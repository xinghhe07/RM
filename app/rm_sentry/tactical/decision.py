"""战术决策层：安全监督 + 复活阶段 + 优先级栈 + 带迟滞的选点。

优先级栈（`哨兵方案设计.md` §3.1 / 团队 2026-10-13 讨论确认）：

  1. 安全监督（最高，战术层不可绕过）
  2. 复活 / 保护 / 虚弱（阶段判定）
  3. 导航有效性（定位丢了就别再规划新目标）
  4. 当前任务保持（迟滞 + 最短驻留，防抖点）
  5. 重新选点（评分）

设计约束（都是被踩过的坑逼出来的）：
  - **不反推状态**（AGENTS.md R2b）：复活/保护/虚弱的跨越全部用显式标志记录，
    绝不用"dead_secs > 0"之外的推断。
  - **不用过期信息当安全**：区域信息超过 `zone_stale_after_s` 一律按 UNKNOWN 处理。
  - **纯函数式**：`decide()` 只读输入、返回结论，跨步状态全部放在
    `DecisionState` 里由调用方持有 —— 便于单测与复盘（日志里能看见"为什么"）。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from rm_sentry.tactical._core import (
    PointEstimate,
    PointKind,
    PointSpec,
    TacticalAction,
    TacticalDecision,
    TacticalPoints,
    TacticalState,
    ZoneOwner,
    collect_nonfinite,
)
from rm_sentry.tactical.config import (
    DecisionPriorities,
    ScoringWeights,
    TacticalConfig,
)

_EPS = 1e-9

#: 非控制类点位的默认到达容差（米）。控制类点位用绑定区域的半径。
# [默认] 待按实车最小转弯半径与定位精度标定。
ARRIVAL_TOL_M = 0.3


# --------------------------------------------------------------------------
# 安全监督层（独立状态机，见 `工程红线.md` §1）
# --------------------------------------------------------------------------


class SafetyState(enum.Enum):
    """安全状态。与战术任务状态机**独立** —— 战术层无法把它降级回去。"""

    NORMAL = enum.auto()    # 一切正常
    DEGRADED = enum.auto()  # 观测失效 / 导航连续失败：保守停滞，待恢复
    RECOVERY = enum.auto()  # 尝试恢复中（重新初始化定位 / 取消过期目标）
    SAFE_STOP = enum.auto()  # 无法保证安全运动：停


@dataclass
class SafetyMonitor:
    """安全监督。只做一件事：**基于新鲜度与数值有效性**决定是否允许战术行动。

    设计取舍（`工程目标.md` §4 冲突仲裁表）："不确定该不该降级 -> 降"。
    因此这里全部是**宁保守勿激进**的判定。

    ⚠️ 升级方向单向：一旦 SAFE_STOP，只有数据持续有效才能逐级回来，
       NaN 出现过则**永久 SAFE_STOP**（数值异常说明链路有硬问题，比赛中
       没有人工确认的机会，停在原地是唯一安全选择）。
    """

    cfg: TacticalConfig
    state: SafetyState = SafetyState.NORMAL
    reason: str = ""
    healthy_s: float = 0.0       # 连续健康时长
    invalid_s: float = 0.0       # 连续异常时长
    loc_lost_s: float = 0.0      # 连续定位丢失时长（单独的升级依据）
    ever_nan: bool = False       # 出现过 NaN -> 永久停
    invalid_for: float = 0.0     # SAFE_STOP 之后仍需持续有效的时长

    def degraded(self) -> bool:
        return self.state is not SafetyState.NORMAL

    def update(self, snapshot: TacticalState) -> SafetyState:
        c = self.cfg
        dt = c.decision_dt_s

        # (a) 数值异常：立即永久 SAFE_STOP（链路有硬问题，比赛中无法人工确认）
        if collect_nonfinite(snapshot):
            self.ever_nan = True
            self._keep(SafetyState.SAFE_STOP, "数值异常（NaN/Inf），永久停机")
            return self.state

        # (b) 更新健康/异常计时
        why = ""
        if not snapshot.robot.position_valid:
            why = "定位失效"
            self.loc_lost_s += dt
        if not snapshot.nav.planner_ok:
            why = why or "规划器不健康"
        elif snapshot.nav.consecutive_failures >= c.nav_fail_limit:
            why = why or f"连续规划失败 {snapshot.nav.consecutive_failures} 次"

        healthy = not why
        if healthy:
            self.healthy_s += dt
            self.invalid_s = 0.0
            self.loc_lost_s = 0.0
        else:
            self.invalid_s += dt
            self.healthy_s = 0.0

        # (c) 状态转移。升级（变保守）总是立即；降级（恢复）必须有持续有效证据。
        if self.ever_nan:
            self._keep(SafetyState.SAFE_STOP, "数值异常过，永久停机")
        elif self.state is SafetyState.SAFE_STOP:
            self.invalid_for += dt if healthy else 0.0
            if not healthy:
                self.invalid_for = 0.0
            if healthy and self.invalid_for >= c.recovery_confirm_s:
                self._set(SafetyState.NORMAL, "持续有效，恢复正常")
            else:
                self._keep(SafetyState.SAFE_STOP, self.reason or "保持停机")
        elif self.loc_lost_s >= c.localization_loss_max_s:
            self._set(SafetyState.SAFE_STOP, "定位丢失超时 -> 停机")
        elif self.state is SafetyState.RECOVERY:
            if healthy and self.healthy_s >= c.recovery_confirm_s:
                self._set(SafetyState.NORMAL, "恢复成功")
            elif not healthy and self.invalid_s >= c.nav_recover_max_s:
                self._set(SafetyState.SAFE_STOP, f"{why} 超时未恢复")
            else:
                self._keep(SafetyState.RECOVERY, why or self.reason)
        elif self.state is SafetyState.DEGRADED:
            if healthy:
                self._set(SafetyState.RECOVERY, "观测恢复，确认中")
            else:
                self._keep(SafetyState.DEGRADED, why)
        else:  # NORMAL
            if not healthy:
                self._set(SafetyState.DEGRADED, why)
        return self.state

    def _set(self, s: SafetyState, reason: str) -> None:
        self.state = s
        self.reason = reason

    def _keep(self, s: SafetyState, reason: str) -> None:
        self.state = s
        if reason:
            self.reason = reason


# --------------------------------------------------------------------------
# 复活阶段（显式状态，不反推 —— AGENTS.md R2b）
# --------------------------------------------------------------------------


class RespawnPhase(enum.Enum):
    ALIVE = enum.auto()
    DEAD = enum.auto()                   # 读条中
    RESPAWNED_PROTECTED = enum.auto()    # 复活保护期（无敌，通常禁止开火）
    RESPAWNED_VULNERABLE = enum.auto()   # 保护结束（据实标记虚弱）


@dataclass
class RespawnTracker:
    """复活阶段的显式跨步记录。

    注意：这里**不**从 `dead_secs` 反推任何东西 —— 只记录"上一步死没死"这一个
    显式标志，然后按边沿触发（`just_respawned`）。这是 R2b 的直接落地。

    虚弱的两种口径（2026-10 发现并修，A1.6）：
      - `vulnerable_known=True`（2027）：**跟随状态位**。规则虚弱持续到接触
        己方补给区，不能用自编秒数倒数 —— 否则战术层会在环境仍禁止占区/开火时
        提前恢复行为，两处口径分叉。此时也不进"保护期原地不动"：复活后应当
        立即回补给区解虚弱（安全点就在补给区内），30 s 无敌正好覆盖这段路程。
      - `vulnerable_known=False`（旧赛季环境未建模）：回退到旧计时器序列
        DEAD -> PROTECTED -> VULNERABLE -> ALIVE，行为逐位不变。
    """

    phase: RespawnPhase = RespawnPhase.ALIVE
    was_dead: bool = False
    protected_s: float = 0.0
    vulnerable_s: float = 0.0

    def update(self, snapshot: TacticalState, cfg: TacticalConfig):
        """返回 (phase, reason, just_respawned)。就地推进内部计数器。"""
        dt = cfg.decision_dt_s
        dead_now = snapshot.robot.dead_secs > _EPS
        just_respawned = self.was_dead and not dead_now

        if snapshot.robot.vulnerable_known:
            # ---- 规则状态位优先（2027）：虚弱不倒数，由现场解除 ----
            if dead_now:
                self.phase = RespawnPhase.DEAD
                reason = f"读条剩余 {snapshot.robot.dead_secs:.1f}s"
            elif snapshot.robot.vulnerable:
                if self.phase is not RespawnPhase.RESPAWNED_VULNERABLE:
                    why = "刚复活且虚弱" if just_respawned else "虚弱"
                    reason = f"{why}：回己方补给区解除"
                else:
                    reason = "虚弱未解除（等待接触己方补给区）"
                self.phase = RespawnPhase.RESPAWNED_VULNERABLE
                self.protected_s = 0.0
                self.vulnerable_s = 0.0
            else:
                reason = ("虚弱已解除，恢复战术行为"
                          if self.phase is RespawnPhase.RESPAWNED_VULNERABLE else "存活")
                self.phase = RespawnPhase.ALIVE
                self.protected_s = 0.0
                self.vulnerable_s = 0.0
            self.was_dead = dead_now
            return self.phase, reason, just_respawned

        # ---- 回退：数据源不提供虚弱状态（2025/2026 环境，旧行为） ----
        if dead_now:
            self.phase = RespawnPhase.DEAD
            reason = f"读条/无敌剩余 {snapshot.robot.dead_secs:.1f}s"
        elif just_respawned:
            # 复活判定成功的那一步：进入保护期（用规则的无敌时长，不是估计值）
            self.protected_s = cfg.respawn_invuln_s
            self.vulnerable_s = 0.0
            self.phase = RespawnPhase.RESPAWNED_PROTECTED
            reason = "刚复活：进入保护期"
        elif self.phase is RespawnPhase.RESPAWNED_PROTECTED:
            self.protected_s = max(0.0, self.protected_s - dt)
            if self.protected_s <= 0.0:
                self.phase = RespawnPhase.RESPAWNED_VULNERABLE
                self.vulnerable_s = cfg.vulnerable_secs_assumed
                reason = "保护期结束"
            else:
                reason = f"保护期剩余 {self.protected_s:.1f}s"
        elif self.phase is RespawnPhase.RESPAWNED_VULNERABLE:
            self.vulnerable_s = max(0.0, self.vulnerable_s - dt)
            if self.vulnerable_s <= 0.0:
                self.phase = RespawnPhase.ALIVE
                reason = "虚弱结束，恢复战术行为"
            else:
                reason = f"虚弱剩余 {self.vulnerable_s:.1f}s"
        else:
            self.phase = RespawnPhase.ALIVE
            reason = "存活"

        self.was_dead = dead_now
        return self.phase, reason, just_respawned


# --------------------------------------------------------------------------
# 决策状态（跨步，由调用方持有；任何时候都能打印出来复盘）
# --------------------------------------------------------------------------


@dataclass
class DecisionState:
    """跨步决策状态。对应 `工作台.md` 的铁律 R2：所有跨步量都在这里显式管理。"""

    respawn: RespawnTracker = None  # type: ignore[assignment]
    prev: TacticalDecision | None = None
    prev_point_id: str | None = None
    dwell_s: float = 0.0     # 当前目标已保持多久（用于最短驻留）
    age_s: float = 0.0       # 距上次**换点**过了多久（用于迟滞）
    mode: str = ""
    vulnerable_seen: bool = False

    def __post_init__(self) -> None:
        if self.respawn is None:
            self.respawn = RespawnTracker()


# --------------------------------------------------------------------------
# 选点评分（可解释：每个分量都要能单独打印）
# --------------------------------------------------------------------------


def control_factor(snapshot: TacticalState, point: PointSpec, cfg: TacticalConfig) -> float:
    """控场收益分量 C ∈ [0,1]：这个点现在值不值得去。

    规则依据（2026 §3.3.4）：控制区同一时间只能被一方占领；双方同在 -> 先到先得。
    """
    owner = snapshot.zone_owner(point.zone_id) if point.zone_id else ZoneOwner.UNKNOWN
    if point.kind is PointKind.CONTROL:
        if owner is ZoneOwner.NOBODY:
            return 1.0        # 空区：最高收益（直接开始扣对方分）
        if owner is ZoneOwner.ENEMY:
            return 0.85       # 敌占：争夺（有风险，见 R 分量）
        if owner is ZoneOwner.CONTESTED:
            return 0.5        # 争夺中：先到先得未定
        if owner is ZoneOwner.FRIENDLY:
            return 0.2        # 已是我们的：只需维持（守着）
        return 0.35           # UNKNOWN：信息过期，不能当"安全"也不能当"能占"
    # 非控制类点位：控场收益按"是否需要"给固定值
    if point.kind is PointKind.PATROL:
        return 0.3
    if point.kind is PointKind.OBSERVE:
        return 0.25
    if point.kind in (PointKind.WAIT, PointKind.RECOVERY):
        return 0.1
    return 0.2


def point_score(
    snapshot: TacticalState,
    point: PointSpec,
    est: PointEstimate,
    cfg: TacticalConfig,
    weights: ScoringWeights,
) -> tuple[float, dict[str, float]]:
    """S = wv*V + wc*C - wd*D - wr*R - ws*S - wt*T，返回 (分数, 各分量)。

    分量全部归一化到 [0,1]（权重之和 = 1.0，见 `ScoringWeights`）。
    返回分量字典是为了**可解释性**：日志里能看见这次选点的完整理由。
    """
    value = min(max(point.value, 0.0), 1.0)

    control = control_factor(snapshot, point, cfg)
    # 陈旧信息惩罚：过期的"空区"可能其实有人
    stale = min(est.age_s / max(cfg.point_stale_after_s, _EPS), 1.0)
    cost = min(est.path_cost_s / max(weights.path_scale_s, _EPS), 1.0)
    risk = min(max(est.risk, 0.0), 1.0)
    conflict = 1.0 if est.teammate_intent else 0.0

    total = (weights.w_value * value
             + weights.w_control * control
             - weights.w_cost * cost
             - weights.w_risk * risk
             - weights.w_stale * stale
             - weights.w_conflict * conflict)
    parts = {"V": value, "C": control, "D": cost, "R": risk, "S": stale, "T": conflict}
    return total, parts


def viable_estimates(snapshot: TacticalState, points: TacticalPoints, cfg: TacticalConfig):
    """过滤出可信、可达、非队友在执行的点位估计。"""
    out: dict[str, PointEstimate] = {}
    for p in points.points():
        est = snapshot.points.get(p.point_id)
        if est is None:
            est = PointEstimate(point_id=p.point_id)  # 缺省视为新鲜可达（保守由调用方覆盖）
        if est.blocked or not est.reachable:
            continue
        if est.age_s > cfg.point_stale_after_s:
            continue
        out[p.point_id] = est
    return out


def _pick_safe_point(snapshot: TacticalState, points: TacticalPoints,
                     cfg: TacticalConfig) -> str | None:
    """选一个**可达**的安全点（等待点/恢复点）。全部不可达 -> None（原地保守）。"""
    best: tuple[float, str] | None = None
    for p in points.safe_points():
        est = snapshot.points.get(p.point_id)
        if est is None or est.blocked or not est.reachable:
            continue
        if est.age_s > cfg.point_stale_after_s:
            continue
        score = -est.path_cost_s - 5.0 * est.risk
        if best is None or score > best[0]:
            best = (score, p.point_id)
    return best[1] if best else None


# --------------------------------------------------------------------------
# 主决策入口
# --------------------------------------------------------------------------


def decide(
    snapshot: TacticalState,
    points: TacticalPoints,
    cfg: TacticalConfig,
    weights: ScoringWeights,
    safety: SafetyMonitor,
    prev_state: DecisionState,
    priorities: DecisionPriorities | None = None,
) -> tuple[TacticalDecision, DecisionState]:
    """一次战术决策。**纯函数 + 显式跨步状态**，返回 (决策, 新状态)。

    调用顺序即优先级栈：安全 -> 复活 -> 导航 -> 当前任务 -> 选点。
    """
    priorities = priorities or DecisionPriorities()
    st = _advance(prev_state, cfg)   # 复制并推进计时器

    # ---- 1) 安全监督（最高优先级，不可绕过） ----
    # 归一化：任何非 NORMAL 的安全状态，战术层一律 SAFE_MODE（停）。
    # 理由：RECOVERY 的定义就是"定位/导航不可信"，此时给移动指令是不负责任的；
    #      DEGRADED 也不该继续做战术机动。稳妥优先（工程目标.md 冲突仲裁表）。
    safety_state = safety.update(snapshot)
    if safety_state is not SafetyState.NORMAL:
        st.prev_point_id = None
        st.prev = TacticalDecision(TacticalAction.SAFE_MODE, None,
                                   f"安全层[{safety_state.name}]：{safety.reason}")
        st.mode = safety_state.name
        return st.prev, st

    # ---- 2) 复活 / 保护 / 虚弱（显式阶段判定，不反推） ----
    phase, why, _just = st.respawn.update(snapshot, cfg)
    st.mode = phase.name
    if phase is RespawnPhase.DEAD:
        st.prev_point_id = None
        st.prev = TacticalDecision(TacticalAction.HOLD_CURRENT_ZONE, None, f"战亡：{why}")
        return st.prev, st
    if phase is RespawnPhase.RESPAWNED_PROTECTED:
        st.prev_point_id = None
        st.prev = TacticalDecision(TacticalAction.HOLD_CURRENT_ZONE, None,
                                   f"复活保护期：{why}（不浪费无敌，也不盲目出击）")
        return st.prev, st

    # 虚弱 / 低血：撤离到安全点。
    # 一旦进入撤离（虚弱或血量进入下限），必须等血量回到 resume 阈值以上才恢复 ——
    # 这是"迟滞带"（retreat < resume），避免在阈值附近反复横跳。
    hp = snapshot.robot.hp_fraction
    if phase is RespawnPhase.RESPAWNED_VULNERABLE:
        st.vulnerable_seen = True
    if hp >= cfg.resume_hp_fraction:
        st.vulnerable_seen = False
    want_retreat = (phase is RespawnPhase.RESPAWNED_VULNERABLE
                    or hp < cfg.retreat_hp_fraction
                    or (st.vulnerable_seen and hp < cfg.resume_hp_fraction))
    if want_retreat:
        safe = _pick_safe_point(snapshot, points, cfg)
        st.prev_point_id = None
        st.prev = TacticalDecision(
            TacticalAction.RETREAT if safe else TacticalAction.HOLD_CURRENT_ZONE,
            safe,
            f"低血量/虚弱（{phase.name}, HP={hp:.0%}, {why}）-> 撤离"
            + ("" if safe else "（无可用安全点，原地保守）"))
        return st.prev, st

    # ---- 3) 导航有效性 ----
    if not snapshot.nav.planner_ok or \
            snapshot.nav.consecutive_failures >= cfg.nav_fail_limit:
        st.prev_point_id = None
        st.prev = TacticalDecision(
            TacticalAction.RECOVER, None,
            f"导航异常（failures={snapshot.nav.consecutive_failures}）-> 取消过期目标")
        return st.prev, st

    # ---- 4) + 5) 当前任务保持 / 重新选点（带迟滞） ----
    return _select(snapshot, points, cfg, weights, st)


def _advance(prev_state: DecisionState, cfg: TacticalConfig) -> DecisionState:
    """推进跨步计时器（dwell/age）。复制一份，保持 decide() 对调用方无副作用。"""
    st = DecisionState(
        respawn=prev_state.respawn,
        prev=prev_state.prev,
        prev_point_id=prev_state.prev_point_id,
        dwell_s=prev_state.dwell_s,
        age_s=prev_state.age_s,
        mode=prev_state.mode,
        vulnerable_seen=prev_state.vulnerable_seen,
    )
    st.dwell_s += cfg.decision_dt_s
    st.age_s += cfg.decision_dt_s
    return st


def _select(snapshot: TacticalState, points: TacticalPoints, cfg: TacticalConfig,
            weights: ScoringWeights, st: DecisionState) -> tuple[TacticalDecision, DecisionState]:
    ests = viable_estimates(snapshot, points, cfg)
    scored: list[tuple[float, str, dict[str, float]]] = []
    for p in points.points():
        if p.point_id not in ests:
            continue
        s, parts = point_score(snapshot, p, ests[p.point_id], cfg, weights)
        scored.append((s, p.point_id, parts))
    scored.sort(key=lambda t: t[0], reverse=True)

    if not scored:
        st.prev_point_id = None
        st.prev = TacticalDecision(
            TacticalAction.HOLD_CURRENT_ZONE, None,
            "无可达点位（全部不可达/过期）-> 原地保持，等待可达性恢复")
        return st.prev, st

    best_s, best_id, best_parts = scored[0]
    cur_id = st.prev_point_id
    cur = next((t for t in scored if t[1] == cur_id), None) if cur_id else None

    # 防抖规则（唯一一条）：换点必须同时满足
    #   (a) 已过最短驻留；(b) 新点优势 >= switch_margin
    # 只要有一个不满足就维持当前点 —— 不叠加"当前点加成"，避免双阈值互相干扰。
    keep_current = False
    note = ""
    if cur is not None:
        if st.age_s < cfg.min_dwell_s:
            keep_current, note = True, f"最短驻留未满（{st.age_s:.1f}/{cfg.min_dwell_s:.1f}s）"
        elif best_s - cur[0] < cfg.switch_margin:
            keep_current, note = True, (f"优势不足（{best_s - cur[0]:.2f} < "
                                        f"{cfg.switch_margin:.2f}），维持不过早换点")
    chosen = cur if keep_current else (best_s, best_id, best_parts)

    if not keep_current and cur_id != chosen[1]:
        st.age_s = 0.0     # 正式换点，重置迟滞计时

    st.prev_point_id = chosen[1]
    st.prev = _to_action(snapshot, points.point(chosen[1]), chosen[2], note,
                         score=chosen[0], is_switch=not keep_current)
    return st.prev, st


def _to_action(snapshot: TacticalState, spec: PointSpec, parts: dict[str, float],
               note: str, *, score: float, is_switch: bool) -> TacticalDecision:
    """把"选中的点"翻译成战术动作（可解释：reason 带完整评分分量）。"""
    owner = snapshot.zone_owner(spec.zone_id) if spec.zone_id else ZoneOwner.UNKNOWN
    if spec.kind is PointKind.CONTROL:
        if owner is ZoneOwner.FRIENDLY:
            action = TacticalAction.DEFEND_POINT
            head = f"防守 {spec.name}（已是我方）"
        else:
            action = TacticalAction.CAPTURE_POINT
            head = f"争夺 {spec.name}（{owner.name}）"
    elif spec.kind is PointKind.PATROL:
        action, head = TacticalAction.PATROL_POINT, f"巡逻 {spec.name}"
    elif spec.kind in (PointKind.WAIT, PointKind.RECOVERY):
        action, head = TacticalAction.RETREAT, f"撤回安全点 {spec.name}"
    else:
        action, head = TacticalAction.CAPTURE_POINT, f"前往 {spec.name}"

    tag = "换点" if is_switch else "保持"
    reason = (f"[{tag}] {head}；score={score:.3f} "
              f"(V={parts['V']:.2f} C={parts['C']:.2f} D={parts['D']:.2f} "
              f"R={parts['R']:.2f} S={parts['S']:.2f} T={parts['T']:.2f})"
              + (f"；{note}" if note else ""))
    return TacticalDecision(action, spec.point_id, reason)


def arrive_at(snapshot: TacticalState, spec: PointSpec) -> bool:
    """到点判定（公开：状态机与决策层共用同一套语义，避免两处口径漂移）。

    控制类点位按**绑定区域的形状**判定 —— 这是"控制区是区域不是点"的落地：
    人在区域里就算到位，不需要追一个精确坐标。
      - 矩形区（2027 三控制区）：`|Δx| <= half_w` 且 `|Δy| <= half_h`，**边界含入**
      - 圆区（2025/2026 中心增益点）：`dist <= radius_m`
    ⚠️ 矩形**不能**退化成内切圆：3000×3000 的方区，四角距中心 1.5·√2 ≈ 2.12 m，
       大于内切半径 1.5 m —— 用圆判会把**站在角落里的机器人**判成"没到位"。
    其余点位按 `ARRIVAL_TOL_M` 容差。
    """
    r = snapshot.robot
    est = snapshot.point_est(spec.point_id)
    dist = math.hypot(r.x_m - spec.x_m, r.y_m - spec.y_m)
    if spec.kind is PointKind.CONTROL:
        if est.half_w_m is not None and est.half_h_m is not None:
            return (abs(r.x_m - spec.x_m) <= est.half_w_m + _EPS
                    and abs(r.y_m - spec.y_m) <= est.half_h_m + _EPS)
        if est.radius_m is not None:
            return dist <= est.radius_m + _EPS
    return dist <= ARRIVAL_TOL_M
