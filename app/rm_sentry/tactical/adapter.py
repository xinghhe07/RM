"""环境适配器：把 `BatchedRMUL3v3` 的张量状态接进战术层。

职责边界（**单向依赖，不许反向**）：
    环境张量 --(本文件)--> `TacticalState` -> 战术层决策 -> `环境动作张量`
    战术层**不认识**张量列索引；环境**不认识**战术动作枚举。两端都只认中间契约。

为什么要有这一层（而不是让决策直接读张量）：
  1. **真车复用**：真车上这些字段从 ROS 话题来（定位/裁判系统/自瞄），
     而不是从 `S.X` 这类列索引来。适配器换掉，战术层一行不改。
  2. **可测**：`to_tactical_state()` 是纯函数（输入张量、输出快照），
     可以直接构造对抗性输入做回归（见 `test_tactical_adapter.py`）。
  3. **不把环境的玩具几何带进战术层**：适配器里出现的几何量（center_r、field_w/h）
     全部来自 `cfg`，真图到位后改 `cfg` 即可，战术层无感。

决策频率：环境 dt=0.05s（20 Hz），战术层 `decision_dt_s=0.1`（10 Hz）
  ⇒ 每 `decide_every` 步决策一次，中间步沿用上次的动作（与真车的"策略慢于控制"一致）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rm_sentry.rmul_3v3_env import E, RMUL3v3Config, S
from rm_sentry.tactical._core import (
    EnemyEstimate,
    NavigationStatus,
    PointEstimate,
    PointKind,
    PointSpec,
    RobotStatus,
    TacticalAction,
    TacticalDecision,
    TacticalPoints,
    TacticalState,
    ZoneOwner,
    ZoneSpec,
    ZoneStatus,
)
from rm_sentry.tactical.config import ScoringWeights, TacticalConfig
from rm_sentry.tactical.decision import DecisionState, SafetyMonitor, decide
from rm_sentry.tactical.machine import TaskState, TacticalMachine

# --------------------------------------------------------------------------
# 点图：由**环境配置**构造（2025/2026 单中心区；2027 三控制区矩形）
# --------------------------------------------------------------------------
CONTROL_POINT_ID = "P_CONTROL"   # 索引 0 = 中央控制区（沿用旧 id，观测槽/测试兼容）
WAIT_POINT_ID = "P_WAIT"

# 2027 追加的 id（三控制区 + 己方补给区；坐标来源见 build_points_from_env）
Z_HIGH_R_ID = "Z_HIGH_R"
Z_HIGH_B_ID = "Z_HIGH_B"
Z_SUPPLY_R_ID = "Z_SUPPLY_R"
P_HIGH_R_ID = "P_HIGH_R"
P_HIGH_B_ID = "P_HIGH_B"


def _assert_zones_rect_order(cfg: RMUL3v3Config) -> None:
    """自检 `cfg.zones_rect` 的顺序约定，错了就抛错而不是静默错标区域。

    环境侧注释声明：**索引 0 = 中央控制区**（旧观测槽 `gen_center` 复用）。
    本函数把这个口头契约变成可执行断言 —— 判据用几何（索引 0 必须最靠近场地中心），
    不用硬编码数值，这样将来改坐标也不会误伤。
    """
    rects = cfg.zones_rect
    if len(rects) != 3:
        raise ValueError(
            f"2027 控制区应为 3 处，cfg.zones_rect 有 {len(rects)} 处；"
            f"若规则改了，需同步改本函数与点图（不要静默继续）")
    fx, fy = cfg.field_w / 2.0, cfg.field_h / 2.0
    d = [math.hypot(float(r[1]) - fx, float(r[2]) - fy) for r in rects]
    if d[0] > min(d[1], d[2]):
        raise ValueError(
            f"cfg.zones_rect[0] 必须是最靠近场地中心的中央控制区，"
            f"但实测距离 {d[0]:.2f} m 不是最小 {min(d[1], d[2]):.2f} m；"
            f"顺序约定被破坏，点图会错标区域")


def build_points_from_env(cfg: RMUL3v3Config) -> TacticalPoints:
    """从**环境配置**构造点图（不是从硬编码坐标）。

    **两条分支**（2027 迁移时加的）：

    (a) `cfg.zones_rect` 非空（2027）→ 按官方三控制区 **矩形** 建图。
        坐标全部来自 `cfg.zones_rect` / `cfg.own_supply_rect`，而这两个字段又逐条对应
        `docs/2027场地规格.md`（官方图纸实测）；本函数**不新增任何坐标假设**。
        点位只放官方图纸能支持的：3 个控制区点 + 1 个补给区安全点。
        ⚠️ **故意不加**的点（加了就是编的）：
          - 隧道观察点：隧道**通行净空未确认**（`2027场地规格.md` §3 待核项 1），
            现在放点可能把哨兵指挥进墙里；
          - 巡逻点：要等队伍定打法，且当前环境没有对应机制。
    (b) 否则（2025/2026）→ 保持原路径：一个中心增益点（圆）+ 一个待机角。
        **行为逐位不变**（`test_tactical_adapter.py` 的历史断言依赖它）。
    """
    cx, cy = cfg.field_w / 2.0, cfg.field_h / 2.0

    if not cfg.zones_rect:
        zones = [ZoneSpec(zone_id="Z_CENTER", name="控制区",
                          x_m=cx, y_m=cy, radius_m=cfg.center_r,
                          strategic_value=1.0, control_zone=True)]
        points = [
            PointSpec(point_id=CONTROL_POINT_ID, name="控制区", kind=PointKind.CONTROL,
                      x_m=cx, y_m=cy, zone_id="Z_CENTER", value=1.0),
            # 等待点：取场地一角（离控制区最远、风险最低）。真图到位后换成真实安全点。
            PointSpec(point_id=WAIT_POINT_ID, name="待机角", kind=PointKind.WAIT,
                      x_m=1.0, y_m=1.0, value=0.5),
        ]
        return TacticalPoints(zones, points)

    # ---- (a) 2027：三控制区（矩形） ----
    _assert_zones_rect_order(cfg)
    c_name, c_rx, c_ry, c_hw, c_hh = cfg.zones_rect[0]   # 中央
    r_name, r_rx, r_ry, r_hw, r_hh = cfg.zones_rect[1]   # 红方高地
    b_name, b_rx, b_ry, b_hw, b_hh = cfg.zones_rect[2]   # 蓝方高地

    zones = [
        ZoneSpec(zone_id="Z_CENTER", name=str(c_name),
                 x_m=c_rx, y_m=c_ry, half_w_m=c_hw, half_h_m=c_hh,
                 strategic_value=1.0, control_zone=True),
        ZoneSpec(zone_id=Z_HIGH_R_ID, name=str(r_name),
                 x_m=r_rx, y_m=r_ry, half_w_m=r_hw, half_h_m=r_hh,
                 strategic_value=1.0, control_zone=True),
        ZoneSpec(zone_id=Z_HIGH_B_ID, name=str(b_name),
                 x_m=b_rx, y_m=b_ry, half_w_m=b_hw, half_h_m=b_hh,
                 strategic_value=1.0, control_zone=True),
    ]
    points = [
        # 三个控制区点：战略价值相等 —— 2027 §3.3.4「每控制 1 处，对方 −1/s」，
        # 三处等价（不是 2026 那种"只有中央分"）。
        PointSpec(point_id=CONTROL_POINT_ID, name=str(c_name), kind=PointKind.CONTROL,
                  x_m=c_rx, y_m=c_ry, zone_id="Z_CENTER", value=1.0),
        PointSpec(point_id=P_HIGH_R_ID, name=str(r_name), kind=PointKind.CONTROL,
                  x_m=r_rx, y_m=r_ry, zone_id=Z_HIGH_R_ID, value=1.0),
        PointSpec(point_id=P_HIGH_B_ID, name=str(b_name), kind=PointKind.CONTROL,
                  x_m=b_rx, y_m=b_ry, zone_id=Z_HIGH_B_ID, value=1.0),
    ]

    if cfg.own_supply_rect:
        scx, scy, shw, shh = cfg.own_supply_rect
        zones.append(ZoneSpec(zone_id=Z_SUPPLY_R_ID, name="己方启动区兼补给区",
                              x_m=scx, y_m=scy, half_w_m=shw, half_h_m=shh,
                              strategic_value=0.0, control_zone=False))
        # 安全点：放在己方补给区内。2027 §3.3.2.2 的虚弱要靠"检测到己方补给区
        # 交互卡"解除，所以安全点必须在补给区里（不是随便找个角落）。
        # 坐标 (1.0, 1.0) 沿用旧实现的"待机角"，未新增假设。
        points.append(PointSpec(point_id=WAIT_POINT_ID, name="己方补给区",
                                kind=PointKind.WAIT, x_m=1.0, y_m=1.0,
                                zone_id=Z_SUPPLY_R_ID, value=0.5))
    else:
        points.append(PointSpec(point_id=WAIT_POINT_ID, name="待机角",
                                kind=PointKind.WAIT, x_m=1.0, y_m=1.0, value=0.5))

    return TacticalPoints(zones, points)


def tactical_config_from_env(cfg: RMUL3v3Config) -> TacticalConfig:
    """用环境的真实规则常量构造战术层配置 —— **避免两处漂移**。

    凡是环境已经定义过的量（复活无敌、决策周期……），一律从这里取，
    不在 `TacticalConfig` 里另写一份默认值（写了就一定会漂）。
    """
    return TacticalConfig(
        decision_dt_s=1.0 / cfg.decision_hz,
        respawn_invuln_s=cfg.respawn_invuln,
    )


# --------------------------------------------------------------------------
# 观测 -> 战术状态（纯函数，可单测）
# --------------------------------------------------------------------------


def to_tactical_state(env, points: TacticalPoints, row: int = 0) -> TacticalState:
    """从批环境的第 `row` 行抽取一个 `TacticalState`（单环境决策用）。

    设计：战术层是单环境的（真车就是一台车），批量仿真里逐环境调用。
    n=512 时每步 512 次 Python 构造有开销，但换来的是
    **战术层与批量维度解耦** —— 这正是可迁移性所在（真车没有"批量"维度）。
    """
    c = env.cfg
    i = row
    s = env.sentry[i]
    sx, sy = float(s[S.X]), float(s[S.Y])

    # ---- 自机 ----
    robot = RobotStatus(
        x_m=sx, y_m=sy,
        hp_fraction=float(s[S.HP]) / max(float(c.sentry_hp), 1.0),
        dead_secs=float(s[S.DEAD]),
        heat_fraction=float(max(s[S.HEAT1], s[S.HEAT2])) / max(float(c.heat_limit), 1.0),
        ammo=int(s[S.AMMO]),
        # 2027 环境已建模虚弱（S.VULN，见 rmul_3v3_env.py 列定义）。
        # `vulnerable_known` 必须跟着 `vuln_on_respawn`：旧赛季该列恒 0 是
        # "未建模"，不是"确认不虚弱"—— 战术层据此决定用状态位还是回退计时器。
        vulnerable=float(s[S.VULN]) > 0.5,
        vulnerable_known=bool(c.vuln_on_respawn),
        position_valid=bool(torch.isfinite(s[S.X]) and torch.isfinite(s[S.Y])),
    )

    # ---- 区域归属（从**位置**现算：先到先得 + 双方同在 -> 争夺中）----
    # 不用 env._holder：那是"上一步的副产物"，且它把"双方都在"记成先到者的。
    # 战术层要的是**当下事实**，因此这里现算（现算是纯几何，无副作用）。
    # ⚠️ 必须走 ZoneSpec.contains()：2027 控制区是 3000×3000 / 1200×1200 的**矩形**，
    #    拿内切圆半径判会把站在四个角上的机器人误判成"没进区"。
    alive = float(s[S.DEAD]) <= 0.0
    vulnerable = float(s[S.VULN]) > 0.5
    # 2027 §3.3.2.2：虚弱状态下"无法激活控制区" ⇒ 人在区内也不算有效占领。
    can_hold = alive and not vulnerable

    vis = env._los_blocked()[i]          # [k]，列 1..n_enemies 是真实敌人
    enemies: list[EnemyEstimate] = []
    sensed: list[tuple[float, float]] = []   # 感知范围内的敌人位置（用于区域归属）
    for m in range(1, c.n_enemies + 1):
        ex, ey = float(env.enemies[i, m, E.X]), float(env.enemies[i, m, E.Y])
        e_alive = float(env.enemies[i, m, E.ALIVE]) > 0.5
        d = math.hypot(ex - sx, ey - sy)
        if e_alive and d <= c.sense_range:
            sensed.append((ex, ey))
        # 只有**本帧可见**的敌人才进快照（哨兵禁止机间通信，拿不到真实坐标）
        if bool(vis[m]) and e_alive:
            enemies.append(EnemyEstimate(x_m=ex, y_m=ey, age_s=0.0,
                                         threat=float(max(0.0, 1.0 - d / max(c.sense_range, 1e-6)))))

    zones: dict[str, ZoneStatus] = {}
    for z in points.zones():
        s_in = z.contains(sx, sy)
        e_in = any(z.contains(ex, ey) for ex, ey in sensed)
        if s_in and e_in:
            owner = ZoneOwner.CONTESTED
        elif s_in and can_hold:
            owner = ZoneOwner.FRIENDLY
        elif e_in:
            owner = ZoneOwner.ENEMY
        elif alive:
            owner = ZoneOwner.NOBODY
        else:
            owner = ZoneOwner.UNKNOWN
        zones[z.zone_id] = ZoneStatus(z.zone_id, owner, age_s=0.0)

    # ---- 点位估计 ----
    # path_cost_s：环境里没有规划器，用"到区**外缘**的直线距离 / v_max"作**代理**并标注。
    #   真实可达性/路径代价必须来自 Nav2（真车上由导航层给），这里只是让战术层能跑。
    #   2027 控制区 3×3 m：算到"区域中心"的直线距离会高估约 1.5 m（区内还要横穿），
    #   所以用 distance_outside_m（已在区内 = 0）—— 与"进区即到位"的规则语义一致。
    v_max = max(float(c.v_max), 1e-6)
    pts: dict[str, PointEstimate] = {}
    for p in points.points():
        zone = points.zone(p.zone_id) if p.zone_id is not None else None
        if zone is not None:
            d = zone.distance_outside_m(sx, sy)
            shape: dict[str, float | None] = (
                {"half_w_m": zone.half_w_m, "half_h_m": zone.half_h_m}
                if zone.is_rect else {"radius_m": zone.radius_m})
        else:
            d = math.hypot(p.x_m - sx, p.y_m - sy)
            shape = {}
        # 风险：可见敌人里最近的有多近（越近越危险）
        risk = 0.0
        if enemies:
            nearest = min(math.hypot(e.x_m - sx, e.y_m - sy) for e in enemies)
            risk = float(max(0.0, 1.0 - nearest / max(c.shot_range, 1e-6)))
        pts[p.point_id] = PointEstimate(
            point_id=p.point_id, reachable=True, path_cost_s=d / v_max,
            risk=risk, age_s=0.0, teammate_intent=False, blocked=False, **shape)

    # 剩余时间：旧实现传的是 `cfg.match_time`（常量，永远 300 s）—— 时间是**不走**的。
    # 这里按环境步数现算（真车上由裁判系统给剩余时间）。
    sc = getattr(env, "step_count", None)
    if sc is not None:
        match_left = max(0.0, float(c.match_time) - float(sc[i]) * float(c.dt))
    else:
        match_left = float(c.match_time)

    return TacticalState(
        robot=robot,
        nav=NavigationStatus(planner_ok=True, consecutive_failures=0),
        zones=zones, points=pts, enemies=tuple(enemies), teammates=(),
        match_time_left_s=match_left,
    )


# --------------------------------------------------------------------------
# 战术决策 -> 环境动作
# --------------------------------------------------------------------------


@dataclass
class _ActionOut:
    """一行的环境动作：`[vx, vy, target_idx, fire]`。"""

    move: torch.Tensor        # [2] ∈ [-1,1]
    target: int               # 0 = 不交战；m = 敌人槽 m
    fire: bool


def _dir_to(target_x: float, target_y: float, sx: float, sy: float,
            cfg, deadband_m: float = 0.3) -> torch.Tensor:
    """朝目标点的单位方向（归一化到 v_max 的动作量）。死区内不微调，避免原地抖。"""
    dx, dy = target_x - sx, target_y - sy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= deadband_m:
        return torch.zeros(2)
    return torch.tensor([dx / dist, dy / dist])


def _inside_bound_zone(points: TacticalPoints, spec: PointSpec,
                       sx: float, sy: float) -> bool:
    """点位绑定的区域是否**已包含自机**（未绑区域 -> False）。"""
    if spec.zone_id is None:
        return False
    try:
        return points.zone(spec.zone_id).contains(sx, sy)
    except KeyError:
        return False


def decision_to_env_action(
    decision: TacticalDecision,
    env,
    points: TacticalPoints,
    i: int = 0,
    *,
    heat_stop_ratio: float = 0.875,
) -> _ActionOut:
    """把战术决策翻译成环境动作（移动来自战术层；打谁/开火由交战逻辑决定）。

    开火门控的取值依据（[开源]）：`pb2025_sentry_behavior` 的
    `IsStatusOK heat_max=350` / 上限 400 = **87.5%**，留余量不贴上限
    （见 `rule_baseline.py` 的同一处注释；两处必须一致，改一处要改另一处）。
    """
    c = env.cfg
    s = env.sentry[i]
    sx, sy = float(s[S.X]), float(s[S.Y])
    alive = float(s[S.DEAD]) <= 0.0
    zero = torch.zeros(2)

    # ---- 移动 ----
    spec: PointSpec | None = None
    if decision.point_id is not None:
        try:
            spec = points.point(decision.point_id)
        except KeyError:
            spec = None

    if not alive or decision.action is TacticalAction.SAFE_MODE:
        move = zero
    elif decision.action in (TacticalAction.HOLD_CURRENT_ZONE, TacticalAction.RECOVER):
        move = zero
    elif spec is not None and _inside_bound_zone(points, spec, sx, sy):
        # 已在目标区域内 -> 原地保持，不再朝区域中心挤。
        # 2027 控制区 3000×3000（半宽 1.5 m）：进区后还往中心走，最坏白送约 2.1 m 的
        # 暴露路程；而规则只要求"检测到控制区交互卡"（进区即可）。
        # 旧环境半径 0.8 m 时"区内"≈"中心"，问题不显著 —— 是本次几何迁移才需要这条。
        move = zero
    elif decision.action is TacticalAction.RETREAT:
        if spec is not None:
            move = _dir_to(spec.x_m, spec.y_m, sx, sy, c)
        else:
            # 无可用安全点：沿"远离控制区中心"的单位向量退（保守，不是随机逃）
            cx, cy = c.field_w / 2.0, c.field_h / 2.0
            ax, ay = cx - sx, cy - sy
            n_ = max((ax * ax + ay * ay) ** 0.5, 1e-6)
            move = torch.tensor([-ax / n_, -ay / n_])
    elif spec is not None:
        move = _dir_to(spec.x_m, spec.y_m, sx, sy, c)
    else:
        move = zero
    move = move.clamp(-1.0, 1.0)

    # ---- 交战（打谁 / 开火） ----
    # 与 `rule_baseline.py` 同口径：现算可见性（不用 env._vis —— 见 AGENTS.md R2c）
    vis = env._los_blocked()[i]
    ex = env.enemies[i, :, E.X]
    ey = env.enemies[i, :, E.Y]
    dist_e = torch.hypot(ex - sx, ey - sy)
    valid = (vis > 0) & (env.enemies[i, :, E.ALIVE] > 0.5)
    valid[0] = False                                   # 槽 0 是自机占位
    masked = torch.where(valid, dist_e, torch.full_like(dist_e, 1e6))
    nearest = int(masked.argmin())
    any_visible = bool(valid.any())
    nearest_dist = float(masked[nearest]) if any_visible else 1e6

    heat = float(max(s[S.HEAT1], s[S.HEAT2]))
    vulnerable = float(s[S.VULN]) > 0.5      # 2027 虚弱锁发射；旧赛季恒 0
    # ⚠️ 所有开火条件必须在**同一个**布尔量里判定：目标选择与开火用两套条件时，
    #    会出现"选了目标但不开火"的中间态（实现时踩过一次，被 T4 断言抓住）。
    #    条件与环境 `action_mask()` 的口径逐条对齐
    #    （弹药 > 0 / 非复活中 / 非虚弱 / 可见）。
    can_fire = (any_visible and alive
                and float(s[S.AMMO]) > 0 and float(s[S.DEAD]) <= 0
                and not vulnerable
                and decision.action not in (TacticalAction.SAFE_MODE, TacticalAction.RECOVER,
                                            TacticalAction.RETREAT)
                and heat < float(c.heat_limit) * heat_stop_ratio
                and nearest_dist < float(c.shot_range)
                and bool(vis[nearest]))
    # 交战中也要尊重掩码：不可见的目标槽不能选
    # ⚠️ 不再调 `env.action_mask()` —— 它读 `env._vis`（`_build_obs()` 的副产物，
    #    可能是上一帧的陈旧值，且新环境上可能尚未初始化；见 AGENTS.md R2c）。
    #    这里用上面**现算**的 `vis`，口径与环境 step() 内部一致。
    target = nearest if can_fire else 0
    fire = bool(can_fire)

    return _ActionOut(move=move, target=target, fire=fire)


# --------------------------------------------------------------------------
# 驱动器：批量环境下跑战术层（每 env 一份决策状态）
# --------------------------------------------------------------------------


@dataclass
class TacticalDriver:
    """在 `BatchedRMUL3v3` 上驱动战术层，输出环境动作张量。

    ⚠️ 跨步状态（`DecisionState` / `SafetyMonitor` / `TacticalMachine`）**每个环境一份**，
       且必须在 `reset()` 里一起重置 —— 这是 `AGENTS.md` R2 的直接落地：
       任何跨 step 状态都要回答"新一局开始时它还合法吗"。
    """

    env: object
    points: TacticalPoints = None            # type: ignore[assignment]
    cfg: TacticalConfig = None               # type: ignore[assignment]
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    # [默认] 每几步决策一次：环境 20 Hz / 战术 10 Hz ⇒ 2
    decide_every: int = 2
    # 决策日志（`tactical/logger.py` 的 DecisionLogger 或 None）。
    # duck typing：为 None 时本层零开销；开启时每决策周期落一条 JSONL（见 logger.py）。
    logger: object = None

    def __post_init__(self) -> None:
        if self.points is None:
            self.points = build_points_from_env(self.env.cfg)
        if self.cfg is None:
            self.cfg = tactical_config_from_env(self.env.cfg)
        self._tick = 0
        self._reset_state(self.env.n)

    def _reset_state(self, n: int) -> None:
        self.decisions: list[TacticalDecision] = [
            TacticalDecision(TacticalAction.HOLD_CURRENT_ZONE, None, "初始") for _ in range(n)]
        self.states = [DecisionState() for _ in range(n)]
        self.safeties = [SafetyMonitor(cfg=self.cfg) for _ in range(n)]
        self.machines = [TacticalMachine(cfg=self.cfg, points=self.points) for _ in range(n)]
        self._dec_count = [0] * n   # 每环境决策序号（日志用；新一局归零，R2）

    def _reset_env(self, i: int) -> None:
        """只重置第 i 个环境的跨步状态（部分 `auto_reset` 时必须用这个）。"""
        self.decisions[i] = TacticalDecision(
            TacticalAction.HOLD_CURRENT_ZONE, None, "初始")
        self.states[i] = DecisionState()
        self.safeties[i] = SafetyMonitor(cfg=self.cfg)
        self.machines[i] = TacticalMachine(cfg=self.cfg, points=self.points)
        self._dec_count[i] = 0

    def reset(self, env_indices: list[int] | None = None) -> None:
        """新一局：重置跨步状态（R2）。

        `env_indices=None` 重置全部（显式整局重置）；
        给定时**只重置这些环境** —— 批环境里只有部分 `done` 时必须走这条路，
        否则仍在进行的环境会丢掉 dwell/age/复活/安全状态机（A2.2）。
        """
        if env_indices is None:
            self._tick = 0
            self._reset_state(self.env.n)
            return
        for i in env_indices:
            self._reset_env(int(i))

    # ---- 主接口 ----
    def act(self) -> torch.Tensor:
        """返回 [n, n_actions] 的环境动作张量。"""
        n = self.env.n
        do_decide = (self._tick % self.decide_every == 0)
        out = torch.zeros(n, self.env.n_actions, device=self.env.device)

        # 逐环境构造快照是 O(n) 次张量切片。n=512 时开销可观但可接受；
        # 需要提速时把 `to_tactical_state` 向量化 —— 但**契约为先**，
        # 战术层不知道"批量"这个概念，提速不能以破坏这层隔离为代价。
        log = self.logger
        for i in range(n):
            snapshot = to_tactical_state(self.env, self.points, row=i)
            tr = None
            if do_decide:
                dec, st = decide(snapshot, self.points, self.cfg, self.weights,
                                 self.safeties[i], self.states[i])
                self.states[i] = st
                self.decisions[i] = dec
                tr = self.machines[i].update(snapshot, dec)
            o = decision_to_env_action(self.decisions[i], self.env, self.points, i)
            out[i, 0:2] = o.move
            out[i, 2] = float(o.target)
            out[i, 3] = 1.0 if o.fire else 0.0
            # 决策日志（可选）。⚠️ step_count.item() 会同步设备，故只在小批量复盘时开启。
            if do_decide and log is not None and log.wants(i):
                frame = int(self.env.step_count[i].item())
                t = frame * float(self.env.cfg.dt)
                self._dec_count[i] += 1
                log.log_decision(i, snapshot, self.decisions[i], self.states[i], o,
                                 t=t, frame=frame, decision_idx=self._dec_count[i])
                if tr is not None:
                    log.log_transition(i, tr, t=t, frame=frame)
        self._tick += 1
        return out

    # ---- 观测（给评估/日志用）----
    def machine_states(self) -> list[TaskState]:
        return [m.state for m in self.machines]
