"""战术层回归测试（几何无关）。

这六类场景是团队 2026-10-13 确认的**必测清单**（对应"回归测试"契约）：
    TC1 定位丢失          TC4 复活（读条/保护/虚弱）
    TC2 点位不可达        TC5 信息过期 / 通信超时
    TC3 控制区易手        TC6 目标反复切换（防抖）
外加：点到区域绑定约束、RL 动作接口稳定性、配置权重自检。

⚠️ 本测试**不依赖任何真实场地坐标**（红线见交接文档 §9）：下面点图里的
   坐标全部是**测试构造值**，与真实赛场无关，禁止复制进任何配置。

运行：  python test_tactical.py
"""

from __future__ import annotations

import sys
import traceback

from rm_sentry.tactical import (
    DecisionState,
    PointEstimate,
    PointKind,
    PointSpec,
    RobotStatus,
    SafetyMonitor,
    SafetyState,
    ScoringWeights,
    TacticalAction,
    TacticalConfig,
    TacticalPoints,
    TacticalState,
    TaskState,
    TacticalMachine,
    ZoneOwner,
    ZoneSpec,
    ZoneStatus,
    decide,
)

# --------------------------------------------------------------------------
# 测试脚手架
# --------------------------------------------------------------------------

OK: list[str] = []
BAD: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else BAD).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def _make_points(all_unreachable: bool = False) -> TacticalPoints:
    """构建**测试用**点图（坐标是构造值，不是真实场地参数）。"""
    zones = [
        ZoneSpec("Z_L", "左测试区", 3.0, 4.0, 1.0, strategic_value=1.0, control_zone=True),
        ZoneSpec("Z_R", "右测试区", 9.0, 4.0, 1.0, strategic_value=1.0, control_zone=True),
    ]
    points = [
        PointSpec("P_L", "左控制点", PointKind.CONTROL, 3.0, 4.0, "Z_L", value=1.0),
        PointSpec("P_R", "右控制点", PointKind.CONTROL, 9.0, 4.0, "Z_R", value=1.0),
        PointSpec("P_SAFE", "测试安全点", PointKind.WAIT, 6.0, 7.0, value=0.5),
    ]
    return TacticalPoints(zones, points)


def _snapshot(
    *,
    x: float = 6.0,
    y: float = 4.0,
    hp: float = 1.0,
    pos_valid: bool = True,
    planner_ok: bool = True,
    dead_secs: float = 0.0,
    vulnerable: bool = False,
    vulnerable_known: bool = False,
    owner_l: ZoneOwner = ZoneOwner.NOBODY,
    owner_r: ZoneOwner = ZoneOwner.NOBODY,
    reach_l: bool = True,
    reach_r: bool = True,
    reach_safe: bool = True,
    age_l: float = 0.0,
    age_r: float = 0.0,
    teammate_l: bool = False,
    teammate_r: bool = False,
    cost_l: float = 4.0,
    cost_r: float = 4.0,
    heat: float = 0.0,
    ammo: int = 750,
) -> TacticalState:
    pts = {
        "P_L": PointEstimate("P_L", reachable=reach_l, path_cost_s=cost_l,
                             age_s=age_l, teammate_intent=teammate_l, radius_m=1.0),
        "P_R": PointEstimate("P_R", reachable=reach_r, path_cost_s=cost_r,
                             age_s=age_r, teammate_intent=teammate_r, radius_m=1.0),
        "P_SAFE": PointEstimate("P_SAFE", reachable=reach_safe, path_cost_s=2.0,
                                radius_m=None),
    }
    return TacticalState(
        robot=RobotStatus(x_m=x, y_m=y, hp_fraction=hp, dead_secs=dead_secs,
                          heat_fraction=heat, ammo=ammo, vulnerable=vulnerable,
                          vulnerable_known=vulnerable_known,
                          position_valid=pos_valid),
        zones={
            "Z_L": ZoneStatus("Z_L", owner_l, age_l),
            "Z_R": ZoneStatus("Z_R", owner_r, age_r),
        },
        points=pts,
    )


class _Harness:
    """把 (快照 -> 决策) 的固定样板收在一处，避免每个用例复制粘贴。"""

    def __init__(self, points: TacticalPoints, cfg: TacticalConfig | None = None):
        self.points = points
        self.cfg = cfg or TacticalConfig()
        self.weights = ScoringWeights()
        self.safety = SafetyMonitor(cfg=self.cfg)
        self.dstate = DecisionState()
        self.machine = TacticalMachine(cfg=self.cfg, points=points)

    def step(self, snapshot: TacticalState):
        dec, self.dstate = decide(snapshot, self.points, self.cfg,
                                  self.weights, self.safety, self.dstate)
        tr = self.machine.update(snapshot, dec)
        return dec, tr

    def run(self, snapshot: TacticalState, n: int):
        last = None
        for _ in range(n):
            last = self.step(snapshot)
        return last


# --------------------------------------------------------------------------
# TC1 定位丢失
# --------------------------------------------------------------------------


def test_localization_loss() -> None:
    print("\n[TC1] 定位丢失：安全层必须接管，且不能自动降级回正常")
    h = _Harness(_make_points())
    dec, _ = h.step(_snapshot())                     # 正常一步，建立基线
    check("基线：正常时给出战术动作", dec.action is not TacticalAction.SAFE_MODE)

    dec, _ = h.step(_snapshot(pos_valid=False))      # 定位失效当步
    check("定位失效 -> SAFE_MODE", dec.action is TacticalAction.SAFE_MODE,
          f"got {dec.action}")
    check("安全状态已降级", h.safety.state is not SafetyState.NORMAL,
          h.safety.state.name)

    # 持续失效 -> 升级到 SAFE_STOP；期间**不得**出现争夺类动作
    h2 = _Harness(_make_points())
    seen: list[TacticalAction] = []
    for _ in range(20):
        d, _ = h2.step(_snapshot(pos_valid=False))
        seen.append(d.action)
    check("持续定位丢失 -> SAFE_STOP", h2.safety.state is SafetyState.SAFE_STOP,
          h2.safety.state.name)
    check("全程无争夺类动作", all(a is not TacticalAction.CAPTURE_POINT for a in seen))

    # 数据恢复：不应立刻跳回 NORMAL（需要持续有效）
    d, _ = h2.step(_snapshot())
    check("刚恢复时不立刻回 NORMAL", h2.safety.state is not SafetyState.NORMAL,
          h2.safety.state.name)


def test_nan_is_permanent() -> None:
    print("\n[TC1b] NaN 输入 -> 永久 SAFE_STOP")
    h = _Harness(_make_points())
    h.step(_snapshot(x=float("nan")))
    check("NaN 当步 -> SAFE_STOP", h.safety.state is SafetyState.SAFE_STOP)
    for _ in range(50):
        h.step(_snapshot())                          # 之后一切正常
    check("数值异常后永久保持 SAFE_STOP",
          h.safety.state is SafetyState.SAFE_STOP, h.safety.state.name)


# --------------------------------------------------------------------------
# TC2 点位不可达
# --------------------------------------------------------------------------


def test_unreachable_points() -> None:
    print("\n[TC2] 点位不可达：不得反复选择不可达目标")
    h = _Harness(_make_points())
    snap = _snapshot(reach_l=False, reach_r=False, reach_safe=False)
    actions = []
    for _ in range(30):
        d, _ = h.step(snap)
        actions.append(d.action)
    check("无可达点 -> 只出现保守动作",
          all(a in (TacticalAction.HOLD_CURRENT_ZONE, TacticalAction.RECOVER)
              for a in actions),
          f"出现 {set(a.name for a in actions)}")
    check("不可达时不产生争夺动作",
          TacticalAction.CAPTURE_POINT not in actions)


def test_stuck_task_times_out() -> None:
    print("\n[TC2b] 任务超时：卡住必须能退出（失败后动作）")
    h = _Harness(_make_points())
    d, _ = h.step(_snapshot(cost_l=2.0))            # 去 P_L，ETA=2s
    check("进入赶路状态", h.machine.state is TaskState.MOVING, h.machine.state.name)
    # 故意用很小的宽限（1s + 2×ETA=5s），模拟"迟迟到不了"的最坏情况
    h.cfg.stuck_grace_s = 1.0
    for _ in range(60):
        d, _ = h.step(_snapshot(cost_l=2.0))
    stuck = [t for t in h.machine.history if "卡住" in t.reason]
    check("卡住后进入过 RECOVERING", len(stuck) >= 1,
          f"history={[(t.previous.name, t.current.name) for t in h.machine.history]}")
    check("迁移记录带原因", stuck and "卡住" in stuck[0].reason,
          stuck[0].reason if stuck else "无卡住迁移")
    # 恢复保持窗口：进入恢复后不能立刻又发同一个目标（否则等于没恢复）
    stuck = [t for t in h.machine.history if "卡住" in t.reason]
    if stuck:
        t0 = stuck[0].at_s
        after = [t for t in h.machine.history if t.at_s > t0 + 1e-9
                 and t.current is TaskState.MOVING]
        if after:
            gap = after[0].at_s - t0
            check(f"恢复保持 >= {h.cfg.recover_hold_s}s（实测 {gap:.1f}s）",
                  gap >= h.cfg.recover_hold_s - 1e-9, f"gap={gap:.2f}")
        else:
            check("恢复期内没有立刻重发目标", True)   # 60 步内未重试也算通过


# --------------------------------------------------------------------------
# TC3 控制区易手
# --------------------------------------------------------------------------


def test_zone_handover() -> None:
    print("\n[TC3] 控制区易手：NOBODY -> FRIENDLY -> ENEMY 的行为切换")
    h = _Harness(_make_points())
    # 只有左点可达 -> 决策必须明确指向左点（消除评分自由度）
    snap_l = _snapshot(owner_l=ZoneOwner.NOBODY, reach_r=False)
    d, _ = h.step(snap_l)
    check("空区 -> 争夺左点",
          d.action is TacticalAction.CAPTURE_POINT and d.point_id == "P_L",
          f"{d.action.name}/{d.point_id}")

    # 机器人已进入左区（区域语义：在半径 1.0 内即算到位），此时归属变为我方
    snap_f = _snapshot(x=3.0, y=4.0, owner_l=ZoneOwner.FRIENDLY, reach_r=False)
    for _ in range(5):
        d, _ = h.step(snap_f)
    check("我方已占 -> 转为防守",
          d.action is TacticalAction.DEFEND_POINT and d.point_id == "P_L",
          f"{d.action.name}/{d.point_id}")
    check("状态机进入执行", h.machine.state is TaskState.EXECUTING, h.machine.state.name)

    snap_e = _snapshot(x=3.0, y=4.0, owner_l=ZoneOwner.ENEMY, reach_r=False)
    d, _ = h.step(snap_e)
    check("敌方占领 -> 回到争夺",
          d.action is TacticalAction.CAPTURE_POINT, d.action.name)
    check("决策理由记录了归属变化", "ENEMY" in d.reason, d.reason[:80])


# --------------------------------------------------------------------------
# TC4 复活（读条 / 保护 / 虚弱）
# --------------------------------------------------------------------------


def test_respawn_sequence() -> None:
    print("\n[TC4] 复活序列：DEAD -> PROTECTED -> VULNERABLE -> 正常")
    h = _Harness(_make_points())
    d, _ = h.step(_snapshot(dead_secs=30.0))
    check("读条中 -> 停在原地", d.action is TacticalAction.HOLD_CURRENT_ZONE and d.point_id is None,
          f"{d.action.name}/{d.point_id}")
    check("阶段 = DEAD", h.dstate.respawn.phase.name == "DEAD",
          h.dstate.respawn.phase.name)

    # 复活：读条结束 -> 保护期
    d, _ = h.step(_snapshot(dead_secs=0.0, hp=0.2))
    check("刚复活 -> 保护期", h.dstate.respawn.phase.name == "RESPAWNED_PROTECTED",
          h.dstate.respawn.phase.name)
    check("保护期不动（不浪费无敌、也不盲目出击）",
          d.action is TacticalAction.HOLD_CURRENT_ZONE)

    # 保护期结束 -> 虚弱 -> 撤离到安全点
    for _ in range(int(h.cfg.respawn_invuln_s / h.cfg.decision_dt_s) + 2):
        d, _ = h.step(_snapshot(dead_secs=0.0, hp=0.2))
    check("保护期结束 -> 虚弱", h.dstate.respawn.phase.name == "RESPAWNED_VULNERABLE",
          h.dstate.respawn.phase.name)
    check("虚弱 -> 撤离且有目标安全点",
          d.action is TacticalAction.RETREAT and d.point_id == "P_SAFE",
          f"{d.action.name}/{d.point_id}")

    # 血量恢复 -> 虚弱倒计时走完 -> 回到战术行为（跑够虚弱时长 + 余量）
    n_after = int(h.cfg.vulnerable_secs_assumed / h.cfg.decision_dt_s) + 10
    for _ in range(n_after):
        d, _ = h.step(_snapshot(dead_secs=0.0, hp=0.9, owner_l=ZoneOwner.NOBODY))
    check("血量恢复 -> 回到战术行为",
          d.action in (TacticalAction.CAPTURE_POINT, TacticalAction.DEFEND_POINT,
                       TacticalAction.PATROL_POINT),
          d.action.name)


def test_rule_vulnerable_follows_state_not_timer() -> None:
    print("\n[TC4c] 2027 规则虚弱：跟随状态位，不跟随自编计时器")
    h = _Harness(_make_points())
    h.step(_snapshot(dead_secs=5.0))

    # 2027：复活瞬间环境同时置 S.VULN=1 -> 必须立即回补给区，而不是原地“保护” 30s
    d, _ = h.step(_snapshot(hp=0.2, vulnerable=True, vulnerable_known=True))
    check("复活即虚弱（状态位优先于保护期序列）",
          h.dstate.respawn.phase.name == "RESPAWNED_VULNERABLE",
          h.dstate.respawn.phase.name)
    check("立即撤离到安全点（补给区）",
          d.action is TacticalAction.RETREAT and d.point_id == "P_SAFE",
          f"{d.action.name}/{d.point_id}")
    check("理由说明需回补给区解除", "补给区" in d.reason, d.reason[:100])

    # 跑过任何旧计时器时长：只要状态位仍为 1，就不得自行结束虚弱
    n = int(max(h.cfg.respawn_invuln_s, h.cfg.vulnerable_secs_assumed)
            / h.cfg.decision_dt_s) + 20
    for _ in range(n):
        d, _ = h.step(_snapshot(hp=0.9, vulnerable=True, vulnerable_known=True))
    check("状态位未清 -> 虚弱不因计时器到期而结束",
          h.dstate.respawn.phase.name == "RESPAWNED_VULNERABLE",
          h.dstate.respawn.phase.name)
    check("血量正常时仍因规则虚弱保持撤离",
          d.action is TacticalAction.RETREAT and d.point_id == "P_SAFE",
          f"{d.action.name}/{d.point_id}")

    # 接触己方补给区（状态位清零）-> 立即恢复战术行为
    d, _ = h.step(_snapshot(hp=0.9, vulnerable=False, vulnerable_known=True))
    check("状态位清零 -> 虚弱立即结束",
          h.dstate.respawn.phase.name == "ALIVE", h.dstate.respawn.phase.name)
    check("虚弱解除 -> 恢复战术行为",
          d.action is TacticalAction.CAPTURE_POINT, d.action.name)

    # 中途接管 / 日志回放（没有死亡边沿）：也必须从状态位识别虚弱
    h2 = _Harness(_make_points())
    d, _ = h2.step(_snapshot(hp=0.9, vulnerable=True, vulnerable_known=True))
    check("无死亡边沿也能识别规则虚弱",
          h2.dstate.respawn.phase.name == "RESPAWNED_VULNERABLE",
          h2.dstate.respawn.phase.name)
    check("中途虚弱同样撤离",
          d.action is TacticalAction.RETREAT and d.point_id == "P_SAFE",
          f"{d.action.name}/{d.point_id}")

    # 数据源未声明已知（旧赛季）-> 不得改走状态位分支，仍用旧回退序列
    h3 = _Harness(_make_points())
    h3.step(_snapshot(dead_secs=5.0))
    h3.step(_snapshot(hp=0.2, vulnerable=True, vulnerable_known=False))
    check("源未知时即使 vulnerable=True 也走旧回退（保护期）",
          h3.dstate.respawn.phase.name == "RESPAWNED_PROTECTED",
          h3.dstate.respawn.phase.name)


def test_no_retreat_point_falls_back_to_hold() -> None:
    print("\n[TC4b] 无可用安全点 -> 原地保守（不得撤离到不可达点）")
    h = _Harness(_make_points())
    # HP 低 + 左点可达（会吸引选点）+ 安全点不可达
    d, _ = h.step(_snapshot(hp=0.1, reach_safe=False, owner_l=ZoneOwner.NOBODY))
    check("安全点不可达 -> 原地保持",
          d.action is TacticalAction.HOLD_CURRENT_ZONE and d.point_id is None,
          f"{d.action.name}/{d.point_id}")


# --------------------------------------------------------------------------
# TC5 信息过期 / 通信超时
# --------------------------------------------------------------------------


def test_stale_information() -> None:
    print("\n[TC5] 信息过期：过期信息不得当'安全'，也不得参与选点")
    h = _Harness(_make_points())
    d, _ = h.step(_snapshot(owner_l=ZoneOwner.NOBODY, reach_r=False, age_l=0.0))
    check("新鲜信息：正常争夺", d.action is TacticalAction.CAPTURE_POINT)

    # 我方标记为"已占领"，但信息已过期 -> 不得按"安全防守"处理
    h2 = _Harness(_make_points())
    stale = h2.cfg.point_stale_after_s + 1.0
    d, _ = h2.step(_snapshot(owner_l=ZoneOwner.FRIENDLY, reach_r=False,
                             age_l=stale))
    check("过期信息 -> 不选该点", d.point_id != "P_L", f"point={d.point_id}")
    check("过期信息 -> 也不当作敌方来争夺",
          d.action is not TacticalAction.CAPTURE_POINT or d.point_id != "P_L",
          d.action.name)


# --------------------------------------------------------------------------
# TC6 防抖：目标反复切换
# --------------------------------------------------------------------------


def test_no_oscillation() -> None:
    print("\n[TC6] 防抖：评分接近时不得在两点间反复切换")
    h = _Harness(_make_points())
    # 交替微调两侧代价，制造"每步评分几乎相同、互相反超"的最坏情况
    switches = 0
    prev = None
    for i in range(40):
        snap = _snapshot(cost_l=4.0 + (0.05 if i % 2 else 0.0),
                         cost_r=4.0 + (0.0 if i % 2 else 0.05))
        d, _ = h.step(snap)
        if prev is not None and d.point_id != prev:
            switches += 1
        prev = d.point_id
    check(f"40 步内换点次数 <= 2（实测 {switches}）", switches <= 2, f"switches={switches}")


def test_clear_advantage_still_switches() -> None:
    print("\n[TC6b] 迟滞不能把策略钉死：明显更优时必须换点")
    h = _Harness(_make_points())
    for _ in range(20):
        h.step(_snapshot(cost_l=2.0, cost_r=2.0))
    d0, _ = h.step(_snapshot(cost_l=2.0, cost_r=2.0))
    point0 = d0.point_id
    # 左点变得极差（代价爆表）+ 右点极好
    for _ in range(3):
        d, _ = h.step(_snapshot(cost_l=20.0, cost_r=0.5))
    check("明显更优 -> 换点", d.point_id != point0,
          f"{point0} -> {d.point_id}")


# --------------------------------------------------------------------------
# 结构约束 / 接口稳定性
# --------------------------------------------------------------------------


def test_zone_binding_enforced() -> None:
    print("\n[TC7] 点到区域绑定：控制类点位必须绑区域")
    try:
        _ = TacticalPoints(
            [ZoneSpec("Z1", "区", 1.0, 1.0, 1.0)],
            [PointSpec("P1", "无绑定控制点", PointKind.CONTROL, 1.0, 1.0, None)],
        )
        check("未绑定区域的控制点应被拒绝", False, "竟然构造成功了")
    except ValueError as e:
        check("未绑定区域的控制点应被拒绝", "必须绑定区域" in str(e), str(e))


def test_action_interface_stable() -> None:
    print("\n[TC8] RL 接口：动作集与需要目标的动作必须显式声明")
    from rm_sentry.tactical import ACTION_NEEDS_TARGET
    check("动作集非空", len(list(TacticalAction)) >= 6)
    check("需要目标的动作集合正确",
          ACTION_NEEDS_TARGET == {TacticalAction.CAPTURE_POINT,
                                  TacticalAction.DEFEND_POINT,
                                  TacticalAction.PATROL_POINT},
          str(ACTION_NEEDS_TARGET))
    ids = _make_points().point_ids()
    check("点 ID 稳定排序（RL 动作索引依赖它）", list(ids) == sorted(ids), str(ids))


def test_weights_are_normalized() -> None:
    print("\n[TC9] 评分配置：权重和必须为 1.0（否则迟滞阈值失效）")
    try:
        ScoringWeights(w_value=0.9, w_control=0.9)
        check("非法权重应被拒绝", False, "竟然构造成功了")
    except ValueError as e:
        check("非法权重应被拒绝", "必须为 1.0" in str(e), str(e))


def test_config_matches_env() -> None:
    print("\n[TC10] 与环境规则常量的一致性（防止两处漂移）")
    from rm_sentry.rmul_3v3_env import RMUL3v3Config
    env = RMUL3v3Config(season="2026")
    tac = TacticalConfig()
    check("复活无敌时长一致",
          abs(tac.respawn_invuln_s - env.respawn_invuln) < 1e-9,
          f"tac={tac.respawn_invuln_s} env={env.respawn_invuln}")
    check("决策周期与 decision_hz 一致（10 Hz -> 0.1s）",
          abs(tac.decision_dt_s - 1.0 / env.decision_hz) < 1e-9,
          f"tac={tac.decision_dt_s} env={1.0/env.decision_hz}")


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

TESTS = [
    test_localization_loss,
    test_nan_is_permanent,
    test_unreachable_points,
    test_stuck_task_times_out,
    test_zone_handover,
    test_respawn_sequence,
    test_rule_vulnerable_follows_state_not_timer,
    test_no_retreat_point_falls_back_to_hold,
    test_stale_information,
    test_no_oscillation,
    test_clear_advantage_still_switches,
    test_zone_binding_enforced,
    test_action_interface_stable,
    test_weights_are_normalized,
    test_config_matches_env,
]


def main() -> int:
    print("=" * 74)
    print("战术层回归测试（几何无关；点图坐标均为测试构造值）")
    print("=" * 74)
    for t in TESTS:
        try:
            t()
        except Exception:  # 单测崩溃也算失败，但继续跑完后
            BAD.append(t.__name__)
            print(f"  [FAIL] {t.__name__} 抛出异常：")
            traceback.print_exc()
    print("\n" + "=" * 74)
    print(f"合计：通过 {len(OK)} 项，失败 {len(BAD)} 项")
    for n in BAD:
        print(f"  失败：{n}")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
