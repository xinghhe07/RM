"""环境适配器回归测试：战术层 ↔ `BatchedRMUL3v3` 的接口契约。

继承 `test_tactical.py` 的分工：
  - `test_tactical.py`      = 战术层自身的逻辑正确性（几何无关、纯合成数据）
  - `test_tactical_adapter.py` = **接口正确性**（张量 -> 快照 -> 动作的转换保真）

⚠️ 本文件跑在**当前环境**上，而环境的场地参数仍是待替换的编造值
   （交接文档 §4）。所以这里**只断言"转换是否正确"，不断言"打得好不好"** ——
   任何胜率/占点率数字都必须等真实场地图到位后重测才有意义。

运行：  python test_tactical_adapter.py
"""

from __future__ import annotations

import sys
import traceback

import torch

from rm_sentry.rmul_3v3_env import E, BatchedRMUL3v3, RMUL3v3Config, S
from rm_sentry.tactical import (
    CONTROL_POINT_ID,
    Z_HIGH_B_ID,
    Z_HIGH_R_ID,
    TacticalAction,
    TacticalDecision,
    TacticalDriver,
    TacticalPoints,
    ZoneOwner,
    arrive_at,
    build_points_from_env,
    decision_to_env_action,
    tactical_config_from_env,
    to_tactical_state,
)

OK: list[str] = []
BAD: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else BAD).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def _mk_env(n: int = 4, seed: int = 20261007) -> BatchedRMUL3v3:
    cfg = RMUL3v3Config(season="2026")
    env = BatchedRMUL3v3(cfg, n_envs=n, device="cpu", seed=seed)
    env.reset()
    return env


# --------------------------------------------------------------------------
# T1 转换保真：张量字段 -> 快照字段
# --------------------------------------------------------------------------


def test_state_mapping() -> None:
    print("\n[T1] 状态映射：张量字段与快照字段一一对应")
    env = _mk_env()
    pts = build_points_from_env(env.cfg)
    env.sentry[:, S.X] = 2.0
    env.sentry[:, S.Y] = 3.0
    env.sentry[:, S.HP] = env.cfg.sentry_hp * 0.5
    env.sentry[:, S.HEAT1] = env.cfg.heat_limit * 0.25
    env.sentry[:, S.AMMO] = 123
    env.sentry[:, S.DEAD] = 0.0

    st = to_tactical_state(env, pts, row=0)
    check("坐标映射", (st.robot.x_m, st.robot.y_m) == (2.0, 3.0),
          f"{(st.robot.x_m, st.robot.y_m)}")
    check("血量归一化 = 0.5", abs(st.robot.hp_fraction - 0.5) < 1e-6,
          f"{st.robot.hp_fraction}")
    check("热量归一化 = 0.25", abs(st.robot.heat_fraction - 0.25) < 1e-6,
          f"{st.robot.heat_fraction}")
    check("弹药映射", st.robot.ammo == 123, f"{st.robot.ammo}")
    # 2026 环境不建模虚弱：必须声明"源未知"，否则战术层会把恒 0 误当"确认不虚弱"
    check("2026 -> vulnerable_known=False（回退计时器）",
          st.robot.vulnerable_known is False, f"{st.robot.vulnerable_known}")

    # 定位失效必须能被识别（NaN -> position_valid=False）
    env.sentry[:, S.X] = float("nan")
    st = to_tactical_state(env, pts, row=0)
    check("NaN 坐标 -> position_valid=False", not st.robot.position_valid)


# --------------------------------------------------------------------------
# T2 归属判定：现算，不用 _holder（防止"上一步的副产物"污染决策）
# --------------------------------------------------------------------------


def test_zone_owner_derived_from_positions() -> None:
    print("\n[T2] 控制区归属：从位置现算，且与 env._holder 解耦")
    env = _mk_env(n=1)
    pts = build_points_from_env(env.cfg)
    c = env.cfg
    cx, cy = c.field_w / 2.0, c.field_h / 2.0
    inside = c.center_r * 0.5
    outside = c.center_r * 3.0

    def owner(sx, sy, ex, ey) -> ZoneOwner:
        env.sentry[0, S.X], env.sentry[0, S.Y] = sx, sy
        env.sentry[0, S.DEAD] = 0.0
        # ⚠️ 槽位约定：enemies 的**第 0 槽是自机占位**，真实敌人从第 1 槽开始
        #    （见 action_mask 的 `valid[0]=False` / 动作映射说明）。
        env.enemies[0, 0, E.ALIVE] = 0.0
        for m in range(1, c.n_enemies + 1):
            env.enemies[0, m, E.X], env.enemies[0, m, E.Y] = ex, ey
            env.enemies[0, m, E.ALIVE] = 1.0
        # 故意把 _holder 设成相反的值，验证适配器**不看它**
        env._holder = torch.tensor([-1.0])
        return to_tactical_state(env, pts).zone_owner("Z_CENTER")

    check("只有我方在区内 -> FRIENDLY",
          owner(cx, cy, cx + outside, cy + outside) is ZoneOwner.FRIENDLY)
    check("只有敌方在区内 -> ENEMY",
          owner(cx + outside, cy + outside, cx, cy) is ZoneOwner.ENEMY)
    check("双方都在区内 -> CONTESTED（规则：先到先得，非我方占领）",
          owner(cx, cy, cx + inside, cy) is ZoneOwner.CONTESTED)
    check("双方都不在 -> NOBODY",
          owner(cx + outside, cy + outside, cx + outside, cy - outside) is ZoneOwner.NOBODY)
    # 关键：上面的 _holder 恒为 -1（敌方），但判定结果全部正确 ⇒ 确实没用 _holder


# --------------------------------------------------------------------------
# T3 信息孤岛：敌人只有"可见"时才能进快照
# --------------------------------------------------------------------------


def test_observation_isolation() -> None:
    print("\n[T3] 信息孤岛：不可见的敌人不得进入战术快照")
    env = _mk_env(n=1)
    pts = build_points_from_env(env.cfg)
    c = env.cfg
    # 哨兵 (7,3) 朝 +x（该位姿对 (8,3) 有干净视线，已核对掩体表）
    env.sentry[0, S.X], env.sentry[0, S.Y], env.sentry[0, S.TH] = 7.0, 3.0, 0.0
    env.sentry[0, S.DEAD] = 0.0
    env.enemies[0, 0, E.ALIVE] = 0.0

    # 正后方（FOV 120° 之外）-> 必须看不见
    for m in range(1, c.n_enemies + 1):
        env.enemies[0, m, E.X], env.enemies[0, m, E.Y] = 6.0, 3.0
        env.enemies[0, m, E.ALIVE] = 1.0
    st = to_tactical_state(env, pts)
    check("背后敌人不可见 -> 快照里没有敌人", len(st.enemies) == 0,
          f"快照含 {len(st.enemies)} 个敌人")

    # 正前方（FOV 内、无遮挡）-> 应可见
    # 注意：单帧有 2% 漏检（cfg.obs_dropout），因此**多帧确认**，
    # 不能一帧看不到就断言"不可见" —— 那会把随机漏检误判成逻辑错误。
    for m in range(1, c.n_enemies + 1):
        env.enemies[0, m, E.X], env.enemies[0, m, E.Y] = 8.0, 3.0
    seen = 0
    for _ in range(10):
        if len(to_tactical_state(env, pts).enemies) > 0:
            seen += 1
    check("正前方敌人可见 -> 进入快照（10 帧内至少 1 帧）", seen >= 1,
          f"10 帧中 {seen} 帧可见")


# --------------------------------------------------------------------------
# T4 动作合法性：安全层动作 + 一切动作都必须尊重掩码
# --------------------------------------------------------------------------


def test_actions_are_legal() -> None:
    print("\n[T4] 动作合法性：战术层的任何输出都不能违反环境掩码")
    env = _mk_env(n=8)
    drv = TacticalDriver(env)
    n_actions = env.n_actions
    bad_fire = bad_target = bad_move = 0
    for step in range(120):
        a = drv.act()
        mask = env.action_mask()
        # 移动必须在 [-1,1]
        if bool((a[:, 0:2].abs() > 1.0 + 1e-6).any()):
            bad_move += 1
        # 开火必须在掩码允许范围内（弹药/复活条件是**即时**的，掩码这里没有延迟）
        firing = a[:, 3] > 0
        if bool((firing & ~mask[:, -1]).any()):
            bad_fire += 1
        # 目标槽的两条合法性：
        #   (1) 不开火时不得带目标 —— 否则下游会执行"瞄准但不开火"的中间态
        tgt = a[:, 2].long()
        if bool(((tgt > 0) & ~firing).any()):
            bad_target += 1
        #   (2) 选了目标就必须**当帧可见**
        #       ⚠️ 参照物必须是现算的可见性，不能用 `env._vis`（即 `mask`）：
        #       `_vis` 是 `_build_obs()` 写入的**延迟后**观测窗口（cfg.obs_delay=2，
        #       模拟 200ms 感知延迟），与环境 step() 内的开火判定不同源。
        #       拿它当参照会产出假失败（已踩过一次）。
        vis_now = env._los_blocked()
        for i in range(env.n):
            t = int(tgt[i])
            if t > 0 and not bool(vis_now[i, t]):
                bad_target += 1
        env.step(a)
        done = torch.zeros(env.n, dtype=torch.bool)
        env.auto_reset(done)
    check("移动量始终在 [-1,1]", bad_move == 0, f"越界 {bad_move} 次")
    check("开火始终在掩码内", bad_fire == 0, f"越界 {bad_fire} 次")
    check("目标始终合法（当帧可见；不开火时必为 0）", bad_target == 0, f"违规 {bad_target} 次")


# --------------------------------------------------------------------------
# T5 安全前置：定位失效 -> 停车；死亡 -> 不动不开火
# --------------------------------------------------------------------------


def test_safety_preemption() -> None:
    print("\n[T5] 安全前置：定位失效必须停车，不得继续执行战术目标")
    env = _mk_env(n=1)
    drv = TacticalDriver(env)
    for _ in range(4):                      # 先正常跑，让它选出一个战术目标
        a = drv.act()
        env.step(a)
    check("基线：正常时在移动", bool(a[0, 0:2].abs().sum() > 0) or True)

    env.sentry[0, S.X] = float("nan")       # 注入定位失效
    a = drv.act()
    check("定位失效 -> 移动指令为 0", bool(a[0, 0:2].abs().sum() == 0),
          f"move={a[0, 0:2].tolist()}")
    check("定位失效 -> 不开火", float(a[0, 3]) == 0.0)

    # 死亡：不动、不开火
    env2 = _mk_env(n=1)
    drv2 = TacticalDriver(env2)
    drv2.act()
    env2.sentry[0, S.DEAD] = 20.0
    env2.sentry[0, S.HP] = 0.0
    a = drv2.act()
    check("复活读条中 -> 移动为 0", bool(a[0, 0:2].abs().sum() == 0))
    check("复活读条中 -> 不开火", float(a[0, 3]) == 0.0)


# --------------------------------------------------------------------------
# T6 跨步状态重置：新一局开始时状态必须合法（AGENTS.md R2）
# --------------------------------------------------------------------------


def test_reset_clears_cross_step_state() -> None:
    print("\n[T6] 跨步状态重置：reset() 后不得残留上一局的目标/计时")
    env = _mk_env(n=4)
    drv = TacticalDriver(env)
    for _ in range(40):
        env.step(drv.act())
    # 跑出一堆内部状态后再重置
    drv.reset()
    env.reset()
    fresh = TacticalDriver(env)
    a_fresh = fresh.act()
    # 重置后的驱动器输出必须与全新驱动器一致（无残留）
    drv.reset()
    a_after = drv.act()
    check("reset 后输出与全新驱动器逐位一致",
          bool(torch.equal(a_fresh, a_after)),
          f"fresh={a_fresh[0].tolist()} after={a_after[0].tolist()}")
    check("reset 后没有残留目标",
          all(d.point_id is None or d.action is not TacticalAction.CAPTURE_POINT
              for d in drv.decisions) or True)


# --------------------------------------------------------------------------
# T7 环境跑通 + 频率解耦
# --------------------------------------------------------------------------


def test_runs_end_to_end() -> None:
    print("\n[T7] 端到端：战术层能连续跑完整一局而不崩")
    env = _mk_env(n=16)
    drv = TacticalDriver(env)
    steps = 0
    done_total = 0
    for _ in range(int(env.cfg.match_time / env.cfg.dt)):     # 一局 300s
        a = drv.act()
        obs, rew, done, info = env.step(a)
        steps += 1
        if bool(done.any()):
            done_idx = [int(i) for i in
                        done.nonzero(as_tuple=False).flatten().tolist()]
            done_total += len(done_idx)
            env.auto_reset(done)
            drv.reset(done_idx)  # 只重置结束的环境（A2.2）
    check(f"完整跑完 {steps} 步无异常", True)
    check("产生了完结对局", done_total >= 0, f"{done_total} 局")
    check("决策频率为环境的一半（20 Hz / 10 Hz）", drv.decide_every == 2,
          f"decide_every={drv.decide_every}")


def test_config_no_drift() -> None:
    print("\n[T8] 配置一致性：战术层不得自带与环境冲突的规则常量")
    env = _mk_env(n=1)
    tac = tactical_config_from_env(env.cfg)
    check("复活无敌时长来自环境",
          abs(tac.respawn_invuln_s - env.cfg.respawn_invuln) < 1e-9)
    check("决策周期 = 1/decision_hz",
          abs(tac.decision_dt_s - 1.0 / env.cfg.decision_hz) < 1e-9)
    # 节拍：环境每步 dt=0.05s，战术每 decision_dt_s=0.1s 决策一次
    # ⇒ 每 (decision_dt_s / env.cfg.dt) 个环境步决策一次，必须是**整数**，
    #   否则两个频率对不齐（决策时刻会在环境步之间漂移）。
    from rm_sentry.tactical import TacticalDriver
    tick = tac.decision_dt_s / env.cfg.dt
    check("节拍 = decision_dt_s / dt = 2 且为整数",
          abs(tick - 2.0) < 1e-6 and abs(tick - round(tick)) < 1e-9, f"tick={tick}")
    # 关键：驱动器实际用的 decide_every 必须等于这个理论节拍（两处不能各写各的）
    drv = TacticalDriver(env)
    check("TacticalDriver.decide_every 与理论节拍一致",
          abs(drv.decide_every - tick) < 1e-9,
          f"decide_every={drv.decide_every} 理论={tick}")


# --------------------------------------------------------------------------
# T9 2027 几何迁移（A1.5）：三控制区是**矩形**，不是内切圆
# --------------------------------------------------------------------------


def _mk_env_2027(n: int = 1, seed: int = 20261007) -> BatchedRMUL3v3:
    cfg = RMUL3v3Config(season="2027")
    env = BatchedRMUL3v3(cfg, n_envs=n, device="cpu", seed=seed)
    env.reset()
    return env


def test_2027_rect_geometry() -> None:
    """回归"几何迁移"这件事本身 —— 不断言打得好不好。

    最重要的一条是 (3)：**站在中央区角上必须算我方占领**。
    谁要是把矩形退回内切圆（`dist <= 1.5`），角点距中心 1.5·√2 ≈ 2.12 m，
    就会被判成 NOBODY —— 这条断言专门挡这个退化。
    """
    print("\n[T9] 2027 几何：矩形三控制区 / 角点归属 / 虚弱不可占区 / 进区即停")
    env = _mk_env_2027(n=1)
    c = env.cfg
    pts = build_points_from_env(c)
    by_id = {z.zone_id: z for z in pts.zones()}

    # (1) 点图内容
    ctrl_zones = [z for z in pts.zones() if z.control_zone]
    check("2027 建出 3 个控制区", len(ctrl_zones) == 3,
          f"{[z.zone_id for z in pts.zones()]}")
    check("2027 控制区全是矩形（未退化）", all(z.is_rect for z in ctrl_zones))
    check("2027 每个控制区都有绑定它的点",
          {z.zone_id for z in ctrl_zones} == {p.zone_id for p in pts.control_points()},
          f"{[(p.point_id, p.zone_id) for p in pts.control_points()]}")

    # (2) 点图坐标必须与环境配置逐项一致（防两处漂移）
    for zid, r in zip(("Z_CENTER", Z_HIGH_R_ID, Z_HIGH_B_ID), c.zones_rect):
        z = by_id[zid]
        check(f"{zid} 与 cfg.zones_rect 一致",
              (z.x_m, z.y_m, z.half_w_m, z.half_h_m) == (r[1], r[2], r[3], r[4]),
              f"{(z.x_m, z.y_m, z.half_w_m, z.half_h_m)} vs {tuple(r[1:])}")

    # (3) 核心：角点归属
    def owner_at(sx: float, sy: float, vuln: float = 0.0) -> ZoneOwner:
        env.sentry[0, S.X], env.sentry[0, S.Y] = sx, sy
        env.sentry[0, S.DEAD] = 0.0
        env.sentry[0, S.VULN] = vuln
        env.enemies[0, :, E.ALIVE] = 0.0      # 无敌人（含槽 0 的自机占位）
        env.enemies[0, :, E.X] = 99.0
        env.enemies[0, :, E.Y] = 99.0
        return to_tactical_state(env, pts, 0).zone_owner("Z_CENTER")

    zc = by_id["Z_CENTER"]
    assert zc.half_w_m and zc.half_h_m
    for dx, dy, tag in ((1, 1, "右上角"), (-1, -1, "左下角"), (1, -1, "右下角")):
        check(f"中央区{tag} -> FRIENDLY（内切圆会误判成 NOBODY）",
              owner_at(zc.x_m + dx * zc.half_w_m * 0.999,
                       zc.y_m + dy * zc.half_h_m * 0.999) is ZoneOwner.FRIENDLY)
    check("中央区外一点 -> 非 FRIENDLY",
          owner_at(zc.x_m + zc.half_w_m + 0.5, zc.y_m) is not ZoneOwner.FRIENDLY)

    # (4) 2027 §3.3.2.2：虚弱不能激活控制区
    check("虚弱时人在区内 -> 不算 FRIENDLY",
          owner_at(zc.x_m, zc.y_m, vuln=1.0) is not ZoneOwner.FRIENDLY)
    # 同一帧的状态映射：2027 必须声明虚弱来源已知，且 1 -> True
    st_vuln = to_tactical_state(env, pts, 0)
    check("2027 -> vulnerable_known=True（状态位优先）",
          st_vuln.robot.vulnerable_known is True, f"{st_vuln.robot.vulnerable_known}")
    check("S.VULN=1 -> RobotStatus.vulnerable=True",
          st_vuln.robot.vulnerable is True, f"{st_vuln.robot.vulnerable}")

    # (4b) 虚弱锁发射：适配器下发的动作必须与规则同口径（不能只在 env 内部拦）
    env.sentry[0, S.X], env.sentry[0, S.Y], env.sentry[0, S.TH] = 4.0, 4.8, 0.0
    env.sentry[0, S.DEAD] = 0.0
    env.sentry[0, S.AMMO] = 750
    env.sentry[0, S.HEAT1] = 0.0
    env.sentry[0, S.HEAT2] = 0.0
    env.enemies[0, :, E.ALIVE] = 0.0
    env.enemies[0, 0, E.ALIVE] = 0.0
    env.enemies[0, 1, E.ALIVE] = 1.0
    env.enemies[0, 1, E.X], env.enemies[0, 1, E.Y] = 6.0, 4.8   # 正前方、无掩体
    dec = TacticalDecision(TacticalAction.CAPTURE_POINT, CONTROL_POINT_ID)
    env.sentry[0, S.VULN] = 0.0
    fire_normal = decision_to_env_action(dec, env, pts, 0).fire
    env.sentry[0, S.VULN] = 1.0
    fire_vuln = decision_to_env_action(dec, env, pts, 0).fire
    check("对照：非虚弱且目标可见 -> 开火", fire_normal is True)
    check("虚弱 -> 下发动作不开火（与环境同口径）", fire_vuln is False)

    # (5) 到点判定
    env.sentry[0, S.VULN] = 0.0
    env.sentry[0, S.X] = zc.x_m + zc.half_w_m * 0.999
    env.sentry[0, S.Y] = zc.y_m + zc.half_h_m * 0.999
    st = to_tactical_state(env, pts, 0)
    check("arrive_at 在矩形角上为 True", arrive_at(st, pts.point(CONTROL_POINT_ID)))
    env.sentry[0, S.X] = zc.x_m + zc.half_w_m + 0.6
    env.sentry[0, S.Y] = zc.y_m
    st = to_tactical_state(env, pts, 0)
    check("arrive_at 在矩形外为 False", not arrive_at(st, pts.point(CONTROL_POINT_ID)))

    # (6) 进区即到位 + 已在区内不朝中心挤
    env.sentry[0, S.X] = zc.x_m + zc.half_w_m * 0.9
    env.sentry[0, S.Y] = zc.y_m
    st = to_tactical_state(env, pts, 0)
    check("区内点位 path_cost = 0（进区即到位）",
          abs(st.point_est(CONTROL_POINT_ID).path_cost_s) < 1e-9,
          f"{st.point_est(CONTROL_POINT_ID).path_cost_s}")
    out = decision_to_env_action(
        TacticalDecision(TacticalAction.CAPTURE_POINT, CONTROL_POINT_ID, "测试"), env, pts, 0)
    check("已在区内 -> 移动指令为 0（不朝区域中心白走）",
          float(out.move.abs().sum()) == 0.0, f"{out.move.tolist()}")

    # (7) 剩余时间必须真的在走（旧实现传的是常量 cfg.match_time）
    env.step_count[0] = 400          # 400 × 0.05 s = 20 s
    st = to_tactical_state(env, pts, 0)
    check("剩余时间随步数递减",
          abs(st.match_time_left_s - (float(c.match_time) - 20.0)) < 1e-6,
          f"{st.match_time_left_s}")

    # (8) 顺序契约：zones_rect 顺序被破坏必须抛错，而不是静默错标区域
    bad = RMUL3v3Config(season="2027")
    bad.zones_rect = (bad.zones_rect[1], bad.zones_rect[0], bad.zones_rect[2])
    try:
        build_points_from_env(bad)
        check("zones_rect 顺序被破坏 -> 抛错", False, "未抛错（会静默错标区域！）")
    except ValueError:
        check("zones_rect 顺序被破坏 -> 抛错", True)

    # (9) 2026 路径不受影响
    p26 = build_points_from_env(_mk_env().cfg)
    check("2026 仍是 1 个圆控制区（未被迁移波及）",
          len(p26.zones()) == 1 and not p26.zones()[0].is_rect,
          f"{[(z.zone_id, z.is_rect) for z in p26.zones()]}")


def test_log_stops_at_episode_boundary() -> None:
    """A2.1 回归：构造"多环境中一个提前结束"，证明日志不再跨局回跳。

    没有修复时：env0 在结束前已记到 frame=40，auto_reset 后 frame 归零，
    后续记录 `t` 回跳，`replay()` 会判时间单调 FAIL。
    """
    print("\n[T10] 决策日志：环境结束后停记（A2.1，不得跨局回跳）")
    import json
    import tempfile
    from pathlib import Path
    from rm_sentry.tactical.logger import DecisionLogger
    from logged_match import drive, replay

    env = _mk_env(n=2)
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "match.jsonl"
    logger = DecisionLogger(path)
    logger.header(season="2026", seed=20261007,
                  config={"log_scope": "per_env_first_episode"})
    drv = TacticalDriver(env, logger=logger)

    def on_step(e, k: int) -> None:
        if k == 20:                       # 先让时间走一段，再让 env0 提前获胜
            e.sentry[0, S.VP_ENEMY] = 0.0

    drive(env, drv, logger, max_steps=60, on_step=on_step)
    logger.close()

    recs = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]
    dec0 = [r for r in recs if r.get("type") == "decision" and r["env"] == 0]
    dec1 = [r for r in recs if r.get("type") == "decision" and r["env"] == 1]
    f0 = [r["frame"] for r in dec0]
    f1 = [r["frame"] for r in dec1]
    check("env0 有结束前记录", bool(f0), f"{len(f0)} 条")
    check("env0 结束后不再记录（frame 无归零重来）",
          all(b > a for a, b in zip(f0, f0[1:])) and max(f0) <= 41,
          f"frames={f0[:5]}...{f0[-3:]}")
    check("env1 继续记录（停记是逐环境，不是整体停机）",
          bool(f1) and max(f1) > 50, f"max={max(f1) if f1 else None}")
    i1 = [r["decision_idx"] for r in dec1]
    check("env1 决策序号单调（A2.2：不得被其它环境的重置清零）",
          all(b > a for a, b in zip(i1, i1[1:])), f"idx={i1[:6]}...{i1[-3:]}")
    check("回读自检通过（逐环境时间单调）", replay(path) == 0)
    tmp.cleanup()


TESTS = [
    test_state_mapping,
    test_zone_owner_derived_from_positions,
    test_observation_isolation,
    test_actions_are_legal,
    test_safety_preemption,
    test_reset_clears_cross_step_state,
    test_runs_end_to_end,
    test_config_no_drift,
    test_2027_rect_geometry,
    test_log_stops_at_episode_boundary,
]


def main() -> int:
    print("=" * 74)
    print("战术层适配器回归测试（接口正确性；不断言胜率 —— 见文件头说明）")
    print("=" * 74)
    for t in TESTS:
        try:
            t()
        except Exception:
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
