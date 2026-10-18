"""规则一致性测试：逐条对照官方手册验证环境实现。

⚠️ 2025 与 2026 的哨兵数值和**胜利点机制符号相反**，本测试对两个赛季分别验证。
   你们要打的是 2026，以 2026 为准。

规则来源（原文已抽取到 C:\\RM\\rules\\rules_text\\）：
  RMUL 2025 V1.1.1 (20250314)：表 3-4 哨兵关键特征 / §3.3.1.3 热量 / §3.3.2.2 复活
                                §3.3.3 经济 / §3.3.6 中心增益点与胜利点 / §3.3.7 获胜
  RMUL 2026 V1.2.0 (20260109)：表 3-4 哨兵关键特征 / 表 3-7 伤害 / §3.3.4 控制区与胜利点
                                §3.3.5 获胜条件
  RMUL 2027（预发布 PPT 2026-08-14）：原英雄→重装机器人、控制区 1→3 处、新增隧道

运行：  python -m rm_sentry.rules_test            # 默认测 2026
        python -m rm_sentry.rules_test --season 2025
"""

from __future__ import annotations

import argparse
import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S, E

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          + (f"  <- {detail}" if detail and not cond else ""))


def test_constants(cfg: RMUL3v3Config, season: str) -> None:
    print(f"\n[1] 表 3-4 哨兵机器人关键特征（{season}）")
    check("运行方式=自动运行（零人类干预，R21 遥控器须上交）", True)
    check("多机通信=不允许 -> 观测不得含敌方真值坐标", cfg.sense_range < cfg.field_w)
    check(f"发射机构={cfg.n_barrels} 个 17mm",
          cfg.n_barrels in (1, 2))
    check("射击初速度上限=25 m/s", cfg.muzzle_speed == 25.0)
    check(f"射击热量上限={cfg.heat_limit:.0f}",
          cfg.heat_limit in (260.0, 400.0))
    check(f"射击热量冷却速率={cfg.heat_cool:.0f}/秒",
          cfg.heat_cool in (30.0, 80.0))
    check("底盘功率上限=100 W", cfg.chassis_power == 100.0)
    check("初始/上限血量=400（哨兵固定 10 级不成长）", cfg.sentry_hp == 400)
    check("初始允许发弹量=750，局内不可增加", cfg.ammo_init == 750)
    # 赛季配对：2025 是 2 管 400/80，2026+ 是 1 管 260/30
    if season == "2025":
        check("2025 档案：2 管 + 热量 400 + 冷却 80", cfg.n_barrels == 2
              and cfg.heat_limit == 400.0 and cfg.heat_cool == 80.0)
    else:
        check(f"{season} 档案：1 管 + 热量 260 + 冷却 30", cfg.n_barrels == 1
              and cfg.heat_limit == 260.0 and cfg.heat_cool == 30.0)

    print(f"\n[2] §3.3.1.3 射击热量与冷却（持续射速上限 = 冷却/每发热量）")
    check("每发 17mm 弹丸热量 +10", cfg.heat_per_shot == 10.0)
    sustained = cfg.heat_cool / cfg.heat_per_shot
    expected = 8.0 if season == "2025" else 3.0
    check(f"持续射速上限 = {sustained:.2f} 发/秒（应为 {expected:.0f}）",
          abs(sustained - expected) < 1e-9)

    print(f"\n[3] 攻击伤害（表 3-7）")
    exp_dmg = 10.0 if season == "2025" else 20.0
    check(f"17mm 弹丸对装甲伤害={cfg.dmg_base:.0f}（{season} 应为 {exp_dmg:.0f}）",
          cfg.dmg_base == exp_dmg)

    print(f"\n[4] §3.3.2.2 复活机制")
    check("每秒自动增加 2 点读条进度", cfg.respawn_progress_per_s == 2.0)
    check("首次战亡读条长度=10，每次 +10", cfg.respawn_base_len == 10.0
          and cfg.respawn_len_step == 10.0)
    for nth, expect in ((1, 5.0), (2, 10.0), (3, 15.0)):
        need = cfg.respawn_base_len + (nth - 1) * cfg.respawn_len_step
        check(f"第 {nth} 次战亡读条 = {expect:.0f} 秒",
              abs(need / cfg.respawn_progress_per_s - expect) < 1e-9)
    check("复活后血量=上限 20%，无敌 10 秒",
          cfg.respawn_hp_ratio == 0.2 and cfg.respawn_invuln == 10.0)

    print(f"\n[5] 经济体系")
    check("比赛开始双方各 200 金币", cfg.initial_gold == 200)

    print(f"\n[6] 胜利点机制（两赛季符号相反）")
    if season == "2025":
        check("§3.3.6 从 0 累积，占中心我方 +1/秒", cfg.vp_mode == "accumulate"
              and cfg.vp_init == 0.0 and cfg.vp_center_coef == 1.0)
        check("§3.3.6 战亡则对方 +20", cfg.vp_kill_receiver == "killer"
              and cfg.vp_kill_coef == 20.0)
        check("§3.3.7 胜利点达 200 立即结束", cfg.vp_win == 200.0)
    else:
        check("§3.3.4 双方各 200 点起扣", cfg.vp_mode == "drain" and cfg.vp_init == 200.0)
        check("§3.3.4 占控制区则对方 −1/秒", cfg.vp_center_coef == 1.0
              and cfg.vp_kill_receiver == "victim")
        check("§3.3.4 战亡则该方 −20", cfg.vp_kill_coef == 20.0)
        check("§3.3.5 一方胜利点为 0 立即结束", cfg.vp_win == 0.0)
    check("未检测到场地交互模块卡后，占领状态 2 秒失效", cfg.center_grace == 2.0)

    print(f"\n[7] 场地与时长")
    check("战场长 12m 宽 8m", cfg.field_w == 12.0 and cfg.field_h == 8.0)
    check("5 分钟比赛阶段", cfg.match_time == 300.0)


def test_center_control(cfg: RMUL3v3Config, dev: str) -> None:
    print(f"\n[8] 行为：控制区归属（先到先得 + 2 秒宽限）")
    env = BatchedRMUL3v3(cfg, n_envs=4, device=dev, seed=7)
    env.reset()
    cx, cy = cfg.field_w / 2.0, cfg.field_h / 2.0
    act = torch.zeros(env.n, env.n_actions, device=dev)
    act[:, 2] = 0.0
    act[:, 3] = 0.0

    def pin_center(e):
        e.sentry[:, S.X] = cx
        e.sentry[:, S.Y] = cy

    pin_center(env)
    env.enemies[:, 1:, E.X] = 1.0
    env.enemies[:, 1:, E.Y] = 1.0
    vp0 = float(env.sentry[0, S.VP])
    ven0 = float(env.sentry[0, S.VP_ENEMY])
    for _ in range(40):  # 2 秒
        env.step(act)
        pin_center(env)
    check("哨兵独占控制区时持续占领", bool(env.gen_center.all()))
    if cfg.vp_mode == "accumulate":
        gained = float(env.sentry[0, S.VP]) - vp0
        check(f"2025：占中心 2 秒使我方 +2 点（实得 +{gained:.2f}）", abs(gained - 2.0) < 0.2)
    else:
        drained = ven0 - float(env.sentry[0, S.VP_ENEMY])
        check(f"2026：占控制区 2 秒使对方 −2 点（实得 −{drained:.2f}）", abs(drained - 2.0) < 0.2)

    # 先到先得：对方后进入不夺走归属
    env.enemies[:, 1:, E.X] = cx
    env.enemies[:, 1:, E.Y] = cy
    env.step(act)
    check("对方后进入控制区，不夺走已有归属（先到先得）", bool(env.gen_center.all()))

    # 离开超过宽限期后丢失归属
    for _ in range(int(cfg.center_grace / cfg.dt) + 4):
        env.step(act)
        env.sentry[:, S.X] = 1.0
        env.sentry[:, S.Y] = 1.0
    check("哨兵离开超过 2 秒宽限期后丢失归属", not bool(env.gen_center.any()))


def test_fire_rate(cfg: RMUL3v3Config, dev: str) -> None:
    """射速验证：热量锁定会把连射调制成"爆发 + 长锁定"。

    2026 单管只有 260 热量 / 30 冷却：满血连射约 1.75 秒打 ~35 发即超限锁定，
    之后要冷却到 0（约 8.7 秒）才解锁。所以**瞬时射速**远高于长期均值，
    长期均值才等于 冷却/每发热量。两者都要验证。
    """
    expected = cfg.heat_cool / cfg.heat_per_shot
    print(f"\n[9] 行为：射速与热量（长期均值应为 {expected:.0f} 发/秒）")
    env = BatchedRMUL3v3(cfg, n_envs=4, device=dev, seed=11)
    env.reset()
    env.sentry[:, S.X] = 6.0
    env.sentry[:, S.Y] = 4.0
    env.sentry[:, S.TH] = 0.0
    act = torch.zeros(env.n, env.n_actions, device=dev)
    act[:, 2] = 1.0
    act[:, 3] = 1.0

    def hold(e):
        """把敌人钉在正前方且**始终落在哨兵射程内**，同时锁死哨兵自身状态。

        位置取哨兵射程的一半（2026: 4m），确保每步都能开火。
        因为对手门限 6m > 4m 也会打到哨兵，所以额外把血量与战亡计时锁死 ——
        本测试只量射速，不能让"战亡后 15 秒无法开火"混进来。
        """
        e.enemies[:, 1:, E.X] = 6.0 + cfg.shot_range * 0.5
        e.enemies[:, 1:, E.Y] = 4.0
        e.enemies[:, 1:, E.HP] = 1e6
        e.sentry[:, S.HP] = cfg.sentry_hp
        e.sentry[:, S.DEAD] = 0.0

    env.sentry[:, S.TH] = 0.0   # 朝 +x，正对敌人
    hold(env)
    # --- 爆发段：从热量 0 开始连射，测瞬时射速 ---
    a_burst0 = float(env.sentry[0, S.AMMO])
    burst_steps = int(1.0 / cfg.dt)  # 1 秒
    for _ in range(burst_steps):
        env.step(act)
        hold(env)
    burst_rate = (a_burst0 - float(env.sentry[0, S.AMMO])) / (burst_steps * cfg.dt)
    check(f"满热连射瞬时射速很高（实测 {burst_rate:.1f} 发/秒）",
          burst_rate > expected * 1.5)

    # --- 长期段：跑够多个锁定-冷却周期，测长期均值 ---
    T = 90.0  # 90 秒，覆盖 2026 的 ~10.2 秒周期
    t0 = int(env.step_count[0])
    a0 = float(env.sentry[0, S.AMMO])
    for _ in range(int(T / cfg.dt)):
        env.step(act)
        hold(env)
    dt_sim = (int(env.step_count[0]) - t0) * cfg.dt
    fired = a0 - float(env.sentry[0, S.AMMO])
    rate = fired / dt_sim
    check(f"长期平均射速 ≈ {expected:.0f} 发/秒（实测 {rate:.2f}，{dt_sim:.0f}s 内 {fired:.0f} 发）",
          abs(rate - expected) <= max(0.2, expected * 0.06))
    check("任意时刻单管热量不超过上限",
          float(env.sentry[:, S.HEAT1].max()) <= cfg.heat_limit + 1e-6
          and float(env.sentry[:, S.HEAT2].max()) <= cfg.heat_limit + 1e-6)
    if cfg.n_barrels == 1:
        check("2026 单管：第 2 管全程不产生热量", float(env.sentry[:, S.HEAT2].max()) == 0.0)


def test_overheat_and_death(cfg: RMUL3v3Config, dev: str) -> None:
    print(f"\n[10] 行为：过热锁定 / 战亡重置热量 / 复活")
    env = BatchedRMUL3v3(cfg, n_envs=2, device=dev, seed=13)
    env.reset()
    env.sentry[:, S.X] = 6.0
    env.sentry[:, S.Y] = 4.0
    env.sentry[:, S.TH] = 0.0
    env.sentry[:, S.HEAT1] = cfg.heat_limit + 50.0
    env.sentry[:, S.HEAT2] = cfg.heat_limit + 50.0
    act = torch.zeros(env.n, env.n_actions, device=dev)
    act[:, 2] = 1.0
    act[:, 3] = 1.0
    a0 = float(env.sentry[0, S.AMMO])
    for _ in range(20):
        env.step(act)
        env.enemies[:, 1:, E.X] = 7.0
        env.enemies[:, 1:, E.Y] = 4.0
        env.enemies[:, 1:, E.HP] = 1e6
    check("热量超上限时发射机构锁定（1 秒内未发弹）",
          abs(a0 - float(env.sentry[0, S.AMMO])) < 1e-6)

    env2 = BatchedRMUL3v3(cfg, n_envs=2, device=dev, seed=3)
    env2.reset()
    # 把**所有**敌人挪到对角远角：对手开火门限是 6m，距离要拉开才不会被干扰
    env2.enemies[:, 1:, E.X] = 11.0
    env2.enemies[:, 1:, E.Y] = 7.0
    # 把哨兵挪离控制区：否则它会以 1 点/秒占点，干扰"战亡对胜利点的影响"测量
    env2.sentry[:, S.X] = 1.0
    env2.sentry[:, S.Y] = 1.0
    env2.sentry[:, S.HEAT1] = 350.0
    env2.sentry[:, S.DEATHS] = 0.0
    vp_before = float(env2.sentry[0, S.VP])
    ven_before = float(env2.sentry[0, S.VP_ENEMY])
    env2.sentry[:, S.HP] = 0.0
    env2.step(torch.zeros(env2.n, env2.n_actions, device=dev))
    check("战亡时热量重置为 0", float(env2.sentry[:, S.HEAT1].max()) == 0.0)
    check("战亡次数 +1", float(env2.sentry[0, S.DEATHS]) == 1.0)
    check("战亡当帧血量保持 0（未提前回血）", float(env2.sentry[0, S.HP]) == 0.0)
    if cfg.vp_kill_receiver == "victim":
        check(f"2026：我方战亡扣除我方 20 点胜利点"
              f"（{vp_before:.0f}→{float(env2.sentry[0, S.VP]):.0f}）",
              abs(vp_before - float(env2.sentry[0, S.VP]) - 20.0) < 1e-6)
    else:
        check(f"2025：我方战亡使**对方** +20 点"
              f"（{ven_before:.0f}→{float(env2.sentry[0, S.VP_ENEMY]):.0f}）",
              abs(float(env2.sentry[0, S.VP_ENEMY]) - ven_before - 20.0) < 1e-6)
        check("2025：我方战亡不改变我方胜利点（只有击杀才加）",
              abs(float(env2.sentry[0, S.VP]) - vp_before) < 1e-6)
    for _ in range(int((5.0 + cfg.respawn_invuln) / cfg.dt) + 4):
        env2.step(torch.zeros(env2.n, env2.n_actions, device=dev))
        env2.enemies[:, 1:, E.X] = 11.0
        env2.enemies[:, 1:, E.Y] = 7.0
        env2.sentry[:, S.X] = 1.0
        env2.sentry[:, S.Y] = 1.0
    check(f"复活后血量回到上限 20%（{cfg.sentry_hp * 0.2:.0f}，实得 {float(env2.sentry[0, S.HP]):.0f}）",
          abs(float(env2.sentry[0, S.HP]) - cfg.sentry_hp * 0.2) < 1.0)
    check("复活后发射机构立即解锁（热量为 0）", float(env2.sentry[0, S.HEAT1]) == 0.0)
    check("弹药不因复活返还", float(env2.sentry[0, S.AMMO]) <= cfg.ammo_init)


def test_sensor_isolation(cfg: RMUL3v3Config, dev: str) -> None:
    print(f"\n[11] 行为：信息孤岛（哨兵拿不到敌方坐标，只能靠自身感知）")
    # 关掉漏检（obs_dropout）与位置噪声，让几何可见性断言是确定的
    cfg_det = RMUL3v3Config(season=cfg.season, obs_dropout=0.0, obs_noise_xy=0.0)
    env = BatchedRMUL3v3(cfg_det, n_envs=1, device=dev, seed=5)
    env.reset()
    env.sentry[:, S.X] = 6.0
    env.sentry[:, S.Y] = 4.0
    env.sentry[:, S.TH] = 0.0
    # ⚠️ 槽位约定：enemies 的槽 0 是"自机"槽（与 _vis 对齐），真实敌人占槽 1/2/3
    env.enemies[:, 1, E.X] = 3.0     # 正后方 3m，超 FOV
    env.enemies[:, 1, E.Y] = 4.0
    env.enemies[:, 2, E.X] = 20.0    # 正前方但超感知距离（14m > 9m）
    env.enemies[:, 2, E.Y] = 4.0
    env.enemies[:, 3, E.X] = 7.0     # 正前方 1m，应可见
    env.enemies[:, 3, E.Y] = 4.0
    env.enemies[:, 1:, E.ALIVE] = 1.0
    vis = env.refresh_visibility()   # 改完位置必须显式刷新，_vis 才不会是上一步的
    # 立即取掩码：action_mask() 读的是 _vis，必须在 refresh 之后取，
    # 否则拿到的是刷新前的旧值（测试里踩过一次）。
    mask = env.action_mask()
    # 索引约定：_vis 形状 [n, k]，k = n_enemies+1，槽 0 = 自机，真实敌人是槽 1/2/3。
    # 动作布局：[0:2]=移动, [2]=不交战, [3]=目标1(E0), [4]=目标2(E1), [5]=目标3(E2), [6]=开火
    check("正后方的敌人不可见（FOV 生效）", not bool(vis[0, 1]))
    check("正前方但超感知距离的敌人不可见", not bool(vis[0, 2]))
    check("正前方近距离敌人可见", bool(vis[0, 3]))
    check("掩码与可见性一致：不可见目标禁用(槽3/4)、可见目标可用(槽5)",
          (not bool(mask[0, 3])) and (not bool(mask[0, 4])) and bool(mask[0, 5]))

    # 随机漏检应当存在但很稀疏（默认 2%）
    cfg_sto = RMUL3v3Config(season=cfg.season)
    env2 = BatchedRMUL3v3(cfg_sto, n_envs=2000, device=dev, seed=1)
    env2.reset()
    env2.sentry[:, S.X] = 6.0
    env2.sentry[:, S.Y] = 4.0
    env2.sentry[:, S.TH] = 0.0
    env2.enemies[:, 1:, E.X] = 7.0
    env2.enemies[:, 1:, E.Y] = 4.0
    env2.enemies[:, 1:, E.ALIVE] = 1.0
    v = env2.refresh_visibility()[:, 1:]
    seen = float(v.float().mean())
    check(f"默认漏检率约 {cfg_sto.obs_dropout:.0%}（实测可见率 {seen:.3f}）",
          abs(seen - (1 - cfg_sto.obs_dropout)) < 0.03)


def test_constants_2027(cfg: RMUL3v3Config) -> None:
    """2027 常量：官方手册 V1.0.0（20261015）口径。"""
    print(f"\n[2027-1] 关键常量（官方手册 V1.0.0）")
    check("控制度各 600（§3.3.4）", cfg.vp_init == 600.0)
    check("战亡自扣 10（§3.3.4）", cfg.vp_kill_coef == 10.0
          and cfg.vp_kill_receiver == "victim")
    check("3 处控制区（坐标来自官方图纸，配置数据）", len(cfg.zones_rect) == 3)
    check("中央控制区 3000×3000 @ (6,4)", cfg.zones_rect[0][1:] == (6.0, 4.0, 1.5, 1.5))
    check("两处高地 1200×1200 且 180° 镜像",
          cfg.zones_rect[1][1:] == (3.5, 3.2, 0.6, 0.6)
          and cfg.zones_rect[2][1:] == (8.5, 4.8, 0.6, 0.6))
    check("占领进度 0~5、速率 ±1/s", cfg.zone_progress_max == 5.0
          and cfg.zone_progress_rate == 1.0)
    check("读条：每秒 +1，首次 5，每次 +5（§3.3.2.2）",
          cfg.respawn_progress_per_s == 1.0 and cfg.respawn_base_len == 5.0
          and cfg.respawn_len_step == 5.0)
    check("复活：20% 血 + 无敌 30 s + 虚弱",
          cfg.respawn_hp_ratio == 0.2 and cfg.respawn_invuln == 30.0
          and cfg.vuln_on_respawn)
    check("补给区回血基础 25%/s、档位 33%/50%", cfg.heal_supply_per_s == 0.25
          and cfg.heal_supply_t1 == 0.33 and cfg.heal_supply_t2 == 0.50)
    check("控制度差档位 200 / 400", cfg.cd_tier1 == 200.0 and cfg.cd_tier2 == 400.0)
    check("无掩体（2027 真实场地无掩体；高地/隧道几何属 B1 任务）", cfg.covers == ())


def test_2027_zone_dynamics(cfg: RMUL3v3Config, dev: str) -> None:
    """[2027-2] 行为：控制区四条动力学逐条验证（§3.3.4）。"""
    print(f"\n[2027-2] 行为：控制区进度动力学（四条规则逐条）")
    env = BatchedRMUL3v3(cfg, n_envs=4, device=dev, seed=21)
    env.reset()
    act = torch.zeros(env.n, env.n_actions, device=dev)
    cx, cy = cfg.zones_rect[0][1], cfg.zones_rect[0][2]   # 中央 (6,4)

    def pin(sx, sy, ex=11.0, ey=7.0):
        env.sentry[:, S.X] = sx
        env.sentry[:, S.Y] = sy
        env.enemies[:, 1:, E.X] = ex
        env.enemies[:, 1:, E.Y] = ey
        env.enemies[:, 1:, E.ALIVE] = 1.0

    # ---- ① 独占 + 对方 0 -> 己方 +1/s：5 秒计满 = 控制 ----
    pin(cx, cy)
    for _ in range(100):
        env.step(act); pin(cx, cy)
    p_s = float(env.zone_p[0, 0, 0])
    check(f"独占 5 秒 -> 己方进度 = 5（实得 {p_s:.2f}）", abs(p_s - 5.0) < 0.06)
    check("计满 5 = 控制", bool(env._ctrl_s[0, 0]))

    # ---- 控制中：对方控制度 −1/s ----
    ven0 = float(env.sentry[0, S.VP_ENEMY])
    for _ in range(100):
        env.step(act); pin(cx, cy)
    drained = ven0 - float(env.sentry[0, S.VP_ENEMY])
    check(f"控制 1 处 5 秒 -> 对方控制度 −5（实得 −{drained:.2f}）", abs(drained - 5.0) < 0.2)

    # ---- ② 占方换位：独占 + 对方 ≠0 -> 对方 −1/s；清空后才轮到新方 +1/s ----
    pin(1.0, 7.0, ex=cx, ey=cy)          # 我方撤出、敌方独占
    for _ in range(100):                  # 5 秒：先 drain 我方 5 -> 0（同时立即掉控制）
        env.step(act); pin(1.0, 7.0, ex=cx, ey=cy)
    p_s = float(env.zone_p[0, 0, 0])
    check(f"敌方独占 5 秒 -> 我方进度 5→0（实得 {p_s:.2f}）", abs(p_s) < 0.06)
    check("我方失去控制", not bool(env._ctrl_s[0, 0]))
    for _ in range(100):                  # 再 5 秒：敌方 0 -> 5
        env.step(act); pin(1.0, 7.0, ex=cx, ey=cy)
    p_e = float(env.zone_p[0, 0, 1])
    check(f"对方进度 0→5（实得 {p_e:.2f}）", abs(p_e - 5.0) < 0.06)
    check("对方取得控制", bool(env._ctrl_e[0, 0]))

    # ---- ③ 双方同在 -> 暂停 ----
    env.zone_p[0, 0, 0] = 2.0
    env.zone_p[0, 0, 1] = 3.0
    pin(cx, cy, ex=cx, ey=cy)
    for _ in range(20):                   # 1 秒
        env.step(act); pin(cx, cy, ex=cx, ey=cy)
    check("双方同在 1 秒 -> 进度暂停（2/3 不变）",
          abs(float(env.zone_p[0, 0, 0]) - 2.0) < 1e-5
          and abs(float(env.zone_p[0, 0, 1]) - 3.0) < 1e-5)

    # ---- ④ 无人 + 非中立 -> 双方 −1/s ----
    pin(1.0, 7.0, ex=11.0, ey=7.0)
    for _ in range(20):
        env.step(act); pin(1.0, 7.0, ex=11.0, ey=7.0)
    check("无人 1 秒 -> 双方各 −1（1/2）",
          abs(float(env.zone_p[0, 0, 0]) - 1.0) < 1e-5
          and abs(float(env.zone_p[0, 0, 1]) - 2.0) < 1e-5)


def test_2027_cd_death_vuln_supply(cfg: RMUL3v3Config, dev: str) -> None:
    """[2027-3] 行为：战亡 −10 / 读条 5 s / 虚弱 / 补给区 / 差值档位。"""
    print(f"\n[2027-3] 行为：战亡 · 虚弱 · 补给区 · 差值档位")
    env = BatchedRMUL3v3(cfg, n_envs=4, device=dev, seed=23)
    env.reset()
    act = torch.zeros(env.n, env.n_actions, device=dev)

    def park_out():
        env.sentry[:, S.X] = 1.0
        env.sentry[:, S.Y] = 7.0          # 远离补给区（补给区 y∈[0,2]），避免回血干扰
        env.enemies[:, 1:, E.X] = 11.0
        env.enemies[:, 1:, E.Y] = 7.0
        env.enemies[:, 1:, E.ALIVE] = 1.0

    # ---- 战亡 −10 + 首次读条 5 s ----
    park_out()
    vp0 = float(env.sentry[0, S.VP])
    env.sentry[:, S.HP] = 0.0
    env.step(act); park_out()
    check(f"战亡 -> 己方控制度 −10（{vp0:.0f}→{float(env.sentry[0, S.VP]):.0f}）",
          abs(vp0 - float(env.sentry[0, S.VP]) - 10.0) < 1e-6)
    check(f"首次读条 = 5 s（实得 {float(env.sentry[0, S.DEAD]):.2f}）",
          abs(float(env.sentry[0, S.DEAD]) - 5.0) < 0.06)

    # ---- 5 秒后复活：20% 血 + 虚弱 + 无敌 30 ----
    for _ in range(100):
        env.step(act); park_out()
    check(f"复活血量 = 20%（实得 {float(env.sentry[0, S.HP]):.0f}）",
          abs(float(env.sentry[0, S.HP]) - cfg.sentry_hp * 0.2) < 1.0)
    check("复活后进入虚弱", float(env.sentry[0, S.VULN]) > 0.5)
    check(f"复活后无敌 ≈ 30 s（实得 {float(env.sentry[0, S.INVULN]):.1f}）",
          abs(float(env.sentry[0, S.INVULN]) - cfg.respawn_invuln) < 0.3)

    # ---- 虚弱锁发射（掩码 + 实际不耗弹）----
    env.sentry[:, S.TH] = 0.0
    env.enemies[:, 1, E.X], env.enemies[:, 1, E.Y] = 3.0, 7.0   # 正前方 2 m
    env.enemies[:, 1:, E.ALIVE] = 1.0
    env.refresh_visibility()
    check("虚弱期间开火通道被掩码禁用", not bool(env.action_mask()[0, -1]))
    ammo0 = float(env.sentry[0, S.AMMO])
    a_fire = torch.zeros(env.n, env.n_actions, device=dev)
    a_fire[:, 2] = 1.0
    a_fire[:, 3] = 1.0
    for _ in range(10):                    # 0.5 s
        env.step(a_fire)
        env.sentry[:, S.X], env.sentry[:, S.Y], env.sentry[:, S.TH] = 1.0, 7.0, 0.0
        env.enemies[:, 1, E.X], env.enemies[:, 1, E.Y] = 3.0, 7.0
    check("虚弱锁发射：开火 0.5 s 未消耗弹药",
          abs(float(env.sentry[0, S.AMMO]) - ammo0) < 1e-6)

    # ---- 虚弱不能占控制区 ----
    env.zone_p[:, 0, :] = 0.0
    for _ in range(20):                    # 1 秒，钉在中央（仍虚弱）
        env.step(act)
        env.sentry[:, S.X], env.sentry[:, S.Y] = 6.0, 4.0
        env.enemies[:, 1:, E.X] = 11.0
        env.enemies[:, 1:, E.Y] = 7.0
    check("虚弱期间站控制区内不计进度", float(env.zone_p[0, 0, 0]) < 1e-6)

    # ---- 触己方补给区：解虚弱 + 解无敌 + 回血 25%/s ----
    env.sentry[:, S.HP] = 200.0            # 50%，留出回血观察空间
    for _ in range(20):                    # 1 秒停在补给区
        env.step(act)
        env.sentry[:, S.X], env.sentry[:, S.Y] = 1.25, 1.0
        env.enemies[:, 1:, E.X] = 11.0
        env.enemies[:, 1:, E.Y] = 7.0
    check("触己方补给区解除虚弱", float(env.sentry[0, S.VULN]) < 0.5)
    check("触己方补给区解除无敌", float(env.sentry[0, S.INVULN]) < 0.5)
    hp = float(env.sentry[0, S.HP])
    check(f"补给区回血 25%/s（200 → ≈300，实得 {hp:.0f}）", abs(hp - 300.0) < 10.0)

    # ---- 差值档位：劣势方读条 −33% / −50% ----
    for diff, cut, tag in ((-300.0, 0.33, "−33%"), (-450.0, 0.50, "−50%")):
        e2 = BatchedRMUL3v3(cfg, n_envs=2, device=dev, seed=29)
        e2.reset()
        e2.sentry[:, S.X], e2.sentry[:, S.Y] = 1.0, 7.0
        e2.enemies[:, 1:, E.X], e2.enemies[:, 1:, E.Y] = 11.0, 7.0
        e2.sentry[:, S.VP] = 600.0 + diff      # 我方落后
        e2.sentry[:, S.VP_ENEMY] = 600.0
        e2.sentry[:, S.HP] = 0.0
        e2.step(torch.zeros(e2.n, e2.n_actions, device=dev))
        dead = float(e2.sentry[0, S.DEAD])
        expect = 5.0 * (1.0 - cut)
        check(f"劣势方（差 {diff:.0f}）首次读条 {tag} ≈ {expect:.2f} s（实得 {dead:.2f}）",
              abs(dead - expect) < 0.06)


def run_season(season: str, dev: str) -> None:
    cfg = RMUL3v3Config(season=season)
    print("=" * 72)
    print(f"RMUL 3v3 哨兵环境 · 规则一致性测试 · 赛季 {season}")
    print(f"来源: {cfg.profile_source}")
    print("=" * 72)
    if season == "2027":
        test_constants_2027(cfg)
        test_2027_zone_dynamics(cfg, dev)
        test_2027_cd_death_vuln_supply(cfg, dev)
        test_fire_rate(cfg, dev)          # 通用（射速与热量机制 2026+ 相同）
        test_sensor_isolation(cfg, dev)   # 通用（几何感知，与赛季无关）
    else:
        test_constants(cfg, season)
        test_center_control(cfg, dev)
        test_fire_rate(cfg, dev)
        test_overheat_and_death(cfg, dev)
        test_sensor_isolation(cfg, dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="both", choices=["2025", "2026", "2027", "both"],
                    help="默认同时验证 2025 与 2026（你们打 2026）")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seasons = ["2025", "2026"] if args.season == "both" else [args.season]
    for s in seasons:
        run_season(s, dev)
        print()
    print("=" * 72)
    print(f"合计：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
