"""RMUL 3v3 哨兵战术强化学习环境（向量化）。

规则来源：RoboMaster 2025 机甲大师高校联盟赛比赛规则手册 V1.1.1 (20250314)
          RoboMaster 2025 机甲大师高校系列赛机器人制作规范手册 V2.0.0 (20250620)

设计原则（三条，都是被规则逼出来的）：
  1. 1 Hz 战术决策层 —— 输出"怎么动 / 打谁 / 是否开火"，云台与扳机交给控制层。
     离线数据也是 1 Hz 战术决策，两者必须同频，否则策略永远学不出来。
  2. 信息孤岛 —— RMUL 3v3 哨兵"多机通信：不允许"，且 3v3 没有雷达兵种。
     所以观测里**绝不能有上帝视角的敌方坐标**，只能有带 FOV + 遮挡 + 噪声的感知。
  3. 奖励围绕"胜利点" —— 占中心 1 点/秒、击杀 20 点/杀、满 200 获胜。
     血量只是第 4 顺位判据，所以"活得久 + 持续占点"优于"换命击杀"。

本文件同时提供：
  - RMUL3v3Config : 全部规则常量集中在此，赛季换版只改这里
  - RMUL3v3Env    : 标准 Gymnasium 环境（单环境接口，内部按批向量化）
  - BatchedRMUL3v3: 纯张量批量环境（4000+ 并行，训练用这个）

用法见文件末尾 __main__ 与 smoke_test.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

import numpy as np
import torch

try:  # gymnasium 只用于标准接口，缺失时降级为纯 torch 使用
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _HAS_GYM = False


# --------------------------------------------------------------------------
# 状态张量列索引（列布局，行 = 并行的环境）
# --------------------------------------------------------------------------
class S:  # sentry
    X, Y, TH = 0, 1, 2
    HP = 3
    HEAT1, HEAT2 = 4, 5
    AMMO, GOLD = 6, 7
    VX, VY = 8, 9
    CD = 10  # 到下一次可开火的剩余秒数
    DEAD = 11  # >0 表示读条复活剩余秒数
    FIRE_T = 12  # 已累计开火时长（用于统计实际射速）
    VP = 13  # 本方胜利点（2026 起从 200 往下扣）
    VP_ENEMY = 14  # 对方胜利点（2026 的扣分机制需要同时跟踪）
    DMG_OUT, DMG_IN = 15, 16
    HOLD = 17  # 本机连续占点秒数（用于判断控制区归属）
    DEATHS = 18  # 累计战亡次数（决定复活读条长度）
    # --- 2027 追加（只追加不改旧索引 —— 保证旧赛季行为逐位不变）---
    VULN = 19  # 虚弱：1 = 锁发射 + 不能占控制区，直到接触己方补给区解除（2027）
    INVULN = 20  # 无敌剩余秒数（2027 与 DEAD 分离：DEAD=读条，INVULN=无敌；旧赛季恒 0）
    N = 21


class E:  # enemy slots
    X, Y, HP, ALIVE, TH, ENGAGE, CD = 0, 1, 2, 3, 4, 5, 6
    N = 7


# --------------------------------------------------------------------------
# 规则常量
# --------------------------------------------------------------------------
@dataclass
class RMUL3v3Config:
    """RMUL 3v3 全部规则常量。2025 V1.1.1 实测值，赛季换版只改这里。"""

    # --- 场地与时间（V1.1.1 §3.2.1 / §3） ---
    field_w: float = 12.0  # 战场长 12 m
    field_h: float = 8.0  # 战场宽 8 m
    match_time: float = 300.0  # 比赛阶段 5 分钟

    # --- 判定频率 ---
    # dt=0.05s 是刻意选的：手册的持续射速上限 = 冷却/每发热量 = 80/10 = 8 发/秒，
    # 两根管交替 ⇒ 每管每 0.125s 一发。若 dt=0.1 会与热量锁定阈值混叠，实测射速
    # 只有 6.0 发/秒（少 25%）。dt=0.05 时每管每 2 步恰好 1 发，4 步升温 40 恰好
    # 抵 4 步冷却 40，可长期稳定在 8 发/秒。见 rules_test.py [10]。
    dt: float = 0.05
    decision_hz: float = 10.0  # 战术决策频率（上层策略每 0.1s 决策一次）

    # --- 哨兵数值（V1.1.1 表 3-4） ---
    sentry_hp: int = 400  # 初始/上限血量，固定 10 级不成长
    heat_limit: float = 400.0  # 单个发射机构射击热量上限
    heat_cool: float = 80.0  # 射击热量冷却 80/秒
    heat_per_shot: float = 10.0  # 每发 17mm 弹丸 +10 热量
    ammo_init: int = 750  # 初始允许发弹量，局内不可增加
    chassis_power: float = 100.0  # 底盘功率上限 100 W
    muzzle_speed: float = 25.0  # 射击初速度上限
    supercap_wh: float = 300.0  # 最大供电总容量 300 Wh
    supercap_j: float = 2000.0  # 超级电容标称能量上限 2000 J
    mass_kg: float = 25.0  # 最大重量 25 kg

    # --- 伤害（17mm 弹丸对机器人装甲） ---
    # 表 3-7：2025 为 10，2026 为 20 —— 由赛季档案套用，此处默认值仅在未选赛季时使用
    dmg_base: float = 10.0
    dmg_falloff: float = 0.35  # 远距离命中率/伤害衰减（建模用，非官方）
    shot_range: float = 8.0  # 有效射程（建模用，非官方）

    # --- 胜利点（由赛季档案套用，勿直接改；见 SEASON_PROFILES） ---
    vp_win: float = 200.0  # 2025: 累加到 200 获胜；2026: 归 0 获胜（即阈值 0）
    center_grace: float = 2.0  # 未检测到场地交互模块卡后，占领状态 2 秒后失效

    # --- 经济（§3.3.3 表 3-10） ---
    initial_gold: int = 200  # 比赛开始（04:59）双方各 200 金币
    gold_by_time: tuple = ((60.0, 200), (120.0, 200), (180.0, 300), (240.0, 300))  # 累计剩余时间的发放

    # --- 复活 / 回血（§3.3.2） ---
    respawn_progress_per_s: float = 2.0  # 每秒自动增加 2 点读条进度
    respawn_base_len: float = 10.0  # 首次战亡所需读条长度
    respawn_len_step: float = 10.0  # 每次战亡后读条长度依次增加 10
    respawn_hp_ratio: float = 0.2  # 复活后血量恢复至上限血量的 20%
    respawn_invuln: float = 10.0  # 复活后无敌 10 秒
    heal_supply_per_s: float = 0.0  # 占领己方补给区回血 25%/s（2027 档案启用；旧赛季 0 = 不建模）

    # --- 2027 机制（控制区双进度 / 控制度 / 虚弱 / 补给 / 差值档位） ---
    # 坐标全部为**配置数据**：来源 docs/2027场地规格.md（官方图纸实测提取）。
    # 坐标系：沿用场地规格的 mm→m 换算（12000×8000 → 12×8 m）。场地 180° 中心对称，
    #   两处高地控制区在该坐标下互为镜像，仿真中红/蓝标签无战术差别。
    zone_progress_max: float = 5.0   # [规则] 占领进度 0~5，计满 = 控制（§3.3.4）
    zone_progress_rate: float = 1.0  # [规则] 进度变化速率 ±1/s
    zones_rect: tuple = ()           # [配置] ((名称, cx_m, cy_m, 半宽_m, 半高_m), ...)；空 = 无进度控制区
    own_supply_rect: tuple = ()      # [配置] 己方补给区 (cx, cy, hw, hh)；空 = 不建模补给
    vuln_on_respawn: bool = False    # [规则] 2027：复活后虚弱（锁发射 + 不能占区），触己方补给区解除
    cd_tier1: float = 200.0          # [规则] 控制度差档位 1（劣势方读条 −33% / 回血 33%/s）
    cd_tier2: float = 400.0          # [规则] 控制度差档位 2（劣势方读条 −50% / 回血 50%/s）
    tier_respawn_cut: tuple = (0.33, 0.50)  # [规则] 读条时长削减比例（档位 1 / 档位 2）
    heal_supply_t1: float = 0.33     # [规则] 劣势方补给回血（档位 1）
    heal_supply_t2: float = 0.50     # [规则] 劣势方补给回血（档位 2）

    # --- 感知模型（规则逼出来的：哨兵拿不到敌方坐标） ---
    fov_deg: float = 120.0
    sense_range: float = 9.0  # 感知距离上限
    obs_delay: int = 2  # 观测延迟步数（10 Hz 下 2 步 = 200 ms）
    obs_noise_xy: float = 0.15  # 位置观测噪声（米）
    obs_dropout: float = 0.02  # 单帧漏检概率

    # --- 建模假设（非官方，必须自行标定） ---
    v_max: float = 3.0  # 哨兵最大水平速度 m/s（需按实车实测）
    a_max: float = 4.0  # 最大加速度 m/s^2（简化模型，当前仅用于能耗统计）
    respawn_pos_jitter: float = 0.5

    # --- 覆盖物（12x8 场地中心对称，建模用） ---
    covers: tuple = (
        (2.0, 1.0, 3.0, 1.4),  # (x, y, w, h)
        (7.0, 1.0, 3.0, 1.4),
        (2.0, 5.6, 3.0, 1.4),
        (7.0, 5.6, 3.0, 1.4),
    )
    center_r: float = 0.8  # 中心增益点半径

    # 权重按"该事件值多少胜利点"来标定，而不是拍脑袋：
    #   占控制区：1 秒 = 扣对方 1 VP（+ 相对优势 1）= 净 2 VP → 2/(1/0.05) = 0.1/步
    #   造成伤害：20 点伤害 ≈ 1 VP（400 血的敌人需 20 发×20 伤害）→ 0.05 VP/伤害
    #   ⇒ **占点项的权重必须显著高于伤害项**，否则策略会学成"脱战刷伤害"
    #     （实测教训：w_dmg_out=0.05 时，伤害每步 +1.0 而占点只 +0.025，差 40 倍，
    #       策略胜率从 0.56 崩到 0.00，净胜点 −179.5）
    w_vp_adv: float = 1.0        # 胜利点净优势的逐步变化（直接以 VP 计，这是"赢"的定义）
    # 占点权重定标（按规则经济算，不是拍脑袋）：
    #   占点 2 VP/秒；打死一个 400 血敌人需 ~6.7 秒（3 发/秒 × 20 伤害），
    #   这 6.7 秒里占点已累计 13.3 VP —— 所以**占点的价值显著高于击杀**。
    #   取 w_hold=2.0 → 占点 2.0/秒 vs 命中 0.3/发 = 6.7 倍，与上述经济关系一致。
    w_hold: float = 2.0
    # 接近控制区的距离势函数塑形（**必需**，不是可选）：
    #   实测教训：只奖励"在圈内"而不奖励"靠近圈"，策略根本发现不了占点行为
    #   —— 控制区半径仅 0.8 m，靠随机探索撞进去的概率极低。结果策略学成
    #   "逃离控制区保命"（探测显示它跑到离中心 5.9~7.2 m 处）。
    #   这是典型的稀疏奖励问题。ROBIO 2019 的结论"战术目标塑形可替代端到端 RL"
    #   正适用此处。
    #   做法：奖励"本步比上一步更接近控制区中心"的位移量。势函数只在移动时
    #   产生信号，原地不动不会累积（避免"绕圈刷分"）。
    w_approach: float = 1.0
    # ⚠️ 势函数塑形必须用**训练时的折扣因子** gamma（Ng et al. 1999 的定理形式就是
    #    F = γΦ(s') − Φ(s)）。环境里若不定义它，训练时会 AttributeError。
    #    必须与 TrainCfg.gamma 保持一致，否则塑形不再是 policy-invariant。
    gamma: float = 0.99
    # 势函数 Φ(s) = −tanh(d/σ) 的尺度 σ（米）。
    # 取 3.0m：控制区半径只有 0.8m，用远大于它的尺度才能在远处仍有可分辨梯度
    # （tanh 在 d≫σ 时饱和、梯度消失）。改这个值等于改"接近塑形的作用范围"。
    approach_sigma: float = 3.0
    # ⚠️ 伤害权重必须**远低于**占点（第三次配错，这次按整局算清楚了）：
    #   整局账本：输出 643 伤害 × 0.015 = +9.6，而站住控制区 300 秒 × 2.0 × 0.01 = +6.0
    #   → 打人比占点赚 60%，策略于是把伤害刷到 643、占点率却只有 0.002~0.011，
    #     胜率虽然能到 1.00，但那是"靠对枪赢"，而 2026 的胜利点**只由占控制区和战亡改变，
    #     伤害本身不给分**。改成 0.003 后：643 伤害 = +1.9，占点 +6.0（占点主导）。
    w_dmg_out: float = 0.003
    # 承受伤害惩罚：**不能太重**，否则"保命逃跑"会压过"占点"。
    # 实测：w_dmg_in=-0.05 时，挨打 6 秒约 -100，远大于占点收益 → 策略逃跑。
    w_dmg_in: float = -0.01
    w_overheat: float = 0.5      # 过热惩罚（2026 是"点射-长冷却"节奏）
    w_death: float = 0.0         # 已由 w_vp_adv 表达（阵亡 −20 VP），避免重复
    w_kill: float = 0.0          # 同上，已包含在 VP 变化里
    w_win: float = 10.0          # 终局胜负（lose 取负）

    # --- 对手建模（脚本 AI，非官方） ---
    enemy_fire_interval: float = 0.5  # 对手开火间隔（秒）。最初设为 0 导致 100 dps，
                                      # 比哨兵自身输出还高，会让策略学到错误示范
    # ⚠️ 对手单发伤害**必须与哨兵分开配置**（踩过的坑，2026-10 诊断发现）：
    #   曾经直接用 c.dmg_base(=20)，于是每台敌人每 0.5s 打满 20 伤害 =>
    #   3 台 = 120 dps，而 400 血的哨兵 **3.3 秒就死**，自己却只有 60 dps。
    #   这个失衡让"站定占点"在物理上不可能（站定 ≥0.47s 即死）。
    #   实测表现：策略每局反复战亡 9 次（9×20=180 VP）、占点率恒为 0.005、
    #   净胜点 −178、胜率 0；且"对手火力关掉"的对照组立刻变成 +2.8 —— 变量被干净隔离。
    #   改为 5.0：单台 10 dps、3 台 30 dps，哨兵存活约 13 秒 —— 才有战术博弈空间。
    enemy_dmg_per_hit: float = 5.0
    enemy_hp: float = 200.0  # 对手基线血量（步兵/英雄档位不同，后续可随机化）

    n_enemies: int = 3

    # --- 赛季选择（由 SEASON_PROFILES 在 __post_init__ 套用） ---
    season: str = "2026"  # 你们要打的赛季；可选 "2025" / "2026" / "2027"(预发布)
    profile_source: str = ""  # 套用后填入出处，便于追溯

    # --- 以下字段由赛季档案套用，故意不再给"2025 默认值"以免误用 ---
    n_barrels: int = 2  # 发射机构数量（2025=2 双管；2026+=1 单管）
    vp_mode: str = "accumulate"  # "accumulate"(2025 累加到 200) / "drain"(2026+ 从 200 扣)
    vp_init: float = 0.0  # 开局胜利点
    vp_center_coef: float = 1.0  # 占控制区对**胜利点**的作用（2026 作用于对方）
    vp_kill_coef: float = 20.0  # 战亡对**胜利点**的作用
    vp_kill_receiver: str = "killer"  # "killer"(2025 对方得) / "victim"(2026 自己扣)
    gain_name: str = "中心增益点"  # 增益点名称（2025 中心增益点 / 2026+ 控制区）
    control_zones: int = 1  # 控制区数量（2025/2026 = 1 处；2027 计划增至 3 处）

    # ======================================================================
    # 赛季档案
    # ======================================================================
    # ⚠️ 2026 与 2025 的哨兵和胜利点机制**完全不同**，不是小改：
    #   2025：2 个发射机构 / 热量 400 / 冷却 80（持续射速 8 发/秒）/ 17mm 伤害 10
    #         胜利点从 0 **累积**到 200；占中心我方 +1/秒；战亡则**对方** +20
    #   2026：1 个发射机构 / 热量 260 / 冷却 30（持续射速 3 发/秒）/ 17mm 伤害 20
    #         双方各从 200 开始**扣**；占控制区则**对方** −1/秒；战亡则**自己** −20
    #         §3.3.4 原文："双方各有200 点胜利点……当一方机器人占领控制区时，对方
    #         每秒扣除1 点胜利点。每当一方机器人战亡，该方扣除20 点胜利点。"
    #         §3.3.5 "一方胜利点为 0 时，当局比赛立即结束"
    # 2027（官方手册 V1.0.0，20261015）：控制区 3 处、双方独立占领进度 0~5（四条动力学：
    #   独占+对方0 → 己方 +1/s；独占+对方≠0 → 对方 −1/s；双方同在 → 暂停；
    #   无人且非中立 → 双方 −1/s）、控制度各 600、战亡自扣 10、读条 5/+5、无敌 30、
    #   复活后虚弱（锁发射 + 不能占区，触己方补给区解除）、补给区回血 25%/s（劣势方档位加速）。
    #   场地坐标见 docs/2027场地规格.md（配置数据）。
    SEASON_PROFILES: ClassVar[dict] = {
        "2025": dict(
            source="RMUL 2025 规则手册 V1.1.1 (20250314) 表 3-4 / §3.3.6",
            n_barrels=2, heat_limit=400.0, heat_cool=80.0, dmg_base=10.0,
            vp_mode="accumulate", vp_init=0.0, vp_win=200.0,
            vp_center_coef=1.0, vp_kill_coef=20.0, vp_kill_receiver="killer",
            gain_name="中心增益点",
        ),
        "2026": dict(
            source="RMUL 2026 规则手册 V1.2.0 (20260109) 表 3-4 / §3.3.4 / §3.3.5",
            n_barrels=1, heat_limit=260.0, heat_cool=30.0, dmg_base=20.0,
            vp_mode="drain", vp_init=200.0, vp_win=0.0,
            vp_center_coef=1.0, vp_kill_coef=20.0, vp_kill_receiver="victim",
            gain_name="控制区",
        ),
        "2027": dict(  # ✅ 官方手册 V1.0.0（20261015）——不再使用预发布 PPT
            source="RMUL 2027 规则手册 V1.0.0 (20261015) §3.3.2.2 / §3.3.4 / 表 3-4；"
                   "场地坐标：docs/2027场地规格.md（官方图纸实测，配置数据）",
            n_barrels=1, heat_limit=260.0, heat_cool=30.0,
            dmg_base=20.0,  # [待核] 2027 表 3-7 需手册精读确认（当前沿用 2026 值）
            vp_mode="control_degree", vp_init=600.0, vp_win=0.0,
            vp_center_coef=1.0,      # 每控制 1 处 → 对方 −1/s
            vp_kill_coef=10.0,       # 战亡 → 该方 −10（§3.3.4）
            vp_kill_receiver="victim",
            gain_name="控制区×3（控制度 600）",
            control_zones=3,
            respawn_progress_per_s=1.0, respawn_base_len=5.0, respawn_len_step=5.0,
            respawn_invuln=30.0,     # 复活后无敌 30 s（2027 由 10 → 30）
            heal_supply_per_s=0.25,  # 补给区基础回血 25%/s
            vuln_on_respawn=True,    # 复活后虚弱：锁发射 + 不能占区
            zones_rect=(
                # 顺序约定：索引 0 = 中央控制区（旧观测槽 gen_center 复用）
                ("中央控制区", 6.0, 4.0, 1.5, 1.5),       # (4500,2500)-(7500,5500) mm → 3000×3000
                ("红方高地控制区", 3.5, 3.2, 0.6, 0.6),   # (2900,2600)-(4100,3800) mm → 1200×1200
                ("蓝方高地控制区", 8.5, 4.8, 0.6, 0.6),   # (7900,4200)-(9100,5400) mm（180° 镜像）
            ),
            own_supply_rect=(1.25, 1.0, 1.25, 1.0),      # 启动区兼补给区 (0,0)-(2500,2000) mm
            covers=(),  # 2027 真实场地无掩体；高地/隧道/工作台几何属 B1 场地模型任务（可运行体）
        ),
    }

    def __post_init__(self) -> None:
        """按 season 套用规则档案。显式传入的字段不会被覆盖（便于实验覆盖）。"""
        prof = self.SEASON_PROFILES.get(self.season)
        if prof is None:
            raise ValueError(f"未知赛季 {self.season}，可选 {list(self.SEASON_PROFILES)}")
        # 注意：不能用 type(self)() 取默认值 —— 那会再次触发 __post_init__ 无限递归。
        # 用 dataclasses.fields 直接读声明的默认值。
        defaults = {f.name: f.default for f in fields(self)}
        for key, val in prof.items():
            if key in ("source", "provisional"):
                continue
            if key not in defaults:
                raise KeyError(f"赛季档案 {self.season} 含未知字段 {key}")
            # 仅当该字段仍是默认值时才套用档案，避免覆盖用户显式设置
            if getattr(self, key) == defaults[key]:
                setattr(self, key, val)
        self.profile_source = prof["source"]


# --------------------------------------------------------------------------
# 纯张量批量环境 —— 训练用
# --------------------------------------------------------------------------
class BatchedRMUL3v3:
    """完全向量化的 RMUL 3v3 战术环境。

    所有并行环境的状态存在两张 GPU 张量里（sentry / enemies），
    全程无 Python 循环，4000 个环境在单张 8GB 卡上远低于显存上限。
    """

    def __init__(self, cfg: RMUL3v3Config | None = None, n_envs: int = 1024,
                 device: str | torch.device | None = None, seed: int | None = None):
        self.cfg = cfg or RMUL3v3Config()
        self.n = int(n_envs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        c = self.cfg
        self.max_steps = int(round(c.match_time / c.dt))
        self.k = c.n_enemies + 1  # 含自机的槽位数

        self.sentry = torch.zeros(self.n, S.N, device=self.device)
        self.enemies = torch.zeros(self.n, self.k, E.N, device=self.device)
        self.step_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        # 奖励用：上一步的"胜利点净优势"，用于计算 advance（见 step 的奖励段）
        self._prev_vp_adv = torch.zeros(self.n, device=self.device)
        # 奖励用：势函数塑形项的历史值（Φ = −tanh(d/σ)，见 step 里的详细注释）
        # _phi_valid=False 表示历史势函数无效，本步不计塑形（避免跨 episode 误计）
        self._prev_phi = self._phi_value()
        self._phi_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.obs_hist = torch.zeros(self.n, c.obs_delay + 1, self.obs_dim,
                                    device=self.device)
        self.gen_center = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        # 2027：控制区双进度 [n, n_zones, 2]（0=我方, 1=对方）。旧赛季 n_zones=0，不参与任何计算。
        self.zone_p = torch.zeros(self.n, len(c.zones_rect), 2, device=self.device)

        self._u = torch.zeros(self.n, 2, device=self.device)
        self._hold_sentry = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self._hold_enemy = torch.zeros(self.n, dtype=torch.bool, device=self.device)

    # ---------------- 尺寸 ----------------
    @property
    def obs_dim(self) -> int:
        base = 6 + 8 + 7 * self.cfg.n_enemies + 1
        if self.cfg.vp_mode == "control_degree":
            # 2027 追加：自机 +2（虚弱/无敌）、每控制区 6 维（双方进度/在场/控制）、记分 +2（双方控制度）
            base += 2 + 6 * len(self.cfg.zones_rect) + 2
        return base

    @property
    def n_actions(self) -> int:
        """[vx, vy] 连续 2 维 + 目标选择 (n_enemies+1) 离散 + 开火 1 离散。"""
        return 2 + (self.cfg.n_enemies + 1) + 1

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, device=self.device, generator=self.gen)

    # ---------------- 感知 ----------------
    def _los_blocked(self) -> torch.Tensor:
        """返回 [n, k] 可见性（False = 被掩体遮挡或超距或超 FOV）。

        注意：这里刻意全部使用 `torch.logical_and` 而非 `ok &= ...`。
        实测在布尔张量 + 广播（[n,k] 与 [n,1]）的组合下，原地 `&=` 会出现
        "部分列未被写入"的静默错误（例如第 3 槽本应保持 True 却被置 False）。
        显式 out-of-place 逻辑与可稳定复现，见 rules_test.py [11]。
        """
        c = self.cfg
        px = self.sentry[:, S.X].reshape(-1, 1)
        py = self.sentry[:, S.Y].reshape(-1, 1)
        th = self.sentry[:, S.TH].reshape(-1, 1)

        ex = self.enemies[:, :, E.X]
        ey = self.enemies[:, :, E.Y]
        dx = ex - px
        dy = ey - py
        dist = torch.hypot(dx, dy)

        ok = self.enemies[:, :, E.ALIVE] > 0.5
        ok = torch.logical_and(ok, dist <= c.sense_range)
        # FOV：哨兵朝向为 +x 轴，逆时针为正
        ang = torch.atan2(dy, dx)
        dth = torch.remainder(ang - th + torch.pi, 2 * torch.pi) - torch.pi
        half = torch.deg2rad(torch.tensor(c.fov_deg / 2.0, device=self.device,
                                          dtype=dist.dtype))
        ok = torch.logical_and(ok, dth.abs() <= half)

        # 掩体遮挡：线段 vs 轴对齐矩形（中心式），分离轴判定
        for (cx, cy, cw, ch) in c.covers:
            hx, hy = cw / 2.0, ch / 2.0
            # 线段两端相对矩形中心
            p0x, p0y = px - cx, py - cy
            p1x, p1y = ex - cx, ey - cy
            # 粗筛：两端都在同一侧则不可能相交
            same = (((p0x > hx) & (p1x > hx)) | ((p0x < -hx) & (p1x < -hx))
                    | ((p0y > hy) & (p1y > hy)) | ((p0y < -hy) & (p1y < -hy)))
            # 分离轴：x 轴与 y 轴投影区间是否重叠
            t = torch.where((p1x - p0x).abs() < 1e-6,
                            torch.full_like(p0x, 0.5),
                            (torch.where(p0x > 0, -hx, hx) - p0x) / (p1x - p0x))
            t = t.clamp(0.0, 1.0)
            iy = p0y + t * (p1y - p0y)
            hit_x = iy.abs() <= hy
            t2 = torch.where((p1y - p0y).abs() < 1e-6,
                             torch.full_like(p0y, 0.5),
                             (torch.where(p0y > 0, -hy, hy) - p0y) / (p1y - p0y))
            t2 = t2.clamp(0.0, 1.0)
            ix = p0x + t2 * (p1x - p0x)
            hit_y = ix.abs() <= hx
            crossed = torch.logical_and(torch.logical_or(hit_x, hit_y),
                                        torch.logical_not(same))
            ok = torch.logical_and(ok, torch.logical_not(crossed))
        return ok

    def _build_obs(self) -> torch.Tensor:
        """构造观测。敌方坐标**必须先经过 FOV + 遮挡 + 噪声 + 漏检**。"""
        c = self.cfg
        n, k = self.n, self.k
        vis = self._los_blocked()

        # 漏检
        vis = vis & (self._rand((n, k)) > c.obs_dropout)

        n1 = (n, 1)

        # ---- 自机块（14 维） ----
        sx = self.sentry[:, S.X].reshape(n1)
        sy = self.sentry[:, S.Y].reshape(n1)
        th = self.sentry[:, S.TH].reshape(n1)
        parts = [
            (sx / c.field_w) * 2 - 1,                                   # 0 x
            (sy / c.field_h) * 2 - 1,                                   # 1 y
            torch.sin(th),                                              # 2 sin
            torch.cos(th),                                              # 3 cos
            self.sentry[:, S.HP].reshape(n1) / c.sentry_hp * 2 - 1,     # 4 hp
            (self.sentry[:, S.DEAD] > 0).float().reshape(n1),           # 5 复活中
            self.sentry[:, S.HEAT1].reshape(n1) / c.heat_limit,         # 6 热量1
            self.sentry[:, S.HEAT2].reshape(n1) / c.heat_limit,         # 7 热量2
            self.sentry[:, S.AMMO].reshape(n1) / max(c.ammo_init, 1),   # 8 弹药
            self.sentry[:, S.CD].reshape(n1),                           # 9 开火冷却
            self.gen_center.float().reshape(n1),                        # 10 中心归属
            self._hold_sentry.float().reshape(n1),                      # 11 连续占点
            self.sentry[:, S.VX].reshape(n1) / c.v_max,                 # 12 vx
            self.sentry[:, S.VY].reshape(n1) / c.v_max,                 # 13 vy
        ]
        if c.vp_mode == "control_degree":
            parts += [
                (self.sentry[:, S.VULN] > 0.5).float().reshape(n1),                     # 14 虚弱
                (self.sentry[:, S.INVULN] / max(c.respawn_invuln, 1e-6)).reshape(n1),   # 15 无敌剩余
            ]
        assert len(parts) == (16 if c.vp_mode == "control_degree" else 14), \
            f"自机块维度不对，实际 {len(parts)}"

        # ---- 敌方块（每台 7 维） ----
        # 关键：哨兵拿不到敌方坐标（RMUL 3v3 多机通信不允许），
        # 因此这里必须经 FOV + 掩体遮挡 + 噪声 + 漏检，看不到就整块置零。
        ex = self.enemies[:, :, E.X]      # [n, k]
        ey = self.enemies[:, :, E.Y]
        ox = ex + (self._rand((n, k)) * 2 - 1) * c.obs_noise_xy
        oy = ey + (self._rand((n, k)) * 2 - 1) * c.obs_noise_xy
        dxn = ((ox - self.sentry[:, S.X].reshape(n1)) / c.field_w * 2)
        dyn = ((oy - self.sentry[:, S.Y].reshape(n1)) / c.field_h * 2)
        hp_norm = self.enemies[:, :, E.HP] / c.sentry_hp * 2 - 1
        alive_f = (self.enemies[:, :, E.ALIVE] > 0.5).float()
        visf = vis.float()                # [n, k]

        for j in range(c.n_enemies):
            vj = visf[:, j + 1].reshape(n1)          # 该敌人是否可见
            parts += [
                dxn[:, j + 1].reshape(n1) * vj,      # 相对 x
                dyn[:, j + 1].reshape(n1) * vj,      # 相对 y
                hp_norm[:, j + 1].reshape(n1) * vj,  # 血量
                alive_f[:, j + 1].reshape(n1) * vj,  # 存活
                vj,                                  # 是否可见
                torch.zeros(n1, device=self.device), # 预留：装甲朝向（感知层填）
                torch.zeros(n1, device=self.device), # 预留：距离
            ]

        # ---- 控制区块（2027：每区 6 维；旧赛季无）----
        if c.zones_rect:
            s_in, e_in = self._zone_presence()
            for zi in range(len(c.zones_rect)):
                p_s = self.zone_p[:, zi, 0] / c.zone_progress_max
                p_e = self.zone_p[:, zi, 1] / c.zone_progress_max
                ctrl_s = (self.zone_p[:, zi, 0] >= c.zone_progress_max - 1e-6).float()
                ctrl_e = (self.zone_p[:, zi, 1] >= c.zone_progress_max - 1e-6).float()
                parts += [
                    p_s.reshape(n1), p_e.reshape(n1),
                    s_in[:, zi].float().reshape(n1), e_in[:, zi].float().reshape(n1),
                    ctrl_s.reshape(n1), ctrl_e.reshape(n1),
                ]

        # ---- 时间 / 记分块 ----
        t_left = 1.0 - self.step_count.float() / max(self.max_steps, 1)
        parts.append(t_left.reshape(n1))
        if c.vp_mode == "control_degree":
            parts += [
                (self.sentry[:, S.VP] / max(c.vp_init, 1e-6)).reshape(n1),
                (self.sentry[:, S.VP_ENEMY] / max(c.vp_init, 1e-6)).reshape(n1),
            ]

        obs = torch.cat(parts, dim=1)
        self._vis = vis
        return obs

    def _push_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """维护延迟缓冲，返回延迟后的观测。"""
        self.obs_hist = torch.roll(self.obs_hist, shifts=1, dims=1)
        self.obs_hist[:, 0] = obs
        return self.obs_hist[:, self.cfg.obs_delay]

    # ---------------- 对手策略 ----------------
    def _opponent_policy(self):
        """脚本对手：优先抢中心，否则向哨兵推进，并按冷却开火。

        故意写得"平庸但合理"—— 先有一个可打败的对手，再逐步加强。
        注意：对手开火必须有冷却，否则每步一发（100 dps）会远高于哨兵自身输出。

        ⚠️ 槽位约定（enemies 张量形状 [n, k, E.N]，k = n_enemies + 1）：
           槽 0 专门对应"自机"，_los_blocked 的可见性输出与动作掩码都依赖这个对齐；
           因此它**不参与对手物理**。真实敌人只占槽 1..n_enemies。
           不要在外部代码里写 enemies[:, 0] 去改敌人 —— 那改的是自机槽（幽灵槽）。
        """
        c = self.cfg
        n, k = self.n, self.k
        # 槽 0 恒定置零，防止任何路径把它当成真实敌人
        self.enemies[:, 0, :] = 0.0
        alive = self.enemies[:, :, E.ALIVE] > 0.5          # [n, k]
        alive[:, 0] = False                                # 槽 0 永不参与
        ex = self.enemies[:, :, E.X]                        # [n, k]
        ey = self.enemies[:, :, E.Y]
        # 注意：这里必须是 [n, 1]，与 [n, k] 广播后才得到 [n, k]
        sx = self.sentry[:, S.X].reshape(n, 1)
        sy = self.sentry[:, S.Y].reshape(n, 1)
        cx, cy = c.field_w / 2.0, c.field_h / 2.0

        # 目标：中心点（若未被本方占据）否则哨兵
        go_center = (~self._hold_enemy).float().reshape(n, 1)
        tx = go_center * cx + (1 - go_center) * sx           # [n, k]
        ty = go_center * cy + (1 - go_center) * sy
        vx = torch.sign(tx - ex)
        vy = torch.sign(ty - ey)
        v = torch.stack([vx, vy], dim=-1)                    # [n, k, 2]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        enemy_speed = c.v_max * 0.85
        ex_new = ex + v[:, :, 0] * enemy_speed * c.dt * alive
        ey_new = ey + v[:, :, 1] * enemy_speed * c.dt * alive

        # 对手开火：距离足够近，且各自冷却完毕（[n, k] -> any -> [n]）
        d = torch.hypot(sx - ex_new, sy - ey_new)            # [n, k]
        dead = (self.sentry[:, S.DEAD] <= 0).reshape(n, 1)   # [n, 1]
        cd = self.enemies[:, :, E.CD]                        # [n, k]
        ready = cd <= 0.0
        shot = (d < c.shot_range * 0.75) & alive & dead & ready   # [n, k]
        # 冷却推进：开火的管重置，其余递减
        self.enemies[:, :, E.CD] = torch.where(
            shot, torch.full_like(cd, c.enemy_fire_interval),
            (cd - c.dt).clamp_min(0.0))
        enemy_fire = shot.any(dim=1)                          # [n]
        return ex_new, ey_new, enemy_fire, v

    def refresh_visibility(self) -> torch.Tensor:
        """按**当前**位置重算可见性，并同步更新动作掩码所依赖的 _vis。

        _los_blocked() 只做几何计算、不更新 _vis（_vis 是在 _build_obs 里写的）。
        因此外部若要"改完位置立刻查可见性/掩码"（测试、诊断、可视化），必须先调本
        方法，否则读到的是上一步的旧值 —— 这是一个静默错误，故单独提供显式接口。
        """
        obs = self._build_obs()
        return self._vis

    # ---------------- 2027 控制区辅助 ----------------
    def _phi_value(self) -> torch.Tensor:
        """势函数 Φ = −tanh(d/σ)。

        2027：d = 到**最近控制区矩形**的距离（区内为 0）—— 三区布局下"靠近任一点"
        都是有效接近信号；旧赛季：d = 到中心圆点的距离（行为与改版前逐位一致）。
        """
        c = self.cfg
        if c.zones_rect:
            d = torch.full((self.n,), float("inf"), device=self.device)
            for (_, zx, zy, hw, hh) in c.zones_rect:
                dx = (self.sentry[:, S.X] - zx).abs() - hw
                dy = (self.sentry[:, S.Y] - zy).abs() - hh
                d = torch.minimum(d, torch.hypot(dx.clamp_min(0.0), dy.clamp_min(0.0)))
        else:
            cx, cy = c.field_w / 2.0, c.field_h / 2.0
            d = torch.hypot(self.sentry[:, S.X] - cx, self.sentry[:, S.Y] - cy)
        return -torch.tanh(d / c.approach_sigma)

    def _zone_presence(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[2027] 返回 (我方在区内 [n, nz], 敌方在区内 [n, nz])。

        我方"在区" = 存活 且 非虚弱（虚弱不能占控制区，§3.3.2.2）；敌方无虚弱机制。
        单一事实来源：step 的进度结算与 _build_obs 都用本函数 —— 避免两处实现漂移
        （同类"两处口径不一致"的坑踩过多次）。
        """
        c = self.cfg
        n, nz = self.n, len(c.zones_rect)
        s_in = torch.zeros(n, nz, dtype=torch.bool, device=self.device)
        e_in = torch.zeros(n, nz, dtype=torch.bool, device=self.device)
        if nz == 0:
            return s_in, e_in
        alive_ok = (self.sentry[:, S.DEAD] <= 0) & (self.sentry[:, S.VULN] <= 0.5)
        e_alive = self.enemies[:, :, E.ALIVE] > 0.5
        e_alive[:, 0] = False          # 槽 0 是自机占位，见 _opponent_policy 的约定
        for zi, (_, zx, zy, hw, hh) in enumerate(c.zones_rect):
            s_in[:, zi] = (((self.sentry[:, S.X] - zx).abs() <= hw)
                           & ((self.sentry[:, S.Y] - zy).abs() <= hh) & alive_ok)
            e_in[:, zi] = (((self.enemies[:, :, E.X] - zx).abs() <= hw)
                           & ((self.enemies[:, :, E.Y] - zy).abs() <= hh)
                           & e_alive).any(dim=1)
        return s_in, e_in

    # ---------------- 主循环 ----------------
    def reset(self, mask: torch.Tensor | None = None):
        c = self.cfg
        n = self.n
        if mask is None:
            mask = torch.ones(n, dtype=torch.bool, device=self.device)
        m = mask
        nm = int(m.sum())
        if nm == 0:
            return self._push_obs(self._build_obs())

        # 自机出生点：
        #   2027 —— 规则要求出生在启动区（= 己方补给区），坐标来自配置数据；
        #   旧赛季 —— 出生在中心增益点附近（建模选择，保持逐位不变）。
        if c.own_supply_rect:
            scx, scy, shw, shh = c.own_supply_rect
            self.sentry[m, S.X] = scx + (self._rand((nm,)) * 2 - 1) * shw * 0.7
            self.sentry[m, S.Y] = scy + (self._rand((nm,)) * 2 - 1) * shh * 0.7
        else:
            ang = self._rand((nm,)) * 2 * torch.pi
            rad = self._rand((nm,)) * 1.5
            self.sentry[m, S.X] = c.field_w / 2.0 + torch.cos(ang) * rad
            self.sentry[m, S.Y] = c.field_h / 2.0 + torch.sin(ang) * rad
        self.sentry[m, S.TH] = torch.pi / 2
        self.sentry[m, S.HP] = c.sentry_hp
        self.sentry[m, S.HEAT1] = 0.0
        self.sentry[m, S.HEAT2] = 0.0
        self.sentry[m, S.AMMO] = c.ammo_init
        self.sentry[m, S.GOLD] = c.initial_gold  # §3.3.3：开局 200 金币
        self.sentry[m, S.VX] = 0.0
        self.sentry[m, S.VY] = 0.0
        self.sentry[m, S.CD] = 0.0
        self.sentry[m, S.DEAD] = 0.0
        self.sentry[m, S.FIRE_T] = 0.0
        self.sentry[m, S.VP] = c.vp_init          # 2025: 0 累加；2026: 200 起扣
        self.sentry[m, S.VP_ENEMY] = c.vp_init
        self.sentry[m, S.DMG_OUT] = 0.0
        self.sentry[m, S.DMG_IN] = 0.0
        self.sentry[m, S.HOLD] = 0.0
        self.sentry[m, S.DEATHS] = 0.0
        self.sentry[m, S.VULN] = 0.0
        self.sentry[m, S.INVULN] = 0.0

        # 对手：随机撒在上半场。注意槽 0 是"自机"槽（见 _opponent_policy 的说明），
        # 恒置零；真实敌人占槽 1..n_enemies。
        self.enemies[m, 0, :] = 0.0
        self.enemies[m, :, E.X] = (self._rand((nm, self.k)) * (c.field_w - 2.0) + 1.0)
        self.enemies[m, :, E.Y] = c.field_h * 0.65 + self._rand((nm, self.k)) * (c.field_h * 0.3)
        self.enemies[m, 0, :] = 0.0
        self.enemies[m, :, E.HP] = c.enemy_hp
        self.enemies[m, 0, E.HP] = 0.0
        self.enemies[m, :, E.ALIVE] = 1.0
        self.enemies[m, 0, E.ALIVE] = 0.0
        self.enemies[m, :, E.TH] = -torch.pi / 2
        self.enemies[m, :, E.ENGAGE] = 0.0
        self.enemies[m, :, E.CD] = 0.0

        self.step_count[m] = 0
        self.gen_center[m] = False
        self.zone_p[m] = 0.0
        self._u[m] = 0.0
        self._hold_sentry[m] = False
        self._hold_enemy[m] = False
        self.obs_hist[m] = 0.0
        # 奖励基准复位：新一局的 vp_adv 起点（2025=0, 2026=0，因为双方都从 vp_init 开始）
        self._prev_vp_adv[m] = 0.0
        # ⚠️⚠️ 势函数塑形的历史值也必须作废（踩过的坑）：
        #   复活时 sentry 会**瞬移**回控制区附近，而 _prev_phi 还记着死前远处的势函数。
        #   若不作废，本步 shaping = γΦ(new) − Φ(old) 会是个巨大的正数 ——
        #   **每局白送奖励**，策略会学会"死掉刷重生奖励"。
        #   （"跨 episode 状态未复位"这一类 bug 已出现三次，务必记住。）
        if hasattr(self, "_phi_valid"):
            self._phi_valid[m] = False
        if hasattr(self, "_prev_phi"):
            self._prev_phi[m] = self._phi_value()[m]
        # ⚠️ 必须复位所有跨步累积的状态，否则 auto_reset 会让新一局带着上一局的
        # 热量锁定 / 控制区归属开局 —— 会直接污染训练数据。
        if hasattr(self, "_lock1"):
            self._lock1[m] = False
            self._lock2[m] = False
        if hasattr(self, "_holder"):
            self._holder[m] = 0.0
            self._grace[m] = 0.0

        # ⚠️ 预热观测延迟缓冲（重要，曾经的静默 bug）:
        #   _push_obs 从 obs_hist 取第 obs_delay 个槽，而上面刚把 obs_hist 清零。
        #   若只 push 一次，返回的会是那个全零的历史槽 —— 于是 reset() 后的前
        #   obs_delay 步，策略看到的观测**全是 0**。不报错，但会污染起步数据、
        #   并让"同种子/异种子"这类检查失效。
        #   注意：必须只调用一次 _build_obs（它含随机漏检/噪声，多调会破坏可复现性），
        #   直接用这一次的结果填满整个延迟缓冲。
        obs_now = self._build_obs()
        self.obs_hist[m] = obs_now[m].unsqueeze(1)
        return obs_now

    def action_mask(self) -> torch.Tensor:
        """[n, n_actions] 布尔掩码，True = 合法。不可见的目标不可选。

        动作布局：[0:2]=移动, [2]=不交战, [3..2+n_enemies]=目标选择(索引 1..n_enemies),
                  [n_actions-1]=开火。
        映射关系：目标选择索引 m 对应 enemies/_vis 的槽 m，也对应敌人 m-1。
        因此动作通道 (2 + m) 的可用性 = _vis[:, m]。
        """
        n = self.n
        mask = torch.ones(n, self.n_actions, dtype=torch.bool, device=self.device)
        vis = self._vis  # [n, k]，列 0 是自机槽，列 1..n_enemies 是真实敌人
        for m in range(1, self.cfg.n_enemies + 1):
            mask[:, 2 + m] = vis[:, m]
        # 弹药耗尽 / 复活中 / 虚弱（2027）不允许开火
        can_fire = ((self.sentry[:, S.AMMO] > 0) & (self.sentry[:, S.DEAD] <= 0)
                    & (self.sentry[:, S.VULN] <= 0.5))
        mask[:, -1] = can_fire
        return mask

    def step(self, action: torch.Tensor):
        """action: [n, n_actions]。前 2 维连续（-1..1），后 n_enemies+2 维为离散 logits/索引。

        约定：动作张量全部当作"已决策"的值传入 —— 便于把 PPO 的 logits 直接喂进来。
          action[:, 0:2]        : vx, vy ∈ [-1, 1]（乘 v_max）
          action[:, 2]          : 目标选择索引 0..n_enemies（整数）
          action[:, 3]          : 是否开火（>0 视为开火）
        """
        c = self.cfg
        n = self.n
        act = action
        if act.shape[1] != self.n_actions:
            raise ValueError(f"action 维度应为 {self.n_actions}，收到 {act.shape[1]}")

        # ---- 运动：匀速 + 限速 + 场地边界 ----
        vx = act[:, 0].clamp(-1, 1) * c.v_max
        vy = act[:, 1].clamp(-1, 1) * c.v_max
        alive_s = self.sentry[:, S.DEAD] <= 0
        vx = vx * alive_s
        vy = vy * alive_s
        self.sentry[:, S.VX] = vx
        self.sentry[:, S.VY] = vy
        self.sentry[:, S.X] = (self.sentry[:, S.X] + vx * c.dt).clamp(0.0, c.field_w)
        self.sentry[:, S.Y] = (self.sentry[:, S.Y] + vy * c.dt).clamp(0.0, c.field_h)

        # 朝向跟随速度方向（云台由控制层负责，这里只用于感知 FOV）
        moving = torch.hypot(vx, vy) > 1e-3
        new_th = torch.atan2(vy, vx)
        self.sentry[:, S.TH] = torch.where(moving, new_th, self.sentry[:, S.TH])

        # ---- 对手推进 ----
        ex_new, ey_new, enemy_fire, _ = self._opponent_policy()
        self.enemies[:, :, E.X] = ex_new.clamp(0.0, c.field_w)
        self.enemies[:, :, E.Y] = ey_new.clamp(0.0, c.field_h)
        self.enemies[:, :, E.ALIVE] = (self.enemies[:, :, E.HP] > 0).float()

        # ---- 控制区归属 ----
        cx, cy = c.field_w / 2.0, c.field_h / 2.0
        if c.vp_mode != "control_degree":
            # 旧赛季（2025/2026）：单中心 + 先到先得 + 宽限期（行为逐位不变）
            s_in = (torch.hypot(self.sentry[:, S.X] - cx, self.sentry[:, S.Y] - cy)
                    <= c.center_r) & alive_s
            e_in = (torch.hypot(self.enemies[:, :, E.X] - cx,
                                self.enemies[:, :, E.Y] - cy) <= c.center_r).any(dim=1)
            e_in = e_in & (self.enemies[:, :, E.ALIVE] > 0.5).any(dim=1)

            # §3.3.6(2025) / §3.3.4(2026) 原文："控制区在同一时间仅可被同一方占领……若双方
            # 机器人同时位于控制区内，占领方遵循先到先得的原则"。因此不能简单地用
            # "双方都在 -> 无人占"。用 holder 追踪当前占领方（1=我方, -1=对方, 0=无人），
            # 配合"未检测到场地交互模块卡后，占领状态将在 2 秒后失效"做宽限期。
            if not hasattr(self, "_holder"):
                self._holder = torch.zeros(n, dtype=torch.float32, device=self.device)
                self._grace = torch.zeros(n, device=self.device)
            only_s = s_in & ~e_in
            only_e = e_in & ~s_in
            both = s_in & e_in
            # 新进入者只有在无人占领时才拿到归属（先到先得）
            free = self._holder <= 0.0
            self._holder = torch.where(only_s & free, torch.ones_like(self._holder),
                                       torch.where(only_e & (self._holder >= 0.0),
                                                   -torch.ones_like(self._holder), self._holder))
            # 宽限期倒计时：仍有本方机器人在点内则刷新
            still_s = (self._holder > 0) & (s_in | both)
            still_e = (self._holder < 0) & (e_in | both)
            self._grace = torch.where(still_s | still_e,
                                      torch.full_like(self._grace, c.center_grace),
                                      (self._grace - c.dt).clamp_min(0.0))
            expired = self._grace <= 0.0
            self._holder = torch.where(expired, torch.zeros_like(self._holder), self._holder)

            self._hold_sentry = self._holder > 0
            self._hold_enemy = self._holder < 0
            self.gen_center = self._hold_sentry
        else:
            # ============ 2027：3 控制区 + 双方独立进度 0~5（§3.3.4）============
            # 四条动力学（手册原文逐条实现；"己方/对方"按先后位置为独占方视角）：
            #   ① 独占激活 且 对方进度为 0 -> 己方 +1/s
            #   ② 独占激活 且 对方进度 ≠ 0 -> 对方 −1/s
            #   ③ 双方同激活               -> 暂停（不加不减）
            #   ④ 双方均不激活 且 非中立    -> 双方 −1/s
            s_in, e_in = self._zone_presence()
            p_s = self.zone_p[:, :, 0]
            p_e = self.zone_p[:, :, 1]
            only_s = s_in & ~e_in
            only_e = e_in & ~s_in
            neither = ~s_in & ~e_in
            step_p = c.zone_progress_rate * c.dt
            inc_s = only_s & (p_e <= 1e-6)          # ①
            drain_e = only_s & (p_e > 1e-6)         # ②
            inc_e = only_e & (p_s <= 1e-6)          # ①（对方向）
            drain_s = only_e & (p_s > 1e-6)         # ②（对方向）
            decay = neither & ((p_s > 1e-6) | (p_e > 1e-6))   # ④
            self.zone_p[:, :, 0] = (p_s + (inc_s.float() - drain_s.float()
                                           - decay.float()) * step_p
                                    ).clamp(0.0, c.zone_progress_max)
            self.zone_p[:, :, 1] = (p_e + (inc_e.float() - drain_e.float()
                                           - decay.float()) * step_p
                                    ).clamp(0.0, c.zone_progress_max)
            self._ctrl_s = self.zone_p[:, :, 0] >= c.zone_progress_max - 1e-6
            self._ctrl_e = self.zone_p[:, :, 1] >= c.zone_progress_max - 1e-6
            self._hold_sentry = self._ctrl_s.any(dim=1)
            self._hold_enemy = self._ctrl_e.any(dim=1)
            # 索引 0 = 中央控制区（zones_rect 顺序约定）：复用旧观测槽 gen_center
            self.gen_center = self._ctrl_s[:, 0]

        # ---- 胜利点（两个赛季机制完全不同，符号相反） ----
        # 2025 §3.3.6：占中心 -> 我方 +1/秒；战亡 -> 对方 +20
        # 2026 §3.3.4：占控制区 -> 对方 −1/秒；战亡 -> 该方 −20
        vp_step = c.vp_center_coef * c.dt
        if c.vp_mode == "accumulate":
            self.sentry[:, S.VP] += self._hold_sentry.float() * vp_step
        elif c.vp_mode == "control_degree":
            # 2027 §3.3.4：每控制 1 处控制区，对方 −1/s（最多 3/s）
            n_s = self._ctrl_s.sum(dim=1).float()
            n_e = self._ctrl_e.sum(dim=1).float()
            self.sentry[:, S.VP_ENEMY] = (self.sentry[:, S.VP_ENEMY]
                                          - n_s * vp_step).clamp_min(0.0)
            self.sentry[:, S.VP] = (self.sentry[:, S.VP] - n_e * vp_step).clamp_min(0.0)
        else:  # drain（2026）：占点扣的是**对方**的胜利点
            self.sentry[:, S.VP_ENEMY] = (
                self.sentry[:, S.VP_ENEMY] - self._hold_sentry.float() * vp_step).clamp_min(0.0)

        # ---- 开火判定 ----
        tgt = act[:, 2].long().clamp(0, c.n_enemies)
        want_fire = act[:, 3] > 0.0
        # 掩码：不可见的目标不能打（杜绝"隔墙打人"这种上帝视角行为）
        # 槽位约定：_vis 的列 m 与动作选择索引 m 一一对应（m=1..n_enemies 是真实敌人）
        #
        # 这里**现算**可见性，不依赖 self._vis —— 原因：
        #   _vis 是 _build_obs() 的副产物，而 reset() 在 mask 全 False 时会提前返回、
        #   跳过 _build_obs()，导致新环境上直接 step() 会 AttributeError。
        #   现算同时消除了"用了上一帧的陈旧可见性"这一隐患。
        #   （_los_blocked 只做几何计算，无副作用，可安全重复调用。）
        vis = self._los_blocked()
        tgt_visible = torch.gather(vis, 1, tgt.unsqueeze(1)).squeeze(1)
        # 目标槽 -> 敌人列（目标索引 m 对应 enemies 槽 m；m=0 表示不交战）
        ecol = tgt.clamp_min(0)
        tx = torch.gather(self.enemies[:, :, E.X], 1, ecol.unsqueeze(1)).squeeze(1)
        ty = torch.gather(self.enemies[:, :, E.Y], 1, ecol.unsqueeze(1)).squeeze(1)
        t_alive = torch.gather(self.enemies[:, :, E.ALIVE], 1,
                               ecol.unsqueeze(1)).squeeze(1) > 0.5
        dist = torch.hypot(tx - self.sentry[:, S.X], ty - self.sentry[:, S.Y])

        fire_ok = (tgt > 0) & tgt_visible & t_alive & want_fire & alive_s
        fire_ok &= (self.sentry[:, S.AMMO] > 0)
        fire_ok &= dist <= c.shot_range
        fire_ok &= (self.sentry[:, S.VULN] <= 0.5)   # 2027：虚弱锁发射机构（旧赛季恒 0）

        # ---- 热量：先冷却结算，再判定能否开火（§3.3.1.3）----
        # 手册语义：Q0=热量上限，Q1=当前热量。每发 17mm +10；冷却按 10Hz 结算，
        # 每周期冷却量 = 每秒冷却值/10。
        #   Q1 > Q0      -> 发射机构锁定，直到 Q1 = 0 才解锁（不是降到 Q0 以下）
        #   Q1 > Q0+100  -> 当局剩余时间永久锁定
        # 关键：持续射速上限 = 冷却/每发热量（2025: 80/10=8 发/秒；2026: 30/10=3 发/秒）。
        # 因此**不需要**再设固定开火间隔 —— 热量收支本身就是射速限制器。
        cool = c.heat_cool * c.dt
        self.sentry[:, S.HEAT1] = (self.sentry[:, S.HEAT1] - cool).clamp_min(0.0)
        self.sentry[:, S.HEAT2] = (self.sentry[:, S.HEAT2] - cool).clamp_min(0.0)
        # 锁定状态：超上限即锁，热量归零才解锁
        if not hasattr(self, "_lock1"):
            self._lock1 = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._lock2 = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._lock1 = (self._lock1 | (self.sentry[:, S.HEAT1] > c.heat_limit)) & \
                      (self.sentry[:, S.HEAT1] > 0.0)
        self._lock2 = (self._lock2 | (self.sentry[:, S.HEAT2] > c.heat_limit)) & \
                      (self.sentry[:, S.HEAT2] > 0.0)

        # 2025 = 2 管交替（热量独立）；2026+ = 1 管（第 2 管恒不参与）
        two_barrels = c.n_barrels >= 2
        b1_ok = (~self._lock1) & (self.sentry[:, S.HEAT1] < c.heat_limit)
        b2_ok = two_barrels & (~self._lock2) & (self.sentry[:, S.HEAT2] < c.heat_limit)
        any_ok = b1_ok | b2_ok
        use_b1 = b1_ok & ((~b2_ok) | (self.sentry[:, S.HEAT1] <= self.sentry[:, S.HEAT2]))
        use_b2 = b2_ok & ~use_b1
        fire = fire_ok & any_ok
        shot_b1 = fire & use_b1
        shot_b2 = fire & use_b2

        # 伤害
        falloff = (1.0 - c.dmg_falloff * (dist / c.shot_range).clamp(0, 1)) * fire.float()
        dmg = c.dmg_base * falloff
        prev_alive = self.enemies[:, :, E.ALIVE].clone()
        self.enemies[:, :, E.HP] = self.enemies[:, :, E.HP] - \
            torch.zeros_like(self.enemies[:, :, E.HP]).scatter_(1, ecol.unsqueeze(1),
                                                                dmg.unsqueeze(1))
        new_alive = (self.enemies[:, :, E.HP] > 0).float()
        kills = ((prev_alive > 0.5) & (new_alive < 0.5)).sum(dim=1).float()
        self.enemies[:, :, E.ALIVE] = new_alive
        # ---- 对手开火（简化：命中即扣血，**复活无敌期免伤**）----
        # 放在胜利点结算之前，保证"本步战亡"与"本步扣分"同步
        #
        # ⚠️ 无敌期判定曾**缺失**（注释写了但没实现），后果严重：
        #   respawn_invuln=10 只被加进了读条长度，从未用于免伤。
        #   于是复活的哨兵（80 血）立刻继续挨打，3 敌 30 dps -> 2.7 秒又死，
        #   实测每局战亡 7.2 次、承受伤害 882；而"关掉对手火力"的对照组是
        #   0 死亡 / 100% 胜率 —— 变量被干净隔离。
        #   DEAD 字段的语义是"读条 + 无敌期"（见 respawn_secs 的构造），
        #   所以 DEAD > 0 即处于无敌/读条状态，应免伤。
        invuln = (self.sentry[:, S.DEAD] > 0) | (self.sentry[:, S.INVULN] > 0)  # 2027：读条与无敌分离
        enemy_dmg = enemy_fire.float() * c.enemy_dmg_per_hit * (~invuln).float()
        self.sentry[:, S.HP] = (self.sentry[:, S.HP] - enemy_dmg).clamp_min(0.0)
        self.sentry[:, S.DMG_IN] += enemy_dmg
        dead_now = ((self.sentry[:, S.HP] <= 0) & (self.sentry[:, S.DEAD] <= 0)).float()

        # 战亡的胜利点作用方向（两赛季不同）：
        #   2025 §3.3.6 "每当一方机器人战亡，另一方获得 20 点胜利点"
        #        => 我方击杀敌人：我方 +20；我方战亡：对方 +20
        #   2026 §3.3.4 "每当一方机器人战亡，该方扣除 20 点胜利点"
        #        => 我方战亡：我方 −20；我方击杀敌人：对方 −20
        self_deaths = dead_now  # 本步我方是否战亡（0/1）
        if c.vp_kill_receiver == "victim":
            # 2026：战亡方自己扣（我方战亡 -> 我方扣）
            self.sentry[:, S.VP] = (self.sentry[:, S.VP]
                                    - self_deaths * c.vp_kill_coef).clamp_min(0.0)
        else:
            # 2025：对方得（我方战亡 -> 对方加）
            self.sentry[:, S.VP_ENEMY] += self_deaths * c.vp_kill_coef
        if c.vp_mode == "accumulate":
            # 2025：击杀敌人 -> 我方得
            self.sentry[:, S.VP] += kills * c.vp_kill_coef
        else:
            # 2026：击杀敌人 -> 对方扣
            self.sentry[:, S.VP_ENEMY] = (self.sentry[:, S.VP_ENEMY]
                                          - kills * c.vp_kill_coef).clamp_min(0.0)

        # 热量入账（每根管独立）
        self.sentry[:, S.HEAT1] = (self.sentry[:, S.HEAT1]
                                   + shot_b1.float() * c.heat_per_shot).clamp_min(0.0)
        self.sentry[:, S.HEAT2] = (self.sentry[:, S.HEAT2]
                                   + shot_b2.float() * c.heat_per_shot).clamp_min(0.0)
        self.sentry[:, S.AMMO] = (self.sentry[:, S.AMMO] - fire.float()).clamp_min(0.0)
        self.sentry[:, S.FIRE_T] += fire.float() * c.dt
        self.sentry[:, S.DMG_OUT] += dmg * fire.float()

        # 对手开火（简化：命中即扣血，无敌期免伤）—— 已提前到胜利点结算之前，见上

        # ---- 阵亡与复活（§3.3.2.2） ----
        # 规则：战亡时射击热量重置为 0；每秒自动 +2 点读条进度；
        #       首次战亡读条长度 10，同一台机器人每次战亡后依次 +10。
        #       即 第 n 次战亡需要 10*n 进度 -> n*5 秒。
        # 战亡 -> 开始读条；读条结束的那一帧回血复活。
        #
        # 实现要点（这里连续错了三次，务必看懂再改）：
        #   错法 1：先把 DEAD 递减并 clamp_min(0)，再判断 (DEAD > 0) -> 归零后永假 -> 永不复活
        #   错法 2：保存 dead_prev 再判断 (dead_prev > 0) -> 阵亡当步 dead_prev 是 0，逻辑被污染
        #   错法 3：用 (DEAD + dt) 反推上一步 -> DEAD 被 clamp 到 0 时反推得 dt(>0)，
        #           于是"读条已结束"被误判成"正在读条"，每局第一步就白送一次复活
        #   结论：**不要反推、不要看状态位跨越**。用一个显式标志记住"上一步是否在读条"。
        just_died = (self.sentry[:, S.HP] <= 0) & (self.sentry[:, S.DEAD] <= 0)
        was_reviving = self.sentry[:, S.DEAD] > 0          # 上一步（本步开始时）在读条中
        self.sentry[:, S.DEATHS] += just_died.float()
        need_len = (c.respawn_base_len
                    + (self.sentry[:, S.DEATHS] - 1.0).clamp_min(0.0) * c.respawn_len_step)
        # 控制度差（2027 差值档位用；旧赛季忽略）
        cd_diff = self.sentry[:, S.VP] - self.sentry[:, S.VP_ENEMY]
        if c.vuln_on_respawn:
            # 2027：DEAD = 读条剩余（**不含**无敌；无敌/虚弱独立跟踪）。
            # 差值档位：劣势方读条时长 −33% / −50%（§3.3.4 追赶机制）
            cut = torch.where(
                cd_diff <= -c.cd_tier2,
                torch.full_like(cd_diff, c.tier_respawn_cut[1]),
                torch.where(cd_diff <= -c.cd_tier1,
                            torch.full_like(cd_diff, c.tier_respawn_cut[0]),
                            torch.zeros_like(cd_diff)))
            respawn_secs = need_len / c.respawn_progress_per_s * (1.0 - cut)
        else:
            respawn_secs = need_len / c.respawn_progress_per_s + c.respawn_invuln

        self.sentry[:, S.DEAD] = torch.where(
            just_died, respawn_secs,
            (self.sentry[:, S.DEAD] - c.dt).clamp_min(0.0))
        # 复活 = 上一步在读条 且 本步读条结束 且 本步不是新的阵亡
        respawning = was_reviving & (self.sentry[:, S.DEAD] <= 0) & ~just_died

        # 战亡瞬间热量归零
        self.sentry[:, S.HEAT1] = torch.where(just_died, torch.zeros_like(self.sentry[:, S.HEAT1]),
                                              self.sentry[:, S.HEAT1])
        self.sentry[:, S.HEAT2] = torch.where(just_died, torch.zeros_like(self.sentry[:, S.HEAT2]),
                                              self.sentry[:, S.HEAT2])
        # 复活瞬间：血量回到上限的 20%，发射机构立即解锁（哨兵特有，热量已为 0）
        self.sentry[:, S.HP] = torch.where(
            respawning, torch.full_like(self.sentry[:, S.HP], c.sentry_hp * c.respawn_hp_ratio),
            self.sentry[:, S.HP])

        # ---- 2027：无敌 / 虚弱 / 补给区（§3.3.2.2） ----
        if c.vuln_on_respawn:
            # 无敌：复活起 30 s（与读条分离）；虚弱：复活起持续，**只有接触己方补给区才解除**
            self.sentry[:, S.INVULN] = torch.where(
                respawning, torch.full_like(self.sentry[:, S.INVULN], c.respawn_invuln),
                (self.sentry[:, S.INVULN] - c.dt).clamp_min(0.0))
            self.sentry[:, S.VULN] = torch.where(
                just_died, torch.zeros_like(self.sentry[:, S.VULN]),
                torch.where(respawning, torch.ones_like(self.sentry[:, S.VULN]),
                            self.sentry[:, S.VULN]))
        # 补给区：解虚弱/无敌（仅 2027 有虚弱）+ 回血（基础 25%/s；劣势方按差值档位加速）
        if c.own_supply_rect and c.heal_supply_per_s > 0.0:
            scx, scy, shw, shh = c.own_supply_rect
            in_supply = (((self.sentry[:, S.X] - scx).abs() <= shw)
                         & ((self.sentry[:, S.Y] - scy).abs() <= shh)
                         & (self.sentry[:, S.DEAD] <= 0))
            if c.vuln_on_respawn:
                self.sentry[:, S.VULN] = torch.where(
                    in_supply, torch.zeros_like(self.sentry[:, S.VULN]),
                    self.sentry[:, S.VULN])
                self.sentry[:, S.INVULN] = torch.where(
                    in_supply, torch.zeros_like(self.sentry[:, S.INVULN]),
                    self.sentry[:, S.INVULN])
            heal_rate = torch.where(
                cd_diff <= -c.cd_tier2,
                torch.full_like(cd_diff, c.heal_supply_t2),
                torch.where(cd_diff <= -c.cd_tier1,
                            torch.full_like(cd_diff, c.heal_supply_t1),
                            torch.full_like(cd_diff, c.heal_supply_per_s)))
            self.sentry[:, S.HP] = (self.sentry[:, S.HP]
                                    + heal_rate * c.sentry_hp * c.dt * in_supply.float()
                                    ).clamp_max(c.sentry_hp)

        # ---- 奖励 ----
        # 组成（注意：2026 的"胜负"来自胜利率先归零，与 2025 相反）
        #   1) 目标导向项：胜利点净优势的**逐步变化** × w_vp_adv
        #      —— 这是唯一能表达"占控制区能赢"的项。少了它，2026 扣分制下
        #         策略会退化成"躲远处不死"（实测占点率掉到 0.000）。
        #   2) 输出/承受伤害、过热惩罚（塑形，帮助早期学习）
        overheat = (self.sentry[:, S.HEAT1] > c.heat_limit * 0.95).float() + \
                   ((self.sentry[:, S.HEAT2] > c.heat_limit * 0.95).float()
                    if c.n_barrels >= 2 else 0.0)
        vp_adv = self.sentry[:, S.VP] - self.sentry[:, S.VP_ENEMY]
        # 注意：drain 模式下开局 vp_adv = 0，accumulate 模式下也是 0，量级一致
        advance = vp_adv - self._prev_vp_adv
        self._prev_vp_adv = vp_adv

        # 3) 接近控制区：**势函数塑形**（potential-based reward shaping）
        #
        # 理论依据（Ng, Harada & Russell 1999 的 policy invariance 定理）：
        #   只有形如 F(s,a,s') = γΦ(s') − Φ(s) 的塑形才**保证不改变最优策略**。
        #   任何其它形式的塑形都可能引入新的局部最优，让策略收敛到错误行为。
        #   arXiv 1911.01417《Sibling Rivalry》用它作为反例基础：
        #   "simple distance-to-goal reward shaping often fails, as it renders
        #    learning vulnerable to local optima" —— 他们实测朴素距离塑形的
        #   PPO 在多个迷宫任务上**完全学不到目标**。
        #
        # ⚠️ 我最初的实现违反了这条定理（三处），这也是反复崩盘的嫌疑之一：
        #   (a) 写成 (prev_dist - dist_now)/(v_max*dt)，即 −Δd，形式是 Φ(s)−Φ(s')
        #       而不是 γΦ(s')−Φ(s)；
        #   (b) 加了 .clamp(-1,1) —— **裁剪会破坏势函数性质**，正是"引入新局部最优"
        #       的来源；
        #   (c) 用 Φ = −d（线性无界），才不得不靠 clamp 压住，恶性循环。
        #   → 改为有界势函数 Φ = −tanh(d/σ)，这样 γΦ(s')−Φ(s) 天然有界，**无需裁剪**。
        #
        # σ 取 3.0 m：控制区半径仅 0.8 m，用远大于它的尺度才能在"离得很远"时
        # 仍有可分辨的梯度（tanh 在 d≫σ 时饱和，梯度消失）。
        phi_now = self._phi_value()
        if not hasattr(self, "_prev_phi"):
            self._prev_phi = phi_now.clone()
            self._phi_valid = torch.zeros_like(phi_now, dtype=torch.bool)
        # 新一局/复活时历史势函数无效 -> 本步不计塑形（否则会瞬移刷分）
        fresh = ~self._phi_valid
        shaping = (c.gamma * phi_now - self._prev_phi) * fresh.float()
        self._prev_phi = phi_now.clone()
        self._phi_valid = torch.ones_like(phi_now, dtype=torch.bool)

        reward = (c.w_vp_adv * advance
                  + c.w_dmg_out * dmg * fire.float()
                  + c.w_dmg_in * enemy_dmg
                  + c.w_kill * kills
                  + c.w_hold * self._hold_sentry.float() * c.dt
                  + c.w_approach * shaping
                  - c.w_overheat * overheat * c.dt)

        # ---- 终止（§3.3.7 / §3.3.5）----
        self.step_count += 1
        timeout = self.step_count >= self.max_steps
        if c.vp_mode == "accumulate":
            # 2025: 一方胜利点到达 200 立即结束
            win = self.sentry[:, S.VP] >= c.vp_win
            lose = torch.zeros_like(win)
        else:
            # 2026+: 一方胜利点为 0 立即结束
            win = self.sentry[:, S.VP_ENEMY] <= c.vp_win
            lose = self.sentry[:, S.VP] <= c.vp_win
        done = timeout | win | lose
        # 超时则比胜利点高低
        if c.vp_mode != "accumulate":
            win = win | (timeout & (self.sentry[:, S.VP] > self.sentry[:, S.VP_ENEMY]))
            lose = lose | (timeout & (self.sentry[:, S.VP] < self.sentry[:, S.VP_ENEMY]))
        reward = reward + win.float() * c.w_win - lose.float() * c.w_win

        obs = self._push_obs(self._build_obs())
        info = {
            "vp": self.sentry[:, S.VP].clone(),
            "vp_enemy": self.sentry[:, S.VP_ENEMY].clone(),
            "dmg_out": self.sentry[:, S.DMG_OUT].clone(),
            "dmg_in": self.sentry[:, S.DMG_IN].clone(),
            "ammo": self.sentry[:, S.AMMO].clone(),
            "hold_ratio": self._hold_sentry.float(),
            "win": win.float(),
            "lose": lose.float(),
            "timeout": timeout.float(),
        }
        if c.vp_mode == "control_degree":
            info["ctrl_ours"] = self._ctrl_s.sum(dim=1).float().clone()
            info["ctrl_enemy"] = self._ctrl_e.sum(dim=1).float().clone()
            info["zone_p"] = self.zone_p.clone()
        return obs, reward, done, info

    def auto_reset(self, done: torch.Tensor):
        if bool(done.any()):
            self.reset(done)


# --------------------------------------------------------------------------
# Gymnasium 单环境封装（用于算法库、可视化、调参）
# --------------------------------------------------------------------------
if _HAS_GYM:

    class RMUL3v3Env(gym.Env):
        """Gymnasium 接口。内部仍走批量实现（n_envs=1），便于与训练代码共用一套逻辑。"""

        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

        def __init__(self, cfg: RMUL3v3Config | None = None, device: str = "cpu",
                     seed: int | None = None):
            super().__init__()
            self.cfg = cfg or RMUL3v3Config()
            self.core = BatchedRMUL3v3(self.cfg, n_envs=1, device=device, seed=seed)
            self.n = 1
            # 混合动作空间：用 Dict 表达"连续速度 + 离散目标 + 离散开火"
            self.action_space = spaces.Dict({
                "move": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "target": spaces.Discrete(self.cfg.n_enemies + 1),
                "fire": spaces.Discrete(2),
            })
            self.observation_space = spaces.Box(-2.0, 2.0,
                                                shape=(self.core.obs_dim,), dtype=np.float32)

        # --- 工具 ---
        def action_to_tensor(self, action) -> torch.Tensor:
            if isinstance(action, dict):
                t = np.zeros(self.core.n_actions, dtype=np.float32)
                t[0:2] = np.asarray(action["move"], dtype=np.float32)
                t[2] = float(action["target"])
                t[3] = float(action["fire"])
            else:
                t = np.asarray(action, dtype=np.float32)
            return torch.as_tensor(t, device=self.core.device).unsqueeze(0)

        def action_mask(self) -> np.ndarray:
            return self.core.action_mask()[0].cpu().numpy()

        # --- API ---
        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.core.gen.manual_seed(seed)
            obs = self.core.reset()
            return obs[0].cpu().numpy(), {}

        def step(self, action):
            obs, rew, done, info = self.core.step(self.action_to_tensor(action))
            terminated = bool(info["win"][0] > 0.5)
            truncated = bool(info["timeout"][0] > 0.5)
            i = {k: (v[0].item() if torch.is_tensor(v) else v) for k, v in info.items()}
            return obs[0].cpu().numpy(), float(rew[0]), terminated, truncated, i

        def render(self):  # pragma: no cover - 轻量文本渲染
            s = self.core.sentry[0]
            return (f"t={float(self.core.step_count[0]) * self.cfg.dt:6.1f}s "
                    f"pos=({float(s[S.X]):5.2f},{float(s[S.Y]):5.2f}) "
                    f"hp={float(s[S.HP]):5.1f} ammo={float(s[S.AMMO]):5.0f} "
                    f"heat=({float(s[S.HEAT1]):5.1f},{float(s[S.HEAT2]):5.1f}) "
                    f"vp={float(s[S.VP]):6.1f} center={bool(self.core.gen_center[0])}")


# --------------------------------------------------------------------------
if __name__ == "__main__":  # python -m rm_sentry.rmul_3v3_env [--season 2026]
    import sys

    season = "2026"
    if "--season" in sys.argv:
        season = sys.argv[sys.argv.index("--season") + 1]
    cfg = RMUL3v3Config(season=season)
    print(f"赛季 {cfg.season}  |  规则来源: {cfg.profile_source}")
    print(f"场地 {cfg.field_w}x{cfg.field_h} m, 时长 {cfg.match_time}s, "
          f"决策 {cfg.decision_hz}Hz, 对手 {cfg.n_enemies} 台")
    print(f"哨兵: {cfg.sentry_hp}hp / {cfg.n_barrels} 管 / 热量{cfg.heat_limit:.0f} / "
          f"冷却{cfg.heat_cool:.0f}/s（持续射速 {cfg.heat_cool / cfg.heat_per_shot:.1f} 发/秒）/ "
          f"{cfg.ammo_init}发 / 底盘{cfg.chassis_power:.0f}W")
    print(f"17mm 单发伤害 {cfg.dmg_base:.0f}")
    if cfg.vp_mode == "accumulate":
        print(f"胜利点(2025): 从 {cfg.vp_init:.0f} 累积, 占{cfg.gain_name} +{cfg.vp_center_coef:.0f}/s, "
              f"击杀 +{cfg.vp_kill_coef:.0f}, 满 {cfg.vp_win:.0f} 获胜")
    elif cfg.vp_mode == "control_degree":
        print(f"控制度({cfg.season}): 双方各 {cfg.vp_init:.0f}; {len(cfg.zones_rect)} 处控制区"
              f"独立进度 0~{cfg.zone_progress_max:.0f}; 每控制 1 处对方 −{cfg.vp_center_coef:.0f}/s; "
              f"战亡自扣 {cfg.vp_kill_coef:.0f}; 归 {cfg.vp_win:.0f} 判负（判定细节待手册精读）")
    else:
        print(f"胜利点({cfg.season}): 双方各 {cfg.vp_init:.0f} 起扣, 占{cfg.gain_name} 使对方 "
              f"−{cfg.vp_center_coef:.0f}/s, 战亡自扣 {cfg.vp_kill_coef:.0f}, "
              f"一方归 {cfg.vp_win:.0f} 获胜")

    if not _HAS_GYM:
        print("\n[提示] 未安装 gymnasium —— 纯张量批量环境仍可用，只是没有 Gym 标准接口。")
        print("       安装: pip install gymnasium")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = BatchedRMUL3v3(cfg, n_envs=8, device=dev, seed=0)
    print(f"\n设备 {dev}, obs_dim={env.obs_dim}, n_actions={env.n_actions}")
    obs = env.reset()
    print("reset obs shape:", tuple(obs.shape))
    print("action_mask:\n", env.action_mask().int().cpu().numpy())
    for _ in range(5):
        act = torch.zeros(8, env.n_actions, device=dev)
        act[:, 0] = 0.5
        act[:, 2] = 1.0
        act[:, 3] = 1.0
        obs, rew, done, info = env.step(act)
    print("5 步后: vp=", info["vp"].cpu().numpy().round(2),
          " dmg_out=", info["dmg_out"].cpu().numpy().round(1),
          " ammo=", info["ammo"].cpu().numpy())
