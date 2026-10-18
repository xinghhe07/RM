"""战术层配置与阈值。

⚠️ **每个常量都必须标注出处或"待标定"** —— 这是 `开源借鉴.md` §0.1 的硬要求：
   照抄阈值前必须核对场景可比性（踩过 hp_min=200 的坑，见 `rule_baseline.py`）。

分类约定：
  [规则]  —— 官方规则手册给定，可直接引用
  [开源]  —— 借鉴开源实现（注明文件/参数）
  [默认]  —— 工程默认值，**待真实数据/标定**，允许调整但必须记录改动原因
  [假设]  —— ⚠️ 无开源依据、规则未确认的工程假设；使用即承担风险，必须复核
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 决策阈值
# --------------------------------------------------------------------------


@dataclass
class TacticalConfig:
    """战术层全部可调阈值。改这里 = 改行为，改前先读本文件顶部的分类约定。"""

    # ---- 迟滞 / 最短驻留（防抖动）----
    # [默认] 目标保持的最短时间（秒）。参考卡的"目标抖动 -> 加迟滞"处置。
    min_dwell_s: float = 1.0
    # [默认] 换点所需的最小分数优势（归一化分数，0~1）。
    # 语义：只有"新点比当前点高出 switch_margin 以上"才换 —— 这是防抖的唯一阈值，
    # 不要再叠加"当前点加成"，否则两个机制会互相干扰（实现时踩过一次）。
    # ⚠️ 必须与评分权重的量级匹配：单个分量能造成的最大分差 = 它的权重。
    #    实测：switch_margin 曾取 0.15，而"代价"分量最大只能拉出 0.06 的差距
    #    -> 任何情况下都换不了点（策略被钉死）。现取 0.05（略高于噪声量级）。
    #    改权重必须重跑 TC6/TC6b（防抖 + 明显更优仍能换点）。
    switch_margin: float = 0.05

    # ---- 血量门控 ----
    # [默认] 低于此血量比例 -> 撤退到安全点。
    # ⚠️ 必须 < 复活血量比例（环境中 0.2）与"可战血量"之间；
    #    取 0.30 的后果：复活后（20%）先撤，直到 30% 才恢复战术行为。
    #    改这个值必须重跑 test_tactical.py 的 respawn 场景。
    retreat_hp_fraction: float = 0.30
    # [默认] 血量回升到该值以上才恢复战术行为（与 retreat 拉开迟滞带）。
    resume_hp_fraction: float = 0.35

    # ---- 信息新鲜度 ----
    # [默认] 区域状态超过该秒数 -> 视为 UNKNOWN（不能把过期信息当安全）。
    zone_stale_after_s: float = 5.0
    # [默认] 点位估计超过该秒数 -> 不可信，不参与选点。
    point_stale_after_s: float = 8.0

    # ---- 复活 / 保护期 / 虚弱 ----
    # [规则] RMUL 2026 §3.3.2.2：复活后无敌 10 秒（与环境 respawn_invuln 同源）。
    # 与环境配置的一致性由 test_tactical.py 检查，防止两处漂移。
    respawn_invuln_s: float = 10.0
    # [回退假设] 复活后"虚弱"持续时间（秒）——**仅在数据源不提供该状态时使用**。
    #   2027 数据源（`RobotStatus.vulnerable_known=True`，来自 S.VULN）以现场状态为准：
    #   虚弱持续到接触己方补给区，**不倒数**；2025/2026 旧赛季环境未建模该状态，
    #   回退到此计时器（旧行为逐位不变）。
    # ⚠️ 2026 规则手册只确认存在虚弱的判定，未（在已抽取文本中）给出秒数。
    #    使用前必须回查手册 §3.3.2.2 / 官方答疑；此值不影响安全性（只影响是否
    #    进入保守行为），但会影响胜率评估口径。
    vulnerable_secs_assumed: float = 3.0

    # ---- 决策周期 ----
    # [规则] 环境 dt=0.05s；[默认] 战术决策 10 Hz（与环境 decision_hz 一致）。
    decision_dt_s: float = 0.1

    # ---- 导航健康 ----
    # [默认] 连续规划失败达到该次数 -> 进入 RECOVER。
    nav_fail_limit: int = 3
    # [默认] RECOVER 状态的最长时间（秒），超时升级为 SAFE_STOP。
    nav_recover_max_s: float = 5.0
    # [默认] 任务卡住的固定宽限（秒）：超时阈值 = 宽限 + 2×预计路径耗时。
    #   待按真实导航的路径跟踪误差与重规划开销标定。
    stuck_grace_s: float = 20.0
    # [默认] 进入 RECOVERING 后的最短保持时间（秒）。
    #   关键：没有它就出现"卡住 -> 恢复 -> 立刻又发同一个目标 -> 又卡住"的抖振
    #   （实现时实测到 80ms 内来回切）。恢复必须真的有持续窗口，否则等于没恢复。
    recover_hold_s: float = 2.0

    # ---- 安全监督 ----
    # [默认] 定位失效后等待数据恢复的最长时间（秒）；超时升级为 SAFE_STOP。
    #   0.5s = 5 个决策周期：定位是导航与战术的共同前提，丢失必须尽快停。
    #   ⚠️ 待用实车的定位置信度定义标定（"失效"的判据本身也来自定位模块）。
    localization_loss_max_s: float = 0.5
    # [默认] 恢复需要"持续有效"的确认时长（秒）。防止观测一跳一恢复就回到战术层。
    recovery_confirm_s: float = 1.0


@dataclass
class ScoringWeights:
    """选点评分权重（归一化分数）。

    分数 = w_value*V + w_control*C - w_cost*D - w_risk*R - w_stale*S - w_conflict*T

    约束（故意做成断言）：正值权重之和 = 1.0，保证分数在 [0,1] 内可比。
    ⚠️ 所有权重都是 **[默认] 待标定**：真实场地与对手数据到位后，用
       `eval_baselines.py` 的固定种子评估来调，不要凭感觉改。
    """

    w_value: float = 0.35
    w_control: float = 0.40
    w_cost: float = 0.15   # 与 switch_margin 的量级直接相关，改前先读 switch_margin 注释
    w_risk: float = 0.05
    w_stale: float = 0.03
    w_conflict: float = 0.02

    # [默认] 路径耗时的归一化尺度（秒）：cost=1 对应"这条路径要走满这么久"。
    #   30s ≈ 慢速横穿一次场地的量级；真实场地图到位后按实际路径耗时标定。
    path_scale_s: float = 30.0

    def __post_init__(self) -> None:
        total = self.w_value + self.w_control + self.w_cost + self.w_risk
        total += self.w_stale + self.w_conflict
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"评分权重之和必须为 1.0（当前 {total:.4f}）—— "
                f"否则点与点之间的分数不可比，迟滞阈值会失效")
        if self.path_scale_s <= 0:
            raise ValueError("path_scale_s 必须为正（它是分子归一化尺度）")


@dataclass
class DecisionPriorities:
    """决策优先级栈（高优先级在前，对应文档里的决策树顺序）。

    ⚠️ 顺序即策略，不要随意调换：
      安全 → 复活/状态限制 → 导航有效 → 当前任务保持 → 重新选点
    """

    order: tuple[str, ...] = field(default_factory=lambda: (
        "safety",          # 安全监督层（最高，不可被战术覆盖）
        "respawn",         # 战亡 / 复活保护 / 虚弱
        "navigation",      # 定位与导航有效性
        "current_task",    # 当前任务是否仍值得执行（迟滞）
        "reselect",        # 重新选点（评分）
    ))
