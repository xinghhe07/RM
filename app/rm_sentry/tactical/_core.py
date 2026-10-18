"""战术层核心数据契约（几何无关）。

⚠️ 本模块与任何具体场地坐标**无关**。理由见交接文档 §9 与 `工作台.md` §0：
   场地参数曾经是编的，禁止再造。这里的每个坐标都只作为**配置数据**在
   `TacticalPoints` 里出现 —— 真实场地图到位后只换数据，不改逻辑。

上游设计依据：`哨兵方案设计.md` §3.1（三层决策）
  V0 固定优先级 + 状态机 → V1 规则决策树 + 动态点位评分 → V2 PPO 选战术动作

本模块提供 V1/V2 共用的**接口**：
  1. 命名点 + 区域绑定  —— 点的唯一 ID 就是战术层的输出取值；RL 的离散动作集
     就是这个集合（与环境的 action_mask 机制天然对齐）。
  2. 运行时状态快照     —— `TacticalState`，决策的唯一输入。
  3. 上一次决策          —— `TacticalDecision`，用于跨步的迟滞与状态机。

设计约束（团队 2026-10-13 讨论确认）：
  - 点 = 命名点 + 区域绑定，**不做网格化**（网格粒度收益≈0，复杂度高，且会毁掉
    "规则层与 RL 层共用同一输出空间"这个前提）。
  - 控制区在规则里是**区域**，用 `zone_id` 绑定点 -> 区域，到达判定按半径而不是点。
  - 所有区间常量必须带单位后缀，避免单位混用（团队 2026-10-13 提的契约要求）。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, TypeVar

from rm_sentry.tactical import config

# --------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------


class PointKind(enum.Enum):
    """点位的战术类型（定义见 `哨兵方案设计.md` §3.1 的"战术点图"）。"""

    CONTROL = enum.auto()    # 控制区（规则里的得分区域）
    PATROL = enum.auto()     # 巡逻点
    OBSERVE = enum.auto()    # 观察点（改善感知，不要求占点）
    WAIT = enum.auto()       # 等待点（低风险待命）
    RECOVERY = enum.auto()   # 恢复点（导航异常后重规划）
    SUPPLY = enum.auto()     # 补给点（建模未启用，规则允许时才用）


class ZoneOwner(enum.Enum):
    """区域归属。UNKNOWN 与"安全"**不等价** —— 信息过期不能当安全用。"""

    FRIENDLY = enum.auto()   # 我方有效占领
    ENEMY = enum.auto()      # 敌方有效占领
    CONTESTED = enum.auto()  # 争夺中（双方同在 / 先到先得未定）
    NOBODY = enum.auto()     # 确认无人占领
    UNKNOWN = enum.auto()    # 无信息或信息过期


class TacticalAction(enum.Enum):
    """战术动作集 —— **规则层（V1）与 RL 层（V2）共用**。

    RL 的离散动作 = 本枚举中"需要目标"的那几项 × 目标点集合。
    增删动作必须同步更新 `ACTION_NEEDS_TARGET` 与回归测试。
    """

    HOLD_CURRENT_ZONE = enum.auto()  # 保持当前区域
    CAPTURE_POINT = enum.auto()      # 前往争夺指定点
    DEFEND_POINT = enum.auto()       # 防守指定点
    PATROL_POINT = enum.auto()       # 巡逻到指定点
    RETREAT = enum.auto()            # 撤离（可带目标安全点）
    RECOVER = enum.auto()            # 恢复定位 / 导航
    SUPPORT_TEAMMATE = enum.auto()   # 支援队友
    SAFE_MODE = enum.auto()          # 最低安全：停


#: 需要目标点的动作（RL 的掩码与状态机的转移都要用）
ACTION_NEEDS_TARGET: frozenset[TacticalAction] = frozenset({
    TacticalAction.CAPTURE_POINT,
    TacticalAction.DEFEND_POINT,
    TacticalAction.PATROL_POINT,
})

T = TypeVar("T")


# --------------------------------------------------------------------------
# 静态配置数据（坐标的**唯一**合法入口）
# --------------------------------------------------------------------------

#: 几何判定用的容差（米）。只为吸收浮点误差，**不是**到达容差 ——
#: 到达容差是 `decision.ARRIVAL_TOL_M`，两者语义不同，别混用。
_GEOM_EPS = 1e-9


@dataclass(frozen=True)
class ZoneSpec:
    """规则区域的静态定义（控制区 / 补给区 / 出生区……）。

    **两种形状**（2027 迁移时加的 —— 官方控制区是矩形，不是圆）：
      - 圆：给 `radius_m`（2025/2026 的中心增益点、以及历史测试用）
      - 矩形：给 `half_w_m` + `half_h_m`（2027 三控制区：中央 3000×3000、两高地 1200×1200）

    为什么不能只用圆代替矩形：3000×3000 的方区，其内切圆半径 1.5 m 覆盖不到四个角
    （角点距中心 1.5·√2 ≈ 2.12 m）。用 `dist <= r` 判"是否在区内"会把**真正站在区里的
    机器人判成没进区** —— 2027 的占点判定直接因此错。故本类提供统一的
    `contains()` / `distance_outside_m()`，调用方**不得**自己拿 `radius_m` 算距离。

    边界语义：**边界含入**（在边上算在区内）。与 `开源借鉴.md` §2.1 记的中科大
    `Area_Square::contains`（边界含入）保持一致的工程习惯。

    ⚠️ `x_m / y_m / radius_m / half_*_m` 一律为**配置数据**：坐标只能来自
       `docs/2027场地规格.md`（官方图纸实测）或环境 `cfg`，禁止在逻辑里写死。
    """

    zone_id: str
    name: str
    x_m: float
    y_m: float
    radius_m: float = 0.0                      # 圆半径（矩形区填 0）
    strategic_value: float = 1.0               # 相对战略价值（控制区之间可比）
    control_zone: bool = False                 # 是否为规则中的得分控制区
    half_w_m: float | None = None              # 矩形半宽（x 方向）
    half_h_m: float | None = None              # 矩形半高（y 方向）

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id 不能为空（点位要绑它、日志要引用它）")
        rect = self.half_w_m is not None or self.half_h_m is not None
        if rect and (self.half_w_m is None or self.half_h_m is None):
            raise ValueError(
                f"区域 {self.zone_id}：矩形必须同时给 half_w_m 与 half_h_m"
                f"（当前 half_w_m={self.half_w_m}, half_h_m={self.half_h_m}）")
        if rect and self.radius_m > 0.0:
            raise ValueError(
                f"区域 {self.zone_id}：圆与矩形二选一，不能同时给 "
                f"radius_m={self.radius_m} 与 half_w/half_h")
        if rect and (self.half_w_m <= 0.0 or self.half_h_m <= 0.0):
            raise ValueError(f"区域 {self.zone_id}：矩形半宽/半高必须为正")
        if not rect and self.radius_m <= 0.0:
            raise ValueError(
                f"区域 {self.zone_id}：必须给 radius_m>0（圆）或 half_w_m/half_h_m（矩形）")

    @property
    def is_rect(self) -> bool:
        return self.half_w_m is not None and self.half_h_m is not None

    def contains(self, x_m: float, y_m: float) -> bool:
        """点是否在区域内（**边界含入**）。"""
        dx, dy = x_m - self.x_m, y_m - self.y_m
        if self.is_rect:
            assert self.half_w_m is not None and self.half_h_m is not None
            return abs(dx) <= self.half_w_m + _GEOM_EPS and abs(dy) <= self.half_h_m + _GEOM_EPS
        return math.hypot(dx, dy) <= self.radius_m + _GEOM_EPS

    def distance_outside_m(self, x_m: float, y_m: float) -> float:
        """点到区域**边界**的距离；已在区内（或恰在边界上）返回 0。

        语义 = "还要走多远才算进区"，供适配器算路径代价代理（真车上换 Nav2 的真实代价）。
        """
        dx, dy = x_m - self.x_m, y_m - self.y_m
        if self.is_rect:
            assert self.half_w_m is not None and self.half_h_m is not None
            ox = max(0.0, abs(dx) - self.half_w_m)
            oy = max(0.0, abs(dy) - self.half_h_m)
            return math.hypot(ox, oy)
        return max(0.0, math.hypot(dx, dy) - self.radius_m)


@dataclass(frozen=True)
class PointSpec:
    """命名点的静态定义。点的 ID 在整局内唯一、稳定、可日志化。"""

    point_id: str
    name: str
    kind: PointKind
    x_m: float
    y_m: float
    zone_id: str | None = None   # CONTROL 类点位应绑定区域
    value: float = 1.0           # 相对战略价值；缺省继承所属区域

    def __post_init__(self) -> None:
        if not self.point_id:
            raise ValueError("point_id 不能为空（决策输出要引用它）")
        if not self.name:
            raise ValueError(f"点 {self.point_id} 必须有可读名称（便于日志与复盘）")


class TacticalPoints:
    """静态点图容器。**坐标只从这里来**。

    校验是故意的：控制类点位必须绑定区域（否则"到点判定"退化成"追一个坐标"，
    违反"控制区是区域不是点"的规则语义）。
    """

    def __init__(self, zones: list[ZoneSpec], points: list[PointSpec]):
        self._zones: dict[str, ZoneSpec] = {}
        for z in zones:
            if z.zone_id in self._zones:
                raise ValueError(f"区域 ID 重复：{z.zone_id}")
            self._zones[z.zone_id] = z

        self._points: dict[str, PointSpec] = {}
        for p in points:
            if p.point_id in self._points:
                raise ValueError(f"点 ID 重复：{p.point_id}")
            if p.zone_id is not None and p.zone_id not in self._zones:
                raise ValueError(f"点 {p.point_id} 绑定了不存在的区域 {p.zone_id}")
            if p.kind is PointKind.CONTROL and p.zone_id is None:
                raise ValueError(
                    f"控制类点位 {p.point_id} 必须绑定区域（zone_id）—— "
                    f"控制区是区域，不是坐标点")
            self._points[p.point_id] = p

    # ---- 查询 ----
    def point(self, point_id: str) -> PointSpec:
        if point_id not in self._points:
            raise KeyError(f"未知点位 {point_id}；已知：{sorted(self._points)}")
        return self._points[point_id]

    def zone(self, zone_id: str) -> ZoneSpec:
        if zone_id not in self._zones:
            raise KeyError(f"未知区域 {zone_id}；已知：{sorted(self._zones)}")
        return self._zones[zone_id]

    def points(self) -> tuple[PointSpec, ...]:
        return tuple(self._points.values())

    def zones(self) -> tuple[ZoneSpec, ...]:
        return tuple(self._zones.values())

    def of_kind(self, kind: PointKind) -> tuple[PointSpec, ...]:
        return tuple(p for p in self._points.values() if p.kind is kind)

    def control_points(self) -> tuple[PointSpec, ...]:
        return tuple(p for p in self._points.values()
                     if p.kind is PointKind.CONTROL)

    def safe_points(self) -> tuple[PointSpec, ...]:
        """撤离/恢复可用的安全点（等待点 + 恢复点）。"""
        return tuple(p for p in self._points.values()
                     if p.kind in (PointKind.WAIT, PointKind.RECOVERY))

    def point_ids(self) -> tuple[str, ...]:
        """稳定排序的点 ID —— RL 动作索引就按这个顺序（契约）。"""
        return tuple(sorted(self._points))


# --------------------------------------------------------------------------
# 运行时状态（每个决策周期从感知/裁判系统刷新，**不**在决策里反推）
# --------------------------------------------------------------------------


@dataclass
class PointEstimate:
    """某个点的运行时状态估计。

    ⚠️ 每个字段都要能回答"它是什么时候的"。没有任何时间戳的状态**不允许**参与决策。
    """

    point_id: str
    reachable: bool = True        # 规划器/可达性模块给出的结论
    path_cost_s: float = 0.0      # 实际可行路径的预计耗时（不是直线距离）
    risk: float = 0.0             # [0,1] 预计风险
    age_s: float = 0.0            # 距最近一次观测/确认过了多久（秒）
    teammate_intent: bool = False  # 是否已有队友执行同一任务
    blocked: bool = False         # 明确不可达（规划失败/被占死）
    # 到点判定用的形状（控制类点位用；None -> 默认容差 ARRIVAL_TOL_M）
    # 圆给 radius_m；矩形给 half_w_m + half_h_m（2027 三控制区）。
    radius_m: float | None = None
    half_w_m: float | None = None
    half_h_m: float | None = None


@dataclass
class ZoneStatus:
    """区域运行时归属。`owner=UNKNOWN` 表示信息不可用或已过期。"""

    zone_id: str
    owner: ZoneOwner = ZoneOwner.UNKNOWN
    age_s: float = 0.0


@dataclass
class RobotStatus:
    """自机状态。所有时间量单位 = 秒。"""

    x_m: float = 0.0
    y_m: float = 0.0
    hp_fraction: float = 1.0       # [0,1]
    dead_secs: float = 0.0         # >0 = 读条剩余（2027）/ 读条+无敌剩余（旧赛季）
    heat_fraction: float = 0.0     # [0,1] 当前热量 / 上限
    ammo: int = 0
    vulnerable: bool = False       # 规则"虚弱"状态（锁发射 + 不能占控制区）
    # 数据源是否**忠实提供**上一条状态。2027 环境/真车状态源为 True；旧赛季
    # 环境为 False（S.VULN 恒 0 是"未建模"，不是"确认不虚弱"）——此时战术层
    # 回退到 `TacticalConfig.vulnerable_secs_assumed` 计时器（见 decision.py）。
    vulnerable_known: bool = False
    position_valid: bool = True    # 定位是否有效


@dataclass
class NavigationStatus:
    """导航健康度（由导航层提供，战术层**不**自己算）。"""

    planner_ok: bool = True
    consecutive_failures: int = 0


@dataclass
class EnemyEstimate:
    """敌方态势的**感知估计**（不是真值 —— 哨兵禁止机间通信，拿不到敌人坐标）。

    本阶段只保留"最近观测"语义；什么时候算过期由 `TacticalConfig` 决定。
    """

    x_m: float
    y_m: float
    age_s: float = 0.0
    threat: float = 0.0            # [0,1]


@dataclass
class TacticalState:
    """决策的**唯一输入快照**。纯数据，可序列化、可构造、可单测。

    ⚠️ 与 `rmul_3v3_env.py` 的向量状态**解耦**：环境适配器负责从张量里
       抽取这些字段，战术层不直接读张量列（真车上从 ROS 话题取同样的字段）。
    """

    robot: RobotStatus = field(default_factory=RobotStatus)
    nav: NavigationStatus = field(default_factory=NavigationStatus)
    zones: dict[str, ZoneStatus] = field(default_factory=dict)
    points: dict[str, PointEstimate] = field(default_factory=dict)
    enemies: tuple[EnemyEstimate, ...] = ()
    teammates: tuple[EnemyEstimate, ...] = ()
    match_time_left_s: float = 300.0

    def zone_owner(self, zone_id: str) -> ZoneOwner:
        zs = self.zones.get(zone_id)
        return zs.owner if zs is not None else ZoneOwner.UNKNOWN

    def point_est(self, point_id: str) -> PointEstimate:
        return self.points.get(point_id) or PointEstimate(point_id=point_id)


@dataclass
class TacticalDecision:
    """一次战术决策的输出 + 决策依据（依据必须能进日志，用于赛后复盘）。"""

    action: TacticalAction
    point_id: str | None = None   # 仅 ACTION_NEEDS_TARGET 的动作使用
    reason: str = ""


# --------------------------------------------------------------------------
# 数值有效性检查（供安全监督层使用）
# --------------------------------------------------------------------------


def collect_nonfinite(obj: Any, prefix: str = "state") -> list[str]:
    """递归找出对象里所有非有限数值（NaN / ±Inf）。

    安全监督层用它判定"数值异常 -> 立即降级"（`工程红线.md` §2）。
    只认数值，不认字符串/枚举，避免误报。
    """
    bad: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, float):
            if node != node or node in (float("inf"), float("-inf")):
                bad.append(path)
        elif isinstance(node, int) or isinstance(node, bool):
            return
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__.values():
                walk(getattr(node, f.name), f"{path}.{f.name}")

    walk(obj, prefix)
    return bad
