"""战术层：战术点图 + 决策树 + 状态机（几何无关）。

定位（`哨兵方案设计.md` §3.1 的 V1 档）：
    这是"保底方案"的核心 —— 规则决策树 + 带迟滞的动态点位评分。
    未来上 RL（V2）时，RL 只是替换"选点/选动作"这一步，本层的
    安全监督、状态机、执行契约全部保留。

⛔ 红线（交接文档 §9）：**坐标只能是配置数据**，不许在逻辑里写死任何场地参数。
   真实场地图到位后，替换 `TacticalPoints` 的构造数据即可，不改本层代码。

用法：
    from rm_sentry.tactical import build_default_stack
    stack = build_default_stack(points, cfg)     # 见 test_tactical.py 的合成点图
    decision, state = stack.decide(snapshot, state)
"""

from rm_sentry.tactical.adapter import (
    CONTROL_POINT_ID,
    P_HIGH_B_ID,
    P_HIGH_R_ID,
    WAIT_POINT_ID,
    Z_HIGH_B_ID,
    Z_HIGH_R_ID,
    Z_SUPPLY_R_ID,
    TacticalDriver,
    build_points_from_env,
    decision_to_env_action,
    tactical_config_from_env,
    to_tactical_state,
)
from rm_sentry.tactical._core import (
    ACTION_NEEDS_TARGET,
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
    collect_nonfinite,
)
from rm_sentry.tactical.config import (
    DecisionPriorities,
    ScoringWeights,
    TacticalConfig,
)
from rm_sentry.tactical.decision import (
    DecisionState,
    RespawnPhase,
    RespawnTracker,
    SafetyMonitor,
    SafetyState,
    arrive_at,
    decide,
    point_score,
)
from rm_sentry.tactical.logger import DecisionLogger
from rm_sentry.tactical.machine import TaskState, TaskTransition, TacticalMachine

__all__ = [
    # 数据契约
    "ACTION_NEEDS_TARGET", "EnemyEstimate", "NavigationStatus", "PointEstimate",
    "PointKind", "PointSpec", "RobotStatus", "TacticalAction", "TacticalDecision",
    "TacticalPoints", "TacticalState", "ZoneOwner", "ZoneSpec", "ZoneStatus",
    "collect_nonfinite",
    # 配置
    "DecisionPriorities", "ScoringWeights", "TacticalConfig",
    # 决策
    "DecisionState", "RespawnPhase", "RespawnTracker", "SafetyMonitor",
    "SafetyState", "arrive_at", "decide", "point_score",
    # 状态机 / 日志
    "TaskState", "TaskTransition", "TacticalMachine", "DecisionLogger",
    # 环境适配
    "CONTROL_POINT_ID", "WAIT_POINT_ID",
    # 2027 三控制区 + 补给区的 id（点图 id = RL 动作索引空间，属公开契约）
    "P_HIGH_R_ID", "P_HIGH_B_ID", "Z_HIGH_R_ID", "Z_HIGH_B_ID", "Z_SUPPLY_R_ID",
    "TacticalDriver", "build_points_from_env",
    "decision_to_env_action", "tactical_config_from_env", "to_tactical_state",
]
