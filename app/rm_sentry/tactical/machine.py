"""执行状态机：把战术决策落到实处，并保证异常也能退出。

契约（团队 2026-10-13 讨论确认的"状态机契约"）：
    每个状态都必须定义 **进入条件 / 退出条件 / 超时条件 / 失败后动作**。
    这是 `工程红线.md` §7 的"兜底要能说清坏在哪"在战术层的落地。

    任务状态机（本模块）：     IDLE -> MOVING -> EXECUTING -> ...
    安全状态机（decision.py）：NORMAL / DEGRADED / RECOVERY / SAFE_STOP

    两者**独立**：安全状态任何时刻都可以把任务状态机打断并压到 IDLE，
    反过来不允许（战术层不得绕过安全层）。

设计取舍：
  - 状态机只认识 `TacticalDecision` + `TacticalState`，不认识环境张量 ——
    这样它既能跑在仿真里，也能跑在真车的 ROS 节点里。
  - 所有计时都由调用方驱动（`decision_dt_s`），不引入隐藏的全局时钟。
  - `history` 是完整审计日志；`update()` 返回本次新增的迁移（没有则 None）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from rm_sentry.tactical._core import (
    ACTION_NEEDS_TARGET,
    PointSpec,
    TacticalAction,
    TacticalDecision,
    TacticalPoints,
    TacticalState,
)
from rm_sentry.tactical.config import TacticalConfig
from rm_sentry.tactical.decision import arrive_at


class TaskState(enum.Enum):
    """战术任务状态。"""

    IDLE = enum.auto()        # 没有有效任务（等待 / 安全层压制 / 无可达点）
    MOVING = enum.auto()      # 正在前往目标点
    EXECUTING = enum.auto()   # 已到位，执行任务（占点 / 防守 / 巡逻）
    RETREATING = enum.auto()  # 撤离中
    RECOVERING = enum.auto()  # 导航 / 定位恢复中


@dataclass
class TaskTransition:
    """一次状态迁移的记录。**必须进日志** —— 赛后复盘全靠它。"""

    previous: TaskState
    current: TaskState
    at_s: float                     # 迁移发生的**绝对**时刻（自状态机启动累计，只增不减）
    task_elapsed_s: float           # 迁移时上一个任务已执行多久（用于归因"卡了多久"）
    reason: str


@dataclass
class TacticalMachine:
    """任务状态机。"""

    cfg: TacticalConfig
    points: TacticalPoints                       # 到点判定需要点的定义（区域半径语义）
    state: TaskState = TaskState.IDLE
    target_point_id: str | None = None
    elapsed_s: float = 0.0                       # 当前任务已执行时长
    total_s: float = 0.0                         # 自启动以来的绝对时间（只增）
    recovering_s: float = 0.0                    # RECOVERING 已保持多久
    history: list[TaskTransition] = field(default_factory=list)

    def update(
        self,
        snapshot: TacticalState,
        decision: TacticalDecision,
    ) -> TaskTransition | None:
        """推进一个决策周期。返回本次新增的迁移记录（无迁移时 None）。"""
        before = len(self.history)
        self._step(snapshot, decision)
        self.elapsed_s += self.cfg.decision_dt_s
        self.total_s += self.cfg.decision_dt_s
        if self.state is TaskState.RECOVERING:
            self.recovering_s += self.cfg.decision_dt_s
        return self.history[before] if len(self.history) > before else None

    # ---- 内部 ----

    def _step(self, snapshot: TacticalState, decision: TacticalDecision) -> None:
        # 安全层压制：无条件压到 IDLE（安全 > 一切战术）
        if decision.action is TacticalAction.SAFE_MODE:
            if self.state is not TaskState.IDLE:
                self._go(TaskState.IDLE, None, f"安全层压制：{decision.reason}")
            return

        if self.state is TaskState.RECOVERING:
            # RECOVERING 最短保持：防止"卡住 -> 恢复 -> 立刻重发同一目标"的抖振
            if self.recovering_s < self.cfg.recover_hold_s:
                return
            # 保持期满 -> 尝试恢复任务（重发当前决策的目标）
            want = self._map(snapshot, decision)
            if want is not TaskState.RECOVERING:
                self._go(want, decision.point_id,
                         f"恢复窗口结束，重试：{decision.action.name}")
            return

        # 超时判定基于**当前**状态与当前目标（先判超时，再看新意图）
        timeout = self._timeout_reason(snapshot)
        if timeout:
            # 超时 -> 强制 RECOVERING（"失败后动作"契约）
            self._go(TaskState.RECOVERING, None, timeout)
            return

        want = self._map(snapshot, decision)
        if want is not self.state or decision.point_id != self.target_point_id:
            self._go(want, decision.point_id, decision.reason or decision.action.name)

    def _map(self, snapshot: TacticalState, decision: TacticalDecision) -> TaskState:
        if decision.action is TacticalAction.RECOVER:
            return TaskState.RECOVERING
        if decision.action is TacticalAction.RETREAT:
            return TaskState.RETREATING
        if decision.action is TacticalAction.HOLD_CURRENT_ZONE:
            return TaskState.EXECUTING       # 决策层已保证"原地"语义

        # 需要目标的动作：按是否**已到点**区分"赶路"与"执行"
        # 到点判定复用决策层的 arrive_at（同一套区域半径语义，避免两处漂移）
        if decision.point_id is None:
            return TaskState.IDLE
        spec = self._spec(decision.point_id)
        if spec is None:
            return TaskState.IDLE
        return TaskState.EXECUTING if arrive_at(snapshot, spec) else TaskState.MOVING

    def _spec(self, point_id: str) -> PointSpec | None:
        try:
            return self.points.point(point_id)
        except KeyError:
            # 未知点位：保守处理（不猜），由决策层负责保证不会出现
            return None

    def _timeout_reason(self, snapshot: TacticalState) -> str:
        """卡住判定。

        超时阈值 = 固定宽限 + 预计路径耗时的 2 倍余量。
        ⚠️ 这里的"2 倍"是 [默认] 待标定值：真实导航的路径跟踪误差与重规划
           开销要用实车数据定，不能照抄。
        """
        if self.state not in (TaskState.MOVING, TaskState.RETREATING):
            return ""
        est = snapshot.point_est(self.target_point_id) if self.target_point_id else None
        eta = est.path_cost_s if est is not None else 0.0
        limit = self.cfg.stuck_grace_s + 2.0 * max(eta, 0.0)
        if self.elapsed_s > limit:
            return (f"任务卡住：已 {self.elapsed_s:.1f}s > 限 {limit:.1f}s"
                    f"（ETA={eta:.1f}s）-> 进入恢复")
        return ""

    def _go(self, new_state: TaskState, target_point_id: str | None, reason: str) -> None:
        self.history.append(TaskTransition(
            previous=self.state, current=new_state,
            at_s=self.total_s, task_elapsed_s=self.elapsed_s, reason=reason))
        self.state = new_state
        self.elapsed_s = 0.0
        self.recovering_s = 0.0
        # 保留目标点的状态：赶路类（超时判定要用 ETA）+ EXECUTING。
        # ⚠️ EXECUTING 也要保留 —— 否则"同状态同目标"会被判为"变了"，每个决策周期
        #    都记一条 EXECUTING->EXECUTING 的假迁移（A2 做日志时实测发现：
        #    5600 条决策配 5600 条迁移，日志被稀释一半）。
        self.target_point_id = (target_point_id
                                if new_state in (TaskState.MOVING, TaskState.RETREATING,
                                                 TaskState.EXECUTING)
                                else None)
