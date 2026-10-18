"""决策日志（JSONL）：把每个战术决策"为什么这么做"落盘 —— 赛后复盘与离线评估的事实来源。

设计约束：
  - **零依赖**：只用标准库（json / time / pathlib），不 import torch —— 真车节点可直接用。
  - **一行一事件**（JSONL，追加写）：崩溃不毁档，事后 grep / jq 皆可。
  - **感知帧时间戳**：每条记录带 `frame`（环境步数）+ `wall`（墙钟）。
    将来接入相机后据此对齐图像帧（多模态数据管线的接缝，现在留好）。
  - **可关**：驱动层持 `DecisionLogger | None`；为 None 时零开销。

schema（version = 1）：
    header     : {"type": "header", "version", "created", "season", "seed", "config": {...}}
    decision   : {"type": "decision", "env", "t", "frame", "wall", "decision_idx",
                  "robot": {...}, "nav": {...}, "match": {...}, "zones": {...},
                  "enemies": [...], "req": {...}, "exec": {...}, "overrides": [...]}
    transition : {"type": "transition", "env", "t", "frame", "from", "to",
                  "task_elapsed_s", "reason"}

`req` = 决策层的输出（requested）；`exec` = 实际下发给环境的动作（executed）。
V1 里两者的差异来自执行门控（开火条件 / 死亡 / 掩码）；V2 引入 RL 后，
"RL 提议被安全层拒绝"这类事件写进 `overrides`（字段已留）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SAFETY_MODES = ("DEGRADED", "RECOVERY", "SAFE_STOP")
_RESPAWN_MODES = ("DEAD", "RESPAWNED_PROTECTED", "RESPAWNED_VULNERABLE")


def _overrides(state: Any, decision: Any) -> list[dict]:
    """非战术路径的压制记录。V1：规则层内部（安全/复活）；V2：RL 提议被拒也写这里。"""
    if state.mode in _SAFETY_MODES:
        return [{"layer": "safety", "state": state.mode, "reason": decision.reason}]
    if state.mode in _RESPAWN_MODES:
        return [{"layer": "respawn", "state": state.mode, "reason": decision.reason}]
    return []


@dataclass
class DecisionLogger:
    """JSONL 决策日志。用法见 `logged_match.py`（跑一局 + 回读验证）。"""

    path: str | Path
    env_filter: tuple[int, ...] | None = None   # None = 记录全部环境行
    _fh: Any = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)   # 总记录数（自检用）
    # 已结束的环境：`auto_reset` 后 frame 会归零，继续记录会让 t 回跳，
    # 且无法判断一条决策属于哪一局。调用方在该环境 `done` 后注销它
    # （`logged_match.py` 的口径：一运行 = 每环境第一局）。
    _stopped: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self._fh = p.open("a", encoding="utf-8")

    # ---- 基础 ----
    def wants(self, env_idx: int) -> bool:
        """该环境行是否要记（`env_filter=None` 时全记；已结束的环境不再记）。"""
        if env_idx in self._stopped:
            return False
        return self.env_filter is None or env_idx in self.env_filter

    def stop(self, env_idx: int) -> None:
        """该环境的一局已结束：此后拒绝其 decision / transition 记录。

        ⚠️ 必须在 `auto_reset` **之前**调用（之后 `frame` 已归零）。
        这是 A2.1 的口径：日志按 `(env, 第一局)` 唯一确定，不再跨局。
        """
        self._stopped.add(env_idx)

    def _write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._n += 1

    @property
    def count(self) -> int:
        return self._n

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "DecisionLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 三类记录 ----
    def header(self, *, season: str, seed: int | None, config: dict) -> None:
        """文件头：规则档案摘要 + 生成时间（复盘时要能回答"这是哪版环境"。）"""
        self._write({
            "type": "header", "version": 1, "created": time.time(),
            "season": season, "seed": seed, "config": config,
        })

    def log_decision(self, env_idx: int, snapshot: Any, decision: Any,
                     state: Any, exec_out: Any, *, t: float, frame: int,
                     decision_idx: int) -> None:
        if not self.wants(env_idx):
            return
        r = snapshot.robot
        self._write({
            "type": "decision", "env": env_idx,
            "t": round(t, 3), "frame": frame, "wall": time.time(),
            "decision_idx": decision_idx,
            "robot": {"x": round(r.x_m, 3), "y": round(r.y_m, 3),
                      "hp": round(r.hp_fraction, 4), "dead": round(r.dead_secs, 2),
                      "vuln": bool(r.vulnerable),
                      # 虚弱来源是否可信：回放时必须一起还原，否则 2027 会被
                      # 当成旧赛季走计时器（A1.6 的两套口径）。
                      "vuln_known": bool(r.vulnerable_known),
                      "heat": round(r.heat_fraction, 3), "ammo": int(r.ammo),
                      "pos_ok": bool(r.position_valid)},
            "nav": {"ok": bool(snapshot.nav.planner_ok),
                    "fails": int(snapshot.nav.consecutive_failures)},
            "match": {"left_s": round(snapshot.match_time_left_s, 2)},
            "zones": {zid: {"owner": zs.owner.name, "age": round(zs.age_s, 1)}
                      for zid, zs in snapshot.zones.items()},
            "enemies": [{"x": round(e.x_m, 2), "y": round(e.y_m, 2),
                         "age": round(e.age_s, 2)}
                        for e in snapshot.enemies],
            "req": {"action": decision.action.name, "point_id": decision.point_id,
                    "mode": state.mode, "reason": decision.reason},
            "exec": {"vx": round(float(exec_out.move[0]), 3),
                     "vy": round(float(exec_out.move[1]), 3),
                     "target": int(exec_out.target), "fire": bool(exec_out.fire)},
            "overrides": _overrides(state, decision),
        })

    def log_transition(self, env_idx: int, tr: Any, *, t: float, frame: int) -> None:
        if not self.wants(env_idx):
            return
        self._write({
            "type": "transition", "env": env_idx, "t": round(t, 3), "frame": frame,
            "from": tr.previous.name, "to": tr.current.name,
            "task_elapsed_s": round(tr.task_elapsed_s, 2), "reason": tr.reason,
        })
