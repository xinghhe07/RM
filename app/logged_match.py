"""A2 验收：跑一局带完整决策日志的比赛，并**立即回读验证**（"每个决策可复盘"）。

用法：
    cd app
    python logged_match.py                    # 默认 2026（战术层已完整适配的赛季）
    python logged_match.py --envs 2 --season 2027

产出：runs/logs/match_<season>_<stamp>.jsonl（一行一事件：header/decision/transition）
回读：动作分布 / 状态迁移 / 时间单调自检 / 抽样打印决策全文。

日志口径（A2.1）：**每环境只记录第一局**，某环境 `done` 后即停记（`log_scope` 写进
header）。这样 `(env, frame)` 才能唯一确定一条决策；跨局聚合请在后续版本显式加
`episode` 字段，而不是把多局混进同一个 `env` 序列。

✅ 2027 说明（**A1.5 已于 2026-10-16 完成**）：战术层的点图/适配器已迁移到官方三控制区
   **矩形**（中央 3000×3000 + 两高地 1200×1200，坐标见 `docs/2027场地规格.md`）。
   决策日志里出现的是真实区域名与真实点位（如 `P_HIGH_R`），不再是"中心圆 0.8 m"代理。

⚠️ 但 2027 本局**仍不是成绩**，剩余占位项（都在环境侧）：
   1. 脚本对手仍只打中央区 -> 哨兵去红方高地时全程零交战、零战亡，评估失真；
   2. 环境无高地/隧道**实体**（`covers=()`，属 B1 场地模型任务）；
   3. 运动学仍是占位 `v_max=3.0`（车体尺寸未定稿）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config
from rm_sentry.tactical import TacticalDriver
from rm_sentry.tactical.logger import DecisionLogger

SEED = 20261016
LOG_DIR = Path("runs/logs")


def drive(env, drv, logger, *, max_steps: int | None = None, on_step=None) -> int:
    """驱动整局；返回结束次数。`on_step` 仅供测试构造"提前结束"。

    A2.1：某环境 `done` 后**先注销日志、再 auto_reset**。否则 `frame` 归零，
    该环境后续记录的 `t` 会回跳，日志既不能忠实回放也不能按局聚合。
    口径：一次运行 = 每环境只记录第一局。
    """
    done_n = 0
    limit = (max_steps if max_steps is not None
             else int(env.cfg.match_time / env.cfg.dt) * 2)
    for k in range(limit):
        if on_step is not None:
            on_step(env, k)
        a = drv.act()
        _obs, _rew, done, _info = env.step(a)
        if bool(done.any()):
            done_idx = [int(i) for i in
                        done.nonzero(as_tuple=False).flatten().tolist()]
            for i in done_idx:
                logger.stop(i)               # 必须在 auto_reset 之前（A2.1）
            done_n += len(done_idx)
            env.auto_reset(done)
            drv.reset(done_idx)              # 只重置结束的环境（A2.2）
        if done_n >= env.n:
            break
    return done_n


def run(season: str, n_envs: int, device: str) -> Path:
    cfg = RMUL3v3Config(season=season)
    env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=device, seed=SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"match_{season}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    logger = DecisionLogger(path)
    logger.header(season=season, seed=SEED, config={
        "dt": cfg.dt, "decision_hz": cfg.decision_hz, "match_time": cfg.match_time,
        "vp_mode": cfg.vp_mode, "vp_init": cfg.vp_init,
        "respawn_base_len": cfg.respawn_base_len, "respawn_len_step": cfg.respawn_len_step,
        "respawn_invuln": cfg.respawn_invuln, "vuln_on_respawn": cfg.vuln_on_respawn,
        "zones": [z[0] for z in cfg.zones_rect] if cfg.zones_rect else ["中心(旧赛季)"],
        "profile_source": cfg.profile_source[:100],
        # 回放口径（A2.1）：每环境只记录第一局，done 后停记。
        "log_scope": "per_env_first_episode",
    })

    env.reset()
    drv = TacticalDriver(env, logger=logger)
    drive(env, drv, logger)
    logger.close()
    return path


def replay(path: Path) -> int:
    """回读验证：证明"每个决策可复盘"。返回非零 = 自检失败。"""
    n_dec = n_tr = 0
    actions: dict[str, int] = {}
    overrides: dict[str, int] = {}
    modes: dict[str, int] = {}
    t_last: dict[int, float] = {}
    monotonic_ok = True
    transitions: list[dict] = []
    first_dec: dict | None = None
    sample_exec_dev: dict | None = None    # req 与 exec 不一致的首例（门控证据）

    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "decision":
                n_dec += 1
                actions[rec["req"]["action"]] = actions.get(rec["req"]["action"], 0) + 1
                modes[rec["req"]["mode"]] = modes.get(rec["req"]["mode"], 0) + 1
                for ov in rec["overrides"]:
                    k = f"{ov['layer']}:{ov['state']}"
                    overrides[k] = overrides.get(k, 0) + 1
                e = rec["env"]
                if e in t_last and rec["t"] < t_last[e] - 1e-6:
                    monotonic_ok = False
                t_last[e] = rec["t"]
                if first_dec is None:
                    first_dec = rec
                if sample_exec_dev is None and rec["exec"]["fire"] is True:
                    sample_exec_dev = rec
            elif rec["type"] == "transition":
                n_tr += 1
                transitions.append(rec)

    print("=" * 84)
    print(f"回读 {path.name}")
    print(f"  决策记录 {n_dec} 条 ｜ 状态迁移 {n_tr} 条 ｜ 时间单调: "
          f"{'OK' if monotonic_ok else 'FAIL'}")
    print(f"  动作分布: {dict(sorted(actions.items(), key=lambda kv: -kv[1]))}")
    print(f"  模式分布: {dict(sorted(modes.items(), key=lambda kv: -kv[1]))}")
    if overrides:
        print(f"  压制统计: {overrides}")
    if transitions:
        print("  状态迁移抽样（前 6 条）：")
        for tr in transitions[:6]:
            print(f"    t={tr['t']:7.2f}s env{tr['env']}  {tr['from']:>10s} -> {tr['to']:<10s}"
                  f" ({tr['reason'][:56]})")
    if first_dec is not None:
        print("  首条决策全文（复盘的最小证据）：")
        print("    " + json.dumps(first_dec["req"], ensure_ascii=False)[:400])
    if sample_exec_dev is not None:
        print("  首条开火决策（req/exec 对照）：")
        print("    " + json.dumps({"req": sample_exec_dev["req"],
                                   "exec": sample_exec_dev["exec"]},
                                  ensure_ascii=False)[:300])
    ok = monotonic_ok and n_dec > 0 and n_tr > 0
    print(f"  自检: {'通过' if ok else '未通过'}")
    print("=" * 84)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.season == "2027":
        print("提示：2027 决策几何已用官方三控制区矩形（A1.5 已完成）；"
              "但对手只打中央 + 无高地/隧道实体 -> 本局不是成绩。")
    path = run(args.season, args.envs, args.device)
    print(f"日志已落盘: {path}")
    return replay(path)


if __name__ == "__main__":
    raise SystemExit(main())
