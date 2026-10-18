"""战术层（V1 规则决策树 + 状态机）评测：与规则基线同口径、同种子、同环境。

⚠️ **本脚本的数字不是成绩**。当前环境的场地/车体参数仍是待替换的编造值
   （交接文档 §4），所以：
     - 可以比：**同一次运行内**战术层 vs 规则基线（同种子、同对手、同记账）——
       用来回答"新架构有没有把旧行为搞坏"
     - 不可以比：与任何历史数字、任何对外宣称 —— 场地一换全部作废

记账口径**逐行照抄 `rule_baseline.evaluate_rule`**（不抽公共函数是为了不惊动
已验证的规则基线路径；两边都改过一次就一定会漂，因此这里加了交叉校验：
同一次运行里把规则基线也跑一遍，看数字是否与 `eval_baselines.py` 一致）。

用法：
    python eval_tactical.py                 # 战术层 + 规则基线（同种子对照）
    python eval_tactical.py --envs 16 --max-steps 4000
"""

from __future__ import annotations

import argparse

import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S
from rm_sentry.rule_baseline import RuleSentry, evaluate_rule
from rm_sentry.tactical import TacticalDriver
from rm_sentry.train_ppo import to_env_action

EVAL_SEED = 20261007     # 与 eval_baselines.py 一致，跨实验可比


def evaluate_tactical(cfg, dev, *, n_envs: int = 16, seed: int = EVAL_SEED,
                      max_steps: int | None = None) -> dict:
    """用与 `evaluate_rule` **完全相同**的统计定义评估战术层。

    额外多记一项 `survive_duty`（存活占空比）—— 它是当前唯一的阻塞项
    （`工作台.md` §四：哨兵 69% 时间是死的），战术层必须能改善它。
    两个来源交叉校验：逐步 `DEAD<=0` 与每局临终的 `S.DEATHS`。
    """
    env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=dev, seed=seed)
    env.reset()
    drv = TacticalDriver(env)
    done_n = 0
    acc = {"vp": 0.0, "vp_enemy": 0.0, "dmg_out": 0.0, "dmg_in": 0.0,
           "hold": 0.0, "win": 0.0, "lose": 0.0}
    alive_steps = 0
    deaths_total = 0.0
    steps = 0
    limit = max_steps if max_steps is not None else int(cfg.match_time / cfg.dt) * 2

    for _ in range(limit):
        a = drv.act()
        obs, rew, done, info = env.step(a)
        acc["hold"] += float(info["hold_ratio"].sum())
        alive_steps += int((env.sentry[:, S.DEAD] <= 0).sum())
        steps += 1
        if bool(done.any()):
            k = done
            for key, ik in (("vp", "vp"), ("vp_enemy", "vp_enemy"),
                            ("dmg_out", "dmg_out"), ("dmg_in", "dmg_in"),
                            ("win", "win"), ("lose", "lose")):
                acc[key] += float(info[ik][k].sum())
            deaths_total += float(env.sentry[k, S.DEATHS].sum())
            done_n += int(k.sum())
            env.auto_reset(done)
            drv.reset()                      # 跨步状态随新一局一起重置（AGENTS.md R2）
        if done_n >= n_envs:
            break

    d = max(done_n, 1)
    return {
        "episodes": done_n,
        "vp": acc["vp"] / d, "vp_enemy": acc["vp_enemy"] / d,
        "dmg_out": acc["dmg_out"] / d, "dmg_in": acc["dmg_in"] / d,
        "win_rate": acc["win"] / d, "lose_rate": acc["lose"] / d,
        "hold_rate": acc["hold"] / max(steps * n_envs, 1),
        "vp_margin": (acc["vp"] - acc["vp_enemy"]) / d,
        "survive_duty": alive_steps / max(steps * n_envs, 1),
        "deaths_per_episode": deaths_total / d,
    }


def fmt(tag: str, r: dict) -> str:
    extra = ""
    if "survive_duty" in r:
        extra = (f"  存活占空比 {r['survive_duty']:6.3f}  "
                 f"场均战亡 {r['deaths_per_episode']:5.1f}")
    return (f"  {tag:20s} 胜率 {r['win_rate']:5.2f}  败率 {r['lose_rate']:5.2f}  "
            f"净胜点 {r['vp_margin']:+7.1f}  占点率 {r['hold_rate']:7.4f}  "
            f"输出 {r['dmg_out']:6.0f}  承受 {r['dmg_in']:6.0f}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-rule", action="store_true", help="只跑战术层")
    args = ap.parse_args()

    cfg = RMUL3v3Config(season=args.season)
    print("=" * 104)
    print(f"战术层评测  season={args.season}  n_envs={args.envs}  EVAL_SEED={EVAL_SEED}"
          f"（⚠️ 环境参数仍是编造值 -> 只做同次运行内的对照，不作成绩）")
    print("=" * 104)

    r_tac = evaluate_tactical(cfg, args.device, n_envs=args.envs,
                              max_steps=args.max_steps)
    print(fmt("战术层(V1)", r_tac))

    if not args.skip_rule:
        rule = RuleSentry(cfg)
        r_rule = evaluate_rule(cfg, args.device, rule, n_envs=args.envs,
                               seed=EVAL_SEED, max_steps=args.max_steps)
        print(fmt("规则基线(旧)", r_rule))

        print("\n" + "-" * 104)
        print("同次运行内对照（看的是『新架构有没有把旧行为搞坏』，不是成绩）")
        print("-" * 104)
        for key, label in (("vp_margin", "净胜点"), ("hold_rate", "占点率"),
                           ("win_rate", "胜率")):
            diff = r_tac[key] - r_rule[key]
            print(f"  {label:8s} 战术层 {r_tac[key]:+8.3f}  规则基线 {r_rule[key]:+8.3f}"
                  f"  差 {diff:+8.3f}")
        print(f"  {'存活占空比':8s} 战术层 {r_tac['survive_duty']:8.3f}  "
              f"（规则基线的同一指标由 evaluate_rule 未采集，需另测）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
