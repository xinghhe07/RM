"""对照基线评测：随机 / 规则 / RL 检查点，全部用同一 EVAL_SEED。

为什么必须有这个脚本
-------------------
AGENTS.md 铁律 R6：没有对照基线 = 不知道在进步还是在原地打转。
本脚本给出三个数字，之后所有"改进了"的说法都必须与它们对比：
    1. 随机策略        —— 下界（不该有策略比它差）
    2. 规则基线        —— **真正的对照**（人类常识水平）
    3. RL 检查点       —— 待评估对象

统一口径（否则不可比）：
    - 同一环境配置、同一 EVAL_SEED = 20261007
    - 同一批指标：胜率 / 败率 / 净胜点 / 占点率 / 场均输出 / 承受伤害
    - 同一对手（环境内置脚本 AI）

用法：
    python eval_baselines.py                      # 随机 + 规则
    python eval_baselines.py --ckpt runs/diag/baseline_upd010.pt
    python eval_baselines.py --ckpt runs/diag/baseline_upd010.pt --ckpt runs/diag/baseline_upd025.pt
    python eval_baselines.py --rule-tune          # 扫规则基线的超参
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rm_sentry.rmul_3v3_env import RMUL3v3Config
from rm_sentry.rule_baseline import RuleSentry, evaluate_rule
from rm_sentry.train_ppo import ActorCritic, evaluate

EVAL_SEED = 20261007          # 所有评估共用，跨实验可比
OUT = Path(__file__).resolve().parent / "runs" / "baseline_eval"


def fmt(tag: str, r: dict) -> str:
    return (f"  {tag:22s} 胜率 {r['win_rate']:5.2f}  败率 {r['lose_rate']:5.2f}  "
            f"净胜点 {r['vp_margin']:+7.1f}  占点率 {r.get('hold_rate', float('nan')):7.4f}  "
            f"输出 {r['dmg_out']:6.0f}  承受 {r['dmg_in']:6.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--ckpt", action="append", default=None)
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--rule-tune", action="store_true", help="扫规则基线的超参")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    cfg = RMUL3v3Config(season=args.season)

    print("=" * 100)
    print(f"对照基线评测  season={args.season}  n_envs={args.envs}  EVAL_SEED={EVAL_SEED}")
    print("=" * 100)

    rows = {}

    if not args.skip_random:
        # 随机策略：用 evaluate(random_policy=True)，与其它项同环境同种子
        import types
        net = ActorCritic(36, cfg.n_enemies + 1, 64).to(dev)
        r = evaluate(net, cfg, dev, n_envs=args.envs, seed=EVAL_SEED,
                     random_policy=True, max_steps=args.max_steps)
        rows["random"] = r
        print(fmt("随机策略(下界)", r))

    # 规则基线
    if args.rule_tune:
        print("\n  --- 规则基线超参扫描 ---")
        best = None
        for heat in (0.7, 0.85, 0.95):
            for dead in (0.3, 0.5, 1.0):
                rule = RuleSentry(cfg, heat_stop_ratio=heat, zone_deadband=dead)
                r = evaluate_rule(cfg, dev, rule, n_envs=args.envs,
                                  seed=EVAL_SEED, max_steps=args.max_steps)
                print(fmt(f"heat={heat} dead={dead}", r))
                if best is None or r["vp_margin"] > best[1]["vp_margin"]:
                    best = (f"heat={heat} dead={dead}", r)
        print(f"\n  最佳规则配置: {best[0]}")
        rows["rule_best"] = best[1]
    else:
        rule = RuleSentry(cfg)
        r = evaluate_rule(cfg, dev, rule, n_envs=args.envs,
                          seed=EVAL_SEED, max_steps=args.max_steps)
        rows["rule"] = r
        print(fmt("规则基线", r))

    # RL 检查点
    for p in (args.ckpt or []):
        path = Path(p)
        if not path.exists():
            print(f"  [跳过] 找不到 {path}")
            continue
        ck = torch.load(path, map_location=dev, weights_only=False)
        net = ActorCritic(36, cfg.n_enemies + 1, ck["cfg"].get("hidden", 256)).to(dev)
        net.load_state_dict(ck["model"])
        net.eval()
        r = evaluate(net, cfg, dev, n_envs=args.envs, seed=EVAL_SEED,
                     max_steps=args.max_steps)
        rows[f"rl:{path.stem}"] = r
        print(fmt(f"RL {path.stem}", r))

    print("\n" + "=" * 100)
    print("判定")
    print("=" * 100)
    if "rule" in rows or "rule_best" in rows:
        rr = rows.get("rule") or rows.get("rule_best")
        base = rr["vp_margin"]
        print(f"  规则基线净胜点 = {base:+.1f}（这是 RL 必须超过的门槛）")
        for k, v in rows.items():
            if k.startswith("rl:"):
                d = v["vp_margin"] - base
                verdict = "超过基线 ✅" if d > 0 else "未超过基线 ❌"
                print(f"    {k:34s} 差 {d:+7.1f}  {verdict}")
    else:
        print("  （本次未跑规则基线）")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baselines.json").write_text(
        json.dumps({"eval_seed": EVAL_SEED, "season": args.season,
                    "n_envs": args.envs, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  已保存 -> {OUT / 'baselines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
