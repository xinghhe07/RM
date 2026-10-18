"""受控诊断：定位"策略训练中期塌缩"的成因。

症状（已观测）
--------------
训练到 upd 40~50 左右，**累计伤害从 ~530 掉到个位数**、开火率 0.6 → 0.42。
梯度范数与 KL 都受控（说明不是梯度爆炸），所以是另一个独立机制。

诊断设计（严格单变量）
--------------------
固定：训练种子、**评估种子**（同一评估集才能横向比较）、环境、网络结构。
变量：一次只改一个（学习率 / rollout 长度 / 熵系数 / 优势归一化 / 是否用对手池）。
记录：每次更新的 loss/KL/梯度/奖励/开火率/占点率 + 每次评估的确定性对局结果。
落盘：JSON，便于逐配置对比。

用法
----
    python diag_collapse.py --config baseline
    python diag_collapse.py --config low_lr
    python diag_collapse.py --all                 # 一次跑完全部配置并对比
    python diag_collapse.py --config baseline --updates 80
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rm_sentry.guards import HealthMonitor
from rm_sentry.reward_scaling import RewardScaler
from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S
from rm_sentry.train_ppo import ActorCritic, RollingEpisodeStats, TrainCfg, to_env_action

OUT = Path(__file__).resolve().parent / "runs" / "diag"
EVAL_SEED = 20261007        # 固定评估种子 —— 所有配置共用同一评估集


# --------------------------------------------------------------------------
# 配置矩阵：每次只改一个变量
# --------------------------------------------------------------------------
def configs() -> dict[str, TrainCfg]:
    base = TrainCfg(n_envs=512, rollout=64, epochs=4, minibatches=8,
                    lr=3e-4, ent_coef=0.01, target_kl=0.03,
                    per_minibatch_adv_norm=True, seed=0, hidden=256)
    return {
        # 基线：当前默认奖励权重（w_dmg_out=0.003，占点主导）
        "baseline": base,
        # 变量 1：每轮只跑 1 个 epoch。优势是 rollout 时算的，只对第 1 个 epoch 真正有效；
        #         跑满 4 个 epoch 等于在 off-policy 数据上多做 3 倍更新。
        "one_epoch": replace(base, epochs=1),
        # 变量 2：学习率降 3 倍
        "low_lr": replace(base, lr=1e-4),
        # 变量 3：rollout 拉长 2 倍（优势估计噪声更小）
        "long_rollout": replace(base, rollout=128),
        # 变量 4：熵系数提高（防止探索塌缩）
        "high_ent": replace(base, ent_coef=0.03),
        # 变量 5：关掉 per-minibatch 优势归一化（看它是否有副作用）
        "no_permb_norm": replace(base, per_minibatch_adv_norm=False),
    }


# --------------------------------------------------------------------------
def run_one(name: str, tc_in: TrainCfg, updates: int, dev: str,
            eval_envs: int, eval_max_steps: int, season: str) -> dict:
    dev_t = torch.device(dev)
    tc = replace(tc_in, n_envs=tc_in.n_envs, rollout=tc_in.rollout)
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    cfg = RMUL3v3Config(season=season)
    env = BatchedRMUL3v3(cfg, n_envs=tc.n_envs, device=dev_t, seed=tc.seed)
    obs = env.reset()
    mask = env.action_mask()
    net = ActorCritic(env.obs_dim, cfg.n_enemies + 1, tc.hidden).to(dev_t)
    opt = torch.optim.Adam(net.parameters(), lr=tc.lr, eps=1e-5)

    n_actions = env.n_actions
    batch = tc.n_envs * tc.rollout
    scaler = RewardScaler()
    hm = HealthMonitor(window=20)
    roll = RollingEpisodeStats(horizon=60)

    print(f"\n{'='*74}")
    print(f"配置 {name}: rollout={tc.rollout} lr={tc.lr:g} epochs={tc.epochs} "
          f"minibatch={tc.minibatches} ent={tc.ent_coef} "
          f"permb_norm={tc.per_minibatch_adv_norm} target_kl={tc.target_kl}")
    print(f"{'='*74}")

    hist: list[dict] = []
    # 训练前的基线评估（未训练网络）
    ev0 = evaluate(net, cfg, dev_t, eval_envs, EVAL_SEED, eval_max_steps)
    print(f"  [评估 upd  0] 胜率 {ev0['win_rate']:.2f} 败率 {ev0['lose_rate']:.2f} "
          f"净胜点 {ev0['vp_margin']:+.1f} 场均输出 {ev0['dmg_out']:.0f}")

    t0 = time.perf_counter()
    for upd in range(1, updates + 1):
        sd = {"dmg": 0.0, "fire": 0.0, "hold": 0.0, "rew": 0.0, "n": 0}
        ob, mb, mvb, tgb, frb, lpb, vb, rb, db = ([] for _ in range(9))

        for _ in range(tc.rollout):
            with torch.no_grad():
                pa, _ = net.act(obs, mask)
            a = to_env_action(pa, n_actions, dev_t)
            for buf, v in ((ob, obs), (mb, mask), (mvb, pa["move"]), (tgb, pa["target"]),
                           (frb, pa["fire"]), (lpb, pa["logp"]), (vb, pa["value"])):
                buf.append(v)
            nobs, rew, done, info = env.step(a)
            rb.append(rew); db.append(done.float())
            sd["fire"] += float(pa["fire"].mean())
            sd["hold"] += float(info["hold_ratio"].mean())
            sd["rew"] += float(rew.mean())
            sd["dmg"] = float(env.sentry[:, S.DMG_OUT].mean())   # 累计量，直接取当前值
            sd["n"] += 1
            if bool(done.any()):
                roll.update(done, info)
            obs = nobs
            env.auto_reset(done)
            mask = env.action_mask()

        with torch.no_grad():
            _, _, _, _, last_v = net.forward(obs, mask)
        O = torch.stack(ob); M = torch.stack(mb)
        MV = torch.stack(mvb); TG = torch.stack(tgb); FR = torch.stack(frb)
        LP = torch.stack(lpb); V = torch.stack(vb)
        R = torch.stack(rb); D = torch.stack(db)

        scaler.update_discounted(R)
        Rn = scaler.scale(R)

        adv = torch.zeros_like(R)
        last = torch.zeros(tc.n_envs, device=dev_t)
        for t in reversed(range(tc.rollout)):
            nt = 1.0 - D[t]
            nv = last_v if t == tc.rollout - 1 else V[t + 1]
            delta = Rn[t] + tc.gamma * nv * nt - V[t]
            last = delta + tc.gamma * tc.lam * nt * last
            adv[t] = last
        ret = adv + V

        f = lambda x: x.reshape(-1, *x.shape[2:])           # noqa: E731
        O, M, MV, TG, FR, LP, V, ADV, RET = map(f, (O, M, MV, TG, FR, LP, V, adv, ret))
        if not tc.per_minibatch_adv_norm:
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        idx = torch.randperm(batch, device=dev_t)
        mb_size = batch // tc.minibatches
        losses, kls, gns, ent_h = [], [], [], []
        epoch_kls = []
        epochs_run = 0
        for ep in range(tc.epochs):
            ep_kl = []
            for st in range(0, batch, mb_size):
                j = idx[st:st + mb_size]
                newlp, ent, newv = net.evaluate(O[j], M[j], MV[j], TG[j], FR[j])
                a_ = ADV[j]
                if tc.per_minibatch_adv_norm:
                    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                ratio = (newlp - LP[j]).exp()
                pg = -torch.min(ratio * a_,
                                ratio.clamp(1 - tc.clip, 1 + tc.clip) * a_).mean()
                vf = 0.5 * (newv - RET[j]).pow(2).mean()
                loss = pg + tc.vf_coef * vf - tc.ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gns.append(float(nn.utils.clip_grad_norm_(net.parameters(), tc.max_grad_norm)))
                opt.step()
                with torch.no_grad():
                    # k3 估计器（无偏低方差），见 Schulman 的 KL 近似博客
                    k3 = float(((ratio - 1.0) - (newlp - LP[j]).log()).mean())
                    ep_kl.append(k3)
                    kls.append(float((LP[j] - newlp).mean()))
                    losses.append(float(loss))
                    ent_h.append(float(ent.mean()))
            epochs_run = ep + 1
            epoch_kls.append(float(np.mean(ep_kl)))
            # ---- KL 早停：**按 epoch 检查**（SB3 的做法）----
            # 之前写成"滑动平均 + 每 minibatch 检查"，有两个问题：
            #   1) 优势是在 rollout 时算的，只对第 1 个 epoch 有效；
            #      跑满 4 个 epoch 等于在 off-policy 数据上多做了 3 倍更新。
            #   2) 按 minibatch 检查会被瞬时波动掩盖真实漂移。
            if tc.target_kl and epoch_kls[-1] > tc.target_kl:
                break

        nd = max(sd["n"], 1)
        row = dict(upd=upd, loss=float(np.mean(losses)), kl=float(np.mean(kls)),
                   epoch_kls=epoch_kls, epochs_run=epochs_run,
                   grad=float(np.mean(gns)), ent=float(np.mean(ent_h)),
                   r_step=sd["rew"] / nd, fire=sd["fire"] / nd,
                   hold=sd["hold"] / nd, dmg=sd["dmg"],
                   r_std=scaler.std, v_mean=float(V.mean()), ret_std=float(RET.std()))

        # 周期性确定性评估（真实对局，不看训练奖励）
        # 评估很密（每 5 次更新）—— 因为崩盘发生在 upd 10~20 之间，
        # 必须抓住崩盘前的那个检查点才能做尸检。
        if upd % 5 == 0 or upd == updates:
            ev = evaluate(net, cfg, dev_t, eval_envs, EVAL_SEED, eval_max_steps)
            row["eval"] = ev
            # 存档：崩盘前后各存一个，便于事后对比"崩前 vs 崩后"的策略行为
            torch.save({"upd": upd, "model": net.state_dict(),
                        "cfg": asdict(tc), "eval": ev},
                       OUT / f"{name}_upd{upd:03d}.pt")
            print(f"  upd {upd:3d} | 伤害 {row['dmg']:6.0f} 开火 {row['fire']:.2f} "
                  f"占点 {row['hold']:.3f} | |g| {row['grad']:6.1f} KL {row['kl']:+.3f} "
                  f"熵 {row['ent']:.2f} | 评估 胜率 {ev['win_rate']:.2f} "
                  f"净胜点 {ev['vp_margin']:+.1f}")
        else:
            print(f"  upd {upd:3d} | 伤害 {row['dmg']:6.0f} 开火 {row['fire']:.2f} "
                  f"占点 {row['hold']:.3f} | |g| {row['grad']:6.1f} KL {row['kl']:+.3f}")

        for msg in hm.observe(loss=row["loss"], kl=row["kl"], grad_norm=row["grad"],
                              reward=row["r_step"], fire_rate=row["fire"],
                              hold_rate=row["hold"], entropy=row["ent"]):
            print(f"     \033[31m[护栏]\033[0m {msg}")
        hist.append(row)

    wall = time.perf_counter() - t0
    result = dict(name=name, cfg=asdict(tc), season=season, updates=updates,
                  wall_s=round(wall, 1), eval_seed=EVAL_SEED, hist=hist,
                  eval_before=ev0)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  完成 {updates} 次更新，用时 {wall/60:.1f} 分钟 -> {p}")
    return result


def evaluate(net, cfg, dev, n_envs: int, seed: int, max_steps: int) -> dict:
    """确定性评估：固定种子，报告真实对局指标。"""
    from rm_sentry.train_ppo import evaluate as _eval
    return _eval(net, cfg, dev, n_envs=n_envs, seed=seed, max_steps=max_steps)


def summarize(results: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("对比汇总（关键看最后一行：伤害是否被保住）")
    print("=" * 88)
    hdr = f"{'配置':16s} {'起始伤害':>8s} {'峰值伤害':>8s} {'末期伤害':>8s} {'末期胜率':>8s} {'净胜点':>8s} {'塌缩?':>6s}"
    print(hdr)
    print("-" * 88)
    for r in results:
        h = r["hist"]
        if not h:
            continue
        dmgs = [x["dmg"] for x in h]
        peak = max(dmgs)
        last = dmgs[-1]
        evs = [x["eval"] for x in h if "eval" in x]
        last_ev = evs[-1] if evs else r["eval_before"]
        # 塌缩判定：末期伤害低于峰值的 40%，且峰值足够大（排除没学起来的）
        collapsed = peak > 50 and last < 0.4 * peak
        print(f"{r['name']:16s} {dmgs[0]:8.0f} {peak:8.0f} {last:8.0f} "
              f"{last_ev['win_rate']:8.2f} {last_ev['vp_margin']:+8.1f} "
              f"{'是' if collapsed else '否':>6s}")
    print("=" * 88)
    print("\n解读：")
    print("  * 塌缩 = 峰值伤害足够大(>50) 但末期掉到 40% 以下；说明策略先学到东西又退化")
    print("  * 哪个配置「否」，那个变量就是解 —— 再单独验证一次它是否稳定")
    print("  * 所有配置都「是」 → 病根不在这些超参，要转向环境/奖励语义")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--eval-envs", type=int, default=128)
    ap.add_argument("--eval-max-steps", type=int, default=6000,
                    help="评估每局最多步数（默认 6000 = 一局时长；调小更省时）")
    args = ap.parse_args()

    allcfg = configs()
    names = list(allcfg) if args.all else [args.config]
    for n in names:
        if n not in allcfg:
            print(f"未知配置 {n}，可选: {list(allcfg)}")
            return 2

    results = []
    for n in names:
        results.append(run_one(n, allcfg[n], args.updates, args.device,
                               args.eval_envs, args.eval_max_steps, args.season))
    if len(results) > 1:
        summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
