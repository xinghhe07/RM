"""尸检：对比"崩盘前"与"崩盘后"检查点的策略行为，定位崩盘机制。

背景
----
受控诊断显示一个高度可复现的现象：
    upd 10~15 评估胜率 0.81 / 净胜点 +26（最好）
    upd 20    评估胜率 0.04 / 净胜点 -160（崩）
之后再也不恢复。梯度范数一直很小（0.4~4），所以不是梯度爆炸。

本脚本加载崩前/崩后两个检查点，在**同一批固定种子**的环境上跑确定性对局，
逐项对比行为差异，回答：
    崩后策略"不占点"是因为
      (a) 它不动了？           -> 看移动指令幅值
      (b) 它乱动但不去控制区？  -> 看位置轨迹与到中心距离
      (c) 它不敢开火？          -> 看开火率
      (d) 它早死/到处挨打？      -> 看存活时间与承受伤害
      (e) 它选了"不交战"？      -> 看目标选择分布

用法
----
    python probe_policy.py --ckpt runs/diag/baseline_upd010.pt runs/diag/baseline_upd020.pt
    python probe_policy.py --auto            # 自动挑崩前(最好)与最后两个检查点
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S
from rm_sentry.train_ppo import ActorCritic

ROOT = Path(__file__).resolve().parent
EVAL_SEED = 20261007


def probe(ckpt_path: Path, season: str, n_envs: int, max_steps: int, dev: str) -> dict:
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = RMUL3v3Config(season=season)
    env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=torch.device(dev), seed=ck["upd"] * 7 + 1)
    net = ActorCritic(env.obs_dim, cfg.n_enemies + 1, ck["cfg"].get("hidden", 256)).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    obs = env.reset()
    mask = env.action_mask()

    # 统计容器
    mv_mag, tgts, fires, holds, alive_t, dmg_in = [], [], [], [], [], []
    dist_from_center, vp_margin = [], []
    n_actions = env.n_actions
    steps_survived = torch.zeros(n_envs, device=dev)
    max_survived = torch.zeros(n_envs, device=dev)

    for t in range(max_steps):
        # 确定性动作：连续头取均值，离散头取 argmax，开火取 logit>0
        # （不能用 net.act()，它只有随机采样；确定性评估要用 forward 直接读分布参数）
        with torch.no_grad():
            mu, logstd, tlog, flog, val = net.forward(obs, mask)
            move = mu.clamp(-1.0, 1.0)
            target = tlog.argmax(dim=-1)
            fire = (flog > 0.0).float()
        pa = {"move": move, "target": target, "fire": fire}

        # 移动指令幅值（连续头，正常应是 [-1,1]）
        mv = pa["move"].abs().mean(dim=-1)
        mv_mag.append(float(mv.mean()))
        tgts.append(pa["target"].float())
        fires.append(float(pa["fire"].mean()))

        # 到控制区中心的距离（控制区 = 场地中心，半径 center_r）—— 与 env 内部一致
        cx, cy = cfg.field_w / 2.0, cfg.field_h / 2.0
        d = torch.sqrt((env.sentry[:, S.X] - cx) ** 2 + (env.sentry[:, S.Y] - cy) ** 2)
        dist_from_center.append(float(d.mean()))
        holds.append(float((d <= cfg.center_r).float().mean()))
        vp_margin.append(float((env.sentry[:, S.VP] - env.sentry[:, S.VP_ENEMY]).mean()))

        a = torch.zeros(n_envs, n_actions, device=dev)
        a[:, 0] = pa["move"][:, 0]
        a[:, 1] = pa["move"][:, 1]
        a[:, 2] = pa["target"]
        a[:, 6] = pa["fire"]
        nobs, rew, done, info = env.step(a)

        dmg_in.append(float(env.sentry[:, S.DMG_IN].mean()))
        alive = (env.sentry[:, S.DEAD] <= 0).float()
        steps_survived = (steps_survived + alive) * alive   # 死亡即归零
        max_survived = torch.maximum(max_survived, steps_survived)

        obs = nobs
        env.auto_reset(done)
        mask = env.action_mask()

    tg = torch.stack(tgts)  # [T, N]
    n_target = tg.shape[-1]
    tg_hist = [(tg == i).float().mean().item() for i in range(n_target)]
    return dict(
        upd=ck["upd"], ckpt=ckpt_path.name,
        eval=ck.get("eval", {}),
        move_mag=float(np.mean(mv_mag)),
        fire=float(np.mean(fires)),
        hold=float(np.mean(holds)),
        dist_center=float(np.mean(dist_from_center)),
        vp_margin=float(np.mean(vp_margin)),
        dmg_in=float(np.mean(dmg_in)),
        survive_s=float(max_survived.mean()) * cfg.dt,
        target_hist=[round(x, 3) for x in tg_hist],
    )


def show(r: dict) -> None:
    ev = r["eval"] or {}
    print(f"\n{'='*72}")
    print(f"检查点 {r['ckpt']}  (upd {r['upd']})")
    print(f"{'='*72}")
    if ev:
        print(f"  评估: 胜率 {ev.get('win_rate', float('nan')):.2f}  "
              f"败率 {ev.get('lose_rate', float('nan')):.2f}  "
              f"净胜点 {ev.get('vp_margin', float('nan')):+.1f}")
    print(f"  行为指纹:")
    print(f"    移动指令幅值       {r['move_mag']:.3f}   (0=完全不动, 1=满速)")
    print(f"    开火率             {r['fire']:.3f}")
    print(f"    到场地中心平均距离  {r['dist_center']:.2f} m")
    print(f"    在控制区内时间占比  {r['hold']:.3f}")
    print(f"    净胜点(过程平均)    {r['vp_margin']:+.1f}")
    print(f"    承受伤害(累计)      {r['dmg_in']:.0f}")
    print(f"    平均存活时长        {r['survive_s']:.1f} s")
    print(f"    目标选择分布        {r['target_hist']}  [0=不交战, 1..3=敌机]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=None)
    ap.add_argument("--glob", default="runs/diag/baseline_upd*.pt")
    ap.add_argument("--auto", action="store_true", help="自动选崩前(胜率最高)与最后两个")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.ckpt:
        paths = [Path(p) for p in args.ckpt]
    else:
        paths = [Path(p) for p in sorted(glob.glob(str(ROOT / args.glob)))]
    if not paths:
        print(f"没找到检查点。确认跑过带存档的训练（{args.glob}）")
        return 1

    # --auto：加载每个检查点的 eval，挑胜率最高的 + 最后两个
    if args.auto:
        scored = []
        for p in paths:
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                wr = (ck.get("eval") or {}).get("win_rate", -1)
                scored.append((wr, p))
            except Exception:
                continue
        if scored:
            scored.sort(key=lambda x: -x[0])
            best = scored[0][1]
            last2 = [p for _, p in scored[-2:]] if len(scored) >= 2 else []
            picked = []
            for p in [best] + last2:
                if p not in picked:
                    picked.append(p)
            paths = picked
            print(f"--auto 选中: {[p.name for p in paths]}")

    results = [probe(p, args.season, args.envs, args.steps, args.device) for p in paths]
    for r in results:
        show(r)

    if len(results) >= 2:
        a, b = results[0], results[-1]
        print(f"\n{'='*72}")
        print(f"差异（{a['ckpt']} -> {b['ckpt']}）")
        print(f"{'='*72}")
        rows = [
            ("胜率", (a["eval"] or {}).get("win_rate"), (b["eval"] or {}).get("win_rate"), "{:.2f}"),
            ("净胜点", (a["eval"] or {}).get("vp_margin"), (b["eval"] or {}).get("vp_margin"), "{:+.1f}"),
            ("移动指令幅值", a["move_mag"], b["move_mag"], "{:.3f}"),
            ("开火率", a["fire"], b["fire"], "{:.3f}"),
            ("到中心距离", a["dist_center"], b["dist_center"], "{:.2f}"),
            ("占点时间占比", a["hold"], b["hold"], "{:.3f}"),
            ("承受伤害", a["dmg_in"], b["dmg_in"], "{:.0f}"),
            ("存活时长", a["survive_s"], b["survive_s"], "{:.1f}"),
        ]
        for name, x, y, fmt in rows:
            if x is None or y is None:
                continue
            print(f"  {name:14s} {fmt.format(x):>10s} -> {fmt.format(y):>10s}")
        print(f"\n  目标选择分布   {a['target_hist']} -> {b['target_hist']}")
        print("\n解读要点：")
        print("  * 移动幅值掉到接近 0  -> 策略学会'不动'（可能被奖励/观测坑了）")
        print("  * 开火率掉到接近 0    -> 过热锁死或学到不开火")
        print("  * 到中心距离变大      -> 主动远离控制区（脱离了目标）")
        print("  * 存活时长大幅变短    -> 学成了送死")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
