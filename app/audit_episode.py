"""审计对局时间线：搞清"净胜点怎么会掉到 -179"。

背景
----
2026 是扣分制：双方各从 200 胜利点开始，占控制区则**对方** −1/秒，
战亡则**自己** −20。而候选策略的评估净胜点长期在 −179 左右，几乎等于
"一方掉到 0 立即结束"的理论最差值。这需要解释：
    - 是"没占点被慢慢磨死"？
    - 还是"反复战亡被扣死"？
    - 对局在第几秒结束？

本脚本逐秒记录，直接回答上面三个问题。

用法
----
    python audit_episode.py                    # 随机策略
    python audit_episode.py --rule             # 规则基线（旧）
    python audit_episode.py --tactical         # 战术层（V1）
    python audit_episode.py --ckpt runs/diag/baseline_upd005.pt   # 指定检查点
    python audit_episode.py --no-enemy-fire    # 关掉对手火力做对照
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S
from rm_sentry.rule_baseline import RuleSentry
from rm_sentry.train_ppo import ActorCritic, to_env_action

ROOT = Path(__file__).resolve().parent


def _to_tensor(policy_act: dict, env, dev: str) -> torch.Tensor:
    """把策略动作字典拼成环境动作张量（与 train_ppo 共用同一映射，避免两处漂移）。"""
    return to_env_action(policy_act, env.n_actions, torch.device(dev))


def run(season: str, dev: str, ckpt: Path | None, n_envs: int,
        no_enemy_fire: bool, use_random: bool,
        use_rule: bool = False, use_tactical: bool = False) -> None:
    c = RMUL3v3Config(season=season)
    if no_enemy_fire:
        c.dmg_base = 0.0        # 对手打不出伤害（对照：隔离"竞争压力"这个因素）
    env = BatchedRMUL3v3(c, n_envs=n_envs, device=torch.device(dev), seed=99)
    rng = torch.Generator(device=dev).manual_seed(99)

    net = None
    if ckpt is not None:
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        net = ActorCritic(env.obs_dim, c.n_enemies + 1, ck["cfg"].get("hidden", 256)).to(dev)
        net.load_state_dict(ck["model"])
        net.eval()

    rule = RuleSentry(c) if use_rule else None
    drv = None
    if use_tactical:
        from rm_sentry.tactical import TacticalDriver
        drv = TacticalDriver(env)

    obs = env.reset()
    mask = env.action_mask()
    n = env.n

    if drv is not None:
        pol_name = "战术层(V1)"
    elif rule is not None:
        pol_name = "规则基线(旧)"
    elif use_random or net is None:
        pol_name = "随机"
    else:
        pol_name = ckpt.name
    print("=" * 78)
    print(f"对局审计  season={season}  n_envs={n}  对手火力={'关' if no_enemy_fire else '开'}"
          f"  策略={pol_name}")
    print("=" * 78)
    print(f"  {'秒':>5s} {'我方VP':>8s} {'对方VP':>8s} {'净':>8s} {'占点率':>7s} "
          f"{'已死次数':>8s} {'我方HP':>7s} {'到中心':>7s}")

    deaths_prev = torch.zeros(n, device=dev)
    died_at_first = None
    ended_at = None
    trace = []
    for t in range(env.max_steps):
        if drv is not None:
            a = drv.act()
        elif rule is not None:
            a = _to_tensor(rule.act(obs, mask, env), env, dev)
        elif net is None:
            # 动作布局是 [vx, vy, target, fire]（见 train_ppo.to_env_action）。
            # ⚠️ 这里曾把开火写到第 6 列（`a[:, 6]`）—— 而那列根本不是开火通道，
            #    于是"随机策略"全程**从未开火**，随机下界的数字一直是失真的。
            #    已改为统一走 `to_env_action`（与环境/训练同一套映射）。
            a = to_env_action({
                "move": torch.rand(n, 2, generator=rng, device=dev) * 2 - 1,
                "target": torch.randint(0, c.n_enemies + 1, (n,), generator=rng, device=dev),
                "fire": (torch.rand(n, generator=rng, device=dev) < 0.3).float(),
            }, env.n_actions, torch.device(dev))
        else:
            with torch.no_grad():
                mu, logstd, tlog, flog, val = net.forward(obs, mask)
                a = to_env_action({
                    "move": mu.clamp(-1, 1),
                    "target": tlog.argmax(dim=-1),
                    "fire": (flog > 0).float(),
                }, env.n_actions, torch.device(dev))
        obs, rew, done, info = env.step(a)

        cx, cy = c.field_w / 2, c.field_h / 2
        dist = torch.hypot(env.sentry[:, S.X] - cx, env.sentry[:, S.Y] - cy)
        vp, vpe = env.sentry[:, S.VP], env.sentry[:, S.VP_ENEMY]
        deaths = env.sentry[:, S.DEATHS]

        if t % 100 == 0 or t < 20:
            print(f"  {t*c.dt:5.1f} {vp.mean():8.1f} {vpe.mean():8.1f} "
                  f"{(vp-vpe).mean():+8.1f} {info['hold_ratio'].mean():7.3f} "
                  f"{deaths.mean():8.1f} {env.sentry[:, S.HP].mean():7.1f} {dist.mean():7.2f}")
        trace.append(dict(t=t * c.dt, vp=float(vp.mean()), vpe=float(vpe.mean()),
                          hold=float(info['hold_ratio'].mean()),
                          deaths=float(deaths.mean()), dist=float(dist.mean())))

        if died_at_first is None and bool(done.any()):
            ended_at = t * c.dt
            died_at_first = True
        deaths_prev = deaths
        env.auto_reset(done)
        if drv is not None:
            drv.reset()                  # 跨步状态随新一局一起重置（AGENTS.md R2）
        mask = env.action_mask()
        if bool(done.all()):
            break

    last = trace[-1]
    print()
    print("  ---- 结论 ----")
    print(f"  对局在 {last['t']:.1f} 秒时结束（上限 {env.max_steps*c.dt:.0f} 秒）")
    print(f"  结束时 我方VP={last['vp']:.1f} 对方VP={last['vpe']:.1f} 净={last['vp']-last['vpe']:+.1f}")
    print(f"  全程平均占点率 = {sum(x['hold'] for x in trace)/len(trace):.4f}")
    print(f"  累计战亡次数   = {last['deaths']:.2f}")
    print(f"  全程平均到中心距离 = {sum(x['dist'] for x in trace)/len(trace):.2f} m (控制区半径 {c.center_r})")
    print()
    print("  判定：")
    if last['t'] < env.max_steps * c.dt - 1:
        print(f"    * 对局**提前结束**（{last['t']:.0f}s < 上限）→ 一方胜利点归零，被扣死了")
        if last['deaths'] > 3:
            print(f"    * 战亡 {last['deaths']:.1f} 次 × 20 = {last['deaths']*20:.0f} VP 来自战亡")
        print(f"    * 占点带来的优势很少（平均占点率 {sum(x['hold'] for x in trace)/len(trace):.3f}）")
    else:
        print("    * 对局打满全程（未提前结束）")
    if no_enemy_fire:
        print("    * 注意：本组**关闭了对手火力**，用于隔离'竞争压力'这一个变量")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rule", action="store_true", help="规则基线（旧）")
    ap.add_argument("--tactical", action="store_true", help="战术层（V1）")
    ap.add_argument("--no-enemy-fire", action="store_true")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.season, args.device,
        Path(args.ckpt) if args.ckpt else None,
        args.envs, args.no_enemy_fire, args.random,
        use_rule=args.rule, use_tactical=args.tactical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
