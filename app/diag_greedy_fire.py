"""隔离实验：只回答一个问题 —— **2026 配置下 1v3 到底打不打得赢？**

为什么需要它
------------
`diag_fire_time.py` 发现规则基线**只打出理论射速的 2.5%**（活着能看见却不开火）。
所以"1v3 打不赢"这个结论被**策略太差**污染了，不能直接下。

本实验用一个**最小人工策略**把开火率拉到接近上限：
  - 永远选第一个可见敌人
  - 只要掩码允许就要求开火
  - 不移动（隔离运动策略的影响）—— 或可选原地小陀螺
这样测出的存活/输出才是"环境本身"的属性，而不是规则策略的缺陷。

只打印事实，不做推断（AGENTS.md「先跑工具再猜」）。
"""
from __future__ import annotations

import argparse

import torch

from rm_sentry.rmul_3v3_env import E, S, BatchedRMUL3v3, RMUL3v3Config
from rm_sentry.train_ppo import to_env_action


def greedy_fire_action(env, mask, n_actions, dev) -> torch.Tensor:
    """最小策略：选第一个可见敌人 + 永远开火 + 不移动。"""
    n = env.sentry.shape[0]
    act = torch.zeros(n, n_actions, device=dev)
    # 通道 2+m 对应 _vis[:, m]（槽 0 是自机）
    vis = env._los_blocked()
    ne = env.cfg.n_enemies
    for m in range(1, ne + 1):
        ch = 2 + m
        if ch >= n_actions - 1:
            break
        ok = mask[:, ch] & vis[:, m]
        act[:, ch] = torch.where(ok, torch.ones_like(act[:, ch]), act[:, ch])
    # 取第一个被选中的（互斥：只保留最低索引）
    chosen = act[:, 2:2 + ne].cumsum(1) <= 1
    act[:, 2:2 + ne] = act[:, 2:2 + ne] * chosen.float()
    # 开火通道
    can = mask[:, -1] & (act[:, 2:2 + ne].sum(1) > 0)
    act[:, -1] = can.float()
    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--season", default="2026")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--enemy-dmg", type=float, default=None,
                    help="覆盖敌方每发伤害（默认用 cfg）")
    ap.add_argument("--move", action="store_true", help="允许移动（否则原地）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = RMUL3v3Config(season=args.season)
    if args.enemy_dmg is not None:
        cfg.enemy_dmg_per_hit = args.enemy_dmg
    dev = args.device
    env = BatchedRMUL3v3(cfg, n_envs=args.envs, device=dev, seed=20261007)
    env.reset()
    mask = env.action_mask()

    ne = cfg.n_enemies
    acc = dict.fromkeys(("alive", "fired", "kills", "dmg_out", "dmg_in",
                         "heat_lock", "deaths"), 0.0)
    steps = 0
    done_n = 0
    win = lose = 0.0

    for _ in range(args.steps):
        a = greedy_fire_action(env, mask, env.n_actions, dev)
        if args.move:
            # 简单朝敌人方向移动（仅用于对照）
            enf = env.enemies[:, 1:1 + ne]
            alive_e = enf[:, :, E.ALIVE] > 0.5
            dx = enf[:, :, E.X] - env.sentry[:, S.X].unsqueeze(1)
            dy = enf[:, :, E.Y] - env.sentry[:, S.Y].unsqueeze(1)
            d = torch.sqrt(dx * dx + dy * dy + 1e-9).masked_fill(~alive_e, 1e9)
            j = d.argmin(1)
            gx = torch.gather(dx, 1, j.unsqueeze(1)).squeeze(1)
            gy = torch.gather(dy, 1, j.unsqueeze(1)).squeeze(1)
            norm = torch.sqrt(gx * gx + gy * gy + 1e-9)
            a[:, 0] = gx / norm * 0.5
            a[:, 1] = gy / norm * 0.5

        ammo_before = env.sentry[:, S.AMMO].clone()
        alive_before = (env.enemies[:, 1:1 + ne, E.ALIVE] > 0.5).sum(1).float()
        alive_s = (env.sentry[:, S.DEAD] <= 0).float()
        lock = ((env.sentry[:, S.HEAT1] > cfg.heat_limit) |
                (env.sentry[:, S.HEAT2] > cfg.heat_limit)).float()

        obs, rew, done, info = env.step(a)

        acc["alive"] += alive_s.mean().item()
        acc["heat_lock"] += lock.mean().item()
        acc["fired"] += (ammo_before - env.sentry[:, S.AMMO]).clamp_min(0).sum().item() / args.envs
        acc["kills"] += (alive_before -
                         (env.enemies[:, 1:1 + ne, E.ALIVE] > 0.5).sum(1).float()).mean().item()
        acc["dmg_out"] += env.sentry[:, S.DMG_OUT].mean().item()
        acc["dmg_in"] += env.sentry[:, S.DMG_IN].mean().item()
        acc["deaths"] += env.sentry[:, S.DEATHS].mean().item()
        steps += 1

        if bool(done.any()):
            win += float(info["win"][done].sum())
            lose += float(info["lose"][done].sum())
            done_n += int(done.sum())
        env.auto_reset(done)
        mask = env.action_mask()

    per_step = cfg.heat_cool / cfg.heat_per_shot * cfg.dt
    print("=" * 70)
    print(f"隔离实验：最小“能开火就开火”策略   season={args.season}  "
          f"敌方每发={cfg.enemy_dmg_per_hit}  移动={args.move}  envs={args.envs}")
    print("=" * 70)
    print(f"  存活率            : {acc['alive']/steps:.3f}")
    print(f"  出弹/步           : {acc['fired']/steps:.3f}  (理论 {per_step:.3f}, "
          f"占比 {acc['fired']/steps/max(per_step,1e-9):.1%})")
    print(f"  热量锁定占比      : {acc['heat_lock']/steps:.3f}")
    print(f"  平均累计击杀      : {acc['kills']/steps:.2f} / {ne} 敌")
    print(f"  平均累计伤害输出  : {acc['dmg_out']/steps:.1f}")
    print(f"  平均累计承伤      : {acc['dmg_in']/steps:.1f}")
    print(f"  平均累计阵亡次数  : {acc['deaths']/steps:.2f}")
    print(f"  结束局数          : {done_n}")
    if done_n:
        print(f"  胜率              : {win/done_n:.3f}")
        print(f"  负率              : {lose/done_n:.3f}")
    print()
    print("  ---- 判读：这个策略把开火率拉满后，1v3 打得赢吗？ ----")
    if done_n == 0:
        print("  局数不足，加 --steps")
    elif win / done_n > 0.5:
        print(f"  ✅ 打得赢（胜率 {win/done_n:.2f}）-> “1v3 不可能”的结论**错误**")
    else:
        print(f"  ❌ 仍打不赢（胜率 {win/done_n:.2f}）-> 支持“1v3 结构性劣势”")


if __name__ == "__main__":
    main()
