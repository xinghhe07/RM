"""诊断：哨兵为什么只打出理论伤害的 41%？

把"无法开火"的原因分解并各自量化，只打印事实、不做推断
（符合 AGENTS.md「先跑工具再猜」）。

口径与 `rm_sentry.rule_baseline.evaluate_rule` **完全一致**：
  - `rule.act(None, mask, env)` -> dict
  - `to_env_action(dict, n_actions, dev)` -> tensor
  - 每步后 `env.auto_reset(done)`（否则状态跨局污染）

分解：
  A 存活 / B 射程内 / C 视野内（FOV+遮挡）/ D 热量锁定 / E 实际出弹（按弹药消耗）
"""
from __future__ import annotations

import argparse

import torch

from rm_sentry.rmul_3v3_env import E, S, BatchedRMUL3v3, RMUL3v3Config
from rm_sentry.rule_baseline import RuleSentry
from rm_sentry.train_ppo import to_env_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--season", default="2026")
    ap.add_argument("--no-enemy-fire", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = RMUL3v3Config(season=args.season)
    if args.no_enemy_fire:
        cfg.enemy_dmg_per_hit = 0.0   # 隔离竞争压力（与 audit_episode.py 同思路）
    dev = args.device
    env = BatchedRMUL3v3(cfg, n_envs=args.envs, device=dev, seed=20261007)
    rule = RuleSentry(cfg)

    env.reset()
    mask = env.action_mask()

    acc = dict.fromkeys(
        ("alive", "in_range", "visible", "heat_locked", "fired",
         "dist", "dmg_out", "dmg_in", "kills", "sfire"), 0.0)
    steps = 0
    ne = cfg.n_enemies

    for _ in range(args.steps):
        a = to_env_action(rule.act(None, mask, env), env.n_actions, dev)

        alive_s = (env.sentry[:, S.DEAD] <= 0).float()
        enf = env.enemies[:, 1:1 + ne]
        alive_e = enf[:, :, E.ALIVE] > 0.5
        dx = enf[:, :, E.X] - env.sentry[:, S.X].unsqueeze(1)
        dy = enf[:, :, E.Y] - env.sentry[:, S.Y].unsqueeze(1)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
        in_range = (dist <= cfg.shot_range) & alive_e
        vis = env._los_blocked()
        visible = vis[:, 1:1 + ne] & alive_e
        locked = ((env.sentry[:, S.HEAT1] > cfg.heat_limit) |
                  (env.sentry[:, S.HEAT2] > cfg.heat_limit)) & (alive_s > 0.5)

        ammo_before = env.sentry[:, S.AMMO].clone()
        n_alive_before = alive_e.sum(1).float()

        obs, rew, done, info = env.step(a)

        fired = (ammo_before - env.sentry[:, S.AMMO]).clamp_min(0)
        kills = n_alive_before - (env.enemies[:, 1:1 + ne, E.ALIVE] > 0.5).sum(1).float()

        acc["alive"] += alive_s.mean().item()
        acc["in_range"] += in_range.any(1).float().mean().item()
        acc["visible"] += visible.any(1).float().mean().item()
        acc["heat_locked"] += locked.float().mean().item()
        acc["fired"] += fired.sum().item() / args.envs
        acc["kills"] += kills.mean().item()
        if bool(alive_e.any()):
            acc["dist"] += dist[alive_e].mean().item()
        acc["dmg_out"] += env.sentry[:, S.DMG_OUT].mean().item()
        acc["dmg_in"] += env.sentry[:, S.DMG_IN].mean().item()
        steps += 1

        env.auto_reset(done)
        mask = env.action_mask()

    rate_max = cfg.heat_cool / cfg.heat_per_shot      # 发/秒
    per_step = rate_max * cfg.dt

    print("=" * 68)
    print(f"诊断：可开火条件分解   season={args.season}  "
          f"enemy_dmg_per_hit={cfg.enemy_dmg_per_hit}  envs={args.envs}  steps={steps}")
    print("=" * 68)
    print(f"  A 存活            : {acc['alive']/steps:.3f}")
    print(f"  B 有敌在射程内     : {acc['in_range']/steps:.3f}   (射程 {cfg.shot_range} m)")
    print(f"  C 有敌在视野内     : {acc['visible']/steps:.3f}   (FOV {cfg.fov_deg}° + 遮挡)")
    print(f"  D 热量锁定中       : {acc['heat_locked']/steps:.3f}")
    print(f"  E 本步实际出弹     : {acc['fired']/steps:.3f} 发/环境/步")
    print()
    print(f"  平均敌人距离       : {acc['dist']/steps:.2f} m")
    print(f"  平均累计伤害输出   : {acc['dmg_out']/steps:.1f}")
    print(f"  平均累计承伤       : {acc['dmg_in']/steps:.1f}")
    print(f"  平均累计击杀       : {acc['kills']/steps:.2f} 敌")
    print()
    print("  ---- 判读 ----")
    al = acc["alive"] / steps
    vi = acc["visible"] / steps
    ir = acc["in_range"] / steps
    print(f"  存活 -> 射程内 -> 视野内 : {al:.3f} -> {ir:.3f} -> {vi:.3f}")
    if al > 1e-9:
        print(f"  存活时能看见敌人   : {vi/al:.3f}   <- 越低说明 FOV/遮挡 越卡")
    print(f"  理论射速上限       : {rate_max:.1f} 发/秒 = {per_step:.3f} 发/步")
    print(f"  实际/理论出弹      : {acc['fired']/steps/max(per_step,1e-9):.3f}")
    # 理论伤害（按实际出弹数 × 平均伤害）
    print(f"  理论应打出伤害     : {acc['fired']/steps * cfg.dmg_base * 0.65 / cfg.dt:.0f}"
          f"  (按实际出弹 x {cfg.dmg_base:.0f} 伤 x 0.65 衰减)")


if __name__ == "__main__":
    main()
