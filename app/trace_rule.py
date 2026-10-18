"""规则基线整局行为追踪：为什么占点率只有 0.143 而不是冒烟测试的 1.000。

只打印事实，不做推断。
"""
from __future__ import annotations

import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, E, RMUL3v3Config, S
from rm_sentry.rule_baseline import RuleSentry
from rm_sentry.train_ppo import to_env_action

cfg = RMUL3v3Config(season="2026")
dev = "cuda" if torch.cuda.is_available() else "cpu"
n = 8
env = BatchedRMUL3v3(cfg, n_envs=n, device=dev, seed=1)
env.reset()
rule = RuleSentry(cfg)
mask = env.action_mask()
cx, cy = cfg.field_w / 2, cfg.field_h / 2
a = torch.zeros(n, env.n_actions, device=dev)

print(f"  {'步':>5s} {'秒':>6s} {'到中心':>7s} {'占点':>6s} {'开火':>6s} {'HP':>6s} "
      f"{'DEAD':>6s} {'死亡数':>6s} {'VP':>6s} {'敌VP':>6s} {'活敌':>5s}")
for t in range(int(cfg.match_time / cfg.dt)):
    act = rule.act(None, mask, env)
    a2 = to_env_action(act, env.n_actions, dev)
    obs, r, done, info = env.step(a2)
    if t % 500 == 0 or t == int(cfg.match_time / cfg.dt) - 1:
        dist = torch.hypot(env.sentry[:, S.X] - cx, env.sentry[:, S.Y] - cy).mean()
        print(f"  {t:5d} {t*cfg.dt:6.1f} {float(dist):7.2f} "
              f"{float(info['hold_ratio'].mean()):6.3f} {float(act['fire'].mean()):6.3f} "
              f"{float(env.sentry[:,S.HP].mean()):6.1f} {float(env.sentry[:,S.DEAD].mean()):6.2f} "
              f"{float(env.sentry[:,S.DEATHS].mean()):6.1f} {float(env.sentry[:,S.VP].mean()):6.1f} "
              f"{float(env.sentry[:,S.VP_ENEMY].mean()):6.1f} "
              f"{float((env.enemies[:,:,E.ALIVE]>0.5).sum(1).float().mean()):5.2f}")
    env.auto_reset(done)
    mask = env.action_mask()
