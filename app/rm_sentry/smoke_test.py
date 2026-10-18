"""冒烟测试 + 吞吐基准：验证环境正确性并测出并行吞吐。

运行：
    cd C:\\RM
    python -m rm_sentry.smoke_test            # 默认 4000 环境，200 步
    python -m rm_sentry.smoke_test --n 1024 --steps 100

验收标准（阶段 2 的目标）：吞吐 > 10 万步/秒。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S, E


def check_static(env: BatchedRMUL3v3) -> None:
    """维度与观测范围的静态检查。"""
    cfg = env.cfg
    expected_dim = 6 + 8 + 7 * cfg.n_enemies + 1
    if cfg.vp_mode == "control_degree":
        expected_dim += 2 + 6 * len(cfg.zones_rect) + 2   # 2027：虚弱/无敌 + 控制区块 + 控制度
    assert env.obs_dim == expected_dim, "obs_dim 公式不符"
    assert env.n_actions == 2 + (cfg.n_enemies + 1) + 1, "n_actions 公式不符"

    obs = env.reset()
    assert obs.shape == (env.n, env.obs_dim), f"obs shape {tuple(obs.shape)} 不符"
    assert torch.isfinite(obs).all(), "reset 出现 NaN/Inf"

    # 规则一致性（赛季相关）
    assert torch.allclose(env.sentry[:, S.HP], torch.full_like(env.sentry[:, S.HP], cfg.sentry_hp))
    assert torch.allclose(env.sentry[:, S.AMMO], torch.full_like(env.sentry[:, S.AMMO], cfg.ammo_init))
    assert float(env.sentry[:, S.HEAT1].abs().max()) == 0.0
    assert torch.allclose(env.sentry[:, S.VP], torch.full_like(env.sentry[:, S.VP], cfg.vp_init)), \
        f"开局胜利点应为 {cfg.vp_init}"
    assert torch.allclose(env.sentry[:, S.GOLD], torch.full_like(env.sentry[:, S.GOLD], cfg.initial_gold)), \
        f"开局金币应为 {cfg.initial_gold}"

    mask = env.action_mask()
    assert mask.dtype == torch.bool and mask.shape == (env.n, env.n_actions)
    print(f"  [ok] 赛季 {cfg.season} | obs_dim={env.obs_dim} n_actions={env.n_actions} "
          f"| {cfg.n_barrels} 管/热量{cfg.heat_limit:.0f}/冷却{cfg.heat_cool:.0f} "
          f"| 胜利点 {cfg.vp_mode} 开局{cfg.vp_init:.0f}")
    print(f"  [ok] 开局金币={cfg.initial_gold} 可用目标数 min/mean = "
          f"{int(mask[:, 2:-1].sum(1).min())}/{float(mask[:, 2:-1].sum(1).float().mean()):.2f}")


def check_episode(env: BatchedRMUL3v3, max_steps: int = 400) -> dict:
    """跑一个完整片段，检查数值稳定性与规则不变量。"""
    cfg = env.cfg
    env.reset()
    vp_hist = []
    for t in range(max_steps):
        mask = env.action_mask()
        # 随机但合法的动作
        act = torch.zeros(env.n, env.n_actions, device=env.device)
        act[:, 0:2] = (torch.rand(env.n, 2, device=env.device) * 2 - 1)
        tgt = torch.randint(0, cfg.n_enemies + 1, (env.n,), device=env.device)
        # 把不可见的换成"不交战"
        legal = mask[:, 2:2 + cfg.n_enemies + 1]
        picked = legal.gather(1, tgt.unsqueeze(1)).squeeze(1)
        tgt = torch.where(picked, tgt, torch.zeros_like(tgt))
        act[:, 2] = tgt.float()
        act[:, 3] = (torch.rand(env.n, device=env.device) < 0.7).float()
        obs, rew, done, info = env.step(act)

        assert torch.isfinite(obs).all(), f"step {t} obs 非有限值"
        assert torch.isfinite(rew).all(), f"step {t} reward 非有限值"
        assert (env.sentry[:, S.HP] >= 0).all() and (env.sentry[:, S.HP] <= cfg.sentry_hp).all()
        assert (env.sentry[:, S.AMMO] >= 0).all()
        assert (env.sentry[:, S.HEAT1] >= 0).all() and (env.sentry[:, S.HEAT1] <= cfg.heat_limit * 1.01).all()
        assert (env.sentry[:, S.X] >= 0).all() and (env.sentry[:, S.X] <= cfg.field_w).all()
        assert (env.sentry[:, S.Y] >= 0).all() and (env.sentry[:, S.Y] <= cfg.field_h).all()
        assert bool((env.enemies[:, :, E.HP] <= 400).all())
        vp_hist.append(float(info["vp"].mean()))
        if bool(done.all()):
            break
    env.auto_reset(done)
    print(f"  [ok] 片段 {t + 1} 步, vp 均值 {vp_hist[0]:.2f} -> {vp_hist[-1]:.2f}, "
          f"平均伤害输出 {float(info['dmg_out'].mean()):.1f}")
    return {"steps": t + 1, "vp_final": vp_hist[-1]}


def bench(env: BatchedRMUL3v3, steps: int) -> float:
    """吞吐基准：步/秒（所有并行环境累计）。"""
    env.reset()
    act = torch.zeros(env.n, env.n_actions, device=env.device)
    act[:, 0] = 0.3
    # 预热（CUDA 首次编译 kernel）
    for _ in range(5):
        env.step(act)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = env.n * steps / dt
    return sps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="并行环境数")
    ap.add_argument("--steps", type=int, default=200, help="基准步数")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--season", type=str, default="2026", choices=["2025", "2026", "2027"],
                    help="规则赛季档案（2026 = 你们要打的赛季）")
    args = ap.parse_args()

    cfg = RMUL3v3Config(season=args.season)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print(f"RMUL 3v3 哨兵环境 冒烟测试  [赛季 {cfg.season}]")
    print("=" * 68)
    print(f"规则来源: {cfg.profile_source}")
    if dev == "cuda":
        print(f"设备: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        print(f"设备: {dev}")
    print(f"torch {torch.__version__} / cuda {torch.version.cuda}")

    print(f"\n[1] 静态检查 (n_envs=8)")
    env8 = BatchedRMUL3v3(cfg, n_envs=8, device=dev, seed=1234)
    check_static(env8)

    print(f"\n[2] 完整片段检查 (n_envs=8, 最多 400 步 = 20 s 对战时间)")
    check_episode(env8, max_steps=400)

    print(f"\n[3] 吞吐基准 (n_envs={args.n}, {args.steps} 步)")
    env = BatchedRMUL3v3(cfg, n_envs=args.n, device=dev, seed=0)
    sps = bench(env, args.steps)
    mem = (torch.cuda.max_memory_allocated() / 1024**2) if dev == "cuda" else 0.0
    print(f"  {sps:,.0f} 步/秒" + (f"  峰值显存 {mem:.0f} MB" if dev == "cuda" else ""))
    # 单局 3000 步，一分钟能跑多少局
    eps_per_min = sps / (cfg.match_time / cfg.dt) * 60
    print(f"  约合 {eps_per_min:,.0f} 局/分钟（一局 {int(cfg.match_time / cfg.dt)} 步）")
    verdict = "达标" if sps > 1e5 else "未达标（阶段 2 目标 > 10 万步/秒）"
    print(f"  验收: {verdict}")
    print("\n" + "=" * 68)
    print("全部通过" if sps > 0 else "失败")


if __name__ == "__main__":
    main()
