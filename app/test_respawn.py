"""复活机制验证 —— 常规回归测试，不是临时排障脚本。

为什么需要它
-----------
复活逻辑连续错了三次，每次都让**所有长于 ~13 秒的评测失真**：
  错法 1  先递减 DEAD 并 clamp(0) 再判断 (DEAD > 0) -> 归零后永假 -> 永不复活
  错法 2  保存 dead_prev 再判断      -> 阵亡当步 dead_prev 是 0，逻辑被污染
  错法 3  用 (DEAD + dt) 反推上一步   -> DEAD 被 clamp 后反推得 dt(>0)，
                                        "读条结束"被误判为"正在读条"，
                                        每局第一步就白送一次复活（HP 400 -> 80）

三次的共同教训：**不要反推状态，不要看状态位跨越；显式记住转换。**
（与 AGENTS.md R2 的"跨 step 状态"陷阱同类）

用法：python test_respawn.py
"""
from __future__ import annotations

import statistics

import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, E, S

FAILS: list[str] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def check_2027(dev: str) -> None:
    """2027 复活链：读条 5/+5（与 2026 的 10/+10 不同）、无敌/虚弱与读条分离。

    2026 把"无敌期"并入 DEAD 倒计时；2027 的 DEAD 只表示读条，复活后有独立的
    30 s 无敌 + 虚弱（触己方补给区才解除）—— 本函数专门验证这个区别。
    """
    cfg = RMUL3v3Config(season="2027")
    env = BatchedRMUL3v3(cfg, n_envs=4, device=dev, seed=17)
    env.reset()
    act = torch.zeros(env.n, env.n_actions, device=dev)

    def park():
        env.sentry[:, S.X] = 1.0
        env.sentry[:, S.Y] = 7.0          # 远离补给区（y ≤ 2），避免回血干扰
        env.enemies[:, 1:, E.X] = 11.0
        env.enemies[:, 1:, E.Y] = 7.0
        env.enemies[:, 1:, E.ALIVE] = 1.0

    park()
    env.sentry[:, S.HP] = 0.0
    env.step(act); park()
    ck("2027：首次读条 = 5 s",
       abs(float(env.sentry[0, S.DEAD]) - 5.0) < 0.06,
       f"实得 {float(env.sentry[0, S.DEAD]):.2f}")

    for _ in range(100):
        env.step(act); park()
    ck("2027：复活后血量 = 20%（80）",
       abs(float(env.sentry[0, S.HP]) - cfg.sentry_hp * cfg.respawn_hp_ratio) < 1.0,
       f"实得 {float(env.sentry[0, S.HP]):.0f}")
    ck("2027：复活后进入虚弱（锁发射 + 不能占区）",
       float(env.sentry[0, S.VULN]) > 0.5)

    # 第二次战亡：读条 5 + 5 = 10 s
    env.sentry[:, S.HP] = 0.0
    env.step(act); park()
    ck("2027：第二次读条 = 10 s",
       abs(float(env.sentry[0, S.DEAD]) - 10.0) < 0.06,
       f"实得 {float(env.sentry[0, S.DEAD]):.2f}")


def main() -> int:
    cfg = RMUL3v3Config(season="2026")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = 16
    print("=" * 72)
    print("复活机制回归验证")
    print("=" * 72)
    print(f"  血量上限 {cfg.sentry_hp}   复活血量 {cfg.sentry_hp*cfg.respawn_hp_ratio:.0f}   "
          f"首次读条 {cfg.respawn_base_len/cfg.respawn_progress_per_s + cfg.respawn_invuln:.0f}s")

    env = BatchedRMUL3v3(cfg, n_envs=n, device=dev, seed=7)
    env.reset()

    ck("reset 后 HP = 血量上限", bool((env.sentry[:, S.HP] == cfg.sentry_hp).all()),
       f"实际 {env.sentry[0, S.HP].item():.0f}")

    a = torch.zeros(n, env.n_actions, device=dev)
    hp0 = env.sentry[:, S.HP].clone()
    env.step(a)
    drop = hp0 - env.sentry[:, S.HP]
    ck("step 后 HP 只小幅减少（未被误写成复活血量）",
       bool((drop >= 0).all() and (drop <= 60).all()),
       f"单步掉血 {drop.min().item():.0f}~{drop.max().item():.0f}")

    # 静止挨打，观察死亡-复活循环（跑满一局长度）
    prev_dead = env.sentry[:, S.DEAD].clone()
    prev_deaths = env.sentry[:, S.DEATHS].clone()
    hp_after: list[float] = []
    n_death = n_revive = 0
    for _ in range(int(cfg.match_time / cfg.dt)):
        _, _, done, _ = env.step(a)
        hp, dead, deaths = env.sentry[:, S.HP], env.sentry[:, S.DEAD], env.sentry[:, S.DEATHS]
        new_death = deaths > prev_deaths          # 同一局内 DEATHS 只增
        revive = (prev_dead > 0) & (dead <= 0) & ~new_death
        n_death += int(new_death.sum())
        if bool(revive.any()):
            n_revive += int(revive.sum())
            hp_after.extend(hp[revive].tolist())
        env.auto_reset(done)
        prev_dead, prev_deaths = dead.clone(), deaths.clone()

    ck("发生了自然死亡（不是永久存活）", n_death > 0, f"{n_death} 次")
    ck("发生了复活（不是永久死亡）", n_revive > 0, f"{n_revive} 次")
    if hp_after:
        m = statistics.mean(hp_after)
        ck("复活后血量正确", abs(m - cfg.sentry_hp * cfg.respawn_hp_ratio) < 1.0,
           f"均值 {m:.1f}（期望 {cfg.sentry_hp*cfg.respawn_hp_ratio:.0f}）")
    ck("死亡-复活形成循环", n_revive >= 2, f"{n_death} 死 / {n_revive} 活")

    print()
    check_2027(dev)

    print("\n" + "=" * 72)
    print(f"失败 {len(FAILS)} 项" + ("  —— 全部通过" if not FAILS else ""))
    for f in FAILS:
        print(f"  - {f}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
