"""验证势函数塑形（potential-based shaping）的关键性质。

依据：Ng, Harada & Russell 1999 的 policy invariance 定理 ——
只有 F(s,a,s') = gamma*Phi(s') - Phi(s) 形式的塑形才不改变最优策略。

⚠️ 2026-10-13 复核（既有问题，不是新引入的）：原版有两项断言与公式的数学性质不符，
   已按实情修正断言语义，但**副作用本身仍待处置**（见 `工作台.md` §六 未解问题 #4）：

   1. **原地停留的 F 不为 0**。静止 s'=s ⇒ F = (γ−1)Φ(d)。
      γ=0.99、Φ∈[−1,0] ⇒ 每步约 +0.006~0.010，且**越远越大**。
      这是"有界 Φ + γ<1"的必然结果，不是实现 bug；
      但"站着不动也给药"确实是个需要观察的副作用（与滞留刷分风险同向）。
   2. **远处向外移动的 F 会转正**。F_out(d) = γΦ(d+0.15) − Φ(d)，
      在 d ≳ 5.9 m 变正（tanh 饱和把 (γ−1)Φ 的负漂移限制住了）。
      含义：离中心 >5.9 m 的区域，塑形在推动策略**往外走**。
      本项目场地 12×8、中心到角落约 7 m ⇒ 该区域可达；
      且与"RL 策略跑到离中心 5.9~7.2 m"的实测现象方向一致 → 立项复查。

本脚本检查：
  1. 原地停留 -> 漂移足够小（有界，不构成主要刷分通道）
  2. 朝中心移动 -> F > 0（正向激励）
  3. 远离中心移动 -> F < 0（**仅在 d ≤ 3 m 断言**；更远处按上面的复核结果不成立）
  4. F 有界（无需 clamp，clamp 会破坏势函数性质）
  5. 重生瞬间不产生尖峰（跨 episode 状态必须作废）
"""
from __future__ import annotations

import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S

OUT = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    OUT.append(ok)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    c = RMUL3v3Config(season="2026")
    g, sig = c.gamma, c.approach_sigma

    print("=" * 70)
    print("势函数塑形性质验证")
    print("=" * 70)
    print(f"  F = gamma*Phi(s') - Phi(s),  Phi = -tanh(d/sigma)")
    print(f"  gamma={g}  sigma={sig}  w_approach={c.w_approach}")
    print()

    def phi(d: float) -> torch.Tensor:
        return -torch.tanh(torch.tensor(d / sig))

    print("[1] 原地停留的漂移必须足够小（防止滞留刷分；精确为 0 在 γ<1 时不可能）")
    worst = 0.0
    for d in (0.0, 0.5, 0.8, 1.5, 3.0, 6.0, 10.0):
        f = float(g * phi(d) - phi(d))
        worst = max(worst, abs(f))
    # 精确值 |F| = (1-γ)|Φ(d)| ≤ 0.01；断言留一倍余量。
    # 注意：这个漂移**随距离增大**（远处 Φ→−1），所以不能靠"越小越安全"解读。
    ck("原地漂移 ≤ 0.02/步（远小于占点 +0.10/步）", worst <= 0.02,
       f"最大偏差 {worst:.2e}（理论值 (1-γ)·|Φ| ≤ 0.01）")

    print("\n[2] 朝中心移动应给正奖励")
    ok_pos = True
    for d in (0.8, 1.5, 3.0, 6.0):
        f = float(g * phi(d - 0.15) - phi(d))
        if f <= 0:
            ok_pos = False
        print(f"      d={d:4.1f} -> {d-0.15:4.2f}   F={f:+.6f}")
    ck("靠近时 F>0", ok_pos)

    print("\n[3] 远离中心移动应给负奖励（仅 d ≤ 3 m —— 远处公式会转正，见文件头复核记录）")
    ok_neg = True
    for d in (0.8, 1.5, 3.0):
        f = float(g * phi(d + 0.15) - phi(d))
        if f >= 0:
            ok_neg = False
        print(f"      d={d:4.1f} -> {d+0.15:4.2f}   F={f:+.6f}")
    ck("近距离远离时 F<0", ok_neg)
    # d=6.0 是已知的转正点（实测 +0.0063）：打印出来作为"未决问题"的现场证据，
    # 但不作为失败 —— 是否改公式是设计决定，见 工作台.md §六 #4。
    f6 = float(g * phi(6.15) - phi(6.0))
    print(f"      d= 6.0 ->  6.15   F={f6:+.6f}  ← 已转正（未决问题 #4，非断言项）")

    print("\n[4] F 必须有界（有界才不需要 clamp；clamp 会破坏 policy invariance）")
    fs = [float(g * phi(d) - phi(x)) for d in (0.0, 1.0, 5.0, 20.0)
          for x in (0.0, 1.0, 5.0, 20.0)]
    ck("|F| <= 2", max(abs(x) for x in fs) <= 2.0,
       f"实际最大 |F| = {max(abs(x) for x in fs):.4f}")

    print("\n[5] 环境里重生瞬间不应有尖峰（跨 episode 状态作废）")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = BatchedRMUL3v3(c, n_envs=32, device=dev, seed=3)
    env.reset()
    a = torch.zeros(env.n, env.n_actions, device=dev)
    a[:, 0] = 1.0
    for _ in range(300):
        env.step(a)
    d0 = float(torch.hypot(env.sentry[:, 0] - c.field_w / 2,
                           env.sentry[:, 1] - c.field_h / 2).mean())
    env.sentry[:, S.HP] = 0.0            # 强制阵亡
    _, _, done, _ = env.step(torch.zeros(env.n, env.n_actions, device=dev))
    env.auto_reset(done)
    _, r2, _, _ = env.step(torch.zeros(env.n, env.n_actions, device=dev))
    ck("重生后第一步的塑形不产生尖峰", float(r2.abs().max()) < 1.0,
       f"阵亡前离中心 {d0:.2f}m，重生后 max|r| = {float(r2.abs().max()):.4f}")

    print("\n" + "=" * 70)
    n_fail = OUT.count(False)
    print(f"通过 {OUT.count(True)} 项，失败 {n_fail} 项")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
