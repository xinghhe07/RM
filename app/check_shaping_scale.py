"""验证势函数塑形在**真实距离范围**内的数值有效性。

动机
----
Φ = −tanh(d/σ) 在 d ≫ σ 时会饱和、梯度趋零。若哨兵常在离控制区很远处
（实测平均 2~7 m），而 σ 取小了，塑形信号会消失 —— 等于没有塑形。

本脚本直接测：在每个距离上，"移动一步"带来的 F = γΦ(s') − Φ(s) 有多大，
并与其它奖励项（占点 +0.1/步、命中 +0.06/发）比较，确认塑形是否"看得见"。
"""
from __future__ import annotations

import torch

from rm_sentry.rmul_3v3_env import RMUL3v3Config

c = RMUL3v3Config(season="2026")
g, sig = c.gamma, c.approach_sigma
step = c.v_max * c.dt          # 一步最大位移（3.0 m/s × 0.05 s = 0.15 m）


def phi(d):
    return -torch.tanh(torch.tensor(d / sig))


print("=" * 78)
print("势函数塑形数值有效性")
print(f"  gamma={g}  sigma={sig}  v_max*dt={step:.3f} m/步   w_approach={c.w_approach}")
print("=" * 78)
print(f"  {'距离 d':>8s} {'靠近一步 F':>12s} {'远离一步 F':>12s} {'原地 F':>10s}  {'占点/步':>8s}")
print("-" * 78)
for d in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
    f_in = float(g * phi(max(d - step, 0.0)) - phi(d)) * c.w_approach
    f_out = float(g * phi(d + step) - phi(d)) * c.w_approach
    f_stay = float(g * phi(d) - phi(d)) * c.w_approach
    hold = c.w_hold * c.dt
    print(f"  {d:8.2f} {f_in:+12.5f} {f_out:+12.5f} {f_stay:+10.6f}  {hold:8.3f}")

print()
print("参照：其它奖励项的量级")
print(f"  占点 1 步        {c.w_hold * c.dt:+.4f}")
print(f"  命中 1 发(20伤害) {c.w_dmg_out * 20:+.4f}")
print(f"  挨 1 发          {c.w_dmg_in * 20:+.4f}")
print(f"  阵亡一次         {c.w_vp_adv * -20:+.4f}")
print()

# 关键判定：在最常见的交战距离（2~7 m）上，塑形信号必须可与占点项相比
print("判定：塑形信号在 2~7 m 范围内是否仍然显著（> 占点项的 5%）")
thr = c.w_hold * c.dt * 0.05
ok = True
for d in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
    f_in = abs(float(g * phi(max(d - step, 0.0)) - phi(d)) * c.w_approach)
    good = f_in > thr
    ok &= good
    print(f"  d={d:4.1f}  |F|={f_in:.5f}  {'OK' if good else '过弱'}")
print()
print(f"  结论：{'塑形在整个距离范围内有效' if ok else '*** sigma 偏小，远处塑形饱和失效 ***'}")
print(f"  （饱和阈值参考：tanh 在 d/sigma > 2 后导数 < 0.08，即 d > {2*sig:.1f} m 时明显衰减）")
