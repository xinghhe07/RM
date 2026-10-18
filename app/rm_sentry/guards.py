"""运行时护栏：让训练「要么健康地跑，要么明确报错」，而不是安静地学错。

为什么需要这个
--------------
训练最贵的失败不是崩溃，而是：
  1. **静默学错** —— 跑完几小时，看曲线觉得在涨，其实策略塌缩/观测错位/超参写反
  2. **慢速劣化** —— 泄漏导致吞吐越来越低、loss 慢慢 NaN，几小时后才发现
  3. **配置错误** —— 参数组合自相矛盾（如 clip<=0、评测频率为负），启动时无人检查

本模块提供两层：
  * `validate_train_cfg()`  启动时立刻拦住自相矛盾的配置（花 0 秒，省几小时）
  * `HealthMonitor`         训练中滚动记录指标，主动判定"是否健康"，并在异常时
                            给出可执行的中文诊断，而不是丢一个 NaN 让你猜

设计取向：**只报明确可判定的问题**，不做玄学判断。每个告警都带具体数值与建议。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# 一层：启动时配置校验
# --------------------------------------------------------------------------
class ConfigError(ValueError):
    """配置自相矛盾 —— 应该在校验阶段拦住，而不是训练中途才炸。"""


def validate_train_cfg(tc: Any, cfg: Any, env: Any) -> list[str]:
    """校验训练配置与环境的匹配性。返回警告列表；致命问题直接抛 ConfigError。"""
    errs: list[str] = []
    warns: list[str] = []

    # ---- 致命：这些组合一定训不出东西 ----
    if getattr(tc, "clip", 0.2) <= 0:
        errs.append(f"clip={tc.clip} 必须 > 0（PPO 的裁剪范围，<=0 会让策略无法更新）")
    if getattr(tc, "gamma", 0.99) <= 0 or tc.gamma > 1:
        errs.append(f"gamma={tc.gamma} 应在 (0, 1]")
    if getattr(tc, "lam", 0.95) <= 0 or tc.lam > 1:
        errs.append(f"lam={tc.lam} 应在 (0, 1]（GAE 的 λ）")
    if getattr(tc, "lr", 1e-4) <= 0:
        errs.append(f"lr={tc.lr} 必须 > 0")
    if getattr(tc, "n_envs", 1) < 1:
        errs.append(f"n_envs={tc.n_envs} 必须 >= 1")
    if getattr(tc, "rollout", 1) < 1:
        errs.append(f"rollout={tc.rollout} 必须 >= 1")
    if getattr(tc, "epochs", 1) < 1:
        errs.append(f"epochs={tc.epochs} 必须 >= 1")
    if getattr(tc, "minibatches", 1) < 1:
        errs.append(f"minibatches={tc.minibatches} 必须 >= 1")
    if getattr(tc, "vf_coef", 0.5) < 0 or getattr(tc, "ent_coef", 0.0) < 0:
        errs.append("vf_coef / ent_coef 不能为负")

    # 观测维度必须与网络一致
    if hasattr(env, "obs_dim") and hasattr(tc, "hidden"):
        pass  # hidden 是网络宽度，与 obs_dim 无关，这里只做存在性检查

    # minibatch 必须能整除 batch
    batch = getattr(tc, "n_envs", 0) * getattr(tc, "rollout", 0)
    mb = getattr(tc, "minibatches", 1)
    if mb and batch % mb != 0:
        warns.append(
            f"batch({batch}) 不能被 minibatches({mb}) 整除 —— "
            f"实际 minibatch 大小会是 {batch // mb}，末尾 {batch % mb} 个样本每轮被丢掉"
        )
    if mb and batch // mb < 8:
        warns.append(
            f"minibatch 仅 {batch // mb} 个样本，梯度噪声会很大。"
            f"建议 n_envs*rollout >= minibatches*64"
        )

    # ---- 致命：赛季 / 动作空间不匹配 ----
    if hasattr(env, "n_actions") and hasattr(cfg, "n_enemies"):
        expect = 2 + (cfg.n_enemies + 1) + 1
        if env.n_actions != expect:
            errs.append(f"动作维度不一致：环境 {env.n_actions} vs 配置推算 {expect}")
    if hasattr(env, "obs_dim") and hasattr(cfg, "n_enemies"):
        expect_obs = 6 + 8 + 7 * cfg.n_enemies + 1
        if env.obs_dim != expect_obs:
            errs.append(f"观测维度不一致：环境 {env.obs_dim} vs 配置推算 {expect_obs}")

    # ---- 警告：容易踩但不致命 ----
    if getattr(tc, "rollout", 128) * getattr(tc, "n_envs", 512) < 4096:
        warns.append(
            f"每次更新只采 {tc.n_envs * tc.rollout} 步，PPO 对批大小敏感，"
            f"建议 >= 8192（当前 {tc.n_envs}x{tc.rollout}）"
        )
    if getattr(cfg, "season", "2026") not in getattr(type(cfg), "SEASON_PROFILES", {}):
        errs.append(f"未知赛季 {cfg.season}")
    if getattr(cfg, "season", "") == "2027":
        warns.append("赛季 2027 为**预发布**版本，正式手册未公布，数值可能变化")

    # 观测延迟不能超过 rollout，否则整个 rollout 都在用同一份历史
    od = getattr(cfg, "obs_delay", 0)
    if od >= getattr(tc, "rollout", 128):
        warns.append(f"obs_delay({od}) >= rollout({tc.rollout})，观测延迟形同虚设")

    if errs:
        raise ConfigError("训练配置存在致命问题：\n  - " + "\n  - ".join(errs))
    return warns


# --------------------------------------------------------------------------
# 二层：训练中健康监控
# --------------------------------------------------------------------------
@dataclass
class HealthMonitor:
    """滚动记录训练指标，主动判定健康度。

    用法：
        hm = HealthMonitor()
        ...
        hm.observe(update=upd, loss=..., kl=..., grad_norm=..., reward=r,
                   fire_rate=..., hold_rate=..., sps=...)
        for warn in hm.snapshot(upd):   # 每 N 次更新调用一次
            print(warn)
    """

    window: int = 20
    _loss: list[float] = field(default_factory=list)
    _kl: list[float] = field(default_factory=list)
    _gnorm: list[float] = field(default_factory=list)
    _reward: list[float] = field(default_factory=list)
    _fire: list[float] = field(default_factory=list)
    _hold: list[float] = field(default_factory=list)
    _sps: list[float] = field(default_factory=list)
    _ent: list[float] = field(default_factory=list)
    _fired_warnings: set[str] = field(default_factory=set)
    nan_seen: bool = False

    # ---- 记录 ----
    def observe(self, *, loss=None, kl=None, grad_norm=None, reward=None,
                fire_rate=None, hold_rate=None, sps=None, entropy=None) -> list[str]:
        """记录一次更新的指标，立即返回本次触发的告警（去重后）。"""
        for name, val, buf in (
            ("loss", loss, self._loss), ("kl", kl, self._kl),
            ("grad_norm", grad_norm, self._gnorm), ("reward", reward, self._reward),
            ("fire_rate", fire_rate, self._fire), ("hold_rate", hold_rate, self._hold),
            ("sps", sps, self._sps), ("entropy", entropy, self._ent),
        ):
            if val is None:
                continue
            v = float(val)
            if not math.isfinite(v):
                self.nan_seen = True
                self._fired_warnings.add(f"非有限值:{name}")
                buf.append(float("nan"))
            else:
                buf.append(v)
            if len(buf) > self.window:
                buf.pop(0)

        out: list[str] = []
        for msg in self._check():
            key = msg.split("（")[0][:24]
            if key not in self._fired_warnings:
                self._fired_warnings.add(key)
                out.append(msg)
        return out

    # ---- 判定 ----
    def _finite(self, buf: list[float]) -> list[float]:
        return [x for x in buf if math.isfinite(x)]

    def _check(self) -> list[str]:
        msgs: list[str] = []
        loss = self._finite(self._loss)
        kl = self._finite(self._kl)
        gn = self._finite(self._gnorm)
        rew = self._finite(self._reward)
        fire = self._finite(self._fire)
        sps = self._finite(self._sps)
        ent = self._finite(self._ent)

        if self.nan_seen:
            msgs.append("出现 NaN/Inf（loss/kl/grad/reward 之一）。"
                        "常见原因：优点归一化除零、学习率过大、观测未归一化。"
                        "建议：先降 lr 到 1e-4，再检查观测范围")

        if len(kl) >= 5 and max(abs(k) for k in kl[-5:]) > 0.5:
            msgs.append(f"KL 过大（{max(abs(k) for k in kl[-5:]):.3f} > 0.5）——"
                        "策略更新步长太大，容易崩。建议降 lr 或减小 epochs")

        if len(gn) >= 5 and max(gn[-5:]) > 100:
            msgs.append(f"梯度范数过大（{max(gn[-5:]):.1f}）——"
                        "存在梯度爆炸风险。建议降低 lr 或收紧 max_grad_norm")

        if len(fire) >= self.window and all(f < 0.02 or f > 0.98 for f in fire[-self.window:]):
            msgs.append(f"开火率塌缩到常量（{np.mean(fire[-self.window:]):.3f}）——"
                        "策略不再探索开火决策。检查熵系数 ent_coef 是否过小，"
                        "或奖励里开火相关项是否被淹没")

        if len(ent) >= self.window and np.mean(ent[-self.window:]) < 0.05:
            msgs.append(f"策略熵接近 0（{np.mean(ent[-self.window:]):.3f}）——"
                        "探索完全停止，之后只会重复同一动作。建议提高 ent_coef")

        if len(rew) >= self.window:
            r = np.array(rew[-self.window:])
            if abs(r.mean()) < 1e-6 and r.std() < 1e-6:
                msgs.append("奖励连续为常量 —— 很可能是环境没被真正推进"
                            "（例如 done 后没有 auto_reset，或动作没生效）")

        if len(sps) >= 10:
            a = np.mean(sps[:5]); b = np.mean(sps[-5:])
            if a > 0 and (a - b) / a > 0.25:
                msgs.append(f"吞吐下降 {(a-b)/a*100:.0f}%（{a:,.0f} → {b:,.0f} 步/秒）——"
                            "疑似内存泄漏或系统被抢占。建议检查显存增长与后台程序")
        return msgs

    # ---- 快照 ----
    def snapshot(self, update: int) -> str:
        def fmt(buf: list[float], nd: int = 3) -> str:
            f = self._finite(buf)
            return f"{np.mean(f):.{nd}f}" if f else "n/a"
        return (f"  [健康] upd {update} | loss {fmt(self._loss, 4)} | "
                f"KL {fmt(self._kl, 4)} | |g| {fmt(self._gnorm, 2)} | "
                f"奖励 {fmt(self._reward, 4)} | 开火率 {fmt(self._fire)} | "
                f"占点率 {fmt(self._hold)} | 吞吐 {fmt(self._sps, 0)}")

    def is_healthy(self) -> bool:
        return not self.nan_seen and not self._fired_warnings
