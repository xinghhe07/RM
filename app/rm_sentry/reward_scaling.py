"""奖励归一化：PPO 在大奖励尺度下会发散，必须把奖励标定到稳定区间。

为什么必须有
------------
本项目实测过：加入"胜利点净变化 × 5.0"的目标导向奖励后，单步奖励跳到几十~几百
（击杀对方 = 20 VP × 5.0 = 100）。价值函数的回归目标随之到上百，于是：
    梯度范数 9.5 → 590      价值 loss 从 +0.2 → +294
    KL 0.008 → 1.24         策略更新崩坏      胜率掉到 0.23
这正是 PPO 最典型的发散模式。

标准做法是维护奖励的滑动标准差，把奖励归一化到 ~单位方差。
本模块同时保留**真实奖励**的统计量（用于对外报告"场均奖励"），
避免归一化后无法解读实际表现。
"""

from __future__ import annotations

import numpy as np
import torch


class RewardScaler:
    """奖励缩放：只做**固定尺度**的线性缩放，不减均值（SB3 VecNormalize 的做法）。

    依据（来自实现细节调研，含 SB3 源码与 Andrychowicz et al.）：
      - **不要对 reward 减均值**：会引入非平稳、破坏 bootstrap 一致性，且对
        "0 = 平局"的零和博弈没有意义。SB3 的做法是 `r / sqrt(Var(回报))`，只除不减。
      - **不要用 `clip_range_vf`（价值裁剪）**：SB3 明示它"依赖奖励尺度"，
        Andrychowicz 的结论是**有害**，Engstrom 也未发现其有帮助。
        根治办法是让 **value target 处于 O(1)**，而不是裁剪。
      - 因此本类只负责把奖励缩放到 O(1)，真正的方差控制交给
        **per-minibatch 优势归一化**（在 train_ppo 的更新循环里做）。

    为什么不减均值也能压住发散：
      把奖励缩放到 O(1) 后，回报 target 自然也是 O(1)，
      critic 不需要跨几个数量级去拟合 → 梯度不会爆炸。这是"根治尺度"而非"裁剪症状"。

    实现要点：
      - 用**折扣回报**的滑动标准差（更贴近 value target 的真实尺度），而非单步奖励
      - 预热期先原样返回，避免用噪声统计量缩放
      - 最后裁剪到 [-clip, clip]，防偶发极端值；注意这会削掉大额 payoff，
        若"一次击杀决定胜负"，要把 clip 调大（本项目 200 VP 量级 → clip=50 足够）
    """

    def __init__(self, eps: float = 1e-4, clip: float = 50.0, gamma: float = 0.99,
                 var_ema: float = 0.99, warmup: int = 3):
        self.eps = eps
        self.clip = clip
        self.gamma = gamma
        self.var_ema = var_ema
        self.warmup = warmup
        self._var = 1.0
        self._disc_return = 0.0
        self.count = 0

    def update_discounted(self, reward: torch.Tensor) -> None:
        """用折扣回报的均值平方估计尺度（SB3 风格，只维护方差，不减均值）。"""
        with torch.no_grad():
            r_mean = float(reward.mean())
        # 折扣回报的稳态方差估计（近似）：Var ≈ r² / (1-γ²)
        self._disc_return = self.gamma * self._disc_return + r_mean
        est = (self._disc_return ** 2) / max(1.0 - self.gamma ** 2, self.eps)
        self.count += 1
        if self.count == 1:
            self._var = max(est, self.eps)
        else:
            self._var = self.var_ema * self._var + (1 - self.var_ema) * est

    def update(self, reward: torch.Tensor) -> None:
        """兼容旧接口：直接按单步奖励方差更新。"""
        with torch.no_grad():
            v = float(reward.var(unbiased=False))
        self.count += 1
        if self.count == 1:
            self._var = max(v, self.eps)
        else:
            self._var = self.var_ema * self._var + (1 - self.var_ema) * v

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self._var, self.eps)))

    def scale(self, reward: torch.Tensor) -> torch.Tensor:
        """按固定尺度缩放（不减均值）。预热期原样返回。"""
        if self.count < self.warmup:
            return reward
        return (reward / (self.std + self.eps)).clamp(-self.clip, self.clip)

    def state(self) -> dict:
        return {"var": self._var, "std": self.std, "count": self.count}

    def load_state(self, s: dict) -> None:
        self._var = s.get("var", 1.0)
        self.count = s.get("count", 0)


class RunningMeanStd:
    """环境观测/回报的滑动均值方差（备用，用于诊断观测范围）。"""

    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        bm = x.mean(axis=0)
        bv = x.var(axis=0)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)
