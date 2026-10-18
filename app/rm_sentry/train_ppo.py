"""PPO 训练脚本（RMUL 3v3 哨兵，2026 规则）。

设计要点
--------
1. **混合动作空间**：移动 2 维连续（高斯）+ 目标选择 4 类（离散，带掩码）+ 开火 1 维伯努利。
   用三个头拼成一个策略，log_prob 相加，PPO 目标函数无需改动。
2. **掩码处理**：不可见目标在 logits 上置 -1e9，保证采样与 log_prob 都合法。
3. **训练/部署不对称**（关键）：仿真用真值算奖励，但策略只吃"哨兵能感知到"的观测。
   这正是 Sim2Real 的标准做法，也使得部署时策略不需要任何特权信息。
4. **设备无关**：同一份代码在 cuda 与 cpu 上都能跑，便于在本机训练、在赛场 CPU 机器上验证。

用法
----
    # 本机用 GPU 训练
    python -m rm_sentry.train_ppo --device cuda --n-envs 512 --total-steps 2000000

    # 在赛场那台只有 CPU 的机器上验证/小规模训练
    python -m rm_sentry.train_ppo --device cpu --n-envs 64 --total-steps 200000

    # 只做速度基准，不训练
    python -m rm_sentry.train_ppo --bench-only
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S
from rm_sentry.guards import ConfigError, HealthMonitor, validate_train_cfg
from rm_sentry.reward_scaling import RewardScaler

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# --------------------------------------------------------------------------
# 策略网络：三头混合动作
# --------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_targets: int, hidden: int = 256):
        super().__init__()
        self.n_targets = n_targets          # 目标选择类别数 = n_enemies + 1（含"不交战"）
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.move_mu = nn.Linear(hidden, 2)
        self.move_logstd = nn.Parameter(torch.full((2,), -0.5))
        self.target_logits = nn.Linear(hidden, n_targets)
        self.fire_logit = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.trunk(obs)
        mu = self.move_mu(h)
        logstd = self.move_logstd.clamp(-4.0, 1.0).expand_as(mu)
        tlog = self.target_logits(h)
        if mask is not None:
            # mask: [n, n_actions]；目标通道位于 [2 : 2+n_targets]
            tm = mask[:, 2:2 + self.n_targets]
            tlog = torch.where(tm, tlog, torch.full_like(tlog, -1e9))
        flog = self.fire_logit(h).squeeze(-1)
        return mu, logstd, tlog, flog, self.value(h).squeeze(-1)

    # ---- 采样 ----
    def act(self, obs, mask):
        mu, logstd, tlog, flog, val = self.forward(obs, mask)
        std = logstd.exp()
        move = torch.distributions.Normal(mu, std).sample()
        # 连续动作裁剪到 [-1, 1]（与动作约定一致）
        move_c = move.clamp(-1.0, 1.0)
        tdist = torch.distributions.Categorical(logits=tlog)
        target = tdist.sample()
        fdist = torch.distributions.Bernoulli(logits=flog)
        fire = fdist.sample()
        # 注意：用裁剪后的动作算 log_prob（否则与真实执行的动作不一致）
        logp_move = torch.distributions.Normal(mu, std).log_prob(move_c).sum(-1)
        logp = logp_move + tdist.log_prob(target) + fdist.log_prob(fire)
        return dict(move=move_c, target=target, fire=fire,
                    logp=logp, value=val), (mu, logstd, tlog, flog)

    # ---- 评估（训练更新用） ----
    def evaluate(self, obs, mask, move, target, fire):
        mu, logstd, tlog, flog, val = self.forward(obs, mask)
        std = logstd.exp()
        logp_move = torch.distributions.Normal(mu, std).log_prob(move).sum(-1)
        tdist = torch.distributions.Categorical(logits=tlog)
        fdist = torch.distributions.Bernoulli(logits=flog)
        logp = logp_move + tdist.log_prob(target) + fdist.log_prob(fire)
        entropy = (torch.distributions.Normal(mu, std).entropy().sum(-1)
                   + tdist.entropy() + fdist.entropy())
        return logp, entropy, val


# --------------------------------------------------------------------------
# 向量化 rollout：把策略动作拼成环境要的张量
# --------------------------------------------------------------------------
def to_env_action(policy_act, n_actions: int, device) -> torch.Tensor:
    n = policy_act["move"].shape[0]
    a = torch.zeros(n, n_actions, device=device)
    a[:, 0:2] = policy_act["move"]
    a[:, 2] = policy_act["target"].float()
    a[:, 3] = policy_act["fire"]
    return a


def evaluate(net, cfg, dev, n_envs: int = 256, seed: int = 12345,
             random_policy: bool = False, max_steps: int | None = None) -> dict:
    """确定性评估：固定种子、固定对局数，报告"是否真的变强"。

    为什么必须有这个函数：
      训练日志里的 loss / KL 只说明优化器在工作，**不能说明策略变强**。
      判定必须靠闭环对局（离线评估已被证明不可信）。这里用固定种子跑完整对局，
      所以两次评估之间的差异可归因于策略本身。

    ⚠️ 成本警告：一局是 6000 步（300s / 0.05s）。若策略很少战死，就必须等到超时，
      总步数 = n_envs × 每局步数，很容易吃掉几百万步。训练中频繁调用会严重拖慢整体。
      因此：
        - 训练中的"趋势"看 `RollingEpisodeStats`（零额外开销）
        - 本函数只在需要**确定性、可比较**的结论时调用（如每 50 次更新、或训练结束后）
    """
    env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=dev, seed=seed)
    obs = env.reset()
    mask = env.action_mask()
    finished = 0
    acc = {"vp": 0.0, "vp_enemy": 0.0, "dmg_out": 0.0, "dmg_in": 0.0,
           "hold": 0.0, "hold_steps": 0, "total_steps": 0, "win": 0.0, "lose": 0.0}
    step_limit = max_steps if max_steps is not None else int(cfg.match_time / cfg.dt) * 2
    for _ in range(step_limit):
        with torch.no_grad():
            if random_policy:
                n = obs.shape[0]
                mv = torch.rand(n, 2, device=dev) * 2 - 1
                logits = torch.where(mask[:, 2:2 + cfg.n_enemies + 1],
                                     torch.zeros(n, cfg.n_enemies + 1, device=dev),
                                     torch.full((n, cfg.n_enemies + 1), -1e9, device=dev))
                tg = torch.distributions.Categorical(logits=logits).sample()
                fr = (torch.rand(n, device=dev) < 0.5).float()
                pact = {"move": mv, "target": tg, "fire": fr}
            else:
                pact, _ = net.act(obs, mask)
        a = to_env_action(pact, env.n_actions, dev)
        obs, rew, done, info = env.step(a)
        # ⚠️ 占点率必须**逐步累积**（不能只在 done 时取）：
        #   评估结果里若没有占点率，就无法回答"胜率 1.00 是靠占点还是靠打人"
        #   —— 而这正是判断策略是否真的完成战术目标的关键。
        acc["hold"] += float(info["hold_ratio"].sum())
        acc["hold_steps"] += 1
        if bool(done.any()):
            k = done
            acc["vp"] += float(info["vp"][k].sum())
            acc["vp_enemy"] += float(info["vp_enemy"][k].sum())
            acc["dmg_out"] += float(info["dmg_out"][k].sum())
            acc["dmg_in"] += float(info["dmg_in"][k].sum())
            acc["win"] += float(info["win"][k].sum())
            acc["lose"] += float(info["lose"][k].sum())
            finished += int(k.sum())
        env.auto_reset(done)
        mask = env.action_mask()
        if finished >= n_envs:
            break
    d = max(finished, 1)
    return {
        "episodes": finished,
        "vp": acc["vp"] / d,
        "vp_enemy": acc["vp_enemy"] / d,
        "dmg_out": acc["dmg_out"] / d,
        "dmg_in": acc["dmg_in"] / d,
        "win_rate": acc["win"] / d,
        "lose_rate": acc["lose"] / d,
        # 占点率：本次评估全部采样步里"我方持有控制区"的比例。
        # 用它区分"靠占点赢"与"靠对枪赢" —— 前者才是本赛题的战术目标。
        "hold_rate": acc["hold"] / max(acc["hold_steps"] * n_envs, 1),
        # 净胜点：2026 是"扣分制"，比较双方剩余胜利点差更直观
        "vp_margin": (acc["vp"] - acc["vp_enemy"]) / d,
    }


class RollingEpisodeStats:
    """零额外开销的对局统计：复用训练 rollout 里已经算出来的 done/info。

    为什么需要：
      确定性 evaluate() 很贵（一局 6000 步）。而训练中想知道"是否在变好"，
      完全可以顺手收集训练时自然结束的那些对局 —— 不花额外算力。
    局限：这些对局是"当前策略"产生的，不同更新之间可比性略弱于固定种子评估，
          但足以看趋势。
    """

    def __init__(self, horizon: int = 60):
        self.horizon = horizon
        self.buf: list[dict] = []

    def update(self, done: torch.Tensor, info: dict) -> None:
        if not bool(done.any()):
            return
        k = done
        n = max(int(k.sum()), 1)
        self.buf.append({
            "n": n,
            "vp": float(info["vp"][k].mean()),
            "vp_enemy": float(info["vp_enemy"][k].mean()),
            "dmg_out": float(info["dmg_out"][k].mean()),
            "win": float(info["win"][k].mean()),
            "lose": float(info["lose"][k].mean()),
        })
        if len(self.buf) > self.horizon:
            self.buf.pop(0)

    def summary(self) -> dict | None:
        if not self.buf:
            return None
        res = sum(b["n"] for b in self.buf)
        return {
            "episodes": res,
            "vp": sum(b["vp"] * b["n"] for b in self.buf) / res,
            "vp_enemy": sum(b["vp_enemy"] * b["n"] for b in self.buf) / res,
            "dmg_out": sum(b["dmg_out"] * b["n"] for b in self.buf) / res,
            "win_rate": sum(b["win"] * b["n"] for b in self.buf) / res,
            "lose_rate": sum(b["lose"] * b["n"] for b in self.buf) / res,
        }


@dataclass
class TrainCfg:
    n_envs: int = 512
    rollout: int = 128          # 每个环境每次采样的步数
    total_steps: int = 2_000_000
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03      # KL 早停阈值（0 = 关闭）。超过 2x 就停止本轮的后续 epoch
                                 # 依据：Spinning Up 明确"clip 不是硬约束，我们靠 early stopping"
    per_minibatch_adv_norm: bool = True  # per-minibatch 优势归一化（ICLR 37 细节 detail 7）
    hidden: int = 256
    seed: int = 0


def train(args) -> dict:
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = RMUL3v3Config(season=args.season)
    tc = TrainCfg(n_envs=args.n_envs, rollout=args.rollout,
                  total_steps=args.total_steps, hidden=args.hidden)

    env = BatchedRMUL3v3(cfg, n_envs=tc.n_envs, device=dev, seed=tc.seed)
    obs = env.reset()
    mask = env.action_mask()

    net = ActorCritic(env.obs_dim, cfg.n_enemies + 1, tc.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=tc.lr, eps=1e-5)

    n_actions = env.n_actions
    batch = tc.n_envs * tc.rollout
    updates = max(1, tc.total_steps // batch)

    # ---- 启动时配置校验：花 0 秒拦住会白跑几小时的配置错误 ----
    try:
        warns = validate_train_cfg(tc, cfg, env)
    except ConfigError as e:
        print(f"\n[配置错误] {e}\n")
        raise SystemExit(2)
    for w in warns:
        print(f"  \033[33m[配置警告]\033[0m {w}")

    print(f"设备 {dev} | 赛季 {cfg.season} | {cfg.n_barrels} 管 "
          f"| 并行 {tc.n_envs} x {tc.rollout} = {batch} 步/更新 x {updates} 次")
    print(f"网络参数量: {sum(p.numel() for p in net.parameters()):,}")

    hm = HealthMonitor(window=20)
    roll = RollingEpisodeStats(horizon=60)   # 零额外开销的对局趋势
    scaler = RewardScaler()                  # 奖励归一化（防发散，见 reward_scaling.py）
    hist = []
    t_start = time.perf_counter()
    total = 0

    for upd in range(1, updates + 1):
        # ---------------- 采样 ----------------
        o_buf, m_buf, mv_buf, tg_buf, fr_buf = [], [], [], [], []
        lp_buf, v_buf, r_buf, d_buf = [], [], [], []
        # 逐步诊断（不依赖对局结束）
        step_diag = {"reward": 0.0, "hold": 0.0, "fire": 0.0, "dmg": 0.0, "n": 0}
        # 对局级累计（一局 6000 步，短训练可能为 0，属正常）
        ep_acc = {"vp": 0.0, "vp_enemy": 0.0, "dmg_out": 0.0,
                  "dmg_in": 0.0, "win": 0.0, "lose": 0.0}
        finished = 0
        t_roll = time.perf_counter()

        for _ in range(tc.rollout):
            with torch.no_grad():
                pact, _ = net.act(obs, mask)
            a = to_env_action(pact, n_actions, dev)
            o_buf.append(obs); m_buf.append(mask)
            mv_buf.append(pact["move"]); tg_buf.append(pact["target"])
            fr_buf.append(pact["fire"]); lp_buf.append(pact["logp"])
            v_buf.append(pact["value"])
            nobs, rew, done, info = env.step(a)
            r_buf.append(rew); d_buf.append(done.float())
            # 逐步诊断量（不必等对局结束，所以短训练也能看到信号）
            step_diag["reward"] += float(rew.mean())
            step_diag["hold"] += float(info["hold_ratio"].mean())
            step_diag["fire"] += float(pact["fire"].mean())
            step_diag["dmg"] += float(env.sentry[:, S.DMG_OUT].mean())
            step_diag["n"] += 1
            if bool(done.any()):
                finished += int(done.sum())
                roll.update(done, info)
                for k in ("vp", "vp_enemy", "dmg_out", "dmg_in", "win", "lose"):
                    ep_acc[k] += float(info[k][done].sum())
            obs = nobs
            env.auto_reset(done)
            mask = env.action_mask()

        total += batch
        t_sample = time.perf_counter() - t_roll

        # ---------------- 计算 GAE ----------------
        with torch.no_grad():
            _, _, _, _, last_v = net.forward(obs, mask)
        O = torch.stack(o_buf)          # [T, N, obs]
        M = torch.stack(m_buf)
        MV = torch.stack(mv_buf); TG = torch.stack(tg_buf); FR = torch.stack(fr_buf)
        LP = torch.stack(lp_buf); V = torch.stack(v_buf)
        R = torch.stack(r_buf); D = torch.stack(d_buf)

        # ---- 奖励缩放（让 value target 落在 O(1)，这是防发散的根治手段）----
        # 依据（含 SB3 源码与 Andrychowicz et al.）：
        #   * **只缩放、不减均值** —— 减均值引入非平稳、破坏 bootstrap 一致性，
        #     且对"0 = 平局"的零和博弈毫无意义。
        #   * **不要用价值裁剪（clip_range_vf）** —— SB3 明示其"依赖奖励尺度"，
        #     Andrychowicz 结论为有害。根治是让 target 处于 O(1)，而非裁剪。
        #   * 方差控制交给 per-minibatch 优势归一化（在下面的更新循环里）。
        scaler.update_discounted(R)
        Rn = scaler.scale(R)

        adv = torch.zeros_like(R)
        last_gae = torch.zeros(tc.n_envs, device=dev)
        for t in reversed(range(tc.rollout)):
            next_v = last_v if t == tc.rollout - 1 else V[t + 1]
            nonterminal = 1.0 - D[t]
            # 用**归一化后**的奖励算 GAE —— 价值头拟合的就是归一化回报
            delta = Rn[t] + tc.gamma * next_v * nonterminal - V[t]
            last_gae = delta + tc.gamma * tc.lam * nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + V

        # 展平（注意 V 也要一起展平：它是 [T, N]，而 newv 是展平后的 [T*N]）
        f = lambda x: x.reshape(-1, *x.shape[2:])
        O, M, MV, TG, FR, LP, V, ADV, RET = map(f, (O, M, MV, TG, FR, LP, V, adv, ret))
        # 注意：优势**不在这里**做全局归一化。
        # 依据（ICLR 2022《37 个实现细节》detail 7）：应该在**每个 minibatch 内**
        # 单独归一化，而不是对整个 batch 做一次。全局归一化会让 minibatch 之间的
        # 优势尺度不一致，是常见的不稳定来源。
        # 全局归一化只作为兜底（若关闭 per-minibatch 模式）。
        if not tc.per_minibatch_adv_norm:
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # ---------------- PPO 更新 ----------------
        idx = torch.randperm(batch, device=dev)
        mb_size = batch // tc.minibatches
        clip_fracs, kls, losses = [], [], []
        grad_norms: list[float] = []
        early_stopped = False
        t_upd = time.perf_counter()
        for ep in range(tc.epochs):
            if early_stopped:
                break
            for start in range(0, batch, mb_size):
                j = idx[start:start + mb_size]
                newlp, ent, newv = net.evaluate(O[j], M[j], MV[j], TG[j], FR[j])
                # ---- per-minibatch 优势归一化（ICLR 37 细节 detail 7）----
                a = ADV[j]
                if tc.per_minibatch_adv_norm:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (newlp - LP[j]).exp()
                s1 = ratio * a
                s2 = ratio.clamp(1 - tc.clip, 1 + tc.clip) * a
                pg = -torch.min(s1, s2).mean()
                # 价值损失：**不加 value clipping**。
                # 依据：SB3 明确说 clip_range_vf "依赖奖励尺度"，Andrychowicz 的结论是
                # 有害、Engstrom 也未发现其有帮助。正确做法是让 value target 处于 O(1)
                # （由 reward_scaling 的固定尺度缩放 + 优势归一化共同保证）。
                vf = 0.5 * (newv - RET[j]).pow(2).mean()
                loss = pg + tc.vf_coef * vf - tc.ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # 记录梯度范数（clip_grad_norm_ 返回的就是裁剪前的总范数）
                gn = float(nn.utils.clip_grad_norm_(net.parameters(), tc.max_grad_norm))
                grad_norms.append(gn)
                opt.step()
                with torch.no_grad():
                    k = float((LP[j] - newlp).mean())
                    clip_fracs.append(((ratio - 1).abs() > tc.clip).float().mean().item())
                    kls.append(k)
                    losses.append(loss.item())
                # KL 早停：本 epoch 的平均 KL 超阈值就停止后续 epoch，
                # 防止策略一步走太远而崩（这是 PPO 论文与 SB3 都采用的保护）
                if tc.target_kl and len(kls) >= tc.minibatches:
                    recent = np.mean(kls[-tc.minibatches:])
                    if recent > tc.target_kl * 2:
                        early_stopped = True
                        break
        t_update = time.perf_counter() - t_upd

        sps = batch / (t_sample + t_update)
        nd = max(step_diag["n"], 1)
        row = dict(update=upd, steps=total, sps=round(sps),
                   sample_s=round(t_sample, 3), update_s=round(t_update, 3),
                   loss=round(float(np.mean(losses)), 4),
                   approx_kl=round(float(np.mean(kls)), 5),
                   clip_frac=round(float(np.mean(clip_fracs)), 4),
                   r_step=round(step_diag["reward"] / nd, 4),
                   r_norm_std=round(scaler.std, 4),
                   hold_rate=round(step_diag["hold"] / nd, 4),
                   fire_rate=round(step_diag["fire"] / nd, 4),
                   dmg_out=round(step_diag["dmg"] / nd, 1),
                   episodes=finished,
                   roll=roll.summary())
        hist.append(row)
        if upd % max(1, updates // 20) == 0 or upd == 1:
            print(f"  upd {upd:4d}/{updates} | {sps:7,.0f} 步/秒 | "
                  f"loss {row['loss']:+.3f} | KL {row['approx_kl']:+.4f} | "
                  f"r/步 {row['r_step']:+.3f} (σ{row['r_norm_std']:.2f}) | "
                  f"占点率 {row['hold_rate']:.3f} | "
                  f"开火率 {row['fire_rate']:.3f} | 累计伤害 {row['dmg_out']:.0f}")

        # ---- 运行时护栏：主动判定健康度，异常时给中文诊断 ----
        for msg in hm.observe(loss=float(np.mean(losses)),
                              kl=float(np.mean(kls)),
                              grad_norm=float(np.mean(grad_norms)) if grad_norms else None,
                              reward=step_diag["reward"] / nd,
                              fire_rate=step_diag["fire"] / nd,
                              hold_rate=step_diag["hold"] / nd,
                              sps=sps):
            print(f"  \033[31m[护栏告警]\033[0m {msg}")
        if upd % max(1, updates // 10) == 0 and upd > 1:
            print(hm.snapshot(upd))
        # 滚动对局趋势（零额外开销，训练中看趋势用这个）
        rs = row.get("roll")
        if rs and upd % max(1, updates // 10) == 0 and rs["episodes"] > 0:
            print(f"  [对局趋势] 已结束 {rs['episodes']} 局 | 胜率 {rs['win_rate']:.2f} | "
                  f"败率 {rs['lose_rate']:.2f} | 场均输出 {rs['dmg_out']:.0f} | "
                  f"净胜点 {rs['vp'] - rs['vp_enemy']:+.1f}")

        # ---------------- 确定性评估（可选，较贵） ----------------
        # 注意：一局 6000 步，评估成本 = eval_envs × 每局步数，很容易吃掉几百万步。
        # 训练中的趋势请看上面的 [对局趋势]（零额外开销）；这里只在显式指定
        # --eval-every 时调用，用于拿到**固定种子、可比较**的结论。
        if args.eval_every and upd % args.eval_every == 0:
            ev = evaluate(net, cfg, dev, n_envs=args.eval_envs, seed=12345,
                          max_steps=args.eval_max_steps)
            row["eval"] = ev
            print(f"       [评估] 对局 {ev['episodes']} 局 | 胜率 {ev['win_rate']:.2f} | "
                  f"败率 {ev['lose_rate']:.2f} | 净胜点 {ev['vp_margin']:+.1f} | "
                  f"场均输出 {ev['dmg_out']:.0f} / 承伤 {ev['dmg_in']:.0f}")
            hist[-1] = row

    wall = time.perf_counter() - t_start
    print(f"\n完成：{total:,} 步，用时 {wall/60:.1f} 分钟，平均 {total/wall:,.0f} 步/秒")

    # 保存
    RUNS_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"{cfg.season}_{dev.type}"
    out = RUNS_DIR / tag
    out.mkdir(exist_ok=True)
    torch.save({"model": net.state_dict(), "obs_dim": env.obs_dim,
                "n_targets": cfg.n_enemies + 1, "hidden": tc.hidden,
                "season": cfg.season, "train_cfg": asdict(tc)},
               out / "policy.pt")
    (out / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps({"season": cfg.season, "profile": cfg.profile_source,
                    **asdict(tc)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存到 {out}")
    return {"wall_s": wall, "sps": total / wall, "updates": updates}


def bench(args) -> None:
    """设备基准：单独测环境吞吐 与 策略前向/反向开销。"""
    dev = torch.device(args.device)
    cfg = RMUL3v3Config(season=args.season)
    print("=" * 68)
    print(f"训练吞吐基准  [设备 {dev}]  赛季 {cfg.season}")
    print("=" * 68)
    for n_envs in (64, 256, 1024, 4096):
        try:
            env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=dev, seed=0)
            env.reset()
            net = ActorCritic(env.obs_dim, cfg.n_enemies + 1).to(dev)
            # CPU 上大规模会慢很多，按规模缩短测量步数，避免基准本身跑太久
            steps = 100 if n_envs <= 1024 else 25
            a = torch.zeros(n_envs, env.n_actions, device=dev)
            # 环境单独吞吐
            t0 = time.perf_counter()
            for _ in range(steps):
                env.step(a)
            t_env = time.perf_counter() - t0
            # 含策略前向
            obs = env.reset(); mask = env.action_mask()
            t0 = time.perf_counter()
            for _ in range(steps):
                with torch.no_grad():
                    pact, _ = net.act(obs, mask)
                o, r, d, i = env.step(to_env_action(pact, env.n_actions, dev))
                obs = o; mask = env.action_mask(); env.auto_reset(d)
            t_full = time.perf_counter() - t0
            print(f"  n_envs={n_envs:5d} | 纯环境 {n_envs*steps/t_env:>10,.0f} 步/秒"
                  f" | 含策略 {n_envs*steps/t_full:>10,.0f} 步/秒")
            del env, net
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  n_envs={n_envs:5d} | 失败: {str(e)[:70]}")
    print("\n说明：纯环境吞吐 ≈ 训练上限；含策略那列才是 PPO 实际可达的上限。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--n-envs", type=int, default=512)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="每多少次更新做一次确定性评估（0 = 关闭，默认；"
                         "评估很贵：成本 ≈ eval_envs × 一局步数）")
    ap.add_argument("--eval-envs", type=int, default=128)
    ap.add_argument("--eval-max-steps", type=int, default=None,
                    help="评估时每局最多跑多少步（默认 2 局时长 = 12000 步；调小可显著省钱）")
    args = ap.parse_args()
    if args.bench_only:
        bench(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
