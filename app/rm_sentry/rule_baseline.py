"""规则基线哨兵 —— 给 RL 提供对照，同时是**兜底层**。

为什么必须有
-----------
没有基线，"RL 胜率 0.9" 没有意义（AGENTS.md 铁律 R6）。
它同时是 `工程红线.md` 里的**第 2 层兜底**：RL 挂了要能顶上，所以必须真的能打。

设计依据（**全部借鉴开源，未自研**）
--------------------------------
主要来源：pb2025_sentry_behavior（47★, **Apache-2.0 可商用**）
  behavior_trees/rmul_2025.xml —— 哨兵决策行为树，已解码见 `开源借鉴.md` §2.1

抄来的三条：
  1. 热量门控 heat_max = 上限的 87.5%（原值 350 / 400）—— **留余量，不贴上限**
  2. 失败兜底结构：主行为失败 -> 走安全动作（原树：回原点）
  3. 行为树组织方式：Fallback + ReactiveSequence（状态检查在前，动作在后）

⚠️ **不能照抄的一条：血量门控 `hp_min=200`**（踩坑记录）
  pb2025 的血量阈值 200 是相对"能回满血"的场景；而本项目环境复活血量只有
  上限的 20%（80），若阈值取 200 会导致**复活即撤退、永久不开火**。
  详见 `HP_RETREAT_RATIO` 处的注释。
  ⇒ **借鉴阈值时必须核对自己的场景是否可比**（这是本条的通用教训）。

次要来源（**仅读思路，未复制代码**）：
  SnowStain/Artinx_Simulator_for_RM26UC  control/behavior_trees/sentry_btcpp.xml
    + decision_plugins/definitions/（emergency_retreat 等）
    ⚠️ 该仓库无 LICENSE，只读思路

⚠️ 无开源依据的自研项（本项目环境特有，公开实现里不存在）：
  - "双方同在控制区 = 无人占领"（RMUL 规则 §3.3.6 先到先得）的应对：
    原树没有这个概念（RMUL 2025 的中心增益点机制不同）。
    本实现选择"卡位"策略：在区内贴近中心待着，等对方离开即接管。

行为树结构（与 pb2025 同构：Fallback + ReactiveSequence）
-------------------------------------------------------
  check_dead          死亡/复活中 -> 不动、不开火
  check_health        HP < 50%     -> **撤退**（远离控制区，先活下来）
  main_sequence       否则         -> 进控制区驻留 + 打最近可见目标 + 开火
    ├ 热量 > 87.5% -> 停火（防热量锁定）
    ├ 弹药 = 0     -> 停火
    └ 无可见目标   -> 只移动
"""

from __future__ import annotations

import torch

from rm_sentry.rmul_3v3_env import E, RMUL3v3Config, S


class RuleSentry:
    """规则哨兵。接口与 ActorCritic.act 对齐（返回 move/target/fire）。"""

    # ---- 借鉴 pb2025_sentry_behavior 的热量阈值（见模块 docstring）----
    HEAT_STOP_RATIO = 0.875     # 原 IsStatusOK heat_max=350 / 上限 400
    #
    # ⚠️ 血量门控**不能照抄 pb2025 的 hp_min=200**（踩过的坑）：
    #   他们能靠补给把血回满，所以 200 是"该收手了"的合理阈值；
    #   而本项目环境**复活血量只有上限的 20%（= 80）**，若阈值取 200，
    #   则复活那一刻就已经"低血量"，策略会永久处于撤退状态：
    #   实测后果 —— 开火率仅 5.7%、低血量占比 92.8%、每局输出只有 200、
    #   净胜点 −103（比不加门控还差）。
    #   ⇒ 阈值必须**低于复活血量**才有意义。取上限的 10%（= 40 < 80）。
    HP_RETREAT_RATIO = 0.10

    def __init__(self, cfg: RMUL3v3Config, *,
                 hp_retreat_ratio: float = HP_RETREAT_RATIO,
                 heat_stop_ratio: float = HEAT_STOP_RATIO,
                 zone_deadband: float = 0.35,
                 fire_range_ratio: float = 0.9,
                 name: str = "rule"):
        self.cfg = cfg
        self.hp_retreat_ratio = hp_retreat_ratio
        self.heat_stop_ratio = heat_stop_ratio
        # 距中心小于此值即"已就位"，停止微调（避免在中心来回抖）
        self.zone_deadband = zone_deadband
        self.fire_range_ratio = fire_range_ratio
        self.name = name

        # 自检：血量门控必须低于复活血量，否则会永久撤退（见上面注释）
        respawn_hp = cfg.sentry_hp * cfg.respawn_hp_ratio
        if self.hp_retreat_ratio * cfg.sentry_hp >= respawn_hp:
            raise ValueError(
                f"hp_retreat_ratio={self.hp_retreat_ratio} 对应的血量 "
                f"{self.hp_retreat_ratio*cfg.sentry_hp:.0f} >= 复活血量 {respawn_hp:.0f}，"
                f"会导致复活后永久撤退。必须让阈值 < 复活血量。")

    @torch.no_grad()
    def act(self, obs, mask, env) -> dict:
        """返回动作 dict。需要 env 引用：掩码只说"谁能看见"，
        而"该往哪走"需要自机/敌人真实坐标（RL 从观测学这些，规则直接读状态）。"""
        c = self.cfg
        n = env.n
        dev = env.device
        zero2 = torch.zeros(n, 2, device=dev)

        cx, cy = c.field_w / 2.0, c.field_h / 2.0
        sx, sy = env.sentry[:, S.X], env.sentry[:, S.Y]
        to_cx, to_cy = cx - sx, cy - sy
        dist_center = torch.hypot(to_cx, to_cy)
        unit_to_center = torch.stack(
            [to_cx / dist_center.clamp_min(1e-6), to_cy / dist_center.clamp_min(1e-6)], -1)

        # ---- check_dead：死亡/复活中 -> 不动、不开火 ----
        alive = env.sentry[:, S.DEAD] <= 0

        # ---- check_health：血量过低 -> 撤退（借鉴 hp_min=50%）----
        # 撤退方向 = 远离控制区中心（沿单位向量反向）
        low_hp = alive & (env.sentry[:, S.HP] < c.sentry_hp * self.hp_retreat_ratio)
        retreat = -unit_to_center

        # ---- main：进控制区并驻留 ----
        want_move = alive & ~low_hp & (dist_center > self.zone_deadband)
        advance = unit_to_center * want_move.float().unsqueeze(-1)

        move = torch.where(low_hp.unsqueeze(-1), retreat, advance)
        move = torch.where(alive.unsqueeze(-1), move, zero2)

        # ---- 选最近可见目标 ----
        # ⚠️ 现算可见性，不用 env._vis（那是 _build_obs 的副产物，可能是上一帧的旧值，
        #    而且新环境上可能尚未初始化 —— 见 AGENTS.md R2c）
        vis = env._los_blocked()                      # [n, k]，列 1..n_enemies 为真实敌人
        ex, ey = env.enemies[:, :, E.X], env.enemies[:, :, E.Y]
        dist_e = torch.hypot(ex - sx.unsqueeze(-1), ey - sy.unsqueeze(-1))
        valid = vis & (env.enemies[:, :, E.ALIVE] > 0.5)
        valid[:, 0] = False                           # 列 0 是自机槽，不可选
        dist_masked = torch.where(valid, dist_e, torch.full_like(dist_e, 1e6))
        nearest = dist_masked.argmin(dim=1)
        any_visible = valid.any(dim=1)
        nearest_dist = dist_masked.gather(1, nearest.unsqueeze(1)).squeeze(1)

        # target 动作：2 = 不交战；3+m 对应槽 m（见 action_mask 映射说明）
        target = torch.where(any_visible, nearest + 2, torch.full_like(nearest, 2))

        # ---- 开火门控：热量 / 弹药 / 射程 / 可见性 ----
        heat = torch.maximum(env.sentry[:, S.HEAT1], env.sentry[:, S.HEAT2])
        can = (any_visible & alive & ~low_hp
               & (heat < c.heat_limit * self.heat_stop_ratio)
               & (env.sentry[:, S.AMMO] > 0)
               & (nearest_dist < c.shot_range * self.fire_range_ratio))
        fire = can.float()

        # ---- 兜底：规则也必须尊重掩码，否则与 RL 口径不一致、不可比 ----
        tgt_ok = torch.gather(mask, 1, target.unsqueeze(1)).squeeze(1)
        target = torch.where(tgt_ok, target, torch.full_like(target, 2))
        fire = fire * tgt_ok.float()

        return {"move": move, "target": target, "fire": fire}


def evaluate_rule(cfg: RMUL3v3Config, dev, rule: RuleSentry,
                  n_envs: int = 256, seed: int = 20261007,
                  max_steps: int | None = None) -> dict:
    """用与 `train_ppo.evaluate` **完全相同的口径**评估规则策略。

    不直接复用 evaluate()：那里调用 net.act(obs, mask)，而规则策略需要读 env 状态。
    为保证可比，逐行照抄同一套统计定义。
    """
    from rm_sentry.rmul_3v3_env import BatchedRMUL3v3
    from rm_sentry.train_ppo import to_env_action

    env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=dev, seed=seed)
    env.reset()
    mask = env.action_mask()
    done_n = 0
    acc = {"vp": 0.0, "vp_enemy": 0.0, "dmg_out": 0.0, "dmg_in": 0.0,
           "hold": 0.0, "win": 0.0, "lose": 0.0}
    steps = 0
    limit = max_steps if max_steps is not None else int(cfg.match_time / cfg.dt) * 2

    for _ in range(limit):
        a = to_env_action(rule.act(None, mask, env), env.n_actions, dev)
        obs, rew, done, info = env.step(a)
        acc["hold"] += float(info["hold_ratio"].sum())
        steps += 1
        if bool(done.any()):
            k = done
            for key, ik in (("vp", "vp"), ("vp_enemy", "vp_enemy"),
                            ("dmg_out", "dmg_out"), ("dmg_in", "dmg_in"),
                            ("win", "win"), ("lose", "lose")):
                acc[key] += float(info[ik][k].sum())
            done_n += int(k.sum())
        env.auto_reset(done)
        mask = env.action_mask()
        if done_n >= n_envs:
            break

    d = max(done_n, 1)
    return {
        "episodes": done_n,
        "vp": acc["vp"] / d, "vp_enemy": acc["vp_enemy"] / d,
        "dmg_out": acc["dmg_out"] / d, "dmg_in": acc["dmg_in"] / d,
        "win_rate": acc["win"] / d, "lose_rate": acc["lose"] / d,
        "hold_rate": acc["hold"] / max(steps * n_envs, 1),
        "vp_margin": (acc["vp"] - acc["vp_enemy"]) / d,
    }
