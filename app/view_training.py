"""把 rm_sentry 的 2D 环境实时渲染到 MuJoCo 窗口 —— 用来"看着训练/对局跑"。

为什么这样做
-----------
战术环境本身是**标量数学**（无几何体），所以它没有可视化的东西。
但"看不到过程"会导致无法判断策略在干什么（本项目的很多误判都源于此：
读数字看不出"哨兵 69% 时间是死的"、"92% 在区时间是双方同在"）。

本脚本把环境的真实状态（位置/朝向/血量/热量/控制区归属/胜利点）
**镜像**到 MuJoCo 场景里，用 GUI 实时显示。数据完全来自 env，不是另写一套。

用法：
    python view_training.py                       # 规则基线 2 局
    python view_training.py --policy random       # 随机策略
    python view_training.py --ckpt runs/diag/baseline_upd010.pt
    python view_training.py --speed 4             # 4 倍速
    python view_training.py --headless --out renders/  # 不开窗，存图
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ⚠️ 不要设 MUJOCO_GL —— 这是本项目第二次踩同一个坑（已记入 参考卡.md）：
#   * `MUJOCO_GL=egl` 是 **Linux** 的后端，Windows 上直接报
#     "invalid value for environment variable MUJOCO_GL: egl"
#   * Windows 上用**默认后端**即可（GUI 窗口与离屏渲染都能工作）
#   * 反过来说，若离屏渲染输出全黑而没有任何报错，先检查是不是后端变量被设错
import mujoco
import mujoco.viewer

import mujoco
import mujoco.viewer

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, E, RMUL3v3Config, S
from rm_sentry.rule_baseline import RuleSentry
from rm_sentry.train_ppo import to_env_action

# --------------------------------------------------------------------------
# 配色
# --------------------------------------------------------------------------
C_FLOOR = "0.20 0.22 0.26 1"
C_COVER = "0.35 0.30 0.26 1"
C_ZONE = "0.25 0.60 0.95 0.45"        # 控制区（半透明蓝）
C_ZONE_HELD = "0.25 0.95 0.45 0.50"   # 我方占领时变绿
C_ZONE_ENEMY = "0.95 0.30 0.25 0.50"  # 对方占领时变红
C_SELF = "0.30 0.65 1.0 1"            # 我方哨兵
C_SELF_LOW = "1.0 0.55 0.15 1"        # 我方低血量
C_ENEMY = "0.95 0.25 0.25 1"          # 敌人
C_DEAD = "0.35 0.35 0.35 0.35"        # 阵亡（半透明灰）
C_SIGHT = "1.0 1.0 0.4 0.30"          # 视线


def build_scene(cfg: RMUL3v3Config) -> str:
    """按环境配置生成 MJCF，坐标直接沿用 env 的场地坐标（x 右、y 上 -> MuJoCo x-y 平面）。"""
    W, H, R = cfg.field_w, cfg.field_h, cfg.center_r
    parts = [f"""<mujoco model="rmul_3v3_view">
  <compiler angle="radian"/>
  <option timestep="0.05" gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
    <quality shadowsize="1024"/>
    <map znear="0.1" zfar="60"/>
    <global offwidth="1280" offheight="800"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.18 0.21"
             rgb2="0.20 0.22 0.26" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="12 8" texuniform="true"/>
    <material name="zone" rgba="{C_ZONE}"/>
    <material name="zone_held" rgba="{C_ZONE_HELD}"/>
    <material name="zone_enemy" rgba="{C_ZONE_ENEMY}"/>
    <material name="cover" rgba="{C_COVER}"/>
    <material name="self_ok" rgba="{C_SELF}"/>
    <material name="self_low" rgba="{C_SELF_LOW}"/>
    <material name="enemy" rgba="{C_ENEMY}"/>
    <material name="dead" rgba="{C_DEAD}"/>
    <material name="sight" rgba="{C_SIGHT}" emission="0.4"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="{W/2+1} {H/2+1} 0.01" material="floor"/>
    <!-- 场地边框 -->
    <geom name="bd_s" type="box" size="{W/2+0.05} 0.05 0.03" pos="{W/2} {-0.05} 0.03"
          rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>
    <geom name="bd_n" type="box" size="{W/2+0.05} 0.05 0.03" pos="{W/2} {H+0.05} 0.03"
          rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>
    <geom name="bd_w" type="box" size="0.05 {H/2+0.05} 0.03" pos="{-0.05} {H/2} 0.03"
          rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>
    <geom name="bd_e" type="box" size="0.05 {H/2+0.05} 0.03" pos="{W+0.05} {H/2} 0.03"
          rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>"""]

    # 控制区（用 site 以便运行时改颜色；MuJoCo 里 geom 的 material 也能改）
    parts.append(f"""
    <geom name="zone" type="cylinder" size="{R} 0.02" pos="{W/2} {H/2} 0.02"
          material="zone" contype="0" conaffinity="0"/>""")

    # 掩体（env 的 covers = (x, y, w, h)）
    for i, (cx, cy, cw, ch) in enumerate(cfg.covers):
        parts.append(f"""
    <geom name="cover{i}" type="box" size="{cw/2} {ch/2} 0.25"
          pos="{cx+cw/2} {cy+ch/2} 0.25" material="cover" contype="0" conaffinity="0"/>""")

    parts.append("""
    <!-- 顶视需要从上方来的光；MuJoCo 的 headlight 跟随相机，两者配合才不黑 -->
    <light pos="6 4 10" dir="0 0 -1" directional="true"
           diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <light pos="0 0 8" dir="0.6 0.5 -1" directional="true"
           diffuse="0.35 0.35 0.4"/>""")

    # 我方哨兵 + 3 个敌人 + 视线指示
    parts.append(f"""
    <body name="self" pos="{W/2} {H/2} 0.35">
      <geom name="self_body" type="cylinder" size="0.16 0.18" material="self_ok"
            contype="0" conaffinity="0"/>
      <geom name="self_nose" type="box" size="0.12 0.03 0.06" pos="0.18 0 0"
            material="self_ok" contype="0" conaffinity="0"/>
      <geom name="self_sight" type="box" size="2.0 0.02 0.01" pos="2.0 0 0.1"
            material="sight" contype="0" conaffinity="0"/>
    </body>""")
    for i in range(cfg.n_enemies):
        parts.append(f"""
    <body name="enemy{i}" pos="{W/2} {H*0.7} 0.35">
      <geom name="enemy{i}_body" type="cylinder" size="0.14 0.16" material="enemy"
            contype="0" conaffinity="0"/>
      <geom name="enemy{i}_nose" type="box" size="0.10 0.03 0.05" pos="0.16 0 0"
            material="enemy" contype="0" conaffinity="0"/>
    </body>""")
    parts.append("\n  </worldbody>\n</mujoco>\n")
    return "".join(parts)


def make_policy(kind: str, ckpt: str | None, cfg, env, dev):
    """返回一个 (obs, mask) -> 动作 dict 的函数。"""
    if kind == "rule":
        rule = RuleSentry(cfg)
        return lambda obs, mask: rule.act(obs, mask, env)
    if kind == "random":
        def f(obs, mask):
            n = env.n
            mv = torch.rand(n, 2, device=dev) * 2 - 1
            tgt = torch.randint(0, cfg.n_enemies + 1, (n,), device=dev)
            fire = (torch.rand(n, device=dev) < 0.5).float()
            return {"move": mv, "target": tgt, "fire": fire}
        return f
    # RL 检查点
    from rm_sentry.train_ppo import ActorCritic
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net = ActorCritic(env.obs_dim, cfg.n_enemies + 1, ck["cfg"].get("hidden", 256)).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    def f(obs, mask):
        act, _ = net.act(obs, mask)
        return act
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description="把 rm_sentry 对局实时渲染到 MuJoCo 窗口")
    ap.add_argument("--policy", default="rule", choices=["rule", "random", "rl"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--season", default="2026")
    ap.add_argument("--episodes", type=int, default=2, help="看几局")
    ap.add_argument("--speed", type=float, default=1.0, help="回放倍速")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.policy == "rl" and not args.ckpt:
        print("[xx] --policy rl 需要 --ckpt")
        return 2

    cfg = RMUL3v3Config(season=args.season)
    dev = torch.device(args.device)
    env = BatchedRMUL3v3(cfg, n_envs=1, device=dev, seed=7)   # 单环境，便于观看
    obs = env.reset()
    mask = env.action_mask()
    policy = make_policy(args.policy, args.ckpt, cfg, env, dev)

    model = mujoco.MjModel.from_xml_string(build_scene(cfg))
    data = mujoco.MjData(model)

    def gid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def bid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    id_zone = gid("zone")
    mat_zone = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n)
                for n in ("zone", "zone_held", "zone_enemy")]
    mat_self = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n)
                for n in ("self_ok", "self_low", "dead")]
    mat_enemy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "enemy")
    mat_dead = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "dead")

    total = int(cfg.match_time / cfg.dt) * args.episodes
    print("=" * 74)
    print(f"实时可视化  策略={args.policy}  场地 {cfg.field_w}×{cfg.field_h}  "
          f"控制区半径 {cfg.center_r}")
    print(f"  关闭窗口即退出。每 {cfg.dt}s 一步，回放倍速 {args.speed}x")
    print("=" * 74)
    if args.headless:
        Path(args.out).mkdir(exist_ok=True)
        renderer = mujoco.Renderer(model, height=600, width=900)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = [cfg.field_w / 2, cfg.field_h / 2, 0]
        cam.distance = 14
        cam.elevation = -75
        cam.azimuth = 90
    else:
        handle = mujoco.viewer.launch_passive(model, data)
        with handle as viewer:
            viewer.cam.lookat[:] = [cfg.field_w / 2, cfg.field_h / 2, 0]
            viewer.cam.distance = 14
            viewer.cam.elevation = -75
            viewer.cam.azimuth = 90
            run_loop(env, cfg, policy, obs, data, model, viewer, args, id_zone,
                     mat_zone, mat_self, mat_enemy, mat_dead, gid, bid)
        return 0

    # headless
    for t in range(total):
        obs = step_env(env, cfg, policy, obs, data, model, id_zone, mat_zone,
                       mat_self, mat_enemy, mat_dead, gid, bid)
        if t % 40 == 0:
            renderer.update_scene(data, camera=cam)
            from PIL import Image
            Image.fromarray(renderer.render()).save(Path(args.out) / f"view_{t:05d}.png")
    renderer.close()
    print(f"已存图到 {args.out}/")
    return 0


def step_env(env, cfg, policy, obs, data, model, id_zone, mat_zone, mat_self,
             mat_enemy, mat_dead, gid, bid):
    """推进环境一步，并把状态镜像到 MuJoCo 的 data。返回新的 obs。"""
    mask = env.action_mask()
    act = policy(obs, mask)
    a = to_env_action(act, env.n_actions, env.device)
    nobs, rew, done, info = env.step(a)
    sync_scene(env, cfg, data, model, id_zone, mat_zone, mat_self,
               mat_enemy, mat_dead, gid, bid)
    env.auto_reset(done)
    # 重置过的环境需要新的观测；这里统一重新构建一次（单环境，开销可忽略）
    return env._build_obs()


def sync_scene(env, cfg, data, model, id_zone, mat_zone, mat_self,
               mat_enemy, mat_dead, gid, bid) -> None:
    """把 env 状态写进 MuJoCo data + 改材质颜色。"""
    W, H = cfg.field_w, cfg.field_h

    # 控制区颜色：我占=绿，敌占=红，无人=蓝
    held = bool(env._hold_sentry[0]) if hasattr(env, "_hold_sentry") else False
    held_e = bool(env._hold_enemy[0]) if hasattr(env, "_hold_enemy") else False
    model.geom_matid[id_zone] = mat_zone[1] if held else (mat_zone[2] if held_e else mat_zone[0])

    # 我方
    b = bid("self")
    dead = float(env.sentry[0, S.DEAD]) > 0
    hp_low = float(env.sentry[0, S.HP]) < cfg.sentry_hp * 0.5
    data.xpos[b] = [float(env.sentry[0, S.X]), float(env.sentry[0, S.Y]), 0.35]
    th = float(env.sentry[0, S.TH])
    data.xquat[b] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    mat = mat_self[2] if dead else (mat_self[1] if hp_low else mat_self[0])
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == b:
            model.geom_matid[g] = mat
    # 视线指示长度：跟随是否有可见目标
    gs = gid("self_sight")
    vis = env._los_blocked()
    valid = vis & (env.enemies[:, :, E.ALIVE] > 0.5)
    valid[:, 0] = False
    model.geom_size[gs][0] = 2.0 if bool(valid[0].any()) else 0.0

    # 敌人
    for i in range(cfg.n_enemies):
        b = bid(f"enemy{i}")
        slot = i + 1
        alive = float(env.enemies[0, slot, E.ALIVE]) > 0.5
        data.xpos[b] = [float(env.enemies[0, slot, E.X]),
                        float(env.enemies[0, slot, E.Y]), 0.35]
        eth = float(env.enemies[0, slot, E.TH])
        data.xquat[b] = [np.cos(eth / 2), 0, 0, np.sin(eth / 2)]
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == b:
                model.geom_matid[g] = mat_enemy if alive else mat_dead


def run_loop(env, cfg, policy, obs, data, model, viewer, args, id_zone, mat_zone,
             mat_self, mat_enemy, mat_dead, gid, bid) -> None:
    total = int(cfg.match_time / cfg.dt) * args.episodes
    ep_len = int(cfg.match_time / cfg.dt)
    ep = 0
    for t in range(total):
        if not viewer.is_running():
            break
        t0 = time.time()
        obs = step_env(env, cfg, policy, obs, data, model, id_zone, mat_zone,
                       mat_self, mat_enemy, mat_dead, gid, bid)
        mujoco.mj_forward(model, data)
        viewer.sync()
        if t > 0 and t % ep_len == 0:
            ep += 1
            print(f"  --- 第 {ep} 局结束 ---")
        # 按倍速回放
        want = cfg.dt / max(args.speed, 1e-6)
        el = time.time() - t0
        if el < want:
            time.sleep(want - el)
    print("  回放结束")


if __name__ == "__main__":
    raise SystemExit(main())
