"""训练前体检（preflight）：在花几小时训练之前，用几分钟把能查的错全查掉。

设计原则
--------
1. **只查能自动判定的东西**，不做"看起来没问题"的主观结论。
2. **每项检查独立**，一项失败不影响其余继续跑，最后汇总。
3. **刻意找错**：不只跑正常路径，还要喂极端值（NaN/Inf/越界观测）、
   检查可复现性、检查长跑稳定性 —— 因为这些才是真实事故的来源。

检查项
------
  [1] 硬件与环境      : GPU/CUDA/显存/内存/磁盘
  [2] 依赖完整性      : 关键包版本、ONNX 链路可选
  [3] 规则一致性      : 调 rules_test（95 项）
  [4] 可复现性        : 同种子两次起步状态必须逐位相同
  [5] 数值稳健性      : 极端观测/动作下不得出现 NaN/Inf
  [6] 不变量          : 血量/热量/弹药/位置/胜利点全程在合法范围
  [7] 训练冒烟        : 真的跑 N 次 PPO 更新，看 loss/KL/梯度是否健康
  [8] 长跑稳定性      : 连续跑一段，看吞吐是否衰减、显存是否增长
  [9] 检查点往返      : 存档→读档→再推理，结果必须完全一致

用法
----
    python -m rm_sentry.preflight                 # 全量（默认）
    python -m rm_sentry.preflight --quick         # 跳过长跑稳定性
    python -m rm_sentry.preflight --device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rm_sentry.rmul_3v3_env import BatchedRMUL3v3, RMUL3v3Config, S, E

ROOT = Path(__file__).resolve().parent.parent

# 结果收集
RESULTS: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    tag = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  — {detail}"
    print(line, flush=True)


# --------------------------------------------------------------------------
def check_env(dev: torch.device) -> None:
    print("\n[1] 硬件与环境")
    print(f"      python {sys.version.split()[0]}  |  torch {torch.__version__}  |  设备 {dev}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"      GPU {p.name}  {total/1024**3:.1f} GB 总 / {free/1024**3:.1f} GB 空闲"
              f"  (sm_{p.major}{p.minor})")
        rec("CUDA 可用", True, f"sm_{p.major}{p.minor}")
        # Blackwell 必须 CUDA 12.8+
        cuda_v = torch.version.cuda or "0"
        need128 = p.major >= 12
        ok = (not need128) or tuple(int(x) for x in cuda_v.split(".")) >= (12, 8)
        rec("CUDA 版本匹配算力", ok,
            f"算力 sm_{p.major}{p.minor} 需 CUDA>=12.8，当前 {cuda_v}")
        rec("显存足够（>=2GB 空闲）", free > 2 * 1024**3, f"空闲 {free/1024**3:.1f} GB")
    else:
        print("      纯 CPU 模式（2D 环境实测约 20 万步/秒，可正常开发）")
        rec("CPU 模式可用", True)
        rec("CPU 线程数已设置", torch.get_num_threads() >= 1,
            f"torch 使用 {torch.get_num_threads()} 线程")

    # 磁盘（训练产物 + 检查点）
    try:
        import shutil
        free = shutil.disk_usage(str(ROOT)).free
        rec("磁盘空间（>=5GB）", free > 5 * 1024**3, f"{ROOT} 剩余 {free/1024**3:.1f} GB")
    except Exception as e:  # noqa: BLE001
        rec("磁盘空间", False, str(e))


def check_deps() -> None:
    print("\n[2] 依赖完整性")
    import importlib
    for mod, need in (("torch", "2.0"), ("numpy", "1.24"), ("gymnasium", None), ("mujoco", None)):
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            rec(f"{mod} 可导入", True, f"v{v}")
        except Exception as e:  # noqa: BLE001
            rec(f"{mod} 可导入", False, str(e))
    # ONNX 链路为可选
    for mod in ("onnx", "onnxruntime"):
        try:
            importlib.import_module(mod)
            rec(f"{mod}（可选）", True)
        except ImportError:
            rec(f"{mod}（可选）", True, "未安装 —— 只在导出部署包时需要")


def check_rules(dev: torch.device) -> None:
    print("\n[3] 规则一致性（调 rules_test）")
    import subprocess
    py = sys.executable
    r = subprocess.run([py, "-X", "utf8", "-m", "rm_sentry.rules_test"],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    tail = (r.stdout or "").strip().splitlines()
    summary = next((l for l in reversed(tail) if "合计" in l), "")
    rec("规则测试全部通过", r.returncode == 0, summary.strip() or f"退出码 {r.returncode}")


def check_reproducibility(dev: torch.device) -> None:
    print("\n[4] 可复现性（同种子两次必须完全一致）")
    cfg = RMUL3v3Config(season="2026")
    a = BatchedRMUL3v3(cfg, n_envs=64, device=dev, seed=1234).reset()
    b = BatchedRMUL3v3(cfg, n_envs=64, device=dev, seed=1234).reset()
    same = bool(torch.equal(a, b))
    rec("同种子 reset 逐位一致", same,
        "若失败：训练结果无法复现，实验对比全部失效")
    if not same:
        rec("（诊断）最大差异", False, f"{float((a-b).abs().max()):.3e}")

    # 不同种子必须不同，否则种子没生效
    c = BatchedRMUL3v3(cfg, n_envs=64, device=dev, seed=999).reset()
    rec("不同种子结果不同", not bool(torch.equal(a, c)),
        "若相同：种子参数没起作用")


def check_numerics(dev: torch.device) -> None:
    print("\n[5] 数值稳健性（喂极端值，不得出现 NaN/Inf）")
    cfg = RMUL3v3Config(season="2026")
    env = BatchedRMUL3v3(cfg, n_envs=32, device=dev, seed=0)
    env.reset()

    # 极端动作：全 +1 / 全 -1 / 越界值 / 目标索引越界
    n_act = env.n_actions
    cases = {
        "全 +1 动作": lambda n: torch.ones(n, n_act, device=dev),
        "全 -1 动作": lambda n: -torch.ones(n, n_act, device=dev),
        "越界动作(±5)": lambda n: torch.full((n, n_act), 5.0, device=dev)
                                  * (torch.randint(0, 2, (n, 1), device=dev) * 2 - 1),
        "目标索引越界(99)": lambda n: torch.cat([
            torch.zeros(n, 2, device=dev),
            torch.full((n, 1), 99.0, device=dev),
            torch.ones(n, n_act - 3, device=dev)], dim=1),
        "目标索引为负(-5)": lambda n: torch.cat([
            torch.zeros(n, 2, device=dev),
            torch.full((n, 1), -5.0, device=dev),
            torch.ones(n, n_act - 3, device=dev)], dim=1),
    }
    for name, mk in cases.items():
        ok_all = True
        detail = ""
        try:
            env.reset()
            for _ in range(50):
                act = mk(env.n)
                assert act.shape == (env.n, n_act), f"动作维度错: {tuple(act.shape)}"
                obs, rew, done, info = env.step(act)
                if not torch.isfinite(obs).all():
                    ok_all = False; detail = "obs 出现 NaN/Inf"; break
                if not torch.isfinite(rew).all():
                    ok_all = False; detail = "reward 出现 NaN/Inf"; break
                env.auto_reset(done)
        except Exception as e:  # noqa: BLE001
            ok_all = False; detail = f"抛异常: {type(e).__name__}: {e}"
        rec(f"极端输入安全：{name}", ok_all, detail)

    # 人为污染状态，检查是否被静默容忍（这类最危险）
    env.reset()
    env.sentry[:, S.HEAT1] = 1e9
    env.enemies[:, 1:, E.X] = 1e9
    try:
        obs, rew, done, info = env.step(torch.zeros(env.n, env.n_actions, device=dev))
        ok = torch.isfinite(obs).all() and torch.isfinite(rew).all()
        rec("状态被污染时仍不产生 NaN", bool(ok),
            "若失败：说明缺少输入校验，实车异常数据会导致策略输出 NaN" if not ok else "")
    except Exception as e:  # noqa: BLE001
        rec("状态被污染时仍不产生 NaN", False, f"抛异常: {e}")


def check_invariants(dev: torch.device) -> None:
    print("\n[6] 规则不变量（随机策略跑 3000 步，全程校验）")
    cfg = RMUL3v3Config(season="2026")
    env = BatchedRMUL3v3(cfg, n_envs=128, device=dev, seed=7)
    env.reset()
    viol: list[str] = []
    for t in range(3000):
        mask = env.action_mask()
        act = torch.zeros(env.n, env.n_actions, device=dev)
        act[:, 0:2] = torch.rand(env.n, 2, device=dev) * 2 - 1
        tgt = torch.randint(0, cfg.n_enemies + 1, (env.n,), device=dev)
        legal = mask[:, 2:2 + cfg.n_enemies + 1].gather(1, tgt.unsqueeze(1)).squeeze(1)
        act[:, 2] = torch.where(legal, tgt, torch.zeros_like(tgt)).float()
        act[:, 3] = (torch.rand(env.n, device=dev) < 0.6).float()
        obs, rew, done, info = env.step(act)

        s = env.sentry
        checks = {
            "血量越界": ((s[:, S.HP] < 0) | (s[:, S.HP] > cfg.sentry_hp)).any(),
            "热量越界": ((s[:, S.HEAT1] < 0) | (s[:, S.HEAT1] > cfg.heat_limit + 1e-6)).any(),
            "弹药为负": (s[:, S.AMMO] < 0).any(),
            "位置越界": ((s[:, S.X] < 0) | (s[:, S.X] > cfg.field_w)
                        | (s[:, S.Y] < 0) | (s[:, S.Y] > cfg.field_h)).any(),
            "胜利点越界": ((s[:, S.VP] < -1e-6) | (s[:, S.VP] > cfg.vp_init + 1e-6)).any(),
            "敌方血量越界": (env.enemies[:, 1:, E.HP] > cfg.enemy_hp + 1e-6).any(),
            "观测非有限": ~torch.isfinite(obs).all(),
            "奖励非有限": ~torch.isfinite(rew).all(),
        }
        for nm, bad in checks.items():
            if bool(bad) and nm not in viol:
                viol.append(f"{nm}@step{t}")
        env.auto_reset(done)
        if len(viol) >= 5:
            break
    rec("3000 步内无规则/数值不变量违规", not viol,
        "、".join(viol[:5]) if viol else "血量/热量/弹药/位置/胜利点/观测 全部合法")


def check_training_smoke(dev: torch.device, updates: int = 30) -> None:
    print(f"\n[7] 训练冒烟（真跑 {updates} 次 PPO 更新）")
    from rm_sentry.train_ppo import ActorCritic, TrainCfg, to_env_action

    cfg = RMUL3v3Config(season="2026")
    n_envs, rollout = 128, 64
    env = BatchedRMUL3v3(cfg, n_envs=n_envs, device=dev, seed=0)
    obs = env.reset(); mask = env.action_mask()
    net = ActorCritic(env.obs_dim, cfg.n_enemies + 1, 128).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

    losses, kls, gnorm, ent_hist, fire_hist = [], [], [], [], []
    crashed = ""
    t0 = time.perf_counter()
    for upd in range(updates):
        ob, mb, mvb, tgb, frb, lpb, vb, rb, db = ([] for _ in range(9))
        for _ in range(rollout):
            with torch.no_grad():
                pa, _ = net.act(obs, mask)
            a = to_env_action(pa, env.n_actions, dev)
            for buf, v in ((ob, obs), (mb, mask), (mvb, pa["move"]), (tgb, pa["target"]),
                           (frb, pa["fire"]), (lpb, pa["logp"]), (vb, pa["value"])):
                buf.append(v)
            nobs, rew, done, info = env.step(a)
            rb.append(rew); db.append(done.float())
            obs = nobs; env.auto_reset(done); mask = env.action_mask()
        with torch.no_grad():
            _, _, _, _, last_v = net.forward(obs, mask)
        O = torch.stack(ob).reshape(-1, env.obs_dim); M = torch.stack(mb).reshape(-1, env.n_actions)
        MV = torch.stack(mvb).reshape(-1, 2); TG = torch.stack(tgb).reshape(-1)
        FR = torch.stack(frb).reshape(-1); LP = torch.stack(lpb).reshape(-1)
        V = torch.stack(vb).reshape(-1); R = torch.stack(rb).reshape(-1)
        D = torch.stack(db).reshape(-1)
        T = rollout
        # GAE（按时间反推，这里用简化展开）
        adv = torch.zeros_like(R); last = torch.zeros(env.n, device=dev)
        for t in reversed(range(T)):
            nt = 1.0 - D[t * env.n:(t + 1) * env.n]
            nv = last_v if t == T - 1 else V[(t + 1) * env.n:(t + 2) * env.n]
            delta = R[t * env.n:(t + 1) * env.n] + 0.99 * nv * nt - V[t * env.n:(t + 1) * env.n]
            last = delta + 0.99 * 0.95 * nt * last
            adv[t * env.n:(t + 1) * env.n] = last
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        try:
            newlp, ent, newv = net.evaluate(O, M, MV, TG, FR)
            ratio = (newlp - LP).exp()
            loss = (-torch.min(ratio * adv,
                               ratio.clamp(0.8, 1.2) * adv).mean()
                    + 0.5 * 0.5 * (newv - ret).pow(2).mean()
                    - 0.01 * ent.mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn_util_clip(net)
            opt.step()
        except Exception as e:  # noqa: BLE001
            crashed = f"{type(e).__name__}: {e}"
            break
        losses.append(float(loss)); kls.append(float((LP - newlp).mean()))
        gnorm.append(gn); ent_hist.append(float(ent.mean())); fire_hist.append(float(pa["fire"].mean()))
    dt = time.perf_counter() - t0

    rec("PPO 冒烟未崩溃", not crashed, crashed)
    if losses:
        rec("loss 全程有限", all(np.isfinite(losses)), f"末值 {losses[-1]:+.4f}")
        rec("KL 未爆炸（|KL|<0.5）", max(abs(k) for k in kls) < 0.5,
            f"最大 {max(abs(k) for k in kls):.4f}")
        rec("梯度范数未爆炸（<1000）", max(gnorm) < 1000, f"最大 {max(gnorm):.1f}")
        rec("策略未塌缩成常量（开火率不是 0 或 1）",
            0.0 < np.mean(fire_hist) < 1.0, f"平均开火率 {np.mean(fire_hist):.3f}")
        rec("熵未归零（策略还在探索）", np.mean(ent_hist) > 0.1,
            f"平均熵 {np.mean(ent_hist):.3f}")
        print(f"      冒烟吞吐: {updates*n_envs*rollout/dt:,.0f} 步/秒（含更新）")


def nn_util_clip(net) -> float:
    """计算梯度范数（模拟 clip_grad_norm_ 的返回值，但不真的裁剪，便于观测）。"""
    total = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def check_longrun(dev: torch.device, seconds: float = 20.0) -> None:
    print(f"\n[8] 长跑稳定性（持续约 {seconds:.0f} 秒）")
    cfg = RMUL3v3Config(season="2026")
    env = BatchedRMUL3v3(cfg, n_envs=1024, device=dev, seed=0)
    env.reset()
    act = torch.zeros(env.n, env.n_actions, device=dev)
    act[:, 0] = 0.5

    buckets: list[float] = []
    mem: list[float] = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        for _ in range(50):
            obs, rew, done, info = env.step(act)
            env.auto_reset(done)
        dt = time.perf_counter() - t0
        buckets.append(50 * env.n / dt)
        if dev.type == "cuda":
            mem.append(torch.cuda.memory_allocated() / 1024**2)
        else:
            mem.append(0.0)

    # 丢弃前 3 个 bucket：GPU 首次 kernel 加载、JIT、显存分配都会压低起始吞吐，
    # 那不是"衰减"。取中段均值作为基线才是公平比较。
    warm = 3
    if len(buckets) <= warm + 3:
        rec("长跑稳定性采样足够", False, f"只采到 {len(buckets)} 个 bucket，建议延长时长")
        return
    base = np.mean(buckets[warm:warm + 3])
    last = np.mean(buckets[-3:])
    decay = (base - last) / base
    rec("吞吐无显著衰减（<15%）", decay < 0.15,
        f"基线(去预热) {base:,.0f} → 末尾 {last:,.0f} 步/秒（变化 {decay*100:+.1f}%）")
    if decay >= 0.15:
        print("      可能原因：内存/显存泄漏、Python 对象累积、或系统被其他程序抢占")
    if dev.type == "cuda":
        grow = max(mem) - min(mem)
        rec("显存无持续增长（<50MB）", grow < 50,
            f"波动 {grow:.1f} MB（峰值 {max(mem):.1f} MB）")
        print(f"      显存峰值: {max(mem):.1f} MB / 8GB")
    rec("长跑未产生 NaN（抽样自观测）", bool(torch.isfinite(obs).all()))


def check_checkpoint(dev: torch.device) -> None:
    print("\n[9] 检查点往返（存档→读档→推理必须一致）")
    from rm_sentry.train_ppo import ActorCritic

    cfg = RMUL3v3Config(season="2026")
    net1 = ActorCritic(36, cfg.n_enemies + 1, 64).to(dev)
    obs = torch.randn(8, 36, device=dev)
    mask = torch.ones(8, 7, dtype=torch.bool, device=dev)
    with torch.no_grad():
        before = net1(obs, mask)

    tmp = ROOT / "runs" / "_preflight_ckpt.pt"
    tmp.parent.mkdir(exist_ok=True)
    try:
        torch.save({"model": net1.state_dict(), "obs_dim": 36,
                    "n_targets": cfg.n_enemies + 1, "hidden": 64, "season": "2026"}, tmp)
        ck = torch.load(tmp, map_location=dev, weights_only=False)
        net2 = ActorCritic(ck["obs_dim"], ck["n_targets"], ck["hidden"]).to(dev)
        net2.load_state_dict(ck["model"])
        with torch.no_grad():
            after = net2(obs, mask)
        same = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(before, after))
        rec("存档→读档推理逐位一致", same,
            "若失败：训练中断后无法可靠续训" if not same else "")
        rec("检查点含必要元信息",
            all(k in ck for k in ("model", "obs_dim", "n_targets", "hidden")),
            "obs_dim/n_targets/hidden/season")
    except Exception as e:  # noqa: BLE001
        rec("存档→读档", False, f"{type(e).__name__}: {e}")
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="跳过长跑稳定性检查")
    ap.add_argument("--json", default=None, help="把报告写到此路径")
    args = ap.parse_args()
    dev = torch.device(args.device)

    print("=" * 72)
    print("训练前体检（preflight）—— 用几分钟换几小时不白跑")
    print("=" * 72)

    check_env(dev)
    check_deps()
    check_rules(dev)
    check_reproducibility(dev)
    check_numerics(dev)
    check_invariants(dev)
    check_training_smoke(dev)
    if not args.quick:
        check_longrun(dev)
    check_checkpoint(dev)

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 72)
    print(f"合计：通过 {n_pass} 项，失败 {n_fail} 项")
    if n_fail:
        print("\n失败项（这些会让训练白跑，先修再训）：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  [FAIL] {name}" + (f"  — {detail}" if detail else ""))
    else:
        print("全部通过 —— 这台机器可以开训。")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"name": n, "pass": o, "detail": d} for n, o, d in RESULTS],
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"报告已写入 {args.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
