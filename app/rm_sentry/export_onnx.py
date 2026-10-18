"""把训练好的策略导出为 ONNX，并验证 PyTorch / ONNX Runtime 数值一致。

为什么要这个脚本
----------------
实机是只有 CPU 的小电脑（AMD 7845HX 级别），而"部署"只需要**前向推理**：
    - 用 torch 做单样本前向，实测约 430 us/次，其中绝大部分是 Python/调度开销
    - 用 ONNX Runtime 可显著降低，并且**车端不需要装 torch**（省几 GB 依赖）
    - 更重要的是：ONNX 是上一句话的"冻结契约"，能避免"仿真里好、车上推理结果不同"
      这类最高频的部署事故

本脚本做三件事：
    1. 导出 policy.pt -> policy.onnx（含动作掩码作为输入）
    2. **逐项对齐验证** PyTorch 与 ONNX Runtime 的输出（不通过就报错退出）
    3. 测量两者在 CPU 上的单次推理延迟

用法
----
    python -m rm_sentry.export_onnx --run runs/smoke_cpu
    python -m rm_sentry.export_onnx --run runs/smoke_cpu --bench-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rm_sentry.train_ppo import ActorCritic

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def build_mask_input(n_targets: int) -> torch.Tensor:
    """构造一个"有真实约束"的掩码，而不是全 True —— 全 True 测不出掩码逻辑对不对。"""
    mask = torch.ones(1, 2 + n_targets + 1, dtype=torch.bool)
    mask[0, 2] = False          # 不交战：一般允许，这里设 False 以验证掩码生效
    mask[0, 3] = False          # 目标 1 不可见
    mask[0, 2 + n_targets] = True   # 允许开火
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/smoke_cpu", help="训练输出目录")
    ap.add_argument("--out", default=None, help="onnx 输出路径")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--repeat", type=int, default=2000)
    args = ap.parse_args()

    run_dir = (RUNS_DIR.parent / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    ckpt_path = run_dir / "policy.pt"
    if not ckpt_path.exists():
        print(f"[xx] 找不到 {ckpt_path}")
        return 1

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["n_targets"], ck["hidden"]).eval()
    net.load_state_dict(ck["model"])
    obs_dim, n_targets, hidden = ck["obs_dim"], ck["n_targets"], ck["hidden"]
    print(f"载入 {ckpt_path}")
    print(f"  obs_dim={obs_dim} n_targets={n_targets} hidden={hidden} "
          f"season={ck.get('season')} 参数 {sum(p.numel() for p in net.parameters()):,}")

    onnx_path = Path(args.out) if args.out else run_dir / "policy.onnx"
    mask = build_mask_input(n_targets)
    obs = torch.randn(1, obs_dim)

    if not args.bench_only:
        export_ok = True
        try:
            # torch 2.6+ 默认走 dynamo 导出路径，需要 onnxscript。
            # 这里显式选择 TorchScript 导出器：依赖更少、对纯 MLP 更稳定。
            torch.onnx.export(
                _Wrapper(net),
                (obs, mask),
                str(onnx_path),
                input_names=["obs", "action_mask"],
                output_names=["move_mu", "move_logstd", "target_logits", "fire_logit", "value"],
                dynamic_axes={"obs": {0: "batch"}, "action_mask": {0: "batch"},
                              "move_mu": {0: "batch"}, "target_logits": {0: "batch"},
                              "fire_logit": {0: "batch"}, "value": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
            print(f"  [ok] 已导出 {onnx_path} ({onnx_path.stat().st_size/1024:.1f} KB)")
        except Exception as e:  # noqa: BLE001
            export_ok = False
            print(f"  [xx] ONNX 导出失败: {e}")

        if export_ok:
            # ---------------- 数值一致性验证（最关键的一步） ----------------
            try:
                import onnxruntime as ort
            except ImportError:
                print("  [!!] 未装 onnxruntime，跳过一致性验证")
                print("       安装: pip install onnxruntime")
                return 0

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                mu_t, ls_t, tl_t, fl_t, v_t = net(obs, mask)
            out = sess.run(None, {"obs": obs.numpy(),
                                  "action_mask": mask.numpy()})
            names = ["move_mu", "move_logstd", "target_logits", "fire_logit", "value"]
            refs = [mu_t, ls_t, tl_t, fl_t, v_t]
            worst = 0.0
            all_ok = True
            print("\n  数值一致性（PyTorch vs ONNX Runtime）:")
            for nm, ref, got in zip(names, refs, out):
                diff = float(np.abs(ref.numpy() - got).max())
                worst = max(worst, diff)
                ok = diff < 1e-4
                all_ok &= ok
                print(f"    {nm:14s} 最大绝对误差 {diff:.3e} {'ok' if ok else 'FAIL'}")
            # 掩码是否真的生效：被禁的目标 logit 应该是很大的负数
            tl = out[2][0]
            print(f"    掩码生效检查: 目标1(应被禁) logit={tl[1]:.1f}, "
                  f"目标2(应可用) logit={tl[2]:.1f}")
            mask_ok = tl[1] < -1e8 and tl[2] > -1e8
            print(f"    掩码: {'ok' if mask_ok else 'FAIL'}")
            if not (all_ok and mask_ok):
                print("\n  [xx] 导出结果与 PyTorch 不一致 —— 车端会出现行为偏差，请勿使用")
                return 1
            print(f"\n  [ok] 一致性通过（最大误差 {worst:.2e} < 1e-4）")

    # ---------------- 延迟对比 ----------------
    print(f"\n  CPU 单次推理延迟（{args.repeat} 次平均，单样本）:")
    with torch.no_grad():
        for _ in range(200):
            net(obs, mask)
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            net(obs, mask)
        t_torch = (time.perf_counter() - t0) / args.repeat
    print(f"\n    PyTorch        {t_torch*1e6:7.1f} us  -> {1/t_torch:,.0f} Hz")
    if not onnx_path.exists():
        print("    ONNX Runtime   跳过（onnx 文件不存在，导出未成功）")
        return 1
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = {"obs": obs.numpy(), "action_mask": mask.numpy()}
        for _ in range(200):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            sess.run(None, feed)
        t_ort = (time.perf_counter() - t0) / args.repeat
        print(f"    ONNX Runtime   {t_ort*1e6:7.1f} us  -> {1/t_ort:,.0f} Hz")
        print(f"    提速 {t_torch/t_ort:.1f}x")
    except ImportError:
        print("    ONNX Runtime   未安装（pip install onnxruntime）")

    print("\n  结论：哨兵决策频率是 1-10 Hz，以上均绰绰有余；")
    print("        选用 ONNX的额外好处是**车端不需要装 torch**。")
    return 0


class _Wrapper(torch.nn.Module):
    """ONNX 只导出前向；把掩码作为显式输入，避免把掩码逻辑藏进 Python 分支。"""

    def __init__(self, net: ActorCritic):
        super().__init__()
        self.net = net

    def forward(self, obs, mask):
        return self.net(obs, mask)


if __name__ == "__main__":
    raise SystemExit(main())
