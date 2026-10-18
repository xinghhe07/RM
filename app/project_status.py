"""项目状态盘点 —— 生成总工作台所需的事实，不靠记忆。

用途：`工作台.md` 里的每个数字都应来自本脚本的输出，避免"文档写了但实际不符"。
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def section(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def run(mod: str, *args) -> tuple[int, str]:
    p = subprocess.run([PY, "-X", "utf8", "-m", mod, *args],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    section("1. 目录结构")
    for p in sorted(ROOT.iterdir()):
        if p.name.startswith("."):
            continue
        kind = "DIR " if p.is_dir() else "file"
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {kind} {p.name:38s} {n:5d} 个文件")
        else:
            print(f"  {kind} {p.name:38s} {p.stat().st_size/1024:7.1f} KB")

    section("2. 关键模块可导入性（R3：能导入≠能运行，但先查导入）")
    mods = [
        "rm_sentry.rmul_3v3_env", "rm_sentry.train_ppo", "rm_sentry.guards",
        "rm_sentry.reward_scaling", "rm_sentry.rule_baseline", "rm_sentry.export_onnx",
        "view_training",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            print(f"  [ok  ] {m}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {m}: {type(e).__name__}: {e}")

    section("3. 规则测试（改环境后必跑）")
    rc, out = run("rm_sentry.rules_test")
    tail = [l for l in out.splitlines() if "合计" in l or "失败" in l]
    print(f"  exit={rc}")
    for l in tail[-2:]:
        print("  " + l.strip())

    section("4. 已保存的检查点")
    for d in sorted((ROOT / "runs").glob("*")):
        if not d.is_dir():
            continue
        pts = sorted(d.glob("*.pt"))
        js = sorted(d.glob("*.json"))
        print(f"  runs/{d.name:16s} 检查点 {len(pts):3d} 个, 结果 {len(js):2d} 个")
        for p in pts[:3]:
            print(f"        {p.name}")
        if len(pts) > 3:
            print(f"        ... 还有 {len(pts)-3} 个")

    section("5. 基线评测结果")
    bf = ROOT / "runs" / "baseline_eval" / "baselines.json"
    if bf.exists():
        d = json.loads(bf.read_text(encoding="utf-8"))
        print(f"  EVAL_SEED={d['eval_seed']}  season={d['season']}  n_envs={d['n_envs']}")
        for k, v in d["rows"].items():
            print(f"  {k:22s} 胜率 {v['win_rate']:5.2f}  净胜点 {v['vp_margin']:+7.1f}  "
                  f"占点率 {v.get('hold_rate', float('nan')):7.4f}  输出 {v['dmg_out']:6.0f}")
    else:
        print("  尚未生成")

    section("6. 奖励账本（确认占点项主导）")
    from rm_sentry.rmul_3v3_env import RMUL3v3Config
    c = RMUL3v3Config(season="2026")
    match, dmg = 300.0, 643.0
    hold_all = match * c.w_hold * c.dt
    dmg_all = dmg * c.w_dmg_out
    print(f"  w_hold={c.w_hold}  w_dmg_out={c.w_dmg_out}  w_dmg_in={c.w_dmg_in}  "
          f"w_approach={c.w_approach}  sigma={c.approach_sigma}")
    print(f"  整局：站住控制区 300 秒 = {hold_all:+.2f}   输出 643 伤害 = {dmg_all:+.2f}")
    print(f"  => 占点/伤害 = {hold_all/max(dmg_all,1e-9):.2f}x  (目标 >=10x)")

    section("7. 势函数塑形数值有效性")
    rc, out = run("check_shaping_scale")
    print(f"  exit={rc}")
    for l in out.splitlines():
        if "结论" in l or "范围有" in l:
            print("  " + l.strip())

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
