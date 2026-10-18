#!/usr/bin/env bash
# RMUL 3v3 哨兵强化学习 —— Ubuntu 环境搭建 / 校验（幂等，可重复运行）
#
# 目标系统：**Ubuntu 22.04 LTS**（比赛机）。同时兼容 24.04。
#
# 用法:
#     bash setup_env.sh                 # 建环境 + 装依赖 + 跑测试
#     bash setup_env.sh --check         # 只校验现有环境（不装东西）
#     bash setup_env.sh --with-onnx     # 额外装 ONNX 导出链（做部署包时才需要）
#
# Ubuntu 22.04 的两个已知坑（脚本已内置检查与提示）:
#   1. 自带 Python 3.10，且 **python3-venv 默认未安装** —— 缺了就建不了 venv
#   2. **RTX 50 系（Blackwell / sm_120）需要较新的 NVIDIA 驱动**，
#      22.04 仓库里的默认驱动版本可能直接认不到显卡
#
# ⚠️ torch 必须用 CUDA 12.8 的 wheel：pip 默认源会给 cu126 或 CPU 版。

set -uo pipefail

VENV_DIR="${RM_VENV:-$HOME/venvs/rl}"
PY="${PYTHON:-python3}"
CHECK_ONLY=0
WITH_ONNX=0
for a in "$@"; do
    case "$a" in
        --check)     CHECK_ONLY=1 ;;
        --with-onnx) WITH_ONNX=1 ;;
        *) echo "未知参数: $a"; exit 2 ;;
    esac
done

c_cyan() { printf '\n\033[36m=== %s ===\033[0m\n' "$1"; }
c_ok()   { printf '  \033[32m[ok]\033[0m %s\n' "$1"; }
c_warn() { printf '  \033[33m[!!]\033[0m %s\n' "$1"; }
c_err()  { printf '  \033[31m[xx]\033[0m %s\n' "$1"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || { echo "无法定位项目根目录"; exit 1; }
cd "$ROOT" || { echo "无法进入 $ROOT"; exit 1; }

# ---------------------------------------------------------------- 0. 系统检查
c_cyan "0. 系统与驱动检查"
c_ok "项目根目录: $ROOT"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    c_ok "系统: ${PRETTY_NAME:-未知}"
    case "${VERSION_ID:-}" in
        22.04) c_ok "目标系统 Ubuntu 22.04 —— 与 ROS 2 Humble / Isaac Sim 官方验证版本一致" ;;
        24.04) c_warn "Ubuntu 24.04：可用，但注意 ROS 2 对应版本是 Jazzy 而非 Humble" ;;
        *)     c_warn "非 22.04/24.04（${VERSION_ID:-未知}），脚本可能不完全适用" ;;
    esac
else
    c_warn "无法读取 /etc/os-release，按未知系统继续"
fi

# --- NVIDIA 驱动（Blackwell 是重点） ---
if command -v nvidia-smi >/dev/null 2>&1; then
    DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    c_ok "GPU: ${GPU:-未知}   驱动: ${DRV:-未知}"
    DRV_MAJOR="${DRV%%.*}"
    if [[ "$DRV_MAJOR" =~ ^[0-9]+$ ]] && (( DRV_MAJOR < 550 )); then
        c_warn "驱动 $DRV 偏旧。RTX 50 系（Blackwell/sm_120）需要较新驱动，"
        c_warn "22.04 仓库默认版本常常认不到卡。参考安装方式："
        c_warn "  https://gigagpu.com/nvidia-driver-555-blackwell-setup-ubuntu/"
    fi
else
    c_warn "未找到 nvidia-smi。可能原因："
    c_warn "  a) 没装驱动；b) 装了但版本太老不认 RTX 50 系（22.04 上常见）"
    c_warn "  本项目的 2D 环境在 CPU 上也能跑（实测 21 万步/秒），不阻塞开发"
fi

# --- Python 与 venv 能力 ---
command -v "$PY" >/dev/null 2>&1 || { c_err "找不到 $PY，请先: sudo apt install python3 python3-venv"; exit 1; }
# 直接取 python 自己的版本号，比解析 `--version` 更稳（也能避免版本号被写进 stderr）
PYVER="$("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || echo "未知")"
c_ok "python 版本: $PYVER  ($PY)"

if ! "$PY" -c 'import venv' >/dev/null 2>&1; then
    c_err "缺少 venv 模块 —— Ubuntu 22.04 默认不装 python3-venv！"
    c_err "请先执行:  sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
fi
c_ok "venv 模块可用"

if ! "$PY" -c 'import ensurepip' >/dev/null 2>&1; then
    c_warn "缺少 ensurepip（通常随 python3-venv 一起装）: sudo apt install -y python3-venv"
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    [[ -f "$VENV_DIR/bin/activate" ]] || { c_err "venv 不存在: $VENV_DIR"; exit 1; }
else
    # ------------------------------------------------------------ 1. 建 venv
    c_cyan "1. 创建 / 复用 venv"
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        c_ok "已存在，跳过: $VENV_DIR"
    else
        mkdir -p "$(dirname "$VENV_DIR")"
        "$PY" -m venv "$VENV_DIR" || {
            c_err "创建 venv 失败。若是 Ubuntu 22.04，先: sudo apt install -y python3-venv"
            exit 1
        }
        c_ok "已创建: $VENV_DIR"
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
VPY="$VENV_DIR/bin/python"
c_ok "使用解释器: $VPY ($("$VPY" -c 'import sys;print(sys.version.split()[0])'))"

if [[ $CHECK_ONLY -eq 0 ]]; then
    # ------------------------------------------------------------ 2. 装依赖
    c_cyan "2. 安装依赖"
    "$VPY" -m pip install -q --upgrade pip 2>/dev/null || c_warn "pip 升级失败（不致命）"

    if "$VPY" -c "import torch" >/dev/null 2>&1; then
        c_ok "torch 已存在: $("$VPY" -c 'import torch;print(torch.__version__)')"
    else
        echo "  安装 torch cu128（约 2.5GB，视网速 3~15 分钟）..."
        # 必须指定 cu128：RTX 5070 是 Blackwell sm_120
        "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || {
            c_err "torch 安装失败。检查网络，或改用国内镜像（注意镜像需同步 cu128）"
            exit 1
        }
        c_ok "torch 安装完成"
    fi

    "$VPY" -m pip install -q -r "$ROOT/requirements.txt" || {
        c_err "requirements.txt 安装失败"; exit 1;
    }
    c_ok "requirements.txt 安装完成"

    if [[ $WITH_ONNX -eq 1 ]]; then
        "$VPY" -m pip install -q onnx onnxruntime onnxscript
        # onnxscript 会经 ml-dtypes 把 numpy 顶到 2.x，而 scipy 要求 numpy<1.27
        "$VPY" -m pip install -q "numpy==1.26.4"
        c_ok "ONNX 工具链安装完成（numpy 已钉回 1.26.4）"
    fi
fi

# ---------------------------------------------------------------- 3. 自检
c_cyan "3. 环境自检"
"$VPY" - <<'PYEOF'
import sys
try:
    import torch, numpy, gymnasium, mujoco
except ImportError as e:
    print(f"  [xx] 缺依赖: {e}")
    print("       请运行:  bash setup_env.sh")
    sys.exit(1)
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(f"  [ok] python    {sys.version.split()[0]}")
print(f"  [ok] torch     {torch.__version__} | cuda {torch.version.cuda} | 可用 {torch.cuda.is_available()}")
if p is not None:
    print(f"  [ok] gpu       {p.name} {p.total_memory/1024**3:.1f} GB (sm_{p.major}{p.minor})")
    if p.major >= 12 and str(torch.version.cuda) < "12.8":
        print(f"  [!!] Blackwell (sm_{p.major}{p.minor}) 需要 CUDA 12.8+ 的 torch，"
              f"当前 {torch.version.cuda} —— 请重装 cu128 wheel")
else:
    print("  [!!] CUDA 不可用，回退 CPU（2D 环境实测 21 万步/秒，可正常开发）")
print(f"  [ok] numpy     {numpy.__version__}")
print(f"  [ok] gymnasium {gymnasium.__version__}")
print(f"  [ok] mujoco    {mujoco.__version__}")
PYEOF

echo ""
echo "  依赖健康检查:"
"$VPY" -m pip check 2>&1 | head -6 || true

# ---------------------------------------------------------------- 4/5. 测试
c_cyan "4. 规则一致性测试（双赛季）"
"$VPY" -X utf8 -m rm_sentry.rules_test | tail -4
RC1=${PIPESTATUS[0]}

c_cyan "5. 环境冒烟测试 + 吞吐基准"
"$VPY" -X utf8 -m rm_sentry.smoke_test --season 2026 --n 4000 --steps 200 | tail -10
RC2=${PIPESTATUS[0]}

c_cyan "6. 训练吞吐基准（本机 CPU）"
"$VPY" -X utf8 -m rm_sentry.train_ppo --bench-only --device cpu

cat <<EOF

$(printf '\033[32m全部完成。\033[0m')
以后运行：
  source $VENV_DIR/bin/activate
  cd $ROOT
  python -m rm_sentry.rules_test
  python -m rm_sentry.train_ppo --bench-only --device cpu     # 确认本机吞吐
  python -m rm_sentry.export_onnx --run runs/<tag>           # 导出部署包

跨平台提示：同一份代码在 Windows 与 Ubuntu 上是同一套，没有平台分支。
若从 Windows 拷贝项目过来，先确认 shell 脚本是 LF 行尾（仓库里有 .gitattributes 兜底）：
  file setup_env.sh        # 应显示 "ASCII text"，若含 "CRLF" 则执行:
  sed -i 's/\r$//' setup_env.sh
EOF

if [[ "${RC1:-1}" -ne 0 || "${RC2:-1}" -ne 0 ]]; then
    c_err "有测试未通过（RC1=$RC1 RC2=$RC2），把上面的报错发出来"
    exit 1
fi
