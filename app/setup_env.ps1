# RMUL 3v3 哨兵强化学习 —— 环境搭建 / 校验（PowerShell）
#
# 用法:  powershell -ExecutionPolicy Bypass -File setup_env.ps1
#        powershell -ExecutionPolicy Bypass -File setup_env.ps1 -WithOnnx
#
# ⚠️ 教训（详见 RM哨兵强化学习备赛总纲.md §2.3）:
#   不要用 `conda create -n rl --clone d2l`！
#   克隆会把 d2l 的全部已装包（jupyter 全家桶、torchvision、d2l 教学包本身）
#   一起复制过来，共 139 个包，其中 45 个与本项目无关；更糟的是那堆包自带的
#   依赖声明会让 pip 在你装别的东西时擅自升级 numpy，破坏其他包的兼容性。
#   正确做法：新建空环境，只装需要的。

param([switch]$WithOnnx)

$ErrorActionPreference = 'Stop'

$EnvName   = 'rl'
$CondaRoot = 'D:\Anaconda3'
$TorchIdx  = 'https://download.pytorch.org/whl/cu128'   # RTX 5070 = Blackwell sm_120，必须 cu128
$PyVersion = '3.11'

function Write-Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "  [ok] $m" -ForegroundColor Green }
function Write-Warn2($m){ Write-Host "  [!!] $m" -ForegroundColor Yellow }

Write-Step "0. 检查 conda 与磁盘"
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "未找到 conda" }
Write-Ok "conda 可用"
foreach ($l in @('C', 'D')) {
    try {
        $di = New-Object System.IO.DriveInfo($l)
        Write-Ok ("{0}: 剩余 {1:N1} GB" -f $l, ($di.AvailableFreeSpace / 1GB))
    } catch { Write-Warn2 "$l 盘空间查询失败（不影响继续）" }
}

Write-Step "1. 创建/复用 $EnvName 环境"
$envList = conda env list
if ($envList -match "(?m)^\s*$EnvName\s") {
    Write-Ok "$EnvName 已存在，跳过创建"
} else {
    Write-Host "  新建空环境（Python $PyVersion，不克隆任何东西）..."
    conda create -n $EnvName "python=$PyVersion" pip -y -q
    if ($LASTEXITCODE -ne 0) { throw "conda create 失败" }
    Write-Ok "已创建 $EnvName"
}
$py = Join-Path $CondaRoot "envs\$EnvName\python.exe"
if (-not (Test-Path $py)) { throw "找不到 $py" }
Write-Ok "python: $py"

Write-Step "2. 安装依赖"
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -eq 0) {
    $tv = & $py -c "import torch;print(torch.__version__)"
    Write-Ok "torch 已存在: $tv"
} else {
    Write-Host "  安装 torch cu128（约 2.5GB，视网速 3~15 分钟）..."
    & $py -m pip install -q --upgrade pip
    & $py -m pip install torch --index-url $TorchIdx
    if ($LASTEXITCODE -ne 0) { throw "torch 安装失败" }
    Write-Ok "torch 安装完成"
}
& $py -m pip install -q gymnasium mujoco numpy
Write-Ok "gymnasium / mujoco / numpy 安装完成"

if ($WithOnnx) {
    Write-Host "  安装 onnx / onnxruntime / onnxscript（导出部署包才需要）..."
    & $py -m pip install -q onnx onnxruntime onnxscript
    # onnxscript 会经 ml-dtypes 把 numpy 顶上 2.x，而 scipy 要求 numpy<1.27。
    # 钉回 1.26.4：实测 onnxruntime 在此版本下工作正常。
    & $py -m pip install -q "numpy==1.26.4"
    Write-Ok "ONNX 工具链安装完成（numpy 已钉回 1.26.4）"
}

Write-Step "3. 环境自检"
& $py -X utf8 -c @"
import sys
import torch, numpy, gymnasium, mujoco
print('  python    ', sys.version.split()[0])
print('  torch     ', torch.__version__, '| cuda', torch.version.cuda, '| 可用', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('  gpu       ', p.name, f'{p.total_memory/1024**3:.1f} GB', f'(sm_{p.major}{p.minor})')
print('  numpy     ', numpy.__version__)
print('  gymnasium ', gymnasium.__version__)
print('  mujoco    ', mujoco.__version__)
"@
Write-Host "`n  依赖健康检查（pip check）:"
& $py -m pip check

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

Write-Step "4. 规则一致性测试（双赛季）"
& $py -X utf8 -m rm_sentry.rules_test | Select-Object -Last 4
$code1 = $LASTEXITCODE

Write-Step "5. 环境冒烟测试 + 吞吐基准"
& $py -X utf8 -m rm_sentry.smoke_test --season 2026 --n 4000 --steps 200 | Select-Object -Last 8
$code2 = $LASTEXITCODE

Write-Step "6. 训练吞吐基准（CPU）"
& $py -X utf8 -m rm_sentry.train_ppo --bench-only --device cpu
Pop-Location

if ($code1 -eq 0 -and $code2 -eq 0) {
    Write-Host "`n全部完成。" -ForegroundColor Green
    Write-Host "以后运行：" -ForegroundColor Green
    Write-Host "  conda activate $EnvName" -ForegroundColor Green
    Write-Host "  cd $root" -ForegroundColor Green
    Write-Host "  python -m rm_sentry.rules_test" -ForegroundColor Green
    Write-Host "  python -m rm_sentry.train_ppo --device cuda --n-envs 512 --eval-every 20" -ForegroundColor Green
} else {
    Write-Warn2 "有测试未通过，把上面的报错发我"
}
