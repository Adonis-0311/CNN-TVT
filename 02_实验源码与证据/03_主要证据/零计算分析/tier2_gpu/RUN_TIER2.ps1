# Tier-2 执行脚本（本机 CUDA 运行；沙箱内无 GPU，未在沙箱验证过训练路径）
#
# 前置：
#   1. 已签署并冻结 PREREGISTRATION_TIER2.md；
#   2. 已在 experiments/run_standard_experiment.py 的 diagnostic_* 工厂块中注册门控变体：
#        from analysis_zero_compute.tier2_gpu.presence_gated_vimd import register
#        factories = register(factories)
#   3. 先跑 0. 冒烟，确认 forward/loss/checkpoint 全通再上多 seed。

$ErrorActionPreference = 'Stop'
$branchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$repoRoot = Join-Path $branchRoot '02_实验源码与证据\01_当前可复现主线'
Set-Location -LiteralPath $repoRoot
$py    = 'D:\Python\python.exe'
$cache = 'standards\cache_factor_headline_1024_v2'

# ---------------------------------------------------------------------------
# 0. 冒烟：1 seed / 1 epoch，仅验证门控变体可训练、可评估、可保存
# ---------------------------------------------------------------------------
& $py experiments\run_standard_experiment.py `
  --cache-root $cache `
  --models a5_vimd_full,diagnostic_vimd_v5_presence_gated `
  --seeds 17 --epochs 1 --batch-size 16 --use-amp `
  --device cuda `
  --output artifacts --run-id tier2_smoke_gated_v1

# ---------------------------------------------------------------------------
# 0b. H2-D：冻结表示线性探针（几分钟，必须先于任何 H2-F 训练）
#     决定 clean 上的星座阶数信息是"表示里有但头用不上"还是"表示里没有"
# ---------------------------------------------------------------------------
#     注意：clean_probe.py（v1）已作废——它用 AGC 前的 jammer 去减 AGC 后的 x，
#     窗口功率 0.046~0.50 而非契约的 0.5。改用 v2。
foreach ($m in @('a5_vimd_full','a0_backbone','cssl_amc_supervised_adaptation','iqformer_inspired')) {
  foreach ($s in @(17,29,43)) {
    foreach ($d in @('counterfactual','cv')) {
      & $py analysis_zero_compute\tier2_gpu\clean_probe_v2.py --model $m --seed $s --design $d `
        --device cuda --out artifacts\tier2_clean_probe_v2
    }
  }
}
& $py analysis_zero_compute\tier2_gpu\summarize_probe_runs.py

# ---------------------------------------------------------------------------
# 1. H2-G：干扰存在性门控对照臂（预期不能单独修好 clean，见预注册）
# ---------------------------------------------------------------------------
& $py experiments\run_standard_experiment.py `
  --cache-root $cache `
  --models a5_vimd_full,diagnostic_vimd_v5_presence_gated `
  --seeds 17,29,43,71,101,131,173,211,257,307 `
  --epochs 30 --batch-size 16 --learning-rate 3e-4 --weight-decay 0.01 `
  --patience 8 --use-amp --device cuda `
  --bootstrap-draws 10000 `
  --output artifacts --run-id tier2_presence_gated_v1

# ---------------------------------------------------------------------------
# 2. H1：A5 容量阶梯（仅改宽度参数，不新增模型代码），5 seeds
#    每档单独 run-id，便于逐档核对参数量与结果
# ---------------------------------------------------------------------------
& $py experiments\run_standard_experiment.py `
  --cache-root $cache `
  --models a5_vimd_full `
  --seeds 17,29,43,71,101 `
  --spectral-channels 40 --embedding-dim 80 --environment-dim 48 `
  --epochs 30 --batch-size 16 --use-amp --device cuda `
  --output artifacts --run-id tier2_capacity_M_v1

& $py experiments\run_standard_experiment.py `
  --cache-root $cache `
  --models a5_vimd_full `
  --seeds 17,29,43,71,101 `
  --spectral-channels 64 --embedding-dim 128 --environment-dim 64 `
  --epochs 30 --batch-size 16 --use-amp --device cuda `
  --output artifacts --run-id tier2_capacity_L_v1

# 显存提示：L 档在 8 GB 卡上可能需要 --batch-size 8；若下调 batch，必须在
# 结果文档中记录，并且不得与 S/M 档的 batch 16 结果混为"同设置对比"。

# ---------------------------------------------------------------------------
# 3. 与冻结证据的逐 seed 配对分析（零算力工具链，读新旧两侧的 prediction NPZ）
# ---------------------------------------------------------------------------
& $py analysis_zero_compute\tier2_gpu\compare_tier2.py `
  --new artifacts\tier2_presence_gated_v1 `
  --new-model diagnostic_vimd_v5_presence_gated `
  --out analysis_zero_compute\outputs\csv\tier2_presence_gated.csv

Write-Host 'Tier-2 完成；请勿把上述任何目录并入 artifacts\tvt_v4r_headline_composite。'
