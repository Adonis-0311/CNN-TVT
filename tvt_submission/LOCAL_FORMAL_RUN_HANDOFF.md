# TVT 本地正式运行交接

状态日期：2026-07-28

本文件给出各阶段的精确合同；日常操作应优先使用
`run_all_local_after_gpu_free.ps1` 和 `LOCAL_EXECUTION_QUEUE.md` 中的统一
队列。它不授权修改冻结配置，也不把诊断结果提升为论文证据。当前正式缓存、
五种子正式运行和 release lock 均未启动或生成；`paper/results_auto.tex`
仍是内部评审占位文件。

## 1. 冻结范围与当前状态

所有命令都从以下项目根目录执行：

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\vimd_amc"
$Python = "D:\Python\python.exe"
$Ack = "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE"
$MinimumFreeGpuMiB = 7000
$MinimumFreeDiskGiB = 20
```

权威冻结文件是
`tvt_submission/configs/formal_tvt_freeze_v1.json`，其当前合同为：

- 正式缓存：`standards/cache_factor_headline_1024_v1`；
- 正式运行：`artifacts/tvt_headline_1024_5seed_v1`；
- 9 个源身份互斥 split，共 47,000 个源序列、94,000 个配对视图；
- 11 个模型、种子 `17,29,43,71,101`，共 55 次拟合；
- 每次拟合最多 30 epoch，batch size 64，CUDA，AMP；
- 10,000 次分层配对 bootstrap；
- 固定主参考是 `cssl_amc_supervised_adaptation`，其选择在训练前完成，
  不能事后描述为“最强”；
- 比较器家族包含
  `cssl_amc_supervised_adaptation`。它只能称为
  “CSSL-AMC official-architecture supervised adaptation”，不是完整 CSSL
  复现、官方数值复现或结构化干扰专用方法。

当前应观察到：

```powershell
Test-Path .\standards\cache_factor_headline_1024_v1
Test-Path .\artifacts\tvt_headline_1024_5seed_v1
Test-Path .\paper\release_lock.json
```

三个结果均应为 `False`。若任一项为 `True`，先人工审计已有目录；不要覆盖、
拼接或静默复用。

## 2. 本地资源与运行纪律

正式缓存依赖 MATLAB R2025a、5G Toolbox 25.1 和本项目 Python 环境；正式
实验的冻结设备是 CUDA，因此 CUDA 不可用会直接失败。论文构建使用本机
MiKTeX `latexmk.exe` 和项目内固定的 `paper/IEEEtran.cls`。

正式缓存规模是筛选缓存的十倍。已审计筛选缓存含 4,700 个源、190 个文件、
517,687,920 bytes；按样本数线性估算，正式缓存约
5,176,879,200 bytes（约 4.82 GiB）。这不是硬性上界，因为 MATLAB 临时
交换文件、55 组 checkpoint、预测 NPZ、CSV、日志和 TeX 构建还会占用空间。
所有 `-Execute` 入口会强制要求项目所在卷至少保留 20 GiB 可用空间。

运行器按模型和种子顺序执行 55 次拟合，不是 55 个并行 GPU 任务。耗时由
MATLAB 缓存生成、GPU 型号、早停点和 10,000 次 bootstrap 决定；不要用本
交接文件承诺完成时长。

三个执行入口使用同一个全局命名 mutex，并在执行前检查所有 Python、MATLAB
和 LibreOffice（含 `soffice.bin`）进程、GPU 剩余显存及项目卷空间；项目内
已有训练不会被白名单放行。长队列在阶段间重复检查，但不会终止任何已有
进程，也不会抢占或接管被占用/遗弃的 mutex。

从正式实验启动到 `run.json` 完成写入之间：

- 不修改 `src/`、`experiments/`、`standards/` 或运行相关脚本；
- 不启动第二个正式运行，不与高负载 GPU 作业共享设备；
- 不改模型、种子、split、epoch、checkpoint 或多重比较设置；
- 不读取结果后调参再覆盖同一运行；
- 不触碰相邻的 `tccn_satellite_amc` 工作流；
- 中断或失败后保留现场。当前入口不支持向同一目录合并或覆盖续跑；任何重跑
  都应先完成失败审计，再以新的、明确批准的 freeze/run ID 开始。

## 3. 阶段 0：只读预检

以下命令会验证冻结配置并打印后续长命令，但不会构建缓存、训练模型或写
release：

```powershell
& .\tvt_submission\run_local.ps1 -Python $Python
```

输出末尾必须包含：

```text
Dry preflight complete. No cache build, model training, or release write was started.
```

预检失败时停止。不要绕过 `validate_formal_freeze.py`，也不要手工复制它本应
生成的参数。预检会把冻结的九个 split 计数和 evidence designation 与实际
`standards/build_factor_cache.py` 的 `headline` 预设静态对齐，并拒绝 runner、
baseline 关键符号的重复顶层绑定；正式 CUDA freeze 还要求 AMP 保持启用。

## 4. 阶段 1：构建正式缓存

只有本地操作者确认 MATLAB、磁盘和运行窗口后，才执行：

```powershell
& .\tvt_submission\run_local.ps1 `
  -Stage cache `
  -Execute `
  -Acknowledgement $Ack `
  -MinimumFreeGpuMiB $MinimumFreeGpuMiB `
  -MinimumFreeDiskGiB $MinimumFreeDiskGiB `
  -Python $Python
```

预期根目录：

```text
standards/cache_factor_headline_1024_v1/
```

至少必须出现：

```text
standards/cache_factor_headline_1024_v1/manifest.json
```

缓存必须保持 schema 2、designation
`headline_formal_tvt_evidence`、恰好九个 split、47,000 个全局源身份互斥
记录，并通过 checksum、component identity、SNR/SIR、guard、factor
coverage 和类别支持检查。目录预先存在、MATLAB 超时、缺少 toolbox、
component 误差越界、checksum/shape/finiteness 错误或源身份碰撞，均是停止
条件；不得把筛选缓存重命名为正式缓存。

缓存完成后，用实验阶段的非执行预检确认 manifest 可见：

```powershell
& .\tvt_submission\run_local.ps1 `
  -Stage experiment `
  -Python $Python
```

该命令仍然不训练。

## 5. 阶段 2：五种子正式运行

仅在阶段 1 完整通过后执行：

```powershell
$CacheSha = (
  Get-FileHash -Algorithm SHA256 `
    .\standards\cache_factor_headline_1024_v1\manifest.json
).Hash.ToLowerInvariant()

& .\tvt_submission\run_local.ps1 `
  -Stage experiment `
  -Execute `
  -AllowValidatedReuse `
  -ExpectedCacheManifestSha256 $CacheSha `
  -Acknowledgement $Ack `
  -MinimumFreeGpuMiB $MinimumFreeGpuMiB `
  -MinimumFreeDiskGiB $MinimumFreeDiskGiB `
  -Python $Python
```

预期根目录：

```text
artifacts/tvt_headline_1024_5seed_v1/
```

关键产物至少包括：

```text
run.json
metrics.csv
seed_aggregates.csv
paired_statistics.csv
headline_paired_statistics.csv
manifests/cache_reference.json
models/<model>_seed<seed>/model.pt
models/<model>_seed<seed>/predictions_<split>.npz
```

成功进程退出不等于论文可用。`run.json` 还必须同时证明：

- `status` 与 `execution_status` 均为 `complete`；
- `evidence_eligibility` 内的 `eligible`、
  `formal_paper_evidence_eligible` 和 `headline_eligible` 均为 `true`；
- 缓存 designation 与 policy version 精确匹配正式合同；
- 11 模型 × 5 种子矩阵完整、无重复、无缺失；
- 每次拟合选择了完整目标生效窗口内的验证 checkpoint；
- `fallback_used=false`；
- 可执行源树的起止 fingerprint 完全相同；
- checksum、component、split、类别、辅助标签支持和统计产物门全部通过；
- 所有结果单元有限且不是占位值。

任一项失败，即保持内部评审状态。不要从失败目录挑选“最好 seed”，也不要
手工修补 `run.json`。

## 6. 阶段 3：自动生成论文宏清单

只有 eligible 正式运行存在后，执行：

```powershell
& $Python -B .\tvt_submission\generate_macro_values.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --output .\tvt_submission\formal_macro_values.json
```

该生成器不接受人工输入的性能值。它必须从 runner-native JSON、CSV 和
source-aligned prediction NPZ 交叉推导宏，并把 CSSL 适配器纳入本地比较器
审计。缺文件、重复单元、NaN/Inf、指标不一致、样本对齐失败、比较器选择
含糊或任何占位值都会非零退出并保持 release 关闭。

如果 `formal_macro_values.json` 已存在，不要默认覆盖。只有在完整来源审计
确认旧文件应被替换后，才可显式使用 `--replace-existing`，并保留旧版本及
原因记录。

当前唯一宏合同为：artifact-derived manifest 含 73 个 provenance 宏；
内部评审 `results_auto.tex` 含 `ResultSource` 加这 73 个宏，共 74 个
non-sentinel 命令；正式锁定后再增加 `EligibleLockedResults`，共 75 个。
宏名只允许 ASCII 字母，百分位名称必须使用 `PFifty` 和
`PNinetyFive`；`P50`/`P95` 必须被 parser 拒绝。

## 7. 阶段 4：release 预检、写锁与复核

先执行不写文件的 release 预检：

```powershell
$RunSha = (
  Get-FileHash -Algorithm SHA256 `
    .\artifacts\tvt_headline_1024_5seed_v1\run.json
).Hash.ToLowerInvariant()

& .\tvt_submission\run_local.ps1 `
  -Stage release `
  -MacroValues "tvt_submission\formal_macro_values.json" `
  -Execute `
  -AllowValidatedReuse `
  -ExpectedCacheManifestSha256 $CacheSha `
  -ExpectedRunJsonSha256 $RunSha `
  -Acknowledgement $Ack `
  -MinimumFreeGpuMiB $MinimumFreeGpuMiB `
  -MinimumFreeDiskGiB $MinimumFreeDiskGiB `
  -Python $Python
```

只有预检零退出后，才允许原子写入结果宏和锁：

```powershell
& .\tvt_submission\run_local.ps1 `
  -Stage release `
  -MacroValues "tvt_submission\formal_macro_values.json" `
  -WriteRelease `
  -Execute `
  -AllowValidatedReuse `
  -ExpectedCacheManifestSha256 $CacheSha `
  -ExpectedRunJsonSha256 $RunSha `
  -Acknowledgement $Ack `
  -MinimumFreeGpuMiB $MinimumFreeGpuMiB `
  -MinimumFreeDiskGiB $MinimumFreeDiskGiB `
  -Python $Python
```

预期新增或替换：

```text
paper/results_auto.tex
paper/release_lock.json
```

合格的 `results_auto.tex` 必须定义：

```tex
\newcommand{\EligibleLockedResults}{eligible_locked_formal_run}
```

`release_lock.json` 必须把 run ID、cache digest、`run.json` SHA-256、宏清单
SHA-256、`results_auto.tex` SHA-256 和 sentinel 绑定在一起。随后只读复核：

```powershell
& .\tvt_submission\run_local.ps1 `
  -Stage release `
  -Execute `
  -AllowValidatedReuse `
  -ExpectedCacheManifestSha256 $CacheSha `
  -ExpectedRunJsonSha256 $RunSha `
  -Acknowledgement $Ack `
  -MinimumFreeGpuMiB $MinimumFreeGpuMiB `
  -MinimumFreeDiskGiB $MinimumFreeDiskGiB `
  -Python $Python
```

已有 lock 不一致、宏文件被改动或来源 digest 漂移均应失败。不要使用
`--replace-existing-release`，除非人类作者明确批准一次可审计的整体替换。

## 8. 阶段 5：科学晋级门

自动完整性门通过后仍必须检查科学效果。当前预注册的最低晋级线是：

- A5 在 hard-interference 主终点的 macro-F1 必须分别高于 A0、MCLDNN、
  IQFormer-inspired 和 CSSL 监督适配器中的每一个至少 5 个百分点，不是
  只高于其中“最佳”或某个事后选定的比较器；
- `unseen_jammer`、`unseen_speed`、`heldout_channel` 三个独立外推域中，
  至少两个达到不低于 3 个百分点的 macro-F1 增益；
- clean retention 需分别报告 seen A/C/D 与 held B/E；非劣性要求点估计
  不低于 -1 个百分点，配对 95% 区间下界不低于 -2 个百分点；
- 机制指标、消融方向、置信区间与 multiplicity 结果不得与论文因果叙述
  冲突。

manifest v3 已序列化 `scientific_release_gate`；generator 必须从正式产物
推导它，release validator 必须独立重推导并锁绑定，paper gate 再动态消费。
人类作者仍须复核门的科学含义，不能因结构完整性验证通过就忽略失败。任何
主门失败都应保持 `\internalreviewtrue`，如实报告负结果或重新设计新的
前瞻性实验，不能在当前正式测试结果上事后调门槛。

## 9. 阶段 6：public build

release lock 与科学晋级门都通过后，人类作者还必须：

1. 创建并审核 `paper/authors_verified.tex`，填写真实作者、单位、通讯作者、
   funding、conflicts、acknowledgment 和适用披露；
2. 在 `paper/main.tex` 中把唯一的 `\internalreviewtrue` 改为
   `\internalreviewfalse`；
3. 复核专利公开时序、全部原始文献、数值、图表和 IEEE/TVT 当日政策。

然后构建：

```powershell
$Latexmk = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64\latexmk.exe"
Push-Location .\paper
& $Latexmk -pdf -interaction=nonstopmode -halt-on-error `
  -file-line-error -outdir=build main.tex
Pop-Location
```

对已构建 PDF 执行只读 release 审计：

```powershell
& $Python -B .\tvt_submission\validate_paper_build.py `
  --paper-root .\paper `
  --mode release `
  --max-pages 14
```

该命令必须返回 `"ok": true`。必须同时满足：public 分支、生效的
`authors_verified.tex`、有效 sentinel/release lock、无 pending/generated
占位、无 undefined citation/reference、无 fatal error、无 overfull box、
PDF 与 log 一致且不旧于源文件、页数不超过 14。最后还需逐页人工检查
`paper/build/main.pdf` 的表格、公式、图、脚注、参考文献和裁切。

## 10. VIMD-v4 与投稿承诺边界

`run_candidate_local.ps1` 对应 VIMD-v4 DSBN 的一次性候选诊断。它不在
`formal_tvt_freeze_v1.json` 的 11 模型正式家族中，输出也不是论文证据。
本交接不要求运行它；即便未来通过候选门，也必须先建立新的前瞻性 freeze，
才能进入正式实验，绝不能把诊断数值复制进当前 TVT 主结果。

这套流程用于降低可避免的实验、可复现性和投稿风险，但无法保证 90% 或任何
具体录用概率。编辑初筛、审稿人判断、同期竞争、期刊范围和政策变化均不受
本地代码控制。
