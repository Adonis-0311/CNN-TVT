# IEEE TVT Reviewer #2/#3 攻防终审

> **只读审计快照，修复后需复审。**
>
> 审计日期：2026-07-28  
> 审计范围：`vimd_amc` 当前论文、正式实验冻结配置、数据协议、统计链、结果发布门禁、CSSL-AMC 比较器、复杂度测量、IEEE TVT 模板及专利/Idea/论文边界。  
> 审计约束：本轮仅检查证据与实现一致性；除本报告外未修改项目文件。正式实验结果尚未执行，因而本报告不认可任何尚未生成的定量结论，也不对期刊接收概率作保证。

## 总体结论

当前状态可概括为“强协议骨架、弱结果闭环”。九域 source-disjoint 协议、paired-source 统计、独立信道随机性、cochannel exclusion、内部 evidence lock 和 CSSL-AMC 来源锁具备较好的可审计基础；但正式配置仍标记为 `preregistered_not_executed`，且结果发布链存在多个会直接阻止合法投稿或允许错误结论进入论文的硬缺口。

在以下致命问题修复并重新审计前，不能把当前稿件视为可投稿 release，更不能把内部 7 页 PDF 视为当前源码的模板合规证明。

## 致命问题：交付前必须修复

### F1. Release-lock 版本互斥，合法 release 必然无法通过最终门禁

- `tvt_submission/validate_release.py:28-29` 生成：
  - `vimd_amc.tvt.macro_values.v2`
  - `vimd_amc.tvt.release_lock.v2`
- `tvt_submission/validate_paper_build.py:23-26` 将 `RELEASE_LOCK_SCHEMA` 硬编码为 `vimd_amc.tvt.release_lock.v1`。
- `tvt_submission/validate_paper_build.py:687-726` 在 release 模式中严格拒绝 schema 不一致的锁。

影响：由当前 release validator 正确生成的 v2 锁无法通过 paper-build gate，最终公开构建链在逻辑上不可达。

可执行修复：

1. 将 paper-build validator 与 release validator 统一到同一 schema；若 v2 是当前权威版本，应删除所有残余 v1 读取逻辑。
2. 增加一项端到端自动测试：合成合格 run → 生成 macro manifest → 生成 `results_auto.tex` 和 release lock → release 模式编译 → `validate_paper_build.py` 通过。
3. 测试同时覆盖篡改 `results_auto.tex`、release lock、PDF 或 manifest 后必须失败。

状态：**交付前必须修复，不能等待正式训练。**

### F2. 公开结果表不会被发布系统填充

- `paper/main.tex:743-805` 的公开分支仍直接输出字面值 `generated`：
  - headline table：`paper/main.tex:754-758`
  - OOD/clean table：`paper/main.tex:774-780`
  - mechanism table：`paper/main.tex:794-800`
- `tvt_submission/validate_release.py:32-41` 仅要求八个结果宏。
- `tvt_submission/generate_macro_values.py:883-962` 也仅生成：
  - strongest/reference 名称
  - 三个 gain 类摘要
  - 一个机制摘要
  - 参数量和单一延迟摘要
- `paper/main.tex:823-824` 却声称表格全部由 eligible machine-readable artifacts 生成，且没有手填性能单元格。

影响：即使正式结果完整，公开 PDF 的主要结果表仍会显示 `generated`；现有 macro gate 也无法证明表中每个性能单元格来自锁定 artifact。

可执行修复：

1. 建议生成独立 `results_tables_auto.tex`、`results_figures_auto.tex`，或为全部表格单元格定义有来源记录的宏。
2. 每个单元格应在 manifest 中记录源 artifact、模型、seed 聚合、regime、metric、统计方法与格式化规则。
3. release gate 不仅扫描 `results_auto.tex`，还应扫描公开模式展开后的正文和最终 PDF，拒绝 `pending`、`generated`、`placeholder`、`TBD`、`N/A` 等占位词。
4. 结果表必须与 `metrics.csv`、`headline_paired_statistics.csv`、预测 NPZ 和 run digest 交叉核对。

状态：**交付前必须修复；数值只能在正式运行后生成。**

### F3. Promotion gate 只证明“跑完”，不证明论文的科学主张成立

- `tvt_submission/configs/formal_tvt_freeze_v1.json:93-100` 的 promotion requirements 只有完成性、源树未变、checkpoint、占位符和人工文献审计等行政布尔条件，没有效果方向或最小效果阈值。
- `experiments/run_standard_experiment.py:1789-1964` 的 `submission_release_source_gate()` 检查：
  - execution complete
  - evidence eligibility
  - required models/seeds
  - checkpoint 合法性
  - 指标是否为有限数
  - 统计输出文件是否存在
- 该函数没有检查 hard-region 效果方向、OOD 外推、clean retention、消融、teacher 贡献或机制证据。
- `tvt_submission/generate_macro_values.py:823-974` 会格式化任何有限数值，包括负增益。
- 公开摘要却直接声称：
  - “improves”：`paper/main.tex:94-97`
  - “preserving clean-condition performance”：`paper/main.tex:97-98`

影响：方法显著变差、clean retention 失败或核心机制不成立时，现有流水线仍可能解锁一份语法上虚假的摘要。

可执行修复：

1. 增加 machine-readable `scientific_promotion_decision`，并把它纳入 release lock。
2. 如果 Idea 中拟定的阈值仍是正式决策标准，应在访问正式结果前锁定，例如：
   - hard-interference macro-F1 相对最佳冻结非 oracle 基线至少 `+5 pp`；
   - 至少两个独立 factor-isolated OOD endpoint 达到 `+3 pp`；
   - clean A/C/D 和 B/E 两个分层分别满足点估计不低于 `-1 pp`，层级配对 95% CI 下界不低于 `-2 pp`；
   - A5 相对 A1/A6 等关键消融满足预注册方向与最小效果；
   - teacher、三掩码和机制量达到预注册的支持门槛。
3. 若门槛失败，应 fail closed，或自动切换为明确的中性/否定性叙述，不得继续使用 “improves/preserving”。
4. 把每条论文主张映射到一条可机读 gate，而不是只检查 artifact 是否存在。

状态：**门禁逻辑必须在正式运行前修复；通过与否只能在正式运行后判定。**

### F4. Clean retention 的 A/C/D 与 B/E 分层当前不可计算、不可复核

- 正文明确要求：
  - `paper/main.tex:779-780` 分别列出 A/C/D 和 B/E；
  - `paper/main.tex:811-815` 明确禁止用 aggregate 替代任一分层。
- 缓存已经保存 profile：
  - `src/vimd_amc/standards/cache.py:1346-1351`
  - `src/vimd_amc/standards/cache.py:1630-1638`
- 但预测链丢弃 profile：
  - `src/vimd_amc/metrics.py:13-20` 的 `PredictionBundle` 没有 profile 字段；
  - `src/vimd_amc/evaluation.py:295-310` 的 `predict()` 只保留 probabilities、labels、source IDs、SNR、SIR；
  - `experiments/run_standard_experiment.py:2456-2465` 的预测 NPZ 也未写入 `target_profile_index`。

影响：当前 runner 只能提供 aggregate `clean_retention`，无法完成正文承诺的 seen-profile 与 held-profile 非劣效性检验。

可执行修复：

1. 将 `target_profile_index` 从 cache dataset 贯通至 `PredictionBundle`、预测 NPZ、artifact audit 和统计函数。
2. 对 A/C/D 与 B/E 分别计算模型指标、A5-reference 配对差、source/seed hierarchical CI。
3. 在 release validator 中验证两个 strata 的 profile 集合、source count、无交叉污染以及 clean 条件。
4. aggregate 只能作为补充描述，不能替代两个预注册分层。

状态：**数据字段和统计管线必须在正式运行前修复；数值只能在正式运行后生成。**

### F5. `StrongestBaseline` 与预注册 reference 的含义互相矛盾

- 正式配置将 IQFormer 固定为 reference：
  - `tvt_submission/configs/formal_tvt_freeze_v1.json:45`
  - `tvt_submission/configs/formal_tvt_freeze_v1.json:46` 明确写明 “not described as strongest”。
- runner 同样记录：
  - `experiments/run_standard_experiment.py:2311-2316`
  - `reference_strength_claimed: False`
  - reference 只是 paired-comparison anchor。
- 宏生成器却：
  - 在 `tvt_submission/generate_macro_values.py:699-721` 按 hard-region mean macro-F1 排名所有基线；
  - 在 `tvt_submission/generate_macro_values.py:728-736` 强制实际第一名必须恰好等于固定 reference，否则拒绝生成；
  - 在 `tvt_submission/generate_macro_values.py:885-915` 将其写入 `StrongestBaseline` 并为其使用 IQFormer-reference 的 CI。
- CSSL-AMC 新增后，IQFormer 并无理由必然成为实际最强基线。

影响：若其他基线表现更好，release 因“身份不一致”失败；若继续声称 IQFormer 最强，则违背预注册语义。

可执行修复，二选一：

1. **保留预注册 reference：** 把摘要和宏统一改为 `PredeclaredReference`，不再使用 strongest；另行报告冻结 family 内所有基线结果。
2. **支持 best evaluated baseline：** 在测试访问前预注册“冻结基线族内最大值”的描述规则，并为 A5 相对每一个非 oracle 基线生成层级配对 CI；如用显著性结论，还需处理多重比较。不得把描述性观察到的最佳者表述为“最强已发表方法”。

状态：**正式运行前必须解决语义和统计设计。**

### F6. 正文声称的 classical/oracle controls 未进入正式实验 family

- `paper/main.tex:656-669` 声称预注册 comparator 包括：
  - classical higher-order/cyclostationary classifier
  - clean-input upper control
  - oracle-teacher upper control
- `tvt_submission/configs/formal_tvt_freeze_v1.json:31-43` 的正式模型 family 并不包含这些行。
- `tvt_submission/configs/formal_tvt_freeze_v1.json:102-108` 还明确禁止从其他运行导入 `oracle-control metrics`。
- runner 的标准模型/seed 循环没有执行 classical/oracle control。

影响：论文把没有进入正式证据链的项目称为“predeclared comparator/control”，Reviewer 可直接要求表格结果与可复核 artifact。

可执行修复：

1. 删除或改写这些“已执行、已预注册”的表述；或
2. 在正式测试访问前把 control 纳入单独冻结、可执行、非部署的辅助 family，并使其 artifact 与 headline family 有清晰边界。

状态：**正文与正式配置必须在正式运行前一致。**

### F7. 当前 PDF 落后于当前源码，不能作为模板或版式证据

审计时文件时间：

- `paper/main.tex`：2026-07-28 13:02:14
- `paper/build/main.pdf`：2026-07-28 12:39:09
- `paper/build/main.log`：2026-07-28 12:39:09
- `docs/IEEE_TEMPLATE_COMPLIANCE_AUDIT.md`：2026-07-28 12:42:34

影响：现有 7 页 PDF 和 compile log 并非当前 `main.tex` 的构建结果，不能支撑当前版本合规结论。

可执行修复：

1. 在全部文本和 release-chain 修改后重新编译当前源。
2. 对最终 PDF 逐页渲染检查表格、公式、引用、图注、浮动位置、字体和页数。
3. 重新运行 build validator，并让 audit 文档绑定 source/PDF hash。

状态：**交付前必须重新构建和视觉复核。**

### F8. 本地 AI 政策说明与当前 TVT 专属要求不一致

- `paper/SUBMISSION_READINESS.md:23-28` 使用较宽泛的 IEEE 规则摘要，表述为 AI-generated content 可以披露后使用，并称编辑/语法辅助通常不强制披露。
- `paper/main.tex:850-857` 的内部说明只提到 “generative-AI-assisted language editing”。
- 审计时 TVT 官方 Instructions for Authors 的表述更严格：AI 工具不得替代作者生成文章内容；对于 AI 修改作者生成文本的使用，应按投稿时有效政策在 acknowledgments 中披露。

官方来源：

- https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-authors/instructions

影响：若实际使用范围超过语言编辑，本地说明可能不足或失实。这属于作者责任、出版伦理与投稿合规问题，不能由软件门禁代替。

可执行修复：

1. 由人类作者实质性核验、重写并承担每个段落、公式、结果和引用的责任。
2. 按实际使用范围准备准确披露，不得把内容生成降格描述为仅语言编辑。
3. 投稿当天重新检查 TVT 页面；政策具有时效性。

状态：**内部文档应在交付前纠正；实际合规只能由人类作者在投稿前完成。**

## 高风险问题

### H1. 复杂度与延迟协议远强于当前实现

- 正文承诺：
  - CPU 和 GPU P50/P95；
  - isolated process；
  - warm-up；
  - device、runtime、window length、synchronization policy；
  - 检测并拒绝其他 GPU 进程占用。
  - 位置：`paper/main.tex:722-728`
- 实际复杂度测量：
  - 在训练/评估进程内直接调用：`experiments/run_standard_experiment.py:2436-2441`
  - 当前 device 上 10 次 warm-up 和若干次计时：`src/vimd_amc/evaluation.py:1635-1687`
  - 只记录 `latency_device`，没有独立进程、GPU occupancy 审计、runtime metadata 或 CPU/GPU 双设备结果。
- 正式配置只指定 CUDA：`tvt_submission/configs/formal_tvt_freeze_v1.json:61`。
- 宏生成只公开 A5 参数量与 P50：`tvt_submission/generate_macro_values.py:740-816,944-959`。

建议：

1. 增加独立 benchmark 子进程，固定 batch=1、输入长度、线程数、电源模式、软件版本和同步规则。
2. 分别报告 CPU/GPU P50/P95、参数量、state size、MAC 范围以及 peak memory。
3. 对全部正式模型执行，而非只公开 VIMD。
4. 如果一小时内无法实现，应把正文收窄到 runner 实际测得的单设备 P50/P95，不得保留 isolated CPU/GPU 协议。

状态：**测量协议必须在正式 benchmark 前锁定；数值只能在运行后填。**

### H2. 当前计数不是完整 FLOPs

- `src/vimd_amc/evaluation.py:1576-1633` 的 operation counter 只注册：
  - `Conv1d`
  - `Conv2d`
  - `Linear`
  - `LSTM`
- 注意力矩阵运算、einsum、归一化、激活、softmax 和其他算子不一定进入统计。
- 正文称 convolution/linear MACs 并单列 STFT：`paper/main.tex:722-725`，这比直接称 FLOPs 更准确，但不足以支持跨异构架构的完整计算量结论。

建议：

1. 保持“已计算算子 MACs”的精确标签，列明排除项。
2. 若报告 FLOPs，统一转换规则并覆盖注意力、循环、FFT 和特殊前端。
3. 表中同时给出参数量、MAC 范围、延迟与状态大小，避免用单一 proxy 判断效率。

### H3. 没有锁定的 SNR–SIR 平面结果链

- `paper/main.tex:671-675` 承诺在 full SNR–SIR plane 与 hard region 上报告结果。
- runner 在 NPZ 中保存 `snr_db` 和 `sir_db`：`experiments/run_standard_experiment.py:2456-2463`。
- 但标准结果链只生成 aggregate regime metrics，没有预注册的 cell-level 指标、样本支持数、CI、热图数据或 release 检查。

影响：正式运行后若临时选择有利 cell/区域，缺乏机器门禁阻止 post-hoc cherry-picking。

建议：

1. 在测试访问前固定所有 SNR×SIR cells、最低支持数和空 cell 规则。
2. 自动输出每模型每 seed 的 cell metrics、跨 seed 聚合与支持数。
3. 生成绑定 manifest 的热图和 hard-region 汇总。
4. 对低支持 cell 明确遮罩，不做选择性删除。

### H4. 多重性叙述与实际统计对象不一致

- `paper/main.tex:684-689` 正确说明 Holm 只用于每个 regime×seed family 的 McNemar，并非 headline inference。
- `paper/main.tex:807-815` 的 falsification gate 又称 held-out gains 必须在 “paired multiplicity correction” 后存在。
- 实现中的 Holm：
  - `experiments/run_standard_experiment.py:2646-2686`
  - 对逐 seed、accuracy-based exact McNemar p-value 调整。
- 多 seed headline macro-F1：
  - `experiments/run_standard_experiment.py:2728-2806`
  - 使用 hierarchical paired bootstrap，但没有跨 OOD endpoint 或 candidate 的 multiplicity correction。

建议：

1. 若不增加层级多重比较，正文应明确 Holm 只覆盖 supplemental per-seed accuracy McNemar，删除“held-out macro-F1 gains 经 multiplicity correction”的表述。
2. 若该 falsification gate 必须保留，应为预注册 OOD endpoints 和 candidate family 实现多 seed 层级推断的 multiplicity-adjusted CI 或 p-value。

### H5. CSSL-AMC 架构适配准确，但不是完整已发表方法复现

正面核查：

- 本地实现保留固定提交中的主要拓扑：
  - noise estimator
  - raw IQ 与 estimated noise 拼接
  - `[2,2]` residual stages
  - 固定 `128×512 → 128` readout
  - `128 → 64 → classes` classifier
  - 官方 forward 未使用的 input BatchNorm 参数
- 代码位置：`src/vimd_amc/models/baselines.py:357-467`
- 来源与适配边界：
  - `tvt_submission/sources/cssl_amc_2025.lock.json:55-96`
  - `docs/RECENT_COMPARATOR_AUDIT.md:86-145`
- 正文 `paper/main.tex:659-667` 使用 “official-architecture supervised adaptation”，并明确：
  - 随机初始化；
  - 不载入官方权重；
  - 不执行完整两阶段协议；
  - 不是 structured-interference-specific method。

公平性风险：

- 官方原生过程约为 200 epoch contrastive pretraining + 200 epoch fine-tuning。
- 本地统一预算只有 30 epoch supervised training。
- `docs/RECENT_COMPARATOR_AUDIT.md:139-145` 已正确承认这可能低估完整 CSSL 方法的可达性能。

建议：

1. 所有表格、摘要、图注始终使用 “CSSL-AMC official-architecture supervised adaptation”。
2. 不得称 official reproduction、two-stage CSSL result、strongest recent method 或 interference-specific baseline。
3. 若目标是高强度 TVT 证据，建议在正式测试访问前另行预注册 native two-stage adapted sensitivity；该结果仍不能冒充原论文结果。
4. 主表必须同时给出训练更新量、参数量、MAC 和延迟，避免仅以相同 epoch 声称完全公平。

状态：**现有描述准确；完整方法公平性仍是 Reviewer 高风险点。**

### H6. 仿真证据范围限制 TVT 外部有效性

- `paper/main.tex:826-848` 已明确：
  - simulation-only；
  - 无 SDR；
  - 无 field/deployment；
  - 无完整 V2X trajectory 结论。
- `docs/PATENT_IDEA_PAPER_TRACEABILITY.md:51-77` 清楚记录相对 Idea 的收窄：
  - 标题收窄为 TR 38.901 TDL-profile simulations；
  - 连续轨迹与 SDR layer 尚不存在；
  - cochannel 和 mixed-jammer 另行处理。

评价：这不是内部逻辑违规，且当前边界表述诚实；但对 TVT Reviewer 而言仍是主要外部有效性缺口。若不增加轨迹/SDR证据，应坚持当前窄标题和窄结论，不能在讨论或摘要中回扩为真实 V2X 部署有效性。

### H7. Patent/Idea/论文边界清楚，但公开时点需人工法律审查

- `docs/PATENT_IDEA_PAPER_TRACEABILITY.md:19-29` 已把 patent root、Idea upgrade、paper implementation 和 evidence state 分开。
- `docs/PATENT_IDEA_PAPER_TRACEABILITY.md:51-77` 对科学上必要的 Idea 修正作了显式记录。
- 当前论文没有把专利本身当成科学证据，这是正确的。

剩余风险：

1. 新增 tri-mask、component-power teacher、cross-condition contrastive 与 release pipeline 是否构成需要先行申请的改进点；
2. 投稿时是否需要向编辑或在论文中披露/引用相关专利；
3. 公开论文是否影响后续专利权利。

这些问题必须由合格专利专业人员在公开上传前决定，不能由本审计或软件自动处理。

### H8. `FeatureSIRGain` 名称可能误导

- 公开摘要：`paper/main.tex:98-99` 称 “counterfactual feature-SIR gain”。
- 机制定义：`paper/main.tex:692-715` 明确它是 oracle-conditioned spectral component ratio，不是 waveform SIR、真实 source separation 或现场 SIR estimator。
- 宏生成：`tvt_submission/generate_macro_values.py:935-942` 接受任何有限值并命名为 `FeatureSIRGain`。

建议：

1. 改名为 `OracleSpectralRatioGain` 或等价的精确术语。
2. 摘要同步使用 “oracle-conditioned spectral component-ratio change”。
3. 若没有预注册正向机制门槛，应中性报告变化值，而不是先验称 “gain”。

## 中风险问题

### M1. 当前只为固定 reference 生成层级 CI，无法严谨比较“实际最佳基线”

- `experiments/run_standard_experiment.py:2566-2806` 的所有 paired 和 headline comparison 都以单一 `reference_model` 为锚。
- 当观察到的最佳基线不是 IQFormer 时，不存在 A5 对该最佳基线的预生成层级 CI。

建议：正式测试访问前生成 A5 对全部冻结非 oracle 基线的层级 paired comparison，并预注册解释与 multiplicity 规则。

### M2. 五 seed 的 percentile hierarchical CI 仍可能较粗

- 当前 seed/source hierarchical bootstrap 的 source alignment、class stratification 和不池化 McNemar 的处理是合理的。
- 但仅五个 algorithm seed 时，seed-level 不确定性的 percentile interval 分辨率有限。

建议：保留当前 primary analysis，同时公开每 seed 原始指标、source-only CI、seed-only CI，并增加 paired permutation 或 BCa/稳健敏感性分析。不得用增加 bootstrap draws 伪装增加独立 seed 数。

### M3. Clean quota 是 per-view 而不是 per-source

- `src/vimd_amc/standards/cache.py:703-725` 在 `2 × source_count` 个 paired-view slots 上放置精确 clean quota。
- 同一 source 的两个 view 可能一 clean、一 jammed。

评价：这与 cross-condition training 兼容，但“20% no-jammer quota”容易被理解为 20% sources。正文和数据协议应明确是 per-view quota，并分别报告 clean views 和 affected sources 的比例。

### M4. 标题缩写影响检索清晰度

- `paper/main.tex:49` 标题直接使用 “AMC”。

建议：标题中展开 “Automatic Modulation Classification”，摘要后续再使用 AMC。

### M5. 3GPP 引用应绑定具体版本

- `paper/references.bib` 的 TR 38.901 条目应记录与 MATLAB/cache manifest 一致的 release/version/date，而不是只给通用标准名和年份。

建议：在最终 primary-source audit 中锁定标准版本、MATLAB 实现版本与实际参数映射。

### M6. 页数预算需要在结果加入后重新评估

- 旧 PDF 为 7 页，但结果表、SNR–SIR 图、完整复杂度表、消融和补充说明尚未填入。
- TVT 当前 regular-paper 初投稿限制应在上传当天重新确认；本地 validator 使用 14 页作为初稿门槛是合理的。

官方页面：

- https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-authors/instructions
- https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-authors/instructions/page-charges

## 低风险问题

### L1. 旧日志存在多处 underfull box

旧 `paper/build/main.log` 中存在 underfull box，涉及当时源文件约：

- 155–161
- 164–168
- 365–370
- 490–516
- 655–670
- 832–843

没有观察到 overfull 或 fatal error。但日志已落后于当前源，最终结果表和图加入后必须重新检查。

### L2. 内部 review fail-closed 设计正确，但提交切换必须严格受控

- `paper/main.tex:15-16` 当前为 `\internalreviewtrue`。
- `paper/main.tex:20-26` 的公开构建要求 `EligibleLockedResults`。
- `paper/main.tex:51-58` 的公开分支要求 `authors_verified.tex`。

评价：这是合理的 fail-closed 设计。不得在 formal release、作者信息和人工审计完成前手工切换。

## 已通过的关键审计

### P1. 九域 factor-isolated protocol 已明确冻结

- 九个 split 定义：`src/vimd_amc/standards/cache.py:50-60`
  - train
  - validation
  - id_test
  - hard_interference
  - unseen_jammer
  - unseen_speed
  - heldout_channel
  - combined_ood
  - clean_retention
- seen/held profile、jammer、speed 和 SNR/SIR 网格：`src/vimd_amc/standards/cache.py:61-76`
- 正式 source counts：`tvt_submission/configs/formal_tvt_freeze_v1.json:15-24`

结论：九域角色和预期规模清楚；正式运行尚未发生。

### P2. Source-disjoint 与 paired-source 设计正确

- cache 创建后强制全 split source ID 不相交：`src/vimd_amc/standards/cache.py:957-958`
- cache 载入时再次检查：`src/vimd_amc/standards/cache.py:1563-1564`
- 推理只消费 view1：`src/vimd_amc/evaluation.py:285-311`
- paired statistics 强制 reference/candidate 的 source IDs 和 labels 完全一致：`src/vimd_amc/metrics.py:188-207`
- paired bootstrap 按 source cluster 且可按 modulation class 分层：`src/vimd_amc/metrics.py:211-342`
- multi-seed headline bootstrap 同时重采样 algorithm seed 与 source cluster：`src/vimd_amc/metrics.py:401-581`

结论：paired-source 统计的核心身份链是当前实现的强项。

### P3. Target 与 jammer 信道随机性相互独立且被记录

- 独立 profile RNG：`src/vimd_amc/standards/cache.py:818-825`
- 独立 delay RNG：`src/vimd_amc/standards/cache.py:826-837`
- 独立 target/jammer channel seed 并显式检查不碰撞：`src/vimd_amc/standards/cache.py:838-857`
- profile 和 channel seed 写入 cache：`src/vimd_amc/standards/cache.py:1338-1358`

结论：不存在明显的 target/jammer 共用同一信道随机实现的问题。

### P4. Cohannel 与 mixed-jammer exclusion 有明确科学理由

- `src/vimd_amc/standards/cache.py:79-103` 记录：
  - unanchored in-taxonomy cochannel mixture 的单标签不可识别性；
  - mixed-jammer 缺少冻结 composition policy；
  - 重新纳入所需的物理 anchor、dominant-emitter rule 或独立 ambiguity/composition protocol。

结论：当前排除不是静默删难例，且正文 `paper/main.tex:75-82,826-848` 对 scope 有对应披露。

### P5. CSSL-AMC 来源、拓扑和适配边界可审计

- 固定 commit、immutable URLs、license 与 claim boundary：
  - `tvt_submission/sources/cssl_amc_2025.lock.json`
- 本地实现：
  - `src/vimd_amc/models/baselines.py:292-467`
- 详细 fairness audit：
  - `docs/RECENT_COMPARATOR_AUDIT.md:86-150`

结论：代码层面可以称 official-architecture supervised adaptation；不能称完整 CSSL reproduction。

### P6. Patent、Idea 与论文证据边界总体清楚

- `docs/PATENT_IDEA_PAPER_TRACEABILITY.md:19-29` 将技术血缘和当前 evidence state 分开。
- `docs/PATENT_IDEA_PAPER_TRACEABILITY.md:31-49` 明确保留的 Idea 指令。
- `docs/PATENT_IDEA_PAPER_TRACEABILITY.md:51-77` 明确科学上必要的收窄和修正。

结论：当前论文没有把专利文字直接当作实验事实，也没有把缺失的 SDR/trajectory 证据伪装为已完成。

### P7. IEEE 模板基础使用正确

- `paper/main.tex:1` 使用 `\documentclass[letterpaper,journal]{IEEEtran}`。
- 审计时 `paper/IEEEtran.cls` 的 SHA-256 与用户提供的 IEEE Transactions 模板包中 class 一致。
- 旧日志确认 IEEEtran V1.8b，成功输出 7 页，无 overfull、undefined-reference 或 fatal compile error。

限制：旧 PDF 落后于当前源码，因此该结论只证明 class/template 基础正确，不证明当前或最终 release 的版式已经通过。

## 交付前修复清单

下列工作不依赖正式模型结果，必须优先完成：

1. 统一 release-lock schema，并添加端到端 release build 测试。
2. 实现完整自动结果表/图导出，禁止公开 PDF 中残留占位符。
3. 将 `target_profile_index` 贯通至预测和统计链，落实 clean A/C/D 与 B/E 两分层。
4. 增加科学 promotion gate，并把每条公开主张绑定到可机读条件。
5. 统一 “predeclared reference” 与 “strongest evaluated baseline” 的定义和统计协议。
6. 让正文 comparator/control 声明与 formal family 一致。
7. 增加 SNR–SIR cell-level exporter、支持数和锁定热图。
8. 使 multiplicity 文字与实际统计对象一致。
9. 实现正文承诺的复杂度/延迟协议，或收窄正文至真实实现。
10. 更正 TVT AI 政策说明。
11. 使用当前模板重新编译当前源码并完成逐页视觉 QA。

## 只能在正式运行后填充的内容

以下内容不得预填、手填或从 screening/diagnostic/oracle 运行导入：

1. 所有模型的 accuracy、macro-F1、worst-class recall、NLL、ECE。
2. hard-interference 与各 factor-isolated OOD 的配对效果和 CI。
3. clean A/C/D 与 B/E 两分层非劣效性结果。
4. SNR–SIR 平面、支持数和热图。
5. A0–A7 消融结论。
6. teacher agreement、第三路线、`alpha`、`beta` 和 oracle spectral ratio 等机制证据。
7. 各模型参数量、MAC、state size、CPU/GPU P50/P95 和 peak memory。
8. scientific promotion decision 的最终通过/失败状态。
9. 公开摘要、结论和结果表中的所有定量叙述。

## 必须由人类在正式投稿前完成

1. 核实作者、单位、基金、利益冲突和 acknowledgments。
2. 逐条核验数学陈述、实验事实与 primary-source 引文。
3. 按实际使用范围完成符合投稿时 TVT 政策的 AI disclosure。
4. 由专利专业人员审查论文公开时点、改进点申请与专利披露。
5. 确认最终标题、scope、页数、稿件类型和投稿系统要求。
6. 阅读并承担全文作者责任；自动门禁不能替代作者判断。

## 最终判定

- 协议与代码基础：**有潜力，部分关键链条已达到可审计水平。**
- 当前正式证据：**未执行。**
- 当前公开 release：**不可生成/不可验证。**
- 当前稿件投稿状态：**阻断。**
- 解除阻断条件：完成 F1–F8 的适用修复，重新运行静态与端到端测试，执行冻结正式实验，生成完整 machine-bound 结果，重新编译并复审。

