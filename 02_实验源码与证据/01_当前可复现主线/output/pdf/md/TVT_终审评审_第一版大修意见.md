# 终审评审报告（Nature-style 三盲审 + 交叉综合）— 第一版大修意见

> 评审对象：`tvt_operating_envelope_integration.pdf`（7 页，IEEE TVT 风格，匿名）
> 标题：An Occupancy-Indexed Operating Envelope for Compact Physics-Guided Automatic
> Modulation Classification under Structured Interference
> 评审框架：`nature-reviewer` skill（3 份互盲评审报告 + 1 份事后交叉综合）
> 评审语言：中文（技术术语、模型名、ID 保留英文）

---

## Review setup

- **Input scope（输入范围）**：完整稿件（7 页 PDF 全文 + LaTeX 源码 `main.tex` + 解析后数值宏 `paper_macros.tex`），含摘要、正文全部小节、Table I–IV、Figure 1–7 及图注、全部解析数值。
- **Assessment boundary（评估边界）**：图的像素内容除图注与嵌入数字外不可核查；无作者 rebuttal、无补充材料、无代码执行、无逐 seed 原始数据文件。据此无法评估的项一律标注，不臆造。
- **Shared manuscript claim summary（共同主张摘要）**：在结构化干扰下，紧凑物理引导谱表示（VIMD-Net / A5，39.5k 参数）相对大容量 I/Q 分类器（354k–406k 参数）的优势可由一条 occupancy–SIR 两变量经验包络索引与预测；该优势在 severe SIR 角落可复现，clean 边界是"表示缺口"且可经 received-I/Q sidecar 修复；稿件明确声明"simulation only、非 uniform superiority"。
- **Visible evidence base（可见证据库）**：sealed confirmatory family（Table I，唯一正效应 A5 vs A0 +4.57 pp [3.65, 5.49]）；exploratory 的 severe-corner（+9.74 / +10.46 pp）、envelope（sign 0.975 / r=0.694 / RMSE 6.20 pp，holdout 81 cells）、fusion、probe–coverage–sidecar 修复链、capacity ladder（Table III–IV）。
- **Missing materials affecting confidence（影响信心的缺失材料）**：逐 seed 原始数据与完整 provenance 字段（path / row selector / precision / SHA-256）未随稿提供；图内数值仅能靠图注与解析宏核对。

---

## Reviewer 1（侧重：技术可靠性 / 技术缺陷）

- **总体评价**：稿件提出明确且有价值的问题，证据分层与诚实陈述异常克制透明，severe 干扰角落结果在统计上有支撑。主要技术保留在于：唯一正面 sealed 结果（A5 vs A0）在容量与捆绑项上混淆，三路由/教师机制未被确认性证据建立；包络外推验证缺机会水平基线，且招牌 severe 角落是 in-sample；统计程序说明不足难以独立复核。诚实但结论自限，技术缺陷需解决后才能完整立论，但不因此否定其描述性贡献。

- **谁会关心这些结果，为什么**：AMC / 无线通信 / 频谱监测研究者（部署取舍、operating envelope 比平均精度更可操作）；资源受限推理（轻量、边缘部署）从业者（39.5k 参数在严重占用干扰下反超大参数模型 + I/Q sidecar 修复路径）；可复现 ML 读者（artifact 溯源 + sealed/exploratory 分层）。跨学科吸引力有限，集中在 wireless/ML 交叉处。

- **主要优点**：① 证据分层与诚实结论；② 严重干扰角落结果有内部一致性支撑（10/10 seeds 同号、12/64 cells 过 Holm 且全在 severe 行、与全图一致）；③ 修复链逻辑清晰（probe → coverage → sidecar → capacity ladder，每个正负结果区分竞争解释）；④ 数值自洽性总体良好（break-even 可由 β0/β1/β2 反解验证 0.58/0.87/0.48/0.66 全部吻合）。

- **主要关切（Major Concerns）**

**R1-M1 [technical soundness / mechanism-evidence]**
严重程度 Major ｜ 阻断 No
主张指针：紧凑物理引导谱分配的机制（三条 soft 分配路由 + simulation-component margin teacher）是 severe 区域优势的来源。
证据指针：Section IV-B（p2）、Section V-A 与 Table I（p3）、Abstract（p1）、macros ParamsAZero=8458 vs ParamsAFive=39500。
关切：sealed 三对比中只有 Full method vs A0 为正（4.57 pp），margin vs proportional teacher（0.16）与 tri-route vs dual-route（0.44）同时区间均跨零，family gate 未通过；而 A5 vs A0 把参数从 8458 提到 39500（约 4.7 倍）并同时捆绑多任务目标、exact-source 对比损失、残差 bypass 等全部改动，+4.57 pp 甚至不能归因于"整个 method"，可能纯粹是容量或任意捆绑项。唯一正 sealed 结果无法把"physics-guided allocation"与"更大的网络"区分开。
为什么重要：创新点与标题都围绕 physics-guided 与机制，而 sealed 证据既未隔离教师、也未隔离三路由、更未在等容量下隔离方法整体，机制主张未被确认性证据建立。
解决判据：提供容量匹配的 sealed 对比（A0 扩到 39500 参数，或 A0+除分配机制外全部非机制项）；或把机制改述为 exploratory，并列出 A5 与 A0 之间除分配机制外的所有差异及其贡献/合计上界。

**R1-M2 [statistical-rigor / reproducibility]**
严重程度 Major ｜ 阻断 No
主张指针：non-studentized max-absolute-deviation 分层配对 bootstrap 的 simultaneous 区间、1.31 pp design resolution、sign-flip 概率 0.0018 支撑了报告的效应量与鲁棒性。
证据指针：Section IV-B（p2）、Section V-A/B（p3）、macros ProtocolBootstrapDraws=10000、DesignResolution=1.31、SevereGainIqformerSignFlipP=0.0018。
关切：四点。（a）正文未报 bootstrap 抽样次数（宏为 10000），层级结构如何落地（seeds 与 class-stratified source clusters 的嵌套、seeds 固定还是随机、cluster 如何重抽样）未说明；（b）"design resolution 1.31 pp" 是单一未推导标量且被标为 exploratory，却被用来解释 sealed 结果大小；（c）sign-flip 0.0018 低于 10/10 同号双侧符号检验精确值 2/1024≈0.00195，检验统计量与精确/蒙特卡洛实现未说明；（d）4.57/1.31≈3.49 被写成 "about three times"。
为什么重要：头条统计结论无法独立复核，削弱 severe 角落与确认性结果可信度。
解决判据：完整报告 bootstrap 方案（抽样次数、重抽样单元、层级嵌套、seeds 处理），给出 1.31 pp 的推导或出处，明确 sign-flip 检验统计量与精确/蒙特卡洛 p；"about three times" 改为"约 3.5 倍"。

**R1-M3 [claim-moderation / experimental-design]**
严重程度 Major ｜ 阻断 No
主张指针：occupancy–SIR 包络 "predicts their sign outside the regimes used to fit it"，在 81 个未见 regime cells 上取 0.975 sign agreement。
证据指针：Abstract（p1）、Section V-C（p3–4）、Fig. 4（p5）、macros EnvelopeHoldoutCells=81、EnvelopeIqformerHoldoutSign=0.975、R=0.694、Rmse=6.20。
关切：三点。（a）81 holdout cells 大多来自 compact 模型输掉的 regime，"总是预测多数符号"的平凡基线可能已取得高 sign agreement，但稿件未报符号分布 prevalence，也无 chance-adjusted 指标（balanced sign accuracy / Cohen's kappa）；（b）招牌 SIR=−15 dB 严重角落属于拟合时的 hard-interference cells，即 in-sample，故"预测未见 regime 符号"的口号不覆盖最亮的胜利；（c）RMSE 6.20 pp 与头条增益（9.74/10.46 pp）同量级，量级预测能力有限，只是符号方向尚可。
为什么重要：贡献一（"可预测符号的 operating envelope"）在摘要中被高估。
解决判据：报告 81 cells 符号 prevalence，加入"预测多数符号"平凡基线与 chance-corrected 指标；明确 severe SIR corner 是 in-sample；收窄 "predicts their sign outside the regimes used to fit it" 措辞，或在真正 held-out 的 SIR/occupancy 范围验证 severe corner 符号预测。

- **次要意见（Minor Comments）**

**R1-m1 [technical soundness]** 受影响要素：IQFormer-inspired 参考 macro-F1 取值。证据指针：Section V-B（p3）、Table IV（p7）、macros AbsIQFormerHard=0.4872 与 CapacityLHardVsIqformerReference=49.61。问题：正文 sealed 边际与 sidecar 参考均为 48.72（10 seeds），capacity ladder 的 "L minus IQFormer" 参考值却是 49.61（5 seeds），相差 0.89 pp 且 Table IV 未说明；MCLDNN 同样有 0.4599 vs 45.41 差异。修正：标注 Table IV 参考值的 seed 数与精确参考，或统一用 10-seed 参考重算。

**R1-m2 [technical soundness]** 受影响要素：sign agreement 与 r 精度。证据指针：Section V-C（p4）、Fig. 4 图注（p5）。问题：正文 sign 0.975（97.5%）与 r=0.694，图注写 "sign 98%" 与 "r=0.69"（底层 79/81≈97.53% 可被同时截断/四舍五入，但两处精度不一）。修正：统一精度，如全文写 79/81（97.5%）与 0.694。

**R1-m3 [claim-moderation]** 受影响要素：design resolution 使用与 A7 差值。证据指针：Section V-A（p3）、Fig. 5（p5）。问题：A7 vs A5 具体差值及区间未给，"well below the design resolution" 无法核验；4.57/1.31≈3.49 写成 "about three times"。修正：给出 A7 vs A5 数值与区间；改"约 3.5 倍"。

**R1-m4 [reproducibility]** 受影响要素：fusion 互补对照设置。证据指针：Section V-D（p4）。问题："same-architecture seed-ensemble control +1.49 pp" 未说明架构、seed 数、ensemble 规模、校准方式。修正：补全对照设置。

**R1-m5 [readability]** 受影响要素：probe 阈值定义。证据指针：Section V-E（p5）。问题："preregistered high-order threshold 0.25" 未定义 probe 度量（accuracy/F1/R²）与取值范围。修正：定义度量、范围与 0.25 单位。

**R1-m6 [figures-and-tables]** 受影响要素：Fig. 2 星号与 Holm 计数。证据指针：Fig. 2 图注（p4）、Section V-B（p3）。问题：图注称星号标记 positive lower paired interval，severe 行合计 15 星号，正文却称 12/64 cells 过 Holm；二者关系（星号是否 Holm 幸存标记）不明。修正：图注说明星号对应未校正区间还是 Holm 幸存。

**R1-m7 [reproducibility]** 受影响要素：artifact provenance 可核验性。证据指针：Section VII（p6–7）。问题：正文称每个 macro 存 path/row selector/precision/SHA-256，但所供宏只含数值与 evidence class 标签，无这些字段。修正：提供完整 provenance 字段或说明公开层级。

**R1-m8 [data-resource-quality]** 受影响要素：CSSL 比较器解释。证据指针：Section V-F 与 Fig. 5（p5）、macros AbsCSSLHard=0.3915 vs AbsAZeroHard=0.3949。问题：8.6M 参数 official CSSL 在 hard 上低于 8.5k 参数 A0，正文未讨论，CSSL 又被用作单调增益参照。修正：一句话说明 CSSL 低分是已知局限还是复现/调参不足。

- **在作者立论成立之前必须解决的技术缺陷**：无 Blocking Yes。R1-M1（机制证据硬缺口）、R1-M2（统计可复核性缺口）、R1-M3（外推验证强度缺口）三条 Major 均需解决后才算完整立论，且 R1-m1/R1-m2 数值不一致需一并消除。

- **对照 Nature 式标准的评估**：originality 中等到良好（原创在评估框架，机制层面未被证据支撑）；scientific importance 中等（simulation only、结论被两个 gate 自限）；interdisciplinary readership 有限；technical soundness 是保留重点（R1-M1–M3 + R1-m1–m8）；readability 中等偏上（统计部分晦涩、关键度量定义不清）。

- **建议姿态**：Supportive if technical concerns are resolved。值得给修稿机会，但机制主张在 sealed 证据下尚未成立，统计程序与外推验证需补强。

---

## Reviewer 2（侧重：原创性 / 科学重要性）

- **总体评价**：稿件纪律性极强且诚实得少见，数值自洽检查未发现矛盾，是重要加分项。但以 originality 与 significance 为重的评审下，核心问题在"贡献实质"与"证据能支撑的分量"不匹配：唯一 sealed 且为正的是 A5 对自身 stripped backbone A0 的 4.57 pp，机制无法归因到任何单个新组件；最有传播力的数字（0.975、9.74/10.46、11.77）全部是 exploratory/post-hoc。结论分量明显弱于摘要与 contribution 列表的排布所暗示的水平。

- **谁会关心这些结果，为什么**：AMC / spectrum monitoring / cognitive radio 社区，尤其"轻量物理引导何时优于大 I/Q 模型"的 ML-for-wireless 研究者；做 benchmark 严谨性研究者（full-region map、frozen artifact、preregistration+negative control 有方法参考价值）；"用物理协变量预测模型优势符号"对 ML 社区有潜在吸引力，但未提炼为可迁移方法，跨领域吸引力目前是潜在而非已实现。

- **主要优点**：① 诚实与 claim moderation；② 证据追踪（macro 存 path/selector/class/precision/SHA-256，figure 由 CSV 渲染）；③ 防选择性报告（Fig. 2 全图、Fig. 6 全曲线）；④ 统计纪律（joint bootstrap、Holm、sign-flip、preregistered directional test、negative control）；⑤ 数值自洽（Table I/II/IV、Eq. (7) break-even 反算一致）。

- **主要关切（Major Concerns）**

**R2-M1 [originality]** Major ｜ 阻断 No
主张指针：核心 advance 是 occupancy–SIR envelope 与三路分配 + simulation-component teacher，并区别于既有 AMC 与 time–frequency masking 工作。
证据指针：Section II（p1–2）；"The main distinction of this work is evaluative"（p2）；"Our use differs in purpose"（p2）；contribution 列表（p1）。
关切：稿件自认 novelty 不在 masking/contrastive/attention，而在 evaluative。三路 softmax 分配是 attention/soft-masking 与 multi-stream AMC 常规成分重组，teacher（Eq. 5–6 的 [q_s−q_j]_+）是监督信号设计，envelope（Eq. 7）只是两变量线性回归且自述 non-causal。Related Work 未引用任何关于 AMC 或信号处理中 regime/operating-envelope 刻画或 condition-dependent model comparison 的既有工作，无法判断 evaluative framing 究竟新在哪、相对谁新。
为什么重要：对 Nature 式 venue，originality 是门槛标准；若 advance 主要是呈现方式而非方法/可证机制，novelty 主张需大幅收窄。
解决判据：区分"方法学新增"与"评估方式新增"；补充并对比至少一类 condition-dependent/regime-dependent 模型比较或 operating-region 刻画工作；否则将 novelty 降级为"一种系统化的评估与审计框架"并收窄标题/摘要。

**R2-M2 [scientific importance / significance]** Major ｜ 阻断 No
主张指针：确立 physics-guided 机制（three-route allocation + simulation-component teacher）及其 reproducible operating region。
证据指针：Table I（p3）、Section V-A（p3）、Section IV-B（p2–3）、Evidence status（p1）、macros MethodEffectDiff=4.57、TeacherFormDiff=0.16、TriRouteDiff=0.44、ComponentEffectBound=1.36、DesignResolution=1.31。
关切：唯一 sealed 正对比是 full method vs A0 的 4.57 pp；两个"新组件"对比（0.16、0.44）都没过零，joint family gate 未通过，单个组件效应在 design resolution（1.31 pp）量级以下。sealed 证据只能确立"VIMD family 在该预算下优于 stripped backbone"，不能确立 margin teacher 或三路分配有任何贡献，更不能确立 physics-guided 具体机制。而摘要里最有分量的数字全部是 exploratory/post-hoc。
为什么重要：significance 的核心——标题与摘要把 advance 锚定在 physics-guided 三路分配与 teacher 上，但 sealed 证据无法把增益归因于它们。
解决判据：提供能分离机制的新 sealed/confirmatory 对比（teacher on/off、route vs no-route，足够 resolution），或把显著性主张重述为"whole-method 对 backbone 正效应已确认，组件机制未解决"，让 sealed 与 exploratory 结论分量严格对应。

**R2-M3 [technical soundness / significance]** Major ｜ 阻断 No
主张指针：occupancy–SIR law 把 gain 索引为物理条件并可作 model-selection diagnostic。
证据指针：Eq. (7) 与 Table II（p4）；"Because occupancy and SIR co-vary…not a causal structural equation"（p4）；Discussion "jammer occupancy is estimated from simulator-tracked components"（p6）；Limitations（p6–7）。
关切：occupancy 与 SIR 设计中共变，且 occupancy 来自 simulator-tracked components、部署时不可得；"future…estimate deployment-available occupancy proxies" 尚未做。标题 "occupancy-indexed" 与 "直接可作 model-selection diagnostic" 把归因强度与可用性说得比证据更满——目前是用不可部署变量拟合的描述性曲面，而非可操作部署诊断量。
为什么重要：冲击"region 可由物理协变量预测"的可迁移性与 significance。
解决判据：明确 envelope 当前需 ground-truth occupancy、不可部署；补充 deployment-available occupancy proxy 并在 held-out regimes 报告其表现，或把 "diagnostic" 降级为 "simulation 内审计/描述工具"。

**R2-M4 [scientific importance / mechanism]** Major ｜ 阻断 No
主张指针：以 "physics-guided" 机制为卖点，报告 gain 随 occupancy 单调上升的经验方向。
证据指针：Section V-C（p4）："The preregistered mechanism expectation was non-increasing gain with occupancy. The sealed directional test rejected that sign. Exploratory quintiles instead show monotonically increasing A5–CSSL gain"。
关切：preregistered 机制期望（gain 随 occupancy 非递增）被 sealed directional test 拒绝，实际方向（单调递增）来自 exploratory quintiles。即 sealed 证据与物理直觉相反，支撑新方向故事的是 exploratory/post-hoc 证据。稿件诚实报告，但没有给出"为什么物理直觉错了、新方向在机制上意味着什么"，也没把反转当作对 physics-guided 标签本身的检验。
为什么重要：削弱 "physics-guided" 可信度——若引导设计的物理直觉被 sealed 测试否定，"物理引导"这一新颖性叙事需重新论证。
解决判据：提供机制层面解释或新实验说明为何 gain 随 occupancy 上升，并与被否定的 preregistered 期望在物理上调和；否则在结论中明确 "physics-guided" 仅指 teacher 构造来源，而非被证实的方向性机制。

**R2-M5 [scientific importance / compactness]** Major ｜ 阻断 No
主张指针：compact spectral 表示的 clean boundary 是 architectural 且可修复，而非 compactness 的不可避免代价。
证据指针：Section V-E（p5–6）、Fig. 7（p6）、Table III（p7）、macros TransferFailShareCompact=100、TransferFailShareHighCapacity=11、SidecarParameters=46794、SidecarCleanVsAFiveDiff=11.77。
关切：clean-retention gate 未通过：6 个未覆盖 clean 窗口类 compact-spectral transfer failure 100%（I/Q 高容量 11%）；修复靠增加 46794-parameter received-I/Q sidecar，即把 pure spectral 补上 I/Q 通道 clean 才修复 +11.77 pp。这实质上承认 pure spectral 表示对 clean 星座阶数判别不足，"compact" 卖点被部分削弱；且整条 probe–coverage–sidecar 链是 exploratory Tier-2。
为什么重要：影响"clean boundary 是 architectural"主张力度与"compact physics-guided spectral allocation"作为独立轻量方案的可信度。
解决判据：明确 sidecar 后总参数量与 MACs 并与 baselines 对比；把 "clean boundary 是 architectural" 限定为 exploratory 支持。

- **次要意见（Minor Comments）**

**R2-m1 [technical soundness]** "about three times" 实为 3.49，改"约 3.5 倍"（p3，macros 4.57/1.31）。
**R2-m2 [readability/consistency]** 正文 r=0.694、sign 0.975 与 Fig. 4 图注 r=0.69、sign 98% 格式/精度不统一。
**R2-m3 [technical soundness]** MCLDNN envelope 只报 sign 0.951，隐去 r=0.511 / RMSE=8.05（明显弱于 IQFormer 的 0.694/6.20），属不对称报告。
**R2-m4 [readability]** 内部 ID（A0/A5/A7、CSSL、S/M/L、Tier-2、ID/combined OOD）密集且无 glossary。
**R2-m5 [technical soundness]** sidecar 后总参数量未给（SidecarParameters=46794 vs ParamsAFive=39500）。
**R2-m6 [readability]** "MCLDNN" 与 "IQFormer-inspired" 命名不对称，易误导复现程度；"anchors, not exact reproduction" 只出现一次。

- **技术缺陷汇总**：无 Blocking Yes；但 R2-M2/M3/M4 叠加意味着 sealed 证据只支撑 modest 的 whole-method-vs-backbone 正效应，而 novelty 核心机制未归因、preregistered 方向被否定、envelope 依赖不可部署变量且共变。significance 目前 asserted 而非 supported，须在机制分离、方向解释、部署代理三方面补齐。

- **对照 Nature 式标准**：originality 偏弱（novelty 主要在 evaluative framing，缺针对性 related work）；significance 被 framing 放大、sealed 支持不足（结构性张力）；interdisciplinary readership 对 AMC 有价值、跨领域潜力有限；technical soundness 强（artifact 追踪、防选择性报告），主要问题是机制归因不足、共变、不可部署变量、精度/对称报告小问题；readability 整体清楚，但 ID 密集无 glossary、baseline 命名易误解。

- **建议姿态**：Supportive only if the originality and significance concerns are resolved。不质疑可复现性或数值有效性，也不认为中心结论被证伪；但 significance 必须由 sealed/confirmatory 证据承担，而非 exploratory 数字承担。

---

## Reviewer 3（侧重：跨学科受众 / 非专家可读性）

- **总体评价**：认真执行了一个有价值的问题，claim moderation 与证据透明度罕见且可作范本，数值自洽性良好。主要不足在跨学科传达：可迁移的方法学贡献（envelope 评估范式、证据分类纪律、repair chain）从未被提炼给非 AMC 读者；摘要、Evidence status box 与大量未解释记号（A0/A5/A7、S/M/L、IQFormer-inspired、CSSL、macro-F1）使非专业读者难以抓住正面 headline。技术层面，唯一 sealed 阳性是对共享 backbone（A0），最醒目的 severe-corner 优势是 post-hoc/exploratory；envelope 幅值预测较弱。这些都被诚实披露，但削弱 "compact beats large" 的 headline 强度。

- **谁会关心这些结果，为什么**：① ML 评估/基准测试与 reproducibility 工程社区（sealed/exploratory 分离、如实报告未通过 gate、SHA-256 溯源、拒绝单一 marginal ranking，是对刷榜式 SOTA 文化的解毒剂）；② 科学机器学习 / physics-guided vs data-driven 社区（"便宜带物理先验的表示何时足以胜过黑箱"在气候、材料、CFD 代理模型同样成立）；③ CV/NLP 中研究小模型对大模型权衡、domain shift 下 rank 交换的研究者（repair chain 是通用诊断流程）。遗憾是 Discussion/Conclusion 未为这些场外读者架桥。

- **主要优点**：① 罕见 claim moderation 与证据纪律；② 问题表述（rank exchange across physical plane）优于又一个准确率表；③ 诊断修复链是真正的机制分析；④ 数值自洽性优秀；⑤ Limitations 明确具体、reproducibility 边界扎实。

- **主要关切（Major Concerns）**

**R3-M1 [interdisciplinary readership interest]** Major ｜ 阻断 No
主张指针：用物理协变量索引性能、sealed/exploratory 证据分离的评估方法，结论对 AMC 之外研究者同样有意义。
证据指针：Section VI Discussion 与 VIII Conclusion（p6–7）全程仅讨论 AMC 模型选择，未提场外迁移；Section IV 也未抽象为通用方法论。
关切：可迁移方法学价值被完全封装在 AMC 术语内。operating envelope、sealed confirmatory family、repair chain 三个概念本可各自成为跨领域读者关心的点，但 Discussion 只谈在何种干扰下选哪个模型。
为什么重要：Nature 式标准要求结论对跨学科读者有意义；当前把本可触达 ML 评估/科学机器学习/CV-NLP 读者的文章降格为纯无线领域增量工作。
解决判据：在 Discussion/Conclusion 增加非 AMC 例子，说明"把性能索引到物理协变量而非对条件取平均"是通用评估原则、"sealed/exploratory 分离与 artifact 可追溯"可移植，并展开 operating envelope 的工程类比（飞行包线/安全工作区）。

**R3-M2 [readability for nonspecialists]** Major ｜ 阻断 No
主张指针：读者能从摘要与正文平滑追踪哪个模型是哪个、赢了什么。
证据指针：Abstract 与 Section III–V 混用 VIMD-Net 与 A5、shared backbone 与 A0、residual-free A7、S/M/L 与 A5 关系均未建立显式映射（p1、3、5、6）。
关切：摘要用 VIMD-Net/shared backbone，Results 改用 A5/A0，读者须自行推断 VIMD-Net=A5 且 A5≠S（同为 39500 参数但 hard 分别为 44.06 与 43.59）；A7 只在 V.A 冠以 residual-free 一次；A1–A4、A6 从未出现却被 Fig. 7 的 (A0–A7) 暗含；CSSL 无展开无引用；macro-F1 从不定义。
为什么重要：记号系统是证据纪律的载体，非专业读者无法解析 A0/A5/A7 与 S/M/L 对应关系，就无法判断 headline 里 compact beats large 到底比什么。
解决判据：提供记号表（VIMD-Net=A5、A0=shared direct spectral backbone、A7=residual-free、S/M/L 参数与 A5 关系、CSSL 全称出处），并说明 A5 与 S（同为 39500 参数）的差异来源。

**R3-M3 [readability for nonspecialists]** Major ｜ 阻断 No
主张指针：摘要与 Evidence status box 能让非专业读者迅速抓住正面 headline。
证据指针：Abstract 全段 + Evidence status box（p1）。
关切：摘要极长且以负性 gate 框架为主导（did not pass、exploratory、outside frozen confirmatory family 收束），正面结果被埋；Evidence status box 在 Introduction 前使用 confirmatory family/clean-retention gate/post-hoc/Tier-2/frozen 等未定义黑话，且与摘要末尾几乎重复。
为什么重要：摘要是跨学科读者唯一入口；headline 被负面框架淹没则读者流失。诚实性与可读性在此产生张力，当前用可读性为诚实性付出过高代价。
解决判据：重写摘要，开头用通俗语言陈述正面发现，gate 状态压缩为一句并移到中后部；Evidence status box 改写为面向非专业读者的一句话式 "what is solid vs what is preliminary"。

**R3-M4 [technical soundness]** Major ｜ 阻断 No
主张指针：SIR=−15 dB 下 compact 模型以 9.74/10.46 pp 显著优于更大 I/Q 模型，是本文 headline。
证据指针：Abstract（p1）、Section V.B（p3）、Fig. 2–3（p3–4）、Section IV.B（p2）。Table I 唯一 sealed 阳性是对 A0 的 4.57 pp。
关切：标题与摘要最醒目、最可能被引用的数字（+9.74/+10.46）来自 post-hoc/exploratory 而非 sealed family。sealed 只证明完整方法优于自身共享 spectral backbone（A0），未证明优于 MCLDNN/IQFormer-inspired 这类大容量 I/Q 基线。作者诚实，但 framing 仍把 exploratory 的 severe-corner 胜利放在比 sealed 结果更突出位置，存在被二次引用者读作已确认 compact > large 的风险。
为什么重要：削弱 "compact beats large at severe SIR" 核心推论强度，暴露缺少一个 preregistered 的 compact vs large 在 severe SIR 的确认性对比。
解决判据：在摘要与 V.B 首句把 severe-corner 数字显式标注为 exploratory/post-hoc 并说明与 sealed A5–A0 结果的证据等级差异；或声明一个 preregistered compact vs large 在 severe SIR 的确认性对比作为未来工作。

**R3-M5 [technical soundness]** Major ｜ 阻断 No
主张指针：occupancy–SIR envelope 能 predict their sign outside the regimes used to fit it，并支撑标题 occupancy-indexed。
证据指针：Section V.C 与 Eq. (7)（p3–4）、Fig. 4（p4–5）。out-of-regime r=0.694、sign 0.975、RMSE 6.20（IQFormer）；MCLDNN sign 0.951、r=0.511、RMSE 8.05。occupancy 操作定义仅在 Discussion 一句带过。
关切：① 幅值预测弱——RMSE 6.20 pp 与 headline 效应量（9.74/10.46 pp）同量级，r 中等，MCLDNN RMSE 8.05 pp 接近其效应本身，predict sign 成立而 predict size 勉强；② occupancy 与 SIR 共变（作者承认 Eq. (7) 非因果），且 occupancy 确切定义（哪个时频单元、何种阈值、如何归一化）未给出，使标题关键词无法被外部复现。
为什么重要：标题把 occupancy 作首要索引变量，但共变、操作化缺失、幅值预测偏弱三者共同削弱核心卖点。
解决判据：补充 occupancy 精确定义与计算流程；区分 sign 与 magnitude 两种预测质量并分别报告；突出 occupancy 与 SIR 无法解耦的限定，或给两者独立化的前瞻设计。

- **次要意见（Minor Comments）**

**R3-m1 [readability]** IQFormer-inspired 称谓：Fig. 5/6/7 轴标签写作 IQFormer，与 [7] 关系不明；统一为 IQFormer-inspired 或说明差异。
**R3-m2 [readability]** CSSL 无全称无引用（约 8.6M 参数）却不在 354984–406070 baseline 区间；给出全称出处并说明为何排除。
**R3-m3 [technical soundness]** severe-corner 聚合口径：+9.74 pp over MCLDNN at SIR=−15 未说明 SNR 条件/聚合方式；按 Fig. 2 severe 行 8 cell 简单平均约 10.75 pp 与 9.74 不符，暗示未披露配对/聚合方式。明确 SNR 条件与 per-cell / source-paired 聚合，说明与 Fig. 2 单 cell 关系。
**R3-m4 [technical soundness]** capacity ladder seed/参考不一致：基于 5 seeds（主结果 10 seeds），IQFormer 参考 49.61 vs 主文 48.72、MCLDNN 45.41 vs 45.99、S tier hard 43.59 vs A5 44.06，且 S(39500) 与 A5(39500) 关系未说明。披露 seed 数、参考配置与 S/A5 差异。
**R3-m5 [readability]** macro-F1 通篇使用从未定义；首次出现处加一句定义。
**R3-m6 [readability]** 摘要/正文 r=0.694、sign 0.975 与 Fig. 4 注 r=0.69、sign 98% 精度不一致。
**R3-m7 [readability]** Section V.D "39500 parameters and 41.8 million MACs, versus 354984 and 355.6 million" 单位不明，且缺 MCLDNN 对应值（406070/398.2M）；标注单位并补齐。
**R3-m8 [technical soundness]** compact 动机与延迟：仅报 parameters/MACs 未报可得 latency（A5 3.172 vs MCLDNN 3.789 ms，仅约 16% 优势，与 10 倍参数差不成比例）；补延迟或把 compactness 限定为 parameters/MACs 而非端到端推理成本。

- **技术缺陷汇总**：无 Blocking Yes。优先 R3-M1 与 R3-M2/M3（跨学科可读性兑现）；技术层面 R3-M4/M5 使 headline 证据等级与表述一致并补强 envelope 定义与幅值评估；R3-m3/m4 修复成本低、显著提升可信度。

- **对照 Nature 式标准**：originality 中等偏上（新意在 evaluative framing 与证据纪律，非架构突破）；significance 中等（simulation-only，提炼为通用方法可显著提升）；interdisciplinary readership 潜力高兑现低（加权重点）；technical soundness 中等偏上且异常诚实（headline 为 exploratory、envelope 幅值弱、occupancy 操作化缺失）；readability 最弱（摘要过载、负性框架主导、记号无映射）。

- **建议姿态**：有条件支持，前提是解决可读性与跨学科传达（R3-M1/M2/M3）并正面处理技术关切（R3-M4/M5）与 minor 披露项（尤其 R3-m3/m4）。核心障碍不在科学诚实，而在把优点与一条通俗清晰的正面结论一起传达给更广读者。

---

## Cross-review synthesis（post-review，不反馈给评审人）

### Consensus strengths（共识优点）

1. **罕见的 claim moderation 与诚实**：三位评审独立一致地肯定如实报告 confirmatory family gate 与 clean-retention gate 未通过、sealed/exploratory 逐项标注、"simulation only、not uniform superiority" 的结论（R1、R2、R3）。
2. **证据分层与 artifact 溯源**：sealed confirmatory family 与 exploratory 分离、macro→SHA-256 溯源、figure 由 CSV 渲染，被三位评审独立视为可作范本的证据工程。
3. **数值自洽性良好**：三位评审均逐项核对 prose/Table I–IV/图注/解析宏，未发现实质数字矛盾（break-even 可由 β0/β1/β2 反解验证）。
4. **防选择性报告**：full-region SNR–SIR map、完整 risk–coverage 曲线主动避免只报有利角落（R1、R3）。
5. **修复链的逻辑设计**：probe→coverage→sidecar→capacity ladder 的干预序列，每个正负结果区分竞争解释（R1、R3）。

### Consensus blocking concerns（共识阻断性关切）

**无。** 三位评审均未将任何 Major 关切标为 Blocking Yes。原因是稿件自身已把中心结论刻意收窄（simulation-only operating envelope + repair path），且无有效性、伦理或数据完整性缺陷。但三位评审独立达成的共识是：**若干非阻断 Major 关切的叠加，使 headline "physics-guided operating envelope" 的显著性主张目前是 asserted 而非 sealed 证据支持**（见下）。

### Other consensus major concerns（其他共识性主要关切）

**S1 机制归因未建立（≥2 人：R1-M1、R2-M2，相关 R3-M4）**
sealed 证据只能确立"完整方法（A5）优于其自身 stripped backbone（A0）+4.57 pp"，且该对比被约 4.7 倍容量差与捆绑项（多任务、对比损失、残差 bypass）混淆；margin teacher（+0.16）与 tri-route（+0.44）两个被点名机制组件均未过零，family gate 未通过。因此"physics-guided 三路分配 + teacher"这一标题级机制主张未被确认性证据建立，最醒目的 severe-corner 胜利（+9.74/+10.46）也是 exploratory/post-hoc。**这是本轮大修的第一优先级。**

**S2 occupancy–SIR envelope 主张超出证据（3 人：R1-M3、R2-M3、R3-M5）**
occupancy 与 SIR 在设计中共变（作者自认非因果），occupancy 来自 simulator-tracked components、部署不可得且操作化定义缺失；外推验证缺符号 prevalence/chance 基线，招牌 severe 角落是 in-sample，幅值预测 RMSE（6.20/8.05 pp）与 headline 效应量同量级。"occupancy-indexed" 与 "可直接作 model-selection diagnostic" 的说法强于证据。**第二优先级。**

**S3 统计程序与参考值披露不足（R1-M2 为 Major；R3-m3/R3-m4 为 Minor 触及同一底层）**
bootstrap 方案（抽样次数、层级嵌套、seeds 处理）、design resolution 1.31 pp 推导、sign-flip 0.0018 的精确统计量均说明不足；severe-corner 聚合口径无法从 Fig. 2 重构（9.74 vs 简单平均 10.75）；capacity ladder 用 5 seeds 且 IQFormer/MCLDNN 参考值与主结果 10 seeds 不一致（48.72→49.61、45.99→45.41）且未披露。**第三优先级（多属低成本高收益的披露修复）。**

### Where emphasis differs across reviewers（评审侧重差异）

- **R2-M4（preregistered 方向反转）为 R2 独有但重要**：sealed directional test 否定了"gain 随 occupancy 非递增"的预注册期望，实际单调递增方向来自 exploratory。这不仅是机制未归因（S1）的另一种表现，更直接冲击 "physics-guided" 标签的物理可信度，需作者解释为何物理直觉被数据否定、新方向在机制上意味着什么。
- **R2-M5（compact 卖点被 sidecar 部分让渡）为 R2 独有**：clean-retention gate 未通过（6/10 类 100% transfer failure），靠 46794-param I/Q sidecar 修复，实质承认 pure spectral 表示对 clean 星座阶数判别不足；sidecar 后总参数未给出。
- **R3-M1/M2/M3（跨学科桥接与可读性）为 R3 独有且为其加权重点**：三位中只有 R3 把"未兑现的跨学科潜力"与"摘要负性框架过载/记号无映射"列为 Major。R1、R2 认可可读性中等偏上，未将其列为阻断；此分歧是侧重差异（R3 的 emphasis 为 readership/readability），非事实分歧。
- 三位在"严重度校准"上一致：**无任何 Major 达到 Blocking Yes**，但一致认为 headline 的 significance 需由 sealed 证据承担。

### Minor revision checklist（次要修订清单，已去重）

1. 统一 holdout 指标精度与格式：正文 r=0.694/sign 0.975 与 Fig. 4 注 r=0.69/sign 98% 不一致（R1-m2、R2-m2、R3-m6）。
2. "about three times"（4.57/1.31≈3.49）改为"约 3.5 倍"（R1-m3、R2-m1）。
3. 统一 baseline 命名：IQFormer-inspired vs IQFormer（含 Fig. 5/6/7 轴标签）、MCLDNN 是否加 inspired 后缀（R2-m6、R3-m1）。
4. 提供记号表与定义：VIMD-Net=A5、A0、A7、S/M/L 与 A5 关系、CSSL 全称出处、macro-F1 定义（R2-m4、R3-m2、R3-m5）。
5. 报告 sidecar 后组合总参数/MACs 并与 baseline 对比（R2-m5，关联 R2-M5）。
6. 对称报告 MCLDNN envelope 的 r=0.511/RMSE=8.05，避免只留有利指标（R2-m3）。
7. 给出 A7 vs A5 具体差值及区间（R1-m3）。
8. 补报告 latency（A5 3.172 vs MCLDNN 3.789 ms）或把 compact 主张限定为 parameters/MACs（R3-m8）。
9. 说明 severe-corner 的 SNR 条件与聚合方式（per-cell / source-paired），使 +9.74 pp 可从 Fig. 2 重构（R3-m3）。
10. 披露 capacity ladder 的 seed 数（5 vs 10）与参考模型值，统一 IQFormer/MCLDNN 参考（48.72/45.99 vs 49.61/45.41），说明 S(39500) 与 A5(39500) 差异（R1-m1、R3-m4）。
11. 补全 fusion 对照设置（架构、seed 数、ensemble 规模、校准）（R1-m4）。
12. 定义 probe 度量、范围与 0.25 阈值单位（R1-m5）。
13. 说明 Fig. 2 星号与 Holm 幸存计数（15 vs 12）关系（R1-m6）。
14. 提供完整 provenance 字段或说明公开层级（R1-m7）。
15. 一句话说明 CSSL 在 hard 上低于 A0 的原因（R1-m8）。
16. Section V.D 参数/MACs 句子标注单位并补齐 MCLDNN 值（R3-m7）。

### Broad-interest / significance readout（广泛兴趣 / 显著性判读）

稿件在 AMC / 频谱监测 / 资源受限 ML 交叉处有明确且可操作的贡献，其证据纪律（sealed/exploratory 分离、如实报告未通过 gate、SHA-256 溯源）对 benchmarking 与可复现 ML 社区有示范价值。但"跨学科广泛兴趣"目前是潜力而非已兑现：作者未把 operating-envelope 评估范式与 repair chain 提炼为可迁移方法，也未为 ML-for-science / CV / NLP 读者架桥。按 Nature 式标准，本文更接近"领域内优秀且有方法示范意义"的稿件，而非"outstanding interdisciplinary importance"；这一判断属评审意见，最终是否达到该门槛由编辑裁定。

### Most important issues to resolve before a strong Nature-style case is established（立论前最需解决的问题）

1. **机制归因（S1）**：提供容量匹配的 sealed 对比（A0 扩参 / 等容量捆绑消融），或把机制主张明确改述为 exploratory，并列出 A5 与 A0 的全部差异及其贡献上界。
2. **envelope 主张收窄（S2）**：报 prevalence + chance-corrected 指标，明确 severe corner 是 in-sample，补 occupancy 精确定义与部署代理，量化 occupancy–SIR 共变程度，区分 sign 与 magnitude 预测。
3. **统计/参考值披露（S3）**：补 bootstrap 细节、design resolution 推导、sign-flip 精确 p、severe-corner 聚合口径，统一并披露 capacity ladder 的 seed 与参考值。
4. **preregistered 反转的机制解释（R2-M4）**：解释为何物理直觉被 sealed 测试否定、新方向在机制上意味着什么。
5. **compact 主张的边界（R2-M5）**：给出 sidecar 组合参数，说明修复后是否仍 compact。
6. **跨学科桥接 + 可读性（R3-M1/M2/M3）**：为场外读者提炼可迁移方法，重写摘要/Evidence status box，建立记号表。

---

## Risk / unsupported claims（风险 / 未获支持主张）

- **未获支持的显著性主张**：标题与摘要把 advance 锚定在 "physics-guided" 机制上，但唯一 sealed 阳性（A5 vs A0 +4.57 pp）被容量/捆绑项混淆，机制组件（margin teacher、tri-route）未过零；severe-corner 优势为 exploratory/post-hoc。headline significance 目前非 sealed 证据支持。
- **超出现有证据的可操作性主张**："occupancy-indexed…model-selection diagnostic" 依赖 simulator-tracked、部署不可得的 occupancy，且未给出 deployment-available proxy；"直接可作诊断"说法强于证据。
- **外推主张被高估**："predicts their sign outside the regimes used to fit it" 的 0.975 sign agreement 缺 prevalence/chance 基线，且招牌 severe 角落是 in-sample；"unseen regimes" 指同模拟器内 held-out cells，非 out-of-distribution。
- **无法从所供材料核验**：artifact provenance 的 path/row selector/precision/SHA-256 字段未随稿提供；逐 seed 数据缺失；Fig. 2 星号与 Holm 计数（15 vs 12）、severe-corner 聚合（9.74 vs 10.75）、capacity ladder 参考值（48.72/45.99 vs 49.61/45.41）无法对账。

### 评审过程补充观察（编辑侧 QA，非任一评审报告）

以下为交叉综合阶段的补充核对（已由解析宏与作者内部文档印证），供作者在回复中一并处理：

- **拟合样本量未披露**：envelope 为 3 参数线性回归，holdout 81 cells；由总单元宏（113）可推得拟合约 32 cells，正文未明示拟合单元数与分布。32 单元拟合 3 参数模型是较小样本，应披露并讨论稳定性（尤其 bootstrap 下系数 CI 是否抗小样本）。
- **occupancy–SIR 共变程度未量化**：正文仅写 "co-vary"，未给相关系数（作者内部口径为 rho≈−0.8，属高度共线）；高共线下两变量系数 CI 的解释需谨慎，应披露并讨论。
- **"receiver stress" split 有声明无结果**：Evidence Protocol 列出 receiver stress 作为一个 evaluation split，但全文无任何对应结果报告，应说明去处或补上。
- **quintile 单调性具体数值未入正文**：A5–CSSL 随 occupancy 五分位单调上升（−7.6→+23.3 pp，ρ=+1.0）的具体值未写入正文，仅定性描述；属可低成本补齐的机制支撑证据。

---

## 第一版大修意见（编辑/作者面向结论）

**总体结论：Major Revision（大修）。** 三位评审一致认为稿件在科学诚实、证据分层、artifact 溯源与数值自洽上"罕见且可作范本"，且无任何单一缺陷达到阻断（Blocking Yes）级别；但一致认为 headline 的显著性主张（physics-guided 机制的 operating envelope）目前由 exploratory/post-hoc 证据承担、而非 sealed 证据承担，原创性定位（evaluative framing）也未与既有工作充分划界，故在机制归因、envelope 证据强度、统计可复核性与跨学科传达四个方向需实质性修订后方可建立完整立论。

**必改（对应共识 Major，决定能否立论）**

1. 机制归因：提供容量匹配的 sealed 对比，或将机制主张改述为 exploratory，并披露 A5 与 A0 间全部差异及其贡献上界。
2. envelope 证据：补符号 prevalence 与 chance-corrected 指标；明确 severe corner 为 in-sample；补 occupancy 精确定义、部署代理与共变程度量化；区分 sign/magnitude 预测质量。
3. 统计可复核：补 bootstrap 方案、design resolution 1.31 pp 推导、sign-flip 精确 p、severe-corner 聚合口径；统一并披露 capacity ladder 的 seed 与参考模型值（消除 IQFormer 48.72 vs 49.61、MCLDNN 45.99 vs 45.41 的不一致）。
4. 机制方向反转的解释：解释 preregistered "非递增" 期望被 sealed 测试否定后，递增方向在物理/机制上意味着什么，或明确 "physics-guided" 仅指 teacher 构造来源。
5. 跨学科桥接与可读性：在 Discussion 提炼可迁移方法（envelope 评估范式 + 证据纪律 + repair chain）；重写摘要与 Evidence status box（正面结论前置、协议术语后置）；建立记号表（VIMD-Net=A5、A0/A7、S/M/L、CSSL、macro-F1）。

**应改（Minor checklist 16 项，成本低、显著提升可信度，见上）**：重点是统一精度/命名、补 latency 与 sidecar 组合参数、披露 capacity ladder 与 severe-corner 口径、补 provenance 字段。

**本轮不需重做实验即可关闭的项**：S3 全部、Minor 全部、以及 S2/S1 的"措辞收窄"部分（把机制与 envelope 主张降级为 exploratory 表述，使其与证据等级一致）。**需要新增 sealed/confirmatory 实验才能关闭的项**：S1 的机制分离对比、S2 的 deployment-available occupancy proxy 验证。
