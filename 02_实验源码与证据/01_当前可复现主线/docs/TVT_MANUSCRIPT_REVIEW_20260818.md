# 稿件全面评审：`tvt_operating_envelope_integration.pdf`

> 评审日期：2026-08-18 ｜ 稿件：7 页正文＋21 条参考文献 ｜ 目标期刊：IEEE TVT  
> 评审方式：逐数字核对工作区证据（sealed composite、`analysis_zero_compute/outputs/`、`artifacts/tier2_*`），并按 TVT 审稿人视角做攻击面测试。

## 0. 总体判定

**证据强度：A−**（同类 AMC 稿件中属前 5%）。**当前形态的接收概率：25–35%**，最可能结局是 Major Revision 后进入边缘区。**按 §4 修改后：55–70%。**

判定依据：证据工程、统计严谨性、诚实性都远高于 AMC 领域平均水平；扣分全部来自**定位与呈现**，以及一项外部有效性缺口（无公开数据集）。三个核心问题里有两个是自伤，可在不新增仿真的前提下修好。

数字核对结论：**抽查的 27 个关键数字全部与工作区产物一致**，未发现夸大或选择性报告。Table I、Table III、Table IV、Fig. 2–7 与 CSV 逐项吻合；预注册失败门在摘要、§IV、§V、§VII 四处披露。这一条是本稿最大的资产——不要在修改中损失它。

---

## 1. 真正强的地方（保住）

1. **严重干扰角结果经得起打**：64 格中 12 格过 Holm，全在 SIR = −15 dB 行；10/10 seed 同号；符号翻转 p = 0.0018（10 seed 下限）；4 种 bootstrap 变体一致。这是全稿最硬的实证。
2. **干预链的科学结构**：探针（诊断）→ 覆盖臂（否定廉价解释）→ sidecar（修复）→ 容量阶梯（排除宽度解释）→ 门控臂（阴性对照）。四臂中三臂产出**否定或阴性**结果，却共同收敛到同一结论。这是审稿人会明确称赞的设计。
3. **有界结论的表述**：把两项未过的消融写成"个体贡献 < 1.36 pp、设计分辨力 1.31 pp"，而不是"不确定"。技术上正确，且比模糊表述更有信息量。
4. **可复核性**：110 个宏逐一带 source path / row selector / SHA-256，图只渲染 CSV，另有复现包 zip。TVT 对可复现性有正面偏好。
5. **诚实披露没有削弱说服力**——摘要里写明两个门未过，反而让 4.57 pp 的 sealed 主效应可信。这一点做对了。

---

## 2. 会被击中的五个点（按严重度）

### P1 🔴 摘要头条指标近乎空洞：sign agreement 0.975

**问题**：81 个留出单元中 **96.3% 的真实增益本来就是负的**。恒定预测"永远为负"可得 0.963，包络律得 0.975——**净提升仅 1.2 个百分点**。摘要与 §V-C 把 0.975 当作外推能力的主证据，一个细心的审稿人 10 分钟内就能算出这一点，届时不只是这条主张失效，全稿的统计自律形象会一并受损。

**核对**（`a14_envelope_holdout.csv`）：

| 指标 | vs IQFormer | vs MCLDNN |
|---|---:|---:|
| 符号一致率 | 0.975 | 0.951 |
| 恒定"永远为负"基线 | 0.963 | 0.901 |
| **净提升** | **+1.2 pp** | **+5.0 pp** |

**修法（数据已在手，不需新计算）**：换成有技巧分数含义的指标。

| 指标 | vs IQFormer | vs MCLDNN |
|---|---:|---:|
| 留出 RMSE | 6.20 pp | 8.05 pp |
| 常数基线（拟合集均值）RMSE | 8.40 pp | 9.63 pp |
| **留出技巧分数 R²_skill** | **0.456** | **0.301** |
| 全样本 R²（仅占用率 / 仅 SIR / 两者） | 0.202 / 0.391 / 0.580 | 0.259 / 0.201 / 0.449 |

"相对常数基线降低 26% 的留出 RMSE、技巧分数 0.46，且两个物理变量各自贡献不可替代的方差（0.20 与 0.39 → 合并 0.58）"——这句话比 0.975 弱看，但**真实且不可攻破**。建议摘要改用 R²_skill 与 RMSE，符号一致率降为正文中的次要说明并**同时给出恒定基线值**。

### P2 🔴 全稿最强结果被埋在 §V-E，且参数表述含糊

**核对**（`artifacts/tier2_iq_sidecar_v1/models/*/result.json`）：`tier2_h2_f2_iq_sidecar` 的**总**参数量是 **46,794**，不是"A5 的 39,500 再加 46,794"。MACs 43.1 M vs A5 的 41.8 M。也就是说：

> **+7,294 个参数（+18%）、+3% MACs，换来 clean +11.77 pp、hard +4.37 pp，并在 hard interference 上与 354,984 参数的 IQFormer 统计持平（Δ = −0.29 pp，[−1.95, +1.46]），在 SIR = −15 dB 仍领先 12.09 pp。**

这是整个项目最有冲击力的一句话，目前却写成"adds a 46794-parameter received-I/Q sidecar"（读者会理解为模型变大了一倍多），并放在第 5 页倒数第二段。

**修法**：
- 明确写"总参数 46,794（相对 A5 增加 7,294，+18%）"，并把 7.6× 参数比与统计持平放进摘要；
- Fig. 5 的 Pareto 图加入 sidecar 点——它会落在 IQFormer 左下方最有说服力的位置，目前图里完全没有它；
- 标题/贡献列表把"repair path"提到与"envelope"并列，而不是附属。

**保留**：exploratory 标注必须保留。但"exploratory"不等于"次要"，可以写成 "prospective Tier-2 experiment, outside the sealed confirmatory family"，并在讨论中说明它是下一轮确认性campaign 的预注册候选。

### P3 🔴 全合成，无任何公开数据集验证

TVT 的 AMC 审稿人几乎必问 RadioML 2016.10a/2018.01a 或同类公开集。本稿引用了 O'Shea [1] 却未在其数据上评估。当前所有外部有效性都押在"3GPP TR 38.901 TDL profile + 自建 jammer 分类学"上。

**风险**：这是最可能直接触发 Reject 的单点，尤其遇到偏经验的审稿人。

**修法（需算力，不需新仿真设计）**：在 RML2018.01a 上跑 A5 / sidecar / MCLDNN / IQFormer 四个模型，只报**相对排序**与包络律在该集上的可迁移性（RML 无干扰占用率标注，可用带内能量占比作代理）。哪怕只作为一节 "external validity check" 且结果为负（律不迁移），也远好于没有。

### P4 🟡 方法新颖性偏弱 + 篇幅偏薄

本稿本质是**评价型/刻画型**论文：所提方法在多数条件下不占优，主张是"边界可预测 + 可修复"。TVT 接受这类论文，但比例低于方法型。7 页正文、21 条参考文献对 TVT 常规论文（通常 9–13 页、35–50 条参考）偏薄，容易被读成 correspondence。

**修法**：
- 参考文献扩到 35+，补 2024–2026 的抗干扰 AMC、轻量化 AMC、operating-envelope/failure-characterization 类工作；
- 把 sidecar 提为共同贡献后，正文自然扩到 9–10 页；
- Related Work 里明确写清"本文与既有工作的差别是给出可外推的边界律与修复链，而非新架构"——主动定位比让审稿人替你定位安全。

### P5 🟡 两处可被逐字核对出来的表述问题

1. **"Because occupancy and SIR co-vary in this design"（§V-C 末）与数据不符**：在参与拟合的 113 个单元上，两者相关系数是 **−0.025**（`a14_envelope_cells.csv`），几乎正交；共变只发生在 hard_interference 内部的占用率分位对比中（ρ ≈ −0.8）。这句本意是自我克制，但写成了一个可被证伪的陈述。**改**为："within hard-interference occupancy quintiles the two covary (ρ ≈ −0.8), while across the pooled cell set they are close to orthogonal (r = −0.03); the law is therefore descriptive rather than causal because occupancy is simulator-derived, not because of collinearity."
2. **占用率在部署时不可观测**：§VI 已承认需要 "deployment-available occupancy proxies"，但摘要与 Table II 把 break-even 表说成 "directly usable as a model-selection diagnostic"。两处口径不一致。**改**：Table II 标题加 "given oracle occupancy"，并在讨论中明确"实用化需要一个占用率估计器，本文未提供"。

---

## 3. 中小问题清单

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | Fig. 2 | 星号只标"下界为正"，未说明是否过多重性校正；正文说 12/64 过 Holm，图上星号数看起来更多 | 图注补一句"star = uncorrected positive lower bound; Holm-surviving cells are listed in §V-B" |
| 2 | Fig. 5 | 缺 sidecar 与 M/L 容量档的点 | 补点，Pareto 图才完整 |
| 3 | Table III | 三行干预缺少参数量列 | 加一列，读者才能看出 sidecar 的性价比 |
| 4 | §V-D | 融合结论用的是 A5，未说明 sidecar 是否仍与 IQFormer 互补 | 用现有预测重算一次融合（零算力，几分钟） |
| 5 | §I | 贡献列表 5 条中有 3 条是"证据/审计"性质 | 压缩为 4 条，把 repair chain 提前 |
| 6 | 全文 | 无算法伪代码/复杂度推导 | 加一小段 sidecar 结构说明，方便复现 |
| 7 | 参考文献 | 无 RadioML 数据集论文的正式引用位（只在 [1]） | 补数据集与 benchmark 类引用 |

---

## 4. 提升接收率的行动清单（按性价比）

| 优先 | 行动 | 成本 | 预计接收率增益 |
|---|---|---|---|
| 1 | 摘要与 §V-C 换掉 sign agreement，改用 R²_skill = 0.456 / RMSE 6.20 vs 8.40，并给出恒定基线 | 1 小时，数据已在 | +8~12 pp（同时消除一个致命攻击点） |
| 2 | sidecar 提为并列贡献：46,794 总参数、7.6× 参数比下与 IQFormer 持平，进摘要、进 Fig. 5、进标题贡献 | 半天 | +10~15 pp |
| 3 | 补 RML2018.01a 外部有效性一节（哪怕结果为负） | GPU 1–2 天 | +10~20 pp |
| 4 | 参考文献扩到 35+，Related Work 明确定位 | 1 天 | +5 pp |
| 5 | 修 P5 两处表述、补中小问题 1–6 | 半天 | +3~5 pp |

做完 1、2、5 → 约 40–50%；再加 3、4 → **55–70%**。

---

## 5. 期刊选择

| 期刊 | 匹配度 | 说明 |
|---|---|---|
| **IEEE TVT** | 中高 | 车载多普勒＋算力受限定位契合；但方法非 SOTA、纯仿真是主要风险。建议完成行动 1–3 后投 |
| IEEE Trans. Cognitive Comm. & Networking | 高 | 更欢迎"刻画与边界"型研究；但你已有 TCCN 支线，需避免自我重复 |
| IEEE Wireless Communications Letters | 高（作为退路） | 把包络律＋severe corner 压成 5 页 letter，接收概率显著更高（约 45–55%），代价是 sidecar 与条件迁移分析要删 |
| IEEE Open Journal of Communications Society | 中高 | 对刻画型与负结果更友好，OA 费用需考虑 |

**建议**：按行动 1–5 修完投 TVT；若一审被拒且理由集中在"无公开数据集"，转投 TCCN 或压成 WCL letter，不要在 TVT 反复申诉。

---

## 6. 一句话给作者

这份稿件的证据质量已经超过它目前的自我呈现：**最强的结果（+18% 参数换来与 7.6 倍大模型持平）被埋着，而最弱的指标（sign agreement）被放在摘要第一句**。把这两件事对调，接收概率的提升幅度会大于任何新实验。
