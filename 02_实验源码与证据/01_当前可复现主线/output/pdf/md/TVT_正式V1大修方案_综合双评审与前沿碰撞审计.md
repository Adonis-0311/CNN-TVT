# TVT 正式 V1 大修方案
## 综合双评审意见 + 第一版大修方案 + 2026-08 前沿碰撞审计

**稿件：** *An Occupancy-Indexed Operating Envelope for Compact Physics-Guided Automatic Modulation Classification under Structured Interference*  
**目标期刊：** IEEE Transactions on Vehicular Technology (TVT)  
**版本：** Formal Revision Plan V1.0  
**日期：** 2026-08-18  
**性质：** 投稿前结构性 Major Revision  
**总目标：** 在不牺牲证据诚实性、不把 exploratory 结果伪装成 confirmatory 结果的前提下，把稿件从“紧凑 physics-guided AMC 模型 + 若干复杂实验”重新定位为一篇 **condition-indexed model-ranking / operating-envelope + representation diagnosis and repair** 的 TVT 论文。

---

# 0. V1 的最终判断

本轮综合两份独立评审、原稿以及第一版修改方案后，V1 不应继续围绕：

> “VIMD-Net 是一个新的 physics-guided compact AMC architecture，并在结构化干扰下比大模型更强”

来组织。

这一表述有三重风险：

1. **机制证据不足**：sealed family 只确认 A5 相对 A0 的 whole-configuration contrast 为正；margin teacher 与 tri-route 两个被点名组件均未单独过零，且 A5/A0 还存在容量和捆绑项差异。
2. **前沿碰撞较高**：2024–2026 已有大量 interference-tolerant、knowledge/physics-guided、multimodal、lightweight、cross-domain AMC 工作；把“physics-guided / compact / I/Q+spectral fusion”作为一级新颖性容易被审稿人判为组件重组。
3. **最强科学贡献其实不是架构本身**：现有证据最有辨识度的是“模型排序随物理条件发生 rank exchange → 经验 envelope 描述边界 → clean failure 被 probe/coverage/sidecar/capacity chain 定位并修复”。

因此，**正式 V1 的核心论文命题应调整为：**

> **在结构化干扰与移动信道条件下，紧凑谱表示与高容量 I/Q 表示不存在统一排序；其相对优势呈现可复现的条件依赖边界。本文用受控、source-disjoint 的证据协议刻画这一边界，并通过前瞻性 representation intervention 诊断和修复 clean-condition failure。**

这条主线既比“新网络刷分”更有辨识度，也更能承受现有失败 gate、post-hoc severe corner 和 simulator-only 的事实。

---

# 1. 三份意见的综合结论

## 1.1 三方一致认可的资产——必须保护

### A. 严重干扰角落不是单点挑结果

现有证据包括：

- 64 个 inspected cells；
- 12 个 Holm-surviving cells，全部位于 SIR = −15 dB；
- 10/10 seeds 同方向；
- sign-flip probability = 0.0018；
- 完整 SNR–SIR map 展示正负区域。

这组证据的价值不在“证明全局 superiority”，而在于：

> **证明 rank exchange 是结构性的，而不是由一个偶然 seed 或单个 cell 造成。**

### B. probe → coverage → sidecar → capacity ladder 是全文最有科学性的链条

这一链条同时包含正结果和阴性结果：

1. probe：定位可用信息不足；
2. coverage arm：增加 clean exposure 不能修复；
3. presence-gated negative control：不能修复；
4. received-I/Q sidecar：显著修复；
5. spectral capacity ladder：加宽能提高平均性能，但不能消除边界。

V1 中应把这条链提到与 operating envelope **并列核心贡献**，而非放在稿件末端作为补充分析。

### C. 证据等级纪律是稿件的独特资产

应继续清楚区分：

- original sealed confirmatory family；
- post-hoc severe-stratification；
- exploratory envelope；
- prospective Tier-2 diagnosis/repair；
- future prospective confirmation。

任何新实验都不能 retroactively 改写 original sealed family。

---

# 2. 前沿碰撞审计：2024–2026 相关方向已经拥挤到什么程度

> 本节的目的不是证明“无人做过”，而是建立 novelty firewall：哪些关键词已经拥挤，哪些组合式贡献仍有明显空间。  
> 检索截至 2026-08-18。公开预印本与正式同行评审论文分开对待。

## 2.1 结构化/恶意干扰 AMC：碰撞风险高

### Sun et al., IEEE TVT, 2025
**Toward Interference-Tolerant Automatic Modulation Recognition via Multi-stage Feature Extraction Network**

已直接把“interference-tolerant AMR”作为 TVT 论文核心问题，并使用频域预处理、多阶段/多尺度特征提取提升低信噪环境下鲁棒性。

**碰撞风险：高**

因此本文不能再把：

> “在干扰条件下提高 AMC 鲁棒性”

当作主要创新。

**必须区别为：**

> 本文研究的是 structured jammer condition 下 **model ranking 本身如何随物理条件发生交换**，而不是设计另一个 universally interference-robust classifier。

---

### Zhang et al., IEEE TVT, 2026
**Modulation Recognition via Tensor Analysis for MIMO Systems with Malicious Interference**

该工作已经覆盖：

- malicious interference；
- 低 SIR；
- tensor/CP decomposition；
- interference separation；
- modulation recognition。

**碰撞风险：高**

尤其不能用：

> “首次解决 malicious/structured interference 下 AMC”

这类表述。

**本文仍可保留的差异：**

- 非 MIMO tensor separation；
- 不以 source reconstruction 为目标；
- 研究 compact spectral vs high-capacity I/Q 的 rank exchange；
- 给出物理条件 plane 上的 operating boundary；
- 用后续 intervention 诊断和修复 failure boundary。

---

## 2.2 Cross-domain / domain adaptation / contrastive AMC：碰撞风险高

近年已有：

- SigDA，IEEE TWC 2024；
- MCLHN，IEEE TWC 2024；
- Contrastive Learner for AMC，IEEE TWC 2025；
- Confidence-Guided Prototypical Contrastive Domain Adaptation，IEEE TCCN 2026。

这些工作已经覆盖：

- domain mismatch；
- masked/contrastive representation；
- hard negatives；
- prototypical adaptation；
- channel-domain generalization。

**碰撞风险：高**

因此：

- exact-source contrastive loss 不能作为一级创新；
- held-channel / cross-domain robustness 也不宜单独包装成 novelty。

这些只能作为 VIMD whole configuration 的组成部分。

---

## 2.3 Multi-domain / I/Q + spectral / physics-guided representation：碰撞风险很高

### IQFormer, IEEE TCCN, 2025
已经是 multi-modality fusion AMC 的代表性强模型。

### RIS-MAE, 2025 preprint
强调 raw I/Q representation 相比 time-frequency transforms 保留 amplitude/phase/frequency-offset information。

### DKDNet, 2026-07 preprint
**Dual Knowledge and Data-Driven Network for Cross-Domain AMC**

非常接近“knowledge/physics prior + multi-representation + lightweight/dynamic fusion + cross-domain”叙事。

### ZoomSpec, 2026 preprint
使用 physics-guided coarse-to-fine sensing，并融合 time-domain I/Q 与 spectral representation。

### MoEformer, 2026 preprint
使用 dynamic experts / adaptive fusion。

**碰撞风险：很高**

因此 V1 必须避免把下列内容单独包装成强 novelty：

- physics-guided；
- knowledge-guided；
- multi-route；
- adaptive gating；
- I/Q + spectral fusion；
- compact multi-domain representation。

尤其 sidecar **不能写成“我们提出一种新的 I/Q-spectral fusion architecture”**。

sidecar 最安全、也是最强的定位是：

> **a prospective representation intervention designed after a frozen failure diagnosis**

即：

> 它不是“新融合网络”，而是一个用来检验 representation-gap hypothesis 的前瞻性修复实验。

---

## 2.4 Lightweight / efficient AMC：碰撞风险很高

2025–2026 已有大量轻量化 AMC，包括：

- LightAMC 既有基础；
- Threshold-Denoising RNN / efficient AMC；
- 约 27k parameter 的 lightweight networks；
- pruning / distillation / compact architectures。

所以：

> “39.5k 参数”

本身已经不能承担 novelty。

它只能承担一个 engineering axis：

> **在某些 severe/high-occupancy conditions 下，紧凑表示的 performance–complexity tradeoff 发生反转。**

即 compactness 是 operating-envelope 的一个维度，不是论文的核心创新标签。

---

# 3. 碰撞风险矩阵

| 主张/组件 | 2026 前沿碰撞风险 | V1 处理 |
|---|---:|---|
| interference-tolerant AMC | 高 | 不作为一级创新 |
| malicious/structured interference AMC | 高 | 强调 rank exchange / condition plane |
| physics-guided AMC | 高 | 降级为方法属性，不作为已证实机制 |
| multi-route / soft masking | 高 | 作为 architecture implementation |
| contrastive condition invariance | 高 | 作为辅助训练目标 |
| lightweight/compact AMC | 高 | 只作为 tradeoff axis |
| I/Q + spectral fusion | 高 | sidecar 定位为 diagnosis-driven intervention |
| cross-domain/channel robustness | 高 | supporting evidence |
| occupancy-based interference characterization | 中 | 明确定义且不声称 deployment-ready |
| condition-indexed model rank exchange | 低–中 | **一级核心创新** |
| occupancy–SIR operating boundary for model comparison | 低 | **一级核心创新，但必须收窄主张** |
| full-region anti-cherry-picking map + evidence hierarchy | 低 | **方法学贡献** |
| failure localization → negative control → repair → capacity exclusion | 低 | **一级核心贡献** |
| prospective repair after failed clean gate | 低 | **一级核心贡献** |

### 结论

截至本轮检索，没有发现与本文**完整同构**的：

> **structured-interference model-rank operating envelope + sealed/exploratory evidence separation + prospective representation-repair chain**

公开工作。

但这不是“绝对无人做过”的证明。正式稿应写：

> “To our knowledge, prior AMC studies have largely optimized or compared marginal performance rather than explicitly modeling the condition-dependent rank exchange between compact spectral and high-capacity I/Q representations and then prospectively intervening on the revealed boundary.”

不要写：

> “This is the first operating envelope for AMC.”

除非后续做形式化 systematic search 并能承担该绝对表述。

---

# 4. V1 的 novelty firewall

## 4.1 不再主张

以下内容从一级贡献中删除：

- “three-route allocation itself is proven to cause the gain”
- “margin teacher is confirmed to outperform proportional teacher”
- “physics-guided mechanism is confirmed”
- “compact VIMD is uniformly superior”
- “occupancy–SIR law is causal”
- “occupancy–SIR boundary is directly deployable”
- “0.975 sign agreement proves strong out-of-regime generalization”
- “I/Q sidecar is a novel multimodal fusion architecture”
- “39.5k parameters is a unique lightweight contribution”

---

## 4.2 正式主张

V1 应只拥有以下四项 headline：

### Contribution 1 — Condition-indexed rank exchange

> 在 structured-interference physical plane 上，compact spectral 与 high-capacity I/Q models 的排序发生可复现交换；完整区域图避免把 severe corner 误读成 uniform superiority。

### Contribution 2 — Empirical operating envelope

> 使用 occupancy + SIR 构建低维 empirical envelope，对同一 frozen simulator campaign 中未参与拟合的 regime cells 具有有意义的 magnitude skill，相比 constant baseline 降低预测误差。

注意：

- 不再以 0.975 sign agreement 作为首要指标；
- 强调 skill relative to a trivial baseline；
- “held regime” 不等于真实-world OOD；
- severe SIR corner 是 fit-side hard-interference region，不能被当作 holdout validation。

### Contribution 3 — Evidence-bounded severe-interference finding

> severe SIR corner 具有多 seed、paired intervals、Holm correction、full-map support，但明确属于 post-hoc / exploratory statistical reinforcement。

### Contribution 4 — Representation diagnosis and repair

> clean gate failure 经 probe、coverage arm、negative control、I/Q sidecar、capacity ladder 形成解释链；sidecar 作为 prospective Tier-2 intervention，以很小的额外复杂度同时修复 clean 和 hard performance。

---

# 5. 标题：正式 V1 建议更换

当前：

> **An Occupancy-Indexed Operating Envelope for Compact Physics-Guided Automatic Modulation Classification under Structured Interference**

当前标题存在两个问题：

1. **“Physics-Guided”过度前置**：现有 sealed evidence 无法确认具体 teacher/tri-route mechanism；
2. 前沿已有大量 knowledge/physics-guided、多域 AMC，碰撞较高。

## 推荐标题 A —— 正式首选

> **A Condition-Indexed Operating Envelope and Representation-Repair Study for Automatic Modulation Classification under Structured Interference**

优点：

- operating envelope 保留；
- repair 提升为并列贡献；
- 避开未经确认的 physics-guided mechanism headline；
- 与最新 knowledge-guided/multimodal architectures 拉开距离。

## 备选标题 B

> **Operating Envelopes and Representation Repair for Automatic Modulation Classification under Structured Interference**

更简洁，冲击力较强。

## 备选标题 C

> **Condition-Dependent Rank Exchange between Compact Spectral and I/Q Modulation Classifiers under Structured Interference**

最直接，但过于像 comparison paper，可作为保守版本。

### V1 决策

**优先标题 A。**

“physics-guided”保留在正文 Method 和 Abstract 中作为 VIMD-Net 的属性，而不是整篇论文的已确认机制结论。

---

# 6. P0-1：立即替换摘要中的 0.975 sign headline

这是两版评审中新增的最严重攻击面之一。

## 6.1 当前问题

81 个 holdout cells 中，真实增益符号高度不平衡。

针对 IQFormer-inspired：

- envelope sign agreement = 0.975；
- “永远预测为负”的 trivial baseline 已经约为 0.963；
- 净增益只有约 1.2 pp。

如果摘要只报 0.975，而不报 prevalence baseline，会让审稿人认为指标选择性偏乐观。

---

## 6.2 V1 的正式替代指标

针对 IQFormer-inspired：

- holdout RMSE = **6.20 pp**；
- constant-baseline RMSE = **8.40 pp**；
- \(R^2_{\text{skill}}\approx 0.456\)。

针对 MCLDNN：

- holdout RMSE = **8.05 pp**；
- constant-baseline RMSE = **9.63 pp**；
- \(R^2_{\text{skill}}\approx 0.301\)。

建议摘要使用：

> “On held regime cells from the same frozen simulator campaign, the two-variable envelope reduces RMSE from 8.40 to 6.20 percentage points relative to a constant baseline, corresponding to an \(R^2_{\rm skill}\) of 0.46.”

然后正文再补：

> sign agreement = 79/81 (97.5%), but the majority-sign baseline is 78/81 (96.3%), so sign accuracy alone is not used as the principal validation metric.

### 必须增加

- sign prevalence；
- majority-sign baseline；
- balanced sign accuracy 或 Cohen’s \(\kappa\)（若现有 cell-level artifact 可直接计算）；
- magnitude metric 与 sign metric 分开讨论。

---

# 7. P0-2：sidecar 参数表述必须纠正，并提升为并列贡献

## 7.1 当前文字存在事实性歧义

原稿：

> “adding a 46794-parameter received-I/Q sidecar”

容易让人理解成：

> 在 39,500 参数 A5 之外再额外增加 46,794 参数。

实际评审核对结果：

- A5 total = **39,500 params**；
- sidecar model total = **46,794 params**；
- 实际增加 = **7,294 params**；
- parameter increase ≈ **18%**；
- A5 MACs ≈ **41.8 M**；
- sidecar MACs ≈ **43.1 M**；
- MAC increase ≈ **3%**。

---

## 7.2 这是全文最值得上摘要的结果之一

sidecar：

- clean +11.77 pp；
- hard +4.37 pp；
- overall hard vs IQFormer-inspired：
  \[
  \Delta=-0.29\ \text{pp},\quad [-1.95,1.46]
  \]
- severe SIR 下仍领先约 12.09 pp；
- total parameters 46,794 vs IQFormer-inspired 354,984，约 **7.6× smaller by parameter count**。

### V1 要求

摘要必须明确：

> “A prospective received-I/Q sidecar raises the total model size only from 39.5k to 46.8k parameters (+18%; about +3% reported MACs), while improving clean macro-F1 by 11.77 pp and hard-interference macro-F1 by 4.37 pp.”

如果篇幅允许，再写：

> “Its overall hard score is statistically near the 355k-parameter IQFormer-inspired reference.”

### 但必须保留

> “This repair is exploratory and outside the original sealed confirmatory family.”

---

# 8. P0-3：重新界定 “physics-guided”

## 8.1 当前 sealed evidence 能支持什么

可以支持：

> A5 whole configuration relative to A0 has a positive sealed contrast of +4.57 pp with simultaneous interval [3.65, 5.49].

不能支持：

> margin teacher 本身带来显著收益。

不能支持：

> tri-route 本身带来显著收益。

更不能支持：

> severe-interference advantage 是由被预注册 physics mechanism 直接造成。

---

## 8.2 V1 的标准措辞

将：

> “confirmed physics-guided mechanism”

统一改为：

> **“physics-informed training construction”**

或：

> **“simulation-component-guided representation allocation”**

并明确：

> “The original sealed family resolves a positive whole-configuration contrast, but does not isolate the contribution of the margin teacher or the tri-route allocation individually.”

### Section V-A 标题

从：

> Confirmed Whole-Method Effect, Bounded Component Effects

改为：

> **Positive Sealed Whole-Configuration Contrast and Bounded Component Effects**

“configuration”优于“method effect”，因为 A5/A0 还存在容量与捆绑项差异。

---

# 9. P0-4：A5 vs A0 容量混淆必须正面处理

两版评审都指出：

- A0 ≈ 8,458 params；
- A5 = 39,500 params；
- A5 相比 A0 约 4.7× 参数；
- 同时还有多任务、contrastive、bypass 等捆绑差异。

因此 +4.57 pp 不能严格解释为“physics-guided allocation effect”。

## V1 最低成本路线（必做）

在 Method / Evidence Protocol 中增加：

### “What differs between A0 and A5?”

用表格列：

| Difference | A0 | A5 |
|---|---|---|
| parameter count | 8,458 | 39,500 |
| three-route allocation | No | Yes |
| simulation-component teacher | No/Reference | Yes |
| multitask objectives | ... | ... |
| exact-source contrastive | ... | ... |
| residual bypass | ... | ... |

并在 Results 明确：

> “The A5–A0 contrast is interpreted as a whole-configuration effect, not as an isolated causal effect of the teacher or route count.”

---

## 可选高价值路线

如果愿意增加一次前瞻实验：

> 新建 **capacity-matched direct-spectral control**（约 39.5k params），尽量控制非机制项。

但必须注明：

> 这是查看原结果后设计的新 prospective experiment，不能纳入 original sealed family。

如果做，应称：

> **prospective confirmatory extension**

而不是“原 sealed 结果”。

---

# 10. P0-5：preregistered mechanism direction 被否定，必须单独解释

原预注册期望：

> gain 随 occupancy 增大应 non-increasing。

sealed directional test：

> 拒绝该方向。

探索结果：

> gain 反而随 occupancy 增加。

这不能被写成：

> “我们的 physics intuition 最终被验证，只是方向不同。”

---

## 正式 Discussion 段落应表达

> The preregistered directional mechanism hypothesis was not supported. Specifically, the data reject the expectation that the compact model’s relative gain should decrease with increasing occupancy. The surviving result is therefore empirical rather than mechanistic: occupancy organizes the observed rank exchange, but the observed direction should not be interpreted as confirmation of the originally hypothesized masking mechanism. In this paper, “physics-guided” refers to the construction of the training supervision, not to a confirmed monotonic physical law.

这是 V1 必须保留的诚实边界。

---

# 11. P0-6：occupancy–SIR “共变”陈述必须重新核对，不能照抄任何一版意见

两份评审之间出现了一个需要回到 artifact 的不一致：

- 一版核对称 pooled cells 中 occupancy–SIR correlation 约为 **−0.025**，接近正交；
- 另一版综合意见提到 hard-interference occupancy quintiles 中约 **ρ≈−0.8**；
- 同时关于 fit cell count 是 32 还是 113，也存在不同口径。

因此 V1 **禁止直接把任意一个值写进主稿，直到重新对 artifact 对账。**

---

## 必做 Artifact Audit

对 `a14_envelope_cells.csv` 或实际 envelope 源文件统一输出：

1. \(N_{\rm fit}\)；
2. \(N_{\rm holdout}=81\)；
3. fit set 中 pooled Pearson \(r(o,\mathrm{SIR})\)；
4. hard-interference subset 中 Pearson / Spearman correlation；
5. occupancy quintile aggregation 中 correlation；
6. 每个 regime 的 cell 数；
7. 哪些 cells 用于 fit，哪些用于 evaluation。

### 预期写法

若最终核对确实是：

- pooled \(r\approx -0.03\)；
- hard subset \(\rho\approx -0.8\)；

则正文写：

> “Occupancy and SIR are nearly orthogonal across the pooled fitting cells but become strongly associated within the hard-interference stratification. Eq. (7) is nevertheless treated as descriptive rather than causal because occupancy is simulator-derived and the design is observational with respect to this covariate.”

不要再简单写：

> “Because occupancy and SIR co-vary in this design…”

---

# 12. P0-7：occupancy 必须给出精确定义

目前标题把 occupancy 放在核心，但操作定义不足。

V1 必须明确：

- occupancy 在哪个时频 lattice 上计算；
- 使用 target/jammer 哪个 tracked component；
- power threshold 如何定义；
- 是否 normalize；
- 是否按 time-frequency cells 的比例；
- threshold 与 noise floor 的关系；
- 一个 window 的 occupancy 如何 aggregation；
- 一个 evaluation cell 的 occupancy 如何 aggregation；
- 是否由 ground-truth simulator component 得到。

最好给公式，例如：

\[
o = \frac{1}{FT}\sum_{f,t}
\mathbf{1}\left(P_j(f,t)>\tau\right),
\]

若实际实现不是这个形式，则按真实 artifact 写，不能臆造。

---

# 13. P0-8：部署主张收窄

当前 occupancy 是 simulator-tracked quantity。

所以 Table II 的 break-even occupancy 不是 deployment-ready control threshold。

### 禁用

> directly usable model-selection diagnostic

### 改为

> **“a simulator-domain reference boundary conditional on an occupancy estimate”**

或：

> **“a physically interpretable reference for future receiver-side model selection, provided that a deployment-available occupancy proxy can be estimated.”**

Table II 标题建议：

> **Exploratory Break-Even Occupancy under Simulator-Tracked Jammer Occupancy**

而不是单纯 “break-even jammer spectral occupancy”。

---

# 14. P0-9：统计程序补全

## 14.1 Bootstrap

必须写：

- draws = 10,000（若 artifact 最终核对一致）；
- resampling unit；
- source cluster 如何分层；
- class stratification；
- seed 是固定 block 还是随机层；
- simultaneous max-absolute-deviation 的计算步骤；
- paired nature 如何保留。

---

## 14.2 Design resolution = 1.31 pp

必须说明：

- 定义；
- 推导；
- 来自哪一组统计量；
- 是 preregistered 还是 exploratory；
- 不能把 exploratory resolution 当成 sealed significance threshold。

并将：

> “about three times”

改为：

> “about 3.5 times”

如果仍保留这句话。

---

## 14.3 Sign-flip p = 0.0018

必须写清：

- exact permutation 还是 Monte Carlo；
- test statistic；
- one-sided / two-sided；
- permutation count；
- 为什么不是简单 2/1024。

如果无法重构 0.0018，应改为可完全复核的 exact sign test 或准确说明实现。

---

# 15. P0-10：severe-corner 聚合必须可从正文重构

当前 SIR = −15 dB：

- A5 vs MCLDNN +9.74 pp；
- A5 vs IQFormer-inspired +10.46 pp。

但 Fig. 2 每个 SNR cell 的简单平均与这一数字不完全相等。

V1 必须说明：

- 是否先在 source level pairing；
- 是否按 class/source weighting；
- 是否跨 SNR pooling；
- macro-F1 是先 class-average 再 seed-average，还是其他顺序；
- Fig. 2 cell average 与 Fig. 3 aggregated contrast 为什么不同。

建议在 §V-B 加一句：

> “The SIR = −15 dB aggregate is computed from source-paired predictions pooled across the eight SNR conditions before macro-F1 aggregation; it is therefore not the arithmetic mean of the eight displayed cell-level differences.”

仅在实际实现确实如此时使用。

---

# 16. P0-11：capacity ladder 的参考口径统一

评审发现：

- 主结果使用 10 seeds；
- capacity ladder 可能只有 5 seeds；
- IQFormer reference 出现 48.72 vs 49.61；
- MCLDNN 45.99 vs 45.41；
- S tier 与 A5 同为 39,500 params，但 hard score 不完全相同。

必须解释：

1. ladder 用多少 seeds；
2. ladder reference 是否重新训练；
3. why 5-seed comparator means differ from 10-seed main result；
4. S tier 与 A5 的 exact config 差异；
5. Table IV 的 reference 是否应统一到 matched five-seed subset。

### 原则

**绝不混用 5-seed candidate 与 10-seed reference。**

如果 ladder 只有 5 seeds，应全部使用相同 5-seed matched subset。

---

# 17. P0-12：Fig. 2 星号与 Holm 必须对应

当前图上星号数量与“12/64 survive Holm”容易让读者误解。

图注必须明确：

若星号是 uncorrected paired CI positive：

> “A star marks a positive unadjusted paired lower interval; Holm-surviving cells are reported separately in the text.”

如果星号实际代表 Holm-surviving，则重新绘图确保数量一致。

优先建议：

- 圆点/星号表示 nominal interval；
- 外框或双星表示 Holm-surviving；
- 或只显示 Holm-surviving 标记。

---

# 18. P0-13：Fig. 5 必须重画

当前 Pareto 图缺：

- sidecar；
- M tier；
- L tier。

这会埋掉最强 performance–complexity 信息。

### 新 Fig. 5 应至少包含

- A0；
- A5；
- sidecar；
- M/L；
- MCLDNN；
- IQFormer-inspired；
- CSSL。

横轴：

- parameters（log scale）；
- 可在 supplement 或第二 panel 加 MACs。

### 图中最重要的视觉信息

sidecar：

- 46.8k total parameters；
- hard score near IQFormer-inspired；
- 远左于 355k 参数点。

但图注必须强调：

> sidecar is prospective Tier-2 exploratory repair.

---

# 19. P0-14：Table III 增加复杂度列

建议修改：

| Intervention | Question | Total Params | MACs | Clean diff [CI] | Hard diff [CI] | Decision |
|---|---|---:|---:|---|---|---|

这样 sidecar 的“+18% params / +3% MACs”一眼可见。

---

# 20. P0-15：Method 从“太短”扩成可复核版本

当前 7 页过度压缩。

TVT Regular Paper 不需要继续维持 7 页。

建议正文扩至 **9–10 页左右**，重点不是“写长”，而是补足审稿人真正会问的复现信息。

---

## 20.1 Architecture Table

至少列：

- input sequence length；
- STFT size；
- window；
- hop；
- FFT；
- number of real/imag/log-mag planes；
- allocation encoder；
- route dimensions；
- \(\lambda\)；
- \(\rho\)；
- pooling；
- classification heads；
- total parameters；
- MACs。

---

## 20.2 Loss 完整数学化

\[
\mathcal{L}
=
\mathcal{L}_{\rm mod}
+\alpha\mathcal{L}_{\rm teacher}
+\beta\mathcal{L}_{\rm jammer}
+\gamma\mathcal{L}_{\rm link}
+\eta\mathcal{L}_{\rm contrast}.
\]

补：

- 每个 loss 定义；
- 权重；
- contrastive positive/negative construction；
- temperature；
- teacher information；
- inference-time information。

---

## 20.3 Information Contract Table

| Quantity | Training | Test inference | Post-hoc analysis |
|---|---:|---:|---:|
| received mixture | ✓ | ✓ | ✓ |
| tracked target component | ✓ teacher only | ✗ | ✓ simulator audit |
| tracked jammer component | ✓ teacher only | ✗ | ✓ occupancy |
| clean counterfactual | ✗/probe only | ✗ | Tier-2 only |
| occupancy \(o\) | not required for classifier | ✗ | ✓ envelope |
| true SIR | simulator condition | ✗/metadata | ✓ envelope |

这张表能一次性消除 oracle-information 混淆。

---

# 21. P0-16：Baseline fairness 表

必须建立统一 retraining protocol。

| Model | Representation | Params | MACs | Seeds | Same source cache? | Training budget | Reproduction status |
|---|---|---:|---:|---:|---|---|---|
| A0 | direct spectral | ... | ... | 10 | Yes | ... | internal control |
| A5/VIMD | spectral allocation | 39.5k | 41.8M | 10 | Yes | ... | proposed config |
| sidecar | spectral + received I/Q | 46.8k | 43.1M | ... | Yes | ... | Tier-2 |
| MCLDNN | I/Q | ... | ... | 10 | Yes | ... | unified retraining |
| IQFormer-inspired | multimodal/IQ | 354,984 | 355.6M | 10 | Yes | ... | architecture-inspired |
| CSSL | ... | ... | ... | ... | Yes | ... | official architecture / adapted training |

### 必须统一命名

全文：

> IQFormer-inspired

不能图上写 IQFormer、正文写 IQFormer-inspired。

---

# 22. P0-17：CSSL 的异常表现要解释

CSSL 参数远大于 A0，但 hard score 低于 A0。

不用长篇辩护，但至少写：

> “CSSL is included as an adaptation-oriented architecture reference; under the unified structured-interference retraining protocol it underperforms the direct spectral A0 baseline, so we do not use it as evidence of state-of-the-art superiority.”

否则 Reviewer 会怀疑复现质量。

---

# 23. P0-18：Related Work 完整重构

建议从“逐网络罗列”改成四层。

## A. Strong AMC representations

- CNN / recurrent；
- complex-valued；
- dual-stream；
- Transformer；
- IQFormer。

## B. Robust / cross-domain AMC

- SigDA；
- MCLHN；
- contrastive learner；
- confidence-guided domain adaptation；
- DenoMAE2.0。

## C. Interference-aware / malicious-interference AMC

重点直接讨论：

- Sun et al., TVT 2025；
- Zhang et al., TVT 2026。

不要只引用而不比较。

## D. Knowledge-guided / multimodal / efficient frontier

正式发表与预印本分开：

- efficient/lightweight AMC；
- DKDNet 2026 preprint；
- MoEformer 2026 preprint；
- ZoomSpec 2026 preprint；
- RIS-MAE 2025 preprint。

### 本文差异段落

> Existing work has made substantial progress in interference-tolerant recognition, domain adaptation, lightweight architectures, and multi-representation fusion. Our contribution is not another claim that a particular representation dominates on average. We instead study the **condition dependence of the ranking itself**, fit a low-dimensional empirical boundary to that rank exchange, expose negative regions rather than reporting only favorable averages, and prospectively intervene on the revealed representation boundary.

---

# 24. P0-19：TVT scope 落脚——加强但不过度

V1 不建议为了 TVT 硬加“完整 V2X geometry”。

更稳妥的定位：

> **terrestrial mobile / high-mobility receiver-side AMC under structured interference**

需要在 System Model 中补：

- carrier frequency；
- speed range；
- Doppler mapping；
- TDL profiles；
- training/held-channel split；
- receiver-side use case。

Introduction 应解释：

> 对移动/车载 receiver，average accuracy 不足以指导受限计算平台何时选择 compact spectral model、何时保留高容量 I/Q path。

但继续明确：

> 本文不模拟完整车联网拓扑、不主张 MAC/network-layer V2X performance。

---

# 25. P0-20：摘要必须重写，但保留证据等级

V1 摘要建议 6 句逻辑。

## Sentence 1 — problem

平均 AMC 排名掩盖不同 representation 在 structured-interference conditions 下的 rank exchange。

## Sentence 2 — protocol/method

介绍 source-disjoint TR 38.901 TDL-profile simulation campaign、compact spectral A5 与强 I/Q comparators。

## Sentence 3 — sealed result

只报告：

> positive sealed whole-configuration A5–A0 contrast +4.57 pp [3.65, 5.49]。

不要说 teacher / route confirmed。

## Sentence 4 — severe corner

明确写：

> “In a post-hoc severe-interference analysis…”

再报 +9.74 / +10.46。

## Sentence 5 — envelope

用 RMSE / \(R^2_{\rm skill}\) 代替 sign 0.975 headline。

## Sentence 6 — repair + boundary

写 sidecar：

- total 46.8k；
- +18% params；
- clean +11.77；
- hard +4.37；
- exploratory；
- simulation-only；
- not uniform superiority / not deployment-ready.

---

# 26. Evidence Status Box：简化

当前第一页 Evidence status 黑话过多。

建议改成三行：

> **Evidence status.** The original sealed family resolves one positive whole-configuration contrast (A5 vs A0); two component contrasts and the clean-retention gate do not pass. The severe-interference, operating-envelope, and Tier-2 repair analyses are exploratory or post-hoc and are labeled as such. All conclusions are restricted to source-disjoint simulation.

这样既诚实，又不会一上来把读者淹没在 Tier-2 / family gate jargon 中。

---

# 27. 记号表必须加

建议新增小表：

| Symbol/ID | Meaning |
|---|---|
| A0 | direct spectral reference backbone |
| A5 | VIMD-Net sealed full configuration |
| A7 | residual-free control |
| S/M/L | prospective spectral capacity ladder |
| sidecar | Tier-2 received-I/Q repair model |
| CSSL | full name + citation |
| IQFormer-inspired | unified-retraining strong I/Q reference |
| macro-F1 | unweighted mean of class-wise F1 |

并解释：

> S 是否等于 A5 architecture，为什么同为 39.5k 但分数不同。

---

# 28. Clean boundary：措辞收紧但地位提升

不要说：

> clean boundary is definitively architectural.

改为：

> “The Tier-2 evidence is more consistent with a representation-information gap than with insufficient clean exposure alone.”

因为：

- probe 是 exploratory；
- coverage arm 是 exploratory；
- sidecar 是 exploratory；
- capacity ladder 是 exploratory。

但链条的一致性仍然非常有价值。

---

# 29. Sidecar 的科学角色

明确：

> sidecar is **not the new main classifier replacing A5**.

它的作用：

> 检验 “missing I/Q information” hypothesis 是否具有可修复性。

正文建议：

> “The sidecar was designed prospectively after the clean-boundary diagnosis. It is therefore interpreted as an intervention on the hypothesized information bottleneck rather than as retroactive evidence for the original A5 architecture.”

---

# 30. Fusion：降级为 supporting evidence

A5 + IQFormer fusion 有互补性，保留。

如果现有 predictions 可直接重算：

> 建议补 sidecar + IQFormer fusion。

这是 **zero/near-zero training cost** 的高性价比检查。

如果 sidecar 与 IQFormer 互补性消失，也不影响主线；如仍有提升，可增强“representation complementarity”。

---

# 31. Selective prediction：继续保留但压缩

风险–coverage curve 是完整性证据，不要变成主贡献。

正文一句：

> “Selective prediction shows that compact and I/Q models converge at low retained coverage, but this result is supporting rather than central evidence.”

---

# 32. 公开数据集外部有效性：高价值，但要避免做错

两份评审中一版强烈建议 RadioML2018.01a。

这个建议有价值，但必须注意：

> RadioML2018.01a 本身不是本文 structured-jammer occupancy protocol，不能天然验证 occupancy–SIR envelope。

---

## 正式建议：分两级

### Level 1 — 强烈推荐，若投稿前算力允许

在 RadioML2018.01a 或同类公开 AMC benchmark 上做：

- A5；
- sidecar；
- MCLDNN；
- IQFormer-inspired。

目标只回答：

> “本文 representation tradeoff 是否完全依赖私有 simulator？”

报告：

- clean/noisy marginal ranking；
- parameter efficiency；
- sidecar 是否改善 pure spectral representation；
- 结果若为负也如实报告。

### 不建议直接做

> 用一个未经验证的“带内能量占比”立即替代 jammer occupancy，然后声称 envelope transfer。

因为这会引入一个新的 proxy-definition 问题。

---

## Level 2 — 更高价值但可留后续

设计 receiver-available occupancy proxy：

- energy concentration；
- spectral kurtosis；
- occupied-bandwidth estimator；
- learned interference-presence/occupancy estimator。

然后在 frozen simulator held regimes 上测试：

- proxy vs oracle occupancy correlation；
- envelope skill degradation；
- sign/magnitude performance。

这是从“simulation reference envelope”走向“deployable selector”的真正一步。

---

# 33. 真实世界碰撞与未来方向

2026 最新公开工作已经出现：

- 真实/野外 jamming datasets；
- occupied bandwidth / jammer characterization；
- real-world spectrum sensing + time/frequency/IQ fusion。

这说明 Reviewer 对“纯仿真”的外部有效性要求只会越来越高。

V1 应在 Limitations 中主动说：

> “The present campaign establishes an internally controlled operating boundary. Validation with receiver-observable occupancy estimates and over-the-air or public interference recordings is a separate external-validity step.”

不要把它写成已经完成。

---

# 34. References：从 21 篇扩到约 30–38 篇

不需要为了数量堆文献。

优先补：

### 必补正式发表

1. Sun et al., *Toward Interference-Tolerant Automatic Modulation Recognition via Multi-stage Feature Extraction Network*, IEEE TVT, 2025.
2. Zhang et al., *Modulation Recognition via Tensor Analysis for MIMO Systems with Malicious Interference*, IEEE TVT, 2026.
3. DenoMAE2.0, IEEE Transactions on Communications, 2026.
4. SigDA, IEEE TWC, 2024.
5. MCLHN, IEEE TWC, 2024.
6. Contrastive Learner for AMC, IEEE TWC, 2025.
7. Confidence-Guided Prototypical Contrastive Domain Adaptation, IEEE TCCN, 2026.
8. IQFormer, IEEE TCCN, 2025.
9. 一篇正式 lightweight/efficient AMC 2025–2026 代表工作。
10. 3GPP TR 38.901 标准。

### 可补前沿预印本，明确标为 preprint

- DKDNet, arXiv:2607.08031, 2026.
- MoEformer, arXiv:2606.09085, 2026.
- ZoomSpec, arXiv:2604.13568, 2026.
- RIS-MAE, arXiv:2508.00274, 2025.
- Jammertest Norway 2025 dataset preprint, 2026（若 Discussion 谈真实 jammer characterization）。

### 原则

预印本用于：

> “frontier positioning”

不能替代正式文献支撑基础科学结论。

---

# 35. 限制条件扩充为 7 条

V1 Limitations 至少明确：

1. simulation-only；
2. not complete V2X geometry；
3. teacher uses simulator component bookkeeping；
4. occupancy is simulator-derived and unavailable directly at receiver；
5. original confirmatory family gate did not pass；
6. clean-retention gate did not pass；
7. severe/envelope/repair/capacity results are exploratory/post-hoc and require future prospective confirmation。

另可补：

8. public/OTA external validation not yet established；
9. IQFormer-inspired 等为 unified-retraining references，不是所有原论文 recipe 的逐细节复制。

---

# 36. Receiver stress split：必须处理

Evidence Protocol 声称存在 receiver stress split，但正文几乎没有结果。

V1 二选一：

### A. 如果该 split 有完整 artifact

在 supplement / brief table 报告。

### B. 如果不参与本文核心

从“11 evaluation splits”描述中说明：

> receiver-stress split is retained in the artifact package but not used for the claims in this paper.

不能列出后完全消失。

---

# 37. Reproducibility：从口号变成可检查内容

正文现在说每个 macro 有：

- source path；
- row selector；
- evidence class；
- precision；
- SHA-256。

正式投稿包必须至少满足其一：

### 方案 A

公开完整 provenance manifest。

### 方案 B

如果匿名/数据限制不能公开：

正文明确：

> what will be released upon acceptance / what remains private.

不能让正文宣称“可追溯”，但 submission material 中完全不可见。

---

# 38. 正式 V1 文章结构

## I. Introduction

- mobile/high-mobility receiver motivation
- why marginal AMC ranking is insufficient
- structured interference
- rank-exchange question
- 4 bounded contributions

## II. Related Work and Claim Boundary

### A. Strong and efficient AMC representations  
### B. Robust/cross-domain AMC  
### C. Interference-aware and malicious-interference AMC  
### D. Distinction: condition-indexed ranking and repair

## III. System Model and VIMD Configuration

### A. Mobile received signal model  
### B. Three-route spectral allocation  
### C. Simulation-component teacher  
### D. Training objective  
### E. Information contract and complexity

## IV. Evidence Protocol

### A. Frozen source-disjoint campaign  
### B. Comparator fairness  
### C. Original sealed family  
### D. Exploratory / Tier-2 evidence classes  
### E. Statistical procedures

## V. Results

### A. Positive Sealed Whole-Configuration Contrast and Bounded Components  
### B. Rank Exchange across the SNR–SIR Plane  
### C. Empirical Occupancy–SIR Operating Envelope  
### D. Complexity and Complementarity  
### E. Clean-Boundary Diagnosis  
### F. Prospective I/Q Repair and Capacity Controls

## VI. Discussion

### A. What the operating envelope establishes  
### B. What it does not establish  
### C. Meaning of the failed directional hypothesis  
### D. Representation repair and practical model selection  
### E. External validity and deployment proxies

## VII. Limitations and Reproducibility

## VIII. Conclusion

---

# 39. 推荐贡献列表：压成四条

### Contribution 1

> We expose a condition-dependent rank exchange between a compact spectral model and stronger I/Q-domain classifiers over the complete structured-interference SNR–SIR plane, rather than reporting a single marginal ranking.

### Contribution 2

> We fit a low-dimensional occupancy–SIR empirical envelope and evaluate its magnitude skill on regime cells excluded from fitting, with explicit trivial-baseline comparison and bounded interpretation.

### Contribution 3

> We separate the original sealed whole-configuration contrast from post-hoc severe-interference evidence, using paired statistics, multiplicity control, and full-region visualization to prevent a favorable corner from being read as uniform superiority.

### Contribution 4

> We turn a failed clean-retention gate into a prospective diagnosis-and-repair chain; a received-I/Q sidecar increases the total model size only modestly while repairing clean and hard-interference performance, and a spectral capacity ladder shows that width alone does not remove the boundary.

---

# 40. 逐项措辞替换表

| 当前/风险措辞 | V1 推荐 |
|---|---|
| confirmed whole-method effect | positive sealed whole-configuration contrast |
| physics-guided mechanism is confirmed | physics-informed training construction; component mechanism unresolved |
| predicts unseen regimes | predicts held regime cells within the frozen simulator campaign |
| out-of-distribution | held regime / held condition，除非真的是新分布 |
| directly usable diagnostic | simulator-domain reference conditional on an occupancy estimate |
| causal law | empirical descriptive/predictive envelope |
| sidecar adds 46,794 parameters | sidecar model totals 46,794 parameters, +7,294 vs A5 |
| compact beats IQFormer | compact model shows a post-hoc advantage in the severe-interference region |
| architecture-level clean boundary | exploratory evidence is more consistent with a representation-information gap |
| about three times design resolution | about 3.5 times，若该 comparison 仍保留 |
| IQFormer | IQFormer-inspired（统一） |

---

# 41. P1 高价值新增工作：按回报排序

| 优先级 | 新工作 | 训练成本 | 价值 | 是否 V1 必须 |
|---|---|---:|---:|---|
| P1-A | RadioML2018.01a public benchmark sanity check | 中 | 很高 | 强烈推荐 |
| P1-B | capacity-matched A0 prospective control | 中 | 很高 | 推荐 |
| P1-C | receiver-available occupancy proxy | 中 | 极高 | 可留下一轮 |
| P1-D | sidecar + IQFormer fusion using existing predictions | 低/零 | 中 | 建议 |
| P1-E | coefficient bootstrap / stability analysis for envelope | 低 | 高 | 建议 |
| P1-F | latency on same hardware | 低 | 中 | 若已有数据则补 |

---

# 42. 新实验原则

任何在 2026-08-18 之后根据当前结果新增的实验：

- 不能称为 original sealed；
- 必须写清“designed after inspecting the frozen campaign”；
- 如要叫 confirmatory，应新建独立 preregistration / prospective protocol；
- 不能用新增正结果覆盖原始失败 gate；
- 负结果同样保留。

这是本文可信度的底线。

---

# 43. V1 Artifact Audit 清单——正文改写前先做

以下内容先对账，再写稿：

- [ ] sidecar 46,794 是否为 total params；
- [ ] sidecar total MACs 43.1M 是否与相同 MAC estimator 一致；
- [ ] A5 total params 39,500；
- [ ] A0 params 8,458；
- [ ] IQFormer-inspired total params / MACs；
- [ ] MCLDNN total params / MACs；
- [ ] envelope \(N_{\rm fit}\)；
- [ ] envelope \(N_{\rm holdout}=81\)；
- [ ] holdout positive/negative prevalence；
- [ ] constant-baseline RMSE；
- [ ] \(R^2_{\rm skill}\)；
- [ ] pooled occupancy–SIR correlation；
- [ ] hard-subset occupancy–SIR correlation；
- [ ] occupancy exact formula；
- [ ] severe SIR aggregate calculation；
- [ ] Fig. 2 star meaning；
- [ ] Holm 12/64 exact cells；
- [ ] bootstrap 10,000 draws implementation；
- [ ] sign-flip p=0.0018 implementation；
- [ ] design resolution 1.31 derivation；
- [ ] A7 vs A5 diff + CI；
- [ ] capacity ladder seed count；
- [ ] capacity ladder matched comparator references；
- [ ] S-tier vs A5 exact configuration difference；
- [ ] receiver-stress split disposition；
- [ ] CSSL reproduction/training status；
- [ ] provenance manifest actual submission availability。

**原则：任何两份评审中出现不一致的数字，都以 artifact 为唯一真值。**

---

# 44. 图表最终布局建议

## Fig. 1
Teacher / information-contract visualization。保留。

## Fig. 2
Full SNR–SIR rank-exchange map。核心图，保留。

## Fig. 3
Severe corner + paired intervals。保留，显式 exploratory。

## Fig. 4
Operating envelope：
- 左：physical plane；
- 右：predicted vs observed；
- 增加 constant-baseline RMSE / skill annotation。

## Fig. 5
重画为 complete performance–complexity frontier：
- A0/A5/sidecar/S/M/L/MCLDNN/IQFormer-inspired/CSSL。

## Fig. 6
risk–coverage，可缩小。

## Fig. 7
clean-transfer taxonomy：
- 星号必须定义；
- caption 自洽。

### 如篇幅超 10 页

优先把：
- Fig. 6 selective prediction；
- fusion 细节；
- extended per-class table；

移 supplement。

不要移：
- Fig. 2；
- Fig. 4；
- sidecar repair；
- evidence table。

---

# 45. 建议新增三张表

## Table A — Comparator and complexity summary

参数、MAC、seed、domain、retraining status。

## Table B — Evidence hierarchy

| Result | Original sealed? | Post-hoc? | Prospective Tier-2? | Allowed claim |
|---|---|---|---|---|

## Table C — Repair chain

在现 Table III 基础上加：
- total params；
- MACs；
- evidence class。

这样全文“证据等级”不需要靠黑话反复解释。

---

# 46. TVT 篇幅策略

正式 Regular Paper 初稿允许的页数高于当前 7 页，但超长页面会带来费用和阅读负担。

本稿最适合：

> **9–10 页左右。**

原因：

- 7 页导致 Method / statistics / comparator fairness 被压缩；
- 9–10 页足以解释；
- 没必要扩成 13–14 页；
- 强稿不是靠长度，而是靠可复核性。

---

# 47. 投稿前格式与内部内容

必须删除：

> Internal integration manuscript...

正式稿补全：

- authors；
- affiliations；
- funding；
- conflicts；
- data/code availability；
- acknowledgments；
- 如使用 AI 进行语言编辑，按 TVT 当前要求做适当披露。

---

# 48. V1 的“最小可投版本”与“增强版”区分

## 最小可投 V1 —— 不新增训练

必须完成：

- 摘要换掉 sign headline；
- sidecar 参数纠错并提升；
- title / novelty 重新定位；
- physics-guided mechanism claim 降级；
- A5/A0 capacity confounding 明示；
- occupancy 定义；
- occupancy/SIR correlation artifact audit；
- constant-baseline skill；
- bootstrap/stat protocol；
- severe aggregation；
- capacity ladder seed/ref consistency；
- Fig. 2/5/7 修复；
- baseline fairness；
- Related Work 扩充；
- TVT scope；
- limitations；
- internal note 删除。

做到这里，论文已经明显强于当前版本。

---

## 增强版 V1 —— 推荐投稿版本

在最小版基础上加：

1. RadioML2018.01a public benchmark sanity check；
2. capacity-matched prospective A0 control；
3. sidecar fusion zero-compute analysis；
4. envelope coefficient stability / bootstrap；
5. latency（若已有统一硬件数据）。

---

# 49. 预计 Reviewer 攻击与 V1 防御

## 攻击 1
“你的 physics-guided 组件根本没被确认。”

### V1 回应
同意组件 effect unresolved；论文不再以单组件 causal mechanism 为 headline，只主张 whole-configuration contrast + condition-dependent boundary。

---

## 攻击 2
“0.975 sign 是 majority-class trick。”

### V1 回应
摘要不再使用该指标作为主证据；报告 majority baseline、RMSE、skill score、chance-adjusted sign metric。

---

## 攻击 3
“occupancy deployment 不可见。”

### V1 回应
明确 simulator-derived；break-even 只是 oracle/reference boundary；部署 proxy 是未来工作/新增实验。

---

## 攻击 4
“你这个 sidecar 不就是别人做过的 IQ + spectral fusion？”

### V1 回应
不把 architecture fusion 当 novelty；sidecar 是在 frozen clean failure 后设计的 prospective intervention，用于检验 representation-gap hypothesis。

---

## 攻击 5
“compactness 只是参数故事。”

### V1 回应
capacity ladder + sidecar + IQFormer comparison共同说明：
- width 能提高平均表现；
- pure spectral width 不消除 boundary；
- small I/Q addition 改变 clean/hard tradeoff。

---

## 攻击 6
“纯仿真，没有 external validity。”

### V1 回应
最好加 public benchmark sanity check；否则明确 controlled simulation scope，不声称 real-world deployment。

---

# 50. 完成判据

正式 V1 只有在以下全部完成后才进入第二轮英文精修：

- [ ] 标题从 architecture-first 调整为 envelope + repair；
- [ ] Abstract 不再以 0.975 sign agreement 为 headline；
- [ ] majority-sign baseline 明示；
- [ ] RMSE skill score 明示；
- [ ] sidecar “46,794 total” 表述纠正；
- [ ] sidecar +18% params / ~3% MACs 信息明示；
- [ ] sidecar 进入 Abstract / contribution / Fig. 5；
- [ ] whole-method effect 改为 whole-configuration contrast；
- [ ] A5/A0 capacity/bundle confounding 明示；
- [ ] teacher/tri-route component mechanism 不再被确认性宣传；
- [ ] failed directional hypothesis 独立讨论；
- [ ] occupancy exact formula 给出；
- [ ] occupancy–SIR correlation 从 artifact 重新核对；
- [ ] fit/holdout cell count 从 artifact 重新核对；
- [ ] severe corner 明确是 post-hoc 且不是 holdout proof；
- [ ] bootstrap procedure 可复核；
- [ ] sign-flip p 可复核；
- [ ] design resolution 可复核；
- [ ] capacity ladder seed/reference 统一；
- [ ] Fig. 2 星号/Holm 关系清楚；
- [ ] Fig. 5 加 sidecar/M/L；
- [ ] Fig. 7 星号定义；
- [ ] comparator fairness table；
- [ ] notation table；
- [ ] CSSL 情况解释；
- [ ] receiver stress split 处理；
- [ ] references 扩充并纳入 2025–2026 直接碰撞工作；
- [ ] TVT high-mobility/mobile receiver relevance 明确；
- [ ] 不声称 complete V2X geometry；
- [ ] deployment limitation 明示；
- [ ] internal integration note 删除；
- [ ] 全稿控制约 9–10 页；
- [ ] 所有新实验 evidence class 明确。

---

# 51. V1 最终定位

正式 V1 不再是一篇：

> **“一个新 physics-guided AMC 网络在干扰下胜过大网络”**

的论文。

它应成为：

> **“一篇研究 AMC representation 排名何时交换、如何用物理条件刻画这一边界、如何在发现失败后用前瞻干预定位并修复表示缺口的证据型 TVT 论文。”**

这也是当前证据最强、与 2024–2026 前沿碰撞最小、最能抵御审稿攻击的定位。

---

# 52. 前沿审计参考（建议纳入或用于定位）

> 以下仅列本轮碰撞审计中最相关的代表工作；正式参考文献仍需按 IEEE 格式统一。

### Peer-reviewed

1. X. Sun et al., “Toward Interference-Tolerant Automatic Modulation Recognition via Multi-stage Feature Extraction Network,” *IEEE Transactions on Vehicular Technology*, 2025. DOI: 10.1109/TVT.2025.3555769.
2. J. Zhang et al., “Modulation Recognition via Tensor Analysis for MIMO Systems with Malicious Interference,” *IEEE Transactions on Vehicular Technology*, 2026. DOI: 10.1109/TVT.2026.3674535.
3. A. Faysal et al., “DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches for Automatic Modulation Classification,” *IEEE Transactions on Communications*, 2026. DOI: 10.1109/TCOMM.2025.3626031.
4. S. Wang et al., “SigDA: A Superimposed Domain Adaptation Framework for Automatic Modulation Classification,” *IEEE Transactions on Wireless Communications*, 2024. DOI: 10.1109/TWC.2024.3399067.
5. C. Xiao et al., “MCLHN: Toward Automatic Modulation Classification via Masked Contrastive Learning with Hard Negatives,” *IEEE Transactions on Wireless Communications*, 2024. DOI: 10.1109/TWC.2024.3412234.
6. M. Du et al., “A Contrastive Learner for Automatic Modulation Classification,” *IEEE Transactions on Wireless Communications*, 2025. DOI: 10.1109/TWC.2025.3532438.
7. W. Li et al., “Confidence-Guided Prototypical Contrastive Domain Adaptation for Cross-Domain Automatic Modulation Classification,” *IEEE Transactions on Cognitive Communications and Networking*, 2026. DOI: 10.1109/TCCN.2026.3677189.
8. M. Shao et al., “IQFormer: A Novel Transformer-Based Model With Multi-Modality Fusion for Automatic Modulation Recognition,” *IEEE Transactions on Cognitive Communications and Networking*, 2025.
9. “Efficient Automatic Modulation Classification for Next-Generation Wireless Networks,” *IEEE Transactions on Green Communications and Networking*, 2026. DOI: 10.1109/TGCN.2025.3574278.

### Recent preprints / frontier watch

10. “Dual Knowledge and Data-Driven Network for Cross-Domain Automatic Modulation Classification (DKDNet),” arXiv:2607.08031, 2026.
11. “MoEformer … Automatic Modulation Classification,” arXiv:2606.09085, 2026.
12. “ZoomSpec … Physics-Guided … Spectrum Sensing,” arXiv:2604.13568, 2026.
13. “RIS-MAE … Automatic Modulation Classification,” arXiv:2508.00274, 2025.
14. “Jammertest Norway 2025 …,” 2026 preprint, for real-world jammer characterization / occupied-bandwidth context.

---

**正式版本状态：V1.0 — ready for execution.**  
**下一阶段：** 完成 Artifact Audit → 逐章节施工式改稿 → 新 Figure/Table → Abstract/Title 精修 → TVT Reviewer Attack Audit V2。
