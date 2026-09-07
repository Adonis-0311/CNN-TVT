# TVT 投稿前终局改稿执行方案 V1.2
## 基于 `tvt_operating_envelope_integration_V1.pdf` 的最终定向修改清单

**目标期刊：** IEEE Transactions on Vehicular Technology (TVT)  
**当前稿件版本：** `tvt_operating_envelope_integration_V1.pdf`  
**当前状态：** 主线、证据等级、TVT 场景定位、方法复现性、baseline fairness、sidecar 与 capacity control 已基本成型。  
**本轮性质：** 投稿前最后一轮技术性修订，不再进行结构性大改。  
**核心原则：** 不改变论文主张层级，不增加新的 architecture-first 叙事，不把 exploratory evidence 升格为 confirmatory evidence。

---

# 0. 本轮结论

当前稿件已经完成以下关键升级：

- condition-indexed rank exchange 成为一级主线；
- operating envelope 不再依赖 sign agreement 作为 headline；
- envelope baseline 已统一为 fit-cell constant predictor；
- Fig. 4 已转为 magnitude prediction；
- high-mobility / TVT 物理场景已定量化；
- implementation details 已显著补全；
- A5–A0 明确为 whole-configuration contrast；
- sidecar 已正确定位为 prospective representation intervention；
- capacity ladder 已解决 five-seed / ten-seed 参考口径；
- recent related work 与 references 已基本补齐。

因此本轮不再重写标题、贡献列表或全文结构，仅处理下列 **5 个关键剩余问题 + 若干投稿前 QA 项**。

---

# 1. P0：修正 Loss 公式编号引用错误

## 当前问题

Implementation Details 中目前出现：

> “Loss weights. In (5), \(\alpha = 0.50, \beta = 0.25, \gamma = 0.05,\eta = 0.10\)...”

以及：

> “A small route-orthogonality regularizer ... is omitted from (5) for readability...”

但总损失函数实际编号为 Eq. (7)。

## 必改

- `In (5)` → `In (7)`
- `omitted from (5)` → `omitted from (7)`

## 推荐最终句

> “For (7), \(\alpha=0.50\), \(\beta=0.25\), \(\gamma=0.05\), and \(\eta=0.10\), with label smoothing 0.05 and contrastive temperature 0.12. A route-orthogonality regularizer with weight 0.01 is also used in the complete configuration but is omitted from (7) for readability.”

---

# 2. P0：修正 “only the architecture changes” 的不准确表述

## 当前句子

> “every model in this paper is trained on the same source-disjoint cache, with the same optimization budget, the same stopping rule, and the same validation-only model-selection criterion; only the architecture changes.”

## 风险

该句与 Table IV 不一致。A0 与 A5 的差异不仅是 architecture，还包括 parameter count、three-route allocation、simulation-component teacher、multi-task objectives、exact-source contrastive 和 residual bypass。

## 推荐替换

> “Unless otherwise stated, all models are trained on the same source-disjoint cache with the same data budget, optimization budget, stopping rule, and validation-only model-selection criterion. Model-specific architectural and objective components follow the predefined configuration of each variant.”

可接：

> “Thus, comparator fairness is enforced at the data, budget, and model-selection levels rather than by forcing structurally different models to share identical internal objectives.”

---

# 3. P0：给 occupancy threshold \(\tau\) 明确定义

## 当前问题

当前 occupancy 已定义为：

\[
o=\frac{1}{FT}\sum_{f,t}\mathbf{1}\left(P_j(f,t)>\tau\right),
\]

正文只说明：

> “a noise-floor-referenced threshold”

但没有明确 \(\tau\) 数值和实际实现方式。

## 必须从 artifact 读取真实实现

核对：

- occupancy threshold 数值；
- 相对 noise floor 的偏移；
- dB 还是线性功率；
- per-window 还是 global；
- 是否所有 split 使用同一阈值。

## 推荐写法

若阈值为固定 noise-floor 偏移：

> “The threshold is fixed at \(\tau=N_0+x\) dB on the tracked jammer-power lattice, where \(N_0\) is the simulator noise-floor reference. The same threshold is used for all splits and model comparisons.”

若实际实现不同，则按代码真实逻辑写，不能凭经验补。

---

# 4. P1：将 “high-order constellation classes” 改成更准确术语

## 当前句子

> “the unweighted mean F1 over the three high-order constellation classes (QPSK, 16QAM, 64QAM)”

## 推荐替换

> **“three constellation-order-sensitive classes (QPSK, 16QAM, and 64QAM)”**

完整句：

> “Probe performance is the unweighted mean F1 over three constellation-order-sensitive classes (QPSK, 16QAM, and 64QAM), ranging from 0 to 1.”

---

# 5. P1：将 envelope 的 “magnitude skill” 精确为 “cross-regime transport skill”

当前正文已经主动披露：

- IQFormer-inspired envelope RMSE = 6.20 pp；
- fit-domain constant baseline = 8.40 pp；
- held-regime oracle constant = 6.03 pp；
- MCLDNN envelope = 8.05 pp；
- fit-domain constant = 9.63 pp；
- held-regime oracle constant = 6.77 pp。

这意味着 envelope 的主要能力是跨 regime 搬运 gain level，而不是强解释 held-domain 内部 cell-to-cell variation。

## Abstract 建议

原：

> “magnitude, not sign, is the validated quantity.”

改为：

> **“the validated skill is cross-regime transport of gain magnitude rather than sign classification.”**

## Contribution 2 建议

> “a low-dimensional occupancy–SIR empirical envelope whose cross-regime gain-transport skill is evaluated on regime cells excluded from fitting, against an explicit fit-domain constant baseline and with a bounded, non-causal interpretation.”

## Section V-C 建议增加

> “Accordingly, we interpret the envelope primarily as a cross-regime gain-level transport model rather than as a fine-grained predictor of within-regime cell variation.”

## Conclusion 建议

原：

> “A low-dimensional occupancy–SIR envelope describes this exchange with measurable magnitude skill...”

改为：

> “A low-dimensional occupancy–SIR envelope transports the gain level across regime cells excluded from fitting with measurable skill relative to a fit-domain constant baseline...”

---

# 6. P1：probe causal language 保持收紧

保持：

> “more consistent with a representation-information gap”

避免：

> “proves that the spectral representation loses constellation information”

推荐：

> “Within the tested architecture family, the Tier-2 probe and intervention sequence is more consistent with a representation-information bottleneck than with insufficient clean exposure or classifier-head failure.”

---

# 7. P1：检查 5.9 GHz / V2X 引用是否精确支持正文措辞

当前：

> “The carrier frequency is \(f_c=5.9\) GHz, in the band used for vehicle-to-everything operation [26]...”

建议检查 [26] 的精确支持范围。

如需收紧：

> “\(f_c=5.9\) GHz, representative of ITS/V2X operation...”

---

# 8. P1：high-mobility coherence-time 论证再做一次物理量核算

投稿前重新核算：

\[
v=250/3.6\ \mathrm{m/s},
\]

\[
f_D=\frac{vf_c}{c},
\]

\[
T_c\approx\frac{0.423}{f_D}.
\]

确保正文中的：

- 1367 Hz
- 309 μs

一致。

推荐写：

> “using the conventional approximation \(T_c\approx0.423/f_D\)”

---

# 9. P1：Fig. 4 baseline 标签精确化

当前图内：

> `constant baseline = 8.40 pp`

建议改：

> **`fit-domain constant = 8.40 pp`**

或：

> **`fit-mean constant baseline = 8.40 pp`**

避免与正文的 held-regime oracle constant 混淆。

---

# 10. P1：Table VI 的 break-even occupancy 增加适用范围

建议补：

> “These thresholds should be interpreted only within the fitted SIR range and the simulator’s occupancy definition; extrapolation beyond that range is not evaluated.”

---

# 11. P1：Comparator fairness 与 Implementation 的 optimizer 口径统一

当前存在潜在张力：

- Implementation：optimizer settings shared across every model；
- Comparator fairness：architecture-specific optimizer settings used only where an architecture cannot train under the shared ones。

## 若所有模型确实共享 AdamW

删除 architecture-specific exception。

## 若确有例外

Implementation 改为：

> “The shared schedule is the default; architecture-specific optimizer substitutions are used only when required for numerical stability and are disclosed in the comparator configuration.”

并列出具体模型。

---

# 12. P1：route-orthogonality regularizer 与 Table IV 对齐

当前新披露：

> route-orthogonality regularizer weight = 0.01

但 Table IV 未列出。

## 推荐

Table IV 增加：

| Property | A0 | A5 |
|---|---|---|
| Route-orthogonality regularizer | No | Yes |

这样避免 Reviewer 认为 A5/A0 bundle 还有未披露差异。

---

# 13. P1：A7 vs A5 的 0.21 pp 表述

若 artifact 中有 CI，补出 paired CI。

若没有，推荐写：

> “A7 is nominally 0.21 pp above A5 on the marginal aggregate; this difference is below the design-resolution scale and is not treated as a resolved effect.”

---

# 14. P1：References 最终 QA

投稿前检查：

- 年份 / 卷 / 期 / 页码；
- 作者拼写；
- title capitalization；
- IEEE journal abbreviation；
- arXiv author metadata；
- URL 换行；
- GitHub commit；
- DeepSig accessed date；
- 3GPP TR 38.901 version；
- TR 37.885 是否适合作为 5.9 GHz / V2X 依据。

---

# 15. P1：最终 submission metadata

当前 PDF 仍为 Anonymous Author(s)。正式投稿前补：

- authors
- affiliations
- corresponding author
- funding
- conflicts
- data/code availability
- acknowledgment
- reproduction package statement
- AI-assisted language-editing disclosure（如实际使用）

AI disclosure 可沿用：

> “The authors used AI solely to assist with English-language editing and improving textual clarity. All technical content, mathematical derivations, data, code, figures, interpretations, and conclusions were produced and verified by the authors, who take full responsibility for the manuscript.”

---

# 16. 不再建议进行的修改

以下内容不要再动：

- 标题主方向；
- condition-indexed rank exchange 主线；
- severe corner 的 post-hoc 定位；
- family gate fail；
- clean-retention fail；
- sidecar Tier-2 身份；
- capacity ladder；
- Fig. 2 full-region map；
- Fig. 5 performance–parameter frontier；
- Evidence status；
- simulation-only scope。

---

# 17. 最终优先级

| 优先级 | 修改项 | 成本 | 必要性 |
|---|---|---:|---:|
| P0 | Eq. (5) → Eq. (7) 两处 | 极低 | 必须 |
| P0 | 删除 “only the architecture changes” | 极低 | 必须 |
| P0 | 补 occupancy threshold \(\tau\) | 低 | 必须 |
| P1 | high-order → constellation-order-sensitive | 极低 | 强烈建议 |
| P1 | magnitude skill → cross-regime transport skill | 低 | 强烈建议 |
| P1 | Fig. 4 baseline 标签精确化 | 极低 | 建议 |
| P1 | optimizer consistency | 低 | 建议 |
| P1 | route-orthogonality 加入 Table IV | 极低 | 建议 |
| P1 | Doppler/coherence time 复算 | 极低 | 建议 |
| P1 | References QA | 中 | 必须 |
| P1 | submission metadata | 投稿前 | 必须 |

---

# 18. 推荐修改后的核心贡献表述

## Contribution 1 — Condition-dependent rank exchange

> Compact spectral and high-capacity I/Q representations exchange rank across the structured-interference SNR–SIR plane, so a single marginal score is insufficient for model selection.

## Contribution 2 — Cross-regime operating-envelope transport

> An occupancy–SIR empirical envelope transports model-gain level from fitting conditions to held-regime cells with measurable skill relative to an explicit fit-domain constant baseline, while sign imbalance and oracle-mean diagnostics bound the interpretation.

## Contribution 3 — Evidence-bounded severe corner

> A post-hoc severe-interference advantage is reinforced by source-paired intervals, seed consistency, multiplicity control, and full-region visualization without being upgraded to confirmatory evidence.

## Contribution 4 — Prospective representation repair

> A failed clean-retention gate triggers a probe–coverage–negative-control–sidecar–capacity sequence; the evidence is more consistent with a repairable representation-information bottleneck than with insufficient exposure or parameter count alone.

---

# 19. 推荐终稿一句话定位

> **The paper is not a claim that a compact AMC network is uniformly superior; it is a study of when representation rankings change, how that change can be transported across physical conditions, and how a revealed representation boundary can be prospectively diagnosed and repaired.**

---

# 20. 本轮完成后的状态

完成以上修改后：

- 不建议再进行结构性重写；
- 不建议继续增加 baseline；
- 不建议临时加入设计不严谨的新实验；
- 直接进入 **TVT 投稿前终审**：
  1. 数字一致性检查
  2. 公式引用检查
  3. 图表/正文一致性检查
  4. references 检查
  5. submission metadata 检查
  6. 英文润色与 typo scan
  7. 最终 acceptance-risk audit

**V1.2 完成后即应冻结技术内容。**
