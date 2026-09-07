# TVT 投稿前最终改稿执行方案 V1.3
## 基于 `tvt_operating_envelope_integration_V2.pdf` 的终审前修订清单

**目标期刊：** IEEE Transactions on Vehicular Technology (TVT)  
**当前稿件：** `tvt_operating_envelope_integration_V2.pdf`  
**版本定位：** 技术内容冻结前最后一轮小修  
**总原则：** 不再改主线、不再扩实验、不再增加 baseline；只修正物理量命名、分析定义、内部一致性和摘要措辞，使所有 headline 与实际计算严格一致。

---

# 0. 当前状态判断

V2 已经完成上一轮几乎全部关键修订：

- Abstract 已将 envelope 主指标收敛为 **cross-regime transport of gain magnitude**；
- Contribution 2 已同步改为 **cross-regime gain-transport skill**；
- A5–A0 已明确为 whole-configuration contrast；
- Eq. (5) → Eq. (7) 交叉引用已修正；
- `only the architecture changes` 已删除；
- route-orthogonality regularizer 已加入 A0/A5 difference table；
- probe 中 “high-order constellation classes” 已改为更准确的 constellation-order-sensitive classes；
- 5.9 GHz / Doppler / coherence time / speed range 已定量化；
- Fig. 4 已使用 fit-domain constant = 8.40 pp；
- capacity ladder 的 five-seed matched subset 已解释清楚；
- sidecar 仍保持 prospective Tier-2 repair 的正确定位。

因此，V1.3 不做结构性重写，只处理以下 **4 个核心问题 + 投稿前 QA**。

---

# 1. P0：重新命名当前所谓 “occupancy”

## 当前真实定义

V2 中的 \(o\) 已不再是：

> “jammer 占据多少比例的 time-frequency cells”

也不是 threshold-based occupied-bin ratio。

当前真实定义是：

1. 对 tracked target 和 tracked jammer 分别计算 STFT power；
2. 按 target power 从高到低排序；
3. 取 target 累积 99% 能量对应的 support \(S\)；
4. 计算 jammer 总功率中有多少比例落入该 target support：

\[
o =
rac{
\sum_{(f,t)\in S} P_j(f,t)
}{
\sum_{f,t} P_j(f,t)
}.
\]

因此 \(o\) 的真实物理意义是：

> **jammer energy overlapping the target-dominant time–frequency support**

而不是通常意义上的 spectral occupancy。

---

## 当前风险

全文仍大量使用：

- jammer spectral occupancy
- occupancy–SIR envelope
- break-even occupancy
- spectrally occupied corner
- occupancy coefficient
- occupancy proxy

这会让 Reviewer 误以为：

> \(o=0.8\) 表示 jammer 占据了 80% 的时频平面。

但当前真实含义是：

> 80% 的 jammer STFT power 落在 target 的 99%-energy support 内。

这两个量不同。

---

## 正式推荐术语

### 首选

> **target-support jammer overlap**

符号继续保留为 \(o\)。

首次定义：

> “We define the target-support jammer overlap \(o\) as the fraction of total jammer STFT power falling inside the target’s 99%-energy time–frequency support.”

这是最准确的版本。

---

## 备选

> **target-support interference overlap**

如果希望避免 “jammer” 一词重复。

---

## 不推荐继续作为主术语

> jammer spectral occupancy

除非重新回到真正的 occupied-bin definition。

---

# 2. 全文术语替换清单

## 标题

当前：

> **A Condition-Indexed Operating Envelope and Representation-Repair Study for Automatic Modulation Classification under Structured Interference**

### 建议

**标题不需要改。**

标题没有直接写 occupancy，因此可保持。

---

## Abstract

当前：

> “A two-variable occupancy–SIR envelope...”

改为：

> **“A two-variable target-support-overlap–SIR envelope...”**

或者为避免复合词过长：

> **“A two-variable overlap–SIR envelope...”**

并在前一句或首次出现说明：

> “where overlap denotes the fraction of jammer energy falling inside the target’s dominant time–frequency support.”

---

## Contribution 2

当前：

> “a low-dimensional occupancy–SIR empirical envelope...”

改为：

> **“a low-dimensional target-support-overlap–SIR empirical envelope...”**

---

## Section V-C 标题

当前：

> `Empirical Occupancy–SIR Operating Envelope`

改为：

> **`Empirical Target-Support-Overlap–SIR Operating Envelope`**

如果标题过长，可用：

> **`Empirical Overlap–SIR Operating Envelope`**

正文第一次定义完整术语。

---

## Table VI

当前：

> `EXPLORATORY BREAK-EVEN OCCUPANCY UNDER SIMULATOR-TRACKED JAMMER OCCUPANCY`

改为：

> **`EXPLORATORY BREAK-EVEN TARGET-SUPPORT JAMMER OVERLAP`**

或：

> **`EXPLORATORY BREAK-EVEN OVERLAP UNDER SIMULATOR-TRACKED COMPONENTS`**

---

## Fig. 4 x 轴

当前：

> `jammer spectral occupancy`

改为：

> **`target-support jammer overlap`**

如果空间不足：

> **`jammer overlap with target support`**

---

## Discussion

将：

> “high-occupancy conditions”

改为：

> **“high-overlap conditions”**

将：

> “occupancy organizes the rank exchange”

改为：

> **“target-support jammer overlap organizes the observed rank exchange”**

---

## Limitations

将：

> “occupancy is simulator-derived”

改为：

> **“the target-support overlap is simulator-derived because it requires separately tracked target and jammer components.”**

这比原句更准确。

---

# 3. P0：解释 envelope analysis STFT 与 model STFT 不同

## 当前事实

### 模型前端

- NFFT = 64
- hop = 16
- Hann window length = 64
- 64×29 lattice

### overlap covariate

- Hann-windowed 128-point STFT
- hop = 32

这是两个不同的 lattice。

---

## 风险

Reviewer 可能问：

> 为什么 physical covariate 不在与 VIMD 相同的 STFT lattice 上计算？

如果不解释，容易被认为：

- 分析口径后改；
- 为了更漂亮的 envelope 选择另一个 STFT；
- 或存在 hidden tuning。

---

## 推荐正文补充

在 Section V-C overlap 定义后加入：

> **“The overlap covariate is computed on a fixed analysis STFT (128-point Hann window, hop 32) that is independent of the 64-point VIMD front-end lattice. This analysis lattice was frozen for the artifact-level physical audit and is applied identically to all models and splits; it is not an input feature of VIMD-Net and is not tuned to any model’s predictions.”**

### 如果真实情况不是 preregistered/frozen

不要写 “frozen” 或 “not tuned”。

改为：

> **“For consistency across all post-hoc physical analyses, the overlap covariate is computed on a common 128-point analysis STFT that is independent of the model front end.”**

必须按真实 artifact provenance 选择。

---

# 4. P0：修正 route-orthogonality regularizer 的逻辑矛盾

## 当前句子

Implementation：

> “A small route-orthogonality regularizer with weight 0.01 is also active in the complete configuration; it is omitted from (7) for readability and is not part of any reported contrast.”

## 问题

Table IV 已明确：

- A0：No
- A5：Yes

所以它事实上属于：

> A5 vs A0 whole-configuration contrast

只是没有独立 component contrast。

---

## 必改

改为：

> **“A small route-orthogonality regularizer with weight 0.01 is also active in the complete configuration and is omitted from (7) for readability. It is part of the A5 whole-configuration bundle but is not isolated in a dedicated component contrast.”**

这样与 Table IV 完全一致。

---

# 5. P0：修正 Abstract 中 constant predictor 的歧义

## 当前句子

> “...RMSE 6.20 percentage points against 8.40 percentage points for a constant predictor fitted on the same cells...”

## 风险

“same cells”容易被理解成：

> constant predictor 是在 held-out cells 上拟合的。

实际不是。

真实流程是：

- constant = 32 个 fitting cells 的 mean；
- 然后 unchanged applied to 81 held-regime cells。

---

## 推荐改法

> **“...with RMSE 6.20 percentage points versus 8.40 percentage points for a constant predictor estimated from the fitting cells (\(R^2_{m skill}=0.46\)); the validated skill is cross-regime transport of gain magnitude rather than sign classification.”**

这是最清楚的版本。

---

# 6. P1：Fig. 4 caption 中 “training cells” 改成 “fitting cells”

当前 caption：

> “relative to a constant predictor fitted on the training cells”

容易与神经网络训练样本混淆。

## 建议

改为：

> **“relative to a constant predictor estimated from the envelope-fitting cells”**

完整：

> “Magnitude prediction is evaluated by RMSE relative to a constant predictor estimated from the envelope-fitting cells...”

---

# 7. P1：将 “spectrally occupied corner” 改成 “high-overlap corner”

当前多处写：

> severe, spectrally occupied corner

在新 overlap 定义下不够准确。

## 推荐

Abstract / Introduction：

> **“a severe, high-overlap corner”**

或者：

> **“a severe-interference corner with strong jammer overlap on the target support”**

Results：

> “with A5 leading only in the severe, high-overlap corner.”

Discussion：

> “the compact spectral configuration becomes relatively most competitive when interference is severe and strongly overlaps the target-dominant time–frequency support.”

---

# 8. P1：改写 Eq. (8) 解释

当前：

\[
\Delta_b = eta_{0,b}+eta_{1,b}o+eta_{2,b}\mathrm{SIR}
\]

建议紧接着写：

> “Here \(o\) is the target-support jammer overlap, not a generic occupied-bandwidth fraction.”

这样 Reviewer 不会把它和传统 spectral occupancy 混淆。

---

# 9. P1：重新解释 break-even 数值

原来：

> break-even occupancy

容易被理解成：

> jammer 要覆盖多少带宽。

改成 overlap 后，应解释：

> break-even overlap = 在某个 SIR 下，jammer 能量落入 target dominant support 的比例达到何值时，A5 与 comparator 的预测 gain 交叉为 0。

推荐正文：

> **“The break-even value therefore denotes a simulator-domain jammer-to-target-support overlap threshold, not an occupied-bandwidth threshold.”**

---

# 10. P1：Future Work 中 occupancy proxy 也要改

当前：

> “estimate a receiver-observable occupancy proxy (for example occupied-bandwidth or spectral-kurtosis estimators)”

如果主变量现在是 target-support overlap，那么简单 occupied bandwidth 并不严格是它的 proxy。

## 推荐改为

> **“estimate a receiver-observable proxy for jammer-to-target spectral overlap, potentially using occupied-bandwidth, spectral-kurtosis, interference-localization, or learned overlap estimators.”**

这样逻辑一致。

---

# 11. P1：External-validity 段同步修改

当前：

> “public benchmarks ... do not expose a tracked occupancy covariate.”

改为：

> **“public benchmarks generally do not expose separately tracked target and jammer components required to compute the present target-support overlap covariate.”**

这比 “no occupancy covariate” 更精确。

---

# 12. P1：结论统一术语

当前：

> “A low-dimensional occupancy–SIR envelope transports the gain level...”

改为：

> **“A low-dimensional target-support-overlap–SIR envelope transports the gain level across regime cells excluded from fitting...”**

或者更简洁：

> **“A low-dimensional overlap–SIR envelope...”**

前文已定义即可。

---

# 13. P1：Index Terms 可调整

当前：

- Automatic modulation classification
- structured interference
- vehicular Doppler
- operating envelope
- condition-dependent model comparison
- representation repair
- compact neural networks

### 建议保持

无需加入 “occupancy”。

如果想体现物理变量，可增加：

> interference overlap

但不是必须。

---

# 14. P1：重新检查 Related Work 中 novelty 是否还与新术语一致

当前 novelty 仍然成立：

> condition-dependent rank exchange + empirical boundary + repair

不受 occupancy 改名影响。

无需重新搜文献或改主线。

---

# 15. P1：A7 0.21 pp 可保持当前写法

当前：

> “nominally above A5 by 0.21 pp... well below the design resolution...”

已经足够谨慎。

如果没有现成 paired CI，不建议为了这一项新增统计。

---

# 16. P1：optimizer consistency 已基本解决

V2 当前：

> “all fits use AdamW...”
> “optimizer settings and schedule are shared across every model”

Comparator fairness 已不再出现 architecture-specific optimizer exception。

因此这一项可以视为完成。

---

# 17. P1：5.9 GHz / Doppler / coherence time 已完成

当前已经写：

- 5.9 GHz
- training 0–150 km/h
- held 180/250 km/h
- 820 / 1367 Hz
- \(T_cpprox309\mu s\)
- 512 μs analysis window

无需继续扩展。

投稿前只做一次数字机械复算即可。

---

# 18. 本轮不建议再做的新实验

不要再增加：

- RadioML 新实验
- 新 baseline
- 新 capacity tier
- 新 sidecar
- 新 occupancy/overlap estimator
- OTA experiment
- 新 preregistration

原因：

当前最重要的是保证：

> **定义、命名、证据等级、图表和结论完全一致。**

而不是继续扩大实验面。

---

# 19. 最终推荐贡献表述

## Contribution 1 — Condition-dependent rank exchange

> We expose a condition-dependent rank exchange between compact spectral and high-capacity I/Q representations over the complete structured-interference SNR–SIR plane, rather than reporting a single marginal ranking.

## Contribution 2 — Cross-regime overlap–SIR envelope

> We construct a low-dimensional target-support-overlap–SIR empirical envelope whose cross-regime gain-transport skill is evaluated on regime cells excluded from fitting against an explicit fit-domain constant baseline, with a bounded and non-causal interpretation.

## Contribution 3 — Evidence-bounded severe corner

> We separate the original sealed whole-configuration contrast from post-hoc severe-interference evidence using paired statistics, multiplicity control, seed consistency, and full-region visualization.

## Contribution 4 — Prospective representation repair

> We turn a failed clean-retention gate into a prospective diagnosis-and-repair sequence in which probe, coverage control, negative control, I/Q sidecar, and capacity ladder collectively support a repairable representation-information bottleneck rather than a pure exposure or parameter-count explanation.

---

# 20. 推荐摘要关键段最终版

> “A two-variable target-support-overlap–SIR envelope, fitted on in-distribution and hard-interference cells only, predicts the gain against IQFormer-inspired on held-out regime cells with RMSE 6.20 percentage points versus 8.40 percentage points for a constant predictor estimated from the fitting cells (\(R^2_{m skill}=0.46\)); the validated skill is cross-regime transport of gain magnitude rather than sign classification.”

如果想减少术语长度，可首次定义后写：

> “overlap–SIR envelope”

---

# 21. 推荐 Discussion 核心句

原：

> “the compact spectral configuration becomes relatively most competitive when interference is severe and sufficiently spectrally occupied.”

改为：

> **“Within the present simulator campaign, the compact spectral configuration becomes relatively most competitive when interference is severe and jammer energy strongly overlaps the target-dominant time–frequency support.”**

这比 “spectrally occupied” 物理上准确得多。

---

# 22. 推荐 Limitations 关键句

> **“The target-support overlap covariate is simulator-derived because it requires separately tracked target and jammer components and is not directly observable at an uncooperative receiver.”**

---

# 23. 推荐 Future Work 关键句

> **“A future prospective campaign should estimate a receiver-observable proxy for jammer-to-target time–frequency overlap and test whether the resulting overlap–SIR envelope retains cross-regime transport skill without simulator component bookkeeping.”**

---

# 24. 最终优先级

| 优先级 | 修改项 | 是否必须 |
|---|---|---:|
| P0 | occupancy → target-support jammer overlap | 必须 |
| P0 | 解释 128/32 analysis STFT 与 64/16 model STFT 的关系 | 必须 |
| P0 | 修正 regularizer “not part of any reported contrast” | 必须 |
| P0 | Abstract constant predictor “same cells” 歧义 | 必须 |
| P1 | Fig. 4 caption training cells → fitting cells | 强烈建议 |
| P1 | spectrally occupied → high-overlap / overlap target support | 强烈建议 |
| P1 | break-even occupancy → break-even overlap | 强烈建议 |
| P1 | deployment proxy 段同步改为 overlap proxy | 强烈建议 |
| P1 | conclusion / limitations / discussion 术语统一 | 必须 |
| P1 | 数字与引用机械 QA | 必须 |

---

# 25. 修改完成判据

完成 V1.3 后应满足：

- [ ] 全文不再把 Eq. (9) 的量称为 generic spectral occupancy；
- [ ] \(o\) 首次定义明确写为 target-support jammer overlap；
- [ ] Fig. 4 x-axis 同步；
- [ ] Table VI 同步；
- [ ] Abstract / Contribution / Discussion / Conclusion 同步；
- [ ] 128-point analysis STFT 与 64-point model STFT 关系解释清楚；
- [ ] regularizer 逻辑矛盾修正；
- [ ] constant predictor 明确来自 fitting cells；
- [ ] “spectrally occupied” 统一替换为符合真实定义的 overlap 表述；
- [ ] deployment limitation 改成 component-tracking limitation；
- [ ] 不增加新实验；
- [ ] 不再改 contribution hierarchy；
- [ ] 最终 PDF 进入纯 QA 阶段。

---

# 26. V1.3 完成后的冻结原则

完成上述修改后：

> **技术内容冻结。**

之后只允许：

- typo
- grammar
- IEEE style
- figure label
- formula reference
- reference metadata
- author metadata
- acknowledgment / funding / disclosure
- submission formatting

不再允许：

- 新 claim
- 新实验
- 新 headline
- 新 evidence class
- 新 architecture narrative
- 新 causal interpretation

---

# 27. 最终一句话定位

> **This paper studies when compact spectral and high-capacity I/Q representations exchange rank under structured interference, describes that exchange using a simulator-domain jammer-to-target-support overlap–SIR envelope, and prospectively diagnoses and repairs a revealed representation boundary.**

---

**版本：V1.3 — 技术冻结前最后一轮改稿。**
