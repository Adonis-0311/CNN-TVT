# TVT 论题重定位与主张阶梯

> 版本：v1 ｜ 2026-08-12  
> 面向对象：论文写作 agent 与人类作者  
> 与 `analysis_zero_compute/RESULTS.md` 的分工：**RESULTS.md 是内部证据账本，措辞钝且完整；本文件是对外稿件的叙事与表述规范。** 两者的数字必须一致，语气不必一致。内部账本对自己狠，对外稿件对证据自信——这是正常的科研写作分工，不是两套事实。

---

## 0. 核心判断

原叙事（"提出的方法在带干扰的 AMC 上更鲁棒"）与证据不匹配，所以显得处处退让。证据实际支持的是一个**更窄但更硬**的命题：

> **在结构化强干扰（低 SIR、高频谱占用）条件下，39.5 k 参数的物理引导紧凑前端优于 9–10 倍规模的高容量基线；随着干扰强度下降、占用率降低，这一优势系统性消失。增益与劣势沿同一条物理轴排列，因此本文给出的不是一个"更好的模型"，而是一条**被定量刻画的适用边界**。**

一旦论题从"更好"改成"在哪里更好、为什么、边界在哪"，此前所有"不利结果"都变成**支持性证据**：它们证明作者知道自己的方法什么时候不该用，而这条边界是可预测的。这是审稿人愿意接受的形态，也不需要隐瞒任何数字。

---

## 0.5 深度分析：五条候选叙事的实证比较

叙事不是文风选择，是可检验的假设。五条候选各自都能写，但只有一条被数据选中。

| # | 候选叙事 | 中心主张 | 用掉的证据比例 | 攻击面 | 判定 |
|---|---|---|---|---|---|
| N1 | **占用率索引的工作区**（被选中） | 一条两变量物理律同时解释增益与退化，且可外推预测 | ~90%（含全部"不利"结果） | 事后分层（已 Holm 校正）、占用率与 SIR 共变（已声明） | **采纳为主线** |
| N2 | 紧凑效率前沿（Pareto 优先） | 同预算下最优 | ~35% | 同价位 A7 名义更高；前沿位置依赖成本轴选择；"小而弱"易被贬为工程琐事 | 降为 C6 支撑 |
| N3 | 互补性/可融合组件 | 即使更弱也贡献独立信息 | ~25% | 把自己定位成别人的附件，削弱独立价值 | 降为 C5 支撑 |
| N4 | 基准条件迁移研究 | 缓存条件覆盖导致类选择性迁移失败 | ~30% | 变成基准批评论文，方法贡献被稀释；TVT 口味偏工程 | 降为 C7 独立小节 |
| N5 | 严重干扰专用模型 | 低 SIR 角落更强 | ~45% | 只讲赢的角落＝选择性报告；无机制解释 | 并入 N1 的一个推论 |

### 决定性证据：包络律可外推预测（A14）

把全部 7 个测试 regime 切成 113 个 (split × SIR × 占用率四分位) 单元，用**两个物理变量**回归相对基线的增益：

```
gain(A5 − IQFormer) [pp] = −16.60 + 11.39 · occupancy − 0.666 · SIR(dB)
gain(A5 − MCLDNN)   [pp] = −14.28 + 13.93 · occupancy − 0.504 · SIR(dB)
```

两个系数的 bootstrap 95% CI 均不含零（占用率 [+7.88, +14.64]，SIR [−0.82, −0.50]）。关键在于**留出验证**：只用 `id_test` 与 `hard_interference` 的 32 个单元拟合，再去预测它从未见过的 81 个单元——包括 held-out 干扰族、未见速度、未见信道、combined OOD 与无干扰条件：

| 留出 regime | 实际增益 | 律预测 |
|---|---:|---:|
| unseen_jammer | −16.05 pp | −13.66 pp |
| combined_ood | −16.47 pp | −13.65 pp |
| unseen_speed | −12.38 pp | −12.27 pp |
| heldout_channel | −11.31 pp | −12.13 pp |
| clean_retention | −19.14 pp | −17.08 pp |

留出单元符号一致率 **97.5%**（vs MCLDNN 95.1%），Pearson r = 0.694，RMSE 6.2 pp。

**这句话就是全文的核心**：论文此前分别报告为五处"泛化退化"的结果，其实是同一条律在低占用率处的取值。方法没有五个失败，它有一条边界。

### 可引用的设计规则（break-even 占用率）

令 gain = 0 解出所需占用率：

| SIR (dB) | 超越 IQFormer 所需占用率 | 超越 MCLDNN 所需占用率 |
|---:|---:|---:|
| −15 | 0.58 | 0.48 |
| −10 | 0.87 | 0.66 |
| −5 | 1.17（不可达） | 0.84 |
| 0 | 1.46（不可达） | 1.03（不可达） |

即："**每放松 1 dB 干扰强度，紧凑模型需要多约 0.04–0.06 的频谱占用率才能维持优势；SIR 高于约 −7 dB 时不存在可达的占用率。**" 这是一条工程师可以直接用来选型的规则，也是 TVT 读者最买账的产出形态。

### 为什么 N1 让"不利结果"变成资产

| 原"不利结果" | 在 N1 下的角色 |
|---|---|
| unseen_jammer −4.14 pp | held-out 族占用率中位数 0.134，律预测为负——**验证律的外推能力** |
| combined_ood −4.12 pp | 同上 |
| clean_retention −9.69 pp | 占用率 = 0，律的极端点；**预测值与实测相差 2 pp** |
| 绝对精度落后强基线 | 律的高 SIR / 低占用区；交叉点由律给出，可计算 |
| occupancy 预注册方向被否 | 律的斜率为**正且显著**，方向被数据改正——从"假设失败"变成"定量律" |
| 三项消融只过一项 | 律描述的是整体架构效应，与"组件个体贡献 < 1.4 pp"的有界结论自洽 |

六项里有五项从减分变成了律的支持证据，第六项（有界结论）与之自洽。这就是"改变叙事范围让证据变强"的正确技术含义：不是换措辞，是找到一个能同时解释成功与失败的更小的命题。

---

## 1. 建议题名与一句话贡献

**题名候选**

1. *An Occupancy-Indexed Operating Envelope for Compact Physics-Guided AMC under Structured Interference*（推荐：把可外推的律放在题名里）
2. *Where Physics-Guided Disentanglement Pays Off: A 39.5k-Parameter Front End for Severe Co-Channel Interference in Vehicular AMC*
3. *Predicting When a Small Model Wins: A Two-Variable Envelope for Compact Spectral AMC against High-Capacity Baselines*

**一句话贡献**

> We show that the advantage of a compact, physics-guided time–frequency disentanglement front end over high-capacity AMC baselines is governed by a two-variable law in jammer spectral occupancy and SIR: fitted on two evaluation regimes, the law predicts the sign of the advantage in 97.5% of cells across five regimes it never saw, including held-out jammer families and the jammer-free condition. Within the region the law identifies, the 39.5k-parameter model exceeds 355k–406k-parameter baselines by about 10 macro-F1 points; outside it, the same law states the price and the break-even occupancy at which it is paid.

---

## 2. 主张阶梯（按论文中出现顺序）

| # | 主张 | 证据 | 证据级别 | 强度 |
|---|---|---|---|---|
| **C0** | **增益与退化由一条两变量物理律统一描述：`gain = β0 + β1·occupancy + β2·SIR`；仅用 ID 与 hard interference 拟合，即可在 5 个未见 regime 的 81 个单元上以 97.5% 符号一致率外推预测** | `a14_envelope_fit.csv`、`a14_envelope_holdout.csv`、`figA14` | exploratory | **强**：系数 CI 均不含零；留出验证跨越 held-out 干扰族与无干扰条件 |
| C1 | 在 SIR = −15 dB 的严重干扰下，紧凑模型以 39.5 k 参数超过 MCLDNN（+9.74 pp）与 IQFormer-inspired（+10.46 pp） | `a1_paired_by_level.csv`、`a11_*` | exploratory，已做 Holm 校正 | **强**：64 格中 12 格过 Holm，**全部落在同一 SIR 行**；10/10 seed 同号；符号翻转检验 p=0.0018（10 seed 的下限）；4 种 bootstrap 变体区间均严格为正 |
| C2 | 完整方法相对共享 backbone 的主效应为 +4.571 pp，联合同时 95% CI [+3.655, +5.488] | sealed `ablation_paired_statistics.csv` | **sealed confirmatory** | **强**：设计分辨力 1.31 pp，效应为其 3.5 倍 |
| C3 | 该增益来自架构整体而非单个组件：两项组件级干预的个体贡献在联合 95% 置信下被**限定在 1.4 pp 以内** | `a12_effect_resolution_ablation.csv` | sealed，重述为有界结论 | **中强**：是有界结论，不是空结论 |
| C4 | 增益与退化沿同一物理轴排列：A5−CSSL 随占用率五分位单调上升（−7.6 → +23.3 pp，ρ=+1.0），退化集中在低占用的窄带/脉冲型 held-out 干扰 | `a6_occupancy_bins.csv`、`a2_strata_paired.csv` | exploratory | **强**：单调、跨分层一致 |
| C5 | 即使在其落后的场景，其决策仍携带强基线没有的信息：A5+IQFormer 融合 +4.71 pp，而同架构双种子集成对照仅 +1.49 pp | `a5_fusion.csv` | exploratory，含对照 | **强**：有排除"集成效应"的对照 |
| C6 | 精度—成本前沿：VIMD 家族在 39.5 k 档占据前沿；30% 覆盖率下选择性正确率 94.4%，与 IQFormer 相差 0.5 pp 而参数少 9 倍 | `a8_pareto.csv`、`a4_selective_points.csv` | sealed descriptive + exploratory | **中强** |
| C7 | 基准性发现：按训练实际使用的两个 view 重算后，4/10 类见过无干扰训练窗，其余 6/10 为条件 OOD；迁移失败集中在"紧凑谱域 × 未覆盖的星座阶数类"（27/27 单元），IQ 域高容量模型为 1/9 | `a9_*`、`a13_*` | exploratory | **强**：覆盖与表示域分开报告；修正了旧 2/10 口径 |

C1、C4、C5、C7 是新增的，全部来自已封存预测，无新仿真。

---

## 3. 每个"不利结果"的正当强表述

原则：**不要用"失败/不足/遗憾"作主语。用"边界/条件/界限/未被本设计分辨"作主语。** 事实一个不少，主语全部换掉。

### 3.1 绝对精度不敌 MCLDNN / IQFormer

- ❌ 弱表述："我们的方法在绝对精度上不如强基线。"
- ✅ 强表述：
  > At matched interference severity the two capacity regimes exchange rank: above SIR = −10 dB the 355k–406k-parameter baselines lead, while at SIR = −15 dB the 39.5k-parameter front end leads both by about 10 macro-F1 points. We therefore report an operating envelope rather than a uniform ranking.
- 配套：SNR×SIR 工作图放正文，交叉点明确标出。**不要把 marginal 平均值当主结果表的第一列**——那是把两个不同工作区强行平均。

### 3.2 三项确认性消融只过一项

- ❌ 弱表述："预注册的确认性家族门未通过。"（照实说，但不要作为小节标题）
- ✅ 强表述：
  > The primary method effect is confirmed with a simultaneous 95% interval of [+3.66, +5.49] pp, 3.5× the design's 1.31 pp resolution. The two component-level interventions are *bounded*, not unmeasured: their individual contributions lie within ±1.4 pp under the same joint interval. The family gate, which required all three contrasts to clear zero simultaneously, therefore did not pass — the effect is carried by the architecture as a whole rather than by any single component, which is the more useful engineering conclusion at this parameter budget.
- 位置：结果章正文，与 C2 同段。**必须出现 "the family gate did not pass"** 一句——它换来的是后面所有统计主张的可信度。

### 3.3 clean-retention −9.69 pp

- ❌ 弱表述："方法在干净信号上有显著退化，是主要代价。"
- ✅ 强表述：
  > The clean-retention split evaluates all ten modulations without interference, whereas the two training views expose jammer-free windows for only four classes; for the remaining six this contrast measures out-of-distribution condition transfer rather than clean-signal accuracy. Among uncovered constellation-order cells, transfer failure occurs in 27/27 compact-spectral cells versus 1/9 IQ-domain high-capacity cells, while envelope-distinctive cells transfer in both families. The preregistered clean-retention gate did not pass; we report the benchmark property and representation boundary that produce it.
- 这一段把最大的减分项变成了 C7 这个**独立贡献**。同时保留门未通过的陈述。

### 3.4 unseen_jammer / combined_ood 退化

- ✅ 强表述：
  > Degradation concentrates on the two held-out jammer families, whose median spectral occupancy (0.134) is less than half that of the trained families (0.299). It is not a separate failure mode: the envelope law fitted on in-distribution and hard-interference cells predicts −13.7 pp on unseen-jammer cells against an observed −16.1 pp, and −17.1 pp on the jammer-free condition against an observed −19.1 pp. Gains and losses are two evaluations of the same law.
- 这一段让退化**支持**机制主张，而不是削弱它。数字来自 `a14_envelope_holdout.csv`。

### 3.5 occupancy 机制方向与预注册相反

- ❌ 弱表述："预注册机制假设未获支持。"
- ✅ 强表述：
  > We preregistered a non-increasing relation between jammer occupancy and gain; the sealed test rejected that direction (rho = +0.714, one-sided p = 0.949). The data instead support the opposite and stronger relation: the advantage grows monotonically with occupancy (quintile-level rho = +1.0 against the CSSL reference, from −7.6 pp to +23.3 pp). We report the corrected direction as an empirical finding, and note that occupancy co-varies with SIR in this design (rho = −0.8), so the relation is descriptive rather than causal.
- 预注册方向错了本身不减分；**藏起来才减分**。改正方向并给出更强的关系是加分项。

### 3.6 同价位 A7 名义高于 A5

- ✅ 强表述：
  > Within the 39.5k-parameter tier the residual-free variant scores nominally 0.21 pp higher; this is far below the 1.31 pp resolution of the design and within seed dispersion, consistent with the bounded component-level effects reported above. We therefore present the frontier position as a property of the VIMD family at this budget.
- 主语从"我们的具体模型"上移到"该家族在该预算档"，既准确又不自伤。

---

## 4. 论文结构建议（实现上述阶梯）

1. **Introduction** — 问题设定为"车载结构化强干扰下的算力受限 AMC"，明确提出"什么时候紧凑物理引导表示优于高容量模型"这一问题。不承诺全面领先。
2. **Related work** — 按"IQ 域高容量" vs "变换域紧凑" 两条线组织，指出既有工作缺少**跨干扰强度的工作区刻画**。
3. **Method** — 三路分配、teacher、多目标；强调参数预算与物理对齐。
4. **Experimental protocol** — 冻结协议、12 模型 ×10 seed ×11 split、配对分层 bootstrap、预注册与其结果（含未通过项）；此处一次性把诚实性交代完，后文不再反复致歉。
5. **Results**
   - 5.1 主效应与有界组件（C2、C3）
   - 5.2 工作区：SNR×SIR 图与严重干扰角（C1）
   - 5.3 **包络律与外推验证**（C0、C4）——本节是全文的转折点，把 5.2 的观察升级为可预测的律，并给出 break-even 占用率表
   - 5.4 互补性与融合（C5）
   - 5.5 成本前沿与选择性预测（C6）
   - 5.6 条件迁移基准性质（C7）
6. **Discussion** — 何时该用、何时不该用；对部署的含义（仅仿真口径）。
7. **Limitations** — 集中一节，简洁、事实性、不道歉：仿真边界、门未通过项、事后分层的性质、占用率与 SIR 共变、A7 同价位。
8. **Reproducibility** — 证据链、哈希、复现包。

---

## 5. 必须保留的披露（及其收益）

| 必须写 | 写在哪 | 为什么它让稿件更强而不是更弱 |
|---|---|---|
| 确认性家族门未通过 | §4 协议 + §5.1 | 换来 C2 的"预注册"身份；这是全文最硬的一个数字 |
| clean-retention 门未通过 | §4 + §5.6 | 换来 C7 这个独立贡献的可信度 |
| SIR 分层是事后的 | §5.2 首句 | 已做 Holm 校正与 10/10 seed 一致性；主动声明 + 校正 = 无懈可击；被审稿人指出 = 致命 |
| occupancy 预注册方向被否 | §5.3 | 证明作者不是在事后编故事，反而使 ρ=+1.0 的结果可信 |
| 绝对精度落后区间 | §5.2 图与正文 | 工作区叙事的另一半；缺了它，交叉点没有意义 |
| 仅仿真 | 全文术语 | 避免最容易触发的 desk 级质疑 |

**判断标准**：一处披露如果换来了一个更强主张的可信度，它就是资产；如果只是自责而不换来任何东西，删掉那句自责，保留事实。本文件所有 ✅ 表述都遵循这条。

---

## 6. 不可做（会直接导致撤稿级风险）

1. 把事后 SIR 分层写成预注册分析；
2. 把 `submission_unlocked=false` 的门写成通过，或不写；
3. 删除 MCLDNN / IQFormer 或失败消融；
4. 只报 SIR=−15 而不给全工作区图（选择性报告）；
5. 把 occupancy 反向结果说成原假设成立；
6. 把 Tier-2 探索性结果与 V4 正式结果同表并列；
7. 任何 measured / onboard / field / operational 措辞。

这些不是道德条款，而是风险条款：TVT 审稿人复核预注册与多重性是常规动作，上述任一被发现都会使全文可信度归零，而它们换来的收益为零——因为按 §3 的表述，这些事实本来就不减分。

---

## 7. 待补强项对主张的增益（GPU 部分，见工作分解 WP3/WP4）

| 工作包 | 若结果为正 | 若结果为负 | 对稿件的净影响 |
|---|---|---|---|
| WP4 容量阶梯 | "差距可由容量解释"，紧凑性定位更纯粹 | "差距非容量所致"，说明是表示域差异，C1 的架构解释更强 | **两种结果都加分** |
| WP3 条件覆盖补救 | clean 迁移可修复，C7 从"性质"升级为"性质＋解法" | 修复无效，则 C7 的表示层解释被加强 | **两种结果都加分** |
| WP2 探针 v2 | 信息在表示中 → 训练层可解 | 信息不在表示中 → 前端层限制，机制解释更硬 | **两种结果都加分** |

这三项的共同特点：**设计成"无论正负都产出结论"的实验**。这才是把不确定性变成强度的正确做法。

### 7.1 2026-08-12 执行更新

- WP2 v2 已完成：两种设计 × A5/A0/CSSL/MCLDNN/IQFormer × 3 seeds 均已落盘。A5 的高阶类 MLP 探针在 6 个设计—seed 单元中 5 个低于 0.25，仅 source-disjoint CV seed 43 为 0.2558；按"两设计一致"的预注册要求，记为**阈值边界、证据偏向 representation-limited**。
- CSSL/MCLDNN/IQFormer 在两种设计的 18/18 个单元全部高于 0.25，说明探针本身有分辨力。
- 修复顺序改为：完整冒烟 → 条件覆盖 F1 → ≤80 k mixture-IQ sidecar 前端臂 → 容量阶梯。Tier-2 执行器的合成契约、归一化、梯度和参数预算测试已通过，但完整缓存 1-epoch GPU 冒烟尚未完成，不得直接进入多 seed 扩大训练。
- 2026-08-14：完整冒烟与条件覆盖 F1 均已完成。F1 在 clean 上仅 +0.15 pp [−0.31, +0.63]，未达 +6 pp 门；同时 hard 非劣、严重干扰角和相对 A0 的六轴保持门均通过。这不是"训练修好了"，而是**条件覆盖单因素解释被排除**；下一个必需臂为 mixture-IQ sidecar 前端修复。
- 2026-08-15：H2-F2 mixture-IQ sidecar 已完成 10 seeds 并通过 H2a–H2d 全部门。其以 46,794 参数将 clean 提升 +11.77 pp [11.06, 12.49]，将 hard 提升 +4.37 pp [3.44, 5.31]，并在 SIR=−15 dB 保持相对 MCLDNN/IQFormer 的显著正优势。因此可以把 clean 边界从"不明原因的代价"收口为**紧凑谱域缺少星座信息的可修复表示缺口**；但它仍为 Tier-2 exploratory，不改写封存确认性结论。
- 2026-08-17：容量阶梯已闭环。M=99,596、L=233,244 参数；两档均提高总体 hard 精度，但 L 仍显著低于 IQFormer −2.84 pp [−4.30, −1.31]，因此禁止把总体差距归因于容量不足。同时 L 在 SIR=−15 dB 上相对 IQFormer 为 +14.07 pp [9.55, 18.51]，说明严重干扰角优势不是小模型欠拟合产物。强表述应为：**容量改善平均拟合，表示域/架构决定工作区边界。**
