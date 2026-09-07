# 审稿意见书（Minor Revision / 小修）

**稿件**：A Condition-Indexed Operating Envelope and Representation-Repair Study for Automatic Modulation Classification under Structured Interference（V3）
**目标期刊**：IEEE Transactions on Vehicular Technology
**审稿日期**：2026-08-18
**推荐意见**：**Minor Revision（小修）**，但小修清单较长，且其中 3 项需要重跑（成本低，均可复用已冻结的 artifact）

---

## 0. 总体判断

### 0.1 分项评分

| 维度 | 评分 (1–10) | 说明 |
|---|---|---|
| 新颖性 / 定位 | 7.0 | "条件索引的秩交换 + 边界修复" 是一个真正未被 AMC 社区正面处理的问题，定位诚实 |
| 技术正确性 | 7.5 | 抽查的数值关系基本自洽（见 §1），但存在若干确凿的内部矛盾 |
| 实验充分性 | 7.0 | 120 fits / 10 seeds / 11 splits 的规模在 TVT 属于上游；但纯仿真、无公开数据集锚点 |
| 统计严谨性 | **8.5** | 密封确认族、多重性控制、失败门公开报告 —— 显著高于 AMC 领域均值 |
| 写作与图表 | 6.0 | 最弱项。术语超载、图文数字不一致、纯差值无绝对值 |
| 期刊契合度 (TVT) | 6.5 | "vehicular" 论证偏薄，且元方法论包袱重于典型 TVT 论文 |
| **综合** | **7.2 / 10** | |

**当前状态录用概率估计：50–60%**
**按本意见完成修改后：68–78%**

### 0.2 核心判断

这篇稿子最不寻常也最有价值的地方，是它**主动报告了两个未通过的预注册门**（confirmatory family gate、clean-retention gate），并把失败门转成了诊断-修复链。这在 AMC 领域几乎见不到，编辑会认可。

但正因为如此，稿件的**举证责任反而更重**：当作者反复强调"我的证据分级很干净"时，任何一处图文数字对不上，都会被审稿人放大成"分级本身不可信"。目前稿件里至少有 **3 处确凿的内部矛盾**（A1–A3），必须在返修中全部消除，否则元方法论的信誉资本会被反噬。

第二个战略问题：**主结果是"我在边际上输给两个大模型"**（0.4406 vs 0.4599 / 0.4872），赢的只是一个 post-hoc 的角落。作者已经很诚实地写了，但摘要里没有出现 0.4406/0.4872 这组数字。编辑可能会认为摘要有选择性。建议主动补上（见 D1）——这反而会增强而非削弱说服力。

---

## 1. 已通过的自洽性核查（供作者放心）

以下关系我逐一验算，**全部正确**，返修时无需改动：

| 项 | 核查 |
|---|---|
| 教师掩码归一性 | `[q_s−q_j]₊ + [q_j−q_s]₊ + q_u + 2min(q_s,q_j) = \|q_s−q_j\| + 2min(q_s,q_j) + q_u = q_s+q_j+q_u = 1` ✅ |
| STFT 格点 64×29 | `floor((512−64)/16)+1 = 29`；1 MHz/64 = 15.625 kHz ✅ |
| 分析格点 128/32 | `floor((512−128)/32)+1 = 13` 帧，7.8 kHz/bin ✅ |
| 包络系数自洽 | 由 β₁=11.39、β₂=−0.666、SIR=−15 时 o\*=0.58 反解 β₀=−16.60；代入 SIR=−10 得 o\*=0.872 ≈ 表 VI 的 0.87 ✅ |
| R²_skill (IQFormer) | `1−(6.20/8.40)² = 0.455 ≈ 0.46` ✅ |
| R²_skill (MCLDNN) | `1−(8.05/9.63)² = 0.301 ≈ 0.30` ✅ |
| 严酷角落绝对值 vs 配对增益 | 0.362−0.265 = 9.7 pp；0.362−0.257 = 10.5 pp，与正文 9.74/10.46 一致 ✅ |
| Holm 计数 | Fig. 2 星标 15 个（MCLDNN 行 7 + IQFormer 行 8），12 个存活，全在 SIR=−15 行 ✅ |
| 64 cells | 32 fitting cells × 2 comparators = 64 ✅；32+81 = 113 ✅ |
| sidecar 参数/MAC | 39500→46794 = +18.5%（"+18%"）；41.8M→43.1M = +3.1%（"~+3%"）✅ |
| 设计分辨率倍数 | 4.57/1.31 = 3.49（"about 3.5×"）✅ |
| 符号检验 | 2/1024 = 0.00195 ≈ 0.0020 ✅ |

---

## A. 必须修改项（内部矛盾 / 硬错误）

### 🔴 A1. Fig. 7 星标数与正文 "4 classes" 直接冲突【最高优先级】

正文 §V-E：

> "the two training views expose jammer-free windows for only **4 classes**; the remaining **6 classes** are a condition-transfer test."

Fig. 7 图题：

> "A star marks a class whose jammer-free condition is **uncovered** during training"

但 Fig. 7 中带星的类别为 **BPSK\*, QPSK\*, 8PSK\*, 16QAM\*, 64QAM\*, GMSK\*, CPFSK\*, 4FSK\*** = **8 个**；无星为 PI2BPSK、256QAM = 2 个。

无论把 "uncovered" 读作"已覆盖"还是"未覆盖"，**8/2 都无法与正文的 4/6 对上**。这是全稿最容易被审稿人抓住的硬伤，因为 clean-retention 失败门是整条 Tier-2 修复链的起点。

**修改要求**：
1. 明确到底哪 4 个类在训练中见过 jammer-free 窗口，逐类列出（例如："BPSK, GMSK, CPFSK, 4FSK 在两个视图中均出现 jammer-free 窗口"）。
2. 废弃 "uncovered" 这个歧义词。改用二值化明示："★ = jammer-free condition **not seen** in training"，或干脆在 Fig. 7 用两种颜色的类别标签区分 covered / not-covered。
3. Fig. 7 图题同步改写，并在 §V-E 加一句交叉引用："the four covered classes are marked without a star in Fig. 7"。

---

### 🔴 A2. SIR = +5 / +10 dB 两行凭空消失

§IV-A 写：

> "the standard SIR levels are **−10 to 10 dB in 5 dB steps** and the hard-interference split adds SIR = −15 dB."

即 SIR ∈ {−10, −5, 0, +5, +10} ∪ {−15} = **6 个等级**，6 × 8 SNR = 48 cells。

但 Fig. 2 只有 **4 行**（−15, −10, −5, 0），§V-C 明确 "Only the **32** in-distribution and hard-interference cells enter the fit"（4×8=32），且 "Of **64** inspected cells" = 32×2。**SIR = +5 和 +10 的 16 个 cell 从未出现在任何图表中**。

**修改要求**：在 §IV-A 明确写出 hard-interference split 的 SIR 支撑集，例如："the hard-interference split spans SIR ∈ {−15, −10, −5, 0} dB; the SIR = +5 and +10 dB cells belong to the in-distribution split and are reported in the reproduction package"。若这 16 个 cell 确实存在但被排除在包络拟合外，必须给出排除理由（这直接影响 §V-C 的"only ... enter the fit"是否是一个 **预设的** 还是 **事后的** 划分）。

> ⚠️ 这一点如果处理不好，会被解读为"拟合域是看了结果之后挑的"，直接击穿包络分析的可信度。建议在返修信中明确声明拟合域划分是在拟合前确定的。

---

### 🔴 A3. clean-retention cell 在包络中的 SIR 取值未定义

§V-C 说 81 个 held-regime cells 包含 **clean retention**。但 Eq. (8) 是 `Δ = β₀ + β₁·o + β₂·SIR`：

- clean cell 上无干扰机 → `o` 无定义（或恒为 0）；
- clean cell 上 **SIR = +∞**（或未定义）。

那么 Fig. 4(b) 中那些 `⊠ clean retention` 点的 **predicted (pp)** 横坐标是怎么算出来的？如果给 clean cell 赋了某个有限 SIR 代理值（比如 +30 dB 或 fitting 域上界），这是一个**从未披露的建模选择**，且直接影响 RMSE = 6.20 pp 这个核心数字。

**修改要求**：
1. 在 §V-C 明确写出 clean-retention cell 的 (o, SIR) 赋值规则；
2. **补一个敏感性分析**：报告排除 clean-retention cell 后的 held-regime RMSE 与 R²_skill。若剔除后 R²_skill 显著下降，必须在正文说明。
3. 若无法定义，应将 clean-retention cell 从包络评估中剔除，重报 RMSE（held cell 数从 81 改为对应值）。

**工作量**：纯 artifact 重算，无需重训。

---

### 🟠 A4. "高机动"论证依赖的是留出速度，不是训练速度

§III-A：

> "at 1367 Hz the coherence time ... is about 309 µs, shorter than the 512 µs analysis window, so the channel is not static within a decision window **at the top of the speed range**."

但 1367 Hz 对应 250 km/h，属于 **held-speed split**（训练中未见）。训练速度上限 150 km/h：

```
f_D = 819.4 Hz  →  T_c = 0.423/819.4 = 516.2 µs  >  512 µs 窗长
```

即：**在训练分布内，信道在一个判决窗内近似静态**（516 vs 512 µs，仅差 0.8%）。"high-mobility" 的定量论证实际上建立在 OOD 速度上。

**修改要求**：把这段改写为分层陈述，而不是删掉——这其实对论文有利，因为它解释了为什么 unseen-speed 是一个真正的 regime shift：

> "At the top of the **training** speed range (150 km/h, f_D ≈ 819 Hz), T_c ≈ 516 µs is comparable to the 512 µs window, so the channel is approximately static within a decision window. At the held-out speeds (180 and 250 km/h, up to 1367 Hz), T_c falls to 430 and 310 µs respectively, so the held-speed split additionally probes intra-window non-stationarity rather than only a Doppler-spread shift."

补一张小表（速度 / f_D / T_c / T_c 与窗长之比）放到 §III-A 或附录。

---

### 🟠 A5. "12.09 pp" 和 "14.07 pp" 的比较对象未指明

§V-F：

> "while at severe SIR it remains ahead by **12.09** percentage points ([8.32, 15.88])"（sidecar）
> "the same L model is ahead by **14.07** percentage points ([9.55, 18.51])"（L tier）

两处均未写明"ahead of **谁**"。上下文暗示是 IQFormer-inspired，但 sidecar 段落上一句刚提到 A5，读者会误读为 "ahead of A5"。

**修改要求**：显式补出比较对象。同时注意 **L 的 14.07 pp 是五种子子集**，而 A5 的 10.46 pp 是十种子，两者不可并列解读；建议加一句限定。

---

### 🟠 A6. 容量阶梯的五种子/十种子落差 (1.36 pp) 超过设计分辨率

由 Table VIII：`L − S = 3.18`，`L − IQFormer = −2.84` ⇒ 五种子下 `S − IQFormer = −6.02 pp`。
由 §V-B：十种子下 `A5 − IQFormer = 0.4406 − 0.4872 = −4.66 pp`。

**落差 1.36 pp > 设计分辨率 1.31 pp**。

作者已声明"comparator means differ slightly from the ten-seed main results"，但 "slightly" 与"超过自己定义的可分辨阈值"是矛盾的。更重要的是，这暴露了 **种子方差比正文暗示的更大**，而密封族的 bootstrap 把 seeds 当作 fixed blocks（见 B3）。

**修改要求**：
1. 在 Table VIII 下方补一个脚注，给出五种子子集上 S、M、L、IQFormer、MCLDNN 的**绝对 macro-F1**，使读者能自行核对；
2. 明确写出该落差为 1.36 pp 并说明它"与设计分辨率同量级，属于种子子集效应"；
3. 不要用 "slightly"。

---

### 🟠 A7. Fig. 5 混绘了五种子与十种子结果

Fig. 5 中 A5-VIMD 位于 0.4406（十种子），而 M / L 标记来自五种子子集（由 A6 推算，五种子 S ≈ 0.4337，与图上 A5 位置差约 0.7 pp）。图题只区分了"密封 / Tier-2"，未区分"十种子 / 五种子"。这与作者自己在 Table VIII 中的告诫（"should not be numerically mixed"）自相矛盾。

**修改要求**：要么在 Fig. 5 中一律使用五种子匹配子集重绘（推荐），要么在图题明确标注哪些点是五种子并在图上加视觉区分（如空心方框加下标 "5-seed"）。

---

### 🟠 A8. Fig. 4(b) 图内标注 "fit-domain constant = 8.40 pp" 语义错误

8.40 pp 是**常数预测器的 RMSE**，不是常数预测器的取值（后者是 −8.27 pp）。当前标注会让读者误以为常数值是 +8.40。

**修改要求**：改为 `fit-domain constant RMSE = 8.40 pp`，并在图题中给出常数值本身（−8.27 pp）以便区分。

---

### 🟡 A9. 符号 γ 在 Eq.(1) 与 Eq.(7) 中冲突；S 在 Eq.(9) 与容量阶梯中冲突

- Eq. (1)：`γ` 控制 SIR
- Eq. (7)：`γ` 是 L_link 的损失权重（γ = 0.05）
- Eq. (9)：`S` 是目标 99% 能量支撑集
- §V-F / Table VIII：`S` 是最小容量档

**修改要求**：把 Eq. (1) 的 SIR 缩放因子改为 `κ` 或 `a_j`；把支撑集改为 `𝒮_τ` 或 `Ω`，容量档改为 `S-tier / M-tier / L-tier` 并全文统一带 "-tier" 后缀。另建议在 Table II 增加符号栏，把 λ、ρ、o、γ、g 一并收进去。

---

## B. 方法学与统计问题（需实质回应，部分需重算）

### 🔴 B1. 包络模型缺少 SNR 项，而操作图显示强 SNR 依赖【方法学最重要】

Eq. (8) 只含 `o` 和 `SIR`。但 Fig. 2 显示，在固定 SIR = 0 dB 行上，A5 − MCLDNN 从 −2 pp（SNR=−10）单调恶化到 −14 pp（SNR=10）；A5 − IQFormer 从 +1 pp 恶化到 −20 pp。**SNR 的边际效应量级与 SIR 相当甚至更大**，却被完全排除在模型之外。

审稿人的自然质疑是：包络的 "cross-regime transport skill"（R²_skill = 0.46）有多少是被遗漏的 SNR 项吸收进 β₀ 的？如果 held-regime cells 的 SNR 边缘分布与 fitting cells 不同，这就是一个直接的遗漏变量偏倚。

**修改要求（必做）**：
1. 拟合并报告扩展模型 `Δ = β₀ + β₁·o + β₂·SIR + β₃·SNR`，给出各系数、held-regime RMSE 与 R²_skill；
2. 若加入 SNR 后 skill 明显提升，应把三变量版本作为主模型，二变量版本作为简化对照（这对论文是**加分**，因为 SNR 在接收端是可观测的，而 overlap 不是——见 B2）；
3. 若作者有意保持二变量（为了"低维"叙事），必须显式论证并报告三变量的结果作为敏感性分析。

**工作量**：纯回归重算，< 1 小时。

---

### 🟠 B2. 包络的实用价值被 overlap 的不可观测性架空——建议主动给出"可观测代理"下界

作者在 §VI-B 和 §VII 已诚实承认 overlap 是 simulator-derived、接收端不可得，并把可观测代理留给"future prospective campaign"（§VI-E）。

问题是：**这使 Table VI 的 break-even 阈值在部署意义上等于零**。审稿人会问："那这个包络到底能用来做什么？"

**强烈建议（半天工作量，收益极高）**：在现有冻结 artifact 上，用**仅接收端可得**的量（例如混合信号 STFT 的谱峰度、占用带宽比、混合谱的 Rényi 熵、或频域能量集中度）拟合一个 `ô` 的代理估计器，报告 `corr(ô, o)`，并给出 **用 ô 替换 o 后的 held-regime RMSE**。

即使代理很差（比如 R² 只有 0.3），这也把论文从"仿真域内部一致性研究"推进到"可部署路径已被量化界定"，直接回应 §VI-E 自己提出的问题，且**不需要任何新的训练**。这一条是本稿从 7.2 提到 8.0 的最高杠杆改动。

---

### 🟠 B3. bootstrap 将 seeds 视为 fixed blocks，区间是"给定种子集"条件下的

§IV-E："resampling class-stratified source clusters while **keeping algorithm seeds as fixed blocks**"。

这意味着所有报告的区间**不包含初始化/训练随机性**，推断是条件于所实现的 10 个种子的。结合 A6 中观察到的 1.36 pp 五/十种子落差，这个假设并非无害。

**修改要求**：
1. 在 §IV-E 显式声明："intervals are conditional on the realized seed set and do not propagate seed-to-seed training variance"；
2. 至少对**密封族的三个对比**补一个种子作为随机效应的变体（分层 bootstrap 中同时重采样 seed），报告区间宽度变化。若结论不变，这是一个很强的稳健性证据；若 A5–A0 的 4.57 pp 在种子随机化后仍显著为正，论文的唯一正向密封结果就被加固了。

---

### 🟠 B4. 用 "design resolution" 作等效界（equivalence bound）是统计误用

§V-A："the residual-free A7 variant is nominally above A5 by 0.21 percentage points ... **well below the design resolution** and consistent with the bounded component-level result"。

1.31 pp 是"80% 功效下的最小可检测效应"，是一个**功效量**，不是等效界。"低于 MDE" ≠ "无差异"。

**修改要求**：直接报告 `A7 − A5` 的配对 bootstrap 区间。如果该区间落在 [−1.31, +1.31] 内，那才是一个合法的等效陈述；否则应改写为"未被本设计分辨"（not resolved at this design's power），而非"consistent with no effect"。

---

### 🟠 B5. 81 个 held-regime cells 高度相关，RMSE 应按 regime 分解

81 个 cell 来自 5 个 regime（unseen jammer / unseen speed / heldout channel / combined OOD / clean retention）。它们**在 regime 内高度相关**，有效样本量远小于 81。当前只报了一个池化 RMSE = 6.20 pp，读者无法判断 skill 是均匀的，还是被某一两个 regime 主导。

**修改要求（必做）**：补一张表：

| Held regime | #cells | 观测增益均值 (pp) | 包络 RMSE (pp) | 常数 RMSE (pp) | R²_skill |
|---|---|---|---|---|---|
| unseen jammer | | | | | |
| unseen speed | | | | | |
| heldout channel | | | | | |
| combined OOD | | | | | |
| clean retention | | | | | |
| **pooled** | 81 | | 6.20 | 8.40 | 0.46 |

这是**低成本、高说服力**的补充。如果各 regime 的 R²_skill 都为正，论文的核心声明大幅加固；如果 skill 集中在一两个 regime，作者也应主动说明（这与作者一贯的诚实基调一致，不会扣分）。

---

### 🟠 B6. 相对 oracle constant 的 R²_skill 为负，应显式给出

正文已说 oracle constant 达到 6.03 pp（vs 包络 6.20 pp）。换算成同一尺度：

```
R²_skill(vs oracle const, IQFormer) = 1 − (6.20/6.03)² = −0.057
R²_skill(vs oracle const, MCLDNN)  = 1 − (8.05/6.77)² = −0.414
```

即**包络在 held-regime 域内不如一个事后最优常数**，MCLDNN 侧尤其明显（−0.41）。作者用文字表达了这一点（"transporting the level ... rather than resolving variation within them"），但没给数字。

**修改要求**：把上述两个负值直接写进 §V-C。这不会削弱论文——它精确界定了 skill 的性质（跨域水平搬运，而非域内解释力），反而堵住了审稿人"你在夸大 0.46"的质疑路径。同时建议在 Fig. 4(b) 的文本框中加一行 `oracle constant RMSE = 6.03 pp`。

---

### 🟡 B7. 严酷角落区间是否经多重性调整未说明

§V-B 报告 `9.74 ([7.52, 12.14])` 和 `10.46 ([6.78, 14.29])`，Fig. 2 图题却说星标是 "**unadjusted** paired lower interval"。正文这两个头条区间到底是同时区间还是边际区间？

**修改要求**：在 §V-B 明确标注（建议同时给出未调整与 Holm/同时调整两版）。既然作者已经在做 Holm 校正，把头条数字的调整状态说清是必要的。

---

### 🟡 B8. 探针"保守侧"的判定逻辑需澄清

§V-E："For A5, 5 of 6 design–seed cells lie below it, and one cell is borderline (0.256); following the preregistered consistency rule we record the **conservative** side."

问题：判定为 "representation-limited" 恰恰是**支持后续 sidecar 干预**的那一侧，很难称之为"保守"。真正保守的判定在 5:1 分裂下应是 "inconclusive"。

**修改要求**：
1. 说明 "design–seed cells" 的构成（2 种探针 × 3 种子？请写明）；
2. 重新表述："conservative" 改为明确的规则陈述，例如"the preregistered rule declares representation-limited unless **all** cells exceed the threshold"，并说明该规则在 Tier-2 protocol 中的预注册位置；
3. 补一句敏感性：若阈值取 0.20 或 0.30，判定是否翻转？

---

### 🟡 B9. sidecar 架构完全未描述

sidecar 是 Tier-2 修复链的落点，也是摘要中的正向结论之一（clean +11.77 pp），但全稿**没有一句话描述它的结构**：I/Q 分支是几层？在哪一层与谱路融合（早融合 / 晚融合 / 特征级）？7294 个新增参数如何分配？

**修改要求**：在 §V-F 补 5–8 行架构描述 + 一个参数分解表（I/Q encoder / fusion / head）。另需说明 MAC 只增 3% 而参数增 18% 的原因（暗示 I/Q 分支是低时间分辨率或强下采样的）。

---

### 🟡 B10. Fig. 6 的度量从 macro-F1 切换到 accuracy，未加说明

全稿主度量是 macro-F1，但 §V-D 和 Fig. 6 使用 "selective **accuracy**"。在选择性预测中，低覆盖率下保留样本的类别分布会严重倾斜，accuracy 可能被"丢掉难类"人为抬高。94.4% @ 30% coverage 与边际 macro-F1 ≈ 0.44 的反差很大，读者会困惑。

**修改要求**：补报 30% 覆盖率下的 **selective macro-F1**，或至少给出保留集合的类别分布熵。同时说明 latency 测量的硬件、batch size 与是否含预处理（§V-D 的 3.172/3.789/4.180 ms 目前无测量条件）。

---

## C. 图表问题

| 编号 | 位置 | 问题 | 修改建议 |
|---|---|---|---|
| C1 | Fig. 1 | 横轴 "STFT frame" 只显示到 ~2.5，但格点有 29 帧 | 核对刻度渲染；若确为 29 帧，补 0/14/28 刻度 |
| C2 | Fig. 2 | 单元格取整（"9\*"）与正文 9.74 不一致；出现 "-0" | 显示一位小数；把 "-0" 改为 "0.0" 或 "−0.0"（并加脚注说明四舍五入） |
| C3 | Fig. 2/4 | 图例称 Eq. (8) 为 "law"，与 §VI-B "not a causal structural equation" 矛盾 | 全文与图例统一改为 "envelope fit" / "fitted envelope"，删除 "law" |
| C4 | Fig. 3(a) | 误差棒定义未说明（种子标准差？bootstrap CI？） | 图题补充 |
| C5 | Fig. 5 | 标记语义不清（蓝色星形 vs 红色圆点 vs 空心）；且混绘五/十种子（见 A7） | 重绘 + 完整图例 |
| C6 | Fig. 7(a) | 标题为 "compact spectral family (A0--A7)"，把 A0–A7 聚合成一组柱 | 明确是族均值还是 A5；若是族均值，须说明为何用族均值论证 A5 的 clean 边界 |
| C7 | Table V/VII/VIII | **只有差值，无绝对 macro-F1** | 三张表均加一列 "reference absolute macro-F1"（尤其 clean 基线，正文全篇未给 A5 的绝对 clean macro-F1，导致 +11.77 pp 无法定位） |
| C8 | 全局 | 每个 cell 的测试窗口数从未给出 | 在 §IV-A 补 "each SNR–SIR cell contains N test windows"，这是 CI 宽度的前提 |

---

## D. 表述、格式与参考文献

**D1（建议采纳）**：摘要补入边际结果。当前摘要只给了 post-hoc 角落的 +9.74/+10.46，未给边际的 0.4406 vs 0.4599/0.4872。建议在 "In a post-hoc severe-interference analysis..." 之前插入一句：

> "Marginally over the hard-interference split the compact model trails both I/Q comparators (macro-F1 0.4406 versus 0.4599 and 0.4872); the ranking reverses only in the severe, high-overlap corner."

这与全文"不主张一致优越"的基调完全一致，且会让编辑对作者的诚信度评价上升一个档次。

**D2**：标题过长（19 个实词）。建议：
> *Condition-Indexed Operating Envelopes for Automatic Modulation Classification under Structured Interference*
或
> *When Does a Compact Spectral AMC Model Win? An Overlap–SIR Operating Envelope and a Representation Repair*

**D3**："Evidence status" 独立段落在 TVT 中非标准。建议改为摘要后的脚注（footnote on first page），或并入 §I 末尾。请核对 TVT 模板是否允许。

**D4**：§IV-A 只说 "Target classes cover PSK, QAM, continuous-phase, and FSK families"，十个类别直到 Fig. 7 才出现。请在 §IV-A 显式枚举。

**D5**：干扰族术语不一致——§IV-A 用 "chirp"，Fig. 1 元数据用 "sweep"。统一。

**D6**：Eq. (3) 中 `Z_m = (M_s + λM_o + ρ) ⊙ Z`。加入 bounded bypass ρ 后，`M_s + λM_o + ρ` 可能 > 1，Eq. (2) 的凸分配性质被破坏，`Z_m + Z_j ≠ Z`。请：(a) 说明 ρ 是标量还是场；(b) 给出其取值范围；(c) 明说分配在加入 ρ 后不再是精确划分。

**D7**：Eq. (7) 为可读性省略了 route-orthogonality 正则项。既然作者在同一段承认它是 A5 bundle 的一部分，建议直接写进 Eq. (7)（多一项不影响可读性），避免"关键式子不完整"的观感。

**D8**：Eq. (1) 未给出 SIR 与 γ 的显式关系。补一行 `SIR = E[|H_s(s)|²] / (γ² E[|H_u(u)|²])`（或按实际定义）。

**D9**：CSSL 参数 8.63 M 但 MAC 仅 155.3 M —— 比 MCLDNN（0.41 M 参数 / 398.2 M MAC）参数多 21× 而 MAC 少 61%。这个组合看起来反常，建议加脚注说明（大投影头 / 编码器下采样激进）。

**D10**：参考文献核查
- [19] *IEEE TCCN*, vol. 12, **pp. 6929–6941**, 2026 —— TCCN 年页码总量通常在 2000 页量级，6929 起页存疑，请核对；
- [21]–[23] 为 2026 年 arXiv 预印本，占据"最新前沿"的定位段落。建议在投稿前检查是否已有正式版本，并至少将其中一篇替换为已发表文献，以免审稿人认为定位段落建立在未经同行评议的材料上；
- [33] GitHub commit 2fbc5b3 已给出，很好；建议对 MCLDNN 与 IQFormer 的复现代码也提供同等级别的 commit 级溯源（见 §E）。

**D11**：语言层面整体偏名词化、抽象度高（"condition-indexed"、"whole-configuration contrast"、"representation-information gap" 密集出现）。建议做一遍以动词为中心的重写，尤其是 §I 与 §VI。

---

## E. 一条会被强烈要求的补充实验（建议主动做）

**E1. 比较器复现保真度的外部锚点（半天，价值极高）**

全稿**没有任何公开数据集结果**。作者在 §VI-E 正确地论证了"公开基准无法验证包络本身"（因为不提供分离的 target/jammer 分量）——这个论证是对的。

但审稿人真正想问的是另一个问题：**"IQFormer-inspired" 和 "MCLDNN" 复现得对吗？** 如果这两个强基线欠训练，那么整篇论文的秩交换结论就是伪的。而这个问题**恰恰可以用公开数据集回答**。

**建议**：在 RML2016.10a（或 RML2018.01a）上跑一次 MCLDNN 与 IQFormer-inspired，报告其准确率-SNR 曲线，与原文献报告值对比，放入附录。加一句：

> "On RML2016.10a our unified-retraining references reach XX% and YY% peak accuracy, within Z pp of the values reported in [5] and [7], indicating that the comparators are faithfully implemented; the public benchmark cannot validate the envelope itself because it does not expose separately tracked target and jammer components."

这一段能同时堵住"基线不公平"和"纯仿真"两条最常见的拒稿路径，而且**不需要触碰任何密封结论**。

---

## F. TVT 契合度（编辑视角）

**F1**：vehicular 论证目前只有两个支点：f_c = 5.9 GHz 和终端速度。而 A4 又显示训练速度区间内信道近似准静态。建议在 §VI 加半页 "Vehicular relevance" 讨论：把 overlap–SIR 平面映射到一个具体的 ITS 场景（例如 5.9 GHz ITS-G5 频段内的邻信道泄漏、隧道内多径引起的高 overlap、以及路侧单元密集部署时的同频干扰几何），说明 break-even overlap 落在哪种车载场景。**不需要新实验，纯讨论。**

**F2**：元方法论（sealed family / evidence tiers / preregistration）的篇幅在 TVT 属于不寻常。建议压缩 §IV-D、§IV-E 各三分之一，并把 §III-F 的部分实现细节移入附录，为 F1 和 B5 的新增内容腾出版面。

---

## G. 修改优先级与工作量

| 优先级 | 条目 | 工作量 | 是否需重算/重训 |
|---|---|---|---|
| 🔴 P0 | A1 (Fig.7 星标)、A2 (SIR 行)、A3 (clean cell 的 SIR) | 各 0.5–2 h | A3 需 artifact 重算 |
| 🔴 P0 | B1 (加 SNR 项) | 1 h | 回归重算 |
| 🔴 P0 | B5 (分 regime RMSE 表) | 1 h | artifact 重算 |
| 🟠 P1 | A4–A9、B3、B4、B6、B7 | 各 0.5–2 h | B3 需 bootstrap 重跑 |
| 🟠 P1 | B9 (sidecar 架构)、C7 (绝对值列) | 2–3 h | 无 |
| 🟠 P1 | E1 (公开数据集锚点) | 0.5 天 | 需 2 次训练 |
| 🟡 P2 | B2 (可观测代理 ô) | 0.5–1 天 | 特征计算 + 回归 |
| 🟡 P2 | B8、B10、C1–C8、D1–D11、F1、F2 | 1–2 天合计 | 无 |

**总计约 3–4 天**，其中真正需要 GPU 的只有 E1（2 次训练）和 B3（bootstrap，CPU 即可）。

---

## H. 建议的回复信框架

分四块组织，与本意见编号对齐：

1. **"内部一致性已全部修复"** —— 逐条列出 A1–A9 的具体改动位置（页/行），并在开头声明"我们重新核查了全稿的每一处数值交叉引用，并附上了自动化一致性检查脚本"。这一句能一次性化解审稿人对元方法论可信度的疑虑。
2. **"包络的规格已收紧"** —— B1（加入 SNR）、B5（分 regime）、B6（vs oracle 的负 R²）、A3（clean cell 处理）打包回复，主线是"我们把 skill 的边界写得更窄、更准了"。
3. **"比较器保真度已有外部锚点"** —— E1。
4. **"未做的以及为什么"** —— B2 若只做到代理相关性而未做完整验证，明确说明；不要含糊。作者在原稿中已建立了"坦率报告失败"的信誉，回复信应延续同一风格。

---

## I. 最终结论

**Minor Revision（小修）**。

本稿的科学内核是成立的：秩交换现象在 10 个种子上方向一致、Holm 校正后存活、并被一个诊断-干预链闭环。**没有任何一条意见要求推翻结论或重做主实验**。

真正的风险不在科学，而在**呈现**：一篇把"证据分级严谨性"作为核心卖点的论文，出现 A1/A2 这类图文数字对不上的问题，代价是不成比例的。P0 五项修完，本稿的说服力会有台阶式提升。

若进一步完成 B2（可观测 overlap 代理）与 E1（公开数据集锚点），本稿有机会从"一份诚实的仿真域研究"上升为"一个可被后续工作直接接续的设计规则"，届时综合评分可达 8.0+，录用概率 75–82%。

---

*审稿意见结束。以上所有数值关系均已独立验算，验算细节见 §1。*
