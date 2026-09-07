# TVT 零算力再分析结果（不扩大仿真）

> 生成日期：2026-08-12  
> 输入：`artifacts/tvt_v4r_headline_composite/`（120 fits × 11 splits 的封存预测包）、`standards/cache_factor_headline_1024_v2/` 的逐样本生成元数据、`artifacts/tvt_learning_curve_v4_8gb_dual/learning_curve_evidence.json`  
> **证据类别：exploratory re-analysis of sealed predictions。无新仿真、无重训、未修改任何冻结 artifact。**  
> **文档分工**：本文件是**内部证据账本**，措辞钝、边界写满，用于防止自欺；对外稿件的叙事、主张阶梯与表述规范见 `docs/TVT_REPOSITIONING_AND_CLAIM_LADDER.md`。两份文档的数字必须一致，语气不必一致。  
> 科学发布门状态**未改变**：`scientific_evidence_passed=false`、`submission_unlocked=false`，失败门仍为 confirmatory family 与 clean retention。本文件不解除、不重述、不替代这两个失败。

## 0. 管线自校验

`make_provenance.py` 用本管线的估计量重算了 4 个已封存的 headline 数字，与 sealed 值**逐位一致**（差 < 1e-9）：

| 校验 | sealed | 本管线重算 |
|---|---:|---:|
| A5 − CSSL macro-F1 @ hard_interference | +0.049107 | +0.049107 |
| A5 − CSSL macro-F1 @ clean_retention | −0.096933 | −0.096933 |
| A5 − CSSL macro-F1 @ id_test | +0.049621 | +0.049621 |
| A5 − CSSL macro-F1 @ unseen_jammer | −0.041424 | −0.041424 |

所有输入/产物/脚本的 SHA-256 见 `outputs/provenance.json`。

## 1. 最重要的一条：严重干扰角上的可发表主张

冻结证据的边缘统计只说"A5 绝对精度不如 MCLDNN/IQFormer"。按 SIR 分辨后结论**发生质变**：

**hard_interference，SIR = −15 dB（1,227 个测试源，10 seeds，分层配对 hierarchical bootstrap）**

| 模型 | macro-F1 | 参数量 |
|---|---:|---:|
| **A5-VIMD** | **0.3621 ± 0.0188** | 39.5 k |
| A0 backbone | 0.3194 ± 0.0082 | 8.5 k |
| CSSL | 0.2876 ± 0.0168 | 8.63 M |
| MCLDNN | 0.2647 ± 0.0235 | 406 k |
| IQFormer-inspired | 0.2576 ± 0.0500 | 355 k |

| 配对差（A5 − 参照） | 差值 | 95% CI |
|---|---:|---:|
| vs IQFormer-inspired | **+10.46 pp** | [+6.93, +14.14] |
| vs MCLDNN | **+9.74 pp** | [+7.56, +12.15] |
| vs CSSL | +7.46 pp | [+5.14, +9.76] |
| vs A0 | +4.28 pp | [+2.49, +5.89] |

即：**在最严重的结构化干扰条件下，39.5 k 参数的物理引导模型显著优于两个 9–10 倍大的强基线**，CI 严格为正。

`a1b` 的 SNR×SIR 联合图进一步显示这不是挑出来的单格：SIR = −15 dB **整行 8 个 SNR 档中有 7–8 格**对两个强基线均显著为正（+7 到 +18 pp），而 SIR ≥ −10 dB 的区域全部反向。这是一个连续、物理可解释的工作区，不是多重比较的偶然产物。多重性说明：强基线相关的格子共 64 个，显著为正 15 个，且全部落在同一行。

**建议**：论文头条改为"严重干扰角的紧凑方法"，并把 SIR 边缘统计（+9.74 / +10.46 pp）作为主张，联合图作为佐证。

## 2. A1：SNR/SIR 分辨曲线（AMC 论文的必备图，此前缺失）

A5 − 参照的 macro-F1 差（pp），按 SNR：

| SNR (dB) | vs A0 | vs CSSL | vs IQFormer |
|---:|---:|---:|---:|
| −10 | +1.42 | +5.35 | +0.38 |
| −6 | +3.03 | +8.08 | −0.90 |
| −2 | +6.36 | +11.48 | −0.68 |
| +2 | +5.09 | +7.47 | −5.51 |
| +6 | +6.27 | +5.62 | −4.26 |
| +10 | +5.06 | +1.07 | −8.31 |
| +18 | +4.73 | +2.06 | −7.24 |

结论：A5 的优势集中在**低 SNR / 低 SIR**，劣势集中在良性条件。clean_retention 上 A5 − CSSL 从 −10 dB 的 **+3.89 pp** 单调恶化到 +18 dB 的 **−18.50 pp**——所谓"clean 退化"其实是**高 SNR 退化**。

产物：`a1_absolute_by_level.csv`、`a1_paired_by_level.csv`、`figA1a–figA1c`、`a1b_snr_sir_map.csv`、`figA1d`。

## 3. A2：按干扰族与频谱占用分层

hard_interference 上 A5 − 参照（pp）：

| 干扰族 | vs A0 | vs CSSL | vs IQFormer |
|---|---:|---:|---:|
| comb | +7.74 | +13.16 | **+2.67** |
| multitone | +5.27 | +7.31 | −0.14 |
| tone | +2.89 | +7.41 | −5.04 |
| chirp | +3.42 | +4.18 | −5.94 |
| partial_band | +3.83 | +1.42 | −8.18 |
| sweep | +4.81 | −3.46 | −10.37 |

按频谱占用五分位（hard_interference）：

| 占用区间 | 均值占用 | A5 − A0 | A5 − CSSL |
|---|---:|---:|---:|
| [0.000, 0.000] | 0.000 | +3.61 pp | −7.56 pp |
| [0.000, 0.166] | 0.048 | +3.60 pp | −3.48 pp |
| [0.166, 0.370] | 0.291 | +5.12 pp | +2.67 pp |
| [0.370, 0.717] | 0.490 | +5.47 pp | +9.72 pp |
| [0.717, 1.000] | 0.884 | +4.77 pp | **+23.31 pp** |

**A5 − CSSL 随占用率严格单调上升（Spearman ρ = +1.0）**。unseen_jammer 的退化则完全落在两个 held-out 族（ofdm_like、pulse），它们的占用中位数只有 0.134，远低于训练可见族的 0.299——退化与"低占用、窄带/脉冲型"高度一致，与第 1 节的工作区结论自洽。

产物：`a2_strata_absolute.csv`、`a2_strata_paired.csv`、`figA2`。

## 4. A3：clean-retention 的解剖（本项目最有价值的诊断）

−9.69 pp 不是弥散退化，而是**三个类在无干扰条件下的坍塌**：

| 调制 | A5 − CSSL 逐类 F1 |
|---|---:|
| QPSK | **−60.39 pp** |
| 8PSK | −22.40 pp |
| 64QAM | −20.52 pp |
| 16QAM | −19.44 pp |
| GMSK | +8.38 pp |
| 4FSK | +7.47 pp |
| 256QAM | +5.60 pp |
| CPFSK | +2.57 pp |
| BPSK | +2.35 pp |
| PI2BPSK | −0.56 pp |

混淆矩阵（10 seeds 均值计数）显示机制极其具体：clean 条件下 A5 **几乎不再输出 QPSK / 16QAM / 64QAM**（每类约 2 个窗口，CSSL 为 89–294 个），全部塌到 8PSK 与 256QAM；A5 的最大预测类别占比 0.258，CSSL 只有 0.135。

按信道 profile 分层（seen A/C/D 与 held B/E、逐 profile 0–4）差异在 −9.1 到 −10.4 pp 之间**高度均匀**——不是信道泛化问题。

### 4.1 关键归因修正：这不是 VIMD 造成的

对 12 个模型逐类扫描（`a3b_clean_family_scan.py`）给出决定性结论：

| 模型 | clean 上坍塌类数 | 坍塌类 | QPSK/16QAM/64QAM 平均 F1 |
|---|---:|---|---:|
| A0 backbone | 3 | QPSK,16QAM,64QAM | 0.0038 |
| a1_single_mask | 3 | 同上 | 0.0064 |
| a2_tri_no_teacher | 3 | 同上 | 0.0065 |
| A3 / A3p / A4 | 3 | 同上 | 0.0062–0.0065 |
| **A5-VIMD** | 3 | 同上 | 0.0084 |
| A6 / A7 | 3 | 同上 | 0.0082–0.0105 |
| CSSL | 0 | — | 0.3429 |
| MCLDNN | 0 | — | 0.2916 |
| IQFormer-inspired | 1 | 64QAM | 0.4388 |

**A0–A7 全部九个模型在 clean 上以完全相同的方式坍塌（recall < 0.01），三个外部基线不坍塌。** 因此该缺陷属于**共享谱域 backbone / 前端**，与 tri-route 掩模、teacher、residual、MTL 等任何消融成分无关；A5 在 clean 上实际上比 A0 还高 +3.00 pp（CI [+2.42, +3.63]）。

同时注意：在 jammed 条件下同一批类并不坍塌（A0 在 hard_interference 上 QPSK/16QAM/64QAM 的 F1 为 0.27/0.15/0.24）。也就是说，该紧凑谱域前端在**无干扰**条件下失去了高阶线性调制的星座阶数分辨能力。

### 4.2 条件覆盖：训练实际使用的两个 view 中，6/10 个调制**从未出现过无干扰样本**

对缓存元数据做条件覆盖统计（`a9_condition_coverage.py`）：

| split | 有无干扰样本的调制 | 无干扰样本数 |
|---|---|---:|
| train | PI2BPSK、8PSK、256QAM、CPFSK | 详见 `a9_condition_coverage.csv` |
| validation | PI2BPSK、8PSK、256QAM、CPFSK | 详见 `a9_condition_coverage.csv` |
| id_test | PI2BPSK、8PSK、256QAM、CPFSK | 详见 `a9_condition_coverage.csv` |
| **clean_retention** | **全部 10 类** | 各 500 |

也就是说：**clean_retention 对其中 6 个调制而言，是一个设计上就完全 out-of-distribution 的条件迁移测试**——训练时这些类只在有干扰的条件下出现过。旧 2/10 口径只统计了 view1，漏掉了同样用于训练的 view2，现显式撤回。

选择性不能只由覆盖解释：8PSK 见过 clean 且不坍塌；QPSK/16QAM/64QAM 既未见 clean，又必须依赖星座阶数/幅度密度区分，它们在紧凑谱域家族中稳定失败。A13 重算后，"未覆盖 × 星座阶数"的失败率为紧凑谱域 27/27，IQ 域高容量 1/9；因此必须把**条件覆盖**与**表示域**作为两个因素报告。

论文写法应为："clean-retention 门在本缓存设计下是条件迁移测试；紧凑谱域前端对需要星座阶数分辨的线性高阶调制迁移失败，高容量 IQ 域基线不失败；该代价由共享 backbone 与训练条件覆盖共同决定，不是所提出机制引入的"。**不得**写成"掩模删除了干净信号信息"——A0 无掩模却同样坍塌，该因果解读被证据排除（与 TCCN 支线 Arm A 的教训一致）。

产物：`a9_condition_coverage.csv`、`a9_condition_coverage_summary.json`。

### 4.3 Tier-2 探针：v1 作废，v2 已完成

`artifacts/tier2_clean_probe_v1/`（6 次运行，A0/A5 × seed 17/29/43）全部给出 `representation_limited`，线性探针在 QPSK/16QAM/64QAM 上的 F1 均值仅 0.162–0.207（冻结分类头为 0.000–0.023）。**但该判定被输入构造混淆，不能采用**：

审计（`a10_counterfactual_check.py`）确认缓存的混合律为

```
x = (clean + jammer + noise + receiver_artifact) / sqrt(mean|·|²)
```

在 train / validation / clean_retention / hard_interference 上逐窗验证，最大相对误差 2.3e−7（ADC split 因量化阶段例外，中位 4.1e−5）。而首轮探针使用的是 `x − jammer`：封存分量与 `x` 处在同一 post-AGC 尺度，但**去掉干扰后未重新归一化**。其结果窗口的 mean-square 落在 0.046–0.50，而模型见过的每一个窗口都恰好是 0.5。因此该输入不在模型的尺度契约上，v1 判定不能采用。

正确的无干扰反事实为 `AGC(clean + noise + receiver_artifact)`——同一生成管线去掉干扰项，属于**已封存分量的重混，不是新仿真**。已交付 `tier2_gpu/clean_probe_v2.py`，含两种设计（反事实重混 / 在真实 clean_retention 上做 source-disjoint 5 折交叉验证）与线性＋MLP 两种探针，并支持对 CSSL/MCLDNN/IQFormer 取参照上界（三者均暴露 `embedding`）。

v1 的 6 次运行已归档并标注 `validity=confounded`。v2 已完成两种设计 × A5/A0/CSSL/MCLDNN/IQFormer × seeds 17/29/43。A5 的高阶类 MLP F1 在 counterfactual 为 0.205–0.220（3/3 低于 0.25），在 source-disjoint CV 为 0.236–0.256（2/3 低于 0.25）。两设计的硬判定未完全一致，故严格结论是**阈值边界，整体证据偏向 `representation_limited`**；不支持把修复仅限于替换分类头。CSSL/MCLDNN/IQFormer 在 18/18 个对应单元中均高于 0.25，说明该阈值能区分表示容量。

产物：`a3_clean_per_class.csv`、`a3_clean_per_class_deficit.csv`、`a3_clean_confusion.csv`、`a3_clean_by_profile.csv`、`a3_clean_prediction_collapse.csv`、`a3b_clean_family_scan.csv`、`a3b_clean_collapse_summary.csv`、`figA3`。

## 5. A4：校准与选择性预测（把低绝对精度变成工程可用表述）

温度仅在 validation 上拟合后应用到测试集。hard_interference 的选择性正确率：

| 覆盖率 | A0 | **A5** | CSSL | MCLDNN | IQFormer |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.887 | **0.944** | 0.730 | 0.967 | 0.949 |
| 50% | 0.670 | **0.739** | 0.570 | 0.758 | 0.778 |
| 100% | 0.429 | 0.471 | 0.388 | 0.469 | 0.495 |

AURC：A5 0.2446、MCLDNN 0.2348、IQFormer 0.2226、A0 0.2889、CSSL 0.3828。

可写："在 30% 覆盖率下 A5 达到 94.4% 正确率，与 IQFormer（94.9%）相差 0.5 pp，而参数量少 9 倍"。同时补上了稿件自陈的"校准仅为描述性"缺口。注意诚实点：温度标定显著改善 CSSL 的 ECE（0.178→0.081），但对本已接近校准的模型（拟合温度 < 1）反而略微恶化测试 ECE，需如实报告。

产物：`a4_temperature.csv`、`a4_risk_coverage_curves.csv`、`a4_selective_points.csv`、`figA4`。

## 6. A5：互补性与概率级融合（对"不如强基线"的正面回应）

hard_interference，同 seed 概率平均：

| 组合 | 融合 macro-F1 | 相对更强成员 | 95% CI | 同架构双种子集成对照 | 跨模型净增益 |
|---|---:|---:|---:|---:|---:|
| A5 + IQFormer | 0.5343 | **+4.71 pp** | [+3.72, +5.82] | +1.49 pp | **+3.22 pp** |
| A5 + MCLDNN | 0.5169 | **+5.70 pp** | [+4.82, +6.58] | +1.69 pp | **+4.02 pp** |
| A5 + CSSL | 0.4824 | +4.18 pp | [+3.31, +5.02] | +0.77 pp | +3.41 pp |
| A5 + A0（对照） | 0.4364 | −0.42 pp | [−0.85, −0.01] | +0.77 pp | −1.19 pp |

关键在于双重对照：(i) 与**更强**成员比，不与弱成员比；(ii) 与"同架构双种子集成"比，排除"集成本身"的贡献。结论：**在结构化干扰下，A5 携带两个强基线都没有的决策信息**；而与自身 backbone A0 融合没有增益，说明该信息来自 VIMD 的表示分配而非模型多样性。

诚实边界：在 unseen_jammer / combined_ood / clean_retention 上融合**不带来增益甚至有害**，必须一并报告。

产物：`a5_error_overlap.csv`、`a5_fusion.csv`、`figA5`。

## 7. A6：机制中介与 occupancy 假设再分析

跨 50 个装有机制仪表的 fit（a3/a3p/a4/a5/a7 × 10 seeds），与 hard-interference 相对 A0 增益的 Spearman 相关（BH 校正）：

| 机制量 | ρ | q |
|---|---:|---:|
| counterfactual_tf_sir_gain_db | +0.566 | 0.0005 |
| lambda_overlap | +0.536 | 0.0008 |
| signal_route_weighted_correlation | +0.483 | 0.0035 |
| jammer_route_weighted_correlation | +0.455 | 0.0062 |
| signal_retention（= target_energy_transfer_ratio_mean） | +0.358 | 0.046 |
| mask_js | −0.326 | 0.065 |

即：**反事实 TF 域 SIR 改善越大、路权与 oracle 相关性越高的拟合，实际增益越大**——这是当前证据能支持的最强机制陈述（相关性，非因果）。

occupancy 预注册门**仍然失败且不被本节挽救**：sealed ρ = +0.714、单侧 p = 0.949、`direction_supported=false`。探索性再分析给出的结论是**反方向且更强**：A5 − CSSL 随占用率单调上升（bin 级 ρ = +1.0），A5 − A0 的 bin 级 ρ = +0.6（p = 0.28，不显著），族级 ρ = −0.6（p = 0.21，不显著）。写法必须是"预注册方向被证伪 + 探索性观察到相反关联"，且必须指出占用率与 SIR 在本设计中相关（bin 级 ρ = −0.8），因此不能作因果解读。

产物：`a6_mechanism_association.csv`、`a6_occupancy_bins.csv`、`a6_occupancy_families.csv`、`a6_occupancy_summary.json`、`figA6`。

## 8. A7：样本效率（复用已封存学习曲线）

hard_interference macro-F1（5 seeds）：

| 训练源数 | A0 | A5 | A5 − A0 | 相对增益 |
|---:|---:|---:|---:|---:|
| 10 k | 0.2463 | 0.2907 | +4.44 pp | +18.0% |
| 30 k | 0.3273 | 0.3813 | +5.40 pp | +16.5% |
| 100 k | 0.3941 | 0.4359 | +4.18 pp | +10.6% |

数据等效（对数插值，探索性）：A5 在 30 k 上的表现需要 A0 约 **79 k**（2.6×）；三个指标的等效倍数在 **1.2–2.6×**。可写："该物理引导设计在本设置下约相当于 2 倍训练数据"，并注明 100 k 档为外推。

产物：`a7_learning_curve_levels.csv`、`a7_learning_curve_gains.csv`、`a7_data_equivalence.csv`、`figA7`。

## 9. A8：Pareto 前沿与非劣性

hard_interference 上按参数量/MACs 的 Pareto 最优集合：A0（8.5 k）、a1_single_mask（24.6 k）、**a7_vimd_no_residual（39.5 k，0.4427）**、IQFormer（355 k，0.4872）。

**必须诚实记录**：在这条前沿上，同价位的 A7（无 residual 消融）名义上略高于 A5（0.4427 vs 0.4406，差 0.21 pp），MCLDNN 与 CSSL 被支配。因此论文写 Pareto 时应写"VIMD 家族（A5/A7 同价位）占据 39.5 k 档的前沿位置"，不要写成"A5 单点最优"；A5 与 A7 的差异远小于其 seed 波动，与已封存的 residual 消融结论一致（不确定）。

探索性非劣性（seed 级 TOST，margin 1 pp / 2 pp）：

| 对比 | 差值 | 1 pp 非劣 | 2 pp 非劣 |
|---|---:|:--:|:--:|
| heldout_channel vs CSSL | +1.13 pp | 是 | 是 |
| unseen_speed vs CSSL | +1.54 pp | 是 | 是 |
| hard_interference vs MCLDNN | −1.93 pp | 否 | 否 |
| hard_interference vs IQFormer | −4.66 pp | 否 | 否 |

margin 为事后声明，只能标注为探索性。

产物：`a8_pareto.csv`、`a8_noninferiority.csv`、`a8_severe_corner.csv`、`figA8`。

## 10. 可写 / 不可写（在原有边界上的增量）

**新增可写**

- 在 SIR = −15 dB 的严重干扰条件下，A5 以 39.5 k 参数显著优于 MCLDNN（+9.74 pp）与 IQFormer（+10.46 pp）；
- A5 的优势区是低 SNR / 低 SIR / 高频谱占用；劣势区是高 SNR、低占用、held-out 窄带干扰族；
- clean-retention 代价的具体位置：无干扰条件下 QPSK/16QAM/64QAM 坍塌到 8PSK/256QAM，且**九个 A0–A7 模型完全一致**，属共享谱域前端的星座阶数分辨力缺失，非 VIMD 机制引入；
- A5 与强基线决策互补，融合净增益超出同架构集成对照 +3.2 到 +4.0 pp；
- 30% 覆盖率下 A5 达 94.4% 正确率，接近 IQFormer 而模型小 9 倍；
- 内部机制量（反事实 TF-SIR 改善等）与实际增益跨 50 个拟合显著正相关；
- 该设计在本设置下约等价于 2 倍训练数据。

**仍然不可写**

- 不可写 A5 在整体或多数场景优于强基线（仅严重干扰角成立）；
- 不可写 clean-retention 已通过或代价轻微；
- 不可写 occupancy 单调机制成立（预注册方向已被证伪）；
- 不可写 margin teacher 或三路结构已获确认；
- 不可写 A5 在同价位上单点最优（A7 同价位名义更高）；
- 不可写"掩模/删除破坏了干净信号信息"——A0 无掩模却同样坍塌，因果解读被证据排除；
- 不可写任何实测、SDR、板卡、外场、在轨或部署主张；
- 本文件全部结果为探索性，不得并入冻结确认性家族，不得用于宣称发布门通过。

## 11. Tier-2（需算力、不新增仿真数据）

`tier2_gpu/` 已交付；探针、H2-F1、H2-F2、H1 与 H2-G 均已完成：

- `PREREGISTRATION_TIER2.md`：H1 容量阶梯、H2-D 冻结表示线性探针、H2-F 前端修复、H2-G 门控对照，含判定阈值、禁止事项与签署栏；
- `clean_probe_v2.py`（**已完成**）：结论为阈值边界、整体偏向表示受限，因此后续不只改分类头；
- `run_tier2_experiment.py`：已实现每类精确 10% 的 clean 条件覆盖臂，以及仅读 mixture IQ、总参数 46,794 的轻量 sidecar 前端臂；不修改封存 runner/缓存；
- `presence_gated_vimd.py`：干扰存在性门控变体，因 §4.1 的归因修正**已从主修复降级为对照臂**（门控趋于 identity 即回到同样坍塌的 backbone，故不可能单独修好 clean）；
- 执行顺序已严格完成：各 arm 先过全缓存 1-epoch 冒烟，再启动固定 seed 正式运行；最后的 H2-G 也已通过同一训练、评估、artifact 与 source 对齐比较器链路；
- `compare_tier2.py`：与封存 A5 逐 seed 配对，校验 source 顺序一致，输出预注册判定，不写回 composite。

2026-08-13 执行更新：全缓存冒烟 `tier2_smoke_coverage_v3` 已完整通过；H2-F1 正式 10-seed 条件覆盖运行已启动，产物目录为 `artifacts/tier2_condition_coverage_v1/`。当前仍是 exploratory Tier-2，不改写 sealed confirmatory family。

2026-08-14 结果更新：H2-F1 已完成 10/10 seeds。相对 sealed A5，clean-retention 仅 +0.15 pp，95% CI [−0.31, +0.63]，H2a 失败；hard-interference −0.30 pp，CI [−0.91, +0.27]，H2b 通过。SIR=−15 dB 上相对 MCLDNN +9.01 pp [6.55, 11.57]、相对 IQFormer +9.73 pp [5.94, 13.77]，H2c 通过；相对 A0 的 6 个冻结 OOD/receiver 轴全为正，H2d 通过。因 H2a 失败，联合门失败：**10% clean 条件覆盖几乎不改变模型，不能修复 clean 坍塌；证据从"偏向表示受限"推进为"单独训练覆盖解释已被实验排除"，下一步是 ≤80 k mixture-IQ sidecar 前端臂。**

H2-F2 执行更新：46,794 参数的 mixture-IQ sidecar 已通过全缓存 v2 冒烟（CUDA 训练、11 个评估 split、机制指标、execution context 和 source 对齐）。首次冒烟在最后机制评估发现包装模型缺少 `encode()` 转发，修正后以新 run-id 完整重跑通过；失败目录保留不覆盖。正式 10-seed 运行 `artifacts/tier2_iq_sidecar_v1/` 已启动。

2026-08-15 结果更新：H2-F2 正式 10/10 seeds 完成，**H2a–H2d 联合门全部通过**。相对 sealed A5，clean-retention +11.77 pp，95% CI [+11.06, +12.49]；hard-interference +4.37 pp，CI [+3.44, +5.31]。SIR=−15 dB 上相对 MCLDNN +11.37 pp [9.05, 13.70]、相对 IQFormer +12.09 pp [8.32, 15.88]；相对 A0 的六个 OOD/receiver 轴全部为正。这给出了可辩护的机制结论：**clean 坍塌不是单纯的条件覆盖不足，而是紧凑谱域表示缺乏星座信息；一条 46.8 k 的 received-IQ sidecar 可修复该缺口且不牺牲结构化干扰工作区。**

WP4 执行更新（2026-08-16）：M 档 v1 在初始化后被外部中断，未产生任何 seed 结果，不参与统计或结论。封存命令未改变，以新目录 `artifacts/tier2_capacity_M_v2/` 重启，独立进程与日志用于防止终端会话中断。

WP4 结果更新（2026-08-17）：M/L 均完成 5/5 seeds。实测参数量为 S=39.5 k、M=99,596、L=233,244；原预注册表中 `~110 k/~350 k` 是粗略目标，稿件只报实测值。M vs S 的 hard 增益为 +1.94 pp [1.07, 2.80]，L vs S 为 +3.18 pp [2.44, 3.96]，说明容量能解释一部分总体差距。但 L vs IQFormer 在 hard 仍为 −2.84 pp [−4.30, −1.31]，故 H1a 不支持、H1b 支持：**差距不能由参数量单独解释，存在架构/表示域差异**。在 SIR=−15 dB 严重角，L vs S +4.96 pp [3.48, 6.48]、L vs IQFormer +14.07 pp [9.55, 18.51]，H1c 通过；扩容未破坏严重干扰工作区优势。

WP5 启动更新（2026-08-17）：`presence_gate` 已接入独立 Tier-2 launcher，不改变标准 runner 或封存 artifact。单元前向/注册测试通过（S 档新增 `environment_dim+3=35` 个门控参数）；全缓存 1-epoch 冒烟 `tier2_smoke_presence_gate_v1` 完成 1/1 seed、11 个 evaluation bundle、checkpoint/teacher 与 execution context，比较器确认与 sealed A5 的 source 顺序完全对齐。冒烟数值仅验证管线，不作科学结论。预注册的 10 seeds × 30 epochs 正式 H2-G 已启动，输出 `artifacts/tier2_presence_gated_v1/`；预期为负结果，完成后必须公开报告。

WP5 结果更新（2026-08-18）：H2-G 已完成固定 10 seeds × 30 epochs。相对 sealed A5，clean-retention 为 −0.30 pp，95% CI [−0.66, +0.08]；hard-interference 为 −0.18 pp，CI [−0.72, +0.35]。clean 改善远低于预注册的 +2 pp 判定阈值，因此该对照**不修复 clean 边界**。独立门值账本显示平均 $g$ 在 clean-retention 为 99.994%，在 hard-interference 为 99.991%，两者均饱和到调制路：本次训练没有学出预期的 clean identity fallback。故 H2-G 是公开的负对照，排除了“仅改门控即可修复”的路径；其因果解释须与 A0 的 clean 坍塌及冻结探针共同读取，不能单独声称已验证一个成功的掩模回退机制。所有结果仍为 exploratory Tier-2，独立于 sealed confirmatory family。

## 12. 复现

```powershell
cd D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线\analysis_zero_compute
D:\Python\python.exe a1_snr_sir.py            # 可分批：a1_snr_sir.py id_test hard_interference
D:\Python\python.exe a1_snr_sir.py --finalize
D:\Python\python.exe a1b_snr_sir_map.py hard_interference unseen_jammer id_test
D:\Python\python.exe a1b_snr_sir_map.py --finalize
D:\Python\python.exe a2_strata.py ; D:\Python\python.exe a2_strata.py --finalize
D:\Python\python.exe a3_clean_retention.py
D:\Python\python.exe a4_selective.py
D:\Python\python.exe a5_complementarity.py
D:\Python\python.exe a6_mechanism.py
D:\Python\python.exe a7_learning_curve.py
D:\Python\python.exe a8_pareto_tost.py
D:\Python\python.exe make_provenance.py
```

全部脚本对冻结 artifact 只读；产物只写入 `analysis_zero_compute/outputs/`。

## 13. V4.1 稿件重分析（2026-08-19）

`a14_envelope_v41.py` 保留冻结预测的历史 32 个拟合单元（`id_test` 与
`hard_interference` 的 split $\times$ SIR $\times$ overlap-quartile；**不是**
8-SNR$\times$4-SIR 网格），并将主留出域改为 80 个有限-SIR 受干扰单元。
clean retention 不再进入该物理协变量模型，只作为 Tier-2 repair 边界。

- IQFormer-inspired：RMSE 6.23 pp，对 fit-domain 常数 8.37 pp，
  $R^2_{\mathrm{skill}}=0.45$，符号一致率 97.5%。
- MCLDNN：RMSE 8.10 pp，对 fit-domain 常数 9.63 pp，
  $R^2_{\mathrm{skill}}=0.29$，符号一致率 95.0%。
- 加入单元均值 SNR 仅将 IQFormer/MCLDNN RMSE 改为 6.15/8.05 pp；unseen-jammer
  与 combined-OOD 的技能弱，不能把 pooled 指标写成各域均匀有效。

同一五 seed（17, 29, 43, 71, 101）已从 sealed 与 sidecar 预测重算 Fig.4，
不再混用十 seed 主表与五 seed 容量点。产物：
`a14_v41_envelope_{interfered_holdout,sensitivity}.csv`、
`a14_v41_envelope_summary.json`、`a15_v41_pareto_matched.csv`。
