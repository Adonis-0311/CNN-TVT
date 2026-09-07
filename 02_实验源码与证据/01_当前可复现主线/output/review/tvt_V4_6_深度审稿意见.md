# 《Rank Instability and Representation Repair for Automatic Modulation Classification under Structured Interference》深度审稿意见（V4.6）

- 审阅对象：`tvt_rank_instability_representation_repair_V4_6.pdf`（主稿，IEEEtran，10 页）+ `tvt_supporting_material_V4_6.pdf`（补充材料，11 页）。
- 审阅方式：核对两份 PDF 的 LaTeX 源（`main.tex`、`supplement.tex`）与全部已解析数值宏（`paper_macros.tex`、`v1_audit_macros.tex`、`v41_envelope_macros.tex`）；抽验了文中关键算术（STFT 帧数、Doppler 频移、相干时间、参数倍数、融合增量、MAC 增幅等）；确认文中引用的图件与 artifact 文件真实存在。
- 证据边界：全部结论基于 3GPP TR 38.901 TDL 信道仿真，无 OTA、实际上板或在轨证据。复现包（代码/数据）承诺录用后释放，本次未审。

---

## 一、结论与建议

**建议：大修（Major Revision），不接受现稿直接录用，但也未到拒稿程度。**

稿件的问题不在单个致命错误，而在三类叠加的短板：

1. 若干**头条数字内部不自洽**、且部分承重实验的**构造与注册信息缺失**，导致关键结论目前无法从所供材料核实；
2. 唯一强正向结果（sidecar 修复）的**归因未在声明所在的评估域上受控**；
3. 论文把自己框定为"排序不稳定 + 前瞻证伪 + 表示修复"，但**新颖性定位与重要性措辞都高于所演示的规模**，且对圈外读者的价值从未被说清。

下面按"数字一致性 → 方法归因 → 定位与表达 → 次要问题"的顺序给出，均给出具体位置与可执行修改。文中不逐条复述"透明、严谨"之类的定性——这类自我定性对录用决策没有增量信息，且与后文暴露的不一致放在一起反而削弱可信度。

---

## 二、必须解决的核心问题（Major）

### M1　两个头条"pooled"增益彼此对不上（数字不自洽）

- 位置：主稿 V-F、摘要；补充材料 S11 与 tab:s5a。
- 事实：迁移章节报"独立 severe-held 集 pooled A5 增益对 IQFormer-inspired 为 **−3.12 pp**、对 MCLDNN 为 **−12.45 pp**"；修复章节给出的同一批模型的 pooled macro-F1 level 为 A5=20.72%、IQFormer-inspired=23.29%、MCLDNN=33.75%。由 level 直接相减得到 A5−IQFormer = **−2.57 pp**、A5−MCLDNN = **−13.03 pp**，与迁移章节的 −3.12 / −12.45 分别相差 **约 0.55 / 0.58 pp**，而补充材料明确声明两检查共用同一锁定缓存与同一批 A5/IQFormer/MCLDNN 预测。
- 影响：这是审稿人一眼能抓到、且最难反驳的一类问题——两个声称度量"同一比较、同一数据"的头条数字不一致，且没有任何聚合规则的说明。它直接动摇"所有数字由 artifact 机械生成、可追溯"这一全文反复强调的主张。
- 要求：逐条写清两个数字各自的聚合定义（是"窗口级 pooled macro-F1"还是"逐 seed 配对差的等权均值"，macro-F1 是非线性量，两种池化不相等），并从逐 seed/逐 SNR 表证明两者同源；对账或修正其一，并在正文给出不依赖读者自行反推的口径声明。

### M2　sidecar 的"表示修复"归因未在 severe-held 集上受控

- 位置：主稿 V-E、V-F；补充材料 S5（容量阶梯仅战役划分）、S11（severe-held 修复）、tab:sidecar。
- 事实：用于分离"表示 vs 容量"的 S/M/L 宽度阶梯只在战役 hard 划分上测得（0.436/0.455/0.468，五 seed 匹配子集），从未在 severe-held 集上评估。而 4.37 pp 修复增益是在 severe-held 集上测得的。即"容量不是主因"这一前提是在 A 域验证、"修复来自 I/Q 信息"这一结论是在 B 域得出。此外 sidecar 相对 A5 不只是加了 I/Q 分支，还新增第二个分类头与残差 logit 融合（tab:sidecar 第 11–13 层），却无"等参数谱分支 / 空 I/Q 输入"的消融。
- 影响：稿件三大贡献之一、也是唯一存活的强正向结果，目前只能支撑"一个更大的混合模型在困难 held 集上有帮助"，支撑不了"恢复 received-I/Q 信息"这一更具体、也更值钱的解释。稿件自己的宽度数据还指向反方向：容量在紧凑模型本已占优的 regime（战役内 SIR=−15 dB，L 对 S +4.96 pp）收益最大，恰是 severe-held 瞄准的 regime。
- 要求：在 severe-held 集上补 S/M/L 容量档与"零 I/Q 输入"对照（同架构、I/Q 分支喂空或仅幅度），并报告 pooled macro-F1 对照 sidecar 的 25.09%。若容量档或空输入对照恢复相近增益，归因失败，需相应改写结论。

### M3　"预注册"这一承重声明无法核实

- 位置：主稿 IV-D、V-F；补充材料 S11–S12。
- 事实：全文反复以"preregistered / frozen before inference"作为把两项 severe-held 验证与事后分析区分开的关键，但审阅包内**没有任何预注册文档、时间戳、内容哈希或逐字引用的判定规则**。迁移证伪的判据（何种 pooled-gain 或 sign-accuracy 构成"证伪"）、修复检查的主对比与池化方式、clean-retention gate 的规则，均只有转述、没有原文。
- 影响：若注册规则与冻结时序不可验证，"预注册证伪"这一核心贡献即不成立，两项验证退化为"看到结果后反向设计的事后检查"——这正是稿件自己反复贬低的证据等级。
- 要求：提供带时间戳与哈希的预注册文件、逐字引用判定规则，并证明报告的结论机械地遵循这些规则；若拿不出，就把两项验证改标为 post-hoc，并相应下调"证伪/修复"的强度措辞。

### M4　severe-held 集构成缺失，证伪结论无法定位

- 位置：主稿 V-F；补充材料 S11（仅"12,820 新源、两个 severe regime、view 0、八 SNR 层"）。
- 事实：未给出 severe-held 集的 SIR 层、SNR 网格、干扰族、速度、信道剖面、逐层窗口数与 overlap/SIR 分布。因此无法判断"两个 severe regime"是否包含 −15 dB（即战役内反转与 break-even 阈值所针对的层），也无法判断迁移失败是"公平检验"还是"测了另一个条件"。
- 影响：证伪与修复结论只有相对 held 集对所声称条件的覆盖才可解读；现在这一步缺位。
- 要求：给出与补充材料 tab:cells 同构的 held 集 regime 表，并明确 SIR=−15 dB 是否出现在 held 集中、占比多少。

### M5　两个强对比器存在预算受限风险，且 sidecar 自身训练史未审计

- 位置：主稿 V-B、V-F；补充材料 tab:convergence。
- 事实：IQFormer-inspired 与 CSSL 在 10/10 seed 中均于第 24–30 epoch 选中 checkpoint（即 30-epoch 上限处验证损失仍在下降），其水平是下界；无扩展预算或平台期敏感性。sidecar 自身的逐 seed 选中 epoch 与早停行为不在审计表内。于是"sidecar 反超 IQFormer-inspired"与"战役内 −15 dB 反转"都可能受 IQFormer-inspired 欠训影响。
- 影响：边际排序与 severe-held 对比的解读建立在"共享预算即公平"上，而审计表自己暴露了这一假设对 IQFormer-inspired/CSSL 不成立。
- 要求：在匹配子集上以扩展预算训练 IQFormer-inspired 与 CSSL，报告边际排序、−15 dB 反转、severe-held 对比是否改变；把 sidecar 的训练史补入收敛审计。

### M6　"rank instability"的新颖性未与通用文献区分

- 位置：主稿 II-C（自承"positioning claim, not an exhaustive systematic review"）。
- 事实：论文只对照 AMC 的"平均主导性 / 普适鲁棒性"文献，未触及"分布漂移下模型排序不稳定、选择不稳定、泛化失败"这一早已存在的通用现象。读者无法判断真正新颖的是现象本身、AMC 特化演示、预注册证伪协议、还是修复链。而唯一强正向交付物 sidecar，在架构上是常规的多流加法（I/Q 分支 + 谱主干），其新颖性落在"诊断+修复"叙事而非架构，且该叙事在 Limitations 里自承非因果。
- 影响：若排序不稳定只是已知效应在 AMC 上的又一次演示，增量贡献就是"一个带证伪协议的有界案例研究"，远小于标题与摘要暗示的分量。
- 要求：在 II 明确写清在 AMC 内何为新的（如系统级 cell 排序图、对自身战役解释的预注册拒绝、修复链），并给出把"表示特化结构"与"数据噪声"区分开的证据（如跨第二个仿真器或模型族验证排序模式是否稳定）。

### M7　重要性措辞高于所演示的规模

- 位置：摘要（"a more consequential failure mode"）、V-F（"strong representation repair"）、VI（"substantial independent OOD degradation"）。
- 事实：按论文自己的数字，诊断包络的实用价值有限——overlap+SIR 选择器在 80 个 held 单元里只选 A5 5 次、oracle 遗憾 0.36 pp、仅 5.5% 的 MAC 节省、realized macro-F1 0.5872 还低于 always-IQFormer 的 0.5894；overlap 相对 SIR 的增量其配对 bootstrap 区间含零；overlap 变量在非合作接收端不可直接观测；包络在独立集上不迁移。修复效应本身在 20.72%→25.09% 的低工作点上，且仍低于 MCLDNN 8.66 pp、只恢复至 MCLDNN 差距的约 34%。
- 影响：这些数字说明的是一个**领域局部、中等强度、且相当一部分为负向结论**的研究。"consequential / substantial / strong"这类措辞与自己的表格直接冲突。
- 要求：要么量化一个具体后果（如计算受限选择策略下的期望 macro-F1 或误分类代价 vs always-best），要么把措辞降为与数字匹配的"有界、局部、中等"表述。摘要与标题应反映正文已承认的有界范围。

### M8　跨界读者与非专业可读性未交付

- 位置：摘要、I、VI、VII、Fig.2 caption。
- 事实：全文未点名 AMC/V2X 之外的任何受众，也未以通用语言陈述可迁移教训（"边际基准掩盖排序翻转""战役内诊断需独立验证""小表示改动修复大 OOD 退化"）。I/Q、STFT、SIR、SNR、TDL、macro-F1、MCLDNN 等核心术语全文不拼写；A0/A5 承担摘要头条对比却不在摘要前引入；"sealed/frozen/preregistered"承载整套证据架构却无通俗注解；IV-E 的 bootstrap 描述是一句无注解的统计墙；Fig.2 caption 要求读者先懂"positive unadjusted paired lower interval"与"Holm-surviving cells"。
- 影响：底层信息其实一句话能说清（最好模型随干扰条件变化、单一战役学到的规则在全新重干扰数据上失败、恢复少量原始接收信号修复大部分损伤），但这句话从未出现。对非专业读者，摘要+结论读完后无法复述核心发现。
- 要求：首次使用处展开所有术语；加一段平实语言陈述"做了什么、一句话要点"；给 sealed/frozen/preregistered 与 bootstrap 各一句注解；用平实语言改写 Fig.2 星号语义。

---

## 三、次要问题（Minor）

1. **两处数值巧合未说明**：sidecar 修复在战役内 hard 与 severe-held 上均为 **+4.37 pp**（区间不同，3.44–5.31 vs 2.96–5.90）；两个参照的 sign-flip 置换概率同为 **0.0018**。需注明独立计算或报第三位小数，否则会被读作复制错误。（对应原 R2-m3/R3-m4、R1-m1/R3-m1）
2. **"resolved" 与家族 gate 失败的措辞冲突**：A5–A0 被报为"resolved"，同时联合家族 gate 未通过。需加一句解释"同一 simultaneous band 的成员在联合 gate 失败后仍受控"，并统一"survives the joint band"与"resolved"的用词。（原 R3-m2）
3. **单一 severe-held 缓存、无第二独立集**：证伪与修复两项头条都只基于 12,820 窗口的单一缓存，修复还复用了迁移检查的三组模型预测。应明确"第二独立集复现"尚缺，并复核"strong repair"措辞。（原 R3-m3）
4. **severe-held RMSE 是 held 单元上的重拟合**：迁移章节的包络 RMSE 与战役包络的 out-of-sample 评估口径不同，正文未说明。需说明重拟合并解释其为何仍能服务证伪论证。（原 R1-m2）
5. **包络未加权拟合**：单元窗口数 229–1053 不等，未加权最小二乘的权重选择未讨论。做一次窗口加权敏感性或说明理由。（原 R1-m3）
6. **27/27 与 1/9 的 cell 计数不可重构**：正文只给 6 个未覆盖类，未给"模型 × 类 × 条件"网格。补网格定义。（原 R1-m4）
7. **源计数漂移**：主稿"12,820 examples" vs 补充材料"12,820 sources"；补充材料"战役 152,000 源" vs 主稿宏"100,000 训练源序列"。三者关系需一段话对账。（原 R1-m5）
8. **五 seed 子集"前五个"的选择规则与敏感性**未给。补理由，并说明其他固定五 seed 子集上结论是否成立。（原 R1-m6）
9. **probe 反事实的重构过程**（"correctly reconstructed jammer-free counterfactual"）未描述、未验证。说明是仿真跟踪真值还是重构信号，并给验证。（原 R1-m7）
10. **交叉引用错误**：V-C 承诺"we return to this reversal in Section VI-C"，实际反转在 VI-B 重访。改为 VI-B。（原 R1-m8，已复核为真实错误）
11. **gap-recovery ratio 未定义**：对 IQFormer 出现 1.70（>1），需定义并说明 sidecar 超过参照水平时会出现 >1。（原 R2-m4）
12. **probe 阈值的临界敏感性未带入讨论**：5/6 单元低于 0.25、唯一高于者仅 0.256、阈值 0.20 即翻转，但讨论以"leans toward missing constellation information"作结。重述其阈值敏感、属佐证而非决定性。（原 R2-m5）
13. **预算受限 caveat 未进 V-B**：IQFormer-inspired 反转（+10.46 pp）的解读应就地带上预算充分性 caveat，而不是只在 Limitations 出现。（原 R2-m6）

---

## 四、数字与一致性核查记录（审稿人亲算）

以下为本次核对中亲手验算、且与正文一致的部分，说明论文在"能直接算账"的地方大体站得住；问题集中在"不能直接算账"的聚合口径上（见 M1）：

- A5–A0 全配置增益 4.57 pp [3.65, 5.49]，约为设计分辨率 1.31 pp 的 3.5 倍（4.57/1.31 ≈ 3.49），一致。
- STFT：N_FFT=64、hop=16、Hann 窗长 64、无补零无居中，1024 采样 → (1024−64)/16+1 = **61 帧**，与正文 64×61 一致。
- Doppler：f_c=5.9 GHz，150 km/h → ≈820 Hz、250 km/h → ≈1367 Hz，与正文一致；T_c≈0.423/f_D 在 250 km/h 下 ≈0.31 ms，为 1024 μs 决策窗的约 30%（180 km/h 约 42%），一致。
- 参数倍数：A5/A0 = 39500/8458 ≈ 4.67×，与"约 4.7×"一致；sidecar 39500→46794（+7294，+18.5%），MAC 41.8→43.1 M（+3.1%），与"约 3% MACs"一致。
- 融合：4.71/5.70 − 1.49/1.69 = 3.22/4.01，与正文 excess 一致。
- 公共基准：IQFormer-inspired 65.23% − 参考 64.19% = +1.04 pp；MCLDNN 58.55% − 62.05% = −3.50 pp，一致。
- clean 迁移：27/27=100%、1/9≈11%，一致。

**不一致或未解释处**：见 M1（−3.12/−12.45 vs −2.57/−13.03）、次要问题 1（4.37 与 0.0018 双现）、次要问题 7（源计数三处漂移）、次要问题 10（交叉引用）。另附两点构建层面的提示：生成的 `paper_macros.tex` 仍保留 81-cell/6.20/8.40/0.46 的旧包络值，主稿靠 `v41_envelope_macros.tex` 覆盖为 80-cell/6.23/8.37/0.45，而补充材料的包络表是硬编码 80-cell 值、且只 `\input` 了 `paper_macros.tex`——当前 PDF 无可见冲突，但属未来重生成时的静默分歧风险；`v1_audit_macros.tex` 中 `\EnvelopeHoldoutNegShareIQ` 的"78/81"注释相对 80-cell 重定义已过时（数值 0.963 实际与 77/80 一致）。

---

## 五、值得保留之处（简述）

- 负向结果没有被藏起来：联合家族 gate 未通过、clean-retention gate 未通过、severe-held 迁移证伪，都在摘要与正文显著位置出现。这一点对录用是加分项，但不必作为卖点反复渲染。
- 对照设计是可用的：公共基准复现（E1）、收敛审计、seed 固定块配对 bootstrap 与 seed 随机化变体、leave-−15-dB-out、clean-sentinel 敏感性、逐 seed/逐 SNR 报告，构成一个能支撑"负向结论可信"的基础。
- 溯源索引是真实工作量：冻结缓存、SHA-256、checkpoint 审计、artifact 路径索引，在复现性上比多数同类稿扎实。

一句话：**可信的是"负向结论 + 一个中等强度的修复效应"；不可信的是围绕它叠加的"排序不稳定作为新范式"的定位与"substantial/strong"的重要性措辞。** 修改应把力气放在 M1–M8，而不是继续扩充证据分级与自我限界的篇幅。

---

## 六、作者逐条答复清单（建议顺序）

1. 对账 M1 的两个 pooled 增益，给出各自聚合公式并修正其一（或解释差值来源）。
2. 补 severe-held 上的容量档与"零 I/Q 输入"消融（M2），并据此改写或保留"表示修复"结论。
3. 提供预注册文件 + 时间戳 + 判定规则原文，或降级为 post-hoc（M3）。
4. 补 severe-held 集 regime 表（SIR/SNR 层、逐层窗口数、overlap/SIR 分布，并说明 −15 dB 是否在集内）（M4）。
5. 扩展预算重训 IQFormer-inspired/CSSL 的敏感性 + sidecar 训练史入审计（M5）。
6. 修订 II 的 novelty 定位，对照通用排序不稳定文献（M6）。
7. 校准重要性措辞或量化具体后果（M7）。
8. 术语首用展开 + 平实语言一段 + Fig.2 caption 改写（M8）。
9. 逐条处理第三、四节的 minor 与数字问题（尤其 M1 之外的两处巧合、源计数、交叉引用）。
