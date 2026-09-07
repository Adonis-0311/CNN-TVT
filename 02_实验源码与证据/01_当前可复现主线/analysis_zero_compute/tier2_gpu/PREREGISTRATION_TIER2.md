# Tier-2 预注册：不扩大仿真的两项补充训练研究

> 状态：**执行参数已冻结；项目所有者于 2026-08-13 通过当前 Codex 任务明确授权"资源充足，继续推进"；作者姓名签署栏仍留空，不代签**  
> 生成日期：2026-08-12；执行口径修正：2026-08-13  
> 数据边界：**不生成任何新仿真数据**。全部使用既有冻结缓存 `standards/cache_factor_headline_1024_v2/`（digest `f2003d4b...c0697d80`）。  
> 证据边界：本文件定义的全部研究均为 **exploratory / post-hoc**，位于冻结确认性家族之外。产物必须写入**新目录**，禁止并入 `artifacts/tvt_v4r_headline_composite/`，禁止修改任何 sealed artifact。

## 0. 为什么需要这两项

零算力分析（`analysis_zero_compute/RESULTS.md`）确认了三件事：

1. A5 在**严重干扰角**（SIR = −15 dB）显著优于两个强基线（vs MCLDNN +9.74 pp，vs IQFormer +10.46 pp，CI 均严格为正），但在良性条件下全面落后；
2. clean-retention 的 −9.69 pp **几乎全部来自 QPSK/16QAM/64QAM 三类在无干扰条件下的坍塌**；且该坍塌在 **A0–A7 全部九个模型上完全一致**（recall < 0.01），三个外部基线不坍塌——即缺陷位于共享谱域前端，**不是 VIMD 机制引入的**；
3. A5 与强基线**决策互补**：A5+IQFormer 概率融合在 hard interference 上比 IQFormer 单体高 +4.71 pp，而"同架构双种子集成"对照只有 +1.49 pp。

由此产生两个审稿人必问、且当前证据回答不了的问题：

- **Q1（容量混淆）**：A5 的绝对精度差距是否只是参数量差 9 倍造成的？
- **Q2（可修复性）**：共享前端在无干扰条件下丢失星座阶数分辨力，是"表示里有信息但分类头用不上"，还是"表示里根本没有"？两者对应完全不同的修复成本。

## 1. 研究 H1：容量阶梯（回答 Q1）

### 设计

在**同一冻结缓存**上训练 A5 的容量阶梯，仅改变既有 CLI 已暴露的宽度参数，不新增模型代码：

| 档位 | spectral-channels | embedding-dim | environment-dim | 目标参数量 |
|---|---|---|---|---|
| S（已有，等于 A5） | 24 | 48 | 32 | 39.5 k |
| M | 40 | 80 | 48 | ~110 k |
| L | 64 | 128 | 64 | ~350 k |

seeds：`17,29,43,71,101`（5 个，成本折中；不得事后追加到 10 个再挑选）。  
其余超参数与 V4 冻结一致（30 epochs 上限、batch 16、lr 3e-4、wd 0.01、patience 8、AMP、1024 输入）。

### 预注册假设与判定

- **H1a**：在 hard interference 上，L 档 macro-F1 ≥ IQFormer（0.4872）。  
  判定：5 seed 均值差的配对 bootstrap 95% CI 下界 > 0 → 支持"容量是主要差距来源"。
- **H1b**：若 L 档仍显著低于 IQFormer（CI 上界 < 0），则记为**架构差距**，论文必须直接写明"差距不能由容量解释"，不得回避。
- **H1c（次要）**：severe-interference corner（SIR = −15 dB）上 S 档相对 L 档的优势不劣化超过 2 pp，用于支持"紧凑模型在该角落不是靠欠拟合取胜"。

两种结果都可发表；H1b 为负结果时，论文定位仍成立（Pareto 前沿 + 严重干扰角），但必须删掉任何"容量受限"暗示。

## 2. 研究 H2：定位并修复 clean 条件下的星座阶数坍塌（回答 Q2）

按成本从低到高分三步，**H2-D 必须先做**，其结果决定 H2-F 的形态。

### H2-D：冻结表示的探针（几分钟，无训练）

**首轮（`clean_probe.py`，`artifacts/tier2_clean_probe_v1/`）作废**：其训练输入为 `x − jammer`。封存分量与 `x` 共享 post-AGC 尺度，但去干扰后未重新归一化，窗口功率落在 0.046–0.50 而非契约的 0.5。判定被输入尺度混淆，不得采用（审计见 `analysis_zero_compute/a10_counterfactual_check.py`）。

**改用 `clean_probe_v2.py`**，两种设计都要跑：

- `--design counterfactual`：训练输入为 `AGC(clean + noise + receiver_artifact)`，即同一生成管线去掉干扰项。混合律 `x = AGC(clean+jammer+noise+receiver_artifact)` 已逐窗验证（最大相对误差 2.3e−7）。属已封存分量重混，不是新仿真。
- `--design cv`：完全不合成，在真实 clean_retention 窗口上做 source-disjoint 5 折交叉验证。因其在测试集来源上拟合，**仅可作表示诊断，禁止作为性能数字引用**。

每种设计跑 A5 与 A0 × seeds 17/29/43，并对 CSSL / MCLDNN / IQFormer 取参照上界（三者均暴露 `embedding`）。

- **判定**：MLP 探针在 QPSK/16QAM/64QAM 上的 F1 均值 > 0.25 → `information_present_head_limited`；否则 → `representation_limited`。
- 两种设计、两个模型、三个 seed 的结论必须一致才可用；不一致则如实报告为不确定，并以更保守的一侧写入论文。

### H2-F：条件修复（形态由 H2-D 决定）

前提事实（`a9_condition_coverage.py`）：按训练实际使用的两个 view 重算，PI2BPSK、8PSK、256QAM、CPFSK 四类存在无干扰训练窗，其余 6 类在训练中从未以无干扰形式出现，而 clean_retention 对全部 10 类求值。旧 2/10 口径只统计 view1，已撤回。因此必须将**训练条件覆盖**与**表示域**分开检验。

- **H2-F1（条件覆盖，首选）**：用 `AGC(clean + noise + receiver_artifact)` 从**已封存分量重混**出全部 10 类的无干扰训练窗口，按预注册比例（建议占训练批次 10%）混入训练。必须在论文与产物中标注为 "counterfactual re-mixing of sealed components; no new channel realisation, no new source sequence, no MATLAB call"，并附混合律验证（最大相对误差 2.3e−7）。**此臂改变了训练分布，因此其结果与 V4 正式证据不可直接同表并列，必须单列。**
- **H2-F2（若 H2-D 判定 `representation_limited` 且 H2-F1 不足）**：前端修订——为紧凑 backbone 增加一条轻量幅度/相位统计或 IQ 侧支路（可复用 `src/vimd_amc/models/temporal_vimd.py` 的既有结构），预算约束为总参数量 ≤ 80 k，否则 Pareto 主张失效。
- **H2-F0（零成本对照，必须报告）**：把"训练从未见过 6/10 类的无干扰条件"本身写进论文局限性。即使 H2-F1/F2 都不做，这一条也必须写；否则读者会误以为 clean-retention 失败是方法缺陷。

seeds：`17,29,43,71,101,131,173,211,257,307`（10 个，与正式运行一致，便于与 sealed A5 逐 seed 配对）。

### H2-G：干扰存在性门控（对照臂，非主修复）

`diagnostic_vimd_v5_presence_gated`：`w' = g·(m + λ·o + ρ) + (1 − g)·1`，新增约 35 个参数。

**预期为负**：`g→0` 时模型退化为 backbone 路径，而 backbone 在 clean 上同样坍塌（A0 的三类 recall < 0.01），因此门控**不可能单独修好 clean**。保留该臂的唯一目的，是验证"掩模并非 clean 退化主因"这一归因，并检查门控是否在 jammed/clean 之间带来更好的折中。若其 clean 增益 < +2 pp，即为归因正确的正面证据，必须如实报告，不得当作失败隐藏。

### 预注册判定（H2-F 的联合门，必须同时满足）

- **H2a（补救有效）**：clean_retention macro-F1，新变体 − sealed A5 ≥ +6 pp，且分层配对 95% CI 下界 > 0；
- **H2b（不牺牲主效应）**：hard_interference macro-F1，95% CI 下界 > −1 pp（非劣性，margin 1 pp）；
- **H2c（不牺牲严重干扰角）**：SIR = −15 dB 子集上相对两个强基线仍显著为正；
- **H2d（不破坏 backbone-relative 结论）**：相对 A0 在 6 个冻结 OOD/receiver 轴上全部保持为正。

四条全过 → 可写"该缺陷已定位并修复，代价是 X"；任意一条不过 → 必须写成"补救尝试及其失败/部分成功"，**不得**改判定阈值、不得只报 clean 而不报 hard、不得把失败的补救从论文里删掉。

### 附加记录（非判定）

- 逐类 F1（重点 QPSK/16QAM/64QAM）与 clean 上的预测类别分布熵；
- H2-G 的门控值 `g` 在 clean / jammed 子集上的分布。

## 3. 执行与产物约束

- 产物目录：`artifacts/tier2_condition_coverage_v1/`、`artifacts/tier2_iq_sidecar_v1/`、`artifacts/tier2_capacity_M_v1/`、`artifacts/tier2_capacity_L_v1/`、`artifacts/tier2_presence_gated_v1/`；全部为**新目录**；
- 每次运行前后记录 `git status --short`，不得 `git reset/checkout/clean`；
- 不得改动：`tvt_submission/configs/formal_tvt_freeze_v4_8gb_dual.json`、`formal_tvt_recovery_v4r_110plus10.json`、`artifacts/tvt_headline_1024_10seed_v4_8gb_dual/`、`artifacts/tvt_v4r_cssl_recovery_workers/`、`artifacts/tvt_v4r_headline_composite/`、`standards/cache_factor_headline_1024_v2/`；
- 与 sealed A5 的配对比较通过 seed 对齐完成，禁止把新 fit 写入 composite 的 `models/`；
- 论文中所有 Tier-2 数字必须带 "exploratory, outside the frozen confirmatory family" 标注。

## 4. 明确禁止

- 不做补种子或换 primary metric 以求通过；
- 不调整 clean-retention 阈值；
- 不删除失败的补救分支或较强基线；
- 不把 Tier-2 结果与 V4/V4R 证据混排在同一张"正式结果"表内而不加区分；
- 不以 Tier-2 结果重述已封存的 confirmatory family 结论。

## 5. 签署

| 角色 | 姓名 | 日期 | 冻结前 SHA-256 |
|---|---|---|---|
| 作者 |  |  |  |
| 复核 |  |  |  |

## 6. 2026-08-13 执行附记（结果前固定）

- WP2 v2 已完成；A5 在 6 个设计—seed 单元中 5 个低于 0.25，1 个临界越线，按本文的一致性规则以更保守一侧记为"阈值边界、整体偏向 representation-limited"。
- 全缓存 1-epoch 冒烟 `tier2_smoke_coverage_v3` 已通过：CUDA 训练、checkpoint/teacher、11 个 evaluation prediction bundle、`run.json=complete`、Tier-2 execution context 和比较器 source 对齐全部通过。冒烟指标不作科学结论。
- H2-F1 固定为：10 seeds `17,29,43,71,101,131,173,211,257,307`；30 epochs；batch 16；lr 3e-4；weight decay 0.01；patience 8；AMP；每类精确 10% 的 `AGC(clean+noise+receiver_artifact)` 重混覆盖；输出 `artifacts/tier2_condition_coverage_v1/`。
- 由于根 manifest 约 1.95 GB，Tier-2 包装器仅在内存中让各 split 共享已审计 manifest 对象，避免 12 次重复 JSON 解析；不改缓存、标准 runner 或 sealed artifact。
