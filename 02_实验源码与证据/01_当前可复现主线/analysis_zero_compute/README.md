# analysis_zero_compute

对已封存 TVT V4R 证据的**零算力再分析**：不生成新仿真数据、不重训、不修改任何冻结 artifact。

- 结果与可写/不可写边界：`RESULTS.md`
- 溯源与自校验：`outputs/provenance.json`（含所有输入/产物 SHA-256，以及与 sealed headline 数字的逐位一致性校验）
- 表格：`outputs/csv/`（`parts/` 为分批运行的分片，可安全删除后重跑）
- 图：`outputs/figures/`（PDF + PNG）
- 需算力但不新增仿真数据的后续研究：`tier2_gpu/`（预注册 + 门控模型 + 运行脚本，**需先在本机 CUDA 跑冒烟**）

模块：

| 脚本 | 内容 |
|---|---|
| `zc_core.py` | 加载、指标、分层配对 hierarchical bootstrap（与 `src/vimd_amc/metrics.py` 同口径） |
| `a1_snr_sir.py` | SNR/SIR 分辨的绝对值与配对差 |
| `a1b_snr_sir_map.py` | SNR×SIR 联合工作区图 |
| `a2_strata.py` | 干扰族 / 信道 profile / 速度 / 频谱占用分层 |
| `a3_clean_retention.py` | clean-retention 缺口的逐类、逐 profile、混淆结构解剖 |
| `a4_selective.py` | validation-only 温度标定与 risk–coverage |
| `a5_complementarity.py` | 错误重叠、概率级融合与"同架构双种子集成"对照 |
| `a6_mechanism.py` | 机制量—增益关联与 occupancy 假设再分析 |
| `a7_learning_curve.py` | 样本效率与数据等效倍数 |
| `a8_pareto_tost.py` | 精度—成本 Pareto 与探索性非劣性 |
| `make_provenance.py` | 溯源与估计量自校验 |

约束：所有产物只写入 `outputs/`；`artifacts/tvt_v4r_headline_composite/`、`artifacts/tvt_headline_1024_10seed_v4_8gb_dual/`、`artifacts/tvt_v4r_cssl_recovery_workers/`、`standards/cache_factor_headline_1024_v2/` 与 `tvt_submission/configs/` 均为只读。
