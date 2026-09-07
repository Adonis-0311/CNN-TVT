# TCCN 治理机制复用审计（TVT / Vehicular AMC）

## 1. 审计结论

本次工作只读审计了
`D:\CNN信号调制识别\tccn_satellite_amc` 中与实验治理有关的源码、配置、
文档和测试，并把可验证、与场景无关的机制改写到
`tvt_submission/tccn_reuse/`。

审计期间：

- 未枚举、读取、写入或以任何方式访问源仓库的 `artifacts/`；
- 未启动、停止、查询或修改 Pilot-1 训练进程；
- 未运行源仓库代码或测试；
- 未复制源项目的数据、模型权重、运行状态、实验数值或结论；
- 未把卫星链路、轨道/过境身份、APSK 制式或卫星 RF 假设移入 TVT 工程；
- 新增代码不启动训练，只实现冻结、QA、统计治理和发布门禁。

本次“采用”表示采用治理不变量并针对 vehicular AMC 重新实现，不表示源文件的
逐字复制。源仓库当前或未来的任何 Pilot-1 数值都不能因本次复用而成为 TVT 证据。

## 2. 来源逐文件 SHA-256、改造与取舍

SHA-256 均在 2026-07-28 对下列明确路径只读计算。路径相对于
`D:\CNN信号调制识别\tccn_satellite_amc`。

| 来源文件 | 来源 SHA-256 | 决策 | 改造、采用或拒绝理由 |
|---|---|---|---|
| `configs/pilot1_training_freeze_v1.json` | `eb2f06c392b261ff5fba31db5b80c40e73282f773aaba88d1cc0dea8ec7e3d4c` | 部分采用 | 采用“结果前冻结”、固定 seed、角色隔离、配置哈希的结构；拒绝全部卫星制式、链路参数、文件路径和 Pilot-1 状态。生成的只是 `template_not_frozen` 模板，不伪装成已冻结实验。 |
| `configs/d0_pilot_design.json` | `8534136a9ccfb767f0ea1ad09cb06e4328965868e6e5e3c89c89416a0b5d161b` | 部分采用 | 采用冻结的完全因子单元计数与期望集合闭包；拒绝卫星场景因子和值。vehicular 版本让调用者显式提供因子字段与期望单元。 |
| `scripts/audit_repository.py` | `a9573fa1a952fff945f1052d1dd4a3e84601bc6bb3aa5e515c315bd50aa2506d` | 采用并缩减 | 采用 JSON 可解析、CSV 主键唯一、受控枚举及发布前仓库审计思想；实现最小审计器并增加外来场景语义防火墙。未移植论文元数据的项目特定文件名。 |
| `src/satamc/pilot_experiment.py` | `11f599de430f551b3ec0bacd2e4580a47afff93450d676f963db037bdb50cf78` | 采用核心治理 | 采用 canonical JSON、create-new 原子发布、配置/产物 SHA 闭包、相对路径约束、完成态 manifest、git dirty 记录、校准隔离、paired cluster 约束。未移植模型、数据加载、训练循环、超参数或任何结果。 |
| `src/satamc/d0_qa.py` | `9f142f8e8a86fb8c4462b30e355b25649d1717e2c1cda5b427fbc7803ba0b99c` | 采用不变量 | 采用样本 ID 唯一、group 跨 split 不相交、冻结单元精确计数、分量重构、实测 SNR/SIR、payload 哈希。身份字段改为调用者给定，默认 `source_id`。未移植源场景字段和数据布局。 |
| `src/satamc/hdf5_v2.py` | `1ed846cad68faad7f4e3949f001d854d250e6d556f6ca7e093ebaef6e49b05b3` | 拒绝直接移植 | HDF5 schema 和身份字段与源场景强耦合。只吸收“manifest—shard—checksum 闭包”的原则；TVT 当前数据布局不因本审计被重写。 |
| `src/satamc/leakage_v2.py` | `f620924c47836124abf3f2325dec1a68866adca358bbad432733f5908175c5c9` | 部分采用 | 采用按组身份隔离和精确 payload 摘要；拒绝源场景组键。近重复 SimHash/相关系数方案保留为后续扩展要求，本次最小实现不宣称已覆盖它。 |
| `src/satamc/decision.py` | `b523b00188270914a90ae4d030965b858493eca4d2c5d3ebf80da3dffc6e88f5` | 拒绝 | 顺序认知决策器不是当前 VIMD-Net 调制识别论证的一部分，移植会扩大论文方法范围且没有 TVT 证据，故不进入代码或 claim。 |
| `governance/CLAIM_EVIDENCE_LEDGER.csv` | `f2adb82fae0cd7f017b06cfa35a083a6e86548a8f42ef7761096d9f0eaf57674` | 采用 schema 思想 | 采用 claim ID 唯一、状态受控、`supported` 必须指向证据产物；拒绝源项目的具体 claim 和状态。 |
| `governance/SUBMISSION_READINESS_GATES.md` | `ad6479b6b26e03cbf24078295fe3c883da969db79e61b2da15251e860e43b805` | 采用 | 把“缺证据即禁止发布”落为可执行 `assess_release`，而非文字清单；所有 QA 和统计控制必须显式为真。 |
| `governance/PILOT1_PREREGISTRATION.md` | `07ce4ecb1feb09fd67e37fb08ed92abb693894c1526766020c86084b81de41a0` | 部分采用 | 采用 3-seed screening、5-seed headline、固定假设族、负控/clean guard 的治理方向；实现了 seed 深度、校准隔离和预注册族检查。拒绝 Pilot-1 的具体方法、阈值和结论。 |
| `governance/D0_FREEZE_GATE.md` | `636a85f7d79a1c8d5805a91c5836fe83b4e925b84f84a3a4baa8dcf03d62c612` | 采用 | 采用 clean data manifest、角色/组隔离、完整 QA 才能解锁下游的 fail-closed 原则；未复制源项目解锁状态。 |
| `docs/17_D0_DATA_QA.md` | `a4e639828bdd7c8954250dc9f1a5cc357ba4a8e6b821f470f073967363b0e97f` | 采用检查逻辑 | 采用 registry/manifest/shard 闭包、分量恒等式、实测 SNR/SIR、精确重复与近重复探测的证据要求；本次代码覆盖前四类核心契约，近重复扫描仍须接入实际 TVT 数据流程后才能宣称通过。 |
| `tests/test_repository_metadata.py` | `93469d86e366a891639b0548313003b40790bafa51251ff0004c2f8143d06a02` | 采用测试方式 | 采用“损坏输入必须 fail closed”的契约测试方法；未复制源项目元数据 fixtures。 |
| `tests/test_d0_qa.py` | `35b812624cf797b889ed2e46d8452de76c3e6e7ef1191381bf927029e3cd28c1` | 采用测试方式 | 以合成数组验证 group 泄漏和分量恒等式；未载入源数据。 |
| `tests/test_hdf5_v2.py` | `245128a3c8350867d4d2fd126b561a09bab05eaa4d257cb38587c28943c534e3` | 拒绝直接移植 | 测试依赖被拒绝的源 HDF5 schema；只保留 checksum/闭包/篡改拒绝的测试意图。 |
| `tests/test_pilot_experiment.py` | `9852d4fffd81247533f016bcf2b96e0f1713c66f69cdc81269624919bf407e6b` | 采用测试方式 | 采用不可覆盖、路径逃逸拒绝、checksum 漂移拒绝、manifest 完成态和发布门禁契约；未运行源 runner，未复制训练 fixtures。 |

## 3. 落地文件及职责

| TVT 文件 | 职责 | 主要来源机制 |
|---|---|---|
| `tvt_submission/tccn_reuse/freeze.py` | 校验结果前冻结状态、seed 唯一、四类 split 角色隔离、数据 manifest SHA，并把加载绑定到配置文件精确字节哈希 | freeze config、freeze gate |
| `tvt_submission/tccn_reuse/manifest.py` | canonical JSON；同目录临时文件、fsync、独占锁、只创建一次；产物路径不得逃逸 attempt 目录；逐产物 SHA 闭包 | `pilot_experiment.py` |
| `tvt_submission/tccn_reuse/data_qa.py` | 样本唯一、可配置 group 身份跨 split 隔离、冻结因子单元精确计数、payload 哈希、分量恒等式和实测 SNR/SIR | `d0_qa.py`、`leakage_v2.py` |
| `tvt_submission/tccn_reuse/statistics.py` | screening/headline seed 下限、校准/测试隔离、配对样本/标签/cluster 完全一致、Holm 校正 | preregistration、`pilot_experiment.py` |
| `tvt_submission/tccn_reuse/claim_ledger.py` | claim 主键、状态枚举、`supported` 证据路径强制 | claim ledger |
| `tvt_submission/tccn_reuse/publication_gate.py` | 对配置 SHA、seed 集合、git commit/dirty、四项数据 QA、三项统计治理和必需产物执行全项 fail-closed 判断 | readiness gates |
| `tvt_submission/tccn_reuse/repository_audit.py` | JSON/CSV 健康检查及选定代码/配置的外来场景 token 防火墙 | repository audit |
| `tvt_submission/tccn_reuse/vehicular_experiment_freeze_template.json` | vehicular AMC 冻结模板；默认状态故意不可运行发布 | source freeze schema，仅保留结构 |
| `tests/test_tccn_reuse.py` | 10 个不训练的合成契约测试 | 上述测试设计原则 |

## 4. Fail-closed 发布条件

一个 TVT 结果包只有同时满足以下条件才可由 `assess_release` 判为 eligible：

1. run manifest schema 正确且 `execution_status=completed`；
2. manifest 的 frozen-config SHA 与结果前登记值精确一致；
3. 完成 seed 集合与冻结集合完全相同；
4. 存在源代码 commit，且默认要求运行时 worktree 干净；
5. `sample_id_unique`、`split_group_disjoint`、`component_identity`、
   `duplicate_scan` 全部显式为 `true`；
6. `paired_by_source`、`family_preregistered`、`calibration_isolated`
   全部显式为 `true`；
7. 每个必需产物位于 attempt 目录内、实际存在且 SHA-256 一致。

缺字段、字段为 `null`、路径越界、校验和漂移、少 seed、dirty source 或任何 QA
未通过，均拒绝发布。门禁不会把“未检查”解释为“通过”。

## 5. 明确未完成、不得提前声称的事项

- 冻结模板仍含替换占位符且状态为 `template_not_frozen`；它不是实验注册。
- `duplicate_scan=true` 必须由接入实际 vehicular AMC 数据后的精确重复与近重复
  检查产生，本工具不会自行把它置真。
- 当前实现没有替实际训练 runner 自动捕获 commit、环境、数据 manifest 或产物；
  runner 集成后仍需端到端测试。
- 3 个 seed 只允许 screening；高水平 headline 证据至少需要 5 个预先冻结的独立
  algorithm seeds。
- 本审计不验证 VIMD-Net 的性能优势，也不增加任何可用于论文的结果数字。
- 顺序决策、源 HDF5 schema、卫星信道和 APSK 设置被明确拒绝，不应进入 TVT
  方法、实验或消融。

## 6. 最小验证

执行命令（未启动训练）：

```text
python -m unittest tests.test_tccn_reuse -v
```

结果：`Ran 10 tests ... OK`。

覆盖的失败模式包括：

- 冻结配置字节被篡改；
- immutable 文件被二次发布；
- 产物相对路径逃逸；
- 产物 checksum 漂移；
- 同一 `source_id` 跨 split；
- 分量重构不成立；
- headline seed 数不足；
- 配对样本顺序不同；
- supported claim 缺少证据；
- 数据 QA 任一项未通过时发布被拒。

此测试结论只证明治理工具的最小契约成立，不等同于真实数据 QA 或论文证据链已经
完成。
