# TVT V4.5 S5R1 改稿操作日志（2026-08-20）

状态：**FROZEN（V4.5 S5R1）**

## 1. 背景与授权

1. S5-A（独立 −15 dB severe held 转移检验）判级 **fail**，已归档暂停
   （`docs/S5_A_OPERATION_LOG.md`，ARCHIVED）。
2. S5-R1（received-I/Q sidecar 表示修复检验）判级 **strong_repair**
   （`docs/S5_R1_OPERATION_LOG.md`）：pooled +4.37 pp，CI [+2.96, +5.90]，
   10/10 seeds、8/8 SNR 档为正，gap recovery vs IQFormer 1.70、vs MCLDNN 0.34。
3. 重定位影响评估 `docs/S5_REPOSITIONING_IMPACT_ASSESSMENT.md` 定位 7 处冲突表述；
   用户决策 = **选项 B：有界披露**——保持现标题与结构，修复冲突表述，V-F 新增
   S5-A/S5-R1 披露段，全表进 supp 新两节，新增宏同步验证器，编译确认 10 页后
   重新 Freeze。

## 2. 改稿原则

* 标题、章节结构、sealed artifact、V4.4 历史身份均不变。
* 新证据一律标注 exploratory / post-hoc / preregistered；不写成 sealed。
* 所有新数字走宏系统：`build_paper_numbers.py` 新增 24 个宏，
  `evidence_class="exploratory"`，源为 `a19_s5_severe_held.json` 与
  `a20_s5_r1_summary.json`（含 source SHA-256 与 row selector）。
* 宏名不含数字（TeX 控制字约束）：前缀 `SevereTransfer*`（S5-A）与
  `SevereRepair*`（S5-R1）。

## 3. 修改文件

* `paper_data_layer/build_paper_numbers.py`：新增 S5 宏段（24 键）。
* `paper_data_layer/outputs/paper_macros.tex` / `paper_numbers.json`：重新生成，
  253 键（sealed 67 / exploratory 186）。
* `paper/main.tex`：版本头 V4.5 S5R1；7 处冲突修复；V-F 末新增独立 severe held
  披露段；Table III（regime 表）移入 supp；既有冗余句压缩以保 10 页。
* `paper/supplement.tex`：版本头 V4.5 S5R1；abstract 增补；新增两节
  （S5-A 转移失败 + S5-R1 表示修复）；Artifact Index 增 3 条；加载
  `paper_macros.tex`。
* `output/pdf/tvt_operating_envelope_integration_V4_5_S5R1.pdf`（10 页）、
  `output/pdf/tvt_supporting_material_V4_5_S5R1.pdf`（11 页）。

## 4. 七处冲突修复对照

1. Abstract：severe corner 数字加 "campaign's" 限定 + 独立 held 不复现句；
   sidecar 增补独立 held +4.37 pp；"partial representation-repair path" →
   "representation-repair path"。
2. V-B：corner 检查 "reinforce" 加 "within the campaign" + 指向 V-F 的转移失败句。
3. V-C：held-regime 运输段尾增补独立 held 不支持 −15 dB 运输（指向 V-F）。
4. V-F 首段："while preserving the structured-interference advantage" 删除，
   改为 "whether the campaign-internal advantage survives beyond the campaign
   is examined at the end of this subsection"；段末新增披露段（S5-A 数字 +
   S5-R1 数字 + exploratory/post-hoc 标注）。
5. Limitations 第 5 条：exploratory 清单加入 independent severe-held validation。
6. Limitations 第 7 条：epoch-cap 论证加 "campaign-internal" 限定 + 独立 held
   不复现该优势。
7. Limitations 第 8 条 / Conclusion：leave-one-SIR-out 句增补独立 held 完全不复现；
   Conclusion "reproducible compact-model advantage" → campaign-internal corner +
   独立 held 不复现 + sidecar 修复 +4.37 pp。

## 5. Freeze 判据对照

* [x] 主稿 10 页（latexmk 编译核验）。
* [x] supp 11 页，编译通过，无 undefined 宏。
* [x] 验证器 547 → 504 条（净减 43；零新增；豁免体系
  `docs/TVT_V4_4_NUMBER_VALIDATION_WAIVER.md` 延续适用）。
* [x] 搜索清单：禁用旧表述（"preserves the structured-interference advantage"、
  "reproducible compact-model advantage"、"partial representation-repair path"）
  全部 0；必备披露（independently generated severe held set、SevereRepair*、
  SevereTransfer*、preregistered、post-hoc）主稿/supp 均就位。
* [x] sealed 67 键数值不变；V4.4 历史 PDF 未覆盖。

## 6. 新增宏值核对（与判级记录一致）

| 宏 | 值 | | 宏 | 值 |
|---|---|---|---|---|
| SevereTransferGainIqformer | −3.12 | | SevereRepairDiff | 4.37 |
| SevereTransferGainMcldnn | −12.45 | | SevereRepairCiLow/High | 2.96 / 5.90 |
| SevereTransferSignAccIqformer/Mcldnn | 0.578 / 0.406 | | SevereRepairRegimeOne/TwoDiff | 4.22 / 4.52 |
| SevereTransferRmse*（Full/SirOnly/Constant） | 9.75/9.05/8.11；22.56/22.29/15.41 | | SevereRepairPositiveSeeds | 10 |
| SevereTransferSelectorRecall | 0.581 | | SevereRepairPositiveSnrStrata | 8 |
| | | | SevereRepairGapRecoveryIqformer/Mcldnn | 1.70 / 0.34 |
| | | | SevereRepairLevel Sidecar/AFive/Iqformer/Mcldnn | 25.09/20.72/23.29/33.75 |

## 7. 内容压缩说明（为保 10 页）

* Table III（held-regime envelope skill）移入 supp：supp 既有表更完整
  （含 Oracle/Sign 列），主稿无对该表的正文引用；主稿改为指向 Supporting Material。
* V-C 协变量阶梯、selector 段、oracle-constant 段、V-E 单元格枚举、VI-D/VI-E/
  Conclusion/Reproducibility 既有冗余句压缩；**无 claim 删除**，所有 sealed 数字
  与披露句保留。

## 8. 未动

sealed 证据管线、figures、references、`v1_audit_macros.tex`、
`v41_envelope_macros.tex`、V4.4 及更早历史 PDF、waiver 文档。
