# TVT 交付清单规划

状态日期：2026-07-28

本文件只定义未来交付包的边界和校验方法；当前不创建 ZIP、不复制大型缓存、
不修改任何共享状态文件。正式缓存、正式运行和 release lock 尚不存在，
因此现在生成“最终投稿包”会产生误导。

## 1. 计划中的两层交付

最终交付应拆成两层，避免把期刊投稿源文件与大型可复现实验数据混在一起。

### A. IEEE 投稿源包

仅在 public build 全部通过后纳入：

- `paper/main.tex`
- `paper/references.bib`
- `paper/results_auto.tex`
- `paper/authors_verified.tex`
- `paper/IEEEtran.cls`
- `paper/figures/` 中被 `main.tex` 实际引用的最终图
- 论文编译需要的其他明确引用源文件
- `paper/release_lock.json` 的只读副本，用于内部交付审计
- `paper/TEMPLATE_PROVENANCE.md`

投稿系统是否接受内部审计 Markdown/JSON 应按上传当日要求决定；不要求的
审计文件保留在作者证据包，而不是强行上传给期刊。

### B. 作者可复现证据包

纳入：

- `src/vimd_amc/`
- `experiments/run_standard_experiment.py`
- `standards/` 中构建和审计正式缓存所需的 Python/MATLAB 桥接代码，但不含
  `.npy` 大型缓存本体
- `tvt_submission/configs/formal_tvt_freeze_v1.json`
- `tvt_submission/run_local.ps1`
- `tvt_submission/generate_macro_values.py`
- `tvt_submission/validate_formal_freeze.py`
- `tvt_submission/validate_release.py`
- `tvt_submission/validate_paper_build.py`
- `tvt_submission/sources/cssl_amc_2025.lock.json`
- CSSL 的本地 Apache-2.0 license 与 third-party notices
- `tvt_submission/MACRO_DERIVATION_AUDIT.md`
- `tvt_submission/MACRO_RELEASE_PROTOCOL.md`
- `tvt_submission/PRE_SUBMISSION_CHECKLIST.md`
- 本文件、`LOCAL_FORMAL_RUN_HANDOFF.md` 与
  `FINAL_CONVERGENCE_REPORT.md`
- `docs/` 中与协议、专利/Idea 溯源、近期比较器和 reviewer remediation
  直接相关的文档
- 最终测试报告、LaTeX release 审计 JSON、PDF 逐页人工 QA 记录
- 正式运行完成后，由 `run.json` 引用的 CSV、manifest、source fingerprint、
  统计摘要和必要 prediction provenance

大型正式 cache、55 组 checkpoint 和 prediction NPZ 应保存在独立证据库，
由 SHA-256、cache digest、run ID、文件大小和受控存储位置引用。是否公开
完整数据由许可、容量、匿名评审和机构政策共同决定。

## 2. 明确排除

以下内容不得进入 TVT 投稿源包：

- `standards/cache_factor_screening_1024_v1` 及其任何筛选指标；
- `diagnostics/`、失败候选和 VIMD-v4 一次性诊断结果；
- `artifacts/` 中非正式、失败、中断、superseded 或 source-mutated 运行；
- 手工填写的性能宏、手工编辑的 `run.json` 或伪造 release sentinel；
- TCCN 卫星场景数据、结果、模型 checkpoint、论文表述或整个
  `tccn_satellite_amc` 目录；
- cochannel/mixed-jammer 的未注册主任务结果；
- MATLAB 临时交换文件、Python cache、TeX 辅助文件和本机绝对路径日志；
- 原始专利 DOCX、Idea 原稿和用户提供的 IEEE 模板 ZIP 的复制件。它们保持
  原位置不变，以 hash 和溯源文档引用；
- 密钥、账号、个人联系方式、未公开审稿信息或其他敏感数据。

VIMD-v4 源码若作为“未来工作候选”保留在作者证据包，必须与正式 A5 结果
明确分层；其诊断输出绝不纳入当前论文结果清单。

## 3. 必须满足的打包前条件

只有以下条件全部满足，才允许开始实际打包：

- 正式缓存目录存在，manifest designation 为
  `headline_formal_tvt_evidence`；
- `artifacts/tvt_headline_1024_5seed_v1/run.json` 完整且 eligible；
- 11 模型 × 5 种子和所有预注册 split/artifact 齐全；
- 宏清单由 `generate_macro_values.py` 自动生成；
- `validate_release.py` 预检、写入和既有锁复核均零退出；
- `paper/results_auto.tex` 含有效 `EligibleLockedResults` sentinel；
- `paper/release_lock.json` 的 schema、run/cache/hash 绑定全部有效；
- 科学晋级门通过，不以完整性门代替效果门；
- CSSL 只使用“官方架构监督适配”准确标签；
- `authors_verified.tex` 与 disclosure 由人类作者审核；
- public build 成功且 `validate_paper_build.py --mode release` 返回
  `"ok": true`；
- 最终 PDF 不超过当日 TVT 限制，并完成逐页人工 QA；
- 专利公开时序和所有主来源已由合适的人类专业人员复核。

任一条件缺失时，清单状态保持
`pre_submission_convergence_not_upload_ready`。

## 4. 未来 manifest 结构

实际打包时，建议生成一个不含手工性能值的 UTF-8 JSON manifest，至少包含：

```json
{
  "schema_version": "vimd_amc.tvt.delivery_manifest.v1",
  "created_utc": "<UTC timestamp>",
  "package_role": "submission_source|author_evidence",
  "run_id": "tvt_headline_1024_5seed_v1",
  "formal_cache_digest": "<from eligible run>",
  "run_json_sha256": "<computed>",
  "release_lock_sha256": "<computed>",
  "paper_pdf_sha256": "<computed>",
  "files": [
    {
      "path": "<package-relative POSIX path>",
      "bytes": 0,
      "sha256": "<64 lowercase hex>",
      "role": "<source|evidence|license|audit|paper>"
    }
  ],
  "excluded_large_artifacts": [
    {
      "logical_path": "<path>",
      "digest": "<content digest>",
      "reason": "<size, privacy, or submission boundary>"
    }
  ]
}
```

文件列表必须按 package-relative path 排序；路径不得包含 `..`、盘符、绝对
路径或重复项。manifest 自身的 hash 应在写完后由外层交付记录保存，不能把
自身 hash 循环写回自身。

未来可用以下 PowerShell 只读命令核查候选文件，但本轮不执行打包：

```powershell
Get-ChildItem -LiteralPath .\paper -Recurse -File |
  Sort-Object FullName |
  Get-FileHash -Algorithm SHA256
```

大型证据库需另行记录每个文件的相对路径、字节数和 SHA-256；仅记录目录名
不足以证明内容。

## 5. 版本与不可变性原则

- 提交包必须绑定同一个 eligible `run.json`、macro manifest、
  `results_auto.tex`、release lock 和最终 PDF；
- release 后对上述任一文件的修改都使旧 manifest 失效，必须重新验证和
  重新生成，而不是改写 hash；
- 已有正式目录不得原位覆盖；
- 失败与负结果不删除，放入非投稿审计存储并保留原因；
- 用户提供的专利、Idea 和 IEEE ZIP 始终只读；
- TCCN 工作流保持隔离；只允许复用已经在 TVT 目录中完成来源登记的通用
  治理资产，不复制卫星结果。

## 6. 录用概率边界

manifest 能证明“交付文件完整、来源一致、声明受锁”，不能证明期刊一定接收。
即使所有门均通过，也不保证 90% 或任何具体录用率。最终提交决定和科学责任
属于人类作者。
