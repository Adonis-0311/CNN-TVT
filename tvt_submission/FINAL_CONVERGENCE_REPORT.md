# IEEE TVT 预投稿收敛状态报告（非最终投稿验收）

状态日期：2026-07-28

项目：VIMD-Net AMC under vehicular-Doppler TDL simulations

## 结论先行

当前工作树已经形成可交给本地机器执行的、fail-closed 的预投稿收敛包：
算法、九域证据协议、11 模型 × 5 种子冻结、近期 CSSL 比较器、自动宏来源链、
release lock、IEEE 模板论文和 public-build 门均已有明确接口。

它目前仍不是可上传稿件。正式 headline 缓存未构建，五种子正式训练未运行，
正式结果宏与 release lock 未生成，作者信息及人类原始文献审计也未完成。
这一状态是有意保留的：本轮没有启动任何长时间缓存或训练任务，也没有把
筛选/诊断结果伪装成正式证据。

## 当前证据状态

| 项目 | 当前状态 | 可用于 TVT 主结果 |
|---|---|---|
| 方法实现与 A0--A7 消融 | 已实现并冻结接口 | 仅待正式运行 |
| 九域 factor-isolated 协议 | 已实现 | 仅正式 designation 可用 |
| 1024 点筛选缓存 | 完整性审计通过，digest 为 `241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3` | 否；designation 明确为 screening |
| 正式 headline 缓存 | 未启动，目标目录不存在 | 否 |
| 11 模型 × 5 种子运行 | 未启动，目标目录不存在 | 否 |
| CSSL-AMC 2025 比较器 | 官方架构来源已锁定，本地监督适配已注册 | 正式运行后才可比较 |
| VIMD-v4 DSBN | 已实现为一次性候选诊断，尚未执行 | 否；不在当前 freeze |
| 自动论文宏 | manifest v3 接口已实现：73 个 provenance records、74 个 non-sentinel TeX commands、release 时 75 个 commands | 当前无 eligible 输入；最终全量回归仍待完成 |
| `release_lock.json` | 不存在 | 否 |
| IEEE Transactions 模板 | 使用用户提供 ZIP 中的本地 `IEEEtran.cls` | 是 |
| 内部评审 PDF | 最新 compile-check 为 8 页、证据锁分支；`paper/build/main.pdf` 的 7 页副本早于最新源文件，已陈旧 | 仅内部评审，不能作为投稿 PDF |
| public PDF | 尚未构建 | 否 |
| 作者/单位/披露 | 待人类作者提供 | 否 |

## 已冻结的预注册设计

正式缓存固定为 1,024 点、96 点 guard、master seed `20260727` 的九个
source-disjoint split：train、validation、ID test、hard interference、
unseen jammer、unseen speed、held-out channel、combined OOD 和 clean
retention。行政规模为 47,000 个源序列和 94,000 个配对视图。

正式模型家族为 A0--A7、MCLDNN reimplementation、IQFormer-inspired 和
`cssl_amc_supervised_adaptation`，种子为 `17,29,43,71,101`，总计 55 次
拟合。主参考 `cssl_amc_supervised_adaptation` 是训练前固定的 CSSL-AMC
官方架构监督适配，不得事后称为“最强”。IQFormer-inspired 仍是必需的
non-oracle 比较器和 Holm family 候选。

CSSL 比较器来自 2025 年 CSSL-AMC 官方代码的编码器/分类器拓扑审计，但本地
实验使用统一 1,024 点输入、统一监督训练预算和统一 split。它没有导入官方
checkpoint，也没有复现原论文的两阶段对比学习过程。因此准确标签是
“官方架构监督适配”，禁止称为完整复现、官方结果或结构化干扰专用 SOTA。

VIMD-v4 DSBN 是前瞻性候选，不属于当前正式家族。它的任何诊断输出都不能
进入当前论文结果；若未来通过候选门，需要新的 freeze 和新的正式运行。

## 强证据链与停止条件

证据链按以下顺序单向推进：

```text
冻结配置预检
  -> 正式 MATLAB-TDL 缓存
  -> 11 模型 × 5 种子正式运行
  -> runner-native JSON/CSV/NPZ
  -> 自动 manifest-v3 宏清单 + scientific_release_gate
  -> 从 artifacts 重推导并执行 release 预检
  -> 仅在全部科学/完整性门通过后写 results_auto.tex + release_lock.json
  -> public LaTeX build
  -> 逐页人工审阅
```

其中任一完整性门非零退出、缓存 designation 错误、源身份重叠、源树运行期
漂移、模型/种子缺失、fallback checkpoint、非有限或占位指标、CSV/NPZ
不一致、release digest 不一致，都会关闭 release。

科学晋级门独立于软件“跑完”状态：

- hard-interference 主终点相对 A0、MCLDNN、IQFormer-inspired 和 CSSL
  监督适配这四个必需 non-oracle 基线中的**每一个**至少 +5 pp macro-F1；
- unseen jammer/speed/held-out channel 相对预注册 CSSL 主参考至少两个域达到
  +3 pp；
- clean retention 在 seen A/C/D 和 held B/E 两层分别满足点估计不低于
  -1 pp、配对 95% 区间下界不低于 -2 pp；
- A5 在 hard-interference 上严格优于 A1 和 A6；
- 必需机制值均有限，两项相关性非负，oracle-conditioned spectral component
  ratio 严格为正；这些量不能改称 waveform SIR、SDR 或源分离证据。

失败时不能从五个 seed 中挑选有利结果、改写主参考、改变门槛或覆盖运行。
正确动作是保留负结果、维持内部评审状态，并在新的前瞻性冻结下迭代。

## 本地执行临界路径

完整、可复制的 PowerShell 命令、预期目录、资源预估、故障处理和 public
build 顺序见 `LOCAL_FORMAL_RUN_HANDOFF.md`。简化顺序为：

1. `run_local.ps1` 无 `-Execute` 预检；
2. `-Stage cache -Execute` 构建正式缓存；
3. `-Stage experiment -Execute` 完成 55 次拟合；
4. `generate_macro_values.py` 自动生成宏清单；
5. release 先预检，再显式 `-WriteRelease`；
6. 人类补齐作者信息、切换 public 分支、编译并运行
   `validate_paper_build.py --mode release`。

本轮没有执行以上长任务。正式缓存、正式运行和 release lock 的缺失必须在
交付时继续如实披露。

## 论文与权利边界

论文保持 simulation-only：证据来自 3GPP TR 38.901 TDL-profile 参数化的
车辆多普勒仿真，不声称 SDR、实测道路、实时部署、波形恢复、源分离或完整
V2X 系统级合规。单天线、同类调制源碰撞在没有物理目标锚、优势发射机规则
或歧义标签时不进入单标签主协议。

专利是方法溯源基础，Idea 是论文设计总纲；本阶段只交付论文和本地正式运行
链。专利后续优化及工程转化应在论文证据稳定、公开时序经专业人员审核后另行
推进。

## 对“90%录用目标”的准确表述

本项目可以通过更强的比较器、冻结协议、源级配对、五种子统计、失败封锁和
IEEE 模板合规来提高稿件质量并降低可避免的拒稿风险，但不能科学地保证
90% 或任何具体录用概率。录用还受编辑范围判断、审稿人评价、同期稿件、
政策和人类作者最终表述影响。最终可主张的只有“证据链经验证且声明与证据
匹配”，不能主张“录用已被保证”。

## 收敛判定

静态设计与本地交接：接口已形成；最终集成、全量回归和最新 PDF 逐页 QA
尚未完成。

正式机器证据：未开始。

上传就绪：否。

下一权威动作：严格按 `LOCAL_FORMAL_RUN_HANDOFF.md` 从只读预检开始。
