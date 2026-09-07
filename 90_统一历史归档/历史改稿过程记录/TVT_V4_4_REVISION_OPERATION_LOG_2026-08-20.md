# TVT V4.4 改稿操作记录（2026-08-20）

依据 `TVT\_V4.4\_Technical\_Freeze前最终改稿方案.md`，对 V4.3 两份 PDF 对应 tex 源稿完成全部修改、重编译与核验。本文件记录本次会话的全部操作、判定与产物。

* 输入方案：`TVT\_V4.4\_Technical\_Freeze前最终改稿方案.md`
* 目标产物（V4.3，只读基线）：

  * `output/pdf/tvt\_operating\_envelope\_integration\_V4\_3.pdf`（10 页）
  * `output/pdf/tvt\_supporting\_material\_V4\_3.pdf`（5 页）
* tex 源稿：`paper/main.tex`、`paper/supplement.tex`
* 交付产物（V4.4）：

  * `output/pdf/tvt\_operating\_envelope\_integration\_V4\_4.pdf`（**10 页**，562,491 字节）
  * `output/pdf/tvt\_supporting\_material\_V4\_4.pdf`（**6 页**，185,486 字节）

\---

## 1\. 操作清单（按执行顺序）

### 1.1 预检与数据核实（只读）

|核实项|来源 artifact|结果|
|-|-|-|
|V4.3 PDF Fig.5 为两 panel 图且星号错误|`output/pdf/...\_V4\_3.pdf` 高分辨率裁切|误标 `8PSK\*`、`CPFSK\*`；需改为 6 星号|
|clean-sentinel cell 身份|`analysis\_zero\_compute/outputs/csv/a14\_envelope\_cells.csv`|`id\_test` / SIR=0 / 最低 overlap quartile，`occupancy\_mean≈1.0165e-11`，`row\_count=1053`|
|E1 MCLDNN 三 seed all-SNR 精度|`artifacts/rml2016\_10a\_fidelity\_v3/metrics.csv`|0.581818 / 0.593886 / 0.580659 → 均值 **58.5454%**|
|E1 IQFormer 三 seed 均值|同上|0.647909 / 0.657864 / 0.651000 → **65.2258%**|
|E1 reference-release|`paper\_data\_layer/outputs/e1\_reference.json`|MCLDNN **62.05%**，IQFormer(ours) **64.19%**|
|MCLDNN unseen-speed R²\_skill|`analysis\_zero\_compute/outputs/csv/a14\_v41\_envelope\_interfered\_holdout.csv`|**0.724847** → 统一 0.72（主稿原 0.73 错误）|
|既有 break-even 不一致（主动发现）|`paper\_data\_layer/outputs/paper\_macros.tex` vs `paper/v41\_envelope\_macros.tex`|宏值 0.58/0.87/0.48/0.66 源自历史 113-cell 拟合，与主稿报告的 32-cell 系数不自洽|

### 1.2 P0-2：31-cell clean-sentinel 敏感性（新分析）

* **新建** `analysis\_zero\_compute/a16\_clean\_sentinel\_sensitivity.py`：从冻结 cell 表同时重算 32-cell 主拟合与 31-cell 敏感性（排除 sentinel cell），held 恒为 80 个 finite-SIR interfered cells，不重训任何模型。
* **运行产出**：

  * `analysis\_zero\_compute/outputs/csv/a16\_clean\_sentinel\_sensitivity.csv`
  * `analysis\_zero\_compute/outputs/a16\_clean\_sentinel\_sensitivity.json`
* 结果（详见 §3）：判定为**情况 A** → 保留 32-cell primary；supplement 新增敏感性小节，主稿仅补一句。

### 1.3 P0-1：Fig.5 两 panel 重绘

* **新建** `paper\_figures/make\_figs\_v44.py`：两 panel（(a) compact spectral family (A0--A7)、(b) I/Q-domain baselines），仅对训练未覆盖的 6 类加星（BPSK\*、QPSK\*、16QAM\*、64QAM\*、GMSK\*、4FSK\*），π/2-BPSK、8PSK、256QAM、CPFSK 无星号；IEEE 单栏 3.5 in。
* **覆盖输出** `paper\_figures/outputs/fig6\_condition\_transfer.pdf` / `.png`（main.tex 经 `\\includegraphics` 引用，无需改引用路径）。
* 布局迭代两次：首版 panel 标题重叠 → 调 `rcParams` 字号（titlesize 5.9 等）、`figsize=(3.5,2.6)`、`subplots\_adjust(left=0.115,right=0.99,wspace=0.12,top=0.90,bottom=0.28)`；(b) 标题右缘裁切 → 缩短为 "(b) I/Q-domain baselines"。

### 1.4 宏管线修正（break-even 源切换）

* **修改** `paper\_data\_layer/build\_paper\_numbers.py`：break-even 4 宏改读 `a16\_clean\_sentinel\_sensitivity.csv` 中 `fit\_cells == 32` 行（selector 记录于宏文件），证据类 `exploratory`，precision 2。
* **重新生成** `paper\_data\_layer/outputs/paper\_macros.tex`：仅 4 个宏变化

  * `\\BreakEvenIqformerSirFifteen` 0.58 → **0.11**
  * `\\BreakEvenIqformerSirTen` 0.87 → **0.47**
  * `\\BreakEvenMcldnnSirFifteen` 0.48 → **0.06**
  * `\\BreakEvenMcldnnSirTen` 0.66 → **0.29**

### 1.5 `paper/main.tex` 编辑

1. 头部版本注释改 V4.4 (2026-08-20)，声明自包含、仅一次提及 optional supplement。
2. E1 段落：`58.55%`（MCLDNN）/ `65.23%`（IQFormer-inspired）vs reference `62.05%` / `64.19%`；`+1.04` pp 与 `−3.50` pp（即 3.5 pp lower）；表述改为 range-level comparator-implementation fidelity。
3. Table III：MCLDNN unseen speed R²\_skill `0.73` → `0.72`。
4. break-even 段落：删除"超过最大观测 overlap 0.84 故不可达"子句（与新 0.11–0.47 值不自洽）；段尾新增 sensitivity 句："A sensitivity excluding the single fitting aggregate dominated by jammer-free sentinel windows leaves the held-regime conclusions materially unchanged (Supporting Material)."
5. Fig.5 caption 换为方案推荐文本："Clean condition-transfer taxonomy for the compact spectral family and I/Q-domain baselines. A star marks a modulation class whose jammer-free condition was not observed during training; the four covered classes are unstarred. The plot localizes the clean boundary rather than aggregating it."

### 1.6 `paper/supplement.tex` 编辑

1. 头部版本注释改 V4.4。
2. 新增 `\\section{Clean-Sentinel Fitting-Cell Sensitivity}`（位于 Envelope Skill by Held Regime 之后），含 `tab:sentinel` 32/31 对照表与结论句（held RMSE 变化 ≤0.20 pp、R²\_skill ≤0.05、系数符号与 break-even 排序不变 → 保留 32-cell primary）。
3. E1 表：All-SNR 行 `58.55% \[58.1,59.4]` / `65.23% \[64.8,65.8]`；Reference release 行 `62.05%` / `64.19%`。
4. E1 正文：`+1.04` pp、`3.50` pp lower。
5. Artifact Index 增加 a16 脚本 / CSV / JSON 三条目。

### 1.7 编译与交付

* `paper/` 目录下 `latexmk -pdf -outdir=build\_v44 main.tex` → 10 页（含 bibtex）。
* `latexmk -pdf -outdir=build\_v44 supplement.tex` → 6 页。
* 复制为 `output/pdf/tvt\_operating\_envelope\_integration\_V4\_4.pdf` 与 `output/pdf/tvt\_supporting\_material\_V4\_4.pdf`（V4\_3 原件保留未动）。

\---

## 2\. 核验记录（方案 §16 / §19）

|核验项|方法|结果|
|-|-|-|
|主稿页数|编译日志 / PDF|**10 页**（硬判据满足）|
|supplement 页数|编译日志|6 页（V4.3 为 5 页，新增小节约半页）|
|Fig.5 星号|`pdftoppm -r 200 -f 8` 裁切目检|仅 6 星号；π/2-BPSK、8PSK、256QAM、CPFSK 无星号|
|Fig.5 panel 标题|同上裁切|"(a) compact spectral family (A0--A7)"、"(b) I/Q-domain baselines" 完整无裁切/重叠|
|Fig.5 caption|`pdftotext`|新推荐文本渲染正确|
|break-even 渲染值|`pdftotext` 主稿|"0.11 (vs. IQFormer-inspired) and 0.06 (vs. MCLDNN) at SIR = −15 dB, rising to 0.47 and 0.29 at −10 dB"|
|sensitivity 句（主稿）|`pdftotext`|"A sensitivity excluding the single fitting …" 存在|
|sentinel 小节（supp）|`pdftotext`|标题与 32/31 表渲染正确|
|E1 数字一致性|grep 双 tex|主稿与 supp 均为 58.55/65.23/62.05/64.19、+1.04、3.50|
|旧值残留|grep `0\\.73\|62\\.1\|58\\.6\|113-cell`|双 tex 零残留|
|星号残留|grep `8PSK\\\*\|CPFSK\\\*\|256QAM\\\*`|tex 零残留（星号仅存在于图内 6 类）|
|命名|grep|正文 `PI/2-BPSK`、图内 `π/2-BPSK`，符合方案 §13|
|supplement 提及次数|grep 主稿|正文仅 1 次（line 729）|

\---

## 3\. a16 关键数值（32 vs 31，held=80 cells）

|Reference|Fit|intercept|overlap coeff|SIR coeff|BE o @−15|BE o @−10|Held RMSE|Const RMSE|R²\_skill|Oracle RMSE|
|-|-|-|-|-|-|-|-|-|-|-|
|IQFormer|32|−17.0766|14.3757|−1.0356|0.1073|0.4675|6.2303|8.3681|0.4457|6.0442|
|IQFormer|31|−18.1541|15.9032|−1.0664|0.1357|0.4710|6.4168|8.2761|0.3988|6.0442|
|MCLDNN|32|−14.5514|19.1607|−0.8938|0.0597|0.2930|8.1012|9.6347|0.2930|6.8088|
|MCLDNN|31|−15.5338|20.5535|−0.9219|0.0830|0.3072|8.1984|9.5915|0.2694|6.8088|

**情况 A 判定**：held RMSE Δ ≤ 0.20 pp（<0.3）；R²\_skill Δ ≤ 0.047（<0.05）；overlap/SIR 系数符号不变；break-even 排序不变（required overlap 随 SIR 升高而升高）。→ 保留 32-cell primary。

\---

## 4\. 问题与处理

|问题|处理|
|-|-|
|PowerShell 无 `head`|去除管道直接运行|
|`KeyError: 'windows'`（cell 表窗口数列名为 `row\_count`）|改用正确列名|
|Fig.5 首版标题重叠 / (b) 标题裁切|字号与布局参数两轮迭代（见 1.3）|
|`latexmk` 在工作区根目录报 "Could not find file 'main.tex'"|切到 `paper/` 目录编译（tex 使用 `../` 相对引用）|
|`validate\_paper\_numbers.py` 报 406 issues|核实为 V4.3 既有状态（表内手写数字等），与本次宏重生成无关，非回归，不追修|

\---

## 5\. 方案外必要一致性修复（特别说明）

`paper\_macros.tex` 的 break-even 4 宏原值（0.58/0.87/0.48/0.66）源自历史 113-cell 拟合，与主稿报告的 32-cell 系数（14.38/−1.036 等）数学上不自洽；新增敏感性表会直接暴露 32-cell break-even（0.11/0.47/0.06/0.29），若不修正将造成主稿与 supplement 数字冲突。故将 `build\_paper\_numbers.py` 宏源切换到 a16 的 `fit\_cells == 32` 行并重生成宏，同步重写主稿 break-even 段落（删除"超过最大观测 overlap 0.84"子句）。该修复为 freeze 判据"Supporting Material 与主稿数值一致"的必要条件。

\---

## 6\. 产物与变更文件清单

**新建**

* `analysis\_zero\_compute/a16\_clean\_sentinel\_sensitivity.py`
* `analysis\_zero\_compute/outputs/csv/a16\_clean\_sentinel\_sensitivity.csv`
* `analysis\_zero\_compute/outputs/a16\_clean\_sentinel\_sensitivity.json`
* `paper\_figures/make\_figs\_v44.py`
* `paper/build\_v44/`（编译中间产物）
* `output/pdf/tvt\_operating\_envelope\_integration\_V4\_4.pdf`
* `output/pdf/tvt\_supporting\_material\_V4\_4.pdf`
* `tmp/v44\_\*.png / tmp/v44\_\*.txt`（QA 裁切与文本抽取，临时）

**修改**

* `paper/main.tex`（版本头、E1、Table III、break-even 段、Fig.5 caption）
* `paper/supplement.tex`（版本头、新增 sentinel 小节、E1 表/正文、Artifact Index）
* `paper\_data\_layer/build\_paper\_numbers.py`（break-even 源切换）
* `paper\_data\_layer/outputs/paper\_macros.tex`（重新生成，4 宏变化）
* `paper\_figures/outputs/fig6\_condition\_transfer.pdf / .png`（覆盖重绘）

**未动**：V4\_3 两份原 PDF、`v1\_audit\_macros.tex`、`v41\_envelope\_macros.tex`、sealed 证据管线。

\---

## 7\. Freeze 判据对照（方案 §19）

* \[x] Fig.5 仅 6 星号、π/2-BPSK 命名、caption 为推荐文本
* \[x] 31-cell 敏感性完成且判定情况 A；supplement 含半页小节与表；主稿一句指向
* \[x] E1 差值统一（−3.50 / +1.04 pp，两位小数自洽）
* \[x] R²\_skill 统一 0.72
* \[x] break-even 主稿/supp/宏三方一致
* \[x] 主稿 10 页
* \[x] §16 搜索清单零冲突

\---

## 8\. FINAL 投稿轮（依据 `TVT\_V4.4\_FINAL投稿指令\_全方位评分\_成功率预测.md`，同日执行）

### 8.1 Step 1：main.tex 三处措辞

1. 删除 sensitivity 句尾 `(Supporting Material)` 字样（主稿不再指向外部文档），句尾改为 `held-regime conclusions materially unchanged.`。
2. abstract：`the ranking reverses in the severe-interference region, with the relative advantage organized by target-support overlap` 改为 `the ranking reverses in the severe-interference region,\\nwith the relative advantage organized by target-support overlap. In a\\npost-hoc severe-interference analysis at`（按指令新 wording）。
3. Section V-B（Rank Exchange）：按指令替换为 `single ranking is adequate: rank changes with SNR and SIR, with A5 reversing the marginal ranking under severe interference; the overlap analysis in Section\~V-C further shows that the required overlap decreases as interference becomes more severe.`。

### 8.2 Step 2：数字验证豁免（WAIVER）

* **新建** `paper\_data\_layer/waiver\_classify\_v44.py`：复用 `validate\_paper\_numbers.py` 规则收集全部 notices，对照扩展 artifact 参照集（macro 文件、`paper\_numbers.json\["keys"]`、`analysis\_zero\_compute/outputs/csv/\*.csv`、`paper\_data\_layer/outputs/\*.csv`、`artifacts/rml2016\_10a\_fidelity\_v3/{metrics.csv,per\_snr.csv,training\_history.csv,run.json}`、`tier2\_iq\_sidecar\_v1/run.json` 等）做 7 桶分类（scale 1/100、按字面小数位 rounding、k-shorthand、±符号匹配；DERIVED/DEFINITIONAL 白名单带 provenance）。
* **产出** `paper\_data\_layer/outputs/v44\_validator\_issues.json`（436 条 manifest）与 **新建** `docs/TVT\_V4\_4\_NUMBER\_VALIDATION\_WAIVER.md`。
* 结果：436 notices = b1 exact 107 + b2 derived/rounded 319 + b4 未引用宏遗留 9 + b5 caption/definitional 1；b3=b6=b7=0。**true mismatch = 0，unclassified = 0**。
* 迭代修复：`KeyError 'entries'` → 顶层键为 `keys`；初版 b6=248/b7=20 → 扩展 corpus 与匹配规则后归零。

### 8.3 Step 3：2026 参考文献 DOI 元数据 QA

* Crossref 逐条核验：`zhang2026malicious`（TVT Vol.75, No.8, 18609–18613）、`faysal2026denomae`（Vol.74, 929–943）、`li2026cgpcda`（TCCN Vol.12, 6929–6941）均与 bib 一致，保留。
* `efficientamc2026tgcn`：公开索引仅确认 Vol.10, 249–259，issue 未登记 → 依"不猜页码/期号"原则从 `references.bib` 删除 `number = {1},`。

### 8.4 Step 4：作者元数据与 AI 披露

* 用户决定：作者元数据**暂缓**（"先不管"），保留 `\\author{Anonymous Author(s)}`；致谢选**仅 AI 普通声明**。
* 在 Conclusion 之后、`\\bibliographystyle` 之前插入 `\\section\*{Acknowledgment}`：AI tools 仅用于英文语言编辑与文本润色；全部技术内容、推导、数据、代码、图表、解释与结论由作者产出并负责。

### 8.5 Step 5：编译核验与页数恢复

* 加入 Acknowledgment 后主稿溢出至 11 页（第 11 页仅 2 条参考文献）。按指令压缩优先级执行两轮压缩（仅文字紧缩，不删 Fig.1–5、不删表格、不动数字）：

  * 轮 1：Implementation Details 的 AGC/学习率句、E1 末句、Related Work 两处。
  * 轮 2：Implementation Details 首段（共享协议一句化）、"no comparator received an architecture-specific sweep"、Loss weights 末句、Seeds 句。
* 终验：**主稿 10 页、0 overfull hbox、字体全部嵌入（emb=yes）**；**supplement 6 页、0 overfull**；新 wording 与 Acknowledgment 经 pdftotext 确认渲染。

### 8.6 Step 6：final search 清单

|搜索项|结果|
|-|-|
|`Supporting Material`（主稿）|0 命中|
|`high-overlap corner`|双稿 0 命中|
|`0.87` / `0.73` / `58.6` / `62.1`|0 命中|
|`8PSK\*` / `CPFSK\*`|0 命中|
|`0.48`（主稿）|1 命中 = 0.4843 sidecar hard score（artifact 真值，非旧 break-even）|
|`0.58` / `0.66`（supp）|命中均为表内 artifact 值（0.584/0.582/0.587/0.588 与 CI −0.66），非旧 break-even|
|`Anonymous Author`|存在（按用户"先不管"暂缓，投稿前需替换作者块）|

### 8.7 Step 7：Freeze 产物

* `output/pdf/tvt\_operating\_envelope\_FINAL\_SUBMISSION.pdf`（**562,594 字节，10 页**）
* `output/pdf/tvt\_supporting\_material\_FINAL.pdf`（**185,486 字节，6 页**）
* 同步更新 `output/pdf/tvt\_operating\_envelope\_integration\_V4\_4.pdf`（562,594 B）与 `tvt\_supporting\_material\_V4\_4.pdf`（185,486 B）。

**遗留项**：作者块仍为 `Anonymous Author(s)`（用户决定暂缓）；正式投稿前需按 `paper/authors\_verified.example.tex` 模板填入已核实的作者元数据。

\---

## 9\. FINAL-S2 轮（按 `TVT\_FINAL\_S2\_基于深度终审意见的筛选式最终改稿方案.md` 同日执行）

原则：不新增仿真、不改证据等级、只做零仿真高 ROI 审计；Fig.1 保留；完成后重回 Technical Freeze。

### 9.1 零仿真审计（新脚本 `analysis\_zero\_compute/a17\_final\_s2\_audits.py`）

产出四份 CSV（`analysis\_zero\_compute/outputs/csv/a17\_\*.csv`），全部只读 sealed artifacts：

* **协变量阶梯**（`a17\_covariate\_ladder.csv`）：IQFormer 参照 M0 常数 8.37 / M1 SIR-only 6.84（R²=0.33）/ M2 overlap-only 7.46（0.20）/ M3 overlap+SIR 6.23（0.45）/ M4 +SNR 6.15（0.46）；MCLDNN 参照 9.63 / 8.71（0.18）/ 8.68（0.19）/ 8.10（0.29）/ 8.05（0.30）。overlap|SIR 增量两参照均 0.61 pp → SIR 为主导协变量，但 overlap 有一致真实增量。
* **selector 评估**（`a17\_selector\_evaluation.csv`）：overlap–SIR 符号规则在 80 held cells 仅选 A5 5/80（regret 0.36 pp，期望 MAC 336.0M，节省 5.5%）；hindsight oracle 也仅选 3/80 → 价值是边界定位而非算力节省，据实表述。
* **收敛审计**（`a17\_convergence\_audit.csv`）：MCLDNN median best epoch 13（11–16），0/10 撞 30 上限、10/10 early-stop；IQFormer 27（24–30）10/10 撞界；CSSL 29（26–30）10/10 撞界；A5 29.5（27–30）10/10 撞界 → 触发指令"普遍撞 30"分支。
* **sidecar 结构核验**：从 sealed checkpoint state\_dict 逐层读出 7294 = 216+24+84+288+48+120+576+48+1752+48+3504+96+490（base 39500 + added 7294 = 46794，与 `\\SidecarParamsAdd` 精确一致）。

### 9.2 `paper/main.tex` 修改清单

|指令项|落点|内容|
|-|-|-|
|P0-A1（ρ 情况 A）|III-D|"It is added only to the modulation route; therefore the retained route weights remain nonnegative but sum to $1+\\rho$ rather than one"；公式不改|
|P0-A2|V-C|oracle 段按指令原文重写，落句 "regime-dependent transport and within-regime structure rather than uniform held-domain superiority"|
|P0-A3|全文|删除 −0.06/−0.42 的 `\\EnvelopeOracleSkillIQ/MC` 引用（宏定义保留但不再使用）|
|P0-A4|Abstract/V-F/VI-D|"a partial representation-repair path"；"substantially narrows the clean-condition failure while preserving the structured-interference advantage"；"recovers a substantial part of the predicted missing-information deficit"；禁用混 seed 差值|
|P0-A5|IV-B E1|"As a public-benchmark comparator-fidelity check,"（post-review 措辞全删）|
|P0-A6|Abstract|压至约 220 词（含删 ten-seeds 句）|
|P0-A7|V-E|"five of the six design–seed cells lie below the 0.25 threshold; the sole cell above it is borderline at 0.256" + "unchanged at 0.30 but flips at 0.20"|
|P0-A8|V-E|分母句：36 cells = 9 compact×3 类 + 3 I/Q×3 类；27/27 vs 1/9|
|P0-A9|Table III|caption 加聚合口径脚注（指令原文）|
|P0-A10|IV-B|CSSL 句："reported as a supervised adaptation reference rather than a reproduction of the full released recipe"|
|P0-A11|Limitations Seventh|IQFormer/CSSL 10/10 撞界（median 27/29）vs MCLDNN 全 early-stop（median 13）；"shared budget may favor faster-converging models and severe-region comparisons inherit this caveat"|
|P0-B1/B2|V-C|ladder/selector headline 段（6.84/7.46/6.23、0.61 pp 增量、5/80 与 3/80、regret 0.36、5.5% MAC），全表指向 Supporting Material|
|§22|VI-E|held-speed coherence 句：0.43/0.31 ms ≈ 决策窗 42%/30%|
|§23|Results/Discussion|系统性减 hedge（四轮页码压缩，见 9.4）|
|§25|V-A|"their simultaneous upper bounds are \\TeacherFormSimHigh and \\TriRouteSimHigh pp, respectively"|
|§26|V-A|family gate 句：A5–A0 仍为有效 family-wise-controlled 结果|

### 9.3 `paper/supplement.tex` 与宏

* 新增三小节：**Comparator Convergence Audit**（tab:convergence，四模型 median/range/撞界/early-stop）+ **Sidecar Architecture Detail**（tab:sidecar，13 行逐层参数表）+ **Covariate Ladder and Selector Evaluation**（tab:ladder M0–M4 + tab:selector 五行，含 boundary-localization 解读段）。
* Table II（regime skill）段改为 regime-wise heterogeneous 表述（3.94 vs 6.71 等），删除被指令禁止的过度贬损句；tab:capabs caption 加口径脚注；Artifact Index 加 a17 与 sidecar 条目。
* `paper/v1\_audit\_macros.tex` 追加 FINAL-S2 宏块（ladder/selector/convergence，provenance 指向 a17 CSV）。

### 9.4 页数压缩（§23 授权，四轮）

新增内容净约 +40 行致 11 页 → 依次压缩：① ladder 段精简、severe hedge、E1 尾句、Fig.4 caption；② 摘要、贡献项、CSSL 句、Limitations 句；③ 摘要 ten-seeds 句、V-C 因果句、V-F 句；④ V-C cell 构造细节、oracle/sign-agreement 句、sentinel 敏感句、fusion control 值、sidecar 架构句、capacity CI、Discussion 三段、Reproducibility、Conclusion、Evidence box、III-A coherence 句、receiver-stress 句、III-B ratio-masking 引用句、cochannel 句、统计程序 sign-test 括注。**未删任何指令强制内容**（§22 句、family gate 句、Table III 脚注、ladder/selector headline、probe/分母措辞全部保留）。

### 9.5 编译终验与 Freeze 产物

* `latexmk -pdf -gg` 重建：**main 10 页**、supplement 8 页；main.log 无 undefined macro/reference。
* 数字一致性抽查：a17 三份 CSV ↔ `v1\_audit\_macros.tex` 宏 ↔ supp 表 ↔ 主稿行文全部吻合（6.84/8.71/7.46/8.68/0.61/5/3/0.36/5.5/13/27/29）。
* pdftotext 复核强制句渲染：ρ（sum to 1+ρ）、CSSL supervised adaptation、family-wise-controlled、42%/30% coherence、regime-dependent transport、36-cell 分母、0.256、6.84、5 of the 80、partial representation-repair、post-review 已消失、−0.06/−0.42 已消失（残留 0.06 为 break-even overlap `\\BreakEvenMcldnnSirFifteen`，合法）。
* 交付：`output/pdf/tvt\_operating\_envelope\_FINAL\_S2.pdf`（10 页）、`output/pdf/tvt\_supporting\_material\_FINAL\_S2.pdf`（8 页）。

**FINAL-S2 Freeze 判据（§33）逐项满足**：无新增仿真、证据等级未变、硬错误（ρ/oracle 表述/分母/post-review）已修、零仿真审计全部回填、主稿保持 10 页。**宣布重新 Technical Freeze（V4.4 FINAL-S2）**。遗留项同第 8 节：作者块仍为 Anonymous。

---

## 10. FINAL-S3 轮（按 `TVT_FINAL_S3_Technical_Freeze前最后零成本改稿方案.md` 同日执行）

原则：四项零成本修正（单变量结论写准确、selector tradeoff 透明、E1 epoch 协议补全、强 reference 系数补强 overlap 方向），完成后直接冻结投稿，不做任何科学扩展。

### 10.1 回查与事实确认（零仿真）

* **E1 epoch 预算**（情况 B）：E1 public-benchmark fidelity 实际为 IQFormer-inspired 60 epochs、MCLDNN 200 epochs（各按其发布协议），与主 campaign 统一 30-epoch 预算不同 → supp 明写披露。
* **overlap 系数**：从 sealed 拟合确认 IQFormer 参照 +14.38、MCLDNN 参照 +19.16 pp/unit overlap（与 sentinel 表 14.38/19.16 一致）。
* **收敛事实修正**：IQFormer-inspired 与 CSSL 均为 10/10 seeds 跑满 30 epoch（supp 原 "nine of ten" 为笔误，据 a17_convergence_audit.csv 改为 all ten）。

### 10.2 `paper/main.tex`

* V-C ladder 段：单变量结论改为 reference-dependent——"SIR is the stronger single covariate against IQFormer-inspired, whereas SIR and overlap have comparable single-covariate skill against MCLDNN"；0.61 pp 增量句加 "against both references"。
* selector 句：保留 boundary-localization 定性（5 of the 80 vs oracle 3、regret 0.36），新增主动披露 "trades approximately 0.22 percentage points of realized macro-F1 for a 5.5% reduction in expected MACs"；不将 selector 包装成优于 always-IQFormer。
* V-B overlap 方向句（§17.3）："The positive overlap direction is not specific to this adaptation reference: the fitted overlap coefficients against IQFormer-inspired and MCLDNN are also positive, at +14.38 and +19.16 pp per unit overlap"——overlap 正方向不再只依赖 CSSL。
* Limitations Seventh（§17.4）：删除严格 "lower bound" 表述，改 "the shared budget is potentially binding for IQFormer-inspired and CSSL"；新增 A5 自身事实——"A5 itself also runs to the 30-epoch boundary in all ten seeds, so the budget concern is not one-sided against the I/Q comparators"。

### 10.3 `paper/supplement.tex` 与宏

* E1 fidelity 小节：新增 epoch 协议句（IQFormer 60 / MCLDNN 200，发布协议原值，非统一 30 预算）。
* 收敛小节："lower bound" 改为 budget-limited 表述；nine-of-ten → all ten seeds 事实修正。
* `v1_audit_macros.tex` 新增 `\SelectorRealizedFTradePp`=0.22。**踩坑**：初版宏名 `\SelectorF1TradePp` 含数字 "1"，TeX 宏名遇数字截断导致编译错误（`\SelectorF` undefined），改名修复。

### 10.4 页数回收与编译终验

新增约 10 行经浮动体重排放大为约 24 行溢出 → 分批压缩（不动 S3 强制内容）回到 **main 10 页**、supplement 8 页；0 overfull、无 undefined macro/reference。

§19 搜索清单逐项核验（渲染 PDF 文本）：`dominant covariate`=0、`lower-bound`=0、`post-review`=0、`CSSL reproduction`=0；`0.61`（main 2/supp 3）、`0.22`（main 1）、`5.5%`（1/3）、`5 of 80`、`epoch 30`（supp 1）、`+14.38`/`+19.16`（main 2/1）、`Supporting Material`、`supervised adaptation` 全部就位且主稿/supp 一致。

§20 Freeze 判据逐项满足：单变量结论 reference-dependent；0.61 pp 增量对两参照明写；selector 0.22 pp 代价主动披露且未包装为优于 always-IQFormer；E1 epoch 预算公开；convergence 不再用严格 lower bound；A5 自身 epoch-boundary 公开；overlap 正方向有 IQFormer/MCLDNN 强 reference 系数支撑；CSSL 保持 supervised adaptation reference 标注；摘要 218 词 ≤250；主稿 10 页；supp 数值与主稿一致；无新增训练/仿真/claim。

交付：`output/pdf/tvt_operating_envelope_FINAL_S3.pdf`（10 页）、`output/pdf/tvt_supporting_material_FINAL_S3.pdf`（8 页），V4_4 副本同步更新。

**宣布重新 Technical Freeze（V4.4 FINAL-S3）**。按指令 §23：不再继续优化强度，直接投稿。遗留项同前：作者块仍为 Anonymous（正式投稿前按 `paper/authors_verified.example.tex` 填入）。

## 11. FINAL-S4 轮（按 `TVT_FINAL_S4_综合第二轮深度终审后的最终改稿执行方案.md` 同日执行）

原则：只补两个真正能被 Reviewer 一问击穿的证据点（held domain 不含 SIR=−15 dB 的披露与外推测试；0.61 pp overlap 增量的配对统计检验），恢复 fusion negative controls，修正 sidecar head 与 convergence 表述；不扩大仿真、不引入新数据、不升级叙事。

### 11.1 回查与事实确认（零仿真）

* **A0 收敛**：直读 sealed per-seed result.json，A0 selected epochs [30,30,30,30,29,30,28,29,30,28]——10/10 seeds 全部撞 30-epoch 上界（median 30，range 28–30）。
* **sidecar head 实现**：`run_tier2_experiment.py` 确认 `logits = output["logits"] + self.classifier(embedding)`——残差 logit 相加，classifier 为 Linear 48→10（490 参数），无独立辅助 loss。
* **fusion 宏**：`\FusionIqformerGain/Control`、`\FusionMcldnnGain/Control` 已在 `paper_macros.tex`（+4.71/+1.49、+5.70/+1.69），本轮仅补 excess 宏。

### 11.2 a18 新审计（`analysis_zero_compute/a18_final_s4_audits.py`，新建并成功运行）

* **leave-−15-out 外推测试**（排除 hard_interference 4 个 −15 cells，28 cells 重拟合预测 4 cells）：结果**参考依赖**——MCLDNN 方向 full model RMSE 6.84 vs constant 16.43 / SIR-only 7.87（保持）；IQFormer 方向 full 12.69 vs SIR-only 12.15 且 sign 1/4（方向失败）。
* **block-paired ΔMSE bootstrap**（20 held blocks，B=10000，seed 20260820）：IQFormer 参照 mean ΔMSE 7.97 pp²、95% CI [−2.06, 18.18]、P(ΔMSE>0)=0.94；MCLDNN 参照 10.18、[−8.29, 29.84]、0.85 → **情况 B**（CI 跨 0），按指令用 "exploratory refinement" 表述。
* **selector 混淆矩阵**（从 held-cell artifacts 直接算）：TP=3/FP=2/FN=0/TN=75，recall 3/3、precision 3/5。
* **−15 同口径 selector 审计**：4 cells hindsight 全 favor A5，但 28-cell sign 规则仅选 1（sign acc 1/4，realized F1 0.285 vs oracle 0.361，regret 7.56 pp）→ 据实写入 supp 作为外推 caveat 的 selector 侧呼应，不加 Table XI 新行。
* 产出：a18_leave15_out.csv、a18_leave15_out_cells.csv、a18_overlap_increment_bootstrap.csv、a18_selector_confusion.csv、a18_leave15out_selector.csv、a18_final_s4_audits.json。

### 11.3 `paper/main.tex`

* V-C ladder 段：0.61 pp 增量句加 bootstrap 结果（P(ΔMSE>0)=0.94/0.85、两 CI 均跨 0）→ "we treat overlap as a consistent exploratory refinement beyond SIR rather than a separately resolved predictor"。
* V-C selector 句：recall 3/3、precision 3/5 明写；新增指令给定 held −15 披露句（held cells 跨 −10 至 +10 dB，−15 仅在 fitting split，break-even 阈值不被 held-regime 独立验证）。
* oracle 句修复（P0-9）：`slightly below` → "lower than ... (marginally for IQFormer-inspired and more noticeably for MCLDNN)"。
* fusion negative controls 恢复："cross-model fusion gains exceed the ensemble controls by 3.22 and 4.01 pp"。
* bolted-on 防御句删除；`\pp over` 空格 typo 修复。
* sidecar head 句按代码如实："10-class linear head whose logits are added residually to the spectral classifier logits under the shared training objective; the side head has no separate auxiliary supervision."
* Limitations Seventh 重写：convergence caveat 非对称——MCLDNN 10/10 early-stop（median 13，range 11–16）非 binding，severe 层 A5–MCLDNN +9.74 pp；第八（新）条：−15 absent 披露 + leave-one-SIR-stratum-out 参考依赖结果 + "conditional extrapolations"。

### 11.4 `paper/supplement.tex` 与宏

* convergence 表加 A0 行（30 & 28–30 & 10/10 & 0/10）+ prose 句（§12）。
* sidecar prose 如实改写（残差相加 + 无独立辅助监督 + inference 只用混合信号）。
* ladder prose 改 reference-dependent；tab:selector caption 加口径警示（ten-seed cell-level 与 matched-five-seed split-level 不可相减，§P0-8）；selector prose 加混淆矩阵句。
* 新 section `\section{Leave-$-15$-dB-Out Extrapolation and Paired Overlap-Increment Tests}`（sec:s4tests）：Table XII（leave-−15-out，2 参照×3 预测器 RMSE/MAE/Sign acc.）+ Table XIII（paired bootstrap：Mean ΔMSE/CI/P/RMSE red.）+ 参考依赖与情况 B 解读段 + −15 selector audit 段。编号自动排为 XII/XIII ✓。
* Artifact Index 加 a18 条目。
* `v1_audit_macros.tex` 追加 FINAL-S4 宏块（LeaveFifteen*/BootDmse*/BootProbPos*/FusionIqformerExcess/FusionMcldnnExcess/ConvAZeroMedianEpoch/ConvAFiveMedianEpoch），provenance 指向 a18。

### 11.5 页数回收与编译终验

S4 新增内容使 main 溢出至 11 页 → 按指令 §23 授权压缩源分批回收（II-C Related Work、intro claim 段、E1 seed-range 句、Discussion V-A/V-B/V-C/V-E hedge 重复），未触碰任何 S4 强制内容 → **main 10 页**（末页为 Conclusion+完整参考文献，止于 [35]）、supplement 10 页；0 编译错误。

§28 搜索清单核验：`bolted`=0、`pointsover`=0、`slightly below`=0（supp 残余 1 处已同步修复）、`dominant covariate`=0、`free of all training-budget`=0；新数字 21.34/6.84/7.97/0.94/TP=3/TN=75、leave-one-SIR-stratum-out、conditional extrapolations、exploratory refinement、residually 全部就位且主稿/supp 一致。

交付：`output/pdf/tvt_operating_envelope_FINAL_S4.pdf`（10 页，SHA256 1CF65E82FB98…）、`output/pdf/tvt_supporting_material_FINAL_S4.pdf`（10 页），V4_4 副本同步更新。

**宣布重新 Technical Freeze（V4.4 FINAL-S4）**。按指令 §18/§26：A0-wide 保留为 Major Revision reserve，不做任何叙事升级。遗留项同前：作者块仍为 Anonymous（正式投稿前按 `paper/authors_verified.example.tex` 填入）。


