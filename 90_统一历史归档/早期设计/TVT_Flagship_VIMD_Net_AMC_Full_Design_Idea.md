# IEEE TVT 旗舰论文全过程设计 Idea  
## VIMD-Net：高机动强干扰环境下的软掩码多任务自动调制识别

> **目标期刊**：IEEE Transactions on Vehicular Technology（TVT）  
> **投稿栏目建议**：Wireless Communications  
> **研究主线**：软掩码 + 多任务 AMC + 干扰不变表征学习  
> **技术底座**：专利《基于神经网络的干扰环境下信号调制识别方法及系统》（DFI257727）及 E-AMR V2 工程优化路线  
> **文件定位**：从科学问题、方法设计、数据与实验、理论分析、代码实现、写作排版到投稿门槛的全过程旗舰方案  
> **版本日期**：2026-07-27

---

# 0. 执行摘要

## 0.1 最终论文定位

本论文不能写成“又一个 CNN/TCN 改进模型”，也不能只在 RadioML 上证明平均准确率提高。面向 TVT，应将科学问题重新定义为：

> **在高速移动、时变多径、频偏和复合干扰共同存在的车辆无线环境中，如何把与调制类别有关的稳定语义，从与干扰、信道和移动性有关的非平稳因素中分离出来？**

围绕该问题，提出：

# **VIMD-Net**
## **Vehicular Interference-Aware Modulation Disentanglement Network**

建议英文主标题：

> **VIMD-Net: Overlap-Aware Soft Decomposition and Multi-Task Contrastive Learning for Robust Modulation Classification in High-Mobility Vehicular Channels**

建议中文题目：

> **VIMD-Net：面向高机动车辆信道鲁棒调制识别的重叠感知软分解与多任务对比学习网络**

论文核心不是单一二值软掩码，而是升级为：

1. **干扰条件化的重叠感知三路软掩码**：将潜在特征连续划分为“调制主导、干扰主导、信号—干扰重叠/不确定”三部分；
2. **潜在理想比例掩码教师监督**：利用仿真训练阶段可获得的干净信号与干扰分量，生成物理意义明确的弱监督掩码；
3. **多任务环境表征学习**：联合调制分类、干扰类型识别及 SNR/SIR/归一化多普勒估计；
4. **跨条件调制对比学习**：同一调制在不同车辆信道、速度和干扰条件下作为正样本，压缩环境变化引起的类内散度；
5. **车辆化评测闭环**：采用 3GPP V2X 场景、跨速度/跨信道/未见干扰、连续行驶轨迹及 SDR 封存数据验证。

## 0.2 为什么必须进行上述升级

截至 2026 年，已有工作已经出现：

- 频域单软掩码与自适应滤波；
- 面向低 SNR 雷达调制识别的 smart-mask attention；
- 面向 AMC 的对比学习和自监督学习；
- TVT 中的复杂值网络、双流 CNN-LSTM、轻量 AMC、RepCCNet、Cross-SKNet、信道鲁棒累积量方法和联邦 AMC。

因此，论文不能再把“使用 sigmoid 产生一个 mask”作为核心创新。旗舰版本必须通过以下差异建立新颖性：

> **三路重叠建模 + 训练期潜在物理教师 + 信号/干扰双支路监督 + 跨车辆条件对比学习 + 高机动实测闭环。**

## 0.3 TVT 适配性

TVT 官方范围的 Wireless Communications 方向明确覆盖：

- vehicular/mobile communications；
- cognitive communications；
- UAV/Vehicle-to-X communications；
- spectrum sharing；
- interference cancellation and coordination；
- machine learning for wireless communications。

因此，论文必须把“车辆环境”落实为可复现实验变量，而不是只在 Introduction 中出现 V2X 文字。至少应具备：

- 3GPP V2X 车辆信道；
- 车辆速度和多普勒范围；
- 同频碰撞、窄带、扫频、脉冲及复合干扰；
- 跨速度与连续轨迹测试；
- 一个真实或半实物 SDR 验证集。

## 0.4 旗舰投稿的最低成稿标准

在固定测试集上，相对于最强可比基线，建议内部阶段门为：

- 全场景 Macro-F1 提高 **≥3 个百分点**；
- `SIR ≤ 0 dB` 困难区 Macro-F1 提高 **≥5 个百分点**；
- 未见干扰/未见参数测试提高 **≥4 个百分点**；
- 高 SNR、无干扰性能下降不超过 **0.5 个百分点**；
- 最差调制类别召回率提高 **≥4 个百分点**；
- 参数量控制在 **2 M 以内**，P95 推理时延不超过强基线的 **1.5 倍**；
- 三随机种子下收益的 95% 置信区间下限仍为正；
- 至少一组真实或半实物 SDR 数据上保持明确正收益。

这些数值是内部 go/no-go 门槛，不是对外性能承诺。

---

# 1. TVT 投稿约束与论文包装原则

## 1.1 当前官方格式约束

按照 TVT 官方作者说明：

- Regular Paper 初投稿最多 **14 页**；
- 修改稿/终稿最多 **16 页**；
- 双栏、最终 IEEE 排版、字体不小于 10 pt；
- 参考文献和作者简介计入页数；
- Regular Paper 超过 10 个免费页面后存在强制超页费；
- 初稿必须直接按照 TVT 格式准备，而不是后期再压缩；
- TVT 对与车辆技术核心范围不匹配的论文可直接拒稿；
- 论文应作为独立完整工作，不允许拆成显式 Part I/Part II；
- TVT 当前作者规则要求：AI 工具不得代替作者生成论文技术内容；若仅用于作者已有文本的语言修改，应按要求披露。

## 1.2 TVT 论文叙事的三个不可妥协项

### 不可妥协项 A：车辆场景必须进入系统模型

标题、摘要、System Model 和实验必须同时出现：

- high-mobility vehicular channel；
- Doppler/time-varying multipath；
- vehicular co-channel interference/jamming；
- V2X spectrum awareness 或 blind receiver configuration。

### 不可妥协项 B：不能只证明“网络更深所以更准”

论文应回答：

- 为什么需要三路掩码而不是普通 attention；
- 为什么重叠区域不能硬分配；
- 为什么辅助干扰任务能改善调制表征；
- 为什么跨条件对比损失能减小车辆信道引起的类内散度；
- 这些机制是否真的降低了干扰泄漏，而非仅增加参数。

### 不可妥协项 C：必须有泛化和实际价值证据

至少包括：

- 未见干扰；
- 未见干扰参数；
- 未见速度或未见车辆信道；
- 连续轨迹稳定性；
- SDR/半实物验证；
- 推理复杂度。

---

# 2. 科学问题、任务边界与应用场景

## 2.1 科学问题

车辆接收端观测到的复基带序列可写为：

\[
x[n]
=
e^{j(2\pi \Delta f nT_s+\phi)}
\left(
\sum_{\ell=0}^{L_h-1}h_v[n,\ell]s_m[n-\ell]
\right)
+
\sum_{q=1}^{Q}a_qj_q[n-\tau_q]
+
w[n],
\]

其中：

- \(s_m[n]\)：调制类别为 \(m\) 的目标信号；
- \(h_v[n,\ell]\)：与车辆速度、道路环境及遮挡有关的时变多径信道；
- \(\Delta f\)、\(\phi\)：载频偏移与初相；
- \(j_q[n]\)：第 \(q\) 类干扰或同频碰撞信号；
- \(a_q\)、\(\tau_q\)：干扰幅度和相对时延；
- \(w[n]\)：接收噪声。

定义：

\[
\mathrm{SNR}=10\log_{10}\frac{P_s}{P_n}, \qquad
\mathrm{SIR}=10\log_{10}\frac{P_s}{P_j}.
\]

目标是从长度为 \(N\) 的未知 I/Q 窗口中估计：

\[
\hat m=\arg\max_m p_\theta(m\mid x),
\]

同时输出辅助环境量：

\[
\hat{\mathbf c}
=
\left[
\hat{\mathbf p}_{jam},
\widehat{\mathrm{SNR}},
\widehat{\mathrm{SIR}},
\hat f_D
\right],
\]

其中 \(f_D\) 为归一化多普勒或多普勒等级。

## 2.2 论文明确不做的事情

为避免主线发散，第一篇 TVT 论文不承担：

- 调制推荐或自适应调制决策；
- 协议识别；
- 完整开放集识别；
- 端到端信号恢复；
- FPGA 资源实现；
- 大型 Transformer；
- 联邦学习；
- 完整领域自适应。

论文只输出：

1. 当前实际调制类别；
2. 干扰/信道辅助状态；
3. 校准置信度和质量标志。

“识别结果”和“推荐调制方式”继续保持彻底分离。

## 2.3 车辆应用场景

推荐统一采用以下故事线：

> 车载单元、路侧单元或无人机中继需要在稠密、动态的 V2X 频谱中进行非合作信号感知。车辆高速运动带来时变多径与多普勒，邻近车辆并发传输形成同频碰撞，窄带设备、脉冲设备或异常发射源进一步污染信号。接收端在没有先验控制信息的情况下，需要快速判断未知信号的调制方式，以支持频谱监测、盲接收机配置、干扰诊断和后续链路处理。

---

# 3. 现有研究格局与论文缺口

## 3.1 TVT 中具有代表性的 AMC 路线

| 工作 | 主要贡献 | 对本论文的启示 | 尚未覆盖的缺口 |
|---|---|---|---|
| Meng 等，TVT 2018 | 深度学习 AMC 完整框架 | 深度 AMC 的经典基线 | 未显式建模强干扰和车辆条件 |
| LightAMC，TVT 2020 | 压缩感知与网络剪枝实现轻量化 | 必须报告参数与时延 | 轻量化不等同于干扰鲁棒 |
| Complex-Valued Networks，TVT 2020 | 复杂值网络保持 I/Q 结构 | 前端应保留复信号耦合 | 未做信号—干扰分解 |
| CNN-LSTM Dual-Stream，TVT 2020 | 双流提取空间与时间特征 | 双流网络是重要强基线 | 仍属于直接分类 |
| Decentralized/Ensemble AMC，TVT 2022 | 分布式学习与模型融合 | 说明 TVT 接受 AMC 系统创新 | 不解决强干扰表征污染 |
| RepCCNet，TVT 2024 | 因果卷积与重参数化，强调噪声鲁棒与效率 | 需要与高效强网络比较 | 无显式干扰分离和未见干扰 |
| Cross-SKNet，TVT 2024 | OFDM 下 IQ 失衡、CFO 补偿与选择核网络 | 硬件损伤实验不可缺失 | 主要针对 OFDM 与前端失真 |
| CSQCC，TVT 2024 | 基于谱商累积量实现未知多径信道鲁棒 | 需要跨信道测试 | 面向 OFDM、手工表示，未覆盖复合干扰 |
| FL-MoLSTM，TVT 2025 | 联邦 AMC 与改进 LSTM | 多端数据与隐私是另一条路线 | 不属于本论文主线 |

## 3.2 2025—2026 年新近重叠风险

### 风险一：Soft mask 已经出现

2026 年的 FAFT 工作对 OFDM 幅度谱产生单一 soft mask，再进入自适应频域滤波；2025 年的低 SNR 雷达调制识别工作采用 smart mask attention。

因此不得声称：

> “本文首次将 soft mask 用于调制识别。”

正确的新颖性表述应是：

> 本文首次系统研究**干扰条件化、重叠显式建模、双分支任务监督和潜在理想比例教师联合驱动**的软分解机制，用于高机动车辆通信 AMC。

“首次”必须在完成最终文献检索后再决定是否保留，初稿宜使用“to the best of our knowledge”。

### 风险二：对比学习 AMC 已经出现

自监督、半监督及 modulation-consistency contrastive learning 已用于 AMC。本文不能把“使用 InfoNCE”单独作为创新。

正确差异是：

- 正样本来自**同一基带信号在不同车辆速度、信道和干扰条件下的成对观测**；
- 负样本优先选择**干扰相同但调制不同**的 hard negatives；
- 对比损失施加在**软分解后的调制支路**；
- 目标不是减少标签需求，而是减少干扰/移动性引起的类内散度。

### 风险三：多任务 AMC 不是新概念

多任务学习本身只能作为机制，不宜单独列为主贡献。本文应突出：

> 多任务头对三路掩码和双支路分解进行结构化监督，而不是简单在共享 backbone 后增加多个输出头。

---

# 4. 最终方法：VIMD-Net

## 4.1 总体结构

```text
Input complex I/Q window x ∈ R^(2×N)
        │
        ▼
Complex-aware causal stem
        │
        ▼
Shared local-temporal encoder F ∈ R^(C×T)
        │
        ├──────────────► Environment encoder
        │                 ├─ jammer multi-label head
        │                 └─ SNR/SIR/Doppler quality head
        │                          │
        │                          ▼
        └────────► condition embedding ec
                                   │
                                   ▼
             Interference-Conditioned Overlap-Aware Tri-Mask
                  Ms: modulation-dominant mask
                  Mj: interference-dominant mask
                  Mo: overlap/uncertain mask
                       │                  │
                       ▼                  ▼
             modulation branch Fs   interference branch Fj
                       │                  │
                       │                  └─ jammer/quality supervision
                       ▼
              temporal modulation encoder
                       │
                       ▼
              modulation embedding zm
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    AMC classification        cross-condition
         head                 contrastive loss
```

---

## 4.2 输入与复值感知前端

输入：

\[
X=
\begin{bmatrix}
I[0:N-1]\\
Q[0:N-1]
\end{bmatrix}
\in\mathbb R^{2\times N}.
\]

首选窗口：

\[
N=1024,
\]

并补充 \(N=256,512,2048\) 的窗口长度消融。

采用实值实现的复卷积：

\[
Y_R=W_R*I-W_I*Q,
\]

\[
Y_I=W_R*Q+W_I*I.
\]

建议结构：

| 模块 | 配置 |
|---|---|
| Complex stem 1 | kernel 7，32 个复值等效通道，stride 1 |
| Complex stem 2 | kernel 5，48 个等效通道，stride 2 |
| 归一化 | GroupNorm 或 LayerNorm，避免 batch 统计依赖 |
| 激活 | SiLU；面向 FPGA 的后续版本可替换 ReLU |

这样既保留 I/Q 耦合，又避免使用难以部署的复杂自定义算子。

---

## 4.3 共享局部—长程编码器

共享编码器由两部分组成：

### 局部编码

4 个深度可分离 Conv1D 残差块：

\[
48\rightarrow64\rightarrow64\rightarrow96.
\]

用于提取：

- 幅相跳变；
- 瞬时频率纹理；
- 脉冲边缘；
- 窄带与部分带干扰结构。

### 长程编码

6 个 TCN 块，膨胀率：

\[
1,2,4,8,1,2,
\]

卷积核为 3，残差连接，输出：

\[
F\in\mathbb R^{C\times T},\quad C=96.
\]

TCN 相比 LSTM 更适合：

- 并行训练；
- 固定时延；
- 单卡训练；
- 后续 ONNX/FPGA 映射。

---

## 4.4 环境编码器

对 \(F\) 做统计池化：

\[
g=\operatorname{Concat}
\left[
\operatorname{Mean}_t(F),
\operatorname{Std}_t(F),
\operatorname{Max}_t(F)
\right].
\]

通过 MLP 产生环境嵌入：

\[
e_c=\operatorname{MLP}(g)\in\mathbb R^{32}.
\]

辅助头包括：

### 干扰类型头

若一个窗口只允许一种干扰，使用 softmax；旗舰数据包含混合干扰时，应使用多标签 sigmoid：

\[
\hat{\mathbf p}_{jam}
=
\sigma(W_j e_c+b_j).
\]

损失：

\[
L_{jam}
=
-\sum_{k=1}^{K_j}
\left[
y_k\log \hat p_k+(1-y_k)\log(1-\hat p_k)
\right].
\]

### 环境质量头

输出：

\[
\hat{\mathbf q}
=
[
\widehat{\mathrm{SNR}},
\widehat{\mathrm{SIR}},
\hat f_D
].
\]

使用归一化 Huber Loss：

\[
L_q
=
\sum_{r}
\operatorname{Huber}(\hat q_r-q_r).
\]

辅助任务只服务于调制识别，不作为最终业务主张。

---

## 4.5 核心创新一：重叠感知三路软掩码

### 4.5.1 为什么不是二值或单掩码

在同频碰撞、宽带干扰和时变多径条件下，同一个潜在单元往往同时含有：

- 调制边沿；
- 信道响应；
- 干扰能量；
- 噪声。

将其强制分为“信号”或“干扰”会造成错误删减。因此定义三个连续掩码：

\[
[M_s,M_j,M_o]
=
\operatorname{softmax}
\left(
\frac{G_\theta(F,e_c)}{T(e_c)}
\right),
\]

满足：

\[
M_s+M_j+M_o=1,
\]

其中：

- \(M_s\)：调制信息主导；
- \(M_j\)：干扰信息主导；
- \(M_o\)：信号与干扰重叠或判断不确定。

环境条件通过 FiLM 或条件仿射调制进入掩码生成器：

\[
\tilde F
=
\gamma(e_c)\odot F+\beta(e_c).
\]

温度：

\[
T(e_c)=T_{\min}+
(T_{\max}-T_{\min})\sigma(g_T(e_c)),
\]

使强干扰时掩码更平滑，避免过度自信的硬选择。

### 4.5.2 重叠分量自适应分配

计算：

\[
\lambda_o=\sigma(g_\lambda(e_c)),
\]

\[
\rho=\rho_{\min}
+
(\rho_{\max}-\rho_{\min})\sigma(g_\rho(e_c)).
\]

调制支路：

\[
F_s=
(M_s+\lambda_oM_o)\odot F+\rho F.
\]

干扰支路：

\[
F_j=
(M_j+(1-\lambda_o)M_o)\odot F.
\]

解释：

- \(\lambda_o\) 决定重叠信息更偏向哪一支；
- \(\rho F\) 为残差保护路径；
- 弱干扰时保留更多原始特征；
- 强干扰时依赖软分解；
- 掩码出错时不会完全切断调制信息。

---

## 4.6 核心创新二：潜在理想比例掩码教师

仅依靠分类损失容易产生“看起来有 mask、实际上只是 attention”的问题。仿真数据生成时具有：

- 干净目标信号 \(s\)；
- 干扰分量 \(j\)；
- 混合信号 \(x=s+j+n\)。

采用指数滑动平均教师编码器 \(E_{\bar\theta}\)，分别编码：

\[
A_s=|E_{\bar\theta}(s)|,\qquad
A_j=|E_{\bar\theta}(j)|.
\]

构造潜在比例：

\[
r_s=\frac{A_s}{A_s+A_j+\epsilon},
\qquad
r_j=1-r_s.
\]

重叠程度：

\[
u=4r_sr_j.
\]

三路教师掩码：

\[
M_s^\star=(1-u)r_s,
\]

\[
M_j^\star=(1-u)r_j,
\]

\[
M_o^\star=u.
\]

由于：

\[
M_s^\star+M_j^\star+M_o^\star=1,
\]

可直接用 Jensen-Shannon divergence 或 KL 散度监督：

\[
L_{mask}
=
D_{\mathrm{JS}}
(
M^\star\Vert M
).
\]

该教师只在训练阶段使用，推理不需要干净信号或干扰参考。

### 对真实数据的处理

真实数据通常没有 \(s/j\) 分量标签：

- 对有参考的合成数据使用 \(L_{mask}\)；
- 对无参考 SDR 数据关闭 \(L_{mask}\)；
- 使用分类损失、环境多任务损失和双视图一致性继续微调；
- 不用伪造“真实干净波形”。

---

## 4.7 调制与干扰双支路

### 调制支路

\[
z_m=P_m(E_m(F_s))\in\mathbb R^{128}.
\]

分类概率：

\[
p(m\mid x)=\operatorname{softmax}(W_mz_m+b_m).
\]

主损失：

\[
L_{mod}
=
-\sum_c y_c\log p_c.
\]

可使用 label smoothing 0.05，但必须与所有基线保持一致。

### 干扰支路

\[
z_j=P_j(E_j(F_j)).
\]

干扰头在 \(F_j\) 上预测干扰类别，环境质量头可同时读取共享环境特征与 \(z_j\)。这样能对掩码形成方向明确的监督：

- \(F_s\) 应适合调制识别；
- \(F_j\) 应适合干扰识别；
- 两支路不能只是随机分流。

### 支路正交约束

批次归一化后：

\[
L_{\perp}
=
\left\|
\bar Z_m^{\mathsf T}\bar Z_j
\right\|_F^2.
\]

该项权重必须很小，避免把物理上相关的信息强行完全独立。

---

## 4.8 核心创新三：跨车辆条件调制对比学习

### 4.8.1 成对样本生成

对同一干净基带序列 \(s_i\)，生成两个视图：

\[
x_i^{(1)}
=
\mathcal H_{v_1,c_1}(s_i)+j_1+n_1,
\]

\[
x_i^{(2)}
=
\mathcal H_{v_2,c_2}(s_i)+j_2+n_2,
\]

两者：

- 调制类别相同；
- 车辆速度不同；
- 信道类型可不同；
- 干扰类型和强度不同；
- CFO/SFO/相位等损伤不同。

在调制支路上获得：

\[
z_i^{(1)},z_i^{(2)}.
\]

### 4.8.2 正负样本设计

正样本：

- 同一基带实例的不同条件视图；
- 同一调制类别、不同条件的其他样本。

优先 hard negatives：

- 干扰类型相同、调制类别不同；
- SIR 接近、调制类别不同；
- 容易混淆的调制对，如 QPSK/8PSK、16QAM/64QAM。

监督对比损失：

\[
L_{xcc}
=
-\sum_i
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
\frac{
\exp(\operatorname{sim}(z_i,z_p)/\tau)
}{
\sum_{a\neq i}
\exp(\operatorname{sim}(z_i,z_a)/\tau)
}.
\]

该损失不是为了减少标签成本，而是为了：

> 在保留调制可分性的同时，降低车辆信道、速度和干扰条件造成的类内漂移。

---

## 4.9 多任务负迁移控制

总损失初版：

\[
L=
L_{mod}
+\lambda_jL_{jam}
+\lambda_qL_q
+\lambda_mL_{mask}
+\lambda_xL_{xcc}
+\lambda_\perp L_\perp.
\]

建议初值：

| 项 | 初始权重 |
|---|---:|
| \(L_{mod}\) | 1.00 |
| \(L_{jam}\) | 0.25 |
| \(L_q\) | 0.05 |
| \(L_{mask}\) | 0.50 |
| \(L_{xcc}\) | 0.10 |
| \(L_\perp\) | 0.005 |

不得只凭最终测试集调参。所有权重在验证集上确定。

若辅助任务出现负迁移，按以下顺序处理：

1. 对辅助头使用不确定性加权；
2. 启用 PCGrad 或只对共享编码器进行梯度投影；
3. 降低质量回归头权重；
4. 删除 Doppler 回归，仅保留干扰类别和 SIR；
5. 若仍无增益，保留多任务作为掩码预训练监督，联合阶段关闭部分辅助头。

---

# 5. 轻量理论分析

## 5.1 命题一：有效特征 SIR 改善

令潜在表示满足：

\[
F=F^{(s)}+F^{(j)}.
\]

经过调制支路有效权重 \(W_s=M_s+\lambda_oM_o+\rho\) 后：

\[
\tilde F^{(s)}
=
W_s\odot F^{(s)},
\]

\[
\tilde F^{(j)}
=
W_s\odot F^{(j)}.
\]

定义信号保留率：

\[
\alpha
=
\frac{
\mathbb E\|\tilde F^{(s)}\|_2^2
}{
\mathbb E\|F^{(s)}\|_2^2
},
\]

干扰泄漏率：

\[
\beta
=
\frac{
\mathbb E\|\tilde F^{(j)}\|_2^2
}{
\mathbb E\|F^{(j)}\|_2^2
}.
\]

则掩码后的潜在信干比为：

\[
\mathrm{SIR}_{feat}^{out}
=
\frac{\alpha}{\beta}
\mathrm{SIR}_{feat}^{in}.
\]

当：

\[
\alpha>\beta,
\]

有：

\[
\mathrm{SIR}_{feat}^{out}
>
\mathrm{SIR}_{feat}^{in}.
\]

这一结论本身简单，但可以明确论文需要测量的中间物理量：

- signal retention；
- interference leakage；
- feature-SIR gain。

## 5.2 命题二：跨条件一致性降低类内散度

假设调制嵌入可分解为：

\[
z=\mu_m+\delta_c+\epsilon,
\]

其中：

- \(\mu_m\)：调制语义中心；
- \(\delta_c\)：由信道、速度和干扰条件引起的偏移；
- \(\epsilon\)：剩余扰动。

对同一调制、不同条件的正样本最小化：

\[
\mathbb E\|z_c-z_{c'}\|^2,
\]

等价于压缩：

\[
\mathbb E\|\delta_c-\delta_{c'}\|^2.
\]

在类间中心距离不降低的前提下，类内散度下降使 Fisher 判别比：

\[
J=
\frac{\operatorname{tr}(S_B)}
{\operatorname{tr}(S_W)}
\]

增大。

论文中应把该结论与以下实验对应：

- 类内/类间距离统计；
- t-SNE/UMAP 仅作辅助；
- Fisher ratio；
- 未见条件泛化性能。

注意：理论部分只在明确假设下给出，不宣称一般条件下的全局最优性。

---

# 6. 数据体系：V2X-JAM-AMC

## 6.1 数据集分层

### D0：公开基础数据

- RadioML 2016.10A：用于实现正确性和经典基线对齐；
- RadioML 2018.01A 子集：用于更大类别和更长序列的补充验证。

公开数据不作为车辆工程可用性的主要证据。

### D1：V2X-JAM-AMC 仿真数据

主论文数据集，建议在线生成与离线封存相结合。

### D2：SDR/半实物数据

在屏蔽、线缆或依法授权的实验环境中采集，按设备和会话隔离。开发阶段不得查看封存测试会话。

---

## 6.2 调制类别

推荐首版 10 类：

1. BPSK；
2. \(\pi/2\)-BPSK；
3. QPSK；
4. 8PSK；
5. 16QAM；
6. 64QAM；
7. 256QAM；
8. GMSK；
9. CPFSK；
10. 4FSK。

若 256QAM 在低 SNR/强干扰下过于不稳定，可将其作为扩展类，不应为了获得好看结果删除困难类别。

可增加一个 OFDM 子集实验：

- CP-OFDM + QPSK；
- CP-OFDM + 16QAM；
- CP-OFDM + 64QAM；
- CP-OFDM + 256QAM。

该子集专门与 Cross-SKNet、CSQCC 等 TVT 工作比较。

---

## 6.3 车辆信道与移动性

建议依据 3GPP TR 37.885 的 V2X 评估场景构建：

- Highway LOS；
- Highway NLOS；
- Urban/Grid LOS；
- Urban/Grid NLOS。

基础载频可采用典型 V2X 频段配置，最终以项目实际设备与当地法规为准。

速度网格：

\[
v\in\{0,30,60,120,180,250\}\ \mathrm{km/h}.
\]

训练阶段连续随机采样；测试阶段固定网格。

必须设置跨条件划分，例如：

- 训练速度：0–160 km/h；
- 测试未见速度：180、250 km/h；
- 训练：Highway LOS、Urban LOS、部分 NLOS 参数；
- 测试：未见延迟扩展和 Doppler 组合。

---

## 6.4 干扰字典

### 单一干扰

1. 单音 CW；
2. 多音；
3. 线性调频 chirp；
4. 扫频；
5. 脉冲/突发；
6. 部分带；
7. 梳状；
8. 邻频泄漏；
9. 同频调制信号；
10. OFDM-like 宽带干扰。

### 复合干扰

随机选取两类，控制：

- 相对功率；
- 时间占空比；
- 频谱重叠率；
- 起止位置；
- 干扰间时延。

论文中至少单独报告：

- 同频；
- 部分带；
- 脉冲；
- 混合干扰。

---

## 6.5 参数矩阵

训练范围：

\[
\mathrm{SNR}\sim U[-12,20]\ \mathrm{dB},
\]

\[
\mathrm{SIR}\sim U[-15,20]\ \mathrm{dB}.
\]

固定测试网格建议：

- SNR：\(-12,-8,-4,0,4,8,12,16,20\) dB；
- SIR：\(-15,-10,-5,0,5,10,15,20\) dB。

同步与硬件损伤：

- 随机载频偏移；
- 相位偏移；
- 符号定时偏差；
- 采样频偏；
- IQ 幅相不平衡；
- 直流偏置；
- AGC 增益变化；
- ADC 量化；
- 温和削顶；
- 接收滤波器响应差异。

---

## 6.6 数据规模

旗舰配置建议：

| 集合 | 建议规模 |
|---|---:|
| 训练 | 100–200 万窗口，在线增强 |
| 验证 | 10–20 万封存窗口 |
| 仿真测试 | 20–40 万固定窗口 |
| 未见干扰测试 | 5–10 万窗口 |
| 未见车辆条件测试 | 5–10 万窗口 |
| SDR 测试 | 每类不少于 5 个独立会话，合计 5–20 万窗口 |

若算力受限，优先减少训练重复样本，不减少测试条件覆盖。

---

## 6.7 防止数据泄漏

必须按“源序列”而非窗口随机切分：

- 同一比特序列派生的所有视图只能属于一个集合；
- 同一信道随机种子不能跨集合；
- 同一 SDR 连续记录的相邻窗口不能跨集合；
- 同一设备/会话应分组切分；
- 数据清单记录哈希、随机种子和生成配置。

---

# 7. Baseline 体系

## 7.1 通用深度基线

- CNN2；
- ResNet1D；
- CLDNN；
- CNN-LSTM Dual-Stream；
- Complex-Valued CNN；
- Transformer AMC；
- RepCCNet。

## 7.2 轻量和车辆相关基线

- LightAMC；
- 积累极坐标特征轻量网络；
- Cross-SKNet（仅 OFDM 子集）；
- CSQCC（仅适用子集）。

## 7.3 干扰处理基线

- 原始 IQ + 强分类器；
- 固定 notch/filter + 强分类器；
- 硬阈值掩码；
- 单一 sigmoid soft mask；
- 普通 SE/CBAM attention；
- 双掩码无 overlap 分支；
- VIMD-Net 三路掩码。

## 7.4 上界与下界

- Clean-signal upper bound；
- Oracle latent mask upper bound；
- 无干扰模型在干扰测试集上的下界；
- 原专利近似链 B0。

## 7.5 公平对比原则

所有可训练模型使用：

- 相同数据；
- 相同窗口；
- 相同训练轮数；
- 相同优化器搜索预算；
- 相同随机种子；
- 相同 early stopping；
- 相同标签平滑和增强策略。

若引用原论文结果，必须与“本文统一重训结果”分开列出。

---

# 8. 完整实验矩阵

## E1：公开数据基础能力

目的：

- 验证实现正确；
- 证明方法未以牺牲普通 AMC 性能换取干扰性能。

输出：

- RadioML Accuracy-SNR；
- 参数量和 MACs；
- 与经典 TVT 基线对比。

## E2：车辆强干扰主结果

在完整 V2X-JAM-AMC 测试集上报告：

- Accuracy；
- Macro-F1；
- 每类 Recall；
- 最差类别 Recall；
- NLL；
- ECE。

这是 Table II 的核心结果。

## E3：SNR–SIR 二维热图

每个模型生成：

\[
\mathrm{MacroF1}(\mathrm{SNR},\mathrm{SIR}).
\]

重点区域：

- SNR ≤ 0 dB；
- SIR ≤ 0 dB；
- 高 SNR、低 SIR；
- 低 SNR、无干扰。

平均值不能掩盖二维失效区。

## E4：不同干扰类型

分别报告：

- CW；
- chirp/sweep；
- pulse；
- partial-band；
- co-channel；
- mixed.

验证软分解是否对频谱重叠干扰最有价值。

## E5：未见干扰泛化

训练不出现：

- 梳状干扰；
- 某类 OFDM-like 宽带干扰；
- 两类特定复合组合。

测试这些条件，并加入未见参数：

- 新扫频速率；
- 新占空比；
- 新频谱占用率；
- 新功率组合。

## E6：跨速度与跨车辆信道

测试：

- 训练未出现的 180/250 km/h；
- LOS→NLOS；
- Highway→Urban；
- 未见延迟扩展/Doppler 组合。

报告：

- 性能下降量；
- 相对最强基线的增益；
- 特征类内散度变化。

## E7：完整消融

| ID | Tri-mask | Latent teacher | MTL | XCC | Residual gate | 目的 |
|---|---:|---:|---:|---:|---:|---|
| A0 | × | × | × | × | × | 共享 backbone |
| A1 | 单 mask | × | × | × | × | 普通 soft mask |
| A2 | √ | × | × | × | √ | 验证 overlap 与残差 |
| A3 | √ | √ | × | × | √ | 验证物理教师 |
| A4 | √ | √ | √ | × | √ | 验证多任务 |
| A5 | √ | √ | √ | √ | √ | 完整模型 |
| A6 | 双 mask | √ | √ | √ | √ | 三路与双路直接比较 |
| A7 | √ | √ | √ | √ | × | 验证残差保护 |

## E8：掩码机制验证

不能只展示漂亮热图，必须量化：

- Mask 与教师 LIRM 的相关性；
- signal retention \(\alpha\)；
- interference leakage \(\beta\)；
- feature-SIR gain；
- 重叠掩码占比随 SIR 变化；
- 旁路/残差系数随环境变化；
- hard mask 与 tri-mask 的误删率。

## E9：表征分析

报告：

- Fisher ratio；
- 类内距离；
- 类间距离；
- 同调制跨条件余弦相似度；
- t-SNE/UMAP，仅作为辅助图。

## E10：复杂度与实时性

指标：

- Params；
- MACs/FLOPs；
- 模型大小；
- GPU batch=1 P50/P95；
- CPU batch=1 P50/P95；
- 峰值内存；
- 256/512/1024/2048 窗口延迟。

## E11：连续车辆轨迹

构建一条连续时变轨迹：

- 速度变化；
- LOS/NLOS 切换；
- SIR 逐渐变化；
- 干扰突然出现/消失；
- 多径延迟扩展变化。

报告：

- classification outage probability；
- temporal label flip rate；
- jammer onset 后恢复到正确类别所需窗口数；
- 置信度随时间变化；
- 与最强基线的时间序列对比。

该实验能显著增强 TVT 场景适配性。

## E12：SDR/半实物验证

最小要求：

- 多个独立采集会话；
- 仿真训练、实测测试；
- 少量实测校准前后均报告；
- 会话级封存；
- 同时报告类别和条件分组。

实测不能只选最好的一次记录。

---

# 9. 统计与可信性要求

## 9.1 重复性

核心模型至少：

- 3 个随机种子；
- 报告均值 ± 标准差；
- 对主要差值给 bootstrap 95% CI。

## 9.2 配对显著性

相同测试样本上：

- 使用 McNemar 检验比较分类错误差异；
- 多个基线时使用 Holm 校正；
- 同时报告绝对百分点和相对错误率下降。

## 9.3 校准

在独立验证集上做 temperature scaling：

\[
p_T(c\mid x)
=
\operatorname{softmax}(l_c/T_c).
\]

报告：

- ECE；
- NLL；
- reliability diagram；
- 错误高置信率。

---

# 10. 训练全过程

## 10.1 阶段 T0：基线冻结

完成：

- 数据接口；
- B0 原专利近似链；
- CNN/ResNet/RepCCNet 强基线；
- 固定数据划分；
- 固定评测脚本。

退出条件：

- 三次重复结果稳定；
- 数据无泄漏；
- 所有指标可自动生成。

## 10.2 阶段 T1：共享编码器预训练

只训练：

\[
L=L_{mod}.
\]

目标：

先得到可靠 AMC backbone，避免掩码早期塌缩。

建议：

- AdamW；
- lr \(3\times10^{-4}\)；
- weight decay \(10^{-2}\)；
- 5 epoch warm-up；
- cosine decay；
- mixed precision；
- gradient clipping 5.0。

## 10.3 阶段 T2：掩码教师预热

部分冻结 backbone，加入：

\[
L=L_{mod}+\lambda_mL_{mask}+\lambda_jL_{jam}.
\]

训练 10–20 epoch。

观察：

- 三路占比；
- mask entropy；
- signal retention；
- interference leakage。

若 \(M_s\) 全 1 或全 0，禁止进入联合训练。

## 10.4 阶段 T3：多任务联合训练

解冻全模型：

\[
L=
L_{mod}
+\lambda_jL_{jam}
+\lambda_qL_q
+\lambda_mL_{mask}
+\lambda_\perp L_\perp.
\]

必要时：

- uncertainty weighting；
- PCGrad；
- 辅助头分阶段解冻。

## 10.5 阶段 T4：跨条件对比微调

加入：

\[
\lambda_xL_{xcc}.
\]

建议：

- 对比损失延后启用；
- 使用类别均衡采样；
- hard negative 比例不超过全部负样本的 50%；
- 温度 \(\tau\) 在验证集搜索；
- 防止过强对比导致调制类内过压缩。

## 10.6 阶段 T5：实测校准

冻结前半 backbone，优先微调：

- 归一化层；
- condition encoder；
- tri-mask generator；
- 分类 head。

论文主结果必须同时报告：

- zero-shot synthetic-to-real；
- few-session calibration。

不能只报告微调后的数字。

---

# 11. 推荐超参数起点

| 项目 | 初值 |
|---|---|
| 输入窗口 | \(2\times1024\) |
| Backbone 输出通道 | 96 |
| 调制嵌入 | 128 |
| 环境嵌入 | 32 |
| 模型参数目标 | 0.8–1.8 M |
| Batch，24 GB GPU | 256 |
| Batch，8 GB GPU | 48–64 + 梯度累积 |
| Optimizer | AdamW |
| 初始学习率 | \(3\times10^{-4}\) |
| Weight decay | \(10^{-2}\) |
| Warm-up | 5 epoch |
| 训练总 epoch | 80–120 |
| 对比温度 | 0.07–0.2 |
| \(\rho_{\min},\rho_{\max}\) | 0.05, 0.35 |
| Mask temperature | 0.5–2.0 条件化 |
| Label smoothing | 0.05 |
| 随机种子 | 至少 3 个 |

---

# 12. 代码工程设计

```text
vimd_amc/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data_v2x.yaml
│   ├── model_vimd.yaml
│   ├── train_baseline.yaml
│   ├── train_vimd.yaml
│   └── eval_all.yaml
├── src/
│   ├── data/
│   │   ├── modulation_generator.py
│   │   ├── paired_view_dataset.py
│   │   ├── metadata_schema.py
│   │   └── split_guard.py
│   ├── channels/
│   │   ├── v2x_3gpp.py
│   │   ├── fading.py
│   │   └── impairments.py
│   ├── jammers/
│   │   ├── tone.py
│   │   ├── chirp.py
│   │   ├── pulse.py
│   │   ├── partial_band.py
│   │   ├── cochannel.py
│   │   └── mixture.py
│   ├── models/
│   │   ├── complex_stem.py
│   │   ├── temporal_encoder.py
│   │   ├── environment_encoder.py
│   │   ├── tri_mask.py
│   │   ├── latent_mask_teacher.py
│   │   ├── heads.py
│   │   └── vimd_net.py
│   ├── losses/
│   │   ├── multitask.py
│   │   ├── mask_distillation.py
│   │   ├── cross_condition_contrastive.py
│   │   └── orthogonality.py
│   ├── train/
│   │   ├── trainer.py
│   │   ├── schedules.py
│   │   ├── checkpoint.py
│   │   └── seed.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── significance.py
│   │   ├── complexity.py
│   │   └── trajectory.py
│   └── visualization/
│       ├── heatmap.py
│       ├── masks.py
│       ├── confusion.py
│       └── paper_figures.py
├── scripts/
│   ├── 00_smoke_test.py
│   ├── 01_build_manifest.py
│   ├── 02_train_baselines.py
│   ├── 03_train_vimd.py
│   ├── 04_run_ablations.py
│   ├── 05_run_generalization.py
│   ├── 06_run_sdr_eval.py
│   └── 07_build_paper_tables.py
├── tests/
│   ├── test_power_control.py
│   ├── test_snr_sir.py
│   ├── test_no_leakage.py
│   ├── test_mask_sum.py
│   └── test_determinism.py
└── artifacts/
    ├── manifests/
    ├── checkpoints/
    ├── metrics/
    ├── figures/
    ├── tables/
    └── logs/
```

## 12.1 数据记录格式

每个样本至少记录：

```yaml
sample_id:
source_sequence_id:
modulation:
snr_db:
sir_db:
jammer_labels:
channel_scenario:
vehicle_speed_kmh:
doppler_norm:
cfo_norm:
sfo_ppm:
iq_gain_db:
iq_phase_deg:
random_seed:
split:
device_id:
session_id:
```

## 12.2 自动化纪律

- 所有长任务支持断点续训；
- 每个实验先执行 100–1000 样本 smoke test；
- 配置文件和 git commit 写入 checkpoint；
- 结果文件不得手工修改；
- 图表从 CSV/JSON 自动生成；
- 不允许训练脚本读取封存测试集进行 early stopping；
- 单卡同时只运行一个主训练作业，数据生成可低优先级并行。

---

# 13. 论文图表规划

## Figure 1：车辆场景与系统问题

展示：

- 高速车辆/RSU；
- 目标信号；
- 同频车辆；
- 窄带/脉冲干扰；
- 时变信道；
- 识别输出。

## Figure 2：VIMD-Net 总体架构

核心流程图，必须一眼看懂：

- shared encoder；
- condition encoder；
- tri-mask；
- signal/jammer branches；
- MTL；
- contrastive pairs。

## Figure 3：三路掩码与潜在教师

展示：

- \(M_s,M_j,M_o\)；
- LIRM 教师生成；
- overlap 分配；
- residual gate。

## Figure 4：SNR–SIR 二维主结果

建议用：

- 最强基线；
- 完整模型；
- 差值热图。

## Figure 5：未见干扰和跨速度泛化

柱状图或折线图。

## Figure 6：连续车辆轨迹

展示：

- speed；
- SIR；
- channel state；
- predicted label；
- confidence；
- recovery time。

## Figure 7：Mask 物理量

展示：

- signal retention；
- interference leakage；
- feature-SIR gain；
- overlap ratio。

## Figure 8：SDR 结果与混淆矩阵

如页数不足，可将部分图放入补充材料或代码仓库，但主文必须保留实测主结果。

---

## Table I：相关工作差异

列：

- vehicular mobility；
- explicit jammer modeling；
- overlap-aware mask；
- physical mask supervision；
- multi-task；
- unseen jammer；
- real data；
- complexity。

## Table II：主性能

所有强基线的 Accuracy、Macro-F1、Worst Recall、ECE。

## Table III：消融

A0–A7。

## Table IV：泛化

未见 jammer、未见 speed、未见 channel。

## Table V：复杂度

Params、MACs、P95 latency、memory。

## Table VI：SDR/半实物

zero-shot 和 calibrated。

---

# 14. TVT 14 页稿件结构与页数预算

| 章节 | 建议页数 |
|---|---:|
| Title/Abstract/Index Terms | 0.5 |
| I. Introduction | 1.25 |
| II. Related Work and System Model | 1.5 |
| III. Proposed VIMD-Net | 3.5 |
| IV. Analysis and Training Objective | 1.0 |
| V. Experimental Setup | 1.5 |
| VI. Results and Discussion | 3.5 |
| VII. Conclusion | 0.35 |
| References | 1.4 |
| 合计 | 约 14 页 |

投稿初稿不建议加入作者简介，以保留空间；最终稿按 TVT 要求处理。

---

# 15. Introduction 写作逻辑

## 第一段：车辆无线感知价值

说明 AMC 对：

- V2X 频谱感知；
- 智能接收机；
- 干扰诊断；
- 无先验信号处理；

的重要性。

## 第二段：车辆条件使 AMC 更困难

强调：

- 高速多普勒；
- 时变多径；
- 同频车辆碰撞；
- 复合干扰；
- 硬件损伤。

## 第三段：现有深度 AMC 的进展

概括：

- CNN/complex CNN；
- dual-stream；
- Transformer；
- lightweight；
- channel-robust feature。

## 第四段：三个明确 gap

1. 多数模型直接从污染表示分类，没有显式任务驱动的干扰分解；
2. 单一 mask/attention 无法表示信号与干扰在同一特征单元内重叠；
3. 仅在同分布 SNR 数据上验证，未充分研究跨车辆条件和未见干扰。

## 第五段：本文方法

一句话提出 VIMD-Net。

## 第六段：贡献

建议只写 3 个主贡献：

### Contribution 1

提出干扰条件化的 overlap-aware tri-mask，结合 adaptive overlap allocation 和 residual protection，在特征层连续分解调制主导、干扰主导和重叠分量。

### Contribution 2

提出 latent ideal-ratio mask teacher 与双支路多任务监督，并在软分解后的调制支路上实施 cross-condition contrastive learning，获得对车辆信道、速度和干扰变化更稳定的表示。

### Contribution 3

建立覆盖 3GPP V2X 信道、强/复合干扰、未见条件、连续车辆轨迹和 SDR 数据的评测闭环，并在精度、校准和复杂度三个维度验证方法。

理论分析可并入 Contribution 1 或 2，不必单独凑第四点。

---

# 16. 摘要骨架

英文摘要必须包含五句逻辑：

1. **背景与场景**：AMC 对 vehicular spectrum intelligence 重要；
2. **问题**：mobility and jamming entangle modulation semantics with nuisance features；
3. **方法**：VIMD-Net + tri-mask + latent teacher；
4. **学习策略**：multi-task + cross-condition contrastive；
5. **结果**：在 V2X channels、unseen jammers 和 SDR 上优于强基线，同时保持轻量。

初稿结果未出来前，不写虚构数字，用占位符：

```text
VIMD-Net improves Macro-F1 by XX.X percentage points over the strongest baseline
under SIR ≤ 0 dB, while requiring only X.XX M parameters.
```

---

# 17. 可能的审稿意见与预先封堵

## 17.1 “这只是 attention/mask 的另一个变体”

预先证据：

- 单 mask、SE、CBAM、双 mask、tri-mask 直接消融；
- LIRM teacher；
- feature-SIR；
- overlap ratio；
- 双支路任务可分性。

## 17.2 “车辆场景只是包装”

预先证据：

- 3GPP V2X channel；
- speed/Doppler grid；
- cross-speed；
- continuous trajectory；
- V2X-specific co-channel collision；
- SDR/半实物结果。

## 17.3 “多任务学习是已有技术”

写作策略：

- 不将 MTL 单独声称为首创；
- 强调其对掩码分解和支路语义的结构监督；
- 展示辅助任务删除后 mask 退化。

## 17.4 “仿真数据不能代表现实”

预先证据：

- session-held-out SDR；
- zero-shot + calibrated；
- 仿真参数由实测统计反标定；
- 报告 synthetic-real gap，而不是回避。

## 17.5 “多个损失堆叠，贡献难归因”

预先证据：

- A0–A7；
- 固定训练预算；
- 逐项增益；
- loss sensitivity；
- 公开默认超参数。

## 17.6 “掩码没有可辨识性”

预先证据：

- 训练期 LIRM；
- mask correlation；
- clean/jammer branch probe；
- signal retention/leakage；
- oracle upper bound。

## 17.7 “与 2025/2026 soft-mask 工作重叠”

Related Work 必须直接引用并区分：

- 其单频域 mask/雷达 attention；
- 本文三路 overlap、车辆干扰条件化、潜在教师、双支路 MTL 与跨条件对比。

---

# 18. 风险、触发门与降级路线

| 风险 | 识别信号 | 处理 | 降级 |
|---|---|---|---|
| Mask collapse | \(M_s\) 近全 1/0 | teacher warm-up、温度、冻结 backbone | 双 mask + residual |
| 过度净化 | 无干扰性能下降 | 增大 residual、降低 mask loss | 仅将 mask 作特征增强 |
| MTL 负迁移 | AMC 主指标下降 | uncertainty weighting/PCGrad | 去掉 Doppler/SNR 头 |
| Contrastive collapse | 类间距下降 | 延迟启用、hard negative 限制 | supervised consistency |
| 合成—实测域差 | zero-shot 大幅下降 | 反标定仿真、BN/LayerNorm 微调 | 论文降低工程主张 |
| 算法收益不足 | 对强基线 <3 pp | 检查数据难度与 mask 物理量 | 改投 DSP/Signal Processing |
| 车辆关联不足 | 普通 RadioML 为主 | 强制增加 V2X 轨迹与实测 | 不投 TVT |
| 页数超限 | >14 页 | 移除非核心分支、压缩图表 | 只保留 3 个贡献 |

---

# 19. 项目时间表

## 旗舰 18 周版本

| 周次 | 工作 | 阶段门 |
|---|---|---|
| W1–W2 | TVT/AMC/soft-mask/contrastive 文献审计，冻结 gap | 新颖性矩阵通过 |
| W3–W4 | V2X-JAM-AMC 生成器与数据审计 | SNR/SIR/功率测试通过 |
| W5–W6 | B0、CNN、ResNet、RepCCNet、dual-stream 基线 | 基线稳定 |
| W7–W8 | Tri-mask + residual + environment encoder | 不塌缩，主任务不下降 |
| W9 | LIRM teacher | mask 相关性与泄漏指标改善 |
| W10 | Multi-task heads | 困难区增益为正 |
| W11 | Cross-condition contrastive | 未见条件增益为正 |
| W12 | 全量训练、三随机种子 | 主阶段门通过 |
| W13 | 消融和超参数敏感性 | 贡献可归因 |
| W14 | 未见 jammer/speed/channel | 泛化阶段门通过 |
| W15 | SDR/半实物封存测试 | 实测收益为正 |
| W16 | 统计检验、复杂度与轨迹实验 | 证据闭环 |
| W17 | TVT 14 页初稿与图表 | 页数/格式通过 |
| W18 | 内部模拟审稿、修改、投稿 | submission-ready |

## 最短 12 周版本

只能在已有可靠数据生成器和 SDR 数据的前提下执行；不得通过删除未见条件和实测来“压工期”。

---

# 20. 投稿前 readiness gate

## G0：范围适配

- 标题和摘要明确 high-mobility vehicular；
- 系统模型包含时变车辆信道和干扰；
- 实验包含 V2X 变量。

## G1：数据可信

- 无源序列泄漏；
- SNR/SIR 功率校验；
- 会话级封存；
- manifest 可复核。

## G2：算法有效

- 全场景 ≥3 pp；
- 困难区 ≥5 pp；
- clean 保持性通过。

## G3：机制成立

- mask teacher correlation 为正且稳定；
- feature-SIR gain 为正；
- tri-mask 优于单/双 mask；
- overlap 分支确有贡献。

## G4：泛化成立

- 未见 jammer；
- 未见 speed；
- 未见 channel；
- 三者至少两项达到显著增益。

## G5：工程性成立

- ≤2 M 参数；
- 时延可接受；
- SDR 正收益；
- 轨迹 flip/outage 改善。

## G6：写作合规

- 初稿 ≤14 页；
- 所有图可读；
- 引用覆盖最新 2024–2026 工作；
- 不虚构性能；
- 贡献不夸大“first”；
- AI 使用按 TVT 当前规则处理。

只有 G0–G6 全部通过，才建议正式投 TVT。

---

# 21. 论文与专利/工程化的衔接

| 论文模块 | 原专利模块 | 后续工程作用 |
|---|---|---|
| Environment encoder | 干扰适配模块 | 离散模板升级为连续条件向量 |
| Overlap-aware tri-mask | 干扰抑制层 | 硬剔除升级为可旁路软分解 |
| LIRM teacher | 干扰贡献度计算 | 提供训练期物理监督 |
| Modulation branch | 调制分类层 | 提升强干扰识别能力 |
| Jammer/quality heads | 干扰类型信息 | 输出干扰与质量估计 |
| Confidence calibration | 关联校验接口 | 后续规则软融合 |
| Residual gate | B0 回退 | 净化失效时保护原始信息 |

论文阶段不把规则库和调制推荐并入模型。论文验收后再进入：

1. 识别/推荐双输出；
2. 规则置信融合；
3. 开放集拒识；
4. ONNX/INT8；
5. FPGA/嵌入式；
6. 专利从属权利点设计。

在论文公开前，应由专利代理人完成新增技术点的可专利性与公开时序审查。

---

# 22. 最终决策

## 22.1 推荐锁定的旗舰版本

\[
\boxed{
\text{VIMD-Net}
=
\text{Complex-aware Encoder}
+
\text{Conditioned Tri-Mask}
+
\text{Latent Mask Teacher}
+
\text{Multi-Task Supervision}
+
\text{Cross-Condition Contrastive Learning}
}
\]

## 22.2 论文真正的主张

不是：

> “本文设计了一个更深的网络。”

而是：

> **本文提出一种面向高机动车辆强干扰环境的调制—干扰重叠表征分解方法，使模型在不硬删除有效信息的情况下学习干扰不变的调制语义。**

## 22.3 最值得守住的三项贡献

1. **Overlap-aware tri-mask**；
2. **Latent ideal-ratio mask teacher**；
3. **Vehicular cross-condition contrastive learning**。

多任务头、TCN、复杂值前端和残差门用于构成完整方法，但不应被包装成彼此独立的“创新点”。

---

# 参考文献与标准（首轮核心清单）

[1] IEEE Vehicular Technology Society, “IEEE Transactions on Vehicular Technology—Scope and Publishing Policies,” accessed Jul. 27, 2026.

[2] IEEE Vehicular Technology Society, “IEEE Transactions on Vehicular Technology—Instructions for Authors,” accessed Jul. 27, 2026.

[3] 3GPP TR 37.885 V15.3.0, “Study on Evaluation Methodology of New Vehicle-to-Everything (V2X) Use Cases for LTE and NR,” 2019.

[4] F. Meng, P. Chen, L. Wu, and X. Wang, “Automatic Modulation Classification: A Deep Learning Enabled Approach,” *IEEE Transactions on Vehicular Technology*, vol. 67, no. 11, pp. 10760–10772, 2018, doi: 10.1109/TVT.2018.2868698.

[5] Y. Wang, J. Yang, M. Liu, and G. Gui, “LightAMC: Lightweight Automatic Modulation Classification via Deep Learning and Compressive Sensing,” *IEEE Transactions on Vehicular Technology*, vol. 69, no. 3, pp. 3491–3495, 2020, doi: 10.1109/TVT.2020.2971001.

[6] A. Tu, Y. Lin, C. Hou, and S. Mao, “Complex-Valued Networks for Automatic Modulation Classification,” *IEEE Transactions on Vehicular Technology*, vol. 69, 2020, doi: 10.1109/TVT.2020.3005707.

[7] Z. Zhang, H. Luo, C. Wang, C. Gan, and Y. Xiang, “Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure,” *IEEE Transactions on Vehicular Technology*, vol. 69, no. 11, pp. 13521–13531, 2020, doi: 10.1109/TVT.2020.3030018.

[8] X. Fu, G. Gui, Y. Wang, H. Gacanin, and F. Adachi, “Automatic Modulation Classification Based on Decentralized Learning and Ensemble Learning,” *IEEE Transactions on Vehicular Technology*, vol. 71, no. 7, pp. 7942–7946, 2022, doi: 10.1109/TVT.2022.3164935.

[9] N. Tang, X. Wang, F. Zhou, S. Tang, and Y. Lyu, “Reparameterization Causal Convolutional Network for Automatic Modulation Classification,” *IEEE Transactions on Vehicular Technology*, vol. 73, no. 6, pp. 8576–8583, 2024, doi: 10.1109/TVT.2024.3361928.

[10] B. Ren, K. C. Teh, H. An, and E. Gunawan, “OFDM Modulation Classification Using Cross-SKNet With Blind IQ Imbalance and Carrier Frequency Offset Compensation,” *IEEE Transactions on Vehicular Technology*, vol. 73, pp. 8389–8403, 2024, doi: 10.1109/TVT.2024.3356606.

[11] S. Huang, Y. Chen, J. He, S. Chang, and Z. Feng, “Channel-Robust Automatic Modulation Classification Using Companding Spectral Quotient Cumulants,” *IEEE Transactions on Vehicular Technology*, vol. 73, no. 11, pp. 17749–17753, 2024, doi: 10.1109/TVT.2024.3416288.

[12] Y. Dong, R. Zhai, Y. Zhong, Z. Rong, Y. Wang, and C. Wang, “A Novel Distributed Solution for Automatic Modulation Classification Based on Federated Learning and Modified LSTM,” *IEEE Transactions on Vehicular Technology*, vol. 74, no. 8, pp. 12290–12302, 2025, doi: 10.1109/TVT.2025.3551765.

[13] J. Cai, F. Gan, X. Cao, and W. Liu, “Signal Modulation Classification Based on the Transformer Network,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 8, pp. 1348–1357, 2022, doi: 10.1109/TCCN.2022.3176640.

[14] X. Tian et al., “A Survey on Deep Learning Enabled Automatic Modulation Classification Methods: Data Representations, Model Structures, and Regularization Techniques,” *Signal Processing*, vol. 242, Art. no. 110444, 2026, doi: 10.1016/j.sigpro.2025.110444.

[15] Y. Li et al., “A Novel Automatic Modulation Recognition Algorithm for OFDM Signals Based on FAFT,” *Scientific Reports*, vol. 16, Art. no. 9614, 2026, doi: 10.1038/s41598-025-33752-7.

[16] C. Huang et al., “A Low-SNR-Adaptive Temporal Network With Smart Mask Attention for Radar Signal Modulation Recognition,” *Digital Signal Processing*, vol. 168, Art. no. 105640, 2025/2026, doi: 10.1016/j.dsp.2025.105640.

[17] D. Liu, P. Wang, T. Wang, and T. Abdelzaher, “Self-Contrastive Learning Based Semi-Supervised Radio Modulation Classification,” arXiv:2203.15932, 2022.

[18] C. Wang et al., “Modulation Consistency-Based Contrastive Learning for Self-Supervised Automatic Modulation Classification,” arXiv:2605.11875, 2026.

---

# 附录 A：给 Codex/开发智能体的首轮执行口令

```text
目标：实现 TVT 旗舰论文 VIMD-Net 的最小可信闭环，不进行工程规则库、开放集、FPGA 和调制推荐开发。

必须按序执行：
1. 冻结数据字典、调制类别、SNR/SIR、车辆信道、干扰类型和 split manifest。
2. 编写功率/SNR/SIR/无泄漏单元测试，未通过不得训练。
3. 复现 CNN、ResNet1D、Complex-CNN、Dual-Stream、RepCCNet 基线。
4. 实现 shared encoder + single soft mask，作为最小对照。
5. 实现 conditioned tri-mask + overlap allocation + residual gate。
6. 实现 EMA latent ideal-ratio mask teacher。
7. 加入 jammer multi-label、SNR/SIR/Doppler quality heads。
8. 加入 cross-condition supervised contrastive loss。
9. 按 A0–A7 自动运行消融；所有实验固定数据、训练预算和随机种子。
10. 自动输出 Accuracy、Macro-F1、Worst Recall、ECE、SNR/SIR heatmap、mask retention/leakage、Params、MACs 和 latency。
11. 先跑 1k 样本 smoke test，再跑 20k 开发集，阶段门通过后才启动全量训练。
12. 所有长任务支持断点续训；单卡仅保留一个主训练作业。
13. 封存测试集不得参与调参；所有最终表格由 CSV/JSON 自动生成。
14. 若 full model 在 SIR≤0 dB 相对最强基线提升不足 5 pp，先审计数据与 mask 机制，不得通过删除困难样本获得好看结果。
```
