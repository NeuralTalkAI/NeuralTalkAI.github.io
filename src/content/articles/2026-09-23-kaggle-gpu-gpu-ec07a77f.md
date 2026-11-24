**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**图形性能指数**；逆向工程；:mdnice-bold-italic[结构化预测]；XGBoost；**特征工程**；Kaggle

> 本文完整复盘了 Kaggle 竞赛「GPU Performance Prediction Challenge」的参赛过程与夺冠方法。**该竞赛要求仅依据 GPU 硬件规格（流处理器数量、显存位宽、时钟频率等）预测主办方自定义的「图形性能指数」（Graphics Performance Index, GPI）**，以原始空间的均方根误差（RMSE）为评估指标。

**前七轮穷尽通用机器学习手段** 😅——LightGBM/XGBoost 选型、目标变换、特征工程、多 seed 集成与外部数据库增强——将公开榜分数从 0.20754 降至 0.18150，逼近通用方法的性能天花板，也参考了 TileSight。

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/01.jpg)

- GPU Performance Prediction Challenge: Predict the Graphics Performance Index (GPI) from GPU hardware specifications including cores, clock speeds, memory, and bus width.
- https://www.kaggle.com/competitions/gpu-performance-prediction-challenge/leaderboard

::: related-title
**相关推荐**
:::

::: related-links
- **[从GEMM、FA再到分布式vLLM：TileSight 用三层 tile 抽象击穿 GPU 性能建模黑盒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904280&idx=1&sn=91a08cdc1338df361894a581f5ee9fb8&scene=21#wechat_redirect)**
- **[解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**

:::

> :mdnice-bold-italic[随后通过观察训练集中同一 GPU 型号的「超频变体」]，**发现 GPI 是一个近乎确定性的函数，反向推导出隐藏公式** 🤣 $\mathrm{gpi} \approx C(\text{静态规格}) \cdot \mathrm{fp32}^{0.55}$，并据此构建「查表 + 回归」的两阶段结构化模型。这个过程中也发现竞赛数据来自我们先前分享的一篇文章的 GPU DataBase 开源项目，通过多字段规格指纹匹配（覆盖率 49.1%）。

[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)

公开榜分数最终降至 0.05543，位列第 1 名，较第二名（0.10781）领先约 48%；同时第二名本身亦为离群值，暗示前两名均掌握了隐藏公式😅。

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/02.jpg)

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/03.jpg)

::: image-caption
图 1：九轮迭代的公开榜分数轨迹，直观展示了这九轮的分数轨迹：前七轮在 0.18 附近反复震荡（CV 噪声淹没了真实增益），第八轮对公式的洞察一举将分数拉低近 70%，第九轮则因验证方式失误而小幅回退。图中不含 v2，因为该轮未提交。
:::

本文记录每一轮提交的方法考量与经验教训，其中「CV 提升不等于公开榜提升」「切分方式必须匹配测试集分布」两条认识最具普适价值，可为结构化回归竞赛提供方法论参考。

## 本文目录

- 一、引言
- 二、问题定义与数据
  - 2.1 任务描述
  - 2.2 数据集
  - 2.3 特征相关性分析
- 三、方法演进：九轮迭代的完整复盘
  - 3.1 第一轮（v1）：LightGBM 基线
  - 3.2 第二至三轮（v2、v3）：模型选型与目标变换
  - 3.3 第四轮（v4）：XGBoost depth=3 + log1p 甜点
  - 3.4 第五至六轮（v5、v6）：过拟合的两条反例
  - 3.5 第七轮（v7）：外部数据库类别特征
  - 3.6 关键转折：发现 GPI 的确定性结构
  - 3.7 第八轮（v8）：结构化模型夺冠
  - 3.8 第九轮（v9）：特征工程的回归与关键教训
  - 3.9 特征选择的演进脉络
- 四、核心方法：两阶段结构化模型
  - 4.1 公式推导
  - 4.2 两阶段预测流程
  - 4.3 关键实现细节
- 五、实验与结果
  - 5.1 误差结构分解
  - 5.2 C 预测的模型对比
  - 5.3 C 的构成分析
  - 5.4 指数与超参数鲁棒性
  - 5.5 最终排行榜
- 六、讨论
  - 6.1 关键经验教训
  - 6.2 局限与未解决问题
  - 6.3 方法论总结
  - 6.4 与第二名的差距：确定性结构带来的「阶跃式」优势
- 七、结论
- 参考文献

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/04.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、引言

> 近年来，随着深度学习与通用计算的爆发，GPU 的性能评估在硬件选型、算力调度与系统设计中日益重要。:mdnice-bold-italic[然而，GPU 的「性能」并非单一指标，是游戏渲染、通用计算、AI 加速、光线追踪与功耗等多个维度的综合。如何仅从一份「规格表」出发，预测一个综合性能指数，是一个兼具工程价值与建模挑战的回归问题。]

Kaggle 竞赛「GPU Performance Prediction Challenge」正是围绕这一问题设计：**给定约 10,000 条 GPU 训练样本（每条含 18 个硬件规格字段与一个目标值 GPI），要求预测 2,709 条测试样本的 GPI**。评估指标为原始空间的 RMSE。:mdnice-bold-italic[值得注意的是，主办方刻意隐藏了 GPI 的具体计算公式（仅说明其为五个维度的加权和，但省略了权重），这为竞赛赋予了「黑箱回归」甚至「逆向工程」的色彩。]

我们提交的作品最终以 0.05543 的分数夺得第一名。

::: block-1
这一结果并非来自复杂的模型堆叠或大规模集成🤣，**而【真的】是来自对问题本质的深入洞察——GPI 本质上是一个由硬件规格决定的确定性函数**。:mdnice-highlight[本文的目的是完整记录这一过程：从最初的基线模型，到每一次提交背后的思路、实验与结论，再到最终的突破，将其组织为一篇可供复现与借鉴的技术文章。]
:::

本文的主要贡献如下：

1. 完整记录了九轮迭代的完整脉络，包括每次提交的模型配置、CV 与公开榜分数、以及后续的归因分析；
2. 揭示并论证了 GPI 的确定性结构 $\mathrm{gpi} \approx C(\text{静态规格}) \cdot \mathrm{fp32}^{0.55}$，并给出严谨的统计证据（组内 $z$ 值标准差精确为零）；
3. **总结了一套面向结构化回归竞赛的方法论**，尤其是「用与测试集分布一致的切分方式验证模型」这一关键教训。

## 二、问题定义与数据

### 2.1 任务描述

> 给定一条 GPU 的硬件规格记录 x，预测其图形性能指数 $y = \mathrm{gpi\_value}$。这是一个监督回归任务，评估指标为原始空间的均方根误差：

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2}
$$

其中 $N=2709$ 为测试样本数，$\hat{y}_i$ 为预测值。

**主办方说明 GPI 是「游戏、计算、AI、光线追踪、功耗」五个维度的加权和**，但权重被刻意省略，这暗示 GPI 背后存在一个可（在理论上）被恢复的确定性公式。

### 2.2 数据集

> 下面这样：

| 项 | 训练集 | 测试集 |
| --- | --- | --- |
| 样本数 | 10,000 | 2,709 |
| 特征数 | 18 | 18 |

特征字段可分为三类：

- 核心架构规格：cores（流处理器/CUDA 核心数）、sm_count（流式多处理器数）、tensor_cores、rt_cores、api_cuda（CUDA 计算能力版本）；
- 时钟与存储规格：base_clock、boost_clock、memory_clock、ram（显存容量）、bus_width（显存位宽）；
- 性能指标字段：fp16/fp32/fp64/bf16/tf32/int8_performance（各精度算力，TFLOPS）、multi_gpu。

目标变量 gpi_value 呈现强右偏分布：最小值为 0，最大值为 42.44，均值 6.955，中位数 4.8（偏度 1.596）。

> 数据质量方面存在若干值得注意的特点：

| 特点 | 具体描述 |
| --- | --- |
| 大量缺失值 | int8_performance、multi_gpu 列 100% 为空；bf16/tf32_performance 99.9% 为空；tensor_cores、rt_cores、sm_count、api_cuda 等存在 27%–64% 不等的缺失。 |
| 强共线性 | fp32_performance 与目标相关系数高达 0.9755，是最强的单一预测特征；其次是 sm_count(0.9505)、cores(0.9490)、rt_cores(0.9104)。 |
| 可重建的冗余特征 | 经验证，fp32_performance 几乎精确等于物理重建公式 $\mathrm{fp32} \approx \mathrm{cores} \times 2 \times \mathrm{boost\_clock} / 10^6$（相关系数 0.9731，比值中位数 1.0），说明该字段就是理论峰值算力。 |
| **隐藏的「超频变体」结构** | 同一 GPU 型号在训练集中以多个「时钟频率略有差异」的变体出现，:mdnice-bold-italic[这是后文推导公式的关键线索]。 |

### 2.3 特征相关性分析

> 下表 1 列出了主要特征与目标 GPI 的 Pearson 相关系数：

| 特征 | 与 GPI 相关系数 |
| --- | --- |
| fp32_performance | 0.9755 |
| sm_count | 0.9505 |
| cores | 0.9490 |
| rt_cores | 0.9104 |
| fp16_performance | 0.8811 |
| boost_clock | 0.6698 |
| api_cuda | 0.674 |
| fp64_performance | 0.0158（几乎无关） |

值得注意的是，**fp64_performance 与 GPI 几乎不相关（0.0158），这为后续的特征选择提供了第一个依据——完整的特征取舍脉络** （删死字段、识冗余、区分静态规格与性能规格）将在第 3 章逐轮展开。

## 三、方法演进：九轮迭代的完整复盘

> 本章按时间顺序记录每一次提交（共九轮）背后的动机、方法、实验结果与结论。下表 2 首先给出全局概览：

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/05.jpg)

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/06.jpg)

::: image-caption
图 1：九轮迭代的公开榜分数轨迹，直观展示了这九轮的分数轨迹：前七轮在 0.18 附近反复震荡（CV 噪声淹没了真实增益），第八轮对公式的洞察一举将分数拉低近 70%，第九轮则因验证方式失误而小幅回退。图中不含 v2，因为该轮未提交。
:::

### 3.1 第一轮（v1）：LightGBM 基线

动机：快速建立可提交的基线，验证数据通路与提交机制。

方法：采用 LightGBM 5 折交叉验证，做基础特征工程（新增 fp16/fp32 比值、每核心性能、显存带宽近似等 23 个特征），直接预测原始目标。特征层面此轮只做了一处「减法」：剔除 int8_performance、multi_gpu 两个全空字段与 bf16/tf32_performance 两个近乎全空的精度字段，其余字段一律保留。

结果：OOF CV RMSE = 0.289，公开榜分数 0.20754。

结论：数据通路打通。fp32 是核心特征，目标强右偏，需要后续处理。此时的特征选择仍停留在「删死字段」的朴素层面，尚未触及冗余与噪声字段。这是整个项目的起点。

### 3.2 第二至三轮（v2、v3）：模型选型与目标变换

动机：探索更强的模型与目标变换。

关键实验：

1. log1p 目标变换：将目标 $y$ 变换为 $\log(1+y)$ 后建模，显著提升（LightGBM 0.290→0.268）。这一发现贯穿后续所有轮次。
2. 模型对比：XGBoost 完胜 LightGBM/CatBoost（CatBoost 因本机 CPU 缺少 AVX2/FMA 指令集而无法导入）；线性模型表现极差（约 0.42），说明存在非线性交互。
3. 集成尝试：多模型简单平均反而拖累性能，说明在单模型明显占优时，弱模型集成有害。

结果：v3（XGB depth=5 + 目标编码 + 3seed）公开榜 0.19688，较基线提升 5.1%。

结论：XGBoost + log1p 是正确方向；目标编码与集成在该阶段无益。与模型选型同步完成的是一次关键的特征扩充：在 13 个原始字段之外，为 12 个右偏数值字段取 log1p 变换、为 8 个稀疏字段构造缺失指示、对 fp64/fp16 两个含极端离群值的字段做上界截断，并派生 3 个精度比值（fp16/fp32、fp64/fp32 等）与 7 个工程特征（每核心/每 SM 算力、显存带宽、算力带宽比等），将特征集从 v1 的基础集扩展为 43 维基准集。这套 43 维基准集在此后五轮中保持基本不变。

### 3.3 第四轮（v4）：XGBoost depth=3 + log1p 甜点

动机：确立最优的模型配置。

关键实验（深度与目标变换的系统扫描）：

- 深度扫描：depth=3 最优（raw RMSE 0.17472），depth 4/5/6/7 单调变差（0.195/0.226/0.225/0.245）——更深过拟合；
- 目标变换扫描：log1p 最优（0.17472），优于 sqrt（0.19471）和 raw（0.22490）。

结果：v4 公开榜 0.18381，成为后续所有优化的基准。

结论：XGBoost depth=3 + log1p 是该数据集的「甜点」。此时我们形成了「更简单、更强正则」的认识。特征层面，43 维基准集在这一轮被正式冻结——后续轮次的注意力从「特征集本身」转向「要不要往里面再加什么」。

### 3.4 第五至六轮（v5、v6）：过拟合的两条反例

这两轮是竞赛中最重要的「负面教材」，直接奠定了后续的方法论纪律。

v5（物理特征 + 5-seed）：**受 TileSight 论文（第一性原理 GPU 性能建模）启发，构造了算力带宽比、CUDA 架构世代拆分（主/次版本）等物理特征**。CV 从 0.17472 降至 0.16953，但公开榜反而变差至 0.18736。:mdnice-bold-italic[从特征选择的角度看，这是一次失败的「加法」：算力带宽比本质上是 fp32 与显存带宽之比，而二者都已在特征集内，新增特征只是让树模型重复编码同一物理量，CV 上的微小增益是过拟合而非信息增益。]

v6（7-seed 平均）：7 个随机种子的 XGBoost 平均使 CV 从 0.17472 降至 0.17268，但公开榜变差至 0.18537。

归因分析：偏差分析显示高端卡（$\mathrm{GPI} \ge 20$）系统性低估（$\mathrm{GPI} \ge 30$ 段 $\mathrm{bias} = -0.154$），且 train/test 分布几乎一致（$\mathrm{fp32} \ge 19.5$ 的占比 train 23.2% vs test 22.5%）。因此 CV 与 public 的约 $0.01$ 差距并非分布漂移，而是公开榜子集噪声 + 高端尾部方差所致。

关键教训：在 10,000 样本规模下，5 折 CV 的噪声约为 ±0.003，任何基于 CV 反复加复杂度的操作都会过拟合。CV 提升不等于公开榜提升。

### 3.5 第七轮（v7）：外部数据库类别特征

动机：尝试用外部数据（GitHub 上的 GPU 数据库）增强特征。

方法：从 RightNow-AI/RightNow-GPU-Database（2,824 个 GPU，Apache 2.0，数据源与比赛同源 TechPowerUp）通过多字段规格指纹匹配（覆盖率 49.1%），提取类别特征 memoryType、vendor、architecture、generation 做 one-hot 编码。

关键实验：

| 特征组合 | OOF raw |
| --- | --- |
| baseline | 0.17472 |
| +memoryType+vendor | 0.17106 ✅ |
| +architecture | 0.18589 ❌ |
| +generation | 0.17137 ❌ |

只有 memoryType + vendor 有效——这是逐项消融筛出来的：architecture 与 generation 因类别数过多、单类样本稀疏而拖累泛化，被剔除。

结果：v7 公开榜 0.18150（新最佳）。

结论：外部数据仅类别特征有微弱增益，数值型增强（tdp、process_size、die_size 等）全部无效，因为它们与现有规格（≈ cores×clock）高度冗余。此轮之后，我们已逼近「通用机器学习」在该问题上的天花板（约 0.17–0.18）。

### 3.6 关键转折：发现 GPI 的确定性结构

::: block-1
在通用方法逼近天花板后，我们开始追问问题的本质：:mdnice-bold-italic[既然主办方刻意隐藏了五个维度的权重，那么 GPI 是否由规格确定性地生成、而非包含不可约的随机噪声？这一怀疑引出一个具体的检验思路——寻找训练集中完全相同的硬件是否多次出现]：若同一 GPU 型号（静态规格完全相同）对应多个样本，且这些样本的 gpi 随某个规格（如时钟频率）单调变化，就说明 gpi 是被规格唯一决定的。**由此得到关键观察：**
:::

> **训练集中同一 GPU 型号（如 RTX 4090：cores=16384, sm=128, tensor=512, rt=128）以 79 个「超频变体」（clock-variant）出现** ——只有 boost_clock 变化（2520→2640），fp32 随之变化（82.58→86.51），而 gpi 单调递增（32.06→32.89）。:mdnice-bold-italic[这说明 GPI 是随规格（尤其 fp32）连续变化的确定性函数。]

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/07.jpg)

::: image-caption
图2：log(GPI) 对 log(fp32) 的散点——同一静态组内呈现斜率为 0.55 的平行直线族
:::

图 2 以五个不同规格（cores 从 384 到 16384）的静态组为例，横轴为 $\log(\mathrm{fp32})$、纵轴为 $\log(\mathrm{GPI})$。同一静态组内的点（仅时钟变化）严格落在一条斜率为 $0.55$ 的直线上，不同组之间的差异只体现在截距 $C(\text{静态规格})$ 上——这正是公式 $\mathrm{gpi} = C \cdot \mathrm{fp32}^{0.55}$ 的可视化证据。

基于此，我们定义了关键变量：

$$
z = \log(\mathrm{gpi}) - p \cdot \log(\mathrm{fp32})
$$

并在「静态特征组」内检验其恒定性。静态特征组定义为以下 9 个非性能列的取值组合：

> cores, sm_count, tensor_cores, rt_cores, base_clock, memory_clock, ram, bus_width, api_cuda

核心发现：在相同的静态特征组内，$z$ 的组内标准差中位数精确为零（587/2253 个组的 z-std == 0），对所有 $p \in [0.5, 0.6]$ 均成立。这意味着：

$$
\mathrm{gpi} \approx C(\text{静态规格}) \cdot \mathrm{fp32}^{p}
$$

其中 $C$ 是仅由静态规格决定的组常数。这是一个近乎精确的确定性公式。

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/08.jpg)

::: image-caption
图3：组内 z 标准差直方图——绝大多数组精确为零
:::

图 3 展示了所有（n≥2）静态组组内 z 标准差的分布：587 个组的标准差精确为零（图中红色高亮区），其余组也集中在接近零的极小范围，进一步印证了确定性结构的普遍性。

指数 $p$ 的估计：池化组内回归斜率 = $0.5662$；但以原始空间 RMSE 为准则，最优 $p = 0.55$（多 seed 验证 $p = 0.55/0.566/0.57$ 无显著差异）。

理论下限：组内精确线性拟合的 raw RMSE = $0.01888$（oracle floor），远低于当时的 $0.18$，说明存在巨大的提升空间。

### 3.7 第八轮（v8）：结构化模型夺冠

**这一发现本身即是一次决定性的特征选择**：原始特征被一分为二——9 个「静态规格」只以组键形式决定常数 $C$，性能字段中只有 fp32 以幂律 $f^{0.55}$ 进入公式，而 fp16/fp64/bf16/tf32 等其余精度字段被整体排除（第 5.3 节证实它们对 $C$ 几乎无解释力）。

这正是从「加法」到「减法」的转折点：不是继续往 43 维集里加特征，而【真的】**是把特征空间的结构本身改写**。基于此，我们构建了两阶段结构化模型（详见第 4 章）。v8 公开榜 0.05543，直接跃升至第 1 名，较上一轮 0.18150 提升 69%，较第二名领先 48%。

### 3.8 第九轮（v9）：特征工程的回归与关键教训

动机：在 rank 1 后，尝试对「未见组」的 C 预测做进一步特征工程。

方法：为 C 预测的 XGBoost 模型新增 tensor_ratio、rt_ratio、ram_per_bus、clock_ratio 等比值特征。

实验结果：整体 OOF 从 0.12055 降至 0.10602（看似提升），但公开榜变差至 0.06343。

归因：测试集的 364 个「未见」样本中，314 个是「clock-variant」（非时钟组在训练集出现过、但完整键因时钟不同而未见），50 个是「novel」（全新非时钟组）。用 clock-variant 专属切分重新验证发现，v9 特征在这些行上反而退化（z-OOF 0.02740 vs v8 的 0.02673）。v9 的 OOF 增益来自不会出现在公开榜上的「非 clock-variant 未见行」。

关键教训（全文最重要的一条）：**任何特征选择与模型改动，必须用与测试集分布一致的切分方式验证（本任务即 clock-variant 专属 CV），否则整体 OOF 会系统性误导** ——v9 的比值特征在整体 OOF 上「看似更优」，却在真正决定成绩的 clock-variant 行上反而退化。

### 3.9 特征选择的演进脉络

> 回顾九轮迭代，特征选择的思路经历了一条由「宽」到「窄」、由「加法」到「减法」再到「结构化拆分」的清晰演进：

1. 朴素删除（v1）：仅剔除全空与近乎全空的死字段，其余一律保留；
2. 构造与冻结（v2–v4）：通过对数变换、缺失指示与比值/派生特征，将特征集扩充并冻结为 43 维基准集；
3. 审慎加法（v5–v7）：物理特征、外部数值与类别特征被逐项加入、逐项消融，最终只有 memoryType/vendor 两个类别特征幸存——绝大多数「加法」被证明是冗余或过拟合；
4. 结构化拆分（v8–v9）：识别出确定性公式后，特征被彻底二分——9 个静态规格决定常数 $C$，fp32 以幂律进入，其余性能字段被整体排除，这是收益最大的一次「减法」。

这条脉络贯穿全文，其核心认识是：**特征选择的价值不在于「特征越多越好」，而在于识别哪些特征真正承载目标信息、哪些只是同一物理量的重复编码**。v8 的突破正是这一认识的极致体现——它不是在 43 维集上继续做加法，而是直接改写了特征空间的结构。

## 四、核心方法：两阶段结构化模型

本章详细描述最终夺冠模型（v8）的技术细节。

### 4.1 公式推导

沿用第 3.6 节的推导结论：设静态特征向量为 $s$（上述 9 列），性能特征 fp32 记作 $f$，隐藏公式为

$$
\mathrm{gpi} \approx C(s) \cdot f^{p}
$$

其中指数 $p$ 已确定为 $0.55$（估计方法与鲁棒性细节见第 3.6、5.4 节，此处不再重复）。定义辅助变量 $z = \log(\mathrm{gpi}) - p \cdot \log(f)$，在同一静态组 $s$ 内 $z$ 为常数（等于 $\log C(s)$），于是预测问题被拆解为两部分：

1. 对训练集中出现的静态组，$C(s)$ 可由组内 $z$ 的均值直接估计；
2. 对未见过的静态组，$C(s)$ 需由一个从 $s$ 到 $z$ 的回归模型预测。

### 4.2 两阶段预测流程

阶段一（seen 行，查表）：对于测试样本，若其静态组键在训练集中出现过，直接取该组的 z 均值作为预测，再还原：

$$
\mathrm{gpi}_{\mathrm{pred}} = \exp\left( z_{\mathrm{mean}} + p \cdot \log(\mathrm{fp32}) \right)
$$

由于组内 $z$ 标准差为零，此阶段误差趋近于零（实测 raw RMSE $\approx 0.027$，接近 oracle 下限 $0.019$）。

阶段二（unseen 行，回归）：对于静态组键未出现的测试样本，用 XGBoost（depth=4）从静态特征预测 z，再按上式还原。这是误差的主要来源。

### 4.3 关键实现细节

> 以下几个实现细节对结果至关重要，任何一处疏忽都会导致显著退化：

1. 静态组键必须用字符串 join + fillna(-999)。若直接使用 set_index 或 dict 分组，含 NaN 键的组会被静默丢弃，导致「seen」判定错误（此前曾因此把 seen 数误算为 908/2709，正确应为 2345/2709）。
2. fp32 缺失处理：测试集有 2 行 fp32 为 NaN，用训练集中位数填充；训练集有 7 行 $\mathrm{gpi} \le 0$，因 $\log(0)$ 无定义需剔除。
3. z 模型的输入特征：对每个静态特征构造 log 与缺失指示，并加入 cores_per_sm、mem_bw、mem_per_core、bw_per_core 等派生特征。
4. XGBoost 参数：objective=reg:squarederror，learning_rate=0.05，max_depth=4，n_estimators=800，subsample/colsample_bytree=0.9，tree_method=hist。深度超过 4 会过拟合（depth6/7 的 z-OOF 变差）。

## 五、实验与结果

### 5.1 误差结构分解

> 将测试集按静态组是否出现分类，得到误差分解：

| 类别 | 数量（/2709） | raw RMSE | 说明 |
| --- | --- | --- | --- |
| seen（静态组在训练集） | 2345 | ≈0.027 | 查表，近 oracle |
| clock-variant（非时钟组在训练集，完整键不在） | 314 | ≈0.27–0.30 | 需回归，主导误差 |
| novel（全新非时钟组） | 50 | $z \approx 0.052$ | 最难，极端规格 |

结论：seen 行几乎无误差，误差几乎全部来自 unseen 行（尤其 clock-variant）。

> 注：上表 raw RMSE 是训练集 5 折 OOF 估算值，其中 clock-variant 的 0.27–0.30 偏保守——OOF 中模型只用 80% 的静态组训练 $C(s)$，外推能力弱于用全量训练集的情况。由最终总分 0.05543 反推（见第 6.4 节），实际测试集中 364 个 unseen 行的 raw RMSE 约为 0.13–0.14。

### 5.2 C 预测的模型对比

对 unseen 行的 C（即 z）预测，我们系统对比了多类模型（OOF z-RMSE）：

| 模型 | unseen-z RMSE |
| --- | --- |
| XGBoost depth=4 | 0.0289 |
| GradientBoosting d4 | 0.0292 |
| ExtraTrees | 0.0301 |
| Ridge | 0.0495 |
| KNN(5, 距离加权) | 0.0568 |
| ElasticNet | 0.0514 |

结论：树模型显著优于线性模型与 KNN；XGBoost 最佳。线性模型虽然在 $\log C$ 的样本内拟合上 $R^2 \approx 0.99$，但在 OOF 上严重过拟合。

### 5.3 C 的构成分析

对组常数 $C$ 的特征重要性分析显示，ram（显存容量）是决定 $C$ 的最主要因素（重要性 0.625），其次是 $\mathrm{memory\_clock} \times \mathrm{bus\_width}$（显存带宽，0.155）、bus_width（0.149）、cores（0.023）。全局拟合 $\log C$ 对 $\log(\mathrm{ram})$ 的 $R^2 = 0.49$、斜率 $0.12$，说明 $C$ 主要但并非纯粹由显存容量决定。

此外，各精度性能指标与 fp32 的比值（fp16/fp64/bf16/tf32/int8 相对 fp32）对 $C$ 几乎无解释力（OLS $R^2 = 0.06$），再次印证这些字段是冗余或噪声。

### 5.4 指数与超参数鲁棒性

- 指数：多 seed 验证 $p = 0.55/0.566/0.57$ 的 clock-variant z-RMSE 分别为 $0.02677/0.02666/0.02663$，overall raw 分别为 $0.11218/0.11331/0.11226$，均在噪声内（$\pm 0.0005$ / $\pm 0.003$），无显著差异，故保持 $p = 0.55$。
- 学习率/树数：对 $(lr, n\_estimators)$ 组合做 3-seed clock-variant CV，所有差异均在噪声内（clock-z $\pm 0.0005$，overall $\pm 0.003$），v8 的 $(lr = 0.05, n = 800)$ 已是局部最优。

### 5.5 最终排行榜

| 名次 | 团队 | 分数 |
| --- | --- | --- |
| 1 | Kirisame Marisa（我们） | 0.05543 |
| 2 | 崔洧瑄 | 0.10781 |
| 3 | sainadhureddy | 0.14295 |
| 4 | ahbab zahin | 0.14395 |
| 5 | রোদশী | 0.15155 |
| 6 | Elias William | 0.15484 |
| 7 | asterask | 0.17602 |
| 8 | Prakash | 0.18015 |

我们以 0.05543 领先第二名 0.10781 约 48%，且第二名同样是离群值（比第三名 0.14295 好 25%），说明前两名都掌握了「公式」这一关键信息，而其余队伍仍停留在通用机器学习方案。

## 六、讨论

### 6.1 关键经验教训

> 回顾全程，以下几条认识最具普适价值：

1. 先理解问题，再堆模型。本项目最大的提升（0.18→0.055）来自对问题结构的洞察（GPI 是确定性函数），而非任何模型/特征微调。通用 ML 方案在此问题上存在约 0.17 的硬天花板。
2. CV 提升不等于公开榜提升。v5（物理特征）、v6（7-seed）、v9（新特征）三轮都出现了 CV 变好但 public 变差的情况。在样本量小、标签噪声存在时，5 折 CV 的方差足以淹没真实增益。
3. 切分方式必须匹配测试集分布。v9 的教训最深刻：整体 OOF 显示提升，但该提升来自测试集中不存在的样本类别，导致 public 反而退化。正确的做法是复现测试集的构成（本任务即 clock-variant 专属切分）来做验证。
4. 确定性结构是巨大的可利用先验。当发现「相同静态规格 → 相同 z」时，问题从「回归」退化为「查表 + 少量外推」，误差从 0.18 直接降到 0.027。

### 6.2 局限与未解决问题

尽管取得了 rank 1，仍有若干问题未完全解决：

1. 119 个非确定行：训练集中有 119 行（1.2%）具有完全相同的特征向量却对应不同 gpi（组内 std $\approx 0.136$，最大 $0.49$）。这可能是主办方注入的标签噪声，或存在未观测的隐藏特征，它限制了可达到的真实下限。
2. $C(s)$ 的精确解析形式未知：XGBoost 对 $C$ 的预测是黑盒近似；$\log C$ 对 $\mathrm{ram}/\mathrm{mem\_bw}/\mathrm{bus\_width}$ 的依赖非纯对数线性（OLS $R^2 = 0.99$ 但残差 $6\%$）。若能解析 $C$，可进一步改善 unseen 行（尤其 50 个 novel 行）的预测。
3. 50 个 novel 行的预测仍是瓶颈：极端/稀有规格组合（如 $\mathrm{ram} = 1$、$\mathrm{bus\_width} = 5120$ 等）缺乏训练先验，z-RMSE 高达 $0.052$。

### 6.3 方法论总结

本项目的方法论可概括为「四步曲」：

1. 建立基线：快速打通数据与提交通路，确立可对比的分数锚点；
2. 穷尽通用方法：模型选型、目标变换、特征工程、集成、外部数据，同时记录每一轮的 CV 与 public 分数；
3. 挖掘问题结构：通过数据观察（超频变体、组内恒定性）发现确定性公式；
4. 结构化建模 + 严格验证：将先验知识编码为模型结构，并用与测试集一致的切分方式验证。

### 6.4 与第二名的差距：确定性结构带来的「阶跃式」优势

> 排行榜呈现明显的两段式分布：**我们（0.05543）与第二名（0.10781）构成第一梯队**，第三名及以后（0.14295–0.18015）构成第二梯队。第三名 0.14295 虽优于我们前七轮通用方案的最优分（0.18150），但仍停留在 0.10 以上的通用机器学习区间，未触及 0.10 以下的「公式区」，**说明第二梯队尚未破解确定性结构；而前两名之间仍有约 48% 的差距，这需要更细的分解才能解释。**

我们对误差的来源分解（第 5.1 节）表明：

::: block-1
2345 个 seen 行通过查表达到近 oracle 精度，总误差几乎全部来自 364 个 unseen 行的 $C(s)$ 外推。据此可反推——假设 seen 行误差约为 0.02–0.03（oracle 下限为 0.019），则我们的 364 个 unseen 行的 raw RMSE 约为 0.13–0.14；而第二名若要达到 0.10781，在 seen 行同样精确的前提下，其 unseen 行 raw RMSE 需达约 0.29，是我们的两倍以上。:mdnice-highlight[也就是说，前两名都发现了公式，但差距并不在「是否发现公式」，而在如何对未见静态组的 $C(s)$ 做外推。]
:::

> 我们在外推上做得更好的原因可归结为三点：

1. 静态组键用字符串 join + fillna(-999) 构造，保证「已见/未见」判定正确（第 4.3 节），避免把大量样本误判为 unseen 而白白损失查表收益；
2. 对 $C(s)$ 单独建模并系统对比模型（第 5.2 节），而非复用全局回归器；
3. 用 clock-variant 专属切分做验证（第 3.8 节），避免了被整体 OOF 误导。这三点叠加，构成了我们相对第二名约 2 倍于「unseen 样本误差」的优势。

## 七、结论

> 本文完整复盘了**我们在「GPU Performance Prediction Challenge」中从基线到冠军的全过程**。核心贡献在于：通过观察训练集中同一 GPU 型号的超频变体，反向推导出隐藏的 GPI 生成公式 $\mathrm{gpi} \approx C(\text{静态规格}) \cdot \mathrm{fp32}^{0.55}$，并据此构建了「查表 + XGBoost 回归」的两阶段结构化模型，最终以 0.05543 的分数夺得第 1 名，领先第二名约 48%。

这一过程清晰地展示了**「理解问题本质」相对于「堆砌模型复杂度」的巨大价值**：九轮迭代中，前七轮的通用机器学习优化合计仅将分数从 0.20754 降至 0.18150（-12.5%），而第八轮对公式的洞察一举将分数降至 0.05543（-69%）。**同时，本文反复强调的「CV 提升不等于公开榜提升」与「切分方式必须匹配测试集分布」两条教训，对任何结构化回归竞赛都具有警示意义。**

我们希望本文的完整记录——包括成功与失败的反例——能为后续参与相似竞赛的研究者提供可复现的路线图与方法论参考。

## 参考文献

1. Williams S., Waterman A., Patterson D. Roofline: An Insightful Visual Performance Model for Multicore Architectures. Communications of the ACM, 2009.
2. TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model. arXiv:2607.22432.
3. RightNow-AI/RightNow-GPU-Database. https://github.com/RightNow-AI/RightNow-GPU-Database (Apache 2.0).
4. Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.
5. Kaggle. GPU Performance Prediction Challenge. https://www.kaggle.com/competitions/gpu-performance-prediction-challenge

::: related-title
**相关推荐**
:::

::: related-links
- **[超越仿真，精准预测推理性能！上交与阿里提出SynPerf：一种混合分析与机器学习的 GPU 性能预测框架，实现 6.1%内核误差与 1.7 倍加速！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897849&idx=1&sn=0e780adf7b65d20826ef5e3b2330cda2&scene=21#wechat_redirect)**
- **[伯克利开源 CHIA：进化 Agent 打通 AI 芯片全链路，单框架实现仿真对齐/RTL 生成/时序优化五大工业级硬件自动化，最高 1.97 倍性能提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903506&idx=1&sn=27650a6cbe9467c5d3d0608e8839b15d&scene=21#wechat_redirect)**
- **[解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**

:::
