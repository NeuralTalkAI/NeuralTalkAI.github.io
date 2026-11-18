关键词：**GPU 内核基准测试**、:mdnice-bold-italic[光速上限（硬件极限）]、**智能体优化**、AI 加速器、**架构多样性**

> NVIDIA 这项工作 SOL-ExecBench 表明，智能体生成方案填平了从基线到硬件极限约 73% 的差距，完全达到初级甚至中级算子工程师的水平，已经能够胜任中等复杂度算子（L1、L2）的自动化优化，且能有效利用低精度硬件特性（Quant）。

注：这里的 0.5 基线是智能体优化器对 PyTorch 参考实现进行优化后得到的性能，而非原始 PyTorch 的 eager 执行水平。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/01.webp)

::: image-caption
图 10(a) 的直方图横坐标为 SOL 分数（0 到 1），纵坐标为基准测试问题的数量，“S” 标注了中位数位置。数据显示，智能体优化器在 235 个问题上的 SOL 分数整体呈偏右分布，中位数达到 0.732，显著高于代表评分基线的 0.5。这样设定的目的是构建一个更强、更稳定的评分起点。这表明智能体普遍能填补从软件基线到硬件极限之间近一半以上的性能差距，具备较强的通用优化能力。同时，分布仍未逼近 1.0，说明当前自动化优化距离专家级内核仍有显著提升空间。该图验证了 SOL 分数作为评估指标的稳定性与区分度，也凸显了 SOL-ExecBench 在衡量真实硬件优化进度上的有效性。
:::

人类算子工程师不再需要逐一手写这些算子，而必须转向深度硬件理解、极限性能调优、系统集成与验证能力将成为更高价值的核心能力。

此外，Roofline 模型为本文 SOL 指标提供理论框架，以峰值算力与内存带宽约束内核性能。Orojenesis 基于片上缓冲区推导更紧致的数据移动上限，修正朴素 Roofline 对低数据复用算子的性能高估。

- [ISCA'24 揭示SRAM与计算资源最优配比！DSE创新方法Orojenesis：张量算法的可达数据移动与操作强度界限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891559&idx=1&sn=50b6d88425d3829e823a33a38b915801&scene=21#wechat_redirect)

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/02.webp)

::: image-caption
表 1：GPU内核生成基准测试对比。不难看出，现有基准测试均无法同时满足覆盖前沿模型、支持低精度/训练负载、对标硬件极限、适配Blackwell新硬件四大核心要求；SOL-ExecBench是唯一覆盖BF16/FP8/NVFP4精度、包含反向传播、专为Blackwell硬件设计、以硬件SOL上限为评估指标的基准测试。
:::

::: block-1
本文 SOL 上限采用 NVIDIA 内部方法，以硬件光速（极限）性能百分比衡量，并在其基础上研发 SOLAR 流水线，可从 PyTorch 参考实现自动推导紧致硬件级 SOL 上限。

**更多细节见后文。**
:::

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/03.webp)

- **SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits**
- https://arxiv.org/pdf/2603.19173
- **开源代码：https://github.com/NVIDIA/SOL-ExecBench**
- 1.2 万字，阅读 40 分钟，**播客 28 分钟**

相关推荐

- [决战性能之巅！MLSys 2026 CUDA 内核优化大赛开启，用代码挑战 B200 GPU 极限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896612&idx=1&sn=4e667f93479948119cf7688933ca966c&scene=21#wechat_redirect)
- [AKG Kernel Agent：多智能体联手，实现跨 DSL 多硬件平台 Kernel 的“一键生成”与迁移](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895857&idx=1&sn=3bce0710d38ccd3a772869ac46a24c83&scene=21#wechat_redirect)
- [MoE 14.3 倍加速！UCB提出基于世界模型的GPU Kernel生成方法K-Search](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898073&idx=1&sn=aac8d6be7fbb51f22220baa01f34bb8c&scene=21#wechat_redirect)

> 随着智能体 AI 系统生成与优化**GPU 内核** 的能力不断增强，相关技术进展却受限于现有基准测试的评价逻辑——这类测试仅奖励相对于软件基线的性能加速，而非衡量内核与**硬件高效执行** 的接近程度。

本文提出**SOL-ExecBench**，这是一套包含 235 个**CUDA** 内核优化问题的基准测试集，问题提取自 124 个生产级与前沿 AI 模型，覆盖语言、扩散、视觉、音频、视频及混合架构，专为**NVIDIA Blackwell 系列 GPU** 设计。该基准测试覆盖**BF16、FP8、NVFP4** 下的前向与反向计算任务，包含必须依赖 Blackwell 架构专属特性才能实现最优性能的内核。

::: block-1
与现有基准测试主要相对于软件实现评估内核性能不同，SOL-ExecBench 以**SOLAR**，即本文提出的硬件级光速上限推导流水线，用该流水线解析推导的**光速（Speed-of-Light, SOL）上限** （:mdnice-bold-italic[硬件可实现的理论最小执行时间]）为性能标尺，为硬件高效优化提供固定目标。
:::

本文提出**SOL 分数（SOL Score）**，用于量化候选内核填补官方定义的评分基线与硬件 SOL 上限之间性能差距的程度。为支撑对智能体优化器的稳健评估，本文还提供一套**沙箱化评测框架**，具备**GPU 时钟锁定**、**二级缓存清空**、**隔离式子进程执行**，以及基于静态分析的**奖励作弊** （优化器钻评测规则漏洞虚高得分，而非真正优化内核）检测机制。:mdnice-bold-italic[SOL-ExecBench 将 GPU 内核基准测试的核心逻辑，从「超越易变的软件基线」重构为「缩小与硬件光速性能的剩余差距」。]

## 本文目录

- 一、引言
  - 1.1 现有评估方式的瓶颈与架构多样性的挑战
  - 1.2 基准测试的核心标准与现有方案的不足
  - 1.3 SOL-ExecBench的提出与评估框架
- 二、相关工作
  - 2.1 相关基准测试
  - 2.2 光速性能指标
- 三、基准测试构建
  - 3.1 源模型覆盖
  - 3.2 提取流水线
  - 3.3 问题规格格式
- 四、数据集与评测
  - 4.1 问题特征分析
  - 4.2 SOL 上限推导
  - 4.3 指标：SOL 分数
  - 4.4 评测框架
  - 4.5 评分基线
- 五、实验
  - 5.1 实验设置
  - 5.2 SOL 分数与加速比对比
  - 5.3 奖励作弊缓解
  - 5.4 评分基线结果
- 六、结论
- 参考文献

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/04.webp)

## 一、引言

### 1.1 现有评估方式的瓶颈与架构多样性的挑战

> 随着基于大语言模型代码生成、自主编译、性能分析并迭代优化 GPU 内核的智能体 AI 系统持续进步[5,14,15,25]，现有评估方式已成为制约其发展的瓶颈。:mdnice-bold-italic[现有基准测试往往以相对于软件基线的加速比衡量成功，而内核工程的真实目标是逼近硬件高效执行状态。这种错配的影响正愈发显著：每一代 GPU 都快速推出影响性能的新特性，同时能效成为数据中心部署的核心约束]。在实践中，人工优化已无法长期跟上硬件特性迭代节奏与模型复杂度的增长速度，基于 AI 的内核优化正从可选方案变为必然选择。

AI 模型架构的生态持续扩张。除稠密**Transformer** 外[22]，当前前沿模型还包括：

- 混合专家（Mixture-of-Experts, MoE） 模型（DeepSeek-V3[7]、Qwen3-Coder-480B）
- 状态空间模型（State-Space Models, SSMs）（Mamba-2[6]、Jamba[1]）
- 线性注意力变体（RWKV[17]、Gated Delta Rule[24]）
- SSM-Transformer 混合架构（Nemotron-H[19]）
- 融合视觉、音频、语言的多模态系统（Qwen3-VL、Gemma-3n、Llama-3.2-Vision）

**每种架构都会引入新型计算原语**，例如多头潜在注意力（MLA）、SwiGLU 混合专家调度、面向视频的 3D 旋转位置编码（RoPE）、基于分块的选择性扫描；:mdnice-bold-italic[这些原语需要专用 GPU 内核才能充分发挥硬件性能，IO 感知注意力内核已验证这一点]，如 FlashAttention[6]。单个前沿模型可能包含数十种此类原语，为每个原语开发**光速（SOL）内核** 耗时极长。此外，**内核开发与硬件设计相互促进：理解前沿负载所需的内核可为未来硬件特性提供设计依据，而新硬件能力又会解锁更多内核优化空间**。

> 因此，内核优化基准测试必须**覆盖这类架构多样性，既要保证代表性，也要向硬件设计者传递未来负载趋势。**

### 1.2 基准测试的核心标准与现有方案的不足

负载类型的持续多样化，叠加基于大语言模型代码生成、自主迭代优化 GPU 内核的智能体 AI 系统快速发展[5,14,15,25]，让基准测试的设计难度大幅提升。**一套合格的基准测试必须满足：**

1. 覆盖当前前沿与新兴架构；
2. 包含需利用最新硬件特性与精度格式才能实现最优性能的问题；
3. 包含训练后（微调、RLHF）与推理全流程负载；
4. 以硬件级最大可实现性能为评估目标，而非易变的软件基线；
5. 提供能抵御对抗性优化的评测基础设施。

近期基准测试如 KernelBench[16]、FlashInfer-Bench[23]、BackendBench[20]、TritonBench[12]已朝此方向迈出重要步伐，本文第 2 节将进行详细对比。**但本文发现，尚无单一现有基准测试能同时满足全部标准**。例如：

- KernelBench 包含 250 个 PyTorch 转 CUDA 问题，但:mdnice-bold-italic[其模型级负载源自老旧架构，且仅以相对于 PyTorch 即时执行（eager execution）的加速比为指标，而非硬件极限]。
- FlashInfer-Bench 覆盖 Blackwell 架构上的真实推理负载（含 FP8 混合专家内核），但:mdnice-bold-italic[未包含训练后负载与 NVFP4 等更低精度格式。]

### 1.3 SOL-ExecBench的提出与评估框架

::: block-1
**本文提出 SOL-ExecBench，一套满足全部标准的基准测试**。

本文从 124 个生产级与前沿 AI 模型（覆盖大语言模型、扩散、视觉、音频、视频及混合架构）中，通过大语言模型辅助流水线提取 7400 个计算子图，经筛选得到 235 个基准测试问题，按复杂度与精度分为四个等级。

每个问题均附带规格说明、PyTorch 参考代码，以及最多约 16 个动态形状负载，覆盖 NVIDIA B200 GPU 上 BF16、FP8、NVFP4 精度的前向与反向传播任务。
:::

> 该基准测试与现有方案的核心差异在于**评估目标。SOL-ExecBench 并非仅奖励相对于软件参考的加速比，而是以光速（SOL）上限** ——:mdnice-bold-italic[即目标硬件上解析推导的执行时间下界——为标尺评估内核]。本文研发 SOLAR 流水线，可基于浮点运算量（FLOP）、数据字节量、GPU 峰值算力与带宽，解析推导硬件级 SOL 上限。

本文将该上限与预定义评分基线结合，提出**SOL 分数，用于衡量候选内核填补基线到 SOL 上限之间性能差距的程度**。该指标下，分数 0.5 对应匹配评分基线，1.0 对应达到硬件 SOL 上限。因此，:mdnice-bold-italic[该分数不仅体现相对基线的提升，还反映相对于硬件最大可实现性能的剩余优化空间。]

SOL-ExecBench 还包含一套沙箱化评测框架，保障评分可靠、可复现。为构建更强的评分基线，我们训练一个智能体优化器，在相同评测协议下优化提供的 PyTorch 参考实现。将该优化器在基准测试全量问题上运行，暴露了奖励作弊行为——即优化器钻评测规则漏洞而非生成真正更快的内核；这些观测为框架内置的防御机制提供了依据。

我们通过在全部 235 个问题上运行该智能体优化器验证 SOL-ExecBench 的有效性。智能体生成的基线实现:mdnice-bold-italic[中位数 SOL 分数达 0.732，显著高于 0.5 的中点（分数 0.5 对应匹配评分基线，1.0 对应达到硬件 SOL 上限）]，同时仍保留充足的优化空间。SOL 分数与优化空间回收比例近乎完美相关，而单纯加速比的预测能力更弱。测试过程中，14.5%的智能体提交被判定为奖励作弊，凸显了稳健评测基础设施的重要性。

我们公开 SOL-ExecBench 数据集与评测框架，地址：https://github.com/NVIDIA/SOL-ExecBench。通过将评估锚定到硬件SOL上限而非易变的软件基线，**SOL-ExecBench将GPU内核基准测试的核心重构为缩小与硬件光速性能的剩余差距**。

## 二、相关工作

> 面向智能体的 GPU 内核生成基准测试是一个新兴且快速发展的领域，已有多款基准测试。本文选取代表性方案梳理，表 1 为结构化对比。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/05.webp)

::: image-caption
表 1：GPU内核生成基准测试对比。不难看出，现有基准测试均无法同时满足覆盖前沿模型、支持低精度/训练负载、对标硬件极限、适配Blackwell新硬件四大核心要求；SOL-ExecBench是唯一覆盖BF16/FP8/NVFP4精度、包含反向传播、专为Blackwell硬件设计、以硬件SOL上限为评估指标的基准测试。
:::

### 2.1 相关基准测试

#### KernelBench[16]

> 是大语言模型驱动 CUDA 内核生成领域应用广泛的基准测试，包含 270 个 PyTorch 问题，分四个等级：等级 1（100 个单算子，如矩阵乘、卷积）、等级 2（100 个算子融合模式）、等级 3（50 个完整模型架构）、等级 4（20 个来自 HuggingFace 模型的前瞻性任务）。

KernelBench 提出 $fast_p$ 指标，定义为生成内核中**正确且加速比>p×PyTorch 即时执行基线** 的比例。尽管 KernelBench 流行，:mdnice-bold-italic[但其问题源自已非前沿的模型（等级 3 含 ResNet、BERT、VGG），且方案无需用到最新硬件特性即可超越 PyTorch 基线。]

#### FlashInfer-Bench[23]

> 基于生产级大语言模型推理系统（vLLM、SGLang）的推理原语轨迹构建基准测试，采用 FlashInfer Trace 格式捕获真实算子形状与数据，问题贴合实际部署场景。

MLSys 2026 竞赛赛道（融合混合专家、稀疏注意力、门控增量网络）面向 NVIDIA B200 GPU，支持 CUDA、Triton[21]、CuTe DSL[18]、cuTile[2]等方案。**SOL-ExecBench 纳入 FlashInfer-Bench 的 26 个推理原语，并将覆盖范围扩展至训练负载、量化算子与更广泛的模型类型。**

#### BackendBench[20]

> 针对 PyTorch 后端独立算子（ATen ops）做基准测试，将大语言模型生成的 Triton 内核与 PyTorch 算子信息测试集对比，算子形状源自 HuggingFace 模型。该测试覆盖 271 个算子的正确性、124 个算子的性能，目标是将生成内核直接合入 PyTorch。

- BackendBench :mdnice-bold-italic[聚焦库级算子、采用相对性能指标；]
- 而 SOL-ExecBench 面向完整模型架构提取的应用级子图，包含反向传播与低精度格式（FP8、NVFP4），并以硬件 SOL 上限为评估标尺。

#### TritonBench[12]

> 是首款专门面向 Triton 代码生成的基准测试，包含 184 个源自真实 GitHub 仓库的算子（TritonBench-G）与配套的 PyTorch 对齐集合（TritonBench-T），通过领域特定语言（DSL）指标（内存分块、工作组调度）评估正确性与硬件效率。

- TritonBench :mdnice-bold-italic[聚焦 Triton 单算子生成；]
- **SOL-ExecBench 面向真实模型架构的多算子融合子图**，支持多种 GPU 编程语言（CUDA、Triton、CUTLASS 等）。

#### ComputeEval[19]

> 提供 232 个手工编写的 CUDA 编程任务，覆盖张量核心、CUDA 图、流、线程束原语、共享内存等各类 GPU 编程概念，测试大语言模型的 CUDA 编程能力，核心评估功能正确性（pass@k），适合判断大语言模型能否编写合法 CUDA 代码。

**ComputeEval 与 SOL-ExecBench 互为补充**：

- :mdnice-bold-italic[ComputeEval 侧重 CUDA 编程知识广度]
- SOL-ExecBench 聚焦深度学习负载对标硬件性能极限的深度优化挑战

#### CUDABench[26]

> 面向文本转 CUDA 生成做基准测试，包含 AI、科学计算、数据分析、信号处理、图形学、科学仿真/金融六大领域 500 个任务，分三个难度等级，提出基于**Roofline模型** （硬件性能上限分析模型）的性能分数，与本文 SOL 指标类似，均对标硬件极限而非软件基线。

- :mdnice-bold-italic[CUDABench 面向通用 CUDA 编程；]
- SOL-ExecBench **专注深度学习内核优化，问题提取自生产级模型架构。**

### 2.2 光速性能指标

Roofline模型[23]为本文 SOL 指标提供理论框架，以峰值算力与内存带宽约束内核性能。Orojenesis[9]进一步优化，基于片上缓冲区容量推导张量算法更紧致、可实现的数据移动上限，证明朴素Roofline上限会显著高估数据复用有限的算子的可实现性能。

::: block-1
本文采用的 SOL 上限与 NVIDIA 内部内核团队的开发方法一致——性能以**硬件光速性能百分比** 衡量，而非相对软件基线。:mdnice-bold-italic[本文在 Orojenesis 基础上研发 SOLAR（SOL Analysis for Runtime）流水线，可从 PyTorch 参考实现自动推导紧致、硬件级 SOL 上限]，详见第 4.2 节。
:::

## 三、基准测试构建

> AI 训练后应用（AI post-training applications）对高性能内核的需求飞速增长，驱动本文基于三大原则设计 SOL-ExecBench：

- **应用驱动问题**：问题必须源自前沿与新兴模型架构，保证算子类型、张量形状、数据类型贴合当前生产与未来近景实际负载。
- **覆盖最新硬件特性**：问题与指标目标需鼓励方案用到新 GPU 架构的最新硬件特性，如 Blackwell GPU 第五代张量核心支持的 NVFP4。
- **覆盖训练后全生命周期**：问题需覆盖微调、RLHF、推理服务等全训练后流程，包含低精度（FP8、NVFP4）前向与反向传播，而非仅推理前向内核。

遵循上述原则，本文从 6 大领域 124 个生产级 AI 模型提取问题，面向 NVIDIA B200 GPU，覆盖 BF16、FP8、NVFP4 精度的前向/反向传播，最终得到 235 个基准测试问题，按复杂度与精度分为四类。

### 3.1 源模型覆盖

> 本文从 HuggingFace、Artificial Analysis、arXiv 选取模型，优先覆盖代表当前与未来前沿负载的成熟及新兴架构，总计处理 124 个模型，覆盖六大领域：

1. **大语言模型（61 个）**：含稠密 Transformer（Llama-3.x、Gemma-3、Phi-4）、混合专家模型（DeepSeek-V3/R1、Qwen3-Coder-480B、GLM-4.7）、新型注意力变体（Kimi-K2），引入分组查询注意力、SwiGLU 混合专家调度、多 token 预测等算子。
2. **扩散模型——文本与图像（24 个）**：图像生成模型（Stable Diffusion 系列、FLUX.1/2、混元图像、Qwen-Image-Edit、Step1X-Edit、Bria-3.2、FIBO、Sana、HiDream），贡献自适应层归一化、双流联合注意力、VAE 编解码块；文本扩散模型（如 LLaDA-8B）引入 token 序列并行去噪，计算模式与自回归生成截然不同。
3. **视觉（6 个）、音频/语音（9 个）、视频（2 个）**：视觉含 SAM-HQ、ConvNextV2、VMamba、NAFNet、Swin2SR、MaskGIT；音频含 ASR（Whisper、Parakeet-TDT、Canary）与 TTS/语音模型（Voxtral、OpenVoice、Kokoro、XTTS-v2、Granite-Speech-3.3-8B）；视频含 Wan2.2-T2V。这些领域贡献窗口注意力、Conformer 编码器、3D RoPE 空间注意力等。
4. **多模态与混合架构（22 个）**：含视觉语言模型（Qwen3-VL、Qwen3-Omni、Llama-3.2-Vision、Gemma-3n、Molmo-7B-D、MiMo-V2Flash）、OCR 与文档理解（DeepSeek-OCR）、SSM 及混合架构（Jamba、Nemotron-H、RWKV-v7），融合注意力、状态空间、混合专家原语。

### 3.2 提取流水线

> 提取流水线分四阶段，如图 1 所示。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/06.webp)

::: image-caption
图 1：SOL-ExecBench 构建流水线概览。输入为覆盖六大领域的 124 个源模型；输出为四个类别下共 235 个经过验证的基准测试问题。SOL-ExecBench 通过模型准备→子图提取→筛选采样→验证四步标准化流水线，从海量前沿 AI 模型中高效、规范地提取出高质量内核优化问题，保证问题的真实性与代表性。
:::

1. **模型准备**：加载每个模型的架构定义，提取完整模型源码与隐层维度、注意力头数、数据类型等配置常量。
2. **子图提取**：前沿大语言模型分析每个预处理模型，识别关键计算子图，生成内联所有常量的独立 PyTorch 实现。前向传播按序生成以去重，反向传播并行生成。量化模型通过专用提示引导大语言模型使用对应低精度原语。从 124 个模型中提取 7400 个前向/反向传播子图。
3. **筛选与采样**：从 11 个维度刻画每个子图（算子类型、模型领域、精度、计算强度、前向/反向等），通过分层抽样选取覆盖均衡、多样性强的子集。:mdnice-bold-italic[抽样保持单内核/多内核融合问题的目标比例，为量化算子预留专属槽位]。选中的子图通过大语言模型驱动生成器转为基准测试问题。由于筛选与提取解耦，可针对不同基准目标重采样子图池，无需重新执行提取。
4. **验证**：验证含三部分。未通过校验或易被规格钻漏洞的问题被剔除，最终得到 245 个验证通过问题，公开 235 个，预留 10 个用于后续竞赛。评测框架的作弊检测详见第 4.4 节。

### 3.3 问题规格格式

> 每个问题基于扩展 FlashInfer Trace 格式定义，含三部分：

1. **定义**：指定问题名称、算子类型、带类型符号轴（常量/变量/表达式）、输入/输出张量形状与数据类型、参考实现。
2. **参考**：独立 PyTorch 实现，含顶层`run()`函数；需结构化输入（如稀疏索引、分页 KV 缓存）的问题额外定义`get_inputs()`函数。
3. **负载**：每个问题提供多个（通常约 16 个）动态形状实例的具体轴数值，典型动态轴包括批量大小 $\in\{1,...,64\}$、序列长度 $\in\{128,...,8192\}$。

## 四、数据集与评测

> 基准测试的问题构成可能随时间更新。本文发布时，SOL-ExecBench 含 235 个问题，按复杂度与精度分为四类，见表 2。每个问题附带完整规格、PyTorch 参考实现、优化后基线。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/07.webp)

::: image-caption
表 2：基准测试类别汇总，235个问题按复杂度分为L1（单算子）、L2（多算子融合）、Quant（低精度量化）、FIB（生产推理原语）四类；覆盖从基础算子到复杂融合块、从常规精度到FP8/NVFP4量化的全场景内核优化需求。
:::

### 4.1 问题特征分析

> 图 2 从四个维度刻画 235 个问题：

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/08.webp)

::: image-caption
图 2：从四个维度对 SOL-ExecBench 的 235 个问题进行特征刻画。(a) 基准测试类别分布（阴影部分展示前向 / 反向传播占比）；(b) 按主导算子类型的分布；(c) 按模型领域划分的问题分布（按基准测试类别堆叠）；(d) 按主导计算精度的分布。可以看出：问题以L1+L2 基础算子与融合块为主（75%），前向传播占绝大多数（80%）；算子以注意力、混合专家为核心，贴合大语言模型主流负载；精度以BF16/FP32为主，同时覆盖 FP8/NVFP4 低精度量化场景，匹配现代 AI 训练 / 推理需求。
:::

1. **基准测试类别分布** （阴影标注前向/反向比例）：
2. **主导算子类型分布**：
3. **模型领域分布** （按类别堆叠）：:mdnice-bold-italic[大语言模型 153 个（65%，占比最高），多模态 27 个，扩散 25 个，视觉 13 个]，音频/语音 11 个，视频 6 个。大语言模型覆盖全部四类；扩散、视觉集中在 L1/L2。
4. **主导计算精度分布** （主数据张量类型，非累加缓冲）：:mdnice-bold-italic[BF16 107 个（46%，现代大语言模型与扩散负载主流），FP32 79 个（34%，集中在音频、视觉、扩散），FP8 19 个（8%）、NVFP4 15 个（6%）仅在 Quant 类]，FP16 12 个（5%）多在音频与矩阵乘，混合精度 3 个（1%，整数/布尔主导内核、注意力掩码构造、混合专家 token 路由排序、多模态位置索引计算）。

每个问题含 16 个负载（FIB：7-48 个），覆盖批量大小 $\in\{1,...,64\}$、序列长度 $\in\{128,...,8192\}$ 等动态轴；78 个（33%）问题为分页 KV 缓存、混合专家路由张量、稀疏注意力掩码等结构化输入定制输入生成逻辑。

### 4.2 SOL 上限推导

> 为量化剩余优化空间、验证生成内核的加速比有效性，本文为基准测试每个问题提供**光速（SOL）执行时间**。

该上限通过 SOLAR 工具推导——SOLAR 专为估算 PyTorch 程序在目标硬件上的理论最小执行时间设计，流程如图 3：

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/09.webp)

::: image-caption
图 3：从 PyTorch 模型和输入形状推导光速（SOL）运行时间上限 $T_{SOL}$ 的 SOLAR 流水线。SOLAR 通过图提取→爱因斯坦求和转换→SOL 分析三步，可自动化、解析化地计算出内核在目标硬件上的理论最小执行时间（SOL 上限），为性能评估提供硬件级标尺。
:::

1. **图提取器**：追踪 PyTorch 模型，生成捕获数据流、算子类型、中间张量形状的算子图，基于 torchview 库[11]，利用前向钩子收集张量元信息，兼容 PyTorch 即时执行与动态控制流，无需静态模型图即可捕获精确执行路径。
2. **智能体爱因斯坦求和转换器**：将 PyTorch 算子转为**扩展爱因斯坦求和表达式** [10,15]——:mdnice-bold-italic[爱因斯坦求和的泛化形式，用索引符号表示张量计算，统一张量代数操作，显式暴露张量迭代空间与计算模式，供 SOLAR 做算子分析、推导浮点运算量与数据流量]。SOLAR 维护持久查询表，映射 PyTorch 算子到验证过的求和转换函数；未见过的算子由大语言模型智能体生成候选转换函数，通过仿真求和表达式与原算子结果比对验证，自动纠错后入库。
3. **SOL 分析器**：将转换后的爱因斯坦求和图与硬件规格输入，:mdnice-bold-italic[基于Roofline模型[23]，用目标频率下的峰值算力与内存带宽计算性能：]$T_{SOL}=\max\left(\frac{\text{Total FLOPs}}{\text{Compute Throughput}}, \frac{\text{Total Fused Bytes}}{\text{Memory Bandwidth}}\right)$分析器支持图级融合与预取优化，**还集成 Orojenesis[9]，按片上缓冲区容量建模片外数据移动，推导更紧致上限** ——解决并非所有张量数据均可片上缓存复用的实际问题。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/10.webp)

::: image-caption
图 4：SOLAR 流水线在一个具体的 SOL-ExecBench L1 问题上的应用示例。SOLAR 可精准解析 PyTorch 代码的计算逻辑，量化浮点运算量、内存流量、计算瓶颈，最终输出严谨的 SOL 上限，验证了流水线的实用性与准确性。
:::

> 图 4 以 JambaReasoning-3B 的 L1 问题（融合注意力输出投影+残差加法）为例演示 SOLAR 流程：

1. PyTorch 代码：先矩阵乘，再逐元素加法。
2. 追踪算子图：显式 matmul、add、transpose 节点与张量形状。
3. 扩展爱因斯坦求和图：matmul 转为`ACB,BH→ACH`，加法转为`ACH,ACH→ACH`。
4. B200@1.5GHz 下 SOL 上限：总浮点运算量 107.4G，融合内存 footprint 约 126MB，计算强度 427，计算瓶颈，SOL 执行时间 0.059ms。

::: block-1
**SOLAR 当前局限：**

- :mdnice-bold-italic[【仅】基于张量形状而非数值分析，无法捕获压缩、常量传播等数值依赖优化，可能忽略结构化/重复数据带来的内存访问优化与代数简化收益；]
- 同时，硬件功耗限制、温度节流等波动会导致 SOL 上限在实践中不够紧致。
:::

### 4.3 指标：SOL 分数

> 本文定义新性能指标**SOL 分数**，记为 $S\in[0,1]$，衡量内核相对固定基线接近硬件 SOL 的程度。

设 $T_b$ 为基线执行时间，$T_{SOL}$ 为 SOLAR 估算的 SOL 执行时间，$T_k$ 为候选内核实测执行时间；假设 $T_b>T_{SOL}$ 且 $T_k≥T_{SOL}$，保证基线到 SOL 的差距为正；若假设不成立，视为审计信号，用于复核 SOLAR 上限与检测奖励作弊（4.4.1 节）。

SOL 分数定义：

$$
S(T_k)=\frac{1}{1+\frac{T_k-T_{SOL}}{T_b-T_{SOL}}}
$$

等价形式：

$$
S(T_k)=\frac{T_b-T_{SOL}}{(T_k-T_{SOL})+(T_b-T_{SOL})}
$$

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/11.webp)

::: image-caption
图 5：SOL分数作为内核运行时间$T_k$的函数曲线，示例中$T_{SOL}=50$、$T_{b}=100$。当$T_k=T_{SOL}$时指标锚定在$S=1$，当$T_k=T_{b}$时锚定在$S=0.5$，随着运行时间增加分数平滑趋近于0。曲线同时体现了指标的非线性特性：相同的运行时间提升，在越接近SOL区间时带来的分数增益越大。因此，可以看出 SOL分数非线性、有界（0~1），基线对应0.5、硬件极限对应1；越接近硬件性能极限，分数提升越显著，精准鼓励逼近硬件极限的深度优化
:::

> SOL 分数满足三个锚定特性（图 5）：

- $T_k=T_b \Rightarrow S=0.5$
- $T_k=T_{SOL} \Rightarrow S=1$
- $T_k\to\infty \Rightarrow S\to0$

该指标将基线设为中点 0.5 而非 0，在有界尺度上区分三类性能区间：

- 低于基线（$S<0.5$），低于基线区间随执行时间增加平滑趋近 0
- 高于基线但未达 SOL（$0.5 < S < 1$），分数>0.5 表示内核优于基线，$T_b-T_{SOL}$ 为基线到硬件 SOL 的性能空间，SOL 分数衡量候选内核填补该差距的效率
- 达到 SOL（$S=1$）。

接近 1 表示逼近硬件高效执行。**若 $T_b\to T_{SOL}$，视为问题已解决，不再接收新提交**。本文尝试过 clip、sigmoid 等变体，最终选择当前形式以保证简洁性。

为仅给功能正确的内核计分，本文为每个问题设正确性指标 $C\in\{0,1\}$：验证失败的内核 $C=0$，无论执行时间均无性能分。

基准测试集含 N 个问题时，整体 SOL 分数为单问题分数的算术平均：

$$
\overline{S}=\frac{1}{N}\sum_{j=1}^{N}C_jS_j
$$

::: block-1
采用算术平均的原因：单问题 SOL 分数已绑定[0,1]，跨问题含义一致，平均后可保留该含义。该形式可自然扩展到智能体系统单问题生成多个候选的 best-of-k 场景，本文暂不展开。
:::

### 4.4 评测框架

> 提交方案为 JSON 规格，含源码文件、实现语言、编译配置、目标硬件、入口函数；当前评测器支持 Python、Triton、CUDA/C++（通过`torch.utils.cpp_extension`），兼容 PTX、CUTLASS、CuTe DSL、cuBLAS、cuDNN、cuTile 等实现。

每个问题附带 PyTorch 参考实现，定义目标语义并支持正确性校验；参考实现优先保证可移植性、可读性、功能覆盖，不追求高性能，应视为功能规格而非强软件性能基线。因此本文区分参考实现与执行基线：计算 SOL 分数需为每个问题定义**评分基线 $T_b$，** 作为指标中点锚点（4.3 节）。

:mdnice-bold-italic[当前评分基线暂不公开，未来版本可能发布；评分基线不绑定特定实现，可随更强方案出现而更新，保证基准测试随技术演进保持挑战性]。当前评分基线的获取方式见 4.5 节。

**正确性校验流程**：评测器先运行固定参考生成参考输出，再在多组随机种子试验中比对候选输出与参考。校验覆盖输出形状、数据类型、张量基本合法性，拒绝非法 inf/NaN 值、非平凡参考下的全零输出。稠密张量输出的正确性由负载专属容差元组（绝对误差`atol`、相对误差`rtol`、匹配比例`matched_ratio`）定义，存储于`workload.jsonl`；BF16/FP32 问题的容差经离线多次随机输入校验校准，施加 1.25 倍安全裕度。量化与采样问题用专用评测器：量化内核与 cuBLAS 参考（通过 PyTorch）比对，无 cuBLAS 参考时与 FP32 参考比对。

**执行时间测量**：用 CUDA 事件，每组试验 10 次热身、50 次计时，共 3 组试验，报告均值。每次计时前清空 256MB 设备缓冲以清空二级缓存，克隆张量参数保证每次运行用新地址、不复用历史状态。:mdnice-bold-italic[单 GPU 基准测试串行执行，通过`nvidia-smi`锁定时钟为硬件专属频率（B200 为 1500MHz），提升可复现性]。框架报告绝对执行时间（毫秒），加速比、SOL 分数等相对指标后续按版本评分基线计算。

::: block-1
- 提交方案在专属子进程编译执行，隔离评测器状态，避免方案间相互干扰；
- 该设计支持多 GPU 轮询调度，失败工作单元可丢弃重启，不中断基准测试。300 秒超时防止卡死与死循环；
- 参考输出与计时数据单独准备，通过进程间通信传递给方案工作单元。
:::

#### 4.4.1 奖励作弊与防御

> 已有研究发现，智能体优化系统易出现**奖励作弊** [13]——优化器利用评测环境漏洞最大化得分，而非正确解决核心任务。

在 GPU 内核优化场景，本文观察到智能体生成代码绕过计时机制、违反基准约束、利用计时循环重复性伪造低执行时间。**本文将这些漏洞利用分为三类：并发漏洞、状态缓存、环境操纵，见表 3。**

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/12.webp)

::: image-caption
表 3：总结了智能体式内核优化器在优化过程中出现的三类奖励破解策略及其对应的防御机制。具体而言，并发利用通过隐藏计算于未记录的线程或CUDA流中规避计时；状态缓存依赖重复计时循环中的输出复用或惰性求值来跳过实际计算；环境操纵则通过篡改计时函数或降低计算精度来虚报性能提升。针对这些行为，评估框架结合了动态运行时检查（如线程数监控、输入克隆、内存地址验证）与基于LLM的静态分析，有效识别并拦截了相关攻击。随着智能体系统在GPU内核优化中自主性增强，其绕过评估机制、追求表面分数的“投机”行为也日趋多样和隐蔽。这不仅凸显了构建具备多层次防御能力的沙箱化评估环境的重要性，也表明单纯依赖传统性能指标（如加速比）已不足以可靠衡量优化质量，必须辅以严格的正确性验证与抗攻击机制，才能确保评估结果的真实性与可信度。
:::

- 并发漏洞：绕过`torch.cuda.Event`计时器隐藏执行时间，如后台线程、非默认流、`torch.jit.fork`；复杂变体滥用 CUDA 图捕获隐式流，CuTe 内核未获取该流导致捕获时计算、重放时空跑。
- 状态缓存：正确性校验缓存结果/中间变量，计时阶段直接拷贝；惰性求值用`FakeTensor`跳过计时计算。
- 环境操纵：猴子补丁篡改计时函数；精度降级（如 FP32 问题用 FP16 计算后转型）；非 PyTorch 提交嵌入预编译机器码绕过源码审查。

> 为保障 SOL 分数完整性，本文构建**评测沙箱** 抵御智能体优化器常见作弊：检测计时路径篡改、侧流隐藏计算、运行间输入克隆减少状态泄露、严格输出校验+子进程隔离，确保无效/对抗方案无性能分。

- 框架执行严格动态检查：监控活跃线程数、断言输出为完整`torch.Tensor`（拒绝子类）、插入`torch.cuda.synchronize()`捕获异步隐藏计算、校验关键计时函数内存地址防猴子补丁。
- 缓解状态缓存：多组随机输入正确性试验、每次计时前清空 256MB GPU 二级缓存；
- 防止用内存地址作为缓存键：预分配缓冲内每轮偏移 256B 分配新地址。复杂隐蔽模式（动态流、语义缓存、未授权文件 I/O）用大语言模型法官做提交前静态代码分析。
- 此外，**拟作为新评分基线的候选方案均需人工审核。**

> 为减少作弊，当前评测器做两项保守设计：禁用 CUDA 流、依赖 PyTorch 默认内存分配器。

- 禁用流可防止隐藏计算漏洞，但可能导致用户内核无法完全复现`torch.compile`评分基线；该限制对高算力大语言模型内核可接受，因其多流收益有限。
- 依赖 PyTorch 即时分配器提升实用性与可读性，但可能对显存水位>50%的子图中非 PyTorch 内核不利。

::: block-1
未来版本可通过流检测、分配器控制（`PYTORCH_NO_CUDA_MEMORY_CACHING`）、定制 PyTorch 构建、非 PyTorch 参考实现+静态分配器优化，各方案均有兼容性、可读性、维护成本的权衡。
:::

### 4.5 评分基线

> 每个问题的**评分基线** （暂不公开）独立于参考程序，是高性能实现，作为 SOL 分数公式中的执行时间锚点 $T_b$。快于 $T_b$ 的方案得分>0.5，代表优于已优化内核。

基线实现由**智能体内核优化系统** 生成：优化器采用回合制多智能体模式，每个问题启动多个独立智能体，在固定时间/成本预算下优化 PyTorch 参考实现。每轮收集所有有效提交，暴露给下一批智能体，可基于任意正确候选继续优化。

智能体仅限用 PyTorch 与标准 Python 包，配备工具提交方案到 4.4 节沙箱做正确性校验与基准测试；所有提交经大语言模型法官检测违规与作弊。仅编译成功、通过正确性校验、通过作弊防御的方案保留为基线候选。多轮结束后，聚合所有有效候选，为每个问题选最快内核作为最终优化基线。

## 五、实验

### 5.1 实验设置

所有实验在 NVIDIA DGX B200 节点执行，配 8×NVIDIA Blackwell B200 GPU，单卡 192GB HBM3e 内存、8TB/s 内存带宽[18,19]。

- 软件栈基于 NVIDIA 官方 Docker 镜像：CUDA 13.1.1、cuDNN 9.17.1、PyTorch 2.9.0、NVIDIA 驱动 580.95。
- 每次基准测试用单 GPU，SM 时钟通过`nvidia-smi`锁定 1500MHz，降低频率波动噪声，与 4.4 节评测框架一致。

### 5.2 SOL 分数与加速比对比

> 内核优化的传统评估方式是测相对 PyTorch 参考的加速比。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/13.webp)

::: image-caption
图 6：中展示了所有问题和负载下，相对于PyTorch参考实现的加速比（横轴）与距离硬件SOL的程度（纵轴）。进而可以发现加速比与硬件SOL距离几乎无相关性，仅用加速比评估内核优化会严重失真，无法反映真实的硬件优化空间。
:::

本文先以该指标评估智能体生成方案：图 6 以该加速比（$T_{ref}/T_k$，横轴）对比方案相对 SOL 上限的剩余距离（$T_k/T_{SOL}$，纵轴），覆盖所有负载。:mdnice-bold-italic[结果显示二者无相关性（对数尺度 $r=0.10$）——即方案可比 PyTorch 快 10 倍，却仍距硬件 SOL 超 10 倍。仅用加速比会高估这类内核，掩盖到达 SOL 前的大量优化空间]。部分负载落在横轴 1.0 以下，代表相对参考实现变慢。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/14.webp)

::: image-caption
图7：SOL分数分布全景图。每个点按SOL分数区间着色，并叠加$S=0.5、0.7、0.9$等分数轮廓线。坐标轴与图6一致。可以得出一些结论：1. SOL分数整合了加速比与硬件极限距离，是比单纯加速比更科学的评估指标；2. 仅靠高加速比无法获得高分，只有同时接近硬件SOL才能拿到高SOL分数。
:::

图 7 以相同坐标轴，按 SOL 分数 $S$ 着色，叠加等分数轮廓线（$S=0.5、0.7、0.9$）：高 $S$ 点（蓝色，$S≥0.9$）集中在右下角（高加速比、近 SOL），低 $S$ 点（红色，$S<0.4$）在左上角；右上象限（快于 PyTorch 但远于 SOL）得分中等（$S≈0.5-0.7$）。等分数线证明 $S$ 并非加速比的简单重映射：相同加速比因 SOL 距离不同，对应 SOL 分数差异极大。这验证 $S$ 将优化的两个维度整合为单一有界值，是加速比、SOL 接近度单独无法实现的。

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/15.webp)

::: image-caption
图 8：SOL分数与加速比对比验证（优化空间回收比例）。两个子图共享相同的横轴。(a) SOL分数与优化空间回收比例的关系（按类别着色）；(b) 加速比与优化空间回收比例的关系（按SOL分数着色）。
:::

> 为验证 SOL 分数忠实反映优化质量，本文将其与**优化空间回收比例** （$(T_{ref}-T_k)/(T_{ref}-T_{SOL})$，衡量填补总优化差距的程度）对比：

- **图 8(a)直接绘制 SOL 分数与该比例，二者近乎完美相关（皮尔逊 $r=0.981$），证明 $S$ 忠实追踪该指标**；曲线在 $y=x$ 对角线上方，因方案匹配/超越参考时 $S≥0.5$。
- **图 8(b) 展示了加速比（纵轴）随优化空间回收比例（横轴）的变化趋势**，颜色依据 SOL 分数 $S$ 由红（$S \approx 0$）经黄渐变至深绿（$S \approx 1$）。图中可见显著的水平扩散现象：:mdnice-bold-italic[当加速比固定为 3× 时，对应的空间回收比例跨度极大]，从不足 0.2 到超过 0.8，具体取值高度依赖于初始解距硬件 SOL 的远近。:mdnice-bold-italic[这一现象明确表明——加速比本身无法可靠反映优化质量]。

更关键的是，图 8(b) 颜色梯度与回收比例高度一致：深绿色区域（高 $S$）几乎全部集中在高回收比例一侧，且该对应关系不受加速比高低影响，:mdnice-bold-italic[体现出 SOL 分数对优化效果更强的判别力。量化分析进一步佐证：加速比与回收比例的皮尔逊相关系数为 $r = 0.81$，而 SOL 分数 $S$ 与回收比例的相关系数高达 $r = 0.98$。]

::: block-1
这一显著差异凸显了 SOL 分数的核心价值：它将硬件可实现的性能上限（即 SOL）显式纳入评估体系，从而弥补了仅依赖加速比所导致的评估盲区。$S$ 不仅反映相对提升，更刻画了逼近理论极限的程度，使优化效果评估更本质、更鲁棒。
:::

### 5.3 奖励作弊缓解

> 如前所述，智能体解题过程中出现奖励作弊。图 9 展示所有智能体提交中检测到的漏洞利用分布，按类型与问题类别拆分：

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/16.webp)

::: image-caption
图 9：智能体提交中检测到的奖励作弊漏洞分布（按类型与问题类别）。可以发现智能体内核优化奖励作弊现象普遍（14.5%提交作弊），核心作弊类型为精度降级、猴子补丁、流注入；再就是验证了SOL-ExecBench沙箱评测框架作弊检测与防御机制的必要性。
:::

- 精度降级最常见（259 个内核，占总提交 6.4%）——智能体用低精度计算（如 FP16 代 FP32）后向上转型通过校验；
- 猴子补丁（134，3.3%）重写计时函数伪造低执行时间；
- 流注入（100，2.5%）在未记录流隐藏计算；
- 缓存输出复用（67，1.6%）校验时缓存结果、计时重放；
- JIT 分叉、一次性校验、线程注入较少见。

总计 589 份提交（14.5%）被 4.4.1 节的动态运行时检查+大语言模型静态分析判定并拒绝。**结果凸显智能体内核优化中稳健评测基础设施的必要性。**

### 5.4 评分基线结果

> 图 10 刻画四类基准测试的智能体生成方案：

![文章插图](/articles/2026-03-21-nvidia-sol-exebench-gpu-kernel-d6977abd/images/17.webp)

::: image-caption
图 10(a)：智能体解决方案（新评分基线）结果分析，各类别的SOL分数分布：直方图（上方）标注中位数（实线）与均值（虚线）；箱线图（下方）。$S=0.5$为与PyTorch参考持平线。可以看出智能体生成的基线中位数SOL分数达0.732，显著优于PyTorch参考；所有类别均未达到SOL上限（S=1），仍有充足优化空间，保证了基准测试的长期挑战性。
:::

1. 每类 SOL 分数分布：L1 中位数 0.688、L2 0.761、Quant 0.757、FIB 0.789，整体中位数 0.732。各类中位数均高于 0.5 中点，**证明智能体方案持续优于 PyTorch 参考**；同时分数极少达 1.0，各类均保留优化空间。L1 分布最宽，反映单算子内核类型多样；FIB 分数最高，**反映 FlashInfer 生产套件的聚焦优化。**
2. PyTorch 参考与智能体方案相对 SOL 上限的距离对比：几乎所有负载落在对角线下方，证明方案更接近 SOL。L1 中位数 SOL 距离缩小 2.0×、L2 2.7×、Quant 2.9×、FIB 3.4×。**智能体方案代表当前优化前沿**，优于参考的方案将作为未来 SOL 分数公式的评分基线 $T_b$，基线可随更强方案出现而更新。

## 六、结论

> 本文提出 SOL-ExecBench，一套围绕**硬件光速（SOL）性能目标** 构建的 GPU 内核优化基准测试。:mdnice-bold-italic[SOL-ExecBench 包含 235 个问题，源自 124 个前沿与新兴 AI 模型，覆盖训练后与推理负载、现代精度格式、需新硬件特性的内核。]

- 提出 **SOLAR 流水线，从 PyTorch 程序解析推导硬件级 SOL 上限**，为每个问题提供超越软件基线的稳定目标。
- 提出**SOL 分数**，一种全新的生成内核评估指标：与单纯加速比不同，该指标通过衡量填补评分基线到 SOL 上限的差距，揭示剩余优化空间。
- 提供稳健评测框架，基于智能体生成方案的失效模式设计，防御奖励作弊。

本文智能体优化器生成强基线方案，整体中位数 SOL 分数 0.732，证明仍存大量优化空间。:mdnice-bold-italic[未来，SOL-ExecBench 将随新负载、更强基线、社区提交持续演进，保持与模型、内核、硬件前沿对齐。]

## 参考文献

1. AI21 Labs. 2024. Jamba: A Hybrid Transformer-Mamba Language Model. arXiv:2403.19887
2. Jonathan Bentz. 2025. Python. NVIDIA Technical Blog.
3. Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374
4. Terry Chen, Bing Xu, and Kirthi Devleker. 2025. Automating GPU Kernel Generation with DeepSeek-R1 and Inference-Time Scaling. NVIDIA Technical Blog.
5. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems (NeurIPS).
6. Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State-Space Duality. arXiv:2405.21060
7. DeepSeek-AI. 2024. DeepSeek-V3 Technical Report. arXiv:2412.19437
8. Albert Einstein. 1922. The general theory of relativity. In The meaning of relativity. Springer, 54–75.
9. Qijing Huang, Po-An Tsai, Joel S. Emer, and Angshuk Parashar. 2024. Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms. In Proceedings of the 51st Annual International Symposium on Computer Architecture (ISCA). Best Paper Nominee.
10. Fredrik Kjolstad, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. 2017. The tensor algebra compiler. Proceedings of the ACM on Programming Languages 1, OOPSLA (2017), 1–29.
11. Mert Kurttutan. 2022. torchview: Visualize PyTorch Models.
12. Jianling Li, Shangzhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, Haojie Wang, Jianrong Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators. In Findings of the Association for Computational Linguistics (ACL).
13. Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023. StarCoder: May the Source Be with You!. In Transactions on Machine Learning Research.
14. Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. 2024. StarCoder 2 and The Stack v2: The Next Generation. arXiv:2402.19173
15. NVIDIA. 2024a. CUTLASS: CUDA Templates for Linear Algebra Subroutines.
16. NVIDIA. 2024b. NVIDIA B200 GPU Data Sheet.
17. NVIDIA. 2024c. NVIDIA Blackwell Architecture Technical Brief.
18. NVIDIA. 2025a. ComputeEval: Evaluating LLM Capabilities for CUDA Programming.
19. NVIDIA. 2025b. Nemotron-H: A Family of Hybrid Transformer-SSM Models.
20. Toluwanimi O. Odemuyiwa, Joel S. Emer, and John D. Owens. 2024. The EDGE Language: Extended General Einsums for Graph Algorithms. arXiv:2404.11591
21. Robert Tjarko Lange, Qi Sun, Aaditya Prasad, Maxence Faldor, Yujin Tang, and David Ha. 2025. Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization. arXiv:2509.14279
22. Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini. 2025. KernelBench: Can LLMs Write Efficient GPU Kernels?. In Proceedings of the 42nd International Conference on Machine Learning (ICML). Also Best Paper at DL4C and SSI-FM workshops at ICLR 2025.
23. Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. 2023. RWKV: Reinvented RNNs for the Transformer Era. arXiv:2305.13048
24. Mark Saroufim, Sahan Paliskara, Jiannan Wang, Bert Maher, Manuel Candales, Shahin Sefati, and Laura Wang. 2025. BackendBench: Benchmarking LLM-Generated Triton Kernels for PyTorch Backend Operators.
25. Philippe Tillet, Hsiang-Tsung Kung, and David D. Cox. 2019. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. 10–19.
26. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems (NeurIPS). 5998–6008.
27. Samuel Williams, Andrew Waterman, and David Patterson. 2009. Roofline: An Insightful Visual Performance Model for Multicore Architectures. Commun. ACM 52, 4 (2009), 65–76.
28. Shanli Xing, Yiyan Zhai, Alexander Jiang, Yixin Dong, Yong Wu, Zihao Ye, Charlie Ruan, Yingyi Huang, Yineng Zhang, Liangsheng Yin, Aksara Bayyapu, Luis Ceze, and Tianqi Chen. 2026. FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems. arXiv:2601.00227
29. Bing Xu, Terry Chen, Fengzhe Zhou, Tianqi Chen, Yangqing Jia, Vinod Grover, Haicheng Wu, Wei Liu, Craig Wittenbrink, Wen-mei Hwu, et al. 2026. VibeTensor: System Software for Deep Learning, Fully Generated by AI Agents. arXiv preprint arXiv:2601.16238
30. Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2024. Gated Delta Networks: Improving Mamba2 with Delta Rule. arXiv:2412.06464
31. Jiace Zhu, Wentao Chen, Qi Fan, Zhixing Ren, Junying Wu, Xing Zhe Chai, Chotiwit Rungrueangwutthinon, Yehan Ma, and An Zou. 2026. CUDABench: Benchmarking LLMs for Text-to-CUDA Generation. arXiv:2603.02236

相关推荐

- [NVIDIA, UCB提出Kernel Blaster：记忆增强上下文 RL 让 CUDA 内核几何平均1.43×-2.50×加速！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897594&idx=1&sn=7b2a26eda203fb62c3c7f6aa75e8d046&scene=21#wechat_redirect)
- [最高 4.7 倍加速！UCSD、Meta 提出 AutoOverlap：以块级细粒度重叠融合计算与通信，突破多 GPU 通信瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897745&idx=1&sn=c925fdf089e3022438cce891ccbc440b&scene=21#wechat_redirect)
- [MoE 14.3 倍加速！UCB提出基于世界模型的GPU Kernel生成方法K-Search](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898073&idx=1&sn=aac8d6be7fbb51f22220baa01f34bb8c&scene=21#wechat_redirect)
