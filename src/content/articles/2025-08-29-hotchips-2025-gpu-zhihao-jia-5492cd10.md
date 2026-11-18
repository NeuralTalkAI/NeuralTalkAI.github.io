关键词：GPU 机器学习优化、并行化（含数据/模型/流水线/张量并行）、内核融合（含巨型内核）、FlexFlow、Mirage Persistent Kernel（MPK）、Scaling Law

> “硬件的摩尔定律会放缓，但系统的摩尔定律才刚刚开始。”

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/01.webp)

::: image-caption
https://catalyst.cs.cmu.edu/
:::

- Programming Techniques for Optimizing ML on GPUs
- Zhihao Jia, CMU，https://catalyst.cs.cmu.edu/
- https://github.com/mirage-project/mirage/
- 本文 4636 字，阅读需 15 分钟，播客 9 分钟如下

相关推荐

- [超大规模训练 1.23 倍加速！WLB-LLM 工作负载均衡 4D 并行解决长上下文训练瓶颈](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890434&idx=1&sn=46c7ec1fd6bccb03606a125b19a89611&scene=21#wechat_redirect)
- [MoE 所有层融到一个分布式算子GPU Kernel！FlashDMoE：GPU内核-硬件协同解锁大规模分布式机器学习性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888883&idx=1&sn=14a76e02fc523d30f613d4a2219c0fea&scene=21#wechat_redirect)
- [HotChips 2025：从 PyTorch 到 CUDA：大模型时代 GPU 内核自动生成的技术跃迁，Tri Dao最新演讲](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891347&idx=1&sn=173b8638e00da3610dd90ba7d5a4b7f6&scene=21#wechat_redirect)

::: block-1
Catalyst 是卡内基梅隆大学（CMU）的一个跨学科机器学习与系统研究小组，由该校机器学习系、计算机科学系和电子与计算机工程系的研究人员合作组建，致力于探索自动化学习系统相关问题，其研究覆盖机器学习和系统栈的多个层面。

研究项目（或领域）主要有：XGrammar、TidalDecode、机器学习编译、FlexFlow Serve、Cortex、Apache TVM Stack、Hyperband 和 ASHA、FlexFlow、TASO 和 XGBoost。
:::

> 在 AI 算力需求呈指数级爆发、硬件架构持续迭代的当下，如何让机器学习系统在 GPU 上实现高效部署，成为突破技术瓶颈的关键命题。

来自卡内基梅隆大学计算机科学系的 Zhihao Jia，以《Programming Techniques for Optimizing ML on GPUs》为主题展开演讲，**深度聚焦 GPU 机器学习系统优化的核心逻辑**。

::: block-1
他从 “每美元每秒算力”（Compute Per Second Per Dollar）这一核心驱动力切入，结合 Scaling Law 揭示模型效果与算力的关联，继而剖析现代 GPU“大规模并行、高度异构、快速演进” 的硬件特征。

随后系统提出并行化（跨设备优化）与内核融合（单设备优化）两大核心技术路径，并通过 FlexFlow 自动并行方案、Mirage Persistent Kernel（MPK）巨型内核工程落地等实践案例，为解决 “硬件 2-3 年迭代、软件 5-7 年成熟” 的行业痛点提供了切实可行的方案。
:::

这场演讲不仅梳理了 GPU 机器学习优化的十年技术跃迁脉络，更指明了自动化编译与运行时优化的未来方向，为从事 AI 系统研发与落地的研究者、工程师提供了极具价值的技术参考。

## 本文目录

- 一、问题的起点：为什么 AI 算力需求永无止境
- 二、硬件的现实：并行、异构、日新月异
  - 存储层级
  - 计算单元
  - 系统互联
  - 顺便给自家实验室打广告
- 三、两大抓手：并行化与内核融合
  - 1. 并行化：跨设备的“外科手术”
  - 2. 内核融合：单卡内部的“微雕艺术”
- 四、自动并行：从 FlexFlow 到 TopoOpt
- 五、巨型内核的工程落地：Mirage Persistent Kernel（MPK）
  - 1. 编程模型：Task Graph（任务图）
  - 2. 编译器：从 Layer 到 Task 的映射
  - 3. 运行时：Event-Driven Execution（事件驱动执行）
- 六、未竟之路：写在摩尔定律之后的思考
- 结语

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/02.webp)

## 一、问题的起点：为什么 AI 算力需求永无止境

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/03.webp)

::: image-caption
机器学习成功的根本驱动力。核心问题：机器学习成功的根本驱动力是什么？
:::

- 1950 年，一台占据整个房间的 UNIVAC 每秒只能做几千次浮点运算；
- 2023 年，一台零售价 1299 美元的 Apple 笔记本电脑在一小时内完成的计算量，已超越 UNIVAC 90 年生命周期的总和。

这种算力爆发并非偶然，Ray Kurzweil 在《The Singularity Is Near: When Humans Transcend Biology》（2005）中提出的“加速回报定律”，恰是对这一现象的精准概括。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/04.webp)

::: image-caption
聚焦 “每秒每美元计算量（Compute Per Second Per Dollar）”，并展示计算能力的指数增长趋势（1900-2045）。引用 Ray Kurzweil 的预测：计算能力将在 2023 年超越人类脑力，暗示技术奇点可能来临。强调硬件并行化和专用化是计算能力提升的关键。
:::

在跨越 1900-2045 年的指数曲线图里，人类脑力的总和被浓缩成一条轻描淡写的虚线，而真实世界的算力增长曲线却以陡峭的斜率向上攀升，早已将前者远远甩在身后。图中清晰标注“Surpass human brainpower in 2023”，意味着在 2023 年，人类创造的算力已实现对自身脑力的超越，这一节点成为 AI 发展的重要里程碑。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/05.webp)

::: image-caption
机器学习的缩放定律，模型精度和能力的提升依赖于训练 / 推理计算量的增加。对比硬件性能增长：GPU 计算性能每年翻倍（10 年提升 1000 倍），远超 CPU（单线程性能年增 1.1-1.5 倍）。硬件并行化和专用化是计算能力提升的关键。
:::

对机器学习而言，这条算力曲线不仅是历史注脚，更是生存法则。Transformer 模型的参数规模每两年就会翻 240 倍，如此迅猛的增长使得算力需求成为制约模型发展的关键瓶颈——训练 GPT-4 的成本若用 2020 年的 V100 集群计算，需要近 30 年。

正是在这样的背景下，“每一美元能买到的算力”（Compute per Second per Dollar）成为驱动 ML 发展的核心指标，Zhihao 将其明确量化，并由此推演出机器学习领域的核心规律：Scaling Law（缩放定律），即模型效果与参数、数据、算力三者呈正相关关系，模型准确性与能力的提升，依赖于训练和推理算力的同步缩放。

## 二、硬件的现实：并行、异构、日新月异

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/06.webp)

::: image-caption
机器学习硬件的特性：大规模并行与异构性。
:::

> Zhihao 用两幅详细的硬件架构图，将现代 GPU 拆解成了清晰的“解剖图”，直观呈现出其复杂的内部结构：

### 存储层级

**从存储层级来看**，GPU 拥有从寄存器（Register Files）、共享内存（Shared Memory）到高带宽内存（HBM）、设备内存（Device Memory）的多级存储体系，不同层级的延迟差异可达 3 个数量级。这种多级存储设计的目的，是通过将高频访问的数据放在低延迟存储中，减少数据传输耗时，提升整体计算效率。

### 计算单元

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/07.webp)

::: image-caption
机器学习硬件特性在快速演进：对比 NVIDIA A100（2020）和 H100（2022）GPU，H100 新增 “处理集群” 和 “张量内存加速器”，硬件并行管理和内存访问专用化升级。
:::

从计算单元来看，GPU 包含多种功能各异的处理单元：负责通用计算的 CUDA Core、专门优化张量运算的 Tensor Core，以及最新一代 GPU（如 NVIDIA H100）中新增的 Tensor Memory Accelerator（张量内存加速器）。

这些单元各司其职，CUDA Core 处理基础算术逻辑运算，Tensor Core 加速矩阵乘法等 AI 核心运算，Tensor Memory Accelerator 则针对性优化内存访问效率，三者协同实现高效算力输出。

### 系统互联

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/08.webp)

::: image-caption
硬件随着机器学习快速演进下：新的硬件层、特化的加速器实现、集群和TMA
:::

在系统互联层面，一台 DGX 节点通常集成 8 张 GPU，这些 GPU 通过 NVLink 或 NVSwitch 实现高速互联，确保节点内部数据传输效率；跨节点场景下，则通过 InfiniBand 网络构建大规模集群，满足分布式训练对跨节点通信的需求。这种多层级互联架构，是实现大规模并行计算的基础。

### 顺便给自家实验室打广告

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/09.webp)

::: image-caption
现代 ML 硬件的特征精准概括：大规模并行、高度异构、快速演进
:::

CMU 的 Catalyst 实验室将现代 ML 硬件的特征精准概括为“Massively Parallel, Highly Heterogeneous, Rapidly Evolving”（大规模并行、高度异构、快速演进）。但残酷的现实是：硬件更新周期仅为 2-3 年，而编译器、框架等软件工具的成熟往往需要 5-7 年，这种“硬件跑在软件前面”的错位，给 ML 系统优化带来了巨大挑战。

## 三、两大抓手：并行化与内核融合

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/10.webp)

::: image-caption
GPU 上优化 ML 的关键挑战：大规模并行、日益异构和快速演进。如何最优并行化、处理异构性、应对硬件快速迭代？
:::

> 面对硬件并行、异构、快速演进的现实，Zhihao 给出了系统级的优化方案：

- 向外拓展，通过并行化（Parallelization）优化跨设备性能；
- 向内深耕，通过内核融合（Kernel Fusion）优化单设备内部性能。

二者共同构成了 GPU 上 ML 优化的核心抓手。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/11.webp)

::: image-caption
ML 训练 / 推理的两大关键技术：并行化、核融合。
:::

从并行化说起。

### 1. 并行化：跨设备的“外科手术”

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/12.webp)

::: image-caption
并行化：优化设备间性能，包括数据并行、模型并行、流水线并行等多种策略。
:::

> 并行化的核心思路是将大规模计算任务拆解，分配到多台设备上协同执行，Zhihao 中详细介绍了多种并行策略及其适用场景。

#### 数据并行（Data Parallelism）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/13.webp)

::: image-caption
数据并行原理：1. 分割数据集为批次；2. 各 GPU 独立执行前向 / 反向传播；3. 聚合梯度。
:::

将规模为 256K 的训练批次（batch）均匀切分为 256 份，每张 GPU 负责处理 1K 数据的前向和反向传播，最后通过梯度聚合（Aggregation）同步各设备的计算结果。这种策略实现简单，适用于数据量庞大但模型结构不复杂的场景，通过流程图清晰展示了其“拆分-计算-聚合”的流程。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/14.webp)

::: image-caption
Transformer 示例：展示多 GPU 上权重共享、MatMul 和 AllReduce 操作的协同。
:::

#### 模型并行（Model Parallelism）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/15.webp)

::: image-caption
模型并行：将模型分割为子图，分配给不同设备，传输中间结果。
:::

针对参数规模庞大的模型（如 1760 亿参数的 BLOOM），将模型纵向拆分为多个子图（如 80 层），每个子图分配到一张 GPU 上，设备间通过传输中间结果实现协同。这种方式解决了单卡无法容纳大模型的问题，架构图直观呈现了模型拆分与设备分配的关系。

#### 流水线并行（Pipeline Parallelism）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/16.webp)

::: image-caption
流水线模型并行（Transformer 示例）：将模型层分配给不同 GPU，通过流水线 overlap 计算与通信，提升效率。
:::

在模型并行的基础上，进一步将拆分后的模型层横向划分为多个阶段（stage），例如将 80 层拆分为 4 个 stage，采用“1 前向（Forward）+1 反向（Backward）”（1F1B）的调度方式，实现各阶段计算的流水线化，减少设备空闲时间。上图以 Transformer 模型为例，展示了流水线并行如何通过阶段划分提升计算效率。

#### 张量并行（Tensor Parallelism）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/17.webp)

::: image-caption
张量模型并行（Transformer）：对 Transformer 权重分割，通过 AllReduce 同步部分结果，实现大张量计算的并行化。
:::

针对模型中的核心运算（如注意力机制、矩阵乘法），对张量进行维度拆分（如对注意力头做列切分），由多台设备同时执行部分运算，再合并结果。例如 4 张 GPU 可同时计算一个矩阵乘法任务，大幅提升单算子的计算速度，通过 Transformer 模型的张量并行示例，清晰呈现了这一过程。

#### 混合并行策略

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/18.webp)

::: image-caption
混合并行策略的重要性。单一并行策略难以扩展，结合数据与模型并行可在 512 GPU 时仍保持 > 70% 效率。引用 Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.
:::

单一并行策略往往存在局限，通过实验数据证明了组合策略的威力：

::: block-1
- 仅使用数据并行时，从 1 张卡逐步加到 8 张，线性加速比只能达到 77%；
- 而采用 2D 并行（模型并行+数据并行）后，即使到了512 张卡，线性扩展性才降低到 74%，这一结果充分体现了多并行策略协同的价值。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/19.webp)

::: image-caption
LLM 训练的 3D 并行结合流水线并行（层分割）、张量并行（权重分割）、数据并行（样本分割），支持万亿参数模型训练。图表显示 GPU 数量与参数规模、吞吐量的正相关关系。https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/
:::

### 2. 内核融合：单卡内部的“微雕艺术”

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/20.webp)

::: image-caption
核（Kernel）的定义。编程抽象：线程块（Thread Block）和线程（Thread）组成的执行单元。硬件映射：线程块对应流式多处理器（SM），包含张量核心、寄存器、共享内存等。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/21.webp)

::: image-caption
现有 “每层一核” 模式。Transformer 各层（MatMul、Attention 等）对应独立核，SM 上核执行存在资源闲置（如部分 SM 等待核切换）。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/22.webp)

::: image-caption
现有 “每层一核” 模式。
:::

> 传统的 ML 计算采用“一层一个内核（Kernel-Per-Layer）”的方式，即每个模型层（如 MatMul、Attention、AllReduce）对应一个 cuBLAS 或 cuDNN 内核，这种方式存在明显缺陷，Zhihao 通过表格和时序图揭示了这些问题：

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/23.webp)

::: image-caption
传统核模式的局限性。无层间流水线（核壁垒阻碍）、计算与通信无法重叠（粗粒度依赖）、动态性差（每次前向传播启动数百至数千核）。
:::

::: block-1
- 一次模型前向传播需要启动数百个内核，而每次内核启动都存在固定开销，大量启动操作累积会显著增加计算延迟；
- 内核之间存在 barrier（同步屏障），必须等待前一个内核完全执行完毕，后一个内核才能开始，无法实现计算与通信（如数据传输）的重叠；
- 依赖 CUDA Graph 进行内核调度时，只能处理静态任务流程，无法应对连续批处理（Continuous batching）、投机解码（Speculative decoding）等动态形状场景，灵活性极差。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/24.webp)

::: image-caption
核融合（Kernel Fusion）。原理：将多个核（如 RMSNorm 和 MatMul）合并为一个，减少内存访问和核启动开销，通过数学公式合并展示优化效果。
:::

为解决这些问题，研究者提出“巨型内核（Mega-Kernel）”的优化思路：将模型中的多个连续算子（如 RMSNorm → MatMul → Attention → AllReduce）全部整合到同一个 CUDA 内核中，利用寄存器实现流水线存储（pipeline register），用共享内存构建生产者-消费者队列，实现算子间数据的高效流转。下图展示了巨型内核的理想效果。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/25.webp)

::: image-caption
大核（Mega-Kernel）的可行性：通过无核壁垒、操作重排、负载均衡，融合所有层为单一核。
:::

::: block-1
- 彻底消除内核启动开销，将原本需要启动上千次的内核浓缩为 1 个，大幅减少延迟；
- 通过软件流水线技术，将矩阵乘法（GEMM）与全局归约（AllReduce）等操作重叠执行，提升设备利用率；
- 无需依赖 CUDA Graph 的静态调度，在动态批处理场景下无需重新捕获任务流程，灵活性显著提升。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/26.webp)

::: image-caption
大核（Mega-Kernel）的优势：支持层间流水线、计算与通信重叠、动态负载均衡（无需 CUDA 图）。
:::

## 四、自动并行：从 FlexFlow 到 TopoOpt

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/27.webp)

::: image-caption
手动设计并行策略存在三大问题：难以设计、性能次优、可移植性差，列举张量并行、序列并行等多种策略的复杂性。
:::

> 随着硬件架构日益复杂、模型规模持续增长，手动设计并行策略的方式已走到尽头——不仅需要投入大量人力，且难以适配不同硬件和模型，优化效果也难以保证。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/28.webp)

::: image-caption
FlexFlow：自动优化 DNN 并行化。详见：https://github.com/flexflow
:::

> Zhihao 介绍了 CMU 与 Stanford 合作的开源项目 FlexFlow（https://github.com/flexflow，https://github.com/flexflow），它通过自动化技术，彻底改变了并行策略的设计模式。

FlexFlow 的核心逻辑是构建“计算图 + 硬件拓扑”的输入模型，通过内置的执行搜索引擎（Execution Search Engine）和模拟器（Simulator），结合代价模型（Cost Model），自动探索最优并行策略。实验数据显示，这种自动化方式仅需几分钟，就能生成优于专家手工调优 10 倍的策略，大幅降低了并行优化的门槛。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/29.webp)

::: image-caption
FlexFlow 在推荐模型训练中的性能。对比数据并行和模型并行，FlexFlow 在 96 GPU 时吞吐量提升 10 倍，验证其大规模扩展能力。Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models. ISCA’22
:::

在实际应用场景中，FlexFlow 展现出强大的性能优势：

::: block-1
- 在大规模推荐系统训练中，仅用 48 张 GPU 就替代了原本需要 200 张 GPU 的纯数据并行方案，训练吞吐量（#samples/second）达到传统方案的数倍，Zhihao 在第 21 页的折线图清晰对比了 FlexFlow 与数据并行、模型并行的性能差异；
- 在 AI4Science 领域的计算流体力学（CFD）模拟任务中，FlexFlow 支持 768 张 A100 GPU 协同计算，将原本需要 3 天的训练时间压缩到 5 小时，效率提升超 14 倍，柱状图直观呈现了不同 GPU 数量下，FlexFlow 与 TensorFlow 的训练时间差异。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/30.webp)

::: image-caption
FlexFlow 加速 AI4Science 模型训练。相比 TensorFlow，在 768 GPU 时训练时间减少 15 倍，支持从单 GPU 到大规模集群的高效扩展。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/31.webp)

::: image-caption
FlexFlow 合作与特性。跨机构合作（CMU、Meta 等），支持 TensorFlow/PyTorch 等框架，自动优化并行策略，提供 Keras 支持和大规模 GNN 加速。
:::

FlexFlow 的后续工作 TopoOpt 进一步拓展了优化边界。

::: block-1
- TopoOpt: Optimizing the Network Topology for Distributed DNN Training NSDI 2023.
- https://people.csail.mit.edu/ghobadi/papers/topoOpt_nsdi_2023.pdf
- https://github.com/hipersys-team/TopoOpt
:::

TopoOpt 将网络拓扑（如节点间的通信带宽、延迟）纳入联合优化范畴，通过优化数据在不同节点间的传输路径，确保网络资源被充分利用，真正实现了“把网线也用满”，为分布式 ML 系统的优化提供了更全面的解决方案。

## 五、巨型内核的工程落地：Mirage Persistent Kernel（MPK）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/32.webp)

::: image-caption
大核的关键挑战：依赖管理（需任务图）、动态性处理（连续批处理等）、性能优化（现有编译器不支持），引入 Mirage 超级优化器。Mirage: A Multi-Level Superoptimizer for Tensor Programs. OSDI’25
:::

> 虽然巨型内核的思路为单卡优化提供了方向，但如何将其工程化落地，仍是一个复杂的挑战。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/33.webp)

::: image-caption
MPK（Mirage Persistent Kernel）。功能：将 LLM 服务编译为高度优化的大核。优势：低开发量（几十行 Python）、性能提升 1.2-6.7 倍、支持新模型（无需手动实现）。MPK 工作流：输入服务配置（批处理、投机解码等）和 LLM，通过编译器生成任务图，运行时执行。
:::

Zhihao 详细介绍了 Mirage 项目提出的 Mirage Persistent Kernel（MPK）方案，从编程模型、编译器、运行时三个维度，构建了完整的巨型内核落地体系。

### 1. 编程模型：Task Graph（任务图）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/34.webp)

::: image-caption
MPK 任务图：由 SM 级任务（计算 / 通信）和事件（同步）组成，任务触发事件，事件驱动后续任务。
:::

> MPK 摒弃了传统的“内核级”调度，提出以“任务”为核心的 Task Graph 编程模型。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/35.webp)

::: image-caption
对比任务图（Task Graph）与 CUDA 图（CUDA Graph）的核心差异。任务图被定义为 “更低级别的” CUDA 图，其关键特点是能够捕获子内核级别的依赖关系，且具有静态、不可变的属性，构建一次后可被多次重放。此外，CUDA 图的节点是 GPU 上的完整内核，例如矩阵乘法（MatMul）、注意力机制（Attn）、AllReduce 等操作对应的独立内核；而任务图的节点则是流多处理器（SM）上的具体任务，涵盖矩阵乘法任务（MM）、事件（E）、AllReduce 任务（AR）、注意力任务（AT）等更细粒度的工作单元。
:::

与 CUDA Graph 不同，Task Graph 的节点是“运行在流多处理器（SM）上的任务”，而非“整个内核”，每个任务可分为三类：

::: block-1
- **计算任务（Compute Task）**：如矩阵乘法（GEMM）、Softmax、LayerNorm 等核心算子运算；
- **通信任务（Comm Task）**：如 NCCL AllReduce（全局归约）、P2P Copy（点对点数据传输）等跨设备或设备内数据交互操作；
- **事件（Event）**：任务完成的信号标识，用于触发后续依赖任务的执行，是实现细粒度同步的关键。
:::

这种差异使得任务图能够更精准地管理内核内部的依赖关系，为大内核（Mega-Kernel）中的层间流水线、计算与通信重叠等优化提供了基础支持。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/36.webp)

::: image-caption
任务图结构由矩阵乘法任务（MM）、AllReduce 任务（AR）、注意力任务（AT）、事件（E）等元素交织构成
:::

任务是流多处理器（SM）上的一个工作单元，事件是任务间的同步机制；任务完成后会通知对应的事件，而事件被触发后会启动所有依赖该事件的任务。上图呈现了 SM 上的通信任务、计算任务以及任务间的事件同步关系，体现了任务启动与事件通知的交互流程。

总得来说，任务图通过“事件驱动”机制，MPK 实现了高效的任务调度：

::: block-1
- 调度器（Scheduler）运行在一个 warp（线程束）上，负责管理事件队列，一旦某个事件被完全触发（即所有依赖任务完成），就立即启动后续关联任务；
- 工作器（Worker）运行在 SM 上，从任务队列中获取任务并执行，执行完成后触发对应事件。
:::

这种无锁队列 + 细粒度依赖的设计，大幅提升了任务执行的灵活性和效率。

### 2. 编译器：从 Layer 到 Task 的映射

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/37.webp)

::: image-caption
MPK 编译器。功能：将层分解为任务（基于 SM 数量）、添加同步事件、生成高效 CUDA 代码。
:::

> MPK 编译器是实现“层-任务”转化的核心组件，其输入为“某层 Transformer 的算子序列”，输出则是可直接在 GPU 上执行的任务分配方案，具体包括三个关键步骤：

::: block-1
- **算子拆分（Operator Decomposition）**：将每个算子（如 MatMul）拆分为多个适合 SM 执行的 tile（子任务块），确保每个 tile 的计算量与 SM 的算力相匹配；
- **任务分配（Task Allocation）**：根据硬件拓扑（如 SM 数量、性能差异），将拆分后的 tile 分配到具体的 SM 上，避免部分 SM 负载过高而 others 空闲的情况；
- **事件插入（Event Insertion）**：分析任务间的依赖关系，在关键节点插入事件同步，确保任务按正确顺序执行，同时最大限度减少同步开销。
:::

目前，MPK 编译器主要依赖启发式算法 + 基于 profiling 的调优方式，已在 NVIDIA A100、H100、B200 等主流 GPU 上完成验证，能稳定输出高效的任务分配方案。但 Zhihao 也指出，其下一步的核心挑战是实现跨 GPU 架构的自动调优，摆脱对特定硬件的依赖，提升方案的通用性。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/38.webp)

### 3. 运行时：Event-Driven Execution（事件驱动执行）

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/39.webp)

::: image-caption
MPK 运行时架构。调度器（每个 warp 一个）管理事件队列，工作器（每个 SM 一个）执行任务并触发事件，利用 PCIe 5.0 和 NVLink 加速通信。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/40.webp)

::: image-caption
Task-Based Parallel Runtime（基于任务的并行运行时）的本质，是“用软件层的任务并行运行时，将 GPU 的硬件并行能力（多 SM、多 Warp）与 LLM 的细粒度计算 / 通信需求对齐”，最终实现 “计算不停顿、资源不空闲、延迟可压缩” 的高效推理
:::

> 在 LLM 推理的并行执行架构中，Task-Based Parallel Runtime（基于任务的并行运行时） 和 Event-Driven Execution（事件驱动执行） 是互补协作的核心组件，共同解决 “如何让 GPU 高效处理细粒度任务” 的问题。

Event-Driven Execution（事件驱动执行）是 Mirage Persistent Kernel（MPK）中实现高效 LLM 推理的核心机制，其核心是通过任务（task）与事件（event）的动态交互，实现 GPU 上计算与通信的细粒度协同、资源高效利用和低延迟执行。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/41.webp)

::: image-caption
事件驱动执行机制，这里展示包含矩阵乘法（MM）、AllReduce（AR）、注意力机制（AT）等任务及事件（E）的任务图
:::

> 这张图是 Event-Driven Execution（事件驱动执行） 在 LLM 推理中的流程示意图，核心展示了 任务并行与流水线化 的实现逻辑。我们可以从 **模块含义、执行流程、设计优势** 三个维度拆解。

#### 模块

| 模块 | 含义 | 角色 |
| --- | --- | --- |
| 蓝色 AT | Attention（注意力计算） | LLM 推理的核心计算任务（如多头注意力的每个头，或每个 batch 的注意力计算） |
| 绿色 E | Event（事件） | 任务完成的同步信号，触发后续依赖任务的启动 |
| 浅蓝色 MM | MatMul（矩阵乘法） | 计算任务（如 QKV 投影、FeedForward 层的 GEMM 操作） |
| 橙色 AR | AllReduce（跨 GPU 通信） | 通信任务（多 GPU 数据并行时，同步中间结果，确保各 GPU 状态一致） |

#### 执行流程

从左到右的任务流转，有下面三个阶段：

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/42.webp)

::: image-caption
阶段1：Attention 并行执行（开头的那个图的左侧 AT）。左：空闲，右：Attention 并行执行
:::

> 阶段 1：Attention 并行执行（开头的那个图的左侧 AT）

- 初始事件`E`（开头的那个图的最左侧绿色圆圈）触发 多个 AT 任务并行执行（图中 4 个 AT 同时运行）。
- 这些 AT 可以是：同一层的多头注意力（每个头独立计算），或连续 batch 的注意力推理（批处理并行）。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/43.webp)

::: image-caption
阶段2：事件触发 MatMul（开头的那个图的中间E→MM）。左：Attention 并行执行，右：事件触发 MatMul
:::

> 阶段 2：事件触发 MatMul（开头的那个图的中间 E→MM）

- 每个 AT 完成后，立即触发专属事件`E`（中间的绿色圆圈，每个 AT 对应一个 E）。
- 这些`E`作为依赖信号，触发对应的 MM 任务（浅蓝色）并行执行——无需等待所有 AT 完成（传统模式的核心瓶颈！）。

例如：第一个 AT 完成后，立即启动第一个 MM；第二个 AT 完成后，立即启动第二个 MM ，实现 “流级并行”（pipeline）。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/44.webp)

::: image-caption
阶段3：事件触发AllReduce（开头的那个图的右侧E→AR）
:::

> 阶段 3：事件触发 AllReduce（开头的那个图的右侧 E→AR）

- 每个 MM 完成后，再次触发事件`E`（开头的那个图的右侧绿色圆圈，每个 MM 对应一个 E）。
- 这些`E`启动 AR 任务（橙色）并行执行——多 GPU 间同步 MM 的计算结果。

Event-Driven Execution 由两类核心组件构成，协同完成任务调度与执行，Zhihao 通过以上多组示意图展示了运行时的工作流程：调度器持续监控事件队列，当某个事件被触发（如一个 Attention 任务完成），立即从任务队列中取出依赖该事件的 MatMul 任务，分配给空闲的 SM 执行；工作器执行完任务后，会触发新的事件，进而驱动下一批任务的执行，形成闭环。

这里涉及到了四个概念：

::: block-1
- 调度器（Scheduler）：运行在 GPU 的 warp 上，负责监听事件队列（Event Queue），当事件被触发时，启动依赖该事件的后续任务。
- 工作器（Worker）：运行在 GPU 的流多处理器（SM）上，负责从任务队列（Task Queue）中获取任务并执行，完成后触发对应的事件，通知调度器启动后续依赖任务。
:::

#### 事件驱动的核心设计优势

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/41.webp)

::: image-caption
事件驱动执行机制，这里展示包含矩阵乘法（MM）、AllReduce（AR）、注意力机制（AT）等任务及事件（E）的任务图
:::

1. **消除“阶段内同步壁垒”** ：传统模式中，必须等同阶段所有任务完成（如所有 AT 都算完），才会启动下一阶段（如 MM）。事件驱动中，单个任务完成就触发后续任务（AT1 完成 → 启动 MM1，AT2 完成 → 启动 MM2），避免“等最慢任务”的空闲浪费。
2. **实现“跨阶段流水线化”**，任务执行形成流水线：
3. **最大化并行效率**

---

其实，对比传统模式，就愈发凸显事件驱动在计算-通信持续重叠的优势。

| 传统模式（层间全局同步） | 事件驱动模式（细粒度事件触发） |
| --- | --- |
| 等所有 AT 完成 → 全局 E → 启动所有 MM | 每个 AT 完成 → 专属 E → 启动对应 MM |
| 等所有 MM 完成 → 全局 E → 启动所有 AR | 每个 MM 完成 → 专属 E → 启动对应 AR |
| 阶段间必须完全串行，空闲多 | 阶段间流水线化，计算-通信持续重叠 |

> Event-Driven Execution 用“任务 → 事件 → 任务”的细粒度依赖链，打破传统“整阶段同步”的枷锁，让 LLM 推理的计算（Attention、MatMul）和通信（AllReduce）像“接力赛”一样连续推进，最终实现低延迟、高并行的推理执行。

之后引出了必备环节，性能 benchmark：

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/45.webp)

::: image-caption
MPK 降低 LLM 推理延迟。在 Qwen3-8B 模型上，MPK 将单 token 延迟从 14.5ms 降至 12.5ms（接近 10ms 理论极限），优于 PyTorch、VLLM 等系统。
:::

实验结果充分验证了 MPK 运行时的性能优势：在 Qwen3-8B 模型（BF16 精度、单批次）的推理任务中，使用 A100 GPU 时，MPK 将单 token 延迟从 PyTorch 的 14.5ms 降至 12.5ms，距离 10ms 的硬件理论下限仅差 2.5ms；在 2-4 卡扩展场景下，MPK 继续保持领先，性能优于 vLLM、SGLang 等主流推理框架。

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/46.webp)

::: image-caption
MPK 项目资源。GitHub 链接（mirage-project/mirage），合作机构包括华盛顿大学、普渡大学等，相关技术获社区关注。
:::

此外，MPK 还具备低工程成本、高扩展性的优势：仅需几十行 Python 代码，就能将一个 LLM 模型“巨型内核化”（megakernelize）；对新模型的支持无需手动实现，可实现“Day-0 支持”；在不同硬件平台上的适配也无需大量修改代码，大幅降低了落地门槛。

## 六、未竟之路：写在摩尔定律之后的思考

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/47.webp)

::: image-caption
开放问题。层到任务的分解：当前依赖启发式 + 调优，需更自动化方法以支持多样 GPU。
:::

![文章插图](/articles/2025-08-29-hotchips-2025-gpu-zhihao-jia-5492cd10/images/48.webp)

::: image-caption
开放问题。任务调度：轮询调度（无同步但负载不均）vs. 动态调度（适应负载但有协调开销）。
:::

Zhihao 最后并未停留在现有成果，而是抛出了几个核心开放问题，这些问题不仅是 MPK 方案的待解方向，更指明了下一个十年 ML 系统优化的研究重点：

::: block-1
1. **“层-任务”切分的自动化**：当前 MPK 依赖人工设计的启发式规则 + profiling 调优来实现“层-任务”切分，这种方式效率低、通用性差。如何构建完全自动化的切分算法，根据模型结构、硬件特性动态决定切分粒度与方式，是实现方案通用化的关键。
2. **硬件快速适配能力**：新 GPU 架构（如未来的 NVIDIA 新系列、AMD 或自研 GPU）每 2-3 年就会发布，而现有方案需要数月甚至半年的手工调优才能适配。如何实现“硬件发布即支持”，在新架构发布第一天就生成高效的并行策略与内核代码，是解决“硬件-软件错位”的核心。
3. **跨硬件架构拓展**：目前巨型内核与并行化的思路主要针对 GPU 设计，如何将其拓展到 CPU、TPU、NPU 乃至存内计算等其他硬件架构，构建统一的 ML 系统优化框架，是实现“全场景高效计算”的必答题。
:::

> 针对这些问题，Zhihao 进一步指出了下一个十年的研究方向：

- **编译器层面**：借鉴数据库优化器的设计思路，将“成本模型 + 搜索算法”打造为编译器的基础设施，实现策略的全自动生成；
- **运行时层面**：将“事件驱动 + 任务窃取”机制深度优化，使其成为 GPU 上的通用高效运行时，类似 Erlang 在分布式系统中的角色；
- **系统设计层面**：打破“并行化”与“内核融合”的独立优化思维，将二者视为一个联合优化问题，通过协同设计实现 1+1>2 的效果。

## 结语

从 1940 年的真空管计算机，到 2025 年的 NVIDIA B200 GPU，人类用 85 年时间，将“每美元算力”提升了 $10^{15}$ 倍。站在这个节点回望，GPU 上 ML 系统优化的十年跃迁，呈现出清晰的演进逻辑：

::: block-1
- 指数级增长的算力需求，是 AI 时代的新“熵增”，它推动着技术不断突破边界；
- 并行化与内核融合，是对抗熵增的两条核心手臂——并行化向外拓展算力规模，内核融合向内挖掘单卡效率；
- 而自动化的编译与运行时（如 FlexFlow、MPK），是让这两条手臂无限延伸的大脑，它解决了“人工优化跟不上硬件与模型发展”的根本矛盾。
:::

当巨型内核把 1000 个分散的 kernel 浓缩成 1 个，当 FlexFlow 把 10 人月的调优工作压缩成 10 分钟的自动化搜索，我们或许终于有资格说：

> “硬件的摩尔定律会放缓，但系统的摩尔定律才刚刚开始。”

交个朋友
