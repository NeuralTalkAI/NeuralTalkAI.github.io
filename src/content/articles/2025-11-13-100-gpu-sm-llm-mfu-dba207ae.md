关键词：:mdnice-bold-italic[GPU 利用率]、**MFU（模型浮点运算利用率）**、**SM 效率**、内核融合、:mdnice-bold-italic[LLM 训练优化]、分布式策略兼容

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/01.webp)

- GPU Utilization is a Misleading Metric
- https://www.trainy.ai/blog/gpu-utilization-misleading
- 8000 字，阅读 23 分钟，**播客 11 分钟**

相关推荐

- [Apple Silicon LLM 推理全方位深度剖析：与 NVIDIA GPU 从性能到成本的终极对决！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890883&idx=1&sn=d05b1dbc8df195d7a60dfc6d2bee4d6f&scene=21#wechat_redirect)
- [LLM推理极限建模：Bandwidth, Compute, Sync and Capacity are All You Need](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890183&idx=1&sn=2fc6e4e2e0c705123df18ad6d9f0dcb4&scene=21#wechat_redirect)
- [MoE 所有层融到一个分布式算子GPU Kernel！FlashDMoE：GPU内核-硬件协同解锁大规模分布式机器学习性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888883&idx=1&sn=14a76e02fc523d30f613d4a2219c0fea&scene=21#wechat_redirect)

> 本文指出 GPU 利用率（常用 nvidia-smi 查询）是易误导的性能指标，Trainy 团队在 LLM 训练优化中发现，即便 GPU 利用率达 100%，模型的 MFU（模型浮点运算利用率）仅 20%，远低于行业常见的 35%-45%。

GPU 利用率的核心缺陷的是**仅衡量** “:mdnice-bold-italic[过去采样周期内 GPU 是否有内核执行] ”，**不反映核心是否充分利用或工作负载并行度，极端情况下纯内存读写就能拉满该指标**。而 SM 效率（流式多处理器活跃度）能更精准反映 GPU 实际算力利用，可定位未充分优化的内核。

团队通过融合 Transformer 层内核（如 Flash Attention、FusedMLP），替代 PyTorch 原生层实现，结合适配的模型并行策略，最终实现训练速度 4 倍提升，MFU 提升至 38%。

::: block-1
作者**建议 AI 团队同时跟踪 GPU 利用率与 SM 效率**，:mdnice-bold-italic[MFU 虽精准但不适用于实时分层监控]，而 SM 效率可通过 Nvidia DCGM 直接获取。
:::

Nvidia DCGM（Data Center GPU Manager，数据中心 GPU 管理器）是英伟达推出的专业 GPU 集群管理与监控工具，核心定位是帮助数据中心、AI 训练集群等场景实现对 GPU 硬件的精细化管控，解决传统工具（如 nvidia-smi）监控维度有限、无法适配大规模集群的问题。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/02.webp)

## 关键问题

### 问题 1：当 GPU 利用率已达 100% 但 MFU 仅 20% 时，核心矛盾是 SM 未充分激活还是内核未实现算力并行，该如何通过工具快速区分这两种场景？

> 核心矛盾本质是“**内核未充分利用 GPU 硬件资源** ”，具体可拆解为 SM 激活不足或算力并行缺失，两者可通过工具快速区分：

1. 用**Nvidia DCGM** 或**PyTorch Profiler** 查看 SM 效率：若 SM 效率低于 10%（如 H100 单 SM 运行时仅 0.7%），直接指向 SM 未充分激活，此时内核未调度足够多的流式多处理器。
2. 用**PyTorch Profiler** 分析内核细节：若 SM 效率较高但 MFU 低，说明核心矛盾是内核未实现算力并行——比如未融合的 Softmax 等内存绑定内核，虽占用多个 SM 但主要时间用于内存读写，浮点运算占比极低，可通过查看内核的“浮点运算耗时占比”或“内存读写耗时占比”验证。

---

### 问题 2：面对 `torch.compile` 与 FSDP 等分布式策略不兼容、手动替换融合内核效率低的问题，AI 团队在平衡开发成本与算力提升（如 4 倍提速、MFU 从 20% 到 38%）时，应优先突破哪类技术瓶颈？

> 应优先突破**“兼容分布式策略的自动内核融合工具”** 这一技术瓶颈，核心原因如下：

1. 手动替换融合内核的开发成本与维护成本极高，且难以规模化推广，原文中团队虽通过手动替换实现 4 倍提速，但明确提到“**识别替换位置是最大挑战** ”，**无法适配快速迭代的模型与分布式策略。**
2. `torch.compile` 的核心价值是自动化优化，其与 FSDP 等分布式策略的不兼容、 graph breaks 导致的提速不及预期，是当前阻碍“低成本高算力提升”的关键——若能解决兼容性问题，可同时覆盖“内核融合”与“分布式训练”两大核心场景，直接降低开发成本，同时复用原文作者已验证的“融合内核=MFU 从 20%提升至 38%”的算力增益。
3. 模型并行策略的优化空间相对有限，原文中仅作为辅助优化（结合 3.2 Tbps Infiniband），而内核融合是算力提升的核心来源，优先解决其自动化与兼容性问题，投入产出比更高。

## 本文目录

- 关键问题
  - 问题 1：当 GPU 利用率已达 100% 但 MFU 仅 20% 时，核心矛盾是 SM 未充分激活还是内核未实现算力并行，该如何通过工具快速区分这两种场景？
  - 问题 2：面对 `torch.compile` 与 FSDP 等分布式策略不兼容、手动替换融合内核效率低的问题，AI 团队在平衡开发成本与算力提升（如 4 倍提速、MFU 从 20% 到 38%）时，应优先突破哪类技术瓶颈？
- 零、业务背景与 Google 提出的 MFU
  - MFU 是什么？
  - Google PaLM 的 Training Efficiency
- 一、GPU 利用率的真实定义是什么？
  - USE 方法论
- 二、深入探究问题根源
- 三、但 SM 效率究竟代表什么？
  - 流式多处理器（SM）
  - SM 效率如何计算？
  - SM 效率
  - 流水线利用率
- 四、实施优化措施
- 五、结论

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/03.webp)

## 零、业务背景与 Google 提出的 MFU

> 在Trainy[1]公司，我们致力于**研发 GPU 集群管理基础设施**，因此会投入大量时间研究这类问题。

去年，我们与一家基金会模型（Foundation Model）公司合作，助力其扩展大型语言模型（LLM）训练规模并提升训练效率。我们遵循了几乎所有 PyTorch 性能调优指南中提到的基础步骤，具体如下：

- 通过修改**数据加载器** （dataloader）默认参数，使 GPU 达到饱和运行状态，如工作进程数`num_workers`、批次大小`batch_size`、内存锁定`pin_memory`、预取因子`prefetch_factor`等；
- 使用**混合精度**，如`fp16`半精度浮点数、`bf16`脑半精度浮点数，从而最大化张量核心的使用率。其中，Tensor Core 是 GPU 中专门用于加速矩阵运算的硬件单元；
- 采用来自 Apex/DeepSpeed（两款常用的 PyTorch 训练优化库）的**融合优化器**，如`FusedAdam`、`FusedAdamW`等，融合优化器**将多个独立计算步骤合并**，减少数据在内存中的往返耗时；
- **使用专为训练设计的实例/网络**，如 H100SXM、A100SXM 型号 GPU。同时，在条件允许的情况下优先选用更新型号的实例，优先级为 H100 > A100 > V100。

这些简单的调整让我们实现了 100%的 GPU 利用率，同时 GPU 功耗也显著提升，从表面看效果极佳！为了确认是否还有优化空间，我们计算了该训练任务的模型 FLOPS 利用率（MFU）。

### MFU 是什么？

> 快速回顾：MFU，即模型 FLOPS（每秒浮点运算次数，Floating point Operations Per Second，衡量计算能力的核心单位）利用率，是了解 GPU 性能的最佳指标之一，该概念在Google 的 PaLM 论文[2]中首次提出。

其定义为“**在峰值 FLOPS 运行状态下，系统的实际吞吐量（每秒处理的 token 数）与理论最大吞吐量的比值** ”。简单来说，它能告诉你当前任务的每秒浮点运算量占 GPU 最大浮点运算能力的百分比。

::: block-1
MFU 唯一的实际缺点是，与 GPU 利用率这类指标相比，其计算难度更高——因为它依赖于模型参数和所使用的深度学习框架。
:::

下面这部分来自Google 的 PaLM 论文[3]附录 Compute Usage and Environmental Impact 部分理解：

稠密 Transformer 仅解码器模型 MFU（模型浮点运算利用率）的计算逻辑，本质是通过 “实际算力消耗” 与 “硬件理论峰值算力” 的比值，衡量模型训练 / 推理时对 GPU 浮点运算能力的利用效率。

#### 核心前提：FLOPs 的构成 —— 模型参数与自注意力的双重贡献

> MFU 计算的基础是先明确 “**模型每处理 1 个令牌（Token）需消耗多少浮点运算（FLOPs）** ”，分为两部分：

- **基础参数相关 FLOPs**：不考虑自注意力时，N 个参数的仅解码器模型，每令牌需 6N 次矩阵乘法 FLOPs。逻辑是：前向传播（推理 / 训练正向计算）中，每个参数参与 1 次乘法 + 1 次加法（即 1 次矩阵乘法 FLOP），共 2N 次；反向传播（训练反向更新）需为前向的每个矩阵乘法对应 2 次矩阵乘法（梯度计算逻辑），共 4N 次，两者合计 6N。
- **自注意力额外 FLOPs**：稠密自注意力会新增 6LH (2QT) 次 FLOPs / 令牌（L = 层数、H = 头数、Q = 头维度、T = 序列长度）。虽然这部分对大型模型占比小，但需纳入以保证计算完整。

#### MFU 的核心公式：实际吞吐量 / 理论峰值吞吐量

- 第一步：算硬件理论峰值吞吐量（R）。若加速器总理论峰值矩阵乘法算力为 P（FLOPs / 秒），则**每秒钟能处理的最大令牌数 R = P /（6N + 12LHQT）** —— 分母是 “每令牌总 FLOPs”，即基础参数 FLOPs 与自注意力 FLOPs 之和。
- 第二步：算MFU。**用模型实际达到的 “令牌吞吐量”（每秒处理多少个令牌）除以 R，得到的比值就是 MFU**。:mdnice-bold-italic[比值越高，说明模型实际消耗的浮点运算越接近硬件极限，算力利用越充分。]

MFU 聚焦 “浮点运算（模型核心计算）的利用效率”，直接反映模型是否在 “高效用算力”—— 比如前文提到的 “100% GPU 利用率但 MFU 仅 20%”，本质是模型实际浮点运算量远低于硬件上限，大量 GPU 资源被非计算任务占用。

### Google PaLM 的 Training Efficiency

> 之前的大模型常用 “硬件 FLOPs 利用率”（HFU）衡量效率，通常反映了在特定设备上观察到的 FLOPs 与该设备理论峰值 FLOPs 的比率估计。然而，硬件 FLOPs 利用率存在几个问题。

- 首先，**所执行的硬件 FLOPs 数量取决于系统、实现方式，并且编译器的设计选择可能会导致操作数量的不同**。例如：如果它们不能全部容纳，一些前向传播操作可以重新计算（使得一些激活值能够被重计算而不是存储）。:mdnice-bold-italic[这就产生了一种权衡：使用额外的硬件 FLOPs 可以节省内存，但训练系统的最终目标是实现每秒高的令牌吞吐量（从而实现快速训练），而不是尽可能多地使用硬件 FLOPs。]
- 其次，观察到的硬件 FLOPs 的测量取决于用于计数或跟踪它们的方法。已报道的观察到的硬件 FLOPs 是基于分析核算（Narayanan 等人，2021b）以及使用硬件性能计数器得出的。

PaLM 团队提出了更合理的 “模型 FLOPs 利用率”（MFU）—— **只计算模型本身必需的算力（前向 + 反向传播）**，不包含额外开销，PaLM 核心效率指标结果如下：

- MFU 达 46.2%（包含自注意力层），如果不算自注意力层是 45.7%；
- 硬件 FLOPs 利用率（包含额外计算）达 57.8%；
- 对比前代模型：GPT-3 的 MFU 只有 21.3%，Gopher（2800 亿参数）是 32.5%，Megatron-Turing NLG（5300 亿参数）是 30.2%，PaLM 的效率几乎是前代的 1.5 倍以上。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/04.webp)

::: image-caption
PaLM 和之前大型模型的 FLOPs 利用率。由于在模型、编译器和并行策略方面进行了多项优化，PaLM 的模型 FLOPs 利用率（MFU）显著较高。PaLM 相应的硬件 FLOPs 利用率为 57.8%
:::

---

**遗憾的是，我们的模型训练的 MFU 仅达到约 20%。** 作为参考，如今大多数 LLM 训练的 MFU 通常在35% - 45%[4]之间。于是问题来了：既然 GPU 利用率已达到 100%，为何 GPU 计算能力的理论最大值我们仅用到了 20%？

要解答这个问题，我们需要先更深入地理解 GPU 利用率究竟在衡量什么。

## 一、GPU 利用率的真实定义是什么？

> 英伟达的 NVML 库是 GPU 管理库，用 C 语言写的，可以用来监控和管理英伟达的 GPU 设备，像查看 GPU 的温度、使用率，管理运行的程序都可以，NVML 文档[5]中对 GPU 利用率的定义较为模糊，仅描述为“当前 GPU 计算资源和内存接口的使用率”。

而在 Datadog 的 NVML（英伟达管理库，用于监控 GPU 状态的接口）文档[6]中，我们能找到更清晰的定义：

::: block-1
“在过去的采样周期内，GPU 上有一个或多个内核（Kernel，GPU 上执行具体计算任务的最小单元）正在运行的时间百分比”。
:::

要理解为何这一定义具有误导性，我们需要先简要了解 GPU 的工作原理。

GPU 包含**计算核心（Cores）和多处理器管理器（Multiprocessing Managers）** 。

- 在英伟达 GPU 中，这些多处理器管理器被称为流多处理器（Streaming Multiprocessors，简称 SM，可理解为“工作组负责人”，负责调度一组计算核心的任务）；
- 而在 AMD 显卡中，它们被称为计算单元（Compute Units，简称 CU）。

下图为 GH100 GPU 的示意图，该 GPU 包含 144 个 SM。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/05.webp)

::: image-caption
GH100 GPU示意图，含144个流多处理器（SM）
:::

我们可以将这些多处理器管理器看作一组“工人”（即计算核心）的“工头”。**当你启动一个 CUDA 内核（基于英伟达 CUDA 框架编写的计算任务）时，任务会由一个或多个 SM 分配给 CUDA 核心执行**。如下图所示，GH100 芯片上的单个 SM 就包含多个 CUDA 核心。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/06.webp)

::: image-caption
一张H100 GPU流多处理器内部架构的示意图。GPU核心以绿色显示，其他计算单元以褐红色显示，调度单元以橙色显示，内存以蓝色显示
:::

这意味着，“GPU 利用率”这一指标仅能衡量**某一时刻** 是否有内核在运行，:mdnice-bold-italic[完全无法反映该内核是否使用了所有可用核心，也无法体现任务是否已达到 GPU 的最大并行处理能力]。在**最极端的情况下，仅对内存进行读写操作** （不执行任何浮点运算，即 0 FLOPS），**也能实现 100%的 GPU 利用率。**

> 在此我们需要澄清一点：该指标仅对缺乏系统领域知识的人（如许多机器学习工程师）具有误导性。

正如“USE 方法论”[7]提到的，GPU 利用率的定义其实具有一定合理性。

### USE 方法论

USE Methodology，一种用于分析系统性能瓶颈的框架，即 Utilization 使用率、Saturation 饱和度、Errors 错误率。

与传统从给定指标反向推导的方式不同，**USE 方法从提问出发，构建检查清单，而非基于已有指标反向推导**。对于服务器分析，可快速定位资源瓶颈。例如在服务器性能问题排查时，按照 USE 方法，能系统地对 CPU、内存等资源进行检查，而不是盲目从某一指标入手 。

::: block-1
但回到我们面临的问题：这一定义恰好解释了疑问——为何 GPU 利用率百分比与 MFU 百分比之间会存在巨大差距！显然，GPU 仍有很大的性能潜力待挖掘，我们只需找到具体的优化方向。
:::

## 二、深入探究问题根源

> 要进一步挖掘性能潜力，下一步自然是**对模型的训练循环（Training Loop）进行性能分析**。我们使用 PyTorch 性能分析器（Pytorch Profiler）对训练循环进行了观察，以获取更详细的信息。

如下图所示，Softmax Kernel的 :mdnice-bold-italic[GPU 利用率很高，但一项名为“SM 效率”（SM Efficiency）的指标却很低]。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/07.webp)

::: image-caption
PyTorch性能分析器显示：Softmax内核运行时，SM效率较低
:::

这一现象立刻引起了我们的警觉——因为原生未优化的 Softmax 是 LLM 的典型性能瓶颈，其特点是“内存带宽受限”，即计算速度受限于数据在内存中的读写速度，而非 GPU 计算能力，为此业界已推出许多“内核融合”（Kernel Fusions），即将多个独立计算步骤合并为一个 GPU 内核，减少内存访问次数的方案，例如FlashAttention[8]（一种针对注意力机制的高效内核实现）。

基于这一信息，**SM 效率指标或许正指向模型执行过程中的效率问题。**

## 三、但 SM 效率究竟代表什么？

### 流式多处理器（SM）

NVIDIA GPU 的流式多处理器（SM）大致类似于 CPU 的核心[9]。也就是说，SM 既执行计算，又在寄存器中存储可用于计算的状态，并配有相关缓存。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/06.webp)

::: image-caption
一张H100 GPU流多处理器内部架构的示意图。GPU核心以绿色显示，其他计算单元以褐红色显示，调度单元以橙色显示，内存以蓝色显示
:::

**与 CPU 核心相比，GPU 的 SM 是简单、性能较弱的处理器**。SM 中的执行在一条指令内进行流水线处理（就像 20 世纪 90 年代以来几乎所有的 CPU 那样），但不存在推测执行或指令指针预测（这与所有当代高性能CPU不同）。

然而，GPU 的流多处理器（SM）可以并行执行更多线程。

::: block-1
- 一块 H100 SXM GPU 的最大功耗为 700 瓦，拥有 132 个 SM
- 每个 SM 配备 4 个 warp 调度器，每个调度器每时钟周期可向 32 个线程（即一个warp）并行发布指令，总共 $128 × 132 > 16,000$ 个并行线程
- 每个线程的功耗约为5 cW。需要注意的是，这是真正的并行：16,000 个线程中的每一个都能在每个时钟周期内取得进展
:::

GPU 的 SM 还支持大量并发线程——即执行指令交错进行的执行线程。

- H100 上的单个 SM 可以同时执行多达 2048 个线程
- 这些线程分为 64 个线程组，每个线程组包含 32 个线程
- 凭借 132 个SM，其总共可支持超过 250,000 个并发线程。

GPU 线程束之间的切换以单个时钟周期的速度进行，比 CPU 上的上下文切换快 1000 倍以上，这同样由 SM 的线程束调度器提供支持。:mdnice-bold-italic[大量可用线程束的数量以及线程束切换的速度有助于隐藏延迟，大量由内存读取、线程同步或其他耗时指令导致的延迟，确保由CUDA 核心和张量核心提供的算术带宽得到充分利用。]

### SM 效率如何计算？

> SM 效率（也称为 SM 活动，SM Activity）是英伟达 GPU 的一项指标，用于描述:mdnice-bold-italic[在特定时间间隔内 处于活跃状态的 SM 占总 SM 的百分比]。

正如我们之前所提及的，**SM 可看作管理一组 CUDA 核心的“工头”**。以英伟达 H100 GPU[10]为例，该 GPU 包含： 132 个 SM，每个 SM 有 128 个 CUDA 核心，因此总核心数为 16896 个。**通过监测 SM 效率，我们可以判断 CUDA 内核是否充分利用了流多处理器。**

::: block-1
假设一个 CUDA 内核连续运行 10 秒，但仅使用了 1 个 SM，那么在 H100 GPU 上，该任务的 GPU 利用率会显示为 100%，而 SM 效率则为 $1 / 132 = 0.7\%$。
:::

这正是我们一直在寻找的指标！我们可以逐层监测 SM 效率，从而找出那些“低垂的果实”——即通过优化可快速提升性能的层。这里带来两点启示：

1. **SM 效率可以成为核心监测指标**。GPU 利用率不反映核心是否真在高效算浮点，而 SM 效率能直接体现 GPU 核心是否被充分激活、算力是否实际投入运算。
2. **“逐层监测 SM 效率 + 寻找低垂果实”的优化逻辑**，具体来说：

### SM 效率

我们再回过头来看看SM 利用率。

> SM 利用率衡量的是流式多处理器（SM）**执行指令的【时间百分比】**。正如前文所述：:mdnice-bold-italic[在特定时间间隔内 处于活跃状态的 SM 占总 SM 的百分比]。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/08.webp)

::: image-caption
这是通过 NVIDIA-SMI 工具查看的 GPU 状态信息，显示有 8 块NVIDIA B200 GPU。驱动版本为 575.57.08，CUDA 版本 12.9。所有 GPU 风扇状态为 N/A，温度在 25℃-27℃之间，性能状态 P0，功耗在 136W-142W（功耗上限 1000W），显存使用均为 0MiB（总显存 183359MiB），GPU 利用率 0%，计算模式为 Default 且 MIG 禁用，当前无 ECC 错误
:::

SM 利用率类似于 GPU 利用率（utilization），后者由 nvidia-smi[11] 报告，更为人熟知，但 **SM 利用率的粒度更细**。它不会报告内核在 GPU 上任何位置执行的时间占比，而是报告:mdnice-bold-italic[所有 SM 用于执行内核的时间占比。]

::: block-1
如果某个内核仅使用一个 SM，例如只有一个线程块，那么在其运行期间，GPU 利用率将达到100%，但 SM 利用率**最多** 为 1 除以 SM 的数量——在H100 GPU中，这一比例不到1%。
:::

尽管 SM 利用率比 GPU 利用率更精细，它仍然不足以细致地反映 GPU 计算资源的使用情况。:mdnice-bold-italic[如果流多处理器利用率很高，但性能仍然不足，程序员应该检查【流水线利用率】]，它用于衡量每个流多处理器对其内部功能单元的有效利用程度。:mdnice-bold-italic[流多处理器利用率高而流水线利用率低，表明你的内核正在许多流多处理器上运行，但没有充分利用每个流多处理器内的计算资源。]

### 流水线利用率

> 流水线利用率衡量内核有效利用每个流式多处理器（SM）内执行资源的程度。

每个 SM 包含多个独立的执行管道，这些管道针对不同的指令类型进行了优化：

- CUDA核心用于通用浮点运算
- 张量核心用于张量收缩
- 加载/存储单元用于内存访问
- 控制流单元用于分支

::: block-1
管道利用率表示**当每个管道至少在积极执行一个线程束时，其达到的峰值速率的百分比**，该百分比是:mdnice-bold-italic[在所有活跃的SM上取平均值得到的。]

在调试管道利用率层面的应用性能【之前】，GPU 程序员:mdnice-bold-italic[应首先考虑 GPU 内核利用率和 SM 利用率。]
:::

管道利用率可在NSight Compute（ncu）的`sm__inst_executed_pipe_*.avg.pct_of_peak_sustained_active`指标中查看，其中星号代表特定管道，如 fma、tensor、lsu或adu（地址单元，主要处理地址计算等控制类操作）。

## 四、实施优化措施

> 既然我们能轻松识别出哪些内核未让 GPU“满负荷工作”，就可以针对性地优化这些层。

由于该模型基于 Transformer 网络结构（LLM 的核心基础结构），**优化的主要方向是对 Transformer 块（Transformer Block）定义中的层进行内核融合**，下图总结了我们的优化内容。

![文章插图](/articles/2025-11-13-100-gpu-sm-llm-mfu-dba207ae/images/09.webp)

::: image-caption
Transformer块内的内核融合示意图
:::

这里的“融合”（Fusing）指：不再使用 PyTorch 原生定义的一组层，而是用一个基于 CUDA 或 Triton 实现的 GPU Kernel 替代，将这组层的计算逻辑合并到一个内核中。之所以能实现加速，是因为对于某些层如Softmax[12]，内核每次读写内存的耗时，远少于其执行数学计算的耗时——**融合后可减少内存读写次数，从而提升效率**。

FlashAttention[13]就是这类融合内核的典型例子。除此之外，需要进行融合的其他内核还包括MLP[14]和dropout 层归一化残差加法[15]操作，后者是一种结合 dropout 正则化、层归一化和残差连接的计算步骤。

这些内核是我们自己编写的吗？并非如此。**大多数融合内核已在现有库中实现**，例如 FlashAttention 就提供了以`nn.Modules`（PyTorch 中构建神经网络层的基础模块）形式封装的层，**因此你无需从零开始实现** `torch.autograd.function`（PyTorch 中自定义自动求导逻辑的基础类）。此外，这些现有实现通常已针对硬件进行了优化，除了提升速度外，还能减少内存占用。

> 优化过程中最大的挑战在于：**如何定位代码中需要替换为融合层的位置**。

::: block-1
尽管 PyTorch 的编译优化工具`torch.compile`试图自动完成这一工作，但在撰写本文时，`torch.compile`与 FSDP（完全共享数据并行，一种用于训练超大规模模型的分布式策略）等较新的分布式策略兼容性不佳[16]。
:::

而且，由于模型计算图中存在无法被编译器优化的中断点，即“图断裂”（Graph Breaks），其实际加速效果往往未达预期。未来，我们期待 PyTorch 编译器能自动完成这些优化，但目前而言，我们仍需手动替换融合实现。

从优化结果来看，我们为该客户实现了**训练速度 4 倍的提升**，**MFU 也从最初的 20%提升至 38%。** 我们所做的优化中：

::: block-1
- 大部分来自上述**融合内核**
- **找到了合适的“模型并行级别”** （Model Parallelism Level，即如何将模型拆分到多个 GPU 上运行），这个过程基于模型大小和可用的 3.2 Tbps InfiniBand（一种高速网络技术，常用于分布式训练集群）
:::

## 五、结论

> 我们强烈建议大多数 AI 团队在监控 GPU 集群时，**同时跟踪** “SM 效率”和“GPU 利用率”。

**SM 效率能更真实地反映你对 GPU 性能的挖掘程度**，而 :mdnice-bold-italic[GPU 利用率【仅】能用于判断机器是否处于空闲状态]。当然，计算 MFU 也很有价值，但它并非一种可实时监控、或逐层监控的指标。

相比之下，英伟达 DCGM[17]（数据中心 GPU 管理器，Nvidia Data Center GPU Manager，用于监控和管理数据中心 GPU）默认就会提供 SM 活动数据。

此外，还有更细粒度的指标，例如 SM 占用率（SM Occupancy，在 PyTorch 性能分析器中称为“实际占用率”，Achieved Occupancy），该指标可反映每个 SM 的工作负载饱和度。

::: block-1
但与“尽可能提升 SM 效率”相比，理解这些细粒度指标的难度更高。

如果你想深入了解相关知识，建议参考PyTorch 性能分析器博客[18]、DCGM 文档[19]、Nsight 内核分析指南[20]（Nsight 是英伟达提供的 GPU 开发与分析工具套件）以及Nsight 文档[21]。
:::

参考资料

[1] https://www.trainy.ai/: *https://www.trainy.ai/*

[2] PaLM: Scaling Language Modeling with Pathways: *https://arxiv.org/pdf/2204.02311*

[3] PaLM: Scaling Language Modeling with Pathways: *https://arxiv.org/pdf/2204.02311*

[4] llm-foundry/scripts/train/benchmarking: *https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking*

[5] management-library-nvml: *https://developer.nvidia.com/management-library-nvml*

[6] Datadog’s NVML docs: *https://docs.datadoghq.com/integrations/nvml/#metrics*

[7] USE 方法论: *https://www.brendangregg.com/usemethod.html*

[8] Dao-AILab/flash-attention: *https://github.com/Dao-AILab/flash-attention*

[9] 什么是流式多处理器？: *https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor*

[10] Nvidia Hopper Architecture in Depth: *https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/*

[11] nvidia-smi: *https://docs.nvidia.com/deploy/nvidia-smi/*

[12] triton-lang.org/main/getting-started/tutorials/02-fused-softmax: *https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html*

[13] Dao-AILab/flash-attention: *https://github.com/Dao-AILab/flash-attention*

[14] fused_dense.py: *https://github.com/Dao-AILab/flash-attention/blob/9a11f440d3a34f618b4ba814c825b109c6d7e8f5/flash_attn/ops/fused_dense.py#L531*

[15] fused_dense.py: *https://github.com/Dao-AILab/flash-attention/blob/9a11f440d3a34f618b4ba814c825b109c6d7e8f5/flash_attn/ops/fused_dense.py#L531*

[16] dev-discuss.pytorch.org/t/torch-compile-fsdp-dec-8th: *https://dev-discuss.pytorch.org/t/torch-compile-fsdp-dec-8th/1718*

[17] dcgm/latest/user-guide/feature-overview.html#profiling-metrics: *https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics*

[18] pytorch.org/blog/pytorch-profiler-1.9-released: *https://pytorch.org/blog/pytorch-profiler-1.9-released/#gpu-metric-on-timeline*

[19] nvidia.com/datacenter/dcgm: *https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics*

[20] nsight-compute/ProfilingGuide: *https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html*

[21] nsight-compute/ProfilingGuide: *https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html*
