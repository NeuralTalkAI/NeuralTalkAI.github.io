关键词：**晶圆级系统**、大语言模型推理、**FlatAttention 数据流**、:mdnice-bold-italic[片上集合通信]、软硬协同设计

> 在当前大语言模型（LLM）推理的演进路线中，随着序列长度的暴增和混合专家架构（MoE）的普及，Attention（注意力机制）算子正成为制约系统整体吞吐量的核心瓶颈。传统的优化手段往往局限于单一维度的算法改进或纯硬件的带宽升级。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/01.webp)

::: image-caption
图 1 (a) 大语言模型在预填充阶段（序列长度）和解码阶段（键值缓存长度）的浮点运算量占比分布。显示 MoE 模型中注意力的浮点运算占比随模型规模飙升，DeepSeek-v3-671B 解码阶段占比达 71%，印证了 “注意力瓶颈” 的存在。(b) 英伟达 GH200 图形处理器上 FlashAttention-3 预填充和 FlashMLA 解码性能的性能上限图，基于十六位浮点精度评估，预填充阶段改变头维度和序列长度，解码阶段改变推测长度和键值缓存长度，表明 GH200 上的主流注意力实现离硬件性能上限差距极大，利用率仅 26%-64%，说明 GPU 架构的注意力优化已达瓶颈。这两点共同证明，针对 LLM 推理的注意力瓶颈，需要全新的 tile 架构与专用数据流协同设计，为 FlatAttention 的研究提供了现实依据。
:::

然而，最新的一项研究为我们揭示了一个硬核的真相：:mdnice-bold-italic[“通过在基于 Tile（平铺）的多 PE（处理单元）加速器上深度协同优化数据流（Dataflow）与 Fabric 互联集合通信（Collectives），我们可以突破单点优化的极限。”]

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/02.webp)

- **FlatAttention: Dataflow and Fabric Collectives Co-Optimization for Large Attention-Based Model Inference on Tile-Based Accelerators**
- https://arxiv.org/pdf/2604.02110
- 7000 字，阅读 28 分钟，**播客 20 分钟**

相关推荐

- [解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)
- [注意力块 4 倍加速！Zen-Attention：面向AMD NPU基于硬件感知的动态注意力层折叠编译框架，端到端32%提速!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891471&idx=1&sn=7b3c57d36ba86b27a277a997b771478f&scene=21#wechat_redirect)
- [MICRO'25 StreamTensor：迭代张量驱动的 LLM 数据流加速器，从PyTorch到FPGA全流程优化实现 0.76x FPGA 延迟与 1.99x GPU 能效突破](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896673&idx=1&sn=c4bb2d42f2cdefa9f557bb4d2ddfa208&scene=21#wechat_redirect)

> 该研究提出了一种名为 FlatAttention 的全新架构数据流范式。它不仅在计算密集型和访存密集型场景下分别实现了惊人的 **86% 和 78% 的平均资源利用率**，更在基准测试中，:mdnice-bold-italic[以平均 1.9 倍的性能加速比，全面超越了在 NVIDIA 顶级 GH200 GPU 上经过极致优化的 FlashAttention 和 FlashMLA 算子实现。]

更令人振奋的是，当我们将目光投向拥有 64 个加速芯片（单芯片算力达 1976 TFLOPS FP8）晶圆级多裸片系统（Wafer-Scale Multi-Die System）时，FlatAttention 助力 DeepSeek-v3-671B 模型在端到端解码推理中，**实现了比现有的最佳商用系统（由 96 张 H800 组成的 DS-Prof 集群）高出 2.9 倍的系统吞吐量**，同时将每个输出 Token 的时间（TPOT）降低了 1.4 倍。

::: block-1
本文将拆解 FlatAttention 的核心创新机制，探讨其如何通过底层数据流与片上网络（NoC）集合通信的深度绑定，重塑未来大模型推理的硬件基础设施。
:::

## 本文目录

- 一、核心痛点：为什么传统的 Attention 优化在多 Tile 架构上失效？
- 二、 核心创新：FlatAttention 的软硬协同艺术
  - 2.1 极致的资源利用与 Tiling 策略
  - 2.2 全面超越单芯片极限 (vs. GH200)
- 三、 晶圆级系统上的端到端性能：DeepSeek-v3 解码推理实战
  - 3.1 晶圆级巨兽的架构配置
  - 3.2 FlatAttention 对吞吐量与延迟的双重优化
  - 3.3 直面通信墙：专家并行与拓扑瓶颈
- 四、 对标行业标杆：DeepSeek-v3 解码性能的全面胜利
- 五、 相关工作：我们在架构演进史上的坐标系
- 结语与展望

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、核心痛点：为什么传统的 Attention 优化在多 Tile 架构上失效？

> 在探讨 FlatAttention 之前，我们必须理解当前架构面临的根本困境。

**大量的早期研究致力于开发优化的 Attention 数据流，其核心目标是消除 Softmax 归一化作为独立阶段带来的高昂片外内存传输开销**。业界普遍采用的解法是进行“算子融合”（Operator Fusion），将 Attention 核心内的基本操作融合，从而确保中间张量始终驻留在片上（On-chip）内存中。

::: block-1
然而，**以往的工作主要聚焦于单 Tile（或 NVIDIA 架构中的单 SM）加速器**。以著名的 FlashAttention-2 为例，它通过优化的切块（Tiling）和 SM 间的工作划分，在 GPU 上实现了业界领先的融合 Attention 数据流。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/04.webp)

::: image-caption
算法 1：FlashAttention-2 算法。该算法是 GPU 上经典的注意力数据流实现，通过分块加载 K、V 矩阵至片上 L1，融合矩阵乘与 Softmax 计算以提升数据复用，减少 HBM 访问。但它仅在单个计算单元（SM）内做优化，无跨单元通信设计，移植到 tile 架构时，各 tile 独立处理数据，完全丧失跨 tile 数据复用可能，且分块大小受单 tile L1 容量限制，无法进一步降低 IO 复杂度。这一缺陷直接凸显了 tile 架构下设计跨 tile 协作数据流的必要性，也是 FlatAttention 提出的核心动因之一。
:::

但是，:mdnice-bold-italic[在 Hopper 之前的经典 GPU 架构中，SM 之间的通信必须通过全局内存（Global Memory）进行，这严重阻碍了跨 SM 的层融合，使得片上 SM 到 SM 的通信成本极其高昂]。因此，**FlashAttention-2 被迫采用了一种“极其并行”（Embarrassingly-parallel）的数据流策略**，刻意避免这种高成本的通信。
:::

- **直到最新的 Hopper 架构，NVIDIA 才在 GPU 处理集群（GPC）内部引入了高效的 SM 到 SM 通信机制**，这为 GPC 内的 SM 之间进行高效的片上集合通信（Collective Communication）操作创造了条件。
- :mdnice-bold-italic[尽管近期有研究尝试利用 Hopper 的片上集合通信来扩大 LLM 工作负载的层融合范围，如 FlashFuser，但其局限于 GEMM 链和 FFN（前馈神经网络）层，并未触及复杂的 Attention 算子。]

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/05.webp)

::: image-caption
图 2：(a) tile 式多处理单元架构模板。(b) 依托片上架构支持的硬件集体通信实现的行式多播，与两种基于软件的集体通信实现（软件树型、软件串行）的对比。(c) 由多个 tile 式多处理单元加速器组成的晶圆级多芯系统，采用二维网状的片间互连拓扑。整体来说，该图清晰展示了本文的硬件研究基底，图 2a 的 tile 架构模板整合了计算、存储、通信单元，是大模型推理加速器的主流设计；图 2b 证明硬件集体通信的行式多播远优于软件实现，为 FlatAttention 采用硬件原语提供了性能依据；图 2c 的晶圆级多芯系统则适配了千亿参数大模型的内存和计算需求，解决了单芯资源受限问题。这三张子图共同构建了 FlatAttention 的硬件应用场景，也解释了为何集体通信原语是 tile 架构优化的关键。
:::

::: block-1
在多 Tile 架构以及更宏大的晶圆级系统中，如果缺乏数据流与硬件底层集合通信的深度协同，简单的算子融合将不可避免地撞上通信延迟和带宽利用率的“互联墙”。
:::

## 二、 核心创新：FlatAttention 的软硬协同艺术

针对上述挑战，本文提出的 FlatAttention 是一种与底层 Fabric 级集合通信原语深度协同设计的数据流，**专门用于在基于 Tile 的多 PE 加速器上高效加速 LLM 中的现代 Attention 变体**，这包括 MHA、GQA 和 MLA。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/06.webp)

::: image-caption
图 3：(a) 当前主流大语言模型架构概述，以及 (b) 预填充阶段的多头注意力、(c) 自回归解码阶段的多头注意力、(d) 自回归解码阶段的分组查询注意力的原理示意图。整体清晰展示了 MHA 在预填充和解码阶段的计算形态差异，以及 GQA 通过分组共享 KV 缓存将解码阶段的矩阵 - 向量乘还原为矩阵乘的设计。这一特性是 FlatAttention 泛化到各注意力变体的关键依据，本文正是基于不同注意力变体的计算形态差异，设计了适配的 tile 分组和分块策略，比如 GQA 解码阶段采用更长的查询序列分组。同时，图 3a 的 MoE+MLA 架构也解释了 DeepSeek-v3 成为本文端到端验证案例的原因。
:::

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/07.webp)

::: image-caption
图 4：(a) FlatAttention 的参数定义。(b) 详细的 FlatAttention 数据流。(c) 朴素的 FlatAttention 调度方式。(d) 优化后的异步 FlatAttention 调度方式。该图是 FlatAttention 数据流的核心可视化，图 4a 的组参数定义为多 tile 协同处理奠定了基础，图 4b 拆解了其 “加载 - 多播 - 计算 - 归约 - 存储” 的核心流程，体现了跨 tile 数据复用的设计核心。图 4c 和 4d 的对比则凸显了异步调度的价值，朴素调度中数据移动和 Softmax 计算会闲置矩阵引擎，而异步调度通过双任务并发让矩阵引擎近乎满负载运行。这一优化是 FlatAttention 实现 92.3% 高利用率的关键，也印证了数据流调度与硬件单元特性协同的重要性。
:::

### 2.1 极致的资源利用与 Tiling 策略

> FlatAttention 在资源调度上的精确度令人叹为观止。

在保证 L1 内存使用量严格控制在 384 KiB 预算内的同时（如系统配置表所规定），该策略首要目标是最大化底层矩阵乘法引擎（RedMulE）的利用率，目标设定为超越 95%。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/08.webp)

::: image-caption
图 11：(a) RedMulE 引擎利用率与分块大小的关系。(b) 异步 FlatAttention 数据流的 L1 本地内存占用与分块大小的关系。该图为 FlatAttention 的最优单 tile 分块大小提供了量化依据，图 11a 显示分块大小为 128 时矩阵引擎利用率超 95%，继续增大分块利用率无明显提升；图 11b 显示该分块大小的 L1 占用未超出 384KiB 的硬件预算，是兼顾计算利用率和存储资源的最优解。这一结果确定了 FlatAttention 的基础分块参数，为后续组规模的缩放提供了基准，也证明了数据流设计需要严格匹配硬件的存储和计算资源特性，才能实现最优性能。
:::

::: block-1
:mdnice-bold-italic[研究团队经过严密的空间探索发现，对于所评估的基于 Tile 的多 PE 配置，最优的 Tiling 和组缩放（Group-scaling）策略配置为 $B_r / G_y = B_c / G_x = 128$。 这一黄金配置使得系统在不超出 L1 容量限制的前提下，将 RedMulE 的利用率推向了惊人的 98% 峰值。]
:::

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/09.webp)

::: image-caption
图 10：FlatAttention 的通用分块和组规模缩放策略。该图是本文从实验中提炼的 FlatAttention 核心调优方法，核心原则是先保证单 tile 的矩阵引擎高利用率，再根据注意力分数矩阵形态和 tile 拓扑增大组规模。这一策略解决了 “过度扁平化” 问题，为 FlatAttention 泛化到 MHA、GQA、MLA 等不同注意力变体提供了统一的调优框架。通过该策略，FlatAttention 能根据不同注意力变体的计算形态，灵活调整 tile 组的行列规模，最大化跨 tile 数据复用的同时，避免计算单元利用率损失，是其实现多变体通用优化的关键。
:::

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/10.webp)

::: image-caption
算法 2：基于 tile 式多处理单元架构的 FlatAttention 算法。该算法是本文核心的注意力数据流实现，依托 tile 架构的片上网络集体原语设计，将多 tile 划分为组协同处理注意力计算。算法中仅对角 tile 从 HBM 加载数据并通过组内多播分发，结合归约操作完成全局 Softmax 计算，大幅降低 HBM 访问。其设计突破了单 tile 的资源限制，还支持异步调度让矩阵计算与数据移动、Softmax 计算重叠。这一设计完美适配 tile 架构的通信特性，是 FlatAttention 实现高利用率和低 HBM 流量的算法层核心保障。
:::

### 2.2 全面超越单芯片极限 (vs. GH200)

> 为了验证其实际效能，研究人员在多种输入形状下，对 Prefill 阶段的 MHA、Decode 阶段的 MHA、GQA 以及 MLA 进行了详尽评估。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/11.webp)

::: image-caption
表 1：系统规格参数。该表格是 FlatAttention 性能验证的核心硬件基准，其 32×32 tile 架构的峰值 FP16 性能与 NVIDIA GH200 持平，保障了对比的公平性。tile 内整合矩阵、向量双引擎，兼顾注意力计算的密集矩阵乘和 Softmax 的向量操作，384KiB 本地内存为片上数据缓冲提供支撑，HBM4 的 2TB/s 带宽匹配计算单元的数据流需求。该配置既体现了 tile 架构的计算 - 存储 - 通信协同设计，也为 FlatAttention 发挥集体原语优势提供了硬件基础。
:::

硬件配置方面，加速器在南侧边缘部署了两个 HBM4 堆栈，提供高达 4 TB/s 的带宽，这在峰值 FP16 算力和片外带宽上直接对标了顶级的 NVIDIA GH200 GPU。

> :mdnice-bold-italic[在与 GH200 上最先进的注意力实现（针对 MHA 和 GQA 的 FlashAttention，以及针对 MLA 的 FlashMLA）的正面交锋中， FlatAttention 展现出了统治级的表现：]

- 计算受限场景 (Compute-bound)： 实现平均 86% 的计算资源利用率。
- 访存受限场景 (Memory-bound)： 实现平均 78% 的 HBM 带宽利用率。

如作者在评估数据总结所示，在绝大多数评估场景下，**FlatAttention 在基于 Tile 的多 PE 加速器上的表现全面碾压 GH200 上的优化实现，平均性能加速比达到 1.9 倍。**

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/12.webp)

::: image-caption
图 12： tile 式多处理单元加速器上的 FlatAttention 与英伟达 GH200 图形处理器上优化的注意力数据流的基准对比，预填充阶段改变头维度和序列长度，解码阶段改变推测长度和键值缓存长度。标注 C:x% 表示计算受限的核（利用率 x%），标注 M:y% 表示内存受限的核（高带宽内存带宽利用率 y%）。该图验证了 FlatAttention 在各注意力变体和不同计算场景下的通用优势，无论是计算受限还是内存受限场景，FlatAttention 均实现了更高的硬件利用率，计算受限场景平均利用率 86%，内存受限场景 HBM 带宽平均利用率 78%。对比 GH200 上的优化实现，FlatAttention 平均实现 1.9 倍的速度提升，且这一优势覆盖预填充、解码阶段以及 MHA、GQA、MLA 全变体。这一结果证明 FlatAttention 的设计能适配不同注意力变体的计算特性，具有良好的通用性。
:::

::: block-1
更重要的是，FlatAttention **不仅是一个优化的结果，它还可以作为一种强大的反馈机制，用于指导和反向优化基于 Tile 的多 PE 加速器的架构设计参数**，实现了真正的“软硬共同演进”。
:::

## 三、 晶圆级系统上的端到端性能：DeepSeek-v3 解码推理实战

单芯片的胜利只是开始，大模型真正的战场在于分布式集群。研究团队将目光转向了在晶圆级多裸片系统上进行 DeepSeek-v3 的端到端解码推理性能分析。

### 3.1 晶圆级巨兽的架构配置

> 该评估系统通过晶圆级中介层（Interposer），以 8 × 8 的 Mesh 拓扑结构互联了 64 个基于 Tile 的多 PE 加速器。节点间通过 D2D（Die-to-Die）链路连接，提供 1 TB/s 的超高带宽和仅 256 ns 的极低延迟。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/13.webp)

::: image-caption
图 5(a) tile 式加速器上用于通用矩阵乘的 SUMMA 数据流。DeepSeek-v3 在 (b) 流水线并行、(c) 全专家并行、(d) 专家 - 流水线混合并行下的工作负载分布。(e) 晶圆级多芯系统的执行模式。该图为大模型在晶圆级系统的并行部署提供了设计依据，图 5a 的 SUMMA 数据流为 FlatAttention 之外的 GEMM 计算提供了高效实现，保障了解码阶段其他核的性能；图 5b-d 对比了三种并行策略的负载分布特性，为本文选择 EP32-PP2 混合并行提供了参考；图 5e 的执行模式则明确了核执行与片间通信的同步方式。本文基于这些设计，实现了 DeepSeek-v3-671B 在 64 tile 系统的高效部署，证明了并行策略与数据流优化的协同性。
:::

由于 DeepSeek-v3 推理完全在 FP8 精度下运行，所有加速器均配置为相同的峰值 FP8 吞吐量（未开启稀疏性时为 1976 TFLOPS）以及 4 TB/s 的 HBM 带宽。在 RedMulE 矩阵引擎中，FP8 的峰值吞吐量与 FP16 完美匹配。

为了达成这一算力目标，加速器的工作频率被提升至 1.9 GHz，这与 NVIDIA GH200 的频率处于同一梯队。每个加速器配备 128 GiB 的 HBM 容量，这使得包含 MTP 特性的 DeepSeek-v3-671B 庞大解码推理任务能够完整部署在这 64 芯片的晶圆级系统上。

### 3.2 FlatAttention 对吞吐量与延迟的双重优化

> 在系统级并行策略上，采用了 EP32–PP2（专家并行 32，流水线并行 2）的配置。我们重点观察不同数据流对解码性能的影响。

通过改变每个加速器批处理的用户 Token 请求数（Batch Size），我们得到了系统级吞吐量与 TPOT（每个输出 Token 的时间）的关系曲线。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/14.webp)

::: image-caption
图 13：(a) 64 加速器系统上 DeepSeek-v3-671B 解码性能对比，分别采用 FlatAttention 和 FlashMLA 数据流。(b) 单芯片批处理量为 256 时，单个解码层的运行时间拆解。专家并行的影响：(c) 晶圆级系统的性能表现；(d) 批处理量为 256 时的片间通信开销。该图展示了 FlatAttention 在端到端 LLM 推理中的性能提升，图 13a 和 b 证明 FlatAttention 将注意力计算的占比从 71% 降至 42%，实现 4.5 倍的注意力核加速和 2.1 倍的端到端解码加速；图 13c 和 d 则揭示了专家并行的调优规律，低中批处理量下增大专家并行能提升吞吐量，但高批处理量下片间通信开销会成为瓶颈。这一结果既验证了 FlatAttention 对端到端推理的实际价值，也为晶圆级系统的并行策略选择提供了参考。
:::

> 结论如下：

- 低 Batch 规模下： 系统的 TPOT 达到最小，但整体吞吐量较低。此时 FlatAttention 与 FlashMLA 性能表现相当。
- 中高 Batch 规模下： 随着每芯片 Batch Size 的增加，FlatAttention 展现出惊人的扩展性，其系统吞吐量最高达到 FlashMLA 的 2.1 倍，并且始终保持更低的 TPOT 延迟。

::: block-1
深入到微观层面的运行时间分解（Runtime Breakdown）：在满足 50 ms TPOT 约束的前提下，当每芯片 Batch Size 为 256 时，Attention 计算环节在整个解码时间中占据绝对主导地位。
:::

- 使用 FlashMLA 时，Attention 耗时占比高达 71%，暴露出极低的加速器利用率。
- :mdnice-bold-italic[使用 FlatAttention 时，加速器利用率飙升至 83%，Attention 部分实现了 4.5 倍的加速，使得整体解码流程占比降至 42%， 最终转化为端到端 2.1 倍的惊人加速比。]

### 3.3 直面通信墙：专家并行与拓扑瓶颈

> 研究进一步探讨了专家并行度（Expert Parallelism）的影响。在完全流水线并行的架构下，若处于低 Batch 区域，增加 Batch Size 无法提升整体吞吐量，反而会劣化 TPOT，根本原因在于并非所有的专家节点都被激活。只有当 Batch Size 超过 16 时，所有专家全面唤醒，系统吞吐量才开始爬坡，尽管 TPOT 也会相应上升。

**数据表明，从低到中等规模的Batch范围看，将专家并行度从零拉满，能够同时改善系统吞吐量和 TPOT。**。然而，:mdnice-bold-italic[在极高的 Batch 规模下，芯片到芯片（Chip-to-chip）的互联通信开销成为了统治解码延迟的罪魁祸首]。

特别是考虑到 2D Mesh 互联拓扑中不可避免的多跳（Multi-hop）传输，:mdnice-bold-italic[盲目增加专家并行度只会进一步急剧放大这一通信开销]。**这直接指出了未来的研究方向：探索更为优化且物理上可实现的晶圆级系统互联拓扑结构。**

## 四、 对标行业标杆：DeepSeek-v3 解码性能的全面胜利

> 为了确立 FlatAttention 晶圆级系统的业界地位，研究人员将其与由商用 GPU 和 NPU 构建的顶尖服务系统进行了对标。下表提炼了原论文中的核心对比数据：

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/15.webp)

::: image-caption
表 2：DeepSeek-v3-671B 解码性能与当前主流 GPU / NPU 方案对比。该表格直观验证了 FlatAttention 在晶圆级多芯系统上的实际部署优势，即便本文 64 Tile 加速器的峰值系统性能低于 96 张 H800 的方案，仍实现了更高的单芯片吞吐量和更低的 TPOT。即使将 D2D 链路带宽降至与 NVLink 持平，优的 Tiling 和组缩放本文方案仍有 1.6 倍的解码吞吐量提升，核心原因是 FlatAttention 大幅提升了注意力核的硬件利用率。这一结果证明，数据流与硬件架构的协同优化，比单纯提升峰值性能更能提升 LLM 推理的实际性能。注：以上数据基于满足 50ms TPOT 约束的操作点得出。表中 Ours1 与 Ours2 均代表搭载 FlatAttention 的晶圆级系统。
:::

> 深度数据透视：我们对比表现最强悍的商用系统 DS-Prof——由 96 张 H800 组成，单卡能力与我们的加速器相近。

1. 全速状态（Ours1）： 搭配 1 TB/s D2D 链路的晶圆级系统，实现了高达 2.9 倍的系统吞吐量飞跃，同时将 TPOT 延迟大幅降低了 1.4 倍（从 50.2 ms 降至 35.8 ms）。
2. 降配状态（Ours2）： 即便我们将极其硬核的 1 TB/s D2D 带宽强行降低至 160 GB/s（与 DS-Prof 中的 NVLink 带宽处于同一水平），该晶圆级多裸片系统依然轻松超越 DS-Prof。
3. **核心优势归因**： 必须指出的是，我们的晶圆级互联采用的是 2D Mesh 拓扑，其网络直径更大，并且缺乏计算内核与通信重叠（Kernel-communication overlap）的优化。:mdnice-bold-italic[在面临这些不利物理条件下，系统仍能达成 1.6 倍的解码吞吐量加速，其核心驱动力完全归功于 FlatAttention 数据流带来的 Attention 内核利用率的暴涨。]

::: block-1
放眼整体系统性能，尽管我们的 64 节点系统在峰值理论算力上比 96 节点的 DS-Prof 集群低了 1.5 倍，但**凭借 FlatAttention 的神级优化，其最终的系统级吞吐量依然实现了高达 1.9 倍的逆势反超。**
:::

## 五、 相关工作：我们在架构演进史上的坐标系

> 为了更清晰地呈现 FlatAttention 的破局点，论文对近期的前沿研究进行了全面的梳理与对比。

![文章插图](/articles/2026-04-07-gh200-flatattention-tile-86-78-88dbbb0a/images/16.webp)

::: image-caption
表 3：相关研究对比。该表格通过与现有研究的多维度对比，凸显了 FlatAttention 的创新性和全面性。现有研究要么仅支持单 tile 优化，要么虽支持多 tile 但未采用硬件级的多播和归约原语，WaferLLM 虽用硬件多播但因假设模型全入片上内存而放弃层融合，无法适配大模型。FlatAttention 是首个同时实现层融合、注意力全变体优化、多 tile 支持，并深度结合硬件级集体通信原语的方案，填补了大模型推理中数据流与 tile 架构协同优化的空白。*注：*Wafer-scale 假设，对于完全适配在片上内存的小模型，无需进行层融合。
:::

- **GPU 上的局限突围**： ClusterFusion 虽然利用了 Hopper GPU 上的集合通信实现了优化的全融合 Attention 数据流，且:mdnice-bold-italic[与本文工作有相似之处，但其灵活性欠佳（仅沿一个轴“展平”），且受限于单一 GPC 簇的架构规模。]
- **Tile/NoC 架构的初步探索**：
- **硬件加速集合的缺失**： 绝大多数前沿工作都未评估硬件加速集合通信的使用。:mdnice-bold-italic[唯一的例外是 WaferLLM，它利用了 Cerebras WSE-2 的硬件多播支持。然而，WaferLLM 极度依赖模型能完全存入片上内存的假设，一旦模型过大溢出，其数据流便显得笨拙且次优。]

::: block-1
在此之前，业界没有任何一项工作全面直面并解决了为 LLM 协同设计优化包含集合通信的多 Tile 加速器这一艰巨挑战，而 FlatAttention 补齐了这块拼图。
:::

## 结语与展望

> FlatAttention 不仅仅是一个简单的算法升级，它代表了 AI 芯片基础软件栈从“适应硬件”到“定义硬件通信”的范式转变。

通过将数据流与 Fabric 级集合通信原语的深度绑定，FlatAttention 在 GH200 级别的单节点上证明了 1.9 倍的绝对优势，更在 64 节点的晶圆级巨兽上，以少胜多，**超越了业界顶尖的庞大商用 GPU 集群。**

::: block-1
未来，随着模型参数量的进一步飙升，:mdnice-bold-italic[算力瓶颈将愈发下沉至通信与互联层。我们期待在 FlatAttention 提供的数据流反馈机制下，下一代晶圆级加速器能在 Mesh 以外孵化出更具物理可实现性的终极互联拓扑结构，彻底击碎阻挡 AGI 演进的“互联墙”。]
:::
