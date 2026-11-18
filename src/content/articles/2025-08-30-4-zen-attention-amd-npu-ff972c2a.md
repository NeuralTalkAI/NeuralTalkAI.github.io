关键词：神经网络处理单元（NPU）、注意力机制（attention）、energy efficiency、memory bound、compiler framework、dynamic attention folding

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/01.webp)

- Zen-Attention: A Compiler Framework for Dynamic Attention Folding on AMD NPUs
- https://arxiv.org/abs/2508.17593
- 本文 8973 字，阅读需 29 分钟，播客 8 分钟

相关推荐

- [吞吐量提高 70%、延迟降 53%！硬件高效注意力：解锁解码速度与性能极限！比 FlashMLA 快 2 倍！TTFT减少94%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889644&idx=1&sn=83b36094e969eec3586c8a965f8c3714&scene=21#wechat_redirect)
- [低功耗高性能！TeLLMe：首个 FPGA 三值 LLM 加速器！三值矩阵乘法查找表和预填充注意力协同优化计算、内存和调度！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888617&idx=1&sn=4ec623ed4d887d2cb7eebf5003057441&scene=21#wechat_redirect)
- [FlashInfer：面向 LLM 服务的可定制且高效的 GPU 注意力引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888144&idx=1&sn=0b3de7ba3a4d204ad4e017c19e2aed69&scene=21#wechat_redirect)

> 基于 Transformer 的深度学习模型正越来越多地部署在笔记本电脑、游戏主机等能量和 DRAM 带宽受限的设备上，这为满足模型的延迟要求带来了重大挑战。

**业界正转向神经网络处理单元** （NPU，一种专门用于加速神经网络计算的硬件单元），以**实现更优异的每瓦性能** （perf/watt，衡量硬件在消耗单位功率时所能提供的性能指标）。然而，:mdnice-bold-italic[将动态注意力层高效映射到 NPU 上仍然是一项艰巨的任务。]

::: block-1
为了优化每瓦性能，AMD XDNA NPU 采用**软件管理的缓存，并与主机共享系统内存**。这需要大量的工程工作来实现**高效的分块** （tiling）、**缓冲区分配** 以及数据在**互连结构** 上的传输，从而充分挖掘设备的最大效率。

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/02.webp)

::: image-caption
图 1 （a）未折叠的多头注意力（MHA）与（b）折叠的多头注意力（MHA）的实现。
:::

本文介绍了 :mdnice-bold-italic[Zen-Attention，这是一个用于优化模型注意力层中 DRAM 带宽利用率的框架]。该框架通过系统性地探索:mdnice-bold-italic[层折叠]、:mdnice-bold-italic[分块以及数据在互连结构] 上的传输等复杂设计空间，并:mdnice-bold-italic[结合张量布局]，最终得出最优解决方案。
:::

我们的评估包括对**端到端模型延迟** 以及每个模型中**特定注意力层** 延迟的对比分析。我们通过改变输入维度和掩码操作，展示了该框架如何增强映射能力。

对于典型的 Transformer 模型，**与基准的非折叠（Unfolded）方法相比**，Zen-Attention 框架实现了:mdnice-bold-italic[注意力块延迟最高 4 倍的提升]，以及:mdnice-bold-italic[端到端网络延迟最高 32%的提升。]

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/03.webp)

## 关键问题

### 问题 1. 在 Zen-Attention 框架中，折叠层级（folding_level）的判定高度依赖 L1 内存容量，而不同型号 AMD NPU 的 L1/L2 缓存规格存在差异（如未来可能推出更大或更小缓存的 NPU），当前框架的硬件感知图优化与分块逻辑是否具备动态适配不同缓存规格的能力？若需适配新硬件，是否需重构核心算法，其迁移成本与兼容性如何保障？

Zen-Attention 框架当前的硬件感知设计具备**基础适配潜力**，但动态性与迁移成本需结合架构细节分两方面分析：

::: block-1
1. **适配能力的底层支撑**：:mdnice-bold-italic[框架的核心模块（如 Tiler、硬件感知图优化）已将“硬件参数”作为输入变量] ——例如 Tiler 明确以“L1 大小”“张量维度”为输入计算子体积（$S_q/S_k$ 等），硬件感知图优化也通过调用 Tiler 返回的“folding_level”动态判断折叠层数，而非硬编码缓存规格。**这种“参数化设计”为适配不同缓存规格 NPU 提供了基础**：若新 NPU 仅调整 L1/L2 容量（如 L1 从 64KB 增至 128KB），理论上只需更新输入的“缓存大小参数”，即可让 Tiler 重新计算适配的分块方案，无需修改核心逻辑。
2. **迁移成本的边界条件**：框架的兼容性依赖“硬件架构一致性”。论文中 AMD XDNA NPU 的核心特性（2D 核心阵列、L1/L2 为软件管理 scratchpad、级联流支持空间归约、DMA 最小 stride 为 4B）是框架设计的前提。:mdnice-bold-italic[若未来新 NPU 突破这一架构范式] （如改为 3D 核心布局、L2 转为硬件缓存、DMA stride 规格变化），:mdnice-bold-italic[则需重构关键模块]：例如，若 L2 不再支持软件显式管理，“L2 层 DMA 块转置”机制将失效，需重新设计转置流程；:mdnice-bold-italic[若核心互连方式改变，“跨列空间归约”的分块展开策略也需调整]。此时迁移成本较高，但**论文未提及针对“架构颠覆性变化”的兼容预案**，推测当前框架的适配性仍局限于“XDNA 架构家族内的缓存规格调整”。
:::

### 问题 2. 实验结果显示，Zen-Attention 对大序列/上下文长度模型的注意力块 latency 提升显著（最高 4 倍），但对小维度模型仅提升~8%且易进入计算受限状态。针对小维度场景，框架是否存在进一步优化空间（如调整分块粒度、复用计算资源等）？若无法突破计算受限瓶颈，该框架在轻量级 Transformer 模型（如边缘设备常用的小参数量模型）中的实用价值是否会大幅受限？

论文虽未直接提及小维度场景的专项优化，但结合框架设计逻辑与 NPU 特性，可明确**存在优化空间，且轻量级模型中的实用价值未被完全限制**：

::: block-1
1. **小维度场景的潜在优化方向**：
2. **轻量级模型中的实用价值**：即使小维度场景 latency 提升有限（仅~8%），框架的“低 DRAM 带宽占用”特性仍为轻量级模型（边缘设备常用）提供核心价值。论文明确提到“即使计算受限，折叠注意力仍降低 DRAM 带宽 utilization”——边缘设备通常存在“多任务并发”场景（如同时运行模型推理与设备监控），DRAM 带宽是共享瓶颈。此时，Zen-Attention :mdnice-bold-italic[虽未大幅加速单模型 latency，但可减少对带宽的占用，避免模型推理与其他任务争抢资源导致的整体性能下降，这对边缘设备的“系统级效率”至关重要，因此实用价值未被完全限制。]
:::

### 问题 3. 框架通过“Folding-Preserving Transpose”机制将转置融入折叠流程，依赖 8×8 块粒度与 SHUFFLE 寄存器指令，而不同注意力架构（MHA/MQA/GQA）的张量维度与转置需求存在差异（如 MQA 共享键值头导致转置规模变化）。这种转置优化方案是否能适配所有三类注意力架构？当遇到非 8×8 整数倍的张量维度时，是否会因块拆分引入额外开销，反而抵消折叠带来的性能增益？

关于 Folding-Preserving Transpose 对三类注意力架构的适配性与非 8×8 维度开销。结合论文中“转置机制设计”与“注意力架构适配细节”，可从适配性与开销两方面来看：

::: block-1
1. **对 MHA/MQA/GQA 的适配性**：该转置机制**具备跨架构适配能力**，核心原因是:mdnice-bold-italic[其优化逻辑与“注意力架构的键值共享方式无关，仅依赖张量维度与 DMA/核指令特性”]：
2. **非 8×8 整数倍维度的开销问题**：作者通过“Padding 机制”间接解决了这一问题，无需额外引入开销：
:::

## 本文目录

- 关键问题
  - 问题 1. 在 Zen-Attention 框架中，折叠层级（folding_level）的判定高度依赖 L1 内存容量，而不同型号 AMD NPU 的 L1/L2 缓存规格存在差异（如未来可能推出更大或更小缓存的 NPU），当前框架的硬件感知图优化与分块逻辑是否具备动态适配不同缓存规格的能力？若需适配新硬件，是否需重构核心算法，其迁移成本与兼容性如何保障？
  - 问题 2. 实验结果显示，Zen-Attention 对大序列/上下文长度模型的注意力块 latency 提升显著（最高 4 倍），但对小维度模型仅提升~8%且易进入计算受限状态。针对小维度场景，框架是否存在进一步优化空间（如调整分块粒度、复用计算资源等）？若无法突破计算受限瓶颈，该框架在轻量级 Transformer 模型（如边缘设备常用的小参数量模型）中的实用价值是否会大幅受限？
  - 问题 3. 框架通过“Folding-Preserving Transpose”机制将转置融入折叠流程，依赖 8×8 块粒度与 SHUFFLE 寄存器指令，而不同注意力架构（MHA/MQA/GQA）的张量维度与转置需求存在差异（如 MQA 共享键值头导致转置规模变化）。这种转置优化方案是否能适配所有三类注意力架构？当遇到非 8×8 整数倍的张量维度时，是否会因块拆分引入额外开销，反而抵消折叠带来的性能增益？
- 一、引言
  - 1.1 注意力机制的特点
  - 1.2 AMD XDNA™ NPU 的特点
  - 1.3 注意力层映射至 NPU 的核心方法：逐层执行与层折叠优化，及带宽敏感问题
- 二、多头注意力机制（MHA）面临的问题
  - 2.1 NPU 现有计算分工的局限与注意力层内存瓶颈
  - 2.2 维度不匹配的填充开销与内核折叠方案的提出
  - 2.3 内核折叠方案的优势与局限：L1 容量约束及 NPU 跨核心优化
  - 2.4 现代语言模型的三类注意力架构及其内存特性
  - 2.5 新型编译器框架的提出及核心贡献
- 三、Zen-Attention 框架
  - A. 硬件感知图优化
  - B. 分块（Tiling）
  - C. 转置（Transpose）
  - D. 填充（Padding）
- 四、实验结果与分析
- 五、结论
- 参考文献

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/04.webp)

## 一、引言

注意力机制作为现代 Transformer 架构的核心，其计算特性本质上具有内存受限的特点。这一现象已通过性能上限图（roofline plots）得到了充分证明，该图将深度神经网络操作划分为两个主要的计算范畴：计算受限操作和内存受限操作。

### 1.1 注意力机制的特点

:mdnice-bold-italic[由于运算强度较低[6]，在不同的硬件架构中，注意力机制始终属于内存受限类别]。这种分类源于多个关键因素：

::: block-1
- 大规模的激活张量（activation tensors，神经网络中经过激活函数处理后的数据张量）
- 归约操作（reduction operations，一种对张量进行压缩计算的操作，如求和、求平均等）
- 元素级操作（element-wise operations，对张量中每个元素单独进行的操作）
:::

上面三类操作的普遍存在，注意力块的内存受限特性给性能带来了重大挑战，因为**这些操作的性能受限于数据传输而非计算吞吐量** （computational throughput，计算单元单位时间内完成的计算量）。

> 为了降低延迟，常见的优化主要集中在两种核心技术上：**最大化数据复用** [5,6]以及**重构计算过程以利用吞吐量密集型的内核指令**。

### 1.2 AMD XDNA™ NPU 的特点

**AMD XDNA™ NPU[7]是一种可重构硬件架构**，它为加速机器学习工作负载提供了一种不同于 GPU 的方案，旨在最大限度地降低能量消耗。该架构:mdnice-bold-italic[通过用暂存缓冲区（scratch buffers，一种用于临时存储数据的高速缓冲区）替代传统的缓存机制]，为程序员提供了对内存层次结构的显式控制，与传统处理器相比，能耗显著降低——传统处理器需要持续执行高能耗的缓存行填充、关联查找（associative searches，在缓存中查找目标数据的操作）、替换策略（replacement policies，当缓存空间不足时决定替换哪些缓存数据的策略）以及推测性预取（speculative fetching，提前从内存中加载可能会被使用的数据到缓存中的操作）。

**但这种方式也带来了一种权衡：内存优化成为了程序员的责任**，因为:mdnice-bold-italic[开发人员必须显式地管理芯片内内存和芯片外内存之间的数据传输模式]，以实现最佳性能。尽管这种方法对编程要求更高，但它能够实现对数据传输和能量使用的精确控制，这使得 NPU 在延迟和能效至关重要的应用场景中具有吸引力。其内存层次结构包含三个级别：L1、L2，而 L3 则是主机内存子系统。

::: block-1
- L1 和 L2 用作暂存内存（scratchpad memory，一种高速的片上内存，由程序员显式管理），L3 则用作 DRAM 上的主机内存。
- NPU 与系统中的其他组件（包括 CPU 和 GPU）共享 L3 内存。
- NPU 核心以二维阵列的形式进行空间布局，每一列共享一个 L2 内存。
- 核心块通过互连网络与其他块进行通信。每个块都配备有一个直接内存访问引擎，以实现流到内存（stream-to-memory）和内存到流（memory-to-stream）的数据传输。
- 专用的级联流（cascade streams，用于在核心阵列内部传输数据的专用数据流）连接阵列中的各个核心，以支持跨核心的空间归约（spatial reduction，在多个核心上分布式地执行归约操作）。
- 垫片块（shim-tiles，用于连接 NPU 与外部内存的接口块）具有访问 DRAM 或主机内存子系统最后一级缓存的端口。
:::

### 1.3 注意力层映射至 NPU 的核心方法：逐层执行与层折叠优化，及带宽敏感问题

**将注意力层映射到 NPU 架构上最常用的方法是采用逐层策略**。为每个层如矩阵乘法、 SoftMax、加法开发优化的内核，使它们能够独立执行——从主机内存读取输入数据，并将输出写回主机内存。一个层会被空间分块为子体积（sub-volumes），并映射到特定的计算核心上。

当输出大小超过 NPU 阵列的计算和存储容量时，会通过时间迭代来处理。该设计受限于 L1 和 L2 内存的容量。层的输出会以子体积的形式发送到 L3（主机 CPU 内存层次结构的一部分）。**更优化的映射方案会将层输出存储在 L1/L2 内存中，并优化层间往返延迟。这种技术被称为层折叠**。层折叠受到多个因素的限制，例如 L1 和 L2 的大小以及中间输出在阵列上的局部性。因此，为了在 NPU 上实现优化的层折叠，一个全面的分块和缓冲区分配框架是必不可少的。

:mdnice-bold-italic[由于 NPU 的内存带宽与计算量之比低于 GPU，因此 NPU 对数据传输非常敏感]。例如，AMD Ryzen™ AI 9 HX 平台为 NPU 分配的带宽仅为 60 GB/s，还不到为 GPU 分配带宽（130 GB/s）的一半。相比之下，独立 GPU 通过专用的高带宽内存以及波前之间较低的上下文切换开销来实现高吞吐量。

NPU 计算核心具有超长指令字流水线，配备有专用的向量和矩阵乘法计算单元。每个块都有一个可显式编程的 DMA 引擎，该引擎具有多个通道，以实现并行性。

## 二、多头注意力机制（MHA）面临的问题

### 2.1 NPU 现有计算分工的局限与注意力层内存瓶颈

当前，神经网络处理器（NPU）相关研究主要聚焦于将高吞吐量密集型的通用矩阵乘法卸载到加速器上执行，而将其余操作留在中央处理器（CPU）上运行。**这种方式会带来显著的 NPU 与 CPU 之间的输入/输出（I/O）开销以及上下文切换开销** [1-4]。

> 现有的逐层执行机制会导致激活数据在动态随机存取存储器中产生过多的往返传输，而这正是 **NPU 上注意力层中内存瓶颈的主要来源。**

此外，注意力层在矩阵乘法操作之前或之后，往往需要进行转置等数据变换，这些变换需要在计算核心内部或 CPU 上进行特殊处理。

### 2.2 维度不匹配的填充开销与内核折叠方案的提出

如果输入和输出的维度无法被计算核心的计算粒度整除，就需要进行大量的填充，此时上述低效问题会呈指数级增长。大多数激活数据变换都需要在 DRAM 中进行往返传输，这不仅效率低下，还会进一步加剧内存瓶颈问题。

**Kao 等人提出通过内核折叠** 来提升查询（Query，Q）与键（Key，K）的乘法（$Q*K$）、softmax 函数、softmax 结果与值（Value，V）的乘法（$SM*V$）等不同独立算子的性能和效率[6]。

### 2.3 内核折叠方案的优势与局限：L1 容量约束及 NPU 跨核心优化

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/02.webp)

::: image-caption
图 1 （a）未折叠的多头注意力（MHA）与（b）折叠的多头注意力（MHA）的实现。
:::

> **如图 1 所示，该方法通过在一级缓存中复用数据，最大限度地减少了每个算子在 DRAM 中的往返延迟**。然而，这种方法受限于执行该内核的每个核心 tile 的一级缓存（L1）容量。

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/05.webp)

::: image-caption
图 2 SoftMax 层中空间归约（Spatial Reduction）的演示。
:::

在 NPU 上，我们可以:mdnice-bold-italic[借助如图 2 所示的跨核心级联流（Cascade Streams，用于在不同计算核心之间传输数据的专用数据流通道），扩大折叠的适用范围。]

### 2.4 现代语言模型的三类注意力架构及其内存特性

此外，现代语言模型采用了三种不同的注意力架构来解决内存瓶颈问题。

::: block-1
- 在多头注意力（MHA）中，每个注意力头（Head，多头注意力机制中用于从不同角度捕捉数据关联的独立计算单元）都维护着独立的查询（Q）、键（K）和值（V）矩阵。其内存需求会随着序列长度（Sequence Length，输入数据中元素序列的长度，如文本中的单词数量）和注意力头数量的增加呈平方级增长。
- 多查询注意力（MQA）通过在多个查询头之间共享单个键-值（K-V）头，降低了内存需求。
- 分组查询注意力（GQA）则是一种折中方案，每个组内部的注意力头共享键-值（K-V）表示，在保证性能具有竞争力的同时，提升了内存效率。
:::

不同类型的注意力块需要不同的内核实现、分块（Tiling）、内存分配以及数据移动编排方案。此外，在部分模型中，注意力块还会引入掩码（Mask）和偏置（Bias），这会进一步增加一级缓存（L1）折叠所需的内存需求。

### 2.5 新型编译器框架的提出及核心贡献

为解决上述问题，我们:mdnice-bold-italic[提出了一种新型编译器框架，该框架将注意力块作为单个折叠层映射到 NPU 阵列上，从而最大限度地减少对 DRAM 的读写操作]。以下是本文的主要贡献：

::: block-1
1. 一种新型的硬件感知图优化启发式算法，用于融合注意力块内的不同层。
2. 一种高效的基于折叠注意力的分块器（Tiler），通过利用一级缓存折叠和空间归约，实现 NPU 的最大利用率。
3. 一种智能处理数据变换的方法，将输入/输出转置、填充和去填充（De-Padding，移除之前为满足计算要求而添加的填充数据）等操作整合到折叠过程中，以减少额外的 DRAM 数据传输量。
:::

## 三、Zen-Attention 框架

### A. 硬件感知图优化

> 在该阶段，我们尝试对计算图进行优化。首先，我们对 ONNX 计算图进行分析，并执行两步注意力折叠和图变换流程。

#### 第一步：折叠

尝试对以下一系列操作进行折叠

$$
A=Q * K^{T} \quad(1)
$$

$$
A=A+B+M \quad(\text{optional}) \quad(2)
$$

$$
S M_{out }=SoftMax(A) \quad(3)
$$

$$
Z=SM_{out} * V \quad(4)
$$

我们**利用表示注意力块中不同算子的特定模式，来匹配并验证输入张量的形状是否能适配一级缓存（L1）内存以进行层折叠**。

::: block-1
- :mdnice-bold-italic[图优化器] 会调用分块器（Tiler），检查一系列算子是否能折叠为单个节点，该节点代表折叠后的注意力操作。
- :mdnice-bold-italic[分块器] 会返回一个折叠级别（folding_level），用于指示注意力块中可折叠的层数。
:::

#### 第二步：批处理注意力头

对不同注意力头的注意力节点进行批处理，生成批处理注意力节点，并将其分类为多头注意力（MHA）、分组查询注意力（GQA）或多查询注意力（MQA）操作。

随后，我们检查是否存在可选的掩码（M）和偏置（B）加法操作，并尝试对这些操作进行优化。在批处理过程中，我们识别出可一起执行的节点，并生成相应的执行序列，以移除计算图中各个注意力头的独立节点。

### B. 分块（Tiling）

> :mdnice-bold-italic[分块器以查询（Q）、键（K）、值（V）、偏置（B）、掩码（M）张量的维度以及一级缓存（L1）容量作为输入]，**寻找能够适配一级缓存（L1）且不违反缓冲区分配约束（由内核规定）的子体积** （subvolumes）${S_q, S_k, S_v, S_b, S_m}$，其中 $S_q$、$S_k$、$S_v$、$S_b$、$S_m$ 分别代表查询、键、值、偏置、掩码张量的子体积大小。

- 分块逻辑首先生成一个有效的分块列表，
- 然后按 $S_q$（查询张量子体积大小）降序对该列表进行排序，以最大限度地增大 $S_q$。
- 之后，分块逻辑会遍历分块列表，直到找到适合适配一级缓存（L1）的子体积。如果键（K）和归约维度被固定在一级缓存（L1）中，那么 $S_q$ 会在同一列的核心之间展开。

类似地，批大小（注意力头数量，Heads）也可以在列之间展开。

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/05.webp)

::: image-caption
图 2 SoftMax 层中空间归约（Spatial Reduction）的演示。
:::

> **利用如图 2 所示的跨列空间归约**，当键（K）、归约维度（r）和掩码（M）进行空间分割（Spatial Split，将数据按空间维度分配到不同的计算核心或 tile 上）时，批大小会按时间维度迭代（Temporal Iteration，按时间顺序依次处理不同批次的数据）。

我们进一步定义了一个名为“折叠级别”的指标，该指标用于确定可折叠的层数。

- 当折叠级别=3 时，表示注意力块中的所有层都已折叠。
- 当折叠级别=2 时，由于一级缓存（L1）容量不足，我们不会融合最后一个矩阵乘法（$SM * V$）操作，但此时的性能仍优于折叠级别=1（代表未折叠注意力）的情况。

### C. 转置（Transpose）

> 在主机或 NPU 上，转置被视为一个独立的算子，但其在计算和 DRAM 访问方面的开销较高。此外，当转置算子与注意力块折叠在一起时，需要额外的一级缓存（L1）缓冲区，这会限制长上下文长度下注意力块融合的可行性。

:mdnice-bold-italic[为解决上述问题，我们提出了一种新型的“折叠保留转置（Folding-Preserving Transpose）”机制，该机制在高效处理转置操作的同时，维持折叠的完整性]。该方法结合了两种操作：

::: block-1
- 一是对二级缓存缓冲区数据执行基于直接内存访问的块转置（Block-Transpose）操作；
- 二是利用专用的矩阵乘法内核，对一级缓存缓冲区数据执行块内（Intra-Block）级别的转置操作。
:::

使用直接内存访问进行数据变换时，关键约束在于最小遍历粒度（指 DMA 一次能访问的数据最小单位）。然而，由于 DMA 支持的最小步长（Stride，指访问内存时相邻数据元素之间的地址间隔）为 4 字节，**我们无法直接使用 DMA 对张量进行完整转置。为解决这一问题**，我们:mdnice-bold-italic[首先以 8×8 的粒度对数据进行块转置，然后在执行矩阵乘法之前，利用寄存器级别的 SHUFFLE 内联函数对块内数据进行转置。]

> 这种经过优化的内核被称为“转置矩阵乘法（Transposed-MatMul）”内核。该混合方法无需单独的转置内核和一级缓存（L1）缓冲区，从而实现了接近最优的性能。

### D. 填充（Padding）

在核心上处理分块输入的计算内核，其每个轴都有由向量乘法累加和向量乘法指令的通道宽度规定的最小所需维度。此外，计算密集型的矩阵乘法和 SoftMax 内核中的循环会被展开以实现软件流水线，这对张量维度设置了严格的下限。

在注意力块的输入和输出张量中，出现奇数维度的情况并不少见。**为满足内核要求，需要将张量填充到内核粒度的下一个倍数**。在 AMD-XDNA2 NPU 中，:mdnice-bold-italic[二级缓存（L2）的内存到流直接内存访问通道支持在$\{D0, D1, D2\}$维度（张量的三个维度方向）上进行填充。利用这一特性，可以将非标准形状（如奇数维度）的张量填充到内核粒度的倍数]。

在极少数直接内存访问（DMA）填充不可行的情况下，我们会依赖前一层将相应的输出张量填充到符合要求的维度，从而避免使用专门的填充（Pad）算子。

## 四、实验结果与分析

> 我们使用 5 个模型开展了实验，这些模型代表了近年来带有注意力模块的语言模型。

- 我们在搭载 AMD 锐龙 AI（Ryzen AI）9 HX 370 处理器的设备上对这些模型的层和模型整体进行了验证，该设备配备的 NPU 包含 32 个核心，以 4×8 网格形式排列，可提供 50 TOPS的算力。
- 该 NPU 阵列中，每个 L1 缓存的容量为 64KB，每个 L2 缓存的容量为 512KB。处理器子系统包含 12 个 Zen 5 系列 CPU 核心。**CPU 和 NPU 共享同一 DRAM，该 DRAM 的读写组合带宽约为 60 GB/s。**

与文献[6]的方案不同，我们的框架利用 NPU 架构中计算资源与数据流资源固有的分离特性，采用了注意力层的预编译内核。

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/06.webp)

::: image-caption
表 1：采用 Zen-Attention 框架评估的模型（Models evaluated using Zen-Attention Framework）
:::

> 表 1 展示了这些模型的部分重要特性：

- 每个模型拥有 1 种或 2 种独特的注意力模块形状；
- 部分模型需要对输入维度进行填充，以适配内核粒度；
- 所有模型的 K 输入（注意力机制中的“键”矩阵）均需进行转置操作，而 **Zen-Attention 框架会将该操作与注意力模块进行折叠优化。**

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/07.webp)

::: image-caption
图 4 不同网络中折叠式与展开式注意力模块的延迟对比
:::

> 图 4 展示了在不同模型上，采用 Zen-Attention 框架所提优化方案后，折叠式与展开式（unfolded，指注意力模块的各层独立处理、数据需频繁往返主存）注意力模块的延迟提升情况。

从直观上看，**序列长度和上下文长度** （context length，指模型能关注的输入序列范围，对应 K 矩阵的维度）**更大的模型，相比序列长度** （Q，注意力机制中的“查询”矩阵维度）和上下文长度（K）:mdnice-bold-italic[更小的模型，能获得更高的加速比]。

**这一现象与各模型对 DRAM 带宽的需求直接相关**：:mdnice-bold-italic[折叠式注意力通过减少注意力模块中带宽密集型层的 DRAM 往返次数（roundtrip，指数据从 NPU 缓存写入 DRAM 再读回的过程）]，实现了最高约 **4 倍的延迟降低。**

对于维度较小的模型，其性能提升并不显著，这是因为：

::: block-1
- 此时计算时间的占比超过了带宽需求的影响——也就是说，模型的瓶颈从“数据传输速度”转变为“计算速度”。
- 此外，较小的张量形状会导致部分 NPU 核心处于未使用状态，这会降低计算效率，进而使注意力模块进入计算受限状态。
:::

即便在**计算受限场景下，折叠式注意力相比展开式仍能实现约 8%的延迟降低**，同时减少了 DRAM 带宽占用——这对于在内存资源受限的系统上同时运行多个应用的场景十分有利。

![文章插图](/articles/2025-08-30-4-zen-attention-amd-npu-ff972c2a/images/08.webp)

::: image-caption
图 5 不同网络中折叠式与展开式注意力模块的端到端网络延迟对比
:::

> 图 5 展示了采用 Zen-Attention 框架对注意力模块进行映射优化后，模型端到端（end-to-end，指从输入数据到输出结果的完整计算流程）延迟的提升情况。

- 在 ViT-base-patch-16 模型中，折叠式注意力使网络**延迟降低了高达 32%，** 这一结果充分体现了注意力层引发的带宽瓶颈对模型性能的显著影响。
- 而在 BERT 模型中，:mdnice-bold-italic[延迟提升幅度最小，仅为 1.4%——这是因为在该模型中，注意力模块并非性能瓶颈的主要来源]。尽管如此，对于同时运行的其他应用而言，折叠式注意力所带来的 DRAM 带宽占用减少仍能带来性能收益。

## 五、结论

> 本文针对Transformer模型在内存带宽与能耗受限设备（如笔记本、游戏主机）部署时的注意力层性能瓶颈，以及AMD XDNA NPU高效映射动态注意力层的技术挑战，提出了Zen-Attention编译框架，为NPU上注意力层的能效优化提供了系统性解决方案。

通过三大核心创新设计，Zen-Attention有效突破了现有方案的局限：:mdnice-bold-italic[硬件感知图优化启发式方法实现了注意力块内多算子的深度融合]，:mdnice-bold-italic[基于折叠注意力的分块器借助L1缓存折叠与空间归约最大化了NPU计算资源利用率]，而:mdnice-bold-italic[折叠保持转置机制与DMA/L1协同填充策略则规避了传统数据变换带来的额外DRAM通信开销]，三者共同构建了适配AMD NPU架构特性的注意力层优化链路。

在AMD Ryzen AI 9 HX平台上，针对ViT-base-patch、CLIP系列、BERT等5个代表性Transformer模型的实验验证表明，与基线展开方案相比：

::: block-1
- Zen-Attention使:mdnice-bold-italic[注意力块延迟 最高降低4倍]，端到端网络延迟**最高优化32%；**
- 即便在:mdnice-bold-italic[计算受限场景下，仍能实现 8% 的延迟降低与DRAM带宽占用缩减。]
:::

这一成果不仅验证了动态注意力折叠策略在提升NPU能效方面的有效性，也为内存带宽受限设备上Transformer模型的低延迟部署提供了关键技术支撑，对推动端侧AI模型的高效落地具有重要实践意义。

## 参考文献

1. Zhuang, Jinming, et al. "ARIES: An Agile MLIR-Based Compilation Flow for Reconfigurable Devices with AI Engines." Proceedings of ACM/SIGDA International Symposium on Field Programmable Gate Arrays. 2025.
2. Zhuang, Jinming, et al. "Ssr: Spatial sequential hybrid architecture for latency throughput tradeoff in transformer acceleration." Proceedings of the ACM/SIGDA International Symposium on Field Programmable Gate Arrays. 2024.
3. Rösti, André, and Michael Franz. "Unlocking the AMD neural processing unit for ML training on the client using bare-metal-programming tools." 2025 IEEE 33rd Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM). IEEE, 2025.
4. Taka, Endri, et al. "Maxeva: Maximizing the efficiency of matrix multiplication on versal ai engine." 2023 International Conference on Field Programmable Technology (ICFPT). IEEE, 2023.
5. Ivanov, Andrei, et al. "Data movement is all you need: A case study on optimizing transformers." Proceedings of Machine Learning and Systems 3 (2021): 711-732.
6. Kao, Sheng-Chun, et al. "Flat: An optimized dataflow for mitigating attention bottlenecks." Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 2023.
7. Rico, Alejandro, Satyaprakash Pareek, Javier Cabezas, David Clarke, Baris Ozgul, Francisco Barat, Yao Fu et al. "AMD XDNA™ NPU in Ryzen™ AI processors." IEEE Micro 2024.
