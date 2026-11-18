关键词：PipeThreader、sTask-graph、软件定义流水线、异构硬件利用、DNN 执行效率

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/01.webp)

- https://www.usenix.org/conference/osdi25/presentation/cheng
- https://tilelang.com
- https://github.com/tile-ai/tilelang
- 本文 14085 字，阅读需 56 分钟，播客 6 分钟如下

> 为了**【有效利用】** 现代 GPU 中如 TensorCores（张量核，专门用于矩阵乘法的硬件单元）和 Tensor Memory Accelerators（TMA，张量内存加速器，用于高效数据搬运的硬件单元）等**异构专用硬件单元**。

本文提出了 **PipeThreader** （管道线程器），一种:mdnice-bold-italic[全新的 DNN（深度神经网络）编译器]。

::: block-1
PipeThreader 提出:mdnice-bold-italic[将调度功能从硬件转移到软件]，从而以最少的人工干预实现更高效、更复杂的计算流水线。这是通过 **sTask-graph** （专用任务图，一种新型的 DNN 计算抽象）、**一种【分层的】硬件抽象** （用于捕捉专用单元的能力）以及**新的调度原语** 实现的。

因此，PipeThreader 能够为已充分研究的 DNN 架构（如 FlashAttention）发现高效的流水线调度方案，实现**与现有手工实现相当甚至更优的性能**。
:::

此外，它还能为新兴模型（如 Mamba2）发现全新的流水线方案，相比最先进的手工实现显著提升性能。代码已开源：https://github.com/tile-ai/tilelang。

> 下面引用 Lei Wang 在不久前 AI 编译器交流会的幻灯片的三页内容：

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/02.webp)

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/03.webp)

## 关键问题

### Q1：PipeThreader 提出的 sTask-graph 计算抽象和分层硬件抽象（EU 与 sEU），相比现有 DNN 编译器的抽象方式（如 Triton 的 tile 抽象、TVM 的算子抽象），在捕捉异构硬件并行性和支持复杂流水线调度上有何本质突破？

现有 DNN 编译器（如 Triton 的 tile 抽象、TVM 的算子抽象）多以粗粒度计算单元（如张量切片、完整算子）为调度单位，难以捕捉异构硬件（TensorCores、TMA 等）的细粒度并行性，且硬件特性封装过深，限制了灵活调度。而 PipeThreader：

::: block-1
- :mdnice-bold-italic[sTask-graph] **将计算拆解为细粒度任务节点** （含计算、数据移动等操作）及依赖关系，可**精确描述硬件单元间的协同需求**；
- :mdnice-bold-italic[分层硬件抽象] （EU 包含异构 sEU）则**将不同类型硬件单元** （如计算型 sEU、数据移动型 sEU）**统一管理**。
:::

这两点打破了传统抽象对硬件功能的“黑盒化”封装，使软件能直接控制异构单元的流水线协同，从而**突破了粗粒度抽象对并行性挖掘的限制。**

### Q2：append、wait、propagate 等调度原语与两层调度策略如何协同工作，实现任务划分与流水线并行的联合优化，从而在保证内存约束的同时最大化硬件单元（如 TensorCores、TMA）的利用率？

`append`、`wait`、`propagate` 等调度原语构成了细粒度控制的基础：append 将 sTask 放入流水线队列，wait 确保任务依赖满足，propagate 实现数据在 sTask 间高效传递（减少显式内存读写）。

::: block-1
两层调度策略与之协同：

- 上层调度将 sTask-graph 划分为适配硬件资源的子图并分配给 EU（硬件单元集合）
- 下层调度在 EU 内通过原语动态构建流水线

使计算型 sEU（如 TensorCores）与数据移动型 sEU（如 TMA）并行工作。
:::

这种协同**既保证了任务划分符合内存约束** （避免数据溢出），又通过原语的灵活控制**最大化硬件单元的重叠执行**，最终提升利用率（接近 TensorCore 理论峰值）。

### Q3：PipeThreader 在新兴模型（如 Mamba2）上实现显著性能提升（远超 Triton），而在 RetNet-65B 上提升有限，这一差异主要源于模型自身特性（如注意力头维度）还是调度策略的适应性限制？其泛化能力的关键瓶颈是什么？

:mdnice-bold-italic[性能差异] 源于模型特性与调度策略的适配性：

- Mamba2 等新兴模型包含**大量细粒度计算与数据移动** （如滑动窗口操作），sTask-graph 能精准捕捉其依赖，调度原语可高效编排流水线；
- 而 RetNet-65B **结构更规整** （如大尺寸注意力头），现有编译器（如 Triton）已通过手工优化实现较优的粗粒度并行，PipeThreader 的细粒度调度增益被其结构限制。

> 泛化瓶颈在于：对**高度规整、已被深度手工优化的模型，细粒度调度的边际效益较低**；同时，**极端复杂的依赖关系** （如超大规模模型的全局依赖）**可能增加调度开销**，限制性能提升。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/04.webp)

## 本文目录

- 关键问题
  - Q1：PipeThreader 提出的 sTask-graph 计算抽象和分层硬件抽象（EU 与 sEU），相比现有 DNN 编译器的抽象方式（如 Triton 的 tile 抽象、TVM 的算子抽象），在捕捉异构硬件并行性和支持复杂流水线调度上有何本质突破？
  - Q2：append、wait、propagate 等调度原语与两层调度策略如何协同工作，实现任务划分与流水线并行的联合优化，从而在保证内存约束的同时最大化硬件单元（如 TensorCores、TMA）的利用率？
  - Q3：PipeThreader 在新兴模型（如 Mamba2）上实现显著性能提升（远超 Triton），而在 RetNet-65B 上提升有限，这一差异主要源于模型自身特性（如注意力头维度）还是调度策略的适应性限制？其泛化能力的关键瓶颈是什么？
- 一、引言
- 二、研究动机
  - 2.1 日益增长的硬件与软件复杂性
  - 2.2 传统数据并行 GPU 执行中的低利用率问题
  - 2.3 观察与机遇
- 三、PipeThreader 抽象
  - 3.1 专用任务与专用执行单元（sTask 与 sEU）
  - 3.2 从 sTask-graph 到 sProgram
  - 3.3 运行示例
- 四、PipeThreader 调度
  - 4.1 调度接口
  - 4.2 调度策略
- 五、PipeThreader 实现
  - 5.1 前端
  - 5.2 sTask Mapping on NVIDIA CUDA GPUs
  - 5.3 sTask Mapping on AMD ROCm GPUs
- 六、评估
  - 6.1 实验设置
  - 6.2 NVIDIA H100 上的算子性能
  - 6.3 NVIDIA H100 上的端到端性能
  - 6.4 调度策略评估
  - 6.5 AMD ROCm GPU 评估
- 七、实验进一步分析与讨论
  - 扩展到多 GPU
  - 支持新设备
  - MoE FFN 内核
- 八、相关工作
  - 深度学习编译器和框架
  - 特定模式的优化
  - 分布式深度学习框架
- 九、结论
- 参考文献

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/05.webp)

## 一、引言

> 深度神经网络（DNN）规模的不断增长对现代 AI 加速器（如 GPU）带来了巨大的计算和内存压力。为了满足日益增长的计算需求，硬件厂商引入了专用的异构硬件单元，如 TensorCores [14–16,20–22] 和 Tensor Memory Accelerators（TMA）[22]。

与此同时，软件开发者倾向于将多个 DNN 算子（如矩阵乘法这种计算密集型单元）融合成一个 GPU 内核，以最大化数据重用，从而缓解内存压力 [27,47]。 然而，:mdnice-bold-italic[这些新兴的硬件和软件趋势为高效的 DNN 执行带来了新的挑战]。

- 首先，**最大化专用硬件单元的利用率需要精心调度 DNN 计算流水线**。过去，GPU 依赖硬件调度器来执行线程，通过大量并发线程来摊销单个流水线中的潜在停顿。然而，这种方法不再有效，因为由于专用单元需要更大的张量粒度以实现更高的计算密度，可用的并发线程数量显著减少。
- 其次，**算子融合引入了更深的计算流水线**。:mdnice-bold-italic[硬件调度器难以理解如此复杂的流水线，使得有效调度变得困难]。

因此，最先进的 DNN 内核（如 FlashAttention [26]）都是手工精心组织的执行流水线。然而，**这种方法难以泛化**。对于每一种 GPU 类型（如 NVIDIA Hopper GPU 或 AMD GPU [46]）、新兴的 DNN 模型（如 Mamba [31]）以及 DNN 模型的新张量形状（如 FlashAttention2/3 [26,46]），都需要新的手工实现。

> 鉴于**软件流水线日益复杂**，以及**硬件调度的固有局限性**，本文提出了 PipeThreader，一种 DNN 编译器，支持在:mdnice-bold-italic[现代 GPU 架构上实现软件定义的流水线]，以高效利用专用的异构硬件单元。

::: block-1
- PipeThreader 将 DNN 计算抽象为 sTask-graph（专用任务图）。
- 图中的每个**节点表示一个 sTask（专用任务）**，这是一个可调度到专用单元的细粒度任务；图中的**有向边指定了 sTask 之间的依赖关系**。
- sTask 在 tile（张量切片，即张量的一部分）上进行计算。
- 现代 DNN 编译器广泛采用了 tile 的概念 [11,23,43,49,53,54,56]。

通过 sTask-graph，PipeThreader 可以提取出适合在专用硬件单元上执行的复杂计算流水线。
:::

:mdnice-bold-italic[为了暴露硬件的细粒度调度能力，PipeThreader 将 GPU 设备抽象为分层的硬件单元]，包括**虚拟化的同构并行执行单元** （即 EU，执行单元）和**每个 EU 内部的异构专用单元** （即 sEU，专用执行单元）。然后，可以使用三种原语（append、wait 和 propagate）来调度 sTask，从而通过软件实现复杂的流水线。PipeThreader 还采用了双层调度策略，在给定的优化空间中找到有效的调度方案，该优化空间由 sTask-graph 和新暴露的硬件能力定义。

我们用 8.5k 行 C++ 和 Python 代码实现了 PipeThreader，基于开源 DNN 编译器 TVM [19] 和 Ladder [51]。实验评估表明：

- PipeThreader 可以在 NVIDIA H100 或 AMD MI300X 等新硬件上发现与 FlashAttention 类似的流水线方案，无需手动实现即可实现相当或更好的性能。
- PipeThreader 还可以为 Mamba2 [28] 等新兴模型找到有效的流水线调度，相比最先进的手工实现显著提升性能。

> 我们相信，PipeThreader 是朝着定义基于 tile 的完整软件/硬件抽象范围的进一步尝试，这与现代的基于 tile 的编程工作（如 Triton [49]、CUTLASS 的 CuTe 抽象 [23] 等）以及新硬件架构的演进高度契合。

## 二、研究动机

### 2.1 日益增长的硬件与软件复杂性

大型深度神经网络（DNN）模型，尤其是大语言模型（LLM）[44]，的迅速扩张，推动硬件厂商开发专门的异构硬件单元，如 TensorCore 和 Tensor Memory Accelerator（TMA），以满足不断攀升的计算需求。同时，像 FlashAttention[27]这样精细的算子融合技术被越来越多地用来降低内存开销、最大化数据局部性（把数据留在离计算单元更近的片上存储，减少反复搬移）。这些趋势虽然提升了计算密度和效率，却也带来了显著的调度和执行挑战，**特别是在配备异构单元** 的现代 GPU 上。

### 2.2 传统数据并行 GPU 执行中的低利用率问题

传统的 GPU 编程模型（如 CUDA[3]）把线程块（threadblock）均匀地分派到各个流式多处理器（SM，GPU 里的一个小计算集群），并把每个 SM 视为一个“均质、独立”的执行单元。这种抽象假设所有 SM 可以互换，并隐藏了它们的内部结构，这在早期架构如 NVIDIA V100[20]上行得通。**然而，现代 GPU** （如 NVIDIA H100[22]）**在每个 SM 内部集成了【异构组件】**：:mdnice-bold-italic[专门做矩阵乘] 的 TensorCore、:mdnice-bold-italic[做通用计算] 的 CUDA Core、以及:mdnice-bold-italic[负责大批量数据搬运] 的 TMA。

这些组件职责不同、执行特性各异。如果仍按“均质”思路均匀分派线程块，而不了解 SM 内部的差异，就会导致资源浪费。**要想充分发挥这类架构的潜力，就必须意识到每个 SM 内部的异构单元，并据此协调任务的放置、调度和流水** （:mdnice-bold-italic[pipeline，让计算和数据搬运重叠进行]）。缺了这层控制，大量性能就会被白白浪费。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/06.webp)

图 1 报告了在 H100 上针对矩阵乘（MatMul）、FlashAttention 和 Mamba2 三种代表性负载的不同实现，各硬件单元的利用率。对于矩阵乘，由于未做流水线执行，数据搬运成为瓶颈，TensorCore 利用率只有 40%；而专家调优的 cuBLAS 可达 97%。FlashAttention-3[46]通过手工优化把 TensorCore 利用率从 40%提高到 72%，相比基于 Triton 的 FlashAttention2[26]。

值得注意的是，FlashAttention-2 升级到 FlashAttention-3 花了将近一年时间。然而，对手工优化的 Mamba2[28]来说，各单元依旧利用率低下，TensorCore 只有 15%。因此，对于新兴 DNN 模型，充分释放现代硬件潜力依旧困难重重。

**在 kernel 里手动管理流水线执行是出了名的困难，因为设计空间巨大且对硬件敏感**。开发者必须在 tile （tile，小数据块）大小与流水线深度之间精心权衡，同时受片上资源（共享内存、寄存器）的严格约束。

**这些挑战又因架构差异** （如内存层级、专用计算单元不同）**而被放大**。手动推理很快变得不可行，因此自动化推导与调度成为在性能与可移植性上取得突破的关键。然而，:mdnice-bold-italic[现有编译器（如 TVM[19]、Triton[49]）缺乏显式表达 tile 级流水线执行的机制]。它们为了隐藏底层控制，牺牲了开发者指定执行顺序、资源分配、计算-通信重叠的能力，阻碍了性能的完全释放。

> 实现高效的 tile 级流水线执行**需要一个新的编译器**，能够系统地探索这一设计空间，生成优化的调度，并跨多硬件平台自适应。

### 2.3 观察与机遇

基于上述事实，我们观察到独特的解决机会。由于**新硬件单元** 以**较大粒度（张量 tile ）处理数据**，已有工作[11,23,43,49,53,54,56]表明， :mdnice-bold-italic[tile 级执行因其在 tile 粒度上的确定性性能，可以在软件层高效调度]。借助这一趋势，我们主张将流水线调度从隐式的硬件行为转移到显式的软件控制。

> 这里所说的“流水线调度”并非指 GPU 底层线程或 warp 的派发，而是指由软件引导的、把 tile 级运算映射到每个 SM 内部专用单元（TensorCore、CUDA Core、TMA 等）的过程。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/07.webp)

图 2(a)展示了融合算子 MatMul-Sum 的执行：MatMul 在 TensorCore 上跑，Sum 在 CUDA Core 上跑。由于现有方法采用均质抽象，调度顺序把二者串行化，忽略了 TensorCore 与 CUDA Core 本身可以并行，导致低效。相反，图 2(b)展示了利用专用执行单元的优化调度，实现流水线执行，充分利用异构硬件。

> 遗憾的是，现有 DNN 模型表示以及 GPU 硬件接口**都没有显式暴露 tile 级流水线执行所需的调度能力**。

## 三、PipeThreader 抽象

第 2 节的观察促使我们提出 PipeThreader——一个把基于 tile 的数据并行与流水线调度相结合的 DNN 编译框架。:mdnice-bold-italic[下面，图 3 给出了系统概览。]

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/08.webp)

当前先进的 DNN 编译器（如 Triton [49]、Roller [56]、Welder [47]）都把硬件加速器抽象为一组同质的执行单元（EU），以支持 SPMD（单程序多数据）式并行。:mdnice-bold-italic[这种做法忽视了现代 GPU 内部本就存在的硬件异构性。]

举例来说，一个 SM 里的 :mdnice-bold-italic[TensorCore 与 CUDA Core 针对的 workload（工作负载）并不相同，而现有编译器无法利用这种多样性]。为弥补这一缺陷，PipeThreader 引入了两项关键抽象：专用任务（sTask）与专用执行单元（sEU）。

::: block-1
- 输入是算子的数据流图（DFG），PipeThreader 先把算子转换成 sTask——它们专门设计来挖掘 sEU 的异构能力，从而支持 MPMD（多程序多数据）式并行。
- 3.1 节会详细介绍 sTask 与 sEU。sTask 在任务粒度上保留了原 DFG 的数据依赖，构成一张 sTask-graph。
- 随后，PipeThreader 把这张 sTask-graph 映射到 sEU 上，形成一份结构化的 sTask-program（简称 sProgram），以便高效执行。
- 借助 sProgram 这一抽象，PipeThreader 打开了一个全新的搜索空间。3.2 节讨论 sTask-graph、sTask-program 以及搜索空间。
:::

### 3.1 专用任务与专用执行单元（sTask 与 sEU）

#### sTask 概念

PipeThreader 把“专用任务”（sTask）作为算子内部、准备被加速器上某一类:mdnice-bold-italic[专用执行单元（sEU）执行的最小计算单位]。这一概念**天然契合现代 DNN 加速器中异构且专用的处理器**，例如 H100 GPU 上的 TMA、CUDA Core、TensorCore。

为了最大化效率，加速器上的计算必须被拆成多种并行（同时异构）的任务，每种任务对应一类专用处理器。这些任务即可用 sTask 表示，既向底层硬件的专用处理器暴露潜在的任务级并行，也向 PipeThreader 编译器暴露同样的信息。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/09.webp)

如图 4 所示，一个 sTask 处理从输入张量切片（tile）得到的数据 tile ，并产生输出张量的数据 tile ；其计算由基于索引的张量表达式描述。sTask 的 shape（第 8 行）沿张量表达式 expr（第 7 行）的每条循环轴定义。此外，target_sEU（第 9 行）属性指定该 sTask 可在哪一类专用单元上执行。**相比之下，传统基于 tile 的任务并未显式分类，因而难以利用专用处理器的并行度**。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/10.webp)

图 5 以 FlashAttention 里 MatMul-Sum 融合算子为例：在传统视角下，一个 [2×2] 的 A tile 与 [2×2×2] 的 B tile 顺序执行，产出 [2×2] 的 C 输出 tile 。

**PipeThreader 引入两种 sTask**：在 TensorCore 上跑矩阵乘加（mma）的 mma-sTask，以及在 CUDA Core 上跑求和的 Sum-sTask。这样就可以**把第二个 mma-sTask** （用 A 乘 B1 得 C1）**与第一个 Sum-sTask** （把 C0 规约成 S0）**流水线化，实现任务级并行**，大幅提升硬件利用率。

#### sEU 概念

现代加速器缺少把 sTask 映射到特定执行单元的接口。为此，PipeThreader 显式暴露 GPU 内的执行单元，并将其抽象为一个层次化的执行阵列，既捕捉并行度，又支持带流水线的数据相关执行。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/09.webp)

如图 4 所示，抽象后的设备由若干并行执行单元（EU）（第 11 行）组成，每个 EU 里又包含若干异构的专用执行单元（sEU）（第 12 行）。这些 **sEU 构成 PipeThreader 进行数据相关任务流水线调度的硬件基础**。

例如，在最新 H100 GPU 上，一个流式多处理器（SM）就是一个 EU，其中包含用于 load-sTask 的 Tensor Memory Accelerator（TMA）和用于 mma-sTask 的 TensorCore。**sEU 通过 Execute 接口（第 15 行）执行给定 sTask**。`is_async` 属性（第 14 行）标记 sEU 是同步（如 CUDA Core）还是异步（如 TMA）。**异步 sEU 可与其他异步或同步 sEU 并发执行**。

### 3.2 从 sTask-graph 到 sProgram

为了执行 DNN，PipeThreader **把输入 DFG 转换成面向现代异构硬件的专用表示**。过程分为两步：先构建保留计算与依赖关系的 sTask-graph，再将其映射到硬件的 sEU 上，形成 sTask-program（简称 sProgram），以编排高效执行。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/08.webp)

#### sTask-graph 概念

> 如图 3 所示，输入 DFG 中的算子经过 sTask-partition 被转换成 sTask，从而构成一张 sTask-graph。

该图在任务粒度上保留了原 DFG 的计算与数据依赖：节点是 sTask，边记录细粒度的任务间依赖。sTask-partition 通过配置 sTask 的 TileShape（即 Map<Axis, Dim>）来划分每个算子，指定可切分的维度与 tile 大小。

传统编译器主要做空间划分以获得数据并行度；**PipeThreader 同时支持空间划分与规约（reduction）划分**，:mdnice-bold-italic[解锁新的流水线机会]。不同划分策略可产生多样的 sTask-graph，为后续执行规划提供更多选择与更高效率。

#### sProgram 概念

给定 sTask-graph 后，PipeThreader 将其映射到硬件 sEU，形成 sTask-program（sProgram）。sProgram 是一个二维数组 sProg[sEU][order]，每一项指定了某个 sTask 被分配到哪个 sEU 以及其执行次序。这种结构化表示便于任务的高效调度与执行。为保证有依赖的 sTask 按正确顺序执行，PipeThreader 引入 barrier-sTask：它在程序中引用 <EU_id, sEU_id, order> 列表，等待所列全部 sTask 完成后再继续。

#### 搜索空间

PipeThreader 的搜索空间由所有合法的 sProgram 构成，每个 sProgram 都是一个二维数组 sProg[sEU][order]，定义了每个 sTask 的 tile 大小与执行顺序（含同步栅栏）。**sTask 的排序与 tile 尺寸组合众多。该空间包含所有在遵守数据依赖前提下能够执行算子的方案**。例如，FlashAttention 的搜索空间包含 37,440 个合法 sProgram；**对复杂融合算子，sTask 调度对搜索空间贡献更大**。FlashAttention 中就有 36 种 tile 配置，但每种 tile 尺寸对应 1,040 种 sTask 排序。

### 3.3 运行示例

#### 示例 1：Mamba2

Mamba 是一种用块扫描（chunk-wise scanning）结合线性注意力机制的流行 DNN 模型。下面以其核心 ChunkScan 算子为例说明。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/11.webp)

- 前端。对 ChunkScan 函数，PipeThreader 接收图 6(a)所示的简明 IR：先把 cb 与 exp(dA)*dt 的乘积相乘（第 6 行），再把 cb 与 x 的矩阵乘结果累加到 acc_o（第 8 行）。PipeThreader 把 load_cb（第 3 行）、load_dA（第 4 行）、load_dt（第 5 行）、exp（第 6 行）、load_x（第 7 行）、mma（第 8 行）视为独立 sTask。
- sTask-graph。PipeThreader 根据依赖关系构建对应的 sTask-graph，并可沿空间维与规约维划分。空间划分（按 batch size）把图拆成更小的子图，分布到不同 EU 做 tile 级数据并行；规约划分（按 sequence length）则在单个 EU 内产生更细粒度的 sTask，为流水线执行提供机会。假设 acc_o 尺寸为 (M,N)，X 尺寸为 (K,N)，PipeThreader 常规地把空间维 (M,N) 划分成 tile (m,n)；同时把规约维 (K) 划分成 loop_range 次迭代，使迭代之间可重叠计算。图 6(b)展示了经规约维划分得到的 sTask-graph。
- sProgram。给定 sTask-graph，PipeThreader 有多种把 sTask 映射到 sEU 的方案，即多条 sProgram。图 6(c)给出三条示例 sProgram：sProg-A 先把 load_x 排在其他 load 之前；sProg-B 与 sProg-C 则相反；sProg-C 还采用更大的 tile 尺寸。
- 评估。图 6(d)评估三条 sProgram：表列出各时间步运行的 sTask 及对应片上内存占用（假设容量 1 KiB）。sProg-A 提前调度 load_x，却因 exp 必须等 load_cb_dA_dt 完成而延迟，整体效率低于 sProg-B。sProg-C 虽用更大 tile ，却使中间缓存占用增加。第 4 步时，工作区超出片上容量，导致该 sProgram 非法。最终选中 sProg-B 作为调度策略。

**PipeThreader 的划分遵循 tile 原则，但把规约 tile 化提升为关键优化策略**。传统 tile 侧重空间划分以重用数据，规约划分常被视作次要。:mdnice-bold-italic[PipeThreader 主动利用规约划分实现流水线]，提高执行效率，并:mdnice-bold-italic[在数据重用与流水线并行之间做出权衡]。

传统策略倾向选大 tile ，却可能与流水线并行所需的额外片上内存冲突。PipeThreader 进一步在 tile 与流水线之间做联合权衡。

#### 示例 2：FlashAttention-3

FlashAttention 是原始全注意力机制的高效实现，输入为 Q、K、V 三 tensor。先算 MatMulQK 得 acc_s = QKᵀ，再经 Softmax 得 P，最后 MatMulPV 得输出 O。FlashAttention 把这三算子融合为一个 kernel。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/12.webp)

PipeThreader 先标注模式，得到依赖 sTask，形成 sTask-graph（图 7）。划分后，load_k、load_v 分给 TMA；mma_qk、mma_pv 分给 TensorCore；softmax、rescale 分给 CUDA Core。PipeThreader 用双层调度策略在该空间内搜索，生成 sProgram，使各 sTask 在对应 sEU 上并行执行，利用 sEU 的异步与异构特性。值得注意的是，我们的调度空间包含了最新 FlashAttention-3[46] 的流水线方案。

## 四、PipeThreader 调度

> sProgram 抽象打开了一个庞大的优化空间。PipeThreader 的目标是在该空间中生成高质量的 sProgram。

为此，PipeThreader 把**调度机制** （mechanism）与**调度策略** （policy）解耦。

::: block-1
- **机制层面** 提供两种能力：
- **策略层面** 提供一种**双层（two-layer）** 策略，在** tile 并行**与** 流水线并行**之间做权衡。该简单策略已能超越现有最佳方案，有时甚至显著领先。我们相信该机制为未来更高级策略的研究奠定了坚实基础，使其能进一步挖掘 sProgram 暴露的优化空间。
:::

### 4.1 调度接口

PipeThreader 提供以下三个接口，用于在新空间中生成高质量 sProgram：

| 接口 | 说明 |
| --- | --- |
| `Append(sTask s, <EU u, sEU v>)` | 将指定 sTask 分配到某个 EU 内的特定 sEU。 |
| `Wait(sTask_id s, list<sTask_id> t)` | 允许 sTask `s` 等待列表 `t` 中的所有 sTask 完成；这会在 `s` 之前隐式插入一个 barrier-sTask。 |
| `Propagate(sTaskGraph g, TileShape shape)` | 用于探索 sTask 划分空间，自动推断 sTask-graph 中每个 sTask 的 TileShape。过程：从最后一个 sTask 的输出 tile 形状出发，沿图反向传播，依据张量表达式与输出 tile 形状推断每个 sTask 所需的输入 tile 形状。 |

举例：若 Softmax sTask 需要 `[4×128]` 的输出 tile ，则可推断其输入 tile 形状也是 `[4×128]`；若该输入又是前一个 mma sTask 的输出，则进一步推断出其输入 tile 形状为 `[4×k]` 与 `[k×128]`，其中 `k` 为规约维度大小。

上述接口允许显式控制 sTask 在 sEU 上的放置与执行顺序（即 sProgram），以探索并行化空间。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/13.webp)

### 4.2 调度策略

- **硬件抽象回顾**：
- **算法概览（图 9）**：

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/14.webp)

1. **Inter-EU 阶段** （第 1–10 行）
2. **Intra-EU 阶段** （第 11–32 行）

- **优先级策略**：优先调度**异步 sTask**，且选择对后续 sTask **解锁潜力大、依赖最少** 的任务。例如图 6(c)中，提前调度 `load_y` 能让 `exp` 更早启动，故算法赋予 `load_y` 更高优先级，从而自然倾向于选择 `sProg-B`。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/11.webp)

- **内存与 tile 权衡**：提高 sTask 重叠度（流水线并行）需更多片上内存缓存中间结果，但可能与更大 tile 尺寸冲突。我们采用**联合搜索策略**，在剖析反馈指导下平衡二者；候选方案若超出内存上限，`check_valid` 会跳过（图 6 中 `sProg-C` 即因此被排除）。

#### Profiler

PipeThreader 引入 **profiler** 指导高效 sProgram 生成：

| 提供的信息 | 作用 |
| --- | --- |
| (1) 每个 sTask 在特定 sEU 上的执行时间 | 用于估算任务启动时刻、保持流水线效率、减少空闲 |
| (2) 资源占用（本地内存、寄存器） | 用于 `check_valid`，确保不超出片上内存 |
| (3) 整个 sProgram 的总体执行时间 | 调度完成后再次实测，为策略提供真实反馈 |

Profiler 通过现有编译器后端（如 TVM）自动生成并测量单个 sTask 的设备代码性能；调度阶段用这些数据做决策，调度完成后再次实测整个 schedule 的延迟。

## 五、PipeThreader 实现

PipeThreader 用约 **8.5k 行 C++ 与 Python 代码** 实现，基于开源 DNN 编译器 TVM [19] 与 Ladder [51]。图 10 总结了整体工作流：

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/15.webp)

### 5.1 前端

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/12.webp)

sTask IR：

- 提供**灵活表达 sTask 级计算** 的中间表示，可表达现有 IR 难以捕捉的模式。
- 图 6(a)、图 7 的伪代码即简化版 sTask IR。
- 涵盖**数据搬运** （DRAM↔SRAM）与** tile 级计算序列**。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/11.webp)

sTask converter（含 sEU 标注）：

- 把 **sTask IR 或 ONNX 图** 转换成 sTask-graph。
- 借助 **Ladder [51]** 做算子融合，输出 TVM TIR 表达的 tile-graph。
- 为每个 tile 任务添加 `target_sEU` 属性，映射到具体 sEU：
- 自动生成常见深度学习 kernel 所需的基础 sTask（load / mma / reduce / softmax …）。
- **用户代码量对比**：

> 开发者无需手动划分任务、指定 tile 形状、sEU 绑定——PipeThreader 自动推断。

### 5.2 sTask Mapping on NVIDIA CUDA GPUs

对于 TMA 和 Tensor Core 的 sEU，我们都将 is_async 属性设为 true，因为可以利用 cp.async.bulk 和 wgmma.mma_async 这两条异步指令。

| 单元类型 | `is_async` | 说明 |
| --- | --- | --- |
| TMA & TensorCore | `true` | 使用 `cp.async.bulk`、`wgmma.mma_async` |
| CUDA Core | `false` | 无原生异步指令，但与 TensorCore 可同时发射 [8] |

与之相反，CUDA Core 的 is_async 设为 false，因为它们不支持任何异步指令。NVIDIA 官方已确认 [8]，CUDA Core 与 Tensor Core 的指令可以并行下发。由于两者共用同一组寄存器，可能会产生干扰；我们通过实现双缓冲机制来缓解这种潜在干扰。我们使用 PTX 中的 mbarrier 对象来实现 barrier-sTask。

- **寄存器冲突**：TensorCore 与 CUDA Core 共用寄存器 → **双缓冲寄存器** 实现重叠。
- **barrier-sTask**：用 PTX `mbarrier` 对象实现。

为了在 sEU 中高效地实现 Execute 函数，我们通过布局推断决定 sTask 操作与数据在不同线程及物理内存中的映射，并借助硬件专用指令（intrinsics）加速运算。

PipeThreader 在支持同一厂商的不同 GPU 型号时无需大量工程投入：当面向 A100 [21]、H100 [22] 或 B100 [24] 等不同架构时，只需对 sEU 布局、intrinsics 及资源上限等硬件相关配置进行轻量级更新，核心编译与调度逻辑完全可复用。

#### Layout Inference（布局推断）

- **Layout 对象** = 映射函数 + 迭代域，描述数据布局与线程绑定。
- **全自动**：高优先级 sTask（如 mma）决定依赖任务布局，冲突用优先算法解决。
- 图 11(a)(b) 给出 mma→sum 子图的简化示例。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/16.webp)

为了在各专用执行单元（sEU）上高效执行 sTask，必须遵守特定的数据布局与线程绑定约束。为此，PipeThreader **引入了一个 Layout 对象，用来描述 sTask 的数据布局及线程绑定方式**。Layout **由映射函数和迭代域组成**，明确规定逻辑数据元素如何映射到物理内存，并可选择性地指定其绑定的线程。

:mdnice-bold-italic[PipeThreader 完全自动地完成布局推断，无需人工指定]。图 11a 展示了一个简化的推断流程。以 mma sTask 为例，它被分配给 Tensor Core sEU，而该单元对布局有严格约束。根据这些约束，系统可推导出相应的布局映射函数，其中 {T(m),n} 表示将数据元素映射到线程 m 的第 n 个位置。

在 sTask-graph 中，相连 sTask 的布局必须保持一致以确保兼容。PipeThreader 依据各 sEU 的布局需求，对 sTask 的 Layout 进行推断，并将这些需求沿计算图传播。**若出现布局冲突，系统使用基于优先级的推断算法解决**：:mdnice-bold-italic[高优先级 sTask（如 mma）的已确定布局将主导其依赖 sTask 的布局]。例如，在图 11b 中，mma 与 sum 两个 sTask 相连。由于 mma 的布局已被确定，系统据此推断出 sum 的布局；在此例中，张量 C_sum 需要被复制一份。

#### Hardware Intrinsics

对于需要在 sEU 上执行批量操作的 sTask，我们将其下沉为基于 tile 级的函数模板。例如，矩阵乘累加运算通过 CUTLASS/CuTe 模板实现下沉，这些模板已内嵌针对 Tensor Core 的硬件专用 intrinsics。进一步的指令级优化（如寄存器分配）则交由 LLVM 等底层编译器完成 [38]。在 **NVIDIA H100 上，我们采用 Warp Specialization [17, 25] 技术进一步优化执行：将线程划分为生产者 warp 和消费者 warp，各自负责不同的流水线阶段。** 生产者 warp 将未使用的寄存器释放后，可被消费者 warp 复用，从而提升寄存器分配和利用效率。

根据 H100 中 TMA 单元的特性，我们把负责在全局内存与共享内存之间搬运数据的 load sTask 分配给生产者 warp；而其余 sTask（如 mma 与 Softmax）则由消费者 warp 处理。**生产者与消费者之间的同步通过以 mbarrier 实现的 barrier-sTask 完成，以确保数据依赖的正确性**。

简单总结如下：

- **TensorCore**：借助 CUTLASS / CuTe 模板 + TensorCore intrinsic。
- **寄存器分配**、指令级优化交由 LLVM [38]。
- H100 Warp Specialization [17,25]：

### 5.3 sTask Mapping on AMD ROCm GPUs

我们还将 PipeThreader 部署到了 AMD 最新一代高性能 GPU MI300X [16] 上。MI300X 的并行执行单元称为计算单元（Compute Units, CUs），与 NVIDIA 的 SM 类似。每个 CU 内包含多种专用执行单元（sEU）：用于矩阵乘累加的 MatrixCore、算术逻辑单元（ALU）以及异步拷贝单元。

:mdnice-bold-italic[与在 CUDA GPU 上的做法一致]，PipeThreader 在 ROCm GPU 上同样执行布局推断，将 sTask 的数据映射到物理地址和线程。**此外，我们显式使用 `lgkmcnt` 和 `s_waitcnt` 指令来管理异步屏障** （barrier），从而**精确控制指令依赖关系** 并**同步内存操作**。

## 六、评估

在本节中，我们在 DNN 微基准测试和端到端模型上评估 PipeThreader，并与最先进的 DNN 编译器、框架和库进行比较，以证明 PipeThreader 的有效性。我们:mdnice-bold-italic[首先总结我们的发现]：

::: block-1
1. PipeThreader 能够为已得到充分研究的 DNN 架构（例如 FlashAttention）发现高效的调度，并达到与最先进方案相当甚至更好的性能；
2. PipeThreader 能够为新兴模型（例如 Mamba2）揭示新的调度方案，显著优于现有最佳的手工实现；
3. PipeThreader 的抽象和设计能够适配 NVIDIA GPU 以外的硬件（例如 AMD MI300X），并取得显著的性能提升。
:::

### 6.1 实验设置

:mdnice-bold-italic[硬件平台的选择]。我们**在 NVIDIA 和 AMD GPU 上评估 PipeThreader**，因为它们是当前最受欢迎的硬件平台。我们的评估包括两款最新的高性能 GPU：NVIDIA H100（80GB）[22] 和 AMD Instinct MI300X GPU（192GB）[16]。我们在 **H100 GPU 上使用 CUDA 12.4**，在 **MI300X GPU 上使用 ROCm 6.1.0**。两款 GPU 均在 Ubuntu 20.04 操作系统上进行评估。

:mdnice-bold-italic[DNN 工作负载的选择]。我们的评估基准使用了六种典型的 DNN 模型，包括 LLAMA3-8B [29]、LLAMA3-70B [29]、Mamba2-1.3B [28]、RetNet-65B [48]、ResNet-50 [32] 和 UNet [45]。

- 对于像 LLAMA3-8B、LLAMA3-70B 和 RetNet-65B 这样的大语言模型，我们使用 (BS, SEQ) 配置 (1, 1)、(32, 1) 和 (1, 4096) 进行测试；
- 其他模型如 ResNet-50 和 UNet 则使用批大小 1 和 128 进行评估，以全面覆盖在线和离线推理场景。
- 对于 Mamba，我们评估了 BS=1 时序列长度分别为 1k、2k、4k、8k，以及 BS=32 或 128 时序列长度为 1 的情况。选择这些配置是因为 Mamba 相比 Transformer 的主要优势在于长序列的计算效率，而这些设置代表了最常用的场景。
- 我们从每个模型中选择最频繁和最昂贵的操作来构建我们的微基准测试。

**表 1 列出了代表性算子、其配置以及每个算子的对应缩写。**

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/17.webp)

下面是我们选择的:mdnice-bold-italic[性能基线参考]：

- 我们将 PipeThreader 与 DNN 框架 ONNXRuntime（v1.19.2）[7] 以及最先进的 DNN 编译器（如 Ladder [51] 和 PyTorch-Inductor（v2.4.0，含 Triton v3.0.0）[10,49]）进行比较。
- PyTorch 集成了 bitsandbytes [30]，这是 HuggingFace transformers 的官方后端，用于 WFP4AFP16 精度。
- 此外，我们还将 PipeThreader 与 NVIDIA 的特定供应商推理库 TensorRT（v10.0.1）[6] 进行评估.
- 我们还将 PipeThreader 与用于 MatMul 的库 cuBLAS [2] 和 rocBLAS（ROCm GPU 上）[1]、用于低精度 MatMul 操作的 Ladder 和 bitsandbytes 库、用于 Conv2D 的 MIOpen [36] 以及使用 CUTLASS [5] 模板为注意力操作编写的专家优化版 FlashAttention-3 [46] 进行比较。
- 对于 LLM 和 Mamba 模型，我们进一步将 PipeThreader 与用于 LLM 推理的最流行库 vLLM（v0.6.3）[37] 进行基准测试。
- 平均性能指标（如加速比）使用所有实验的几何平均值计算。所有评估均从预热迭代开始，然后对每个工作负载重复执行至少 5 秒，以确保准确和稳定的结果。

### 6.2 NVIDIA H100 上的算子性能

图 12 显示了微基准测试中所有算子配置的性能。x 轴表示不同的算子，y 轴表示相对于 PipeThreader 的归一化延迟。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/18.webp)

#### MatMul

图 12 的第一行显示了 PipeThreader 与其他基线在从 LLAMA3-8B（M0-M7）和 LLAMA3-70B（M8-M15）派生的 MatMul 算子上的性能。**尽管现有编译器或库已经提供了经过良好优化的 MatMul kernel，但结果表明 PipeThreader 仍然获得了显著的加速**。与 PyTorch、Triton 和 Ladder 相比，PipeThreader 分别实现了平均 1.24 倍（最高 1.40 倍）、1.13 倍（最高 1.26 倍）和 2.07 倍（最高 2.25 倍）的加速。这一改进源于 PipeThreader 利用其 sTask 抽象将 MatMul 建模为 load 和 mma sTask 的流水线，并充分探索高级调度机会。值得注意的是，PipeThreader 匹配了 cuBLAS MatMul 的性能，平均加速 1.06 倍。它还在大多数 MatMul 算子上实现了超过 750 TFLOPS 的性能，接近 H100 GPU 上 TensorCore 的理论峰值性能。

#### Convolution

PipeThreader 使用隐式 GEMM [39] 实现卷积，这也利用了 load 和 mma 的流水线优化。如图 12 第二行所示，PipeThreader 在从 ResNet-50 模型派生的卷积算子上（批大小为 1 和 128）显著优于基线。PipeThreader 分别比 PyTorch 和 Ladder 实现了平均 1.94 倍（最高 3.52 倍）和 2.56 倍（最高 8.66 倍）的性能提升。由于官方未提供 Conv2D 实现，我们使用 Triton 实现了一个 Conv2D kernel 并进行自动调优以达到其最佳性能。与 Triton 相比，PipeThreader 提供了平均 1.85 倍（最高 2.47 倍）的加速。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/18.webp)

#### Low-bit MatMul

图 12 的第三行显示了从 WFP4AFP16 量化的 LLAMA3-8B（DM0-DM7）和 LLAMA3-70B（DM8-DM15）派生的低比特 MatMul 算子的性能。由于当前 TensorCore 不直接支持 WFP4AFP16 MatMul，低比特 MatMul 必须首先在 CUDA Core 上将数据转换为 FP16，这引入了额外的 dequant 阶段。随着流水线阶段的增加，我们观察到 PipeThreader 比标准 MatMul 获得了更大的加速。例如，在低比特 MatMul 上，PipeThreader 分别比 PyTorch（使用 bitsandbytes）和 Ladder 实现了 3.92 倍（最高 4.76 倍）和 2.48 倍（最高 3.81 倍）的加速，而在标准 MatMul 上的平均加速仅为 1.24 倍和 2.07 倍。

#### FlashAttention 和 FlashDecoding

与 MatMul 和低比特 MatMul 相比，FlashAttention 涉及更深的计算阶段，为 PipeThreader 提供了更大的优化空间。PipeThreader 可以通过探索由 sTask-graph 和新暴露的硬件能力定义的新优化空间，自动找到有效的调度方案。

图 12 的第四行显示了从 LLAMA3-8B（FA0-FA9）和 LLAMA3-70B（FA10-FA19）派生的 FlashAttention 算子的性能。评估涵盖了从 512 到 8k 的序列长度，批大小为 1 和 64，包括带和不带因果掩码的配置。PipeThreader 分别比 Triton 实现了平均 1.36 倍（最高 1.50 倍）的性能提升，而 MatMul 操作的相应提升仅为 1.13 倍（最高 1.26 倍）。这突出了系统利用 FlashAttention 更复杂计算流水线中固有更大优化空间的能力。

> 这种通用调度能力使 PipeThreader 能够达到与专家优化的特定模型实现相当的性能，并在某些配置中甚至优于它们。

FlashAttention-3 是为 NVIDIA Hopper GPU 量身定制的手工优化注意力 kernel，是这种专家设计方法的一个例子。PipeThreader 比 FlashAttention-3 实现了平均 1.07 倍（最高 2.18 倍）的性能提升。作为一种手工方法，FlashAttention-3 无法有效地针对不同的工作负载进行优化。我们观察到，**特别是对于较小的序列长度，由于固定 tile 大小，FlashAttention-3 的性能并不理想**。PyTorch 使用手工 FlashAttention-2 kernel，没有包含更细粒度的流水线。与 PyTorch 相比，PipeThreader 实现了平均 1.82 倍（最高 2.29 倍）的加速。

FlashDecoding 算子选自 LLAMA3-8B（FD0，FD1）和 LLAMA3-70B（FD2，FD3）模型，批大小设为 1，上下文长度设为 8192 以模拟解码场景。与 FlashAttention-3 和 Triton 相比，PipeThreader 分别实现了平均 1.12 倍（最高 1.23 倍）和 2.27 倍（最高 3.06 倍）的加速。

#### Linear Attention

图 12 的第五行和第六行分别展示了 Mamba2 模型的关键线性注意力操作：ChunkScan 和 ChunkState 操作。我们与官方 Triton 实现进行比较。我们测试的配置序列长度从 1k 到 16k，批大小为 1 或 64。PipeThreader 分别比 Triton 实现了平均 1.71 倍（最高 1.99 倍）和 1.98 倍（最高 2.59 倍）的加速。此外，Triton 在某些配置（如序列长度 8k（CC14、CT14）和 16k（CC7、CC15、CT7、CT15））上会失败。这些结果强调了 PipeThreader 在处理新兴 DNN 操作时的适应性，无需手工实现。

### 6.3 NVIDIA H100 上的端到端性能

> 图 13 显示了 NVIDIA H100 GPU 上所有八个 DNN 模型的端到端性能。

由于 GPU 内存限制，我们使用单个解码层评估 LLAMA3、Mamba2 和 RetNet 等大语言模型的推理延迟，作为完整模型性能的代理，因为所有层都是相同的，延迟随层数线性扩展。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/19.webp)

#### LLM 模型

在 FP16 精度的 LLAMA3-8B 和 LLAMA3-70B 模型上，PipeThreader 分别比 Ladder 和 ONNXRuntime 实现了平均 2.17 倍和 2.45 倍的加速。Ladder 的调度策略无法有效地表示和生成 FlashAttention kernel，而 ONNXRuntime 本身不提供对 FlashAttention 的支持，导致性能不佳。

尽管 PyTorch-Inductor、TensorRT 和 vLLM 将工业界流行的 FlashAttention kernel 作为其后端，但 PipeThreader 平均分别比它们快 1.79 倍（最高 2.15 倍）、1.28 倍（最高 1.47 倍）和 1.10 倍（最高 2.05 倍）。这一改进归功于 PipeThreader 能够为模型中的操作（例如 MatMul 和 FlashAttention）探索更高效的流水线配置。

在 WFP4AFP16 量化场景中，PipeThreader 平均分别比 Ladder、PyTorch-Inductor 和 vLLM 快 2.01 倍（最高 3.39 倍）、3.03 倍（最高 11.98 倍）和 2.16 倍（最高 5.16 倍），因为它优化了低比特 MatMul 和 FlashAttention 的流水线调度。我们还在表 2 中提供了部分绝对性能。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/20.webp)

#### 线性注意力模型

在 Mamba2-1.3B 模型上，Ladder、ONNXRuntime 和 TensorRT 无法生成高效的线性注意力 kernel，导致在某些配置下出现内存错误。PyTorch-Inductor 使用 Triton 作为后端，能够运行融合的线性注意力，但与 PipeThreader 相比性能不佳。PipeThreader 分别比 PyTorch-Inductor、ONNXRuntime、TensorRT、vLLM 和 Ladder 快 1.92 倍（最高 2.76 倍）、2.71 倍（最高 5.10 倍）、1.21 倍（最高 2.44 倍）、1.78 倍（最高 2.41 倍）和 45.93 倍（最高 84.41 倍），因为它能够为模型中的操作探索更高效的流水线配置。

对于线性注意力模型 RetNet-65B，与 PyTorch-Inductor、ONNXRuntime、TensorRT 和 Ladder 相比，PipeThreader 分别实现了 1.16 倍（最高 1.31 倍）、1.60 倍（最高 2.12 倍）、1.06 倍（最高 1.46 倍）和 1.04 倍（最高 1.10 倍）的加速。在 RetNet-65B 模型上，PipeThreader 实现的加速相对较小。这是因为 RetNet-65B 模型的**注意力头维度较大** （query 和 key 为 256，value 为 432），**导致共享内存使用量高，限制了流水线调度的优化**。

#### CNN 模型

对于 ResNet-50 和 UNet，通过生成更高效的 Conv2D kernel，PipeThreader 在端到端推理延迟上分别比 Ladder、PyTorch-Inductor 和 ONNXRuntime 快 2.01 倍、2.54 倍和 3.99 倍。与 TensorRT 相比，PipeThreader 实现了相当的性能（0.97 倍）。

### 6.4 调度策略评估

#### 联合优化

我们的调度策略联合优化了 sTask-graph 的划分和流水线调度。为了证明其优势，我们创建了一个 PipeThreader 的变体：“PT-decouple”，它在独立的优化过程中分别优化划分（例如，为了单个 sTask 的高内存利用率）和流水线调度（例如，为了高重叠）。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/21.webp)

如表 3 所示，在 Mamba2-ChunkScan（BS=64，SEQ=8k）算子上，使用 PT-decouple 时，编译器专注于最大化数据重用，并选择更大的 tile 形状（例如 64×128）。然而，这种**更大的 tile 形状限制了 sTask-graph 在一个 EU 上实现有效流水线并行的能力**，导致执行时间为 12.150 毫秒。**通过联合优化 sTask-graph 的划分和调度，编译器选择了更小的 tile 形状** （例如 64×64），从而**实现了更高效的流水线**，并将执行时间减少到 6.981 毫秒。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/22.webp)

#### 编译时间

联合优化需要相对较长的编译时间。**表 4 给出了 PipeThreader 在 H100 上为 FlashAttention 和 Mamba2 的几个典型配置的编译时间**。由于所有任务划分都在图 9 的第 2 行生成，调度过程可以并行化以加快编译。对于具有少量流水线深度的简单 kernel，如 MatMul，PipeThreader 的编译时间仅为 0.13 分钟，而 Triton 和 CUTLASS 分别为 0.17 分钟和 3.36 分钟。对于复杂的融合 kernel，如 FlashAttention，尽管它探索了比 Triton 更大的流水线搜索空间，**PipeThreader 的编译时间仍然很短，为 5.26 分钟**。

### 6.5 AMD ROCm GPU 评估

#### 算子性能

我们使用最初为 NVIDIA H100 GPU 设计的微基准测试套件中的部分算子来评估 AMD MI300X GPU。评估侧重于关键算子：MatMul（与 PyTorch、rocBLAS、Triton 和 Ladder 比较）、Conv2D（与 PyTorch、MIOpen、Triton 和 Ladder 比较）、FlashAttention（与 FlashAttention-2 和 Triton 比较）和 Linear Attention（与 Triton 比较）。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/23.webp)

图 14 显示，PipeThreader 在不同类型的算子上分别比 Triton 实现了 1.16 倍到 5.42 倍的加速，比 PyTorch 实现了最高 6.21 倍的加速。

::: block-1
- PipeThreader 在 MatMul 和 Conv2D 上分**别超过了 rocBLAS 和 MIOpen，加速比分别高达 1.77 倍和 2.21 倍**。
- PipeThreader 还**比 FlashAttention-2 实现了最高 2.82 倍的加速**。
- 此外，PipeThreader 平均**比 Ladder 快 1.45 倍**，证明了其效率和可扩展性。
:::

#### 端到端性能

> 我们在 AMD Instinct MI300X GPU 上评估了 PipeThreader 与 Ladder、PyTorch-Inductor、ONNXRuntime 和 vLLM 的性能。

图 15 显示了在八个模型上的端到端性能结果，包括 FP16 和 WFP4AFP16 格式的 LLAMA3-8B 和 LLAMA3-70B，以及 Mamba2-1.3B、RetNet-65B、ResNet-50 和 UNet。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/24.webp)

在 FP16 精度的 LLAMA3-8B 和 LLAMA3-70B 模型上，PipeThreader 分别比 PyTorch-Inductor、ONNXRuntime、vLLM 和 Ladder 实现了 1.48 倍（最高 2.77 倍）、6.33 倍（最高 15.51 倍）、1.02 倍（最高 1.32 倍）和 1.07 倍（最高 1.29 倍）的加速。在 WFP4AFP16 LLAMA3-8B 和 LLAMA3-70B 模型上，PipeThreader 分别比 PyTorch-Inductor 和 Ladder 实现了 3.97 倍（最高 12.66 倍）和 1.12 倍（最高 1.34 倍）的加速，而 ONNXRuntime 和 vLLM 在 ROCm 平台上不支持 WFP4AFP16 量化。

对于 Mamba2-1.3B，PipeThreader 比 Ladder 实现了惊人的平均 32.93 倍（最高 61.33 倍）的加速。这一显著的性能提升主要归因于 **Ladder 无法融合线性注意力组件**。与 PyTorch-Inductor 相比，PipeThreader 实现了平均 1.31 倍（最高 1.54 倍）的加速。在 RetNet-65B 模型上，PipeThreader 分别比 PyTorch-Inductor、ONNXRuntime 和 Ladder 实现了 1.03 倍（最高 1.36 倍）、4.75 倍（最高 5.73 倍）和 1.01 倍（最高 1.02 倍）的加速。

在传统 CNN 模型（ResNet-50 和 UNet）上，PipeThreader 分别比 PyTorch-Inductor、ONNXRuntime 和 Ladder 实现了 2.74 倍（最高 5.66 倍）、5.84 倍（最高 15.47 倍）和 2.14 倍（最高 6.54 倍）的加速。

MI300X GPU 的异步能力相对较低，共享内存比 NVIDIA H100 GPU 更小，这降低了 PipeThreader 启用的流水线并行带来的潜在性能（例如，相对于基线的加速）。

## 七、实验进一步分析与讨论

> 与手工实现的内核相比，PipeThreader 显示出**【根本性的优势】**，特别是在**自动流水线调度** 和**跨架构可移植性** 方面。

- 首先，它:mdnice-bold-italic[消除了专家级手工调整的需要]。手工设计高效的流水线调度是容易出错、耗时且对输入配置敏感的。即使是专家手工优化的内核，如 FlashAttention-3（FA3），最初也缺乏对某些维度（例如，头大小为 256）的支持，这说明了其难度。PipeThreader 自动化了这一过程，系统地在硬件约束下探索调度空间。它在 FA3 上实现了高达 2.18 倍的加速，并且在 vLLM 的 Triton 基 Mamba2 上实现了 2.41 倍的加速。
- 其次，它:mdnice-bold-italic[在不同硬件上具有良好的泛化能力]。虽然手工调整的内核通常与特定平台紧密耦合，尤其是 NVIDIA GPU，但 PipeThreader 能够在 AMD 硬件上实现显著的性能提升。它的抽象也自然地映射到类似 TPU 的架构 [34,35]（例如，TPU 核心和 DMA 引擎），从而实现高效的流水线执行。
- 最后，它:mdnice-bold-italic[降低了高性能的门槛]。在 Multihead Latent Attention（MLA）[40,41] 上，PipeThreader 仅用 80 行 Python 代码就实现了高达 5 倍的加速，与 DeepSeek 的 500+ 行 CUDA 实现 [4] 相当，但开发工作量要小得多。

### 扩展到多 GPU

PipeThreader 可以**自然地扩展到多 GPU，并与张量并行性引入的集体通信相互作用**，通过将 1）GPU 之间的通信单元（例如，RDMA、NVLINK、IB）作为 sEUs，2）集体通信作为 sTasks 包含进来。这样，PipeThreader 可以重用策略，在 GPU kernel 级别搜索集体通信和计算之间的高效流水线，并扩展到多 GPU 或多节点环境。当前结果显示，PipeThreader 在常见通信模式上与 TileLink [55] 等最先进的系统表现相当。

### 支持新设备

我们发现广泛使用的硬件（例如，NVIDIA/AMD GPU 或 TPUs）与 sEU 抽象对齐，它们包含所有 sEU 的完整集合。为了将 sTask-graph 编译到设备上，编程模型只需要为每个 sEU 实现其自身的 load/store/compute 指令的 Execute 接口（图 4）。设备虚拟化类似于 Roller [56] 和 Welder [47] 的硬件抽象，但进一步暴露了细粒度的异构 sEUs。

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/09.webp)

### MoE FFN 内核

PipeThreader 也可以支持 MoE FFN 内核中的分组 MatMul。与批量 MatMul 不同，每个组可以有不同的形状。为了处理这种情况，PipeThreader 可以将每个组分解为一个独立的 sTask-subgraph，每个子图都有自己的输入形状，并分别应用策略，而不是共享单一调度。

## 八、相关工作

### 深度学习编译器和框架

大多数现有的 DNN 编译器将硬件抽象为同质执行单元（EU）。Rammer [43] 引入了 rTask 的概念，用于跨 EU 的并行执行，而 Welder [47] 专注于通过垂直融合进行整体内存优化。相比之下，PipeThreader 引入了 sTask 和 sEU，:mdnice-bold-italic[明确暴露硬件的异构性]，以实现流水线并行的优化和调度。

算子融合已在 DNN 编译器中广泛采用，例如 TVM [19]、Ansor [54]、XLA [13] 和 TensorRT [6]，以减少内存开销，从而导致更深的计算阶段。像 Triton [49]、Welder [47]、Roller [56]、Cocktailer [52]、TensorIR [11]、ThunderKittens [12]、FractalTensor [42] 和 Ladder [51] 这样的编译器都基于 tile 抽象进行调度优化。然而，这些主要侧重于空间 tile 以增强数据局部性，实现跨 EU 的数据并行，但**忽略了跨 sEU 的流水线并行机会**。

尽管 Triton [49] 和 CUTLASS [5] 等工作引入了流水线执行，但它们依赖于**特定算子的临时规则**，无法推广到多样化的工作负载。PipeThreader 通过引入抽象来解决这些限制，从而实现流水线并行的自动调度和优化。

像 ALCOP [33] 这样的框架专注于数据加载和计算之间的流水线，以优化内存层次结构的利用率。然而，它们**未能充分利用现代计算单元的异构性**，也无法探索具有深计算阶段的工作负载（如 FlashAttention）的流水线调度。PipeThreader 通过引入更细粒度的抽象来填补这一空白，从而实现跨异构硬件组件的全面流水线优化。

### 特定模式的优化

由于现有编译器中缺乏 sTask 和 sEU 抽象，流水线并行优化通常针对特定模式手工定制。例如，FlashAttention [46] 和 CUTLASS [5] 中的 Hopper MatMul 提供了特定模式的调度，但需要大量的手工工作。此外，FlashAttention 为不同的输入提供了单独的调度，而 CUTLASS [5] 要求用户进行剖析并选择最佳调度。相比之下，PipeThreader 通过利用 sTask 和 sEU 抽象来推广流水线并行，从而无需手工干预即可自动跨多种算子和配置进行调度。

### 分布式深度学习框架

Centauri [18]、PrimePar [50] 和 TileLink [55] 通过分层调度、时间张量划分和基于 tile 的抽象来提高通信 - 计算重叠。PipeThreader 可以将通信和计算建模为独立的 sTask，从而表示这些工作中提出的调度策略，同时启用更广泛的调度优化。

## 九、结论

> 随着 DNN 模型的不断增大以及专用异构硬件单元的出现，**硬件调度器已不足以实现高效的流水线执行**。

本文介绍了 PipeThreader，这是一个 DNN 编译器，它:mdnice-bold-italic[通过 sTask-graph 抽象和层次化的硬件能力] （包括虚拟化的 EU 和专用的 sEU）:mdnice-bold-italic[来实现软件定义的流水线]。**借助关键的调度原语，PipeThreader 自动化了流水线调度**，在 NVIDIA H100 和 AMD GPU 上实现了与现有最佳方法（如 FlashAttention）相当甚至更优的性能，同时推广到新兴模型（如 Mamba2）。

我们相信 PipeThreader 为基于编译器的优化的进一步发展奠定了基础，为不断发展的 GPU 架构和 DNN 工作负载的高效利用铺平了道路。

## 参考文献

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/25.webp)

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/26.webp)

![文章插图](/articles/2025-07-15-pipethreader-dnn-flashattention-mamba2-5c3d9639/images/27.webp)
