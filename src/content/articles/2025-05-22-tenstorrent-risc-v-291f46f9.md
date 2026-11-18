关键词：RISC-V、Tenstorrent Grayskull、Matrix Multiplication、Hardware Acceleration、Energy Efficiency

> 对生成式人工智能（如大型语言模型，LLMs）服务的需求不断增加，推动了对专门硬件架构的需求，这些架构旨在优化计算效率和能耗。

::: block-1
本文**评估了 Tenstorrent Grayskull e75 RISC-V 加速器** 在**低精度下对矩阵乘法的性能表现**，矩阵乘法是大型语言模型计算中的一个基本操作。文中详细描述了 Grayskull 的**执行模型、网格大小、矩阵维度、数据格式** 以及**数值精度对计算效率的影响**。
:::

此外，我们**将 Grayskull 的性能与支持张量加速的最新架构进行了比较**，包括英特尔 Sapphire Rapids 处理器和两款英伟达 GPU（V100 和 A100）。尽管英伟达 GPU 在原始性能上占据主导地位，但 **Grayskull 在功耗和计算吞吐量之间展现出了具有竞争力的平衡**，使用 **BF16 格式时达到了 1.55 TFLOPs/Watt 的峰值**。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/01.webp)

- https://arxiv.org/abs/2505.06085

## 本文目录

- 一、引言
- 二、研究背景
  - Tenstorrent 加速器
- 三、相关工作
- 四、Tenstorrent 软件栈和 Tensix 核心的软件抽象
  - 4.1 TTNN Python 库：提供类 PyTorch API 接口
  - 4.2 软件抽象
- 五、实验结果
  - 5.1 实验设置
  - 5.2 卸载和执行模型
  - 5.3 MatMul 内核性能
  - 5.4 优化的 MatMul 内核
  - 5.5 能效
- 六、结论
- 参考文献

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/02.webp)

加群请在 NerualTalk 公众号后台回复：加群

## 一、引言

基于 Transformer 架构的大型语言模型（LLMs）在各种自然语言处理（NLP）任务中取得了最先进的（SoA）结果。它们能够泛化多种任务的能力与模型的大小有关，这使得它们即使对于未直接训练的任务也能取得出色的结果。

随着对 LLMs 服务需求的增加，对能够在性能和能耗方面高效使用的架构的需求也在增加。像 GPT-4 这样的模型拥有数十亿甚至数千亿个参数，**需要大量的内存和计算资源，这使得高效部署成为一个重大挑战**。从计算角度来看，**Transformer 和深度神经网络（DNN）中的大部分工作负载由矩阵-矩阵乘法（MatMul）操作组成**。

> 因此，这些操作的效率取决于底层硬件架构和具体的数据移动策略，这可能会极大地影响整体性能。

在文献中，用于 LLMs 执行的各种架构从通用设备（如 CPU 和 GPU）到更专业的加速器（如基于 FPGA 的设备）不等。**尽管 GPU 目前主导市场，但各种研究都在探索新的架构以进一步提升计算性能并降低能耗**。一些新的解决方案示例包括 Sambanova [5]、Groq [1]、Cerebras [3] 和 Tenstorrent [7]。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/03.webp)

**特别是，Tenstorrent 生产基于 RISC-V 的加速器，专门用于加速 AI 工作负载。它们的产品系列涵盖了从轻量级工作负载到大规模计算密集型应用的广泛需求**。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/04.webp)

其架构灵感来自于将 AI 模型视为图（graph）的想法，**通过将图的组件组织成处理器网格来开发架构**。这种安排**允许数据在各种操作之间轻松流动，最大化计算与通信之间的重叠**，从而成为一个值得进一步探索的有前景的解决方案。

::: block-1
在本文中，我们提出了以下内容：

1. 对 Tenstorrent Grayskull e75 加速器**在不同配置下执行 MatMul 内核的性能和功耗进行测试**，**讨论执行模型** 并展示获得的结果。
2. 我们的实验表明，首次执行计算内核与后续执行之间存在显著的性能差异。我们的评估显示，在**首次运行中，执行时间主要由矩阵分块（tiling）和矩阵乘法内核的编译时间主导，分别占总时间的 31% 和 66%**。相比之下，** 后续运行主要由数据传输时间主导** （62%）。
3. 仅考虑内核执行时间下，我们**观察了处理器网格大小、矩阵维度、数据格式和数值精度对计算效率的影响**。我们的结果突出了基于不同配置的可实现性能的显著差异。
4. **与最新的架构** （即英特尔 Sapphire 处理器、英伟达 V100 GPU 和英伟达 A100 GPU）**进行比较**，展示了 Tenstorrent 加速器的显著效率。
:::

## 二、研究背景

### Tenstorrent 加速器

> Tenstorrent 开发了一系列基于相同架构的加速器，其中 **Grayskull e75 是该系列中最小的加速卡**。其架构**由 96 个 Tensix 核心组成的网格构成**，**每个核心** 的设计**将通信组件与计算组件分开**，以提高效率。

采用 12 纳米工艺制造，该卡具有八个 LPDDR4 内存通道（位于处理器网格的顶部和底部），**提供总共 8 GB 的容量和 102.4 GB/秒的带宽**。以 1 GHz 的频率运行，它**提供了 55 TFLOPs 的半精度浮点峰值性能**。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/05.webp)

虽然 Grayskull e75 的峰值性能较低，但在 Wormhole 系列中，更大的 Tensix 核心网格在板级和系统级互连，从而扩展了性能。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/06.webp)

::: block-1
具体来说，每个 Tensix 核心由五个可编程的“小型”RISC-V 核心、1 MB 的 Local SRAM 内存（也称为 L1 内存）、一个 SIMD 矩阵与向量引擎以及片上网络（NoC）路由器组成，如图 1 所示。

- **第一个和最后一个核心（RISC-V 0 和 RISC-V 4）执行数据移动内核，管理对其他核心的 SRAM、外部 DRAM bank和 Local SRAM 的异步读写操作**。
- 剩下的三个核心与矩阵与向量引擎接口，执行计算内核。
:::

更具体地说，**这三个计算核心分别执行不同的任务：第一个核心解包数据，第二个核心执行计算内核，第三个核心处理数据打包**。Tensix 核心内的核心使用 SRAM 中的循环缓冲区（CB）进行通信，如图 1 所示，存储来自外部存储器的中间结果。

**Tenstorrent 的 Grayskull 支持多种数据格式**。除了标准的浮点格式外，它还支持脑浮点（BF）和块浮点（BFP）。BFP 允许存储由 16 个元素组成的块，通过将它们组合在一个共享的公共指数下，减少内存占用并提高性能。BFP 格式支持 4、8 和 16 位配置。

此外，它**支持四种数学精度级别，这些级别通过指定用于计算操作的位数来确定计算精度。这些级别从最低精度（Low Fidelity）到最高精度（High Fidelity 4）**。较低的精度级别只消耗操作数尾数的最高有效位（MSB），而最高的 High Fidelity 4 则消耗操作数的所有位。中间级别 High Fidelity 2 和 3 分别消耗第一个输入的 MSB 和第二个输入的最低有效位（LSB），反之亦然。较高的精度级别需要更多的计算位，导致每个操作的周期数增加，从而降低 FLOPs。

## 三、相关工作

> 尽管许多研究工作已经对最新 CPU 和 GPU 架构的性能进行了实验，但只有少数研究分析了 Tenstorrent 架构的效率，并与其他当前的张量代数加速技术进行了比较。

在文献 [2] 中，作者探索了在 Grayskull e150 上实现的用于求解拉普拉斯方程的雅可比迭代方法，并将其结果与英特尔至强铂金 CPU 进行了比较。**他们的工作重点在于优化，探索和比较利用低级 C++ 内核的最佳数据访问策略。他们实现了与英特尔 CPU 相当的性能，同时能耗仅为英特尔 CPU 的五分之一**，尽管英特尔 CPU 运行在 FP32 而 Grayskull 运行在 BF16 模式下。

在文献 [6] 中，展示了通过利用 SRAM 核心内存，使用融合内核优化注意力层，显著提高了速度。

此外，Tenstorrent 在其文档中发布了一份报告，展示了其设备 Wormhole 上的 MatMul 内核的绝对性能。尽管有这些初步的实验，但**目前尚缺乏对简单密集张量代数内核与具有张量加速功能的最新架构的比较分析**。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/07.webp)

> 在本文中，我们提供了在 Tenstorrent Grayskull 加速器上运行的 MatMul 内核的分析，并将其性能与其他最新的计算设备进行了基准测试和能效比较。

## 四、Tenstorrent 软件栈和 Tensix 核心的软件抽象

本节描述了 Tenstorrent 软件栈和 Tensix 核心的软件抽象，并展示了一些代码示例。

### 4.1 TTNN Python 库：提供类 PyTorch API 接口

> TTNN Python 库是 Tenstorrent 的开源库，基于 TT-Metal 软件栈构建。它提供了**类似于 PyTorch 的 API，以函数式风格实现了许多 PyTorch 操作**。在底层，TTNN 利用 C++ 内核执行操作，同时提供用户友好的 API。

TTNN 的执行模型包括五个步骤：

1. **程序初始化**：
2. **创建 DRAM 缓冲区用于数据存储**。
3. **将数据从主机加载到设备的 DRAM 中**。
4. **程序执行**。
5. **将结果存储回 DRAM**。

### 4.2 软件抽象

与 PyTorch 相比，**TTNN 要求用户管理 Tenstorrent 的内存和计算配置**。为此，它提供了对数据在底层硬件上的存储方式的控制，允许用户指定目标内存（DRAM 或 L1）和张量内存布局。

#### 4.2.1 张量布局

张量在内存空间中作为二维对象存储，通过展平外部维度。例如，维度为 [1x2x4x8] 的张量被存储为 [8x8]。**在最低级别，包含张量部分的内存块称为页面**。

有两种可能的方式将张量映射到其页面。

- 在行主布局中，每一行被分配到一个单独的页面，从上到下逐行存储张量。
- 或者，张量可以被分块，即存储在固定大小的块中，默认的 Tenstorrent 分块大小为 32x32。

为了用于 MatMul，张量必须在分块布局中。

#### 4.2.2 内存布局

**张量的页面可以使用两种机制存储在内存中：交错或分片**。

1. 在**交错配置中，张量被分成多个页面，然后以循环方式分布在不同的内存 bank 中**。这是 DRAM 和 L1 内存的默认存储方式。
2. 在**分片内存配置中，张量被分成分片，并根据指定的映射分布在 L1 核心的内存中**。这种方法允许用户定义数据在处理网格上的特定分布，确保每个核心对其 L1 内存中的数据具有 Local 访问权限。可以定义分片策略（在高度、宽度或按块）和方向，这决定了分片在网格上放置的顺序。

#### 4.2.3 代码示例

在代码清单 1.1 中，展示了如何将矩阵分配到设备上，设置其布局、存储策略和格式。使用这种方法，还可以应用更高级的内存策略，如分片布局，如列表 1.2 所示，这可以传递给 `from_torch()` 方法以应用于输入。

```
代码清单 1.1：输入卸载1 import torch2 import ttnn3 device = ttnn.open_device(device_id=0)4 in0 = torch.randn((512, 512))5 in0_t = ttnn.from_torch(6     in0,7     device=device,8     tile=ttnn.Tile((32, 32)),9     layout=ttnn.TILE_LAYOUT,10    memory_config=ttnn.DRAM_MEMORY_CONFIG,11    dtype=ttnn.bfloat16 )
```

```
代码清单 1.2：内存配置定义1 memory_config = ttnn.create_sharded_memory_config(2     (1, 1, 512, 512),3     core_grid=ttnn.CoreGrid(y=8, x=8),4     strategy=ttnn.ShardStrategy.BLOCK,5     orientation=ttnn.ShardOrientation.ROW_MAJOR,)
```

在代码清单 1.3 中，展示了如何执行简单的 MatMul 内核，以及精度配置。可以通过向程序`_config` 参数传递内核配置来执行更高级的内核。

```
代码清单 1.3：MatMul 内核执行1 output = ttnn.matmul(2     in0_t,3     in1_t,4     dtype=ttnn.bfloat16,5     memory_config=ttnn.DRAM_MEMORY_CONFIG,6     compute_kernel_config=ttnn.GrayskullComputeKernelConfig(7         math_fidelity=ttnn.MathFidelity.HiFi4),8     # program_config=...)
```

## 五、实验结果

> 在本节中，我们展示了 Tenstorrent Grayskull 在执行矩阵乘法内核时的效率、能耗和性能的实验结果，并将其与其他最新的计算系统进行了比较，这些系统包括针对矩阵乘法和张量线性代数优化的两个英伟达 GPU 和英特尔至强铂金 8480+ 处理器（来自 Sapphire Rapids 服务器系列）。

**测试在 Grayskull e75 上进行，其峰值性能为 55 TFLOPs（BF16）。Grayskull e75 是 Tenstorrent 最经济的加速器，专为边缘计算设计**。

在 Tenstorrent 的产品线中，其他加速器（如 Wormhole 卡）专为大规模计算设计，提供高达 131 TFLOPs（BF16）的峰值性能，并可集群化以构建高性能计算集群。

### 5.1 实验设置

Grayskull e75 通过 PCIe 4.0 x16 连接到运行 Ubuntu 20.04 的英特尔酷睿 i7 Coffee Lake 主机。**为了分析 Tenstorrent 加速器，我们进行了多种测试，以探索不同条件下矩阵乘法的执行情况**：

- **卸载和执行**：分析 Tenstorrent 执行模型，比较首次执行（需要编译计算内核）与后续执行。
- **MatMul 内核性能**：在不同配置下评估性能，包括不同的数据格式、数学精度和网格核心选择。
- **优化的 MatMul 内核**：评估**利用核心 L1 内存和更复杂内核的优化 MatMul 实现的性能提升**。
- **能效**：测量和比较相对于性能的功耗。

> - 使用 pynvml Python 模块测量 GPU 的功耗
> - 使用 Tenstorrent 的遥测工具 TT-SMI 测量 Grayskull 的功耗
>
> 这两种工具都报告瞬时功耗，我们在计算过程中对其进行了平均。

### 5.2 卸载和执行模型

在 Tenstorrent Grayskull 上首次执行内核需要编译程序，这与执行本身相比需要大量的时间。**图 2 显示了首次运行和后续运行的时间**。从首次运行可以看出，编译时间在首次执行内核时会导致显著的开销。我们可以注意到，**首次运行的执行时间主要由分块和 MatMul 内核运行主导**。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/08.webp)

图 2：首次运行和后续运行的时间，对于不同矩阵维度。柱状图显示了所需步骤的时间：将两个输入传输到设备、从行主布局到分块布局的分块，以及 MatMul 运行。

首次运行后，程序不需要重新编译，从而在后续运行中显著加快了分块和 MatMul 操作的速度。**通过计算首次运行和后续运行之间的编译时间差异，我们得出分块内核需要 296 毫秒来编译，与矩阵维度无关**。而分块执行时间从 256x256 矩阵的 351 微秒到 4096x4096 矩阵的 2256 微秒不等。

另一方面，MatMul 内核需要 620 毫秒来编译，执行时间从 328 微秒到 4783 微秒不等。**在后续运行中，从主机到设备的数据传输时间成为主要开销，平均占总时间的 62%。** 然而，与编译一样，当矩阵位于主机中时，才会产生成本。一旦数据加载到设备上，操作就可以在没有传输成本的情况下执行。

> 因此，在接下来的实验中，我们将**只考虑 MatMul 内核的执行时间，假设数据已经加载在设备上**。

### 5.3 MatMul 内核性能

下面表 1中测试了不同配置下的矩阵乘法性能：

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/09.webp)

```
# 关于"Configuration Name" 中的 Math Fidelity"FP32 M4" 中的 "M" 指的是 "Math Fidelity"（数学精度）。这是一个衡量【计算过程】中的精度指标，它通过指定用于计算操作的位数来确定。在本文中，"M4" 表示 "High Fidelity 4"，意味着在计算过程中使用了所有位（即最高精度），这通常涉及到使用更多的计算资源来保持更高的数值精度。这种高精度配置对于需要高精度计算的应用场景非常重要，但可能会增加计算的能耗和时间。
```

上面表格对应下面图 3，图 3 是使用不同配置和网格大小的性能。我们测试并比较了表 1 中报告的一组配置，结合了不同的数据类型和精度。其中：

- (a) 使用表 1 中的不同配置获得的 TFLOPs。
- (b) 使用不同核心集（从 1 个核心到 64 个核心）的 MatMul 加速比，使用 BF16 M4 配置。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/10.webp)

在图 3 (a) 中，我们比较了不同配置下的 MatMul TFLOPs。正如预期的那样，性能随着数据格式的变长和数学精度的提高而下降。对于较大的矩阵维度，性能从使用 32 位浮点数和最高数学精度的 14.72 TFLOPs 到使用 4 位块浮点数和低精度的 49.78 TFLOPs 不等。**考虑到使用 BF16 的理论峰值为 55 TFLOPs，当前结果仍远未达到最佳效率**。

在**接下来的部分中，我们将探讨利用内部 L1 SRAM 和供应商优化的 MatMul 内核的 MatMul 内核优化**。

**图 3 (b) 显示了使用不同核心集（从 1 个核心到 64 个核心）的 MatMul 加速比**。

> 如图所示，**较小的矩阵尺寸可以用少量核心达到性能饱和**，而**对于较大的矩阵**，可以使**用更大的核心集实现几乎线性的加速比**，使用 64 个核心达到 56 倍的加速比。

### 5.4 优化的 MatMul 内核

除了默认的 MatMul 配置外，Tenstorrent 的**软件栈允许通过利用高级内核进一步优化性能**，这些内核使用分片内存配置，将矩阵分布在多个核心的 Local SRAM 上。

在之前的实验中，两个输入矩阵都使用交错内存存储策略分配在 DRAM 中。然而，**性能可以通过分片一个输入矩阵并将其分片分布到计算核心的 L1 内存上来进一步提高**。利用这种存储策略和中间数据重用的内核是 `MatmulMultiCoreReuseMultiCast` 内核，它实现了最高的性能。

下面图 4 中，是我们使用默认内核配置、分片内存配置和优化内核（BF16 配置下高精度 2 和 4）实现的 TFLOPs 的比较。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/11.webp)

- 在**图 4a 中，我们比较了默认的 MatMul 内核** （两个输入都使用交错配置存储在 DRAM 中，如蓝色所示），**分片内存配置（黄色）和优化内核**。对于较小的矩阵尺寸，优化内核和 L1 内存配置表现出相似或略低的性能，这是由于分片内存的开销。对于 BF16 M2 配置和 2048x2048 MatMul（这是可以存储在 L1 核心内存中的矩阵集中的最大矩阵尺寸），内存配置的优势尤为明显。**从图中可以看出，对于超过 L1 内存容量的较大矩阵尺寸，共享内存优化不再有效，但优化内核仍有边际性能提升**。
- 从**图 4b 中可以看出，使用 BF16 M4 配置时，分片内存优化提供了与 M2 配置相似的性能提升**。然而，在这种情况下，优化内核似乎并不特别有效。 我们将使用性能最高的优化内核进行后续比较。

#### 5.4.1 与英伟达 A100 的性能比较

**在 PyTorch 的 MatMul 内核中与下面设备进行了比较：**

- 英伟达 A100 SXM4 40GB（BF16 峰值吞吐量为 312 TFLOPs）
- 英伟达 V100S PCIe 32GB（FP16 峰值吞吐量为 32 TFLOPs）
- 英特尔 Sapphire Rapids 服务器处理器（BF16 峰值吞吐量为 229 TFLOPs）

性能比较结果如图 5 所示。其中：

- Sapphire Rapids 英特尔处理器，配备了英特尔高级矩阵扩展（AMX），以优化 BF16 和 INT8 格式的 MatMul 执行，这些功能被 PyTorch 的 MatMul 内核利用。
- 英伟达 GPU 通过引入 Tensor Core 在 BF16 和 FP16 格式下加速 MatMul 计算，从而实现了显著的性能提升。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/12.webp)

- **图 5a 所示，两个英伟达 GPU 在每个矩阵尺寸下都占据性能主导地位，其次是 Grayskull 加速器和 Sapphire Rapids 处理器**。尽管这些设备设计用于不同的市场细分，但该比较可以作为其他解决方案的参考。然而，Grayskull 在原始性能上超过了英特尔 Sapphire Rapids 处理器。
- **图 5b 显示了相对峰值性能的效率比，作为实现的 TFLOPs 与理论峰值 TFLOPs 的百分比**。对于较小的矩阵，计算效率有限，但随着矩阵尺寸的增加而提高。Grayskull 的效率与其他设备相当，在测试集中最大的矩阵尺寸（2048x2048）中达到 79.36% 的峰值，该矩阵尺寸适合 L1 核心内存并可以从分片内存优化中受益。

### 5.5 能效

能效是当今系统可持续性、可扩展性和运营成本的关键因素。

> 内存访问和算术操作是功耗和能耗的最相关因素。虽然可以通过先进制程和降低数值精度来降低算术操作的能耗，但**内存访问能耗强烈依赖于所使用的内存技术和内存设备与计算单元之间的距离。**

Tenstorrent Grayskull 架构通过其 Tensix 核心网格将数据存储在接近处理单元的地方，以实现更高的能效并减少内存访问时间。

![文章插图](/articles/2025-05-22-tenstorrent-risc-v-291f46f9/images/13.webp)

**图 6 显示了所研究设备实现的每瓦特 TFLOPs**。与之前的考虑一致，Grayskull 实现了最高的效率，在 BF16 M2 配置下达到 1.56 TFLOPs/Watt 的峰值。**值得注意的是，在这种特定场景之外，当使用最高的数学精度时，Grayskull 的 TFLOPs/Watt 比率低于英伟达 A100，但在降低精度运行时超过了它。**

## 六、结论

在本文中，我们**评估了 Tenstorrent** Grayskull e75 **RISC-V 加速器在执行矩阵-矩阵乘法（MatMul）方面的性能和效率**，这是深度学习中的一个基本操作。我们的**分析了其执行模型，揭示了首次运行和后续运行之间的显著差异**，这是由于编译和数据移动开销所致。

我们**研究了处理器网格大小、矩阵维度、数据格式和数值精度对计算性能的影响。**

> 结果表明，Grayskull 在每瓦特 TFLOPs 方面实现了与英伟达 GPU（A100 和 V100）和英特尔 Sapphire Rapids 处理器相当的性能。

尽管 GPU 提供了更高的原始吞吐量，但 **Grayskull 提供了一个有希望的替代方案**，在性能和能效之间取得了良好的平衡。总体而言，**本研究突出了基于 RISC-V 的加速器在加速 AI 工作负载方面的潜力**，并为**高效的 AI 硬件设计的讨论** 做出了贡献。

## 参考文献

::: block-1
1. Abts, D., Kimmell, G., Ling, A., Kim, J., Boyd, M., Bitar, A., Parmar, S., Ahmed, I., DiCecco, R., Han, D., Thompson, J., Bye, M., Hwang, J., Fowers, J., Lillian, P., Murthy, A., Mehtabuddin, E., Tekur, C., Sohmers, T., Kang, K., Maresh, S., Ross, J.: A software-defined tensor streaming multiprocessor for large-scale machine learning. In: Proceedings of the 49th Annual International Symposium on Computer Architecture. p. 567–580. ISCA ’22, Association for Computing Machinery, New York, NY, USA (2022). https://doi.org/10.1145/3470496.3527405, https://doi.org/ 10.1145/3470496.3527405
2. Brown, N., Barton, R.: Accelerating stencils on the tenstorrent grayskull risc-v accelerator (2024), https://arxiv.org/abs/2409.18835
3. Dey, N., Gosal, G., Zhiming, Chen, Khachane, H., Marshall, W., Pathria, R., Tom, M., Hestness, J.: Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster (2023), https://arxiv.org/abs/2304.03208
4. Mutlu, O., Ghose, S., Gómez-Luna, J., Ausavarungnirun, R., Sadrosadati, M., Oliveira, G.F.: A modern primer on processing in memory (2025), https://arxiv. org/abs/2012.03112
5. Prabhakar, R., Sivaramakrishnan, R., Gandhi, D., Du, Y., Wang, M., Song, X., Zhang, K., Gao, T., Wang, A., Li, X., Sheng, Y., Brot, J., Sokolov, D., Vivek, A., Leung, C., Sabnis, A., Bai, J., Zhao, T., Gottscho, M., Jackson, D., Luttrell, M., Shah, M.K., Chen, Z., Liang, K., Jain, S., Thakker, U., Huang, D., Jairath, S., Brown, K.J., Olukotun, K.: Sambanova sn40l: Scaling the ai memory wall with dataflow and composition of experts. In: 2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO). p. 1353–1366. IEEE (Nov 2024). https://doi.org/10. 1109/micro61859.2024.00100, http://dx.doi.org/10.1109/MICRO61859.2024.00100
6. Thüning, M.: Attention in sram on tenstorrent grayskull (2024), https://arxiv.org/ abs/2407.13885
7. Vasiljevic, J., Bajic, L., Capalija, D., Sokorac, S., Ignjatovic, D., Bajic, L., Trajkovic, M., Hamer, I., Matosevic, I., Cejkov, A., Aydonat, U., Zhou, T., Gilani, S.Z., Paiva, A., Chu, J., Maksimovic, D., Chin, S.A., Moudallal, Z., Rakhmati, A., Nijjar, S., Bhullar, A., Drazic, B., Lee, C., Sun, J., Kwong, K.M., Connolly, J., Dooley, M., Farooq, H., Chen, J.Y.T., Walker, M., Dabiri, K., Mabee, K., Lal, R.S., Rajatheva, N., Retnamma, R., Karodi, S., Rosen, D., Munoz, E., Lewycky, A., Knezevic, A., Kim, R., Rui, A., Drouillard, A., Thompson, D.: Compute substrate for software 2.0. IEEE Micro 41(2), 50–55 (2021). https://doi.org/10.1109/MM.2021.3061912
8. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need (2023)
:::
