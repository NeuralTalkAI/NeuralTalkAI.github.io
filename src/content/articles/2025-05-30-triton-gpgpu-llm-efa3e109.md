> 为大型语言模型（LLM）提供服务对于人工智能驱动的应用程序至关重要，但这需要大量的计算资源，特别是在**内存带宽和计算吞吐量** 方面。

低精度计算已成为提高效率和减少资源消耗的关键技术。**现有的低精度内核生成** 方法**仅限于支持 2 的幂次位宽的权重**，并且由于高层 GPU 编程抽象的限制，其**性能并不理想**。这些**抽象限制了关键优化**，例如细粒度的**寄存器管理** 和优化的**内存访问模式**，这些对于高效的低精度计算至关重要。

::: block-1
在本文中，我们介绍了一个**为通用图形处理器（GPGPU）计算设计的虚拟机（VM）**，它**支持任意位宽的低精度数据类型**，同时**保持 GPU 的可编程性**。

该虚拟机具有**基于线程块的编程模型、层次化内存空间、新颖的代数布局系统**，并广泛**支持多种低精度数据类型**。VM 程序被编译成高效的 GPU 程序，具有**自动向量化和指令选择功能**。
:::

广泛的实验表明，我们的 VM 能够**高效地支持全系列的低精度数据类型**，并且在支持的类型上**超越了现有的低精度内核**。与现有的编译器（如 Triton 和 Ladder）以及手工优化的内核（如 QuantLLM 和 Marlin）相比，我们的 **VM 分别实现了 1.75 倍、2.61 倍、1.29 倍和 1.03 倍的性能提升**。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/01.webp)

- https://arxiv.org/abs/2504.12984

## 本文目录

- 一、引言
- 二、背景知识
  - 2.1 大型语言模型服务与量化
  - 2.2 GPGPU 编程
  - 2.3 GPGPU 语言与编译器
- 三、系统概述
  - 3.1 关键思想
  - 3.2 虚拟机程序示例
- 四、代数布局系统
  - 4.1 参数化原始布局
  - 4.2 布局组合
- 五、线程块级编程模型
  - 5.1 状态空间和类型系统
  - 5.2 程序结构和控制流
  - 5.3 线程块级指令集
- 六、任意低精度数据类型
  - 6.1 低精度数据的存储
  - 6.2 LLM 中高效的低精度支持
- 七、实现 Tilus
  - 程序编译和运行时
  - 步骤3：每个指令的代码生成
  - 步骤4：降低低精度数据类型
  - 虚拟机运行时
- 八、评估
  - 8.1 实验设置
  - 8.2 低精度内核的性能
  - 8.3 任意数据类型支持
  - 8.4 端到端性能
  - 8.5 案例研究
- 九、相关工作
- 十、结论
- 参考文献

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/02.webp)

加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

> 大型语言模型（LLM）的发展彻底改变了自然语言处理任务，在文本生成[8]、摘要[28]、翻译[53]和对话式人工智能[38]等领域实现了先进功能。然而，为 LLM 提供服务带来了巨大的计算挑战，因为模型体积庞大且计算需求高。

为了在延迟和功耗限制下高效地为 LLM 提供服务，优化 LLM 推理已成为行业和研究中的优先事项。量化[7, 9, 17, 27, 29, 49]已成为提高 LLM 服务效率的主要方法。通过减少模型参数和激活的位宽，量化减少了权重存储、DRAM 带宽使用，并实现了更快的计算。例如，A16W4 量化（16 位激活和 4 位权重）与 A16W16 方案相比，将 DRAM 消耗和吞吐量减少了 4 倍，从而将生成一个标记的时间缩短了约 4 倍[18]。

然而，最先进的 4 位量化方法[7, 9, 29]仍然存在不容忽视的准确性下降问题。尽管使用 5 到 7 位量化可以缓解这种准确性损失[3, 54]，但缺乏对这些位宽的高效 GPU 内核支持限制了它们的采用。为硬件不友好的位宽（例如 3、5、6 和 7 位）生成优化内核仍然是一个未解决的问题。

::: block-1
**现有的内核生成方法主要分为两类：手动编写的内核[18, 54]和编译器生成的内核[10, 47, 52]。** 尽管手动编写的内核针对特定硬件高度优化，但它们开发起来耗时且容易出错，并且难以推广到新的架构和不断发展的量化方法。例如，QuantLLM[54]仅支持 5 位和 6 位数据的浮点量化，但不支持子通道量化粒度。同样，Marlin[18]仅限于 4 位有符号整数量化，并且不支持 Hopper GPU[30]。

为了解决这些限制，提出了基于编译器的方法[10, 47, 52]以自动化内核生成。其中，Triton[47]通过基于块的模型简化了 GPGPU 编程。Triton 程序定义了线程块在张量块上的计算。然而，**Triton 缺乏对低精度数据类型的内置支持，需要用户手动实现低级位操作**。此外，它**没有完全暴露 GPU 的内存层次结构，限制了低精度 LLM 推理的优化机会**。

Ladder[52]则扩展了 TVM 的调度系统[10]以支持低精度计算，但它仅限于位宽为 2 的幂次的数据类型。此外，其原语无法表达诸如软件流水线[22]等关键优化，导致在批处理大小大于一且启用了连续批处理[57]的 LLM 解码期间性能不佳。
:::

**为了解决这些挑战，我们提出了 Tilus，一个具有专门支持低精度计算的 GPGPU 虚拟机**。Tilus**将 GPU 程序执行抽象为线程块级指令，简化了 GPGPU 编程，同时暴露了层次化内存空间，以精细地操作片上内存中的子张量**。这种双重方法使得能够**高效地处理任意精度的数据类型**，同时**降低 GPU 编程的复杂性**。

为了实现这些目标，Tilus 引入了：

1. **一个代数布局系统**，用于**指定张量元素在块内的分布方式**。该布局系统使得能够灵活地将低精度块在寄存器中重新解释为具有硬件友好数据类型的块，从而实现高效处理；
2. **一个具有精细内存管理的线程块级编程模型**，提供了**对 GPU 内存层次结构中不同级别的数据移动、放置和计算的显式控制** 以及广泛支持；
3. 包括有符号整数、无符号整数和浮点数在内的**任意位宽（1 到 8 位）的低精度数据类型**。

广泛的实验表明，**Tilus 扩展了高效低精度内核的范围，支持任意位宽（1-8 位）和数据类型种类** （例如整数和浮点数），并且在支持的低精度内核上分别比最先进的编译器 Triton[47]、Ladder[52]和手工制作的内核 QuantLLM[54]和 Marlin[18]快 1.75 倍、2.61 倍、1.29 倍和 1.03 倍。

我们总结关键贡献如下：

- 我们提出了一个具有**专门支持低精度** 计算的 **GPGPU 虚拟机**，解决了现有方法在覆盖范围（例如**支持 5 到 7 位量化**）和性能差距方面的问题。
- 在虚拟机内部，我们引入了一个**新颖的布局系统**、**具有层次化内存空间的线程块级编程模型**，以及**对任意位宽的低精度数据类型的支持**。
- 通过广泛的评估，我们证明 Tilus 生成了一系列高效的低精度内核，**在支持的内核上比最先进的解决方案快达 2.6 倍**。

## 二、背景知识

### 2.1 大型语言模型服务与量化

> 在大型语言模型（LLM）服务中，推理过程包括两个阶段：预填充（prefill）和解码（decode）。预填充阶段处理输入提示以建立上下文，而解码阶段则基于先前的标记迭代生成输出标记。

在 LLM 的所有层中，矩阵乘法占主导地位，其计算时间和内存消耗巨大，因此对它们的优化对于高效的 LLM 服务至关重要。量化[11, 17]通过将模型权重和激活减少到更低精度的格式（例如 8 位或 4 位整数）来提高效率。它减少了内存使用、带宽需求，并尝试在保持模型准确性的同时减少推理延迟。

**尽管 4 位量化提供了显著的计算节省，但最先进的方法** [7, 9, 29]**仍然存在不容忽视的准确性下降问题**。**增加精度到 5 位、6 位或 7 位量化[3, 54]可以帮助在保持效率的同时保留准确性**，但这些位宽**缺乏优化的 GPU 支持**，限制了它们的采用。

当前的 GPU 架构和软件堆栈**主要针对 2 的幂次位宽** （例如 4 位和 8 位）进行优化，使得**非标准位宽的计算效率低下**。然而，灵活的量化需求不断增长，因为 4 位对于某些模型来说可能过于激进，而 8 位则浪费资源。支持更广泛的位宽范围可以实现更好的准确性与效率权衡，推动对能够高效处理非标准低精度格式（例如 3、5、6、7 位宽）的新内核生成技术的需求。

### 2.2 GPGPU 编程

> 通用图形处理器（GPGPU）编程通过组织任务在结构化的执行和内存层次结构中来实现并行计算[32]。

- 执行层次结构从线程开始，线程是最小的执行单元，具有独立的指令执行和寄存器及本地内存。
- 线程被分组为线程块，线程块内的线程可以通过共享内存进行数据共享，并支持同步执行。
- 网格（grid）由多个独立的线程块组成，通过组织数千或数百万个线程来实现大规模并行性。

GPGPU 内存层次结构包括寄存器、共享内存和全局内存。寄存器提供最快且线程私有的存储。共享内存可被线程块内的所有线程访问，速度比全局内存快。全局内存可被整个网格访问，但具有较高的延迟。

这种结构通过利用执行和内存层次结构实现了高效的并行执行。

### 2.3 GPGPU 语言与编译器

#### 2.3.1 GPGPU 虚拟机与语言

> GPGPU 编程涉及多种语言和编译器，它们在硬件抽象与控制之间取得平衡。低级语言如 SASS[37]和 CDNA3[5]提供了直接的硬件访问，用于细粒度优化，但需要深入的架构知识。

稍高抽象层次的 NVIDIA 的 PTX[36]作为中间表示，将高级语言（例如 CUDA）与特定于 GPU 的指令链接起来，同时保留优化的灵活性。高级语言如 CUDA[35]和 HIP[6]通过扩展 C 编程语言简化了编程。尽管如此，GPGPU 编程仍然复杂。它**受到硬件特定的内存和计算层次结构的限制，并需要针对工作负载的特定优化**。

**为了解决这些挑战，研究人员引入了更高层次的语言和编译器**，分为**两类**：

- **过程导向型编译器**，通过**超出 CUDA 的抽象简化编程**；
- 以及**调度导向型编译器**，通过**声明式调度原语优化计算与硬件的映射**。

#### 2.3.2 过程导向型编译器（Procedure-Oriented Compilers）

这种类型的编译器[12, 20, 26, 47]**使程序员能够直接编写内核，并提供简化编程过程的抽象**。例如，Triton[47]引入了基于块的编程模型，其中线程块的行为由程序定义，块取代标量作为基本数据类型。

这种方法结合了编程的简单性和高性能内核的生成，使 Triton 被广泛采用。**然而，Triton 缺乏对低精度数据类型** （如 uint4）**的原生支持**。处理这些类型**需要手动从较大存储类型（例如 uint32）中解包子字节数据** [21]。

此外，**Triton 没有暴露 GPU 的内存层次结构，限制了程序员对数据加载和内存范围使用的控制**，从而使得低精度内核的性能优化变得复杂。这些限制导致了低精度内核执行的效率低下。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/03.webp)

**图 1（a）展示了 Triton 生成的低精度内核的权重加载流程**，以 uint4 权重加载管道为例。该流程包括四个步骤：

1. 使用管道化的 cp.async 指令将权重从全局内存异步复制到共享内存；
2. 将共享内存数据加载到寄存器；
3. 执行解包和类型转换操作；
4. 转换寄存器张量布局以满足张量核心指令的要求。

其中，**步骤 4 是一个主要瓶颈**，因为它**依赖于共享内存进行布局转换**，从而产生显著的开销。

#### 2.3.3 调度导向型编译器（Schedule-Oriented Compilers）

> **调度导向型编译器将计算与调度分离，以优化计算与硬件的映射**。Halide[40]率先采用了这种方法，TVM[10]以及后续的深度学习领域工作[16, 22, 43, 51, 52, 60, 61]对此进行了扩展。

其中，Ladder[52]是第一个支持低精度计算的编译器，通过引入专用原语将低精度数据（例如 4 位整数）打包到较大类型（例如 8 位整数）中。然而，Ladder 有两个限制。

- 首先，它无法高效处理非 2 的幂次位宽，因为其类型级打包将低精度类型打包到存储类型中。
- 其次，其原语风格的调度阻止了如软件流水线[22]等优化，导致性能不佳。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/03.webp)

**图 1（b）展示了 Ladder 的低精度内核中的权重加载过程**。该过程包括：

1. 从全局内存加载权重到寄存器，不进行流水线操作；
2. 向量化类型转换；
3. 将转换结果存储到共享内存；
4. 使用 ldmatrix 指令将权重从共享内存加载到寄存器，以供后续张量核心操作使用。

由于 Ladder **权重加载与计算之间缺乏流水线操作，性能受到显著限制**。

## 三、系统概述

### 3.1 关键思想

> **我们的工作引入了一种新颖的 GPGPU 虚拟机，专门设计用于克服编程高效低精度深度学习内核的挑战**。该虚拟机**原生支持任意位宽** （从 1 位到 8 位）的低精度数据类型，从而实现高效的权重加载和计算。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/03.webp)

**图 1（c）展示了虚拟机的权重加载流程，以 uint4 为例**。它首先从全局内存到共享内存进行流水线化的异步内存复制，然后从共享内存加载寄存器张量。接下来，它将寄存器张量重新解释为不同的数据类型和布局且无需成本，然后进行向量化类型转换。

**与图 1 中的其他方法相比，我们的流水线实现了更高的效率，因为它消除了布局转换** （与 Triton[47]不同）**并引入了流水线操作** （与 Ladder[52]不同）。

更重要的是，**我们的流水线是通用的，使我们的工作成为第一个无缝支持任意位宽** （从 1 位到 8 位）**的低精度数据类型的方法**。为了实现这种效率，我们的**设计基于以下几个关键思想**：

- **GPGPU 虚拟机的灵活性**：我们选择实现 GPGPU 虚拟机，以实现更大的 GPU 编程灵活性。与传统的基于循环的转换不同，我们的虚拟机**允许程序员直接实现和微调超出传统循环转换的优化**。**这种灵活性对于低精度计算至关重要**，因为对执行策略的精细控制可以带来显著的性能提升。
- **线程块级编程模型与层次化内存空间**：我们的虚拟机**显式地暴露了 GPU 的内存层次结构，包括寄存器、共享内存和全局内存**，而这些在现有的解决方案（如 Triton[47]）中是被抽象掉的。通过授予程序员对数据放置和移动的精细控制，我们的方法实现了内存流水线操作，并消除了不必要的布局转换，如图 1 所示。
- **代数布局系统**：我们引入了一个**代数布局系统，精确地定义了寄存器张量中的元素在块内线程中的分布方式**。这种结构化的表示简化了张量布局的构建、分析和解释。特别是，它使得低精度寄存器张量能够无缝地重新解释为标准数据类型，如图 1（c）中的步骤 3 所示。
- **对任意低精度数据类型的原生支持**：我们的虚拟机提供了对广泛低精度数据类型的内置支持，包括有符号整数、无符号整数和浮点数，位宽从 1 位到 8 位。支持的类型包括 int2 到 int8、uint1 到 uint8，以及 float3 到 float8，浮点数类型具有任意的指数和尾数分布。**这些创新共同增强了现代 GPU 上低精度内核开发的可编程性、效率和灵活性。我们选择不扩展 Triton[47]，因为其编程模型固有地抽象掉了张量布局，使其与我们对显式布局控制的方法不兼容**。同样，Ladder[52]依赖于类型级打包，而 **Tilus 采用块级重新解释，这使得两者从根本上不兼容**。

下一节将介绍虚拟机中的低精度矩阵乘法示例。

### 3.2 虚拟机程序示例

> 图 2 展示了虚拟机中的低精度矩阵乘法。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/04.webp)

矩阵乘法定义为 $A \times B=C$，其中 A 是浮点 16 位（f16）类型，B 是 6 位有符号整数（int6）类型。内核执行给定的 M、N 和 K 尺寸的矩阵乘法，每个线程块计算 C 矩阵的一个 $BM×BN$ 的块（Line 1）。因此，必须启动一个由 ($M / BM, N / BN$) 线程块组成的网格（Line 2）。

在内核内部，BlockIndices 指令检索线程块索引 bi 和 bj（Line 3），这些索引确定计算相应 C 块的偏移量 ($bi \times BM, bj \times BN$)。通过指定地址和形状创建输入和输出张量的全局内存张量视图（Line 4-6）。然后，创建一个 f32[16, 8]类型的寄存器张量，布局为：

`local(2, 1).spatial(8, 4).local(1, 2).`

它将 $16 \times 8=128$ 个元素分布在 32 个线程中，每个线程存储 4 个元素（Line 7）。此布局由三个原始布局组成（第 4 节），并与 PTX[36]中用于 C 矩阵布局的 `mma.m16n8k16` 张量核心指令对齐。K 维度上的约简循环（Line 8-13）反复从全局内存加载 A 和 B 的块到寄存器并执行矩阵乘法累加（mma）。

在每次迭代中，我们首先使用 LoadGlobal 指令从全局内存加载一个 f16[16, 16]块到寄存器（Line 9）。加载的寄存器块的布局由张量核心指令指定和要求。偏移参数指定了全局张量中加载块的位置。加载数据类型为 int6 的张量 B 涉及一个更复杂的过程，第 6 节将详细讨论。

**我们在此总结高层次思想**。

- 在启动内核之前，作为预处理步骤，权重张量在全局内存中的布局从 `i6[K, N]`转换为 `u8[K / BK, N / BN, BK * BN * 6 / 8]`，从而可以通过 LoadGlobal 指令高效加载（“更改布局”步骤）。
- 接下来，在内核中，转换后的块被加载到寄存器张量（Line 10）并重新解释为不同数据类型和布局的张量（Line 11）。这种重新解释是有效的，因为两个张量在相同的线程数（32）和每个线程持有的位数（24 位，例如 `3×u8` 或 `4×i6`）上是兼容的，如图 2（c）所示。
- 然后，将 i6 张量转换为 f16 张量（Line 12），并将其馈送到张量核心以执行矩阵乘法累加（mma）（Line 13）。
- 最后，将累积张量从 f32 转换为 f16 并存储在全局内存中（Line 14-15）。

为了简单起见，此程序未使用共享内存，并省略了诸如软件流水线[22]等优化；每个 k 迭代仅执行一个张量核心指令[35]。优化实现可在附录 B 中找到。

**以下各节将介绍所提出的虚拟机的三个核心组件**。

- **第 4 节介绍代数布局公式**，系统地定义块内线程中张量元素的存储方式。
- **第 5 节介绍显式暴露的基于线程块的编程模型和层次化内存空间**。
- **第 6 节介绍对任意低精度数据类型的原生支持**，以满足深度学习工作负载中日益增长的低精度计算需求。

## 四、代数布局系统

> 我们的虚拟机向程序员暴露了具有层次结构的内存空间，包括全局内存、共享内存和寄存器。

我们**需要一种方法来模拟张量元素的逻辑索引** 与**相应元素在内存中位置之间的映射**，**这就是所谓的张量布局**。

**虚拟机向程序员展示了一个具有全局内存、共享内存和寄存器的分层内存空间**。我们**需要一种方法来模拟张量元素的逻辑索引与所有三个内存范围内相应元素在内存中的位置之间的映射**。这种映射通常被称为张量的布局（layout）。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/05.webp)

**图 3 展示了一个张量核心指令**：`mma.m16n8k8.f32.f16.f16.f16.f32 D，A，B，C` 所使用的布局示例。它执行以下计算：

$$
D_{16,8} = A_{16,8}B_{8,8} + C_{16,8}
$$

其中 A、B、C、D 是存储在线程寄存器中并分布在一个 warp（线程块）中的 32 个线程上的张量。由于元素分布在不同的线程上，我们称这种布局为分布式布局[47]。这种布局可以定义为一个函数 f，它将线程索引 t 和线程内的局部索引 i 映射到相应元素的逻辑索引 $f(t, i)$。例如，**图 3 中的布局可以表示为**：

$$
f(t, i) = (t/4 + i/2 \times 8, t \% 4 \times 2 + i \% 2)
$$

这里，t 的范围是从 0 到 31，i 的范围是从 0 到 3。函数 $f(t, i)$ 表示线程 t 中元素 i 的逻辑索引。由于线程块中的所有线程都可以访问共享和全局内存，这种形式化也可以在单线程假设下用来描述它们的布局。也就是说，通过设置 t = 0，我们定义 $f(0, i)$ 作为共享或全局内存中地址 i 处元素的逻辑索引。

### 4.1 参数化原始布局

**在形式化定义布局的基础上，我们引入参数化原始布局作为布局代数的基本构建块**。给定一个形状为（n1，n2）的 tile（瓦片，即数据块），有以下两种主要存储方式：

1. 将所有 n1×n2 个元素**存储在一个线程中**
2. 将**所有元素分布在 n1×n2 个线程中**，**每个线程仅持有一个元素**。

我们**分别称这两种类型为局部布局** （local layout）**和空间布局** （spatial layout），分别表示为 local（n1，n2）和 spatial（n1，n2）。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/06.webp)

这种概念可以自然地扩展到任意维度的 tile。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/07.webp)

### 4.2 布局组合

> 现代深度学习工作负载中使用的布局，以及硬件指令中定义的布局，通常展现出层次化的结构。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/08.webp)

以图 5 中的布局（c）为例进行考虑。该布局的形状为（4，6），在 6 个线程中存储 24 个元素，每个线程持有四个元素。我们将每个线程中存储的四个元素表示为 $a_0, a_1, a_2, a_3$。比较其前两行与最后两行，我们观察到类似的结构，只不过最后两行存储的是 $a_2$ 和 $a_3$ 元素，而不是 $a_0$ 和 $a_1$。

为了模拟这种结构上的不变性，布局（c）可以视为布局（a）和布局（b）的组合，其中布局（a）中的每个元素代表一个具有布局（b）的 tile（瓦片，即数据块）。实际上，布局（a）和（b）可以通过以下方式组合来表示布局（c）：

$$
c(t, i) = a(t / 6, i / 2) \odot (2, 6) + b(t \% 6, i \% 2)
$$

其中 $0 \leq t < 6$，$0 \leq i < 4$，其中 $\odot$ 表示逐元素乘法，而（2，6）代表布局（c）的形状。这一组合原则可以推广。给定两个具有相同维度数的布局 $f$ 和 $g$，我们定义它们的组合 $h = f \circ g$ 为：

$$
h(t, i) = f(t / T_g, i / N_g) \odot S_g + g(t \% T_g, i \% N_g),
$$

其中 $T_g, N_g, S_g$ 分别表示布局 $g$ 的线程数、每个线程的局部元素数和形状。我们可以证明组合是结合的，即对于任意三个布局 $f$、$g$ 和 $h$，等式 $f \circ (g \circ h) = (f \circ g) \circ h$ 成立。然而，组合通常不是交换的，即通常 $f \circ g \neq g \circ f$。

**空间布局和局部布局分别为线程和局部元素采用行主序**。通过组合，我们还可以构造它们的列主序对应物，即 `column_spatial(...)` 和 `column_repeat(...)`，**如图5中的布局（e）所示**。

回到图3中的张量核心指令布局，它可以表示为组合布局 `local(2, 1).spatial(8, 4).local(1, 2)`。**利用布局组合，我们还可以定义其逆操作**。如果 $h = f \circ g$，我们定义 $f = h / g$ 作为将布局 $h$ 除以布局 $g$ 的结果。例如，将 `local(2, 4)` 除以 `local(1, 2)` 得到 `local(2, 2)`。在附录A中，我们机型了相关组合的证明，这里略。

## 五、线程块级编程模型

> 现代 GPU 编程模型（如 PTX 和 CUDA）在单个线程级别上定义操作，遵循单指令多线程（SIMT）范式。为了简化 GPU 编程，我们采用基于线程块的编程模型，将操作定义在线程块级别而非单个线程级别。

此外，基于前面介绍的布局系统，我们提出显式暴露现代 GPU 中层次化内存结构的模型，从而在降低编程复杂性的同时实现对内存的精细控制。我们将此模型称为单指令多块（SIMB）。

### 5.1 状态空间和类型系统

虚拟机支持三种类型的变量。

- 标量变量存储单个值，如整数（例如 int32）或浮点数（例如 float16）。
- 指针变量存储内存地址而非直接的数据值。
- 张量变量表示多维数组，其类型指定形状、元素类型、内存范围和布局。张量驻留在不同的内存范围中，包括全局内存、共享内存和寄存器。

张量布局决定了高维张量元素如何映射到线性内存。虚拟机中的所有变量都在线程块级别上操作，即线程块内的所有线程协作维护这些变量。

### 5.2 程序结构和控制流

**虚拟机程序的结构包括程序名称、网格形状、参数列表和程序体**。网格形状在 `<...>` 中指定，可以是基于程序参数的表达式。程序体由一系列控制流语句或块级指令组成。与低级虚拟机或其他指令集架构不同，我们的虚拟机保留了高级控制结构，以提高可读性和可编程性。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/09.webp)

### 5.3 线程块级指令集

> 虚拟机的指令集中的每条指令都在线程块级别上操作，而非单个线程级别。**表 1 列出了指令集中的指令及其签名和指令语义**。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/10.webp)

这些指令在指定的内存空间（如全局内存、共享内存、寄存器）中分配特定数据类型的张量，将张量在内存空间之间传输，并对寄存器张量执行计算或转换。现代处理器执行指令时可能会乱序执行，即后续指令可能在当前指令完成之前就开始执行，前提是它们之间没有依赖关系。

**我们的虚拟机中的指令执行也表现出这种行为**。然而，当两条指令访问共享内存或全局内存的同一区域且第二条指令依赖于第一条指令的完成时，必须插入同步指令以确保所有先前指令在后续指令执行之前完成。

## 六、任意低精度数据类型

> 现代处理器以字节（8 位）为最小处理单元。因此，现代编程语言中的标准数据类型通常具有 8 的倍数字节的位宽。

然而，大型语言模型（LLM）对计算和内存的高需求使得低精度数据类型（小于 8 位）对于减少资源消耗至关重要。本节介绍虚拟机如何高效支持低精度数据类型。

### 6.1 低精度数据的存储

> 由于现代处理器（包括 CPU 和 GPU）以字节为最小内存访问和计算单元，我们将低精度数据（每个元素少于 8 位）紧凑地存储在字节内，如图 7 所示。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/11.webp)

紧凑存储消除了低精度值之间的位间隙，有时会导致单个值跨越两个 uint8 条目（例如图 7 中的 b[1]）。使用位运算提取、操作和存储打包字节数组中的低精度值。加载低精度值时，首先使用位与操作提取相关位，用位移操作调整它们的位置，最后用位或操作组合分离的部分（如果该值跨越多个字节）。同样，存储低精度值时，首先用位掩码清除目标位位置，然后用位或操作插入新值，同时保留其他位。低精度数据在算术计算前转换为标准数据类型，并在之后转换回原类型。

尽管这些方法支持任意位宽的数据类型，但效率通常较低，仅作为后备机制。对于 LLM 服务中的低精度计算，需要更高效的处理方式。

### 6.2 LLM 中高效的低精度支持

> LLM 中的低精度内核通常在计算前执行两个步骤：（1）从全局内存加载权重到片上内存（寄存器或共享内存），（2）将低精度权重转换为高精度（例如 float16）以进行反量化。

因此，高效的内存加载和转换对于性能至关重要。高效的低精度权重加载。通过前面讨论的低精度支持，我们的虚拟机可以使用 LoadGlobal 指令加载低精度张量。然而，直接加载效率低下，因为需要多次位运算和非连续内存访问[35]。为解决此问题，我们转换全局内存中的权重张量布局，以便更高效地加载。如果未进行转换，加载 dtype 为 i6 且布局为 local(2, 1).column_spatial(4, 8).local(2, 1)的寄存器张量将导致非连续内存访问，增加内存访问事务次数[35]。此外，提取低精度位需要额外的位运算。为此，我们识别出一个兼容的张量类型，dtype 为 uint8，布局为 local(n2).spatial(T).local(n1)，其中 n1 = gcd(24, 16)，n2 = 24/gcd(24, 16) = 3。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/12.webp)

如图 8 所示，我们将权重张量[K, N]划分为形状为[BK, BN]的瓦片。每个瓦片（tiling）从 i6[BK, BN]重新解释为 u8[BK * BN * 6 / 8]（Line 19），并连续存储（Line 20）。这允许我们使用硬件友好的指令高效加载瓦片（图 2 中的 Line 10, 11），同时启用标准数据类型的流水线异步内存传输，避免依赖共享内存的布局转换。这种方法可以推广到任意布局的低精度张量加载。

更正式地说，给定一个每个线程有 B bits 的张量和 T 线程，我们将其重新解释为 dtype uint8 和布局 local(n2).spatial(T).local(n1)，其中 n1 = gcd(B, 16)，n2 = B/gcd(B, 16)。高效的类型转换。加载后，权重必须从低精度转换为高精度（例如 float16）以进行计算，特别是当硬件不支持给定的低精度格式时。

我们利用特定于目标的指令进行高效的向量化类型转换。在 CUDA 上，我们使用 PRMT（在 32 位寄存器中排列字节）、LOP3（对三个输入进行任意逻辑操作）和位指令以最小的开销执行类型转换，因为所有操作都在寄存器内执行，不需要线程间通信。

## 七、实现 Tilus

> 由五个主要组件组成：Python 中的领域特定语言（DSL）、中间表示（IR）、优化过程、代码生成器和运行时系统。

DSL 使开发人员能够用 Python 编写 Tilus 程序，然后将这些程序翻译成 VM 的 IR 以便进一步处理。**优化过程通过消除冗余和简化算术表达式来改进 IR。代码生成器将优化后的 IR 翻译成 Hidet IR** [12]，一种类似 CUDA C 的中间表示。之后，我们应用第 6 节中的转换来实现低精度类型，同时保留原始语义。

**最终的 CUDA C 代码从 Hidet IR 生成，并使用 nvcc 编译器[35]编译成硬件二进制文件**。运行时系统管理动态加载的二进制文件并提供执行环境。整个系统由大约 20,000 行 Python 和 C++ 代码组成。

下面我们看一下整个过程。

### 程序编译和运行时

> **我们为虚拟机提供了一种特定领域的语言（DSL），以便程序员能够直接在Python中编写虚拟机程序**，从而方便地将生成的内核与丰富的Python深度学习生态系统集成。

#### 步骤1：全局和共享内存规划

每个GPU内核在启动时可以使用已知大小的共享内存空间。**为了简化GPU编程**，我们**允许用户在需要时多次分配共享内存**。

因此，我们**需要一个共享内存规划器来计算虚拟机所需的共享内存的大小**，并将**共享张量映射到内核共享内存空间的一个区域**。与共享内存规划类似，我们还需要一个**全局内存规划器** 来管理所有线程块共享的全局内存的分配。

我们将请求虚拟机的运行时系统在全局内存中分配一个工作区，使内核在执行期间可以使用此工作区。

### 步骤3：每个指令的代码生成

注：你没看错，这里原文就是步骤1完了就是步骤3，不信你去看原文。

我们**为每个VM指令逐一生成低级GPU代码**。在我们的实现中，我们使用HIDet IR [12] 来表示低级GPU代码。

在这个过程中，我们会进行指令选择，尽可能选择最有效的低级指令。例如，我们使用1ds pt x指令 [36] 从共享内存加载数据到寄存器。然而，如果加载的寄存器张量的布局可以被`layout spatial(8, 4).repeat(1, 4)`整除，则也可以使用更高效的PTX指令 `ldmatrix` 。

此外，我们还尝试对内存加载和存储指令进行自动向量化。例如，我们将尝试使用诸如`cp.async.v4、1ds128`和`1dg128`等向量化指令，以最大化内存访问效率。

### 步骤4：降低低精度数据类型

在我们将线程块级别的指令排放到低级IR之后，我们将应用第6.1节中讨论的规则所实现的通道，将低级IR中的所有低精度操作转换为相应的硬件友好类型操作。

在大多数情况下，只会应用向量化类型转换，从低精度类型到标准类型（例如，float16将被替换为标准类型），由于我们的布局规范化和寄存器张量重解释，低精度数据的内存加载将被标准类型所取代。

之后，我们从低级IR生成CUDA（用于NVIDIA GPU）代码，最后使用编译器nvcc将源代码编译为硬件二进制文件。

### 虚拟机运行时

编译后的二进制文件可以被虚拟机运行时加载。运行时还维护内部状态以服务于内核执行：

1. 按需分配的工作区内存，可以由编译后的CUDA内核请求；
2. 执行上下文，存储内核将要启动的CUDA流；
3. 缓存在内存中的内核。

## 八、评估

### 8.1 实验设置

**工作负载**：我们对三个具有不同模型大小的代表性大型语言模型（LLM）进行基准测试：Gemma-2-9B[46]、QWen2.5-32B[56] 和 Llama-3.3-70B-Instruct[19]。我们评估了预填充和解码两个阶段的性能。对于算子级别的分析，我们重点关注从这些模型中提取的矩阵乘法内核。**Tilus 原则上支持 Triton 支持的所有内核，但本文重点关注量化矩阵乘法。**

**基线**：我们将我们的方法 Tilus 与供应商库 cuBLAS[34]、最先进的深度学习编译器 Triton[47] 和 Ladder[52] 以及手工制作的内核 QuantLLM[54] 和 Marlin[18] 进行比较。启用了 Triton[47] 和 Ladder[52] 的自动调优，而 QuantLLM[54] 使用其启发式策略选择内核超参数。对于端到端评估，我们将量化内核集成到最先进的 LLM 服务框架 vLLM[25] 中，并在端到端执行中与 vLLM[25] 和 Ladder[52] 进行比较。工具的具体版本为：vLLM v0.5.3、Triton v3.1.0、bitblas v0.0.1.dev15（Ladder）、QuantLLM 提交哈希为 9802c5a，Marlin v0.1.1。

**硬件配置**：实验主要在配备 NVIDIA L40S GPU（48 GiB）的服务器上进行，GPU 驱动为 565.57.01，CUDA Toolkit 为 12.6.3。我们还在 NVIDIA A100 和 H100 GPU 上进行了基准测试，以展示我们的方法在不同硬件平台上的通用性。

**实验协议**：对于算子实验，每个内核执行 50 次；对于模型实验，每个模型执行 10 次。使用 CUDA Events[35] 测量延迟，并报告中位数延迟。为消除连续运行中的异常值，在每次执行前清除 L2 缓存。

### 8.2 低精度内核的性能

> 我们实现了一个单一的虚拟机程序模板来支持所有量化类型的矩阵乘法，并将瓦片大小作为可调超参数。在评估中，我们将此自动调优程序的性能表示为 Tilus。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/13.webp)

**图 9 比较了 Triton[47]、Ladder[52]、QuantLLM[54]、Marlin[18] 和 Tilus（我们的方法）在各种低精度矩阵乘法（uint8（u8）、float6_e3m2（f6）、uint4（u4）、int4（i4）、uint2（u2）和 uint1（u1））相对于 cuBLAS[34] 的加速比。**

尽管每个基线仅支持有限的量化数据类型，Tilus 在所有情况下均能实现加速。对于小批量情况，主要瓶颈是从全局内存加载权重到寄存器以进行 SIMT 或 Tensor Core 上的计算。

**Triton 在此方面表现不佳，因为权重加载到寄存器后需要进行代价高昂的布局转换**。尽管可以提前在全局内存中进行转换以缓解此问题，但 Triton 的编程模型缺乏对显式布局控制的支持，使得此类优化不可行。

**Ladder** 通过修改全局内存中的数据布局改进了 Triton，避免了冗余转换。然而，**它缺乏关键优化，如软件流水线** [22, 33]，其类型级打包限制了对任意位宽的高效支持，导致内存带宽利用率低下。

**QuantLLM[54] 和 Marlin[18] 的专家级手工内核针对特定量化方案进行了优化，但缺乏灵活性和可维护性**。相比之下，Tilus 使用单一参数化的虚拟机模板超越了所有基线，通过良好的抽象编程模型高效支持了全系列量化类型。

### 8.3 任意数据类型支持

> 我们的虚拟机支持形式为 matmul(A, B) 的低精度矩阵乘法，其中操作数 A 可以具有 32 位、16 位或 8 位的数据类型，而权重 B 支持从 32 位到 1 位的广泛位宽。

支持的标准数据类型包括 float32、float16 和 int8，还支持位宽小于 8 位的定制低精度类型，包括有符号整数、无符号整数和具有任意指数及尾数分布的浮点格式。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/14.webp)

利用第 4 节和第 6.2 节中的代数布局系统，我们的 VM 能够高效访问低精度数据。**图 10 展示了针对量化权重数据类型的全系列的加速比：从 uint1 到 uint8、int2 到 int8 和 float3 到 float8**。我们选择了代表性指数-尾数分布的浮点数据类型，如 e4m3、e3m3、e3m2、e2m2、e2m1 和 e1m1。每个行代表类型种类（例如无符号整数、有符号整数或浮点数据类型），而每列代表位宽。

我们采用矩阵乘法维度 BS=16、K=8192 和 N=57344 进行评估，**结果表明我们的方法具有显著的加速效果**。值得注意的是，所有内核均来自同一程序模板，**通过参数化瓦片（tiling）大小来实现，因此编程工作量有限。每个算子大约有 200 种配置，编译大约需要一分钟**。实验中激活数据类型为 float16，我们也支持 bfloat16 和 int8。

### 8.4 端到端性能

> 我们评估了代表性 LLM 的端到端性能：Gemma-2-9B[46]、QWen-2.5-32B[56] 和 Llama-3.3-70B[19]，涵盖预填充和解码两个阶段。

预填充阶段一次性处理所有提示标记，生成解码阶段的 kv-cache，解码阶段则迭代生成一个标记。预填充延迟决定了首次生成标记的时间（TTFT），而解码延迟影响后续标记的生成速度。这两个阶段对于优化用户体验和系统利用率都至关重要。我们使用了连续批处理[25, 57]来高效批处理多个解码请求。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/15.webp)

**图 11 显示了这些模型在这两个阶段的延迟。我们的方法在所有情况下均超越了 Ladder** [52]，特别是在解码阶段生成多个标记时（图 11 中的中间列）。

对 Ladder 生成的内核的分析表明，其在 CUDA Core 上的 1-15 个标记的利用率不佳，且未实现关键优化如软件流水线[22] 和 k 维并行化[39]，导致性能不佳。对于预填充阶段，我们解码量化权重到 float16 并使用标准的 f16xf16 矩阵乘法内核进行计算，因为此阶段计算成为瓶颈。**我们对量化权重布局的高效处理确保了解码过程中最小的解码开销，从而带来了卓越的性能**。

### 8.5 案例研究

#### 8.5.1 不同硬件上的加速

> 我们评估了 QWen2.5-30B 模型在 NVIDIA A100、L40S 和 H100 GPU 上的端到端性能，这些 GPU 分别对应 Ampere、Ada Lovelace 和 Hopper 架构。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/16.webp)

**图 12 展示了 vLLM[25]（float16）、Ladder[52]（uint4）和 Tilus（uint4，我们的方法）在解码和预填充阶段的性能比较**。

在 Hopper 架构（H100）上，Ladder 无法生成有效的内核，导致 CUDA 错误（图中标记为 ERR）。在 L40S GPU 上，vLLM[25] 超出了可用的 48 GiB DRAM 容量，导致内存不足（OOM）错误。在所有其他配置中，Tilus 在所有 GPU 和两个处理阶段中均超越了 Ladder，突出了其稳健的性能和跨架构的适应性。

#### 8.5.2 不同批量大小的加速

> 我们分析了批量大小与加速比之间的关系，通过基准测试矩阵乘法在不同批量大小下的性能。

对于解码阶段，我们评估了批量大小为 1、4、8 和 16 的情况，而对于预填充阶段，我们使用了批量大小为 4096、8192 和 12288。实验在 Llama-3.3-70B-Instruct[19] 模型上进行，量化数据类型为 float6_e3m2（f6）和 uint4（u4），其中 $M = 8192$ 和 $K = 57344$。

![文章插图](/articles/2025-05-30-triton-gpgpu-llm-efa3e109/images/17.webp)

**如图 13 所示，Tilus 在 LLM 服务中解码和预填充阶段使用的批量大小下均超越了基线方法**。为了进一步理解这些性能差异，我们对 cuBLAS、Ladder 和 Tilus 的内核进行了性能分析，详细分析见附录 D。

## 九、相关工作

> 许多深度学习编译器采用基于循环的调度[10, 40]，并在其上构建自动调优框架[2, 4, 16, 43, 50–52, 55, 59–62]。

**与之不同，Tilus 采用过程导向的方法，更好地模拟 GPU 硬件**，提高了可编程性和灵活性。除了基于循环的调度外，张量程序还经常使用供应商库（例如 cuBLAS[34]）、预定义模板进行高效的矩阵乘法[33]、硬件感知的瓦片（tiling）策略[64] 以及针对线性代数的专用编译器[42]。

**虽然这些方法优先考虑性能，但它们缺乏对任意低精度数据类型的可扩展性**。其他研究关注优化不规则或 ragged 张量程序[15, 45]、操作符融合[58, 63]、动态形状处理[14, 44, 59, 65]以及调度独立操作符[13, 24, 31]。微缩放数据类型[41] **可以看作是更精细的量化，我们也可以支持它**。**这些技术与我们关注高效低精度计算的重点是互补的**。**Triton[47] 引入了基于瓦片（tiling）的编程模型**。然而，**它缺乏对低精度数据类型的显式支持**，并且**没有暴露 GPU 的内存层次结构，限制了优化机会**。

同样，作为我们后端的 Hidet[12] 也没有提供对低精度类型的内置支持。Graphene[20] 提出了一个带有布局表示的中间表示（IR）。与 Graphene 关注 stride 和计算不同，**我们的代数布局系统强调层次化组织**。实际上，我们可以将 Graphene 的布局表示表达为我们的系统中的一个原始组件。

## 十、结论

> 我们介绍了 Tilus，一个**为高效低精度 LLM 服务设计的 GPGPU 虚拟机**，解决了现有解决方案的关键限制。

Tilus 配备了用于**在块内寄存器** 中分布张量的**代数布局系统**、具有**精细内存管理** 的**线程块级编程模型** 以及对 **1 到 8 位任意精度的广泛支持**。

实验结果表明，**与 Triton 和 Ladder 等先进框架相比，我们的方法具有显著的性能提升**，证明了我们方法的灵活性和可扩展性。这项工作为高效且可扩展的 LLM 推理奠定了基础，为新兴硬件、先进量化技术和多样化的低精度格式的进一步优化铺平了道路。

## 参考文献

::: block-1
1. Tor M Aamodt, Wilson Wai Lun Fung, and Timothy G Rogers. 2018. The SIMT Core: Instruction and Register Data Flow. In General-Purpose Graphics Processor Architectures. Springer, 21–66.
2. Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, TzuMao Li, Michaël Gharbi, Benoit Steiner, Steven Johnson, Kayvon Fatahalian, Frédo Durand, and Jonathan Ragan-Kelley. 2019. Learning to Optimize Halide with Tree Search and Random Programs. ACM Trans. Graph. 38, 4, Article 121 (jul 2019), 12 pages. doi:10.1145/3306346. 3322967
3. Aditya Agrawal, Matthew Hedlund, and Blake Hechtman. 2024. eXmY: A Data Type and Technique for Arbitrary Bit Precision Quantization. arXiv:2405.13938 [cs.LG] https://arxiv.org/abs/2405.13938
4. Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, and Hadi Esmaeilzadeh. 2020. Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation. In International Conference on Learning Representations. https://openreview.net/forum? id=rygG4AVFvH
5. AMD Corporation. 2024. CDNA 3 Architecture for Accelerated Computing. Available at https://www.amd.com/en/technologies/cdna.html.
6. AMD Corporation. 2024. HIP: Heterogeneous-Compute Interface for Portability. Available at https://rocm.docs.amd.com/projects/HIP/en/latest/
7. Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems https://openreview.net/forum?id=dfqsW38v1X
8. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 1877–1901. https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf
9. Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. 2023. QuIP: 2-bit quantization of large language models with guarantees. In Proceedings of the 37th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’23). Curran Associates Inc., Red Hook, NY, USA, Article 196, 34 pages.
10. Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Q. Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. 2018. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. In OSDI
11. Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2024. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article 2198, 15 pages.
12. Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, and Gennady Pekhimenko. 2023. Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 370–384. doi:10.1145/3575693.3575702
13. Yaoyao Ding, Ligeng Zhu, Zhihao Jia, Gennady Pekhimenko, and Song Han. 2021. Ios: Inter-operator scheduler for cnn acceleration. Proceedings of Machine Learning and Systems 3 (2021), 167–180.
14. Pratik Fegade, Tianqi Chen, Phillip Gibbons, and Todd Mowry. 2021. Cortex: A compiler for recursive deep learning models. Proceedings of Machine Learning and Systems 3 (2021), 38–54.
15. Pratik Fegade, Tianqi Chen, Phillip Gibbons, and Todd Mowry. 2022. The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding. In Proceedings of Machine Learning and Systems, D. Marculescu, Y. Chi, and C. Wu (Eds.), Vol. 4. 721–747. https://proceedings.mlsys.org/paper/2022/file/d3d9446802a44259755d38e6d163e820-Paper.pdf
16. Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, and Yong Yu. 2023. TensorIR: An Abstraction for Automatic Tensorized Program Optimization. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 804–817. doi:10.1145/3575693.3576933
17. NVIDIA Corporation. 2021. CUTLASS: CUDA Templates for Linear Algebra Subroutines and Solvers. https://github.com/NVIDIA/cutlass
18. NVIDIA Corporation. 2023. NVIDIA cuBLAS Library. https://developer.nvidia.com/cublas Version 12.2..
19. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
20. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
21. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
22. Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman Amarasinghe. 2013. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. In Acm Sigplan Notices, Vol. 48. ACM, 519–530.
23. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.07723 (2022).
24. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, WeiChen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. In MLSys.
25. Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.
26. Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. 2021. MLIR: scaling compiler infrastructure for domain specific computation. In Proceedings of the 2021 IEEE/ACM International Symposium on Code Generation and Optimization (Virtual Event, Republic of Korea) (CGO ’21). IEEE Press, 2–14. doi:10.1109/CGO51591.2021.9370308
27. NVIDIA Corporation. 2024. NVIDIA cuBLAS Library. https://developer.nvidia.com/cublas Version 12.2..
28. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
29. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
30. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
31. NVIDIA Corporation. 2023. CUTLASS: CUDA Templates for Linear Algebra Subroutines and Solvers. https://github.com/NVIDIA/cutlass
32. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
33. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
34. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
35. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
36. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
37. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
38. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
39. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
40. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
41. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
42. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
43. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
44. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
45. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
46. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
47. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
48. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
49. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
50. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
51. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
52. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
53. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
54. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
55. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
56. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
57. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
58. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
59. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
60. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
61. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
62. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
63. NVIDIA Corporation. 2024. Parallel Thread Execution ISA Version 12.0 Available at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
64. NVIDIA Corporation. 2024. SASS: Streaming Assembler for NVIDIA GPUs. Available at https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html.
65. NVIDIA Corporation. 2024. CUDA C++ Programming Guide. Version 12.0. Available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
:::
