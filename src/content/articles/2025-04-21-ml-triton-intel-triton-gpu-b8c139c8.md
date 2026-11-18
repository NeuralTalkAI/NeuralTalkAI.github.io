### 研究背景

> ❝
>
> - 大型语言模型（LLMs）时代，密集运算（如通用矩阵乘法GEMM和多头注意力MHA）至关重要，且适合用基于tile的方法在GPU上并行执行。
> - 传统GPU编程依赖低级接口（如CUDA或SYCL），而Triton作为一种新兴的领域特定语言（DSL），提供了更友好且可移植的高层编程方式。
> - **当前Triton的编译方式存在局限性**，本文提出了一种新的，具有**多级编译流程和编程接口的Triton语言扩展**，以更好地利用**GPU的层次化结构**，提升编译器的**解耦合性与清洁性**，满足**内核开发者对最新硬件性能的精细控制需求**。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/01.webp)

https://github.com/intel/intel-xpu-backend-for-triton/blob/main/third_party/intel/backend/compiler.py#L111

third_party/intel/lib/TritonToTritonGPUWarp/TritonToTritonGPUWarpPass.cpp

third_party/intel/lib/TritonIntelGPUTransforms

比较适合A100 like的架构，有明确的warp level。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/02.webp)

## 1. 背景知识

### 1.1 Intel GPU架构

Intel Ponte Vecchio GPU（PVC）是专为AI推理和训练工作负载设计的，具有模块化架构：

- 包含两个tile（瓦片，可以理解为GPU的基本构建模块）
- 每个tile包含64个XeCore（执行核心，类似于Nvidia GPU中的CUDA核心）
- 每个XeCore包含8个执行单元（EU，Execution Unit）
- 每个EU支持8个硬件上下文，使用原生SIMD16指令（单指令多数据指令，一种并行计算指令集）进行操作

PVC具有先进的内存层次结构，包括全局内存、L2缓存、共享本地内存（SLM，类似于Nvidia GPU中的共享内存）和L1缓存，以提高数据访问效率和计算速度。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/03.webp)

此外，PVC还支持1D/2D块加载/存储/预取指令以及DPAS指令（点积累积指令，用于加速矩阵乘法等计算密集型任务，见下图，详见：`https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md`），以进一步优化数据传输和计算性能。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/04.webp)

> ❝
>
> 简单来说，PVC GPU就像是一个超级计算工厂，专门用于处理AI模型中的大量计算任务，能够快速、高效地完成复杂的数学运算，从而加速AI模型的训练和推理过程。与我们常见的CPU不同，它更像是一个专门设计用来处理图形和并行计算任务的艺术家，拥有许多小工作站（核心），可以同时处理大量相似的任务。

### 1.2 Triton语言及其编译流程

Triton是一种专为GPU内核开发设计的高性能编程语言，具有Python风格的语法和工作组级别（workgroup level，类似于CUDA中的线程块级别）编程接口，使GPU编程对AI研究人员和工程师更加友好。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/05.webp)

Triton编译器基于MLIR（多级中间表示，一种编译器基础设施）构建，其编译流程包括生成Triton IR（中间表示）、转换为Triton GPU IR、中间端优化以及最终转换为LLVM IR（低级虚拟机中间表示）。

#### 1.2.1 Triton Dialect

Triton方言（Triton dialect）是Triton编译器引入的一种特定的语法和操作集合，专门用于描述张量上的块级操作。在Triton中，张量（tensor）表示一个N维数组，可以是值或指针。默认情况下，Triton使用指向块的指针（block pointer）作为主要的内存访问机制，这种指针显式地传达了数据的连续性信息，对于密集操作来说，这种方式比默认的指针数组方法更高效。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/06.webp)

Triton方言的主要操作包括获取程序ID（get_program_id）、加载张量（load）、存储张量（store）、矩阵乘法（dot）、归约操作（reduce）、创建张量指针（make_tensor_ptr）以及调整张量指针的偏移量（advance）等。

#### 1.2.2 Layout Encoding

布局编码（layout encoding）是Triton中用于描述张量在内存中布局以及数据如何在线程之间分配的一种元数据机制。通过布局编码，可以指定数据在不同线程中的分布方式，从而帮助编译器优化内存访问和计算。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/07.webp)

Triton中的布局编码包括块编码（BlockedEncoding）、点操作数编码（DotOperandEncoding）和切片编码（SliceEncoding）等。

- 块编码表示张量的一个连续部分，其参数包括每个线程操作的块大小（sizePerThread）、线程在Warp中的排列方式（threadsPerWarp）、Warp在CTA中的排列方式（warpsPerCTA）以及内存访问顺序（order）。
- 点操作数编码用于点操作的参数，表示其父布局编码和在点操作中的位置。
- 切片编码表示在父布局编码的基础上沿某个维度进行压缩。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/08.webp)

> ❝
>
> ## 内容补充：CTA（Cooperative Thread Array）和Warp的关系
>
> 在GPU编程领域，CTA（Cooperative Thread Array）和Warp是两个重要的概念，它们在并行计算中扮演着不同的角色，但又紧密相关。
>
> ### CTA（Cooperative Thread Array）
>
> **CTA，即协作线程组，是CUDA编程模型中线程块（thread block）的实现形式**。它是线程的集合，这些线程被调度到同一个流式多处理器（SM）上执行。CTA中的线程可以共享同一块内存空间，通过共享内存和同步机制（如`__syncthreads()`）进行通信和数据交换。CTA是GPU上执行并行任务的基本单位之一，它让一组线程能够方便地一起完成复杂的计算任务。
>
> ### Warp（线程束）
>
> **Warp是GPU中执行相同指令的线程集合，作为GPU的硬件SM调度单位**。在NVIDIA GPU中，一个Warp通常由32个线程组成，它们在同一个时钟周期内并行执行相同的指令，实现了单指令多线程（SIMT）的并行计算模型。Warp是SM中最小的调度单位，一个SM可以同时处理多个Warp。
>
> ### CTA与Warp的关系
>
> CTA和Warp之间的关系可以理解为一种层次结构：
>
> - 包含关系：一个CTA（线程块）由多个Warp组成。例如，如果一个CTA包含256个线程，那么这个CTA将被划分为8个Warp（256/32 = 8）。
> - 调度关系：CTA是程序员定义的逻辑分组，而Warp是硬件定义的调度单元。在执行过程中，CTA被分解为多个Warp，这些Warp被调度到SM的Warp调度器上，等待执行。
> - 执行关系：在同一个CTA内的Warp可以共享同一块内存空间，进行数据交换和同步。而不同CTA中的线程则不能直接通过屏障同步，它们必须通过全局内存进行协调。
>
> CTA和Warp是GPU编程中不可或缺的概念。**CTA提供了逻辑上的线程组织方式**，方便程序员管理和协调线程；而**Warp则是硬件实际调度和执行的单位**，确保了线程能够高效地利用GPU的并行计算资源。理解它们之间的关系有助于更好地设计和优化GPU上的并行算法。

#### 1.2.3 Compilation Flow

Triton的编译流程（compilation flow）包括以下几个阶段：

- 首先，内核函数被装饰为`triton.jit`，Triton编译器会遍历内核函数的抽象语法树（AST）以生成Triton IR，使用标准的静态单赋值（SSA）构造算法。
- 然后，Triton IR被转换为Triton GPU IR，通过添加一个简单的布局编码来指定数据在Warp之间的分布。
- 接下来，Triton GPU IR会经历一系列中间端优化，旨在分析和简化代码，包括内存合并、点积特定增强、软件流水线等优化操作。
- 最后，Triton GPU IR被转换为LLVM IR，可以传递给GPU后端编译器以生成可执行的二进制代码。

整个流程如下图所示：

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/09.webp)

## 2. ML-Triton的多级编译流程

而本文，我们提出的编译流程如图3所示，**通过多级降低来反映GPU的层次结构**。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/10.webp)

### 多级逐步降低的编译方法

这种方法**将不同层次的考虑因素解耦，使得优化更加高效和有针对性**。最初，Triton IR在Workgroup级别（工作组级别，可以理解为一组线程的集合）上运行，然后我们通过添加适当的布局编码将其转换为TritonGPU IR，以指定数据在Warp（warp，一种线程组的逻辑划分）之间的分布。接下来的将工作负载分布到Warp的pass将把内核工作负载转换为Warp级别，即每个Warp应该处理的内容。

匹配目标大小的pass会进一步拆分操作，以匹配硬件目标支持的LLVM固有函数大小，我们将其称为Intrinsic级别（固有级别）。最后，TritonGPU IR被转换为LLVM IR，采用SIMT或SIMD风格。

接下来，我们将使用一个GEMM示例来说明编译流程，因为GEMM是人们最熟悉的典型AI工作负载。给定$A \in \mathbf{R}^{m \times k}, B \in \mathbf{R}^{m \times k}, C \in \mathbf{R}^{m \times k}$，GEMM操作是 $C+= A ∗ B$。用户可以配置Triton内核以处理$256x256 += 256x32 * 32x256$的工作负载，并在循环体中将Warp的数量（numWarps）设置为32。列表2展示了解析抽象语法树（AST）后生成的Triton IR。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/11.webp)

### 具体编译流程阶段

#### 2.1 Convert-triton-to-tritongpu-warp

> ❝
>
> 转换Triton到TritonGPU Warp这一阶段。这个pass首先**分析内核的工作负载模式** （例如元素级操作、归约操作、GEMM、注意力机制等），然**后确定根操作** （如`tt.dot`操作）**的最佳布局编码**。随后，我们通过**定义-使用链传播来推导出其他所有值的布局编码**。

我们使传播规则变得简单明了：除了之前介绍的`tt.dot`和`tt.reduce`规则外，其他操作（包括`tt.load`、`tt.store`、`tt.advance`以及arith/math的单目/双目操作）要求所有源操作数和结果共享相同的布局编码。

但与上游Triton相比，有三个主要区别。

1. 工作负载感知。根据工作负载的不同，点操作可能需要不同的分区策略来实现最佳性能。例如，对于典型的GEMM，方形分区最为有利，而FlashAttention-2则更倾向于沿行维度分区，以最小化Warp之间的通信并实现峰值性能。
2. 一次性布局编码：我们的方法在一步中确定布局编码，而上游Triton最初分配一个简单的布局编码，并在后续pass中进行细化。
3. 关注sizePerWarp：旨在获取每个Warp处理的块大小（sizePerWarp），而不是每个线程的块大小。在一开始就获取每个线程处理的内容是一种过早的下沉（lowering）。

因此，对于GEMM示例，首先需要确定根操作`c += tt.dot a, b`的布局编码。

- 已知c的工作组大小为`[256, 256]`，Warp数量为32。

通过在Warp之间进行平方分区，我们得到c的块编码：

- 每个Warp在CTA中的排列方式为`[8, 4]`。
- 每个Warp处理的块大小为`[32, 64]`（工作组大小除以每个Warp在CTA中的排列方式）。

然后，a和b分别具有点操作数编码。最后，通过之前描述的传播规则，所有张量类型都被标注了布局编码，如列表2所示。注意，我们生成的Triton GPU IR保留了与Triton IR相同的操作。通常来说，上游Triton会引入许多“convert-layout”操作来帮助降低工作负载。

#### 2.2 Distribute-to-warps

> ❝
>
> 这个pass**根据相应的布局编码将工作组的工作负载分布到各个Warp**。在这个pass之后，我们得到每个Warp应该处理的内容。

之前我们修改了块编码以包含每个Warp处理的块大小和每个CTA中的Warp数量，这些参数决定了工作负载如何分布。因此，第一步是获取每个布局编码的等效块编码。对于点操作数编码和切片编码，我们从其父布局编码推导而来。映射规则详细列在表3中。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/12.webp)

对于GEMM示例，图4展示了Warp之间的数据分布。矩阵C被均匀分布，每个Warp处理一个32x64的块。对于矩阵A，尽管每个Warp处理的第二维度大小已经与工作组大小匹配，但我们仍有4个Warp需要排列，因此Warp 0-3处理A的同一个32x32子块。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/13.webp)

同样，Warp 0,4,8..28处理B的同一个32x64子块。在这个pass之后，如列表3所示，`tt.dot`被从$256x256 = 256x32 * 32x256$转换为$32x64 = 32x32 * 32x64$，`tt.make_tensor_ptr`的偏移量也从`tt.program_id`的函数调整为`tt.program_id`和`gpu.subgroup_id`（即warp_id）的函数。

#### 2.3 Match-target-size

> ❝
>
> **将操作拆分为更小的操作，以匹配目标LLVM固有函数（intrinsic）的大小**。所有共享相同布局编码的值将被一致地拆分，除非某个操作需要不同大小。用户可以指定最大加载大小和最大点积大小等选项。

对于GEMM示例，PVC的最大加载大小为$32x32$，最大点积大小为$8x16 = 8x16 * 16x16$。数据分区如图5所示。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/14.webp)

在这个pass之后，如列表4所示，A的加载大小为$32x32$，已经匹配目标加载大小，因此保持不变。然而，B的加载大小为$32x64$，被拆分为2个加载操作。点积操作从$32x64 = 32x32 * 32x64$拆分为32个更小的点积操作，每个大小为$8x16 = 8x16 * 16x16$。

所有在定义-使用链中的加载值都被拆分为处理$32x32$块的操作。由于`tt.dot`需要不同的块大小，添加了`tt.extract`来从A中提取$8x16$的子块和从B中提取$16x16$的子块，这些子块随后被馈送到点积操作中。注：`tt.extract`在实际应用中，被移动到triton_intel_gpu方言中。

#### 2.4 Convert-tritongpu-to-llvm

> ❝
>
> **将所有操作转换为LLVM IR，利用PVC GPU后端编译器提供的固有函数（intrinsic）**。对于**SIMT转换**，**数据将均匀分布到每个线程通道**，固有函数中的向量大小需要除以每个Warp的线程数。

这个pass它重用了上游MLIR转换对于arith、math和scf操作的转换模式。对于Triton操作，使用单独的转换模式将它们映射到LLVM固有函数（intrinsic）。**PVC GPU后端编译器提供了两组固有函数**：**用于SIMD编程的VectorCompute固有函数** 和**用于SIMT编程的GenISA固有函数**。

基本上，转换是一个一对一的机械映射，因为我们已经使操作匹配目标支持的固有函数大小。只是对于SIMT转换，数据被均匀分布到每个线程通道，意味着固有函数中的向量大小需要除以每个Warp的线程数，见表4。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/15.webp)

以`tt.load A`为例，工作负载大小`32x32xf16`被展平为`v512i32`，当除以PVC的每个Warp的线程数（16）时，结果为`v64i16`。列表5显示了转换后的LLVM IR。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/16.webp)

#### 2.5 FlashAttention-2

> ❝
>
> 正如编译流程中所展示的，**每个值的布局编码是关键**。**布局编码决定了工作在不同Warp之间的分配方式**，并作为指导，**指示每个操作应如何拆分以匹配目标固有函数大小**。一旦每个值都被正确标注了布局编码，后续的pass就可以有效地应用。**一旦我们确定了根操作的布局编码，所有其他值的编码都可以通过跟踪定义-使用链（use-def chain）来推导**。这种方法自然地促进了操作的预融合和后融合，因为它们可以从根操作无缝扩展。

以FlashAttention-2为例。它可以看作是两个背靠背的GEMM的融合内核，中间有一个在线softmax，如算法1中概述的。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/17.webp)

我们采用了原始论文中的工作分区方式，该方式沿**行维度在所有Warp之间分布输出矩阵O，并访问K/V**。在**平衡Warp之间共享的数据和寄存器压力后**，我们得到了PVC的以下内核配置：

- O的工作组大小为`[128, 64]`，Warp数量为8
- 通过在Warp之间进行水平分区，我们得到O的块编码：
- 每个CTA中的Warp排列方式为`[numWarps, 1] = [8, 1]`，
- 每个Warp处理的块大小为`[workgroupSize / warpsPerCTA] = [16, 64]`

随后，所有其他张量的布局编码通过推导得到，如表5中总结的。图6展示了它们之间的关系。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/18.webp)

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/19.webp)

## 3. ML-Triton的语言扩展

### 3.1 编译器提示（Compiler Hint）

> ❝
>
> 在**代码生成流程中**，我们借助一种**工作负载感知的pass来识别特定模式**，并**确定根操作的tile分区** ——布局编码。实际上，这一设置**源自专家内核调优经验的最佳实践**。然而，在某些情况下，研究人员可能希望手动定义tile分区。例如，FlashAttention-2 明确提出了如何在不同线程组（Warp）之间分配工作以实现最优性能。

因此，我们引入了一种**编译器提示，允许用户指定根操作在tile分区时的策略**。以下是针对二维张量可用的tile选项：

- 水平分区：沿第一维度（行）均匀地将数据划分为多个部分。
- 垂直分区：沿最后一维度（列）均匀地将数据划分为多个部分。
- 方形分区：将数据划分为方形子块。

对于 FlashAttention-2，仅需设置第二个点积的tile方式为水平，而无需更改其他源代码。**编译器会根据这一提示，自动推导出所有其他值的布局编码**。

示例代码：

```
o = tl.dot(p, v, o, tiling="horizontal")
```

### 3.2 线程组（Warp）级别 API

> ❝
>
> 尽管在**工作组级别编写内核可以减轻开发人员的负担，但性能往往高度依赖于编译器**，而编译器的优化需要时间来不断发展和完善。与其完全依赖编译器特定的优化，**我们认为更有效的方法是赋予开发人员对其代码的细粒度控制权**。

例如，FlashAttention-3 提出了更好的线程组管理方式，以充分利用最新的硬件功能。同样，**许多内核库（如 CUTLASS）也提供了线程组级别的 C++ 模板**。因此，我们**引入了线程组级别的语言扩展**。

关键特性包括：

- `warp_level`：元数据，表明这是一个线程组级别的内核。
- `tl.warp_id()`：返回当前线程组在线程组内的线性 ID。
- `tl.alloc(shape, data type)`：分配并返回一个指向共享本地内存（SLM）中具有指定形状和数据类型的块的指针。
- `tl.reduce(..., cross_warp, dst_warps)`：为归约操作（如求最大值、求和等）添加了关键字参数。当 cross_warp 设置为 true 时，表示在所有线程组之间进行归约；否则，表示在当前线程组内进行归约。dst_warps 参数允许将归约结果仅广播到指定的目标线程组。如果未设置，则结果将广播到所有线程组。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/20.webp)

通过**Warp级别API，开发者可以**：

- 分配共享内存：在GPU的**共享本地内存（SLM）中分配空间，用于存储临时数据**，这类似于在高速缓存中开辟一块专属的“工作区”。
- 跨Warp通信：实现**不同线程组之间的数据同步和交换**，例如在执行归约操作（如求和、求最大值）时，能够高效地整合各个线程组的计算结果。
- 精细控制计算流程：针对每个线程组定制计算逻辑，这对于处理分页注意力（Paged Attention）等复杂任务至关重要，因为它允许开发者明确**指定每个线程组的工作内容，避免不必要的计算重复**。

这些功能使得开发者能够以更细粒度的方式优化代码，充分发挥GPU的并行计算优势。

大型语言模型（LLM）推理过程通常涉及键值（KV）对的长序列。为了提高吞吐量，Flash Decoding 提出将键和值分开处理。为了高效管理内存，分页注意力（Paged Attention）将请求的KV缓存划分为多个块。

列表6展示了如何实现分页注意力Triton内核。尽管核心算法与FlashAttention-2相似，但由于查询序列长度通常仅为1，线程组（warp）之间的分布存在显著差异。这使得**编写分页注意力内核相对容易，但要实现开箱即用的最优性能却颇具挑战。编译器必须进行特定的分析和优化**，以提升性能。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/21.webp)

为了确保GPU上足够的并行性，一种方法是进一步在Warp之间划分KV缓存，如图7所示。**每个Warp处理不同的KV缓存块，并且需要在Warp之间进行归约操作，以同步每个Warp的部分结果**。

然而，**通过在Warp级别编程Triton，用户可以轻松表达上述分解**。列表7展示了Warp级别内核的实现方式。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/22.webp)

## 4. 实验结果与分析

### 实验环境与方法

研究团队在Intel PVC max 1550 GPU上进行实验，这是一款专为人工智能推理和训练设计的高性能图形处理器。他们使用OneAPI 2024.1作为开发平台，并**通过SYCL分析事件记录内核的GPU执行时间**，以评估性能表现。为了确保评估的公平性，将Triton与Intel的XeTLA（一款针对Intel GPU优化的线性代数模板库，类似于Nvidia的CUTLASS）进行对比，并且在两种情况下使用相同的配置参数，包括tile大小等，以最小化因配置差异导致的性能差异。

### 具体实验结果

#### 4.1 GEMM性能

GEMM（通用矩阵乘法）是AI领域中的一个基本操作，构成了大量计算工作负载。研究团队使用了Triton教程中的GEMM内核进行测试，**评估了两种类型的GEMM操作：内存受限和计算受限**。所有**矩阵形状都来自LLama-2和LLama-3等LLM模型**。

- **计算受限** GEMM：适用于LLM训练和推理，当研究人员关注单个GPU上的长上下文长度时，测试了从m=1k到16k的矩阵大小，足够大以充分利用GPU并实现峰值硬件吞吐量。结果如图8所示，**Triton实现了XeTLA性能的96%的几何平均值**。
- **内存受限** GEMM：在LLM推理期间的下一个标记预测阶段很常见。在**具有大m、大k和大n的案例上进行了评估**，以展示Triton的稳健性。如图9所示，**Triton的性能与XeTLA相当，几何平均值为94%。**

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/23.webp)

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/24.webp)

#### 4.2 FlashAttention-2性能

FlashAttention-2广泛用于现代Transformer模型中的多头注意力（MHA）机制。研究团队使用Triton教程中的内核进行测试，评估了具有总共32k标记和1k到32k序列长度的前向传递，与大多数流行LLM的上下文长度对齐。隐藏维度设置为2048，头部维度为64或128（即32个头部或16个头部）。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/25.webp)

如图10和图11所示，**与专家调优实现相比，ML-Triton生成的代码性能差距小于5%，** 证明了其代码生成的高质量。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/26.webp)

#### 4.3 分页注意力性能

分页注意力广泛用于LLM推理引擎。与FlashAttention不同，**分页注意力中的键/值对不是连续存储的**，必须通过块表映射进行访问，这增加了**内存访问的压力**。

![文章插图](/articles/2025-04-21-ml-triton-intel-triton-gpu-b8c139c8/images/27.webp)

**与传统的Triton工作组级别实现相比，ML-Triton的Warp级别内核直接表达了Warp之间的分布**，只需添加几行代码。如图12所示，**Triton的性能超过XeTLA的95%**，证明了其** 处理复杂内核的能力**。

## 5. 结论与未来工作

### 5.1 研究总结

**ML-Triton通过引入多级编译流程和语言扩展，显著提升了GPU编程的易用性和性能**。多级编译流程**从Workgroup级别逐步降低到Warp级别和Intrinsic级别**，使编译过程更加高效且具有针对性。

语言扩展部分，**编译器提示** 允许用户直接指导编译器如何优化关键操作，而**Warp级别API** 则赋予开发者对线程组更精细的控制能力。

实验结果表明，**ML-Triton在GEMM、FlashAttention-2和分页注意力等AI核心运算** 上的性能表现优异，**与专家调优的实现相比差距极小**，充分证明了其设计的有效性和实用性。

### 5.2 未来展望

随着AI技术的飞速发展，计算需求日益增长，ML-Triton的改进方向也逐渐明晰。首先，研究团队计划持续优化和完善ML-Triton的设计，以更好地适应AI领域不断涌现的新挑战和新需求。例如，随着模型规模的进一步扩大和计算复杂度的提升，如何在保持易用性的同时进一步挖掘硬件性能潜力，将是重要的研究课题。

其次，团队还期望将ML-Triton的编程和编译范式拓展至其他多核架构。现代计算硬件种类繁多，除了GPU，还有各种专用加速器和异构处理器。**将ML-Triton的成功经验推广到这些硬件平台**，有望为开发者提供**统一、高效的编程模型，降低跨平台开发的难度** 和成本，推动整个计算行业的发展。
