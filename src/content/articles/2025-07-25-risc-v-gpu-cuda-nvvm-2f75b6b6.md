关键词：CUDA, RISC-V GPU, Code Migration、NVIDIA

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/01.webp)

- Supporting CUDA for an extended RISC-V GPU architecture
- https://arxiv.org/abs/2109.00673
- https://github.com/gthparch/NVPTX-SPIRV-Translator
- 本文 8370 字，阅读需 32 分钟，播客 5 分钟如下

相关文章

- [ZLUDA：让 CUDA 程序在非英伟达 GPU 上“零修改”运行的开源兼容层——现状、原理与 2025 最新进展](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889691&idx=1&sn=ef6cd986f95c897029ba4b2981231cbe&scene=21#wechat_redirect)
- [一次编译，多平台运行！GPU 二进制文件兼容性在NVIDIA、AMD、Intel 和 Tenstorrent上的探索！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889722&idx=1&sn=ee787fcb0d896221430035a780fb5dff&scene=21#wechat_redirect)
- [迁移 CUDA 代码到其它硬件！HPCTransCompile：用 LLM 对 CUDA 代码转义！用AI编译器生成高质量代码对](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889060&idx=1&sn=f73025e12265f3ee62a373c5f3b03177&scene=21#wechat_redirect)
- [汇编级 NVIDIA 与 AMD GPU 代码转换新 SOTA！跨架构转换方案 CASS，远超商业转换基线，数据代码全开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889091&idx=1&sn=a50fd9383f13b3c1ab7b6d3393369123&scene=21#wechat_redirect)

> 随着科学计算的快速发展，越来越多的研究者和开发者致力于在不同设备上实现各种工作负载/操作。在所有这些设备中，NVIDIA GPU（图形处理器）是最受欢迎的选择，这得益于其全面的文档和优秀的开发工具。

因此，存在大量手写高性能 CUDA（英伟达推出的并行计算平台和编程模型）代码的资源。然而，CUDA 主要仅由商业产品支持，且目前尚无对开源硬件平台的支持。

::: block-1
RISC-V（一种开源的指令集架构）凭借其优雅的设计和开源许可，成为硬件指令集架构（ISA，即处理器能够理解和执行的指令集合）的最受欢迎选择。

在本项目中，我们旨在将这些现有 CUDA 代码用于 RISC-V 设备。具体而言，我们设计并实现了一个流水线，能够在 RISC-V GPU 架构上执行 CUDA 源代码。
:::

我们已成功在 RISC-V GPU 架构上执行了具有多线程、原子指令等多项重要特性的 CUDA 内核（内核，指在 GPU 上运行的函数）。

## 关键问题（TL;DR）

### Q1. 核心转换机制如何解决CUDA与RISC-V架构在指令集、内存模型等底层特性上的本质差异？

> 转换流水线通过“多级中间表示（IR）转换”和“后端适配”两层机制解决底层差异：

- **在指令集层面**这一过程中，:mdnice-bold-italic[NVVM IR到SPIR-V的转换会将CUDA特有的指令] （如原子操作、 warp 级同步指令）**映射为SPIR-V的通用操作**，再**由RISC-V后端根据自身指令集特性** （如向量扩展、原子指令支持）**生成等效指令**，从而屏蔽指令集差异。
- **在内存模型层面**

### Q2. 未被当前方法支持的CUDA特性（如纹理内存、特定数学函数）难以支持的根本技术障碍是什么？

> 根本障碍主要来自“硬件依赖”和“实现复杂度”两方面：

- :mdnice-bold-italic[纹理内存] （texture memory）的核心障碍是:mdnice-bold-italic[硬件专用单元的缺失]。
- :mdnice-bold-italic[特定数学函数] （如cuBLAS中的高精度 transcendental 函数）的障碍:mdnice-bold-italic[在于“硬件优化绑定”。]

### Q3. 性能表现与原生NVIDIA GPU存在本质差异的核心原因是什么？这种差异是否可通过优化弥合？

> 核心原因包括三方面：

- **硬件架构成熟度**：NVIDIA GPU经过多代迭代，拥有更完善的流处理器集群、更高的内存带宽（如HBM显存）和专用加速单元（如Tensor Core），而RISC-V GPU原型多为早期设计，硬件资源（如计算核心数量、缓存容量）和并行效率较低。
- **转换流水线开销**：从CUDA到RISC-V指令的多级转换（CUDA→NVVM→SPIR-V→RISC-V）可能引入额外的优化损失，例如NVVM IR中的硬件相关优化（如针对NVIDIA warp调度的指令重排）在转换为SPIR-V时被削弱，导致最终生成的RISC-V代码效率低于NVIDIA nvcc直接编译的原生代码。
- **软件栈生态**：NVIDIA拥有成熟的CUDA runtime和库（如cuDNN），其调度器能高效管理线程块分配、内存分配等，而RISC-V的软件栈（如runtime、设备驱动）尚在开发中，同步操作（如线程块同步）的软件实现开销高于NVIDIA的硬件支持版本。

这种差异部分可通过优化弥合：例如，优化SPIR-V到RISC-V的后端编译器（如增加针对RISC-V向量扩展的指令选择优化）、提升RISC-V GPU的硬件设计（如增加专用加速单元、提高内存带宽）、完善软件栈（如实现硬件级同步机制）；但完全弥合难度较大，因NVIDIA在硬件-软件协同优化上的长期积累形成了生态壁垒，RISC-V需长期迭代才能逐步缩小差距。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/02.webp)

## 本文目录

- 关键问题（TL;DR）
  - Q1. 核心转换机制如何解决CUDA与RISC-V架构在指令集、内存模型等底层特性上的本质差异？
  - Q2. 未被当前方法支持的CUDA特性（如纹理内存、特定数学函数）难以支持的根本技术障碍是什么？
  - Q3. 性能表现与原生NVIDIA GPU存在本质差异的核心原因是什么？这种差异是否可通过优化弥合？
- 一、引言
- 二、背景和相关工作
  - 2.1 程序迁移
  - 2.2 中间表示
  - 2.3 现有可用框架
- 三、流水线概述
  - 3.1 输入 CUDA 源代码
  - 3.2 编译为 NVVM
  - 3.3 将 NVVM 转换为 SPIR-V
  - 3.4 将 SPIR-V 转换为 OpenCL IR
  - 3.5 在 Vortex 上执行 OpenCL IR
- 四、详细描述与实验
  - 4.1 支持内置函数
  - 4.2 基准测试实验
- 五、结论
- 参考文献

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/03.webp)

## 一、引言

> RISC-V 是学术界研究者和硬件公司工程师最受欢迎的选择。最重要的原因是其开源精神。这些开源许可鼓励许多研究者投身于 RISC-V 成熟生态的开发，进而吸引了更多人加入该社区，因为社区中已有丰富的优质代码、硬件设计等资源。

**在 RISC-V 生态中，软件支持是制约 RISC-V 社区蓬勃发展的瓶颈**。虽然 OpenCL（开放计算语言，一种用于异构计算的开源平台）是异构计算的开源平台，但由于其稳定性和软件工具链支持，CUDA 得到了广泛应用。不幸的是，CUDA 源代码只能在 NVIDIA 的设备上编译和执行，这是 RISC-V 在广泛应用（尤其是高性能计算和机器学习工作负载）中的主要障碍。

**解决这一困境的一种方法是采用代码迁移** [8,10,13]。

- 一些研究者没有使用 NVIDIA 编译器的默认方法来编译 CUDA 源代码，而是尝试将源代码解析并修改为其他高级语言，详情见 2.1 节。然而，这些方法高度依赖 CUDA 与目标高级语言的高度相似性，并非通用解决方案。
- 另一种解决方案是构建一个直接将高级 CUDA 语言编译为低级 RISC-V 二进制文件的编译器。据我们所知，虽然存在支持生成 RISC-V 的转换器，但没有一个能够处理 CUDA 源代码。

因此，在:mdnice-bold-italic[本项目中，我们提出并构建了一个支持端到端 CUDA 迁移的流水线]：该流水线接受 CUDA 源代码作为输入，并在扩展的 RISC-V GPU 架构上执行。我们的流水线包括以下步骤：

::: block-1
- 将 CUDA 源代码转换为 NVVM IR[4]（NVVM IR，NVIDIA 推出的用于表示 GPU 计算内核的编译器中间表示）
- 将 NVVM IR 转换为 SPIR-V IR[7]（SPIR-V，一种用于并行计算和 GPU 图形学的中间表示）
- 将 SPIR-V IR 转发到 POCL[5]（POCL，一种 OpenCL 标准的高效实现框架）以获取 RISC-V 二进制文件
- 最后在扩展的 RISC-V GPU 架构上执行该二进制文件
:::

我们选择:mdnice-bold-italic[使用中间表示（SPIR-V）有两个原因]：

1. **RISC-V 仍在发展中**，存在许多扩展，因此不应直接将 CUDA 转换为 RISC-V，因为这会使我们的流水线在未来支持新特性时变得困难；
2. 我们**希望使流水线更通用**，以便能够支持 CUDA 作为前端和 RISC-V 作为后端。

我们的流水线如图 1 所示。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/04.webp)

::: block-1
总之，本文的主要贡献包括：

- 提出并实现了一个在 RISC-V GPU 上执行 CUDA 源代码的流水线；
- 构建了支持从 NVVM 到 SPIR-V 的转换器 ¹；
- 该流水线易于维护，并能进一步支持其他前端语言和后端设备；
- 该流水线轻量，可在没有 NVIDIA GPU 的情况下执行；
- 扩展现有 POCL 以支持 RISC-V 后端。
:::

本文的其余部分组织如下：

- 第 2 节概述了各种 CUDA 源代码迁移尝试，并介绍了我们的流水线中使用的现有框架。
- 第 3 节通过一个简单示例介绍整个流水线，以简要说明流水线的每个阶段。
- 第 4 节记录了如何支持 CUDA 的几个重要特性。
- 最后，第 5 节是我们的结论。

## 二、背景和相关工作

### 2.1 程序迁移

> 许多现有研究[1,14]旨在将 CUDA 迁移到其他表示形式，以适用于不同的后端设备。

#### HIPIFY

HIPIFY 是一个将 CUDA 源代码转换为可移植的 HIP C++（AMD 设备提出的另一种高级语言）的工具。:mdnice-bold-italic[HIPIFY 有两种将 CUDA 转换为 HIP C++的方法：hipify-perl 和 hipify-clang。]

::: block-1
- **Hipify-perl** 非常直接，它大量使用**正则表达式** 将 CUDA 关键字转换为相应的 HIP C++关键字。
- 与逐词替换不同，**Hipify-clang** 采用了更复杂的方法：它在 Clang（一种 C/C++编译器前端）中添加了一个适配器。它**利用 Clang 的语法分析，将 CUDA 源代码转换为抽象语法树** （一种表示代码结构的树状数据结构），使用**转换匹配器对该语法树进行转换**，最后生成 HIP C++。这两种方法都只能用于 HIP C++或其他语法与 CUDA 高度接近的源语言。
:::

HIPIFY 是一种从一种高级语言到另一种高级语言的转换器，因为其大部分工作都在词法层面完成。为了在 AMD 设备上执行 CUDA，用户需要先使用 HIPIFY 将 CUDA 转换为 HIP C++，然后使用 HIP C++编译器编译生成的代码。

**我们不能使用相同的方法在 RISC-V 设备上迁移 CUDA，因为 RISC-V 没有对应的高级语言。**

#### SYCL

SYCL[6,15]是 Khronos 组织提出的另一个致力于在不同设备上部署源内核（在 GPU 等设备上运行的函数）的项目。它是一种高级编程模型，旨在提高各种硬件加速器上的编程效率。它可以看作是 C++的一系列库。

这些库提供了编写可在各种后端设备上部署和执行而无需修改代码的程序所需的 API（应用程序编程接口）。这很像 OpenCL（开放计算语言，一种用于异构计算的框架），OpenCL 也为高级用户使用 C/C++，并自动为不同的后端设备生成主机/设备程序并执行。:mdnice-bold-italic[与 OpenCL 相比，SYCL 的层次更高；它提供了原语（基本操作或函数），支持用户直接为设备编写程序，而不是将代码视为字符串并直接转发给后端设备的驱动程序]。因此，SYCL 凭借其高级抽象具有高度的可移植性。**然而，SYCL 无法解决我们的困境，因为它不支持 CUDA。SYCL 不能迁移现有的 CUDA 源代码以在 RISC-V 上执行**。相反，它要求用户使用 SYCL 提供的 API 重新实现。

### 2.2 中间表示

在现代编译器设计中，编译器总能针对不同的后端设备，根据各种输入语言生成目标文件。为了支持多种源语言和多种后端设备，编译器首先将不同的源语言编译为标准的中间表示（IR，一种编译器在处理代码过程中使用的内部表示形式），然后根据目标后端设备将该 IR 生成不同的二进制文件。我们的:mdnice-bold-italic[流水线涉及三种 IR]：**NVVM IR、SPIR-V 和 OpenCL IR。**

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/05.webp)

在图 2 中，我们展示了这三种 IR 如何表示一个向量加法（vecadd）示例。图 2（a）和（c）中的 IR 格式相似，**除了一些调用指令外，大多数指令都相同**。这是:mdnice-bold-italic[因为 NVVM 和 OpenCL IR 之间的差异仅在于内置函数：对于相同的原语] （如获取 x 维度的线程索引），:mdnice-bold-italic[它们具有不同的内置函数] （`llvm.nvvm.read.ptx.srge.tid.x`和`getlocalid`）。

> SPIR-V IR 与 NVVM 和 OpenCL IR 完全不同，不仅是内置函数不同，加载/存储/二进制操作等指令也不同。

#### 2.2.1 NVVM

**NVVM IR[4]是 NVIDIA 提出的一种编译器 IR，用于表示 GPU 计算内核** （在 GPU 上运行的函数）。

通常情况下，要在 NVIDIA GPU 上执行 CUDA 源代码，需要调用 nvcc（NVIDIA CUDA 编译器）将 CUDA 编译为 NVVM IR，然后 GPU 驱动程序会进一步将 NVVM IR 转换为可在 NVIDIA GPU 处理核心上运行的二进制代码。:mdnice-bold-italic[NVVM IR 与 LLVM[11]（一个编译器框架）高度兼容]，它支持 LLVM IR 的一个子集以及一组定义好的用于表示 GPU 编程概念的内置函数。

#### 2.2.2 SPIR-V

SPIR-V[7]是 SPIR 的第五个版本。SPIR（标准可移植中间表示）是一种用于表示并行计算和基于 GPU 的图形学的中间表示。:mdnice-bold-italic[SPIR 是一种通用 IR，它允许高级用户使用各种前端编程语言]。**SPIR-V 可以用作隔离层，将前端编程语言** （如 MLIR、OpenCL、OpenGL）**与底层计算架构分离开来**。

借助 SPIR，我们首先将源代码编译为 SPIR IR，然后将这些 IR 转发到设备。这样，我们就可以消除设备驱动程序中对高级语言前端编译器的需求。这也将减少内核启动时间，因为我们不需要处理复杂且抽象的高级语言，而是处理更适合硬件的 SPIR-V IR。此外，通过 SPIR 作为隔离层，高级用户无需关心底层硬件，他们可以选择使用任何喜欢的编程语言，而不必担心程序的可靠性和可移植性。

对于硬件研究者来说，他们只需专注于 SPIR 的优化，而无需关心前端语言。

#### 2.2.3 OpenCL IR

OpenCL[12]（开放计算语言）是一个用于编写可在异构平台上执行的程序的框架。它支持多种后端设备：CPU（中央处理器）、GPU、FPGA（现场可编程门阵列）等。

尽管 OpenCL 可以支持使用原生语言（如 CUDA、HIP C++等）为每个后端设备实现内核，但它也**提供了 OpenCL C 编程语言，这是一系列用于高性能计算的高级 API**。这种语言提供了**丰富的内置函数集，用于标量和向量运算。这些函数支持标量和向量参数类型**，可用于为不同的后端设备实现高性能程序。当我们提到 LLVM IR 时，通常指的是 LLVM IR 的 LLVM bc 格式。

### 2.3 现有可用框架

#### 2.3.1 OpenCL-SPIR-V 转换器

Khronos 集团开发了一个双向转换器，支持在 OpenCL IR 和 SPIR-V 文件之间进行转换。尽管它生成的 SPIR-V 可以进一步部署并在 RISC-V 后端设备上执行，但它仅接受 OpenCL IR 作为输入。:mdnice-bold-italic[在我们的项目中，我们在原始转换器的基础上添加了一些扩展，以支持它处理 NVVM]。如第 2 节所述，**NVVM 是 LLVM IR 的一个子集，同时包含一组定义好的内置函数**。该转换器无法处理这些 NVVM 特有的内置函数。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/05.webp)

例如，在图 2 中，（c）是 OpenCL IR，可以被该转换器转换。然而，（a）是 NVVM IR，它包含一个 NVVM 特有的内置函数`llvm.nvvm.read.ptx.srge.tid.x()`。当我们将这个 NVVM 传递给转换器时，它会报错，因为它不识别这个内置函数，只识别 OpenCL IR 中的`get_local_id`。

#### 2.3.2 POCL

POCL（可移植计算语言）[5]是**作为 OpenCL 标准的高效实现而开发的**。POCL 是一个**接受 SPIR-V 二进制文件作为输入的框架**。

POCL :mdnice-bold-italic[首先使用上述转换器将 SPIR-V 转换为 OpenCL IR，将转换后的 OpenCL IR 与运行时库链接，最后生成目标文件]。POCL 支持多种后端设备，如 x86、CUDA、MIPS 等。POCL 具有高度的可扩展性，要支持新的后端设备，研究者只需提供 OpenCL IR 的运行时 API 库，该库将在生成目标文件时用于链接。

:mdnice-bold-italic[尽管原始 POCL 不支持 RISC-V，但其他研究者[3]已经在 POCL 上支持了 RISC-V，] 并在多个后端设备上执行了这些文件。在我们的实验中，我们使用这个版本的 POCL。

#### 2.3.3 Vortex

**Vortex[3]是一个开源的基于 RISC-V 的通用计算 GPU（GPGPU）处理器**。Vortex :mdnice-bold-italic[实现了一种 SIMT] （单指令多线程，一种 GPU 常用的并行执行模型）架构，:mdnice-bold-italic[并对 RISC-V 进行了最小化的指令集扩展] （tmc：激活线程，wspan：生成线程束，split/join：处理分支发散的指令，bar：线程束停滞指令）。

目前，Vortex 可以通过 POCL 运行时系统执行 OpenCL IR。因此，我们可以直接在 Vortex 上执行生成的目标文件。我们还在 POCL 的链接阶段使用了 Vortex 的 RISC-V 库。

## 三、流水线概述

> 在本节中，我们将展示如何在 RISC-V GPU 上执行一个简单的向量加法 CUDA 源代码（代码 1）。

**代码1：VectorAdd CUDA 源码**![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/06.webp)

### 3.1 输入 CUDA 源代码

CUDA 可以看作是标准 C++的扩展。唯一的区别是 CUDA 具有一些额外的特性，如内存层次结构的定义和一些独特的内置函数。**由于在 RISC-V 后端设备上执行 C++相当容易，我们只需处理 CUDA 特有的额外部分**。

例如，在我们的示例中，只有两个 CUDA 特有特性不属于标准 C++：`__global__`和`threadIdx.x`。`__global__`是一个 CUDA C 关键字，用于标识该函数可从主机端调用。`threadIdx.x`可以看作是一个内置函数，用于记录 x 维度上工作项（work-item，OpenCL 中对线程的称呼）的本地索引。

### 3.2 编译为 NVVM

我们的流水线将:mdnice-bold-italic[使用 CUDA 工具包将 CUDA 源代码编译为 NVVM IR]。因此，我们可以得到以下 NVVM IR（代码 2）（为清晰起见，我们删除了一些不重要的内容）：

**代码2：NVVM VectorAdd IR**![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/07.webp)

代码 2 中的 IR 开头有一个元数据变量“target triple”。这个元数据记录了目标后端设备，用于进一步生成目标文件。对于 NVVM，它的值始终是“nvptx64-nvidia-cuda”。

`nvvm.annotations`是另一个重要的元数据。它记录了该程序的内核函数。如 3.1 节所述，CUDA 有一个特殊的关键字`__global__`来标记一个函数为内核函数。**由于在 NVVM 中不能使用这个关键字，我们必须使用额外的元数据来记录此信息。**

这个 NVVM IR 有一个函数`llvm.nvvm.read.ptx.srge.tid.x`，它被声明但未定义。这是一个将与 CUDA 库链接的内置函数。由于**这些库只能用于 NVIDIA GPU，我们必须在下一步替换这些内置函数**。在表 1 中，我们展示了几个在高性能计算程序中广泛使用的内置函数。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/08.webp)

### 3.3 将 NVVM 转换为 SPIR-V

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/09.webp)

在图 3 中，我们展示了转换器的概述。:mdnice-bold-italic[输入的 NVVM IR 包括三个部分]：**元数据** （如设备版本、函数属性）、**NVVM 内置函数声明** （如用于获取线程索引的函数）和**设备无关指令**。

#### 3.3.1 元数据

> 元数据用于记录函数名称、数据布局、地址长度等信息。大多数元数据与后端设备无关，因此我们可以直接复制它们。因此，在我们的向量加法示例中，我们删除了所有元数据，只保留与后端相关的元数据。

对于“target triple”，我们必须将其修改为相应的 SPIR-V triple。对于一些 CUDA 特有的元数据，我们要么直接丢弃，要么将其转换为相应的 SPIR-V 元数据。在 NVVM 中，`nvvm.annotations`用于表示整个程序的内核函数，这些函数在 CUDA 源代码中用`__global__`标记。然而，:mdnice-bold-italic[SPIR-V 使用不同的方法记录内核函数：那些具有“kernel_arg_type”元数据的函数]。

因此，对于每个给定的 NVVM，我们必须记录`nvvm.annotations`所记录的函数，删除该元数据，并为这些函数添加新的“kernel_arg_type”元数据。

最后，由于 CUDA 的一些假设，某些元数据在 NVVM 中是不需要的。但这些假设在 SPIR-V 中并不存在。因此，我们必须显式添加这些元数据。例如，在 CUDA 中，当输入变量是指针时，它们应该始终指向全局内存。因此，没有元数据来表示每个输入指针的内存层次结构。而 SPIR-V 没有这个限制，我们必须添加一个额外的元数据“kernel_arg_addr_space”来记录此信息，以告诉编译器和后端设备这些指针都指向全局内存中的地址。

#### 3.3.2 内置函数

内置函数是特定于给定后端设备或框架的函数。它们是每个设备的关键，因为它们可以被视为框架的原语。尽管对于单指令多数据（SIMD）协议，所有设备都具有相同的原语：如获取线程索引、获取块索引、获取每个维度的长度等。但它们使用不同的内置函数来实现这些原语。

例如，在 NVVM 中，程序直接调用内置函数`llvm.nvvm.read.ptx.srge.tid.x`来获取 x 维度上的全局线程索引。**SPIR-V 使用不同的方法**：在 SPIR-V 中，有一个全局变量“GlobalInvocationId”，程序需要先从内存中加载这个变量，然后提取不同维度的索引。

#### 3.3.3 设备无关指令

在上述两节中，我们修改了 NVVM 和 OpenCL IR 之间不同的内容，这些内容无法通过现有的 OpenCL-SPIR-V 转换器进行转换。然而，**通常一个程序中只有一小部分属于这两类，大多数指令是设备无关的**。对于这些指令，我们可以直接使用现有的 OpenCL-SPIR-V 转换器。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/10.webp)

例如，在图 4 中，NVVM IR 中有四条指令。只有第一条指令（红色文本所示）是 NVVM 特有的，必须使用 NVVM-SPIR-V 转换器处理。其余指令（绿色文本所示）是设备无关的，可以直接使用现有的 OpenCL-SPIR-V 转换器进行转换。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/11.webp)

**分别处理设备相关和设备无关指令可以避免工作重复**，因为在开发 NVVM-SPIR-V 转换器时，我们不需要实现 OpenCL-SPIR-V 转换器中已有的功能。我们**在图 5 中展示了处理 NVVM IR 中指令的流程图**。转换后，我们可以得到图 2（b）所示的 SPIR-V IR。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/05.webp)

### 3.4 将 SPIR-V 转换为 OpenCL IR

我们需要这一步骤有两个原因：

::: block-1
1. 我们希望在 Vortex 上执行 SPIR-V，而 Vortex 只接受 OpenCL IR 作为输入；
2. SPIR-V 不具有人类可读性，换句话说，难以调试。
:::

因此，我们添加了这一步骤将 SPIR-V 转换为 OpenCL IR。在这一步骤中，我们直接调用 2.3.1 节中描述的 LLVM-SPIR-V 转换器。完成这一步后，我们将得到图 2（c）所示的 OpenCL IR。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/05.webp)

### 3.5 在 Vortex 上执行 OpenCL IR

最后一步是**在 Vortex 上执行 OpenCL IR**。在这一步骤中，我们有一个定制的主机代码（主机代码在 x86 处理器上运行，而内核程序在 RISC-V 或扩展的 RISC-V 处理器上运行），用于准备输入数据、设置内核函数的参数类型、为输入和输出分配内存缓冲区、调用 Vortex 相应的内核启动函数，最后获取结果并进行验证。

## 四、详细描述与实验

在本节中，我们将详细介绍为了支持 CUDA 的几个重要特性（所需要开展的工作）。

### 4.1 支持内置函数

#### 4.1.1 网格/块信息

CUDA 最重要的特性之一是它支持 SIMD（单指令多数据，一种并行计算方式，一条指令同时操作多个数据）。实现 SIMD 的关键是通过独特的 API（应用程序编程接口，用于不同软件组件之间交互的规则）为多个线程分配不同的索引。虽然我们已经介绍了如何在 RISC-V 上支持 CUDA 中的`threadIdx.x`（CUDA 中用于获取线程在 x 维度上的局部索引的内置变量），但还有一些细节需要明确说明。NVVM（NVIDIA 的中间表示，一种编译器用于表示 GPU 计算内核的中间代码）会调用不同的函数（代码 3）来获取不同维度的索引。

**代码3：用于获取索引的 NVVM 内置函数**![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/12.webp)

而 SPIR-V（一种用于并行计算和 GPU 图形的中间表示，由 Khronos 集团制定）会先从内存中加载一个全局变量，然后提取不同位置的维度，如代码 4 所示。

**代码 4：用于获取索引的 OpenCL 内置函数**![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/13.webp)

为了将 NVVM 内置函数翻译为 SPIR-V，我们必须将`@llvm.nvvm.read.ptx.sreg.ctaid.x()`的调用指令替换为两条连续的指令：加载指令和提取指令。此外，我们需要分析调用指令的函数名以获取该指令所需的维度，并将该维度作为参数传递给提取指令。如果函数名以 x 结尾，则对应维度 0；如果以 y 或 z 结尾，则分别对应维度 1 或 2。

#### 4.1.2 屏障（barrier，用于线程同步的机制）

NVVM 有一个简单的 API（代码 5）来实现块内所有线程之间的同步。

**代码 5：用于同步的 NVVM 内置函数**

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/14.webp)

OpenCL 的同步函数 barrier 具有不同的原型；它需要一个参数来指定需要同步的内存地址空间：`CLK_LOCAL_MEM_FENCE`（用于局部内存）和`CLK_GLOBAL_MEM_FENCE`（用于全局内存）。实际上，OpenCL 的 barrier 是一个确保全局/局部内存的内存操作按正确顺序执行的函数，而不像 NVVM 中的`barrier0`那样是一个实际的同步屏障（所有线程在此时等待，直到所有线程都到达该点）。对于 NVVM，`barrier0`用于将块内所有线程同步到同一点。由于块内线程可以访问全局内存和局部内存（共享内存），我们必须确保它们对这两种内存的操作顺序一致。因此，我们将 NVVM 的 barrier 翻译为 OpenCL 的 barrier，并使用一个参数来确保局部内存和全局内存的操作顺序，如代码 6 所示。

**代码 6：用于同步的 OpenCL 内置函数**

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/15.webp)

#### 4.1.3 原子操作（Atomic，一种不可被中断的操作，确保多线程访问共享数据时的正确性）

CUDA 提供了多种原子操作，如`atomicAdd`（原子加法）、`atomicSub`（原子减法）、`atomicExch`（原子交换）等。在 NVVM 中，所有这些操作都通过`atomicrmw`指令（原子读写修改指令，一种在操作过程中不会被其他线程中断的指令）来表示，并带有不同的操作（如加法、减法、交换等）（代码 7）。

**代码 7：NVVM 中的原子操作**

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/16.webp)

在 OpenCL 中，这些原子操作被视为普通的函数调用（代码 8）。:mdnice-bold-italic[为了将 NVVM 翻译为 OpenCL IR，我们的流水线需要提取每个`atomicrmw`指令中的操作，并根据该操作选择对应的 OpenCL 函数名]。之后，将 NVVM 指令中的参数直接复制到 OpenCL 指令中，因为它们具有相同的参数原型（第一个参数为指针，第二个参数为 32 位整数）。

**代码 8：OpenCL 中的原子函数**

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/17.webp)

### 4.2 基准测试实验

我们还尝试对基准测试程序进行翻译。在表 2 中，我们记录了 Rodinia[2]基准测试集中应用程序的翻译结果。

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/18.webp)

:mdnice-bold-italic[在支持了网格/块信息、屏障和原子指令等特性后，我们成功翻译了大部分应用程序]。然而，**仍有一些应用程序尚未支持**。这些应用程序要么使用了**纹理** （texture，GPU 中用于高效处理图像等数据的特殊内存），要么使用了一些**数学函数**。

## 五、结论

> 我们展示了一种在 RISC-V 后端设备上执行 CUDA 源代码的方法。为了验证其可行性，我们构建了一个流水线，该流水线能够成功执行具有多种特性的多个 CUDA 源代码，包括多线程、多块、原子操作和同步。

我们的流水线包括四个步骤：将 CUDA 源代码编译为 NVVM，将 NVVM 翻译为 SPIR-V，使用修改后的 POCL（Portable Computing Language，一种 OpenCL 标准的高效实现）生成目标文件，最后在开源 RISC-V GPU 架构上执行生成的目标文件。

::: block-1
除了编译 NVVM 所需的 CUDA 工具包外，所有其他组件都是开源的，并且可以在 Github 上轻松找到。

我们还构建了一个支持将 NVVM 翻译为 SPIR-V 的翻译器。该翻译器轻量，仅依赖 LLVM。它可以在没有 CUDA 工具包和 GPU 的情况下执行。
:::

我们的实验结果表明，我们的翻译器能够支持 Rodinia 中的大多数应用程序。未来，我们将尝试支持剩余的应用程序。具体来说，我们将**支持纹理内存和数学函数**，不仅要:mdnice-bold-italic[实现从 NVVM 到 SPIR-V 的转换]，还要支持**在 RISC-V GPU 上执行我们生成的 SPIR-V 所需的相应库**。

## 参考文献

![文章插图](/articles/2025-07-25-risc-v-gpu-cuda-nvvm-2f75b6b6/images/19.webp)
