关键词：AIEBLAS、AMD AI Engine、BLAS（基础线性代数程序库）、空间架构（Spatial Architectures）、数据流编程（Dataflow Programming）、自动代码生成（Automatic Code Generation）

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/01.webp)

- Developing a BLAS library for the AMD AI Engine
- https://arxiv.org/abs/2410.00825
- https://github.com/atlarge-research/AIE-BLAS
- 本文 4494 字，阅读需 15 分钟，播客 7 分钟

相关推荐

- [基于 LLM 跨硬件GEMM优化框架 QiMeng-GEMM：自动搜索元提示组合下特定规模显著优于cuBLAS、OpenBLAS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889763&idx=1&sn=21f80712d58b5241461c6a1fe4098951&scene=21#wechat_redirect)
- [GPGPU BLAS 计算库 CLBlast: A Tuned OpenCL BLAS Library](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886014&idx=1&sn=87ac9e9eee71b5f625df41912990115a&scene=21#wechat_redirect)
- [7.46倍性能提升！结果重用GEMM加速器Transitive Array，LLaMA模型实现3.97倍加速与1.65倍能耗降低](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890742&idx=1&sn=981ff2fb5277b9aa18d08e0c875b2909&scene=21#wechat_redirect)

> 当 Dennard 缩放定律与摩尔定律逐渐走向终结，机器学习等领域激增的计算需求，正推动计算机架构从传统冯·诺依曼模型向更高效的空间架构转型。

以 AMD AI Engine 为代表的空间设备，凭借数十至数千个处理单元构成的二维网格与可重构片上网络，成为当下热门的机器学习推理加速器，且已融入消费级 CPU 与数据中心加速器。

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/02.webp)

::: block-1
然而，这些具备海量并行潜力的设备，却在计算科学、图处理等非机器学习领域遭遇普及瓶颈——底层编程接口（如AMD ADF）学习曲线陡峭，可复用软件库的缺失，让非专家群体难以驾驭其算力。

在此背景下，来自阿姆斯特丹自由大学的研究团队提出了:mdnice-bold-italic[开源项目 AIEBLAS：一款专为 AMD AI Engine 设计的基础线性代数程序库（BLAS）]。该库通过自动代码生成，将用户的高层需求转化为硬件适配代码，既提供可定制、易集成的数值计算接口，又能借助数据流编程优化片上通信，无需用户深入理解底层硬件细节。
:::

本文不仅:mdnice-bold-italic[详述了 AIEBLAS 的设计架构与开发流程，还通过在AMD VCK5000平台的测试，验证了数据流组合对性能的提升作用]，并指出了后续优化方向，为空间架构在更广泛科学领域的应用开辟了新路径。尽管本文的研究重点是 AIE 架构，但我们认为，类似的设计原则和思路也可应用于其他空间架构。

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/03.webp)

## 关键问题

### 问题1. 研究团队开发AIEBLAS库的背景和动机是什么？为何要针对AMD AI Engine（一种空间架构）构建BLAS库，现有技术或生态存在哪些不足？

背景上来讲：

::: block-1
- 一方面，**Dennard缩放和摩尔定律的终结**，以及机器学习等领域日益增长的计算需求，促使业界和研究者重新思考传统计算机架构，而:mdnice-bold-italic[空间架构（如AMD AI Engine所在的ACAP平台）作为脱离冯·诺依曼模型的替代方案，能更高效利用晶体管和芯片空间]，当前主要作为机器学习推理加速器推广；
- 另一方面，尽管空间架构具备大规模并行性和高性能潜力，但其在计算科学、图处理等非机器学习科学领域的应用未被充分探索。
:::

核心障碍在于**低级别API（如AMD ADF）学习曲线陡峭，且缺乏可重用软件**，导致非专家难以利用这些设备的能力。基于此，研究团队开发AIEBLAS库，以解决上述痛点，推动空间架构在更多领域的应用。

### 问题2. AIEBLAS库的核心定位、主要目标以及开发流程是怎样的？如从用户输入到最终生成可运行代码

- **核心定位**：AIEBLAS是一个**开源、可扩展的Basic Linear Algebra Routines（BLAS）实现**，专门针对AMD AI Engine（一种空间架构）设计，旨在让数值例程易于重用、定制和集成到数据流程序中，且无需用户深入理解底层硬件和编程模型。
- **主要目标**：一是提供即用型数值例程，支持定制和与其他代码集成，避免用户编写复杂低级别代码；二是易于扩展新功能和优化；三是通过数据流方式自然支持片上通信，提升效率。
- **开发流程**：

::: block-1
- 首先，用户通过JSON文件指定所需例程特征（如例程类型、核的唯一名称，可选非功能参数如窗口大小，未指定则使用默认值）；
- 接着，AIEBLAS基于该JSON规范，通过基于模板的代码生成器，生成四部分设计内容——实现BLAS例程的AIE核、与设备DRAM交互的PL核、执行程序并连接AIE核的数据流图、用于构建设计的CMake项目；
- 最后，编译生成的代码并在目标设备（如AMD VCK5000）上运行，用户还可在JSON中指定核间连接或布局约束（针对大型设计）。
:::

### 问题3. AIEBLAS库在AMD VCK5000平台上的初步性能评估有哪些关键结论？研究团队后续计划从哪些方向完善该库？

- **关键性能结论**：评估基于axpy（向量加法）、gemv（矩阵-向量乘法）、axpydot（组合例程），使用AMD Vitis v2022.2、GCC 11.4.0编译，对比CPU（OpenBLAS 0.3.27）性能：
- **后续完善方向**：一是:mdnice-bold-italic[提升性能，包括优化片外内存访问、系统支持多AIE例程以利用空间并行性、支持分块减少片外请求]；二是扩大BLAS覆盖范围，实现更多例程（参考现有先进实现）；三是通过开源（库地址：https://github.com/atlarge-research/AIE-BLAS）吸引社区参与，共同开发适用于当前及未来空间架构的软件库。

## 本文目录

- 关键问题
  - 问题1. 研究团队开发AIEBLAS库的背景和动机是什么？为何要针对AMD AI Engine（一种空间架构）构建BLAS库，现有技术或生态存在哪些不足？
  - 问题2. AIEBLAS库的核心定位、主要目标以及开发流程是怎样的？如从用户输入到最终生成可运行代码
  - 问题3. AIEBLAS库在AMD VCK5000平台上的初步性能评估有哪些关键结论？研究团队后续计划从哪些方向完善该库？
- 一、引言
- 二、背景
  - 2.1 高层架构
  - 2.2 可编程逻辑（Programmable Logic, PL）
  - 2.3 自适应数据流（Adaptive Dataflow, ADF）
- 三、设计与实现
- 四、初步结果
  - 单个例程
  - axpydot 例程
- 五、结论与未来工作
- 参考文献

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/04.webp)

## 一、引言

> Dennard 缩放定律和摩尔定律的终结，以及机器学习等应用领域日益增长的计算需求，正促使业界和研究人员重新思考传统的计算机体系结构。

::: block-1
- Dennard 缩放定律：指指晶体管尺寸缩小带来的功耗与性能提升规律，现已逐渐失效。
- 摩尔定律：指集成电路上可容纳的晶体管数量约每两年翻一番的规律，目前增速放缓
:::

一种极具潜力的替代方案是**空间架构（Spatial Architectures）** ——这种架构脱离了冯·诺依曼模型，:mdnice-bold-italic[旨在更高效地利用可用晶体管和芯片空间。]

如今，空间架构主要作为机器学习（ML）工作负载加速器推向市场，例如 AMD/赛灵思（Xilinx）的 ACAP 平台[1]、Sambanova 可重构数据流架构[2]以及 Cerebras 晶圆级引擎[3]。

> 这类设备拥有数十到数千个**处理单元（Processing Elements）**，这些单元以二维网格形式排列，并通过高速、可重构的:mdnice-bold-italic[片上网络（Network-On-Chip, NoC，一种在芯片内部连接多个处理单元的专用网络，能减少数据传输延迟）] 进行通信。所有**空间架构的共同特点是**，它们适合采用:mdnice-bold-italic[数据流编程模型（dataflow programming model，一种以数据流动路径为核心的编程范式，数据在哪里需要就流向哪里，而非依赖指令控制流，能减少控制开销并优化并行性）] 进行**编程，从而更利于芯片内数据移动，并降低控制开销。**

考虑到其核心用户群体，空间架构制造商提供了与主流高级机器学习框架（如 PyTorch 和 TensorFlow[4]-[6]）的集成支持，并发布了预定义模型[7]、[8]，方便用户将这些设备用于推理任务。

尽管空间架构具有大规模并行性和高性能的潜力，但科学界尚未充分探索其在机器学习之外领域（如计算科学或图处理）的应用。在这些领域中，程序员必须依赖:mdnice-bold-italic[底层应用程序接口（lower-level APIs，抽象程度低、直接与硬件交互的编程接口，使用难度较高）] （例如 AMD ADF[9]、Cerebras CSL[10]）才能充分发挥设备的性能。然而，这些 API 陡峭的学习曲线以及可复用库的缺乏，使得非专业人士难以探索和利用这些新型设备。

> 在本文中，我们介绍正在进行的项目**AIEBLAS** ——这是一个面向 AMD AI 引擎（AIE，一种空间架构，目前已集成到商用 CPU[11]和数据中心加速器[1]中）的开源、可扩展:mdnice-bold-italic[基本线性代数子程序（Basic Linear Algebra Routines, BLAS，一套用于数值计算的标准子程序库，包含向量运算、矩阵运算等基础功能）] 实现。我们为 AIEBLAS 库设定的目标如下：

::: block-1
1. 提供即用型数值子程序，这些子程序可定制且能与其他代码集成，无需用户编写冗长复杂的底层代码；
2. 便于扩展新功能和优化手段；
3. 采用数据流方法，自然地优先支持芯片内通信。
:::

尽管本文的研究重点是 AIE 架构，但我们认为，类似的设计原则和思路也可应用于其他空间架构。

## 二、背景

> AIEBLAS 的目标平台是:mdnice-bold-italic[Versal 自适应计算加速平台（Versal Adaptive Compute Acceleration Platform, ACAP，AMD 推出的异构计算平台，集成了 AI 引擎、可编程逻辑、处理器等组件，专为高性能计算场景设计）] 设备。

### 2.1 高层架构

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/02.webp)

图 2 展示了 VCK5000 开发板[12]的高层架构：

- AIE 阵列由 8×50 的网格组成，共包含 400 个 AI 引擎（AIE）。
- 每个 AI 引擎内置一个:mdnice-bold-italic[超长指令字向量处理器（Very Long Instruction Word vector processor，一种能一次性处理多条指令、专门优化向量数据运算的处理器，可提升数据并行处理效率）] 和 32KB:mdnice-bold-italic[本地内存（local memory，每个 AI 引擎专属的小型高速内存，访问速度远快于外部内存，用于暂存高频访问数据）]。

### 2.2 可编程逻辑（Programmable Logic, PL）

> 可编程逻辑（Programmable Logic, PL），是指芯片中可通过编程灵活配置功能的硬件部分，类似 FPGA，可根据需求实现自定义逻辑电路。

可编程逻辑组件包含**逻辑块、内存** 和:mdnice-bold-italic[数字信号处理单元（Digital Signal Processing units, DSPs，专门用于高效执行数字信号滤波、变换等运算的硬件模块）]，可用于在硬件中实现自定义逻辑。

AIE 阵列与 PL 之间通过多个 AXI 接口（312 个从 PL 到 AIE 的接口，234 个从 AIE 到 PL 的接口）通信，每个接口的传输速率为 4GB/s。

### 2.3 自适应数据流（Adaptive Dataflow, ADF）

> AI 引擎可通过**自适应数据流（Adaptive Dataflow, ADF）API[9]** 进行编程：:mdnice-bold-italic[应用程序以“部署在 AI 引擎上的内核数据流图”形式表示]，计算内核代码通过底层 NoC 和相邻接口，以“块（即窗口，window）”或“逐元素（stream）”的方式交换数据。

注：**“窗口数据” 是指以 “块（Block）” 为单位存储在 AIE 本地内存（Local Memory）中的数据集合**，:mdnice-bold-italic[主要用于承载向量、矩阵等非标量数据（标量数据则通过 “流（Streams）” 传输）。] 每个 AIE 拥有 32KB 本地内存，窗口数据直接存于此处，而非依赖片外 DRAM 或片上共享内存 —— 这是它与普通 “数据块” 最核心的区别：存储位置绑定 AIE 本地内存，且访问方式适配 AIE 的硬件架构。

PL 则可通过:mdnice-bold-italic[高级综合（High-Level Synthesis, HLS，一种将 C/C++等高级编程语言自动转换为硬件描述语言的技术，降低硬件开发门槛）] 或:mdnice-bold-italic[寄存器传输级（Register Transfer Level, RTL，硬件描述语言的一种抽象层级，用于精确描述数字电路中寄存器与组合逻辑之间的数据传输和运算过程）] **进行编程。**

```
{"kernels": [    { "blas_op": "axpy", "name": "vadd", "type": "float", "vsize": 16 ,...},    { "blas_op": "dot", "name": "vdot", ...}],"connections" : [    { "in": {"kernel": "vadd", "parameter": "out"}, "out": {"kernel": "vdot", "parameter": "x"} }, ...]}
```

上述为 AIEBLAS 中用于定义内核和连接关系的 JSON 配置示例，其中“axpy”、“dot”为 BLAS 标准子程序，“vadd”“vdot”为自定义内核名称，“connections”定义了内核间的数据流向——将“vadd”的输出“out”作为“vdot”的输入“x”。

## 三、设计与实现

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/05.webp)

> 图 1 展示了 AIEBLAS 的通用开发流程。用户需在 JSON 文件中指定所需的例程特性，说明例程类型信息以及用于内核生成的唯一名称。用户还可选择性地指定非功能参数（如窗口大小），若不指定则默认使用预定义值。

从 JSON 高层级规范出发，AIEBLAS 会生成一个设计方案，该方案包含：

::: block-1
1. 用于实现所需 BLAS 例程的 AIE 内核；
2. 用于与设备 DRAM 进行数据收发的 PL 内核；
3. 用于执行程序且（若适用）按指定方式连接 AIE 内核的数据流图；
4. 用于构建该设计方案的 CMake 项目。不同的基于模板的代码生成器负责为各类设计组件生成代码。所有代码生成器均可便捷扩展，以实现新功能（如新增例程）或改进现有功能（如对某给定例程进行优化实现）。
:::

AIEBLAS 例程通过流（streams）接收和生成标量数据。对于向量和矩阵，我们让例程通过窗口（windows）接收和生成数据。该方法具有多项优势：

- 首先，窗口存储在本地内存中，相较于 AXI4 流互连，可通过更宽的数据通路对其进行访问，这使我们能够充分利用 AIE 向量处理器；
- 其次，窗口可实现两个通信 AIE 之间的通信解耦，这对于片上通信十分有用。内核代码经过向量化处理，以充分利用 AIE 的计算能力。用户可在 JSON 规范中设置向量宽度，该宽度默认值为支持的最大宽度（512 位）。

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/05.webp)

数值计算可由两个或多个共享数据的例程构成。例如，图 1 中的示例计算 axpydot（即$\beta=z^{T} u$，其中$z=w-\alpha v$；w、v、u 均为向量，α 和 β 均为标量[13]）。该计算可通过以下方式实现：首先执行向量加法（axpy），然后将其输出作为后续点积（dot product）的输入。我们希望:mdnice-bold-italic[优先采用片上通信，将例程构建为数据流图，而非通过片外内存交换数据。通过这种方式，我们可减少开销高昂的片外访问次数，并支持多个例程的流水线执行。]

AIEBLAS 允许用户在 JSON 规范中指定 BLAS 例程之间的连接关系，代码生成器将据此生成相应的数据流图定义。若某例程的输入/输出未与其他例程连接，AIEBLAS 将创建一个 PL 内核，用于从片外内存加载/存储数据。默认情况下，AIEBLAS 依赖 AIE 编译器进行内核布局。然而，对于规模较大的设计，可能需要向编译器提供布局提示，以使其在合理时间内生成布局规划（floorplan）。为实现这一点，用户可在 JSON 配置中设置一个可选字段，为每个内核指定布局约束。

## 四、初步结果

> 我们在 AMD VCK5000 开发板上对 AIEBLAS 的当前实现版本进行了评估。

- 代码使用 AMD Vitis v2022.2（AMD 推出的嵌入式开发工具套件）和 GCC 11.4.0（GNU 编译器套件的 C/C++编译器）进行编译。
- 测试所用主机配备两台 10 核 Intel Xeon Silver 4210R 处理器（主频 2.4GHz，禁用超线程技术），并搭载 256GB DDR4 内存（第四代双倍数据速率同步动态随机存取存储器）。
- 在 CPU 基准测试中，我们采用 OpenBLAS 0.3.27（一款开源的高性能基本线性代数子程序库），并启用-O3 优化标志（GCC 编译器的高级优化级别，可提升代码执行效率）。
- 我们**测试的例程包括向量加法** （axpy，BLAS 中的向量缩放与加法函数）、**矩阵-向量乘法** （gemv，BLAS 中的矩阵-向量乘法函数），以及**组合运算 axpydot** （由 axpy 和 dot 运算组合而成的复合线性代数运算）。

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/06.webp)

::: image-caption
AIEBLAS 在不同输入规模下的评估结果。我们考虑了两种实现方案：一种是数据存储在片外内存且通过可编程逻辑（PL）中的数据传输单元实现数据移动的方案，另一种是数据在 AIE 阵列（无 PL 参与）上人工合成生成的方案。对于 axpydot 例程，我们同时考虑了数据流（w/ DF，即 with Dataflow）和非数据流（w/o DF，即 without Dataflow）两种实现方案。
:::

> 图 3 展示了不同输入规模下的平均执行时间。

### 单个例程

对于单个例程，我们测试了两种实现方案：一种是使用可编程逻辑内核从 DRAM读写数据，另一种是数据在芯片内部直接生成。

后者的执行时间更短，这**凸显了片外访问对内存受限型计算性能的影响**。这一结果强调，:mdnice-bold-italic[有必要优化片外内存读取] （例如通过突发传输——一种内存访问优化技术，通过一次性连续传输多个数据块减少数据传输延迟），:mdnice-bold-italic[并考虑使用多个 AXI 接口] （一种常用的片上总线接口标准，用于芯片内部不同组件间的高速数据传输），:mdnice-bold-italic[充分利用各类 AIE-PL 接口。]

### axpydot 例程

对于 axpydot 例程，我们评估了数据流和非数据流两种实现方案。**正如预期，数据流方案的性能提升了一倍**，这:mdnice-bold-italic[表明流水线执行（一种并行处理技术，将计算任务分解为多个步骤，使不同步骤可同时执行以提升效率）能显著优化性能]。

在这种情况下，优化内存访问同样有助于提升性能。

---

最后，在所有测试场景下，:mdnice-bold-italic[CPU 的性能总体上优于 AIE 实现（最高可达 10 倍）。这是由于 OpenBLAS 具备经过优化的多核实现方案，这也表明 AIE 若要实现具备竞争力的性能，还需进一步优化。]

## 五、结论与未来工作

> 在本文中，我们介绍了 AIEBLAS 的设计方案——这是一款:mdnice-bold-italic[面向 AMD AIE 空间架构的 BLAS 库。]

该库利用自动代码生成技术，根据用户的高层级规格说明**生成特定于架构的代码**，大幅提升了用户的工作效率。初步性能结果表明，:mdnice-bold-italic[为了优先支持片上通信并实现多个例程的流水线执行，数据流组合是必不可少的]。结果还**凸显了对更高空间并行度的需求**。

事实上，关键在于：

- :mdnice-bold-italic[挖掘 PL 中更高的并行度，充分利用多个 PL-AIE 接口]，并使可用的:mdnice-bold-italic[片外内存带宽达到饱和] （即使片外内存的传输能力被充分利用，无空闲带宽）；
- 同时，:mdnice-bold-italic[通过多 AIE 例程实现，即利用多个 AIE 计算单元共同执行一个例程]，即使对于通信受限型例程（指性能受限于数据通信速度而非计算速度的例程），也能提升其性能。

> 我们计划从多个方向继续开发 AIEBLAS：

::: block-1
1. 首先，我们计划**从以下方面提升其性能**：
2. 其次，我们希望通过实现更多例程来扩大 BLAS 的覆盖范围，同时也会参考当前最先进的实现方案[14]-[16]。
3. 最后，通过公开发布 AIEBLAS，我们希望吸引社区参与进来，为当前及未来的空间架构开发该库及其他软件库。
:::

## 参考文献

![文章插图](/articles/2025-09-09-amd-ai-engine-blas-aieblas-22b36204/images/07.webp)
