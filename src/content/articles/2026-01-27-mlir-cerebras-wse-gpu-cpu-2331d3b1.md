关键词：**Cerebras WSE**、:mdnice-bold-italic[MLIR]、模板计算、异步执行模型、**领域特定语言**、:mdnice-bold-italic[编译器流水线]

> 无需修改一行应用代码，即可在 90 万个核心的异步架构上实现超越 GPU 集群的性能

**当芯片规模突破单颗晶片的物理极限**，当计算核心数量飙升至 **90 万个**，传统编程模型与硬件执行模型之间的鸿沟便如同天堑。:mdnice-bold-italic[这就是 Cerebras 晶圆级引擎（WSE）面临的现实——一款为 AI 而生，却在科学计算领域展现出惊人潜力的革命性硬件。]

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/01.webp)

::: image-caption
同为台积电5nm工艺，Cerebras WSE-3在核心硬件参数上远超英伟达H100：前者裸片面积46225mm²，是后者826mm²的57倍；AI优化核心达90万颗，相较H100的16896颗核心+528颗张量核心，实现52倍增长。同时WSE-3性能数据亮眼，内存带宽21PB/s，是H100的7000倍，结构带宽214Pbit/s为其3715倍，片上内存44GB，更是达到H100的880倍，硬件优势显著。
:::

::: block-1
然而，为其编程犹如在迷宫中进行精密布线，需要开发者彻底重构算法以适配其**高度异步、基于 actor（执行者）的执行模型**。长期以来，这成为阻碍 WSE 在高性能计算（HPC）领域广泛应用的“阿喀琉斯之踵”。
:::

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/02.webp)

- :mdnice-bold-italic[An MLIR Lowering Pipeline for Stencils at Wafer-Scale]
- https://arxiv.org/pdf/2601.17754
- 7000 字，阅读 23 分钟，**播客 20 分钟**

相关推荐

- **[GEMM 95% FPU 利用率！基于 MLIR 面向 RISC-V 指令集扩展的加速Micro-kernels的多层编译器后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889407&idx=1&sn=f5640780406954ae498d2c98d162d454&scene=21#wechat_redirect)**
- **[谈谈AI编译软件栈、软硬件协同、TVM和MLIR的看法](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886637&idx=1&sn=477371ef67890c91b565b383493959b6&scene=21#wechat_redirect)**
- **[A100和MI300A上评估跨GPU语言Mojo的性能与可移植性：基于MLIR跨NVIDIA/AMD的Python兼容高性能语言](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892724&idx=1&sn=ad19190aaa36ff58f8c5030d071739d9&scene=21#wechat_redirect)**

> 今天，一项来自英国爱丁堡大学、剑桥大学等机构的研究打破了这一僵局。研究团队构建了一套基于 **MLIR（多级中间表示）** 的编译器流水线，能够**自动将描述数学问题的“模板计算”代码，转换为高度优化、适配 WSE 异步模型的底层代码**。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/03.webp)

::: image-caption
图 3 | 我们的分阶段编译流程扩展了现有基础设施，通过多个阶段面向新型硬件生成代码，实现了 Cerebras WSE 的高效代码生成。该流程整合多前端输入（Devito、PSyclone、Flang），先将 Python 或 Fortran 代码转化为 stencil 方言，再经五组核心变换（如维度分解、内存实例化、角色模型映射等），逐步将 IR lowering 至 csl-ir 方言。流程关键在于保留模板高层信息至编译后期，以适配 WSE 无缓存数据流架构，同时借助 xDSL 与 MLIR 的兼容性，复用如 stencil-inlining 等现有优化 passes，提升编译效率与代码性能。
:::

::: block-1
其神奇之处在于：:mdnice-bold-italic[科学家无需修改其用 Fortran 或 Python DSL 编写的应用代码]，即可让程序在 WSE 上高效运行，性能甚至超越了手工精心优化的版本。
:::

## 本文目录

- 一、晶圆级引擎：性能巨兽与编程噩梦
  - 1.1 什么是 Cerebras WSE？
  - 1.2 异步执行模型：与经典编程的范式冲突
- 二、核心洞见：用领域知识驱动编译器自动化
- 三、创新架构：三级 MLIR 方言桥接语义鸿沟
  - 3.1 csl-stencil 方言：面向通信的模板抽象
  - 3.2 csl-wrapper 方言：管理布局与分阶段编译
  - 3.3 csl-ir 方言：CSL 语言的 MLIR 镜像
- 四、编译流水线：五步转换的魔法
  - 4.1 第一组：分解与数据依赖分析
  - 4.2 第二组：物理映射与通信实现
  - 4.3 第三组：内存具体化
  - 4.4 第四组：映射到 Actor 执行模型
  - 4.5 第五组：降至 csl-ir 并生成代码
- 五、性能评估：超越手工代码与 GPU 集群
  - 5.1 对阵手工优化代码：更优且更通用
  - 5.2 对阵传统 CPU/GPU 集群：数量级的优势
  - 5.3 屋顶线分析：从“内存墙”到“计算墙”
  - 5.4 开发者生产力：代码行数锐减
- 六、相关研究与定位
  - 6.1 早期 WSE 编程探索
  - 6.2 同期 WSE 编译技术
  - 6.3 其他 MLIR 生态工作
- 七、结论与展望

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、晶圆级引擎：性能巨兽与编程噩梦

### 1.1 什么是 Cerebras WSE？

> 想象一下，不是将数十个小型芯片封装在一起，而是将一整片晶圆（Wafer）直接作为一颗芯片。

Cerebras WSE 正是如此，它在单一晶圆上集成了高达 **90 万个独立处理单元**，通过晶圆级片上互联网络相连。其第三代产品 WSE-3 拥有 4 万亿个晶体管，**40GB 的片上 SRAM** （WSE-3 为 44GB），内存带宽高达 21 PB/s。这种规模将传统 CPU/GPU 的“内存墙”问题推向了新的维度。

::: block-1
但与 CPU/GPU 的统一内存架构不同，WSE 的 90 万个处理单元各自拥有独立的 **48KB SRAM**，没有缓存，没有内存层次结构。数据在 PE 间的共享必须通过显式的、**异步的消息传递** 来完成。
:::

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/05.webp)

::: image-caption
除了WSE-3晶圆级引擎芯片外，Cerebras Systems还发布了其CS-3人工智能超级计算机。得益于其庞大的内存池，该超级计算机能够训练比GPT-4和Gemini大10倍的模型。CS-3人工智能解决方案专为企业和超大规模用户设计，与现代GPU相比，其性能效率要高得多。
:::

### 1.2 异步执行模型：与经典编程的范式冲突

> WSE 的执行模型根植于 **“actor 模型”**。每个 PE 被视为一个**硬件 actor**：

- 拥有自己的时钟、程序计数器和私有内存。
- 通过四个双向通道（东、西、南、北）与邻居 PE 交换 32 位的“小波”消息。
- 通过 24 个虚拟通道管理路由。

在软件层面，计算被组织成**任务**，它们是**软件 actor**，由消息到达或内部事件触发执行。这意味着，传统的**同步循环、屏障等待** 等概念在 WSE 上不复存在。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/06.webp)

::: image-caption
图 1 | CSL 中围绕异步数据交换的时间步循环：上方为 Fortran 语言编写的简单雅可比模板程序，下方为对应的 Cerebras 专属编程语言 CSL 代码，清晰对比了传统同步编程与 WSE 异步编程的差异。上方 Fortran 代码以直观的嵌套循环实现时间步迭代，符合 CPU/GPU 编程习惯；下方 CSL 代码需拆解为任务与回调函数（如 for_cond0、seq_kernel0 等），通过 stencil.communicate 处理异步数据交换，还需应对 PE 内存限制采用分块数据传输，体现了 WSE 编程需重构算法执行逻辑的核心挑战
:::

如图 1 所示，一个在 Fortran 中简单的 `do step = 1, timesteps` 循环，在 CSL 中必须被“打碎”成一系列由回调函数链接的任务。程序员需要手动管理数据的分块传输、任务调度和回调链，极易陷入复杂的“回调地狱”。

这使得将现有科学计算代码移植到 WSE 上，不仅需要学习全新的语言，更需要进行**算法层面的根本性重构**，成本极高。

## 二、核心洞见：用领域知识驱动编译器自动化

> 面对这一挑战，研究团队的核心理念是：**将移植负担从应用科学家身上转移到编译器**。

他们提出了一个关键假设：:mdnice-bold-italic[用户代码中已有的、领域特定的抽象信息（如模板计算的结构），足以驱动编译器自动生成适合 WSE 的代码]。换言之，编译器可以利用“这是一个模板计算”这一高级语义信息，来智能地处理数据依赖、通信和异步执行调度。

为此，他们选择了 **MLIR** 作为实现这一愿景的基石。MLIR 不是一个单一的中间表示，而是一个可扩展的**编译器基础设施框架**，允许开发者创建不同抽象级别的“方言”，并定义它们之间的转换规则。

## 三、创新架构：三级 MLIR 方言桥接语义鸿沟

> 研究团队的核心贡献之一是设计了三层新的 MLIR 方言，像阶梯一样，逐步将高层次的数学问题描述，降低到符合 WSE 执行模型的底层表示。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/07.webp)

::: image-caption
图 2 | 通过使用三种 WSE MLIR 方言，我们逐步从模板的数学表示过渡到基于角色模型的 WSE 执行模型。三种方言分工明确且层层递进。stencil Dialect 保留模板计算的数学本质，csl-stencil Dialect 明确 WSE 特有的通信与计算操作，csl-wrapper Dialect 处理布局元程序与参数，最终通过 csl-ir Dialect 映射为 CSL 代码。这种设计让编译器能逐步弥合数学描述与 WSE 异步执行模型的语义鸿沟，同时复用 MLIR 生态优势，减少定制化开发工作量。
:::

上图通过三级 WSE MLIR 方言，从数学表示降至基于 Actor 的执行模型。现有的、与架构无关的 `stencil` 方言首先被转换为 WSE 特定的 `csl-stencil` 方言，然后被包装进 `csl-wrapper` 方言，最后所有内容被转换为 `csl-ir` 方言的操作，从而可以生成最终的 CSL 代码。

### 3.1 csl-stencil 方言：面向通信的模板抽象

> 这是从通用 `stencil` 方言到 WSE 世界的**第一个特化步骤**。它不再隐藏通信细节，而是将其**显式化**：

- `csl-stencil.prefetch`：描述从远端 PE 获取单块数据到本地缓冲区。
- `csl-stencil.apply`：包含**两个区域**。
- `csl-stencil.access`：根据偏移量访问模板值，自动判断是从本地内存还是通信缓冲区读取。

::: block-1
**关键优化**：`csl-stencil.apply` 支持**分块部分归约**。由于每个 PE 仅有 48KB 内存，无法一次性容纳所有邻居数据。该方言允许将通信分为 N 块，并在每块数据到达时立即进行归约，累加到一个单一的累加器缓冲区中，极大降低了瞬时内存需求。
:::

### 3.2 csl-wrapper 方言：管理布局与分阶段编译

> CSL 采用**分阶段编译**：首先运行一个**布局元程序**，该程序决定各个内核在 WSE 二维网格上的放置和路由；然后，编译器利用布局信息，为每个 PE 专门化并编译其程序。

`csl-wrapper` 方言是一个与领域无关的包装器，用于封装：

- 程序级参数。
- 布局元程序（描述如何在 WSE 网格上放置内核）。
- 实际的 PE 程序。 它处理了 CSL 特有的这种两阶段编译流程，为后续代码生成做好准备。

### 3.3 csl-ir 方言：CSL 语言的 MLIR 镜像

这是对 CSL 编程语言一个**大型子集的直接重新实现**。该方言中的操作与 CSL 语言结构一一对应。编译流水线的最终目标，就是将高级表示一直“降低”到 `csl-ir` 方言，然后通过一个打印机将其直接转换为可被 Cerebras SDK 编译的 CSL 源代码。

## 四、编译流水线：五步转换的魔法

> 从科学家编写的 Fortran 或 Python DSL 代码，到最终在 WSE 上飞奔的 CSL 代码，需要经过一系列精妙的转换。研究团队将其归纳为**五个核心转换组**。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/03.webp)

::: image-caption
图 3 | 我们的分阶段编译流程扩展了现有基础设施，通过多个阶段面向新型硬件生成代码，实现了 Cerebras WSE 的高效代码生成。该流程整合多前端输入（Devito、PSyclone、Flang），先将 Python 或 Fortran 代码转化为 stencil 方言，再经五组核心变换（如维度分解、内存实例化、角色模型映射等），逐步将 IR lowering 至 csl-ir 方言。流程关键在于保留模板高层信息至编译后期，以适配 WSE 无缓存数据流架构，同时借助 xDSL 与 MLIR 的兼容性，复用如 stencil-inlining 等现有优化 passes，提升编译效率与代码性能。
:::

上图图详细展示了从 Python/Fortran 源代码开始，经由现有前端（Devito, PSyclone, Flang）生成 stencil 方言，再通过论文提出的五组转换，最终生成 csl-ir 并输出 CSL 代码的完整流程。图中绿色虚线框标明了五组主要转换。

### 4.1 第一组：分解与数据依赖分析

> 目标：将三维模板计算映射到 WSE 的二维 PE 网格上。

- **步骤 1：空间分解**。对于一个 `X × Y × Z` 的三维网格，编译器将其分解为 `X × Y` 的二维网格，:mdnice-bold-italic[每个 PE 负责一个完整的 Z 维“柱子”]。这种分解已被证明在 WSE 上高效。
- **步骤 2：张量化**。将 Z 维度从标量循环转换为:mdnice-bold-italic[张量操作]。这样，每个 stencil 元素就变成了一个张量，整个计算变成了对二维张量网格的操作，完美匹配每个 PE 处理一个数据柱子的模型。

:mdnice-bold-italic[代码清单 3：在跨 PE 分布 X, Y 维度并张量化 Z 维度之后]

```
// 插入 dmp.swap 操作，指明必要的数据交换%1 = "dmp.swap"(%input) (strategy = ...) : ... -> tensor<...>// 接下来的 stencil.apply 操作现在是在张量上进行的result = "stencil.apply"(%1) ({ ... }) : ... -> ...
```

*注：此时代码中插入了 `dmp.swap` 操作，它来自 MPI 通信抽象方言，用于高级别地描述跨 PE 的数据交换需求。同时，计算变成了对 `tensor<512xf32>` 的操作，而非单个浮点数。*

### 4.2 第二组：物理映射与通信实现

> 目标：将逻辑数据依赖转换为 WSE 上具体的、分块的通信操作。

- **步骤：stencil 到 csl-stencil 的转换**。将上一步的 `dmp.swap` 操作替换为具体的 `csl-stencil.prefetch` 操作，并整合进 `csl-stencil.apply` 的两个区域中。同时，生成 `csl-wrapper` 来封装布局信息。

:mdnice-bold-italic[代码清单 4：转换到 csl-stencil 方言后]

```
csl-stencil.apply(...) num_chunks = 2 {  // 区域1：处理远端数据块 (例如，块0)  ^remote0(%chunk: tensor<255xf32>, %offset: index, %acc: tensor<510xf32>):    %val = csl-stencil.access %chunk[...]    %new_acc = tensor.insert_slice %val into %acc[%offset]    csl-stencil.yield %new_acc  // 区域2：处理本地数据，并使用区域1归约后的累加器  ^local(%local_field: ..., %acc: tensor<510xf32>):    %c = arith.constant ...    %local_val = csl-stencil.access %local_field[0,0]    %sum = arith.add %acc, %local_val    %result = arith.mul %sum, %c    csl-stencil.yield %result}
```

*注：这里清晰地展示了分块通信与计算的重叠。`num_chunks=2` 表示通信分两次进行，每次数据到达都触发 `^remote` 区域执行，进行部分归约。所有通信完成后，`^local` 区域执行，完成最终计算。*

### 4.3 第三组：内存具体化

> 目标：从“值语义”转换为“引用语义”，为 DPS 风格计算做准备。

CSL 的数学运算采用**目标传递风格**：函数直接对作为操作数传入的物理内存地址进行读写。而之前的 MLIR 操作使用的是不可变的张量（值语义）。

- **转换**：通过 MLIR 内置的 `bufferization` 通行，将张量转换为 `memref`（内存引用）。同时，将 `arith` 方言中的计算操作转换为支持 DPS 的 `linalg`（线性代数）方言操作。

### 4.4 第四组：映射到 Actor 执行模型

> 这是**最核心、最具挑战性** 的一步，旨在解决同步控制流与异步硬件之间的矛盾。

- **步骤 1：将计算区域拆分为软件 Actor**。`csl-stencil.apply` 的 `^remote` 区域被转换为一个**数据任务**，每次接收到一块数据时被触发。`^local` 区域被转换为一个**本地任务**，在所有数据块接收完毕后被触发。
- **步骤 2：将顶层控制流转换为任务图**。例如，一个顶层的 `scf.for` 时间步循环，不能被直接翻译为 CSL 循环。编译器会将其:mdnice-bold-italic[转换为一个由回调函数链接的控制流任务图]。每个“迭代”的发起，依赖于前一个迭代最终任务完成的回调。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/06.webp)

::: image-caption
图 1 | CSL 中围绕异步数据交换的时间步循环：上方为 Fortran 语言编写的简单雅可比模板程序，下方为对应的 Cerebras 专属编程语言 CSL 代码，清晰对比了传统同步编程与 WSE 异步编程的差异。上方 Fortran 代码以直观的嵌套循环实现时间步迭代，符合 CPU/GPU 编程习惯；下方 CSL 代码需拆解为任务与回调函数（如 for_cond0、seq_kernel0 等），通过 stencil.communicate 处理异步数据交换，还需应对 PE 内存限制采用分块数据传输，体现了 WSE 编程需重构算法执行逻辑的核心挑战
:::

这个过程自动解决了图 1 中程序员需要手动解决的“回调地狱”问题，将复杂的异步协调工作完全隐藏在编译器背后。

### 4.5 第五组：降至 csl-ir 并生成代码

> 最后一步是生成高效的 CSL 代码。

- **利用数据结构的描述符**：WSE 的最佳实践是使用 **DSD**。编译器将 `linalg` 操作（如 `linalg.add`）转换为调用 CSL 内置的高吞吐量 DSD 函数，例如 `@fmacs(dest, src1, src2, val)` 实现融合乘加，而不是生成低效的逐元素循环。
- **生成布局代码和最终模块**：将 `csl-wrapper.module` 转换为两部分 `csl-ir` 代码：一是布局元程序，二是每个 PE 上运行的核心程序。

## 五、性能评估：超越手工代码与 GPU 集群

> 研究团队使用来自 **Flang、Devito、PSyclone** 三个前端的五个模板基准测试程序，在 **WSE-2 和 WSE-3** 上评估了其编译器的性能。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/08.webp)

::: image-caption
图 4 | 在采用我们的方法、运行大型问题规模的情况下，不同基准测试程序在 WSE2 与 WSE3 上的性能对比。我们的代码生成器能利用最新硬件特性实现最优性能。测试涵盖 Jacobian（Flang）、Diffusion（Devito）、Seismic（Cerebras）、UVKBE（PSyclone）四类基准程序，均采用大型问题规模以充分利用硬件资源。WSE3 性能提升主要源于其改进的交换逻辑 —— 无需像 WSE2 那样让 PE 向自身传输数据，通信效率显著提高。结果验证了该编译方法能跨 WSE 硬件代际适配，且有效发挥新一代硬件性能优势。
:::

### 5.1 对阵手工优化代码：更优且更通用

> Cerebras 工程师曾手工优化了一个 **25 点** 地震建模模板内核，作为 WSE-2 的标杆。

- **性能**：手工代码达到了 WSE-2 理论峰值性能的 28.2%。而:mdnice-bold-italic[本文编译器生成的代码，性能反超手工代码高达 7.9%。]
- **优势分析**：
- **可移植性**：手工代码仅限于 WSE-2。而本文编译器**无缝支持 WSE-2 和 WSE-3**。在 WSE-3 上，得益于硬件交换逻辑的改进，生成代码的性能相比 WSE-2 版本进一步提升 **38.1%**。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/09.webp)

::: image-caption
图 5 | 地震应用基准测试程序中，我们基于编译器的方法与 Cerebras 团队开发的手写代码 [19] 的性能对比。高效的通信与任务利用使我们生成的代码在性能上更具优势。该测试针对 25 点地震模板，对比了三种方案在不同问题规模（100×100×450、500×500×450、750×994×450）下的性能。手写代码仅支持 WSE2，而生成代码同时兼容 WSE2 与 WSE3，且在 WSE2 上性能比手写代码高 7.9%。优势源于生成代码更优的内存效率（单块传输替代双块）、精准数据通信（仅传输必要数据）及高效任务管理（任务数量减少约 50%），突破了手写代码的优化瓶颈。
:::

### 5.2 对阵传统 CPU/GPU 集群：数量级的优势

> 为了体现 WSE 的绝对性能，团队将 Devito 声波基准测试在 **WSE-3** 上的表现，与文献中报道的使用 **128 个 NVIDIA A100 GPU** 和 **128 个 CPU 节点** （共 16384 核）的 Cray-EX 超级计算机的结果进行对比。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/10.webp)

::: image-caption
图 6 | Devito 声学基准测试程序在 WSE3（大型问题规模，100,000 次迭代）上的性能，与在 Tursa 超级计算机 128 块 A100 GPU 上运行的 MPI+OpenACC 内核、在 ARCHER2 Cray-EX 超级计算机 128 个 CPU 节点上运行的 MPI+OpenMP 内核的性能对比 [5]。单晶圆实现了与大型 GPU 或 CPU 集群相当的吞吐量，彰显了 WSE3 的高吞吐量能力。尽管测试中 GPU/CPU 采用更大问题规模以降低通信开销，WSE3 仍展现显著优势 —— 比 128 块 A100 GPU 快 14 倍，比 128 个 CPU 节点（16384 核）快 20 倍。这源于 WSE3 的大规模 PE 阵列与高带宽互连，能避免传统集群的跨节点通信瓶颈。同时，Devito 代码无需修改即可跨 WSE3、GPU、CPU 运行，验证了该编译方法在保证性能的同时，还能提升程序的硬件可移植性。
:::

::: block-1
**结论令人震撼**：:mdnice-bold-italic[单颗 WSE-3 晶圆，在特定问题上的“求解时间”性能，超越了由 128 块顶级 GPU 或 128 个高端 CPU 节点组成的大规模集群。] 这充分展示了 WSE 架构在解决通信密集、计算规则的科学计算问题时的巨大吞吐量优势。
:::

### 5.3 屋顶线分析：从“内存墙”到“计算墙”

> 屋顶线模型揭示了性能瓶颈所在。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/11.webp)

::: image-caption
图 7 | 五个基准测试程序在 WSE3 上的性能上限图（Roofline Plot），以及声学基准测试程序在单块 A100 GPU 上的性能上限图。图中虚线代表硬件理论性能上限，WSE3 标注了本地内存带宽（18.22 PB/s）、互连带宽（3.30 PB/s）与峰值计算性能（1.52 PFLOP/s），A100 标注峰值计算性能（17.59 TFLOP/s）。所有 WSE3 基准程序在假设数据本地存储时均为计算受限，仅 Jacobian 在假设数据通过互连传输时为内存受限；而 A100 上的声学程序为内存受限。这表明该编译方法能让程序充分利用 WSE3 的存储与计算资源，摆脱传统 GPU 的内存瓶颈，大幅提升性能。
:::

::: block-1
**核心洞见**：传统的 GPU/CPU 架构运行这些模板代码时，性能受限于内存带宽（“内存墙”）。而**通过本文编译器的转换，相同的代码在 WSE 上运行时，瓶颈转移到了计算能力上（“计算墙”）**。这意味着，:mdnice-bold-italic[编译器成功地将应用从一个受限于内存的架构，迁移到了一个能充分发挥其计算潜力的架构上，且无需用户干预]。
:::

### 5.4 开发者生产力：代码行数锐减

> 编程复杂度的降低，直接体现在需要编写的代码行数上。

![文章插图](/articles/2026-01-27-mlir-cerebras-wse-gpu-cpu-2331d3b1/images/12.webp)

::: image-caption
表 1 | 不同 WSE 硬件代际的关键架构参数对比，包含计算单元（PE）数量、互连带宽、本地内存容量与带宽等核心指标。WSE3 相较 WSE2 在核心硬件参数上全面升级，PE 数量提升至 8 倍，互连带宽、本地内存容量与带宽均实现翻倍，峰值计算性能提升超 7 倍。这些硬件升级让 WSE3 能处理更大规模问题，同时降低数据通信与内存访问的瓶颈，也对编译流程的硬件适配能力提出了更高要求，需充分利用新增的 PE 与带宽资源优化代码执行逻辑。
:::

如表 1 所示，使用 DSL 和本编译器，科学家需要维护的代码量**减少了一个数量级以上**。更重要的是，DSL 代码是领域科学家熟悉的高层数学描述，而手写 CSL 代码则混杂了大量难以理解的底层架构细节。

## 六、相关研究与定位

### 6.1 早期 WSE 编程探索

- Rocki et al. (2020)：在 CS-1 上移植 Krylov 子空间法组件，展示了 WSE 的性能潜力，但使用私有接口。
- Woo et al. (2022)：开发了基于 Python 的 API，提高了易用性，但要求用户改用特定 API，且生成不公开的汇编代码。

### 6.2 同期 WSE 编译技术

- :mdnice-bold-italic[Sai et al. (2024)：提出了基于 Python 的模板 DSL，同样生成 CSL。其重点在于软件路由算法] 的生成，与本文工作互补。但使用了完全定制的编译器栈。
- **SPADA (Gianinazzi et al., 2025)**：一种面向 WSE 的空间数据流语言，并集成了 GT4Py DSL。它更侧重于** 底层栈的深度优化**。:mdnice-bold-italic[本文工作与 SPADA 是互补关系]：:mdnice-bold-italic[未来两者结合（例如 SPADA 实现为 MLIR 方言）将非常有益。]

### 6.3 其他 MLIR 生态工作

MLIR 社区已为 AMD AI Engine、Tenstorrent 等新型加速器开发了方言。

::: block-1
**本文工作的独特之处在于，它是首个支持将现有语言/DSL 中未经修改的代码，加速运行在目标加速器上的 MLIR 流水线**，真正实现了“编写一次，到处运行”的硬件无关编程梦想。
:::

## 七、结论与展望

> 这项工作成功地证明了：:mdnice-bold-italic[通过利用领域特定知识（如模板计算）来驱动基于 MLIR 的编译器自动化，可以从根本上解决像 Cerebras WSE 这样奇特架构的可编程性和性能挑战。]

其贡献不仅是**第一个面向 WSE 的 MLIR 编译流水线**，更提供了一套**可重用的方法论**：:mdnice-bold-italic[如何设计渐进式降低的方言，如何将同步算法自动转换为异步 actor 模型，以及如何将硬件特定优化封装在编译器内部]。未来方向主要有：

1. **探索 Actor 模型作为通用抽象**：本文发现 Actor 模型能很好地描述 WSE 的执行。这或许能为更广泛的类 CGRA 加速器提供一个统一的编程模型抽象。
2. **扩展通信模式**：当前运行时库主要优化星型模板。支持更复杂的通信模式（如盒型）是下一步。
3. **与更多前端和底层优化集成**：如与 SPADA 这样的底层优化框架结合，形成更强大的完整工具链。

这项研究标志着高性能计算编译技术向前迈出了一大步。它不仅在性能和生产力上取得了突破，更重要的是，它为未来日益异构、 specialized 的硬件世界，指明了一条通往“**程序员友好型极致性能** ”的可行路径。科学计算的开发者们，或许终将能从繁琐的底层架构适配中解放出来，重新专注于他们真正擅长的领域：探索科学本身。

::: related-title
相关推荐
:::

::: related-links
- [打破库依赖与 93% 峰值效率！Intel 提出 MLIR 驱动的编译器自动生成 NanoKernel 实现高性能矩阵乘法内核！性能媲美手工优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895855&idx=1&sn=23bb841f0a281cb91e82b9e37fbddd8c&scene=21#wechat_redirect)
- [NiceToMeetYou：MLIR 抽象变换器助力 LLVM 静态分析，17% 精度超越手工版 + 编译性能稳定](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894462&idx=2&sn=85080a15f7ef9de3d3d065857a0c2a54&scene=21#wechat_redirect)
- [MLIR-AIR：AMD 基于空间分区与显式同步原语的开源编译器栈，融合通信-计算重叠，实现矩阵乘法 78.7%计算效率](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893191&idx=1&sn=fba9af0b8819b0766c12b58bb2adfef9&scene=21#wechat_redirect)

:::
