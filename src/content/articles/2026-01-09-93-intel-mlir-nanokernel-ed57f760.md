关键词：:mdnice-bold-italic[MLIR]、**nanokernels**、:mdnice-bold-italic[microkernels]、matmul、vectorization、**compiler**

::: block-1
注：**超微内核（Nanokernel）** 指寄存器级别的最小计算单元，专为特定硬件指令集优化，可作为可组合的、目标无关的编译器 IR 到目标特定指令的 kernel。
:::

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/01.webp)

- **Library Liberation: Competitive Performance Matmul Through Compiler-composed Nanokernels**
- https://arxiv.org/pdf/2511.13764
- 4000 字，阅读 13 分钟，**播客 27 分钟**

相关推荐

- [GEMM 95% FPU 利用率！基于 MLIR 面向 RISC-V 指令集扩展的加速Micro-kernels的多层编译器后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889407&idx=1&sn=f5640780406954ae498d2c98d162d454&scene=21#wechat_redirect)
- [浅谈C/C++性能优化背后方法论：Top-down Micro-architecture Analysis](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886603&idx=1&sn=5a003bbe593e571325eded84f4117632&scene=21#wechat_redirect)
- [MLIR-AIR：AMD 基于空间分区与显式同步原语的开源编译器栈，融合通信-计算重叠，实现矩阵乘法 78.7%计算效率](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893191&idx=1&sn=fba9af0b8819b0766c12b58bb2adfef9&scene=21#wechat_redirect)

> 在 AI 和机器学习 workload 飞速发展的今天，矩阵乘法（GEMM）作为核心计算原语，其性能直接决定了模型训练和推理的效率。但长期以来，想要让矩阵乘法逼近硬件峰值性能，要么需要专家手工编写目标硬件专属内核，要么依赖 CUTLASS、libxsmm 等厂商优化库——前者门槛极高，后者则存在兼容性滞后、依赖复杂、难以适配新兴硬件的问题。

最近 Intel 团队在 arXiv 上发表的论文《Library Liberation: Competitive Performance Matmul Through Compiler-composed Nanokernels》，给出了一套颠覆性解决方案：**让编译器直接自动生成高性能超微内核（Nanokernels）**，彻底摆脱对外部库和运行时 JIT 编译的依赖，同时性能媲美甚至超越主流优化库。这篇文章我们就来深入拆解这项工作的核心创新和技术细节。

## 本文目录

- 一、核心问题：为什么传统方案难以兼顾性能与灵活性？
- 二、核心创新点：三大突破打破传统局限
  - 创新 1：编译器原生超微内核生成，彻底去库依赖
  - 创新 2：寄存器级智能分块+向量化，实现近最优寄存器利用率
  - 创新 3：多架构+多精度统一适配，靠 MLIR dialect 打通底层
- 三、工作方法详解：从矩阵乘法到高性能超微内核的完整流程
  - 第一步：寄存器分块（Register Tiling）—— 让计算“住进”寄存器
  - 第二步：向量化（Vectorization）—— 激活 SIMD 并行
  - 第三步：超微内核生成—— 适配硬件的指令级优化
- 四、相关工作对比：本文到底强在哪？
- 五、实验结果：性能媲美甚至超越手工优化库
  - 1. 性能表现
  - 2. 关键洞察
- 六、总结与展望

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/02.webp)

## 一、核心问题：为什么传统方案难以兼顾性能与灵活性？

> 在深入创新点前，我们先搞懂传统矩阵乘法优化的痛点：

1. **性能与硬件强绑定**：不同 CPU 架构（如 AMX、AVX512、AVX2）的向量寄存器宽度、指令集特性不同，手工编写的内核无法跨架构复用，优化库也需针对性适配。
2. **库依赖的“枷锁”**：主流方案依赖 libxsmm 等外部库，这些库不仅会引入 runtime 开销，还难以快速支持新硬件特性（如 VNNI 指令）和新型算子模式。注，VNNI 是向量神经网络指令，x86 架构中针对 AI workload 优化的指令集，支持打包数据的融合乘加运算；
3. **内存墙瓶颈**：矩阵乘法的计算强度高，但连续操作间的数据搬运容易成为瓶颈，传统方案的内存布局优化难以与编译流程深度融合。

而这篇论文的核心目标，就是:mdnice-bold-italic[通过编译器原生能力，解决“性能、灵活性、可扩展性”三者不可兼得的难题。]

## 二、核心创新点：三大突破打破传统局限

> 这篇工作的创新并非单点优化，而是**一套端到端的编译方案重构**，核心包含三大突破：

### 创新 1：编译器原生超微内核生成，彻底去库依赖

传统方案中，微内核（Microkernel）的优化逻辑被封装在外部库中，编译器仅负责调用。而本文提出：**将库的专家优化知识直接嵌入编译器，通过 MLIR 的多级下转换（Multi-level Lowering）自动生成超微内核**。

::: block-1
- **超微内核（Nanokernel）**：:mdnice-bold-italic[是比微内核更精细的计算单元，专为寄存器级计算设计]，可作为可组合的、目标无关的编译器 IR 到目标特定指令的降低核心，进而形成适配不同硬件的微内核。
- **关键优势**：:mdnice-bold-italic[无需外部库调用、无 JIT runtime 开销]，实现了从高级算子到机器码的纯编译器 AOT（提前编译）统一路径，且能天然适配新硬件（只需扩展 MLIR dialect 支持）。
:::

### 创新 2：寄存器级智能分块+向量化，实现近最优寄存器利用率

寄存器溢出是内核性能的“隐形杀手”，本文通过精细化的寄存器分块策略，确保计算全程无溢出：

- **分块维度**：覆盖 M（矩阵 A 的行）、N（矩阵 B 的列）、K（公共维度）和 batch 维度，batch 固定为 1，K 根据指令集适配（1/VNNI/32）。
- **动态适配**：根据硬件寄存器资源精确计算分块大小，以实现“用满但不超”的资源利用。例如 AVX512（32 个向量寄存器）采用 M=4、N=64，恰好使用 29 个寄存器（24 累加器+4 广播+1 加载）；AVX2（16 个向量寄存器）采用 M=3、N=32，用满 16 个寄存器。
- **兜底策略**：当分块可能导致溢出时，自动交换 A/B 的广播/加载角色，确保寄存器充分利用（如图 2(d)）。

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/03.webp)

::: image-caption
图 2：文献《Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning Workloads》中提出的 BRGEMM 寄存器分块方案，适用于 (a) 32 个向量寄存器、(b) 16 个向量寄存器、(c) 8 个二维寄存器瓦片，以及 (d) 16 个向量寄存器（A 和 B 的加载顺序互换）的场景。该图展示了针对不同 CPU 架构向量资源的寄存器分块策略，核心目标是避免寄存器溢出并最大化寄存器利用率。例如，AVX512 架构（32 个向量寄存器，图 a）采用 M:4、N:64 分块，仅使用 29 个寄存器（24 个累加器、4 个 A 广播寄存器、1 个 B 加载寄存器）；AVX2 架构（16 个向量寄存器，图 b）用 M:3、N:32 分块，耗尽所有寄存器；Intel AMX 架构（图 c）则用 8 个 16×16 FP32 二维瓦片存储数据。当分块导致寄存器不足时（如图 d 的 16 寄存器场景），通过互换 A 广播与 B 加载角色，仍能实现无溢出计算，为跨架构超微核生成提供了分块设计依据
:::

### 创新 3：多架构+多精度统一适配，靠 MLIR dialect 打通底层

本文的核心技术底座是 MLIR 的 dialect 系统，通过抽象不同硬件的指令特性，实现“一套编译逻辑，多架构适配”：

- **核心 dialect 组合**：amx dialect（适配 AMX 架构 tile 操作）、x86vector dialect（适配 AVX 系列指令）、vector dialect（通用向量操作）。
- **多精度支持**：天然支持 FP32、BF16（重点优化）、INT8（未来扩展），其中 BF16 通过 VNNI 打包布局提升 SIMD 效率。

## 三、工作方法详解：从矩阵乘法到高性能超微内核的完整流程

这套方案的核心流程可概括为“三步曲”：寄存器分块 → 向量化 → 超微内核生成，全程基于 MLIR 上游变换，无自定义黑盒逻辑。

### 第一步：寄存器分块（Register Tiling）—— 让计算“住进”寄存器

> 目标是将 Batch-Reduce GEMM（BRGEMM，支持批量矩阵乘法积累，适合 Transformer 等模型中的注意力机制计算）分解为寄存器可容纳的子问题，避免数据频繁进出内存。

BRGEMM 是本文的核心计算模型，相比传统 GEMM 有两大优势：

::: block-1
1. 积累 tile 全程驻留寄存器，减少内存访问；
2. 支持元素级操作融合（如 bias+ReLU），提升数据局部性。
:::

分块策略需严格匹配硬件特性，论文给出了不同架构的最优分块配置：

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/03.webp)

::: image-caption
图 2：文献《Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning Workloads》中提出的 BRGEMM 寄存器分块方案，适用于 (a) 32 个向量寄存器、(b) 16 个向量寄存器、(c) 8 个二维寄存器瓦片，以及 (d) 16 个向量寄存器（A 和 B 的加载顺序互换）的场景。该图展示了针对不同 CPU 架构向量资源的寄存器分块策略，核心目标是避免寄存器溢出并最大化寄存器利用率。例如，AVX512 架构（32 个向量寄存器，图 a）采用 M:4、N:64 分块，仅使用 29 个寄存器（24 个累加器、4 个 A 广播寄存器、1 个 B 加载寄存器）；AVX2 架构（16 个向量寄存器，图 b）用 M:3、N:32 分块，耗尽所有寄存器；Intel AMX 架构（图 c）则用 8 个 16×16 FP32 二维瓦片存储数据。当分块导致寄存器不足时（如图 d 的 16 寄存器场景），通过互换 A 广播与 B 加载角色，仍能实现无溢出计算，为跨架构超微核生成提供了分块设计依据
:::

如上图所示，分块的核心是“无溢出+高利用率”，例如(a)中通过选择 M=4、N=64，精确使用了 29 个寄存器（24 个累加器+4 个广播+1 个加载），在 32 个寄存器限制内实现了无溢出计算。

### 第二步：向量化（Vectorization）—— 激活 SIMD 并行

将分块后的 BRGEMM 转换为`vector.contract`操作，这是超微内核生成的入口：

- **数据移动**：通过`vector.transfer_read/write`实现向量加载/存储；
- **计算表达**：通过`vector.contract`定义矩阵乘法的收缩运算，明确并行维度和归约维度。

### 第三步：超微内核生成—— 适配硬件的指令级优化

这是最核心的环节，根据目标架构、数据类型（FP32/BF16）和数据布局（VNNI-packed/Flat），生成针对性超微内核。

#### 关键技术 1：数据布局优化（VNNI 打包）

对于 BF16 类型，x86 架构的 VNNI 指令要求数据按特定格式打包（2 个 BF16 元素一组），否则会导致内存访问不连续。论文通过 MLIR 变换自动完成 Flat 布局到 VNNI-packed 布局的转换（下图）。在生成的代码中，对于 AVX 目标，该转换通过映射到底层的`vpunpcklwd/hwd`指令实现；而在 AMX 等目标的 IR 层面，则通过`vector.shuffle`操作来表达。

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/04.webp)

::: image-caption
图 1：该图是 x86 架构下 BF16 数据适配 VNNI 指令的核心格式转换示意图。将 4×4 行优先顺序矩阵打包为 VNNI 格式(a) 平坦的 4×4 矩阵；(b) VNNI:2 参数下打包后的 2×4×2 矩阵。行优先存储的 4×4 平坦矩阵（图 a），因直接用于 VNNI 指令会导致非连续内存访问，需按 VNNI:2 规则打包为 2×4×2 结构（图 b）——BF16 元素成对组合，确保与 AVX512 向量寄存器宽度（512 位）对齐。这种打包避免了 runtime 数据重排开销，使 TDPBF16PS/VDPBF16PS 等 VNNI 指令能高效加载数据，显著提升缓存局部性和 SIMD 计算效率，是 BF16 矩阵乘法在 Intel CPU 上接近峰值性能的关键预处理步骤
:::

#### 关键技术 2：架构感知的 dialect 映射

不同架构的超微内核通过 MLIR dialect 操作组合生成，论文给出了清晰的映射关系：

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/05.webp)

::: image-caption
表 1 MLIR dialect 与目标架构的映射关系：注释：通过 dialect 抽象，FP32 内核可跨架构复用，BF16 则根据硬件指令集选择最优操作，例如 AMX 用 tile 级乘法，AVX512 用专用 dot 指令
:::

#### 关键技术 3：超微内核生成算法

论文给出了统一的超微内核生成框架（下面是原论文 Algorithm 1的大致流程），核心流程为：

1. 加载 C 矩阵的子块作为累加器（驻留寄存器）；
2. 对 batch 和 K 维度循环，加载 A/B 的子块；
3. 根据数据类型和架构，执行乘法-累加运算（FMA 或 dot 产品）；
4. 若为 Flat 布局，完成结果重排后存储回 C 矩阵。

以 BF16-AMX 架构为例，超微内核的生成逻辑为：

- 用`amx.tile_load`加载 A/B 的 16×32 子块；
- 用`amx.tile_mulf`执行 tile 级 dot 产品，结果积累在 FP32 寄存器；
- 用`amx.tile_store`写回结果，如需 BF16 输出则完成类型转换。

## 四、相关工作对比：本文到底强在哪？

> 为了更清晰地凸显创新，我们对比主流方案的核心差异：

| 方案 | 核心依赖 | 跨架构适配 | 性能 | 灵活性（支持新硬件） |
| --- | --- | --- | --- | --- |
| 手工编写内核（如 DeepSeek） | 硬件专家经验 | 差（专属架构） | 高 | 差（需重新编写） |
| 厂商优化库（如 libxsmm/CUTLASS） | 外部库+JIT | 中（需库适配） | 高 | 中（库更新滞后） |
| TPP-MLIR（原版） | 外部库（libxsmm） | 中（MLIR+库） | 中（90%峰值） | 中（依赖库扩展） |
| **本文方案（Compiler-composed Nanokernels）** | **MLIR 编译器** | **优（dialect 抽象）** | **高（88%-132% libxsmm 性能）** | **优（扩展 dialect 即可）** |

上表是主流矩阵乘法优化方案对比。本文的核心优势是“去库依赖”和“编译器原生适配”，既保留了手工优化库的性能，又具备跨架构和快速支持新硬件的灵活性。

## 五、实验结果：性能媲美甚至超越手工优化库

> 论文在 Intel 最新三代处理器（Emerald Rapids、Sierra Forest、Arrow Lake）上做了全面验证，核心结论如下：

### 1. 性能表现

以 libxsmm 为基准（100%），不同平台和配置下的性能对比如下：

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/06.webp)

::: image-caption
表 2：在 EMR 和 SRF 两台机器上，不同线程数与矩阵大小下，编译器生成超微核与 libxsmm 的几何平均性能百分比对比。实验平台中，EMR 为双路 Xeon Platinum 8592+（支持 AMX/AVX512，2×64 核），SRF 为双路 Xeon 6780E（能效核架构，2×144 E 核，支持 AVX2）。
:::

上面表格数据显示，FP32 场景下超微核性能接近 libxsmm（EMR 达 91%-94%、SRF 达 91%-93%）；BF16 场景中，VNNI 布局性能达 73%-93%，而 Flat 布局在 SRF 上反超 libxsmm（112%-113%）。**这表明编译器生成的超微核在多线程扩展（1-64 线程）和不同矩阵规模（1024-4096）下均稳定**，尤其在 SRF 的 E 核架构上，BF16-Flat 布局因避免格式转换开销展现出显著优势。

![文章插图](/articles/2026-01-09-93-intel-mlir-nanokernel-ed57f760/images/07.webp)

::: image-caption
表 3：在 ARL 机器的性能核（P 核）和能效核（E 核）上，不同线程数与矩阵大小下，编译器生成超微核与 libxsmm 的几何平均性能百分比对比。ARL 平台为单路 Intel Core Ultra 9 285K（8 个性能核 P+16 个能效核 E，支持 AVX2）
:::

上面表格数据凸显了**超微核对异构核心的适配能力**。FP32 场景下，P 核性能达 86%-91%、E 核达 85%-93%，接近 libxsmm；BF16 场景优势显著，VNNI 布局达 89%-98%，Flat 布局更超 libxsmm（P 核 131%-132%、E 核 119%-126%）。这是因为 Flat 布局直接利用 AVX2 打包操作，规避了 libxsmm 将 Flat 转换为 VNNI 的额外开销，:mdnice-bold-italic[在 ARL 的异构核心上更能发挥硬件能效，验证了编译方案的异构核心优化效果。]

### 2. 关键洞察

- **内存带宽是 AMX 架构的瓶颈**：虽然 AMX 的计算吞吐量更高，但 misaligned 加载/存储会导致吞吐量降至峰值的 1/4，需优化数据布局。
- **AVX2 架构的 Flat 布局优势明显**：由于本文直接利用 AVX2 的 packed 操作，无需布局转换，性能反超依赖 VNNI 的方案。
- **多线程扩展性良好**：在部分高线程配置下（如 64 线程），AMX Flat 布局的性能甚至能反超 libxsmm 约 5%。整体上，AMX Flat 布局几何平均性能达到 libxsmm 的 93%（VNNI 布局为 91%）。

## 六、总结与展望

> 这篇工作的核心价值，是为矩阵乘法等核心计算原语提供了一套“**编译器原生、去库依赖、跨架构** ”的优化方案——通过 MLIR 的 dialect 抽象和寄存器级精细优化，让编译器具备了专家级的内核生成能力。

未来工作方向也很明确：

1. **上游化** 到 LLVM/MLIR 主干，让社区直接复用；
2. **扩展支持** INT8 等更多数据类型，适配量化模型；
3. **自动化** 推导最优分块大小，无需人工调参。

对于 AI 框架开发者和硬件厂商来说，这套方案意味着：:mdnice-bold-italic[无需再为不同架构维护大量手工优化内核，只需扩展 MLIR dialect 即可快速支持新硬件]，同时保持高性能。而对于终端用户，这意味着模型编译将更高效、更灵活，无需依赖复杂的第三方库。

::: block-1
**编译器正在成为硬件性能释放的核心枢纽**，这篇工作无疑为“**编译驱动的高性能计算** ”提供了一个绝佳范例。
:::
