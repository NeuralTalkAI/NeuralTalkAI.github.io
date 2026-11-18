关键词：**TorchInductor**、**CuteDSL**、GEMM、GPU 推理、:mdnice-bold-italic[自动调优]

> “在抽象-性能权衡的赛道上，每一种优秀的领域特定语言（DSL）都占据着独特位置。” PyTorch 的 TorchInductor 此前已支持 Triton、CUTLASS（C++）和 cuBLAS 三大自动调优后端，而 CuteDSL 的加入，不仅填补了 Python 编写、低复杂度与高性能兼具的技术空白，更满足了 TorchInductor 对新后端“低维护成本、无性能退化、目标负载更优”的严苛要求。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/01.webp)

- **Generating State-of-the-Art GEMMs with TorchInductor’s CuteDSL backend**
- https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/
- 代码：http://github.com/pytorch/pytorch/tree/main/torch/_inductor/codegen/nv_universal_gemm
- 6000 字，阅读 35 分钟，**播客 21 分钟**

相关推荐

- [性能超传统库达 1.79×！AMD 提出 Triton 原生多 GPU 通信库 Iris：以 tile 级抽象实现 GEMM+All-Scatter 近峰值带宽](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895934&idx=1&sn=d02bb97055c0f7586e8a3d71e0b3b4cb&scene=21#wechat_redirect)
- [用 Python 写一个开源 RISC-V 向量编译器：MLIR 与 xDSL 的 GEMM 代码生成，较 OpenBLAS 最高提升 35%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899103&idx=1&sn=1729342f5820122c5fad0a641be01943&scene=21#wechat_redirect)
- [对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)

> 作为 NVIDIA 积极研发的 DSL，它继承了 CUTLASS C++的底层抽象能力，却拥有更快的编译速度和更简洁的维护特性，尤其在矩阵乘法（GEMM）这一 LLM 核心计算场景中表现突出。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/02.webp)

::: image-caption
硬件迭代与编译器矩阵乘法（GEMM）内核开发的脱节问题：硬件演进速度远超 GEMM 编译器内核的更新节奏，而饼图显示在 Llama 3.3 70B 模型中，GEMMs 占运行时开销的 89%，是 Transformer 模型的性能瓶颈。用户本期望 torch.compile 能提供无需自定义内核、手动调优的高性能 GEMM，但 Blackwell 等新架构每代新增 Tensor Core 模式、低精度格式等特性，打破了这一预期，凸显了快速适配新硬件 GEMM 方案的迫切需求。
:::

通过整合 cutlass_api 与 nvMatmulHeuristics 分析模型，CuteDSL 后端能精准筛选最优内核配置，**在 B200 GPU 上实现最高 1.78 倍的内核吞吐量提升，端到端推理 latency 最大降低 6.5%。**

::: block-1
本文将详细拆解其技术架构、集成策略、性能表现及使用方法，展现这一后端如何为 GPU 计算注入新活力。
:::

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/03.webp)

::: image-caption
CuTeDSL作为实现SOTA矩阵乘法（GEMMs）的可扩展方案的三大核心优势：一是完整暴露线程与内存层次，可充分利用硬件架构特性，释放高性能上限；二是编译效率优异，耗时与Triton相当，支持epilogue融合、跨内核配置的基准测试与自动调优；三是依托NVIDIA的主动投入，借助早期硬件访问权限，能快速适配新硬件特性，为高性能计算提供高效支撑。
:::

## 本文目录

- 一、引言
  - 1.1 概述
- 二、策略：为何聚焦 GEMM 计算
  - 2.1 不同运算的后端适配差异
- 三、背景：TorchInductor 如何生成 GEMM 内核
  - 3.1 自动调优流程解析
- 四、CuteDSL 后端的架构设计
  - 4.1 核心工作流程
- 五、性能测试结果
  - 5.1 测试环境与评估指标
  - 5.2 内核级性能提升
  - 5.3 端到端 vLLM 推理性能
- 六、CuteDSL 后端支持的功能
  - 6.1 功能清单
- 七、如何试用 CuteDSL 后端
  - 7.1 安装步骤
  - 7.2 使用方法
- 八、未来工作：开发路线图
- 结论：总结与展望

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

### 1.1 概述

> TorchInductor 目前支持三种矩阵乘法（GEMM）自动调优后端：Triton、CUTLASS（C++）和 cuBLAS。**本文将介绍第四种后端 CuteDSL 的集成方案、技术研发动机以及目前已观测到的性能结果。**

内核编写领域的领域特定语言（DSL）发展势头迅猛，Triton、Helion、Gluon、CuTile 和 CuteDSL 各自在抽象程度与性能表现的权衡中占据着独特位置。**当评估是否将新后端集成到 TorchInductor 时，我们遵循三大标准：**

1. 集成不会给团队带来沉重的维护负担，或有厂商提供长期持续的支持；
2. 相较于现有后端，不会增加编译时间或基准测试时间；
3. 在目标工作负载上能提供更优性能。

::: block-1
**CuteDSL 完全满足这三大标准**。NVIDIA 正在积极开发 CuteDSL，并提供优化后的内核模板，这极大降低了 TorchInductor 团队的维护压力。其编译时间与我们其他后端持平，相较于需要完整 nvcc 调用的 CUTLASS C++方案，实现了显著提升。
:::

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/05.webp)

除了这些即时收益，CuteDSL 更是一项长期战略投资。它基于与 CUTLASS C++相同的抽象架构——后者在 FP8 精度 GEMM 计算和尾处理融合方面已展现出卓越性能[1]——但 CuteDSL 采用 Python 编写，编译速度更快，维护复杂度更低。**随着 NVIDIA 持续投入 CuteDSL 的性能优化，它有望在新一代硬件上逐步替代 CUTLASS C++集成方案，简化 TorchInductor 的代码库**。

::: block-1
厂商与社区的激励对齐、日益增长的开源采用率（如 Tri Dao 的 Quack 库、Colfax International 的 Jay Shah 均已采用），以及能充分暴露线程和内存层级的底层编程模型，使得 CuteDSL 成为当前及未来 NVIDIA 硬件上实现最优 GEMM 性能的理想后端。
:::

## 二、策略：为何聚焦 GEMM 计算

### 2.1 不同运算的后端适配差异

> 并非所有运算都能从新后端中获得同等收益。对于内存受限型运算——如逐元素数学运算、激活函数和归约运算——Triton 已能生成高质量代码。其块级编程模型非常适合这类仅需向量化内存访问的工作负载，且 Triton 生成的内核与手工编写内核的性能差距极小。

CuteDSL 虽然也能表达逐点运算和归约运算，但由于其底层特性，从零自动生成 CuteDSL 内核的复杂度较高。**实际测试中，两种 DSL 在这类工作负载上的性能表现不相上下，因此额外的实现复杂度无法带来实质性收益**。:mdnice-bold-italic[我们通过实验验证了这一结论：在 GB200 GPU 上，对 Triton 和 CuteDSL 实现的 softmax 内核进行不同输入尺寸的测试，两者均能接近硬件的极限带宽。]

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/06.webp)

::: image-caption
表格对比了 GB200 平台上 FP16 精度下，Triton 与 CuTeDSL 实现的 Softmax 算子性能。在从 3096² 到 16384² 的多规模测试中，两者耗时几乎持平，Speedup（Triton/CuTe DSL）稳定在 0.98x-1.00x 之间，带宽随规模提升最高达约 7.2K GB/s。结果显示 CuTeDSL 的 Softmax 内核性能与主流 Triton 实现相当，验证了它不仅在 GEMM 上，在通用算子上也能达到业界标杆级性能。
:::

> GEMM 计算则是另一番景象。矩阵乘法在基于 Transformer 的模型中占据着主导地位：在典型的大语言模型（LLM）前向传播过程中，注意力投影、前馈网络（FFN）层和输出头中的 GEMM 运算消耗了大部分 GPU 计算周期。

要在这些运算上实现接近峰值的硬件利用率，需要对每一代新 GPU 引入的硬件特性进行精准控制——包括适配张量核心流水线的分块大小、共享内存级缓存的显式管理、 warp 级调度，以及在 B200 等新型架构上的线程块集群和分布式共享内存技术。这些正是高级语言为简化使用而抽象隐藏的细节。**为了降低底层代码生成的复杂度，我们没有选择从零构建内核，而是基于手工优化的模板进行开发，这些模板暴露了针对不同问题规模调整性能所需的可配置参数。**

现有的 CUTLASS C++后端通过提供底层控制能力解决了这一问题，但 C++编译的高昂开销带来了实际限制：每个内核变体都需要完整的 nvcc 调用，这使得在自动调优过程中评估大量候选方案变得成本极高，也无法在调度阶段对尾处理融合决策进行基准测试。

> CuteDSL 通过定制化的 Python 到 MLIR 编译器解决了这一问题。

该 DSL 与 CUTLASS C++基于相同的抽象架构——:mdnice-bold-italic[相同的分块代数、相同的内存层级原语、相同的尾处理融合模型——但编译速度可与 TorchInductor 的其他后端相媲美]。这种特性组合使得 TorchInductor 能够将其用于其他后端的完整自动调优和融合基准测试流程，应用到具备 CUTLASS 级硬件控制能力的 GEMM 内核中。实现这一目标的核心特性包括：

- **完整的线程和内存层级暴露**：CuteDSL 提供了同步机制、warp 级控制、线程块集群以及完整的线程/内存层级原语。这使得能够利用特定架构的特性，例如 H100 和 B200 上的分布式共享内存。
- **编译时间优化**：CUTLASS C++方案需要为每个内核变体执行完整的 nvcc 调用。这种开销使得基准测试融合（编译器在调度阶段评估多种带有不同尾处理融合的 GEMM 候选方案）变得不切实际。:mdnice-bold-italic[CuteDSL 的编译速度与其他后端相当，消除了这一限制，为新的自动调优策略创造了可能。]
- **NVIDIA 优化的 GEMM 模板**：NVIDIA 专门团队正在积极开发 CuteDSL，提供优化后的 GEMM 内核模板和尾处理融合支持，并致力于实现与 CUTLASS C++后端相当的性能。对于下一代硬件，CuteDSL 将凭借更早接触最新硬件的优势，在硬件特定优化方面占据先发地位。

::: block-1
简而言之：Triton 在逐点运算上表现出色，因此 **CuteDSL 后端的研发重点放在了性能提升空间最大的领域——最新硬件上的 GEMM 计算、注意力机制和尾处理融合。**
:::

## 三、背景：TorchInductor 如何生成 GEMM 内核

### 3.1 自动调优流程解析

> 随着深度学习和 AI 应用场景的发展，GPU 架构变得极其复杂。因此，设计 GEMM 内核时需要做出诸多决策，例如分块大小、warp 专业化配置、指令形状，以及是否使用异步内存传输（如 Hopper 和 Blackwell 架构上的 TMA 技术）。

Torch.compile 作为即时编译（JIT）编译器，能够在运行时识别模型的问题规模，并利用这些信息选择性能最优的配置，这种针对特定工作负载自动优化内核的技术称为自动调优。下图展示了 TorchInductor 的 Triton 自动调优系统流程。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/07.webp)

::: image-caption
TorchInductor为`torch.mm`（矩阵乘法）选择最优GEMM实现的调度流程：它根据输入的问题形状、数据类型/布局等参数，同时评估三类方案——直接选择cuBLAS/HipBLAS，或对Triton、CUTLASS/CK的多种配置执行并行JIT编译，再通过统一的性能基准测试，最终选出适配当前场景的最优GEMM内核，以此在不同硬件与参数下，保障矩阵乘法的性能最优。
:::

> TorchInductor 的 GEMM 自动调优流程分为多个阶段。

当编译器在 lowering 过程中遇到矩阵乘法时，首先会查询每个已启用的后端（Triton、CUTLASS、cuBLAS），判断该后端是否支持当前的问题规模、数据布局和数据类型。不支持该配置的后端会在此阶段被过滤掉。

对于每个符合条件的后端，TorchInductor 会从该后端的模板库中生成一组候选内核。这些候选内核在分块大小、warp 配置和其他后端特定参数上存在差异。随后，所有候选内核都会在目标硬件上进行基准测试，并选择性能最快的内核。

选中的内核及其编译输出会被写入 TorchInductor 的缓存中，因此后续针对相同问题配置的编译可以完全跳过基准测试步骤。这种缓存机制同时作用于单个内核级别（编译后的代码）和选择级别（针对特定问题规模和后端组合的最优候选内核）。

> 在基础流程之上，TorchInductor 还支持 GEMM 内核的尾处理融合。

在调度阶段，编译器会评估将下游逐点运算融合到 GEMM 尾处理中的收益。对于 Triton，这一功能通过 MultiTemplate 缓冲区实现：lowering 阶段筛选出的前 N 个 GEMM 候选内核会被保留，调度阶段会对可能的融合方案进行基准测试，判断融合变体是否优于未融合的 GEMM 内核加独立逐点内核的组合。内核的最终选择会推迟到融合过程完成后进行。完整流程如下图所示。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/08.webp)

::: image-caption
TorchInductor 为torch.mm算子调度最优 GEMM 内核的完整流程：它根据输入的问题形状、数据类型 / 布局等参数，先检查代码缓存，命中则直接复用；未命中时，并行编译 Triton、CUTLASS/CK 的多种配置，同时评估 cuBLAS/HipBLAS 方案，再通过基准测试选出最优内核。此外还会判断 epilogue 融合是否有收益，必要时重编译融合内核，在保证性能的同时兼顾编译效率。
:::

CUTLASS C++后端通过尾处理访问树（EVT）支持尾处理融合，但每个变体的 nvcc 编译成本限制了实际可评估的配置数量。这种编译时间约束是引入 CuteDSL 作为替代方案的主要动机之一。

注：目前 CuteDSL 后端尚未支持尾处理融合，但该功能已列入开发计划（见“未来工作”部分）。

## 四、CuteDSL 后端的架构设计

### 4.1 核心工作流程

> CuteDSL 后端接入了上述自动调优流程。

当 Inductor 在 lowering 过程中遇到矩阵乘法时，该后端会按以下三个步骤执行：

1. 查询 cutlass_api，获取所有与当前问题兼容的内核配置；
2. 利用 nvMatmulHeuristics 对这些配置进行排序，筛选出最优候选方案；
3. 将这些候选方案在目标硬件上进行编译和基准测试，并与 ATen 和 Triton 的结果进行对比。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/09.webp)

::: image-caption
TorchInductor 为torch.mm算子调度最优 GEMM 内核的完整流程：输入矩阵形状、数据类型 / 布局等参数后，框架同时评估 ATen、Triton、Cutlass、CuTeDSL 四类实现；其中 Triton、Cutlass、CuTeDSL 支持 JIT 并行编译（CuTeDSL 并行编译功能即将上线），所有候选方案经统一基准测试对比，最终选出适配当前场景的最优 GEMM 内核，为不同硬件与计算场景提供高性能矩阵乘法支持。
:::

> 它与 Triton 和 CUTLASS C++方案的核心差异体现在两个方面：

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/10.webp)

::: image-caption
CuTeDSL 的核心支撑组件：Cutlass API 与 nvMatmulHeuristics。Cutlass API 作为 NVIDIA 维护的内核库，可根据矩阵形状、数据类型等查询兼容内核，新硬件特性无需修改 Inductor 即可适配；nvMatmulHeuristics 通过分析模型预估硬件吞吐量，从数百候选中筛选 Top5 进行设备侧 profiling，与 ATen、Triton 对比选出最优配置。CuTeDSL 依托并行 JIT 编译，结合二者能力，实现了高效可扩展的 SOTA GEMM 内核调度。
:::

#### **差异1：通过 cutlass_api 选择内核**

Triton 后端从 TorchInductor 内部维护的模板中生成候选内核。而 CuteDSL 后端采用了不同的方式：它查询由 NVIDIA 维护的 Python 库 cutlass_api，该库包含了完整的 CuTeDSL GEMM 内核配置集合——包括分块形状、集群大小和调度参数。

Inductor 会向该 API 描述当前问题（规模、数据类型、布局、缩放模式和 GPU 计算能力），API 则返回所有兼容的内核。当 NVIDIA 添加新的内核配置或硬件支持时，这些更新会直接纳入 cutlass_api，无需修改 Inductor 的代码。

该 API 还具备可扩展性：TorchInductor 可以将自定义的内核类注册到该库中。我们曾利用这一特性，在 upstream 支持之前就添加了 FP4 精度 GEMM 支持（NVFP4、MXF4）——我们的定制内核与 NVIDIA 官方内核遵循相同的筛选、排序和基准测试流程。

#### **差异2：基于启发式模型的搜索空间缩减**

针对特定问题查询 cutlass_api 可能返回数百个兼容的内核配置。对所有配置进行基准测试的成本过高，因此 CuteDSL 后端集成了 nvMatmulHeuristics——这是一个 NVIDIA 开发的分析性能模型，能够通过评估分块效率、内存带宽和占用率等指标，为每个配置打分并预测其硬件吞吐量。

这一过程将数百个候选配置缩减至少数几个（默认 5 个，可通过 nvgemm_max_profiling_configs 参数配置）。只有这些排名靠前的配置会在目标硬件上进行编译和基准测试。Triton 和 CUTLASS 均未采用这种分析模型，它们依赖于在较小的、由模板定义的搜索空间内进行基准测试。

当自动调优选择出最优内核后，编译产物会被缓存在内存中——后续调用会直接执行编译后的函数，无需重复编译。

::: block-1
**重要的是，CuteDSL 后端是纯增量式集成**。

- 如果某个问题与 NVGEMM 不兼容——例如不支持的数据类型、布局或硬件——则不会生成任何 NVGEMM 候选方案，自动调优会照常使用 ATen 和 Triton。
- 如果生成了 NVGEMM 候选方案但在基准测试中表现不佳，系统会自动选择性能更优的后端。:mdnice-bold-italic[启用 NVGEMM 不会导致性能退化。]
:::

## 五、性能测试结果

### 5.1 测试环境与评估指标

> 所有基准测试均在单块 NVIDIA B200 GPU 上运行，功耗设置为 850W，启用动态时钟（无张量并行），使用 PyTorch nightly 版本和 CUDA 13.1。

内核级测试结果通过 Inductor 自动调优测量独立 GEMM 运算的延迟。

端到端测试结果测量 vLLM V1 的解码延迟，测试模型包括 Llama 3.1 8B、Qwen3 32B 和 Llama 3.3 70B，输入提示为 32 个 token，生成 128 个 token，采用串行执行，每次运行前清理缓存。

### 5.2 内核级性能提升

> 我们在 LLM 相关的 GEMM 规模上，针对三种数据类型方案，将 Inductor NVGEMM 与现有 Inductor 后端进行了对比。

下图展示了内核吞吐量（单位：TFLOPS），标注部分显示了相较于性能最优的现有后端的提升倍数（原文此处有三组柱状图，分别对应 BF16、MXFP8、NVFP4 三种数据类型，横轴为不同 GEMM 规模，纵轴为吞吐量，清晰展示了 NVGEMM 在中小规模场景下的性能优势）。

**BF16 精度**：NVGEMM 在解码阶段常见的规模（M=8 至 M=64）中性能提升显著，最高达 1.73 倍；在（4096, 256, 4096）这类高瘦矩阵规模下，性能提升达 1.54 倍；预填充阶段的大规模矩阵运算性能与现有后端持平。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/11.webp)

::: image-caption
柱状图对比了NVIDIA B200（Blackwell架构）平台上，BF16精度下不同矩阵乘法内核的吞吐量（TFLOPS）。横轴为不同(M,N,K)的矩阵尺寸，纵轴为运算吞吐量。结果显示，Inductor NVGEMM（CuTeDSL实现）在所有测试场景中性能全面领先，相比ATen与Triton实现，最高可带来1.68倍的速度提升，且在大矩阵尺寸下优势尤为显著，充分验证了新内核在新硬件上的性能优势。
:::

**MXFP8 精度**：NVGEMM 在中等规模矩阵运算中性能最高提升 1.78 倍，在大规模矩阵运算中与现有后端持平；宽 N 型矩形矩阵规模下，ATen 表现更优。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/12.webp)

::: image-caption
柱状图对比了NVIDIA B200平台上，MXFP8精度下Inductor ATen与Inductor NVGEMM矩阵乘法内核的吞吐量。横轴为不同(M,N,K)矩阵尺寸，纵轴为运算吞吐量（TFLOPS）。结果显示，NVGEMM在中小矩阵场景优势显著，最高可带来1.83倍性能提升；随着矩阵规模增大，两者性能趋同，但NVGEMM在多数场景仍保持领先，展现了其在低精度计算上的优化优势。
:::

**NVFP4 精度**：NVGEMM 在解码阶段规模（M≤256）的吞吐量显著提升，相较于性能最优的现有后端最高达 1.6 倍；当 M≥512 时，ATen 的优化已非常成熟，各后端性能趋于一致。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/13.webp)

::: image-caption
NVIDIA B200平台上，NVFP4精度下Inductor ATen与Inductor NVGEMM矩阵乘法内核的吞吐量表现。横轴为不同(M,N,K)矩阵尺寸，纵轴为运算吞吐量（TFLOPS）。中小矩阵场景中，NVGEMM优势显著，最高可带来1.93倍性能提升；部分大矩阵场景下ATen略高，但多数场景NVGEMM仍保持领先，验证了其在低精度高算力场景的适配优化能力。
:::

### 5.3 端到端 vLLM 推理性能

> 我们使用 vLLM 的 V1 模型运行器，在批量大小 2 至 128 的范围内测量了推理延迟。

由于 vLLM 在批量维度使用动态规模，Inductor 在编译时无法获知实际批量大小。因此我们通过 autotune_batch_hint 参数指定目标批量大小，使 Inductor 能够在运行时实际使用的规模下对候选内核进行基准测试——这一点至关重要，因为最优内核配置与矩阵规模密切相关。

**BF16 精度**：启用 NVGEMM 后，90%的配置（21 个数据点中的 19 个）实现了延迟降低。最大提升出现在 Llama 3.3 70B 模型、批量大小 16 的场景，延迟降低 6.5%；Llama 3.1 8B 在所有批量大小下均实现了 2%-4%的稳定提升；Qwen3 32B 的提升相对温和，为 0.5%-2.4%。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/14.webp)

::: image-caption
NVIDIA B200平台、BF16精度下，vLLM推理中引入NVGEMM（CuTeDSL实现）后的性能提升，基线为ATEN+TRITON方案。不同模型收益差异明显：Llama 3.3 70B提升最显著，在batch=16时最高达6.5%；Llama 3.1 8B在各batch下均有1.9%-4.3%的稳定收益；Qwen3 32B提升较弱，仅大batch场景有小幅优化。整体验证了NVGEMM在大模型推理中的加速优势，对GEMM占比高的场景收益尤为突出。
:::

**NVFP4 精度**：89%的配置（18 个数据点中的 16 个）性能提升。Llama 3.1 8B 最高提升 4.2%，Qwen3 32B 最高提升 3.5%，Llama 3.3 70B 最高提升 3.3%；在批量大小 16-64 的范围内，性能提升最为稳定。

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/15.webp)

::: image-caption
NVIDIA B200 平台、NVFP4 精度下，vLLM 推理引入 NVGEMM 后的性能提升（基线为 ATEN+TRITON 方案）。Llama 3.1 8B 在 batch=8 时提升最高达 4.2%，中高 batch 下也保持 2%-3.8% 的稳定收益；Qwen3 32B 在 16-128 batch 下有 1%-3.5% 的提升；而 Llama 3.3 70B 在小 batch 下出现性能损失，仅大 batch 有小幅优化，体现出 NVGEMM 在低精度推理中对中小模型和中高 batch 场景的适配优势。
:::

## 六、CuteDSL 后端支持的功能

### 6.1 功能清单

![文章插图](/articles/2026-04-20-gpu-gemm-pytorch-cutedsl-bf16-06c66017/images/16.webp)

::: image-caption
PyTorch中不同矩阵乘法算子的数据类型与配置支持：基础的`mm/bmm`支持fp16、bf16、fp8、int8等多种A/B与输出数据类型，布局无限制；`_scaled_mm`系列针对MXFP8、MXF4、NVF4等低精度格式，明确了缩放数据类型、块大小与布局约束，适配低精度缩放计算；`_grouped_mm`支持fp16/bf16/fp8，布局仅支持TN，为分组矩阵乘提供支持，整体覆盖了通用、低精度、分组等多样矩阵乘场景。
:::

## 七、如何试用 CuteDSL 后端

### 7.1 安装步骤

> CuteDSL 后端需要依赖 cutlass_api 库，目前需从 CUTLASS 仓库的特定分支安装。预计未来版本中，cutlass_api 将合并到 CUTLASS 的主分支，届时无需再执行该单独安装步骤。

```
# 安装CuTeDSL和矩阵乘法启发式库pip install nvidia-cutlass-dsl==4.3.5pip install nvidia-matmul-heuristics# 克隆并安装cutlass_api（来自cutlass_api分支）git clone --branch cutlass_api https://github.com/NVIDIA/cutlass.gitcd cutlass/python/cutlass_apipip install -e ".[torch]"
```

此外，还需满足以下要求：

- PyTorch 2.11 及以上版本（支持 mm、bmm、scaled_mm、grouped_mm 等核心 NVGEMM 功能）。FP4 内核支持（NVFP4、MXF4）及多项性能优化需要 PyTorch nightly 版本。

注：cutlass_api 目前要求 CuTeDSL 版本为 4.3.5 或更早。

### 7.2 使用方法

安装完成后，通过将 NVGEMM 添加到 Inductor 自动调优后端列表来启用该后端。以下是最小可运行示例：

```
import torchimport torch._inductor.config as configconfig.max_autotune_gemm_backends = "ATEN,TRITON,NVGEMM"A = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)B = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)@torch.compile(mode="max-autotune-no-cudagraphs")def f(a, b):    return a @ bout = f(A, B)  # 首次调用会触发自动调优
```

当 Inductor 在编译过程中遇到 GEMM 运算时，会同时评估 NVGEMM、ATen 和 Triton 的候选内核，并选择性能最优的方案。NVGEMM 不支持的运算会自动回退到其他后端。

也可通过环境变量进行配置：

```
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,NVGEMM" python my_script.py
```

如需控制每个 GEMM 运算的基准测试配置数量：

```
config.nvgemm_max_profiling_configs = 10  # 默认值为5；设为None则测试所有配置
```

## 八、未来工作：开发路线图

> 以下是 CuteDSL 后端的计划开发内容：

**尾处理融合基准测试**：随着 CuteDSL 编译时间瓶颈的消除，TorchInductor 能够对 GEMM 内核的尾处理融合决策进行基准测试。:mdnice-bold-italic[这一点至关重要，因为单独替换 cublas 的 GEMM 运算未必总能带来收益，而尾处理融合为持续超越无法进行任何融合的 cublas 提供了可能]。这项工作包括将内核最终选择推迟到融合过程完成后、跨后端评估融合与未融合变体，以及选择全局最优配置。cutlass_api 已提供支持尾处理融合（EFC）的内核，可支持辅助张量加载/存储、逐元素运算（加法、乘法、减法、除法）和激活函数（relu、sigmoid、tanh）。剩余工作集中在 Inductor 侧：将 Inductor 的融合决策映射到 EFC 内核接口，并将其集成到调度流程中。cutlass_api 计划后续支持更多尾处理运算——包括归约和行列广播。

**异步并行预编译与持久化缓存**：目前，候选内核通过 cute.compile()进行串行内联编译。我们正在添加基于子进程的并行预编译功能和编译产物的磁盘持久化缓存，使得预热阶段的自动调优能够完全跳过编译步骤。

**可导出的配置缓存**：设计一种可移植、人类可读的格式（JSON 或 protobuf）用于存储自动调优后的 GEMM 配置，并提供导入/导出 API 用于缓存操作。这将实现不同自动调优运行和环境之间的配置可移植性。

**FlexAttention 风格的矩阵乘法 API**：提供高阶 API，允许用户在矩阵乘法调用处指定后端偏好、分块配置和尾处理操作。这将为自动调优行为提供显式控制，并与可导出的配置缓存实现互操作。

**Quack GEMM 集成**：Tri Dao 的 Quack 库包含针对 Blackwell 架构优化的 GEMM 实现，我们将评估其性能与现有模板的对比，如果表现更优，将集成这些模板。

**AOT 编译支持**：对于推理部署场景，在导出时预编译 CuteDSL 内核可消除自动调优开销。这一功能依赖于 CuteDSL 4.4 版本计划推出的预编译 API，且需要研究 AOTI 集成所需的 C++可访问性。

**CUTLASS C++后端替代**：在新一代硬件上，CuteDSL 有望实现与 C++后端的完全功能对等。届时，CuteDSL 将作为替代方案，通过将 CUTLASS 集成统一为单一的 Python 路径，简化 TorchInductor 的代码库。

## 结论：总结与展望

本文详细介绍了 TorchInductor CuteDSL 后端的架构设计、当前的使用方法和基准测试结果。正如未来工作部分所示，这只是该项目的初步成果，后续还有大量开发工作正在进行中。如果您在使用过程中遇到任何问题、有疑问或新的想法，欢迎在 GitHub 上提交 issue[2]并@我们！

参考资料

[1] FP8 精度 GEMM 计算和尾处理融合方面已展现出卓越性能: *https://pytorchconference.sched.com/event/2A7dg/generating-state-of-the-art-gemms-for-heterogeneous-hardware-with-torchinductor-michael-lazos-and-henry-tsang-meta*

[2] PyTorch GitHub Issue页面: *https://github.com/pytorch/pytorch/issues*
