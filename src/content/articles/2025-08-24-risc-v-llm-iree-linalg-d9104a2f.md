关键词：GenAI、RISC-V、MicroKernel、decode、Llama、IREE

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/01.webp)

- Accelerating GenAI Workloads by Enabling RISC-V Microkernel Support in IREE
- RISC-V Summit Europe, Paris, France, 12-15th May 2025
- https://arxiv.org/abs/2508.14899
- 本文 3490 字，阅读需 14 分钟，播客 6 分钟如下

相关推荐

- [RISC-V CPU 上 3 倍推理加速！V-SEEK：在 Sophon SG2042 上加速 14B LLM](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890316&idx=1&sn=c170c9c6aab8c43a8c2facf7579a9f14&scene=21#wechat_redirect)
- [Triton 内核在 RISC-V 平台基于 MLIR 的编译与优化探索](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890238&idx=1&sn=cf15b6e26b823d3d5a3921f852c0cf43&scene=21#wechat_redirect)
- [GEMM 95% FPU 利用率！基于 MLIR 面向 RISC-V 指令集扩展的加速Micro-kernels的多层编译器后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889407&idx=1&sn=f5640780406954ae498d2c98d162d454&scene=21#wechat_redirect)

> 本项目在 IREE（一种基于 MLIR 的机器学习编译器和运行时）中实现了 RISC-V 微内核支持。

该方法首先在 IREE 的 passes 流水线中，**为 RISC-V64 目标实现了将 MLIR linalg 方言的收缩操作** （linalg dialect contraction ops，linalg 是 MLIR 中用于线性代数计算的方言，收缩操作可理解为一类包含矩阵乘法等的运算）**下转为 `linalg.mmt4d` 操作**，随后为 RISC-V 开发了经过优化的微内核。

针对 Llama-3.2-1B-Instruct 模型，将所获性能提升与上游 IREE 和 Llama.cpp（一种 Llama 模型的 C++实现）进行了对比。该方案**成效显著**：

::: block-1
- 基于优化微内核的编译版本，在预填充阶段吞吐量**较上游 IREE 提升 2.3 倍**；
- 解码阶段性能**较上游 IREE 提升 3.6 倍**，**较 Llama.cpp 提升 2.1 倍**；
- 在主流基准测试中，**tokens/s 平均性能达上游 IREE 的 3.2 倍**，:mdnice-bold-italic[全面刷新 RISC-V 架构下大语言模型的运行效率纪录。]
:::

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/02.webp)

## 关键问题

### 问题 1：作者针对 RISC-V64 目标设计的 VLEN 感知分块策略（Prefill 阶段 K=6、VLEN/8，Decode 阶段 K=1、VLEN/4）是基于何种具体的硬件特性（如寄存器数量、缓存结构）和 GenAI 计算需求推导得出的？该分块策略是否在不同 VLEN（如 256bits 以外）的 RISC-V 向量核心上具备可迁移性，且仍能维持相近的性能提升效果？

本文中 RISC-V64 的 VLEN 感知分块策略（Prefill 阶段 K=6、VLEN/8，Decode 阶段 K=1、VLEN/4），核心推导依据是**硬件寄存器利用率与寄存器压力的平衡**：作者明确提到，小于该尺寸的分块会导致硬件寄存器未充分利用，而大于该尺寸则会增加寄存器压力，引发寄存器溢出和重新加载，最终降低性能，该策略的制定未进一步提及具体硬件特性（如寄存器数量、缓存结构）的详细关联逻辑。

:mdnice-bold-italic[关于该策略在不同 VLEN（如 256bits 以外）RISC-V 向量核心的可迁移性，作者未提供相关测试数据或分析]，仅在基准测试中使用了 VLEN=256bits 的 MILK-V Jupiter 板，无法确定其在其他 VLEN 规格硬件上是否能维持相近性能提升效果。

### 问题 2：作者仅实现了 f16xf16->f32 精度场景下的 RISC-V mmt4d 微内核，而 GenAI 工作负载中还存在 f8、bf16 等更主流的低精度格式，未支持这些精度格式是否会限制该方案在实际 RISC-V GenAI 硬件中的应用范围？后续是否有明确计划扩展微内核的精度支持，且扩展后如何保证仍能实现类似单线程 50 倍解码加速的性能增益？

对于未支持 f8、bf16 等低精度格式是否限制应用范围，作者未直接讨论，但从研究定位来看，其核心目标是解决 RISC-V 微内核缺失的基础问题，:mdnice-bold-italic[优先实现 f16xf16->f32 精度场景的微内核以验证方案可行性]，**未涉及对其他精度格式应用限制的评估。**

关于后续扩展微内核精度支持的计划，作者未提及明确安排，也未分析扩展后如何保证类似单线程 50 倍解码加速的性能增益，仅聚焦于已实现的 f16xf16->f32 精度场景的性能与精度验证。

### 问题 3：论文性能基准测试仅基于 Llama-3.2-1B-Instruct 这一特定小参数 LLM 模型，且测试硬件限定为 MILK-V Jupiter 板（1.66GHz×8 RISC-V 向量核心），该方案在更大参数规模的 LLM 模型（如 7B、13B）或不同架构的 RISC-V 硬件（如更高频率核心、不同缓存层级）上，是否仍能保持与上游 IREE、Llama.cpp 的显著性能差距（如多线程 17 倍解码加速）？

关于方案在**更大参数 LLM 模型（如 7B、13B）或不同架构 RISC-V 硬件上的性能表现**，**作者未提供相关测试结果**：性能基准测试仅针对 Llama-3.2-1B-Instruct 小参数模型，且测试硬件限定为 MILK-V Jupiter 板（1.66GHz×8 RISC-V 向量核心、VLEN=256bits、RVA22 profile）。

因此，无法从本文内容中确定该方案在更大参数模型或其他架构 RISC-V 硬件上，是否仍能保持与上游 IREE、Llama.cpp 的显著性能差距（如多线程 17 倍解码加速），其**性能结论仅适用于论文限定的模型与硬件环境。**

## 本文目录

- 关键问题
  - 问题 1：作者针对 RISC-V64 目标设计的 VLEN 感知分块策略（Prefill 阶段 K=6、VLEN/8，Decode 阶段 K=1、VLEN/4）是基于何种具体的硬件特性（如寄存器数量、缓存结构）和 GenAI 计算需求推导得出的？该分块策略是否在不同 VLEN（如 256bits 以外）的 RISC-V 向量核心上具备可迁移性，且仍能维持相近的性能提升效果？
  - 问题 2：作者仅实现了 f16xf16->f32 精度场景下的 RISC-V mmt4d 微内核，而 GenAI 工作负载中还存在 f8、bf16 等更主流的低精度格式，未支持这些精度格式是否会限制该方案在实际 RISC-V GenAI 硬件中的应用范围？后续是否有明确计划扩展微内核的精度支持，且扩展后如何保证仍能实现类似单线程 50 倍解码加速的性能增益？
  - 问题 3：论文性能基准测试仅基于 Llama-3.2-1B-Instruct 这一特定小参数 LLM 模型，且测试硬件限定为 MILK-V Jupiter 板（1.66GHz×8 RISC-V 向量核心），该方案在更大参数规模的 LLM 模型（如 7B、13B）或不同架构的 RISC-V 硬件（如更高频率核心、不同缓存层级）上，是否仍能保持与上游 IREE、Llama.cpp 的显著性能差距（如多线程 17 倍解码加速）？
- 一、引言
- 二、理论框架
- 三、方法论
  - 3.1 实现 linalg 收缩操
  - 3.2 实现微内核函数
- 四、测试与性能基准测试
- 五、结论
- 参考文献

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

> IREE（中间表示执行环境，Intermediate Representation Execution Environment）[1]是一种:mdnice-bold-italic[基于 MLIR[2]（一种面向摩尔定律末期的编译器基础设施）的编译器和运行时]，其**开发目的是为各种平台（如 CPU、GPU 和加速器）编译机器学习模型。**

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/04.webp)

::: image-caption
IREE 工作流概览：IREE 对机器学习模型编译采用一种整体式方法：生成的中间表示（IR）既包含将数据依赖关系传达给诸如 Vulkan 这类低级并行流水线硬件/应用程序编程接口（API）所需的调度逻辑，也包含以诸如 SPIR-V 这类特定于硬件/API的二进制文件形式对硬件上的密集计算进行编码的执行逻辑。
:::

IREE 接收 MLIR 代码形式的机器学习工作负载作为输入，对其**应用算子融合** （operator fusion，将多个算子合并以减少计算开销的优化方法）和**分块** （tiling，将大矩阵分成小块进行计算的优化方法）等经典优化，并生成经过优化的二进制文件用于执行。

尽管编译器生成的代码在大多数情况下性能尚可，但在:mdnice-bold-italic[混合精度计算（mixed precision calculations，使用不同精度的数值类型进行计算，以平衡性能和精度）中，自定义内核的表现更优。]

IREE 包含一个针对各种 CPU 和 GPU 架构的优化微内核库。IREE 包含适用于 x86 和 ARM64 架构 CPU 的微内核，

> 然而，尽管 RISC-V 在 AI 硬件领域的应用日益增多，**RISC-V 微内核仍然缺失**。这一局限导致生成式人工智能（GenAI）模型在基于 RISC-V 的硬件上性能不佳。**为解决这一问题，本项目在 IREE 中实现了 RISC-V 微内核支持。**

## 二、理论框架

矩阵乘法是生成式人工智能（GenAI）工作负载中的关键计算。IREE 使用 MLIR 的 linalg 方言（一种用于表示线性代数操作的中间表示形式）中的收缩操作（contraction ops）来进行矩阵乘法，该操作在编译流程中会经历分块（tiling，将大矩阵分割成小块进行计算，以提高缓存利用率）和融合（fusion，将多个操作合并以减少数据传输开销）处理。**然而，如果数据没有预先排列，分块矩阵乘法（tiled matmul）的性能会不理想，导致较高的缓存未命中率** [3]。

> 为解决这一问题，**IREE 利用 `tensor.pack` 这一 MLIR 操作对数据进行重新排列，确保分块在内存中连续存储，之后再应用 `linalg.mmt4d` 进行优化计算**。`linalg.mmt4d` 生成的 4 维矩阵随后会通过 `tensor.unpack` 操作转换回原始布局。

::: block-1
1. `tensor.pack`：接收一个二维矩阵并将其转换为分块的四维矩阵，其中所有分块在内存中连续存储。
2. `linalg.mmt4d`：对由 pack 操作生成的四维左矩阵和右矩阵执行矩阵乘法。其中的“t”代表右矩阵的转置。
3. `tensor.unpack`：将 mmt4d 生成的四维结果矩阵转换回二维布局。
:::

这些 MLIR 操作会被 IREE 的编译过程降级（lowered，指将高级中间表示转换为低级表示）为对微内核的调用。尽管 IREE 的架构[4]设计初衷是便于用户集成任意微内核，但目前它仅包含适用于 x86 和 ARM64 架构、支持多种精度的 pack、unpack 和 mmt4d 操作的微内核。

针对 RISC-V 目标，我们已实现了 mmt4d 微内核。

## 三、方法论

我们所提出的方法论可分为两个步骤：

### 3.1 实现 linalg 收缩操

> 第一部分涉及实现 linalg 收缩操作到 `tensor.pack`、`tensor.unpack` 和 `linalg.mmt4d` 操作的转换。

目前，针对 x86-64 和 ARM64 目标，linalg 收缩操作到 `linalg.mmt4d`（以及 `tensor.pack`、`tensor.unpack`）的转换是在 `ireecodegen-materialize-device-encoding` 编译阶段（pass，编译器中的一个处理步骤）内进行的。

该阶段会根据目标架构确定输入矩阵 M、N 和 K 维度的分块大小。**我们修改了这一阶段，以支持将收缩操作转换为 `linalg.mmt4d`，并为 RISC-V64 目标执行 VLEN 感知分块** （VLEN 即向量长度，指根据向量寄存器长度调整分块大小以适配硬件）。

一旦完成这一转换，`linalg.mmt4d` 操作会通过后续阶段转换为对微内核函数的调用。分块大小的选择基于以下策略[5]：

::: block-1
1. 预填充（Prefill）分块大小 = M、N，K=6，VLEN/8.1
2. 解码（Decode）：分块大小 = M、N，K=1，VLEN/4，1
:::

经观察发现，:mdnice-bold-italic[选择比上述更小的分块大小会导致硬件寄存器利用不足]，而:mdnice-bold-italic[使用更大的分块大小则会增加寄存器压力]，导致寄存器溢出和重新加载，从而降低性能。

### 3.2 实现微内核函数

第二步包括实现微内核函数。针对$f16 \times f16 \rightarrow f32$的情况（即右操作数（rhs）和左操作数（lhs）为 f16 类型，结果操作数为 f32 类型），我们实现了 `mmt4d` 微内核。

由于**预填充阶** 段涉及通用**矩阵乘法** （GEMM），而**解码阶段** 涉及通用矩阵**向量乘法** （GEMV），因此我们为大语言模型（LLM）的预填充和解码阶段分别实现了 `mmt4d` 微内核。

## 四、测试与性能基准测试

> 为了验证新实现的微内核（指用于执行特定计算任务的小型核心程序）的**准确性**，我们使用基于 LM-Evaluation-Harness[6]构建的框架，对采用我们的微内核编译的 Llama-3.2-1B-Instruct 模型进行了评估。

结果汇总于表 1 中。使用 10x-IREE 编译的模型得分与从 Huggingface 获取的模型得分完全相同。

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/05.webp)

::: image-caption
表 1：LLaMA-3.2-1B-Instruct 模型在选定基准测试上的评估结果。该表对比了模型两个版本的性能：一个是从 Hugging Face 下载的版本，另一个是使用 10x-IREE 编译的版本。
:::

> 在**性能基准测试** 中，我们使用 10x-IREE 编译了 Llama-3.2-1B-Instruct 模型，并记录了预填充阶段（大语言模型处理输入时，首次生成序列前的准备阶段）和解码阶段（大语言模型逐 token 生成输出的阶段）的每秒处理的 token 数（token 是模型处理文本的基本单位，类似词语或子词）。

结果汇总于表 2 中。在单线程运行时，我们观察到解码性能相较于上游 IREE 有 50 倍的提升。在多线程运行时，预填充阶段的性能提升为 2 倍，解码阶段的性能提升为 17 倍。

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/06.webp)

::: image-caption
表 2：使用 llama.cpp、IREE 和 10x-IREE 编译的 LLaMA-3.2-1B-Instruct 模型在预填充阶段和解码阶段的性能（以每秒处理的 token 数表示）。基准测试是在 MILK-V Jupiter 开发板上进行的，该开发板配备 1.66GHz×8 的 RISC-V 向量核，向量长度（VLEN）为 256 位，采用 RVA22 架构。
:::

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/07.webp)

::: image-caption
图 1：使用 IREE 和 10x-IREE 编译的 Llama-3.2-1B-Instruct 模型在预填充阶段的性能对比。基准测试是在 MILK-V Jupiter 开发板上进行的，该开发板配备 1.66GHz×8 的 RISC-V 向量核，向量长度（VLEN）为 256 位，采用 RVA22 架构。
:::

![文章插图](/articles/2025-08-24-risc-v-llm-iree-linalg-d9104a2f/images/08.webp)

::: image-caption
图 2：使用 IREE 和 10x-IREE 编译的 Llama-3.2-1B-Instruct 模型在解码阶段的性能对比。基准测试是在 MILK-V Jupiter 开发板上进行的，该开发板配备 1.66GHz×8 的 RISC-V 向量核，向量长度（VLEN）为 256 位，采用 RVA22 架构。
:::

- 基准测试在 MILK-V Jupiter 开发板上进行，该开发板配备 1.66GHz × 8 的 RISC-V 向量核，向量长度（VLEN）为 256 位，采用 RVA22 架构配置。

## 五、结论

> 本研究通过对基于 10x-IREE 编译的 Llama-3.2-1B-Instruct 模型进行系统性的测试与性能基准评估，验证了新型微内核实现的有效性与优越性。

**在准确性验证方面**，基于 LM-Evaluation-Harness 框架的测试结果显示，采用 10x-IREE 编译的模型在 ARC_c 和 GPQA 基准测试中均取得了与原始 Hugging Face 版本完全一致的得分（59.4% 与 27.2%），这:mdnice-bold-italic[表明 10x-IREE 编译方案在实现性能优化的同时，能够完美保留模型的推理准确性，为其实际应用奠定了关键的可靠性基础。]

**性能基准测试结果进一步凸显了 10x-IREE 在 RISC-V 架构硬件上的显著优势**。在配备 8 核 1.66GHz RISC-V 向量核（VLEN 256 位，RVA22 架构）的 MILK-V Jupiter 开发板上，10x-IREE 编译的模型展现出突破性的性能提升：

- **单线程** 运行时，解码阶段性能较上游 IREE 提升 **50 倍**；
- **多线程** （8 线程）配置下，**预填充阶段性能提升 2 倍，解码阶段性能提升 17 倍。**

这一结果充分证明了针对 RISC-V 向量架构优化的微内核设计在高效执行大语言模型推理任务中的核心价值，尤其在资源受限的边缘计算设备中具有重要的实践意义。

综上，本研究:mdnice-bold-italic[通过实证数据验证了 10x-IREE 编译方案在保证模型推理准确性的前提下，能够显著提升 Llama 系列模型在 RISC-V 架构硬件上的运行效率，为大语言模型在嵌入式与边缘计算场景的部署提供了可行的技术路径]。未来工作可进一步扩展至更大规模模型（如 7B/13B 参数模型）及更多 RISC-V 硬件平台的适配验证，以全面评估该方案的普适性与扩展性。

## 参考文献

1. “IREE,” accessed: Feb. 7, 2025. [Online]. Available: https://iree.dev/
2. C. Lattner, M. Amini, U. Bondhugula, A. Cohen, A. Davis, J. Pienaar, R. Riddle, T. Shpeisman, N. Vasilache, and O. Zinenko, “Mlir: A compiler infrastructure for the end of moore’s law,” 2020. [Online]. Available: https://arxiv.org/abs/2002.11054
3. “Matrix Multiplication with MMT4D,” accessed: Feb. 7, 2025. [Online]. Available: https://iree.dev/community/blog/ 2021-10-13-matrix-multiplication-with-mmt4d/
4. “IREE Source,” Github, accessed: Feb. 7, 2025. [Online]. Available: https://github.com/iree-org/iree
5. “LLM Optimization and Deployment on SiFive RISC-V Intelligence Products,” accessed: Feb. 7, 2025. [Online]. Available: https://www.sifive.com/blog/llm-optimization-and-deployment-on-sifive-intellig
6. “lm-evaluation-harness Source,” Github, accessed: Feb. 7, 2025. [Online]. Available: https://github.com/EleutherAI/lm-evaluation-harness
