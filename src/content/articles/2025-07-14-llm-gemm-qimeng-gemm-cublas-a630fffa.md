关键词：GEMM、大语言模型、QiMeng-GEMM、自动调优框架、硬件平台、提示词工程

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/01.webp)

- QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models
- https://qimeng-team.github.io/QiMeng-GEMM/
- https://github.com/QiMeng-Team/QiMeng-GEMM
- 本文 5365 字，阅读需 22 分钟，播客 9 分钟如下

相关文章

- [GPU Kernel Scientist：LLM 驱动下的 GPGPU Kernel 迭代框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889551&idx=1&sn=606751b8d2b4190238f0fb800f1d5751&scene=21#wechat_redirect)
- [比 Ansor 平均快 2.7 倍！调优快70倍！基于分块表达式与搜索空间剪枝的 MBCI 算子链快速融合框架 MCFuser！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889535&idx=1&sn=4f85a584809f1533ef57644a39212741&scene=21#wechat_redirect)
- [芯片软硬件全自动化设计！计算所处理器国重实验室提出 QiMeng 方案：从特定领域处理器芯片大模型，到编译器工具链、高性能库！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888640&idx=1&sn=22f8a7efccb43f1b0bdbf4186529b167&scene=21#wechat_redirect)

> 作为众多科学与工程计算应用中的关键算子，充分利用不断演进的硬件架构（如 GPU 和 RISC-V）对通用矩阵乘法（General Matrix Multiplication, GEMM）进行自动优化至关重要。

尽管大语言模型（Large Language Models, LLMs）能够为简单任务生成功能正确的代码，但**尚未能产出高性能代码**。:mdnice-bold-italic[核心挑战在于深度理解多样化的硬件架构]，以及设计能有效释放 LLM 潜力以**生成高性能代码** 的:mdnice-bold-italic[提示词]。

::: block-1
本文提出一种名为 QiMeng-GEMM 的:mdnice-bold-italic[【新型提示机制】]，它使 LLM 能够理解不同硬件平台的架构特征，并自动搜索 GEMM 的优化组合。

:mdnice-bold-italic[QiMeng-GEMM 的核心是一组信息丰富、自适应且迭代的元提示] （meta-prompts）。基于此，**通过一种针对【元提示】最优组合的搜索策略，迭代生成高性能代码**。在 4 种主流 LLM、多种典型硬件平台以及代表性矩阵维度上进行的广泛实验，明确证明了 QiMeng-GEMM 在自动生成优化 GEMM 代码方面的卓越性能。
:::

与原始提示（vanilla prompts）相比，我们的方法在 **CPU 上实现了高达 113 倍** 的性能提升，在 **GPU 上提升达 9.65 倍**。即便与人类专家优化的库相比，我们的方法在 NVIDIA GPU 上可**达 cuBLAS 性能的 115%，在 RISC-V CPU 上可达 OpenBLAS 的 211%。**

值得注意的是，人类专家优化 GEMM 通常需要数月时间，而我们的方法将:mdnice-bold-italic[开发成本降低了 240 倍以上。] *

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/02.webp)

## 本文目录

- 一、引言
- 二、预备知识
  - 2.1 优化原语
  - 2.2 硬件特性
- 三、方法
  - 3.1 元提示（Meta-Prompt）
  - 3.2 自动调优（Auto-Tuning）
- 四、实验评估
  - 4.1 实验设置
  - 4.2 整体性能
  - 4.3 消融实验
- 五、结论
- 参考文献

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/03.webp)

## 一、引言

作为计算机科学中最重要的计算之一（Tan 等人，2011），GEMM 在包括 LLM 在内的深度学习应用中占据超过 90%的计算量。GEMM 的高性能实现可实现超过 1300 倍的加速（Hennessy 和 Patterson，2019），这使其**对大多数科学与工程应用的执行效率至关重要** （Jia，2014；Zhu 等人，2024b）。因此，设计高性能 GEMM 代码已受到学术界（Sui 等人，2024；Jang 等人，2024；Wei 等人，2024）和工业界（如 Intel、NVIDIA 和 AMD 等硬件厂商）的广泛关注。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/04.webp)

> 设计高性能 GEMM 主要有两种范式：手动实现和自动编译。

- 手动实现方法需要对硬件组件进行大量手工调优（Rasch 等人，2021），耗时且易出错，且无法在不同平台间移植。Intel 和 NVIDIA 等硬件厂商提供了手动优化的 GEMM 库（如 MKL（Krainiuk 等人，2021）和 cuBLAS（NVIDIA，2023a）），以充分利用底层硬件的计算能力。
- 自动编译方法将代码生成表述为在大型程序空间中的探索（Purini 等人，2013；Shin、Nam 等人，2021；Bi 等人，2023；Chen 等人，2024；Shi 等人，2023，2022），这可显著减少人力投入，但其性能往往落后于手动实现。

**随着硬件异构性和复杂性** （如 X86 CPU、RISC-V CPU（Greengard，2020）、NVIDIA GPU、Google TPU（Jouppi 等人，2017；Norrie 等人，2021；Jouppi 等人，2023））的持续增长，自动生成性能:mdnice-bold-italic[可媲美甚至超越手动实现] 的 GEMM 代码，:mdnice-bold-italic[对硬件平台的成功至关重要]。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/05.webp)

- GPT-4（Achiam 等人，2023）和 CodeLlama（Roziere 等人，2023）等 LLM 已在代码生成方面展现出显著能力（Bairi 等人，2024；
- Zhong 等人，2024；Li 等人，2024；Holt 等人，2024；Wang 等人，2023；Li 等人，2022；Chen 等人，2021；Nijkamp 等人，2022），为自动生成高性能 GEMM 代码提供了有前景的新途径。

然而，**现有 LLM 无法解决此问题，因为它们无法理解硬件平台的复杂特征** （Olausson 等人，2023），更**无法以复杂方式操作硬件组件**。我们观察到，适当的提示词设计可帮助 LLM 更好地理解硬件特征，从而显著提升功能准确性和性能。

> 为高性能 GEMM 代码生成设计 LLM 提示词面临三大挑战。

- 第一大挑战是设计鲁棒的提示词，以在性能优化过程中生成功能正确的代码。
- 第二大挑战是为最优提示词排列提供搜索空间。
- 第三大挑战是设计兼容不同硬件平台的通用提示词。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/06.webp)

> 为应对上述挑战，我们提出 QiMeng-GEMM，一种用于 LLM 在跨硬件平台自动生成高性能 GEMM 代码的新型提示机制，**如图 1 所示**。它通过**总结人类专家进行 GEMM 优化的核心启发式方法，定义了一组元提示**。然后，通过**自动搜索** 策略确定用于代码生成的最优元提示组合。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/07.webp)

::: block-1
QiMeng-GEMM 很好地解决了上述挑战：

1. 将整个代码生成过程表述为一系列定义明确的元提示，以降低推理复杂度并提高功能准确性；
2. 自动调优（auto-tuning）支持对元提示的所有可能组合进行穷举搜索，有可能超越手动优化以获得更好性能；
3. 元提示包含平台无关描述和平台特定提示，以实现跨不同平台的可移植性。
:::

据我们所知，这是首次通过利用 LLM 自动生成高性能 GEMM 代码。**本文做出以下贡献**：

- 提出一组包含五种**常见 GEMM 优化技术** 的**元提示**，使 LLM 能够理解各种硬件平台的**架构特征**。
- 引入**自动调优机制**，该机制自动**搜索元提示的最优排列**，以生成**针对特定硬件** 平台的高性能 GEMM 代码。
- 在各种平台和 GEMM 形状配置上进行了全面评估。结果表明，我们的方法生成的代码性能优于原始提示和人类专家手工优化的库。
- 最后但同样重要的是，与人类专家相比，所提出的 QiMeng-GEMM 框架将**开发成本降低了 240 倍以上。**

## 二、预备知识

### 2.1 优化原语

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/08.webp)

> GEMM 优化技术本质上可分为若干基本且可执行的操作（称为“原语”），这些操作构成了 GEMM 优化的基石。这些原语的图示如图 2 所示。

::: block-1
- **分块（Tiling）**：将矩阵划分为能放入缓存的小块，从而提升内存访问效率（Faingnaert 等人，2022）。
- **重排序（Reordering）**：通过交换代码中 for 循环的顺序来优化内存访问（Anam 等人，2013）。
- **向量化（Vectorization）**：将矩阵中的多个数据打包在一起，以实现高效的内存访问和计算（Katel 等人，2021）。
- **布局（Layout）**：改变矩阵的排列方式，以匹配特定硬件的内存访问特性（Kurzak、Tomov 和 Dongarra，2012）。
- **流水线（Pipeline）**：通过重叠计算和数据移动的不同阶段来提升性能，实现连续处理并减少操作间的空闲时间（Tan 等人，2011）。
:::

### 2.2 硬件特性

> 多样化的硬件特性为高性能 GEMM 代码生成带来了重大挑战，这些挑战主要源于不同的内存层次结构（memory hierarchy）和计算模型（computing model）。

- **内存层次结构**：这是决定合适分块策略的关键因素。例如，此外，GEMM 代码中的内存访问模式必须与硬件特性匹配，如 GPU 上的混洗（swizzle，即重新排列数据元素的操作）（Phothilimthana 等人，2019）和 CPU 上的数组打包（Jian Weng 等人，2023）。
- **计算模型**：直接影响 GEMM 的计算方式和整体效率。例如，GPU 和 CPU 分别采用 SIMT（单指令多线程，Single Instruction, Multiple Threads）和 SIMD（单指令多数据，Single Instruction, Multiple Data）模型。因此，

## 三、方法

> 在本节中，我们首先介绍 QiMeng-GEMM :mdnice-bold-italic[框架的概述]，该框架有效释放了 LLM 的潜力，从而在不同平台上自动生成高性能 GEMM 代码，**如图 3 所示。两个关键组件** 将在后续子节中详细阐述。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/09.webp)

### 3.1 元提示（Meta-Prompt）

元提示为各种通用优化技术和平台特定的优化细节提供了通用模板。它由平台无关描述、平台特定提示和实例化方法组成。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/10.webp)

#### 平台无关描述

平台无关描述是:mdnice-bold-italic[针对不同硬件平台的通用 GEMM 优化技术的【自然语言描述】]。如预备知识部分所示，我们**已将 GEMM 的通用优化技术分为五个原语**：分块、重排序、布局、向量化和流水线。

对于每个优化原语，相应的元提示由人类专家预定义，以降低推理复杂度，从而提升 LLM 的功能准确性。

#### 平台特定提示

> 平台特定提示包括针对特定硬件平台的自然语言描述和代码骨架，为:mdnice-bold-italic[跨不同硬件平台生成优化的 GEMM 代码提供详细的硬件特性]。平台特定的自然语言描述将通用 GEMM 优化策略与目标平台的硬件特性相结合。

- 在 NVIDIA GPU 上，数据流向从全局内存到共享内存再到本地内存，这需要使用多级分块；
- 而在 RISC-V CPU 上，只需考虑缓存局部性，因此单级分块即可满足需求。

平台特定的代码骨架对于提升 LLM 生成适当优化的 GEMM 代码的能力也至关重要。**它不需要完整的代码，而是仅包含嵌套 for 循环结构** （如分块优化）、**指令示例** （如向量化优化）或表示代码**计算阶段的伪代码** （如流水线优化）。

为帮助 LLM 全面理解不同平台的架构特性（如内存层次结构和计算模型），我们开发了:mdnice-bold-italic[提示存储库]，其中包含**预定义的自然语言描述和相应的代码片段**。这允许 GEMM 原语和平台:mdnice-bold-italic[特定提示的灵活组合]，以生成多样化的提示，该过程将在下一子节中详细说明。

#### 实例化（Instantiation）

> 实例化主要描述从通用元提示到针对特定平台提示的转换过程。

具体而言，在向 LLM 请求生成特定平台的高性能代码:mdnice-bold-italic[之前]，QiMeng-GEMM 首先从预定义的提示存储库中检索关键线索，并将相应的提示集成到元提示模板中。**RISC-V CPU 和 NVIDIA GPU 上分块元提示的实例化示例见图 4。**

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/11.webp)

- 对于 RISC-V CPU，由于**只需考虑缓存局部性**，分块元提示实例化所需的平台特定提示非常简单，例如将原始 GEMM 输出分解为更小的块。
- 对于 NVIDIA GPU，鉴于其独特的 warp（GPU 中一组协同执行的线程）编程抽象，平台**特定提示包括 warp 的基本解释和 warp 级分块的基本定义**。

最后，提供相关的代码骨架以帮助 LLM 理解此 GEMM 优化的语义。这种实例化机制减少了手动干预的需求，并且仅依赖于预定义的提示存储库（概述了硬件的架构特性），即**可自适应地执行特定平台的 GEMM 优化。**

### 3.2 自动调优（Auto-Tuning）

为充分探索 GEMM 优化的大型搜索空间并实现更高性能，我们提出了自动调优机制，该机制允许 LLM 以树状结构自动搜索优化原语的不同排列组合。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/12.webp)

算法 1 详细说明了自动调优过程。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/13.webp)

具体而言，它从 LLM 生成的原始 GEMM 代码作为根节点开始。

- 在 GEMM 优化的后续迭代中，LLM 分析当前代码的特性，并从给定的五个 GEMM 优化原语中**推荐多个候选。**
- 对于每个候选，从提示存储库中提取相关的平台特定优化提示，并**将元提示实例化为用于 LLM 交互的提示，以生成新的优化代码**。

在此过程中，**不正确的** GEMM 代码将从整个优化树中**剪枝，仅保留基于评估性能的前 K 个结果**。当优化后的 GEMM 代码不再有性能提升或达到最大迭代次数时，自动调优过程结束，生成最终的优化代码。

## 四、实验评估

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/14.webp)

### 4.1 实验设置

为研究提出的 QiMeng-GEMM 的效果，实验围绕影响性能的四个主要因素展开。由于篇幅限制，额外实验在附录中展示（NeuralTalk评：But 本文没有附录`<_<`）。

::: block-1
**硬件平台**：实验在多种硬件平台上进行，包括玄铁 C910 RISC-V CPU（Chen 等人，2020）、NVIDIA RTX 4070 GPU 和 A100 GPU。

**大语言模型（LLMs）**：该框架在四种前沿 LLM 上进行了验证，包括闭源模型 GPT-4o（OpenAI，2024）和 Claude 3.5 sonnet（Claude3.5，2024），以及开源模型 DeepSeek-CoderV2（Zhu 等人，2024a）和 Codestral-22B（Mistral.AI，2024）。

**提示变体**：将提出的元提示与原始提示（vanilla prompt）和思维链提示（CoT prompt，Wei 等人，2022）进行比较。原始提示包含基本的 LLM 激活短语、GEMM 的基本概念和所需的函数接口。思维链提示在原始提示基础上添加了鼓励 LLM 逐步思考的短语。

**矩阵维度**：实验在深度学习中常见的典型矩阵乘法维度上进行（LeCun 等人，2015），如 Llama2-7b（Touvron 等人，2023）、Mistral-7B（Jiang 等人，2023）等。
:::

### 4.2 整体性能

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/15.webp)

表 1 展示了 QiMeng-GEMM 在不同硬件平台上与几种前沿 LLM 结合的性能。表 2 展示了 QiMeng-GEMM 与简单 CoT 方法的性能对比。从表中可以清楚地看出，我们的方法能够使 LLM 生成兼具功能正确性和高性能的代码。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/16.webp)

QiMeng-GEMM 在所有四种前沿 LLM 上均优于原始提示。以 GPT-4o 为例，在 C910 CPU 上，我们的方法比原始提示性能提升高达 113.67 倍。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/17.webp)

在 RTX 4070 和 A100 GPU 上，我们的方法分别实现了高达 9.65 倍和 3.92 倍的性能提升。更重要的是，我们的方法甚至可以实现比手动优化库（如 OpenBLAS 和 cuBLAS）更好的性能。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/18.webp)

在 C910 CPU 上，QiMeng-GEMM 生成的代码显著优于 OpenBLAS，达到其 211%的性能（10.23/4.85）；在 RTX 4070 GPU 上，达到 cuBLAS 115%的性能（10.55/9.14）。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/19.webp)

随后，我们将呈现与四个影响性能的主要因素相关的实验观察，并深入解析其背后的原理。主要实验结果在正文部分展示，更多细节请参考附录。

#### 硬件平台的影响

使用我们精心设计的元提示，LLM 自动生成的代码在 RISC-V CPU 上的性能通常优于 NVIDIA GPU。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/20.webp)

C910 CPU 和 RTX 4070 GPU 上的性能提升分别高达 113.67 倍和 9.65 倍，这凸显了 CPU 和 GPU 在加速效果上的差异。

与 CPU 上的 SIMD（单指令多数据）并行化不同，GPU 使用 SIMT（单指令多线程）并行化，这导致 GPU 的编码模式与通常用于 CPU 代码的传统 C 编程范式不同。

不同于 CPU 中的传统 C 代码风格，GPU 上的 CUDA C 编程范式通过编写由每个线程大规模执行的内核函数来实现功能。**这导致 CUDA C 中的外层循环是隐式的，与 LLM 通常训练的大量 C 风格代码形成领域偏移。**

#### LLM 的影响

> LLM 越强大，使用 QiMeng-GEMM 生成复杂任务代码时的性能提升越大。

在不同 LLM 的实验中，两个闭源 LLM 在不同硬件平台上展示了可靠的性能优化，在 RTX 4070 上提升高达 9.65 倍。在开源模型中，参数更大的 DeepSeek-v2（2360 亿参数）获得了相对更好的 8.79 倍结果，而**参数较小的 Codestral-22B（220 亿参数）在 GPU 上表现出较低的性能**。

**解决 GEMM 优化等挑战性任务需要 LLM 具备强大的能力**。

- 在预训练阶段，LLM 需要从训练数据中学习足够的领域知识，以便通过最小的提示就能泛化到新任务。
- 在推理阶段，LLM 需要理解呈现各种领域知识和示例的超长上下文。

此外，LLM 必须能够遵循复杂的提示指令，自动调度多轮对话并整合结果。

#### 提示机制的影响

> 信息丰富、自适应且迭代的提示可以高效激活 LLM 生成高性能代码的潜力。

我们的:mdnice-bold-italic[元提示始终优于原始提示和简单的 CoT 提示]。它不仅**包含对优化方法的高度精炼和关键的描述**，还融入了实现**优化技术所必需的核心代码结构和代码提示**。

例如，在 GPU 的 Warp 分块（Warp Tiling，Warp 是 GPU 中一组协同执行的线程）方法中，需要在分块策略中加入额外的粒度级别。我们只需将“for (wm; wm<WM; wm++)”循环添加到模板中，即可生成通过 Warp 分块优化的代码。

我们的提示能够以简洁高效的方式充分释放 LLM 的潜力，引导它们专注于高质量的解决方案空间，并快速找到正确的最优解。

#### 矩阵维度的影响

> 矩阵维度越大，QiMeng-GEMM 实现的性能增益越大。

以 C910 CPU 为例，在维度为 4096 的矩阵上实现了 113.67 倍的加速，而在 512 维度上为 34.15 倍。在 CPU 上的 GEMM 算法中，**原始实现通常随着矩阵维度的增加面临更高的缓存未命中率，导致随着矩阵尺寸增大，GFLOPS** （每秒十亿次浮点运算）**下降**。

更重要的是，近年来随着 LLM 规模的快速增长和长上下文的广泛使用，GEMM 维度也迅速增加。**对更大矩阵的性能优化已成为影响整体性能的关键因素**。QiMeng-GEMM 对更大矩阵实现了显著更高的加速，这与这一趋势完美契合。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/21.webp)

### 4.3 消融实验

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/22.webp)

#### 平台特定提示的影响

如表 3 所示，提出的平台特定提示帮助 LLM 全面理解平台架构特性，从而显著提升 LLM 生成代码的功能正确性和性能。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/23.webp)

具体而言，**对于块分块等简单优化技术，使用和不使用我们提示的性能差距并不显著**。然而，对于**线程分块**，提出的提示可以将 LLM 生成代码的**性能从 4.31 TFLOPS** （每秒万亿次浮点运算）**提升到 14.59 TFLOPS**。

此外，没有平台特定提示，LLM 无法为特定硬件定制的 GEMM 优化生成正确的可执行代码，如 Warp 分块、线程分块和 Float4 向量化。

#### 自动调优的影响

为了更好地说明 QiMeng-GEMM 生成高性能 GEMM 代码的能力，我们检查了如图 5 所示的自动调优过程。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/24.webp)

需要注意的是，我们保留了所有搜索轨迹，没有修剪不正确或低性能的代码，以便对搜索空间质量进行更全面的评估。结果表明，在 25 条搜索轨迹中，QiMeng-GEMM 持续生成功能正确的 GEMM 代码（76%），**25 条轨迹中有 9 条（36%）超越了手动优化的 OpenBLAS 库的性能**。值得注意的是，QiMeng-GEMM 实现的**最佳性能甚至比 OpenBLAS 高出 100%。**

#### 开发成本的影响

表 4 比较了 QiMeng-GEMM 在两个代表性平台上针对 1024 维度矩阵的开发成本。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/25.webp)

两名有三年编码经验的计算机科学硕士学生和两名有五年经验的软件工程师分别作为初级和高级编码员。结果显示：

::: block-1
- 在 GPU 上，该方法将开发时间**从 24 天缩短到 10 分钟**，同时还超越了初级编码员 30%的性能。与高级编码员相比，我们的方法将**开发成本降低了 240 倍** （假设每天工作八小时），:mdnice-bold-italic[同时保持了相当的性能]。
- 同样，在 CPU 上，它将开发时间**从 3 天缩短到 7 分钟**，:mdnice-bold-italic[超越了高级编码员 221%的性能]。
:::

## 五、结论

> 在本文中，我们提出了一种名为 QiMeng-GEMM 的新型提示机制，以充分激活 LLM 在高性能 GEMM 代码生成中的潜力。

通过广泛的实验比较，我们发现:mdnice-bold-italic[关键挑战] 在于**通过元提示释放 LLM 理解各种架构特性的潜力，并自动搜索用于优化 GEMM 的元提示的最佳组合**。

::: block-1
结果表明，我们的方法生成的代码性能可以超越人类专家手工优化的库。至关重要的是，虽然人类专家优化 GEMM 通常需要数月时间，但我们的方法将:mdnice-bold-italic[开发成本降低了 240 倍以上]。

QiMeng-GEMM 的进展在快速演进的架构中展示了巨大的潜力。
:::

未来，我们计划将**适配扩展到更多架构和指令集**。本文中描述的新方法将作为专用工具提供，以鼓励该领域的未来研究。

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/26.webp)

## 参考文献

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/27.webp)

![文章插图](/articles/2025-07-14-llm-gemm-qimeng-gemm-cublas-a630fffa/images/28.webp)

交个朋友
