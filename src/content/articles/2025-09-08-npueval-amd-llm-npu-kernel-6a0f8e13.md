关键词：NPUEval、NPU Kernels、LLMs、Vectorization、Open Source Compilers、Performance Evaluation

> **【NeuralTalk简评】* AMD NPU 的 AIE tile 依赖 VPU（向量处理单元），向量化代码是释放其算力、降低延迟与功耗的关键。而 NPU 编程生态不成熟、架构碎片化，LLM 生成的代码常“功能正确但低效”，只有向量化 kernel 能匹配其为 AI 负载设计的并行架构，因此本文在代码生成重点聚焦向量化。*

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/01.webp)

- NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers
- https://arxiv.org/abs/2507.14403
- https://amdresearch.github.io/NPUEval/
- https://github.com/AMDResearch/NPUEval
- 本文 9470 字，阅读需 32 分钟，播客 7 分钟

相关推荐

- [端到端加速4.5倍！端侧SoC动态稀疏注意力 shadowAttn：关键token稀疏计算架构且兼容主流框架释放NPU性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891725&idx=1&sn=1c315204fb9f1edc0b26e679d8561983&scene=21#wechat_redirect)
- [高通端侧生成式AI：Hexagon NPU架构创新与异构实现98%性能提升、0.6秒图像生成与20tokens/sLLM推理](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891580&idx=1&sn=dafeb4d9f81e0675149ef8293a7fdba4&scene=21#wechat_redirect)
- [HotChips 2025：从 PyTorch 到 CUDA：大模型时代 GPU 内核自动生成的技术跃迁，Tri Dao最新演讲](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891347&idx=1&sn=173b8638e00da3610dd90ba7d5a4b7f6&scene=21#wechat_redirect)

> 神经处理单元（NPUs）在消费级设备等功耗敏感型设备中日益凸显重要性，而 AI PC 的定义便包含了这类专用处理器。要在这些设备上高效运行 AI 工作负载，需要由经过优化的核函数构成的库。

创建高效核函数不仅要求开发者**掌握包含向量内在函数（vector intrinsics）的领域专用 C++语言，还需深入了解目标架构的知识。**

与经过多年发展已趋于成熟的 GPU 编程不同，:mdnice-bold-italic[NPU 编程尚处于新兴阶段，不同硬件平台的开发者社区规模更小且更分散。这种分散性给利用大型语言模型（LLMs）辅助编写 NPU 核函数带来了挑战] ——因为:mdnice-bold-italic[领域专用的优化代码示例在 LLM 预训练数据中的占比极低。]

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/02.webp)

::: block-1
本文提出了 **NPUEval——一个用于编写和评估 NPU 核函数的基准测试集**，该测试集包含 102 个机器学习工作负载常用的算子。我们利用面向 AMD NPU 的开源编译器工具，在实际硬件上从**功能正确性** 和**向量化效率** 两个维度，对 LLM 生成的代码进行评估。

我们**评估了一系列最先进的 LLM，涵盖专有模型和开源权重模型**。DeepSeek R1 等最新推理模型展现出了良好前景，在部分核函数上实现了开箱即用（out-of-the-box）的 50%以上向量化率。:mdnice-bold-italic[然而，即便借助编译器反馈和向量化核函数示例，整个数据集的平均向量化分数仍维持在约 10%——这表明即便对于前沿模型而言，该数据集仍是一项艰巨的挑战。]
:::

该数据集和评估代码将以宽松的开源许可证发布，为推进代码生成和 NPU 核函数优化领域的研究提供关键基准。

## 本文目录

- 一、引言
- 二、向量化
- 三、数据集
  - 3.1 数据集结构
  - 3.2 数据集考量因素
- 四、内核评估
  - 4.1 评估工具链设置
- 五、生成
  - 5.1 系统提示词
  - 5.2 检索增强生成
  - 5.3 编译器反馈
  - 5.4 LLM 设置
- 六、结果
  - 6.1 开箱即用的大语言模型评估
  - 6.2 功能正确性
  - 6.3 性能
  - 6.4 生成失败原因分析
- 七、局限性
- 八、未来工作
- 九、结论
- 参考文献

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/03.webp)

## 一、引言

大型语言模型（LLMs）已成为极具吸引力的代码生成辅助工具，在各类编码基准测试中实现了突破性性能[1][2][3][4][5][6][7][8]。对于 SWE-Bench[9]这类侧重问题解决的更具挑战性的基准测试，前沿模型目前的成功率已超过 62%，且仍在快速提升[10]。然而，尽管这些模型在典型的 Python 任务中表现出色，但许多基准测试更看重“通过/失败”指标，而非代码质量。**一个编写粗糙的算法或许能通过功能测试，但却无法适用于生产系统**。这一点在为硬件加速器生成高效核函数时尤为棘手——**未优化的核函数在加速应用中毫无用处**。

目前，各大芯片厂商[11][12][13][14][15][16]纷纷推出 NPUE 以支持 AI 加速工作负载。但现有关于优化核函数生成的相关研究主要针对 GPU[17][18][19]。

::: block-1
- GPU 编程语言及其生态系统经过多年发展已趋于成熟，且在 LLM 预训练数据中的覆盖率较高。
- 与之相反，NPU 编程的开发者社区规模更小，且因硬件架构差异而更加分散，软件栈也不够成熟。
:::

我们发现，即便前沿模型能够很好地解决通用软件问题，它们在生成厂商特定代码的优化方案时仍常常面临困难。

本文:mdnice-bold-italic[提出了 NPUEval——一个专为评估 LLM 生成 AMD NPU 向量化核函数能力而设计的 数据集]。由于 NPUE 在消费级设备中的可及性，这类加速器成为核函数代码生成研究的理想平台——通过在消费级笔记本电脑和迷你主机（miniPCs）上运行，可最大限度地保证研究的可复现性。整个评估工具集（包括编译器）均基于开源工具构建，且我们将以宽松许可证发布该数据集及相关代码。该数据集包含每个核函数的提示词（prompts）、行为模型（behavioral models）和数据传输信息（data movement information）。

我们的评估框架从两个维度衡量 LLM 性能：

- 一是功能正确性（与 HumanEval[1]等基准测试类似）
- 二是周期精确的性能指标（如向量化分数——由执行向量指令所消耗的周期占总周期的百分比决定）

我们的研究结果表明，LLM 能够生成可在 NPU 硬件上运行的功能正确代码，:mdnice-bold-italic[且通过添加编译器反馈、检索增强生成（RAG）等简单技术，它们能够编写向量化实现。]

本文的贡献如下：

::: block-1
- 提出 NPUEval——据我们所知，这是首个用于评估 LLM 生成 NPU **核函数向量化代码能力** 的基准测试集。
- 提供**一套面向 AMD NPU 编程的全开源技术栈** （编译器和驱动程序），该技术栈聚焦于单核函数开发，可在笔记本电脑上运行。
- 开发一套全面的评估工具集，**支持正确性评估、周期精确性能指标计算和核函数微码输出**，且可基于智能体工作流（agentic workflows）进一步优化。
- 设计一个参考 :mdnice-bold-italic[LLM 流水线，该流水线利用编译器反馈和向量化核函数示例的检索增强生成（RAG），引导 LLM 输出向向量化方案靠拢，并减少生成内容的“幻觉”] （hallucinations）。
:::

## 二、向量化

> 本文所针对的硬件是最新 AMD 笔记本电脑和迷你 PC（如基于 Phoenix、Hawk Point 或 Strix Point 架构的设备）中的 NPU，这类设备在内部采用人工智能引擎（AI Engines，AIE）来执行内核计算。

以 Phoenix 架构设备为例，其包含 20 个 AIE 瓦片（AIE tiles）——每个瓦片都是可独立处理数据的计算单元。每个 AIE 瓦片均配备一个向量处理单元（Vector Processing Unit，VPU）和一个标量单元，这种结构使得这些计算单元具备较高的灵活性，因为标量单元可执行任意 C++代码。

与 GPU 类似，这些 AIE 瓦片也需要内核（kernels）；而为了充分利用硬件架构以实现最高性能，采用经过优化的向量化代码编写这些内核至关重要。低效的内核会大幅增加延迟和功耗，这将完全违背在设备中集成 AIE 的初衷。然而，:mdnice-bold-italic[要尽可能榨取性能，我们需要最大限度地利用 VPU 进行计算，这意味着需通过 C++ AIE API 和向量内在函数（vector intrinsics）编写向量化代码。]

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/04.webp)

> 图 1 展示了一个向量化的简单示例。

- 其中，标量代码（图 1a）是一个直通内核（passthrough kernel），它通过 for 循环逐元素迭代，将输入数据复制到输出缓冲区；
- 而图 1b 中的向量化代码实现了相同的内核功能，但通过向量指令利用 VPU——该代码会一次性处理一块数据而非单个元素，从而显著提升吞吐量。

用于向量化的优化内核可能比这个简单示例更复杂，代码行数也更多，但向量化方法保持一致（参考如下向量化代码示例）。:mdnice-bold-italic[大语言模型（LLMs）面临的挑战在于，生成的向量化代码需同时满足功能正确和 VPU 高效利用的要求，且需使用正确的 API。]

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/05.webp)

::: image-caption
Listing 1: Elementwise Add example (Vector score: 13%)
:::

## 三、数据集

NPUEval 数据集是一组提示（prompts）的集合，这些提示包含：

- AIE 内核定义，并附带文档字符串（docstrings）；
- 文档字符串中包含内核描述、输入/输出示例，以及预期的数据移动（data movement）和运行时参数。

下图展示了一个提示示例。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/06.webp)

### 3.1 数据集结构

数据集中的每个内核均包含以下组件：

::: block-1
- **提示（Prompt）**：受 HumanEval[1]启发的提示，包含内核功能的基本说明和函数签名。
- **数据移动信息（Data movement information）**：指定进出 AIE 瓦片的数据大小。
- **行为模型（Behavioral model）**：基于 NumPy 的目标内核 Python 实现。
- （可选）**标准 C++解决方案（Canonical C++ solution）**：Python 解决方案用于正确性评估，而标准 C++解决方案则用于测试框架（harness）的回归测试。
:::

:mdnice-bold-italic[为了生成能充分利用目标硬件的优化内核，内核可能需要针对特定的固定输入数据大小编写]。因此，我们为每个内核包含了数据移动信息，用于指定进出瓦片的数据大小[20]。

### 3.2 数据集考量因素

由于核心任务是为专用硬件编写代码，在构建该数据集时，需考虑一些在常规 LLM 代码生成基准中通常不会涉及的问题与挑战。

#### 3.2.1 数据类型支持

> 除了 int8 等常用于加速量化神经网络的通用 NPU 数据类型外，AMD 的 NPU 还支持 bfloat16 这一相对较新的数据类型。

目前，NumPy 等 Python 库尚未原生支持这类新型机器学习数据类型。为了生成 bfloat16 的行为模型，我们使用了 Google Jax 的 ml_dtypes 库[21]——该库提供了与 NumPy 兼容的 bfloat16 类型。

此外，:mdnice-bold-italic[AMD NPU 的 bfloat16 实现支持可编程舍入模式（programmable rounding mode），我们利用这一特性确保相同操作能产生完全一致的输出]，从而最大限度减少累积舍入误差。

#### 3.2.2 浮点内核评估

浮点内核的输出结果可能无法与 Python 行为模型的输出完全一致。同一问题可通过不同算法解决，而这些算法在精度上可能存在细微差异。

::: block-1
- 例如，在计算 sin 等三角函数时，NPU 可能采用与行为模型中 CPU 实现不同的多项式逼近方法；
- 此外，浮点运算不满足结合律，因此运算顺序不同也会导致结果略有差异。
:::

在评估生成内核的正确性时：

::: block-1
- 我们采用了 1e-2 的默认绝对误差阈值（与相关内核生成研究[17]的默认阈值一致）。
- 该默认阈值适用于逻辑算子和整数算子，但对于低精度浮点运算，我们将容差设为 2e-2，为 LLM 在算法搜索空间中提供一定灵活性；
- 对于 tanh 或 softmax 等更复杂的激活函数内核，容差则设为 3e-2。
:::

#### 3.2.3 数据移动

为了使用 MLIR-AIE 工具[22]编译 AIE 图（AIE graph），我们为每个内核预定义了数据移动规则，并将其包含在提示中。

- 测试向量（test vectors）用于推断数据移动方式，并自动生成用于配置设备的底层 MLIR 代码。
- 在数据流编程（dataflow programming）中，内核需知晓进出瓦片的数据大小[23]——这对于编译器优化缓冲区分配和存储体分配（bank assignment）至关重要[20]。

#### 3.2.4 硬件可获取性（Hardware availability）

> 大多数针对 Python 或其他高级语言问题的代码生成基准具有便利性，因为它们可在任意商用硬件上运行。而针对硬件加速器的代码编写则面临挑战：**需在目标设备上进行测试，或使用模拟器（通常速度慢得多，且限制迭代次数）。**

尽管在嵌入式应用的开发板[24]上也能找到 AIE，但这类平台对主流用户而言可及性较低。

我们选择 AMD NPU 作为目标硬件，是因为其在消费级硬件上易于获取，这使得更广泛的研究人员能够复现我们的实验结果。本文所呈现的结果均通过搭载 Ryzen 9 7940HS 芯片的笔记本电脑获得。

## 四、内核评估

> 生成的内核将通过以下标准进行评估：

::: block-1
- 编译性（Compilation）——内核是否可编译？其是否为使用有效 NPU 向量单元 API 调用和内部函数（intrinsics）的语法正确的 C++代码？
- 功能正确性（Functional correctness）——在内置误差容限下，内核针对给定输入是否能生成正确输出？
- 性能（Performance）——内核执行耗时多久？向量处理单元（VPU）的利用率如何？
:::

### 4.1 评估工具链设置

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/02.webp)

> 评估流水线的高层级概览如图 3 所示。

- 该数据集包含内核提示词（prompt）、行为模型（behavioral model）和测试向量（test vector）。
- 评估工具链会编译生成的 C++内核代码，并在目标硬件（on-target）上运行。
- 随后，将输出结果与 Python 行为模型生成的预期模拟输出进行对比。

| 类别 | 子项（Subitem） | 详细说明 |
| --- | --- | --- |
| 数据集（Dataset） | 提示词（Prompt） | 受 HumanEval 启发，包含内核功能说明与函数签名，为 LLM 生成 NPU 核代码提供基础输入上下文 |
|  | 行为模型（Behavioral） | 基于 NumPy（或 Google Jax ml_dtypes 库，支持 bfloat16）的 Python 实现，作为 NPU 核输出的参考基准，用于正确性验证 |
|  | 参考输入（Reference inputs） | 数据集包含的测试向量输入，用于驱动 NPU 核执行，与行为模型的输入一致 |
|  | 参考输出（Reference output） | 由行为模型基于参考输入生成的预期输出，用于对比 NPU 核的实际输出以判定功能正确性 |
| 评估系统（Evaluation System） | LLM | 代码生成主体，接收数据集的提示词等信息，输出目标 NPU 核的 C++代码 |
|  | C++代码 | LLM 的核心输出产物，需符合 AIE 向量 API 与内在函数规范，是后续编译与运行的基础 |
|  | LLVM-AIE 编译器（Compiler） | 用于编译 LLM 生成的 C++代码，为 LLVM 的分支，专门适配 AIE 内核编程，完全开源（可从 GitHub 仓库安装） |
|  | MLIR-AIE | 开源框架，用于配置 AIE 阵列、将内核加载到正确 AIE 瓦片，是 NPU 编程的核心配置工具 |
|  | 内核目标文件（Kernel object） | LLVM-AIE 编译器编译 C++代码后生成的中间产物，尚未成为可直接在 NPU 上运行的文件 |
|  | 二进制文件（Binary） | 内核目标文件经进一步处理后生成的最终可执行文件，可直接在 NPU 上加载并执行 |
|  | NPU | 代码运行的目标硬件（如 AMD Phoenix、Hawk Point 架构设备的 NPU），负责执行二进制文件并输出计算结果 |
| 性能（Performance） | 周期计数（Cycle count） | 对 NPU 核执行过程的周期统计，是衡量内核运行效率的基础指标 |
|  | 总周期（Total cycles） | NPU 核从启动到执行完成消耗的全部周期，反映内核整体运行耗时 |
|  | VPU 周期（VPU cycles） | 向量处理单元（VPU）执行指令消耗的周期，与总周期的比值即为 VPU 利用率，是评估内核向量化效率的关键指标 |
|  | 微码（Microcode） | NPU 核执行时的底层指令，可用于分析内核指令级执行细节，辅助优化性能 |
|  | 编译器反馈（Compiler feedback） | 若 LLM 生成的 C++代码编译失败，LLVM-AIE 编译器会输出错误信息，反馈给 LLM 用于迭代修正代码（最多重试 10 次，10 次后收益递减） |
| 正确性（Correctness） | 输出数据（Output data） | NPU 执行内核后输出的实际计算结果，需与行为模型的参考输出对比 |
|  | 通过/失败（Pass/Fail） | 对比输出数据与参考输出后的判定结果：若误差在预设阈值（如 int8 为 1e-2，bfloat16 为 2e-2）内则“通过”，否则“失败” |
|  | 误差统计（Error stats） | 包括最大绝对误差（输出值与参考值的数值差）和最大相对误差（相对于参考值的百分比差），用于量化输出数据与参考输出的偏差程度 |
|  | 其他指标（Additional metrics） | 暂无额外细化，可根据实际评估需求补充（如误差分布、异常值比例等） |
| 编译器（Compiler） | LLVM-AIE 编译器 | 同“评估系统”中的 LLVM-AIE 编译器，x86 机器即可复现带编译器反馈的代码生成步骤，即使无 NPU 也可验证代码语法正确性 |
| 应用构建器（Application builder） | MLIR-AIE 框架 | 同“评估系统”中的 MLIR-AIE，与 IRON 绑定配合使用，共同完成 AIE 阵列配置 |
|  | IRON 绑定（bindings） | 辅助 MLIR-AIE 实现 AIE 阵列配置与内核加载的工具，提升配置过程的便捷性与兼容性 |
|  | 模板化图表（templated graphs） | NPUEval 评估工具链提供，支持多数内核数据类型与输入/输出端口数量，简化 MLIR-AIE 的配置流程 |
|  | 数据传输（data movement）信息 | 通过解析数据集的测试向量和行为模型输出生成，传递给 MLIR-AIE，用于优化缓冲区分配与存储体分配，确保数据高效进出 AIE 瓦片 |
| 运行时（Runtime） | NPU 驱动程序的 Python 绑定 | 实现 NPU 的数据传入/传出与内核执行控制，为 Python 环境提供接口，降低操作复杂度 |
|  | NumPy 数组处理 | 借助 Python 绑定，可轻松处理 NumPy 格式的输入/输出数据，适配行为模型的 Python 生态 |
|  | 输出对比验证 | 快速将 NPU 输出数据与行为模型的参考输出对比，自动完成正确性判定（Pass/Fail） |
| 性能指标（Performance metrics） | 执行时间 | 内核从启动到输出结果的实际耗时，是用户可直观感知的效率指标，与周期计数存在对应关系（需结合 NPU 主频换算） |
|  | 精度测量（最大绝对误差、最大相对误差） | 同“正确性”中的误差统计，可辅助智能体迭代优化（如误差过大则改进算法，VPU 利用率低则调整向量化策略） |
| 后处理（Post-processing） | 代码块提取（正则表达式） | 针对 LLM 输出中混入自然语言、用```标记代码块的问题，通过正则表达式提取 Markdown 中的代码块，确保代码可用于后续编译 |
|  | 无关函数截断 | 截断 LLM 额外生成的 main()等测试函数（LLM 常输出 main()以测试内核），仅保留目标 NPU 核函数，避免干扰编译与执行 |

## 五、生成

> 除数据集和评估工具链外，我们还提供了一个参考代码生成流水线，可用于多种 LLM。:mdnice-bold-italic[该流水线包含一个系统提示词（用于引导 LLM 生成独立的 C++解决方案）]、:mdnice-bold-italic[一个基于开源 AIE 内核构建的向量数据库（用于检索增强生成，RAG）]，以及:mdnice-bold-italic[向 LLM 提供的编译器反馈（用于减少错误和幻觉输出）。]

### 5.1 系统提示词

系统提示词通常用于通过提供上下文和预期结果来引导模型行为。

初步实验表明，**添加系统提示词对于让许多 LLM 生成格式规范的代码至关重要**。部分模型可能会尝试将解决方案拆分为多个代码块、添加解释说明，或生成过于冗长的输出，这会导致难以自动解析解决方案。NPUEval 使用的提示词如下：

```
1. You are a part of a code generation system for AIE (AI Engines).2. 3. * Your job is to write C++ code for a single kernel that will run on an AIE tile.4. * Produce only the C++ code for the requested kernel including any required headers and imports.5. * Make sure the C++ code is complete and self contained in a single code block.6. * Name the function exactly as specified in the request, and output only the kernel (no main(), examples, explanations or extra code).
```

尽管系统提示词无法保证模型一定会遵循规则并生成独立的 C++内核代码块，但从经验来看，它非常有效。部分模型在遵循指令方面表现欠佳，这种情况下仍需进行更多后处理（例如，移除多余的 `main()`函数）。

### 5.2 检索增强生成

> RAG 数据库由来自开源 GitHub 仓库[22][26]的内核组成。我们对这些内核进行了手动修改，**移除了所有标量实现，仅保留向量化代码示例。**

我们使用 llama_index[27]来管理向量数据库和示例检索。嵌入模型（embeddings model）采用 OpenAI 的 text-embedding-ada-002（llama_index 的默认模型）。对于每个测试提示词，会向 LLM 提供两个示例。RAG 系统本身是一个丰富的优化领域[28]，但这超出了本文的研究范围——我们将其留作未来工作，评估过程中使用 llama_index 的默认设置。

### 5.3 编译器反馈

生成阶段和评估阶段使用相同的编译器（LLM-AIE）。将生成的内核代码传入编译器，若编译失败，则将错误信息反馈给模型。每个 LLM 最多可重试代码生成 10 次。虽然:mdnice-bold-italic[更多的编译尝试可能会进一步提升结果，但我们发现 10 次尝试后收益开始递减]。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/07.webp)

性能与迭代次数的关系上，我们进行了进一步的探究，如图 8 所示，发现在约 5 次迭代时出现边际效益递减的趋势。DeepSeek V3、Claude Sonnet 和 GPT-4.1 等模型对编译器反馈尤为敏感，其测试通过率出现了显著提升。

### 5.4 LLM 设置

在条件允许的情况下，我们会锁定随机种子（random seeds），以尽可能提高结果的可复现性。

- 对于 OpenAI 模型，我们将随机种子设置为 42。
- 所有模型的温度参数（temperature）均设为 0.0，top_p 设为 1.0。
- 对于推理模型（reasoning models），温度参数设为默认值 1.0，且系统提示词作为用户消息的一部分传入。

## 六、结果

我们在该数据集上评估了主流的前沿大语言模型（LLM），包括 OpenAI 最新版本的 GPT-4.1、Anthropic 的 Claude 3.7 Sonnet，同时也纳入了 Meta 的 Llama 系列等开源权重模型，以及 DeepSeek 等其他在代码生成任务中表现出色的开源模型（如 DeepSeek R1）。

### 6.1 开箱即用的大语言模型评估

我们**首先为模型设定基准：仅在提示词中加入 5.1 节所述的系统提示词，直接对模型进行测试**。基准测试结果汇总于图 4。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/08.webp)

### 6.2 功能正确性

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/02.webp)

如图 3 所示，功能正确性的评估方式是将编程后的 NPU 输出的核结果与 Python 行为模型生成的参考输出进行比较。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/09.webp)

我们通过最多五次重新编译来探究编译器反馈的有效性，结果汇总如表 1 所示。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/10.webp)

令人意外的是，GPT-4o Mini、Claude 3.5 Haiku 等小型模型在开箱即用的情况下，通过的测试数量反而多于 Claude Sonnet、GPT-4o 等更强大的模型。我们观察到的原因之一是：

::: block-1
- 小型模型默认倾向于生成标量 C++代码（即使用 for 循环的常规 C++代码，而非向量内在函数）；
- 而性能更强的模型最初可能会尝试生成向量化代码，但最终会像图 6b 所示那样生成存在“幻觉”（虚构函数）的代码。
- 在获得编译器反馈后，这些强模型通常会退而采用标量解决方案，从而通过更多测试。
:::

:mdnice-bold-italic[DeepSeek V3、Claude 3.7 Sonnet 和 GPT-4o 从多次编译尝试中获益显著，而其他模型的收益则较为有限]。尤其值得注意的是，DeepSeek V3 在几乎所有首次尝试中都会试图引入“adf.h”头文件——该头文件在评估框架所使用的开源工具集中并不存在，因此在首次重新编译后，其通过率骤升至 42%。

### 6.3 性能

生成核的效率通过以下方式计算：将 NPU 处理测试输入时用于 VPU 执行的周期数除以总周期数。尽管这种方法并非绝对完美，但它能很好地反映核对 VPU 的利用程度。未通过功能测试的核，其效率评估得分为 0%。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/11.webp)

各模型在所有核上的平均得分如图 5 所示。尽管这些得分看似较低，但需注意的是，当前最先进（SoTA）的开源核[22]的向量化比例通常在 10%-30%之间。

为引导模型生成向量化代码，我们在代码生成流程中引入了由向量化核组成的检索增强生成（RAG）模块。将包含公开可用 AIE 核的 RAG 引入提示词后，所有模型的 VPU 利用率均有所提升，其中 GPT-4.1 的向量化分数提升最为显著。:mdnice-bold-italic[奇怪的是，DeepSeek R1 是唯一在引入向量化示例后平均 VPU 利用率下降的模型] ——即便如此，该模型在不使用向量化示例的情况下，仍保持着最高的向量化分数。

> 其性能下降的原因在于：我们数据库中的部分核包含编译器特定的编译指示（pragma），而这些编译指示与 LLVM-AIE 所用的编译指示并不相同。

在获得编译器反馈后，DeepSeek V3 和 GPT-4.1 是生成向量化代码表现最佳的模型之一，尽管它们在功能正确性测试中的得分相当低。这一现象符合以下趋势：:mdnice-bold-italic[性能更强的模型在尝试编写高性能代码时更具“创新性”，但由于缺乏对 NPU 编程环境的深入了解，它们也更容易出现错误。]

#### DeepSeek R1 结果深入分析

> 本研究中一个意外发现是：当向 DeepSeek R1 模型提供向量化代码样本后，其向量处理单元（VPU）利用率反而下降。在所有受测模型中，这是唯一出现性能下降的模型。

观察该模型在无 RAG 情况下生成的解决方案（清单 3），并将其与有 RAG 情况下生成的解决方案（清单 4）进行对比，**我们发现该模型使用了编译器特定的编译指示（pragma）**。显然，**R1 模型精通如何使用编译指示进行底层优化（如循环展开）**，而我们 RAG 数据库中的示例代码是专为 Chess 编译器定制的——该编译器所使用的编译指示集与 LLVM-AIE（Peano）完全不同。

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/12.webp)

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/13.webp)

尽管 Peano 编译器在遇到 Chess 特定的编译指示时不会报错，但会在优化过程中忽略这些指示。**若未向该大语言模型展示这些示例（如清单 3 所示），模型会退回到标准 AIE API** ——这些 API 已封装了优化操作，无需关注底层内联函数（intrinsic）或编译指示。

**未来工作中解决此问题的方法是构建“编译器后端特定的 RAG”**。这些信息也可能通过系统提示词传递，并且随着更多 NPU 架构的发布（每种架构都有其独特的内联函数集和编译器特定优化路径），这种方式很可能成为必要手段。

### 6.4 生成失败原因分析

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/10.webp)

> :mdnice-bold-italic[如图 6a 所示，许多通过功能正确性测试的 LLM，最终生成的却是效率极低] 的解决方案——这些核未能充分利用 NPU 硬件。**而不正确的解决方案通常会因“幻觉”（如图 6b 所示）** 或**对向量 API 使用方式的误解** （如图 6c 所示，模型虽成功使用了 AIE API，但仍通过循环逐元素处理数据）而编译失败。

这些失败或者未充分利用 NPU 硬件的情况包括：

::: block-1
1. 逐元素循环（GPT-4o）——代码示例为标量循环处理 int8 类型数据；
2. 虚构不存在的 API（Llama-3.1 70B）——代码示例中包含虚构的 API 调用；
3. 了解向量概念及数据分割方式，但仍用标量循环处理（Claude 3.7 Sonnet）——代码示例虽定义向量大小，但内部仍逐元素处理 bfloat16 类型数据）
:::

不过，这些模型中似乎蕴含着一定的固有知识。例如，Claude 3.7 Sonnet（图 6c）能够理解“vector_size（向量大小）”的概念，并且知道应将输入缓冲区分割为 16 元素的向量块。**通过更先进的提示工程和更多高质量示例，这些模型有望生成高效的 NPU 核。**

## 七、局限性

本研究主要**针对 AMD NPU 中的 AIE 架构**。尽管我们设计的方法**可扩展到其他加速器系列**，并且在 AIE 系列推出支持最新内在函数和 API 改进的新设备时也可适配，**但仍存在以下局限性**：

::: block-1
- 首先，本文呈现的结果仅基于单一编译器后端（LLVM-AIE），而仅 AIE 架构就存在多种编译器选择。正如 DeepSeek R1 的结果所显示的，代码中的某些优化是编译器特定的——这在将方法:mdnice-bold-italic[推广到不同 NPU 架构和编程平台时，将构成一项有趣的挑战。]
- 其次，由于 NPU 是一类相对较新的设备，其编程模型不像其他加速器那样成熟，目前仍:mdnice-bold-italic[缺乏高质量的开源核来建立人类性能基准]。我们在提供了几个示例以对比结果，包含逐元素和卷积操作，但篇幅过长此处省略，需指出：**当前最先进的向量化比例不超过 30%，且该比例会因核的类型不同而有显著差异。**
- 最后，我们的评估:mdnice-bold-italic[虽涵盖了一系列主流 LLM，但仍有更多模型有待测试]。所有结果均基于贪心解码的单次生成；尽管“pass@k”评估（多次生成后选择最佳结果）可能会获得更高分数，但单次解码仍能有效反映当前 LLM 在 NPU 核生成任务中的能力。
:::

## 八、未来工作

> NPU 仍是一类新兴的加速器，而部分模型在为其编写代码方面已展现出令人印象深刻的能力——这仅仅是一个开始。

未来，代码生成智能体（code generating agents）等新技术有望进一步改进该基准的性能，这一前景令人期待。

:mdnice-bold-italic[尽管本文的研究聚焦于单一 NPU 架构，但该方法可轻松扩展到不同厂商、不同系列的 NPU]。Python 行为模型可与 PromptConstructor 类复用，以生成针对任意编程模型的数据集。我们计划采用相同的方法，将这项工作扩展到其他加速器系列。

## 九、结论

> 本文提出了 NPUEval——首个系统性评估 LLM 生成 NPU 核代码能力的基准。NPUEval 包含一个全面的评估框架（**提供周期精确的性能指标**）和一条参考代码生成流程。

我们的结果表明：:mdnice-bold-italic[大多数 LLM 能够轻松生成标量代码，但难以生成经过优化的向量化解决方案]。有趣的是，GPT-4o Mini、Claude 3.5 Haiku 等小型模型倾向于生成功能正确但未优化的代码，而 GPT-4o、Claude 3.7 Sonnet、DeepSeek R1 :mdnice-bold-italic[等更强模型则会尝试优化，却因存在“幻觉”（虚构内容）而导致功能得分更低。]

我们相信，NPUEval 为推进 LLM 驱动的加速器核生成研究提供了宝贵基础。期望该基准能成为衡量和改进 LLM 针对新兴硬件架构生成代码能力的行业标准。

## 参考文献

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/14.webp)

![文章插图](/articles/2025-09-08-npueval-amd-llm-npu-kernel-6a0f8e13/images/15.webp)
