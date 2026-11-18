> 人工智能，尤其是大型语言模型（LLMs），已经深刻地改变了我们日常的工作和交流方式。然而，由于计算密集型需求、繁重的内存访问、多样的操作符类型以及编译难题，将大型语言模型部署在资源受限的边缘设备（例如机器人）上仍然是一个挑战。

在本文中，我们提出了 **EdgeLLM 加速器** 来解决上述问题。

::: block-1
- 首先，**【计算效率】** 方面，我们设计了**混合精度处理单元阵列** 以及**分组 systolic 架构**，能够高效**支持多头注意力（MHA）块的 FP16∗FP16 计算** 和**前馈神经网络（FFN）层的 FP16∗INT4 计算**。
- 我们对**【对数级结构化权重稀疏性】** 进行了**特定优化**，以进一步提高效率。
- 其次，**为了解决【编译和部署】** 问题，我们分析了大型语言模型中的所有操作符（算子），并**开发** 了一种**通用的数据并行方案**，使得**所有输入和输出特征保持相同的数据形状**，从而**无需任何数据重排** 即可**处理不同的操作符**。
- 然后，我们**提出了一个【端到端的编译器】**，将**整个大型语言模型映射到 CPU-FPGA 异构系统** （AMD Xilinx VCU128 FPGA）。
:::

该加速器在**吞吐量上比商用 GPU** （NVIDIA A100-SXM4-80G）**高出 1.91 倍，能效高出 7.55 倍**。与最先进的 FPGA 加速器 FlightLLM 相比，在 HBM 带宽利用率、能效和大型语言模型吞吐量方面，性能提升了 10% 到 24%。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/01.webp)

- https://arxiv.org/abs/2407.21325v2

## 本文目录

- 一、引言
  - 算法随时代的变革：从 CNN、RNN、Transformer 到 LLM
  - GPU 架构限制与边缘 AI
  - 大预言模型给硬件的挑战
  - 我们的贡献
- 二、背景和相关工作
  - A. Transformer 和大型语言模型
  - B. 支持 Transformer 和大型语言模型的 AI 芯片
- 三、硬件设计（HARDWARE DESIGN）
  - A. 总体架构（Overall Architecture）
  - B. 混合精度计算处理单元设计（Design of Mix-Precision Computing PEs）
  - C. 进一步优化：对数级稀疏性（Further Optimization With Log-Scale Sparsity）
- 四、软件设计（SOFTWARE DESIGN）
  - A. 基于统一数据格式的大型语言模型运算符图（Unified Data Format Based LLM Operator Graph）
  - B. 软件架构和大型语言模型推理系统（Software Architecture and LLM Inference Systems）
- 五、实验结果（EXPERIMENTAL RESULTS）
  - A. 实验设置（Experiment Setup）
  - B. 硬件效率分析（Analysis on the Hardware Efficiency）
- 六、结论（CONCLUSION）
- 参考文献

## 一、引言

> 在过去的十年中，人工智能引起了广泛关注和全球关注。这主要与三个关键进展有关。

- 第一，计算能力的指数级增长，这主要归功于 GPU；
- 第二，AI 算法的演变，特别是用于计算机视觉（CV）的卷积神经网络和用于自然语言处理（NLP）的循环神经网络；
- 第三，AI 大数据的发展，这是 AI 模型训练的基础支持，推动了它们的精度和效率。

这些因素共同推动了 AI 的变革时代。

### 算法随时代的变革：从 CNN、RNN、Transformer 到 LLM

自 2017 年以来，Transformer 算法在人工智能领域迅速成为主导力量，无论是在 NLP 还是 CV 研究领域。通过引入自注意力机制，Transformer 改变了传统的序列建模方法，不再依赖卷积神经网络（CNN）或循环神经网络（RNN）进行特征提取和序列处理。这一创新不仅增强了模型的并行化能力，使大规模模型的训练成为可能，还显著提高了模型性能。

在 Transformer 取得成功之后，研究人员开始探索更大的模型，这些模型通常拥有数十亿甚至更多的参数，被称为“大型语言模型”。例如，OpenAI 的 GPT（生成预训练 Transformer）系列模型展现了理解和生成高质量人类语言的能力。如今，大型语言模型开始与其他模态（如视觉和音频）集成，显示出在跨域任务中的出色性能。这种交互能力使 AI 助手、聊天机器人和虚拟代理更加人性化和高效。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/03.webp)

### GPU 架构限制与边缘 AI

总之，大型语言模型不仅增强了 AI 的语言处理能力，还推动了更广泛的 AI 技术进步，为教育、娱乐、医疗和商业等行业带来了革命性的变化。然而，**当代 AI 加速器架构的主流，特别是针对大型语言模型需求的架构，主要依赖于 GPU 范式**。尽管这种传统设计功能强大，但其固有限制使其在边缘设备上的部署不够理想。

GPU 架构的限制，如高功耗和庞大的计算开销，给在网络边缘实施 AI 能力的实际性和效率带来了重大挑战。尽管 GPU 具有高灵活性和特定任务的高性能能力，**但作为一种异构计算的关键组成部分，FPGA 却鲜有提及**。**基于 FPGA 的大型语言模型** 实现对于大幅**提高计算效率** 和性能具**有相当大的潜力**。

### 大预言模型给硬件的挑战

然而，**几个关键挑战** 仍有待有效解决。

#### 挑战 1：计算与访存

大型语言模型的**第一个挑战是计算和内存访问的繁重负担**。大型语言模型通常在矩阵 - 矩阵乘法（MatMUL）或向量 - 矩阵乘法（VMM）中包含数十亿个参数。我们通常将一个模型称为 LLM-6B 或 LLM-7B，其中 6B 或 7B 表示矩阵乘法中的权重参数数量。如此庞大的参数数量转化为巨大的内存需求，使得将这些模型放入标准硬件的内存中变得困难。**为了解决这一问题，已经开发了几种策略，如模型剪枝和量化**。现有解决方案通常将这些权重参数量化为 INT4 格式。然而，为了确保计算精度，模型中的激活函数仍保持 FP16 格式。因此，系统需要在 FFN（前馈神经网络）层中使用 FP16∗INT4 格式的矩阵乘法。此外，尽管稀疏表示技术在理论上具有优势，但在行业中尚未广泛采用。因此，针对稀疏和 INT4 量化模型的独特需求的专用硬件组件存在明确的机会。

#### 挑战 2：混合精度

**另一方面，在 MHA（多头注意力）块中，KVcache 是作为激活数据动态生成的**，因此**涉及 KVcache 的矩阵乘法需要 FP16∗FP16 格式**。因此，非常需要为 LLMs 的加速**设计混合精度计算单元**。

#### 挑战 3：模型图结构的编译优化

此外，在大型语言模型中，定义计算流程的操作符图可能极为复杂，包含数百乃至数千个以复杂方式相互关联的操作符。确保**从一个操作符到另一个操作符的过渡** 既**无缝又高效成为一个重大挑战**。

传统编译器旨在优化顺序执行的代码，但深度学习模型的本质需要不同的方法。这些**模型的数据流图呈现出并行处理和流水线的机会**，而传统编译器可能无法充分利用这些机会。

因此，专门用于深度学习的编译器和运行时系统应运而生，以应对这些独特的需求。这些编译器会分析操作符图，以识别优化的机会，例如操作符融合，将相邻的操作符组合成一个更高效的操作。

#### 挑战 4：多层次内存结构的管理

另一个关键方面是内存访问模式的管理。**高效地处理数据在不同内存层次之间的移动**，可以显著影响性能。总之，大型模型的编译系统必须足够复杂，以处理操作符图的复杂性，优化并行处理，并有效管理内存访问。

### 我们的贡献

在这项工作中，我们开发了一种**用于大型语言模型的高效 CPU-FPGA 异构加速系统**。其中，FPGA 执行核心计算操作符，CPU 执行动态编译过程。在操作效率方面，我们的创新设计提案在性能上**超越了传统的 GPU 芯片和最先进的 FPGA 系统**。这种**卓越的性能是通过优化的硬件架构、高效的数据处理机制以及先进的动态编译方案的结合来实现的。**

主要贡献如下：

::: block-1
1. 我们**分析了前馈神经网络（FFN）和多头注意力（MHA）的计算需求**，并**提出了高效的混合精度计算单元** 及**分组 systolic 架构**。此外，我们提出了对数级结构化稀疏性以及块级量化方法，以平衡硬件效率和算法准确性。
2. 我们**分析了大型语言模型的编译需求**，并为 AI 算法中的**所有操作符和所有高维张量结构设计了统一的通用数据格式**。这使系统能够快速执行操作符，无需进行任何数据重排。
3. 我们**开发了端到端的编译方案**，其中针对**不同的输入标记长度使用动态编译**，并采用指令流水线策略以减少延迟。该方案可以动态编译所有的操作符，并将整个模型映射到 CPU-FPGA 异构系统上。
4. 最后，整个设计**已成功部署到 AMD Xilinx VCU128 FPGA** 上。我们的结果在**吞吐量上比 GPU 高出 1.91 倍，在能效上高出 7.55 倍**，并且比最先进的 FlightLLM FPGA 加速器的性能高出 10 到 24%。
:::

## 二、背景和相关工作

### A. Transformer 和大型语言模型

> Transformer 模型由 2017 年的著名论文《Attention is All You Need》首次提出，通过采用自注意力（self-attention）机制，突破了传统循环神经网络（RNN）和卷积神经网络（CNN）在处理序列数据时的局限性。

这种机制使得模型在处理序列中的任何位置时能够考虑整个序列的信息，从而在理论上实现了更优越的并行处理能力和捕捉长距离依赖关系的能力。

2018 年，OpenAI 推出的 GPT（生成预训练 Transformer）系列模型展示了语言模型在生成和理解方面的巨大潜力。尤其是 GPT-3，拥有 1750 亿（175 billion）参数，展现了理解和生成高质量人类语言的能力。

2019 年，Google 的一个研究团队引入了文本到文本转移 Transformer（T5），这是一个统一框架，将几乎所有的自然语言处理（NLP）任务重新表述为文本到文本的转换问题。通过预训练和微调，T5 在多种 NLP 任务中表现出色，进一步验证了 Transformer 架构的灵活性和强大的适应能力。

随着研究的进展，**Transformer 架构开始应用于多模态任务**，如图像描述生成和视频理解，实现跨模态信息的有效整合。著名的模型包括 GPT 系列、ChatGLM、PaLM、Sora 等。这些大型模型在各自的领域中表现出色，从简单的文本生成到复杂的多模态理解和生成，不断推动人工智能技术的边界。

### B. 支持 Transformer 和大型语言模型的 AI 芯片

> 支持大型语言模型运行的 AI 芯片通常需要高计算能力、高内存带宽和高效的并行处理架构。

::: block-1
到目前为止，大多数当前的 AI 加速器系统，**特别是那些针对大型语言模型的系统，几乎都基于 CPU-GPU 架构**，如 NVIDIA 的 A100 GPU、Google 的 TPU（张量处理单元）、Intel 的 Habana Gaudi AI 训练处理器、Graphcore 的智能处理单元（IPU）和 AMD 的 Instinct MI200 系列 GPU。这些 AI 芯片通常提供丰富的计算资源（>100 万亿次操作每秒，TOP/s）、高内存带宽（>1 太比特每秒，TB/s）和高功耗（>200 瓦）以支持大规模模型的训练和推理。
:::

为了促进 AI 在社会和工业各个领域的广泛应用，一个关键的需求出现：即必要性，使得 **AI 能够无缝部署在各种边缘硬件上，超越云端接入的限制**。FPGA 实现提供了设计灵活性，允许针对特定应用对模型进行定制和优化，因此特别适合边缘应用。目前，该领域已经取得了几项进展。例如，有工作将一个 1720 亿次浮点运算每秒（GFLOP）的神经机器翻译模型（采用混合精度表示）映射到单个 FPGA 板上，这是首次在 FPGA 上实现实际的端到端神经机器翻译（NMT）模型。但该工作在 100MHz 的低频率下运行，显示出相对较低的吞吐量。

另一个例子是 **ViA，这是一种基于 FPGA 的新型视觉 Transformer 加速器架构，能够高效执行 Transformer 应用**。在 FP16 数据类型下，ViA 在峰值时可实现约 309.6 十亿次运算每秒（GOP/s）。

还有研究提出了一种**仅支持整数运算的 Transformer 加速器**，其平均吞吐量高达 762.7 GOPs，与之前的最先进加速器相比显示出显著的加速性能提升。然而，整数模型不适合大型语言模型。

**FlightLLM 是第一个基于 FPGA 的高性能大型语言模型加速器**。通过引入可配置的稀疏数字信号处理器（DSP）链以支持各种稀疏模式，以及始终在线的芯片解码方案并支持混合精度，**FlightLLM 在 Llama2-7B 模型上实现了比 NVIDIA V100S GPU 高 6.0 倍的能效和 1.8 倍的成本效率**，并且在解码过程中比 NVIDIA A100 GPU 的吞吐量高出 1.2 倍，显示出 FPGA 基大型语言模型加速系统的巨大潜力。然而，其带宽利用率仅为 65.9%，表明在数据传输和计算之间存在相对较长的流水线气泡（pipeline bubbles），可以进一步改进。

## 三、硬件设计（HARDWARE DESIGN）

在本节中，我们将概述所提出的加速器的基本架构，并讨论混合精度计算单元以及稀疏策略。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/04.webp)

### A. 总体架构（Overall Architecture）

> **图 2 提供了所提出的 CPU-FPGA 异构加速器的基本架构图**。整个系统包含四个元素：中央处理单元（CPU）、高带宽存储器（HBM）、DDR 内存和大型语言模型（LLM）加速器。

**CPU 作为系统的大脑，通过外设组件互连高速总线（PCIe）与其他组件接口**。它通过 AXI-full 协议对 DDR 和 HBM 拥有直接访问权限，能够进行读写操作。此外，它还利用 AXI lite 协议与加速器 IP 内部的寄存器数组进行交互，从而控制加速器的操作动态。

**在加速器 IP 内部，精心设计了各种 LLM 操作符（算子）**，包括 MatMUL（矩阵乘法）操作符、MHA（多头注意力）操作符、LayerNorm（层归一化）操作符等。其中，MatMUL 和 MHA 操作符由于对权重参数和 KVcache 的极高内存访问速度要求，与 HBM 相连（图 2 中的红色区域）。

在加速 MatMUL 操作符时，**所有权重参数将被预处理并存储在 HBM 中，而动态生成的激活数据将被写回 DDR**。对于 **KVcache，构建了特殊的写直接内存访问（DMA）路径**，将在线生成的 KVcache 传输到 HBM（图 2 中的红色线条）。因此，MHA 操作符也可以充分利用高带宽存储器。其他操作符，如 LayerNorm、RMSNorm（均方根归一化）、Rotary Embedding（旋转位置嵌入）和非线性激活操作符，通过不同的定制 DMA 策略与 DDR 内存相连。所有操作符都专门设计了定制的 DMA 模块，用于在片外 DDR 和片上块随机访问存储器（BRAM）之间获取激活数据。同时，只有 MatMUL 和 MHA 操作符连接到 HBM 以实现高速权重访问。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/05.webp)

> **图 3 展示了在加速矩阵乘法（MatMUL）操作符时的性能模型** （roofline model），这是一种分析工具，用于对比内存受限和计算受限领域的边界。

- 操作强度（operational intensity）以每字节（TFLOP/Bytes）的浮点运算次数为单位，位于横轴
- 而纵轴则表示可实现的性能，以每秒浮点运算次数（TFLOP/s）为单位。

**屋顶线模型（Roofline Model）能够清晰地划分从内存受限到计算受限场景的转变**。这种视觉对比提供了深刻见解，展示了不同的时钟策略如何影响 MatMUL 的操作强度和可实现性能，从而阐明了内存带宽和计算吞吐量之间的复杂相互作用。**为了充分利用 HBM 的高速通信能力，我们设计了一个与屋顶线模型完美匹配的计算阵列。**

在使用 KVcache 框架的 LLM 操作中，MatMUL 任务可以简化为向量 - 矩阵乘法（VMM）。在此任务中，HBM 的带宽为每周期 8192 位（32 个 AXI 端口，每个端口每周期传输 256 位）。为了充分利用此高带宽，我们设置系统每周期消耗相应的 8192 位权重参数，**这意味着在 INT4 量化方案下的系统计算并行度为 2048**。为了进一步提高计算强度，我们将计算并行度翻倍，因此 HBM-AXI 通信模块和 DMA 模块的运行频率应比计算模块高出一倍。

具体来说，在基于 FP16 INT4 的 FFN 层中，我们将计算并行度设置为 4096（FFN 权重消耗 40964=16384 位/周期），而在基于 FP16 FP16 的 MHA 块中，计算并行度为 1024（MHA KVcache 消耗 102416=16384 位/周期），同时 HBM 提供 16384 位/周期。**在这种情况下，系统既不会受到内存限制，也不会受到计算限制**。

### B. 混合精度计算处理单元设计（Design of Mix-Precision Computing PEs）

> 尽管 LLM 模型包含大量的操作符，但**超过 95% 的计算操作仍然是矩阵乘法（MatMUL）或向量 - 矩阵乘法（VMM）**。为了支持这些计算，已经提出了各种 systolic（即流水线）阵列。

其中，**Google 的 TPU（张量处理单元）是最著名的 systolic（即流水线）阵列**，它以**细粒度流水线的方式执行矩阵乘法**。然而，它需要大量的寄存器来存储临时数据，导致动态功耗和电路面积较大。

为了**提高效率，我们开发了分组向量 systolic（即流水线）计算阵列** （G-VSA）。与 TPU 风格或广播风格的 systolic（即流水线）阵列不同，**G-VSA 以逐行的方式将输入特征和权重传输到 PE（处理单元）阵列中，从而大大降低了功耗和面积**。

虽然现有的 LLM 通常将权重参数量化为 INT4 格式，但模型中的激活函数仍保持 FP16 格式以确保计算精度。因此，系统需要同时支持 FFN 层的 FP16 INT4 格式矩阵乘法和 MHA 块的 FP16 FP16 格式矩阵乘法。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/06.webp)

**混合精度计算单元（processing element，PE）的结构如图 4 所示**，其中 `Tin = 128` 表示向量长度。对于 FFN 层，该模块可以支持`Tin = 128 个 FP16 特征数据 \* INT4 权重数据`。而对于 MHA 块，我们设置并行度为`Tin / 4`，因此该模块需要每时钟周期处理`Tin / 4 = 32 个 FP16 激活数据`和`Tin / 4 = 32 个 FP16 KVcache 数据`。由于 INT4 权重数据和 FP16 KVcache 都来自 HBM，因此在上述两种情况下，计算单元需要相同的带宽。

此外，**我们采用了时间分割复用方法**，并在每次 FP16*INT4 计算中共享 DSP（数字信号处理器）。通过这种设计，系统既不会受到内存限制，也不会受到计算限制。最后，为了与块级 INT4 量化算法的计算要求对齐，该单元还支持缩放值（Scale value，以 FP16 数据格式表示）的乘法调整，以调整最终输出。

**计算单元中的详细数据流如图 4（b）所示**。整个**计算过程可分为四个阶段**：

- **Stage-0**：处理所有输入。通常，FP16 被分为符号位（S）、指数位（E）和尾数位（M），而 INT4 数据被分为符号位（S）和尾数位（M）。在 MODE-0 情况下，每个 FP16 KVcache 数据由四个相邻的 INT4 数据组成。
- **Stage-1**：对每个 FP16 特征（DAT）和 INT4 权重（WT）的符号位进行异或（XOR）操作。同时，指数比较模块分析 FP16 数据的指数位，识别所有 FP16 输入中的最大值。该模块计算每个输入与最大值之间的指数距离。与此同时，FP16 和 INT4 的尾数位被发送到乘法器阵列进行乘法操作。在乘法器中，刻意保留了所有尾数部分，以显著保留计算结果的精度，确保算术过程中没有分数细节丢失。
- **Stage-2**：将 Stage-1 中获得的指数距离和乘法结果输入到对齐移位器模块中，以对齐小数点。然后，移位后的尾数结果被发送到加法树模块进行累加。需要注意的是，初步的移位操作会增加输入数据的位宽。为了避免过度消耗资源，我们将加法树的位宽指定为 19 位，在资源利用和计算精度之间取得平衡。
- **Stage-3**：将前一层得到的乘法结果和最大指数通过前导零预测（LZA，Leading Zero Anticipate）模块和指数调整模块转换为规范化的 FP16 数据。然后，它们与初始的缩放值相乘，使用浮点乘法得到最终结果。最后，结果被输入到积分模块中，调整为 FP16 格式输出。

为了验证我们所提方案的效率，我们设计了两组额外的对照实验。在这些实验中，采用了标准的成对加法基加法树；但中间计算的精度有所不同。在基线 1 情况下，我们配置中间临时结果以 FP16 格式表示。而在基线 2 情况下，使用定制的 FP20 数据格式（S1-E6-M13）来避免计算过程中的溢出，同时由于尾数位宽较大，保持高精度。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/07.webp)

在**表一中列出了不同计算单元设计方法的性能比较。与两种基线方案相比，我们所提出的混合精度模块在效率和计算错误率方面均显著更高**。在进行 100,000 次随机输入测试后，我们的方法在 FP16 INT4 模式下仅表现出 0.047% 的错误率，在 FP16 FP16 模式下仅表现出 0.0044% 的错误率。而基线方法则产生了较大的错误率。此外，**我们的设计在功耗、性能和面积（PPA）方面表现出最高的综合性能**。

我们的设计总面积仅为 71,664 平方微米，而基线 1（采用 FP16 基加法树）为 80,675 + 26,762 = 107,437 平方微米，基线 2（采用 FP20 基加法树）为 140,677 平方微米。与基线 1 相比，面积减少了 33.2%，与基线 2 相比，减少了 49.1%。其次，我们的模块在时钟频率方面表现优于竞争对手，在 28 纳米 CMOS 上达到了 1.11 吉赫兹的峰值，足以满足实际应用需求。最后，在功耗方面，基线 1 由于其在 FP16 操作中内在的计算复杂度较低，自然消耗的功率最少（35.03 + 14.66 = 49.69 毫瓦）。然而，**我们的方案仅显示出 2% 的功耗增加** （FP16 INT4 为 40.34 毫瓦，FP16 FP16 为 10.39 毫瓦），显示出优异的效率。与基线 2 相比，**我们的方法显示出更少的功耗和电路面积。这些属性证实了我们的方法在 PPA（功耗、性能和面积）方面具有最佳性能**。

### C. 进一步优化：对数级稀疏性（Further Optimization With Log-Scale Sparsity）

> 大型语言模型通常包含数十亿个参数。如此庞大的规模在存储、内存和计算效率方面带来了显著的挑战，尤其是在将这些模型部署在资源有限的设备上或尝试扩展分布式训练过程时。

**INT4 量化通过将权重表示为 4 位整数而非典型的 32 位或 16 位浮点数，提供了多项优势**：更高的内存效率、更快的计算速度和更好的带宽优化，因此在 LLM 的推理中得到了广泛应用。此外，结构化稀疏性提供了显著的优势。通过跳过零权重，它可以加速计算，减少大型模型的存储需求，节省能源，并使复杂模型能够在资源有限的设备上运行，同时保持准确性。

**基于密度边界块（DBB）的结构化稀疏性已被证明是一种有效的方法**，但哪种结构化稀疏性是最好的仍然是一个开放的问题。以往工作主要关注线性分布的稀疏性（即 1/8、2/8、3/8...），但硬件利用率通常较低，导致效率低下。基于 GPU 的固定稀疏性（2/4 权重稀疏性）显示出计算效率提高 2 倍，但其内存访问开销几乎没有减少。

**为了克服上述挑战，本文提出了对数级结构化稀疏性** （即 1/2、1/4、1/8...）加速器。**借助时间展开的微架构，硬件利用率可以在广泛的稀疏性范围内达到 100%，更重要的是，内存访问开销可以大幅降低**。我们**将结构化稀疏性应用于矩阵乘法操作符的权重参数中**。

在矩阵乘法执行过程中，首先将更广泛的激活数据输入到系统中。然后，根据权重参数中的掩模选择性地挑选出必要的激活数据。最后，将选定的激活数据和权重数据共同输入到计算阵列中进行处理。我们为特征数据设计了一种高效的稀疏 DMA。对于权重数据，我们采用了块量化和混合稀疏性方案，其中 128 个相邻参数对称量化并共享相同的量化尺度参数。

对数级混合稀疏性定义为：每组八个相邻的数据块中必须至少包含 N 个零，而其他则是非零元素。例如，当稀疏性为 75% 时，意味着每组八个相邻的数据块中必须至少包含 6 个零和最多 2 个非零元素。基于这些分析，参数可以分为三类：量化尺度、掩模和权重本身，所有这些都将存储在 HBM 中。由于该 FPGA 中的单个 HBM AXI 端口具有 256 位的位宽，每个量化尺度是 FP16 数据类型，组大小将是 256/16 * 128 = 2048 个输入通道组（CHin）。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/08.webp)

**图 5 展示了经典权重数据形状**。首先，我们将属于同一 CHout 的所有权重参数打包在一起。例如，所有具有 CHout 索引 0、32、64 的数据将依次放置在 HBM-AXI-port00 中，所有具有 CHout 索引 1、33、65 的数据将依次放置在 HBMAXI-port01 中，依此类推。然后，对于每个权重包，将其分为 CHin/2048 个部分，任何少于 2048 个通道的部分将被填充到 2048。每个部分按比例、掩模和 wt 的顺序排列。

- 案例 1：当模型密集时，2048 个 CHin 组包含 2048 * 4 = 8192 位的 INT4 权重，并包含（2048/128）* 16 = 256 位的 FP16 比例。因此，比例、掩模和 wt 的总比例为 256:0:8192 = 1:0:32。
- 案例 2：当模型的层稀疏性为 50% 时，2048 个 CHin 组包含 2048 * 4 * 50% = 4096 位的 INT4 权重。这里我们使用单热编码方案表示非零掩模，使得 2048 个 CHin 组包含 2048 位的掩模。比例、掩模和 wt 的总比例为 256:2048:4096 = 1:8:16。需要注意的是，如果使用块内地址方案（这种编码方案直接指示每个权重的偏移地址），每个 2048 个 CHin 组将包含 4096 位的掩模，这在这种情况下效率不高。
- 案例 3：当模型的层稀疏性为 75% 时，单热编码掩模需要 2048 位，而权重数据将是 2048 * 25% * 4 = 2048 位。因此，比例、掩模和 wt 的总比例为 256:2048:2048 = 1:8:8。如果使用块内地址方法，掩模将需要 1536 位，因此比例、掩模和 wt 的总比例为 256:1536:2048 = 1:6:8。
- 案例 4：当模型的层稀疏性为 87.5% 时，单热编码掩模需要 2048 位，而权重数据将是 2048 * 12.5% * 4 = 1024 位。因此，比例、掩模和 wt 的总比例为 256:2048:1024 = 1:8:4。当稀疏性非常高时，块内地址方法的成本更为高效。如图 5 所示，2048 个 CHin 组仅包含 1024 位的掩模。因此，总成本为 2304 位，性能提升比率达到 3.66 倍，在 87.5% 的稀疏性下。

根据上述分析，**有效权重位宽分别为 4.125 位、3.125 位、1.875 位和 1.125 位**，而**在不同的稀疏性下性能提升比率为 1.32 倍、2.2 倍、3.67 倍。总之，结构化对数级稀疏性提供了多项优势**：更高的内存效率、更快的计算速度和更好的带宽优化。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/09.webp)

**表 II 显示了在不同稀疏策略下 GLM-6B 算法的性能**。经过**混合稀疏性简化后，权重参数** （以及位置编码参数）的**总量可以显著减少，同时算法的准确性仍保持在较高水平**。

与以往工作相比，**我们的主要创新在于硬件支持混合稀疏性**，允许**为不同操作符层选择不同的稀疏性水平**。例如，**QKVO 层可以为 50% 的稀疏性，而 h-to4h 层可以为 75% 的稀疏性等**。最关键的是，由于我们的设计中 M 和 N 都是 2 的幂，硬件计算单元在不同的稀疏性下始终可以实现 100% 的计算效率。

此外，我们的稀疏块更大。例如，对于相同的 50% 稀疏性，GPU 使用 2:4，而我们可以使用 4:8、8:16 或 32:64。尽管这增加了硬件成本，但可以在算法层面实现更好的性能。此外，我们提出了混合编码方案来记录非零权重的位置，其中在低稀疏性下使用单热编码方案，在高稀疏性下使用块内地址方案，以进一步提高效率。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/10.webp)

## 四、软件设计（SOFTWARE DESIGN）

在本节中，我们将介绍旨在优化执行硬件操作符的编译策略。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/11.webp)

### A. 基于统一数据格式的大型语言模型运算符图（Unified Data Format Based LLM Operator Graph）

> 当执行人工智能模型时，系统的编译过程依赖于运算符图以实现连贯性和效率。**图 6 展示了 ChatGLM-6B 的一个示例，其中一个块将被融合成 17 个硬件步骤**。关键特征是每个运算符的输入和输出保持一致的数据结构，以实现最佳的执行速度。

这种一致性允许从一个运算符平滑地过渡到下一个运算符。为了保持一致的数据结构，**我们为大型语言模型运算符开发了一种通用的数据并行方案。该方案是通用的，可以适应任何类型的 AI 算法**。

文本类型的输入/输出激活数据本质上具有二维结构（tokens， channels）（注：这里的“tokens”指的是文本被分割后的最小单元，比如单词或子词；而“channels”可以理解为不同的特征维度）。这些数据可以轻松转换为三维张量表示，即 [CH/Tout, token, Tout]。同样，图像类型的输入/输出激活数据原本以高度（H）、宽度（W）和通道（CH）为特征，可以转换为四维张量表示 [CH/Tout, H, W, Tout]。**可以看出，文本类型和图像类型数据共享相同的张量化方案。此外，在多头注意力机制引入额外的头维度或批次维度的情况下，数据维度可以类似地扩展为 [head 或 Batch, CH/Tout, H, W, Tout]，确保框架保持通用性**。

以 ChatGLM 模型为例。输入的形状为（token，4096），Wq 的权重矩阵为（4096，4096），Wk 和 Wv 的权重矩阵为（4096，256）。经过矩阵乘法后，我们将得到 Query-Key-Value 三个矩阵，形状分别为（token，4096）、（token，256）、（token，256）。Query-Key-Value 矩阵将进一步嵌入并根据头数量进行重塑，生成三维矩阵（head，token，4096/head）。接下来，通过 Q*K^T 计算上下文信息，其中 Q 是 Query 矩阵，K^T 是 Key 矩阵的转置。然后执行注意力矩阵的 Softmax 操作，最后乘以 Value 矩阵。需要注意的是，K 和 V 是由之前的 MatMUL 运算生成的，分别为 K = input _ Wk 和 V = input _ Wv。因此，KT 和 V 在推理过程中不能预先处理。

此外，转置操作破坏了 V 矩阵的数据访问模式，阻碍了细粒度流水线所需的连续性，需要粗粒度流水线。值得注意的是，本工作中引入的通用数据结构非常适合解决此类问题。由于所有激活数据都可以形成 [CH/Tout, token, Tout] 的形状，因此在地址块中自然存在两个维度的数据，即 [token, Tout]。利用这种数据结构，我们可以设计分段连续执行的转置操作，从而无需更改数据格式。

**另一个优势在于最大化利用 AXI 突发传输的效率**，AXI 突发传输指的是在单次事务中传输多个连续数据，而不是启动新的事务。在我们的设计中，AXI 写模式和 AXI 读模式的数据宽度固定为 Tout*16，这正好与 [Batch, CH/Tout, H, W, Tout] 中的最小数据包相同。因此，AXI 突发传输中的增量地址将正好是宽度维度或 token 维度中的张量。这允许高效顺序数据访问，特别适用于读取或写入内存中的数据块。

### B. 软件架构和大型语言模型推理系统（Software Architecture and LLM Inference Systems）

> 除了硬件运算符设计外，本工作还关注大型语言模型的软件部署和优化。图 8 展示了在 VCU128 FPGA 平台上大型语言模型的编译和部署过程。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/12.webp)

经过稀疏化和量化后，大型语言模型将被导入编译器，**编译器将执行特殊优化，包括动态控制和 KV-cache，并根据加速器的特性生成指令、预处理权重和运行时控制代码，以促进模型部署**。根据编译后的模型，我们**设计了一套基于局域网（LAN）的大型语言模型推理框架**。该框架使用 FPGA 和部署的大型语言模型作为服务器端，负责大型语言模型的实际推理。Python 用作客户端，对 token id 进行编码和解码，并与用户进行直接交互。

此外，还需要精心准备运算符寄存器的配置数据，确保所有组件在即将到来的测试阶段正确初始化。在大型语言模型推理阶段，首先，为了简化所有运算符的动态地址，编译器引入了 MAX Token 属性，该属性来自 RTL 宏定义。它在内存空间大小和动态数据地址的计算中替代表面 token，使地址静态化，减少运行时的额外计算。

其次，硬件指令还需要适应动态 token。当编译器评估对应运算符的硬件指令时，token 作为变量参与计算，参数以有向无环图（DAG）形式的数字表达式记录。如果可以直接计算该参数，编译器将返回该指令的结果；否则，它将以简化的代码表达式形式嵌入到运行时代码中进行实时更新。

在这种方法中，硬件指令所需的空间非常小，使得 KV-cache 的推理空间非常充足。此外，由于最后一个注意力结构之后的所有运算符的有效数据仅在最后一个 token 的向量上，编译器对此进行了额外优化，并根据 token 参数提供实际数据偏移量。在后续计算中移除最后一个 token 的数据，以最小化计算量。最后，我们的设计提供了预配置的寄存器模式和辅助路径，以实现指令流水线的模式操作。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/13.webp)

**编译器可以在推理阶段提供延迟隐藏的优化**。加速器的辅助模式可以直接从片上 DDR 通过 AXI 总线将多个序列化运算符指令编码到缓冲区。在加速器的主机控制中，只需要将给定序列化指令的配置信息写入寄存器，例如地址信息、有效运算符的数量等。然后加速器将完成序列化指令的快速启动和连续操作。在加速器读取指令后，主机有额外的时间在加速器计算完成前进行其他计算。如图 9 所示，动态控制指令更新所需的延迟可以隐藏在等待加速器计算完成的时间中。

在实际推理中，我们只需要在第一次模型推理前更新完整指令，随后的指令更新将隐藏在最后一次模型推理中。

## 五、实验结果（EXPERIMENTAL RESULTS）

在本节中，我们展示了**使用 AMD Xilinx VCU128 板加速大型语言模型（LLM）的实验结果**。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/14.webp)

### A. 实验设置（Experiment Setup）

> 所提出的加速器已使用 Vivado 2023.1 在 AMD Xilinx VCU128 板上进行了综合和实现。该板拥有 1303K 逻辑单元（LUT）、2607K 触发器（FF）、9024 个 DSP 模块和 2016 个片上块存储器（Block RAM）。此外，FPGA 配备了 8GB 的高带宽存储器（HBM），带宽为 460GB/s，非常适合加速 GB 级大型语言模型。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/15.webp)

图 10 展示了系统的实验设置。主机计算机（CPU）通过外设组件互连高速总线（PCIe）控制 DDR 和 HBM，并利用 AXI lite 协议与加速器 IP 内部的寄存器数组进行交互，从而控制加速器的操作动态。矩阵乘法操作符由于需要处理大量参数，以 280MHz 的高频连接到 HBM，而其他部分和操作符则以 140MHz 工作。由于时序优化，操作频率可以进一步提高。

**图 10 的子段展示了 GLM-6B 加速的实验结果**，其中**考虑了不同的稀疏策略**。对于官方提供的**原始模型**，我们的芯片计算**几乎没有精度损失**。经过**稀疏优化后，精度略有下降，但模型速度提高了约 63%。**

此外，我们还测量了 Qwen-7B LLM 模型。Qwen 的解码速度（42.5 token/s 至 69.4 token/s）比 GLM 慢，这不仅是因为其 FFN 参数较少，还因为 MHA 块中高度共享的权重头导致了更长的延迟。

### B. 硬件效率分析（Analysis on the Hardware Efficiency）

> HBM 带宽利用率是 AI 芯片的一个关键点。效率可以通过实际操作时间和理想操作时间的比值来计算。以 GLM 模型为例，输入的形状为（token，4096），Wq 的权重矩阵为（4096，4096）。

在解码阶段，输入 token 始终为 1，因此理想操作时间将为 409640964bit / (8192 bit/cycle) * 3.571ns = 29.25µs。同时，实际操作时间可以直接测量为 38.5µs，因此 HBM 带宽利用率为 29.25µs/38.5µs = 75.97%。同样，我们可以计算每个 MatMUL 层的利用率，并最终观察到它们都介于 70% 和 80% 之间，平均约为 75%。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/16.webp)

**图 11 展示了 GLM-6B 模型加速的性能**。当解码 token 少于 512 时，解码速度保持相对稳定（约 90 token/s）。

图 11（b）显示了详细的延迟分解。整个模型可分为三个部分，即多头注意力（MHA）、前馈神经网络（FFN）和其他部分。FFN 的运行时间与解码长度无关，但 MHA 的运行时间与输入 token 长度呈二次关系。随着 token 的增加，MHA 延迟将逐渐主导系统。

图 11（c）和图 11（d）分别显示了预填充（prefill）的运行时间。由于预填充中 token 长度的增加，相应的计算延迟将成比例增加。在这种情况下，系统的瓶颈将是计算吞吐量，而不是数据访问。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/17.webp)

**图 12 展示了稀疏 GLM 模型的加速性能**。首次解码延迟仅为 10.8 毫秒，对应在 56.86W 的功率下达到约 85.8 token/s 的峰值速度，显示出比 GPU 高 1.91 倍的吞吐量和高 7.55 倍的能效。我们还测量了 Qwen-7B 模型，所有结果与 GLM-6B 相似。在 Qwen-7B 模型中，采用稀疏策略 3 的解码速度为 69.4 token/s，由于 VMM 参数数量更多（7B 对比 6B），以及 MHA 块中更多的 KV 头（4 对比 2），导致 KVcache 参数加载的延迟更长，因此速度比 GLM-6B 慢。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/18.webp)

**为了证明 EdgeLLM 框架的高效率，我们进行了额外的实验**，并评估了**加速器在非 HBM 系统上的性能**，其中使用 DDR 代替 HBM。在边缘系统中，DDR 是最常见的内存芯片，带宽约为 60 GB/s。如表 III 所示，我们展示了解码和预填充测试用例。可以观察到，在解码模式下，与矩阵 - 向量乘法相关的几个操作符是 DDR 系统的主要瓶颈。这导致解码阶段的 token 生成速度仅为 HBM 系统的约 25%。在预填充阶段，由于权重参数的可重用性降低了系统对通信带宽的需求，速度下降相对较低。进一步的实验表明，如果预填充 token 长度更大，DDR 系统的性能下降将变得更小。

总体而言，**尽管在纯 DDR 系统中预填充和解码阶段的 token 生成速度有所下降，但 EdgeLLM 的持久性仍然足够满足边缘应用的需求**。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/19.webp)

**表 IV 显示了加速每个单独操作符时的能耗**。总体而言，FPGA 的功耗在 140/280MHz 下为 56.86W，其中在加载 140/280MHz 位流后的待机功耗为 40.36 W。可以看出，大多数非线性函数（如 LayerNorm/RMSNorm、Rotary Embedding、Softmax）消耗约 41 W，实际功耗仅为约 0.7W。而基于矩阵乘法的操作符显示出相对较高的功耗，范围从 2W 到 18W，主要由于 HBM 工作负载的高带宽和密集计算。

![文章插图](/articles/2025-04-28-edgellm-cpu-fpga-16cc0cdb/images/20.webp)

**表 V 显示了不同平台上的效率比较**。**GPU** 在大型语言模型中得到了广泛应用，在加速大型批次数据（通常用于云场景）时，它们的性能和效率非常高。然而，**当批次大小为 1（通常用于边缘应用场景）时，效率通常低于 30%**。此外，** GPU 的能耗也非常高**。

**FlightLLM 是目前最先进的 FPGA 加速器**，能够高效地进行大型语言模型推理，其**带宽利用率达到约 66%，能效比 GPU 高 6 倍**。由于协同效应，我们的**集成解决方案在性能上比 FlightLLM 提高了 11% 的带宽利用率和高达 24% 的能效提升**，显示出在实际边缘 AI 应用中的巨大潜力。

目前，**几乎所有的大型语言模型都遵循类似的运算符结构：MHA+FFN**。我们对这些运算符进行了全面优化，因此我们的**框架具有很好的通用性**，可以**轻松应用于 GLM 和 Qwen 之外的其他类型的大型语言模型**。然而，根据我们的分析，潜在的局限性在于旋转位置嵌入（Rotary Embedding）运算符，它在不同的大型语言模型中通常有所不同。

为了执行此类运算符，可以采用两种方法：

- 一种是为不同模型定制设计的运算符
- 另一种是使用 CPU+逐元素运算符（ElementWise Operator）

前者对特定模型的硬件效率高，但不能直接用于其他模型；后者具有更高的通用性，但速度较低。此外，本提案中的所有运算符都以时间模式（temporal-mode）执行，即一个运算符在前一个运算符完成后才开始。未来的优化可以探索不同运算符的并行执行。

## 六、结论（CONCLUSION）

> 本研究提出了 EdgeLLM，一种用于大型语言模型（LLM）的高性能 CPU-FPGA 异构加速方案。

**在硬件方面，我们分析了 FFN** （前馈神经网络）**和 MHA** （多头注意力）**的计算需求**，并提出了**高效的混合精度计算单元以及分组 systolic 架构**。同时，**我们对对数级结构化权重稀疏性进行了专门优化**，虽然算法精度略有下降，但硬件效率得到了极大提升。

**在软件方面，我们分析了整个运算符图**，并为 AI 算法中的**所有运算符设计了统一且通用的数据格式**，使不同运算符能够**无需任何数据重排即可进行处理**。然后，我们提出了**端到端的编译方案，可以动态编译所有运算符并映射整个模型**。

**该解决方案具有良好的通用性，可以适应多种大型模型算法**。例如，该加速器**已成功部署在 AMD Xilinx VCU128 FPGA 平台** 上。我们的结果在**吞吐量上比英伟达（NVIDIA）A100SXM4-80G GPU 高出 1.91 倍**，**能效高出 7.55 倍**，并且**比最先进的 FlightLLM FPGA 加速器的性能高出 10 到 24%。**

总体而言，EdgeLLM 解决了大型语言模型计算需求密集、内存访问繁重、运算符类型多样以及编译难题，显示出在不久的将来在边缘场景（如机器人）中的实际应用潜力。

## 参考文献

1. D. Silver et al., “Mastering the game of go with deep neural networks and tree search,” *Nature*, vol. 529, no. 7587, pp. 484–489, Jan. 2016, doi: 10.1038/nature16961.
2. K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 2014, arXiv:1409.1556.
3. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.
4. A. Vaswani et al., “Attention is all you need,” in *Proc. Adv. Neural Inf. Process. Syst.*, Long Beach, CA, USA, 2017, pp. 5998–6008. 5. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” 2018, arXiv:1810.04805.
5. A. Dosovitskiy et al., “An image is worth 16×16 words: Transformers for image recognition at scale,” 2020, arXiv:2010.11929.
6. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, Miami, FL, USA, Jun. 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.
7. T. Y. Lin et al., “Microsoft COCO: Common objects in context,” in *Proc. Eur. Conf. Comput. Vis.*, in Lecture Notes in Computer Science, vol. 8693, Cham, Switzerland: Springer, 2014, pp. 740–755, doi: 10.1007/978-3-319-10602-1_48.
8. R. Bommasani et al., “On the opportunities and risks of foundation models,” 2021, arXiv:2108.07258.
9. E. Kasneci et al., “ChatGPT for good? On opportunities and challenges of large language models for education,” *Learn. Individual Differences*, vol. 103, Apr. 2023, Art. no. 102274, doi: 10.1016/j.lindif.2023.102274.
10. OpenAI, [Online]. Available: https://chat.openai.com.chat.
11. A. Chowdhery et al., “PaLM: Scaling language modeling with pathways,” 2022, arXiv:2204.02311.
12. A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting, “Large language models in medicine,” *Nature Med.*, vol. 29, no. 8, pp. 1930–1940, Aug. 2023, doi: 10.1038/s41591-023-02448-8. 14. W. Xin Zhao et al., “A survey of large language models,” 2023, arXiv:2303.18223. 15. D. Demszky et al., “Using large language models in psychology,” *Nature Rev. Psychology*, vol. 2, pp. 688–701, Oct. 2023, doi: 10.1038/s44159-02300241-5. 16. S. Zeng et al., “FlightLLM: Efficient large language model inference with a complete mapping flow on FPGAs,” in *Proc. ACM/SIGDA Int. Symp. Field Programm. Gate Arrays*, Apr. 2024, pp. 223–234, doi: 10.1145/3626202.3637562. 17. Z. Li and Q. Gu, “I-ViT: Integer-only quantization for efficient vision transformer inference,” 2022, arXiv:2207.01405. 18. B. Li et al., “FTRANS: Energy-efficient acceleration of transformers using FPGA,” in *Proc. ACM/IEEE Int. Symp. Low Power Electron. Design*, Aug. 2020, pp. 175–180, doi: 10.1145/3370748.3406567. 19. M. Huang et al., “A high performance multi-bit-width booth vector systolic accelerator for NAS optimized deep learning neural networks,” *IEEE Trans. Circuits Syst. I, Reg. Papers*, vol. 69, no. 9, pp. 3619–3631, Sep. 2022, doi: 10.1109/TCSI.2022.3178474.
13. M. Huang, J. Luo, C. Ding, Z. Wei, S. Huang, and H. Yu, “An integer-only and group-vector systolic accelerator for efficiently mapping vision transformer on edge,” *IEEE Trans. Circuits Syst. I, Reg. Papers*, vol. 70, no. 12, pp. 5289–5301, Dec. 2023, doi: 10.1109/TCSI.2023.3312775.
14. Y. Su et al., “LLM-barber: Block-aware rebuilder for sparsity mask in one-shot for large language models,” 2024, arXiv:2408.10631. 22. J. Lin et al., “AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration,” in *Proc. Mach. Learn. Syst.*, vol. 7, 2024, pp. 87–100.
15. Z. Liu et al., “Deja Vu: Contextual sparsity for efficient LLMS at inference time,” in *Proc. Int. Conf. Mach. Learn.*, 2023, pp. 22137–22176.
16. E. Frantar and D. Alistarh, “SparseGPT: Massive language models can be accurately pruned in one-shot,” in *Proc. Int. Conf. Mach. Learn.*, 2023, pp. 10323–10337.
17. Z. Liu et al., “Scissorhands: Exploiting the persistence of importance hypothesis for llm KVcache compression at test time,” in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 36, 2023, pp. 52342–52364.
18. C. Raffel et al., “Exploring the limits of transfer learning with a unified text-to-text transformer,” *J. Mach. Learn. Res.*, vol. 21, no. 140, pp. 1–67, 2020.
19. L. Xue et al., “MT5: A massively multilingual pre-trained text-to-text transformer,” 2020, arXiv:2010.11934.
20. T. Glm et al., “ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools,” 2024, arXiv:2406.12793.
21. R. Anil et al., “PaLM 2 technical report,” 2023, arXiv:2305.10403.
22. OpenAI, “Sora: Creating Video from Text,” [Online]. Available: https://openai.com/sora.
23. N. Jouppi et al., “In-datacenter performance analysis of a tensor processing unit,” in *Proc. 44th Annu. Int. Symp. Comput. Architect.*, Toronto, ON, Canada, Jun. 2017, pp. 1–12, doi: 10.1145/3079856.3080246.
24. N. Jouppi et al., “TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings,” in *Proc. 50th Annu. Int. Symp. Comput. Archit.*, Jun. 2023, pp. 1–14, doi: 10.1145/3579371.3589350.
25. H. Peng et al., “Accelerating transformer-based deep learning models on FPGAs using column balanced block pruning,” in *Proc. Int. Symp. Quality Electron. Design*, 2021, pp. 142–148, doi: 10.1109/ISQED51717.2021.9424344.
26. T. Wang et al., “ViA: A novel vision-transformer accelerator based on FPGA,” *IEEE Trans. Comput.-Aided Design Integr. Circuits Syst.*, vol. 41, no. 11, pp. 4088–4099, Nov. 2022, doi: 10.1109/TCAD.2022.3197489.
27. C. Zhang, P. Li, G. Sun, Y. Guan, B. Xiao, and J. Cong, “Optimizing FPGA-based accelerator design for deep convolutional neural networks,” in *Proc. ACM/SIGDA Int. Symp. Field-Program. Gate Arrays*, Feb. 2015, pp. 161–170, doi: 10.1145/2684746.2689060.
28. Z.-G. Liu, P. N. Whatmough, Y. Zhu, and M. Mattina, “S2TA: Exploiting structured sparsity for energy-efficient mobile CNN acceleration,” in *Proc. IEEE Int. Symp. High-Perform. Comput. Archit. (HPCA)*, Seoul, South Korea, Apr. 2022, pp. 573–586, doi: 10.1109/HPCA53966.2022.00049.
29. NVIDIA A100 Tensor Core GPU Architecture, NVIDIA Whitepaper, Santa Clara, CA, USA, 2021, [Online]. Available: https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf.
30. THUDM, “ChatGLM2-6B,” [Online]. Available: https://github.com/THUDM/ChatGLM2-6B.
31. Vitanova, “Qwen,” [Online]. Available: https://github.com/vitanova/Qwen2.
