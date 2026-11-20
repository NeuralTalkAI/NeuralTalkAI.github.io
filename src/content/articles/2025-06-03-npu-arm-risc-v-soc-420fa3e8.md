> 在设备上进行神经网络（NN）推理相比于基于云端的处理具有诸多优势，包括可预测的延迟、增强的隐私性、更高的可靠性以及为供应商降低运营成本。

这激发了微控制器级神经网络加速器（通常称为神经处理单元（μNPUs））的快速发展，**这些加速器专门针对超低功耗应用而设计**。

::: block-1在本文中，我们**首次对多种商用可用的 μNPUs 进行了比较评估，并为其中一些平台提供了独立的基准测试。**

我们**开发并开源了一个模型编译框架**，以**实现跨不同 μNPU 硬件的量化模型** 的**一致性基准测试**。我们的基准测试针对端到端性能，包括模型**推理延迟**、**功耗** 和**内存开销** 以及其他因素。

分析结果揭示了**硬件规格与实际性能之间的预期趋势以及令人惊讶的差异**，包括一些 μNPUs 在**模型复杂度增加时表现出意外的扩展行为**。
:::

我们的框架为进一步评估 μNPU 平台提供了基础，并为这一快速发展的领域的**硬件设计人员和软件开发人员提供了宝贵的见解**。![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/01.png)

- https://arxiv.org/abs/2503.22567

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/02.png)

## 本文目录

- 一、引言
- 二、背景与动机
  - 2.1 受限环境中的神经计算
  - 2.2 μNPU 硬件设计
  - 2.3 基准测试 μNPU 平台
- 三、架构与方法论
  - 3.1 硬件
  - 3.2 模型
  - 3.3 评估指标
  - 3.4 性能分解
- 四、结果与讨论
  - 4.1 功率和效率分析
  - 4.2 延迟和内存 I/O 分析
  - 4.3 特定任务的考量
  - 4.4 结果总结
  - 4.5 未来发展方向
  - 4.6 实用建议
  - 4.7 限制
- 五、相关工作
  - 端到端基准测试
  - 针对 MCUs 的 NN 加速器
  - 高效的设备端推理
- 六、结论
- 参考文献

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/03.jpg)

加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

在受限设备上执行神经网络（NN）推理在众多领域都有应用，包括可穿戴健康监测[1]、智能农业[2]、实时音频处理[3]和预测性维护[4]。

::: block-1与基于云端的替代方案相比，**设备端推理具有诸多优势**：

- 对于时间敏感型应用，可以**改善延迟**；
- 增强**隐私性**，因为无需传输敏感数据；
- 通过**消除对网络连接的依赖**，提高了可靠性；
- 并且通过消除数据传输需求，降低了供应商的运营成本。
:::

鉴于其独特的外形尺寸和低功耗特性，微控制器（MCUs）被广泛应用于资源受限的环境。然而，它们的性能通常受到内存容量、吞吐量和计算能力的限制。

现代神经网络（NNs）的计算需求推动了从高性能数据中心到超低功耗和嵌入式设备的整个计算范围内的专用硬件加速器的发展。在资源受限的低端，**微控制器级神经处理单元（μNPUs）最近开始出现，它们设计用于在极其严格的功耗范围内运行（毫瓦或亚毫瓦级别），同时仍能为实时推理提供低延迟**。这些设备代表了一类新的加速器，结合了微控制器的功耗效率和此前仅属于更强大计算平台的认知能力。

μNPUs 的**核心优势** 在于它们能够利用神经网络的固有并行性，通过专用的乘累加（MAC）阵列以及用于权重存储的专用内存结构来实现。

> 这种架构上的专门化使得 μNPUs 能够在执行等效工作负载时，与通用微控制器相比，实现数量级的延迟改进。尽管 μNPU 平台的数量不断增加，但该领域缺乏标准化的评估或全面的基准测试套件。

**现有的基准测试仅关注 Analog Devices 的 MAX78000[5–7]，缺乏与其他平台的比较**。硬件供应商基于专有的评估框架提供性能指标，通常使用不同的神经网络模型、量化策略以及其他各种优化。

**这种评估方法的异构性以及缺乏对供应商提供的性能声明的独立验证**，为硬件设计人员和嵌入式软件开发人员在选择最适合其应用约束的 μNPU 平台时带来了不确定性。**缺乏标准化的基准测试** 也阻碍了研究，因为它**掩盖了架构设计与实际性能之间的关系**。

::: block-1鉴于 μNPU 平台的快速发展和日益多样化，**为该领域建立可靠的比较基准已成为当务之急**。为此，我们做出了以下贡献：

1. **基准测试 μNPU 平台**：我们**首次** 对商用可用的 μNPU 平台进行了比较评估，使得能够**在一致的工作负载和测量条件下**，直接**比较不同硬件架构** 的性能。
2. 独立基准测试：我们还为一些此前未经过第三方评估的 μNPU 平台提供了**首次细致且独立** 的性能基准测试，**提供对供应商性能声明的无偏验证**。
3. **开源模型编译框架**：我们开发并开源了一个框架，能够**简化跨不同 μNPU 平台移植神经网络模型的过程**，降低跨平台评估的工程开销。
4. 开发者建议：基于我们的基准测试结果，我们**为开发者提供了关于平台选择、模型优化的关键关注点** 以及**各种应用场景和约束下的权衡** 的**可行建议**。
:::

通过开发**统一的编译和基准测试框架**，我们在各种 μNPU 平台上**标准化了模型表示，使得可以直接比较延迟、内存和能耗性能**。

> 我们的评估还包括**对模型执行各个阶段的细致分析**，从 NPU **初始化和内存输入/输出开销到 CPU 预处理/后处理** ——这些方面可能会显著影响端到端性能，但在技术评估中常常被忽视。

分析**结果揭示了硬件规格与实际性能之间的预期趋势以及令人惊讶的差异**，包括**一些 μNPUs 在模型复杂度增加时表现出意外的扩展行为**。

> 我们希望我们的发现能够为**开发者和硬件架构师提供宝贵的见解**。

## 二、背景与动机

### 2.1 受限环境中的神经计算

从云端计算转向设备端神经计算在实时数据处理方面具有诸多优势，尤其是在数据隐私和安全日益受到关注的情况下[8]。与基于云端的解决方案不同，本地推理通过在本地处理敏感数据来降低安全风险，这在医疗诊断和监控等领域尤为有利[9, 10]。

此外，本地处理还可以减少端到端的延迟，并降低模型供应商的运营成本。然而，传统的神经网络加速器（如 GPU 和 TPU）由于功耗较高且外形尺寸较大，不适合在资源受限的环境中使用[11, 12]。**微控制器（MCUs）是一种紧凑型、低功耗的计算平台**，通常**依赖于单个 CPU 和共享内存总线** [13]。尽管 MCUs 常被用于资源受限的物联网（IoT）应用[14–16]，但它们通常缺乏高效的神经网络推理计算能力。

具体来说，**典型 MCUs 的计算能力通常仅能达到每秒数百万次乘累加（MAC）操作，远低于实时神经网络推理所需的每秒数十亿次 MAC 操作**。它们**缺乏专用硬件加速**，导致在处理神经网络时出现较大的延迟开销和较高的功耗。此外，**有限的 SRAM 和闪存内存也常常给高效管理神经网络模型所需的大型权重矩阵带来挑战**。

> 鉴于传统 MCUs 的诸多不足，微控制器级 μNPUs 应运而生。

这些**专用的神经网络加速器** 提供了**专用的神经处理硬件**，为神经网络工作负载提供了更高的吞吐量，满足了实时神经网络推理的严格要求[17–19]，同时保持了低功耗运行。

> 总的来说，μNPUs 成为了在资源受限环境中高效、实时处理神经网络的关键解决方案。

### 2.2 μNPU 硬件设计

**μNPU 硬件设计通过专用的乘累加（MAC）单元和可并行化的内存层次结构来优化高效的张量操作** [20, 21]。**图 1 展示了典型 μNPU 的架构**，由一个脉动阵列（systolic array）的处理单元（PEs）组成。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/04.png)

值得注意的是，**每个 PE 都包含自己的 MAC 单元**，更重要的是，它还**有自己的权重内存空间，以避免内存争用并最大化并行化**。PE 阵列通过 PE 间通信网格连接，该网格连接到一个大型全局缓冲区和通过片上网络连接的 SRAM/DRAM[22]。通过划分可用 RAM，并实现高带宽内存接口和数据预取机制，实现了高效的内存层次结构优化，解决了传统 MCUs 在处理大型神经网络模型权重时面临的内存瓶颈问题。

**μNPUs 主要在 PE 的数量、PE 布局和聚类、内存层次结构布局以及每个 PE 中存储/ MAC 单元的可用性和数量上有所不同**。这些架构优势，加上低功耗优化技术（如电源门控），使得 μNPU 平台能够为实时神经网络推理提供低功耗、高吞吐量的性能。

### 2.3 基准测试 μNPU 平台

**采用 μNPU 平台的需求**：对设备端神经计算的需求增加加速了 μNPUs 的开发和商业化。这一点从越来越多的供应商中得到了证明，例如 Arm[23]，他们已经向市场推出了 μNPU 平台。

**对全面基准测试的需求**：**现有的关于 μNPU 平台的工作** 主要关注实际应用和/或模型优化[24–26]，**缺乏从系统角度对性能进行细致分析**。在评估 μNPU 平台的内存使用、延迟、功耗和吞吐量时，**我们的目标是揭示关键的性能瓶颈**，引导研究人员**朝着更高效的软件和神经网络模型设计方向发展**。

**现有基准测试的局限性**：现有的 μNPU 基准测试主要**集中在单一平台上，缺乏对现在种类繁多的可用平台进行横向比较** [6, 7, 27]。这种狭窄的视角限制了对不同 μNPUs 之间性能变化和基于任务的适用性的理解。现有的独立基准测试也存在显著的不足。最重要的是，**大多数基准测试仅关注模型的前向推理过程，忽略了端到端模型推理或应用程序流程** （s）中的其他相关操作，**例如 NPU 初始化、内存输入/输出（I/O）以及 CPU 预处理/后处理**。尽管这些因素经常被忽视，但它们**可能会显著影响整体性能和效率。**

## 三、架构与方法论

我们首先详细介绍我们的基准测试硬件和模型，然后提供一个关于我们基准测试框架和模型推理流程的全面概述

### 3.1 硬件

为了提供一个全面的基准测试，我们评估了一系列广泛使用的、商用的 μNPU 平台，**从超低功耗的 μNPUs 到高性能的 NPU 集成系统级芯片（SoC）架构**。这些平台与没有专用神经硬件的微控制器（MCUs）一起进行评估，以便进行比较。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/05.png)

我们的选择涵盖了广泛的计算能力（小于 5 到大于 500 GOPs，即每秒十亿次操作）、内存配置（128 KB 到 2 MB RAM）以及位宽支持（从 1 位量化到 32 位浮点运算）。**图 2 提供了我们基准测试中包含的各种 μNPU 平台的峰值 GOPS（每秒十亿次操作）与峰值功耗的可视化（以对数刻度表示）**。

**表 3 细列出了我们的基准测试 μNPUs，我们将在下面对每个平台进行更详细的介绍**。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/06.png)

**MAX78000（或 MAX78K）[5]是由美国 Analog Devices 公司推出的**，它包含一个 **Cortex-M4F 和一个 RISC-V 协处理器**，每个都可以作为主处理器，**以及一个专有的 30-GOPS CNN 加速器。** 后者的加速器拥有一个专门的 512 KB SRAM 用于输入数据，442 KB 用于权重，2 KB 用于偏置，并且支持 1、2、4 和 8 位精度的量化操作。这种细粒度的位宽量化尚未在其他 μNPU 平台上广泛支持，或者在为资源受限设备设计的常见机器学习软件库中也不支持；例如 TFLite/LiteRT [28]仅支持 8 位整数和 16 位浮点权重量化。**MAX78000 还拥有 512 KB 的闪存和 128 KB 的 CPU 专用 SRAM从超低功耗的 μNPUs 到高性能的 NPU 集成系统级芯片（SoC）架构从超低功耗的 μNPUs 到高性能的 NPU 集成系统级芯片（SoC）架构**。这个平台是商用 μNPUs 中记录最完善的平台之一；之前的研究已经对它的 CNN 加速器在各种配置下进行了基准测试[6, 7, 27]，并且探索了针对其 2D 内存布局的最佳模型和数据加载策略[29]。

**GAP8 [30]是 GreenWaves Technologies 的 GreenWaves 应用处理器系列的一部分，它包含一个 8 核 RISC-V 集群和一个 22.65-GOPS 硬件卷积引擎，用于 8 位或 16 位精度的神经网络加速**。该平台拥有 512 KB 的 L2 RAM，最多 8 MB 的 L3 SRAM，以及 20MB 的闪存存储，使其能够存储和运行更大、更复杂的模型或混合专家（MoE）架构。GAP 系列的 μNPUs 也是最近几项研究的主题，这些研究主要集中在模型优化[26, 31, 32]上；目前还没有平台基准测试。

**Himax HX6538 WE2（或 HX-WE2）[33]是来自台湾半导体制造商 Himax Technologies 的更强大的 μNPU 平台**。该平台采用了 Corstone-300 配置，包含一个 Cortex M55 CPU 和一个 Ethos U55 NPU，能够提供高达 512 GOPS 的性能。该平台还配备了 512KB 的 TCM（紧密耦合存储器）、2MB SRAM 和 16MB 闪存，适合运行大型或更复杂的模型，但会增加功耗。 NXP 的 MCXN947 [34]是 MCX N94x 系列微控制器的一部分，它包含两个 Cortex-M33 CPU 和 NXP 的 eiQ Neutron NPU。MCXN947 是为低功耗应用设计的，其 8 位神经网络加速能力为 4.8 GOPS。该平台拥有 512 KB RAM 和 2 MB 闪存存储。

**我们的基准测试还包括没有神经硬件的 MCUs 进行比较**，以量化专用 NPU 架构所获得的效率提升。STM32H7A3ZI [35]是一个**基于 Cortex M7 的高性能 MCU**，拥有 2 MB 的闪存和 1.4 MB 的 SRAM。它由总部位于瑞士的 ST Microelectronics 制造，经常与板载神经网络[16, 36]一起使用。

**ESP32s3 MCU [37]拥有两个 Tensilica LX6 处理器核心，512 KB 的 SRAM，2MB 的 PSRAM 和 8MB 的闪存**。值得注意的是，尽管它主要是一个低功耗 MCU，但它声称具有神经网络加速能力，通过支持“矢量指令……为神经网络计算提供加速”。这是通过扩展指令集实现的，包括 128 位矢量操作、复数乘法、加法、减法、移位和比较等操作。

我们还包括了 **MILK-V Duo [38]，这是一个基于 CVITEK CV1800B 处理器的 RISC-V SoC**。与前面的 MCUs/μNPUs 不同，它运行 Linux 操作系统，支持更灵活的神经网络工作负载，但以更高的功耗预算为代价。

#### 3.1.1 关于 CPU 频率的说明

我们**将各种 μNPU 平台配置为以统一的 CPU 频率运行**。虽然这允许直接比较架构效率，但需要注意的是，许多平台能够以比评估中更高的频率运行——在某些情况下接近 GHz 范围。**我们的方法是有意隔离架构效率的，但进一步的实验可以探索不同 CPU 频率对端到端延迟和功耗的影响**。其他硬件参数大多默认标准化。进一步的研究可以调查它们对性能的影响；不同 μNPU 平台上的活动 PE 数量、内存布局配置以及其他硬件特定优化是可以配置的，并且会影响整体效率。

### 3.2 模型

**表 2 详细列出了我们在基准测试中使用的各种基于 CNN 的模型**，涵盖图像分类、目标识别和信号重建应用。我们在下面对每个模型进行更详细的介绍。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/07.png)

**CIFAR10-NAS**：通过神经架构搜索（NAS）为 CIFAR-10 数据集生成的最优 CNN 模型，结合了多种卷积单元、池化层和独特的连接模式。该模型是使用 Once-for-All（OFA）NAS 框架生成的，这是一个基于权重共享的框架，通过构建一个超网络模型来解耦搜索和训练，从而可以从该超网络模型中推导出各种硬件特定的子网络模型[39]。**这是我们最大的模型，拥有 7430 万次 MAC（乘累加）操作和 3640 万次 FLOPs（浮点运算）**。它在 CIFAR-10 数据集上进行训练，**输入尺寸为 3x32x32，输出为 10 类**。

**ResidualNet**：一个基于残差函数构建的 CNN 框架，有助于缓解梯度消失问题。ResidualNet 拥有 **3770 万次 MAC 操作和 1850 万次 FLOPs**。它在 CIFAR100 数据集上进行训练，**输入尺寸为 3x32x32，输出为 100 类。**

**SimpleNet**：一个更简单的 CNN 框架，由基本的卷积层和池化层堆叠而成，没有复杂的分支或残差函数。尽管其架构简单，但 SimpleNet 通常能够在某些情况下超越更复杂的模型，包括某些版本的 ResidualNet[40]。SimpleNet 拥有 **3800 万次 MAC 操作和 1850 万次 FLOPs**。它在 CIFAR-100 数据集上进行训练，**输入尺寸为 3x32x32，输出为 100 类。**

**Autoencoder**：一个对称的编码器-解码器模型。这是我们的最简单模型，仅拥有 **5455 次 MAC 操作和 2020 次 FLOPs**。它在一个机器故障检测数据集上进行训练，该数据集是使用 SpectraQuest 机械故障模拟器生成的[41]。**输入/输出尺寸为 3x256。**

**YoloV1**：一个单阶段目标检测 CNN，它使用多尺度输出特征图来预测边界框和类别概率。YoloV1 拥有 **4383 万次 MAC 操作和 2120 万次 FLOPs**。它在 COCO 数据集[42]上进行训练，仅用于检测人物，**输入尺寸为 3x96x96**，有 3 个输出层，这是通过修剪其最终层以实现跨平台一致性（更多细节见下文）的结果。**非最大抑制（NMS）后处理步骤是在 CPU 上执行的。**

#### 3.2.1 确保模型一致性

**我们在基准测试平台上遇到了操作（算子）支持的巨大差异**。

- 例如，NXP-MCXN947 的 eiQ Neutron NPU 缺乏对 softmax 操作的原生支持，因此需要将其作为相关模型的 CPU 后处理步骤来实现。
- 同样，YoloV1 模型中的非最大抑制（NMS）操作在各个平台上的支持也不一致，这要求我们将整个 NMS 操作移至 CPU 后处理。

这也解释了我们 YoloV1 模型的多组件输出形状（见表 2）。**基准测试平台还概述了不同水平的操作兼容性**。例如，MAX78000 仅支持 1D 卷积，核大小为 1 到 9，以及 2D 卷积，核大小为 1x1 或 3x3。**不支持的操作将回退到 CPU 执行，并产生延迟惩罚**。

> **通过识别并构建使用所有 μNPUs 普遍支持的核心操作子集的模型**，我们旨在**确保任何测量到的性能差异源于基本架构差异**，而不是模型编译和优化的变化。

#### 3.2.2 量化

> 我们将所有基准模型量化到 INT8 精度，因为所有评估的 NPUs 都支持它。

然而，重要的是要注意，虽然这使得架构比较更加直接，但它可能无法反映每个平台上精度与性能之间的最佳权衡；一些 NPUs（如 MAX78000）支持更低的位宽（例如 1、2、4 位），而其他 NPUs（如 HX-WE2）支持浮点加速（例如 FLOAT16 和 32 位）。**我们对所有模型/平台上执行后训练量化（PTQ）。**

尽管一些平台（如 MAX78000）支持量化感知训练（QAT）和融合算子，但这些优化会产生平台特定的模型，与其他 NPUs 不兼容。**PTQ 使我们能够保持所有平台上的模型结构一致性。此外，由于我们主要感兴趣的指标是延迟和功耗，而不是推理精度，PTQ 为性能评估提供了一个足够有代表性的模型。** PTQ 是使用针对每个模型领域的代表性校准数据集进行的。我们没有对权重应用每通道量化，而是使用每张量量化以确保跨所有平台的兼容性。

#### 3.2.3 编译

各种 μNPUs 支持广泛的模型格式，从平台优化的常见模型格式（例如 TFLite）到平台特定的格式（例如 CVITEK 的 CVIMODEL）。为了便于跨平台部署，**我们开发了一个自定义模型编译工作流，用于将基础模型转换为每个目标 NPU 的优化格式**。

我们的工作流输入 Torch（或 ONNX/TFLM）基础模型以及各种编译器标志（例如目标 NPU 平台、模型输入维度、位精度要求、代表性 PTQ 校准数据等），输出针对目标 NPU 平台优化的模型以及相应的推理代码。编译过程因平台而异。例如，针对 ARM Ethos-U55（在 HX-WE2 上）的模型是使用 ARM Vela 编译器编译的，它接受 TFLiteMicro（TFLM）模型，并生成针对 Ethos-U 架构优化的二进制文件。Vela 应用平台特定的优化，包括内存减少策略。

::: block-1我们评估了尺寸（Size）优化策略：

- HX-WE2（S），它最小化 SRAM 使用，以及性能策略
- HX-WE2（P），它优先考虑执行速度，如果指定了可用的 arena 缓存则使用
- 对于其他平台，我们使用它们各自的工具链（例如 MAX78k 的 SDK 或 NXP eIQ 门户工具）。
:::

在每种情况下，我们配置这些工具以保持模型结构等价性，同时应用平台适当的优化。请注意，在我们的模型编译工作流中，通常会生成模板推理代码以及编译后的模型。然而，这不包括模型特定的预处理/后处理步骤，这些步骤应由开发人员手动实现，开发人员可以根据需要更新模板代码。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/08.png)

**上图 3 下面详细介绍了我们的模型编译框架，用于将基础（Torch/ONNX/TFLM）模型转换为各种平台特定的格式**。我们将开源我们的框架，希望它的使用可以简化跨平台模型编译和基准测试的过程。

### 3.3 评估指标

我们**在每个基准平台和模型上测量延迟、功耗、能效** （以每毫焦耳的推理操作次数表示）以及**内存使用情况**。各个平台上特定模型优化或编译工作流对模型精度的影响不在我们研究的范围内。

#### 3.3.1 延迟

**使用每个平台的内部计时器测量延迟**。值得注意的是，除了 MILK-V 外，所有 MCUs 都被配置为以 100 MHz 运行。MILK-V 不支持手动频率缩放，仅支持动态电压频率调整（DVFS）。然而，延迟与 CPU 频率成反比，如公式 $T = N / f$ 所示（$T$ 表示延迟，$N$ 表示任务所需的周期数，$f$ 表示运行频率）。因此，我们将 MILK-V 的延迟标准化，以使其与统一频率条件下的性能相当。每个模型运行了 10 次连续推理。我们报告平均延迟和标准差，以考虑任何运行间的变化。

#### 3.3.2 功耗和能量

使用 Monsoon High Voltage Power Monitor [43]**以 50 Hz 的采样率计算功耗和能量**。输入电压（$U$）设置为 3.3 V，捕获推理持续时间（$t$）和平均功耗（$P$），从而得出平均能量消耗（$E = Pt$）。为了确保稳定的测量结果，我们仅分析运行 1 分钟后记录的数据。功耗测量是在 10 次重复推理上进行的，我们报告平均值和标准差。图 4 详细展示了在 HX-WE2 的 Ethos-U55μNPU 上进行 YOLOv1 推理的功率曲线。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/09.png)

#### 3.3.3 推理次数/毫焦耳

**为了量化能效，我们引入了“推理次数/毫焦耳”** （$I_{mJ}$），表示每消耗 1 毫焦耳能量所执行的端到端推理次数（包括内存传输、CPU 预处理/后处理，以及可选的 NPU 初始化）。

#### 3.3.4 内存使用

**通过分析编译工具链生成的链接器（$.map$）文件来评估内存使用情况**。该文件提供了内存分配的详细分解，包括代码（$.text$）、初始化数据（$.data$）和未初始化数据（$.bss$）段。闪存内存使用量计算为代码和初始化数据段的总和（$.text + .data$），而 RAM 使用量包括初始化和未初始化数据段（$.data+.bss$）。对于 MAX78000，由于其专用的 NPU-only 内存，RAM 使用量分别针对 CPU 和 NPU 进行计算。

### 3.4 性能分解

> 我们将模型执行的每个阶段进行分解，并测量每个阶段的延迟和功耗。这种**细致的分析有助于识别推理流程中的特定瓶颈**，同时测量整体端到端性能。

我们**还测量了各个平台在没有进行活动计算时的空闲功耗** （例如，各个平台在没有活动计算时的基本功耗）。对于没有神经硬件的 MCUs（例如 STM32H7A3ZI 和 ESP32），初始化和内存 I/O 是合并在一起的。

- **NPU 初始化**：这涵盖了任何 NPU 设置开销，包括内存缓冲区分配和内核配置。
- **内存 I/O**：模型和输入数据加载的成本/开销，包括将输入张量和模型权重从闪存移动到 NPU DRAM，反之亦然（例如，输出张量从 NPU 到 CPU SRAM）。
- **推理**：在 NPU 上执行模型的前向传递。
- **后处理**：需要在 CPU 上执行的任何额外操作。这包括为 ResidualNet、SimpleNet 和 CIFAR10-NAS 模型计算 softmax 输出。YoloV1 的后处理包括非最大抑制（NMS）以及输出类别的 softmax。Autoencoder 模型不需要后处理，因为它产生直接的重建输出。
- **空闲**：各个平台在不进行活动计算时的基本功耗。

---

对于 MCUs 没有神经硬件（例如 STM32H7A3ZI 和 ESP32），初始化和内存 I/O 是合并的。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/10.png)

## 四、结果与讨论

下图是位于补充材料中的表 6 ，详细列出了我们在每个阶段、模型和平台上测量的完整延迟和功耗数据。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/11.png)

### 4.1 功率和效率分析

我们的结果揭示了在基准测试平台之间的效率存在显著差异，如表 3 和表 4 所示。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/12.png)

当考虑 NPU 初始化开销时，使用 Cortex-M4 CPU 的 MAX78000（C）在所有评估模型中展现出最佳的整体效率，端到端延迟始终保持在 30 毫秒以下。使用 RISC-V CPU 的 MAX78000（R）略逊一筹。这与之前的独立基准测试结果一致[6]。NXP-MCXN947 也实现了持续低于 30 毫秒的延迟，其快速初始化和内存 I/O 抵消了（相对）较慢的推理延迟带来的影响，尽管其加速器的吞吐量较低，但在某些情况下仍展现出与 MAX78000 相当（甚至更优）的效率。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/13.png)

**值得注意的是，功耗较高但延迟较低的 HX-WE2 平台** （采用 Arm Corstone-300，包含 Cortex-M55 和 Ethos-U55 NPU）**在各种模型的端到端延迟方面始终优于 MAX78000** （C/R），**原因在于后者的内存 I/O 开销较大**。与 MAX78000（C）和（R）相比，HX-WE2（S）和（P）分别平均实现了约 1.93 倍和 3.07 倍的端到端延迟加速，但平均功耗分别增加了约 3.13 倍和 3.33 倍。我们发现，对于 HX-WE2 平台，经过 Vela 性能优化的模型通常比经过尺寸优化的模型实现略低的延迟。**然而，随着模型复杂度的增加，其效率提升逐渐减弱** ——例如，在性能优化的 YoloV1 模型上，其效率低于尺寸优化的变体。

**没有专用神经硬件的通用 MCUs** （如 STM32H7A3ZI 和 ESP32s3）**在所有模型上都表现出显著较低的效率**。这一结果从实证角度验证了在资源受限的环境中进行设备端推理时，专用神经硬件所带来的优势，某些情况下端到端延迟可提升多达两个数量级。**然而，STM32H7A3ZI 在推理时的功耗（54.91-56.11 毫瓦）与某些模型的 MAX78000（C/R）相当，甚至更低**。这在 Autoencoder 模型中尤为明显，STM32H7A3ZI 实现了 3.483 次/毫焦耳的竞争力指标，与我们测试套件中表现最佳的平台相当。**这种异常现象可能归因于 STM32H7A3ZI 相对高效的 Cortex-M7 核心在处理 Autoencoder 简单计算结构** （0.5455 百万 MAC）时的表现。相比之下，ESP32 在推理时始终表现出较高的功耗（129.74-157.17 毫瓦）和延迟（7.11-536.22 毫秒），尽管其声称支持 CPU 加速的张量运算。

**总体而言，虽然通用 MCUs 可以实现简单模型的合理效率，但对于更复杂的神经网络，它们很快就会变得不切实际**。我们的 RISC-V SoC（MILK-V）在所有模型上也表现出较低的效率，这是由于其 NPU 初始化开销所致。然而，如果忽略初始化开销（例如，在连续操作中），MILK-V 在几乎所有基准模型中的效率排名最高。值得注意的是，尽管其空闲功耗较高，但 MILK-V 实现了极快的推理时间（0.17-0.61 毫秒）。因此，**对于那些功耗不是主要限制因素、工作负载不需要频繁 NPU 初始化/去初始化，但需要低延迟和紧凑外形的应用场景，MILK-V 可能会是一个有效的选择**。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/14.png)

**图 7 详细展示了所有评估平台的功耗分布情况**。在这些平台中，MAX78000（功耗范围为 10.87-80.41 毫瓦）和 NXP-MCXN947（功耗范围为 22.91-36.69 毫瓦）在所有基准模型中展现出最低的功耗，其中 NXP 在执行阶段的峰值功耗变化最小，使得能量预算更加可靠。

**除了峰值功耗外，空闲功耗也是低功耗部署中的一个关键考虑因素，尤其是在工作负载运行不频繁的情况下——空闲功耗在我们的基准测试平台之间也存在显著差异**。MAX78000 在各种 μNPU 平台中展现出最低的空闲功耗（使用 RISC-V 时为 10.87 毫瓦，使用 Cortex-M4 时为 13.21 毫瓦）。而 HX-WE2 平台的空闲功耗最高（89.09 毫瓦），这**引发了对其在极度功耗受限场景下的适用性的担忧** （例如，**在空闲时间占主导的场景中**，空闲功耗将主导整体能量使用）。

### 4.2 延迟和内存 I/O 分析

#### 4.2.1 NPU 初始化

不同基准测试平台的 NPU 初始化时间存在**显著差异**，从 MAX78000 上的最低 0.07 毫秒到 GAP8 上的 12.94 毫秒不等。然而，除了 GAP8（在各种基准模型上，初始化延迟在 7.46 到 12.92 毫秒之间）之外，**大多数 μNPU 平台上的实际初始化开销相对于端到端延迟来说几乎可以忽略不计**。这种开销对于需要频繁加载/卸载模型的占空比应用来说可能是一个问题。

#### 4.2.2 内存 I/O

**表 5 详细列出了我们在各种基准测试平台和模型上的闪存和 RAM 使用情况**。

![文章插图](/articles/2025-06-03-npu-arm-risc-v-soc-420fa3e8/images/15.png)

在所有模型上，MAX78000 的显著内存 I/O 延迟形成了一个明显的推理瓶颈，其平均内存 I/O 时间分别是实际推理时间的 6.10 倍和 9.80 倍（使用 Cortex-M4 和 RISC-V 时）。例如，对于在 MAX78k（R）上运行的 ResidualNet，内存 I/O 时间为 44.89 毫秒，而实际推理时间为 2.96 毫秒，这意味着**超过 90%的端到端推理时间被用于内存操作，而不是计算**。这表明 MAX78000 的**性能在很大程度上受到内存的限制，这也与之前的独立基准测试结果一致** [6]。

> 值得注意的是，与 RISC-V 相比，MAX78000 的 Cortex-M4 CPU 在内存 I/O 操作上更为高效。

相比之下，其他具有共享 SRAM 的基准测试平台的内存 I/O 操作引入的开销可以忽略不计。与传统的依赖于一维连续内存空间的 CPU 和 GPU 不同，μNPU 硬件采用了二维内存布局；**在这种布局中，一个轴映射到并行计算核心，另一个轴组织逻辑地址空间。如图 1 所示，每个 PE（处理单元）都配备了自己独立的权重内存空间，以避免内存争用并最大化并行化**。这导致了一个具有通道级和权重级并行性的层次化架构，尽管权重必须使用相同的偏移量。最近的研究[29]探索了针对这种二维内存布局优化权重加载策略，以减少在单个设备上切换模型时的 I/O 延迟，包括：

- 在加速器内虚拟化权重内存，以减少碎片化；
- 优化动态权重分配，以最小化加载/卸载开销；
- 权重预加载，即在执行前，由空闲的 CPU 将下一个模型的权重加载到未使用的内存区域。

进一步的研究应该包括自动化内存管理，以及使用诸如即时预取、动态量化或输入自适应剪枝等技术来减少单模型执行的 I/O 延迟。

#### 4.2.3 推理

另一个意外的发现是**尽管其理论计算能力** （30 GOPs）**远低于 HX-WE2** （512 GOPs），**MAX78000 在推理延迟方面的优越性**。例如，MAX78000（C）平均比 HX-WE2（P）实现了约 2.48 倍的延迟改进。这可能是由于与 Arm Ethos-U55 相比，**MAX78000 采用了更适合 CNN 工作负载的权重静止数据流模式** （weight-stationary dataflow patterns）。然而，HX-WE2 尽管它在内存 I/O 延迟方面大幅减少，在端到端延迟方面仍然胜出。**HX-WE2 在不同模型上的相对一致的推理时间也表明其架构更适合于比我们基准测试套件中的模型更大的模型**。

MAX78000 在不同模型复杂度上的推理延迟更具可变性（范围从 0.14 毫秒到 4.63 毫秒），表明其在不同模型复杂度之间具有更大的可扩展性。GAP8 在所有模型上都表现出最高的端到端延迟——平均比 MAX78000 慢 17 倍，尽管其计算能力与 MAX78000 相似（22.65 GOPs 对比 30 GOPs）。然而，GAP8 较大的闪存和 RAM 尺寸使其更适合部署大型模型或混合专家（MoE）架构。

#### 4.2.4 CPU 后处理

**尽管在基准测试研究中经常被忽视，但后处理操作可能会对端到端延迟和整体效率产生影响。** 例如，对于某些平台上的 YoloV1 的非最大抑制（NMS），MAX78000 使用 RISC-V CPU 进行后处理需要 3.82 毫秒，相比之下，实际推理时间仅为 2.62 毫秒。这**突显了最小化依赖 CPU 的后处理操作的重要性，并强调了我们基准测试的一个关键设计考虑因素：通过确保所有模型在各种平台上完全兼容 NPU**，我们旨在实现端到端延迟的公平比较，避免由于不支持的操作回退到 CPU 执行而导致的瓶颈或惩罚。

然而，在实际应用中，开发人员会构建针对特定目标平台优化的模型，因此有必要考虑支持的操作范围（某些 NPUs 上的支持范围非常有限），以及使用其他更强大或未修改的模型时可能出现的精度或性能权衡。

### 4.3 特定任务的考量

#### 内存限制和模型复杂度

内存容量显著影响每个平台上可部署的模型复杂度。例如，GAP8 的庞大内存（8MB RAM，20MB 闪存）使其能够部署比 MAX78000（512KB NPU 内存，128KB CPU 内存）加载更大的模型。这种差异对于需要更复杂模型的应用场景（如多类别目标检测或具有大词汇表集的音频分类）变得至关重要。

详细的内存 I/O 时序数据还提供了关于不同平台如何处理模型加载的额外见解。例如，MAX78000 **较长的内存 I/O 时间** （8.84-26.53 毫秒）**更适合于持续模型部署**。相比之下，HX-WE2 **相对较大的闪存内存和低延迟的内存 I/O** （0.03-1.11 毫秒），**以及较长的初始化时间** （2.56-2.60 毫秒），**使其更适合于连续推理或动态模型切换**。

#### 操作模式和功耗特性

支持不同操作模式的能力显著影响一个平台对特定应用的适用性。例如，MAX78000 在空闲（10.87-13.21 毫瓦）和推理（21.13-81.67 毫瓦）状态之间的功耗变化较大；因此，可以在占空比应用中有效利用功耗门控机制。需要对我们的基准测试平台上可用的各种低功耗模式进行进一步研究，包括唤醒时间、功耗门控机制和动态电压频率调整（DVFS）。

此外，具有非对称协处理能力的双 CPU 平台可以改善核心之间的任务分配——或者实现基于任务的 CPU 核心分层唤醒，从而带来进一步的节能优势。例如，MAX78000 结合了低功耗的 RISC-V 和计算能力较强的 Cortex-M4 核心，当与动态低功耗推理策略一起使用时，可以进一步优化模型部署期间的能量使用。

#### 精度要求和量化支持

每个平台支持的位宽是另一个针对特定应用部署的重要考虑因素。例如，MAX78000 对 1、2、4 和 8 位操作的支持使得在可以接受较低精度或模型适合于激进量化的应用中，能够实现高度优化的模型部署。相反，对于需要更高数值精度的应用，像 HX-WE2 这样的平台可能更有优势，它支持高达 32 位的浮点加速。

### 4.4 结果总结

我们在各种商用可用的 μNPU 平台上测量了不同模型架构的功耗和延迟。

> 我们发现，仅凭 GOPS（每秒十亿次操作）无法可靠地预测端到端延迟，而内存带宽对性能的影响巨大。

- 使用 Cortex-M4 CPU 的 MAX78000μNPU 在考虑 NPU 初始化的情况下，提供了最佳的整体效率，在所有模型上都实现了持续低于 30 毫秒的端到端延迟。然而，其性能主要受到内存限制，高达 90%的执行时间都用于内存 I/O 操作。
- HX-WE2 平台平均实现了比 MAX78000 快约 1.93 倍的端到端延迟，但功耗增加了约 3.13 倍。
- NXP-MCXN947 也提供了相对可比的（低于 30 毫秒）端到端延迟，具有快速初始化和内存 I/O；尽管其计算吞吐量较低，但在我们较低复杂度、内存轻量级的基准模型上展现出高效率。
- 通用 MCUs 表现出显著较低的效率，从而从实证角度验证了拥有专用神经硬件的优势。
- 在排除初始化开销（例如，对于需要连续运行的应用）的情况下，MILK-V 在效率方面总体排名最高，其较大的空闲功耗被快速端到端推理延迟所抵消。
- 值得注意的是，NXP-MCXN947 在 Autoencoder 模型上排名最高，无论是否考虑 NPU 初始化，都比其最近的竞争对手实现了超过 2 倍的效率提升。

关键平台差异化因素包括内存容量（影响模型复杂度）、功耗特性（例如，MAX78000 在空闲和推理状态之间显示出显著的功耗变化，这对于占空比应用是有益的）以及精度支持（例如，MAX78000 支持 1-8 位操作，而 HX-WE2 支持高达 32 位）。

### 4.5 未来发展方向

#### 推进硬件架构

> 开发具有更大片上缓存和改进内存吞吐量的下一代 μNPU 架构是一个显而易见的优先事项。这将：

1. 减少某些平台上观察到的显著内存 I/O 开销（例如，MAX78000）
2. 使得能够部署更大、更强大的模型或混合专家（MoE）架构，用于上下文感知推理

#### 优化模型权重加载

与硬件联合优化，改进模型架构和加载策略以最大化数据重用也是至关重要的。在某些平台上观察到的显著内存 I/O 瓶颈突显了需要针对 μNPU 的专门权重虚拟化、动态分配优化和预取策略。

#### 扩展算子支持

目前，大多数 μNPU 平台的算子支持极为有限，主要集中在 CNN 基础算子上。未来的设计应该纳入更广泛的算子集，以支持更多样化的模型架构，例如 Transformer。

#### 改进量化和模型压缩

细粒度的位宽量化以及其他非标准的模型优化在 μNPU 平台上也得不到充分的支持。这包括硬件和软件方面，现有的为资源受限设备设计的神经网络模型软件库通常也缺乏灵活性；例如，TFLite/LiteRT 仅支持 8 位整数和 16 位浮点权重量化。

#### 启用设备端训练

当前的 μNPU 平台仅支持神经网络推理，而不支持设备端训练。然而，设备端模型训练将实现个性化、持续学习和对动态分布变化的适应，而无需依赖云端处理——这对于数据隐私和远程部署至关重要。

未来的 μNPU 设计应该旨在支持量化设备端训练，这需要既有效率的训练算法，也需要硬件对反向传播的支持。

#### 标准化模型格式

我们各种基准测试平台支持的模型格式的异构性也是一个问题。硬件供应商应该朝着统一的模型格式迈进，以减少跨平台编译和部署的开销。

#### 开发准确的模拟器

最后，对于 μNPUs（以及 MCUs）来说，可靠的软件模拟器和预测推理延迟、功耗和内存使用的预测模型是显著缺失的。这样的工具将使开发人员能够在没有物理硬件的情况下优化部署，加速端到端的开发周期。

### 4.6 实用建议

> 我们为嵌入式开发人员和硬件设计人员提供以下建议：

#### 对于能效

MAX78000 在考虑 NPU 初始化的情况下，在能效方面大幅领先于其他 μNPU 平台，特别适合于电池供电的应用。为了延长电池寿命，可以考虑利用其**在空闲期间动态关闭系统部分区域的能力。**

#### 对于延迟敏感型应用

- HX-WE2 平台提供了低延迟，具有快速的 NPU 初始化、内存 I/O 和推理本身，使其最适合于需要响应式模型切换、实时适应变化条件或间歇性/占空比操作的应用。
- NXP-MCXN947 也实现了低端到端推理延迟，并且功耗预算显著较低，使其成为功耗受限工作负载的理想选择。

与此同时，对于延迟敏感但空间受限的应用，如果功耗不是主要限制因素，并且工作负载不需要频繁的 NPU 初始化/去初始化，更强大的 SoC 架构（如 MILK-V）也可能是合适的。未来的工作可以探索其他 SoC 平台[44, 45]的性能。

#### 对于大型模型

GAP8 庞大的内存使其特别适合于部署更大、更复杂的模型，或者实施基于操作条件的多专家网络模型切换方法，尽管其初始化时间较长且推理延迟较高。然而，如果功耗不是主要问题，MILK-V 的低推理延迟和大内存容量（支持 SD 卡）也可能使其成为一个强有力的选择。

#### 对于安全与效率

作为基于 Arm 的平台，NXP-MCXN947 的 CPU 包括 Arm 的 TrustZone[46]，它允许在安全和非安全区域之间进行硬件隔离。凭借其具有竞争力的端到端推理延迟（0.96-25.95 毫秒）和较低的功耗（22.91-36.69 毫瓦），它可能适合于对安全性要求较高但没有极端约束的应用。

未来的工作可以探索将其安全执行环境扩展到集成 μNPU 加速（通过 I/O 直通），以实现受保护但高效的神经网络推理。

#### 对于简单模型

对于足够简单的模型，像 STM32H7 这样的通用 MCU 可以在没有专用神经加速的情况下实现有竞争力的效能，从而消除了对专用硬件的需求。

### 4.7 限制

> 在解释我们的基准测试结果时，应考虑以下限制：

#### 频率标准化

虽然在所有平台上强制使用统一的 CPU 频率可以实现架构效率的直接比较，但它未能展示每个平台的峰值性能——许多基准测试平台可以以比评估中更高的频率运行。

#### 固定量化位宽

我们采用的标准化 INT8 量化方法虽然可以实现公平的比较，但它没有充分利用支持较低位宽操作（例如，MAX78000 的 1/2/4 位支持）或更高精度计算（例如，HX-WE2 的 FLOAT16/FLOAT32 支持）的平台的全部能力。我们还只关注量化作为减少模型大小的一种手段，排除了其他优化方法。

#### CPU 配置

我们还在实验中强制使用统一的 CPU 分频器设置；然而，许多平台支持可变分频器配置，这可能会影响整体效率曲线。

#### 模型适应性限制

为了保持跨所有平台的结构一致性，需要在模型优化方面做出妥协。针对特定平台的优化可能会产生与我们的标准化方法略有不同的效率曲线。

#### 算子支持

同样，通过确保所有模型在各种评估平台上完全兼容 NPU，我们排除了不受支持的 NN 算子的影响。

未来的工作应该研究在具有不同支持算子集的平台上，使用更复杂或未修改的模型以及针对每个平台精度优化的模型的性能扩展，以及平台特定架构优化的影响。

#### 开发工具链成熟度

我们专注于性能指标的研究。然而，另一个在选择目标平台时经常被忽视的因素，值得在未来的研究中关注，是开发工具链的成熟度和模型优化工具的完善程度。

## 五、相关工作

### 端到端基准测试

在资源受限和移动计算平台上对神经网络（NN）模型进行基准测试的研究逐渐增多。

- Japana 等人提出的 MLPerf 基准测试引入了首个行业标准的开源框架，用于评估配备多样化神经网络加速器和软件堆栈的移动设备上的 NN 性能[47]。
- Laskaridis 等人最近研究了大型语言模型（LLMs）在各种最先进的移动平台（包括 Android、iOS 和 Nvidia Jetson 设备）上的效率[48]。
- Reuther 等人探索了从蜂窝 GPU、FPGA 加速器到数据中心硬件等广泛范围的 NN 加速器的性能和功耗特性[49]。

然而，**以往关于 μNPU 平台的研究仅限于应用层面的性能评估[18, 19]或单一平台的独立基准测试** [6, 27]。此外，现有的单一平台基准测试通常忽略了端到端模型流程中的某些操作[7]。据我们所知，**我们的工作详细描述了首个对多个商用可用 μNPU 平台上 NN 模型进行并排和细致的基准测试研究。**

### 针对 MCUs 的 NN 加速器

NN 加速器在缓解传统 MCUs 用于 NN 推理的计算和内存瓶颈方面具有巨大潜力。除了商用加速器（例如 Arm Ethos-U55）之外，近期的研究还引入了一些更高效的定制设计。

- 例如，Venkataramani 等人设计了 RaPiD 加速器，专门用于超低功耗 INT4 推理，实现了 3-13.5 TOPS/W（平均 7 TOPS/W）的能量效率[50]。
- Conti 等人开发了 XNOR 神经引擎，这是一种用于二值神经网络的数字可配置硬件加速器 IP，集成到具有自主 I/O 子系统和混合 SRAM/标准单元存储器的 MCU 中[51]。

### 高效的设备端推理

在 MCUs 上部署 NNs 受到底层硬件内存容量和吞吐量的限制，功耗也常常成为瓶颈[52, 53]。众多研究探索了模型压缩[12, 54, 55]、设计更高效的 NN 算子和架构以降低资源使用[56–58]，以及基于输入复杂度和工作负载的自适应 NN 推理[59–61]。

此外，还研究了各种基于硬件的优化方法，例如并行数据流处理[21]。我们的工作旨在通过识别当前的最佳实践以及现有的瓶颈，进一步推动在 μNPU 平台上高效部署 NNs。

## 六、结论

我们对多种商用可用 μNPUs 上的各种 NN 模型进行了全面评估，揭示了预期趋势以及意外发现，为嵌入式神经计算领域的知识体系做出了贡献。专用神经加速的显著性能优势得到了清晰展示，**与通用 MCUs 相比，专用平台在某些情况下实现了高达两个数量级的能效提升**。

此外，我们还发现，**仅凭理论计算能力（GOPs）无法准确预测实际性能**。通过按阶段分解模型推理，揭示了**某些平台上的关键瓶颈（尤其是在内存 I/O 操作方面），并为未来的硬件和模型设计提供了重要见解**。我们敦促开发人员**在实际部署中权衡延迟、能效、模型复杂度和操作灵活性**，以实现最佳性能。

我们**开源了我们的基准测试框架**，希望其使用能够简化跨平台模型编译和评估。

## 参考文献

::: block-1
[1] Pietro Mercati and Ganapati Bhat. Self-Sustainable Wearable and Internet of Things (IoT) Devices for Health Monitoring: Opportunities and Challenges. IEEE Design and Test, 42(2):35–60, 2025.

[2] Sarah Condran, Michael Bewong, Md Zahidul Islam, Lancelot Maphosa, and Lihong Zheng. Machine Learning in Precision Agriculture: A Survey on Trends, Applications and Evaluations Over Two Decades. IEEE Access, 10:73786–73803, 2022.

[3] Chanwoo Kim, Dhananjaya Gowda, Dongsoo Lee, Jiyeon Kim, Ankur Kumar, Sungsoo Kim, Abhinav Garg, and Changwoo Han. A Review of On-Device Fully Neural End-to-End Automatic Speech Recognition Algorithms. In 2020 54th Asilomar Conference on Signals, Systems, and Computers, pages 277–283, 2020.

[4] Emil Njor, Mohammad Amin Hasanpour, Jan Madsen, and Xenofon Fafoutis. A Holistic Review of the TinyML Stack for Predictive Maintenance. IEEE Access, 12:184861–184882, 2024.

[5] Maxim Integrated. MAX78000. 2025. https://www.analog.com/en.

[6] Arthur Moss, Hyunjong Lee, Lei Xun, Chulhong Min, Fahim Kawsar, and Alessandro Montanari. Ultra-Low-Power DNN Accelerators for IOT: Resource Characterization of the MAX78000. In Proceedings of the 20th ACM Conference on Embedded Networked Sensor Systems, pages 934–940, 2022.

[7] Mitchell Clay, Christos Grecos, Mukul Shirvaikar, and Blake Richey. Benchmarking the MAX78000 artificial intelligence microcontroller for deep learning applications. In Real-Time Image Processing and Deep Learning 2022, volume 12102, pages 47–52. SPIE, 2022.

[8] Linghe Kong, Jinlin Tan, Junqin Huang, Guihai Chen, Shuaitian Wang, Xi Jin, Peng Zeng, Muhammad Khan, and Sajal K Das. Edge-computing-driven Internet of Things: A survey. ACM Computing Surveys, 55(8):1– 41, 2022.

[9] Ruijin Wang, Jinshan Lai, Zhiyang Zhang, Xiong Li, Pandi Vijayakumar, and Marimuthu Karuppiah. Privacy-preserving Federated Learning for Internet of Medical Things under Edge Computing. IEEE journal of biomedical and health informatics, 27(2):854–865, 2022.

[10] Cheng Wang, Zenghui Yuan, Pan Zhou, Zichuan Xu, Ruixuan Li, and Dapeng Oliver Wu. The security and privacy of mobile-edge computing: An artificial intelligence perspective. IEEE Internet of Things Journal, 10(24):22008–22032, 2023.

[11] Jinhyuk Kim and Shiho Kim. Hardware accelerators in embedded systems. In Artificial Intelligence and Hardware Accelerators, pages 167–181. Springer, 2023.

[12] Ji Lin, Wei-Ming Chen, Yujun Lin, Chuang Gan, Song Han, et al. MCUNet: Tiny deep learning on iot devices. Advances in neural information processing systems, 33:11711–11722, 2020.

[13] Swapnil Sayan Saha, Sandeep Singh Sandha, and Mani Srivastava. Machine learning for microcontroller-class hardware: A review. IEEE Sensors Journal, 22(22):21362–21390, 2022.

[14] Ji Lin, Ligeng Zhu, Wei-Ming Chen, Wei-Chen Wang, Chuang Gan, and Song Han. On-device training under 256kb memory. Advances in Neural Information Processing Systems, 35:22941–22954, 2022.

[15] Young D Kwon, Rui Li, Stylianos I Venieris, Jagmohan Chauhan, Nicholas D Lane, and Cecilia Mascolo. TinyTrain: resource-aware task-adaptive sparse training of DNNs at the data-scarce edge. arXiv preprint arXiv:2307.09988, 2023.

[16] Yushan Huang, Ranya Aloufi, Xavier Cadet, Yuchen Zhao, Payam Barnaghi, and Hamed Haddadi. Low-Energy On-Device Personalization for MCUs. In 2024 IEEE/ACM Symposium on Edge Computing (SEC), pages 45–58. IEEE, 2024.

[17] Erez Manor and Shlomo Greenberg. Custom Hardware Inference Accelerator for Tensorflow Lite for Microcontrollers. IEEE Access, 10:73484–73493, 2022.

[18] Guanchu Wang, Zaid Pervaiz Bhat, Zhimeng Jiang, Yi-Wei Chen, Daochen Zha, Alfredo Costilla Reyes, Afshin Niktash, Gorkem Ulkar, Erman Okman, Xuanting Cai, et al. Bed: A Real-Time Object Detection System for Edge Devices. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 4994–4998, 2022.

[19] Weining Song, Stefanos Kaxiras, Thiemo Voigt, Yuan Yao, and Luca Mottola. TaDA: Task Decoupling Architecture for the Battery-less Internet of Things. In Proceedings of the 22nd ACM Conference on Embedded Networked Sensor Systems, pages 409–421, 2024.

[20] Luca Caronti, Khakim Akhunov, Matteo Nardello, Kasım Sinan Yıldırım, and Davide Brunelli. Fine-grained hardware acceleration for efficient batteryless intermittent inference on the edge. ACM Transactions on Embedded Computing Systems, 22(5):1–19, 2023.

[21] Taesik Gong, Fahim Kawsar, and Chulhong Min. DEX: Data Channel Extension for Efficient CNN Inference on Tiny AI Accelerators. Advances in Neural Information Processing Systems, 37:43925–43951, 2025.

[22] Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer. How to Evaluate Deep Neural Network Processors: TOPS/W (Alone) Considered Harmful. IEEE Solid-State Circuits Magazine, 12(3):28–41, 2020.

[23] ARM. ARM Ethos-U Processor Series Brief, 2022. Accessed: 2025-0312.

[24] Marco Giordano and Michele Magno. A Battery-Free Long-Range Wireless Smart Camera for Face Recognition. In Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems, pages 594–595, 2021.

[25] Bakar, Abu and Goel, Rishabh and De Winkel, Jasper and Huang, Jason and Ahmed, Saad and Islam, Bashima and Pawełczak, Przemysław and Yıldırım, Kasım Sinan and Hester, Josiah. Protean: An energy-efficient and heterogeneous platform for adaptive and hardware-accelerated battery-free computing. In Proceedings of the 20th ACM Conference on Embedded Networked Sensor Systems, pages 207–221, 2022.

[26] Edward Humes, Mozhgan Navardi, and Tinoosh Mohsenin. Squeezed Edge YOLO: Onboard Object Detection on Edge Devices, 2023.

[27] Yushan Huang, Taesik Gong, SiYoung Jang, Fahim Kawsar, and Chulhong Min. Energy Characterization of Tiny AI Accelerator-Equipped Microcontrollers. In Proceedings of the 2nd International Workshop on Human-Centered Sensing, Networking, and Multi-Device Systems, pages 1–6, 2024.

[28] TensorFlow.org. TensorFlow Lite for Microcontrollers. https://www.tensorflow.org/lite/microcontrollers, 2022. Accessed: 2025-03-13.

[29] Changmin Jeon, Taesik Gong, Juheon Yi, Fahim Kawsar, and Chulhong Min. TinyMem: Boosting Multi-DNN Inference on Tiny AI Accelerators with Weight Memory Virtualization. In Proceedings of the 26th International Workshop on Mobile Computing Systems and Applications, HotMobile ’25, page 1–6, New York, NY, USA, 2025. Association for Computing Machinery.

[30] GreenWaves Technologies. GAP8 Product Brief, 2021. Accessed: 2025-03-12.

[31] Cristian Ramírez, Adrián Castelló, Héctor Martínez, and Enrique S. Quintana-Ortí. Communication-Avoiding Fusion of GEMM-Based Convolutions for Deep Learning in the RISC-V GAP8 MCU. IEEE Internet of Things Journal, 11(21):35640–35653, 2024.

[32] Julian Moosmann, Hanna Müller, Nicky Zimmerman, Georg Rutishauser, Luca Benini, and Michele Magno. Flexible and Fully Quantized Lightweight TinyissimoYOLO for Ultra-Low-Power Edge Systems. IEEE Access, 12:75093–75107, 2024.

[33] Himax Technologies. WiseEye2 AI Processor, 2025. Accessed: 2025-03-12.

[34] NXP Semiconductors. FRDM-MCXN947 Development Board, 2025. Accessed: 2025-03-12.

[35] STMicroelectronics. STM32H7A3ZI Microcontroller, 2025. Accessed: 2025-03-12.

[36] Tommaso Addabbo, Ada Fort, Marco Mugnaini, Valerio Vignoli, Matteo Intravaia, Marco Tani, Monica Bianchini, Franco Scarselli, and Barbara Toniella Corradini. Gravimetric system for enhanced security of accesses to public places embedding a mobilenet neural network classifier. IEEE Transactions on Instrumentation and Measurement, 71:1–10, 2022. [37] Espressif Systems. ESP32-S3 Datasheet, 2025. Accessed: 2025-03-12.

[38] Milk-V. Milk-V Duo, 2025. Accessed: 2025-03-12.

[39] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-All: Train One Network and Specialize it for Efficient Deployment, 2020.

[40] Seyyed Hossein Hasanpour, Mohammad Rouhani, Mohsen Fayyaz, and Mohammad Sabokrou. Lets keep it simple: Using simple architectures to outperform deeper and more complex architectures, 2023.

[41] Inc. Analog Devices. Motor Fault Sample Dataset. https://github.com/analogdevicesinc/CbM-Datasets/tree/main, 2024. Accessed: 2024-04-01.

[42] Tsung-Yi Lin, Peizhao Ma, Serge Belongie, and Fei-Fei Li. Microsoft COCO: Common Objects in Context, 2014. Accessed: 2025-03-12.

[43] Monsoon Solutions Inc. Monsoon High voltage power monitor. 2024. https://www.msoon.com/.

[44] Luckfox. Luckfox Pico. https://www.luckfox.com/Luckfox-Pico. Accessed: 2025-03-17.

[45] Canaan. K230. https://developer.canaan-creative.com/k230/zh/dev/00_hardware/K230_datasheet.html. Accessed: 2025-03-17.

[46] ARM. ARM TrustZone, 2025. Accessed: 2025-03-17.

[47] Vijay Janapa Reddi, David Kanter, Peter Mattson, Jared Duke, Thai Nguyen, Ramesh Chukka, Ken Shiring, Koan-Sin Tan, Mark Charlebois, William Chou, et al. MLPerf mobile inference benchmark: An industry-standard open-source machine learning benchmark for on-device AI. Proceedings of Machine Learning and Systems, 4:352–369, 2022.

[48] Stefanos Laskaridis, Kleomenis Katevas, Lorenzo Minto, and Hamed Haddadi. Melting point: Mobile evaluation of language transformers. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, pages 890–907, 2024.

[49] Albert Reuther, Peter Michaleas, Michael Jones, Vijay Gadepally, Siddharth Samsi, and Jeremy Kepner. Survey and benchmarking of machine learning accelerators. In 2019 IEEE high performance extreme computing conference (HPEC), pages 1–9. IEEE, 2019.

[50] Swagath Venkataramani, Vijayalakshmi Srinivasan, Wei Wang, Sanchari Sen, Jintao Zhang, Ankur Agrawal, Monodeep Kar, Shubham Jain, Alberto Mannari, Hoang Tran, et al. RaPiD: AI accelerator for ultra-low precision training and inference. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 153–166. IEEE, 2021.

[51] Francesco Conti, Pasquale Davide Schiavone, and Luca Benini. Xnor neural engine: A hardware accelerator IP for 21.6-fj/op binary neural network inference. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 37(11):2940–2951, 2018.

[52] Sayed Saad Afzal, Waleed Akbar, Osvy Rodriguez, Mario Doumet, Unsoo Ha, Reza Ghaffarivardavagh, and Fadel Adib. Battery-free wireless imaging of underwater environments. Nature communications, 13(1):5546, 2022.

[53] Yuchen Zhao, Sayed Saad Afzal, Waleed Akbar, Osvy Rodriguez, Fan Mo, David Boyle, Fadel Adib, and Hamed Haddadi. Towards batteryfree machine learning and inference in underwater environments. In Proceedings of the 23rd Annual International Workshop on Mobile Computing Systems and Applications, pages 29–34, 2022.

[54] Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Sarangapani. A comprehensive survey on model compression and acceleration. Artificial Intelligence Review, 53:5113–5155, 2020.

[55] Muhammad Zawish, Steven Davy, and Lizy Abraham. Complexity-driven model compression for resource-constrained deep learning on edge. IEEE Transactions on Artificial Intelligence, 5(8):3886–3901, 2024.

[56] Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, and Qun Liu. Reusing pretrained models by multi-linear operators for efficient training. Advances in Neural Information Processing Systems, 36:3248–3262, 2023.

[57] Jakub M Tarnawski, Amar Phanishayee, Nikhil Devanur, Divya Mahajan, and Fanny Nina Paravecino. Efficient algorithms for device placement of dnn graph operators. Advances in Neural Information Processing Systems, 33:15451–15463, 2020.

[58] Lingda Li, Robel Geda, Ari B Hayes, Yanhao Chen, Pranav Chaudhari, Eddy Z Zhang, and Mario Szegedy. A simple yet effective balanced edge partition model for parallel computing. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(1):1–21, 2017.

[59] Stefanos Laskaridis, Alexandros Kouris, and Nicholas D. Lane. Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions. In Proceedings of the 5th International Workshop on Embedded and Mobile Deep Learning, EMDL’21, page 1–6, New York, NY, USA, 2021. Association for Computing Machinery.

[60] Bita Darvish Rouhani, Azalia Mirhoseini, and Farinaz Koushanfar. Delight: Adding energy dimension to deep neural networks. In Proceedings of the 2016 International Symposium on Low Power Electronics and Design, ISLPED ’16, page 112–117, New York, NY, USA, 2016. Association for Computing Machinery.

[61] Noam Shazeer, Kayvon Fatahalian, William R. Mark, and Ravi Teja Mullapudi. HydraNets: Specialized Dynamic Architectures for Efficient Inference. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8080–8089, 2018.
:::
