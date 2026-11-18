关键词：On-Device GPU、Inference、Large Generative Model

> 受生成式人工智能发展的推动，大型机器学习模型已在图像处理、音频合成和语音识别等领域引发变革。

尽管基于服务器的部署仍是性能巅峰的核心，但出于隐私和效率考量，设备端推理的需求依然存在。

::: block-1
鉴于 GPU 是应用最广泛的**设备端机器学习加速器**，我们**提出 ML Drift—— 一个优化框架，它扩展了最先进的 GPU 加速推理引擎的能力**。ML Drift **使设备端能够执行参数数量** 比现有设备端生成式 AI 模型**多 10 到 100 倍** 的生成式 AI 工作负载。

ML Drift **解决了与跨 GPU API 开发相关的复杂工程挑战**，并**确保跨移动和桌面 / 笔记本平台的广泛兼容性**，从而便于在资源受限的设备上**部署复杂** 得多的**模型**。
:::

**我们的 GPU 加速 ML/AI 推理引擎** 相**对于现有的开源** GPU 推理引擎实现了**一个数量级的性能提升**。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/01.webp)

- https://arxiv.org/abs/2505.00232

## 本文目录

- 一、引言
- 二、相关工作
  - 通用 GPU 推理
  - 异构边缘设备的推理引擎
  - 大型生成模型推理
- 三、大型模型的 GPU 推理扩展
  - 3.1 逻辑张量和物理 GPU 对象
  - 3.2 张量虚拟化
  - 3.3 坐标转换
  - 3.4 设备专业化
  - 3.5 内存管理
  - 3.6 操作融合
  - 3.7 LLM 推理的阶段感知优化
  - 3.8 GPU 优化的 KV 缓存布局
- 四、性能评估
  - 4.1 大型扩散模型
  - 4.2 大型语言模型
- 五、结论
  - 未来工作
- 参考文献

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/02.webp)

加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

过去十年见证了大型生成模型的快速扩散，尤其在服务器端，它们革新了图像合成和自然语言处理等领域。尽管基于服务器的部署提供了大量计算资源，但设备端执行对于用户隐私、降低延迟、离线功能和降低服务器成本仍然至关重要。

在流行的设备端处理单元中，如 CPU、GPU 和 NPU，移动 GPU 因其广泛的可用性和内在的计算能力而脱颖而出，从而为这些复杂模型的部署提供了一种普遍的解决方案。

> 然而，**现代生成模型** 以其庞大的规模为特征，其**参数通常比它们的前身大 10 到 100 倍**，这为在资源受限的**移动 GPU 上部署** 带来了重大的工程**挑战**。

现有的设备端推理框架难以管理增加的内存和计算需求，这些需求可能大 1 到 2 个数量级，导致性能瓶颈和有限的可扩展性。**下表是 ML Drift 在移动设备** （高通 Adreno 750）和**笔记本电脑** （英特尔 Ultra 7 258V）**GPU 上的推理性能**。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/03.webp)

::: block-1
本文介绍了 **ML Drift**，这是一个**优化的推理框架**，旨在**解决大型生成模型在 GPU 上的部署挑战**。关键**创新包括**：

1. **张量虚拟化**，将**逻辑索引与物理 GPU 索引解耦**，并结合坐标转换以**实现灵活的内存布局和内核优化**；
2. 对**各种 GPU 的全面优化**，包括通过**特定于后端的着色器**、**内存管理**、操作**融合** 和阶段感知的 LLM 优化实现设备专业化。

此外，本文还介绍了 ML Drift 在**移动设备（Arm 和高通）**、桌面设备**（英特尔和 NVIDIA）** 和 **Apple GPU 上的全面性能评估**，展示了**相对于现有开源引擎一个数量级的改进**。
:::

## 二、相关工作

边缘设备的资源受限特性对机器学习模型的有效部署提出了重大挑战。这一限制推动了大量研究，旨在优化这些环境中的推理性能。这些工作大致可分为利用通用 GPU 的方法、专用推理框架，以及最近对在边缘设备上部署大型生成模型的复杂性的研究。

### 通用 GPU 推理

现有的 GPU 加速推理工作大多依赖于特定厂商的库，如 TensorRT [41] 和 ROCm [3]。**虽然这些解决方案在各自的硬件平台上能最大限度地提高性能，但它们本质上受限于架构特性，导致在不同 GPU 生态系统中的可移植性有限**。

> **ML Drift 的不同之处** 在于，它**优先考虑跨多种后端** （如 OpenCL [1]、Metal [20]、OpenGL ES [27]、Vulkan [2] 和 WebGPU [53]）**的性能优化，超越了其他解决方案的范围** [26, 31]。

### 异构边缘设备的推理引擎

针对边缘设备（尤其是利用异构硬件，如 CPU、GPU 和 NPU）的推理引擎优化一直是主要研究重点。硬件和操作系统厂商已通过提供专用 SDK 来响应这一需求，这些 SDK 旨在最大限度地提高其专有硬件上的推理性能。

值得注意的示例包括苹果 CoreML [6]、Arm 计算库 [7]、华为 HiAI 引擎 [19]、英特尔 OpenVINO [21]、联发科 NeuroPilot [32]、微软 DirectML [35] 和高通神经处理 SDK [44]。

> 尽管这些 SDK 在其指定的生态系统中提供了显著的性能优势，但其固有的厂商**依赖性引发了跨平台可移植性和部署灵活性的问题**。

与这些以厂商为中心的方法不同，各种跨平台框架应运而生，旨在提供更广泛的硬件和平台覆盖。这些框架努力抽象掉硬件复杂性，并提供更通用的部署解决方案。著名的与厂商无关的框架包括 ExecuTorch [33]、ONNX Runtime Mobile [36]、MACE [55]、MNN [24]、NCNN [51] 和 TensorFlow Lite [18]。

此外，基于中间表示（IR）的运行时，如 IREE [23]、TVM [8] 和 XLA [48]，专注于通过将机器学习模型通过中间表示降低为特定于硬件的代码来优化模型。

### 大型生成模型推理

大型生成模型的最新出现进一步加剧了对机器学习推理的需求。作为回应，机器学习推理社区提出了专门为服务器端部署设计的库，包括 LMDeploy [10]、SGLang [59] 和 vLLM [25]。

目前，正在努力实现大型生成模型的高效边缘推理，如 llama.cpp [16]、ollama [42] 和 torchchat [50] 等库所示。MLC LLM [38] 利用 TVM 运行时和 WebLLM [47] 来促进跨各种 GPU 后端的大型语言模型推理。

**针对移动 GPU 上大型扩散模型的特定实现级优化也在探索中** [9]。**其他推理加速研究** [4,56-58] 关注与我们工作正交的挑战，包括 DRAM 限制、NPU/CPU 协调、模型协作等。旨在改善边缘部署的模型压缩技术 [13,29,54] **与我们的方法兼容且相辅相成**。

## 三、大型模型的 GPU 推理扩展

> 为解决在各种 GPU 架构上部署大型生成模型的挑战，ML Drift 扩展了成熟的 GPU 加速 ML 推理引擎的基本架构 [26]。

我们**引入了一种新方法**，该方法**从手动优化的着色器模板在运行时执行动态代码生成**，将 ML Drift 归类为**优化数据布局和内核选择** 的专用推理框架。

为了便于优化和性能探索，我们**引入了张量虚拟化** （一种将物理 GPU 对象与逻辑张量表示抽象分离的方法），**实现了多样化的内存布局和内核配置，然后与坐标转换结合**，实现对这些**多样化内存布局的灵活访问**。

进一步的优化包括减少占用空间的内存管理策略、操作融合、LLM 推理的阶段感知优化，以及专用的 KV 缓存布局。这种架构范式使 ML Drift 能够在广泛的 GPU 平台上提升性能和可扩展性。

### 3.1 逻辑张量和物理 GPU 对象

在本文的上下文中，**逻辑张量指的是具有语义有意义轴的多维数组**，通常在数学或机器学习场景中构想。另一方面，**物理 GPU 对象是 GPU 上实现此逻辑张量的实际内存缓冲区或存储结构**，例如 GPU 缓冲区、图像缓冲区、纹理数组、2D 纹理和 3D 纹理。

由于神经网络中的中间张量通常对其轴没有固有含义，我们隐式地为它们分配语义。

::: block-1
具体来说，我们为每个轴分配以下语义（对于最多 5D 的张量）：

- 0D：标量
- 1D：线性
- 2D：H W（H 代表高度，W 代表宽度）
- 3D：HWC（H 为高度，W 为宽度，C 为通道）
- 4D：B H W C（B 为批次，H 为高度，W 为宽度，C 为通道）
- 5D：B H W D C（B 为批次，H 为高度，W 为宽度，D 为深度，C 为通道）
:::

**5D 张量的 D 轴仅用于 3D 卷积**，对于其他网络，D 通常设置为 1。我们在离线测试期间凭经验确定了每个边缘设备的最佳 GPU 对象。在运行时初始化期间，我们根据检测到的硬件为每个 GPU 内核选择预先确定的最佳 GPU 对象。根据所选的 GPU 对象，张量元素以不同的内存布局存储，因此需要不同的索引方法来访问相应的张量元素。

**受 PHWC4 内存布局 [26] 的启发** （该布局通过将数据组织成连续的 **4 通道切片来利用 GPU 的 4 元素 SIMD**，针对 GPU 缓冲区和纹理进行了优化），我们采用了多种 4 元素切片感知内存布局。例如，考虑一个大小为 **(B, H, W, D, C) 的 5D 张量**。当**用 2D 纹理实现时**，我们**使用内存布局 HSWBDC**，其中 **S 表示切片（[C/4]）**，C₄ 表示切片内的索引（c mod 4），以对 H 维度应用自动零填充。为了确保与 4 元素 SIMD 兼容，通道数不能被 4 整除的张量会被零填充。

我们**对权重张量的最佳 GPU 对象和内存布局进行了类似的探索**。

::: block-1
卷积和全连接权重最常用的布局是 OHWI 或 OHWDI，其中 I 和 O 分别是输入通道数和输出通道数，根据内核设计，它们会重新排列为 (G, Sₒ, O₄, H W D, Sᵢ, I₄) 布局的排列。

这里，Sᵢ 表示 I 轴的切片数，O₄ 和 I₄ 分别表示 O 和 I 切片内的元素。虽然 G 和 Sₒ 的具体值取决于内核设计，但它们的乘积 G・Sₒ 始终对应于 O 轴的切片数。
:::

**通过为权重张量战略性地选择最佳内存布局，与使用简单布局相比，我们在矩阵乘法操作中实现了高达 20% 的加速**，这些操作是卷积和全连接层的基础。

> 这种权重张量布局优化是高性能矩阵乘法的**基础**，也是 ML Drift **性能优势** 的**主要驱动力**。

### 3.2 张量虚拟化

依赖于所有张量的单一统一布局对于某些 GPU 内核可能不是最优的，这些**内核的内存访问与最初预期的模式不同**。**为解决这一限制，我们引入了 “张量虚拟化”**，这是一种受虚拟内存启发的新技术。张量虚拟化**将张量的逻辑表示与其在 GPU 上的物理存储解耦，允许张量使用各种类型和数量的 GPU 内存对象** （纹理、缓冲区等）实现。一个抽象层管理逻辑张量索引和物理 GPU 对象索引之间的映射，处理底层的张量数据碎片化和分布。这使内核作者无需关注低级内存管理问题，能够专注于算法逻辑。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/04.webp)

这种**虚拟化在张量表示中提供了显著的灵活性**。传统上，维度为 (1, 2, 3, 5) 的张量会被严格限制为 PHWC4 格式，需要大小为 (2, 3×⌈5/4⌉)=(2,6) 的 RGBA 纹理，而**我们的方法允许该逻辑张量任意映射到各种内存布局。如图 1 所示**，它可以实现为大小为 (2, 3, ⌈5/4⌉)=(2,3,2) 的 3D 纹理、大小为 (2×⌈5/4⌉, 3)=(4,3) 的 2D 纹理，或具有 2×3×⌈5/4⌉=12 像素的 1D 图像缓冲区，**具体取决于特定内核实现的最高效率**。

> 这种适应性对于跨具有不同并行特性的各种内核优化性能至关重要。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/05.webp)

**张量虚拟化** 还**便于高效处理单个张量需要由多个 GPU 对象表示的场景**。例如，我们的通用卷积内核需要**同时从多个纹理读取以优化内存缓存使用**。我们的抽象层通过允许大小为 (5, 2, 1, 7) 的张量使用四个大小为 (4, 2) 的纹理来无缝支持这一点，**每个纹理保存张量的一个切片，如图 2 所示**。这使得**复杂操作成为可能**，而这些操作在张量和 GPU 对象之间采用刚性一对一映射时可能不可行。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/06.webp)

尽管张量虚拟化增加了实现复杂性，但其性能开销可以忽略不计。具体来说，逻辑和物理索引之间的映射是在初始化时的着色器代码生成过程中建立的，从而最大限度地减少运行时延迟。

### 3.3 坐标转换

> 为了为生成着色器程序提供上述灵活性，消费着色器程序必须具备以灵活内存布局读取和写入张量的能力。

换句话说，我们**在着色器代码生成中引入** 了一个**预处理阶段**，使用辅助函数（例如 `args.src.Read (b, x, y, s)`）**将请求的张量元素坐标转换为底层实际的 GPU 内存对象坐标**，**如表 1 所示**。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/06.webp)

> 这使实现神经网络操作的着色器能够从 GPU 缓冲区和纹理中抽象出张量元素的访问，并简化了着色器程序的编写，而不必担心内存布局的组合爆炸问题。

与张量虚拟化类似，**坐标转换过程也在着色器代码生成期间** 执行。这种预处理方法避免了在 GPU 内核执行时增加任何运行时延迟。

### 3.4 设备专业化

> 为了促进**跨各种 GPU 架构的执行**，我们为各种 GPU 后端（例如 OpenCL、Metal、WebGPU）开发了一套着色器生成器，这些生成器将与平台无关的抽象转换为目标 GPU 语言。

::: block-1
在对目标 GPU 的属性进行**运行时分析后**，特别是**识别支持的厂商扩展** 并**确定最佳 GPU 对象存储类型**，**生成器对每个 GPU 操作执行一系列转换**。这些转换包括：

- 自适应内核选择：从针对特定 GPU API 和设备的一组候选实现中选择最快的实现。可能采用专用内核，例如 Winograd 快速卷积，以获得更好的性能。
- 利用厂商特定扩展：利用厂商扩展，例如专用矩阵乘法扩展，如 cl_arm_matrix_multiply。
- 语法转换：将 ML Drift 表示转换为特定于语言的着色器代码。
- 权重转换：预处理权重并以最佳内存布局将其存储在 GPU 内存中。
:::

### 3.5 内存管理

> 大型生成模型的内存占用主要受模型权重和中间张量的影响。虽然**模型权重大小是整体内存消耗的主要决定因素**，但**运行时内存的高效管理至关重要**。

先前对中间张量管理的研究表明，由于神经网络的顺序执行范式，这些张量不需要同时占用内存。因此，**实现高效的内存缓冲区重用方法是减少运行时内存占用的重要手段** [43]。

::: block-1
内存共享可以通过两种主要方法实现：

- 为生命周期不重叠的张量分配内存缓冲区
- 预分配一个大内存块并在其中为张量分配偏移量

这些策略**利用了** 神经网络的**有向无环图表示和顺序执行**。
:::

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/07.webp)

作为一个说明性示例，Stable Diffusion 1.4 对于半精度浮点数（FP16）激活需要 4.31 GB 的运行时内存。如图 3 所示，GREEDY BY SIZE 策略将运行时内存占用减少到 387 MB（节省 93%）。

```
科普：GREEDY BY SIZE 策略# 基本原理在编译器的运行时内存重用中，GREEDY BY SIZE 是一种贪心算法策略，旨在通过按大小顺序分配和管理内存来优化内存使用。# 实现方式该策略通常会根据内存块的大小进行降序排序。在内存分配时，优先选择最大的可用内存块来满足分配请求。这样做的目的是尽量减少内存碎片的产生，因为如果先分配小的内存块，可能会导致大的内存块被分割成许多小的碎片，从而无法满足后续较大的内存分配请求。# 应用场景这种策略在许多编译器相关的内存管理场景中都有应用。例如，在 TensorFlow Lite 中，对于中间张量的内存管理就可以采用 GREEDY BY SIZE 策略。通过分析中间张量的大小和使用时间等信息，利用该策略来决定如何分配和重用内存缓冲区，以减少推理引擎占用的总内存。# 优缺点优点：实现相对简单，能够在一定程度上有效地减少内存碎片，提高内存的利用率，从而优化运行时的内存使用，减少内存分配和释放的开销。缺点：它是一种贪心算法，可能无法保证在所有情况下都能得到最优的内存分配方案。例如，在某些特定的内存分配请求序列下，可能会导致一些大的内存块被过早地分配，而后续又有一些较小的分配请求无法被满足，即使这些小请求的总和小于剩余内存的大小，但由于内存碎片的存在，无法找到连续的足够大小的内存块来分配。
```

### 3.6 操作融合

操作融合（即算子融合）是一种常见的优化，它将多个内存绑定操作合并为单个内核，以减少内核启动开销和内存传输。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/08.webp)

如图 4 所示，当 ML Drift 检测到元素级操作、张量重新排序操作或残差连接的序列时，会自动应用操作融合。

> 除了自动化优化策略外，我们还**结合分析来识别性能瓶颈**，并**手动实现优化** 以实现高效的大型模型推理。

作为注意力块内的一个具体实例，我们设计了一个自定义内核来将旋转嵌入与查询（Q）、键（K）和值（V）投影的布局转换相结合。这包括将查询投影从初始布局 $(B, 1, S, h_q・d_h)$ 转换为结果布局 $(B・h_{kv}, S・h_q/h_{kv}, d_h)$。这里，B 表示批次大小，$h_kv$ 表示 KV 头数，S 表示序列长度，$h_q$ 表示查询头数，$d_h$ 表示头维度。这种优化的 QKV 布局对于跨各种注意力机制（包括多头、多查询和分组查询注意力）的高效推理至关重要。

### 3.7 LLM 推理的阶段感知优化

尽管使用驻留在 GPU 内存中的相同权重，ML Drift 区分了 LLM 推理的预填充和解码阶段，这是由于它们根本不同的性能特征所必需的。

对于涉及外部权重的操作，例如计算查询、键、值、输出投影以及前馈网络中的线性层，计算密集型的预填充阶段受益于专用的 GPU 量化内核。该内核将浮点输入激活转换为 8 位整数并计算所需的量化参数，从而使后续内核能够利用具有预量化权重的快速 int8 指令，并对输出激活进行反量化。

相反，**内存绑定的解码阶段通过将输入激活量化直接集成到操作内核中来优化**，这种方法减轻了内存传输开销，从而提高了整体性能。

> 此外，ML Drift 根据输入数据的特征为每个阶段选择专用的 GPU 内核。

**处理较长输入序列的预填充阶段受益于高度优化的卷积内核**，而解码阶段（其特征是单个 Token 的迭代生成）更有效地由全连接内核提供服务。

### 3.8 GPU 优化的 KV 缓存布局

ML Drift 在 LLM 推理中使用卷积内核进行矩阵乘法。KV 缓存作为卷积权重，以与 QKV 布局转换兼容的布局存储（第 3.6 节）。K 缓存使用 OHWI 布局（$O = 缓存大小，I=d_h$），表示 $K^T$ 用于 $QK^T$ 计算。V 缓存使用维度反转的 OHWI 布局（$O=d_h，I = 缓存大小$），确保涉及 V 的卷积产生所需的注意力输出布局 $(B・h_{kv}, S・h_{q}/h_{kv}, d_h)$。

## 四、性能评估

> 本节对 ML Drift 在大型文本到图像模型和大型语言模型上跨多种平台的性能进行了全面评估。

我们**提供了 ML Drift 的 OpenCL、Metal 和 WebGPU 后端在一系列硬件上的详细基准测试**，包括移动 GPU（**Arm Mali 和高通 Adreno**）、桌面 / 笔记本 GPU（**英特尔和 NVIDIA**）以及 **Apple Silicon**。

### 4.1 大型扩散模型

我们使**用开源模型 Stable Diffusion 1.4 评估了 OpenCL、Metal 和 WebGPU 后端的性能**，配置为使用 **FP16 权重进行 FP16 推理**。该流水线包括文本编码器、UNet 和 VAE 解码器。基准测试**在 Apple、Arm、英特尔和高通的一系列 GPU 上进行**。

#### 移动 GPU

**ML Drift 的 OpenCL 后端是 Android 平台的主要执行引擎**。在配备高通 Adreno 740 GPU 的三星 S23 Ultra 上，我们生成 512×512 图像（20 次迭代）的端到端延迟为 10.96 秒。这比之前报道的亚 12 秒基准 [9] 提高了 8%，比另一项 14.42 秒的测量 [46] 提高了 26%。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/09.webp)

**图 5 详细说明了 Stable Diffusion 1.4 在 Android 上的延迟**，展示了各种移动 GPU 上单个模型组件的性能。ML Drift OpenCL 在配备高通 Adreno 750 的三星 S24 上将延迟降低到 9 秒以下。

#### 桌面 / 笔记本 GPU

在配备**英特尔 Meteor Lake Ultra 7 165U 的 Windows 笔记本** 电脑上，我们的 **OpenCL 和 WebGPU 后端表现出显著优于使用 DirectML 的 ONNX Runtime 的性能优势** [37]。

具体来说，如**表 3 详细所示，ML Drift OpenCL 实现了 2.7 倍的加速，而 WebGPU 实现了 1.3 倍的加速**。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/10.webp)

为了进一步突出 ML Drift 的能力，我们与**英特尔在较新的 Lunar Lake Ultra 7 288V 上报告的性能进行了有针对性的比较** [22]。在我们最接近的可用平台 Lunar Lake Ultra 7 258V 上，**ML Drift OpenCL 生成 512×512 图像** （20 次迭代）**耗时 3.4 秒**，比英特尔为 288V 报告的 3.89 秒**提高了 14.4%。**

#### Apple Silicon

**ML Drift 的 Metal 后端为 Apple Silicon 提供了优化的执行**，从而实现了更快的扩散模型推理。在 M1 Ultra 和 MacBook Pro M4 Pro（20 核 GPU）上使用 Stable Diffusion 1.4 进行性能测试，运行时间分别为 3.86 秒和 5.34 秒。这些数字表明，**与 Apple 的 CoreML** Stable Diffusion 中观察到的 5.03 秒和 6.16 秒**相比 [5]，性能有了显著提升**。

### 4.2 大型语言模型

为了评估 ML Drift 在大型语言模型上的推理性能，我们使用四个开放权重模型进行了基准测试：Gemma 2B [15]、Gemma2 2B [14]、Llama 3.2 3B [34] 和 Llama 3.1 8B [30]。

这些**评估跨越了移动和桌面平台**，能够**与成熟的 LLM GPU 推理解决方案** （如 llama.cpp [16]、ollama [42]、torchchat [50]、MLC LLM [38] 和 MLX LM [39]）进行比较。

**ML Drift 主要对激活使用 FP16，NVIDIA 平台除外**，由于 **OpenCL 驱动程序限制，使用单精度浮点数（FP32）**。

ML Drift 实现了两种量化策略：

- q8（所有权重的每通道 int8 量化）
- 8/4/4（混合精度每通道量化，注意力为 int8，嵌入 / 前馈权重为 int4）

---

相比之下，其他开源解决方案通常使用 GGUF [17] q4 组量化，其产生的模型大小介于 ML Drift 的 q8 和 8/4/4 方法之间。

对于 LLM 基准测试，ML Drift 实现关闭了投机解码 [28] 和 FlashAttention [11]，并在每个 Token 生成后执行 CPU/GPU 同步。所有评估使用 1280 个 Token 的固定上下文长度，包括 1024 个预填充和 256 个生成 Token 。

#### 移动 GPU

我们在 Android 设备上使用 ML Drift 的 OpenCL 实现与 llama.cpp 的基准测试工具和 MLC Chat 演示应用程序进行了基准测试，**评估了五个移动 GPU 的性能**：

- 高通 Adreno 830（小米 15 Pro 16GB RAM）
- Adreno 750（三星 S24 8GB RAM）
- Adreno 740（三星 S23 Ultra 8GB RAM）
- Arm Immortalis-G720（vivo X100 Pro 16GB RAM）
- Arm Mali-G715（谷歌 Pixel 9 12GB RAM）

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/11.webp)

**在高通 Adreno GPU 上，如图 6 所示**，**ML Drift 的 OpenCL 后端** 在 Token **预填充方面比其他开源** LLM 推理解决方案**实现了 5 到 11 倍的加速**。

此外，ML Drift 的 OpenCL 后端在 Llama 3.2 3B 的 Token 生成速度上比高通 AI Hub 基准测试 [45] 高出 29%，使用相同的 8/4/4 量化方案。

类似地，在 Arm Mali GPU 上（目前 llama.cpp 不支持），ML Drift 在预填充和生成阶段都表现出与 MLC LLM 相当的加速。例如，运行 Llama3.2 3B 时，ML Drift 使用 q8 量化平均达到 791 Token / 秒的预填充和 12.5 Token / 秒的解码，而 MLC LLM 使用 q4f16 分别达到 89.2 和 11.2 Token / 秒。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/12.webp)

Token 预填充速度在很大程度上不受量化方案的影响，这意味着这是一个计算受限的过程。然而， **Token 生成速度通过量化优化表现出高达 1.9 倍的性能提升，表明这是一个内存受限阶段，其中内存带宽是关键约束**。上面**表 2 提供了一系列 GPU 和模型的综合性能指标**。

#### 桌面 / 笔记本 GPU

> 除了移动平台，ML Drift 还通过 OpenCL 和 WebGPU（通过 Dawn [52] 原生执行）接口促进在集成和专用 GPU 上的执行。

**尽管这些 GPU 中存在专用的矩阵乘法加速单元** （如 NVIDIA 的 Tensor Cores [40]），**但当前** 的 **ML Drift 实现利用这些硬件特性的能力有限**，这一限制**源于 OpenCL 和 WebGPU 中** 对这些功能的**厂商支持有限**。

因此，如经验观察到的，计算受限的 Token 预填充阶段经历了 4 到 7 倍的性能下降。

> **尽管存在这一限制**，ML Drift **仍保持有竞争力** 的 Token 生成速度，证明了其即使**在没有硬件加速的情况下的鲁棒性**。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/13.webp)

具体来说，**如图 7 所示，在 NVIDIA GeForce RTX 4090 上，ML Drift 的 OpenCL 实现** （使用 FP32 精度）**与基于 CUDA 的 llama.cpp 相比，性能降低了 5% 到 25%，** 这是通过 llama.cpp 的基准测试工具测量的。

然而，**在对 q4f16 量化模型进行推理时，ML Drift 的 OpenCL 保持了优于 ollama 和 torchchat（使用 CUDA 后端）的性能优势**。对于支持 8 位协作矩阵扩展的 GPU（如英特尔 Ultra 7 平台），ML Drift 表现出更快的 Token 预填充速度，如表 4 所示。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/14.webp)

ML Drift 的 WebGPU 后端通过在 RTX 4090 上支持 FP16 和 FP32 精度提供了操作灵活性。比较评估表明，在模型推理期间，**WebGPU 后端相对于 ML Drift 的 OpenCL 实现存在明显的性能下降**。需要进一步研究来调查这种性能下降的根本原因，并探索潜在的优化策略。

Apple Silicon 对于配备专有 Apple Silicon 芯片的 Apple 设备，ML Drift 通过其 Metal 后端促进 LLM 推理，通常使用 FP16 精度。对 Apple M4 Pro 20 核 GPU 上 ML Drift 的性能进行了严格分析，详细结果如图 8 所示。

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/15.webp)

实验结果表明，**ML Drift 的 Metal 实现在 Token 预填充阶段表现出性能优势**。值得注意的是，对于 Gemma2 2B，ML Drift 比 llama.cpp 基准测试显示出 14% 的速度提升，比 MLX LLM 快 20%。在所有测试模型中，**ML Drift 在 Token 生成方面始终优于 llama.cpp 和 ollama，并且对于 Gemma 模型也比 MLX LLM 快**。

这与移动设备上的经验观察一致，比较 q8 和 8/4/4 量化模型时，在预填充阶段发现了性能差异。尽管量化方法之间的性能差异仍然存在，但在 Apple M4 Pro 上，与移动平台相比，其幅度有所减小。Apple Silicon 架构的高内存带宽特性有助于减轻性能差异。

## 五、结论

> 在这项工作中，我们介绍了 ML Drift，这是一个新的推理框架，旨在促进大型生成模型在各种 GPU 上的高效部署。

通过**张量虚拟化** 将逻辑张量索引与物理 GPU 索引解耦，ML Drift 在**内存布局** 和**内核优化** 方面实现了无与伦比的灵活性。结合设备专业化、内存管理策略和操作融合，我们的框架带来了**显著的性能提升**。

**在各种硬件平台** （包括移动、桌面和 Apple Silicon GPU）上的**综合评估验证了 ML Drift 的有效性**，展示了相**对于现有开源解决方案一个数量级的改进**。ML Drift 展示了执行比最新最先进的 GPU 推理引擎大 1 到 2 个数量级的工作负载的能力。

### 未来工作

**未来** 的工作将侧重于通过**整合高级量化** 技术（如子通道量化）和**稀疏性来扩展 ML Drift 的能力**。随着移动 GPU 越来越多地**集成** 用于 ML 工作负载的**专用指令** （包括通过厂商扩展可访问的点积和矩阵乘法），深入探索这些功能对于进一步的性能增强至关重要。

为了便于针对各种模型优化 ML Drift，将进行**消融研究以量化每个优化组件的开销和单独贡献**。此外，**评估将扩展到更新的扩散模型** （如 Stable Diffusion 3.5 [49]）和基于 **Transformer 的架构** [12]，这将涉及研究为这些模型类型定制的先进最先进构建块的集成。还将**探索与异构处理器** 的**高效互操作性** 和混合匹配操作，利用**零拷贝缓冲区**。

## 参考文献

::: block-1
[1] The OpenCL C++ 1.0 Specification. The Khronos Group Inc., 2019.

[2] Vulkan 1.4.309 - A Specification(with all registered extensions). The Khronos Group Inc., 2025.

[3] Advanced Micro Devices, Inc. AMD ROCm Software. https://www.amd.com/en/products/software/rocm.html. [Online; accessed March 17, 2025].

[4] Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. LLM in a flash: Efficient Large Language Model Inference with Limited Memory, 2024.

[5] Apple Inc. Core ML Stable Diffusion. https://github.com/apple/ml-stable-diffusion. [Online; accessed March 17, 2025].

[6] Apple Inc. Core ML. https://developer.apple.com/documentation/coreml. [Online, accessed March 17, 2025].

[7] Arm Ltd. Compute Library. https://www.arm.com/products/development-tools/embeddedand-software/compute-library. [Online; accessed March 17, 2025].

[8] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. TVM: An automated End-to-End optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), pages 578–594, Carlsbad, CA, 2018. USENIX Association.

[9] Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, and Matthias Grundmann. Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations, 2023.

[10] LMDeploy Contributors. LMDeploy: A Toolkit for Compressing, Deploying, and Serving LLM. https://github.com/InternLM/lmdeploy, 2023.

[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022.

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. CoRR, abs/2010.11929, 2020.

[13] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, 2023.

[14] Gemma Team. Gemma 2: Improving Open Language Models at a Practical Size, 2024.

[15] Gemma Team. Gemma: Open Models Based on Gemini Research and Technology, 2024.

[16] ggml.ai. llama.cpp. https://github.com/ggmlorg/llama.cpp. [Online; accessed March 17, 2025].

[17] ggml.ai. GGUF. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md. [Online; accessed March 17, 2025].

[18] Google LLC. LiteRT Overview. https://ai.google.dev/edge/litert. [Online, accessed March 17, 2025].

[19] Huawei Technologies Co., Ltd. Huawei HiAI. https://developer.huawei.com/consumer/en/hiai. [Online; accessed March 17, 2025].

[20] Apple Inc. Metal. https://developer.apple.com/documentation/metal. [Online; accessed March 17, 2025].

[21] Intel Corporation. OpenVINO. https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html. [Online; accessed March 17, 2025].

[22] Intel Corporation. Intel Core Ultra Series 2 Media Deck. https://download.intel.com/newsroom/2024/client-computing/Intel-Core-UltraSeries-2-Media-Deck.pdf, 2024.

[23] IREE Organization. IREE. https://iree.dev/. [Online, accessed March 17, 2025].

[24] Xiaotang Jiang, Huan Wang, Yiliu Chen, Ziqi Wu, Lichuan Wang, Bin Zou, Yafeng Yang, Zongyang Cui, Yuezhi Cai, Tianhang Yu, Chengfei Lv, and Zhihua Wu. MNN: A Universal and Efficient Inference Engine. ArXiv, abs/2002.12418, 2020.

[25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[26] Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk, Mogan Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, and Matthias Grundmann. On-Device Neural Net Inference with Mobile GPUs. In Proceedings of the CVPR Workshop on Efficient Deep Learning for Computer Vision, 2019.

[27] Jon Leech, editor. OpenGL ES Version 3.1. The Khronos Group Inc., 2016.

[28] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding, 2023.

[29] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, WeiMing Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. In MLSys, 2024.

[30] Llama Team. The Llama 3 Herd of Models, 2024.

[31] Huynh Nguyen Loc, Youngki Lee, and Rajesh Krishna Balan. DeepMon: Mobile GPU-based Deep Learning Framework for Continuous Vision Applications. Proceedings of the 15th Annual International Conference on Mobile Systems, Applications, and Services, 2017.

[32] MediaTek Inc. NeuroPilot. https://neuropilot.mediatek.com. [Online; accessed March 17, 2025].

[33] Meta Platforms, Inc. ExecuTorch. https://pytorch.org/executorch-overview. [Online; accessed March 17, 2025].

[34] Meta Platforms, Inc. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. https://ai.meta.com/blog/llama-3-2-connect2024-vision-edge-mobile-devices/. [Online; accessed March 17, 2025].

[35] Microsoft Corporation, Inc. DirectML Overview. https://learn.microsoft.com/en-us/windows/ai/directml/dml. [Online; accessed March 17, 2025].

[36] Microsoft Corporation, Inc. Get started with ONNX Runtime Mobile. https://onnxruntime.ai/docs/get-started/with-mobile.html. [Online; accessed March 17, 2025]. [37] Microsoft Corporation Inc. Stable Diffusion Optimization with DirectML. https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion. [Online; accessed March 17, 2025].

[38] MLC team. MLC-LLM. https://github.com/mlcai/mlc-llm. [Online; accessed March 17, 2025].

[39] MLX Community. MLX LM. https://github.com/ml-explore/mlx-lm. [Online; accessed March 17, 2025].

[40] NVIDIA Corporation. NVIDIA Tensor Cores. https://www.nvidia.com/en-us/data-center/tensor-cores. [Online; accessed March 17, 2025].

[41] NVIDIA Corporation. TensorRT. https://developer.nvidia.com/tensorrt-gettingstarted. [Online; accessed March 17, 2025].

[42] ollama. ollama. https://github.com/ollama/ollama. [Online; accessed March 17, 2025].

[43] Yury Pisarchyk and Juhyun Lee. Efficient Memory Management for Deep Neural Net Inference. In Proceedings of the MLSys Workshop on Resource-Constrained Machine Learning, 2020.

[44] Qualcomm Inc. Snapdragon Neural Processing Engine SDK. https://www.qualcomm.com/developer/software/neural-processing-sdk-for-a. [Online; accessed March 17, 2025].

[45] Qualcomm Inc. QualComm AI Hub Llama-v3.2-3B-Chat. https://aihub.qualcomm.com/models/llama_v3_2_3b_chat_quantized. [Online; accessed March 17, 2025].

[46] Qualcomm Inc. World’s first on-device demonstration of Stable Diffusion on an Android phone. https://www.qualcomm.com/news/onq/2023/02/worlds-first-on-device-demonstration-of-stable-diffusion-on-android. [Online; accessed March 17, 2025].

[47] Charlie F. Ruan, Yucheng Qin, Xun Zhou, Ruihang Lai, Hongyi Jin, Yixin Dong, Bohan Hou, Meng-Shiun Yu, Yiyan Zhai, Sudeep Agarwal, Hangrui Cao, Siyuan Feng, and Tianqi Chen. WebLLM: A High-Performance In-Browser LLM Inference Engine, 2024.

[48] Amit Sabne. XLA: Compiling Machine Learning for Peak Performance, 2020.

[49] Stability AI, Ltd. Introducing Stable Diffusion 3.5. https://stability.ai/news/introducing-stablediffusion-3-5. [Online; accessed March 17, 2025].

[50] Team PyTorch. Introducing torchchat: Accelerating Local LLM Inference on Laptop, Desktop and Mobile. https://pytorch.org/blog/torchchat-local-llminference. [Online; accessed March 17, 2025].

[51] Tencent Holdings Ltd. NCNN. https://github.com/Tencent/ncnn. [Online; accessed March 17, 2025].

[52] The Dawn and Tint Authors. Dawn, a WebGPU implementation. https://dawn.googlesource.com/dawn?pli=1. [Online; accessed March 17, 2025].

[53] W3C. WebGPU. https://www.w3.org/TR/webgpu. [Online; accessed March 17, 2025].

[54] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. In Proceedings of the 40th International Conference on Machine Learning, 2023.

[55] Xiaomi Corporation. MACE. https://github.com/XiaoMi/mace. [Online; accessed March 17, 2025].

[56] Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, and Xuanzhe Liu. LLMCad: Fast and Scalable On-device Large Language Model Inference, 2023.

[57] Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, and Xuanzhe Liu. Fast On-device LLM Inference with NPUs. In International Conference on Architectural Support for Programming Languages and Operating Systems, 2024.

[58] Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, and Haibo Chen. PowerInfer-2: Fast Large Language Model Inference on a Smartphone. ArXiv, abs/2406.06282, 2024.

[59] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient Execution of Structured Language Model Programs, 2024.
:::

![文章插图](/articles/2025-05-31-gpu-ml-drift-layout-llm-3f4e0808/images/16.webp)
