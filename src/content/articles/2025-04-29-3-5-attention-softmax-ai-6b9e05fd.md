> 基于 Transformer 的模型（一种多头自注意力机制的神经网络架构）是高性能大型语言模型（LLMs）的核心，但它们相较于传统的深度神经网络（DNNs）需要更多的算力资源。

::: block-1
因此，**为了降低 LLMs 的计算成本和功耗，需要更高效的 Transformer 算法和加速器**。我们观察到，随着序列长度的增加，Transformer 自注意力机制中的关键操作 **softmax 运算成为了主要瓶颈**。

本文提出了 Cross-Road Softmax，这是**为 LLMs 推理中的注意力层所设计的优化算法**。
:::

我们通过 8 个自然语言处理基准测试进行了软件实验来评估其性能。此外，我们还**设计了基于 Cross-Road Softmax 的 Cross-Road Accel 加速器来加速自注意力层的 softmax 函数**。

我们**使用 RTL 语言实现了 Cross-Road Accel，并通过 Synopsys 设计编译器使用 Nangate 15nm 开源单元库进行合成**，以获取功耗和面积统计信息。

总的来说，**与现有的最先进的 Transformer 加速器相比，CrossRoad Accel 平均实现了约 3.5 倍的能效提升。**

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/01.webp)

## 本文目录

- 一、引言
- 二、背景和动机
  - A. Transformer 和注意力机制
  - B. 动机
- 三、CROSS-ROAD ACCEL
- 四、硬件实现
  - A. 整体架构
  - B. Cross-Road Softmax 实现
- 五、实验评估
  - A. 实验设置
  - B. 软件评估
  - C. 硬件评估
- 六、结论
- 参考文献

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/02.webp)

## 一、引言

> 基于 Transformer 的大型语言模型（LLMs）在深度神经网络的多个领域展现出了突破性的性能。其中一个成功的关键因素是自注意力机制。

我们观察到，自注意力机制的计算成本和功耗会随着序列长度的变化而变化。对于长度小于 1K 的情况，线性操作在自注意力机制中占主导地位。然而，随着最近的模型开始支持超过 1K 的长度，非线性操作已成为主要瓶颈。

我们发现，随**着序列长度的增加，softmax 计算成为了关键瓶颈**。为了解决这个问题，**我们提出了 Cross-Road Softmax，它通过选择性地应用独热编码** （one-hot encoding）和**基 2 softmax 方法来计算注意力层中的 softmax 函数**。

我们的独热 Softmax 设计**利用独热编码来跳过 softmax 计算，利用了 softmax 在向量中分配主导概率给离群值的特性**。通过适当的阈值（通过与知名 LLMs 和任务的实验确定），可以**在不需要完整 softmax 计算的情况下跳过该操作**。

对于**无法跳过 softmax 操作的向量，我们引入了基 2 Softmax，它将基转换为 2 以实现高效计算**。

总之，Cross-Road Softmax 结合了独热 Softmax 和基 2 Softmax 方法。此外，我们还设计了 Cross-Road Accel 硬件加速器来验证我们方法的优势。我们**进行了软件和硬件实验来评估这些方法和设计**。

总的来说，我们的贡献如下：

::: block-1
1. 我们提出了 **Cross-Road Softmax**，这是一种**新颖的 LLMs 推理方法**，具有两种模式：**独热 Softmax 和基 2 Softmax**，专门针对 LLMs 和 softmax 的特性进行优化。（独热编码是一种将分类变量转换为二进制向量的编码方式，例如将类别 1、2、3 分别编码为[1,0,0]、[0,1,0]、[0,0,1]，这样可以让模型更好地处理分类数据。）
2. 我们**对知名的基准测试进行了分析**，以描述 LLMs 中的 softmax 特性，并**确定我们方法的最佳阈值**。
3. 我们提**出了一种硬件友好的注意力加速器 Cross-Road Accel**，与现有的先进加速器相比，它在**能效方面平均提升了 3.5 倍**。
:::

## 二、背景和动机

### A. Transformer 和注意力机制

> Transformer 模型主要由堆叠的编码器或解码器模块组成。每个编码器模块包含两个阶段：多头自注意力（MHA）层和位置前馈网络（FFN）层。

MHA 层并行执行多个头的注意力机制，每个头由单头注意力块组成。这些块通过将输入标记与预训练权重相乘来生成查询（Query，Q）、键（Key，K）和值（Value，V）矩阵。缩放点积注意力机制（Scaled Dot-product Attention）通过将 Q 与 K 的转置相乘来计算注意力分数，然后使用 softmax 函数对这些分数进行归一化并转换为概率。softmax 对于将输入值转换为概率至关重要。

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{i=0}^{k-1} e^{x_i}}
$$

如公式（1）所示，softmax 函数通过归一化输入的指数来计算概率。它在注意力机制中是必要的，将 Q 和 K 之间的相关性转换为概率矩阵，也就是注意力分布。最终的注意力结果是通过将该分布与 V 矩阵相乘获得的，称为上下文。最后，将所有头的结果连接起来以匹配 MHA 层的输入维度。

### B. 动机

#### Transformer 层的运行时分解

> 在早期的 LLMs 中，标记长度通常小于 1K。当标记长度小于 1K 时，除了注意力之外，QKV 生成和 FFN 操作的计算负载在大多数 LLMs 中占主导地位。然而，LLMs 需要支持的参数数量和最大序列长度迅速增加。**随着 LLMs 能够处理的最大序列长度远远超过 1K，注意力的负载急剧增加**。

因此，加速注意力计算变得非常重要。**为了了解注意力层中各个组件的计算瓶颈，我们进行了性能分析**。我们的分析结果证实，随着序列长度的增加，**注意力层中 softmax 函数的计算负载显著增加**。

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/03.webp)

**图 1 展示了** 在 LLMs 中 **softmax 的计算负载随序列长度的增加而增加** 的趋势。X 轴表示序列长度，Y 轴表示运行时分解。

#### 传统硬件的局限性

> 因此，**许多研究试图实现高效且快速的硬件来计算 softmax 函数**。然而，由于 softmax 具有非线性特性，它将无理数 $e$ 的幂次进行划分，因此在硬件中实现它具有挑战性。因此，许多现有的硬件架构使用查找表（LUT）或 CORDIC 算法来实现指数函数。

- 然而，LUT 方法需要大量的存储器来实现高精度，这比逻辑电路实现消耗更多的功耗。
- CORDIC 方法当迭代次数足够高时相当准确，但随着迭代次数的增加，延迟会急剧增加。

```
NeruralTalk：简单科普 CORDIC 算法CORDIC 是一种用于高效计算三角函数、对数、乘法和除法等数学运算的数字算法。它通过一系列简单的移位和加法操作来逐步逼近目标值，非常适合在硬件资源有限的环境中使用，例如嵌入式系统和早期的计算机。
```

在本文中，我们提出了一种新的 softmax 函数计算方法。

## 三、CROSS-ROAD ACCEL

> 我们引入了一种新的计算 softmax 的方法——Cross-Road Softmax（十字路口 Softmax）。

在大型语言模型（LLMs）的推理过程中，我们只需要从 softmax 输出中获取相对概率，因此如果一个值显著大于其他值，那么精确的概率就不需要了。

```
NerualTalk：简单来说，不算所有的，只算部分。
```

在所提出的 Cross-Road Softmax 中，softmax 结果由以下两个模块之一生成。独热 Softmax 模块输出一个向量，该向量仅在最高值处为 1，其余位置为 0，通过使用独热编码来简化 softmax 计算。如下表。

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/04.webp)

**当最高值与次高值之间的差距超过预定义的阈值时，使用这种方法**。应用**独热编码可以提高性能并降低功耗，因为它消除了完整 softmax 计算的需求。**

因此，**设置一个合适的阈值对于最佳的性能和效率至关重要**。另一方面，如果最高值和次高值之间的差异没有超过阈值，则不执行独热 Softmax 操作。相反，在这种情况下将运行基 2 Softmax 模块。基 2 Softmax 模块基于将 softmax 中的非线性函数 $e^x$ 转换为硬件友好的形式。![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/05.webp)

**Cross-Road Softmax 的整个过程在算法 1 中** 进行了概述，其**架构在图 3 中** 进行了展示。

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/06.webp)

## 四、硬件实现

### A. 整体架构

> **图 2 展示了整体系统架构和 Cross-Road Accel**。如图 2 所示，我们**实现了一个名为 Core Wrapper 的硬件系统**，由直接内存访问（DMA）、SRAM、核心控制器、调度器和 Cross-Road Accel 组成。

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/07.webp)

在 Core Wrapper 内，核心控制器负责管理系统的状态，调度器则负责从/向 SRAM 进行数据的读/写操作，SRAM 的容量为 16KB。Cross-Road Accel 从调度器接收 Q、K、V 和有效信号作为输入，并加速缩放点积注意力操作。

在 Cross-Road Accel 中，**实现了两个 8x8 的 systolic arrays（一种用于矩阵运算的硬件架构**，可高效执行矩阵乘法），利**用输出 stationary（一种数据流方式，可以减少数据移动**，提升计算效率）进行操作。

```
NeuralTalk"Stationary" 意为静止不动，实际指的是【数据尽量不动】和【计算靠近数据】。1. 【数据尽量不动】在传统的数据处理中，数据往往需要频繁地在不同的存储设备、内存或计算节点之间移动，这会带来额外的延迟和能耗。而Stationary数据流动技术的核心思想是尽量让数据保持“静止”，即数据在存储位置上不频繁移动，而是通过计算逻辑的调整，让计算靠近数据，减少数据的物理传输。2. 【计算靠近数据】这种技术强调将计算逻辑部署到数据所在的位置，而不是将数据传输到计算中心。例如，在分布式存储系统中，当需要对某个节点上的数据进行处理时，直接在该节点上启动计算任务，而不是将数据传输到其他节点进行处理。
```

这些 systolic arrays 采用流水线技术以提高计算吞吐量。此外，分配控制器模块对 Cross-Road Softmax 的输出进行预处理，而后处理模块对最终输出（上下文）进行后处理。

### B. Cross-Road Softmax 实现

> Cross-Road Softmax 模块由四个子模块组成：检测分数、行级前两名、独热 Softmax 和基 2 Softmax。

检测分数模块和行级前两名模块负责 Cross-Road Softmax 模块的预处理。独热 Softmax 在接收到行级前两名模块的标志后激活。它将 16’hffff（十六进制表示的 1.0）作为注意力分布（AD）分配给存储在索引指针中的最大值的索引，而所有其他输出设置为 0（16’h0000）。

这种操作**将最大概率分配给最大输入的索引，其余为零，一个周期内完成**。另一方面，**基 2 Softmax 模块通过将移位和加法以及位切片操作的结果存储在缓冲区中以进行除法操作**。这些操作并行执行，与 PE 数组的 8x8 大小保持差异。当生成基 2 Softmax 标志时，执行除法操作，所需周期等于最大序列长度（例如，序列长度为 128 时需要 128 个周期）。

然而，如果独热 Softmax 标志开启，则基 2 Softmax 跳过分法并清空缓冲区以节省能源。

## 五、实验评估

### A. 实验设置

> 进行了两种评估：（1）硬件实现与软件相比的精度下降；（2）硬件在延迟、功耗和面积方面的性能。

为了评估精度下降，我们实现了两个知名的 LLM 模型：具有 2 层的 BERT-Tiny 和具有 24 层的 BERT-Large。Cross-Road Softmax 的实现使用了 PyTorch、Hugging Face、CUDA 和 NVIDIA GeForce RTX 3090 GPU。

精度测量也使用 PyTorch 进行。Cross-Road Accel 使用 Verilog HDL 实现，使用 Xilinx 的 Vivado XSIM 和 API 进行功能验证。与之前的 Transformer 研究类似，**我们的设计主要涉及乘积累加（MAC）操作。因此，我们通过测量 MAC 操作的周期数来计算吞吐量**，通常称为有效吞吐量。

### B. 软件评估

> 使用 GLUE 基准测试中的八项任务进行的实验表明，Cross-Road Softmax 的硬件实现导致的精度下降可以忽略不计，甚至略有提高。

**如表 II 所示。对于 GLUE 中最大的任务 MNLI，在 BERT-TINY 和 BERT-LARGE 模型中没有精度下降，甚至提高了多达 3%。**

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/08.webp)

### C. 硬件评估

> 硬件实现使用 Synopsys 设计编译器与 Compile-Ultra 选项和 Nangate 15nm 单元库进行合成。**Cross-Road Accel 在吞吐量、功耗和电路面积方面进行了评估**。

吞吐量定义为总 MAC 操作数除以处理时间，约为 1.29 TOPS（每秒万亿次操作）在 2GHz 频率下。Synopsys Primetime 验证了在此频率下没有时序违规。

Cross-Road Accel 消耗 283.47mW 的功率，实现了 1.29TOPS 的吞吐量，占地面积为 37728um²。Cross-Road Accel 的平均能效比以往知名作品[5]-[7]提高了约 3.5 倍。**评估结果总结在表 III 中**。

![文章插图](/articles/2025-04-29-3-5-attention-softmax-ai-6b9e05fd/images/09.webp)

## 六、结论

本文我们提出了 **Cross-Road Accel，它显著降低了自注意力机制中 softmax 操作的计算成本**。降低计算成本的关键是在**一定条件下选择性地跳过** softmax 计算。对于**必要的 softmax 计算**，我们**提出了更高效的硬件设计**。

我们在 LLMs 中实验了 Cross-Road Softmax，但我们的方法可以应用于任何使用 softmax 的模型和任务。总之，所**提出的硬件设计 Cross-Road Accel 在能效方面平均比现有的先进 Transformer 加速器提高了 3.5 倍**。

## 参考文献

1. Ashish Vaswani et al. "Attention is All you Need," In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.
2. Alex Wang et al. "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding," In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, November 2018. Association for Computational Linguistics. URL https://aclanthology.org/W18-5446, doi: 10.18653/v1/W18-5446.
3. Jacob Devlin, Ming-wei Chang, Kenton Lee, and Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), vol. 1. pp.4171–4186, 2019.
4. Chang Gao, Daniel Neil, Enea Ceolini, Shih-Chii Liu, and Tobi Delbruck. "DeltaRNN: A Power-efficient Recurrent Neural Network Accelerator," In Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA ’18), pages 21–30, 2018. Association for Computing Machinery. ISBN 9781450356145. doi: 10.1145/3174243.3174261.
5. Hanrui Wang, Zhekai Zhang, and Song Han. "SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning," In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 97-110, March 2021. doi: 10.1109/HPCA51647.2021.00018.
6. Tae Jun Ham et al. "ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention Mechanism in Neural Networks," In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 692-705, 2021. doi: 10.1145/3464432.3451656.
7. Yubin Qin et al. "FACT: FFN-Attention Co-optimized Transformer Architecture with Eager Correlation Prediction," In Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA ’23), Orlando, FL, USA, 2023. Association for Computing Machinery. ISBN 9798400700958. doi: 10.1145/3579371.3589057.
