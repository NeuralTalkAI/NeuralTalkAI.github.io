关键词：Systolic Array, Flash Attention, Microarchitecture, Accelerators

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/01.webp)

- SystolicAttention: Fusing FlashAttention within a Single Systolic Array
- https://arxiv.org/abs/2507.11331
- RTL 实现与 Python 软件栈：https://github.com/VCA-EPFL/FSA
- 本文 11195 字，阅读需 44 分钟，播客 8 分钟如下

> Transformer 模型严重依赖于缩放点积注意力（scaled dot-product attention, SDPA），该操作通常通过 FlashAttention 算法实现。然而，当前基于脉动阵列（systolic array，一种通过处理单元网络流式传输数据来高效执行矩阵乘法的硬件架构）的加速器在执行 FlashAttention 时面临重大挑战。

脉动阵列仅能在连续的大型矩阵乘法中实现高利用率，相比之下，FlashAttention 需要频繁交替进行矩阵乘法和 softmax（软 max 函数，一种将向量映射到概率分布的激活函数）操作。

::: block-1
脉动阵列与外部向量单元之间的频繁数据交换导致脉动阵列利用率低下。更严重的是，softmax 涉及大量非矩阵操作，这些操作并不适合在脉动阵列上执行。此外，脉动阵列上的矩阵乘法与向量单元上的 softmax 并发执行会导致寄存器文件和 SRAM（静态随机存取存储器，一种高速缓存）端口冲突，进一步降低性能。

为克服这些限制，我们提出了 **FSA（一种增强型脉动阵列架构）**，它能:mdnice-bold-italic[使整个 FlashAttention 算法在单个脉动阵列内运行——无需外部向量单元]。FSA 的核心是 S**ystolicAttention，这是一种新颖的调度算法**，它能将 FlashAttention 操作:mdnice-bold-italic[以细粒度、元素级重叠的方式映射到脉动阵列上]。这显著提高了阵列利用率，同时保留了原始浮点运算顺序以维持数值稳定性。
:::

我们**在可综合的 RTL** （寄存器传输级，硬件描述语言的一种抽象级别）**中实现了 FSA，并将其性能与最先进的商用加速器进行了评估**。结果表明，:mdnice-bold-italic[FSA 的注意力 FLOPs/s] （每秒浮点运算次数）:mdnice-bold-italic[利用率分别是 AWS NeuronCore-v2 的 1.77 倍和 Google TPUv5e 的 4.83 倍]，而:mdnice-bold-italic[面积开销仅约为 10%。]

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/02.webp)

## 本文目录

- 一、引言
- 二、背景与动机
  - 2.1 systolic array 上的矩阵乘法
  - 2.2 systolic array 优化
  - 2.3 FlashAttention 算法
  - 2.4 在基于 systolic array 的加速器上实现 FlashAttention
- 三、SystolicAttention 调度算法
  - 3.1 FSA 概述
  - 3.2 FSA 上的 rowmax
  - 3.3 FSA 上的 exp 和 exp2
  - 3.4 FSA 上的 rowsum
  - 3.5 使用 SystolicAttention 重叠计算
- 四、FSA 微架构
  - 4.1 微架构概述
  - 4.2 FSA 指令集
  - 4.3 Systolic 阵列控制器
- 五、FSA 内核编程模型
  - 5.1 类型安全张量
  - 5.2 FSA 指令的 Python API
  - 5.3 JIT 内核编译器
- 六、评估
  - 6.1 FlashAttention 性能
  - 6.2 精度分析
  - 6.3 FSA 面积开销
- 七、讨论
  - 7.1 脉动阵列上的非线性函数计算
  - 7.2 向量单元的必要性
  - 7.3 FSA 变体：面积优化设计
  - 7.4 FSA 的局限性
- 八、结论
- 参考文献

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/03.webp)

## 一、引言

人工智能的快速发展，特别是在自然语言处理和计算机视觉领域，很大程度上**得益于 Transformer 模型的成功** [13,17,38]。这些模型彻底改变了机器学习应用，从语言翻译和文本生成[31]到图像识别[29]和自动驾驶[14]。**然而，Transformer 模型的计算需求巨大[37]，需要专用硬件加速器才能实现实用的性能和能效**。:mdnice-bold-italic[随着这些模型在规模和复杂度上的不断增长[27]，对高效硬件加速的需求变得越来越迫切]。

> 脉动阵列[30]是一种专用硬件架构，旨在通过处理单元（PE）网络流式传输数据来高效执行矩阵乘法。在这些架构中，数据每时钟周期仅移动一跳，最大限度地减少了硬件数据移动开销，并实现了高可扩展性。由于其高效性和可扩展性，大型脉动阵列已被广泛应用于 AI 加速器中。

典型示例包括 Google 的 TPU[20,25,26]（TPUv6e 中为 256×256，v6e 之前为 128×128）和 AWS NeuronCore[9]（128×128）。

Transformer 模型中的一个关键操作是缩放点积注意力（SDPA），该操作通常通过 FlashAttention[16,35]算法实现。FlashAttention 通过将 Q、K 和 V 矩阵分割为小瓦片（tile，指数据分块），并将矩阵乘法与在线 softmax 计算融合，从而减少内存开销。**当将 FlashAttention 映射到基于脉动阵列的加速器时，矩阵乘法操作可由脉动阵列执行，而 softmax 相关操作则必须卸载到向量单元**。为了:mdnice-bold-italic[最大化硬件利用率，不同迭代中的矩阵乘法和 softmax 操作通常通过软件流水线并行运行]。向量单元:mdnice-bold-italic[必须] 能够**提供足够的每秒浮点运算次数（FLOPs/s），以跟上脉动阵列的速度。**

> :mdnice-bold-italic[然而，向量单元的 FLOPs/s 通常远低于矩阵乘法的 FLOPs/s，这导致了显著的性能瓶颈。] 例如，AWS NeuronCore-v2 的矩阵乘法性能超过 90 TFLOPs/s，而向量操作仅为 2.3 TFLOPs/s[8]。**当脉动阵列较大时，这种差异会变得尤为严重**，因为矩阵乘法的 FLOPs/s 随阵列大小呈二次方增长，而向量操作的吞吐量则保持有限。

除了矩阵乘法和向量 FLOPs 之间的不平衡外，在:mdnice-bold-italic[脉动阵列和 GPU 上实现 FlashAttention 时的另一个常见挑战是寄存器文件或 SRAM 端口冲突]。当脉动阵列和向量单元同时尝试访问寄存器文件/SRAM 的同一存储体时，矩阵计算和向量计算的并发执行会导致端口冲突，从而降低性能。尽管**现有工作（如 Mosaic[40]和 Virgo[28]）旨在缓解这种冲突，但它们并未完全消除冲突**。

:mdnice-bold-italic[如果完整的 FlashAttention 算法能够在单个脉动阵列上执行，而不依赖任何外部向量单元，上述问题就能得到解决]。在这种情况下，非矩阵乘法操作可以利用脉动阵列的矩阵乘法 FLOPs/s 来执行，并且由于所有操作都在同一阵列上执行，端口冲突也能被消除。实现这一点的**关键挑战** 是：

::: block-1
1. **【仅】使用脉动阵列高效执行【非】矩阵乘法操作**，如 exp（指数函数）、rowmax（行最大值）和 rowsum（行求和）函数。
2. **高效地重叠** 不同操作，以最大化脉动阵列利用率。
:::

为应对这些挑战，我们提出了 :mdnice-bold-italic[FSA，这是一种增强型脉动阵列架构，它能使整个 FlashAttention 算法在单个脉动阵列上执行，无需任何外部向量单元]。这基于以下关键观察：

- FlashAttention 中的 **exp** 操作输入值范围有限，这使得我们可以通过**线性插值来实现** 它。
- rowmax 和 rowsum 操作可以使用基于脉动阵列的归约（reduction，指将数组元素合并为单个值的操作）来实现，这是一个研究成熟的问题。由于行在脉动阵列上以流式方式生成，归约可以实时执行。
- 通过:mdnice-bold-italic[对数据流的仔细研究]，我们**发现 FlashAttention 算法的操作可以完全融合和调度，实现元素级操作重叠**，使脉动阵列能够并行执行多个操作，而无需等待先前操作完成。

我们进一步:mdnice-bold-italic[引入了 SystolicAttention 调度算法，该算法将 FlashAttention 映射到 FSA 上，实现元素级操作重叠]，显著提高阵列利用率。

::: block-1
我们的主要贡献如下：

1. 我们提出了 SystolicAttention，这是一种新的调度算法，它能在脉动阵列上实现 FlashAttention 中所有操作的广泛元素级重叠，同时保留原始浮点运算顺序以维持数值稳定性。
2. 我们设计了 FSA，这是一种增强型脉动阵列，能够执行完整的 FlashAttention 算法，无需依赖任何额外的向量单元。这完全消除了端口冲突，并在内存带宽充足的情况下能够维持理想性能。通过我们提出的 SystolicAttention，FSA 的 FLOPs/s 利用率分别是 AWS NeuronCore-v2 的 1.77 倍和 TPUv5e 的 4.83 倍。
3. 我们在可综合的 RTL 中完整实现了 FSA，并开发了一个 Python 编程接口，允许用户编写自定义 FSA 内核。RTL 实现和 Python 软件栈均已开源，地址为：https://github.com/VCA-EPFL/FSA。
:::

> 这项工作不仅:mdnice-bold-italic[证明了在单个脉动阵列上执行完整 FlashAttention 算法的可行性]，还为脉动阵列架构处理非线性函数和其他以前被认为不适合脉动阵列的复杂操作开辟了新的可能性。

## 二、背景与动机

在本节中，我们**首先简要介绍 systolic array** （一种用于高效矩阵运算的硬件架构，数据在处理单元间按节拍流动）以及基于**其提出的一些优化方法**。然后，我们将**介绍 FlashAttention 算法** 及其在 systolic array 加速器上**硬件实现所面临的挑战**。

### 2.1 systolic array 上的矩阵乘法

要在 systolic array 上执行矩阵乘法，必须将其中一个输入矩阵或输出矩阵**预加载** 到阵列中，而另一个输入矩阵则**流式输入**，输出矩阵则**流式输出**。当一个输入矩阵被预加载时，该阵列被称为权重固定或输入固定的 systolic array。为简洁起见，本文使用“输入固定 systolic array”来指代这两种情况。如果输出矩阵被预加载，则该阵列被称为输出固定 systolic array。由于大多数深度学习工作负载更倾向于输入（权重）复用，因此大多数基于 systolic array 的加速器采用输入固定架构。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/04.webp)

图 1 展示了一个输入固定 systolic array 的示例。它由 $N×N$ 个处理单元（PE，用于执行基本运算的硬件单元）组成，每个 PE 每周期可以执行一次浮点乘加（MAC，乘法和累加的组合运算）操作。其中一个输入矩阵被分块为 $N×N$ 的形状并预加载到阵列中，称为固定矩阵。另一个输入矩阵被分块为 $N×M$ 的形状并从输入 SRAM（静态随机存取存储器，一种高速缓存）流式输入，称为移动矩阵。输出矩阵因此被分块为 $N×M$ 的形状并流式输出。

由于输入固定阵列不会在内部保留输出矩阵，因此输出分块存储在位于阵列底部的小型专用累加 SRAM 中。**累加器单元从该累加 SRAM 读取值，将 systolic array 新计算的结果与现有值相结合，并将更新后的结果写回**。:mdnice-bold-italic[这种近内存累加方案在开源加速器（如 Gemmini [18]）和商业产品（如 AWS NeuronCore [9]）中被广泛采用。]

在计算开始时，预加载固定矩阵需要 $N$ 个周期。为确保正确性，移动矩阵的每一行在进入阵列前必须延迟 $0$ 到 $N-1$ 个周期，输出矩阵的每一列在读出前必须延迟 $0$ 到 $N-1$ 个周期。这引入了总计 $2N-1$ 个周期的同步开销。

- 总计算时间为 $M + 3N-1$ 个周期，因此阵列利用率为 $\frac{M}{M + 3N-1}$。
- **当 $M \gg N$ 时，这种预加载和同步开销会在计算过程中被有效分摊。**

### 2.2 systolic array 优化

已经提出了几种优化方法来减少 systolic array 中的同步开销。

Meissa [7] 引入了在 systolic array 的**每一列使用加法树来累加乘法结果**。**加法树的输入可以并行生成，从而消除了同步需求**。然而，这种方法:mdnice-bold-italic[仅限于整数加法]，因为加法树是使用进位保存加法器构建的，而进位保存加法器由于需要舍入和指数对齐，不适合浮点运算。

DiP [2] 提出了一种**对角线输入数据流来减少同步延迟**。但是，它需要将最左侧的 PE 连接到最右侧的 PE，这:mdnice-bold-italic[违反了 systolic array 中数据每周期只移动一跳的基本通信特性]。这种改动增加了硬件布线复杂度和延迟。

### 2.3 FlashAttention 算法

算法 1 展示了 FlashAttention 算法[16,35]，重点关注计算步骤，为清晰起见省略了内存加载/存储操作。遵循官方开源的 FlashAttention 实现，指数函数被实现为 $\exp(x)=2^{x\log_2 e}$，因为在硬件加速器上，exp2 通常比 exp 更高效。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/05.webp)

> 矩阵乘法用红色突出显示，而非矩阵运算（如归约和元素级计算）用蓝色显示。

- 在类 GPU 架构上，矩阵乘法在张量核心上执行，而非矩阵运算在 SIMT（单指令多线程，一种并行计算架构）核心上执行。
- 在基于 systolic array 的加速器上，矩阵乘法使用 systolic array 执行，其余操作由单独的向量单元或标量核心处理。

### 2.4 在基于 systolic array 的加速器上实现 FlashAttention

要在当前基于 systolic array 的加速器上实现 FlashAttention，:mdnice-bold-italic[必须将第一次矩阵乘法的结果从 systolic array 传输到向量单元以执行 softmax 相关操作]。然后，:mdnice-bold-italic[中间结果必须传输回 systolic array 以执行第二次矩阵乘法]。

算法 1 的第 15-16 行首先计算局部输出矩阵，然后将其累加到全局输出矩阵中。累加可以与第二次矩阵乘法融合，利用 systolic array 的累加器，因此可以将其视为单个矩阵乘法操作。

由于在 systolic array 内外移动数据时会产生额外的同步和预加载开销（如 2.1 节所述）。**为了最大化 systolic array 利用率，必须将分块大小设置得足够大以分摊这些开销**。然而，:mdnice-bold-italic[更大的分块大小也会导致矩阵乘法和向量 FLOPs 之间的不平衡加剧]，以及存储中间结果所需的 SRAM 需求增加。

因此，:mdnice-bold-italic[除了软件流水线之外，还需要【大量的】软件调优]，并且在基于 systolic array 的加速器上实现 FlashAttention 时，要实现高性能仍然具有挑战性。

## 三、SystolicAttention 调度算法

我们设计的目标是在单个 systolic array 上执行 FlashAttention 算法，从而提高非矩阵乘法操作的 FLOPs/s，并消除对外部向量单元的需求。

为了能够在单个 systolic array 上执行 FlashAttention 算法的:mdnice-bold-italic[【非】矩阵乘法操作]，我们:mdnice-bold-italic[在阵列顶部添加了一行比较单元以执行实时行最大值（rowmax）操作]。

- 此外，我们在**每个 PE 中引入了一个拆分（Split）单元来实现指数（exp）操作**。
- 为了使行最大值、指数和行求和（rowsum）操作能够**与矩阵乘法重叠** 执行，我们在 systolic array 中**加入了上行数据路径**。

在本节中，我们首先概述我们提出的用于加速 FlashAttention 算法的增强型 systolic array 微架构 FSA。然后，我们描述如何使用 FSA 执行 FlashAttention 中的非矩阵操作。最后，我们展示如何使用我们提出的 SystolicAttention 调度算法，在 FSA 中以 systolic 方式广泛重叠不同的计算。

### 3.1 FSA 概述

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/06.webp)

图 2 中用红色突出显示了 FSA 对标准 systolic array 的三项修改：

- **一行比较单元**：我们在阵列顶部添加一行比较单元，用于实时计算 systolic array 输出每一列的最大值。每个比较单元由一个浮点加法器和两个寄存器组成。一个寄存器 $old\_m$ 保存来自上一次迭代的最大值。另一个寄存器 $new\_m$ 存储当前迭代的临时最大值。$new\_m$ 的值根据比较结果更新，比较通过浮点加法器对两个值进行减法来实现。输入之间的差值也可以使用浮点加法器生成，:mdnice-bold-italic[这对于计算算法 1 的第 9 行很有用。]
- **每个 PE 中的拆分单元**：我们在每个 PE 中添加一个拆分单元，用于将输入浮点数拆分为其整数部分和小数部分。这用于:mdnice-bold-italic[以分段线性方式实现 exp2 操作。]
- **上行数据路径**：标准 systolic array 仅支持矩阵乘法的部分和沿一个方向传播。我们添加了一条额外的数据路径，用于沿相反方向传播部分和。:mdnice-bold-italic[这允许从阵列的顶部或底部开始计算。]

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/05.webp)

### 3.2 FSA 上的 rowmax

图 3 说明了融合的矩阵乘法和行最大值计算，对应于算法 1 的第 6-8 行。矩阵乘法 $S = Q_i K_j^T$ 在 systolic array 中沿上行数据路径执行，使得 $S$ 的行成为 systolic array 的列并从顶部输出。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/07.webp)

行最大值操作在比较单元中与 $S$ 的每一行的生成同时执行。当 $S$ 的行生成时，它们从阵列顶部重新流式传输，通过比较单元向下传播。当计算出每一行的最大值时，$S$ 已经驻留在 systolic array 中。

因此，可以立即执行算法 1 第 11 行中的减法操作 $N = S - new\_m$。这是通过从阵列左侧流式传输常量 1 作为被乘数，并从阵列顶部流式传输 $new\_m$ 作为加数来实现的。

### 3.3 FSA 上的 exp 和 exp2

在行最大值和减法操作之后，必须对矩阵 $N$ 应用元素级指数函数，如算法 1 的第 12 行所示：

$$
P = \exp\left(\frac{N}{\sqrt{d}}\right) = \exp2\left(\frac{\log_2 e}{\sqrt{d}} \cdot N\right)
$$

指数函数使用 exp2 实现，因为在硬件加速器上 exp2 通常更高效[23]。常量乘法 $\frac{\log_2 e}{\sqrt{d}} \cdot N$ 可以通过从阵列左侧流式传输常量 $\frac{\log_2 e}{\sqrt{d}}$ 作为被乘数来计算，因为减法操作后矩阵 $N$ 已经驻留在 systolic array 中。对于元素级 exp2 操作，我们使用分段线性插值来近似该函数[36]。

一个浮点数可以拆分为其整数部分和小数部分：$x = x_i + x_f$，其中 $x_i$ 是整数部分，$x_f$ 是小数部分[1]。这导致以下恒等式：

$$
exp2(x) = 2^{x_i + x_f} = 2^{x_i} \cdot 2^{x_f}
$$

由于 $N = S - rowmax(S)$，$N$ 中的所有元素都小于或等于零。这意味着 $x_f \in (-1, 0]$，因此 $2^{x_f} \in (0.5, 1]$。鉴于 $x_f$ 的这个有限范围，我们发现 8 段均匀分段线性插值足以在最终 FlashAttention 结果中实现约 $10^{-3}$ 到 $10^{-2}$ 的相对误差：

$$
exp2(x) \approx 2^{x_i} \cdot (slope_k \cdot x_f + intercept_k), k \in [0, 8)
$$

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/08.webp)

如图 4 所示，$x_f$ 的分段线性插值可以通过重用 systolic array 中的 MAC 单元来实现。

添加了一个简单的拆分单元，用于将输入 $x$ 分离为其整数部分和小数部分。拆分单元通过移位尾数位以使指数对齐到零来提取 $x_i$ 和 $x_f$。项 $2^{x_i}$ 仅影响最终结果中的指数位，实际上是将指数增加 $x_i$。

为了避免在 systolic array 中存储分段线性插值系数，我们分别从阵列的左侧和顶部分式传输 $slope_k$ 和 $intercept_k$ 值。我们进一步观察到所有截距都在范围(0.5, 1]内，因此它们的指数只能是 0 或-1。它们指数的最高有效位（MSB）可用于编码插值索引 $k$。这使每个 PE 能够基于 $k$ 更新其寄存器，而不需要额外的控制信号。

### 3.4 FSA 上的 rowsum

得益于 systolic array 中 $P = \exp\left(\frac{N}{\sqrt{d}}\right)$ 的原地指数计算，一旦生成 $P$ 的左上角元素，就可以开始行求和操作。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/09.webp)

如图 5 所示，$P$ 的行求和可以通过从阵列左侧流式传输常量 1 作为被乘数，并从阵列顶部流式传输 0 作为加数来计算。然后，第二次矩阵乘法 $O = P \cdot V$ 可以在 systolic array 中沿下行数据路径执行，在行求和操作开始后一个周期开始。一旦行求和 $local\_l$ 和输出 $local\_O$ 从底部离开阵列，它们就在累加器中与先前的值 $old\_l$ 和 $old\_O$ 累加，结果分别更新累加 SRAM 中的 $new\_l$ 和 $new\_O$。

### 3.5 使用 SystolicAttention 重叠计算

> 到目前为止，我们已经介绍了在 systolic array 中执行 FlashAttention 所需的关键操作，包括 rowmax、exp2 和 rowsum。**现在，我们描述如何以 systolic 方式重叠这些操作以最大化效率。**

FlashAttention 算法的每次迭代涉及两次矩阵乘法。为了避免 systolic array 和外部 SRAM 之间昂贵的数据移动，我们通过选择分块大小 $B_r = N\_COLS$ 和 $B_c = N\_ROWS = d$，在 $N\_ROWS × N\_COLS$ 的 systolic array 上构建计算。这允许第一次矩阵乘法的输出 $S = Q_i K_j^T$ 直接重新流式传输到阵列中，作为第二次矩阵乘法的输入。

在第一次矩阵乘法期间，$Q$ 和 $K$ 的头维度 $d$ 映射到 $N\_ROWS$，而序列长度 $B_r$ 映射到 $N\_COLS$。计算出 $S$ 后，我们将其从阵列顶部重新流式传输，将 $S$ 的 $B_c$ 维度映射到 $N\_ROWS$。然后，逐周期执行一系列元素级操作，将 $S$ 转换为 $N$，然后转换为 $P$。

对于第二次矩阵乘法 $O = P \cdot V$，$V$ 的 $B_c$ 维度映射到 $N\_ROWS$，并与 systolic array 内的固定矩阵 $P$ 收缩。

我们使用上行数据路径进行第一次矩阵乘法，使得 $S$ 的元素从阵列的左上角到右下角生成。然后，通过利用下行数据路径，一旦 $P$ 的左上角元素可用，我们就可以立即开始第二次矩阵乘法。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/10.webp)

图 6 说明了整个内循环的执行。第五个操作 $a = old\_m - new\_m$ 将先前最大值和当前最大值之间的差值从比较单元传递到累加器。该操作与常量乘法 $\frac{\log_2 e}{\sqrt{d}} \cdot N$ 并行运行，因为在原地常量乘法期间下行数据路径处于空闲状态。

**总体而言，每次迭代仅需要 $2 \cdot N\_COLS + 3 \cdot N\_ROWS + 10$ 个周期，这意味着在 $N × N$ 的 systolic array 上，处理 $N × N$ 的 FlashAttention 分块只需 $5N + 10$ 个周期。** 相比之下，在 naive 的 $N × N$systolic array 上，由于 2.1 节中讨论的预加载和同步开销，两次独立的矩阵乘法本身可能需要多达 $8N - 2$ 个周期。

算法 1 第 21 行中的重新缩放操作仅在每个外循环执行一次。它在第二次矩阵乘法完成后使用累加器执行。在我们的实现中，这个重新缩放步骤需要 $2N + 20$ 个周期，与内循环执行时间相比可以忽略不计。

## 四、FSA 微架构

FSA 对标准 systolic 阵列（一种通过处理元素网络流式传输数据来高效执行矩阵乘法的硬件架构）进行了修改，以支持 SystolicAttention 算法。**为了验证其可行性并量化相关的硬件开销，我们在 RTL** （寄存器传输级，硬件设计的一种抽象级别，描述寄存器之间的数据传输和逻辑操作）**中实现了 FSA，并设计了一套最小但灵活的指令集，能够执行完整的 FlashAttention。**

该 RTL 设计是用 Chisel HDL（一种基于 Scala 的硬件描述语言，用于构建可参数化的硬件生成器）实现的，并使用 Chipyard[3-5,15]框架进行片上系统（SoC）构建。**与之前的工作** （如 Gemmini[18]，其重点是灵活的设计空间探索）不**同，我们的目标是提供 FSA 的具体且完整的实现，以验证 SystolicAttention 算法的实用性及其硬件成本。**

本节概述了 FSA 的微架构，详细介绍了简化硬件实现的指令集设计，并描述了协调 SystolicAttention 算法执行的 systolic 阵列控制器。

### 4.1 微架构概述

FSA 微架构的概述如图 7 所示。FSA 通过 AXI4-Lite 接口（一种轻量级的 AXI 总线协议，用于低速控制通信）接收指令，并通过可配置数量的 AXI4 内存接口访问后备存储器。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/11.webp)

指令解码器将指令分为三类：加载、存储和计算。不同类别的指令异步执行，而同一类别的指令按顺序执行但可能交错进行。

一旦所需数据加载到 SRAM（静态随机存取存储器，一种高速内存）中，systolic 阵列控制器就会发出计算指令。为了简化控制逻辑，FSA 优先考虑 SRAM 访问而不是计算操作，使得计算指令的延迟在发出后完全确定。这种设计允许控制器根据每条指令的已知执行周期静态调度控制信号。

DMA 引擎（直接内存访问引擎，用于在内存和外设之间直接传输数据，无需 CPU 干预）遵循 iDMA 风格的接口[12]，并支持二维数据传输。每条 DMA 指令可以指定 SRAM 和后备存储器中的源地址和目的地址，以及传输大小和步长。

当有多个 AXI4 通道可用时，DMA 引擎会自动将传输划分为并行的 AXI4 事务，以最大化内存带宽利用率。

### 4.2 FSA 指令集

为了实现 FlashAttention 各种操作之间的细粒度重叠，FSA 中的 systolic 阵列需要比标准 systolic 阵列更复杂的控制逻辑。:mdnice-bold-italic[FSA 指令集的主要设计目标是确保计算指令以完全确定的方式执行，与内存访问延迟无关。] 这种**简化大大降低了控制复杂度。**

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/12.webp)

为实现这一目标，我们根据其 SRAM 数据依赖性定义了五种计算指令，如图 8 所示。**FlashAttention 内循环分为三个阶段**：

::: block-1
- 加载固定矩阵（Load Stationary）：将 Q 加载到 systolic 阵列中。
- 注意力分数（Attn Score）：使用 Q 和 K 执行第一次矩阵乘法，并与元素级在线 softmax 融合，在 systolic 阵列内原地计算 P 矩阵。在此步骤中还会计算指数和的对数。
- 注意力值（Attn Value）：使用 V 和 P 执行第二次矩阵乘法。
:::

:mdnice-bold-italic[FlashAttention 外循环] 还包括**两个额外阶段**：

::: block-1
- 倒数（Reciprocal）：从累加 SRAM 加载 P 的指数和的对数，并计算其倒数。结果存储在累加器中，作为下一阶段的缩放因子。
- 注意力对数和归一化（Attn LSE Norm）：从累加器加载输出矩阵 O，并应用倒数缩放因子。
:::

每条计算指令从 SRAM 读取一个输入瓦片（tile，指数据块），并向累加 SRAM 写入一个输出瓦片。**这种“一入一出”的设计确保计算指令一旦对应的 SRAM 瓦片准备好就可以执行，无需等待其他瓦片加载或处理。**

DMA 指令的结构与计算指令类似，由一个内存瓦片描述符和一个 SRAM 瓦片描述符组成。然而，它们使用更宽的位域来支持大内存地址空间。

### 4.3 Systolic 阵列控制器

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/13.webp)

FSA systolic 阵列控制器的微架构如图 9 所示。该控制器包括两个相同的基于计数器的有限状态机（FSMs，用于控制硬件操作序列的逻辑电路），每个负责为一条计算指令生成控制信号。

控制信号分为三类：

::: block-1
- PE 控制信号：定义 systolic 阵列的 PE（处理元素，阵列中的基本计算单元）内要执行的操作。
- 累加器控制信号：指定在累加器中执行的操作。
- SRAM 控制信号：管理执行计算指令所需的 SRAM 读写操作。
:::

由于计算指令的执行是完全确定的，计数器在每条计算指令开始时初始化，每个周期递增，完成后重置为零。因此，FSMs 可以仅根据指令类型和当前计数器值生成控制信号。

**为了简化控制逻辑开发，我们在 Chisel 和 Scala 之上实现了一种领域特定语言** （DSL，针对特定应用领域设计的编程语言）。这种 DSL :mdnice-bold-italic[允许设计人员指定每个周期应调度哪些控制信号]。 控制逻辑随后从 DSL 规范自动合成。

**双 FSM 架构支持两条计算指令的重叠执行**。例如，一旦“注意力分数”（Attn Score）指令生成 P 矩阵的第一个元素，“注意力值”（Attn Value）指令就可以开始。组合器单元合并来自两个 FSM 的控制信号，确保信号按正确顺序发出并避免冲突。

## 五、FSA 内核编程模型

AWS Neuron 内核接口（NKI）[10]提供了一种富有表现力的 Python 接口，允许用户为 AWS Neuron 设备编写自定义内核。它通过 Python API 提供指令级控制，兼具便利性。

**受 AWS NKI 的启发，FSA 提供了一个类似的、简化的 Python 库，用于编写自定义 FSA 内核**。该库由三个主要组件组成：类型安全张量、FSA 指令集的 Python API，以及轻量级即时（JIT）编译器。

### 5.1 类型安全张量

FSA Python 库实现了类 PyTorch[6,32]的张量抽象，用于表示多维数组。由于内存层次结构完全由用户管理，张量可以分配在以下三种内存空间之一：主内存、暂存 SRAM（scratchpad SRAM）或累加 SRAM（accumulation SRAM）。相应地，我们定义了三种张量类型：

- MTile - 用于主内存张量，
- STile - 用于暂存 SRAM 张量，
- ATile - 用于累加 SRAM 张量。

所有张量类型都支持 PyTorch API 的一个子集，包括形状（shape）、数据类型（dtype）、拆分（split）和反转（reverse）。通过明确区分张量类型，用户可以编写自定义内核函数，清晰地声明预期的输入和输出张量内存范围。

### 5.2 FSA 指令的 Python API

每条 FSA 指令都通过相应的 Python API 暴露。这些 API 设计为类型安全的，确保张量类型的正确使用。支持的完整 API 集如清单 1 所示。

清单 1：FSA Python 指令。

```
# DMA指令def store_tile(src: ATile, dst: MTile, ...)defload_tile(src: MTile, dst: STile, ...)defload_stationary(tile: STile, ...)# 计算指令defattn_score(K: STile, l: ATile, ...)defattn_value(V: STile, O: ATile, ...)defreciprocal(l: ATile, ...)defattn_lse_norm(O: ATile, ...)
```

在内部，这些函数使用输入张量的元数据（如形状、步长和数据类型）构造硬件指令。这种抽象允许用户编写高级内核，而无需管理低级硬件细节。

### 5.3 JIT 内核编译器

我们提供了一个轻量级 JIT 编译器，将带装饰器的 Python 函数编译为 FSA 二进制程序。这些程序由一系列二进制指令组成，可以提交到设备的指令队列执行。

使用 JIT 编译器在 FSA 中实现的 FlashAttention 内核如清单 2 所示。

```
import fsa as F@F.kernel(device="verilator_sim")def attention(Q: MTile, K: MTile, Vt: MTile) -> MTile:    # 分配输出张量    Ot: MTile = F.alloc_mem(..)    # 将大张量拆分为瓦片    Ot_MTiles = Ot.split(br, dim=-1)  # [d, br]    Q_MTiles = Q.split(br, dim=-2)  # [br, d]    K_MTiles = K.split(bc, dim=-2)  # [br, d]    Vt_MTiles = Vt.split(bc, dim=-1)  # [d, bc]    # Q、K、Vt的双缓冲    K_STiles = (F.alloc_spad(..), F.alloc_spad(..))    Vt_STiles = (F.alloc_spad(..), F.alloc_spad(..))    # 为累加结果分配空间    log_expsum = F.alloc_accum(..)  # [1, br]    Ot_ATile = F.alloc_accum(..)  # [d, br]    for i, Q_i in enumerate(Q_MTiles):        F.load_tile(Q_i, Q_STiles[i % 2])        for j, (K_j, Vt_j) in enumerate(zip(K_MTiles, Vt_MTiles)):            F.load_stationary(Q_STiles[i % 2])            F.load_tile(K_j, K_STiles[j % 2])            F.attn_score(K_STiles[j % 2], log_expsum)            F.load_tile(Vt_j, Vt_STiles[j % 2])            F.attn_value(Vt_STiles[j % 2], Ot_ATile)        F.reciprocal(log_expsum)        F.attn_lse_norm(Ot_ATile)        F.store_tile(Ot_ATile, Ot_MTiles[i])Ot = attention(Q, K, Vt)  # 运行模拟并返回结果O = Ot.to_numpy().T  # 主机端转置
```

JIT 编译器处理用@F.kernel 装饰的 Python 函数。该装饰器还指定目标设备（例如，基于 Verilator 的模拟器[39]），并自动处理设备与主机之间的通信。

在上面的示例中，主机使用 DRAMSim2[34]将输入张量加载到设备内存中，将编译后的指令分派到 FSA 设备，并在完成后检索输出张量。然后将输出转换为 NumPy[22]数组进行后处理。

由于 FSA 目前不支持硬件中的矩阵转置操作，输入矩阵 V 必须预先转置，并且生成的输出矩阵 O 以转置形式返回。

## 六、评估

### 6.1 FlashAttention 性能

我们将 FSA 的 FlashAttention 性能:mdnice-bold-italic[与最先进的商用脉动阵列加速器进行了比较，包括谷歌的 TPUv5e[21]和 AWS NeuronCore-v2[8]。] 我们选择 **TPUv5e 是因为它与 AWS NeuronCore-v2 具有相同的脉动阵列大小** （128×128）。为了公平起见，我们还将 FSA 配置为 128×128 的脉动阵列。所有加速器都使用 16 位浮点激活和 32 位累加。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/14.webp)

评估中使用的加速器配置汇总于表 1。我们评估了所有三种加速器上单个注意力头的 FlashAttention 前向传播性能，使用 128 的头维度，不使用因果掩码，序列长度范围从 2048 到 16384。

需要说明的是，我们**使用“FLOPs”表示浮点运算的数量，使用“FLOPs/s”表示每秒执行的浮点运算数量**。尽管所有加速器具有相同的脉动阵列大小，但它们的绝对性能仍然因时钟频率的不同而有所差异，而时钟频率又取决于所采用的特定节点技术。因此，我们将 FSA 模拟在 1GHz 下（如 6.3 节所述，这在 16nm 技术中很容易实现），并比较矩阵乘法的 FLOPs/s 利用率而非实际执行时间。

FlashAttention 内循环中的浮点运算数量由下式给出：

$$
\text{FLOPs/InnerLoop} = 2 \times 2 \times B_r \times B_c \times d
$$

其中每个矩阵乘法贡献 $2 \times B_r \times B_c \times d$。总运算数量为：

$$
FLOPs = \frac{SeqLen}{B_r} \times FLOPs/InnerLoop \times \frac{SeqLen}{B_c} = 4 \times SeqLen^2 \times d
$$

这与 FlashAttention 中使用的公式一致[16]。

::: block-1
我们计算每个加速器上的实际 FLOPs/s 如下：

$$
Achieved\ FLOPs/s = \frac{Total\ FLOPs}{Execution\ Time}
$$

FLOPs/s 利用率为：

$$
FLOPs/s\ Utilization = \frac{Achieved\ FLOPs/s}{Peak\ FLOPs/s}
$$
:::

我们使用两种商用加速器的官方 FlashAttention 内核实现，这些实现针对各自的架构进行了优化：

- 用于 TPUv5e 的 `jax.pallas.ops.tpu.flash_attention`[24]。
- 用于 AWS NeuronCore-v2 的 `neuronxcc.nki.kernels.attention.flash_fwd`[10]。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/15.webp)

实验结果如图 10 所示。**得益于高度重叠的计算，FSA 的平均 FLOPs/s 利用率分别是 TPUv5e 和 AWS NeuronCore-v2 的 1.77 倍和 4.83 倍**。此外，结果证实:mdnice-bold-italic[我们的 FSA 寄存器传输级实现] （硬件设计的一种底层实现方式）:mdnice-bold-italic[与 3.5 节中概述的【理论性能】非常接近，验证了我们寄存器传输级设计的正确性]。

### 6.2 精度分析

SystolicAttention 产生与标准注意力在数学上等效的结果，并保留浮点运算的顺序。然而，由于 exp2 计算中使用的分段线性插值，最终结果可能与商用加速器生成的结果略有不同。

在本节中，我们首先分析单个 exp2 函数的分段线性近似的精度。然后评估其对最终注意力结果的整体影响。

##### 6.2.1 分段线性近似的误差分析

由于 FlashAttention 中 exp2 函数的输入始终为负数，我们在所有负的标准 fp16 值上全面评估分段线性近似。我们排除了次正规值，因为在大多数加速器中它们通常被 flush 为零[19]。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/16.webp)

图 11 显示了 exp2 函数的分段线性近似的平均绝对误差（MAE）和平均相对误差（MRE），随近似中使用的段数变化。

> 平均绝对误差随着段数的增加而显著减小，而平均相对误差则保持相对稳定。这是因为近似中的断点是均匀分布的，而 fp16 表示本质上是对数的，使得接近零的值的相对误差更敏感。

我们选择**在 FSA 实现中使用 8 个段，这实现了 0.00014 的平均绝对误差和 0.02728 的平均相对误差**。尽管非均匀分布的断点可以进一步减小平均相对误差，但我们将此探索留待未来工作。

##### 6.2.2 整体 FlashAttention 精度

为了评估 FSA 上 FlashAttention 注意力的精度，我们将其与 PyTorch 中的 `torch.nn.functional.scaled_dot_product_attention` 进行了比较[6]。我们使用与性能评估相同的头维度和序列长度。输入矩阵 Q、K、V 是随机生成的，服从[0,1]范围内的正态分布。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/17.webp)FSA 上 FlashAttention 的整体精度结果如表 2 所示。平均绝对误差在 $10^{-4}$ 到 $10^{-3}$ 范围内，而平均相对误差在 $10^{-3}$ 到 $10^{-2}$ 左右。这些结果表明，**exp2 函数的分段线性近似对最终注意力结果的影响可以忽略不计**。

### 6.3 FSA 面积开销

**FSA 的面积开销** 来自以下附加组件：

::: block-1
- 向上数据路径，除了标准的向下路径外，还能使部分和向两个方向传播。
- 顶部一行 CMP 单元，负责识别每个脉动阵列列中的最大值。
- 每个处理元素（PE）内的 Split 单元，将输入值分为整数部分和小数部分以进行 exp2 操作。
:::

为了评估面积开销，我们在 16nm 商用工艺下以 1GHz 综合了 FSA 的脉动阵列部分（不包括 SRAM 和 DMA）。所得的面积分解如表 3 所示。

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/18.webp)

如表 3 所示，标准脉动阵列占总面积的 89.73%，而 FSA 的附加组件占 10.26%。:mdnice-bold-italic[开销的主要来源是向上数据路径和 Split 单元]，**分别占总面积的 4.83%和 4.78%。** 这是因为它们在 128×128 阵列的所有处理元素中重复。相比之下，CMP 单元仅占 0.65%，因为它们仅在阵列的顶行实例化。

## 七、讨论

### 7.1 脉动阵列上的非线性函数计算

> 我们观察到脉动阵列上 **exp2 函数的分段线性近似具有推广到其他非线性函数的巨大潜力。**

:mdnice-bold-italic[一般来说，如果一个非线性函数可以被分割成多个区间，每个区间都可以用线性函数很好地近似，那么就可以应用分段线性近似]。这使得能够**利用脉动阵列的高吞吐量 FLOPs/s 能力来计算非线性函数**，这大大**超过了传统向量单元的能力**。

### 7.2 向量单元的必要性

虽然 FSA 能够完全在脉动阵列上执行 FlashAttention 而不依赖于向量单元，但在完整的加速器系统中，仍然需要向量或特殊功能单元（SFU）来支持 FlashAttention 之外的激活函数。

然而，由于 FSA 将所有与注意力相关的计算从向量单元卸载，整体向量 FLOPs/s 需求可能会显著降低。

### 7.3 FSA 变体：面积优化设计

本文重点介绍了性能最佳的 FSA 变体，它使用向上和向下两种数据流来最大化吞吐量。

然而，:mdnice-bold-italic[在开发过程中，我们还探索了一种更简单的变体，具有单一数据流方向，类似于标准脉动阵列]。这种配置仍然支持 SystolicAttention 算法，但**会引入额外的延迟**：第二次矩阵乘法（O=PV）不能在 P 矩阵的第一个元素计算完成后立即开始，而必须等待整个 P 矩阵。

这导致内循环延迟从 5N+10 增加到 6N+10 个周期。尽管如此，这种面积优化设计仍然会比商用加速器提供更高的 FLOPs/s 利用率。

### 7.4 FSA 的局限性

尽管 FSA 为 FlashAttention 提供了高 FLOPs/s 利用率，但它:mdnice-bold-italic[不太适合 LLM（大语言模型）推理的解码阶段]。

**FSA 非常适合（1）LLM 训练和（2）LLM 推理的预填充阶段**，其中查询矩阵 Q 具有较长的序列长度，使得注意力计算是计算密集型的[33]。在这种情况下，**FSA 的高 FLOPs/s 吞吐量有效地加速了工作负载。**

相比之下，在解码阶段，查询 Q 的序列长度为 1，使得计算是内存密集型的。此外，由于 :mdnice-bold-italic[FSA 以 128×128 的 tiles 运行，Q 必须被填充以适应这种形状，导致大量的计算浪费和资源使用效率低下。]

## 八、结论

> 我们提出了 FSA，这是一种增强的脉动阵列架构，能够在单个脉动阵列内完成 FlashAttention 计算。

我们:mdnice-bold-italic[方法的核心是 SystolicAttention 算法]，它通过**精心的调度和静态数据流管理** 实现了:mdnice-bold-italic[所有 FlashAttention 操作的广泛元素级重叠]。

::: block-1
实验结果验证了我们的方法，显示出与最先进的商用加速器相比，在保持高精度和需要最小片上内存的同时，**FLOPs/s 利用率有显著提高**。
:::

除了证明在单个脉动阵列上执行完整的 FlashAttention 计算的可行性外，我们的工作还为脉动阵列架构有效处理以前被认为不适合此类架构的非线性函数和复杂操作开辟了新的可能性。

## 参考文献

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/19.webp)

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/20.webp)

![文章插图](/articles/2025-07-23-tpuv5e-4-83-flashattention-10-84c3a2b5/images/21.webp)

交个朋友
