关键词：SOLE、**Transformer**、:mdnice-bold-italic[软硬件协同设计]、**Softmax 优化**、**LayerNorm 优化**、低精度计算

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/01.webp)

- SOLE: Hardware-Software Co-design of Softmax and LayerNorm for Efficient Transformer Inference
- https://arxiv.org/pdf/2510.17189
- 11000 字，阅读 40 分钟，播客 21 分钟

- [端侧 NPU 的 LLM 测试时计算扩展：硬件感知块量化与 LUT 优化实现 19.0×GEMM与 2.2×Softmax 加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892974&idx=1&sn=ef25e0315fc32464d4a9739fce435928&scene=21#wechat_redirect)
- [性能提高 162 倍、能耗降 74 倍：基于 VEXP 扩展 RISC-V 指令集的 Softmax 加速，面积开销仅 1%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888122&idx=1&sn=fc8c4aa4aa191272e7fe061f8fc569bf&scene=21#wechat_redirect)
- [GPU上的高效softmax近似](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885412&idx=1&sn=0a6a1c0ff5128cba5d39d4deaf3540b1&scene=21#wechat_redirect)

> 本文提出 SOLE，一种针对 Transformer 中 **Softmax 和 LayerNorm 的软硬件协同设计方案**，旨在解决这两种非线性操作的**效率瓶颈**。

现有基于函数逼近的优化方法存在内存开销大、需重训练补偿误差的问题，而 SOLE 由 E2Softmax 和 AILayerNorm 组成，实现**低精度计算与低比特存储的结合**。

::: block-1
- E2Softmax 通过:mdnice-bold-italic[对指数函数输出进行 log2 量化，将中间数据压缩至 4 位]，并用**近似对数除法替代浮点除法**，无需乘法器和查找表；
- AILayerNorm 基于:mdnice-bold-italic[动态压缩和幂次因子]，实现 **8 位数据缓冲和 4 位乘法的低精度统计计算**。
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/02.webp)

::: image-caption
图 2. SOLE 的整体架构。上图为软件层的 E2Softmax 和 AILayerNorm 算法设计，下图为硬件层的定制计算单元设计，展示了从算法逼近到硬件实现的全流程优化
:::

实验表明，SOLE 无需额外训练即可维持推理精度，在 CV 和 NLP 任务中精度损失可忽略：

- 硬件层面，**相比 GPU，SOLE 对 Softmax 和 LayerNorm 分别实现 36.2 倍和 61.3 倍平均提速**，能效提升显著；
- 相比现有定制硬件，其 Softmax 和 LayerNorm 的**能效分别提升 3.04 倍、3.86 倍**，**面积效率分别提升 2.82 倍、3.32 倍**，且可**与量化等压缩方法兼容**。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/03.webp)

::: image-caption
表III SOFTMAX 与 Softermax、NN-LUT 及 GPU 在能耗和面积方面的唯一对比
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/04.webp)

## 关键问题

### 问题 1：超大规模模型与极端量化场景下的精度稳定性问题

> 论文中 SOLE 在 DeiT、BERT 等中大规模模型上实现了可忽略的精度损失，但面对 GPT-4、ViT-22B 这类**超大规模 Transformer，或低资源设备下的极端量化场景** （如低于 4 位存储），其 log2 量化和低精度统计计算的误差累积是否会突破可接受阈值？**现有方案未涉及该场景的验证，如何保证超大规模模型与极端场景下的精度稳定性？**

- 从核心机制来看，**SOLE 的精度稳定性源于其对 Softmax 和 LayerNorm 特性的针对性优化**：
- 针对超大规模模型（如 ViT-22B、GPT-4），虽论文未直接验证，但 **SOLE 无需重训练的特性可适配大规模模型的训练成本约束**，且其低精度计算未改变模型原有数学逻辑，仅优化计算与存储方式，误差累积大概率仍处于可控范围——超大规模模型的参数冗余性反而可能进一步容忍这种微小误差。
- 对于低于 4 位的极端量化场景，论文未开展相关实验，但 log2 量化的离散特性决定了进一步降比特可能导致精度损失突破阈值。不过，**SOLE 的量化方案可通过调整量化粒度（如动态适配比特宽度）进行扩展**，结合其乘法-free、LUT-free 的硬件友好设计，理论上可通过微调量化参数（如对数量化的步长）适配更低比特场景，但需额外实验验证误差边界。

### 问题 2：现有硬件集成成本与参数动态调整的适配灵活性问题

> SOLE 的定制硬件单元（E2Softmax Unit、AILayerNorm Unit）依赖专用数据路径和低比特存储设计，而现有主流加速器（如 GPU 张量核、Transformer ASIC）已**固化成熟的计算架构，其与这些现有硬件的无缝集成成本如何**？当 Transformer 的**序列长度、头数等参数动态调整时，定制硬件的适配灵活性** 是否会下降，进而抵消其能效优势？

- 集成成本方面，SOLE 的硬件设计天然具备低集成门槛：其 **E2Softmax Unit 和 AILayerNorm Unit 以移位器、加法器、小型 LUT 为核心，无复杂专用组件**，且输入输出均为 8 位整数格式，完全兼容现有 GPU 张量核、Transformer ASIC 中的 8 位整数 MAC 数据路径，可直接作为插件模块集成，**无需重构现有硬件架构**。
- **参数动态调整（序列长度、头数等）的适配性上，SOLE 通过两大设计保障灵活性**：这种设计使 SOLE 在参数调整时，**【仅】需适配缓冲大小和并行度，无需修改核心计算逻辑，不会抵消其能效优势。**

## 本文目录

- 关键问题
  - 问题 1：超大规模模型与极端量化场景下的精度稳定性问题
  - 问题 2：现有硬件集成成本与参数动态调整的适配灵活性问题
- 一、引言
- 二、相关工作
  - A. Transformers 中的主要非线性运算
  - B. 非线性运算的加速方法
- 三、SOLE 软件设计
  - 3.1 预备知识
  - 3.2 E2Softmax 算法
  - 3.3 AILayerNorm 算法
- 四、SOLE 硬件设计
  - 4.1 E2Softmax 单元
  - 4.2 AILayerNorm 单元
- 五、实验评估
  - 5.1 实验设置
  - 5.2 算法性能
  - 5.3 加速比
  - 5.4 硬件效率
- 六、结论
- 参考文献

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/05.webp)

## 一、引言

近年来，基于 Transformer 的网络[1]在提升计算机视觉（CV）任务[2]-[4]和自然语言处理（NLP）任务[5]-[7]性能方面取得了显著成功。

尽管其性能令人瞩目，但其计算特性已成为不可忽视的缺陷。与卷积神经网络（CNNs）[8]、循环神经网络（RNNs）[9]等其他网络相比，Transformer 拥有大量参数和极高的计算开销，例如 ViT-22B[10]、GPT-3[11]、PaLM[12]等模型。

::: block-1
研究人员已采取多种措施来缓解这一问题：

- 一方面，量化[13]-[15]等算法被开发用于减少内存占用和计算开销；
- 另一方面，各类加速器也被提出以加快推理速度[16]-[19]。
:::

**这些努力主要集中在加速 Transformer 中的矩阵乘法** ——这种运算被广泛认为是卷积神经网络（CNNs）、多层感知机（MLPs）等传统网络的主要瓶颈。**然而，对于自注意力机制中广泛存在的非线性计算，包括 Softmax 和层归一化（LayerNorm），相关研究却关注甚少。**

> 正如文献[20]所指出的，这些运算通常由 32 位浮点数（FP32）运算单元实现，严重制约了端到端 Transformer 推理性能的提升。

如图 1(a)所示，在 2080Ti GPU 上，输入图像尺寸为 448×448 时 Deit-Tiny 模型的延迟分解，**当矩阵乘法通过量化采用 8 位整数（INT8）运算实现后，这一问题愈发突出**，因此迫切需要一种解决方案。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/06.webp)

::: image-caption
图 1. （a）FP32 与 INT8 量化下各运算的延迟分解；（b）矩阵乘法与 Softmax/LayerNorm 运算在数据复用和瓶颈类型上的差异
:::

> 以往研究已提出基于函数逼近的方法来优化这些运算，但这类方法的效果仅限于计算层面，且往往忽视了存储问题。

图 1(b)展示了 Softmax/LayerNorm 运算与传统矩阵乘法的两个显著差异：

::: block-1
- 首先，Softmax/LayerNorm **采用两阶段数据流**，由于无法直接生成输出，需要缓存中间数据；
- 其次，这些运算中**输入数据复用率极低**，难以通过计算分摊内存成本。

**这两个因素共同导致了 Softmax/LayerNorm 运算的“内存受限”** （指计算速度受限于内存数据访问速度，而非计算单元本身速度）问题。
:::

此前的研究工作（如 Softermax[20]、I-BERT[21]）:mdnice-bold-italic[未能解决这一问题，因为它们在处理过程中仍需分别缓存 16 位和 32 位数据。此外，这类方法通常需要通过重新训练或微调] （微调指在预训练模型基础上用少量数据适配特定任务的过程）来弥补逼近误差，而随着 Transformer 模型规模不断增大，这种额外的训练开销也日益高昂。

> 本文提出 SOLE——一种软硬件协同设计方法，旨在解决上述问题。在 SOLE 中，我们分别为 Softmax 和 LayerNorm 设计了 E2Softmax 和 AILayerNorm 算法：

- 对于 Softmax，我们**对指数函数的输出进行 log2 量化，将中间数据压缩至 4 位，显著缓解内存受限问题**。借助 log2 量化，指数函数可通过仅由移位器和加法器构成的专用 log2 指数单元（Log2Exp Unit）实现。此外，我们还**采用提出的“近似基于对数的除法”替代浮点除法**，充分利用 log2 量化后的输出。通过这种方式，E2Softmax 可实现无乘法、无查找表（LUT，一种预存数据以快速查询结果的硬件结构）的设计。
- 对于 LayerNorm，我们**发现统计计算对输入微小变化引入的误差具有鲁棒性** （即误差对结果影响较小）。因此，我们**采用动态压缩，并将其与 2 的幂因子（PTF）[22]结合，统一优化数据流以实现低精度统计计算**。最终，AILayerNorm 仅需 8 位数据缓存和 4 位乘法即可完成统计计算。

综上，本文的贡献可总结如下：

::: block-1
1. 提出高效 log2 量化 Softmax（E2Softmax）——一种基于指数函数 log2 量化的硬件友好型 Softmax 算法；
2. 提出近似整数层归一化（AILayerNorm）——一种支持低精度统计计算的高效 LayerNorm 算法；
3. 在多种基于 Transformer 的网络上，**针对 CV 和 NLP 任务开展了专门的软件实验，结果表明 SOLE 无需额外训练即可实现可忽略的精度损失；**
4. 硬件实验表明，相较于 GPU，SOLE 在 Softmax 和 LayerNorm 上分别实现了 36.2 倍和 61.3 倍的平均加速比，以及数量级的能效提升；与当前最先进的定制硬件相比，SOLE 在 Softmax 和 LayerNorm 上的**面积效率分别提升 2.82 倍和 3.32 倍，能效分别提升 3.04 倍和 3.86 倍。**
:::

## 二、相关工作

### A. Transformers 中的主要非线性运算

典型的 Transformer 模型由堆叠的编码器和解码器块组成，每个块包含多个层，且由两个子模块构成：多头自注意力块和前馈网络[1]。

::: block-1
**非线性运算在 Transformer 计算中占比显著** [20]，其中 **Softmax 和 LayerNorm 尤为关键**。Softmax 函数用于计算多头自注意力机制中的注意力权重，而 LayerNorm 则用于对模型每一层的隐藏状态进行归一化。
:::

Softmax 和 LayerNorm 的定义如下：

$$
\begin{array}{r}
Softmax\left(X_{i}\right)=\frac{\exp \left(X_{i}-X_{\text {max }}\right)}{\sum_{j} \exp \left(X_{j}-X_{\text {max }}\right)} \\
LayerNorm\left(X_{i}\right)=\frac{X_{i}-\mu}{\sigma} \cdot \gamma+\beta
\end{array} \tag{1}
$$

其中，$\mu=\frac{1}{C} \sum_{i=1}^{C} X_{i}$（$\mu$ 为输入特征的均值），$\sigma=\sqrt{\frac{1}{C} \sum_{i=1}^{C}\left(X_{i}-\mu\right)^{2}}$（$\sigma$ 为输入特征的标准差）。当采用高成本的 FP32 算术运算实现时，这些操作在 Transformer 推理的运行时间中占比极高[20]；对于面向低端设备的硬件而言，这一开销更是显著[23]。因此，已有多项研究提出针对这一问题的解决方案。

### B. 非线性运算的加速方法

**加速非线性运算的传统方法是函数逼近** [24],[25]。例如：

- Softermax[20]提出了 Softmax 的低精度计算方案；
- I-BERT[21]采用 32 位整数算术来逼近 Softmax、LayerNorm 及其他非线性运算；
- NNLUT[26]则基于 I-BERT 代码库，采用硬件友好的查找表（LUT）和线性分段逼近（用多个线性段近似非线性函数的方法）来实现非线性运算，其 LUT 内容通过单隐藏层 ReLU 网络（ReLU 为常用激活函数，形式为 $f(x)=\max(0,x)$）获取。

::: block-1
**尽管这些方法具有一定优势，但存在两个显著缺陷**：

1. 此前的研究工作[20],[21]**需要通过微调或重新训练来恢复模型性能**，而随着 Transformer 规模增大，这一成本也不断攀升；
2. 这些方法**未能实现 Softmax 和 LayerNorm 的高效部署**，因为高精度乘法和高位宽数据存储仍是不可或缺的环节。
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/02.webp)

::: image-caption
图 2. SOLE 的整体架构。上图为软件层的 E2Softmax 和 AILayerNorm 算法设计，下图为硬件层的定制计算单元设计，展示了从算法逼近到硬件实现的全流程优化
:::

图 2 展示了 SOLE 的整体架构。与此前研究不同：

- SOLE 针对 Softmax 和 LayerNorm 分别设计了 E2Softmax 和 AILayerNorm 算法——这些**算法无需重新训练** 即可实现最小的精度损失。
- 此外，SOLE 还引入了**定制硬件单元来支持这些算法，通过低精度计算和低位宽存储实现高效部署。**

## 三、SOLE 软件设计

> 在 SOLE 中，我们分别为 Softmax 和 LayerNorm 设计了 E2Softmax 与 AILayerNorm 算法，以实现高效的运算。

本节首先介绍算法相关的预备知识，随后详细阐述 E2Softmax 和 AILayerNorm 算法的具体内容。

### 3.1 预备知识

本节将介绍所提算法中涉及的关键技术。

#### $\log_2$ 量化

通常而言，$log_2$ 量化可将（0,1）区间内的连续值转换为一系列离散值，例如 $2^0$、$2^{-1}$、$2^{-2}$ 等[27]。假设量化位宽为 $b$，则 $log_2$ 量化过程可定义为：

$$
log_2\ Q(X)=Clip(\lfloor -log_2(X)\rceil,0, 2^b-1 ), X \in(0, 1) \tag{2}
$$

#### 基于对数的近似除法

该方法由 Mitchell 提出[28]，其核心是**通过线性近似计算二进制数的对数，并利用减法和移位运算实现除法**。假设存在一个无符号 N 位整数 $X$，则 $X$ 可表示为：

$$
X=\sum_{i=0}^{N-1} 2^{i} b_{i}=2^{k_{x}}+\sum_{i=0}^{k_{x}-1} 2^{i} b_{i}=2^{k_{x}}(1+x) \quad(3)
$$

其中：

- $k_x$ 表示 $X$ 的最高有效位（即二进制数中最左边“1”的位置，用于表征数值的量级），
- $x \in(0,1)$（表示数值的尾数部分，用于表征数值的精度）。对数的近似表达式为 $log_2(X)=k_x + x$，其中 $k_x$ 为阶码部分，$x$ 为尾数部分。

以除法运算 $Q=\frac{X_1}{X_2}$ 为例，商的对数近似可表示为：

$$
log_2(Q) = k_1 + x_1 - k_2 - x_2 \tag{4}
$$

通过对数的逆运算（即指数运算）可计算出近似商：

$$
Q'=\begin{cases} 2^{k_{1}-k_{2}-1}\left(2+x_{1}-x_{2}\right), & x_{1}-x_{2}<0 \\ 2^{k_{1}-k_{2}}\left(1+x_{1}-x_{2}\right), & x_{1}-x_{2} \geq 0 \end{cases} \tag{5}
$$

由于对数的逆运算要求尾数部分处于（0,1）区间内，因此当 $x_1 - x_2 < 0$ 时，阶码部分会产生借位，故需将表达式分为上述两种情况讨论。

#### 2 的幂因子（Power-of-Two Factor, PTF）

PTF 由文献[22]提出，用于 LayerNorm 的量化过程，其核心是为不同通道分配不同的因子，以解决 LayerNorm 输入中通道间的数值差异问题。假设输入激活值为 $X \in \mathbb{R}^{B \times L \times C}$（其中 $B$ 为批大小、$L$ 为序列长度、$C$ 为通道数），量化位宽为 $b$，层级量化参数 $s$（缩放因子）、$zp$（零点，用于将有符号数转换为无符号数）$\in \mathbb{R}^1$，PTF 为 $\alpha \in \mathbb{N}^C$（每个通道对应一个整数因子），则量化后的激活值 $X_Q$ 可定义为：

$$
X_{Q}=Clip\left(\left\lfloor\frac{X}{2^{\alpha} s}\right\rceil+zp, 0,2^{b}-1\right) \tag{6}
$$

通常情况下，激活值范围较大的通道会分配更大的 $\alpha$，以确保其缩放因子能与自身数值范围相匹配。

### 3.2 E2Softmax 算法

E2Softmax（高效 $log_2$ 量化 Softmax）通过多种技术提升运算效率：

- 首先，对指数函数的输出进行 $log_2$ 量化；
- 其次，采用基于对数的近似除法（ALDivision）和 Log2Exp 函数，分别替代浮点数除法和指数函数（如算法 1 所示）。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/07.webp)

::: image-caption
算法 1：高效 $log_2$ 量化 Softmax（Efficient $log_2$ Quantized Softmax）
:::

所有计算均基于 8 位量化输入的定点精度完成，并引入在线归一化机制[29]（一种边计算边归一化的方法，可减少数据缓存和延迟）以进一步降低延迟。

#### 指数输出的 $\log_2$ 量化

指数函数是 Softmax 的核心组件，但其非线性特性导致硬件实现成本较高——**传统方法通常需要乘法器和大型查找表（LUT）来进行线性分段近似，这会增加功耗和面积开销**。

> 为解决该问题，我们**对指数运算的输出进行 $log_2$ 量化，使其更易于硬件实现**，具体定义如下：

$$
log2\ Exp(x)=-\left\lfloor log_2\left(e^{x}\right)\right\rceil=-\left\lfloor(x) \times \frac{1}{\ln 2}\right\rceil \quad(7)
$$

其中，$\frac{1}{\ln 2}$（自然对数的倒数）可近似为 1.4375，因此上式可简化为：

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/08.webp)

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/09.webp)

::: image-caption
图3 $Exp(X_i − X_{max})$ 的对数分布。横轴为 $Log2(Exp(X_i - X_{max}))$，纵轴为概率密度（Density），数值分布集中在-16 至 0 之间，符合近似正态分布的特征
:::

我们采用该转换方法主要基于以下三点原因：

::: block-1
1. 指数函数的输入需先减去对应向量的最大值（以避免数值溢出），这一操作**天然满足 Log2Exp 函数的输入范围** （$x \in(-\infty, 0]$）；
2. **指数输出在 $log_2$ 尺度下的分布近似于正态分布** （如图 3 所示），因此 $log_2$ 量化是理想的量化方式；
3. **Softmax 函数关注的是指数输出的【相对值】而非绝对值**，因此指数输出经 $log_2$ 量化引入的误差，会在后续除法运算中被进一步减小[17]。
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/10.webp)

::: image-caption
表I ImageNet-1K 基准上的算法实验结果
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/11.webp)

::: image-caption
表 II 基于 BERT-BASE 模型在 SQuAD 与 GLUE 任务上的算法实验结果
:::

::: block01 通过该方法，我们能以更高效的方式实现指数函数。表 I 和表 II 的实验结果表明，**指数输出经 $log_2$ 量化后可压缩至 4 位，且精度损失极小**，从而进一步减小了内存占用。 :::

#### 基于对数的近似除法

除法是 Softmax 中的另一项非线性运算，传统方法同样需要大型 LUT 和高精度乘法器。由于通过 $log_2$ 量化已获得 4 位精度的指数输出对数，因此我们可利用基于对数的近似除法进一步降低计算复杂度。假设经 $log_2$ 量化后的指数输出为 $k_y = log2\ Exp(Q_y)$，归约和（即所有指数输出的和）为 $S = 2^{k_s} \cdot (1 + s), s \in(0,1)$，则有：

$$
\frac{Q_y}{S}=2^{-(k_y + k_s)} \cdot \frac{1}{1 + s} \tag{9}
$$

其中，$k_s$ 表示归约和 $S$ 的最高有效位位置，$s$ 表示 $S$ 的尾数部分。我们采用线性近似将 $s$ 量化为 1 位精度的 $q(s)$：

$$
\begin{array}{r}
q(s) = \frac{\lfloor 2 \cdot s \rceil}{2}, q(s) \in \{0, 0.5\} \\
\frac{Q_y}{S} = 2^{-(k_y + k_s + 1)} \cdot (1 - q(s))
\end{array} \tag{10}
$$

则近似除法与全精度除法之间的误差可表示为：

$$
\delta=2^{-(k_y + k_s)} \cdot \frac{s - 1 - q(s) \cdot (s + 1)}{2(1 + s)} \tag(11)
$$

假设 $s$ 为[0,1]区间内的均匀随机变量，可计算误差的期望：

$$
\begin{aligned}
E(\delta) & =2^{-(k_y + k_s)}\left(\int_{0}^{0.5} \frac{s - 1}{2(s + 1)} d s+\int_{0.5}^{1} \frac{s - 3}{4(1 + s)} d s\right) \\
& =2^{-(k_y + k_s + 1)} \times(-0.636)
\end{aligned} \tag{12}
$$

为使近似除法无偏（即误差期望为 0），我们对基于对数的近似除法（ALDivision）进行修正，修正后的表达式为：

$$
\text{ALDivision} \left(k_y, S\right)=2^{-(k_y + k_s + 1)} \cdot (1.636 - q(s)) \quad(13)
$$

其中 1.636 是为抵消误差期望而引入的修正系数。

通过上述方式，除法运算可通过硬件友好的移位和减法实现。实际上，由于 $q(s)$ 是 1 位数值（仅 0 或 1），减法运算还可替换为二路多路选择器（根据 $q(s)$ 的值选择不同的输入）。具体实现细节将在第 4 节（硬件设计）中介绍。

### 3.3 AILayerNorm 算法

AILayerNorm（近似整数层归一化）算法基于 PTF 量化，采用动态压缩技术实现低精度的统计量计算（统计量指均值 $\mu$ 和方差 $\sigma^2$）。AILayerNorm 的具体流程如算法 2 所示。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/12.webp)

::: image-caption
算法 2：近似整数层归一化（Approximate Integer Layer Normalization）
:::

#### 动态压缩（Dynamic Compression）

LayerNorm 通过均值和方差对输入进行归一化，因此统计量计算是该算法的核心环节。为得到均值和方差，通常需要计算输入的期望 $E(x)$ 和输入平方的期望 $E(x^2)$[30]。其中，$E(x)$ 仅需加法运算即可实现，而 $E(x^2)$ 则需要大量乘法运算。基于“在 $x^2$ 的归约计算中，小数值的重要性低于其在 $x$ 归约计算中的重要性”这一特性，我们提出动态压缩技术，以实现低精度的统计量计算。

$$
\frac{x_1^2}{x_1^2 + x_2^2} \lt \frac{x_1}{x_1 + x_2}, x_1 \lt x_2\ and\ x_1,x_2 \gt 0 \tag{14}
$$

为降低计算复杂度并最小化性能损失，我们针对 8 位无符号整数 $x[7:0]$（下标表示比特位范围，从第 0 位到第 7 位）设计动态压缩方法，生成 4 位近似值 $y[3:0]$。

该方法根据输入数值的大小动态压缩数据：如![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/13.webp)所示，通过过滤部分低有效位实现压缩，且过滤的比特长度随输入数值动态调整，动态压缩函数具体可表示为：

$$
\text{Dynamic Compress(x)}=\begin{cases} Clip(\lfloor x / 2^4 \rceil , 0, 15),\ 1,\ x[7 : 6] \ne 0 \\ Clip(\lfloor x / 2^4 \rceil , 0, 15),\ 0,\ \text{else} \end{cases} \tag{15}
$$

通过该压缩方法，$x^2$ 的计算可通过 4 位整数运算和移位操作实现。同时，算法还会生成 1 位信号 $s$，用于在恢复原始输入时确定移位长度（2 位或 4 位）。**实验表明，对于均匀分布的输入数据，该方法仅会导致 $E(x^2)$ 产生 0.2%的误差、标准差产生 0.4%的误差。**

#### 低精度统计量计算

我们的方法基于 PTF 量化技术，该技术可解决 LayerNorm 输入的通道间差异，并将输入激活值压缩至 8 位整数。然而，在统计量计算前，需对输入进行通道级移位操作以恢复准确数值，这会导致方差计算需要高精度运算——具体而言，**8 位整数 $X$ 经移位后，计算 $X^2$ 需采用 12 位乘法器**。因此，我们通过等价的数学变换优化数据流，以解决该问题。平方运算可通过移位实现：

$$
\begin{aligned}
Square (X, \alpha) & =(X<<\alpha) \cdot (X<<\alpha) \\
& =(X \cdot X)<<(2\alpha)
\end{aligned} \tag{16}
$$

“$X<<\alpha$”表示将 $X$ 左移 $\alpha$ 位，等价于 $X$ 乘以 $2^\alpha$。

该优化操作与动态压缩的解码过程共同安排在“解压缩（Decompress）”阶段。结合前文提到的动态压缩技术，**该优化可完全避免统计量计算中的 12 位乘法，转而采用 4 位整数运算，从而实现低精度统计量计算。**

## 四、SOLE 硬件设计

> 在 SOLE 中，由于通**用硬件平台（如 CPU、GPU）不支持算法中涉及的特殊功能单元，我们设计了定制化硬件单元**。:mdnice-bold-italic[该定制硬件可集成到其他专用加速器或张量处理硬件（如 GPU 张量核心）中]，以加速 Transformer 推理。

### 4.1 E2Softmax 单元

我们设计了 E2Softmax 单元来实现 E2Softmax 算法，其**核心特征是包含 Log2Exp 单元和基于对数的近似除法器**，且两者均采用无查找表（LUT-free）、无乘法器（multiplication-free）的实现方式，以提升效率。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/14.webp)

::: image-caption
图 4：E2Softmax 单元的硬件设计，输入经 Log2Exp 单元处理后，一部分送入归约和（ReducedSum）计算模块，另一部分存入缓冲区；归约和与缓冲区中的数据再经近似除法器处理，输出 Softmax 结果，整体采用两阶段流水线设计
:::

**如图 4 所示，计算过程分为两个阶段以优化数据流，同时采用乒乓缓冲区** （Ping-pong buffers，通过两个缓冲区交替读写实现数据连续传输、避免流水线停滞的硬件结构）支持流水线操作。

#### Softmax 第一阶段：非归一化 Softmax 计算

> 第一阶段包含三个子单元：最大值单元（Max Unit）、Log2Exp 单元（Log2Exp Unit）和归约单元（Reduction Unit）。

由于输入向量长度可能高达 1024，该阶段每次接收输入向量的一个切片（slice，即部分数据）进行处理。

- **最大值单元**：通过比较树（comparison tree，一种并行比较结构，可快速找出一组数据中的最大值）确定输入向量切片的局部最大值（LocalMax）。随后，将局部最大值与从缓冲区中读取的全局最大值（GlobalMax）进行比较，若局部最大值更大，则用其更新全局最大值。
- **Log2Exp 单元**：接收“输入向量切片与局部最大值的差值”作为输入，通过移位和加法运算生成输出（而传统硬件通常需要 LUT 和乘法器）。最后通过舍入操作得到 4 位整数输出，包括“Log2Exp 输出”（用于后续归约和除法）和“修正值”（用于在线归一化）。
- **归约单元**：Log2Exp 输出一方面存入输出缓冲区以备第二阶段使用，另一方面送入归约单元；修正值则用于调整来自“和缓冲区（Sum Buffer）”的数据，以实现在线归一化。

::: block-1
由于 Log2Exp 输出被量化为 4 位，存储中间结果所需的内存极小，大幅降低了内存开销。
:::

#### Softmax 第二阶段：归一化计算

> 当归约和计算完成后，第二阶段开始执行除法运算并生成 Softmax 最终输出。传统方法（如[20]、[26]）中，除法通常通过依赖 LUT 和高精度乘法器的线性分段近似实现；而本设计中，我们引入了基于对数的近似除法器。

具体流程如下：

- 首先计算修正值并与 Log2Exp 输出相加，得到修正后的 Log2Exp 值；
- 随后将该值与归约和（ReducedSum）一同送入近似除法器，以得到最终输出。

基于对数的近似除法器结构简单、轻量化，仅包含最高有效位检测器（LOD，Leading-one detector，用于检测二进制数中最左边“1”的位置）、减法器、二路多路选择器和两个移位器。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/07.webp)

::: image-caption
算法 1：高效 $log_2$ 量化 Softmax（Efficient Log2 Quantized Softmax）
:::

根据算法 1 的逻辑，除法器的工作步骤为：

1. 利用 LOD 检测归约和的阶码部分（$k_s$），并将其与 Log2Exp 值的阶码部分（$k_y$）相减；
2. 通过移位操作选取归约和尾数部分中最高有效位的下一位，生成 1 位结果（即 $q(s)$）；
3. 该 1 位结果作为多路选择器的选择信号（$s$），用于确定输出：$O=\frac{1.636 - 0.5s}{2}=\begin{cases} 0.818, & s=0 \\ 0.568, & s=1 \end{cases} \tag{17}$
4. 根据步骤 1 中阶码相减的结果，通过右移位器对上述输出进行缩放，最终得到 E2Softmax 的输出。

::: block-1
与传统设计相比，该除法器充分利用了 Log2Exp 单元生成的对数输入，**仅需减法、LOD 检测、二路多路选择和两次移位操作** 即可实现，硬件开销极低。
:::

### 4.2 AILayerNorm 单元

> 我们设计了 AILayerNorm 单元来实现 AILayerNorm 算法，其**核心特征是支持低精度统计量计算**。该单元同样分为两个阶段（统计量计算和仿射变换），并采用乒乓缓冲区设计流水线结构。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/15.webp)

::: image-caption
图 5：AILayerNorm 单元的硬件设计，输入向量经预处理后分为两路：一路送入 Ex 单元（计算输入期望），另一路经动态压缩后送入 Ex2 单元（计算输入平方期望）；两路结果分别经累加、除法（1/n）后得到均值和标准差，再与权重结合完成仿射变换，输出最终结果
:::

与批归一化（Batch Normalization）不同，**层归一化（LayerNorm）对批大小（mini-batch size）无限制**，因此可在批大小为 1 的纯在线模式下工作[31]。

#### 统计量计算

层归一化需计算均值（$\mu$）和方差（$\sigma^2$）以实现输入归一化，第一阶段负责完成统计量计算，包含两个子单元：Ex 单元（计算 $E(x)$）和 Ex2 单元（计算 $E(x^2)$）。

传统方法（如[21]中的 I-BERT）需采用高精度计算（如 32 位整数）来保证均值和方差的精度，这会导致功耗和面积开销显著增加；而本设计通过低精度运算即可实现同等精度，具体流程如下：

1. 预处理：8 位量化输入首先减去零点（zp），将无符号数转换为有符号数，随后分别送入 Ex 单元、Ex2 单元和输入缓冲区（Input Buffer）。
2. Ex 单元：输入经 PTF 因子（$\alpha_i$）移位后，送入 12 位累加器（Accumulator）进行归约累加，以计算输入的总和（用于后续求均值）。
3. Ex2 单元：输入首先经过动态压缩得到 4 位压缩值；由于 4 位输入的平方仅有 16 种可能结果，我们采用 16 项的 LUT（而非乘法器）实现平方运算，以提升效率；随后，解压缩单元（Decompress Unit）通过移位操作补偿 PTF 和动态压缩带来的缩放，恢复近似的原始平方值；最后，该值送入累加器进行归约累加，以计算输入平方的总和（用于后续求平方的期望）。
4. 后处理：当得到输入总和（$\sum_{i=1}^n X_i$）和输入平方总和（$\sum_{i=1}^n X_i^2$）后，预处理单元（Preprocess Unit）通过除法（乘以 $1/n$）计算出 $E(x)$ 和 $E(x^2)$，并进一步通过 $x^{-1/2}$ 单元（计算平方根的倒数，用于后续归一化）生成第二阶段所需的参数（均值 $\mu$ 和标准差的倒数 $stdinv$）。其中，$x^{-1/2}$ 单元同样采用 LUT 实现，由于其运算密度小，占用的面积和功耗极低。

#### 仿射变换

当第一阶段完成统计量计算后，第二阶段开始对输入进行归一化并通过仿射参数（$\gamma$、$\beta$）进行缩放偏移。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/12.webp)

::: image-caption
算法 2：近似整数层归一化（Approximate Integer Layer Normalization）
:::

根据算法 2 的逻辑，仿射单元（Affine Unit）将“归一化”和“缩放偏移”融合为两次乘法和两次加法运算，具体流程如下：

1. 参数计算：仿射单元首先计算仿射参数 $A = \gamma_i \cdot stdinv$（权重与标准差倒数的乘积，用于归一化后的缩放）和 $B = \beta_i$（偏置，用于归一化后的偏移）。
2. 归一化：从输入缓冲区中读取的输入数据，经 PTF 因子移位后减去均值 $\mu$，完成归一化操作。
3. 仿射输出：归一化结果与 $A$ 相乘后再加上 $B$，得到最终输出 $Y = A \cdot X + B$。

> 需注意的是，本设计中仿射权重（$\gamma$）同样被量化为 8 位整数，以保持整体的低精度特性，进一步降低硬件开销。

## 五、实验评估

### 5.1 实验设置

#### 软件设置

> 为验证所提算法，我们在计算机视觉（CV）和自然语言处理（NLP）任务上开展了专项实验。

- **CV 任务**：选取 ImageNet-1K 数据集（ImageNet 数据集的 1000 类子集，常用于图像分类模型评估）作为基准，测试不同视觉 Transformer 模型，即 DeiT 系列（DeiT-T：小型 DeiT 模型、DeiT-S：中型 DeiT 模型、DeiT-B：基础型 DeiT 模型）和 Swin Transformer 系列（Swin-T：小型 Swin 模型、Swin-S：中型 Swin 模型、Swin-B：基础型 Swin 模型）。
- **NLP 任务**：在 GLUE 基准（通用语言理解评估基准，包含多个自然语言理解子任务）和 SQuAD v1.1 数据集（斯坦福问答数据集 1.1 版本，用于机器阅读理解任务）上，基于 BERT-Base 模型（基础型 BERT 模型，含 12 层 Transformer 编码器）进行实验。

特别说明：

- 实验选取 32 位浮点数（FP32）模型和 8 位整数（INT8）模型作为基准。
- **选择 INT8 模型作为基准，是为了证明 SOLE 可作为其他压缩方法（如量化）的插件模块**。
- 我们进一步将 SOLE 算法替换原有浮点版本，评估其对精度的影响，记为“FP32+SOLE”和“INT8+SOLE”。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/10.webp)

::: image-caption
表I ImageNet-1K 基准上的算法实验结果
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/11.webp)

::: image-caption
表 II 基于 BERT-BASE 模型在 SQuAD 与 GLUE 任务上的算法实验结果
:::

- 表 I 中的 INT8 模型通过“训练后量化”（无需重新训练，直接对训练好的 FP32 模型进行量化）得到，量化设置参考[22]；
- 表 II 中的 INT8 模型则通过“量化感知微调”（训练过程中模拟量化误差，提升量化后模型精度）得到，权重和激活均采用 8 位量化，设置参考[38]。

#### 硬件设置

为评估 SOLE 在加速比和硬件效率上的表现，我们采用寄存器传输级 Verilog 硬件描述语言（RTL-Verilog）实现 SOLE 硬件，并使用 Synopsys Design Compiler（逻辑综合工具）在 28nm 台积电（TSMC）工艺库下进行综合，目标时钟周期为 1ns（对应时钟频率 1GHz）。

功耗评估通过 PrimeTimePX（Synopsys 公司的静态功耗分析工具）在典型工艺角（typical corner，指正常电压、温度下的工艺参数）下完成，仿真轨迹采用 Value Change Dump（VCD，值变化转储）格式的仿真文件。

SOLE 硬件的向量长度设置为 32，以匹配传统加速器的乘加（MAC）吞吐量。

- **加速比评估**：选取 NVIDIA 2080Ti GPU 作为基准，将 E2Softmax 单元和 AILayerNorm 单元的资源规模扩大 32 倍，以实现相对公平的对比。
- **面积与功耗评估**：除 GPU 外，还与当前最先进的定制硬件进行对比：Softmax 对比基准为 Softermax[20]，LayerNorm 对比基准为 NN-LUT[26]（注：NN-LUT 中非线性操作采用基于神经网络的查找表（LUT）设计，其他部分遵循 I-BERT 的方法[21]）。为保证公平性，我们在与 SOLE 相同的设置下重新实现了这些基准设计，以提取其功耗和面积数据。

### 5.2 算法性能

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/10.webp)

::: image-caption
表I ImageNet-1K 基准上的算法实验结果
:::

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/11.webp)

::: image-caption
表 II 基于 BERT-BASE 模型在 SQuAD 与 GLUE 任务上的算法实验结果
:::

表 I 和表 II 总结了不同 Transformer 模型在 CV 和 NLP 任务上的精度对比结果，核心结论如下：

1. **FP32 与 FP32+SOLE 的对比**：SOLE 导致的精度损失可忽略不计。在不同数据集和模型上，最大精度损失低于 0.9%，平均精度损失约为 0.38%。
2. **INT8 与 INT8+SOLE 的对比**：SOLE 与 INT8 量化兼容良好。最大精度损失为 0.8%，平均精度损失为 0.2%，证明 SOLE 可与其他模型压缩方法（如量化）集成。

::: block-1
需注意：INT8 基准模型中，仅矩阵乘法采用 INT8 格式，而 Softmax、LayerNorm 等其他操作仍采用 FP32 格式；**集成 SOLE 后，Softmax 和 LayerNorm 的输入与输出均可采用 8 位格式计算。**
:::

> SOLE 的另一显著优势是：**无需额外训练或微调即可维持推理精度**，从而避免了高昂的训练开销，使用便捷性大幅提升。

### 5.3 加速比

> **图 6 展示了 SOLE 相对于 2080Ti GPU 的加速比**，实验设置如下：

- 在 DeiT-Tiny 模型上测试独立的 Softmax/LayerNorm 操作，批次大小（batchsize）从 1 到 16 变化，令牌长度（token length）设为 785（对应 448×448 图像尺寸）；
- 同时测试 INT8 模型在有无 SOLE 软硬件优化时，相对于 FP32 模型的端到端加速比。

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/16.webp)

::: image-caption
图6. (a) 在 Softmax 和 LayerNorm 运算上相较于 GPU 的加速比。(b) 端到端加速比与延迟分解，延迟均以 FP32 实现为基准进行归一化。
:::

结论如下：

- **图 6(a)：独立操作加速比**：SOLE 在 Softmax 上的加速比为 29.3 倍~57.5 倍（平均 36.2 倍），在 LayerNorm 上的加速比为 38.4 倍~86.8 倍（平均 61.3 倍）。:mdnice-bold-italic[加速比主要来源于“高度专用化的流水线数据通路”设计——通过乒乓缓冲（ping-pong buffers）和在线归一化（online normalization）最大化吞吐量，从而降低延迟。]
- **图 6(b)：端到端加速比与延迟分解**：在 GPU 上，INT8 模型相对于 FP32 模型的加速比仅为 1.10 倍~1.28 倍，:mdnice-bold-italic[原因是 INT8 量化仅缓解了矩阵乘法的开销，而 Softmax、LayerNorm 等非线性操作仍占据推理时间的很大比例]；集成 SOLE 优化非线性操作后，端到端加速比提升至 1.50 倍~2.09 倍（延迟以 FP32 实现为基准进行归一化）。

结合 5.2 节的结论（SOLE 与 INT8 量化集成仅导致可忽略的精度损失），SOLE 硬件还可轻松集成到现代加速器中——其 8 位输入/输出与现有加速器、GPU 张量核（tensor cores）中的 8 位整数向量 MAC 数据通路兼容，因此可实现全面优化以获得更高加速比。

> 需注意：本实验**【仅】使用 32 个 E2Softmax 单元和 AILayerNorm 单元与 GPU 对比** （GPU 的硬件资源仍远多于 SOLE），但 SOLE 在 Softmax、LayerNorm 及 Transformer 端到端推理上仍**具备可扩展的加速能力。**

### 5.4 硬件效率

#### 能效

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/03.webp)

::: image-caption
表III SOFTMAX 与 Softermax、NN-LUT 及 GPU 在能耗和面积方面的唯一对比
:::

> 表 III 对比了 SOLE 与 Softermax、NN-LUT 及 2080Ti GPU 的能效（实验设置与 5.3 节一致，表中报告平均能效）。

为深入分析，我们**分别评估了“计算子单元”** （即 4 节提到的归一化单元（Normalization Unit）和**统计单元（Statistic Unit）**，仅反映计算功耗开销）和**“完整硬件单元”** （即 Softmax 单元和 LayerNorm 单元，同时反映计算与存储功耗开销）的功耗：

- 计算子单元：SOLE 的归一化单元能效比 Softermax 高 2.46 倍，统计单元能效比 NN-LUT 高 11.3 倍；
- 完整硬件单元：SOLE 的 Softmax 单元能效比 Softermax 高 3.04 倍，LayerNorm 单元能效比 NN-LUT 高 3.86 倍。

::: block-1
子单元与完整硬件单元的能效差异，源于完整硬件的功耗主要由存储开销决定（如 1 节所述，Softmax/LayerNorm 存在内存瓶颈）。
:::

整体而言，SOLE 相对于基准的能效提升如下：

- Softmax：相对于 Softermax 提升 3.04 倍，相对于 GPU 提升 4925 倍；
- LayerNorm：相对于 NN-LUT 提升 3.86 倍，相对于 GPU 提升 4259 倍。

能效提升的核心原因：

1. **Softmax 优化**：E2Softmax 单元:mdnice-bold-italic[采用 Log2Exp 单元和近似对数除法器，通过移位和加法实现非线性操作] （无需查找表和乘法）；同时，:mdnice-bold-italic[指数输出的 $log_2$ 量化使中间结果仅需 4 位存储] （相比 Softermax 的 16 位存储，大幅降低内存访问成本）。
2. **LayerNorm 优化**：AILayerNorm 采用:mdnice-bold-italic[动态压缩和低精度统计计算]，完全避免了 NN-LUT 中需的 32 位乘法，:mdnice-bold-italic[【仅】使用 16 项 LUT 和移位操作] （降低计算功耗）；此外，SOLE :mdnice-bold-italic[将输入数据量化为 8 位] （相比基准的 32 位存储，大幅减少中间内存访问）。

#### 面积效率

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/03.webp)

::: image-caption
表III SOFTMAX 与 Softermax、NN-LUT 及 GPU 在能耗和面积方面的唯一对比
:::

如表 III 所示，SOLE 相对于当前最先进设计在面积上也有显著优化：

- 计算子单元：**归一化单元面积效率比 Softermax 高 2.89 倍**，统计单元面积效率比 NN-LUT 高 3.79 倍；
- 完整硬件单元：**Softmax 单元面积效率比 Softermax 高 2.82 倍**，LayerNorm 单元面积效率比 NN-LUT 高 3.32 倍。

面积节省的来源：

1. **计算单元简化**：Softmax 中:mdnice-bold-italic[【仅】使用移位器和加法器（替代 Softermax 的乘法器和 LUT）；LayerNorm 中用 16 项 LUT 和移位器（替代 NN-LUT 的 32 位乘法器）]，且整个计算过程基于 8 位输入数据，降低计算面积开销。
2. **存储单元缩减**：SOLE :mdnice-bold-italic[将缓冲中存储数据的位宽从基准的 16 位（Softmax）/32 位（LayerNorm）降至 4 位（Softmax）/8 位（LayerNorm）]，缓冲面积大幅减小。

## 六、结论

> 本文提出 SOLE——一种用于 Transformer 高效推理的软硬件协同设计方法，旨在优化 Softmax 和 LayerNorm 操作的推理效率。

SOLE 由 E2Softmax 和 AILayerNorm 两部分组成，通过**低精度计算** 和**低位宽数据存储** 实现 Softmax 与 LayerNorm 的高效部署，核心成果如下：

1. **精度保障**：SOLE :mdnice-bold-italic[无需额外训练即可维持推理精度，精度损失可忽略不计，且能与量化等正交压缩方法] （如 INT8 量化）有效集成；
2. **性能提升**：相对于 GPU，SOLE 在 Softmax 和 LayerNorm 上实现了:mdnice-bold-italic[数量级的加速比和能效提升]；
3. **硬件效率优化**：相对于:mdnice-bold-italic[当前最先进的定制硬件（Softermax 和 NN-LUT）]，SOLE 的 Softmax 单元能效提升 3.04 倍、面积效率提升 2.82 倍，LayerNorm 单元能效提升 3.86 倍、面积效率提升 3.32 倍。

综上，SOLE 为 Transformer 中非线性操作的高效推理提供了软硬件协同优化方案，可有效缓解 Softmax 和 LayerNorm 的内存瓶颈与计算开销，为 Transformer 模型的低延迟、低功耗部署提供支撑。

## 参考文献

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/17.webp)

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/18.webp)

![文章插图](/articles/2025-11-10-sole-softmax-layernorm-36-2-b6fc37fa/images/19.webp)
