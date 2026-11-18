关键词：**Manifold-Constrained Hyper-Connections**、Hyper-Connections、**Residual Connection**、Manifold Projection、Large Language Models

> 通过高效的基础设施级优化（如核融合、重计算、通信重叠），mHC 在实现这些改进的同时，仅引入可忽略的计算开销。:mdnice-bold-italic[TileLang 简化了具有复杂计算流程的核的实现过程，并使我们能够以最小的工程工作量充分利用内存带宽。4.3.1 部分有比较多的公式和工程优化细节描述。]

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/01.webp)

- **mHC: Manifold-Constrained Hyper-Connections**
- https://arxiv.org/pdf/2512.24880
- 1.7 万字，阅读 60 分钟，**播客 27 分钟**

相关推荐

- [PipeThreader：软件定义流水线驱动的 DNN 编译器，FlashAttention与Mamba2性能显著超越现有方案！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889807&idx=1&sn=a8f2bfdca2091144fcd6059a575dda66&scene=21#wechat_redirect)
- **[超越英伟达 B200 19%计算密度：7nm 四 Chiplet RISC-V 开源架构——面向 AI/HPC 的代际进化路线图](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894155&idx=1&sn=a148162fc540c7a1a4677e210c439dc4&scene=21#wechat_redirect)**
- **[性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)**

> 本文聚焦深度神经网络中**残差连接的优化问题**，针对 Hyper-Connections（HC）扩展残差流宽度、丰富连接模式时，因破坏恒等映射特性导致的训练不稳定性、可扩展性受限及内存访问开销大等痛点，提出 Manifold-Constrained Hyper-Connections（mHC）框架。

**Hyper-Connections** （简称 HC）是 Zhu 等人于 2024 年提出的深度神经网络**残差连接扩展方案**，属于模型宏观架构设计范畴，:mdnice-bold-italic[核心是突破传统残差连接（如 ResNet、Transformer 中的残差范式）的单一流限制，通过扩展残差流宽度与丰富连接模式提升模型表达能力，同时尽量保留计算效率]，适用于大语言模型（LLMs）等基于 Transformer 的深度架构。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/02.webp)

::: image-caption
图1 残差连接范式示意图。该图对比了（a）标准残差连接、（b）超连接（HC）以及（c）本文提出的流形约束超连接（mHC）的结构设计。与无约束的HC不同，mHC通过将矩阵投影到受约束的流形上来优化残差连接空间，从而确保模型训练的稳定性。可以把这三种结构理解成“信号传递通道”：（a）标准残差连接是“单条稳定通道”，信号按固定路径传递，虽稳定但连接方式单一；（b）HC是“多条无规则通道”，为提升性能加宽了通道、增加了连接数量，却因没设“通行规则”，训练时信号易混乱（比如强度突然变大或变小）；（c）mHC则给HC的“多条通道”加了“规则”——通过流形约束让通道里的“信号传递矩阵”符合特定规律，既保留了多通道的优势，又避免了信号混乱，让模型训练更稳定
:::

而 mHC 的核心创新，其实就是给 HC 的残差连接“立规矩”：先把 HC 原本没什么约束的残差连接空间，“放进”一个有特定规则的空间（也就是特定流形）里。具体怎么放呢？靠 Sinkhorn-Knopp 算法这个“调整工具”，把残差映射（可以理解成“信号传递的方式”）改成双随机矩阵——这种矩阵有个死规矩：每行的数加起来是 1，每列的数加起来也得是 1，就像“分东西时，每行每列的总量都固定，不会多拿也不会少给”。

这么改有两个好处：

- 一是**没破坏HC原本的能力** ——依然能把模型学到的各种特征（比如文本的语义、语法信息）好好整合起来（也就是保留特征融合能力）；
- 二是**让残差连接恢复了恒等映射的“信号守恒特性”** ——简单说就是“信号在模型里传递时，不会莫名变超强，也不会莫名变消失”（恒等映射本质就是让信号尽量“原汁原味”传，信号守恒就是避免信号强度失控）。这样一来，模型训练时就不会出现“调整参数的依据突然变超大（梯度爆炸），或者突然变超小（梯度消失）”的问题了。

除此之外，mHC还对输入输出的映射加了个“非负约束”——就是不让输入输出的数值出现负数，避免正负数值互相抵消导致信号混乱，这样能让模型训练更稳定。

::: block-1
为保障效率，mHC 采用:mdnice-bold-italic[三重基础设施优化：核融合减少内存带宽瓶颈，选择性重计算降低显存占用，扩展 DualPipe 调度实现通信与计算重叠。]

实验表明，mHC 在 3B、9B、27B 参数模型上均表现优异，27B 模型在 8 项下游基准测试中**全面超越基线和 HC**，BBH 和 DROP 任务性能分别**提升 2.1%和 2.3%**，** 训练稳定性显著增强，且【仅】引入 6.7%的额外时间开销。**
:::

> 该框架为深度模型的宏观架构设计提供新思路，通过平衡可塑性与稳定性，为下一代基础模型的规模化训练提供了高效解决方案。

## 本文目录

- 一、引言
- 二、相关工作
  - 2.1 微观设计（Micro Design）
  - 2.2 宏观设计（Macro Design）
- 三、预备知识
  - 3.1 数值不稳定性（Numerical Instability）
  - 3.2 系统开销（System Overhead）
- 四、方法
  - 4.1 流形约束超连接（Manifold-Constrained Hyper-Connections）
  - 4.2 参数化与流形投影（Parameterization and Manifold Projection）
  - 4.3 高效基础设施设计（Efficient Infrastructure Design）
- 五、实验
  - 5.1 实验设置
  - 5.2 主要结果
  - 5.3 缩放实验（Scaling Experiments）
  - 5.4 稳定性分析
- 六、结论与展望
- 参考文献

## 一、引言

> 自残差网络（ResNets）[He et al., 2016a]提出以来，深度神经网络架构经历了快速发展。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/02.webp)

::: image-caption
图1 残差连接范式示意图。该图对比了（a）标准残差连接、（b）超连接（HC）以及（c）本文提出的流形约束超连接（mHC）的结构设计。与无约束的HC不同，mHC通过将矩阵投影到受约束的流形上来优化残差连接空间，从而确保模型训练的稳定性。可以把这三种结构理解成“信号传递通道”：（a）标准残差连接是“单条稳定通道”，信号按固定路径传递，虽稳定但连接方式单一；（b）HC是“多条无规则通道”，为提升性能加宽了通道、增加了连接数量，却因没设“通行规则”，训练时信号易混乱（比如强度突然变大或变小）；（c）mHC则给HC的“多条通道”加了“规则”——通过流形约束让通道里的“信号传递矩阵”符合特定规律，既保留了多通道的优势，又避免了信号混乱，让模型训练更稳定
:::

如图 1(a)所示，单一层的结构可表示为如下形式：

$$
\begin{array}{r}
x_{l+1}=x_{l}+\mathcal{F}(x_{l},\mathcal{W}_{l}),
\end{array} \tag{1}
$$

其中$x_l$和$x_{l+1}$分别表示第$l$层的$C$维输入和输出，$\mathcal{F}$表示残差函数（即对输入进行变换的函数，如卷积、注意力操作等）。尽管在过去十年中，残差函数$\mathcal{F}$已发展出多种操作形式（包括卷积、注意力机制和前馈网络等），但残差连接（指将浅层特征直接与深层特征相加，缓解梯度消失问题的连接方式）的范式始终保持其原始形式。随着 Transformer 架构[Vaswani et al., 2017]的发展，该范式目前已成为大型语言模型（LLMs）[Brown et al., 2020; Liu et al., 2024b; Touvron et al., 2023]中的基础设计元素。

这一成功主要归功于残差连接的简洁形式。更重要的是，:mdnice-bold-italic[早期研究[He et al., 2016b]发现，残差连接的恒等映射特性（指浅层信号不经过任何修改直接传递到深层的特性）能够在大规模训练过程中保持稳定性和效率]。通过在多层之间递归扩展残差连接，公式（1）可扩展为：

$$
x_{L}=x_{l}+\sum_{i=l}^{L-1} \mathcal{F}\left(x_{i}, \mathcal{W}_{i}\right), \tag{2}
$$

其中$L$和$l$分别对应较深层和较浅层。**恒等映射中的$x_l$项强调了“浅层信号无需修改即可直接传递到深层”这一核心特性。**

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/02.webp)

::: image-caption
图1 残差连接范式示意图。该图对比了（a）标准残差连接、（b）超连接（HC）以及（c）本文提出的流形约束超连接（mHC）的结构设计。与无约束的HC不同，mHC通过将矩阵投影到受约束的流形上来优化残差连接空间，从而确保模型训练的稳定性。可以把这三种结构理解成“信号传递通道”：（a）标准残差连接是“单条稳定通道”，信号按固定路径传递，虽稳定但连接方式单一；（b）HC是“多条无规则通道”，为提升性能加宽了通道、增加了连接数量，却因没设“通行规则”，训练时信号易混乱（比如强度突然变大或变小）；（c）mHC则给HC的“多条通道”加了“规则”——通过流形约束让通道里的“信号传递矩阵”符合特定规律，既保留了多通道的优势，又避免了信号混乱，让模型训练更稳定
:::

近年来，以超连接（Hyper-Connections, HC）[Zhu et al., 2024]为代表的研究为残差连接引入了新的维度，并通过实验证明了其性能潜力。HC 的单一层架构如图 1(b)所示。通过扩展残差流宽度（即增加并行传递的特征流数量）和提高连接复杂度，HC 在不改变单个单元计算量（以浮点运算次数 FLOPs 衡量）的前提下，显著提升了拓扑复杂度。形式上，HC 中的单一层传播可定义为：

$$
x_{l+1}=\mathcal{H}_{l}^{\text{res}} x_{l}+\mathcal{H}_{l}^{\text{post}\top} \mathcal{F}\left(\mathcal{H}_{l}^{\text{pre}} x_{l}, \mathcal{W}_{l}\right), \tag{3}
$$

其中$x_l$和$x_{l+1}$分别表示第$l$层的输入和输出。与公式（1）不同，HC 中$x_l$和$x_{l+1}$的特征维度从$C$扩展到$n \times C$（$n$为扩展率，即残差流的并行数量）。$\mathcal{H}_{l}^{\text{res}} \in \mathbb{R}^{n \times n}$是一个可学习映射，用于对残差流内部的特征进行混合；同样作为可学习映射，$\mathcal{H}_{l}^{\text{pre}} \in \mathbb{R}^{1 \times n}$将$nC$维残差流特征聚合为$C$维的层输入，而$\mathcal{H}_{l}^{\text{post}} \in \mathbb{R}^{1 \times n}$则将层输出映射回残差流。

::: block-1
**然而，随着训练规模的扩大，HC 引入了潜在的不稳定性风险**。核心问题在于：:mdnice-bold-italic[当架构扩展到多层时，HC 的无约束特性会破坏恒等映射]。在包含多个并行流的架构中，理想的恒等映射是一种“守恒机制”——它确保在正向传播和反向传播过程中，各流之间的平均信号强度保持不变。
:::

通过公式（3）将 HC 递归扩展到多层后，可得：

$$
x_{L}=\left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right) x_{l}+\sum_{i=l}^{L-1}\left(\prod_{j=1}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}}\right) \mathcal{H}_{i}^{\text{post}\top} \mathcal{F}\left(\mathcal{H}_{i}^{\text{pre}} x_{i}, \mathcal{W}_{i}\right), \tag{4}
$$

其中$L$和$l$分别对应较深层和较浅层。与公式（2）不同，HC 中的**复合映射**$\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$:mdnice-bold-italic[无法保持特征的全局均值——这种偏差会导致信号无界放大或衰减，进而在大规模训练中引发不稳定性]。另一个需要考虑的问题是：尽管 HC 在计算量（FLOPs）方面保持了效率，但:mdnice-bold-italic[原始设计并未解决“宽残差流带来的内存访问成本”这一硬件效率问题]。这些因素共同限制了 HC 的实际可扩展性，阻碍了其在大规模训练中的应用。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/02.webp)

::: image-caption
图1 残差连接范式示意图。该图对比了（a）标准残差连接、（b）超连接（HC）以及（c）本文提出的流形约束超连接（mHC）的结构设计。与无约束的HC不同，mHC通过将矩阵投影到受约束的流形上来优化残差连接空间，从而确保模型训练的稳定性。可以把这三种结构理解成“信号传递通道”：（a）标准残差连接是“单条稳定通道”，信号按固定路径传递，虽稳定但连接方式单一；（b）HC是“多条无规则通道”，为提升性能加宽了通道、增加了连接数量，却因没设“通行规则”，训练时信号易混乱（比如强度突然变大或变小）；（c）mHC则给HC的“多条通道”加了“规则”——通过流形约束让通道里的“信号传递矩阵”符合特定规律，既保留了多通道的优势，又避免了信号混乱，让模型训练更稳定
:::

为解决这些挑战，我们提出了**流形约束超连接（Manifold-Constrained Hyper-Connections, mHC）**，如图 1(c)所示。这是一个通用框架，通过将 HC 的残差连接空间投影到特定流形（指高维空间中具有特定几何结构的子集，此处为双随机矩阵构成的集合）上，恢复恒等映射特性，同时通过严谨的基础设施优化确保效率。

具体而言，mHC 利用 Sinkhorn-Knopp 算法，将$\mathcal{H}_{l}^{\text{res}}$熵投影到 Birkhoff 多面体上。补充下概念：

- Sinkhorn-Knopp 算法：[Sinkhorn and Knopp, 1967]一种迭代算法，通过交替归一化矩阵的行和列，将矩阵转化为双随机矩阵。
- 熵投影：指在投影过程中考虑熵值，确保映射的平滑性。
- Birkhoff 多面体：由所有双随机矩阵构成的凸多面体。

该操作可将残差连接矩阵有效约束在“由双随机矩阵构成的流形”内。由于双随机矩阵的行和与列和均为 1**，$\mathcal{H}_{l}^{\text{res}} x_l$的操作本质上是对输入特征的凸组合，** :mdnice-bold-italic[即权重非负且和为 1 的特征组合方式，确保输出特征在输入特征的凸包内，避免信号异常]。

**这种特性使得信号传播具有良好的条件数** （指衡量矩阵可逆性的指标，条件数小表示矩阵更接近可逆，信号传播更稳定）：:mdnice-bold-italic[特征均值得以守恒，衡量信号强度的信号范数指标受到严格正则化] （也就是限制信号范数的范围），:mdnice-bold-italic[从而有效缓解信号消失或爆炸的风险]。此外，由于双随机矩阵集合对矩阵乘法具有封闭性（指两个双随机矩阵相乘的结果仍为双随机矩阵），复合映射$\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$也能保持这种守恒特性。

因此，mHC 能够有效维持任意深度之间恒等映射的稳定性。为确保效率，我们采用：

::: block-1
- **核融合技术，并基于 TileLang[Wang et al., 2025] 开发混合精度核**；
- 此外，通过**选择性重计算** （selective recomputing，训练时不存储部分中间激活值，反向传播时重新计算以节省内存）降低内存占用
- 并在 **DualPipe 这种流水线并行调度策略** [Liu et al., 2024b]中**精心设计通信与计算的重叠**。
:::

在语言模型预训练的大量实验中，mHC 在保持 HC 性能优势的同时，展现出卓越的稳定性和可扩展性。内部大规模训练表明，mHC 支持大规模训练，且当扩展率$n=4$时，仅引入 6.7%的额外时间开销。

## 二、相关工作

> 深度学习中的架构改进主要可分为**微观设计（Micro-Design）** 和**宏观设计（Macro-Design）** 两类。

- 微观设计关注计算块（如注意力层、前馈网络层）的内部架构，定义特征在空间、时间和通道维度上的处理方式；
- 宏观设计则确定块间的拓扑结构，即特征表示在不同层之间的传播、路由和融合方式。

### 2.1 微观设计（Micro Design）

在参数共享（parameter sharing，指多个计算单元复用同一组参数，减少参数总量）和平移不变性（translation invariance，指输入平移后输出也相应平移，是图像识别中重要的特性）的驱动下，卷积操作最初在结构化信号（如图像、语音）处理中占据主导地位。

尽管后续的变体（如深度可分离卷积[Chollet, 2017]（将标准卷积拆分为深度卷积和逐点卷积，减少计算量）和分组卷积[Xie et al., 2017]（将输入通道分组，每组独立进行卷积，平衡计算量和性能））优化了效率，但 T**ransformer[Vaswani et al., 2017]的出现仍使注意力机制（Attention）和前馈网络（FFNs）成为现代架构的基础构建块**。:mdnice-bold-italic[注意力机制实现了全局信息传播（指任意位置的特征可直接与其他位置的特征交互），而前馈网络则增强了单个特征的表示能力。]

为平衡大型语言模型（LLMs）的性能与计算需求，注意力机制已发展出多种高效变体：

::: block-1
- 例如多查询注意力（MQA）[Shazeer, 2019]：多个查询头共享一组键值对，减少内存和计算开销
- 分组查询注意力（GQA）[Ainslie et al., 2023]：将查询头分组，每组共享一组键值对，介于多头注意力和多查询注意力之间
- 多头潜在注意力（MLA）[Liu et al., 2024a]：一种基于潜在空间的注意力机制，进一步提升效率
:::

同时，前馈网络通过专家混合模型（MoE）[Fedus et al., 2022; Lepikhin et al., 2020; Shazeer et al., 2017]（由多个“专家”子网络构成，输入仅由部分专家处理，实现参数规模扩展而不成比例增加计算量）被泛化为稀疏计算范式，使得模型能够在不显著增加计算成本的前提下实现大规模参数扩展。

### 2.2 宏观设计（Macro Design）

**宏观设计控制网络的全局拓扑** [ Srivastava et al., 2015]。继残差网络（ResNet）[He et al., 2016a]之后，**密集连接网络** （DenseNet）[Huang et al., 2017]（通过密集连接将每一层的输出直接传递到后续所有层，增强特征复用）和**分形网络** （FractalNet）[Larsson et al., 2016]（采用分形结构构建网络，通过多路径并行提升性能和鲁棒性）:mdnice-bold-italic[等架构分别通过密集连接和多路径结构增加拓扑复杂度，以提升性能]。*深层聚合（DLA）[Yu et al., 2018]（通过递归方式聚合不同深度和分辨率的特征，优化跨层信息流动）进一步扩展了这一范式，实现了对不同深度和分辨率特征的递归聚合。*

**近年来，宏观设计的焦点逐渐转向“扩展残差流宽度”** [Chai et al., 2020; Fang et al., 2023; Heddes et al., 2025; Mak and Flanigan, 2025; Menghani et al., 2025; Pagliardini et al., 2024; Xiao et al., 2025; Xie et al., 2023; Zhu et al., 2024]。其中：

- 超连接（HC）[Zhu et al., 2024]引入可:mdnice-bold-italic[学习矩阵来调节不同深度特征间的连接强度]；
- 残差矩阵 Transformer（RMT）[Mak and Flanigan, 2025]:mdnice-bold-italic[用外积记忆矩阵（outer-product memory matrix，一种通过外积运算存储特征信息的矩阵）替代标准残差流，以实现特征存储；]
- 类似地，MUDDFormer[Xiao et al., 2025]采用:mdnice-bold-italic[多路径动态密集连接] （multiway dynamic dense connections，通过动态调整连接方式优化跨层信息流动）:mdnice-bold-italic[优化跨层信息流]。

:mdnice-bold-italic[尽管这些方法具有潜力，但它们【均破坏】了残差连接固有的恒等映射特性，进而引入不稳定性并阻碍可扩展性]；此外，**扩展的特征宽度还导致了显著的内存访问开销**。

::: block-1
本文提出的 mHC 基于 HC 框架，将残差连接空间约束在特定流形上以恢复恒等映射特性，同时通过严谨的基础设施优化确保效率——该方法在保留“扩展连接”拓扑优势的同时，提升了稳定性和可扩展性。
:::

## 三、预备知识

> 首先，我们明确本文使用的符号定义。在 HC 的形式化表述中，第$l$层的输入$x_l \in \mathbb{R}^{1 \times C}$通过$n$倍扩展构建隐藏矩阵$x_l=(x_{l,0}^{\top}, \dots, x_{l,n-1}^{\top})^{\top} \in \mathbb{R}^{n \times C}$，该矩阵可视为$n$个并行的残差流（即$n$条同时传递特征的路径）。这一操作有效拓宽了残差流的宽度。

为控制残差流的“读出（read-out，指从残差流中提取特征用于层计算）”、“写入（write-in，指将层输出特征写入残差流）”和“更新（updating，指调整残差流中特征的混合方式）”过程，HC 引入了三个可学习线性映射：$\mathcal{H}_{l}^{\text{pre}}$、$\mathcal{H}_{l}^{\text{post}} \in \mathbb{R}^{1 \times n}$和$\mathcal{H}_{l}^{\text{res}} \in \mathbb{R}^{n \times n}$。这些映射修改了公式（1）所示的标准残差连接，得到公式（3）的形式化表述。

$$
\begin{array}{r}
x_{l+1}=x_{l}+\mathcal{F}(x_{l},\mathcal{W}_{l}),
\end{array} \tag{1}
$$

$$
x_{l+1}=\mathcal{H}_{l}^{\text{res}} x_{l}+\mathcal{H}_{l}^{\text{post}\top} \mathcal{F}\left(\mathcal{H}_{l}^{\text{pre}} x_{l}, \mathcal{W}_{l}\right), \tag{3}
$$

在 HC 中，可学习映射由两部分系数构成：

- 依赖输入的系数，称为动态映射（dynamic mappings），即映射参数随输入特征变化而变化
- 全局固定的系数，称为静态映射（static mappings），即映射参数不依赖输入，为全局可学习偏置。

形式上，HC 的系数计算如下：

$$
\left\{
\begin{array}{l}
\tilde{x}_{l}=\text{RMSNorm}\left(x_{l}\right) \\
\mathcal{H}_{l}^{\text{pre}}=\alpha_{l}^{\text{pre}} \cdot \tanh \left(\theta_{l}^{\text{pre}} \tilde{x}_{l}^{\top}\right)+b_{l}^{\text{pre}} \\
\mathcal{H}_{l}^{\text{post}}=\alpha_{l}^{\text{post}} \cdot \tanh \left(\theta_{l}^{\text{post}} \tilde{x}_{l}^{\top}\right)+b_{l}^{\text{post}} \\
\mathcal{H}_{l}^{\text{res}}=\alpha_{l}^{\text{res}} \cdot \tanh \left(\theta_{l}^{\text{res}} \tilde{x}_{l}^{\top}\right)+b_{l}^{\text{res}},
\end{array}
\right. \tag{5}
$$

其中， - RMSNorm(·)[Zhang and Sennrich, 2019]（Root Mean Square 归一化，一种不依赖批量大小的归一化方法，通过计算特征的均方根对特征进行归一化，避免 BatchNorm 的批量依赖问题）作用于最后一个维度；
- 标量$\alpha_{l}^{\text{pre}}$、$\alpha_{l}^{\text{post}}$和$\alpha_{l}^{\text{res}} \in \mathbb{R}$是可学习的门控因子（gating factors，用于调节动态映射和静态映射的权重），初始化为较小值；
- 动态映射由线性投影（linear projections，通过矩阵乘法将输入特征映射到目标维度）生成，其参数分别为$\theta_{l}^{\text{pre}}$、$\theta_{l}^{\text{post}} \in \mathbb{R}^{1 \times C}$和$\theta_{l}^{\text{res}} \in \mathbb{R}^{n \times C}$；
- 静态映射则由可学习偏置$b_{l}^{\text{pre}}$、$b_{l}^{\text{post}} \in \mathbb{R}^{1 \times n}$和$b_{l}^{\text{res}} \in \mathbb{R}^{n \times n}$表示。

值得注意的是，引入$\mathcal{H}_{l}^{\text{pre}}$、$\mathcal{H}_{l}^{\text{post}}$和$\mathcal{H}_{l}^{\text{res}}$**这三个映射带来的计算开销可忽略不计** ——因为典型的扩展率$n$（如$n=4$）远小于输入维度$C$。:mdnice-bold-italic[通过这一设计，HC 将残差流的信息容量（指残差流能够存储和传递的特征信息量）与层的输入维度解耦——而层的输入维度通常与模型的计算复杂度（以 FLOPs 衡量）强相关]。

::: block-1
因此，HC 提供了一种新的扩展维度：**通过调整残差流宽度实现模型扩展**，这与预训练缩放定律[Hoffmann et al., 2022]中讨论的“模型计算量（FLOPs）”和“训练数据量”等传统扩展维度形成互补。
:::

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/03.webp)

::: image-caption
表 1 | 超连接（HC）组件的消融实验。该表为HC各组件（$H_{l}^{pre}$、$H_{l}^{post}$、$H_{l}^{res}$）的消融实验结果。当某个映射被禁用时，会采用固定映射保证维度一致性：$H_{l}^{pre}$使用权重为$1/n$的均匀分布，$H_{l}^{post}$使用全1权重，$H_{l}^{res}$使用单位矩阵。从表中能明显看出：只启用$H_{l}^{res}$时，绝对损失差距就降了0.022；再加上$H_{l}^{pre}$和$H_{l}^{post}$，损失差距仅再降0.005。这说明$H_{l}^{res}$（残差映射）是HC提升性能的核心，它负责残差流内部的特征混合，对模型效果影响最大
:::

尽管 HC 需要通过三个映射来处理“残差流与层输入之间的维度不匹配”问题，但表 1 所示的消融实验表明：**残差映射$\mathcal{H}_{l}^{\text{res}}$带来的性能提升最为显著**。这一发现凸显了“残差流内部有效信息交换”的关键重要性。

### 3.1 数值不稳定性（Numerical Instability）

> 尽管残差映射$\mathcal{H}_{l}^{\text{res}}$对性能至关重要，但其顺序应用（即多层中连续使用$\mathcal{H}_{l}^{\text{res}}$）会带来显著的数值不稳定性风险。

$$
x_{L}=\left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right) x_{l}+\sum_{i=l}^{L-1}\left(\prod_{j=1}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}}\right) \mathcal{H}_{i}^{\text{post}\top} \mathcal{F}\left(\mathcal{H}_{i}^{\text{pre}} x_{i}, \mathcal{W}_{i}\right), \tag{4}
$$

如公式（4）所述，当 HC 扩展到多层时，从层$l$到层$L$的有效信号传播由复合映射$\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$控制。**由于可学习映射$\mathcal{H}_{l}^{\text{res}}$是无约束的（即其参数可自由更新，无几何或数值限制），该复合映射必然偏离恒等映射** ——导致在正向传播和反向传播过程中，信号幅度（signal magnitude，即信号的强度）容易出现爆炸（explosion，信号幅度急剧增大）或消失（vanishing，信号幅度急剧减小）。**这种现象破坏了残差学习的基本前提（即依赖无阻碍的信号流动），进而导致深层或大规模模型训练过程的不稳定。**

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/04.webp)

::: image-caption
图 2 超连接（HC）的训练不稳定性。该图展示了（a）超连接（HC）相对于流形约束超连接（mHC）的绝对损失差距，以及（b）两者梯度范数的对比情况。所有实验结果均基于 270 亿参数（27B）的模型。 从图中能直观理解 HC 的训练问题：（a）里训练到约 12000 步时，HC 的绝对损失差距突然增大，意味着它的预测误差突然变高，而 mHC 的损失差距始终平稳；（b）中 HC 的梯度范数波动剧烈，梯度范数可理解为模型调整参数的 “力度”，力度忽大忽小会让训练失控，mHC 的梯度范数更平缓，确保了训练过程稳定
:::

实验证据支持这一分析。在大规模实验中，我们观察到了不稳定的损失行为，如图 2 所示。以 mHC 为基准，HC 在约 12k 步时出现意外的损失骤增（loss surge），这与梯度范数（gradient norm，衡量梯度大小的指标，梯度范数异常通常标志训练不稳定）的不稳定性高度相关。此外，对$\mathcal{H}_{l}^{\text{res}}$的分析也验证了这种不稳定性的产生机制。

为量化复合映射$\prod_{i=1}^{L-1} \mathcal{H}_{L-i}^{\text{res}}$在残差流中放大信号的程度，我们采用两个指标：

::: block-1
- 第一个指标基于复合映射行和的最大绝对值（maximum absolute row sum），用于**捕捉正向传播中的最坏情况信号扩展；**
- 第二个指标基于列和的最大绝对值（maximum absolute column sum），对应**反向传播中的信号扩展**。

我们将这些指标称为“复合映射的最大绝对增益幅度（Amax Gain Magnitude）”。
:::

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/05.webp)

::: image-caption
图3 超连接（HC）的传播不稳定性。270亿参数（27B）模型中，（a）单层映射(H_{l}^{res })和（b）复合映射(\prod_{i=1}^{L-l} H_{L-i}^{res})的传播动态。x轴为层索引，将每个标准Transformer块拆分为注意力（Attention）和前馈网络（FFN）两个独立层；y轴为最大增益幅度，计算方式为（前向信号的）最大绝对行和与（反向梯度的）最大绝对列和的平均值，且该平均值基于所选序列中的所有token。可以把这理解成看HC“信号传递的失控程度”：（a）里单层映射的增益就有波动，（b）里多层复合映射的增益更是飙到3000，远超稳定值1，这就像信号在多段管道里传递时不断放大，最终“爆管”，直接证明HC的信号传播存在严重不稳定性
:::

如图 3(b)所示，HC 的最大绝对增益幅度出现极端值（峰值达 3000），与理想值 1（恒等映射的增益幅度）存在显著偏差——这证实了残差流中存在信号爆炸问题。

### 3.2 系统开销（System Overhead）

> 尽管额外映射的线性特性使得 HC 的计算复杂度可控，但系统级开销（指硬件层面的资源消耗，如内存访问、通信成本等）带来了不可忽视的挑战。具体而言，**内存访问（I/O）成本通常是现代模型架构的主要瓶颈之一**，这一问题被广泛称为“:mdnice-bold-italic[内存墙（memory wall）”[Dao et al., 2022]（指内存访问速度远低于计算单元的处理速度，导致计算单元因等待内存数据而闲置，限制整体性能）]。**这一瓶颈在架构设计中常被忽视，但它对运行时效率（runtime efficiency，即模型训练或推理的速度）具有决定性影响。**

以广泛采用的预归一化 Transformer（pre-norm Transformer）[Vaswani et al., 2017]（指在注意力层和前馈网络层之前应用归一化的 Transformer 架构）为研究对象，我们分析了 HC 固有的 I/O 模式。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/06.webp)

::: image-caption
表 2 | 每token内存访问成本对比。分析仅统计前向传播中残差流维护带来的额外开销，不包含层函数$\mathcal{F}$的内部输入输出（I/O）成本。内存访问成本可理解为“模型读写数据的字节数”，是训练效率的关键指标（俗称“内存墙”）。从表中可见，标准残差连接每token总I/O是$2C$（读）+$C$（写），而HC因多流设计，总I/O变成$(5n+1)C + n²+2n$（读）和$(3n+1)C + n²+2n$（写）——若$n=4$，HC的内存访问量约是标准残差的4倍，会大幅拖慢训练速度，这也是HC的核心系统开销问题
:::

表 2 总结了“n 流残差设计”在单个残差层中引入的每 token 内存访问开销。分析表明，**HC 的内存访问成本增加幅度约与$n$成正比** ——若不采用融合核（fused kernels，将多个操作合并为一个计算核以减少内存访问）缓解，这种过高的 I/O 需求会显著降低训练吞吐量。

此外，由于$\mathcal{H}_{l}^{\text{pre}}$、$\mathcal{H}_{l}^{\text{post}}$和$\mathcal{H}_{l}^{\text{res}}$涉及可学习参数，反向传播（backpropagation）需要存储它们的中间激活值（intermediate activations，即前向传播过程中生成的特征值）——这导致 GPU 内存占用大幅增加，通常需要通过梯度检查点（gradient checkpointing，一种通过重计算部分激活值来节省内存的技术）才能维持可行的内存使用。再者，HC 在流水线并行中需要额外的$n$倍通信成本[Qi et al., 2024]，这会导致更大的“气泡（bubbles，指流水线中因等待数据而产生的空闲时间）”，进一步降低训练吞吐量。

## 四、方法

### 4.1 流形约束超连接（Manifold-Constrained Hyper-Connections）

> 受恒等映射原理[16]（指残差连接中浅层信号不经过修改直接传递到深层的特性，是残差网络训练稳定的关键）启发，**mHC 的核心思想是将残差映射$\mathcal{H}_{l}^\text{res}$ 约束在特定流形** （数学中具有局部欧几里得空间性质的拓扑空间，这里指通过约束将映射限制在特定结构的空间内，确保映射满足稳定性条件）上。:mdnice-bold-italic[原始恒等映射通过强制$\mathcal{H}_{l}^{\text{res}} = \boldsymbol{I}$（$\boldsymbol{I}$为单位矩阵）保证稳定性，但这一做法从根本上阻碍了残差流内部的信息交互——而这种交互对于最大化多流架构的潜力至关重要]。

因此，我们提出将残差映射投影到一个特定流形上，该流形需同时满足两个条件：

- 一是维持跨层信号传播的稳定性
- 二是促进残差流之间的相互作用以保留模型的表达能力

为此，我们将 $\mathcal{H}_{l}^{\text{res}}$ 限制为双随机矩阵（指每行和每列元素之和均为 1 且所有元素非负的矩阵）。形式上，令$\mathcal{M}^{\text{res}}$表示双随机矩阵构成的流形（也称为伯克霍夫多面体，指所有 n×n 双随机矩阵构成的凸多面体，是置换矩阵的凸包），我们将 $\mathcal{H}_{l}^{\text{res}}$ 约束为 $\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}_{l}^{\text{res}})$，其定义如下：

$$
\begin{array}{r}
\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}_{l}^{\text{res}}) := \left\{ \mathcal{H}_{l}^{\text{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_{l}^{\text{res}} \mathbf{1}_{n} = \mathbf{1}_{n}, \mathbf{1}_{n}^{\top} \mathcal{H}_{l}^{\text{res}} = \mathbf{1}_{n}^{\top}, \mathcal{H}_{l}^{\text{res}} \geq 0 \right\},
\end{array} \tag{6}
$$

其中$\mathbf{1}_{n}$表示所有元素均为 1 的 n 维向量。

值得注意的是，当$n=1$时，双随机矩阵条件退化为标量 1，从而恢复原始的恒等映射。**选择双随机矩阵这一约束，为大规模模型训练带来了多项严格的理论性质**，具体如下：

::: block-1
1. **范数保持性**：双随机矩阵的谱范数（用于衡量矩阵对向量的最大拉伸程度）被限制在 1 以内（即 $\|\mathcal{H}_{l}^{\text{res}}\|_{2} \leq 1$ ）。这意味着该:mdnice-bold-italic[可学习映射具有非扩张性，能有效缓解梯度爆炸问题]。
2. **复合封闭性**：双随机矩阵集合对矩阵乘法运算具有封闭性（:mdnice-bold-italic[指两个双随机矩阵相乘的结果仍为双随机矩阵]）。这确保了跨多层的复合残差映射$\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$仍保持双随机特性，从而在模型的整个深度范围内维持稳定性。
3. **基于伯克霍夫多面体的几何解释**：集合$\mathcal{M}^{\text{res}}$构成伯克霍夫多面体，它是置换矩阵（指每行和每列恰好有一个 1，其余元素均为 0 的矩阵，对应元素的置换操作）集合的凸包（指包含该集合所有点的最小凸集）。这提供了清晰的几何意义：残差映射本质上是置换操作的凸组合（指多个置换矩阵按权重加权求和，权重非负且和为 1）。:mdnice-bold-italic[从数学角度看，重复应用此类矩阵会单调地增强流之间的信息混合，从而起到稳健的特征融合作用。]
:::

此外，我们还对输入映射 $\mathcal{H}_{l}^{\text{pre}}$ 和输出映射 $\mathcal{H}_{l}^{\text{post}}$**施加非负性约束**。:mdnice-bold-italic[该约束可避免因正负系数组合导致的信号抵消] （指正负系数相乘后相互抵消，削弱有效信号的现象），这也可视为一种特殊的流形投影。

### 4.2 参数化与流形投影（Parameterization and Manifold Projection）

> 本节详细介绍 mHC 中 $\mathcal{H}_{l}^{\text{pre}}$、$\mathcal{H}_{l}^{\text{post}}$ 和 $\mathcal{H}_{l}^{\text{res}}$ 的计算过程。

给定第 $l$ 层的输入隐藏矩阵 $\boldsymbol{x}_{l} \in \mathbb{R}^{n \times C}$ ，我们首先将其向量化（即将矩阵 $\boldsymbol{x}_{l}$ 按列堆叠成一维向量）得到 $\overrightarrow{\boldsymbol{x}}_{l} = \text{vec}(\boldsymbol{x}_{l}) \in \mathbb{R}^{1 \times nC}$，以保留完整的上下文信息。随后，我们沿用原始 HC 的形式，计算动态映射（依赖输入数据的映射，随输入变化而变化）和静态映射（不依赖输入数据的映射，由可学习偏置参数决定），具体如下：

$$
\left(
\begin{array}{l}
\overrightarrow{\boldsymbol{x}}_{l}' = \text{RMSNorm}\left( \overrightarrow{\boldsymbol{x}}_{l} \right) \\
\tilde{\mathcal{H}}_{l}^{\text{pre}} = \alpha_{l}^{\text{pre}} \cdot \left( \overrightarrow{\boldsymbol{x}}_{l}' \boldsymbol{\varphi}_{l}^{\text{pre}} \right) + b_{l}^{\text{pre}} \\
\tilde{\mathcal{H}}_{l}^{\text{post}} = \alpha_{l}^{\text{post}} \cdot \left( \overrightarrow{\boldsymbol{x}}_{l}' \boldsymbol{\varphi}_{l}^{\text{post}} \right) + b_{l}^{\text{post}} \\
\tilde{\mathcal{H}}_{l}^{\text{res}} = \alpha_{l}^{\text{res}} \cdot \text{mat}\left( \overrightarrow{\boldsymbol{x}}_{l}' \boldsymbol{\varphi}_{l}^{\text{res}} \right) + b_{l}^{\text{res}},
\end{array}
\right) \tag{7}
$$

其中 - $\boldsymbol{\varphi}_{l}^{\text{pre}}$、$\boldsymbol{\varphi}_{l}^{\text{post}} \in \mathbb{R}^{nC \times n}$以及 $\boldsymbol{\varphi}_{l}^{\text{res}} \in \mathbb{R}^{nC \times n^2}$ 是用于动态映射的线性投影矩阵；
- $\text{mat}(\cdot)$ 表示矩阵化操作（即将一维向量重塑为$n \times n$矩阵）；
- $\text{RMSNorm}(\cdot)$ 是根均方层归一化（一种归一化操作，通过计算输入的均方根来标准化数据，减少训练中的内部协变量偏移）[33]。

随后，通过以下操作得到最终的约束映射：

$$
\left\{
\begin{array}{l}
\mathcal{H}_{l}^{\text{pre}} = \sigma\left( \tilde{\mathcal{H}}_{l}^{\text{pre}} \right) \\
\mathcal{H}_{l}^{\text{post}} = 2\sigma\left( \tilde{\mathcal{H}}_{l}^{\text{post}} \right) \\
\mathcal{H}_{l}^{\text{res}} = \text{Sinkhorn-Knopp}\left( \tilde{\mathcal{H}}_{l}^{\text{res}} \right),
\end{array}
\right. \tag{8}
$$

其中， - $\sigma(\cdot)$表示 Sigmoid 函数（一种常用激活函数，公式为$\sigma(x) = 1/(1+e^{-x})$，输出值在 0 到 1 之间，用于引入非线性并限制映射输出范围）；
- $\text{Sinkhorn-Knopp}(\cdot)$（辛克霍恩-克诺普算法）是一种通过迭代行和列归一化将矩阵转化为双随机矩阵的算法，其操作步骤如下：首先通过指数运算使矩阵所有元素变为正值，然后交替进行行归一化和列归一化，直到矩阵满足双随机条件。

具体而言，以正矩阵$\boldsymbol{M}^{(0)} = \exp(\tilde{\mathcal{H}}_{l}^{\text{res}})$（$\exp(\cdot)$为指数函数，确保元素非负）为初始值，归一化迭代过程定义为：

$$
\boldsymbol{M}^{(t)} = \mathcal{T}_{r}\left( \mathcal{T}_{c}(\boldsymbol{M}^{(t-1)}) \right), \tag{9}
$$

其中 $\mathcal{T}_{r}$ 和 $\mathcal{T}_{c}$ 分别表示行归一化（将每行元素除以该行元素之和，使每行和为 1）和列归一化（将每列元素除以该列元素之和，使每列和为 1）操作。当迭代次数 $t_{\text{max}} \to \infty$ 时，该过程收敛到双随机矩阵 $\mathcal{H}_{l}^{\text{res}} = \boldsymbol{M}^{(t_{\text{max}})}$。在实验中，我们:mdnice-bold-italic[选择 $t_{\text{max}} = 20$ 作为实际迭代次数（兼顾收敛效果和计算效率）。]

### 4.3 高效基础设施设计（Efficient Infrastructure Design）

> 本节详细介绍为 mHC 定制的基础设施设计。通过严格的优化，我们在大规模模型中实现了 mHC（扩展率$n=4$），且**训练额外开销仅为 6.7%（** 边际开销，指相比基础模型增加的少量开销）。

#### 4.3.1 核融合（Kernel Fusion）

我们观察到，在 mHC 中，:mdnice-bold-italic[当对高维度（$nC$维）隐藏状态$\overrightarrow{\boldsymbol{x}}_{l} \in \mathbb{R}^{1 \times nC}$执行 RMSNorm 操作时，会产生显著的延迟]。为解决这一问题，我们：

::: block-1
- **调整了归一化操作的顺序，将“除以范数”的操作放在矩阵乘法之后执行**。该优化在保持数学等价性（指优化前后的计算结果完全相同）的同时提升了效率。
- 此外，我们**采用混合精度策略** 以最大化数值精度且不损失速度，并将多个具有共享内存访问的**操作融合** 为一个统一的计算核（指将多个独立的计算步骤合并为一个硬件可直接执行的计算单元），从而减少内存带宽瓶颈。
:::

$$
\boldsymbol{\varphi}_{l}: \text{tfloat32} \quad [nC, n^2 + 2n] \tag{10}
$$

$$
\overrightarrow{\boldsymbol{x}}_{l}: \text{bfloat16} \quad [1, nC] \tag{11}
$$

$$
\alpha_{l}^{\text{pre}}, \alpha_{l}^{\text{post}}, \alpha_{l}^{\text{res}}: \text{float32} \quad \text{Scalar} \tag{12}
$$

$$
b_{l} : \text{float32} \quad [1, n^2 + 2n] \tag{13}
$$

基于式(10)至式(13)中详细的输入和参数，我们实现了三个专门的 mHC 核，分别用于计算 $\mathcal{H}_{l}^{\text{pre}}$、$\mathcal{H}_{l}^{\text{post}}$ 和 $\mathcal{H}_{l}^{\text{res}}$。在这些核中，偏置项和线性投影矩阵被整合到 $b_{l}$ 和 $\boldsymbol{\varphi}_{l}$ 中，且 RMSNorm 的权重也被吸收到 $\boldsymbol{\varphi}_{l}$ 中（即通过矩阵运算将 RMSNorm 的权重融入线性投影过程，避免单独的权重乘法操作）。

$$
\left[ \tilde{\mathcal{H}}_{l}^{\text{pre}}, \tilde{\mathcal{H}}_{l}^{\text{post}}, \tilde{\mathcal{H}}_{l}^{\text{res}} \right]: \text{float32} = \overrightarrow{\boldsymbol{x}}_{l} \boldsymbol{\varphi}_{l} \tag{14}
$$

$$
r : \text{float32} = \frac{\|\overrightarrow{\boldsymbol{x}}_{l}\|_{2}}{\sqrt{nC}} \tag{15}
$$

式(14)至式(15)：我们设计了一个统一的核，该核融合了对 $\overrightarrow{\boldsymbol{x}}_{l}$ 的两次扫描操作（指对输入数据的遍历计算），并利用矩阵乘法单元（硬件中专门用于执行矩阵乘法的计算模块）最大化内存带宽利用率。反向传播过程（包含两次矩阵乘法）也被整合到一个单独的核中，避免了 $\overrightarrow{\boldsymbol{x}}_{l}$ 的冗余重新加载。这两个核都采用了精心调整的流水线，以高效处理混合精度计算。

$$
\left[ \tilde{\mathcal{H}}_{l}^{\text{pre}}, \tilde{\mathcal{H}}_{l}^{\text{post}}, \tilde{\mathcal{H}}_{l}^{\text{res}} \right]: \text{float32} = \frac{1}{r} \left[ \alpha_{l}^{\text{pre}} \tilde{\mathcal{H}}_{l}^{\text{pre}}, \alpha_{l}^{\text{post}} \tilde{\mathcal{H}}_{l}^{\text{post}}, \alpha_{l}^{\text{res}} \tilde{\mathcal{H}}_{l}^{\text{res}} \right] + b_{l} \tag{16}
$$

$$
\mathcal{H}_{l}^{\text{pre}}: \text{float32} = \sigma\left( \tilde{\mathcal{H}}_{l}^{\text{pre}} \right) \tag{17}
$$

$$
\mathcal{H}_{l}^{\text{post}}: \text{float32} = 2\sigma\left( \tilde{\mathcal{H}}_{l}^{\text{post}} \right) \tag{18}
$$

式(16)至式(18)：这些针对小系数的轻量级操作（指计算量小、涉及参数维度低的操作）被 opportunistically（ opportunistically 此处指“适时地”，即根据计算流程的空闲时间或数据可用性，将这些小操作合并执行）融合到一个核中，显著减少了核启动开销。

$$
\mathcal{H}_{l}^{\text{res}} : \text{float32} = \text{Sinkhorn-Knopp} \left( \tilde{\mathcal{H}}_{l}^{\text{res}} \right) \tag{19}
$$

式(19)：我们在一个单独的核中实现了辛克霍恩-克诺普迭代。对于反向传播，我们设计了一个自定义反向核，该核在芯片上（片上缓存）重新计算中间结果，并遍历整个迭代过程（完整执行辛克霍恩-克诺普算法的所有迭代步骤）。

利用上述核得到的系数，我们额外引入两个核来应用这些映射：一个用于计算 $F_{\text{pre}} := \mathcal{H}_{l}^{\text{pre}} \boldsymbol{x}_{l}$，另一个用于计算 $F_{\text{post,res}} := \mathcal{H}_{l}^{\text{res}} \boldsymbol{x}_{l} + \mathcal{H}_{l}^{\text{post}\top} F(\cdot, \cdot)$。**通过将 $\mathcal{H}_{l}^{\text{post}}$ 和 $\mathcal{H}_{l}^{\text{res}}$ 的应用与残差合并（指将残差流与主计算流的结果相加的操作）过程融合，该核将读取的元素数量从 $(3n+1)C$ 减少到 $(n+1)C$，并将写入的元素数量从 $3nC$ 减少到 $nC$。**

> 我们使用 TileLang[32]高效实现了大部分核（式(14)至式(15)除外）。该框架简化了具有复杂计算流程的核的实现过程，并使我们能够以最小的工程工作量充分利用内存带宽。

#### 4.3.2 重计算（Recomputing）

n 流残差设计在训练过程中引入了显著的内存开销。为缓解这一问题，我们在前向传播后丢弃 mHC 核的中间激活值（前向传播中生成的、用于反向传播计算梯度的数据），并在反向传播时通过重新执行 mHC 核（不包含计算量较大的层函数$F$）来实时重新计算这些激活值，这一过程称为重计算。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/07.webp)

::: image-caption
表 3 | 存储与重计算的中间激活值。该表列出了反向传播中需保留的每token激活值，以及在$L_{r}$个连续层中需重计算的临时激活值。其中$l_{0}$代表$L_{r}$层中的第一层，$l$代表$L_{r}$层内的任意一层（即$l \in [l_{0}, l_{0}+L_{r}-1]$）。中间激活值是模型训练中产生的“临时数据”，存多了会占满GPU内存。mHC因多流设计内存开销大，所以采用“存关键+重计算”策略：只存每$L_{r}$层的第一层输入$x_{l_{0}}$和每层的$\mathcal{F}(H_{l}^{pre}x_{l},\mathcal{W}_{l})$，其他临时激活值（如$x_{l}$）反向传播时再重新计算。这样既能省内存，又通过公式算出最优$L_{r}$（约$\sqrt{\frac{nL}{n+2}}$），避免重计算过多影响速度
:::

因此，对于由$L_{r}$个连续层组成的块，我们只需存储第一个层的输入$\boldsymbol{x}_{l_0}$。表 3 总结了为反向传播保留的中间激活值，其中不包含轻量级系数（指维度小、占用内存少的映射系数），但包含$F$内部的预归一化（pre-norm，指在层函数$F$的计算前对输入进行归一化的操作）相关激活值。

由于 mHC 核的重计算是针对 $L_r$ 个连续层组成的块进行的，因此对于总层数为 $L$ 的模型，我们必须为所有 $\lceil \frac{L}{L_r} \rceil$ 个块（ $\lceil \cdot \rceil$ 表示向上取整，即当 $L$ 不能被 $L_r$ 整除时，最后一个块包含剩余的层）持久化存储第一个层的输入 $\boldsymbol{x}_{l_0}$，以供反向传播使用。除了这部分常驻内存外，重计算过程还会为当前活跃块（指正在进行反向传播的层块）引入 $(n+2)C \times L_r$ 个元素的临时内存开销（指仅在重计算过程中临时占用、计算结束后释放的内存），这部分开销决定了反向传播过程中的峰值内存使用量（指训练过程中内存占用的最大值）。

因此，我们通过最小化与 $L_r$ 相关的总内存占用量来确定最优块大小 $L_r^*$，计算公式如下：

$$
L_r^* = \arg\min_{L_r} \left[ nC \times \left\lceil \frac{L}{L_r} \right\rceil + (n+2)C \times L_r \right] \approx \sqrt{\frac{nL}{n+2}}. \tag{20}
$$

::: block-1
此外，**大规模训练中的流水线并行对重计算块有一个约束：重计算块不能跨越流水线阶段边界** （指不同设备负责的层范围，称为“阶段”，块需完全包含在单个阶段内）。:mdnice-bold-italic[我们观察到，理论最优块大小$L_r^$通常与每个流水线阶段的层数一致，因此我们选择将重计算块的边界与流水线阶段边界同步] （即重计算块的范围与流水线阶段的层范围完全相同）。
:::

#### 4.3.3 DualPipe 中的通信重叠（Overlapping Communication in DualPipe）

> 在大规模训练中，流水线并行是缓解参数和梯度内存占用的标准方法。具体而言，我们采用 DualPipe 调度策略[22]（一种高效的流水线并行调度方法，通过双流水线设计减少设备间通信与计算的空闲时间），该策略能有效重叠横向扩展互联通信流量（指多设备训练中设备间的数据传输，如专家并行和流水线并行中的数据交换）。

然而，**与单流设计相比，mHC 中提出的 n 流残差会导致流水线阶段间的通信延迟（指设备间传输数据所需的时间）显著增加**。此外，在阶段边界处，对所有$L_r$层的 mHC 核进行重计算会引入不可忽略的计算开销。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/08.webp)

::: image-caption
图 4 | 流形约束超连接（mHC）的通信-计算重叠调度。可以看出我们对双管道调度方案（DualPipe）的扩展，以应对mHC引入的额外开销。图中各模块的长度仅为示意，不代表实际耗时。其中（F）、（B）、（W）分别指代前向传播、反向传播、权重梯度计算；$F^A$ 和 $F^M$ 则分别对应注意力机制（Attention）和多层感知机（MLP）的核心计算单元。可以把这个调度理解成“交通疏导方案”：mHC因多流残差设计，跨管道的数据通信变多，容易出现“堵车”。扩展后的DualPipe让数据传输（通信流）和模型计算（计算流）同时进行，还把MLP的关键计算放在高优先级流里，避免注意力计算“占着车道不放”，最终减少等待时间，即使mHC有额外开销，训练效率也不会大幅下降
:::

**为解决这些瓶颈，我们扩展了 DualPipe 调度**，如图 4 所示，以在流水线阶段边界处更好地实现通信与计算的重叠。

值得注意的是：

- 为避免阻塞通信流（指数据传输的流程，若计算占用设备资源，可能导致通信无法进行），我们:mdnice-bold-italic[在专用的高优先级计算流（指硬件中用于执行计算的独立流程，高优先级流会优先获得计算资源）上执行 MLP 核] （即 FFN）层的$F_{\text{post,res}}$。
- 此外，我们:mdnice-bold-italic[不对注意力层中的长耗时操作使用持久核] （指一旦启动就持续占用计算资源的核，会阻碍其他操作执行），从而避免长时间的停滞（指设备因资源被占用而无法执行其他操作的状态）。

::: block-1
**该设计支持对重叠的注意力计算进行抢占，从而在保持计算设备处理单元高利用率的同时实现灵活调度**。此外，由于每个阶段的初始激活值$\boldsymbol{x}_{l_0}$已在本地缓存，重计算过程与流水线通信依赖解耦，换句话说，**重计算无需等待通信完成，可独立执行**。
:::

## 五、实验

### 5.1 实验设置

> 我们通过语言模型预训练（指在大规模无标注文本数据上训练模型，以学习通用语言知识的过程）验证所提方法的有效性，并对基线模型（指未引入 HC 或 mHC 的原始模型）、HC（超连接）和所提 mHC（流形约束超连接）进行对比分析。

我们基于 DeepSeek-V3[22]的混合专家（Mixture-of-Experts，MoE）架构（指将模型分为多个专家子网络，通过门控机制选择部分专家处理输入，在保证性能的同时降低计算成本）**训练了四个不同的模型变体**，以覆盖不同的评估场景。具体而言，**HC 和 mHC 的扩展率$n$均设置为 4。**

> 我们的主要研究对象是一个 270 亿参数（27B）的模型，其训练数据集大小与参数规模成比例，该模型用于验证系统级的主要结果。

- 在此基础上，我们**通过纳入较小的 30 亿参数（3B）和 90 亿参数（9B）模型（训练数据集大小同样与参数成比例）来分析计算量缩放行，从而观察不同计算量下的性能趋势**。
- 此外，为专门研究令牌（token）缩放行为（指模型性能随训练数据中令牌数量增加的变化趋势），我们还在 1 万亿令牌的固定语料库上训练了一个单独的 3B 模型。

详细的模型配置和训练超参数（学习率、批大小等）见下表。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/09.webp)

::: image-caption
表 5 | 详细模型规格与超参数。该表展示了基于DeepSeek-V3架构的30亿（3B）、90亿（9B）和270亿（27B）参数模型的架构配置，同时列出了mHC和超连接（HC）的特定超参数（包括残差流扩展率、Sinkhorn-Knopp迭代次数），以及实验中使用的优化策略和训练协议。模型规格是“模型的硬件配置”，比如层数、注意力头数决定模型“大小”；超参数是“训练的软件设置”，比如学习率控制模型“学习速度”。像mHC的扩展率n=4、Sinkhorn迭代20次，这些参数确保它既保留多流优势又稳定；不同模型的配置不同（比如27B有30层，3B只有12层），是为了适配不同计算资源，也方便其他研究者复现实验
:::

### 5.2 主要结果

> 我们首先分析 27B 模型的训练稳定性和收敛性）。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/10.webp)

::: image-caption
图 5 | 流形约束超连接（mHC）的训练稳定性。该图展示了（a）mHC和超连接（HC）相对于基准模型的绝对训练损失差距，以及（b）三种方法（基准、HC、mHC）的梯度范数对比。所有实验均基于270亿参数（27B）的模型。结果表明，mHC在损失和梯度范数两方面均展现出更优的稳定性。简单来说，绝对损失差距就是“模型预测误差和基准模型的差值”，梯度范数是“模型调整参数的‘力度’”。从图中能看到，mHC的损失差距一直平稳且低于HC，梯度范数也没出现HC那样的剧烈波动——这就像开车时，mHC始终匀速前进，而HC一会儿猛踩油门、一会儿急刹车，充分证明mHC训练更稳定
:::

如图 5（a）所示，**mHC 有效缓解了 HC 中观察到的训练不稳定性**，与基线模型相比，最终损失降低了 0.021。梯度范数（用于衡量梯度向量的大小，梯度范数异常如过大或过小通常表示训练不稳定）分析（图 5（b））进一步证实了这种稳定性提升：**mHC 的梯度范数行为显著优于 HC，保持了与基线模型相当的稳定曲线。**

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/11.webp)

::: image-caption
表 4 | 270亿参数（27B）模型的系统级基准测试结果。该表对比了基准模型、超连接（HC）和流形约束超连接（mHC）在8个不同下游基准任务上的零样本（0-shot）和少样本（few-shot）性能。mHC在所有任务上均持续优于基准模型，且在大部分任务上超过HC，证明其在大规模预训练中的有效性。这些下游任务就像“模型的考试”：比如BBH考推理、GSM8K考数学、MMLU考多学科知识；零样本是“不给例题直接考”，少样本是“给几道例题再考”。mHC在8个“考试”里都比基准好，还在BBH（+2.1%）、DROP（+2.3%）等任务上超过HC，说明它不仅稳定，实际解决问题的能力也更强
:::

表 4 展示了模型在多个不同下游基准测试，涵盖不同任务类型上的性能[3,9,17,18,21,36]。**mHC 实现了全面的性能提升，始终优于基线模型，并且在大多数任务上超过了 HC**。值得注意的是，与 HC 相比，mHC 进一步增强了模型的推理能力（指模型解决复杂逻辑问题的能力）：

- 在 BBH[35]（一种衡量多步推理能力的基准测试）上性能提升了 2.1%
- 在 DROP[10]（一种衡量阅读理解中离散推理能力的基准测试）上性能提升了 2.3%

### 5.3 缩放实验（Scaling Experiments）

> 为评估所提方法的可扩展性，我们报告了不同规模下 mHC 相对于基线模型的相对损失改进（指 mHC 的损失与基线模型损失的差值，*负值表示 mHC 损失更低，性能更好*）。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/12.webp)

::: image-caption
图 6 | 流形约束超连接（mHC）与基准模型的缩放特性对比。该图展示了mHC相对于基准模型的缩放特性：（a）计算缩放曲线，实线代表不同计算资源预算下的性能差距，每个点对应模型大小（从30亿、90亿到270亿参数）和数据集规模的最优配置；（b）token缩放曲线，展示30亿参数（3B）模型在训练过程中的性能变化轨迹，每个点代表模型在不同训练token数量时的表现。详细的架构和训练配置见表5。计算缩放可理解为“给模型更多计算资源（更大参数、更多数据）时的性能变化”，token缩放是“喂给模型更多训练数据（token）时的性能变化”。图中可见，不管是参数变大还是数据变多，mHC的性能优势都能保持，不会像有些方法那样“资源加了性能却不涨”，说明它很适合大规模训练
:::

- 图 6（a）展示了覆盖 3B、9B 和 27B 参数的计算量缩放曲线（指随着模型计算量（通常用 FLOPs，浮点运算次数衡量）增加，模型性能变化的曲线）。该曲线表明，:mdnice-bold-italic[即使在更高的计算预算（指训练过程中可使用的总计算资源）下，mHC 的性能优势仍能稳健保持，仅出现轻微的衰减] （指性能提升幅度随计算量增加而略微减小）。
- 图 6（b）展示了 3B 模型的令牌缩放曲线（指随着训练数据中令牌数量增加，模型性能变化的曲线），反映了训练过程中的性能动态变化。

这些结果共同验证了 mHC 在大规模场景下的有效性，**内部大规模训练实验进一步证实了这一结论。**

### 5.4 稳定性分析

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/05.webp)

::: image-caption
图3 超连接（HC）的传播不稳定性。270亿参数（27B）模型中，（a）单层映射(H_{l}^{res })和（b）复合映射(\prod_{i=1}^{L-l} H_{L-i}^{res})的传播动态。x轴为层索引，将每个标准Transformer块拆分为注意力（Attention）和前馈网络（FFN）两个独立层；y轴为最大增益幅度，计算方式为（前向信号的）最大绝对行和与（反向梯度的）最大绝对列和的平均值，且该平均值基于所选序列中的所有token。可以把这理解成看HC“信号传递的失控程度”：（a）里单层映射的增益就有波动，（b）里多层复合映射的增益更是飙到3000，远超稳定值1，这就像信号在多段管道里传递时不断放大，最终“爆管”，直接证明HC的信号传播存在严重不稳定性
:::

与图 3 类似，图 7 展示了 mHC 的传播稳定性（指信号在前向传播和梯度在反向传播过程中幅度保持稳定的特性）。

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/13.webp)

::: image-caption
图 7 | 流形约束超连接（mHC）的传播稳定性。该图展示了270亿参数（27B）模型中，（a）单层映射$P_{M^{res }}(H_{l}^{res })$ 和（b）复合映射$\prod_{i=1}^{L-l} P_{M^{res }}(H_{L-i}^{res })$ 的传播动态。结果表明，与HC相比，mHC显著提升了传播稳定性。这里的“映射”可理解为“信号传递规则”：单层映射是单个网络层的传递规则，复合映射是多个层叠加后的规则。Y轴的“最大增益幅度”代表信号放大或衰减的程度——mHC的增益始终接近1（理想稳定值），最高也才1.6，而HC曾飙到3000，就像mHC的信号在“匀速管道”里传递，HC的却在“过山车管道”里，稳定性差距很明显
:::

理想情况下，单层出映射应满足双随机约束，即前向信号增益（指信号经过映射后幅度的变化倍数，行和表示）和反向梯度增益（指梯度经过映射后幅度的变化倍数，列和表示）均应等于 1。然而，在实际实现中，由于辛克霍恩-克诺普算法需要限制迭代次数以保证计算效率，我们在实验中设置了 20 次迭代以获得近似解（这里就是说接近理想双随机矩阵的解，但存在微小误差）。

因此，如图 7（a）所示，反向梯度增益与 1 存在轻微偏差。在图 7（b）的复合映射（指跨多层的映射乘积）情况下，偏差有所增大但仍保持有界（指偏差在有限范围内，不会无限制增大），最大值约为 1.6。

值得注意的是，HC 的最大增益幅度接近 3000，而 mHC 将其降低了三个数量级（指从 10³ 量级降至 10⁰ 量级）。**这些结果表明，与 HC 相比，mHC 显著提升了传播稳定性，确保了前向信号和反向梯度的稳定流动。**

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/14.webp)

::: image-caption
图 8 | 可学习映射的可视化对比。该图展示了超连接（HC，第一行）和流形约束超连接（mHC，第二行）的代表性单层映射与复合映射。每个矩阵均通过对所选序列中所有token取平均计算得到；Y轴和X轴标签分别代表前向信号增益（行和）与反向梯度增益（列和）。这些矩阵就像“信号传递强度表”，数值越大代表信号传递越强，越小则越弱。HC的矩阵里有几百、甚至负数的数值，说明信号有的被放得极大、有的被抵消，传递混乱；而mHC的数值大多在0-1之间，行和列和接近1，就像信号在“均匀分配的管道”里传递，不会出现局部过载或断流，直观体现了稳定性
:::

此外，图 8 展示了代表性映射的具体数值。我们观察到，对于 HC，当最大增益较大时，其他增益值也往往较大，这表明*所有传播路径都存在普遍的不稳定性*。相比之下**，mHC 在所有情况下均能产生稳定的结果。**

## 六、结论与展望

> 本文研究发现，尽管超连接（HC）所提出的“扩展残差流宽度并多样化连接方式”能带来性能提升，但这些连接的无约束特性会导致信号发散（就是说信号在传播过程中幅度无限制地放大或衰减）。这种现象破坏了跨层信号能量的守恒性（即信号幅度在传播过程中保持相对稳定，不发生剧烈变化），引发训练不稳定性，并阻碍深度网络的可扩展性。

为解决这些挑战，我们提出了流形约束超连接（mHC）——**一个将残差连接空间投影到特定流形上的通用框架**。:mdnice-bold-italic[通过辛克霍恩-克诺普算法对残差映射施加双随机约束，mHC 将信号传播转化为特征的凸组合（多个特征按非负权重加权求和，权重和为 1，确保信号幅度稳定）。实验结果证实，mHC 有效恢复了恒等映射特性，支持稳定的大规模训练，且相比传统 HC 具有更优的可扩展性]。

::: block-1
关键在于，通过高效的基础设施级优化（如核融合、重计算、通信重叠），mHC 在实现这些改进的同时，仅引入可忽略的计算开销。
:::

作为 HC 范式的通用扩展，**mHC 为未来研究开辟了多个有前景的方向**。尽管本文采用双随机矩阵确保稳定性，但该框架仍可:mdnice-bold-italic[探索适用于特定学习目标的多种流形约束]。我们期望通过:mdnice-bold-italic[进一步研究不同的几何约束]，能开发出更优的方法，以更好地平衡模型的可塑性与稳定性。

- 几何约束：基于几何特性的映射限制，如双随机矩阵的凸包特性。
- 可塑性：模型适应数据、学习新特征的能力。
- 稳定性：模型在训练过程中避免梯度爆炸或消失、保持训练过程平稳的能力。

::: block-1
此外，我们**希望 mHC 能重新激发社区对宏观架构设计的关注**，如模型层间的拓扑结构设计，如残差连接、多流结构等。通过**深化对“拓扑结构如何影响优化过程和有效特征的表示学习”的理解**，:mdnice-bold-italic[mHC 将有助于解决当前模型的局限性，并可能为下一代基础架构的发展指明新路径。]
:::

## 参考文献

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/15.webp)

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/16.webp)

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/17.webp)

![文章插图](/articles/2026-01-02-deepseek-hc-manifold-constrained-hyper-920732df/images/18.webp)
