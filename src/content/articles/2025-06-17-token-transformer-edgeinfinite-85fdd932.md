关键词：Edge Devices、Memory-Efficient、Transformer、Infinite-Context、Inference

> 基于 Transformer 的大型语言模型（LLMs）在边缘设备上处理长序列时面临两大挑战：注意力机制的二次复杂度，以及键值（KV）缓存带来的内存需求增长。

现有的 KV 缓存优化方法在长输出任务中难以避免不可逆的 Token 逐出问题，而采用替代序列建模架构则需要在现有 Transformer 框架内付出较高的适配成本。

::: block-1为此，我们提出 EdgeInfinite，这是一种**内存高效** 的**无限上下文解决方案**，通过**可训练的内存门控模块** 将**压缩内存** 集成到基于 Transformer 的 LLMs 中。

该方法与标准 Transformer 架构完全兼容，仅需微调少量参数，并能针对长上下文和短上下文任务路由选择性激活内存门控模块。
:::

实验结果表明，EdgeInfinite 在**长上下文基准测试中性能与基于 Transformer 的基线 LLM 相当**，同时**优化了内存消耗和首 Token 生成时间（TTFT）**。![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/01.png)

- https://arxiv.org/abs/2503.22196

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/02.png)

## 本文目录

- 一、引言
- 二、相关工作
  - 2.1 创新序列模型
  - 2.2 KV 缓存优化
- 三、EdgeInfinite
  - 3.1 架构
  - 3.2 推理策略
- 四、实验
  - 4.1 实验设置
  - 4.2 结果
  - 4.3 消融研究
  - 4.4 效率
- 五、结论
- 参考文献

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/03.png)

## 一、引言

Transformer（Vaswani 等人，2017）已成为大型语言模型（LLMs）的基础框架。然而，基于 Transformer 的模型中经典注意力机制的**二次时间复杂度在处理长序列时带来显著挑战**。此外，**随着上下文长度增加，键值（KV）缓存的持续增长导致内存使用量上升**。无论是时间复杂度还是有限内存，这些挑战在智能手机等资源受限的边缘设备上尤为突出。

为应对这些挑战，目前主要提出了两种解决方案。

- 一种方法**聚焦于 KV 缓存优化** （Li 等人，2024b；Xiao 等人，2023；Zhang 等人，2023），主要**通过逐出被认为不重要的 Token 来降低注意力计算复杂度**。尽管这些方法可以提高效率，但可能面临逐出的 Token 未来无法被使用的潜在问题（Tang 等人，2024），尤其是在多轮交互（Li 等人，2024a；Qin 等人，2024）和长生成链推理（CoT）等实际场景中（Wei 等人，2022；Guo 等人，2025）。
- 第二种解决方案**探索更高效的序列建模方法**，如线性循环模型（Katharopoulos 等人，2020；Li 等人，2025）和状态空间模型（Gu 等人，2021；Gu 和 Dao，2023），以解决计算复杂度问题。然而，当前大部分工作仍围绕基于 Transformer 的模型展开，采用新结构模型将产生较高成本，阻碍其在边缘设备上的部署。

---

在这项工作中，**我们提出 EdgeInfinite，一种在边缘设备上高效处理长序列的新方法**。通过**在现有基于 Transformer 的 LLMs 上继续预训练，EdgeInfinite 保持与当前 Transformer 架构的兼容性，实现更精简和资源高效的模型开发**。

我们设计了一个**可训练的内存门控模块，仅需微调少量参数。该模块可针对长文本任务选择性加载，同时为短文本任务保留 Transformer 模型的原始参数**。这种灵活性确保基础模型参数无需额外微调，可在长文本任务上实现快速高效推理，因此非常**适合在边缘设备上部署**。

> 推理过程中，我们将 Sink Token 和 Window Token 保留在 KV 缓存中，而其他 KV 对则压缩到内存块中。这种方法使模型在推理过程中能够保留更多语义和位置信息。

::: block-1我们的贡献可总结如下：

- 提出 EdgeInfinite，**一种边缘侧无限上下文方法**，**将压缩内存与可训练内存门控模块集成**，同时**保持与原始 Transformer 架构的兼容性**。
- EdgeInfinite 在**短文本** 任务上**保持原始** 基于 Transformer 的 LLM **性能**，同时**支持长文本任务** 的高效推理，非常**适合资源受限的边缘设备** 部署。
- 在长上下文基准测试中评估 EdgeInfinite 性能，其在优化内存消耗和 TTFT 的同时，实现了与基线 Transformer 模型相当的性能。
:::

## 二、相关工作

基于经典 Transformer 的 LLM 中，**注意力机制的二次时间复杂度** 和 **KV 缓存不断增长的内存占用**，给资源受限的**边缘设备处理长序列带来了挑战**。本节重点介绍针对这些问题的最新研究进展。

### 2.1 创新序列模型

- Mamba（Gu 和 Dao，2023）和 Mamba-2（Dao 和 Gu，2024）代表了状态空间模型（SSM）（Gu 等人，2021）发展中的重要里程碑，在自然语言处理和其他任务中表现出色。
- RWKV（Peng 等人，2023，2024）结合了 RNN 和 Transformer 的优势，引入 Token 移位和优化时间混合等创新，实现了推理的线性复杂度。
- Titans（Behrouz 等人，2024）将注意力作为短期记忆与神经长期记忆模块相结合。
- Infini-Transformer（Munkhdalai 等人，2024）将长序列分割为多个块，在原始注意力机制中引入压缩内存，并在单个 Transformer 块中同时构建掩码局部注意力和长期线性注意力机制。

### 2.2 KV 缓存优化

**KV 缓存优化** 主要旨在**通过识别和丢弃不重要的 Token** 来**减少** 整体**计算** 需求。

- StreamingLLM（Xiao 等人，2023）是一种基于滑动窗口注意力的方法，通过保留最近的 Token 和 Sink Token，帮助在有效管理内存使用的同时维持模型性能。
- H2O（Zhang 等人，2023）利用注意力分数识别并保留重要 Token，同时保留最近的 Token。
- SnapKV（Li 等人，2024b）基于观察窗口识别关键注意力特征，并相应压缩 KV 缓存。
- PyramidKV（Cai 等人，2024）通过分析不同层的注意力特征，减少后续层的 KV 缓存预算。
- SCOPE（Wu 等人，2024）通过将预填充和解码阶段的 KV 缓存预算问题分开考虑，创新性地优化了 KV 缓存预算。

## 三、EdgeInfinite

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/04.png)

### 3.1 架构

**如图 1** 所示，**EdgeInfinite 的架构包含三个核心组件**：

1. 用于局部上下文建模的带旋转位置嵌入（ROPE）的分段注意力机制
2. 用于压缩和解压缩历史上下文的内存机制
3. 平衡局部注意力和基于内存的注意力的自适应内存门控模块

#### 3.1.1 带 ROPE 的分段注意力

给定输入序列 $X=[x_{1}, ..., x_{L}]^{T} \in \mathbb{R}^{L ×d}$，将其划分为大小为 $L_{seg}$ 的段，得到 N 个长度为 $L_{seg}$ 的完整段和一个长度为 $L_{res}$ 的剩余段，它们的关系可表示为：

$$
L=N \cdot L_{seg} + L_{res} \quad (1)
$$

完整段 $X_{seg} \in \mathbb{R}^{L_{seg} ×d}$ 或剩余段 $X_{res} \in \mathbb{R}^{L_{res} ×d}$ 可统一表示为 $X_{s/r} \in \mathbb{R}^{L_{s/r} ×d}$，其中“s/r”表示完整段或剩余段。我们计算注意力的查询 Q、键 K 和值 V 状态：

$$
Q=X_{s/r} W^{Q}, \quad K=X_{s/r} W^{K}, \quad V=X_{s/r} W^{V}
$$

其中，$W^{K}$、$W^{V}$ 和 $W^{Q}$ 是可训练的投影矩阵。$Q=[q_{1}, q_{2}, ..., q_{L_{s/r}}]$ 和 $K=[k_{1}, k_{2}, ..., k_{L_{s/r}}]$ 表示段 $X_{s/r}$ 中的查询和键状态，其中 $q_{i}$ 和 $k\_{i}$ 分别对应第 $i$ 个 Token 的查询和键状态。

接下来，集成 ROPE 模型（Su 等人，2024）以将位置信息融入注意力计算：

$$
q_{m}^{r} = \mathcal{R}_{m} q_{m}, \quad k_{n}^{r} = \mathcal{R}_{n} k_{n}
$$

其中，$\mathcal{R}_{m}$ 和 $\mathcal{R}_{n}$ 是位于位置 m 和 n 的旋转矩阵，$q_{m}^{r}$ 和 $k_{n}^{r}$ 表示旋转变换后的查询和键状态。应用旋转变换后，修改后的查询和键状态记为 $Q^{r}$ 和 $K^{r}$。

随后，每个段的注意力计算方式与原始 Transformer 架构（Vaswani 等人，2017）类似：

$$
A_{\text{dot}} = \text{softmax}\left(\frac{Q^{r}(K^{r})^{T}}{\sqrt{d}}\right) V \quad (4)
$$

该计算使模型能够捕获每个段内 Token 之间的依赖关系，同时通过 ROPE 模型融入位置信息。

#### 3.1.2 内存压缩-解压缩

受 Infini-Transformer（Munkhdalai 等人，2024）和线性化注意力（Katharopoulos 等人，2020）的启发，**我们引入内存压缩和解压缩机制**。**除剩余段外的所有段均进行内存压缩**。对于第 $i$ 个段，内存 $M_{i}$ 和归一化项 $z_{i}$ 计算如下：

$$
M_{i} = M_{i-1} + \sigma\left(K^{r}\right)^{T} V
$$

$$
z_{i} = z_{i-1} + \sum_{j=1}^{L_{s/r}} \sigma\left(k_{j}^{r}\right)
$$

其中，$\sigma$ 表示非线性激活函数。第一个段（i=1）的 $M_{i} \in \mathbb{R}^{d ×d}$ 和 $z_{i} \in \mathbb{R}^{d ×1}$ 均初始化为零矩阵。此处，内存 $M_{i}$ 存储前序段的键和值之间的关联，非线性激活函数和归一化主要用于确保模型训练的稳定性。

对所有段执行内存解压缩：

$$
A_{\text{mem}} = \frac{\sigma\left(Q^{r}\right) M_{i-1}}{\sigma\left(Q^{r}\right) z_{i-1}} \quad (7)
$$

其中，$A_{\text{mem}} \in \mathbb{R}^{L\_{r/s} ×d}$ 表示由当前段的内存和查询状态计算的注意力。

> 由于内存编码了前序段键值对的关联，解压缩使我们能够计算当前查询状态与过去键值状态之间的注意力，该过程使分块计算能够近似原始长序列的注意力计算。

#### 3.1.3 内存门控模块

与需要训练整个模型的 Infini-Transformer 不同，**我们的方法仅需微调内存门控模块。该模块可将基于内存的注意力与基于局部段的注意力相集成，增强模型处理长距离依赖的能力**。此外，我们的方法**支持切换到原始模型** 以进行短上下文任务的推理。

---

**内存门控模块** 是一个**可训练** 组件，**由多层感知机（MLP）和门控向量组成**。具体而言，内存注意力 $A_{\text{mem}}$ 首先通过 MLP 进行变换：

$$
\tilde{A}_{\text{mem}} = W_{2} \cdot \text{ReLU}\left(W_{1} A_{\text{mem}} + b_{1}\right) + b_{2} \quad (8)
$$

其中，$W_{1}$ 和 $W_{2}$ 是可训练权重矩阵，$b_{1}$ 和 $b\_{2}$ 是偏置向量。ReLU 激活函数引入非线性，使 MLP 能够优化基于内存的注意力，并捕获当前段与累积内存之间的复杂交互。

变换后的内存注意力 $\tilde{A}_{\text{mem}}$ 随后通过门控机制与基于局部段的注意力 $A_{\text{dot}}$ 结合。组合注意力 $A_{\text{com}}$ 计算如下：

$$
\begin{aligned}
A_{\text{com}} = \text{sigmoid}(g) \odot \tilde{A}_{\text{mem}} + (1 - \text{sigmoid}(g)) \odot A_{\text{dot}}
\end{aligned}
$$

其中，$g$ 是可训练门控向量。应用于 $g$ 的 sigmoid 函数生成门控因子，自适应控制 $\tilde{A}_{\text{mem}}$ 和 $A_{\text{dot}}$ 对组合注意力的贡献。这种自适应加权机制确保模型能够基于长序列的特定特征，动态平衡前序上下文（编码在 $\tilde{A}_{\text{mem}}$ 中）和当前上下文（编码在 $A_{\text{dot}}$ 中）的重要性。

内存门控模块作为旁路集成到注意力流程中。如果序列长度不足以划分为段，则内存为“None”，绕过内存机制，恢复为标准多头注意力。最终注意力输出 O 为：

$$
\begin{cases}
O = \left[A_{\text{com}}^{1}; \ldots A_{\text{com}}^{H}\right] W_{o} & \text{如果 Memory} \neq \text{None} \\
O = \left[A_{\text{dot}}^{1}; \ldots A_{\text{dot}}^{H}\right] W_{o} & \text{如果 Memory} = \text{None}
\end{cases}
$$

其中，$A_{\text{com}}^{h}$ 和 $A_{\text{dot}}^{h}$ 分别表示第 h 个头的组合注意力和基于局部段的注意力。该设计确保短上下文任务与基础模型的一致性，避免灾难性遗忘。

### 3.2 推理策略

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/05.png)

> EdgeInfinite 的推理策略在算法 1 和图 2 中进行了形式化和可视化。  
> 它的特点是两个主要组件：
>
> 1. 选择性 Token 保留以确保高质量推理
> 2. 处理不同输入长度的自适应长短文本路由

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/06.png)

#### 3.2.1 选择性 Token 保留

**与丢弃若干 Token** 以减少计算开销的 **KV 缓存优化方法类似**，**EdgeInfinite 显著压缩与多个 Token 相关的键状态和值状态**，但这种方法**可能潜在降低** 整体推理性能。

::: block-1
> 为解决这一问题，EdgeInfinite 在推理过程中**在 KV 缓存中保留两种重要 Token：Sink Token 和 Window Token。**

- Sink Token 表示序列的初始 Token
- Window Token 对应最近的 Token

这些 Token 对于保留语义和位置信息（Xiao 等人，2023）至关重要，它们未被压缩以确保高质量推理输出。
:::

#### 3.2.2 长短文本推理路由

EdgeInfinite 的推理策略动态适应以高效处理长文本和短文本输入。整个推理过程可分为预填充阶段和解码阶段：

##### 预填充阶段

- 对于**长输入序列** （$L \geq L_{\text{sink}} + L_{\text{window}} + L_{\text{seg}}$），**除去 Sink Token 和 Window Token 的序列被划分为 N 个块**，每个块长度为 $L_{\text{seg}}$。**每个块被压缩到内存中，Sink Token 串联在前面**。**剩余部分** （包括剩余段）**存储为 KV 缓存**。
- 对于**短输入序列** （$L < L_{\text{sink}} + L_{\text{window}} + L_{\text{seg}}$），模型**保留完整 KV 缓存，类似于原始注意力机制**。此处，$L_{\text{sink}}$ 和 $L\_{\text{window}}$ 分别是保留的 Sink Token 和 Window Token 的长度。

##### 解码阶段

模型迭代生成新 Token，直到输出序列长度达到 $L_{\text{max}}$。如果剩余序列长度等于 $L_{\text{seg}}$，则通过压缩相应段（前面串联 Sink Token）来更新内存，然后基于更新后的内存、Sink Token 的 KV 缓存和 Window Token 的 KV 状态生成输出。否则，模型直接使用当前 KV 缓存和内存生成下一个 Token。

## 四、实验

### 4.1 实验设置

**【模型与数据】**：在实验中，我们使用 BlueLM-3B（Lu 等人，2024）作为主干模型评估 EdgeInfinite，这是一种适用于边缘部署的基于 Transformer 的 LLM。训练数据集包含约 100,000 个样本，涵盖文本摘要和生成等多样化任务。

**【超参数】**：模型训练 2 个 epoch，学习率设置为 0.005。仅训练**内存门控模块（占总权重的 0.15%）**。我们将其他超参数配置如下：

::: block-1
- $L_{\text{seg}}$ 设置为 2048
- $L_{\text{sink}}$ 为 300
- $L\_{\text{window}}$ 为 200

对于不同长度的序列，保留的 KV 缓存总大小平均约为 1524 个 Token，包括 300 个 Sink Token、200 个 Window Token 和平均长度为 1024 的剩余段。
:::

**【基准测试】**：我们使用 LongBench（Bai 等人，2023）评估 EdgeInfinite，这是一个多任务长上下文基准，用于评估不同数据集上的长上下文理解能力。

**【基线】**：我们**将 EdgeInfinite 与三种基线 KV 缓存优化方法进行比较**，包括：

::: block-1
- SnapKV（Li 等人，2024b）
- PyramidKV（Cai 等人，2024）
- StreamingLLM（Xiao 等人，2023）
- 具有完整 KV 缓存的原始模型

**这三种基线的缓存大小设置为 2048，略大于 EdgeInfinite 的设置。**
:::

### 4.2 结果

**基线方法与我们的方法之间的性能比较如表 1 所示**。我们报告了每个类别的平均性能，以及所有 21 项任务的整体平均性能。

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/07.png)

总体而言，**EdgeInfinite 与其他基线相比表现出具有竞争力的性能优势**，甚至**超过了 FullKV 的性能**。在特定任务中，EdgeInfinite 在摘要生成和代码完成任务中表现相对更好，在多文档问答（QA）和少样本学习中取得了显著更优的结果。

**结果表明，KV 缓存优化方法的性能通常与 FullKV 相近或略好**。然而，EdgeInfinite 在某些任务（如 HotpotQA 和 TriviaQA）中显著优于 FullKV。

> 性能提升归因于其将长上下文序列分割为多个较短序列的策略，减少了处理过长序列导致的性能下降。

同时，EdgeInfinite 在单文档 QA 中的性能相对弱于多文档 QA，这是因为单文档 QA 需要精确答案，而多文档 QA 侧重于内容摘要。EdgeInfinite 中的内存压缩导致 KV 状态的精度损失，使其更适合生成摘要答案，而非精确检索。

### 4.3 消融研究

为**评估 EdgeInfinite 推理过程中保留特定 KV 缓存的影响**，我们进行了**消融研究**，以评估 Sink Token 和 Window Token 对推理性能的影响。这些消融实验也在 LongBench 上进行。表 2 展示了在三种条件下（移除 Sink Token、移除 Window Token、同时移除两者）不同任务类别的平均分数和整体平均分数（NeuralTalk注：原文没有表 2）。

- 移除 Sink Token 显著影响了大多数任务的结果，因为初始 Token 通常包含许多任务的重要位置和语义信息。
- 此外，移除 Window Token 也会影响整体性能。保留固定数量的 Window Token 可避免 $L\_{\text{res}}$ 过短的问题（这会导致内存压缩期间序列末尾保留为 KV 缓存的 Token 过少），该机制有效维持了推理过程中的语义连续性。

### 4.4 效率

我们**比较了 EdgeInfinite 与原始 BlueLM-3B 模型的 TTFT（首 Token 生成时间）和内存使用情况，如图 3 所示。**

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/08.png)

结果表明，**EdgeInfinite 在处理长序列时具有显著优势，资源消耗不会随序列长度快速增加**。

> 这归因于我们的方法能够在段大小内分块处理长序列，从而大幅降低计算资源需求。

## 五、结论

在本研究中，我们提出 EdgeInfinite，一种**适用于边缘设备长上下文任务** 的高效方法。

::: block-1通过**将压缩内存与可训练内存门控模块集成到基于 Transformer 的 LLMs 中**，我们**实现了无限上下文的高效推理**，同时保持**与原始 Transformer 架构的兼容性**。

此外，我们设计了**一种有效的策略，在长上下文任务推理期间保留重要 Token 以提升推理性能**，并针对短上下文任务切换至原始主干模型。
:::

我们在长上下文基准上的评估表明，EdgeInfinite 性能与基线方法相当。总之，EdgeInfinite **为资源受限的边缘设备上的长上下文任务提供了一种高效解决方案。**

## 参考文献

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/09.png)

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/10.png)

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/11.png)
