关键词：**长上下文推理**、KV 缓存、**张量并行**、:mdnice-bold-italic[低秩注意力]、MLRA

> 随着大语言模型（LLM）能力的不断提升，处理超长上下文（例如百万级 token）已成为现实需求。无论是检索增强生成（RAG）还是长思维链（CoT）推理，模型都需要在生成最终答案前维护一个庞大的上下文窗口。然而，这种能力背后隐藏着一个严峻的性能瓶颈——**KV 缓存的数据搬移**。

在基于 Transformer 的解码阶段，每个生成步骤都需要从高带宽内存（HBM）中重新加载所有历史 token 的 Key-Value 缓存。随着上下文长度增长，KV 缓存的大小线性增加，导致**数据搬移成为主要延迟来源，而非计算**。GPU 的计算单元在大多数时间里处于等待状态，利用率极低。

为了缓解这一问题，研究人员提出了多种高效注意力机制。其中，**Multi-Head Latent Attention（MLA）** 通过将 KV 缓存压缩到一个低维的**潜在头（latent head）** 中，显著减少了缓存大小，并被 DeepSeek-V2/V3 等模型采用。

::: block-1
然而，MLA 在分布式推理中遇到了新的挑战：虽然它的上投影权重可以按头切分，但其核心的 KV 缓存是一个**单一的、不可分割的潜在向量（$C^{KV}$）**。这导致在使用张量并行（TP）时，每个设备仍需加载完整的 $C^{KV}$，而无法像切分权重那样切分 KV 缓存。因此，MLA 的 KV 缓存加载量不会随 TP 设备数增加而减少，严重抵消了 TP 带来的好处。
:::

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/01.webp)

- MULTI-HEAD LOW-RANK ATTENTION
- https://arxiv.org/pdf/2603.02188
- https://huggingface.co/Soughing/MLRA
- **https://github.com/SongtaoLiu0823/MLRA**

相关推荐

- [吞吐量提高 70%、延迟降 53%！硬件高效注意力：解锁解码速度与性能极限！比 FlashMLA 快 2 倍！TTFT减少94%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889644&idx=1&sn=83b36094e969eec3586c8a965f8c3714&scene=21#wechat_redirect)
- [NVIDIA提出长上下文加速方案BLASST：动态 Softmax 阈值稀疏注意力，74%稀疏度下预填充1.6倍速解码1.4倍速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895015&idx=1&sn=00cd71529724bbe0c497f4908cb45490&scene=21#wechat_redirect)
- [超大规模训练 1.23 倍加速！WLB-LLM 工作负载均衡 4D 并行解决长上下文训练瓶颈](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890434&idx=1&sn=46c7ec1fd6bccb03606a125b19a89611&scene=21#wechat_redirect)

> 针对这一局限，来自宾夕法尼亚州立大学、康涅狄格大学、卡内基梅隆大学和加州大学洛杉矶分校的同学们，提出**Multi-Head Low-Rank Attention（MLRA）**，一种原生支持 4 路张量并行的新型注意力机制。MLRA 不仅保持了 MLA 的高性能，还通过将潜在头分解为多个可并行的分支，大幅降低了分布式解码时的 KV 加载开销。

实验表明，MLRA-4 在 2.9B 规模模型上取得了优于 MLA 的困惑度和下游任务准确率，并在**长上下文解码中实现高达 2.8 倍的加速**。下面，我们拆解 MLRA 的设计思想与技术细节。

## 本文目录

- 一、背景知识：从 MHA 到 MLA
  - 1.1 标准多头注意力（MHA）与 KV 缓存
  - 1.2 分组查询注意力（GQA）与多查询注意力（MQA）
  - 1.3 多头潜在注意力（MLA）
  - 1.4 MLA 的张量并行瓶颈
- 二、块分解的洞察：MLA 的另一种视角
- 三、MLRA 的核心设计：将求和移到注意力之外
  - 3.1 MLRA-4
  - 3.2 MLRA-2
- 3.3 方差分析与缩放：保证训练稳定性
  - 旋转位置编码（RoPE）
  - 无位置编码（NoPE）
  - 引入缩放因子
- 四、实验验证：性能与效率的双重提升
  - 4.1 实验设置
  - 4.1 主要结果
  - 4.2 解码效率
  - 4.3 消融实验
- 五、相关工作简述
- 结论

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/02.webp)

## 一、背景知识：从 MHA 到 MLA

### 1.1 标准多头注意力（MHA）与 KV 缓存

> 标准的 Transformer 使用多头注意力（MHA），每个头独立计算查询（Q）、键（K）、值（V）。假设输入序列长度为 $n$，模型隐藏维度为 $d$，头数为 $h$，每个头的维度为 $d_h$，通常 $d = h \times d_h$。

MHA 首先将输入 $\mathbf{H} \in \mathbb{R}^{n \times d}$ 通过三个投影矩阵 $\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V \in \mathbb{R}^{d \times (h d_h)}$ 得到查询、键、值的初始表示，并应用旋转位置编码（RoPE）。

在解码阶段，已生成 token 的 K 和 V 被缓存下来（即 KV 缓存），避免重复计算。每个 token 的 KV 缓存大小为 $2 \times h \times d_h$，即 $2hd_h$。**当上下文长度 $n$ 达到百万级别时，KV 缓存的大小可达数十 GB，成为内存带宽的瓶颈。**

### 1.2 分组查询注意力（GQA）与多查询注意力（MQA）

为了减少 KV 缓存，GQA 将查询头分组，组内共享同一个 KV 头；MQA 则极端地使用单个 KV 头共享给所有查询头。设分组数为 $g$，则缓存大小降为 $2 \times g \times d_h$。**这虽然降低了缓存大小，但可能影响模型质量。**

### 1.3 多头潜在注意力（MLA）

> MLA 是 DeepSeek 提出的一种更高效的注意力机制。其核心思想是：将 KV 进一步压缩到一个更小的**潜在空间**，存储时只保留压缩后的潜在向量，在计算注意力时再通过上投影恢复完整的 K 和 V。具体流程如下：

1. **下投影**：对输入隐藏状态 $\mathbf{H}$ 进行 RMSNorm 后，用可学习矩阵 $\mathbf{W}^{\mathrm{DQ}} \in \mathbb{R}^{d \times d_c'}$ 和 $\mathbf{W}^{\mathrm{DKV}} \in \mathbb{R}^{d \times d_c}$ 分别压缩查询和 KV。其中 $d_c'$ 和 $d_c$ 是远小于 $h d_h$ 的潜在维度（DeepSeek-V3 中取 $d_c = 4d_h$，$d_c' = 12d_h$，$d_h^R = 0.5d_h$）。得到查询潜在向量 $\mathbf{C}^Q \in \mathbb{R}^{n \times d_c'}$ 和 KV 潜在向量 $\mathbf{C}^{\mathrm{KV}} \in \mathbb{R}^{n \times d_c}$。同时，还保留一个较小的 RoPE 键 $\mathbf{K}^{\mathrm{RoPE}} = \mathrm{RoPE}(\mathbf{H}\mathbf{W}^{\mathrm{KR}})$，$\mathbf{W}^{\mathrm{KR}} \in \mathbb{R}^{d \times d_h^R}$，用于位置编码。
2. **缓存**：仅缓存 $\mathbf{C}^{\mathrm{KV}}$ 和 $\mathbf{K}^{\mathrm{RoPE}}$，总大小为 $d_c + d_h^R$ 每 token。在 DeepSeek-V3 的设置中，每 token 缓存量为 $4d_h + 0.5d_h = 4.5d_h$，远小于 MHA 的 $2h d_h$。
3. **上投影与吸收**：在解码时，通过上投影矩阵 $\mathbf{W}^{\mathrm{UK}},\mathbf{W}^{\mathrm{UV}} \in \mathbb{R}^{d_c \times (h d_h)}$ 从 $\mathbf{C}^{\mathrm{KV}}$ 恢复完整的 K 和 V。但 MLA 巧妙地将这些上投影矩阵**吸收** 到查询端，避免了显式物化所有头的 K 和 V，从而大幅减少计算量。

MLA 的高效解码三步算法：

- **Step 1**：将查询与上投影矩阵结合，得到 $\tilde{\mathbf{Q}}^{\mathrm{NoPE}} = \mathrm{einsum}('nhp, hpc → nhc', \mathbf{Q}^{\mathrm{NoPE}}, \tilde{\mathbf{W}}^{\mathrm{UK}})$，其中 $\tilde{\mathbf{W}}^{\mathrm{UK}}$ 是将 $\mathbf{W}^{\mathrm{UK}}$ 按头重排后的张量。这一步相当于把 $\mathbf{W}^{\mathrm{UK}}$ 乘到查询上，从而后续只需与 $\mathbf{C}^{\mathrm{KV}}$ 交互。
- **Step 2**：将 $\mathbf{C}^{\mathrm{KV}}$ 和 $\mathbf{K}^{\mathrm{RoPE}}$ 视为共享的 KV，执行 MQA 风格的注意力，得到中间输出 $\mathbf{Z} \in \mathbb{R}^{h \times d_c}$。
- **Step 3**：用值上投影矩阵 $\tilde{\mathbf{W}}^{\mathrm{UV}}$ 将中间输出映射回最终输出 $\mathbf{O}$。

这一算法借助 FlashAttention 等内核实现极高性能。

### 1.4 MLA 的张量并行瓶颈

> 尽管 MLA 在单卡上非常高效，但当使用张量并行（TP）在多个 GPU 上分布式推理时，问题出现了：MLA 的潜在头 $\mathbf{C}^{\mathrm{KV}}$ 是**单一的**，无法像 MHA 那样按头切分。

因此，在张量并行（TP）设置下，无论 TP 的并行度如何，每个设备都必须**完整加载** 整个 $\mathbf{C}^{\mathrm{KV}}$（每 token 占用 $4.5d_h$ 的内存）。这意味着，设备所需的 KV 缓存加载量**不会随 TP 规模增大而减少**，从而导致大量冗余的数据搬移，严重制约系统扩展性。表 1 清晰地印证了这一现象。

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/03.webp)

::: image-caption
表 1 | 不同注意力机制的参数与键值缓存加载量对比，部分实验结果引自 Zhang 等人（2025）和 Zadouri 等人（2025）的研究，该表是论文的核心对比表之一，以 Qwen3-32B 和 Kimi-K2 为基础架构，对比了 MHA、MLA、MLRA 等 9 种机制的参数量、KV 缓存及不同 TP 设备数下的单设备加载量。表中数据清晰展现了 MLRA-4 仅需 4 路 TP 就能实现 $1.5 d_h$ 的加载量，优于 GQA 的 8 路 TP 要求，为其高效解码提供了量化依据。
:::

## 二、块分解的洞察：MLA 的另一种视角

> 为了设计可切分的潜在注意力，论文作者首先对 MLA 的 KV 上投影进行**块分解** 分析。

将上投影矩阵 $\mathbf{W}^{\mathrm{UK}} \in \mathbb{R}^{d_c \times (h d_h)}$ 按行切分为 4 个块（因为 $d_c = 4d_h$），每个块大小为 $d_h \times (h d_h)$。

更精细地，对于每个头 $i$，可以提取出对应列的块，记 $\mathbf{W}_{(b),(i)}^{\mathrm{UK}}$ 为 $\mathbf{W}^{\mathrm{UK}}$ 中第 $b$ 个行块（$b \in \{0,1,2,3\}$）的第 $i$ 个头对应的列块，大小为 $d_h \times d_h$。同时将潜在矩阵 $\mathbf{C}^{\mathrm{KV}} \in \mathbb{R}^{n \times 4d_h}$ 按列切分为 4 个块 $\mathbf{C}_{(b)}^{\mathrm{KV}} \in \mathbb{R}^{n \times d_h}$。于是，第 $i$ 个头的 NoPE 键可以表示为：

$$
\mathbf{K}^{\mathrm{NoPE}}_{:,i,:} = \sum_{b=0}^{3} \mathbf{C}_{(b)}^{\mathrm{KV}} \mathbf{W}_{(b),(i)}^{\mathrm{UK}}, \quad \mathbf{V}_{:,i,:} = \sum_{b=0}^{3} \mathbf{C}_{(b)}^{\mathrm{KV}} \mathbf{W}_{(b),(i)}^{\mathrm{UV}} \tag{2}
$$

::: block-1
这意味着，每个头的 NoPE 键和值是**四个块投影之和**。这一观察启发我们：**能否将求和操作从 KV 计算挪到注意力输出之后，从而让每个分支独立？**
:::

## 三、MLRA 的核心设计：将求和移到注意力之外

基于上述洞察，MLRA 的核心创新在于**将块求和从 KV 构造阶段移到注意力输出阶段**。具体来说，:mdnice-bold-italic[对于每个块 $b$，独立地计算该块的注意力输出（使用该块的 KV 投影），然后将所有块的输出相加]。这样做的好处是：每个块的潜在状态 $\mathbf{C}_{(b)}^{\mathrm{KV}}$ 可以独立地分配到不同设备上，实现真正的切分。

### 3.1 MLRA-4

> 对于 4 路切分，MLRA-4 的公式为：

$$
\mathbf{O}_{:,i,:} = \sum_{b=0}^{3} \mathrm{Softmax}\left( \tau \mathbf{Q}_{:,i,:}^{\mathrm{NoPE}} (\mathbf{C}_{(b)}^{\mathrm{KV}} \mathbf{W}_{(b),(i)}^{\mathrm{UK}})^\top + \tau \mathbf{Q}_{:,i,:}^{\mathrm{RoPE}} (\mathbf{K}^{\mathrm{RoPE}})^\top \right) (\mathbf{C}_{(b)}^{\mathrm{KV}} \mathbf{W}_{(b),(i)}^{\mathrm{UV}}) \tag{5}
$$

其中 $\tau = 1/\sqrt{d_h + d_h^R}$ 是缩放因子，$\mathbf{Q}^{\mathrm{NoPE}}$ 和 $\mathbf{Q}^{\mathrm{RoPE}}$ 来自查询潜在向量的上投影（与 MLA 相同）。每个分支的 RoPE 键 $\mathbf{K}^{\mathrm{RoPE}}$ 是共享的，因此位置信息得以保留。注意，每个分支的 Softmax 是独立计算的，这意味着**不同分支的注意力权重可能不同，从而增加了模型的表达能力。**

### 3.2 MLRA-2

> 类似地，可以构造 2 路版本（MLRA-2），对应如下公式：

$$
\begin{equation}
\begin{split}
O_{:, i,:} &= \operatorname{Softmax}\left(\tau Q_{:, i,:}^{\text{NoPE}}\left(C_{:,(2 \gamma(i))}^{\text{KV}} W_{(0),(i)}^{(\gamma(i)),\text{UK}}\right)^{\top}+\tau Q_{:, i,:}^{\text{RoPE}}\left(K^{\text{RoPE}}\right)^{\top}\right)\left(C_{:,(2 \gamma(i))}^{\text{KV}} W_{(0),(i)}^{(\gamma(i)),\text{UV}}\right) \\
&+ \operatorname{Softmax}\left(\tau Q_{:, i,:}^{\text{NoPE}}\left(C_{:,(2 \gamma(i)+1)}^{\text{KV}} W_{(1),(i)}^{(\gamma(i)),\text{UK}}\right)^{\top}+\tau Q_{:, i,:}^{\text{RoPE}}\left(K^{\text{RoPE}}\right)^{\top}\right)\left(C_{:,(2 \gamma(i)+1)}^{\text{KV}} W_{(1),(i)}^{(\gamma(i)),\text{UV}}\right).
\end{split}
\end{equation}
$$

它将头分为两组（每组 $h/2$ 个头），每组使用两个块分支，即每组的 KV 潜在维度为 $2d_h$。设分组函数 $\gamma(i)$ 将头 $i$ 映射到组 0 或组 1，则：

$$
\mathbf{O}_{:,i,:} = \sum_{b=0}^{1} \mathrm{Softmax}\left( \tau \mathbf{Q}_{:,i,:}^{\mathrm{NoPE}} (\mathbf{C}_{(2\gamma(i)+b)}^{\mathrm{KV}} \mathbf{W}_{(b),(i)}^{\gamma(i),\mathrm{UK}})^\top + \tau \mathbf{Q}_{:,i,:}^{\mathrm{RoPE}} (\mathbf{K}^{\mathrm{RoPE}})^\top \right) (\mathbf{C}_{(2\gamma(i)+b)}^{\mathrm{KV}} \mathbf{W}_{(b),(i)}^{\gamma(i),\mathrm{UV}})
$$

> 图 8 和图 9 直观展示了 MLRA-2 和 MLRA-4 的结构：

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/04.webp)

::: image-caption
图 8 双分支多头低秩注意力（MLRA-2）的结构示意图。该图直观展示了 MLRA-2 将 KV 潜态分为两个块的核心设计，输入隐藏层经处理得到两个缓存潜态，分别经 RoPE 和多头注意力计算，再拼接输出。图中清晰标注了缓存部分、RoPE 应用环节与注意力计算分支，体现了其可支持 2 路 TP 的分区潜态设计逻辑。
:::

上图是 MLRA-2 架构，两个分支分别处理部分潜在块，输出求和。每个头根据其所属组选择对应的两个块。

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/05.webp)

::: image-caption
图 9 四分支多头低秩注意力（MLRA-4）的结构示意图。该图展示了 MLRA-4 将 KV 潜态拆分为四个独立块的核心架构，输入隐藏层生成四个缓存潜态，各分支独立完成 RoPE 应用与多头注意力计算，最终拼接得到输出隐藏层。图中标注了缓存模块、分支计算流程，直观体现了其原生支持 4 路 TP 的设计优势。
:::

上图是 MLRA-4 架构，四个分支并行计算后求和。每个头都使用所有四个块。

::: block-1
这种设计使得 MLRA 天然支持张量并行：可以将不同的块 $\mathbf{C}_{(b)}^{\mathrm{KV}}$ 分配到不同的设备上，每个设备只需加载自己负责的块（大小为 $d_h$ 每 token），从而将每设备 KV 加载量从 MLA 的 $4.5d_h$ 降低到 $1.5d_h$。

对于 4 路 TP，因为 $d_c=4d_h$ 被切分到 4 个设备，每个设备负责一个大小为 $d_h$ 的块。加上所有设备仍需加载的、大小为 $0.5d_h$ 的共享 RoPE 键，**每个设备实际加载量为 $d_h + 0.5d_h = 1.5d_h$。**
:::

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/06.webp)

::: image-caption
表 1 | 不同注意力机制的参数与键值缓存加载量对比，部分实验结果引自 Zhang 等人（2025）和 Zadouri 等人（2025）的研究，该表是论文的核心对比表之一，以 Qwen3-32B 和 Kimi-K2 为基础架构，对比了 MHA、MLA、MLRA 等 9 种机制的参数量、KV 缓存及不同 TP 设备数下的单设备加载量。表中数据清晰展现了 MLRA-4 仅需 4 路 TP 就能实现 $1.5 d_h$ 的加载量，优于 GQA 的 8 路 TP 要求，为其高效解码提供了量化依据。
:::

如表 1 所示，MLRA-4 在 **4 路 TP 下每设备加载量仅为 $1.5 d_h$，与 8 路 TP 的 GTA 相当，但 MLRA-4 只需 4 路即可达到。**

## 3.3 方差分析与缩放：保证训练稳定性

> 论文作者发现，MLA 及类似结构中，不同路径产生的张量（如 NoPE 键和 RoPE 键）可能存在**方差失配** 问题，影响训练稳定性。

通过假设各权重矩阵元素独立同分布且均值为 $0$，以及输入经过 RMSNorm 后均方约为 $1$，作者推导了注意力机制中关键部分的方差表达式。

### 旋转位置编码（RoPE）

> 下面公式表示经旋转位置编码（RoPE）处理后的键 $\mathbf{K}^{\mathrm{RoPE}}$ 的方差。

$$
\mathrm{Var}(\mathbf{K}^{\mathrm{RoPE}}) \approx d \sigma_w^2
$$

在MLA中，$\mathbf{K}^{\mathrm{RoPE}}$ 由原始隐藏状态 $\pmb{H} \in \mathbb{R}^{n \times d}$ 经线性投影 $\pmb{W}^{\mathrm{KR}} \in \mathbb{R}^{d \times d_h^R}$ 后，再应用RoPE得到，即 $\mathbf{K}^{\mathrm{RoPE}} = \mathrm{RoPE}(\pmb{H}\pmb{W}^{\mathrm{KR}})$。由于 $\pmb{H}$ 经过RMSNorm，其元素均方近似为1，而权重 $\pmb{W}^{\mathrm{KR}}$ 的方差为 $\sigma_w^2$，根据方差传播性质，其输出 $\overline{\mathbf{K}}^{\mathrm{RoPE}} = \pmb{H}\pmb{W}^{\mathrm{KR}}$ 的方差即为 $d\sigma_w^2$。由于RoPE是正交变换，不改变向量的模长，因此 $\mathbf{K}^{\mathrm{RoPE}}$ 的方差也近似为 $d\sigma_w^2$。

### 无位置编码（NoPE）

> 下面公式表示无位置编码（NoPE）的键 $\mathbf{K}^{\mathrm{NoPE}}$ 的方差。

$$
\mathrm{Var}(\mathbf{K}^{\mathrm{NoPE}}) \approx d_c \sigma_w^2
$$

在MLA中，首先通过一个共享的潜变量 $\mathbf{C}^{\mathrm{KV}} \in \mathbb{R}^{n \times d_c}$（由 $\pmb{H}$ 经低秩投影得到）来压缩KV缓存。随后，通过一个上投影矩阵 $\pmb{W}^{\mathrm{UK}} \in \mathbb{R}^{d_{c}\times (h d_{h})}$ 将其投影回高维空间，得到 $\overline{\mathbf{K}}^{\mathrm{NoPE}} = \mathbf{C}^{\mathrm{KV}}\pmb{W}^{\mathrm{UK}}$。最后，将 $\overline{\mathbf{K}}^{\mathrm{NoPE}}$ 重塑（Reshape）为 $[n, h, d_h]$ 的张量，即得到 $\mathbf{K}^{\mathrm{NoPE}}$，它包含了所有注意力头的键信息。由于 $\mathbf{C}^{\mathrm{KV}}$ 也经过RMSNorm，其元素均方近似为1，且权重 $\pmb{W}^{\mathrm{UK}}$ 的方差为 $\sigma_w^2$，因此 $\overline{\mathbf{K}}^{\mathrm{NoPE}}$ 的方差约为 $d_c\sigma_w^2$，此方差在随后的重塑操作中保持不变。

::: block-1
将这两个方差表达式对比，可以发现一个关键现象：**方差失配（Variance Mismatch）**。由于潜空间维度 $d_c$ 通常远小于原始隐藏维度 $d$（即 $d_c \ll d$），导致 $\mathrm{Var}(\mathbf{K}^{\mathrm{NoPE}})$ 远小于 $\mathrm{Var}(\mathbf{K}^{\mathrm{RoPE}})$。

:mdnice-bold-italic[这种量级上的差异会破坏注意力计算中不同组分的平衡，影响模型训练的稳定性与最终性能。正是基于这一观察，论文在后续引入了方差校准（Calibration）机制，通过引入缩放因子 $\alpha_{kv} = \sqrt{d/d_c}$ 对潜变量 $\mathbf{C}^{\mathrm{KV}}$ 进行缩放，使得 $\mathrm{Var}(\mathbf{K}^{\mathrm{NoPE}})$ 与 $\mathrm{Var}(\mathbf{K}^{\mathrm{RoPE}})$ 的量级重新匹配，从而保证了模型各部分在统计上的协调性]。因此，上面两个方差的公式不仅是理论上的分析，更是后续提出关键改进方法（Scaling）的理论基石。
:::

### 引入缩放因子

> 由于 $d_c$ 通常远小于 $d$（例如 $d_c = 4d_h$，而 $d = h d_h$，在 2.9B 模型中 $h=24$，$d=3072$，$d_c=512$，$d_c$ 远小于 $d$），NoPE 键的方差可能远小于 RoPE 键。因此，需要对潜在状态进行**缩放**，使得后续组件的方差匹配。

论文引入了缩放因子 $\alpha_q = \sqrt{d/d_c'}$ 和 $\alpha_{kv} = \sqrt{d/d_c}$（对于 MLA）。对于 MLRA，由于多个分支求和，还需要对最终输出进行缩放：

$$
\mathbf{O} \leftarrow \frac{1}{\sqrt{2}} \mathbf{O} \quad (\text{MLRA-2}), \quad \mathbf{O} \leftarrow \frac{1}{2} \mathbf{O} \quad (\text{MLRA-4})
$$

> 图 2 的消融实验证明了缩放的有效性：

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/07.webp)

::: image-caption
图 2 缩放对训练损失的影响。模型未使用缩放策略与使用缩放策略的损失差值，计算方式为用未缩放模型的损失减去缩放模型的损失。该差值用于验证方差校准缩放策略的有效性，差值为正代表缩放能降低训练损失。实验仅针对 MLA、GLA-2 和 MLRA-2 展开，随训练推进差值保持稳定且大于 0，说明缩放能解决 RoPE 键与其他组件的方差失配问题，提升模型收敛效果。
:::

可以看出应用缩放后，MLA、GLA-2、MLRA-2 的损失曲线更低，收敛更好。此外，论文还通过初始化、双倍头数等消融实验验证了设计选择的合理性。

## 四、实验验证：性能与效率的双重提升

### 4.1 实验设置

作者在 2.9B 参数规模下，基于 Llama-3 架构，在 FineWeb-Edu 等数据集上预训练了多种注意力机制（MHA、GQA、MLA、GLA、MLRA 等），保持总参数大致相同（通过调整 FFN 维度）。训练了 100B token，并在多个下游任务上评估零样本准确率。

### 4.1 主要结果

表 3 和表 4 展示了关键结果：

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/08.webp)

::: image-caption
表 3 各模型在多个数据集上的困惑度。七种数据集上的验证集困惑度对比（数值越低越好），数据集包括维基百科、C4、Pile、RefinedWeb、Cosmopedia、FineWeb 和 FineWeb-Edu，最优结果用粗体标注，次优结果用下划线标注。该表是 2.9B 参数量模型的核心性能对比表，所有模型均在 FineWeb-Edu100B 预训练 100K 步。实验结果显示 MLRA-4 在所有数据集上均取得最优或次优困惑度，平均困惑度 13.672 显著低于 MLA 的 13.727 和 GQA 的 14.139，验证了 MLRA 在语言建模能力上的优势。
:::

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/09.webp)

::: image-caption
表 4 各模型在 7 个常识推理下游任务上的平均准确率。数据集包括 ARC-E、ARC-C、OpenBookQA、BoolQ、HellaSwag、Winogrande 和 PIQA。ARC-E/C、OpenBookQA、HellaSwag 和 PIQA 采用标准化准确率（%），其余数据集采用标准准确率（%），最优结果粗体标注，次优结果下划线标注。该表采用零样本评估方式，基于 lm-evaluation-harness 工具完成。MLRA-4 以 58.84% 的平均准确率位居第一，略高于 MLA 的 58.75%，显著优于 GQA 的 57.89%，其中在 Winogrande 和 OpenBookQA 上的准确率表现最优，证明 MLRA 的架构设计未牺牲下游任务能力，反而提升了推理性能。
:::

MLRA-4 在困惑度和下游任务上均达到最优，证明了其强大的表示能力。

### 4.2 解码效率

> 图 5 和图 6 展示了解码速度与吞吐量：

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/10.webp)上图 5，展示了在单 H100 上，MLRA-4 解码延迟随序列长度增长最慢，**相比 GQA 有 1.05-1.26 倍加速，相比 MLA 稳定快约 2.8 倍（得益于 4 路 TP）。**

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/11.webp)

::: image-caption
图 6 GQA、MLA、GLA-2 和 MLRA-4 四种机制在批次大小为 128 时，解码吞吐量与序列长度的关系。实验基于 8 张 NVIDIA H100 80GB GPU，序列长度 1K 至 16Ktokens，各机制采用不同的 TP/DP 配置以消除冗余 KV 缓存加载。结果表明 MLRA-4（TP=4/DP=2）在全序列长度下吞吐量最高，短序列时与 GQA 持平，长序列优势显著，优于 MLA 的纯 DP 配置。
:::

上图 6 在 8 卡 H100 上，MLRA-4（TP=4/DP=2）在**各种序列长度下吞吐量最高，尤其长序列优势明显。**

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/12.webp)

::: image-caption
表 2 不同注意力机制的注意力解码算术强度对比。算术强度是衡量计算与内存访问比的核心指标，决定模型是计算受限还是内存受限。该表计算了各机制在长上下文解码中的算术强度，结果显示 MLRA-2 和 MLRA-4 分别达到 $h$ 和 $2h$，保持了与 MLA、GLA-2 相当的高算术强度，将解码从内存瓶颈转向计算瓶颈，提升 GPU 利用率。
:::

表 2 还对比了各方法的算术强度（AI），MLRA-4 达到 $2h$，远高于 GQA 等，**表明其计算密集度更高，更接近计算瓶颈而非内存瓶颈。**

### 4.3 消融实验

> **初始化**：图 1 和表 38 显示，将输出投影初始化为零比高斯初始化更好。

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/13.webp)

::: image-caption
图 1 | 高斯分布初始化（均值 0、标准差 0.02）与零初始化的损失差值，计算方式为用高斯分布初始化的损失减去零初始化的损失。该指标直观展现两种初始化策略的训练损失差异，差值为正则说明零初始化损失更低。实验覆盖所有对比的注意力机制，随训练 token 量增加，差值变化趋势稳定，验证了零初始化在注意力和 FFN 输出投影层的优越性，为后续所有实验确定了初始化基准方案。
:::

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/14.webp)

::: image-caption
表 38 七种数据集上的验证集困惑度对比（数值越低模型性能越好），数据集包含维基百科、C4、Pile、RefinedWeb、Cosmopedia、FineWeb 和 FineWeb-Edu。本实验对比了零初始化与高斯分布初始化（均值 0、标准差 0.02）两种策略，且该初始化仅作用于注意力和前馈网络的输出投影权重。该表是初始化消融实验的核心量化结果，聚焦模型关键的输出投影层权重初始化，覆盖主流的语言建模数据集，对比了所有实验涉及的注意力机制。实验结果清晰显示，零初始化在绝大多数机制下能降低验证集困惑度，为论文后续所有实验统一了初始化方案，保障了实验结论的有效性与一致性。
:::

> **双倍头数**：图 3 和表 40 表明，单纯增加头数而不改变 KV 缓存大小无益。

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/15.webp)

::: image-caption
图 3 模型使用双倍注意力头数与未使用的损失差值，计算方式为用双倍头数模型的损失减去原头数模型的损失。该实验为验证注意力头数增加的效果，仅针对 GQA、MLA、GLA-2 调整，且保持 KV 缓存大小与参数量不变。差值始终为正，说明双倍头数会提升训练损失，无法降低困惑度，证明原头数配置更合理，也为 MLRA 多分支设计的合理性提供了对照依据。
:::

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/16.webp)

::: image-caption
表 40 七种数据集上的验证集困惑度对比（数值越低模型性能越好），数据集包含维基百科、C4、Pile、RefinedWeb、Cosmopedia、FineWeb 和 FineWeb-Edu。本分析专门对比了模型使用双倍注意力头数与未使用双倍注意力头数的性能表现。该表是双倍注意力头数消融实验的核心结果，实验仅针对 GQA、MLA、GLA-2 展开，且保持 KV 缓存大小与模型参数量不变。表中数据显示，使用双倍注意力头数后模型困惑度均上升，证明原头数配置更合理，也为 MLRA 采用多分支而非单纯增加头数的设计提供了关键实验依据。
:::

> **门控机制**：图 4 和表 5 显示，引入门控（gating）可进一步提升性能，MLRA-4+门控平均困惑度降至 13.621。

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/17.webp)

::: image-caption
图 4 模型未使用门控机制与使用门控机制的损失差值，计算方式为用未使用门控模型的损失减去使用门控模型的损失。该实验为验证门控机制对模型训练的影响，选取 GQA、MLA、GLA-2、MLRA-2 和 MLRA-4 为对象，且调整 FFN 维度保持参数量一致。差值为正且随训练 token 量增加趋势平稳，证明门控机制能有效降低训练损失，提升模型收敛效率。
:::

![文章插图](/articles/2026-03-07-2-8-sota-mlra-multi-aaad19e5/images/18.webp)

::: image-caption
表 5 加入门控机制后，七种数据集上的验证集困惑度对比（数值越低越好），最优结果用粗体标注，次优结果用下划线标注。该实验为验证门控机制的性能增益，在 GQA、MLA、GLA-2、MLRA-2 和 MLRA-4 中加入门控并调整 FFN 维度保持参数量不变。结果显示 MLRA-4 仍以 13.621 的平均困惑度居首，所有模型加入门控后困惑度均有下降，证明门控机制能在不增加参数量的前提下提升模型语言建模能力。
:::

## 五、相关工作简述

> MLRA 的提出与以下研究方向密切相关：

| 类别 | 方法 | 特点 |
| --- | --- | --- |
| KV 缓存压缩 | 量化（KVQuant）、Token 丢弃（H2O、Scissorhands）、低秩近似（LoRA、Palu） | 减少存储量，多数不改变注意力计算方式 |
| 效率注意力机制 | MQA、GQA、MLA、GLA | 通过共享或压缩 KV 减少缓存；GLA 分组潜在头，仍需加载整组向量，部分存在并行性限制 |
| 系统级优化 | FlashAttention 系列、vLLM 的 PagedAttention、FlashMLA 内核 | FlashAttention 采用 IO 感知 tiling 降低内存读写；PagedAttention 优化 KV 缓存管理；FlashMLA 是 MLA 的高效实现基础 |
| 线性注意力 | Mamba、RWKV | 时间复杂度降至 O(n)，但可能影响生成质量或难以扩展至超大规模 |

**MLRA 融合了低秩压缩与并行设计**，:mdnice-bold-italic[在保持模型质量的同时，解决了分布式推理的扩展性问题。]

与 MLA 在 TP 下加载量不变的困境相比，MLRA 通过设计实现了线性的加载量下降。与同样采用分组设计的 GLA-2 相比，**MLRA-4 的优势在于其更细粒度的切分效率和更低的并行门槛**。如表 1 所示，GLA-2 需要 4 路 TP 才能将每设备加载量降至 $1.5d_h$，而 **MLRA-4 仅需 4 路 TP 即可原生实现这一优势**，且在 TP=2 时 ($2.5d_h$) 的表现也与 GLA-2 的最佳情况持平，展示了其在各种并行配置下的灵活性和高效性。MLRA 的求和式设计允许更细粒度的切分，而 GLA-2 的分组设计则引入了头与组的绑定，灵活性稍逊。

## 结论

> Multi-Head Low-Rank Attention（MLRA）通过将潜在头分解为多个可并行的分支，巧妙地绕过了 MLA 在张量并行中的冗余加载问题。

MLRA-4 在 2.9B 规模下达到 SOTA 性能，并在长上下文解码中实现 2.8 倍加速。这一设计为未来超长上下文模型的分布式推理提供了新的思路，其代码和预训练权重已开源，值得深入探索。
