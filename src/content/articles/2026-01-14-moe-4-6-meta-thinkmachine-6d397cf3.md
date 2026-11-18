关键词：:mdnice-bold-italic[MoEBlaze]、内存墙、**MoE 训练**、**索引化路由**

> 在当今大模型浪潮中，参数规模已突破万亿，训练成本与内存压力成为制约模型规模继续扩大的关键瓶颈。**混合专家模型（Mixture-of-Experts, MoE）** 因其能够以稀疏激活的方式实现万亿参数级别的模型训练，已成为大规模语言模型的主流架构之一。

然而，MoE 的稀疏性在降低计算密度的同时，也引入了巨大的**激活内存开销**，:mdnice-bold-italic[尤其是在 token 路由、中间张量物化等环节，形成了显著的“内存墙”问题。]

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/01.webp)

- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs
- https://arxiv.org/pdf/2601.05296v1
- 3000 字，阅读 10 分钟，**播客 20 分钟**

相关推荐

- [MoE 所有层融到一个分布式算子GPU Kernel！FlashDMoE：GPU内核-硬件协同解锁大规模分布式机器学习性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888883&idx=1&sn=14a76e02fc523d30f613d4a2219c0fea&scene=21#wechat_redirect)
- [64块H100媲美96块！开源MoE软硬件协同加速方案SonicMoE：基于IO和tile感知优化实现45%内存节省+1.86倍吞吐量！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894859&idx=1&sn=c50a765749e1cb035469b0fa38334033&scene=21#wechat_redirect)
- [树莓派跑 10B+参数 MoE！EdgeMoE：首款端侧稀疏 LLM 高性能引擎且代码开源！5.4×推理加速与3.2×内存节省！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889845&idx=1&sn=690674e0b64195ed9071603fd24c4ec5&scene=21#wechat_redirect)

> 来自 Meta Platforms Inc 和 Thinking Machines Lab 的研究团队在近期发布的论文《MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs》中，提出了一套**内存高效的 MoE 训练框架 MoEBlaze**，:mdnice-bold-italic[通过系统级的联合设计，显著降低了训练内存占用，提升了计算效率]。

本文将对这项工作进行深入解读，理解 MoEBlaze 的核心创新、实现方法及其对 MoE 训练范式的深远影响。

## 本文目录

- 一、MoE 训练的内存困境：为什么我们需要 MoEBlaze？
  - 1.1 从“内存墙”说起
  - 1.2 MoE 中的内存瓶颈来源
- 二、MoEBlaze 的核心创新：如何突破内存墙？
  - 2.1 内存高效的 Token 路由：从“物化”到“索引化”
  - 2.2 高效数据结构构建：避免全局排序
- 三、训练内核协同设计：应对复杂激活函数的内存挑战
  - 3.1 SwiGLU 带来的内存压力
  - 3.2 融合内核与激活检查点
- 四、实验验证：MoEBlaze 到底有多强？
  - 4.1 实验配置
  - 4.2 内存效率提升
  - 4.3 训练速度提升
  - 4.4 SwiGLU 下的表现
- 五、相关技术背景与先前工作
  - 5.1 MoE 架构演进
  - 5.2 MoE 训练系统
  - 5.3 MoEBlaze 的定位
- 六、总结与展望

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/02.webp)

## 一、MoE 训练的内存困境：为什么我们需要 MoEBlaze？

### 1.1 从“内存墙”说起

> 自上世纪 90 年代起，处理器计算能力的提升速度远快于内存带宽与延迟的改进，形成了所谓的“**内存墙** ”。在大规模深度学习训练中，即使拥有充足的计算单元，端到端的吞吐量也常受限于参数与激活值的读写与交换速度。

MoE 架构通过每 token 仅激活少数专家（如 Top-K）来实现稀疏计算，显著降低了训练 FLOPs，但其**稀疏性也导致计算密度下降**，尤其在分布式训练中，内存压力进一步加剧。随着序列长度和批大小的增长，系统性能逐渐受限于内存与通信子系统，而非单纯的计算能力。

### 1.2 MoE 中的内存瓶颈来源

MoE 训练中的内存瓶颈主要来自两个方面：

1. **Token 路由缓冲区**：传统实现中，需要为每个专家分配缓冲区来紧凑存储被路由的 token，其内存占用与 token 数量 $L$、模型维度 $d$、激活专家数 $K$ 成正比。
2. **中间激活存储**：尤其在采用复杂激活函数（如 SwiGLU）时，前向传播中生成的中间张量（如 GEMM 输出、激活值等）需在反向传播中保留，内存开销巨大。

#### 以 DeepSeek-MoE 为例

- token 数：$L \approx 2 \times 10^6$ tokens
- 激活专家数：$K = 4$
- 模型维度：$d = 6144$
- 使用 bfloat16（2 字节/元素）

::: block-1
**重要说明**：这一计算（94GB）对应的是**无丢弃路由（Dropless Routing）** 策略在 worst-case 下的理论内存占用。在实际传统实现中：
:::

1. :mdnice-bold-italic[容量限制路由（Capacity-Limited Routing）]：如 Switch Transformers 采用的方法，会为每个专家设置固定容量的缓冲区（$C \approx \gamma \cdot \frac{Bk}{E}$），超出容量的 Token 会被丢弃。这种方法内存占用可控但会牺牲模型质量。
2. :mdnice-bold-italic[无丢弃路由（Dropless Routing）]：确保所有 Token 都被处理，不丢弃任何 Token。由于每个专家分配的 Token 数量动态变化，系统通常需要分配足够的缓冲区来容纳所有可能的 Token（接近 $L \times K$ 的规模）。这正是本文 94GB 示例所针对的情况，也是现代高质量 MoE 模型（如 DeepSeek-V3）采用的策略。

> 这个例子搁这儿，就是想强调**无丢弃路由** 在当前大规模训练中的内存瓶颈问题，这也是 MoEBlaze 要解决的核心问题之一。

## 二、MoEBlaze 的核心创新：如何突破内存墙？

MoEBlaze 通过两种系统级优化手段，显著降低了内存占用并提升了计算效率：

1. **内存高效的 token 路由与训练算法**，避免中间缓冲区分配；
2. **训练内核与激活检查点的协同设计**，减少复杂激活函数的内存占用。

### 2.1 内存高效的 Token 路由：从“物化”到“索引化”

> 传统 MoE 流程中，token 路由后会被**物化** 到每个专家的缓冲区中，再执行 FFN 计算，最后聚合回原顺序。这一过程引入了大量中间张量。

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/03.webp)

::: image-caption
图 1 | 传统 MoE 与 MoEBlaze 流程对比图。左图（传统 MoE 流程）：展示传统 MoE 的三步计算流程 ——token 分配（Token Dispatch）、专家计算（Expert Compute）、加权聚合（Weighted Aggregation），需单独存储分配后的 token 缓存，中间数据冗余明显。 右图（MoEBlaze 流程）：呈现 MoEBlaze 的内存高效设计，通过融合 token 路由与专家计算，省去中间缓存存储，直接基于索引实时访问数据，流程更紧凑且资源利用率更高
:::

MoEBlaze 提出一种**索引化路由机制**，核心思想是：

- 不物化路由 token，仅构建轻量级索引结构；
- 在专家计算时动态收集输入 token；
- 在输出聚合时直接归约，避免中间存储。

#### 关键数据结构

> 首先要说明的是，MoEBlaze 设计了四种紧凑的数据结构来支持高效路由：

- `expert_token_indices`：按专家拼接的 token 索引列表，尺寸 $L \times K$
- `expert_token_offsets`：每个专家的 token 起始偏移，长度 $E+1$
- `token_expert_indices`：每个 token 对应的专家 ID 列表，尺寸 $L \times K$
- `token_index_map`：每个 token 在专家列表中的位置索引

下图展示 token-expert 与 expert-token 映射关系：

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/04.webp)

::: image-caption
图 2 | 内存高效 MoE 训练的数据结构示意图，展示 token-expert 与 expert-token 映射，其中包含专家 - token 索引（expert-token-indices）、token - 专家索引（token-expert-indices）、token 索引映射（token-index map）等，清晰展示 6 个 token 与 4 个专家（每个 token 选择 2 个专家）的关联关系。这些轻量化数据结构总占用内存远低于传统方法的中间缓存，其中专家 token 偏移量（expert token offsets）可快速定位每个专家的 token 范围，token 索引映射则支持高效聚合专家输出，共同支撑 MoEBlaze 的并行化计算与低内存占用特性
:::

#### 前向传播流程

1. **Token 分发**：仅生成索引结构，不分配 token 缓冲区；
2. **专家计算**：基于 `expert_token_indices` 动态收集输入，执行 FFN；
3. **输出聚合**：基于 `token_expert_indices` 直接归约到输出张量。

由于这段内容过于抽象，下面是一段伪代码：

```
# 输入: tokens [L, d]# 步骤1: 生成索引expert_token_indices, token_expert_indices = build_indices(gating_scores)# 步骤2: 专家计算（动态收集）for each expert e:    token_ids = expert_token_indices[e]    inputs = gather(tokens, token_ids)  # 从原张量收集    outputs[e] = FFN_e(inputs)# 步骤3: 输出聚合（直接归约）final_output = zeros(L, d)for each token t:    expert_ids = token_expert_indices[t]    for e in expert_ids:        final_output[t] += gating_score[t,e] * outputs[e][position_of_t_in_expert_list]
```

通过这种方式，全程无需将 tokens 复制到专家缓冲区，仅通过索引进行“虚拟路由”。

#### 反向传播流程：

同样利用索引结构避免梯度张量的物化展开，直接通过映射关系在适当位置累加梯度。

### 2.2 高效数据结构构建：避免全局排序

传统方法通常通过对（专家 ID, token ID）元组进行全局排序来构建路由结构，这在 GPU 上会引起多次全局内存访问与高延迟。

MoEBlaze 提出一种**基于稠密位图的并行构建方法**，分为三步：

1. **构建稠密 token-专家映射表**：使用 $L \times E$ 的稠密矩阵记录每个 token 的 Top-K 专家；
2. **计算专家长度**：统计每位专家的 token 数量，生成 `expert_lengths` 与 `expert_offsets`；
3. **生成专家 token 索引列表**：基于偏移将 token 索引写入 `expert_token_indices`。

该方法全程无锁、无原子操作，完全并行，极大降低了构建开销。

#### 为什么全局排序在 GPU 上效率低？

传统方法需对 $(\text{expert}_{\text{id}}, \text{token}_{\text{id}})$ 元组进行全局排序，这在 GPU 上意味着多次全局内存读写，且排序算法（如多通道基数排序）会导致 $O(LK)$ 数据被多次移动，带宽利用率低。

#### MoEBlaze 的 GPU 友好设计：

- 稠密映射表：使用 $L \times E$ 的位图表，每个 token 在其 Top-K 专家对应位置写入自身 token_id。由于每个 token 的专家 ID 唯一，写入操作无锁。
- 专家长度统计：每个线程块负责一个专家列，在共享内存中进行局部计数，再通过线程束内归约得到全局计数。
- 索引写入：根据上一步计算的全局偏移，将 token_id 直接写入 expert_token_indices 对应位置，全程无原子操作。

该方法将复杂度从 $O(LK \log LK)$ 降至 $O(LK)$，且完全并行，更适合 GPU 的大规模并行架构。

## 三、训练内核协同设计：应对复杂激活函数的内存挑战

### 3.1 SwiGLU 带来的内存压力

现代 MoE 常使用 SwiGLU 等复杂激活函数，其定义为：

$$
\text{SwiGLU}(x;W_1,W_2) = \text{SiLU}(xW_1) \cdot (xW_2)
$$

其中 $\text{SiLU}(u) = u \cdot \sigma(u)$。一次 SwiGLU 计算涉及：

- 两次投影：$a = xW_1$, $b = xW_2$
- SiLU 计算与逐元素乘

传统实现中，$a$、$b$、$\sigma(a)$、$\text{SiLU}(a)$ 等中间结果均需保存至全局内存，供反向传播使用，内存占用巨大。

### 3.2 融合内核与激活检查点

> MoEBlaze 提出**融合内核设计** 与**智能激活检查点** 策略：

1. **融合前向内核**：将两次投影、SiLU 计算、逐元素乘法融合为单个内核，避免中间结果写回内存；
2. **激活重计算**：在反向传播中重新计算 SiLU 值，而非保存，因 SiLU 计算轻量且内存带宽受限；
3. **融合反向内核**：将两条路径的梯度计算融合，避免额外缓冲区分配。

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/05.webp)

::: image-caption
算法 1 | 融合 SwiGLU MoE 训练流程。前向传播通过单次融合内核同时计算两个投影（$A=XW_1, B=XW_2$）与 SwiGLU 激活 $Y_{swi}=SiLU(A)⊙B$，仅保存$A、B$和$Y_{swi}$。反向传播时重新计算轻量的$SiLU(A)$，而非从内存读取，既避免了保存大中间张量，又通过融合梯度计算减少内存访问，实现了内存与计算效率的双重优化
:::

#### 内存节省原理与效果

传统实现中，SwiGLU 的前向计算需要物化多个中间结果（如两个投影输出 $a, b$ 以及激活中间值）供反向传播使用。这些中间张量的大小都与 $L \times h$ 成正比，在批量训练中会占用大量激活内存。

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/06.webp)

::: image-caption
图 5 | SwiGLU 激活函数下激活内存占用对比图。纵轴为激活内存占用（单位：MB），横轴为实验配置，聚焦更复杂的 SwiGLU 激活函数场景。SwiGLU 因需存储更多中间计算结果，传统方法内存压力显著增加，而 MoEBlaze 通过内核融合与激活检查点技术，仍实现近 4 倍内存节省，配置 3 中 Megablocks 需占用超 40000MB 内存，MoEBlaze 仅需约 10000MB，验证了其在复杂激活场景下的稳定性
:::

根据图 5 的实验结果：

- 在 CONF3 配置下（$L\approx 2\times 10^6, h=24576$），基线系统 Megablocks 的峰值激活内存占用超过 **40,000 MB**
- MoEBlaze 在相同配置下仅占用约 **10,000 MB**
- **内存降低约 4 倍**

这一显著的节省主要来自：

1. **融合内核设计**：将两次投影、SiLU计算和逐元素乘法融合为单次内核执行，避免中间结果写回全局内存
2. **智能激活重计算**：在反向传播中重新计算轻量的 SiLU 值（而非从内存读取），因 SiLU 是逐元素操作，计算开销远小于内存读写开销

#### 为什么重计算可行？

$\text{SiLU}(x) = x \cdot \sigma(x)$ 是逐元素操作，计算开销极小（约 2 FLOPs/元素），而内存读写需 4 字节/元素（bfloat16）。:mdnice-bold-italic[在现代 GPU 上，这类逐元素操作通常是内存带宽受限的，因此重计算比额外的内存读写更高效。]

## 四、实验验证：MoEBlaze 到底有多强？

实验在 NVIDIA H100 GPU 上进行，对比基线为当前最先进的 MoE 训练系统 **Megablocks**。

### 4.1 实验配置

> 表 1 列出了七种具有代表性的 MoE 配置，覆盖不同输入维度、专家数量、路由数量等：

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/07.webp)

::: image-caption
表 1 | MoE 实验配置表。表格涵盖 7 种贴近真实大语言模型的 MoE 配置，FFN 隐藏层维度固定为输入维度的 4 倍，通过变量输入维度、专家数量等关键参数，全面测试 MoEBlaze 在不同场景下的性能。配置 4 因输入维度大（2048）且专家数量多（16），成为验证模型极限性能的核心场景，后续实验中也展现出最显著的提速与内存节省效果
:::

### 4.2 内存效率提升

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/08.webp)

::: image-caption
图 3 | SiLU 激活函数下激活内存占用对比图。纵轴为激活内存占用（单位：MB），横轴为 7 种实验配置，对比 MoEBlaze 与基准模型 Megablocks 的内存消耗。MoEBlaze 在所有配置下均显著降低内存占用，配置 4 中差距最大（MoEBlaze 约 6100MB vs Megablocks 约 22000MB），而配置 1 因 token 选择专家数少（K=1），内存节省效果相对温和，这与内存节省量随序列长度和激活专家数成正比的规律一致
:::

在 CONF4（大输入维度、多专家）配置下：

- Megablocks：22,000 MB
- MoEBlaze：6,100 MB**内存节省约 3.6 倍**

### 4.3 训练速度提升

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/09.webp)

::: image-caption
图 4 | SiLU 激活函数下训练速度对比图。纵轴为提速倍数（Speedup），横轴为实验配置，数值越大表示相对于 Megablocks 的训练速度越快。MoEBlaze 提速范围在 1.4 倍至 3.7 倍之间，配置 4 凭借优化的内核与高效数据调度，实现最大提速。这一成果得益于其避免了传统方法的多轮内核调度与数据排序开销，充分发挥了 H100 GPU 的硬件加速能力
:::

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/10.webp)

::: image-caption
图 6 | SwiGLU 激活函数下训练速度对比图。纵轴为提速倍数（Speedup），横轴为实验配置，展现 MoEBlaze 在复杂激活函数下的性能优势。该场景下 MoEBlaze 提速更显著（2 倍至 6.2 倍），因 SwiGLU 的复杂计算让传统方法的内存带宽瓶颈更突出，而 MoEBlaze 的内核融合技术将计算从内存受限转为计算受限，同时减少全局内存访问，使提速效果优于 SiLU 激活场景
:::

#### 内存节省如何转化为速度提升？

1. 减少内存传输：传统方法中，中间张量需多次写入和读出 HBM（高带宽内存），占用大量内存带宽。MoEBlaze 通过消除中间缓冲区，将这部分带宽释放给计算单元。
2. 提高缓存命中率：由于数据局部性增强，更多数据可驻留在 GPU 的 L2/L1 缓存中，降低访问延迟。
3. 降低内核启动开销：融合内核将多个小操作合并为一个大内核，减少了 CUDA 内核启动次数，提升了 GPU 利用率。

因此，内存效率的提升直接转化为计算吞吐的提升，尤其在 SwiGLU 等复杂激活函数中效果更显著。

### 4.4 SwiGLU 下的表现

![文章插图](/articles/2026-01-14-moe-4-6-meta-thinkmachine-6d397cf3/images/06.webp)

::: image-caption
图 5 | SwiGLU 激活函数下激活内存占用对比图。纵轴为激活内存占用（单位：MB），横轴为实验配置，聚焦更复杂的 SwiGLU 激活函数场景。SwiGLU 因需存储更多中间计算结果，传统方法内存压力显著增加，而 MoEBlaze 通过内核融合与激活检查点技术，仍实现近 4 倍内存节省，配置 3 中 Megablocks 需占用超 40000MB 内存，MoEBlaze 仅需约 10000MB，验证了其在复杂激活场景下的稳定性
:::

在 CONF3 配置下：

- Megablocks：>40,000 MB
- MoEBlaze：≈10,000 MB**内存降低约 4 倍**

## 五、相关技术背景与先前工作

### 5.1 MoE 架构演进

- **Sparsely-Gated MoE** （Shazeer et al., 2017）：首次提出稀疏门控 MoE 层；
- **Switch Transformer** （Fedus et al., 2022）：采用 Top-1 路由与容量限制；
- **GLaM** （Du et al., 2021）：大规模 MoE 训练，强调效率-质量平衡；
- **DeepSeek-V3** （2025）：671B 参数 MoE，每 token 激活 37B 参数。

### 5.2 MoE 训练系统

> 下面对主要 MoE 训练系统进行的对比整理。

| 系统名称 | 发表年份 | 核心路由/通信机制 | 内存优化关键方法 | 核心技术特点 | 主要优势 |
| --- | --- | --- | --- | --- | --- |
| **FastMoE** | 2021 | 容量限制路由，支持分布式专家放置 | 传统缓冲区管理 | 基于PyTorch的分布式训练框架，提供灵活的专家并行策略。 | 易于使用，是早期实用的分布式MoE训练解决方案。 |
| **Tutel** | 2023 | 运行时自适应并行与流水线 | 动态负载均衡 | 提出“Flex”设计，智能处理路由带来的动态、不规则工作负载。 | 在超大规模集群上实现了显著的性能提升和良好的可扩展性。 |
| **Megablocks** | 2023 | 无丢弃路由，**块稀疏矩阵运算** | 避免填充（Padding）和令牌丢弃 | 将MoE计算重构为块稀疏的GEMM操作，完美映射到GPU的高性能稠密计算单元。 | 首次通过块稀疏计算避免路由开销，获得了接近理论峰值的计算吞吐。 |
| **TurboMoE** | 2025 | 元数据驱动的融合路由 | 数据布局变换，减少稀疏计算开销 | 识别并优化门控路径瓶颈，通过融合内核和智能数据重组来降低开销。 | 大幅降低了稀疏计算和通信的开销，提升了大规模训练的端到端吞吐量。 |
| **MoEBlaze** | 2025 | **索引化路由**，零缓冲区 | **消除中间激活物化**，智能激活检查点与融合内核 | 1. **索引化路由**：用轻量元数据代替物理缓冲区。2. **内核协同设计**：深度融合计算与通信，对复杂激活函数（SwiGLU）进行内存-计算联合优化。 | **革命性内存效率**：彻底突破“内存墙”，在实现最高计算性能的同时，内存占用仅为基线的1/4到1/2，训练速度提升最高达6.2倍。 |

上表展示了 MoE 训练系统的演进脉络，从早期的**分布式框架** （FastMoE），到针对**动态负载** （Tutel）和**计算模式** （Megablocks）的优化，再到专注于**降低稀疏开销** （TurboMoE），最终由 **MoEBlaze** 通过系统级协同设计，实现了对**内存瓶颈** 的根本性解决。

### 5.3 MoEBlaze 的定位

MoEBlaze 在以下方面推进了现有工作：

- **消除每专家激活缓冲区**，通过元数据驱动；
- **深度融合路由与专家计算**，适配现代 GPU 微架构；
- **联合优化内核与激活检查点**，应对复杂激活函数。

## 六、总结与展望

> MoEBlaze 通过**索引化路由** 与**内核协同设计**，系统性地解决了 MoE 训练中的内存瓶颈问题，在保持模型精度的同时，实现了**内存降低 50%以上，训练速度提升最高 6.2 倍**。

这项工作不仅为单 GPU 训练提供了高效解决方案，其核心机制也适用于分布式训练环境。**未来，研究团队计划将 MoEBlaze 扩展至多节点多 GPU 场景**，进一步推动万亿参数模型的高效训练，这个扩展潜力进一步来说：

- 索引化路由可跨节点保持轻量：仅需传输 token_id 和 expert_id，而非完整的 token 张量，极大降低跨节点通信量。
- 融合内核可减少跨设备同步：将计算与通信融合，可在一次内核中完成本地计算与梯度聚合，减少通信次数。
- 适用于专家并行：每个 GPU 可持有一部分专家，MoEBlaze 的索引结构可自然映射到分布式专家布局中，支持动态负载均衡。

MoEBlaze 的出现，标志着 MoE 训练在系统优化上迈出重要一步，也为未来更大规模、更高效的大模型训练指明了方向。
