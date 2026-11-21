关键词：BLASST、**稀疏注意力**、大语言模型、**长上下文推理**、:mdnice-bold-italic[动态剪枝]

> *在大语言模型（LLM）领域，“更长的上下文”始终是核心追求——从处理百万 token 的代码库，到分析超长文档、维持数轮对话，都需要模型具备高效的长序列理解能力。但**传统注意力机制的$O(n^2)$计算复杂度**，却像一道“枷锁”：:mdnice-bold-italic[序列长度翻倍，计算量和内存占用就会翻四倍，即使是最先进的 GPU 也难以承载。]*

- FlashAttention 虽通过分块和核融合优化了带宽，但仍需计算完整注意力矩阵；
- 现有稀疏注意力要么需要昂贵的预计算，要么依赖静态模式缺乏灵活性。

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/01.webp)

- :mdnice-bold-italic[BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding]
- https://arxiv.org/pdf/2512.12087
- **BLASST 的代码已集成到 flashinfer 框架，开源可用**
- 5000 字，阅读 15 分钟，**播客 38 分钟**

相关推荐

- [FlashInfer：面向 LLM 服务的可定制且高效的 GPU 注意力引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888144&idx=1&sn=0b3de7ba3a4d204ad4e017c19e2aed69&scene=21#wechat_redirect)
- [注意力块 4 倍加速！Zen-Attention：面向AMD NPU基于硬件感知的动态注意力层折叠编译框架，端到端32%提速!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891471&idx=1&sn=7b3c57d36ba86b27a277a997b771478f&scene=21#wechat_redirect)
- [端到端加速4.5倍！端侧SoC动态稀疏注意力 shadowAttn：关键token稀疏计算架构且兼容主流框架释放NPU性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891725&idx=1&sn=1c315204fb9f1edc0b26e679d8561983&scene=21#wechat_redirect)

直到这篇来自 Rice 大学、UC Davis 和 NVIDIA 的联合研究——**BLASST（Dynamic Blocked Attention Sparsity via Softmax Thresholding）** ，提出了:mdnice-bold-italic[一种“即插即用”的动态稀疏方案]，在:mdnice-bold-italic[无需预计算、不损失精度] 的前提下，实现了 **1.62 倍 prefill 提速和 1.48 倍 decode 提速。**

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 本文目录

- 一、痛点：长上下文 LLM 的“注意力困境”
- 二、相关工作：稀疏注意力的“前世今生”
  - 1. 计算优化型稀疏：聚焦“减少无效计算”
  - 2. 内存优化型稀疏：聚焦“压缩 KV 缓存”
  - 3. 新型注意力变体：需改架构或重训练
- 三、BLASST 核心创新：三大关键设计
  - 1. 动态剪枝：用“已有统计信息”替代“预计算”
  - 2. 自动校准：让阈值“适配任意上下文长度”
  - 3. 稀疏感知训练：让模型“适应稀疏模式”
- 四、CUDA Kernel 设计：针对 Prefill/Decode 的“差异化优化”
  - 1. Prefill 阶段：优化“计算密集”瓶颈
  - 2. Decode 阶段：优化“内存密集”瓶颈
- 五、实验验证：速度与精度的“双丰收”
  - 1. 精度：稀疏度 75%仍保持高准确率
  - 2. 速度：75%稀疏度下最高 1.62 倍速
- 六、总结与展望

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、痛点：长上下文 LLM 的“注意力困境”

> 要理解 BLASST 的价值，首先得明确长上下文 LLM 面临的核心瓶颈：

对于长度为 n 的序列，传统注意力需要完成三步关键操作：

::: block-1
1. 计算 Query（Q）与 Key（K）的相似度矩阵 S，复杂度 $O(n^2)$；

$$
S = \frac{Q K^\top}{\sqrt{d_k}}
$$

1. 对 S 做 Softmax 得到注意力权重 P，每个元素需指数运算，计算密集；
2. 权重与 Value（V）相乘得到最终输出 $O = PV$，复杂度 $O(n^2)$。
:::

当 n 达到 128K 甚至 1M 时，$O(n^2)$的复杂度会导致：

- **计算爆炸**：GPU 核心长时间满负荷，推理 latency 飙升；
- **内存承压**：KV 缓存和注意力矩阵占用海量 HBM（高带宽内存），甚至超出硬件限制。

现有方案要么“治标不治本”如 FlashAttention 未解决 $O(n^2)$本质，要么“顾此失彼”如静态稀疏缺乏灵活性，动态稀疏需预计算。**BLASST 的核心目标，就是在“无额外开销”和“高精度保留”之间找到平衡点。**

## 二、相关工作：稀疏注意力的“前世今生”

> 在 BLASST 之前，稀疏注意力已发展出三大技术方向，但均存在明显局限。理解这些方案的不足，才能更清晰地看到 BLASST 的创新之处。

### 1. 计算优化型稀疏：聚焦“减少无效计算”

这类方案通过筛选“重要注意力交互”减少计算量，但存在两类问题：

- **静态模式** （如 LongFormer、BigBird）：用固定的块/滑动窗口划分注意力，:mdnice-bold-italic[无法适配不同任务的注意力分布]，比如代码理解和对话的注意力模式完全不同；
- **动态模式** （如 MInference、XAttention）：:mdnice-bold-italic[需要先做一轮“预计算”]，如计算代理重要性分数）来识别关键块，这部分开销常抵消掉稀疏带来的收益，且仅支持 prefill 阶段。

### 2. 内存优化型稀疏：聚焦“压缩 KV 缓存”

这类方案通过裁剪 KV 缓存减少内存占用（如 H2O、RocketKV、StreamingLLM），但**核心目标是“降内存”而非“减计算”** ——即使 KV 缓存压缩了，注意力矩阵的计算仍需 $O(n^2)$操作，无法缓解 compute-bound（计算密集）场景的瓶颈。

### 3. 新型注意力变体：需改架构或重训练

如 Mamba（状态空间模型）、DeepSeek Sparse Attention，虽能降低复杂度，但需要修改模型架构或重新训练，无法作为“即插即用”的优化方案适配现有 LLM。

::: block-1
**BLASST 的突破点**：:mdnice-bold-italic[同时解决“计算+内存”瓶颈，无需预计算/代理分数，支持 prefill+decode 全阶段，且无需修改模型架构] ——完美弥补了现有方案的短板。
:::

## 三、BLASST 核心创新：三大关键设计

> BLASST 的核心思想非常简洁：**在 FlashAttention 的分块计算过程中，用已有的 Softmax 统计信息（块最大值、运行最大值）动态判断“哪些块可以跳过”** ，既无额外开销，又能精准筛选无效计算。

### 1. 动态剪枝：用“已有统计信息”替代“预计算”

FlashAttention 的核心是“分块计算注意力”——将 Q、K、V 拆分为多个小块（Tile），逐块计算并维护“运行最大值（m⁽ʲ⁾）”以保证 Softmax 数值稳定性。:mdnice-bold-italic[BLASST 正是利用了这一“天然存在”的统计信息，设计了一套无开销的剪枝规则。]

#### 核心原理：Softmax 的“数值特性”

Softmax 的计算逻辑决定了：**只有与“全局最大值”接近的元素，才会对最终权重有显著贡献**。对于某一注意力块 Sⱼ（Q 与 K 的第 j 块乘积）：

::: block-1
- 计算块内最大值 $\tilde{m}^{(j)}$（块最大值）；
- 对比块最大值与当前运行最大值 $m^{(j)}$：若 $\tilde{m}^{(j)} - m^{(j)} < \ln(\lambda)$（λ 为阈值），则该块所有元素经过指数运算后会接近 0（$exp(\tilde{m}^{(j)} - m^{(j)}) < \lambda$），对最终输出贡献可忽略。
:::

此时，BLASST 会直接跳过该块的三项昂贵操作：

::: block-1
1. 跳过 Softmax 的指数运算（节省 CUDA 核心）；
2. 跳过从 HBM 加载对应的 V 块（节省内存带宽）；
3. 跳过注意力权重与 V 块的矩阵乘法（节省张量核心）。
:::

下图清晰展示了这一过程：

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/04.webp)

::: image-caption
图1. BLASST核心流程。注意力矩阵一行中的块会被依次处理。我们（1）如FlashAttention中那样更新运行行最大值（m(j)），（2）为每个Sj块（QK⊤j）计算块最大值（m˜(j)），（3）如果块最大值比运行最大值低超过输入阈值ln(λ)，则跳过后续工作。详细内容参见算法1
:::

逐块处理注意力矩阵行时，先更新运行最大值（m⁽ʲ⁾），再计算块最大值（$\bar{m}^{(j)}$），若块最大值与运行最大值差距超过阈值 ln(λ)，则跳过该块后续计算，无需预计算或代理分数。

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/05.webp)

::: image-caption
算法1是结合BLASST的FlashAttention流程：输入Query/Key/Value块与阈值λ，输出注意力结果。遍历每个Query块i，初始化运行最大值、输出、行和；再遍历Key块j，计算注意力分数S_ij，求局部最大值并更新运行最大值。若局部与运行最大值的差小于ln(λ)，则跳过该Key块；否则计算注意力权重，更新行和与输出。最后归一化得到输出块，通过动态跳块实现稀疏注意力的高效计算
:::

#### 关键公式（核心剪枝条件）：

当满足以下条件时，块可被安全跳过：

$$
\tilde{m}^{(j)} - m^{(j)} < \ln(\lambda)
$$

其中：

- $\tilde{m}^{(j)}$：当前注意力块的局部最大值；
- $m^{(j)}$：截至当前块的全局运行最大值；
- λ：可调节阈值（需通过校准确定，下文详解）。

### 2. 自动校准：让阈值“适配任意上下文长度”

> :mdnice-bold-italic[BLASST 的第二个创新，是解决了“阈值选择难题”——固定阈值在不同上下文长度下表现差异极大] （比如 8K 序列的 $\lambda=1e-4$，64K 序列需 $\lambda=1e-5$ 才能维持相同稀疏度）。

研究团队通过大量实验发现：**最优阈值 λ 与上下文长度 L 呈反比关系**，即 $\lambda = a/L$（$a$ 为模型专属常数）。基于这一规律，他们设计了一套自动校准流程（原论文算法 2，见下文）：

1. 选取多个典型上下文长度（如 4K、8K、16K、32K）；
2. 对每个长度，搜索能达到目标稀疏度（如 50%、75%）的最优 $\lambda$；
3. 对（$1/\lambda, \lambda$）做线性回归，拟合出模型专属的 $a$ 值；
4. 部署时，根据输入序列长度 $L$，自动计算 $\lambda = a/L$。

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/06.webp)

::: image-caption
表5. 不同上下文长度下的稀疏度稳定性：Llama-3.1-8B模型上的校准阈值与固定阈值对比。我们的校准方法在不同上下文长度下能保持一致的稀疏度水平，而固定阈值则会产生较大差异。括号中的数值表示实际稀疏度与目标稀疏度的偏差
:::

上表为 BLASST 校准效果对比。括号内为“实际稀疏度与目标稀疏度的偏差”。固定阈值在不同长度下偏差极大，而校准后的阈值能稳定稀疏度，确保部署时性能可预测。

::: block-1
**校准效果验证**：固定阈值在 4K 序列下稀疏度仅 23%，64K 下却高达 75%；而校准后的阈值能将稀疏度稳定在目标值 ±5%以内，完全满足生产部署需求。
:::

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/07.webp)

::: image-caption
算法2是BLASST的校准流程，用于得到适配不同上下文长度的阈值参数$a$：输入目标稀疏度$S$、校准数据集$\mathcal{D}$、上下文长度集$\{L_k\}$、$\lambda$候选集等，遍历各$L_k$，采样对应长度序列，从$\lambda$集中选使实测稀疏度最接近$S$的$\lambda_{\text{best}}$；若差距小于容忍度$\delta$，则将$(1/L_k, \lambda_{\text{best}})$加入数据点集$\mathcal{P}$，最终拟合线性回归$\lambda = a \cdot (1/L)$，返回系数$a$
:::

### 3. 稀疏感知训练：让模型“适应稀疏模式”

> BLASST 的第三个创新，是将“稀疏优化”从“推理阶段”延伸到“训练阶段”——传统稀疏注意力是“推理时硬剪枝”，而 **BLASST 提出“训练时就用稀疏模式”，让模型主动将重要信息集中到高分数块中。**

具体做法非常简洁：

- 微调阶段，在注意力计算中启用 BLASST 的动态剪枝；
- 反向传播时，被跳过的块无梯度更新，模型会逐渐学会“将关键交互集中到不被剪枝的块中”。

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/08.webp)

::: image-caption
图 6：稀疏感知训练对精度的提升效果。蓝色曲线为“推理时加稀疏”，橙色曲线为“训练时加稀疏”。在训练期间启用BLASST进行微调的模型，与训练后应用稀疏性的模型相比，在高稀疏度水平下保持了更高的精度。通过稀疏注意力训练，模型学会将信息集中在高分块中，从而使其对剪枝更具鲁棒性
:::

::: block-1
**效果验证**：经过稀疏感知训练的模型，在 75%高稀疏度下，精度比“推理时才加稀疏”的模型高 1.7%，且在低稀疏度下甚至超过 dense 基线——证明模型学会了更高效的注意力分布。
:::

## 四、CUDA Kernel 设计：针对 Prefill/Decode 的“差异化优化”

LLM 推理的两个核心阶段:mdnice-bold-italic[（Prefill 和 Decode）性能瓶颈完全不同，BLASST 为此设计了两套专用 CUDA 核，确保在不同场景下都能最大化提速。]

### 1. Prefill 阶段：优化“计算密集”瓶颈

> Prefill 是处理输入序列、生成 KV 缓存的阶段，核心瓶颈是**计算** （CUDA 核心的 Softmax 指数运算、张量核心的矩阵乘法）。

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/09.webp)

::: image-caption
图 3：FlashAttention与BLASS在Prefill 阶段流水线对比（50%稀疏度）。深色/浅色代表不同 Tile 行的操作，B1/B2 为矩阵乘法（MMA）操作。BLASST 通过跳过无效块的 Softmax（EX2）和 B2 操作，压缩了总耗时，且决策逻辑被隐藏在现有操作中，无额外 latency
:::

**上图核心呈现 FlashAttention 与 BLASST在 50% 稀疏度下的预填充流水线调度差异**，覆盖 4 次循环迭代（L0-L3）:

- FlashAttention（上图(a)）：需完成所有块的 Softmax 和矩阵乘法，总耗时 18 个时间单位；
- BLASST（上图(b)）：跳过无效块的计算，总耗时压缩至 14 个时间单位，且无额外决策开销。

:mdnice-bold-italic[BLASST 的优化策略：跳过剪枝块的计算操作，同时通过流水线重叠隐藏决策开销]。进一步来说：

- 图内行数按 warp/warpgroup 功能分工划分，如负责矩阵乘法的 MMA warp 与处理归一化的 softmax warpgroup 各司其职；
- 深色、浅色分别对应不同瓦片行（T0/T1）的运算，清晰标记数据块的计算轨迹。

::: block-1
其中：

- MMA warp执行的BMM1（B1）、BMM2（B2）运算，是 Query-Key 相似度矩阵计算的核心步骤；
- softmax warpgroup 虽主要受指数运算（EX2）性能瓶颈制约，还隐性执行 BLASST 的动态修剪关键操作——跳过检查（块是否可修剪）、行求和（累积未归一化权重和）及softmax缩放，仅未标注细节。
:::

主循环迭代（L0-L3）以实线框定，明确流水线时序流程，直观体现 BLASST 在稀疏场景下的调度效率优势。

### 2. Decode 阶段：优化“内存密集”瓶颈

> Decode 是逐 token 生成输出的阶段，核心瓶颈是**内存带宽** （需从 HBM 加载 KV 缓存中的 V 块）。

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/10.webp)

::: image-caption
图4. FlashAttention和BLASST在跳过循环1、2和4时的解码流水线调度对比。未显示序幕部分，我们重点关注前6个循环迭代（L0-L5）的稳定状态。我们拆分了TMA warp的流水线阶段，以展示多个TMA加载是如何同时发出的。图4b中的加载完成得更快，因为同时进行的加载更少。箭头表示BMM1之后来自跳过检查的记分板依赖关系。请注意，MMA warp的BMM1和BMM2操作分别用B1和B2表示
:::

上图对比了两者的 Decode 流水线，TMA 为内存加载操作，B1/B2 为矩阵乘法。

- FlashAttention（上图(a)）：需加载所有 V 块，总耗时 30 个时间单位；
- BLASST（上图(b)）：**跳过无效块的 V 块加载，总耗时降至 23 个时间单位，内存带宽占用减少 73.2%。**

::: block-1
:mdnice-bold-italic[BLASST 的优化策略是：重点跳过剪枝块的 V 块加载（TMA 操作），减少 HBM 访问量]。总耗时从 30 降至 23 个时间单位，**完美适配内存密集场景**。
:::

## 五、实验验证：速度与精度的“双丰收”

BLASST 在 H200/B200 GPU 上的实验结果，彻底证明了其工业价值——:mdnice-bold-italic[在 75%左右的稀疏度下，速度显著提升，精度几乎无损失，甚至在部分任务上超过 dense 基线。]

### 1. 精度：稀疏度 75%仍保持高准确率

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/11.webp)

::: image-caption
表1. BLASST在所有模型和基准测试中不同稀疏度下的性能。我们在Llama-3.1-8B和Qwen3-8B上，在三种部署场景下进行了评估：仅预填充优化（长上下文任务：RULER、LongBench）、仅解码优化（推理任务：MATH500、AIME 2024、GPQA）以及预填充+解码组合优化。结果显示，即使在约75%的稀疏度下，精度下降也极小，偶尔还会优于密集基线
:::

> 原论文表 1 对比了 Llama-3.1-8B 和 Qwen3-8B 在不同稀疏度下的精度：

- 即使在 75%稀疏度下，Llama-3.1-8B 在 RULER（长上下文任务）上的精度:mdnice-bold-italic[仅从 92.33 降至 91.67，LongBench 精度完全不变；]
- Qwen3-8B 在 MATH500（数学推理）上甚至从 :mdnice-bold-italic[95.87 提升至 96.23——原因是剪枝无效块相当于“去噪”，让模型更聚焦关键推理链。]

### 2. 速度：75%稀疏度下最高 1.62 倍速

![文章插图](/articles/2025-12-26-nvidia-blasst-softmax-74-1-c1b9cc17/images/12.webp)

::: image-caption
表4. 在B200（ Blackwell ）GPU上，随着稀疏度水平的提高，BLASST的预填充和解码加速情况。我们改变阈值（λ）以展示在不同稀疏度范围内的性能。预填充配置：批处理大小148，1个Q头，1个KV头，32K序列长度，128头维度。解码配置：批处理大小148，32个Q头，4个KV头，32K序列长度，128头维度。
:::

> 上表展示了 B200 GPU 上的速度提升，对比 FlashAttention 基线可以看出，BLASST 在 B200 GPU 上的速度提升。**速度随稀疏度线性增长，73%左右稀疏度下，Prefill 达 1.41 倍速，Decode 达 1.48 倍速**，且 0%稀疏度时性能接近基线（0.98-0.99x），无额外开销。

::: block-1
- Prefill 阶段：73.47%稀疏度下实现 1.41 倍速，H200 上更是达到 1.62 倍速；
- Decode 阶段：73.15%稀疏度下实现 1.48 倍速，且速度随稀疏度线性提升，完全可预测。
:::

## 六、总结与展望

> BLASST 的最大价值，在于它是一款“工业级友好”的稀疏注意力方案——**无需预计算、无需改架构、无需重训练** （可选稀疏感知微调），却能:mdnice-bold-italic[同时解决计算和内存瓶颈，适配 Prefill/Decode 全阶段。]

对于 LLM 部署而言，这意味着：

- 现有模型可“即插即用”BLASST，无需重新训练；
- 长上下文任务（如代码理解、文档分析）的推理成本降低 40%以上；
- 支持 MHA、GQA、MQA 等多种注意力变体，兼容性极强。

::: block-1
未来，随着硬件（如更大带宽的 GPU）与算法的进一步结合，BLASST 这类“动态稀疏+硬件优化”的方案，有望彻底打破长上下文 LLM 的 $O(n^2)$枷锁，让百万 token 级推理成为常态。
:::

如果你正在做 LLM 推理优化，这篇论文绝对值得深入研读——**BLASST 的代码已集成到 flashinfer 框架，开源可用**，落地门槛极低。

相关推荐

- [FlashInfer：面向 LLM 服务的可定制且高效的 GPU 注意力引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888144&idx=1&sn=0b3de7ba3a4d204ad4e017c19e2aed69&scene=21#wechat_redirect)
- [注意力块 4 倍加速！Zen-Attention：面向AMD NPU基于硬件感知的动态注意力层折叠编译框架，端到端32%提速!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891471&idx=1&sn=7b3c57d36ba86b27a277a997b771478f&scene=21#wechat_redirect)
- [端到端加速4.5倍！端侧SoC动态稀疏注意力 shadowAttn：关键token稀疏计算架构且兼容主流框架释放NPU性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891725&idx=1&sn=1c315204fb9f1edc0b26e679d8561983&scene=21#wechat_redirect)
