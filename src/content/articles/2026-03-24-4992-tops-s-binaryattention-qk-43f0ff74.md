关键词：Transformer、**二值注意力**、:mdnice-bold-italic[硬件加速]、**极低比特量化**

> *当注意力机制“瘦身”到**极致**。Transformer 架构的成功，很大程度上归功于其强大的注意力机制，它能捕捉序列中任意两个位置之间的依赖关系。然而，这种能力是有代价的：**注意力计算的时间复杂度和空间复杂度随序列长度 $N$ 呈二次方增长**。在视觉任务中，当处理高分辨率图像（如`1024x1024`）时，:mdnice-bold-italic[这意味着巨大的计算开销和内存占用，成为了制约模型部署的关键瓶颈。]*

为了突破这一瓶颈，业界已经尝试了各种方法：从架构层面的稀疏注意力、线性注意力，到硬件层面的 FlashAttention，再到数值精度层面的模型量化。

其中，模型量化通过将高精度浮点数（如 FP32）压缩为低精度整数（如 INT8）来加速计算和节省内存，是实践中非常有效的一环。然而，**量化注意力模块，尤其是其中的 $QK^T$ 矩阵乘法，面临着独特的挑战——它负责动态地计算词元间的相似度，对精度损失极其敏感。**

::: block-1
那么，我们能否将注意力机制的量化推向极致？能否用 1 比特，即每个数值:mdnice-bold-italic[仅用 $+1$ 或 $-1$ 来表示，来替代原本复杂的浮点数计算]，从而在根本上改变规则？这听起来似乎有些疯狂，因为**将信息压缩到 1 比特，意味着巨大的信息损失，难道不会导致模型性能断崖式下跌吗？**
:::

![文章插图](/articles/2026-03-24-4992-tops-s-binaryattention-qk-43f0ff74/images/01.webp)

- **BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers**
- https://arxiv.org/pdf/2603.09582
- **代码：https://github.com/EdwardChasel/BinaryAttention**
- 7000 字，阅读 24 分钟，**播客 22 分钟**

相关推荐

- [端到端加速4.5倍！端侧SoC动态稀疏注意力 shadowAttn：关键token稀疏计算架构且兼容主流框架释放NPU性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891725&idx=1&sn=1c315204fb9f1edc0b26e679d8561983&scene=21#wechat_redirect)
- [1.25bit登顶！腾讯开源三元量化方案Sherry：用3:4稀疏榨干硬件最后一滴血，25%位宽节省+10%速度暴增屠榜LLM边缘推理](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897385&idx=1&sn=81cd50636e85207bb71b6fa881cdda53&scene=21#wechat_redirect)
- [走向ARM CPU 1-bit推理的极致道路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885109&idx=1&sn=0f32a578c71b66d6ceffb7952859c575&scene=21#wechat_redirect)

> 来自香港理工大学和 OPPO 研究院的团队，在最新论文《BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers》中给出了一个令人振奋的答案：**不仅可以，而且在保持甚至超越全精度性能的同时，还能实现 2 倍于 FlashAttention2 的加速！**

核心洞察：论文指出，:mdnice-bold-italic[注意力机制的核心在于捕捉查询（Query）与键（Key）之间的相似性关系，而非精确的数值大小。]

![文章插图](/articles/2026-03-24-4992-tops-s-binaryattention-qk-43f0ff74/images/02.webp)

::: image-caption
图 1 上半部分：视觉任务中 FlashAttention2 与 BinaryAttention 的性能对比。下半部分：由 BinaryAttention 驱动的 DiT-XL/2 模型生成的图像示例。该图从性能与生成效果双重维度验证 BinaryAttention 的有效性。上半部分对比视觉任务性能，证实 BinaryAttention 在速度与精度上均优于 FlashAttention2；下半部分 DiT-XL/2 生成图像细节完整、结构合理，说明其可适配扩散 Transformer。结论：BinaryAttention 能兼顾计算效率与模型效果，适配视觉 Transformer 与扩散 Transformer 两大场景。
:::

::: block-1
通过理论证明，他们将原始的浮点点积相似性，巧妙地转化为二进制空间中的汉明距离或角度余弦。:mdnice-bold-italic[更重要的是，作者提出的定理 1 揭示了二进制化后的查询和键的外积期望， 与原始高斯分布下的协方差矩阵之间，存在一个简洁的 $arcsin$ 关系，这意味着二进制表示保留了原始数据的统计结构]。这为 1 比特注意力提供了坚实的理论基础，着急的读者可翻到本文 2.1 节。
:::

![文章插图](/articles/2026-03-24-4992-tops-s-binaryattention-qk-43f0ff74/images/03.webp)

::: image-caption
图 2 BinaryAttention 的整体架构与对比分析。(a) BinaryAttention 的计算包含三个核心模块：将查询与键转换为缩放二值表示、添加偏置增强、对注意力系数与值进行量化。子图 (b)(c) 分别展示标准注意力与本文 BinaryAttention 的注意力图（上）及对应激活图（下），证明即便经过 1 比特量化，BinaryAttention 仍具备与标准注意力相当的表达能力。BinaryAttention 的核心设计通过缩放二值化、偏置增强、混合量化三模块协同实现 1 比特 QK 注意力高效计算。对比注意力与激活图，二者特征聚焦区域高度一致，说明 1 比特量化未破坏关键特征捕捉能力。该设计用位运算替代浮点点积，既大幅降计算成本，又完整保留注意力表达能力，为极端低比特注意力设计提供可行路径。
:::

基于这一洞察，BinaryAttention 应运而生。它不仅**将 $QK$ 相似度计算完全二值化**，还通过**引入可学习偏置来弥补信息损失**，并对后续的**注意力权重和值（Value）进行 8-bit 量化**，实现了**端到端的硬件友好加速。**

实验数据有力地支撑了他们的论点：

- 速度：在 A100 GPU 上，BinaryAttention 的**注意力核心速度比 FlashAttention2 快 2 倍以上**，在`1024x1024`的高分辨率下，**端到端推理吞吐量提升高达 1.5 倍。**
- 精度：在 ImageNet 分类任务上，BinaryAttention-Base 模型达到了 **82.04% 的 Top-1 准确率，超越了全精度 DeiT-Base (81.8%) 和 SageAttention (81.83%)。** 在 COCO 目标检测和实例分割任务上，BinaryAttention 同样展现了媲美甚至超越全精度基线的性能。

> 那么，BinaryAttention 是如何在 1 比特的极限压缩下，做到“既快又好”的？接下来，我们将深入其技术细节，一探究竟。

![文章插图](/articles/2026-03-24-4992-tops-s-binaryattention-qk-43f0ff74/images/04.webp)

::: image-caption
图 6：基于 DiT-XL/2（cfg=1.50），用 FlashAttention2、SageAttention 与 BinaryAttention 生成图像的更多定性对比。此图补充三种方法的生成效果，FlashAttention2 与 SageAttention 结果高度相近，BinaryAttention 生成内容略有差异但细节完整、质量达标。进一步验证 BinaryAttention 在扩散 Transformer 中稳定有效，1 比特量化未牺牲生成性能，为轻量化高效图像生成提供可靠方案。
:::
