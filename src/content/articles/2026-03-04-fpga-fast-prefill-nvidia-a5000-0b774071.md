> FAST-Prefill 这项工作非简单映射算法，本质是硬件-算法深度协同的范例，其基于 Flex-Prefill 的计算模式与数据流特征，定制数据通路与访存策略。其不仅提供实用加速方案，更确立了一种方法论启示——针对特定工作负载，精巧的架构设计往往比通用算力堆砌更高效。

关键词：:mdnice-bold-italic[FPGA 加速]、**稀疏注意力**、**长上下文 LLM**、预填充阶段、**动态稀疏模式**

> 当大模型需要处理 128K Token的长文本时，预填充阶段的计算量爆炸式增长。FAST-Prefill 提出了一种基于 FPGA 的动态稀疏注意力加速方案，通过创新的硬件架构设计，**在保持模型精度的同时实现了 2.5 倍的 TTFT 速度提升和 4.5 倍的能效提升。**

让大语言模型（LLM）阅读一本数百页的小说并回答其中的问题，在这个过程中，模型首先需要处理整本书的内容——这就是 LLM 推理中的"预填充阶段"（prefill stage）。**当输入文本变长时，预填充阶段的自注意力计算量呈平方级增长，成为整个推理过程的性能瓶颈。**

::: block-1
近年来，**稀疏注意力** 技术被提出来解决这个问题。它的**核心思想很直观：每个Token不需要与所有其他Token交互，只需关注其中的一小部分**。:mdnice-bold-italic[但问题在于，这种稀疏模式是"动态"的] ——不同输入文本有不同的稀疏模式，同一个文本中不同的注意力头也有各自独特的模式。
:::

这种动态特性让 GPU 陷入了尴尬境地：

- 一方面，生成稀疏索引需要复杂的控制流和数据依赖；
- 另一方面，稀疏的 KV 缓存访问模式导致数据重用率低、内存带宽利用率不足。

GPU 虽然计算能力强，但在这种**内存密集型** 任务中难以发挥优势。

![文章插图](/articles/2026-03-04-fpga-fast-prefill-nvidia-a5000-0b774071/images/01.webp)

- **FAST-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill**
- https://arxiv.org/pdf/2602.20515
- 7000 字，阅读 25 分钟，**播客 22 分钟**

相关推荐

- **[低功耗高性能！TeLLMe：首个 FPGA 三值 LLM 加速器！三值矩阵乘法查找表和预填充注意力协同优化计算、内存和调度！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888617&idx=1&sn=4ec623ed4d887d2cb7eebf5003057441&scene=21#wechat_redirect)**
- **[EdgeLLM：数据并行、端到端编译器与混合精度的 CPU-FPGA 加速架构](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888046&idx=1&sn=59307cdc16ca341e201680d6a02f8db0&scene=21#wechat_redirect)**
- **[卷积提速 55 倍！为 TinyML 打造的 FPGA 全栈开源框架 CFU Playground，敏捷探索轻量加速器设计空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888011&idx=1&sn=3ea5446297b91cdd15378baa2ed83125&scene=21#wechat_redirect)**

> 这正是 FPGA（现场可编程门阵列）的机会所在。FPGA 具有可定制的存储层次和计算流水线，能够为特定工作负载设计专用的数据通路。来自南加州大学的研究团队在 FPGA'24 会议上提出的 FAST-Prefill，正是**首个针对动态稀疏注意力长上下文预填充阶段的 FPGA 加速器。**

本文将深入解读 FAST-Prefill 的工作机制，探讨它如何通过创新的硬件架构设计，:mdnice-bold-italic[在保持模型精度的同时，实现比 NVIDIA A5000 GPU 快 2.5 倍的性能提升和 4.5 倍的能效提升。]

![文章插图](/articles/2026-03-04-fpga-fast-prefill-nvidia-a5000-0b774071/images/02.webp)

::: image-caption
图 5 | FAST-Prefill 与基准 GPU 实现的首令牌生成时间（TTFT）对比。该图对比了 Nvidia A5000 GPU 与 FAST-Prefill 在 Llama3.2-1B、Qwen-2-1.5B、Llama3.2-3B 模型上，4K 至 128K 上下文长度的归一化 TTFT。结果显示 FAST-Prefill 在所有模型和上下文长度下均表现更优，且随上下文长度增加优势更显著，核心原因是其定制化的稀疏索引核与双级缓存大幅减少了片外访存，而 GPU 存在不规则访存与算力利用率低的问题。
:::
