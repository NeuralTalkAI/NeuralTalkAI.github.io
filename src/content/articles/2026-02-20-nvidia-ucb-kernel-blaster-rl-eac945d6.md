关键词：CUDA 优化、**记忆增强**、:mdnice-bold-italic[上下文强化学习]、**跨架构适配**、性能加速

> 当大语言模型学会反思自己的优化策略，GPU 代码的性能天花板将被不断推高。

过去十年，GPU 架构从 Pascal、Volta 一路进化到 Ampere、Hopper，每一代新硬件都带来了更复杂的并行计算能力。但随之而来的问题是：**为上一代 GPU 精心调优的 CUDA 内核，在新硬件上可能立刻失去优势**。

![文章插图](/articles/2026-02-20-nvidia-ucb-kernel-blaster-rl-eac945d6/images/01.webp)

- **Kernel Blaster: Continual Cross-task Cuda Optimization Via Memory-augmented In-context Reinforcement Learning**
- https://arxiv.org/pdf/2602.14293
- 1 万字，阅读 40 分钟，**播客 22 分钟**

相关推荐

- [AutoTriton：通过强化学习实现自动Triton编程！8B参数媲美Claude-4 与 DeepSeek-R1！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890530&idx=1&sn=2f4b9743242cd200c0bae8e8f8c68e67&scene=21#wechat_redirect)
- [Meta 提出 TritorX：面向 ML ASIC 的 Agentic 算子生成系统，84.7% OpInfo 覆盖与 2 万+测试验证，简化加速器软件生态构建](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895656&idx=1&sn=8a348cee4243a478fe9218a0f317c9ab&scene=21#wechat_redirect)
- [ProfilingGuided+LLM协同：TritonForge突破 Triton内核优化瓶颈，成功率42.7%且最高提速5倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894462&idx=1&sn=ba65fbb1cade2e2982477a013f23beda&scene=21#wechat_redirect)

> 作者开篇举了两个典型案例：

- **FlashAttention-2** 刚移植到 NVIDIA H100 时，性能直接掉了 47%，直到重新设计为 FlashAttention-3 才挽回颓势。
- **DeepSeek-V3** 针对 H800 的 FP8 计算和集群通信做了深度优化，但如果换到其他 GPU 型号，这些优化可能完全不适用。

这种“硬件演进速度 > 软件优化速度”的矛盾，正在成为大模型时代的性能瓶颈。

- 手工调优需要深厚的硬件知识，且难以跨任务复用；
- 传统编译器依赖固定启发式，缺乏灵活性；
- 用大语言模型（LLM）生成 CUDA 代码虽然已有探索，但现有方法要么成本高（微调模型），要么无法积累经验（每次从零开始搜索）。

> 那么，有没有一种方法，能让 LLM 在不断试错中**学习通用的优化策略**，并将经验持久化，在新任务上快速找到高效内核？

这正是 Kernel Blaster 要解决的问题。它提出了一种名为**记忆增强上下文强化学习** （:mdnice-bold-italic[Memory-Augmented In-Context Reinforcement Learning, MAIC-RL]）的框架，让基于 LLM 的 CUDA 优化智能体具备了**“长期记忆”和“经验迁移”** 的能力。

![文章插图](/articles/2026-02-20-nvidia-ucb-kernel-blaster-rl-eac945d6/images/02.webp)

::: image-caption
表 2 | KernelBench Level 1 与 Level 2 上的端到端性能指标对比。指标包含验证通过率、几何平均加速比、标准差、最小值、最大值、平均令牌消耗、高加速占比及低加速占比，全面量化了优化效果与成本。数据显示 Kernel Blaster 以 93%–95% 的高验证率实现 4.5–9.4 倍的几何平均加速，令牌效率显著优于基线。高加速占比超 59%，说明其不仅能稳定优化，还能挖掘出极致性能潜力，在复杂 Level 2 任务中优势尤为突出。
:::
