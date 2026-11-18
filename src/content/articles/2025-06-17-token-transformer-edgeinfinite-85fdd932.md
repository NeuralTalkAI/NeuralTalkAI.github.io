关键词：Edge Devices、Memory-Efficient、Transformer、Infinite-Context、Inference

> 基于 Transformer 的大型语言模型（LLMs）在边缘设备上处理长序列时面临两大挑战：注意力机制的二次复杂度，以及键值（KV）缓存带来的内存需求增长。

现有的 KV 缓存优化方法在长输出任务中难以避免不可逆的 Token 逐出问题，而采用替代序列建模架构则需要在现有 Transformer 框架内付出较高的适配成本。

::: block-1
为此，我们提出 EdgeInfinite，这是一种**内存高效** 的**无限上下文解决方案**，通过**可训练的内存门控模块** 将**压缩内存** 集成到基于 Transformer 的 LLMs 中。

该方法与标准 Transformer 架构完全兼容，仅需微调少量参数，并能针对长上下文和短上下文任务路由选择性激活内存门控模块。
:::

实验结果表明，EdgeInfinite 在**长上下文基准测试中性能与基于 Transformer 的基线 LLM 相当**，同时**优化了内存消耗和首 Token 生成时间（TTFT）**。![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/01.webp)

- https://arxiv.org/abs/2503.22196

![文章插图](/articles/2025-06-17-token-transformer-edgeinfinite-85fdd932/images/02.webp)
