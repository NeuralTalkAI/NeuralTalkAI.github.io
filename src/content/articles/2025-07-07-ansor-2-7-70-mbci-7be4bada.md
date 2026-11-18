关键词：Operator Fusion, Compute-Intensive, Automatic Exploration, Performance Model, GPU

![文章插图](/articles/2025-07-07-ansor-2-7-70-mbci-7be4bada/images/01.webp)

- MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators
- https://arxiv.org/abs/2506.22169

本文 13136 字，阅读需 52 分钟， 播客 8 分钟如下

> :mdnice-bold-italic[算子融合] 是提高数据局部性和缓解 GPU 内存带宽压力的关键技术，但:mdnice-bold-italic[由于计算吞吐量饱和，该技术往往无法扩展到多个计算密集型算子的融合]。

然而，张量维度大小的动态性可能会导致这些算子受限于内存，因此需要生成融合内核。但融合策略的搜索空间有限、存在冗余内存访问以及调优时间过长等问题阻碍了这一任务的进行，从而导致性能欠佳和部署效率低下。

::: block-1
为了解决这一问题，我们引入 MCFuser，这是一个开创性的框架，旨在通过为我们定义的:mdnice-bold-italic[内存受限计算密集型（MBCI）算子链] 生成高性能**融合内核** 来克服这些障碍。

MCFuser 利用:mdnice-bold-italic[高级分块表达式（high-level tiling expression]）来划定一个全面的搜索空间，并结合**有向无环图（DAG）分析** 来**消除冗余内存访问**，从而简化内核优化。

通过实施**修剪搜索空间** 的准则，并**结合分析性能模型和启发式搜索**，MCFuser 不仅显著加速了调优过程，还展现出卓越的性能。
:::

在 NVIDIA A100 和 RTX3080 GPU 上与 Ansor 等领先编译器进行基准测试时，MCFuser 的内核性能提升高达 5.9 倍，超越了其他基线如 Ansor 等，同时将调优时间缩短了 70 多倍，彰显了其敏捷性。

![文章插图](/articles/2025-07-07-ansor-2-7-70-mbci-7be4bada/images/02.webp)
