关键词：System-Algorithm Co-Design, Domain-Specific Acceleration, Ascend NPU, Multi-Scale Deformable Attention, Grid Sampling, Vision Transformer

> 多尺度可变形注意力（MSDA）是视觉任务中一种灵活且强大的特征提取机制，但其基于随机访问的网格采样策略带来了显著的优化挑战，**尤其是在诸如 NPU** （神经网络处理单元）等领域**专用加速器** 上。

::: block-1
在这项工作中，我们提出了一种**协同设计方法**，系统地**重新思考 Ascend NPU 架构** 上 **MSDA 的内存访问和计算策略**。

通过这种协同设计方法，我们的**实现支持高效的前向和反向计算**，完全适配训练工作负载，并**融入了一套硬件感知优化**。
:::

大量实验表明，我们的解决方案相比基于网格采样的基线，在**正向传播、反向传播和端到端训练** 中分别实现了高达 **5.9 倍、8.9 倍和 7.3 倍的加速**，相**比最新的厂商库** 分别实现了 **1.9 倍、2.4 倍和 2.0 倍的加速。**

![文章插图](/articles/2025-06-18-grid-sample-npu-2-4-f10cb762/images/01.webp)

- https://arxiv.org/abs/2505.14022

![文章插图](/articles/2025-06-18-grid-sample-npu-2-4-f10cb762/images/02.webp)
