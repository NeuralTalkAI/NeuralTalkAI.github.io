关键词：**Blackwell GPU**、注意力机制、:mdnice-bold-italic[非对称硬件缩放]、**算法-内核协同设计**、张量内存、:mdnice-bold-italic[2-CTA MMA 模式]

> 随着 Transformer 架构成为大语言模型的核心支柱，注意力机制的计算效率始终是制约长上下文应用的关键瓶颈。面对 Blackwell GPU 上张量核心吞吐量翻倍而指数单元与共享内存带宽提升缓慢的非对称硬件缩放趋势，FlashAttention-4 应运而生——它通过算法与内核的协同设计，重新定义了注意力计算的性能边界。

- 利用完全异步的 MMA 指令与更大 Tile 尺寸重构前向与反向流水线，最大化张量核心与 Softmax 的重叠；
- 采用基于 FMA 单元的软件模拟指数函数与条件 Softmax 重缩放，缓解指数单元瓶颈；
- 借助张量内存（TMEM）与 2-CTA MMA 模式减少共享内存流量，并将反向传播中的全局原子归约操作减半。

实测表明，在 B200 GPU 上，BF16 精度下 FlashAttention-4 较 cuDNN 9.13 实现最高 1.3 倍加速，较 Triton 实现最高 2.7 倍加速，峰值性能达 1613 TFLOPS（71%理论利用率）。

::: block-1
更令人瞩目的是，全部实现基于 CuTe-DSL 嵌入式 Python 语言，在保持完全表达力的同时将编译速度提升 20-30 倍，为未来硬件迭代下的敏捷开发开辟了新路径。
:::

![文章插图](/articles/2026-03-06-flashattention-4-3e6c2542/images/01.webp)

- **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling**
- https://research.colfax-intl.com/FlashAttention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- **代码：https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute**

相关推荐

- [实现 FA3 峰值性能 78.4%！基于Triton低级扩展的2-Simplicial Attention 硬件对齐方案](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891754&idx=1&sn=ded08de19b2c8c4d4235d9fc2de453a7&scene=21#wechat_redirect)
- [性能直逼FA3/4！斯坦福NVIDIA提出Tensor Core GPU最优流水线和Warp Sepcialization自动调度策略Twill，跨Hopper/Blackwell适配！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895214&idx=1&sn=8ab4120451fcfdff70e3900acab5469c&scene=21#wechat_redirect)
- [性能较 TPUv5e 提升 4.83 倍！单脉动阵列内实现 FlashAttention 全流程且开源！面积开销仅 10%!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890052&idx=1&sn=36b6413d652d086eb7fa34e59ed83999&scene=21#wechat_redirect)

> 本文提出了 FlashAttention-4，一种针对 NVIDIA Blackwell GPU 架构（B200/GB200）进行算法与内核协同设计的注意力机制优化方案。

随着硬件演进呈现非对称缩放趋势——Blackwell 的张量核心吞吐量较 Hopper 翻倍，而共享内存带宽与指数运算单元吞吐量提升缓慢，导致传统注意力内核的瓶颈从矩阵乘法转移至非矩阵运算单元。

针对这一变化，FlashAttention-4 引入四项核心创新：

- 一是在前向传播中重新设计软件流水线，利用 Blackwell 完全异步的 MMA 指令与更大 Tile 尺寸最大化张量核心与 Softmax 计算的重叠；
- 二是通过基于 FMA 单元的软件模拟指数函数并引入条件 Softmax 重缩放，缓解指数单元瓶颈；
- 三是在反向传播中利用张量内存（TMEM）存储中间结果以减少共享内存流量，并借助 2-CTA MMA 模式重构 dQ 梯度计算，将全局原子归约操作减半；
- 四是采用最长处理时间优先调度策略优化因果掩码等负载不均场景。

实现层面，论文采用 CuTe-DSL 嵌入式 Python 语言编写内核，相较于传统 C++模板方法将编译速度提升 20-30 倍。

::: block-1
在 B200 GPU 上的实测结果表明，FlashAttention-4 在 BF16 精度下相比 cuDNN 9.13 实现最高 1.3 倍加速，相比 Triton 实现最高 2.7 倍加速，峰值性能达 1613 TFLOPS（理论峰值 71%）**更多数据见文末**。
:::

![文章插图](/articles/2026-03-06-flashattention-4-3e6c2542/images/02.webp)

::: image-caption
这是B200 GPU上因果注意力关闭（causal=False）、注意力头维度128时的前向计算TFLOPS对比柱状图，对比了FA2、Triton、Gluon、cuDNN和FA4等方案。随序列长度从1K增至32K，各方案算力均提升，FA4与cuDNN 9.19.1.2性能接近且领跑，FA2表现最弱。
:::

![文章插图](/articles/2026-03-06-flashattention-4-3e6c2542/images/03.webp)

::: image-caption
这是B200 GPU上因果注意力开启（causal=True）、注意力头维度128时的前向计算TFLOPS对比柱状图，对比了FA2、Triton 3.6.0、Gluon 3.6.0、cuDNN 9.13.0、cuDNN 9.19.1和FA4六种方案。随序列长度从1K增至32K，各方案算力均提升，FA4与cuDNN 9.19.1表现领先，FA2的算力表现最弱。
:::
