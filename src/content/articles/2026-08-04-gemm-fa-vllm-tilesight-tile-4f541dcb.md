关键词：GPU 性能建模、**Tile 抽象**、:mdnice-bold-italic[流水线重叠]、**缓存重用距离**、分布式推理

> *当 Triton、TileLang、甚至 NVIDIA 刚刚在 CUDA 13.1 中推出的 CUDA Tile **都把 tile 作为一等编程公民时，性能分析工具还停留在 roofline 的粗粒度边界、ML 模型的黑盒预测、或者 profiler 的事后采样**。黄仁勋称 :mdnice-bold-italic[CUDA Tile 是 CUDA 诞生 20 年来最重要的进步]，但写 kernel 的人依然要靠 autotuning 在上千种配置里暴力搜索，靠 NSight Compute 的计数器猜瓶颈在哪。*

这个 gap 在 AI workload 下变得愈发尖锐。现代 LLM kernel 早就不是简单的 GEMM 了，FlashAttention-3 在 H100 上涉及十几种异构操作：Tensor Core 上的两个 GEMM、CUDA Core 上的 reduction 和 softmax、SFU 上的特殊函数，**它们之间有精细的数据依赖，占用不同的硬件资源，理论上可以重叠，但重叠程度完全取决于调度顺序和流水线深度**。

::: block-1
再往上走，tensor parallelism、expert parallelism、sequence parallelism 把计算切到多张 GPU 上，collective communication 要和计算重叠，:mdnice-highlight[全局 tile 网格怎么切、用什么通信原语、计算和通信流水线怎么交错，这些决策现在全靠直觉和昂贵的试错。]
:::

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/01.webp)

- **TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters**
- https://arxiv.org/pdf/2607.22432

::: related-title
**相关推荐**
:::

::: related-links
- **[LLM推理极限建模：Bandwidth, Compute, Sync and Capacity are All You Need](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890183&idx=1&sn=2fc6e4e2e0c705123df18ad6d9f0dcb4&scene=21#wechat_redirect)**
- **[超越仿真，精准预测推理性能！上交与阿里提出SynPerf：一种混合分析与机器学习的 GPU 性能预测框架，实现 6.1%内核误差与 1.7 倍加速！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897849&idx=1&sn=0e780adf7b65d20826ef5e3b2330cda2&scene=21#wechat_redirect)**
- **[面向下一代NPU架构的仿真模拟器EONSim：双模式协同建模支撑DNN workloads，TPUv6e验证误差低至1.4%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896021&idx=1&sn=cf4ec890f976d47302297253dd1897be&scene=21#wechat_redirect)**

:::

> TileSight 的核心洞察极其朴素：**既然 tile 已经是编程的一等公民，那它也应该成为性能分析的一等公民**。这个抽象天然具备三个特性：

| 特性 | 描述 |
| --- | --- |
| 确定性 | 给定 tile 形状、流水线深度、内存布局，每个 tile 的资源使用完全确定，不需要模拟 |
| 可组合性 | 每个 tile 自带流水线资源分解，tile 之间通过依赖、并发、执行顺序关联，tile 网格跨设备延伸 |
| 可移植性 | 从 A100 到 B200 再到 AMD MI210，所有现代 GPU 都通过分层内存和计算结构执行 tile 形 workload |

基于这个洞察，**TileSight 构建了一个统一的三层分析引擎**。在单核内，它建模计算和内存流水线的重叠；在同芯片的多个核之间，它建模缓存层次；在跨 GPU 之间，它建模节点间通信。

::: block-1
:mdnice-highlight[三层共享同一个 tile 抽象]：

- **intra-tile 层** 把每个 tile 的工作表达为横跨网络、内存、计算流水线的资源向量；
- **inter-tile 层** 调度有依赖和有序的 tile 动作来暴露合法重叠，并从 tile 重用距离推导多级缓存命中率；
- **cross-device 层** 把远程 tensor 访问映射到 placement，通过 α-β 阶段成本路由。
:::

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/02.webp)

::: image-caption
图 3：以 AllGather-GEMM (AG-GEMM) 为例的 TileSight 设计总览。(a) 仅通过算子和张量排布描述工作负载（X 矩阵按列切分到 N 张 GPU）；(b) TileSight 将其提升为 tile 调度方案，对应的 DAG 跨越 L₀到 L₄五级访存层级；(c) 统一硬件抽象将寄存器、SMEM、L2、HBM 和 GPU 间互联抽象为五级层次结构；(d) tile 内资源向量与 tile 间 DAG / 并发分析输入到递归的 “序言 - 稳态 - 尾声” 三段式流水线模型中；(e) 引擎将三段式流水线模型渲染为时间线：软件流水线化的加载与计算重叠，AllGather 通信通过 Net 通道从张量排布中推导得出；(f) 包含时延、利用率、缓存命中率、重叠率的逐 tile 性能报告。
:::

实验结果是硬的（👀，solid）。

- 在 A100、H200、B200、B6000 四款 GPU 上，:mdnice-bold-italic[TileSight 预测单 GPU kernel 延迟的聚合 MAPE 是 12.35%，超过了所有 SOTA 基线，而且在新架构上迁移性更好。它的 L2 缓存命中率预测在每款 GPU 上都和实测值相差约 1 个百分点。]
- 扩展到 **32GPU 部署，TileSight 在融合分布式 kernel 上达到 16.18% wMAPE，在端到端 vLLM 服务上达到 13.52% wMAPE**。把它放进优化循环里，TileSight 选出的 tile 配置能和厂商库、专家手写的基线打平甚至更好。

## 本文目录

- 一、问题背景：为什么 GPU 性能建模这么难
- 二、TileSight 整体架构：三层 tile 流水线模型
- 三、Intra-tile：资源向量与三段式流水线模型
- 四、Inter-tile：依赖、缓存与调度顺序
- 五、Cross-device：从 placement 到网络路由
- 六、实验评估：从单算子到端到端 vLLM
- 七、局限、启示与未来方向
- 结语

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[比真 GPU 硬件快5-17 倍的时间跳跃仿真框架 REVATI！实现 vLLM/SGLang 跨框架无 GPU 加速仿真，预测误差低于 5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896002&idx=1&sn=7b81c53ef9a19c14aab1b6bb1d0a5361&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**

:::
