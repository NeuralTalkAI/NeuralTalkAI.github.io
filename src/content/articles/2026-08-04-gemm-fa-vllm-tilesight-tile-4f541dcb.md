**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：GPU 性能建模、**Tile 抽象**、:mdnice-bold-italic[流水线重叠]、**缓存重用距离**、分布式推理

> 当 Triton、TileLang、甚至 NVIDIA 刚刚在 CUDA 13.1 中推出的 CUDA Tile **都把 tile 作为一等编程公民时，性能分析工具还停留在 roofline 的粗粒度边界、ML 模型的黑盒预测、或者 profiler 的事后采样**。黄仁勋称 :mdnice-bold-italic[CUDA Tile 是 CUDA 诞生 20 年来最重要的进步]，但写 kernel 的人依然要靠 autotuning 在上千种配置里暴力搜索，靠 NSight Compute 的计数器猜瓶颈在哪。

这个 gap 在 AI workload 下变得愈发尖锐。现代 LLM kernel 早就不是简单的 GEMM 了，FlashAttention-3 在 H100 上涉及十几种异构操作：Tensor Core 上的两个 GEMM、CUDA Core 上的 reduction 和 softmax、SFU 上的特殊函数，**它们之间有精细的数据依赖，占用不同的硬件资源，理论上可以重叠，但重叠程度完全取决于调度顺序和流水线深度**。

::: block-1
再往上走，tensor parallelism、expert parallelism、sequence parallelism 把计算切到多张 GPU 上，collective communication 要和计算重叠，:mdnice-highlight[全局 tile 网格怎么切、用什么通信原语、计算和通信流水线怎么交错，这些决策现在全靠直觉和昂贵的试错。]
:::

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/01.jpg)

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

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/02.jpg)

::: image-caption
图 3：以 AllGather-GEMM (AG-GEMM) 为例的 TileSight 设计总览。(a) 仅通过算子和张量排布描述工作负载（X 矩阵按列切分到 N 张 GPU）；(b) TileSight 将其提升为 tile 调度方案，对应的 DAG 跨越 L₀到 L₄五级访存层级；(c) 统一硬件抽象将寄存器、SMEM、L2、HBM 和 GPU 间互联抽象为五级层次结构；(d) tile 内资源向量与 tile 间 DAG / 并发分析输入到递归的 “序言 - 稳态 - 尾声” 三段式流水线模型中；(e) 引擎将三段式流水线模型渲染为时间线：软件流水线化的加载与计算重叠，AllGather 通信通过 Net 通道从张量排布中推导得出；(f) 包含时延、利用率、缓存命中率、重叠率的逐 tile 性能报告。
:::

实验结果是硬的（👀，solid）。

- 在 A100、H200、B200、B6000 四款 GPU 上，:mdnice-bold-italic[TileSight 预测单 GPU kernel 延迟的聚合 MAPE 是 12.35%，超过了所有 SOTA 基线，而且在新架构上迁移性更好。它的 L2 缓存命中率预测在每款 GPU 上都和实测值相差约 1 个百分点。]
- 扩展到 **32GPU 部署，TileSight 在融合分布式 kernel 上达到 16.18% wMAPE，在端到端 vLLM 服务上达到 13.52% wMAPE**。把它放进优化循环里，TileSight 选出的 tile 配置能和厂商库、专家手写的基线打平甚至更好。

## 本文目录

- 一、问题背景：为什么 GPU 性能建模这么难
  - 1.1 现有工具的三层缺失
  - 1.2 Tile 抽象为什么是答案
- 二、TileSight 整体架构：三层 tile 流水线模型
  - 2.1 核心数据结构：tile 执行计划
  - 2.2 算法主循环
  - 2.3 可移植硬件抽象
- 三、Intra-tile：资源向量与三段式流水线模型
  - 3.1 资源向量：九维时间分解
  - 3.2 三段式流水线模型：prologue、steady、epilogue
  - 3.3 稳态重叠：拓扑序搜索
  - 3.4 边界开销与 wave 分解
- 四、Inter-tile：依赖、缓存与调度顺序
  - 4.1 为什么传统重用距离分析太慢
  - 4.2 Tensor 访问与 tile 重用距离
  - 4.3 命中概率与快速评估
  - 4.4 归约轴采样与两级级联
- 五、Cross-device：从 placement 到网络路由
  - 5.1 远程 tensor 访问推导
  - 5.2 逻辑交换与拓扑
  - 5.3 Per-stage 路由成本
- 六、实验评估：从单算子到端到端 vLLM
  - 6.1 单算子预测准确率
  - 6.2 端到端 vLLM 推理
  - 6.3 应用：性能诊断与 cost model
- 七、局限、启示与未来方向
  - 7.1 当前局限
  - 7.2 对领域的启示
  - 7.3 未来方向
- 结语

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/03.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、问题背景：为什么 GPU 性能建模这么难

> 写高性能 GPU kernel 的人都有过这种体验：roofline 模型告诉你这个 kernel 是 memory bound 的，但你改了 tile 大小、加了软件流水线、调了 swizzle 顺序，性能差了 2 倍，FLOP 和 byte 数却完全没变。

这就是现有性能工具的根本困境——**它们抽象的粒度和现代 GPU 程序实际运行的粒度对不上**。

- 你跟它讲 FLOP 和带宽，它跟你讲实际 kernel 里十几种操作的依赖关系、不同流水线的重叠、L2 里到底命中了多少；
- 你跟它讲 autotuning，它跟你讲上千种配置跑一遍要几个小时，换个架构又得重来；
- 你跟它讲 profiler，它跟你讲 NSight 跑一遍 kernel 慢三倍，计数器告诉你 TC 利用率 57%但不告诉你为什么是 57%、怎么改成 80%。

### 1.1 现有工具的三层缺失

论文把现有工具的问题精准归纳为三个层次的缺失。

> **第一层是 intra-tile 缺失**。每个 tile 使用横跨计算、内存、网络的异构流水线，单一瓶颈标量根本捕捉不到决定重叠度的 per-pipeline 结构。

:mdnice-bold-italic[就像下图 1 展示的 FlashAttention-3，同一个 tile 迭代里，Tensor Core 在算 GEMM0 的时候，LD Q/K/V 这些内存操作可以同时进行，但它们能不能重叠、重叠多少，取决于操作的顺序和资源竞争]。roofline 只看总的 FLOP 和 byte，它告诉你这个 kernel 算得够快、内存也够快，但实际跑起来就是有流水线气泡，它看不见。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/04.jpg)

::: image-caption
图 1：FlashAttention-3在H100上的执行结构。(a) 十几种异构操作横跨Tensor Core、CUDA Core和SFU；(b) 它们的数据依赖DAG；(c) 调度顺序如何决定计算-内存流水线重叠。
:::

这张图非常直观地展示了问题所在。最下边(c)部分你可以看到，**Pipe 0 和 Pipe 1 两条流水线，不同操作在不同硬件单元上，理论上可以重叠，但实际重叠多少完全取决于你怎么排这个顺序**。GEMM0、GEMM1、softmax、各种 LD 操作，:mdnice-bold-italic[它们之间有依赖，但依赖之间有空隙，这个空隙就是优化空间。现有工具不建模这个调度结构]，所以它们预测不了为什么同样的 FLOP 数、同样的 byte 数，换个 tile 顺序性能差很多。

> **第二层是 inter-tile 缺失**。tile 之间的依赖决定了融合体内的合法动作顺序，而 tile 在网格上的执行顺序决定了缓存复用。现代 GPU 的缓存层次早就不是一个平的带宽数字了。:mdnice-bold-italic[图 2 展示了 B200 和 B6000 的有效带宽随 working set 大小的变化，你可以清晰地看到悬崖式下降。]

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/05.jpg)

::: image-caption
图 2：B200和B6000上L2带宽随working set大小的变化，揭示了多级缓存层次。B200（双die）在约22.5 TB/s处暴露了L1.5/LRC层，在约83 MB处有L2悬崖；B6000（单die）在约130 MB处显示陡峭悬崖。TileSight用这些扫描来校准每款GPU的有效缓存容量。
:::

- B200 是双 die 设计，所以你看到两个台阶：第一个是 L1.5/LRC 层，带宽高达 22465 GB/s，第二个才是 L2 悬崖在 83.12 MB。
- B6000 是单 die，悬崖更陡，在 130.62 MB 处直接从 7434 GB/s 掉到 1563 GB/s。

一个简单的平带宽数字怎么可能描述这种行为？**更关键的是，L2 命中率不是 tile 本身的属性** ——同样的 load tile 坐标，swizzle 方式不同、wave 占用不同、相邻 tile 共享数据多少不同，它可能命中缓存也可能打到 DDR。:mdnice-bold-italic[行优先遍历能保留 B tile 在相邻 M 行的复用，block swizzle 能把 L2 命中率从 35%变到 72%，这些都是调度决定的。]

> **第三层是 cross-device 缺失**。分布式场景下，切分后的 tile 网格通过通信流水线交换数据，这些通信要和计算重叠，而不是简单加一个独立的通信时间。

AllGather+GEMM 这种融合 kernel，通信和计算的重叠方式对性能影响巨大，但现有模型要么把通信当成黑盒，要么需要用户手动指定重叠率，不能从 tile placement 自动推导。

### 1.2 Tile 抽象为什么是答案

> 论文在这里提出了一个非常关键的观察：**tile 天然适合做性能建模的抽象，因为它有三个性质。**

| 特性 | 描述 | 优势 |
| --- | --- | --- |
| 确定性 | 给定 tile 配置（形状、流水线深度、内存布局），每个 tile 的资源使用完全确定；无需周期精确模拟或指令集仿真，纯分析即可计算 | 区别于需学习映射关系的 ML 方法，直接计算更高效、更可靠 |
| 可组合性 | tile 信息分层组合：intra-tile（单 tile 内 per-pipeline 资源分解）、inter-tile（tile 间依赖、并发发射、执行顺序）、cross-device（tile 网格通过 placement 跨设备延伸） | 支持模块化建模，三个小模型独立构建再组合，避免单体模型复杂性 |
| 可移植性 | tile 抽象被主流 GPU 架构广泛采用，包括 NVIDIA Ampere/Hopper/Blackwell 与 AMD CDNA2；均基于相似的分层内存与计算结构执行 tile 形 workload | 核心逻辑无需重写，仅需通过微基准测试校准硬件参数（如带宽、吞吐量、延迟），数分钟即可适配新平台 |

::: block-1
:mdnice-highlight[现有 GPU 性能工具的根本问题是抽象不匹配：编程模型已经进化到 tile-centric，但分析工具还停留在 FLOP/byte 标量、黑盒 ML 预测、事后 profiler 三个层次]。Tile 作为抽象同时具备确定性、可组合性、可移植性，这三个性质让第一性原理的白盒建模成为可能。
:::

## 二、TileSight 整体架构：三层 tile 流水线模型

> TileSight **把 tile 作为一等建模单元，在程序的每一层递归应用 prologue-steady-epilogue 三段式流水线模型*。一个 tile 携带 intra-tile 信息，参与 inter-tile 关系，分布式执行只是同一个抽象的 placement 扩展。

整个系统的输入是一个高层 workload 描述，可以是 tiled GEMM、融合 attention kernel、AllGather 后跟 GEMM、或者跨 GPU 的 MoE 路由。workload 固定了 tensor 和它们的 placement，但不指定调度。**TileSight 把它提升为一个 tile 执行计划，暴露调度相关的选择：tile 形状、循环和归约顺序、block swizzle、软件流水线深度、每个 SM 驻留 block 数、分布式切分、collective 实现。**

这个计划刻意避开了线程级细节。它只保留 tile 程序员和分布式 runtime 实际会改的那些选择，而这些选择恰恰决定了哪些 tile 进入流水线、它们占用什么资源、它们之间怎么依赖或并发。:mdnice-bold-italic[缓存流量、wave 效应、通信阶段、流水线重叠都是推导出来的]，不是单独加进去的。

### 2.1 核心数据结构：tile 执行计划

> 表 2 总结了 tile 执行计划的字段，按描述对象分组：单个 tile 本身（intra）还是 tile 之间的关系（inter）。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/06.jpg)

::: image-caption
表 2：tile 执行计划的字段按描述对象分类：描述单个独立 tile 的属性（intra，tile 内），以及描述 tile 间关系的属性（inter，tile 间）
:::

> **intra 部分（单个 tile 本身）** 包括三个东西。

- 第一是 tensor 访问，每个 tile 的 footprint 和 placement 描述符，记录一个 tensor tile 在哪里产生、驻留在哪里——寄存器、架构特定的 TMEM、SMEM、本地缓存/DDR、或者 GPU 组上的分片/副本。reuse 维度喂给 inter-tile 缓存建模。
- 第二是操作类型，tile 执行的动作：load、store、Tensor Core 或 CUDA Core matmul、归约、指数、缩放、远程传输、或者融合组合。
- 第三是资源向量，每个 tile 在独立可调度资源上的时间分解。

> **inter 部分（tile 之间）** 也有三个东西。

- 第一是 tile 网格，空间 tile 形状、启动顺序、swizzle、循环/归约深度、分布式切分，决定了 tile 执行顺序、wave、每个设备上的本地工作。
- 第二是生产者-消费者 DAG，基于 tensor 生产和消费的 tile 之间的边，和 placement 一起固定了一次迭代内的合法顺序。
- 第三是并发和深度，软件流水线阶段、每个 SM 驻留 block 数、哪些 tile 可以一起发射，设置有效流水线深度。

下面图 3 完整展示了 TileSight 在 AllGather+GEMM 这个例子上的设计概览。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/07.jpg)

::: image-caption
图 3：TileSight在AllGather-GEMM上的设计概览。(a) workload仅由算子和tensor placement描述（X在N个GPU上按列分片）；(b) TileSight将其提升为tile调度，DAG横跨内存层次L0-L4；(c) 单一硬件抽象把寄存器、SMEM、L2、HBM、GPU间网络暴露为5层层次；(d) intra-tile资源向量和inter-tile DAG/并发分析喂入递归的prologue-steady-epilogue三段式流水线模型；(e) 引擎把三段式流水线模型渲染为时间线：软件流水线的load和计算重叠，AllGather从placement推导为Net通道；(f) per-tile性能报告，包含延迟、利用率、缓存命中、重叠率。
:::

### 2.2 算法主循环

> 算法 1 展示了整个评估流程的伪代码。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/08.jpg)

::: image-caption
这个流程非常清晰。先处理分布式部分：如果有分布式映射，就切分 tile 计划，推导远程 tensor 访问，把每个远程访问分解成阶段，每个阶段分解成逻辑交换，路由到物理拓扑，用 α-β 模型算时间，然后把这个时间填到对应 tile 的 Net 资源项里。然后处理缓存：CacheTraffic 算 L1.5/L2/DDR 的流量分布，填到内存 tile 的对应资源项里。最后算三段式流水线模型：根据软件流水线阶段数和每个 SM 驻留 tile 数算有效深度，然后用三段式流水线模型算每个 tile 的时间，最后按 wave 聚合。
:::

注意这里的关键设计：缓存和网络都不是后处理修正项，它们都是填充资源向量的某个维度，然后资源向量统一流过三段式流水线模型。这就是三层共享同一个抽象的真正含义——**不管是本地计算、本地内存、还是跨设备网络，在三段式流水线模型看来都是资源向量的一个维度，同一套重叠逻辑适用。**

### 2.3 可移植硬件抽象

> TileSight 只需要影响 tile 执行计划和 placement 描述符的参数。这个抽象镜像了 tensor-placement 层次：本地 placement 映射到寄存器/TMEM/SMEM/缓存/DDR 资源，远程 placement 映射到跨 GPU 和节点的校准网络层次。数值来自厂商规格和轻量微基准测试的实际带宽、利用率上限、网络参数。

表 3 展示了论文评估的几款 GPU 的硬件参数。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/09.jpg)

::: image-caption
表 3：本文评估的 GPU 架构硬件规格：理论峰值（spec）/ 微基准校准值（meas.），注：TileSight 的硬件抽象还包括缓存层级、架构专属 TMEM 带宽、共享内存 / 占用率限制，以及跨 GPU 组的网络层级。为简洁起见未在此列出。*H200 最高频率 1980 MHz，默认频率 1830 MHz。
:::

**注意这里有个重要细节：规格峰值和实测峰值是有差距的，而且差距还不小**。比如 B6000 的 FP32 规格是 117 TFLOPS，但实测只有 88.6，差了 24%。B200 的 FP32 规格 74.5，实测 57.7，差了 23%。:mdnice-bold-italic[TileSight 不用纸面峰值，而是用微基准测试校准的实际可达峰值，这是它准确率高的重要原因之一]。黑盒 ML 方法需要每个架构重新训练，而 TileSight 只需要跑几分钟微基准测试校准这些数字，核心逻辑完全不用改。

TileSight 不建模 warp 级指令发射、编译器寄存器分配、指令粒度的硬件调度、或者包级网络效应。它建模的是 tile 级程序员和分布式 runtime 控制的调度可见效应：tile 形状、tensor placement、复用模式、swizzle 顺序、流水线深度、每个 SM 驻留 block 数、分布式切分、collective 算法、拓扑感知路由。这正是让模型既跨 GPU 代际可移植、又快到能放进调度搜索里的关键取舍。

::: block-1
总得来说，:mdnice-highlight[TileSight 的核心架构哲学是统一：用同一个 tile 资源向量抽象贯穿计算、内存、网络三个维度，用同一个 prologue-steady-epilogue 三段式流水线模型贯穿 intra-tile、inter-tile、cross-device 三层]。缓存和网络不是后处理修正，而是填充资源向量的一等维度。
:::

## 三、Intra-tile：资源向量与三段式流水线模型

> 对于每个 tile，TileSight 把它的属性转换成一个独立可调度硬件资源上的时间向量。这个向量比 roofline 标量表达力强得多，因为不同流水线上的 tile 可以重叠，而竞争同一流水线的 tile 会串行化。

这是 TileSight 和传统 roofline 模型最根本的区别。roofline 说"这个 kernel 要么 compute bound 要么 memory bound"，:mdnice-bold-italic[但实际 GPU 有十几条独立的硬件流水线：Tensor Core、CUDA Core、SFU、TMEM、SMEM、L1.5、L2、DDR、Net。 它们可以并行工作，性能不是由某一个瓶颈决定的，而是由这些流水线怎么重叠决定的。]

### 3.1 资源向量：九维时间分解

> 形式化地，每个 tile 操作 o 对应一个资源向量：

$$
u(o) = \left\langle t_{TC}, t_{CUDA}, t_{SFU}, t_{TMEM}, t_{SMEM}, t_{L1.5}, t_{L2}, t_{DDR}, t_{Net} \right\rangle \tag{1}
$$

这个向量从 tile 的操作、footprint、src/dst placement、以及微基准校准的硬件速率计算而来。

- 一个纯 Tensor Core matmul tile 只填充 TC 项；
- 一个 Blackwell attention tile 还要给 softmax 和 correction 的 load/store 算显式 TMEM 流量；
- 一个从 DDR 来的 load tile 填充 DDR 项（如果访问命中缓存则填充 L1.5/L2 项）；
- 一个远程 load tile 填充 Net 项。

::: block-1
举个具体例子。一个 BF16 的 Tensor Core matmul tile，形状是 128×128×32，那它的 TC 时间就是：$\frac{2 \times 128 \times 128 \times 32\ FLOP}{校准的 TC\ FP16 吞吐量}$。如果它的 A 矩阵从 L2 来，B 矩阵从 DDR 来，那 L2 项就是$\frac{128 \times 32 \times 2 \ byte}{L2 带宽}$，DDR 项就是 $\frac{ 32×128×2 \ byte}{DDR 带宽}$。:mdnice-bold-italic[这些时间不是加起来的，因为它们在不同硬件单元上可以并行 ——TC 在算的时候，L2 和 DDR 可以同时在下个 tile 的数据。]
:::

**资源向量里有两项不是 tile 单独能确定的**。第一个是内存 tile 的 L1.5/L2/DDR 拆分，取决于访问是否命中缓存，这从 3.5 节的 tile 重用距离推导。第二个是远程 tile 的 Net 项，取决于底层通信阶段的路由成本，这从 3.6 节推导。这两项填好之后，资源向量才完整。

### 3.2 三段式流水线模型：prologue、steady、epilogue

> 给定一组带资源向量和 inter-tile 关系的 tile，TileSight 把执行当作流水线来评估。对于一个有 N 次逻辑迭代、有效深度 d 的重复单元：

$$
T = T_{pro} + \max(N - d, 0) \cdot T_{steady} + T_{epi}
$$

其中 $T_{pro}$ 是填充开销，$T_{steady}$ 是每个重复单元的重叠后开销，$T_{epi}$ 是排空开销。:mdnice-bold-italic[同一个三段式流水线模型递归应用在 tile 执行计划的每一层：外层三段式流水线模型（tile-block wave）的 steady 体本身可以是一个流水线（K 循环），它的 steady 体又可以是内部动作序列上的流水线。]

有效深度 d 组合了显式软件流水线阶段和驻留 tile 交错：

$$
d = stages \times resident\_tiles\_per\_SM - 1
$$

所以"每个 SM 两个 block"不是什么特殊情况：它加深了流水线，因为一个 SM 可以在一个驻留 tile-block 等内存的时候发射另一个的工作。这个公式非常重要——很多人知道软件流水线 num_stages，但不知道**每个 SM 驻留多个 block 本质上也是在加深流水线，而且两者是相乘的关系**。Triton 里 num_stages=3、num_warps=4 导致每个 SM 住 2 个 block，有效深度就是 3×2-1=5。

### 3.3 稳态重叠：拓扑序搜索

> 稳态下 tile 序列的开销取决于选择哪个合法顺序，因为使用同一硬件维度的 tile 在那个维度上累加，而独立维度的 tile 重叠：

$$
T_{steady}(\sigma) = \max_r \sum_{o \in \sigma} u_r(o)
$$

受 DAG 中所有数据依赖边约束。选择的稳态是最优合法顺序：

$$
T_{steady} = \min_{\sigma \in Topo(D)} T_{steady}(\sigma)
$$

这在实践中是个很小的搜索，因为真实融合 kernel 的 DAG 约束很强。比如 MLA decode，11 个 tile 动作，无约束排列是 11!接近 4000 万，但合法拓扑序只有 132 个。这个搜索不是 autotuning——它是 tile 计划上的分析调度步骤，足够轻量，可以跑在 cost model 内部。

::: block-1
用 FlashAttention 的例子解释这个公式。

假设一个迭代里有四个动作：LD Q/K（DDR 项 10 拍）、GEMM0（TC 项 8 拍）、softmax（CUDA 项 4 拍）、GEMM1（TC 项 6 拍）。

依赖是 LD Q/K → GEMM0 → softmax → GEMM1。那合法顺序只有一个，稳态时间就是 max(10, 8+6, 4) = max(10, 14, 4) = 14 拍。但如果还有 LD V，它和 GEMM0、softmax 都没有依赖（只要在 GEMM1 之前就行），那 LD V 可以和 GEMM0、softmax 重叠。如果 LD V 也是 DDR 项 10 拍，那它可以和 GEMM0+softmax 的 12 拍重叠，总时间还是 max(10+10, 14, 4) = 20？**不对，DDR 是同一个资源，所以两个 DDR 操作不能重叠，它们的时间要加起来**。:mdnice-highlight[这就是资源向量的意义——同一维度的累加，不同维度的取 max。]
:::

### 3.4 边界开销与 wave 分解

> prologue 和 epilogue 用同样的资源向量计算，但重叠更少。对于 load-compute 流水线，prologue 主要由填充流水线的内存 tile 组成，而 epilogue 由剩余计算和最终 store 组成。融合 tile 会给一个或两个边界增加归约或归一化。这个分离很重要，因为两个稳态瓶颈相同的调度，**在循环次数少或者只发射几个 wave 的时候，端到端时间可能不同——短 kernel 里 prologue 和 epilogue 占比很大。**

占用率改变的是重叠结构，不仅仅是利用率。如果 p 个 tile-block 驻留在一个 SM 上，模型把它们当作同一个 tile 流水线的交错实例；驻留数受 SMEM、寄存器、warp 限制、架构特定的每 SM 最大 block 数约束。

同一个 wave 分解处理 tail 效应：tail wave 可能只用一部分 SM，那些活跃 SM 分到更大份额的共享 L2/DDR 带宽，所以要用活跃 SM 数重新计算三段式流水线模型。这个细节很多模型忽略，但实际影响不小——比如一个 GPU 有 108 个 SM，你 launch 了 200 个 block，第一个 wave 跑 108 个 block，第二个 wave 只跑 92 个，第二个 wave 里每个 SM 能分到的 L2 和 DDR 带宽比第一个 wave 多，所以每个 block 跑得更快。

> 算法 2 会递归遍历 tile 循环结构，枚举依赖合法的顺序。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/10.jpg)

::: image-caption
算法 2：递归三段流水线模型评估。该算法是 TileSight 流水线重叠分析的核心，采用递归结构处理嵌套循环的多层流水线。最外层做波次分解，区分满波和尾波分别计算，尾波因为活跃 SM 更少，每 SM 分到的共享带宽更高，需要单独评估。中间层递归遍历循环嵌套结构，每一层循环都可以有自己的软件流水线深度，有效深度由软件级数和每 SM 驻留 tile 数共同决定。最内层枚举所有合法拓扑序，选择总时延最小的调度，计算对应的序言 - 稳态 - 尾声包络。这种递归设计完美匹配 GPU 内核的多层循环结构，无论是简单的两层 GEMM 循环还是复杂的多层融合内核，都可以统一处理，并且能自然捕捉多 block 驻留对流水线深度的提升作用。
:::

这个递归结构非常优雅。外层循环和内层循环用同一个 ModelOverlap 函数，只是有效深度不同。外层循环的 stage 是每个 SM 驻留的 tile-block 数，内层循环的 stage 是软件流水线阶段数，两者相乘减一就是有效深度。

::: block-1
总得来说，:mdnice-highlight[Intra-tile 层的核心是九维资源向量加 prologue-steady-epilogue 三段式流水线模型。性能是多条独立流水线在依赖约束下的最优重叠]。有效深度由软件流水线阶段和每 SM 驻留 tile 数共同决定，递归应用到多层循环嵌套。
:::

## 四、Inter-tile：依赖、缓存与调度顺序

> 对于内存 tile，L1.5/L2/DDR 拆分不是 tile 孤立的属性：同一个 load-tile 坐标，根据 swizzle、wave 占用、相邻 tile 共享数据多少，可能命中缓存也可能打到 DDR。

这是 TileSight 最有技术含量的部分之一。传统缓存建模要么是 trace 驱动的周期精确模拟，太慢；要么是简单的启发式，太不准。TileSight 把重用距离提升到 tile 粒度，配合高斯近似和采样，把缓存建模速度提升了约五个数量级，同时保持了接近 1 个百分点的准确率。

### 4.1 为什么传统重用距离分析太慢

> 重用距离分析是缓存建模的经典方法，理论上很准，但传统形式化工作在缓存行 trace 上，粒度太低，没法放进分析调度搜索里。图 4 对比了缓存行重用距离和 tile 重用距离。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/11.jpg)

::: image-caption
图 4：缓存行重用距离vs tile重用距离。左：传统缓存行重用距离跟踪数万行条目，在行粒度评估精确SDCM。右：TileSight把重用距离提升到tile大小的块，应用高斯SDCM近似，并在归约轴上采样，保持调度敏感性同时让缓存建模轻量。
:::

- 左边是传统方法：一个 GEMM tile wave 里，A0、B0、C00、C01 这些 tile，每个 tile 包含很多缓存行。重用距离 $D_T=11$ 意味着两次访问同一个缓存行之间有 11 个其他缓存行访问。:mdnice-bold-italic[问题是跟踪表有 K×K 个条目，复杂度是 $K^2$。]
- 右边是 tile 粒度：同样的访问模式，重用距离 $D_T=2$，因为两次访问 A0 之间只有 B0 和 C00 两个其他 tile 块。:mdnice-bold-italic[跟踪表只有 $1^2=1$ 个条目，复杂度降低了几个数量级。]

**关键在于，GPU kernel 本来就是以 tile 为单位调度的，硬件的缓存分配和替换也是以 tile 粒度的内存块为单位进行的，那为什么要降到缓存行粒度去分析呢？** 建模的是 8KB 的 tile 而不是 128 字节的缓存行，跟踪条目减少 64 倍，匹配 tile-centric 调度暴露的粒度，让 block swizzle 和遍历顺序对缓存模型直接可见，还避免了 trace 级缓存模拟。

### 4.2 Tensor 访问与 tile 重用距离

> TileSight 为 tile 网格关联的每个 tensor 引入一个 tensor 访问：per-tile footprint、placement 描述符、重复访问次数、以及同一个数据块被复用的网格维度。reuse_dims 让一个规则覆盖不同算子：tensor 的 reuse key 是 tile 坐标投影到非复用维度上的值。

形式化地，对于有 reuse_dims R 的 tensor，reuse key 计算为：

$$
key(x, R) = \text{LINEARIZE}\left( x_d \mid d \notin R \right)
$$

其中 x 是 tile 坐标，R 是复用维度集合。

::: block-1
这个公式很抽象，但举几个例子就清楚了。对于 GEMM 网格 $(M_t, N_t)$，A 矩阵的 tile 在 $N_t$ 方向复用——同一行 M 的 A tile 被所有 N 列的 tile 共享，所以 A 的 $R = \{N_t\}$，所有同一 M 行的 tile 共享同一个 A key。B 矩阵的 tile 在 $M_t$ 方向复用，所以 B 的 $R = \{M_t\}$，所有同一 N 列的 tile 共享同一个 B key。对于 MLA decode，KV-cache tile 在同一个 batch 元素的不同 attention head 之间复用。对于卷积，权重和激活在 batch、输出通道、空间轴上有不同的复用维度。
:::

**这个设计避免了为每个算子写专门的缓存公式，同时保留了决定复用的调度信息**。具体的 tile 执行顺序——包括 swizzle、行面板遍历、Z-order、持久化 kernel——决定了这些 key 出现的序列，因此决定了它们的重用距离。

tile 重用距离 $D_T$ 是两次访问同一个 tensor 块之间访问的不同 tile 大小数据块的数量。传统重用距离问两次访问之间隔了多少缓存行或内存事务，tile 重用距离问同样的问题，但单位是 GPU kernel 调度暴露的单位。

### 4.3 命中概率与快速评估

> 给定重用距离 $D_T$、相联度 A、以 tile 为单位的缓存容量 $B_T$，随机距离缓存模型（SDCM）估计类 LRU 缓存的命中概率。精确的二项式形式是：

$$
P(h \mid D_T) = \sum_{a=0}^{A-1} \binom{D_T}{a} \left( \frac{A}{B_T} \right)^a \left( \frac{B_T - A}{B_T} \right)^{D_T - a}
$$

这个公式虽然准，但对大 tile 网格里的每个 tensor key 计算都很贵。所以 TileSight 用高斯近似：

$$
P(h \mid D_T)_{approx} = 1 - Q\left( \frac{|A - 1 - \mu|}{\sqrt{\sigma^2}} \right)
$$

其中

$$
\mu = D_T \cdot \frac{A}{B_T}, \quad \sigma^2 = D_T \cdot \frac{A}{B_T} \cdot \left(1 - \frac{A}{B_T}\right)
$$

$Q(x)$ 是标准正态分布的互补累积分布函数。为了进一步降低开销，用 Zelen-Severo 近似计算 CDF $\Phi(x)$：

$$
\Phi(x) \approx 1 - \left( a_1 t - a_2 t^2 + a_3 t^3 \right) \frac{e^{-x^2/2}}{\sqrt{2\pi}}
$$

其中 $t = (1 + 0.33267x)^{-1}$，$a_1, a_2, a_3$ 是常数。

这个近似可以这么理解，重用距离 $D_T$ 意味着在两次访问同一个块之间，有 $D_T$ 个其他块被访问。如果缓存大小是 $B_T$ 个块，相联度 A，那这 $D_T$ 个块里有多少个会把我们要的块挤出去？这本质上是个二项分布问题：每个新来的块有 $A/B_T$ 的概率映射到同一个 set，把我们的块挤出去。当 $D_T$ 很大时，二项分布近似正态分布，所以可以用高斯近似快速计算。

### 4.4 归约轴采样与两级级联

> 除了高斯近似，TileSight 还在归约轴上采样。tile 执行计划暴露归约轴（比如 GEMM 里的 K 轴），TileSight 在这个粒度采样重用事件，而不是重放每个内层循环访问。一个 K=8192、tile_K=32 的 GEMM，检查次数减少 256 倍，准确率损失可以忽略。加上 tile 级重用距离和高斯近似，缓存模型评估速度总共提升了约五个数量级，让缓存建模能在分析循环内部运行，而不是作为离线 trace 分析。

在有中间 L1.5/LRC 层的 GPU 上（H200、B200），TileSight 把 SDCM 作为级联应用——L1.5 在每个物理 SM 组内部，L2 全局，DDR 承载剩余的失效流量。没有这个设计，L1.5 命中概率就是 0，模型退化为单 L2 评估。

**Block swizzle、行面板、Z-order、持久化 block 调度，本质上都是喂给重用距离模拟的具体 tile 坐标序列。** 在一个 wave 内部，TileSight 对 $D_T$ 加扰动来模拟硬件非确定性、顺序 tensor 加载、跨 tensor 缓存老化，这些都从 tile 执行计划和硬件分组推导，不需要 kernel 特定的 profiling。Tail wave 用一部分 SM，因此分到更大份额的共享带宽，所以要为 tail 重新计算三段式流水线模型。

$$
u(o) = \left\langle t_{TC}, t_{CUDA}, t_{SFU}, t_{TMEM}, t_{SMEM}, t_{L1.5}, t_{L2}, t_{DDR}, t_{Net} \right\rangle \tag{1}
$$

算出来的 L1.5/L2/DDR 字节数填充公式 1 的对应项（这里我们再把公式 1 贴出来）所以缓存行为改变的是三段式流水线模型本身，不仅仅是最终延迟。**这一点很重要——缓存命中率高意味着 DDR 项小，DDR 不再是瓶颈，TC 可能成为瓶颈，整个重叠结构都变了。**

> 图 6 展示了 L2 命中率预测的准确率，在 4680 个 GEMM 持久化 kernel case 上和 NCU ground truth 对比。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/12.jpg)

::: image-caption
图 6：TileSight L2命中率预测vs NCU ground truth，横跨4680个GEMM持久化kernel case。A100: MAE=1.46 pp, MAPE=2.33%；H200: MAE=0.88 pp, MAPE=1.50%；B200: MAE=1.05 pp, MAPE=1.52%；B6000: MAE=0.78 pp, MAPE=1.09%。
:::

这张图验证了 TileSight tile 重用距离缓存模型的精度，覆盖四款 GPU 的数千种配置，这个结果相当惊人。每款 GPU 上的平均绝对误差都在 1 个百分点左右，B6000 甚至只有 0.78 个百分点。4680 个点，覆盖不同 shape、不同 tiling 配置，大部分点都落在对角线附近。**要知道这是纯分析模型预测的，没有跑任何 kernel，没有任何 ML 训练，纯粹从 tile 执行顺序和重用距离算出来的。**

文中也提到了一个局限性：重用距离模型假设 tile 以统一速率推进，但实际上 SM 之间会不同步，同时处理不同 K 切片，把并发访问的 tile 扩散到 L2 之外。对于大 K 的 GEMM，这会让实测命中率低于 TileSight 的同步预测（比如 H200 上 M=N=8192、K=28672 的 GEMM，预测 82%，实测 43%）。但这种配置很少见，所以整体误差还是在 1 个百分点左右，模型在这个区间系统性偏乐观。

::: block-1
:mdnice-highlight[Inter-tile 层的核心创新是把重用距离从缓存行粒度提升到 tile 粒度，配合高斯近似和归约轴采样]，把缓存建模速度提升五个数量级，同时保持约 1 个百分点的 L2 命中率预测误差。缓存不是独立后处理，它填充资源向量的内存维度，直接改变流水线重叠结构。
:::

## 五、Cross-device：从 placement 到网络路由

> 跨设备执行是同一个 intra-tile 抽象的 placement 扩展：一个 tile 的源或目的指向另一个 GPU 上的分片或副本，它的资源向量就获得一个非零的 Net 项。

这是 TileSight 最优雅的设计之一。分布式是和本地计算、本地内存完全对称的另一个资源维度。同一个三段式流水线模型、同一个重叠逻辑，自动处理计算-通信重叠。

### 5.1 远程 tensor 访问推导

tensor、expert、sequence、数据并行映射同时切分 tile 网格和它的 tensor tile，产生 GPU 组上的 placement 描述符。切分之后，本地 tile wave 可能需要另一个设备产生的 tensor tile、复制的激活、或者需要先归约才能被后续 tile 消费的部分结果。**TileSight 把这些当作远程 tensor 访问：需要的 collective 或点对点传输直接从生产者-消费者 placement 推导**，每个都变成一个带源/目的设备、字节量、Net 资源使用的 tile。

举个例子，AllGather+GEMM。权重 W 在本地，激活 X 按列分片在 N 个 GPU 上。GEMM 需要完整的 X，所以每个 GPU 上的 GEMM tile 需要从其他所有 GPU 获取 X 的分片。这些远程访问不是用户手动指定的——TileSight 看到 X 的 placement 是"按列分片在 N 个 GPU 上"，而 GEMM tile 要访问整个 X 行，自动推导出需要 AllGather，然后把 AllGather 分解成阶段和逻辑交换。

### 5.2 逻辑交换与拓扑

对于每个推导出来的远程 tensor 访问，TileSight 把需要的 tensor-tile 移动分解成有序阶段。一个阶段由逻辑源-目的交换 $(s, d, b)$ 表示，其中 s 是拥有或产生 tensor tile 的设备，d 是 tile wave 消费它的设备，b 是从 tensor 访问推导的 tile 或分片字节量。

**Collective 算法只是提供不同的阶段分解**：ring all-reduce 用 reduce-scatter 和 all-gather 步骤，tree 算法用归约和广播层级，不规则路由保持点对点。这个表示是 tile 级别的，不是包级别。它保留了推理通信量所需的 tensor-placement 信息，同时让硬件拓扑决定哪条物理链路承载每个交换。

这个设计的好处是解耦了算法和拓扑。同样的 ring AllGather 逻辑，在 NVLink 全互联拓扑上和在 PCIe+IB 拓扑上，路由不同、瓶颈链路不同、性能不同，但 collective 分解逻辑不用改。用户也可以自定义拓扑，提供任意链路的 per-hop 带宽和延迟。

### 5.3 Per-stage 路由成本

> 路由完一个阶段里的交换后，TileSight 用 α-β 通信模型估计阶段时间，匹配跳步延迟和瓶颈链路串行化的分解：

$$
T_k = \underbrace{\max_{(s, d, b) \in \mathcal{E}_k} \sum_{l \in \mathcal{P}_{sd}} \alpha_l}_{\text{路由跳步延迟}} + \underbrace{\max_{l \in \mathcal{L}} \beta_l B_{l,k}}_{\text{瓶颈链路串行化}}
$$

其中 $\mathcal{E}_k$ 是阶段 k 中的逻辑交换集合，$\mathcal{P}_{sd}$ 是 $(s, d, b)$ 的物理路由，$B_{l,k}$ 是经过链路 l 的字节数，$\alpha_l$ 和 $\beta_l$ 是链路 l 校准的启动延迟和逆带宽。

::: block-1
这个公式非常经典，但用在 tile 级别的逻辑交换上是新的。

- **第一项是所有交换中最长路径的总延迟** ——不管你有多少数据，首先要经过这么多跳，每跳有固定启动延迟。
- **第二项是所有链路中最堵的那条的传输时间** ——链路带宽是共享的，多个交换经过同一条链路要排队。两项相加就是这个阶段的总时间。
:::

推导出来的通信序列的代价是其阶段的有序和：$T_c = \sum_{k \in K_c} T_k$。对于有重复相同阶段的算法比如 ring collective，TileSight 评估一个阶段然后乘以阶段数。结果进入公式 1 的 Net 维度，所以跨设备移动被表示为 intra-tile 资源需求，和本地计算通过同一个稳态机制重叠。

> 图 7 展示了纯 collective 的预测准确率，在 H200×8 和 B200×8 上覆盖 AllGather、AllReduce、ReduceScatter、All-to-All 四种操作。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/13.jpg)

::: image-caption
图 7：H200×8 和 B200×8 上纯集合通信算子的预测效果，涵盖 AllGather、AllReduce、ReduceScatter、All-to-All 四种原语。TileSight整体wMAPE 12.22%，GenZ 20.82%，PipeWeave在支持的行上65.72%。
:::

**TileSight 在所有八种配置上都显著优于基线**。特别是 PipeWeave，因为它没有 H200/B200 的原生 collective 模型，只能回退到 H800 的随机森林模型，所以完全反映不出 NVLink4/5 的带宽差异，误差高达 60%以上。:mdnice-bold-italic[而 TileSight 只需要校准 NVLink5 的 α 和 β 参数，核心逻辑完全不用改，误差就在 10-20%之间。]

> 图 8 展示了融合 compute-communication kernel 的结果，这才是真正的考验——纯 collective 是独立的，但融合 kernel 里通信要和计算重叠。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/14.jpg)

::: image-caption
图 8：H200×8和B200×8上融合compute-communication kernel预测（AllGather+GEMM、GEMM+ReduceScatter、Ulysses Attention）。TileSight在融合kernel上达到14.83% wMAPE，两个基线都不支持这种场景。
:::

三个场景：AllGather+GEMM、GEMM+ReduceScatter、Ulysses Attention。H200 上的误差分别是 15.3%、12.2%、14.3%，B200 上是 14.0%、24.6%、14.8%。**注意 B200 上 GEMM+ReduceScatter 误差稍高，24.6%，但考虑到这是零样本预测、没有跑任何 kernel、纯分析推导，这个结果已经非常好了**。更关键的是，两个基线在这种融合场景下完全不支持——它们要么把通信和计算分开算，要么需要用户手动指定重叠率，不能自动推导重叠。

::: block-1
:mdnice-highlight[Cross-device 层的核心洞察是：分布式是 tile placement 的一个特例]，而没有为分布式设计独立模块。远程访问自动推导为 collective 阶段，路由到物理拓扑，用 α-β 模型计算，填充资源向量的 Net 维度。计算-通信重叠是同一个三段式流水线模型自动处理的另一种资源重叠，也并非特化的处理。
:::

## 六、实验评估：从单算子到端到端 vLLM

> 我们从单 GPU 算子到多 GPU LLM 服务评估 TileSight，横跨 A100、H200、B200、B6000、H200-NVL、B200×32 系统，证明**统一的 tile 级抽象可以建模单算子延迟、缓存行为、分布式 kernel、端到端服务**，不需要 per-target 重训练或 profiling。

总共评估了 703 个 GEMM shape、4680 个持久化 kernel 缓存 case、166 个 vLLM decode 配置。硬件覆盖 SXM、PCIe、NVLink4/5、多节点设置，还有 AMD MI210 做跨厂商验证。

### 6.1 单算子预测准确率

> 图 5 展示了 703 个 BF16/FP16 Tensor Core GEMM shape 在五款 GPU 上的结果，以 cutlass_profiler 测量为 ground truth。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/15.jpg)

::: image-caption
图 5：A100、B200、B6000、H200、MI210上GEMM延迟预测vs实测延迟。每个点是一个BF16/FP16 Tensor Core GEMM shape；对角线表示精确预测。TileSight聚合MAPE 12.35%，PipeWeave 21.97%，重训练的NeuSight 32.95%，Roofline 33.85%，GenZ 34.89%。
:::

TileSight 聚合 MAPE 12.35%，**在新架构 B200、B6000、H200 上都是最好的**。NeuSight 在 A100 上微弱领先，因为 A100 在它的训练分布里，但这个优势迁移不到新 GPU 上——这正是 ML 预测器过拟合风险的体现。在 MI210 上，因为 CK 没有提供显式的 rasterization 或 swizzle 控制，TileSight 跑在默认缓存模式，但依然以 23.4% MAPE 领先，PipeWeave 25.5%、NeuSight 26.4%、Roofline 38.8%、GenZ 40.4%。

> 表 4 对比了 TileSight 和 NCU 在融合 FA-3 kernel 上的结果。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/16.jpg)

::: image-caption
表 4：H100 上 FlashAttention-3 建模结果与 NCU 的对比（Qwen 配置：batch=1，64 头，头维度 128）。NCU 为真值。该表是 TileSight 在复杂融合内核上的细粒度验证，覆盖了时延与多维度硬件利用率指标。FlashAttention-3 是典型的异构融合算子，包含 Tensor Core 矩阵乘、CUDA Core 归约、SFU 特殊函数等十余种操作，调度逻辑复杂，是传统性能模型的难点场景。
:::

延迟预测误差只有 2.7%，而且各个资源利用率分量都跟得很紧。这是在非 GEMM 的融合执行上的验证，说明 tile-pipeline 模型不仅仅对 GEMM 有效，对 FlashAttention 这种复杂融合 kernel 也有效。

### 6.2 端到端 vLLM 推理

> 真正的端到端考验是 vLLM 服务。图 9 展示了 166 个健康配置的 decode 吞吐量，覆盖 dense、MoE、单节点、多节点服务。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/17.jpg)

::: image-caption
图 9：dense LLM、MoE模型、多节点配置上的vLLM decode吞吐量预测。Dense行覆盖A100×1、B6000×2、B200×8、H200-NVL。MoE行覆盖B200×8、B200×32、H200-NVL×8。
:::

系统规模从单卡 A100 到 32 卡 B200 多节点，模型从 Qwen、Llama 到 DeepSeek MoE。TileSight 整体达到 13.52% wMAPE，而 PipeWeave 在支持的 dense 行上是 31.84% wMAPE。PipeWeave 不支持 MoE。

> 图 10 把所有点汇总到散点图上。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/18.jpg)

::: image-caption
图 10：所有有效配置下的解码吞吐量预测值与实测值对比。TileSight 整体加权 MAPE 为 13.52%；PipeWeave 在支持的稠密配置上加权 MAPE 为 31.84%。这张汇总散点图直观呈现了两款模型在全场景下的预测精度差异。TileSight 的预测点紧密分布在对角线两侧，误差分布均匀，没有明显的系统性偏差，无论是小吞吐量的单卡场景还是大吞吐量的多卡集群场景都保持了一致的相对误差。而 PipeWeave 的点离散度更高，且在高吞吐量区域偏差明显增大，反映出机器学习模型在分布外场景的泛化短板。
:::

从 10 tok/s 到 20000 tok/s，跨越三个数量级，TileSight 的点基本都落在 y=x 和 y=2x/y=x/2 之间。注意最右边 B200×32 的点，接近 20000 tok/s，:mdnice-bold-italic[TileSight 预测得也很准。这是非常强的结果——从单 kernel 到 32 卡端到端服务，一个统一模型，没有 per-architecture 重训练，13.52%的误差。]

### 6.3 应用：性能诊断与 cost model

> 因为是白盒模型，TileSight 可以直接做性能诊断 🐂。**图 11 展示了 TileSight 指导 tile 配置选择的结果，替代暴力 autotuning。**

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/19.jpg)

::: image-caption
图 11：在 H100 和 MI210 上，由 TileSight 指导 Triton 和 TileLang 选择 tile 配置的内核性能，替代了穷举式自动调优。参考基线包括：多头注意力 / 分组查询注意力 (MHA/GQA) 对应 FlashAttention-3，MLA 对应 FlashMLA，矩阵乘对应 cuBLAS/rocBLAS，反量化矩阵乘对应厂商库。
:::

在 MHA/GQA、MLA、GEMM、反量化 GEMM 四类 kernel 上，TileSight 选出的配置都能和 FA-3、FlashMLA、cuBLAS、rocBLAS 这些高度优化的厂商库打平，在很多 case 上甚至更好。**特别是 MI210 上的 MLA，Triton 默认配置性能极差（有的 case 慢 10 倍以上），而 TileSight 选出的配置和厂商库相当。**

> 表 5 总结了几个具体的诊断 case，TileSight 定位到瓶颈并给出具体优化建议，带来 1.07× 到 8.97× 的加速。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/20.jpg)

::: image-caption
表 5：TileSight 诊断优化的内核性能提升。覆盖 7 个不同场景的真实优化案例。TileSight 能够精准定位四类典型的 tile 级性能瓶颈：间接寻址开销、流水线重叠不足、L2 缓存局部性差、架构专属的内存布局与体冲突问题，并对应给出可落地的优化方向。所有案例均获得显著性能提升，加速比从 1.07 倍到 8.97 倍不等，其中 MLA 内核通过解决多重瓶颈获得了近 9 倍的性能提升。这充分体现了白盒分析模型的核心优势：不仅能预测时延数值，还能解释瓶颈成因并指导优化，这是黑盒机器学习预测器完全无法实现的能力，尤其在 AMD 等工具链相对薄弱的平台上价值更为突出。
:::

最夸张的是 MI210 上的 MLA kernel，基线 66.38ms，TileSight 诊断出 tiling 不合理、内存分配不好、SMEM 冲突三个问题，给出对应解决方案后跑到 7.40ms，加速 8.97 倍。**这就是白盒模型的价值——它不只是告诉你"这个 kernel 慢"，而是告诉你"慢在什么地方、改成什么会快"。**

> 图 12 展示了 TileSight 作为 TileLang cost model 的效果：保留预测前 5%的调度，剪枝 95%的候选，平均能达到穷举搜索最优性能的 99.66%。

![文章插图](/articles/2026-08-04-gemm-fa-vllm-tilesight-tile-4f541dcb/images/21.jpg)

::: image-caption
图 12：TileSight作为TileLang中的cost model：剪枝95%候选调度，保留预测前5%，在10个LLaMA派生的GEMM-FP16 workload上平均达到穷举搜索最优性能的99.66%。
:::

10 个 GEMM shape，最差的也有 98.4%的最优性能，大部分都是 100%。**这意味着 autotuning 不需要跑上千种配置，只需要跑 TileSight 推荐的前 5%，就能得到几乎最优的性能**，autotuning 时间直接减少 20 倍。

::: block-1
总之，**实验结果验证了第一性原理建模的威力**：12.35%单 GPU 误差、1 个百分点 L2 命中率误差、13.52%端到端 vLLM 误差，跨五款 GPU 架构包括 AMD，没有 per-architecture 重训练。:mdnice-highlight[白盒特性让它不仅能预测，还能诊断瓶颈、指导优化、作为 cost model 剪枝搜索空间。]
:::

## 七、局限、启示与未来方向

> TileSight :mdnice-bold-italic[瞄准的是规则的、tile 结构化的程序，其运行时间由资源利用率主导]。**它不建模数据依赖控制流、高度不规则内存访问、指令级编译器决策、未文档化的 warp/CTA 调度、或者闭源 runtime 行为。**

这是论文非常诚实的局限性讨论。了解这些边界，才能正确使用这个工具，也能看到未来的改进方向。

### 7.1 当前局限

| 局限点 | 描述 | 具体表现或影响 |
| --- | --- | --- |
| 硬件抽象聚焦吞吐量 | 延迟受限场景（如小 batch decode attention）及多 die 效应（如 B200 中 SM 到 HBM 的亲和性）需更细粒度的延迟与拓扑参数；小 batch 下 tile 尺寸小、流水线填充不足，prologue/epilogue 占比高，固定延迟（如指令发射、warp 调度）影响显著，而 TileSight 当前主要建模吞吐量项 | 小 batch 场景下性能预测偏差增大，无法准确反映实际延迟瓶颈 |
| 假设 tile 跨 SM 以统一速率执行 | 实际 SM 间存在执行不同步，导致大 K GEMM 的 L2 命中率预测偏乐观 | K=28672 时预测命中率 82%，实测仅 43%；因 SM 异步推进不同 K 切片，L2 中驻留 tile 块增多，命中率下降 |
| 编译器集成与 workload 范围有限 | TileSight 已在选定 GEMM workload 上作为 TileLang cost model 验证，但更广泛编译器集成及非 GEMM 调度搜索尚未支持 | 用户需手动编写 tile-action DAG；理想状态应由编译器自动从 Triton/TileLang 程序中提取 DAG，无需人工干预 |
| 不建模数据依赖控制流与不规则内存访问 | 无法处理含动态索引、数据依赖循环次数（如 while 循环）、稀疏访问等高度不规则模式 | 当前 tile 执行计划表达能力不足，无法支持此类 kernel 的准确建模与调度优化 |

### 7.2 对领域的启示

> TileSight 带来的更大启示是方法论层面的。论文结尾说得很好："The broader lesson is a first-principles one: begin with a compact set of physically grounded mechanisms, and let their composition explain complex execution."

在 ML 方法大行其道的今天，很多人觉得 GPU 这么复杂，肯定得用 ML 来学，纯分析模型搞不定。但 :mdnice-bold-italic[TileSight 证明了，只要找对了抽象层次——也就是 tile——一组紧凑的物理机制（资源向量、流水线重叠、重用距离、α-β 路由）组合起来，就能解释从单核到 32 卡集群的复杂执行，准确率和黑盒 ML 相当甚至更好，同时还是白盒可解释的、跨架构可移植的、不需要训练数据的。]

这对 AI Infra 领域有普遍意义。我们经常急于用 ML 去拟合复杂系统，但有时候问题只是我们没有找到正确的抽象。tile 作为编程抽象已经存在好几年了，但直到这篇论文才有人认真把它作为性能建模的统一抽象。好的抽象能让复杂问题变简单，这是工程的真谛。

另一个启示是可组合性的力量。:mdnice-bold-italic[TileSight 没有一个巨大的单体模型，它是三个小模型：intra-tile 资源向量、inter-tile 缓存与调度、cross-device 路由。它们通过共享的 tile 抽象组合起来，每一层都可以独立改进]。比如未来想建模更细粒度的 warp 调度，只需要改进 intra-tile 层；想建模更复杂的网络拓扑，只需要改进 cross-device 层。这种模块化设计是系统能持续演进的基础。

### 7.3 未来方向

> 基于 TileSight 开辟的方向，能看到几个有价值的未来工作。

| 方向 | 描述 |
| --- | --- |
| 自动 DAG 提取 | 当前需用户手动描述 tile-action DAG；未来可直接从 Triton、TileLang、CUDA Tile 程序的 IR 中自动提取该 DAG，使 TileSight 成为编译器中的一个 pass，透明提供性能反馈。 |
| 更细粒度的延迟和多 die 建模 | Blackwell 架构的双 die 设计引入新层次：SM 到本地 HBM、SM 到远程 die 的 HBM、L1.5 缓存跨 die 共享；需增强拓扑感知建模；小 batch 延迟敏感场景还需纳入指令发射延迟、warp 调度开销等非吞吐量因素。 |
| SM 间不同步的缓存建模 | 当前同步假设在大 K 场景下偏差显著；可引入 SM 间偏移的概率模型，或基于 wave 内并发访问分布精细修正重用距离。 |
| 更广泛的 kernel 支持 | 当前已验证 GEMM、FlashAttention、MLA、归一化等规则 kernel；后续扩展至稀疏算子、动态 shape kernel、控制流主导的 kernel。 |
| 和分布式系统的集成 | 当前 cross-device 层已支持建模 TP、EP、SP 等并行策略的 kernel 性能；未来将与 DistServe、vLLM 等分布式调度系统集成，自动搜索最优并行策略与 tile 配置，替代人工经验。 |

::: block-1
:mdnice-highlight[TileSight 证明了第一性原理建模在 GPU 性能分析领域的可行性。找对抽象层次，一组紧凑的物理机制组合起来就能匹敌甚至超越黑盒 ML 方法，同时保持白盒可解释、跨架构可移植]。它开辟了 tile-centric 性能分析这个新方向，未来从自动 DAG 提取到多 die 建模到分布式系统集成，还有大量工作可以做。
:::

## 结语

> TileSight 这篇论文读下来最大的感受是"优雅"。它**没有发明什么惊天动地的新算法，用的都是经典技术**：三段式流水线模型、重用距离分析、α-β 通信模型。但:mdnice-bold-italic[它找到了正确的抽象——tile——然后把这些经典技术用这个抽象重新组织起来，就解决了一个看起来很复杂的问题。]

有一句话是这么说的**："好的抽象是这样的：它把问题变得如此简单，以至于你会觉得'这不是很显然吗？'，但在它出现之前，没有人想到可以这么做。** "tile 作为编程抽象已经存在很久了，Triton 2019 年就出来了，但直到 2026 年才有人系统地把它作为性能建模的统一抽象。这就是洞察力的价值。

- 对于写 GPU kernel 的工程师来说，TileSight 意味着你不用再靠 autotuning 暴力搜索上千种配置了，不用再对着 NSight 的计数器猜瓶颈了。你描述 tile 计划，它告诉你性能是多少、瓶颈在哪、怎么改会快。
- 对于编译器和 runtime 开发者来说，TileSight 提供了一个白盒 cost model，可以用来剪枝搜索空间、自动调度优化。
- 对于架构师来说，TileSight 提供了一个快速建模工具，可以在流片前评估不同硬件参数对真实 workload 的影响。

::: block-1
:mdnice-highlight[论文说 TileSight 发表后会开源。很期待看到它集成到 Triton、TileLang 这些框架里，成为每个 GPU 程序员的标配工具]。GPU 性能优化这件事，正在从"手艺"变成"科学"。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[比真 GPU 硬件快5-17 倍的时间跳跃仿真框架 REVATI！实现 vLLM/SGLang 跨框架无 GPU 加速仿真，预测误差低于 5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896002&idx=1&sn=7b81c53ef9a19c14aab1b6bb1d0a5361&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**

:::
