关键词：**CudaProfiler**、:mdnice-bold-italic[TIRx]、内核内性能剖析、**GPU流水线优化**、CUDA内核调优

> 本文介绍 TIRx 框架下的 CUDA 内核剖析工具 `CudaProfiler`。该工具参考了 FlashInfer 的内核侧 profiling 实现，针对 TIRx API 做了适配改造，基于 PTX 的 `%globaltimer` 指令采集内核级时序数据，能够精准呈现 GPU 内核中各线程组的流水线重叠状态。

在 Hopper、Blackwell 等新一代架构上，高性能机器学习内核普遍采用线程组专用化的流水线设计：:mdnice-bold-italic[不同的 warp 或 warp-group 分别承担数据搬运、矩阵计算、归一化、结果写回等不同职责。此时仅统计内核整体启动耗时，只能得到最终的性能结果，无法定位流水线内部的瓶颈——比如数据加载是否及时、计算单元是否存在空等、阶段衔接是否产生尾部延迟。]

![文章插图](/articles/2026-07-03-cudaprofiler-tirx-warp-2f0a6698/images/01.webp)

- **TIRx notes: 一个简洁的plug-in in-kernel profiler实现**
- 作者：侯博涵，https://zhuanlan.zhihu.com/p/2054305616391304228
- Zhihu 主页：https://www.zhihu.com/people/spec-64

::: related-title
**相关推荐**
:::

::: related-links
- **[TIRx：TVM新一代硬件内核DSL，FP8/NVFP4/FlashAttention4实测性能逼近行业最优实现，底层统一支撑人工、Agent、MegaKernel三类高性能内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902815&idx=1&sn=e440508ea6c74f9393cf6d5406e56ed8&scene=21#wechat_redirect)**
- **[MLC.ai 开源《Modern GPU Programming For MLSys》，从 GEMM 到 FlashAttention：Blackwell 时代的 GPU 编程方法论](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902890&idx=1&sn=3e4f0a13f3c75cc78f033969d0d53e0c&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**

:::

> `CudaProfiler` 支持在内核中对数据加载、计算、存储、校正等逻辑阶段手动打点，生成精细化的时序轨迹。工具允许自定义线程组划分规则，既支持规整的 warp-group 拆分，也兼容跨 warp 边界的非规整分组；每个分组由唯一的主导线程采集时间戳并写入缓冲区，内核执行结束后，既可在主机侧解码得到各阶段耗时，也可导出为 Perfetto 格式的可视化轨迹文件。

**整体实现采用纯插件式设计，无需修改 TIRx 编译器核心，接入成本极低**。当然工具本身也存在一定性能开销：额外的计时读取、全局内存写入与线程块屏障操作，会轻微干扰内核原本的执行时序，在资源极度紧张的极端场景下可能影响分析结果的准确性。

同类工具在 Triton、CuTeDSL 中也有对应 API，FlashInfer 同样内置了类似的实现（profiler.cuh[1]）。**本文介绍的实现思路与 FlashInfer 版本大体一致，主要差异在于适配了 TIRx 的编程接口与代码生成体系。** 相关代码与文档：

- bench.py 实现源码[2]
- 官方文档：CUDA 内核内剖析[3]

![文章插图](/articles/2026-07-03-cudaprofiler-tirx-warp-2f0a6698/images/02.webp)

::: image-caption
上图展示了 FlashAttention-4 前向内核中一个线程块（CTA）的运行时序：group_0 负责发起 TMA 加载（issue-tma-*），group_3 / group_4 执行 softmax 流水线（softmax-max / -exp2 / sum），group_5 负责校正计算。生产者与消费者 warp-group 之间是否真正实现了执行重叠，正是内核内 profiling 要回答的核心问题。
:::

## 本文目录

- 一、设计初衷：流水线内核需要更细粒度的性能观测
- 二、基础使用：内核内打点与结果导出
  - 2.1 内核侧打点
  - 2.2 结果解析与可视化
- 三、分组机制：灵活控制剖析粒度
  - 3.1 典型场景：按 warp-group 划分
  - 3.2 灵活适配：非 warp 倍数的分组
- 四、实现原理：插件式设计与底层机制
- 五、开销与局限性
- 总结

![文章插图](/articles/2026-07-03-cudaprofiler-tirx-warp-2f0a6698/images/03.webp)

咕咕嘎嘎🐧

## 一、设计初衷：流水线内核需要更细粒度的性能观测

> Hopper、Blackwell 架构上的高性能机器学习内核，普遍已经演进为 warp 专用化的细粒度流水线：

- 部分 warp / warp-group 负责数据搬运
- 部分负责矩阵乘计算
- 部分负责 softmax、结果校正、尾声处理等环节

决定内核最终性能的关键，往往不只是单段代码的执行速度，而是不同角色的执行单元之间是否充分重叠、哪一方在等待另一方、哪个环节拖长了流水线的尾部延迟。

**仅靠内核整体启动耗时，只能得到最终的性能数字，无法拆解内部的执行状态**。比如 TMA 生产者是否及时把数据送到计算单元、消费者 warp-group 有没有空转等待、softmax 的多步计算是否产生了尾部开销，这些问题都需要深入内核内部，按逻辑执行阶段分别打点才能回答。

::: block-1
`CudaProfiler` 解决的就是这个问题：它让内核开发者可以把 `issue-tma`、`softmax-exp2`、`correction` 这些自定义的逻辑阶段，标记到同一条内核内时间轴上，直观呈现各阶段的起止与重叠关系。
:::

## 二、基础使用：内核内打点与结果导出

工具的使用方式非常简洁，核心是在内核代码中用 `start` / `end` 包裹需要观测的代码区域。

### 2.1 内核侧打点

> 开发者只需要在内核中构造 `CudaProfiler` 实例，然后对目标代码段进行标记：

```
prof.start(Event.Load)# 执行数据加载逻辑prof.end(Event.Load)prof.start(Event.Compute)# 执行核心计算逻辑prof.end(Event.Compute)prof.start(Event.Store)# 执行结果写回逻辑prof.end(Event.Store)
```

运行过程中，设备侧每个线程块/线程组的主导线程（leader thread）会读取 GPU 全局计时器 `%globaltimer`，将时间戳与事件标签一同写入预先分配的 profiling 缓冲区。内核执行结束后，主机侧读取该缓冲区，即可解码出每个线程块、每个线程组的事件时间线。

### 2.2 结果解析与可视化

> 主机侧可以直接从缓冲区中解析出各阶段的耗时，也可以调用工具提供的接口将结果导出为 Perfetto 轨迹文件，在浏览器中通过 Perfetto UI 查看交互式的时间轴。

由于所有时间戳都来自 GPU 全局统一计时器，而非单个 SM 的周期计数器，不同线程块的事件处于同一时间坐标系下，可以直接对比先后与重叠关系。

## 三、分组机制：灵活控制剖析粒度

> `CudaProfiler` 支持将一个线程块内的线程划分为 `num_groups` 个逻辑分组，每条 `(block, group)` 对应时间轴上的一条独立轨道。分组规则完全由开发者定义：一个分组可以是一个 warp-group、一个单独的 warp，也可以是任意数量的线程集合，且不需要严格对齐 warp 边界。这是因为记录逻辑只用到了带断言的单线程写入和线程块屏障，不依赖 warp 集合操作。

使用分组机制需要遵循两条基本规则：

1. 线程通过调用 `init(group_id)` 加入对应分组，该调用会将线程的写入指针定位到分组对应的缓冲区轨道；
2. 每个分组有且仅有一个主导线程负责实际写入记录，需要通过断言条件选出，且该线程必须已通过 `init` 加入对应分组。

由于每个主导线程都持有独立的写入指针，同一条 `start` / `end` 语句会同时在全部分组中记录事件——每个分组的主导线程会在自己的轨道上打下时间戳。

### 3.1 典型场景：按 warp-group 划分

> 最常见的用法是将分组与 warp-group 一一对应。例如一个 256 线程的线程块可以拆分为两个 warp-group，每个 warp-group 分配独立的 `group_id`，并以组内第一个线程作为主导线程。

以下示例中两个 warp-group 执行的计算量不同，最终两条轨道会呈现出不同的耗时：

```
NUM_GROUPS = 2p = CudaProfiler(    prof,    write_stride=NUM_BLOCKS * NUM_GROUPS,    num_groups=NUM_GROUPS,    default_leader=(tid % 128 == 0),  # 每个 warp-group 的第一个线程)if tid < 128:    p.init(0)else:    p.init(1)# ... 数据加载阶段 ...p.start(Ev.Compute)if tid < 128:    for _ in range(1000):             # warp-group 0: 轻量计算        acc = acc * T.float32(1.0001) + xelse:    for _ in range(5000):             # warp-group 1: 重度计算        acc = acc * T.float32(1.0001) + xp.end(Ev.Compute)
```

输出示例：

```
block 0 group 0: load=96 ns, compute=3040 ns,  store=64 nsblock 0 group 1: load=96 ns, compute=10816 ns, store=64 nsblock 1 group 0: load=96 ns, compute=3072 ns,  store=64 nsblock 1 group 1: load=128 ns, compute=10784 ns, store=64 ns
```

### 3.2 灵活适配：非 warp 倍数的分组

> 分组不要求对齐 warp 边界，非整数倍 warp 的划分同样可以正常工作。例如一个 128 线程的线程块可以按 48 / 48 / 32 的比例拆分，每个分组以起始线程作为主导，即使前两个 48 线程的分组跨越了 warp 边界，也能各自记录出准确的时间轨迹：

```
NUM_GROUPS = 3  # 分组范围：[0, 48) [48, 96) [96, 128)p = CudaProfiler(    prof,    write_stride=NUM_BLOCKS * NUM_GROUPS,    num_groups=NUM_GROUPS,    default_leader=((tid == 0) | (tid == 48) | (tid == 96)),)if tid < 48:    p.init(0)elif tid < 96:    p.init(1)else:    p.init(2)
```

输出示例：

```
block 0 group 0: load=96 ns, compute=4544 ns, store=64 ns   # 48 线程（1.5 个 warp）block 0 group 1: load=64 ns, compute=4512 ns, store=96 ns   # 48 线程，跨越 warp 边界block 0 group 2: load=64 ns, compute=4576 ns, store=64 ns   # 32 线程
```

## 四、实现原理：插件式设计与底层机制

> `CudaProfiler` 采用完全插件化的实现，没有对 TIRx 编译器核心做任何修改。

```
@T.meta_classclass CudaProfiler:    """A lightweight wrapper around T.timer_* CUDA intrinsics.    Stores repeated arguments used by timer_init/start/end/finalize so users can    call concise methods in kernels. Intended to mirror Pipeline/TileScheduler helpers.    When ``profiler_enabled`` is False (or a false-y PrimExpr), calls to    ``init/start/end/finalize`` become no-ops. This allows constructing a    profiler unconditionally and eliminating external ``if PROFILER_ON:`` guards.    """
```

它是一个解析期（parsing-time）的类，内部方法通过调用 TIRx 预置的设备端内联算子完成功能；对应的 CUDA 内建函数代码生成逻辑注册在后端 intrinsic 模块中，整个工具完全基于现有编译器框架扩展实现。

核心类的定位是对 `T.cuda.timer_*` 系列内建函数的轻量封装，它会缓存初始化时传入的公共参数（缓冲区、步长、分组数等），让内核代码中只需要调用简洁的 `start` / `end` 方法，无需重复传递参数。同时工具支持通过 `profiler_enabled` 参数一键关闭：传入 `False` 后，所有方法都会变为空操作，profiling 代码可以保留在源码中，编译时自动消除，不需要额外的条件判断包裹。

::: block-1
底层的计时与写缓冲逻辑，最终会生成对应 PTX 指令：通过 `mov.u32` 读取 `%globaltimer_lo` 得到低 32 位纳秒级时间戳，与打包后的事件标签组合成 64 位记录，写入全局缓冲区。事件标签按位域打包，包含 SM ID、线程块 ID、分组 ID、事件索引与事件类型（开始/结束/瞬时/终止），主机侧通过位运算即可完成解码。
:::

## 五、开销与局限性

> 任何内核内剖析工具都会引入额外开销，`CudaProfiler` 也不例外，它对执行时序的影响主要来自四个方面：

- 读取 GPU 全局计时器的指令开销
- 向全局 profiling 缓冲区写入数据的内存开销
- 每个事件前后的线程块内存屏障（`__threadfence_block`）
- profiling 缓冲区本身的显存占用与主机侧解码成本

因此 profiling 工具的介入会一定程度上改变内核原本的执行时序。如果内核本身资源非常紧张，额外的指令与内存操作可能大幅改变编译产物的寄存器分配、指令调度，此时分析结果的准确性会有所下降。

除此之外，工具本身还有一些固有约束：

- `%globaltimer_lo` 仅提供 32 位纳秒计时，约每 4.29 秒发生一次溢出，跨越溢出点的时间段会得到错误的耗时结果；
- 计时精度为数十纳秒级别，极短的代码段可能只能读到 0 或单个计时刻度；
- 事件记录仅包含时间戳与事件 ID，不支持额外载荷，需要区分的信息必须编码在事件枚举或分组编号中。

::: block-1
总体而言，这是一款面向调试与调优阶段的工具，适合定位流水线瓶颈，不建议在生产环境中开启。
:::

## 总结

> TIRx 内置的 `CudaProfiler` 为 GPU 机器学习内核的精细化调优提供了一套轻量化、低成本的剖析方案，填补了传统性能工具无法观测内核内部流水线状态的空白。

它跳出了“仅靠整体耗时判断性能”的局限，能够精准定位线程组间的阻塞、空等、尾部延迟等深层问题；灵活的分组机制也可以适配各类专用化内核的剖析需求。

尽管存在一定的运行开销与精度限制，仅适用于调试分析场景，但纯插件式的接入方式、零侵入的编译器适配设计，让开发者可以快速完成内核时序打点与可视化分析，大幅降低了高性能 CUDA 内核的调优门槛，对于优化 FlashAttention 等复杂流水线内核的并行效率、挖掘 GPU 硬件算力潜力具有很高的实用价值。

参考资料

[1] profiler.cuh: *https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/profiler.cuh*

[2] bench.py 实现源码: *https://github.com/apache/tvm/blob/main/python/tvm/tirx/bench.py#L570*

[3] 官方文档：CUDA 内核内剖析: *https://tvm.apache.org/docs/tirx/native_basics/cuda/profiling.html*

::: related-title
**相关推荐**
:::

::: related-links
- **[TIRx：TVM新一代硬件内核DSL，FP8/NVFP4/FlashAttention4实测性能逼近行业最优实现，底层统一支撑人工、Agent、MegaKernel三类高性能内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902815&idx=1&sn=e440508ea6c74f9393cf6d5406e56ed8&scene=21#wechat_redirect)**
- **[MLC.ai 开源《Modern GPU Programming For MLSys》，从 GEMM 到 FlashAttention：Blackwell 时代的 GPU 编程方法论](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902890&idx=1&sn=3e4f0a13f3c75cc78f033969d0d53e0c&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**

:::
