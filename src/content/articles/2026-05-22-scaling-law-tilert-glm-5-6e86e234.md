关键词：**TileRT**、大模型推理、低延迟、**异构执行**、:mdnice-bold-italic[Scaling Law]、**软硬件协同**

> 本文深度解析 TileRT 高性能推理引擎，提出速度是大模型推理领域新的 Scaling Law。文章梳理 AI 推理发展三阶段，行业需求从模型质量、Token 吞吐逐步转向低延迟响应。

当前主流推理框架以吞吐为核心设计，算力损耗在算子切换、跨卡同步、内存往返等执行边界，硬件性能无法有效转化为实际响应速度。

![文章插图](/articles/2026-05-22-scaling-law-tilert-glm-5-6e86e234/images/01.webp)

- **速度：大模型推理的下一个 Scaling Law**
- https://www.tilert.ai/blog/speed-as-the-next-scaling-law-zh.html
- 项目：https://github.com/tile-ai/TileRT
- 5000 字，阅读 26 分钟，**播客 21 分钟**

相关推荐

- [TileRT v0.1.3 重磅发布：超低延迟 LLM 推理再提速，GLM-5 支持上线，在线 Demo 即刻体验！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897357&idx=1&sn=5ca8dfaeb17f34d4ac452018efd9fffb&scene=21#wechat_redirect)
- [性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)
- [无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)

> TileRT 重构推理执行模型，通过编译期常驻内核、瓦片级流水线、线程专属化削减单卡执行空耗；创新异构工作器机制，拆分 GPU 运算职责，适配 GLM-5.1 稀疏注意力计算场景。

该引擎已正式上线并承载真实生产流量，多轮版本迭代不断优化长尾延迟、动态路径与长上下文负载问题。

::: block-1
表明单纯算子优化已逼近性能上限，未来需推进模型、编译器、硬件深度协同设计，依靠极致推理速度赋能智能体、实时交互等高阶 AI 应用。
:::

![文章插图](/articles/2026-05-22-scaling-law-tilert-glm-5-6e86e234/images/02.webp)

## 本文目录

- 一、延迟正在成为新的智能
  - AI 推理三阶段演进
- 二、推理性能与硬件上限之间的鸿沟
  - Runtime 开始进入关键路径
- 三、TileRT：重新思考推理引擎
  - 从 Runtime Scheduling 到 Persistent Execution
  - Warp Specialization 与 Tile-Level Pipeline
  - TileRT 想解决的，其实是 Execution Gap
  - 从算子 Runtime 到 GPU 驻留式调度
- 四、从 Warp Specialization 到 Heterogeneous Worker
  - 把 Specialization 从 SM 扩展到整个 NVL 域
  - GLM-5.1 中的异构执行
  - 通信开始进入 Tile Flow
- 五、Production-Ready：从实验室极速版到真实流量
  - 从“跑得快”到“持续跑得快”
  - 真正困难的问题，开始越来越“系统”
- 六、下一阶段：协同设计
  - 模型、编译器与硬件开始重新耦合
  - TileRT 只是开始
- 七、The speed is all you need
  - 合作交流

![文章插图](/articles/2026-05-22-scaling-law-tilert-glm-5-6e86e234/images/03.webp)

## 一、延迟正在成为新的智能

> LLM 服务化以来，推理系统的核心矛盾已经经历了几次迁移。

最早的问题，是如何让超大规模模型真正运行起来。随后，行业开始追求单位成本下更高的请求吞吐，更大的 batch、更深的队列，以及多级 KV cache，逐渐成为推理系统的核心设计方向。

但实时 AI 交互正在改变这一切。

### AI 推理三阶段演进

![文章插图](/articles/2026-05-22-scaling-law-tilert-glm-5-6e86e234/images/04.gif)

- **2023 — 2024 · Chat Era** 代表产品：ChatGPT 经济特征：value < cost 核心诉求：model quality
- **2025 — Now · Agentic Era** 代表产品：Cursor、Claude Code、Codex 经济特征：value ≈ cost 核心诉求：token throughput
- **Next two years · Autonomous Era** 应用场景：工厂智能体、金融、车载系统 经济特征：value »» cost 核心诉求：**SPEED**

> AI 推理的三个时代 — 从模型质量，到 token 吞吐，再到速度，成为下一个 scaling 维度。

Agent、语音交互、代码补全、工具调用，以及 Test-Time Scaling 的兴起，正在把推理重新推向 latency-first。用户不再只关心系统总共能生成多少 token，而开始关心下一次响应是否足够快。这也意味着，那些过去被吞吐掩盖的固定开销，正在重新回到关键路径。

与此同时，Test-Time Scaling 也开始让系统速度直接影响模型能力：在固定延迟预算下，更快的推理意味着更多 rollout、更深的推理路径，以及更强的自验证能力。

但现有推理系统，大多并不是围绕这一目标设计的。

无论是传统推理框架，还是长期围绕单个 Kernel 优化的编译器栈，本质上仍建立在“吞吐优先”的执行假设之上。它们能够在请求充足时有效摊薄开销，却很难在实时交互场景下，把硬件性能真正转化为端到端响应速度。

正是在这样的背景下，TileRT 首次赋能 GLM-5.1 模型，在其官方极速版推理服务 MaaS 平台上线部署，从最初的实验性原型系统，到今天承载真实生产流量，TileRT 又跨出了重要一步。它不仅完成了一次从 prototype 到 production 的跃迁，也代表着我们对大模型推理执行模型的一次系统性重构。

## 二、推理性能与硬件上限之间的鸿沟

> 今天，一台 8×H200 NVL 服务器的聚合内存带宽已经接近 38 TB/s。

对于 GLM-5.1 模型而言，单次 decode 过程中实际激活的参数量约为 42 GB。仅从理论带宽估算，在不启用 MTP 的情况下，token 生成速度的理论上限接近 1000 token/s。

但落到真实系统中，端到端速度往往只有几十 token/s，距离理论上限仍隔着一个数量级的性能鸿沟。也正是从这个落差出发，TileRT 团队开始追问：问题的根源究竟在哪里？

最初，我们以为问题只是 kernel 不够快。直到 profiler 暴露出一些反直觉的现象：GPU 利用率并不低，理论 FLOPS 也并不差，但 token 延迟依然居高不下。

这意味着，**GPU 并不是没有算力。真正的问题在于，算力被困在了一道道执行边界之间。**

### Runtime 开始进入关键路径

> 今天的大多数推理框架，仍然沿用一种经典执行模型：模型被拆成一连串独立算子，每个 operator 单独启动、同步，并完成各自的内存往返。这套抽象在训练时代非常成功，因为 kernel 足够“大”，计算能够自然掩盖 launch、同步与 runtime 调度的固定成本。

但 decode 改变了这一时间尺度。

在 latency-first 要求下，单个 kernel 的生命周期开始进入微秒级。过去在吞吐场景下被摊薄与掩盖的固定开销，重新浮出水面：kernel launch、跨 kernel barrier、global memory spill，以及网络通信的同步开销，都开始进入关键路径，成为端到端延迟中无法忽视的一部分。

在性能分析过程中，我们频繁地看到一种反直觉的现象：kernel 还没真正“热起来”，就已经结束了。

GPU 被迫不断重复执行流程：

```
launch → load → compute → store → synchronize
```

每一次 kernel 执行的边界，都会打断数据流、破坏局部性，并迫使执行单元重新同步。很多时候，真正限制系统速度的，并不是某个 GEMM 的计算速度，而是：下一次计算什么时候才能开始。

过去，runtime 是 GPU 编程的便利层；但在超低延迟推理下，它开始越来越接近性能墙本身。问题开始不再只存在于 kernel 内部，而开始存在于：kernel 与 kernel 之间。

这也是 TileRT 的起点。

## 三、TileRT：重新思考推理引擎

> TileRT 的一个核心判断是：当运行时系统（runtime）的任务编排开始进入延迟关键路径时，问题的答案可能已经不只是“继续优化运行时系统”，而是重新思考推理系统的执行模型。

传统推理框架中，GPU 执行的是一串彼此独立的 kernel，执行流会不断被 kernel launch、同步、operator 边界与 memory round-trip 切碎。

而在 TileRT 中，我们开始尝试另一种思路：GPU 不再执行一串离散 kernel，而是持续执行一个长期驻留的任务流水线。

### 从 Runtime Scheduling 到 Persistent Execution

> TileRT 会在编译期（AOT）将整个模型静态展开为一个常驻 Engine Kernel。

![文章插图](/articles/2026-05-22-scaling-law-tilert-glm-5-6e86e234/images/05.webp)

::: image-caption
Tile-level Task Scheduling：算子被拆解为 tile-level task，调度到 warp group 上，CTA 通过 warp/block specialization 形成异构 worker。operator 被拆解为 tile，task 调度到 warp group，CTA 通过 warp/block specialization 成为 heterogeneous worker。
:::

在整个解码生命周期中：

- Host 只启动一次
- 任务执行流（execution flow）持续驻留在 GPU 内部
- 大量运行时调度被前移到编译期

传统执行模型更接近“operator-by-operator”的连续调度，而在 TileRT 中，任务执行流会进一步被组织成持续推进的 tile 计算流水线。

这里，tile 的意义并不只是更细粒度的任务划分。它本质上是一种新任务调度抽象：计算、通信与异步 IO 会被统一拆解为 tile-level task，并在 GPU 内部持续推进。

### Warp Specialization 与 Tile-Level Pipeline

> 为了维持这种持续执行，TileRT 在 Engine Kernel 内部进一步采用了激进的 Warp / Block Specialization。

不同 warp group 会承担不同职责：

- 一部分负责异步数据搬运
- 一部分负责张量计算
- 一部分负责通信重叠

过去很多执行阶段必须串行推进：

```
load → barrier → compute → barrier
```

而在 TileRT 中，数据搬运、张量计算与通信会开始以 tile 为粒度持续重叠，大量中间结果也不再反复写回 global memory，而会直接停留在 register、shared memory 与 L2 cache 中继续向后流动。

从 profiler 的角度看，这种变化会非常明显：GPU 不再表现得像“不断启动新 kernel”，而更像一个持续运行的 execution pipeline。

### TileRT 想解决的，其实是 Execution Gap

> 随着 decode 延迟被不断压缩，我们逐渐发现，真正吞噬延迟的很多时候并不是计算，而是：

- kernel 与 kernel 之间的等待
- 通信与计算之间的停顿
- runtime 与 device 之间的同步

TileRT 的很多设计——persistent kernel、tile pipeline、warp specialization——本质上都围绕着同一个目标：尽可能压缩 execution gap。

### 从算子 Runtime 到 GPU 驻留式调度

> 当越来越多调度逻辑被下沉进 kernel 内部后，推理系统本身也开始呈现出一种不同于传统 inference runtime 的形态。

**过去执行模式**

- 调度发生在 host runtime
- 同步发生在算子边界
- 通信由外部框架组织

**当前执行模式** 越来越多 execution orchestration 开始驻留在 GPU 内部。

Runtime 不再不断“驱动 GPU”，而更像是在初始化与维护一个持续运行的 execution pipeline。

## 四、从 Warp Specialization 到 Heterogeneous Worker

> Persistent 执行流解决了单卡内部的大量执行空泡。然而，当系统扩展到 8×NVL 后，我们开始遇到另一种 boundary：同构并行 abstraction 本身。

今天的大多数张量并行（TP）框架，都默认所有 GPU rank 执行相同逻辑，并通过同步共同推进执行流。这种方式在稠密训练时代非常自然，但推理开始改变这一点。

随着稀疏路由、Top-K 选择、动态索引、长上下文 attention 与 MTP 执行不断进入系统，越来越多执行阶段开始表现出一种新的特征：它们并不天然适合同构扩展。

很多阶段本身计算量并不大，但高度依赖全局信息、动态依赖与同步。如果仍然强行让所有 rank 执行相同逻辑，往往会引入大量重复计算、广播与同步放大。

### 把 Specialization 从 SM 扩展到整个 NVL 域

> 我们后来意识到：既然 warp 可以 specialization，GPU 本身为什么不可以？

于是，TileRT 开始进一步把 specialization 的思想从 SM 内部向外扩展：

```
warp specialization → block specialization → GPU specialization
```

不同 GPU 不再被视为完全对称的执行单元，而会根据计算密度、通信成本与数据依赖承担不同职责。某种意义上，Heterogeneous Worker 本质上是 Warp Specialization 在更大尺度上的延伸。

### GLM-5.1 中的异构执行

> 下图中，GLM-5.1 的 attention 被拆成异构 worker：GPU 1–7 运行 MLA（Projection、Q/K/V、Selection、Attention、All Reduce），GPU 0 运行 Sparse Indexer（Index Q/K、Sparse Index Score、Top-K、Broadcast、All Reduce）。Sparse Index 反馈给 Selection。

![文章插图](/articles/2026-05-22-scaling-law-tilert-glm-5-6e86e234/images/06.webp)

::: image-caption
GLM-5.1 中 attention 的异构拆分 — GPU 0 运行稀疏索引 Worker，GPU 1–7 运行 MLA Worker。
:::

在 GLM-5.1 的执行流中，我们最终把 attention layer 拆成了两类 heterogeneous worker：

- **GPU0：** 稀疏索引 Worker
- **GPU1–7：** MLA Worker

其中，稀疏索引 Worker 负责 Top-K 选择、稀疏索引构建与路由决策；而 MLA Worker 则负责 RMSNorm、GEMM、Flash Sparse Attention、AllReduce 等计算密集型 stage。

这种拆分的关键，并不只是“功能划分”。更重要的是，不同 execution stage 开始拥有不同的扩展策略。一些阶段更依赖全局信息，同步开销主导计算，因此更适合集中式 execution；另一些阶段则天然适合张量并行。

### 通信开始进入 Tile Flow

> 在传统推理框架中，通信往往仍然是执行流之外的独立阶段。但在 TileRT 中，通信会进一步被下沉进执行流水线内部。整个 attention layer 在 host 侧只对应一次 Engine Kernel 启动，大量广播、归约与同步会直接在 tile-level 任务流水线内部完成。

**传统执行逻辑**

```
compute → sync → compute
```

**TileRT 重叠执行逻辑**

```
compute ↔ communication ↔ compute
```

形成持续重叠的执行流水线。

## 五、Production-Ready：从实验室极速版到真实流量

极致 benchmark 往往是脆弱的。

真正困难的问题，从来不是如何在实验室里跑出一个漂亮数字，而是如何让系统在真实生产流量下，持续稳定地接近硬件极限。

TileRT 从 prototype 走向 production 的过程中，我们越来越明显地感受到：超低延迟推理，本质上是在对执行流水线的持续推进做极限维护。很多 benchmark 环境下不会暴露的问题，在真实流量里会被迅速放大。在 benchmark 中，序列长度往往相对稳定，路由模式更加理想，请求到达也更规整，KV cache 生命周期通常较短。

但真实生产流完全不同。

长短上下文会持续交织，KV cache 会不断增长、碎片化与迁移，不同请求之间的路由模式也会持续波动。而在 MTP 场景下，accept / reject path 甚至会不断改变执行流本身。

### 从“跑得快”到“持续跑得快”

> 过去半年里，TileRT 经历了几轮关键的执行模型重构。但这些重构的核心目标，并不只是继续推高峰值速度，而是解决一个在真实生产环境中更难的问题：如何让执行流持续稳定地跑下去。

在 v0.1.1 版本中，我们进一步压缩了执行流水线的空泡，并引入更细粒度的 overlap，对 Engine Kernel 内部调度方式进行了重新组织。这类优化不一定会显著抬高理论计算性能上限，却会实实在在地改善端到端延迟，尤其是 tail latency。因为在真实服务里，用户感受到的往往不是“最快的一次”，而是“最慢的那几次”。

到了 v0.1.2-alpha.1，MTP（Multi-Token Prediction）开始进入 TileRT 的执行流。表面上看，MTP 是为了“一次生成更多 token”；但从系统角度看，它真正带来的挑战，是执行流开始变得动态。accept / reject path 会持续改变 pipeline 的推进方式，而 draft / verify 之间也会引入新的 synchronization dependency。换句话说，系统不仅要跑得快，还要在执行路径不断变化时，依然保持流水线不断流。

当 v0.1.3 正式支持 GLM-5 模型后，新的系统问题又进一步浮出水面。随着 FP8 与超长上下文场景进入生产环境，KV cache 压力、访存局部性、communication amplification 开始成为影响稳定低延迟的关键因素。此时，瓶颈已经不再只是某个 kernel 是否足够快，而是整个系统能否在复杂负载下持续维持执行流的持续推进。

接下来，我们即将公布 v0.1.4 版本。这个版本将进一步面向超长上下文生产环境进行优化，并引入全新的 Heterogeneous Worker 执行模型。它的目标，是让 TileRT 不只在理想条件下跑出漂亮数字，而是在真实流量、动态路径、长上下文和复杂通信模式交织的生产环境中，持续稳定地跑得快。

### 真正困难的问题，开始越来越“系统”

> 随着性能不断逼近硬件边界，我们越来越明显地感受到，真正困难的问题已经不再只是 GEMM 优化、kernel 调优或 operator 融合。

很多时候，系统不是“算得不够快”，而是**任务执行流水线无法持续稳定地流动** 。

今天，TileRT 已经正式承担 GLM-5 / GLM-5.1 的生产流量。但我们越来越觉得，benchmark 其实只是整个系统问题里相对容易的一部分。真正困难的问题，开始越来越接近：

- runtime architecture
- distributed scheduling
- memory system behavior
- model-system co-design

而这，也正在推动推理系统从“算子优化”逐渐演化为真正的 AI execution infrastructure。

## 六、下一阶段：协同设计

> 当系统性能开始逼近硬件边界后，我们越来越明显地感受到，很多问题已经无法继续通过局部优化解决。

过去几年，大模型系统优化的大部分工作，本质上都发生在算子 / kernel 层：更快的 GEMM、更激进的融合、更高效的 communication overlap，以及更复杂的 scheduling policy。但很多 bottleneck 已经不再局限于某一个 operator，而开始存在于整个任务流水线本身。

我们开始越来越频繁地撞上“结构性边界”：

- memory hierarchy 与模型结构之间的不匹配
- communication topology 与 routing pattern 的冲突
- KV cache 增长与 locality 之间的矛盾
- dynamic sparsity 对传统同构并行 abstraction 的破坏

这些问题有一个共同点：它们已经无法仅靠 runtime 修复。

很多时候，系统本身已经非常接近硬件极限，但模型结构仍然会不断制造新的执行流碎片。于是我们逐渐意识到：未来的极致性能，不可能只来自推理框架本身，而必须来自更深层次的协同设计。

### 模型、编译器与硬件开始重新耦合

> 过去，模型、编译器与硬件之间更像一种单向适配关系：

```
model → compiler → hardware
```

模型先被设计出来，然后 compiler 尝试 lowering，runtime 尝试调度，kernel 工程师再去“补救”执行效率问题。

但在超低延迟推理场景下，这种方式开始越来越困难。随着执行流水线被不断压缩，任何访存局部性被破坏、communication amplification 或 synchronization expansion，都可能直接进入延迟关键路径。

模型结构本身，也开始越来越直接地影响系统执行行为。

### TileRT 只是开始

> TileRT 并不是这个方向的终点，它更像是一次尝试：我们开始不再把推理系统视为“一串被 launch 的 kernel”，而开始把它视为一个持续流动的任务流水线。

过去，大模型 Scaling Law 讨论的是参数量、数据量与训练算力；但当推理开始成为 AI 产品真正的核心环节后，速度本身也开始成为一种新的 Scaling Law。

因为推理速度开始越来越直接地影响模型在固定延迟预算内能够完成的推理深度、交互质量与 Agent 响应能力。

## 七、The speed is all you need

> 如果说过去十年，GPU 的核心任务是尽可能高效地完成大规模并行计算，那么未来几年，它可能会越来越多地承担另一种任务：在极短延迟预算内，持续维持任务流水线。

这会迫使模型结构、编译器、运行时系统与硬件架构一起发生变化。

因为速度正在越来越直接地影响：

- 推理深度
- 交互质量
- Agent 响应能力
- 真实世界生产力

而 TileRT，只是这个方向上的一步。

### 合作交流

> TileRT 团队正在进行高性能 AI 系统的底层构建与极限探索。

如果你对 GPU architecture、compiler optimization、distributed execution 或 large-scale inference system 有兴趣，欢迎与我们交流或加入我们。

- 开源代码库（部分模块）：github.com/tile-ai/TileRT
- 技术与合作交流：tile-ai@outlook.com

> **The speed is all you need.**
