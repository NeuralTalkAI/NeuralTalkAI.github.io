关键词：:mdnice-bold-italic[ThunderKittens 2.0]、**GPU Kernel**、性能优化、CUDA、**Tensor Core**、:mdnice-bold-italic[GEMM]

> 本文是 ThunderKittens（TK）2.0 版本的发布说明，这是一款轻量的嵌入式 CUDA 领域特定语言（DSL，一种针对特定应用场景设计的计算机语言），同时也为感兴趣的读者带来深度的技术解析。

自两年前发布 ThunderKittens（TK）[1]以来，我们将大部分精力投入到新功能的开发中，例如通用代码模板[2]、定制化设备端调度器[3]、英伟达 Blackwell 架构支持[4]、FP8 精度支持[5]、多 GPU 支持[6]以及超级内核（Megakernels，整合多种计算任务的大内核）[7]。

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/01.webp)

- **ThunderKittens 2.0: Even Faster Kernels for Your GPUs**
- https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
- **https://github.com/HazyResearch/ThunderKittens**
- 1 万字，阅读 40 分钟，**播客 20 分钟**

相关推荐

- **[端到端 LLM 编译器 nncase：基于 e-graph 的异构存储架构高性能统一编译框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895279&idx=1&sn=cfe132ed7791edfd649561561681c72d&scene=21#wechat_redirect)**
- **[性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**

> 本次版本更新的特别之处在于，**做“减法”的比重与做“加法”相当**：我们重构了内部代码、剔除了冗余的内存指令与汇编低效代码、简化了构建系统的复杂度，:mdnice-bold-italic[还发现了现代英伟达 GPU 上诸多出人意料的特性，这些发现也为内核优化划定了“避坑指南”。]

本文的写作目的，一是简要宣布版本发布，二是分享我们的部分研究成果。ThunderKittens 2.0 在功能上的更新包括：

| 更新点 | 说明 |
| --- | --- |
| 新增功能 | 支持 MXFP8/NVFP4 低精度格式、CLC 调度、张量内存可控性；新增多款实用工具及 PDL 功能。 |
| 内部重构 | 对核心代码进行大规模重构，过程中识别出大量细微性能低效问题，后续将逐项分析与优化。 |
| 示例简化 | 全面简化所有示例内核的构建结构，显著降低使用门槛，便于用户或智能体快速适配与二次开发。 |
| 行业贡献 | 多家企业基于 ThunderKittens 衍生出内部分支；部分企业主动将优化成果开源回馈，形成良性协作生态。 |

这些改进让我们能够以更少的代码、全新的优化策略编写速度更快的内核。作为实例，我们推出了全新的业界领先级 BF16[8]/MXFP8[9]/NVFP4[10] 通用矩阵乘法（GEMM）内核，**在英伟达 B200 显卡上，其性能持平甚至超越了 cuBLAS**：

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/02.webp)

::: image-caption
图 1 | 全新内核性能测试。所有内核（TK 与 cuBLAS）均采用按位完全相同的随机输入进行基准测试，测试包含 500 次预热迭代、100 次性能分析迭代，并开启 L2 缓存淘汰机制。基准测试的详细方法将在本文后续部分说明。
:::

我们还对所有现有示例内核[11]进行了更新，适配新版 API，同时正基于 TK 积极开发更多业界领先的内核，如 Flash Attention 4、分组式通用矩阵乘法、通用矩阵向量乘法。

版本发布的内容就介绍到这里！更多细节可查阅我们的代码仓库[12]。

> 本文后续将精选我们在优化 TK 过程中发现的有趣技术细节，具体包括：

1. 内存一致性：想要榨干 GPU 最后一点浮点运算性能，这一问题绕不开！通过合理的逻辑设计优化内存同步，是实现峰值性能的关键；
2. 张量核心与内存流水线：部分张量核心指令支持隐式流水线执行，但相关文档说明并不完善，针对不同计算负载的最优内存流水线策略也并非显而易见；
3. 合理向 PTX 汇编器传递优化提示：否则汇编器并不会信任开发者的代码！逻辑完全相同的代码，写法不同也可能生成差异显著的汇编指令；
4. 线程占用率：别轻信代码表面的表现！分布式共享内存在不同流多处理器（SM）上的表现并非完全一致，张量核心指令还会暗中限制线程占用率；
5. 合理开展 GPU 内核基准测试：需兼顾 L2 缓存使用与功耗因素。

我们特意挑选了这些有趣且鲜有资料提及的主题，这些都是榨干 GPU 性能的最后关键。如果只想实现 90%的浮点运算性能，推荐阅读 Modular 发布的这篇优质博文[13]。

## 本文目录

- 关键问题
  - 问题一：关于 NVIDIA 在 `tcgen05.copy` 文档的错误与验证流程
  - 问题二：关于张量内存与 Occupancy 硬限制
- 一、内存一致性
  - 1.1 理解 PTX 内存一致性模型
  - 1.2 张量核心能否感知到输入数据？
- 二、张量核心与内存流水线
  - 2.1 `tcgen05.cp`与`tcgen05.mma`的流水线化执行
  - 2.2 张量内存的流水线化设计
- 三、SM90+架构下单线程指令的 PTX 汇编器行为
- 四、线程占用率
  - 4.1 并非所有流多处理器（SM）都支持所有 cluster 尺寸
  - 4.2 TCGEN05 指令会硬性限制每个流多处理器的线程占用率
- 五、合理开展 GPU 内核基准测试
- 六、未来规划

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/03.webp)

## 关键问题

### 问题一：关于 NVIDIA 在 `tcgen05.copy` 文档的错误与验证流程

> 是怎么发现 `tcgen05.copy` 在 PTX 文档中是一个“ typo”，并因此优化了 NVFP4 GEMM 性能。:mdnice-bold-italic[作为顶尖的优化团队，验证流程为何未能更早发现这一关键性文档错误？这是否反映了 NVIDIA 在 PTX 文档维护上的系统性缺陷？] 在未来，开发者应如何避免被文档误导，甚至反向推导出硬件行为的真实语义？

1. **为何未能更早发现？**
2. **这是否反映 NVIDIA 文档的系统性缺陷？**
3. **未来开发者应如何应对？**

### 问题二：关于张量内存与 Occupancy 硬限制

> 发现 `tcgen05` 指令会**硬限制每 SM 的 occupancy 为 1**，即使只使用了四分之一的张量内存。这是否意味着 Blackwell 架构中“张量内存”实际上并非设计为可被多个线程块共享，而是一个独占资源？如果是这样，那此前关于“分布式共享内存”的宣传是否在误导开发者？ThunderKittens 未来是否会为此设计新的调度策略来绕过这一限制？

1. **张量内存是否是独占资源？**
2. **这是否误导了开发者？**
3. **ThunderKittens 未来是否会设计新调度策略？**

## 一、内存一致性

> CUDA/PTX 提供了多种内存一致性原语。缺少内存同步会导致竞态条件——多个线程同时访问同一内存地址引发的结果异常，而冗余的内存同步则会造成浮点运算性能损失。事实上，优化获取/释放模式的使用方式，是我们的超级内核（MegaKernels）[14]实现性能超越 SGLang 的最后几步优化之一：我们发现，几条冗余的内存栅栏指令会导致性能损失超 10%。

本文将演示如何合理设计内存栅栏（memory fence）的使用方式，以及其如何带来性能提升。本节将以 ThunderKittens 中 Blackwell 架构下块缩放张量核心矩阵乘法流程（`mma(...)`函数）的代码为例展开说明：

```
... // 准备指令与矩阵描述符asm volatile("tcgen05.fence::after_thread_sync;\n");asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");... // 执行五代张量核心矩阵乘法
```

先简单介绍这段代码的作用：该代码执行的是第五代张量核心矩阵乘法运算，操作数 A、B 矩阵分块存储在共享内存中，运算结果累加到张量内存中。同时，该代码还实现了块缩放矩阵乘法（如 MXFP8 或 NVFP4 精度），因此 A、B 矩阵的输入缩放因子也存储在张量内存中。读者无需深入理解块缩放矩阵乘法的原理，感兴趣可查阅另一篇博文[15]。

那为何要使用这两条内存栅栏指令？在张量核心开始处理数据（执行`tcgen05.mma`指令）前，我们需要确保所有输入数据都已加载至指定内存位置（张量内存与共享内存），且张量核心能够读取到这些数据。**这两条栅栏指令作为安全保障，确保所有前置的加载与复制操作均已完成，且操作结果可被读取。**

> 但对该流程的性能分析显示，这两条栅栏指令导致通用矩阵乘法内核的计算吞吐量损失了约 20-30 TFLOP/s。那么问题来了：我们真的需要这两条指令吗？

### 1.1 理解 PTX 内存一致性模型

要弄清楚这两条栅栏指令的作用，首先需要深入理解 PTX 的内存一致性模型。

::: block-1
在 PTX 中，若一个线程对共享内存或全局内存的写操作，与另一个线程对该内存的读操作满足**因果序**，则读操作一定能感知到写操作的结果。
:::

两个内存操作 X 和 Y 满足因果序的条件为：

1. X 和 Y 由同一个线程发起；
2. X 与 Y 完成同步；
3. 存在第三个操作 Z，X 与 Z、Z 与 Y 均满足因果序（传递性）。

我们称 X 与 Y“完成同步”的情况为：

1. X 和 Y 均为在同一屏障上执行的屏障操作（`bar/barrier`）；
2. X 为写-释放操作，Y 为读-获取操作，且 Y 读取到了 X 写入的值。

> 该规则还有一些细节与特殊情况，本文仅介绍核心内容。

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/04.webp)

::: image-caption
图 2 | PTX 中满足因果序的内存操作示例。这张图清晰呈现了 PTX 内存一致性模型的因果序核心规则：同一线程内操作天然有序，跨线程时可通过“释放-获取”同步建立因果关系，还能借助传递性让非直接关联的线程操作形成有序关联（如线程 0 的 X 操作经线程 1 的中间操作，使线程2的Y操作感知其结果），这一规则是 GPU 内核避免竞态条件、优化内存同步效率的关键理论支撑，也是 ThunderKittens 2.0 精简冗余内存栅栏、提升内核性能的重要依据。
:::

PTX 中还有**内存代理** 的概念：代理指一组内存访问方法，大多数内存操作（如`st`存储、`ld`加载）使用通用代理，而部分异步操作（如`tcgen05.mma`、张量内存异步加载`cp.async.bulk.tensor`（TMA））使用异步代理。上述的因果序仅在同一代理内有效，若要让不同代理的操作满足因果序，必须在其中插入代理栅栏指令（`fence.proxy`）。

### 1.2 张量核心能否感知到输入数据？

当张量核心开始执行矩阵乘法（执行`tcgen05.mma`指令）时，必须能感知到两类数据：

1. 通过 TMA 加载至共享内存的输入矩阵 A、B 分块（即通过`cp.async.bulk.tensor`指令加载）；
2. 通过`tcgen05.cp`指令加载至张量内存的输入缩放因子。

需要说明的是，张量核心还需确保收尾线程（epilogue threads）不再访问累加器，本文暂不讨论该问题。

我们必须全面分析写操作（`cp.async.bulk.tensor`与`tcgen05.cp`）与读操作（`tcgen05.mma`）之间的执行逻辑，判断张量核心是否一定能感知到输入数据。要回答这一问题，需要整合 PTX 文档中分散的相关信息，并进行严谨的逻辑推导。下文将结合文档相关章节逐一分析。

> **第一，张量核心能否感知到 TMA 加载至共享内存的数据？** 我们逐步分析：

1.首先执行从全局内存到共享内存的内存复制操作，该操作通过 TMA 加载指令（`cp.async.bulk.tensor`）完成，属于**弱内存操作** （参考文档 9.7.9.27.1.2 节[16]），且基于异步代理执行（参考文档 9.7.9.25.2 节[17]）；

2.紧随 TMA 加载指令的是`mbarrier`（内存屏障）的`complete-tx`操作，该操作与前置的内存复制操作满足隐式序（参考文档 8.9.1.1 节[18]），且属于释放操作（参考文档 9.7.9.27.1.2 节[19]）；同时，操作完成后会自动插入一条通用-异步代理栅栏指令（参考文档 9.7.9.25.2 节[20]）；

3.张量核心矩阵乘法执行前的`mbarrier`·`try-wait`操作，默认属于获取操作（参考文档 9.7.13.15.16 节[21]），ThunderKittens 正是使用了该默认行为，因此该操作与 TMA 加载操作建立了因果序；

4.最后，`tcgen05.mma`指令由发起`mbarrier·try-wait`操作的同一线程执行，继承了因果序，且该指令通过异步代理读取共享内存数据（参考文档 9.7.16.6.5 节[22]），与 TMA 加载使用的代理一致。

::: block-1
**结论**：在该特定场景下，通过传递性，TMA 写操作与张量核心读操作已满足因果序，读取共享内存时无需额外的内存栅栏指令。
:::

第一个问题解决，接下来分析第二个。

> **第二，张量核心能否感知到`tcgen05.cp`加载至张量内存的数据？**

1.根据文档 9.7.16.6.2 节[23]，`tcgen05.mma`与`tcgen05.cp`指令支持**隐式流水线执行** （下文将详细说明）；

2.根据文档 9.7.16.6.4.1 节[24]，支持隐式流水线执行的指令无需额外的内存排序机制。

**结合以上两点可得出结论：若`tcgen05.cp`与`tcgen05.mma`由同一线程发起，二者会自动满足有序性**。下文将说明，这是编写块缩放通用矩阵乘法内核的标准范式，且我们尚未发现跨线程发起这两条指令的优势。因此，用于确保`tcgen05.mma`能感知`tcgen05.cp`结果的栅栏指令（即示例代码中的`after_thread_sync`）并无必要。

::: block-1
这一发现意义重大：:mdnice-bold-italic[两条内存栅栏指令实际均为冗余。在通用矩阵乘法与注意力机制内核中剔除这类冗余栅栏指令后，性能提升了约 20 TFLOP/s。]
:::

## 二、张量核心与内存流水线

### 2.1 `tcgen05.cp`与`tcgen05.mma`的流水线化执行

> 在 MXFP8/NVFP4 精度的内核中，**将缩放因子加载至张量内存** 是主要的性能瓶颈。

先列出一组关键数值，供后续分析：

- MXFP8 精度下，每 32 个元素对应 1 个缩放因子；
- NVFP4 精度下，每 16 个元素对应 1 个缩放因子。

要充分利用 Blackwell GPU 的张量核心性能，MXFP8 精度要求每个线程块（CTA）的通用矩阵乘法形状为 128×128×32，NVFP4 精度则为 128×128×64。实现`tcgen05.mma`流水线化执行的标准方式，是为连续 4 次矩阵乘法运算准备足够的无中断数据，即 MXFP8 精度下每个线程块为 128×128×128，NVFP4 精度下为 128×128×256。我们将这连续 4 次矩阵乘法称为**一个 MMA 阶段**。

根据上述数值可推导（本文忽略缩放因子重排）：

- MXFP8 精度：每个线程块、每个 MMA 阶段、每个操作数需要 $128\times128/32=512$ 个缩放因子；
- NVFP4 精度：每个线程块、每个 MMA 阶段、每个操作数需要 $128\times256/16=2048$ 个缩放因子。

::: block-1
问题在于，块缩放矩阵乘法需要将相同的缩放因子广播至张量内存中的 4 个线程束（warp），而支持该功能的指令仅有`tcgen05.cp.32x128.warpx4`。从指令名可知，该指令每次调用仅能复制 $32\times128/8=512$ 个值（其中 128 指字节数）。
:::

**这对 MXFP8 精度非常友好**：单个操作数仅需 1 次`tcgen05.cp`调用，即可为一个 MMA 阶段供数。

但对 NVFP4 精度而言，单个操作数一个 MMA 阶段就需要 4 次`tcgen05.cp`调用；同时处理 A、B 两个操作数，调用次数翻倍；再加上 B 矩阵内存布局的细节限制，还需再乘以 2，最终**一个 MMA 阶段需要 12 次`tcgen05.cp`调用**。

调用次数过多会带来显著开销，我们最初的内核设计如下：

```
if (warp_id == 0) {    通过TMA加载A、B矩阵分块（高带宽内存HBM -> 共享内存SMEM）} else if (warp_id == 1) {    通过TMA加载A、B矩阵缩放因子（高带宽内存HBM -> 共享内存SMEM）} else if (warp_id == 2) {    等待缩放因子加载至共享内存    通过tcgen05.cp加载缩放因子（共享内存SMEM -> 张量内存TMEM）} else if (warp_id == 3) {    等待矩阵分块加载至共享内存    等待缩放因子加载至张量内存    执行4次矩阵乘法运算}
```

在上述伪代码中，线程束 2 在一个 MMA 阶段需要执行 12 次`tcgen05.cp`调用，线程束 3 还需显式等待所有调用完成。这些开销导致内核吞吐量大幅下降，比业界领先水平低约 10%。

经过数周的调试，我们终于发现了关键线索——PTX 文档 9.7.16.6.2 节关于**tcgen05 流水线指令** 的说明：

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/05.webp)

::: image-caption
图 3 | PTX 文档 9.7.16.6.2 节内容。这张图展示了英伟达Blackwell架构中tcgen05异步指令的流水线规则：异步tcgen05操作的执行顺序可能与发起顺序不同，但特定指令对可形成流水线，保证执行顺序与发起顺序一致。这些配对包括同N的mma→mma、copy→mma、shift→mma、shift→cp、mma→shift等，这一机制是ThunderKittens 2.0实现NVFP4精度GEMM性能提升约10%的关键优化依据。
:::

> 文档中提到`tcgen05.copy`与`tcgen05.mma`支持隐式流水线执行，我们反复查阅文档后才意识到，`tcgen05.copy`实际是`tcgen05.cp`的笔误——:mdnice-bold-italic[文档中再未出现过`tcgen05.copy`指令。同时，文档中也没有`tcgen05.cp`与`tcgen05.mma`流水线执行的示例，这让我们在很长一段时间内都未能发现这一关键特性。]

基于这一发现，我们重新设计了内核：将复制操作与矩阵乘法操作合并至同一个线程执行，并移除了不再需要的屏障等待，最终恢复了约 500 TFLOP/s 的性能损失，**NVFP4 精度通用矩阵乘法的性能提升约 10%。** 优化后的内核设计如下：

```
if (warp_id == 0) {  通过TMA加载A、B矩阵分块（高带宽内存HBM -> 共享内存SMEM）} else if (warp_id == 1) {  通过TMA加载A、B矩阵缩放因子（高带宽内存HBM -> 共享内存SMEM）} else if (warp_id == 3) {  等待矩阵分块加载至共享内存  等待缩放因子加载至共享内存  通过tcgen05.cp加载缩放因子（共享内存SMEM -> 张量内存TMEM）  执行4次矩阵乘法运算}
```

### 2.2 张量内存的流水线化设计

> 现有资料中常见的设计范式，是将矩阵乘法与张量内存读取做流水线化整合，核心思路为：张量内存的存储结构为 128×512 的数组，非块缩放的`tcgen05.mma`指令每次最多只能将结果累加到 128×256 的张量内存区域。

因此，自然的设计思路是将张量内存分为两个 128×256 的区域，交替使用：

- 一个区域供张量核心执行累加运算；
- 另一个区域供收尾线程读取上一次矩阵乘法的结果。

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/06.webp)

::: image-caption
图 4 | 张量内存缓冲设计：每个流多处理器（SM）的 128×512 张量内存分为两个区域，可同时供张量核心与收尾线程访问。将 128×512 的张量内存划分为两个128×256的区域，左侧区域用于存储已完成的第 i 次 MMA（矩阵乘法累加）结果，由收尾线程读取并写回 HBM（高带宽内存）；右侧区域用于正在进行的第 (i+1) 次 MMA 运算，由张量核心进行累加。这种双缓冲设计实现了无流水线气泡的并行执行，让张量核心和收尾线程同时工作，避免了硬件闲置，是提升小尺寸 GEMM（通用矩阵乘法）性能的关键优化手段。
:::

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/07.webp)

::: image-caption
图 5 | 上述缓冲设计的流水线执行流程：张量核心依次执行 MMA 0、MMA 1、MMA 2 三次矩阵乘法累加操作，而收尾线程则在 MMA 0 完成后启动 Epi 0，在 MMA 1 完成后启动 Epi 1，将张量内存中的结果写回高带宽内存。这种设计通过双缓冲交替使用张量内存，实现了无气泡的流水线并行，让张量核心和收尾线程同时工作，避免了硬件闲置，是提升小尺寸通用矩阵乘法（GEMM）性能的关键优化手段。
:::

> 该设计在理论上无懈可击——不存在张量核心因等待收尾线程完成而闲置的**流水线气泡**。实际测试中，该设计在**小尺寸通用矩阵乘法** （约 2048×2048×2048 以下）中表现出最优性能。

但针对**大尺寸通用矩阵乘法**，我们发现**双累加模式** 的性能更优。:mdnice-bold-italic[该模式会同时运行两条矩阵乘法流水线，共享同一个 A 矩阵分块，分别处理不同的 B 矩阵分块]。这意味着整个 128×512 的张量内存都会被用于累加运算，张量核心会在收尾线程读取内存结果时进入短暂等待。

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/08.webp)

::: image-caption
图 6 | 张量内存缓冲优化设计：每个流多处理器（SM）的 128×512 张量内存分为两个区域，均由张量核心或收尾线程独占访问。具体来说，针对大尺寸 GEMM 优化的**双累加模式**张量内存布局为：将 128×512 的张量内存划分为两个 128×256 的独立累加区，分别用于存储同一A矩阵分块与不同B矩阵分块（B0、B1）的 MMA 结果。这种设计通过共享A矩阵分块大幅减少了内存访问量，虽然会在矩阵乘法之间引入轻微的流水线气泡，但整体性能优势抵消了开销，可使 BF16 精度 GEMM 性能提升约 100 TFLOP/s，是大尺寸矩阵运算场景下的关键优化策略。
:::

![文章插图](/articles/2026-02-21-20-500-tflop-s-thunderkittens-2514c5f1/images/09.webp)

::: image-caption
图 7 | 双累加模式的流水线执行流程：注，张量核心硬件会对 A×B0 与 A×B1 两条流水线做串行化处理，图片为简化示意图未展示该细节。具体来说，张量核心会先串行执行A×B0的三次 MMA（MMA 0、MMA 1、MMA 2），再串行执行 A×B1 的三次 MMA，而收尾线程则在 A×B0 的 MMA 完成后启动 Epi 0，在 A×B1 的 MMA 完成后启动 Epi 1，将结果写回高带宽内存。这种设计通过共享 A 矩阵分块大幅减少了内存访问量，虽然会在两组 MMA 之间引入轻微的流水线气泡，但整体性能优势抵消了开销，可使 BF16 精度 GEMM 性能提升约 100 TFLOP/s，Again再次说明！是大尺寸矩阵运算场景下的关键优化策略。
:::

该模式会在矩阵乘法之间引入轻微的流水线气泡：张量核心需要等待收尾线程从张量内存中读取完全部 256KB 数据（每个元素 4 字节，$128\times512\times4$），才能开始下一轮运算。但共享 A 矩阵分块大幅减少了内存访问量，这一优势抵消了流水线气泡的开销，**BF16 精度通用矩阵乘法的性能提升约 100 TFLOP/s**。

## 三、SM90+架构下单线程指令的 PTX 汇编器行为

特别感谢纳什·布朗[25]发现了这一特性！

> 现代 GPU 内核的常见设计范式是**线程束专用化** （Warp Specialization）：为每个线程束分配单一的功能角色。在英伟达 Hopper 与 Blackwell 架构的 GPU 中，许多角色仅需线程束内的**单个线程** 发起指令即可，例如“加载线程束”仅需一个线程发起 TMA 加载指令。

ThunderKittens 原本的设计代码如下：

```
if (warpgroup::groupid() == 0) {    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {        tma::expect_bytes(arrived, sizeof(smem));        tma::load_async(smem, A, {0, 0}, arrived);    }}
```

其中，`arrived`为`mbarrier`（内存屏障）对象，`smem`为共享内存分块。对 ThunderKittens 不熟悉的读者可参考：这段代码会选中第一个线程束组的第一个线程束，再选中该线程束内的第 0 个线程。从源码层面看，该设计合理，因为 TMA 加载仅需单个线程发起。

但查看该代码生成的 SASS 代码（GPU 硬件可直接执行的机器码）后，我们发现了问题：

```
0x0260 @P0 ELECT P1, URZ, PT ;0x0270     UTMALDG.5D [UR8], [UR4] ;0x0280 @P1 PLOP3.LUT P0, PT, P1, PT, PT, 0x8, 0x80 ;0x0290     PLOP3.LUT P1, PT, PT, PT, PT, 0x8, 0x80 ;0x02a0 @P0 BRA.U.ANY 0x260 ;
```

需要注意的是，这段代码已经在`warpgroup::warpid() == 0 && warp::laneid() == 0`的分支内，理论上仅有一个线程会执行该代码。但 PTX 汇编器却插入了一个**循环**：先选中一个线程发起 TMA 加载，将该线程从活跃集中移除，再跳回 0x0260 地址，从剩余 31 个线程中重新选一个，以此类推，遍历线程束内的所有 32 个线程。

出现该问题的原因有两点：

1. `UTMALDG` 指令（TMA 加载对应的 SASS 指令）不允许同一个线程束内的多个线程同时发起；
2. PTX 汇编器无法验证该代码路径仅由单个线程执行，因此会保守地插入**串行化循环**。

如何规避这一问题？我们需要使用汇编器能识别的**单线程选择指令** ——`elect.sync`。该 PTX 指令会从当前线程束中选中唯一的一个线程，功能与`warp::laneid() == 0`一致，但汇编器能识别其“单线程执行”的设计意图，从而避免插入循环。

在 ThunderKittens 中，可通过`warp::elect_leader()`函数实现该功能，修改后的代码如下：

```
if (warpgroup::groupid() == 0) {    if (warpgroup::warpid() == 0 && warp::elect_leader()) {        tma::expect_bytes(arrived, sizeof(smem));        tma::load_async(smem, A, {0, 0}, arrived);    }}
```

修改后生成的 SASS 代码如下，**循环被成功移除**：

```
0x01e0*/      ELECT P0, URZ, PT ;0x01f0*/ @!P0 BRA 0x2d0 ;... (地址计算)0x02b0*/      UTMALDG.5D [UR8], [UR4] ;0x02d0*/      BSYNC.RECONVERGENT B0 ;
```

将该设计范式应用到内核中所有的单线程指令后，**小尺寸通用矩阵乘法的计算吞吐量提升了约 10%。**

## 四、线程占用率

> 本节将介绍我们在优化过程中踩过的坑，帮助读者规避同类问题：

### 4.1 并非所有流多处理器（SM）都支持所有 cluster 尺寸

大于 2 的线程块集群（threadblock clusters）[26]，可通过启用分布式共享内存、减少内存控制器流量提升计算吞吐量，但存在一个隐患：**部分集群尺寸会导致调度器无法充分利用所有流多处理器**。

::: block-1
具体表现为：以 B200 GPU 上的持久化网格内核（persistent grid kernel）为例，B200 拥有 148 个流多处理器，常规设计思路为：将网格尺寸设为 148，让每个线程块占用一个流多处理器的大部分或全部共享内存，通过`__cluster_dims__(n)`指定集群尺寸，然后启动内核。
:::

- 集群尺寸为 2 时，该设计表现良好：每两个流多处理器组成一个集群，共可运行 $148/2=74$ 个线程块集群。
- 但当集群尺寸为 4 时，我们原本预期每 4 个流多处理器组成一个集群，共运行 $148/4=37$ 个集群，实际结果却并非如此。

为验证这一问题，我们编写了测试内核：打印集群与线程块信息，休眠一段时间后退出，代码如下：

```
__global__ __cluster_dims__(2) void kernel(const __grid_constant__ globals<C> g) {    printf("num_clusters=%d, cluster_idx=%d, cluster_size=%d, cta_rank=%d, blockIdx.x=%d\n", nclusterid().x, clusterIdx().x, cluster_nctarank(), cluster_ctarank(), blockIdx.x);    for (int i = 0; i < 500000; i++) __nanosleep(10000);}
```

> 运行测试后发现了令人意外的结果：**同一时间仅有 132 个流多处理器处于活跃状态，剩余 16 个线程块需等待前 132 个执行完毕后才会被调度**。为进一步验证，我们测试了所有 2 的幂次 cluster 尺寸，结果如下：

| 集群尺寸 | 活跃流多处理器数量 |
| --- | --- |
| 2（网格尺寸 148） | 148 |
| 4（网格尺寸 148） | 132 |
| 8（网格尺寸 144） | 120 |
| 16（网格尺寸 144） | 112 |

**表 1 B200 GPU 不同集群尺寸下的活跃流多处理器数量**

::: block-1
这意味着，:mdnice-bold-italic[在持久化网格内核中盲目设置大于 2 的集群尺寸，会导致难以排查的性能下降。我们推测，这一问题的根源在于：分布式共享内存需要流多处理器之间的硬件连线，而英伟达为简化硬件设计，未为部分流多处理器设计该连线。]
:::

> 这是否意味着英伟达的线程块集群与分布式共享内存设计存在误导性？

`__cluster_dims__()`属性的设计确实不够直观，但并非设计缺陷。研究发现，**放弃使用`__cluster_dims__`，转而通过`cudaLaunchKernelEx`启动内核**，可指定两个集群尺寸：首选尺寸与最小尺寸。

调度器会先以首选尺寸为流多处理器分组，例如以 4 为集群尺寸为 132 个流多处理器分组，再以最小尺寸为剩余的流多处理器分组。当然，内核代码需要兼容两种集群尺寸。ThunderKittens 2.0 中新增了 LaunchConfig 工具类[27]实现该功能，该工具是 CUDA API 的轻量封装，读者也可直接使用原生 API。

### 4.2 TCGEN05 指令会硬性限制每个流多处理器的线程占用率

> 我们还发现一个令人意外的特性：**内核一旦访问张量内存，每个流多处理器的最大线程占用率会被硬性限制为 1**。以下面的简单测试代码即可验证：

```
__global__ void empty_kernel() {}__global__ void tcgen05_kernel() {    __shared__ uint32_t tmem_addr;    asm volatile(        "{tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;}"    ::  "l"(reinterpret_cast<uint64_t>(&tmem_addr)), "n"(128)    );}void print_occupancy(const char* name, const void* kernel, int block_size) {    int max_blocks_per_sm = -1;    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(        &max_blocks_per_sm,        kernel,        block_size,        0    ));    printf("%s: Max blocks per SM: %d\n", name, max_blocks_per_sm);}int main() {    constint block_size = 256;    print_occupancy("empty_kernel", reinterpret_cast<constvoid*>(empty_kernel), block_size);    print_occupancy("tcgen05_kernel", reinterpret_cast<constvoid*>(tcgen05_kernel), block_size);    return0;}
```

编译并运行该代码后，输出结果显示：`tcgen05_kernel`在每个流多处理器上的最大线程块数量为 1。

这一结果令人意外，因为该内核仅占用了张量内存的 1/4 空间，且 PTX 中所有张量内存分配与管理指令都隐含“张量内存可由多个驻留线程块共享”的设计意图。但实际测试表明，**在 Blackwell 架构的内核中使用张量核心时，提升每个流多处理器的线程占用率并非可行的优化策略**。

## 五、合理开展 GPU 内核基准测试

> 在数周的通用矩阵乘法内核基准测试中，我们一直被一个问题困扰：使用 CUTLASS 自带的性能分析器[28]测试时，CUTLASS 的通用矩阵乘法内核性能犹如一座大山始终无法超越，比我们的内核快 100-150 TFLOP/s。我们甚至在自己的内核名前加上`cutlass`前缀[29]，该操作是怀疑英伟达为自家代码预留了专属的汇编器优化通道，不少知乎上的讨论也不少确实有效。

::: block-1
幸运的是，事实并非如此。我们最终发现，**CUTLASS 性能分析器会默认将输入值四舍五入为整数后，再传入内核**。这一设计选择本身就颇具争议，且会对性能测试结果产生显著影响：:mdnice-bold-italic[类整数的输入矩阵会减少 GPU 晶体管的位翻转次数，从而降低功耗，减少时钟频率节流的概率。]
:::

这一发现让我们开始深入研究，最终意识到**诸多因素都会影响浮点运算性能的测试结果**，基准测试的每一个细节设计都至关重要：

- 使用 2 个 CUDA 事件还是 $N\times2$ 个。N 为性能分析迭代次数
- 迭代间如何清理 L2 缓存：显式刷新 vs 通过输入组自然淘汰）
- 基于 C++还是 PyTorch 做测试
- 使用何种随机分布与随机种子
- 等等

> 根据我们的经验，上述设计选择的差异，可能导致测试结果的偏差高达 10%。

经过一系列实验，我们制定了一套**GPU 内核基准测试规范**，并基于该规范发布性能测试数据，具体内容为：

1. 使用**按位完全相同的随机输入**：通常为均匀分布，数值范围根据精度调整；
2. 若输入总尺寸小于 3 倍 L2 缓存大小（如 B200 的 L2 缓存为 128MB），则使用**多组输入**，让每组输入自然淘汰前一组的 L2 缓存数据，模拟冷缓存场景；:mdnice-bold-italic[我们发现，即使合理使用 CUDA 事件，显式缓存刷新也会显著增加测试耗时；]
3. 性能分析前执行**500 次预热迭代**，让 GPU 达到功耗稳定状态；
4. 执行**100 次性能分析迭代**，内核连续启动，迭代间不进行同步；
5. 使用**2 个 CUDA 事件** 记录性能分析迭代的开始与结束时间，计算耗时；
6. 两款内核的测试之间，让 GPU**短暂闲置**，完成散热。

我们推荐业界采用并优化该规范，以下伪代码展示了核心流程。代码基于 CUDA C++，也可通过 PyTorch 的`torch.cuda`API 实现完全相同的逻辑：

```
int l2_cache_size;cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);int num_input_groups = (input_size >= l2_cache_size * 3) ? 1 : int(l2_cache_size * 3 / input_size) + 1;// 准备num_input_groups组输入，确保所有测试内核的输入按位完全相同cudaDeviceSynchronize();for (int i = 0; i < 500; i++) {    input_group_index = i % num_input_groups;    launch_kernel(inputs[input_group_index]);}cudaEventRecord(start_event);for (int i = 0; i < 100; i++) {    input_group_index = i % num_input_groups;    launch_kernel(inputs[input_group_index]);}cudaEventRecord(end_event);cudaEventSynchronize(end_event);float milliseconds;cudaEventElapsedTime(&milliseconds, start_event, end_event);// 保存耗时与吞吐量数据sleep_ms(500);
```

## 六、未来规划

> ThunderKittens 现已完成针对 Blackwell 架构的全面优化，且相比其他 CUDA 领域特定语言，能以更少的代码编写业界领先的 GPU Kernel！同时，TK 2.0 也已开始在工业界落地应用：例如，Cursor 的 Composer 模型的训练 Kernel、Together AI 的推理内核，均基于 TK 开发。

但仍有大量极致的优化方向，能进一步提升内核性能。其中最关键的是：**现代 GPU 采用多芯粒设计**，每个芯粒都有独立的 L2 缓存与高带宽内存（HBM）。

::: block-1
以英伟达 B200 为例，其由两个芯粒组成，芯粒间的互连带宽为 10TB/s，远低于 L2 缓存的带宽（我们实测约为 21TB/s）。这带来了不可避免的:mdnice-bold-italic[非统一内存访问（NUMA）效应：即使是 L2 缓存命中的场景，内存吞吐量也会受芯粒间互连带宽的限制。若能通过内核设计或模型架构优化解决该问题，性能将迎来大幅提升。]
:::

我们对超级内核的研究[30]也尚未止步！我们已验证，即使在计算密集、GPU 友好的负载场景（如预填充）中，超级内核的性能也显著优于业界领先的推理引擎。

> 目前的核心问题是**如何降低超级内核的使用门槛**：我们认为，无需再开发新的领域特定语言或独立编译器，而是需要对机器学习基础设施进行**根本性改造**。

虽然 PyTorch 是一款优秀的计算图表示前端，我们对此深表感谢；但它已有十余年历史的后端架构，可能并非榨干现代 GPU 性能的最优选择。未来会变成什么样，让我们拭目以待！

参考资料

[1] ThunderKittens: *https://github.com/HazyResearch/ThunderKittens*

[2] 通用代码模板: *https://hazyresearch.stanford.edu/blog/2024-10-29-tk2*

[3] 定制化设备端调度器: *https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla*

[4] 英伟达 Blackwell 架构支持: *https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell*

[5] FP8 精度支持: *https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8*

[6] 多 GPU 支持: *https://hazyresearch.stanford.edu/blog/2025-11-17-pk*

[7] Megakernels: *https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main*

[8] BF16: *https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/gemm/bf16_b200*

[9] MXFP8: *https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/gemm/mxfp8_b200*

[10] NVFP4: *https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/gemm/nvfp4_b200*

[11] 现有示例内核: *https://github.com/HazyResearch/ThunderKittens/tree/main/kernels*

[12] 代码仓库: *https://github.com/HazyResearch/ThunderKittens*

[13] matrix-multiplication-on-nvidias-blackwell-part-1-introduction: *https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction*

[14] MegaKernel: *https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main*

[15] cursor.com/blog/kernels: *https://cursor.com/blog/kernels*

[16] data-movement-and-conversion-instructions-cp-async-bulk-tensor: *https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor*

[17] async-proxy: *https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy*

[18] program-order-async-operations: *https://docs.nvidia.com/cuda/parallel-thread-execution/#program-order-async-operations*

[19] data-movement-and-conversion-instructions-cp-async-bulk-tensor: *https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor*

[20] async-proxy: *https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy*

[21] parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait: *https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait*

[22] tcgen05-memory-consistency-model-smem-access: *https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-smem-access*

[23] tcgen05-memory-consistency-model-pipelined-instructions: *https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions*

[24] tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread: *https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread*

[25] nashbrown1: *https://www.linkedin.com/in/nashbrown1/*

[26] thread-block-clusters: *https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters*

[27] LaunchConfig 工具类: *https://github.com/HazyResearch/ThunderKittens/blob/9df1476fe54d56fcb255aa7cf3128cc9222827f3/include/common/util.cuh#L454*

[28] CUTLASS 自带的性能分析器: *https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html*

[29] 内核名前加上`cutlass`前缀: *https://news.ycombinator.com/item?id=45458948*

[30] MegaKernels: *https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main*
