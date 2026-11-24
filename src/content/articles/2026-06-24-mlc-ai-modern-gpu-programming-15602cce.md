**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：:mdnice-bold-italic[Blackwell]、**TIRx**、GEMM、**FlashAttention**、异步协作

> 如果说过去的 GPU 优化还像“给流水线加速”，那么 Blackwell 时代的 GPU 编程更像“给一支交响乐团编排乐谱”。TMA 负责搬运，Tensor Core 负责计算，TMEM 负责中转，mbarrier 负责协调，warpgroup 和 CTA 负责分工，而 TIRx 则把这一切翻译成可运行、可验证、可推导的代码。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/01.webp)

- **Modern GPU Programming For MLSys**
- 本书循序渐进讲解现代GPU内核编程：认识GPU硬件→掌握GPU编程方法→编写顶尖内核程序。全书以 Blackwell GPU为核心研究对象，详细介绍其存储层次、张量存储、张量核心、异步数据传输引擎、线程束组与计算簇等关键组件。书中依托TIRx（新一代张量中间表示语言）展开讲解，这是一门可在中间表示层级编写GPU内核的Python领域专用语言。
- https://github.com/mlc-ai/modern-gpu-programming-for-mlsys
- 📖 Read it online: https://mlc.ai/modern-gpu-programming-for-mlsys/

::: related-title
**相关推荐**
:::

::: related-links
- **[暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)**
- **[NVIDIA 开源 GPU 编程教学仓库 accelerated-computing-hub，从单线程 kernel 到协作归约](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900741&idx=1&sn=8a88431b7a8d1d46aa3b03f9889bc151&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::

> `modern-gpu-programming-for-mlsys` 并不是一本只讲技巧的书，它更像一张地图：**它先告诉你现代 GPU 为什么变复杂，再带你一步步走进 GEMM、FlashAttention、布局系统和异步调度的核心腹地**。读懂它，:mdnice-bold-italic[意味着你不仅能写出更快的 kernel，还能真正理解“快”是如何被系统性设计出来的。]

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/02.webp)

::: image-caption
这本《Modern GPU Programming For MLSys》书中有大量的交互式例子，比方上图。这张图展示 GPU 内存重排单元 Swizzle Atoms 的对比效果，swizzle atom 是内存中执行地址重排的连续存储块，当前采用 128B 格式、f16 数据，基础原子块为 8 行 8 列，重排规则为重排列号 = 逻辑列号异或行号。交互式的教程，让你能更好地理解很多看似晦涩复杂的概念。
:::

## 本文目录

::: related-links
- 一、这不是“GPU 技巧合集”，而是一套现代算子工程学
- 二、仓库的核心目标：把现代 GPU 的“复杂协作”讲清楚
- 三、Part I：先把“硬件世界”讲明白，才谈得上优化
- 四、Part II：TIRx 的价值，不只是“能写”，而是“能表达硬件语义”
- 五、Part III：GEMM 是试金石，也是方法论的落地点
- 六、Part IV：FlashAttention 是“把所有前面学到的东西一次性用上”
- 七、这本书最难得的地方：它把“工程细节”讲成了“可迁移的方法论”
- 八、为什么这本书值得 AI 系统工程师认真读
- 结语：现代 GPU 编程的终点，不是更快，而是更懂

:::

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/03.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、这不是“GPU 技巧合集”，而是一套现代算子工程学

> 这份仓库的第一印象很明确：它不是一个零散代码集，也不是单纯的课程讲义，而是一部围绕现代 GPU 编程重建认知结构的教材。README 开门见山地给出三段式主线：**理解 GPU 硬件 → 学会编程 → 写出 SOTA kernel**。这个顺序非常关键，因为它承认一个现实：现代 GPU 的性能上限，已经不只是“把循环写对”，而是“把硬件当系统来组织”。

仓库主页写得很直接：它的主角是 **Blackwell-class GPU**，而不是泛化的“某种 CUDA GPU”。:mdnice-bold-italic[这意味着书中讨论的不是传统的共享内存 + 寄存器 + warp 这种老式心智模型，而是更现代的执行与存储层次：]

- Tensor Memory Accelerator（TMA）
- Tensor Memory（TMEM）
- Tensor Core 的异步 MMA
- warpgroup / cluster 级协作
- mbarrier 这样的异步同步机制

这套技术栈背后的逻辑也很明确：**性能不是来自单点爆发，而是来自“搬运、计算、同步、布局”四个环节的整体协同。** 从仓库首页的内容结构可以看出作者的写作哲学：

1. 先讲硬件世界如何运作；
2. 再讲 TIRx 这种接近硬件的 DSL 如何表达 kernel；
3. 最后用 GEMM 和 FlashAttention 两个最具代表性的算子，把所有机制串起来。

::: block-1
这种设计很像编译器课程里的“从语义到 IR 再到 lowering”，但这里的对象不是语言，而是 GPU kernel。**它把“优化”从经验主义提升为结构化工程：你知道每一步为什么发生，也知道它依赖哪种硬件能力。**
:::

## 二、仓库的核心目标：把现代 GPU 的“复杂协作”讲清楚

> 如果只看目录，你会发现它覆盖的章节非常完整：`chapter_background`、`chapter_performance`、`chapter_data_layout`、`chapter_tma`、`chapter_tensor_cores`、`chapter_tmem`、`chapter_async_barriers`、`chapter_clc`、`chapter_gemm_basics`、`chapter_gemm_async`、`chapter_gemm_advanced`、`chapter_flash_attention` 等等。

这不是“内容多”，而是一个明显的结构化铺陈：**先给出 GPU 的物理和抽象边界，再给出性能模型，再给出布局与异步调度工具，最后落地到实际 kernel。**

从 README 和 `index.md` 的叙述能看出，作者其实在回答一个更深的问题：

> 当 GPU 不再是简单的“SIMT 加速器”，而是一个拥有多级存储、专用搬运引擎和独立计算引擎的复杂系统时，程序员该如何建立正确的心智模型？

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/04.webp)

::: image-caption
这张图展示 Blackwell 架构流式多处理器 SM 内部硬件构成，单 SM 搭载第五代 tcgen05 张量核心、通用 CUDA 核心、228KB 共享内存 SMEM、128 通道张量内存 TMEM、寄存器堆与 TMA 数据搬运引擎。TMA 作为专用 DMA 实现全局内存与 SMEM 异步块传输，张量核心读取 SMEM 数据完成乘累加并写入 TMEM，数据再载入寄存器，硬件自动处理地址与访存重排，多组 SM 并行完成张量计算。
:::

这个问题的重要性，在 AI 系统里尤其突出。因为训练和推理的瓶颈，往往不在模型本身，而在**少数关键算子**：

- Attention kernel
- LLM prefill / decode kernel
- 低精度 block-scaled GEMM
- fused MoE 层
- 数据布局与重排

这些算子一旦慢，整条推理链路就慢；它们一旦写得好，系统吞吐就会明显上升。也就是说，**现代 AI 系统的性能很大程度上被这些 kernel 的质量定义。**

::: block-1
这也是为什么这本书的标题里强调了 “for MLSys”。它不是为了 GPU 而 GPU，而是把 GPU 放回到机器学习系统的语境中：**GPU 是系统性能的发动机，而不是孤立的计算设备。**
:::

## 三、Part I：先把“硬件世界”讲明白，才谈得上优化

> 仓库首页把第一部分命名为 **Understanding the GPU**，这里的章节包括执行与内存模型、性能模型、数据布局、TMA、Tensor Core、TMEM、mbarrier、CLC 等。这个顺序不是随意的，它对应的是现代 GPU 优化的认知栈。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/05.webp)

::: image-caption
该图展示双 CTA 集群协同 MMA 计算架构，两个 CTA 分别部署在独立 SM，各自存放分片矩阵 A 与 B，依托集群互联实现跨 CTA 共享内存直读对方 B 分片，无需往返全局内存。单 CTA 仅能完成 128×128 矩阵运算，双 CTA 协同可输出 256×256 结果，分布式共享内存 DSMEM 是跨 CTA 协同张量乘累加的核心支撑。
:::

### 3.1 为什么性能模型必须先于代码优化

> 在 `img/scripts/gen_roofline.py` 中，作者明显采用了 roofline 的思路来解释优化空间：算力上限、内存带宽、算术强度，共同决定程序到底受限于什么。图里甚至直接标出了几个对比点：

- 元素级 / RMSNorm 属于 memory-bound
- naive GEMM 远远没有吃满 SM
- SOTA GEMM 才接近峰值性能

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/06.webp)

::: image-caption
面向 B200 芯片的屋顶线性能分析图，采用双对数坐标，横轴代表算术强度（每字节内存访问对应的浮点运算量），纵轴是硬件可实现浮点性能。黑色折线为硬件性能上限 “屋顶线”，左侧斜段受内存带宽限制，右侧水平段是 fp16 精度约 2PFLOPS 的计算峰值，两者分界阈值约 250FLOP/byte。图中三类负载清晰体现瓶颈差异：红色点位逐元素、RMSNorm 算子，算术强度偏低，属于内存受限场景；橙色是朴素实现的 4096³ 矩阵乘，虽算术强度极高，但算力利用率极低，流式多处理器闲置；绿色为优化后的同款矩阵乘，沿竖轴大幅攀升逼近计算峰值，标注直观展示优化带来的巨大性能增益，清晰区分内存、计算两类性能瓶颈，体现高算术强度场景下算法优化对释放硬件算力的核心价值。
:::

这张图很重要，因为它告诉读者：**优化不是“哪里慢改哪里”，而是要先知道自己卡在 roofline 的哪一侧。**

如果算术强度太低，你再怎么加 Tensor Core 也救不了；如果搬运路径不对，再强的计算单元也会饿死。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/07.webp)

::: image-caption
Blackwell GPU 完整线程层级，由内至外依次为单线程、32 线程构成的 warp、含 4 个 warp 共 128 线程的 warpgroup、驻留单 SM 的 CTA 线程块、跨 SM 共享分布式内存的 CTA 集群，最外层 Grid 对应一次内核启动。warpgroup 是 MMA 运算与张量内存访问的协作单元，该层级单元自 Hopper 架构引入并延续至 Blackwell。
:::

这本书在叙事上非常克制：它没有上来就扔大量公式，而是用“性能模型”先建立判断框架。这样做的好处是，读者在后续看 TMA、TMEM、warp specialization 时，不会把它们当“玄学加速技巧”，而会把它们理解为对瓶颈的针对性修复。

### 3.2 数据布局不是附属品，而是性能本体的一部分

> `chapter_tirx_layout_api/index.md` 直接点明：布局 API 的目标是把书里提出的布局记号变成编译器对象，比如 `TileLayout`、`SwizzleLayout`、`ComposeLayout`。这说明作者对布局的理解不是“存储地址怎么映射”，而是**布局本身就是程序语义的一部分**。

这在高性能 kernel 里非常关键。因为很多时候，性能问题并不来自“算错了”，而来自“存得不对”。例如：

- 同样的数据，如果共享内存排布不合适，会有 bank conflict；
- 同样的 tile，如果没有 swizzle，Tensor Core 可能无法高效消费；
- 同样的 accumulator，如果 TMEM 布局没对齐后续 MMA 需求，就会引入额外搬运。

这本书把布局写成可组合对象，而不是零散参数，这是很“编译器”的做法。它意味着：**布局不是临时决定，而是被静态表达、被复用、被检查、被下传到 lowering pipeline 的。**

### 3.3 TMA：把“搬运 tile”从线程工作变成硬件工作

> 在 `chapter_tma/index.md` 里，TMA 被定义得非常清楚：它是一个在全局内存和共享内存之间进行异步 tile copy 的硬件引擎。最重要的点有两个：

1. **一个线程发起，硬件搬运。**不再是整个 warp 去逐元素读写，而是由单线程发出描述，TMA 引擎完成 tile 迁移。
2. **搬运时顺便解决布局问题。**TMA load 可以在写入 shared memory 时做 swizzle，让 tile 直接落到后续 MMA 需要的形态。

::: block-1
这背后体现的是现代 GPU 编程的一个巨大变化：过去你优化的是“访存指令本身”；现在你优化的是“让访存变成专门引擎的职责”。
:::

`img/scripts/gen_memory_dataflow.py` 把这个路径画得很直观：**GMEM → TMA load → SMEM → tcgen05 MMA → TMEM → tcgen05.ld → RF → thread write → SMEM → TMA store → GMEM**

这条链路非常像一个工业流水线，而不是传统意义上的“一个 kernel 里做所有事”。它揭示了 Blackwell 时代的真实分工：

- TMA 负责移动 tile
- Tensor Core 负责矩阵乘法
- TMEM 负责中间承载
- 寄存器负责局部消费
- 线程只做少量协调工作

这就是现代 GPU 编程的本质：**减少通用线程对搬运和同步的直接介入，让专用单元完成专用工作。**

### 3.4 mbarrier：异步世界里的“完成信号”

> 一旦有了异步搬运和异步计算，就会出现一个根本问题：**谁来证明数据已经好了？**`chapter_async_barriers/index.md` 就是在回答这个问题。

这里的 mbarrier 不只是一个同步原语，而是异步硬件协议的一部分。它有几个关键概念：

- arrival counter：当前还差多少到达
- phase bit：这一轮是第几阶段
- arrive / wait：生产者和消费者的协议动作
- byte-count tracking：针对 TMA 的字节级完成追踪

这类机制的意义在于，它让“程序顺序”不再是唯一可靠的正确性来源。因为在异步硬件中，**发起操作和完成操作不是一回事**。

如果没有 barrier，线程可能在数据还没到时就读它；或者在消费者还没用完时就覆盖它。

书里强调 phase tracking 很重要，因为 barrier 是可复用的。对于一个 pipeline 来说，单次使用 barrier 没价值；关键是它可以跨多个迭代反复工作，而 phase bit 保证旧完成不会冒充新完成。

这实际上是在训练读者掌握一种新心智：**同步不再是“全体停下来等一下”，而是“把每一次 ownership transfer 都用 barrier 表达出来”。**

## 四、Part II：TIRx 的价值，不只是“能写”，而是“能表达硬件语义”

> 如果说 Part I 在建立硬件直觉，那么 Part II 就是在告诉你：**如何把这种直觉写进代码。**

`chapter_intro_tirx` 和 `chapter_tirx_layout_api` 透露出一个很重要的设计理念：TIRx 不是普通 DSL，而是一个尽可能贴近硬件执行模型的 IR 级 Python DSL。

这意味着它要同时满足三种要求：

1. **可读**：代码看起来要像教材里的布局和数据流；
2. **可控**：能表达底层线程、warpgroup、布局、同步；
3. **可编译**：最终能被 lowering 成真正可执行的 kernel。

这本书非常强调“写法和解释几乎一致”。比如在布局 API 中，类似：

```
S[(128, 256) : (1@TLane, 1@TCol)]
```

这种写法并不是“漂亮的注释”，而是在直接构造 `TileLayout`。

::: block-1
这极其重要，因为它让“概念”和“代码”之间的距离缩到最短。很多 GPU 编程教程的问题就在于：讲解时是一个抽象，代码时又是另一个抽象，中间的翻译成本全由读者承担。TIRx 的思路正相反，它要让结构表达本身就是代码的一部分。
:::

> 从编译器角度看，这种设计有三大好处：

- 容易做静态检查
- 容易做 layout propagation
- 容易让 lowering pipeline 保持语义透明

也就是说，TIRx 不是为了让你“少写几行代码”，而是为了让 GPU kernel 的低层结构可以被更精确地描述、分析和优化。

## 五、Part III：GEMM 是试金石，也是方法论的落地点

> 如果说 Part I 和 Part II 是“理论基础 + 编程接口”，那么 Part III 的 GEMM 才是整本书真正进入实战的地方。:mdnice-bold-italic[因为 GEMM 是几乎所有高性能 GPU 编程课程都会拿来讲的算子，但这本书讲 GEMM 的方式明显更现代：它不是停留在 tile 分块，而是一路走到 TMA、persistent scheduling、warp specialization、2-CTA cluster。]

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/08.webp)

::: image-caption
该图展示 Blackwell 架构下矩阵乘 D=A×B 的完整数据流水线：全局内存 GMEM 的 A、B 经 TMA 引擎加载至共享内存 SMEM，由第五代 tcgen05 Tensor Core 完成 MMA 乘累加并暂存张量内存 TMEM，数据再载入寄存器，写回 SMEM 后通过 TMA 存回 GMEM 得到结果 D。下方拆解 SM 内部硬件，包含 TMA 搬运单元、SMEM、TMEM、寄存器、Tensor Core 与通用 CUDA Core，多组 SM 并行工作，清晰呈现 GEMM 全链路数据流转路径。
:::

### 5.1 为什么 GEMM 依然是最好的入门对象

> GEMM 有一个不可替代的教学价值：它足够简单，能把硬件协作讲清楚；它又足够复杂，足以容纳现代 GPU 的关键机制。

从 `chapter_gemm_basics` 到 `chapter_gemm_advanced`，作者显然不是想把 GEMM 只当“矩阵乘法”，而是把它当成一条性能优化路径。

这种路径的结构在 `img/scripts/gen_gemm_perf.py` 里非常明显：

- Step 3: Sync tiled
- Step 4: TMA
- Step 7: Warp spec
- Step 8: 2-CTA
- Step 9: Multi-cons.
- cuBLAS

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/09.webp)

::: image-caption
运行时间从几十毫秒级一路降到 0.094ms 量级，说明优化不是“修修补补”，而是分阶段吞掉主要瓶颈。该图为 B200 芯片下 4096 阶 FP16 精度矩阵乘（GEMM）的分步优化耗时柱状图，纵轴采用对数刻度直观放大时延差异。基线 Step3 同步分块实现耗时高达 53.6ms，性能最差；随后逐层叠加硬件优化手段，Step4 引入 TMA 硬件访存优化，耗时骤降至 0.49ms；Step7 加入 Warp 特殊调度压缩至 0.23ms，Step8 开启双 CTA 并行降至 0.104ms；Step9 采用多生产者优化后耗时 0.094ms，与 cuBLAS 标准库性能完全持平。图表完整呈现访存、线程调度、多计算流等多层优化带来的断崖式提速，清晰展现从低效原始实现逐步逼近硬件理论峰值、追上成熟商用算子库的完整优化路径。
:::

更重要的是，这张图传达了一个非常现实的信号：**现代 GPU kernel 的性能提升，往往来自架构协同而不是单纯算子替换。**

### 5.2 从同步 tiled 到 TMA：先把搬运从主路径上移走

> `chapter_gemm_async/index.md` 讲得非常朴素：同步 tiled GEMM 的问题，不是算错，而是“轮流干活”浪费时间。一个阶段 copy，一个阶段 compute，再 copy 下一块，再 compute 下一块，看起来没错，但实际上两个完全不同的硬件单元被人为串行了。

引入 TMA 后，GEMM 开始像真正的流水线：

- 线程只负责发起；
- TMA 自己搬 tile；
- 计算阶段继续吃当前 tile；
- 下一 tile 在后台已经在路上。

这里的变化不只是“异步”二字，而是**调度责任被重新分配**。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/10.webp)

::: image-caption
TMA 同步流图
:::

原来线程做的是“搬运工”；现在线程做的是“调度员”。

### 5.3 Warp specialization：让不同 warp 只做自己擅长的事

> 从 `gen_warp_specialization_timeline.py` 的脚本可以看出，warp specialization 的核心是把不同阶段分配给不同 warp：

- 有的 warp 负责 TMA
- 有的 warp 负责 MMA
- 有的 warp 负责 writeback

这就是“特化”的真正含义：**不让所有 warp 都做同样的事情，而是让每一组 warp 成为某个阶段的专家。**

这类设计带来的好处很明显：

- 减少 warp 间无效等待
- 提高硬件单元利用率
- 便于构建更深的 pipeline

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/11.webp)

::: image-caption
这张图对比矩阵乘优化前后的硬件执行流程。上方优化前为串行执行模式，每一轮先执行 TMA 数据加载、再执行 MMA 矩阵计算，加载与计算完全交替串行，硬件资源长期闲置，空闲占比达 50%。下方是流水线优化方案，设置流水线深度为 2，拆分 TMA 加载、MMA 计算、结果写回三类任务至不同 warp 并行运行，依靠 buf0、buf1 双缓冲轮转缓存数据；通过 smem_pipe.full、smem_pipe.empty 同步信号约束，让 TMA 预取最多领先 MMA 一级，实现加载与计算阶段时间重叠。该方案打破串行壁垒，充分并发利用硬件，大幅削减闲置周期，提升算力吞吐。
:::

它也很像操作系统中的 producer-consumer 拆分：不是所有线程都在同一条队列上抢同一份活，而是按照阶段和角色明确分工。

### 5.4 2-CTA cluster：把协作扩展到 CTA 层级

> GEMM 的 advanced 章节进一步往前走到 2-CTA clusters。这个层级的意义在于，优化不再只发生在线程和 warpgroup 内部，而是扩展到 CTA 之间的协作。

这说明书中的“现代 GPU 编程”并不满足于单 CTA 的极限，而是在探索更高阶的并行组织方式。

在 AI kernel 里，这种扩展非常重要。因为随着模型规模变大、块结构更复杂，单个 CTA 的局部最优不一定等于全局最优。

2-CTA cluster 的思路就是：**把更多资源视为一个协同单元，而不是一个个孤立的执行块。**

## 六、Part IV：FlashAttention 是“把所有前面学到的东西一次性用上”

> 如果说 GEMM 是直线式优化，那 FlashAttention 就是综合题。`chapter_flash_attention/index.md` 明确指出：Attention 不是重复一个 MMA，而是 **两个 MMAs 中间夹着 softmax、causal masking、rescaling、GQA 等逻辑。**

这意味着 attention kernel 的复杂性，远高于 GEMM。它不是单纯追求吞吐，而是要在中间阶段维持数值稳定和在线更新。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/12.webp)

::: image-caption
Flash Attention 4 里两类 MMA 矩阵乘运算的输入就绪闸门同步逻辑，图例区分 SMEM 张量分片、TMEM 张量分片、MMA 运算单元与屏障同步信号。上方 Score MMA 闸门执行 Q、K 计算分数 S 的矩阵乘，依赖 SMEM 中 Q、K 分片就绪，需 q_load.full、kv_load.full 两道屏障全部完成信号同步后才启动运算。下方 Value MMA 闸门是两段式 P、V 生成输出 O 的矩阵乘，输入包含 SMEM 的 V 分片、TMEM 中 0:96 与 96:128 两段 P 列分片，分别对应 kv_load.full、p_o_rescale.full、p_ready_2.full 三道屏障；其中 kv_load.full 同时管控 K、V 分片加载，因此在两组闸门中均出现，整套机制依靠屏障信号约束张量分片加载时序，保证全部输入就绪后再触发对应 MMA 运算。
:::

### 6.1 为什么 Attention 比 GEMM 难

> 从算法结构看，FlashAttention 要计算：

$$
O = \text{softmax}(QK^T / \sqrt{d})V
$$

问题是，完整的 score matrix 太大，根本不可能全部放在片上。

所以它必须流式处理 K/V block，维护在线 softmax 的中间状态。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/13.webp)

::: image-caption
表格从多维度对比标准 GEMM 与 Flash Attention 4 的硬件执行差异：GEMM 仅单一重复 MMA 运算，运行状态、中间缓存简洁，线程分工与同步屏障较少；Flash Attention 4 包含打分、数值两轮 MMA，穿插在线 softmax、掩码等运算，需维护多类运行状态与多段张量缓存，线程分工、同步阶段更多，调度以注意力任务为最小单元，计算逻辑更复杂。
:::

这带来一个关键区别：

**GEMM 的中间值只会累加；Attention 的中间值会被重标定。**

这也是为什么书里强调 softmax rescaling 是关键难点。因为当新的 block 带来更大的最大值时，之前累积的输出就处于错误尺度，必须重新缩放。

于是，attention kernel 的执行顺序不再只是“算完一个块，再算下一个块”，而是“在块间不断更新全局统计量，同时保证已有结果仍然正确”。

### 6.2 FlashAttention 其实是在做“跨阶段协作编排”

> `img/scripts/gen_flash_attention_pipeline.py` 显示了一个很好的直觉：attention pipeline 中的关键不是某个单独步骤，而是**score MMA、softmax、value MMA、rescale、store** 之间的时序关系。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/14.webp)

::: image-caption
Flash Attention 4 的时序流水线架构，沿时间轴排布不同 warp 与工作组的并行任务，依靠多线程分组重叠执行隐藏延迟。WG3 warp1 负责 TMA 分批加载 Q、K、V 张量分片；WG3 warp0 调度 Tensor Core MMA 运算，交错执行当前 V 分片的 Value 矩阵乘与下一组 K 分片的 Score 矩阵乘，充分复用算力；WG0、WG1 分别对应两个 Q 计算阶段，串行完成 Softmax 归一并写出 P 分片；WG2 负责输出校正、缩放归一等收尾操作；最后 WG3 warp2 通过 TMA 完成输出 O 的全局存储。整套流水线将张量加载、矩阵乘、Softmax 归一、输出后处理、结果落存多阶段并行重叠，不同工作组各司其职、任务交错调度，最大化 GPU 张量核心与访存通道的硬件利用率。
:::

这张图的启示在于：

- 软计算（softmax）并不是旁支，它是 critical path；
- rescale 不是额外修补，而是 correctness 的组成部分；
- 线程/warp 之间的协作必须精确到阶段，否则就会出错。

可以说，FlashAttention 是整本书最能体现“现代 GPU 编程是系统工程”的章节。因为它把前面所有概念——TMA、TMEM、barrier、warp specialization、layout、MMA、在线数值稳定性——全部拉进了同一个执行图里。

### 6.3 Attention 的本质，是把“算”与“改写历史”同时做好

这句话听起来抽象，但其实很贴切。

> GEMM 只是把两个矩阵乘起来；Attention 则是在流式输入中，持续更新“当前最可信的归一化解释”。

所以它不仅要算，还要重写先前结果的尺度。

这意味着：

- 计算正确只是底线；
- 流水线不断推进是性能目标；
- 中间状态的数值重标定是算法约束。

这三者必须同时满足。FlashAttention 的高明之处，就在于它把看似不相容的目标统一到了一个硬件友好的实现框架里。

## 七、这本书最难得的地方：它把“工程细节”讲成了“可迁移的方法论”

很多技术资料的问题在于，它们能让人“看懂某个 kernel”，却不能让人“学会如何设计下一个 kernel”。**而这份仓库最值得称道的地方，是它不断在教读者迁移方法：**

### 7.1 先找瓶颈，再选硬件机制

比如：

- 如果瓶颈是搬运，就引入 TMA；
- 如果瓶颈是同步，就引入 mbarrier；
- 如果瓶颈是局部布局，就引入 swizzle 和 layout API；
- 如果瓶颈是流水线深度，就引入 warp specialization 或 persistent scheduling。

这是一套非常成熟的工程判断逻辑：**不是“有什么用什么”，而是“问题是什么，就选什么机制”。**

### 7.2 让布局、同步、搬运、计算各司其职

现代 GPU kernel 最怕的不是慢，而是职责混乱。

> 把本该给硬件做的事交给线程，把本该静态化的布局写成运行时分支，把本该局部化的同步写成全局 barrier，都会把性能拖垮。

这本书不断强调：

- 布局写在 layout object 里
- 同步写成 mbarrier protocol
- 搬运交给 TMA
- 计算交给 Tensor Core
- 调度交给 warpgroup / CTA / cluster

这其实是在帮读者建立一种非常“编译器化”的思维：**把程序拆成不同职责层，再把每层映射到最合适的硬件单元。**

### 7.3 把“正确性”当作性能优化的一部分

这一点在 FlashAttention 尤其明显。

> 很多人做优化时默认“先快再对”，但这本书展示的是另一种方式：**只有正确地表达异步协议，性能优化才真正成立。**

因为一旦 barrier、phase、rescale、布局有一个地方错了，速度再快也只是快错。

所以这里的“性能工程”不是和“正确性”对立，而是建立在正确性协议之上。

## 八、为什么这本书值得 AI 系统工程师认真读

如果你的工作是大模型推理、训练系统、编译器优化或 GPU kernel 开发，那么这本书的价值会远超过“一个教程”：

### 8.1 它帮你理解为什么性能瓶颈总是出现在少数关键算子

大模型系统里，真正吞噬时间的，往往不是 Python 逻辑，而是那几个核心 kernel。

本书把 GEMM 和 Attention 作为主线，本质上就是在告诉你：**系统性能的下限，常常由这类 kernel 决定。**

### 8.2 它帮你建立跨层心智模型

> 一名优秀的 AI 系统工程师，不只看 API，也要能理解：

- 这一层的数据怎么排布
- 为什么这个 barrier 必须存在
- 这个 TMA 什么时候发起
- 哪个 warp 在做什么
- 为什么 rescale 会影响后续写回

这本书恰恰在训练这种跨层能力。它不是让你记住“某个参数怎么填”，而是让你理解“每个参数为什么存在”。

### 8.3 它把 GPU 编程从“经验手艺”提升为“可推导工程”

这是整本书最核心的意义。

> 当你学完这本书之后，你不应该只是会复述 TMA、TMEM、mbarrier、warp specialization 这些名词，而应该能够像编译器设计者一样回答问题：

- 为什么这个 tile 应该 swizzle？
- 为什么这里需要 phase tracking？
- 为什么这条 pipeline 可以 overlap？
- 为什么这个 kernel 应该 persistent？
- 为什么 GEMM 和 FlashAttention 的优化路径会相似，但又不能完全照搬？

能回答这些问题，才说明你真正进入了现代 GPU 编程的语境。

## 结语：现代 GPU 编程的终点，不是更快，而是更懂

> `mlc-ai/modern-gpu-programming-for-mlsys` 最打动人的地方，不是它展示了多少高性能技巧，而是它把现代 GPU 的“复杂”讲得非常诚实。它没有把 Blackwell 时代包装成一个可以靠口诀通关的世界，而是承认：**想把现代 GPU 用好，你必须理解硬件分工、布局语义、异步协议和调度策略之间的整体关系。**

这也正是它的价值所在。

::: block-1
它让读者看到，真正的高性能不是某个神奇指令带来的，而是从数据布局、搬运机制、同步协议、计算调度到算法重写的一整套系统设计。
:::

如果把传统 GPU 编程比作“在单车道上开快车”，那么这本书展示的，是 Blackwell 时代如何把车道、信号灯、引擎、变速箱和导航系统一起重构。

> 最后你会发现，所谓“现代 GPU 编程”，并不是更复杂地写代码，而是更准确地理解：**代码到底应该把哪件事交给哪块硬件来做。**

- **Modern GPU Programming For MLSys**
- https://github.com/mlc-ai/modern-gpu-programming-for-mlsys
- 📖Read it online: https://mlc.ai/modern-gpu-programming-for-mlsys/

::: related-title
**相关推荐**
:::

::: related-links
- **[暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)**
- **[NVIDIA 开源 GPU 编程教学仓库 accelerated-computing-hub，从单线程 kernel 到协作归约](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900741&idx=1&sn=8a88431b7a8d1d46aa3b03f9889bc151&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::
