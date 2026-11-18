关键词：LLM 编译、:mdnice-bold-italic[e-graph]、**异构存储架构**、统一分布式编译、自动优化、**端到端编译框架**

- 本文转载自知乎账号：郑启航[1]
- https://zhuanlan.zhihu.com/p/1989088940733510928

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/01.webp)

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/02.webp)

- **nncase: An End-to-End Compiler for Efficient LLM Deployment on Heterogeneous Storage Architectures**
- https://arxiv.org/pdf/2512.21571
- https://github.com/kendryte/nncase
- 技术交流 QQ 群：790699378 答案：人工智能
- 5000 字，阅读 18 分钟，**播客 52 分钟**

相关推荐

- [性能直逼FA3/4！斯坦福NVIDIA提出Tensor Core GPU最优流水线和Warp Sepcialization自动调度策略Twill，跨Hopper/Blackwell适配！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895214&idx=1&sn=8ab4120451fcfdff70e3900acab5469c&scene=21#wechat_redirect)
- [平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)
- [PipeThreader：软件定义流水线驱动的 DNN 编译器，FlashAttention与Mamba2性能显著超越现有方案！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889807&idx=1&sn=a8f2bfdca2091144fcd6059a575dda66&scene=21#wechat_redirect)

> 很高兴为大家介绍我们的工作： nncase, 一个支持 LLM 多存储架构端到端部署的 AI 编译器。

具体而言，nncase 是一个支持动态 shape、自动向量化布局、自动分布式切分、自动多内存层级 Fusion Tiling 的 AI 编译器，可以**将 LLM 模型端到端编译为 Single Kernel 执行**，达到超越推理框架的性能。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/03.webp)

::: image-caption
图1：nncase 架构概览，展示了 nncase 编译器的端到端编译流程，通过五个核心阶段连接高层模型定义与底层硬件原语，实现异构存储架构下 LLM 的高效部署：先通过基于 e-graph 的优化阶段（含自动向量化、自动分布式策略搜索）解决布局冲突与阶段排序问题；再经自动调度阶段（结合 MCTS 与 MINLP 优化循环结构和参数）提升片上缓存复用率；最后通过代码生成阶段，依托缓冲调度与 NTT 库实例化高效内核，最终实现 “一次编译，多端适配”，为异构硬件下 LLM 部署提供统一优化路径
:::

我们设计 nncase 的动机在于：

::: block-1
1. 作为一个工具链团队，是需要**向后向前兼容各种硬件**。
2. 同时目前模型以及硬件 scaling 已成事实，**编译器自然也需要具备 scaling 的能力** 来支持不同场景/不同规格硬件的编译优化，因为我们相信不远的将来依旧会出现组合爆炸的问题，这是人力难以 handle 的。
3. 考虑到多 chip/多 die，UMA/NUMA 等不同的组合的硬件架构，是以计算资源为粒度，以分布式视角将计算图进行拆分，并针对不同层级的存储/互联做相应的编译优化。
:::

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/04.webp)

::: image-caption
图9：单核心场景（1T）下不同框架的LLM token 生成吞吐量对比。其中 llama.cpp 代表手动优化基准，其余框架代表基于编译器的实现方案。基于 AMD Ryzen 9 5900X 单线程实验，清晰呈现性能层级：llama.cpp 因手动优化内核领先，nncase 紧随其后，远超 Intel IPEX 与 MLC LLM。具体来看，Qwen3-0.6B-F32 配置下，nncase 达 8.7 tokens/s，较 Intel IPEX 高 15%；F16 精度时提升至 13.87 tokens/s；Qwen3-1.7B-F16 仍保持 5.09 tokens/s，这验证了其单核心下自动向量化与调度模块的有效性，虽略逊手动优化，但在编译器方案中表现最优
:::

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/04.webp)

::: image-caption
图9：单核心场景（1T）下不同框架的LLM token 生成吞吐量对比。其中 llama.cpp 代表手动优化基准，其余框架代表基于编译器的实现方案。基于 AMD Ryzen 9 5900X 单线程实验，清晰呈现性能层级：llama.cpp 因手动优化内核领先，nncase 紧随其后，远超 Intel IPEX 与 MLC LLM。具体来看，Qwen3-0.6B-F32 配置下，nncase 达 8.7 tokens/s，较 Intel IPEX 高 15%；F16 精度时提升至 13.87 tokens/s；Qwen3-1.7B-F16 仍保持 5.09 tokens/s，这验证了其单核心下自动向量化与调度模块的有效性，虽略逊手动优化，但在编译器方案中表现最优
:::

## 本文目录

- 一、背景与动机
  - 1.1 数据布局优化
  - 1.2 分布式策略搜索
  - 1.3 kernel 调度优化
  - 1.4 核心观察和解决方案
- 二、nncase 编译器架构
  - 2.1 nncase 架构概览
  - 2.2 Equality Saturation
  - 2.3 Auto Vectorize
  - 2.4 Auto Distribution
  - 2.5 Auto Scheduling
  - 2.6 Code Generation
- 三、性能实验评估
- 总结与展望

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/05.webp)

## 一、背景与动机

### 1.1 数据布局优化

目前通用的计算架构都会存在 tensor/vector 级别的计算单元：

- 传统做法会采用在单个 kernel 内部进行 vectorize/tensorize，但是这样**对于单个算子的调度要求较高**，类似 BLIS 需要进行 online packing。并且也**错失了多个 kernel 存在数据布局复用的情况**。
- 也有一些类似 TPP MLIR 的方案，尝试以 tensor 为粒度将数据布局重写，但是他并**没有考虑矩阵单元和向量单元数据布局的转换的开销。**

### 1.2 分布式策略搜索

分布式策略搜索是充分释放芯片并行算力的关键，但是目前大部分的工作都是都是在 high level 的框架中完成，**没有在一个编译器中考虑上下游的数据布局/算子调度来同步完成。**

### 1.3 kernel 调度优化

> :mdnice-bold-italic[调度优化直接决定了算力的有效率，但是不同硬件平台在指令集、片上缓存结构、计算单元架构等方面的固有差异]，使得“单一调度策略适配全硬件平台”的目标难以达成。

目前有的工作基本分为两类：

::: block-1
- **基于学习的调度方法** （looper/ansor/meta scheduler），这种方法通常需要真实硬件并且学习时间过长不太适合快速部署；
- **基于分析建模的方法** （tuna/chimera），但是他们目前没办法支持复杂的 fused 算子求解，并且有的工作是基于标量为最小粒度，很难实现寄存器级别的优化，因此距离手工优化水平还差的较远。
:::

### 1.4 核心观察和解决方案

1. :mdnice-bold-italic[数据布局优化本质上需要从全局考虑数据转换的开销]，但这样会遇到 phase ordering 问题，错失更优的数据布局优化方案。 因此 nncase 实现了**基于 e-graph 的 pattern rewrite system，把所有可行布局存入 egraph，基于 roofline model 来评估 cost**，最终提取出最优化的数据布局方案。
2. :mdnice-bold-italic[分布式策略的本质也是一个路径提取问题，nncase 通过自定义 e-graph 的构建过程，将其纳入到图优化的范畴]，复用计算/通信 cost model 以及 extractor 来进行最优解提取。并且可以结合一些 memory constraint 来保证不会出现 OOM 的问题，以及配合硬件架构做内存分配上的优化， 比如 UMA 架构下可以实现逻辑 Broadcast 而物理上只存储一份、逻辑 Split 而物理紧凑排列等。
3. :mdnice-bold-italic[调度优化核心问题是调度粒度过细以及搜索时间过长，nncase 提出了 nncase tensor template library(NTT library)，基于手写的 micro kernel 向编译器隐藏寄存器级别的优化复杂性]。并且以 µkernel 所形成的 tile 作为基本调度单元，借鉴 welder/tileflow 的思想，**将搜索空间区分为两个维度：structural part 以及 parametric part**。后面采用蒙特卡洛树搜索（MCTS）求解结构层优化问题，借助解析建模完成参数层优化，既保障寄存器级优化性能，又规避了传统方法搜索时间过长的缺陷。

## 二、nncase 编译器架构

### 2.1 nncase 架构概览

> nncase 的工作流程如下图所示，采用基于 e-graph 的 rewrite system 来解决各类图优化问题。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/03.webp)

::: image-caption
图1：nncase 架构概览，展示了 nncase 编译器的端到端编译流程，通过五个核心阶段连接高层模型定义与底层硬件原语，实现异构存储架构下 LLM 的高效部署：先通过基于 e-graph 的优化阶段（含自动向量化、自动分布式策略搜索）解决布局冲突与阶段排序问题；再经自动调度阶段（结合 MCTS 与 MINLP 优化循环结构和参数）提升片上缓存复用率；最后通过代码生成阶段，依托缓冲调度与 NTT 库实例化高效内核，最终实现 “一次编译，多端适配”，为异构硬件下 LLM 部署提供统一优化路径
:::

输入计算图以及硬件配置，在图优化中进行自动向量化/自动分布式， 而后将支持 affine 的表示的 operation 转换为携带 affine map 信息的 tiered tile graph 上。

在 auto schedule 中先用 MCTS 自由组合各种 tile graph 的结构以及其 affine map 属性，为每一个具体的 tile graph 采用 MINLP 建模求解他的 tile size/buffer placement 和 total time，这样动态的把 fusion/tiling/buffer planing 结合到一起。

然后把最优的 tile grpah 以及参数生成为 c++代码，通过 c++去实例化手写的 NTT library，最终使用传统编译器得到一个可执行文件。

::: block-1
**nncase 编译器大量使用求解器作为优化手段**：

- 比如 e-graph 的 extractor 实际上是一个 WPMAXSAT 求解
- auto schedule 中采用 MINLP 求解
- buffer schedule 转化为 bin packing 问题采用 CP-SAT 求解。

**总之，Make OR Great Again！**
:::

### 2.2 Equality Saturation

> 这里先解释一下为什么需要使用 e-graph 作为 rewrite system。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/06.webp)

::: image-caption
图2：A comparison of traditional term rewriting and e-graph based rewriting. 图中：(a) 为待优化的计算图，目标是通过后文 表1 中的转置优化规则消除冗余操作。传统重写（b、c）受阶段排序影响：贪心选择先应用 CombineBinaryRightTrans 会导致次优结果（c，残留冗余转置），而正确顺序（先 CombineBinaryLeftTrans 再 CombineUnaryTrans）才能全消除冗余（b）；基于 e-graph 的重写（d-f）通过等式饱和保留所有等价节点（d），结合 Roofline 成本模型与 WPMAXSAT 求解，提取出全局最优子图（f），彻底规避传统方法的局部最优陷阱
:::

考虑我们有左上的计算图，以及如下的优化 rule。我们就会发现 rule 的执行顺序影响了最终优化的结果， 考虑 b 中的情况，先向下移动左边的 transpose，会在右边以及下面插入新的 transpose，运气比较好，左边和下面可以和原始的 transpose 做抵消，最终得到只剩一个 transpose 的结果。

然而，如果先向下移动右边的 transpose，那么很不幸，下面以及左边的与新插入的 transpose 只能做合并，最终还剩两个 transpose，得到了次优化的结果。

那么 e-graph 则是通过 Equality Saturation 来解决上述问题，通过重复的 apply rewrite rule，将整个 graph 填满可能的节点，最终通过 cost model + extractor 来拿到最优的计算图。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/07.webp)

::: image-caption
表1：转置优化相关重写规则。该表的规则为 nncase 解决转置优化的核心规则集，为“等式饱和” 机制提供关键支撑。例如，FoldNopTrans 可直接剔除计算图中无实际作用的多重转置操作，避免冗余数据移动；FoldTwoTrans 则能融合连续转置变换，减少中间计算步骤。这些规则并非按固定顺序执行，而是在 e-graph 中同时生效，保留所有等价转置优化路径，有效规避传统编译器因 “阶段排序问题” 导致的冗余转置残留，为后续通过 Roofline 成本模型提取全局最优子图提供规则基础
:::

> 关于 e-graph 的更多内容可以参考这些文章：

::: block-1
Term Rewriting, Equality Saturation and Deep Learning Compilers[2]

Equality Saturation 优化在 AI 编译器中遇到的挑战[3]

Glenside : 如何自动发现 im2col 布局转换?[4]
:::

### 2.3 Auto Vectorize

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/08.webp)

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/08.webp)

::: image-caption
e-graph based auto vectorize for the Attention-like subgraph
:::

基于 e-grpah，其实完成自动向量化布局并不是难题，主要是插入各种 pack op/ packed op/ unpack op， 然后配合 fold 进行优化， 当然实现起来还是有许多细节问题要处理， 最麻烦的其实还是在 µkernel 中支持各类 packed op。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/09.webp)

::: image-caption
表2：向量化与布局优化相关重写规则。MetaPackOperation 通过生成多硬件布局变体（如 Tensor Core 的分块布局、Vector Unit 的扁平布局），在 e-graph 中保留所有可行的打包方案；FoldNopPack 则剔除 “Pack 后立即 Unpack” 的冗余操作。二者配合使 Attention 子图等场景中，数据能以最优硬件格式贯穿计算链，避免反复布局转换的内存开销，为 Roofline 成本模型筛选全局最优布局提供基础
:::

### 2.4 Auto Distribution

自动分布式搜索我们是基于 oneflow 的 SBP 抽象来实现的，SBP 的具体表示大家可以参考 OneFlow 源码解析：Eager 模式下的 SBP Signature 推导[5]， 配合下面这张图一起看更加直接：

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/10.webp)

::: image-caption
图4：SBP 抽象与设备布局示意图。(a) 展示 1D 设备布局下 [2,2] 张量的三种 SBP（Split、Broadcast、Partial）策略实例；(b) 呈现 2D 嵌套拓扑（两组设备，每组含两个物理设备）及对应的 2D SBP 配置，直观体现 SBP 对不同拓扑下张量分布式状态的表征能力
:::

在接入 SBP 表示的时候， 我发现实际上 SBP 表示相同的节点是等价节点， 基于这一点， 那么就可以在 e-graph 中构建 Boxing Sharding -> Computation -> Boxing UnSharding 的节点，并通过 fold Sharding/UnSharding 来进行 SBP 的传播。 这样我们又复用了 e-graph 的 rewrite system，并在其中为 boxing 添加了基于 alpha beta 的 cost model 和计算操作做统一的 cost 评估。下图是一个简单的矩阵乘分布式切分搜索逻辑：

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/11.webp)

::: image-caption
Visualization of the e-graph for distributed strategy search
:::

### 2.5 Auto Scheduling

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/12.webp)

::: image-caption
图7：计算内核的设计空间，拆分为结构维度与参数维度。左侧展示结构维度（y 轴）的层级化 Tile 图，含算子融合逻辑；右侧及下方呈现参数维度（x 轴）的配置差异，对比不同分块大小与缓冲区位置对数据局部性的影响，绿色框为优化配置，红色框为低效配置
:::

这是我们所设计的，以 tile 为粒度的 design space ，纵轴是结构部分，横轴是参数部分。从结构部分采样到的 tile graph 决定了具体的 fusion 方式以及循环顺序， 而后参数部分采样到的则是这个 tile grpah 的 tile size 以及 buffer location。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/13.webp)

这是一个具体的调度流程，我们将一个标准的计算图 lower 到 affine 的形式，然后在此基础上结合硬件提供的参数配置，将单个的 affine operation 转换为嵌套的 tile grpah 的表示， 实际上等同于做 symbolic 的 post-tiling fusion（和赵捷老师的思路类似 AKG: 使用 post-tiling fusion 策略完成无副作用的内存优化[6]），在 tile graph 的结构上借鉴了部分 welder/tile flow 的思路。

与他们不同的是，我们采用 MCTS 配合多面体分析来搜索各种程序组织形式，fusion 形式则是包含在其中， 关于多面体分析的内容可以参考我的文章：[Hands-On Polyhedral] 动手多面体编译[7]

在 MCTS 搜索中，搜索树的一个节点对应 TileGraph，然后再对这个 TileGraph 做树遍历来进行 MINLP 建模。 经过 MCTS 的迭代过程，很好地解决了早期无法评估 ABC 还是 BCD 做 fusion 的问题：

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/14.webp)

对具体的 tile graph 进行 MINLP 建模是一个非常复杂的过程，具体细节可以参考我们的论文。**这里也是我们和其他工作不一样的地方，我们用离散变量支持了 buffer location 的搜索，并且解耦和存储和计算的层级** （:mdnice-bold-italic[对应 halide 里面的 compute at/store at 调度]）。然后用公式评估了内存足迹、内存复用、 µkernel 计算开销，最终一次求解直接得到 tilesize 以及在各个层级上的 buffer 大小以及 offset。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/15.webp)

### 2.6 Code Generation

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/16.webp)

::: image-caption
图8：利用 NNCase 张量模板库（NTT，Tensor Template Library）原语生成的 C++ 代码。代码通过`ntt::fixed_shape_v`等静态维度接口，将编译期可知的张量形状（如模型权重维度）提前解析为立即值，消除运行时形状计算开销；同时调用`ntt::vector`等显式向量类型，精准匹配Auto Vectorize模块确定的SIMD指令需求。此外，代码隐式集成NTTD（分布式扩展）的拓扑感知逻辑，可直接对接Auto Distribution模块输出的SBP策略，确保生成的代码在单核心或多核心场景下，均能高效复用片上缓存、减少数据移动，成为连接前端优化与硬件执行的关键纽带
:::

上图是代码生成的结果，在生成之前我们会有 buffer alias 分析以及结合 SBP 属性进行 memory planing。

此外非常值得一提的是我们的 NTT library，是 @SunnyCase[8] 亲自操刀的设计，支持 fixed/dynamic shape 混合, 以及 type based sharding, 除了高效以外，编写分布式 kernel 还可以减少许多心智负担。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/17.webp)

::: image-caption
summa example by NTTD
:::

上图是一个 summa 的例子，summa是一种分布式matmul算法，见《分布式存储架构下的矩阵乘与编译器》[9]。其基于 sharding 的类型信息，通过 remote 函数配合 runtime 提供的能力可以很方便取得对等存储上的 tensor。

## 三、性能实验评估

> 我们在 x86 上测试了无分布式情况，:mdnice-bold-italic[通过自动向量化+自动调度，我们拿到了 llama.cpp 大约 80%的性能，超过其他的编译器。]

这里 MLC LLM 性能较弱，主要是他们在 meta schedule 实际上没办法在 LLM 场景下用起来，是使用的 DLight （一套预先设定的调度模板）。

但是 :mdnice-bold-italic[nncase 则是使用多面体编译技术，处理了动态 shape 问题，把自动调度的结果和 µkernel 组合在一起，尽可能的达到自动调度所搜索到的性能。]

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/04.webp)

::: image-caption
图9：单核心场景（1T）下不同框架的LLM token 生成吞吐量对比。其中 llama.cpp 代表手动优化基准，其余框架代表基于编译器的实现方案。基于 AMD Ryzen 9 5900X 单线程实验，清晰呈现性能层级：llama.cpp 因手动优化内核领先，nncase 紧随其后，远超 Intel IPEX 与 MLC LLM。具体来看，Qwen3-0.6B-F32 配置下，nncase 达 8.7 tokens/s，较 Intel IPEX 高 15%；F16 精度时提升至 13.87 tokens/s；Qwen3-1.7B-F16 仍保持 5.09 tokens/s，这验证了其单核心下自动向量化与调度模块的有效性，虽略逊手动优化，但在编译器方案中表现最优
:::

> 而后我们测试了 4/8 核的情况，nncase 则是略微超越了 llama.cpp。

主要原因其实是 llama.cpp 采用的是基于 shared memory 的并行加速方案，但是这种方案实际上每个算子结束都需要进行一次同步。

而 **nncase 则是以原生分布式的视角，尽量减少同步/通信开销为准则进行分布式搜索，实际上的同步次数是明显少于 llama.cpp 的**， 所以才可以在自动生成算子性能不如手写的情况下做到媲美他的性能。

![文章插图](/articles/2025-12-31-llm-nncase-e-graph-2b7e3dce/images/18.webp)

::: image-caption
图10：多核心场景（4T/8T）下LLM token 生成吞吐量对比。基于 AMD Ryzen 9 5900X 平台实验，nncase 展现出卓越的扩展性，性能超过手动优化的库。Qwen3-0.6B-F16 配置下，4T 时 nncase（23.5 tokens/s）略超 llama.cpp（23.2 tokens/s），8T 仍保持领先；Qwen3-1.7B-F16 的 4T 性能较 1T 提升 74%，远超 llama.cpp 的 32%。这得益于其静态任务划分（规避 OpenMP 动态调度开销）与核心级分布式粒度优化，突破传统共享内存并行的效率瓶颈
:::

## 总结与展望

::: block-1
nncase 解决了 dynamic shape、auto vectorize、auto distribution、auto schedule 的问题，**以 24 contributors + 编译器达到了 1396 contributors 手写所取得的性能**，是一个还不错的成果。
:::

但是目前还有许多问题没办法解决，比如 computation-communication overlap、flash/paged attention，这些都是最棘手的问题。

通算融合之前做了一些探索，但还不怎么完善 [Hands-On Polyhedral] 自制通算融合 DSL Part1[10]。

> 这个项目因为不可抗力的原因的停止了，但也给了我们机会去探索新的方向。

**目前 graph compiler 失败的主要原因还是过度依赖自动调度**，目前的一些优化，一部分无法被表示，另一部分虽然可以通过 schedule 到达，但是这个过程依旧很漫长，所以 kernel compiler 目前更加火热。

:mdnice-bold-italic[graph compiler 需要把中间表示打开，提供给用户操作的接口，但 tvm.te 这种表现力还是不足，tile based IR 就是一个很好的入口]，可以让用户手工编写提供调度，编译器管理资源，最后嵌入到 graph 中做整体优化，形成编算协同。

参考资料

[1] 郑启航: *https://www.zhihu.com/people/28-18-94-5*

[2] Term Rewriting, Equality Saturation and Deep Learning Compilers: *https://zhuanlan.zhihu.com/p/437972991*

[3] Equality Saturation 优化在 AI 编译器中遇到的挑战: *https://zhuanlan.zhihu.com/p/605459519*

[4] Glenside : 如何自动发现im2col布局转换?: *https://zhuanlan.zhihu.com/p/456616977*

[5] OneFlow源码解析：Eager模式下的SBP Signature推导: *https://zhuanlan.zhihu.com/p/605108910*

[6] AKG: 使用post-tiling fusion策略完成无副作用的内存优化 : *https://zhuanlan.zhihu.com/p/535606722*

[7] [Hands-On Polyhedral] 动手多面体编译: *https://zhuanlan.zhihu.com/p/21062125487*

[8] SunnyCase: *https://www.zhihu.com/people/01688e8d818824d2d21b48434f74c7cc*

[9] 分布式存储架构下的矩阵乘与编译器: *https://zhuanlan.zhihu.com/p/6906641426*

[10] [Hands-On Polyhedral] 自制通算融合DSL Part1: *https://zhuanlan.zhihu.com/p/1954332062753493221*
