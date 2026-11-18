关键词：**TileLang**、:mdnice-bold-italic[DeepSeek-V4]、**DSL**、大模型、小算子

> 最近 V4 的技术报告讨论度很高，其中 TileLang 在报告中也占了一部分内容，主要是总结了社区最近的一些技术进展、以及其**投入实际工业界应用** 后的一些打磨经验。

这篇文章带大家简单解读一下技术报告里 TileLang 部分，同时分享一些关于编译器 DSL 在真正的模型 Infra 中定位的思考。

![文章插图](/articles/2026-05-12-deepseek-v4-tilelang-eb0dcd25/images/01.webp)

- **DeepSeek-V4 技术报告背后的 TileLang：如何高效实现大模型中的小算子**
- 作者：SiriusNEO，https://zhuanlan.zhihu.com/p/2033034202720022871
- TileLang 本体：https://github.com/tile-ai/tilelang
- :mdnice-bold-italic[由 TileLang 实现的高效 LLM 小算子：https://github.com/deepseek-ai/TileKernels]
- 3000 字，阅读 16 分钟，**播客 7 分钟**

相关推荐

- [DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)
- [最高砍掉90%计算成本！揭开DeepSeek-V4超长上下文的高效密码与未解难题](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900326&idx=1&sn=e82f732df8909fb737de3a25027bc5e4&scene=21#wechat_redirect)
- [DeepGEMM #304浅析：Mega MoE、FP4 Indexer 与全面架构升级](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900340&idx=1&sn=aec2362e63f9118347f000b30a428114&scene=21#wechat_redirect)

::: block-1
开头附上 TileLang 本体的项目位置，以及最近DeepSeek 开源的 TileKernels，一系列由 TileLang 实现的高效 LLM 小算子。
:::

## 本文目录

- DSL v.s. 专家手写
- TileLang 的设计精髓
- V4 设计报告里的几个点
- 总结一下
- 读者评论

![文章插图](/articles/2026-05-12-deepseek-v4-tilelang-eb0dcd25/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## DSL v.s. 专家手写

> 随着基模架构的收敛，模型中的计算变得固定，因此现在的 Infra 基本选择手写 kernel 来获得更高的性能上限，由此似乎 DSL/编译器看起来重要性越来越低了。现在甚至更极端地有着 **MegaKernel** 这样的概念，选择发挥手写调度的极致。在这个大背景下，DSL 的定位看起来似乎就比较尴尬。

从 TileKernels 开源里可以看到，其中的算子以不带 Tensor Core 的操作算子为主，包括 elemenwise+reduce 的组合、type cast 与 indexing 等。同时在技术报告中也提到，TileLang 在 V4 Infra 中通过用一系列 fused kernel 来取代模型中的细粒度小算子。**在 Infra 上，形成了 DSL 负责零碎小算子，Attention、GEMM 采用专家定制优化的格局**，个人觉得这个 Infra 整体结构还是很健康的。:mdnice-bold-italic[这里自然也有一个问题：为什么这部分零碎算子的 fused kernel 不手写？当然，手写也是可以的。但是 DSL 在这类算子的优势就比较明显。]

1. **开发优势：** 通常我们对 DSL 的认知是开发成本与性能的 trade-off，但对这类小算子，几乎就没有什么 trade-off。主要是由于这类 mem-bound 算子大多不需要 Tensor Core，像各种 Warp Specialization 的问题也就更少了，因此 DSL 的性能基本可以顶到手写的上限。在这个前提下，使用 DSL 开发确实是快得多。一来一回，性能无损加上开发快，自然优势大。
2. **维护与迁移优势**：除了开发上的优势，DSL 在维护上的心智负担也很低，用 DSL 维护的算子库会更加简洁，如果锅是编译器的甚至不需要改动算子库本身，同时对硬件的依赖也较低，在不同后端间迁移也更加愉快。

同时对于带 Tensor Core 的算子，TileLang 也能做不错的开发，典型例子比如TileLang 仅用 80 行 Python 代码就实现了 FlashMLA 95%的性能[1]，以及最近 Qwen 团队开源的FlashQLA[2]，使用 TileLang 实现的 **GDN** 在特定场景下达到了超越 **FlashInfer** 的性能。个人认为在这方面，DSL 的意义主要是

1. **学术 Idea 的快速验证**，即用少量的开发成本换取 80%-90% 的性能，对于学术界尤其是各种魔改 Attention 的工作还是很不错的。
2. 同时还能作为 dataflow 的建模参考，放在 examples 里大家也更好**阅读学习**。
3. 此外，TileLang 的编译栈可以生成 source 级源码（例如 CUDA 后端的`.cu`，作为对比 Triton 是直接 MLIR lower 到 PTX），因此也可以幻想利用 DSL 打模板，**生成源码后继续专家优化** 的场景。
4. 对于未来的**Agent**，DSL 会不会比源码更好读？这点暂时无法确认，目前 AI 写 CUDA 水平还是比较高，主要还是语料充足，像 TileLang 这样的自制语言语料太少了。不过目前基于 SKILL 写不复杂的 kernel 体验上也不错，对比 zero-shot 写 CUDA 甚至体验会略好一些。后续如果能收集足够语料的话，TileLang 提供的抽象可以让 AI 只关心数据流，不用学复杂的 Layout 怎么写，感觉理论上还是有利于 Agent 的。

## TileLang 的设计精髓

> 讲完了关于 DSL 的一些思考，这一小节我简单分享一些我对于 TileLang 设计亮点的理解。对于 TileLang 相比 Triton 的直接区别，第一印象可能是显式暴露内存层级（L0、L1、L2）。不过个人认为 TileLang 最精髓的部分还是`Fragment`和`Parallel`抽象。

`Fragment`是对同一个 block 中所有线程的寄存器（registers）构成的集合体的抽象。 写过 CUDA 的同学们都知道，CUDA 的 SIMT 编程模型分为 block、warp、thread 等层级，对于一个并行计算任务我们通常是要在这些层级上拆之又拆。往往将任务拆分到不同的 block 上是简单的，但是对于任务到 warp、线程的映射，这一问题就变得比较复杂，尤其是当 MMA 这种带有严格 Layout 要求的指令出现时。

::: block-1
TileLang 的想法就是利用`Fragment`抽象让用户把一个 block 内的任务当成一个整体来考虑，而`Fragment`的具体实现——即逻辑上的`A[i,j]`具体映射到哪个物理寄存器——则借由编译器的自动推导实现。同时相对应地引入`Fragment`上的迭代，即`Parallel`抽象。
:::

它背后当然就是不同的线程，对于`A[i,j]`这样的访问实际就是对应线程对自己本地寄存器的访问。**与 Triton 通常使用 Tile-Level 的 micro-op（`tl.op` ) 相比，拥有`Fragment`与`Parallel`加持的 TileLang 支持更加细粒度、精确到 element-wise（`A[i, j]`）的操作逻辑。** 这样用户编程上的体感就是这样的：

```
# 节选自：https://github.com/tile-ai/tilelang-puzzles/blob/main/ans/06-softmax.py# online softmax 实现的片段log2_e = 1.44269504A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)cur_max_A = T.alloc_fragment([BLOCK_N], dtype)cur_exp_A = T.alloc_fragment([BLOCK_N, BLOCK_M], dtype)for i, j in T.Parallel(BLOCK_N, BLOCK_M):    cur_exp_A[i, j] = T.exp2(A_local[i, j] * log2_e - cur_max_A[i] * log2_e)
```

对寄存器做了这种抽象后，我们发现原来的多层编程模型有了一种优美的一致性——**所有东西都是 Tile**，shared memory、registers 不过是不同内存层级的 Tile，我们只需要关心不同层级的 Tile 搬运以及 Tile 的计算。

![文章插图](/articles/2026-05-12-deepseek-v4-tilelang-eb0dcd25/images/03.webp)

::: image-caption
基于统一切块（tile）抽象的四个核心 AI 编译技术。 《微软亚洲研究院推出AI编译器界“工业重金属四部曲”》提到 Rammer、Roller、Welder、Grinder 四大系统，均基于统一切块抽象，相关论文收录于 OSDI 大会。四款工具分别针对性提升硬件并行利用率、压缩编译耗时、优化全局内存访存、高效支持控制流执行，在多硬件平台实现显著性能加速，已落地微软多款产品。文章还展望了大模型时代 AI 编译器在硬件适配、架构定制等领域的发展前景。
:::

这种抽象个人感觉是MSRA 的深度学习编译器四部曲[3]积累总结而来，最后在 TileLang 中呈现的精华形态。不过限于篇幅笔者只能浅谈，感兴趣的读者可以前往TileLang 在 ICLR'26 的文章[4]细读。

## V4 设计报告里的几个点

> 最后带大家读一下 V4 报告里的 TileLang 部分的几个点。

首先是**Host CodeGen**，这部分主要是 TileLang 通过利用TVM-FFI[5]极大减少了主机侧的开销，包括 kernel launch、一些 Tensor 的检查等。做法是把这部分开销从 Python 侧移动到 C++ 侧，作为 kernel 的 host 侧逻辑一起被编译进去。:mdnice-bold-italic[实际收益还是很明显的，也要感谢伟大的 TVM 社区做出 TVM-FFI 这样的高质量基建。]

第二点是**Z3 Prover** 的集成。之前 TileLang 使用的代数系统其实是 tvm 自带的一套 arith，这玩意证明能力还是比较弱的，有不少简单的条件它是没法证出来的。什么，:mdnice-bold-italic[你问我为什么需要编译器里需要这种证明工具？这是因为 Tensor Program 里本来就自带很多 index 计算，最简单的需求例如边界判断：]

```
if i < N:    A_local[i] = 1 # Buffer write guarded by conditon
```

TileLang 是会自动帮你插入这些边界判断的 if 的，不过如果程序上下文信息能够“证明”这个 i 一定不会越界，即`i < N`恒真，则我们可以不插这个 if，这里就要用到 Prover 来做这件事。当然还有更多的例子，例如向量化等。这些问题本质上都能被转化成**整数表达式** 的证明。集成完 Z3 后我们也杯具的发现 bug 更多了——因为原来并不是没有 bug，而是证明器太弱了导致采用了相对保守的做法。

第三点是**精度与 bitwise 对齐** 方面，这方面包括后面许多算子的 batch invariant 性质对像 RL 这种高精度需求的场景是非常重要的，这一点也是只有把编译器投入真实的工业界使用中才会发现的问题。这部分 TileLang 主要是做了很多小活来保证这件事，包括默认关 fast-math、采用 IEEE 的 intrin，同时与 NVCC 这种主流编译器对齐代数变形的逻辑。这里有很多的规则怪谈，例如 NVCC 做各种神秘的 fma fuse 导致一些看起来相等的表达式其实并不 bit-identical。

## 总结一下

TileLang 在 V4 Infra 里的作用还是很足的，也作为一个例子让我们思考在如今的时代，DSL 需要是一个什么定位。总之欢迎大家试用学习：目前官方文档[6]、TileLang-Puzzles[7]、TileOPs[8] 以及XPUOJ[9] 都提供了大量的资料、例子以及评测方法。相信你会 get 到其中的一些美妙之处的！

## 读者评论

![文章插图](/articles/2026-05-12-deepseek-v4-tilelang-eb0dcd25/images/04.webp)

![文章插图](/articles/2026-05-12-deepseek-v4-tilelang-eb0dcd25/images/05.webp)

::: image-caption
读者评论中有人提到 TileLang 借 DeepSeek-V4 热度近半年迭代进步显著，但仍沿用老旧 CK、Cutlass3.x 设计范式，内存层级上缺少 Hopper 架构必备的 TMA 描述，期待官方后续补齐支持；也有读者感慨文章入门内容浅显易懂，容易产生自己也能上手的错觉，可读到 V4 三大核心技术细节时又顿感理解门槛陡增；同时还有人询问是否有 CuteDSL 相关解析博客，也十分关注 TileLang 对昇腾硬件的适配进展，已有网友及文章作者证实，目前已专门开设了 tilelang-ascend 适配仓库。
:::

参考资料

[1] TileLang 仅用 80 行 Python 代码就实现了 FlashMLA 95%的性能: *https://zhuanlan.zhihu.com/p/27965825936*

[2] FlashQLA: *https://github.com/QwenLM/FlashQLA*

[3] MSRA 的深度学习编译器四部曲: *https://www.microsoft.com/en-us/research/articles/ai-compilor/*

[4] TileLang 在 ICLR'26 的文章: *https://openreview.net/pdf?id=Jb1WkNSfUB*

[5] TVM-FFI: *https://github.com/apache/tvm-ffi/*

[6] 官方文档: *https://tilelang.com/deeplearning_operators/gemv.html*

[7] TileLang-Puzzles: *https://github.com/tile-ai/tilelang-puzzles/*

[8] TileOPs: *https://github.com/tile-ai/TileOPs*

[9] XPUOJ: *https://xpuoj.com/*
