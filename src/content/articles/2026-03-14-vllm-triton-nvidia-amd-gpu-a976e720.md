关键词：**vLLM**、**Triton**、:mdnice-bold-italic[注意力后端]、分页注意力、**性能移植**、GPU

> 过去一年中，IBM 研究院、红帽和超威半导体（AMD）的团队合作开发并向主仓库贡献了一款基于 Triton 的 vLLM 注意力后端，旨在实现顶尖的性能表现，同时保证在不同 GPU 厂商的硬件上具备良好的可移植性。

开展这项工作的原因，是当下加速器硬件的种类愈发丰富，而维护大量高度专用化核函数的成本也在不断攀升。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/01.webp)

- **vLLM Triton Attention Backend Deep Dive**
- https://vllm.ai/blog/vllm-triton-backend-deep-dive
- 7000 字，阅读 36 分钟，**播客 15 分钟**

相关推荐

- [为什么 Autotuning 对 GPGPU 的性能可移植性来说是必须的？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888410&idx=1&sn=afcf753b2aa0f8b60aefc0bdf2042f9f&scene=21#wechat_redirect)
- [如何仅用 Triton，实现跨平台 GPU 的 Attention 顶尖性能？5.9 倍提速，达 105.9% SOTA 水准](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894314&idx=1&sn=3286d5f41de88dd002d8a1e1a53c7f89&scene=21#wechat_redirect)
- [从Kernel生成到SGLang/vLLM部署！FlashInfer-Bench：重塑 AI 生成内核的高效落地路径](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895801&idx=1&sn=fb695179b9930b27800a2ce00a71b25c&scene=21#wechat_redirect)

> 本文将对这项工作进行全面的技术解析，阐释 Triton 为何适配 vLLM 的技术需求，介绍 Triton 注意力后端的相关情况及其使用场景，随后深入讲解高性能分页注意力核函数的实现过程。
>
> 文中还将依次介绍核函数级别的优化手段、并行化策略、与 CUDA 图的交互方式以及基准测试结果，最后对 Helion 做简要介绍。

## 本文目录

- 关键问题
  - 问题一、性能可移植的边界：Triton 能否全面替代手写专用内核？
  - 问题二、动态调度与 CUDA Graph 的平衡：持久化内核会引发负载不均吗？
- 一、为何 Triton 能为 vLLM 提供助力
- 二、vLLM 中的 Triton 注意力后端
- 三、Triton 注意力后端的使用场景
- 四、基于 Triton 编写高性能、可移植的分页注意力核函数
- 五、Tips：分页注意力核函数的作用
- 六、基于查询块优化 tl.dot 的分块尺寸
- 七、借助并行分块 Softmax 实现并行化
- 八、CUDA 图、启动网格与 GPU 执行波次
- 九、从可变启动网格到持久化核函数
- 十、基准测试结果
- 十一、前瞻：Helion 中的分页注意力
- 结论

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/02.webp)

## 关键问题

### 问题一、性能可移植的边界：Triton 能否全面替代手写专用内核？

> Triton 的抽象是否真的能在所有场景下无损替代专家优化的 CUDA 内核？虽然作者给出了 H100 上 100.7% 的性能对比，但这只是特定模型、特定 batch size 下的结果。现实中，像 FlashAttention 这样的库之所以有 7 万行代码，正是因为它们在为不同架构（如 Hopper、Blackwell）做极致的手工调优。:mdnice-bold-italic[Triton 能否在更复杂的模型、更长的序列、更细粒度的硬件特性（如 Tensor Core、TMA、WGMMA）上真正匹敌这些手工调优？如果不能，那“可移植”是否意味着“平庸的性能”？]

Triton 用 800 行代码实现了与 7 万行 FlashAttention 3 相当的性能（H100 上达 100.7%），并在 AMD MI300 上取得 5.8 倍加速，这确实证明了其“性能可移植”的潜力。

::: block-1
但作者承认，微基准测试显示不同内核变体在不同负载（prefill/decode/混合）下各有优劣，没有单一配置统治所有场景。Triton 依赖 autotuner 自动选择 tile 大小，但 autotuner 的搜索空间有限，且新硬件特性（如 Hopper 的 TMA、WGMMA）可能尚未被 Triton 编译器充分映射。
:::

因此，**彻底放弃手写内核为时尚早——在极端长序列、稀疏注意力或未来架构上，专家优化的 CUDA 仍可能领先**。但 Triton 大幅降低了维护成本，使 vLLM 能在多数场景下接近最优性能，是走向硬件无关的重要一步。

### 问题二、动态调度与 CUDA Graph 的平衡：持久化内核会引发负载不均吗？

> “持久化内核”是为了解决 CUDA Graph 与可变 launch grid 的冲突，但这种设计是否引入了新的调度开销或负载不均问题？:mdnice-bold-italic[在极端动态的工作负载下（如混合长短请求），这种固定线程块+动态任务分配的策略是否会导致 GPU 资源利用率下降？我们是否在用一种“通用”的调度机制来掩盖底层硬件调度器的不足？持久化内核通过固定数量的线程块来适配 CUDA Graph，但这也意味着某些线程块可能处理更多任务，而另一些可能空闲]。这种“静态启动+动态负载”的模式在高度动态的在线推理场景中是否真的稳健？

持久化内核通过固定线程块数量、动态从全局内存中获取任务，解决了 CUDA Graph 与可变 launch grid 的冲突。**作者未直接讨论负载不均，但理论上，若某线程块处理超长 KV 遍历而其他块空闲，会导致部分 SM 闲置，降低利用率。**

然而，这种设计隐含了任务窃取或动态分配的机制——每个块循环获取下一个任务，直到队列为空，天然具备一定的负载均衡能力。

- 在典型推理 workload 中，请求长度分布较集中，效果应可接受。
- 但极端动态场景（如长短请求剧烈混合）可能仍需启发式规则或更精细的分组策略。

作者提到“heuristics 用于确定并行化方法”，表明团队已考虑此 trade-off，实际效果有待后续性能数据验证。

## 一、为何 Triton 能为 vLLM 提供助力

> vLLM 的目标是在各类平台、模型和执行策略下，实现最优的推理性能。在实际应用中，这意味着需要支持多种加速器及不同代际的硬件、各类模型架构，还要适配多样化的工作负载特征，比如不同的批次大小、序列长度和注意力模式。

一种实现思路是编写大量高度专用化的核函数，为每种模型和 GPU 架构单独调优。这种方式虽能实现高性能，但不具备可扩展性。要在多款 GPU 平台（如英伟达 Hopper、Blackwell 架构，超威半导体 MI300，英特尔 GPU，以及未来可能出现的各类平台）上维护数百个核函数，很快会变得难以操作。

因此，我们更倾向于采用**性能可移植的核函数**，Triton 后端正是遵循这一思路设计的。

::: block-1
Triton 是一门领域特定语言，开发者可以用 Python 编写 GPU 核函数如矩阵乘法、注意力计算相关核函数，这些核函数会被编译为可在多款平台上高效运行的 GPU 代码。**Triton 的分块编程模型实现了很好的平衡：其底层特性足以让开发者实现与硬件相关的优化，高层特性又能让代码在很大程度上与硬件解耦。**
:::

如图 1 所示，开发者以逻辑分块为单位编写计算逻辑，Triton 编译器和自动调优工具会决定这些逻辑分块如何映射到底层硬件。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/03.webp)

::: image-caption
图 1：Triton 的分块编程模型，编译器和自动调优工具会将逻辑分块映射为适配特定硬件的执行布局。代码通过tl.arange生成BLOCK_SIZE_N与BLOCK_SIZE_M维度的索引，拼接为 2D tile 坐标。左图 GPU 1 将 2D 块拆分为两个子块并行；右图 GPU 2 则将块拆分为多行细粒度任务，其中BLOCK_SIZE_M可由自动调优器（autotuner）动态确定，体现了 Triton 分块策略的灵活性与硬件适配性。
:::

不同 GPU 上的分块形状和执行布局可能差异显著，但这些决策会由工具自动完成，通常会由自动调优工具指导（更多细节可参考我们的论文《GPU Performance Portability needs Autotuning》，arxiv.org）[1]。

之前我们推送过这篇文章：[为什么 Autotuning 对 GPGPU 的性能可移植性来说是必须的？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888410&idx=1&sn=afcf753b2aa0f8b60aefc0bdf2042f9f&scene=21#wechat_redirect)

## 二、vLLM 中的 Triton 注意力后端

> 在大语言模型中，注意力计算通常是对性能影响最大的核心操作。为了简化复杂度，vLLM 引入了**注意力后端** 这一抽象层，**通过统一的应用程序编程接口（API）封装各类注意力计算的实现**，使其与线性层、层归一化等相对简单的组件解耦。

在这一抽象架构下，vLLM 支持多款注意力后端，包括 CUDA 平台上的 FlashAttention 和 FlashInfer、基于 ROCm 的注意力后端，以及适用于 MLA（多线性注意力）风格的专用注意力后端（完整列表见此处[2]）。Triton 注意力后端完全基于 Triton 实现[3]，是 vLLM 的原生后端。

研发该后端的初衷，是解决性能可移植性和依赖项管理的问题。**它的同一份源码可在英伟达、超威半导体、英特尔的 GPU 上运行，仅依赖 PyTorch 和 Triton 两个库，且作为 vLLM 的内置组件随时可用**。该后端最初由 IBM 研究院和红帽人工智能团队开发，目前已由更广泛的社区负责维护和扩展。

## 三、Triton 注意力后端的使用场景

在基于 ROCm 运行的超威半导体 GPU 上，Triton 注意力后端是默认选择；在英特尔 XPU 上也会启用该后端。当运行 32 位浮点精度计算时，vLLM 会回退到 Triton 注意力后端，因为 FlashAttention 在该精度下无法在上述硬件上运行。此外，该后端还支持需要特定特性的模型，比如 StepFun 音频模型所使用的 ALiBi 平方根注意力机制、sink tokens（锚定令牌，用于稳定注意力计算的特殊令牌）和 GPT-OSS 相关行为，尤其在英伟达 Hopper 架构之前的 GPU（如 A100）上表现良好。

同时，它还支持小注意力头尺寸的模型、编码器-解码器注意力计算以及多模态前缀注意力计算。由于 Triton 注意力后端是 vLLM 的内置组件，当 FlashAttention、FlashInfer 或其他依赖项无法使用或导入失败时，它会作为**兜底后端** 生效，也支持批次不变性等特性。

## 四、基于 Triton 编写高性能、可移植的分页注意力核函数

> Triton 注意力后端的开发工作启动时，研发团队首先在 vLLM 框架外实现了核心核函数，并通过大量微基准测试对其进行评估。核函数的 API 按 vLLM 的需求设计，而性能调优工作则在端到端集成前独立完成。

微基准测试[4]至关重要，它能帮助研发团队理解核函数在**预填充密集型** （模型推理初期填充上下文的计算密集型场景）、**解码密集型** （模型生成新令牌的逐步计算场景）、混合负载场景下的性能表现，也能分析不同批次大小和上下文长度对性能的影响。

下面图 2 展示了具有代表性的微基准测试结果，横轴为令牌总数，纵轴为延迟，不同子图分别对应纯预填充、混合、纯解码负载场景。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/04.webp)

::: image-caption
图 2：多款 Triton 分页注意力核函数变体在预填充、解码、混合负载下的微基准测试对比。这张图对比了四种注意力实现（Triton naive、Triton parallel tiled、Triton GQA opt、flash_attn）在不同 decode_share（0.0/0.5/1.0）下的归一化延迟随 token 数的变化。decode_share=0.0 时，flash_attn 延迟最低，Triton naive 次之，Triton parallel tiled 随序列增长延迟攀升最快。decode_share=1.0 时，Triton GQA opt 延迟显著降低，与 flash_attn、Triton naive 差距缩小，凸显 GQA 优化在高 decode_share 场景下的性能优势。
:::

结果表明，不同的核函数变体在不同场景下各有优势，没有单一的配置能在所有场景中表现最优。

微基准测试是对端到端基准测试的补充，它能暴露核函数级别的性能表现，而这些细节往往会被系统级的影响所掩盖。

## 五、Tips：分页注意力核函数的作用

> 分页注意力通过对键值缓存进行分页管理，实现了内存高效的注意力计算。对于批次中的每个查询，核函数会处理每个查询令牌；针对每个令牌，核函数会遍历查询头及对应的键值头，随后遍历分页的键值缓存，计算注意力分数并应用值向量。

下面图 3 展示了这一计算结构：查询令牌沿横轴排列，查询头沿纵轴排列，对分页键值缓存的遍历构成最内层循环。为了简洁，文中省略了因果掩码、滑动窗口等细节。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/05.webp)

::: image-caption
图 3：分页注意力的概念示意图，展示了查询令牌、查询头以及对分页键值缓存的遍历过程。以num_query_heads、sequence_length为轴，将扁平化的变长批次（Flattened “varlen” batch）划分为多个计算块：BLOCK_Q 与查询头数相关，BLOCK_N=16 固定键值维度，BLOCK_M 适配查询序列长度。不同长度的查询序列（q_len[0]、q_len[1]、q_len[2]）被规整为等粒度块，实现高效硬件并行。
:::

- 如需了解该核函数底层优化的详细解释，推荐阅读核函数开发者发布在 PyTorch 博客的相关文章：https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
- 核函数代码地址：https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py

## 六、基于查询块优化 tl.dot 的分块尺寸

> 注意力计算的核心是矩阵乘法，在 Triton 中由 `tl.dot`（Triton 的矩阵乘法原语）实现。但要实现高性能，需要设置足够大的分块尺寸以充分利用硬件资源，而单纯加载分页的键值缓存并不能达到理想的性能效果。

键值侧的分块尺寸受键值缓存的页大小限制，因此优化工作主要集中在查询侧。对于**分组查询注意力** （GQA，将多个查询头绑定到同一个键值头的注意力机制，平衡性能和内存占用），可将单个键值头对应的所有查询头批量处理，提升缓存的复用率。为了进一步提升并行度，研发团队将多个查询令牌组合为一个工作单元，称为**查询块（Q block）**。

下图 4 阐释了这一方法：启动网格覆盖批次大小和键值头维度，而查询块则决定每个核函数实例处理的查询令牌数和查询头数，自动调优工具会为各平台选择合适的块尺寸。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/06.webp)

::: image-caption
图 4：查询块将多个查询头和查询令牌组合为单个工作单元，提升 tl.dot 的硬件利用率和缓存复用率。左侧 2D 网格以batch_size和num_heads为维度，规整排布计算块；右侧 Q Block 为三维结构，维度包含query tokens（BLOCK_Q）、query heads（BLOCK_M/BLOCK_Q）与features（HEAD_SIZE），且满足BLOCK_M/BLOCK_Q = num_query_heads/num_kv_heads，体现了从 2D 并行到三维分块的资源调度逻辑。
:::

## 七、借助并行分块 Softmax 实现并行化

> 批量处理多个查询令牌的方式在预填充负载场景下表现优异，但在解码负载场景中毫无优势——因为解码时每次仅处理一个查询令牌。为解决这一问题，研发团队通过**并行分块 Softmax（平行分块归一化指数函数）** 引入了额外的并行化机制，该核函数也被称为“3D 核函数”。

这一方法将对键值缓存的遍历操作拆分到多个核函数实例中执行，每个实例计算部分结果，后续通过归约操作得到最终输出。由于 Triton 不提供全局屏障（用于同步所有核函数实例执行的同步机制），该归约操作需要启动第二个核函数，这就需要在**提升并行度** 和**增加核函数启动开销** 之间做权衡，研发团队通过启发式算法判断该方法的适用场景。

## 八、CUDA 图、启动网格与 GPU 执行波次

> CUDA 图通过记录并复现固定的执行图，减少核函数的启动开销。但注意力核函数的实现存在难点：其启动网格（核函数实例的组织方式）通常由批次大小和序列长度决定，具有可变性。

GPU 通过固定数量的**流多处理器（SM）** 执行核函数，当启动的线程数超过流多处理器的数量时，计算会以**波次（wave）** 为单位依次执行。图 5 展示了这一行为，后续的波次会导致硬件资源利用不充分。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/07.webp)

::: image-caption
图 5：启动线程数超过可用流多处理器数量时的 GPU 执行波次示意图。本示例中，GPU 配备 8 个流多处理器，需执行 12 个线程。任务分为两个阶段：wave 0 阶段线程满负载运行（绿色区域），算力利用率高；wave 1（尾波次）阶段因启动网格不匹配，仅部分线程参与计算，大量线程处于闲置状态（红色区域），直观体现了尾计算阶段因任务粒度不匹配导致的硬件算力损耗。
:::

- 参考链接：https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

当该执行过程被捕获到 CUDA 图中后，即便实际的工作负载规模减小，这种资源利用的低效问题也会被复现。图 6 展示了固定的启动网格如何导致额外的无效计算，进而增加延迟。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/08.webp)

::: image-caption
图 6：通过 CUDA 图复现固定启动网格时产生的额外无效计算。在 SM（流式多处理器）上，计算分为 wave 0 与 wave 1（tail）两个阶段：wave 0 中线程满负载执行，而 wave 1 仅 4 个线程有实际计算任务，其余 8 个线程需等待。因启动网格不匹配，大量线程空转或等待，右侧红色区域为未利用的硬件资源，凸显了尾波次计算时的算力闲置与效率损耗。
:::

## 九、从可变启动网格到持久化核函数

> 早期版本的分页注意力核函数采用**可变启动网格**，网格规模随工作负载大小动态调整，如下图 7 所示。这种方式虽具备灵活性，但与 CUDA 图的兼容性较差。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/09.webp)

::: image-caption
图 7：早期分页注意力核函数所使用的可变启动网格。左图 2D 策略以 batch_size 和 num_heads 为二维维度，计算块规整排布，头间独立但序列维度未显式拆分；右图 3D 策略新增 sequence 维度，将计算沿 batch、head、序列三维展开，通过更细粒度的分块并行提升硬件资源利用率，更适配长序列场景下的高效算力调度。
:::

::: block-1
为解决这一问题，研发团队设计了**持久化核函数**。该方案会启动固定数量的核函数实例，实例数与硬件的可用计算资源相等，每个实例通过读取 GPU 内存中的元数据，动态确定自身需要处理的计算量。这一设计让启动网格保持固定，从而实现 CUDA 图的高效复用。
:::

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/10.webp)

::: image-caption
图 8：持久化核函数方案，采用固定启动网格和动态任务分配机制。左图中，2D 策略将每个注意力头独立分配至 SM / 核心，头内计算串行展开；右图中，3D 策略在固定分块数下，让注意力头数与 SM / 核心数呈线性扩展，通过分块并行提升资源利用率，更适配长序列场景下的硬件算力调度。
:::

## 十、基准测试结果

> 2025 年末的基准测试结果验证了该方案的有效性。图 9 展示了在英伟达 H100 和超威半导体 MI300 显卡上，运行 Llama 3.1 8B 模型、批次大小为 1、输入长度为 500 令牌时的端到端延迟结果，横轴为输出令牌长度。

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/11.webp)

::: image-caption
图 9(a) 展示了在 H100 GPU 上不同注意力实现方案的归一化延迟随输出序列长度变化的趋势。FlashAttention V3（CUDA 实现）与 Triton 优化版本（Triton Q Block、Triton Q Block + Par TS、Triton Static Launch Grid）均保持极低延迟，且随序列长度增长平缓上升；而 Triton Baseline 延迟随输出长度增长急剧攀升，在长序列场景下显著高于其他方案，凸显了优化后 Triton 方案与 FlashAttention V3 的性能优势。
:::

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/12.webp)

::: image-caption
图 9(b) 展示了MI300 GPU上不同 Triton 注意力实现的归一化延迟随输出序列长度的变化趋势。Triton Baseline 延迟随输出长度增长急剧攀升，在长序列场景下性能最差；而 Triton Q Block、Triton Q Block + Par TS 及 Triton Static Launch Grid 三种优化方案延迟显著更低，其中 Triton Static Launch Grid 的延迟控制最优，充分体现了优化策略在长序列推理场景下的性能优势。
:::

- 在 H100 显卡上，对于长解码请求，Triton 注意力后端的性能达到了 FlashAttention 3 的 100.7%；
- 在 MI300 显卡上，其性能较早期实现提升了约 5.8 倍。

值得注意的是，两款平台使用的是同一份 Triton 核函数源码。需说明的是，基于 Triton 的分页注意力实现仅有约 800 行代码，而 FlashAttention3 的代码量约为 7 万行。

## 十一、前瞻：Helion 中的分页注意力

> Helion[5] 是 PyTorch 团队推出的一款全新领域特定语言，可将其视为更高层级的 Triton，或支持分块编程的 PyTorch。相比 Triton 提升了抽象层级，更易编写高效且正确的内核，同时优化自动调优流程，其名称取自氦 - 3 原子核，与 Triton（氚核）对应，也可看作带分块的 PyTorch 或高阶 Triton。

它会通过约 10 分钟自动调优评估海量 Triton 实现方案，提升内核在不同硬件的性能移植性，可自动完成张量索引、掩码处理、网格尺寸与 PID 计算、搜索空间定义、内核参数管理、循环归约等工作，还支持 PID 重排、循环重排等自动优化。

该工具提供 Colab 演示和 2025 PyTorch 大会讲解视频，矩阵乘法示例展现其基于标准 PyTorch 语法，循环内代码会编译为单一 GPU 内核。

::: block-1
Helion 首次运行会执行自动调优，用户可硬编码最优配置跳过，生产环境推荐预调优配置；其参数分为管控 GPU 执行的 Config（性能导向、支持自动调优）和控制编译的 Settings（开发调试用），同时提供日志、打印生成代码等调试相关的环境变量与装饰器参数。
:::

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/13.webp)

::: image-caption
使用vllm基准测试工具，在H100上结合Llama3.1 8B模型和ShareGPT数据集进行的端到端性能测量。左侧为归一化总 token 吞吐量，以 triton_attn 为基准（1.0），helion_attn（动态形状）约 0.95，静态形状版仅约 0.25。右侧为对数刻度的归一化延迟，静态版 median TTFT 和 ITL 延迟显著更高，动态版延迟接近 triton_attn 水平。
:::

![文章插图](/articles/2026-03-14-vllm-triton-nvidia-amd-gpu-a976e720/images/14.webp)

::: image-caption
使用vllm基准测试在MI300X上对Llama3.1 8B和ShareGPT数据集进行的端到端性能测量。左侧为归一化总token吞吐量（以triton_attn为基准1.0，越高越好），helion_attn（动态形状）约0.6，静态形状版仅约0.15。右侧为对数刻度归一化延迟（越低越好），静态版median TTFT和ITL延迟显著更高，动态版延迟更接近triton_attn水平。
:::

**研发团队已在 Helion 中实现了一个简化版的分页注意力核函数作为实验，初期结果表现喜人**。相关工作已在 PyTorch 博客[6]发布，代码也以草稿形式[7]提交了拉取请求至 vLLM 代码仓库。

## 结论

> 随着模型、推理优化技术和硬件平台的不断发展，**性能可移植性** 的重要性日益凸显。vLLM 中的 Triton 注意力后端证明，通过单一、可移植的核函数实现，也能达到顶尖的注意力计算性能。

通过精心的核函数设计、大量的微基准测试，以及持久化核函数、CUDA 图等系统级优化手段，Triton 后端在保持跨 GPU 厂商可移植性的同时，性能持平甚至超越了高度专用化的实现方案。如今，它已成为超威半导体 GPU 的默认注意力后端，且同一份源码能在英伟达、英特尔平台上高效运行。

本文仅概述了 Triton 注意力后端中最重要的优化手段，更多细节和基准测试结果可参考我们的相关论文《The Anatomy of a Triton Attention Kernel》（arxiv.org）[8]。

也就是先前 NeuralTalk 推送的这篇文章：[如何仅用 Triton，实现跨平台 GPU 的 Attention 顶尖性能？5.9 倍提速，达 105.9% SOTA 水准](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894314&idx=1&sn=3286d5f41de88dd002d8a1e1a53c7f89&scene=21#wechat_redirect)

参考资料

[1] GPU Performance Portability needs Autotuning: *https://arxiv.org/abs/2505.03780*

[2] vllm/v1/attention/backends: *https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends*

[3] vllm/v1/attention/backends/triton_attn.py: *https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py*

[4] vllm-triton-backend: A Triton-only attention backend for vLLM: *https://github.com/foundation-model-stack/vllm-triton-backend*

[5] A Python-embedded DSL that makes it easy to write fast, scalable ML kernels with minimal boilerplate.: *https://github.com/pytorch/helion*

[6] Portable Paged Attention in Helion: *https://pytorch.org/blog/portable-paged-attention-in-helion/*

[7] Experimental attention backend in helion: *https://github.com/vllm-project/vllm/pull/27293*

[8] The Anatomy of a Triton Attention Kernel: *https://arxiv.org/abs/2511.11581*
