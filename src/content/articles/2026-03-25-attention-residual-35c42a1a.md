关键词：Attention Residual、Block AttnRes、**推理延迟优化**、两阶段计算、**软硬协同设计**

> 作为月之暗面 AI Infra 团队的一员，这篇文章我想从 AI Infra，尤其是推理架构的角度（关于训练，我们的同事有一篇非常好的回答，推荐读一下，这里就不多谈了，见下面链接）：

- Kimi 新发布的「注意力残差」有什么亮点？马斯克和 Karpathy 为何同时点赞？
- https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331

::: block-1
这里想聊一聊 Attention Residual 背后的一些设计思考，也把论文里受限于篇幅没有展开的一些工程分析讲得更清楚一点。
:::

![文章插图](/articles/2026-03-25-attention-residual-35c42a1a/images/01.webp)

- **谈谈 Attention Residual 架构一些背后的想法**
- 作者：YyWangCS，https://zhuanlan.zhihu.com/p/2017528295286133070
- 6000 字，阅读 30 分钟，**播客 18 分钟**

相关推荐

- [从清华到 OpenAI：一位 AI Infra 工程师的成长与思考——对话翁家翌](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896307&idx=1&sn=61a45f13fdab2f25a3b0d70944f03d2e&scene=21#wechat_redirect)
- [NPU 设计的系统级艺术：权衡、适配与协同优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897198&idx=1&sn=4013a7097a200009e9439429e7ea61b9&scene=21#wechat_redirect)
- [“历史性的项目，错过了遗憾终生”：清华特奖游凯超与 vLLM 的三次蜕变与技术远征](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896591&idx=1&sn=689f8a0351d8ea9ed0c2c0ebd2f89ca2&scene=21#wechat_redirect)

> 熟悉我的朋友可能知道，我工作中一个很核心的内容就是模型架构设计和性能优化。所以我自己一直有一个认知：一个模型架构最后设计成什么样子，往往不只是算法问题，它同时也反映了团队对工程实现、硬件约束，以及算法和系统如何协同设计的理解。

Attention Residual 这项工作就很典型，它并不是单纯提出了一个“效果更好”的结构，而是从一开始就把模型能力、训练开销、推理延迟和硬件特点放在一起思考后，做出来的一套面向真实系统的架构设计。这篇文章大致会讲三件事：

1. **先聊推理性能优化，特别是延迟优化**。
2. **再聊这个架构本身的演进过程，尤其是我们是怎么从 Full AttnRes 一步步走到 Block AttnRes 的**。
3. 最后也会谈谈我自己对**团队协作这件事的一些新的感受。**

## 本文目录

- 一、Block AttnRes: 极致延迟和成本优化
  - 1.1 延迟和成本优化的差异
  - 1.2 基于 two-phase computation 的性能优化
  - 1.3 显存空间优化
  - 1.4 性能分析
- 二、从 Full AttnRes 到 Block AttnRes 的迭代历程
  - 2.1 算法的提出与初步验证
  - 2.2 Full Attention 显存问题的解决
  - 2.3 Full Attention 访存问题的解决
  - 2.4 训练的开销与 Block AttnRes 算法的确定
- 三、团队协作

![文章插图](/articles/2026-03-25-attention-residual-35c42a1a/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、Block AttnRes: 极致延迟和成本优化

这一节里，我会先说明大模型推理中延迟优化和成本优化的差异，以及这种差异是如何影响我们对 AttnRes 架构的设计选择的；进一步会说明，当前的 Block AttnRes 是如何在相比 standard residual connection 显著增强表达能力的同时，把额外的延迟和成本开销压到非常低的。

### 1.1 延迟和成本优化的差异

> 通常来说，优化延迟和优化成本是高度相关的。比如通过算子 fusion 优化 IO，或者优化 Flash Attention 这类核心计算，往往会同时降低延迟和成本。

但两者并不完全等价，尤其到了今天的大模型推理场景，这种差异已经越来越明显了。比较典型的有下面两类情况：

1. **延迟低，但成本不一定低**：有些方案主要是通过增加并行度、缩短 critical path 来降低端到端延迟。这样做对用户体验很有帮助，但模型总的访存量和计算量未必真的下降，所以成本改善可能并不明显。类似 ScMoE 这一类方案，就比较接近这个方向。
2. **成本低，但延迟不一定低**：另一类情况是一些典型的 latency bound 算子。比如 topk（sparse attention 或者 MoE router 中的 topk 计算）、一些小尺寸矩阵乘法（比如 HyperConnection 里的 `tf32_hc_prenorm_gemm`，或者 router gate 计算）等。这些算子在大 batch 下可能有不错的带宽和算力利用率，所以从吞吐和成本角度看问题不大；但在小 batch decode 场景下，它们对端到端延迟的影响却比较明显。

所以如果只从成本或者平均吞吐的角度看问题，很容易低估一些结构在真实推理里的 latency 代价。**尤其是在 decode 场景下，更需要警惕的是关键路径上有没有引入新的 latency bound 开销，因为这些开销虽然未必占大头，但往往会对端到端延迟带来比较明显的影响**。对于成本优化，过去已经有很多成熟的方法，比如通过更大的 batch、优化 GPU kernel pipeline 等方式提升计算利用率和带宽利用率。但对于延迟优化，尤其是 decode 小 batch 场景，问题要更棘手一些。正如我在之前的文章中写到的，Latency bound 算子通常会带来下面两个挑战：

1. **很难靠增加并行度来解决**：像 FFN 或者 Attention 里很多 memory bound 的计算，还可以通过增加并行度来换延迟；但 topk 和小矩阵 gemm 这类计算本身并行度就不高。虽然可以用 Split-K 之类的方法增加一些并行性，但由于 merge 开销很大，最终能带来的收益其实有限。
2. **硬件升级不一定带来收益**：新一代硬件比如 Blackwell，算力和带宽都可能成倍提升，但这并不意味着这些算子的 latency 会同步下降。很多时候它们的瓶颈并不在理论算力上，而在 launch、同步或者访存。所以我们甚至会看到某些 topk 实现在更新的 GPU 上反而更慢。

也正因为如此，这类开销的优化思路和普通 compute-bound/memory bound 算子不太一样。对 latency bound 问题，我们通常更关心下面几件事：

1. **能不能减少 IO**：最直接的方式就是通过 fusion 去掉不必要的 HBM 访存，因为很多时候真正贵的不是算，而是访存延迟。
2. **能不能和其他模块 overlap**：如果一个小算子本身打不满 GPU，那么最理想的情况不是单独把它做快，而是让它和别的计算并行起来，不落在关键路径上。
3. **能不能把开销摊薄**：比如配合投机采样，某些小算子的延迟在 batch 增加时增长并不明显，那么它的边际成本就有机会被摊薄。

后面我们会看到，这些认知直接影响了 AttnRes 的最终架构设计。Block AttnRes 并不是单纯为了“省一些计算”而设计的，而是在一开始就围绕一个目标展开：在尽量不引入额外计算开销的前提下，把 residual 结构的表达能力做强。后面的 Block Residual + two-phase computation，本质上都是围绕这个目标展开的。

### 1.2 基于 two-phase computation 的性能优化

> 如果直接 naively 实现 Block AttnRes，那么每一层都要去读取所有之前的 block representation，然后算一次 attention residual。这样会导致两个问题：

1. **对于 prefill workload**：虽然通过 Block AttnRes 将 attention 计算的 key 和 value 压缩到了 8 个 block，但是 $batch\times8\times 7168$ 的访存仍然有比较大的访存开销。
2. **对于 decode workload**：除了访存开销之外，由于 attention 计算本身相对复杂，受限于 shared memory 空间等限制，比较难和前后的 all reduce、RMSNorm 等计算有效的 fuse 到一起从而降低 high OTPS 场景的 latency。

::: block-1
Block AttnRes 的一个关键设计是：每一层 attention query 是一个和当前 hidden state 解耦的可学习参数。这个设计从算法上看很轻量，但从系统角度其实非常关键，因为它意味着同一个 block 内的所有 query 都可以提前拿出来，提前统一做一遍 batched inter-block attention。
:::

基于这一点，我们最终把计算拆成了两个 phase：

1. **Phase 1: batched inter-block attention** 对一个 block 中的所有层，统一和之前所有的 block representation 做一次 batch attention，得到每一层对应的 inter-block 部分结果以及 softmax 统计量（lse、max 等）。
2. **Phase 2: sequential intra-block attention + online softmax merge** 在 block 内仍然按层顺序推进，用不断更新的 partial sum 计算 intra-block 部分，然后通过 online softmax 把两部分结果精确 merge 起来。

> 现在我们来分析 two-phase computation 带来的好处：

1. **优化 IO**：通过 Phase 1 的 batch 计算，原来每一层都要重复读取的历史 block representation 被平摊掉了，本质上把“每层读一次”变成了“每个 block 读一次”。
2. **更容易做 fusion**：Phase 2 中的 online softmax merge 是 elementwise 计算，这意味着它可以很自然地和 all-reduce、RMSNorm 等算子融合，从而进一步减少额外 IO。
3. **提升并行度**：Phase 1 可以和 block 内第一个 layer 的部分计算 overlap，因此即使它本身引入了额外工作，也并不完全落在关键路径上。
4. **保持数值精确**：这里非常重要的一点是，two-phase 不是一个近似算法，而是通过 online softmax 实现了和原始 attention 完全等价的精确合并。这一点决定了它不仅仅是一个“工程 trick”，而是可以作为正式架构的一部分。最后要说明一下，这个 two-phase 的方法我在真实模型上是做了正确性验证。因此，Block AttnRes 的推理延迟开销并不是靠牺牲精度换来的，而是靠更合理的计算组织方式实现的。

### 1.3 显存空间优化

> 看到关于 Attention Residual 的讨论中，很多人第一印象就是由于要存更多层的输出结果，显存会不会炸。这里我们会解释在推理阶段为什么显存开销是可以忽略不计的；甚至在下一节，我们会说明即使是对于 Full Attention，这个显存开销也是完全可以接受的。

对于 Block AttnRes 来说，如果直接存完整的 $N * T * d$ 表示，那么在长上下文下显存压力会非常大。比如 128K token、8 个 blocks、hidden_dim=7168 的情况下，仅仅 block representation 就可能需要大约 15GB 显存。这显然不是一个可以忽略的开销。整体优化思路是沿 sequence 维度做 shard，把 block representation 分摊到 tensor parallel 的各个设备上。这样每张卡只需要保留本地 sequence shard 对应的 block cache，而不是整段序列的完整 cache。

1. **显存占用线性下降**：如果有 $P$ 张 TP 卡，那么单卡上的 block cache 从 $N * T * d$ 下降到 $N * (T / P) * d$。对于上面的 128K 场景，单卡显存大概可以从 15GB 降到 1.9GB 左右。
2. **能融入现有 TP 通信路径**：Phase 2 的 online softmax merge 本质上是 elementwise 的，因此可以自然嵌入标准的 TP all-reduce 路径中，具体来说就是 reduce-scatter 之后做本地 merge，做完 RMSNorm 后再通过 all-gather 恢复结果。这让它更容易和已有的 fused kernel 体系衔接起来。

进一步地，如果再结合 chunked prefill，比如 32K 一次的 chunk size（已经能够打满稀疏模型 MOE 计算的计算了），那么这部分显存开销还可以继续下降到远低于 1GB 的量级。

### 1.4 性能分析

> 这里我们对 AttnRes 的推理开销做一个更具体的分析，并主要和 standard residual connection 做对比。为了避免 `layer`、`block` 这些叫法混在一起，先约定一下术语：

1. **Transformer Decoder Block**：指一个完整的 Transformer block，里面包含一次 Attention 和一次 FFN。
2. `layer`：沿用论文里的记法，专指一次 AttnRes 执行的位置。由于 AttnRes 会在 Attention 前和 FFN 前各执行一次，所以 `1 个 Transformer Decoder Block = 2 个 layer`；一个有 64 个 `Transformer Decoder Block` 的模型，在论文记法里其实对应 128 个 `layer`。
3. `AttnRes block`：two-phase computation 里的分块单位，由若干个 `layer` 组成。
4. D：hidden dimension，这里默认使用 7168 进行计算。

先看 baseline。对于标准的 residual connection，在实现里通常会直接和后面的 PreNorm fuse 到一起，也就是常见的 `FusedAddRMSNorm`。这个算子的输入是 `hidden_states` 和 `residual`，输出是归一化后的新 `hidden_states` 以及更新后的 `residual`，因此总访存大致可以记为 4D。

> 再看 Block AttnRes。它的计算分成两个阶段。

**Phase 2：online-softmax merge** 这一步的输入是 `hidden_states`、`prefix_sum` 和 `partial_attention`，输出是新的 `hidden_states` 和 `prefix_sum`，因此总访存是 5D。也就是说，单看这一步，它相比标准的 residual connection 只多了 D 的访存。

**Phase 1：inter-block attention** 这里记 AttnRes 的 block 数为 N，每个 AttnRes block 内包含的 `layer` 数为 S。在一个典型配置里，如果模型有 64 个 `Transformer Decoder Block`，那就对应 128 个 `layer`；如果我们把 AttnRes 按 16 个 `layer` 一组来分块，那么就有 S=16，整个模型对应 N=8 个 AttnRes block。对于最后一个 AttnRes block，Phase 1 需要读取前面 8 个 block representation。如果把 key 和 value 的读取都算进去，那么这一阶段最坏情况下的读取开销可以近似记为 16D。但这只是最后一个 block 的情况，并不是平均成本，平均只有 8D。更重要的是平摊后的开销：因为这 8D 的读取会被当前 block 内的 16 个 layer 共同分摊，所以平均到每个 layer 上，额外读取大约只增加 0.5D。除此之外，Phase 1 还需要为当前 block 内的每个 layer 写出一个 `partial_attention` 结果，因此一共会有 16D 的写回，平摊到每个 layer 上就是 D。所以从平摊后的角度看，Phase 1 对每个 layer 带来的额外访存大约是 1.5D。

> 把两部分合起来看，Block AttnRes 每个 layer 相比标准 residual connection 增加的主要访存开销，大致是：

$$
\underbrace{1D}_{\text{Phase 2 相比 baseline 多出的部分}} + \underbrace{1.5D}_{\text{Phase 1 平摊后的额外开销}} = 2.5D
$$

这个增量其实是比较小的，大致可以理解成只多了一次很轻的 activation 计算。这也是为什么我们会说，在很多实际 workload 下，Block AttnRes 带来的推理开销是非常低的。实际测试里，它带来的额外开销类似 RMSNorm：即使在大于 50 OTPS 的输出场景下，这部分 decode 延迟也通常小于 2% 这个量级。

比如在 `batch=128`、64 个 `Transformer Decoder Block` 的配置下，端到端 decode 延迟增加不到 0.5ms；如果再考虑 MTP，这部分开销还会被进一步平摊。对于 32K prefill 这样的场景，由于 attention 等计算占比很高，这部分额外开销通常可以忽略不计，甚至在性能测试抖动下会出现 baseline 延迟略差的情况。

## 二、从 Full AttnRes 到 Block AttnRes 的迭代历程

> 如果只从算法表达能力的角度看，Full AttnRes 其实是最直接、也最“正确”的版本。它让每一层都可以直接对所有历史 layer output 做 selective aggregation，这个形式最完整，也最贴近“把 attention 迁移到 depth 维度”的原始直觉。

在这一节，我会简单介绍一下我们对于 Full Attn 的优化，以及为什么最终综合考虑硬件限制、训练、推理开销、算法效果选择了 Block AttnRes。

我们经历过如下的一些分析历程，下面这些部分是我根据所知道信息总结的，更多的是从我自己的视角，不一定代表全貌。

### 2.1 算法的提出与初步验证

首先是我们伟大的苏神苏剑林老师一开始提出了非常通用的数学形式，即 general full attention，而且给出了非常完善的理论分析，包括各种数学推导，然后苏神以及我们算法同事的初步实验也真证明了效果上的优势，这个其实给了我们 Infra 同事非常强的信心，这点重要性我会在之后的算法工程协作部分说明。

### 2.2 Full Attention 显存问题的解决

与大家一样，我看到这个架构之后，第一个顾虑点就是显存占用问题，64 层模型就要存储 128 份 hidden states。32K 上下文就需要 60G 的存储空间。但是很快就发现这个问题可以解决，因为完全可以按照序列维度做切分，比如 8 卡并行，其实 7.5G 显存就可以。

### 2.3 Full Attention 访存问题的解决

在优化显存过程中，我反而意识到真正难解决的问题不是显存，而是访存，因为访存开销是层数的平方量级，一个 64 层的大模型，一共需要访存 $2\times batch\times128\times128\times7168$ bytes，假设 32K 上下文，这个就对应 7.7 TB 访存，也就是说在 H200 这种 GPU 上要引入 2s 额外计算时间；即使按照上面说的将序列切分到 8 卡，也需要 0.3 s 的计算时间，这个已经相当于 40B 激活参数量的计算时间了。

为了解决这个问题，才想到去设计论文中的 two-phase computation 算法（full 版本 two-phase computation 见论文的 Appendix 部分），**核心是通过 batching 的方式来平摊访存开销**。通过 batching 方式可以将每层的访存量从 $O(L)$ 优化到 $O(L/N+N)$，即从 $O(L)$ 优化到 $O(\sqrt L)$，这个点可以将访存量优化到之前的六分之一左右。结合一些 IO fusion 以及计算并行，这个延迟是可以接受的。所以我们才拍：从推理 Infra 角度觉得 Full AttnRes 是可行的。

### 2.4 训练的开销与 Block AttnRes 算法的确定

同时，训练框架的同事也在非常极致的优化显存、通信等，但是后续发现训练的跨 PP 通信问题还是比较难解决，这个在文章中有提到，细节我不谈了。所以最终大家综合考虑确定了 Block AttnRes 结构。

最终的 block size 设置，综合考虑了训练效率、算法效果、推理效率等；block size=8，既可以满足训练效率，又能拿到大部分算法收益，同时又和上面的 $O(\sqrt L)$ 这个最佳 block size 相对比较接近。而从推理效率角度，当时新增加了一个考虑因素，那就是公司流量疯狂上涨，所以无论是延迟优化还是成本优化的压力都非常大，因此在最终定版的时候，我们在设计目的上就是做到几乎无性能开销。

::: block-1
说了这么多，更多的是想说明我们在设计 Block AttnRes 中的整体历程，以及最终方案的一些权衡。但是我们从来没有放弃表达能力更好的 Full AttnRes，也许之后有一天，我们有更多更强的 GPU，我们会看到 Full AttnRes 的上线。
:::

## 三、团队协作

> 这个项目其实做了挺久的，最终技术报告只体现了结果，但是中间大家做了很多的尝试和努力，甚至有些尝试都并不顺利。项目的进行很依赖整个团队的努力，这里既有算法同事的理论分析、制定方案、scaling 实验、效果排查等等，也有工程同事做的很多事情，比如训练的同事优化显存和通信，负责基建的同事保障稳定性，推理的同事优化性能和支持各种评测，包括我自己都花了很多时间去做性能分析和对分验证等等。

这里面其实我感受最深的是**互相间的正向激励**。算法同事和工程同事之间会积极的去给对方正反馈和信心，这个对于项目最终能够完成挺重要的。

剑林老师提过很多次，当时我这边设计完两阶段算法后，直接拍 Full AttnRes 推理能够搞定这点给了算法同事很大的鼓励和推进作用；但是反过来，其实正是他一开始对整个方案非常完备的理论分析，以及算法同事（张宇 @yzhangcs，我们的天才少年 Nathan，剑林老师，以及很多我不知道知乎账号无法 at 的同事）的实验结果，给了我们 Infra 同事很大的信心。而我正是认真读完剑林老师写的理论分析之后觉得非常信服才开始非常认真的去进行性能优化的，所以这个是双向的激励。

公司有一个梗图，在遇到算法或者架构问题的时候有同事会拿出来，里面有两句话，我觉得很能体现我们这种互相理解和鼓励：

1. 第一句是我从 Infra 角度说的，背景是剑林老师经常会有各种算法上的 idea，但是他总是担心系统上开销太大，系统搞不定，经常会来和我讨论。而我有一次很认真的回了一句话，**“数学上合理的架构，工程上没道理实现不了，如果实现不了，大概率是工程这边哪个地方理解不对”**。
2. 第二句是张宇作为算法角度说的：**“如果系统上实现不了，大概率是算法没真正想清楚”**。

我还挺喜欢这种大家互相理解、互相支持的感觉，也非常相信互相的正向激励才能更好的推进整个项目的完成。

事实上，最近我并不是完成一个项目后开香槟的快乐状态，而是始终有一些压力感。这种压力到并不是来自于人或者工作，我们团队气氛挺好的，我工作本身是非常开心的。**这种压力更多的是来自于当前的技术挑战：其实大家已经逐步感受到一个 AI tipping point 的到来，而我们所面临的技战也越来越大**。

- 比如流量指数增长，算法层面探索的越来越深入，参数量和模型架构等继续探索带来的未知性等等，这些都其实带来了挺大的技术挑战。
- 比如我做模型架构，经常会担心自己什么地方稍微算错一点就带来巨大的成本开销等等。

而每当我和同事聊起这些事情，同事之间互相鼓励对我还挺有帮助的。

相关推荐

- [重读《机器学习科研的十年》：未来最难的，还是保持初心和热情！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892043&idx=1&sn=fb5fe29f30a2155db388a13e01d44f3e&scene=21#wechat_redirect)
- [SGLang RL Lead 赵晨阳谈 AI 领域发展与职业选择：从学术到工业的深度思考](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895608&idx=1&sn=7cfc6c867cde88444e2dedd2ee950dc3&scene=21#wechat_redirect)
- [HotChips 2025：从 PyTorch 到 CUDA：大模型时代 GPU 内核自动生成的技术跃迁，Tri Dao最新演讲](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891347&idx=1&sn=173b8638e00da3610dd90ba7d5a4b7f6&scene=21#wechat_redirect)
