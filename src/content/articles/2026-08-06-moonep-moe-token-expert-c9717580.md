**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：MoonEP、:mdnice-bold-italic[专家并行负载均衡]、**MoE 动态调度**、权重预取、**Token 路由**

> **说明** ：**本文只介绍MoonEP[1] 的基本思路和接口抽象，不展开 kernel 实现**。对于代码细节，我倾向于使用 AI 辅助阅读源码来理解。:mdnice-bold-italic[本文主要讨论三个问题：为什么 MoonEP 要同时调度 token 和 expert、这样做要付出什么成本，以及它适合哪些场景。]

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/01.jpg)

- **从 MoonEP 出发：MoE 负载均衡，搬 Token 还是搬 Expert？**
- https://zhuanlan.zhihu.com/p/2068137220662473580
- 本文作者 Leo z，https://www.zhihu.com/people/yin-yin-yue-yue-yy-97

::: related-title
**相关推荐**
:::

::: related-links
- **[MoE吞吐达94%理想上限！北大联合小红书提出UltraEP：破解大规模专家并行的负载失衡难题](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903917&idx=1&sn=ec7e38d0846e941c1a01280fb70257ed&scene=21#wechat_redirect)**
- **[NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903850&idx=1&sn=e7418452bcd7df13a7917f2463127104&scene=21#wechat_redirect)**
- **[把 MoE 权重做成“分页内存”！UIUC 提出 PagedWeight，72%显存节省破解 KV 缓存与权重的显存死局](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903812&idx=1&sn=8884a16353e79af577a4e1bbeba1c872&scene=21#wechat_redirect)**

:::

> 本文以“MoE 负载均衡应调度 Token 还是 Expert”为核心问题，**针对传统专家并行（EP）中极端路由偏斜导致的计算与通信瓶颈展开分析**。传统方案如 EPLB+LPLB 采用阶段性调整 Expert 副本拓扑、运行时动态分配 Token 的模式，无法应对单批次突发热点。

作者详细阐释 **MoonEP 的核心设计**：在每次层调度前生成统一通信计划，通过全局汇总与本地落位的两级规划，同时决定 Token 分发路径与临时 Expert 权重预取方案，确保各 Rank 处理等量 Token 任务，维持静态通信缓冲区与稳定的计算负载。文中建立成本模型量化权重预取与通信收益的平衡关系，得出单份 Expert 权重约对应 1536 个 Token 的通信节省平衡点。

**后文也进一步对比三种不同时间粒度的调度方案**，结合 SIGCOMM 2026 相关研究分析各自适配场景，:mdnice-bold-italic[并以 Kimi 系列模型为例，说明 Expert 数量增多、单 Expert 体积减小的演进趋势将释放动态调度优化空间，指出 MoonEP 更适配大批次训练与高吞吐 Prefill 场景。]

## 本文目录

- 一、一个极端例子：所有 Token 都路由到 Expert 0
- 二、MoonEP 之前：阶段性调整 Expert，运行时调度 Token
- 三、MoonEP 的核心：Plan 同时决定 Token 去向和权重预取
- 四、成本分析：一份 Expert 权重，对应多少 Token？
- 五、场景决定设计：不同时间尺度的取舍
- 六、最后的补充：从 Kimi K2.6 到 Kimi K3
- 小结

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、一个极端例子：所有 Token 都路由到 Expert 0

> 假设一个 Expert Parallelism（EP）组中的每个 rank 都有 10,000 个 token，hidden size 为 4,096，路由采用 Top‑1。现在出现了极端不均衡：**所有 token 都被路由到 expert 0。**

Expert 0 的 FFN 包含 gate、up 和 down 三个权重矩阵 ，每个矩阵都是 $4096 \times 1024$。Token 和权重都使用 BF16。

每个 rank 上的 token 数据量为

$$
10{,}000 \times 4096 \times 2\ \text{Byte} =78.125\ \text{MiB}.
$$

Expert 0 的三块权重一共是

$$
3 \times 4096 \times 1024 \times 2\ \text{Byte} =24\ \text{MiB}.
$$

在之前的 EP 中，其他 rank 需要把 token 发给 expert 0 的 owner rank (dispatch)，计算结束后再把结果收回来(combine)。对每个远端 rank 来说，单向 token 数据为 78.125 MiB，算上 dispatch 和 combine，往返为 156.25 MiB。与此同时，所有 FFN 计算都落在一个 rank 上，其他 rank 基本空闲。

::: block-1
:mdnice-bold-italic[这个例子提出了一个很直接的问题：既然一份 expert 权重只有 24 MiB，为什么一定要把大量 token 发给它？能不能把 expert 临时复制到其他 rank？更一般地说，系统应该只动态调度token，或者只动态调度 expert，还是同时调度两者？]
:::

## 二、MoonEP 之前：阶段性调整 Expert，运行时调度 Token

> DeepSeek 开源的EPLB（Expert Parallelism Load Balancer）[2]会根据历史负载，复制高负载 expert，并重新安排物理 expert 副本在 GPU 上的位置。它提供分层和全局两种放置策略。

EPLB 解决的是较慢时间尺度上的问题：**哪些 expert 需要副本，以及副本放在哪里。** 它可以定期重新运行，但一次重排完成后，expert 拓扑会在一段时间内保持不变。因此，它是阶段性重排，不是每次 MoE dispatch 前都改变 expert 布局。

有了副本，还需要决定 token 去哪个副本。LPLB（Linear‑Programming‑Based Load Balancer）[3] 在预先给定的副本拓扑上，根据当前 batch 的负载，通过线性规划把 token 的目标逻辑 expert ID 重定向到不同物理副本，以降低各 rank 的 token 数差异。

从运行时角度看，这类方案仍然可以概括为**相对静态的 expert + 动态 token** ：expert 拓扑按阶段更新，token 在已有副本之间动态分配 。如果当前 batch 突然出现一个预设拓扑没有覆盖的热点，planner 可以改变 token 分配，却不能在这次 dispatch 中临时增加新的副本。

MoonEP 的主要变化，是把临时 expert 副本也放进当前 dispatch 的规划中。

## 三、MoonEP 的核心：Plan 同时决定 Token 去向和权重预取

> MoonEP 根据当前 router 输出生成`MoonEPCommPlan` 。这份 plan 描述本 rank 的 token‑expert assignments 应写到哪里，以及各 rank 需要预取哪些 experts。这里的“动态 expert”不是改变参数的长期 owner，而是为当前计算创建临时权重副本。

**从实现上看，planning 分为全局规划和本地落位两层**。每个 rank 先把本地`[E]` 的`tokens_per_expert` 写入 rank 0 的对称 meta buffer；rank 0 汇总成`[R,E]` 的全局计数 ，计算`alloc[dest_rank, expert]` ，即每个 expert 有多少 token assignments 交给各目标 rank，同时生成各 rank 的权重预取选择和 FFN 分段布局。随后，每个 rank 结合这份全局 allocation 与自己的本地 topk，为本地每个 token‑expert pair 计算最终目标位置，写入本地`plan.dst` 。

从上层接口看，前向过程可以简化为：

```
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(    hidden_sh,              # [S, H] bf16    route_weights_sk,       # [S, K] fp32    topk_experts_sk,        # [S, K] int32    tokens_per_expert,      # [E] int32)buffer.prefetch_weight(    plan=plan,    full_gate_weight=full_gate_weight,    full_up_weight=full_up_weight,    full_down_weight=full_down_weight,)# grouped_ffn 由上层框架提供，不属于 MoonEPexpert_output_nvsh = grouped_ffn(    hidden_nvsh,    full_gate_weight,    full_up_weight,    full_down_weight,    cu_seqlens,)output_sh, _, _ = buffer.combine(    plan=plan,    hidden_nvsh=expert_output_nvsh,    route_weights_nvs=route_weights_nvs,)
```

当前代码中，`dispatch` 和`prefetch_weight` 是两个独立步骤。`dispatch` 先运行 planner，再根据结果发送 token；`prefetch_weight` 随后读取`plan.experts_to_copy` ，复制 plan 选中的权重。

从上层框架的角度，plan 中最关键的决策可以归纳为两类：

- `plan.dst` ：本 rank 的逐 token‑expert‑pair 结果，编码目标 rank 和目标位置。它由本 rank 根据本地 topk 和 rank 0 生成的全局 allocation 前缀表计算。
- `plan.experts_to_copy` ：形状为`[R, B]` ，指定每个 rank 的 B 个 prefetch slots 分别复制哪些 experts；未使用的 slot 记为`-1` 。

MoonEP 要求每个 projection 使用连续的`[E+B, H, H']` 权重张量。前`E` 行对应原始 experts，并通过对称内存映射到各 rank；后`B` 行是本 rank 的临时 prefetch slots。`dispatch` 还会返回`[E+B]` 的`cu_seqlens` ，用来告诉 grouped GEMM 每一行权重负责哪一段 token。

Planner 保证每个 rank 接收恰好 $S\times K$ 个真实 token assignments (S: num_tokens_per_rank, K: topK)，不受原始路由偏斜影响。实际 FFN 输入是`[NvS, H]` ，其中`NvS` 还包括各 expert segment 的 padding。所有 rank 使用相同的静态容量，因此计算量、通信 buffer 大小和 FFN 输入 shape 都不会随热点 rank 改变。

同一份 plan 还会在 combine 和 backward 中使用。Combine 根据 plan 把 expert 输出恢复到 token 顺序；combine backward 可以复用前向 plan 重新 dispatch 输出梯度，不再运行 planner。`zero_copy=True` 可以让 FFN 直接读写通信 buffer。

## 四、成本分析：一份 Expert 权重，对应多少 Token？

> 假设平均每个 rank 接收 $\bar T$ 个 Token，热点 rank 接收了 $2\bar T$ 。

- 在 dispatch 阶段，热点 rank 要接收两倍的输入，是接收侧瓶颈；
- 在 combine 阶段，它又要把两倍的 Expert 输出发回各个源 rank，成为发送侧瓶颈。

MoonEP 把多出的 Token 分给临时 Expert 副本，可以同时消除这两个热点。热点 rank 的 FFN 计算时间也会随之缩短，但这里因为无法精确量化暂时不计，只比较通信时间。

继续使用 hidden size 4,096、intermediate hidden size 1,024，并假设 Token 和权重都是 BF16。一个 Expert 的 gate、up、down 三块权重共计：

$$
3\times4096\times1024\times2\ \text{Byte} =24\ \text{MiB}.
$$

一个 Token 的 hidden activation 是 8 KiB。每从热点 rank 迁走一个 Token assignment，dispatch 接收侧少 8 KiB，combine 发送侧再少 8 KiB，关键路径上一共减少 16 KiB：

$$
8\ \text{KiB}+8\ \text{KiB}=16\ \text{KiB}.
$$

因此，若为了从热点 rank 迁走 $\Delta T$ 个 Token，某 rank 需要预取 $M$ 份 Expert 副本，那么两边通信时间大致相当时：

$$
M\times24\ \text{MiB} \approx \Delta T\times16\ \text{KiB},
$$

也就是：

$$
M:\Delta T\approx1:1536.
$$

> 每预取 1 个 Expert，大约需要迁走 1,536 个 Token，权重预取时间才能与消除 dispatch 接收侧和 combine 发送侧瓶颈所节省的时间相当。

例如，平均负载是 10,000 个 Token、热点 rank 收到 20,000 个，那么需要迁走 10,000 个 Token，对应大约 6.5 份 Expert 权重:mdnice-bold-italic[。这个比例假设两类通信的有效带宽相近，且暂未计入 planning 和训练时的 gradient reduce。]

## 五、场景决定设计：不同时间尺度的取舍

> 系统设计没有脱离场景的最优解。多久调整一次 expert、每次允许复制多少 expert、是否追求每一步完全均衡，都是成本与收益之间的取舍。

我们团队被 SIGCOMM 2026 录用的《Balancing and Beyond: Communication‑Centric Optimizations in Expert Parallelism》[4]中，也有一部分研究**动态 expert 权重传输和动态 token 调度**。我刚看到 MoonEP 时便感到很熟悉，:mdnice-bold-italic[但两项工作针对的场景不同，因此选择了不同的时间粒度粒度。]

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/03.jpg)

::: image-caption
表格梳理 EPLB+LPLB、本文工作、MoonEP 三种 MoE 专家负载均衡方案，从适用场景、Expert 调整粒度、Token 调度方式、方案取舍四个维度展开对比。EPLB+LPLB 面向长期稳定热点，阶段性调整拓扑，以固定拓扑压低运行开销；本文工作针对中小 batch 推理，按若干解码步少量调整专家，借助热点时间局部性降低迁移开销；MoonEP 适配大规模训练与高吞吐 prefill，每层按需选择临时副本，以细粒度调度开销换取负载完全均衡。
:::

**我们的方案利用同一 batch 在连续 decode steps 中的热点 expert 时间局部性**。Token 每个 step 都动态 dispatch，但权重分布只在若干 steps 内调整一次。由于 decode 单步的 token 量较小，我们限制每个 rank 每次只能迁移少量 experts，再通过贪心算法决定迁移哪些 experts、迁移到哪些 ranks，以及如何配合 token dispatch。:mdnice-bold-italic[在这个场景中，目标是提高整体 decode 吞吐，不是为每一个 step 支付任意代价来实现完全均衡。]

MoonEP 面对的成本结构不同。大 $S$ 会放大计算不均衡和动态 shape 的问题，也更容易摊薄 per‑dispatch planning 与权重预取成本。因此，MoonEP 在每次 layer dispatch 中根据当前路由选择临时副本，并让每个 rank 获得相同数量的 token assignments。从成本模型看，这种方案更容易在大 batch 训练或高吞吐 prefill 中体现收益。

::: block-1
:mdnice-highlight[从 EPLB、我们的工作到 MoonEP，可以看到三个不同时间尺度：阶段性重排、若干 decode steps 调整一次、每次 layer dispatch 都规划]。它们已不是简单的替代关系，是针对不同 batch 大小和执行阶段做出的不同选择。
:::

SIGCOMM 2026 将于**2026 年 8 月 17–21 日** 在美国丹佛举行。欢迎大家关注我们的相关工作。

## 六、最后的补充：从 Kimi K2.6 到 Kimi K3

> 模型结构也会改变动态调度的成本。

Kimi K2.6 官方配置[5] 给出的字段是`n_routed_experts=384` 、`num_experts_per_tok=8` 、`hidden_size=7168` 和`moe_intermediate_size=2048` ；

Kimi K3 官方说明[6] 给出的参数是 896 个 experts、每个 Token 选择 16 个、Latent MoE dimension 为 3584、每个 Expert 的 MoE hidden dimension 为 3072。

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/04.jpg)

::: image-caption
表格对比 Kimi K2.6 与 Kimi K3 两款 MoE 模型的核心架构参数，涵盖路由专家数量、Top‑K、MoE 输入维度、Expert 中间维度以及三矩阵 BF16 等价大小。Kimi K2.6 有 384 个路由专家，Top‑K 为 8，MoE 输入维度 7168；Kimi K3 升级为 896 个路由专家、Top‑K 为 16，采用 Latent MoE，输入维度降至 3584。Kimi K3 专家中间维度提升至 3072，三矩阵 BF16 等价大小下降至 63MiB，体现出潜在 MoE 架构在参数配置上的优化特点。
:::

最后一列采用与本文前文相同的简化口径：将每个 Expert 近似为两个输入投影和一个输出投影，共有 $3HH'$ 个元素，并统一按 BF16 计算。由此，Kimi K2.6 的单 Expert 大小为 $3\times7168\times2048\times2=84$ MiB，Kimi K3 为 $3\times3584\times3072\times2=63$ MiB。

因此，Kimi K3 的 routed experts 数量是 K2.6 的 $896/384\approx2.33$ 倍，而单 Expert 的 BF16 等价大小减少了 25%。Expert 数量更多、单个 Expert 更小，意味着运行时选择临时副本时有更细的调度粒度，复制单个 Expert 的成本也更低。

从这个统一口径看，Kimi K3 的 Expert 更多、单个 Expert 更小，确实给动态 Expert 调度留下了更大的优化空间；**实际收益仍取决于部署采用的数据格式以及通信实现。**

## 小结

> MoonEP 的核心不是单独优化 token 通信，也不是简单增加固定副本，确切来说，**是根据当前路由，用同一份 plan 决定 token 去向和临时权重副本**。这样可以让每个 rank 处理相同数量的 token assignments，并保持静态 buffer。

代价也很明确：**plan 需要时间，权重需要预取，训练还需要归并副本梯度**。

- 中小 batch decode 更适合降低权重调整频率，利用多个 step 的时间局部性；
- 大 batch 训练或高吞吐 prefill 更容易承担每次 dispatch 的动态规划成本。

**最终采用哪种方案，取决于具体 workload**，而不是某一种方法在所有场景下都更好。

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/05.jpg)

::: image-caption
该图表对比 MoonEP 与 DeepEP v2 在 MoE 通信场景下的时延表现，测试配置为 ep=8、E=384 等，分别展示 dispatch fwd、dispatch bwd、combine fwd、combine bwd 四类模块的时延，横轴 MaxVio 代表负载不均衡程度，数值越大负载越倾斜。堆叠柱状图拆分出通信、规划、预取等 MoonEP 开销，橙色为 DeepEP v2。随着负载失衡加剧，两种方案时延均上升，MoonEP 整体时延显著低于 DeepEP v2，体现其在不均衡专家负载下的通信性能优势，备注说明 grad‑reduce 未纳入本次基准测试。
:::

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/06.jpg)

::: image-caption
折线图对比 MoonEP 与 DeepEP 在 MoE 端到端训练下的归一化迭代耗时，横轴 target maxvio 代表专家负载不均衡程度，纵轴为归一化迭代时间。负载较轻时 DeepEP 性能略优，随着负载失衡加剧，DeepEP 迭代时间持续上涨，相比 MoonEP 性能分别下降 3.9%、7.6%、11.1%，在 maxvio=20 时出现 OOM 显存溢出报错。MoonEP 迭代耗时基本保持平稳，全程正常运行，体现出该方案在专家负载高度不均衡场景下，具备更稳定的训练性能与显存鲁棒性。
:::

参考资料

[1]

MoonEP: *https://github.com/moonshotAI/moonep*

[2]

EPLB（Expert Parallelism Load Balancer）: *https://github.com/deepseek-ai/eplb*

[3]

LPLB（Linear‑Programming‑Based Load Balancer）: *https://github.com/deepseek-ai/LPLB*

[4]

《Balancing and Beyond: Communication‑Centric Optimizations in Expert Parallelism》: *https://conferences.sigcomm.org/sigcomm/2026/program/papers/*

[5]

Kimi K2.6 官方配置: *https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json*

[6]

Kimi K3 官方说明: *https://github.com/MoonshotAI/Kimi-K3*

::: related-title
**相关推荐**
:::

::: related-links
- **[面向大规模专家并行 MoE 推理的故障容错方案 EEP，让 671B MoE 单节点故障停机从 348 秒压缩至 19 秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901916&idx=1&sn=bb370fd7a527a3da6a08327163854740&scene=21#wechat_redirect)**
- **[重新思考 FP4 预训练！收缩偏置才是 E2M1 FP4 训练不稳定的根源，UFP4 均匀格式将 124B MoE 损失降解降低 20%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902714&idx=1&sn=f461543fe701a86413ad7da6ccd7156a&scene=21#wechat_redirect)**
- **[前Google TPU架构师访谈：从批大小经济学到MoE物理布局，从流水线并行陷阱到RL对训练策略颠覆，从API定价技术到神经网络与密码学趋同进化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901219&idx=1&sn=95ebfde4d930adbfd424999bc41188e6&scene=21#wechat_redirect)**

:::
