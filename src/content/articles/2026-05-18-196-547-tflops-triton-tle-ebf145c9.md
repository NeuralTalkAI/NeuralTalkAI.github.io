关键词：**SparseMLA**、TLE（Triton Language Extensions）、**Warp Specialization**、:mdnice-bold-italic[生产消费模型]、FlashMLA

> SparseMLA 是 DeepSeek 系列模型的关键稀疏注意力算子，但在 Hopper GPU 上优化极为困难：稀疏索引导致不连续访存，宽向量 QK 计算与 PV 累积带来巨大的寄存器与同步压力，手写 CUDA 门槛极高。

![文章插图](/articles/2026-05-18-196-547-tflops-triton-tle-ebf145c9/images/01.webp)

- :mdnice-bold-italic[Triton-TLE：告别手动 barrier，用生产消费模型实现 FlashMLA 90% 的性能]
- 作者：SunnyCase，https://zhuanlan.zhihu.com/p/2039745753778804443
- 本文首发于 ❤️ 心爱的 Code House[1]
- **代码：https://github.com/flagos-ai/FlagTree/pull/592**
- 安装手册与文档：https://github.com/flagos-ai/FlagTree
- 本文 6000 字，阅读 33 分钟，播客 19 分钟

相关推荐

如何用 Triton-TLE 写出比 FlashInfer 更快的 TopK Selector

如何用 Triton 写出比 SGLang 更快的 MoE Align Block Size

微秒级决策，近最优性能！AMD 提出 tritonBLAS：告别耗时参数搜索，用“数学”算出最快矩阵乘法

> 本文提出基于 TLE（Triton Language Extensions）的两种优化方案：

- TLE-Pipe 复刻 TileLang-Pipelined 的 producer-consumer 流水线，将 KV 加载与左右半 V 计算拆分到三个 warp group，**性能从 Triton 基线的约 196–236 TFLOPS 提升至 382–484 TFLOPS（最高约 2×）；**
- TLE-FlashMLA 进一步用多条结构化 pipe 表达 FlashMLA 的 seesaw 双消费者协同模式，实现 QK 并行、max 交叉合并与交错 PV，**在 H800 上达到 FlashMLA 的 77%–90% 性能（最高 547 TFLOPS）。**

::: block-1
TLE 通过 `tle.pipe` 和 `warp_specialize` 将复杂的 warp 间生产消费关系抽象为可读的 acquire/commit/wait/release 语义，由编译器自动推导 mbarrier 同步，彻底免除手工配平屏障的负担。
:::

**这项工作证明，纯 Triton 语法即可接近手写 CUDA 的极致性能**，显著降低了在 Hopper 架构上探索先进数据流优化的工程成本。

## 本文目录

- 1. TL;DR
  - SparseMLA 是什么？
  - 为什么优化难？
  - TLE 带来了什么？
  - 各实现概览
- 2. SparseMLA 算法流程，以及朴素 Triton 怎么写
  - 核心循环代码
  - 重要细节
- 3. TileLang-Pipelined 的优化思路
- 4. TLE 是什么：用 TLE-Pipe 复刻 TileLang-Pipelined
  - TLE-Pipe 管道定义
  - 编排逻辑说明
  - Producer 核心代码
  - Left Consumer 核心代码
  - Right Consumer 核心代码
  - TLE-Pipe 实测性能
- 5. FlashMLA 的优化思路，以及用 TLE 复刻
  - TLE 实现 FlashMLA 管道声明
  - 线程组资源分配
  - 交叉同步核心逻辑
  - TLE-FlashMLA 完整性能对标
- 6. 总结

![文章插图](/articles/2026-05-18-196-547-tflops-triton-tle-ebf145c9/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 1. TL;DR

### SparseMLA 是什么？

> SparseMLA 是 DeepSeek 系列模型中最关键的算子之一。它对每个 query，只访问 TopK 个最相关的 KV token。它的 QK 计算被拆成两段：

- 主维度`D = 512` ：`q_main @ k_main`
- 尾部维度`TD = 64` ：`q_tail @ k_tail`
- 最终 score =`q_main @ k_main + q_tail @ k_tail` ，输出仍然是`softmax(score) @ v` 。

### 为什么优化难？

- 稀疏 indices 导致 KV gather 不连续，对 global memory 的访问很不友好。
- `D = 512` 太宽，一个 warp group 既算完整 QK 又做 PV，寄存器、shared memory 和同步很容易撞墙。

### TLE 带来了什么？

> TLE（Triton Language Extensions）让你**不用写 CUDA** ，就能把 TileLang-Pipelined 和 FlashMLA phase1 中那些复杂的 warp-specialized 数据流，直接写成可维护的 Triton 代码。

- 安装及详细手册：https://github.com/flagos-ai/FlagTree
- TLE Pipeline PR 和 SparseMLA 算子实现：https://github.com/flagos-ai/FlagTree/pull/592

尤其是在写 WASP / warp-specialized pipeline 时，你只需要声明不同 warp group 之间的**生产-消费关系** 。至于每个 mbarrier 需要多少`arrive_count` 、`cp.async` 的`arrive/commit/wait_group` 该怎么插入，编译器会从`pipe` 的读写语义自动推导。这对写过手写同步的人意味着：**再也不用翻 PTX 手册配平屏障了。**

### 各实现概览

> 测试环境：H800，`B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=2048` ，无`attn_sink`

| 实现 | 主要优化手段 | CTA / 并行形态 | H800 TFLOPS |
| --- | --- | --- | --- |
| Triton baseline | 每 K tile 直接从 global gather KV/tail，两段 QK、online softmax、PV | 普通 Triton program，8 warps，2 stages | 196.0 - 236.1 |
| TileLang-Pipelined | 显式 shared memory staging，producer/consumer 分工，V 左右拆半，score pipe 供右半输出复用 | ~384 threads，3 个 WG | 327.1 - 430.1 |
| TileLang-Seesaw | 两个 consumer 分别处理偶/奇 K block，交换 max/prob/sum，左右半输出交错累加 | ~384 threads，producer + WG0/WG1 seesaw | 394.9 - 452.8 |
| TLE-Pipe | 用 tle.pipe + warp_specialize 复刻 TileLang-Pipelined 数据流 | producer + left consumer + right consumer | 381.7 - 483.5 |
| FlashMLA | phase1 seesaw：两条 consumer 分别处理 K0/K1，互换 max/prob/sum，重叠两半 V 的 PV | CUDA/Hopper 专用 | 515.3 - 606.9 |
| TLE-FlashMLA | 用多个 TLE pipe 复刻 FlashMLA prefill seesaw 数据流，producer-last low-reg 映射 | consumer0 + consumer1 + producer | 396.9 - 546.8 |

![文章插图](/articles/2026-05-18-196-547-tflops-triton-tle-ebf145c9/images/03.webp)

::: image-caption
H800 GPU上不同SparseMLA实现的Prefill阶段吞吐量（单位：TFLOPS，越高越好），测试条件为B=1、S=4096、H=128，横轴为SKV值（8192~131072）。结果显示，FlashMLA（绿色） 性能始终领先，吞吐量在510~610 TFLOPS区间；TLE-FlashMLA（黑色） 次之，约395~545 TFLOPS；TLE-Pipe、TileLang-Seesaw、TileLang-Pipelined性能依次递减；Triton（红色）垫底，仅约195~235 TFLOPS。所有实现的吞吐量均随SKV增大而下降，其中FlashMLA在高SKV下的性能衰减也相对平缓，展现出显著的效率优势。
:::

从图中可以直观看到：Triton baseline 在最长的 SKV 上只有约 196 TFLOPS；TLE-Pipe 拉到约 382 TFLOPS，接近 2x；TLE-FlashMLA 再推到最高 547 TFLOPS，大约已经摸到 FlashMLA 的 0.77x - 0.90x。虽然还没追上，但这是在**纯 Triton 语法** 下做到的。

## 2. SparseMLA 算法流程，以及朴素 Triton 怎么写

> 先明确 kernel 的形状，所有后续优化都建立在同一个数据布局上。

- Query：`q: [B, SQ, H, DQK]`
- KV：`kv: [B, SKV, HKV, DQK]` ，这里`DQK = DV + TD = 512 + 64 = 576`
- TopK indices：`indices: [B, SQ, HKV, topk]`

对于 prefill 场景（`B=1, S=4096` ），每一个 query head block 要处理`topk=2048` 个 KV token。朴素 Triton 的做法是把`topk` 按`BK=64` 的 tile 逐个遍历，每个 tile 做：

1. 根据 indices 从 global 中 gather 出`kv_main` 和`kv_tail`
2. 计算`qk = q_tail @ tkv_tail^T + q_main @ kv_main^T`
3. online softmax 更新`max_prev / sum_exp / acc`
4. 累积`acc = softmax(score) @ kv_main`

### 核心循环代码

```
NK = tl.cdiv(topk_len, BK)for ck in tl.range(NK, num_stages=2):    t_ptr = BK * ck + offs_t    kv_ids = tl.load(t_base + t_ptr, t_ptr < topk_len, other=-1)    mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)    kv_blk = tl.load(kv_base + kv_ids[:, None] * stride_kvn + offs_d[None, :],                     mask_ids[:, None] & mask_d[None, :], other=0.0)    tkv_blk = tl.load(tkv_base + kv_ids[:, None] * stride_kvn + offs_td[None, :],                      mask_ids[:, None] & mask_td[None, :], other=0.0)    qk = tl.full([BH, BK], 0.0, dtype=tl.float32)    qk = tl.where(mask_ids[None, :], qk, float("-inf"))    qk = tl.dot(tq_blk, tl.trans(tkv_blk), qk, out_dtype=tl.float32)    qk = tl.dot(q_blk, tl.trans(kv_blk), qk, out_dtype=tl.float32)    new_max = tl.maximum(max_prev, tl.max(qk, axis=1))    alpha = tl.math.exp2((max_prev - new_max) * log_scale)    exp_qk = tl.math.exp2(qk * log_scale - new_max[:, None] * log_scale)    sum_exp = sum_exp * alpha + tl.sum(exp_qk, axis=1)    acc = tl.dot(exp_qk.to(tl.bfloat16), kv_blk, acc * alpha[:, None],                 out_dtype=tl.float32)    max_prev = new_max
```

### 重要细节

- `q_blk` 和`tq_blk` 在循环外只加载一次，但每个 K tile 都要重新 gather`kv_blk` 和`tkv_blk` 。gather 访存模式本身已经拉胯，再加上`DQK=576` 的大宽度，一个 warp group 既要从 global 搬数，又要做两段矩阵乘，最后还要把 exp 后的结果再乘一遍`kv_blk` ，寄存器压力极大。
- `qk` 不是完整的 attention matrix，而是`[BH, BK]` 的小块，online softmax 靠`max_prev` 和`sum_exp` 把每个 tile 串起来。
- `num_stages=2` 给了两层软件流水，但因为 gather 和 compute 都挤在同一个 warp group 里，pipeline 的效果很有限。实测中这个朴素实现作为 reference 是正确且清晰的，但性能只能在 196 - 236 TFLOPS 徘徊。

::: block-1
到这里 SparseMLA 已经能跑了。但如果看 SM 的硬件利用情况，会发现：真正在干活的 cycle 大概只有一半，另一半都在等 memory。要继续提速，就得把“搬运”和“计算”拆给不同 warp group，同时让左右两半 V 并行消费同一份 softmax 概率。
:::

## 3. TileLang-Pipelined 的优化思路

> TileLang-Pipelined 重新组织了每个 CTA 内的执行分工。它的核心变化是：

- 把`DV=512` 拆成左右两半`DPH=256` 。
- Q 的三个部分`q_l` 、`q_r` 和`q_tail` 只 staging 一次。
- 三个 warp group 各司其职：

| 泳道 | 角色 | 每个 ck 的动作 |
| --- | --- | --- |
| Producer WG | KV staging | acquire kv[ck] -> load indices -> load kv_l / kv_r / tkv / valid -> commit kv[ck] |
| Left Consumer WG | QK + softmax + 左半 PV | wait q -> wait kv[ck] -> q_tail@tkv + q_l@kv_l + q_r@kv_r -> online softmax -> acc_l += P@kv_l -> commit score[ck] |
| Right Consumer WG | 右半 PV | wait q -> wait kv[ck] -> wait score[ck] -> acc_r = acc_r*alpha + P@kv_r -> release |

![文章插图](/articles/2026-05-18-196-547-tflops-triton-tle-ebf145c9/images/04.webp)

::: image-caption
TLE-Pipe/TileLang-Pipelined的流水线数据流动，核心是WASP（warp-specialized）区域的并行协作：Producer工作组先一次性将Query（Q）加载到共享内存，随后由Producer分ck0-ck3四阶段异步加载稀疏KV数据；同时，Left/Right两个Consumer工作组并行执行计算，Left Consumer按序完成等待KV、QK点积、Softmax、结果计算与提交，Right Consumer则同步执行结果存储与后续点积，两者通过流水线调度实现数据加载（蓝色TMA操作）、矩阵乘（黄色dot）与CUDA Core同步计算（粉色操作）的完全重叠，大幅隐藏了内存访问延迟，提升SparseMLA的Prefill阶段吞吐量。
:::

一个 K tile 的两个半区（`kv_l` 和`kv_r` ）由 producer 一块儿准备好，left consumer 完整地算 QK 和 softmax，然后自己只用左半 V 做 PV；右半 V 的 PV 则交给 right consumer，它复用的是 left consumer 产出的`prob` 和`alpha / sum_exp` 。

::: block-1
这个设计很聪明：**它把算 QK 的 consumer 从 PV 的负担中解放出一半，同时让 idle 的 warp 也有活干。** 但代价是引入了显式的`score_pipe` ，左 consumer 要 commit probability/alpha，右 consumer 要 wait 后才开始累加自己的输出。这种跨 warp-group 的生产消费关系，用常规 Triton 是无法实现的。
:::

TileLang 版本用显式 shared memory + barrier 把它实现出来了，把 Triton baseline 从 200 TFLOPS 拉到 327 - 430 TFLOPS。接下来要解决的问题是：**能不能用更少的同步心智，在 Triton 里重新表达这套 producer / consumer 流水？**

这就是 TLE 要补的那块拼图。

## 4. TLE 是什么：用 TLE-Pipe 复刻 TileLang-Pipelined

> **TLE** （Triton Language Extensions）不是一个新的 DSL。你不用放弃熟悉的`tl.arange` 、`tl.load` 、`tl.dot` 。它补的是一层**结构化编排能力** ：shared memory allocation、local pointer view、typed pipe 和显式 warp specialization。

对开发者来说，最重要的两个 API 是：

- `tle.pipe(...)` —— 定义 CTA 内的一条数据流边，支持 SPSC / SPMC，payload 是 shared memory buffered tensor。写端用`acquire / commit` ，读端用`wait / release` 。每一个 pipe 同时是编译器推导 arrive / wait 关系的源级契约。
- `tle.gpu.warp_specialize(...)` —— 把一个 kernel 中的不同 JIT 函数分配到不同的 warp partition，可以指定每个 partition 的 warp 数和 requested registers。

### TLE-Pipe 管道定义

在 TLE-Pipe 实现里，我们直接把上文 TileLang-Pipelined 的数据流映射成三个 pipe 和三个 warp specialized 函数：

```
q_pipe = tle.pipe(    capacity=1, scope="cta", name="sparse_mla_q",    readers=("left", "right"), one_shot=True,    q_l=q_l_smem, q_r=q_r_smem, q_tail=q_tail_smem,)kv_pipe = tle.pipe(    capacity=PIPE_CAPACITY, scope="cta", name="sparse_mla_kv",    readers=("left", "right"),    kv_l=kv_l_pipe_smem, kv_r=kv_r_pipe_smem,    tkv=tkv_pipe_smem, valid=valid_pipe_smem,)score_pipe = tle.pipe(    capacity=1, scope="cta", name="sparse_mla_score",    alpha=score_alpha_smem, sum_exp=score_sum_exp_smem,)tle.gpu.warp_specialize(    [        (_tle_pipe_sparse_mla_producer, (...)),        (_tle_pipe_sparse_mla_left_consumer, (...)),        (_tle_pipe_sparse_mla_right_consumer, (...)),    ],    [4, 4],    [240, 168],)
```

### 编排逻辑说明

- `kv_pipe` 是一个 SPMC 管道，left 和 right consumer 都会等待同一个 KV tile。但 right reader 只取`kv_r` 字段，避免传递不需要的`kv_l` ，shared memory 不至于被无用的数据填满。
- `score_pipe` 只传`alpha` 和`sum_exp` ，概率`prob` 本身放在一个单独的 shared buffer 里，走另一个路径。这样 TLE 编译器可以只针对 pipe fields 生成读端和写端的 mbarrier，而不是把整个 buffer 打包交给一个粗粒度的 barrier。
- `warp_specialize` 中 producer 是 default partition，两个 consumer 是 worker，各自的 warp 数和寄存器预算也直接写在了参数里。这比自己去数`__launch_bounds__` 要省心得多。

### Producer 核心代码

> producer / consumer 里面真正的 pipe API。下面是删掉地址计算后的核心结构：

```
@triton.jitdef _tle_pipe_sparse_mla_producer(kv_writer, kv_base, tkv_base, t_base, ...):    for ck in tl.range(NK):        # writer.acquire(ck) 取得这一轮可写的 shared-memory slot。        kv_slot = kv_writer.acquire(ck)        kv_l_smem_ptr = tle.gpu.local_ptr(kv_slot.kv_l)        kv_r_smem_ptr = tle.gpu.local_ptr(kv_slot.kv_r)        tkv_smem_ptr = tle.gpu.local_ptr(kv_slot.tkv)        valid_smem_ptr = tle.gpu.local_ptr(kv_slot.valid)        kv_ids = tl.load(t_base + BK * ck + offs_t, mask=t_msk, other=-1)        kv_l_blk = tl.load(kv_l_ptr, mask=kv_l_msk, other=0.0)        kv_r_blk = tl.load(kv_r_ptr, mask=kv_r_msk, other=0.0)        tkv_blk = tl.load(tkv_ptr, mask=tkv_msk, other=0.0)        tl.store(kv_l_smem_ptr, kv_l_blk, mask=kv_l_msk)        tl.store(kv_r_smem_ptr, kv_r_blk, mask=kv_r_msk)        tl.store(tkv_smem_ptr, tkv_blk, mask=tkv_msk)        tl.store(valid_smem_ptr, mask_ids.to(tl.int32))        # commit 后，等待这个 ck 的 reader 才能继续。        kv_writer.commit(ck)
```

### Left Consumer 核心代码

left consumer 对 `kv_pipe.reader("left")` 调 `wait`，拿到 producer 刚 commit 的 slot；算完 QK、softmax 和左半输出后，再把右 consumer 需要的状态写入 `score_pipe`：

```
@triton.jitdef _tle_pipe_sparse_mla_left_consumer(q_reader, kv_left_reader,                                       score_writer, score_prob_smem, ...):    q_slot = q_reader.wait(0).slot    q_l = tl.load(tle.gpu.local_ptr(q_slot.q_l))    q_r = tl.load(tle.gpu.local_ptr(q_slot.q_r))    q_tail = tl.load(tle.gpu.local_ptr(q_slot.q_tail))    for ck in tl.range(NK):        kv_wait = kv_left_reader.wait(ck)        kv_slot = kv_wait.slot        kv_l = tl.load(tle.gpu.local_ptr(kv_slot.kv_l))        kv_r = tl.load(tle.gpu.local_ptr(kv_slot.kv_r))        tkv = tl.load(tle.gpu.local_ptr(kv_slot.tkv))        valid = tl.load(tle.gpu.local_ptr(kv_slot.valid)) != 0        qk = tl.dot(q_tail, tl.trans(tkv), out_dtype=tl.float32)        qk = tl.dot(q_l, tl.trans(kv_l), qk, out_dtype=tl.float32)        qk = tl.dot(q_r, tl.trans(kv_r), qk, out_dtype=tl.float32)        prob, alpha, sum_exp = online_softmax_update(qk, valid)        acc_l = tl.dot(prob, kv_l, acc_l, out_dtype=tl.float32)        score_slot = score_writer.acquire(ck)        tl.store(tle.gpu.local_ptr(score_slot.alpha), alpha)        tl.store(tle.gpu.local_ptr(score_slot.sum_exp), sum_exp)        tl.store(tle.gpu.local_ptr(score_prob_smem), prob)        score_writer.commit(ck)        # release 告诉 kv_pipe：left reader 已经不再使用这个 slot。        kv_left_reader.release(ck)
```

### Right Consumer 核心代码

right consumer 则只拿自己需要的字段。这里 `kv_pipe.reader("right", fields=("kv_r",))` 会让编译器知道它只消费 `kv_r`，不需要把 `kv_l/tkv/valid` 也纳入这个 reader 的等待/可见性关系：

```
@triton.jitdef _tle_pipe_sparse_mla_right_consumer(q_reader, kv_right_reader,                                        score_reader, score_prob_smem, ...):    q_slot = q_reader.wait(0).slot    q_r_smem_ptr = tle.gpu.local_ptr(q_slot.q_r)    for ck in tl.range(NK):        kv_wait = kv_right_reader.wait(ck)        score_wait = score_reader.wait(ck)        kv_r = tl.load(tle.gpu.local_ptr(kv_wait.slot.kv_r))        prob = tl.load(tle.gpu.local_ptr(score_prob_smem))        alpha = tl.load(tle.gpu.local_ptr(score_wait.slot.alpha))        sum_exp = tl.load(tle.gpu.local_ptr(score_wait.slot.sum_exp))        acc_r = acc_r * alpha[:, None]        acc_r = tl.dot(prob, kv_r, acc_r, out_dtype=tl.float32)        score_reader.release(ck)        kv_right_reader.release(ck)    tl.store(q_r_smem_ptr, (acc_r / sum_exp[:, None]).to(OUT_DTYPE))
```

这就是 TLE-Pipe 最核心的可读性：`writer.acquire -> store payload -> writer.commit` ，consumer 侧`reader.wait -> load payload -> reader.release` 。同步语义就在这些 API 里，不需要在业务代码旁边手写一堆 barrier phase 和 arrive 数量。

### TLE-Pipe 实测性能

| SKV | Triton | TileLang-Pipelined | TLE-Pipe |
| --- | --- | --- | --- |
| 8192 | 236.1 | 430.1 | 483.5 |
| 32768 | 208.4 | 363.5 | 439.1 |
| 65536 | 200.5 | 340.3 | 407.7 |
| 98304 | 197.4 | 327.3 | 391.2 |
| 131072 | 196.0 | 327.1 | 381.7 |

TLE-Pipe 没有改变 SparseMLA 的任何数学步骤，只是把“加载 KV”和“两半输出计算”拆给了不同 warp group。结果它超过了 TileLang-Pipelined，并把 Triton baseline 从约 196 - 236 TFLOPS 推到了约 382 - 484 TFLOPS。

## 5. FlashMLA 的优化思路，以及用 TLE 复刻

> 如果说 TileLang-Pipelined 的优化是“把 PV 拆给两个 consumer”，那 FlashMLA phase1 做得更激进：**每个 consumer 都参与 QK 计算，而且 K tile 成对处理，两个 consumer 做 seesaw** 。

一个 pair 包括`k0` 和`k1` ，数据流大致如下：

![文章插图](/articles/2026-05-18-196-547-tflops-triton-tle-ebf145c9/images/05.webp)

::: image-caption
TLE-FlashMLA Prefill Seesaw 数据流架构，核心是Producer工作组与WG0/WG1两个消费者工作组的协同流水线：Producer先加载稀疏索引并分阶段异步提交K0、K1的左右半部分；WG0/WG1分别负责左、右半部分输出，同时交叉计算对方的K分片（WG0算K0L、WG1算K1R，再交换远程分片），通过max、probability、sum三次边界交换同步Softmax相关状态，最终完成结果输出。该设计让数据加载（蓝色TMA操作）、矩阵乘（黄色dot）与CUDA Core同步计算（粉色操作）完全重叠，大幅隐藏内存延迟，提升稀疏MLA Prefill阶段的并行效率。
:::

| 泳道 | 角色 | 关键块 |
| --- | --- | --- |
| Producer | sparse gather / staging | ids0/ids1 -> K0L -> K1R+tail -> K0R+tail -> K1L -> valid |
| WG0 | K0 owner，left output owner | QK0 -> local max0 -> sM0 -> P0*K0L -> wait sM1 -> scale P0 -> sS0 -> wait sS1/K1L -> P1* K1L -> sum exchange -> O_l |
| WG1 | K1 owner，right output owner | QK1 -> wait sM0 -> max_next/sM1 -> P1*K1R -> sS1 -> wait sS0/K0R -> P0* K0R -> sum exchange -> O_r + lse |

不再是一个 consumer 负责完整 score 然后“施舍”给另一个；而是**QK 并行做，max 交叉合并，PV 两半交错吃** 。这极大提升了 compute 和 data movement 的重叠，但对应的同步网比 TileLang-Pipelined 复杂得多：`sM` 、`sS` 、`sL` 多条小管道在 pair 间交替使用。

### TLE 实现 FlashMLA 管道声明

> 用 TLE 写这个 seesaw，pipe 声明几乎就是数据流图的直接翻译：

```
k0_l_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sK0_l", sK=sK0_smem)k0_r_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sK0_r",                     readers=("qk", "remote"), sK=sK0_smem, sK_tail=sK0_tail_smem)k1_l_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sK1_l",                     readers=("qk", "remote"), sK=sK1_smem)k1_r_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sK1_r",                     sK=sK1_smem, sK_tail=sK1_tail_smem)sM_wg0_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_wg0_bunch_0_ready", sM=sM_smem)sM_wg1_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_wg1_bunch_0_ready", sM=sM_smem)sS0_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sS0", sS0=sK0_tail_smem)sS1_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sS1", sS1=sS1_smem)
```

### 线程组资源分配

> warp specialization 的分配也体现了资源权衡：

```
tle.gpu.warp_specialize(    [        (_tle_flashmla_prefill_consumer0, (...)),        (_tle_flashmla_prefill_consumer1, (...)),        (_tle_flashmla_prefill_producer, (...)),    ],    [4, 4],    [216, 72],)
```

这里第一个函数是 default partition，因此`consumer0` 位于 default partition；producer 被放到最后一个 worker 并分配了较少的寄存器（72）。这是因为 producer 只负责稀疏 gather 和 staging，计算量远小于两个 consumer，把寄存器留给更需要的人。

### 交叉同步核心逻辑

> seesaw 中的同步代码也极具可读性，比如 WG0 发布 local max 然后等待合并后的 max_next：

```
# WG0: publish local max for k0sM_wg0_slot = sM_wg0_writer.acquire(pair)tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)sM_wg0_writer.commit(pair)sM_wg1_wait = sM_wg1_reader.wait(pair)max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))final_scale = tl.math.exp2((local_max - max_next) * log_scale)prob0_scaled = prob0 * final_scale[:, None]
```

WG1 的合并逻辑：

```
# WG1 合并逻辑sM_wg0_wait = sM_wg0_reader.wait(pair)candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))max_next = tl.maximum(candidate1, candidate0)sM_wg1_slot = sM_wg1_writer.acquire(pair)tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)sM_wg1_writer.commit(pair)
```

不需要手动管理`mbarrier` 的`arrive_count` ，也不用去数这个 barrier 究竟有几个 named reader。编译器从 pipe 的`readers` 声明自动推导。

::: block-1
这个实现非常完整地证明了：**FlashMLA 级别的 WG 协同数据流，可以在 Triton 中用结构化 pipe 表达出来。**
:::

### TLE-FlashMLA 完整性能对标

> 性能上，TLE-FlashMLA 把基线从 196 - 236 TFLOPS 推到 397 - 547 TFLOPS，大约达到手写 FlashMLA 的 0.77x - 0.90x：

| SKV | Triton | TLE-Pipe | TLE-FlashMLA | FlashMLA | TLE-FlashMLA / FlashMLA |
| --- | --- | --- | --- | --- | --- |
| 8192 | 236.1 | 483.5 | 546.8 | 606.9 | 0.90x |
| 32768 | 208.4 | 439.1 | 483.3 | 567.5 | 0.85x |
| 65536 | 200.5 | 407.7 | 440.4 | 537.4 | 0.82x |
| 98304 | 197.4 | 391.2 | 417.3 | 525.4 | 0.79x |
| 131072 | 196.0 | 381.7 | 396.9 | 515.3 | 0.77x |

TLE-FlashMLA 并不是“神奇地自动等于手写 CUDA FlashMLA”。**它真正证明的是：FlashMLA 这类复杂 WG 数据流，可以在 Triton kernel 里用 TLE 的结构化 pipe 表达出来，且性能已经非常接近**。对于不想常年抱着 CUDA 和 Hopper 手册的工程师来说，这条路径显著降低了尝试极致优化数据流 kernel 的门槛。

## 6. 总结

1. **SparseMLA 算法不复杂，难的是让 sparse gather、QK、softmax 和 PV 在 Hopper 的 warp group 之间流起来。** 没有合适的编排工具，开发者很容易在“先算哪一半、在哪同步”里踩坑。
2. **TLE-Pipe 证明了 TileLang-Pipelined 这种 producer/consumer 分工可以用 Triton + TLE 清晰表达，并带来接近 2x 的实际性能提升。** 写 pipe 和 warp_specialize 比手拼 mbarrier 直观一个数量级。
3. **TLE-FlashMLA 进一步证明，FlashMLA phase1 的 seesaw 双 consumer 同步模式也能用 TLE 复刻。** 它不是对手写 CUDA 的全面替代，但已经把复刻成本从“翻几百行 PTX 和 CUDA barrier 代码”降低到“写几十行 pipe 声明 + 几个消费者函数”。

::: block-1
如果你已经能写 Triton attention kernel，TLE 要你多理解的不是一门新语言，而是三件事：**shared memory buffer、typed pipe、warp partition。**
:::

SparseMLA 正好说明，这三件事足够把一个朴素 kernel 推到 FlashMLA 附近。

参考资料

[1] 心爱的 Code House: *https://www.zhihu.com/column/c_1015646974233718784*
