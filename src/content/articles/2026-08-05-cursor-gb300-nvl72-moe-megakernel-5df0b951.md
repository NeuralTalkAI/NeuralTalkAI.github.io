**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：MoE、**megakernel**、NVL72、:mdnice-bold-italic[计算通信重叠]、专家并行

> Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel——Mixture-of-Kittens（MoK），将全部 MoE 通信与计算融合进单个完全确定性的 kernel，已支撑横跨数万张 GPU 的 Composer 模型训练。

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/01.png)

- **Mixture-of-Kittens: our open-source MoE megakernel for NVL72s**
- https://github.com/cursor/mixture-of-kittens
- https://cursor.com/cn/blog/mixture-of-kittens

::: related-title
**相关推荐**
:::

::: related-links
- **[64块H100媲美96块！开源MoE软硬件协同加速方案SonicMoE：基于IO和tile感知优化实现45%内存节省+1.86倍吞吐量！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894859&idx=1&sn=c50a765749e1cb035469b0fa38334033&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**
- **[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903011&idx=1&sn=7872d080a5475a44bba4cfabe7717b56&scene=21#wechat_redirect)**

:::

> 今天，我们正式开源 Mixture-of-Kittens（MoK）[1]——**我们面向 NVL72 的生产级 MoE 训练 megakernel。**

随着我们的智能体编程模型 Composer[2] 在训练和推理规模上不断扩大，:mdnice-bold-italic[混合专家（mixture-of-experts，MoE）层始终是最大的瓶颈。视工作负载和训练配置而定，它可能消耗端到端训练时间的一半以上。]

::: block-1
:mdnice-highlight[MoK 通过将全部 MoE 通信与计算融合进单个完全确定性的 kernel 来解决这一瓶颈。它目前支撑着横跨数万张 GPU 的 Composer 训练]。你可以在 GitHub[3] 上试用 MoK 并浏览代码。我们期待你的反馈与贡献。MoK 诞生于此前多次加速 MoE 层的尝试。过去一年里，我们编写了自己的 MXFP8[4] 和 NVFP4 训练 kernel，并开发了面向 MoE 推理的 "warp decode"[5] 方案。
:::

但这些技术只优化了该层的计算部分，并假设 GPU 间通信会被单独处理。而在我们的生产工作负载中，通信已经成为限制因素。**这促使我们从第一性原理出发重新设计整个 MoE 层，把通信直接构建进 kernel 之中。**

此外，我们迁移到 GB300 NVL72 平台，从两个重要方面改变了问题。

- 第一，NVL72 是一个处于单一 NVLink 域内的多节点机架，使得全部 72 张 GPU 之间能够快速、细粒度地重叠计算与通信。
- 第二，集成的 Grace CPU（GB300 中的"G"）相对于 GPU 而言偏慢。我们发现 :mdnice-bold-italic[GPU 流很容易追上 CPU 侧的工作，导致 GPU 在这段时间内完全空转。因此我们必须激进地压缩 CPU 工作量和 CPU-GPU 同步。]

我们针对这一系列挑战给出的答案就是 Mixture-of-Kittens（MoK）——一个从第一性原理出发、为 NVL72 高度优化的 MoE 训练 megakernel。**MoK 将全部 MoE 通信与计算融合进单个 kernel，完全确定，并且相对公开可用的实现达到了最先进的性能。**

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/02.jpg)

::: image-caption
在 GB300 NVL72 上，Mixture-of-Kittens 的 MXFP8 前向吞吐量最高达到最快公开基线的 2.37 倍。对比对象包括：NCCL + PyTorch、DeepEP + PyTorch、DeepEP + TransformerEngine、HybridEP + Megatron、Mixture-of-Kittens。
:::

在我们横跨多个 NVL72 机架的生产训练栈中，MoK 将端到端每秒 token 数提升了 1.41 倍。

::: block-1
本文的其余部分将讲解 MoK 背后的关键思想，包括：:mdnice-highlight[我们如何选定正确的通信方向、如何组织计算与通信之间的重叠、以及如何用环形 token 缓冲区消除 CPU-GPU 同步]。我们还会介绍 **megakernel 设计、确定性、MXFP8 支持，以及其他若干实现细节。**
:::

## 本文目录

- 一、MoE 中计算与通信的重叠
- 二、选择正确的通信方向
  - 2.1 调度
  - 2.2 NVLink 带宽利用率
  - 2.3 信号传递
- 三、合理组织计算-通信重叠的结构
  - 3.1 计算-通信粒度
  - 3.2 重叠的调度
- 四、用环形 token 缓冲区彻底消除 CPU-GPU 同步
  - 4.1 反转环形缓冲区，最小化反向时的前向重放
- 五、其他特性与实现细节
  - 5.1 Megakernel
  - 5.2 确定性
  - 5.3 用于 RDMA 重叠的 Cluster Launch Control（CLC）
  - 5.4 MXFP8 支持
  - 5.5 路由权重梯度计算
  - 5.6 计算/通信 SM 调优
- 六、性能提升
  - 6.1 MoE 层基准测试
  - 6.2 端到端基准测试
- 七、面向智能体时代的 Megakernel

## 一、MoE 中计算与通信的重叠

> MoK 面向 DeepSeek-V3（DSV3）风格的 MoE 层，这类结构被 GLM、Qwen、Kimi（截至 K2.7）以及 DSV 本身等开放权重模型广泛采用。这些层将一个共享专家与大量路由专家（往往多达数百个）组合在一起。

对于进入 MoE 层的每个 token，一个路由投影会选出 top-k 个路由专家，并为每个专家分配一个路由权重。随后，每个被选中的专家执行标准的前馈网络（FFN）计算，包括 up 投影与 gate 投影、SwiGLU 激活以及 down 投影。该层再按照路由权重，将共享专家与路由专家的输出组合起来。

后文我们公式中会使用如下记号：

| 符号 | 含义 |
| --- | --- |
| $D$ | 模型维度 |
| $I$ | 专家中间维度 |
| $E$ | 路由 top-k 专家的集合 |
| 输入 token $x \in \mathbb{R}^{D}$ | 输入 token，维度为 $D$ 的实向量 |
| 路由权重 $s \in \mathbb{R}^{|E|}$ | 对应于集合 $E$ 中每个专家的路由权重，长度为 $|E|$ |
| 专家权重 $W_{\text{up}} \in \mathbb{R}^{I \times D}$ | 上投影权重矩阵，将输入从 $D$ 维映射到 $I$ 维 |
| 专家权重 $W_{\text{gate}} \in \mathbb{R}^{I \times D}$ | 门控权重矩阵，与 $W_{\text{up}}$ 同形，用于计算激活门控 |
| 专家权重 $W_{\text{down}} \in \mathbb{R}^{D \times I}$ | 下投影权重矩阵，将中间表示从 $I$ 维映射回 $D$ 维 |

> MoE 层的计算为：

$$
\text{MoE}(x, s) = E_{\text{shared}}(x) + \sum_{i \in E} g_i \, E_i(x)
$$

其中：

$$
E_i(x) = W_{\text{down}}^{(i)}\left(\text{SiLU}(W_{\text{gate}}^{(i)} x) \odot W_{\text{up}}^{(i)} x\right) \quad \text{且} \quad g_i = \frac{s_i}{\sum_{j \in E} s_j}
$$

**在专家并行（EP）下，我们把路由专家切分，将它们的权重摊到多张 GPU（即多个 rank）上，并把共同持有全部专家权重的 rank 数量称为 EP 度**。例如，有 256 个路由专家、EP 度为 64 时，每个 rank 持有 4 个路由专家，外加共享专家。这样一来，token 就必须按照路由投影的结果，在 MoE 层前后跨 GPU 传输。

分布式 MoE 最直接的实现方式是：把每个 token 发送到持有其指定专家的 rank（dispatch all-to-all），执行 FFN，再把结果送回各 token 原本的 rank（combine all-to-all），最后对专家输出做加权和。但由于通信耗时可能与计算本身相当，将两者串行执行是低效的。

::: block-1
标准的补救办法是通过流水线把 dispatch/combine 通信与逐专家 FFN 重叠起来：先传输一块 token，在其上计算 FFN 的同时重叠传输下一块，如此往复。:mdnice-highlight[MoK 正是这一方案的一个变体，并配有一组新颖的、针对特定硬件的技术，使其快于现有基线。]
:::

*注: 在本文中，dispatch 指 dispatch all-to-all 通信，combine 指 combine all-to-all 通信。通常包含在"combine"中的加权归约，在本文中被单独称为"加权和"（weighted sum）。*

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/03.png)

::: image-caption
MoE 前向中计算与通信的重叠示意图（上：串行执行 Dispatch → Expert FFNs → Combine；下：通信与计算分块重叠）
:::

## 二、选择正确的通信方向

在跨 GPU 发送 token 时，可以选择基于推送（push）的机制——由拥有 token 的 GPU 主动将其写入远端目标 GPU；也可以选择基于拉取（pull）的机制——由需要 token 的 GPU 从远端源 GPU 读取。**现有方案通常依赖 push 式通信来在 GPU 间散布和收集 token（例如 DeepEP）。**

> **常见的观念是，push 涉及的协议通信更少，因而能更好地打满 GPU 间链路，于是成为默认选择**。:mdnice-bold-italic[但我们的观察是，每种机制各有取舍]，为每个通信算子选对机制对最大化性能至关重要，理由有以下三点。

### 2.1 调度

为了:mdnice-bold-italic[尽可能快地 dispatch 和 combine token，必须满足以下条件：]

1. :mdnice-bold-italic[机架内互连 72 张 GPU 的所有 NVLink 通道必须保持打满]。我们不能容忍 token 只在一部分（源 → 目的地）通道上传输的时间段，因此在任意时刻，必须让每张源 rank 的发送均匀散布到所有目标 rank 上。
2. :mdnice-bold-italic[发往某个 rank 的 token 应按该 rank 本地专家的顺序到达]。如果到达顺序是乱的，专家分组 GEMM 就要等更久才能凑满一个完整的 token tile，张量核矩阵乘法才能开始。
3. :mdnice-bold-italic[本地零拷贝]。属于某个专家的 token 应直接落进连续内存，分组 GEMM 无需在本地重排任何东西即可开始。
4. :mdnice-bold-italic[满足上述三个条件的开销必须保持最小]。我们希望绝大多数时间花在真正发送 token 上，而不是花在调度或寻找待发送的 token 上。

::: block-1
使用 push 式 dispatch 时，我们需要生成一张调度表，列为 `{src_index, dst_rank, dst_index}`，其中 `src_index` 是本地输入激活缓冲区的索引，`dst_index` 是 token 在目标 rank 内存中必须落到的位置。

该表的行索引决定 token 经 NVLink 发送的顺序。:mdnice-highlight[我们希望表的行按 `dst_rank` 轮转循环，从而让每条互连通道都保持繁忙。]
:::

在指向同一 `dst_rank` 的那些行中，`dst_index` 的值必须在所有源 GPU 之间均匀交错，同时还必须遵循目标 rank 上本地专家递增的顺序。构建这张表涉及多次排序，而且每个 rank 的调度都必须考虑其他所有 rank，因为任意两个源 rank 都不能写入同一个 `dst_index`。

| Schedule Idx | Src Idx | Dst Rank | Dst Idx |
| --- | --- | --- | --- |
| 0 | 17 | 0 | 0 |
| 1 | 0 | 1 | 0 |
| 2 | 29 | 2 | 0 |
| 3 | 3 | 3 | 0 |
| … | … | … | … |

*上表是 Push 式调度：相邻表项必须交错目标 rank，才能让网络完全打满。*

| Schedule Idx | Src Rank | Src Idx |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 5 |
| 2 | 2 | 13 |
| 3 | 3 | 7 |
| … | … | … |

*上表是 Pull 式调度：相邻表项必须交错源 rank，才能让网络完全打满。*

> 使用 pull 式 dispatch 时，调度表简化为两列 `{src_rank, src_index}`，且表的行索引直接对应本地目标缓冲区中的 token 索引。

理论上，表的行数与目标缓冲区的行数一致（这需要较大的内存分配，后文会详述）。这里不需要任何排序。我们遍历路由投影的结果，每当发现一个应落到当前 rank 的 token，就把它的源 rank 和索引写入我们的调度表。算法如下：

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/04.png)

算法 1：Pull-based Dispatch 调度表生成完整 GPU 设备端算法。输入为全局路由张量 E、本地 GPU 编号 c、单卡专家数量 L；输出为各本地专家 token 计数 T、源 GPU-token 二维计数矩阵 M、专家缓冲区偏移 S、最终 Pull 调度表 Φ；遍历全部 token 路由结果统计分布，计算缓冲区偏移，按源 GPU 交错规则填充调度表，全程无 CPU 参与。该算法是 Pull 通信架构的底层支撑，完全卸载至 GPU 核函数执行，规避 GB300 Grace CPU 算力短板。算法仅单次遍历路由张量，无多轮全局排序，时间复杂度远低于 Push 调度构建逻辑；填充调度表时按专家本地顺序排布 token，使远端传输数据直接落入连续显存，消除本地重排拷贝开销。生成的调度表可复用至前向、反向全部通信算子，减少显存读写；同时天然规避跨 GPU 信令等待，从算法层面解决 72 卡 NVLink 域同步延迟问题，整套调度开销控制在总 MoE 耗时 3% 以内，是 MoK 通信层创新核心。

实践中，我们实现该算法的调度 kernel 仅占 MoE 总运行时的不到 3%，并且完全在设备侧运行，不涉及任何 CPU-GPU 通信。

通过选择 push 式 combine，我们还可以原样复用这份调度——只需把 `{src_rank, src_index}` 读作 `{dst_rank, dst_index}`。事实上，只需构建一次调度，就能复用于前向与反向的全部四个通信操作，具体组合为：

- Pull 式前向 dispatch
- Push 式前向 combine
- Pull 式反向 reverse-combine
- Push 式反向 reverse-dispatch

这份调度在**最坏情况下也只有几 MB，因此可以一直留着复用，毫无内存压力**。另一个额外好处是，它**完全消除了跨 GPU 的多通道信号传递**，后文会解释这一点。

### 2.2 NVLink 带宽利用率

> 与任何网络系统一样，经 NVLink 传输的所有用户数据（载荷）都会附带额外的协议元数据，其中包含源与目的地信息、确认信息等，具体取决于网络协议。虽然 NVLink 通信协议的细节并未公开，但我们可以观察链路上实际发送的数据并仔细推理。

我们发现，push 式 NVLink 通信搬运的总字节数（即协议元数据加载荷）更少，而且几乎所有流量都朝一个方向走，因此在所有通道都满载时带宽利用率更高。

另一方面，pull 式传输搬运的总字节数更多，但协议元数据在链路的两个方向之间分摊得更均匀：拉取方先朝一个方向发送元数据，再从另一个方向接收更多元数据加载荷。

我们可以写一个简单的跨 GPU 传输 kernel，并用 NCU 做性能分析来验证这一点。例如，在我们的微基准测试中，经 NVLink 发送一个 256×256 的 BF16 tile（131,072 字节），观测结果如下：

|  | 总计 | 总 RX | 总 TX | 协议 RX | 协议 TX | 载荷 RX | 载荷 TX |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Push | 159.6 KB | 2.9 KB（1.84%） | 155.6 KB（99.16%） | 2.7 KB | 24.6 KB | 0 KB | 131.1 KB |
| Pull | 172.0 KB | 147.5 KB（85.71%） | 24.6 KB（14.29%） | 16.4 KB | 24.6 KB | 131.1 KB | 0 KB |

上表中，RX 指接收方向，TX 指发送方向。**可以看到，push 的总字节数大约少 12.4 KB，理想情况下能带来更好的整体 NVLink 饱和度**。然而，NVLink 的每个方向都有独立通道。:mdnice-bold-italic[要达到 NVLink 规格标称的带宽（GB300 上第五代 NVLink 为 1.8 TB/s），流量必须双向同时打满]。而对于 MoE dispatch/combine 这样的工作负载，每次传输往往只有几千字节且并不均衡，会在链路上留下执行气泡。

::: block-1
根据实测，:mdnice-highlight[在专家负载不均衡的情况下 dispatch token 时，pull 式通信的 NVLink 带宽利用率最高比 push 式通信高出 29%，因而是更好的选择。]
:::

### 2.3 信号传递

> 信号传递是必要的，因为 GPU 必须知道 dispatch/combine 通信何时完成。GPU 只有在收到完成信号之后，才能开始处理接收到的 token。最小化这部分信号开销很重要，而选择正确的通信方向对此有显著影响。

- 使用 push 式 dispatch 或 pull 式 combine 时，完成信号必须跨 GPU 传递。在专家分组 GEMM 开始之前，一个 rank 必须等待来自多达 71 个对端的信号，并通过发出内存栅栏（memory fence）在整个机架范围内冲刷内存。此外，在信号传输的时间里，数据其实早已待在目的地了。
- 而使用 pull 式 dispatch 和 push 式 combine 时，跨 GPU 信号传递消失了。每个 rank 发起一次加载，等数据到达后即可立即使用，无需任何多节点同步。同时，只有单一主体执行信号传递，因此该开销不会随专家并行度的增长而增长。

我们发现这种信号开销相当大。在我们的多节点微基准测试中，push 式 dispatch 的信号延迟约为 pull 式 dispatch 的 5.8 倍——103 µs 对 18 µs——而这一成本在 megakernel 内会迅速累积。

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/05.png)

::: image-caption
Push 与 Pull 式 dispatch 对比图（左：push 式，GPU 0 向各 GPU 推送 token 并发送信号；右：pull 式，GPU 0 从各 GPU 拉取 token，信号在本地完成）。具体来说，左侧推送模式下，GPU 0 主动向 GPU1、GPU2、GPU3 推送 token 数据，同步发送信号通知各接收 GPU。右侧拉取模式中，GPU1、GPU2、GPU3 主动向 GPU 0 拉取 token，GPU 0 接收信号。两种方式数据流向相反，推送由源 GPU 发起传输，拉取由目标 GPU 主动请求数据，清晰展示两类 MoE token 分发的通信交互逻辑差异。
:::

::: block-1
基于这些结果，:mdnice-highlight[MoK 采用：pull 式前向 dispatch、push 式前向 combine、pull 式反向 reverse-combine、push 式反向 reverse-dispatch。]
:::

## 三、合理组织计算-通信重叠的结构

### 3.1 计算-通信粒度

> 计算-通信粒度，指的是计算任务与通信任务相互通知对方"已完成"的粒度。它带来了一个有趣的权衡。

在一个极端，通信可以非常细粒度：我们只发送足够张量核发射一条完整矩阵乘累加指令的 token——在 Blackwell 上是 256 个——并持续地把每次传输与矩阵乘法重叠起来。Comet[6] 就是这条路线的代表。

在另一个极端，通信可以是粗粒度的：一次发送数千乃至数万个 token，等整批全部到达后才开始张量核运算。DeepEP[7] 是这方面的代表。

我们的观察是，**两个极端都不是正确选择。最优点位于中间的某个位置，并且取决于工作负载。**

如果通信过细，张量核永远无法完全打满。英伟达的张量核是深度流水化、为带宽优化的加速器，**我们希望持续喂饱它们，而不是每隔几条 MMA 指令就撞上一次屏障等待。**

如果通信过粗，张量核又会长久地等待第一轮 token 到达、等待最后一轮输出被合并。

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/06.png)

::: image-caption
MoE 前向传播计算与通信重叠方案对比，分为细粒度（Fine-grained）与粗粒度（Coarse-grained）上下两组时序图，横轴代表时间。细粒度模式下通信拆分为四段 Dispatch，在首段调度空闲期结束后，依次启动三段专家计算。粗粒度模式仅有两段长 Dispatch，长时间空闲后仅执行一轮专家计算。两种方案均存在初始空闲时段，细粒度划分能够更早启动专家 FFNs 计算，实现通信与计算更大程度重叠。
:::

我们把每轮发送的 token 集合称为一个 minibatch。**MoK 把 minibatch 大小做成可调参数**，方便在运行工作负载之前扫掠各种可能的配置。

> 选择最优 minibatch 大小的一个粗略启发式方法，是考虑每个算子（例如专家分组 GEMM）需要多少个波（wave）。

一个"波"是 GPU 上所有 SM 并发执行的一轮。如果一项工作太小（例如专家分组 GEMM 收到的 token 太少），就会产生一个不完整的波，其中一些 SM 空转。另一种情况是出现拖尾效应：最后一个波只占用一小部分 SM。

::: block-1
:mdnice-highlight[我们发现，让 FFN 中每个专家分组 GEMM 算子至少获得两个完整的波，是一个好用的 minibatch 大小选择标准]。这有两个原因：

1. 一个完整的波就足以完全打满张量核；
2. 有两个波时，第二个波的 MMA 会与第一个波的收尾（epilogue）及依赖算子（例如 SwiGLU 激活或下一个分组 GEMM）重叠，从而摊薄拖尾效应。
:::

具体来说，在 Blackwell GPU 上，要让一个 $M \times N \times K$ 的 GEMM（$K$ 为归约维度）充分利用张量核，无论使用 1-SM 还是 2-SM MMA，每个 SM 都要处理一个 $128 \times 256$ 的输出 tile。

因此，给定：

| 符号 | 含义 |
| --- | --- |
| $T$ | 每次传输的 token 数（即 minibatch 大小） |
| $H$ | 隐藏维度 |
| $I$ | 专家中间维度 |
| $C$ | SM 数量 |

我们希望每个 GEMM 产生足够多的任务来填满至少两个波，同时让通信粒度尽可能小，以获得最大程度的重叠。对于 up 投影和 gate 投影——两者互不依赖、可以同时运行——这意味着：

$$
\frac{T}{128} \cdot \frac{2I}{256} \geq 2 \cdot C
$$

对于无法与其他算子并行运行的 down 投影：

$$
\frac{T}{128} \cdot \frac{H}{256} \geq 2 \cdot C
$$

将两者结合，我们要找的约束是：

$$
T \geq \frac{2C \cdot 128 \cdot 256}{\min(2I, H)}
$$

举例来说，Composer 2.5 的基座模型 Kimi 2.5，其 $H = 7168$、$I = 2048$。在 Blackwell GPU 上 $C = 148$，于是可得：

$$
T \geq 2368
$$

在我们以 Kimi 2.5 的形状运行 MoK 前向的微基准测试中，观测结果如下，与该启发式基本吻合：

| $T$ | ms |
| --- | --- |
| 512 | 5.981 |
| 1024 | 4.669 |
| 1536 | 3.981 |
| 2048 | 3.666 |
| 2560 | 3.425 |
| 3072 | 3.447 |
| 3584 | 3.524 |
| 4096 | 3.473 |

### 3.2 重叠的调度

> 通信粒度确定之后，**下一个问题是如何在 GPU 上调度不同的任务（dispatch、combine、FFN），使重叠高效发生**。为此，:mdnice-bold-italic[我们采用 SM 间重叠（inter-SM overlapping）技术：把一部分 SM 分配给专家 FFN（计算 SM），一部分分配给 dispatch/combine（通信 SM），两组之间通过本地计数器相互发信号]。

这之所以可行，是因为**借助 Tensor Memory Accelerator（TMA）的加载与存储，我们用不到三分之一的 SM[8] 就能打满 NVLink 带宽。**

我们以 SM 间重叠作为基础策略。在 MoE 前向中，通信 SM 启动 pull 式 dispatch all-to-all，每当一整批 minibatch 的 token 到达、可以开始计算时，就通知计算 SM。计算 SM 随后原子地在这批 token 上运行 MLP-SwiGLU；当直到 down 投影的全部计算完成后，它们再通知通信 SM：这些 token 可以 combine 了。

通信 SM 会先完成所有 dispatch 任务，再开始任何 combine all-to-all 任务，因此等通信 SM 做完 dispatch 时，计算 SM 发来的许多完成信号已经到位。随后通信 SM 转入 push 式 combine，而计算 SM 继续为最后到达的 token 跑完 FFN。此外，在计算 SM 等待第一轮 dispatch 完成期间，我们还可以把共享专家 FFN 插进来执行。

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/07.png)

::: image-caption
MoE 前向中 dispatch、FFN、combine 在计算 SM 与通信 SM 上的调度示意图。纵向分为计算 SM 与通信 SM 两条时序通道，横轴代表时间。计算 SM 依次执行共享专家计算以及多个子批次路由专家的 Up+gate、SwiGLU、Down 运算。计算任务完成后发出信号，驱动通信 SM 先后执行各子批次 Dispatch 分发与 Combine 合并操作，依靠信号同步协调两条通路，有序完成通信任务，实现前向传播阶段计算与通信任务流水执行。
:::

反向过程类似，只是由于涉及前向重放（forward replay）而稍微复杂一些（下一节详述）。通信 SM 执行 reverse-combine，并通知计算 SM 运行 FFN 反向——不包括 wgrad（即 3 个 dgrad 对应的 3 个专家分组 GEMM，以及 SwiGLU 激活的反向）；计算 SM 在 FFN 完成、token 可以 reverse-dispatch 时通知通信 SM。

> 我们把剩余的三个 wgrad 推迟到所有 minibatch 的 dgrad 全部完成之后再执行。推迟 wgrad 有两个原因：

1. reverse-dispatch 并不依赖它们，因此 wgrad 可以与最后几轮 reverse-dispatch 重叠；
2. 由于 token 轴是 wgrad 的归约维度，一次性沿整个 token 轴累加可以把数值不稳定性降到最低。

与前向一样，我们把完整的共享专家反向与第一轮 reverse-combine 重叠起来。

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/08.png)

::: image-caption
MoE 反向中 reverse-combine、FFN 反向、reverse-dispatch 与 wgrad 的调度示意图。纵向划分计算 SM 与通信 SM 两条时序通道，横轴代表时间。计算 SM 依次执行共享专家运算、各子批次路由专家反向梯度计算与权重梯度计算。计算任务完成后发出信号，驱动通信 SM 交替执行反向合并、反向分发任务。依靠信号同步协调上下两条通路，有序完成各个子批次对应的通信操作，实现反向传播阶段计算与通信任务交错执行。
:::

## 四、用环形 token 缓冲区彻底消除 CPU-GPU 同步

> 最后一块拼图是处理 MoE 的动态性：我们事先无法知道会有多少 token 到达某个目标 rank。现有两种处理方式。

- 第一种是 token 丢弃（token dropping）：超出固定缓冲区大小的 token 直接被忽略，永远不进入专家 FFN。
- 第二种是 CPU-GPU 同步：把由路由结果算出的每 rank token 计数发给 CPU，由 CPU 据此分配大小精确的缓冲区。

**两者都有明显缺点**。为了训练质量，我们不想丢 token；而 CPU-GPU 同步会阻碍 GPU 流跑到 CPU 线程前面。

在 GB300 NVL72 上，CPU-GPU 同步的代价尤其高，因为集成的 Grace CPU 很慢。:mdnice-bold-italic[trace 经常显示 GPU kernel 被 CPU 侧工作卡住——比如打日志、往远程服务器推训练指标——而同样的训练栈在配备 Intel CPU 的 DGX 机器上完全没有这类问题。鉴于此，我们希望尽可能减少 CPU-GPU 同步。]

> 在 MoK 中，我们用一个环形 token 缓冲区来解决这个问题，称之为 macrobatch。我们不再为通信 SM 分配一个很大却基本用不满的目标缓冲区来接收 token，而是创建一个几百 MB 的固定大小环形缓冲区，以 minibatch 为粒度在其中循环使用。

要高效实现 macrobatch，需精细重叠计算与通信：任一方都不应因缓冲区槽位仍被占用或数据未就绪而等待。理想情况下，重叠应跨越环形缓冲区边界——即在缓冲区尾部执行计算的同时，向头部发起 dispatch。

::: block-1
:mdnice-highlight[核心原则是：在性能允许前提下，尽早释放每个槽位，以便尽快复用]。前向过程中，dispatch 填充槽位，combine 清空槽位，从而为下一轮 dispatch 预留空间。
:::

若沿用前文调度方式——先完成全部 dispatch，再执行全部 combine——则下一 macrobatch 必须等待所有 combine 结束后才能启动，导致计算与通信在第一轮 dispatch 完成前无法重新重叠。

**为避免此问题，前向采用 dispatch 与 combine 交错执行**：:mdnice-bold-italic[上一 macrobatch 某 minibatch 的 combine，与下一 macrobatch 对应 minibatch 的 dispatch 并发进行，共用环形缓冲区同一区域]。如此，槽位一经释放即被立即复用。*需注意：首个 macrobatch 的 dispatch 与末个 macrobatch 的 combine 不参与交错，因其无对应配对操作。*

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/09.png)

::: image-caption
前向中跨 macrobatch 的 dispatch-combine 交错示意图。纵向分为计算 SM 与通信 SM 两条时序通道，横轴代表时间。计算 SM 依次执行多个子批次路由专家计算，包含 Up+gate、SwiGLU、Down 运算。计算任务完成后发送信号，驱动通信 SM 交错执行 Combine 合并、Dispatch 分发操作，交替处理不同宏批次、子批次任务，实现计算任务与通信任务流水衔接，调度时序随时间持续向后延伸。
:::

### 4.1 反转环形缓冲区，最小化反向时的前向重放

反向过程稍显复杂，因为反向 megakernel 需支持前向激活重放。macrobatch 缓冲区采用环形结构，当存在多个 macrobatch 时，前向过程中部分已保存的激活可能被后续 macrobatch 覆盖，因而需在反向时重新计算。重放只需执行至 SwiGLU 激活层即可，因反向仅需 down 投影层的输入激活及其输出梯度，无需其前向输出结果。

::: block-1
**一项关键优化是采用“反转环”策略**：若前向按 macrobatch 递增顺序处理 token，末尾 macrobatch 将导致环形缓冲区处于部分填充状态，从而增加反向重放量。:mdnice-highlight[为此，MoK 在前向中按 macrobatch 逆序遍历 token，确保环形缓冲区最终要么完全填满，要么完整保留全部 token，最大限度减少反向阶段的前向重放。]
:::

## 五、其他特性与实现细节

### 5.1 Megakernel

> 我们使用 megakernel[9] 技术构建 MoK：指令在 SM 任务级别重叠，而不是被 kernel 启动边界隔开。**Megakernel 通常适合用来融合整个前向或反向步骤，但我们发现它对 MoK 有用有两个原因：**

1. 如果不用 megakernel 融合，实现 minibatch 和 macrobatch 将需要多次 kernel 启动，增加启动边界开销；
2. 对于 SM 间重叠，我们发现多流加 green context 的方式无法可靠地按我们的意图精确划分 SM，而软件划分能给出精确的资源分配保证。

### 5.2 确定性

为了内部消融实验和在策略（on-policy）RL 后训练，我们把 MoK 设计为完全确定性的。浮点运算的顺序是固定的，因此无论硬件调度和指令发射顺序如何，相同输入都会产生逐比特一致的输出。

### 5.3 用于 RDMA 重叠的 Cluster Launch Control（CLC）

MoK megakernel 通过 CLC 调度任务，以允许 RDMA 重叠。CLC 是 Blackwell 引入的硬件原生工作窃取（work-stealing）特性，是实现持久化网格 kernel（persistent grid kernel）的一种新方式。

**基于 CLC 的 kernel 可以自然地让位于更高优先级流上的 kernel，而无需等待当前 kernel 完全执行完**。:mdnice-bold-italic[这一点很重要，因为在训练期间，我们经常需要把机架内的通信与计算机架间的 InfiniBand/RoCE 通信（例如 FSDP all-gather）重叠起来，而后者需要自己的 SM]。没有 CLC 的话，机架间通信会被串行化到 megakernel 之后，造成显著的性能下降。

### 5.4 MXFP8 支持

MoK 同时支持 BF16 和 MXFP8 两种精度模式。我们采用 MXFP8 模式进行训练，因其速度更快，且未引发数值问题。在 MXFP8 模式下，共享专家仍使用 BF16 精度，因为我们发现将其改为 MXFP8 会降低训练稳定性。

MXFP8 的一项额外成本是：张量必须先量化，才能喂给 Tensor Core。为了把这部分开销降到最低，我们 (1) 提供了一个优化的 MXFP8 量化 kernel，用于权重的预量化；(2) 把激活量化融合进 dispatch all-to-all、专家分组 GEMM 和 SwiGLU 之中。

### 5.5 路由权重梯度计算

MoK 的反向过程还会计算路由权重的梯度，采用 SonicMoE[10] 提出的计算方式。它不保存完整的 down 投影输出，而是利用 SwiGLU 激活值与 down 投影梯度（dgrad）的内积来计算路由权重梯度。该计算被融合到 SwiGLU 的反向传播中，从而避免额外的内存访问开销。

### 5.6 计算/通信 SM 调优

> 计算 SM 与通信 SM 的最优比例取决于具体工作负载。本地 token 数量、模型结构、硬件计算能力以及网络带宽，都会影响 dispatch/combine 任务相对于专家 FFN 任务的执行时间。

**即使在同一工作负载下，前向传播与反向传播的最优比例也存在差异**。因此，:mdnice-bold-italic[MoK 提供了可调节参数，允许分别为前向和反向过程独立配置通信 SM 的数量。]

## 六、性能提升

我们用两类基准测试评估了 MoK：

- 一是单个 MoE 层的基准测试（全部基准测试代码均已开源）
- 二是在我们横跨多个 NVL72 机架的内部生产训练栈上做的端到端基准测试。

### 6.1 MoE 层基准测试

> 我们基准测试单个 MoE 层的完整执行：全局 token 调度、token dispatch all-to-all、专家 FFN、token combine all-to-all，以及最后的加权和。前向和反向分别测试，并覆盖 BF16 与 MXFP8 两种模式。所有测试都在单个 NVL72 机架内运行，EP 度为 64，路由前每 GPU 2,048 个 token。

选取的基线满足：(1) 同时支持 MoE 前向和反向；(2) 能在 NVL72 上运行：

- NCCL + PyTorch
- DeepEP + PyTorch
- DeepEP + TransformerEngine
- HybridEP + Megatron——这是英伟达在 NVL72 上推荐的方案，优于 DeepEP + Megatron

我们测试了被广泛使用的开放权重模型的形状：

- Kimi K2.7 Code 形状（$E$：384，$H$：7168，$I$：2048，top-k：8）
- GLM-5.2 形状（$E$：256，$H$：6144，$I$：2048，top-k：8）
- Qwen3.5-397B-A17B 形状（$E$：512，$H$：4096，$I$：1024，top-k：10）
- DeepSeek-V4-Pro 形状（$E$：384，$H$：7168，$I$：3072，top-k：6）

> 下面的图表展示了全部四种形状在所有基线下的结果（原网页为交互式图表，此处以数据表格呈现，单位为 TFLOPS；EP 度 64，路由前 2,048 token/GPU，GB300 NVL72）：

> **专家并行 MoE 吞吐量：MXFP8 前向（GB300 NVL72）**

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/10.jpg)

::: image-caption
在 GB300 NVL72s 硬件平台上，Mixture-of-Kittens 的 MXFP8 前向吞吐量最高可达最优公开基线的 2.37 倍。纵轴单位为 TFLOP/s，横轴包含 Kimi K2.7、GLM 5.2、Qwen 3.5-397B-A17B、DeepSeek V4 Pro 四款模型。图例列出五种方案，橙色柱代表 Mixture-of-Kittens。在全部模型测试中，该方案吞吐量均显著高于 NCCL+PyTorch、DeepEP 系列、HybridEP+Megatron 等基线方案。
:::

| 实现 | Kimi K2.7 | GLM 5.2 | Qwen 3.5-397B-A17B | DeepSeek V4 Pro |
| --- | --- | --- | --- | --- |
| NCCL + PyTorch | 568.1 | 510.0 | 198.1 | 683.1 |
| DeepEP + PyTorch | 558.4 | 477.3 | 208.4 | 665.2 |
| DeepEP + TransformerEngine | 369.6 | 332.4 | 127.1 | 434.0 |
| HybridEP + Megatron | 631.8 | 563.1 | 224.4 | 738.9 |
| **MoK（本文）** | **1,255.4** | **1,218.9** | **531.1** | **1,360.3** |

> **专家并行 MoE 吞吐量：MXFP8 反向（GB300 NVL72）**

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/11.jpg)

::: image-caption
专家并行 MoE 吞吐量：MXFP8 反向传播（GB300 NVL72s），纵轴单位 TFLOP/s，横轴依次为 Kimi K2.7、GLM 5.2、Qwen 3.5-397B-A17B、DeepSeek V4 Pro 四类模型。图例包含五种方案，橙色柱代表 Mixture-of-Kittens。在全部测试模型场景中，Mixture-of-Kittens 吞吐量均高于 NCCL+PyTorch、DeepEP 系列、HybridEP+Megatron 等对比方案，在各个模型下都展现出领先的性能水平。
:::

| 实现 | Kimi K2.7 | GLM 5.2 | Qwen 3.5-397B-A17B | DeepSeek V4 Pro |
| --- | --- | --- | --- | --- |
| NCCL + PyTorch | 1,051.5 | 997.2 | 471.2 | 1,226.6 |
| DeepEP + PyTorch | 1,030.1 | 928.8 | 376.7 | 1,217.1 |
| DeepEP + TransformerEngine | 978.1 | 894.9 | 328.5 | 1,128.5 |
| HybridEP + Megatron | 1,231.6 | 1,153.2 | 500.3 | 1,334.2 |
| **MoK（本文）** | **1,587.4** | **1,602.1** | **890.8** | **1,631.8** |

> **专家并行 MoE 吞吐量：BF16 前向（GB300 NVL72）**

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/12.jpg)

::: image-caption
专家并行 MoE 吞吐量：BF16 前向传播（GB300 NVL72s），纵轴单位为 TFLOP/s，横轴包含 Kimi K2.7、GLM 5.2、Qwen 3.5-397B-A17B、DeepSeek V4 Pro 四类模型。图例展示五种实现方案，橙色柱代表 Mixture-of-Kittens。在全部测试模型下，Mixture-of-Kittens 吞吐量均显著高于 NCCL+PyTorch、DeepEP 系列、HybridEP+Megatron 等对比方案，在各场景中均具备明显性能优势。
:::

| 实现 | Kimi K2.7 | GLM 5.2 | Qwen 3.5-397B-A17B | DeepSeek V4 Pro |
| --- | --- | --- | --- | --- |
| NCCL + PyTorch | 597.1 | 563.2 | 258.7 | 719.2 |
| DeepEP + PyTorch | 658.8 | 605.8 | 316.9 | 752.7 |
| DeepEP + TransformerEngine | 494.7 | 451.9 | 181.0 | 591.0 |
| HybridEP + Megatron | 651.5 | 610.8 | 252.6 | 746.7 |
| **MoK（本文）** | **981.8** | **980.4** | **608.6** | **1,024.6** |

> **专家并行 MoE 吞吐量：BF16 反向（GB300 NVL72）**

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/13.jpg)

::: image-caption
专家并行 MoE 吞吐量：BF16 反向传播（GB300 NVL72s），纵轴单位 TFLOP/s，横轴依次为 Kimi K2.7、GLM 5.2、Qwen 3.5-397B-A17B、DeepSeek V4 Pro 四类模型。图例包含五种方案，橙色代表 Mixture-of-Kittens。在全部模型测试场景中，Mixture-of-Kittens 吞吐量均处于较高水平，尤其在 Qwen 3.5-397B-A17B 模型上优势突出，整体性能普遍优于 NCCL+PyTorch、DeepEP 系列、HybridEP+Megatron 等对比方案。
:::

| 实现 | Kimi K2.7 | GLM 5.2 | Qwen 3.5-397B-A17B | DeepSeek V4 Pro |
| --- | --- | --- | --- | --- |
| NCCL + PyTorch | 935.8 | 914.0 | 532.9 | 1,039.6 |
| DeepEP + PyTorch | 1,012.9 | 985.6 | 575.8 | 1,125.3 |
| DeepEP + TransformerEngine | 919.8 | 857.5 | 390.9 | 951.1 |
| HybridEP + Megatron | 984.1 | 1,012.6 | 560.8 | 1,046.9 |
| **MoK（本文）** | **1,114.5** | **1,112.5** | **910.6** | **1,108.7** |

总体而言，与最快的基线相比，MoK 在 MXFP8 前向上最高快 2.37 倍，MXFP8 反向最高快 1.78 倍，BF16 前向最高快 1.92 倍，BF16 反向最高快 1.58 倍。

### 6.2 端到端基准测试

> 除了与公开可用实现的对比之外，我们还报告了内部生产训练栈上的加速效果。此前，**我们的训练栈依赖基于 DeepEP 的实现，并配有自研的 MXFP8 MoE 计算 kernel 来做专家并行**。通过更换 MoE 层的实现（DeepEP 对比 MoK），比较端到端每秒 token 数这一指标。

![文章插图](/articles/2026-08-05-cursor-gb300-nvl72-moe-megakernel-5df0b951/images/14.jpg)

::: image-caption
End-to-end training throughput comparison (GB300 NVL72s)，横轴为训练 Step，范围 300 至 500，纵轴单位为 Tokens/second/GPU。橙色曲线代表 Mixture-of-Kittens，整体稳定维持在 900 至 1100 区间；灰色曲线代表 DeepEP-based，大致处于 600 至 750 区间。两条曲线均存在周期性小幅波动与阶段性低谷，全程橙色曲线吞吐量显著高于灰色曲线，直观体现 Mixture-of-Kittens 拥有更高单 GPU 训练吞吐性能。
:::

基准测试在横跨多个 GB300 NVL72 机架的 512 张 GPU 上进行。

| 端到端训练吞吐量对比（GB300 NVL72） | 基于 DeepEP | MoK |
| --- | --- | --- |
| Token / 秒 / GPU | 760.9 | 1,070.2（1.41 倍） |

总体而言，MoK 相对我们此前基于 DeepEP 的生产配置带来了约 41% 的每秒 token 数提升，使我们能够在 GB300 NVL72 基础设施上更高效地训练模型。

## 七、面向智能体时代的 Megakernel

> Mixture-of-Kittens 自今日起完全开源[11]。我们计划持续维护它，并欢迎社区的反馈与贡献。我们希望这能降低 AI 研究的门槛，让更多研究者和实验室能够高效地训练模型。MoK 同时也力求灵活——我们在"易于修改"上投入了大量精力，这样在智能体的帮助下，你可以快速把它适配到我们所用平台之外的硬件上。

顺便一提，随着能够编写高性能 kernel 的 AI 模型出现，kernel 优化这个领域变得有趣多了。过去需要手工编写的单算子 kernel，如今几乎完全可以交给 AI。**只要给出正确的设计方向（例如如何做 warp 特化、使用哪些 PTX 指令），智能体就能一次写出最先进的 kernel。**

不过更有趣的是这给我们带来的新能力。

- 去年，编写一个 megakernel 还需要先造一层简化的抽象，比如这个 SonicMoE 也是我们的工作[12]。
- **今年，我们不再需要框架，而是移除了那层额外的抽象，在智能体的帮助下从零开始直面复杂性本身。**

::: block-1
智能体自动完成了较简单的任务（例如单算子 kernel），并帮助我们以非常小的团队、快得多地攻克了更难的任务（例如编写分布式 MoE megakernel）。:mdnice-highlight[话虽如此，在优化横跨数十万张 GPU 运行的高风险 ML 工作负载方面，前方还有大量工作]。如果你对这篇文章和这类工作感兴趣，我们很期待你的来信：hiring@cursor.com[13]。
:::

参考资料

[1]

Mixture-of-Kittens（MoK）: *https://github.com/cursor/mixture-of-kittens*

[2]

Composer: *https://cursor.com/blog/composer-2*

[3]

GitHub: *https://github.com/cursor/mixture-of-kittens*

[4]

MXFP8: *https://cursor.com/blog/kernels*

[5]

warp decode: *https://cursor.com/blog/warp-decode*

[6]

Comet: *https://arxiv.org/abs/2502.19811*

[7]

DeepEP: *https://github.com/deepseek-ai/DeepEP*

[8]

不到三分之一的 SM: *https://arxiv.org/abs/2511.13940*

[9]

megakernel: *https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles*

[10]

SonicMoE: *https://arxiv.org/abs/2512.14080*

[11]

完全开源: *https://github.com/cursor/mixture-of-kittens*

[12]

这个: *https://github.com/HazyResearch/Megakernels*

[13]

hiring@cursor.com: *mailto:hiring@cursor.com*

::: related-title
**相关推荐**
:::

::: related-links
- **[64块H100媲美96块！开源MoE软硬件协同加速方案SonicMoE：基于IO和tile感知优化实现45%内存节省+1.86倍吞吐量！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894859&idx=1&sn=c50a765749e1cb035469b0fa38334033&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**
- **[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903011&idx=1&sn=7872d080a5475a44bba4cfabe7717b56&scene=21#wechat_redirect)**

:::
