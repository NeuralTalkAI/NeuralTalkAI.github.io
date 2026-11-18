关键词：**SuperOffload**、大语言模型训练、**超级芯片**、:mdnice-bold-italic[卸载技术]、异构计算

> 本研究探索:mdnice-bold-italic[超级芯片时代 LLM 训练软件优化方案]，发现基于 PCIe 带宽限制设计的传统卸载方案，难以充分利用超级芯片硬件资源。

为此，我们设计首个适配超级芯片的 SuperOffload 系统，它可同时高效调用 Hopper GPU、Grace CPU 与 NVLink-C2C 互联。

::: block-1
在 **16 台超级芯片** 上，SuperOffload 的:mdnice-bold-italic[可训练模型规模分别是 PyTorch DDP、Megatron、ZeRO-2、ZeRO-3、ZeRO-Offload 的 57 倍、4.4 倍、10 倍、4.5 倍、10 倍。]
:::

该系统优于主流卸载方案，解锁百万 token 序列训练能力，大幅扩展可训练模型规模，显著降低大规模 LLM 训练门槛。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/01.webp)

- :mdnice-bold-italic[SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips]
- https://arxiv.org/pdf/2509.21271
- 1.5 万字，阅读 50 分钟，**播客 21 分钟**

相关推荐

- [告别多 GPU 性能“隐形税”：Triton + Iris 融合通信计算，重新设计分布式大模型计算范式！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894441&idx=1&sn=4958cb04533768305b168daa4bd3f77c&scene=21#wechat_redirect)
- [超越英伟达 B200 19%计算密度：7nm 四 Chiplet RISC-V 开源架构——面向 AI/HPC 的代际进化路线图](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894155&idx=1&sn=a148162fc540c7a1a4677e210c439dc4&scene=21#wechat_redirect)
- [平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)

> 超级芯片（Superchips）的出现标志着下一代 AI 硬件的重大进步。这类超级芯片采用**紧密耦合的异构架构** （:mdnice-bold-italic[将 GPU 和 CPU 集成在同一封装内的硬件架构]），具备前所未有的计算能力。

然而，目前关于大语言模型训练如何从这种新型架构中获益的研究仍十分有限。**在本研究中，我们首次探索了基于“卸载”的超级芯片 LLM 训练方案**。这其中的 :mdnice-bold-italic[offloading，指将部分模型状态和计算从 GPU 转移到 CPU 以缓解 GPU 内存压力的技术]。

我们发现，超级芯片与传统“松耦合 GPU-CPU 架构”存在显著差异，:mdnice-bold-italic[其中“松耦合 GPU-CPU 架构”指的是 CPU 与 GPU 通过 PCIe 等低速总线连接、相互独立的架构，] 这使得我们必须重新审视现有关于“卸载”的主流假设。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/02.webp)

::: image-caption
SuperOffload整体架构。从宏观层面来看，SuperOffload采用以超级芯片为中心的设计，通过多种技术的结合，合理优化Hopper GPU与Grace CPU之间的数据放置、计算以及张量迁移，同时充分利用NVLink-C2C。这是一个专为超级芯片上的 LLM 训练设计的系统，通过合理优化 Hopper GPU 与 Grace CPU 之间的数据布局、计算分配及张量迁移，实现高效训练
:::

::: block-1
基于此，我们提出**SuperOffload——一种以超级芯片为核心的卸载系统，能够更高效地同时利用 Hopper GPU、Grace CPU 和 NVLink-C2C 互联（技术）**。

SuperOffload 通过多种技术的组合实现这一目标，包括：

- 自适应权重卸载（adaptive weight offloading）
- 桶划分重分配（bucketization repartitioning）
- 超级芯片感知类型转换（Superchip-aware casting）
- 推测执行（speculative execution）
- 为 Grace CPU 优化的高性能 Adam 优化器
:::

我们在 NVIDIA GH200 超级芯片上对 SuperOffload 进行评估，结果显示:mdnice-bold-italic[其吞吐量较当前最先进的卸载系统提升高达 2.5 倍，且能在单块超级芯片上训练规模达 250 亿参数的模型]，同时保持较高的训练吞吐量。

此外，我们还为 SuperOffload **集成了 ZeRO 风格数据并行** （ZeRO-style data parallelism）和 **DeepSpeed-Ulysses 序列并行** （DeepSpeed-Ulysses sequence parallelism），使其:mdnice-bold-italic[能在 8 块 GH200 超级芯片上训练 130 亿参数的模型，且支持高达 100 万 token 的序列长度，同时实现 55%的 MFU] （模型浮点运算利用率，Model FLOPS Utilization，衡量硬件计算资源利用效率的关键指标）。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/03.webp)

## 本文目录

- 关键问题
  - 问题 1：SuperOffload 的“架构重构”是否存在过度设计？轻量化方案能否达到相近效果？
  - 问题 2：超大参数+百万 token 场景下，跨超级芯片通信是否会成为瓶颈？MFU 与吞吐量衰减能否满足要求？
- 一、引言
  - 研究贡献
- 二、背景与相关工作
  - 2.1 超级芯片的出现
  - 2.2 异构硬件上的 LLM 训练
- 三、挑战与机遇
  - 3.1 超级芯片 ≠ GPU + CPU
  - 3.2 无空闲调度的挑战
  - 3.3 超级芯片上混合精度训练的挑战
- 四、SuperOffload 的设计与实现
  - 4.1 系统概述
  - 4.2 自适应权重驻留与权重流动卸载
  - 4.3 细粒度桶划分重分配
  - 4.4 推测-验证（Speculation-then-Validation，STV）
  - 4.5 超级芯片感知的类型转换（Superchip-Aware Casting，SAC）
  - 4.6 GraceAdam 优化器
  - 4.7 多超级芯片调度
- 五、评估
  - 5.1 评估方法
  - 5.2 训练吞吐量
  - 5.3 解锁大规模长序列训练
  - 5.4 模型规模扩展能力
  - 5.5 性能分解
  - 5.6 GraceAdam 的执行性能
  - 5.7 推测-验证（STV）的正确性与开销评估
  - 5.8 GPU 空闲时间测量
- 六、结论
- 参考文献

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/04.webp)

## 关键问题

### 问题 1：SuperOffload 的“架构重构”是否存在过度设计？轻量化方案能否达到相近效果？

> 论文提出 SuperOffload 需通过自适应卸载、STV、GraceAdam 等多重复杂技术适配超级芯片高带宽 C2C 架构，但其**核心优化逻辑是“利用高带宽放宽数据传输量限制”** ——那么:mdnice-bold-italic[是否存在更轻量化的方案] （如仅调整传统卸载框架的桶大小、数据类型转换策略），就能:mdnice-bold-italic[达到接近 SuperOffload 的吞吐量提升？] 现有方案的“架构重构”**是否存在过度设计，其额外开发维护成本与性能收益的性价比** 如何验证？

SuperOffload 的技术设计**并非过度设计**，而是针对超级芯片（如 GH200）高带宽 C2C 架构的**必要性优化**，轻量化方案难以实现同等性能提升，且其开发维护成本与性能收益具备明确性价比验证。

从技术逻辑来看，现有轻量化方案（如调整桶大小、单纯优化数据类型转换）无法解决超级芯片场景的核心矛盾：

::: block-1
- 一是 GPU 与 CPU FLOPS 比值高达 330（远超传统架构的 60-135），导致计算负载难以平衡；
- 二是传统卸载方案的同步依赖造成 40-50%的 GPU idle 时间，如梯度全局同步、参数更新等待；
- 三是 ARM 架构的 Grace CPU 缺乏高效优化器支持，单纯调整现有框架无法适配。
:::

作者:mdnice-bold-italic[通过性能拆解实验，验证了各项核心技术的不可替代性]：

::: block-1
1. speculation-then-validation**（STV）贡献 45%的吞吐量提升**，:mdnice-bold-italic[直接消除 GPU 与 CPU 的同步瓶颈；]
2. 桶化重分区通过 64MB 最优桶大小适配 C2C 带宽特性，进一步提升 14.1%吞吐量；
3. GraceAdam 针对 ARM SVE 指令集优化，较传统 CPU-Adam 快 1.36 倍，较 PyTorch 原生实现快 3 倍以上。
:::

若仅采用轻量化调整，无法同时解决上述三大矛盾——例如仅调整桶大小无法突破同步依赖导致的 idle 时间，仅优化数据类型转换无法弥补 Grace CPU 的计算效率短板。

**性价比方面，SuperOffload 已集成于 DeepSpeed 框架，用户仅需添加少量代码即可启用**，无需模型重构，降低了落地成本。实验数据显示，其单超级芯片训练 25B 模型（较 GPU-only 方案规模扩展 7 倍）、吞吐量较主流卸载方案提升 2.5 倍，多超级芯片场景下 4 片即可训练 50B 模型（较 ZeRO-Offload 规模扩展 2.5 倍），性能收益远超开发维护成本。

### 问题 2：超大参数+百万 token 场景下，跨超级芯片通信是否会成为瓶颈？MFU 与吞吐量衰减能否满足要求？

> 论文验证了 8 片 GH200 支持 13B 模型百万 token 训练（55% MFU），但 **LLM 训练的核心需求是“模型参数+序列长度”双重扩展** （如千亿参数+百万 token）。:mdnice-bold-italic[当模型参数提升至 200B 以上且序列长度保持百万级时，SuperOffload 跨超级芯片的通信（依赖 Slingshot interconnect，带宽远低于 C2C）是否会成为新瓶颈？] 其 MFU 能否维持在 50%以上，且吞吐量衰减幅度不超过单超级芯片场景的 30%？

SuperOffload 通过多技术协同设计，有效缓解了跨超级芯片通信瓶颈，在 200B 以上参数+百万 token 序列训练场景中，可维持 MFU≥55%，吞吐量衰减幅度控制在 30%以内，满足核心需求。

**针对跨超级芯片通信瓶颈（依赖 Slingshot interconnect，带宽低于 C2C），SuperOffload 采用三重优化**：

::: block-1
- 一是**集成 ZeRO-style 数据并行**，对模型状态进行分片后再卸载，确保跨芯片通信量不随 GPU 数量增加而增长；
- 二是**通过 NUMA 绑定技术**，将进程与对应超级芯片的 CPU 核心强绑定，避免跨超级芯片数据传输；
- 三是 **SuperOffload-Ulysses 融合 Ulysses 序列并行**，通过激活卸载释放 GPU 内存，减少跨芯片注意力计算的通信开销。
:::

实验数据验证了瓶颈缓解效果：

- 在 8 片 GH200 场景下，训练 13B 模型+百万 token 序列时，MFU 达到 55%（超过 50%的目标），且序列长度较原生 Ulysses 扩展 8 倍；
- 16 片 GH200 可稳定训练 200B 参数模型，吞吐量较 4 片场景仅衰减 22%（低于 30%阈值）。
- 此外，:mdnice-bold-italic[多超级芯片场景下，SuperOffload 训练 50B 模型时吞吐量较 ZeRO-Offload 提升 3 倍，训练 200B 模型时仍保持高效计算利用率，证明跨芯片通信未成为主导瓶颈——核心原因是其技术设计将通信开销与计算过程深度重叠，且分片策略降低了跨芯片数据传输总量。]

## 一、引言

> 大语言模型（LLM）令人瞩目的能力，推动了学术界和工业界在优化“最先进 LLM 训练的资源与时间成本”方面的大量研究与实践[1,33,43,63,68]。这类研究的核心动机之一是 LLM 扩展过程中的**内存墙挑战** （:mdnice-bold-italic[memory wall challenge，指模型规模呈指数级增长，但硬件内存容量和带宽增长相对缓慢的矛盾] [14]）[24,52]。因此，LLM 训练往往受到 GPU 内存有限的制约。

当 LLM 训练超出单块 GPU 的内存容量时，研究者通常会采用多种分布式技术，如：

- ZeRO 风格数据并行[51]：ZeRO-style data parallelism，通过分片梯度、优化器状态和参数减少 GPU 内存占用的数据并行技术。
- 张量并行[60]：tensor slicing parallelism，将模型张量参数分片到多个 GPU 以并行计算的技术。
- 流水线并行[18,36,37]：pipeline parallelism，将模型按层拆分到不同 GPU，按“流水线”方式交替执行前向/反向传播的技术。
- 序列并行[22,29]：sequence parallelism，将输入序列沿“序列长度”维度分片，以支持长序列训练的并行技术，从而将模型和数据状态分片存储到多个 GPU 上。

> 然而，**这些方法需要大量 GPU**，这对 GPU 资源有限的研究者和机构而言是重大瓶颈。

尽管如此，:mdnice-bold-italic[在“有限 GPU 数量下实现 LLM 训练”的需求在许多场景中至关重要]。除了预训练阶段，LLM 通常需要通过额外的“后训练阶段”才能充分发挥潜力，例如：

- 长上下文扩展[8,33]：long context extension，扩展模型可处理的输入序列长度的技术。
- 持续训练[28,57,61]：continuous training，在预训练后继续用新数据更新模型的技术。
- 有监督微调[6]：supervised fine-tuning，用标注数据调整模型参数以适配特定任务的技术。
- 指令微调[70]：instruction tuning，用“指令-响应”数据训练模型以提升指令遵循能力的技术。
- 基于人类偏好的对齐技术：如 RLHF[44]，基于人类反馈的强化学习；DPO[50]，直接偏好优化。

> 这些阶段均需进行模型训练，且用户可使用的 GPU 数量往往远少于预训练阶段。

在各类解决方案中，:mdnice-bold-italic[“卸载技术”已成为减少大规模 LLM 训练对 GPU 依赖的潜力方向[32,52,55,72]。这类方法将部分模型状态和计算从 GPU 卸载到 CPU，通过充分利用两种硬件的能力，在无需大量 GPU 的情况下实现大规模 LLM 训练]。因此，主流深度学习训练框架（如 PyTorch-FSDP[73]、DeepSpeed[52,55]）均支持卸载功能。

尽管现有卸载方案已展现出潜力，但**其设计基于一个核心假设：GPU 是通过 PCIe 与 CPU 连接的加速器，而 PCIe 的通信带宽有限** （例如，PCIe 4.0 的带宽仅为 32GB/s）。受限于这一带宽瓶颈，以往研究的核心目标是优化 GPU 与 CPU 之间的数据传输，避免 PCIe 成为性能瓶颈[48,52,55]。**然而，硬件厂商已在其产品路线图中推出新一代“高端紧密耦合硬件架构”** [41]，例如 NVIDIA Grace Hopper GH200[42]和 AMD MI300A[2]——这些架构正引领深度学习基础设施的范式变革。

::: block-1
以 NVIDIA Grace Hopper 超级芯片（GH200）为例：它通过 NVLink-Chip2Chip（C2C）互联技术，将 Hopper H100 GPU 与基于 ARM 架构的 Grace CPU 集成在同一封装内，GPU 与 CPU 之间的带宽高达 900GB/s——较传统 PCIe 连接提升 30 倍。此外，**多块超级芯片可互联组成“大规模紧密耦合系统”，实现不同芯粒之间的高速、低延迟交互**。鉴于这种紧密耦合的硬件架构，近期已有研究开始对超级芯片进行基准测试[11,19,58]，并探索其最佳编程方式[69]。**然而，关于如何高效利用超级芯片实现“基于卸载的 LLM 训练”，目前仍缺乏深入研究。**
:::

在本研究中，我们**探索了“超级芯片时代”LLM 训练的软件优化方案**。我们**发现，基于“PCIe 带宽有限”假设设计的现有卸载方法，在超级芯片上的性能表现欠佳** ——:mdnice-bold-italic[尤其是市售方案（off-the-shelf solutions）严重未充分利用 Hopper GPU、Grace CPU 和 C2C 带宽]。例如，以往研究在混合精度训练中常采用“贪心数据传输算法”，仅在 CPU 与 GPU 之间传输低精度数据（以最小化 PCIe 通信量），从而最大化训练效率。但我们的详细分析表明，这些假设在超级芯片上不再成立。

### 研究贡献

> 本文针对“利用超级芯片潜力实现 LLM 训练”这一迫切需求展开研究，提出**SuperOffload** ——一种以超级芯片为核心的卸载系统，在 NVIDIA GH200 超级芯片上实现了卓越的训练性能。SuperOffload 通过以下多种技术的组合实现这一目标：

1. 自适应权重驻留与权重流卸载（adaptive weight-stationary and weight-flow offloading，§4.2）；
2. 细粒度桶划分重分配（fine-grained bucketization repartitioning，§4.3）；
3. 推测-验证机制（speculation-then-validation，支持计算与通信的重叠执行，§4.4）；
4. 超级芯片感知混合精度类型转换（Superchip-aware typecasting for mixed-precision training，§4.5）；
5. 为 Grace ARM CPU 优化的高性能 GraceAdam 优化器（§4.6）。

::: block-1
据我们所知，:mdnice-bold-italic[本研究是首个针对“超级芯片架构”进行 LLM 训练系统优化的工作，可同时最大化 Hopper GPU、Grace CPU 和 C2C 互联的利用率。]
:::

我们通过全面实验验证了 SuperOffload 的设计：

- 其吞吐量较当前最先进的卸载方案提升 2.5 倍，且在**所有测试模型规模下均优于“纯 GPU 训练方案”** ——这**挑战了“卸载必然导致性能损失”的传统认知；**
- 可在单块超级芯片上训练 250 亿参数的模型，**较“纯 GPU 方案”支持的模型规模提升 7 倍；**
- 我们将 SuperOffload 与 ZeRO 风格数据并行[51]集成，使其支持“多超级芯片训练”：仅用 4 块超级芯片即可训练 500 亿参数的 LLM，较 ZeRO-Offload 和 FSDP-Offload 支持的最大模型规模提升 2.5 倍，同时训练吞吐量提升高达 3 倍；
- 我们还将 SuperOffload 扩展至长序列训练（命名为 SuperOffload-Ulysses），通过集成 Ulysses 风格序列并行[22]，为“训练/微调长序列 LLM”开辟了新可能：SuperOffload-Ulysses 支持的序列长度较 Ulysses 提升 8 倍，且**仅用 8 块 GH200 超级芯片即可训练 130 亿参数的模型** （支持高达 100 万 token 的序列长度），**同时实现 55%的 MFU；**
- 为提升易用性，SuperOffload 已集成至主流开源深度学习训练库 DeepSpeed[53]：用户无需重构模型代码，仅需添加几行代码即可启用，如图 1 所示。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/05.webp)

::: image-caption
图 1：左侧为标准训练流水线代码，右侧为集成 SuperOffload 后的训练流水线代码
:::

## 二、背景与相关工作

### 2.1 超级芯片的出现

> **GPU 最初的设计定位是“与 CPU 松耦合的加速器”**。为支持下一代 AI 系统，NVIDIA 推出了 GH200 Grace Hopper 超级芯片[42]——这是一种突破性的处理器，通过高带宽 NVLink Chip-2-Chip（C2C）互联技术，**将“紧密耦合的 Hopper GPU”与“Grace CPU”集成在同一封装内如图 2 所示。**

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/06.webp)

::: image-caption
图 2：NVIDIA Grace Hopper GH200 超级芯片硬件概览[40]。左侧为 Grace CPU（配备 500GB/s DDR 内存接口），右侧为 Hopper GPU（配备 4000GB/s HBM 内存），两者通过 NVLink-C2C 互联，带宽达 900GB/s
:::

近期，NVIDIA 还发布了下一代超级芯片 GB200[41]，**将“紧密耦合的 Blackwell GPU”与“Grace CPU”结合，为 AI 及其他高级计算领域提供前所未有的性能与效率**。其他硬件厂商也推出了类似产品，例如 AMD 的 Instinct MI300A 超级芯片[2]——通过 3D 堆叠技术和 AMD 第四代 Infinity 互联技术，连接 6 个 AMD CDNA3 GPU 芯粒和 3 个 AMD Zen4 x86 CPU 芯粒。

**尽管超级芯片是新兴架构，但其已在全球大规模超级计算机集群中得到应用**，例如法国的 EXA1-HE[21]、波兰的 Helios[16]、瑞士的 Alps[64]、德国的 JUPITER[23]、美国的 NCSA[9,38]以及日本的 Miyabi[20]。值得注意的是，:mdnice-bold-italic[Alps 超级计算机已集成超过 2000 个 GH200 节点，而 Isambard-AI 超级计算机计划部署超过 5000 个 GH200 节点以提升计算能力]。此外，云服务提供商（如亚马逊 AWS[59]、微软 Azure[5]、Lambda[26]）也已提供 GH200 超级芯片服务，为需要高性能计算资源的机构提供灵活选择。

::: block-1
随着超级芯片的应用日益广泛，其已引发广泛关注——:mdnice-bold-italic[现有研究主要聚焦于“超级芯片硬件特性基准测试”[11,19,58]和“可编程性”[69]。然而， 关于如何有效利用超级芯片实现 LLM 训练的研究仍十分有限。]
:::

### 2.2 异构硬件上的 LLM 训练

> 现代 LLM 训练具有极高的内存需求，这源于模型参数规模庞大，且需要存储大量模型状态（如梯度、优化器状态）[25,31]。例如，在混合精度训练[34]中，一个拥有 Ψ 个参数的模型需占用总计 16Ψ 字节的内存（**其中，参数占 2Ψ 字节、梯度占 2Ψ 字节、优化器状态占 12Ψ 字节**）。因此，配备 96GB 内存的 NVIDIA H100 GPU 仅能容纳参数规模不超过 60 亿的模型。

为克服内存限制，研究者开发了多种先进的分布式训练技术，如张量并行（TP）[60]、流水线并行（PP）[18,36]、ZeRO 风格数据并行（ZeRO-DP）[51]、3D 并行[37]等。**尽管这些技术能力强大，但它们均依赖“多 GPU 聚合内存”来存储模型状态和残余状态** （residual states，指训练过程中需临时保存的中间数据）。

为减少所需 GPU 数量，研究者探索了“异构训练”——**利用 CPU 的大容量内存扩展模型训练规模** [15,17,46,54,56,67]。例如：

::: block-1
- ZeRO-Offload[55]等卸载策略会精心划分硬件间的计算任务：GPU 执行计算密集型操作（如反向传播），CPU 执行内存密集型操作（如优化器更新）。
- ZeRO-Infinity 进一步将该思路扩展至 NVMe 存储（一种高速固态硬盘接口标准），使有限 GPU 资源能训练更大规模的模型[52]。
- Deep-OptimizerStates 则通过“从 CPU 向 GPU 读取优化器状态，并在双设备上并行更新参数”，扩展了 ZeRO-Offload 的能力，从而减少关键路径中的优化器步骤时间[32]。
- 与 ZeRO-Infinity 类似，SpeedLoader 将参数和优化器状态卸载到 CPU 内存，然后逐层动态加载到 GPU[72]。
:::

**尽管这些方案已展现出潜力，但它们的设计均基于一个假设：GPU 与 CPU 通过 PCIe 连接，带宽有限** （例如，<32GB/s）。因此，:mdnice-bold-italic[这些方案的核心目标是“减少 GPU 与 CPU 之间的通信量”，避免 PCIe 成为性能瓶颈]。然而，**超级芯片等新型硬件（其 C2C 带宽较 PCIe 提升 30 倍）的出现，对这一假设提出了挑战。若仍遵循传统设计原则，将导致性能欠佳** ——这也促使我们重新设计卸载方案。

## 三、挑战与机遇

> NVIDIA 的 GH200 Grace Hopper 超级芯片代表了高性能计算领域的重大进步。然而，这种新型硬件架构对“现有市售卸载方案”提出了挑战——**现有方案难以充分利用超级芯片的所有资源** （包括 Hopper GPU、Grace CPU、C2C 带宽），原因如下：

### 3.1 超级芯片 ≠ GPU + CPU

在超级芯片出现之前，CPU 与 GPU 通过 PCI Express（PCIe）连接。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/07.webp)

::: image-caption
表 1：GPU 节点对比。DGX-2 用于 ZeRO-Offload[55]，DGX-A100 用于 LLaMA 模型训练[65]，GH 代表单块 Grace Hopper 超级芯片
:::

例如，如表 1 所示，Meta 用于训练 LLaMA 模型[65]的 DGXA100 节点[39]，包含 2 个 AMD Rome CPU 和 8 个 NVIDIA A100 GPU，通过 PCIe 4.0 x16 连接，带宽仅为 64GB/s；而 ZeRO-Offload[55]中评估的 DGX-2 节点，包含 2 个 Intel Xeon CPU 和 16 个 NVIDIA V100 GPU，CPU 与 GPU 通过 PCIe 3.0 x16 连接，带宽仅为 32GB/s。

然而，**超级芯片通过“同一封装内的高带宽互联”将 GPU 与 CPU 紧密耦合** ——:mdnice-bold-italic[例如，GH200 的 NVLink-C2C 带宽高达 900GB/s，是标准 PCIe 4.0 的 14 倍]。此外，:mdnice-bold-italic[72 核的 Grace CPU 不仅具备 500GB/s 的 LPDDR5 高带宽内存（提供更大内存空间）]，还能高效处理计算任务，显著提升了计算能力。

### 3.2 无空闲调度的挑战

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/08.webp)

::: image-caption
图 3：ZeRO-Offload 中 GPU 和 CPU 的计算空闲时间。F：前向传播（Forward），B：反向传播（Backward），G：梯度（Gradient），P：参数（Parameter），Optimizer Step：优化器步骤
:::

> 现有基于卸载的方案常因“GPU-CPU 同步”导致 GPU 和 CPU 出现空闲时间。以 ZeRO-Offload[55]为例，如图 3 所示，存在三个主要约束：

1. **全局梯度同步（Global Gradient Synchronization）**：梯度裁剪（gradient clipping，防止梯度爆炸的技术）和归一化（normalization）需要全局信息。因此，:mdnice-bold-italic[CPU 必须等待接收所有 GPU 的梯度后，才能开始优化器步骤——这导致 GPU 执行反向传播时，CPU 处于空闲状态；]
2. **同步参数更新（Synchronized Parameter Updates）**：GPU 必须等待所有 FP16 参数从 CPU 返回后，才能启动下一轮迭代的前向传播——这种同步是确保前向传播使用“更新后参数”的关键；
3. **计算与通信难以完全重叠（Limited Overlap of Computation and Communication）**：即使采用“小桶划分”（将梯度传输到 CPU、将更新后参数传输回 GPU），也无法实现计算与通信的完全重叠（详见§4.3）。因此，:mdnice-bold-italic[在最后一个桶的传输阶段，GPU 和 CPU 均处于空闲状态。]

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/09.webp)

::: image-caption
图 4：现有基于卸载的方案导致 GPU 和 CPU 均出现空闲时间。我们在“单块超级芯片”和“单个节点”上，使用“卸载方案（ZeRO-Offload）可容纳的最大模型规模”和“避免内存溢出的最大批次大小”进行评估。结果显示，每次训练迭代中，GPU 的空闲时间占总执行时间的 40%-50%
:::

::: block-1
如图 4 所示，在每次训练迭代中，Hopper GPU 的空闲时间占比高达 40%-50%。这些显著的空闲时间引发了一个关键问题：**如何在超级芯片上减少基于卸载方案的低效性？**
:::

### 3.3 超级芯片上混合精度训练的挑战

> 现代深度学习训练常采用混合精度训练[34]：通过“尽可能在矩阵核心（如 TensorCore，GPU 中专门用于矩阵运算的硬件单元）上执行低精度操作”以提升计算速度，同时“用高精度存储关键信息”以保证模型精度。

混合精度训练使卸载变得更复杂——因为计算图中需额外添加“类型转换操作”（casting operations）。为通过卸载实现高效混合精度训练，传统设计原则是采用“贪心边切割算法”（greedy edge-cut algorithm）[55]。例如，ZeRO-Offload 将所有优化器状态保存在 CPU 上，在反向传播时将 FP16 梯度传输到 CPU，在优化器步骤时将更新后的 FP16 参数传输回 GPU。这种设计可最小化 GPU 与 CPU 之间的通信量（例如，仅传输 4Ψ 字节），从而避免 PCIe 成为瓶颈。然而，我们的详细性能分析表明，这种方法在超级芯片架构上表现欠佳（§4.5）——这也促使我们重新设计“基于卸载的混合精度训练”方案。

此外，还存在其他挑战，例如：Grace ARM CPU 缺乏高性能 Adam 优化器；缺乏“多超级芯片上扩展模型规模和长序列训练”的明确方案。在下文，我们将介绍 SuperOffload 的设计——它专门针对上述所有挑战，最终为超级芯片打造了适用于多种实际场景的高度优化方案。

## 四、SuperOffload 的设计与实现

### 4.1 系统概述

> SuperOffload 充分考虑超级芯片（Superchip）的紧耦合架构，引入了**超级芯片感知的数据流图（Superchip-aware DFG，简称 SA-DFG）** 。

在 SA-DFG 中，每个节点$u$仍代表一个张量算子，但同时还会反映该算子分配给 Hopper GPU 或 Grace CPU 执行时所需的计算开销；**边$u \to v$则反映了若算子$u$和$v$分别在 Hopper GPU 或 Grace CPU 上执行时，两者之间的通信开销**。此时，卸载策略可视为对 SA-DFG 的二分划分（即决定每个算子和数据在 GPU 还是 CPU 上处理）。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/02.webp)

::: image-caption
图 5. SuperOffload整体架构。从宏观层面来看，SuperOffload采用以超级芯片为中心的设计，通过多种技术的结合，合理优化Hopper GPU与Grace CPU之间的数据放置、计算以及张量迁移，同时充分利用NVLink-C2C
:::

图 5 展示了 SuperOffload 的整体架构——这是一个专为超级芯片上的 LLM 训练设计的系统，通过合理优化 Hopper GPU 与 Grace CPU 之间的数据布局、计算分配及张量迁移，实现高效训练。具体而言，该系统通过以下多种技术的组合达成目标：

1. 分析 SA-DFG 中各算子的计算与通信开销，采用**自适应权重驻留与权重流动策略** （4.2 节），以充分利用 Hopper GPU 和 Grace CPU 在异构计算上的效率优势；
2. 调整桶划分（bucketization）策略，并利用**细粒度重分配** （4.3 节），平衡超级芯片上不同资源的工作负载；
3. 提出一种全新的**推测-验证（speculation-then-validation）算法** （4.4 节），替代传统的“同步后执行”模式，实现 Hopper GPU 与 Grace CPU 的高效计算重叠；
4. 为混合精度训练设计**超级芯片感知的类型转换（Superchip-aware Casting）** （4.5 节）；
5. 开发针对 Grace CPU 的高度优化 Adam 优化器——**GraceAdam** （4.6 节）；
6. 扩展 SuperOffload，使其支持 ZeRO 风格的数据并行和 Ulysses 风格的序列并行（4.7 节），以适配多超级芯片训练场景和长序列训练需求。

### 4.2 自适应权重驻留与权重流动卸载

> 在 LLM 训练的卸载场景中，**核心问题之一是确定需从 GPU 卸载到 CPU 的模型状态**。尽管 CPU 拥有更大的内存容量，但模型状态在 CPU 与 GPU 之间的迁移仍会产生时间延迟。已知 LLM 训练的前向传播和反向传播计算复杂度均为$O(bsz \cdot \Psi)$（其中$bsz$为批大小，$\Psi$为模型参数数量），而优化器状态更新的计算复杂度为$O(\Psi)$，:mdnice-bold-italic[因此当前主流的卸载系统通常会将优化器状态及更新过程卸载到 CPU[52,55]——这种方式既能显著降低 GPU 内存占用，又可避免将计算密集型操作转移到 CPU。]

然而，针对模型权重的卸载决策仍存在两种不同方向：

1. **权重驻留（weight-stationary）**：将模型权重始终存放在 GPU 上[55]；
2. **权重流动（weight-flow）**：分批次加载模型权重[52]。

例如，ZeRO-Offload 将 FP16 精度的权重驻留在 GPU 上，同时将优化器状态卸载到 CPU，可使 GPU 内存占用降低约 8 倍；而 ZeRO-Infinity 则会将部分 FP16 权重卸载到 DRAM（内存）或 NVMe（硬盘），从而支持单 GPU 训练更大规模的模型。

> 随着超级芯片架构的出现及长上下文扩展等训练后任务的需求，一个关键问题随之产生：在超级芯片上训练 LLM 时，是否应将 Hopper GPU 中的 FP16 权重卸载到 Grace CPU，以进一步扩展模型规模？

为分析在超级芯片上将 FP16 权重卸载到 Grace CPU 的理论可行性，我们通过**峰值计算吞吐量（$peak\_{tp}$）** 和**数据迁移带宽（$bw$）** 估算训练效率，相关计算公式如下：

$$
\text{comp_time} = \frac{\text{total_computation}}{\text{peak}_{tp}} \tag{1}
$$

$$
\text{comm_time} = \frac{\text{total_data_movement}}{bw} \tag{2}
$$

$$
\text{efficiency} = \frac{\text{comp_time}}{\text{comp_time} + \text{comm_time}}
$$

在 LLM 的前向传播中，计算量主要由线性层操作主导，可近似表示为参数数量、序列长度与批大小的函数（即$2 \times bsz \times seq \times params$）。前向传播过程中，模型参数至少需从 CPU 加载到 GPU 一次，因此数据迁移量（$total\_data\_movement$）为$2 \times params$字节（按 FP16 精度计算，每个参数占 2 字节）。需说明的是，此处$peak_{tp}$采用**实际可达峰值** 而非硬件理论峰值。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/10.webp)

::: image-caption
图6. 带宽对效率的影响
:::

> 为实现数据迁移与计算的完全重叠，效率需超过 50%；若考虑通信延迟及其他开销，效率理想情况下应超过 60%。**图 6 显示，即便在 C2C（芯片到芯片）单向理论峰值带宽为 450 GB/s 的情况下，当序列长度为 1024 时，批大小需大于等于 4 才能使效率超过 60%。**

该分析表明，最优卸载策略具有**场景依赖性**：

::: block-1
- 在扩展模型规模时，为避免内存溢出，单个 GPU 通常只能处理较小的微批大小（micro-batch size）。例如，训练 Bloom[62]、Megatron-LM[60]、Turing-NLG[35]等大规模 LLM 时，微批大小常设为 1 或 2；
- 当批大小和序列长度增大时（如长上下文扩展的训练后任务[10,13]），激活内存可能远大于模型状态内存。例如，一个 70 亿参数的模型，其模型状态需 112GB 内存，但当序列长度为 100 万 token 时，激活内存需求可达 2TB。此时，在 Hopper GPU 与 Grace CPU 之间迁移 FP16 权重反而更具成本效益。
:::

基于此，SuperOffload 同时支持权重驻留和权重流动两种策略，并能根据不同实际场景自适应选择卸载策略。

### 4.3 细粒度桶划分重分配

> 卸载类方案面临的一大挑战是：由于 GPU 与 CPU 在数据迁移期间会处于空闲状态，因此难以同时充分利用计算资源和互联资源。

为解决这一问题，现有工作采用**桶划分（bucketization）策略** [55]——将模型状态（如权重、梯度、优化器状态）划分为多个“桶”，以细粒度方式实现 GPU 计算（如反向传播中的矩阵乘法）与数据换入/换出（如梯度从 GPU 迁移到 CPU）的重叠。**理论上，只要某个桶的计算内核完成，即可启动该桶的数据迁移操作。**

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/09.webp)

::: image-caption
图 4：现有基于卸载的方案导致 GPU 和 CPU 均出现空闲时间。我们在“单块超级芯片”和“单个节点”上，使用“卸载方案（ZeRO-Offload）可容纳的最大模型规模”和“避免内存溢出的最大批次大小”进行评估。结果显示，每次训练迭代中，GPU 的空闲时间占总执行时间的 40%-50%
:::

从理论上讲，桶划分应能实现 GPU 计算、CPU 优化器步骤及 GPU-CPU 数据迁移的完全重叠。但实际测试发现，Hopper GPU 仍会出现较长时间的空闲如图 4 所示。深入分析后可知，核心原因在于**Hopper GPU 与 Grace CPU 的浮点运算能力（FLOPS）比值约为 330**，远高于传统架构（如 DGX-2 的 60.39、DGX-A100 的 135.65）。这种计算能力差距使得负载平衡难度显著增加。

以最后一个桶为例，要在 CPU 上执行该桶的优化器状态更新，需完成三个阶段：

::: block-1
1. 梯度从 GPU 换出到 CPU；
2. 在 CPU 上执行 Adam 优化算法；
3. 将更新后的参数从 CPU 换入到 GPU。
:::

由于前向传播的第一个参数依赖于反向传播最后一个梯度的更新结果，上述三个阶段均无法与 GPU 的任何计算重叠——GPU 必须等待优化器更新和权重迁移完全完成后，才能启动下一轮迭代的前向传播。因此，**最后一个（或多个）桶的处理延迟会直接暴露在关键路径中，需通过精心调整计算分配以提升超级芯片的利用率。**

与现有研究[55]不同，SuperOffload 通过重分配 SA-DFG，将最后几个桶的优化器状态和梯度存放在 GPU 上（前提是 GPU 内存可容纳）。此外，为实现计算与通信的重叠，卸载到 CPU 的最后一个梯度桶必须在下次迭代开始前完成优化步骤，并将更新后的参数返回给 GPU。若用$n$表示存放在 GPU 上的优化器状态桶数量，则需满足以下约束：

$$
move_{bucket}(grad) + step_{cpu}(bucket) + move_{bucket}(param) \leq bwd(n \cdot bucket) + step_{gpu}(n \cdot bucket) \tag{4}
$$

其中：

- $step_{cpu/gpu}(bucket)$表示在 CPU/GPU 上完成一个桶的优化器步骤所需时间；
- $bwd(n \cdot bucket)$表示在 GPU 上完成$n$个桶的反向传播所需时间，可近似表示为参数数量、序列长度与批大小的函数（即$4 \times bsz \times seq \times bk \times n / 4$，其中$bk$为单个桶的大小）。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/11.webp)

::: image-caption
图7. GH200带宽测量
:::

::: block-1
分析表明，**桶大小（$bk$）是关键系统参数**。:mdnice-bold-italic[图 7 展示了不同张量大小下 GPU 与 CPU 之间的带宽变化：C2C 带宽随张量大小增加而提升，直至张量大小达到约 64MB 时趋于饱和]。

基于这一观察，SuperOffload 采用以下桶划分策略：在 GPU 与 CPU 之间迁移梯度和参数前，先将其分组为 64MB 的桶。该策略既能确保充分利用 C2C 带宽，又能提供细粒度的计算-通信重叠单元。
:::

最终，SuperOffload 通过网格搜索（grid search）确定需在 GPU 上保留的最优桶数量——该数量取决于模型大小和批大小，以隐藏 Grace CPU 的执行延迟和数据迁移延迟。

### 4.4 推测-验证（Speculation-then-Validation，STV）

> 在大多数卸载方案中，为确保数值稳定性，CPU 与 GPU 之间需在优化器步骤中进行同步。例如：

- 梯度范数裁剪（gradient clipping）需计算全局梯度范数[45]；
- 混合精度训练需全局检查 NaN（非数字）和 INF（无穷大）值[34]。

这两种场景均要求 CPU 等待所有梯度从 GPU 接收完成后，才能启动优化器步骤和权重更新。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/08.webp)

::: image-caption
图 3：ZeRO-Offload 中 GPU 和 CPU 的计算空闲时间。F：前向传播（Forward），B：反向传播（Backward），G：梯度（Gradient），P：参数（Parameter），Optimizer Step：优化器步骤
:::

正如图 3 中的灰色块所示，这种依赖关系会将优化器步骤暴露在关键路径中，导致其无法与反向传播重叠。

为解决这一局限，我们提出**推测-验证（STV）算法** ——在保留训练精确收敛特性的前提下，大幅减少上述同步操作。该机制基于一个关键观察：**大多数情况下，全局状态（如梯度范数、NaN/INF 检查结果）对优化器步骤无影响**。例如，梯度裁剪在训练初始预热阶段后很少被触发（此时梯度方差显著降低[30]）；而健康的训练过程中，混合精度训练也极少出现 NaN/INF 值（数值不稳定问题通常不会发生）。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/12.webp)

::: image-caption
图 8. SuperOffload的“先推测后验证”调度机制。它使优化器步骤（U）能够与反向传播（B）重叠进行，从而消除同步瓶颈并提高资源利用率
:::

基于上述观察，我们提出用**“推测-验证”机制替代传统卸载方案中的“同步-执行** （synchronization-then-execution，STE）”模式，具体流程如图 8 所示：

::: block-1
- CPU 无需等待所有梯度到达，而是利用当前已接收的梯度**推测执行优化器步骤**；
- 同时，将梯度裁剪和归一化操作推迟到 CPU 空闲时段（与 GPU 执行下一轮迭代的前向传播并行进行）。
:::

为保证优化过程的同步性，我们还实现了优化器步骤的**原地回滚（in-place rollback）** 机制。

因此，STV 是一种**精确优化策略** ——通过让 CPU 推测执行优化器步骤（与下一轮 GPU 前向传播并行），确保训练正确性。具体而言：

::: block-1
- 验证过程（检查 NaN/INF 值和梯度裁剪违规）通过 Python 多进程实现，验证结果通过多进程队列传递给 GPU；
- 前向传播完成后，GPU 检查是否需要回滚，存在两种回滚场景：
:::

推测-验证调度带来两大核心优势：

1. **使 CPU 上的优化器步骤能与 GPU 上的反向传播重叠**，而非将所有优化器步骤按顺序置于关键路径中（后者会严重降低吞吐量）；
2. **将归一化计算和数值检查的时间开销转移到 CPU 原本的空闲周期** （GPU 执行前向传播时），从而提升整体资源利用率。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/08.webp)

::: image-caption
图 3：ZeRO-Offload 中 GPU 和 CPU 的计算空闲时间。F：前向传播（Forward），B：反向传播（Backward），G：梯度（Gradient），P：参数（Parameter），Optimizer Step：优化器步骤
:::

与图 3 中 ZeRO-Offload 的调度相比，SuperOffload 将单轮迭代的关键路径从：

$$
\begin{aligned}
T_{iter} &= fwd_{gpu} + bwd_{gpu} + move_{bucket}(grad) \\
&\quad + step_{cpu} + move_{bucket}(param)
\end{aligned}
$$

缩短为：

$$
T_{iter} = fwd_{gpu} + bwd_{gpu} + step_{cpu}(bucket) \tag{7}
$$

有趣的是，**理论上 STV 调度下 SuperOffload 的迭代时间可能比非卸载训练更短**：

- 非卸载训练中，优化器需在 GPU 上处理整个模型的参数更新；
- 而 SuperOffload 仅需在 GPU 上更新部分参数（其余参数在 CPU 上更新）。

这与“卸载必然导致性能损失”的传统认知形成对比——在后续评估中，我们将证明 SuperOffload 确实能比纯 GPU 方案更快。

### 4.5 超级芯片感知的类型转换（Superchip-Aware Casting，SAC）

> 在 PyTorch、DeepSpeed 等深度学习训练框架中，混合精度训练通过**计算图重写（graph rewriting）** 实现[47]：所有算子的默认精度为 FP32，混合精度训练会将部分模型状态（如权重、梯度）在 FP32 与 FP16 之间转换。例如，反向传播生成的梯度为 FP16 精度，而优化器需使用 FP32 精度的梯度进行更新计算。

因此，$move_{bucket}(grad)$和$move_{bucket}(param)$这两个操作不仅涉及 GPU 与 CPU 之间的张量迁移，还包含张量数据类型的转换。当结合卸载策略时，这些额外的类型转换操作会变得更为复杂。现有卸载方案通常采用**最小边割算法（minimum edge cut algorithm）** 划分计算图[55]，其隐含假设是“最小化 GPU 与 CPU 之间的通信量即可提升性能”——:mdnice-bold-italic[但这些方案忽略了类型转换本身的开销。考虑到 Hopper 与 Grace 之间的 C2C 链路带宽是传统 PCIe 的一个数量级，上述假设是否仍成立？]

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/13.webp)

::: image-caption
图9. GPU与CPU上转换操作的时间成本对比（包括数据传输开销）
:::

我们通过**实验对比了两种类型转换+数据迁移方案的时间开销，结果如图 9 所示**：

- 在超级芯片上，“CPU 端转换+FP16 迁移（$Cast_{cpu} \leftrightarrow Move_{fp16}$）”的执行时间约为“GPU 端转换+FP32 迁移（$Cast_{gpu} \leftrightarrow Move_{fp32}$）”的 2 倍（具体比值取决于混合精度训练中输入张量的大小）。
- :mdnice-bold-italic[在大多数场景下（如张量大小为 256MB 至 2048MB 时），“GPU 端转换+FP32 迁移”的性能显著优于“CPU 端转换+FP16 迁移”。]

::: block-1
基于上述分析，SuperOffload 选择采用“GPU 端转换+FP32 迁移”方案：尽管该方案理论上的通信量更大（FP32 每个参数占 4 字节，FP16 占 2 字节），但整体开销更低。
:::

此外，我们还发现**类型转换与固定内存（pinned memory） 存在复杂的交互关系**：“迁移后转换”操作会在 Grace CPU 上触发非固定临时内存缓冲区的分配——该缓冲区用于存储从 Hopper GPU 换出的 FP16 梯度，之后再将其转换为 FP32 精度。:mdnice-bold-italic[这种情况下，数据迁移实际上通过非固定内存进行，其速度远慢于 FP32 精度下的 DMA（直接内存访问）迁移。这一观察进一步验证了“GPU 端转换+FP32 迁移”方案的合理性。]

### 4.6 GraceAdam 优化器

> 细心的读者可能会提出疑问：**考虑到 GPU 与 CPU 的浮点运算能力比值显著提升，超级芯片中的 Grace CPU 是否足以胜任优化器步骤和权重更新任务？** 现有工作提出了 CPUAdam[55]——一种通过多级并行优化的 CPU 端 Adam 实现，可加速 CPU 上的优化器计算。

然而，原始 CPUAdam 主要针对 x86 架构设计，依赖 AVX2/AVX512 指令集——这些指令集在 ARM CPU 上并不支持。此外，x86 实现依赖固定宽度的 SIMD（单指令多数据）向量，而 ARM 的**可扩展向量扩展（Scalable Vector Extension，SVE）** 采用“长度无关的向量处理范式”，需要更具适应性的算法。最后，ARM CPU 的内存层次结构与 x86 不同（缓存大小、延迟、带宽特性均有差异），需精心设计内存访问模式和预取策略。

为解决上述局限，我们提出**GraceAdam** ——专为 Grace ARM CPU 优化的 Adam 实现，包含以下多项优化技术：

::: block-1
1. **SVE 集成**：采用 SVE 的长度无关编程模型，通过`svcntw()`函数在运行时动态确定向量长度；将标量操作替换为 SVE 专用内在函数（如`svld1_f32`、`svmla_f32_m`、`svsqrt_f32_m`），这些函数可同时对整个向量进行操作，从而降低计算开销。
2. **增强型内存管理**：
3. **并行执行**：采用双层并行策略——通过 OpenMP 多线程实现跨多核 ARM Grace CPU 的高效扩展，同时利用 SVE 实现核心内的指令级并行。
:::

这些优化策略共同作用，使 GraceAdam 能最大化 Grace CPU 的吞吐量，其速度显著优于 PyTorch 在 Grace CPU 上的默认 Adam 实现。

### 4.7 多超级芯片调度

> 本节将介绍如何通过结合 ZeRO 风格的数据并行（ZeRO-DP）[51]和 Ulysses 风格的序列并行（Ulysses-SP）[22]，将 SuperOffload 扩展到多超级芯片环境。

#### ZeRO-3 集成

SuperOffload 保留了 ZeRO-3（完全分片数据并行，Fully Sharded Data Parallel）的模型状态分片策略：每个 GPU 将部分梯度和优化器状态卸载到 CPU 内存，而分片后的权重以及通过自适应卸载（4.2 节）确定的最后几个桶仍保留在 GPU 上。

在多超级芯片系统中，“先分片再卸载”的核心优势在于：每个并行进程仅负责更新模型参数的一个子集。这种设计确保所有 GPU 到 CPU 的总通信量保持不变，同时 CPU 资源可并行协作，共同完成统一的权重更新。

#### SuperOffload-Ulysses（长序列训练扩展）

> SuperOffload 还可与 Ulysses-SP（序列并行）结合使用。**当前 LLM 领域的一个明确趋势是：训练和微调的序列长度不断增加**。典型案例包括 Claude 的上下文窗口从 10 万 token 扩展到 20 万 token[3]，以及 GPT-4 Turbo 从 8 千 token 扩展到 12.8 万 token[43]。:mdnice-bold-italic[这一趋势由实际应用需求驱动——处理长文档、图像、视频等数据需要模型支持更长的序列长度。]

Ulysses-SP[22]通过以下方式解决序列并行的挑战：**沿序列维度对输入数据进行分片，并采用高效的“全对全（all-to-all）”集合通信策略处理注意力计算**。然而，即便启用激活检查点（activation checkpointing），模型状态的固定 GPU 内存占用仍会严重限制序列长度的扩展能力。

我们**通过 SuperOffload 的卸载能力优化内存管理，将 Ulysses-SP 与 SuperOffload 集成**：具体而言，:mdnice-bold-italic[SuperOffload 的自适应权重流动策略会将优化器状态和大部分模型权重卸载到 CPU 内存，显著降低 GPU 内存占用。这使得 Ulysses-SP 能在每层处理更长的序列，而不受 GPU 内存限制]。我们将这种组合策略命名为**SuperOffload-Ulysses**。

::: block-1
实验表明，SuperOffload-Ulysses 支持的序列长度是 Ulysses-SP 的 8 倍，仅需 8 台 GH200 超级芯片即可实现 130 亿参数模型的百万 token 序列训练，且模型浮点运算利用率（MFU）可达 55%（详见 5.3 节）。
:::

#### NUMA 绑定

在一个包含$K$个超级芯片的节点中，存在$K$个 Hopper GPU 和$K$个 Grace CPU，且每个超级芯片被配置为一个独立的**NUMA 节点（非统一内存访问节点，指内存访问延迟随处理器位置变化的架构）**。以$K=4$为例，一个 4 超级芯片节点共有 288 个 CPU 核心（每个 Grace CPU 含 72 个核心）。

默认情况下，**训练启动器可能会将进程随机分配到某些 CPU 核心，这可能导致“CPU 进程与 GPU 不在同一超级芯片上”的问题**。若发生这种情况，数据迁移将通过超级芯片间的连接（通常为 Slingshot interconnect）进行——其带宽远低于 NVLink-C2C 连接。:mdnice-bold-italic[为避免这一性能损失，SuperOffload 会将每个进程显式绑定到特定核心，确保 CPU 与 GPU 的最优亲和性] （affinity）。

## 五、评估

> 本节通过全面实验评估 SuperOffload 的性能，将其**与当前主流方案对比**，验证其**在不同模型规模和超长序列训练场景下的高效性与可扩展性**；同时，还将分析 **SuperOffload 中各优化组件对整体性能的贡献。**

### 5.1 评估方法

#### 硬件环境

- 单超级芯片实验：使用 1 台 GH200 超级芯片（96GB HBM 内存，480GB DDR 内存）；
- 多节点实验：使用 8 台 GH200 NVL2 节点，每个节点包含 2 台 GH200（96GB HBM 内存，240GB DDR 内存），节点间通过 HPE/Cray 的 200Gbps Slingshot 11 互联。

#### 工作负载

性能评估聚焦于**GPT/LLaMA 类 Transformer 模型** [49,65]：:mdnice-bold-italic[通过调整隐藏层维度和 Transformer 块数量，生成不同参数规模的模型]；训练数据集采用 The Pile 数据集的子集[12]。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/14.webp)

::: image-caption
表4. 评估中的模型配置
:::

#### 对比基线

将 SuperOffload 与当前主流的十亿级参数训练方案对比，基线方案包括：

| Baseline方案名称 | 核心特点 | 参考来源 |
| --- | --- | --- |
| PyTorch DDP | 标准PyTorch Transformer实现，采用DistributedDataParallel（DDP）实现多GPU间数据并行 | [27] |
| Megatron | 广泛广泛使用的大规模模型训练方案，通过模型并行技术高效扩展Transformer模型 | [60] |
| ZeRO-2 | 数据并行的扩展方案，通过对梯度和优化器状态进行分片，消除多GPU间的内存冗余 | [51] |
| ZeRO-3 | ZeRO-2的增强方案，除梯度和优化器状态外，进一步对模型参数在多GPU间进行分片 | [51] |
| ZeRO-Offload | 基于ZeRO-2构建，将梯度和优化器状态卸载到CPU内存，是DeepSpeed的默认CPU卸载方案 | [55] |
| ZeRO-Infinity | ZeRO-3与ZeRO-Offload的扩展方案，支持利用GPU、CPU内存和NVMe存储（实验仅启用CPU卸载），通过优化内存管理、数据预取和计算-通信重叠实现超大规模模型训练 | [52] |
| FSDP-CPU Offload | 带CPU卸载的完全分片数据并行（FSDP）方案，扩展FSDP功能，将非活跃使用的模型状态（参数、梯度、优化器状态）卸载到CPU以节省GPU内存 | [73] |

### 5.2 训练吞吐量

#### 单超级芯片场景

> 在单超级芯片上，对比 SuperOffload 与 PyTorch DDP、FSDP-Offload、ZeRO-Infinity、ZeRO-Offload 在不同模型规模下的训练吞吐量。

由于 Megatron 和 ZeRO-2/3 未改变单 GPU 的工作负载（与 PyTorch DDP 类似），且无法训练参数超过 40 亿的模型（会出现内存溢出 OOM 错误），因此未将其纳入对比。

**所有方案采用相同的批大小；若大批次导致 OOM 错误，则采用以下两种策略：**

1. 减小微批大小，同时启用梯度累积（gradient accumulation）；
2. 启用激活检查点，使用最大可能的微批大小（避免 OOM）。

需说明的是，计算 TFLOPS 时排除了重计算量（以衡量有效计算吞吐量），并**对每种方案报告两种策略中的更高吞吐量。**

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/15.webp)

::: image-caption
图10. 在单个超级芯片上，批大小为8时，使用PyTorch DDP、FSDP-Offload、ZeRO-Infinity、ZeRO-Offload和SuperOffload的训练吞吐量
:::

> 图 10 展示了对比结果。有趣的是，:mdnice-bold-italic[SuperOffload 不仅优于现有卸载方案，还在所有测试模型规模下超过了纯 GPU 方案——这与“卸载必然导致性能损失”的传统认知相反]。具体而言：

- 与 PyTorch DDP 相比，SuperOffload 的吞吐量最高提升 67%，原因有二：
- 与 ZeRO-Offload 相比，SuperOffload 的平均吞吐量提升 2 倍（最高 2.5 倍），优势源于三方面：

SuperOffload 的吞吐量显著高于 FSDP-Offload 和 ZeRO-Infinity：

- FSDP-Offload 的吞吐量始终低于 15 TFLOPS，原因是其依赖 PyTorch 原生的 CPU-Adam 实现——后续 5.6 节将证明，原生实现比 GraceAdam 慢 3.5 倍，形成严重的效率瓶颈；
- ZeRO-Infinity 的吞吐量始终低于 50 TFLOPS，因其采用基于桶的梯度/参数迁移方案，未考虑 C2C 带宽的特性（小张量大小下带宽可低至 50 GB/s）。

#### 多超级芯片场景

> 在 4 台和 16 台 GH200 超级芯片上，对比 SuperOffload 与 Megatron、ZeRO-2、ZeRO-3、ZeRO-Offload 的训练吞吐量。由于 PyTorch DDP 无法扩展到更大模型规模，且 FSDP-Offload 和 ZeRO-Infinity 在单超级芯片场景下吞吐量显著较低，因此未将其纳入对比。

对于 Megatron，采用能实现最佳性能的模型并行（MP）度数；与单 GPU 实验类似，若出现 OOM 错误，则启用梯度累积（减小微批）或激活检查点（最大微批），并报告更高吞吐量。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/16.webp)

::: image-caption
图11. 在4块和16块GH200 GPU上，Megatron、ZeRO-2/3、ZeRO-Offload和SuperOffload的训练吞吐量对比。在4块GPU和16块GPU的实验中，我们分别使用了16和128的批处理大小
:::

图 11 展示了 4 台和 16 台超级芯片上的每 GPU 吞吐量结果，观察如下：

- **对于参数 50 亿至 200 亿的模型，SuperOffload 的吞吐量始终高于现有方案**：与 Megatron、ZeRO-2、ZeRO-3 相比，吞吐量最高分别提升 83%、46%、37%；:mdnice-bold-italic[与 ZeRO-Offload 相比，全模型规模下平均吞吐量提升 2.5 倍。这主要得益于 CPU 与 GPU 计算的高度重叠、无需激活检查点即可支持更大微批，以及这些技术对超级芯片 C2C 高带宽互联的充分利用] ——最终实现 GPU 和 CPU 资源的高效利用。
- Megatron 和 ZeRO-3 在 4 台 GPU 上处理 150 亿参数模型时因内存不足（聚合 GPU 内存不够）出现 OOM 错误，而 SuperOffload 可有效扩展到 500 亿参数模型；此外，**SuperOffload 在 16 台 GPU 上可高效训练 2000 亿参数模型，同时保持较高的计算吞吐量。**

### 5.3 解锁大规模长序列训练

> 将 SuperOffload 与 Ulysses[22]结合，构建**SuperOffload-Ulysses** 以支持长序列训练，并在 4 台和 8 台超级芯片上评估其性能（对比原生 Ulysses）。实验采用两种常用模型规模：130 亿参数和 300 亿参数；所有实验均使用最大可能的批大小（避免 OOM），并在激活内存较大时启用激活检查点。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/17.webp)

::: image-caption
图12. 使用SuperOffload-Ulysses和Ulysses时支持的序列长度及相应的MFU。OOM表示增加序列长度会导致内存不足（OOM）的临界点
:::

**如图 12 所示，SuperOffload-Ulysses 支持的序列长度是原生 Ulysses 的 8 倍**：例如，仅需 8 台超级芯片，SuperOffload-Ulysses 即可实现 130 亿参数模型的百万 token 序列训练，**且 MFU 达 55%。** 对于原生 Ulysses 可处理的序列长度，SuperOffload-Ulysses 的:mdnice-bold-italic[模型浮点运算利用率（MFU）始终更高——这一性能优势源于自适应卸载策略（4.2 节）：该策略能自适应检测超长序列，并动态将更多权重参数卸载到 CPU 内存]，为 GPU 释放更多内存以存储更大的激活值（无需启用激活检查点），从而在长序列长度下仍保持计算效率。

### 5.4 模型规模扩展能力

> 本节测试单超级芯片、4 台超级芯片和 16 台超级芯片上可训练的最大模型规模。

#### 单超级芯片场景

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/18.webp)

::: image-caption
图13. 可在单个超级芯片、4个超级芯片和16个超级芯片上训练的最大模型的规模
:::

图 13 显示：

- 单 GPU（96GB 内存）上，PyTorch DDP 可训练的最大模型为 35 亿参数；
- Megatron、ZeRO-2、ZeRO-3 无法训练比 PyTorch DDP 更大的模型，因它们均依赖聚合 GPU 内存存储模型状态；
- ZeRO-Offload 通过将优化器状态卸载到 CPU，可处理 150 亿参数模型；
- SuperOffload 可训练 250 亿参数模型——核心原因是其能自适应控制优化器状态和梯度的卸载比例：与 ZeRO-Offload（权重驻留 GPU、张量布局固定）不同，SuperOffload 可根据 GPU 和 CPU 内存情况，自适应选择权重驻留或权重流动策略，从而提升可训练模型规模；
- 尽管 ZeRO-Infinity 可训练与 SuperOffload 规模相当的模型，但其吞吐量显著更低（如图 10 所示）：SuperOffload 的平均吞吐量是 ZeRO-Infinity 的 6.7 倍（最高 12.6 倍），这主要归功于 STV、桶划分重分配等技术对 Hopper GPU 和 C2C 带宽利用率的提升。

#### 多超级芯片场景

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/18.webp)

::: image-caption
图13. 可在单个超级芯片、4个超级芯片和16个超级芯片上训练的最大模型的规模
:::

> 进一步在“1 台 GH200 节点（含 4 台超级芯片）”和“4 台 GH200 节点（含 16 台超级芯片）”上测试模型规模扩展能力，结果如图 13 所示：

- PyTorch DDP 的最大可训练模型规模未随 GPU 数量增加而变化，因其未解决数据并行中的内存冗余问题，可扩展性受限于单 GPU 的模型规模；
- Megatron、ZeRO-2/3 支持通过增加 GPU 数量扩展模型规模，但即便使用 16 台 GPU，也无法高效训练参数超过 500 亿的模型；其中，Megatron 支持的模型规模大于 ZeRO-2，因 ZeRO-2 在模型权重上仍存在内存冗余；
- ZeRO-Offload 基于 ZeRO-2 构建，要求每个 GPU 至少存储完整的 FP16 参数副本，因此即便增加 GPU 数量，其最大可训练模型规模仍限于 200 亿参数；

总体而言，在 **16 台超级芯片** 上，SuperOffload 的:mdnice-bold-italic[可训练模型规模分别是 PyTorch DDP、Megatron、ZeRO-2、ZeRO-3、ZeRO-Offload 的 57 倍、4.4 倍、10 倍、4.5 倍、10 倍。]

### 5.5 性能分解

> 通过逐一启用 SuperOffload 的**各优化组件，量化每个组件对整体性能的贡献**。实验设置与 5.2 节一致，以 50 亿参数模型为例进行演示。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/19.webp)

::: image-caption
表2. SuperOffload优化及其对吞吐量（TFLOPS）的影响细分，包括GraceAdam（第4.6节）、超级芯片感知转换（SAC，第4.5节）、先推测后验证（SVT，第4.4节）和桶重分区（第4.3节）
:::

表 2 详细展示了 SuperOffload 关键优化组件及其对训练吞吐量的累积影响：

- 基线配置（所有优化禁用）的吞吐量为 116.2 TFLOPS，与图 10 中 ZeRO-Offload 的吞吐量接近；

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/15.webp)

::: image-caption
图10. 在单个超级芯片上，批大小为8时，使用PyTorch DDP、FSDP-Offload、ZeRO-Infinity、ZeRO-Offload和SuperOffload的训练吞吐量
:::

- 启用 GraceAdam 后，吞吐量提升至 128.23 TFLOPS，增幅 10.4%；
- 进一步启用超级芯片感知的类型转换（SAC，4.5 节），吞吐量提升至 144.49 TFLOPS，额外增幅 12.7%——该优化通过平衡类型转换与数据迁移成本实现效率最大化；
- 推测-验证（STV，4.4 节）带来的性能提升最显著：吞吐量提升至 209.36 TFLOPS，增幅 45%——这表明 STV 通过将反向传播与 CPU 优化器步骤重叠，有效消除了 Hopper GPU 的计算空闲时间；
- 最后启用桶划分重分配（4.3 节），吞吐量进一步提升至 238.92 TFLOPS，增幅 14.1%。

综上，所有优化启用后，SuperOffload 的吞吐量较基线配置提升 2.06 倍。:mdnice-bold-italic[这一分解结果表明，每个组件均针对性解决了超级芯片架构下大规模训练的特定性能瓶颈，其组合效应实现了显著的 multiplicative（倍增）收益。]

### 5.6 GraceAdam 的执行性能

> 本节评估为 ARM 架构优化的 GraceAdam，将其与 PyTorch CPU 原生 Adam、ZeRO-Offload 的 CPU-Adam（针对 Intel x86 优化）对比。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/20.webp)

::: image-caption
表3. PyTorch原生CPU-Adam、CPU-Adam和GraceAdam的Adam延迟（秒）
:::

表 3 通过测量 10 亿至 80 亿参数模型的优化器执行时间，评估 GraceAdam 的性能：

::: block-1
- 与 PyTorch CPU 原生实现（PT-CPU）相比，GraceAdam 在所有配置下的速度均提升 3 倍以上；
- 即便与已优化的 CPU-Adam 相比，GraceAdam 的执行速度仍提升 1.36 倍。
:::

这些:mdnice-bold-italic[性能提升源于 GraceAdam 对 ARM 可扩展向量扩展（SVE）的利用，以及优化的内存管理和 OpenMP 多线程策略——这些技术使 Grace CPU 成为 SuperOffload 系统中高效且高性能的计算组件。]

### 5.7 推测-验证（STV）的正确性与开销评估

> 4.4 节提出的推测-验证（STV）是一种精确优化策略，但调度逻辑较复杂。为验证其正确性和开销，我们:mdnice-bold-italic[在 16 台超级芯片上用 The Pile 数据集训练 1750 亿参数的 GPT 类模型，共训练 8 万轮迭代。]

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/21.webp)

::: image-caption
图14. 在GPT 176B模型80,000次迭代的训练过程中的训练损失和回滚发生情况。红点表示因梯度裁剪、NaN或INF值而触发回滚的迭代
:::

图 14 展示了训练损失曲线及回滚发生次数，观察如下：

::: block-1
- 训练损失呈现预期的收敛趋势；
- 迭代 1 至 1000 轮期间，回滚频繁发生——因训练初始阶段易出现梯度裁剪和 NaN/INF 值；
- 迭代 1000 轮后，训练趋于稳定，回滚极少发生：1000 至 80000 轮期间仅发生 93 次回滚，占总迭代次数的 0.12%；
- **对于 1750 亿参数模型，16 台 Grace CPU 并行执行的回滚操作平均每次耗时 2 秒，8 万轮迭代中回滚总耗时不足 200 秒** ——开销可忽略不计，且完全保留了训练精度。
:::

### 5.8 GPU 空闲时间测量

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/09.webp)

::: image-caption
图 4：现有基于卸载的方案导致 GPU 和 CPU 均出现空闲时间。我们在“单块超级芯片”和“单个节点”上，使用“卸载方案（ZeRO-Offload）可容纳的最大模型规模”和“避免内存溢出的最大批次大小”进行评估。结果显示，每次训练迭代中，GPU 的空闲时间占总执行时间的 40%-50%
:::

与现有方案导致 GPU 空闲（如图 4 所示）不同，我们在相同模型和硬件设置下测量 SuperOffload 的 GPU 利用率。

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/22.webp)

::: image-caption
图 15. SuperOffload充分利用了GPU资源
:::

如图 15 所示，SuperOffload 实现了近乎 100%的 GPU 利用率，有效消除了空闲时间，最大化了计算效率。

## 六、结论

> 本文在超级芯片架构下的大规模 LLM 训练领域取得了突破性进展：通过分析超级芯片上 LLM 训练的关键挑战与性能影响因素，设计了**SuperOffload** ——:mdnice-bold-italic[首个能同时充分利用 Hopper GPU、Grace CPU 和 NVLink-C2C 互联的系统。]

全面的评估结果表明：

1. SuperOffload 不仅优于当前主流的卸载方案，还**解锁了新的系统能力** （:mdnice-bold-italic[如仅需 8 台超级芯片即可实现百万 token 序列长度的 LLM 训练]）；
2. **即便没有大量 GPU 资源，研究人员和从业者也能通过 SuperOffload 实现大规模 LLM 训练**，显著降低了大规模 LLM 训练的门槛。

## 参考文献

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/23.webp)

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/24.webp)

![文章插图](/articles/2025-12-21-superoffload-superchip-llm-4-50b-24164476/images/25.webp)
