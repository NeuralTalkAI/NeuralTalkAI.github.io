关建词：PD-Multiplexing（PD多路复用）、LLM Serving、GreenContext、Goodput（有效吞吐量）、SLO（服务级别目标）、GPU Resource Scheduling（GPU资源调度）

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/01.webp)

- PD-Multiplexing: Unlocking High-Goodput LLM Serving with GreenContext
- https://lmsys.org/blog/2025-09-28-pdmux/
- 简要参考文档：https://github.com/ykcombat/sglang/blob/e84aa1bdd055df93e603a46fa6ca5e60afd213f5/docs/advanced_features/pd_multiplexing.md
- 后续会发布更详细文档
- 本文 6603 字，阅读需 22 分钟，播客 15 分钟

相关推荐

- [训练即部署！SpecForge：加速 SGLang 推测性解码训练，让16GB GPU支持70B模型训练，加速比达2.18倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892155&idx=1&sn=5c2c834d047a75a4da3e1b52ff14a30d&scene=21#wechat_redirect)
- [整套软硬件协同系统设计！攻克内存墙(带宽和容量)：极长上下文推理的优化路径！实现 A100 2.24 倍吞吐量的推理加速器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892472&idx=1&sn=a65b145aced249a6066b19964f402ccb&scene=21#wechat_redirect)
- [SGLang 新手指南：理解高效执行LLM系统的秘诀](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888085&idx=1&sn=d9373b4ff322eda9e4ca5a84db846ff2&scene=21#wechat_redirect)

> 在大语言模型（LLM）服务中，“预填充”（处理输入生成初始缓存）和“解码”（生成后续输出）两个阶段常抢用GPU资源，导致服务效率低、难保证响应速度（SLO）。

为此，SGLang框架推出了新方案：**预填充/解码的多路复用方案（PD-Multiplexing），核心是借助NVIDIA GPU的GreenContext能力** ——能在单个进程里精细划分GPU资源，破解资源竞争难题（https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html）。

::: block-1
![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/02.webp)

::: image-caption
图 1. PD 多路复用（PD-Multiplexing）概述
:::

我们认为这种范式是 **LLM 服务部署的一种极具潜力的新方案**，能够为“**模型即服务（MaaS）** ”提供更可靠的服务级别目标（SLO，指服务提供者与用户约定的服务质量标准，如延迟上限、可用性等）保障和更高的有效吞吐量。
:::

之前的解决办法要么把GPU资源固定分给不同任务，要么把预填充拆成固定块，前者资源浪费多、后者难平衡效率，都有明显缺陷。**而PD-Multiplexing让同一服务实例里的GPU资源能灵活共享**：缓存池大家一起用，计算资源按需分配，还设计了**“无空闲引擎”** （拆预填充块避免GPU空等）和**“保速度调度器”** （既保证响应达标，又最大化预填充效率）。

实验显示，在H200、8块A100等硬件上，**这个新方案比旧方案的“有效吞吐量”最高提升3倍多，还能加快首条输出的生成速度，同时满足响应时间要求**。未来它还会支持更多功能，进一步优化工业场景的适配效果。

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/03.webp)

## 关键问题

### 问题 1：PD-Multiplexing 跨非 NVIDIA 硬件架构的适配性与瓶颈

> PD-Multiplexing 核心依赖 NVIDIA GPU 的 GreenContext 轻量级分区能力，若部署环境切换至 AMD GPU、自研 AI 芯片等非 NVIDIA 架构硬件，其“GPU 空间共享”“SM 动态分配”等核心设计将面临何种技术瓶颈？目前是否有明确的跨硬件架构适配路径，以避免方案陷入“硬件绑定”的落地局限？

基于原文信息，PD-Multiplexing 的核心设计**深度依赖 NVIDIA GPU 专属的 GreenContext 能力** ——该能力是实现“进程内轻量级 GPU 资源细粒度分区”“同一实例内 SM 资源动态分配”的前提，:mdnice-bold-italic[原文未提及对 AMD GPU、自研 AI 芯片等非 NVIDIA 架构的适配方案。]

从技术逻辑看，其面临的**核心瓶颈有两点：**

::: block-1
1. **硬件功能缺失**：非 NVIDIA 芯片普遍缺乏与 GreenContext 对等的“细粒度资源分区”硬件接口，无法支持 KV 缓存池的高效共享与 SM 资源的动态调度，若强行通过软件层模拟，会引入额外开销，可能抵消 PD-Multiplexing 的性能优势；
2. **架构依赖固化**：方案中“无气泡复用引擎”对 GPU 计算单元（SM）的调度逻辑、KV 缓存的存储管理，均基于 NVIDIA GPU 的架构特性设计，跨硬件后需重构核心调度模块，原文未提及相关技术储备或适配路径。
:::

原文仅在“未来规划”中提到“优化工业级 SLO 适配”，:mdnice-bold-italic[未涉及跨硬件架构的方向]，因此目前**尚无明确的非 NVIDIA 硬件适配方案**，存在“硬件绑定”的落地局限。

### 问题 2：超大规模 LLM 与多模态模型的适配性及 SLO 风险

> 实验中 goodput 提升 3.06 倍的结论基于 CodeLlama-34b-hf（H200）、Llama3.1-70B（8 A100）两类特定规模的纯 LLM 模型，若迁移至 100B+超大规模 LLM 或包含文本、图像等输入的多模态模型，预填充与解码阶段的资源竞争逻辑会发生何种变化？SLO 感知调度器能否避免因模态/参数规模扩容导致的 SLO 违约或性能骤降？

原文实验仅覆盖**CodeLlama-34b-hf（中规模）、Llama3.1-70B（大规模）两类纯文本 LLM**，未涉及 100B+超大规模 LLM 或多模态模型，因此需基于原文设计逻辑推导潜在问题：

::: block-1
1. **超大规模 LLM 的资源竞争变化**：100B+模型的 KV 缓存体积会显著增大（相同序列长度下，参数规模翻倍可能导致 KV 缓存占用量翻倍），此时“KV 缓存池共用”可能面临资源耗尽风险——若多个请求同时生成长序列，会挤占解码阶段的缓存空间，导致解码效率下降；同时，预填充阶段的计算量增加，可能延长单块预填充时间，削弱“无气泡复用引擎”消除空闲的效果。
2. **多模态模型的额外挑战**：多模态输入（如图像、音频）会在预填充阶段增加“模态编码”步骤（如将图像转为特征向量），导致预填充阶段的计算复杂度远超纯文本 LLM，预填充与解码的资源竞争从“文本计算资源争夺”变为“模态编码+文本计算+缓存存储”的多重竞争，:mdnice-bold-italic[原文的 SLO 感知调度器【仅】针对“文本 LLM 的 ITL（Inter-Token Latency）”设计，未考虑多模态场景下“模态编码延迟”对 SLO 的影响，可能出现“ITL 达标但整体请求延迟突破 SLO”的情况。]
:::

原文未提及对超大规模 LLM 或多模态模型的适配规划，因此目前方案在这类场景下**存在 SLO 违约或性能骤降的风险**，需进一步验证与优化。

### 问题 3：突发流量与长序列场景下的兜底机制

> 工业级 LLM 服务常面临突发流量（如请求量瞬时翻倍）与长序列生成（如 1000+token 输出）场景，PD-Multiplexing 的“无气泡复用引擎”在拆分预填充块时，是否会因突发请求挤占解码资源，或因长序列 KV 缓存占用过高导致新请求 TTFT 延迟突破 SLO？针对这类极端场景，是否有动态资源优先级调整的兜底机制？

原文明确 PD-Multiplexing 通过“无气泡复用引擎（拆分预填充块）”和“SLO 感知调度器（保障 ITL SLO）”提升性能，但**未提及针对“突发流量”“长序列生成”等极端场景的动态兜底机制**，基于原文设计可推导【潜在问题】：

::: block-1
1. **突发流量的资源挤占**：当请求量瞬时翻倍时，“无气泡复用引擎”会拆分更多预填充块以填充 GPU 空闲时间，但过多预填充块可能挤占解码阶段的 SM 资源——原文未设计“预填充/解码资源优先级动态调整”逻辑，可能导致解码延迟升高，突破 ITL SLO；
2. **长序列的 TTFT 延迟风险**：长序列（如 1000+token）生成会长期占用大量 KV 缓存资源，而原文“KV 缓存池共用”未设置“缓存占用上限”或“长序列缓存驱逐策略”，新请求可能因缓存资源不足需等待释放，导致 TTFT（Time to First Token）延迟超出 SLO 要求。
:::

:mdnice-bold-italic[原文“未来规划”仅提及“支持 MTP、投机解码”“优化工业级 SLO 适配”，【未明确】针对极端场景的【兜底机制】]，因此目前方案在突发流量、长序列场景下**缺乏明确的资源优先级调整或缓存管理策略**，可能无法稳定满足 SLO。

## 本文目录

- 关键问题
  - 问题 1：PD-Multiplexing 跨非 NVIDIA 硬件架构的适配性与瓶颈
  - 问题 2：超大规模 LLM 与多模态模型的适配性及 SLO 风险
  - 问题 3：突发流量与长序列场景下的兜底机制
- 一、LLM 服务中的有效吞吐量：一项长期存在的挑战
- 二、PD 多路复用：一种全新的服务范式
- 三、借助 GreenContext 实现无气泡多路复用引擎
- 四、分析与制定调度策略
- 五、基准测试
  - 5.1 不同块大小下与分块预填充（Chunked-prefill）的对比
  - 5.2 真实世界轨迹（Real-world Traces）上的测试结果
  - 5.3 多样化任务上的测试结果与调度可视化
- 六、未来工作
  - 6.1 支持多 token 预测与投机解码
  - 6.2 面向更真实的工业级 SLO 与细粒度参数控制
  - 6.3 与 PD 分离（PD-Disaggregation）的集成
  - 6.4 兼容 PyTorch 2.6 以上版本
  - 6.5 支持专家混合模型（MoE Model）

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/04.webp)

## 一、LLM 服务中的有效吞吐量：一项长期存在的挑战

> 大规模提供模型即服务（MaaS），要求 LLM 服务系统在不牺牲吞吐量的前提下，持续满足严格的服务级别目标（SLO）。

在实际场景中，这意味着要满足推理两个阶段已明确的延迟 SLO：

::: block-1
- 预填充阶段（prefill phase，指 LLM 处理输入序列并构建初始键值缓存的阶段）的**首 token 生成时间（TTFT，即模型生成第一个输出 token 所需的时间）**；
- 解码阶段（decode phase，指模型基于预填充阶段的结果，持续生成后续输出 token 的阶段）的**token 间延迟（ITL，也称为 token 间隔时间 TBT，即连续两个输出 token 之间的时间间隔）**。
:::

挑战在于，同一服务实例上会交替进行预填充和解码操作，导致 GPU 资源竞争。目前已出现两种常见的 SLO 合规保障方案：

::: block-1
1. **实例级 PD 分离（PD-disaggregation）**：将预填充和解码拆分到不同实例中。但:mdnice-bold-italic[该方案存在【权衡取舍】]：GPU 资源会被静态划分，且:mdnice-bold-italic[实例间的键值缓存迁移会增加额外复杂度]，需要高性能的互连设备和:mdnice-bold-italic[通信库支持]。
2. **序列级分块预填充（chunked-prefill）**：将长序列拆分为更小的块，每个块与一次解码迭代融合，以控制 ITL。:mdnice-bold-italic[该方案同样存在精细的【权衡问题】：块大小需在严格的 ITL 保障与高 GPU 利用率之间找到平衡。]
:::

尤其在为实际 LLM 服务设定严格的 SLO 阈值时，上述两种方案（分离和分块）的缺陷会愈发明显。

## 二、PD 多路复用：一种全新的服务范式

> 为此，我们提出了一种全新的服务范式——PD 多路复用（PD-Multiplexing），以实现更高的有效吞吐量。

该范式**通过 GPU 内部空间共享** （intra-GPU spatial sharing，指:mdnice-bold-italic[在单块 GPU 内部将计算资源按空间维度分配给不同任务]），**在同一实例内对预填充和解码阶段进行多路复用**。它具备以下几项重要优势：

::: block-1
- 预填充和解码在同一实例内共享同一个键值缓存（KV cache）池，无需进行代价高昂的跨实例缓存迁移；
- GPU 内部空间共享支持 GPU 计算资源（即流多处理器 SMs）随工作负载变化，在预填充和解码之间动态分配；
- 这种共享还实现了预填充与解码执行的分离，确保在满足严格 ITL SLO 的同时，不会影响预填充性能。
:::

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/02.webp)

::: image-caption
图 1. PD 多路复用（PD-Multiplexing）概述
:::

如图 1 所示，PD 多路复用（PD-Multiplexing）包含两个核心模块：

1. **无气泡多路复用引擎**：可独立且高效地执行预填充和解码阶段；
2. **SLO 感知调度器**：可迭代生成符合 SLO 要求的多路复用计划。

## 三、借助 GreenContext 实现无气泡多路复用引擎

> 我们基于 GreenContext 构建了这一全新范式——GreenContext 是 NVIDIA GPU 从 CUDA 12.4 版本开始引入的一项功能，支持轻量级的进程内空间共享。

简单来说，从 CUDA 12.6 版本起，我们可以创建多个具有专属流多处理器（SM）分配的 CUDA 流（CUDA stream，GPU 上用于调度内核执行的序列），以实现并发 GPU 内核运行。**借助 GreenContext，GPU 资源可在预填充和解码阶段之间动态划分，实时适应 SLO 要求、工作负载模式及其他服务需求。**

为兼容现有的服务架构，我们采用单线程调度方式对预填充和解码进行多路复用，而非为每个阶段创建单独的线程。选择这种设计的另一原因是，Python 的全局解释器锁（GIL，Python 中的一种机制，确保同一时间只有一个线程执行 Python 字节码）仍会阻碍真正的并行执行，且在未来版本中仍将作为默认机制。幸运的是，:mdnice-bold-italic[预填充和解码均为异步操作，这使得该设计具备可行性。通过在预填充和解码的专属 GreenContext 流之间切换调度，我们即可实现多路复用。]

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/05.webp)

::: image-caption
图 2. 消除气泡以实现高效的预填充-解码多路复用
:::

然而，这种 GreenContext 集成方式会引入 GPU 气泡（bubble，指 GPU 资源处于空闲状态的时间片段）。如图 2(a)所示，气泡的产生源于两个原因：

::: block-1
1. **预填充阶段的启动时间远长于解码阶段** （解码阶段仅涉及单个 CUDA 图，CUDA 图是一种将多个 CUDA 操作记录为一个图形结构以加速重复执行的技术）。在某些情况下，启动预填充阶段的时间甚至比执行整个解码阶段的时间还要长，导致 GPU 资源闲置；
2. **解码阶段的迭代次数具有不确定性**。当一个解码批次中的所有请求提前完成时，若预填充已启动，则预先分配的流多处理器（SM）可能会处于未充分利用的状态。
:::

**为解决这一问题，我们将预填充阶段拆分为更小的预填充块，如图 2(b)所示**。由于预填充的计算密集度通常远高于解码，这种块级拆分带来的开销可忽略不计，同时能有效消除多路复用过程中的 GPU 气泡。

## 四、分析与制定调度策略

> 在实现无气泡多路复用引擎后，下一个挑战是对预填充块和解码批次进行调度。

离线分析（offline profiling，指在实际服务运行前，对代表性工作负载进行的性能数据采集与分析）表明，在 GreenContext 环境下，预填充和解码两个阶段会竞争资源。**【根本原因】在于，尽管 GreenContext 会对 SM 进行划分，但不会对内存带宽进行划分，这使得资源竞争难以建模。**

为解决这一问题，我们对代表性工作负载进行离线分析，并利用分析结果训练一个延迟预测器（latency predictor，一种通过历史数据预测任务延迟的模型），该预测器将用于驱动我们的 SLO 感知调度策略。由于建模过程依赖于特定的模型和硬件环境，此处暂不展开细节，但未来我们将提供详细教程，包含实用的分步指导。

调度策略的核心思路十分简单：为解码阶段分配刚好满足 ITL SLO 所需的流多处理器（SM），剩余所有 SM 均分配给预填充阶段。同时，确定要启动的预填充块数量。通过这种方式，解码阶段始终能在严格的 SLO 保障下运行，而预填充阶段则能以最快速度执行，从而扩大解码批次规模。

## 五、基准测试

> 总而言之，我们在多种工作负载和设备上，将 PD 多路复用（PD-Multiplexing）与多个基准方案进行了对比评估。

评估过程如下：

::: block-1
- 首先展示一个易于复现的实验；
- 然后通过真实世界轨迹（real-world traces，指记录实际服务中用户请求特征的数据序列）和多样化任务，证明 PD 多路复用的优势；
- 最后提供运行时调度细节的全景可视化结果。
:::

在广泛的评估中，**PD 多路复用相比当前最先进的基准方案**，有效吞吐量提升最高可达 **3.06 倍**。

以下结果仅用于研究目的。在实际应用中，SLO 要求通常更为具体。此处通过这些基准测试展示 PD 多路复用的潜力。

### 5.1 不同块大小下与分块预填充（Chunked-prefill）的对比

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/06.webp)

::: image-caption
图 3. 在单块运行 CodeLlama-34b-hf 模型的 H200 GPU 上，ShareGPT 和 LooGLE 工作负载的测试结果
:::

我们首先在单块运行 CodeLlama-34b-hf 模型的 H200 GPU 上，将 PD 多路复用与不同块大小的分块预填充方案进行对比评估。图 3 报告了 99 分位数的首 token 生成时间（TTFT）和 token 间延迟（ITL）。我们将 ITL 的 SLO 目标设定为 60 毫秒（ms），未对 TTFT 施加 SLO 约束，而是通过报告 TTFT 的 99 分位数来体现 PD 多路复用的效率。图中实心点表示对应基准方案满足 ITL SLO 要求，空心点表示该基准方案违反 ITL SLO 要求。

结果清晰地体现了 PD 多路复用的优势：**在持续满足严格 ITL SLO 目标的同时，PD 多路复用实现了最快的 TTFT**。相比之下，:mdnice-bold-italic[分块预填充方案往往需要将块大小减小到 1024 以下才能满足如此严格的 ITL 要求，这会导致预填充性能下降，且 GPU 资源未被充分利用]。这种优势在长上下文工作负载（如 LooGLE）中更为显著——此类场景下，分块预填充的低效问题会被放大。复现说明可参考：https://github.com/sgl-project/sglang/pull/10692。

### 5.2 真实世界轨迹（Real-world Traces）上的测试结果

> 我们还利用真实世界轨迹“Mooncake-Tool&Agent”对 PD 多路复用进行了评估，并将其与分块预填充、PD 分离方案进行对比

- 所有方案均基于 SGLang 代码库实现，以确保对比的公平性。
- 实验在一台配备 8 块 A100 GPU 的服务器上进行，分块预填充的块大小设为 512，PD 分离方案中预填充与解码实例的比例（P:D）为 1:1，并启用了前缀缓存共享（prefix cache sharing，指多个请求共享输入序列前缀对应的 KV 缓存，以减少重复计算的优化技术）。

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/07.webp)

::: image-caption
图 4. 在 8 块 A100 GPU、运行 Llama3.1-70B 模型的服务器上，Mooncake-Tool&Agent 轨迹的测试结果
:::

- 图 4(a)展示了 TTFT 和 ITL 的测试结果。**与分块预填充相比，PD 多路复用在这两个指标上均有提升；与 PD 分离相比，PD 多路复用的 TTFT 显著更短，且两种方案均满足解码阶段的 SLO 要求**。
- 为评估 PD 多路复用对有效吞吐量的影响，我们逐步提高请求速率，并测量 SLO 达标率。如图 4(b)所示，PD 多路复用能维持显著更高的有效吞吐量——**相比分块预填充和 PD 分离，有效吞吐量分别提升了最高 3.06 倍和 1.62 倍。**

### 5.3 多样化任务上的测试结果与调度可视化

> 我们进一步在三个代表性任务上评估 PD 多路复用：OpenThoughts、ShareGPT 和 LooGLE。这些任务具有不同的工作负载模式：

- OpenThoughts：预填充输入最短，解码输出最长；
- ShareGPT：预填充输入较长，解码输出较短；
- LooGLE：预填充输入最长，解码输出最短。

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/08.webp)

::: image-caption
图 5. 在 8 块 A100 GPU、运行 Llama3.1-70B 模型的服务器上，OpenThoughts、ShareGPT 和 LooGLE 任务的测试结果。PD 多路复用始终保持优异性能，有效吞吐量显著高于基准方案。
:::

![文章插图](/articles/2025-10-03-pd-sglang-nvidia-greencontext-4fc91240/images/09.webp)

::: image-caption
图 6. 在 8 块 A100 GPU、运行 Llama3.1-70B 模型的服务器上，OpenThoughts、ShareGPT 和 LooGLE 任务的调度可视化结果
:::

为直观展示其实际工作机制，图 6 提供了运行时调度决策的全景时间线。如图所示，PD 多路复用能动态调整资源分配：

- 对于 OpenThoughts 任务，为预填充分配最少资源；
- 对于 LooGLE 任务，为解码分配最少资源。

时间线还进一步展示了调度器如何随工作负载变化，无缝切换不同的 SM 分配方案，确保同时满足 SLO 合规性和高资源利用率。

## 六、未来工作

> PD 多路复用为模型即服务（MaaS）部署展现出巨大潜力。我们接下来将重点关注以下几个方向：

### 6.1 支持多 token 预测与投机解码

具备多 token 预测（MTP，Multi-Token Prediction，指模型一次生成多个 token 的机制，如 DeepSeek-V3 模型采用的机制）的模型，需要调整 PD 多路复用中的调度策略，以保障其 SLO。

此外，采用更复杂验证方法（涉及多个协作 LLM）的投机解码方案，也需要不同的资源划分策略。我们将为这些场景扩展 PD 多路复用的专属策略，并报告最终的端到端性能提升。

### 6.2 面向更真实的工业级 SLO 与细粒度参数控制

基于 PD 多路复用实现高有效吞吐量服务，需要针对目标模型和硬件进行一次性分析（profiling）。

- **目前已提供简要参考文档**：https://github.com/ykcombat/sglang/blob/e84aa1bdd055df93e603a46fa6ca5e60afd213f5/docs/advanced_features/pd_multiplexing.md

**我们还将发布详细的实操教程**，帮助用户复现我们的分析流程，并设计符合真实工业场景的高效调度策略。

### 6.3 与 PD 分离（PD-Disaggregation）的集成

尽管在上述评估中，我们将 PD 多路复用与 PD 分离进行了对比，但这两种方案并非互斥——它们可以**集成以进一步提升有效吞吐量**。:mdnice-bold-italic[例如，在工作负载变化的场景下，解码实例往往无法维持最大批次规模，此时可将其替换为 PD 多路复用实例。]

通过**将解码阶段与多路复用的预填充阶段重叠执行，系统可充分利用原本会被浪费的资源。**

### 6.4 兼容 PyTorch 2.6 以上版本

目前所有测试结果均基于 PyTorch 2.6 版本复现。

对于更新版本，我们在结合 CUDA Graphs 与 NCCL（NVIDIA Collective Communications Library，NVIDIA 集体通信库，用于多 GPU 间数据通信）时，遇到了 NVIDIA 内部问题。

具体而言，当使用 PyTorch 2.6 以上版本时，启动分布式 CUDA 图会产生显著开销。我们正与上游开发者密切合作，解决这一漏洞，以实现与未来 PyTorch 版本的顺畅兼容。

### 6.5 支持专家混合模型（MoE Model）

我们在 Qwen-235B 模型上的初步实验表明，**PD 多路复用仍能实现稳定的性能提升**。

我们**计划发布针对更大规模专家混合模型的全面测试结果**，例如 DeepSeek-V3，并在更大规模的分布式环境中进行评估。

目前，我们已实现 PD 多路复用的概念验证（proof-of-concept）版本(https://github.com/sgl-project/sglang/pull/10692)，其全面集成到 SGLang 的路线图(https://github.com/sgl-project/sglang/issues/10813)正在推进中。
