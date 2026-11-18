关键词：AutoTriton、Triton 编程、强化学习（RL）、监督微调（SFT）、TritonBench、KernelBench、GRPO 算法

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/01.webp)

- AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs
- https://arxiv.org/abs/2507.05687
- https://github.com/AI9Stars/AutoTriton
- 本文 7618 字，阅读需 30 分钟，播客 7 分钟如下

Triton 相关文章

- [GPU Kernel Scientist：LLM 驱动下的 GPGPU Kernel 迭代框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889551&idx=1&sn=606751b8d2b4190238f0fb800f1d5751&scene=21#wechat_redirect)
- [模块化 Triton GPU 内核代码生成 Agent 架构 GEAK：生成正确率达 63%，速度提升 2.59 倍！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890465&idx=1&sn=a73eec7d10391ef7371018529fb2405c&scene=21#wechat_redirect)

- [ML-Triton：Intel 在 Triton GPU 编程的多级编译与语言扩展的工作](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887927&idx=1&sn=07061b7b0fe72cdb4bfd91e42f9431b2&scene=21#wechat_redirect)
- [超越 Triton？任意低精度 GPGPU 计算虚拟机，加速 LLM 服务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888475&idx=1&sn=c452ca77513dd283a6850181735ecaa0&scene=21#wechat_redirect)

- [KPerfIR：扩展 Triton 面向 GPU 的开源编译器中心性能工具，以 8.2% 开销换 24.1% FA3内核性能提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890015&idx=1&sn=cafc8d3ec7baebab7417835090e56242&scene=21#wechat_redirect)
- [PipeThreader：软件定义流水线驱动的 DNN 编译器，FlashAttention与Mamba2性能显著超越现有方案！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889807&idx=1&sn=a8f2bfdca2091144fcd6059a575dda66&scene=21#wechat_redirect)

> 深度学习中的核（kernel，指深度学习中执行特定计算任务的函数或模块）开发需要**跨硬件优化计算单元，同时通过大量经验调优平衡内存管理、并行性和硬件特定优化**。

尽管像 Triton 这样的领域特定语言通过抽象底层细节简化了 GPU 编程，但开发者仍必须通过迭代实验手动调整关键参数（如块大小和内存访问模式），这对实现最佳性能和更广泛的采用造成了巨大障碍。

::: block-1
在这项工作中，我们提出了 AutoTriton，这是**第一个专门用于 Triton 编程的模型，由强化学习** （RL，一种让智能体通过与环境交互学习最优策略的机器学习方法）**驱动**。

:mdnice-bold-italic[AutoTriton 通过高质量的数据收集管道执行监督微调] （SFT，一种利用标注数据训练模型的方法），以具备基本的 Triton 编程知识，随后:mdnice-bold-italic[使用组相对策略优化（GRPO）算法进行强化学习]，结合基于规则的奖励和基于执行的奖励来进一步提升 Triton 编程能力。
:::

在 TritonBench 和 KernelBench 的五个评估渠道上的实验表明，我们的 **8B 参数模型 AutoTriton 实现了与主流大型模型** （包括 Claude-4-Sonnet 和 DeepSeek-R1-0528）**相当的性能**。进一步的实验分析证明了 AutoTriton 中每个模块的关键作用，包括 SFT 阶段、RL 阶段和奖励设计策略。

这些发现**凸显了强化学习在自动生成高性能核方面的潜力**，由于高性能核是 AI 系统的核心组件，这一突破为构建更高效的 AI 系统奠定了重要基础。该模型和代码将在：https://github.com/AI9Stars/AutoTriton 上发布。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/02.webp)

## 关键问题

### Q1. 在 AutoTriton 的两阶段训练中，RL 阶段通过 GRPO 算法和混合奖励机制实现了性能提升，但实验显示 SFT 阶段已通过高质量数据赋予模型基础能力。那么，在复杂 Triton 编程任务（如 TritonBench-G 中的真实世界内核）中，RL 带来的性能增益究竟是源于策略优化对探索空间的有效利用，还是更多依赖于 SFT 阶段数据 pipeline 提供的高质量先验？如何量化两者在突破“模仿学习天花板”中的具体贡献？

在复杂 Triton 编程任务中，:mdnice-bold-italic[RL 的性能增益同时依赖于 SFT 阶段高质量数据先验与策略优化对探索空间的利用]，两者贡献可通过实验数据初步量化。

::: block-1
- SFT 阶段通过精心设计的数据管道（如 GitHub 爬取、LLM 蒸馏与验证）为模型**奠定了 Triton 编程基础**，使模型在正确性和基本性能上超越原始骨干模型（Seed-Coder-8B-Reasoning）。
- RL 阶段通过 GRPO 算法和混合奖励机制，**进一步提升了性能**。
:::

在 TritonBench-T 中，AutoTriton（带 RL）的 Execution Accuracy 为 39.16，显著高于仅 SFT 版本的 34.94，且 fast1 指标从 15.06 提升至 17.04。表明 **RL 在突破模仿学习天花板中起到了关键作用**。这种:mdnice-bold-italic[增益源于 RL 对更广泛策略空间的探索]，尤其是在 :mdnice-bold-italic[OOD 数据上的适应能力]，但需依赖 SFT 提供的高质量初始策略以避免无效探索。

### Q2. 论文指出当前奖励机制虽通过规则校验（如@`triton.jit`）减少了“奖励黑客（hack）”行为，但仍存在模型生成“伪正确代码”的现象（如仅实现简单算子而将复杂部分保留为 PyTorch 调用）。这种情况下，现有奖励函数（执行正确性+语法规则）是否本质上无法区分“真正高性能的 Triton 实现”与“仅满足最低要求的敷衍实现”？若引入性能导向的奖励（如运行速度），如何解决不同硬件环境下性能基准不一致导致的训练不稳定问题？

现有奖励函数确实无法完全区分“真正高性能的 Triton 实现”与“敷衍实现”，而引入性能导向奖励面临硬件环境差异的挑战。

论文指出，现有奖励（执行正确性+规则校验）虽减少了“奖励黑客”（如缺失`@triton.jit`），但模型仍会生成低质量代码（如仅实现简单算子而保留复杂部分为 PyTorch 调用），这表明**当前奖励难以衡量实现的“深度”与性能。**

关于引入性能奖励，**论文在局限性中提到，当前缺乏性能导向训练，未来需整合性能感知框架**，但:mdnice-bold-italic[未解决硬件差异问题——不同硬件的性能基准不一致可能导致奖励信号不稳定]，这与 RL 中奖励设计需鲁棒性的要求冲突，可能**需要结合硬件抽象层或相对性能指标** （如对比 PyTorch 基准的加速比）来缓解。

### Q3. AutoTriton 在基准测试中表现接近大模型，但依赖于精心设计的数据 pipeline（含 GitHub 爬取、LLM 蒸馏等）。当面对超出训练数据分布的极端场景（如超大规模张量运算、新型硬件架构适配）时，其数据 pipeline 的泛化能力是否会显著下降？RL 阶段的探索策略在缺乏高质量先验数据的“未知领域”中，能否避免陷入低效试错或退化至 SFT 水平？

数据管道在:mdnice-bold-italic[极端场景下的泛化能力可能下降]，且 :mdnice-bold-italic[RL 在缺乏高质量先验数据的“未知领域”中可能陷入低效试错]。论文提到：

::: block-1
- TritonBench-G（真实世界内核）中所有模型表现均较差，反映出数据管道对复杂、真实场景的覆盖不足；
- 而数据管道依赖现有 PyTorch 和 Triton 代码，面对超大规模张量或新型硬件时，训练数据分布与实际需求的偏差可能导致泛化下降。
:::

此外，RL 阶段虽纳入 OOD 数据，但论文指出其探索依赖 SFT 阶段的数据质量，**若缺乏高质量先验，可能难以突破 SFT 水平** ——正如“w/o SFT”模型存在严重奖励黑客行为，:mdnice-bold-italic[表明 SFT 是 RL 有效探索的基础，未知领域中先验不足可能导致 RL 退化。]

## 本文目录

- 关键问题
  - Q1. 在 AutoTriton 的两阶段训练中，RL 阶段通过 GRPO 算法和混合奖励机制实现了性能提升，但实验显示 SFT 阶段已通过高质量数据赋予模型基础能力。那么，在复杂 Triton 编程任务（如 TritonBench-G 中的真实世界内核）中，RL 带来的性能增益究竟是源于策略优化对探索空间的有效利用，还是更多依赖于 SFT 阶段数据 pipeline 提供的高质量先验？如何量化两者在突破“模仿学习天花板”中的具体贡献？
  - Q2. 论文指出当前奖励机制虽通过规则校验（如@`triton.jit`）减少了“奖励黑客（hack）”行为，但仍存在模型生成“伪正确代码”的现象（如仅实现简单算子而将复杂部分保留为 PyTorch 调用）。这种情况下，现有奖励函数（执行正确性+语法规则）是否本质上无法区分“真正高性能的 Triton 实现”与“仅满足最低要求的敷衍实现”？若引入性能导向的奖励（如运行速度），如何解决不同硬件环境下性能基准不一致导致的训练不稳定问题？
  - Q3. AutoTriton 在基准测试中表现接近大模型，但依赖于精心设计的数据 pipeline（含 GitHub 爬取、LLM 蒸馏等）。当面对超出训练数据分布的极端场景（如超大规模张量运算、新型硬件架构适配）时，其数据 pipeline 的泛化能力是否会显著下降？RL 阶段的探索策略在缺乏高质量先验数据的“未知领域”中，能否避免陷入低效试错或退化至 SFT 水平？
- 一、引言
- 二、相关工作
  - 2.1 用于核生成的大语言模型
  - 2.2 用于代码的强化学习
- 三、AutoTriton
  - 3.1 问题表述
  - 3.2 监督微调
  - 3.3 强化学习
- 四、实验
  - 4.1 评估设置
  - 4.2 主要结果
  - 4.3 分析
- 五、结论
- 参考文献

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/03.webp)

## 一、引言

高效的 Kernel 工程实现是高性能深度学习系统的基石，使模型能够在日益异构的硬件环境（指由不同类型硬件组成的计算环境，如 CPU、GPU、TPU 等混合架构）中实现最优执行[Abadi et al., 2016; Paszke et al., 2019]。历史上，在 CUDA 等低级语言中编写此类核是性能工程师的专属领域，这需要对硬件架构和复杂的并行编程模式有深入了解[Tillet et al., 2019]。类 Python 的 GPU 编程框架（最著名的是 Triton[Tillet et al., 2019]）的出现标志着可编程性的重大飞跃。**尽管有这些进步，但这种高层抽象并未完全消除性能调优的复杂性**。开发者:mdnice-bold-italic[仍然需要手动配置关键参数（如分块配置和数据布局），这一经验性的试错过程是实现性能可移植性和广泛采用的主要瓶颈。]

当前，AI 辅助 Kernel 生成的研究引起了越来越多的关注。

::: block-1
- 一些基准测试（如 TritonBench[Li et al., 2025]和 KernelBench[Ouyang et al., 2025]）已被引入，用于**系统评估大型语言模型（LLMs）生成高性能核的能力**。
- 除了基准测试外，最近的工作（如 AI CUDA Engineer[Lange et al., 2025]）也获得了广泛关注。该框架**利用通用大型语言模型作为基础组件构建自动化工作流**。然而，由于底层模型固有的能力边界，其适应性和灵活性仍然有限。
:::

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/04.webp)

> 在这项工作中，我们提出了 AutoTriton，这是第一个专门用于 Triton 编程的模型，由强化学习驱动。

- AutoTriton 基于 Seed-Coder-8B-Reasoning[Zhang et al., 2025]（一种专门用于编程的推理模型）构建，通过监督微调（SFT）和强化学习的协同结合进一步增强，如图 1 所示。
- 在 SFT 阶段，我们首先设计并实现了一个专门的数据构建管道。该管道有助于组装高质量的 Triton 数据集，该数据集明确阐述了 Triton 固有的关键编程概念和推理步骤，从而使 AutoTriton 具备基础的编程能力。
- 随后，我们再次利用从该管道生成的数据，:mdnice-bold-italic[结合基于规则和基于执行的奖励进行强化学习]。这一阶段鼓励模型探索和内化有效的 Triton 编程策略，**使其能够捕捉【仅】通过监督微调难以灌输的实际细节和效率**。

在两个典型基准测试 TritonBench 和 KernelBench 上的实验结果表明，**AutoTriton 仅用 8B 参数，就在所有五个基准测试渠道上实现了与主流大型模型** （包括 GPT-4o、Claude-4-Sonnet、Qwen3-32B、DeepSeek-R1-0528）**相当的性能**，这表明 :mdnice-bold-italic[AutoTriton 在 Triton 编程任务上的有效性]，并凸显了我们提出的数据收集管道和强化学习训练策略的关键影响。进一步的分析**强调了 SFT、RL 和奖励设计组件在 AutoTriton 中的关键作用**。这些发现为该领域的未来研究提供了我们认为至关重要的指导。

## 二、相关工作

### 2.1 用于核生成的大语言模型

> 计算核生成对于在多样化硬件上优化 AI 工作负载至关重要。典型方法包括 MLIR（LLVM 项目，2019）、TVM（Apache 软件基金会，2018）等，它们共同提升了 AI 模型的性能和可移植性，解决了现代异构计算环境的复杂性[Al-Dujaili et al., 2024]。

近年来，GPU 核生成的自动化（这对优化机器学习性能至关重要）引起了广泛的研究关注。

- 诸如 KernelBench（Ouyang et al., 2025）之类的基准测试为系统评估大语言模型在该领域的能力提供了便利，它**通过“fast p”等指标评估在多样化工作负载上生成快速且正确的核的能力**。尽管前沿模型在通用编程任务上表现出色，但它们:mdnice-bold-italic[在核生成任务上往往存在不足，这凸显了通用编码能力与专门的核优化需求之间的差距。]
- 同样，**TritonBench（Li et al., 2025）强调了大语言模型在处理 Triton 等领域特定语言时面临的挑战**，揭示了由于:mdnice-bold-italic[不熟悉 Triton 规范和 GPU 编程细节，生成高效核存在困难]。

除了基准测试外，像 **AI CUDA Engineer（Lange et al., 2025）这样的框架采用智能体方法，利用大语言模型进行 PyTorch 到 CUDA 的转换和迭代优化**。尽管实现了显著的加速，但这些:mdnice-bold-italic[无训练的方法从根本上受到基础大语言模型固有能力边界的限制]。为了直接增强模型能力，Kevin-32B（Baronio et al., 2025）**采用多轮强化学习，使模型能够从环境反馈中学习，并通过自我改进显著提高核的正确性和性能**，特别是在复杂任务上。此外，DeepSeek-R1 模型:mdnice-bold-italic[通过测试时扩展（指在推理时分配更多计算资源进行迭代优化和验证），展示了其在 KernelBench 任务上实现高正确性的有效性] （NVIDIA 开发者博客，2025）。

这些进展共同表明，:mdnice-bold-italic[一种趋势是采用迭代、反馈驱动的方法来增强大语言模型] 在专门的高性能代码生成方面的熟练程度。此外，**KernelBench（Fisches et al., 2025）通过监督微调生成 Triton 核**。尽管取得了合理的性能，但它从根本上受到模仿学习上限的限制。它无法利用探索（指模型主动尝试不同策略以发现更优解的过程），因此限制了其生成更高质量 Triton 核的能力。

> 与上述工作不同，在本研究中，我们提出了 AutoTriton，这是**第一个专为 Triton 编程设计的、基于强化学习的模型**，在五个典型基准测试通道上实现了显著改进。

### 2.2 用于代码的强化学习

强化学习提供了一种强大的范式，使智能体能够通过与动态环境的交互学习最优策略，最大化累积奖励。早期在代码生成中的应用将问题表述为马尔可夫决策过程（MDP，一种描述序贯决策的框架，其中每个状态的决策仅依赖于当前状态），其中部分程序构成状态，语法产生式作为动作（Chen et al., 2020）。这种表述凸显了**强化学习在适应不同抽象层次方面的灵活性**。现代进展利用:mdnice-bold-italic[大语言模型，将代码生成模型视为执行者，将代码生成视为动作，而从单元测试结果中获得的功能正确性作为奖励信号] （Le et al., 2022）。这种方法使 AlphaCode 等系统在复杂编码任务中取得了具有竞争力的性能（Li et al., 2022）。除了生成之外，**强化学习还广泛应用于代码优化，特别是用于学习编译器优化 passes 的最优序列** （Bendib et al., 2024；Shahzad et al., 2022）。在这里，:mdnice-bold-italic[状态通常由中间表示（IR）的统计分析或基于图的模型表示，奖励与诸如周期计数、面积或资源利用率等性能指标相关联] （Shahzad et al., 2022）。像 CYCLE 这样的框架的成功进一步证明了强化学习在对有缺陷的代码生成进行迭代自我改进、从执行反馈中学习以及显著提高改进能力方面的能力（Ding et al., 2024）。

> 尽管取得了这些进展，用于代码的强化学习仍面临重大挑战。

::: block-1
- **设计健壮的奖励函数仍然是一个主要问题**。设计:mdnice-bold-italic[不佳的奖励可能导致意外行为或“奖励欺骗”] （指智能体利用奖励结构而非实现预期目标）（Milvus, 2023）。
- **训练不稳定性**，特别是在**微调大型语言模型** 时，是另一个障碍，像 REINFORCE++ Hu（2025）这样的算法经常遭受不稳定的策略更新。
:::

为了解决这些不稳定性并增强训练收敛性，已开发出**改进的算法，如群相对策略优化（GRPO），它还有助于消除大语言模型强化学习框架中的奖励欺骗** （Shao et al., 2024）。我们提出的 AutoTriton 与这一观点一致。在核生成领域，:mdnice-bold-italic[我们采用 GRPO 算法并设计合理的奖励，以建立大语言模型对核查询的理解，并相应地进行基于强化学习的 Triton 编程。]

## 三、AutoTriton

在本节中，我们介绍 AutoTriton，这是一个**专为 Triton 编程任务定制的专用模型**。AutoTriton 具有序贯的两阶段过程。

- 最初，模型经过**监督微调（SFT）** 以:mdnice-bold-italic[建立 Triton 编程原理的坚实基础]。
- 在此之后，**应用强化学习框架**，该框架允许 GPU 代码的基于执行的反馈，:mdnice-bold-italic[指导模型进一步优化生成的核的正确性。]

> 为了阐明 AutoTriton，我们将首先正式进行任务表述，然后详细说明监督微调过程，随后介绍强化学习框架的设计。

### 3.1 问题表述

实现高性能的自定义算子 kernel 传统上需要大量的领域专业知识，并涉及大量的经验性试错。**为了加速这一开发生命周期，我们定义了 Triton 编程任务**。该任务旨在学习从全面的 kernel 规范到其相应的可执行 Triton 实现的映射。 :mdnice-bold-italic[kernel 规范 D 通常包括两个主要组件]：

::: block-1
- 具体的 PyTorch 实现或正式的接口定义，其中详细说明了其功能描述、输入和输出参数的签名（包括数据类型）。
- 所涉及张量的维度（形状）。
:::

核心挑战是开发一个模型 M，给定 D，合成一个 Triton 核 T，该 T 不仅在语法上正确且可执行，而且在语义上忠实于 D 中概述的所有要求。

### 3.2 监督微调

最近的研究（Li et al., 2025）强调，即使是精通通用编程的模型，在生成专门的 Triton 核方面也表现出有限的能力。为了弥合这一能力差距并使我们的模型具备基本的 Triton 编程专业知识，我们:mdnice-bold-italic[开发了一个细致的数据收集流水线，以生成高质量的数据 用于监督微调]。该:mdnice-bold-italic[流水线自动化了数据收集、合成和验证的关键步骤]，明确目标是仅保留高保真、语法正确且可证明正确执行的数据用于训练。**流水线的架构如图 2 所示。**

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/05.webp)

> 我们提出的数据获取流水线首先**系统地收集 PyTorch kernel**。

::: block-1
1. 这需要从 GitHub 和 HuggingFace 等成熟的开源平台收集核，并通过 PyTorch 接口补充基本核的算法组合。
2. 然后，我们利用精通编程的开源大语言模型（如 Qwen 2.5 Coder（Hui et al., 2024））自动生成测试用例，这些测试用例随后用于验证和保留可执行的 PyTorch 核。
3. 在收集 PyTorch 核之后，我们采用两种不同的策略来生成相应的 Triton 核：指令引导的大语言模型蒸馏和带有大语言模型增强改进的编译。
:::

认识到通用大语言模型在熟练生成 Triton 代码方面的固有局限性（Li et al., 2025），我们**还利用基于编译的方法来提高数据获取效率**。具体而言，使用 `torch.compile` 处理 PyTorch 代码片段。

- 然后，由大语言模型改进生成的编译产物以提高人类可读性；
- 这涉及诸如插入解释性注释、移除无关装饰器以及将变量重命名为更具语义意义的名称等任务。

在使用测试用例验证与 PyTorch 的功能等效性之后，我们**利用大语言模型来设计指令**。这些指令:mdnice-bold-italic[与经过验证的 Triton 代码一起，用于提示大语言模型生成详细的思维链叙述]，旨在在监督微调期间灌输 Triton 编程范式。

最后，包含`<指令，带思维链的 Triton 代码>`对的最终数据集用于监督微调。模型学习在指令提示的条件下预测 Triton 代码及其思维链合理性，从而培养 Triton 编程的基础能力。

### 3.3 强化学习

**为了进一步突破** AutoTriton 的**编码能力边界**，我们:mdnice-bold-italic[采用了常见的带可验证奖励的强化学习（RLVR）流水线]。我们的训练过程基于 GRPO 算法（Shao et al., 2024），使用群归一化目标更新策略：

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/06.webp)

其中 $\pi_{\theta}$ 和 $\pi_{\theta_{old }}$ 是策略模型和参考模型，$\hat{A}_{i, t}$ 是群-wise 优势：

$$
\hat{A}_{i,t}=\frac{r_{i}-\text{mean}\left(\left\{r_{j}\right\}_{j=1}^{N}\right)}{\text{std}\left(\left\{r_{i}\right\}_{i=1}^{N}\right)}
$$

奖励函数由以下方程定义，它结合了基于执行的组件和基于规则的组件：

$$
R(\hat{a})= \begin{cases}1, & \text{is_Triton} (\hat{a}) \& \text{test_passed} (\hat{a}) \\ 0, & \text{otherwise} \end{cases}
$$

其中:

- $\text{test_passed}$ 组件确认功能正确性。它验证生成的代码是否通过所有测试用例，并且在 ϵ 的 tolerance（容差，指允许的误差范围）内与参考 PyTorch 实现语义等效。
- $\text{is_Triton}$ 组件通过使用基于规则的检查器检查代码是否符合 Triton 语言规范来确保语法有效性。该组件规范策略，以阻止“奖励欺骗”，即模型可能生成非 Triton 代码（如更简单的 PyTorch 代码）而轻易通过功能测试的情况。

我们强化学习阶段的数据也是使用第 3.2 节中的流水线生成的。在这个阶段，我们只保留`<指令，PyTorch 代码>`对，因为**参考 PyTorch 代码足以通过测试用例执行导出奖励信号，无需标记的 Triton 代码**。这使得:mdnice-bold-italic[能够包含更难的、分布外（OOD，指与训练数据分布不同的数据）的数据]，:mdnice-bold-italic[这些数据非常适合强化学习探索]。最终的训练数据是这些新实例和来自监督微调阶段的一小部分分布内数据的策略性混合，以确保平稳的策略过渡。

## 四、实验

> 在本节中，我们评估 AutoTriton 的性能，并从多个方面进行综合分析。

### 4.1 评估设置

#### 评估基准

我们使用两个已有的基准对 AutoTriton 进行评估：TritonBench（Li 等人，2025）¹ 和 KernelBench（Ouyang 等人，2025）²。

::: block-1
- **TritonBench 用于评估大语言模型（LLM）生成 Triton 内核的能力**，它分为两个评估渠道：TritonBench-G 包含 184 个来自 GitHub 的真实世界内核，TritonBench-T 包含 166 个与 PyTorch 接口对齐的内核。
- **KernelBench 则评估大语言模型在 250 个任务上生成用于神经网络优化的高效 GPU 内核的能力**，这些任务分为 3 个级别：Level 1（100 个单内核任务，例如卷积，用于 CUDA 替代）、Level 2（100 个简单融合任务，例如 conv+bias+ReLU，用于融合 CUDA 内核）和 Level 3（50 个完整架构任务，例如 MobileNet，用于端到端 CUDA 优化）。
:::

推理过程中使用的提示词详见图 5。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/07.webp)

#### 评估指标

关于评估指标，我们综合了上述两个基准的指标，将其分为两个方面：

::: block-1
1. 编译准确率（无错误编译）；
2. 调用准确率（无错误调用）；
3. 执行准确率（正确的输入输出行为）；
4. 两个基准上的加速比（相对执行时间改进）。
:::

遵循两个基准的原始设置：

- 在 KernelBench 上评估编译准确率、执行准确率和加速比
- 在 TritonBench 上分别评估调用准确率、执行准确率和加速比

> 对于 TritonBench-G 的评估，加速比是通过与提供的参考 Triton 代码比较得出的；对于所有其他评估，加速比是相对于 PyTorch 实现计算的。为遵循 KernelBench，我们报告 $\text{fast}_{p}$ 来衡量整个基准上 Triton 代码的绝对加速比，其计算方式如下：

$$
\text{fast}_{p}=\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left(\text{correct}_{i} \land \left\{\text{SpeedUp}_{i}>p\right\}\right)
$$

#### 训练细节

- 在**监督微调（SFT）阶段，我们使用 LLaMA-Factory 框架** （Zheng 等人，2024），数据集包含 14,102 个样本。
- 在**后续的强化学习（RL）阶段，我们采用 VeRL 框架** （Sheng 等人，2025），使用包含 6,302 个样本的数据集。

### 4.2 主要结果

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/08.webp)

> 表 1 和表 2 分别展示了 AutoTriton 在 TritonBench 和 KernelBench 上的实验结果。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/09.webp)

实验在 TritonBench 和 KernelBench 的**五个评估渠道上系统地进行**。

- 在:mdnice-bold-italic[正确性方面（表格中的 Exec），AutoTriton 明显优于包括 DeepSeek-R1-0528、GPT-4o、Claude-4-Sonnet、DeepSeek-R1-0120 和 Qwen3-32B 在内的强大模型]，这**证明了 AutoTriton 在其监督微调阶段和强化学习阶段的有效性**，也凸显了我们提出的**数据收集管道和训练框架的显著贡献**。
- AutoTriton 的优势在:mdnice-bold-italic[运行时性能评估中也很明显]。在大多数评估渠道上，它取得了与主流大型模型（包括 Claude-4-Sonnet 和 DeepSeek-R1-0528）相当的性能，这强调了我们提出的数据收集管道在生成高保真训练样本方面的有效性。

此外，我们**观察到在 TritonBench-G 渠道上，所有被评估的模型在正确性和性能指标上都表现不佳**。该渠道的固有难度（它:mdnice-bold-italic[基于 GitHub 上的真实世界需求]，将模型生成的代码与参考 Triton 实现进行评估）:mdnice-bold-italic[凸显了自动 Triton 编程中仍然存在的重大挑战]。这**表明该任务远未解决，值得深入研究。**

### 4.3 分析

#### Triton 和 CUDA 模型的交叉比较

为了**进一步评估 AutoTriton 的性能**，我们将其与并非专门针对 Triton 编程的主流内核生成模型进行了比较分析，即 AI CUDA Engineer（Lange 等人，2025）和 Kevin-32B（Baronio 等人，2025）。我们选择 KernelBench 基准进行此评估，因为它能够同时**评估 Triton 和 CUDA 内核**，确保公平比较。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/10.webp)

如表 3 所示，我们报告了在 pass@10 设置下相对于 PyTorch 基线的 P75 和 P50 加速比，它们分别代表内核性能分布的第 75 百分位和第 50 百分位的加速比。

我们的**分析得出的一个关键发现是，Triton 和 CUDA 在自动编程能力上存在持续的、系统性的差距**，这凸显了:mdnice-bold-italic[高性能 Triton 代码生成所面临的巨大挑战]。尽管如此，**AutoTriton 仍优于最近的特化框架 AI Cuda Engineer（Lange 等人，2025），在正确性和运行时效率指标上都取得了更好的定量结果**。虽然这些结果验证了 AutoTriton 的优势，但它仍然:mdnice-bold-italic[落后于 Kevin-32B] （Baronio 等人，2025）模型，我们将其**归因于三个潜在原因**：:mdnice-bold-italic[CUDA 和 Triton 之间固有的编程模型差异、参数规模不匹配] （80 亿对 320 亿），以及 :mdnice-bold-italic[Kevin-32B 使用了 90%的评估数据进行训练]，这可能带来更高的评估分数。

#### 强化学习的影响

> 表 1 和表 2 的最后几行展示了**没有强化学习** 阶段的 AutoTriton 的性能。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/08.webp)

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/09.webp)

将 AutoTriton 与其仅经过监督微调的版本进行比较，可以观察到明显的性能提升，这表明**强化学习能有效提高 Triton 编程任务的性能上限**。这一结果表明，:mdnice-bold-italic[强化学习使模型能够超越模仿学习的固有局限性]，这与许多其他领域的研究结果一致。此外，强化学习阶段取得的性能提升也验证了我们提出的数据收集管道的有效性。该管道有效地生成了非常适合强化学习探索的训练数据集，这在突破 Triton 编程任务的边界方面发挥了关键作用。

#### 奖励设计的影响

如 3.3 节所述，Triton 编程任务中的一个**主要挑战是“奖励欺骗”（指模型学会满足测试用例但不生成正确的 Triton 代码）**。:mdnice-bold-italic[为了解决这个问题，我们引入了辅助的基于规则的奖励]，与主要的基于执行的奖励相结合，以明确激励模型遵守 Triton 语言规范。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/11.webp)

表 4 量化了这一策略的影响。通过检查强制性的 `@triton.jit` 装饰器，我们发现**基于规则的奖励显著减少了** TritonBench-T（从 18 个减少到 5 个）和 KernelBench-Level1（从 25 个减少到 6 个）上的**无效生成，证实了奖励机制中明确语法指导的重要性**。:mdnice-bold-italic[尽管有这些改进，基于规则的奖励函数仍然可能被“欺骗”。模型可能学会生成满足明确规则但未能满足 Triton 完整语义要求的低质量代码。]

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/12.webp)

例如，如图 3 所示，当任务是实现一个由卷积和 ReLU 组成的内核时（图 3(a)），模型通常会为较简单的 ReLU 部分生成有效的 Triton 内核，而将更复杂的卷积部分保留为 PyTorch 的 fallback（后备）实现（图 3(b)）。**更关键的是，模型可能通过生成一个从未被调用的假 Triton 内核来完全规避奖励规则（图 3(c)）**。这种低质量实现的现象在所有被评估的模型中都非常普遍。:mdnice-bold-italic[一个潜在的解决措施是纳入基于运行时的性能奖励，我们将其留作未来的工作。]

#### 监督微调的影响

如表 1 和表 2 所示，**经过监督微调后，模型的性能优于原始的基础模型** （Seed-Coder-8B-Reasoning）。这一初步结果表明，监督微调能有效帮助模型熟悉 Triton 编程的基本范式，并进一步证明了我们提出的数据收集管道在生成高质量监督微调数据方面的有效性。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/13.webp)

图 4 中的训练动态进一步证实了这一结论。尽管没有经过监督微调的模型的奖励曲线也呈现出明显的上升趋势，但它遭受了严重的“奖励欺骗”，大多数样本都表现出图 3(c)所示的行为。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/12.webp)

具体来说，虽然其生成的代码可以通过测试用例，但它们往往不遵守 Triton 的基本语法，而是默认使用更简单的 Torch 实现，这与表 4 中观察到的现象一致。

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/11.webp)

这种行为表明，**监督微调** 不仅**对于学习正确的语法至关重要**，而且对于:mdnice-bold-italic[防止“奖励欺骗”] （即模型学会用简单的 Torch 代码利用测试用例，而不是掌握真正的 Triton 编程）也至关重要。

#### 局限性

基于对 AutoTriton 的多方面评估，其主要**局限性是我们当前的训练框架缺乏基于性能的训练**。由于:mdnice-bold-italic[编译或蒸馏得到的内核缺乏有效的运行时反馈，AutoTriton 在当前阶段不包含基于性能的训练]。未来的工作可以**专注于获取或生成更高质量的数据**，同时**整合一个感知性能的训练框架**，奖励模型生成不仅功能正确而且在目标硬件上实现高性能的内核，从而引导模型朝着更高效的解决方案发展。

## 五、结论

> 在这项工作中，我们研究了自动 Triton 编程任务，并提出了 AutoTriton，这是第一个专门用于 Triton 编程的、由强化学习驱动的模型。

AutoTriton 采用两阶段训练过程：

::: block-1
- 在监督微调阶段，AutoTriton 从我们新颖的数据整理管道生成的高质量数据中学习基本的 Triton 编程知识；
- 随后在强化学习阶段，它通过探索更具挑战性的问题实例进一步改进。

在 TritonBench 和 KernelBench 的五个渠道上的评估表明，AutoTriton 取得了与主流大型模型（包括 Claude-4-Sonnet 和 DeepSeek-R1-0528）相当的性能。
:::

我们对每个组件的深入分析验证了基于强化学习的方法在自动 Triton 编程方面的巨大潜力。最终，AutoTriton 展示了一条有前景的自动化生成高效内核的路径，为构建高性能 AI 系统提供了新的范式。

## 参考文献

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/14.webp)

![文章插图](/articles/2025-08-09-autotriton-triton-8b-claude-4-6d7d3615/images/15.webp)

交个朋友
