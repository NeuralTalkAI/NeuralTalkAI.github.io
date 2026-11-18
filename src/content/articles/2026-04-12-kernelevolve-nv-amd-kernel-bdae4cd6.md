关键词：**KernelEvolve**、:mdnice-bold-italic[硬件分层知识库]、**异构 AI 硬件**、写算子的那种 SKILL 动态更新、:mdnice-bold-italic[AI 基础设施]

> AI 模型复杂度持续攀升、硬件生态日趋多元，编译器与底层内核作为连接模型与异构硬件的核心枢纽，已然成为制约 AI 基础设施效率、决定规模化部署可行性的核心引擎。

模型架构、内核原语、硬件代际与架构的三维多样性构成内核优化的维度灾难，人工调优内核难以适配 NVIDIA、AMD GPU 及 Meta 自研 MTIA 等异构硬件，内存层级、编程模型的代际断层更让手动跨平台开发在工程与经济上均不可行；而预处理算子的内核覆盖度更是决定部署架构的二进制约束，缺失单一内核便会引发 10-20ms 纯网络开销的架构惩罚。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/01.webp)

- **KernelEvolve: How Meta’s Ranking Engineer Agent Optimizes AI Infrastructure**
- https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/
- 7000 字，阅读 25 分钟，**播客 29 分钟**

相关推荐

- [算子 编译器天塌了吗。。NVIDIA 提出 AVO：当 AI Agent 成为进化算子，探索突破 Blackwell 性能极限，超越 FlashAttention‑4 最高10.5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899076&idx=1&sn=64a6532410f9d22f65043945582d4f1a&scene=21#wechat_redirect)
- [Meta AI 基础设施的十年演进、下一代基础设施蓝图与 AI 自研芯片之路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896790&idx=1&sn=a13b82d8c267fcdbd6f3e20c53a1429b&scene=21#wechat_redirect)
- [Meta 超大规模基础设施概述：10万块GPU与服务器规模下的97% 自动化部署、每 3 小时一次更新！支撑 10 亿用户请求！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890290&idx=1&sn=4080e61b4591b57896c593c551703c7d&scene=21#wechat_redirect)

> 为此，Meta 打造的 KernelEvolve 彻底改写内核开发范式，让其成为排名工程师智能体的核心组件。它将内核优化形式化为结构化搜索问题，融合 LLM 合成器、树搜索引擎、检索增强知识库等六大核心组件，实现跨硬件内核的自动化生成、评测与迭代 。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/02.webp)

::: image-caption
图 1：内核优化请求在 KernelEvolve 六大组件中的流转流程。系统以UI/API为入口，通过外循环树搜索（MCTS+进化策略）筛选优质内核（绿）、淘汰缺陷内核（红）；内循环依托多硬件/多语言知识库，经LLM生成候选、CPU池调度、AI硬件池（MTIA/NVIDIA/AMD/CPU）多维度评测，生成性能报告。数据底座沉淀会话与树节点，通过后训练（SFT/DPO等）迭代Meta大模型，形成“搜索-生成-评测-迭代”的闭环，完美支撑异构硬件自动化内核优化，数小时替代专家数周的效率突破。
:::

该系统:mdnice-bold-italic[仅需数小时即可完成内核专家数周的调试优化工作]，在 NVIDIA GPU 上为广告模型带来超 **60%推理吞吐提升**，在 **MTIA 芯片上实现超 25%训练吞吐增益**，还在标准基准测试中达成 100%通过率。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/03.webp)

::: image-caption
KernelEvolve生成的内核与PyTorch的一维卷积及优化后的二维卷积基准模型进行对比，结果显示在英伟达、超威半导体以及MTIA架构上，其加速比最高可达6.22倍。
:::

::: block-1
KernelEvolve 不仅破解了异构硬件适配难题，还将工程师从底层编码中解放，助力 Meta 每日万亿级推理请求高效运行，为大规模 AI 基础设施的快速迭代开辟全新路径。
:::

## 本文目录

- 一、KernelEvolve：Meta 排名工程师智能体如何优化 AI 基础设施
- 二、核心挑战：内核数量爆发式增长带来的瓶颈
  - 2.1 硬件异构性
  - 2.2 模型架构多样性
  - 2.3 标准库之外的内核多样性
- 三、KernelEvolve 的破局方案
- 四、KernelEvolve：最优内核的搜索之路
  - 4.1 大语言模型合成器（LLM Synthesizer）
  - 4.2 树搜索引擎
  - 4.3 检索增强知识库
  - 4.4 自动化评测框架
  - 4.5 共享数据底座
  - 4.6 智能体强化学习
- 五、赋能自研 AI 芯片
- 六、KernelEvolve 在基准测试与生产环境的效果
  - 6.1 开发效率
- 七、整体工作流程
- 八、未来展望

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：交个朋友

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/05.webp)

## 一、KernelEvolve：Meta 排名工程师智能体如何优化 AI 基础设施

> 本文将阐述如何优化底层基础设施，让这类模型在大规模场景下高效运行。我们推出 KernelEvolve，这是一款供排名工程师智能体使用的智能体内核开发系统，且普遍适用于广告排序之外的各类 AI 模型。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/06.webp)

::: image-caption
Triton多目标编译架构。源代码通过渐进式MLIR降级阶段进行转换——与平台无关的Triton-MLIR、硬件特定的GPU/AMDGPU/MTIA方言、LLVM中间表示（LLVM-IR），最终为英伟达（PTX/CUBIN）、超威（AMDGCN/HSACO）平台生成原生二进制文件。
:::

- **Meta 运营着大规模异构硬件集群，涵盖 NVIDIA GPU、AMD GPU、Meta 自研 MTIA 芯片以及 CPU**。想要高效利用这些硬件，需开发能将高层模型运算转化为芯片专属指令的高性能程序，这类程序被称为优化内核。针对每一代新芯片与机器学习模型架构，都要完成内核的开发与优化。:mdnice-bold-italic[除厂商库覆盖的通用矩阵乘法、卷积等标准内核算子外，生产负载还需要排序模型中的大量定制算子。随着模型数量、硬件类型与代际不断增加，内核专家的手工调优方式已无法满足规模化需求。]
- 为应对模型数量、硬件类型与代际叠加带来的**海量性能优化工作**，我们打造了KernelEvolve[1]，这是一款供Meta Ranking 工程师智能体[2]使用的性能优化智能体，实现三大核心价值：
- KernelEvolve 将内核优化视为搜索问题：专用任务执行框架评测每个候选内核，将诊断信息反馈给大语言模型，持续搜索数百种备选方案，性能超越人类专家编写的内核。
- 更多技术细节详见论文 KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta[5]，该论文将发表于第 53 届国际计算机架构研讨会（ISCA）2026[6]。

Meta 每日为全球用户提供数十亿次 AI 驱动服务，从个性化推荐到生成式 AI 助手，背后依托的是包含 NVIDIA、AMD 及 Meta 自研 MTIA 芯片的多元化硬件基础设施。

> 每一次训练或推理请求的背后，都有一层高度优化的底层硬件内核——这类小程序能将高层模型运算转化为特定芯片可高效执行的指令。**随着 AI 模型愈发复杂、硬件生态日趋多元，内核数量随硬件平台、模型架构、算子类型同步增长，形成数千种配置组合，已无法依靠人类专家逐一调优**，这成为关键瓶颈，拖慢硬件适配、性能调优与模型迭代节奏，阻碍机器学习技术与应用的核心突破。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/07.webp)

::: image-caption
针对生产环境数据形状的一维卷积实现方案性能分析轨迹对比。PyTorch一维卷积（上图）会启动五个独立内核，涵盖布局转换与通用矩阵乘法操作。PyTorch二维卷积（中图）通过优化的通道优先存储格式路径将内核数量缩减至四个。KernelEvolve（下图）通过跨操作融合技术，将所有运算整合为两个内核。请注意：性能分析轨迹中显示的耗时包含分析工具自身开销，无法代表内核的实际执行延迟。
:::

> 本文我们将分享KernelEvolve[7]，这款自主化 AI 系统仅通过数小时实验，就将广告模型推理吞吐提升 60%，而这项工作人类专家需耗时数周。

KernelEvolve 可自主生成并优化训练与推理所用异构硬件的生产级内核，支持 NVIDIA GPU、AMD GPU、Meta 自研 MTIA 芯片、CPU。**与常规一次性生成代码的大语言模型智能体不同，KernelEvolve 将内核优化视为搜索问题，探索数百种内核实现方案，快速找到性能比肩甚至超越人类专家的方案**。在 Meta 生产环境中，KernelEvolve 正优化支撑每日万亿级推理请求的代码。

KernelEvolve 标志着我们对 AI 软件与硬件关系认知的根本性转变。内核开发曾是依赖专家的手工流程，难以跟上硬件与模型迭代速度，而 KernelEvolve 让这一过程实现持续化、自动化，随硬件与模型变化动态适配。随着 Meta 持续丰富 AI 硬件布局，快速为新芯片生成优化内核的能力，大幅降低了异构硬件集成到训练与推理流程的工程成本。

## 二、核心挑战：内核数量爆发式增长带来的瓶颈

> **内核数量呈爆发式增长，源于总内核数由三大因素乘积决定：硬件类型与代际 × 模型架构 × 算子数量**。这一乘积产生数千种独特内核配置，均需编写、测试与维护。:mdnice-bold-italic[手工调优每个内核无法规模化，仅靠内核专家也跟不上迭代节奏。]

### 2.1 硬件异构性

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/08.webp)

::: image-caption
在Meta，Triton超越CUDA成为主流内核编程模型。左图：Triton的内核数量已增至8000余个，超过了停滞不前的CUDA遗留代码库，而新兴领域特定语言（CuTe、TLX、Helion）的内核数量仍不足600个。右图：增长趋势显示，Triton 60%的扩张速率推动了这一转变，CuTe在2025年11月部署后增速达到50%。这种向更高层级领域特定语言的转型——同时保留传统CUDA并引入新的抽象层（TLX）——导致编程模型在五种以上语言间呈现碎片化，这也成为研发KernelEvolve自动化合成方案的动因。
:::

> Meta 的加速芯片集群涵盖 NVIDIA GPU、AMD GPU、Meta 自研MTIA[8]芯片，各类硬件的内存架构与层级、指令集、执行模式存在本质差异。

在一个平台上最优运行的内核，在另一平台上可能性能极差甚至完全失效。复杂性不止存在于厂商之间，即便同一硬件系列，迭代代际的架构变化也需要不同优化策略。**Meta MTIA 路线图两年内迭代四代芯片** （MTIA 300 至 500[9]），:mdnice-bold-italic[每一代都新增计算能力、内存带宽特性、数值数据类型，适配持续演进的负载。为某一代优化的内核，在同架构下一代芯片上运行会出现性能下降。]

### 2.2 模型架构多样性

Meta 的推荐模型历经三大发展阶段：

从早期基于嵌入的深度学习推荐模型，到采用注意力机制处理交互历史的序列学习模型[10]，再到生成式广告推荐模型（GEM）[11]，以及最新落地大语言模型规模广告能力的Meta 自适应排序模型[12]。

::: block-1
每一代都引入前代未有的算子类型。

除代际迭代外，**Meta 生产栈同时支撑多种差异显著的模型家族，各有专属算子**，:mdnice-bold-italic[单次广告请求可能在一次服务调用中遍历多个模型家族]。生产环境中模型种类繁多且持续增长，每一种新架构都会拓展跨硬件需优化的算子矩阵。
:::

### 2.3 标准库之外的内核多样性

> cuBLAS、cuDNN 等厂商库覆盖通用矩阵乘法、卷积、标准激活函数等常规运算，但即便这类标准算子也不存在通用解决方案。

以矩阵乘法为例，同一算子在不同场景表现不同：:mdnice-bold-italic[训练批次的最优内核与推理服务请求不同，张量形状在排序阶段与模型间差异极大，形成组合式配置空间]，**人类专家与当前基于编译器的自动调优、算子融合技术都无法大规模全覆盖**。

除标准算子外，**生产负载以大量库覆盖范围外的长尾算子为主**，包括特征哈希、分桶、序列截断等数据预处理转换（为模型推理准备原始输入），以及融合特征交互层、专用注意力变体等 Meta 架构专属定制模型算子。

**这些定制算子均未出现在厂商库中，且多数因负载特异性强，无需纳入库实现**。若无加速器原生实现，这类算子要么降级到 CPU 运行（导致分离式服务架构产生显著延迟开销），要么通过未优化代码路径运行，无法充分利用硬件性能。

**硬件多样性让问题进一步加剧**。手工优化的 NVIDIA 内核无法直接编译适配 AMD GPU 或 MTIA 芯片。每一种新模型架构都会拉长长尾算子，每一款新芯片都会让适配工作量成倍增加。

## 三、KernelEvolve 的破局方案

> 各项挑战均对应专属架构设计，具体如下：

| 挑战 | KernelEvolve 的解决方案 |
| --- | --- |
| 硬件异构性 | 检索增强知识库将平台**专属文档** （架构手册、指令集、优化范式）注入生成上下文。大语言模型在推理阶段基于该文档推理，无需提前针对目标硬件训练。统一通用提示接口，无需为每个平台定制提示模板。 |
| 模型架构多样性 | 树搜索探索任意算子（含全新算子）的实现方案。**成功优化方案提炼为可复用范式，跨模型家族迁移** ——为某一架构发现的优化方法，可加速后续同类算子优化。 |
| 内核多样性/长尾算子 | **自动化评测并行验证数百个候选方案**。基于**搜索的优化替代手工调优**，让原本不值得投入数周手工优化的算子具备落地可行性。 |

## 四、KernelEvolve：最优内核的搜索之路

> KernelEvolve 与常规 AI 编码助手的思路截然不同。它并非提示大语言模型生成单个内核再测试，而是**将内核优化形式化为跨实现空间的结构化搜索问题。**

底层由专用长时任务执行框架驱动每一轮迭代——编译候选方案、校验正确性与性能、分析硬件利用率、生成评测报告，同时处理数分钟构建周期与基础设施故障，让原生方案难以落地的场景变为可行。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/09.webp)

::: image-caption
图 1：内核优化请求在 KernelEvolve 六大组件中的流转流程。这张图完整呈现了Meta KernelEvolve的端到端架构，体现“将内核优化转化为结构化搜索问题”的核心设计思路。系统以UI/API为入口，通过外循环树搜索（MCTS+进化策略）筛选优质内核（绿）、淘汰缺陷内核（红）；内循环依托多硬件/多语言知识库，经LLM生成候选、CPU池调度、AI硬件池（MTIA/NVIDIA/AMD/CPU）多维度评测，生成性能报告。数据底座沉淀会话与树节点，通过后训练（SFT/DPO等）迭代Meta大模型，形成“搜索-生成-评测-迭代”的闭环，完美支撑异构硬件自动化内核优化，数小时替代专家数周的效率突破。
:::

### 4.1 大语言模型合成器（LLM Synthesizer）

> 大语言模型生成多编程语言、多硬件目标的候选内核，涵盖 Triton、TLX[13]、CuTe DSL、FlyDSL 等高层领域特定语言，以及 CUDA、HIP、MTIA C++等底层后端。

合成器不使用静态提示，而是构建**动态上下文感知提示**，持续融入运行时诊断信息、硬件约束、历史候选优化评测信号。该设计替代了传统为调试、性能调优、正确性校验分别维护提示模板的方式，以单一自适应接口整合所有工作流，驱动持续的反馈式优化循环。

### 4.2 树搜索引擎

> 系统采用蒙特卡洛树搜索、进化策略等图搜索算法探索优化空间。每个内核候选方案作为搜索树的一个节点，引擎筛选优质候选、执行转换、评测结果，决定继续探索或回溯——平衡利用成熟方案与探索全新思路。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/10.webp)

::: image-caption
针对一维卷积核生成历经300步的搜索树可视化。绿色：生成成功；红色：编译/正确性校验失败。
:::

::: block-1
关键在于，节点并非独立进化。每个节点搭载可配置记忆算子，决定生成下一轮候选时如何从搜索树提取上下文。节点可继承父节点优化轨迹以深化优质方向，对比兄弟节点以区分高性能变体，融合父节点与兄弟节点的思路，或重置状态以跳出局部最优。
:::

**这种选择性记忆机制让树搜索超越简单独立采样**：:mdnice-bold-italic[兄弟节点通过互补策略协同，父子链保留并深化成功优化路径，无记忆重启则在搜索停滞时注入多样性。]

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/11.webp)

::: image-caption
图 2：树搜索引擎在优化空间中探索高性能内核的逻辑。图中总计 51 个搜索节点，最优内核（Step 26）以蓝色选中节点呈现，得分 1.9729，位列 TOP5 榜首。节点以颜色区分状态：绿色为改进型有效内核，红色为缺陷内核，紫色为父节点，橙色为子节点，直观展示从根节点出发的迭代探索路径。系统通过结构化搜索迭代筛选、淘汰缺陷方案，最终找到性能远超基线的最优内核，完美验证自动化搜索替代人工调优、数小时完成专家数周工作的核心价值。
:::

### 4.3 检索增强知识库

> 为给大语言模型**未训练过的硬件** 生成优化代码，KernelEvolve 维护分层知识库，分为三类：保障内核实现合法的正确性约束、覆盖调试与调优策略的平台无关优化指南、包含各加速芯片架构细节的硬件专属文档。

系统基于运行时信号动态检索相关知识，例如内存带宽瓶颈触发内存层级文档检索，编译错误激活调试指南。

**该知识库并非静态**。系统解决新优化问题时，会将成功方案提炼为可复用技能（紧凑优化范式与调试启发式规则）并回写知识库。:mdnice-bold-italic[这种自进化技能库实现上下文内强化学习]：每一次成功探索都丰富后续会话的上下文，让系统以更少搜索步骤更快解决同类问题，无需重新训练模型。

### 4.4 自动化评测框架

> 所有生成内核都经过严格校验流程，同时核查功能正确性（与参考实现逐位精度一致）与性能表现，且评测远不止单一运行时长数据。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/12.webp)

::: image-caption
端到端评估流程。树搜索生成具备标准化双重实现（PyTorch基准版、Triton优化版）的算子候选，在硬件解释器（GPU、AMD、MTIA）上运行，并通过TritonBench、NCU、MPP以及MTIA Insight采集各平台专属的性能分析指标。性能分析反馈将指导后续的搜索迭代。
:::

KernelEvolve 整合多层性能分析工具，各有侧重：

- TritonBench[14]对比 PyTorch 基线校验数值正确性，测算生产输入形状的端到端加速比；
- PyTorch Profiler 捕获系统级执行时间线，含内核启动开销、主机-设备同步；
- 针对 GPU 目标，NCU[15]提供内核级硬件指标（占用率、内存吞吐、指令组合），Proton[16]**提供内核内指令级延迟与流水线表** 现；
- 针对 MTIA 目标，MTIA Insight 提供芯片专属全维度检测（处理单元利用率、固定功能引擎指标、缓存行为、处理单元内存带宽计数器）。

KernelEvolve 未将这些工具视为独立步骤，而是**通过【编】【译】【器】中心抽象统一整合**。框架通过任务图完成分析：:mdnice-bold-italic[编译器转换插入多层级检测工具，性能分析模块采集指标，轨迹合成生成结构化输出]。这让搜索引擎不仅能判断“内核 A 比内核 B 快 1.2 倍”，还能明确原因——瓶颈是内存受限、计算受限还是占用率不足，并将诊断信号反馈给大语言模型合成器，指导下一轮候选生成。

### 4.5 共享数据底座

每一次优化会话都为共享数据底座贡献价值。**若某工程师的探索发现某类算子的有效分块策略，该思路会对后续同类负载会话开放，形成复利效应** ——:mdnice-bold-italic[系统随使用次数增加愈发智能]。早期用户完成最难的探索工作，后续用户直接接近最优起点并进一步优化。

### 4.6 智能体强化学习

每一次优化会话都会自然生成结构化训练数据：记录高性能内核背后推理逻辑、代码转换、评测反馈的智能体轨迹。**这类领域专属数据稀缺且价值极高**，编码了公共数据集不具备的优化直觉。

**我们利用这些数据通过智能体强化学习微调小型专用模型，奖励信号直接来自内核实测性能**。由此形成良性循环：:mdnice-bold-italic[更优模型以更少推理 token、更少搜索步骤生成更优内核，进而产出更高质量训练数据]。多轮迭代后，这种复利飞轮让我们自主部署更高效的模型，模型足够紧凑可规模化低成本运行，同时保留超大前沿模型的优化能力。

## 五、赋能自研 AI 芯片

> 该架构的核心价值之一，是能为公共训练数据中不存在的硬件生成优化代码。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/13.webp)

::: image-caption
Meta MTIA 是一款专为人工智能工作负载优化定制设计的芯片。该图从四个维度展示了MTIA硬件：其在数据中心或服务器集群中的集成情况，着重呈现整体设施布局与环境；在面向高带宽应用的机架式系统中的部署形态；芯片电路与板载连接的特写视图；以及芯片核心的细节结构。这些图像共同展现了MTIA的先进设计、互联特性，及其在提升元旗下各平台人工智能任务性能与效率方面所发挥的作用。
:::

**Meta 自研 MTIA 芯片带来独特编程挑战**。由于该芯片为自研，公共大语言模型均未基于 MTIA 代码训练，常规编码助手因未接触 MTIA 文档、指令集细节、编程范式，无法编写优化的 MTIA 内核。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/14.webp)

::: image-caption
MTIA 2i架构采用8×8处理单元（PE）阵列，各单元通过片上网络互联。每个处理单元包含双RISC-V内核及专用固定功能单元：用于数据转换的内存布局单元（MLU）、用于矩阵运算的点积引擎（DPE）、用于聚合运算的归约引擎（RE）、用于向量运算的单指令多数据流引擎（SE），以及用于控制流处理的命令处理器（CP）
:::

**KernelEvolve 通过系统化知识注入解决该问题**。:mdnice-bold-italic[我们将 MTIA 专属文档（架构手册、指令集参考、内存层级规范、优化范式）直接编码至检索增强知识库]。系统针对 MTIA 优化时，会检索并融入该自研知识，实现实时“学习”硬件特性。

**该方案可拓展至任意新加速芯片**。:mdnice-bold-italic[新芯片面世后，工程成本从手工编写数千个内核，转为整理硬件文档并注入知识库]。系统随后自主为新平台生成优化内核，让软件栈适配速度匹配硬件部署速度，而非受制于人工工程效率。

## 六、KernelEvolve 在基准测试与生产环境的效果

> KernelEvolve 在标准基准测试与生产负载中均取得亮眼表现。

**基准测试性能**：在斯坦福涵盖三个难度等级、包含 250 个内核优化问题的 KernelBench 测试集上，KernelEvolve 实现**100%通过率** ——:mdnice-bold-italic[所有生成内核功能正确，且速度优于 PyTorch 参考实现。系统还在三大硬件平台验证 160 个 PyTorch ATen 算子，480 种配置组合均实现 100%正确性。]

**生产环境加速**：

- 在 Meta MTIA 芯片上，KernelEvolve 生成的计算密集型、内存密集型、定制操作内核，为一款广告模型实现超 25%训练吞吐提升；
- 在 NVIDIA GPU 上，相比已集成 torch.compile 与厂商库的高度优化内核，实现超 60%推理吞吐提升——该性能增益直接转化为服务容量与基础设施效率。

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/15.webp)

::: image-caption
KernelEvolve在各类异构人工智能硬件上，针对 Meta 平台的大语言模型及实际生产应用场景实现了1.25至17倍的性能加速，应用场景涵盖卷积Transformer架构、数据预处理算子以及推荐系统。
:::

**硬件覆盖**：系统通过单一统一框架，为 NVIDIA GPU、AMD GPU、Meta 自研 MTIA 芯片、CPU 生成优化内核。它无需为每个平台维护独立提示模板，而是动态检索硬件专属约束与优化范式，通过检索增强适配各目标，无需手工提示工程。

### 6.1 开发效率

此前需内核专家耗时数周的内核开发工作（性能分析、迭代分块策略、跨硬件调试边界场景），如今通过自动化搜索与评测数小时即可完成。这让工程师从底层编码工作中解放，投入模型架构设计、训练方法优化、定义优化目标等高价值工作。

## 七、整体工作流程

> 工程师指定目标算子、硬件平台、性能目标后，系统自主完成以下步骤：

1. 从知识库**检索** 相关硬件文档与优化知识。
2. 由大语言模型合成器通过上下文感知提示**生成** 初始内核候选集。
3. 通过分布式基准测试基础设施**评测** 每个候选的正确性与性能。
4. 将结果**反馈** 至搜索引擎，筛选优质候选并执行进一步优化。
5. :mdnice-bold-italic[迭代 1-4 步，探索搜索树直至满足终止条件] ——达成性能目标、耗尽搜索预算或进度停滞。
6. **输出** 性能最优、完整校验的内核，可直接部署至生产环境。

该流程运行在 Meta 分布式基础设施上，并行评测数千个候选方案。搜索树与实现方案的持久化存储，让系统在适配新模型变体或硬件代际时，可基于过往成果持续优化。

## 八、未来展望

> 支撑 KernelEvolve 的智能体技术（结构化推理、检索增强知识、闭环评测），可应用于混合模型搜索、编译器优化、内存管理、系统配置等场景。KernelEvolve 是迈向**排名工程师智能体** 愿景的早期一步，该智能体可持续优化自身的性能关键基础设施。

在排名工程师智能体体系中，机器学习探索模块[17]发现更优模型，KernelEvolve 让模型具备生产落地能力，二者协同加速排序优化方案落地广告业务的速度。

如需更多技术细节，可阅读发表于ISCA 2026[18]的论文 KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta[19]。

参考资料

[1] KernelEvolve: *https://arxiv.org/abs/2512.23236*

[2] Meta’s Ranking Engineer Agent: *https://engineering.fb.com/2026/03/17/developer-tools/ranking-engineer-agent-rea-autonomous-ai-system-accelerating-meta-ads-ranking-innovation/*

[3] Andromeda: *https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/*

[4] MTIA: *https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/*

[5] KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta: *https://arxiv.org/abs/2512.23236*

[6] 53rd International Symposium on Computer Architecture (ISCA) 2026: *https://iscaconf.org/isca2026/*

[7] KernelEvolve: *https://arxiv.org/abs/2512.23236*

[8] MTIA: *https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/*

[9] MTIA 300 through 500: *https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/*

[10] sequence learning models: *https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/*

[11] Generative Ads Recommendation Model (GEM): *https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/*

[12] Generative Ads Recommendation Model (GEM): *https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/*

[13] TLX: *https://github.com/facebookexperimental/triton*

[14] TritonBench: *https://github.com/meta-pytorch/tritonbench*

[15] Nsight Compute: *https://developer.nvidia.com/nsight-compute*

[16] Proton: *https://ieeexplore.ieee.org/document/11395207*

[17] ML Exploration: *https://engineering.fb.com/2026/03/17/developer-tools/ranking-engineer-agent-rea-autonomous-ai-system-accelerating-meta-ads-ranking-innovation/*

[18] ISCA 2026: *https://arxiv.org/abs/2512.23236?ref=engineeringatmeta*

[19] KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta: *https://arxiv.org/abs/2512.23236?ref=engineeringatmeta*

更多推荐

- [Meta 提出 TritorX：面向 ML ASIC 的 Agentic 算子生成系统，84.7% OpInfo 覆盖与 2 万+测试验证，简化加速器软件生态构建](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895656&idx=1&sn=8a348cee4243a478fe9218a0f317c9ab&scene=21#wechat_redirect)
- [Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)
- [国产GPU生死线：10亿营收验尸报告，谁在裸泳？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899576&idx=1&sn=28cfd027d5f571ef6e024493b8852d14&scene=21#wechat_redirect)

交流加群请在 NeuralTalk 公众号后台回复：交个朋友

![文章插图](/articles/2026-04-12-kernelevolve-nv-amd-kernel-bdae4cd6/images/05.webp)
