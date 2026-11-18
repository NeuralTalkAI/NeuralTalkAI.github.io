关键词：**JIT-Agent**、:mdnice-bold-italic[Agent Harness]、**Harness Intelligence**、Evo-GDPO、**任务自适应编排**

> 所有人都在卷基础模型的参数、推理速度和能力边界，却很少有人关注包裹在模型外层的运行脚手架——**Agent Harness，而它对 Agent 最终性能的贡献甚至可以超过基础模型本身。**

长期以来，行业默认 Agent 能力是基础模型的附属品，更强的模型自然能完成更复杂的任务。:mdnice-bold-italic[但实际工程实践中，同样的模型搭配不同的记忆策略、规划流程和工具调用方式，最终表现可能天差地别。一套糟糕的 Harness 可以让顶级模型频频出错，而一套适配良好的 Harness 可以让中等模型发挥出超预期的水平。]

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/01.webp)

::: image-caption
图 1：四个代表性智能体基准测试排行榜。JIT-Agent 生成的执行框架在深度研究、日常工作、规划和工作空间任务中均持续提升了底层主干模型的性能，表明即时执行框架综合所带来的收益超越了单纯的模型缩放。从结果分布来看，无论是侧重推理效率的轻量化模型还是能力更强的大参数模型，搭载 JIT 生成的定制化 harness 后性能都出现显著抬升，且在规划类复杂任务上提升幅度最为突出。
:::

::: block-1
传统的 Agent Harness 始终停留在人工设计、提前优化的阶段。无论是经典的 ReAct、Plan-and-Execute 范式，还是 Claude Code、OpenCode 等生产级 Agent 运行时，都属于“提前构建、全局通用”的 AOT（Ahead-of-Time）模式：开发者预先设计好固定的架构和流程，然后用这套统一的 Harness 去适配所有类型的任务。这种模式在任务分布稳定的场景下表现良好，:mdnice-bold-italic[但面对异构的任务类型时，必然会出现适配性不足的问题——广度搜索任务需要并行探索能力，终端操作任务需要精简的串行循环，深度研究任务需要长时工作记忆，代码开发任务需要文件系统状态管理，用单一 Harness 覆盖所有场景，结果只能是处处妥协、处处平庸。]
:::

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/02.webp)

- **JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution**
- 论文：https://arxiv.org/pdf/2608.25593
- 网站：https://bingreeky.github.io/JIT-site
- 代码：https://github.com/bingreeky/JIT
- 模型：https://huggingface.co/JIT-Agent

::: related-title
**相关推荐**
:::

::: related-links
- **[默认最强模型，吃掉编码 Agent 大半预算！Databricks 同时路由 Model 和 Harness，成本最高节省 56%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904783&idx=1&sn=d71d64ab6fafe2a7b4217dadd19f3da4&scene=21#wechat_redirect)**
- **[DeepSeek-V4-Pro 0813 官方文档已上线，DeepSeek Harness 蓄势待发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904626&idx=1&sn=5960582b3f87e6e94cd7eef299ed3fd7&scene=21#wechat_redirect)**
- **[记忆是被重构的而非重放的！浙大提出 MemHarness，用 GRPO 炼出超越 GPT-4o 的 7B 智能体](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904205&idx=1&sn=d511f9fea5385a822ac1844841af6596&scene=21#wechat_redirect)**

:::

> 针对这一核心痛点，来自新加坡国立大学的研究团队提出了 JIT-Agent，这是**首个专门用于即时生成任务自适应 Harness 的元模型。它的核心洞察是：既然不同任务需要不同的 Harness，那就不要预先设计通用版本，而是在见到具体任务之后，当场合成专属的运行脚手架**。JIT-Agent :mdnice-bold-italic[将 Agent Harness 形式化为由记忆、规划、动作、能力编排四大模块组成的可组合构件，通过定制学习、修复学习、进化学习三阶段训练，掌握了 Harness 的生成、修复与自我进化能力。]

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/03.webp)

::: image-caption
图 2：JIT-Agent 系统概览。给定任务后，JIT-Agent 通过实例化（而非简单拼接）记忆、规划、行动和能力编排四个模块，构建面向特定问题的智能体执行框架。因此，不同的任务结构会催生截然不同的可执行协议与状态组织形式，正如深度研究、产品生成和自主研究三类场景的专用框架所展示的那样。
:::

实验结果显示，搭载 JIT 生成的 Harness 后，DeepSeek-V4-Flash 在 DeepSearchQA 基准上**超越 GPT-5.6 达 9.1 分**，在 OdysseyBench 上领先 4.3 分；原本表现强劲的 **GLM-5.2 更是获得最高 20.2 分的性能提升**。

::: block-1
在成本端，JIT 生成的 Harness 相**比成熟的固定 Harness 平均降低 36%的 API 调用成本，最高降幅达 54.1%，同时保持更优的性能表现**。这项工作:mdnice-highlight[不仅提出了一种提升 Agent 性能的新方案，更定义了“Harness 智能”这一全新的能力维度，证明它是可训练、可迁移、可复利的，与模型缩放完全正交。]
:::

## 本文目录

- 一、Agent 能力的第二曲线：被忽视的 Harness 层
  - 1.1 什么是 Agent Harness
  - 1.2 传统 Harness 工程的固有局限
- 二、从 AOT 到 JIT：Harness 设计的范式转移
  - 2.1 JIT 范式的核心思想
  - 2.2 两种范式的系统性对比
  - 2.3 Harness 智能的定义
- 三、四模块协议：构建可生成的标准化 Harness 设计空间
  - 3.1 四模块分解框架
  - 3.2 HarnessFactory 种子库
- 四、三阶段训练：从定制、修复到进化的能力闭环
  - 4.1 第一阶段：任务条件下的 Harness 定制
  - 4.2 第二阶段：失败 Harness 的修复学习
  - 4.3 第三阶段：在线进化的 Evo-GDPO
  - 4.4 两种推理模式
- 五、全面实证：性能、成本与泛化性的多重验证
  - 5.1 同模型下的性能提升
  - 5.2 与成熟固定 Harness 的对比
  - 5.3 跨模型家族的泛化性
  - 5.4 流式进化的实际效果
- 六、具象观察：不同任务下生成 Harness 的形态差异
  - 6.1 工件生产类任务：DAG 式图执行架构
  - 6.2 深度研究类任务：递归委托式架构
- 七、总结与展望：Harness 智能的长期价值
  - 7.1 核心贡献总结
  - 7.2 未来展望

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/04.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、Agent 能力的第二曲线：被忽视的 Harness 层

> 过去几年，大模型能力的提升主要沿着参数缩放的路径前进，从百亿到万亿参数，每一次代际升级都带来显著的能力跃迁。但当大模型走向 Agent 落地，人们逐渐发现，**模型能力只是基础，真正决定任务完成效果的，是模型之外的运行框架。**

### 1.1 什么是 Agent Harness

::: block-1
Agent Harness 是包裹在基础大模型之外的一层运行时脚手架，是将模型的推理能力转化为闭环执行能力的核心中间层。如果把基础模型比作 CPU，负责核心的逻辑运算，那么 **Harness 就是整个计算机系统的主板加操作系统，负责管理内存、调度任务、对接外设、控制执行流程。**
:::

具体来说，一套完整的 Agent Harness 涵盖四大核心功能：

| 组件 | 描述 |
| --- | --- |
| 记忆管理 | 决定哪些历史交互需要保留、如何组织、如何呈现给模型，是简单的全量上下文，还是分层摘要，或是结构化的事实库 |
| 规划策略 | 如何将复杂任务拆解为子目标，如何跟踪进度，如何应对意外情况调整计划 |
| 动作协议 | 定义模型输出的动作如何被执行，出错如何处理，结果如何返回给模型 |
| 工具与技能编排 | 从庞大的工具库中选择当前阶段需要的能力，如何组合调用多个工具，如何管理更复杂的 Agent 技能 |

**Agent 的实际能力是基础模型与 Harness 共同作用的结果，而非模型权重的单一属性**。:mdnice-bold-italic[强模型配上错误的记忆策略、混乱的规划流程，可能连简单任务都完成不好；中等模型配上高度适配的 Harness，却能超出预期地完成复杂任务]。这也是为什么很多时候，开源模型经过精心的 Harness 调优后，表现可以接近甚至超过闭源旗舰模型。

### 1.2 传统 Harness 工程的固有局限

> 当前主流的 Agent Harness 都属于人工设计的固定架构，也就是 AOT 模式。开发者根据经验设计好记忆、规划、动作的完整流程，然后将其封装成可复用的运行时，所有任务都在这套固定架构下执行。

这种模式存在三个不可避免的短板：

- 第一，**人力成本高，泛化性依赖开发者经验**。Harness 设计需要大量的 Agent 工程经验，:mdnice-bold-italic[优秀的 Harness 往往是团队反复试错打磨出来的，很难快速复制到新的任务领域。]
- 第二，**全局架构无法适配所有任务**。不同类型的任务对 Harness 的需求天差地别：信息检索任务适合并行探索加证据聚合，日常办公任务适合线性分步执行，多约束规划任务需要状态跟踪与回溯，代码项目任务需要文件系统与版本管理。:mdnice-bold-italic[用一套固定架构跑所有任务，必然会在很多场景下出现架构冗余或者能力缺失。]
- 第三，**优化周期长，迭代效率低**。AOT 模式下的 Harness 优化需要积累大量的任务执行轨迹，从中发现问题再调整架构，:mdnice-bold-italic[整个循环周期长，而且优化后的架构很难快速迁移到其他任务类型。]

::: block-1
随着 Agent 应用场景越来越广泛，任务类型越来越异构，传统 AOT Harness 的适配瓶颈也越来越明显。:mdnice-highlight[行业需要一种新的 Harness 生成范式，能够【针对每个任务】【自动生成】最适配的运行架构]，而不是用一套通用方案去勉强适配所有场景。
:::

## 二、从 AOT 到 JIT：Harness 设计的范式转移

> 提前优化的 :mdnice-bold-italic[AOT 范式本质上是用过去的经验适配未来的任务，其性能上限取决于训练分布与真实任务的匹配度。JIT 范式则彻底反转了这个逻辑：不为未来预先生成通用 Harness，而是在见到具体任务之后，当场合成专属的任务级 Harness。]

### 2.1 JIT 范式的核心思想

::: block-1
Just-in-Time 的概念最早来自编译领域：AOT 编译器在程序运行前就把代码全部编译成机器码，启动快但无法适配运行时的实际负载；:mdnice-highlight[JIT 编译器则在程序运行过程中，根据当前的执行情况动态编译热点代码，生成针对性的优化指令，最终获得更好的运行时性能。]
:::

Agent Harness 的 JIT 范式与之异曲同工：它不追求预先设计出一套能打所有场景的通用 Harness，而是:mdnice-bold-italic[把 Harness 生成本身变成一个模型可执行的任务。当一个新任务到来时，专门的元模型会根据任务的类型、目标、可用工具，当场生成一套最适合这个任务的 Harness，然后交给基础模型执行]。任务结束后，这套 Harness 可以被归档，作为未来生成类似任务的参考。

> 这种模式下，每个任务都能获得专属的运行架构，不再受通用 Harness 的架构限制。

- 任务需要并行探索，就生成多分支的 Harness；
- 任务需要严格的状态跟踪，就生成带结构化状态管理的 Harness；
- 任务不确定性高，就生成带动态规划与修复机制的 Harness。

### 2.2 两种范式的系统性对比

> 论文中明确对比了:mdnice-bold-italic[不同 Harness 优化方法的范式差异。表 1 对比了 8 种代表性方法在五个维度上的差异，包括构建方式、是否支持实例合成、是否有可训练的 Harness 模型、是否支持学习修复、是否支持在线进化。]

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/05.webp)

::: image-caption
表 1：Harness 框架优化范式对比。“构建方式” 区分了三类方法：通过提前搜索得到的框架、经测试时反馈后续编辑的提前构建框架、为每个任务即时生成的框架。“实例合成”“框架模型”“学习修复”“在线演化” 分别表示方法是否直接合成面向具体实例的框架、是否训练生成器、是否从失败执行轨迹中学习修复能力、是否在部署后持续改进。
:::

- AutoHarness、Meta-Harness、AHE 都属于纯 AOT 搜索模式，既不能生成实例专属 Harness，也没有可训练的模型，更不支持修复和进化；
- Adaptive AH、TTHE、RHI 属于 AOT 加测试时编辑模式，可以在部署后根据反馈修改 Harness，但依然没有统一的生成模型；
- Harness-R1 开始引入可训练的 Harness 模型和修复能力，但构建方式依然是 AOT 编辑；

::: block-1
只有 JIT-Agent 属于真正的 JIT 构建模式，同时具备实例合成、可训练 Harness 模型、学习修复、在线进化全部四项能力。:mdnice-highlight[这意味着 JIT-Agent 是首个完整实现 Harness 全生命周期自动化的方案，从生成、修复到进化全部由模型完成，不再依赖人工架构设计]。
:::

### 2.3 Harness 智能的定义

> 基于 JIT 范式，论文正式提出了“Harness Intelligence（Harness 智能）”的概念：

::: block-1
**Harness intelligence is the capacity to construct and refine the operational scaffold through which a model acts**. It has three defining properties: adaptivity, matching the harness to the task and backbone; reliability, producing executable behavior and recovering when synthesis fails; and evolvability, turning execution feedback into stronger future harnesses.
:::

简单来说，Harness 智能就是构建和优化 Agent 运行脚手架的能力，它包含三个核心属性：

- **适应性**：能够根据任务类型和基础模型的特性，匹配最合适的 Harness 架构；
- **可靠性**：生成的 Harness 能够稳定执行，出现错误时可以自动修复；
- **进化性**：能够从执行反馈中学习，不断生成更优的 Harness，持续推进性能边界。

::: block-1
Harness 智能是独立于基础模型能力之外的全新维度。:mdnice-bold-italic[基础模型负责思考，Harness 智能负责组织思考的方式。两者正交，也就是说，无论模型强弱，都可以通过提升 Harness 智能来获得性能增益；同样的 Harness 智能，也可以迁移到不同的基础模型上生效。]
:::

## 三、四模块协议：构建可生成的标准化 Harness 设计空间

> **要让模型自动生成 Harness**，首先不能让它自由生成无约束的代码，**必须先定义一套标准化的接口协议，把 Harness 拆成可组合、可验证的模块**。这就好比搭乐高，先规定好每块积木的接口尺寸和连接方式，模型只需要在预制的模块库中选择组合，就能搭出符合规范的成品，而不是从零开始雕刻。

### 3.1 四模块分解框架

研究团队将完整的 Agent Harness 抽象为四个互操作的模块：记忆模块（M）、规划模块（P）、动作模块（A）、能力编排模块（F）。**任何一套 Harness 都可以表示为这四个模块的组合，用公式表示为：**

$$
h = (M, P, A, F) \in \mathfrak{M} \times \mathfrak{P} \times \mathfrak{A} \times \mathfrak{F}
$$

其中 $\mathfrak{M}$、$\mathfrak{P}$、$\mathfrak{A}$、$\mathfrak{F}$ 分别是符合协议的记忆、规划、动作、能力编排模块的实现空间。

> 四个模块在运行时形成单向依赖的执行闭环：

1. **记忆模块 M**：接收完整的历史交互和当前控制器状态，加工生成模型可见的上下文视图 $v_t$。:mdnice-bold-italic[它决定了模型能看到什么、记得什么，是保留全部对话历史，还是只保留关键事实，或是分层组织的记忆结构。]
2. **规划模块 P**：根据任务目标、当前状态和记忆视图，生成本地执行指令 $d_t$。:mdnice-bold-italic[它决定了任务的执行节奏，是没有显式规划走一步看一步，还是线性分步执行，或是基于 DAG 的并行执行]，甚至是动态分解子问题的递归规划。
3. **能力编排模块 F**：根据当前执行指令，从全局工具/技能库中激活当前阶段需要的能力集合 $C_t$。它:mdnice-bold-italic[决定了模型在当前步骤能调用哪些工具，是暴露全部工具，还是只开放当前阶段相关的工具，甚至可以动态生成子 Agent 等高级能力。]
4. **动作模块 A**：整合所有上下文信息，更新控制器状态，输出下一个动作 $e_t$，可能是工具调用，也可能是最终答案输出。它是:mdnice-bold-italic[执行的核心引擎，决定了动作的执行方式和错误处理逻辑。]

这个四模块框架把形态各异的 Agent Harness 都统一到了同一个坐标空间中。比如经典的 ReAct 范式可以表示为 $h_{ReAct} = (M_{full}, P_{\emptyset}, A_{react}, F_{all})$，也就是全量记忆、无显式规划器、标准 ReAct 循环、全量工具暴露；而递归式 Agent 可以表示为 $h_{rec} = (M_{subproblem}, P_{decomp}, A_{rec}, F_{route})$，也就是子问题隔离记忆、分解式规划、递归执行、工具路由。

::: block-1
:mdnice-highlight[所有的 Harness 差异，本质上都是这四个模块的选择和内部实现的差异]。有了这个统一的表示，生成 Harness 就是在四个维度的设计空间中做选择和实例化，大大降低了生成难度，也保证了生成结果的协议合规性。
:::

### 3.2 HarnessFactory 种子库

> 为了支撑模型训练，研究团队基于同一个四模块协议，重新实现了 13 种代表性的 Agent Harness，构建了初始种子库 $B_0$。这 13 种 Harness 涵盖了当前主流的 Agent 架构，包括 ReAct、Plan-and-Execute、ReSum、Flash-Searcher、GAM、MemoBrain、AggAgent、OAgent、AgentFold、HiAgent、DeepAgent、ROMA、AOrchestra。

**每一种 Harness 都对应四模块的不同组合**，比如有的用 DAG 规划，有的用分层记忆，有的用多分支聚合，有的用递归子智能体。:mdnice-bold-italic[这些种子 Harness 一方面作为训练时的参考模板，让模型知道有哪些成熟的架构可以借鉴；另一方面作为性能基准，用来衡量新生成的 Harness 是否实现了性能超越。]

随着系统运行，种子库会不断进化，加入更多性能更优的新 Harness，成为不断增长的经验库，支撑后续的生成和进化。

## 四、三阶段训练：从定制、修复到进化的能力闭环

> 有了标准化的设计空间，接下来就是**训练 JIT-Agent 模型逐步掌握 Harness 的生成、修复和进化能力**，整个训练分为三个阶段，层层递进。:mdnice-bold-italic[三个阶段分别对应 Harness 智能的三个核心属性：第一阶段训练适应性，第二阶段训练可靠性，第三阶段训练进化性。]

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/06.webp)

::: image-caption
图 3：JIT-Agent 训练流水线。第一阶段学习任务条件下的执行框架定制；第二阶段将失败的框架与执行诊断信息转化为有界修复轨迹；第三阶段通过将候选框架与现有库进行对比执行在线演化，优化解耦的奖励、延迟和成本优势，并保留能推进性能边界的设计。
:::

### 4.1 第一阶段：任务条件下的 Harness 定制

第一阶段的目标是**教会模型根据任务生成符合协议的 Harness，并且初步学会偏好性能更优的设计。**

::: block-1
训练数据由更强的教师模型生成：:mdnice-highlight[对于每个训练任务，先从种子库中采样 3 个同类型的 Harness 作为参考上下文，然后让教师模型生成任务专属的 Harness]，只有通过协议验证和执行检查的样本才会被保留，形成训练数据集。
:::

阶段一的训练包含两个耦合的目标：

- 第一个是**生成模仿目标，让模型学习教师生成的 Harness 结构**，保证输出的 Harness 符合四模块协议，语法和接口正确。这是最基础的合规性要求。
- 第二个是偏**好学习目标，针对同一个任务，对比不同 Harness 的实际表现，奖励更高、延迟更低、成本更优的 Harness 会获得更高的偏好分数**。模型需要学习不仅生成能用的 Harness，更要生成好用的 Harness，在保证效果的同时兼顾执行效率。

通过第一阶段训练，:mdnice-bold-italic[JIT-Agent 已经可以针对给定任务生成合规且性能不错的专属 Harness，但还不能保证 100%的可执行性，也不具备自我优化的能力。]

### 4.2 第二阶段：失败 Harness 的修复学习

> 第一阶段生成的 Harness 中，有一部分会通不过静态验证或者运行时检查，比如存在接口不匹配、工具调用错误、运行时异常等问题。**第二阶段的目标就是教会模型修复这些有缺陷的 Harness，提升系统的可靠性。**

研究团队没有丢弃这些**失败样本，而是将它们转化为修复训练数据**。对于每一个失败的 Harness，连同对应的诊断报告，:mdnice-bold-italic[由教师模型生成结构化的修复补丁，应用补丁后重新验证，最多两轮修复内能够正常执行的轨迹就会被保留为训练样本。]

第二阶段的训练目标不是模仿一次性生成正确的 Harness，而是**模仿修复的过程：给定初始的失败 Harness 和诊断信息，模型学习生成对应的修复补丁，逐步修正问题，直到 Harness 可以正常执行。**

::: block-1
:mdnice-highlight[这个设计让 JIT-Agent 具备了容错能力：即使生成的 Harness 有小问题，也可以通过有限次数的自动修复恢复可执行性，而不是直接失败]。这对于实际部署至关重要，它保证了系统的稳定性，不会因为生成过程中的小偏差导致任务完全无法执行。
:::

### 4.3 第三阶段：在线进化的 Evo-GDPO

> 第三阶段是整个训练流程最核心的创新，研究团队提出了**进化式组解耦策略优化（Evolutionary Group-Decoupled Policy Optimization，Evo-GDPO）**，目标是:mdnice-bold-italic[让 JIT-Agent 学会生成超越现有水平的 Harness，持续推进性能边界，实现自我进化。]

传统的强化学习方法通常只关注组内相对表现，而 **Evo-GDPO 的核心是让模型学会超越历史最优**。具体的训练流程如下：

::: block-1
对于每个任务实例，先从当前的 Harness 库中取出一组历史最优设计作为参考，然后让模型生成一组候选 Harness。所有候选 Harness 和历史最优 Harness 在完全相同的条件下执行，对比三个维度的表现：任务奖励（核心指标）、执行延迟、API 成本。
:::

奖励计算遵循严格的优先级：

- 首先看任务奖励，只有候选 Harness 的奖励不低于历史最优，延迟和成本的优势才会被计入；
- 如果奖励更低，直接判定为更差。

三个维度的信号分别做归一化处理，避免数值量级差异互相干扰，然后加权合并成综合优势分数，其中任务奖励的权重最高。

最后使用 PPO 风格的裁剪目标更新模型策略，同时加入 KL 散度约束避免更新幅度过大。更新后，只有那些奖励不低于当前前沿，并且至少在延迟或成本上有严格提升的候选 Harness，才会被加入到 Harness 库中。

**这种机制保证了 Harness 库只会越来越好，不会出现退化**。同时模型:mdnice-bold-italic[在训练过程中不断学习如何设计出比历史最优更好的 Harness，逐渐掌握进化的能力]，而不是仅仅模仿已有的设计。

### 4.4 两种推理模式

> 训练完成的 JIT-Agent 支持两种推理模式：

- **静态推理模式**：针对单个任务，并行生成 N 个候选 Harness，从中选择最优的一个执行。这种模式可以增加生成多样性，提升单次任务的表现，任务之间相互独立，不共享经验。
- **流式推理模式**：针对连续的任务流，每处理一个任务，就将表现优秀的 Harness 更新到库中，后续任务可以检索这些历史经验作为参考。这种模式下，系统会随着任务处理量的增加不断积累经验，越跑越强，实现持续的在线进化。

## 五、全面实证：性能、成本与泛化性的多重验证

> 研究团队**在 9 个 Agent 基准测试上开展了全面实验，覆盖深度研究、日常工作、规划、工作空间四大类任务**，测试了多个主流模型的适配效果。

实验结果从多个维度:mdnice-bold-italic[验证了 JIT-Agent 的效果，不仅性能提升显著，成本优势明显，还展现出极强的跨模型泛化能力。]

### 5.1 同模型下的性能提升

> 首先是控制变量实验：**保持基础模型不变，将默认的 Harness 替换为 JIT 生成的 Harness**，对比性能差异。实验覆盖了 GLM-5.2 和 DeepSeek-V4-Flash 两个主流模型，9 个基准测试，共 18 组直接对比。**表 3 呈现了完整的实验结果。**

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/07.webp)

::: image-caption
表 3：九项智能体基准测试的主要结果。前缀标注 “JIT-Agent” 的行与其对应的原版主干模型使用相同的基础模型，仅将默认执行架构替换为 JIT 生成的执行框架。分数采用 0–100 分制，分数越高代表表现越好。每列中的第 1、第 2、第 3 名结果分别以浅绿、黄绿、黄色高亮标注；第 1 名同时加粗显示。未进行的评测以 “-” 表示。
:::

18 组对比中 JIT 生成的 Harness 全部实现性能提升，没有任何一组出现下降。其中：

- GLM-5.2 搭载 JIT-Agent 后，九个基准的平均得分从 74.1 提升到 81.8，平均提升 7.7 分，最高在 DeepPlanning-Travel 任务上提升 20.2 分；
- DeepSeek-V4-Flash 搭载 JIT-Agent 后，平均得分从 66.7 提升到 75.5，平均提升 8.8 分，最高在 DeepPlanning-Shopping 任务上提升 24.8 分。

:mdnice-bold-italic[从任务类型来看，提升最明显的是需要持续状态管理和约束跟踪的规划类任务，其次是深度研究和开放交互类任务。这说明 JIT 生成的 Harness 不仅仅是优化了提示词，更是真正改进了模型分配上下文、分解长程目标、协调外部动作的方式。]

::: block-1
和闭源旗舰模型对比，JIT-Agent 加 GLM-5.2 在 9 个基准中的 7 个取得第一名，其中 DeepSearchQA 达到 93.9 分，AgentIF 达到 69.9 分，全面超越 GPT-5.6、Gemini 3.5 Flash 等旗舰模型。DeepSeek-V4-Flash 搭载 JIT-Agent 后，也全面超过了更强的 DeepSeek-V4-Pro 原生版本，平均领先 8.7 分。

:mdnice-highlight[这意味着通过 Harness 智能的提升，完全可以弥补模型本身的代际差距，用更低级别的模型实现超越更高级别模型的效果]。
:::

### 5.2 与成熟固定 Harness 的对比

> :mdnice-bold-italic[为了单独验证 Harness 设计的贡献，实验控制基础模型不变，将 JIT-Agent 与 Claude Code、Codex、OpenCode、Hermes、NanoBot 等 5 种成熟的固定 Harness 进行对比，同时统计了每个方案的 token 消耗量和 API 成本。]

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/08.webp)

::: image-caption
表 4：与先进智能体执行框架的对照比较，对于每个固定主干模型，报告任务性能（Perf.）、单例平均 token 消耗量（千令牌，#Tokens (K)）以及单例 API 成本（美元）。性能数值越高越好，token 数与成本数值越低越好。每个主干模型分组内的最优值已加粗。
:::

上表 4 呈现了详细的对照数据。结果显示，:mdnice-bold-italic[在 6 组模型-基准组合中，JIT-Agent 有 4 组取得了最高的性能。更值得关注的是成本表现：JIT-Agent 在所有 6 组场景中，都是 token 消耗最少、API 成本最低的]。相比每个场景中成本最低的固定 Harness，JIT-Agent 将单任务成本降低了 14.9%到 54.1%，平均降幅达到 36.0%。

::: block-1
一个典型的例子是 DeepSeek-V4-Flash 跑 xBench-DS 任务：JIT-Agent 的性能为 82.0 分，每任务消耗 212K token，成本 0.039 美元；而之前表现最好的固定 HarnessNanoBot，性能为 78.0 分，消耗 527K token，成本 0.075 美元。:mdnice-highlight[JIT-Agent 不仅性能更高，token 消耗还不到一半，成本直接砍半。]
:::

> 图 4 的成本-性能帕累托前沿可以更直观地看到这种优势。

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/09.webp)

::: image-caption
图 4：DeepSearchQA 与 AgentIF 任务上的成本 - 性能权衡对比。包含 (a) DeepSearchQA 和 (b) AgentIF 两个子图，分别对应深度研究与日常办公两类场景。标记颜色区分不同执行框架，实心与空心圆圈分别代表 DeepSeek-V4-Flash 与 Qwen3.6-Flash；横轴为单例 API 成本，纵轴为任务性能，深绿色线为全局帕累托最优边界。
:::

**JIT-Agent 的点都明显偏向图的左上角，也就是高性能、低成本的最优区域**。:mdnice-bold-italic[其中 DeepSeek-V4-Flash 加 JIT-Agent 在 DeepSearchQA 任务上是强帕累托最优点]，相比最强的固定 HarnessNanoBot，性能高 4.7 分，成本低 49.6%，实现了性能和成本的双重优化。

### 5.3 跨模型家族的泛化性

> **Harness 智能是否只针对特定模型有效？为了验证泛化性，实验测试了 DeepSeek V4、Qwen 3.6、Mimo 2.5 三个模型家族**，每个家族包含 Flash 和 Pro 两个版本，共 6 个基础模型，在 4 个基准上进行测试，对比默认 ReAct Harness 和 JIT 生成 Harness 的差异。**图 5 呈现了完整的对比结果。**

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/10.webp)

::: image-caption
图 5：JIT 生成的执行框架相对 ReAct 框架持续提升配对主干模型的性能。行分组展示三个模型家族及每个家族的两个变体；列对应 DeepSearchQA、AgentIF 单日任务、深度规划 - 购物、OfficeBench 四个基准。每个子图对比同一主干模型在固定 ReAct 框架（虚线）与 JIT 生成框架（实线）下的表现，标注数字为绝对分数提升。DeepSearchQA 使用 100 样本子集，其余三个基准使用 50 样本子集。
:::

**全部 24 组对比中，JIT 生成的 Harness 性能全部超过 ReAct 基准，平均提升 7.6 分**。:mdnice-bold-italic[其中 DeepSeek V4 家族平均提升 10.2 分，Qwen 3.6 家族平均提升 4.0 分，Mimo 2.5 家族平均提升 8.6 分。] 提升效果在所有模型家族和所有版本上都一致存在，没有出现任何负向迁移。

::: block-1
这说明 **Harness 智能是一种可迁移的通用能力，它不依赖于特定模型的内部特性**，而是基于任务结构来优化运行架构。:mdnice-highlight[无论是什么模型，只要接入 JIT 生成的 Harness，都能获得稳定的性能提升。]
:::

### 5.4 流式进化的实际效果

> **最后验证在线进化的能力，对比静态 JIT 和流式 JIT 的表现**。静态模式下每个任务独立生成 Harness，流式模式下 Harness 库会随着任务推进不断更新，后续任务可以复用之前的经验。图 6 呈现了三个基准上的对比曲线。

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/11.webp)

::: image-caption
图 6：任务流上的流式测试时执行框架演化，展示深度规划 - 购物、深度规划 - 旅行、OfficeBench 三个任务上的累积准确率（上）、单任务 API 成本（中）、单任务工具调用次数（下）。虚线代表静态 JIT 模式，各任务的框架生成相互独立；实线代表流式 JIT 模式，随新任务到达持续融入执行反馈。阴影区域为流式轨迹的局部波动范围。
:::

实验结果显示，在三个测试基准上，**流式 JIT 的最终累计准确率都高于静态 JIT，并且随着任务数量增加，优势逐渐显现**。而 API 成本和工具调用量并没有因为进化而显著增加，和静态模式保持在相近水平。

::: block-1
这符合 Evo-GDPO 的设计目标：:mdnice-bold-italic[只有真正能推进性能前沿的 Harness 才会被保留入库，不会为了进化而无限制地增加计算开销。随着处理的任务越来越多，Harness 库会积累越来越多的优质设计，系统的整体表现也会持续提升，形成正向循环。]
:::

## 六、具象观察：不同任务下生成 Harness 的形态差异

> 数据上的提升是结果，:mdnice-bold-italic[更有价值的是观察 JIT-Agent 到底生成了什么样的 Harness，为什么能带来这么大的性能提升。针对不同结构的任务，JIT-Agent 会生成完全不同的运行架构，而不是简单调整提示词或者增减工具。]

研究团队可视化了多个典型任务生成的 Harness，**这里选取两个差异最大的案例，展示 JIT 定制化的实际形态。**

### 6.1 工件生产类任务：DAG 式图执行架构

> 针对联系人处理、表格生成、邮件发送这类有明确产出物的跨应用任务，JIT-Agent 生成了名为 Palimpsest 的 Harness 架构。图 7 展示了这套架构的完整结构。

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/12.webp)

::: image-caption
图 7：Palimpsest：基于图规划的工件执行模式。GraphPlanPlanning 模块将联系人处理请求转化为有向无环图（DAG）；GraphPlanAction 模块以受限的宽度与深度执行该图；GraphPlanMemory 模块则存储可复用的工件与推理状态。其核心是基于依赖图的执行：规划模块将整个任务拆解为有依赖关系的 DAG 节点，比如发现联系人文件、读取表结构、筛选记录、创建工作簿、发送邮件，下游节点必须等上游节点完成才能开始执行。动作模块按照 DAG 调度执行，支持有限的并行度，同时严格控制深度。记忆模块专门维护工件存储，把中间生成的表格、文档等结构化产物存起来，下游节点直接读取使用，不需要从对话历史中重新提取。
:::

**这种架构非常适合有明确交付物的生产类任务**，整个执行流程清晰可控，中间产物可复用，避免了传统 ReAct 模式下反复从长对话中找信息的低效问题，也减少了模型的上下文消耗。

### 6.2 深度研究类任务：递归委托式架构

> 针对多跳线索查证这类不确定性很高的深度研究任务，JIT-Agent 生成了名为 Trapdoor 的 Harness 架构。图 8 展示了这套架构的运行逻辑。

![文章插图](/articles/2026-08-30-jit-agent-agent-harness-2ae380ec/images/13.webp)

::: image-caption
图 8：Trapdoor：委派工具调用背后的有界研究模式。合成的委派能力由编排器循环拦截，由其运行一个有界子智能体，并将提取的事实写入事实图记忆中。这类任务无法预先规划好固定路径，因为研究方向会随着发现的线索不断变化。所以这套架构没有用固定 DAG，而是采用动态分解加递归委托的模式：规划模块动态生成子问题，能力编排模块合成一个委托工具，调用这个工具就会启动一个独立的子 Agent，子 Agent 有自己的私有记忆和研究工具，在有限步数内完成子问题研究，然后把结果返回给主 Agent。主 Agent 的记忆模块专门维护事实图谱，把各个分支返回的结构化事实存起来，不需要保留子 Agent 的完整执行轨迹。
:::

:mdnice-bold-italic[这种架构既可以并行探索多个研究分支，又不会让子 Agent 的冗余历史挤占主循环的上下文空间，非常适合开放式的深度研究任务。它相当于让模型自动学会了分而治之的策略，把复杂问题拆成小问题分头解决，再汇总结果。]

::: block-1
从以上两个案例对比可以看到，**同样的 JIT-Agent，针对不同的任务结构，生成了完全不同的运行架构**：一个是有向无环图的流水线执行，一个是动态递归的分支探索。:mdnice-highlight[协议只约束模块接口，不约束具体行为，任务是什么结构，Harness 就长成什么结构。这正是 JIT 范式的核心价值所在。]
:::

## 七、总结与展望：Harness 智能的长期价值

JIT-Agent 这项工作提出了一个全新的 Agent 能力增长维度——Harness 智能。在此之前，行业提升 Agent 能力的路径无非两条：堆更大的模型，或者堆更多的推理算力，而 **JIT-Agent 证明，专门优化 Harness 层带来的提升，完全可以媲美甚至超过模型本身的缩放。**

### 7.1 核心贡献总结

> 这项工作的核心贡献可以归纳为三点：

- 第一，:mdnice-bold-italic[提出并定义了 Harness 智能的概念，证明它是一种可训练、可迁移、可复利的能力，与基础模型缩放完全正交]。这打破了 Agent 能力由模型决定的固有认知，为 Agent 性能提升打开了全新的技术路径。
- 第二，:mdnice-bold-italic[设计了四模块 Harness 协议和三阶段训练流水线，实现了 Harness 生成、修复、进化的完整自动化闭环]。这套框架具有很强的通用性，可以兼容不同的基础模型和不同的任务类型。
- 第三，:mdnice-bold-italic[通过全面的实验验证了效果：JIT 生成的 Harness 可以稳定提升不同模型的性能，最高提升超过 20 分，成本平均降低 36%，性能可以超越更高量级的基础模型]，同时具备良好的跨模型泛化性和持续进化能力。

### 7.2 未来展望

> 从长期来看，JIT 范式代表了 Agent 系统演进的一个重要方向。未来的 **Agent 系统走向模型与 Harness 协同设计、共同进化的新范式。**

- 短期内，生产级 Agent 运行时可以借鉴 JIT 的思路，保留稳定的核心架构，同时让模型根据任务动态调整部分模块，比如动态选择记忆策略、规划模式和工具集合，在稳定性和灵活性之间取得平衡。
- 长期来看，基础模型本身可能会内化 Harness 智能，不再需要单独的元模型来生成 Harness。模型不仅学会在给定的架构下执行任务，还学会根据任务特点调整自己的运行架构，实现真正的自我组织、自我进化。

::: block-1
总之，:mdnice-highlight[JIT-Agent 开创了即时 Harness 生成的全新范式，将 Agent Harness 从人工设计的提前固化工件，转变为可训练、可进化的即时合成能力]。它证明 **Harness 智能是独立于模型参数与推理算力的第三增长曲线，为 Agent 能力提升开辟了全新的技术方向**，也将推动整个 Agent 领域从模型为王走向架构与模型并重的新阶段。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**
- **[刚刚，SpaceXAI 开源 Grok Build：一个 Rust AI Agent Harness 的系统级拆解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903721&idx=1&sn=20b3a6235e2c8a3bae1df8e27de39301&scene=21#wechat_redirect)**
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**

:::
