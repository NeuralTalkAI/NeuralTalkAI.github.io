关键词：**Harness 自演化**、LLM 智能体、Harness 更新、Harness 获益、指令遵循

> *当 LLM 智能体在真实环境中执行任务时，系统行为并非仅由底层模型决定。提示词、技能、记忆、工具——这些围绕在模型周围的可编辑外部组件，即所谓的"Agent Harness"——同样深刻地塑造着智能体的观察、推理、行动与错误恢复方式。:mdnice-bold-italic[一个自然的想法是：能否让智能体从执行经验中自动学习，将失败教训、成功策略写回 Harness，使后续任务受益？这便是 Harness 自演化的核心愿景。]*

然而，现有研究几乎都在问同一个端到端问题：某种:mdnice-bold-italic[自演化方法（harness self-evolution）是否提升了智能体性能？这个问题虽然重要，却掩盖了性能提升的真实来源。]

::: block-1
增益可能来自 :mdnice-highlight[Evolver 产出了更高质量的 Harness 更新]，也可能来自:mdnice-highlight[任务求解智能体更有效地利用了更新后的 Harness]。端到端的分数无法区分这两种贡献。
:::

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/01.webp)

- Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents
- https://arxiv.org/pdf/2605.30621

::: related-title
**相关推荐**
:::

::: related-links
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**
- **[异构智能体之间的“读心术”：稠密 KV 缓存高效通信实现 2-3 倍算力优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902336&idx=1&sn=d667567b5407191944b276e13a16ba7d&scene=21#wechat_redirect)**
- **[LLM负责思考，SLM负责记忆。LightMem小模型天团重塑 Agent 记忆，延迟仅 83 毫秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900501&idx=1&sn=2311904700b9bfd0c8da1f96e2e98a4b&scene=21#wechat_redirect)**

:::

> 来自宾夕法尼亚大学、加州大学圣克鲁兹分校的团队做了一件关键的事：:mdnice-bold-italic[将 Harness 自演化拆解为两种独立能力——Harness 更新（Harness-updating）与 Harness 获益（Harness-benefit），并在三个代表性智能体基准（SWE-bench Verified、MCP-Atlas、SkillsBench）上对七个 LLM（从 Qwen3.5-9B 到 Claude Opus 4.6）进行了系统性交叉实验。]

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/02.webp)

::: image-caption
图 1：Harness 自进化整体框架示意图：该图搭建论文整套实验的底层运行闭环，清晰划分三大固定模块：权重冻结的大模型、由记忆 / 工具 / 提示词 / 技能组成的外部配套、负责迭代的更新器。整套流程以任务执行产生的轨迹、失败反馈作为原始输入，更新器完成诊断后修改配套组件，全程不改动模型参数。该框架最核心价值是拆分任务求解模型与更新模型两个独立角色，让本文能够解耦配套更新、配套受益两类完全独立的模型能力；若二者共用同一模型，端到端性能指标会混淆两种增益来源，无法开展分层定量分析，是全文实验设计的基础前提。
:::

两个反直觉的发现浮出水面：

- 其一，**Harness 更新能力在基础能力维度上惊人地扁平** ——即使是 Qwen3.5-9B 产出的 Harness 更新，其下游增益也与 Claude Opus 4.6 相当；
- 其二，**Harness 获益能力与基础能力呈非单调关系** ——中型模型获益最大，而弱模型获益最少。弱模型的问题出在两个失效模式上：要么根本没有把 Harness 加载进工作上下文（激活失败），要么加载了却不按指导执行（遵循失败）。

::: block-1
这些发现指向一个明确的实践原则：:mdnice-highlight[把算力预算分配给任务求解智能体而不是 Evolver，并将 Harness 调用和长程指令遵循能力纳入智能体训练目标。]
:::

## 本文目录

- 一、引言
- 二、相关工作
  - 2.1 Harness 工程
  - 2.2 LLM 智能体的自演化
- 三、Harness 演化能力
  - 3.1 预备知识：Harness 状态与 Evolver
  - 3.2 演化协议
  - 3.3 能力度量
- 四、实验
  - 4.1 实验设置
  - 4.2 Evolver 侧分析
  - 4.3 智能体侧分析
- 五、局限与结论

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/03.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、引言

> 大语言模型（LLM）已成为语言理解、推理和任务求解的通用基础。越来越多的 LLM 还被用于驱动智能体系统，这些系统与外部环境交互、调用工具、操作软件接口并完成长程任务。:mdnice-bold-italic[在这些场景中，系统行为不仅取决于底层模型，还取决于一个外部的 Agent Harness：提示词、技能、记忆、工具等组件，它们塑造着模型如何观察、推理、行动以及从错误中恢复。改进一个智能体系统，越来越意味着不仅要改进基础模型，还要改进围绕其周围的可编辑 Harness。]

在当前实践中，Harness 通常由人工设计。然而，这种人工设计在部署环境中显得脆弱：:mdnice-bold-italic[任务分布会漂移，边界情况会浮现，有用的操作流程只有在系统与真实任务交互之后才能被发现。一个自然的应对之策是，根据执行证据自动更新 Harness：失败案例、反馈信息、执行轨迹和成功流程都可以写回 Harness，并在未来任务中被复用。]

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/04.webp)

::: image-caption
图 1：配套组件自进化整体框架示意图：该图搭建论文整套实验的底层运行闭环，清晰划分三大固定模块：权重冻结的大模型、由记忆 / 工具 / 提示词 / 技能组成的外部配套、负责迭代的更新器。整套流程以任务执行产生的轨迹、失败反馈作为原始输入，更新器完成诊断后修改配套组件，全程不改动模型参数。该框架最核心价值是拆分任务求解模型与更新模型两个独立角色，让本文能够解耦配套更新、配套受益两类完全独立的模型能力；若二者共用同一模型，端到端性能指标会混淆两种增益来源，无法开展分层定量分析，是全文实验设计的基础前提。
:::

::: block-1
我们将这种设定称为**Harness 演化** （如图 1 所示）：模型权重保持冻结，而外部 Agent Harness 随时间推移不断被修订。:mdnice-highlight[近年来涌现的自演化智能体方法[18-22]正是沿着这一思路，覆盖了多种 Harness 组件，并在端到端任务上展现了对非演化基线的改善]。在这些工作中，Harness 更新通常由一个 LLM 从执行证据中生成；我们将承担这一更新角色的模型称为**Evolver**。
:::

尽管进展迅速，这些方法的评估仍然只问一个端到端的问题：某种自演化方法是否有效提升了智能体性能？这个问题固然重要，却掩盖了性能提升的来源。增益可能来自 Evolver 产出了更高质量的 Harness 更新，也可能来自任务求解智能体在任务执行中更有效地利用了更新后的 Harness。端到端的分数无法区分这些贡献，留下两个实际的问题悬而未决：**哪些模型能产出有用的 Harness 更新？哪些模型能从中获益最大？**

> 为了回答这些问题，我们在三个智能体基准和七个 LLM 上，分析了模型在 Harness 自演化中行使的两种演化能力：

1. **Harness 更新能力** （harness-updating）——从执行证据中产出有用的持久化 Harness 更新的能力；
2. **Harness 获益能力** （harness-benefit）——在任务求解中从更新后的 Harness 中获益的能力。

模型作为 Evolver 时行使 Harness 更新能力，作为任务求解智能体时行使 Harness 获益能力。我们通过将七个 LLM（覆盖开源和闭源系列、跨越不同能力层级）分别作为智能体和 Evolver 进行配对，在三个代表性智能体基准上开展了全面的交叉实验。

> 我们的分析揭示了 Harness 演化能力与基础能力（即模型在没有 Harness 演化时的任务求解能力）之间的两种系统性脱钩现象，如图 2 所示。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/05.webp)

::: image-caption
图 2：研究发现概览。（A）Harness 更新能力在基础能力维度上呈扁平分布——跨越不同能力层级的模型产出的 Harness 更新带来了相似的增益。（B）Harness 获益能力在基础能力维度上呈非单调分布——中型模型获益最多，弱模型因 Harness 激活失败（无法加载 Harness）和 Harness 遵循失败（加载后无法忠实遵循）而获益极少。
:::

- **首先，Harness 更新能力在基础能力维度上呈扁平分布。** 当我们固定任务求解智能体、仅变化 Evolver 模型时，来自不同能力层级的模型产出的 Harness 更新带来了惊人相似的增益，且没有任何一个 Evolver 能在所有基板上都占据主导地位。:mdnice-bold-italic[我们的案例研究进一步表明，即使是 Qwen3.5-9B 作为 Evolver，其产出的 Harness 更新带来的下游增益也与 Claude Opus 4.6 相当——尽管两者在基础能力上存在巨大差距。]
- **其次，Harness 获益能力在基础能力层级上呈非单调分布。** 中型模型（如 GPT-OSS-120B）从更新后的 Harness 中获益最大，而强模型（如 Claude Opus 4.6）因触达性能天花板而获益较少。然而，弱模型一端却不能用同样的天花板效应来解释：:mdnice-bold-italic[像 Qwen3-32B 这样的模型，其基础能力之上的提升空间最大、按理应获益最多，却恰恰获益最少]。

> 这些发现转化为 Harness 自演化系统的三条设计指导原则。

1. **将算力预算分配给任务求解智能体而非 Evolver**：在所有基准上，不同 Evolver 之间的 Harness 更新增益差距至多只有 3.1 个百分点，因此扩大 Evolver 规模带来的回报有限；演化后的性能更多地取决于任务求解智能体而非 Evolver。
2. **将 Harness 调用能力纳入智能体训练**：弱模型常常根本不加载 Harness（例如，Qwen3-32B 的技能加载率仅为 25%，而强模型约为 96%），因此 Harness 调用应被视为一项需要学习的一等技能。
3. **加强长程指令遵循能力**：即使加载了 Harness，弱模型的遵循度在轨迹展开过程中衰减的速度是强模型的四倍以上，使得持续指令遵循成为下游智能体训练的第二个关键目标。

## 二、相关工作

### 2.1 Harness 工程

> **LLM 智能体越来越多地被部署为复合系统：一个冻结的模型被外部组件所包围，这些组件负责中介推理、工具使用、记忆访问和环境交互。** 我们将这一外部层称为 Agent Harness。:mdnice-bold-italic[近期工作将 Harness 视为一等设计对象，主要区别在于暴露给智能体的组件类型。]

- 提示词和指令提供自然语言指导；
- 工具暴露外部服务并定义智能体如何发现、调用和验证它们；
- 记忆存储先前的观察、事实和策略以供后续检索；
- 技能将可复用的操作流程打包为可调用模块；
- 代码则将 Harness 本身视为可被智能体提案优化的可执行源代码。

这些工作确立了 Harness 作为可编辑智能体状态的地位。我们的工作将关注重心从 Harness 的表示形式转向模型在更新和获益于 Harness 时所表现出的能力。

### 2.2 LLM 智能体的自演化

> 在 Harness 包含什么内容之外，一条互补的研究线路追问的是 Harness 如何从执行经验中更新。

早期系统通过回合级或任务级的语言反馈来适配智能体：口头自反思和迭代自反馈通过将经验教训反馈到上下文中来改善后续尝试。

更近期的方法则将持久化的 Harness 组件作为自演化的基本单位，从执行轨迹中更新提示词、记忆、技能或工具。

::: block-1
**总体而言，这些方法表明，将执行经验写回 Harness 可以改善下游任务表现**。然而，该领域中的评估通常报告的是某一种更新流程与某一个目标智能体在某一基板上配对后的端到端增益。

:mdnice-bold-italic[这样的分数混淆了三种贡献来源：智能体的基础能力，Evolver 的 Harness 更新能力，以及智能体的 Harness 获益能力。我们的工作以受控分析补充了这些方法：独立地变化任务求解智能体和 Evolver，分别测量 Harness 更新能力与 Harness 获益能力，并检验两者是否与基础能力正相关。]
:::

## 三、Harness 演化能力

> 为了探索 Harness 自演化中的演化能力，我们考虑 Harness 自演化的设定：通过对固定模型周围的外部 Harness 进行更新，在执行任务的过程中适配 LLM 智能体——智能体尝试一系列任务，Harness 基于智能体的执行证据而被更新。

**在本节中，我们形式化 Harness 演化协议，并定义两种演化能力**：

- Harness 更新能力（产出有用的 Harness 更新的能力）
- Harness 获益能力（从更新的 Harness 中获益的能力）

### 3.1 预备知识：Harness 状态与 Evolver

> **Agent Harness。** 我们用 Agent Harness 指代 LLM 执行任务时所依赖的、外部的、非参数化的上下文和基础设施。

形式上，在演化步 $t$，LLM 智能体定义为：

$$
A_t = (f, H_t) \tag{1}
$$

其中 $f$ 是智能体的模型主干，$H_t$ 是第 $t$ 步之后的 Harness 状态。遵循常见的 Harness 自演化设定，我们保持 $f$ 冻结，仅更新 $H_t$ 中可编辑的组件（如提示词、技能、记忆），而固定其他组件（如工具接口和执行策略）。

> **Evolver。** Evolver 是将智能体的执行证据转化为 Harness 更新的更新流程。

近期的自演化智能体系统[49,50,15,19]越来越多地用 LLM 智能体来实例化这一流程。形式上，给定前一步的 Harness $H_{t-1}$ 和累积的执行证据 $D_t$，Evolver $e$ 提出 Harness 更新并将其施加到 $H_{t-1}$ 上，得到下一步的 Harness：

$$
\Delta H_t = e(H_{t-1}, D_t), \quad H_t = \text{Apply}(H_{t-1}, \Delta H_t) \tag{2}
$$

其中 $\text{Apply}$ 表示将 $\Delta H_t$ 提交到 $H_{t-1}$ 上的操作。

### 3.2 演化协议

> 遵循常见的 Harness 自演化流程，我们将该协议形式化为任务求解与 Harness 演化之间的迭代循环。从初始 Harness $H_0$ 出发，协议迭代 $T$ 步。在每一步中，智能体在一批任务上运行，收集执行证据，然后 Evolver 为下一步更新 Harness。

形式上，给定智能体 $A_{t-1} = (f, H_{t-1})$ 和任务批次 $X_t$，$A_{t-1}$ 尝试求解每个任务 $x \in X_t$ 并输出：

$$
(\tau_{t,x}, y_{t,x}) = \text{Solve}(A_{t-1}, x) \tag{3}
$$

其中 $\tau_{t,x}$ 是执行轨迹，$y_{t,x}$ 是最终输出。执行证据 $D_t$ 为：

$$
D_t = \{(x, \tau_{t,x}, y_{t,x}) : x \in X_t\} \tag{4}
$$

Evolver 从 $H_{t-1}$ 和 $D_t$ 生成更新后的 Harness $H_t$（如公式 2 所示），从而得到下一个智能体 $A_t = (f, H_t)$。这一循环重复 $T$ 步，最终产出 Harness $H_T$。

### 3.3 能力度量

> 为了分析哪些模型产出有用的 Harness 更新以及哪些模型从中获益，我们形式化定义**三个度量指标，分别衡量两种 Harness 演化能力（Harness 更新能力与 Harness 获益能力）以及每个模型的基础能力。**

#### 指标1：基础能力与演化增益

给定任务集 $X = \bigcup_{t=1}^{T} X_t$，模型 $f$ 的基础能力是初始智能体 $A_0 = (f, H_0)$ 在 $X$ 上的任务求解表现：

$$
M_{\text{base}}(f) = J_X(f, H_0) \tag{5}
$$

其中 $J_X(f, H)$ 是衡量智能体 $(f, H)$ 在 $X$ 上表现的评分函数。

给定一个模型 $f$ 和一个 Evolver $e$，令 $H_T^{(f,e)}$ 表示以 $f$ 为智能体、$e$ 为 Evolver、从 $H_0$ 出发经过 $T$ 步演化后产生的最终 Harness。我们进一步定义成对演化增益为一个特定智能体-Evolver 配对 $(f, e)$ 相对于智能体演化前任务求解表现所获得的提升：

$$
\Delta(f, e) = J_X(f, H_T^{(f,e)}) - M_{\text{base}}(f) \tag{6}
$$

#### 指标2：Harness 更新能力

Evolver $e$ 的 Harness 更新能力是其产出能改善智能体任务求解表现的 Harness 更新的能力。形式上，定义为在一个锚定智能体集合 $F^\star$ 上的平均成对增益：

$$
\Delta_{\text{update}}(e) = \frac{1}{|F^\star|} \sum_{f \in F^\star} \Delta(f, e) \tag{7}
$$

#### 指标3：Harness 获益能力

模型 $f$ 的 Harness 获益能力是其在任务求解中从 Harness 自演化中获得的最大提升。在实践中，我们将其估计为一个固定锚定 Evolver 集合 $E^\star$ 上的最大成对增益：

$$
\Delta_{\text{benefit}}(f) = \max_{e \in E^\star} \Delta(f, e) \tag{8}
$$

## 四、实验

在本节中，我们对第三节中定义的两种 Harness 演化能力进行实证分析。我们在第 4.2 节展示 Evolver 侧的 Harness 更新能力分析，在第 4.3 节展示智能体侧的 Harness 获益能力分析。

### 4.1 实验设置

**数据集**我们在三个代表性智能体基准上系统评估，三者互补，兼顾任务复杂性、环境真实性与技能广度：

- SWE-bench Verified（SWE）——聚焦真实软件工程任务；
- MCP-Atlas（MCP）——依托真实MCP服务器，考验工具调用能力；
- SkillsBench（SB）——覆盖编程、数学、推理等多领域技能执行。

**模型**

- 共选用7个LLM主干，横跨开源与闭源、强基与轻量：Claude Opus/Sonnet/Haiku（4.5–4.6版）、Qwen3-235B-A22B、Qwen3-32B、GPT-OSS-120B，构成智能体侧分析主力；
- Evolver侧额外加入Qwen3.5-9B——本文最小模型，用于验证轻量开源模型能否驱动有效Harness更新。

**评估协议**

- 沿用第3.3节定义的三大指标：基础能力 $M_{\text{base}}(f)$、Harness更新增益 $\Delta_{\text{update}}(e)$、Harness获益增益 $\Delta_{\text{benefit}}(f)$。
- 以通过率（pass rate）为统一性能度量 $J_X$，采用原位（in-place）评估：每个任务 $x_t \in X_t$ 先在旧Harness $H_{t-1}$ 下执行并评分，其执行证据再用于生成 $H_t$。最终结果按任务流逐项聚合。

**实现细节**严格遵循第3.2节演化协议，采用固定轮次的“求解→演化”循环。所有配对共享提示模板、轨迹窗口、初始Harness $H_0$、任务流 $X$、演化预算 $\beta$ 与每任务回合上限。SWE与SB中演化技能；MCP中进一步扩展至技能、提示词与记忆。

### 4.2 Evolver 侧分析

> 为理解 Harness 更新能力在 LLM 之间如何变化，我们固定任务求解智能体，在第 4.1 节的七个 LLM 中变换 Evolver。具体而言，我们使用三个代表性 LLM——Opus 4.6、Sonnet 4.6 和 Qwen3-235B——作为锚定智能体集合 $F^\star$。对每个 Evolver $e$，我们在图 3 中报告了三个基准上的 $\Delta_{\text{update}}(e)$（定义见第 3.3 节）。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/06.webp)

::: image-caption
图 3：各更新器的配套组件更新增益$\Delta_{update}$（按Claude、Qwen、GPT-OSS模型家族分组）；每个子图加粗标记对应数据集最优、最差更新器，最优更新器随数据集发生变化。图表分为SWE、MCP、SB三组独立子图，量化7款更新器的平均跨智能体增益。任意数据集内最优与最差更新器的收益差值不超过3.1个百分点，且不存在能适配全部任务领域的通用最优更新器，Qwen3.5-9B轻量模型在SB数据集增益反超全部闭源大模型。该量化结果佐证图2(A)平缓趋势，证明更新仅需要提炼标准化业务流程，对模型推理算力门槛极低；任务领域会小幅影响更新产出效果，但模型规模无法作为筛选更新器的核心指标。
:::

> 观察 1：**Harness 更新能力在基础能力维度上呈扁平分布**。图 3 展示了两个模式：

1. $\Delta_{\text{update}}$ 在不同 Evolver 之间的分布很窄。在所有基准上，最佳与最差 Evolver 之间的差距至多仅为 3.1 个百分点，且没有模型能在不同基准上都占据绝对优势。Qwen3-235B 体现了这种重新洗牌：它在 SWE 上领先（8.2 个百分点），却在 MCP 上排名垫底（0.6 个百分点）。
2. 模型规模不具有预测性。最小的 Evolver Qwen3.5-9B 在 SB 上取得了最高增益（3.8 个百分点），超过了 Opus 4.6（2.3 个百分点）和 Qwen3-235B（1.5 个百分点）。

#### 案例研究：9B 的 Evolver 写出了与 Opus 程序同构的技能。

> 为了理解产生这些相当增益的机制，我们详细检视了一个代表性 SkillsBench 任务`flink-query`。我们固定任务求解智能体主干为 Opus 4.6，**比较其在三种 Evolver 条件下的轨迹（图 4）：无 Evolver、Qwen3.5-9B 作为 Evolver、Opus 4.6 作为 Evolver。**

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/07.webp)

::: image-caption
图 4：Qwen3.5-9B 与 Claude Opus 4.6 作为 Evolver 所产出的 Harness 更新对比。在 SkillsBench 的`flink-query`任务上，使用 Opus 4.6 作为固定任务求解智能体，比较三种条件：无演化技能（左，得分 0.67）、Qwen3.5-9B 演化的技能（中，得分 1.0）和 Opus 4.6 演化的技能（右，得分 1.0）。两种演化技能编码了程序上同构的指导，使同一智能体成功求解了任务。
:::

我们观察到：

- 没有演化技能时，智能体遗漏了 FINISH 事件过滤步骤，未能解决此任务（得分 0.67）；
- 在被 Qwen3.5-9B 或 Opus 4.6 注入技能后，同一智能体成功求解了该任务（得分 1.0）。检视两种技能的内容，我们发现它们在程序上是同构的——规定了相同的操作步骤序列，仅在实现的表面细节和冗长度上有所不同。9B 的开源 Evolver 因此达到了与前沿 Evolver 相同的程序内容。

> 观察 2：**演化后分数由模型的基础能力主导**，而非 Evolver 的身份。

为了理解任务求解智能体与 Evolver 对演化后表现的相对贡献，我们绘制了 $F^\star$ 中三个 LLM（Opus 4.6、Sonnet 4.6、Qwen3-235B）在第 4.1 节中七个 LLM 作为 Evolver 所更新的 Harness 下的任务求解表现，并与其各自的基础能力进行对比。MCP-Atlas 上的结果如图 5 所示。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/08.webp)

::: image-caption
图 5：MCP-Atlas 上的演化后分数分布。每个锚定智能体对应七个蓝点，分别表示其在七个不同 Evolver 的 Harness 更新下的表现；黑色刻度线为无演化的基线分数。同一智能体在不同 Evolver 之间的分数变动相对于不同智能体基础能力之间的差距而言很小。
:::

我们观察到：

1. **同一智能体在不同 Evolver 之间的分数散布远小于不同智能体之间的差距**。七个 Evolver 带来的同一智能体内部散布至多为 5.1 个百分点（Qwen3-235B），而 Opus 与 Qwen3-235B 基础能力之间的差距高达 36.0 个百分点。这一模式在 SWE 和 SB 上也持续存在。
2. **极端配对仍然有利于强智能体**。:mdnice-bold-italic[即使将最弱的锚定智能体与其表现最好的 Evolver 配对，去对抗最强的锚定智能体与其表现最差的 Evolver 配对，强智能体在每一个基准上仍领先 18.6 至 35.2 个百分点]。这两种模式在 SWE 和 SB 数据集上同样持续存在。**因此，演化后性能的瓶颈在智能体侧而非 Evolver 侧**，这促使了第 4.3 节的智能体侧分析。

::: block-1
将算力预算分配给任务求解智能体而非 Evolver：（i）在所有基准上，不同 Evolver 之间的 $\Delta_{\text{update}}$ 差距至多仅为 3.1 个百分点；（ii）演化后分数由智能体的基础能力所主导。
:::

### 4.3 智能体侧分析

> 为搞清楚 Harness 的获益能力在不同大模型（LLM）间怎么变化，我们保持 Evolver 不变，只在第 4.1 节提到的 LLM 主干中切换任务求解智能体：Opus 4.6、Sonnet 4.6、Haiku 4.5、Qwen3-235B、Qwen3-32B 和 GPT-OSS-120B。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/09.webp)

::: image-caption
表 1：各基准上的基础通过率（%）与 Harness 获益增益 $\Delta_{\text{benefit}}$（百分点）。每行为一个 LLM 主干作为任务求解智能体。粗体标记各基准内最大的 $\Delta_{\text{benefit}}$。数据表明，Opus 4.6 在 SWE 上基础通过率高达 74.2%但增益仅为 2.6 个百分点，而 Qwen3-235B 基础通过率 20.7%却获得 19.3 个百分点的增益。
:::

其中，Opus 4.6、Sonnet 4.6 和 Qwen3-235B 作为三个基准 Evolver（记为 $E^\star$），用于统一比较。对每个智能体 $f$，我们直接在表 1 和图 6 中给出其相对于基准的获益变化量 $\Delta_{\text{benefit}}(f)$——也就是它比基准 Evolver 多带来多少提升（或少多少）。所有实验均在相同设置下运行，确保结果可比。不引入额外调参或适配，纯粹考察模型原生能力对 Harness 效果的影响。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/10.webp)

::: image-caption
图 6：SWE 上 $\Delta_{\text{benefit}}$ 与基础通过率的关系。每个点代表一个 LLM 主干作为任务求解智能体，按基础通过率升序连接。Harness 获益增益并非随基础能力单调递增——中型模型峰值最高，两端均低。
:::

> **观察 1：$\Delta_{\text{benefit}}$ 在基础能力维度上呈非单调分布。** 如表 1 和图 6 所示，$\Delta_{\text{benefit}}$ 并不随基础能力单调递增。

在 SWE 上，增益在 Qwen3-235B 处达到峰值（19.3 个百分点），而更弱的 Qwen3-32B 仅获得 4.4 个百分点、更强的 Opus 4.6 仅获得 2.6 个百分点的增益。在 MCP 上，峰值移至 GPT-OSS-120B（7.0 个百分点），同样是基础能力两端增益更低。这一模式在能力谱的两端有不同的解释。

- 在高能力端，较小增益与天花板效应一致：强模型在初始 Harness 下已经能解决许多任务，留下了较少的改善空间。
- 在低能力端，较小增益反映的是另一种瓶颈——我们接下来对此进行诊断。

> **观察 2：弱层级模型因两种失效模式而获得较低的 $\Delta_{\text{benefit}}$。** 为理解为何基础能力低的弱层级模型获得的 $\Delta_{\text{benefit}}$ 低，我们在 SkillsBench 上开展了深入分析，识别出两种互补的失效模式：Harness 激活失败与 Harness 遵循失败，如图 7 所示。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/11.webp)

::: image-caption
图 7 Qwen3-32B 在 SkillsBench 上的两种 Harness 获益失效模式。左（`threejs`任务）：Harness 激活失败——无效的多键加载动作导致技能体从未进入上下文。右（`pg-essay-to-audiobook`任务）：Harness 遵循失败——技能虽已被加载，但智能体将其视作字面脚本，跳过了规定的 TTS 引擎回退链。
:::

#### 第一种模式是Harness 激活失败

> **弱层级模型常常无法将相关 Harness 组件（如技能）纳入其工作上下文**。为了在 SkillsBench 上量化这一点，我们报告了每个智能体的:mdnice-bold-italic[技能加载率（Skill-Load Rate, SLR），即其轨迹中至少主动加载了一个技能进入上下文的轨迹占比]。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/12.webp)

::: image-caption
表 2 报告了 HFR 及两个互补指标：SLR（衡量 Harness 激活）和加载通过率（LPR，衡量该模型在技能已加载轨迹中的通过率）。我们观察到两个模式。（i）强层级模型的 Harness 遵循率远高于弱层级模型。Opus 4.6 的 HFR 达到 0.757，而 Qwen3-32B 仅为 0.142。（ii）加载 Harness 并不足以从中获益。Qwen3-235B 在激活与遵循之间提供了最干净的分野：其技能加载率为 0.961，几乎与 Opus 4.6 相同，然而其 HFR 仅为 0.350。其加载通过率同样反映了这一差距：0.022，而 Opus 4.6 为 0.177。
:::

表 2 显示，Opus 4.6、Sonnet 4.6 和 Qwen3-235B 的技能加载率接近天花板（0.957–0.961），但 GPT-OSS-120B 降至 0.446，Qwen3-32B 更是降至 0.251。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/13.webp)

::: image-caption
图 7 Qwen3-32B 在 SkillsBench 上的两种 Harness 获益失效模式。左（`threejs`任务）：Harness 激活失败——无效的多键加载动作导致技能体从未进入上下文。右（`pg-essay-to-audiobook`任务）：Harness 遵循失败——技能虽已被加载，但智能体将其视作字面脚本，跳过了规定的 TTS 引擎回退链。
:::

图 7 左侧面板展示了这一激活失败。**具体来说，Qwen3-32B 识别到了相关技能，但将加载请求嵌入了一个更宽泛的动作中，而非将其作为独立的技能加载动作发出**。SkillsBench 环境因此不将其视为有效的加载请求，技能体从未进入上下文。

#### 第二种模式是Harness 遵循失败

> **即使相关 Harness 组件被加载了，弱层级模型也常常无法在任务求解过程中忠实地遵循其指导**。我们用量化指标:mdnice-bold-italic[Harness 遵循率（Harness-Following Rate, HFR）来衡量这一失效，该指标在至少加载了一个技能的轨迹上计算]。对于每条加载了技能的任务求解轨迹，由 LLM 裁决者判断任务求解模型是否遵循了已加载技能的指导。HFR 是技能已加载轨迹中被判定为遵循技能的占比。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/14.webp)

::: image-caption
表 2：SkillsBench 上各模型的激活、遵循与结果指标。SLR 为模型轨迹中加载了至少一个技能的轨迹占比；HFR 为加载后轨迹中被判定为遵循技能指导的占比；LPR 为加载后轨迹中的通过率。该表清晰展示了弱模型（Qwen3-32B）在激活（0.251）和遵循（0.142）两个维度上的双重不足。
:::

图 7 右侧面板的`pg-essay-to-audiobook`案例展示了这一遵循失败：Qwen3-32B 成功加载了程序性技能，但将指导视为现成脚本而非应遵循的流程。在第一次尝试失败后，它终止执行而非尝试技能所规定的替代步骤。

---

#### 诊断：长程执行中的弱指令遵循能力

> 为了检验 Harness 遵循度是否随轨迹展开而退化，我们进行了阶段级遵循度分析。

由 LLM 裁决者在不同执行阶段分配 0 到 1 之间的遵循度评分。我们使用 Qwen3-32B、GPT-OSS-120B 和 Opus 4.6 分别作为弱、中、强层级的代表模型。

![文章插图](/articles/2026-07-09-harness-harness-llm-aa7f9576/images/15.webp)

::: image-caption
表 3：代表性弱、中、强层级模型（Qwen3-32B、GPT-OSS-120B、Opus 4.6）的阶段级遵循度评分。粗体和下划线分别标记每个阶段中的最佳和最差分数。Qwen3-32B 从 Harness 加载后的 0.52 跌至最终验证时的 0.13，衰减幅度（-0.39）是 Opus 4.6（-0.09）的四倍以上。
:::

表 3 报告了三个代表性阶段的评分：Harness 加载后、轨迹中点、最终验证时，分数为各模型已裁决轨迹上的均值。:mdnice-bold-italic[我们观察到，Qwen3-32B 从 Harness 加载后的 0.52 急剧下降到最终验证时的 0.13，GPT-OSS-120B 从 0.67 较为温和地下降到 0.43。相比之下，Opus 4.6 保持稳定，从 0.89 到 0.80。 这种梯度的漂移表明存在长程指令遵循瓶颈：较弱模型随着轨迹展开逐渐丧失遵循度，而非仅仅在加载时误读了 Harness。]

#### 总结

> 智能体训练应沿两个轴向来针对性地提升 Harness 获益能力。

1. **将 Harness 调用能力内化到训练中**：弱层级模型的技能加载率低（Qwen3-32B 为 25.1%，而强层级模型约为 96%），:mdnice-bold-italic[智能体必须学会可靠地将相关 Harness 组件纳入上下文。]
2. **加强长程指令遵循能力**：即使在加载了 Harness 之后，弱层级模型在轨迹中会逐渐丧失遵循度（Qwen3-32B 从 0.52 漂移到 0.13），:mdnice-bold-italic[因此智能体必须学会在长程任务中持续遵循 Harness 指导。]

## 五、局限与结论

> 我们研究 Harness 自演化——一种模型权重冻结、仅通过更新外部 Harness 组件实现适应的机制。为深入理解其内在机制，我们将该能力解耦为两类模型能力：**Harness 更新能力** （产出有效更新的能力）与**Harness 获益能力** （在任务求解中切实从更新后的 Harness 中获益的能力）。

实验覆盖七个主流 LLM 与三个基准，揭示出关键差异：

- Harness 更新能力在基础能力维度上呈**扁平分布** ——不同层级模型（如 Qwen3.5-9B 与 Claude Opus 4.6）产出的更新带来相近增益，表明该能力对模型“智力”门槛要求较低；
- 而 Harness 获益能力则呈**非单调分布**：中等能力模型表现最优，弱模型获益最少，主因两类失效——**未能激活相关 Harness 组件**，或**激活后无法忠实遵循指令**。

值得注意的是，本研究严格限定于纯 Harness 自演化范式：不涉及参数微调、权重端强化学习，亦未混合 Harness 更新与权重更新。所选模型兼顾开源与闭源、覆盖能力梯度，具代表性但非穷尽；未来在更广模型网格（含不同架构、规模、训练配方及推理成本）上的扩展，有望进一步刻画两类能力的演化规律与部署边界。

::: block-1
这些发现指向务实的资源分配策略：:mdnice-highlight[算力预算应优先投向智能体（Agent）侧优化，而非 Evolver 侧升级]；同时，将:mdnice-highlight[Harness 调用能力（精准定位并触发组件）与长程指令遵循能力（跨步骤、跨组件保持意图一致性）确立为智能体训练的核心目标。]
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**
- **[异构智能体之间的“读心术”：稠密 KV 缓存高效通信实现 2-3 倍算力优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902336&idx=1&sn=d667567b5407191944b276e13a16ba7d&scene=21#wechat_redirect)**
- **[LLM负责思考，SLM负责记忆。LightMem小模型天团重塑 Agent 记忆，延迟仅 83 毫秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900501&idx=1&sn=2311904700b9bfd0c8da1f96e2e98a4b&scene=21#wechat_redirect)**

:::
