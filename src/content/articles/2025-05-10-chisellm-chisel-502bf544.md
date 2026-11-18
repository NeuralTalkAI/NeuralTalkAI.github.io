关键词：大型语言模型、特定领域架构、Chisel 语言、敏捷硬件开发

> **随着特定领域架构** （Domain-Specific Architecture, DSA）**需求的不断增长**，**敏捷硬件开发** 方法（Agile Hardware Development Methodology, AHDM）**得到了快速发展**。像 Chisel 这样的硬件构造语言（Hardware Construction Language, HCL）提供了高级抽象功能，使其成为基于 HCL 的 AHDM 的理想语言。

尽管大型语言模型（Large Language Models, LLMs）在代码生成任务中表现出色，但在**Chisel 代码生成方面仍面临挑战，尤其是在语法正确性和设计可变性方面**。最近的推理模型通过**测试时扩展（test-time scaling）技术显著增强了代码生成能力**。然而，我们发现没有经过领域适配的推理模型并不能为 Chisel 代码生成任务带来显著的好处。

::: block-1
**本文提出了 ChiseLLM，它包括数据处理和转换、提示引导的推理轨迹合成以及领域适配模型训练**。我们**从公共 RTL 代码资源中构建了高质量的数据集，并通过提示增强方法引导模型采用结构化思维模式**。

实验表明，我们的**ChiseLLM-7B 和 ChiseLLM-32B 模型在语法正确性方面分别比基础模型提高了 18.85%和 26.32%，在设计可变性方面比基线推理模型提高了 47.58%。** 我们的数据集和模型将公开提供，**为基于 HCL 的 AHDM 提供高性能、低成本的模型**，并为未来研究提供有效的基线。

- GitHub 仓库：https://github.com/observerw/ChiseLLM
:::

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/01.webp)

- https://arxiv.org/abs/2504.19144

## 本文目录

- 一、引言
- 二、相关工作
  - A. 硬件构造语言
  - B. 推理模型在代码生成中的应用
- 三、前置知识
  - A. 敏捷硬件开发与设计可变性
  - B. Chisel 生成任务
- 四、我们的方法
  - A. 原始数据集收集与预处理
  - B. 提示引导的推理轨迹蒸馏
  - C. 模型微调
- 五、实验
  - A. 研究问题
  - B. 实验设置
  - C. 正确性评估 (RQ1)
  - D. 可变性评估 (RQ2)
  - E. 消融研究 (RQ3)
- 六、VI. 案例研究
- 七. 结论
- 参考文献

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/02.webp)

## 一、引言

> 随着人工智能和大数据的快速发展，对特定领域架构（Domain-Specific Architecture, DSA）的市场需求显著增长。这一趋势推动了敏捷硬件开发方法（Agile Hardware Development Methodology, AHDM）的发展。

**敏捷硬件开发方法** （AHDM）**对缩短硬件设计周期和增强设计可变性提出了更高要求**，**设计可变性是指硬件设计适应需求或约束变化的能力**。为了满足这些要求，**下一代硬件构造语言** （Hardware Construction Language, HCL），**以 Chisel 为代表，将现代编程语言特性融入硬件设计**，显著增强了硬件设计的抽象表达能力。

因此，基于 HCL 的 AHDM 展现出光明的前景，并已在多个大型项目中得到应用。大型语言模型（LLMs）在代码生成方面表现出色，促使研究人员探索使用 LLMs 进行 AHDM 开发，包括自动化模块级设计和架构设计。**将 LLMs 应用于基于 HCL 的 AHDM 开发已受到初步研究关注，表明这种方法可以更有效地管理项目复杂性、降低迁移成本并增强敏捷方法的实用性。**

::: block-1
然而，我们观察到**现有的小规模开源模型在 Chisel 代码生成方面缺乏足够的能力**：模型在**生成 Chisel 代码时经常出现语法/语义错误**，并且**难以** 利用 Chisel 的语言特性进行有效的**可变性设计**。这一限制阻碍了生成代码在参数化模块生成或设计空间探索等任务中的潜力。这些挑战阻碍了 LLMs 在基于 HCL 的 AHDM 开发中的实际应用。最近的推理模型，如 OpenAI-o1 和 Deepseek-R1，引入了测试时扩展，通过在生成最终结果之前进行深思熟虑的思考和反思来进行推理。
:::

**相关研究表明，推理模型比缺乏思考过程的非推理模型能更有效地生成高质量代码**。因此，我们认为引入测试时扩展是增强模型在 Chisel 代码生成任务中性能的有希望的方法。

> 然而，我们**的实验表明，没有经过领域适配的推理模型并不能在 Chisel 代码生成场景中实现显著的额外好处**，因为通用推理模型**缺乏可靠的硬件逻辑思维模式**，导致输出容易出现幻觉。

因此，使推理模型适应以特定任务的思维模式进行推理成为关键挑战。为解决这一挑战，**我们提出了 ChiseLLM 系列数据集和模型**。我们的**方法包括**：

1. **原始数据集处理和转换**：我们收集了来自公共 RTL 代码源的 Chisel 和 Verilog 代码，经过处理和转换后形成了两个高质量的领域特定指令微调数据集；
2. **提示引导的推理轨迹数据合成**：我们采用提示引导的方法构建了具有特定思维模式的推理数据集 ChiseLLM{Completion,Decompile}，使模型能够学习如何以这些特定的思维模式进行推理；
3. **推理模型训练**：基于生成的数据，我们对 Qwen2.5-Coder 系列进行了微调，产生了 ChiseLLM-{7,32}B 模型。

::: block-1
实验结果表明，ChiseLLM 成功地使推理模型适应，实现了卓越的 Chisel 代码生成能力。

- **在语法/语义正确性方面**，ChiseLLM-7B 模型平均比预微调基础模型提高了 18.85%的语法正确性，而 ChiseLLM-32B 模型在 Pass@5 方面平均比基础模型提高了 26.32%，在所有任务中平均比最优基线提高了 20.44%，在语法正确性和反编译准确性方面实现了与商业模型相当的性能。
- **在可变性能力方面**，我们的模型比基线推理模型提高了 47.58%，**表明 ChiseLLM 模型能够有效地生成具有高可变性的设计**。
:::

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/03.webp)

我们的主要贡献如下：

1. 我们**引入了 ChiseLLM 系列模型和数据集，成功地将开源 LLMs 在 Chisel 代码生成任务中的性能提升到实际有效的水平**，并在某些任务中实现了**与商业模型相当的性能**，同时将模型大小控制在 32B 以下，以降低计算成本并提供更广泛的应用性。
2. 我们的**提示引导方法成功地使推理模型适应特定领域的使用，为 LLM 辅助硬件设计引入了新方法**。我们构建了一个标准化的评估系统，全面评估主流代码生成模型在 Chisel 任务上的性能。
3. 我们的**模型、数据集和构建过程将开源** 供后续研究人员使用和改进。我们希望 ChiseLLM 能作为基于 HCL 的 AHDM 领域的基础模型，推动相关研究的发展。

## 二、相关工作

### A. 硬件构造语言

为了满足**敏捷硬件开发的主要目标** ——**增强设计灵活性和缩短开发周期**，人们探索了多种方法，特别是“硬件生成器”的概念，它可以自动生成满足特定需求的设计。在硬件构造语言（HCL）出现之前，例如 Genesis2 项目尝试通过将硬件代码视为模板，并通过嵌入式语言自动生成硬件设计来满足这一需求。

::: block-1
这种方法的演变促成了硬件构造语言（HCL）的发展，它将生成逻辑集成到高级编程语言中。HCL 被定义为在高级语言之上设计的领域特定语言（Domain Specific Languages, DSLs），例如 Chisel 基于 Scala，PyRTL 基于 Python。**HCL 的最初设计意图是标准化和形式化这些设计生成器**，**利用宿主语言** 的表达能力，根据一组高级设计参数和约束**自动生成设计**，从而无需为每类设计规范重新实现设计。
:::

> 这种方法减少了从头开发新语言的成本，同时将软件工程中的重要概念和技术引入硬件设计。

目前，Chisel 语言已被应用于多个大规模敏捷硬件开发项目，包括 RocketChip、XiangShan、Berkeley BOOM 处理器等。

### B. 推理模型在代码生成中的应用

**测试时扩展** （Test-Time Scaling）技术指的**是对大型语言模型进行后训练修改**，使其能够**在训练后进行扩展的思考过程（称为推理轨迹）**。具备这种能力的模型被称为推理模型，包括 OpenAI-o 系列、Deepseek-R1、Claude-3.7 Sonnet、Gemini 2.5 等。推理模型在代码生成任务中表现出色，成为该领域的一个重要研究方向。

> 然而，**LLM 在特定领域语言（DSLs）中的表现仍然受到限制，因为这些语言的语料库稀缺**，这已被**广泛认为需要领域适配**。

**我们的工作证实了这一结论**，表明**推理模型仍然需要领域适配，才能在 Chisel 代码生成任务中实现最佳性能**。

## 三、前置知识

### A. 敏捷硬件开发与设计可变性

敏捷硬件开发方法（AHDM）借鉴了软件敏捷开发的概念，旨在通过快速迭代和持续反馈增强硬件设计的灵活性和响应能力。因此，采用 AHDM 的项目应该具有高度的可变性，即项目适应市场需求或环境条件变化的能力。

在**硬件领域，可变性主要强调硬件设计对需求或约束变化的适应性，从而支持快速迭代的需求**。本文采用以下直接推断：**高度可变的硬件设计应该能够以最小的代码重构成本满足不同的设计需求变化**。例如，可变性设计可以通过仅修改一组高级参数，而无需对底层实现进行大量修改，来满足不同的功能和性能约束，如在[8]中所展示的。

### B. Chisel 生成任务

**当前使用 LLMs 进行 AHDM 开发的研究主要集中在将设计规范转换为传统硬件描述语言（HDL）代码**。然而，**Chisel 作为设计规范和传统 HDL 之间的中间层语言，需要考虑两种不同的场景进行代码生成**：

- 直接从设计规范生成 Chisel 代码（Spec-to-Chisel）
- 将现有的传统 HDL 代码转换为等效的 Chisel 代码（Decompile-to-Chisel）

这种分类反映了**两种实际工程需求**：**开发新设计** 和**升级现有设计**。表 I 总结了这两种任务类型。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/04.webp)

## 四、我们的方法

在本节中，我们将介绍 ChiseLLM 的**完整数据构建** 和**模型训练** 流程，包括：

- 原始数据收集与预处理（第 A 节）
- 提示引导的推理轨迹蒸馏（第 B 节）
- 模型微调（第 C 节）

### A. 原始数据集收集与预处理

> 公开可用的硬件描述语言（HDL）代码数据极为稀缺。（System）Verilog 在 the-stackv2-dedup 数据集中仅占 3.2×10⁻²%，而像 Chisel 这样的硬件构造语言（HCL）的占比更低（大约 1×10⁻³%）。

据我们所知，**目前没有公开的、专门针对 Chisel 语言的大规模训练数据集**。为解决这一限制，我们**利用 Chisel 作为中间层语言的特性，同时使用 Chisel 和 Verilog 数据来扩充数据集的规模和多样性。**

我们收集了公开可获取的 Chisel 和 Verilog 数据集，分别进行处理和转换，最终形成了两个高质量的基础数据集：ChiseLLM-{Completion,Decompile}Base。这两个数据集分别包含 Chisel 代码补全和代码反编译任务。

ChiseLLM-Completion-Base 数据集源自 the-stackv2-dedup 数据集，这是一个包含多种编程语言的大型预训练数据集。我们从该数据集中提取了 Chisel3 代码文件，排除了包含 chiseltest 或 scalatest 的文件，以去除所有测试代码。这种过滤是必要的，因为 Chisel 语言测试框架的不稳定性以及它与我们任务目标的不一致性。

**经过过滤和长度筛选后，我们获得了大约 8.3K 个 Chisel 代码文件**。随后，我们**利用大型语言模型为代码添加详细注释，并将完整的代码转换为包含设计规范和上下文信息的问题-答案对**，用于代码补全任务。

### B. 提示引导的推理轨迹蒸馏

我们**遵循 DeepseekR1 论文中提出的方法，使用 Deepseek 为指令微调数据集生成推理轨迹**，从而创建推理数据集以**训练较小的模型以获得推理能力**。这个过程被称为蒸馏。

> 相关研究表明，大型语言模型在**微调时主要学习如何有效地利用预训练期间获得的知识**，**而不是学习新知识**。基于这一见解，我们专注于构建具有特定“思维模式”的推理数据。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/05.webp)

**如图 2(b) 所示，为了获得具有所需思维模式的推理数据，我们采用提示引导的方法**，为 Deepseek-R1 提供额外的任务相关提示和引导信息，**引导模型生成更符合任务要求的推理轨迹**。我们称此过程为提示引导的蒸馏。

对于 Spec-to-Chisel 和 Decompile-to-Chisel 任务，我们**分别采用了不同的提示引导方法，以适应每个任务目标**。通过学习这些数据，**模型在面对特定任务时能够采用适当的思维模式，展现出比通用思维模式更强的问题解决能力**。

通过这种方法，我们处理了 ChiseLLM-{Completion,Decompile}-Base 数据集，最终创建了两个高质量的推理数据集：ChiseLLM-{Completion,Decompile}。

#### 1）Spec-to-Chisel 任务

> Spec-to-Chisel 任务的目标是将设计规范转换为 Chisel 代码。

我们观察到，**当生成代码时，模型通常只推理功能实现，而缺乏对语法的反思**。这导致了更多的幻觉现象，包括引用不存在的 Chisel API 或错误地使用其他语言的语法。

> 为了使模型能够在推理轨迹中合理地利用 Chisel 语言特性，我们提供了两种类型的提示引导：文档引用和基准答案。

::: block-1
预期的思维模式是：在推理过程中，模型首先形成功能实现的逻辑步骤，然后在语法层面进行反思（通过“回忆”文档），最终生成语法和功能都正确的答案。
:::

##### 文档引用

我们从 Chisel 官方网站下载了所有语言文档，并将其处理成可独立引用的 Markdown 文档片段。我们为官方文档创建了一个索引文档，包括所有文档片段的标题和内容摘要，并为每个片段分配了一个唯一的章节 ID。随后，

我们将此文档作为上下文，让 Qwen2.572B-Instruct 以注释的形式在代码行级别进行标注。之后，根据章节 ID 匹配文档片段，并将其作为相应代码数据的部分信息。经过处理后，每段代码都对应几个相关的文档，平均在 5-10 个之间。

我们**将这些文档作为上下文信息的一部分纳入蒸馏过程**，要求模型**在反思语法时利用这些文档**。

##### 基准答案

目前构建推理数据集的常见方法是对独立可验证的高质量代码数据集进行拒绝采样。然而，现有的 Chisel 代码数据以预训练数据集的形式存在，缺乏高质量的参考答案和测试用例。由于缺少单元测试用例以及编译的困难，生成的 Chisel 代码很难判断其正确性。为解决这一问题，我们将基准答案作为引导的一部分，允许推理模型在给出参考答案时生成中间推理轨迹。这种方法防止了模型的推理轨迹导致错误的答案。

#### 2）Decompile-to-Chisel 任务

Decompile-to-Chisel 任务的目标是将低级硬件描述语言（HDL）代码转换为 Chisel 代码。现有的工作可以程序化地将 Verilog 代码转换为 Chisel 代码，但它只能实现代码结构的非优化映射，映射后的优化仍然需要人工完成。

> 我们认为 LLM 在这个任务中具有更大的潜力，能够通过理解和推理代码来实现高质量的转换和优化。为此，我们提供了两种类型的提示引导：**变体模式规范** 和 **Chisel 特性描述**。
>
> ::: block-1
> 我们的预期思维模式是：在推理过程中，模型能够理解原始代码的核心功能，通过将变体模式应用于原始代码来推断潜在的设计需求变化，并利用 Chisel 语言特性重构代码以涵盖所有变体。
> :::

##### 变体模式规范

基于可变性设计的特性（见第三节），我们要求模型首先考虑该模块可能面临的需求变化，然后相应地优化代码，以降低适应每种变化的重构成本。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/06.webp)

**为了引导模型产生合理的变体设计，我们将变体的形式限制为以下三种类型**：

::: block-1
1. **可配置变体：修改版本，将可调整属性** （如数据宽度、缓存深度）提**取为可配置参数，而不改变核心功能或接口**。这种类型的变体在**设计空间探索** 中非常常见，通常**用于在不同设计需求下进行性能优化**。
2. **功能变体：修改版本，改变输入输出行为** （如协议版本支持、异常处理机制、算法实现），**直接修改设计的功能需求**。这种类型的变体通常出现在**早期设计阶段的快速原型设计中，或后期的功能扩展** 中。
3. **结构变体：修改版本，在保持功能等价的同时调整硬件架构** （如流水线阶段、存储器组织、有限状态机编码方案），**影响时序/面积**。这种类型的变体通常出现在**功能需求确定后**，**进行功耗、性能、面积（PPA）优化时**。
:::

我们要求模型提供可配置变体，并且只提供功能变体或结构变体中的一个，以避免引导模型产生过于复杂且不必要的设计。随后，我们要求模型设计一个能够涵盖所有变体的 Chisel 模块。

##### Chisel 特性描述

**为了使模型在推理时合理地利用 Chisel 语言特性，我们提供了 Chisel 语言特性及设计模式的简要描述**，包括参数化、函数组件和函数式编程、组合逻辑生成、通用参数、可选接口、继承和特质。

我们**要求模型在生成推理轨迹时，根据所选的变体模式和问题本身的特性，选择合适的语言特性**。例如，在将一个向上计数器模块转换为向下计数器模块的功能变体时，模型应首先将计数器行为抽象为抽象类 CounterBase，然后通过实现抽象接口来实现相应功能。

### C. 模型微调

我们使用 LLaMA-Factory 进行模型的全参数微调。我们首先分析了 ChiseLLM-Completion 和 ChiseLLM-Decompile **数据集的数据特性，如图 2(c) 所示**。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/05.webp)

反编译数据集的标记长度显著长于补全数据集，平均约为 9K 个标记。鉴于两种任务类型的难度特征，这种统计分布是符合预期的。因此，我们**按照比例混合数据，采样比例为 3:7，并在 Qwen2.5-7B-Instruct 和 Qwen2.5-32B-Instruct 上进行训练，最终获得了一系列具有推理能力的模型，称为 ChiseLLM 模型**。所有模型都将公开提供。

## 五、实验

在本节中，我们**对 ChiseLLM 模型进行全面评估**。我们**评估了 ChiseLLM 模型和基线模型** 在 Spec-to-Chisel 和 Decompile-to-Chisel 任务上的**性能**，并**分析了模型在可变性设计方面的能力**。我们还进行了消融研究，以分析 ChiseLLM **数据集对模型性能的影响**。

### A. 研究问题

我们的实验关注以下研究问题：

- RQ1：**ChiseLLM 模型** 在 Spec-to-Chisel 和 Decompile-to-Chisel 任务上的表现**与基线模型相比如何**？
- RQ2：ChiseLLM 模型**是否能更有效地产生可变性设计**？
- RQ3：ChiseLLM **数据集对模型性能的影响** 是什么？

### B. 实验设置

- 基准测试：据我们所知，目前没有专门用于 Chisel 的公开评估数据集。然而，由于 Chisel 可以生成可综合的 Verilog 代码，我们利用广泛使用的 Verilog 评估数据集进行评估。我们**使用 RTLLM v2.0 和 VerilogEval-Human 数据集作为评估语法/语义正确性的基准**。
- **正确性评估方法**：为了衡量生成的 Chisel 代码的语法和语义正确性，我们**采用常见的 Pass@k 指标**。

我们首先使用 Scala 2.13 编译器编译生成的 Chisel 代码，检查其语法正确性；然后使用 Chisel3 的 `ChiselStage.emitSystemVerilog` API 将生成的 Chisel 代码转换为等效的 SystemVerilog 代码。在大多数情况下，通过此 API 生成的 SystemVerilog 代码在语法上是正确的且可综合的。最后，我们使用商业工具（因为 Chisel 自动生成的 SystemVerilog 语法无法被 iVerilog 编译）编译并仿真代码，检查其功能正确性。

- **可变性评估方法**：为了评估 Chisel 代码的可变性，我们采用主流的 LLM-as-a-Judge 方法，这种方法被认为在代码质量评估中接近人类评估。为了确保评估结果的一致性和公平性，我们：

所有评估数据集都是公开可访问的。Judge LLM 对同一段 Chisel 代码进行多次评估。最终评估分数是所有评估分数的平均值。对于评估方差较高的样本，不将其纳入评估。类似于 Pass@k 指标，我们要求模型多次实现相同的设计需求，并对每次实现进行评估，所有实现的平均分数作为评估结果。

- **基线模型**：我们选择具有验证代码生成性能的模型作为基线，包括：

### C. 正确性评估 (RQ1)

**表 II 显示了各种模型在 VerilogEval 和 RTLLM 数据集上的 Spec-to-Chisel 和 Decompile-to-Chisel 任务的性能**。

> 我们观察到以下事实：**通用推理模型并不总是优于其基础模型**。

例如，在 RTLLM 数据集的 Spec-to-Chisel 任务中，Deepseek-R1-Distill-Qwen-32B 的 pass@1 仅为 14.86%，而其基础模型 Qwen2.5-Coder-32B-Instruct 达到 26.50%；同样，Deepseek-R1-Distill-Llama-8B 在 VerilogEval 上的 pass@1 为 9.31%，低于其基础模型 Llama3.1-8B-Instruct。

> 这些观察表明，推理模型相对于其基础模型并没有一致且显著的改进。ChiseLLM 模型在所有任务中均优于类似大小的基线模型，甚至在某些任务中超过了商业模型。

以 RTLLM 的 Decompile-to-Chisel 任务为例，ChiseLLM-32B 的 pass@1 达到 54.32%，显著高于 Qwen2.5-Coder-32B-Instruct 的 13.00%，甚至超过了 GPT-4o 的 44.50%；ChiseLLM-7B 在 VerilogEval 数据集上的 pass@1 达到 50.47%，几乎是 Qwen2.5-Coder-7B-Instruct（27.60%）的两倍。

值得注意的是，ChiseLLM-32B 在所有指标中始终排名首位或第二，展现出与 Deepseek-R1 相当的性能，尽管其参数规模显著较小（32B 对比 671B）。

### D. 可变性评估 (RQ2)

下面**图 4 显示了不同模型在执行 Decompile-to-Chisel 任务时对 RTLLM 数据集的可变性设计能力**。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/07.webp)

> 我们观察到以下事实：ChiseLLM 模型在可变性设计方面优于基线模型，并且与商业模型相当。

ChiseLLM-32B 的可变性得分显著高于非商业基线模型，如 Qwen-2.5-Coder-32B（3.79）和 LLaMA-3.3-70B（2.39），甚至超过了商业模型，如 Deepseek-R1（5.03）和 Deepseek-V3（4.84）。

**这表明 ChiseLLM-32B 通过特定领域的微调获得了卓越的可变性设计能力**。**推理模型在可变性设计任务中并没有显著优势**。尽管 Deepseek-R1-Distill 系列模型在某些代码生成任务中表现出色，但其在 Chisel 可变性设计中的表现有限，Deepseek-R1-Distill-32B 和 Deepseek-R1-Distill-70B 的平均值仅为 3.82 和 3.20。

这些值与它们对应的基线模型 Qwen-2.5-Coder-32B（3.79）和 LLaMA-3.3-70B（2.39）相比提升有限。

> **表明通用推理能力并不总是与特定领域的可变性设计能力正相关。更好的可变性设计能力伴随着更大的不确定性。**

尽管 ChiseLLM-32B 的平均值最高（5.19），但其方差也最大；同样，Deepseek-R1 和 Deepseek-V3 也表现出高平均值和大方差。这表明，当模型尝试实现更灵活的设计时，可能会引入更多的不确定性。较小的模型，如 LLaMA-3.3-70B，方差较小（1.23），但可变性设计能力也最弱（平均值仅为 2.39）。

**总体而言，较高的平均值表示更强的可变性设计能力，但设计的多样性也可能带来更大的性能波动。**

### E. 消融研究 (RQ3)

下面的**表 III 显示了我们的消融研究结果**。为了展示 ChiseLLM **数据集对模型性能的影响，我们分别使用这两个数据集对 ChiseLLM-32B 模型进行微调**，得到了 ChiseLLM-32B-Completion 和 ChiseLLM-32B-Decompile 模型，并对其性能进行了评估。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/08.webp)

> 我们观察到以下事实：两个数据集单独使用都能提升模型性能。ChiseLLM-{Completion,Decompile} 数据集各自对模型性能有积极影响。

- 在 VerilogEval 的 Decompile-to-Chisel 任务中，ChiseLLM-32B-Completion 在 pass@1 上提升了 2.88 个百分点，而 ChiseLLM-32B-Decompile 提升了 14.78 个百分点；
- 在 RTLLM 的 Decompile-to-Chisel 任务中，ChiseLLM-32B-Completion 在 pass@1 上提升了 23.33 个百分点，ChiseLLM-32B-Decompile 提升了 37.56 个百分点。

两个数据集的结合产生了显著的协同效应。在所有任务中，使用两个数据集训练的 ChiseLLM-32B。

## 六、VI. 案例研究

在本节中，我们**通过两个案例研究来展示和比较经过微调的 ChiseLLM-32B 模型的设计可变性能力**。本节的目的是通过提供直观的示例来**补充第 五-D 节中呈现的结果**，这些示例既展示了设计过程中的思考过程，也**展示了 ChiseLLM 模型产生的最终设计的独特特征**。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/09.webp)

**图 6 展示了一个实际的例子，ChiseLLM-32B 对 Verilog 源代码进行反编译**。在这个例子中，原始的 Verilog 代码实现了一个具有固定数据宽度和移位模式的简单移位模块。在反编译过程中，ChiseLLM 考虑了配置和结构设计变体，通过引入宽度配置参数和抽象的 ShifterBase 类。这些增强将模块转换为一个具有高可变性和可扩展性的 Chisel 实现，具有可配置的数据（可变宽度）和结构（可定制的移位行为）。

![文章插图](/articles/2025-05-10-chisellm-chisel-502bf544/images/10.webp)

尽管由于篇幅限制，这里展示的代码相对简单，但 ChiseLLM 模型可以将相同的逻辑方法应用于实际应用中的更复杂实例。**图 5 展示了一个比较示例，其中 ChiseLLM 和 Qwen2.5-32B-Coder-Instruct 分别对相同的 Verilog 源代码进行反编译**。在这个案例中，名为 sync_signal 的 Verilog 模块旨在将输入数据延迟多个时钟周期后输出。由 Qwen 模型生成的 Chisel 实现包含了一些优化，但仅考虑了通过参数化数据宽度和同步周期的配置变体。

相比之下，**ChiseLLM 遵循一个结构化的推理过程**，首先分析模块的核心功能，然后识别当前实现的关键缺陷——固定的数据结构和缺乏流控制，并通过引入适当的 Chisel 语言特性来解决这些问题。**ChiseLLM 生成的最终 Chisel 模块能够处理任意复杂类型的数据输入，并通过使能信号控制数据流传输，从而展现出更大的可变性和功能可扩展性**。

## 七. 结论

在本文中，我们**提出了 ChiseLLM**，它**包括一个高质量的推理数据集和一个推理模型**。我们从公共资源中收集了训练数据，并**通过提示引导的蒸馏增强了 LLM 在硬件构造语言（HCL）生成中的性能**。

::: block-1
我们的工作**为未来在 LLM 辅助的基于 HCL 的敏捷硬件开发（AHDM）领域提供了两个有前景的视角**：首先，**高质量的领域适配训练语料库可以显著提高 LLM 在 HCL 生成中的性能**；其次，通过额外的引导，**LLM 可以掌握解决硬件设计复杂性所需的“思维模式”**。
:::

作为 LLM 辅助的基于 HCL 的 AHDM 的初步探索，我们希望吸引更多研究人员关注这一领域，并推动 LLM 在硬件设计中的应用。

## 参考文献

[1] Y. Lee and A. e. a. Waterman, “An agile approach to building risc-v microprocessors,” IEEE Micro, vol. 36, no. 2, pp. 8–20, 2016.

[2] R. Bahr and C. e. a. Barrett, “Creating an agile hardware design flow,” in 2020 57th ACM/IEEE Design Automation Conference (DAC), 2020, pp. 1–6.

[3] M. Acher and J. G. e. a. Duarte, “On programming variability with large language model-based assistant,” in Proceedings of the 27th ACM International Systems and Software Product Line Conference - Volume A. Tokyo Japan: ACM, Aug. 2023, pp. 8–14.

[4] C. Brink and E. e. a. Kamsties, “On hardware variability and the relation to software variability,” in 2014 40th EUROMICRO Conference on Software Engineering and Advanced Applications, 2014, pp. 352–355.

[5] J. Bachrach and H. e. a. Vo, “Chisel: Constructing hardware in a scala embedded language,” in Proceedings of the 49th Annual Design Automation Conference. San Francisco California: ACM, Jun. 2012, pp. 1216–1225.

[6] C. Papon and Y. Xiao, “SpinalHDL,” Apr. 2025.

[7] Y. Xu and Z. e. a. Yu, “Towards Developing High Performance RISCV Processors Using Agile Methodology,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2022, pp. 1178–1199.

[8] “Chipsalliance/rocket-chip,” CHIPS Alliance, Apr. 2025.

[9] N. Jain and K. e. a. Han, “Livecodebench: Holistic and contamination free evaluation of large language models for code,” arXiv preprint arXiv:2403.07974, 2024.

[10] N. Pinckney and C. e. a. Batten, “Revisiting VerilogEval: A year of improvements in large-language models for hardware code generation,” Feb. 2025.

[11] S. Liu and W. e. a. Fang, “Rtlcoder: Outperforming gpt-3.5 in design rtl generation with our open-source dataset and lightweight solution,” in 2024 IEEE International Workshop on LLM-Aided Design. IEEE, 2024.

[12] K. Chang and Y. W. et al., “Chipgpt: How far are we from natural language hardware design,” 2023. [Online]. Available: https: //arxiv.org/abs/2305.14019

[13] J. Niu and X. e. a. Liu, “Rechisel: Effective automatic chisel code generation by llm with reflection,” in Proceedings of the 62nd ACM/IEEE Design Automation Conference (DAC). San Francisco, CA, USA: ACM/IEEE, Jun. 2025.

[14] T. Liu and Q. e. a. Tian, “ChatChisel: Enabling Agile Hardware Design with Large Language Models,” in 2024 2nd International Symposium of Electronics Design Automation (ISEDA). Xi’an, China: IEEE, May 2024, pp. 710–716.

[15] “Introducing OpenAI o1,” https://openai.com/o1/, Sep. 2024.

[16] DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” Jan. 2025.

[17] C. Snell and J. e. a. Lee, “Scaling llm test-time compute optimally can be more effective than scaling model parameters,” arXiv preprint arXiv:2408.03314, 2024.

[18] S. Welleck and A. e. a. Bertsch, “From decoding to meta-generation: Inference-time algorithms for large language models,” arXiv preprint arXiv:2406.16838, 2024.

[19] “O1 tops aider’s new polyglot leaderboard,” https://aider.chat/2024/12/21/polyglot.html, Dec. 2024.

[20] D. Li and S. C. et al., “S*: Test time scaling for code generation,” 2025. [Online]. Available: https://arxiv.org/abs/2502.14382

[21] B. Hui and J. e. a. Yang, “Qwen2.5-coder technical report,” arXiv preprint arXiv:2409.12186, 2024.

[22] A. Solomatnikov and A. e. a. Firoozshahian, “Chip multi-processor generator,” in Proceedings of the 44th Annual Design Automation Conference, ser. DAC ’07. New York, NY, USA: Association for Computing Machinery, 2007, p. 262–263. [Online]. Available: https://doi.org/10.1145/1278480.1278544

[23] J. Clow and G. e. a. Tzimpragos, “A pythonic approach for rapid hardware prototyping and instrumentation,” in 2017 27th International Conference on Field Programmable Logic and Applications (FPL). IEEE, 2017, pp. 1–7.

[24] “Motivation of chisel,” https://www.chisellang.org/docs/explanations/motivation.

[25] “Riscv-boom/riscv-boom: SonicBOOM: The berkeley out-of-order machine,” https://github.com/riscv-boom/riscv-boom.

[26] K. Kumar and T. e. a. Ashraf, “Llm post-training: A deep dive into reasoning large language models,” arXiv preprint arXiv:2502.21321, 2025.

[27] J. Wei and X. e. a. Wang, “Chain-of-thought prompting elicits reasoning in large language models,” 2023.

[28] “Introducing OpenAI o3 and o4-mini,” https://openai.com/index/introducing-o3-and-o4-mini/.

[29] “Claude 3.7 sonnet and claude code,” https://www.anthropic.com/news/claude-3-7-sonnet.

[30] “Gemini 2.5: Our most intelligent AI model,” https://blog.google/technology/google-deepmind/gemini-modelthinking-updates-march-2025/, Mar. 2025.

[31] “Introducing SWE-bench verified,” https://openai.com/index/introducingswe-bench-verified/.

[32] X. Gu and M. e. a. Chen, “On the effectiveness of large language models in domain-specific code generation,” ACM Trans. Softw. Eng. Methodol., vol. 34, no. 3, Feb. 2025. [Online]. Available: https://doi.org/10.1145/3697012

[33] S. Joel and J. J. e. a. Wu, “A survey on LLM-based code generation for low-resource and domain-specific programming languages,” Nov. 2024.

[34] “Manifesto for agile software development,” https://agilemanifesto.org/.

[35] V. Cortellessa and D. e. a. Varr´o, Eds., Fundamental Approaches to Software Engineering: 16th International Conference, FASE 2013, Held as Part of the European Joint Conferences on Theory and Practice of Software, ETAPS 2013, Rome, Italy, March 16-24, 2013. Proceedings, ser. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, vol. 7793.

[36] J. Bruant and P.-H. H. et al., “(system)verilog to chisel translation for faster hardware design,” in Proceedings of the 31th International Workshop on
