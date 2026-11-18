关键词：LLM、RTL 代码自动生成、Verilog、数据集。

继昨天《[ChiseLLM：推理大模型加速 Chisel 特定领域架构的敏捷硬件开发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888214&idx=1&sn=fbc5d29d58edbafe9731fc5b15113433&scene=21#wechat_redirect)》，今天分享一篇关于基于 LLM 实现 RTL 代码自动生成的文章。

> 最近大型语言模型的进展，激发了人们将它们**应用于电子设计自动化** （EDA，即利用软件工具辅助设计电子系统，比如芯片设计）任务的兴趣，**特别是在寄存器传输级（RTL**，这是数字电路设计中的一种抽象层次，介于高层算法和底层门电路之间，用于描述数据在寄存器之间的传输和处理）**代码生成方面**。

虽然已经引入了一些 **RTL 数据集，但大多数只关注语法有效性** （即代码能否被正确编译），而**没有通过测试来验证功能正确性** （即代码是否实现了预期的功能行为），这导致训练样本**虽然能编译通过，但可能无法实现预期的功能**。

::: block-1
我们**提出了 VERICODER**，这是一个**在功能正确性验证的数据集上进行微调的 RTL 代码生成模型**。我们通过一种**新方法构建了这个微调数据集**，该方法**结合了单元测试生成和反馈引导的细化** （一种根据反馈逐步改进代码的方法）。

**给定一个自然语言规范** （即用日常语言描述的硬件设计需求）和一个**初始的 RTL 设计**，我们**提示一个教师模型** （GPT-4o-mini，这是一个功能强大的语言模型版本）**生成单元测试，并根据模拟结果** （即运行测试用例来验证代码功能的过程）**迭代地修订 RTL 设计**。如有必要，教师模型还会更新测试，以确保它们符合自然语言规范。
:::

通过这个过程，我们数据集中的每个示例都经过了功能验证，包括一个自然语言描述、一个 RTL 实现和通过的测试。在超过 125，000 个示例的这个数据集上进行微调后，VERICODER 在 VerilogEval 和 RTLLM 基准测试（这是两个用于评估 RTL 代码生成模型性能的标准测试套件）中，在**功能正确性方面取得了最先进的指标，在 VerilogEval 上的相对增益高达 71.7%，在 RTLLM 上高达 27.4%。**

消融研究进一步表明，在我们功能验证数据集上训练的模型，比在未功能验证数据集上训练的模型表现更好，强调了**在 RTL 代码生成中高质量数据集的重要性**。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/01.webp)

- https://arxiv.org/abs/2504.15659
- Standford、CMU、UIUC、Intel、 Visa Research

## 本文目录

- 一、引言
- 二、背景与相关工作
  - A. 语言建模与微调
  - B. RTL 代码数据集
- 三、我们的方法
  - A. 概述
  - B. 动机示例
  - C. 算法和提示
- 四、实验设置
  - A. 数据集
  - B. LoRA 微调设置
  - C. 基准测试和指标
  - D. 评估模型
- 五、结果
  - A. 主要评估结果
  - B. 数据集消融研究
  - C. 非验证数据集的测试通过率
- 六、讨论与未来工作
- 七、结论
- 参考文献

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/02.webp)

## 一、引言

> 大型语言模型（LLMs）在自然语言处理任务中展现出了卓越的性能，激发了人们将它们的能力应用于广泛的电子设计自动化（EDA）问题的兴趣[1]–[4]。

最近的努力探索了 LLMs 在代码生成[5]–[12]、架构设计[13]–[15]、验证[16]，[17]、工具辅助[18]，[19]和调试[1]，[20]等方面的应用。

::: block-1
在**本工作中**，我们**专注于从自然语言规范生成寄存器传输级（RTL）代码**。自动化 RTL 代码生成有潜力显著**提升硬件设计效率**，**减少复杂设计** 任务中的**人工工作量**，使其成为一个**及时且影响深远的研究领域**。
:::

**开发开源的、轻量级的 RTL 代码生成模型对于推进研究和部署至关重要**。像 GPT-4o 和 Claude 3.7 这样的专有模型限制了定制化，并且缺乏透明度，因此不适合深入分析和学术探索。此外，当处理可能**包含敏感知识产权的 RTL 设计时，它们还会引发隐私和安全问题**。

相比之下，可以在本地运行的轻量级模型提供了一种安全、保护隐私的替代方案，使硬件工程师能够将人工智能直接集成到他们的设计工作流中。

> 然而，现有的开源模型在 RTL 任务上的表现仍然不佳，这主要是由于它们的训练语料库中缺乏高质量、功能验证的 RTL 数据集[21]，[22]。

尽管训练算法 readily available，但进展受到缺乏经过功能正确性验证的开源数据集的限制。**构建这样的数据集的一个关键挑战在于构建大规模、高质量的训练数据，将自然语言规范与 RTL 实现配对**。尽管有努力从开源存储库中挖掘 RTL 代码[23]–[26]，但收集到的**大部分数据缺乏验证**，可能与预期功能不一致。

**为了解决这个问题，最近的研究转向了 LLMs** ——要么提示它们**根据基于关键字的规范合成 RTL 设计** [6]，[7]，要么**利用它们重写现有的 RTL 代码并生成匹配的规范** [8]，[24]，[26]。在两种情况下，通常使用语法检查器来过滤无法编译的代码或提供反馈以进行迭代细化，但这些技术仍然无法验证功能正确性。

> 据我们所知，所有这些先前的工作[6]–[8]，[24]，[26]都只关注确保语法正确性，忽略了功能正确性。因此，许多数据集示例虽然编译成功，但可能无法实现其自然语言规范中描述的行为。

语法正确性与功能正确性之间的区别对于模型评估和实际部署具有重要意义。虽然功能正确的代码本质上满足语法约束，但语法正确性本身并不能保证正确的功能。这一差距在 RTLLM 基准测试[10]报告的结果中显而易见，GPT-4o 在语法准确性上达到了 100.0%的高分，但在功能正确性上只有 69.0%。最终，在实际场景中，真正重要的是功能正确性，而不是语法有效性。

::: block-1
**在本工作中，我们引入了 VeriCoder，这是一个在经过功能正确性验证的高质量数据集上进行微调的 RTL 代码生成模型**，该数据集包含 125,777 个示例。

**为了构建这个数据集，我们开发了一种新流程，将单元测试生成与反馈引导的细化相结合，由教师 LLM（GPT-4o-mini）指导**。给定一个自然语言规范和一个初始的 RTL 实现，教师模型首先生成一个单元测试。如果 RTL 代码在模拟中失败，模型将根据观察到的错误消息迭代地修订设计。如有必要，单元测试也会被更新，以更好地反映规范所描述的预期功能。

**这个过程会一直持续到设计通过模拟或者达到重试限制为止。** 由此产生的微调数据集包含超过 125k 个经过验证的三元组：自然语言规范、正确的 RTL 设计和自检单元测试。我们从 Qwen2.5-14B-Instruct 开始对 VeriCoder 进行微调，并在两个已建立的 RTL 代码生成基准测试：VerilogEval [9]和 RTLLM [10]上对其进行评估。VeriCoder 在这些基准测试中取得了新的最先进的性能，在 pass@k 指标上相对于之前最佳的微调模型 OriGen [8]分别实现了高达 71.7%和 27.4%的相对增益。

我们还进行了一项**消融研究，表明在功能验证数据集上训练的模型比在非验证数据上训练的模型表现更好**，在相同的基础模型和训练设置下。这些**结果突出了在 RTL 代码生成中使用高质量、功能验证数据集的重要性**。我们的贡献如下：

- 我们**引入了 VeriCoder1**，这是一个**在功能正确性验证的数据集上进行微调的 RTL 代码生成模型**。在 VerilogEval 和 RTLLM 基准测试中，VeriCoder 在开源微调模型中实现了最先进的性能，**在 pass@k 指标上相对于之前最佳模型分别实现了高达 71.7%和 27.4%的相对增益**。
- 我们开发了**一种数据集增强流程，将单元测试生成与教师 LLM 引导的反馈引导细化相结合**。据我们所知，这产生了迄今为止最大规模的经过功能验证的微调数据集，包含**超过 125k 个经过验证的三元组，包括自然语言规范、RTL 设计和通过的测试**。
- 我们进行了**一项消融研究**，表明在**数据集构建过程中进行功能验证可以提高模型性能**，强调了在 RTL 代码生成中使用高质量功能验证数据集的重要性。
:::

## 二、背景与相关工作

### A. 语言建模与微调

> 大型语言模型（LLMs）是深度神经网络，用于执行语言建模任务，即学习预测序列中的下一个词。简单来说，语言建模就是教会模型如何预测文本中下一个可能出现的词。

形式上，给定一个词序列 $x = (x_1, x_2, \ldots, x_T)$，训练目标是最大化其对数似然：

$$
\mathcal{L}_{LM} = \sum_{t=1}^{T} \log P(x_t | x_{ < t}; \theta)
$$

其中 $\theta$ 表示模型参数，$x_{<t} = (x_1, \ldots, x_{t-1})$ 表示上下文词序列。这种自回归目标使模型能够生成连贯的文本，并捕捉不同领域中的长距离依赖关系。大型语言模型的训练通常分为两个阶段：

- 预训练：模型在海量、多样化的语料库（例如网络数据、书籍、源代码）上进行训练，以获取广泛的知识和语言理解能力。这个阶段成本高昂，每个模型只进行一次训练。
- 后训练 预训练模型通过较小的、精心策划的数据集适配到特定任务。这包括监督微调（SFT），即在特定于任务的输入-输出对上训练模型。

由于从头开始微调大型模型资源成本高，研究人员开发了参数高效的微调方法。一种广泛使用的方法是低秩适应（LoRA）。与其更新每个线性层中的完整权重矩阵 $W \in \mathbb{R}^{d \times k}$，不如冻结原始权重并引入可训练的低秩更新：

$$
W' = W + \Delta W = W + AB
$$

其中 $A \in \mathbb{R}^{d \times r}$，且 $B \in \mathbb{R}^{r \times k}$，其中 $r \ll \min(d, k)$。在训练期间，只更新 $A$ 和 $B$，而 $W$ 保持不变。这种技术降低了适应过程中的内存和计算开销，使得将大型语言模型针对特定领域（如 RTL 生成）进行专业化成为可能，同时计算资源需求有限。

### B. RTL 代码数据集

> 由于缺乏大型、高质量的数据集，开源 RTL 代码生成的进展受到限制。为了缓解这一问题，最近的努力集中在自动化数据挖掘和增强技术上，以丰富现有的 RTL 示例语料库。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/03.webp)

**表 I 对比了构建微调数据集的不同策略。从开源 RTL 设计中挖掘数据是构建数据集的常见策略**。

- VeriGen 从 GitHub 和教科书编译 Verilog 模块，使用自动化语法检查创建结构化语料库。
- BetterV 从互联网收集 Verilog 模块，然后根据编码风格和语法有效性过滤设计。
- 其他工作也采用了类似的方法来挖掘和预处理 RTL 代码。

**另一项研究利用商业 LLM 生成合成数据**。RTLCoder 提示 GPT-3.5 使用领域关键词生成任务描述和相应的 RTL，丢弃任何无法编译的输出。Origen 进一步在代码到代码的流程中使用 Claude 3.5：首先将挖掘的 RTL 代码转换为自然语言规范，然后在编译器的指导下从这些规范重新生成代码。ChipGPT 将 Verilog 抽象语法树（AST）转换为自然语言规范。

**尽管表 I 中列出的大多数现有工作都确保了语法有效性，但没有一个提供了功能正确性的证据**。

> 在数据集构建期间，如果没有全面的单元测试或基于模拟的反馈，微调在这些语料库上的模型可能会产生编译成功但不符合预期自然语言规范的代码。

最近的一项工作 OpenLLM-RTL 探索了使用 LLM 生成断言，生成了包含 7k 示例的功能验证数据集。尽管与提高微调数据集中的功能正确性这一目标相同，但我们的工作采用了不同的方法，通过生成单元测试进行验证。

> 我们的最终数据集包含超过 125k 示例，是迄今为止最大的功能验证 RTL 数据集。

## 三、我们的方法

### A. 概述

> 我们的目标是改进由自然语言规范与语法正确的 Verilog 设计组成的微调数据集的质量，此前的相关工作[6]–[8], [24], [26]均存在一定的局限性。

这些数据集，包括 Origen [8] 在内，虽然包含了通过语法检查的 Verilog 设计，但却**未通过单元测试来确保功能正确性。为解决这一局限，我们引入了一种自动化数据集增强流程**，借助教师语言模型（例如 GPT-4o-mini）对每个示例进行迭代细化以完成验证。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/04.webp)

**如图 1 所示，给定一个自然语言规范和一个初始 RTL 设计，教师模型首先生成一个单元测试。如果 RTL 设计未能通过模拟测试，模型则基于错误信息迭代修订设计**。如有必要，它还会更新单元测试，使其更好地符合自然语言规范所描述的预期功能。尽管我们的实验主要侧重于增强 Origen 数据集，但这一方法同样适用于任何缺乏测试验证的数据集。

增强流程从原始数据集 $D = \left\{ (\text{specification}, \text{design}) \right\}$ 开始，其中每个 RTL 设计旨在实现对应的自然语言规范。然而，由于缺乏测试，无法确定这些设计是否具备预期的功能行为。**对于每一对规范和设计，我们提示教师模型 GPT-4o-mini 生成一个单元测试。随后，该测试将与设计一起编译并模拟以检查正确性，这里的正确性意味着设计能够通过模拟测试。如果模拟失败，我们将提取错误信息，并使用细化提示重新调用教师模型。** 该提示包括规范、当前设计和测试以及错误信息。模型将通过尽可能少地修改设计、测试或两者来解决失败问题。

> 这一细化过程将重复进行：每次候选设计都会被重新模拟，直到设计通过测试或达到最大尝试次数。

最终输出是一个经过验证的数据集 $D' = \left\{ (\text{specification}, \text{design}, \text{test}) \right\}$，其中每个三元组包含一个自然语言规范、一个 Verilog 设计以及单元测试。**一个具体的实例将在第 III-B 节中展示，算法及提示的详细信息将在第 III-C 节中提供**。

### B. 动机示例

**图 2 展示了一个直接取自 Origen 数据集 [8] 的动机示例，突显了仅依赖语法检查进行验证的数据集的关键局限性**。此前 RTL 生成工作通常假设语法正确性足以用于微调，而无需通过单元测试验证功能。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/05.webp)

该示例包含一个自然语言规范（图 2a）、一个来自原始数据集的有缺陷 RTL 设计（图 2b）以及由我们的流程生成的修正设计（图 2c）。规范描述了一个简单的组合模块 and3，用于计算三个一位输入 a、b 和 c 的按位与。原始设计虽然语法正确，但由于几个语义问题导致功能不正确。

1. 首先，在组合 always @* 块中误用了非阻塞赋值（<=），这可能导致综合结果出人意料。
2. 如果在时序块中使用非阻塞赋值序列 —— y <= a，随后 y <= y & c，最后 y <= y & b —— 则无法正确计算并存储 a、b 和 c 的按位与。特别是非阻塞赋值会将更新推迟到当前时间步结束，这意味着所有赋值都基于 y 的初始值进行操作，最终只有最后一次赋值会生效。
3. 最后，如果将非阻塞赋值替换为阻塞赋值（=），则会引入组合反馈环路，导致无法稳定。这些错误的出现是因为包括 Origen [8] 在内的此前 RTL 数据集中的 RTL 代码是由教师 LLM（例如 Claude 3.5）合成生成的，并且仅通过语法检查进行过滤。在没有模拟或基于测试的验证的情况下，影响功能正确性的语义错误无法被发现。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/06.webp)

我们向教师模型 GPT-4o-mini 提供自然语言规范和有缺陷的 RTL 设计，并使用下面图 4a 所示的提示（第三-C 节将详细说明）提示其生成单元测试。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/07.webp)

**生成的测试如图 3 所示，它将三个输入均设置为 1，并检查输出 y 是否如预期评估为 1。** 当对有缺陷的设计（图 2b）进行该测试模拟时，模拟会卡住并最终超时。这个错误表明存在组合环路。always @* 块用于组合逻辑，其评估由块内读取的变量中的任何一个发生变化触发。在此案例中，块内既读取 y（在右侧）又写入 y（在左侧）。

在评估块时，它会安排更新 y，这导致 y 发生变化。此变化会重新触发块，导致再次安排更新 y，依此类推。这一循环无限进行，导致模拟无法收敛。修正版本将非阻塞赋值替换为单个阻塞赋值（=），确保 y 立即更新为 a & b & c 的结果，符合规范要求。此版本通过了教师模型生成的测试，并在模拟中表现正确。

**该示例突显了在 RTL 数据集中进行功能验证的重要性**。**仅靠语法检查无法捕获影响功能正确性的微妙但关键的语义错误**。我们的方法通过教师驱动的测试生成和迭代细化，确保增强数据集中的每个设计不仅语法正确，而且通过单元测试进行了功能验证。

### C. 算法和提示

**下面算法 1 展示了我们将未验证的 RTL 数据集转换为功能验证数据集的自动化流程**。从数据集 $D = \left\{ (s_i, d_i) \right\}_{i=1}^N$ 开始，其中每个示例包含一个自然语言规范 $s_i$ 和一个对应的 RTL 设计 $d_i$（例如来自 Origen [8]），我们的目标是为每个设计生成一个单元测试 $t_i$，以验证其功能正确性。如果设计未能通过测试，我们将调用迭代细化循环，在达到最大尝试次数 $T$ 之前更新设计和测试。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/08.webp)

在实验中，我们将 $T$ 设为 5。该流程由教师模型 GPT-4o-mini 驱动，对应算法 1 中的 LLMInvoke 调用。尽管 GPT-4o 或 o3-mini 等更强大的模型可能带来更好的性能，但在实际应用中我们选择了 GPT-4o-mini，这是由于 Origen 数据集规模庞大（包含 217,462 个示例）且反复调用 OpenAI 模型 API 成本高昂。流**程首先使用图 4a 所示的测试生成提示** （第 III-C 节将详细描述）**向教师模型提示自然语言规范和初始 RTL 设计** （例如图 2a 和图 2b）。模型**随后生成候选单元测试（例如图 3），旨在通过模拟检查设计是否满足预期功能**。

使用标准 Verilog 工具对设计和测试进行编译和模拟。如果测试失败，例如由于超时、输出错误或其他运行时错误，我们将构建细化提示（图 4b），包括规范、失败的设计和测试以及模拟错误信息（对应算法 1 中的 err 变量）。

**此提示随后传递给教师模型，模型将尝试通过修改设计、测试或两者来解决问题。细化过程重复进行，直到更新后的设计通过模拟或达到最大尝试次数**$T$。一旦设计成功通过测试，验证后的三元组 $(s_i, d_i, t_i)$ 将被添加到输出数据集 $D'$ 中。**该策略能够系统检测并修正无法通过语法检查发现的微妙 RTL 错误。**

通过将基于 LLM 的测试生成和迭代细化集成到数据集构建流程中，我们生成的数据集不仅语法正确，而且通过模拟进行了功能验证。尽管我们无法保证增强数据集中的每个设计在所有可能输入下都功能正确，但单元测试的加入使其比仅依赖语法检查的先前方法更健壮。我们认为这是朝着构建更高质量 RTL 生成微调数据集所迈出的实际且可扩展的一步。

## 四、实验设置

### A. 数据集

> 按照第三节所述的方法，我们构建了一个包含 125,777 个示例的微调数据集。每个示例均包含一个自然语言规范、一个相应的 RTL 设计以及关联的单元测试。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/09.webp)

**表 II 汇总了关键统计信息**：规范平均包含 247 个词（范围从 116 到 549），RTL 实现平均为 35 行代码（范围从 5 到 225），单元测试平均为 55 行（范围从 6 到 197）。我们使用该数据集中的规范 - 解决方案对来训练我们的模型 VeriCoder。

### B. LoRA 微调设置

> 遵循标准的大型语言模型（LLM）微调实践，我们使用低秩适应（LoRA，一种参数高效的微调方法，通过在模型的每个线性层上添加低秩矩阵来更新模型，而非更新整个模型的所有参数，从而减少计算资源和内存的消耗）对基础模型 Qwen2.5-14B-Instruct 进行微调，秩设为 16，扩展因子为 32，应用于 Transformer 中所有线性投影层。

训练进行了 3 个周期，批量大小为 40。我们采用恒定的学习率 $1×10^{−5}$，搭配线性衰减调度器和 $0.05$ 的预热比率。优化器采用权重衰减为 $1×10^{−4}$，并应用最大范数为 1 的梯度裁剪。

### C. 基准测试和指标

遵循先前工作 [7], [8] 建立的评估协议，我们针对 VerilogEval [9] 和 RTLLM [10] 进行基准测试。对于 VerilogEval，我们报告标准的 Pass@k 指标（$k \in \{1, 5, 10\}$），用于估计前 k 个生成程序中至少有一个通过所有测试用例的概率。该指标定义为：

$$
\text{Pass}@k = \mathbb{E} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]
$$

其中 $n$ 为生成程序的总数，$c$ 为正确的数量。所有测试用例均由专家设计。在所有评估中，我们设 $n = 10$。对于 **RTLLM，我们同时报告语法正确性和功能正确性，均采用 Pass@5 指标**。此评估设置与先前工作 [8] 一致。

### D. 评估模型

我们评估了两组模型。

- 第一组包括仅预训练的基础模型，如 OpenAI 最新发布的模型（o4-mini, o3-mini, GPT-4o, GPT-4o-mini）、Google 的 Gemini 2.0 Flash、DeepSeek 的 R1 和 DeepSeek-Coder-7B-v1.5（先前工作 [8] 中使用的基模型）、Meta 的 LLaMA2-7B 模型以及阿里巴巴的 Qwen2.5-14B-Instruct（我们微调的基础模型）。
- 第二组包括先前工作中发布的具有已发布权重的微调模型：Origen [8]、RTLCoder [6] 和 ChipGPT [27]。为确保公平比较，我们对所有模型使用相同的输入提示和后处理脚本。对于先前工作的模型，我们未采用其模型特定的提示 [8] 或推理管道 [6], [27]。

我们应用统一的评估脚本，**唯一变量是受测模型**。**这种标准化至关重要，因为输入格式和后处理都会显著影响性能**。通过控制这些因素，我们隔离了模型能力，实现了公平比较。

## 五、结果

### A. 主要评估结果

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/10.webp)

表 III 显示了评估结果。我们的主要发现如下：

1. **与先前工作的比较**：VeriCoder **在两个 RTL 代码生成基准测试中均取得了最先进的成果，超越了所有先前发布的开源微调模型**。 在 VerilogEval-Machine 上，VeriCoder 的 pass@1 准确率达到 55.7%，相比先前最佳模型 OriGen 提出了 19.8 个百分点的改进。在 VerilogEval-Human 上，它达到了 38.3%，超过了 OriGen 的 22.3%，提升了 16.0 个百分点。在所有评估的 k-shot 设置（k=1,5,10）中，VeriCoder 在 Human 分割集上始终保持领先地位。在 RTLLM 基准测试中，VeriCoder 实现了 79.3% 的语法正确性和 48.3% 的功能正确性，分别超过了 OriGen 的 51.7% 和 37.9%。综上所述，VeriCoder 在 VerilogEval 和 RTLLM 的 pass@k 准确率上分别实现了高达 71.7% 和 27.4% 的相对改进，超越了先前在两个基准测试中的最佳模型。
2. **我们微调的有效性**：以 Qwen-2.5-14B-Instruct 作为基础模型，VeriCoder 在 VerilogEval 上实现了显著提升。在 VerilogEval-Machine 分割集上，pass@1 提升了 7.6%，pass@5 提升了 4.0%，pass@10 提升了 2.1%；VerilogEval-Human 显示出相同的趋势。在 RTLLM 上，功能 pass@5 比基础模型高出 7%。特别是，VeriCoder 甚至在 VerilogEval 的 Eval-Machine 的 pass@5 和 pass@10 指标上，以及 RTLLM 上，超过了 Google 的 Gemini2.0-flash 商业模型。这些结果共同证明了我们的微调过程和验证数据集在提升 RTL 生成的 pass@k 指标和语义正确性方面的显著效果。
3. **模型差距依然存在**：尽管取得了上述改进，但 **VeriCoder 与最强的大型模型之间仍存在显著的性能差距**。例如，o3mini 在 VerilogEval Pass@1 上达到了 66.4%，而 VeriCoder 为 55.7%。DeepSeek-R1 在人类评分的 Pass@5 上达到了 69.1%，而 VeriCoder 为 49.2%。像 GPT-4o 这样的商业 LLM 在 Syntax-VCS 有效性上达到了完美的 100.0%，功能正确性达到了 69.0%，而 VeriCoder 分别记录为 79.3% 和 48.3%。尽管存在性能差距，但开源轻量级模型具有显著的优势。它们提供了透明度，允许本地部署，并确保知识产权保护，这些能力在 RTL 设计工作流程中尤为重要，因为安全性、可定制性和与现有工具链的集成至关重要。

### B. 数据集消融研究

为了**评估数据集质量对 RTL 代码生成的影响**，我们进行了消融研究，使用相同的基线模型 Qwen2.5-14B-Instruct，分别在以下两个数据集上进行微调：

1. 先前工作 [8] 中未经验证的 OriGen 数据集
2. 我们新策划的功能验证数据集。为了确保公平比较，除了数据集外，所有因素（包括数据集规模、微调超参数、训练程序和评估设置）都保持不变。

> 在所有指标上，随着数据集质量的提高，我们观察到了一致的改进。

在 VerilogEval 基准测试中，基线模型实现了 46.8% 的 Pass@5。在未经验证的数据集上微调后性能提升至 53.5%，而我们的验证数据集进一步将其提升至 55.8%。**对于 RTLLM 语法正确性，趋势相似**：**基线模型为 69.0%，未经验证版本为 75.9%，经过验证的数据为 79.3%。**

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/11.webp)

功能正确性的提升更为显著，从基线的 41.4% 提升到未经验证的 44.8%，最终达到验证后的 48.3%。**这些结果表明，功能验证数据比现有的未经验证数据提供了更有效的监督**。这也**强调了在微调大型语言模型以进行 RTL 代码生成时数据集质量的重要性**。

### C. 非验证数据集的测试通过率

我们通过在教师模型 GPT-4omini 生成的合成单元测试上评估先前工作的微调数据集的质量。对于每个语料库，我们随机抽取了 1,000 个 Verilog 实现，并应用了第 III 节中描述的测试生成和细化管道。然后，我们对原始设计运行相应的单元测试，并测量原始设计成功通过生成测试的比例。

![文章插图](/articles/2025-05-11-cmu-vericoder-llm-rtl-verilogeval-c53f5339/images/12.webp)

如表 V 所示，RTLCoder 数据集 [6] 中只有 24.4% 的示例通过了我们的功能测试，而 OriGen [8] 达到了 53.5%。OriGen 较高的通过率与其在表 III 中更强的代码生成结果相一致，**暗示了数据集有效性与下游性能之间的正向关联**。这些发现突出了在微调数据集策划中纳入功能正确性验证的潜在价值，以实现更好的 RTL 代码生成。

## 六、讨论与未来工作

> 尽管 VeriCoder 结合单元测试生成与反馈驱动的细化流程提升了生成 RTL 代码的功能正确性，但却**无法全方位保障代码的准确性**。**合成测试用例难以涵盖所有潜在的边缘情况**。

为攻克这一难题，**未来研究可探索将形式化验证技术纳入数据集构建流程，从而严苛确保生成代码的准确性**。近期进展已彰显将自然语言指令转化为形式化规范，以及在基于 LLM 的代码生成过程中强制实施形式化约束的光明前景。此外，包括 VeriCoder 在内的多数现有方法聚焦于小规模 RTL 生成。

然而，**实际的硬件开发往往涉及大型、存储库级别的代码库，这类代码库具有错综复杂的跨文件依赖关系，并对长程上下文提出要求**。近**期研究已着手应对这些挑战，提出融合微调与检索增强式 RTL 代码生成的技术**。将 VeriCoder 的单元测试生成及反馈驱动精细化**组件拓展至存储库层面**，将赋能 LLM 处理**更具现实意义的 RTL 任务**。

再者，强化学习（RL）为超越监督微调的大型语言模型性能优化提供了强大框架。近期研究已证实 RL 在融合多样化反馈形式（例如测试用例结果、编译器诊断信息以及形式化验证结果）以提升基于 LLM 的代码生成质量方面的效用。基于这些进展**，未来研究可探讨针对 VeriCoder 数据集运用 RL 技术，借助配套测试用例作为反馈信号，迭代优化 RTL 代码生成质量。**

## 七、结论

> 大型语言模型（LLMs）的近期发展为电子设计自动化（EDA）领域，尤其是 RTL 代码生成带来了新的机遇。然而，现有大多数数据集侧重语法有效性而忽视功能正确性，限制了微调模型的效能。

我们推出了 **VERICODER，一个基于 125,000 个经功能正确性验证示例微调的模型**。**借助教师 LLM 引导的反馈驱动精细化流程构建数据集，生成并迭代更新 RTL 设计与单元测试，直至设计通过模拟**。由此产生的**数据集包含经功能验证的三元组：自然语言规范、RTL 实现与通过测试**。

经此数据集微调的 VERICODER 在两大既定 RTL 基准测试中取得领先成果，**在 VerilogEval 和 RTLLM 上较之前最佳模型分别达成高达 71.7% 和 27.4% 的相对改进**。消融研究**证实功能验证对模型性能的影响，凸显高质量训练数据对 RTL 代码生成的重要性**。

展望未来，研究可融合形式化验证与强化学习以进一步提升模型在人工智能辅助硬件设计中的性能表现。

## 参考文献

::: block-1
[1] M. Liu, T.-D. Ene, R. Kirby, C. Cheng, N. Pinckney, R. Liang, J. Alben, H. Anand, S. Banerjee, I. Bayraktaroglu et al., “Chipnemo: Domainadapted llms for chip design,” arXiv preprint arXiv:2311.00176, 2023.

[2] L. Chen, Y. Chen, Z. Chu, W. Fang, T.-Y. Ho, R. Huang, Y. Huang, S. Khan, M. Li, X. Li et al., “The dawn of ai-native eda: Opportunities and challenges of large circuit models,” arXiv preprint arXiv:2403.07257, 2024.

[3] R. Zhong, X. Du, S. Kai, Z. Tang, S. Xu, H.-L. Zhen, J. Hao, Q. Xu, M. Yuan, and J. Yan, “Llm4eda: Emerging progress in large language models for electronic design automation,” arXiv preprint arXiv:2401.12224, 2023.

[4] Z. He and B. Yu, “Large language models for eda: Future or mirage?” in Proceedings of the 2024 International Symposium on Physical Design, 2024, pp. 65–66.

[5] X. Yao, Y. Wang, X. Li, Y. Lian, R. Chen, L. Chen, M. Yuan, H. Xu, and B. Yu, “Rtlrewriter: Methodologies for large models aided rtl code optimization,” in Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design, 2024, pp. 1–7.

[6] S. Liu, W. Fang, Y. Lu, J. Wang, Q. Zhang, H. Zhang, and Z. Xie, “Rtlcoder: Fully open-source and efficient llm-assisted rtl code generation technique,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2024.

[7] S. Liu, W. Fang, Y. Lu, Q. Zhang, H. Zhang, and Z. Xie, “Rtlcoder: Outperforming gpt-3.5 in design rtl generation with our open-source dataset and lightweight solution,” in 2024 IEEE LLM Aided Design Workshop (LAD). IEEE, 2024, pp. 1–5.

[8] F. Cui, C. Yin, K. Zhou, Y. Xiao, G. Sun, Q. Xu, Q. Guo, D. Song, D. Lin, X. Zhang et al., “Origen: Enhancing rtl code generation with code-tocode augmentation and self-reflection,” arXiv preprint arXiv:2407.16237, 2024.

[9] M. Liu, N. Pinckney, B. Khailany, and H. Ren, “Verilogeval: Evaluating large language models for verilog code generation,” in 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD). IEEE, 2023, pp. 1–8.

[10] Y. Lu, S. Liu, Q. Zhang, and Z. Xie, “Rtllm: An open-source benchmark for design rtl generation with large language model,” in 2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC). IEEE, 2024, pp. 722–727.

[11] Y. Tsai, M. Liu, and H. Ren, “Rtlfixer: Automatically fixing rtl syntax errors with large language model,” in Proceedings of the 61st ACM/IEEE Design Automation Conference, 2024, pp. 1–6.

[12] Y. Liao, T. Adegbija, and R. Lysecky, “Are llms any good for highlevel synthesis?” in Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design, 2024, pp. 1–8.

[13] Y. Fu, Y. Zhang, Z. Yu, S. Li, Z. Ye, C. Li, C. Wan, and Y. C. Lin, “Gpt4aigchip: Towards next-generation ai accelerator design automation via large language models,” in 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD). IEEE, 2023, pp. 1–9.

[14] Z. Yan, Y. Qin, X. S. Hu, and Y. Shi, “On the viability of using llms for sw/hw co-design: An example in designing cim dnn accelerators,” in 2023 IEEE 36th International System-on-Chip Conference (SOCC). IEEE, 2023, pp. 1–6.

[15] Z. Liang, J. Cheng, R. Yang, H. Ren, Z. Song, D. Wu, X. Qian, T. Li, and Y. Shi, “Unleashing the potential of llms for quantum computing: A study in quantum architecture design,” arXiv preprint arXiv:2307.08191, 2023.

[16] M. Cosler, C. Hahn, D. Mendoza, F. Schmitt, and C. Trippel, “nl2spec: Interactively translating unstructured natural language to temporal logics with large language models,” in International Conference on Computer Aided Verification. Springer, 2023, pp. 383–396.

[17] C. Sun, C. Hahn, and C. Trippel, “Towards improving verification productivity with circuit-aware translation of natural language to systemverilog assertions,” in First International Workshop on Deep Learning-aided Verification, 2023.

[18] H. Wu, Z. He, X. Zhang, X. Yao, S. Zheng, H. Zheng, and B. Yu, “Chateda: A large language model powered autonomous agent for eda,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2024.

[19] Z. Xiao, X. He, H. Wu, B. Yu, and Y. Guo, “Eda-copilot: A ragpowered intelligent assistant for eda tools,” ACM Transactions on Design Automation of Electronic Systems, 2025.

[20] K. Xu, J. Sun, Y. Hu, X. Fang, W. Shan, X. Wang, and Z. Jiang, “Meic: Re-thinking rtl debug automation using llms,” in Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design, 2024, pp. 1–9.

[21] R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Marone, C. Akiki, J. Li, J. Chim et al., “Starcoder: may the source be with you!” arXiv preprint arXiv:2305.06161, 2023.

[22] A. Lozhkov, R. Li, L. B. Allal, F. Cassano, J. Lamy-Poirier, N. Tazi, A. Tang, D. Pykhtar, J. Liu, Y. Wei et al., “Starcoder 2 and the stack v2: The next generation,” arXiv preprint arXiv:2402.19173, 2024.

[23] E. Dehaerne, B. Dey, S. Halder, and S. De Gendt, “A deep learning framework for verilog autocompletion towards design and verification automation,” arXiv preprint arXiv:2304.13840, 2023.

[24] Z. Pei, H.-L. Zhen, M. Yuan, Y. Huang, and B. Yu, “Betterv: Controlled verilog generation with discriminative guidance,” arXiv preprint arXiv:2402.03375, 2024.

[25] S. Thakur, B. Ahmad, Z. Fan, H. Pearce, B. Tan, R. Karri, B. DolanGavitt, and S. Garg, “Benchmarking large language models for automated verilog rtl code generation,” in 2023 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2023, pp. 1–6.

[26] S. Thakur, B. Ahmad, H. Pearce, B. Tan, B. Dolan-Gavitt, R. Karri, and S. Garg, “Verigen: A large language model for verilog code generation,” ACM Transactions on Design Automation of Electronic Systems, vol. 29, no. 3, pp. 1–31, 2024.

[27] K. Chang, K. Wang, N. Yang, Y. Wang, D. Jin, W. Zhu, Z. Chen, C. Li, H. Yan, Y. Zhou et al., “Data is all you need: Finetuning llms for chip design via an automated design-data augmentation framework,” in Proceedings of the 61st ACM/IEEE Design Automation Conference, 2024, pp. 1–6.

[28] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.

[29] Y. Zhang, Z. Yu, Y. Fu, C. Wan, and Y. C. Lin, “Mg-verilog: Multigrained dataset towards enhanced llm-assisted verilog generation,” in 2024 IEEE LLM Aided Design Workshop (LAD). IEEE, 2024, pp. 1–5.

[30] E. Goh, M. Xiang, I. Wey, T. H. Teo et al., “From english to asic: Hardware implementation with large language model,” arXiv preprint arXiv:2403.07039, 2024.

[31] S. Liu, Y. Lu, W. Fang, M. Li, and Z. Xie, “Openllm-rtl: Open dataset and benchmark for llm-aided design rtl generation,” in Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design, 2024, pp. 1–9.

[32] D. Mendoza, C. Hahn, and C. Trippel, “Translating natural language to temporal logics with large language models and model checkers,” in 2024 Formal Methods in Computer-Aided Design (FMCAD), 2024, pp. 1–11.

[33] P. Aggarwal, B. Parno, and S. Welleck, “Alphaverus: Bootstrapping formally verified code generation through self-improving translation and treefinement,” arXiv preprint arXiv:2412.06176, 2024.

[34] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan, “Swe-bench: Can language models resolve real-world github issues?” arXiv preprint arXiv:2310.06770, 2023.

[35] T. Suresh, R. G. Reddy, Y. Xu, Z. Nussbaum, A. Mulyar, B. Duderstadt, and H. Ji, “Cornstack: High-quality contrastive data for better code retrieval and reranking,” in The Thirteenth International Conference on Learning Representations, 2025.

[36] N. Jain, M. Shetty, T. Zhang, K. Han, K. Sen, and I. Stoica, “R2e: Turning any github repository into a programming agent environment,” in ICML, 2024.

[37] P. Wu, N. Guo, J. Lv, X. Xiao, and X. Ye, “Rtlrepocoder: Repositorylevel rtl code completion through the combination of fine-tuning and retrieval augmentation,” arXiv preprint arXiv:2504.08862, 2025.

[38] Z. Li, C. Xu, Z. Shi, Z. Peng, Y. Liu, Y. Zhou, L. Zhou, C. Ma, J. Zhong, X. Wang et al., “Deepcircuitx: A comprehensive repository-level dataset for rtl code understanding, generation, and ppa analysis,” arXiv preprint arXiv:2502.18297, 2025.

[39] N. Wang, B. Yao, J. Zhou, X. Wang, Z. Jiang, and N. Guan, “Large language model for verilog generation with golden code feedback,” arXiv preprint arXiv:2407.18271, 2024.

[40] J. Wang, Z. Zhang, Y. He, Y. Song, T. Shi, Y. Li, H. Xu, K. Wu, G. Qian, Q. Chen et al., “Enhancing code llms with reinforcement learning in code generation,” arXiv preprint arXiv:2412.20367, 2024.
:::
