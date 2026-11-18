> 近年来，大型语言模型（LLMs）通过优化传统的自然语言处理（NLP）流程，在提升人工智能性能和泛化能力方面取得了显著进展。

这推动了它们在各种系统中的集成应用。包括我们许多 NLP 系统，都**采用了直接将 LLM 集成到流程中的 “单阶段” 流式架构**。尽管这种方法效果显著，但**由于需要庞大的模型参数来实现满意的结果**，因而**导致了巨大的成本支出和推理延迟**。

::: block-1
本文介绍了一种**三阶段的面向成本效益的端到端 LLM 部署流程**，包括**原型设计、知识转移和模型压缩**，以应对基于 LLM 的框架在**实际部署中的 “成本 - 性能” 困境**。

我们的方法**最终** 生成了一个**超小型模型**，该模型在我们的**在线系统中得到部署，实现了最优的成本 - 性能指标，同时大幅简化了系统架构**。
:::

- 首先，通过将**复杂任务转化为基于函数调用的 LLM 驱动流程**，**构建出性能最优的原型系统**，以产生高质量数据，**作为教师模型**。
- 在**第二阶段**，结合拒绝微调（RFT）、强化学习（RL）和知识蒸馏（KD）等技术，**将知识转移到** 仅有 0.5B 参数的**小型学生模型中**，在极低成本下实现了有效性能。
- **最后阶段，通过量化和剪枝技术将模型进一步压缩** 至 0.4B 参数，**实现了超低延迟和成本**。

该框架的模块化设计和跨领域能力表明其在其他 NLP 领域也具有潜在适用性。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/01.webp)

- https://arxiv.org/abs/2504.13471

## 本文目录

- 一、引言
- 二、相关工作
  - 用于函数调用及评估利用的大型语言模型
  - 知识蒸馏
  - 大型语言模型剪枝与量化
  - 强化学习
- 三、我们的方案
  - 3.1 原型设计
  - 3.2 知识转移
  - 3.3 模型压缩
- 四、实验
  - 4.1 实验设置
  - 4.2 实验结果
- 五、分析
  - 模型规模与计算效率之间的权衡
  - 训练范式的特征
  - 模型压缩策略的比较
- 六、结论
  - 超小型模型是低成本的实用替代方案
  - 超越通用人工智能的平衡关注
  - 可扩展的三阶段成本效益管道
- 七、限制与未来工作
- 参考文献

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/02.webp)

## 一、引言

> 近年来，大型语言模型（LLMs）在提升人工智能方面取得了显著进展[1; 2; 3]，通过解决传统管道（流程）的不足——包括手动标注、性能欠佳以及泛化和适应能力有限——从而推动了其在各种自然语言处理（NLP）系统中的集成需求。

我们发现，**包括我们的系统在内的许多 NLP 系统，都直接采用了将 LLM 集成到管道中的方式，通常被称为 “单阶段” 流程**。尽管这种流程促进了突破，但它也**导致了显著的成本增加[4]和高推理延迟** [5]。在我们的 NLP 系统中，**成本 - 性能是一个尤其关键的指标**，这可能是该领域其他人的共识[6]。

这些巨大的成本限制了 LLM 的更广泛应用，并影响了它们的商业可行性。我们的分析指出，正是这种单阶段管道的简单训练方法，需要较大的模型参数才能实现满意的结果，从而导致了显著的成本。

**在本文中，我们介绍了一种三阶段的高性能且面向成本效益的端到端 LLM 部署流程，包括原型设计、知识转移和模型压缩**，以应对基于 LLM 的框架在实际部署中普遍存在的 “成本 - 性能” 困境。这种方法最终生成了一个超小型模型，该模型在我们的在线系统中得到部署，实现了最优的成本 - 性能指标，同时大幅简化了系统架构。

首先，在原型设计阶段，通过将传统复杂任务转化为基于函数调用的 LLM 驱动工作流（通过提示工程），构建出一个性能最优的原型，与传统管道相比具有优势。

此原型作为教师模型，在第二阶段高效地生成大量高质量数据。在知识转移阶段，利用第一阶段得出的最优原型，我们采用拒绝微调（RFT）、强化学习（RL）[8; 9; 10] 和多策略知识蒸馏（KD）等技术，针对 Qwen 2.5 系列模型[11]，将知识有效地转移到一个仅有 0.5B 参数的学生模型中。由此得出的学生模型比原型模型小得多，同时在极低成本下实现了几乎相当的性能。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/03.webp)

在第三阶段，我们利用量化和剪枝技术进一步压缩第二阶段开发的学生模型，最终得到一个仅有 4 亿参数（如图 1 所示）的极其紧凑的模型。

最终，这个超小型模型被部署在我们的在线系统中。**该模型实现了数百倍的压缩，并且以微不足道且可接受的性能损失为代价，换取了极低的延迟和成本**。该框架的模块化设计和卓越的跨领域能力表明其在其他 NLP 领域也具有潜在的适用性。

该论文的**主要贡献如下**：

- 我们首次验证了一种**三阶段的面向成本效益的端到端 LLM 流程，简化了原始系统复杂性，实现了最佳结果（SOTA），解决了 “成本 - 性能” 困境**，实现了相对于 72B LLM 的 **180 倍压缩**，同时与传统基于 BERT 的管道相比，**绝对准确率（AR）提高了 14%。**
- 我们**提出了首个混合策略**，通过**结合 RL 和 KD**，显著增强了超小型模型在特定领域任务上的泛化能力。这种方法**有效地将 LLM 的能力转移到较小的模型中**。
- 所**提出的框架具有低训练、推理和部署成本的优势，项目开发成本低且周期短，并且具有强大的跨领域迁移潜力**。它为学术界和工业界开发标准化的跨领域解决方案提供了灵感，促进了 LLM 的广泛采用。

## 二、相关工作

### 用于函数调用及评估利用的大型语言模型

大型语言模型（LLMs）凭借在上下文中的学习和推理能力，能够根据用户意图从提供的列表中选择函数及参数[12]。经过微调的 7B 模型（如 Gorilla[13]）在包含数千个 API 的数据集上实现了 GPT-4 级别的函数调用准确率。基于此，**TinyAgent[14] 使参数少于 1.1B 的模型具备了边缘设备上的函数调用能力**。

通过拒绝采样优化 API 检索的查询到 API 重写，可以减少检索延迟并提高准确性[15]。如今，LLMs 可用于评估模型生成内容[16; 17]。CriticGPT[18] 表明 LLMs 能够构建出性价比高且精准的评判者，超越人类在检测细微错误方面的能力。

### 知识蒸馏

**知识蒸馏（KD）[19] 通过模拟教师模型的概率分布，将知识转移到较小的学生模型中**。常见的方法包括基于 logits[20; 21]、基于特征[22] 和序列级（Seq-KD）[23] 蒸馏。最初，基于 logits 的 KD 采用前向 KL 散度（FKL）[24; 25] 作为损失函数。一些 KD 方法更倾向于逆向 KL 散度（RKL）[26; 27; 28; 21]。最近，自适应 Kullback-Leibler（AKL）散度[29] 实现了模仿分布和避免过拟合之间的平衡。

### 大型语言模型剪枝与量化

> 高效的 LLM 部署主要依赖于剪枝[30; 31]和量化[32; 33; 34]。

- 结构化剪枝[35]采用基于梯度[31; 38]或激活感知[37]的重要性标准来移除层[36; 37]或降低维度[31; 37; 38]，通过可学习的掩码[38; 39]或剪枝 - 重新训练管道[36; 31; 37]来实现。
- 仅权重量化的 LLM 方法[32; 33; 40]提高了内存效率，而联合权重 - 激活的方法[34; 41]及其组合[42]则加速了计算。

### 强化学习

**近期的研究[8; 9; 10]聚焦于利用强化学习（RL）来增强模型的推理能力。** 出现了诸如 DeepSeek-R1[43]、QWQ-32B[44] 和 OpenAI o1[45] 等显著的推理模型，它们采用了诸如近端策略优化（PPO）[46]、组相对策略优化（GRPO）[47] 等 RL 算法。

对于通用 LLMs，偏好优化算法[51; 52; 53; 54]得到了广泛应用。此外，还提出了自我改进管道[55; 56; 57]，以迭代方式提高模型性能。**我们主要利用 RL 来增强模型的泛化能力。与直接蒸馏推理模型相比，RL 能够产生更短且更精确的思维链（CoT）。**

## 三、我们的方案

最近大型语言模型（LLMs）的发展促进了自然语言处理（NLP）任务（如分类和总结）的显著进步。我们利用这些能力**通过一种面向成本效益的端到端 LLM 流程来重构传统对话系统**，**该流程** 包括原型设计、知识转移和模型压缩，**如图 2a 和图 2b 所示**。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/04.webp)

首先，我们利用当代 LLMs 的卓越性能来开发原型系统。然后，我们将知识从大型模型转移到小型模型。最后，我们通过进一步的模型压缩技术来实现模型尺寸的额外缩减。**最终，我们的目标是以最低的成本提供智能的用户体验，如图 2c 所示。**

### 3.1 原型设计

#### LLM 原型

> 传统的对话系统通常采用**意图分类和槽填充（ICSF）框架**，常常使用 BERT[60; 61]。

然而，**单一模型往往难以处理复杂的现实问题，如表 1 所示**。因此，通常会使用领域分类器（DC）来**首先将查询分类到不同领域，从而实现特定领域的 ICSF** 。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/05.webp)

**尽管分层的 DC-ICSF 架构提高了准确性，但随着特征更新，系统复杂性也随之增加**。这导致了各模块之间的高度耦合，增加了开发和维护成本。此外，领域边界的模糊性使 DC 成为整体端到端准确率的瓶颈，形成 “特征扩展 - 操作难度” 的反馈循环，阻碍了对话体验的优化。

现代 LLMs 展现出了足以处理这些 NLP 问题的能力。特别是在相关的工具使用工作中，**LLMs 能够通过在提示中附加函数描述来学习调用函数，而无需额外训练。本质上，ICSF 可以被视为一个函数调用任务，模型从用户查询中提取关键信息作为特定模块执行的函数参数**。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/04.webp)

**按照这种方法，我们基于 Qwen2.5-72B-Instruct[11] 设计了一个原型系统。如图 2a 所示**，该原型系统接受用户查询和上下文信息（包括聊天历史和时间状态信息）作为输入，在 API 池中搜索相关 API，然后输出函数名和参数以供调用。

传统的系统由一个 DC 模块和多个基于 BERT 或正则表达式的 ICSF 模型组成，每个模块都需要单独维护，并且需要频繁更新 DC。相比之下，基于 LLM 的系统具备函数调用能力，通过一个统一的模型处理所有任务，如图 1 所示，从而减少了系统复杂性并提高了操作效率。

为了增强时间感知能力并减少幻觉现象，我们通过 RAG[62] 丰富了输入中的上下文知识，从策划的知识库中检索实体相关术语的解释和扩展。**我们没有将 API 池中所有可调用的 API 添加到 LLM 提示中，而是实现了一个受[15]启发的 API 检索模块**。

::: block-1
LLM 首先生成潜在的 API 描述，然后通过向量相似性（使用 GTE-Qwen2-1.5B[63] 作为嵌入模型和 FAISS[64] 进行索引）检索前 K 个 API。然后，LLM 从候选 API 中选择最佳的函数 / 参数。

- 如果不存在合适的函数，系统将返回 “无合适函数”；
- 在查询不明确的情况下，将生成 “请求不清晰”。

为了确保向后兼容性，我们设计了一个适配器，将函数 / 参数映射到传统的领域 - 意图 - 槽结构。
:::

#### 评判

> 传统的 ICSF 评估方案依赖于测试集，要求意图和槽完全匹配。然而，在现实场景中，函数和槽的选择存在多种解决方案。

例如：（1）“稍微调高音量” 可以通过音量设置或音量调高 API 来解决；（2）“晚上 11:00” 和 “23:00” 在语义上表示相同的时间。此外，由于**众包标注人员对特定业务逻辑的理解有限**，手动标注的准确性较低。**如图 2c 所示，众包标注人员的标注准确性已成为系统的瓶颈**。

**为了解决上述问题，我们引入了基于 LLM 的评判来评估原型的准确性**。我们基于 QwQ-32B[44] 设计了一个评判模型：给定上下文和一组函数及其参数，评判模型判断该组合是否在语义上满足用户查询。按照 Samuylova 的最佳实践，我们使用 1k 专家标注数据集对评判模型进行了优化。随后，在相同条件下，由 LLM 和 9 位领域专家共同评估了 3k 测试样本。

结果表明，评判模型能够提供逻辑连贯的思维链（CoT）理由，并且与专家判断的一致性达到了 93%。基于这些发现，我们得出**结论：基于 LLM 的评判方法是该任务的最佳评估方式。**

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/06.webp)

**如图 4 所示，我们将对话管道从基于 BERT 的 DC-ICSF 架构重构为基于 LLM 的函数调用架构**。通过 LLM 作为评判的评估表明，**与原始架构相比，新架构的准确性从 81.43% 提高到了 94.74%。**

- 更多关于 LLM 评判的内容，可以阅读：https://www.evidentlyai.com/llm-guide/llm-as-a-judge

### 3.2 知识转移

#### 拒绝采样微调

基于我们面向 Agent 的大模型设计，我们**提出了使用大模型生成的数据对小型模型进行监督微调（SFT）**，而**不是仅仅依赖于大模型的上下文学习** （ICL）。这种方法提供了两个关键优势：

::: block-1
1. 在经过足够**高质量数据训练后，小型模型可以实现与大型模型相当的性能**；
2. 它显著**提高了推理和优化效率**。对于包含提示组件 $x$ 和响应组件 $y$ 的数据集 $D$，我们定义 $D(x, y) = \{x_i, y_i\}_{i=1}^N$，其中 $N$ 表示数据集的大小。为了最小化交叉熵（CE）损失，我们首先使用 SFT 优化，然后采用进一步优化的拒绝采样微调（RFT）。
:::

我们使用在 3.1 节中引入的 LLM 评判模型来评估由 LLM 原型生成的训练数据，过滤掉低质量数据，并生成最终的训练集。实际训练数据表示为：

$$
D_{filter}=\{(x_i, y_i)|\text{judge_data}(x_i, y_i)=1\}_{i=1}^N
$$

其中， $$
\text{judge_data}(x, y)= \begin{cases}1, & \text{如果 LLM 评判模型认为 } y \text{ 是 } x \text{ 的好答案} \\ 0, & \text{其它情况}\end{cases}
$$

#### 强化学习

> 自从 OpenAI 引入了基于人类反馈的强化学习（RLHF）[65] 以来，它已成为对齐 LLMs 的重要组成部分，这一观点在主要技术报告[66; 67; 68; 11; 69]中得到了证实。

**DeepSeek-R1[43] 提出了一种从头开始的具有成本效益的 RL 训练方法，** 该方法结合了基于规则的 ORM 和 GRPO[47]。**后续工作如 Logic-RL[8; 9; 10] 将这种高效方法应用于各种任务，使模型能够在训练期间自动生成思维链（CoT），从而提高性能[70]。**

基于我们观察到 RFT 后模型的泛化能力下降，我们尝试采用 RL 来提高模型在领域外数据集上的性能。遵循 Logic-RL[9] 的方法，我们针对特定任务优化了奖励设计（见附录 B），并进行了广泛的尝试来结合知识蒸馏（见附录 K）。

#### 蒸馏

利用大模型的函数调用相关数据和 RFT 方法，我们**通过训练将能力从大模型转移到小模型**。为了加速模型训练和推理，我们在**训练期间通过移除输入 - 输出对中的不必要内容** （包括**输入中的格式规范和输出中的思维链** （CoT）理由）来进行**提示压缩**。有关详细信息，请参见附录 C。

**知识蒸馏** [19] 是一种将知识从大模型转移到小模型的有效方法，**与 SFT 相比，它需要的数据更少**，并且**过拟合风险更低**。我们采用了基于 logits 的自适应 Kullback-Leibler（AKL）蒸馏[29]，其优化目标为 ：

$$
\mathbf{L_{\text{AKL}}}= \mathbb{E}_{(x,y) \sim D} [\text{AKL}(\pi_\phi(y|x), \pi_\theta(y|x))]
$$

$$
\text{AKL}(p, q)= \alpha_{\text{head}} \text{KL}(p, q) + (1 - \alpha_{\text{head}}) \text{KL}(q, p)
$$

$$
\text{KL}(p, q)=\sum_t p_t \log \frac{p_t}{q_t}
$$

其中 $\pi_\theta$ 表示可训练的学生模型，$\pi_\phi$ 表示教师模型，$\alpha_{\text{head}}$ 是 AKL 中的头部比例。设置 $\alpha_{\text{head}} = 1$ 得到基于前向 KL 散度的蒸馏[25]，而设置 $\alpha_{\text{head}} = 0$ 则对应于基于逆向 KL 散度的蒸馏[21]。

我们对原始方法进行了以下优化：

1. 教师校准：遵循[71]的做法，我们使用特定任务的数据对教师模型进行校准，以实现更低的学生模型损失；
2. logits 预存储：为了减少训练时间，我们预先存储教师 logits，从而实现从单个缓存的 logits 集中对多个学生模型进行蒸馏；
3. top-k 剪枝：我们通过 top-k 剪枝修剪低概率 logits，以减轻存储压力并加速训练速度。

### 3.3 模型压缩

**剪枝** ：为了进一步压缩模型，我们**按照 NVIDIA[71]、Llama3.2[72] 和 Apple** [73] 的方法**设计了两个阶段的剪枝**：

1. **离线一次性结构化剪枝**：遵循 Minitron[31] 的做法，我们探索了两种剪枝策略。
2. 恢复训练：

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/07.webp)

**量化** ：量化通常可以减少 GPU 内存消耗并提高推理性能。**在本工作中，我们对通过蒸馏 / 剪枝获得的小型模型应用了两种量化方法**：4 位仅重量化 GPTQ 算法[33]和全 8 位重量化 FP8 量化[41]。

- **4 位仅重量化** [33]**将线性权重从 FP16 压缩到每组 INT4，通过整数量化实现，并在计算过程中将它们反量化回 FP16**。这种方法在大型模型参数和低 GPU 内存带宽的内存受限场景中实现了显著的推理加速。
- **FP8 量化** [41]**将权重和激活都量化为 8 位浮点格式**。与 8 位整数量化[41]相比，FP8 在准确性方面显示出更显著的改进，并且通过利用 NVIDIA 的 FP8 张量核心，FP8 可以实现有效的推理加速[74]。

## 四、实验

### 4.1 实验设置

**数据集** ：在实验中，我们利用团队多年来在该领域的实践和经验构建了一个大规模数据集。我们将现实系统中的意图分类和语义槽框架重构为 API 文档规范，创建了一个包含 74 个功能模块和 225 个参数的函数池。

我们从单一业务场景中采样了 20 万个数据点作为训练集，并根据 3.1 节中概述的 LLM 原型合成标签。通过分层和随机抽样的组合，我们从该场景的数据中获得了 11,922 个样本作为领域内测试集。

此外，我们使用相同的方法从另一个业务场景中采样了 7,867 个数据点，作为领域外测试集，该测试集缺少训练集中的一些可调用函数。**与开源数据集相比，我们自行构建的数据集更为复杂，且更贴近现实场景（见表 1）**。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/08.webp)

---

**指标** ：**根据 3.1 节中概述的基于 LLM 的评判模型，我们建立了评估指标**。假设有一个大小为 $N$ 的数据集，表示为 $D(x, y) = \{x_i, y_i\}_{i=1}^N$，其中 $x$ 表示包含用户查询、函数列表和对话历史等元素的提示，而 $y$ 表示包含函数及其参数的响应。

函数 $\text{judge_data}$ 评估函数及其参数列表是否能够满足指定的用户查询。我们定义可实现率（Achievable Rate, AR）如下 ：

$$
\text{Achievable Rate (AR)}= \frac{1}{N} \sum_{i=1}^N \text{judge_data}(x_i, y_i)
$$

---

**训练** ：在训练阶段，所有基础模型均来自 Qwen2.5 模型家族的小型指令模型（0.5B-7B）。我们对它们进行 RFT（拒绝采样微调）和在 7B 模型上进行 RL（强化学习）。

我们在 0.5B 模型上进行剪枝，并获得 Qwen2.5-0.4B-Width 和 Qwen2.5-0.4B-Depth，剪枝比例均为 20%。然后，我们通过知识蒸馏（KD）将从 7B RFT 训练模型中获得的知识传递给剪枝后的和 0.5B 模型。此外，我们将从 7B RL 训练模型中获得的知识通过 KD 传递给 0.5B 模型，随后再进行 RL 阶段（记为 KD*+RL）。

最后，我们使用 W8A16/W4A16/W8A8 等量化策略对经过 KD 训练的 0.5B 模型进行量化，具体细节见附录 E。整个训练过程成本效益高，仅使用了 8 个 NVIDIA H20 GPU。计算资源详见附录 F。

---

**评估设置** ：我们通过与众包标注人员进行比较以及基于 LLM 作为评判（见公式 6）的指标 AR 来评估我们方法的有效性。我们还实现了一个检索模块，从函数池中选择前 10 个函数，旨在加速训练和推理阶段，具体细节见附录 D。

在现实场景中，我们在 NVIDIA L20 GPU 上部署模型，并通过 TensorRT-LLM[74] 进行有效性评估，具体配置细节见附录 H。在比较训练后的模型和基于 BERT 的系统时，我们采用每秒查询数（QPS）作为系统吞吐量的主要指标，响应时间（RT）作为端到端延迟指标，以及并发客户端数（CC）来衡量高并发性能。

### 4.2 实验结果

下表 2 是在 Qwen2.5 上训练的模型的主要结果。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/09.webp)

下图 3，可以看出成本和性能结果，其中 FLOPs 减少了 10 倍，AR 几乎无损。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/10.webp)

下表 3 是 RFT 与 RL 在领域内数据和领域外数据上的比较。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/11.webp)

#### RFT 和 KD 作为有效的知识转移方法

**如表 2 所示，经过 RFT 训练的模型通常比 LLM 原型具有更高的 AR**。具体而言，3B 和 7B 模型的性能差异可以忽略不计，两者都比 LLM 原型的 AR 提高了 0.5%。0.5B 模型的 AR 比 1.5B 模型低 0.5%，而 KD 成为 RFT 的有力补充，使 0.5B 模型的性能额外提高了 0.19%。

**在推理性能方面，0.5B 模型与 7B 模型相比仅表现出 0.4% 的 AR 差异，却实现了 13.57 倍的 QPS 提高和 91.3% 的延迟降低（见表 4）**。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/12.webp)

#### RL 的卓越泛化能力

根据下面表 3，尽管 RFT 在领域内数据上表现良好，但与 LLM 原型相比，在领域外数据上表现不佳，AR 降低了 3.61%。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/11.webp)

相比之下，**经过 RL 训练的 7B 模型在领域内和领域外数据集上均优于其他模型，表明 RL 训练方法具有更好的泛化能力**。

> 值得注意的是，即使是小至 0.5B 的模型，采用 KD+RL 的混合训练方法也能在领域内取得更好的结果。

并在领域外数据集上比 RFT 方法提高 1% 的性能。有关 RL 结果的更详细探索，请参见附录 K。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/13.webp)

#### 小型 LLM 作为基于 BERT 系统的成本效益升级

如表 2 和表 4 所示，**量化对准确性的影响微乎其微**，AR 损失不到 0.1%，同时显著提高了推理性能，在最佳情况下实现了高达 1.29 倍的 QPS 提高和 0.76 倍的 RT 降低。

**剪枝导致的准确性损失稍高**，为 0.2%，但性能始终提高了 20%。对于端到端推理，在并发客户端数为 64 时，基于 BERT 的系统达到了 1110.88 QPS 和 55.84 ms RT。尽管其成本大约是基于 BERT 系统的两倍，但 LLM 基础带来的 AR 提高了多达 14%。

## 五、分析

### 模型规模与计算效率之间的权衡

我们的实验**分析突出了模型大小与计算效率之间的引人注目的权衡**。尽管较大的模型通常表现出更优的性能，但实际差距并没有预期的那么大。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/10.webp)

**如图 3 所示，0.4B 模型的性能几乎与 7B 模型相当，仅需其 11.8% 的浮点运算次数（FLOPs）**，并实现了超过 17.4 倍的查询每秒（QPS）提升和 92.9% 的响应时间（RT）减少。

在 16 个并发客户端下，7B 模型的 RT 为 882.552 毫秒。尽管与传统的基于 BERT 的模型相比，该模型在成本上略有增加，但其源于 LLM 的基础性能提升使其成为多种应用的有益选择。

### 训练范式的特征

**表 3 中的结果强调了不同训练方法的显著特征**。微调主要关注于记忆数据，使其容易对训练集过拟合。相比之下，强化学习（RL）展现出更大的鲁棒性，因为它不仅仅拟合单个数据点。我们认为，RL 的卓越泛化能力源于其能够从由正在训练的模型自身生成的负样本中学习。**对于结构良好且明确的问题，RL 非常有效，但在面对现实任务中的模糊数据或缺乏领域知识时，其收敛性可能不佳。**

此外，将** RL 直接应用于超小型模型往往效果不佳**，这可能是由于这些模型在利用思维链（CoT）推理方面的能力有限。在这种情况下，** 知识蒸馏（KD）可以弥补这一不足**，通过** 对由 RL 训练的教师模型生成的大量 CoT 数据进行训练，使超小型模型也能掌握 CoT 推理**。与微调相比，KD 能更好地从概率分布中提取 CoT 推理。

> 因此，KD 和 RL 的混合方法可以显著提升性能，代表了未来发展的有前景的方向。

### 模型压缩策略的比较

我们的分析表明，**深度剪枝是特定领域模型的最佳方法**。深度剪枝在相同的压缩率下实现了 20% 的推理加速且无 AR 损失（表 2），**这源于深度剪枝的性质，即移除整个 Transformer 层及其相关的注意力计算**。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/14.webp)

尽管宽度剪枝在一般指令任务训练后表现出稍好的效果（表 9），但在**处理长提示时，深度剪枝在效率上更为出色，而宽度剪枝则侧重于矩阵优化** （详见附录 J）。

![文章插图](/articles/2025-05-13-cost-efficient-llms-large-super-b27c5554/images/15.webp)

在量化方面：

- **W4A16 策略在【单线程场景】中表现出色**
- **FP8 在【高并发条件下】以无损的准确性成为更优的选择，是【吞吐量部署的卓越典范】**。

## 六、结论

基于我们的综合研究，关于在实际场景中开发和部署有效的深度学习模型，得出以下关键见解：

### 超小型模型是低成本的实用替代方案

实验表明，经过**端到端优化的超小型模型能够在保持准确性的同时，实现与大型模型相当的性能水平**。这为在低成本场景下应用 LLM 提供了一条替代路径。

### 超越通用人工智能的平衡关注

尽管通用人工智能（AGI）是 LLM 的一个富有远见的长期目标，但目前进展已陷入停滞，促使人们更加关注实际应用。我们的工作表明，将 LLM 驱动的技术集成到传统任务中可以带来即时且可衡量的改进。这倡导了一种平衡的研究议程，既强调基础进展，也注重特定领域的应用。

### 可扩展的三阶段成本效益管道

所提出的包括**原型设计、知识转移和模型压缩的三阶段管道**，提供了一个**旨在在训练和推理中都保持低成本** 的同时实现高性能的**系统框架**。我们相信，这种**方法可以有效地适应其他领域**，从而可能推动这些领域的研究进展，并促进 LLM 的广泛部署。

## 七、限制与未来工作

我们的整个操作管道已经展示了最优的成本 - 性能结果，我们的超小型模型展现出了显著的能力，包括低延迟、成本效益、云独立性和强大的隐私保护。进一步的压缩可能带来的边际收益有限。因此，我们**更加关注增强这些模型的能力，并扩展其更广泛的应用，例如移动设备**。

**基于强化学习和知识蒸馏的混合方法**：**基于我们现有的实验，强化学习和知识蒸馏的混合方法已有效提升了超小型模型的泛化能力**。这表明了一种潜在强大的混合方法，**使超小型模型能够在特定领域的任务中几乎完全保留 LLM 的能力**。在未来的研究中，我们将基于我们之前的工作，对这种混合方法在将 LLM 的能力转移到超小型模型中的应用进行更深入的研究和理论分析。

**设备端部署优化**：尽管我们已经实现了 8 位量化和在线部署，但现实世界的边缘设备面临着诸如内存碎片化和神经处理单元（NPU）兼容性等异构硬件限制。**在未来，为了在边缘设备上部署超小型模型，我们将开发面向硬件的压缩算法和自适应运行时内核，以优化跨不同物联网生态系统的延迟 - 能源权衡**，并探索更资源高效的边缘设备部署技术，例如稀疏计算、以边缘为中心的量化和联邦学习。

## 参考文献

[1] Meng-Hao Guo, Zheng-Ning Liu, Tai-Jiang Mu, and Shi-Min Hu. Beyond self-attention: External attention using two linear layers for visual tasks, 2021.

[2] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022.

[3] Alan Akbik, Duncan Blythe, and Roland Vollgraf. Contextual string embeddings for sequence labeling. In Emily M. Bender, Leon Derczynski, and Pierre Isabelle, editors, Proceedings of the 27th International Conference on Computational Linguistics, pages 1638–1649, Santa Fe, New Mexico, USA, August 2018. Association for Computational Linguistics.

[4] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm, 2021.

[5] Mo Zhao, Ya Ma, Zhendong Li, and Hao Liu. Cross-dataset pose estimation of faces in the wild. In 2021 5th Asian Conference on Artificial Intelligence Technology (ACAIT), pages 718–724, 2021.

[6] Julia Gusak, Daria Cherniuk, Alena Shilova, Alexander Katrutsa, Daniel Bershatsky, Xunyi Zhao, Lionel Eyraud-Dubois, Oleg Shlyazhko, Denis Dimitrov, Ivan Oseledets, and Olivier Beaumont. Survey on large scale neural network training, 2022.

[7] Yan Li, So-Eon Kim, Seong-Bae Park, and Soyeon Caren Han. Midas: Multi-level intent, domain, and slot knowledge distillation for multi-turn nlu, 2025.

[8] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Open-reasoner-zero: An open source approach to scaling reinforcement learning on the base model. https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero, 2025.

[9] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning, 2025.

[10] Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.

[11] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.

[12] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[13] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

[14] Lutfi Eren Erdogan, Nicholas Lee, Siddharth Jha, Sehoon Kim, Ryan Tabrizi, Suhong Moon, Coleman Hooper, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Tinyagent: Function calling at the edge, 2024.

[15] Mohammad Kachuee, Sarthak Ahuja, Vaibhav Kumar, Puyang Xu, and Xiaohu Liu. Improving tool retrieval by leveraging large language models for query generation, 2024.

[16] Tom Kocmi and Christian Federmann. Large language models are state-of-the-art evaluators of translation quality. In Mary Nurminen, Judith Brenner, Maarit Koponen, Sirkku Latomaa, Mikhail Mikhailov, Frederike Schierl, Tharindu Ranasinghe, Eva Vanmassenhove, Sergi Alvarez Vidal, Nora Aranberri, Mara Nunziatini, Carla Parra Escartín, Mikel L. Forcada, Maja Popovic, Carolina Scarton, and Helena Moniz, editors, Proceedings of the 24th Annual Conference of the European Association for Machine Translation, EAMT 2023, Tampere, Finland, 12-15 June 2023, pages 193–203. European Association for Machine Translation, 2023.

[17] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[18] Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron Uribe, Evgenia Nitishinskaya, Maja Trebacz, and Jan Leike. Llm critics help catch llm bugs, 2024.

[19] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015.

[20] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

[21] Yixing Li, Yuxian Gu, Li Dong, Dequan Wang, Yu Cheng, and Furu Wei. Direct preference knowledge distillation for large language models. arXiv preprint arXiv:2406.19774, 2024.

[22] Chuanguang Yang, Xinqiang Yu, Zhulin An, and Yongjun Xu. Categories of response-based, feature-based, and relation-based knowledge distillation, 2023.

[23] Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Jian Su, Xavier Carreras, and Kevin Duh, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 1317–1327. The Association for Computational Linguistics, 2016.

[24] Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, and Jungwook Choi. Token-scaled logit distillation for ternary weight generative language models, 2023.

[25] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter, 2020.

[26] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models, 2024.

[27] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes, 2024.

[28] Gyeongman Kim, Doohyuk Jang, and Eunho Yang. Promptkd: Distilling student-friendly knowledge for generative language models via prompt tuning, 2024.

[29] Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, and Ngai Wong. Rethinking kullback-leibler divergence in knowledge distillation for large language models. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 5737–5755. Association for Computational Linguistics, 2025.

[30] Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot, 2023.

[31] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models, 2023.

[32] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8(): 8-bit matrix multiplication for transformers at scale, 2022.

[33] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers, 2023.

[34] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models, 2024.

[35] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets, 2017.

[36] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect, 2024.

[37] Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via pruning and knowledge distillation, 2024.

[38] Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. Sheared llama: Accelerating language model pre-training via structured pruning, 2024.

[39] Song Guo, Jiahang Xu, Li Lyna Zhang, and Mao Yang. Compresso: Structured pruning with collaborative prompting learns compact large language models, 2023.

[40] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration, 2024.

[41] Andrey Kuzmin, Mart Van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, and Tijmen Blankevoort. Fp8 quantization: The power of the exponent, 2024.

[42] Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. Qserve: W4a8kv4 quantization and system co-design for efficient llm serving, 2024.

[43] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.

[44] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.

[45] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024.

[46] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.

[47] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024.

[48] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12248–12267. Association for Computational Linguistics, 2024.

[49] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective and efficient reinforcement learning method for aligning large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

[50] Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models, 2025.

[51] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 16, 2023, 2023.

[52] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

[53] Jiwoo Hong, Noah Lee, and James Thorne. ORPO: monolithic preference optimization without reference model. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 11170–11189. Association for Computational Linguistics, 2024.

[54] Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

[55] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

[56] Eugene Choi, Arash Ahmadian, Matthieu Geist, Olivier Pietquin, and Mohammad Gheshlaghi Azar. Self-improving robust preference optimization. CoRR, abs/2406.01660, 2024.

[57] Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, and Mao Yang. Mutual reasoning makes smaller llms stronger problem-solvers. CoRR, abs/2408.06195, 2024.

[58] Zhaohan Daniel Guo, Gökhan Tür, Wen-tau Yih, and Geoffrey Zweig. Joint semantic utterance classification and slot filling with recursive neural networks. In 2014 IEEE Spoken Language Technology Workshop, SLT 2014, South Lake Tahoe, NV, USA, December 7-10, 2014, pages 554–559. IEEE, 2014.

[59] Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Snips voice platform: an embedded spoken language understanding system for private-by-design voice interfaces. CoRR, abs/1805.10190, 2018.

[60] Qian Chen, Zhu Zhuo, and Wen Wang. Bert for joint intent classification and slot filling, 2019.

[61] Diogo Tavares, Pedro Azevedo, David Semedo, Ricardo Gamelas Sousa, and João Magalhães. Task conditioned BERT for joint intent detection and slot-filling. In Nuno Moniz, Zita Vale, José Cascalho, Catarina Silva, and Raquel Sebastião, editors, Progress in Artificial Intelligence 22nd EPIA Conference on Artificial Intelligence, EPIA 2023, Faial Island, Azores, September 5-8, 2023, Proceedings, Part I, volume 14115 of Lecture Notes in Computer Science, pages 467–480. Springer, 2023.

[62] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021.

[63] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.

[64] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019.

[65] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.

[66] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.

[67] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[68] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023.

[69] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, Dimple Vijaykumar, Dominika Rogozińska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucińska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Görner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sébastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. Gemma 2: Improving open language models at a practical size, 2024.

[70] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022, 2022.

[71] Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Gerald Shen, Jiaqi Zeng, Zijia Chen, Yoshi Suhara, Shizhe Diao, Chenhan Yu, Wei-Chun Chen, Hayley Ross, Oluwatobi Olabiyi, Ashwath Aithal, Oleksii Kuchaiev, Daniel Korzekwa, Pavlo Molchanov, Mostofa Patwary, Mohammad Shoeybi, Jan Kautz, and Bryan Catanzaro. Llm pruning and distillation in practice: The minitron approach, 2024.

[72] Meta. Llama 3.2: Vision and Lightweight Models for Edge and Mobile Devices. https://ai.meta.com/blog/llama-3.2-connect-2024-vision-edge-mobile-devices/, January 2024. Accessed: 2025-03-14.

[73] Tom Gunter, Zirui Wang, Chong Wang, Ruoming Pang, Andy Narayanan, Aonan Zhang, Bowen Zhang, Chen Chen, Chung-Cheng Chiu, David Qiu, Deepak Gopinath, Dian Ang Yap, Dong Yin, Floris Weers, Guoli Yin, Haoshuo Huang, Jianyu Wang, Jiarui Lu, John Peebles, Ke Ye, Mark Lee, Nan Du, Qibin Chen, Quentin Keunebroek, Sam Wiseman, Syd Evans, Tao Lei, Vivek Rathod, Xiang Kong, Xianzhi Du, Yanghao Li, Yongqiang Wang, Yuan Gao, Zaid Ahmed, Zhaoyang Xu, Zhi Ouyang, Alp Aygar, Alvin Wan, Andrew Walkingshaw, Andy Narayanan, Antonie Lin, Arsalan Farooq, Brent Ramerth, Colorado Reed, Chris Bartels, Chris Chaney, David Riazati, Eric Liang Yang, Erin Feldman, Gabriel Hochstrasser, Guillaume Seguin, Irina Belousova, Joris Pelemans, Karen Yang, Keivan Alizadeh Vahid, Liangliang Cao, Mahyar Najibi, Marco Zuliani, Max Horton, Minsik Cho, Nikhil Bhendawade, Patrick Dong, Piotr Maj, Pulkit Agrawal, Qi Shan, Qichen Fu, Regan Poston, Sam Xu, Shuangning Liu, Sushma Rao, Tashweena Heeramun, Thomas Merth, Uday Rayala, Victor Cui, Vivek Rangarajan Sridhar, Wencong Zhang, Wenqi Zhang, Wentao Wu, Xingyu Zhou, Xinwen Liu, Yang Zhao, Yin Xia, Zhile Ren, and Zhongzheng Ren. Apple intelligence foundation language models, 2024.

[74] NVIDIA. Tensorrt-llm. https://github.com/NVIDIA/TensorRT-LLM, 2023.

[75] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics, 2019.

[76] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024.

[77] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

[78] YiMing Liu. Rethinking kl divergence in rlhf: From single sample to mini-batch to expectation, 2025. Notion Blog.

[79] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on GPU clusters using megatron-lm. In Bronis R. de Supinski, Mary W. Hall, and Todd Gamblin, editors, International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2021, St. Louis, Missouri, USA, November 14-19, 2021, page 58. ACM, 2021.

[80] Jian Hu, Xibin Wu, Weixun Wang, Xianyu, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance RLHF framework. CoRR, abs/2405.11143, 2024.

[81] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[82] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph Gonzalez, Clark Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs, 2024.

[83] Beijing Academy of Artificial Intelligence (BAAI). Infinity instruct. arXiv preprint arXiv:2406.XXXX, 2024.

[84] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models, 2022.
