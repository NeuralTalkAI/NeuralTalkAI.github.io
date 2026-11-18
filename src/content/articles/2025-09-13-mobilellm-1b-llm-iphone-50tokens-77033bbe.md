关键词：MobileLLM、1B参数语言模型、深度薄架构、权重共享、移动端部署、零样本推理

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/01.webp)

- MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases
- https://arxiv.org/abs/2402.14905
- https://github.com/facebookresearch/MobileLLM
- https://huggingface.co/collections/facebook/mobilellm-r1-68c4597b104fac45f28f448e
- https://huggingface.co/spaces/akhaliq/MobileLLM-R1-950M
- 本文 9257 字，阅读需 30 分钟，播客 8 分钟

相关推荐

- [ACL'25 TTFT 最高降 81.3%！MobiLoRA：端侧 LoRA-LLM 推理框架！基于上下文感知和KV缓存优化加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890383&idx=1&sn=d106a2482a5c13e1d66e1ddf37a3c698&scene=21#wechat_redirect)
- [16KB RAM跑MobileNet？分块多阶段融合方案msf-CNN：探索 TinyML 精度与延迟极致灵活性！适用广泛硬件！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888703&idx=1&sn=a7806a7e47b81b9b09c8e40b3339d3fe&scene=21#wechat_redirect)
- [端侧微调 10B LLM！MobiLLM：低精度量化与端云协同，微调同时提供推理服务！让微调内存减少 4 倍，收敛加快2.3倍!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890342&idx=1&sn=800ff8697cdb07639497ba250d04e9db&scene=21#wechat_redirect)

> 本文针对移动设备上对高效大型语言模型（LLM）日益增长的需求展开研究，这一需求主要由云成本上升和延迟问题所驱动。

我们:mdnice-bold-italic[专注于设计参数数量少于 10 亿的高质量 LLM，这一选择对于移动设备部署具有实际可行性。]

::: block-1
与“数据和参数数量是决定模型质量的关键因素”这一普遍观点相反，我们的研究强调了模型架构对十亿以下规模 LLM 的重要性。**通过采用深而窄的架构，并结合嵌入共享和分组查询注意力机制，我们构建了一个性能强劲的基准网络，将其命名为 MobileLLM。**

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/02.webp)

该模型相比之前 125M/350M 参数的最先进模型，准确率显著提升了 2.7%/4.3%。此外，我们提出了一种**即时块级权重共享方法**，该方法不会增加模型规模，仅带来微小的延迟开销。**由此产生的模型（命名为 MobileLLM-LS）** 相比 MobileLLM 125M/350M 模型，**准确率进一步提升了 0.7%/0.8%。**
:::

此外，在对话基准测试中，MobileLLM 模型家族相比以往的十亿以下参数模型表现出显著优势；在 API 调用任务中，:mdnice-bold-italic[其正确性与 LLaMA-v2 7B 模型相近，这凸显了小型模型在常见端侧使用场景中的能力。]

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/03.webp)

## 关键问题

### 问题 1：“深度优先”架构在参数超 10 亿后的可扩展性回答

> 【具体问题】本文提出亚十亿参数 LLM 中“深度比宽度更重要”的结论，与传统缩放定律（Kaplan et al., 2020）强调参数数量主导性能的观点相悖，但仅验证了 125M-350M 等亚十亿模型；那么当模型参数规模突破 10 亿后，这种“深度优先”的架构设计是否仍能保持性能优势？目前是否有初步实验证据或理论依据支撑该设计理念的可扩展性？

本文虽核心验证 125M/350M 亚十亿参数模型，但已通过将“深度优先”等设计原则（SwiGLU、分组查询注意力、嵌入共享）扩展至 MobileLLM-600M、1B、1.5B 版本，如下图。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/04.webp)

实验显示:

- MobileLLM-1.5B 在零样本常识推理任务中平均准确率达 59.4 分，较参数更多的 Qwen1.5-1.8B（平均 56.5 分）高出 2.9 分，且优于同规模的 TinyLlama-1.1B、Falcon-1B 等模型；
- MobileLLM-1B 平均准确率 57.3 分，同样显著超过同参数级别的 Pythia-1B（48.7 分）、BLOOM-1.1B（46.9 分）。

**这表明即便参数突破 10 亿，“深度优先”架构结合权重共享等策略仍能保持性能优势**，具备初步可扩展性支撑，但未进一步验证 10 亿以上更大规模（如 2B+）模型的表现，**相关实验证据集中于 1.5B 及以下版本。**

### 问题 2：块级权重共享在低端低 SRAM 设备上的有效性

> 【具体问题】MobileLLM-LS 的块级权重共享策略依赖硬件缓存（如移动端 SRAM）减少权重迁移以控制延迟，且仅在 iPhone 13 等中高端设备（SRAM 24MB）验证了“微小延迟 overhead”；若部署在 SRAM 容量更小（如仅 4-8MB）的低端移动设备上，该策略是否会因缓存无法容纳共享块权重而失效，导致延迟反而显著高于非共享模型？

本文**块级权重共享（MobileLLM-LS）的设计前提是“SRAM 能容纳单个 Transformer 块”，以避免权重在 SRAM 与 DRAM 间迁移**。文中仅在 iPhone 13（SRAM 24MB）、iPhone 15（24MB）、Google Pixel 8（8MB）等 SRAM 容量 8-32MB 的:mdnice-bold-italic[中高端设备验证，显示 MobileLLM-LS 较非共享模型仅增加 2.2%加载延迟、2.6%执行延迟。]

:mdnice-bold-italic[对于 SRAM 仅 4-8MB 的低端设备，本文未开展实验，但从设计逻辑看，若 SRAM 无法容纳单个 Transformer 块，块级共享“避免权重迁移”的核心优势将失效] ——需频繁在 DRAM 与 SRAM 间传输权重，反而可能导致延迟高于非共享模型。

此外，文中提及移动端 DRAM 通常 6-12GB，且应用需控制在 DRAM 的 10%以内，低 SRAM 设备往往伴随 DRAM 容量限制，进一步增加块级共享的适配难度，但尚无直接实验数据佐证该场景下的失效情况。

### 问题 3. API 调用任务在真实场景与真实数据集的性能

> 【具体问题】本文指出 MobileLLM-350M 在 API 调用任务中“意图与结构精确匹配分数接近 LLaMA-v2 7B”，但同时承认其 Rouge 分数（反映 agent 回复质量）显著低于 7B 模型，且测试基于 5000 样本的合成数据集；那么在真实场景中，当 API 调用需结合复杂上下文生成自然且准确的 agent 回复时，该模型的性能是否会因回复质量不足而受限？其“接近 7B 模型”的 API 调用能力在非合成的真实数据集上能否复现？

针对 API 调用任务，**本文明确其核心需求是“正确预测 API 意图与结构”** （而非 agent 回复的 Rouge 分数），MobileLLM-350M 的 EM_intent（65.3%）、EM_structure（48.8%）与 LLaMA-v2 7B（EM_intent 62.8%、EM_structure 50.9%）接近，已满足“准确调用 API”的核心目标；Rouge 分数较低（R1 46.8% vs LLaMA-v2 7B 的 56.5%）虽影响 agent 回复文本质量，但在 API 调用场景中属于次要需求。

关于真实数据集复现，本文 API 调用实验基于 5000 条训练、2500 条测试的合成数据集，**未在真实场景数据集（如含复杂上下文、真实用户交互的 API 调用数据）中验证。但文中指出“API 调用等常见移动端场景难度较低”**，且 MobileLLM-350M 在 Chat 任务（AlpacaEval 赢率 48.2%接近 GPT-3）、阅读 comprehension（RACE 高难度数据集准确率 33.8%）等下游任务中，**均展现出对“半真实/真实任务”的适配能力，间接暗示其 API 调用性能存在真实场景复现的可能性，但缺乏直接实验证据。**

## 本文目录

- 关键问题
  - 问题 1：“深度优先”架构在参数超 10 亿后的可扩展性回答
  - 问题 2：块级权重共享在低端低 SRAM 设备上的有效性
  - 问题 3. API 调用任务在真实场景与真实数据集的性能
- 一、引言
- 二、改进十亿参数以下规模的大语言模型设计
  - 2.1 训练设置
  - 2.2 构建性能强劲的基准模型
  - 2.3 层共享
- 三、实验
  - 3.1 实验设置
  - 3.2 主要实验结果
  - 3.3 下游任务
  - 3.4 与量化的兼容性
  - 3.5 知识蒸馏
  - 3.6 端侧性能分析
- 四、相关工作
  - 4.1 模型压缩（Model Compression）
  - 4.2 小模型设计（Small Model Design）
  - 4.3 神经架构搜索（Neural Architecture Search, NAS）
  - 4.4 权重共享（Weight Sharing）
  - 4.5 高效注意力机制与实现（Efficient Attention and Implementation）
- 五、结论
- 参考文献

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/05.webp)

## 一、引言

大型语言模型（LLM）正渗透到人类生活的各个方面，不仅影响着人们的沟通和工作方式，还在塑造日常娱乐体验。当前 LLM 产品的典型例子（如 ChatGPT 和 Perplexity AI）主要在云环境中运行。像 ChatGPT-4 这样的领先模型，其参数数量甚至超过 1 万亿。然而，设想未来人类在前端对话界面和后端操作（如推荐系统）中广泛依赖 LLM 的场景——这相当于个人每日时间的约 5%。在这一假设场景下，若以 50 tokens/秒的速率运行 GPT-4，需部署约 1 亿块 H100 GPU（每块 GPU 的算力为 60 TFLOPs/秒）。这一计算规模（未包含通信和数据传输开销）堪比 160 个 Meta 规模公司的算力总和，由此产生的能耗和二氧化碳排放将带来巨大的环境挑战。因此，缩小 LLM 的规模势在必行。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/02.webp)

此外，从便携性和计算成本角度考量，在智能手机和移动设备上部署 LLM 的需求日益迫切。在当前移动技术环境下，由于主内存（DRAM）容量的限制，集成像 LLaMA-v2 7B（Touvron 等人，2023b）这样的 8 位权重 LLM 成本过高，难以实现。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/06.webp)

**图 2 展示了主流移动设备中的内存层级结构**。iPhone 15 的 DRAM 容量为 6GB，Google Pixel 8 Pro 的 DRAM 容量为 12GB（Hristov，2022；Google，2023），而移动应用程序占用的 DRAM 不应超过 10%——因为 DRAM 需与操作系统及其他应用程序共享（Malladi 等人，2012）。**这一限制推动了十亿以下参数 LLM 的部署需求。**

此外，考虑到 LLM 的能耗特性（每 10 亿模型参数每 token 消耗 0.1 焦耳（Han 等人，2016；Malladi 等人，2012）），70 亿参数的 LLM 每 token 将消耗 0.7 焦耳。:mdnice-bold-italic[充满电的 iPhone 约含 50 千焦能量，在 10 tokens/秒的速率下，该模型仅能支持不到 2 小时的对话，每处理 64 个 token 就会消耗 0.2%的电池电量。]

> 这些需求共同指向一个核心目标：采用紧凑模型实现端侧部署。

- **若使用 350M 参数的 8 位模型（每 token 仅消耗 0.035 焦耳）**，iPhone 可支持全天的对话使用。
- 此外，解码速度也能显著提升——例如，125M 参数模型的基准测试结果显示其解码速度可达 50 tokens/秒，而目前最先进的 iPhone 应用 MLC Chat 使用 LLaMA 7B 模型，解码速度仅为 3∼6 tokens/秒。

基于这些考量，**本文致力于设计和实现参数数量少于 10 亿的 LLM**。为构建当前性能最优的十亿以下参数 LLM，我们做出了以下贡献：

::: block-1
- 与缩放定律（Kaplan 等人，2020）相反，我们证明**对于小型 LLM 而言，模型深度比宽度更重要**。深而窄的模型结构在捕捉抽象概念方面表现更出色，最终性能更优。
- 我们重新审视了嵌入共享方法（Zhang 等人，2022），并**在小型 LLM 中实现了分组查询注意力** （Ainslie 等人，2023），以最大化权重利用率。
- 我们提出了**即时块级权重共享方法**。在内存迁移成为延迟瓶颈的场景中，相邻两个块之间的权重共享可避免权重迁移，只需计算该块两次，且仅产生微小的延迟开销。
- 我们提出了**全新的 MobileLLM 模型家族**，该家族模型展现出最先进（SOTA）的性能。在一系列零样本任务中，MobileLLM 相比之前 125M/350M 参数的最先进模型，准确率提升了 2.7%/4.3%。
- 在对话（Chat）和 API 调用等下游任务中，MobileLLM 模型家族显著优于同等规模的模型。在 API 调用任务中，MobileLLM-350M 甚至与规模大得多的 LLaMA-v2 7B 模型达到相近的精确匹配分数。
- 我们进一步证明，我们的设计理念可有效扩展到更大规模的模型，MobileLLM-600M/1B/1.5B。
:::

## 二、改进十亿参数以下规模的大语言模型设计

> 在本节中，我们将介绍从基准十亿参数以下规模模型到新型最先进模型的演进过程（图 3）。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/07.webp)

我们对 1250 万参数（125M）和 3.5 亿参数（350M）两种模型规模进行了研究，并证明在这两种情况下模型性能均能得到持续提升。

> 对于模型大小是主要限制因素的端侧应用场景，如何有效分配有限的权重参数变得前所未有的关键。我们首先通过测试四种对十亿参数以下规模大语言模型有益的模型设计技术，提出了一个性能强劲的基准模型，将其命名为 MobileLLM。

这四种技术包括：

1. 采用 SwiGLU 前馈网络（FFN）（Dauphin 等人，2017）；
2. 采用“深而薄”（deep and thin）的架构；
3. 重新采用嵌入共享方法（Zhang 等人，2022）；
4. 采用分组查询注意力机制（Chowdhery 等人，2023）。

随后，我们提出了一种**即时块级层共享** （immediate block-wise layer-sharing）方法，该方法:mdnice-bold-italic[在不增加任何额外内存开销的情况下进一步提升了模型准确率，且在内存受限的大语言模型解码过程中仅带来微小的延迟开销]。我们将采用这种层共享方法的模型命名为 MobileLLM-LS。

### 2.1 训练设置

我们的实验在 32 块 A100 GPU 上进行，每块 GPU 的批大小（batch size）为 32。

- 我们**首先** 在 0.25 万亿个 token 上进行了 12 万次迭代的**探索性实验**。
- 随后，**对于表 3 和表 4 中报告的最优模型**，我们在 **1 万亿个 token** 上进行了 **48 万次迭代** 的训练。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/08.webp)

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/09.webp)

我们在**零样本常识推理任务上对预训练模型进行评估**，这些任务包括：

::: block-1
- ARC-easy、ARC-challenge（Clark 等人，2018）、BoolQ（Clark 等人，2019）、PIQA（Bisk 等人，2020）、SIQA（Sap 等人，2019）、HellaSwag（Zellers 等人，2019）、OBQA（Mihaylov 等人，2018）、WinoGrande（Sakaguchi 等人，2021）；
- 同时，还在问答任务和阅读理解任务上进行评估，所使用的数据集分别为 TQA（Joshi 等人，2017）和 RACE 数据集（Lai 等人，2017）。
:::

### 2.2 构建性能强劲的基准模型

#### 2.2.1 前馈网络选择

> 我们首先研究了前馈网络（FFN）中常用的激活函数，发现当前最先进的 **SwiGLU 激活函数** （Dauphin 等人，2017）对小型模型同样有益。

**将传统的前馈网络（$(FC \to ReLU \to FC)$）替换为 SwiGLU 后**，:mdnice-bold-italic[125M 参数模型在零样本推理任务上的平均性能从 42.6 提升至 43.9]。因此，在后续实验中，我们的前馈网络均采用 SwiGLU 激活函数。

#### 2.2.2 架构深度与宽度的权衡

> 该领域的一个:mdnice-bold-italic[普遍观点（Kaplan 等人，2020）认为，Transformer 模型的性能主要由参数数量、训练数据集大小和训练迭代次数决定]，而架构设计对 Transformer 模型性能的影响微乎其微。**然而，我们的研究结果表明，对于小型模型而言，情况可能并非如此。**

我们针对模型容量有限的小型模型开展的实验结果显示，在提升模型性能方面，增加模型深度比增加模型宽度更为关键。我们进行了一项广泛的研究，共训练了 19 个模型，其中 9 个模型约为 125M 参数，10 个模型约为 350M 参数。**每个模型的大小相近，但深度和宽度有所不同**。我们在 8 个零样本常识推理任务以及问答和阅读理解基准测试上进行了实验。**研究结果一致表明，“深而薄”的模型性能优于“浅而宽”的模型。**

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/10.webp)

图 4（a）和（b）显示，在大多数零样本推理任务（包括 ARC-easy、ARC-challenge、PIQA、HellaSwag、OBQA、WinoGrande）中，**深度更大的网络均表现出更优的性能**。尤其:mdnice-bold-italic[在 TQA 和 RACE 数据集上，这种趋势更为明显，如图 4（c）-（f）所示]。

我们的研究发现，对于**参数规模在 125M 左右的 Transformer 模型，具有 30 层甚至 42 层的模型性能显著优于仅有 12 层的模型**。考虑到此前大多数 125M 参数模型（Zhang 等人，2022；Black 等人，2022）的层数均限制在 12 层，这一发现令人意外。

#### 2.2.3 嵌入共享

> **十亿参数以下规模的语言模型中，嵌入层占总参数数量的比例较高**。例如，当嵌入维度为 512、词汇表大小为 32k 时，输入嵌入层和输出嵌入层各包含 1600 万参数，两者合计占 125M 参数模型总参数的 20%以上。

**相比之下，在更大规模的语言模型中，这一比例要低得多**。

- 例如，在 LLaMA-7B 模型（Touvron 等人，2023a）中，输入嵌入层和输出嵌入层的参数仅占总参数的 3.7%；
- 而在 LLaMA-70B 模型中，这一比例仅为 0.7%。

:mdnice-bold-italic[这种差异或许可以解释为何嵌入共享最初在 OPT 模型（Zhang 等人，2022）中被提出并应用，但在后续的大语言模型设计中却被忽略。]

在十亿参数以下规模语言模型的开发过程中，我们重新研究了输入-输出嵌入共享的概念。

::: block-1
- 大语言模型中的输入嵌入层将词汇表中的 token ID 映射为对应的 token 嵌入，其维度为（vocab_size，embedding_dim）；
- 相反，输出全连接层则将嵌入维度映射回词汇表上的 logits 预测，其权重大小为（vocab_size，embedding_dim）。
:::

通过**嵌入共享**，我们将输入嵌入层的权重**复用** 为输出全连接层的权重，从而得到更高效、更紧凑的模型架构。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/11.webp)

我们在一个 30 层的 125M 参数模型上进行了实验。:mdnice-bold-italic[表 1 显示，共享输入嵌入层和输出嵌入层的参数可使模型参数数量减少 1600 万（约占总参数的 11.8%），而平均准确率仅下降 0.2 个百分点]。通过将节省下来的参数重新分配以增加更多层数，这种微小的准确率下降很容易得到恢复：将模型深度增加到 32 层后，模型准确率提升了 0.4 个百分点，同时参数数量仍比原始的 135M 参数模型少 1000 万。在 350M 参数模型上也观察到了类似的结果。这些发现进一步表明，**在模型存储预算有限的情况下，嵌入共享是一种提升权重利用率和优化模型性能的有效技术。**

#### 2.2.4 注意力头与键值头数量

> 我们现在研究小型 Transformer 模型的**最优注意力头（head）大小**。选择注意力头大小时，**关键需权衡** “:mdnice-bold-italic[每个头维度可捕捉的语义信息数量] ”与“:mdnice-bold-italic[多个头可实现的非线性组合数量] ”。此外，此前大多数研究在十亿参数以下规模的语言模型中，通常设置键值头（kv-head）数量与查询头（query head）数量相同。

然而，我们发现，最初为减少大语言模型中键值缓存大小而设计的**分组查询注意力** （Chowdhery 等人，2023；Ainslie 等人，2023），同样**能有效减少小型语言模型中键值头的冗余**。:mdnice-bold-italic[分组查询注意力可被视为另一种形式的权重共享（用于权重复用）]：在这种机制下，:mdnice-bold-italic[键值头的数量为查询头数量的$1/n$（其中$n$为正整数，且查询头数量可被$n$整除），在计算注意力分数时，键值头会被重复$n$次，并与查询头的输出共同参与后续计算。]

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/12.webp)

为构建性能最先进的小型 Transformer 模型基准，我们通过实验确定了 125M 参数和 350M 参数模型的理想注意力头大小。**图 5 的结果显示，使用 16 个查询头时模型性能最佳**。此外，将键值头数量从 16 减少到 4 时，125M 参数模型的准确率基本保持不变，350M 参数模型的准确率仅下降 0.2 个百分点，而模型大小却减少了近 10%。这些结果为我们的模型架构设计提供了指导。通过采用分组查询注意力（GQA），同时增加嵌入维度以保持模型大小不变，125M 参数模型的准确率进一步提升了 0.4 个百分点，这**表明分组查询注意力是进一步挖掘小型模型潜力的有效方法。**

> 综上所述，我们测试了**四种对小型模型设计有益的最先进技术**，包括采用 SwiGLU 的前馈网络、“深而薄”的架构、嵌入共享和分组查询注意力。将这些技术相结合，我们构建了一个性能强劲的小型大语言模型基准，并将其命名为 MobileLLM。

### 2.3 层共享

2.2.2 节中关于层深度与宽度影响的研究结果表明，对于小型 Transformer 模型而言，增加层深度是更有利的选择。这:mdnice-bold-italic[促使我们研究层共享策略，旨在不增加模型存储成本的前提下增加隐藏层数量]。这种方法在模型大小是主要限制因素的端侧场景中尤为实用。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/13.webp)

令人意外的是，:mdnice-bold-italic[实验结果表明，只需重复 Transformer 块（无需修改架构或增大模型大小），就能提升模型准确率]。我们进一步**研究了三种不同的权重共享策略** （如图 6 所示）。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/14.webp)

表 2 的结果显示，在即时块级重复、全局重复和反向共享这三种策略中，**全局重复层共享策略的性能最佳。**

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/06.webp)

然而，结合硬件内存层级结构（图 2）来看，用于计算的静态随机存取存储器（SRAM）容量通常限制在 20MB 左右，该容量通常仅能容纳一个 Transformer 块。因此，:mdnice-bold-italic[将共享权重置于缓存中并立即进行两次计算，可避免在 SRAM 和动态随机存取存储器（DRAM）之间传输权重，从而提高自回归推理的整体执行速度]。基于此，我们**在模型设计中选择了即时块级共享策略**，并将采用这种层共享策略的模型命名为 MobileLLM-LS。

## 三、实验

### 3.1 实验设置

我们使用 Adam 优化器（Kingma & Ba，2014）从头开始训练 MobileLLM，权重衰减（weight decay）设置为 0.1。实验在 32 块 A100 GPU 上进行，每块 GPU 的批大小为 32。初始学习率设置为 2e-3，并采用余弦学习率衰减策略。

我们**首先** 在 0.25 万亿个 token 上进行了 12 万次迭代的**快速探索** 实验，**随后对表 3 和表 4 中报告的最优模型**，在 :mdnice-bold-italic[1 万亿个 token 上进行了 48 万次迭代的训练。]

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/08.webp)

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/09.webp)

### 3.2 主要实验结果

我们在零样本常识推理任务、问答任务和阅读理解任务上对比了模型的最终性能。为确保评估流程的一致性，基准方法的结果均通过其开源的 Hugging Face 模型进行评估。

#### 零样本常识推理

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/08.webp)

**表 3 对比了我们提出的 MobileLLM 模型与当前最先进的十亿参数以下规模模型的性能，** 这些对比模型包括早期开源的大语言模型（如 OPT（Zhang 等人，2022）、BLOOM（Scao 等人，2022））以及近期发布的模型（如 Galactica（Taylor 等人，2022）、Cerebras-GPT（Dey 等人，2023）、GPT-neo（Black 等人，2022）），还包括大语言模型分析套件 Pythia（Biderman 等人，2023）和 Transformer 变体模型 RWKV（Peng 等人，2023）。

> 在 125M 参数模型规模下，MobileLLM 显著优于 OPT、GPT-Neo、Galactica 等相同规模的现有模型**。

此外，尽管 MobileLLM-125M 的参数规模比 Pythia-160M 小 22%、比 RWKV-169M 小 26%，但其准确率分别比这两个模型高出 3.8 个百分点和 2.7 个百分点。

进一步地，在 MobileLLM-125M 中引入层共享（即 MobileLLM-LS-125M）后，模型准确率又提升了 0.7 个百分点。:mdnice-bold-italic[值得注意的是，MobileLLM-LS-125M 的性能与大多数现有 350M 参数模型相当，甚至更优]。

> 在 350M 参数模型规模下，MobileLLM 的性能比当前最先进的模型高出 4 个多百分点，且模型规模相当或更小。

为了在更广泛的内存约束条件下验证我们的设计原则，我们还将模型扩展到了 MobileLLM-600M、1B（10 亿参数）和 1.5B（15 亿参数）等配置如下图。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/15.webp)

#### 问答与阅读理解

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/09.webp)

我们在 TQA 问答基准（Joshi 等人，2017）和 RACE 阅读理解基准（Lai 等人，2017）上对预训练模型进行了评估，并遵循（Touvron 等人，2023a）中的评估设置，结果如表 4 所示。

::: block-1
- 在 125M 参数模型规模下，与前代模型相比，MobileLLM-125M 在 TQA 基准上的性能提升了 4.3 个多百分点；
- 此外，MobileLLM-350M 模型的性能比其他 350M 参数规模的模型高出约 10 个百分点。在阅读理解任务上，MobileLLM 系列模型的分数也显著高于此前的十亿参数以下规模模型。
:::

### 3.3 下游任务

> 为验证十亿参数以下规模模型在端侧应用中的有效性，我们:mdnice-bold-italic[评估了其在两个关键端侧任务中的性能：对话（Chat）和 API 调用（API Calling）。]

#### 3.3.1 对话任务

我们对 MobileLLM 模型以及从 Hugging Face checkpoint 中获取的当前最先进（SoTA）模型进行对话任务微调，并在相同设置下进行评估以确保一致性。我们采用两个基准进行评估：AlpacaEval（Li 等人，2023）（单轮对话基准）和 MT-Bench（Zheng 等人，2023）（多轮对话基准）。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/16.webp)

> 表 5 的结果显示，MobileLLM 系列模型显著优于此前最先进的十亿参数以下规模模型，甚至超过了部分 10 亿参数规模的模型。

值得注意的是，MobileLLM-LS-350M 与基准模型 GPT-3（text-davinci-001）相比，获胜率（win rate）达到了 48.2%。考虑到 GPT-3 的自获胜率（self-win rate）为 50%，**这意味着 MobileLLM-LS-350M 的对话性能已与该基准模型相当**。下图中的对话示例可视化结果也进一步证明了 MobileLLM 模型生成回复的高质量。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/17.webp)

#### 3.3.2 API 调用任务

API 调用是一种常见的端侧应用，尤其在与语音转文本模型协同实现助手功能时。利用大语言模型进行 API 调用，需要将自然语言输入转换为 JSON 配置，以调用相应的 API。例如，给定输入“帮我把闹钟设置在早上 7:30”，模型需输出{API: "alarm(time="7:30 am")"}；同时，模型还需生成智能体回复（agent response）：“好的！您的闹钟已设置为早上 7:30。”

为使大语言模型适配该任务，我们构建了一个合成数据集，包含 5000 个训练样本和 2500 个测试样本，每个样本平均包含 8 轮对话。我们使用 Adam 优化器对预训练模型在训练集上进行 4 个 epoch 的微调，学习率采用线性衰减策略，初始学习率为 2e-5，权重衰减为 0.01。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/18.webp)

> 表 6 显示，MobileLLM-350M 在意图精确匹配（$EM*{intent}$）和结构精确匹配（$EM*{structure}$）分数上与 LLaMA-v2 7B 模型相当。

其中，:mdnice-bold-italic[高意图分数表明模型能正确预测用户想要调用的 API，而结构精确匹配分数则反映了模型对 API 函数内部内容的预测能力]。尽管 MobileLLM-350M 的 Rouge 分数低于 7B 规模的模型，但需注意的是，API 调用任务的核心是正确调用 API。这一**结果表明，端侧应用中的某些常见场景并非特别复杂，像 MobileLLM-350M 这样的小型模型完全能够胜任。**

### 3.4 与量化的兼容性

> 我们进一步在 125M 参数和 350M 参数规模的 MobileLLM 和 MobileLLM-LS 模型（均在 0.25 万亿个 token 上训练）上进行了逐 token 最小-最大训练后量化（PTQ）实验。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/19.webp)

图 7 显示，:mdnice-bold-italic[采用 8 位权重 8 位激活（W8A8）的训练后量化时，模型准确率仅下降不到 0.5 个百分点，且该量化方法与层共享策略兼容。]

### 3.5 知识蒸馏

截至目前，我们通过将“下一个 token”作为硬标签，从头开始训练紧凑模型。我们还探索了以 LLaMA-v2 7B 为教师模型，对 125M 参数和 350M 参数模型进行知识蒸馏（KD）的效果。遗憾的是，**知识蒸馏会增加训练时间（训练速度减慢 2.6-3.2 倍），且模型准确率与基于标签的训练相当甚至更低**。

### 3.6 端侧性能分析

> 我们通过 ExecuTorch 在 **iPhone 13** （iOS 17.2.1）上（采用 Metal Performance Shaders（MPS）后端）**测量了 MobileLLM-125M 和 MobileLLM-LS-125M FP16 模型的延迟。**

表 7 报告了:mdnice-bold-italic[模型的加载时间、初始化时间和执行时间]，其中执行时间为 50 次迭代的平均值。

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/20.webp)

表 7 的结果表明，通过权重共享将层数翻倍后，MobileLLM-LS 的加载时间和初始化时间仅比 MobileLLM 增加 2.2%（这归因于两者模型大小相近）；得益于数据局部性，执行时间也仅增加 2.6%的开销。相比之下，在不采用权重共享的情况下将层数翻倍的模型，其加载时间和初始化时间大幅增加 143%，执行时间增加 86%。

## 四、相关工作

> 大语言模型（LLM）的优异性能推动了其广泛应用。考虑到 LLM 的计算成本与能耗，一个新的研究方向应运而生——通过缩小 LLM 规模以实现端侧推理。这些方法包括：

### 4.1 模型压缩（Model Compression）

研究人员已为 LLM 开发了多种模型压缩方法，包括剪枝（Xia et al., 2023b）、稀疏化（Sun et al., 2023；Xia et al., 2023a；Frantar & Alistarh, 2023）和量化（Liu et al., 2023a；Dettmers et al., 2022；Kim et al., 2023；Frantar et al., 2022；Xiao et al., 2023；Yao et al., 2022；Liu et al., 2023c；2023b；Frantar et al., 2022）。:mdnice-bold-italic[我们的研究与这些技术具有互补性。正如 3.4 节所证实的，我们的方法与量化技术兼容。]

### 4.2 小模型设计（Small Model Design）

已有少量研究探索了紧凑型模型架构，例如 TinyLLaMA（Timiryasov & Tastet, 2023）。然而，即便是最小的 TinyLLaMA，其参数数量也超过了 10 亿，这使得它们在许多端侧应用中仍难以部署。

部分研究在模型家族中提出了大型模型架构及其小型 LLM 变体（Zhang et al., 2022；Scao et al., 2022；Black et al., 2022；Dey et al., 2023），或开发了包含小型 LLM 变体的分析套件（Biderman et al., 2023）。

:mdnice-bold-italic[但这些模型并未在十亿参数以下的约束条件下进行优化，因此可能并非最优方案。]

### 4.3 神经架构搜索（Neural Architecture Search, NAS）

在卷积神经网络领域，尤其是视觉任务场景下，NAS 已获得广泛关注（Tan & Le, 2019；Zoph & Le, 2016；Wu et al., 2019；Guo et al., 2020）。与之相反，在 Transformer 领域，当前普遍共识认为：在参数总量不变的前提下，模型架构对精度的影响微乎其微（Kaplan et al., 2020）。

仅有少量研究针对语言 Transformer（以 BERT 模型为目标）开发了 NAS 算法（Xu et al., 2021；Jawahar et al., 2023；Ganesan et al., 2021）。

> 我们当前的研究聚焦于:mdnice-bold-italic[模型深度与宽度之间的相互作用，本质上可视为在深度空间内进行的精细网格搜索]。该研究结果对当前围绕缩放定律的主流观点提出了挑战，提出对于紧凑型 LLM 而言，**深而薄的架构能展现出更优性能。**

### 4.4 权重共享（Weight Sharing）

在固定参数约束下，权重共享是优化模型权重利用率的一种直观策略。尽管 OPT 系列模型（Zhang et al., 2022）及后续研究（Black et al., 2022）利用了输入嵌入与输出嵌入之间的权重共享，但针对 Transformer 中间层权重共享的研究仍较为有限（Shen et al., 2022；Reid et al., 2021）。

以往的研究往往需要为共享层设计专门的结构。与之不同，**我们的贡献在于提出了一种更简洁却有效的方法——只需重复 Transformer 块**，:mdnice-bold-italic[即可在模型规模固定且延迟增幅极小的前提下提升精度。]

### 4.5 高效注意力机制与实现（Efficient Attention and Implementation）

在高效 Transformer 设计领域，大量研究聚焦于通过低秩近似（Wang et al., 2020；Katharopoulos et al., 2020；Xiong et al., 2021）和稀疏注意力（Kitaev et al., 2020；Roy et al., 2021）等方法优化注意力计算过程。另一类研究则探索硬件调度与权重迁移，例如 FlashAttention（Dao et al., 2022）和 FlexGen（Sheng et al., 2023）等工作。

与之不同，:mdnice-bold-italic[我们的核心目标是在不引入新的注意力计算方式或高效硬件实现方法的前提下，优化模型规模。]

## 五、结论

> 本研究聚焦于优化十亿参数以下规模的模型，以适用于端侧应用场景。

研究结果表明，**对于小模型而言，优先考虑深度而非宽度能够提升模型性能**。此外，通过采用先进的:mdnice-bold-italic[权重共享技术包括嵌入共享、分组查询注意力（GQA）和块级权重共享]，我们在存储受限场景下显著提升了权重利用率。

由此得到的 MobileLLM 模型系列，在零样本常识推理、问答及阅读理解任务上，**相较于以往的最先进方法展现出显著优势**。

最后，我们证实了经过微调的 MobileLLM 模型**在两种主流端侧应用场景如聊天和 API 调用中的有效性**，突显了其在处理此类任务时的出色能力。

## 参考文献

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/21.webp)

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/22.webp)

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/23.webp)

![文章插图](/articles/2025-09-13-mobilellm-1b-llm-iphone-50tokens-77033bbe/images/24.webp)

加群
