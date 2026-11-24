**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：AMDKernelVault、**智能体内核优化**、HIP 内核、Triton 内核、**ROCm 生态**

> 本文这项工作说明了：没有数据，就没有好的模型；没有好的模型，就只能依赖人工和通用大模型，效率上不去，生态差距就越拉越大。此外也有三点洞察：

- **内核优化的核心是架构原生的优化范式**，只有经过原生硬件编译、执行、实测的数据，才能训练出真正懂架构的模型。:mdnice-bold-italic[语法翻译永远是二流方案，原生数据才是根本]。
- **垂直领域小模型+专用语料，效果可以超越通用大模型**。内核生成是高度垂直的领域，:mdnice-bold-italic[通用大模型再强，知识面再广，也不如经过专用语料深度微调的小模型。]
- **Agent 闭环既是数据生产工具，也是最终的部署形态**。训练的时候用 Agent 流水线生产数据，:mdnice-bold-italic[部署的时候用同样的 Agent 闭环做优化，训练和部署的范式完全统一]，模型学的就是部署时要用的能力，没有落差。

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/01.jpg)

::: image-caption
表6：与同类内核生成系统的定位对比。对比Kevin、AutoTriton、GEAK、AMDKernelVault四个系统。GEAK本身不训练模型，推理阶段反复调用前沿大模型，成本高；AMDKernelVault直接训练单个8B模型，一个模型同时承担生成器、反思器、优化器，全部在本地AMD GPU运行。奖励信号同时融合编译、数值正确性、性能延迟。
:::

::: block-1
现有的基于大模型的内核智能体大多以 CUDA/NVIDIA 为中心。这种一边倒的产业格局，使得 AMD 的 ROCm 生态长期缺乏大规模经过原生硬件执行验证的内核数据，:mdnice-highlight[无法训练出适配 AMD 架构的专用内核大模型，只能依赖通用大模型做语法层面的翻译式优化，效果不佳且部署成本高、灵活性差。]
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/02.jpg)

- **AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization**
- 组织机构：Advanced Micro Devices, Inc.（AMD）
- 论文链接：https://arxiv.org/pdf/2609.12471
- :mdnice-bold-italic[开源代码地址：https://github.com/AMD-AGI/hip_kernel_llm_lab]
- :mdnice-bold-italic[开源数据集地址：https://huggingface.co/datasets/amd/AIG-Datasets]

::: related-title
**相关推荐**
:::

::: related-links
- **[模块化 Triton GPU 内核代码生成 Agent 架构 GEAK：生成正确率达 63%，速度提升 2.59 倍！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890465&idx=1&sn=a73eec7d10391ef7371018529fb2405c&scene=21#wechat_redirect)**
- **[AutoTriton：通过强化学习实现自动Triton编程！8B参数媲美Claude-4 与 DeepSeek-R1！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890530&idx=1&sn=2f4b9743242cd200c0bae8e8f8c68e67&scene=21#wechat_redirect)**
- **[超越AutoTriton/KernelLLM的 Triton 内核生成模型 TritonRL：融合SFT蒸馏与RL分层奖励设计](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893752&idx=1&sn=e5fdd6a30c934972634fa013748e8333&scene=21#wechat_redirect)**

:::

> 为此，AMD 推出 **AMDKernelVault，这是一套面向 CDNA 架构 GPU 的开源 HIP 与 Triton 内核语料库，同时配套了完整的 Agent 化数据生产流水线与模型训练框架**。通过生成-评估-反思的智能体闭环，项目产出了 :mdnice-bold-italic[6.4 万+经过 ROCm 编译、正确性验证与延迟实测的 HIP 内核样本，以及近 4 万个 Triton 内核样本。]

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/03.jpg)

::: image-caption
图 1：AMDKernelVault整体流水线架构。展示HIPKernelGen与TritonKernelGen两套Agent数据生成流水线。输入是PyTorch参考实现，Agent循环执行：生成候选HIP/Triton代码 → ROCm编译 → AMD真机运行校验数值正确性 → 采集延迟性能数据 → 根据报错做反思修正，循环迭代直到产出合格样本。合格样本存入数据集，同时记录编译状态、正确性、延迟元数据。这套流水线是数据集的源头，和普通文本爬取最大区别：每一条样本都经过AMD硬件真机验证。该流水线可以大规模自动制造训练数据，不需要人工手写大量kernel，解决AMD生态缺少实测代码样本的痛点。
:::

基于这套语料训练的 **Qwen3-8B 模型**，多项正确性指标:mdnice-bold-italic[超越 GPT-5、Claude Sonnet 4 等千亿级前沿大模型：]

- 在 PyTorch-to-HIP 任务上达到 34.0% Pass@1
- 在 TritonBench-G 上达到 33.2% Corr@3
- 在 ROCmBench 上达到 41.94% Corr@3

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/04.jpg)

::: image-caption
表 4：固定 3 轮 GEAK OptimAgent-v2 评估协议下 TritonBench-G 测试结果。本表是论文最核心的横向对比，所有模型使用完全相同的 Agent 循环预算，变量仅为模型权重。训练后的 Qwen3-8B 在 Corr@3 达到 33.2%，超过 GPT-5、Gemini、Claude，同时带来 1.46 倍平均加速。这组结果直接证明，在 AMD 硬件上，用 AMDKernelVault 数据集训练的小模型，在 Triton 算子生成任务上超越通用闭源大模型。单纯的通用大模型缺少 AMD GPU 硬件知识，很容易写出语法能过、但在 AMD 上数值错误或访存低效的代码。该表同时给出编译准确率与最终加速，兼顾正确性和性能两个内核生成任务的核心指标，不是只看能不能编译。
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/05.jpg)

::: image-caption
表 5：固定 3 轮 GEAK OptimAgent-v2 评估协议下 ROCmBench 测试结果。ROCmBench 是 AMD 专属算子测试集，更贴近 ROCm 真实生产场景。训练后的 Qwen3-8B 取得最高 Corr@3（41.94%），优于所有对比模型。Claude 编译成功率更高，但结果正确性弱于本文模型。这一结果说明，AMDKernelVault 学到是适配 AMD MI 系列硬件的算子逻辑。通用大模型容易直接套用 CUDA 经验，在 AMD 特有 LDS、wavefront 调度场景出错。
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/06.jpg)

::: image-caption
图 3：ROCmBench 上 Pass@K 对比曲线（31 个内核任务）。本文模型在全部 K 取值下都取得最高通过率，证明在内核生成任务上拥有更优的样本效率。Pass@K 曲线衡量：生成 K 个候选 kernel，至少一个正确的概率。训练后的 Qwen3-8B 整条曲线高于所有对比模型。它的核心价值是体现样本效率：不需要大量采样候选代码，就能产出正确内核。传统 Agent 方案要生成大量候选再筛选，推理成本高；本工作把反思能力内化进模型权重，单次生成有效概率更高。曲线在 K 增大后增速放缓，也暴露局限：高难度算子无论采样多少候选，成功概率都难以显著提升，对应硬件深度调优能力存在天花板。
:::

::: block-1
这项工作不仅:mdnice-highlight[从根源上极大扩充了 AMD 生态的内核训练数据，更证明了垂直领域【专用小模型】可以在【特定任务】上反超通用大模型]，为 ROCm 生态的 AI 驱动内核优化建立了完整的基础设施闭环。
:::

## 本文目录

- 一、CUDA 垄断下的 AMD 内核优化困局
  - 1.1 语法翻译的本质局限
  - 1.2 数据缺失是根源问题
- 二、AMDKernelVault：十万级原生验证内核语料库全解析
  - 2.1 HIP/ROCm 数据集：四大子集覆盖全场景
  - 2.2 Triton 数据集：多源融合的 AMD 适配版本
  - 2.3 难度分级与质量控制
- 三、Agent 驱动的数据生产流水线：生成-评估-反思闭环
  - 3.1 统一流水线架构：两阶段闭环范式
  - 3.2 后端专属生成器：HIP 与 Triton 双流水线
  - 3.3 生产级 QA 的提取方法论
- 四、小模型也能打：SFT+执行感知 RL 的训练范式
  - 4.1 监督微调：解决冷启动的梯度稀疏问题
  - 4.2 执行感知强化学习：针对低成功率的定制设计
  - 4.3 多轮同策略反思：让模型学会自我调试
- 五、实验结果：8B 模型如何反超前沿大模型
  - 5.1 HIP 任务：正确性全面领先前沿大模型
  - 5.2 Triton 任务：AMD 专属基准优势更明显
  - 5.3 消融实验：验证每个设计的价值
- 六、技术本质与边界：这项工作到底改变了什么
  - 6.1 三个核心的本质洞察
  - 6.2 客观看待当前的局限性
- ROCm 生态的新起点

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/07.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、CUDA 垄断下的 AMD 内核优化困局

> 过去几年，基于大模型的 GPU 内核优化技术几乎全部围绕 CUDA 生态展开，从 KernelBench 到 GEAK，从 CUDA-LLM 到 Kevin Agent，所有主流数据集、训练框架和基准测试都默认面向 NVIDIA 硬件设计。

这种一边倒的局面，给 AMD 的 ROCm 生态留下了一个致命的缺口：:mdnice-bold-italic[没有大规模经过原生硬件验证的内核数据，就无法训练出适配 AMD 架构的专用大模型，只能反复调用云端通用大模型做翻译式优化，效果差、成本高、还没法本地部署。]

### 1.1 语法翻译的本质局限

::: block-1
很多人以为 CUDA 转 HIP 只是语法替换，用 hipify 工具就能搞定。实际远非如此。**CUDA 和 HIP 虽然暴露了相似的编程抽象，但高性能内核的设计深度依赖底层架构特性**。:mdnice-highlight[AMD GPU 的 wavefront 执行模式、LDS 容量与使用规则、内存合并机制、矩阵核心调度方式，以及 Triton 后端的约束条件，都和 NVIDIA 有显著差异。]
:::

> 直接通过 hipify 翻译的代码，虽然保留了 API 结构，但会沿用 NVIDIA 风格的分块常数、warp 大小假设、输入专用索引方式和内存布局，最终要么无法通过 ROCm 编译，要么在 AMD 硬件上性能低下。

举个最简单的例子：CUDA 的 warp 默认是 32 线程，而 AMD 的 wavefront 是 64 线程。:mdnice-bold-italic[如果内核里的块大小、共享内存分块都是按 32 线程设计的，搬到 AMD 上要么浪费一半的线程资源，要么出现共享内存 bank 冲突，性能直接打对折。这本质上远超语法层面的差异，是两套优化范式的根本不同。]

### 1.2 数据缺失是根源问题

要训练出懂 AMD 架构的内核大模型，就必须有大量在 AMD 硬件上编译通过、结果正确、性能经过实测的内核数据。但在 AMDKernelVault 之前，这样的数据集几乎不存在。

现有的内核数据集要么是纯 CUDA 的，要么是面向 NVIDIA 的 Triton 基准，直接拿到 AMD 上用会有大量样本编译失败或者结果错误。而生产级的 ROCm 库代码虽然质量高，但数量有限，且没有结构化的任务-内核对，不适合直接用来训练大模型。

::: block-1
**这种数据层面的差距，是 ROCm 生态在 AI 内核优化上落后于 CUDA 的核心原因**。:mdnice-highlight[没有数据，就没有好的模型；没有好的模型，就只能依赖人工和通用大模型，效率上不去，生态差距就越拉越大。]
:::

## 二、AMDKernelVault：十万级原生验证内核语料库全解析

> AMDKernelVault 的核心定位，:mdnice-bold-italic[是首个面向 AMD CDNA 架构 GPU、经过完整 ROCm 编译与执行验证的开源内核语料库，同时配套了完整的 Agent 化数据生产与模型训练框架。]

整个语料库分为两大体系：HIP/ROCm 内核数据集和 Triton 内核数据集，外加生产级 ROCm 库的 QA 监督数据，覆盖了从 PyTorch 转写、内核优化到生产级代码理解的全场景。

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/08.jpg)

::: image-caption
表1：本表统计AMDKernelVault整套数据集各个子集的样本总量，分为HIP与Triton两大分支。HIP侧包含HIP-CodeSearch、HIP-GPUMode、HIP优化样本、ROCm库QA条目，合计64530条；Triton侧包含Triton-Standard、TritonBench-G、Triton-Curated、Triton-API，合计39893条。全部HIP样本都经过ROCm编译、真机运行验证，区别于网上仅文本爬取的GPU代码数据集。该表直接证明数据集规模，也体现数据集来源多元化：开源代码、算子自动生成、ROCm生产库。这个体量填补AMD生态缺少实测kernel数据集的空白，是后续SFT与RL训练的基础。
:::

### 2.1 HIP/ROCm 数据集：四大子集覆盖全场景

> **HIP 部分总共有 64530 条数据**，其中 62153 条是经过执行验证的内核样本，2377 条来自生产级库的 QA 数据，分为四个子集。

| 数据集名称 | 描述 |
| --- | --- |
| HIP-CudaAgent | 源自 CUDA-Agent-Ops-6K 数据集，**经过 HIPKernelGen 流水线转换和验证**，最终得到 5388 条 PyTorch 到 HIP 的有效样本，**覆盖基础算子场景**。 |
| HIP-GPUMode | 源自 GPUMODE-KernelBook 数据集，包含 :mdnice-bold-italic[5910 个独立的 PyTorch 任务，生成了 22397 个 HIP 内核变体，平均每个任务有 3.8 种不同的实现，覆盖不同的块大小、线程映射策略、分块方案和内存访问顺序]。 |
| HIP2HIP | **专门面向内核优化任务的数据集**，包含 34368 条验证过的 HIP 内核优化对，也就是:mdnice-bold-italic[从一个基线内核生成优化后的等价内核，用于训练模型的优化能力。] |
| ROCm Libraries QA | 2377 条生产级 QA 数据，其中 1858 条来自 rocBLAS，519 条来自 rocSOLVER。这部分是接口级的问答监督，**覆盖计算语义、索引到张量的映射、内存访问模式、同步策略等生产级知识。** |

### 2.2 Triton 数据集：多源融合的 AMD 适配版本

> **Triton 部分总共有 39893 条经过 AMD 硬件验证的内核样本**，由四个来源融合而成，所有样本都经过统一的验证流水线处理。

| 数据集名称 | 描述 |
| --- | --- |
| Triton-Stack | 从 Stack-v2-dedup 数据集中筛选而来，过滤了约 1.3 万个候选样本，最终得到 2269 条中等难度的 Triton-PyTorch 对，**全部经过 AMD 兼容性验证**。 |
| Triton-Bench | 源自 TritonBench-8k 数据集，:mdnice-bold-italic[移除了 DSL 特有的人工痕迹，验证了 7713 个样本，确保在不同形状、步长和精度下的正确性]。 |
| Triton-GPUMode | 源自 GPUMODE-KernelBook 的约 1.8 万个条目，**将原本的 Torch Inductor Triton 代码重构为通用实现，能够处理所有形状和参数**。 |
| Triton-AICE | 源自 AI-CUDA-Engineer-Archive 的 11911 个样本，将原本面向 CUDA 的 Triton 代码转换为 AMD 验证过的实现。 |

### 2.3 难度分级与质量控制

> 整个语料库按照难度分为三级：Level 1 是独立的单函数内核，Level 2 是融合算子或单文件实现，Level 3 是带有跨模块依赖的多文件内核。

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/09.jpg)

::: image-caption
表7：各数据集样本难度分布。Triton 语料的难度分布为：29.6%的 L1、62.1%的 L2、8.3%的 L3，和实际开发中的难度分布基本一致。ROCm Libraries QA 的 L3 占比更高，因为生产级接口通常涉及主机端的调度和多文件配合。作者采用课程学习策略，训练时从简单样本逐步过渡到复杂样本，降低RL训练崩溃风险。L3样本来自rocBLAS、rocSOLVER生产库，带有真实工程中的跨文件调用、host与device交互逻辑。这种分级设计让模型循序渐进学习，避免直接学习复杂工程级代码导致的训练不稳定，也是SFT+RL训练能收敛的关键设计。
:::

质量控制上，所有内核都必须通过 ROCm 编译、数值正确性验证、延迟测试三道关卡。

- **只有编译通过、数值结果和 PyTorch 参考一致、且在 AMD 硬件上稳定运行的样本，才会被收入语料库。**
- :mdnice-bold-italic[性能导向的子集还会根据延迟实测结果，保留不同优化策略的变体。]

::: block-1
总之，AMDKernelVault 不只是简单的代码收集仓库，更是一套经过原生编译、执行验证、延迟测试的高质量结构化语料体系。:mdnice-highlight[它既覆盖了从基础算子到融合内核的全难度范围，也包含了生产级库的工程经验，从根源上解决了 AMD 生态内核大模型训练数据缺失的问题。]
:::

## 三、Agent 驱动的数据生产流水线：生成-评估-反思闭环

> 十万级的高质量语料是**靠一套完整的 Agent 化流水线自动生成、验证、筛选出来的，这才是 AMDKernelVault 最核心的技术底座。**

这套流水线统一采用:mdnice-bold-italic[“生成-评估-反思”的闭环范式，针对 HIP 和 Triton 分别实现了 HIPKernelGen 和 TritonKernelGen 两套专用流程，所有产出的内核都必须过 ROCm 编译、数值正确性验证、延迟 profiling 三道关。]

### 3.1 统一流水线架构：两阶段闭环范式

> 整个流水线分为两个核心阶段，所有任务都遵循相同的处理逻辑。

- :mdnice-bold-italic[第一阶段是 PyTorch 参考标准化。对于每个目标算子或模块，先将其转换为统一的函数式 PyTorch 参考实现。这个参考实现包含了完整的输入构造和正确性检查逻辑，确保后续不管生成 HIP 还是 Triton 内核，都用同一个基准做验证]，避免了不同来源测试标准不一致的问题。
- **第二阶段是后端专属内核合成。针对不同的后端，由专属的 Agent 生成候选内核，然后依次经过编译、正确性验证、延迟测试。** 如果失败，就由反思器根据错误信息生成改进建议，重新生成内核，直到通过或者达到最大尝试次数。

正确性验证的标准非常严格：*HIP 默认使用 torch.allclose，相对误差 1e-4，绝对误差 1e-4，支持 NaN 相等；Triton 默认相对误差 1e-4，绝对误差 1e-3，不支持 NaN 相等*，不同任务可以配置不同的公差。

### 3.2 后端专属生成器：HIP 与 Triton 双流水线

> 针对 HIP 和 Triton 的不同生态特点，两套流水线采用了不同的生成策略。

**HIPKernelGen 面向 HIP 内核，数据生成主要在 2025 年 8 月到 12 月完成，使用 GPT-5 作为生成器**。每个候选内核通过 hipcc 编译，和 PyTorch 参考做数值对比，然后测试延迟。:mdnice-bold-italic[是否保留样本只看 ROCm 上的执行结果，和生成器是什么模型无关。]

**TritonKernelGen 面向 Triton 内核，为了增加实现的多样性，使用了 GPT-oss 120B、DeepSeek-R1、Qwen2.5-32B 多个模型生成**。:mdnice-bold-italic[每个内核都通过 Triton 3.3.0 编译执行，测试约 10 种正确性配置和 3 种性能配置，覆盖不同的张量形状、步长、精度类型和边界情况。]

::: block-1
测试用例的构建也很讲究。除了继承现有数据集的测试用例外，还会用 LLM 生成更多样化的测试用例，覆盖更多形状、步长、数据类型和边界条件。:mdnice-highlight[但 LLM 生成的测试用例不会直接作为监督信号，只会作为额外的测试用例加入验证循环，避免模型过拟合特定的测试输入。]
:::

### 3.3 生产级 QA 的提取方法论

> **ROCm Libraries QA 部分的生产级数据**，是通过一套接口级的提取流程得到的，:mdnice-bold-italic[这部分是很多同类数据集没有的。]

- **首先是接口分解。从用户可见的 API 接口出发，遍历调用图，定位到和 GPU 内核执行直接相关的代码区域**，包括设备端内核、主机端启动逻辑、参数准备和紧密耦合的辅助工具。这样保留了公共 API 和底层内核之间的关联，而不是把文件当成无结构的文本处理。
- 然后是范围控制。生产库里有大量和内核行为无关的基础设施代码，比如日志、调试、测试、辅助工具。通过项目专属的允许列表和禁止列表，只遍历内核相关的子系统，避免范围膨胀。
- 最后是:mdnice-bold-italic[内核中心化 QA 生成。基于提取到的接口上下文，由 Agent 生成面向内核的问答对，问题针对具体的实现细节，答案基于真实的代码上下文]。这样得到的监督数据强调可落地的内核逻辑，不是泛泛的功能描述。

::: block-1
回顾一下，**这套 Agent 流水线的本质，是把编译器反馈、运行时正确性检查、性能 profiling 都变成了闭环里的反馈信号**，让大模型在迭代中自动学习 AMD 架构的优化规则，最终产出大规模的高质量原生数据。:mdnice-highlight[相比人工整理或者直接翻译，这种方式效率更高、质量更可控、还能自动覆盖更多边界场景]。
:::

## 四、小模型也能打：SFT+执行感知 RL 的训练范式

> 有了高质量语料，下一步就是训练专门的内核大模型。AMDKernelVault 的目标很明确：不用动辄千亿参数的云端大模型，**训练一个 8B 参数的本地小模型，就能在 Agent 闭环里同时承担生成器、反思器、优化器三个角色。**

**他们采用的是“监督微调（SFT）+ 执行感知强化学习（RL）”的两阶段训练方案**，这套方案并非全新的算法发明，:mdnice-bold-italic[但针对 AMD 内核生成的低成功率场景做了非常关键的适配设计。]

### 4.1 监督微调：解决冷启动的梯度稀疏问题

::: block-1
#### 为什么不能直接上强化学习？

**因为内核生成的初始成功率极低，基础模型生成的内核大部分要么编译不过，要么结果不对，二元奖励下大部分批次都是零梯度，模型根本学不动**。所以必须先通过监督微调，给模型打好基础，让它先能写出大概率能编译、能跑对的代码。
:::

SFT 阶段的训练样本就是语料库里的指令-响应对。对于 HIP 任务，输入是 PyTorch 任务或者原始 HIP 内核，输出是验证过的 HIP 实现，直接做指令跟随微调。对于 Triton 任务，在代码前面加上思维链推理过程，让模型先思考分块策略、掩码逻辑、边界检查、内存布局，再输出最终的内核代码。这样模型学会先规划再写代码，而不是直接生成。

值得注意的是，他们用的是**全参数微调**。:mdnice-bold-italic[原因是内核生成任务既要学习底层的语法规则，又要学习 Agent 式的自我反思行为，需要的参数更新幅度很大]，适配器的表达能力不足以支撑，在初步实验中效果明显更差。

### 4.2 执行感知强化学习：针对低成功率的定制设计

> SFT 之后，就进入强化学习阶段，用真实的执行反馈来优化模型。HIP 训练使用单轮 GRPO，Triton 训练使用多轮的 GEAK 式闭环。

:mdnice-bold-italic[这部分最核心的设计是分层奖励函数。标准的二元奖励只有对和错两种，在低成功率场景下会导致严重的梯度消失。他们设计了三级奖励：]

$$
r = r^{compile} + r^{correct} + r^{perf}
$$

其中 $r^{compile}$ 是编译成功奖励，值为 0.4；$r^{correct}$ 是正确性通过奖励，值为 1.0；$r^{perf}$ 是性能奖励，最高为 1.5，和加速比的对数平方成正比，计算公式为：

$$
r^{perf} = min\left(0.375 \cdot\left(\log _{2} s\right)^{2}, 1.5\right)
$$

这里的 $s$ 是内核相对于基线的加速比。

::: block-1
**这样设计的好处是，即使内核结果不对，只要编译成功了，也能拿到 0.4 的奖励，和完全失败的 0 分拉开差距**。这样就大大增加了有梯度信号的样本比例，解决了早期训练梯度稀疏的问题。
:::

> **另一个关键设计是禁用方差归一化**。标准 GRPO 会用组内奖励的标准差来归一化优势值，:mdnice-bold-italic[但在早期训练中，组内的奖励大多是 0 或者 0.4，方差非常小，归一化之后微小的奖励差异会被放大成巨大的梯度，导致训练崩溃。]

**他们直接用奖励减去组内均值作为优势值**：

$$
\hat{A}_{i} = r_{i} - \overline{r}
$$

虽然形式简单，但完美解决了低方差下的梯度不稳定问题。

### 4.3 多轮同策略反思：让模型学会自我调试

Triton 的 RL 训练采用多轮闭环，同一个模型同时承担生成器、反思器、优化器三个角色。每一轮中，模型看到任务提示、上一轮的代码、编译器输出、错误信息，然后生成反思和改进后的代码。

多轮的信用分配采用折扣机制，折扣因子 $\gamma=0.6$，总共 3 轮。经过归一化后，第一轮的权重约占 51%，鼓励模型一次就生成正确的代码，同时也奖励后续轮次的反思改进。

训练还采用了课程学习策略，分三个阶段逐步提升难度。

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/10.jpg)

::: image-caption
表 9：Triton RL 训练的课程学习。课程从只包含简单的 L1 样本，逐步过渡到以 L2 和 L3 为主，逐渐减少简单样本的比例，避免模型过拟合简单任务，同时确保难的任务也有足够的学习信号。
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/11.jpg)

::: image-caption
图 2：Triton 内核生成的强化学习训练结果。(a) 奖励变化曲线，展示从 SFT 基线（约 0.19）提升至峰值超过 1.0；(b) 训练过程中的基准评测结果。子图 (a) 奖励曲线直观体现训练过程奖励稳步抬升，说明分层奖励机制可以持续传递有效梯度信号，不会因大量样本编译失败导致奖励稀疏而直接训崩。子图 (b) 跟踪训练迭代中 TritonBench-G 与 ROCmBench 两套测试集的编译、正确性指标，两者同步上涨，证明学到的能力不是单纯过拟合训练分布，而是可以泛化到未见过的算子任务。该图验证课程学习搭配多轮反思 RL 的组合策略有效；同时曲线后期增速放缓，侧面说明模型能力存在上限，高难度算子上提升空间收窄。
:::

::: block-1
整体看下来，:mdnice-highlight[这套训练方案的核心洞察，是针对 AMD 内核生成“早期成功率极低、反馈信号稀疏”的特点，用分级奖励和禁用归一化解决了 GRPO 的梯度消失问题，用多轮同策略反思让模型学会自我调试]，**最终用 8B 的小模型就达到了甚至超过了云端大模型的效果**。
:::

## 五、实验结果：8B 模型如何反超前沿大模型

> 所有实验都在相同的 Agent 预算下进行对比：同样的 3 轮迭代、同样的生成/反思/优化角色分配、同样的测试基准，只换模型本身。这样的对比是受控的，能真实反映模型本身的能力差异。

实验分 HIP 和 Triton 两大部分，测试对象包括训练后的 Qwen3-8B，以及 GPT-5、Gemini 2.5 Pro、Claude Sonnet 4 三个前沿通用大模型。

### 5.1 HIP 任务：正确性全面领先前沿大模型

HIP 部分测试了两个任务：PyTorch 到 HIP 的转换，以及 HIP 到 HIP 的内核优化。

PyTorch 到 HIP 转换任务使用 200 个来自 KernelBench 的 L1 和 L2 级任务，指标是 Pass@1，也就是一次生成就正确的概率。结果如下表：

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/12.jpg)

::: image-caption
表2：PyTorch到HIP翻译任务的Pass@1指标。本表对比基线大模型、原始Qwen3-8B、SFT微调、SFT+RL四阶段模型在PyTorch转HIP算子翻译任务上的编译成功率Comp Acc与正确性Corr Acc。前沿大模型编译通过率很高，但数值正确性偏低；原始Qwen3-8B两项指标都弱。经过AMDKernelVault数据集SFT+RL训练后，8B小模型正确性达到34.0%，超过GPT-5、Gemini、Claude等前沿模型。这说明执行感知强化学习可以教会小模型掌握AMD硬件语义，而不是单纯模仿CUDA语法。短板是编译率不如Claude Sonnet4，说明部分底层语法、host/device交互逻辑仍然容易出错。
:::

HIP 到 HIP 优化任务使用 100 个留出样本，结果如下表：

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/13.jpg)

::: image-caption
表3：HIP内核自优化任务Pass@1指标。指标包含编译、正确性、Speed@10加速。SFT+RL训练后的Qwen3-8B正确性84%，非常接近GPT-5的87%，同时可以带来1.14倍平均加速。这个任务比翻译任务更考验对AMD硬件内存、wavefront调度、访存合并的理解。基础模型只有18%正确率，证明原生大模型缺少AMD硬件优化知识。该实验证明数据集不仅可以做代码翻译，也能训练模型做kernel性能调优。但加速上限低于GPT-5，说明小模型在复杂分块、LDS缓存调度这类深度硬件优化上仍有差距。
:::

### 5.2 Triton 任务：AMD 专属基准优势更明显

Triton 部分测试了两个基准：通用的 TritonBench-G 和 AMD 专属的 ROCmBench，都使用固定 3 轮迭代的 GEAK OptimAgent-v2 评估器。

> TritonBench-G 包含 184 个内核，结果如下表：

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/14.jpg)

::: image-caption
表4：固定3轮GEAK OptimAgent-v2评估协议下TritonBench-G测试结果。在完全一致的Agent循环预算下，对比各模型生成Triton算子的Corr@1/Corr@3、Comp@3、加速比。基础Qwen3-8B几乎无法生成正确算子；经过SFT+RL训练，Corr@3达到33.2%，Comp@3达到67.9%，加速比1.46倍，全面超过GPT-5、Gemini，高于Claude Sonnet4。核心原因是训练数据包含大量经过AMD MI系列GPU实测验证的Triton样本，并且多轮强化学习模拟Agent“生成-报错-反思”的闭环。但模型在极高难度算子上仍会失败，常见问题是Triton分块策略不匹配AMD wavefront与LDS存储。
:::

> ROCmBench 是 AMD 专属的基准，包含 31 个内核，更能反映在 AMD 硬件上的真实表现。结果如下表：

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/15.jpg)

::: image-caption
表5：固定3轮GEAK OptimAgent-v2评估协议下ROCmBench测试结果。ROCmBench作为AMD专属算子测试集，更贴近ROCm生态真实场景。本表中SFT+RL训练的Qwen3-8B取得最高Corr@3（41.94%），超过Claude与GPT-5；但Claude的编译成功率Comp@3和速度指标更高。这说明模型在保证计算结果正确这件事上优势明显，但生成的代码有时会出现编译语法、硬件资源分配问题，导致编译失败，且最优kernel的性能上限不如顶级闭源大模型。ROCmBench的结果非常关键，证明这个数据集训练出来的模型，在AMD特有算子场景具备差异化优势，而不是只能在通用算子测试集刷指标。
:::

### 5.3 消融实验：验证每个设计的价值

> 论文还做了详细的消融实验，验证了各个设计的必要性。

数据规模消融显示，**只用 10%的训练数据，效果就已经超过基础模型，并且性能随着数据量增加稳步提升，证明了数据集的高质量。**

Triton 组件消融显示，去掉多轮反思，正确率下降 13.1 个百分点；去掉 SFT 冷启动，正确率下降 24.7 个百分点。**这说明 SFT 是基础，多轮反思是关键提升，两者缺一不可。**

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/16.jpg)

::: image-caption
表 11 展示了针对Triton训练框架的消融实验，使用智能体训练评估器（GEAK OptimAgent-v2）在TritonBench-G（包含184个内核）上经过3轮迭代完成评估。
:::

奖励设计的消融也验证了分级奖励和禁用归一化的重要性，这两个设计共同解决了低成功率场景下的梯度消失问题，是 RL 能够生效的核心。

::: block-1
总之，:mdnice-highlight[在相同的 Agent 预算下，8B 的小模型经过 AMDKernelVault 的语料训练之后，在 HIP 和 Triton 的多项正确性指标上都超过了 GPT-5、Claude 等千亿级前沿大模型]。虽然在编译通过率、极致性能优化上还有差距，但考虑到参数规模差了两个数量级，还能本地部署，**这个结果已经非常有产业冲击力**。
:::

## 六、技术本质与边界：这项工作到底改变了什么

> 很多人会说，不就是做了个数据集吗？其实不然，AMDKernelVault 真正的价值，是打通了“原生数据生产-专用模型训练-本地 Agent 部署”的完整链路，给 AMD 生态建立了一套内核优化的自循环体系。

在此之前，AMD 的内核优化要么靠人工手写，要么靠翻译 CUDA 的成果，要么反复调用云端大模型试错，始终没有自己的数据驱动的优化闭环。这套体系补上了最关键的基础设施。

### 6.1 三个核心的本质洞察

> 透过现象看本质，这项工作背后有三个非常重要的洞察。

- 第一个洞察：**内核优化的核心是架构原生的优化范式**。CUDA 的分块策略、warp 调度、内存访问模式，放到 AMD 架构上就是水土不服。:mdnice-bold-italic[只有经过原生硬件编译、执行、实测的数据，才能训练出真正懂 AMD 架构的模型]。语法翻译永远是二流方案，原生数据才是根本。
- 第二个洞察：**垂直领域小模型+专用语料，效果可以超越通用大模型**。内核生成是高度垂直的领域，通用大模型再强，知识面再广，也不如经过专用语料深度微调的小模型。而且小模型可以本地部署，不用反复调用 API，成本更低、延迟更低、数据更安全。
- 第三个洞察：**Agent 闭环既是数据生产工具，也是最终的部署形态**。训练的时候用 Agent 流水线生产数据，部署的时候用同样的 Agent 闭环做优化，训练和部署的范式完全统一，模型学的就是部署时要用的能力，没有落差。

### 6.2 客观看待当前的局限性

当然，这项工作也有明确的边界。

**首先，专家级内核依然很难**。:mdnice-bold-italic[TritonBench-G 中最难的 D5 级内核，所有模型都解不出来] 😅。常见的失败模式包括：针对特殊张量形状的块大小选择错误、多步归约中的同步错误、AMD wavefront 执行模式下的内存合并失败。

其次，性能优化能力还有差距。**在 HIP 优化任务上，模型的加速比还不如 GPT-5**；在 ROCmBench 上，加速比也低于 Claude Sonnet 4。:mdnice-bold-italic[说明现在的训练更侧重正确性，极致的性能优化还需要更丰富的 profiler 反馈和更多专家级内核数据。]

另外，评估集的规模有限，主要覆盖深度学习算子，HPC、稀疏计算、图计算等场景还没有覆盖。**语料库也没有做全量的语义去重审计，可能存在功能相近的样本。**

## ROCm 生态的新起点

> AMDKernelVault 的开源，对整个 ROCm 生态来说是里程碑式的事件。**它第一次让 AMD 生态拥有了和 CUDA 同量级的内核大模型训练基础设施，也第一次证明了小模型可以在垂直领域反超通用大模型**。对于开发者最直接的好处就是以后做 AMD 内核优化，不用再反复调用云端大模型，:mdnice-bold-italic[本地跑一个 8B 模型就能搞定大部分场景，成本更低、速度更快、还不用担心数据隐私。]

以前写 AMD 内核需要懂架构、懂编译器、懂算子优化的资深工程师，现在普通开发者也可以用 AI 工具快速生成可用的内核，再人工调整极致性能🤣。:mdnice-bold-italic[这会吸引更多开发者加入 ROCm 生态，丰富生态的软件栈，逐步缩小和 CUDA 生态的差距。]

对于其他硬件平台比方国产 GPU，这套方法论提供了完整的参考。**怎么从零搭建一套内核大模型的训练体系，怎么设计 Agent 化的数据生产流水线，怎么适配低成功率的训练场景，都可以直接借鉴。**

::: block-1
**未来这个方向还有很大的想象空间**。

- 更大规模的语料库，覆盖更多算子家族和应用场景；
- 更丰富的性能反馈，把内存占用、寄存器压力、占用率都加入奖励；
- 更强大的小模型，进一步拉近和专家的差距；甚至跨平台的内核优化模型，一套模型适配多种硬件。

但无论如何，:mdnice-highlight[AMDKernelVault 都是一个标志性的开始，从它开始，AMD 的 GPU 内核优化正式进入了数据驱动的智能体时代。]
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 开源 hip-ep：用 MLIR 贯通 ONNX Runtime 与 ROCm，落地 LLVM bitcode 编译](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905194&idx=1&sn=2f1918c0623bc9f0918a730bcb28a992&scene=21#wechat_redirect)**
- **[无源码改写汇编优化 GPU Kernel！AMD 联合南方科技大学提出 AsmEvo，让已编译 GPU 内核最高提速 3.88 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905090&idx=1&sn=53df28ba322bdc929e8df59b9f391f98&scene=21#wechat_redirect)**
- **[34.93 倍加速！MLSys 26 DSA 智能体赛道冠军方案，一套规则手册与三个子智能体替代人类内核工程师](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901644&idx=1&sn=ddd845b9bc6fd48c0c73a0ce535f45e0&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[Triton 首次生产级落地自研 ASIC：Meta MTIA-2i 上 60 种模型、50% 的层与 47% 非 GEMM 时间被攻占](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904593&idx=1&sn=48f27e651841785d00dcf73aaf58a55f&scene=21#wechat_redirect)**

:::
