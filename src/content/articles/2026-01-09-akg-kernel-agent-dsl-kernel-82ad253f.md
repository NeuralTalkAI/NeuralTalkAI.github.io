关键词：**Kernel Generation**、:mdnice-bold-italic[AI for AI System]、**Multi-Agent System**、:mdnice-bold-italic[Code Optimization]、**Hardware Acceleration**

> 近年来，随着大模型、多模态架构和推荐系统日益复杂，加上稀疏化、量化等技术的广泛应用，AI 模型对底层计算内核的性能要求越来越高。

与此同时，硬件更新频繁，芯片架构多样，不同平台往往需要专门优化的内核实现。

::: block-1
手写优化已经跟不上需求，**内核开发正在成为 AI 系统发展的瓶颈**。大模型代码生成能力的突破，为自动化内核开发带来了新希望。但如果让通用大模型直接生成内核，往往会编译出错、数值不对，甚至性能还不如基础版本。
:::

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/01.webp)

- **Akg Kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis**
- https://arxiv.org/pdf/2512.23424
- **代码：https://atomgit.com/mindspore/akg**
- 5000 字，阅读 18 分钟，**播客 11 分钟**

相关推荐

- **[ProfilingGuided+LLM协同：TritonForge突破 Triton内核优化瓶颈，成功率42.7%且最高提速5倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894462&idx=1&sn=ba65fbb1cade2e2982477a013f23beda&scene=21#wechat_redirect)**
- **[超越AutoTriton/KernelLLM的 Triton 内核生成模型 TritonRL：融合SFT蒸馏与RL分层奖励设计](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893752&idx=1&sn=e5fdd6a30c934972634fa013748e8333&scene=21#wechat_redirect)**
- **[MultiKernelBench：首个覆盖英伟达GPU、华为NPU、谷歌TPU的Kernel生成基准，涵盖14类算子285项测试](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891504&idx=1&sn=d1558faf582d556f3ac231f985efc057&scene=21#wechat_redirect)**

> 本文分享华为技术有限公司与湖南大学合作的最新研究成果——**AKG Kernel Agent（AI-driven Kernel Generator）**。

聚焦于通过多智能体（Multi-Agent）技术解决现代 AI 系统中内核开发的效率、跨平台适配与性能优化难题，实现内核的自动化生成、迁移与性能调优。

## 本文目录

- 01 内核生成：AI 时代的性能瓶颈
- 02 核心创新：多智能体协同作战
- 03 统一草图：硬件无关的优化蓝图
- 04 文档驱动：灵活支持新平台
- 05 检索增强：精准找到最佳范例
- 06 迭代优化：向性能巅峰进化
- 07 性能评估：正确性与速度双优
  - 正确性评估
  - 性能评估
- 08 未来展望：挑战与方向
- 09 结语

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/02.webp)

## 01 内核生成：AI 时代的性能瓶颈

> 高性能计算内核（Kernel）是 AI 模型高效运行的基石。它们负责在 GPU、NPU 等硬件上执行具体的张量运算，如矩阵乘法、卷积、注意力计算等。

一个优秀的计算内核，既要理解算法逻辑，又要精通硬件特性：内存层次、并行执行模型、指令集、线程块大小、访存模式……

这些都需要开发者通过大量实验反复调优，过程耗时费力，且结果往往与特定硬件强耦合。

以 FlashAttention 为例，其针对新架构的优化常常滞后于硬件发布数年。

高级 DSL（领域特定语言）如 Triton，虽然提升了抽象层次，但仍需专家经验才能在不同硬件上高效映射算法，且往往难以完全弥合 **“性能”** 与 **“可移植性”** 之间的鸿沟。

与此同时，**大模型（LLM）在代码生成方面展现出惊人潜力**。GPT-4、DeepSeek-R1 等模型已能生成语法正确、功能有效的代码。这催生了利用 LLM 自动生成高性能内核的研究。

::: block-1
但通用大模型常因不熟悉内核编程的语义和性能约束，生成错误或低效代码；而专门微调的模型（如 Meta 的 KernelLLM、Stanford 的 Kevin-32B）又面临高质量内核数据稀缺的问题。

如何让 AI 既懂算法又懂硬件，生成既正确又高效的内核？华为与湖南大学的研究团队提出了一个全新的解决方案：**AKG Kernel Agent**。
:::

## 02 核心创新：多智能体协同作战

> AKG Kernel Agent 是一个**多智能体系统**，通过结构化协作实现内核的自动生成、迁移与性能调优。:mdnice-bold-italic[它不像传统方法那样依赖单一模型“一步到位”，而是将复杂任务拆解，交给四个各司其职的智能体协同完成。]

整个系统的架构如下图所示，清晰展示了各智能体的分工与协作流程：

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/03.webp)

::: image-caption
图 1：AKG Kernel Agent 的整体架构。Conductor（协调器）作为核心调度者，协调 Designer（设计器）、Coder（编码器）和 Verifier（验证器）代理形成迭代优化循环。该系统借助专家知识库、优化核数据库以及多个 LLM 后端，生成高性能的核代码。此架构遵循 “策略与实现解耦”“文档驱动可扩展性”“闭环优化” 三大原则。Designer 负责生成含并行策略等的 Unified Sketch，Coder 将其转为 Triton 等目标 DSL 代码，Verifier 验证正确性与性能，三者在 Conductor 调度下，支撑从算子规格到优化核的全生命周期自动化生成
:::

这四个智能体分别是：

1. **Designer（设计师）**：分析硬件规格和算子需求，生成**统一草图（Unified Sketch）**。这是一种硬件高效但独立于具体 DSL 的中间表示，用于捕获并行化策略、数据流和内存访问模式。
2. **Coder（程序员）**：将统一草图翻译成目标 DSL（如 Triton、CUDA-C、TileLang 等）的可执行代码。
3. **Verifier（验证器）**：在目标硬件上验证生成内核的正确性，并分析性能。
4. **Conductor（调度器）**：监控工作流，通过自适应重路由处理错误。实现错误发送给 Coder 修正，结构性问题则返回给 Designer 重新设计。

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/04.webp)

::: image-caption
图 3：Conductor（协调器）的动态工作流调度。Conductor 分析验证结果，将任务路由至对应代理并提供针对性反馈，实现高效的错误解决。Conductor 先通过 LLM 分析错误类型：语法错误、API 误用等实现类问题，会路由给 Coder 并提供修改建议；算法逻辑、内存访问等设计类问题，则转给 Designer 调整。它还维护执行历史，避免固定流水线的无效迭代，大幅提升核生成成功率
:::

这种**策略与实现解耦** 的设计是核心创新之一。

它将高层优化决策（Designer 负责）与底层代码合成（Coder 负责）分离，降低了单次 LLM 调用的认知负担，提高了生成过程的可解释性。

统一草图是关键枢纽，它用硬件无关的方式描述“做什么”，再由 Coder 基于目标平台的文档将其转化为“怎么做”，从而实现了**一次设计，多平台部署**。

## 03 统一草图：硬件无关的优化蓝图

> 什么是统一草图？它是一种极简的领域特定语言，专门用于以 LLM 易于理解、Coder 易于实现的形式表达算法意图。

其设计遵循四大原则：

- **最小化原语**：仅包含 alloc（分配）、load（加载）、store（存储）、compute（计算）等基本操作。
- **统一语法**：所有操作采用函数调用风格。
- **标准控制流**：使用 Python 的 for/range 语法。
- **提示分离**：通过@llm_hint 装饰器表达复杂优化提示，而不污染主逻辑。

形式化定义中，一个草图 $\mathcal{S}$ 包含四个部分：

$$
\mathcal{S}=(\mathcal{D},\mathcal{O},\mathcal{C},\mathcal{H})
$$

其中：

- $\mathcal{D}$：声明（符号变量、张量规格、编译时常量）
- $\mathcal{O}$：核心操作（alloc/load/store/compute）
- $\mathcal{C}$：控制流（嵌套 for 循环，带显式分块）
- $\mathcal{H}$：优化提示（通过@llm_hint 指定并行级别、内存层次、循环变换等）

下图展示了一个用于 RMSNorm 算子的统一草图示例：

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/05.webp)

::: image-caption
图 2：RMSNorm 算子的 Unified Sketch（统一框架）示例。该框架展示了：（1）包含符号与张量的声明；（2）带有 @llm_hint ("parallel", "coreidx") 装饰器的嵌套 for 循环，用于并行化；（3）使用 alloc 进行内存操作，附带 “accumulator”（累加器）、“input_cache”（输入缓存）、“output_buffer”（输出缓冲区）等语义提示；（4）用于 RMS 归一化的计算操作（乘法 mul、加法 add、除法 div、平方根 sqrt）。Unified Sketch 是硬件无关的中间表示，由 D（声明）、O（核心操作）、C（控制流）、H（优化提示）构成。示例中 “fast”“fastest” 等语义提示，能让 Coder 精准映射到寄存器、共享内存等硬件资源，确保同一框架可跨 GPU、NPU、CPU 等后端移植
:::

草图`alloc`中的语义提示（如"fastest"表示寄存器，"fast"表示共享内存/L1 缓存）是硬件无关的。

Coder 根据目标硬件的文档将其映射到具体资源。这种抽象使得同一草图可被翻译到 GPU（映射到 grididx/threadidx）、NPU（映射到 coreidx）或 CPU（映射到 OpenMP/SIMD）后端。

## 04 文档驱动：灵活支持新平台

> 另一个关键创新是**文档驱动的集成框架（DDI）**。传统的系统要为每种 DSL-硬件组合硬编码支持，工程量大且难以维护。

AKG Kernel Agent 则将**文档视为系统扩展的一等公民**。它定义了一套标准化文档规范（DocSpec），将所需知识分为四类：

1. **基础文档**：核心概念、语法、执行模型。
2. **API 文档**：可用接口的签名和使用模式。
3. **专家建议**：性能优化策略和常见陷阱。
4. **参考示例**：按前端框架组织的工作实现。

要集成新的 DSL 或硬件目标，用户只需按照 DocSpec 结构准备文档。系统在运行时自动加载并利用这些知识，无需修改核心代理逻辑。

对于冗长的文档，系统采用基于 LLM 的压缩，根据算子规格和统一草图动态提取与当前任务最相关的 API 子集，避免在每次提示中包含全部内容。

## 05 检索增强：精准找到最佳范例

> 代码生成中，检索相关示例至关重要。但内核代码检索面临独特挑战：形状或分块配置的细微差异可能导致语义迥异；而语义相似的优化可能表现为语法多样的实现。

AKG Kernel Agent 采用了**分层检索机制**，逐步缩小搜索空间，流程如下图所示：

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/06.webp)

::: image-caption
图 4：分层代码检索流程。系统通过三步逐步缩小候选集：基于计算逻辑的语义搜索、基于 DSL / 后端 / 算子类型的硬过滤、基于形状兼容性的语义匹配。该流程解决了标准 RAG 在核代码检索中的缺陷（如表面相似代码语义差异大）。先提取算子特征生成嵌入做语义搜索，再过滤出适配目标平台的候选，最后按张量形状嵌入评分选 Top-k 示例，为 Coder 提供符合语法规范且含优化策略的参考代码
:::

具体步骤如下：

1. **特征提取**：使用 LLM 提取查询算子的计算特征。
2. **语义搜索**：将特征转化为嵌入向量，通过余弦相似度在数据库中检索计算逻辑相似的候选代码。
3. **硬过滤**：基于 DSL、后端、算子类型进行精确匹配，确保示例适用于目标平台。
4. **形状匹配**：对过滤后的候选代码进行形状兼容性排序，返回最相关的 Top-k 示例。

这种分层方法确保了检索到的示例既在语义上相关，又在结构和平台上适用，为 Coder 提供了高质量的参考。

## 06 迭代优化：向性能巅峰进化

> 生成正确的内核只是第一步，达到峰值性能还需要探索多种优化策略。

AKG Kernel Agent 将其建模为搜索问题，并采用**基于迭代搜索的优化** 策略。系统通过多轮并发生成、执行和分析，持续进化内核性能。

为了避免早熟收敛，系统引入了**岛屿模型**：将种群划分为 K 个半独立的岛屿，定期让精英个体在岛屿间迁移，既传播成功策略，又保持多样性。

$$
\text{Island}_{i}.\text{population}\leftarrow\text{Island}_{i}.\text{population}\cup\text{Elite}(\text{Island}_{i-1})
$$

在每轮优化中，系统会:mdnice-bold-italic[对高性能和低性能实现进行对比分析，识别出有效的分块方案、内存模式和并行化选择，从而指导下一轮的优化方向。]

## 07 性能评估：正确性与速度双优

> 为了全面评估，团队构建了自己的评测基准，包含 198 个（动态形状）/214 个（静态形状）算子，涵盖元素级操作、归约、归一化、张量操作、矩阵乘法、索引、排序和融合操作八大类别。

该基准的一个显著特点是**系统化动态形状测试**，确保生成的内核在实际部署中鲁棒。

### 正确性评估

在权威测试集 KernelBench Level 1 上，AKG Kernel Agent 在 Triton-CUDA 组合上实现了**100%的 Pass@4 准确率** （在所有算子类别上），展现了卓越的代码生成正确性。

具体分类表现如下表所示：

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/07.webp)

::: image-caption
表 1：KernelBench Level 1 正确性评估（Pass@4），按算子类别细分
:::

在自建更具挑战性的动态形状基准上，系统也表现优异：

- Triton-CUDA: **90.9%** (动态) / **87.4%** (静态) 总体通过率
- Triton-Ascend: **85.4%** (动态) / **82.2%** (静态) 总体通过率

### 性能评估

在性能方面，与 PyTorch Eager 基线相比，AKG 生成的内核带来了显著加速。评估使用几何平均来聚合加速比，避免异常值带来的偏差。

$$
\text{Geom. Mean}=\left(\prod_{i=1}^{n}s_{i}\right)^{1/n}=\exp\left(\frac{1}{n}\sum_{i=1}^{n}\ln s_{i}\right)
$$

关键性能数据如下表所示：

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/08.webp)

::: image-caption
表 3：KernelBench Level 1 上三个 DSL-后端组合的性能评估（原论文 Table 3）_  _速度计算为 $T_{\text{baseline}}/T_{\text{generated}}$，基线为 PyTorch Eager。GM：几何平均加速比。Fastₚ (%)：加速比 ≥p 的内核百分比。卷积内核仅对 CPP-CPU 进行评估（Triton 后端不支持）
:::

其中，Triton-Ascend 实现了**1.46 倍的总体几何平均加速**，在归约与归一化类内核上表现尤为突出（1.66 倍）。

CPP-CPU 在扫描与损失类内核上更是实现了惊人的**9.00 倍加速**。这是因为 PyTorch Eager 通常将此类操作分解为多个小算子执行，而 AKG 生成的内核将其融合，消除了中间内存访问和内核启动开销。

下图更直观地展示了不同后端在不同内核类别上的加速比分布：

![文章插图](/articles/2026-01-09-akg-kernel-agent-dsl-kernel-82ad253f/images/09.webp)

::: image-caption
图 5：KernelBench Level 1 上三个 DSL-后端组合的每内核加速比分布按内核类别分组，并列比较 Triton-Ascend（蓝）、Triton-CUDA（绿）和 CPP-CPU（红）。箱线图显示四分位距（IQR），中位数为水平线。红色虚线表示加速比=1.0（PyTorch Eager 基线）。注意，卷积内核仅适用于 CPP-CPU，因为 Triton 后端在当前评估中不支持卷积操作
:::

## 08 未来展望：挑战与方向

> 尽管成果显著，AKG Kernel Agent 仍面临一些挑战：

- 复杂的融合操作和专用内核仍是难点。
- 对非 Triton DSL 的卷积支持有限，反映了将此类操作映射到不同编程模型的内在复杂性。
- 性能优化基础设施有待进一步完善。

展望未来，团队指出了几个方向：

1. **引入强化学习**，基于性能反馈更智能地指导搜索过程。
2. **扩展统一草图语言**，以捕获更复杂的优化模式（如张量核利用、异步执行）。
3. **开发自动化技术**，从现有代码库生成高质量文档，加速新目标的集成。
4. **探索微调策略**，利用 AKG 自身生成的内核来提升基座 LLM 的内核编程能力。

## 09 结语

> AKG Kernel Agent 展示了**多智能体协作、文档驱动的知识集成以及系统化搜索** 如何实质性推动自动化内核生成的边界。

它将高深的硬件优化知识封装在智能体协作的流程中，让开发者能够更专注于算法创新，而非底层实现的繁琐调优。

通过开源该系统及其基准测试，研究团队希望能加速弥合算法创新与硬件优化实现之间的鸿沟，为 AI 计算的下一波性能突破提供自动化引擎。当 AI 开始为自己编写更高效的“发动机”代码时，我们距离真正的智能计算又近了一步。

更多推荐

- [NPUEval：评估AMD硬件LLM向量化NPU Kernel——102个机器学习算子、编译器反馈与 50%+向量化峰值效率](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891826&idx=1&sn=313200a89680b665f85f56718ab3f7af&scene=21#wechat_redirect)
- [NVIDIA ComputeEval：从基准建立到难度升级，全面评估 LLMs 的 CUDA 代码生成能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894186&idx=1&sn=4bcf0b3458bf4a638fc3d18efc85e67c&scene=21#wechat_redirect)
- [CUDA Kernel 自动生成新范式！多轮RL+反馈迭代方案 Kevin 让生成速度超o4-mini 41%、正确性领先44%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891083&idx=1&sn=511eaf94b8c3948457d107d8e8853641&scene=21#wechat_redirect)

加群
