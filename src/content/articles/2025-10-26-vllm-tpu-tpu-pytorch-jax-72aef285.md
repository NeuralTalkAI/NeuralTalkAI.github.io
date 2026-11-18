# 关键词：vLLM、Google TPU、大模型推理、适配优化、分布式部署、性能提升

# ![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/01.webp)

- vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU
- https://blog.vllm.ai/2025/10/16/vllm-tpu.html
- 8000 字，阅读 40 分钟，播客 13 分钟

相关推荐

- [PD 多路复用：SGLang 借助 NVIDIA GreenContext 释放高有效吞吐量的大语言模型服务能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892879&idx=1&sn=fdd05fbc29cd1001cd1479034c4b0930&scene=21#wechat_redirect)
- [训练即部署！SpecForge：加速 SGLang 推测性解码训练，让16GB GPU支持70B模型训练，加速比达2.18倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892155&idx=1&sn=5c2c834d047a75a4da3e1b52ff14a30d&scene=21#wechat_redirect)
- [大模型 Scaling Law 的发展历程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887535&idx=1&sn=b7deccfd6753760b21d6bece2962b676&scene=21#wechat_redirect)

> 2025 年 10 月，谷歌团队推出 vLLM TPU 新统一后端，:mdnice-bold-italic[基于 tpu-inference 硬件插件实现 PyTorch 与 JAX 的统一支持]，**通过 JAX→XLA lowering 路径重构架构，在性能、兼容性和扩展性上实现重大突破。**

该后端彻底**重写 TPU 底层，采用 SPMD 编程模型替代原 MPMD**，自动实现张量分区与通信计算重叠，搭配 Ragged Paged Attention V3（RPA v3）内核，支持任意模型规格、量化类型和张量并行，融合 KV 缓存更新操作隐藏延迟，适配预填充、解码等混合批次场景，吞吐量较 RPA v2 提升 10%。

![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/02.webp)

PyTorch 模型**无需代码修改即可通过 Torchax 适配 TPU**，JAX 模型原生支持，安装仅需单条命令，默认调用 tpu-inference 优化模型，**无适配成本。**

::: block-1
性能方面，Llama 3.1-8B 在 v6e-1 上吞吐量提升 3.6 倍，70B 版本在 v6e-8 上提升 2.1 倍，**较 2025 年 2 月原型性能最高提升近 5 倍。**

- 目前支持 Trillium（v6e）、v5e 等 TPU 型号
- 覆盖稠密型、多模态模型
- 提供前缀缓存、结构化解码、量化等核心功能
- 实验性支持 v5p、多 LoRA 等特性。
:::

![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/03.webp)

**当前仍存在局限，XL MoE、视觉编码器等复杂模型需持续优化，v5p 需额外调优**。未来将推进稀疏核卸载、TPU 友好型 MoE/MLA 算子开发，完善 RL 集成、分布式解耦部署等功能，定期更新开源验证模型列表，持续联动谷歌云服务与开源社区。

## 关键问题

### 问题 1：vllm 对 TPU 的适配是否真正解决了 TPU 生态中算子支持不足的核心痛点，其与 GPU 部署的迁移成本差异如何？

- 已针对性解决算子支持不足痛点：通过统一 JAX→XLA lowering 路径，复用 JAX 成熟高性能原语，搭配 RPA v3 内核（支持任意模型规格、量化 dtype、张量并行），覆盖更多模型和场景，同时提供 Collective Communication Matmul、量化 Matmul 等 TPU 友好算子，**补齐了过往 TPU 生态算子适配局限。**
- **迁移成本显著低于 GPU 部署**：PyTorch 模型无需任何代码修改即可通过 Torchax 在 TPU 上高效运行，安装仅需单条`pip install vllm-tpu`命令，依赖统一无额外配置成本；而 GPU 部署常需适配不同框架的硬件插件、调整内核参数，迁移复杂度更高。

---

### 问题 2：同等硬件投入下，vllm+TPU 的性能提升（吞吐量、延迟）在真实高并发场景中能否稳定复现，是否存在场景局限性？

- **真实高并发场景可稳定复现性能提升**：依托 SPMD 编程模型（编译器自动分区张量、重叠通信与计算）、RPA v3 内核（融合 KV 缓存更新、隐藏延迟、适配混合批次），官方数据显示 Llama 3.1-8B 在 v6e-1 上吞吐量提升 3.6x，70B 在 v6e-8 上提升 2.1x，新架构较 2025 年 2 月原型快近 5x，且通过量化、结构化解码等优化进一步保障高并发下的稳定性。
- **存在明确场景局限性**：目前**对 XL MoE、带视觉编码器的复杂模型仍在优化算子支持**；v5p 型号需额外调优才能稳定运行；:mdnice-bold-italic[实验性功能（多 LoRA、树状 Eagle 3 推测解码、分布式解耦部署）尚未完全成熟]，:mdnice-bold-italic[暂不适合核心生产场景]；超大规模多主机部署的负载均衡仍需依赖第三方项目（如 llm-d）补充。

## 本文目录

- 关键问题
  - 问题 1：vllm 对 TPU 的适配是否真正解决了 TPU 生态中算子支持不足的核心痛点，其与 GPU 部署的迁移成本差异如何？
  - 问题 2：同等硬件投入下，vllm+TPU 的性能提升（吞吐量、延迟）在真实高并发场景中能否稳定复现，是否存在场景局限性？
- 一、vLLM TPU：一款支持在 TPU 上运行 PyTorch 与 JAX 的全新统一后端
- 二、vLLM TPU
  - 挑战1：vLLM V1 版本集成
  - 挑战2：多程序多数据
  - 挑战3：PyTorch/XLA
- 三、由 TPU-inference 支持的 vLLM TPU
  - 3.1 面向 PyTorch 与 JAX 的统一后端
  - 3.2 深入解析
  - 3.3 不规则分页注意力 V3（RPA V3）：开源领域中用于 TPU 推理的最灵活、高性能注意力内核
  - 3.4 单程序多数据（SPMD）
  - 3.5 整体总结
- 四、支持的模型、功能及未来规划
  - 4.1 支持的模型类型
  - 4.2 支持/已验证的 TPU 型号
  - 4.3 功能特性
  - 4.4 对 TPU 友好的内核
  - 4.5 实验性功能
  - 4.6 未来规划
- 总结

![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/04.webp)

## 一、vLLM TPU：一款支持在 TPU 上运行 PyTorch 与 JAX 的全新统一后端

> 如今，vLLM TPU 由tpu-inference[1]提供支持——这是一款功能强大且表达性强的全新硬件插件，能将JAX[2]（谷歌开发的机器学习框架，擅长自动微分和并行计算）与PyTorch[3] 统一到单一的 lowering 路径下。

它不仅比上一代 vLLM TPU 速度更快，还能支持更多类型的模型，提供更丰富的功能支持。vLLM TPU 作为一款面向开发者的框架，可实现以下目标：

::: block-1
1. 在开源领域突破 TPU 硬件的**性能** 极限。
2. 为 JAX 和 PyTorch 用户提供更高的**灵活性**：无需修改任何代码，就能让 PyTorch 模型定义在 TPU 上高效运行，同时还能为 JAX 提供原生支持。
3. 保留 vLLM 的**标准化** 特性：保持一致的用户体验、遥测能力（指对系统运行数据的采集和监控）与接口。
:::

![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/03.webp)

## 二、vLLM TPU

> 2025 年 2 月，正当vLLM 的 V1 版本集成[4]工作初具雏形时，一支由谷歌员工和 vLLM 核心贡献者组成的“小而精”的团队，为自己设定了一个目标：赶在谷歌云 2025 年 Next 大会[5]（谷歌云举办的重要技术与产品发布会）之前，推出一款能在少量模型上运行的高性能 TPU 后端。

在接下来的两个月里，他们遇到了多项挑战，具体如下：

### 挑战1：vLLM V1 版本集成

团队需集成到全新的 V1 代码路径中，这就需要开发一款新的不规则分页注意力内核（RPA v2[6]，RPA 为 Ragged Paged Attention 的缩写，指处理长度不一致的输入序列时的分页注意力机制）。

**开发该内核的主要目的是支持分块预填充** （指将输入序列分成多个块进行预处理，优化内存使用和计算效率）**和前缀缓存** （指缓存输入序列的前缀部分，避免重复计算，提升推理速度）等功能。

尽管这些键值缓存管理技术在 TPU 中较为常见，但要结合 vLLM 的分页注意力，设计出“对 TPU 友好”的方案仍颇具难度。

### 挑战2：多程序多数据

当时，vLLM 完全依赖 MPMD[7]（:mdnice-bold-italic[多程序多数据，一种并行计算模型，每个进程运行不同程序处理不同数据]）来协调进程间通信。这与 TPU 以编译器为中心的编程模型形成了鲜明对比——TPU 的编程模型高度依赖单程序多数据:mdnice-bold-italic[（SPMD[8]，单程序多数据，并行计算模型，所有进程运行相同程序、处理不同数据）] 来实现多设备与多主机通信的重叠。

### 挑战3：PyTorch/XLA

尽管 PyTorch/XLA 框架（也称为PTXLA[9]，其中，XLA 是加速线性代数计算的编译器，可让 PyTorch 代码在 TPU 上原生运行）能让 PyTorch 代码在 TPU 上原生运行，从而降低了 vLLM 的集成难度，但团队在对技术栈底层进行优化时，仍遇到了不少挑战。

尽管面临这些障碍，团队仍实现了性能突破：在 v6e-1 型号 TPU 上运行 Llama 3.1-8B 模型时，吞吐量性能提升了**3.6 倍**；在 v6e-8 型号 TPU 上运行 Llama 3.1-70B 模型时，吞吐量性能提升了**2.1 倍**。此外，vLLM TPU 还成功登上了谷歌云 Next 大会的重要舞台[10]。

## 三、由 TPU-inference 支持的 vLLM TPU

> 尽管基于 PTXLA 的 vLLM TPU 已是一项重大成就，但我们仍需在开源领域继续突破 TPU 的性能极限。同时，我们还**希望通过以最高效的方式在 TPU 上原生支持 PyTorch 和 JAX 模型，将 TPU 与 vLLM 生态系统整合起来。**

### 3.1 面向 PyTorch 与 JAX 的统一后端

这款基于tpu-inference[11]重新设计的 vLLM TPU，旨在通过单一统一的 JAX→XLA lowering 路径支持 PyTorch（通过Torchax[12]，谷歌开发的工具，可将 PyTorch 代码转换为 JAX 兼容形式以在 TPU 上运行）和JAX[13]，从而实现性能与可扩展性的优化。

与 PyTorch/XLA 相比，:mdnice-bold-italic[JAX 是一套更成熟的技术栈——其原语（指基础的计算单元或操作，是构建复杂算法的基础）通常具有更广泛的覆盖范围和更优的性能，尤其是在实现复杂并行策略（如数据并行、模型并行等）时表现更为突出。]

正因如此，**vLLM TPU 现在通过 JAX 为所有 vLLM 模型提供 lowering 路径** ——即便模型定义是用 PyTorch 编写的，也能受益于显著的性能提升。这一决策能让我们更高效地推进开发：通过封装高层框架，将精力集中在核开发（开发硬件级的计算核心代码）与编译器性能优化上。需注意的是，**对于 XLA 而言，Torchax 和 JAX 在编译前会使用相同的高性能原语**。在jax-primitives[14]，可以了解更多相关信息。

尽管当前采用这一设计，但我们始终致力于在 TPU 上实现最佳性能，并**计划在未来** 为 vLLM TPU 评估 TPU 上的 PyTorch 原生移植版本，即**无需通过 JAX，直接为 TPU 适配的 PyTorch 版本。**

> **核心要点 1**：如今，vLLM TPU 通过 JAX 对所有模型进行 lowering 处理。无需修改任何模型代码（例如llama.py[15]，Llama 模型的代码文件），**vLLM TPU 的吞吐量性能就能提升约 20%——这完全得益于 JAX 成熟的高性能原语**：JAX 会生成 HLO 图（HLO 为 High-Level Optimizer 的缩写，指 XLA 的高级中间表示图，用于编译器优化），再由 XLA 对该图进行编译。

### 3.2 深入解析

1. 安装：`pip install vllm-tpu`

由于 Torchax 和 JAX 在底层本质上都基于 JAX，因此无论模型代码是用 PyTorch 还是 JAX 编写的，我们都能使用相同的安装路径。这确保了依赖项的一致性，用户无需担心为不同模型管理不同的依赖要求。

1. 模型部署 MODEL_ID="google/gemma3-27b-it" # 在tpu-inference或vllm中已注册的模型 vllm serve $MODEL_ID

在 TPU 上部署模型时，有两个模型注册表可用于拉取模型代码：

1. tpu-inference（默认，列表[16]）
2. vllm（由 vLLM 上游代码维护，上游指开源项目的主分支或核心开发分支，列表[17]）

下面，让我们深入了解底层的工作原理：

![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/05.webp)

::: block-1
这项统一工作通过**利用 vLLM 社区的现有成果，减少了重复工作，从而让团队有更多时间优化 TPU 内核与 XLA 编译器**。对于 PyTorch（通过 Torchax）和 JAX 模型，所有内核与编译器都是共享的。
:::

> **核心要点 2**：如今，vLLM TPU 会默认优先运行 tpu-inference 中经过 TPU 优化的模型代码（若存在）；若不存在，则会回退到 vLLM 上游代码中的 PyTorch 模型代码（通过Torchax[18]借助 JAX 进行 lowering 处理）。对于大多数用户而言，这属于实现的内部技术细节。

*既然 Torchax 能让 PyTorch 模型代码在 TPU 上开箱即用（OOTB，Out Of The Box，指无需额外配置即可直接使用），且仍通过 JAX JIT（即时编译，指在程序运行时动态编译代码以提升性能）进行编译，那我们为何还要在 tpu-inference 中重写部分模型呢？这难道不是重复工作吗？*

我们为开发者提供了少量参考模型，以降低他们在开始为 TPU 优化模型前的上手难度（详见vllm-project/tpu-inference/tree/main/tpu_inference/models/jax[19]）。有趣的是，我们观察到：通过 Torchax 进行 lowering 处理的模型，与简单重写的 JAX 模型（指未进行额外优化、仅基础重写的模型）性能大致相当——**这充分体现了 Torchax 在转换高层模型（指将 PyTorch 等高层框架的模型转换为 JAX 可处理的形式）时的高效性。**

而我们支持重写模型的真正原因，以及重写带来的性能提升，源于对 JAX 代码的 TPU 优化——即直接利用 TPU 架构的优势。

:mdnice-bold-italic[我们需要这种灵活性的原因在于：vLLM 开发者在实现模型时的逻辑设计选择（指开发者的设计思路和方案）并非总能适配 TPU]。导致这种适配问题的原因，并非 JAX 与 Torchax 的差异，而是 **GPU 与 TPU 本身的架构不同——二者需要不同的优化策略。**

> **核心要点 3**：对于任何模型而言，其底层实现都基于 JAX！除非实现中的逻辑差异导致 TPU 性能下降，否则将模型原生重写为 JAX 版本通常无法带来性能提升。尽管如此，:mdnice-bold-italic[若重写模型能充分发挥 TPU 性能，那么保留这种重写的灵活性仍十分重要。]

### 3.3 不规则分页注意力 V3（RPA V3）：开源领域中用于 TPU 推理的最灵活、高性能注意力内核

尽管不规则分页注意力 V2（RPA V2）内核已实现了性能的大幅提升，但为了开箱即用地支持更多模型与使用场景，它还需要具备更高的灵活性。

1. RPA V2 仅支持头维度（注意力机制中每个注意力头的维度）为 128 的模型规格如头维度、层数等。
2. RPA V2 由于需顺序执行 KV 缓存更新与注意力操作（先执行 KV 缓存更新，再执行注意力操作），导致流水线效率低下存在等待时间。
3. 在以解码为主或预填充任务输入序列长度可变的场景下，RPA V2 会造成大量资源浪费。
4. 尽管 RPA V2 相较于首个 TPU 原型版本已实现了显著的吞吐量提升，但仍缺乏灵活性。

我们即将发布关于 RPA V3 的技术深度解析，敬请关注我们的文档（更新。

> **核心要点 4**：RPA V3 兼具灵活性与高性能，是开源领域中生产级 Pallas 内核（Pallas 是 JAX 生态中的库，用于开发高性能硬件内核）开发的优秀参考案例。我们期待未来能以类似方式，在开源领域推出对 TPU 友好的混合专家模型和多语言注意力内核。

### 3.4 单程序多数据（SPMD）

此次发布将单程序多数据（SPMD[20]，Single Program, Multi-Data）作为 vLLM TPU 的默认编程模型。与之前的多工作进程模型（借鉴自 GPU 范式，指通过多个独立工作进程实现并行的模式）不同，**SPMD 是 XLA 编译器原生支持的**。开发者:mdnice-bold-italic[可将多个 TPU 设备视为一个单一的“巨型设备”来编写代码，XLA 编译器会自动对模型和张量进行分割，并插入通信操作以实现优化执行。]

> **核心要点 5**：SPMD 支持通信与计算重叠（:mdnice-bold-italic[指让设备间的数据传输和计算操作同时进行，减少等待时间]）等高级优化。**SPMD 标志着 vLLM TPU 向更深度的 TPU 原生集成** （更贴合 TPU 硬件特性的集成方式）迈出了战略性一步——**通过以 TPU 为中心、编译器优先的运行模式** （设计围绕 TPU 硬件，依赖编译器进行优化的模式），有望实现更高性能。

### 3.5 整体总结

![文章插图](/articles/2025-10-26-vllm-tpu-tpu-pytorch-jax-72aef285/images/06.webp)

自 2025 年 2 月的原型版本以来，vLLM TPU 已取得了长足进步：在相同工作负载下，性能提升了近**2 至 5 倍**，同时模型支持范围和易用性（指用户使用该框架的便捷程度）也得到了提升。

> **核心要点 6**：如今，vLLM TPU 的性能**相较于 2025 年 2 月的首个 TPU 原型版本提升了近 5 倍**。随着这一新基础的搭建完成，开发者和研究人员将能在开源领域前所未有地突破 TPU 推理性能的边界。

## 四、支持的模型、功能及未来规划

> 我们可将此次发布视为一项基础性成果——今后，vLLM TPU 将在开源领域定期发布版本。每发布一个新版本，持续集成/持续部署系统都会发布经审核的 vLLM 原生模型的文档化表格。

我们还将维护一份经过压力测试的 tpu-inference 模型列表，主要作为 JAX 用户的参考。所有功能在发布前也将经过严格测试。

### 4.1 支持的模型类型

::: block-1
- 稠密模型（Dense，指所有层都参与计算的传统模型架构，区别于 MoE 等稀疏架构）
- 多模态模型（Multimodal，指支持文本、图像等多种输入类型的模型，仅 tpu-inference 中的模型支持）
:::

> **模型支持说明**：在加入更多功能前，我们建议从经过压力测试的模型列表此处[21]）开始尝试。我们仍在向 tpu-inference 中集成组件，这些组件将提升更大规模、更高复杂度模型（如超大型混合专家模型 XL MoE，XL 为 Extra Large 的缩写，指规模更大的 MoE 模型；视觉编码器，指处理图像等视觉数据的编码模块；MLA 等）的性能。 若您希望我们优先处理特定需求，请通过此处[22]在 GitHub 上提交功能请求。

### 4.2 支持/已验证的 TPU 型号

- Trillium（v6e，谷歌 TPU v6e 型号的代号）、v5e（TPU 型号）

### 4.3 功能特性

- 前缀缓存（Prefix Caching，缓存输入序列前缀以减少重复计算）
- 分块预填充（Chunked Prefill，分块处理预填充任务以优化内存）
- 多模态输入（Multimodal Inputs，支持多种类型的输入数据）
- 单程序多数据（SPMD，详见前文解释）
- 结构化解码（Structured Decoding，指按照特定结构生成输出的解码方式，如生成 JSON 格式文本）
- 推测性解码：Ngram（Speculative Decoding，通过小模型提前预测 tokens 以加速大模型解码的技术，此处采用 Ngram 算法）
- 树外模型支持（Out-of-tree model support，指支持不在 vLLM 核心代码库中的外部模型）
- 优化的运行时采样（Optimized Runtime Sampling，指在推理运行时对生成结果进行优化采样，包括 top k（选取概率前 k 个 token）、top p（选取累积概率达 p 的 token 集合）、temperature（调整采样随机性）、logit 输出（输出模型预测的 logit 值）等策略）
- 量化

### 4.4 对 TPU 友好的内核

- 不规则分页注意力 V3（Ragged Paged Attention V3，详见前文解释）
- 集合通信矩阵乘法（Collective Communication Matmul，支持多设备间协同通信的矩阵乘法操作）
- 量化矩阵乘法、注意力与 KV 缓存（Quantized Matmul, Attention and KV Cache，对这些计算模块进行量化优化）

### 4.5 实验性功能

- v5p（TPU 型号，处于实验阶段）
- 多模态（通过 Torchax 支持，处于实验阶段）
- 多 LoRA（Multi-LoRA，同时加载多个 LoRA 适配器，LoRA 为 Low-Rank Adaptation 的缩写，指通过低秩矩阵微调模型的技术）
- 推测性解码：基于树的 Eagle 3（Speculative Decoding，采用树结构的 Eagle 3 算法，处于实验阶段）
- 单主机预填充/解码解耦部署（Single-host P/D disaggregated serving，P/D 指 Prefill/Decode，即在单主机上分别部署预填充和解码模块，处于实验阶段）

### 4.6 未来规划

- 稀疏核心卸载：Sparsecore offloading，将计算任务卸载到 TPU 的稀疏核心处理，稀疏核心专门优化稀疏计算
- 推测性解码：Eagle 3、MTP
- 对 TPU 友好的内核：

## 总结

> vLLM TPU此次基于tpu-inference打造的统一后端，无疑是其在TPU生态适配与LLM推理性能优化上的里程碑突破。

它通过将JAX作为统一lowering路径，既实现了PyTorch模型“**零代码改动** ”在TPU上的高效运行，**又延续了JAX生态的成熟性能优势**，仅凭借底层框架优化就带来约20%的吞吐量提升；而RPA v3内核的灵活化、 pipeline 效率提升与部署场景拓展，以及SPMD编程模型的引入，更是**从核心计算逻辑与分布式执行范式层面，彻底解决了前代版本在模型覆盖、性能释放与TPU原生适配上的痛点** ——最终让vLLM TPU相比2025年2月的原型实现，实现了2-5倍的性能飞跃，同时覆盖了密集型、多模态模型及v5e/v6e等TPU型号，还支持前缀缓存、推测解码等关键推理特性。

未来，随着稀疏核心卸载、XL MoE/MLA专用TPU内核、多主机分布式推理等功能的落地，vLLM TPU将进一步释放TPU在大模型推理场景下的硬件潜力，为开发者提供更贴近生产需求的开源工具链。

目前，开发者可通过Google Cloud的GKE[23]、Compute Engine[24]与Vertex AI[25]快速体验这一框架，而vLLM社区与Google团队的持续协作，也将让这一工具在模型覆盖广度与性能深度上持续进化，成为推动TPU生态开源化与大模型推理效率提升的重要力量。

参考资料

[1] http://tpu.vllm.ai/: *http://tpu.vllm.ai/*

[2] JAX: *https://docs.jax.dev/en/latest/index.html*

[3] https://pytorch.org: *https://pytorch.org/get-started/locally/*

[4] vLLM的V1版本集成: *https://docs.vllm.ai/en/latest/usage/v1_guide.html*

[5] Cloud Next 2025: *https://cloud.withgoogle.com/next/25*

[6] RPA v2: *https://github.com/pytorch/xla/blob/master/torch_xla/experimental/pallas_kernels/ragged_paged_attention_v2.py*

[7] MPMD: *https://en.wikipedia.org/wiki/Flynn%27s_taxonomy#Multiple_programs,*multiple_data_streams* (MPMD "MPMD")*

[8] SPMD: *https://en.wikipedia.org/wiki/Single_program,_multiple_data*

[9] PTXLA: *https://github.com/pytorch/xla*

[10] the big stage at Cloud Next: *https://www.youtube.com/live/Md4Fs-Zc3tg?si=t3V52Kac5Y5VTNN0&t=1137*

[11] http://tpu.vllm.ai/: *http://tpu.vllm.ai/*

[12] https://google.github.io/torchax/: *https://google.github.io/torchax/*

[13] https://docs.jax.dev/en/latest/index.html: *https://docs.jax.dev/en/latest/index.html*

[14] https://docs.jax.dev/en/latest/jax-primitives.html: *https://docs.jax.dev/en/latest/jax-primitives.html*

[15] https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py: *https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py*

[16] https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax: *https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax*

[17] https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py: *https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py*

[18] Torchax: *https://google.github.io/torchax/user_guide/how-it-works/*

[19] https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax: *https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax*

[20] https://en.wikipedia.org/wiki/Single_program,_multiple_data: *https://en.wikipedia.org/wiki/Single_program,_multiple_data*

[21] https://github.com/vllm-project/tpu-inference/blob/main/support_matrices/model_support_matrix.csv: *https://github.com/vllm-project/tpu-inference/blob/main/support_matrices/model_support_matrix.csv*

[22] https://github.com/vllm-project/tpu-inference/issues/new/choose: *https://github.com/vllm-project/tpu-inference/issues/new/choose*

[23] Google Cloud的GKE: *https://cloud.google.com/tpu?hl=en#cloud-tpu-in-gke*

[24] Compute Engine: *https://cloud.google.com/tpu?hl=en*

[25] Vertex AI: *https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm-tpu*

加群
