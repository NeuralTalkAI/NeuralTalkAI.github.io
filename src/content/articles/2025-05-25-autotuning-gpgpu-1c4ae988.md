关键词：LLM Portability, Domain-specific Languages, Performance of Systems, Code tuning

不久前，分享过一篇注意力引擎库的文章[《 FlashInfer：面向 LLM 服务的可定制且高效的 GPU 注意力引擎》](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888144&idx=1&sn=0b3de7ba3a4d204ad4e017c19e2aed69&scene=21#wechat_redirect)，FlashInfer 支撑了 SGLang、vLLM 和 MLC-Engine 。今天再分享一篇来自 IBM Research Europe 在 Trition 上针对 GPGPU 进行注意力内核自动调优的工作。

> 随着大语言模型（LLMs）的复杂性不断提高，要实现最先进的性能，需要在**算法、软件和硬件方面** 进行**紧密的协同设计**。如今对单一主导平台的依赖限制了可移植性，造成了供应商锁定，并给**新型人工智能硬件** 设置了障碍。

在这项工作中，我们**提出将即时（JIT）编译与内核参数自动调优相结合的方案**，以实现**【无需代码更改】** 即可具备**【一流性能】** 的**【可移植大语言模型推理】**，且**【代码开源】**！

::: block-1
我们聚焦于 FlashAttention（一种广泛应用且对性能至关重要的大语言模型内核），基于 Triton，我们证明了这种方法能够探索多达 15 倍的内核参数配置，在多个维度上生成更多样化的代码，甚至**比硬件厂商优化的实现快 230%，同时将内核代码大小缩小 70 倍**，省去手动代码优化。

我们的结果表明，**自动调优是实现跨 GPU 供应商** 模型**可移植性** 的一条**有前景的途径**。
:::

- `github.com/IBM/triton-dejavu`：`triton-dejavu` 是一个用于减少 triton-lang 自动调优开销的框架，特别适用于已知的部署场景
- `ibm.biz/vllm-ibm-triton-lib`：有多个性能优化点值得关注，集中在 Triton 内核实现、内存访问模式和自动调优等方面

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/01.webp)

- https://arxiv.org/abs/2505.03780

## 本文目录

- 一、引言
- 二、背景与相关工作
  - A. 基于模板库的代码生成
  - B. 编译与自动调优
  - C. Triton：一种平铺（tiling）领域特定语言
  - D. Triton 自动调优
- 三、我们的自动调优方法
  - 需求 1：缓存自动调优结果
  - 需求 2：扩大搜索空间且通用化
  - 需求 3：提供简单易用的调优接口
- 四、实验评估
  - A. 跨平台性能
  - B. 自动调优对 Triton 性能的影响
  - C. 自动调优对代码多样性的影响
- 五、结论
- 附录1：vllm-ibm-triton-lib
  - 1. Triton 内核优化
  - 2. 自动调优与性能适配
  - 3. 计算效率优化
  - 4. 内存管理优化
  - 5. 后端集成与平台适配
  - 6. 微基准测试框架
- 附录2：triton-dejavu 项目
  - 主要功能
  - 核心模块介绍
- 参考文献

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/02.webp)

加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

在过去几年中，大语言模型（LLMs）取得了巨大发展。除了模型架构和训练流程的改进外，在针对现代硬件优化大语言模型应用方面也有诸多创新（[1] - [4]）。

然而，这种在**特性和性能方面的竞争**，给新的人工智能（AI）或机器学习（ML）范式**带来了 “硬件彩票” 问题** [5]，并**围绕最主流的硬件平台形成了一种 “引力斜坡”**。

除了对哪些算法被认为切实可行有所限制外，这还导致了人工智能在部署和应用场景方面的局限性。

> 因为大多数特性仅在有限的一组硬件或输入问题规模下得到支持[5]。

因此，用于部署具备最先进（SOTA）性能的大语言模型的库的数量和规模都大幅增长。

我们在图 1 中展示了这种动态变化，图中展示了核心 FlashAttention 层[6]的四种不同实现方式在来自不同厂商的两种 GPU 架构上的性能。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/03.webp)

FlashAttention 的实现方式列于表 1 中，性能结果均已根据每个平台上的基线 PyTorch 原生实现进行了归一化处理。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/04.webp)

可以看出，**通用的原生 PyTorch 实现仅需 29 行代码** （LoC），但比针对 NVIDIA GPU 优化的流行 flash_attn 库或在 AMD MI250 上提供**最优性能** 的 FlashAttention 库的 ROCm 版本**慢 6 到 13 倍**。

然而，**这两个优化库也比 PyTorch 原生实现复杂得多（代码行数多出 2300 倍）**。最后，图 1c 量化了在 NVIDIA 和 AMD 架构之间移植大语言模型（LLM）注意力层所需的底层代码更改。**为了在 MI250 上实现最优性能，最初的 FlashAttention 中有超过 40%的部分必须手动优化**。

> 拥有兼具 PyTorch 的简洁性和可移植性，同时又具备最先进性能的内核代码，仍然是一个有待研究的开放性问题。

为了移植一个一行代码的内核而编写数万行代码 [6]，这不仅会减缓研究进度，极大地增加新机器学习方法的部署复杂性，还会阻碍新硬件的应用。此外，代码量增加几个数量级也意味着出错的概率会相应提高。

在当下 AI 快速发展，这些问题愈发紧迫，因为上述提到的注意力库都并非 “最终版本”。**为了整合新算法、满足新需求或支持新硬件，需要不断进行更改** （[4]，[5]）。例如，**flash_attn 的第二个版本耗时一年多** [1]，[2] 才开发出来。然后，**又过了一年，针对英伟达新一代 Hopper GPU 的改进版本** [3] 发布，**而发布相应的适用于生产环境的库又花了八个月时间**。

---

而在我们这项工作中，我们重新审视了零改动性能可移植性问题，旨在实现简洁、可移植且高效的 GPU 内核。

- 首先，我们回顾了**导致当前手动移植内核现状的潜在问题**。
- 我们提**出了一种在 GPU 上实现零改动性能可移植性的实用方法**，并展示了**在两家供应商的两种 GPU 平台上仅用 1100 行代码** 实现的高级快速注意力（flash attention）的最先进性能。
- 为了与前面讨论的激励性示例形成对比，我们**还在图 1 中展示了使用我们的解决方案** （Triton 自动调优）**得到的相应结果**。

> 我们的研究结果表明，**利用自动调优** 是追求大语言模型（LLM）性能**可移植性的关键组成部分**。

## 二、背景与相关工作

### A. 基于模板库的代码生成

一种**针对不同硬件架构定制内核的常用方法是通过使用模板库**。例如，一组流行的模板库旨在尽可能高效地实现注意力层[6]，并适用于广泛的使用场景，它们是 flash_attn [1]-[3]和 FlashInfer [8]。这些库并非简单易做。

总的来说，flash_attn 有近 70000 行代码，而 Flashinfer 有 51000 行代码。这些模板库根据使用场景（例如，取决于张量形状、数据类型或批量大小）选择哪些手写代码片段（用像 CUDA 这样的低级语言编写）。然后，低级函数被编译成相应的加速器指令集（例如，用于英伟达 GPU 的 PTX/SASS）。

**基于模板的方法可以生成逐点最优解，并且比简单方法覆盖更大的优化空间。然而，它并不完美**。

1. 首先，这种模板库的灵活性相当有限，因为通常只有一种或几种实例化模板的方法。如果在与模板开发时不同的硬件上使用模板，这会**导致次优的代码生成** [9]。次优的代码生成可能会**导致人工智能应用中 GPU 的计算利用率较低** [3]。
2. 其次，由于硬件在不断发展（即我们现在有**数十种基于不同硬件架构的加速器选项**，涵盖**多代产品** 且来自越来越多的供应商），这使得选择**合适的模板版本变得更加困难** [9]。
3. 第三，**为一组新的加速器添加优化的新模板版本需要大量的人工工作** ——以数万行代码来衡量——这使得大语言模型软件的进入门槛很高。

### B. 编译与自动调优

::: block-1
一般来说，有**两种方法** 可以**避免编写手动优化代码** 的需求：

- 第一，依靠提前编译（Ahead-of-Time）或即时编译（Just-in-Time，JIT）编译器，为大语言模型（LLM）生成可执行文件；
- 第二，利用自动调优来优化大语言模型的内核参数。
:::

#### B.1 编译器方法

**基于编译器的方法依赖预定义的启发式算法** 来生成通用代码。为了提高性能，编译器可能会使用许多优化步骤，并利用各种预先确定的启发式算法。然而，由于**将高级语言编译为低级语言的问题是 NP 难问题**，**优化性能可能会导致非常长的编译时间** [9]，这是不可容忍的，尤其是对于即时编译器而言。此外，编译器必须考虑各种可能的内核参数，以便生成有效的二进制文件，因此无法针对所有参数组合实现性能最大化。

#### B.2 自动调优方法

相比之下，**自动调优内核通过实证性能调优来增强编译过程，生成并对大量内核变体进行基准测试，以便为目标硬件和场景选择性能最佳的配置**。

自动调优减少了编译器在特定内核编译时需要考虑的参数空间。因此，它能够实现针对特定场景的更优编译优化，代价是为每个调优后的目标场景生成更多编译产物。

这种方法可以探索更大的优化空间——通常能生成数量级更多的变体[10, 11]，从而带来更高的性能和更好的代码专用化。

> 虽然自动调优会引入额外开销，但其无需手动调整就能实现近乎最优性能的能力，使其成为在异构平台上部署大语言模型（LLMs）的极具吸引力的解决方案。

---

在自动调优器与编译器的深度集成中，自动调优器可以将实验结果的性能细节反馈给编译器，以便编译器生成与先前最佳实现类似的代码[10]。

这些优势使得自动调优更适合在异构硬件平台上部署大语言模型。一般来说，必须对自动调优进行权衡，以使其在性能方面的优势超过编译和执行时间开销方面的劣势。

### C. Triton：一种平铺（tiling）领域特定语言

OpenAI 最近发布了 Triton 领域特定语言（DSL），这是一种很有前景的开源方案，可替代编写自定义 CUDA 内核。

Triton 允许使用简单的 Python 代码编写和调试内核，这些代码可以在各种 GPU 上执行。

> 事实证明，Triton 内核在不同的 GPU 平台上都具有高性能和可移植性。

因此，Triton 越来越受欢迎；它被用于许多大语言模型（LLM）堆栈，并集成到了`pytorch.compile`中。**Triton 利用即时（JIT）编译器**，并基于**分层瓦片（tiling）** 的理念，**实现内存合并、共享内存分配以及线程间同步的自动化** [12]。

```
# Listing 1：Triton 中的简单向量加法程序instance_id = tl.program_id(axis=0)my_block_start = instance_id * BLOCK_SIZEoffsets = my_block_start + tl.arange(0, BLOCK_SIZE)mem_mask = offsets < n_elementsx = tl.load(x_ptr + offsets, mask=mem_mask)y = tl.load(y_ptr + offsets, mask=mem_mask)result = x + ytl.store(output_ptr + offsets, result, mask=mem_mask)
```

**上面的代码清单 1 展示了 Triton 中一维并行向量加法**。使用超参数（也称为内核配置），可以针对不同的工作负载大小或目标架构对 Triton 内核进行微调。在清单 1 中，`BLOCK_SIZE`是一个影响 GPU 内核心调度的配置。

### D. Triton 自动调优

在学术文献之外，大语言模型应用的自动调优工作由例如 PyTorch Inductor [14] 完成。**Inductor** 是 `torch.compile` 的调优前端，`torch.compile` 也**是一个即时编译器，它可以使用 Triton 作为后端**。Inductor **通过简单地按顺序运行不同的算法来实现自动调优。**

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/05.webp)

Triton 内核有一个内置的自动调优器，它需要程序员提供一份潜在内核配置列表，然后依次尝试所有配置。然而，这一功能在实际中很少使用。通常，Triton 内核是针对特定 GPU 进行手动优化的，在不同的 GPU 平台上表现并不相同。在表 II 中，我们对流行的大语言模型**框架和推理服务器进行了简要调研。可以看出，只有一小部分 Triton 内核利用了自动调优。**

> 接下来，我们总结一下在**自动调优方面，理论与实践存在差距的原因**：

**使用内置的自动调优器除了会增加即时编译的开销外，还会给 Triton 内核启动带来更多的额外开销**。这种额外开销源于这样一个事实：对于内核参数的每一种变化，自动调优器都需要确定哪个内核版本的性能最佳。此外，每次启动新进程时都必须重复这个过程，因为自动调优器的结果仅在创建它们的进程中有效。

自动调优**没有被广泛用** 于在不同平台间移植大语言模型（LLM）的**另一个原因是**，这并非大语言模型社区关注的重点。**通常，研究界和工业界的直接目标是在 “事实上的默认平台” 上的一组有限的基准** 测试中展现性能。几乎没有任何工作致力于提升平台和性能的可移植性。因此，认识不足或代码成熟度不够，为自动调优为何未在当今实践中得到应用提供了另一种解释。

> 此外，Triton 自动调优器仍需手动优化。开发人员必须为每个不同的内核变体提供一份可供探索的配置选项列表。

**这份列表的选择基于程序员的直觉，并且会显著影响最终性能**。例如，我们的结果表明，对于**复杂内核，性能差异接近 20 倍（见图 4a）**。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/06.webp)

总体而言，我们观察到，在**超过 90% 的情况下，自动调优器并未得到使用**，而且**大多数 Triton 内核都是针对特定 GPU 架构手动优化的**。即使在启用自动调优器的情况下，也**未充分发挥其潜力**。例如，要么只探索了非常有限的一组超参数，要么基于错误的内核参数触发了自动调优器。

## 三、我们的自动调优方法

基于我们对 Triton 自动调优器的分析，我们确定了**使该方法切实可行** 所需的**三项改进**：

1. 必须**将自动调优的开销移出关键路径**。理想情况下，自动调优的执行运行结果应该被缓存。
2. **自动调优器应采用一种搜索策略**，该策略能够**扩展到大型配置空间**，同时具有足够的**通用性**，以**适应各种内核、内核参数和平台**。
3. 需要**一个应用程序编程接口** （API）来**表述内核参数空间和约束条件**。

### 需求 1：缓存自动调优结果

我们开发了“DejaVu”，这是**一种针对 Triton 自动调优器的缓存机制**，它**消除了 Triton 自动调优器的运行时开销**，这是在生产环境中使用自动调优器的关键要求。在缓存实验运行结果时，主要的挑战在于全面确定自动调优器的环境、软件和硬件依赖关系，并准确判断何时重用先前的结果而不是触发全新的执行。

### 需求 2：扩大搜索空间且通用化

**我们采用了一种简单但有效的策略**：我们将自动调优器配置为**仅评估 2 的幂次方的问题规模**。例如，对序列长度为 450 的内核进行自动调优时，会选择与对长度为 512 进行自动调优时相同的结果。

其依据是，Triton 使用了基于 2 的幂次方建模的 Triton 分层内存模型[12]。对于 Flash Attention 内核，2 的幂次方探索方法将 790,000 个可能参数值的搜索空间缩小了三个数量级，降至每个 GPU 类型更易于管理的 768 个。

### 需求 3：提供简单易用的调优接口

**我们在 auto-tuning 配置空间中添加了一个简单的功能，该功能能够清晰且灵活地定义大型配置**。

```
# Listing 2：自动调优的配置空间定义示例ConfigSpace({    "BLOCK_M": [16, 32, 64, 128, 256],    "BLOCK_N": [16, 32, 64, 128, 256],    "PRE_LOAD_V": [True, False],},kwarg_conditions=[    lambda c: c["BLOCK_M"] >= c["BLOCK_N"],    lambda c: c["BLOCK_N"] >= 32or"H100"notin gpu_name,],num_warps=[2, 4, 8], num_stages=[1, 2, 4, 6, 8])
```

清单 2 是一个这种配置空间定义的示例。它将一个空间描述为所有超参数的笛卡尔积，再加上过滤函数，以确保只搜索有效的配置。请注意，`kwargs` 条件对于过滤那些在某些平台上无效但在其他平台上表现良好的配置也是必要的。

---

我们将**工作成果开源（github.com/IBM/triton-dejavu/，详见附录）**，很高兴看到类似的想法出现在其他开源项目中，例如 Triton [15]、[16] 以及 PyTorch [17]。

## 四、实验评估

我们在 Triton 中实现的 FlashAttention 内核[2]有 1134 行代码，其中包括自动调优代码，它是**在现有开源内核** [7]的**基础上** 结合全面的**自动调优** 进行的**改进版本**。我们在 `ibm.biz/vllm-ibm-triton-lib` 上开源了我们的实现。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/04.webp)

### A. 跨平台性能

为了测试我们的框架，我们在 Triton 中使用注意力内核实现运行微基准测试。作为基线，我们与表 I 中列出的最先进（SOTA）实现进行比较。**图 2 展示了这些不同实现在 A100 - 80GB 和 MI250 128GB 上的微基准测试结果**。请注意，针对 NVIDIA 和 AMD GPU 的 flash_attn 库有所不同，而我们自动调优的 Triton 内核保持不变。**图 2a 和图 2b 展示了针对不同最大序列长度的四个不同图表**。在每个图表中，x 轴表示批量大小，y 轴表示延迟（数值越低越好）。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/07.webp)

延迟值通过 flash_attn 最左侧的延迟值进行归一化。如第三节所述，对于自动调优，我们仅使用 2 的幂次方的问题规模。然而，测试微基准测试也包括非 2 的幂次方的批量大小。**每个平台上的自动调优时间为 24.1 小时，其中包括编译时间，编译时间约占自动调优时间的 80%。** 所有实验均使**用 CUDA 图（或 HIP 图）来评估内核性能**，并避免测量软件开销。

> 如图 2 所示，**无论批量大小或序列长度如何，我们自动调优的 Triton 内核在两个平台上都具有竞争力**，代码行数（LoC）不到 2%。

**在最佳情况下，我们的 Triton 内核比 flash_attn 快 2.3 倍**，在最坏情况下，它仍能达到 78%的性能（针对非常大的批量大小和长序列）。

### B. 自动调优对 Triton 性能的影响

接下来，我们**评估自动调优对 Triton 是否真的必要**，或者是否有可能在所有情况下找到接近最先进性能的配置或简单启发式方法。为了评估这一点，我们**进行了两组实验**。首先，**我们为每个 GPU 上的每个基准测试选取最佳配置，并使用此配置在其他 GPU 上运行**。例如，我们为序列长度为 512 选取了最佳的 MI250 配置，并在 A100 上进行测量。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/06.webp)

结果如图 4 所示。可以看出，**简单移植配置的影响非常显著**，在 A100 上使用 MI250 配置的情况下，**执行速度减慢到只有 7%**（图 4a）。如图 4b 右侧的缺失值所示，** 有些配置在其他平台上甚至无效**。

> 我们的结果表明，跳过自动调优而仅使用为另一款 GPU 优化的配置时，性能至少会下降 20%。

再次回顾开篇图 1（如下），**图 1 中手动调优的 Triton 内核的较大误差线也表明了 Triton 内核性能的这种高可变性**，在该图中，我们评估了 5 种不同的超参数，这些超参数是在自动调优中使用的配置空间中均匀采样得到的。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/03.webp)

### C. 自动调优对代码多样性的影响

最后，我们希望**量化自动调优是否真的能带来更好的代码生成机会**。

> 为此，我们**分析了在对一个模型和一种序列长度设置** （LLama3.1 - 8b 的注意力层，批量大小 64，序列长度 2048，见图 2）**进行自动调优时评估的所有 450 个 Triton 配置所生成的 PTX 汇编代码**。

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/08.webp)

**分析结果如图 3 所示**，我们**进行了三种定量代码分析**：

1. 首先，我们**统计唯一汇编代码指令的数量**，仅评估操作码和前缀，而不评估操作数。结果显示在蓝色曲线上（左侧 y 轴）。
2. 其次，我们**统计`.cubin` 文件中的 PTX 指令总数**，如绿色曲线所示（右侧 y 轴）。图 3a 显示这两个指标之间没有关联。
3. 此外，单独观察这两条曲线时，并不清楚为什么自动调优器会选择配置#67 作为最佳配置，如红色标记所示。

为了与 CUDA 模板进行比较，我们仅考虑了 30 个包含适用于我们场景的模板的二进制文件（微架构 sm80，数据类型 fp16）。

**比较图 3a 和图 3b 可以发现三个显著差异**：

- 首先，所有 CUDA 库模板使用的 PTX 指令种类较少，例如，唯一 PTX 指令的最大数量（224）不到 Triton 生成数量（475）的一半。
- 其次，生成的.cubin 文件不仅指令种类较少
- 而且长度也显著缩短了一个数量级以上。

这些差距可以用 **Triton 编译器大量利用软件流水线和循环展开** 这一事实来解释。出于篇幅考虑，并且由于 AMD GPU 的有效配置数量要低得多，因此参考价值较小，**我们将代码分析限制在 NVIDIA GPU 上**。

> 总体而言，我们的代码分析表明，**自动调优** 能够更广泛、更高效地探索各种可能情况，**比手动编写的模板库的解决方案空间。**

---

总之，通过我们的实验评估，我们认为**具有全面自动调优功能** 的 **Triton 内核** 提供了性能可移植性，其定义为 “**相同的源代码在各种不同架构上都能以具有竞争力的性能运行** ” [18]。

## 五、结论

通过**这项工作**，我们**证明了自动调优** 是实现当今大语言模型（LLM）堆栈**性能可移植性** 的**关键** 组成部分。

我们**专注于注意力内核**，并表明**将 Triton 即时编译器与整体自动调优相结合**，可以**实现不同代际和不同厂商的 GPU 之间** 的**性能可移植性**。

::: block-1
我们表明：

- **自动调优能够探索多达 15 倍的内核参数配置**
- 在**多个维度上生成** 显著更多样化的代码
- 甚至**比厂商优化的实现快 2.3 倍**
- 同时将**内核代码大小缩小 70 倍**
- 并消除手动代码优化
:::

## 附录1：vllm-ibm-triton-lib

我们在 `ibm.biz/vllm-ibm-triton-lib` 上开源了我们的实现。在 `ibm-triton-lib` 目录下的代码中，有多个性能优化点值得关注。这些优化主要集中在 Triton 内核实现、内存访问模式和自动调优等方面。以下是关键的性能优化点分析：

### 1. Triton 内核优化

#### 3D 分页解码注意力实现

- **`triton_paged_decode_attention_3d.py`** 中的 `kernel_paged_attention_3d` 函数实现了高效的注意力计算内核。
- **并行优化**：利用 Triton 的并行计算能力，通过多线程并行处理不同的注意力头和序列位置，提高计算效率。
- **内存访问优化**：

#### 内存布局优化

- 内核中使用的内存布局（如张量排列方式）经过优化，以匹配 GPU 的内存访问模式，减少内存带宽瓶颈。
- 通过共享内存（shared memory）进一步提高数据访问速度，减少全局内存访问。

### 2. 自动调优与性能适配

#### triton-dejavu 集成

- 项目依赖 triton-dejavu 库来减少自动调优的开销。
- **预计算调优参数**：通过存储和重用预计算的调优参数，避免每次运行时都进行耗时的自动调优过程，从而显著提高性能。
- **平台感知优化**：针对不同的 GPU 平台和配置，自动选择最优的内核参数，确保在各种环境下都能获得最佳性能。

#### 元数据生成优化

- **`metadata_fn` 函数**：为内核生成元数据，包括浮点运算次数和字节数，帮助调度器做出更优的调度决策。
- 通过准确的元数据，系统可以更好地评估内核性能，并为不同的工作负载选择最合适的执行配置。

### 3. 计算效率优化

#### 减少冗余计算

- 内核实现中避免了不必要的计算，例如在注意力计算中，只计算需要的部分，减少计算开销。
- 通过算法优化（如 FlashAttention 类似的方法）减少中间结果的存储和计算，提高整体效率。

#### 优化的矩阵运算

- 注意力计算中的矩阵乘法和累加操作经过优化，利用 GPU 的张量核心（Tensor Cores）进行高效计算。
- 通过调整块大小和线程布局，最大化 GPU 计算资源的利用率。

### 4. 内存管理优化

#### KV 缓存优化

- 在 `triton_attn.py` 中，KV 缓存的管理和使用经过优化，减少内存分配和释放的开销。
- 通过复用内存块和优化缓存策略，降低内存碎片，提高内存使用效率。

#### 高效的数据加载

- 内核中实现了高效的数据加载模式，例如批量加载和预取（prefetching）技术，减少数据加载的等待时间。

### 5. 后端集成与平台适配

#### Triton 平台插件

- **`platform.py`** 中的 `TritonPlatform` 类提供了与 vLLM 平台的集成，确保 Triton 后端能够高效地与 vLLM 协同工作。
- 通过平台插件机制，系统可以根据硬件环境和工作负载特性选择最优的注意力实现。

#### CUDA 版本兼容性处理

- **`utils/triton_utils.py`** 中的工具函数处理了不同 CUDA 版本之间的兼容性问题，确保内核在各种环境下都能正常运行并保持高性能。

### 6. 微基准测试框架

- 项目提供的微基准测试框架（`scripts/benchmark.py`）允许开发者评估不同配置和参数下的内核性能，从而进一步优化代码。
- 通过不断测试和调整，开发者可以找到针对特定硬件和工作负载的最优参数配置。

---

`ibm-triton-lib` 目录下的代码通过 Triton 内核优化、内存访问模式改进、自动调优技术和高效的后端集成，实现了显著的性能提升。这些优化措施使得基于 Triton 的注意力计算在 vLLM 中能够达到接近硬件极限的性能表现，同时保持了良好的通用性和适应性。

## 附录2：triton-dejavu 项目

我们将工作成果开源（github.com/IBM/triton-dejavu/），`triton-dejavu` 是一个用于减少 triton-lang 自动调优开销的框架，特别适用于已知的部署场景。以下是对该仓库的详细介绍：

### 主要功能

1. **自动调优状态的存储和恢复**：基于 Triton 自动调优器，可将自动调优状态存储在 JSON 文件中，并在需要时安全恢复，从而将 Triton 自动调优器的开销降为零，使自动调优器能在生产环境中使用。
2. **配置空间（ConfigSpaces）**：允许用户定义一个配置空间，自动调优器会对该空间进行详尽探索。
3. **贝叶斯优化**：加速自动调优过程。
4. **启发式方法结合**：允许将启发式方法与自动调优器结合使用。
5. **JIT 工件缓存**：提供一个 JIT 工件缓存，方便在使用 Triton 时进行缓存管理。

### 核心模块介绍

1. `triton_dejavu/__init__.py`：定义了框架的版本号，并导入了一些核心模块和全局变量。
2. `triton_dejavu/autotuner.py`：包含 `ConfigSpace` 类，用于表示自动调优器可评估的内核配置空间。它可以生成所有可能的有效配置列表，并支持将配置空间转换为 Bohb 配置空间。
3. `triton_dejavu/cache_manager.py`：实现了自定义的 Triton 缓存管理器 `CustomCacheManager`，允许为每个进程和每次 `triton-dejavu` 运行设置不同的缓存目录。
4. `triton_dejavu/dejavu_storage.py`：包含 `DejavuStorage` 类，用于存储和管理自动调优器的缓存数据。它会根据多个因素（如 CUDA 运行时版本、Triton 版本、GPU 类型等）来确定缓存的适用性。
5. `triton_dejavu/jit_cache.py`：实现了 `JitCache` 类，用于缓存 JIT 编译的内核。它可以减少重复编译的开销，并支持静态和动态模式。

## 参考文献

::: block-1
[1] T. Dao et al. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.” (Jun. 23, 2022), [Online]. Available: http://arxiv.org/abs/2205.14135 (visited on 01/24/2024), pre-published.

[2] T. Dao. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” (Jul. 17, 2023), [Online]. Available: http://arxiv.org/abs/2307.08691 (visited on 03/20/2024), pre-published.

[3] J. Shah et al. “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.” (Jul. 12, 2024), [Online]. Available: http://arxiv.org/abs/2407.08608 (visited on 10/17/2024), pre-published.

[4] W. Kwon et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” (Sep. 12, 2023), [Online]. Available: http://arxiv.org/abs/2309.06180 (visited on 01/24/2024), pre-published.

[5] S. Hooker, “The hardware lottery,” Communications of The Acm, Nov. 2021. DOI: 10.1145/3467017.

[6] A. Vaswani et al., “Attention is All you Need,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017.

[7] AMD Triton kernels team. “Triton flash attention.py.” (2024), [Online]. Available: https://github.com/rasmith/vllm/blob/71f89c5595a6d1c0b9ce29b68344959cb69aa21a/vllm/attention/ops/triton_flash_attention.py (visited on 03/10/2025).

[8] Z. Ye et al. “FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.” (Jan. 2, 2025), [Online]. Available: http://arxiv.org/abs/2501.01005 (visited on 01/20/2025), pre-published.

[9] B. Ringlein et al., “Advancing compilation of DNNs for FPGAs using operation set architectures,” IEEE Computer Architecture Letters, Jan. 2023. DOI: 10.1109/LCA.2022.3227643.

[10] D. Diamantopoulos et al., “Agile autotuning of a transprecision tensor accelerator overlay for TVM compiler stack,” in Proceedings of the 30 IEEE International Conference on Field-Programmable Logic and Applications (FPL), Gothenburg, Sweden: IEEE, Aug. 31–Sep. 4, 2020. DOI: 10.1109/FPL50879.2020.00058.

[11] T. Moreau et al., “A HardwareSoftware blueprint for flexible deep learning specialization,” IEEE Micro, Sep. 2019. DOI: 10.1109/MM.2019.2928962.

[12] P. Tillet et al., “Triton: An intermediate language and compiler for tiled neural network computations,” in Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, Phoenix AZ USA: ACM, Jun. 22, 2019. DOI: 10.1145/3315508.3329973.

[13] L. Zheng et al. “SGLang: Efficient Execution of Structured Language Model Programs.” (Jun. 5, 2024), [Online]. Available: http://arxiv.org/abs/2312.07104 (visited on 09/16/2024), pre-published.

[14] PyTorch Community. “PyTorch Inductor (Algorithm Selection).” (2025), [Online]. Available: https://github.com/pytorch/pytorch/tree/main/torch/inductor/select_algorithm.py (visited on 10/03/2025).

[15] B. Ringlein. “[RFC] ”autotuner deja-vu” save and restore autotuner cache persistently.” (May 28, 2024), [Online]. Available: https://github.com/tritonlang/triton/issues/4020 (visited on 03/10/2025).

[16] B. Maher. “Cache autotune timings to disk.” (Mar. 20, 2025), [Online]. Available: https://github.com/triton-lang/triton/pull/6261 (visited on 03/20/2025).

[17] PyTorch Community. “PersistentCache.” (2025), [Online]. Available: https://github.com/pytorch/pytorch/blob/main/torch/inductor/codecache.py (visited on 03/10/2025).

[18] S. J. Pennycook et al. “A Metric for Performance Portability.” (Nov. 22, 2016), [Online]. Available: http://arxiv.org/abs/1611.07409 (visited on 03/21/2025), pre-published.
:::

![文章插图](/articles/2025-05-25-autotuning-gpgpu-1c4ae988/images/09.webp)
