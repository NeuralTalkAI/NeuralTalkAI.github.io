关键词：**Hexagon-MLIR**、MLIR、**Qualcomm Hexagon NPU**、:mdnice-bold-italic[PyTorch / Triton]、**算子融合**、生成式编译

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/01.webp)

- **Hexagon-MLIR: An AI Compilation Stack For Qualcomm’s Neural Processing Units (NPUs)**
- https://arxiv.org/pdf/2602.19762
- **代码：https://github.com/qualcomm/hexagon-mlir**
- 1.2 万字，阅读 40 分钟，**播客 20 分钟**

相关推荐

- **[高通端侧生成式AI：Hexagon NPU架构创新与异构实现98%性能提升、0.6秒图像生成与20tokens/sLLM推理](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891580&idx=1&sn=dafeb4d9f81e0675149ef8293a7fdba4&scene=21#wechat_redirect)**
- **[评估高通 100 Ultra 加速卡在 HPC 集群中的 LLM 推理性能与能效表现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890141&idx=1&sn=b1a763f1b960132a6fca7158194725f3&scene=21#wechat_redirect)**
- **[HeteroLLM：利用移动端 SoC 实现 NPU-GPU 并行异构 LLM 推理！以 高通8 Gen 3的NPU GPU为例](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888774&idx=1&sn=dc4a0c912248f7f3f1a48806d2d7bfe9&scene=21#wechat_redirect)**

> 在本文中，我们提出了 Hexagon-MLIR[17,18]，这是一个开源的编译栈，面向高通 Hexagon 神经处理单元[10]，为 Triton 内核[24]和 PyTorch 模型[15]的下推编译提供统一支持。

该编译器基于 MLIR 框架[11]构建，:mdnice-bold-italic[通过执行一系列结构化的编译 pass，充分利用 NPU 的架构特性来加速 AI 工作负载。它能实现新的 Triton 内核（手写内核或来自 PyTorch 2.0[2]的子图）向目标硬件的快速部署，提供从内核到二进制文件的自动化编译能力。]

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/02.webp)

::: image-caption
该图呈现了 Hexagon-MLIR 的整体编译流程框架，是整个编译栈的核心视觉化呈现，清晰展示了 PyTorch 模型和 Triton 内核分别通过 Torch-MLIR、triton-to-linalg 转换为 MLIR Linalg 算子的统一入口，为后续所有优化、下推编译遍提供了结构化基础，确保语义保留的同时完成多阶段 IR 转换。
:::

::: block-1
通过接收 Triton 内核，我们**生成超大型内核（mega-kernel），最大化 NPU 紧密耦合内存（TCM，Tightly Coupled Memory）中的数据局部性**，减少基于库的方法中固有的带宽瓶颈。**这一方案是对我们商用工具链的补充，为开发者提供了一个基于 MLIR 的开源编译栈**，使其能够通过更灵活的方式提升 AI 编译能力。
:::

通过多层优化技术——包括向量化（最高 63.9 倍加速）、多线程（最高 3.95 倍加速）、双缓冲（最高 15.4 倍加速）以及 TCM 优化，Hexagon-MLIR 能够充分利用 NPU 的硬件特性。

Hexagon-MLIR 目前仍在开发中，我们正持续为其添加更多的优化手段和功能。

## 本文目录

- 关键问题
  - 问题一：Hexagon-MLIR 声称采用“生成式方法”（Generative Approach）来替代传统的算子库方法，但其性能是否真的能够逼近手写汇编的水平？在哪些关键场景下仍存在显著差距？
  - 问题二：Hexagon-MLIR 的“融合优先”策略在提升局部性和减少 DRAM 访问方面具有理论优势，但其在实际部署中是否可能因 TCM 容量限制或 DMA 调度开销而失效？
- 一、引言
- 二、文章结构
- 三、Hexagon-MLIR：概述
  - 3.1 域定义
  - 3.2 语义保留
  - 3.3 Hexagon-MLIR 编译 Pass流水线
- 四、Hexagon-MLIR：核心细节
  - 4.1 Triton 到 Linalg 的转换
  - 4.2 线性代数通用操作（linalg.generic）
  - 4.3 算子融合
  - 4.4 面向存储层次的分块
  - 4.5 多线程
  - 4.6 双缓冲
  - 4.7 数学库
  - 4.8 其他优化与下推编译
- 五、实验结果与分析
  - 5.1 向量化加速比
  - 5.2 单线程与多线程对比
  - 5.3 双缓冲优化
  - 5.4 多编译 Pass的相互作用
- 六、相关工作
- 七、结论
- 参考文献

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/03.webp)

## 关键问题

### 问题一：Hexagon-MLIR 声称采用“生成式方法”（Generative Approach）来替代传统的算子库方法，但其性能是否真的能够逼近手写汇编的水平？在哪些关键场景下仍存在显著差距？

> 作者多次强调传统算子库方法虽然性能高但开发周期长、难以应对新兴算子组合，而 Hexagon-MLIR 通过融合、向量化、多线程和双缓冲等编译优化手段，能够自动生成高效内核。然而，文中并未提供与手写算子库（如 QHL）或手写汇编的:mdnice-bold-italic[直接性能对比。尤其是在涉及复杂数据流、非规则访存（如 TopK、动态形状）或高度调优的矩阵运算时，编译器自动生成的代码是否能达到手工调优的 90%以上？是否存在某些场景下编译优化反而引入 overhead 或无法有效利用硬件特性？]

生成式方法性能逼近手写汇编，但在复杂场景仍有差距。

Hexagon-MLIR 的生成式编译通过融合、向量化、多线程、双缓冲等优化，在规整的 tensor 运算上取得了显著加速。例如 GELU 在 float16 下获得 63.9 倍向量化加速，RMS‑Norm 达 46.5 倍，表明编译器能有效利用 HVX 向量单元和多级存储层次。**这些数据间接说明其生成的代码已接近硬件峰值，尤其对于元素级操作和固定形状的规约操作。**

然而，与手写汇编相比，仍有若干场景存在差距：

- **不规则数据流**：如 TopK、动态形状或稀疏操作，编译器难以自动推导最佳内存布局和线程划分，而手工库可针对性调优。
- **复杂算子融合**：长算子链融合后可能引入数据依赖冲突或寄存器溢出，导致实际性能低于手工拆分的库调用。文中虽提及“生成式方法”能处理任意长链，但【未给出】类似 Transformer 中复杂融合的实测对比。
- **专用硬件特性利用**：手写汇编可精确控制向量寄存器分配、流水线停顿、指令序等微观细节，编译器受限于 IR 抽象层次，可能无法完全发掘硬件微架构潜力。例如，论文表 1 中 SiLU 在 float16 下性能低于 float32，推测因内部精度提升引入额外开销，手工代码则可避免此类问题。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/04.webp)

::: image-caption
该表格实测了多款 AI 核心算子在 Hexagon NPU 的 HVX 向量化后的性能提升，覆盖激活、归一化、注意力等典型算子，还包含基础的二维向量加法。float16 数据类型因更适配 HVX 的向量宽度和数据打包规则，加速比远高于 float32；Flash Attention 加速比偏低，是因其包含复杂的归约操作和非连续内存访问逻辑，限制了向量化效率。
:::

::: block-1
综上，对于规整、计算密集型算子，Hexagon‑MLIR 有望达到手写性能 90% 以上；但对于不规则、动态或需深度调优的场景，自动编译仍存在性能缺口，需结合手工库或 DSL 扩展来弥补。
:::

### 问题二：Hexagon-MLIR 的“融合优先”策略在提升局部性和减少 DRAM 访问方面具有理论优势，但其在实际部署中是否可能因 TCM 容量限制或 DMA 调度开销而失效？

> 论文中提到通过算子融合和分块（tiling）将数据保留在 TCM 中，以减少 DDR 带宽瓶颈。但在实际 LLM 或 MoE 模型中，某些算子链可能非常长，导致中间结果无法完全容纳在有限的 TCM 中，进而引发频繁的 DMA 交换。文中虽提到双缓冲（double buffering）来隐藏 DMA 延迟，但并未讨论**当融合区域过大时，编译器如何自动决定分裂点（split point）**，以及这种决策是否总能优于手动分割。此外，DMA 调度的同步开销在多核场景下是否可能成为新的瓶颈？

融合优先策略面临 TCM 与 DMA 挑战，但双缓冲与智能分裂可缓解。

::: block-1
Hexagon‑MLIR 通过算子融合将中间数据保留在 TCM，减少 DDR 访问。然而，TCM 容量有限（通常数百 KB），长算子链可能导致中间结果无法完全容纳，迫使编译器插入 DMA 换入换出，反而增加开销。:mdnice-bold-italic[论文中虽采用分块（tiling）将大张量切分为 TCM 尺寸的 tile，但融合区域若跨越多个算子且各算子对 tile 形状要求不同，可能产生大量小 tile 或跨 tile 依赖，引发频繁的 DMA 传输。]
:::

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/05.webp)

::: image-caption
该图以基础的二维向量加法为测试用例，展示了其在融合、向量化、多线程、双缓冲等优化编译遍依次作用下的性能变化曲线，直观体现了各优化手段的叠加增益效应，其中向量化带来基础且显著的性能提升，多线程与双缓冲则进一步挖掘了硬件的并行执行潜力，验证了编译遍流水线的有效性。
:::

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/06.webp)

::: image-caption
该图针对计算密集型的指数级数内核，开展了多优化编译遍组合的性能拆解分析，向量化与多线程优化大幅降低了内核的计算耗时，却让内存传输开销成为新瓶颈，而双缓冲优化通过重叠计算与 DMA 传输，有效缓解了这一问题，各编译遍的协同作用显著提升了内核整体吞吐率，为计算密集型算子的优化提供了参考范式。
:::

双缓冲（double buffering）通过 ping‑pong 机制隐藏 DMA 延迟，如文中图 5、6 所示，叠加双缓冲后可进一步提升性能。**但该技术有效的前提是计算时间足够覆盖下一次 DMA 预取；若计算量过小（如简单逐元素操作），DMA 仍会成为瓶颈**。此外，多线程环境下多个硬件上下文同时发起 DMA，可能竞争带宽，导致队列拥塞，如 1M 元素时多线程加速比略有下降（图 3），暗示 DMA 资源饱和。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/07.webp)

::: image-caption
该图量化展示了 GELU 内核多线程相对单线程的加速比随数据规模的变化趋势，32K 规模是性能拐点，此后加速比稳步提升至 512K 时的 3.95 倍，1M 元素时加速比略有回落，这一变化反映了大尺度数据下 TCM 存储和 DMA 传输队列的资源压力，为后续分块参数的重调优提供了直接的实验依据。
:::

编译器应对策略包括：

- **智能分裂**：当融合区域超过 TCM 容量时，自动选择合适的分割点（split point），将过长的链拆分为多个子图，每个子图能完全装入 TCM。:mdnice-bold-italic[文中虽未详述分裂算法，但提及“融合作为第一类编译 pass”，暗示可分析数据依赖和内存占用，启发式决定融合边界。]
- **自适应分块**：根据算子类型动态调整 tile 尺寸，使计算与 DMA 比例最优。例如，:mdnice-bold-italic[对计算密集型算子采用较大 tile，对访存密集型采用较小 tile 以提高并行度。]
- **DMA 调度优化**：通过标记优先级或使用多通道 DMA，避免多核冲突。文中双缓冲第二阶段将 memref.copy 转为 dma_start/wait，并分配独立 tag，为细粒度同步打下基础。

::: block-1
总之，融合优先策略在容量与带宽约束下并非万能，但结合编译器动态分析和双缓冲等机制，可在绝大多数场景中取得收益。未来需进一步优化分裂决策和 DMA 调度，以应对日益复杂的模型结构。
:::

## 一、引言

> Hexagon-MLIR 是一套用于在高通 Hexagon 神经处理单元（NPU）[10]上编译和执行 Triton 内核[24]与 PyTorch 计算图[15]的编译栈，该栈的部分模块近期已开源[17,18]。

该编译器基于 MLIR 框架[11]构建，通过执行一系列结构化的转换、优化和下推编译 Pass，充分利用 NPU 的各类架构特性——包括：

- 硬件向量扩展（HVX，Hardware Vector eXtension）；
- 可利用硬件线程和多个 HVX 上下文的多线程技术；
- 包含紧密耦合内存（TCM）的存储层次结构；
- 直接内存访问（DMA，Direct Memory Access）传输与计算的交错执行；
- 经过优化的向量化数学库；
- 以及高吞吐的矩阵乘法引擎——以此实现 AI 工作负载的加速。

::: block-1
Hexagon-MLIR 采用生成式方法：**将算子融合作为一级编译 Pass，能够为任意长的算子链生成专用内核。**
:::

传统的机器学习（ML）编译方法通常依赖算子库，这类库能实现峰值性能，但需要耗费大量时间进行细致的手动实现[4,21]。

- 随着新的算子和优化方法不断通过 Triton 及其他领域特定语言（DSL，Domain Specific Language）[19,25]被开发出来，从 PyTorch 算子、手写或由 Inductor 生成的 Triton 内核，到生成经过合理优化的二进制文件，这条快速的自动化路径能带来显著的部署速度优势。
- 此外，算子库会成为优化的不透明壁垒，这种碎片化问题会导致数据在算子调用之间往返于双倍数据率同步动态随机存取存储器（DDR，Double Data Rate Synchronous Dynamic Random Access Memory），进而造成带宽瓶颈。就像上世纪 90 年代提出的区域编译[7,9]（该技术实现了数据局部性和并行性的挖掘）一样，**通过融合[28]实现的 Triton 内核，能挖掘出算子库无法利用的数据局部性和并行性潜力。手写汇编代码（极致性能的代码）在单操作下的执行速度更快，但面对快速迭代的 AI 模型，自动代码生成才是唯一具备可扩展性的解决方案**。
- 此外，Triton 生态系统正通过更高层的抽象不断发展，例如 Helion[16]，它提供了以 PyTorch 为中心的领域特定语言，可编译为 Triton 代码，进一步降低了内核开发的门槛。

Hexagon-MLIR 为接收这些大尺度计算区域（例如通过 PyTorch Inductor[2,15]生成的计算区域）并将其编译为二进制文件奠定了架构基础。

> 如何高效编译新的大语言模型（LLM，Large Language Model）和近期提出的混合专家模型（MoE，Mixture-of-Experts）架构[6]，仍是一个活跃的研究方向。

近期的研究表明，Triton 能在注意力机制[5,20,26]等大语言模型核心组件上实现当前最优的性能。此外，多项式组合激活函数（PolyCom）[31]等新型激活函数被提出，用于提升 Transformer 模型的表达能力和动态特性，这也进一步凸显了开发灵活编译框架的必要性——这类框架需能高效支持新兴的模型架构创新。**借助 PyTorch 2.0 的 Inductor 流程，Triton 内核可被自动生成；同时，开发者也能通过 Triton 语言快速高效地编写新的算子或算法**。因此，基于手工优化算子库的方法不具备可扩展性。

> 本编译器基于 MLIR[11]构建，MLIR 并非特定的编译器，而是一个可用于构建编译器的框架。

- IREE[27]就是一款基于 MLIR 构建的 AI 编译器，它通过硬件抽象层（HAL，Hardware Abstraction Layer）实现了跨硬件的可移植性。
- 另一款基于 MLIR 的编译器是张量处理原语库（TPP，Tensor Processing Primitives）[8]，其面向英特尔和超威半导体的处理器，**据作者称，该编译器“能达到手工优化等效程序 90%的性能”。**
- 编译期优化方法也被应用于分布式训练优化，DeepCompile[23]就是典型案例，它通过基于性能分析的优化遍，提升了分布式场景下通信与计算的重叠执行效率。:mdnice-bold-italic[虽然对手工优化库进行封装（例如英特尔针对 libxsmm 库的做法）能保证标准操作的峰值性能，但无法解决现代 Transformer 模型中大量的算子融合模式长尾问题，还会面临所需内核数量组合式爆炸的难题。]

## 二、文章结构

> 本文的剩余部分结构如下：

- 第 3 节**概述 Hexagon-MLIR 的编译流程**，形式化定义包括语义保留、编译 Pass流水线架构在内的核心概念；
- 第 4 节详细**阐述编译器的核心编译 Pass** ——包括 Triton 到 Linalg 的转换、线性代数通用操作（linalg.generic）抽象、算子融合、面向存储层次的分块（tiling）、多线程、双缓冲、数学库集成以及其他优化手段。**全文将以 softmax 和 GELU 为典型内核案例，阐释代码在编译流水线中逐步转换的过程；**
- 第 5 节展示实验结果与性能分析，验证各个独立优化手段（向量化、多线程、双缓冲）的有效性，以及在内核从计算密集型转为内存密集型的过程中，这些优化手段之间的相互作用；
- 第 6 节讨论深度学习编译器和优化技术领域的相关研究；
- 第 7 节总结本文的研究贡献，并展望未来的研究方向。

## 三、Hexagon-MLIR：概述

> 图 1 展示了 Hexagon-MLIR 的编译流程概览。PyTorch 模型通过 Torch-MLIR[1]转换为 Linalg 算子及其他 MLIR 核心算子；同理 MLIR 核心算子；同理，Triton 内核通过 triton-to-linalg[13]转换为 MLIR 算子。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/08.webp)

::: image-caption
该图呈现了 Hexagon-MLIR 的整体编译流程框架，是整个编译栈的核心视觉化呈现，清晰展示了 PyTorch 模型和 Triton 内核分别通过 Torch-MLIR、triton-to-linalg 转换为 MLIR Linalg 算子的统一入口，为后续所有优化、下推编译遍提供了结构化基础，确保语义保留的同时完成多阶段 IR 转换。
:::

### 3.1 域定义

设 $\mathcal{M}$ 为 PyTorch 模型的集合，$\mathcal{K}$ 为 Triton 内核的集合，$\mathcal{L}$ 为 MLIR Linalg 算子的集合，则存在如下映射关系：

$$
f: \mathcal{M} \to \mathcal{L} (\text{Torch-MLIR转换})
$$

$$
g: \mathcal{K} \to \mathcal{L} (\text{Triton到Linalg的转换})
$$

### 3.2 语义保留

对于任意的 $m \in \mathcal{M}$ 和 $k \in \mathcal{K}$，向 Linalg 算子下推编译的过程需保证指称语义的保留，即：

$$
\lbrack\!\lbrack m \rbrack\!\rbrack=\lbrack\!\lbrack f(m) \rbrack\!\rbrack, \quad \lbrack\!\lbrack k \rbrack\!\rbrack=\lbrack\!\lbrack g(k) \rbrack\!\rbrack
$$

部分与精度相关的选择（如量化）可能会弱化严格的语义保留，但机器学习模型通常对这类偏差具有容忍性。此外，若 MLIR 计算图中出现自定义命名的算子，编译器会将其转换为语义等价的 MLIR 核心算子。

### 3.3 Hexagon-MLIR 编译 Pass流水线

编译过程可建模为不同中间表示（IR，Intermediate Representation）之间的一系列态射，每个编译 Pass将一种中间表示转换为另一种：

$$
Obj(IR) = \{ {IR}_{\alpha }| \alpha \text{ 为中间表示的抽象级别}\}
$$

$$
Hom(IR) = \{ p:{IR}_{\alpha }\mapsto {IR}_{\beta }| p\in \text{编译 Pass流水线}\}
$$

编译 Pass至少可分为两类：

- **转换类**：重写中间表示，以实现代码简化（如公共子表达式消除（CSE）、规范化）或性能提升。例如向量化编译 Pass：$p: IR_{\text{标量}} \mapsto IR_{\text{向量化}}$
- **下推类**：降低代码的抽象级别，向目标硬件靠拢。例如将结构化控制流转换为非结构化控制流：$p: IR_{\text{scf}} \mapsto IR_{\text{cf}}$

> Hexagon-MLIR 的编译 Pass流水线包含大量编译 Pass，本文将重点介绍其中的核心编译 Pass，各编译 Pass的符号及说明如下表所示：

| 符号 | 说明 |
| --- | --- |
| C | 规范化、公共子表达式消除（CSE）或常量传播（CP） |
| F | 算子融合 |
| L | 布局转换与传播 |
| M | 多线程 |
| $p_i$ | 某一优化或下推编译 Pass |
| T | 面向存储层次的分块 |
| Q | 量化 |
| V | 向量化及相关编译 Pass |

Hexagon-MLIR 的编译 Pass流水线是作用于输入中间表示的一系列编译 Pass的组合，即：

$$
\Pi=p_{n} \circ \cdots \circ T \circ F \circ \cdots \circ p_{0}: IR_{0} \to IR_{\text{最终}}
$$

本文后续将详细介绍其中部分重要的编译 Pass。

## 四、Hexagon-MLIR：核心细节

> 本节将深入讲解 Hexagon-MLIR 的核心编译 Pass，并通过若干案例辅助说明。

### 4.1 Triton 到 Linalg 的转换

softmax 函数是机器学习中应用广泛的函数，从数学角度，对于给定的 $Z = (z_1, ..., z_K)$，softmax 的计算公式为：

$$
\sigma\left(z_{i}\right)=\frac{\exp \left(z_{i}-\max _{j} z_{j}\right)}{\sum_{k=1}^{K} \exp \left(z_{k}-\max _{j} z_{j}\right)}.
$$

将每个元素减去 $\max_j z_j$ 的操作，能在不改变计算结果的前提下提升数值稳定性。

#### Triton 内核实现

> 下面代码清单 1 展示了实现数值稳定的按行 softmax 的 Triton 内核片段，计算过程为：

- 该内核将每一行的所有元素减去该行的最大值，避免指数运算过程中出现数值溢出；
- 随后通过`tl.exp`计算指数值，并通过按行求和操作`tl.sum`得到归一化因子；
- 最后将每个指数值除以该归一化因子，得到 softmax 概率值。

*代码清单 1：Triton 实现 softmax 的内核片段*

```
@triton.jitdef softmax_kernel(output_ptr , input_ptr ,    row_minus_max = row - tl.max(row , axis =0)    numerator = tl.exp(row_minus_max)    denominator = tl.sum(numerator , axis =0)    softmax_output = numerator / denominator
```

Triton 编译器会先将代码清单 1 中的 Triton 内核下推编译为 Triton-IR，再通过 Triton-to-Linalg 转换器转换为 Linalg 算子，:mdnice-bold-italic[下面代码清单 2 展示了转换后的 Linalg 中间表示，其中包含一系列线性代数通用操作（linalg.generic）。 因该操作是编译流程的核心，本文将先对其进行简要说明。]

*代码清单 2：softmax 片段的 Linalg 中间表示*

```
%9 = linalg.generic{indexing_maps = [affine_map <(d0) -> (d0)>, affine_map <(d0) -> (d0)>,affine_map <(d0) -> (d0) >],iterator_types = ["parallel"]}ins(%3, %8 : tensor <16384 xf32 >, tensor <16384 xf32 >)outs (%4 : tensor <16384 xf32 >) {^bb0(%in: f32 , %in_5: f32 , %out: f32):%17 = arith.subf %in , %in_5 : f32linalg.yield %17 : f32} -> tensor <16384 xf32 >%10 = tensor.empty () : tensor <16384 xf32 >%11 = linalg.generic{indexing_maps = [affine_map <(d0) -> (d0)>, affine_map <(d0) -> (d0)>],iterator_types = ["parallel"]}ins (%9 : tensor <16384 xf32 >) outs (%10 : tensor <16384 xf32 >) {^bb0(%in: f32 , %out: f32):%17 = math.exp %in : f32linalg.yield %17 : f32} -> tensor <16384 xf32 >
```

### 4.2 线性代数通用操作（linalg.generic）

> linalg.generic 是 MLIR 线性代数（Linalg）方言中的核心算子，该算子包含索引映射、迭代器类型、输入输出张量、计算体（linalg.body）等核心组成部分。

#### 索引映射

设 $I \subseteq \mathbb{Z}^{d}$ 为迭代域（一个受仿射约束的整数格），linalg.generic 算子会定义一组有限的仿射映射：

$$
\phi_{\ell}: \mathcal{I} \to \mathbb{Z}^{n_{\ell}}
$$

每个仿射映射定义了对输入或输出张量 $X_{\ell}$ 坐标空间的视图。这些映射共同决定了公共迭代域中的点如何投影到各个操作数的索引集合上。

#### 迭代器类型

迭代域被标记为并行（parallel）或归约（reduction）类型，该标记定义了计算体中沿每个轴的计算方式，linalg.body 的计算逻辑必须与该声明保持一致。

#### 计算负载区域

> linalg.body 是一个基本块，以标量为操作对象，从标量输入生成标量输出并返回。该计算体定义了 linalg.generic 在迭代空间中单个点的计算逻辑，每个标量都会通过索引映射从张量操作数中提取，并作为块参数传入 linalg.body。

linalg.generic 是一种结构化算子，其内部封装了迭代的几何特征（索引集合、仿射映射、迭代器类型），并将具体计算逻辑委托给 linalg.body 实现。**这种设计将数据遍历的控制逻辑与数据上的计算语义解耦，使得在将代码下推编译为显式循环之前，能对其进行规范化的转换优化**。例如，:mdnice-bold-italic[所有命名的 Linalg 算子（如 linalg.add、linalg.conv_2d）都能唯一地转换为 linalg.generic，这一特性是其核心价值所在——它能为机器学习中种类繁多的算子，提供统一的转换、优化和下推编译机制]。尽管 linalg.generic 可基于缓冲区（buffer）执行操作，但最佳实践是使用基于张量的 Linalg（linalg-on-tensors），**本文采用的也是该方式，因其能充分利用静态单赋值（SSA，Static Single Assignment）表示的优势。**

::: block-1
需要说明的是，**部分算子因自身特性——如输出张量形状不固定、存在数据相关的控制流、非仿射映射等** ——无法通过单个 linalg.generic 表示。softmax、前缀和、TopK 等重要的机器学习算子，:mdnice-bold-italic[需要结合其他控制流算子与 linalg.generic 才能完成表示]。

此外，Linalg 中新增了一层中间抽象，即 linalg.contraction（收缩操作）和 linalg.elementwise（按元素操作），这些算子同样能唯一地转换为 linalg.generic。
:::

### 4.3 算子融合

> 算子融合是一种保持语义不变的转换操作，它将多个结构化算子（如 linalg.generic）合并为单个算子，消除中间张量的实例化过程，提升数据局部性。

考虑两个连续的算子操作：

$$
X \stackrel{P}{\to} Y \stackrel{Q}{\to} Z
$$

其中 $P$ 和 $Q$ 为 linalg.generic 算子，对应的迭代域分别为 $\mathcal{I}_{P}$ 和 $\mathcal{I}_{Q}$。生产者算子 $P$ 通过以下映射写入张量 $Y$ 的元素：

$$
\phi_{\text{out }}^{P}: \mathcal{I}_{P} \to \text{dom}(Y),
$$

而消费者算子 $Q$ 通过以下映射读取张量 $Y$ 的元素：

$$
\phi_{\text{in }}^{Q}: \mathcal{I}_{Q} \to \text{dom}(Y)
$$

::: block-1
算子融合的核心是将这两个 linalg.generic 合并为一个算子，这样就无需将张量 $Y$ 的所有元素写回内存，而是在融合后的算子计算体中，读取输入张量 $X$ 的标量值 $x$，生成中间标量值 $y$ 后立即消费该值，生成输出张量 $Z$ 的标量值 $z$。
:::

因此，若：

$$
P= \text{linalg.generic} \left\{\mathcal{I}_{P},\left\{\phi_{\ell}^{P}\right\}, y:=f(x)\right\},
$$

$$
Q= \text{linalg.generic} \left\{\mathcal{I}_{Q},\left\{\phi_{\ell}^{Q}\right\}, z:=g(y)\right\},
$$

则融合后的算子为：

$$
\text{linalg.generic} \left\{\mathcal{I},\left\{\phi_{\ell}\right\}, z:=g(f(x))\right\}.
$$

> 代码清单 3 展示了清单 1 中减法（subf）和指数（exp）算子融合后的中间表示，该案例为了便于说明，采用了恒等映射的简单形式，而实际的算子融合需要考虑广播、转置和重计算等复杂场景。

*代码清单 3：算子融合后 softmax 片段的中间表示*

```
%7 = linalg.generic{indexing_maps = [affine_map <(d0) -> (d0)>,affine_map <(d0) -> (d0)>],iterator_types = ["parallel"]}ins (%3 : tensor <16384 xf32 >) outs (%4 : tensor <16384 xf32 >) {^bb0(%in: f32 , %out: f32):%11 = arith.subf %in , %extracted : f32%12 = math.exp %11 : f32linalg.yield %12 : f32} -> tensor <16384 xf32 >...
```

### 4.4 面向存储层次的分块

> 分块操作会将通常存储在低速大容量的 DDR 内存中的大张量，划分为多个小的分块（tile），并将这些分块复制到高速小容量的片上 TCM 中，随后基于 TCM 中的数据执行 HVX 向量化等计算操作。:mdnice-bold-italic[为抵消 DDR 与 TCM 之间的数据传输延迟，提高 TCM 中读取或生成数据的复用率至关重要。上一节介绍的算子融合能提升向量寄存器级别的数据复用，而通过将张量结果保留在 TCM 中，可进一步提升融合算子间的数据复用率。]

下文将详细介绍 Hexagon-MLIR 中的分块机制：对于一个迭代域为 $I \subseteq \mathbb{Z}^{d}$ 的 linalg.generic 算子，给定分块向量 $t=(t_1, ..., t_d)$ 和可选的交换向量，分块操作会引入两层迭代结构：外层为结构化控制流 for 循环（scf::for）或结构化控制流并行 for 循环（scf::forall），循环体中仍为 linalg.generic 算子，该算子将基于 TCM 中的分块数据执行计算。

**为充分利用存储层次结构，Hexagon-MLIR 中的分块编译 Pass引入了 DDR 和 TCM 两个独立的内存空间，通过 DMA 操作实现两者间的数据传输**。

- 分块操作在基于张量的 Linalg 级别执行，该编译 Pass会通过带内存空间属性的`bufferization.alloc_tensor`操作，插入显式的数据移动指令，这些属性能让编译器后续生成 DDR 与 TCM 之间的 DMA 传输代码。
- 此外，分块编译 Pass还会执行循环交换，将可向量化的维度置于最内层，并为分块后的循环添加注解，为软件流水线等后续优化提供支持。

> 本文将暂时放下 softmax 案例，以高斯误差线性单元（GELU）为新的内核案例，介绍分块、向量化和双缓冲优化。

GELU 的近似计算公式为：

$$
{\displaystyle \text{GELU}(x)\approx 0.5x\left( 1+\tanh\left( \sqrt {\frac {2}{\pi }}(x+0.044715x^{3})\right) \right) }
$$

下面的代码清单 4 展示了经过算子融合和分块后的 GELU 内核，此时尚未执行缓冲区化，因此分块后的 linalg.generic 仍基于张量操作。

*代码清单 4：算子融合和分块后的 GELU 内核*

```
#map = affine_map <(d0) -> (d0)>module attributes {llvm.target_triple = "hexagon"} {func.func @gelu_kernel (%arg0: memref <*xf32 > {tt.divisibility = 16 : i32}, ..) {%cst_2 = arith.constant 7.978850e-01 : f32.. %4 = scf.for %arg8 = %c0 to %c1048576 step %c262144iter_args (%arg9 = %0) -> (tensor <1048576 xf32 >) {%extracted_slice = tensor.extract_slice %3[% arg8] [262144] [1]: tensor <1048576 xf32 > to tensor <262144 xf32 >%5 = bufferization.alloc_tensor ()copy(% extracted_slice) {memory_space = 1 : i64} : tensor <262144 xf32 >..%7 = linalg.generic{indexing_maps = [#map , #map], iterator_types = [" parallel "]}ins (%5 : tensor <262144 xf32 >) outs (%6 : tensor <262144 xf32 >) {^bb0(%in: f32 , %out: f32):%13 = arith.mulf %12, %cst : f32%14 = math.exp %13 : f32%15 = arith.addf %14, %cst_0 : f32%16 = arith.subf %cst_0 , %14 : f32%17 = arith.divf %16, %15 : f32linalg.yield %20 : f32} -> tensor <262144 xf32 >%inserted_slice = tensor.insert_slice %7 into %arg9[%arg8] [262144] [1]: tensor <262144 xf32 > into tensor <1048576 xf32 >scf.yield %inserted_slice : tensor <1048576 xf32 >} {all_parallel , tiled_generic}}}
```

可以看到代码中出现了 scf.for 循环，在每次循环迭代中，通过`tensor.extract_slice`从大的输入张量中提取一个切片，随后将该切片从默认内存空间复制到 TCM 中，linalg.generic 的计算体将基于 TCM 中的数据执行计算。尽管当前计算体的代码为标量形式，但经过后续转换后，将成为支持多线程的 HVX 指令。下文将为该分块后的 linalg.generic 算子添加多线程支持。

### 4.5 多线程

> 高通 NPU 具备硬件多线程能力，可支持硬件向量扩展，可支持硬件向量扩展（HVX）指令的并行执行。:mdnice-bold-italic[向量上下文是一组能支持 HVX 指令独立执行的硬件资源，包括向量寄存器文件和向量谓词寄存器文件，多个硬件线程可并行运行，每个线程占用不同的向量上下文]。本节将为上一节中完成分块的 GELU 内核添加多线程支持。

本文实现了两种多线程方案，一种基于 scf::for 循环实现，另一种基于 linalg.generic 实现，本节将介绍后者。

::: block-1
Hexagon-MLIR 的多线程方案基于 MLIR 的异步（Async）方言实现，该方言提供了结构化的分叉-合并（fork–join）中间表示，核心操作包括：
:::

- `async.execute`：创建一个异步执行区域，返回一个令牌（可选返回计算值），通过`async.yield`完成异步执行；
- `async.await`：挂起当前协程，等待令牌/计算值返回；
- `async.create_group`、`async.add_to_group`和`async.await_all`：实现多令牌的屏障同步。

> 上述异步表示最终会被下推编译为 LLVM 协程（`llvm.coro.id`、`llvm.coro.begin`、`llvm.coro.suspend`、`llvm.coro.end`），并结合 MLIR 异步运行时执行。Hexagon-MLIR 通过两阶段流水线实现 HVX 多线程：

1. **构建虚拟线程**：对计算任务进行划分，通过 scf.forall 编码并行结构；
2. **构建异步线程**：将上述并行结构下推编译为 MLIR 的 Async 方言，该方式能让并行语义保持声明式和可分析性。

**阶段 1**：通过多面体大小启发式算法，分析 linalg.generic 算子并判断并行化的收益；对并行迭代器执行分块操作，通过块分布（当迭代范围能被线程数整除时）或块循环分布（用于平衡线程计算量）的方式，生成 scf.forall（虚拟线程）。

**阶段 2**：将每个 scf.forall 重写为显式的异步分叉-合并模式。首先通过`async.create_group(N)`为 N 个线程建立屏障；随后通过外层 scf.for 循环遍历所有分块，每次迭代通过以下操作生成异步线程：

$$
\text{async.execute} \{ /* \text{for all分块的计算体} */ \} \mapsto \text{token}
$$

通过以下操作将生成的令牌添加到屏障组中：

$$
\text{async.add_to_group(group, token)}
$$

最后通过`async.await_all(group)`确保所有异步线程执行完成后，再执行后续的依赖计算。该方式在保留原始并行语义的同时，为基于协程的代码生成做好了准备。

> 下面代码清单 5 展示了多线程化后的 GELU 内核中间表示，其输入为清单 4 中分块后的 GELU 内核中间表示。

*代码清单 5：创建异步线程后的 GELU 内核*

```
%12 = async.create_group %11 : !async.groupscf.for %arg9 = %c0_10 to %c262144_11 step %c8_12 {%token = async.execute {%subview_14 = memref.subview %2[% arg9] [8] [1]: memref <262144 xf32 , 1> to memref <8xf32 , strided <[1], offset: ?>, 1>scf.for %arg10 = %c0 to %c8 step %c1 {%14 = memref.load %subview_14 [%arg10] : memref <8xf32 , strided <[1], offset: ?>, 1>%15 = arith.mulf %14, %cst_1 fastmath <fast > : f32%21 = math.exp %20 fastmath <fast > : f32%22 = arith.addf %21, %cst_0 fastmath <fast > : f32%23 = arith.subf %cst_0 , %21 fastmath <fast > : f32%24 = arith.divf %23, %22 fastmath <fast > : f32memref.store %27, %subview_15 [%arg10] : memref <8xf32 , strided <[1], offset: ?>, 1>}async.yield}%13 = async.add_to_group %token , %12 : !async.token}async.await_all %12
```

### 4.6 双缓冲

> 软件流水线中的双缓冲技术，通过让计算与数据移动操作重叠执行，缓解层次化存储系统中的内存延迟问题。该技术借助两个缓冲区（乒乓缓冲区）实现：当一个缓冲区被用于计算时，另一个缓冲区同时执行数据填充，随后交换两个缓冲区的角色，从而实现数据传输与计算的并行执行。

在以张量操作为主的 AI/ML 工作负载中，数据量往往超过高速片上内存的容量，双缓冲技术因此变得至关重要。:mdnice-bold-italic[通信与计算的交错执行，能让内存密集型内核转变为对延迟不敏感的流水线，充分挖掘内存单元与计算单元之间的并行性。类似的重叠执行技术也被应用于分布式 AI 系统中，基于编译器的方法能为分布式工作负载实现原生的重叠执行优化[29]。]

> Hexagon-MLIR 中的双缓冲实现分为两层编译 Pass，将结构化的软件流水线转换与异步数据移动解耦：

#### 阶段 1：结构化转换

该阶段会寻找符合“标准形式”的单缓冲分块循环 scf.for，这些循环带有注解，表明其由分块后的 linalg.generic 生成。分块操作会为每个分块留下一组调度三元组：

$$
\text{memref.subview} \mapsto \text{memref.alloc} \mapsto \text{memref.copy},
$$

同时还会生成计算区域和写回序列。

::: block-1
双缓冲编译 Pass会构造乒乓执行流程：

- 首先通过一个带判断的序章（prologue），将第一个分块预取到主缓冲区（ping buffer）中；
- 随后重构内核，生成两个互斥的子内核（分别标注为`db_ping_kernel`/`db_pong_kernel`）。

每个子内核会为下一次迭代发起预取操作，将计算逻辑克隆并映射到当前的主/副缓冲区（ping/pong buffer），同时通过在当前归纳变量处重新实例化子视图，重构数据存储操作。通过一个驻留在内存中的布尔标志位控制主/副缓冲区的切换，每次迭代更新该标志位；同时通过`db_generic`（唯一标识）、`db_prologue`/`db_prefetch`等属性为后续编译 Pass提供中间表示锚点。

该设计能让转换过程具有确定性且无风险，在保留原始分块语义的同时，实现通信（预取）与计算的重叠执行，且无需绑定到特定的异步传输方式。
:::

#### 阶段 2：异步 DMA 集成

> 该阶段会实现真正的异步数据传输和同步操作：

- 通过解析阶段 1 中插入的属性，从中间表示中恢复序章、内核以及主/副缓冲区的调度序列。随后，该编译 Pass会将所有预加载和预取的`memref.copy`操作重写为`memref.dma_start`——为主/副缓冲区分配不同的标签，确保每个阶段能独立等待数据传输完成；
- 并在对应的预取块后立即插入`memref.dma_wait`，确保数据传输完成后再执行计算。

对于数据写回流程，同样将每个`memref.copy`替换为`dma_start`和后续的`dma_wait`（使用独立分配的写回标签），并在内核执行完成后，为所有标签执行平衡的释放操作。

::: block-1
该设计将合法性/调度（阶段 1）与传输语义（阶段 2）解耦，能通过显式的`dma_start`/`dma_wait`隐藏延迟，且不会干扰计算相关的中间表示。
:::

> 清单 6 展示了双缓冲优化后的中间表示，代码中添加了 C++风格的注释，并对部分变量进行了自定义命名，便于理解。

*代码清单 6：双缓冲优化后的 GELU 内核 MLIR 代码*

```
// -----// IR Dump After hexagon -double -buffer -generic -s2func.func @gelu_kernel (%arg0: memref <*xf32 > {tt.divisibility = 16 : i32}, ..) {%false = arith.constant false// allocate ping -pong buffers on TCM%ping_buffer = memref.alloc () {alignment = 2048 : i64} : memref <262144 xf32 , 1>%pong_buffer = memref.alloc () {alignment = 2048 : i64} : memref <262144 xf32 , 1>%dma_ping_tag = memref.alloc () : memref <1xi32 >%dma_pong_tag = memref.alloc () : memref <1xi32 >.../// Double Buffering - Prologuescf.if %loop_executes_at_least_once {%subview = memref.subview %Input [0] [262144] [1]: memref <1048576 xf32 , ..>> to memref <262144 xf32 ,..>>memref.dma_start %subview [%c0], %ping_buffer [%c0],%c262144 ,% dma_ping_tag [%c0]: memref <262144 xf32 ,.>>, memref <262144 xf32 ,1>, memref <1xi32 >} {db_generic = 0 : i64 , db_prologue}/// Double Buffering - Kernel (main loop)scf.for %arg8 = %c0 to %c1048576 step %c262144 {%is_ping_stage = memref.load %ping_pong_toggle [] : memref <i1>scf.if %is_ping_stage {scf.if %is_not_last_iteration {// Prefetch to pong buffers for next iteration.%subview_17 = memref.subview %Input [%4] [262144] [1]: memref <1048576 xf32 , ..>> to memref <262144 xf32 , ..>>memref.dma_start %subview_17 [%c0], %pong_buffer [%c0],%c262144 , %dma_pong_tag [%c0]: memref <262144 xf32 , ..>>, memref <262144 xf32 , 1>, memref <1xi32 >...} {db_prefetch}memref.dma_wait %dma_ping_tag [%c0], %c262144 : memref <1xi32 >...// Do multi -threaded HVX execution on ping buffers on TCMscf.forall (%arg9) = (0) to (262144) step (65536) {%subview_17 = memref.subview %ping_buffer [%arg9] [65536] [1]: memref <262144 xf32 , 1> to memref <65536 xf32 , ...>, 1>scf.for %arg10 = %c0 to %c65536 step %c32 {%subview_19 = memref.subview %subview_17 [%arg10] [32] [1]: memref <65536 xf32 ,..>, 1> to memref <32xf32 , ..>, 1>%15 = math.exp %14 fastmath <fast > : vector <32xf32 >vector.transfer_write ... : vector <32xf32 >, memref <32xf32 , ..>, 1>}}memref.dma_start ... // write -back ping buffer results.memref.store %false , %ping_pong_toggle [] : memref <i1> // ping -pong state} {db_ping_kernel}scf.if %is_pong_stage {... // pong -stage} {db_pong_kernel}... // deallocations etc.}
```

本文以 GELU 内核为例，完整阐释了 Hexagon-MLIR 的核心编译流程：

- 代码清单 3 为算子融合后的中间表示，融合操作在基于张量的 Linalg 级别执行；
- 清单 4 为分块后的内核，实现了将 DDR 中的数据切片复制到 TCM 中，该分块代码是张量与内存引用（memref）之间的桥梁，后续还会执行缓冲区化等其他编译 Pass；
- 清单 5 为添加 HVX 多线程支持后的中间表示；
- 本节介绍了双缓冲优化，对应清单 6 的中间表示。

经过上述优化后，编译器还会执行更多的下推编译操作，最终生成带有运行时调用的 LLVM-IR。

### 4.7 数学库

> 在 AI/ML 工作负载中，一种常见的优化手段是将指数（exp）等超越函数替换为低次多项式近似，以可控的数值误差为代价提升吞吐率。Hexagon-MLIR 调用高通 Hexagon 库（QHL）——一组为向量化计算预优化的数学库，实现超越函数的高效计算。

MLIR 本身也为数学操作提供了多项式近似展开，可实现无需依赖自定义库函数的快速近似计算。以指数函数 $e^x$ 的近似计算为例，其核心思路为：

- 首先将指数函数变形为： $e^{x}=e^{\epsilon +b}=e^{\epsilon +n\log 2}=e^{\epsilon }\cdot 2^{n}.$
- 随后将 $\epsilon$ 取值为较小的数，通过泰勒级数的前若干项计算 $e^{\epsilon}$，并常使用霍纳法则减少多项式计算中的乘法和加法次数。

::: block-1
除上述方法外，Hexagon-MLIR 还采用了其他优化手段和算法，实现数学操作的快速执行，快速平方根倒数（fast inverse square root）就是典型案例。
:::

### 4.8 其他优化与下推编译

> 本文未覆盖 Hexagon-MLIR 中已实现或开发中的所有编译 Pass和优化手段，也未讨论所使用的 MLIR 核心编译 Pass（如缓冲区化、释放、向 LLVM 的下推编译）。

上文介绍的编译流程，已能实现 PyTorch 和 Triton 内核的成功编译。此外，本文未提前介绍卷积和矩阵乘法的优化，这类操作通常会交由专用的矩阵乘法引擎执行；为让数据满足引擎的布局要求，编译器会通过 linalg.pack/linalg.unpack 实现数据空间转换；同时通过布局传播编译 Pass，减少数据打包/解包的次数。

## 五、实验结果与分析

> 本节的实验结果将与前文介绍的编译器优化手段一一对应，通过表格和柱状图展示性能提升效果。

实验中，我们在 Hexagon-MLIR 编译 Pass流水线中单独或组合启用了以下编译 Pass：

- 融合（F）——消除中间张量实例化，提升数据局部性；
- 分块（T）——将工作集移入 TCM；
- HVX 向量化（V）——利用单指令多数据（SIMD，Single Instruction Multiple Data）宽向量执行；
- 多线程（M）——将分块后的计算任务分配到多个硬件 HVX 上下文；
- 双缓冲（DB）——让 DMA 传输与计算重叠执行。

:mdnice-bold-italic[这种结构化的展示方式，能将技术章节介绍的每一种优化手段与实际的运行时性能提升关联起来，并凸显这些编译 Pass之间的相互作用——在内核从计算密集型转为内存密集型的过程中，这些相互作用有时是协同的，有时是竞争的。]

### 5.1 向量化加速比

> 表 1 总结了不同内核在 HVX 向量化后的性能加速比。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/09.webp)

::: image-caption
该表格实测了多款 AI 核心算子在 Hexagon NPU 的 HVX 向量化后的性能提升，覆盖激活、归一化、注意力等典型算子，还包含基础的二维向量加法。float16 数据类型因更适配 HVX 的向量宽度和数据打包规则，加速比远高于 float32；Flash Attention 加速比偏低，是因其包含复杂的归约操作和非连续内存访问逻辑，限制了向量化效率。
:::

- GELU 在 float16 数据类型（16384 个元素）下实现了 63.9 倍的加速，在 float32 数据类型下实现了 16.1 倍的加速，**这表明 HVX 向量宽度的利用率和数据类型的打包方式是吞吐率的核心影响因素。**
- 均方根归一化（RMS-norm）在 127×513 的张量形状下，向量化加速比达到 46.5 倍，这证明即使包含部分归约操作，以按元素操作为主的 linalg.elementwise 类计算，也能从向量化中显著获益。回顾一下，RMS-Norm 的计算公式为：

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_{i}^{2}+\epsilon}} \cdot g
$$

下面清单 7 展示了其 Triton RMS Norm 内核的计算部分。

```
sq = tl.sum(x * x, axis =0) / NUM_COLSrms = tl.sqrt(sq + EPSILON)y = (x / rms) * g
```

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/10.webp)

::: image-caption
该表格实测了多款 AI 核心算子在 Hexagon NPU 的 HVX 向量化后的性能提升，覆盖激活、归一化、注意力等典型算子，还包含基础的二维向量加法。float16 数据类型因更适配 HVX 的向量宽度和数据打包规则，加速比远高于 float32；Flash Attention 加速比偏低，是因其包含复杂的归约操作和非连续内存访问逻辑，限制了向量化效率。
:::

- 表 1 还包含了 Sigmoid 线性单元（SiLU）激活函数的实验结果，SiLU 的计算公式为：

$$
\text{SiLU}(x)=x \cdot \sigma(x)=x \cdot \frac{1}{1+e^{-x}},
$$

该算子在**向量化后实现了 5-8 倍的性能提升**，其计算过程主要受 $e^x$ 操作的影响；在 float16 数据类型下，若操作内部被提升为 float32，性能会出现下降。

### 5.2 单线程与多线程对比

> 图 2 对比了单线程与基于多个 HVX 上下文的 HVX 多线程，在不同问题规模（8K 到 1M 个元素）下的执行时间。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/11.webp)

::: image-caption
该图直观对比了不同数据规模下 GELU 内核单、多线程实现的实际执行耗时，小数据规模（≤16K）下多线程因线程启动、令牌管理和屏障同步的固定开销，耗时反而更高；当数据规模扩大后，并行计算的负载摊薄了固定开销，多线程执行耗时呈显著下降趋势，体现了多线程优化的规模效应。
:::

- 在小问题规模（≤16K）下，多线程的运行时间更长，这是由线程启动、令牌管理和屏障同步的开销导致的；
- 当问题规模超过 32K 后，多线程的性能持续优于单线程，加速比从 32K 时的 2.28 倍提升到 512K 时的 3.95 倍，在 1M 元素时为 3.40 倍（见加速比柱状图）。

当并行计算的负载足以抵消`async.execute`和`async.await_all`的固定开销时，多线程的性能优势就会显现。**在 1M 元素时，加速比比 512K 时略有下降，这是由 TCM 和 DMA 队列的压力增大导致的，这种情况下可通过重新调优分块参数提升性能。**

> 下面图 3 以更直观的形式展示了多线程 GELU 的性能加速比（加速比定义为单线程执行时间与多线程执行时间的比值）。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/12.webp)

::: image-caption
该图量化展示了 GELU 内核多线程相对单线程的加速比随数据规模的变化趋势，32K 规模是性能拐点，此后加速比稳步提升至 512K 时的 3.95 倍，1M 元素时加速比略有回落，这一变化反映了大尺度数据下 TCM 存储和 DMA 传输队列的资源压力，为后续分块参数的重调优提供了直接的实验依据。
:::

### 5.3 双缓冲优化

> 图 5 展示了向量化、多线程（MT）和双缓冲（DB）组合优化后的性能提升效果：向量化能带来显著的性能增益，若将向量化后的性能作为基准，多线程和双缓冲能在此基础上进一步提升性能。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/13.webp)

::: image-caption
该图以基础的二维向量加法为测试用例，展示了其在融合、向量化、多线程、双缓冲等优化编译遍依次作用下的性能变化曲线，直观体现了各优化手段的叠加增益效应，其中向量化带来基础且显著的性能提升，多线程与双缓冲则进一步挖掘了硬件的并行执行潜力，验证了编译遍流水线的有效性。
:::

> 图 7 展示了双缓冲及其他优化手段为 GELU 带来的性能增益，图 6 则展示了这些优化手段对计算密集型的指数级数内核的性能提升效果。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/14.webp)

::: image-caption
该图对 GELU 这一经典的 AI 激活算子完成了多编译遍优化的性能拆解，清晰展现了向量化、多线程、双缓冲优化的逐层增益效果，也直观体现了编译遍之间的相互作用 —— 计算类优化会放大内存开销，而双缓冲能针对性弥补。该图为 GELU 及同类激活算子在 Hexagon NPU 上的部署，提供了最优的优化组合参考。
:::

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/15.webp)

::: image-caption
该图针对计算密集型的指数级数内核，开展了多优化编译遍组合的性能拆解分析，向量化与多线程优化大幅降低了内核的计算耗时，却让内存传输开销成为新瓶颈，而双缓冲优化通过重叠计算与 DMA 传输，有效缓解了这一问题，各编译遍的协同作用显著提升了内核整体吞吐率，为计算密集型算子的优化提供了参考范式。
:::

### 5.4 多编译 Pass的相互作用

> 下文将分析多个编译 Pass之间的相互作用：

向量化和多线程主要减少计算时间，这会让内存传输的开销变得更加突出，而双缓冲通过让数据移动与计算重叠执行，能提升整体性能。**但这种优化的副作用是，经过优化的内核可能会从计算密集型转为内存密集型**。

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/16.webp)

::: image-caption
该图揭示了双缓冲优化的理论性能规律，在理想的计算 - 数据传输重叠状态下，当内存开销或计算开销单独占比 100% 时，双缓冲无法带来任何性能增益；而实际 AI 工作负载的开销分布介于两者之间，因此该优化能发挥作用，也为编译器选择性启用双缓冲编译遍提供了核心的理论参考。
:::

图 4 展示了一种理想情况：

- 当内存传输的开销占比 100%时，双缓冲无法带来任何性能增益；
- 同理，当计算开销占比 100%时，该技术也无效。

**在实际场景中，大多数工作负载的开销分布介于这两种极端情况之间**，因此双缓冲能带来性能增益，而开销、调度等次要因素也会对性能产生重要影响。:mdnice-bold-italic[选择性地启用优化编译 Pass，能更深入地分析内核的特性以及这些次要因素的相互作用。]

## 六、相关工作

| 作者与文献 | 主要贡献 | 关键技术/特点 |
| --- | --- | --- |
| Li 等人 [12] | 深度学习编译器综述，涵盖设计架构与优化策略 | 多级中间表示（IR）、前端/后端优化；算子融合、内存规划、布局转换、硬件映射、自动调优 |
| Yifan 等人 [28] | 提出张量编译器 Neptune，支持归约算子循环融合 | 滚动更新融合、Split-K 融合；重计算满足依赖；代数修正自动推导；基于 TVM 与 Triton 实现 |
| Yifan 等人 [30] | 《深度学习编译器中的图优化策略》 | 聚焦前端图优化技术，提升神经网络训练与推理效率 |
| Chin 等人 [14] | 《HloEnv：面向深度学习编译器优化研究的图重写环境》 | 量化评估编译 Pass影响：影响数据集比例 + 平均性能变化（启用/禁用运行时比值） |

同时优化计算速度、内存占用和数据传输需求，是 AI 编译工具的核心目标。文献[3]讨论了一款 XLA 编译器扩展，该扩展能根据用户指定的内存限制，调整算法的计算数据流表示。Benoit 等人[22]提出了 MODeL，这是一种基于整数线性规划（ILP，Integer Linear Programming）的方法，通过优化神经网络训练过程中张量的生命周期和内存位置，减少模型的内存占用。

## 七、结论

> 本文提出了 Hexagon-MLIR，这是一款基于 MLIR 的编译栈，专为高通 Hexagon NPU 设计，用于加速 AI 工作负载。Hexagon-MLIR **结合了 MLIR 的可扩展基础设施和面向硬件的编译 Pass**，为 PyTorch 模型和 Triton 内核提供了系统化的下推编译流水线。

算子融合、分块、HVX 向量化、多线程执行和双缓冲等核心优化手段，能充分利用紧密耦合内存（TCM）、异步 DMA 传输和专用数学库的性能优势。

实验结果表明，代表性内核在该编译栈上均实现了显著的性能提升，验证了本方案的有效性。除性能提升外，本文:mdnice-bold-italic[还阐释了 MLIR 的设计原则——结构化中间表示、声明式转换和分阶段下推编译——为构建面向 AI 工作负载的编译器奠定了坚实的基础。]

尽管当前的实现已取得了良好的效果，我们仍在持续开展相关工作，包括扩展对新兴模型的支持、优化调度启发式算法、提升功能支持度和性能。最终，Hexagon-MLIR 旨在为下一代 AI 工作负载在 Hexagon NPU 上的部署，提供一套具备可扩展性的高性能解决方案。

## 参考文献

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/17.webp)

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/18.webp)

![文章插图](/articles/2026-03-08-hexagon-mlir-mlir-pytorch-triton-39979d1a/images/19.webp)
