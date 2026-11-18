关键词：**torch.compile**、归一化算子、:mdnice-bold-italic[LayerNorm]、**RMSNorm**、GPU 性能优化

> LayerNorm 与 RMSNorm 作为深度学习模型的基础归一化算子，其用于对输入数值进行归一化处理，是保障大模型训练平稳收敛、提升推理效率的关键模块。

尤其在高算力 GPU 平台上，**其内核性能直接影响整体训练吞吐。**

::: block-1
当前业内顶尖归一化内核多依赖手工深度优化，而 **PyTorch 原生 torch.compile 在该类算子上长期存在性能差距，内存带宽利用率低、归约逻辑与分块策略不合理等问题突出**，难以充分发挥 H100、B200 等新一代显卡的硬件潜力。
:::

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/01.webp)

- SOTA Normalization Performance with torch.compile
- https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/
- 6000 字，阅读 15 分钟，**播客 8 分钟**

相关推荐

- [TorchAO：PyTorch原生端到到框架，打通训练到移动端全链路！FP8/PTQ缩容4倍/延迟降21%！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891983&idx=1&sn=f03133d7a88c10a28d6e83c1ec26acb1&scene=21#wechat_redirect)
- [HotChips 2025：从 PyTorch 到 CUDA：大模型时代 GPU 内核自动生成的技术跃迁，Tri Dao最新演讲](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891347&idx=1&sn=173b8638e00da3610dd90ba7d5a4b7f6&scene=21#wechat_redirect)
- [SOLE：Softmax/LayerNorm软硬件协同优化，36倍加速与2.8+倍面效和能效比的低精度Transformer架构](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893805&idx=1&sn=b82015bdc4d1d0bee906f5471e88de62&scene=21#wechat_redirect)

> 本文以 Quack 手工优化内核为性能基准，针对 torch.compile 的 Inductor 后端展开针对性改进，:mdnice-bold-italic[优化内部归约分块、线程束配置等核心参数，提出混合阶数归约方案并结合拆分归约、软件流水线与自动调优策略]，大幅提升前向与反向传播内核效率，**最终让编译生成的归一化算子达到行业顶尖水平，同时保留自动融合周边算子的优势**，为大模型编译优化提供了实用方案。

我们在英伟达 H100 与 B200 显卡上对 LayerNorm/RMSNorm 的`torch.compile`性能进行了评估与优化，**使其在逐内核层面达到接近行业顶尖（SOTA）的性能，同时还能通过自动融合能力实现进一步的速度提升。**

## 本文目录

- 一、前向传播
  - 1.1 层归一化（LayerNorm）
  - 1.2 均方根归一化（RMSNorm）
  - 1.3 Quack
  - 1.4 torch.compile
  - 1.5 基准测试结果
- 二、反向传播
  - 2.1 融合归约（Fused Reductions）
  - 2.2 Inductor 生成的融合归一化反向传播内核
  - 2.3 拆分尺寸（Split-Size）自动调优
  - 2.4 软件流水线（Software Pipelining）
  - 2.5 基准测试结果
- 结论

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、前向传播

### 1.1 层归一化（LayerNorm）

> 层归一化最早在这篇论文中被提出：Layer Normalization, Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton[1]。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/03.webp)

它通过计算均值与方差对输入进行归一化，同时搭配可学习参数 $\gamma$（权重）与 $\beta$（偏置）进行缩放与平移。

### 1.2 均方根归一化（RMSNorm）

> 均方根归一化（RMSNorm）作为层归一化的改进方案，在这篇论文中被提出：Root Mean Square Layer Normalization, Biao Zhang, Rico Sennrich[2]。

该方法不再以均值为中心进行中心化处理，而是**采用输入 $x$ 的平方和的均方根（RMS）完成归一化**；依旧使用可学习参数 $\gamma$（权重）进行缩放，但不再包含偏置项。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/04.webp)

LayerNorm 与 RMSNorm 两者的前向传播逻辑较为相似，通常会在连续维度上执行归约操作，并搭配若干额外的逐元素运算；**RMSNorm 的计算量通常更低，因为其浮点运算量更少且无偏置项，效率会略高一些。**

::: block-1
出于本次研究的目的，鉴于两类内核的相似性，我们会交替展示 LayerNorm 与 RMSNorm 的基准测试结果。
:::

### 1.3 Quack

> Quack 是由 Tri Dao 团队开发的、基于 CuteDSL（一种 GPU 内核领域专用语言）实现的极致优化内核库：https://github.com/Dao-AILab/quack[3]。

该项目的 README 文档展示了在 H100 显卡上，Quack 针对此类归约内核的性能优于`torch.compile`。我们将 Quack 作为行业顶尖（SOTA）基准，以此评估`torch.compile`的性能水平。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/05.webp)

::: image-caption
bf16精度下，RMSNorm、Softmax、Cross-Entropy三大大模型核心算子的性能对比，纵轴为内存带宽（数值越高性能越强），横轴为不同(batch, 归约维度)输入尺寸。QuACK（蓝线）在三类算子、全尺寸输入下全面领先，大幅优于torch.compile、Liger Kernel、cuDNN，大尺寸输入优势更显著。
:::

Quack 的 README 中展示了`torch.compile`此前的性能表现，从中可以看出**，`torch.compile`的性能通常仅能达到 Quack 的 50%左右。**

### 1.4 torch.compile

> 下方我们会阐述`torch.compile`为 LayerNorm 前向传播生成内核的通用逻辑，这和 RMSNorm 的实现思路也完全一致。

我们假设输入的归约维度元素数量（`rnumel`）是连续的，在 Inductor（PyTorch 编译器后端）中我们将其称为**内部归约（Inner reduction）**。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/06.webp)

::: image-caption
RMSNorm/LayerNorm前向的分块并行计算流程。输入X按XBLOCK（行块）、RBLOCK（列块）分块，先对R_BLOCK内元素累加，计算各XBLOCK的均值；再将均值传入归一化单元，完成元素级除/乘/加操作并输出；若需梯度，缓存均值、rsqrt(var)供反向使用。该方案适配GPU并行架构，优化访存，提升大模型归一化层计算效率。
:::

尽管内核逻辑看起来略显复杂，但其核心原理非常简单：

1. 为输入 $X$ 的每一行维护大小为`R_BLOCK`的部分和；
2. 利用部分和计算均值与方差；
3. 依据层归一化公式对 $X$ 执行逐元素运算；
4. 存储逐元素运算的输出结果：

::: block-1
补充说明：若归约维度大小 $R$ 小于某一经验阈值（1024），Inductor 会生成**持久化归约（persistent reduction）** 内核——此时无需循环遍历归约维度 $r$，可直接计算均值。
:::

在 H100 与 B200 显卡上对比`torch.compile`与 Quack 实现的 RMSNorm 前向传播时，**我们复现了`torch.compile`性能远逊于 Quack 的问题**。但在完成自动调优并以此优化 Inductor 默认参数后，`torch.compile`在 H100 与 B200 上达到了行业顶尖（SOTA）性能。

> 总体而言，实现该效果的核心优化手段如下：

- **基准测试中插入`torch._dynamo.reset()`：** 确保`torch.compile`不启用自动动态形状推导；此前针对每个形状单独调用一次`torch.compile`，会导致编译器判定使用动态形状。
- **修复不合理的自动调优配置决策**：H100 与 B200 上的默认自动调优配置会做出次优选择，进而导致性能不佳；通过设置`mode='max-autotune'`可缓解该问题。我们对默认启发式规则做了多项改进：

### 1.5 基准测试结果

> 下方我们展示`torch.compile 2.11`与 Quack（2026 年 3 月 24 日主干版本）在 Quack 基准测试形状，以及实际业务中常见的大 M、小 N 形状下的性能对比数据。**结果表明，`torch.compile`的性能基本与 Quack 持平。**

测试中存在两类性能回退场景：

1. 在 N=384 时出现小幅性能回退，:mdnice-bold-italic[原因是 Triton 无法简洁地表示非 2 的整数次幂的分块大小；]
2. H100 在 N 尺寸极大时出现大幅性能回退，:mdnice-bold-italic[原因是 Triton 暂不支持分布式共享内存的表达。]

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/07.webp)

::: image-caption
B200 GPU在bf16精度、CUDA 13、750W功耗下的RMSNorm 性能测试，纵轴为吞吐量（GB/s），对比Quack与Static torch.compile+启发式优化方案，横轴为不同输入尺寸。多数场景Quack性能持平或反超，大尺寸输入优势显著，说明Quack对B200硬件适配更优，大模型场景收益突出。
:::

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/08.webp)

::: image-caption
H100 GPU在bf16精度、CUDA 12.9、650W功耗下的RMSNorm算子性能测试，纵轴为吞吐量（GB/s），对比Quack与Static torch.compile+启发式优化方案，横轴为不同输入尺寸。常规尺寸两者性能接近，特殊/大尺寸输入Quack稳定领先，说明Quack对H100硬件适配更优，大模型场景表现更稳。
:::

## 二、反向传播

> LayerNorm/RMSNorm 的反向传播过程比前向传播更复杂。至少需要计算两类梯度：输入的梯度 $\text{d}X$、权重的梯度 $\text{d}W$；LayerNorm 中还需额外计算偏置梯度 $\text{d}B$。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/09.webp)

::: image-caption
反向传播的梯度归约布局示意图，用于高效计算三类梯度：水平蓝色箭头对应输入梯度dx的归约，沿x轴聚合；垂直红色箭头对应权重梯度dw、偏置梯度dB的归约，沿y轴聚合。图标注了维度步长：rnumel维度步长为1，numel维度步长为rnumel，通过优化内存排布适配GPU并行架构，提升访存效率，加速梯度计算。
:::

为简化实现并规避复杂的数学公式，从性能角度考虑，这些梯度计算需要对反向传播的输入梯度 $\text{d}Y$（即前向传播上一步输出的梯度）的两个维度分别执行归约操作。

最朴素的实现方式（也是归约维度极大时无法避免的方案）是在独立内核中分别完成归约：一个内核计算 $\text{d}X$，另一个内核计算 $\text{d}W$、$\text{d}B$。但该方案会导致同一输入 $\text{d}Y$ 被两次读取，读取数据量翻倍。鉴于归一化内核属于内存受限型任务，这会带来显著的额外延迟。

### 2.1 融合归约（Fused Reductions）

> 对于元素数通常不大、且单行数据可充分放入线程块的合理形状（一般 ≤16384），可以采用性能更优的融合内核，同时不会造成共享内存/寄存器占用爆炸。

简单来说，该内核会照常执行 $\text{d}W$、$\text{d}B$ 的归约计算，同时为每一行并行完成 $\text{d}X$ 的列维度归约。已有多篇文献对这类融合方式进行了研究，例如Liger[4]、Meta 提出的融合半持久化归一化反向传播内核[5]、Quack 基于 CuteDSL 实现的融合内核[6]。

在 Inductor 中，我们用不同类型标识归约操作，具体包括：

- **内部归约（INNER reduction）**：对步长为 1 的连续维度执行归约；
- **外部归约（OUTER reduction）**：对其余维度执行归约。

::: block-1
基于上述定义，融合内核会对同一输入张量同时执行内部归约与外部归约：其中内部归约对应 $\text{d}X$（连续维度），外部归约对应 $\text{d}W$、$\text{d}B$。
:::

#### 拆分归约（Split Reduction）

> 在实际业务的大量形状中，批维度`xnumel`远大于归约维度`rnumel`。此时更优的方案是：**对 $X$ 维度的归约计算部分和，最终通过`torch.sum`汇总所有部分和，以此提升并行度。**

Triton 官方教程中的层归一化示例就采用了拆分归约，但该实现通过原子操作加锁，让单个线程块负责处理单行数据；在批维度 $X$ 较大时，该方式性能极差，且**会引发数值计算不一致问题**。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/10.webp)

::: image-caption
反向传播中权重梯度（DW）的两阶段并行计算方案。Stage1：将输入X分块，并行计算局部权重梯度DW_partial，通过Lock规避多线程写冲突；Stage2：对各DW_partial做归约累加，得到最终权重梯度Final_DW。该方案通过分阶段并行，大幅减少原子操作开销，显著提升GPU上大模型卷积层梯度计算的并行效率。
:::

Inductor 当前也具备类似的拆分归约能力：与上述方案一致，会分配一块工作区张量（workspace tensor）存储部分和，但不使用原子操作；而是保证单个线程块协处理器（CTA）处理多行数据，并向工作区张量的唯一位置写入结果。

### 2.2 Inductor 生成的融合归一化反向传播内核

> 结合上文提到的融合归约与拆分归约方案，我们让 TorchInductor 能够自动生成达到行业顶尖水平的融合归一化反向传播内核。此外，编译器自动生成这类内核的方式，还支持更灵活的自动调优，以及**与周边算子的自动融合能力**。

由于该优化的核心难点是对同一输入执行不同归约顺序的归约融合，我们将该优化命名为**混合阶数归约（MixOrderReduction）**。

针对形状为 $[M, N]$ 的输入，生成的内核执行流程如下：

1. 按照`SPLIT_SIZE`分块，对 $M$ 维度执行拆分归约；
2. 对每个分块，工作区张量中的一行存储外部归约的部分归约结果（例如每一列的部分和，即 $\text{d}W$、$\text{d}B$）；
3. 对每个分块，通过循环加载分块内的每一行：
4. 额外执行一次归约，将工作区张量中的部分归约结果汇总，得到外部归约的最终结果。该额外内核的输入张量尺寸极小，因此单独执行不会带来明显的性能损耗。

> 在 Inductor 代码生成逻辑中，识别到混合阶数归约模式后，会执行以下步骤：

1. 针对外部归约内核，将归约与`store_reduction`节点替换为全新的`partial_accumulate`节点。该节点会记录待归约数值、归约类型等信息。该变换会将外部归约内核转换为逐元素内核（PW1）；
2. 依托已有的循环重排序能力，对转换后的逐元素内核（PW1）执行循环重排序，得到新的逐元素内核（PW2）；
3. 此时 PW2 与内部归约内核的循环顺序一致，可完成两者的融合。

### 2.3 拆分尺寸（Split-Size）自动调优

> 拆分尺寸`SPLIT_SIZE`对混合阶数归约内核的性能至关重要。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/11.webp)

::: image-caption
NVIDIA H100上RMSNorm反向传播的性能测试，纵轴为内存吞吐量（TB/s，越高性能越好），横轴是不同优化方案，不同色线对应不同输入尺寸（batch×隐层维度）。eager/inductor性能偏弱，fused系列优化大幅提效，中等分段（ss_16~ss_64）达性能峰值，大尺寸输入吞吐量更高，分段过大会因开销导致性能回落，需匹配输入选最优参数。
:::

在 H100 显卡上，针对形状为 $(1152000, 384)$、数据类型为 bfloat16 的输入，Liger 的 RMSNorm 反向传播内核默认性能为 0.417 TB/s；将`SPLIT_SIZE`缩小 32 倍后，性能可提升至 1.912 TB/s。

> 我们在 H100 显卡上，针对 bfloat16 数据类型，测试了不同基准形状与拆分尺寸下的性能结果。从上述结果可以得出结论：

- **拆分尺寸选择不当，会导致超过 2 倍的性能下降；**
- :mdnice-bold-italic[性能曲线近似呈抛物线形态。采用「持续将拆分尺寸放大 2 倍或缩小 1/2，直至找到性能峰值」的自动调优策略，可高效解决该问题。]

Inductor 原有的拆分归约特性会为了提升性能拆分外部归约，但基于原有启发式规则选择的拆分尺寸（图表中标记为`fused_split_reduction`列）往往效果不佳。我们让混合阶数归约忽略拆分归约选定的尺寸，转而使用专属的启发式规则或自动调优机制，选择更优的拆分尺寸。

### 2.4 软件流水线（Software Pipelining）

> 在为反向传播内核冲击峰值带宽的过程中，我们还发现了软件流水线（也称作加载预取）的优化价值。

以往只有矩阵乘法（GEMM）、注意力机制这类计算密集型任务会使用流水线优化，内存受限型任务通常无需该操作；Inductor 针对逐元素/归约内核、以及 Liger 示例均未实现`num_stages`（流水线阶段数）自动调优。但我们观察到 Quack 内核中实现了类似预取的逻辑。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/12.webp)

::: image-caption
NVIDIA B200 GPU上RMSNorm反向传播的性能测试，纵轴为内存吞吐量（GB/s，越高性能越好），横轴为不同输入尺寸，对比4种优化方案。基础PyTorch 2（蓝）性能垫底，开启MixOrder（红、黄）大幅提效，带num_stages的MixOrder（黄）更优；Quack（绿）整体表现顶尖，多数场景持平或领先，适配大模型多尺寸输入。
:::

我们将`num_stages`作为通用自动调优参数加入 Inductor 内核，**在部分形状下实现了显著加速，尤其在大 M、小 N 的场景中，应用于混合阶数归约后提速最高可达 20%。**

### 2.5 基准测试结果

> 下方我们展示混合阶数归约（MixOrderReduction）对比 PyTorch 即时执行（eager）、旧版编译版本，以及 Quack、liger 等开源基准的测试结果。所有测试均在 2025 年末、搭载 CUDA 12.9 的 750W B200 机器上完成。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/13.webp)

::: image-caption
NVIDIA B200 GPU上RMSNorm反向传播的性能测试，纵轴为内存带宽（GB/s，数值越高性能越强），横轴为大模型常见输入尺寸。原生eager模式性能垫底；PyTorch编译优化（compiled）大幅领先，峰值超6000GB/s；Quack、Liger等专用优化次之，显著优于无融合编译方案，大尺寸输入下编译方案优势更突出。
:::

从结果可以观察到：

- 搭载混合阶数归约的`torch.compile`比即时执行模式快 17.07 倍，而未搭载该优化的`torch.compile`仅比即时执行快 9.93 倍；
- 搭载混合阶数归约的`torch.compile`比 Liger 快 1.45 倍，比 Quack 快 1.34 倍。

> 我们同时展示了 LayerNorm 的基准测试结果，鉴于内核逻辑相似，其表现与 RMSNorm 基本一致。

![文章插图](/articles/2026-04-09-torch-compile-normalization-80b2b58e/images/14.webp)

::: image-caption
NVIDIA B200 GPU上LayerNorm反向传播的性能对比，纵轴为内存带宽（GB/s，越高性能越强），横轴为大模型常见输入尺寸。原生eager模式性能垫底；PyTorch融合编译（compiled，橙线）大幅领先，峰值超6000GB/s；无融合编译（黑）次之，Liger（红）在大尺寸输入下性能明显回落，融合编译是该算子的最优优化方案。
:::

**结果呈现出与 RMSNorm 相同的趋势：**

- 未搭载混合阶数归约的`torch.compile`相比 PyTorch 即时执行已有显著提速；
- 而搭载全新混合阶数归约方案的`torch.compile`，相比旧版`torch.compile`实现了近 2 倍提速，性能更接近峰值内存带宽。

## 结论

> 本文围绕 PyTorch 的`torch.compile`在 LayerNorm 与 RMSNorm 归一化算子上的性能优化展开深入研究，最终实现了突破性成果：通过一系列针对性改进，让编译生成的归一化内核在 H100、B200 等新一代 GPU 上达到行业顶尖水平，全面对标甚至超越 Quack 手工优化内核与 Liger 等开源基准。

从具体表现来看，反向传播环节搭载混合阶数归约方案后：相较原有`torch.compile`实现近 2 倍提速，比 Liger 快 1.45 倍、比 Quack 快 1.34 倍，前向传播也通过参数调优与自动调优策略，**解决了非 2 的整数次幂分块等场景的性能回退问题，内存带宽利用率大幅提升。**

这些成果的达成，源于对`torch.compile` Inductor 后端核心痛点的精准突破：

- 前向传播中通过优化内部归约的 RBLOCK 大小、调整持久化归约的 XBLOCK 配置、动态适配线程束数量，解决了原有编译内核硬件适配性差的问题；
- 反向传播创新性提出混合阶数归约，融合内外归约逻辑并搭配拆分归约与软件流水线技术，规避了数据重复读取的效率损耗，同时通过拆分尺寸自动调优，避免了不当参数导致的 2 倍以上性能下降。

::: block-1
更重要的是，:mdnice-bold-italic[本文优化方案保留了`torch.compile`的核心优势——自动融合周边逐元素与归约算子，这是手工编写内核难以实现的灵活性，能进一步提升大模型训练与推理的端到端效率。]
:::

该优化不仅为归一化算子提供了高性能编译方案，更验证了通过编译器后端优化、结合硬件特性的自动调优策略，能够充分发挥新一代 GPU 的硬件潜力，为深度学习框架中基础算子的编译优化提供了可复用的思路，对推动大模型高效部署具有重要的实践价值。

参考资料

[1] Layer Normalization, Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton: *https://arxiv.org/abs/1607.06450*

[2] Root Mean Square Layer Normalization, Biao Zhang, Rico Sennrich: *https://arxiv.org/abs/1910.07467*

[3] https://github.com/Dao-AILab/quack: *"https://github.com/Dao-AILab/quack"*

[4] Liger-Kernel/src/liger_kernel/ops/rms_norm.py: *https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py#L459*

[5] Meta 提出的融合半持久化归一化反向传播内核: *https://fb.workplace.com/groups/257735836456307/posts/999376505625566*

[6] Quack 基于 CuteDSL 实现的融合内核: *https://github.com/Dao-AILab/quack/blob/main/quack/rmsnorm.py*
