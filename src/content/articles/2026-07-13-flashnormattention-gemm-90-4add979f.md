**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：内核融合、**归一化优化**、Multi-CTA Norm、**Lazy Pre-Norm**、:mdnice-bold-italic[FlashNormAttention]

> 本文针对 LLM 与广告推荐模型中广泛使用的 **LayerNorm、RMSNorm 性能痛点提出全套内核融合优化方案**。归一化（Normalization）算子是典型内存受限操作，无法调用 TensorCore，在推荐模型训练中占 20%总时延、大模型占 10%，传统算子融合因 GEMM 与归一化分块逻辑冲突，仅小维度场景有效。

作者推出三大核心创新：

- Lazy Pre-Norm 借助数学恒等变换破解预归一化循环依赖，将归一化计算后置至矩阵乘收尾阶段；
- Multi-CTA Norm 依靠多 CTA 集群与分布式共享内存协同归约，实现通用后归一化融合，最高可掩盖 90%归一化时延；
- FlashNormAttention 一体化融合前后归一化与 GDPA 注意力，构建低 IO MegaKernel。

::: block-1
作者设计前向、反向差异化融合策略，规避反向计算冗余，基于 NVIDIA B200 GPU 实测，:mdnice-bold-italic[矩阵场景归一化延迟掩盖率最高 91.61%，注意力模块正向总时延最高降低 35.39%、反向近 20%，大幅削减 HBM 读写开销，整套优化代码开源，为生成式广告模型、大语言模型提供通用高性能内核优化路径。]
:::

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/01.jpg)

- **Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels**
- https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#
- multi_cta_norm_fusion 代码地址：https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion
- gdpa_megakernel 代码地址：https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel

::: related-title
**相关推荐**
:::

::: related-links
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**
- **[Helion DSL 赋能 vLLM 实现跨硬件高效推理：Qwen3 模型 FP8 推理加速实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902433&idx=1&sn=5e65e917235f732c27c1cd121a2d6d24&scene=21#wechat_redirect)**
- **[借助 torch.compile 实现 Normalization 顶尖性能表现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899700&idx=1&sn=dad00ee6111ffb73275bae4c98c8107f&scene=21#wechat_redirect)**

:::

> 本研究主要基于两种核领域特定语言（DSL）实现：

::: related-links
- TLX[1]：Triton DSL 的扩展集，提供底层、硬件感知的 GPU 执行控制支持
- Helion[2]：高层 DSL，在开发效率、可移植性和全面自动调优方面表现出色

:::

基准测试采用 bfloat16 数据类型，在 Meta 数据中心的 NVIDIA B200 GPU 上进行，功率限制为 750W。

::: block-1
我们感谢 Tri Dao、Markus Hoehnerbach、Jay Shah、Ted Zadouri、Vijay Thakkar、Wentao Guo 在Flash Attention[3] 和Quack[4] 方面的开源工作——这些工作为本文讨论的许多优化技术奠定了基础并提供了灵感。我们感谢 PyTorch 和 Triton 团队对Helion[5] 和TLX[6] 的开发和维护——它们使本文提出的思路能够高效、顺利地探索和实现。
:::

## 本文目录

- 引言
- 一、归一化融合的挑战
- 二、延迟预归一化：一种将预归一化与线性层融合的新颖技术
- 三、多 CTA 归一化：将后归一化与线性层作为后缀融合
  - 反向传播怎么办？融合重组
- 四、FlashNormAttention：将预归一化和后归一化同时融合到 FlashAttention 变体核中
  - 反向传播实现
- 结语

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/02.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 引言

> 归一化技术已成为大多数深度学习架构中不可或缺的组成部分，因其在稳定训练和加速收敛方面的卓越效果而被广泛应用。**特别是，沿着最内层嵌入维度的传统归一化（如 LayerNorm、RMSNorm），已成为现代大语言模型以及 Meta 广告模型等推荐系统模型中最常见、最普遍的类型。**

例如，在部署于 Meta 最大推荐系统训练基础模型——生成式广告模型（GEM）[7] ——的昆仑（Kunlun）[8]架构中，:mdnice-bold-italic[LayerNorm/RMSNorm 几乎存在于所有关键组件中，包括多头注意力（Multi-Head Attention）、分层种子池化（Hierarchical Seed Pooling）和 GDPA 增强型位置前馈网络（PFFN）。]

::: block-1
然而，**归一化的普遍性也带来了严峻的性能挑战：它是高度内存绑定的操作，且无法利用 TensorCore**。这使得我们无法在模型训练中充分发挥硬件的计算能力。:mdnice-highlight[以昆仑（Kunlun）为例，归一化操作约占总训练延迟的 20%。这意味着如果不进行优化，我们将直接损失 20%的硬件计算吞吐量。在典型的计算绑定型 LLM 中，归一化仍可能占据约 10%的总延迟。]
:::

> 为解决这一问题，我们必须以 IO 感知的方式设计归一化核，通过核融合在不影响计算精度的前提下精心节省内存 IO 成本，并将这些内存/CUDACore 密集型操作与 TensorCore 密集型操作重叠执行。:mdnice-bold-italic[由于大多数归一化操作紧跟或先于矩阵乘法操作（如 MLP、注意力机制），本研究重点关注如何高效地将归一化与矩阵乘法融合。]

我们首先介绍多种与单个 GEMM 高效融合的策略，最后提出 FlashNormAttention 算法，将 LayerNorm 和 RMSNorm 同时融合到注意力机制中。

*注：除非另有说明，以下基准测试结果均禁用了逐元素仿射变换[9]（elementwise affines），因为我们发现这些操作在我们的模型中会带来显著的性能开销，而对模型质量的影响微乎其微。此外，除非另有说明，所提出的核心优化思路和算法思想均适用于是否包含逐元素仿射变换的场景，尽管性能结果可能有所不同。*

## 一、归一化融合的挑战

> 本节讨论以常规方式将归一化与 GEMM 等计算密集型核融合时面临的挑战——:mdnice-bold-italic[这些挑战本质上源于不同的分块（tiling）策略。我们随后提出一种“朴素”融合方案：强制 GEMM 算法遵循与归一化算法相同的分块方式。实验表明，这种方案在超小 N 值场景下表现良好，但随着 N 增大，由于分块约束和低效性，会变得次优甚至不可行。]

与标准激活函数融合（如 GEMM+ReLU）相比，**归一化融合的核心挑战在于分块方式的差异**。归一化本质上是一种归约（reduction）操作，需要访问整个维度的数据才能计算出正确结果。:mdnice-bold-italic[具体而言，对于 LayerNorm 和 RMSNorm，典型的核会沿着外层维度分块，但不会对内层维度分块——这意味着每个 CTA（线程块）始终需要加载完整的行数据]。相比之下，典型的 GEMM 会在两个维度上都进行分块，因此每个分块不会跨越完整的行，这使得后续的行方向归一化无法实现。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/03.webp)

::: image-caption
对比两种GPU分块（tiling）布局：左侧是matmul tiling矩阵乘法分块，采用零散、分散排布的小块tile，排布不规整；右侧是norm tiling归一化分块，tile以整行长条状连续横向排布，结构规整。二者是针对矩阵乘法和归一化两类不同计算算子的显存/计算分块策略，适配各自的计算访问模式。
:::

最直接的解决方法是扩大 GEMM 的分块大小，使每个分块跨越完整的内层维度。对于典型的（MxK）@（KxN）GEMM，这意味着 N 维度上的分块大小必须大于 N（通常是 N 的下一个 2 的幂）。高层算法如下图所示：

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/04.webp)

::: image-caption
图中展示矩阵运算过程：一维行矩阵A元素为1、2、3、4、5，与列矩阵B（元素同样是1、2、3、4、5）做内积运算，得到矩阵C，该行结果为1+2+3+4+5的求和结果，之后对C进行按行求范数（row-wise norm）处理得到norm(C)，标注k loop代表沿k维度循环做求和运算，直观演示行向量与列向量相乘及行范数计算。
:::

> 不过，这种方法存在两个主要问题：

1. **它偏离了纯 GEMM 的最优分块策略**，由于缓存行为、流水线行为等方面的次优性，会降低 GEMM 本身的性能；
2. **它对输入形状施加了严格限制** ——特别是 N 的最大取值。过大的 N 将无法放入共享内存。

通过简单计算可以大致了解 N 的最大可能值：假设 Blackwell GPU 的共享内存大小为 228KB，数据类型为 bfloat16（2 字节/元素），为实现高效流水线/重叠执行至少需要 2 个流水线阶段，且 M 维度和 K 维度的最小分块大小均为 32。则：

```
2 stages x 2 bytes / element x (tile_m x tile_k + tile_k x tile_n + tile_m x tile_n) < 228KB=> 32 x 32 + 32 x tile_n + 32 x tile_n < 228KB / 4=> 512 < tile_n < 1024
```

由于分块大小通常为 2 的幂，这意味着 tile_n（进而 N）的最大值限制为 512，否则核无法运行。

::: block-1
我们将在后续章节讨论解决这些限制的方法。**但尽管存在这些问题，我们发现此类融合策略在小 N 值场景下仍能带来显著收益**。本实验采用Helion[10]实现，因其具有高开发效率和全面的自动调优能力，在这种分块大小受严格约束的非规范场景中表现出色。
:::

> 以下是广告模型中典型输入形状的基准测试结果。注意：延迟节省百分比是相对于 PyTorch Inductor 的归一化核延迟计算的——这种指标设计使结果不受基础 GEMM 核延迟（及其与归一化核的延迟对比）的影响，并本质上反映了此类融合尝试的提升空间（即 100%是理论最优值，意味着归一化完全与 GEMM 重叠执行）。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/05.webp)

::: image-caption
这张柱状图展示full-N epilogue融合后LayerNorm算子延迟被掩盖的占比，横轴为不同(K,N)参数组合（M固定为256×1024），纵轴为占比百分比。前四组参数为正值：(64,64)25.63%、(64,128)28.11%、(128,64)17.24%、(128,128)32.20%；后三组出现负值：(128,256)-2.46%、(256,128)-2.90%、(256,256)-64.38%，可见参数变化对融合效果影响极大。
:::

- 对于 64、128 等小形状，该融合策略可为 LayerNorm 核带来 17%-32%的显著延迟节省。
- 但当 K/N 超过 128 后，性能提升开始消失，甚至出现大幅退化。**这是因为随着 N 增大，强制 tile_n=N 会与纯 GEMM 的最优分块大小偏差越来越大，节省内存 IO 的收益逐渐被破坏基础 GEMM 算法带来的负面影响所抵消。**

## 二、延迟预归一化：一种将预归一化与线性层融合的新颖技术

> **本节介绍一种新颖的前缀融合（prologue fusion）技术，用于将预 RMSNorm 融合到 GEMM 核中**。我们讨论此类前缀融合的动机和挑战，并提出一种名为**延迟预归一化（Lazy Pre-Norm）** 的创新算法——:mdnice-bold-italic[该算法通过数学技巧将部分预归一化计算延迟到 GEMM 完成后执行，从而解决这些挑战并实现良好的性能提升。]

我们提出的第一个思路是通过前缀融合将预归一化与后续的 GEMM 融合。尽管通常应避免前缀融合，但仍有以下几点值得探索：

1. **前缀融合避开了后缀融合（epilogue fusion）中遇到的分块问题**：GEMM 核中的每个 CTA 无法访问输出张量的完整行，但每个 CTA 在算法执行过程中会自然扫描输入张量 A 的完整行；
2. 实际上，**预归一化在现代模型（尤其是大语言模型）中已比后归一化更为普遍**。

为实现高效的前缀融合，我们针对无逐元素仿射变换的 RMSNorm 这一特殊场景，设计了**延迟预归一化（Lazy Pre-Norm）** 优化技术。具体来说，我们希望融合以下计算：

```
C = rmsnorm(A) @ Bwhere rmsnorm(A) = A * rstd(A)[:, None]and rstd(A) = rsqrt((A ** 2).sum(dim=-1) / A.shape[-1] + 1e-5)
```

延迟预归一化旨在解决的核心难点：在典型的分块 GEMM 中，尽管我们最终会访问完整的行（这是计算归约结果 rstd 的前提），但实际访问是按分块进行的——而我们需要 rstd 才能处理每个分块！这形成了循环依赖：必须等待 k 循环结束才能计算 rstd，但开始 k 循环又需要 rstd。

> 为解决这一问题，第一个关键观察是：两个相互依赖的部分本质上是不同类型的计算：归约（reduction）和逐元素应用（elementwise application）。

1. rstd 计算部分是对完整行的**归约** 操作；
2. 使用 rstd 进行归一化是对 A 中每个元素的**逐元素** 计算。

我们将这两个组件分开处理：对于归约部分，首先需要注意它本身不会阻塞任何操作——这是一个很好的特性，意味着我们可以将其与 TensorCore 计算并行执行。由于每个 CTA 会自然地沿着 A 的内层维度扫描，我们可以在执行矩阵乘法的同时，并行累积 A 的平方和。

逐元素部分的问题更为突出，因为它依赖于归约结果，导致循环依赖。解决方案来自一个数学技巧，核心观察是：无仿射变换的 RMSNorm 中的逐元素乘法本质上是**行方向的乘法** ——**即 A 中同一行的所有元素都乘以相同的 rstd。这意味着以下关键性质成立：**

```
(A * rstd[:, None]) @ B = (A @ B) * rstd[:, None]
```

> **证明**： 行方向乘法等价于用对角矩阵 M 与 A 相乘（M 的对角线元素为 rstd）。因此：`(A * rstd) @ B = (M @ A) @ B = M @ (A @ B) = (A @ B) * rstd`

这一性质非常重要，因为它允许我们将逐元素计算“延迟”到整个 k 循环结束后执行，实际上成为了后缀操作！综合以上思路，延迟预归一化算法的核伪代码如下：

```
def GEMM_norm_fusion_kernel(A, B, C):    compute the m_tile and n_tile of this CTA    square_sum = zeros(m_tile)    acc = zeros(m_tile, n_tile)    for each k_tile:        tile_A = A[m_tile][k_tile]        tile_B = B[k_tile][n_tile]        acc += tile_A @ tile_B        square_sum += (tile_A * tile_A).sum(-1) # computed in parallel to the GEMM!    rstd = rsqrt(square_sum / A.shape[-1] + 1e-5)    acc *= rstd[:, None]    C[m_tile][n_tile] = acc
```

注意：尽管每个 k 迭代中仍会产生一些额外计算，但这些计算可以与矩阵乘法计算重叠执行。通过 warp 特化（warp specialization），kernel 的 warp 划分和执行流程如下：

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/06.webp)

::: image-caption
这张示意图分为三层计算分区：load分区负责逐段加载A、B矩阵数据（load A1、load B1等）；mma分区同步完成矩阵乘累加运算（acc += A1@B1等）；norm分区同步累加求和（sum += A1.sum()等），最后在尾声（epilogue）执行rmsnorm归一化计算。整体通过任务并行排布，实现数据加载、矩阵运算与归一化前置计算的流水线并行执行。
:::

需要注意的是，**该算法仍存在前缀融合的一个关键缺点：RMSNorm 计算会在多个 CTA 中重复执行** （试想所有处理相同行但不同列输出张量的 CTA；详见第 3 节）。然而，由于延迟预归一化确保了大部分计算与 TensorCore 完全重叠，这种冗余是可接受的，且仍能带来良好的性能提升。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/07.webp)

::: image-caption
Lazy Pre-Norm优化下RMSNorm算子延迟被掩盖占比的基准测试结果，横轴为不同(K,N)参数组合（M固定为256×1024），纵轴为占比百分比。各组数值均为正值，其中(2048,512)达98.01%为最高，(1024,2048)为41.49%为最低，整体掩盖效果显著，不同参数配置下优化效果差异明显。
:::

> 延迟预归一化算法的几个局限性：

1. 难以支持逐元素仿射变换——这类变换本质上是列方向乘法，会破坏我们“逐元素操作必须是行方向乘法”的前提；
2. 不适用于 LayerNorm——因为 LayerNorm 的逐元素部分包含减法，并非简单的行方向乘法；
3. 该融合的反向传播实现较为复杂——因为前向传播中从未显式生成 rmsnorm(A)，因此在计算 dA 和 dB 时，需要根据 A 和 rstd 实时重建 rmsnorm(A)。

## 三、多 CTA 归一化：将后归一化与线性层作为后缀融合

> 尽管延迟预归一化带来了良好的速度提升，但它仍有局限性，无法推广到大多数归一化场景。本节讨论一种更通用的后归一化与 GEMM 融合技术，回归后缀融合的思路，通过**CTA 集群（CTA clusters）** 和**分布式共享内存（Distributed Shared Memory, DSMEM）** 直接解决第 1 节提出的分块不匹配问题。

我们借鉴了Quack[11] 的思路，并将其从独立归一化核扩展到融合核。Quack 归一化核利用CTA 集群[12] 将大 N 维度分配给同一集群中的不同 CTA，让它们通过**分布式共享内存** 相互通信必要的数据，协作完成跨 N 维度的归约操作。这使得多个 CTA 能够协作分工处理同一行数据，并根据归一化需求进行通信，而无需承担全局内存 IO 成本。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/08.webp)

::: image-caption
GPU并行计算硬件架构：整体为包含多个Cluster的Grid结构，底层是可供所有GPU内核访问的全局内存Global Memory；上层分为多个Cluster，每个Cluster内含多个CTA，配备分布式共享内存，仅对同一Cluster内所有线程可见；CTA内部包含大量执行线程，形成层级化线程与内存布局，区分全局内存和集群级共享内存的访问范围。
:::

如前所述，大多数归一化操作可分解为归约部分（如 RMSNorm 的 rstd、LayerNorm 的均值和方差）和利用归约结果的后续逐元素部分。只有归约部分需要扫描整个 N 维度——我们通过 CTA 集群分而治之。由于归约结果通常较小（毕竟是归约操作），通过 DSMem 进行通信的开销非常小。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/09.webp)

::: image-caption
这张图对比了 single CTA 与 multi-CTA 两种 GPU 归约计算模式，整体流程分为加载行、归约计算、逐元素计算三步。single CTA 由单个 CTA 完成整体归约运算，直接输出结果；multi-CTA 采用分治策略，先由多个 CTA 各自做局部归约得到中间结果，再进行全局完整归约，最后完成逐元素计算，适配更大规模并行计算场景。
:::

需要注意的是，该思路解决的问题与我们在归一化融合中面临的问题完全一致——即 N 过大！（尽管 N 过大的原因和阈值不同）。这意味着我们可以直接将这种多 CTA 算法嵌入到 GEMM 的后缀中，即可完成融合：

```
def GEMM_norm_fusion_kernel(A, B, C):    计算当前CTA的m_tile和n_tile    acc = zeros(m_tile, n_tile)    for each k_tile:        tile_A = A[m_tile][k_tile]        tile_B = B[k_tile][n_tile]        acc += tile_A @ tile_B    acc = multi_cta_norm(acc)  # 此处进行DSMem通信    C[m_tile][n_tile] = acc
```

> 需要注意的是，这种融合并非无代价的，除了引入 DSMem 通信开销外，还会对核施加一些约束，可能导致基础 GEMM 核性能退化：

1. 对 CTA 调度有严格限制——特别是集群中的相邻 CTA 必须共享相同的 m_tile，但使用不同的 n_tile；
2. 由于约束 1，配对 CTA（paired-CTA）[13] 矩阵乘法难以实现；
3. 同样由于约束 1，无法进行分块超分组（tile super-grouping）[14] ；
4. 仍无法支持无限大的 N——最大 N 值为单个 CTA 限制（约 512）乘以最大集群大小。在 Blackwell GPU 上，可移植的最大集群大小为8[15]，因此 N 的最大限制为 4096。

尽管如此，节省大量内存 IO 的收益仍远大于这些局限性。Kernel 采用TLX[16] 实现——它在灵活性/开发效率和底层硬件控制之间取得了良好平衡，这在本案例研究中至关重要。我们基于支持 warp 特化的TLX GEMM 核[17] 构建了融合 Kernel，并在广告模型的典型形状（M=256k，K≈512，N≈512）上进行了基准测试，结果如下：

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/10.webp)

::: image-caption
基准测试柱状图，展示残差融合后归一化内核延迟占原始归一化内核延迟的百分比，对比 RMSNorm（蓝色）与 LayerNorm（红色）在不同 (K,N) 参数组合下的结果，M 固定为 256×1024。整体上多个参数组里 LayerNorm 占比显著更高，最高达 91.61%，体现融合对两种归一化算子延迟隐藏效果存在明显差异。注：我们将 K 和 N 的最大值限制为 2048，因为当它们达到 4096 时，延迟完全由 GEMM 主导，归一化仅占总延迟的 5%以下。
:::

### 反向传播怎么办？融合重组

> 本节讨论将上述融合思路应用于反向传播时的额外挑战：**前向传播中的后缀融合在反向传播中自然转变为前缀融合**。我们分析前缀融合的核心问题，并提出一种新颖的解决方案——在前向传播和反向传播中，将归一化与不同的 GEMM 融合，**最终在两者中均实现高效的后缀融合**。

LayerNorm 和 RMSNorm 的反向传播也涉及归约操作——这一点与前向传播类似，我们可以通过相同的方式解决。:mdnice-bold-italic[高效的反向传播还需要利用前向传播中存储的中间归约结果——这也不是大问题，因为归约结果是一维的，IO 开销极小。真正的问题在于：前向传播中的后缀融合在反向传播中会转变为前缀融合。]

```
# 前向传播公式C = norm(A @ B)# 反向传播公式dA = norm_backward(dC) @ B.TdB = A.T @ norm_backward(dC)
```

注：在反向传播中，norm_backward 计算发生在 GEMM 之前，因此潜在的融合形式是前缀融合。我们在第 2 节中讨论了前缀融合的一些普遍缺点，但在这个特定场景中，让我们通过一个潜在的前缀融合方案来更深入地理解问题所在：

```
def GEMM_norm_bwd_fusion_kernel(dC, BT, dA):    计算当前CTA的m_tile和n_tile    acc = zeros(m_tile, n_tile)    for each k_tile:        tile_dC = dC[m_tile][k_tile]        tile_dC = multi_cta_norm_bwd(tile_dC)  # 此处进行DSMem通信        tile_BT = BT[k_tile][n_tile]        acc += tile_dC @ tile_BT    dA[m_tile][n_tile] = acc
```

这种方案存在几个性能问题：

1. 归一化反向传播计算位于关键路径上，阻塞了每个迭代的 GEMM 计算——相比之下，后缀融合中仅需在主循环结束时执行一次计算；
2. 归一化反向传播存在重复计算——注意，dC 的每一行会被处理相同 m_tile 但不同 n_tile 的多个 CTA 分别加载，每个 CTA 都会对相同的 dC 分块执行相同的归一化反向传播计算；
3. 该融合核仅计算 dA，而 dB 的计算还需要再次执行 norm_backward(dC)，导致更多重复计算。

> 解决方案是什么？除非是第 2 节讨论的特殊场景，否则几乎没有更好的选择——因此，只需避免前缀融合即可。要实现这一点，我们需要在融合策略上更加灵活：由于归一化层通常位于线性层之间，我们是否可以在前向传播和反向传播中将归一化与不同的线性层融合？

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/11.webp)

::: image-caption
前向传播将 Linear1 矩阵乘法与 Normalization 归一化算子融合，结果再输入 Linear2 计算；反向传播则把 Linear2 梯度计算和归一化反向传播算子融合，再求得 Linear1 的梯度 dA、dB 以及 Linear2 梯度 dY，通过算子融合减少显存访问开销，提升计算效率。
:::

不难看出，此时反向传播也成为了后缀融合，归一化反向传播不再存在重复计算。更重要的是，融合的结构与前向传播完全一致！只需将前向核中的`multi_cta_norm`替换为`multi_cta_norm_bwd`，即可得到反向传播核。以下是与前向传播相同形状的基准测试结果：

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/12.webp)

::: image-caption
这张柱状基准测试图展示残差融合后归一化反向内核延迟占原反向内核延迟的百分比，对比蓝色 RMSNorm 与红色 LayerNorm，M 固定为 256×1024，横轴为不同 (K,N) 参数组合。两组算子数值随参数变化存在差异，LayerNorm 最高达 70.71%，RMSNorm 最高达 69.21%，直观体现 epilogue 融合对反向归一化算子延迟隐藏效果的区别。
:::

::: block-1
**该思路不仅适用于线性层**。例如，在 LLM 架构中，可能存在“注意力 → 归一化 → 线性”或反之的结构——:mdnice-highlight[只要归一化前后是计算密集型操作（融合后能带来收益且技术可行），都可以采用相同的优化技术。]
:::

下一节将讨论一个将归一化融合到注意力机制中的示例。

## 四、FlashNormAttention：将预归一化和后归一化同时融合到 FlashAttention 变体核中

> 本节以GDPA 核[18] 为案例研究，应用上述融合思路，提出**FlashNormAttention** 算法。GDPA 核在 Meta 广告模型（尤其是昆仑（Kunlun）[19] 架构）中被广泛使用，是基于FlashAttention[20] 重新设计的通用注意力核。

因此，以下优化思路大多可推广到 FlashAttention 等其他注意力核。该算法采用与上述多 CTA GEMM+归一化融合完全相同的核心算法技巧，但复杂度更高（注意力机制 vs GEMM，融合两个归一化 vs 一个）。我们采用了多种优化技术以确保其性能，具体如下：

- SMEM/TMEM 重用，减少内存压力；
- 寄存器子分块（Register subtiling），降低寄存器压力；
- 精细调优的 warp 特化，并行化密集型 CUDACore 操作；
- 反向传播中重计算归一化，避免前向传播中存储额外张量的 IO 成本；
- 利用 TMA_REDUCE_ADD[21] 和 TensorCore 累加等高级硬件特性，提升流水线效率。

昆仑（Kunlun）架构中的典型 PFFN 块以 GDPA 核为核心，但在其前后还包含多个归一化和残差连接。下图展示了 PFFN 块内部的数据流：

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/13.webp)

::: image-caption
Transformer 注意力模块的结构流程图。输入包含查询 q、键值 k 和 v；q 先经过 LayerNorm 归一化，再与 k、v 送入 GDPA 注意力计算；随后与原始 q 残差相加，经 RMSNorm 归一化后再次残差连接，最终输出结果。整体融合两种归一化算子与残差连接，构成完整自注意力前向计算链路。
:::

注：该结构的 IO 密集度很高，归约和逐元素核分散在 GDPA 核的前后。我们的目标是将所有这些操作融合到一个核中——即**FlashNormAttention**。这类似于执行整个模块计算的“巨核（megakernel）”，但在意图和意义上与原始MegaKernel[22]不同：它不仅旨在节省核启动成本，更重要的是**减少与 HBM 之间的总数据传输量** ——这通常是更大的性能瓶颈。

::: block-1
整体融合方案保持不变：利用 CTA 集群和分布式共享内存协作计算归一化。需要注意的是，此处的 GDPA 是多头注意力机制，而归一化是跨所有头执行的。因此，尽管在典型的 GDPA/FA 核中，单个 CTA 可以访问完整的头维度，但仍需要来自其他头的数据才能执行归一化——这使得多 CTA 归一化算法成为必需。
:::

首先，我们给出原始 GDPA 核算法的伪代码（为简洁起见，仅保留与融合相关的核心部分）。有关详细的优化算法，请参考原始GDPA 博客[23]。

```
# 输入：Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]# 元参数：BLOCK_M（seq_len_q上的m_tile分块大小），BLOCK_N（seq_len_kv上的n_tile分块大小）# 网格划分：(batch_size, seq_len_q // BLOCK_M, H)def gdpa_fwd_kernel(Q, K, V, output):    计算当前CTA的batch_idx、m_tile和head_idx    q = Q[batch_idx, m_tile, head_idx, :]  # [BLOCK_M, head_dim]，索引B和H维度    acc = zeros(BLOCK_M, head_dim)    for n_tile in 遍历整个seq_len_kv:        k = K[batch_idx, n_tile, head_idx, :]  # [BLOCK_N, head_dim]，索引B和H维度        v = V[batch_idx, n_tile, head_idx, :]  # [BLOCK_N, head_dim]        p = elementwise_activation(q @ k.T)  # [BLOCK_M, BLOCK_N]        acc += p @ v  # [BLOCK_M, head_dim]    output[batch_idx, m_tile, head_idx, :] = acc
```

在开始融合工作之前，我们先对该算法进行一个关键调整：在昆仑（Kunlun）的使用场景中，:mdnice-bold-italic[我们观察到 seq_len_q 通常较大（约 1k），而 seq_len_kv 通常较小（约 128）。这导致内层循环的流水线深度很浅，暴露了前缀和后缀开销。为改善这种场景的性能，我们交换了 Q 和 K/V 在核中的角色：基于 KV 进行网格划分，遍历 Q]。需要注意的是，只有当不对 seq_len_kv 进行分块时，该算法在数值上才是正确的。

```
# 输入：Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]# 元参数：BLOCK_M（seq_len_q上的m_tile分块大小）# 网格划分：(batch_size, H)def gdpa_fwd_kernel_short_kv(Q, K, V, output):    计算当前CTA的batch_idx和head_idx    k = K[batch_idx, :, head_idx, :]  # [seq_len_kv, head_dim]，索引B和H维度    v = V[batch_idx, :, head_idx, :]  # [seq_len_kv, head_dim]    for m_tile in 遍历整个seq_len_q:        q = Q[batch_idx, m_tile, head_idx, :]  # [BLOCK_M, head_dim]，索引B和H维度        p = elementwise_activation(q @ k.T)  # [BLOCK_M, BLOCK_N]        output[batch_idx, m_tile, head_idx, :] = p @ v  # [BLOCK_M, head_dim]
```

尽管该优化与本文主题没有直接关联，但我们仍在此提及——因为我们的融合核是基于该版本构建的，且基准测试也是基于该版本进行的。为保持完整性，我们补充了原始 GDPA 博客中未提及的这一细节。

现在，我们将归一化和残差连接融合到核中。思路与上述多 CTA 归约一致。**唯一需要注意的是：同一集群中的 CTA 应共享相同的 batch_idx，并处理不同的 head_idx**。此外，此处的 LayerNorm 是前缀融合——尽管流水线和矩阵乘法对其的依赖仍是问题（我们将在下文解决），但幸运的是，由于 KV 较短，我们无需担心 LayerNorm 的重复计算。因为我们不对 KV 的长度进行分块，所以 Q 的每个分块只会被一个 CTA 加载和处理。

```
# 输入：Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]# 元参数：BLOCK_M（seq_len_q上的m_tile分块大小）# 网格划分：(batch_size, H)def gdpa_fwd_fusion_kernel_short_kv(Q, K, V, output):    计算当前CTA的batch_idx和head_idx    k = K[batch_idx, :, head_idx, :]    v = V[batch_idx, :, head_idx, :]    for m_tile in 遍历整个seq_len_q:        q = Q[batch_idx, m_tile, head_idx, :]        ln_q = multi_cta_layernorm(q)  # 多CTA归一化1        p = elementwise_activation(ln_q @ k.T)        gdpa_out = p @ v        gdpa_out += ln_q  # 残差连接1        out = multi_cta_rmsnorm(gdpa_out)  # 多CTA归一化2        out += q  # 残差连接2        output[batch_idx, m_tile, head_idx, :] = out
```

> 尽管上述伪代码看起来简洁明了，但隐藏着两个关键问题：

1. **内存压力**：这种大规模融合会带来巨大的寄存器和共享内存压力。我们需要保留许多以前不存在的变量（如 ln_q 和 rmsnorm(gdpa_out)）。如果执行是纯顺序的，这不会有问题——因为生成一个操作的输出后，其输入可以立即释放。:mdnice-bold-italic[但由于残差连接的存在，情况并非如此：注意，q 和 ln_q 的生命周期很长，因为后续的残差连接需要使用它们。这意味着我们必须为这些变量分配专门的共享内存，从而增加了总内存需求。] 实际上，在这个朴素版本的算法中，我们观察到共享内存使用量翻倍，显著超出了硬件限制。
2. **CUDACore 主导与流水线阻塞**：尽管融合消除了大部分内存 IO，但归一化和残差连接的 CUDACore 计算仍然存在，并会阻塞 TensorCore 的利用率。需要通过精细调优的 warp 特化和流水线设计，尽可能隐藏 CUDACore 的延迟。

> 针对内存压力，我们采用了三项主要优化思路：

- **内存缓冲区重用**：节省内存使用的关键技术是让非重叠数据共享相同的内存缓冲区。例如，输出变量 out 的共享内存缓冲区——在上述代码的最后一行，看似是直接将寄存器中的 out 存储到 HBM，但实际上我们首先将其放入 SMEM 缓冲区，然后调用 TMA 异步将 SMEM 中的数据存储到 HBM。显然，该缓冲区的生命周期很短，我们可以重用它来临时存储 ln_q 计算完成后、第二次残差连接使用前的 q。
- **利用张量内存（Tensor Memory）和 TensorCore 累加**：注意，ln_q 会立即与矩阵乘法结果 p@v 相加。我们无需将 ln_q 存储在 SMEM 中，然后在矩阵乘法完成后读取——因为这正是 tcgen05 在 TMEM 中支持的MMA[24] 语义。因此，我们可以直接将 ln_q 保存在为 p@v 分配的 TMEM 缓冲区中，并将 ln_q 的加法操作卸载到 TensorCore！这有助于同时节省 SMEM 占用和计算时间。
- **寄存器子分块**：特别是在我们这种 CUDACore 密集型核中，寄存器比 SMEM/TMEM 更为稀缺。除了精细的寄存器分配调优外，我们还通过寄存器子分块来缓解寄存器压力：将 SMEM/TMEM 中的张量分割成多个块，每次加载一个块到寄存器中执行归一化计算（包括归约和逐元素操作）。这有助于避免寄存器溢出（register spilling）——这会导致显著的性能退化。

> 针对流水线阻塞，我们采用了以下技术来提升流水线效率：

- **Warp 特化**：在原始 GDPA 设计中，我们有 4 个主要的特化 warp 分区（加载、MMA、激活、后缀）。在 FlashNormAttention 中，我们将 RMSNorm 计算分配给激活 warp，同时新增第五个分区专门用于前缀 LayerNorm 计算——以便更好地将其与 TensorCore 以及其他 CUDACore 操作（如前一次迭代的 RMSNorm 计算）重叠执行。执行流水线如下所示：我们为激活分区分配 8 个 warp（0-7），为 LayerNorm 分配 4 个 warp（8-11）。我们为激活 warp 分配最大数量的寄存器，同时限制 LayerNorm warp 的寄存器使用量，通过寄存器子分块在寄存器占用和延迟之间取得最优平衡。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/14.webp)

::: image-caption
GPU CUDA kernel 的 warp 执行时序视图，展示 gdpa_kernel 在 Blackwell 架构上各 warp 的任务调度。不同 warp 执行差异化算子任务：包含 GDPA 注意力计算、RMSNorm/LayerNorm 归一化运算、数据加载 / 存储、矩阵乘 mma、等待同步等阶段，色块表示不同计算 / 访存 / 等待周期，体现多 warp 并行执行、算子融合后的 GPU 指令排布与执行时序特征。
:::

- **寄存器预加载**：导致执行流水线阻塞的关键因素是残差连接引入的复杂数据依赖。q 和 ln_q 都需要长时间保存在内存中，阻碍了下一次迭代中这些张量的预取和预计算。由于 TMA 预取利用 SMEM，上述针对内存压力的优化 2 有效地将 ln_q 移出了关键路径。对于 q，我们让激活 warp 在 q 准备就绪后立即从 SMEM 预加载，并一直持有到第二次残差连接执行完毕。这样，我们可以立即释放 q 的 SMEM 空间，以便在处理当前迭代时加载下一次迭代的 q。由于我们为激活 warp 分配了最大数量的寄存器，寄存器压力仍然可控。

以下是我们使用的典型 GDPA 形状的基准测试结果：KV 是长度为 128 的密集序列；Q 是稀疏序列，平均稀疏度为 0.5，最大长度可变；批量大小为 768；头维度设为 128；头数可变（以反映不同归一化维度和不同 CTA 集群大小下的性能表现）。由于融合复杂度较高，我们将延迟节省百分比定义为相对于总基准延迟（而非仅归一化/逐元素核延迟）的提升。基准测试采用 Inductor 编译结果。

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/15.webp)

::: image-caption
该柱状图展示 GDPA Megakernel 前向传播的延迟节省率，横轴为查询序列长度，三组颜色分别代表 H=2、H=3、H=4。整体均实现显著延迟节省，H=2 的优化效果整体最优，最高可达 35.39%；H=3 次之，H=4 节省率相对偏低。随序列长度变化，不同头数设置的优化收益呈现差异化波动，验证算子融合 megakernel 的加速效果。
:::

### 反向传播实现

> 为简洁起见，我们省略反向传播的详细算法，仅说明其关键优化以及与前向传播的异同：

- **算法结构**：首先需要注意，反向传播的融合与前向传播非常相似——因为归一化等归约操作的反向传播也是归约操作，残差连接的反向传播也是残差连接；
- **重计算（Recompute）**：原始 GDPA 反向传播核通过重计算 q@k 来节省前向传播的内存 IO。我们在融合核中采用了相同的思路：首先，我们的核中 q@k 变为 ln(q)@k，因此需要先重计算 ln(q)。我们通过在前向传播中存储均值和方差（成本很低，因为是一维的）来避免此处的 DSMem 开销，以便反向传播可以轻松推导 ln(q)；其次，RMSNorm 反向传播需要其输出，我们通过 rmsnorm_out = kernel_out – q 重计算；最后，残差连接反向传播还需要 RMSNorm 的输入，我们通过 rmsnorm_in = rmsnorm_out / rstd 重计算；
- **Warp 特化**：原始 GDPA 反向传播核有 4 个 warp 分区：MMA、激活、加载、归约（用于原子累加 dQ）。我们采用相同的结构，将除 LayerNorm 反向传播外的所有新计算分配给激活分区；将 LayerNorm 反向传播计算分配给归约分区，以促进更好的流水线（类似于前向传播中将前缀 LayerNorm 分配给独立分区）；
- **内存压力**：反向传播面临的内存压力比前向传播更为严峻——仅仅是因为它需要存储和计算的数据量更大。为缓解这一问题，我们采用了与前向传播类似的激进 SMEM/TMEM 重用策略，但也不得不缩小分块大小；
- **流水线效率**：与前向传播类似，残差的长生命周期会阻塞流水线。对于内部残差，我们仍可以使用 TMEM 和 TensorCore 累加来打破生命周期限制；但对于外部残差，由于反向传播中寄存器压力更高，我们无法承担寄存器预加载的成本。因此，我们的解决方案是使用TMA_REDUCE_ADD[25]，在残差准备就绪后立即将其从 SMEM 直接累加到 HBM。

这乍看之下可能违背直觉——:mdnice-bold-italic[因为 FlashNormAttention 的核心目标是减少内存 IO，但实际上这是合理的：融合后，内存 IO 不再是核的瓶颈，因此我们愿意牺牲少量内存 IO 来缓解当前的瓶颈（计算和流水线阻塞）。]

> 下图展示了反向传播融合的性能提升，使用与前向传播完全相同的形状：

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/16.webp)

::: image-caption
这张柱状图展示 GDPA Megakernel 反向传播的延迟节省率，横轴是查询序列长度，蓝色 H=2、红色 H=3、黄色 H=4 代表不同头数配置。整体实现正向延迟优化，H=2 在中短序列优化效果最佳，最高达 18.08%；H=4 整体节省率最低。随序列长度变化，三种头数的优化效果呈现不同波动趋势，验证反向 megakernel 融合的加速收益。
:::

## 结语

> 本文针对大语言模型与生成式广告推荐模型中 LayerNorm、RMSNorm 带来的内存带宽瓶颈，提出一整套归一化内核融合优化方案，有效解决传统 GEMM 与归一化因分块逻辑冲突难以高效融合的行业难题。

文中先后提出三类核心算法：Lazy Pre-Norm 借助数学恒等变换破除预归一化计算循环依赖，实现无仿射 RMSNorm 与矩阵乘的前序融合；Multi-CTA Norm 依靠多 CTA 集群与分布式共享内存协同归约，实现通用后归一化融合，最高可掩盖 90% 归一化算子时延；FlashNormAttention 将前后归一化、残差连接与 GDPA 注意力整合为单 MegaKernel，配合分片、 warp 特化等手段缓解显存压力，正向训练最高降低 35.39% 整体时延。同时本文设计前后向差异化融合策略，规避反向归一化冗余计算。

::: block-1
整套方案在 NVIDIA B200 GPU 完成验证并开源，但仍存在适用约束：Lazy Pre-Norm 不支持带仿射变换与 LayerNorm，多 CTA 集群维度存在上限。**未来可拓展至更多注意力变体、混合精度训练场景，为深度学习训练内核 IO 优化提供通用落地范式，充分释放 GPU TensorCore 硬件算力。**
:::

参考资料

[1]

TLX: *https://arxiv.org/abs/2605.10905*

[2]

Helion: *https://pytorch.org/blog/helion/*

[3]

Flash Attention: *https://github.com/Dao-AILab/flash-attention*

[4]

Quack: *https://github.com/Dao-AILab/quack*

[5]

Helion: *https://github.com/pytorch/helion*

[6]

TLX: *https://github.com/facebookexperimental/triton/tree/main*

[7]

生成式广告模型（GEM）: *https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/*

[8]

——的[昆仑（Kunlun）: *https://arxiv.org/abs/2602.10016*

[9]

逐元素仿射变换: *https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html*

[10]

Helion: *https://pytorch.org/blog/helion/*

[11]

Quack: *https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md*

[12]

CTA 集群: *https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-of-cooperative-thread-arrays*

[13]

配对 CTA（paired-CTA）: *https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair*

[14]

分块超分组（tile super-grouping）: *https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations*

[15]

8: *https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#thread-block-clusters*

[16]

TLX: *https://pytorch.org/blog/enabling-cluster-launch-control-with-tlx/*

[17]

TLX GEMM 核: *https://github.com/facebookexperimental/triton/blob/main/third_party/tlx/tutorials/blackwell_gemm_ws.py*

[18]

GDPA 核: *https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/*

[19]

为案例研究，应用上述融合思路，提出**FlashNormAttention** 算法。GDPA 核在 Meta 广告模型（尤其是[昆仑（Kunlun）: *https://arxiv.org/abs/2602.10016*

[20]

架构）中被广泛使用，是基于[FlashAttention: *https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute*

[21]

TMA_REDUCE_ADD: *https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278*

[22]

MegaKernel: *https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles*

[23]

GDPA 博客: *https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/*

[24]

MMA: *https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma*

[25]

TMA_REDUCE_ADD: *https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278*

::: related-title
**相关推荐**
:::

::: related-links
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**
- **[Helion DSL 赋能 vLLM 实现跨硬件高效推理：Qwen3 模型 FP8 推理加速实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902433&idx=1&sn=5e65e917235f732c27c1cd121a2d6d24&scene=21#wechat_redirect)**
- **[借助 torch.compile 实现 Normalization 顶尖性能表现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899700&idx=1&sn=dad00ee6111ffb73275bae4c98c8107f&scene=21#wechat_redirect)**

:::
