关键词：**Lean4 形式化推理**、:mdnice-bold-italic[TileLang GPU Kernel 生成]、张量程序超优化器、算子融合与显存优化、**Attention 4.08x 加速**

![文章插图](/articles/2026-05-15-gpu-lean4-tilelang-attention-4-7a32fcc8/images/01.webp)

- **Lean4-TileLang 张量程序超级优化器[开发中]**
- https://leloykun.github.io/ponder/lean4-tilelang/
- 5000 字，阅读 18 分钟，**播客 25 分钟**

相关推荐

- [无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)
- [DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)
- [PipeThreader：软件定义流水线驱动的 DNN 编译器，FlashAttention与Mamba2性能显著超越现有方案！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889807&idx=1&sn=a8f2bfdca2091144fcd6059a575dda66&scene=21#wechat_redirect)

> 本文是对 Franz Louis Cesista 在博客文章 Lean4-TileLang Tensor Program Superoptimizer [WIP][1] 中相关内容的解读，并结合 `tile-ai/tilelang` 仓库，解释 Lean4、TileLang、张量计算图、GPU kernel 优化和 superoptimizer 之间的关系。

原文展示了一个仍在开发中的系统：用 Lean4 描述神经网络计算图，再自动搜索等价但更高效的实现方式，最后生成 TileLang kernel，在 GPU 上执行。

本文会尽量用更直观的语言解释它在做什么、为什么有意义，以及其中几个典型 kernel 的优化思路。

## 本文目录

- 一、先给出整体结论
- 二、TileLang 是什么？
- 三、为什么需要 Lean4？
- 四、为什么要做这种自动化优化？
- 五、实验结果先看一眼
- 六、Attention：接近 FlashAttention 的优化思路
- 七、SwiGLU：融合两个矩阵乘和激活函数
- 八、Matmul：基础 GEMM 的优化空间相对有限
- 九、RMSNorm：正确的变换不一定更快
- 十、RMSNorm-MLP：融合后收益重新出现
- 十一、从这些 case 中能看出什么规律？
  - 1. 显存读写往往比计算更关键
  - 2. 算子融合是主要收益来源
  - 3. 基础算子单独优化空间有限
  - 4. 自动搜索需要好的 cost model
- 十二、这种系统的长期价值
- 十三、结合 TileLang 来看它的位置
- 十四、一个形象比喻
- 总结

![文章插图](/articles/2026-05-15-gpu-lean4-tilelang-attention-4-7a32fcc8/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、先给出整体结论

根据 Lean4-TileLang Tensor Program Superoptimizer[2] 的内容，可以把整个系统理解为：

> 用 Lean4 描述“要算什么”，用 superoptimizer 搜索“怎么等价地改写”，再用 TileLang 生成“真正跑得快的 GPU kernel”。

也就是说，它不是单纯手写几个高性能 kernel，而是尝试搭建一条自动化链路：

```
神经网络算子的数学定义        ↓Lean4 计算图        ↓等价变换与搜索        ↓生成 TileLang kernel        ↓JIT 编译        ↓在 GPU 上运行并 benchmark
```

在这个流程里：

| 组件 | 作用 |
| --- | --- |
| Lean4 | 描述计算图，表达和约束等价变换 |
| Superoptimizer | 在多个等价实现中搜索更快的方案 |
| TileLang | 把搜索结果编译成高性能 GPU kernel |
| Autotune | 搜索 block size、线程数、pipeline stage 等参数 |
| Benchmark | 判断哪个实现实际更快 |

::: block-1
换句话说：Lean4 负责“算得对”，TileLang 负责“跑得快”。
:::

## 二、TileLang 是什么？

> `tile-ai/tilelang` 是一个用于写高性能 GPU / CPU kernel 的 DSL，也就是领域专用语言。

它的目标是让开发者用相对简洁的 Python 风格代码，写出接近 CUDA、CUTLASS、TVM 等底层系统性能的 kernel。

在 `tile-ai/tilelang` 仓库中，TileLang 支持的典型算子包括：

- GEMM / Matmul；
- Dequant GEMM；
- FlashAttention；
- Linear Attention；
- Convolution；
- DeepSeek MLA；
- 其他复杂 AI kernel。

TileLang 代码看起来像 Python，但它不是普通 Python。它会通过 JIT 编译成底层 kernel。常见写法包括：

| TileLang 写法 | 通俗解释 |
| --- | --- |
| `@tilelang.jit` | 把函数编译成可执行 kernel |
| `@T.prim_func` | 定义一个底层计算函数 |
| `T.Kernel(...)` | 定义 GPU kernel 的 grid / block 结构 |
| `T.alloc_shared(...)` | 分配 shared memory |
| `T.alloc_fragment(...)` | 分配寄存器级别的局部累加区 |
| `T.copy(...)` | 在显存、shared memory、寄存器之间搬数据 |
| `T.gemm(...)` | 调用高性能矩阵乘核心 |
| `T.Pipelined(...)` | 让数据搬运和计算流水化 |
| `T.use_swizzle(...)` | 改变 block 调度，提高 cache 利用 |
| `tilelang.autotune(...)` | 自动搜索更好的性能参数 |

可以把 TileLang 理解成：

> 比 PyTorch 更底层，比 CUDA 更易写的 kernel 编程语言。

PyTorch 更适合表达模型逻辑，而 TileLang 更适合表达“这个算子在 GPU 上到底怎么分块、怎么搬数据、怎么用 shared memory、怎么调用 tensor core”。

## 三、为什么需要 Lean4？

如果 TileLang 已经能写高性能 kernel，为什么还要引入 Lean4？

关键原因是：GPU kernel 优化经常涉及数学等价变换。

例如 Attention 的普通写法是：

```
scores = Q @ Kᵀprobs = softmax(scores)out = probs @ V
```

但是高性能 FlashAttention 类实现不会真的把完整的 `scores` 矩阵写入显存，而是分块计算，并用在线 softmax 维护中间状态。

这就需要保证：

```
分块在线计算的结果
```


```
完整 scores + softmax + matmul 的结果
```

在数学上是等价的。

再比如 RMSNorm 中有一个变换：

```
out = (X * S) @ W
```

在某些条件下可以改写成：

```
out = (X @ W) * S
```

这里要求 `S` 是按行广播的缩放因子。如果条件不满足，这个变换就可能是错的。

Lean4 的作用，就是为这些计算图和变换提供更严谨的表达基础。

它可以帮助系统描述：

- 输入输出 shape；
- axis 约束；
- reduction 语义；
- broadcast 规则；
- matmul 的 contraction axis；
- elementwise 操作；
- 哪些变换是合法的；
- 哪些计算图是等价的。

所以，在这个系统中，Lean4 更像是“数学与推理层”，TileLang 更像是“高性能执行层”。

## 四、为什么要做这种自动化优化？

> 神经网络模型越来越复杂，新结构层出不穷。比如新的 Attention 变体、新的 MLP 结构、新的 Norm、新的门控机制等。

如果每提出一个新结构，都要手写一套 CUDA kernel，会有几个问题：

1. 开发成本很高；
2. 需要非常专业的 GPU kernel 经验；
3. 很容易写错；
4. 很难快速比较大量候选结构；
5. 小规模实验结果可能无法代表大规模训练表现。

博客原文开头提到，作者并不看好只在小规模上做“代码 fuzzing”式的自动研究，因为小规模表现好并不代表大规模也好。要公平比较不同架构，至少要同时考虑：

- GPU / TPU kernel 是否优化充分；
- optimizer 是否合适；
- 参数化方式是否合理；
- 超参数如何随宽度、深度、batch size、训练步数变化；
- 小规模规律能不能迁移到大规模。

因此，自动生成高性能 kernel 是更大目标中的一部分：

```
用形式化系统定义神经网络结构        ↓自动生成高性能 kernel        ↓自动调整 optimizer / parametrization        ↓研究 scaling law        ↓更公平地比较不同架构
```

本文主要解读其中“Lean4 → TileLang kernel”这一段。

## 五、实验结果先看一眼

> 根据原文给出的 benchmark，几个 workload 的结果如下：

| Workload | Case | TileLang | torch.compile | Speedup |
| --- | --- | --- | --- | --- |
| Attention | h16_tq4096_tkv4096_dh128 | 0.531 ms | 2.169 ms | 4.079x |
| SwiGLU | m1024_n2048_d2048 | 0.057 ms | 0.122 ms | 2.140x |
| Matmul | m1024_d4096_n4096 | 0.125 ms | 0.162 ms | 1.294x |
| RMSNorm | m1024_n4096_d4096 | 0.207 ms | 0.174 ms | 0.842x |
| RMSNorm-MLP | m1024_n1024_d1024 | 0.038 ms | 0.072 ms | 1.912x |

可以先得到几个直观结论：

1. Attention 加速很明显，约 4 倍；
2. SwiGLU 也有明显提升，约 2 倍；
3. Matmul 提升有限，因为矩阵乘本身已经被高度优化；
4. RMSNorm 单独 case 反而慢了；
5. RMSNorm-MLP 融合后又变快了。

这说明一个很重要的事实：

> TileLang 和自动优化器的优势，更多体现在“算子融合”和“减少显存读写”上，而不仅仅是把单个基础算子写得更快。

## 六、Attention：接近 FlashAttention 的优化思路

Attention 的数学形式是：

```
scores = Q @ Kᵀprobs = softmax(scores)out = probs @ V
```

对应的 Lean4 描述大致是：

```
let q ← input "Q" shapeHTqDhlet k ← input "K" shapeHTkvDhlet v ← input "V" shapeHTkvDhlet scores ← matmul q k "d_h" "d_h"let probs ← softmaxSubdag scores "t_kv"let out ← matmul probs v "t_kv" "t_kv"output out
```

这表达的是标准 Attention 计算图。

问题在于，`scores` 会非常大。以原文配置为例：

```
h = 16t_q = 4096t_kv = 4096d_h = 128
```

那么 `scores` 的形状是：

```
16 × 4096 × 4096
```

如果完整生成这个矩阵，再写入显存、读回来做 softmax、再和 `V` 相乘，显存流量会非常大。

![文章插图](/articles/2026-05-15-gpu-lean4-tilelang-attention-4-7a32fcc8/images/03.gif)

::: image-caption
Derivation of top-1 kernel
:::

高性能实现的关键，是避免完整 `scores` 落地。

生成的 TileLang kernel 采用类似 FlashAttention 的分块在线计算方式：

```
固定一块 Q循环读取 K/V 的小块每次计算局部 QKᵀ做局部 softmax 统计更新全局 softmax 状态累加输出
```

TileLang 代码中有几个关键状态：

```
state_pass0_mstate_pass0_lstate_pass0_o
```

它们可以这样理解：

| 状态变量 | 含义 |
| --- | --- |
| `state_pass0_m` | 已处理 block 中的最大值 |
| `state_pass0_l` | softmax 分母累积值 |
| `state_pass0_o` | attention 输出累积值 |

为什么需要这些状态？

因为 softmax 不能简单地分块各算各的。要保证分块计算和完整 softmax 等价，就必须维护全局最大值和归一化因子。

这个在线 softmax 过程可以理解为：

```
每来一块新的 K/V就把它纳入当前 softmax 统计同时修正旧的累积结果
```

这样就不需要存储完整 `scores`。

这正是 FlashAttention 系列算法的核心思想：

> 用更多片上计算，换取更少显存读写。

因此 Attention case 能达到约 4 倍加速，并且原文中标注其效果相当于 Flash Attention 2。

## 七、SwiGLU：融合两个矩阵乘和激活函数

SwiGLU 是大语言模型 MLP 中常见的结构。它可以简化写成：

```
up = X @ W_upgate = X @ W_gateout = up * silu(gate)
```

其中：

```
silu(x) = x * sigmoid(x)
```

Lean4 计算图描述为：

```
let x ← input "X" shapeMDlet wUp ← input "W_up" shapeDNlet wGate ← input "W_gate" shapeDNlet gatePre ← matmul x wGate "d" "d"let up ← matmul x wUp "d" "d"let gate ← silu gatePrelet out ← mul gate upoutput out
```

如果用普通框架执行，可能会产生多个中间张量：

```
gatePreupgateout
```

每个中间张量都可能写回显存，再读出来做下一步。

![文章插图](/articles/2026-05-15-gpu-lean4-tilelang-attention-4-7a32fcc8/images/04.gif)

::: image-caption
Derivation of top-1 kernel
:::

TileLang kernel 的做法是把这些步骤融合：

1. 同一个 kernel 中计算 `X @ W_up`；
2. 同一个 kernel 中计算 `X @ W_gate`；
3. 直接对 `gate` 做 SiLU；
4. 直接和 `up` 相乘；
5. 最后只写最终输出。

核心代码形态类似：

```
T.gemm(input_0, input_4, matmul_8, clear_accum=False)    # upT.gemm(input_0, input_10, matmul_14, clear_accum=False)  # gate
```

随后融合 SiLU 和乘法：

```
matmul_14[i0, i1] = matmul_14[i0, i1] / (1.0 + T.exp2((-(matmul_14[i0, i1])) * scale))matmul_14[i0, i1] = matmul_8[i0, i1] * matmul_14[i0, i1]
```

这里还用到了 fast math 技巧：

```
scale = 1.44269504  # log2(e)
```

因为：

```
exp(x) = exp2(x * log2(e))
```

在 GPU 上，`exp2` 往往更容易高效实现。

SwiGLU 的提速来自两个方面：

1. 两个 matmul 共享同一个输入 tile；
2. SiLU 和乘法不再单独写中间结果到显存。

所以该 case 获得了约 2.14 倍加速。

## 八、Matmul：基础 GEMM 的优化空间相对有限

Matmul 是最基础的矩阵乘：

```
O = X @ W
```

Lean4 图非常简单：

```
let x ← input "X" shapeMDlet w ← input "W" shapeDNlet out ← matmul x w "d" "d"output out
```

![文章插图](/articles/2026-05-15-gpu-lean4-tilelang-attention-4-7a32fcc8/images/05.gif)

::: image-caption
Derivation of top-1 kernel
:::

生成的 TileLang kernel 是标准 tiled GEMM：

```
把 X 的一块搬到 shared memory把 W 的一块搬到 shared memory用 T.gemm 做 tile-level 矩阵乘沿 d 维循环累加最后写回 O
```

核心代码类似：

```
for k_pass0 in T.Pipelined(T.ceildiv(d, block_d), num_stages=num_stages):    T.copy(X[gy * block_m, k_pass0 * block_d], input_0)    T.copy(W[k_pass0 * block_d, gx * block_n], input_4)    T.gemm(input_0, input_4, matmul_8, clear_accum=False)
```

它使用了典型 GPU GEMM 优化手段：

- tiling；
- shared memory；
- fragment 累加；
- pipelined loop；
- swizzle；
- tensor core GEMM。

不过，矩阵乘本身是 GPU 上被优化得最充分的算子之一。cuBLAS、CUTLASS、PyTorch 后端、torch.compile 等都已经做了大量优化。

因此，TileLang 版本只比 `torch.compile` 快约 1.29 倍，并不意外。

这里可以得到一个重要判断：

> 自动生成 TileLang kernel 的最大价值，不一定在于单独打败成熟 GEMM 库，而在于把 GEMM 和其他操作融合起来。

## 九、RMSNorm：正确的变换不一定更快

RMSNorm 的简化公式是：

```
S = 1 / sqrt(mean(X²) + eps)norm = X * Sout = norm @ W
```

Lean4 中先描述 RMSNorm 子图：

```
let x2 ← square xlet mean ← meanAlongSubdag x2 axislet eps ← epsConstlet denom ← add mean epslet scale ← rsqrt denommul x scale
```

再接一个 matmul：

```
let out ← matmul norm w "d" "d"
```

自动生成的 TileLang 实现把它拆成两个 kernel。

第一个 kernel 计算每一行的 scale：

```
S[gy * block_m + i0] = 1.0 / T.sqrt(row_state[i0] / d + eps)
```

第二个 kernel 做矩阵乘，并在最后乘上 scale：

```
T.gemm(X_tile, W_tile, acc_0, clear_accum=False)for i0, i1 in T.Parallel(block_m, block_n):    acc_0[i0, i1] = acc_0[i0, i1] * S[gy * block_m + i0]
```

这里使用了一个代数变换：

```
(X * S) @ W = (X @ W) * S
```

只要 `S` 是按行广播的缩放因子，这个变换就是合理的。

但是性能上出现了问题。

因为这个实现需要：

1. 启动一个 kernel 计算 `S`；
2. 把 `S` 写入显存；
3. 再启动一个 kernel 做 matmul；
4. 从显存读取 `S`；
5. 最后乘到输出上。

对于 RMSNorm 这种相对简单的算子，多一次 kernel launch 和中间读写可能会抵消优化收益。

因此结果是：

```
TileLang: 0.206 mstorch.compile: 0.174 msSpeedup: 0.842x
```

![文章插图](/articles/2026-05-15-gpu-lean4-tilelang-attention-4-7a32fcc8/images/06.gif)

::: image-caption
Derivation of top-1 kernel
:::

也就是说，这个 case 中 TileLang 版本反而慢了。

这说明：

> Superoptimizer 能找到数学上等价的实现，但数学等价不代表性能一定更优。

真实 GPU 性能还受到很多因素影响，例如：

- kernel launch 开销；
- 显存读写；
- shared memory 利用率；
- 寄存器压力；
- occupancy；
- cache 命中率；
- 编译器优化质量。

## 十、RMSNorm-MLP：融合后收益重新出现

RMSNorm-MLP 是更接近实际模型结构的组合算子：

```
norm = rmsnorm(X)up = norm @ W_upgate = norm @ W_gateout = up * gate
```

Lean4 图可以理解为：

```
let norm ← rmsNormSubdag x "d"let up ← matmul norm wUp "d" "d"let gate ← matmul norm wGate "d" "d"let out ← mul up gateoutput out
```

TileLang 实现仍然先算每行的 `S`：

```
S = scale_kernel(values['X'], eps_value)
```

然后第二个 kernel 同时计算两个 matmul：

```
T.gemm(X_tile, W_gate_tile, acc_0, clear_accum=False)T.gemm(X_tile, W_up_tile, acc_1, clear_accum=False)
```

再乘上 scale：

```
acc_0[i0, i1] = acc_0[i0, i1] * S[gy * block_m + i0]acc_1[i0, i1] = acc_1[i0, i1] * S[gy * block_m + i0]
```

最后做逐元素乘法：

```
acc_0[i0, i1] = acc_0[i0, i1] * acc_1[i0, i1]
```

虽然它也有单独计算 `S` 的开销，但第二个 kernel 融合了更多计算：

- `X @ W_gate`
- `X @ W_up`
- 两次 RMSNorm scale
- 最后的逐元素乘法

这样减少了多个中间张量的显存读写，收益重新超过开销。

所以 RMSNorm-MLP 获得了约 1.91 倍加速。

这个例子很好地说明：

> 单独一个小算子可能不值得特化优化，但多个算子融合后，特化 kernel 的优势会明显放大。

## 十一、从这些 case 中能看出什么规律？

综合几个 workload，可以看到一些通用规律。

### 1. 显存读写往往比计算更关键

Attention 和 SwiGLU 的加速，主要不是因为少算了多少 FLOPs，而是因为减少了中间张量的显存读写。

GPU 计算很快，但显存带宽有限。高性能 kernel 的核心目标之一就是：

```
让数据尽量留在寄存器或 shared memory 中不要频繁写回 global memory
```

### 2. 算子融合是主要收益来源

这些例子中收益最大的往往是 fusion：

```
Attention: matmul + softmax + matmulSwiGLU: matmul + matmul + silu + mulRMSNorm-MLP: rmsnorm + matmul + matmul + mul
```

融合越充分，越有机会省掉中间 tensor。

### 3. 基础算子单独优化空间有限

Matmul 本身已有非常成熟的优化库。自动生成 kernel 能赢一点已经不错，但很难轻松大幅领先。

真正有价值的是：

```
围绕 Matmul 做图级融合
```

而不是只优化孤立 GEMM。

### 4. 自动搜索需要好的 cost model

RMSNorm 变慢说明，搜索到的“等价实现”不等于“最快实现”。

一个好的 superoptimizer 不只要知道数学上能不能变换，还要估计：

- 多一个 kernel launch 值不值；
- 多写一个中间 tensor 值不值；
- 融合后寄存器压力会不会太高；
- shared memory 是否够用；
- tensor core 是否能充分利用；
- block size 是否合适。

因此，形式化推理和性能建模都很重要。

## 十二、这种系统的长期价值

根据原文设想，这套基础设施的目标并不只是优化几个算子，而是服务于更大的自动化 AI 研究流程。

未来如果要比较大量新模型结构，理想流程可能是：

```
在 Lean4 中定义模型结构        ↓自动生成候选计算图        ↓自动验证等价变换        ↓自动生成 TileLang kernel        ↓自动 benchmark        ↓自动调整 optimizer 和超参数        ↓研究 scaling law
```

这样做的意义是：

1. 降低新架构的 kernel 开发成本；
2. 避免因为 kernel 没优化好而误判架构优劣；
3. 更快探索新的 Attention、MLP、Norm 设计；
4. 让实验从“小规模代码试错”走向更系统的 scaling law 比较；
5. 为 AI 编译器引入形式化验证能力。

如果这个方向成熟，未来研究者可能只需要描述数学结构，系统就能自动生成足够高性能的实现。

## 十三、结合 TileLang 来看它的位置

> `tile-ai/tilelang` 仓库本身提供的是高性能 kernel 编程和编译能力。它有点像一个“面向 AI kernel 的底层执行语言”。

而 Lean4 superoptimizer 则更像是 TileLang 上方的一层“自动生成器”。

二者关系可以画成：

```
Lean4 / Superoptimizer负责：- 计算图描述- 等价变换- 搜索 derivation- 选择实现策略        ↓ 生成TileLang负责：- shared memory tiling- fragment accumulation- pipelining- swizzle- T.gemm- JIT 编译- autotune- 后端代码生成        ↓ 编译运行GPU / CPU / Accelerator
```

所以，TileLang 在这里承担的是后端角色：

> 它把上层搜索得到的计算方案，变成真正可执行且高性能的 kernel。

## 十四、一个形象比喻

可以把整个系统比作自动做菜。

| 做菜类比 | 技术对应 |
| --- | --- |
| 菜谱 | Lean4 计算图 |
| 判断步骤能不能调整 | 形式化等价变换 |
| 尝试不同做法 | Superoptimizer |
| 选择最快方案 | Benchmark / autotune |
| 下锅执行 | TileLang kernel |
| 切块大小 | block size |
| 锅具和火候 | shared memory、tensor core、pipeline |
| 厨房 | GPU |

传统 CUDA 优化像是大厨亲自下厨，每一步都靠经验。

Lean4 + TileLang 的思路是：

> 用户描述要做什么菜，系统自动推理哪些步骤可以合并、怎么切块、怎么下锅，最后生成一套高效做法。

## 总结

根据 Lean4-TileLang Tensor Program Superoptimizer[3] 的内容，可以将其核心思想概括为：

> 用 Lean4 严谨描述张量计算，用 superoptimizer 搜索等价实现，用 TileLang 生成高性能 GPU kernel。

实验结果显示：

- Attention：约 4.08 倍加速；
- SwiGLU：约 2.14 倍加速；
- Matmul：约 1.29 倍加速；
- RMSNorm：反而慢于 `torch.compile`；
- RMSNorm-MLP：约 1.91 倍加速。

这些结果说明：

1. 算子融合是主要性能来源；
2. 减少显存读写非常关键；
3. 单独基础算子优化空间有限；
4. 自动搜索仍需更好的性能模型；
5. Lean4 + TileLang 的组合有潜力把 AI kernel 优化变得更加自动化和可靠。

最简洁的总结是：

> Lean4 让系统知道“哪些计算变换是对的”，TileLang 让系统把这些计算“在 GPU 上跑得足够快”。

参考资料

[1] Lean4-TileLang Tensor Program Superoptimizer [WIP]: *https://leloykun.github.io/ponder/lean4-tilelang/*

[2] Lean4-TileLang Tensor Program Superoptimizer: *https://leloykun.github.io/ponder/lean4-tilelang/*

[3] Lean4-TileLang Tensor Program Superoptimizer: *https://leloykun.github.io/ponder/lean4-tilelang/*
