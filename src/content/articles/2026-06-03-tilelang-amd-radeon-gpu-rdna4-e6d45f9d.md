**投稿/项目文章寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：TileLang、**AMD Radeon**、:mdnice-bold-italic[RDNA架构]、**GPU性能优化**、ROCm

> 本文依托TileLang编程语言，选取RX 7900 XTX、Radeon AI PRO R9700、Strix Halo三款覆盖RDNA3/3.5/RDNA4的AMD硬件，完成10种主流GPU计算内核的落地与深度调优。

研发过程先后定位threads参数限制、warpSize硬编码、exp2向量化编译异常四类易静默出错的编译器缺陷，通过PR#2313[1]完成源码修复，解除BM取值、BLOCK_M等参数隐性约束。

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/01.webp)

- **用 TileLang 在 AMD Radeon 上榨干 GPU 性能**
- 作者：Jason Zhang，https://zhuanlan.zhihu.com/p/2045169991885096417
- 测试代码：https://github.com/zhangnju/tilelang_notebooks
- 1.2 万字，阅读 50 分钟，**播客 35 分钟**

::: related-title
**相关推荐**
:::

::: related-links
- **[从形式化推理到 GPU 加速！Lean4 + TileLang 张量程序超优化器，让 Attention 快 4.08 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901134&idx=1&sn=f689cf4d266599f5873320f98ae75d4b&scene=21#wechat_redirect)**
- **[无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**

:::

> 针对iGPU低统一带宽痛点，落地BN=1访存优化、2-pass Online注意力等专属方案；独立显卡则以高并行块尺寸提升硬件利用率。

为保证评测公允，修正自动微分、循环卷积等失真基准，规避基准虚高加速比问题。:mdnice-bold-italic[实测RDNA4显卡GEMM峰值118.5TFLOPS对标rocBLAS]，反量化算子93TFLOPS，:mdnice-bold-italic[iGPU场景部分算子相较PyTorch最高提速193%].，**文章同步总结出适配三代RDNA的内核参数调优规律。**

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/02.webp)

::: image-caption
三款AMD GPU上TileLang相对PyTorch的算子加速倍率，黑色虚线为1倍基准线，蓝、红、绿柱分别对应gfx1100独显、gfx1151核显、gfx1201专业卡。多数算子TileLang实现性能优于PyTorch，gfx1151在反向传播Bwd bwd达到全图最高2.9倍提速，gfx1201的Flash Attn提速2.4倍；Outer Add算子里gfx1100与gfx1201同为2.2倍、Mul+RELU的gfx1100达2.0倍。仅少量GEMM等场景个别硬件略低于基准，整体iGPU优化收益普遍突出，得益于BN=1连续访存等定制优化，独显与专业卡在注意力、逐元素算子上同样收获明显加速。
:::

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/03.webp)

::: image-caption
三款AMD GPU下TileLang相对Triton的算子加速倍数，黑色虚线为1倍性能基准，蓝、红、绿色柱依次代表gfx1100独显、gfx1151集成iGPU、gfx1201专业卡，测试前已针对各硬件单独调优Triton参数。整体上TileLang多数算子性能优于Triton，gfx1151在反量化Dequant算子取得2.4倍全图表最高加速，Copy算子里iGPU与R9700同为2.2倍提速；反向传播、Softmax、GEMM等算子同样具备可观增益，仅Reduce等个别算子小幅落后基准，iGPU凭借访存定制优化在各类算子中加速表现尤为突出。
:::

> 本文完整记录调优过程与踩坑细节，附核心代码与实测数据。

## 本文目录

- 一、背景：为什么要在 AMD GPU 上做这件事
  - 三款 GPU 实测规格
- 二、先踩一遍 TileLang 的“静默陷阱”
  - 2.1 `T.Kernel(threads=...)` 必须是 Python 字面量，不能是变量
  - 2.2 `T.reduce_sum` 结果错误 + BM 上限：同一个 warpSize bug
  - 为什么 BM 小的时候结果是对的？
  - 修复后验证
  - 2.3 gfx1151 上 `T.exp2` 向量化的 BLOCK_M 上限问题
  - 2.4 三款 GPU 都用 WMMA 而非 MFMA
- 三、iGPU 的内存哲学：慢带宽倒逼精细设计
  - 3.1 核心优化：BN=1，所有线程写同一行
  - 3.2 2-pass Online Attention：利用 L2 缓存减少 DRAM 读写
- 四、公平对比：修正 PyTorch、Triton 错误基准
  - 4.1 autograd backward ≠ PyTorch 真实计算能力
  - 4.2 Conv1D：Python for 循环是假基准
  - 4.3 逐元素算子：torch.compile Inductor 算子融合，一行代码省掉一次 DRAM 往返
  - 4.4 iGPU 小卷积：unfold+matmul 绕过 MIOpen 启动开销
  - 4.5 Triton 的块尺寸需要针对每款 GPU 单独调整
- 五、各 GPU 的调优配置规律
  - 5.1 BLOCK_N 与 CU 数量负相关（访存密集算子）
  - 5.2 WMMA 参数：寄存器压力与 occupancy 的权衡
  - 5.3 小问题规模：减少 threads 反而更快
- 六、最终性能实测数据
  - gfx1100（RX 7900 XTX，RDNA3，96 CU，规格带宽 ~0.96 TB/s）
  - gfx1151（Radeon 8060S / Strix Halo，RDNA3.5 iGPU，40 CU，规格带宽 ~0.26 TB/s）
  - gfx1201（R9700 RDNA4 64CU，规格带宽 ~0.64TB/s）
- 七、调优系统性结论
  - 7.1 线程利用率最优原则：利用率要刚好 100%，多余的线程和闲置的线程都是浪费
  - 7.2 带宽分析采用硬件规格峰值而非实测值，防止误导分析
  - 7.3 iGPU 的”1309% 加速比”是怎么来的
  - 7.4 PyTorch 的默认写法往往低估了 PyTorch 的能力
- 八、全文总结
  - TileLang 框架层面的关键规则
  - iGPU（gfx1151）特化优化
  - 基准公平性
  - 仍然存在的差距

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/04.webp)

## 一、背景：为什么要在 AMD GPU 上做这件事

> AI 推理和训练领域，CUDA/NVIDIA 生态固然成熟，但 AMD GPU 的性价比优势越来越明显。特别是 RDNA 系列——从 RDNA3 独立显卡 RX 7900 XTX，到专业 AI 加速卡 R9700，再到 Strix Halo 这样的 iGPU 旗舰，硬件跨度很大，适用场景也各不相同。

TileLang[2] 是 Tile-AI 开发的面向 GPU 内核的 Python DSL，目标是**让写 kernel 像写 Python 一样简单，但性能接近手写 CUDA**。它可以直接在 ROCm 上为 AMD GPU 生成高效 kernel，语法比 Triton 更高层，比 CUDA 更易读。

项目将 10 个经典 GPU 内核（向量加法、外积、Softmax、Flash Attention、GEMM、Conv1D、反量化矩阵乘等）用 TileLang 实现，在三款硬件上系统调优。目标只有一个：**在每个平台上，TileLang 性能必须好于 PyTorch 和 Triton，或者清楚知道性能落后的原因。**

### 三款 GPU 实测规格

> 注：AMD ROCm 的 `multi_processor_count` 是 WGP 数，1 WGP = 2 CU，本文所有 CU 数均为实际值。

| GPU | 架构 | CU 数 | 显存 | 规格峰值带宽 |
| --- | --- | --- | --- | --- |
| RX 7900 XTX | gfx1100 (RDNA3) | 96 CU | 24 GB GDDR6 | ~0.96 TB/s |
| Radeon AI PRO R9700 | gfx1201 (RDNA4) | 64 CU | 32 GB GDDR6 | ~0.64 TB/s |
| Radeon 8060S (Strix Halo) | gfx1151 (RDNA3.5 iGPU) | 40 CU | 128 GB 统一内存 | ~0.26 TB/s |

**测试环境**：TileLang 0.1.10+ rocm · PyTorch 2.12.0+rocm7.2 · Triton 3.7.0

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/05.webp)

::: image-caption
对比三款不同架构AMD显卡的峰值显存带宽与CU计算单元数量，左纵轴代表峰值带宽（TB/s）、右纵轴标注CU数量；RX7900 XTX(gfx1100/RDNA3)拥有0.96TB/s带宽、96CU，两项参数在三者中最高；Radeon AI PRO R9700(gfx1201/RDNA4)规格居中，带宽0.64TB/s、64CU；Radeon 8060S(gfx1151/RDNA3.5集成iGPU)带宽仅0.26TB/s、40CU，硬件规格最低，直观体现三款产品定位差异：RX7900XTX为高性能游戏独显、R9700是中端专业加速卡、8060S是低带宽统一内存架构的移动端核显，三者带宽与CU数量呈同步递减规律。
:::

## 二、先踩一遍 TileLang 的“静默陷阱”

在开始性能调优之前，我们先被 TileLang 自身的几个约束坑了——而且都是**运行不报错、结果错误的** 那种，查起来非常费劲。

### 2.1 `T.Kernel(threads=...)` 必须是 Python 字面量，不能是变量

#### T.Kernel 和 threads 是什么

> TileLang 用 T.Kernel(grid, threads=N) 来声明一个 GPU kernel 的执行配置，类似 CUDA 的 <<<grid, block>>>：

```
@tilelang.jit(...)def my_kernel(A, BLOCK_N: int):    with T.Kernel(N // BLOCK_N, threads=256) as pid:        # 这里的代码在 GPU 上并行执行        # grid = N // BLOCK_N 个 block，每个 block 256 条线程        # pid = 当前 block 的编号（0, 1, 2, ...）        ...
```

- `grid`：Block 数量，运行时可变，支持变量/表达式；
- `threads`：单 Block 线程数，编译期固定，决定寄存器、Warp、共享内存布局，**只允许字面常量**。

threads 的值直接影响 GPU kernel 的寄存器分配、warp 调度和共享内存布局——这些在编译期就必须固定下来，无法在运行时动态调整。RDNA 架构（gfx1100/gfx1201/gfx1151）的最优线程数因 GPU 型号而异，例如 gfx1201 有 64 个 CU 且性能特性略有不同，直觉上想在运行时根据 arch 动态选择线程数——这就踩了下面这个坑。

> 这个问题折腾了我们大半天。表现是：编译通过，结果正确，但性能始终达不到预期——同样的算法在 Triton 上跑 0.046ms，TileLang 却要 0.18ms，慢了将近 4 倍，怎么排查都找不到原因。

最后排查发现，问题出在这里：

```
# ❌ threads 使用运行时变量arch = torch.cuda.get_device_properties(0).gcnArchNamethreads = 128 if arch.startswith("gfx1201") else 256@tilelang.jit(...)def my_kernel(A, BLOCK_N: int):    with T.Kernel(N // BLOCK_N, threads=threads) as pid:        ...
```

TileLang 的 jit 在编译时把 T.Kernel 里的 threads 当成编译期常量来决定寄存器分配、warp 调度等。如果传入的是运行时变量，编译器会忽略它，用默认值或者编译出错误的 kernel。更糟糕的是，这不会抛出任何异常。

正确方案：按架构拆分独立 Kernel，threads 硬编码字面量：

```
# ✅ 每个 arch 有独立的 kernel，threads 是字面量@tilelang.jit(...)def kernel_gfx1201(A, BLOCK_N: int):    with T.Kernel(N // BLOCK_N, threads=128) as pid:  # 字面量 128        ...@tilelang.jit(...)def kernel_others(A, BLOCK_N: int):    with T.Kernel(N // BLOCK_N, threads=256) as pid:  # 字面量 256        ...# 根据 arch 选择编译哪个if arch.startswith("gfx1201"):    k = kernel_gfx1201.compile(...)else:    k = kernel_others.compile(...)
```

这个约束的根本原因是：TileLang 需要在 Python 编译时（不是 GPU 运行时）确定线程数，以便生成正确的 HIP kernel。变量在 Python 编译时不是常量，所以无法参与这个过程。

### 2.2 `T.reduce_sum` 结果错误 + BM 上限：同一个 warpSize bug

#### T.reduce_sum 和 BM 是什么

> `T.reduce_sum 是 TileLang 对跨线程归约求和的封装。GPU kernel 里，数据分散在各个线程的寄存器（fragment）中，T.reduce_sum 负责把这些寄存器里的值汇聚成一个标量——内部通过 warp 内的 shuffle 指令和 warp 间的共享内存来实现：

```
# 典型用法：对矩阵 A 的每一行求和，结果存入 Bwith T.Kernel(N // BN, threads=128) as pid:    Al  = T.alloc_fragment((BN, BM), T.float32)  # 寄存器 fragment    acc = T.alloc_fragment((BN,),    T.float32)  # 累加器    T.clear(acc)    for m_blk in T.Serial(M // BM):              # 串行遍历列方向的 tile        T.copy(A[pid * BN, m_blk * BM], Al)      # 从全局内存加载一个 tile        T.reduce_sum(Al, acc, dim=1, clear=False) # 对 BM 个元素求和，累加到 acc    T.copy(acc, B[pid * BN])
```

BM（Block M）是每次归约的 tile 大小，即每次串行迭代中处理的列数。BM 越大，串行迭代次数越少（M / BM 次），性能越好；但 BM 越大，参与每次归约的线程数越多，对 AllReduce 内部的实现要求也越高。

AllReduce 的内部实现分两段：

- offset ≥ warpSize：当前参与归约的线程跨越了多个 warp，需要借助共享内存（LDS）在 warp 之间传递数据
- offset < warpSize：所有参与线程在同一个 warp 内，直接用寄存器 shuffle（shfl_xor）完成，无需同步

::: block-1
BM 越大 → 参与归约的线程数越多 → 越有可能触发跨 warp 的共享内存路径 → 对 warpSize 的判断越依赖。这正是 warpSize 写错时 BM 大了才出错的原因。
:::

---

这个问题折腾了我们最久，而且我们先后观察到两种截然不同的症状，花了很长时间才意识到它们是同一个根因。

- **症状 A（最先发现）**：BM 超出某个阈值时，T.reduce_sum 结果静默出错，不报任何异常——correctness 测试显示 max_diff ≈ 490。正确答案是几千量级，计算结果也是几千，差了 490，不是精度误差，是计算逻辑崩了。
- **症状 B（后来发现）**：gfx1201/gfx1151 上，T.copy 填充 fragment 时 BM 只能 ≤ 64，换成 T.Parallel 填充才能用到 BM=128。两种填充方式”上限不同”，让我们误以为是布局（layout）差异导致的。

> 这两个症状看起来毫无关联，**实际上根因只有一个：constexpr int warpSize = 64**

`src/tl_templates/hip/reduce.h` 里有一行：

```
// ❌ TileLang 旧版本（已修复）constexpr int warpSize = 64;   // CDNA（MI300/MI350）的值！RDNA 是 32！
```

RDNA 系列（gfx1100/gfx1201/gfx1151）的 warp size 是 32，不是 64。AllReduce 模板是一个递归展开的折半归约树：

```
AllReduce<threads=128, scale=1> 在 RDNA（warpSize=32）上的展开：  旧版（warpSize=64）              新版（warpSize=32）  offset=64: 64≥64 → 共享内存 ✓   offset=64: 64≥32 → 共享内存 ✓  offset=32: 32<64 → shfl_xor(32) ❌  offset=32: 32≥32 → 共享内存 ✓  offset=16: 16<64 → shfl_xor(16) ✓  offset=16: 16<32 → shfl_xor(16) ✓  ...                              ...
```

关键差异在 `offset=32` 这一步：旧版走 `shfl_xor(x, 32)`，在 32-lane 的 RDNA warp 上，`lane XOR 32` = lane 32~63，这些 lane 不存在，读到的是未初始化的寄存器，产生垃圾值。

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/06.webp)

::: image-caption
该图表展示PR#2313针对RDNA架构Wave32的AllReduce归约warpSize缺陷修复前后对比，左侧是修复前硬编码warpSize=64的旧逻辑，右侧为调用`__builtin_amdgcn_wavefrontsize()`动态获取硬件波前尺寸的新方案；绿色代表寄存器内shfl_xor波内洗牌运算、蓝色代表LDS共享内存路径，红色标注错误项，修复前offset=32时沿用64线程Warp的shfl_xor(32)，在RDNA单波32线程下出现lane索引越界报错，修复后该偏移场景切换为LDS共享内存实现，全档位offset（1~64）运算均可正常执行，彻底消除因CDNA与RDNA波前尺寸差异带来的归约计算出错问题。
:::

### 为什么 BM 小的时候结果是对的？

> BM 决定了参与归约的线程数（reducing_threads）。当 BM 足够小，使得 reducing_threads ≤ 32（全部在一个 wave 内），AllReduce 的展开不会到达 offset=32 这一层，全程走 shfl_xor(16/8/4/2/1)，结果正确。一旦 BM 增大使得 reducing_threads > 32，就必须经过 offset=32 这一步，立刻出错。

这就解释了”BM 上限”现象：不是 T.copy 和 T.Parallel 的布局差异，而是两者在特定 BM 下触发了不同的 reducing_threads 值，某些配置恰好把 reducing_threads 压在 32 以下从而侥幸正确。

PR #2313 的修复：[AMD][RDNA4] Fix gfx12 (RDNA 4 / Wave32) related CI issues，详见：https://github.com/tile-ai/tilelang/pull/2313

```
// ✅ 修复后：运行时查询实际 warp sizeconst int warpSize = __builtin_amdgcn_wavefrontsize();  // RDNA 返回 32，CDNA 返回 64
```

`offset=32` 时 `32 ≥ 32`，改走共享内存路径，正确。同步修复还覆盖了：

- `scan.h`：添加 wave32 支持（`threads==32` 断言 + `run_auto()` 自动分发）
- `finalize_reducer.cc`：`WarpSize()` 从 `return 64` 改为 `return TargetGetWarpSize(target)`
- `arch/rdna.py`：gfx12（RDNA4）WMMA 指令注册
- `wmma_macro_generator.py`：k_pack 索引修复

我们将这些改动移植到 Tilelang最新代码，并重新编译（`USE_ROCM=1 USE_CUDA=0 pip install -e . -v`）。

### 修复后验证

> 实测（gfx1100/RX 7900 XTX，PR #2313 已应用，TileLang 0.1.10+rocm）：

```
Method        BN   BM  Correct      ms----------------------------------------T.copy         2   64      ✅   0.046msT.copy         2  128      ✅   0.031ms   ← 修复前此处出错（结果错误）T.copy         2  256      ✅   0.019msT.copy         2  512      ✅   0.020msT.Parallel     2  128      ✅   0.031msT.Parallel     2  256      ✅   0.019msT.Parallel     2  512      ✅   0.020ms
```

::: block-1
修复结论：PR #2313[3] 合入后，`T.copy`与`T.Parallel`行为统一，彻底解除 BM 取值上限。

教训：改参数后必须跑 correctness 测试。这类 warpSize 错误在编译期完全无提示，只有静默的错误结果。
:::

### 2.3 gfx1151 上 `T.exp2` 向量化的 BLOCK_M 上限问题

#### T.exp2 和 T.Parallel 是什么

> `T.exp2(x)` 是 TileLang 的内置 GPU 数学函数，计算以 2 为底的指数 $2^x$。在 Softmax 和 Flash Attention 中，用它替代自然指数 $e^x$：

```
log2_e = 1.44269504   # log₂(e)# e^x = 2^(x × log₂e)，用 exp2 实现：速度更快，精度相同y = T.exp2(log2_e * x)
```

`T.Parallel(BLOCK_M)` 用于声明并行循环，loop body 中的每次迭代由不同线程同时执行，类似 CUDA 的线程并行：整个 block 的 `threads` 条线程瓜分 `BLOCK_M` 次迭代任务。

当 `T.exp2` 出现在 `T.Parallel` 循环内时，编译器会尝试对其进行向量化——将多个元素的 exp2 合并成一条 SIMD 指令（如 `v_exp_f32` 的向量版本）。这个向量化的宽度（每条指令处理几个元素）取决于每个线程分到的元素数量，也就是 `BLOCK_M / threads`。

#### 三款 GPU 的 exp2 向量化宽度对比

> 每个线程分到的元素数 = `(BLOCK_B × BLOCK_M) / threads`，这就是编译器需要向量化的宽度。三款 GPU 对这个宽度的支持程度差异很大：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/07.webp)

::: image-caption
这份表格对比三款AMD RDNA架构GPU的exp2向量化参数，gfx1100、gfx1201分别对应RDNA3、RDNA4独显，二者exp2向量化宽度上限均≥8，采用BLOCK_M=4096、threads=512的配置，单线程分配8个运算元素；gfx1151为RDNA3.5集成核显，经PR#2313修复后向量化宽度上限提升至≥4，选用BLOCK_M=1024、threads=256，每线程仅处理4个元素。差异化的硬件指令能力，决定了三者Softmax算子的编译参数区别，修复后iGPU解除BLOCK_M≤256的编译枷锁，得以放大分块尺寸优化运行效率。
:::

- **gfx1100/gfx1201 的情况：AMD GPU 为 exp/log/sin 等数学函数提供专门的硬件指令（如 v_exp_f32、v_log_f32），由 Shader Core 内的数学流水线执行**。TileLang 对向量化循环进行展开，生成多条连续的 v_exp_f32 指令；每条指令在一个 Wavefront 的所有 lane 上同时执行（Wave32 = 32 个 exp2 并行），整体吞吐量极高。
- :mdnice-bold-italic[gfx1151 的情况：gfx1151 硬件本身完整支持 v_exp_f32 等数学指令，限制来自 TileLang 编译器的 HIP 代码生成层]。当 T.exp2 出现在 T.Parallel 循环内，若编译器推导出向量化数据类型（vector dtype），会生成向量形式的 tirx.exp2 调用，而 PR #2313 之前 HIP codegen 只注册了标量版本的 lowering，找不到对应处理，报错：

```
InternalError: Unresolved call ir.Op(name="tirx.exp2")
```

这是 PR #2313[4] 之前的编译器实现缺失，而非 gfx1151 硬件能力受限。规避方式是将每线程元素数压到 1，使编译器不触发 vector dtype 路径，经验性约束为：

`(BLOCK_B × BLOCK_M) / threads ≤ 1`，即 `BLOCK_B × BLOCK_M ≤ threads`

实测验证（threads=256，gfx1151）：

```
BLOCK_B=1, BLOCK_M=256  → 256/256=1 元素/线程 → ✅ 编译通过（修复前后均可）BLOCK_B=1, BLOCK_M=512  → 512/256=2 元素/线程 → ❌ 修复前：Unresolved call tirx.exp2  ✅ 修复后（[PR #2313](https://github.com/tile-ai/tilelang/pull/2313 "PR #2313")）：正常编译BLOCK_B=8, BLOCK_M=64   → 512/256=2 元素/线程 → ❌ 修复前：Unresolved call tirx.exp2  ✅ 修复后：正常编译BLOCK_B=4, BLOCK_M=64   → 256/256=1 元素/线程 → ✅ 编译通过（修复前后均可）
```

PR #2313 修复前，gfx1151 上 `T.exp2` 在 `T.Parallel` 内有隐性上限：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/08.webp)

::: image-caption
该表格罗列三款不同RDNA架构AMD显卡的exp2向量化硬件规格与内核配置，gfx1100、gfx1201分属RDNA3、RDNA4独立显卡，硬件exp2向量化宽度上限均可达到8及以上，工程上采用BLOCK_M=4096、threads=512的参数，单线程负责8个数据元素；gfx1151是RDNA3.5架构集成核显，在PR#2313代码修复后exp2向量化上限提升至4，配套BLOCK_M=1024、threads=256，每线程处理4个元素。硬件原生向量运算能力的差距，造成三者Softmax算子分块配置分化，本次修复解除了iGPU原本BLOCK_M不得超256的编译约束，有效优化算子运行效率。
:::

超出这个限制，编译器报 InternalError: Unresolved call tirx.exp2，而不是静默出错（比 2.2 节的 warpSize bug 好一点——至少会报错）。

#### PR #2313 的修复

PR #2313 在 `src/backend/rocm/codegen/intrin_rule_hip.cc` 中为 HIPMath 结构体添加了向量类型的自动降级分发：

```
// ✅ 修复：向量类型（如 float32x4）自动降级到标量元素类型if (!t.is_scalar()) {  return (*this)(t.element_of(), name);  // 递归调用标量路径}
```

这让 T.Parallel(BLOCK_M) 内的 T.exp2 在 gfx1151 上即使每线程处理多个元素（vector dtype），也能正确 lower 到 exp2f。gfx1151 上 BLOCK_M 不再受 256 上限约束：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/09.webp)

实测（gfx1151，N=2048 行，M=8192 列，float32）：PR #2313 让 BLOCK_M=1024 能够编译通过，从而使用更大的 tile，减少 global memory round-trip 次数，Softmax Online 延迟从 2.83ms 降至 2.56ms（≈ +10%），最终与 PyTorch/MIOpen 持平（2.56ms vs 2.55ms）。注意性能提升来自 tile 更大、迭代轮数减少，而非 exp2 指令本身加速。

解决方法：给 gfx1151 写独立 kernel 并使用 BLOCK_M=1024：

```
# Softmax Online：gfx1151 修复后用 BLOCK_M=1024（[PR #2313](https://github.com/tile-ai/tilelang/pull/2313 "PR #2313") 后不再限制 256）if arch.startswith("gfx1151"):    BM_online = 1024   # 修复前受限于 256；intrin_rule_hip.cc 修复后无此限制    k_online = tl_softmax_online_gfx1151.compile(N=N, M=M, BLOCK_M=1024)else:    BM_online = 4096    k_online = tl_softmax_online_gfx11xx.compile(N=N, M=M, BLOCK_M=4096)# Flash Attention：gfx1151 2-pass online（BB=1, BS=256）if arch.startswith("gfx1151"):    k = tl_attn_gfx1151.compile(B=B_sz, S=S, BLOCK_B=1, BLOCK_S=256)
```

gfx1151 修复后用 BLOCK_M=1024，Softmax Online 实测 2.56ms，与 PyTorch 2.55ms 持平（≈ 0%），比修复前 BM=256 时的 2.83ms 快约 10%。差距来自 iGPU 统一内存带宽（规格峰值 0.26 TB/s），与 BLOCK_M 配置无关。Flash Attention 的 BB=1, BS=256 配置不变。

### 2.4 三款 GPU 都用 WMMA 而非 MFMA

> 这是一个容易混淆的概念。AMD GPU 有两种矩阵乘法指令路径：

- **WMMA（Wave Matrix Multiply-Accumulate）**：RDNA 系列，指令 v_wmma_f32_16x16x16_f16，**warp_size = 32**
- **MFMA（Matrix Fused Multiply-Add）**：CDNA 系列 GPU（MI300/MI350 系列），**warp_size = 64**

我们的三款测试 GPU（gfx1100、gfx1201、gfx1151）全部是 RDNA，用 WMMA。

TileLang 提供两条 AMD 矩阵指令路径：

- `WMMAIntrinEmitter` → RDNA3/RDNA4（WMMA）✅
- `MatrixCoreIntrinEmitter` → CDNA（MFMA）✅

RDNA GPU（gfx1100/gfx1151/gfx1201）应使用 `WMMAIntrinEmitter`。在 RDNA 上误用 `MatrixCoreIntrinEmitter` 通常会直接编译报错，因为 MFMA 指令仅 CDNA 支持。

另一个细节：WMMA 要求 B operand 以转置布局提供，在 `WMMAIntrinEmitter` 中通过 `b_transposed=True` 声明。这并不意味着一定要对输入做物理转置（`B.T.contiguous()`）——转置布局也可以在 kernel 数据搬运阶段（如写入 LDS 时）构造出来。下面的示例采用预先物理转置的写法：

```
# WMMA 的标准用法B_T = B.T.contiguous()  # (K,N) → (N,K)，提前转置wmma_e = WMMAIntrinEmitter(    a_dtype=T.float16, b_dtype=T.float16, accum_dtype=T.float32,    a_transposed=False,    b_transposed=True,   # B operand 以转置布局解释（此处由 B.T.contiguous() 提前准备）    block_row_warps=brw, block_col_warps=bcw,    warp_row_tiles=wrt, warp_col_tiles=wct,    chunk=chunk, k_pack=1, target=target,)
```

## 三、iGPU 的内存哲学：慢带宽倒逼精细设计

> gfx1151（Strix Halo）有 40 个 CU，硬件规格并不算弱，但 iGPU 的根本限制是内存带宽——128 GB 统一内存与 CPU 共享，规格峰值带宽约 **0.26 TB/s（256-bit LPDDR5X）**，相比 gfx1100 的 0.96 TB/s 差了约 3.7 倍。

在这种硬件上写 kernel，必须把内存访问当成最稀缺的资源来精打细算。

### 3.1 核心优化：BN=1，所有线程写同一行

> 我们的 outer_vec_add kernel（计算 C[i,j] = A[i] + B[j]）在 gfx1100 上用 BN=256, BM=64, threads=256 效果很好，带宽利用率很高，比 PyTorch 快 121%。

把同样的配置搬到 gfx1151 上，结果是 **1.377ms，比 PyTorch 慢了 3.5 倍（PyTorch 0.302ms）。这是怎么回事？**

::: block-1
根本原因在于 工作集分散：`BN=256, BM=64` 时，一个 block 同时覆盖 256 行 × 64 列的输出区域（32KB fp16），这 256 行的地址在 row-major 内存中彼此间距 `M × sizeof(fp16)`（即数千字节），访问模式对 iGPU 的低带宽内存非常不友好。

此外，向量 `A[i]` 本应被 `BM=64` 次复用，但 256 行同时活跃时 A 的 cache locality 也随之下降。
:::

gfx1151 规格带宽仅约 0.26 TB/s（gfx1100 的 1/3），低带宽使得低效的访问模式代价被大幅放大——这是性能差距的核心原因，而非延迟本身。

解决方案：让 BN=1，整个 block 只处理 1 行输出，所有写操作连续：

```
# ❌ gfx1151 上性能差：BN=256，一个 block 同时覆盖 256 行，工作集分散with T.Kernel(N // 256, M // 64, threads=256) as (pid_n, pid_m):    for i, j in T.Parallel(256, 64):        C[pid_n * 256 + i, pid_m * 64 + j] = A[pid_n * 256 + i] + B[pid_m * 64 + j]# ✅ gfx1151 上的正确姿势：BN=1，一个 block 处理同一行的连续列with T.Kernel(N, M // 4096, threads=256) as (row, pid_m):    for j in T.Parallel(4096):        C[row, pid_m * 4096 + j] = A[row] + B[pid_m * 4096 + j]# 所有线程写同一行的连续地址，访问完全 coalesced
```

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/10.webp)

::: image-caption
内存访问模式：BN=256 vs BN=1。该图对比iGPU两种BN参数下的内存访问模式，左侧BN=256为红色区块，数据跨行分散分布，属于跨步离散访存，访存步长远超缓存行长度，频繁失效拉高缓存压力、浪费带宽，是iGPU性能短板；右侧BN=1为绿色区块，单个Block仅读取单行的连续列数据，实现合并连续访存，数据顺序排布贴合硬件缓存读取逻辑，能够高效利用缓存、大幅减少DRAM读写开销。这也是gfx1151核显固定BN=1优化算子的核心原理，依靠规整访存大幅提速向量运算与反向传播。
:::

**gfx1151 优化前后性能对比**

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/11.webp)

同样的原则也适用于 Triton：把 backward kernel 的 BN=64, BM=64 改成 BN=1, BM=2048：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/12.webp)

::: block-1
**为什么同样的改动在 gfx1100 上反而变慢？**

gfx1100 拥有 96 CU、960 GB/s GDDR6 以及大容量 Infinity Cache，BN=256 能让更多 CU 同时工作、保持高 occupancy。改成 BN=1 后并行度下降，可能出现 CU 利用率不足，损失大于访问局部性的收益。gfx1151 带宽受限，局部性收益更显著，两者权衡结果相反。**这是严格的架构特化优化**，必须用 arch 检测控制。
:::

### 3.2 2-pass Online Attention：利用 L2 缓存减少 DRAM 读写

> 2-pass Online 算法的核心优势是省去了将中间结果 exp(QK-max) 写回 DRAM 并再次读取这一步：

**3-pass 实现**：三次显存读写，中间 Attention Score 写入 DRAM；

- Pass 1：读 Q, 读 K → 求每行最大值
- Pass 2：读 Q, 读 K → 写 exp(QK-max) 到 DRAM
- Pass 3：读 exp(QK-max), 读 V → 写 O

**2-pass Online**：Q、K 两次读取，第二次命中 L2 缓存，省去中间张量落地 DRAM。

- Pass 1：读 Q, 读 K → 在线维护 (max, sum)
- Pass 2：重新读取 Q, K（往往仍驻留 L2），读 V → 写 O

Q,K 虽被读了两次，但 Pass 2 的第二次读取往往可以命中 L2，避免了写入再读取中间 attention score 的大额 DRAM 开销。

gfx1151 的可用内存带宽远低于独立显卡（约 0.26 TB/s vs 0.96 TB/s），因此减少 DRAM 往返带来的收益更加明显，这个差异在 gfx1151 上非常显著：

- 3-pass：~0.65ms（旧配置 BB=8, BS=128）
- 2-pass online（BB=1, BS=256）：0.411ms（提升约 37%，反超 PyTorch 0.576ms 和 Triton 0.530ms，最终 +29% vs Triton）

在 gfx1100/gfx1201 上，2-pass 需要额外维护在线 (max, sum) 状态，fragment 数量增加，带来寄存器压力上升和 occupancy 下降。在高带宽的独立显卡上，这一 occupancy 损失超过了减少 DRAM 流量的收益，因此仍用 3-pass。只在 gfx1151 上启用 2-pass：

```
if arch.startswith("gfx1151"):    k = tl_scalar_attn_2pass.compile(...)   # 2-pass：利用 L2 缓存else:    k = tl_scalar_attn_3pass.compile(...)   # 3-pass：寄存器压力小
```

## 四、公平对比：修正 PyTorch、Triton 错误基准

优化 TileLang 只完成了一半工作。如果 PyTorch 基准本身有问题，加速比数字没有意义。我们在对比过程中发现了几个经典的”基准陷阱”，修复之后有些数字变化很大。

### 4.1 autograd backward ≠ PyTorch 真实计算能力

> 对 broadcast mul + ReLU 的反向传播，最直觉的 PyTorch 写法用 autograd：

```
# ❌ 这个测的是"autograd 框架的调度开销"，不是"GPU 计算速度"def ref_bwd_wrong(A, B, dC):    A_ = A.clone().float().requires_grad_(True)    torch.relu(A_ * B.float()).backward(dC.float())    return A_.grad.half()
```

在 gfx1151 上这跑出 **8.6ms**。但 autograd 在做很多 TileLang 根本不需要做的事情：

- 在 CPU 上构建计算图（overhead）
- float16 → float32 类型转换（两次内存来回）
- 通过计算图执行反向传播（额外调度逻辑）
- float32 → float16 转换（再次内存来回）

涉及多个 GPU kernel 启动（具体数量取决于 autograd dispatch 路径，通常明显多于手写融合 kernel），而实际需要的计算 `dA = dC × B × 1[A×B > 0]` 本身极为简单。

公平的对比：

```
# ✅ 与 Triton/TileLang 做完全相同的数学运算# 用 in-place 操作减少临时张量分配（iGPU 上 3次 kernel 启动 vs autograd 的 5~6次）def ref_bwd_fair(A, B, dC):    tmp = A * B          # tmp 只计算一次    return dC * B * (tmp > 0)
```

修正后行性能（gfx1151）：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/13.webp)

注意：修正基准后，TileLang 的加速比从 +105% 变成了 +193%，似乎”更夸张”了——实际上是 PyTorch 基准变合理了（2.6ms 比 8.6ms 更接近真实计算时间），TileLang 的 0.88ms 依然通过 BN=1 优化获得。

### 4.2 Conv1D：Python for 循环是假基准

多通道 conv 的原始实现：

```
# ❌ 32 次独立 GPU kernel 调用，90% 时间在 kernel 启动和调度def ref_conv1d_mc_wrong(X, K):    N, L = X.shape; KL, F = K.shape; pad = (KL-1)//2    O = torch.zeros(N, L, F, dtype=X.dtype, device=X.device)    for f in range(F):      # F=32 次 GPU kernel！        O[:, :, f] = torch.conv1d(            X.unsqueeze(1), K[:, f].view(1,1,-1), padding=pad        ).squeeze(1)    return O
```

在 gfx1151 上跑出 0.658ms，而 TileLang 的 3D 并行 kernel 只需 0.005ms，加速比 131×。这个数字当然是假的——问题不在于”TileLang 有多强”，而在于”for 循环跑了 32 次 GPU kernel”。

正确实现，一次处理所有 F 个输出通道：

```
# ✅ 单次 GPU kernel 调用处理所有通道def ref_conv1d_mc_fair(X, K):    pad = (K.shape[0] - 1) // 2    weight = K.T.contiguous().unsqueeze(1)   # [F, 1, KL]    out = torch.conv1d(X.unsqueeze(1), weight, padding=pad)    return out.permute(0, 2, 1).contiguous()
```

修正后（gfx1151）：PyTorch 0.658ms → 0.018ms，加速比 131× → 3.6×（+260%）。数字变小了很多，但这才是真实的差距。原始实现测的是 kernel 启动调度效率，而非计算内核本身的质量；修正后的对比才反映真实的 kernel fusion 差异。

### 4.3 逐元素算子：torch.compile Inductor 算子融合，一行代码省掉一次 DRAM 往返

> torch.relu(A * B) 看起来简单，实际上在 PyTorch eager mode 下是两个独立的 GPU kernel：

```
Kernel 1：A * B  读 A（8 MB）+ 读 B（8 MB）→ 写 temp（8 MB）= 24 MB DRAM 访问Kernel 2：relu(temp)  读 temp（8 MB）→ 写 result（8 MB）= 16 MB DRAM 访问合计：40 MB 显存流量（未融合）融合成一个 kernel 只需读一遍 A,B、写一遍 result，显存流量降至 24 MB。`torch.compile` 里的 Inductor 后端可以自动做这个融合（实际融合效果取决于后端 fusion 决策）：```python# gfx1151 专用（compile 有约 100ms 启动开销，推理场景均摊后划算）if arch.startswith("gfx1151"):    _fused = torch.compile(lambda a, b: torch.relu(a * b))    _fused(A, B)  # 第一次调用触发 JIT 编译，后续直接使用融合 kernel    def ref_mul_relu(A, B):        return _fused(A, B)else:    def ref_mul_relu(A, B):        return torch.relu(A * B)
```

结果：gfx1151 上 PyTorch Mul+ReLU 从 **0.142ms → 0.037ms，提升 281%，** 与 TileLang 和 Triton 持平。

### 4.4 iGPU 小卷积：unfold+matmul 绕过 MIOpen 启动开销

> 对于`N=64, L=1024, KL=7`这样的小卷积，`torch.conv1d` 走 MIOpen 路径。MIOpen 需要匹配 solver、设置 descriptor、分配工作空间——这些操作在 GDDR6 上因为内存操作快，开销可以忽略；但在 iGPU 统一内存上，每一步都拖慢了，导致 torch.conv1d 在 gfx1151 上耗时 **0.054ms**。而实际数据只有 256KB，在规格带宽 0.26 TB/s 下理论只需约 1µs——MIOpen 的启动开销放大了几十倍。

换成 `unfold + matmul` 路径，一个融合操作完成整个卷积：

```
if arch.startswith("gfx1151"):    def ref_conv1d(X, K):        pad = (K.shape[0] - 1) // 2        Xp = F.pad(X, (pad, pad))            # 零填充        Xu = Xp.unfold(1, K.shape[0], 1)    # [N, L, KL]（view，通常零拷贝）        # 单次 matmul 完成所有卷积计算，1 个 GPU kernel        return (Xu.float() @ K.float().unsqueeze(-1)).squeeze(-1).half()
```

**PyTorch 耗时 0.054ms→0.035ms，提速 54%。** 对于这类小规模负载，PyTorch 选择了调度开销较大的 MIOpen 路径；unfold+matmul 通过单次 GEMM 调用绕开了这一开销。实际差距受具体 workload 大小影响。

### 4.5 Triton 的块尺寸需要针对每款 GPU 单独调整

> 很多 Triton 示例代码的块参数是为 NVIDIA GPU 调优的，直接在 AMD GPU（尤其是 iGPU）上用会有性能损失。

**Scalar Attention 的 BLOCK_S** （B_sz=256, S=16384）：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/14.webp)

BLOCK_S=2048 在 gfx1151 上更快，因为更大的 tile 减少了串行迭代次数，降低了 serial loop 开销（iGPU 上每次迭代的调度代价更大）。代价是更高的寄存器和 LDS 压力，在 gfx1151 的 occupancy 特性下这一权衡是合算的。

**2D 内核的块形状**：outer_vec_add 和 backward 的 Triton kernel 也需要 BN=1：

```
# gfx1151：必须用 BN=1，否则跨步访问导致 cache miss 严重if arch.startswith("gfx1151"):    BN_t, BM_t = 1, 2048    # 每个 program 处理 1 行，完全 coalescedelse:    BN_t, BM_t = 64, 64     # 标准 2D tile
```

## 五、各 GPU 的调优配置规律

经过上面所有调优，**三款 GPU 的最优配置形成了清晰的规律。**

### 5.1 BLOCK_N 与 CU 数量负相关（访存密集算子）

> 对于 copy/vector_add 这类纯内存密集型 kernel，BLOCK_N 的选择与 CU 数密切相关：

问题规模：N = 262,144 个 fp16 元素（512KB 数据）

- gfx1100（96 CU）→ BLOCK_N=1024 → 256 blocks → 每 CU 约 2.7 个 block ✓
- gfx1201（64 CU）→ BLOCK_N=2048 → 128 blocks → 每 CU 约 2.0 个 block ✓
- gfx1151（40 CU）→ BLOCK_N=8192 → 32 blocks → 每 CU 约 0.8 个 block ✓

```
BLOCK_N_BEST = {    "gfx1151": 8192,   # 40 CU → 32 blocks    "gfx1201": 2048,   # 64 CU → 128 blocks}.get(arch[:7], 1024)  # gfx1100: 96 CU → 256 blocks
```

### 5.2 WMMA 参数：寄存器压力与 occupancy 的权衡

> GEMM 的 WMMA 配置核心是 wrt × wct（warp tile 大小），决定了 tile 尺寸和寄存器占用：

```
if arch.startswith("gfx1151"):    # iGPU：wrt=64 大 tile + panel=4 小 swizzle，sweep 发现比 wrt=32 快 +24%    brw, bcw, wrt, wct, chunk, panel = 2, 2, 64, 64, 64, 4    # block_M=128, block_N=128；panel=4 对 128-block grid 的 L2 命中率最优    # 结果：~28.5 TFLOPS（在测试 workload 下约为 rocBLAS 的 84%）else:    # 独立显卡：寄存器充足，用大 tile 提高计算密度    brw, bcw, wrt, wct, chunk, panel = 2, 2, 64, 64, 64, 10    # block_M=128, block_N=128（大 tile）    # 结果：gfx1100 85.7 TFLOPS，gfx1201 118.5 TFLOPS
```

**iGPU 上 TileLang GEMM 在测试 workload 下约为 rocBLAS 的 84%（参数扫描后 wrt=64 比 wrt=32 提升约 +24%）**，这是目前参数搜索难以完全弥补的——rocBLAS 可能受益于针对该架构的 vendor-level kernel 启发式选择和调度优化。

### 5.3 小问题规模：减少 threads 反而更快

> 当 grid 很小时（conv1d single-ch 的 N=64，grid 只有 16×16=256 个 block），每个 block 的线程数选择变得反直觉：

gfx1201 单通道小卷积：

| threads | BN | grid | TileLang 耗时 |
| --- | --- | --- | --- |
| 256 | 4 | 16×16 | 0.014ms |
| 128 | 4 | 16×16 | 0.004ms（提速 3.5×） |

threads=128 比 threads=256 快 3.5 倍。每个 block 的工作量相同（4×64=256 个输出元素），但 256-thread 配置增加了每 block 的寄存器占用（8 个 wavefront vs 4 个），在 low-grid 场景下 occupancy flexibility 下降，调度效率随之降低。减小 thread block 尺寸降低了每 kernel 的资源占用，在 grid 小、launch overhead 主导的场景下更有利：

```
# gfx1201 conv1d single-ch：threads=128 是最优的TH_1D = 128 if arch.startswith("gfx1201") else 256
```

## 六、最终性能实测数据

> 修复所有基准之后，三款 GPU 的完整性能对比：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/15.webp)

::: image-caption
三款AMD GPU上TileLang相对PyTorch的算子加速倍率，黑色虚线为1倍基准线，蓝、红、绿柱分别对应gfx1100独显、gfx1151核显、gfx1201专业卡。多数算子TileLang实现性能优于PyTorch，gfx1151在反向传播Bwd bwd达到全图最高2.9倍提速，gfx1201的Flash Attn提速2.4倍；Outer Add算子里gfx1100与gfx1201同为2.2倍、Mul+RELU的gfx1100达2.0倍。仅少量GEMM等场景个别硬件略低于基准，整体iGPU优化收益普遍突出，得益于BN=1连续访存等定制优化，独显与专业卡在注意力、逐元素算子上同样收获明显加速。
:::

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/16.webp)

::: image-caption
三款AMD GPU下TileLang相对Triton的算子加速倍数，黑色虚线为1倍性能基准，蓝、红、绿色柱依次代表gfx1100独显、gfx1151集成iGPU、gfx1201专业卡，测试前已针对各硬件单独调优Triton参数。整体上TileLang多数算子性能优于Triton，gfx1151在反量化Dequant算子取得2.4倍全图表最高加速，Copy算子里iGPU与R9700同为2.2倍提速；反向传播、Softmax、GEMM等算子同样具备可观增益，仅Reduce等个别算子小幅落后基准，iGPU凭借访存定制优化在各类算子中加速表现尤为突出。
:::

### gfx1100（RX 7900 XTX，RDNA3，96 CU，规格带宽 ~0.96 TB/s）

> 软件版本：TileLang 0.1.10+rocm.gitf791a9ca · PyTorch 2.12.0+rocm7.2 · Triton 3.7.0

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/17.webp)

### gfx1151（Radeon 8060S / Strix Halo，RDNA3.5 iGPU，40 CU，规格带宽 ~0.26 TB/s）

> - GEMM：wrt=wct=64, panel=4（参数扫描后 wrt=64 比 wrt=32 快 +24%），TileLang 28.5 TFLOPS，**差距从 −50% 缩至 −16% vs rocBLAS。**
> - Softmax Online：BM=1024（PR #2313 修复后不再受 256 上限约束）。
> - Flash Attention：BB=1, BS=256（2-pass online），+29% vs Triton。

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/18.webp)

- ★：PyTorch 使用 torch.compile/unfold+matmul 优化；
- ★★：Triton Dequant 调参 BM=64,BN=256,BK=16，比默认 64×64×32 快 3.2×（从 29.8ms → 9.4ms）。

### gfx1201（R9700 RDNA4 64CU，规格带宽 ~0.64TB/s）

> - Softmax Online：BM=4096, TH=512（与 gfx1100 相同）。
> - Flash Attn：TH=128, BS=1024，TileLang **+1% vs Triton。**
> - GEMM：118.5 TFLOPS 与 rocBLAS 118.8 TFLOPS 基本持平（PR #2313 正式启用 RDNA4 WMMA 路径）。
> - Dequant：BM=256,BN=128,BK=32,TH=256 → 93.1 TFLOPS，+16% vs PyTorch。

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/19.webp)

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/20.jpg)

::: image-caption
这张图由左右两张雷达图构成，分别展示TileLang相对PyTorch、Triton在三款AMD GPU上全算子的加速倍率，虚线圆环为1倍性能基准，蓝、红、绿线依次对应gfx1100独显、gfx1151核显、gfx1201专业卡，径向数值越高代表加速幅度越大。左侧对标PyTorch的图表里，红色iGPU曲线在Bwd bwd反向传播延伸最远，加速优势最突出；右侧对标Triton图表中，gfx1151在Dequant反量化、Copy算子增幅领跑。整体三款硬件绝大多数算子的多边形都突破基准环线，仅少量GEMM场景小幅低于基线，iGPU凭借访存定制优化在多数算子的性能提升幅度普遍优于两款独显。
:::

## 七、调优系统性结论

做完这些调优，有几个结论与我最初的直觉相反：

### 7.1 线程利用率最优原则：利用率要刚好 100%，多余的线程和闲置的线程都是浪费

> gfx1151 reduce_sum 的完整 config sweep（BN∈{1,2,4}，BM∈{64..2048}，TH∈{128,256,512}）揭示了一个精准的规律：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/21.webp)

关键观察：

- `BN×BM < TH`：每次迭代中部分线程处于 predicated-off 状态（未充分利用），等效带宽大幅下降
- `BN×BM = TH 但迭代多`：线程充分利用，但每行仍需 64 次串行迭代，比 BM=1024 慢 20%——说明迭代次数是独立的关键因素
- `BM 足够大`：迭代次数降低，访存延迟被流水掩盖，接近带宽天花板；但 BM 和 TH 并非越大越好——寄存器压力与 LDS 需求同步上升，存在 occupancy 与吞吐的权衡

> 真正的规律：对 memory-bound 的 reduce kernel，最优配置需要同时满足：

1. 提供充分的线程级并行度（避免大量线程 predicated-off）
2. BM 足够大（迭代次数少，流水掩盖效率高）
3. 最优配置是线程并行度与每线程工作量的平衡点，而非任一维度的单纯最大化

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/22.webp)

::: image-caption
该热力图是gfx1151 iGPU下reduce_sum算子的分块与线程参数遍历测试，纵轴为BM分块尺寸、横轴是单Block线程数TH，配色由红到绿代表耗时由高到低，灰色区域为未测试配置。TH=128全档位均偏红黄、运行耗时偏高，BM=256时达1.398ms；TH=256搭配BM=256/512/1024呈现绿色，性能大幅优化，其中BM=1024、TH=256取得最优0.570ms；TH=512全部空缺未做测试。数据可见提升单块线程、合理放大BM可掩盖访存延迟，该最优配置下iGPU访存逼近硬件带宽上限，是reduce_sum算子定型的关键参数。
:::

gfx1151 上 `BN=1, BM=1024, TH=256` 达到 0.241 TB/s，与 PyTorch、Triton 完全持平——这是 iGPU 统一内存的实测带宽天花板。

### 7.2 带宽分析采用硬件规格峰值而非实测值，防止误导分析

> 分析”某个 kernel 是否已到带宽极限”时，用的是规格峰值带宽而非实测值——实测带宽受 cache 热身状态、kernel 重复次数、并发 workload 等因素影响，存在测量方差；规格值更客观、可复现，也避免读者质疑测试本身的有效性。

三款 GPU 规格峰值带宽：

```
RX 7900 XTX (gfx1100)：~0.96 TB/s（384-bit GDDR6 @ 20 Gbps）R9700       (gfx1201)：~0.64 TB/s（256-bit GDDR6）Radeon 8060S(gfx1151)：~0.26 TB/s（256-bit LPDDR5X-8000）
```

写错了会导致完全错误的分析结论——”kernel 还没到带宽极限”和”已达带宽天花板”对应完全不同的优化方向。

### 7.3 iGPU 的”1309% 加速比”是怎么来的

> gfx1151 上 Conv 单通道 TileLang vs PyTorch 的 818% 加速比，拆开看：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/23.webp)

TileLang 在三款 GPU 上的绝对时间差异不大（0.004~0.006ms，主要在 kernel launch overhead 的测量方差范围内），而 PyTorch 在 gfx1151 上却比另外两款 GPU 慢得多（0.035ms vs ~0.020ms），导致加速比在 gfx1151 上看起来最高。

> 本质上，这是 **kernel launch 延迟与统一内存带宽限制交互放大的结果：**

1. TileLang kernel 极小，主要受 launch overhead 控制，不同 GPU 间差异有限
2. PyTorch 在 gfx1151 上走 unfold+matmul 路径，承受统一内存（~0.26 TB/s）带宽代价；在独立显卡上走 MIOpen 路径，代价不同
3. 低带宽 iGPU 上，通用实现的内存流量代价更高，专用 kernel 的相对优势被放大

这反映的是测试场景（小 problem size + 低带宽统一内存）对两种实现结构差异的放大，而非 TileLang 在 iGPU 上有特殊优势。

### 7.4 PyTorch 的默认写法往往低估了 PyTorch 的能力

> 这个项目让我们认真审视了所有 PyTorch 基准：

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/24.webp)

许多常见的 PyTorch baseline 并未反映 Inductor 等现代编译器后端所暴露的优化执行路径。API 层面的简洁写法往往隐藏了多条性能差异显著的执行路径——修正 baseline 后，性能差距显著缩小，残差主要来自 kernel fusion 策略与内存流量优化，而非框架本身的计算能力限制。

## 八、全文总结

### TileLang 框架层面的关键规则

1. `T.Kernel(threads=N)` 中 N 必须是 Python 字面量，不能是变量（静默 bug）
2. `T.reduce_sum` 改参数后必须跑 correctness，某些后端 lowering 路径下编译期不会明确报错
3. 所有三款 RDNA GPU 使用 `WMMAIntrinEmitter`，B 需预转置为 (N×K)
4. PR #2313（warpSize=64 bug + gfx12 支持 + T.exp2 向量类型修复）：旧版 HIP 模板硬编码 `constexpr warpSize=64`，在特定归约配置下导致 wave32 架构上的正确性偏差；修复后放宽了对 BM 的约束，`T.copy` 与 `T.Parallel` 在多数情况下行为等价
5. gfx1151 `T.exp2` 约束（已修复）：旧版要求 `BM ≤ 256`；PR #2313 在 HIP 数学函数 intrinsic lowering 路径（`intrin_rule_hip.cc`）中为 HIPMath 添加向量类型自动降级，gfx1151 现可使用 `BM=1024`

### iGPU（gfx1151）特化优化

- `BN=1` 行并行：消除跨步写，`outer_vec_add` 和 backward 效果最显著
- 2-pass Online Attention（`BB=1, BS=256`）：减少显存往返，实测 +29% vs Triton（受 workload 大小影响）
- Softmax Online 独立 kernel（`BM=1024, TH=256`）：PR #2313 修复后解除 256 限制，实测与 PyTorch/MIOpen 持平
- `torch.compile` 融合 Mul+ReLU（2 kernel → 1 kernel）
- Conv1D 用 `unfold+matmul` 绕过 MIOpen 启动开销
- WMMA：`wrt=wct=64, panel=4`（参数扫描中观测到比 `wrt=32` 快约 +24%，在测试 workload 下与 rocBLAS 差距从 50% 缩至约 16%）
- Triton 也需要 `BN=1 + GPU-adaptive BLOCK_S`

### 基准公平性

- 用直接计算梯度，不用 autograd backward
- 用单次融合 conv1d，不用 Python for 循环
- gfx1151 上用 `torch.compile` 融合 element-wise ops
- 分析带宽瓶颈时使用规格峰值带宽，避免实测方差影响判断

### 仍然存在的差距

- GEMM vs rocBLAS（iGPU 在测试 workload 下约差 16%，`wrt=64` 配置）：rocBLAS 可能受益于 vendor-level 针对该架构的 kernel 启发式与调度优化
- gfx1151 Reduce Sum：当前配置 `BN=1, BM=1024, TH=256`，带宽约 0.24 TB/s，与 PyTorch/Triton 接近（差距在测量方差范围内），接近当前配置下统一内存的带宽饱和区间

测试代码见：tilelang-notebooks[5] ， 软件版本：TileLang 0.1.10+rocm · PyTorch 2.12.0，rocm7.2，Triton 3.7.0，ROCm 7.2

发现问题或有补充欢迎在评论区指出。

参考资料

[1]

PR#2313: *https://github.com/tile-ai/tilelang/pull/2313*

[2]

TileLang: *https://github.com/tile-ai/tilelang*

[3]

PR #2313: *https://github.com/tile-ai/tilelang/pull/2313*

[4]

PR #2313: *https://github.com/tile-ai/tilelang/pull/2313*

[5]

tilelang-notebooks: *https://github.com/zhangnju/tilelang_notebooks*

::: related-title
**相关推荐**
:::

::: related-links
- **[从形式化推理到 GPU 加速！Lean4 + TileLang 张量程序超优化器，让 Attention 快 4.08 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901134&idx=1&sn=f689cf4d266599f5873320f98ae75d4b&scene=21#wechat_redirect)**
- **[无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**

:::
