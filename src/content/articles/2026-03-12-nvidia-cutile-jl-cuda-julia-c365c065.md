关键词：**cuTile.jl**、:mdnice-bold-italic[CUDA Tile]、Julia、GPU 编程、**分块编程**、高性能

> NVIDIA CUDA Tile 是 CUDA 编程中最重要的新增特性之一，可让程序自动调用张量核心（NVIDIA GPU 中专门加速矩阵运算的硬件单元）及其他专用硬件。今年年初，NVIDIA 发布了Python 版 cuTile[1]，为 Python 开发者提供了编写高性能 GPU 内核的自然方式。

![文章插图](/articles/2026-03-12-nvidia-cutile-jl-cuda-julia-c365c065/images/01.webp)

- **cuTile.jl Brings NVIDIA CUDA Tile-Based Programming to Julia**
- https://developer.nvidia.com/blog/cutile-jl-brings-nvidia-cuda-tile-based-programming-to-julia/
- **代码：https://github.com/JuliaGPU/cuTile.jl**
- 3000 字，阅读 13 分钟，**播客 7 分钟**

相关推荐

- [对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)
- [取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)
- [从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)

> cuTile.jl[2] 是一款实验性开源软件包，首次将 NVIDIA CUDA Tile 基于分块的编程模型引入 Julia 语言。

传统 CUDA 编程需开发者手动管理线程、内存层级等底层细节，而该工具通过抽象这些复杂操作，让开发者聚焦数据分块运算，大幅简化高性能 GPU 内核开发。**其语法与 Python 版 cuTile 高度兼容**，便于代码迁移，同时适配 Julia 1-based 索引、广播机制等原生特性。

::: block-1
在 NVIDIA GeForce RTX 5080 上，:mdnice-bold-italic[向量加法、矩阵乘法等核心内核性能达 Python 版的 99%-100%，仅复杂控制流内核因编译器尚在成熟阶段存在小幅差距。]

该工具依托 Julia GPU 生态，支持内存访问、矩阵运算、归约等多种分块操作及丰富示例，需搭配 Blackwell 架构 GPU、CUDA 13+ 驱动与 Julia 1.11+ 版本，**虽部分特性待完善，但为 Julia 开发者提供了高效便捷的 GPU 编程新方案。**
:::

本文将介绍 cuTile.jl 如何简化高性能 CUDA 内核开发，展示其符合 Julia 语言习惯的语法，并对比其与现有 Python 版 cuTile 的性能一致性。

## 本文目录

- 关键问题
  - 问题一、性能对等的代价
  - 问题二、编译策略的挑战
- 一、什么是基于分块的 GPU 编程？
- 二、符合 Julia 语言习惯的内核
- 三、cuTile.jl 的性能
- 四、cuTile.jl 的工作原理
- 五、cuTile.jl 的当前状态
- 六、快速上手

![文章插图](/articles/2026-03-12-nvidia-cutile-jl-cuda-julia-c365c065/images/01.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 关键问题

### 问题一、性能对等的代价

> cuTile.jl 声称与 Python 版性能相当，但 Tile 抽象是否会在内存访问模式或非计算密集型任务中引入隐藏的性能损失或限制优化空间？

作者指出，cuTile.jl **与 Python 版均基于相同的 NVIDIA Tile IR 后端，因此在计算密集型任务中可达到性能对等**。然而，Tile 抽象通过分块隐藏了底层线程调度，虽然简化了编程，但在内存访问模式上可能引入潜在开销。

例如，:mdnice-bold-italic[若算法访存模式不规则（如随机访问），分块策略可能导致缓存未命中或冗余数据加载，降低有效带宽]。对于非计算密集型任务（如纯内存拷贝），Tile 抽象可能因分块管理开销而无法达到最优性能，且编译器可能难以自动优化访存模式。

此外，分块大小需手动调优，不当选择会限制占用率，带来性能损失。作者未提及这些场景，但实践表明，抽象层需在通用性与极致优化间权衡。

### 问题二、编译策略的挑战

> cuTile.jl 通过自定义编译器拦截标准库调用，这种设计与 Julia 的 JIT 编译特性如何协同？是否会带来调试难度或与其他 Julia 包互操作的障碍？

Julia 采用多次分派的 JIT 编译，而自定义编译器可能通过宏或编译器钩子在 AST 层面转换代码，将标准操作映射到 Tile IR。

这种设计可能依赖 Julia 的元编程能力，但需确保与 JIT 流程兼容，避免编译时冲突。调试难度方面，由于代码经过转换，用户追踪原始逻辑与生成代码的对应关系可能复杂，且错误信息可能难以理解。与其他包互操作时，若拦截了全局标准库调用，可能影响其他包中未预期此行为的代码，导致意外行为或性能下降。作者未详述实现细节，但此类深层编译技术需谨慎设计以避免生态系统冲突。

## 一、什么是基于分块的 GPU 编程？

> 传统基于 CUDA 的 GPU 编程需要开发者关注线程（GPU 最小执行单元）、线程束（GPU 执行的基础线程组）和内存层级（GPU 从寄存器到显存的多级存储结构）。

这套方案虽功能强大，但要求开发者将算法高效映射到硬件上。而使用 CUDA Tile 时，开发者只需描述对**数据分块** 的操作，编译器会自动完成硬件映射。

以向量加法为例。在传统 GPU 编程模型中，使用 CUDA.jl 时开发者必须显式管理单个线程：

```
using CUDAfunction vadd(a, b, c, n)    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x    if i <= n        @inbounds c[i] = a[i] + b[i]    end    returnendthreads = 512blocks = cld(vector_size, threads)@cuda threads blocks vadd(a, b, c, vector_size)
```

而通过 cuTile.jl 使用 CUDA Tile 时，相同操作可直接在**分块层级** 表达，屏蔽了索引计算、越界检查等细节：

```
import cuTile as ctfunction vadd(a, b, c, tile_size)    pid = ct.bid(1)    tile_a = ct.load(a, pid, (tile_size,))    tile_b = ct.load(b, pid, (tile_size,))    ct.store(c, pid, tile_a + tile_b)    returnendtile_size = 1024grid = cld(vector_size, tile_size)ct.launch(vadd, grid, a, b, c, ct.Constant(tile_size))
```

与 Python 版本对比如下：

```
@ct.kerneldef vadd(a, b, c, tile_size: ct.Constant[int]):    pid = ct.bid(0)    tile_a = ct.load(a, index=(pid,), shape=(tile_size,))    tile_b = ct.load(b, index=(pid,), shape=(tile_size,))    ct.store(c, index=(pid,), tile=tile_a + tile_b)tile_size = 1024grid = ceil(vector_size / tile_size)ct.launch(stream, grid, vadd, (a, b, c, tile_size))
```

两者语法高度相似，这是设计之初的刻意选择。cuTile.jl 保持内核的抽象层级与 Python 版 cuTile 完全一致，降低代码迁移难度，也方便开发者参考 Python 版 cuTile 文档。同时，它尽可能采用 Julia 语言习惯，让 Julia 开发者上手更直观，包括**从 1 开始索引** 和用于逐元素运算的广播表达式。

## 二、符合 Julia 语言习惯的内核

> 真正体现优势的是超越简单读写操作的复杂内核。下面是一个**行归一化内核** ——层归一化（深度学习中常用的数据标准化方法）的核心逻辑，不含权重与偏置项：

```
function normalize_rows(X, Y, tile_n)    bid = ct.bid(1)    tile = ct.load(X, (bid, 1), (1, tile_n))    mean = sum(tile; dims=2) / size(X, 2)    centered = tile .- mean    var = sum(centered .^ 2.0f0; dims=2) / size(X, 2)    ct.store(Y, (bid, 1), centered ./ sqrt.(var .+ 1f-5))    returnend
```

在这个示例中，`sum`、`size` 和 `sqrt` 都是经扩展后可作用于**数据分块** 的标准 Julia 函数。点运算符（`.^`、`.-`、`./`）是 Julia 标准广播语法，表示操作按逐元素执行。这个内核看起来和普通的 Julia 数组代码完全一样。cuTile.jl 内核越接近普通 Julia 代码，CPU 与 GPU 之间的代码共享和复用就越容易。

## 三、cuTile.jl 的性能

> cuTile.jl 与 Python 版 cuTile 采用相同的 NVIDIA Tile IR 后端，因此两个软件包生成的 GPU 机器码类型一致。在 NVIDIA GeForce RTX 5080（计算能力 12.0，NVIDIA Blackwell 架构，NVIDIA 新一代 GPU 硬件架构）上，计算密集型内核的性能与 Python 版本基本持平：

![文章插图](/articles/2026-03-12-nvidia-cutile-jl-cuda-julia-c365c065/images/02.webp)

::: image-caption
表 1：使用 Julia 或 Python 作为前端时，常见 GPU 内核的性能对比
:::

部分控制流更复杂的内核（如层归一化、快速傅里叶变换 FFT）尚未达到完全的性能一致，这是因为 cuTile.jl 编译器仍在完善中。这些问题已被记录为已知问题，开发团队正在持续优化。

## 四、cuTile.jl 的工作原理

> cuTile.jl 使用定制的 Julia 编译器，拦截 `+`、`sum`、`reshape` 等标准库调用，并将其路由为 Tile IR 操作。生成的中间表示会被进一步编译为 Tile IR 字节码——这与 Python 版 cuTile 生成的二进制格式完全相同。之后，NVIDIA `tileiras` 编译器负责最终编译为 GPU 机器码。

任意内核生成的 Tile IR 都可以被查看：

```
julia> ct.@device_code_tiled ct.launch(vadd, grid, a, b, c, ct.Constant(16))cuda_tile.module @kernels {  entry @vadd(%arg0: tile<ptr<f32>>, %arg1: tile<i32>, ...) {    ...    return  }}
```

这种透明性对调试、理解高层 Julia 代码如何映射为分块操作非常有价值。

## 五、cuTile.jl 的当前状态

> cuTile.jl 是一款仍在积极开发中的实验性开源软件包，代码托管于JuliaGPU/cuTile.jl[3]。它已支持大量分块操作，包括内存访问、算术运算、归约操作、前缀扫描、矩阵乘法、形状变换和原子操作。同时提供了可运行的示例，涵盖向量加法、矩阵乘法、转置、批量矩阵乘法、层归一化和快速傅里叶变换。

尽管如此，该软件仍处于早期阶段，存在以下限制：

- 并非所有 cuTile 特性都已实现。
- 部分 Julia 语言特性（尤其是基于迭代器的 `for` 循环）在内核中暂不支持，或会生成低效代码。
- 与 CUDA.jl 的集成仍需优化，以更好地与单指令多线程 SIMT 内核（GPU 的并行执行模式）共存。
- API 可能在无预告的情况下变更。

该项目依托 Julia 现有的 GPU 生态构建，与CUDA.jl[4]集成完成数组管理和内核启动。已使用 CUDA.jl 编写 Julia GPU 代码的用户，能轻松过渡到基于分块的编程模式。

## 六、快速上手

> 与 Python 版 cuTile 一样，cuTile.jl 需要搭载 NVIDIA Blackwell 架构的 GPU，以及支持 CUDA 13 及以上版本的 NVIDIA 驱动。同时要求 Julia 版本为 1.11 及以上。

启动 Julia，在交互式命令行中按下 `]` 进入内置包管理器，安装 cuTile.jl：

```
pkg> add cuTilepkg> # 如需可运行测试套件      test cuTile
```

GitHub 仓库中包含所有支持操作的完整列表，以及 cuTile.jl 与 Python 版 cuTile、标准 Julia 语法差异的详细文档。

参考资料

[1] Simplify GPU Programming with NVIDIA CUDA Tile in Python: *https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/*

[2] https://github.com/JuliaGPU/cuTile.jl: *https://github.com/JuliaGPU/cuTile.jl*

[3] JuliaGPU/cuTile.jl: *https://github.com/JuliaGPU/cuTile.jl*

[4] CUDA.jl: *https://github.com/JuliaGPU/CUDA.jl*
