关键词：**Rust**、GPU 编程、**cuTile**、CUDA、:mdnice-bold-italic[领域特定语言(DSL)]

> 近年来，Rust 语言以其内存安全、高性能和无畏并发的特性，在系统编程领域赢得了广泛关注。

然而，在 GPU 编程方面，Rust 生态仍然相对年轻——开发者通常需要编写 CUDA C++或 OpenCL 内核，再通过 FFI 与 Rust 宿主代码交互，这不仅繁琐，而且难以保证内存安全。

![文章插图](/articles/2026-03-15-nvlabs-cutile-rs-rust-gpu-15de04ba/images/01.webp)

- NVlabs/cutile-rs
- cuTile Rust provides a safe, tile-based kernel programming DSL for the Rust programming language. It features a safe host-side API for passing tensors to asynchronously executed kernel functions.
- 项目地址：https://github.com/NVlabs/cutile-rs

相关推荐

- [NVIDIA 技术博客：cuTile.jl 将 CUDA 基于分块的编程模型引入 Julia 语言](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898470&idx=1&sn=13df29073e4622bf1ac3f5dd8741f097&scene=21#wechat_redirect)
- [从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)
- [取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)

::: block-1
今天，我们要介绍一个来自 NVIDIA Labs 的研究项目——**cuTile-rs**，它试图填补这一空白，为 Rust 带来一个基于 tile 的、安全的、异步的 GPU 内核编程 DSL。
:::

## 本文目录

- 一、项目概览
- 二、为什么需要 tile-based 编程？
- 三、快速上手：一个简单的加法内核
  - 3.1 定义设备端内核
  - 3.2 Tile 内核的逻辑
  - 3.3 Host 端组合与执行
- 四、异步执行与调度
  - 4.1 调度策略
  - 4.2 避免频繁同步
- 五、安全访问的奥秘：Arc 与 Partition
- 六、深入 tile 内核：以矩阵乘法为例
- 七、编译流程：从 Rust 到 PTX
- 八、安装与配置
- 九、项目状态与未来
- 结语

![文章插图](/articles/2026-03-15-nvlabs-cutile-rs-rust-gpu-15de04ba/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、项目概览

> cuTile-rs 是一个研究性质的开源项目，旨在提供一种**安全的、基于 tile 的内核编程领域特定语言（DSL）**，完全嵌入 Rust 中。它的核心设计围绕三个关键组件展开：

- **cuda-async**：一个允许程序员异步组合 CUDA 操作（构成 DAG）的库，可与任何异步 Rust 运行时（如 tokio）配合使用。
- **cuda-tile-rs**：CUDA Tile MLIR 方言的 Rust 绑定，负责将高级内核定义编译为高效的 GPU 代码。
- **cutile**：主库，包含编译器插件、运行时和 host-side API，为用户提供简洁的编程接口。

项目的目标非常明确：让 Rust 开发者能够用纯 Rust 编写 GPU 内核，并利用 Rust 的所有权系统保证内存安全，同时通过 tile-based 编程模型获得接近手写 CUDA 的性能。

## 二、为什么需要 tile-based 编程？

> 在深入代码之前，有必要理解“tile”（ tile，中文常译为“瓦片”或“块”）的概念。

现代 GPU 架构由大量流多处理器（SM）组成，每个 SM 拥有共享内存和寄存器文件。为了达到高利用率，我们需要将问题分解成适合在 SM 上并行执行的小块——这些小块就是 tile。tile-based 编程允许开发者显式地控制数据的局部性，将数据分成适合共享内存大小的块，从而减少全局内存访问，大幅提升性能。

cuTile-rs 正是围绕 tile-based 编程思想设计的。它提供了一种方式，让开发者编写针对单个 tile 的内核函数，然后由框架自动将整个计算网格划分为多个 tile，并在 GPU 上并行执行。

## 三、快速上手：一个简单的加法内核

> 让我们从一个最简单的例子开始，感受一下 cuTile-rs 的使用方式。

下面是一个完整的向量加法程序：

```
use cutile::prelude::*;use my_module::add_async as add;#[cutile::module]mod my_module {    use cutile::core::*;    #[cutile::entry()]    fn add<const S: [i32; 2]>(        z: &mut Tensor<f32, S>,        x: &Tensor<f32, {[-1, -1]}>,        y: &Tensor<f32, {[-1, -1]}>,    ) {        let tile_x = load_tile_like_2d(x, z);        let tile_y = load_tile_like_2d(y, z);        z.store(tile_x + tile_y);    }}fn main() {    let x = api::ones([32, 32]).arc();    let y = api::ones([32, 32]).arc();    let z = api::zeros([32, 32]).partition([4, 4]);    let (_z, _x, _y) = add(z, x, y).sync();}
```

> 这段代码清晰地展示了 cuTile-rs 的几个核心概念：

### 3.1 定义设备端内核

使用`#[cutile::module]`属性标记一个模块，该模块内的函数将成为设备端可调用的内核。`#[cutile::entry()]`标记了入口函数`add`。注意函数签名中的参数类型：

- `z: &mut Tensor<f32, S>`：这是一个可变引用，表示输出张量。但是，在设备端执行时，每个 tile 线程只能访问张量的一个子区域。因此，在 host 端，我们必须将`z`**划分** 成多个子张量，每个 tile 线程获得一个独占的可变引用。这正是通过`partition([4, 4])`做到的——它将`32x32`的张量划分成`4x4`的 tile 网格，每个 tile 大小为`8x8`。
- `x`和`y`是只读输入张量。由于多个 tile 线程可能同时读取同一个位置，我们需要将它们包装在`Arc`（原子引用计数）中，实现安全的共享访问。

### 3.2 Tile 内核的逻辑

内核函数本身非常简单：它加载与输出 tile 形状相同的输入 tile（`load_tile_like_2d`），执行逐元素加法，然后将结果存储回输出 tile。这里的`load_tile_like_2d`是一个内置函数，它从全局内存中加载一块数据到寄存器或共享内存中（具体由编译器优化决定），并返回一个`Tile`对象，可以像普通 Rust 值一样进行运算。

### 3.3 Host 端组合与执行

在`main`函数中，我们通过`api::ones`和`api::zeros`在设备上分配张量。注意`x`和`y`调用了`.arc()`，将它们转换为`Arc<Tensor<f32>>`；而`z`调用了`.partition([4, 4])`，得到一个`Partition<Tensor<f32>>`。然后我们将这三个参数传递给`add`函数——这个`add`是框架自动生成的异步版本（名字由`add_async`指定），它返回一个实现了`DeviceOperation`的对象。

最后调用`.sync()`：这个方法会选择默认设备，同步地 JIT 编译内核（针对该设备的架构），并立即执行。执行完成后，我们得到一个元组`(_z, _x, _y)`，其中包含了修改后的张量（所有权被归还）。

## 四、异步执行与调度

> 上述例子是同步执行的，但 cuTile-rs 真正的威力在于异步组合。看看下面这个例子：

```
async fn main() {    let a = 2.0;    let x = api::ones::<f32>([16, 16]).arc();   // impl DeviceOperation    let y = api::ones::<f32>([16, 16]).partition([4, 4]); // impl DeviceOperation    let op = api::zip((a, x, y)).apply(my_kernels::saxpy);    let (a, x, y) = op.await;}
```

这里我们构建了一个更复杂的操作：首先创建两个张量（一个用`arc`，一个用`partition`），然后将它们与一个标量`a`组合（`zip`），最后应用一个用户定义的`saxpy`内核。`zip`将多个参数打包成一个元组，`apply`将这个元组传递给内核。

整个`op`是一个`impl DeviceOperation`，它并没有立即执行，而是构建了一个计算图。当我们`.await`它时，它会被隐式转换为一个`DeviceFuture`，并提交给调度器执行。

### 4.1 调度策略

`DeviceOperation`可以通过`schedule`方法显式指定调度策略：

```
let zf: DeviceFuture<Output=T> = z.schedule(global_policy(1));let z: Tensor<f32> = zf.await;
```

这里使用了全局策略，指定在设备 1 上执行。调度器负责为操作分配资源（如 CUDA 流、事件等），并确保依赖关系正确。默认情况下，连续的操作会被放在同一个流上，从而避免不必要的同步开销。

### 4.2 避免频繁同步

初学者可能会写出这样的代码：

```
let x: Tensor<f32> = api::ones([16, 16]).await;let y: Tensor<f32> = api::ones([16, 16]).await;let z: Tensor<f32> = api::add(x.into(), y.into()).await;
```

虽然正确，但效率低下：每个`.await`都会导致与异步运行时的同步。更好的做法是将所有操作组合成一个大的`DeviceOperation`，只同步一次：

```
let xf = api::ones([16, 16]);let yf = api::ones([16, 16]);let z: Tensor<f32> = api::add(x, y).await;
```

这里`x`和`y`是操作而不是已完成的张量，`api::add`接收这两个操作，返回一个新的操作，最后一次性`.await`。调度器会将它们安排到同一个流上，执行完毕后通过 CUDA host callback 通知运行时。

## 五、安全访问的奥秘：Arc 与 Partition

> cuTile-rs 最巧妙的设计之一是如何利用 Rust 的所有权系统保证设备端内存访问的安全性。

在传统的 CUDA 编程中，同一个设备指针可能被多个线程同时读写，数据竞争的风险完全由程序员负责。cuTile-rs 通过类型系统将这种风险消除：

- **只读共享**：用`Arc<Tensor<T>>`表示。多个 tile 线程可以同时读取同一个张量，因为`Arc`保证引用计数的线程安全，且张量本身是不可变的。在设备端，内核参数接收到的是`&Tensor<T>`，表明只读。
- **独占可变**：用`Partition<Tensor<T>>`表示。一个张量被划分为多个不相交的子张量，每个子张量只会被一个 tile 线程独占访问。在设备端，内核参数接收到的是`&mut Tensor<T, S>`，其中的常量`S`表示该子张量的形状（由 host 端的划分决定）。这保证了同一个内存位置绝不会被两个线程同时写入。

这种设计完全符合 Rust 的“共享不可变，可变不共享”原则，并且扩展到了设备端。更重要的是，`Partition`的划分是静态的——划分的 tile 大小在编译时已知（通过常量泛型），这为编译器优化提供了丰富的信息。

## 六、深入 tile 内核：以矩阵乘法为例

> 矩阵乘法是 GPU 计算的经典例子。让我们看看 cuTile-rs 如何实现一个高效的 GEMM 内核（来自`cutile-examples/examples/gemm.rs`）：

```
#[cutile::entry()]fn gemm<const M: [i32; 2], const N: [i32; 2], const K: [i32; 2]>(    c: &mut Tensor<f32, M>,   // 输出矩阵 C (M行 x N列)    a: &Tensor<f32, {[-1, -1]}>, // 输入矩阵 A (M行 x K列)    b: &Tensor<f32, {[-1, -1]}>, // 输入矩阵 B (K行 x N列)) {    // 加载A和B的tile到共享内存    let tile_a = load_tile::<f32, _, _, _>(a, ());    let tile_b = load_tile::<f32, _, _, _>(b, ());    // 执行矩阵乘法累加    letmut acc = Tile::zeros();    for k in0..K[0] {        let a_tile = tile_a.slice(k..k+1, ..);        let b_tile = tile_b.slice(.., k..k+1);        acc += a_tile * b_tile;    }    // 将结果存储到C    c.store(acc);}
```

这个内核展示了 tile-based 矩阵乘法的经典模式：每个 tile 线程负责计算输出矩阵 C 中的一个 tile。它从 A 和 B 中加载对应的 tile（可能通过`load_tile`加载到共享内存），然后进行累加。由于每个 tile 的形状由常量泛型参数`M, N, K`指定，编译器可以生成高度优化的代码，利用流水线和张量核心（如果可用）。

host 端的调用可能如下：

```
let a = api::random([1024, 1024]).arc();let b = api::random([1024, 1024]).arc();let c = api::zeros([1024, 1024]).partition([32, 32]);gemm(c, a, b).sync();
```

这里我们将输出矩阵`c`划分为`32x32`的 tile，每个 tile 大小为`32x32`（因为 1024/32=32）。内核将在一个包含`32x32`个 tile 线程的网格上执行。

## 七、编译流程：从 Rust 到 PTX

> cuTile-rs 的编译流程涉及多个阶段，充分利用了 LLVM/MLIR 基础设施：

1. **宏展开与 AST 分析**：`#[cutile::module]`宏会分析模块内的内核函数，提取参数类型、tile 形状等元数据，并生成 host-side 的 wrapper 代码（如`add_async`函数）。
2. **MLIR 代码生成**：内核函数的 Rust 代码（实际上是一个 AST）被转换为 MLIR（Multi-Level Intermediate Representation），具体来说是`cuda-tile`方言。这个方言提供了 tile 操作、加载/存储、算术运算等高级抽象。
3. **MLIR 优化与降级**：MLIR passes 将`cuda-tile`方言逐步降级到 LLVM 方言，同时执行循环展开、共享内存分配、向量化等优化。这个阶段利用了 MLIR 的 tile-based 优化能力，例如自动将 tile 加载到共享内存并生成同步指令。
4. **LLVM IR 与 PTX 生成**：最终，LLVM IR 被编译为 PTX（Parallel Thread Execution）代码，并嵌入到最终的二进制中。运行时通过 cuModuleLoadData 加载并执行。

这个流程使得 cuTile-rs 能够生成与手写 CUDA 相媲美的代码，同时让开发者用高级的 Rust DSL 编写内核。

## 八、安装与配置

> 要尝试 cuTile-rs，你需要满足一些依赖条件：

- **NVIDIA GPU**：支持`sm_80`或`>= sm_100`计算能力（`sm_90`暂不支持）。
- **CUDA 13.2**：需要安装对应版本的 CUDA Toolkit。
- **LLVM 21 with MLIR**：可以从 apt.llvm.org 安装，并确保`llvm-config`指向 LLVM 21。
- **Rust nightly** （部分功能需要）。

具体安装步骤可参考项目 README，大致如下：

```
# 安装LLVM 21wget https://apt.llvm.org/llvm.shchmod +x llvm.shsudo ./llvm.sh 21sudo apt-get install libmlir-21-dev mlir-21-tools# 设置环境变量export CUDA_TOOLKIT_PATH=/usr/local/cuda-13export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21# 构建并测试git clone --recursive https://github.com/NVlabs/cutile-rs.gitcd cutile-rscargo test
```

## 九、项目状态与未来

> cuTile-rs 目前处于“alpha”阶段，由 NVIDIA Labs 作为研究项目发布。这意味着它还存在 bug，功能不完整，API 也可能发生变化。

尽管如此，它已经展示了一个令人兴奋的方向：**将 Rust 的安全性和现代 GPU 编程结合**，有望彻底改变 GPU 计算的开发体验。

项目的未来计划包括：

- 支持更多的 GPU 架构（如 sm_90）。
- 完善异步运行时集成。
- 提供更多的内核优化策略。
- 改进编译时错误信息。

## 结语

> NVIDIA 这次开源的 `cutile-rs`，展示了如何将 Rust 的编译期安全性设计哲学引入到原本极其危险的 GPU 共享内存和并发执行编程中。通过：

1. `Partition` 实现了内核级别的**数据竞争安全**。
2. `Arc` 实现了跨流或跨内核的**安全资源共享**。
3. `cuda-async` 提供了一流的异步**流调度抽象**。
4. 底层依赖了先进的 `MLIR` (LLVM 21) 进行高效的中间代码生成。

目前 `cutile-rs` 虽然在功能覆盖上可能比不上完整的 CUDA C++ 甚至 Triton，但从语言工程学的角度讲，它迈出了重要的一步。如果你有兴趣深入，可以前往仓库进一步查阅：

- 更多官方用例位于 `cutile-examples/examples` 下。
- 自动化测试可通过脚本 `./scripts/run_all_tests.sh` 全量运行。

开源世界需要勇士去探索和反馈。带上你的 `RTX 3090` 或 `RTX 4090` (`sm_80` 级以上），去尝试用 Rust 点亮你的 GPU 吧！

项目地址：https://github.com/NVlabs/cutile-rs

相关推荐

- [NVIDIA 技术博客：cuTile.jl 将 CUDA 基于分块的编程模型引入 Julia 语言](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898470&idx=1&sn=13df29073e4622bf1ac3f5dd8741f097&scene=21#wechat_redirect)
- [从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)
- [取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)

交流加群请在 NeuralTalk 公众号后台回复：加群
