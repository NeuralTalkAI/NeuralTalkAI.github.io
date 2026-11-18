关键词：CUDA 并行计算、**CCCL 三层架构**、CUB 块级原语、:mdnice-bold-italic[Thrust 执行策略]、**libcudacxx 硬件抽象**

> 在当今 GPU 加速计算生态中，NVIDIA CCCL（CUDA Core Compute Libraries）是一个不可忽视的底层基石。

![文章插图](/articles/2026-04-09-nvidia-cccl-cuda-core-compute-3088ef5b/images/01.webp)

- **CUDA Core Compute Libraries.**
- CCCL的目标是为CUDA C++开发者提供基础组件，让编写安全且高效的代码变得更加简便。将这些库整合在一起能够简化开发流程，并拓展发挥CUDA C++强大性能的能力。
- https://github.com/NVIDIA/cccl
- https://nvidia.github.io/cccl
- 5000 字，阅读 19 分钟，**播客 8 分钟**

相关推荐

- **[对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)**
- **[暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)**
- **[取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)**

> 从推理框架 SGLang、vLLM，到训练框架 PyTorch、深度学习编译器 TVM，再到科学计算库 RAPIDS，**几乎所有主流 CUDA C++ 项目都直接或间接依赖 CCCL 提供的 Thrust、CUB 和 libcudacxx。**

**它之于 CUDA 生态的地位，正如 C++ 标准库之于 C++ 本身** ——是并行计算的通用基础设施。

![文章插图](/articles/2026-04-09-nvidia-cccl-cuda-core-compute-3088ef5b/images/02.webp)

::: image-caption
CCCL（CUDA C++核心库）仓库的架构图，整体分为核心库、实验性组件、基础设施三大模块。核心库中，Thrust（高层并行算法库）依赖CUB（块/线程束/设备原语库），二者与实验区的cudax组件，均依赖底层libcudacxx（C++标准库与硬件抽象层）；实验区还包含c/parallel组件；基础设施则提供CMake构建、命名空间管理、统一版本管理等底层支撑。
:::

::: block-1
本文将从源码层面剖析 CCCL 的三层抽象架构、核心算法实现、执行策略分发机制与统一版本治理体系。
:::

## 本文目录

- 一、项目定位与设计动机
- 二、三层抽象架构
  - 2.1 libcudacxx — 基础层：C++ 标准库的 CUDA 实现
  - 2.2 CUB — 中间层：CUDA 特定的高性能并行原语
  - 2.3 Thrust — 高层：跨后端的并行算法库
- 三、统一构建系统
- 四、命名空间与版本管理
  - 4.1 命名空间包装机制
  - 4.2 统一版本控制
  - 4.3 ABI 版本管理
- 五、整体架构总结

![文章插图](/articles/2026-04-09-nvidia-cccl-cuda-core-compute-3088ef5b/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、项目定位与设计动机

> CCCL 是 NVIDIA 将三个独立发展的 CUDA C++ 核心库——**Thrust**、**CUB**、**libcudacxx** ——统一到单一仓库中的项目。

其使命在 README.md 中明确表述为"make CUDA more delightful"，目标是为 CUDA C++ 开发者提供类似于 C++ 标准库之于标准 C++ 的角色：**通用的、极致性能的基础工具集**。

::: block-1
三个库在合并前各自独立维护，导致重复开发、版本不一致和用户集成复杂等问题。CCCL 2.2.0 是从统一仓库发布的第一个版本，此后所有组件共享统一的语义版本号，本文写作时 `main` 分支版本为 **3.4.0**。
:::

## 二、三层抽象架构

> CCCL 的核心设计呈现清晰的**三层抽象**，从底层到高层依次为：

![文章插图](/articles/2026-04-09-nvidia-cccl-cuda-core-compute-3088ef5b/images/04.webp)

### 2.1 libcudacxx — 基础层：C++ 标准库的 CUDA 实现

> libcudacxx 是整个 CCCL 的**地基**，提供可在 host 和 device 代码中同时工作的 C++ 标准库实现，以及 CUDA 硬件特定的抽象（同步原语、缓存控制、原子操作等）。

其头文件组织在 `libcudacxx/include/cuda/` 下，涵盖极其广泛的功能模块：

| 目录 | 功能 | 关键公开头文件 |
| --- | --- | --- |
| `__atomic/` | 原子操作 | `cuda/atomic` |
| `__barrier/` | 屏障同步 | `cuda/barrier` |
| `__memory_resource/` | 内存资源管理 | `cuda/memory_resource` |
| `__ptx/` | PTX 指令封装 | `cuda/ptx` |
| `__stream/` | CUDA 流抽象 | `cuda/stream` |
| `__mdspan/` | 多维数组视图 | `cuda/mdspan` |
| `__warp/` | Warp 级原语 | `cuda/warp` |
| `__launch/` | 内核启动抽象 | `cuda/launch` |
| `__execution/` | 执行策略与确定性控制 | — |
| `std/` | C++ 标准库兼容层 | `cuda/std/*` |

#### 2.1.1 原子操作的作用域模型

以原子操作为例，公开头文件 `cuda/atomic` 通过双重包含策略实现兼容性：

```
#include <cuda/__atomic/atomic.h>   // CUDA 特定实现（带 thread_scope）#include <cuda/std/atomic>          // C++ 标准库兼容接口
```

核心类 `cuda::atomic<T, Sco>` 定义在 `libcudacxx/include/cuda/__atomic/atomic.h`，它继承 `cuda::std::__atomic_impl` 并添加了 CUDA 扩展操作 `fetch_max`/`fetch_min`：

```
template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>struct atomic :public ::cuda::std::__atomic_impl<_Tp, _Sco>{// 标准 atomic 接口通过基类继承_CCCL_API constexpr atomic(_Tp __d) noexcept      : ::cuda::std::__atomic_impl<_Tp, _Sco>(__d) {}// CUDA 扩展：原子 fetch_max / fetch_min_CCCL_API inline _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) noexcept  {    return ::cuda::std::__atomic_fetch_max_dispatch(      &this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});  }// ... fetch_min 类似};
```

同样，`cuda::atomic_ref<T, Sco>` 提供对已有变量的原子引用，定义在 同一文件的 L79-L111。

**线程屏障 (`atomic_thread_fence`)** 根据作用域分发到不同的底层 CUDA 原语，展示了 libcudacxx 如何将硬件特性抽象为统一接口：

```
_CCCL_API inline voidatomic_thread_fence(memory_order __m, thread_scope _Scope = thread_scope::thread_scope_system){  NV_DISPATCH_TARGET(    NV_IS_DEVICE,    (switch (_Scope) {      case thread_scope::thread_scope_system:        ::cuda::std::__atomic_thread_fence_cuda((int) __m, __thread_scope_system_tag{});        break;      case thread_scope::thread_scope_device:        ::cuda::std::__atomic_thread_fence_cuda((int) __m, __thread_scope_device_tag{});        break;      case thread_scope::thread_scope_block:        ::cuda::std::__atomic_thread_fence_cuda((int) __m, __thread_scope_block_tag{});        break;      case thread_scope::thread_scope_thread:        break; // 线程级作用域无需 fence    }),    NV_IS_HOST,    (::cuda::std::atomic_thread_fence(__m);))  // host 端回退到标准实现}
```

`NV_DISPATCH_TARGET` 宏在编译时分发 device/host 代码路径，消除运行时分支开销。

#### 2.1.2 屏障同步 (`cuda::barrier`)

`libcudacxx/include/cuda/__barrier/barrier.h` 定义了 `cuda::barrier<Sco, CompletionF>`，同样通过 `thread_scope` 模板参数实现作用域控制：

```
template <thread_scope _Sco, class _CompletionF>class barrier : public ::cuda::std::__barrier_base<_CompletionF, _Sco>{public:  _CCCL_API constexpr barrier(::cuda::std::ptrdiff_t __expected, _CompletionF __completion = _CompletionF())      : ::cuda::std::__barrier_base<_CompletionF, _Sco>(__expected, __completion) {}  // 就地初始化 (placement new) 友元函数，用于 __shared__ 内存中的 barrier  _CCCL_API inline friend void init(barrier* __b, ::cuda::std::ptrdiff_t __expected) {    new (__b) barrier(__expected);  }};
```

#### 2.1.3 PTX 指令封装

`libcudacxx/include/cuda/__ptx/instructions/get_sreg.h` 提供了对 PTX 特殊寄存器的 C++ 封装（如 `cuda::ptx::get_sreg_laneid()`），CUB 的底层实现直接使用这些封装而非原始内联汇编。

用户使用时指定**作用域** （如 `cuda::thread_scope_device`）和**内存序** （如 `cuda::memory_order_relaxed`），原子操作直接编译为 GPU 硬件原子指令：

```
cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(result.front());atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
```

### 2.2 CUB — 中间层：CUDA 特定的高性能并行原语

> CUB 是面向 CUDA 内核开发者的底层库，提供 **block 级**、**warp 级** 和 **device 级** 的并行算法原语。

#### 2.2.1 Block 级原语：以 `BlockReduce` 为例

Block 级原语是 CUB 的核心特色。`cub/cub/block/block_reduce.cuh` 通过 `BlockReduceAlgorithm` 枚举（L43-L153）提供四种算法变体：

| 算法变体 | 特点 | 执行阶段 |
| --- | --- | --- |
| `BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY` | 高效 raking 算法，仅支持交换律操作 | ① 寄存器规约 → ② 共享内存 raking → ③ Kogge-Stone warp 规约 |
| `BLOCK_REDUCE_RAKING` | 支持非交换操作的 raking 算法 | 同上，但每个线程均写入共享内存 |
| `BLOCK_REDUCE_WARP_REDUCTIONS` | 低延迟的 warp 级规约 | ① 寄存器规约 → ② warp 内 Kogge-Stone → ③ warp 间串行传播 |
| `BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC` | 使用原子操作的非确定性变体 | ① 寄存器规约 → ② warp 内 Kogge-Stone → ③ 原子累加 |

其实现依赖三个特化文件，在编译时根据模板参数选择最优路径：

```
#include <cub/block/specializations/block_reduce_raking.cuh>#include <cub/block/specializations/block_reduce_raking_commutative_only.cuh>#include <cub/block/specializations/block_reduce_warp_reductions.cuh>
```

**`BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY` 的核心实现** （`block_reduce_raking_commutative_only.cuh`）展示了 raking 算法的典型工作流——非 raking 线程将数据写入共享内存网格，raking warp 线程串行归约各自的段，最后 warp 内规约得到最终结果：

```
template <bool FULL_TILE>_CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T partial, int num_valid){if (USE_FALLBACK || !FULL_TILE) {    return FallBack(temp_storage.fallback_storage).template Sum<FULL_TILE>(partial, num_valid);  } else {    // 阶段 1: 非 raking 线程将部分结果写入共享内存网格    if (linear_tid >= RAKING_THREADS) {      *BlockRakingLayout::PlacementPtr(        temp_storage.default_storage.raking_grid, linear_tid - RAKING_THREADS) = partial;    }    __syncthreads();    // 阶段 2: Raking warp 串行归约 + warp 内规约    if (linear_tid < RAKING_THREADS) {      T* raking_segment = BlockRakingLayout::RakingPtr(        temp_storage.default_storage.raking_grid, linear_tid);      auto span = ::cuda::std::span<T, SEGMENT_LENGTH>(raking_segment, SEGMENT_LENGTH);      partial   = cub::ThreadReduce(span, ::cuda::std::plus<>{}, partial);  // 串行段规约      partial   = WarpReduce(temp_storage.default_storage.warp_storage).Sum(partial);  // warp 规约    }  }return partial;}
```

**非确定性变体的原子累加实现** （`block_reduce_warp_reductions.cuh` L112-L133）展示了 CUB 如何利用 libcudacxx 的原子操作原语：

```
template <typename ReductionOp>_CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregatesNonDeterministic(ReductionOp reduction_op, T warp_aggregate){// Warp 0 的 lane 0 (即 linear_tid == 0) 写入初始值if (linear_tid == 0) {    detail::uninitialized_copy_single(temp_storage.warp_aggregates, warp_aggregate);  }  __syncthreads();// 其他 warp 的 lane 0 使用原子操作累加（顺序不确定）if (lane_id == 0 && warp_id != 0) {    NV_IF_TARGET(NV_PROVIDES_SM_60,      (::cuda::atomic_ref<T, ::cuda::thread_scope_block> atomic_target(temp_storage.warp_aggregates[0]);       atomic_target.fetch_add(warp_aggregate, ::cuda::memory_order_relaxed);),      (atomicAdd(&temp_storage.warp_aggregates[0], warp_aggregate);));  }  __syncthreads();return temp_storage.warp_aggregates[0];}
```

注意此处直接使用了 `cuda::atomic_ref<T, cuda::thread_scope_block>` —— 这是 CUB 层依赖 libcudacxx 层的具体体现。

#### 2.2.2 Block 级原语完整列表

`BlockScan` 同样采用算法变体模式，定义在 `cub/cub/block/block_scan.cuh`，提供 `BLOCK_SCAN_RAKING`、`BLOCK_SCAN_RAKING_MEMOIZE`、`BLOCK_SCAN_WARP_SCANS` 三种变体，依赖两个特化文件：

```
#include <cub/block/specializations/block_scan_raking.cuh>#include <cub/block/specializations/block_scan_warp_scans.cuh>
```

Block 级原语的完整列表包括：`block_reduce`、`block_scan`、`block_exchange`、`block_radix_sort`、`block_merge_sort`、`block_histogram`、`block_load`、`block_store` 等，均位于 `cub/cub/block/` 目录。

#### 2.2.3 Device 级算法

CUB 提供丰富的 device 级算法，位于 `cub/cub/device/` 目录，是 Thrust 高层算法的底层实现基础：

| 文件 | 算法 |
| --- | --- |
| `device_reduce.cuh` | 规约（Sum, Min, Max, ArgMin, ArgMax） |
| `device_scan.cuh` | 前缀扫描 |
| `device_radix_sort.cuh` | 基数排序 |
| `device_merge_sort.cuh` | 归并排序 |
| `device_select.cuh` | 流压缩 / 条件选取 |
| `device_partition.cuh` | 分区 |
| `device_histogram.cuh` | 直方图 |
| `device_for.cuh` | 并行 for-each |
| `device_transform.cuh` | 变换 |
| `device_merge.cuh` | 合并 |
| `device_topk.cuh` | Top-K 选取 |

Device 级算法的 Dispatch 层位于 `cub/cub/device/dispatch/`，以 `DeviceReduce` 为例，其 `device_reduce.cuh` 引入了三种 dispatch 策略：

```
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>#include <cub/device/dispatch/dispatch_reduce_deterministic.cuh>#include <cub/device/dispatch/dispatch_reduce_nondeterministic.cuh>#include <cub/device/dispatch/dispatch_streaming_reduce.cuh>
```

`reduce_impl` 方法（L106-L145）展示了 CUB 的策略选择机制——通过 `execution::determinism` 类型标签在编译时选择确定性或非确定性路径：

```
// 通用路径：根据 Determinism 模板参数选择 dispatchCUB_RUNTIME_FUNCTION static cudaError_t reduce_impl(    void* d_temp_storage, size_t& temp_storage_bytes,    InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items,    ReductionOpT reduction_op, TransformOpT transform_op, T init,    ::cuda::execution::determinism::__determinism_holder_t<Determinism>,    cudaStream_t stream){  using accum_t = ::cuda::std::__accumulator_t<ReductionOpT, ...>;  using policy_selector = detail::reduce::policy_selector_from_types<accum_t, offset_t, ReductionOpT>;  return detail::reduce::dispatch<accum_t>(    d_temp_storage, temp_storage_bytes, d_in, d_out,    static_cast<offset_t>(num_items), reduction_op, init, stream, transform_op,    policy_selector{});}
```

### 2.3 Thrust — 高层：跨后端的并行算法库

> Thrust 是面向应用开发者的高层并行算法库，其设计灵感直接影响了 C++ 标准库中并行算法的引入。

#### 2.3.1 执行策略（Execution Policy）

Thrust 的核心设计模式是**执行策略**。用户通过指定执行策略来控制算法在哪个后端执行：

```
int const thrust_result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);
```

执行策略的类型系统定义在 `thrust/thrust/execution_policy.h`。`host_t` 和 `device_t` 分别映射到主机和设备后端的具体实现（L42-L43）：

```
using host_t   = thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::detail::par_t;using device_t = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::detail::par_t;
```

全局常量 `thrust::host` 和 `thrust::device` 分别是这两个类型的实例（L203, L248）：

```
inline constexpr detail::host_t host;
```

```
_CCCL_GLOBAL_CONSTANT detail::device_t device;
```

Thrust 支持四种后端系统，通过 `thrust/system/` 的子目录组织：`cpp`（C++ 串行）、`cuda`（GPU）、`omp`（OpenMP）、`tbb`（Intel TBB）。编译时通过 L24-L25 的宏展开包含对应后端的执行策略：

```
#include __THRUST_HOST_SYSTEM_ALGORITH_HEADER_INCLUDE(execution_policy.h)#include __THRUST_DEVICE_SYSTEM_ALGORITH_HEADER_INCLUDE(execution_policy.h)
```

#### 2.3.2 CUDA 后端的 CRTP 类型标签分发

CUDA 后端的执行策略定义在 `thrust/thrust/system/cuda/detail/execution_policy.h`，通过 CRTP（Curiously Recurring Template Pattern）实现类型标签分发。完整的类型层次如下（L38-L62）：

```
// CRTP 基类：所有 CUDA 执行策略继承此模板template <class Derived>struct execution_policy : thrust::execution_policy<Derived>{  using tag_type = tag;  _CCCL_HOST_DEVICE operator tag() const { return {}; }};// tag 同时继承 execution_policy 和 allocator_aware_execution_policystruct tag    : execution_policy<tag>    , thrust::detail::allocator_aware_execution_policy<execution_policy>{};
```

#### 2.3.3 流绑定与异步执行

`execute_on_stream_base` 类（L64-L87）允许用户将算法绑定到特定 CUDA 流：

```
template <class Derived>struct execute_on_stream_base : execution_policy<Derived>{private:  cudaStream_t stream;public:  _CCCL_HOST_DEVICE execute_on_stream_base(cudaStream_t stream_ = cuda_cub::default_stream())      : stream(stream_) {}  _CCCL_HOST_DEVICE Derived on(::cudaStream_t s) const {    Derived result = derived_cast(*this);    result.stream  = s;    return result;  }};
```

面向用户的 `par_t` 类型（L137-L148）组合了执行策略和流绑定：

```
struct par_t    : execution_policy<par_t>    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_base>{  using stream_attachment_type = execute_on_stream;  _CCCL_HOST_DEVICE stream_attachment_type on(::cudaStream_t s) const {    return execute_on_stream(s);  }};
```

此外还提供了 `par_nosync_t`（L153-L171）用于跳过可选同步的场景，用户代码示例：

```
cudaStream_t stream;cudaStreamCreate(&stream);auto nosync_policy = thrust::cuda::par_nosync.on(stream);thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});// for_each 可能在内核完成前返回，用户需手动同步cudaStreamSynchronize(stream);
```

## 三、统一构建系统

> 根目录 `CMakeLists.txt` 是整个项目的构建入口。三个核心库按依赖顺序集成（L101-L103）：

```
add_subdirectory(libcudacxx)  # 基础层，无外部依赖add_subdirectory(cub)          # 依赖 libcudacxxadd_subdirectory(thrust)       # 依赖 libcudacxx 和 CUB
```

构建系统通过 CMake 选项控制可选组件的条件编译（L47-L66）：

| 选项 | 功能 |
| --- | --- |
| `CCCL_ENABLE_UNSTABLE` | 启用实验性组件（cudax） |
| `CCCL_ENABLE_TESTING` | 启用测试 |
| `CCCL_ENABLE_C_PARALLEL` | 启用 C 语言并行库 |
| `CCCL_ENABLE_C_EXPERIMENTAL_STF` | 启用 CUDASTF 实验库 |
| `CCCL_ENABLE_BENCHMARKS` | 启用性能基准测试（NVHPC 编译器强制关闭） |

项目支持两种使用模式（L1-L21）：作为**顶层项目** 独立构建（开发者模式，要求 CMake ≥ 3.21），或通过 `add_subdirectory()`**嵌入** 到其他项目中（用户模式，仅需 CMake ≥ 3.18）：

```
# 3.18 is the minimum for including the project with add_subdirectory.# 3.21 is the minimum for the developer build.cmake_minimum_required(VERSION 3.18)
```

条件组件的集成（L105-L111）：

```
if (CCCL_ENABLE_UNSTABLE)  add_subdirectory(cudax)       # 实验性 CUDA 扩展库endif()if (CCCL_ENABLE_C_PARALLEL OR CCCL_ENABLE_C_EXPERIMENTAL_STF)  add_subdirectory(c)           # C 语言并行库 / CUDASTFendif()
```

## 四、命名空间与版本管理

### 4.1 命名空间包装机制

CCCL 通过宏驱动的命名空间抽象层解决符号冲突问题，定义在 `thrust/thrust/detail/config/namespace.h`。

`THRUST_CUB_WRAPPED_NAMESPACE` 宏可同时包装 `thrust::` 和 `cub::` 命名空间（L31-L33）：

```
#ifdef THRUST_CUB_WRAPPED_NAMESPACE#  define THRUST_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE#endif
```

`THRUST_WRAPPED_NAMESPACE` 宏将 `thrust::` 放入自定义外层命名空间（L42-L49）：

```
#ifdef THRUST_WRAPPED_NAMESPACE#  define THRUST_NS_PREFIX             \    namespace THRUST_WRAPPED_NAMESPACE \    {#  define THRUST_NS_POSTFIX }#  define THRUST_NS_QUALIFIER ::THRUST_WRAPPED_NAMESPACE::thrust#endif
```

此外，当使用 CUDA 设备后端时，Thrust 还会嵌入 **ABI 内联命名空间** 以编码版本号和 CUDA 架构列表（L86-L114），使得不同编译配置产生的符号名天然不同，避免链接冲突：

```
#define THRUST_DETAIL_ABI_NS_BEGIN                                                       \  inline namespace _CCCL_PP_CAT(                                                         \    _CCCL_PP_CAT(_V_, THRUST_VERSION), _CCCL_PP_SPLICE_WITH(_, _SM, __CUDA_ARCH_LIST__)) \  {#define THRUST_DETAIL_ABI_NS_END }
```

最终的 `THRUST_NAMESPACE_BEGIN/END` 宏组合所有层次（L122-L137）：

```
#define THRUST_NAMESPACE_BEGIN \  THRUST_NS_PREFIX             \  namespace thrust             \  {                            \  THRUST_DETAIL_ABI_NS_BEGIN#define THRUST_NAMESPACE_END   \  THRUST_DETAIL_ABI_NS_END     \  } /* end namespace thrust */ \  THRUST_NS_POSTFIX
```

### 4.2 统一版本控制

CCCL 采用 `MMMmmmppp` 格式的统一版本号，定义在 `libcudacxx/include/cuda/std/__cccl/version.h`（当前为 `3004000`，即 3.4.0）：

```
#define CCCL_VERSION 3004000#define CCCL_MAJOR_VERSION (CCCL_VERSION / 1000000)#define CCCL_MINOR_VERSION (((CCCL_VERSION / 1000) % 1000))#define CCCL_PATCH_VERSION (CCCL_VERSION % 1000)
```

此外还有编译时防护（L22-L24）：

```
#if CCCL_PATCH_VERSION > 99#  error "CCCL patch version cannot be greater than 99 for compatibility with Thrust/CUB's MMMmmmpp format."#endif
```

Thrust 和 CUB 保留各自的版本宏（`MMMmmmpp` 格式）以保持向后兼容，但通过 `static_assert`**强制** 与 CCCL 版本同步：

```
#define THRUST_VERSION 300400 // MMMmmmpp// ...static_assert(THRUST_MAJOR_VERSION == CCCL_MAJOR_VERSION, "");static_assert(THRUST_MINOR_VERSION == CCCL_MINOR_VERSION, "");static_assert(THRUST_SUBMINOR_VERSION == CCCL_PATCH_VERSION, "");
```

```
#define CUB_VERSION 300400 // MMMmmmpp// ...static_assert(CUB_MAJOR_VERSION == CCCL_MAJOR_VERSION, "");static_assert(CUB_MINOR_VERSION == CCCL_MINOR_VERSION, "");static_assert(CUB_SUBMINOR_VERSION == CCCL_PATCH_VERSION, "");
```

### 4.3 ABI 版本管理

`cuda::` 命名空间中的符号嵌入了 ABI 版本号（通过 inline namespace），ABI 变更时递增版本号，允许多个 ABI 版本并存。`thrust::` 和 `cub::` 命名空间的 ABI 则不提供稳定性保证——它们的符号名中编码了 CUDA 架构列表，以不同架构编译或以 CUDA/C++ 模式编译都会产生不同符号。

## 五、整体架构总结

![文章插图](/articles/2026-04-09-nvidia-cccl-cuda-core-compute-3088ef5b/images/05.webp)

::: image-caption
这是CCCL（CUDA C++核心库）仓库的架构图，整体分为核心库、实验性组件、基础设施三大模块。核心库中，Thrust（高层并行算法库）依赖CUB（块/线程束/设备原语库），二者与实验区的cudax组件，均依赖底层libcudacxx（C++标准库与硬件抽象层）；实验区还包含c/parallel组件；基础设施则提供CMake构建、命名空间管理、统一版本管理等底层支撑。
:::

> CCCL 的核心本质可以归纳为三点：

1. **分层抽象**：libcudacxx 提供硬件抽象和标准库兼容（`cuda::atomic`、`cuda::barrier`、`cuda::ptx::*`）。CUB 提供 CUDA 特定的高性能原语（`BlockReduce`、`DeviceReduce` 等，依赖 libcudacxx 的原子操作和 PTX 封装），Thrust 提供跨后端的高层算法接口（通过执行策略标签分发到 CUDA/OMP/TBB/CPP 后端）。:mdnice-bold-italic[三层之间依赖关系单向且清晰。]
2. **编译时多态**：从 Thrust 的 CRTP 执行策略标签分发（`execution_policy<Derived>`），到 CUB 的算法变体模板特化（`BlockReduceAlgorithm` 枚举 + 确定性/非确定性 `if constexpr` 分支），再到 libcudacxx 的作用域参数化原子操作（`__scope_to_tag<_Sco>{}` 标签分发），整个库栈大量使用 C++ 模板元编程实现**零运行时开销** 的抽象。
3. **统一治理**：通过单一仓库、统一版本号（`CCCL_VERSION 3004000` + 编译时 `static_assert` 强制 Thrust/CUB 同步）、宏驱动的命名空间包装机制（`THRUST_CUB_WRAPPED_NAMESPACE` + ABI inline namespace），以及模块化的 CMake 构建系统（按依赖顺序 `add_subdirectory`），将三个历史独立的项目整合为一个内聚的开发和分发单元。

整个项目完全是 **header-only** 的，用户无需链接任何预编译库，只需正确设置头文件包含路径即可使用。

::: block-1
综上，CCCL凭借清晰的三层抽象架构、零运行时开销的编译时多态机制，以及统一的工程治理体系，筑牢了CUDA并行计算的底层根基。

它作为主流CUDA项目的核心依赖，源码设计完美平衡了硬件性能与工程化实践。深入理解这套架构，能帮助开发者高效挖掘GPU算力，精准构建高性能并行应用。
:::
