关键词：**CUDA 编程教学**、标准并行算法、共享内存优化、:mdnice-bold-italic[容器化部署]、**渐进式课程设计**

> GPU 编程长期以来被视为一座陡峭的学习悬崖——初学者不仅需要理解线程层次结构、内存空间模型这些抽象概念，还要与编译工具链、运行时环境搏斗。

NVIDIA 的`accelerated-computing-hub`仓库试图填补这个鸿沟：它不是一份简单的示例代码集合，而是一套经过精心编排的、从"Hello GPU"到多 GPU 分布式计算的完整课程体系。更值得关注的是，它的代码呈现了一种"教学驱动的软件工程"——每一个源文件都服务于特定的认知目标，代码复杂度严格递进，配合 Docker 容器实现环境零配置。

![文章插图](/articles/2026-05-03-nvidia-gpu-accelerated-computing-hub-710f8438/images/01.webp)

- **NVIDIA 精选的通用 GPU 编程相关教育资源合集**
- https://github.com/NVIDIA/accelerated-computing-hub
- 4000 字，阅读 20 分钟，**播客 16 分钟**

相关推荐

- [NVIDIA CCCL (CUDA Core Compute Libraries) 核心架构深度分析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899677&idx=1&sn=7cbf73b7430dba1208d735f80a832ca1&scene=21#wechat_redirect)
- [暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)
- [暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)

> 该仓库给出了一份教程大纲，汇总了 NVIDIA 加速计算中心所有可在 NVIDIA Brev、Google Colab 上运行的交互式 GPU 编程教程。

在下表清晰列出**教程内容主题**、配套**Docker Compose 配置文件**、适配的**Brev 硬件实例型号**，以及支持部署运行的**Brev 服务提供商和系统环境要求**，方便开发者快速匹配硬件与平台开展学习实操。其中，NVIDIA Brev 是 NVIDIA 面向人工智能原型开发与落地交付的开发者云平台。

| 内容 | Brev 实例 | Brev 提供程序 |
| --- | --- | --- |
| CUDA Tile 教程[1] | 1xB300、1xB200 或 1 块 RTX Pro6000 | 任意 Blackwell 服务商；目前均未提供灵活端口 |
| CUDA C++ 教程[2] | L40S、L4 或 T4 | Crusoe 或其他支持灵活端口的设备 |
| 标准并行性教程[3] | 4 个 L4、2 个 L4、2 个 L40S 或 1 个 L40S | GCP、AWS 或其他支持灵活端口，且搭载 Linux 6.1.24+ / 6.2.11+ / 6.3+（适配 HMM）的平台 |
| 加速 Python 教程[4] | L40S、L4 或 T4；分布式场景使用 4xL4 或 2xL4 | Crusoe 或其他支持灵活端口的设备 |
| nvmath-python 教程[5] | 4 个 L4、2 个 L4、2 个 L40S 或 1 个 L40S | Crusoe 或其他支持灵活端口的设备 |
| CUDA Python 教程 - CuPy、cuDF、CCCL 及内核（8 小时版）[6] | L40S、L4 或 T4 | Crusoe 或其他支持灵活端口的设备 |
| CUDA Python 教程 - cuda.core 与 CCCL（2 小时版）[7] | L40S、L4 或 T4 | Crusoe 或其他支持灵活端口的设备 |
| PyHPC 教程 - NumPy、CuPy 与 mpi4py（4 小时版）[8] | 4 个 L4、2 个 L4、2 个 L40S 或 1 个 L40S | Crusoe 或其他支持灵活端口的设备 |
| GPU 部署教程[9] | - | - |

> 本文将剖析这个仓库，核心教学代码的细节，以及它如何通过 Standard Parallelism 这条"零 CUDA 代码"路径，为传统 HPC 开发者打开 GPU 加速的大门。

## 本文目录

- 快速上手
- 一、仓库整体架构与课程编排逻辑
  - 1.1 三条并行的学习路径
  - 1.2 CUDA C++课程的三阶递进
- 二、从单线程到多线程：Kernel 编程的渐进式代码设计
  - 2.1 起点：一个单线程的 GPU kernel
  - 2.2 引入线程并行：stride loop 模式
  - 2.3 kernel 启动语法的最小化演示
- 三、共享内存与线程同步：从原理到优化的完整闭环
  - 3.1 共享内存的最小可运行示例
  - 3.2 实战：基于共享内存的直方图优化
  - 3.3 CUB 协作算法：站在巨人肩膀上
- 四、Standard Parallelism：零 CUDA 代码的 GPU 加速之路
  - 4.1 用标准 C++算法直接跑 GPU
  - 4.2 二维热传导方程：stdpar 的 HPC 实战
- 五、容器化基础设施：教学环境的工程保障
  - 5.1 Docker Compose 的多服务编排
- 总结

![文章插图](/articles/2026-05-03-nvidia-gpu-accelerated-computing-hub-710f8438/images/02.webp)

## 快速上手

克隆仓库并通过 Docker Compose 一键启动 CUDA C++教程的 JupyterLab 环境：

```
git clone https://github.com/NVIDIA/accelerated-computing-hub.gitcd accelerated-computing-hub/tutorials/cuda-cpp/brevdocker compose up
```

启动后访问 `http://127.0.0.1:8888` 即可进入交互式 Notebook。也可直接通过 Google Colab 在线运行，无需本地 GPU。更多部署方式（NVIDIA Brev 云实例、GPU 型号选择等），参考 `docs/brev.md`。

## 一、仓库整体架构与课程编排逻辑

### 1.1 三条并行的学习路径

> 整个仓库围绕三大教学模块展开，对应三类目标受众：

| 路径 | 目录 | 目标受众 |
| --- | --- | --- |
| CUDA C++ | `tutorials/cuda-cpp/` | 想深入理解 GPU 硬件模型的系统程序员 |
| Standard Parallelism | `tutorials/stdpar/` | 希望保持代码可移植性的 HPC 开发者 |
| Accelerated Python | `tutorials/accelerated-python/` | 数据科学家和 Python 开发者 |

三条路径相互独立又共享相同的基础设施（Docker 镜像、Brev 配置、测试框架），这种设计使得不同背景的开发者可以找到最契合自己知识结构的入口。

### 1.2 CUDA C++课程的三阶递进

> CUDA C++教程被切分为三个精心设计的阶段：

1. **第一阶段** （01.xx）：并行算法——不写 kernel，仅通过`thrust`/CCCL 的高层抽象理解执行空间与内存空间
2. **第二阶段** （02.xx）：异步与流——理解 GPU 的异步本质、CUDA Stream、Pinned Memory
3. **第三阶段** （03.xx）：自定义 Kernel——从最朴素的单线程 kernel 逐步演进到共享内存和协作算法

这种"先会用、再理解、最后深入"的教学哲学，在代码层面体现得淋漓尽致。

## 二、从单线程到多线程：Kernel 编程的渐进式代码设计

### 2.1 起点：一个单线程的 GPU kernel

> 教程的 kernel 部分从一个刻意"低效"的实现开始：

```
// 来源：tutorials/cuda-cpp/notebooks/03.02-Kernels/Sources/simple-kernel.cpp#include "ach.h"__global__ void single_thread_kernel(ach::temperature_grid_f in, float *out) {  for (int id = 0; id < in.size(); id++) {    out[id] = ach::compute(id, in);  }}void simulate(ach::temperature_grid_f temp_in, float *temp_out,              cudaStream_t stream) {  single_thread_kernel<<<1, 1, 0, stream>>>(temp_in, temp_out);}
```

注意`<<<1, 1>>>`——仅启动一个 block 内的一个线程。这段代码的教学意图极为明确：让学习者首先理解 kernel 的语法结构和启动方式，暂时搁置并行性的讨论。

- `ach::temperature_grid_f`是教程封装的温度网格类型
- `ach::compute`封装了 stencil 计算逻辑，使学习者聚焦于并行化改造本身

### 2.2 引入线程并行：stride loop 模式

> 接下来代码进化为多线程版本：

```
// 来源：tutorials/cuda-cpp/notebooks/03.02-Kernels/Sources/block-256-kernel.cpp#include "ach.h"constint number_of_threads = 256;__global__ void block_kernel(ach::temperature_grid_f in, float *out) {int thread_index = threadIdx.x;for (int id = thread_index; id < in.size(); id += number_of_threads) {    out[id] = ach::compute(id, in);  }}void simulate(ach::temperature_grid_f temp_in, float *temp_out,              cudaStream_t stream) {  block_kernel<<<1, number_of_threads, 0, stream>>>(temp_in, temp_out);}
```

这里展示了经典的**stride loop** （跨步循环）模式：每个线程从自己的`threadIdx.x`出发，以线程总数为步长遍历所有数据。教程还故意提供了一个`number_of_threads = 2048`的"错误版本"（`failed-block-kernel.cpp`），让学习者亲身体验超过硬件线程块上限（1024）时的运行时错误——这种"先犯错、再纠正"的教学手法贯穿整个仓库。

### 2.3 kernel 启动语法的最小化演示

> 为了单独解释`<<<blocks, threads, shmem, stream>>>`这个 CUDA 特有语法，教程提供了一个极简的"Hello World" kernel：

```
// 来源：tutorials/cuda-cpp/notebooks/03.02-Kernels/Sources/kernel-launch.cpp#include <cstdio>__global__ void kernel(int value) {std::printf("value on device = %d\n", value);}int main() {int blocks_in_grid = 1;int threads_in_block = 1;  cudaStream_t stream = 0;  kernel<<<blocks_in_grid, threads_in_block, 0, stream>>>(42);  cudaStreamSynchronize(stream);}
```

每个启动参数都被赋予具体的变量名，自文档化程度极高。这比直接展示`kernel<<<1,1>>>(42)`要清晰得多——变量名本身就是注释。

## 三、共享内存与线程同步：从原理到优化的完整闭环

### 3.1 共享内存的最小可运行示例

> 共享内存是 CUDA 性能优化的核心武器之一。教程用 4 个线程的极简案例引入概念：

```
// 来源：tutorials/cuda-cpp/notebooks/03.05-Shared-Memory/Sources/simple-shmem.cpp#include <cstdio>__global__ void kernel(){  __shared__ int shared[4];  shared[threadIdx.x] = threadIdx.x;  __syncthreads();if (threadIdx.x == 0)  {    for (int i = 0; i < 4; i++) {      std::printf("shared[%d] = %d\n", i, shared[i]);    }  }}int main() {  kernel<<<1, 4>>>();  cudaDeviceSynchronize();return0;}
```

这段代码精准地展示了三个关键点：

- `__shared__`声明将数组放入片上共享内存；
- `__syncthreads()`确保所有线程完成写入后再读取；
- 仅由 0 号线程执行打印避免输出混乱。

学习者可以立即编译运行，观察到`shared[0]=0, shared[1]=1, ...`的输出，建立对共享内存可见性的直觉。

### 3.2 实战：基于共享内存的直方图优化

> 掌握基础后，教程直接进入实战——使用共享内存优化直方图 kernel：

```
// 来源：tutorials/cuda-cpp/notebooks/03.05-Shared-Memory/Solutions/shmem.cpp__global__ void histogram_kernel(cuda::std::span<float> temperatures,                                 cuda::std::span<int> histogram) {  __shared__ int block_histogram[10];if (threadIdx.x < 10) {    block_histogram[threadIdx.x] = 0;  }  __syncthreads();int cell = blockIdx.x * blockDim.x + threadIdx.x;int bin = static_cast<int>(temperatures[cell] / 10);cuda::atomic_ref<int, cuda::thread_scope_block> block_ref(      block_histogram[bin]);  block_ref.fetch_add(1);  __syncthreads();if (threadIdx.x < 10) {    cuda::atomic_ref<int, cuda::thread_scope_device> ref(        histogram[threadIdx.x]);    ref.fetch_add(block_histogram[threadIdx.x]);  }}
```

这段代码的教学价值极高：

- **两级归约**：先在 block 级别的共享内存中累加（使用`thread_scope_block`的原子操作），再将各 block 结果汇总到全局内存（使用`thread_scope_device`的原子操作）
- **现代 CUDA C++风格**：使用`cuda::std::span`代替裸指针，使用`cuda::atomic_ref`代替传统的`atomicAdd`，展示了 libcu++的现代接口
- **性能意义**：block 内共享内存的原子操作比全局内存快一个数量级以上，因为它避免了跨 SM 的互联争用

### 3.3 CUB 协作算法：站在巨人肩膀上

> 教程最终引入 CUB 库的 block-level 原语，展示工业级的协作归约：

```
// 来源：tutorials/cuda-cpp/notebooks/03.06-Cooperative-Algorithms/Sources/block-sum.cpp#include <cub/block/block_reduce.cuh>constexprint block_threads = 128;__global__ void block_sum(){usingblock_reduce_t = cub::BlockReduce<int, block_threads>;usingstorage_t = block_reduce_t::TempStorage;  __shared__ storage_t storage;int block_sum = block_reduce_t(storage).Sum(threadIdx.x);if (threadIdx.x == 0)  {    printf("block sum = %d\n", block_sum);  }}int main() {  block_sum<<<1, block_threads>>>();  cudaDeviceSynchronize();return0;}
```

CUB 的`BlockReduce`内部实现了 warp-level shuffle 指令和多轮归约树，性能远超手写的 shared memory 归约。:mdnice-bold-italic[教程的编排逻辑是：先让你手写 shared memory 理解原理，再告诉你"实际工程中应该用 CUB"——这种从原理到最佳实践的过渡非常自然。]

## 四、Standard Parallelism：零 CUDA 代码的 GPU 加速之路

### 4.1 用标准 C++算法直接跑 GPU

> `stdpar`教程展示了一条完全不同的 GPU 编程路径——通过 ISO C++标准并行算法实现 GPU 加速：

```
// 来源：tutorials/stdpar/notebooks/cpp/lab1_daxpy/solutions/exercise5.cpp#include <algorithm>#include <execution>#include <ranges>#include <vector>void initialize(std::vector<double> &x, std::vector<double> &y) {std::for_each_n(std::execution::par, std::views::iota(0).begin(), x.size(),    [x = x.data()](int i) { x[i] = (double)i; });std::fill_n(std::execution::par, y.data(), y.size(), 2.);}void daxpy(double a, std::vector<double> const &x, std::vector<double> &y) {std::for_each_n(std::execution::par,                  std::views::iota(0).begin(), x.size(),    [a, x = x.data(), y = y.data()](int i) {        y[i] += a * x[i];  });}
```

这段代码**没有任何 CUDA 关键字** ——没有`__global__`，没有`<<<>>>`，没有`cudaMalloc`。仅通过传递`std::execution::par`执行策略，:mdnice-bold-italic[配合`nvc++ -stdpar=gpu`编译选项，算法就自动在 GPU 上执行。教程还通过对比四种编译器（g++、clang++、nvc++ CPU、nvc++ GPU）的运行结果，让学习者直观感受同一代码在不同执行后端上的性能差异。]

### 4.2 二维热传导方程：stdpar 的 HPC 实战

教程进一步展示了如何用标准算法解决真实的科学计算问题——二维热传导方程的并行 stencil 计算：

```
// 来源：tutorials/stdpar/notebooks/cpp/lab2_heat/exercise2.cppdouble apply_stencil(double* u_new, double* u_old, grid g, parameters p) {auto xs = std::views::iota(g.x_begin, g.x_end);auto ys = std::views::iota(g.y_begin, g.y_end);auto ids = std::views::cartesian_product(xs, ys);returnstd::transform_reduce(    std::execution::par, ids.begin(), ids.end(),    0., std::plus{}, [u_new, u_old, p](auto idx) {      auto [x, y] = idx;      return stencil(u_new, u_old, x, y, p);  });}
```

`std::views::cartesian_product`生成二维索引空间，`std::transform_reduce`在并行执行 stencil 的同时完成能量求和归约——这正是 map-reduce 模式在标准 C++中的优雅表达。结合 MPI 实现多 GPU 通信，这段代码展示了标准并行算法在工业级 HPC 应用中的完整实力。

## 五、容器化基础设施：教学环境的工程保障

### 5.1 Docker Compose 的多服务编排

教程使用精心设计的 Docker Compose 配置实现一键部署：

```
# 来源：tutorials/cuda-cpp/brev/docker-compose.ymlservices:base:    <<:[*gpu-config,*common-service]    image:*image    entrypoint:["/accelerated-computing-hub/brev/entrypoint.bash","base"]jupyter:    <<:[*gpu-config,*common-service,*persistent-service]    image:*image    entrypoint:["/accelerated-computing-hub/brev/entrypoint.bash","jupyter"]    ports:      -"127.0.0.1:8888:8888"nsys:    image:nvcr.io/nvidia/devtools/nsight-streamer-nsys:2026.1.1    ports:      -"127.0.0.1:8080:8080"
```

一次`docker compose up`同时启动 JupyterLab、Nsight Systems 性能分析器和 Nsight Compute 分析器三个服务。学习者在 Notebook 中写完 kernel 后，可以直接在浏览器中查看 GPU timeline 和指令级分析——这种"编码-分析"的即时反馈循环是传统教学难以提供的。

## 总结

> NVIDIA 的`accelerated-computing-hub`不仅仅是一个代码仓库，它代表了一种**教学驱动的软件工程范式**：

1. **认知负载递进**：每个代码文件只引入一个新概念，从`<<<1,1>>>`到`cub::BlockReduce`的跨度被分解为十余个可消化的步骤
2. **多入口设计**：三条路径（CUDA C++/stdpar/Python）让不同背景的开发者都能找到起点
3. **环境即代码**：Docker Compose + 预构建镜像消除了"环境配置"这个最大的学习障碍
4. **现代 C++优先**：全面使用`cuda::std::span`、`cuda::atomic_ref`、`std::mdspan`等现代接口，而非传统的 C 风格 CUDA API

::: block-1
对于想要系统学习 GPU 编程的开发者来说，这个仓库提供了目前最具结构化、最可复现的开源学习路径。而对于教学设计者而言，它展示了如何通过代码的渐进式组织来实现复杂知识的有效传递。
:::

参考资料

[1] CUDA Tile 教程: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-tile/README.md*

[2] CUDA C++ 教程: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/README.md*

[3] 标准并行性教程: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/README.md*

[4] 加速 Python 教程: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/README.md*

[5] nvmath-python 教程: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/README.md*

[6] CUDA Python 教程 - CuPy、cuDF、CCCL 及内核（8 小时版）: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/syllabi/cuda_python__cupy_cudf_cccl_kernels__8_hours.ipynb*

[7] CUDA Python 教程 - cuda.core 与 CCCL（2 小时版）: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/syllabi/cuda_python__cuda_core_cccl__2_hours.ipynb*

[8] PyHPC 教程 - NumPy、CuPy 与 mpi4py（4 小时版）: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/syllabi/pyhpc__numpy_cupy_mpi4py__4_hours.ipynb*

[9] GPU 部署教程: *https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/gpu-deployment/gpu-deployment-from-scratch.md*
