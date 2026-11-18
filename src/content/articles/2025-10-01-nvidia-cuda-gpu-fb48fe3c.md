关键词：CUDA、GPU性能优化、全局内存访问、合并内存访问、跨步访问、Nsight Compute

> 读完本文，最大收获在两点：一是理解CUDA中**全局内存的合并访问是解锁 GPU 性能的关键**，需避免跨步访问，二是文章末尾提到**始终** 用 Nsight Compute 验证访问模式，**确保** 线程束连续线程访问连续内存地址，以最大化内存带宽利用率。

- Unlock GPU Performance: Global Memory Access in CUDA, Sep 29, 2025 By Rajeshwari Devaramani and Jonathan Bentz
- https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/
- 本文 5842 字，阅读需 18 分钟，播客 10 分钟

相关推荐

Apple Silicon LLM 推理全方位深度剖析：与 NVIDIA GPU 从性能到成本的终极对决！

Arm Mali GPU 四大微架构概述深入浅出 | 谈谈MNN GPU性能优化策略

在编写 GPU 内核时，内存管理是需要考虑的最重要性能特征之一。本文将带你了解关于**全局内存及其性能你需要知道的重要方面**。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/01.webp)

## 本文目录

- 一、什么是全局内存
  - 示例：动态分配、数据传输、内核执行与清理
- 二、全局内存的合并访问与性能剖析
- 三、跨步访问与带宽变化
- 三、多维数组下内存访问与剖析
- 四、总结：1 个务必与 1 个始终

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/02.webp)

## 一、什么是全局内存

> CUDA 设备上有多种类型的内存，每种内存的作用域（指可访问该内存的线程范围）、生命周期（指内存从分配到释放的存在时间）和缓存行为（指内存数据在缓存中的存储、读取和更新规则）都不同。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/03.webp)

::: image-caption
A100 GPU 中的内存层级结构图
:::

全局内存（也称为设备内存）是 CUDA 设备上的主内存空间，位于设备 DRAM（动态随机存取存储器，是 GPU 上用于临时存储数据的高速内存）中，其功能与 CPU 系统中的 RAM（随机存取存储器）类似。**“全局”一词指的是它的作用域，主机和内核网格中的所有线程都可对其进行访问和修改。**

可在全局作用域下使用`__device__`声明说明符静态声明全局内存，也可使用`cudaMalloc()`或`cudaMallocManaged()`等 CUDA 运行时 API（应用程序编程接口，是一组预先定义的函数，用于实现不同软件组件间的交互）动态分配全局内存。数据可通过`cudaMemcpy()`从主机传输到设备，也可通过`cudaFree()`释放。这些分配的内存会一直存在，直到被释放。

全局内存也可通过统一内存（指一种能让 CPU 和 GPU 共享同一内存地址空间，无需显式在主机和设备间传输数据的内存管理技术）进行分配和释放。关于全局内存的分配、释放以及与设备间的数据传输，涉及一些复杂内容，我们将在后续文章中介绍。**本文将重点讨论在 CUDA 内核中使用全局内存对性能的影响。**

典型使用模式的一个简单示例流程如下：主机在启动内核前分配并初始化全局内存，接着执行内核，CUDA 线程从全局内存读取数据并将计算结果写回全局内存，最后在核执行完成后，主机获取结果。

### 示例：动态分配、数据传输、内核执行与清理

```
// 主机分配全局内存float* d_input;float* d_output;cudaMalloc(&d_input, n * sizeof(float));cudaMalloc(&d_output, n * sizeof(float));// 将数据传输到设备cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);// 调用内核在设备上执行操作someKernel<<<1024, 1024>>>(d_input, d_output, n);// 将结果复制回主机cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);// 清理（释放已分配的内存）cudaFree(d_input);cudaFree(d_output);
```

## 二、全局内存的合并访问与性能剖析

> 在深入探讨全局内存访问性能之前，我们需要进一步**理解 CUDA 执行模型**。

我们已经讨论过线程是如何分组为线程块（指 GPU 中由多个线程组成的、可共享局部内存的基本计算单元）的，而线程块又会被分配到设备的多处理器（指 GPU 中负责执行线程块的计算核心单元）上。在执行过程中，线程会被进一步细分为线程束。GPU 的多处理器以 SIMT（单指令多线程，指一个指令单元同时向多个线程单元发送相同指令，每个线程单元处理不同数据的执行模式）方式为每个线程束执行指令。当前所有支持 CUDA 的 GPU 的线程束大小（实际即 SIMT 宽度）均为 32 个线程。

> #### 线程块 vs. 线程束(warp)
>
> 线程块（Thread Block）不是线程束（Warp），二者是 CUDA 执行模型中不同层级、不同作用的线程分组单元，存在明确的层级关系和功能差异，需结合线程的 “分组逻辑” 和 “执行逻辑” 区分。

::: block-1
单个线程 → 组成线程块（开发者定义的逻辑分组，共享局部内存）→ 拆分为线程束（硬件定义的执行单元，固定 32 线程）→ 由 SM 按线程束执行指令。

简言之：线程块是 “逻辑上的分组”，线程束是 “执行上的最小单元”，二者不是同一概念。
:::

在 CUDA 中访问全局内存时，有一个关键方面需要关注：**同一线程束内不同线程所访问的内存地址之间的关系**。因为，:mdnice-bold-italic[这些内存访问模式直接影响内存访问效率和应用程序的整体性能。]

全局内存通过 32 字节的内存事务（指内存控制器与内存之间一次完整的数据传输操作）进行访问。当一个 CUDA 线程请求从全局内存读取数据时，该线程束中所有线程的内存访问请求会被合并成尽可能少的内存事务。所需的内存事务数量取决于每个线程访问的数据字（指一次内存访问中可读取或写入的数据单位，通常与数据类型的字节数一致，如 float 类型为 4 字节）大小以及各线程内存地址的分布情况。

```
__global__ void coalesced_access(float* input, float* output, int n) {    int tid = blockIdx.x * blockDim.x + threadIdx.x;    if (tid < n) {        // 每个线程访问连续的4字节数据        output[tid] = input[tid] * 2.0f;    }}
```

以上代码展示了一种场景：**线程束内的连续线程访问连续的 4 字节数据元素**，形成了**【最优】的内存访问模式**。一个线程束发出的所有加载请求（指线程从内存读取数据到寄存器的操作）可由内存中的 4 个 32 字节扇区（指内存中按固定大小划分的连续数据块，此处为 32 字节）来满足，这样就能实现内存带宽的最高效利用。下面图 1 展示了每个线程如何访问连续内存中的 4 字节数据元素。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/04.webp)

::: image-caption
图1. 合并内存访问模式，展示了一个线程束的线程（箭头所示）在四个32字节的扇区中访问一个连续的128字节内存块
:::

该图展示了一个线程束的线程（用箭头表示）如何**访问一个连续的 128 字节内存块**，具体是**通过 4 个 32 字节扇区实现** 的。

```
__global__ void uncoalesced_access(float* input, float* output, int n) {    int tid = blockIdx.x * blockDim.x + threadIdx.x;    if (tid < n) {        // 以32（即128字节）的跨步进行访问，通过取模操作确保索引在有效范围内        int scattered_index = (tid * 32) % n;        output[tid] = input[scattered_index] * 2.0f;    }}
```

相反，如果线程以较大的跨步访问内存，那么每次内存事务获取的数据会远多于实际所需，如上面代码。对于每个线程请求的 4 字节元素，全局内存会读取一整个 32 字节的段，而传输的数据中大部分都未被使用。图 2 展示了这种模式的一个示例。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/05.webp)

::: image-caption
图 2：非合并内存访问模式。该图展示了每个线程（用箭头表示）如何访问不同 32 字节内存扇区中的数据
:::

**接下来，我们将使用 NVIDIA Nsight Compute** （简称 NCU，是 NVIDIA 推出的一款用于分析和优化 CUDA 内核性能的专业 profiling 工具，可提供详细的内存、指令、缓存等性能指标）来**分析这两个对比鲜明的 CUDA 内核的内存访问模式**。NCU 提供了强大的指标，可用于量化内存访问模式。

要开始对一个内核进行性能分析，我们通常会运行以下命令：

```
ncu --set full --print-details=all ./a.out
```

该命令会收集所有可用的性能分析部分的数据，包括内存、指令、启动、占用率（指 GPU 多处理器的计算资源被线程块利用的比例）、缓存等。但当我们专门关注内存访问效率时，会将范围缩小到那些可量化内存工作负载模式的指标。若要仅隔离出与内存工作负载相关的细节，使用以下命令会更合适：

```
ncu --section MemoryWorkloadAnalysis_Tables --print-details=all ./a.out
```

下面是该命令的输出结果（为清晰起见已进行简化）：

```
coalesced_access(float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9 uncoalesced_access(float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9   Section: Memory Workload Analysis Tables   OPT   Est. Speedup: 83%         The memory access pattern for global loads from DRAM might not be optimal. On average, only 4.0 of the 32         bytes transmitted per sector are utilized by each thread. This applies to the 100.0% of sectors missed in         L2. This could possibly be caused by a stride between threads. Check the Source Counters section for         uncoalesced global loads.
```

从输出结果中可以看到，**NCU 已识别出“uncoalesced_access** ”内核在全局加载（指线程从全局内存读取数据的操作）方面**存在性能改进空间**，实际上，结果显示每个线程平均仅利用了所读取的每个 32 字节扇区中的 4 字节数据。**NCU 甚至提示“这可能是由线程之间的跨步导致的”。**

> 我们设置这个问题的目的就是为了**同时展示内存性能的优劣情况**，因此这个结果并不意外。为了更深入地了解，我们可以看看 NCU 还能提供哪些其他类型的内存分析表（https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables）。

由于 NCU 的初始输出指出了从 DRAM 加载数据时存在问题，接下来我们将尝试使用以下命令更深入地分析 DRAM 统计数据：

```
ncu --metrics group:memory__dram_table ./a.out
```

```
coalesced_access(float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9   Section: Command line profiler metrics   Metric Name                                 Metric Unit Metric Value   dram__bytes_read.sum                        Mbyte       268.44   dram__bytes_read.sum.pct_of_peak_sustained_elapsed            %        46.76   dram__bytes_read.sum.per_second                 Gbyte/s     159.76   dram__bytes_write.sum                       Mbyte       248.50   dram__bytes_write.sum.pct_of_peak_sustained_elapsed           %        43.28   dram__bytes_write.sum.per_second                Gbyte/s     147.89   dram__sectors_read.sum                      sector      8,388,900   dram__sectors_write.sum                     sector      7,765,572 uncoalesced_access(float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9   Section: Command line profiler metrics   Metric Name                                 Metric Unit Metric Value   dram__bytes_read.sum                        Gbyte       2.15   dram__bytes_read.sum.pct_of_peak_sustained_elapsed            %        84.92   dram__bytes_read.sum.per_second                 Gbyte/s     290.16   dram__bytes_write.sum                       Mbyte       263.70   dram__bytes_write.sum.pct_of_peak_sustained_elapsed           %        10.43   dram__bytes_write.sum.per_second                Gbyte/s     35.63   dram__sectors_read.sum                      sector      67,110,368   dram__sectors_write.sum                     sector      8,240,680
```

从这个结果中，我们可以看到**两个内核在`dram__sectors_read.sum`（DRAM 读取扇区总数）指标上存在巨大差**。我们的两个内核都是读取一个数组，然后将结果写回同一个数组，因此理论上读取和写入的数据量应该相同，但在非合并访问的情况下，我们可以看到`sectors_read`（读取扇区数）与`sectors_write`（写入扇区数）之间存在 8 倍的差异。

现在，让我们使用以下命令分析 L1（一级缓存，是 GPU 中与计算核心距离最近、速度最快的缓存，用于存储频繁访问的数据）的行为：

```
ncu --metrics group:memory__first_level_cache_table ./a.out
```

该命令会输出大量信息（此处已省略部分内容），但如果你实际运行该命令，**关键是要关注两个内核之间存在差异的指标**。其中有两个指标我们需要进一步研究：`l1tex_t_requests_pipe_lsu_mem_global_op_ld.sum`和`l1tex_t_sectors_pipe_lsu_mem_global_op_ld.sum`。NCU 提供了一个表格（https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder），可帮助你理解这些指标所收集的信息：**第一个指标本质上是发出的内存请求数量，第二个指标是获取的扇区数量。**

如果我们只想收集这两个指标的数据，可以使用以下命令：

```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./a.out
```

得到的输出结果如下：

```
coalesced_access(float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 9.0    Section: Command line profiler metrics    Metric Name                                 Metric Unit Metric Value    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  2097152    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     sector      8388608  uncoalesced_access(float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 9.0    Section: Command line profiler metrics    Metric Name                                 Metric Unit Metric Value    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  2097152    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     sector     67108864
```

**在合并访问内核中，请求数与扇区数的比例为 1:4**，这与我们的预期一致。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/04.webp)

::: image-caption
图1. 合并内存访问模式，展示了一个线程束的线程（箭头所示）在四个32字节的扇区中访问一个连续的128字节内存块
:::

回顾图 1，我们**展示了一次完美的合并内存事务（128 字节）需要 4 个 32 字节扇区**。:mdnice-bold-italic[从内存中获取的每个字节都被内核使用，实现了 100%的内存带宽效率。]

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/05.webp)

::: image-caption
图 2：非合并内存访问模式
:::

**在非合并访问内核中，请求数与扇区数的比例为 1:32**，这也符合我们的预期——回顾图 2，每个线程都会从不同的 32 字节扇区中请求 4 字节数据。因此，一个线程束发出的每个请求都需要获取 32 个扇区。虽然内存系统会获取 32 个扇区（总共 1024 字节），但每个线程仅需要从其对应的扇区中获取 4 字节数据。

这种 8 倍的效率差异对 GPU 性能有着深远的影响，因为内存带宽往往决定了 GPU 内核的最终性能上限。有关性能分析的更多信息（包括内存扇区相关内容），可参考《性能分析指南》章节：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id51。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/06.webp)

::: image-caption
有关性能分析的更多信息（包括内存扇区相关内容），可参考《性能分析指南》章节：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id51
:::

## 三、跨步访问与带宽变化

> 我们再来看看跨步对内存带宽的影响。在 CUDA 内存访问模式的背景下，跨步指的是**一个线程束的线程所访问的连续内存地址之间的距离** （以数组元素或字节为单位）。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/07.webp)

::: image-caption
图 3：GH200 GPU 上跨步与带宽的关系。该图展示了在 GH200 GPU 上，当跨步从 0 到 31 变化时，带宽呈现下降趋势。当跨步较大时，有效带宽会显著降低。当一个线程束的线程访问物理内存中距离较远的地址时，硬件无法高效地合并这些访问请求。
:::

这类内核在不同访问跨步下的带宽测量结果如图 3 所示。该图并非旨在展示可实现的最大带宽，而只是为了说明当全局内存访问存在跨步时，一个简单内核的带宽会如何变化。

## 三、多维数组下内存访问与剖析

> 再来看多维数组（即矩阵）的内存访问问题。

为了获得最佳性能并实现合并内存访问，与一维数组的情况类似，**连续线程应访问数组中的连续元素**。

在 CUDA 内核中使用二维或三维线程块时，线程会按 X 索引（即`threadIdx.x`）优先的顺序线性排列，其次是 Y 索引（`threadIdx.y`），最后是 Z 索引（`threadIdx.z`）。例如，如果我们有一个大小为`(4,2)`的二维线程块，那么线程的排列顺序为：`(0,0)(1,0)(2,0)(3,0)(0,1)(1,1)(2,1)(3,1)`。

在 CUDA 中访问二维数据时，通常会使用二维线程块。当我们考虑使用二维线程块访问以一维内存数组形式存储的矩阵时，由于 C++采用行优先（指多维数组在内存中按行的顺序依次存储，即先存储第一行的所有元素，再存储第二行，以此类推）的方式存储二维数据，因此按行访问是连续的。**如果我们能让连续线程访问一行中的连续内存地址，那么这些访问就是高效的（合并访问）**，而按列访问则是低效的（跨步访问，非合并访问）。

```
__global__ void coalesced_matrix_access(float* matrix, int width, int height){    int row = blockIdx.y * blockDim.y + threadIdx.y;    int col = blockIdx.x * blockDim.x + threadIdx.x;    if (row < height && col < width) {        int idx = row * width + col;          // 行优先存储 ⇒ 合并访问        matrix[idx] = matrix[idx] * 2.0f + 1.0f;    }}
```

由于要实现合并访问，线程束内连续的`threadIdx.x`值对应的线程应访问连续的内存元素，因此`threadIdx.y`值相同的线程应访问矩阵的同一行。这样可以确保当一个线程束中的线程访问矩阵元素时，它们遵循自然的行优先内存布局，从而实现高效的合并内存事务，最大限度地提高内存带宽利用率。

对于`coalesced_matrix_access`合并访问内核，其内存访问模式之所以能实现高效的合并访问，是因为在行优先存储顺序下，线程索引被映射到了矩阵坐标。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/08.webp)

::: image-caption
图 4：二维合并访问。展示了二维线程块如何映射到二维矩阵，以及如何映射到矩阵所在的线性内存。连续线程访问行中的连续元素，而这些元素在内存中是连续的。
:::

在这里，每个线程块的 x 维度（`threadIdx.x`）被分配给矩阵的列索引，这意味着**当一个线程束内连续线程的`threadIdx.x`值递增时，它们会访问矩阵同一行中的连续列（如图 4 所示）**。由于行优先顺序会将同一行中的元素存储在连续的内存地址中，因此按行访问可使线程束中的每个线程都访问连续的内存地址。

对于下面展示的非合并访问内核（`uncoalesced_matrix_access`），其内存访问模式会导致低效的非合并访问。

```
__global__ void uncoalesced_matrix_access(float* matrix, int width, int height){    int row = blockIdx.y * blockDim.y + threadIdx.y;    int col = blockIdx.x * blockDim.x + threadIdx.x;    if (row < height && col < width) {        int idx = col * height + row;         // 列优先计算索引 ⇒ 非合并访问        matrix[idx] = matrix[idx] * 2.0f + 1.0f;    }}
```

在这里，为了说明问题，该内核通过使用`col * height + row`的索引计算方式，人为地将行优先存储的矩阵当作列优先存储的矩阵来访问。这意味着，当一个线程束内连续线程的`threadIdx.x`值（对应列索引）递增时，它们访问的元素在列优先布局中是连续的，但在实际的行优先内存布局中却是跨步的。

![文章插图](/articles/2025-10-01-nvidia-cuda-gpu-fb48fe3c/images/09.webp)

::: image-caption
图 5：二维非合并访问。该图展示了二维线程块如何映射到二维矩阵，以及如何映射到矩阵所在的线性内存。连续线程访问列中的连续元素，而这些元素在内存中是不连续的。
:::

:mdnice-bold-italic[由于数据实际是以行优先顺序存储的，却采用列优先的索引方式进行访问，连续线程最终会访问间隔为`height`个元素的内存地址，形成较大的跨步模式，这使得 GPU 无法将这些访问合并成高效的事务（如图 5 所示）]。这种存储顺序与访问模式的不匹配，会导致全局内存带宽利用率大幅下降。

我们可以通过分析下面的性能分析结果来观察这种行为：

```
coalesced_matrix_access(float *, int, int) (512, 512, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 9.0    Section: Command line profiler metrics    Metric Name                                 Metric Unit Metric Value    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum  8388608    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     sector     33554432  uncoalesced_matrix_access(float *, int, int) (512, 512, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 9.0    Section: Command line profiler metrics    Metric Name                                 Metric Unit Metric Value    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum  8388608    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     sector    268435456
```

两个内核产生的内存请求数量完全相同（均为 8,388,608），但合并访问版本仅需要 33,554,432 个扇区，而非合并访问版本则需要 268,435,456 个扇区。这意味着：

- 合并访问内核的“每请求扇区数”比例为 4，而非合并访问内核为 32。合并访问内核每请求仅需 4 个扇区的低比例，表明其内存合并效率很高——由于采用了连续的访问模式，GPU 只需通过较少的内存扇区就能满足多个线程的请求。
- 相反，非合并访问内核每请求需 32 个扇区的高比例，则表明其内存访问是非合并的——**跨步访问模式迫使内存子系统获取远多于满足相同内存请求所需的扇区。**

## 四、总结：1 个务必与 1 个始终

要使 GPU 代码获得最佳性能，高效利用 GPU 内存是你需要关注的最重要标准之一。**全局内存的最佳性能依赖于采用合并内存访问方式**。

> 请**务必** 尽量减少对全局内存的跨步访问，并且**始终** 使用 Nsight Compute 对 GPU 内核进行性能分析，以确保内存访问是合并的。

这种方法将帮助你最大限度地发挥 GPU 代码的性能。
