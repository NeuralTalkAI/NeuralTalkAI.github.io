
> **CLBlast是一个可用于生产环境，且高性能的OpenCL开源计算BLAS库**，支持平台包括AMD、NVIDIA、Intel的CPU、GPU，也包括移动端领域如Adreno、Mali GPU。系统支持Windows、macOS以及Linux系统。
>
> **虽然这个库有历史年代了，但是相信仍然有一些可以学习的东西。**

介绍提到可用于生产环境，是因为确实**不少开源项目是基于CLBlast实现的**，如支持**OpenCL后端的Caffe：ck-caffe**，**支持OpenCL后端的TensorFlow：tensorflow-cl**，当然，也有基于该项目的Python API的PyCLBLast，也有Java API。

## CLBlast的特点

其实说到这里，不得不提及相关历史。*最相关的是AMD最早开源的CLBlas和NVIDIA闭源的cuBLAS，作者可能之前是这个AMD计算库的使用者，由于AMD不再维护，因而作者独自开发了名为CLBlast的OpenCL BLAS库*，**相比AMD，CLBlast有下面几个优点**：

1. **为调优（Tune）而生。** 设计起始，在kernel的实现上便有很多预设参数，**即kernel高度参数化**。而clBLAS则由于更多较为low-level的硬编码优化如指针运算（pointer arithmetic）留给硬件编译器的优化空间相比CLBlast有限。如在GEMM上，clBLAS缺少循环展开系数、不支持带步幅的（strided）读以及local memory的cache相关调优取，且在GEMM上clBLAS没有对B矩阵做转置，导致在某些调优场景下无法连续地对内存访问；
2. **可以为特定问题调优。** 得益于第一点，可为特定问题如输入尺寸针对性地调优kernel。也因此有相关的CLTune以及CLBlast-database项目，根据前者跑出的数据，可以做如参数建模预测性能，未知设备性能建模等有意思的工作；
3. **支持FP16精度计算。** 在内存带宽（bandwidth，GB/s）和每秒乘加计算次数（GFLOPS）上，计算带来的收益是FP32的2倍，有时甚至超过2倍可能是tune参数空间的那个值针对特定某个case特别好。但并非所有的FP16都有速度收益，如AMD Polaris和Vega架构的GPU虽也支持FP16，也仅是内存和能耗上有节省，实际计算的速度仍然是FP32，和硬件底层实现有直接关系；
4. **支持批量操作（Batch）**，如BachedGEMM等。这点包括FP16，都是在深度学习/机器学习方面最常用的；
5. **支持Cuda。** CLBlast更强调通用性，在率先支持AMD GPU后支持NVIDIA GPU的OpenCL计算，也支持将OpenCL kernel代码通过 `opencl_to_cuda.h`转为Cuda代码，以及更高层次上对Cuda Host端代码做了抽象即CLCudaAPI这个header only的库。提一下CUDA版本的CLBlast优势：
6. 可集成到CUDA支持的项目中，将CUDA buffer直接作为输入；
7. 无OpenCL实现的NVIDIA硬件平台，如非X86的Jetson（ARM架构）和Driver PX系列以及IBM Power架构的超算平台等；
8. 性能与OpenCL版本的Kernel又有不同：主要原因是CUDA/OpenCL编译器不同，底层上NVIDIA可以对齐做更多的优化如 `__ldg`（读取数据获取最佳性能，但是用前提是通过constant缓存存储，且数据必须相对较小等）、 `shfl`（warp级广播操作）等intrinsic指令集的优化或者是基于混合精度的操作，所有这些都是OpenCL所不具备的。

此外，CLBlast主机端代码以C++11写就，而OpenCL Kernel代码则遵从OpenCL C规则。而clBLAS的kernel代码则是通过C++代码生成，难以阅读扩展和维护。

## 架构设计

在BLAS的实现上，完全兼容NNetlib的BLAS接口，也提供C/C++/Java接口。**BLAS routines的三级设计，根据下表，可以看出CLBlast完全支持**，且在此基础上增加extra，对于每个级别的每种情况，CLBlast都尽可能提供5种不同精度的实现：半精度如HGEMM，单精度如SGEMM，双精度如DGEMM，复数单精度2xFP32如CGEMM，以及复数双精度2xFP64如ZGEMM。

**表1：CLBlast支持的BLAS routines从Level1到3且有extra支持batch操作的GEMM和im2col等**

| level | routines | num |
| --- | --- | --- |
| 1 | AXPY COPY SCAL SWAP AMAX ASUM DOT DOTC DOTU NRM2 | 10 |
| 2 | GBMV GEMV HBMV HEMV HPMV SBMV SPMV SYMV TMBV TPMV TRMV TRSV GER GERC GERU HER HER2 HPR HPR2 SPR SPR2 SYR SYR2 | 23 |
| 3 | GEMM HEMM HER2K HERK SYMM SYR2K SYRK TRMM TRSM | 9 |
| extra | SUM MAX MIN AMIN OMATCOPY IM2COL AXPYBATCHED GEMMBATCHED GEMMSTRIDEDBATCHED | 9 |

**3级的BLAS中，前2级别性能主要反应在带宽上，是IO访存密集型（bandwidth-bound，主要指标为GB/s）操作，而level3则是计算密集型（compute-bound，主要指标为GFLOPS）**。

尽管总共有51种实现，且每种又有不同精度，但在**设计上尽可能遵从复用**：

1. **每种实现的kernel都并非精度固定的**，虽然C++模板在OpenCL1.1中不支持，但通过类型别名以及kernel运行时编译传入相关定义的精度是可实现对精度的控制的；
2. **有多种kernel的实现是复用的**，如axpy、dot等，在实现GBMV时，通过使用预处理宏定义复用了gemv的实现，数据的读取等。

## 参数化的kernel实现

**所有的CLBlast kernel在实现时，都结合了预处理宏，以参数化的形式实现，这样的实现在不同设备上都可以做调优。** 下面以 `axpy`这个level1的BLAS routines为例：

- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

```
#define WSG 64 // The local-group size#define WPT 4  // The amount of work-per-thread#define VW 2   // Width of vectors X and Ytypedef float dtype;  // Example data-type#ifdef VW == 1  typedef float dtypeV;#elif VW == 2  typedef float2 dtypeV;#endif  // and similarity for VW = {4, 8, 16}__kernel __attribute_(reqd_work_group_size(WGS))void Xaxpy(const int n, const dtype alpha,           const __global dtypeV* restrict xgm,           __global dtypeV* ygm) {  #pragma unroll  for (int w = 0; w < WPT; ++w) {    int i = w * get_global_size(0) + get_global_id(0);    ygm[i] = ygm[i] + alpha * xgm[i];  }}
```

其中 `WSG`、 `WPT`、 `VW`均为参数化的工作组大小、每个线程工作量、指令宽度的参数化宏定义，这些可以做调优（Tune）的一部分。axpy操作并非计算密集型任务，而是带宽密集型任务，在作者对其与clBLAS和cuBLAS的性能比较中，可以看出实际前两个参数 `WSG`和 `WPT`相比指令宽度 `VW`的调优对性能影响并不大。

![文章插图](/articles/2021-07-24-gpgpu-blas-clblast-a-tuned-bce6a81e/images/01.webp)

**图：SAXPY routine在GTX750Ti、TitanX和HD7970上的性能比较**

## GEMM的参数化调优

GEMM的调优则相比AXPY更加复杂，这方面的工作参考了《Performance Tuning of Matrix Multiplication in OpenCL on Different GPUs and CPUs》，做了比较多的假设。

假设包括：对输入参数的假设如矩阵的尺寸是work group的倍数，偏移量（offset）为0，矩阵B是提前做好转置的形式，这其中一部分假设计算属于前后处理的相关kernel实现。基于假设能很好地对较大的问题做分析，例如时间复杂度是O(n^2)的开销相比O(n^3)有一个数量级的差别。

![文章插图](/articles/2021-07-24-gpgpu-blas-clblast-a-tuned-bce6a81e/images/02.webp)

**图：矩阵长发中的调优参数。其中蓝色区域是单个线程完成计算的任务，橘色是每个work group完成的计算工作**

也正因为这个原因，**通用的直接kernel实现则更复杂，因为没有这么多的假设，是一个单独的kernel实现。** **GEMM的调优参数多达14个**，其中6个是上图展示的：**包括两个维度的work size（Mwg，Nwg），2D寄存器tiling设置（Mwi，Nwi），输入矩阵的矢量宽度，循环展开的系数Kwi，以及是否使用local memory等**。

更多细节有在**《CLTune: A Generic Auto-Tuner for OpenCL Kernels》**这篇文章写到（后续推送我们会分享），其中这篇文章**《Performance Tuning of Matrix Multiplication in OpenCL on Different GPUs and CPUs》**也给作者带来了设计kernel时的灵感。

**作者在优化BLAS时，几乎所有精力都在优化GEMM**，因为GEMM被BLAS level 3的多数routines用到。在性能比较中，作者发现不同输入规模下，**矩阵乘法CLBlast比clBLAS性能整体情况上好的更多**，体现在整体性能的稳定性和对于特殊尺寸如非2次幂的情况，作者**分析CLBlast性能更好的原因，主要是由于clBLAS没有对B矩阵做转置操作，导致访存不连续。**

## 调优过程

对于调优的详细过程，需要参考CLTune项目，一方面是前文所述，每种调优参数都至少有4~5个值的尝试的值，拿GEMM来说，完整的搜索空间规模，即使在过滤掉软硬件限制如最大work group size、local memory size后，仍然可以爆炸增长到10万种。

为此，CLBlast设定了两个搜索集合，一个可能的情况组合比方有500个，其中**结合了最有可能的参数包含一些经验**，在搜索时会一一尝试；另一个集合的组合则是**完全的情况组合，** 在搜索时则采用**随机采样** 的方式来找。

迄今为止，在社区的支持下CLBlast已经在多达50种不同硬件设备上做过参数调优(Tune)。对于及时是没有见过的硬件设备，则会采用该厂商已调优过的设备中的平均最佳性能的设定，保证性能还不错，在这方面有一个CLBlast-database项目收集了tune过程中时长和tune参数的信息。

默认情况下的tune结果是基于预设问题尺寸的，这主要是考虑到若情况太多tune的耗时过长，用户当然也可以自定义tune设置，找寻适合自己问题的最佳tune结果。

## 针对特定问题调优

调优出的一种情况，并不是一招鲜吃遍天的。下图是在2个硬件设备：Intel核心显卡Skylake ULT GT2（下左图），以及AMD显卡Radeon M370X（下右图）在单精度矩阵乘法上的tune效果。每个图的对角线是tune的结果作为性能基准，即100%，横坐标是在当前尺寸的tune效果，同列其它值是在该调优设定下其它尺寸下的性能提升或下降百分比。

紫色较多的竖列，可能是调优效果较通用的，比方下左图，前5列，即对应在这5种情况下的调优效果在其它的尺寸上的性能提升或者下降都在10%以内，而后面的其它列存在较多黄色区域，即这种调优设置不适用有明显性能下降。

在Skylake ULT GT2上，小尺寸下 `m=n=k=64`的调优设置不适用大尺寸，而大尺寸下的调优则比较通用；而Radeon M370X上则表现则较大尺寸上的调优效果的通用性较好。

![文章插图](/articles/2021-07-24-gpgpu-blas-clblast-a-tuned-bce6a81e/images/03.webp)

**图：CLBlast的SGEMM性能，在不同尺寸下tune的结果针对所有尺寸的性能情况**

综上，**针对特定领域问题的调优，在不同用例和硬件设备上是不同的，带来的性能提升是很有必要挖掘的。**

## 批量操作

批量操作，将原本跑n次，每次跑1张图片的操作，聚合为跑1次但这一次却跑n张图片的操作，这其中节省的时间不仅是主机端的CPU代码如for循环，也有GPU这一侧的处理代码。**深度学习训练用批量操作的情况较常见，推理尤其是端侧则少见一些**。

**表：map操作的耗时统计。第一行为首次，后两行为第2、3次执行的时间统计**

| total time(ms) | queue->submit(ms) | submit->start(ms) | start->end(ms) |
| --- | --- | --- | --- |
| 0.660 | 0.013 | 0.594 | 0.009 |
| 0.130 | 0.024 | 0.080 | 0.005 |
| 0.129 | 0.010 | 0.079 | 0.004 |

当开启Profiling后，可以统计OpenCL Kernel在不同时间点：命令入队（queue）、提交命令到队列（submit）、命令执行（start），命令完成（end）的时间戳，根据两个时间戳的差值计算得到该阶段的GPU耗时。

如上表 `start->end`才是kernel实际执行时间，而如果不是批量操作的kernel，不可避免会有多次的 `queue->submit`和 `submit->start`的开销。这也是为何批量操作聚合如算子融合能节省时间的主要原因，尤其是多个小尺寸的操作连续执行，导致了GPU硬件层面的线程和work group未被充分利用，且过少的线程导致GPU的内存延迟不能得到很好地隐藏，而批量操作如Batched BLAS通过一次同时进行大量相似的计算，减轻这个问题。

有些情况下，**批量操作的kernel实现在性能上相比非批量的，性能相差可能达到一个数量级。但这种优势也有可能随着输入尺寸增大而减小**。此外，对于批量操作的kernel进行调优（tune）时，需要与非批量操作的进行区分。

## 未来与总结

虽然CLBlast有以上众多的特性，但还有些地方存在不足，也是**未来重点发力** 的方向：

1. 特定用例的最佳性能，需要用户实际去跑auto-tuner，这个过程是必不可少的。而若默认的参数就是考虑如最大尺寸和小尺寸矩阵的最优权衡结果，那就更好了，但当前还不具备。要做出这种权衡后的参数估计并不容易，需要对kernel和硬件进行更精细的建模，如用上一些机器学习的方法；
2. **预测** 不可见设备要调优的参数，而不是参数值；
3. **特定领域** 如深度学习的应用，优化或**增加kerne**l如FP16的批量GEMM、im2col，类似cuDNN基于tensor的卷积网络的算子kernel等。

综上，CLBlast是一个可用于生产环境且高性能的OpenCL加速库，未来会继续在高性能计算和深度学习领域发力。

## Reference

- code: https://github.com/CNugteren/CLBlast
- paper: https://arxiv.org/abs/1705.05249

> 点击【阅读原文】，看往期文章
