> 不恰当地说，这篇 CLTune 与前两天发的 [**《** **GPGPU BLAS 计算库 CLBlast: A Tuned OpenCL BLAS Library** **》**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886014&idx=1&sn=87ac9e9eee71b5f625df41912990115a&chksm=b290405c85e7c94ac662d54c9981ba656cbf31430e0b332de9dea84b1eae26a182bf1765581b&scene=21#wechat_redirect) 犹如 AutoTVM 与 TVM 两篇文章的关系，一个是 Tunner ， 一个是 Tunner 生成的数据产品 —— CLBlast ，不涉及编译器底层。
>
> ![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/01.webp)
>
> CLTune 与 CLBlast 是 2016 GPU Technology Conf 上的技术分享，虽然有些老，思想。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/02.webp)

**图 GPU Tech Conf 2016年上的分享，作者是 Cedric Nugteren**

# **CLTune: A Generic Auto-Tuner for OpenCL Kernels**

作者在 [**CLBlast 的文章**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886014&idx=1&sn=87ac9e9eee71b5f625df41912990115a&chksm=b290405c85e7c94ac662d54c9981ba656cbf31430e0b332de9dea84b1eae26a182bf1765581b&scene=21#wechat_redirect) 里并没有谈及较为细致的 tune 说明，而在这篇 CLTune ，作者在实验部分**以矩阵乘法和二维卷积为例，讲了自己 CLTune 的工作**，**在 2D 卷积和 GEMM 的实验结果上，都达到甚至超过现今最好性能****，实验在 NVIDIA / AMD / Intel GPU 上进行，且统一为 FP32 精度**。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/03.webp)

**图：需要auto-tuner的理由**

作者在 2016 年 NVIDIA 举办的GPU技术大会上也做了主题为**《Better Than All the Rest: Finding Maximum-Performance GPU Kernels Using Auto-Tuning》**的演讲。

> 下面的内容将结合作者的文章摘要、演讲 Slide 以及我浅薄的理解。

## CLTune 与现有很多 Tuner 工具不同的地方（**不考虑 AutoTVM 、AutoKernel这类的**），**在其对 Kernel 的普适性 tune 支持、易用、支持多种搜索策略（随机/粒子群PSO/模拟退火）且开源**。多种搜索策略，尤其是启发式算法，也是由于搜索空间太大而不得不选择的，例如 GEMM 的搜索空间就达到 20 万种组合。说到这里，不得不提一下 AutoTune 的使用场景，也是作者在设计之初考虑的：

1. **搜索空间巨大。** 如向量宽度（vector width），workgroup，线程的工作粒度(work per thread)，是否使用local memory（cuda里称为shared memory），就光是目前说到的这几个参数，组合起来的情况是巨大的；
2. **各种设备上都要保证高效运行。** 如GPU厂商对自家GPU优化确保性能符合预期需考虑不同架构代数，手机厂商需要支持发布的多款手机需考虑不同架构及驱动，APP应用开发者需考虑兼容性和不同设备上的性能；
3. **最优参数的设置随输入数据改变的情况。** kernel要能以不变应万变，不变的是性能一直保持领先如硬件峰值百分之XX以上，而输入的数据是会变化的。当这种输入数据改变时，也需要调整最优的参数设置；
4. **通用性**，这点是我附加的，考虑到CLTune并非最早的auto-tuner工具，最为人所知的应该是ATLAS，这一类都带有一个用于解决OpenCL或GPU相关问题的auto-tuner，用于解决如卷积/稀疏或稠密矩阵向量乘法/FFT等等，但这些项目针对特定问题而有局限性的特点。虽后来有较为通用的OpenCL auto-tuner也就是Maestro data-orchestration tuner，但其重在数据的传输而非计算。此外，也有更高级概念表示的tuner，如Thean中的数学表达，但他们也因为过于high-level导致很难细粒度的对参数做调优。

此外，CLTune 的 tuner 是 C++ API，在使用方式上**可以离线或在线集成到项目使用（这方面是作者架构设计带来的优点）**。CLBlast **将 OpenCL API 的调用完全隐藏**，如设备初始化/ Kernel 调用/内存管理等。

## 1. 模板化实现Kernel

为有一个直观的描述，下面从一个简单例子 `copy`这个io密集型的kernel开始。

### 举例1：copy

在kernel实现之初，就以类似**模板kernel** 的形式来实现，其**模板参数** `WPT`（work per thread）单位线程的工作量，表示了每个线程做多少个元素的拷贝，该参数可以是 `{1,2,4}`等，由于对任务数量做了切分，因而主机端设置全局线程数量时，就**需要对原始的 `GlobalSize`除以 `WPT`单位线程的工作量**，得到实际需要的全局线程总数。

- 
- 
- 
- 
- 
- 
- 

```
__kernel void copy(__global float* in,                   __global float* out) {    const int tid = get_global_id(0);    for (int w = 0; w < WPT; ++w) {        out[tid * WPT + w] = in[tid * WPT + w];    }}
```

在实际tune过程中，会根据 `WPT`在 `{1,2,4}`里选择最优性能下的 `WPT`参数值。下面是CLTune主机端的调用代码：

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
// Creates a new tuner on device 1 of platform 0cltune::Tuner tuner(0, 1);// Kernel: 2048/WPT global and 64 local threadstuner.AddKernel("copy.cl", "copy", {2048}, {64});tuner.AddParameter("WPT", {1, 2, 4});tuner.DivGlobalSize({"WPT"});// Specifies the input and output host arraystuner.AddArgumentInput(in_vector);tuner.AddArgumentOutput(out_vector);// Starts the tunning processtuner.Tune();
```

其中传入给 `tuner.AddParameter`的参数 `WPT`以及其搜索的候选值。`WPT`会作为OpenCL的 `BuildProgram`的 `build_option`，作为宏并传到kernel文件里进行编译。这样也就需要编译3次。

### 举例2：matvec_tiled

第二个例子是矩阵mata（M行N列）与向量vecx（N列1行）的乘操作，且结果为向量vecy（M行1列），属于BLAS level2 routine。其kernel模板化的实现中有一个可调优的参数TS，即tile size，该参数用于对vecx的分片缓存，即预先放到local memory中，再在与矩阵mata计算的时warp内的work item就能共享使用。

vecx的维度是tilesize的整数倍，且vecx能被分成N/TS个tile size，cl kernel执行一次（即一个线程执），其内部内的for循环会将所有N/TS个tile size的第一个元素保存到对应local mem的tile_x中，供一个warp内的work item共享。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/04.webp)

**图：matvec_tiled kernel的host端和kernel代码**

可以看到这个mattiled的外部global work size用到了1维，gid(0)遍历mata的M行即[0,M-1)，用到local work size的1维，lid(0)遍历分片（tiled size）的尺寸即[0,TS-1)，遍历的目的是对vecx做部分的缓存（local memory），缓存大小即分片大小TS行1列，由于是local memory，**因而这些是一个warp内所有work item所共享的，且需要在填充完设置barrier(CLKLOCALMEMFENCE)，用于后续计算vecy元素的部分结果**，即mata的1行TS列，与tile_x的TS行1列的尺寸相对应。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/05.webp)

**图：matvec_tiled kernel计算示意图**

将上面的代码画成了示意图：

- 左侧代码，蓝色的表示对vecx做local memory的部分，绿色是部分matvec和部分vec_x做计算的部分；
- 右侧示意图，在读左侧代码的绿色内容时，发现对mata取元素是列优先的方式。外层for循环每执行一次，会计算TS大小的mata和vecx计算得到一个vecy元素的部分结果。

Local memory的使用场景是当work item访问相同内容的数据大于2次时，如计算3x3的滤波计算，在滑动窗口步长很小时，两次计算的数据有较多重复，就可用到，减少对去Global memory的频繁加载。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/06.webp)

**图：Adreno OpenCL内存说明**

对于local mem见图：Adreno OpenCL内存使用，能看出其特点是在片上即Shader Processor里，相比global mem有性能优势，特点是一个work group内的所有work item共享。此外，Adreno官方文档有罗列使用要点：

- 数据同步的目的是当两个work item存在对同一块local mem一个读一个写时，可能会导致不一致性。因而需要同步。在matvec_tiled中先加载数据，再计算，在计算前需要同步，确保计算时数据全部拿到；
- 同步需要设置同步栅栏（fence/barrier），但这会阻碍GPU算术运算单元（ALUs）导致利用率降低，意味着高延迟。甚至有些场景，同步带来的延迟（synchronization latency）会抵消乃至超过使用local mem的使用收益。换言之，干脆直接使用global memory算了，起码也比用local带来barrier好，上面的例子，相同GPU上使用local memory有收益需要在问题规模上跑跑看；
- 既然用local mem，搭配向量化的操作更好如128bit的vload4float，推荐这种32bit对齐的用法。上面matvectiled计算过程中没有用到，可以优化；
- 让一个work group中的每个work item参与local memory的读取，而不是一个work item做整个local memory的读取。上面例子中，通过使用local work size，很好地让一个work group里的work item都参与local memory的创建（tiledid这个由getlocal_id(0)获取到的索引）；
- 当用local mem时，不要用asyncworkgroup_copy来实现异步拷贝操作，无论是dst和src哪个是global mem，只要是存在local mem，都会因编译器对对local memory读取优化的不好带来性能问题，有这种local和global mem的异步操作建议用户手动完成。

对local mem扯远了，模板kernel的写法由于引入了和local mem有关的参数TS（Tile Size），我们不得不去关注性能相关的使用限制。

### 归纳

从copy和matvec_tiled两个例子中，可以将这个tune的完整步骤归纳为：

1. 实现带有宏的kernel，宏作为tune参数模板，在调优时会根据预先设置的各种情况，得出一系列排列组合；
2. 实现host端代码，即将kernel中需要替换的参数宏加入到tuner的设置中，以及可能的值；
3. 实现参考kernel，即用来验证调优kernel的正确性的reference实现一般为naive的实现
4. tune进行，这其中根据设定选择搜索策略，有全局搜索，随机搜索，模拟退火，粒子群这4种搜索策略；
5. tune完成，得到最佳的参数组合。

但不难看出也存在一些问题，tune场景一般来说分为离线和在线，离线调优的场景如固定设备的安防厂商/IOT厂商/GPU厂商等，花多久的时间都能容忍，但是在线调优的场景如APP开发者，需要兼容适配尽可能多的手机，为了性能最佳，从APP采集到的信息根据机型占有量，离线做当然可以，提前采购该APP占有率最多如80%的机型，分别看GPU型号进行离线适配，将离线调优好的参数加载。

## 2. 搜索过程中的主要耗时

**但当用户量达到一定规模时，这种方式也可以，但数量太过庞大，可能需要on-line在线方式调优，这就要考虑GPU可用性和兼容性，也要考虑到在线调优的时长。**

像上面以宏参数的形式传入调优的各种值，是比较好的，但是每次需要编译，在手机上编译一次入mobilenetv1模型，如骁龙8系列的BuildProgram就要100ms这个数量级，模型更大的情况下OpenCL的Program Build如Yolov3模型则500ms到1秒之间，这还是复用了编译过的Program的情况。

所以**在移动端上做on-line tune，可能就需考虑避免二次Build Program的调优**，可以尝试将原本的宏参数改为 `setKernelArg`，以参数的形式来做如在较小粒度上调优是否使用某种inline的方法的哪一种实现（当然这种方式在使用上不如加 `build_options`来的方便，毕竟kernel代码里一堆 `if-else`的也影响性能），或是调优不需要二次编译Program的local work size（即work group size），还有可以在更大粒度上调优选择要执行同一个Op的kernel的多种实现如卷积的不同实现方式等。

但搜索空间特别庞大时，即使是离线，考虑调优的时间包括：

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/07.webp)**图：Profiling flags**

| clprofilinginfo类型（剖析时间信息） | 解释 |
| --- | --- |
| CLPROFILINGCOMMAND_QUEUED | 当主机（host）将由事件（Event）标识的命令（command）排入命令队列（command queue）时。该值为64bit值，当前设备以纳秒为单位的时间计数器，这些信息下同，略 |
| CLPROFILINGCOMMAND_SUBMIT | 主机将带有时间标识的命令从host提交到device相关联的命令队列中。因为考虑到host上可能有多个device，每个device可能都有各自的命令队列。命令队列由设备、上下文创建，上下文由设备创建，API是这么定义的，一般每个设备只有一个命令队列、一个上下文 |
| CLPROFILINGCOMMAND_START | command在设备上开始执行 |
| CLPROFILINGCOMMAND_END | command在设备上执行结束 |

**表：clprofilinginfo剖析时间信息类型**

1. GPU Kernel时间。命令队列（command queue）中命令的4个阶段：`queued->submit->start->end`，其中 `start->end`是GPU kernel执行时间，更多见**表clprofilinginfo剖析时间信息类型**。关于这三个阶段的时间，上一篇有AMD GPU的数据，本文略。为拿到剖析时间，需要创建命令队列时设置 `CL_QUEUE_PROFILING_ENABLE`的标志；
2. 二次Build Program的时间。下面在骁龙835对mobilenetv1模型做了耗时方面的统计：
3. **首次运行=加载模型+在线编译opencl program+其他琐碎的时间+首次运行，总计800+ms，****加载模型和在线编译opencl program是大头****；**
4. **保存binary后，再首次运行（加载编译好的opencl program）=加载模型+其它琐碎时间+首次运行的时间量级为：100+ms****；**
5. 因而，二次加载时节省在线编译Build Program的时间量级：500~600ms；
6. 保存binary的时间量级：0.77ms；
7. 加载编译好的opencl program的时间：0.5ms；
8. 保存的opencl program binary的文件大小：92KB；
9. opencl program binary在线编译对应的 `*.cl`文件个数：6个，即在线编译 `cl::Program`对象的次数为6次；
10. binary包含的kernel func数量：31个，即由 `cl::Program`对象创建的 `cl::Kernel`对象个数。
11. 等待/确保gpu kernel计算完成。用于获取当前调优设定下的 kernel计算时间，即 `start->end`的时间。该过程是否需要clwait/clfinish/clflush，先说结论是需要 `clWaitForEvent`的（实测中发现也可以不要），下面再说说区别；
12. OpenCL runtime enqueue API函数分为阻塞调用和非阻塞调用，对非阻塞调用如 `clEnqueueNDRangeKernel`，真实的GPU kernel执行时间并非在该函数前后计时，而是两次打点中间要有 `clWaitForEvent`（前提是有非阻塞调用的事件ID）来保证 `CL_COMPELTE`状态，或者是 `clFinish`。
13. clflush：目的是为了加快命令command提交。简述下背景：**交给gpu要执行的任务可以理解为一个个命令，这些命令在执行时都要到命令队列中即入队，再提交，再开始gpu的计算，然后是计算完毕**。**flush的是加快提交（submit）的进行，但它不保证执行完成（不是同步点）且不能加快gpu的计算（start->end），目前该api很少用。**入队和提交的两个阶段点，分别可理解为软件的开销和cpu cache操作的开销，并非gpu硬件的开销，当命令队列中的opencl kernel足够多时，就会将kernel入队，然后提交，因这个过程有一定gpu的自己调度，但为了加快提交进程，才有这个clflush api。command queue四个时间点是：`queued->submit->start->end`，clflush是加快 `queued->submit`的阶段。
14. **clFinish：会确保一个command queue中所有命令都执行完毕，khroonos的官网文档也说道，这个会block阻塞的**，它返回一个cl_int作为status，这个API一执行性，只有 `command_queue`中入队的所有命令的都被处理完且完成时，才会返回status，clFinish也是一个同步点（synchronization point）。多说两句，这个会影响调优的时间，调优过程不建议用这个确保完成。clFlush和clFinish都是barrier操作，只是barrier的阶段不同。
15. clwait：没有clwait这个api，具体说应该是clWaitForEvents，Events实际是OpenCL中的事件，**一般用于调度调整任务的逻辑顺序（比方a要在执行b之前，那就在b执行的时候在api上设置对a的event list来调整顺序），还可以获取统计的时间信息等**。咱们这只关注执行时间信息， `clWaitForEvents`等待的是gpu命令队列中的命令的执行状态成为已完成，即 `CL_COMPLETE`，表示该命令已完成，此外由于OpenCL也支持OpenGL扩展，如果是gl的事件那么也能反映gl同步对象的状态。clwaitforevent和clfinish可以阻塞直到kernel执行完成。
16. 主机端代码，如切换各种调优策略时的C/C++代码等。

## 3. 搜索空间的特点

搜索过程不是基于一堆已有的性能数据和选项做预测最佳设定，即没有性能数据库，而是基于候选的选项如WPT各种候选值、VW各种候选值等在这些设定下，跑出最好的性能。即使如此，也有一些人为的设定限制，但即使在有这些限制下，搜索空间还是很大，如下图是5个参数下，排列组合且去除不合理设定下仍有3424种组合。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/08.webp)

**图：直接卷积的实现下的搜索空间**

这其中也能发现一些空间上的规律：

1. 每种参数实际上候选值是有限的：比方指令宽度（VW）往往是1，2，4，而每线程的工作量往往在2到8，work group的大小也是在2的5次幂，3个维度且再算上默认的(0,0,0)就是16种，是否做for循环的展开，是否使用local mem等等；
2. 卷积的搜索空间只有5维，但若是写的更复杂些如达到10个参数即10维度以上，是轻而易举的；
3. 参数离散且非线性：如WPT可以是1，2，4，8，而且对性能来说，从4到8很可能由于寄存器压力从4到8导致性能急剧下降；
4. 参数间的强相关。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/09.webp)

**表：在矩阵乘法中7项参数在不同硬件上的最佳选择**

由于非线性（且值非常接近）和布尔变量参数值的存在，基于导数、自动微分、无导数来寻找最优值的三种方法也不适用。因而选择启发式、以及随机搜索的方式。其实随机搜索是最简单的策略，其采样并测试随机的组合情况。其执行效率完全取决于搜索空间的形状，如果高性能排列组合的参数在搜索空间里挨得近，那么搜索（到高性能的参数）自然效率就低。

## 4. 矩阵乘法

作者介绍了两个例子：2D卷积和矩阵乘法。矩阵乘法介绍的更详细一些，这里我展开一下。

矩阵乘法也是计算密集型算子，且作为2D卷积的实现方式之一，在深度学习和机器学习领域被广泛使用。也是大多数BLAS调优库的重点优化对象。矩阵乘法可以表示为 `C=α*A^T B+βC`，其中 `α`、 `β`为常数， `A^T`为转置后的矩阵A，假设矩阵维度是2次幂，且维度是tile size（后续会说道）的整数倍。

在调优参数上，为了尽可能粒度能细一些，CLTune实现了一个高度可调优的版本，其中调优参数有14个：

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
- 
- 
- 
- 

```
// Parameters determined by the tuner// 1. MWG : M维度上的Tile-size，如64, 128// 2. NWG : N维度上的Tile-size，如64, 128// 3. KWG : K维度上的Tile-size，如8,16// 4. MDIMC : M维度上每个workgroup的线程数，如8, 16, 32// 5. NDIMC : N维度上每个workgroup的线程数，如8, 16, 32// 6. MDIMA : 矩阵A的Re-shaped tile的M方向长度，reshape tile A的维度为KDIMA * MDIMA// 7. NDIMB : 矩阵B的Re-shaped tile的N方向长度，reshape tile B的维度为KDIMB * NDIMB// 8. KWI : KWG循环的展开系数，小于等于KWG// 9. VWM : 矩阵A和C向量宽度，支持包括1, 2, 4, 8// 10. VWN : 矩阵B的向量宽度，支持包括1, 2, 4, 8// 11. STRM : 在M维度上是(1)否(0)使用带步长的线程访问// 12. STRN : 在N维度上是(1)否(0)使用带步长的线程访问// 13. SA : 是(1)否(0)使用local/shared内存来对矩阵A做缓存// 14. SB : 是(1)否(0)使用local/shared内存来对矩阵B做缓存此外，还有基于上述14个调优参数的辅助参数：#define MWI (MWG/MDIMC) // 每线程的M维度工作量，即M方向的tile size大小除以M方向的workgroup线程数#define NWI (NWG/NDIMC) // 每线程的N维度工作量，即N方向的tile size大小除以N方向的workgroup线程数#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // 矩阵A的Re-shaped tile的K方向长度，reshape tile A维度为KDIMA * MDIMA#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // 矩阵B的Re-shaped tile的K方向长度，reshape tile B维度为KDIMB * NDIMB#define MWA (MWG/MDIMA) // 每线程在矩阵A的M方向的load总数#define KWA (KWG/KDIMA) // 每线程在矩阵A的K方向的load总数#define KWB (KWG/KDIMB) // 每线程在矩阵B的K方向的load总数#define NWB (NWG/NDIMB) // 每线程在矩阵B的N方向的load总数
```

作者在其实现中，有10个函数，除 `gemm_fast`外其余9个均为inline函数：

1. gemm_fast：矩阵乘实现入口，也是骨架，根据传入的build option参数，会选择性地调用其余9个内联函数。其流程大致为：
2. 若开启SA/SB，则分配一个workgroup内共用的A和B的local mem `alm`、 `blm`；
3. 分配work-item独占的private mem `apm`、 `bpm`、 `cpm`；
4. 初始化累加寄存器 `cpm`；
5. k方向循环遍历搜有workgroup tiles：
6. 对local mem设置同步点；
7. 存储private mem计算结果到global mem。即从 `cpm`到 `cgm`。
8. GlobalToPrivateA（不开启SA）：缓存A的global mem（非片上），到每个线程的private mem（寄存器）；
9. GlobalToPrivateB（不开启SB）：同上；
10. GlobalToLocalA（开启SA）：缓存global mem（非片上）到local mem（一般是片上）；
11. GlobalToLocalB（开启SB）：同上；
12. LocalToPrivateA：缓存A的local mem（warp内共享），到每个线程的private mem（寄存器）；
13. LocalToPrivateB：同上；
14. StoreResults：将private mem的C结果写回到global mem的C中；
15. MultiplyAddVector：单纯乘加操作，底层可选是基于 `mad`或原生的乘法操作；
16. MultiplyAccumulate：调用 `MultiplyAddVector`，计算 `Cpm+=Apm*Bpm`。

---

下面结合示意图，来具体说明这14个参数对应的优化点：

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/10.webp)

**图：** **矩阵乘法和调优参数示意** **图**

### 4.1 workgroup 2D tile

对应上图青色部分，为3个参数，通过三个参数 `M_{wg}`、 `N_{wg}`、 `K_{wg}`对应矩阵乘法的 `M`、 `N`、 `K`三个维度来进行调优。

> 在前文中 `matvec_tiled`实现的矩阵向量乘法中，tiled含义为对向量的一部分做local mem上的缓存，在后续计算中用到，这里在矩阵乘法中的2D tiling类似。

### 4.2 thread tile

对应上图橘色部分，2个参数。local work size（即workgroup size）在2个维度上分别为 `M_{dimC}`和 `N_{dimC}`，即定义了在M和N维度每workgroup内单线程的工作量：`M_{wi}=M_{wg}/M_{dimC}、N_{wi}=N_{wg}/N_{dimC}`，其中 `M_{wg}、N_{wg}`为2D tile size参数，设定每线程工作量是为了线程粗化（coarsening）**增加每线程的利用率/操作数**；

### 4.3 memory缓存：global->local / global->private

是否输入矩阵A或B做大小为2D workgroup tile的local mem缓存，如果不使用则将tile size大小cache到private mem中。因为是A和B两个矩阵，是4种可能，作者因此**分别实现了名为 `GlobalToLocalA`、 `GlobalToLocalB`、 `GlobalToPrivateA`、 `GlobalToPrivateB`4种情况的inline kernel**；

### 4.4 memory调优：local mem reshape

该优化点需确保开启即对A或B使用local mem，在该情况下，决定是否对local mem做reshape操作。遵循对矩阵A\B\C workgroup维度上的要求，即：`M_{dimC}*N_{dimC}=M_{dimA}*K_{dimA}=N_{dimB}*K_{dimB}`。其中，workgroup上的 `M_{dimC}`和 `N_{dimC}`是两个可以调优的参数。其实，这里我没看明白，对于local mem做reshape，贴出原文：

> The local memory (when enabled) can be re-shaped according to `MdimC·NdimC=MdimA·KdimA=KdimB·NdimB`. Here, `MdimA` and `NdimB` are extra tuning parameters and `KdimA` and `KdimB` are calculated according to the above equality.

不太清楚是指后续做矩阵分块还是什么意思；根据后文的最佳参数值，该值候选值为 `8,16,32`，可以确定的是，对local mem做reshape会改变内存排布，**影响访问读取时候的效率，本质上也是优化L1 cache利用率**。

### 4.5 访存调优

单个线程在非片上内存访问的步长。实际我在阅读过程中也没太理解做这个的目的，因而贴出原文：

> A stride for accessing off-chip memory within a single thread can be enabled or disabled through Mstride (for matrices A and C) and Nstride (for matrix B). If enabled, the stride is set to MdimA and NdimB respectively, otherwise it is set to 1 (no stride).

1. 当带步长访问矩阵A和C时， `M_{stride}=M_{dimA}`，不带步长为1；
2. 若带步长访问矩阵B时， `N_{stride}=N_{dimB}`，不带步长为1。

但有一点是可以明确的，访问内存的方式对对性能有极大的影响，**最理想的方式则是：一个workgroup内的线程访问连续的内存地址**，这可以高效利用GPU L1 Cache。即使是调优LWS，也是提高L2 Cache的利用率（这部分参考ARM Compute Library相关的演讲，其中有提到，最理想的情况下是：不同计算单元复用相同的内存块）。

### 4.6 访存-level调优：向量宽度

**通过调整访问内存（即读取和存储）的向量宽度，增加操作数来提升性能**。对矩阵A为 `M_{vec}`、对矩阵B为 `N_{vec}`；

### 4.7 访存-level调优：循环展开

对应上图A矩阵红色部分，通过开启或者关闭循环展开系数，来实现编译器级别的动态循环展开。`K_{wg}`即kernel内循环可以以系数 `K_{wi}`展开的值 `n`。

循环展开可以由程序员完成，也可由编译器自动优化完成。**循环展开通过将循环体代码复制多次实现。增大指令调度的空间，减少循环分支指令的开销。循环展开可以更好地实现数据预取技术**，这其中加入 `unroll`告诉编译器来自动完成。

下面是该操作的优点和缺点，这部分内容摘自 CPU 在循环展开时候的特点：

- **优点：性能提升。增加并行操作数，增加实现的内存带宽使用率，增加kernel在硬件执行过程中每个时钟周期的操作数，消除展开前的分支判断，管理归纳变量，****优化调度(管道过长)带来的延迟即延迟隐藏**；
- **缺点：可能** **增加指令缓存未命中风险****（含分支的情况可能比递归更慢），代码不可读，代码体积增大。**

### 4.8 不同设备上的最佳参数

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/11.webp)

**表：不同设备在矩阵乘法上搜索到的最佳参数值**

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/12.webp)

**图：GEMM案例总结**

矩阵乘法上，作者在K40m上性能**没有拼过cuBLAS的主要原因还是CUDA在汇编级别的优化上做到了减少寄存器压力，移除寄存器bank冲突，其实本质上是拿不到类似CUDA `ldg`这种OpenCL的指令**， `ldg`对于只读global memory数据可以直接从更快的texture缓存中读取，**texture有用到L1 cache**。

## 5. 卷积

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/13.webp)

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/14.webp)

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/15.webp)

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/16.webp)

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/17.webp)

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/18.webp)

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/19.webp)

6. 搜索策略的经验

两种启发式算法：模拟退火和粒子群优化，都有其各自的特点，不同的问题哪一种更合适需要尝试的。

![文章插图](/articles/2021-07-26-cltune-a-generic-auto-tuner-65ac49e5/images/16.webp)

**表：作者实验调优的硬件**

通过作者的尝试，也发现一些经验：

1. **当用户自定义卷积核比较小时，可以将其放到OpenCL constant mem中**；
2. 在2D卷积实验中，对完整搜索空间的搜索结果的性能分布上观察，**只有极少的设置下性能很好**。我的理解是，**参数间的强相关，整个搜索空间的较好性能情况还是非常稀疏的**；
3. 在2D卷积实验中，模拟退火和粒子群在某些硬件上表现好，但有些反之，应该是落入到了局部最优后续也出不来了；
4. 在矩阵乘法实验中，最佳的7类参数在下标中，可以看出不同的设备上基本都是不同的。

其实类似的实验经验还有一些，但是都是设备相关的，不具有普适性。总的来说，**CLTune提供了在OpenCL Kernel上为每一个硬件设备、以模板化方法实现来调优的思路，将异构计算的通用性思维发扬光大**。

> 但其实手写常用算子+tuning的成本确实不高，但是**长远来看，长尾算子、算子融合这些，实现成本就太高了。还是需要将tune策略与codegen结合起来的**。这方面的工作，大家都知道就不多说了。

## Reference

- paper: https://www.researchgate.net/profile/Valeriu-Codreanu/publication/301521441CLTuneAGenericAuto-TunerforOpenCL_Kernels/links/571755ce08ae09ceb2642db1/CLTune-A-Generic-Auto-Tuner-for-OpenCL-Kernels.pdf
- slide: http://www.cedricnugteren.nl/downloads/handouts2016a.pdf
- How to use on Win7: https://williamjshipman.wordpress.com/2016/01/31/autotuning-opencl-kernels-cltune-on-windows-7/

> 点击【阅读原文】，看往期文章
