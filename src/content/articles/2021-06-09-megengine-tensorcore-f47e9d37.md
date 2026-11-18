![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/01.webp)

**前言**

2020年5月Nvidia发布了新一代的GPU架构安培（Ampere）。其中和深度学习关系最密切的莫过于性能强劲的第三代的TensorCore，新一代的TensorCore支持了更为丰富的DL(Deep Learning)数据类型，包括了新的TesorFloat-32(TF32)，Bfloat16(BF16)计算单元以及INT8，INT4和INT1的计算单元，这些计算单元为DL推理提供了全面的支持。

为了发挥这些计算单元的能力，以往会由资深的HPC工程师手写GPU汇编实现的卷积、矩阵乘算子来挖掘硬件的能力。然而凭借人力手工优化算子的方式已经没有办法应对如此多的数据类型，因此对于DL应用的优化渐渐地越来越依赖一些自动化的工具，例如面向深度学习领域的编译器。

在这样的趋势下，Nvidia开发了线性代数模板库CUTLASS，抽象了一系列高性能的基本组件，可以用于生成各种数据类型，各种计算单元的卷积、矩阵乘算子。MegEngine在CUTLASS的基础上进行了二次开发，可以高效地开发新的高性能的算子，快速地迁移到新的GPU架构。

本文将会深入介绍MegEngine CUDA平台的底层卷积算子的实现原理，并将会对Nvidia CUTLASS的Implicit GEMM卷积文档进行解读和补充。

因此，读者在阅读本文之前必须要了解的CUDA知识有：

•访问全局存储（Global Memory）时，同一Warp中的相邻线程访问连续的地址，访存请求会被合并，合并的访存能够最大化Global Memory的吞吐。

•访问Global Memory时，尽可能使用最宽的数据类型（float4）进行访问，这样可以最大化访存指令的利用率。

•CUDA的共享存储（Shared Memory）按照每 4Bytes划分为一个bank，共分为32个bank。当同一 Warp中的线程访问同一bank的不同地址时会发生冲突（bank conflict）。无bank conflict的访存模式才能最大化 Shared Memory 的吞吐。

•GPU有显存（Global Memory）、L2、L1（Shared Memory）、寄存器 4 个层次的存储，直接访问显存的延迟很高，在优化GEMM、Convolution这样的计算密集型算子时，需要

–通过 L1 和寄存器的缓存来减少Global Memory的访存请求。

–通过大量的计算来隐藏不可避免的Global Memory访存延迟。

首先，我们需要了解CUTLASS引入的一些抽象概念

•**TileIterator**:用于访问存储中的一个Tile的数据。TileIterator实现了advance()方法，支持在Matrix、Tensor等数据类型上进行遍历。

•**Fragment****:** 数组类型，用于存放TileIterator读取进来的数据。Fragment的数据通常存放在寄存器中。

然后我们简单回顾一下CUTLASS设计的高性能的GEMM算子的Pipeline，按照Pipeline实现的算子能够在CUDA平台上达到cublas的90%以上的性能。下图演示了CUTLASS设计的Pipeline化的GEMM算子：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/02.webp)

1.图中第一行演示了由**PredicatedTileIterator**和**SmemTileIterator** 配合完成从Global Memory到Shared Memory的数据搬运。

2.第二行演示了**WarpTileIterator**负责从Shared Memory搬运数据到**Fragment**寄存器中。

3.第三行展示了**WarpMmaOperator** 用**Fragment**寄存器中的矩阵数据执行矩阵乘加 (Matrix-Multiply-Add) 操作。

**Implicit GEMM 算法**

**卷积映射为矩阵乘法**

我们首先来看一下前向卷积算子的定义，假设输入的feature map是x，卷积层的weight是w，输出是y，其中x，y，w都是4维的Tensor，x的四个维度分别是NxICxIHxIW，w的四个维度分别是OCxICxFHxFW，y的四个维度分别是NxOCxOHxOW。那么输出y和输入x，w的数学关系式可以写成

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/03.webp)

公式里的小写字母代表了Tensor在每一维的坐标，其中ih，iw和oh，ow，fh，fw的关系式可以写为

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/04.webp)

这里的stride_h, stride_w, pad_h, pad_w是卷积层的参数。根据im2col算法的原理，公式里定义的卷积运算可以转化为一个矩阵乘法，也即

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/05.webp)

其中 •矩阵A由weight转化而来，是一个OC X IC·FH·FW的矩阵。

•矩阵B由feature map转化而来，是一个IC·FH·FW X N·OH·OW的矩阵

•矩阵C代表了输出的Tensor y，是一个OC X N·OH·OW的矩阵。

矩阵和Tensor在各个位置上的元素的对应关系为

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/06.webp)

其中矩阵的下标**i，j，k** 和Tensor的坐标之间的关系为

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/07.webp)

当**j** 已知时，可以用下面的关系式推算出feature map的坐标

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/08.webp)

当**k**已知时，可以推算出weight的坐标

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/09.webp)

同时结合oh，ow，fh，fw,就可以计算出ih和iw。

根据上面的讨论，我们可以把卷积的运算过程，写成一个隐式矩阵乘法(Implicit GEMM)的形式：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/10.webp)

上面的Implicit GEMM算法仍然是串行的形式，接下来我们要把它改造成CUDA上的并行算法。首先我们对整个计算任务进行分块，让每个线程块负责计算并输出大小为TILE_MxTILE_N的矩阵。于是算法变成了下面的形式：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/11.webp)

为了提高访存的效率，我们可以在GEMM_K这一维上也进行分块，每次将TILE_MxTILE_K的矩阵A和TILE_KxTILE_N的矩阵B缓存到Shared Memory里，避免重复的Global Memory访存。于是，算法就变成了如下形式：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/12.webp)

因为我们可以直接复用CUTLASS里已经实现好了高性能的WarpMmaOperator，所以实现基于Implicit GEMM的卷积算子只需要

• 适配DeviceConvolution、KernelConvolution

和ThreadblockConvolution，支持传入Tensor类型和Convolution Layer的参数。

• 添加PredicateTileIterator支持读取Tensor的一个 Tile 的数据到Shared Memory中，并隐式地将读入的数据组织成矩阵的形式。

• 算法的main loop中直接调用WarpTileIterator从Shared Memory读取数据，然后由WarpGemmOperator完成Warp-level的GEMM运算。

• EpilogueOperator适配卷积算子,将Accumulator的数据写回Global Memory的Tensor中。

接下来我们会以INT8数据类型的TensorCore卷积算子来介绍MegEngine底层的卷积实现，本文会重点介绍 2、3、4 是如何实现的，关于如何使用已经写好的卷积算子，可以参考之前的[文章](http://mp.weixin.qq.com/s?__biz=MzU4NTkwMDM1NA==&mid=2247487659&idx=1&sn=02aa4e7242d33699158c3491eff78fe5&chksm=fd82d886caf5519020d306a9a3d7e91d276678b5563cb87e24cb168328a9e2d3d7ccd675997c&scene=21#wechat_redirect)。

**Global Memory 数据布局（Layout）**

为了最大化TensorCore类型的卷积算子的吞吐，MegEngine使用了128位的Global Memory访存指令，因此在访问Tensor的数据的时候要求地址满足128位对齐。MegEngine使用了NCHW32的格式来存储Tensor,NCHW32格式的特点为：

•Tensor的通道维度按照32个channel进行分组，每 32个channel连续的存放在存储中。

•Tensor的其余维度按照W、H、C、N的顺序地址变化由快到慢的存放在存储中。

由于采用了32个通道对齐的存储格式，因此卷积layer要求输入和输出feature map的通道数都是32的倍数。

**预处理访存偏移量**

MegEngine的卷积实现在GEMM_K的维度上是按照(IC/32)·FH·FW·32的顺序累加，写成伪代码的形式如下：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/13.webp)

如果写成一层循环，那么应该写成：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/14.webp)

可以看到在迭代过程中，如果直接计算指针的偏移量的话，会引入很多除法和求余运算。而在CUDA平台上，整数的除法和求余的开销是非常大的，因此我们将一些地址的偏移量在host端预先计算好，存到kernel param的buffer中，需要时从constant memory中直接读取地址，避免除法和求余运算。

对于每个线程来说，在主循环中指针移动的offset如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/15.webp)

如果地址的增量可以用delta来表示的话，那么delta是以FH*FW为周期的，即：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/16.webp)

因此我们只需要大约O(FH·FW)的存储空间。其中地址偏移量的计算逻辑可以参考代码conv2d_tile_iterator_nt_src_fprop_precomp.h

由于kernel param buffer的大小为4KB，我们用了大约3KB来存储地址的增量，所以MegEngine的卷积实现要求Convolution Layer的FH*FW的大小不能太大，但是一般情况下，3x3，5x5，7x7的卷积都可以处理。Nvidia官方实现的迭代顺序与本文介绍的略有不同：

•官方实现需要将IC补齐为TILE_K的倍数，这样在通道数较小时会浪费一些计算量。

•官方实现的线程块在访问输入feature map的时候地址的跨度比较大，降低了访存的局部性，对cache不够友好。

因此在性能方面，MegEngine的实现会更有优势，而官方实现的优点是对Convolution Layer的参数没有太多限制，通用性更好。

**Warp-level Mma****(Matrix-multiply-add)** **指令**

cuda10.2引入了新的Warp-level的mma和ldmatrix指令，用户可以通过mma指令使用 TensorCore 来进行高速的矩阵乘加运算，通过ldmatrix精细地控制Warp给TensorCore喂数据。其中mma指令的用法如下：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/17.webp)

这条指令的语义是由一个Warp的32个线程同步地完成8x8x16的矩阵乘加运算，它有三个输入操作数，其中参与矩阵乘法运算的分别是一个8x16的矩阵A 和一个16x8的矩阵B，这两个输入矩阵的数据分布在同一Warp的32个线程中。矩阵A的布局如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/18.webp)

• 同一Warp中的32个线程分为8组，每组四个线程，负责读取8x16的矩阵中的一行。

• 每一组中的一个线程读取每一行中相邻的4个int8的数据，恰好填满一个32位的寄存器。

类似的矩阵B的布局如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/19.webp)

• 每4个线程一组，共分为8组，每组负责读取16x8的矩阵中的一列。

• 每一组中的一个线程负责读取一列中相邻的4个数据。

参与累加运算的矩阵C和输出矩阵D的数据也同样分布在32个线程中，它们的布局如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/20.webp)

• 同样每4个线程一组，每组负责读入/输出一行的数据。

• 每个线程负责输出一行中的相邻两个int32类型的数据，恰好构成一个64位的寄存器。

通过对mma指令的分析，如果Global Memory/Shared Memor中的数据是以行优先 (RowMajor) 或者列优先 (ColumnMajor) 的格式存储的，那么当同一Warp执行空间上连续的两个8x8x16的矩阵乘加运算时，每个线程读取的数据将会是跳跃的，执行每次乘法都只能读取32位宽的数据到寄存器中，而低位宽的Load指令通常没有办法最大化利用存储的带宽。因此Nvidia提供了ldmatrix的指令，可以让同一Warp一次性读取4个8x16的矩阵到寄存器中，这样恰好可以让Warp中的每个线程一次读取128位的数据，最大化带宽的利用率。

ldmarix的用法如下所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/21.webp)

上述这条指令恰好读取了4个8x16的矩阵，每个线程恰好负责读取矩阵的一行数据，读取完成后，线程之间会进行数据交换，将矩阵的数据重新分布到各个线程，读取的过程如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/22.webp)

这一节介绍了TensorCore相关的mma和ldmatrix指令，有了这两条高性能的指令，我们还需要为数据设计巧妙的Shared Memory存储格式，消除从Shared Memory读取数据的bank conflict，从而提升Shared Memory的读取效率。

**Shared Memory的数据布局**

在介绍Shared Memory中的数据布局之前，我们需要了解Shared Memory的访存特点。Shared Memory按照每4个字节组成一个bank，共划分成了32个bank，同一Warp的线程访问了相同bank的不同地址时会发生conflict，导致访存的效率变慢。在同一Warp的线程访问不同位宽的数据时，会有不同的行为：

• 每个线程访问Shared Memory中32位的数据，访存将在一个阶段内完成。

• 每个线程访问Shared Memory中64位的数据，访存会在两个阶段内完成：

– 第一个阶段：前16个线程访存128字节的数据。

– 第二个阶段：后16个线程访存128字节的数据。

• 每个线程访问Shared Memory中的128位的数据，访存会在四个阶段内完成：

– 每个阶段由8个线程完成128字节的数据的访存。

如果上述过程中每个阶段都没有bank conflict，则能够达到最大的Shared Memory访存效率。

通常为了避免Shared Memory的bank conflict，我们会对Shared Memory的数据进行padding，让线程访问的数据错开，避免落在同一bank中。但是这样做的问题是会使得kernel需要Shared Memory的Size变大，但是SM上的L1 cache(Shared Memory)又是有限的，所以padding会降低kernel的occupancy，进而就会降低kernel的性能。

因此CUTLASS设计了一种Shared Memory的交错布局方式，它能够在不进行padding的前提下，使得线程访存的地址没有bank conflict。接下来，我们以64x64的矩阵为例来详细介绍数据在Shared Memory中的布局。

首先，线程读取数据的粒度都是128位，也即16个INT8类型的数据，因此我们在演示数据的布局时总是以16个数据为一组。如果矩阵是以行优先(RowMajor)的格式来组织的，那么在逻辑上的布局如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/23.webp)

从图中可以看到

• 每16个元素分为一组，被称为一个Vector，被染上了不同的颜色。

• 每行相邻的32个元素被称为一个Crosswise，恰好是NCHW32格式中的一组channel的数据。

在Shared Memory的物理存储中，矩阵的数据进行了重新排列，如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/24.webp)

我们可以看到Shared Memory的物理布局有以下特点：

• 每4行的一个Crosswise的数据作为一组，连续存放在Shared Memory中，紧接着会存放这4行的下一个Crosswise的数据。

• 每组数据包含了8个Vector，占据了128个字节，恰好是Shared Memory中的32个不同的bank。

•每组数据在排列是进行了交错，保证了ldmatrix时不会发生bank conflict。

**显存 -> Shared Memory 的数据搬运**

这一节我们会介绍从显存(Global Memory)到Shared Memory的数据搬运。显存到Shared Memory的数据搬运是由**Conv2dTileSrcIteratorFpropPrecomp** 来完成的，本文并不会详细地解读代码的实现，而是描述线程搬运数据的过程，帮助大家建立直观的印象，更好地理解代码。

如果以上一节中Shared Memory的逻辑布局为例，同一Warp中每个线程读取的数据的逻辑布局如下图所示，每个线程读取16个INT8类型的数据，恰好构成一个Vector。

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/25.webp)

而在实际的物理显存中，线程访问的数据分布如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/26.webp)

• 我们可以看到每个线程读取了128位的数据。

• 相邻的线程读取的数据在物理上是连续的。

因此线程从Global Memory读取数据的pattern可以满足合并访存的要求，同时以最大的数据位宽进行访存，最大化了显存带宽的利用率。

然后如果将线程读取的数据映射到Shared Memory的物理地址，我们可以看到

• 每8个线程向Shared Memory写入128字节的数据，恰好落在Shared Memory的32个不同的bank中。

• 同一Warp的访存分为四个阶段完成，每个阶段都没有bank conflict。

下图演示了一个Warp写入Shared Memory的过程：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/27.webp)

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/28.webp)

**Shared Memory -> 寄存器的数据搬运**

Shared Memory到寄存器的数据搬运是由MmaTensorOpMultiplicandTileIterator完成的。同一 Warp 在每一轮迭代过程会读取4个8x16的矩阵到寄存器中，每个线程会读取一行的数据。例如第一轮迭代时，线程读取的数据在逻辑上的布局如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/29.webp)

而实际上数据在Shared Memory里的物理布局如下图：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/30.webp)

可以看到：

• 每个线程读取了128位的数据，因此访存分为四个阶段来进行。

• 每一阶段的8个线程读取的数据恰好落在了Shared Memory的32个bank中，并且线程访存的数据之间不存在冲突。

当进行到第二轮迭代时，每个线程访问的数据的物理布局如下图：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/31.webp)

同样的访存的每一个阶段都不存在bank conflict。

**Accumulator 写回全局存储**

在int8的情况下，同一Warp负责输出64x64的结果，kernel会分成8次写回Global Memory，每次写回32x8的矩阵。这样保证了每次将Tensor按照 NCHW32格式写回显存时，同一Warp的32个线程恰好写了物理上连续的256字节的数据，而每个线程写回8个字节，保证了可以使用64位宽的数据类型进行显存的写操作，尽可能提高带宽的利用率。

由于mma指令的特点，输出矩阵的数据分布在各个线程上，而为了能够合并访存，即：让相邻线程写回的地址是连续的，我们利用Shared Memory对同一Warp中32个线程的数据进行了交换。数据交换后，每个线程拥有连续的8个通道的数据，且线程写的地址是连续的，保证了写回Global Memory满足合并访存的要求。线程交换数据的过程如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/32.webp)

每一轮迭代，Warp中的32个线程将32x16的矩阵数据写入到Shared Memory中。接着如下图所示，每个线程会把连续的8个channel的数据读到寄存器中。

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/33.webp)

Shared Memory的数据交换是由以下两个Iterator完成的

•**InterleavedTileIteratorTensorOp**完成了每一轮迭代将32x8的数据写入到Shared Memory中。

•**InterleavedSharedLoadIteratorTensorOp**负责将连续的8个channel的数据读到Fragment寄存器中。

当线程将交换后的数据读到Fragment寄存器之后，会由EpilogueOp，在卷积的基础上完成BiasAdd的运算。

以**BiasAddLinearCombinationRelu** 为例，它实际上完成了下面的运算：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/34.webp)

其中bias是一个PerChannel的Tensor，代表了每个输出通道的偏置，z是一个和卷积输出大小一致的Tensor，用于Convolution和ElemwiseAdd的融合。

最后EpilogueOp的输出会由**TensorPredicatedTileIteratorTensorOp** 真正地写回到 Global Memory中。每个线程写回的数据如下图所示：

![文章插图](/articles/2021-06-09-megengine-tensorcore-f47e9d37/images/35.webp)

可以看到线程写回的pattern满足合并访存的要求，因此能最大化Global Memory写的效率。

**总结**

本文介绍了MegEngine底层的卷积算子实现原理，算子性能可以达到cudnn的80%以上，测速结果可以参见[文章](http://mp.weixin.qq.com/s?__biz=MzU4NTkwMDM1NA==&mid=2247487659&idx=1&sn=02aa4e7242d33699158c3491eff78fe5&chksm=fd82d886caf5519020d306a9a3d7e91d276678b5563cb87e24cb168328a9e2d3d7ccd675997c&scene=21#wechat_redirect)

MegEngine会对卷积实现进行持续优化，进一步提升算子的性能，目前来看有以下两点可做的优化：

• 借鉴Nvidia官方CUTLASS ImplicitGEMM Convolution实现对mask的处理，提高TileIterator对于mask判断的效率。

• 现在的卷积实现在写回显存时利用Shared Memory进行数据交换是存在bank conflict的。后续会考虑两点优化：

–对Shared Memory的数据布局进行探索，消除 bank conflict，优化Shared Memory数据交换的效率。

–对Global Memory中的Weight Tensor的布局进行探索，提高每个Thread上accumulator的局部性，避免在Shared Memory中进行数据交换。

**参考资料**

- Warp-level Matrix Fragment Mma PTX文档  
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8816
- CUTLASS Implicit GEMM Convolution官方文档  
https://github.com/MegEngine/cutlass/blob/61ff64e3ab6ad05b5ce2f205216901e8d030013d/media/docs/implicit_gemm_convolution.md
- Volta architecture and performance optimization  
https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf
- Developing CUDA kernels to push Tensor Cores to the absolute limit on Nvidia A100  
https://developer.download.nvidia.cn/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
