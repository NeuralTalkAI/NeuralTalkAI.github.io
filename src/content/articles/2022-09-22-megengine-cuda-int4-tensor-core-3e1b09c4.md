> 前阵子旷视的技术分享有个直播讲到其CUDA INT4的推理，这两天有了录播学习一下：https://www.bilibili.com/video/BV1wP411H7D2 。分享中提到：**整个模型哪个部分应该被量化为INT8，取决于模型开发者本身对精度和速度的平衡，** 这里及后文所说的INT4指的是int4_t（4bit），而非vector数据类型里4个int32t，需要注意。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/01.webp)

**图1 整个模型的量化方案**

其实在硬件上也有有一些经验供模型开发者参考。在CUDA上会使用Tensor Core加速，对同一个计算任务，带宽总量是一定的前提下，数据类型不同那么每秒钟传输的数据量是不同的，如下图右边 Roofline Model，

相同计算密度I的计算任务，理论最大带宽高低依次是INT4、INT8和FP32

。![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/02.webp)

**图2 软硬件结合经验**

因此，**Tensor Core可以用int4_t这种类型来实现相比32bit数据类型的加速，但要求其数据排布要遵循NCHW64的存储格式**，即相同宽高位置、连续的64个数排列在一起，即要求输入输出的通道数能被64整除，这是NVIDIA的要求。但作为**MegEngine框架，内部会对数据数据检查是否对齐到64，不满足则会补齐的倍数**。

## 0. Tensor Core与CUDA Core的区别

在知乎上有个提问：`请问英伟达GPU的tensor core和cuda core是什么区别？`（https://www.zhihu.com/question/451127498），下面摘要如下。

简单来说，**概念上，二者都是运算单元，差异体现在算力和运算场景，CUDA core是全能通用型运算单元**，而**tensor core专为深度学习矩阵运算而设计**，历史上先有cuda core再有tensor core。

> **一个cuda core 包含了一个整数运算单元integer arithmetic logic unit (ALU) 和一个浮点运算单元floating point unit (FPU)**。然后，这个core能进行一种fused multiply-add (FMA)的操作，通俗一点就是一个加乘操作的融合。特点：在不掉精度的情况下，单指令完成乘加操作，并且这个是支持32-bit精度。
>
> 比方深度学习中常用的计算：z=w*x+b，这些cuda core 在显卡里面是并行运算，就是说大家分工计算。cuda core越多，算力就越强，
>
> 于是，我们看到了接来的几代Kepler -> Maxwell -> Pascal，这些cuda core肉眼可见的增长。
>
> **虽然cuda core可以分工干活，但是对于一些场景，比如混合精度的矩阵操作，core算得似乎不是那么高效，于是NVIDIA就开始琢磨专门针对tensor的硬件单元了**，即tensor core，**tensor core 第一代是在volta架构上推出，专为深度学习而设计的**。也就是说他在tensor矩阵场景下算得更快，先上一段视频感受一下，在4X4 矩阵乘法运算时的差异。

图3 Pascal CUDA与Volta Tensor Core矩阵乘法计算效率可视化比较

> 其通过 FP16 和 FP32 下的混合精度矩阵乘法提供了突破性的性能 – 与 NVIDIA Pascal 相比，用于训练的峰值 teraFLOPS (TFLOPS) 性能提升了高达 **12 倍**，用于推理的峰值 TFLOPS 性能提升了高达 6 倍。这项关键功能使 Volta 提供了比 Pascal 高 3 倍的训练和推理性能。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/03.webp)

**图4 支持多种精度类型的mma指令**

> 首先，这个Tensor Core基础能力在于，**一个时钟周期内可以完成一个64 floating point 的FMA**，而cuda core是搞不定的，分多次。其次，Tensor Core也能堆叠，V100上面就堆了640个。而且tensor core经过了几次升级，其操作的精度更加丰富了，从最初Volta的Tensor Core只支持FP16，到Turing架构，扩展支持了INT8/INT4/INT1，再到A100再次扩展支持了FP64/TF32/bfloat16类型。当然性能也更强。
>
> 作者：知乎用户
>
> 链接：https://www.zhihu.com/question/451127498/answer/1813864500

图5 PASCAL CUDA与TURNING架构 Tensor Core不同精度下矩阵乘法计算效率比较

## 1. 哪些算子会用到INT4

好了科普完了Tensor Core，我们再看看那哪些算子会用到INT4。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/04.webp)

**图6 **软硬件结合经验：** Roofline Model**

不过在此之前，我们先看上图的 Roofline Model，横轴是计算密度I（FLOPs/Byte），可算出一个输出元素对应的内存占用（Byte）与其需要的计算量（FLOPs），注意这里是计算操作数，s不是时间单位；纵轴是性能P（FLOPS），S是时间单位是每秒（per second）；斜率是最大带宽B（Byte/s），也就是性能P除以计算密度I，**根据公式，FLOPS在分子分母上下都有被消去，只剩下分子Byte和分母时间，也即每秒钟的传输数据带宽**。

- 当应用的计算密度I小于拐点（图上红线与蓝线交汇处）时，性能P将由当前设备所能达到的最大带宽决定，即带宽受限（Memory Bound）；
- 当应用的计算密度I超过拐点时，性能P最高只能达到设备的理论峰值，即TP。此时，性能被理论峰值被限制，无法与计算密度I成正比，即计算受限（Compute Bound）。

**不同硬件的理论峰值不同，因而拐点出现的位置则不同，** Tensor Core在INT4类型上能达到的理论峰值，相比INT8增加了2倍，要想达到理论峰值，则要计算与访存的比例比较高才可以，卷积的一种实现方式矩阵乘法，就是计算与访存比比较高的例子，因而在INT4、INT8和FP32三种数据类型下，同样最大带宽下，INT4能传输的元素最多，比较适合使用INT4来计算。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/04.webp)

**图7 软硬件结合经验：适用于INT4算子的场景**

这个分享里，也就是上图左侧提到的除了大算存比之外的两种情况：**小feature map以及厚通道**：小feature map我的个人理解是Height和Width相对于Channel比较小，再加上Channel方向会补通道到64，feature map会变成一个瘦条；而厚通道，即Channel相对于H和W很大，和小feature map的情形类，在模型设计经验上来说，厚通道可以在同样性能上，获得更好的精度，**如输入输出通道是56，而框架会自动补齐到64，但如果模型开发人员将模型补齐到64，精度可能会更高**。

## 2. 如何优化INT4推理性能

但实际部署INT4还有不少坑，这里谈一下解决思路，不过分为三个方面：算子融合、张量存储格式、核心算子开发，先来说算子融合。

### 2.1 算子融合

老生常谈了，模型结构上的优化，**图融合通常是通过减少访存次数来优化性能**，特别是游离的散碎算子，比方计算量很低的RELU，即前面Roofline Model里的带宽受限算子。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/05.webp)

**图8 算子融合例子：左边将conv和relu融合，右边将conv-add-relu融合**

如上图通过左边Pass1，减少1次读和1次写，同理右边是Pass2，将Conv Add Relu做了合并，即减少了2次读和2次写。所以当我们对ResNet50的核心重复结构（下图左）进行融合后，成为下图右边的结构。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/06.webp)

**图9 ResNet结构片段经过2个Pass融合后，算子变少**

### 2.2 张量存储格式

图融合后，计算瓶颈由散碎算子转移到核心算子上，一般来讲是卷积，不过在此之前，我们需要考虑张量在MegEngine中存储格式，而MegEngine有多种排布格式：NCHW、NHWC、NCHWx（x可以是4/32/64）。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/07.webp)

**图10 MegEngine对CUDA支持的三类数据排布：NCHW、NHWC、NCHWx（x=4/32/64）**

其中，NCHW分别表示Batch大小、通道数、宽度、高度。而NCHWx，x表示相同H和W的每x个通道连续存储在一起的格式。

**NCHW64的排布方式使得4 bits类型的数据在传输过程能被连续访存，可以充分利用硬件资源的特点****。**

### 2.3 核心算子开发

MegEngine为了享受到Tensor Core加速，而且Tensor Core本质上对矩阵乘有很好的优化，因而卷积通过矩阵乘来实现。

见下图左边上部分image to column算法，是将卷积计算中的输入feature map（即下图右边公式中的X），进行变换，变换后得到矩阵乘法的矩阵B（会比原始输入feature map会大一些），而权重（Weight）是W，进行一个reshape操作，reshape不改变数据摆放的位置，只是对原始数据的维度重新生成一下变为矩阵A，其宽为 `ic*ih*iw`（ic是input channel、ih和iw分别是输入的宽度和高度），其高为'oc`（即output channel）。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/08.webp)

**图11 卷积转为矩阵乘法的计算过程**

经过矩阵乘法，得到结果矩阵C的高、宽、通道数分别为 `oh、ow、oc`。上图右边是使用隐式通用矩阵乘法计算的公式与伪代码，其核心还是通用矩阵乘，表现为三层for循环，通过i、j、k三个循环索引，可以还原出weight的下标，feature map的下标。

隐式矩阵乘法（Implicit GEMM）在cutlass文档里有提到，表示将卷积表示为矩阵乘法的方式。

> Implicit GEMM is the **formulation of a convolution operation as a GEMM** (generalized matrix-matrix product). Convolution takes an activation tensor and applies a sliding filter on it to produce an output tensor.
>
> 参考：cutlass/implicitgemmconvolution.md at master · NVIDIA/cutlass (github.com)

在实现上，参考了NVIDIA开源的cutlass的类矩阵乘算子的编程框架，cutlass适配了多GPU硬件的多显存布局以及CUDA编程模型，里面的高性能算子遵循分层分块策略。

下图上面部分：每个thread block tile访存global memory，并将分块需要的数据写入shared memory，一个warp tile以及thread tile将数据从shared中取出到regitster中做计算，而Epilogue是用shared memory做如bias、激活函数的计算等，并将结果写回global memory，以上就是cutlass conv的过程。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/09.webp)

**图12 cutlass实现的高效率矩阵乘法充分利用内存层级结构实现延迟隐藏**

上图下面部分展示的是，这个过程在不同阶段的执行时间上是可以相互掩盖，如在访问下次迭代的global memory时，当前迭代会对shared memory访问并计算，在下一级循环中，取下一个迭代的shared memory的同时，CUDA Core、Tensor Core还在进行各种当前迭代的数学计算。cutlass中的具体计算细节不赘述，可以参考下面两个文档：

- https://github.com/NVIDIA/cutlass/blob/master/media/docs/implicitgemmconvolution.md
- https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md

#### 2.3.1 技术改进

下面介绍MegEngine在cutlass上的技术改进，Tensor Core也是NVIDIA为了加速矩阵乘法而设计的硬件单元。PTX文档中，对INT4的指令描述是 `mma.m8n8k32`,，m、n、k后面的数字是矩阵乘法中各自的维度，即m=8、n=8、k=32，一个Tensor Core的mma指令，可以计算m=8、n=8、k=32的矩阵乘法。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/10.webp)

**图13 使用mma指令实现局部矩阵乘法的实现1**

一个warp是32个线程，其中排布的方式即上图Fragment A、B，且二者数据类型都是INT4/UINT4，矩阵C的类型是INT32，Fragment A和B一个格子是8个INT4，即32bit，而Fragment C的一个格子是2个32bit，需要注意的是，结果矩阵是均匀地分布在32个线程（即1个warp）的寄存器中，因为每个mma算得是8x8的结果，一个warp只有32个线程，所以1个线程会处理得到2个32bit的结果。

CUDA编程中针对global memory的访存性能有下图经验准则：

1. 首先字节（1byte=8bit）是访问global memory的最小单元，因此对INT4类型而言，需要凑齐2个（int4_t）数据才能填满1个字节；
2. 相同warp中的连续线程，访问连续的地址，会触发访存合并；
3. 由于l2 cache line是128bit的，所以最好用64bit或128bit来访存，这样所需的访存指令会少且性能更好

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/11.webp)

**图14 global memory访存准则与局部矩阵乘法的处理流程**

NCHW64存储格式的特点是，每隔64个通道数据是连续存储的，所以在Tensor Core指令处理时，如果像是上图左边，直接对Fragment A和B乘，得到结果矩阵C，那么这个结果矩阵C在OC的维度上是不连续的——因为结果矩阵的行是OC，而不是列是OC。导致不满足经验准则2和3。由于①结果类型是int32，在写回前，会做重量化变回INT4，②结果矩阵C中每个OC不连续，所以写回时甚至凑不满一个字节，导致准则1也不满足，需要线程间做数据交换来凑齐字节数，才可以写回，导致性能巨大下降。

**因而，MegEngine在实现该过程中，对A和B矩阵交换了顺序，即将Fragment A视作Fragment B，Fragment B视为Fragment A，来做矩阵乘，这样结果矩阵在OC上连续的，这就满足了经验准则1，且对性能友好。**

---

**为了迎合Tensor Core中数据的排布方式，CUDA提供从shared memory取数据到数据寄存器的指令ldmatrix，** 除了对Tensor Core指令做了修改，**MegEngine还适配了ldmatrix指令**，下图以weight矩阵的读取为例，展示了ldmatrix的操作语义。

像下图展示的，CUDA中一个warp的32个线程，1次ldmatrix将取到4次mma所需要的数据，初始形态下每个thread指向shared memory中的连续4个int32，共128bit的数据，经过ldmatrix后，数据将按照中间逻辑的关系分别存放在32个线程中。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/12.webp)

**图15 使用ldmatrix指令对数据做读取操作：得到4次mma的数据**

若单独看一个线程如thread0（上图右边），thread0取到的数据在OC的维度上就是oc0、oc8、oc16和oc24。

下面我们考虑单个warp的矩阵乘法，经过一次ldmatrix A和ldmatrix B后，我们分别对这两部分都取到4块数据，见下图，那么我们可以进行16次的mma计算，根据output数据排布的规律，每个线程持有的结果并不连续。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/13.webp)

**图16 局部矩阵乘法的ldmatrix与mma指令调用细节描述**

以thead0为例，取到的是oc0、oc8、oc16、oc24，又因为结果是均匀分布在32个线程中的，1个线程均摊2个32bit的结果，那么我们t0（即treahd0）持有的结果数据是oc01，oc89、oc1617、oc2425，虽然t0、t1、t2、t3的4个线程加在一起写的是连续的地址，但t3和t4写的地址是不连续的，因为t3和t4写入地址在图上处于与t0和t1不同的行，那么就打破准则2（不符合每个地址的地址连续，对性能也不友好），由于每个线程写入地址的数据是1个字节，因此也不满足准则3。

**为了更好的适配ldmatrix的行为，MegEngine提前排列好了原始数据在oc维度上的顺序，按照下图左边的顺序存放数据，那么ldmatrix后，线程0取到oc维度上的索引就是oc0、oc2、oc4、oc6，这样在经过mma计算后，每个线程持有的结果将在oc上连续。**

需要注意的是，在oc上对数据的重排，都是在weight矩阵上进行的，因为weight是卷积权重，训练完后是固定的，**可以提前重排好不会影响运行时的性能**。因此这个也是megengine做的，前者能拿到更多的计算图信息，知道哪个矩阵是静态地在运行时不会改变，这也是cutlass不合适做这个的原因。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/14.webp)

**图17 重排oc方向后ldmatrix的读取方式**

经过重排后，每个线程持有的数据已经连续了，如thread0持有oc01234567这8个，即这8个int4，可以拼成1个int32，再经过epilogue操作后，共占用1个int32空间大小，且每个线程写回的地址时连续的，满足了经验准则2——可以触发访存合并。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/15.webp)

**图18 重排后需要2次ldmatrix对应16次mma指令调用的细节描述**

但**目前仍然存在优化空间：**

1. **NCHW64中每64个通道是连续的，但此时我们只计算了32个通道；**
2. **根据global memory的性能准则第三条，32位写回，并不足够好。**

所以，在实际中，可以分别对A和B做2次ldmatrix（对应A和B分别8个， `8*8=64`），那A和B分别有即对应64次mma操作，见下图。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/16.webp)

**图19 4次ldmatrix与对应64次mma指令调用的高效组合**

此时，我们便进行了NCHW64中连续64个OC都计算得到了，且每个线程持有2个int32，以64bit位宽写回global memory，比优化前的32bit写回的实现更好，且所需访存次数更少，性能也更好。

## 3. 性能&总结&展望

性能测试基于Ampere架构的A2卡上测试ResNet50 Batch=128，TensorRT版本7，见下图**TRT的性能被标准化为1，MegEngine的INT4推理速度相比TRT INT8快了将近1.3倍左右**。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/17.webp)

**图20 MegEngine-v1.11.0 vs TensorRT-v7.2.2.3@A2@ResNet-50@INT4速度比较**

而在精度上的表现，见下图，则与TRT差不多。

![文章插图](/articles/2022-09-22-megengine-cuda-int4-tensor-core-3e1b09c4/images/18.webp)

**图20 MegEngine-v1.11.0 vs TensorRT-v7.2.2.3@A2@ResNet-50@INT4精度比较**

最后有个MegEngine对低bit量化的小总结，其实还有待挖掘的事情：

1. 在模型的实践上，更多业界模型还有待挖掘其在INT4量化下的精度和性能表现；
2. 目前只做了CUDA INT4，其他计算设备还没有适配。

---

​有GPU卡的同学，根据下面链接里的文档，将数据准备好后，可以一键测试，感受CUDA INT4的推理速度✈。

**https://github.com/MegEngine/examples/tree/main/int4_resnet50_test**

**更多相关文章：**

- **[深入浅出 | 谈谈MNN GPU性能优化策略](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885606&idx=1&sn=4dbf5d6c52018485eea4076d4b61c1de&chksm=b29043c485e7cad256f73405a2fafecead5f36a99da7dc43ad658c4cf0fa7c5d44ab4f9da677&scene=21#wechat_redirect)**
- **[华为诺亚方舟实验室 Bolt v1.2.1 在 QualComm GPU 上的优化总结](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886197&idx=1&sn=5e99d9a8fc07d16047e4924b72830bf4&chksm=b290419785e7c881bd0dfbadc29c69cc1004ade09225057463c4a35f3da4771d39831e627365&scene=21#wechat_redirect)**
- **[移动端GPU矩阵乘优化](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885554&idx=1&sn=7f3078d27555c151e05aa967b5de68b8&chksm=b290421085e7cb06432f9d4fe2e7009819a3cd322c4e7cb86fff7ceaeddd31c2ad451ed9d4a5&scene=21#wechat_redirect)**

> 点击【阅读原文】，看往期文章
