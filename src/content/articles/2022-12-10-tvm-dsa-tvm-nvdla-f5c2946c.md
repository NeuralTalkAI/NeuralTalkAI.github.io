> 本文转载自知乎（https://zhuanlan.zhihu.com/p/571045566），作者凉风（https://www.zhihu.com/people/banzhuantou123456），是**一位GPGPU/NPU方面的从业者，目前聚焦于AI编译器的工作**，欢迎大家关注。
>
> **注：对原文做了一些修改和补充。**

## 前言

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/01.webp)

**图 AI加速器的工作流**

目前AI DSA（领域专用架构）架构五花八门，不像CPU/GPU那么统一，面向DSA的编译软件栈也是各种轮子，未来何时能收敛是大家密切关注的话题。目前看来，硬件收敛应该还需要一些时间，毕竟DSA是结合应用场景定制的产物。

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/02.webp)

**图 各种类型的AI芯片**

**对于编译软件栈来说**，早些年的时候**基于Caffe/Onnx甚至NCNN进行扩展开发**，那个时候也更习惯叫AI工具链。**用户换一个DSA平台就有新的学习成本，配置复杂还会导致很难发挥DSA性能。**

因此在**造新的DSA编译软件栈轮子的时候，按照传统编译器的思想** 进行解耦分层、减少用户的上手成本、性能提升等角度考虑。目前TVM/MLIR等开源AI编译器生态有一定起色，于是进入了TVM的坑（端侧对TVM的接受度更高）。

**对于TVM来说，针对CPU/GPU等通用架构做了很多优化工作**。我们的开发没有走[**BYOC路线**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887008&idx=1&sn=6ce7355369abbbaf987b1e44d040827a&chksm=b2904c4285e7c5548038d091351add76261ff2f715d20b709e0c5cb9053ecf18a3e2b91408f3&scene=21#wechat_redirect)（**自身也没有现成编译软件栈**，都需要0-1造所有轮子），于是就**想像CPU/GPU那样，走relay->te->tir->(llvm)->dsa的路子，想沿用TVM上各种优秀的feature**。

1. DSA优化基本思路

以CPU来说，cache不能显式操作，需要利用数据局部性来减少cache miss提高算力利用。对于DSA来说，是由一个可编程操作但容量较小的SRAM进行数据调度，需要尽可能的减少数据搬运而导致DSA处于空闲状态。

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/03.webp)

## **图 CPU与DSA结构**

### a、DSA指令粒度

对于CPU来说，Conv最终都归结为乘加运算。但对于DSA来说，Conv内部的操作已经固化在硬件里面，只需要进行输入输出配置就行了，下面是DSA上配置CONV指令的函数接口示例：（数据计算流可参考tflite）

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/04.webp)

40+的参数需要一次性全部配置才能运行Conv操作，粒度非常粗，一个for循环也没有。

### b、优化思路

对于CPU来说，layer-by-layer优化，减少cache miss。layer-fused是DSA上优化的一种思路，指定是把多个CONV进行切分后运算，TVM是layer-by-layer的，第一个conv全部完成才会开始第二个conv，两种不同的思路。我们主要做的就是把layer-fused优化思路放到TVM里面去。

> conv 这种操作几乎就是在 height 方向和 out channel 方向是解耦的，那么 height 和 out channel 分解得越多单次 memory cost 就越小，小到可直接放到 SRAM 里面；**layer-fuse 就是连续几个 conv，都每次只算单个 conv 的一个分解，而不是算完整个 conv 再算下一个**，这样就**把 conv 之间的 DDR 读写给省了**。这种做法有的文章里面叫 layer tiling。

对于DSA来说，layer-fused优化，减少中间数据倒腾。举个实际例子，假设有三个卷积如下：

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/05.webp)

**图 截取自resnet50**

如果**按照layer-by-layer的思路，对每个Conv output_h切割成2份**（**假设只有这样才能放入SRAM**），假定DSA有512个MAC，数据总线宽度为128bit，可以如下折算layer-by-layer总共花了1159328个cycle。

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/06.webp)

**图 layer-by-layer cost cycle**

如果**按照layer-fused的思路，同样对每个Conv oh切割成2份**，Cost折算如下：

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/07.webp)

**图 layer-fused cost cycle**

layer-fused总共花了1141152个cycle，比layer-by-layer的方式减少了18176个cycle。

此外，数据的总带宽也变少了，表格的总线latency都是按照0测算的。

> **在SoC上，** **DSA的总线响应优先级比较低**，outstanding配置也会被限制，一次burst可能需要数千cycle才能被响应。对应的cycle就多出数倍甚至数十倍。**这也是单跑DSA和SoC满载业务下的性能差距比较大的重要原因**。

性能上看起来是有提升了，需要注意的**layer-fused减少了中间conv的数据搬运，但是会带来计算量（1111040>1103872）的增加**，这是**因为切分多卷积后输出锥形减少的，输入存在重叠部分导致计算量上升**。

**fused的layer数量不是越多越好，需要结合硬件特性、网络模型结构综合来看。** 当然这里只是一个**基本的优化思路**，**实际操作要考虑SRAM容量、切割调度策略、满算力约束、系统带宽等诸多因素**。除了layer-fused优化思想外，也还有一些其他的优化思想。

c、性能指标

目前来说，绝大部分的DSA宣传跑resnet50是多少多少fps，对于用户来说无可厚非。但是**开发人员要计算一下MAC利用率来推测性能水准，而不是看fps的绝对值****。** 以NVDLA为参考如下：

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/08.webp)

**图 NVDLA运行Int8 ResNet50时，不同MACs单元下的性能（fps）**

以1T算力来说（512MAC），resnet50 MAC利用率为70.3%；对于4T（2048 MAC），MAC利用率为50.8%。1T算力下，毕竟面积小，这个MAC利用率算正常，当然可以增加一些feature来提升性能（面积功耗代价）。1T的resnet 50是可以把MAC利用率做到85%以上（利用率：fps*模型需要算力/芯片每秒钟算力，这里MAC是MAC Unit）。

**对于4T以上算力来说，NVDLA架构确实存在一些需要改进的点**，所以MAC利用率下降比较多，做**一些改进是可以将MAC利用率进行提升的**，比如4T下resnet50到340fps，8T下Resnet50到650fps都可以不需要增加太多的开销。如果要**更高算力可以通过叠加更多核来实现**，对应的MAC利用率也会有一定下降，但不能太多。

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/09.webp)

**图 不同算力下的Resnet50性能**

不同的DSA对应着不同的场景，关注的模型是不一样的，这里用最常见的Resnet50来举例在不同算力下的MAC利用率情况。

### d、PPA Trade-Off

如果只看模型fps的话，是可以通过硬件堆MAC（增大面积、提高了芯片的成本）来实现的，因此我们在看fps的同时也需要关注设计算力。当然，仅仅看MAC利用率是不够的，除了fps/T之外，还需要关注mm^2/T，W/T，MB/T等指标综合对比。

**就以93 fps/T来说，业内也很少看到有芯片能达到这个性能**。**一方面是优化是否到位，另外一方面则是因为芯片需要进行功耗面积妥协**。如用20%的面积来换10%的性能，就是一个非常划算的Trade-Off，毕竟在这样性能折损下MAC利用率也是比较可观的。

## 2. TVM中的挑战

前面说了DSA的指令粒度、优化思路、性能结果，在**TVM中进行落地开发面则临的一些对应的挑战**。

- a、指令粒度：由于DSA有的指令粒度粗，而relay/te非常细粒度，需要进行packed，尤其是在**选配参数问题上一不小心就容易出Bug**；
- b、layer-fused：从TVM的层次结构看，layer-fused放在relay层还算比较合适，但是一组layer-fused里面有很多conv等各种算子，因此**lower的时候需要考虑拓扑结构和大量参数的传递**；
- c、tensor memory：进行layer-fused的时候，模型中间的tensor是不需要进行DDR交互的，因此涉及到内存分配和优化的问题；
- d、inferbound：比如原始in shape=[1,ic,14,14],ksize=1,kstride=2,这里out shape是[1,oc,7,7]。TVM的Inferbound会把in_shape变成[1,ic,13,13]，**看起来减少了数据量的操作，但是对于DSA来说会显著增加带宽消耗**；
- e、tir：指令排布、并行调度、时间片分配，**SRAM管理**等需要较多复杂的pass进行处理；
- f、cost-model：当前的cost是**直接利用程序运行的时间差** 来计算的。DSA一般是具备Cmodel进行cycle模拟的，而cycle是不关注程序执行时间，只看指令配置的。通过RPC连接DSA芯片在回片后也会介入，这样得到的最准确。此外还有以带宽作为cost的，毕竟SoC上系统带宽对性能的影响极大甚至挤压其他模块。

但是对TVM支持过程也有不少挑战与问题：

1. **粒度很粗的DSA**要用起TVM来确实很麻烦，很多功能**需要自己开发且受限**于TVM的一些约束；
2. **tvm与mlir的选择上**。如果基于MLIR来开发，可以自己发挥度更高。但是要打造各种完备的基础功能，在公司里面落地项目来说太难了（团队人员数量与能力要求）；
3. **dynamic shape和control flow确有困扰**，看具体项目的需求了。做项目来说确实很快可以去交付一个不错的软件栈；
4. 深入**改造/Debug TVM确实很费解**。

## 3. 硬件的一些思考

从PPA的角度看，软件能直接可见的是Performance。AI DSA芯片本身就是基于CNN定制化的产品，因此AI编译器的开发也需要对于Power、Area密切关注，可以从更好的维度进行优化提升。

### 1、架构

对于端侧来说，从layout的角度看，DSA架构看起来其实也就那么几种，大家基本上都会参考NVDLA(NCHWc)。评估一个架构，既要结合落地场景需求，也要看看架构本身的性能、带宽瓶颈(系统带宽与SRAM带宽)。

### 2、面积

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/10.webp)

**图 PPA影响因素**

面积分两部分，DC面积和PR面积。

- DC面积：主要是**面积性能的trade-off**，毕竟DSA是一个需要结合场景需求进行定制化的。比如对某些必须的配置做性能增强，对一些不太常用的配置进行面积裁剪，或者可以通过软件进行规避的配置，开合适大小的FIFO，前后级模块的流水承接。AI DSA芯片设计不仅仅是DE人员的事情，作为**AI编译器开发**人员，需要**结合场景算法特点、优化策略参与**进来，**对面积性能**都有较好的帮助。
- PR面积：从物理实现角度上来看，内部各个模块的设计复杂度，是否有存在绕线拥塞问题导致物理实现代价比较大的，这个需要结合架构来看。

### 3、功耗

功耗分成峰值功耗和平均功耗：一般来说峰值功耗是通过错峰来对应，但核心问题还是怎么减少。比如设计阶段**尽可能考虑减少数据的翻转、增加复用**。而DSA本身是定制化的，那就进一步**从cell的级别进行定制化固化**对功耗就非常有帮助。功耗仿真分析的周期实在是太长了，前期设计的时候需要尽可能的考虑到可能的问题。

## 4. NVDLA架构思考

知乎上有不少NVDLA的文章已经分析的很清楚，作为一个开源的DSA架构，具备很高的参考价值。

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/11.webp)

**图 NVDLA核心内部架构**

### a、满算力约束

前面提到的4T以上算力的MAC利用率下降的比较厉害。NVLDA的MAC是按照icowoc来进行配置的，其中ic=oc。比如512MAC就是888，2048MAC就是16816，4096MAC就是161616，也就是**ic/ow/oc是对应的整数倍才能把算力打满**。一般来说模型的第一层是13224*224，ic=3与8/16就不能将算力打满，ow=8/16，但是模型的输出是7/13这种也会存在算力的浪费问题。

### b、带宽功耗

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/12.webp)

**图 NVDLA系统软件的内部数据流**

取数进行计算的量对SRAM带宽和功耗有较大的影响，NCHWc的小c是8的整数倍，取数比较方便。进一步考虑**通过数据的复用来减少取数的频率**，**对功耗有很大的帮助**，当然也会带来一些面积的开销，需要结合具体情况来取舍。

## 5. 总结

做一个自定义推理框架想把性能做好都很不容易，要结合TVM来深度开发就完全是给自己挖了个大坑。从当前的工作来看，总体上把TVM串起来了，同时性能上也有不错的结果，但是还有很多地方需要完善。在开发的过程中，**软硬件的密切交流配合****对PPA的提升****非常有帮助**，

**对AI编译器开发** 的人员来说，能**更好的理解各种优化策略的影响与局限**，并且结合场景也可以**让芯片PPA指标更有竞争力**，也会让工作更有趣。

![文章插图](/articles/2022-12-10-tvm-dsa-tvm-nvdla-f5c2946c/images/13.webp)

**图 PPA分析流程**

对于AI来说，生态问题一直是大家密切关注的点，有**TVM的生态**、**优异的性能**、**灵活的异构** (DSA+GPU/DSP)，相信**未来在端侧会有更大的落地空间与更小的落地成本**，并成为行业发展的趋势。欢迎各位大佬指点交流，一起学习提高。

## 相关文章

- [AI、ML加速器发展趋势综述](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886740&idx=1&sn=4e66cc0346e914168191425145cc8c4a&chksm=b2904f7685e7c660f2bfce6bb3738aeffd4a030dfb273b068b0ec30137cecf3f1e79ece0970d&scene=21#wechat_redirect)
- [新一代深度学习编译技术变革和展望](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886255&idx=1&sn=1b55f4fe1335260a6086082577ec8951&chksm=b290414d85e7c85bbe0b14e41e3b36b6266bb51f1a4dee67d9d4eef0f797e2eb231e1de406f4&scene=21#wechat_redirect)
- [AI 硬件公司接入框架与 AI 框架算子层级的思考](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886093&idx=1&sn=bc56c3a0843a39dbd66b0b513f5803b9&chksm=b29041ef85e7c8f9c484a5a2e5646d8a8822b37b98fce0d6d57fab0e1ea12b5d9c7b09541336&scene=21#wechat_redirect)
- [谈谈AI编译软件栈、软硬件协同、TVM和MLIR的看法](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886637&idx=1&sn=477371ef67890c91b565b383493959b6&chksm=b2904fcf85e7c6d9fd752119c1487b08a2e9c8c7d847067f6b481467a862ad0508100470a792&scene=21#wechat_redirect)
- [BYOC：加速器硬件接入深度学习编译器的统一框架](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887008&idx=1&sn=6ce7355369abbbaf987b1e44d040827a&chksm=b2904c4285e7c5548038d091351add76261ff2f715d20b709e0c5cb9053ecf18a3e2b91408f3&scene=21#wechat_redirect)
- [MegPeak——让你更懂你的处理器：RooflineModel/最优指令组合/探索优化空间](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886335&idx=1&sn=39fd09c6873580df3e31da1070ee9f7a&chksm=b290411d85e7c80b46dd7ca21ed1894c2d5c1c43c82f9c71983b56f02eae56139b329f199122&scene=21#wechat_redirect)
- [浅谈C/C++性能优化背后方法论：Top-down Micro-architecture Analysis](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886603&idx=1&sn=5a003bbe593e571325eded84f4117632&chksm=b2904fe985e7c6ff4c36cf01774edca8ad0a2f3412687470bd387beecdff947b620e85e9ec67&scene=21#wechat_redirect)

**- 点击【阅读原文】，查看往期文章 -**
