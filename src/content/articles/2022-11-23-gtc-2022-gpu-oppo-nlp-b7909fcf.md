导读：在已经结束的Nvidia GTC 2022 会议上， OPPO数智工程事业部机器学习团队分享了主题为

《Evolution Path of GPU Inference Accelerating on Breeno Bot/NLP Scenario》

的报告，主要讲述在GPU推理加速在小布助手NLP场景的演进路线，包含以下4个模块内容：

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/01.webp)

1. 此次技术优化的业务背景，及面临的挑战

2. 解决方案，及推理加速的演进过程

3. 此次GPU推理加速实践的具体工作

4. 后续的技术演进方向与规划

---

**一、业务背景及挑战**

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/02.webp)

小布助手简称小布，也叫Breeno，是

OPPO智能手机和IoT设备上内置的多模态AI助手

， 包含语音、建议、指令、识屏和扫一扫5大能力模块

。小布助手以“机智”、“有趣”、“温暖”为理念，致力于

提供多场景、智慧有度的用户体验。

自然语言处理（Natural Language Processing, NLP）是提升AI助手能力的关键技术。大规模预训练语言模型是NLP领域近几年来取得的最显著的技术突破，在很多NLP下游任务上都显示出了压倒性的威力。然而，

由于大规模预训练模型本身结构复杂，计算量巨大，在落地应用方面存在推理成本高、推理预测性能难以满足在线业务需求等问题

。小布助手也包含信息查询分类、 生成式闲聊、 指代消解、 文本向量表征

等应用场景， 需要使用NLP大规模模型来提升任务的效果

。在小布助手预训练模型落地的场景中，我们探索了一系列GPU推理加速技术，使模型推理性能不断提升，同时也构建了一套高性能推理服务框架。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/03.webp)

基于小布助手的应用场景特性，我们的通用推理服务框架需要满足以下需求：

**(1) 灵活性**：能够**支持不同的场景，包括NLP/CV**，也包含搜索/个人化推荐/广告等；在推理服务维度，也需要**支持不同的推理引擎**以及**异构的AI硬件设备**。

**(2) 功能性**：依据性能需求支持**动态batch合并**，提升吞吐能力；支持**多模型连续推理**。

(3) 高性能：能够支持大规模的模型，并且服务响应时延Latency和成本将会成为我们性能优化首要考虑因素。

**二、解决思路及演进过程**

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/04.webp)

基于这些需求，我们的解决思路是这样的：

1、选取合适的推理引擎，在运行时，它需要**满足低时延高吞吐的性能要求**，同时也能**支持CPU、GPU设备的推理，支持多种模型场景**

2、利用**Cache缓存的方式**，来**提升** 推理服务的**吞吐能力**

3、应用**MPS技术**，帮助**多进程共享context**，**减少**context**切换的开销**

4、应用**Multi-stream多流**优化技术，通过**不同streams之间共享GPU上下文提升GPU核心并发利用率**，同时也可以提升内核job的并发度，减少内存显存拷贝的时延。

5、使用**最新驱动版本的GPU**，因为新版本包含了Nvidia最丰富的优化经验沉淀，往往拥有最优的性能；在**升级过程中，我们需要考虑在线服务滚动升级问题**

6、利用**图优化**，子图融合、高性能算子等方式帮助我们**规避低效的子图**，获取更好的模型计算性能

7、在**调度框架方面**，使用可**灵活调整的batch size策略**，推理**服务与业务逻辑解耦**，**专注模型推理性能**

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/05.webp)

基于上述解决思路， OPPO在线推理服务的实践经历了五个阶段

（如上图所示）：

TensorFlow推理引擎、TVM编译优化、Model Structure优化、Nvidia生态优化、Subgraph优化。

在这个演进过程中，每个阶段我们做了如下工作：

1. 在TensorFlow引擎推理阶段，我们使用了GPU推理对NLP大模型进行加速，

相比于CPU推理性能提升了9倍，推理时延6.6ms， 开启了NLP大模型上线部署之路

2. 在TVM编译优化阶段，我们运行了

CUDA、CuBLAS计算库、TVM Ansor等优化方式，相比于TensorFlow GPU推理降低了45%的推理时延，提升了80%的吞吐量，解决了流量高峰期产生的超时问题

，使得NLP大模型推理变得更稳定；同时MPS技术落地也在对应场景带来了12%的时延降低

3. 在

模型结构优化阶段，我们使用了ONNXRuntime引擎，结合Transformer优化

，获得更好性能的同时，支持了模型的Dynamic Shape功能，顺利支撑了变长序列的业务场景上线

4. 在Nvidia生态优化阶段，我们进一步使

用FP16混合精度的方式提升计算性能，同时Multi-Stream多流技术、CUDA Driver升级

也都为模型推理带来了不小的性能提升5.在

Subgraph优化阶段，我们深入模型结构，在计算图、算子维度进行深度优化

，在部分子图融合场景可以带来10倍的性能提升下面我们将详细解读上述推理加速技术的演进过程。

**三、GPU推理加速实践的具体工作**

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/06.webp)

第一阶段：TensorFlow引擎推理

在应用BERT等预训练模型之前，我们使用简单模型在CPU上进行推理，推理框架使用TensorFlow/Pytorch。然而BERT模型计算量特别大，在CPU上的单次推断耗时往往需要十几甚至上百毫秒，远远超过业务场景可接受的范围。

鉴于BERT模型或者Transformer网络拥有大量的矩阵张量乘法操作，这一特性是非常适合使用Nvidia GPU进行加速的

。在小布助手的场景中，对话的语句以短Query为主，通常长度小于32个字词。因此我们设置了Benchmark场景：

- 模型：基于BERT-BASE，参数量：1.1亿
- Batch size: 1，Sequence length: 32
- Hardware: CPU: Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz、GPU: Nvidia Tesla T4

单次Query推理TensorFlow-CPU需要55ms，此时单颗CPU设备的利用率是100%，在Tesla T4上，单Query推理只需要6.66ms。这意味着使用T4 GPU加速后，我们可以得到9倍的性能提升。升级GPU加速后使得我们语义理解等场景达到了基本的上线需求。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/07.webp)

第二阶段：TVM编译优化

在使用TensorFlow-GPU加速BERT模型之后， 单并发请求的推理时延降低至了6.66ms，然而由于线上服务存在较大的并发请求，在高峰期部分请求存在超时（场景限制响应时间需要小于20ms）的情况，因此需要进一步优化推理的耗时

。此时我们应用了深度学习编译器TVM（Tensor Virtual Machine）。TVM借鉴了传统编译器LLVM的思路，它是一个端到端的深度学习编译器框架，可以加速模型在Nvidia GPU、CPU、NPU、手机芯片等Target设备的执行，特别在计算密集型场景有这不俗的优化效果。作为一个开源项目，TVM社区技术也不断演进，TVM Ansor等技术的出现进一步提升了模型在各设备端的性能。

通过应用与优化NLP生产模型在TVM编译中的性能表现，单Query的请求耗时优化至3.68ms，相比于TensorFlow-gpu的方式减少了45%的推理时延，同时单卡的吞吐量也提升了80%。

TVM的编译优化优雅地解决了流量高峰期产生的超时问题，使得NLP大模型推理变得更稳定。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/08.webp)

第三阶段：模型结构优化

随着越来越多的服务使用大模型推理，GPU推理成本也逐渐升高，需要我们进一步探索

更低成本的方式

， Transformer网络结构优化以及低精度计算

是我们关注的方向。

ONNX（Open Neural Network Exchange），是由Microsoft、Facebook等公司为解决模型转换与部署的开源项目。随着ONNX生态的发展，ONNXRuntime迭代迅速（接近3个月一次大版本Release），其易用性、生态发展、性能表现也越来越出色。

在小布助手的场景中，Query的长度通常小于32， 相比于TVM，ONNXRuntime擅长计算图和访存密集型算子优化，它的推理耗时略优于TVM

（TVM在Sequence_length大于64时，有更好的性能表现）。

另外， ONNXRuntime社区针对Transformer网络有定制优化工具，专项优化了Layer Normalization、Attention、Gelu等结构与算子

，可以带来一定的性能提升。

在精度优化方面，Transformer优化过程中，可以将部分算子使用FP16精度计算，利用Tesla T4卡的TensorCore计算单元加速

，同时以黑名单的形式将精度敏感的算子，保持使用FP32进行计算。混合精度的计算方式可以在保持推理结果不变的前提下，更大限度的提升推理性能。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/09.webp)

在推理性能和准确率评测上，ONNXRuntime 表现出了更好的效果，在如上图的Query理解分类场景，FP16混合精度的应用在分类结果的偏差可以控制在FP32精度的0.2%偏差之下，符合其业务上线要求。

ONNXRuntime单Query的耗时可以控制在1.85ms，相比TVM Ansor方式进一步降低了50%，达到2倍的性能提升。

在推理结果精度、推理性能方面ONNXRuntime Transformer优化结合FP16混合精度的方式拥有最佳的性能表现

，更低的成本，因此广泛地应用在了小布助手等NLP模型推理场景。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/10.webp)

第四阶段：Nvidia生态优化我们先来看下Multi-Stream多流的优化。

Nvidia GPU的调度在默认stream机制下，CUDA tasks将按顺序一个接一个执行。然而，

CUDA也提供了多个CUDA tasks放置到多个streams执行的能力，我们可以用不同的stream来并行执行计算任务

。Multi-Stream可以节省创建CUDA上下文的时间，同时也能提升GPU核心并发的利用率和计算并行度

。Multi-Stream在GPU利用率较低的任务场景可以获得不错的收益，然而我们的场景中GPU util利用率较高，在上面的GPU Profiles分析图中，Streams 会花费更多的时间在等待资源分配，并且相互阻塞，因此没有获得正向的收益。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/11.webp)

多进程服务（MPS，Multi Process Service)使用时，tasks之间有更好的隔离性，他们可以共享GPU核心。不同的Process之间具有显存隔离的能力，也可以指定每个进程的GPU 核心使用率。

Nvidia的MPS使用起来的比较简单的，它主要包含三个组件：

(1) Control Daemon Process：负载开启和关闭MPS Servers、连接clients和servers

(2) Client Runtime：构建于CUDA中，并且可以无感的被CUDA调用。

(3) Server Process：负责与Clients连接，并且提供GPU的并发能力。

MPS技术在OPPO的个性化推荐场景中，TP99时延降低了12%。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/12.webp)

第五阶段：Subgraph优化

为了获取更好的性能表现，我们也尝试了Subgraph图优化策略。

策略1：将低效的算子和计算方式转变成高性能的算子组合。

Split+Concat”算子将会创建1024个split操作，带来大量重复冗余的计算。

替换为”Reshape+Expand”之后，它在计算逻辑上可以保持与原来一致的计算结果，但是它的计算性能可以提升10倍。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/13.webp)

策略2：自定义算子与算子融合。

例如，上面的例子中我们将左边这6个算子组成的pattern，自定义为BiasGelu融合算子，这个

融合减少了算子调度引起的global memory拷贝，带来了2倍的性能提升

。此外，对于

推理引擎中已有的算子，有的实现效率会比较低，也可以修改其代码使之更高效。

但这种需case by case去抠细节，效益没那么明显。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/14.webp)

第六阶段：在线推理服务优化上文中有介绍到我们选取推理引擎的标准: （1）支持GPU与CPU推理（2）支持Dynamic shape （3）支持算子融合（4）支持Multi-Stream多流等等在调研和对比了各种推理引擎之后，我们选取了

ONNXRuntime作为最核心的推理引擎，同时也将支持不同的推理后端

，增强推理服务的灵活性。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/15.webp)

接下来的优化工作为高频Query的缓存Cache方案。

如上图所示，这里统计了一段时间内小布助手收到的用户Query请求频次，例如“小布小布”被请求了1331次，是最高频的请求。

考虑到高频语句请求场景，经过统计分析，如果我们

缓存TOP500的高频Hot Query，它可以帮助我们减少30%的模型推理请求；如果缓存最近20万条请求的话，可以减少60%的模型推理开销。

据此，我们设计了Cache缓存方案，以一部分Cache开销，获得了更好的整体响应性能。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/16.webp)

通常，保持使用最新的Nvidia CUDA Driver是推荐的方式，因为在驱动升级的过程中，CUDA Toolkit、cuDNN等配套加速库均获得了升级，另外新版本的驱动也可以支持更高版本的推理引擎，由此也能更好的获得ONNXRuntime等推理引擎迭代带来的收益。

然而，在

英伟达生态下，某个组件的更新，往往要设计其他一系列组件相应的更新。对于线上服务，这种更新相当于“在高速路上一边开车一边换轮子”，这个过程需非常谨慎细心。我们投入了大量精力，利用“滚动停服升级”的方式，实现了各组件的平滑升级。

从上图表格中我们可以看到从Driver 440升级到470后，在各类业务的推理场景均取得了不错的性能提升，P99时延有最高达41%的降低。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/17.webp)

基于上面提到的各种优化方案，我们的推理框架设计也逐步形成。我们

将模型推理服务分为离线模型预处理和在线模型服务两个阶段

（如上图流程所示）：

1. 离线模型预处理阶段：我们将训练好的模型进行剪枝、常量折叠、算子融合、混合精度、编译优化等模型加速优化，在完成准确性和性能验证后，模型可进入线上推理服务流程。
2. 在线推理服务阶段：我们开发了Hot Query缓存模块“Cache Manager”；“DynamicBatch Manager”模块负责对Query请求进行打包以提升服务的吞吐能力；“DagScheduler and Model Manager”模块用来调度模型选用ONNXRuntime等引擎在CPU、GPU的执行。

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/18.webp)

在以上推理框架层优化之后，我们在小布的推理服务场景提升了两个核心指标：

(1) 新推理框架**P99分位的时延**降低了7ms，下降了35% (2) 新推理框架也有更低的**时延波动**，P99的波动降低为2ms。

**四、未来工作规划**

![文章插图](/articles/2022-11-23-gtc-2022-gpu-oppo-nlp-b7909fcf/images/19.webp)

首先，我们的预训练规模

已经向10亿、100亿级别演进，这些模型也将逐步落地到

各个场景中，进一步提升算法效果

，因此后续推理的模型参数也将进一步提升，单卡推理会遇到显存墙的问题，我们需要

探索模型并行、 流水并行等分布式推理方案。

在AI芯片层面

，我们也将使用Ampere以及Hopper架构的GPU卡来提升推理性能。降低成本。经过实际性能评测，

A30/A100单卡推理时延小于T4卡时延的1/3，可以帮助我们承载更大规模模型

。A30 在MIG切分下，BERT类测试1/4 A30 MIG卡的时延仍低于T4整卡，

MIG技术可进一步降低推理成本。同时我们也将

尝试Graphcore、昆仑芯、寒武纪芯片等异构AI芯片在推理场景带来的成本收益

。在推理架构层面，我们将引入“

Triton + Python backend + 用户自定义代码

” 的方式更大限度的拓展推理服务的灵活性。

在模型性能层面，我们将持续积累计算图优化patterns 、 高性能算子、 编译优化等技术方向，继续模型加速的技术演进。

---

附件：GTC 2022议题及分享视频链接：

https://www.nvidia.com/gtc/session-catalog/?tab.scheduledorondemand=1583520458947001NJiE&search=42152#/session/1639570246493001REaf

推荐阅读

- [阿里云GPU计算平台首次超百万级性能纪录！异构计算平台打破MLPerf推理纪录，每秒处理107万张图片！](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885995&idx=1&sn=48bf2e33e19369e84e2f024ecb210638&chksm=b290404985e7c95fb8fd7dcd0c2b358735b726e5074912156e6a6c7e2c84e1d39271b0c7fda9&scene=21#wechat_redirect)
- [深度学习框架MegEngine CUDA INT4推理方案学习：Tensor Core、INT4算子特点、优化思路](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886428&idx=1&sn=1daf0e1e9b4f1fe6bec13fd0594464f0&chksm=b2904ebe85e7c7a8aedb485b789d7db266b5a92045505bb711f0eea8b5fbc1c2624971182ffe&scene=21#wechat_redirect)
- [GPGPU BLAS 计算库 CLBlast: A Tuned OpenCL BLAS Library](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886014&idx=1&sn=87ac9e9eee71b5f625df41912990115a&chksm=b290405c85e7c94ac662d54c9981ba656cbf31430e0b332de9dea84b1eae26a182bf1765581b&scene=21#wechat_redirect)
- [CLTune: A Generic Auto-Tuner for OpenCL Kernels](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886072&idx=1&sn=818b1e6e9b44ecc9cd4a7309ea8be1cd&chksm=b290401a85e7c90c8175b1bb1bf6355d26d2e533b6f20c9953b5b87ed7c1c68136b3406e0165&scene=21#wechat_redirect)
- [从GPU谈异构，这是ARM处理器架构师的趣谈](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885856&idx=1&sn=fe738c91a44d69ddc2e7ea3e1c2ef9f6&chksm=b29040c285e7c9d43b7afce201c8f5d548964faf808b3941bb899e044629c3e58621a75465a9&scene=21#wechat_redirect)
- [GPU上的高效softmax近似](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885412&idx=1&sn=0a6a1c0ff5128cba5d39d4deaf3540b1&chksm=b290428685e7cb908ad28d761ae7416b430a504466eab5256d53562c52402b5e16ba9e5f644a&scene=21#wechat_redirect)
- [Arm Mali GPU 四大微架构概述](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884956&idx=1&sn=b0fd51f51a5dfd47075d4bb0f545fda2&chksm=b290447e85e7cd68c4dd8db5461c138996eb25a0bdbe238c59b0abe5b26f50f4dde02642f163&scene=21#wechat_redirect)
- [深入浅出 | 谈谈MNN GPU性能优化策略](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885606&idx=1&sn=4dbf5d6c52018485eea4076d4b61c1de&chksm=b29043c485e7cad256f73405a2fafecead5f36a99da7dc43ad658c4cf0fa7c5d44ab4f9da677&scene=21#wechat_redirect)
- [华为端引擎Bolt GPU性能优化：让上帝帮忙掷骰子](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885381&idx=1&sn=cab9666a15cc0012732b537c52d9ca89&chksm=b29042a785e7cbb1eacb9a3f0a5b842973c1f3e1ba8084405223ae02c2f8daed8165c376503d&scene=21#wechat_redirect)
- [PaddleBox：百度超大规模离散DNN模型训练解决方案](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886609&idx=1&sn=fd8ed2af117fdb4a671b69f954b17693&chksm=b2904ff385e7c6e589131098333548b29e769bc14f3d0a07ed16826de78b8e61a7e100dc29bc&scene=21#wechat_redirect)
- [阿里妈妈high_service：打造多媒体AI推理的极致性能！](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886674&idx=1&sn=27bda729baa1177fb9aeaf6efe8e34b6&chksm=b2904fb085e7c6a67ff0ed5810c93f9eaf36489aa4105cf45f196b8e6d0413f1782bfe491e00&scene=21#wechat_redirect)
- [小红书LarC：应用大规模深度学习的分钟级实时推荐系统](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886683&idx=1&sn=6562f5e7c79c04ab076cf19c7a67eabb&chksm=b2904fb985e7c6afb39a633e743ececc2e094541010728dbca0f90ce3dc201aa3c487e39bbc9&scene=21#wechat_redirect)
- [快手 AI 工程化落地：挑战、思考和实践](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886147&idx=1&sn=ecfa3de9054a30b34eae444096ddaa0b&chksm=b29041a185e7c8b784d738cd1303780560118a54da9fc8e4c758146ef3a87995f18255527944&scene=21#wechat_redirect)

- 点击【阅读原文】查看往期文章 -
