> 算子融合也称符算融合，作为面向DL模型推理的一种关键图优化技术，**通过减少计算过程中的访存次数达到提升模型推理性能的目的**，该技术**在不同时期、不同框架下的落地方式各不相同**。
>
> 本文将首先带大家**回顾算子融合技术的发展历程**，并**浅析其中颇有代表性的Apollo方案**。

## 1.算子融合技术的发展历程

> **大约16年前后**，业界对于推理应用的性能诉求还不普遍，对于有性能需求的场景，最常见的做法是**利用设备供应商提供的API** 加速计算图中的部分计算密集型（Compute-bound，以下称CB）算子，如Conv、Dense等。
>
> 这一时期，用户自定义的**算子融合多半是相邻访存密集型（Memory-bound，以下称MB）算子的融合**；而**典型的CB+MB形式的融合（例如Conv+ReLU）则受限于供应商API的支持程度**。

随着AI模型在各领域的发展、成熟和落地，模型推理在具体设备上的性能变得越发重要，**17年到18年**，业界出现了**大量面向DL模型推理的计算框架**，算子融合技术在不同框架上呈现出两种典型的发展路线：

1. **遍历路线：针对特定设备，枚举实现典型的CB+MB形式的融合算子**，如Conv+ReLU/Conv+BN+ReLU/Dense+ReLU/Conv+Sum等等，Intel的oneDNN以及国内很多大厂的推理框架走的都是这个路线；
2. **规则路线：基于规则实现算子融合**，以TVM为例，其将所有算子分为Opaque/Injective/Reduction/Complex-out四类，并设定融合规则形如Complex-out+Injective/Injective+Injective/Injective+Reduction等，以此实现所有符合规则的子图融合。

两种路线相比各有优劣：

1. 对于**业务范围、应用设备十分明确** 的场景，**遍历路线更适合在短期内迅速提升模型的执行性能**。假设某公司的业务是在ARM设备上做Resnet模型的推理，那么提升模型推理性能的最直接手段必然是实现不同Workload的Conv+BN+ReLU子图在ARM设备上的最优流水排布、数据并行方式、循环拆分合并等。同时，这一路线的**缺点也十分明显：没有泛化性，如果部署设备、模型等要素发生变化，之前的工作对当前工作没有任何帮助**；
2. 规则路线显然具备**更佳的应用泛化性，但融合后子图的执行性能却不一定足够好**：以Conv为例，假设在TVM中某个Conv算子在X86设备上按照调度方案A被拆分为9层循环，那么后继的ReLU算子应该在哪层循环计算才是最优？最优位置是否和Conv算子的Workload相关？如果有新类型的CB算子，以上问题的答案是什么？想要解决以上问题，在当前的TVM框架下可以通过对特定子图注册不同的compute-schedule模板，配合AutoTVM搜索给出近似最优解，但**schedule的改动和调整仍然需要人工参与**。

以上提到的**算子融合解决的是典型的“内存墙”问题**，即**减少缓存和内存间数据搬运量以提升计算性能**。除此以外，还有一些算子融合技术解决的是**“并行墙”问题**，即如何**保证计算图执行非CB部分的子图时也能充分利用设备的计算资源**，这方面典型的工作是TASO（TASO: Time and Space Optimization for Memory-Constrained DNN Inference，https://arxiv.org/abs/2005.10709）和RAMMER（Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks，https://www.usenix.org/system/files/osdi20-ma.pdf），通过将计算过程中互不依赖的低计算量子图打包为一个kernel并行计算以提升资源利用率。

## 2. Mindspore的算子融合方案Apollo

Apollo是Mindspore框架的图算融合团队提出的**多层规约算子融合方案**，其分为两个主要阶段：

![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/01.webp)

**图1 算子融合方案Apollo**

1. 首先是图划分阶段，提取DNN模型的所有subgraph，并将subgraphs根据预定的规则**重新分解**并聚合成更合理（或是自家IR的）micrograph，这一步的目的是让所有可能后续做融合的子图聚合为micrograph，为后续所有融合工作划定范围；
2. 再就是基于以上拆分得到的**micrographs做了bottom-up的重新融合**，分为三步：

### 2.1 图划分的具体步骤

图划分是Apollo方案非常重要的一步，比如对于CB算子和injective算子的融合，图划分起的作用是**将可以融合的计算尽可能分到一个group里**，后续的Polyhedral融合就是在这个group里决定融合的具体循环位置，该步骤可以分为以下三步：

1. 将所有合适做融合的算子提取出来，这里的“合适”指**除了** **自定义算子、计算量超复杂的算子以及控制流算子之外的**所有算子；
2. **分解compound算子**，比如对于以下例子中的LogSoftMax算子，该算子是由reduce和substract算子组合的，这一步要把这种算子拆分，为后面的merge提供更多可能性 图 分解算子  
![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/02.webp)**图2 分解Compute Bound算子如LogSoftMax**
3. **primitive算子聚合**，这一步类似于TVM的规则定义和应用，Apollo把算子定义为element-wise/broadcast/reduce/opaque四种算子，根据以下rules做的算子聚合分组：![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/03.webp)  
**图3 基于规则的算子聚合**

### 2.2 图融合的具体步骤

### Apollo采用了**bottom-up多层规约融合的方法实现的融合**，依次解决了buffer/loop级别融合、“内存墙”突破融合以及“并行墙”突破融合：

### 2.2.1 loop级别融合：polyhedral

loop级别的融合采用Polyhedral启发式算法，实现循环合并工作，例如下面例子中实现的add+pooling的融合：

![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/04.webp)

**图4 基于Polyheral通过循环合并实现add+pooling的融合**

通过polyhedral技术进行自动的算子优化和生成。借助于Polyhedral编译技术，能够对算子的迭代空间进行依赖分析，自动生成兼顾并行性与数据局部性的调度。同时，迭代空间的依赖关系还能帮助我们完成循环重排、循环融合、循环切分等变换优化，进而达到较优的执行性能。

相比业界其它自动算子生成能力，**采用Polyhedral技术主要好处**：

1. **不需要手工指定schedule****，以及schedule空间的tuning。** 从而支持任意的融合算子编译生成。使得图算融合应用于JIT训练场景成为可能；
2. 能够**有效提升算子融合深度及融合算子泛化能力。** 在**针对单算子或一个独立的循环嵌套进行优化时，Polyhedral模型并不能轻易地得到与手工优化算子库或其它通过手工指定调度的方法相当的性能**，但是**在** **多个算子融合或多个循环嵌套之间快速寻找优化手段方面，Polyhedral却有着相对较大的优势**。在ModelZoo上的泛化验证结果也证明，无论是在融合算子的规模（200+融合算子），还是融合pattern的复杂程度（多输出、多算子类型）上，AKG（[华为MindSpore开源AKG(Auto Kernel Generator)项目](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884982&idx=1&sn=dee0729be97c3ab8773bbfe76f7310fd&chksm=b290445485e7cd4297f01a1b34260130b00ef341e6313a57642050047c70ed580acb4bf0359e&scene=21#wechat_redirect)）都能够快速高效的生成对应的kernel。

### 2.2.2 内存墙突破：Stitch

“内存墙”突破参考的是阿里的AStitch工作，“内存墙”问题体现比较明显的一个例子是**BERT这种transformer模型的图融合场景**：BERT模型中的CB算子是矩阵乘，相邻的矩阵乘算子间会有数个MB算子，对这些MB算子做融合会遇到两难问题，即**不做会有大量kernel的启动开销影响性能；做了会因为broadcast算子和后续计算的自动合并导致硬件计算资源被空耗（重复计算）**。因此阿里提出了AStitch工作，**对MB算子图做了op依赖和element依赖分析，可以避免一些MB算子融合带来的重复计算问题**，Apollo复用了这个工作。

### 2.2.3 “并行墙”突破：合并无依赖

“并行墙”突破主要是**将无依赖计算合并到一起做并行计算**，这里的关键在于如何在避免组合爆炸的前提下对可以并行的子图分组，Apollo的做法是**基于经验总结了以下几种常见的模式**：

![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/05.webp)

**图5 合并可并行子图**

### 2.3 技术效果

MindSpore 1.3版本发布Bert GPU网络，Apollo方案的应用实现了相对XLA 3.1~3.6倍的性能加速。

![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/06.webp)

**图6 Apollo方案相比XLA的性能提升**

其中对GPU的支持做了极大的完善和增强。通过对Model Zoo大量GPU网络进行验证，图算融合平均性能提升：NLP类96.4%↑；推荐类136.6%↑; CV类30.7%↑。

![文章插图](/articles/2022-11-06-bottom-up-buffer-loop-5e0b2e91/images/07.webp)

**图7 NLP推荐类与CV类MindSpore性能自身对比**

## 3. 观点总结

1. 受限于通用计算设备的算力增长瓶颈，**算子融合技术仍然是提升模型推理** 在这类设备上推理**性能的重要手段**；
2. 解决“并行墙”问题的算子融合技术相对独立，适合大部分框架应用和借鉴；
3. 面向“内存墙”的算子融合技术，Apollo的方案的一个重要基础是基于Polyhedral的自动循环优化，这对于很多小伙伴来说是个门槛；
4. **Polyhedral****不一定****是算子自动循环优化的最优解法**，基于此的CB+MB融合也不是唯一方案，例如小节1提到的**“半自动”搜索方法也能在同样的任务下获得相似的性能，实现难度也更低**。

## 参考

1. FusionStitching: Boosting Execution Efficiency of Memory Intensive Computations for DL Workloads https://arxiv.org/abs/1911.11576
2. ASO: Time and Space Optimization for Memory-Constrained DNN Inference https://arxiv.org/abs/2005.10709
3. Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTask https://www.usenix.org/system/files/osdi20-ma.pdf

## 相关文章

- [AI框架图层IR的优势、特殊需求和挑战](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885305&idx=1&sn=bbca62d8de3794156ad35254a7c471e6&chksm=b290451b85e7cc0d4744486d176d5a071141c0e7643b2932df57f07b59f3e070df8914a0e1ea&scene=21#wechat_redirect)
- [Paddle Lite底层backend的kernel选择策略](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884913&idx=1&sn=66acdf5b2db0aabb58f0d34ce638ce8a&chksm=b290449385e7cd85ebf4056dcea410b370b7085c6c2582731af7b6dac6f54b70947c7f2b3014&scene=21#wechat_redirect)
- [AI 硬件公司接入框架与 AI 框架算子层级的思考](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886093&idx=1&sn=bc56c3a0843a39dbd66b0b513f5803b9&chksm=b29041ef85e7c8f9c484a5a2e5646d8a8822b37b98fce0d6d57fab0e1ea12b5d9c7b09541336&scene=21#wechat_redirect)
- [AI编译优化---访存密集算子优化](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885230&idx=1&sn=8c81fcf79d10decaf6fea7968d3203bf&chksm=b290454c85e7cc5a96aae15580d105c3603ecc8226feb33af9473a5e0a7d506a6c267095437d&scene=21#wechat_redirect)
- [AI编译优化--计算密集算子优化](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885202&idx=1&sn=7ee2c929b55b21f5981d6ae3186d15f2&chksm=b290457085e7cc66372ca7b84ce0bf0279da704ad3f416e2bd8f281d22070133ae13302c1418&scene=21#wechat_redirect)
- [AI编译优化--业务实践](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885245&idx=1&sn=f19f72507a44bed1b5d964ce499978f8&chksm=b290455f85e7cc493af5a5a7c6a5e97172c33eef647dbdef8a2192ae7ecd3eb89c1dd740e639&scene=21#wechat_redirect)
- [AI编译优化--总纲](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885158&idx=1&sn=c5b2f5399d133f1b4389cbf034fabf06&chksm=b290458485e7cc9201483b2411c2dffd481c4fee4510dcf2b54de9bd30a6a641cb595c9be114&scene=21#wechat_redirect)
- [新一代深度学习编译技术变革和展望](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886255&idx=1&sn=1b55f4fe1335260a6086082577ec8951&chksm=b290414d85e7c85bbe0b14e41e3b36b6266bb51f1a4dee67d9d4eef0f797e2eb231e1de406f4&scene=21#wechat_redirect)

> 点击【阅读原文】，查看往期文章
