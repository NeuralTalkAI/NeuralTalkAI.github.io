文 / Ellie Zhou、Tian Lin、Cong Li、Shuangfeng Li 以及 Sushant Prakash

**简介和动机**

TFLite 开源了一个端到端解决方案来解决设备端的推荐任务。该解决方案可提供个性化、低延迟和高质量的设备端侧推荐，同时还能保护用户隐私。

- 端到端解决方案  
https://tensorflow.google.cn/lite/models/recommendation/overview

高质量的个性化推荐对于许多应用至关重要，如音乐、视频、购物、应用、新闻等。目前，典型的推荐系统完全在服务器端构建，可收集用户活动日志、使用收集到的日志训练推荐模型以及提供推荐模型。

尽管已有数据表明完全基于服务器的推荐系统功能十分强大，**但我们希望能另辟蹊径，探索并展示一种更轻量级的方法：在设备端部署推荐模型**。这种设备端推荐解决方案可以实现低延迟推理，速度要比服务器端模型快出几个数量级。并且，它可以实现全新的用户体验，如更新排名和响应每个用户点击或交互的 UI，这一点令基于服务器的传统推荐系统望尘莫及。

设备端模型推理尊重用户隐私，无需将用户数据发送到服务器进行推理，所用的全部数据都保留在设备上。我们的解决方案支持使用公共数据或通过现有的代理数据集来训练模型，从而避免为每个新用例收集用户数据。对于设备端训练，推荐感兴趣的读者使用[联邦学习](http://mp.weixin.qq.com/s?__biz=MzAwODY4OTk2Mg==&mid=2652049407&idx=1&sn=1d39d15bc8b5bfb341ec77748225dbf6&chksm=808cafbab7fb26ac181a5abadc20db162e81bed9ba50e0f876729b062c3f43f5968545b09f32&scene=21#wechat_redirect)或 [TFLite 模型个性化](https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487363&idx=1&sn=7ccd84341d537ce072a506e29d0bbc61&scene=21#wechat_redirect)作为替代方案。

- 联邦学习  
https://federated.withgoogle.com/

解决方案包含以下部分：

- 源代码，为设备端场景构建和训练高质量个性化推荐模型  
https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml
- 一款电影推荐演示应用，可在设备上运行模型  
https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android
- 在 Github 代码库中的准备训练示例和预训练模型的源代码https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation

**模型**

推荐问题通常被表述为未来活动预测问题。因此，我们会对推荐模型进行训练，使其能够基于用户之前的活动来预测用户的未来活动。模型采用以下架构构建：

![文章插图](/articles/2021-05-20-tensorflow-lite-f9b7d3e2/images/01.webp)

在上下文方面，每个用户活动（如观看电影）都被嵌入到嵌入向量中。编码器将聚合来自过去用户活动的嵌入向量，以生成上下文嵌入。该方案支持三种不同类型的编码器：

- 词袋法 (Bag-of-Words)：计算活动嵌入向量的简单平均值。
- CNN：对活动嵌入向量应用一维卷积，然后进行最大池化。
- RNN：对活动嵌入向量应用 LSTM。

在标签方面，标签项（例如，用户看过的电影或感兴趣的下一部电影）被视为“正样本”，而所有其他项（例如，用户未看过的所有其他电影）通过负采样被视为“负样本”。正项和负项都被嵌入，点积结合上下文嵌入来产生对数几率并提供给 softmax 交叉熵损失。今后还会支持标签不是二进制的其他建模情况。完成训练后，可以导出模型并将其部署在设备上以提供服务。我们采用 top-K 推荐，即上下文嵌入和所有标签嵌入之间的 K 个置信度最高的预测 (K-highest logits) 。

![文章插图](/articles/2021-05-20-tensorflow-lite-f9b7d3e2/images/02.webp)

**示例**

为了演示设备端推荐模型的质量和用户体验，我们使用 MovieLens 数据集训练了一个示例电影推荐模型，并开发了一款演示应用（模型和应用仅用于演示目的）。MovieLens 的 1M 数据集包含 6039 位用户对 3951 部电影的评分，每个用户仅对一小部分电影进行评分。为简化流程，我们忽略了评分分数，并训练了一个模型以在给定 N 部以前电影的条件下预测用户将对哪些电影进行评分，其中 N 称为**历史长度**。

- MovieLens  
https://grouplens.org/datasets/movielens/

下面显示了模型在历史长度不同的所有三种编码器上的性能：

![文章插图](/articles/2021-05-20-tensorflow-lite-f9b7d3e2/images/03.webp)

我们可以发现，所有模型均达到了高召回率指标，而 CNN 和 RNN 模型通常在更长的历史长度下表现更好。在实践中，开发者可以使用不同的历史长度和编码器类型进行实验，并针对他们要解决的特定推荐问题找到最佳方法。

我们要强调的是，所有已发布的设备端模型都具有非常低的推理延迟。例如，对于与演示应用集成的 N=10 的 CNN 模型，在我们的实验中，Pixel 4 手机的推理延迟仅为 0.05ms。如简介中所述，如此低的延迟使得我们能够对手机上的每一次用户交互做出即时、流畅的响应，这一点在我们的应用中得到了证明：

![文章插图](/articles/2021-05-20-tensorflow-lite-f9b7d3e2/images/04.gif)

- 0.05ms  
https://tensorflow.google.cn/lite/models/recommendation/overview#performance_benchmarks

**未来工作**

欢迎大家提供各种各样的扩展，参与模型的完善工作。目前的开源模型不支持一个以上的特征列来表示每个用户的活动。在下一个版本中，我们将支持多个特征作为活动表示。而且，我们正计划设计更高级的用户编码器，例如基于 Transformer 的编码器 *(Vaswani, A., et al., 2017)*。

**参考文献**

Vaswani, A., et al."Attention is all you need. arXiv 2017." arXiv preprint arXiv:1706.03762 (2017), https://arxiv.org/abs/1706.03762.

点击 【阅读原文】

，查看往期文章。
