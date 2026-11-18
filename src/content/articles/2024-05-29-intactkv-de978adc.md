> 本文介绍我们针对大语言模型量化的工作 IntactKV，可以**作为插件****有效提升****GPTQ、AWQ、QuaRot等****现有主流量化方法效果**。论文作者来自清华大学、华为诺亚、中科院自动化所和香港中文大学。论文代码已经开源，欢迎大家使用！

- 论文链接：**arxiv.org/abs/2403.01241**
- 开源代码：**https://github.com/ruikangliu/IntactKV**

大语言模型（LLM）在自然语言处理的各项任务上都取得了出色的表现，但其庞大的参数量也给实际部署带来了新的挑战。**模型量化是一种常用的模型压缩方法，可以通过将模型权重和激活值压缩到低比特来有效降低模型推理时的算力和内存开销，然而量化往往伴随着模型的精度损失**。对此，我们提出了一种新的量化方法IntactKV。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/01.webp)

我们发现**LLM中存在关键词元（pivot tokens）**，这些**关键词元的表征上存在极大的离群值**，这会导致模型**将大量自注意力得分分配到这些词元上**，而模型**量化会使** 得这些**关键词元表征受损**，严重**损害**了**量化模型精度**。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/02.webp)

**图1 IntactKV示意图**

**IntactKV通过预先缓存关键词元的无损KV cache来保证量化模型中关键词元表征无损**，理论推导证明这能有效降低模型的量化误差上界。此外，缓存的无损KV cache还可以作为模型的额外参数做进一步校准，进一步弥补量化误差。IntactKV实现简单并且与GPTQ[1]、AWQ[2]、QuaRot[3]等当前主流的LLM量化方法正交，**实验表明，IntactKV可以作为插件有效提升现有量化模型精度，同时不会带来额外的推理开销，在权重量化和权重激活值量化上均取得了当前的SOTA精度**。

## 关键词元（Pivot Tokens）

现有研究[3][4][5]发现，LLM**激活值的特定通道上存在离群值**，这些**离群值往往出现在** **所有词元的激活值上，使得LLM的激活值难以量化**。而我们发现LLM中**还存在另一种特殊的离群值**，如图2所示，它**只存在于** LLM**输入的首词元[BOS]以及开头的一些特定词元上**（如“,”、“.”等标点符号），并且这类离群值比特定通道上的离群值还要大得多。更为重要的是，**这类特定词元上的离群值**会导致LLM将大**量的自注意力得分分配到这些词元上，形成Attention Sink** [6]，因此我们**将这些特定词元称为关键词元（pivot tokens）**。**由于汇聚了大量自注意力得分，关键词元的表征对LLM来说至关重要**。类似的现象也在同期工作Massive Activations [7]中被发现并得到了更为细致的研究。然而，**还没有其他研究针对这一现象对于量化模型的影响做深入探索**。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/03.webp)

**图2 关键词元示意图**

## 方法概述

我们发现，**模型量化会不可避免地导致关键词元表征受损，进而严重影响模型精度**，因此**如果能保证关键词元的表征，也就是其KV cache无损，就能有效降低量化误差**。

为此，我们提出使用IntactKV—**先使用** **全精度模型生成关键词元的无损KV cache****并将其缓存下来** （i.e., IntactKV），**量化模型在推理时就能直接使用无损的关键词元表征，从而有效提升量化模型精度**。如图3所示，IntactKV包含了从首词元[BOS]开始的若干词元，随着IntactKV包含词元数的增加，量化误差也不断降低，尤其是当IntactKV包含关键词元时，量化误差会得到明显下降，这也进一步验证了**关键词元的重要性**。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/04.webp)

**图3 保证关键词元表征无损能有效降低量化误差**

然而，**模型在每次推理时的输入序列各不相同，这意味着我们并不能使用任意长度的IntactKV**，否则在每次量化模型推理前都必须调用全精度模型来生成对应输入的IntactKV，这会带来额外的推理开销。对此，我们**利用了LLM推理时的公共前缀**，如果IntactKV**只包含公共前缀**，那么**全精度模型在生成IntactKV后就可以被释放**，**不会** 给之后量化模型的推理**带来任何额外开销**。对于预训练LLM，输入序列前一般都会加上[BOS]词元作为公共前缀，因此可以使用只包含[BOS]词元的长度为1的IntactKV，称为IntactKV[B]。如图3所示，**由于首词元总为关键词元，因此使用IntactKV[B]就可以明显地降低量化误差**。而**对于SFT模型，输入序列前一般会包含一段系统提示词作为公共前缀，因此可以使用长度为系统提示词的IntactKV**，称为IntactKV[P]。由于**使用了更长的IntactKV**，IntactKV[P]**能覆盖更多关键词元**，**进一步降低量化误差**。我们在实验中发现，**对于Vicuna模型，长度为34的系统提示词就足以覆盖全部关键词元**。

此外，**缓存的IntactKV** 还可以**作为模型的额外参数**做校准来进一步**弥补量化误差**。具体来说，我们**在大小为128的校准集上做了轻量化训练**，可学习参数仅为缓存下来的IntactKV，损失函数**采用量化模型和全精度模型输出的MSE loss**：

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/05.webp)

其中，![文章插图](/articles/2024-05-29-intactkv-de978adc/images/06.webp)为IntactKV，![文章插图](/articles/2024-05-29-intactkv-de978adc/images/07.webp)为第![文章插图](/articles/2024-05-29-intactkv-de978adc/images/08.webp)个Transformer层的映射函数，![文章插图](/articles/2024-05-29-intactkv-de978adc/images/09.webp)为LLM中Transformer层的层数，![文章插图](/articles/2024-05-29-intactkv-de978adc/images/10.webp)为输入激活值，![文章插图](/articles/2024-05-29-intactkv-de978adc/images/11.webp)分别为全精度权重和量化权重。

## 实验验证

### 1. 权重量化

我们首先在PPL任务上进行了实验。如表1所示，**IntactKV能很好地与GPTQ、OmniQuant、AWQ等主流量化方法结合**，**无需任何训练或额外推理开销**即可在LLaMA系列的各种不同模型上有效**提升量化模型精度**，其中AWQ+IntactKV在各个模型上都达到了最优效果。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/12.webp)

**表1 权重量化PPL任务实验结果**

我们基于Vicuna模型，在MMLU和QA任务上做了进一步实验。如表2所示，实验结论与PPL任务基本一致。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/13.webp)

**表2 权重量化MMLU和QA任务实验结果**

我们对量化模型的生成质量也进行了评估，通过GPT-4对模型生成回答进行打分来衡量量化模型的生成能力。如表3所示，IntactKV_[P]已经**能明显提升量化模型的回复得分**，对IntactKV的轻量化校准则进一步提升了模型的回复质量。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/14.webp)

**表3 权重量化MT-bench任务实验结果**

### 2. KV Cache量化

为了将IntactKV用于KV cache量化，我们采用**混合精度量化策略**，将**IntactKV保存为全精度**，**其余KV cache量化到低比特**。对于KV cache量化，由于模型仍然需要先将量化的KV cache反量化为全精度才能进行自注意力计算，因此上述**混合精度方案在推理时只需要将IntactKV和反量化后的KV cache重新拼接到一起即可**，并**不会带来额外的推理开销**。如图4所示，IntactKV在**不同的量化比特设置下均可以有效提升AWQ量化模型精度**，尤其是**在低比特时提升更加明显**。目前，IntactKV已经被集成到了SOTA KV cache量化方法KVQuant[8]中。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/15.webp)

**图4 权重和KV Cache量化MMLU任务实验结果**

### 3. 权重和激活值量化

对于**权重激活值量化**，**KV cache需要全部量化到低比特**才能**有效利用低比特矩阵乘进行运算加速**。我们发现，关键词元的KV cache表征明显比其余KV cache表征更加平坦，就算直接将IntactKV量化到低比特也只会带来极少的精度损失，因此对于激活值量化，我们选择**将IntactKV进一步压缩到低比特以免引入额外的推理开销**。如表4所示，IntactKV能**有效提升当前SOTA量化方法精度**，进一步**缩小了**W4A4**量化模型与原始模型间的精度损失**。

![文章插图](/articles/2024-05-29-intactkv-de978adc/images/16.webp)

**表5 权重和激活值量化PPL任务实验结果**

## 参考文献

[1] Frantar, Elias, et al. "OPTQ: Accurate quantization for generative pre-trained transformers." The Eleventh International Conference on Learning Representations. 2022. [2] Lin, Ji, et al. "Awq: Activation-aware weight quantization for llm compression and acceleration." arXiv preprint arXiv:2306.00978 (2023). [3] Ashkboos, Saleh, et al. "Quarot: Outlier-free 4-bit inference in rotated llms." arXiv preprint arXiv:2404.00456 (2024). [4] Dettmers, Tim, et al. "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale." Advances in Neural Information Processing Systems 35 (2022): 30318-30332. [5] Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." International Conference on Machine Learning. PMLR, 2023. [6] Xiao, Guangxuan, et al. "Efficient Streaming Language Models with Attention Sinks." The Twelfth International Conference on Learning Representations. 2023. [7] Sun, Mingjie, et al. "Massive Activations in Large Language Models." arXiv preprint arXiv:2402.17762 (2024). [8] Hooper, Coleman, et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." arXiv preprint arXiv:2401.18079 (2024).

## 相关文章

- [低比特离线量化EasyQuant: 比TensorFlow更友好的离线量化方式](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885024&idx=1&sn=fc2f5ea0db0a61eb9e7f72aea46d0688&chksm=b290440285e7cd14ae098380e7cecfb35b36aea42739e3fed49c56c856e02c2a57c7ff7a1b14&scene=21#wechat_redirect)
- [IEEE 754 Floating NaN](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887217&idx=1&sn=ac3c93ac58b1f2eddd074ee3d5874951&chksm=b2904d9385e7c485882347eebf7602540a9aa3849ebb07e706fb166df66e4cabd7fcba46f830&scene=21#wechat_redirect)
- [爱奇艺在DCN、EDVR等 4K 超分模型上的 TensorRT 10倍加速实践](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886743&idx=1&sn=987497f322f403029d0e6e8b08a61b49&chksm=b2904f7585e7c6639491bc5696596f34accda01fa5f9ee015101752be0dac34c4f2676980711&scene=21#wechat_redirect)
- [AI、ML加速器发展趋势综述](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886740&idx=1&sn=4e66cc0346e914168191425145cc8c4a&chksm=b2904f7685e7c660f2bfce6bb3738aeffd4a030dfb273b068b0ec30137cecf3f1e79ece0970d&scene=21#wechat_redirect)
- [一篇关于深度学习编译器架构的综述论文](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884994&idx=1&sn=2415d682852bd3d29f9f522e4ac2e8d9&chksm=b290442085e7cd367068d267e2633e376e17a5bb2f47c09488a762cdec8e8bf800e327a98937&scene=21#wechat_redirect)

- 查看往期，【阅读原文】 -
