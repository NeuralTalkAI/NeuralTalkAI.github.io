# 以大模型百万分之一参数量的预测小模型，在LongBench数据集上使用4%的KV Cache达到Full Cache 99%的模型准确率。

> 近日，中科大王杰教授团队（MIRA Lab）和华为诺亚方舟实验室（Huawei Noah's Ark Lab）联合提出了基于时序预测的可学习稀疏注意力计算方法AttentionPredictor，使用仅大模型百万分之一大小的小模型辅助大模型推理，在十六倍的压缩比下取得近乎无损的推理准确率，推理速度提升1.4倍，这为长序列推理任务的高显存占用带来了新的解决方案。代码已开源！

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/01.webp)

论文：https://arxiv.org/abs/2502.04077

代码：https://github.com/MIRALab-USTC/LLM-AttentionPredictor

## 太长不看版

随着大语言模型（LLM）的广泛应用，长上下文生成场景下通过键值缓存（KV Cache）压缩实现高效推理成为研究热点。现有方法基于启发式注意力评分筛选关键token，但忽略了注意力分数的时间模式特性，导致大模型性能显著下降。

本文提出AttentionPredictor，一种可学习的注意力预测方法，通过轻量级卷积模型捕捉注意力时空特征，使用LLM百万分之一的参数量即可精准预测下一token的注意力分数。结合跨token的KV缓存预取框架，AttentionPredictor在16倍压缩率下仍保持大模型性能近乎无损，推理速度提升1.4倍，显著超越现有最优方法。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/02.webp)

与近期大热的DeepSeek的NSA、Kimi的MoBA、微软的SeerAttention比较，我们工作的相同点都是分块压缩和检索近似注意力分数，并且都使用可学习的小模型加速注意力计算。我们工作与近期工作的不同点有：

- 我们分块并压缩的是计算出的注意力分数，而这些工作分块的是Key。
- 我们估算Attention方式是从历史注意力分数中预测下一步的注意力分数，而近期工作是使用压缩后的Key与Query相乘得到估算的注意力分数。
- 我们的可学习部分是预测下一步attention的模型，而近期工作是学习一个模型来表征压缩后的key。
- 我们的方法是post-training的，不涉及LLM训练阶段，而近期工作都在训练阶段就加入了稀疏注意力。

## 引言

> 随着大语言模型（LLM）在长上下文推理任务中的广泛应用，键值缓存（KV Cache）的内存与计算开销成为制约其部署效率的核心瓶颈。传统方法通过启发式规则（如历史注意力累加、局部窗口筛选）或近似检索技术压缩KV Cache，但这些方法普遍面临两大挑战：
>
> 其一，静态启发式评分难以捕捉注意力动态演化的时间模式，导致关键token识别不准确，模型性能显著下降；
>
> 其二，现有检索方法依赖当前步骤的Query信息，无法通过异步预取机制隐藏计算与传输延迟，限制了实际加速效果。

针对上述问题，本文提出AttentionPredictor——首个学习Attention动态时空模式的KV Cache压缩框架。通过系统分析注意力分数的演化规律，我们发现注意力分布呈现强时间特征, 如重复访问（Re-access）、顺序访问（Sequential）、周期性（Seasonal）特征。

基于此，本文将注意力序列建模为时空信号，利用轻量级卷积网络预测下一token的注意力分数，突破传统静态评分的局限性。同时，设计跨token的KV Cache预取框架，通过异步加载与并行化调度，将评估与传输时间隐藏于模型推理过程中，显著提升解码效率。

与现有方法相比，本文的创新性主要体现在三个方面：

1. 动态时间模式建模：首次将注意力分数建模为时空序列，通过卷积网络学习重复访问（Re-access）、顺序访问（Sequential）、周期性（Seasonal）等模式，实现高精度预测下一步注意力。
2. 跨令牌预取框架：提出异步加载下一token关键缓存的机制，将token评估与传输时间隐藏于推理过程中，显著降低解码延迟。
3. 高效压缩与校准：引入分块注意力压缩和分布误差校准技术，在减少计算量的同时保持预测准确性，16倍压缩率下模型性能损失小于1%。。

在长序列任务上，AttentionPredictor在16倍压缩率下平均性能损失<1%。

在长CoT任务上，在16k上下文长度下，AttentionPredictor仅损失2.05%准确率，显著优于Quest的16.91%下降。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/03.webp)

图1. H2O、Quest 和 AttentionPredictor使用历史注意力分数识别下一步的关键token的方法比较。我们基于学习的时空预测器可以捕捉动态注意力模式，并准确预测下一步的注意力得分。

### 1. 背景与问题介绍

当前高效LLM推理与KV缓存压缩方法主要分为四类：

- 缓存驱逐方法：基于启发式规则筛选历史关键token，如StreamingLLM（保留初始与近期token）、H2O（历史注意力累加）、SnapKV（窗口内注意力筛选）、MInference（垂直-斜线、block等模式）。此类方法依赖静态评分，难以捕捉动态时间模式，导致长上下文场景下性能显著下降。
- 缓存检索方法：通过近似Query-Key交互检索关键token，如Quest（分页键近似计算）、PQCache（键值量化）。这类方法的计算开销较大，且准确率随分页大学的增加而大幅下降（如Quest在page size 从16增加到64时，精度下降11%）。且这依赖当前步骤Query，无法通过预取隐藏延迟。
- 可学习的稀疏注意力：Kimi的MoBA 与Quest类似地计算分块attention，再取top-K作为稀疏mask。进一步地，MoBA将稀疏Attention加入了模型训练，使模型在稀疏注意力上的性能得以提高。微软的SeerAttention将按块pooling后的Keys再经过一个可学习的Linear层，以对压缩后的Keys编码，使计算出的近似Attention接近原始分布。DeepSeek的NSA使用可学习的块编码模型代替常见的Pooling来建模每个分块，并结合了多种缓存压缩策略。这些工作都需要在模型训练阶段就加入稀疏注意力的使用，以达到更好的模型效果。
- 跨层预取方法：结合缓存检索与跨层预取（如InfiniGen），但单层推理时间不足以覆盖长序列传输延迟和估算时延，扩展性受限。

现有方法均面临动态模式建模不足与计算-传输延迟耦合两大瓶颈，制约了高压缩率下的模型性能与推理速度。

## 2. 动机实验——注意力具有时序模式

为揭示注意力演化的内在规律，本文通过大量实验分析发现解码过程中注意力分布呈现三类可预测的模式（见图2）：

- 重复访问（Re-access）：特定token在多步骤中被反复关注（垂直带状分布）；
- 顺序访问（Sequential）：注意力沿token序列逐步推移（对角线分布）；
- 周期性（Seasonal）：关键token周期性出现（交替带状分布）。

我们发现Query具有很强的连续性，相邻解码步骤的查询向量余弦相似度高达87%。推导表明注意力分数的差异主要由微小增量Δq主导，使得相邻步骤关键token高度重叠，支持跨token预取。详细推导见论文第2章节。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/04.webp)

图2. 三种时序注意力模式的可视化。Re-access显示对特定标记的重复关注。Sequential注意力向下一个标记推移。Seasonal显示出周期性的模式，如较为集中的高注意力分数和较为均匀分布的注意力交替出现。

## 3. 方法介绍

本文方法包含注意力预测小模型（AttentionPredictor）与跨令牌KV缓存预取框架两部分（如图3），前者通过动态时空建模精准筛选关键token，后者通过异步加载机制隐藏计算与传输延迟，共同实现高效长上下文推理。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/05.webp)

图3. 我们提出的KV Cache压缩方法AttentionPredictor和跨token预取框架。（a）AttentionPredictor将历史注意力分数建模为时空序列，并借助预训练模型预测下一步的注意力。为了提升效率，在每个decoding步骤中，历史注意力分数会以压缩形式进行更新。（b）跨token预取框架。在LLM推理过程中，异步评估关键token，并为下一个token获取KV，从而有效加速解码阶段。

### 3.1 AttentionPredictor

问题建模: 在LLM解码阶段，KV缓存压缩的目标是选择预算B个的关键令牌位置pi最大化注意力恢复率：

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/06.webp)

即保留的注意力分数占比。传统方法依赖静态启发式评分，而AttentionPredictor通过预测下一token的注意力分数动态筛选关键位置。

时空序列建模：本文将历史注意力分数![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/07.webp)建模为时空信号，利用轻量级卷积网络（2层2D卷积+1层1D卷积）捕捉多尺度时空特征。模型输入为分块压缩后的注意力![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/08.webp) （块大小为b），输出为下一token的预测注意力![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/09.webp)，再通过Top-K筛选确定关键token位置。

模型训练：仅需3%的注意力数据（如LongBench中每个任务选取5个样本）即可完成训练，并支持跨任务泛化（如从LongBench迁移至GSM8K）。模型参数量极小，仅为LLM的百万分之一，对存储资源的占用可忽略。

误差抑制技术：本文使用了分块压缩来减少计算量，即对注意力矩阵执行Max pooling，将计算量降低至1/b，同时保留局部关键信息。本文还使用了误差校准技术，即每隔M步计算完整注意力分数，修正因稀疏计算累积的分布偏差，确保长期预测稳定性。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/10.webp)

算法1. AttentionPredictor识别关键KV Cache的算法流程。 更多细节可见原文第4章节。

### 跨token预取框架

在解码阶段，本文通过异步并行化机制隐藏关键token评估与传输延迟。具体来说，在GPU执行LLM推理时，利用AttentionPredictor预测下一步的关键token索引，并异步从CPU加载p对应的KV缓存至GPU。通过掩盖数据传输和估算时延（如图4），与现有跨层预取方法（如InfiniGen）相比，本方法通过跨token粒度，在32k上下文下实现1.4倍解码加速。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/11.webp)

图4. 我们提出的跨token预取的流程图。通过异步加载下一个token的关键KV缓存，我们的框架隐藏了token评估和传输延迟，从而加速了LLM推理的解码。

## 4. 实验介绍

我们实验的数据集包括长序列任务LongBench和数学推理任务GSM8K，平均输入token数达13K。实验包含4个部分：

1. 评估AttentionPrdictor在不同缓存预算下的注意力重建率。
2. 在两大类任务上评估使用我们方法后大模型的性能。
3. 通过消融实验展示我们方法各部分的效用。
4. 评估我们方法的推理效率。

我们在此文章中详细介绍实验2，其余实验请参见原论文的第5章节。

### LongBench数据集

部分实验结果见表1。实验结果显示，我们的方法在不同budget下模型精度损失均<1%。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/12.webp)

表1. 我们提出的AttentionPredictor相较于其他方法，模型精度损失大幅降低。其中H2O方法使用64步注意力分数，与我们方法参数对齐，记为H2O+。

### 数学推理数据集

部分实验结果见表2。我们通过调整few-shot的个数模拟长输入CoT任务。实验结果表明，我们的方法显著优于现有方法，在16k长度下，AttentionPredictor仅损失2.05%准确率，显著优于Quest的 16.91%下降。

![文章插图](/articles/2025-02-26-llm-attentionpredictor-16-kv-attention-a04d2d33/images/13.webp)

表2. 我们方法在长CoT任务上的模型精度表现优于其他方法。

## **相关文章**

- [端侧大模型比赛暨研讨会 | NeurIPS'24](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887513&idx=1&sn=6471e3e91d02a4d4aa167b27f1e2e8ac&scene=21#wechat_redirect)
- [LLM 量化新篇章，4-bit 权重激活量化几乎无损！FlatQuant 的平坦之道](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887505&idx=1&sn=ea42dec96bf6e76d5f37f57e4ffab0bf&scene=21#wechat_redirect)
- [让量化大模型无损，IntactKV：保持关键词元无损，轻松提升量化精度！](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887239&idx=1&sn=b976d340b7ba0b1b0b120874b1763377&chksm=b2904d6585e7c473a8a3980798c6600456127d7985327efed2ec8aafa1ca4716a1512e6c05c6&scene=21#wechat_redirect)
- [走向ARM CPU 1-bit推理的极致道路](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885109&idx=1&sn=0f32a578c71b66d6ceffb7952859c575&chksm=b29045d785e7ccc185ad120904c2e4e14987b473e2e2aa6817c7ebd66c75b3c5a359ad58d8c8&scene=21#wechat_redirect)
- [MLCEngine: 全平台大语言模型推理引擎](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887268&idx=1&sn=0f7b0f591ed129eace8d036662b44b11&chksm=b2904d4685e7c450ee0b3763c447258f87d56c894073608d0c12cc40c081f7ff144ace5e6812&scene=21#wechat_redirect)
- [SIMD 指令集与数据并行程序](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887352&idx=1&sn=c579010b3cb554095be51ebc6d95748c&chksm=b2904d1a85e7c40c0c7d239b03a73abae86c573382803e35c538b22bb487c2c307dd39b2d7b7&scene=21#wechat_redirect)
- [低比特离线量化EasyQuant: 比TensorFlow更友好的离线量化方式](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885024&idx=1&sn=fc2f5ea0db0a61eb9e7f72aea46d0688&chksm=b290440285e7cd14ae098380e7cecfb35b36aea42739e3fed49c56c856e02c2a57c7ff7a1b14&scene=21#wechat_redirect)
- [IEEE 754 Floating NaN](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887217&idx=1&sn=ac3c93ac58b1f2eddd074ee3d5874951&chksm=b2904d9385e7c485882347eebf7602540a9aa3849ebb07e706fb166df66e4cabd7fcba46f830&scene=21#wechat_redirect)
- [爱奇艺在DCN、EDVR等 4K 超分模型上的 TensorRT 10倍加速实践](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886743&idx=1&sn=987497f322f403029d0e6e8b08a61b49&chksm=b2904f7585e7c6639491bc5696596f34accda01fa5f9ee015101752be0dac34c4f2676980711&scene=21#wechat_redirect)
- [AI、ML加速器发展趋势综述](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886740&idx=1&sn=4e66cc0346e914168191425145cc8c4a&chksm=b2904f7685e7c660f2bfce6bb3738aeffd4a030dfb273b068b0ec30137cecf3f1e79ece0970d&scene=21#wechat_redirect)
- [一篇关于深度学习编译器架构的综述论文](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884994&idx=1&sn=2415d682852bd3d29f9f522e4ac2e8d9&scene=21#wechat_redirect)

**- 查看往期，【阅读原文】 -**
