
> 知乎作者：江雪 原文标题：深入硬件特性加速TinyBert，首次实现手机上Bert 6ms推理 原文链接：https://zhuanlan.zhihu.com/p/154526064 本文已获原作者转载授权

## Bert模型应用场景

自然语言处理问题可以分为四大类：序列标注，分类任务，句子关系判断和生成式任务。2018年10月底Google团队提出预训练语言模型Bert[1]，刷新了11项自然语言处理任务的成绩，Bert可以用于问答系统，情感分析，垃圾邮件过滤，命名实体识别，文档聚类等任务。【参与投票，可查看结果】

Google Bert模型参数量巨大，base版本包含12个隐藏层，模型参数总数110M左右；large版本24个隐藏层，模型参数高达340M。许多云侧服务器利用GPU加速Bert的研究陆续提出，但是如果搭建提供给用户访问的服务，用户数据的传输，高并发的服务请求，大量的计算需求等等，对云侧服务器是巨大的考验。

如何利用智能手机的计算资源，在手机上解决Bert计算加速问题显得意义重大。

## 探索Bert在端侧手机落地的可能

Bert在模型存储和推理加速方面，对端侧落地是巨大挑战，华为诺亚方舟实验室通过**两阶段知识蒸馏** 的方法，**将老师的大型Bert 模型中编码的大量知识可以很好地迁移到小型的学生TinyBert[2] 模型中**，**学生模型参数量只有原来的1/8**。使用**通用数据集** 进行**第一阶段的蒸馏**，使用**特定任务的数据集**进行**第二阶段的蒸馏**，既保证了计算精度，又降低了模型参数量。

![文章插图](/articles/2020-07-05-bert-6ms-tinybert-60b9c467/images/01.webp)

::: image-caption
诺亚TinyBert知识蒸馏
:::

实证研究结果表明，TinyBert 是有效的，在 GLUE 基准上实现了与 BERT 相当（下降 3 个百分点内）的效果。

## Bert端侧推理优化策略

TinyBert中的主要运算是矩阵和向量运算，比较容易实现向量化，我们主要使用了如下方法：

1. 将**访存算子** （Batch Normalization, Scale, ReLU, GeLU, TanH等激活函数）尝试**与计算密集的算子** （卷积，矩阵乘法，矩阵向量乘法）进行**合并；**
2. **cache内计算**，减少函数调用和内存访问次数；
3. 使用**合并访存优化Transpose和Reshape等格式转换算子**；
4. **基于快速索引方式计算Embedding** 算子实现。

通过一系列图优化手段，Bert优化问题转化为矩阵向量乘法和矩阵矩阵乘法优化问题。下面，着重讲一下其中的几点。

### 权重预处理、指令优化加速

对核心耗时算子矩阵向量乘法，**首先采用权重预处理，将列方向求和问题转化为行方向求和**，列方向求和缓存利用率只有1/8，行方向充分利用CPU cache line和预读取策略，提高缓存命中率。利用ARM CPU的SIMD高吞吐计算单元和FMA计算指令，1次可以计算8个乘加运算，如下图所示，采用4x8的窗口从左向右滑动，做成对的乘加运算，滑动到行尾结束时利用成对相加指令求和，避免规约指令的高延迟开销。

![文章插图](/articles/2020-07-05-bert-6ms-tinybert-60b9c467/images/02.webp)

::: image-caption
矩阵向量乘法
:::

下一次再起一行滑动计算。对于超长的K采用block循环处理，充分**利用缓存减少对向量的内存访问次数**。

### 矩阵分块、缓存优化加速

采用**分块** （8x24,8x16,8x8,4x24等）**拼图算法加速矩阵矩阵乘法**，拼图有利于解决边界的性能烂尾问题。对规约维度K进行分块处理，每一次处理block大小，首先读取矩阵A的8xblockK的块到缓存，转置读取矩阵B的blockKx24的块缓存。利用SIMD更新计算矩阵C的8x24的块，利用8x(24/8个吞吐)=24个寄存器保存C的中间结果，避免重复读取矩阵C。利用3个寄存器循环读取B，1个寄存器读取A。

![文章插图](/articles/2020-07-05-bert-6ms-tinybert-60b9c467/images/03.webp)

::: image-caption
矩阵矩阵乘法
:::

ARM CPU一般有1~2个FMA计算单元，内存总带宽也是有限的，但是读内存通路和计算通路是相互独立的，可以同时处理，内存访问的延迟是制约程序性能的瓶颈，Bolt采用的缓存优化技术缓解了降低了访存延迟。Bolt同时采用**双缓冲区机制，实现寄存器流水线**，将寄存器分成2组，当第一组寄存器做计算时，可以读取内存数据到第二组寄存器，当第一组寄存器计算完毕空闲出FMA计算单元时，第二组数据也就绪了，第二组开始计算，换成第一组读取数据，ARM CPU的32个向量寄存器被充分利用32=24+(3+1)*2。

### fp16数据类型加速

采用**半精度数据类型float16**加速自然语言处理网络计算，取得了很好的加速效果。**将模型存储和计算内存占用再降低一半**，float16半精度存储的TinyBert模型只有22MB。在分类任务上Bolt混合精度与float32推理准确率基本一致，**性能相对float32提升将近一倍**，在华为nova10, 麒麟810的2.2GHz A76大核心和1.7GHz A55小核心上，我们做了一些单核性能测试，如下图所示。

![文章插图](/articles/2020-07-05-bert-6ms-tinybert-60b9c467/images/04.webp)

::: image-caption
加速Bert
:::

Google Bert base的输入序列长度768，Albert[3]序列长度128，TinyBert序列长度32，采用可变长度机制灵活处理实际落地输入序列各种各样，序列长度不一的问题，减少数据对齐的计算和存储开销，在A76大核心上，TinyBert推理时间平均只用6ms，达到了超实时落地的效果。

## tiny-Bert结合Bolt欢迎使用！

相关成果我们已经开源到Github：https://github.com/huawei-noah/bolt，方便大家复现和使用，欢迎大家试用反馈体验效果，欢迎大家在社区积极讨论。高性能，精确，轻量级，易用和安全是我们不懈追求的目标，未来我们会继续利用高性能计算优化技术和编译技术，加速深度学习，在计算机视觉和自然语言处理领域发力，将更多的研究结果带到社区。

![文章插图](/articles/2020-07-05-bert-6ms-tinybert-60b9c467/images/05.webp)

::: image-caption
HUAWEI Noah's Ark Lab
:::

欢迎高性能计算和深度学习算法方向的童鞋加入我们，获取简历投递邮箱，请在后台回复「诺亚」、「华为诺亚」。

---

[1] Devlin, Jacob, Chang, Ming-Wei, Lee, Kenton,等. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[J]. [2] Jiao, Xiaoqi, Yin, Yichun, Shang, Lifeng,等. TinyBERT: Distilling BERT for Natural Language Understanding[J]. [3] Lan, Zhenzhong, Chen, Mingda, Goodman, Sebastian,等. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations[J].

> 点击**【阅读原文】** 跳转知乎，一起关注 @江雪
