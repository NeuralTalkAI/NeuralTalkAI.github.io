> 原文标题：Hello, EasyQuant! 知乎作者：圈圈虫 原文链接：https://zhuanlan.zhihu.com/p/151292945 本文已获原作者原创转载授权

## ![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/01.webp)

## 背景

深度网络模型的嵌入式应用和加速是人工智能落地的重要方向，而 Int8 量化已经被证明可以满足应用上对精度和速度的需求。但是形形色色的 Int8 量化算法要么是难以实现，逻辑复杂，要么是精度保持差。那么是否存在一种低成本，**无需训练，任务无关，易用简单**，且精度保持比较好的量化方法了？鱼和熊掌不可兼得？

当然有，**@格灵深瞳DeepGlint** 最近开源量化算法 EasyQuant，包括论文**《EasyQuant: Pose-training Quantization via Scale Optimization》** 及代码：

> 项目代码：https://github.com/deepglint/EasyQuant 论文地址：https://github.com/deepglint/EasyQuant/blob/master/docs/EasyQuant.pdf

## 简介

EasyQuant 通过**精细搜索每层量化参数 Scale 值来实现量化精度的提升**。我们的实验表明，**即使是在 Int7 或更低比特的量化条件下，也能达到接近 Int8 和 Float32 的性能**。基于当前 arm NEON 指令集，更低的 Int7 量化，也会带来更快的推理速度，且**更容易减少溢出** 问题的产生。

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/02.webp)

EasyQuant **需要的样本量极低**，在用 KLD 方法得到初始量化值后，一般只需要 50 张左右的典型样本就可以完成整个量化调优过程。

由于这是一个非训练量化（Post Training Quantization）的方法，所以它也是一个**任务无关** 的方法，不需要考虑各种各样任务相关的 Loss 函数和数据 Label，相对于其它需要搭建完整的训练流程才能达到量化目标的方法来说，EeayQuant 体现出了应用上简单友好的特点。

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/03.webp)

## 应用

EasyQuant 已经被工业界其它公司所应用，并得到了对易用性和量化效果的肯定。

@开放智能OPENAILAB 公司的 Tengine（https://github.com/OAID/Tengine）团队在市场调研中发现，目前市面上**大多数 NPU 均采用 TFLite UInt8 量化策略（这是众多 NPU 厂家对 TensorFlow 强大生态妥协的结果）**，**精度下降比较明显，虽然 TensorFlow 提供了 QAT（Quantization Aware Training）的 Finetune 精度补偿方案，但在实际部署环境中，较难具备 Finetune 的条件。**

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/04.webp)

因此他们基于 EasyQuant 中提到的策略，针对市面上常见的某款主流 NPU 提供的离线量化（Post Training Quantization）工具进行改进，取得较好的精度补偿结果，推动了该款 NPU 在实际项目中快速稳定落地。部分实验结果如下：

## 分类任务

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/05.webp)

ImageNet2012, Top1

## 检测任务

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/06.webp)

Wider Face, Easy/Medium/Hard

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/07.webp)

## 展望

希望 EasyQuant 独特的量化方式可以被更多的工业界朋友所使用，也希望能给学术界在量化方法的探索上提供一个有价值的参考。

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/08.webp)

![文章插图](/articles/2020-06-29-easyquant-tensorflow-ed4b2800/images/09.webp)

> 点击**【阅读原文】** 跳转知乎，一起关注 @圈圈虫
