> It provides the ability to automatically fuse ops with specific patterns. AKG works with MindSpore-GraphKernel to improve the performance of networks running on different hardware backends.

## MindSpore / AKG

## AKG全称 Auto Kernel Genterator，用于 MindSpore 的算子融合等特定模式的优化。AKG 适配 MindSpore 的 GraphKernel（这个概念文档没有细说，mindspore文档里也找不到，不过可结合后面的MindSpore架构图猜测一下），针对不同硬件后端，来做性能优化。

![文章插图](/articles/2020-06-21-mindspore-akg-auto-kernel-generator-6e28012a/images/01.webp)

AKG由4个基本优化模块构成：Normalization、auto schedule、Instruction emit和backend optimization。

- Normalizaton：three address transform，公共common subexpression elimination, copy propagation and so on；
- auto schedule. The auto schedule module mainly have vectorization, loop tiling, mem promotion and loop distribution.
- instruction emit. The instruction emitting module has the optimization about loop normalization, auto pragma and emit instruction.
- backend optimization. The backend optimization module consists of double buffer optimization, storage rewrite optimization and inject sync optimization.

目前，支持的硬件是Ascend910，后续会有更多硬件支持。据说，0.3版本的MindSpore的BERT整体（训练）性能在达芬奇芯片上是GPU v100的两倍。ResNet性能据说有很大提升，我猜测应该是也结合了THOR大杀器（二阶优化）。

> 二阶优化：ResNet达到精度（0.759）仅用了42个迭代（epoch），比同软硬件环境的一阶优化快了近1倍（一阶优化使用了81epoch达到目标精度）。
>
> 具体见文档：https://gitee.com/mindspore/mindspore/tree/r0.3/example/resnet50imagenet2012THOR

## MindSpore总体架构

这部分文档来自官网Docs(0.2.0-alpha)，MindSpore框架架构总体分为MindSpore前端表示层、MindSpore计算图引擎和MindSpore后端运行时三层。

![文章插图](/articles/2020-06-21-mindspore-akg-auto-kernel-generator-6e28012a/images/02.webp)

图：MindSpore整体架构（来自0.2.0-alpha）

- MindSpore前端表示层（Mind Expression，简称ME）该部分包含Python API、MindSpore IR（Intermediate representation，简称IR）、计算图**高级别优化（Graph High Level Optimization，简称GHLO）** 三部分。
- MindSpore计算图引擎（Graph Engine，简称GE） 该部分包含计算图**低级别优化（Graph Low Level Optimization**，简称GLLO）、图执行。
- MindSpore后端运行时 该部分包含云、边、端上不同环境中的高效运行环境。

---

这么看来AKG应该是在GE的GLLO这一层，目前看代码底层有C++的实现codegen/poly/schedule等等模块，此外Python打通底层调用。

## Roadmap

另外，MindSpore项目推进非常快，这里摘一下有关性能优化、架构、推理方面的roadmap：

### 性能优化

1. 优化编译时间
2. 低比特混合精度训练/推理
3. 提升内存使用效率
4. 提供更多的融合优化手段
5. 加速PyNative执行性能

### 架构演进

- 图算融合优化：使用细粒度Graph IR表达算子，构成带算子边界的中间表达，挖掘更多图层优化机会。
- 支持更多编程语言
- 优化数据增强的自动调度及分布式训练数据缓存机制
- 持续完善MindSpore IR
- Parameter Server模式分布式训练

### 推理框架

- 提供Tensorflow/Caffe/ONNX模型格式支持
- IOS系统支持
- 完善更多的CPU算子
- 更多CV/NLP模型支持
- 在线学习
- 支持部署在IOT设备
- 低比特量化
- CPU和NPU异构调度

最后放上链接：

- MindSpore/akg: Auto Kernel Generator. - 码云 - 开源中国 https://gitee.com/mindspore/akg/tree/master
- mindspore: MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. - 码云 Gitee.com https://gitee.com/mindspore/mindspore/tree/r0.3/example/resnet50imagenet2012THOR
- Roadmap MindSpore https://www.mindspore.cn/docs/zh-CN/0.3.0-alpha/roadmap.html

放上链接：

- MindSpore/akg: Auto Kernel Generator. - 码云 - 开源中国 https://gitee.com/mindspore/akg/tree/master
- mindspore: MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. - 码云 Gitee.com https://gitee.com/mindspore/mindspore/tree/r0.3/example/resnet50imagenet2012THOR
