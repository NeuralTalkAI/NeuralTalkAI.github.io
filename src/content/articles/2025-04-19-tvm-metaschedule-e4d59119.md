> 边缘深度学习正在消费类设备（如家用电器、汽车导航系统和物联网设备）中迅速部署。鉴于这些设备有限的计算能力和内存，优化作为张量程序表达的深度学习模型至关重要。

本论文来自三菱电机株式会社，创立于1921年1月15日的日本制造业企业，**评估了 Apache TVM 的 MetaSchedule 在优化边缘设备上图像分类模型的推理性能**。MetaSchedule 是 Apache TVM 开源机器学习编译器的一部分，可**自动为各种硬件平台优化张量程序**。

::: block-1
对 52 个图像分类模型的评估表明，**MetaSchedule 在 45 个模型中优于 ONNX Runtime，速度提升高达 4.77 倍（平均为 1.68 倍）**。然而，对于像 EfficientNet-v2、VGG 和 ResNet50 这样**具有较大批量大小的模型，MetaSchedule 则稍显逊色**。
:::

未来的工作将集中于扩展 MetaSchedule 的模型支持，并增强优化性能，从而有助于实现在边缘设备上的低延迟、保护隐私和节省带宽的推理过程。

![文章插图](/articles/2025-04-19-tvm-metaschedule-e4d59119/images/01.webp)

## 本文目录

- 1. 引言
- 2. MetaSchedule 简介
- 3. 实验设置
- 4. 结果与讨论
- 5. 结论
- 参考文献

![文章插图](/articles/2025-04-19-tvm-metaschedule-e4d59119/images/02.webp)

## 1. 引言

> 将基于深度学习的图像分类部署在边缘设备（如家用电器和物联网设备）上，可以带来诸如低延迟、提高隐私保护和减少网络带宽使用等好处。由于边缘设备计算资源有限，优化模型推理性能至关重要。

尽管存在诸如专门的硬件加速器（例如 Google 的 EdgeTPU）和轻量级模型（例如 EfficientNet）等进步，但高效深度学习推理的需求依然存在。

本研究**评估了 Apache TVM 的 MetaSchedule** （一个张量程序优化系统）与 **ONNX Runtime 相比在优化近期深度学习模型方面的有效性**。我们的贡献包括**对 MetaSchedule 优化特性的综合性能分析和深入见解**。

## 2. MetaSchedule 简介

> **MetaSchedule 是 Apache TVM 的一部分**，可**自动为特定硬件平台优化张量程序**。

图 1 展示了 MetaSchedule 的处理流程。它从一个预训练模型开始，该模型被转换为 TVM 的 Relay 格式。TVM 将其划分为小的子图，并将它们降低为张量表达式（TE），表示张量计算。TE 包括用于优化的调度原语，如矢量化、并行化、展开、分块和循环融合等。

![文章插图](/articles/2025-04-19-tvm-metaschedule-e4d59119/images/03.webp)

MetaSchedule 通过在设备上进行测量来寻找每个 TE 的最佳调度策略。TVM 随后将 TE 图转换为张量中间表示（TIR），目标编译器将其转换为模型二进制文件，该文件准备好进行部署，预期将优于 ONNX Runtime。

## 3. 实验设置

> 我们选择了由 PyTorch Torchvision 提供的 52 个图像分类模型，排除了不受支持的模型。**目标硬件是 Jetson Orin Nano 8GB 上的 NVIDIA Ampere GPU**。我们将功耗模型设置为 15W。

在系统软件部分，我们使用 Ubuntu 20.04、**Apache TVM 0.14（Commit: 318fa863）**、LLVM 14.0.0、**ONNX Runtime 1.16.0、Jetpack 5.1.2 和 CUDA 11.4.315**。

我们使用**默认的 MetaSchedule 调优参数**，除了 max_trials_global，将其设置为模型中子图数量的 400 倍，因为模型大小可能会有所不同。这里 max_trials_global 可以理解为一种全局试验次数限制，用于控制整个模型调优过程中的最大尝试次数，而子图数量指的是模型被分解成的小块图的数量。

## 4. 结果与讨论

我们的结果总结如图 2 所示，**MetaSchedule 在多达 4.77 倍（平均为 1.68 倍）的情况下优于 ONNX Runtime**。MetaSchedule 在**小型模型（如 mobilenet_v3_large）上表现出色**，适用于所有批量大小，但在**批量大小超过 2 的 EfficientNet-V2、VGG 和 ResNet50 模型中则稍显逊色**。

![文章插图](/articles/2025-04-19-tvm-metaschedule-e4d59119/images/04.webp)

图 3 和图 4 分别提供了 mobilenet_v3_large 和 resnet50 的不同批量大小性能的详细信息。这些发现突出了 MetaSchedule 的优势和局限性，为未来的优化工作提供了指导。

![文章插图](/articles/2025-04-19-tvm-metaschedule-e4d59119/images/05.webp)

## 5. 结论

本研究对 Apache TVM 的 MetaSchedule 在优化边缘设备上的深度学习推理进行了全面分析。**MetaSchedule 在多个模型中显示出显著的性能提升，超过 ONNX Runtime 高达 4.77 倍（平均为 1.68 倍）**。这些结果强调了深度学习编译器在实现边缘设备上高效深度学习推理方面的重要性。

然而，对于像 EfficientNet-V2、VGG 和 ResNet50 这样**具有较大批量大小的模型**，MetaSchedule 则**稍显逊色**。未来的研究将集中于扩展 MetaSchedule 的模型支持并进一步增强其优化能力，从而有助于实现边缘设备上高效、低延迟和保护隐私的深度学习推理。

## 参考文献

[1] S. Cass, “Taking AI to the edge: Google's TPU now comes in a maker-friendly package,” IEEE Spectrum, vol. 56, no. 5, pp. 16–17, 2019, doi: 10.1109/MSPEC.2019.8701189.

[2] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” CoRR, 2019.

[3] T. Chen et al., “TVM: An Automated End-to-End Optimizing Compiler for Deep Learning,” in 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), Carlsbad, CA: USENIX Association, Oct. 2018, pp. 578–594.

[4] J. Shao et al., “Tensor Program Optimization with Probabilistic Programs,” in Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 – December 9, 2022, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., 2022.

[5] S. Marcel and Y. Rodriguez, “Torchvision the machine-vision package of torch,” in Proceedings of the 18th ACM International Conference on Multimedia, in MM ’10. Firenze, Italy: Association for Computing Machinery, 2010, pp. 1485–1488.

[6] NVIDIA, “NVIDIA Jetson Orin Nano Developer Kit.” [Online]. Available: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
