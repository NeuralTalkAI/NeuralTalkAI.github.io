> 本文对基于华为昇腾神经处理单元（Ascend NPU）的 MindSpore 和 PyTorch 进行性能评估。华为的昇腾 NPU 是**一款专为神经网络计算加速而设计的芯片**，它不仅**为自研框架 MindSpore 提供支持**，还**对 PyTorch 进行了适配**。

然而，目前**缺乏在昇腾 NPU 上对 MindSpore 和其他框架的比较评估研究**，这使得人们难以了解它们的实际性能。

::: block-1
因此，**本文在模型层面和算子层面进行了全面的评估实验，分析并比较了这两个框架的性能**。通过模型评估实验和算子评估实验得出有关**框架性能差异**、**首个Epoch** 初始化现象、**AMP性能提升**、**动静态图性能** 差异和**多 NPU 并行加速（Scaling）** 相关的结论，为深度学习研究人员和从业者**在框架选择** 和**昇腾 NPU 优化策略方面** 提供了参考和建议。
:::

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/01.webp)

## 本文目录

- 1. 介绍
- 2. 相关研究
  - A. 深度学习框架
  - B. 华为昇腾 NPU
  - C. 深度学习框架性能评估的相关研究
- 3. 实验步骤
  - A. 评估实验的整体框架
  - B. 模型评估实验设置
  - C. 算子评估实验设置
  - D. 评估实验的公平性设置
- 4. 结果分析
  - A. 硬件和软件详情
  - B. 模型评估实验结果分析
  - C. 算子评估实验结果分析
- 5. 结论
- 参考文献

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/02.webp)

## 1. 介绍

> 深度学习作为机器学习的一个分支，主要目的是从数据中自动学习有效的特征表示。随着深度神经网络（DNNs）在图像分类、语音识别和自然语言处理等任务中取得巨大成功，基于深度神经网络的深度学习不断发展。

深度神经网络的训练通常需要大规模的数据集，这对设备的计算能力提出了要求。因此，各种类型的**深度学习处理器应运而生**，包括英伟达的图形处理单元（GPU）和谷歌的张量处理单元（TPU）。此外，华为发布的**昇腾 NPU** 提供了强大的计算性能，并且兼容多种计算场景。它基于计算架构面向神经网络（CANN）的软件栈，实现了对多种深度学习框架的支持，为深度学习的发展提供了新的可能性。

为了帮助研究人员和从业者提高编码效率并有效利用复杂的计算设备，出现了大量**支持自动梯度计算和设备资源管理的深度学习框架**。这些框架包括 Caffe、CNTK、MXNet、Jittor、PaddlePaddle、TensorFlow、PyTorch、MindSpore 等。然而，不同的深度学习框架在深度神经网络的实现上具有**不同的功能特性和优化**，因此**在不同的计算设备上的性能可能会有很大差异**。因此，深度学习研究人员和从业者需要仔细选择最适合他们需求的框架。

关于**在不同计算设备上** 评估**不同深度学习框架的研究有很多**。然而，这些研究大多集中在 GPU 或 TPU 设备上的 TensorFlow、PyTorch 或 Caffe 等框架，而**针对华为昇腾 NPU** 和 MindSpore 框架的**评估研究相对较少**。特别是缺乏在昇腾 NPU 上对 MindSpore 和其他框架的比较评估研究。因此，本文旨在通过进一步的研究实验来解决这一问题。

::: block-1
本文的主要贡献包括：

1. 在昇腾 NPU 设备上训练 MindSpore 和 PyTorch 中**三个典型场景的 DNN 模型**，**分析和比较不同设置下的性能**。
2. **分析和比较在不同参数大小下 MindSpore 和 PyTorch 在昇腾 NPU 上的算子性能**，包括卷积、全连接和批量归一化。
3. 基于性能评估实验的结果，为深度学习研究人员和从业者在**昇腾 NPU 上的框架选择** 和**优化策略提供建议**。
:::

## 2. 相关研究

### A. 深度学习框架

**Caffe** 由加州大学伯克利分校提出，用 C++ 编写，为 Python 和 MATLAB 提供接口。它在卷积神经网络方面表现出色。

**CNTK** 是微软提出的开源深度学习工具包。它为 C++、**C#**、Python 和名为 BrainScript 的自定义语言提供接口。它在** 语音识别领域** 取得了显著成果。

**MXNet** 由亚马逊提出，为 Python、C++、Julia、Go、R 等编程语言提供接口。它**轻量级且可扩展**，适合快速深度学习开发。

**Jittor** 是清华大学提出的**即时编译（JIT）** 深度学习框架。它提供类似 Numpy 的元算子以及结合了静态计算图和动态计算图特点的统一计算图。

**PaddlePaddle** 由百度提出。它通过**基于程序的方式动态描述** 神经网络的计算过程。

**TensorFlow** 由谷歌提出，用**数据流图** 表示神经网络的数值计算，其中节点表示操作，边表示数据（即张量）。它为 Python、Java、Go、C++ 和 R 提供接口。

**PyTorch** 由 Facebook 提出，是**从 Torch 演变而来** 的深度学习框架。PyTorch **采用动态计算图设计**，允许在执行期间修改图，这为调试和优化带来了简单性和灵活性。PyTorch 保留了命令式和 Python 风格的编程风格，并将其扩展到深度学习工作流程的几乎所有方面，包括层定义、模型构建、数据加载、优化器执行、训练并行化等。

华为的 **MindSpore** 是一个**支持云、边、设备场景** 的深度学习框架。MindSpore 提供了**结合静态计算图和动态计算图** 的统一执行模式，并允许轻松地从动态图转换到静态图。类似于 PyTorch，它还保留了命令式和 Python 风格的编程风格，使开发人员能够有效地构建神经网络模型。

### B. 华为昇腾 NPU

> 由华为发布的**昇腾 NPU 是一种基于特定领域架构（DSA）的芯片**，主要设计用于加速神经网络计算。昇腾 NPU 的计算核心是基于达芬奇架构的 AI Core，主要包括以下单元：用于张量计算模式的 Cube 单元，用于向量计算模式的 Vector 单元，以及用于标量计算模式的 Scalar 单元。

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/03.webp)

为了充分发挥昇腾 NPU 的性能，华为提出了**神经网络计算架构（CANN）软件栈**，包括神经网络软件流程和相关工具链，以便于 DNN 模型的生成、加载和执行。在深度学习框架层面，CANN 软件栈**不仅支持 MindSpore 的使用，还为 TensorFlow 和 PyTorch 提供了适配**，使得 **TensorFlow 和 PyTorch** 开发者**能够在昇腾 NPU 上训练** DNN 模型。

### C. 深度学习框架性能评估的相关研究

> 由于各种深度学习框架之间的差异，许多研究工作致力于评估深度学习框架的性能。

**TBD** 为 CNTK、MXNet 和 TensorFlow 构建了**一个性能分析工具链**，重点是在 DNN 模型**训练过程中测量和分析 GPU 内存消耗**，覆盖主流深度学习应用。

**DAWNBench** 重点关注 DNN **模型达到一定准确率阈值所需的训练时间**，并提出了一个名为**时间到准确率（TTA）的新复合指标**。在 GPU 和 TPU 设备上对 TensorFlow、PyTorch、Caffe 和 MXNet 进行了基准测试实验。

**ParaDnn** **扩大了基准测试的范围，生成了数千个参数化的 DNN 模型**，并使用 TensorFlow 和 CUDA 在 GPU 和 TPU 设备上评估性能差异。

**MLPerf** 通过使用各种 DNN 模型**对不同机器学习系统的训练收敛时间和吞吐量进行基准测试**，来比较其性能。

**AI Matrix 覆盖了计算机视觉、推荐和自然语言处理的应用领域**，在 GPU 上对 TensorFlow 和 Caffe 进行了特定电子商务环境数据的基准测试。在参考文献 [25]-[31] 中，为了比较和评估包括 Caffe、TensorFlow、PyTorch、Neon、CNTK、MXNet、Torch 和 Theano 在内的深度学习框架在 GPU 上的性能差异，进行了多次测试实验。

对于 MindSpore 的性能评估研究有限。文献 [18] 在模型层面和算子层面对 MindSpore 在昇腾 NPU 上的性能进行了评估，同时也对 PyTorch 和 TensorFlow 在 GPU 上的性能进行了评估。然而，**缺乏在昇腾 NPU 上对 MindSpore 和其他深度学习框架的比较实验评估**，这阻碍了我们进一步了解基于 CANN 软件栈的 PyTorch 适配情况。本研究旨在解决这一差距。

## 3. 实验步骤

### A. 评估实验的整体框架

**本文评估实验的整体框架如图 2** 所示。它包括**模型层面和算子层面** 的工作负载。使用标准数据集和随机生成的数据集来评估基于昇腾 NPU 的 PyTorch 和 MindSpore。

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/04.webp)

### B. 模型评估实验设置

为了尽可能全面地进行分析，我们选择的工**作负载包括残差网络（ResNet）、门控循环单元（GRU）和图卷积网络（GCN）**，涵盖了计算机视觉、自然语言处理和图结构数据处理的应用场景。这些工作负载涉及当前深度学习中的一些主流架构。

- ResNet：是一种引入残差块的深度卷积神经网络，旨在缓解深度神经网络训练过程中的退化问题。我们选择了 ResNet50 的实现，并基于 CIFAR-10 数据集进行图像分类实验。
- GRU：是循环神经网络（RNN）的一种变体，引入门机制以解决捕捉长期依赖关系时的梯度消失或爆炸问题。我们选择了编码器-解码器结构，并基于 Multi30K 翻译数据集进行英法翻译实验。
- GCN：是为图结构数据设计的神经网络。它利用类似于卷积神经网络（CNN）的图卷积滤波器来捕捉图结构中节点和边的局部信息，并将其组合成全局特征表示。我们基于 Cora 数据集进行图节点分类评估实验。Cora 数据集包含 2708 篇学术论文及其引用关系，所有论文被分为 7 类，并由 1433 维特征向量表示。

在进行**模型评估实验** 时，我们**考虑了以下与自动混合精度（AMP）、计算图模式和并行加速相关的设置**：

- **AMP**：是一种在**模型训练** 期间使**用低精度和高精度进行计算的优化技术**。PyTorch 和 MindSpore 均实现了 AMP，可以有效提高训练速度。
- **计算图模式**：对训练性能有显著影响。PyTorch 采用动态图模式，而 MindSpore 支持动态图模式（MindSpore 中的 PYNATIVE_MODE）和静态图模式（MindSpore 中的 GRAPH_MODE）。
- **通信**：通过 CANN 软件栈和华为集体通信库（HCCL）实现昇腾 NPU 设备之间的通信，从而实现多设备之间的加速并行训练，提高性能。

模型评估**实验的评估指标包括收敛时间和吞吐量**。收敛时间表示 DNN 模型达到一定评估指标阈值所需的时间，而吞吐量表示训练过程中每秒处理的样本数。通过考虑这两个指标，我们可以全面评估在昇腾 NPU 上使用 PyTorch 和 MindSpore 构建的模型的性能。

### C. 算子评估实验设置

除了模型评估外，我们还对一些常用的算子进行了评估实验，包括卷积、全连接和批量归一化。具体来说，卷积的 API 实现是 PyTorch 和 MindSpore 中的 Conv2d；全连接的是 PyTorch 中的 Linear 和 MindSpore 中的 Dense；批量归一化的是 PyTorch 和 MindSpore 中的 BatchNorm2d。

在评估实验中，**我们结合了不同的超参数（如批量大小、输入通道和输出通道）来生成不同参数规模的算子**。随后，我们**在各种参数规模下对这些算子进行实验分析**。

算子评估实验主要测量两个深度学习框架下算子的吞吐量比率，从而获得不同场景下算子的性能信息，为优化算子提供依据。

### D. 评估实验的公平性设置

为了确保评估实验的公平性，我们在超参数、算子实现、计算设备、缓存和预热等方面进行了设置。在 DNN 模型训练过程中，我们严格统一了学习率、批量大小、权重衰减、优化器等超参数。

对于两个框架下算子的不同实现，我们参考了 PyTorch 和 MindSpore API 映射表，选择了匹配的 API，并对 API 之间的差异进行了调整。在评估实验中，我们确保计算设备完全空闲，防止计算设备的内存调度影响实验结果。为了进一步提高实验结果的准确性，我们在正式实验前缓存了数据并进行了训练预热。

## 4. 结果分析

### A. 硬件和软件详情

**模型和算子评估实验在配备 8 个华为昇腾 910 NPU 和 4 个华为鲲鹏 920 CPU 的计算服务器上进行**。

::: block-1
每个华为昇腾 910 NPU 拥有 32GB HBM 存储容量，带宽为 1200GB/s，可在 FP16 精度下实现 320TFLOPS 的计算能力。每个华为鲲鹏 920 CPU 拥有 48 个内核，典型频率为 2.6 GHz / 3.0 GHz。服务器的 RAM 存储容量为 768GB，SSD 存储容量为 1.5TB。
:::

在计算服务器上，操作系统的版本是 EulerOS 2.0 (SP8)，CANN 的版本是 5.1.RC2.1，Python 的版本是 3.7.10，PyTorch 的版本是 1.8.1+ascend.rc2，MindSpore 的版本是 1.8.1。

### B. 模型评估实验结果分析

我们**绘制了 DNN 模型训练过程中验证数据集的评估指标随时间变化的图表，如图 3 所示**。ResNet50 和 GCN 的评估指标是准确率，而 GRU 的评估指标是 BLEU。每个评估指标都设置了不同的阈值：ResNet50 的准确率阈值为 85%，GCN 的准确率阈值为 70%，GRU 的 BLEU 阈值为 10%。

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/05.webp)

此外，我们还绘制了在不同训练设置下每个 epoch 的吞吐量图表，如表 I 所示。**训练设置包括在动态计算图或静态计算图中训练，以及使用 AMP 或不使用 AMP 训练**。我们区分了第一个 epoch 的吞吐量和其他 epoch 的平均吞吐量。吞吐量单位方面，ResNet50 以 images/s 为单位，GCN 以 nodes/s 为单位，GRU 以 sentence pairs/s 为单位。

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/06.webp)

最后，为了**评估昇腾 NPU 并行加速的可扩展性**，我们使**用不同数量的昇腾 NPU 训练** 了 MindSpore 和 PyTorch 中实现的 ResNet50 模型，并绘制了训练过程中平均吞吐量的图表，如图 4 所示。

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/07.webp)

根据上述图表和表格的分析，我们得出以下结论：

- **在昇腾 NPU 上，PyTorch 在密集计算场景中的性能不如 MindSpore**，而在**稀疏计算场景中的性能则优于 MindSpore**。ResNet50 的计算核心包括卷积、批量归一化和激活函数，这些都属于计算密集型操作。GCN 和 GRU 的计算核心涉及稀疏图矩阵计算和具有时间依赖性的门控循环机制。在图 3 中，使用 PyTorch 实现的 ResNet50 的训练达到准确率阈值所需的时间比使用 MindSpore 实现的 ResNet50 长 5.03 倍。使用 PyTorch 实现的 GCN 和 GRU 的训练达到准确率和 BLEU 阈值所需的时间分别比使用 MindSpore 实现的短 1.49 倍和 2.11 倍。
- PyTorch 和 MindSpore 在昇腾 NPU 上都表现出明显的**第一个 epoch 初始化现象**。根据表 I 的结果，可以观察到在所有训练设置下，第一个 epoch 的吞吐量低于其他 epoch 的平均吞吐量。这可以**归因于昇腾 NPU 和 CANN 软件栈在第一个 epoch 中执行创建上下文和优化编译等操作**，以优化模型的计算图构建，从而提高后续 epoch 的训练效率。
- **合理设置 AMP 可以提高 PyTorch 和 MindSpore 的性能**。分析表 I 的结果可以发现，**当使用 AMP 训练时，DNN 训练的平均吞吐量得到提高，带来显著的性能提升**。进一步实验表明，将 PyTorch 的 AMP 级别设置为 'O2'，将 MindSpore 的 AMP 级别设置为 'O3'，不仅可以提高训练吞吐量，还可以确保训练损失的稳定收敛。
- **MindSpore 在动态图模式下的性能较差**。MindSpore 是一个同时支持动态计算图和静态计算图的深度学习框架，其**训练性能会受到所采用的计算图模式的显著影响**。如表 I 所示，当其他训练设置保持不变时，MindSpore 在动态图模式下的吞吐量明显低于其静态图模式下的吞吐量，也低于 PyTorch 的吞吐量。据分析，**MindSpore 编译器在静态图模式下利用图优化技术对执行图进行更大程度的优化**，从而提高训练性能。因此，建议在**调试网络时使用 MindSpore 的动态图模式**，而在**正式训练网络时使用静态图** 模式。
- 与 PyTorch 相比，**MindSpore 更能充分利用多个昇腾 NPU 带来的计算能力提升**。从图 4 可以看出，在 CANN 软件栈和 HCCL 的帮助下，增加多个昇腾 NPU 显著提高了 PyTorch 和 MindSpore 框架中 DNN 模型的训练。然而，**随着昇腾 NPU 的数量增加，MindSpore 在模型训练吞吐量方面的提升比 PyTorch 更显著**，从而更好地利用了多个昇腾 NPU 带来的计算能力增强。

### C. 算子评估实验结果分析

我们绘制了**在不同参数数量下，MindSpore 和 PyTorch 在昇腾 NPU 上** 的卷积、全连接和批量归一化算**子的吞吐量比率图表，如下图 5 所示**。图中的纵轴表示它们的吞吐量比率，水平线表示比率等于 1 的情况。

- 当值高于水平线时，表示 MindSpore 的吞吐量更高；
- 当值低于水平线时，表示 PyTorch 的吞吐量更高。

![文章插图](/articles/2025-04-20-npu-pytorch-mindspore-vs-b4f3b7b4/images/08.webp)

从图中可以看出，**PyTorch 在参数数量较少的情况下具有更高的算子吞吐量**，而 **MindSpore 在参数数量较多的情况** 下具有**更高的算子吞吐量**。

此外，不同算子受参数数量的影响也不同。例如:

- MindSpore 的批量归一化算子即使在参数数量较少的情况下也能超过 PyTorch；
- 而 MindSpore 的卷积算子只有在参数数量较多的情况下才能超过 PyTorch。

**这些特性为我们提供了在昇腾 NPU 上构建 DNN 模型时的优化见解**。

## 5. 结论

> 为了弥补在昇腾 NPU 上对深度学习框架评估的不足，本文对基于昇腾 NPU 的 MindSpore 和 PyTorch 进行了模型层面和算子层面的性能评估。

- 在**模型层面** 评估中，构建了**三个典型的 DNN 模型** （ResNet、GCN、GRU），并**分析和比较了在不同训练设置** （如 **AMP、图模式和多 NPU 并行**）下的结果。
- 在**算子层面** 评估中，设置了**不同数量的参数**，以评估 MindSpore 和 PyTorch 在昇腾 NPU 上的算子（**卷积、全连接、批量归一化**）性能。

通过分析评估结果，我们**发现了以下结论**：

1. 在昇腾 NPU 上，**PyTorch 在密集计算场景中的性能不如 MindSpore**，而在**稀疏计算场景中的性能则优于 MindSpore**；
2. PyTorch 和 MindSpore 在昇腾 NPU 上都表现出明显的第一个 epoch 初始化现象；
3. **合理设置 AMP 可以提高** PyTorch 和 MindSpore 的**性能**；
4. **MindSpore 在动态图** 模式下的**性能较差**；
5. 与 PyTorch 相比，**MindSpore 更能充分利用多个昇腾 NPU 带来的计算能力提升**。

基于这些结论，本研究为深度学习研究人员和从业者**在昇腾 NPU 上的框架选择** 和**优化策略** 提供了参考和建议。

## 参考文献

1. I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT press, 2016.
2. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2012.
3. C. Szegedy et al., “Going deeper with convolutions,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1–9, 2015.
4. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, Jun. 2016.
5. “Strategies for training large scale neural network language models,” IEEE Xplore, Dec. 01, 2011.
6. G. Hinton et al., “Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups,” IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 82–97, Nov. 2012.
7. R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa, “Natural Language Processing (Almost) from Scratch,” arXiv.org, Nov. 2011.
8. N. P. Jouppi et al., “In-Datacenter Performance Analysis of a Tensor Processing Unit,” in Proceedings of the 44th Annual International Symposium on Computer Architecture - ISCA’17, 2017.
9. Y. Jia et al., “Caffe: Convolutional Architecture for Fast Feature Embedding,” in Proceedings of the ACM International Conference on Multimedia - MM’14, 2014.
10. F. Seide and A. Agarwal, “CNTK: Microsoft’s Open-Source Deep Learning Toolkit” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Aug. 2016.
11. T. Chen et al., “MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems,” arXiv.org, Dec. 2015.
12. S. M. Hu, D. Liang, G. Y. Yang, G. W. Yang, and W. Y. Zhou, “Jittor: a novel deep learning framework with meta-operators and unified graph execution,” Science China Information Sciences, vol. 63, no. 12, Nov. 2020.
13. Y. Ma, D. Yu, T. Wu, and H. Wang, “PaddlePaddle: An Open-Source Deep Learning Platform from Industrial Practice,” Frontiers of Data and Computing, vol. 1, no. 1, pp. 105–115, Oct. 2019.
14. M. Abadi et al., “TensorFlow: A system for large-scale machine learning,” Google Research, 2016.
15. A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems – NIPS’19, 2019.
16. F.Yu, “Research on the next-generation deep learning framework,” Big Data Research, 2020.
17. R. Collobert, K. Kavukcuoglu, and C. Farabet, “Torch7: A Matlab-like Environment for Machine Learning,” NIPS Workshop BigLearn, Jan. 2011.
18. W. Z. Lu, F. Zhang, Y. X. He, Y. G. Chen, J. D. Zhai, and X. Y. Du, “Evaluation and Optimization for Huawei Ascend Neural Network Accelerator,” vol. 45, no. 8, pp. 1618–1637, Chinese Journal of Computers, Aug. 2022.
19. X. Y. Liang, Ascend AI Processor Architecture and Programming: Principles and Applications of CANN. THU press, 2019.
20. H. Zhu et al., “Benchmarking and Analyzing Deep Neural Network Training,” in 2018 IEEE International Symposium on Workload Characterization, Dec. 2018.
21. C. Coleman et al., “Analysis of DAWNBench, a Time-to-Accuracy Machine Learning Performance Benchmark,” ACM SIGOPS Operating Systems Review, vol. 53, no. 1, pp. 14–25, Jul. 2019.
22. Y. Wang, G. Y. Wei, and D. Brooks, “A SYSTEMATIC METHODOLOGY FOR ANALYSIS OF DEEP LEARNING HARDWARE AND SOFTWARE PLATFORMS.” Accessed: Jun. 29, 2023. [Online]. Available: https://vlsiarch.eecs.harvard.edu/files/vlsiach/files/mlsys 2020-a-systematic-methodology-for-analysis-of-deep-learning-hardware-and-software-platforms-paper.pdf
23. P. Mattson et al., “MLPerf: An Industry Standard Benchmark Suite for Machine Learning Performance,” IEEE Micro, vol. 40, no. 2, pp. 8–16, Mar. 2020.
24. W. Zhang, W. Wei, L. Xu, L. Jin, and C. Li, “AI Matrix: A Deep Learning Benchmark for Alibaba Data Centers,” arXiv.org, Sep. 2019.
25. L. Liu, Y. Wu, W. Wei, W. Cao, S. Sahin, and Q. Zhang, “Benchmarking Deep Learning Frameworks: Design Considerations, Metrics and Beyond,” IEEE Xplore, Jul. 2018.
26. S. Shi, Q. Wang, P. Xu, and X. Chu, “Benchmarking State-of-the-Art Deep Learning Software Tools,” IEEE Xplore, Nov. 2016.
27. H. Dai, X. Peng, X. Shi, L. He, Q. Xiong, and H. Jin, “Reveal training performance mystery between TensorFlow and PyTorch in the single GPU environment,” Science China Information Sciences, vol. 65, no. 1, Dec. 2021.
28. A. Shatnawi, G. Al-Bdour, R. Al-Qurran, and M. Al-Ayyoub, “A comparative study of open source deep learning frameworks,” IEEE Xplore, Apr. 2018.
29. Y. Wu, W. Cao, S. Sahin, and L. Liu, “Experimental Characterizations and Analysis of Deep Learning Frameworks,” in 2018 IEEE International Conference on Big Data (Big Data), Dec. 2018.
30. S. Bahrampour, N. Ramakrishnan, L. Schott, and M. Shah, “Comparative Study of Deep Learning Software Frameworks,” arXiv.org, 2015.
31. A. Karki, C. P. Keshava, S. M. Shivakumar, J. Skow, G. M. Hegde, and H. Jeon, “Detailed Characterization of Deep Neural Networks on GPUs and FPGAs,” in Proceedings of the 12th Workshop on General Purpose Processing Using GPUs - GPGPU’19, Apr. 2019.
32. K. Cho et al., “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.
33. H. Zhang, G. Lu, M. Zhan, and B. Zhang, “Semi-Supervised Classification of Graph Convolutional Networks with Laplacian Rank Constraints,” Neural Processing Letters, Jan. 2021.
34. A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” Semantic Scholar, 2009.
35. J. Elman, “Finding structure in time,” Cognitive Science, vol. 14, no. 2, pp. 179–211, Jun. 1990.
36. I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” in Proceedings of the 27th International Conference on Neural Information Processing Systems – NIPS’14, Dec. 2014.
37. D. Elliott, S. Frank, K. Sima’an, and L. Specia, “Multi30K: Multi lingual English-German Image Descriptions,” arXiv.org, May 2016.
38. A. K. McCallum, K. Nigam, J. Rennie, and K. Seymore, “Automating the Construction of Internet Portals with Machine Learning”, Information Retrieval, vol. 3, no. 2, pp. 127–163, 2000.
39. Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, “Gradient-based learning applied to document recognition,” in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998.
40. G. E. Hinton, “Reducing the Dimensionality of Data with Neural Networks,” Science, vol. 313, no. 5786, pp. 504–507, Jul. 2006.
41. S. Ioffe, and C. Szegedy, “Batch normalization: accelerating deep network training by reducing internal covariate shift,” in Proceedings of the 32nd International Conference on International Conference on Machine Learning - ICML’15, vol. 37, pp. 448–456, Jul. 2015.
42. K. Papineni, S. Roukos, T. Ward, and W. J. Zhu, “Bleu: a Method for Automatic Evaluation of Machine Translation” in Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02, 2001.
