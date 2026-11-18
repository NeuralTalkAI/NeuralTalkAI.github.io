> 机器学习（ML）编译器随着人工智能算法在各种设备上的广泛应用，已成为部署过程中的关键工具。当前的 ML 框架支持针对多种硬件后端的多级优化，但仅关注 ML 生命周期中的推理阶段。

然而，在**真实环境中将 ML 模型部署到硬件平台上时**，会**面临经典 ML 中已知的挑战**，例如**模型性能下降和分布偏移等问题**。通过在部署环境中执行领域自适应技术，可以为硬件平台解决这些问题。

::: block-1
本文提出了一种**基于 Apache TVM 的通用编译方法**，用于**嵌入式系统中的设备上训练**，并展示了**针对嵌入式 GPU 和 CPU 平台以及 AI 加速器** 的性能测量初步结果。这种编译方法将**简化** 深度神经网络的**部署过程**，并**使设备具备预建的训练能力**。
:::


![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/01.webp)

## 本文目录

- ![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/01.webp)
- 一、引言
- 二、相关工作
  - A. 领域自适应和迁移学习
  - B. 部署过程
  - C. Apache TVM 编译堆栈
  - D. Gemmini 加速器
- 三、针对设备上训练的新型编译方法
- 四、用于设备上训练的硬件加速器
  - A. 元素级矩阵乘法函数
- 五、实验评估
  - A. 在 Gemmini 加速器中实现逐元素矩阵乘法的硬件
  - B. 设备上的迁移学习
- 六、结论与未来工作
- 参考文献

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/02.webp)

## 一、引言

> 在过去十年中，机器学习（ML）方法已被应用于各种不同领域，并部署在多种设备上，例如人工智能（AI）加速器、中央处理单元（CPU）和图形处理单元（GPU）。在 ML 生命周期中可以识别出两个主要阶段：训练和推理。

- 在训练阶段，ML 模型正在学习，准备好的数据集被输入到选定的模型中，以发现模型将来能够识别的模式和依赖关系。
- 在推理阶段，模型参数是固定的，模型本身已经训练完成，可以在未见过的数据上进行预测。

将 ML 模型部署到真实环境中以处理实时数据是一项具有挑战性的任务，尤其是在嵌入式设备上：这些设备的内存和功率能力有限，同时要求实时推理，而 ML 模型通常复杂，包含数百万个参数，大小可能达到数百 MB。

为了在这些约束条件下运行，可以**对模型应用通用压缩方法以适应硬件** [1][2]。**另一种用于优化嵌入式设备上模型推理性能的方法是使用机器学习编译器** [3][4]。这些编译器包括上述技术，并考虑目标平台。然而，这些方法虽然必要，但还不足以解决 ML 模型部署到真实环境中后可能出现的模型性能下降问题。

**当当前环境的数据（目标域）与训练数据（源域）显著不同时**，部署的模型的**准确性可能会下降** [5]。在这种情况下，**应用设备上的领域自适应（DA）技术可以在不同域中使用新收集的数据对预训练模型进行适应，** 从而提高模型的预测能力。

此外，嵌入式系统在设备上训练并考虑本地收集的数据的能力，可以将数据保留在数据源附近，并限制需要传输到云端进行处理的数据量。这种策略有潜力缓解隐私和数据存储安全问题。为了使部署的设备能够重新训练，并自**动将 DA 算法集成到 ML 编译过程中**，本文的贡献如下：

1. **我们提出了一种基于 Apache TVM 的通用编译方法**，用于在**嵌入式设备上进行训练的图构建**，并**插入反向传播节点**。
2. 我们介绍了一种扩展方法，**将开源 AI 加速器 Gemmini 转换为支持设备上训练的硬件加速器**。

我们**在一系列嵌入式设备上评估** 了所提出的概念，包括 **Raspberry Pi 4、Jetson Orin 和 Gemmini 加速器**。以下各节将详细讨论这些贡献。

---

本文接下来的内容：

::: block-1
- 第三节：介绍了所提出的带有设备上训练能力的编译方法的描述。
- 第四节：介绍了将开源硬件 AI 加速器转换为支持训练的加速器的初步步骤，这些加速器由所提出的编译堆栈支持。
- 第五节：展示了在一系列嵌入式系统上评估我们的通用编译方法的结果。
:::

## 二、相关工作

在本节中，我们简要介绍受益于预训练模型知识的 ML 技术。我们考虑了硬件平台上的现有部署过程、当前的 ML 编译器堆栈以及 AI 硬件加速器。

### A. 领域自适应和迁移学习

> 传统的 DA 方法旨在从一个相关领域提取知识并转移到另一个领域。这一过程可以改善目标域中模型的性能，通过用目标数据对模型进行微调，可以减少两个域之间的差异[6]。

一般来说，深度神经网络（DNN）具有分层结构，其中不同层学习不同的特征。底层的作用是捕获源数据集的高级特征，而顶层则更专注于目标域中的显式任务。根据 DA 技术，保持底层冻结，解冻并重新训练模型的一些顶层，将使模型能够学习本地数据集的特定特征[7]。迁移学习（TL）允许在预训练模型的顶部添加一个新的分类器，以利用先前网络学习的表示来为新样本提取特征。

### B. 部署过程

> 众所周知的端到端开源机器学习平台 TensorFlow 或 PyTorch 对于深度学习任务至关重要。这些框架支持完整的 ML 生命周期，但分别专注于训练和推理。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/03.webp)

例如，**图 1 中的蓝色块表示将模型部署到嵌入式系统的传统 ML 工作流程**。该过程采用预训练模型，并应用优化技术，包括压缩、量化和剪枝，以将模型转换到下一阶段。然后，考虑到当前嵌入式平台的独特硬件约束，对模型进行编译。

最终，可以在目标设备上执行固定模型的推理。然而，这种方法无法在设备上进行训练并考虑本地数据。**TensorFlow Lite 为大多数常见的嵌入式平台提供了推理 API** [8]。此外，它们**还考虑了设备上训练** 技术，这允许模型根据设备或最终用户的输入进行改进或个性化[9]。

然而，TensorFlow Lite 的这一功能**主要面向移动设备**。**TensorFlow Lite for Microcontrollers 旨在在微控制器和其他资源受限的设备上运行 ML 模型** [10]，但它**尚未引入设备上训练** 的能力。尽管如此，像关键字识别[11]这样的用例可能需要在微控制器上部署后进行设备上适应。还有最先进的框架支持嵌入式设备上的训练或集成反向传播[12]。

**MiniLearn** 旨在启用**低功耗物联网（IoT）设备上的重新训练**，并应用量化和反量化技术，以使模型能够**在资源受限的设备上重新训练** [13]。这项工作仅涵盖了一个目标设备，即 nRF-52840 片上系统。

**TinyOL 通过在线学习进行设备上训练**，这涉及在推理时间对模型进行训练[14]。

::: block-1
**本文的目标设备是 Arduino Nano 33 BLE**。

**MCUNet 通过算法-系统协同设计实现了微小的设备上训练** [15]。不幸的是，它**缺乏灵活性**，仅适用于作者提供的少数预训练模型。最先进的框架旨在在不同设备上启用设备上训练，但**所提出的方法并未涵盖广泛的平台**。

然而，这些新解决方案可以集成到 ML 编译堆栈中，并与我们的编译程序结合，以提供一种通用方法，该方法可以应用于任何硬件后端。
:::

### C. Apache TVM 编译堆栈

> 随着人工智能算法应用的增加，当前的 ML 编译器变得必要。它们确保深度学习工作负载的自动移植性，并提供从高级框架（如 TensorFlow、PyTorch）到广泛硬件后端的端到端编译过程。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/04.webp)

然而，它们大多数也仅专注于推理阶段。**图 2 中的蓝色块显示了 Apache TVM 的编译流程** [3]。它包括以下步骤：第一步是**导入 DNN 模型，并将其表示为高级中间表示（IR）**，该表示包含用于内部模型表示的一组函数（Relay）。

下一步是**将模型转换为几个子图**，这些子图用张量表达式（TE）描述。TE 是一种特定领域的语言，用于描述张量计算。它提供了几种调度原语，用于指定低级循环优化，例如分块、矢量化、并行化、展开和融合[16]。

随后，**应用自动调优模块 AutoTVM，以搜索目标设备的最佳调度**。自动调优步骤的结果是，为当前目标平台找到了最佳配置，子图通过低级优化传递进行了优化，**最终为硬件平台生成了代码**。**最后步骤显示为优化和目标转换**。在这一过程的最后，编译器生成的特定于编译器的代码仅包含给定 DNN 模型的推理，可以将其部署到设备上。

### D. Gemmini 加速器

用于训练 AI 加速器之一是 Google 提出的张量处理单元（TPU）[17]，其矩阵乘法单元具有脉动阵列机制。另一个**基于脉动阵列的加速器是 Gemmini，它是开源的**，并且尚未广泛用于训练。Gemmini **具有一个全栈 DNN 加速器生成器**，用于 DNN 工作负载[18]，并且它是 **Apache TVM 支持的可能硬件后端之一** [19]。一个额外的好处是，该加速器不需要额外的编译器。**Gemmini 运行在 RISC-V 指令集架构上，可以通过从宿主处理器直接传输的一系列自定义 RISC-V 指令对其进行编程**。这也意味着开发人员在**扩展指令集以执行训练方面具有灵活性**。

## 三、针对设备上训练的新型编译方法

> 在现有技术中，存在一个空白，即没有框架能够将领域自适应（DA）技术应用于一系列嵌入式计算平台上的模型部署。如果不更改当前的堆栈或对模型进行任何更改，则需要重复描述的编译过程，以生成新的推理。对于已经集成到生产中的设备来说，这种方法是不可行的。

尽管可以通过远程方式在设备上更新模型参数并上传它们，但**当前域的数据需要被收集并发送到另一个设备上以重新训练和改进模型**。为了支持嵌入式设备重新训练深度神经网络（DNN）模型，我们提出了一个新的机器学习（ML）部署工作流程范式，该范式不仅支持推理，还支持在设备上训练模型。

**建议的流程以红色块显示在图 1 中**。工作流程的起点是一个 ML 模型，前两个训练和转换模型的步骤保持不变。然而，**与传统程序的主要区别在于，模型部署阶段还考虑了需要训练的模型部分。这些新节点在编译过程开始之前被插入到推理部分中**。因此，推理和选定的训练部分可以在目标设备上执行。

为了弥合这一差距并可能支持一系列目标平台，本文提出了**基于 Apache TVM 的设备上训练的编译堆栈，如图 2 所示**。流程的蓝色主干块已经讨论过，而红色块（虚线）代表本文贡献，并显示了在部署 DNN 模型后**提供训练的附加功能**。在堆栈中集成**附加功能的合适位置是在高级 IR 之后**。

> 此时，模型已经被翻译成 Relay，一种用于 DNN 的函数式语言，并被表示为 IR。这个位置允许我们拥有一个与原始模型相似的表示，并利用编译器后续自动应用的优化。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/05.webp)

因此，梯度调用（Gradient Call）会访问损失函数图的所有部分，并返回关于模型参数的偏导数作为附加图。**图 3 以单层神经网络（NN）为例，展示了这一过程的高级 IR 的简化版本**。该 NN 由一个密集层和一个 ReLU 激活函数组成。如前文 II-A 节所述，密集层通常是被重新训练用于本地数据的顶层分类层，以提供迁移学习（Transfer Learning），而 NN 的其余层则保持冻结状态。前向传播（Inference）的子图被表示为前向（fwd），反向传播（Backward）的子图被表示为后向（bwd），返回的元组是（fwd，bwd）。**这个元组是构建将在嵌入式设备上执行的推理和训练节点的初始基础**。

因此，前向子图对应于推理，后向子图对应于训练节点。反向传播阶段之后，用于更新模型参数的下一步是使用梯度下降（Gradient Descent）。为了在设备上执行梯度下降，我们在梯度调用之后的编译堆栈中集成了梯度下降。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/06.webp)

**图 4 展示了损失函数梯度背后的计算过程**。为了更好地理解，以下是这些公式的解释：

- 公式 1 表示单层 NN，其中 $Y$ 对应于 fwd，$\hat{Y}$ 对应于 y labels（真实标签），$X$ 对应于数据，$W$ 对应于权重，$B$ 对应于偏置。均方误差（Mean Squared Error，MSE）函数被选为损失函数，其导数在公式 2 中展示。
- 公式 3 是关于权重的误差导数（grad w in 图 3），$f'(Y)$ 是 ReLU 激活函数的导数（grad b in 图 3）。图 3 中以 bwd 表示的图返回模型参数的偏导数（grad w，grad b）。
- 最后一步是从权重和偏置中减去它们乘以学习率的结果。更新后的参数分别在公式 6 和公式 7 中展示。参数更新的图表示被堆叠在主推理图之后，完成了反向传播计算。

**在编译过程结束时，有两个节点如图 2 所示，分别表示推理和训练节点**。在降低到机器代码之前的中间表示由两个图组成：推理和参数更新。这两个子图中的一个将在设备上单独执行。**为了重新训练模型并确保模型参数更新，需要多次执行第二个子图。**

## 四、用于设备上训练的硬件加速器

> 本文的**第二个贡献是将一个开源硬件 AI 加速器转换为支持所提出编译堆栈的训练加速器的初步步骤**。一旦生成了特定于编译器的代码，就可以将其降低到机器代码，然后部署到目标设备上。然而，**在此过程中，必须考虑硬件架构**，以确保模型针对当前设备进行了优化。

### A. 元素级矩阵乘法函数

由于大多数提供的编译器或加速器功能都专注于推理阶段，因此缺乏与训练相关的功能。

其中之一是元素级矩阵乘法（matmul）。它在反向传播的优化过程中是一个基本功能。雅可比矩阵（Jacobian matrix）的乘法计算了每个输入变量相对于每个输出变量的偏导数[20]，通过逐元素计算激活函数的导数来实现。因此，应用独立的逐元素计算是这些乘法的最有效方式。

为了进一步支持高效实现并简化在编译过程中将降低的 TVM 函数映射到提供的 Gemmini API 的过程，我们在 Apache TVM 以及 Gemmini 加速器上实现了这一功能。

## 五、实验评估

> 在本节中，我们将提出的编译方法应用于迁移学习（Transfer Learning，TL），并在一系列嵌入式设备上对其进行评估。此外，我们还展示了在 Gemmini 加速器上集成的逐元素矩阵乘法（element-wise matrix multiplication）函数的加速效果。

### A. 在 Gemmini 加速器中实现逐元素矩阵乘法的硬件

为了评估部分，我们构建了以下硬件设置：将 Rocket Core [21]与一个 16×16 的脉动阵列（systolic array）Gemmini 加速器相连，该加速器配备了 256 KB 的暂存区（scratchpad，由 4 个 SRAM 银行组成，每个银行有 4096 行）和 64 KB 的累加器（1 个 SRAM 银行，1024 行）。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/07.webp)

**Gemmini 生成器可以生成硬件和软件堆栈，有两种可能的级别可以扩展加速器以支持新功能**。由于它连接到 CPU，因此可以在 CPU 上执行指令，也可以在加速器上执行。因此，**实现新功能有三种方法**：

- 第一种方法是在 CPU 上执行，通过两个循环逐个乘以元素，标记为（a），如表 I 所示。
- 第二种方法（标记为 b）是使用生成的软件堆栈的低级部分，其中可以通过提供的 C/C++ API 对加速器进行编程，并基于已有的标准矩阵乘法函数添加该功能。在这种情况下，两个矩阵的所有元素都变成对角线元素，然后应用标准乘法。之后需要提取对角线元素以获得结果。
- **第三种方法（标记为 c）是最高效的**：它利用**脉动阵列架构**，在加速器上执行逐元素乘法。

为此，我们为第三种方法实现了一个新指令，并扩展了当前的指令集架构。表 I 显示了总体结果：与在 CPU 上计算（a）或在加速器上使用矩阵乘法函数并随后提取对角线（b）相比，Gemmini 实现（c）**Gemmini 生成器可以生成硬件和软件堆栈，有两种可能的级别可以扩展加速器以支持新功能 **的性能提升。

需要注意的是，结果以时钟周期表示，Gemmini 和 CPU 使用相同的时钟频率。结果表明，**在 Gemmini 上执行集成的逐元素矩阵乘法比现有的 CPU 解决方案快 8.5 倍**。

### B. 设备上的迁移学习

> 在本节中，我们**在不同的架构（嵌入式 CPU 和 GPU，以及专用 AI 加速器）上评估了所提出的通用编译方法**，用于迁移学习任务。使用**编译器堆栈作为我们概念的基础的主要优势是，编译器自动支持具有不同架构的设备**。

这意味着，利用这一编译器特性**可以在中间级别进行必要的更改，这些更改在这一阶段对所有设备都是相同的**，最终覆盖不同的目标平台，例如嵌入式 GPU 和 CPU 以及专用加速器。

我们以迁移学习为例来展示这一概念。迁移学习是一种机器学习方法，其中预训练的模型被用作新任务模型的起点。我们从四个不同大小的模型（MobileNetV2、ResNet-50、DenseNet-121 和 VGG-16）创建了基础模型，这些模型已在 ImageNet 数据集上预训练了 1000 个类别 [24]。下一步是为每个基础模型创建并训练一个分类头部。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/08.webp)

有两种方法可以实现这一过程。

- 第一种方法是使用具有冻结权重和偏置的预训练模型，并在其输出顶部创建一个新的头部模型，然后将由两部分组成的模型作为一个整体进行训练。
- 第二种方法是将数据集通过预训练模型（图 5 中的基础模型），记录输出并将其保存为一个新的数据集。此外，创建一个新模型，并在新创建的数据集上对其进行训练。

在本工作中，为了将迁移学习任务集成到我们的编译方法中，我们选择了**第二种方法。这种方法更高效**，因为数据集只通过基础模型传递一次，而不是每个训练周期都传递一次。因此，我们将开源的 2 类和 5 类分类数据集 [22]、[23] 通过基础模型传递，并创建了新的数据集，以便在设备上重新训练每个头部模型。

**如图 5 所示，头部模型由 DenseNet-121 和 VGG-16 网络的两层组成，而 MobileNetV2 和 ResNet-50 则有一个分类层（图 5 中的头部模型）**。选择当前的基础模型和头部模型中的层数是为了展示不同设备的训练能力和训练所花费的时间。将我们的编译堆栈应用于当前的迁移学习任务后，我们编译了具有重新训练能力的模型。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/09.webp)

一旦构建完成，中间表示被部署在三个嵌入式平台上（NVIDIA Jetson Orin、Raspberry Pi 4（ARM Cortex-A72）和 Gemmini 加速器）。这个中间表示包括头部模型的反向传播节点，以及基础模型和头部模型的前向传播节点，用于推理。这使得模型在部署过程完成后可以在目标设备上开始适应。

**表 II 显示了四个模型和三个平台的设备上迁移学习的结果**。为了验证我们的方法和设备上训练的成功，表 II 包括了一个 TensorFlow 列，并展示了在 NVIDIA RTX A6000 上使用 TensorFlow 进行迁移学习时的准确率。表中的“Fwd”和“Bwd”分别代表前向传播和反向传播，反映了这些操作每次运行的持续时间。

由于模型被拆分为基础模型和头部模型，推理的执行时间（“Fwd”）分别对两个网络进行了测量，而反向传播（“Bwd”）仅对头部模型进行了测量，考虑到基础模型的参数是冻结的，没有重新训练。对于所有测量的示例，批次大小等于 32。**Gemmini 加速器上的模型性能测量是通过 Verilator（一个周期精确的模拟器）获得的** [25]。这些数字被转换为时间，假设加速器的时钟频率为 1 GHz，如 [18] 中所述。

![文章插图](/articles/2025-05-03-ai-ai-tvm-risc-v-a6133d5b/images/10.webp)

其他测量结果是直接从板卡上使用分析工具获得的。由于基础模型已经预训练过，并且特征已经由该模型提取，因此在设备上从头开始训练头部模型只需要少量的训练周期。我们**在 Raspberry Pi 4 上使用我们的编译方法在验证集上达到了 94%的准确率，经过 10 个训练周期（如图 6 所示）**，而使用 TensorFlow 在 NVIDIA RTX A6000 上训练头部模型在相同数据集上的准确率为 97%。其他模型的训练周期也设置为 10 个。

**如表 II 所示，Gemmini 上 ResNet-50 的一次推理需要 6.3 秒。头部模型的全连接层的一次反向传播迭代需要 8.3 毫秒**。因此，对于 10 个训练周期和每个周期 61 个批次，整个适应过程将花费 5 秒，这与 ResNet-50 的推理时间相当。

- 将相同的计算应用于其他硬件平台，Jetson Orin 上训练一个新的分类头部需要 0.3 秒，占推理执行时间的 14%
- 而对于 Raspberry Pi 4，这一过程需要 0.7 秒，占推理时间的 7%。

需要注意的是，并非所有模型和设备都需要一个比推理运行更短或相当的训练过程。

然而，重要的是要强调，适应过程不需要像推理那样每次都进行。此外，设备上的学习可以在没有互联网连接的情况下进行，它减少了数据流量，因为不需要将数据上传到云端，并且设备上的训练本质上是保护隐私的。

**图 6 证明了我们方法的有效性，并展示了 MobileNetV2 迁移学习的训练和验证损失（图6上图）以及训练和验证准确率（图6下图）**，这些结果是针对嵌入式设备的，并与在 NVIDIA RTX 上使用 TensorFlow 训练的模型进行了比较。因此，迁移学习任务的评估示例确认了我们通用编译方法的功能以及嵌入式平台在部署过程完成后的训练能力。

## 六、结论与未来工作

> 我们提出了一个**基于 Apache TVM 的通用编译方法** 的概念，用于**在真实环境中对嵌入式系统进行设备上训练的机器学习（ML）模型**。

我们展示了一个**编译堆栈**，该堆栈使设备**在完成部署过程后具备适应能力**。在一系列具有**不同架构的硬件平台上评估了** 这一概念，包括**嵌入式 GPU 和 CPU 以及专用 AI 加速器**。

我们展示了在这些**设备上执行迁移学习任务的示例**。此外，我们**在 Gemmini 加速器上实现了一个逐元素矩阵乘法函数**，以实现该设备上的**训练**。我们获得了**比当前 CPU 解决方案快 8.5 倍的加速效果**。

我们对未来设备上训练的编译流程的工作包括：为基于脉动阵列的加速器进行进一步优化，探索量化推理和浮点反向传播的结合，以及集成领域自适应算法。

## 参考文献

1. Jacob B., Kligys S., Chen B., Zhu M., Tang M., Howard A., Adam H., Kalenichenko D. “Quantization and training of neural networks for efficient integer-arithmetic-only inference.” In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.
2. Hotfilter T., Hoefer J., Kreß F., Kempf L., Kraft T., Harbaum T., Becker J. “A hardware-centric approach to increase and prune regular activation sparsity in cnns.” In 2023 IEEE 5th International Conference on Artificial Intelligence Circuits and Systems (AICAS), 2023, pp. 1–5.
3. Chen T., Moreau T., Jiang Z., Zheng L., Yan H., Shen M., Cowan L., Wang L., Hu L., Ceze L. “TVM: An automated End-to-End optimizing compiler for deep learning.” In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), 2018, pp. 578–594.
4. Lattner C., Amini U., Bondhugula A., Cohen A., Davis J., Pienaar J., Riddle R., Shpeisman T., Vasilache N., Zinenko O. “MLIR: Scaling compiler infrastructure for domain specific computation.” In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), 2021, pp. 2–14.
5. Zhang A., Xing L., Zou J., Wu J. “Shifting machine learning for healthcare from development to deployment and from models to data.” Nature biomedical engineering, vol.6, no. 12, p. 1330—1345, December 2022. Available: https://www.nature.com/articles/s41551-022-00898-y.pdf
6. Wang M., Deng W. “Deep visual domain adaptation: A survey.” Neurocomputing, vol. 312, pp. 135–153, 2018. Available: https://www.sciencedirect.com/science/article/pii/S0925231218306684
7. Too E., Yujian L., Njuki S., Yingchun L. “A comparative study of fine-tuning deep learning models for plant disease identification.” Computers and Electronics in Agriculture, vol. 161, pp. 272–279, 2019. bigData and DSS in Agriculture. Available: https://www.sciencedirect.com/science/article/pii/S0168169917313303
8. “Tensorflow lite inference.” 2024. Available: https://www.tensorflow.org/lite/guide/inference#supported\ platforms
9. “On-device training with tensorflow lite.” 2024. Available: https://www.tensorflow.org/lite/examples/on\ device\ training/overview#setup
10. “Tensorflow lite for microcontrollers.” 2024. Available: https://www.tensorflow.org/lite/microcontrollers
11. “Ai speech recognition with tensorflow lite for microcontrollers and sparkfun edge.” 2024. Available: https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow/#0
12. Zhu S., Voigt T., Ko J., Rahimian F. “On-device training: A first overview on existing systems.” ArXiv, vol. abs/2212.00824, 2022. Available: https://api.semanticscholar.org/CorpusID:254208667
13. Profentzas C., Almgren M., Landsiedel O. “Minilearn: On-device learning for low-power iot devices.” In Proceedings of the 2022 International Conference on Embedded Wireless Systems and Networks, ser. EWSN ’22. New York, NY, USA: Association for Computing Machinery, 2023, p. 1–11.
14. Ren H., Anicic D., Runkler T. “Tinyol: Tinyml with online learning on microcontrollers.” 2021 International Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2021. Available: https://api.semanticscholar.org/CorpusID:23222741
15. Lin J., Zhu L., Chen W., Wang C., Gan C., Han S. “On-device training under 256kb memory.” In Annual Conference on Neural Information Processing Systems (NeurIPS), 2022.
16. “TVM. An overview of tvm and model optimization.” 2023. Available: https://tvm.apache.org/docs/tutorial/introduction.html#sphx-glr-download-tutorial-introduction-py
17. Jouppi N., Young C., Patil N., Patterson D., Agrawal G., Bajwa R., Bates S., Bhatia S., Boden A., Borchers A., Boyle R., Cantin C., Chao P., Clark C., Coriell M., Daley M., Dau J., Dean J., Gelb T., Ghaemmaghami R., Gottipati R., Gulland W., Hagmann R., Ho C., Hogberg J., Hu J., Hundt R., Hurt C., Ibarz J., Jaffey A., Jaworski A., Kaplan H., Khaitan S., Killebrew A., Koch N., Kumar S., Lacy J., Laudon J., Law C., Le D., Leary Z., Liu K., Lucke A., Lundin G., MacKean A., Maggiore M., Mahony K., Miller R., Nagarajan R., Narayanaswami R., Ni K., Nix T., Norrie M., Omernick M., Penukonda A., Phelps J., Ross R., Ross R., Salek A., Samadiani E., Severn G., Sizikov M., Snelham J., Souter D., Steinberg A., Swing M., Tan G., Thorson B., Tian H., Toma E., Tuttle V., Vasudevan R., Walter W., Wang E., Wilcox D., Yoon D. “In-datacenter performance analysis of a tensor processing unit.” SIGARCH Comput. Archit. News, vol. 45, no. 2, p. 1–12, jun 2017. Available: https://doi.org/10.1145/3140659.3080246
18. Genc H., Kim S., Amid A., Haj-Ali V., Iyer P., Prakash P., Zhao J., Grubb H., Liew H., Mao A., Ou C., Schmidt S., Steffl J., Wright I., Stoica I., Ragan-Kelley J., Asanovic K., Nikolic B., Shao Y. “Gemmini: Enabling systematic deep-learning architecture evaluation via full-stack integration.” In Proceedings of the 58th Annual Design Automation Conference (DAC), 2021.
19. Peccia F., Bringmann O. “Integration of a systolic array based hardware accelerator into a dnn operator auto-tuning framework.” 12 2022.
20. Parr T., Howard J. “The matrix calculus you need for deep learning.” 02 2018.
21. Asanovi´c K., Avizienis R., Bachrach J., Beamer S., Biancolin D., Celio C., Cook D., Dabbelt D., Hauser J., Izraelevitz A., Karandikar S., Keller B., Kim D., Koenig Y., Lee E., Love M., Maas A., Magyar H., Mao M., Moreto A., Patterson D., Richards C., Schmidt S., Twigg D., Waterman A. “The rocket chip generator.” EECS Department, University of California, Berkeley, Tech. Rep. UCB/EECS-2016-17, Apr 2016. Available: http://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-17.html
22. Cukierski W. “Dogs vs. cats.” 2013. Available: https://kaggle.com/competitions/dogs-vs-cats
23. Mamaev A. “Flowers dataset.” https://universe.roboflow.com/joseph-nelson/flowers, apr 2020, visited on 2024-07-28. Available: https://universe.roboflow.com/joseph-nelson/flowers
24. Deng J., Dong W., Socher R., Li L., Li K., Fei-Fei L. “Imagenet: A large-scale hierarchical image database.” In 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.
25. “Verilator.” 2024. Available: https://www.veripool.org/verilator/
