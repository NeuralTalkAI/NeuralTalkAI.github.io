关键词：**Apple Silicon**、:mdnice-bold-italic[专用硬件加速器(DSA)、] ANE、NUC、HPC、性能评估

> 当我们在谈论苹果自研芯片 M1、M2、M3 乃至最新的 M4 时，我们在谈论什么？绝大多数人的第一反应是其惊人的能效比、无风扇的轻薄本体验，或是那颗用于剪辑 ProRes 视频的强大媒体引擎。

但在这些光鲜的表面之下，苹果芯片中其实一直藏着一个极为低调却又潜力巨大的单元——**苹果神经引擎**。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/01.webp)

- Evaluation of Domain-Specific Architectures for General-Purpose Applications in Apple Silicon
- https://arxiv.org/pdf/2511.13450
- 1 万字，阅读 40 分钟，**播客 18 分钟**

::: related-title
相关推荐
:::

::: related-links
- [Apple Silicon LLM 推理全方位深度剖析：与 NVIDIA GPU 从性能到成本的终极对决！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890883&idx=1&sn=d05b1dbc8df195d7a60dfc6d2bee4d6f&scene=21#wechat_redirect)
- [生产级 Apple Silicon LLM 推理框架深入分析对比：MLX, MLC-LLM, Ollama, llama.cpp](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894128&idx=1&sn=1af2f122be249b52eb4a764227b28c83&scene=21#wechat_redirect)
- [苹果 M5 芯片发布，AI 性能狂飙 4 倍！GPU 架构重构 + 16 核神经引擎 + 10 核 CPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893129&idx=2&sn=3a2db8a1d0c83334ff424c4b7cfea9c5&scene=21#wechat_redirect)

:::

> 人工智能的崛起及其日益增长的计算需求，推动了领域专用加速器，如 GPU、TPU 和 NPU 在整个计算基础设施中的集成。继 GPGPU 使 GPU 普及于通用任务的先例之后，本研究提出了一个问题：这种模式是否能在新的背景下，通过像 NPU 这样的专用加速器得以复制？

**本文评估了专为机器学习工作负载设计、具有高能效特性的苹果神经引擎（ANE）在通用高性能计算应用中的潜力**。我们通过:mdnice-bold-italic[在 M1 及最新的 M4 架构上运行经典的 HPC 算法]，如 GEMM、Jacobi 和 Multigrid（多重网格法），评估了苹果 ANE 的性能和能耗。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/02.webp)

::: image-caption
表 8 | 以 M1 芯片 CPU 的 FP16 精度执行性能为基准的加速比。该表量化了各单元相对 M1 CPU FP16 的加速能力，M4 Pro 的 GPU 加速比提升最显著，达 20 倍以上，CPU 也有 3-4.5 倍提升。ANE 虽绝对加速比最高，但代际提升仅 1.5-2.5 倍，远低于 CPU 和 GPU，印证了 M4 Pro 大幅缩小了通用计算单元与 ANE 的性能差距。
:::

::: block-1
结果证实，当算法被恰当地适配后，ANE 在 M4-Pro 芯片上实现了具有竞争力的性能：**GEMM 运算达到 3.8 TFlops，与同一 SoC 上 GPU 的 4.7 TFlops 相当**，同时展现出显著**更优的能效**，例如，在 M4 架构上，GEMM **在 ANE 上的功耗为 5.2 瓦，而在 GPU 上则为 24 瓦。**
:::

## 本文目录

- 一、引言
- 二、苹果硅芯片
  - A. CPU
  - B. GPU
  - C. ANE
  - D. CoreMLTools
- 三、实验环境
  - 3.1 M1
  - 3.2 M4
- 四、目标应用
  - A. YOLOv3
  - B. YOLOv11
  - C. GEMM
  - D. Jacobi
  - E. Multigrid
- 五、结果
  - 5.1 实验环境
  - 5.2 实验结果
  - 5.3 YOLOv11
  - 5.4 GEMM
  - 5.5 Jacobi
  - 5.6 Multigrid
  - 5.7 讨论
- 六、结论
- 七、参考文献

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/03.webp)

交流加群请在

NeuralTalk 公众号

后台回复：

加群

## 一、引言

> 随着百亿亿次计算时代的到来，我们见证了由人工智能的崛起和大数据时代催生的大量数据处理所驱动的工作负载爆炸式增长。这一趋势，加上高性能计算系统计算能力的不断提升，推动了更复杂技术的创造，其中最显著的例子便是最近的生成式人工智能。

**得益于技术和系统架构的进步，HPC 节点不仅集成了越来越多的高端处理器，还集成了领域专用加速器**。例如，GPU 在过去几十年中用于通用应用，促进了其在各个科学领域的普及[1]并推动了其使用。此外，GPU 的广泛可用性和进步也助推了最近的人工智能革命[2]，使得训练和部署比以往更复杂的模型成为可能。

人工智能历史上的一个关键里程碑是 AlexNet 的出现[3]，它是现代图像分类与识别、自然语言处理和生成式人工智能模型发展的先驱。这些应用不断升级的计算需求，推动了专用加速器的设计——例如 **TPU（张量处理单元）和 NPU（神经网络处理单元）——这些加速器现在已被集成到整个计算堆栈中** [4]，从嵌入式系统、台式计算机到高性能服务器、HPC 基础设施和大型数据中心。

**NPU[5], [6]已成为台式机计算的基本组件**，特别是在推动了如 ChatGPT 和 DeepSeek 等广泛使用工具的 Transformers 架构[7]问世之后。这种日益增长的兴趣反映在领先科技公司（包括 AMD、三星、英特尔和苹果）对专用 NPU 硬件的开发上。

::: block-1
:mdnice-bold-italic[这一趋势的明显例子是苹果公司在 2020 年推出了基于系统级芯片的 M1 芯片[8]，它集成了高性能和高能效的多核 CPU、GPU、一个矩阵乘法协处理器以及专用的神经引擎。这种集成使得设备能够高效地利用其能力，为从日常任务到繁重图形处理和 AI 应用的各个方面提供高性能]。自那时起，苹果已将 M1 发展到更新版本，随后发布了 M2、M3 和 M4 芯片，M5 预计于 2026 年推出。
:::

正如 GPU 通过过去的 GPGPU 运动在非图形上下文中获得普及一样，我们不排除这种现象在其他专用加速器（如 NPU）上重演。**这项工作评估了在通用应用背景下使用苹果 ANE 及其他加速器的情况，以确定这类设备是否能在性能和能效方面与台式机系统中当前可用的其他设备相竞争**。

探索 NPU 用于其他用途的相关工作中，我们可以找到：

- 像 Parrot Transformation [9]这样的方法，它探索用神经网络替换命令式代码段，以提高程序性能和能效。
- 另一个例子是 Hubner 等人[10]的研究，他们对苹果硅芯片架构进行了深入分析，重点介绍了其主要计算单元：CPU、GPU 和 ANE。

> Apple vs. oranges: Evaluating the apple silicon m-series socs for hpc performance and efficiency,” 2025. [Online]. Available: https://arxiv.org/abs/2502.05317

除了这些论文，ANE 仍然相对未被充分探索，尽管已有一些研究开始对其进行检查。

本文的其余部分结构如下：

| 章节 | 主要内容 |
| --- | --- |
| 第二部分、第三部分 | 介绍苹果硅芯片架构和开发环境 |
| 第四部分 | 介绍本研究中使用的应用程序 |
| 第五部分 | 讨论所取得的性能结果 |
| 第六部分 | 提供结论性意见和潜在的未来研究方向 |

## 二、苹果硅芯片

### A. CPU

> 苹果硅芯片 CPU 采用 big.LITTLE（大小核）架构，结合了两种类型的核心：高性能核心和高效能核心。**这种设计允许将工作负载分配到不同的核心上，旨在提高性能的同时优化功耗**。CPU 支持使用 C、C++、Objective-C 和 Swift 进行编程。此外，CPU 通过 128 位 NEON 指令提供对向量操作的支持，并依赖一个专门的协处理器进行矩阵计算：苹果矩阵扩展。[10]

- AMX：:mdnice-bold-italic[AMX 是一个协处理器，旨在与 CPU 一起加速矩阵运算。在最新的苹果硅芯片世代（M4）中引入了一项重要增强，即增加了对 ARM 可扩展矩阵扩展的支持，这是一套专为矩阵计算设计的 ARM 指令集]。从编程角度来看，可以通过 Accelerate 库（一个用于数值计算的高级 C++框架）来访问 AMX。AMX 也可以通过 CoreML 间接利用；然而，**在这种情况下，执行流程是抽象的，框架会自动决定计算是在 CPU 上执行还是卸载到 AMX**。目前，没有可用于低级编程 AMX 的官方文档。

### B. GPU

M 系列 GPU 使用基于图块的延迟渲染技术，将场景分割成图块并逐块渲染。该设备可以使用 Metal API（一种低级编程框架）配合 Metal 着色语言进行编程。最后，对于高级编程，可以使用 Metal Performance Shaders 和 CoreML 框架。

### C. ANE

> 苹果神经引擎是一个硬件加速器[10]，专门针对机器学习工作负载进行了优化。

:mdnice-bold-italic[与 AMX 不同，ANE 作为一个独立的处理单元运行，而不是协处理器。它首次作为 A11 芯片的一部分在 iPhone X 中发布，并被集成到苹果硅芯片设备中。它专门优化用于处理 FP16（16 位浮点数）数据]，尽管除了美国专利外，官方技术文档仍然稀缺。:mdnice-bold-italic[只能通过 CoreML 框架对 ANE 进行编程，但开发者无法直接控制其配置。]

### D. CoreMLTools

> CoreMLTools 是苹果用于机器学习操作的 Python 框架。

需要注意的是，选择此框架的优点是具有:mdnice-bold-italic[可移植性，允许在苹果 SoC 设备（如 CPU、GPU 或 ANE）之间执行任何应用程序。然而，它不允许单独直接控制每个设备]。

::: block-1
在实践中，如果 CoreML 认为最优，或者某个操作或层不被 GPU/ANE 支持，那么在所有执行模式下都仍可能使用 CPU。可用的执行选项有：
:::

| 枚举值 | 描述 |
| --- | --- |
| `ct.ComputeUnit.ALL` | 在 CPU、GPU 和 ANE 之间分布执行。 |
| `ct.ComputeUnit.CPU_ONLY` | 执行仅限于 CPU。 |
| `ct.ComputeUnit.CPU_AND_GPU` | 在 CPU 和 GPU 之间分布执行。 |
| `ct.ComputeUnit.CPU_AND_ANE` | 在 CPU 和 ANE 之间分布执行。 |

Core ML Tools 通过将模型从 TensorFlow 和 PyTorch 等流行的第三方框架转换为原生的 Core ML 格式，简化了模型的集成。然而，不同硬件支持的数据类型存在重要差异。**虽然 CPU 和 GPU 都支持 32 位浮点数和 16 位浮点数格式，但 ANE 仅限于使用 fp16 格式进行模型推理。**

## 三、实验环境

> 实验在两种不同的 Mac 系统上进行，以比较性能并观察苹果硬件的演进：Mac Mini M1 和 Mac Mini M4 Pro。

选择这些设备是因为：

- Mac Mini M1 代表了苹果用于 Mac 设备的系统级芯片的第一代
- 而 M4 Pro 是本研究进行时可用的最新型号

使用第一代和最新的苹果硅芯片型号，可以为了解苹果硅芯片的性能演进提供一个广阔的视角。这不仅可以在**芯片上集成的三种处理单元之间进行比较**，还可以**评估每个组件跨代际的改进情况。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/04.webp)

::: image-caption
表 1 | M1 与 M4 Pro 硬件平台的核心规格参数。该表完整呈现两代苹果硅芯片的核心硬件配置，M4 Pro 在 CPU 核心数、主频、GPU 核心数、NPU 算力、内存带宽上全面升级，还新增 ARM SME 指令集支持，双 AMX 单元也是关键提升。M1 仅单 AMX、8 核 GPU，而 M4 Pro 的 16 核 GPU、38TOPS NPU 让通用与 AI 计算能力实现质的飞跃，为性能差距奠定硬件基础。
:::

### 3.1 M1

M1 对苹果来说是一个里程碑式的产品，因为它是苹果硅芯片系列中的第一台计算机——这是苹果目前销售的 SoC 家族。

M1（2020 年）包括一个 16 核的 NPU、一个 8 核 GPU 以及一个基于 ARM big.LITTLE 架构的 8 核 CPU，具有四个效率核心和四个性能核心。该型号还集成了一个与 CPU 一起工作的 AMX 协处理器。

### 3.2 M4

与第一款苹果硅芯片型号相比，M4 Pro（2024 年）显示出显著的进步。它拥有一个 12 核 CPU，包含八个性能核心和四个效率核心。GPU 的核心数翻了一番，达到十六个，而 NPU 保持相同的核心数。

> 两台系统均使用 CoreML 通过 Python 进行编程，因为该框架提供了对所有三个处理单元的访问。所使用的库和框架版本总结如下：

| 工具/库 | 版本 | 用途 |
| --- | --- | --- |
| Python | 3.9.6 | 主要编程语言，因 CoreMLTools 支持选用 |
| PyTorch | 2.3.0 | 转换为 CoreML 格式前用于模型创建 |
| Torchvision | 0.18.0 | 处理模型架构和数据集 |
| NumPy | 2.0.2 | 矩阵操作 |
| CoreMLTools | 8.0b2 | 用于机器学习操作的框架 |

此外，**能耗测量需要使用 powermetrics**，以及最初使用的 asitop-tools。:mdnice-bold-italic[powermetrics 是一个 macOS 命令行工具，提供有关系统能耗使用的详细信息]。**在本研究中，功耗采样的时间间隔设置为 100 毫秒**。最后，Xcode 用于通过在其界面上直接检查模型来分析运算符与 ANE 的兼容性。本研究开发的代码可在公共代码仓库中找到。

## 四、目标应用

> 本研究主要分为两个阶段：

1. 使用广泛采用的 AI 模型，特别是 YOLO（v3 和 v11 版本），在 CPU、GPU 和 ANE 上评估苹果硅芯片设备的 AI 性能
2. 通过使用诸如 GEMM、Jacobi 和 Multigrid 等基准测试，将实验扩展到 HPC 背景下进行通用算法测试。

为此，实现了两种算法来求解热传导方程（见公式 1），该方程描述了热量随时间在离散化网格内的扩散。

$$
\frac{\partial u}{\partial t} = \left(\frac{\partial^2u}{\partial x_1^2} +\frac{\partial^2u}{\partial x_2^2} +\dots +\frac{\partial^2u}{\partial x_n^2}\right) \tag{1}
$$

> 以下小节描述了代码迁移到 CoreML 的过程。

### A. YOLOv3

YOLOv3（2018 年）是一个能够对 80 种不同类别进行分类的目标检测模型。选择此模型是为了在 AI 工作负载背景下**评估 CPU、GPU 和 ANE 的性能**。该模型来自苹果官方的 CoreML 示例代码库。实验执行了 100 次以收集实际的性能测量数据。

### B. YOLOv11

YOLOv11（2024 年）是由 Ultralytics 开发的 YOLO 家族最新版本之一。模型来自官方 GitHub 代码库。**由于提供的实现是 PyTorch 格式，因此必须转换为 CoreML 的.mlmodel 格式** （遵循苹果官方的转换指南）。选择此模型也是因为 YOLOv3 现在已相对过时。YOLOv11 提供了一个与当前 AI 模型发展状态相符的更更新的基准测试。

### C. GEMM

GEMM 是本研究中评估的第一个通用算法。选择此算法是因为它在高性能计算领域中的核心作用，及其在科学和工业应用中的广泛使用，此外它还被大量用于 AI 操作。对于后一种用途，GEMM 在 PyTorch 中已有原生实现，这有助于将其集成到我们的实验框架中。

通过在前向方法中**使用 PyTorch 操作 torch.matmul(A, B)，GEMM 很容易适配以在 ANE 上执行。**

### D. Jacobi

> Jacobi 方法是一种用于求解微分方程组的迭代算法。

$$
\frac{\partial u}{\partial t} = \left(\frac{\partial^2u}{\partial x_1^2} +\frac{\partial^2u}{\partial x_2^2} +\dots +\frac{\partial^2u}{\partial x_n^2}\right) \tag{1}
$$

特别地，我们选择了公式 1 中描述的热传导方程作为基准。**与 GEMM 不同，将 Jacobi 适配以在 ANE 上执行需要额外的修改，概述如下：**

1. 使用每个点的初始温度初始化网格。
2. 在每次迭代中，使用五点模板计算每个点的新温度。
3. 重复此过程多次，或直到达到收敛标准。

还需要考虑边界值，这些值随时间保持不变。:mdnice-bold-italic[将 Jacobi 方法迁移到 ANE 上是可行的，通过使用 conv2D（二维卷积）操作来计算每个点的更新温度。该计算与一个预计算的掩码相结合，以确保在每次迭代中维持网格的正确值。]

具体来说，创建了一个卷积核来计算每个点基于其直接邻居的新值，从而重新创建五点模板操作。该掩码（mask）在图像处理卷积中充当卷积核，因此可以使用 torch.tensor 定义创建一个 3x3 矩阵，在相关位置设置值为 0.25（参见代码清单 1）。应用卷积后，结果乘以预计算的掩码以保留边界值。

```
class JacobiMachine (nn.Module):    def __init__(self, nt = 1000, datatype = torch.float32):        super(JacobiMachine, self).__init__()        self.datatype = datatype        self.nt = torch.tensor(nt, dtype = self.datatype)    def forward(self, X, Xprev, Mask):        x = X        x_prev = Xprev        mask = Mask        kernel = torch.tensor([[0.0, 0.25, 0.0],                               [0.25, 0.0, 0.25],                               [0.0, 0.25, 0.0]], dtype = self.datatype).view(1, 1, 3, 3)        i = torch.tensor(0, dtype = self.datatype)        while torch.ne(i, self.nt):            x_prev = x.clone()            x_next = F.conv2d(x_prev, kernel, padding = 1)            x = x_next * mask            diff = torch.max(torch.abs(x - x_prev))            i = torch.add(i, 1)        return x
```

*代码清单 1. Jacobi 方法迁移*

### E. Multigrid

在所选算法中，**Multigrid 的实现最为复杂**。该模型遵循 V 循环结构设计，可分为三个主要阶段：下降、粗网格校正和上升。除了这些阶段外，还需要像 Jacobi 方法一样保留边界值。

#### Multigrid 阶段1：V 循环下降

> 下降阶段需要两个主要操作：平滑和残差计算。平滑使用前述 Jacobi 方法的一次迭代执行，并且需要计算平滑解的残差误差。实现在代码清单 2 中描述。

```
# V-cycle descentfor level in range(1, num_levels):    # ... (mask operations)    masked_output = F.conv2d(masked_input, kernel, padding = 1)    unmasked_output = F.conv2d(unmasked_input, kernel, padding = 1)    residual = masked_output + unmasked_output #
```

*代码清单 2. V 循环下降*

下降阶段的最后一步是网格的限定。在 ANE 上，此限定是使用 AI 模型中常用的平均池化层实现的。此操作的实现如代码清单 3 所示。

```
def restriction(self, residual):    return nn.AvgPool2d(kernel_size = 2)(residual)
```

*代码清单 3. 限定*

此限定应用于之前计算的残差，正如我们在代码清单 4 中看到的那样。

```
# ... (Inside the descent loop)coarse_residual = self.restriction(residual)grids.append(coarse_residual)
```

*代码清单 4. 限定的使用*

#### Multigrid 阶段2：最粗网格求解

一旦网格通过所有期望的层级被限定，最粗网格必须通过迭代求解。为此，我们应用 Jacobi 方法。代码清单 5 展示了在最粗网格上执行的计算实现。

```
coarse_solution = self.jacobi(coarse_solution, Mask9)grids[-1] = coarse_solution
```

*代码清单 5. 最粗网格求解*

#### Multigrid 阶段3：V 循环上升

在上升阶段，需要执行延拓以将网格恢复到原始大小。此延拓还需要进行一次校正，以补偿因粗网格近似而在每个层级引入的误差。延拓通过 PyTorch 中的 interpolate 函数使用双线性插值实现。此函数基于每个新点的四个最近邻点计算其值，从而在每一步有效地增加网格大小。此过程的实现如代码清单 6 所示。

```
def interpolate(self, f, target_size):    return F.interpolate(f, size = target_size, mode = 'bilinear', align_corners = False)
```

*代码清单 6. 插值代码*

在应用延拓之后，必须执行校正。此校正的实现如代码清单 7 所示。

```
1 # Upward phase2 for level in range(num_levels - 2, -1, -1):3     target_size = grids[level].shape[-2:]4     # Interpolate to finer grid5     fine_solution = self.interpolate(grids[level + 1], target_size = target_size)6     # Add correction to finer grid7     fine_solution = torch.add(fine_solution, grids[level])8     # ... (post-smoothing)
```

*代码清单 7. 应用于延拓后网格的校正*

最后，应用平滑步骤以精化解。此平滑过程即为 Jacobi 实现。

## 五、结果

### 5.1 实验环境

> 所选用于评估性能和能效的基于苹果 SoC 的平台是 M1 和 M4 Pro Mac-Mini 设备，其主要特性总结在表 1 中。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/05.webp)

::: image-caption
表 1 | M1 与 M4 Pro 硬件平台的核心规格参数。该表完整呈现两代苹果硅芯片的核心硬件配置，M4 Pro 在 CPU 核心数、主频、GPU 核心数、NPU 算力、内存带宽上全面升级，还新增 ARM SME 指令集支持，双 AMX 单元也是关键提升。M1 仅单 AMX、8 核 GPU，而 M4 Pro 的 16 核 GPU、38TOPS NPU 让通用与 AI 计算能力实现质的飞跃，为性能差距奠定硬件基础。
:::

从软件方面来看，所有机器学习模型均使用 PyTorch 框架开发和实现。为了部署到 Core ML，模型使用 CoreMLTools 进行转换。

### 5.2 实验结果

> 本小节对一系列不同计算任务的性能和能耗进行了比较分析。

具体来说，我们报告了两个不同的 AI 模型——**YOLOv3 和 YOLOv11——以及几个通用应用：标准矩阵乘法**、Jacobi 迭代方法和一个用于热传导方程的更复杂的多重网格求解器的结果。**下面表 2 总结了所有实验使用的规模大小。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/06.webp)

::: image-caption
表 2 | 各类基准测试所采用的问题规模。该表明确了 GEMM、雅可比、多重网格三类算法在 M1 和 M4 Pro 上的测试维度，M4 Pro 支持更大的矩阵 / 网格规模，如 GEMM 达 16384×16384、雅可比达 32768×32768。M1 因硬件限制无法运行超大规模测试，且三类算法的测试规模梯度设计，能充分验证各计算单元在不同数据量下的性能表现。
:::

#### YOLOv3

> 表 3 展示了 YOLOv3 模型处理单张图像的推理时间。由于模型规模较小，执行时间非常短（每次推理小于 15 毫秒），这本身就限制了性能比较的统计结论性。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/07.webp)

::: image-caption
表 3 | YOLOv3 模型的推理执行时间。该表记录了 M1、M4 Pro 在 CPU、GPU、ANE 上运行 YOLOv3 的单图推理耗时，单位为毫秒，ANE 在两代芯片上均为最快，M4 Pro 各单元耗时均低于 M1。但因 YOLOv3 模型规模小，耗时均不足 15 毫秒，统计参考性有限，无法充分体现各单元的性能差异与加速能力。
:::

尽管模型执行时间有局限性，但从能耗数据（见表 4）中可以得出一些有趣的观察结果。**在 M1 芯片上，GPU 模式的能耗最低，而在 M4 Pro 芯片上，ANE 模式的能效最高。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/08.webp)

::: image-caption
表 4 | YOLOv3 模型经过 100 次执行后的平均能耗。该表为 YOLOv3 百次推理的平均功耗数据，M1 上 GPU 功耗最低，M4 Pro 则是 ANE 功耗最优且与 M1 ANE 接近。M4 Pro 的 CPU、GPU 功耗均高于 M1，体现性能提升的功耗代价，而 ANE 保持低功耗优势，成为该轻量 AI 模型推理的能效最优选择。
:::

### 5.3 YOLOv11

> 为了获得更具统计结论性的结果，我们使用更新、更大的模型 YOLOv11 重复了实验。**我们评估了所有可用 YOLOv11 变体的推理时间（见表 5）：YOLOv11n、YOLOv11s、YOLOv11m、YOLOv11l 和 YOLOv11x。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/09.webp)

::: image-caption
表 5 | YOLOv11 模型的推理执行时间。该表覆盖 YOLOv11 全系列模型，对比两代芯片在不同浮点精度下的推理耗时，ANE 仅支持 FP16 且始终最快，M4 Pro 各单元耗时较 M1 大幅降低。M1 的 ANE 对 CPU 加速比达 13-20 倍，而 M4 Pro 仅 3-6 倍，因 M4 Pro 的 CPU、GPU 基准性能提升，大幅缩小了与 ANE 的差距。
:::

- 获得的结果确认了预期的性能层级：**专为 AI 工作负载设计的 ANE 始终产生最快的推理时间，其次是 GPU**。然而，在 M1 和 M4 Pro 芯片之间观察到了加速幅度的代际变化。:mdnice-bold-italic[聚焦于 ANE vs. CPU，M1 芯片上 ANE 相对于 CPU 实现了显著的加速，范围从 13 倍到 20 倍。相比之下，M4 Pro 显示出小得多的增益，加速因子仅在 3 倍到 6 倍之间。]
- :mdnice-bold-italic[从 GPU vs. CPU 的分析来看，性能差距的缩小甚至更为明显：M1 的 ANE 相对于 GPU 提供了 3 倍到 5 倍的加速，而在 M4 Pro 上则减少到 1.7 倍到 2.0 倍。]

::: block-1
:mdnice-bold-italic[这些结果表明，M4 Pro 的 CPU 和 GPU 的基线性能有了显著提升，暗示对于这个特定的推理任务，ANE 可能已经达到了其性能上限。]
:::

> 关于能耗，YOLOv11 模型的结果总结在表 6 中。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/10.webp)

::: image-caption
表 6 | YOLOv11 模型经过 100 次执行后的平均能耗。该表为 YOLOv11 百次推理的功耗数据，M4 Pro 各单元功耗均高于 M1，GPU 始终是功耗最高的单元。ANE 虽功耗高于 M1 同单元，但结合其最快的推理速度，仍是能效最优选择，且小模型 YOLOv11n 与大模型 YOLOv11x 的功耗差异，也反映了模型规模对硬件功耗的影响。
:::

M4 Pro 上的能耗明显更高，这证实了苹果硅芯片在代际性能提升的同时也带来了功耗的增加。在 M1 和 M4 Pro 上，GPU 始终记录着最高的功耗。考虑到能效角度，ANE 仍然是更高效的设备。

### 5.4 GEMM

> 为了有效评估基于 GEMM 的基准测试的性能，我们必须首先考虑苹果硅芯片架构的某些方面。该架构在每个多核簇内集成了一个名为 AMX 的专用单元。该协处理器卸载密集的矩阵运算，从而释放主 CPU 核心。

通过苹果的高级库（主要是 Accelerate 框架或 Core ML）对 AMX 进行编程。因此，我们希望通过两种编程模型在 CPU 设备上评估 GEMM 的行为。为此，**我们实现了一个使用 Accelerate 框架的基线 C++代码，以评估纯 CPU 执行与启用 AMX 的执行之间的性能差异**。Accelerate 框架可以通过其自身的 GEMM 实现轻松使用，我们只需要像代码清单 8 所示那样调用 cblas_sgemm 方法。

```
#include <Accelerate/Accelerate.h>// ... 初始化矩阵 A, B, C 和维度 m, n, k ...cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,            m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
```

*代码清单 8. 通过 Accelerate 使用 AMX 进行矩阵乘法*

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/11.webp)

::: image-caption
表 7 | M1 与 M4 Pro 芯片的 CPU 执行 GEMM 算法的性能（单位：千兆次浮点运算 / 秒）。该表对比了 Accelerate 单 / 双线程与 CoreML 在不同矩阵规模下的 GEMM 性能，M4 Pro 因双 AMX 单元，Accelerate 双线程性能大幅提升，CoreML 默认调用 AMX 但无法利用双单元。M1 仅单 AMX，CoreML 在小维度下性能更优，整体体现了 AMX 单元对 CPU 矩阵计算的核心加速作用。
:::

表 7 分析了 GEMM 的执行时间，比较了基线 C++实现和使用 Accelerate 库，与使用 Core ML 并指定 CPU 设备在 M1 和 M4 Pro 系统上达到的性能。从结果中得到的：

- 第一个观察结果源于架构：由于 **M4 Pro 拥有两个 P 核簇，因此它集成了两个 AMX 加速器**，正如一些作者所报道的那样[11]。相比之下，M1 平台仅使用一个 AMX 单元。
- 第二个关键发现是，当通过 ct.ComputeUnit.CPU_ONLY 选择 CPU 设备时，**通过 Core ML Tools 实现的 GEMM 操作默认使用 AMX 加速器**。然而，Core ML 不允许利用 M4 Pro 中可用的两个加速器。

> 聚焦于 M1 上使用 CoreML 的 GEMM 性能，如图 1 所示。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/12.webp)

::: image-caption
图 1 | Mac M1 芯片上通用矩阵乘法（GEMM）的性能表现。该图展现了 M1 芯片下 AMX、GPU、ANE 在不同浮点精度下执行 GEMM 的 GFLOPS 表现，矩阵维度 k 是核心变量。可见当 k≥1024 时，ANE（FP16）性能反超其他单元成为最优，而小维度下 AMX 加持的 CPU 更具优势，且 ANE 仅支持 4096 及以下矩阵运算，体现了其在特定规模矩阵计算中的针对性优势。
:::

- 对于大于或等于 1024 的矩阵规模，ANE 始终达到最高性能，但其使用仅限于 4096 的矩阵规模。
- 对于较小的矩阵，由于 AMX 的利用，CPU 性能更好。

> 在 M4 Pro 系统上取得的性能结果显示在图 2 中。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/13.webp)

::: image-caption
图 2 | Mac M4 Pro 芯片上通用矩阵乘法（GEMM）的性能表现。此图呈现 M4 Pro 的 AMX、GPU、ANE 执行 GEMM 的性能差异，GPU（FP16）成为全维度性能标杆，相较 M1 实现大幅提升。ANE 仅在小维度下保持竞争力，k≥4096 后性能显著下降，根源是其内部内存缓冲区的硬件限制，且 M4 Pro 的双 AMX 单元也未让 CPU 在该任务上超越 GPU。
:::

**M4 Pro 的结果与在 M1 上观察到的结果显著不同**。在此实验中：

- ANE 没有达到最高性能，尽管它在较小的矩阵规模上表现良好。:mdnice-bold-italic[GPU 成为领先设备，与 M1 相比显示出显著的改进。]
- :mdnice-bold-italic[对于 4096 及更大规模的矩阵，ANE 的性能下降]，可能是因为 ANE 内部内存缓冲区的限制。

> 关于能效，M1 的结果如图 3 所示。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/14.webp)

::: image-caption
图 3 | Mac M1 芯片上通用矩阵乘法（GEMM）的能效（千兆次浮点运算 / 瓦）。该图反映 M1 执行 GEMM 时各计算单元的能效水平，能效以 GFLOPS/W 为衡量标准。小维度下 GPU（FP16）能效最优，而测试的最大矩阵维度 4096 时，ANE（FP16）实现能效登顶，CPU 因 AMX 协处理器的加持，也展现出可观的能效表现，各单元能效随矩阵维度呈现明显的差异化变化。
:::

- 对于测试的最大矩阵规模（4096），ANE 展现出最高的能效。
- **对于较小的规模，GPU 表现出最佳能效，特别是在使用 FP16 数据时**。:mdnice-bold-italic[CPU 也显示出显著的能效优势，这在很大程度上归因于 AMX 的利用。]

> M4 Pro 的能效结果显示在图 4 中。

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/15.webp)

::: image-caption
图 4 | Mac M4 Pro 芯片上通用矩阵乘法（GEMM）的能效（千兆次浮点运算 / 瓦）。此图展示 M4 Pro 执行 GEMM 的各单元能效，ANE（FP16）在绝大多数矩阵维度下均为能效最优者，GPU 虽性能领先但能效劣势显著，其高性能以高功耗为代价。CPU 依托双 AMX 单元的硬件升级，保持了相对良好的能效，成为平衡性能与功耗的折中选择。
:::

**在大多数配置中，ANE 展现出最高的能效**。该图**也突显了 GPU 显著的性能提升是以显著更高的能耗为代价** 的。此外，CPU 保持了相对良好的能效，这可归因于 AMX 的利用。

### 5.5 Jacobi

> **图 5 和图 6 分别展示了 Jacobi 算法在 M1 和 M4 Pro 上达到的性能率。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/16.webp)

::: image-caption
图 5 | M1 上的雅可比算法性能。该图呈现 M1 执行雅可比算法的性能，因算法计算流程复杂、内存访问特性特殊，整体 GFLOPS 远低于 GEMM。其中 ANE（FP16）在网格维度 k＜1024² 时性能领先，成为小尺度网格计算的最优选择，而维度增大后，GPU（FP16）的性能优势开始显现。
:::

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/17.webp)

::: image-caption
图 6 | Mac M4 Pro 芯片上雅可比算法的性能表现。此图反映 M4 Pro 执行雅可比算法的性能差异，GPU（FP16）在全维度占据性能主导，相较 M1 的 GPU 实现跨越式提升。ANE 仍仅在小网格维度下保持竞争力，大维度下性能被 GPU 拉开差距，且各单元的性能表现均随网格维度增大呈现不同幅度的波动。
:::

> 由于 Jacobi 方法的计算流更复杂且可能存在内存访问模式问题，与计算密集度更高的 GEMM 测试相比，整体性能率显著下降。可以看出：

- 对于较小的网格规模（小于 1024），ANE 仍然是最强大的设备
- 而对于较大的问题规模，GPU 报告了更好的 GFLOPS

这种性能行为在 M1 和 M4 Pro 平台上是一致的。

> 关于能效，**如图 7 和图 8 所示，与 GEMM 相比，Jacobi 算法表现出的 ANE 效率较低。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/18.webp)

::: image-caption
图 7 | Mac M1 芯片上雅可比算法的能效（千兆次浮点运算 / 瓦）。该图展示 M1 执行雅可比算法的各单元能效，相较 GEMM，ANE 在该任务上的能效出现明显下降，各计算单元的能效差距大幅缩小。GPU（FP16）在多数维度下能效表现更优，CPU 的能效则保持稳定，整体能效水平因算法特性远低于 GEMM 任务。
:::

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/19.webp)

::: image-caption
图 8 | Mac M4 Pro 芯片上的雅可比算法能效（千兆次浮点运算 / 瓦）。此图呈现 M4 Pro 执行雅可比算法的能效，ANE（FP16）成为全维度能效最优单元，平均能效是 GPU 的两倍以上，展现出在该算法上的能效优势。GPU 虽性能领先但能效偏低，CPU 则依托硬件升级，保持了与 GPU 相近的能效水平，各单元能效随网格维度增大缓慢下降。
:::

**在 M4 Pro 上，ANE 是更高效的设备**，与 GPU 相比，其平均效率（从 20 到 40 GFLOPS/瓦特）高出两倍多。

### 5.6 Multigrid

> Multigrid 算法的性能结果遵循与 Jacobi 平滑器观察到的总体趋势。**图 9 和图 10 显示了 Multigrid 在多个问题规模下的平均执行时间。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/20.webp)

::: image-caption
图 9 | Mac M1 芯片上多重网格算法的执行时间（秒）。该图反映 M1 执行多重网格算法的耗时情况，算法因 V 循环架构、多网格层级求解的复杂性，耗时远高于雅可比算法。GPU（FP16）在全维度耗时最少，ANE 仅在小网格维度下耗时与 GPU 接近，大维度下耗时急剧增加，性能劣势随维度增大愈发明显。
:::

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/21.webp)

::: image-caption
图 10 | Mac M4 Pro 上的多重网格算法耗时（秒）。此图展示 M4 Pro 执行多重网格算法的耗时，GPU（FP16）依旧是全维度最快的计算单元，相较 M1 的 GPU 耗时大幅降低，硬件升级效果显著。ANE 仅在小维度下保持一定竞争力，大维度下因内存限制耗时骤增，甚至无法支撑超大规模网格的运算。
:::

**考虑到此方法的更高复杂性，它涉及到求解一个网格大小的层级结构，GPU 是最快的设备，与问题大小无关**。与之前的分析一样，ANE 在较小的问题规模上具有竞争力，但随着网格大小的增加，ANE 的执行时间显著增加，导致其效率下降。

> 从能耗方面来看，ANE 展现了其优势，尽管 GPU 仍然表现出卓越的性能。**图 11 和图 12 分别显示了 M1 和 M4 Pro 上的平均功耗。**

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/22.webp)

::: image-caption
图 11 | Mac M1 芯片上多重网格算法的平均能耗（焦耳）。该图呈现 M1 执行多重网格算法的能耗表现，GPU（FP16）虽耗时最少，但能耗显著高于其他单元，是典型的高性能高功耗特征。ANE 在小网格维度下能耗最低，相较 CPU 和 GPU 可降低 15%-50%，大维度下能耗随耗时增加同步上升，能耗优势逐步消失。
:::

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/23.webp)

::: image-caption
图 12 | Mac M4 Pro 芯片上多重网格算法的平均能耗（焦耳）。此图反映 M4 Pro 执行多重网格算法的能耗，ANE 在网格维度＜4096² 时能耗最优，4096² 维度下仅消耗 0.5 焦耳，远低于 CPU 的 13.47 焦耳和 GPU 的 1.49 焦耳。大维度下 ANE 因内存限制无法执行，GPU 成为该场景下的能效最优选择，CPU 则始终保持高能耗特征。
:::

再次，**强大的 GPU 以高能耗为代价实现了更好的性能率**。

- 对于小问题规模，ANE 表现出更低的以焦耳为单位的能耗，可以比其他设备低 15%到 50%。例如，对于 4096² 的网格规模，M4 Pro 上的 ANE 消耗 0.5 焦耳，而 CPU 为 13.47 焦耳，GPU 为 1.49 焦耳。
- 对于更大的网格规模，ANE 的内存限制阻止其执行，使得 GPU 成为最能效的设备。

### 5.7 讨论

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/24.webp)

::: image-caption
表 8 | 以 M1 芯片 CPU 的 FP16 精度执行性能为基准的加速比。该表量化了各单元相对 M1 CPU FP16 的加速能力，M4 Pro 的 GPU 加速比提升最显著，达 20 倍以上，CPU 也有 3-4.5 倍提升。ANE 虽绝对加速比最高，但代际提升仅 1.5-2.5 倍，远低于 CPU 和 GPU，印证了 M4 Pro 大幅缩小了通用计算单元与 ANE 的性能差距。
:::

> 表 8 中呈现的结果以 M1 CPU 的执行时间为基线，使我们能够量化每个处理单元在 M1 和 M4 Pro 芯片之间的代际性能提升。这种比较揭示了一个显著的扩展改进差异：**虽然 CPU 和 GPU 经历了显著的加速，但 ANE 的改进幅度明显较小。**

通用单元（CPU 和 GPU）表现出了强劲的扩展能力：

- GPU 性能提升了 3 倍到 5 倍。例如，在 Multigrid (8192)基准测试中，相对于 M1 CPU 的加速比从 M1 GPU 的 6.57 倍增加到 M4 Pro GPU 的 25.75 倍，代表代际改进因子为 3.92 倍。
- CPU 性能也经历了相当大的加速，达到了 4.5 倍。Multigrid (12288)基准测试中，M4 Pro CPU 相对于 M1 CPU 显示了 4.54 倍的加速。

相比之下，尽管 ANE 在绝对项上通常仍然是最快的单元，但它表现出了最低的代际改进因子：

- 在 AI 任务 YOLOv11 中，加速比从 M1 的 19.36 倍增加到 M4 Pro 的 28.96 倍。这转化为仅 1.50 倍的代际改进因子。
- 在 Jacobi (8192)中，ANE 的加速比从 M1 的 3.79 倍移动到 M4 Pro 的 7.34 倍，产生 1.94 倍的改进因子。
- ANE 的最高因子在 Multigrid (4096)中观察到，改进为 2.41 倍。

::: block-1
总之，虽然 CPU 和 GPU 的性能提升范围在 3 倍到 4.5 倍之间，但 ANE 的改进仅限于 1.5 倍到 2.5 倍的范围内。**这种不均衡的扩展表明，M4 Pro 显著缩小了 ANE 在 M1 世代相对于 CPU 和 GPU 所拥有的性能差距。**
:::

## 六、结论

> 本研究评估了苹果硅芯片 M1 和 M4 Pro 架构在 AI 和通用计算工作负载组合中的性能和能效。:mdnice-bold-italic[研究结果证实了 ANE 在加速机器学习任务方面的有效性，使其成为推理领域的领先设备，并且在这些工作负载中在功耗方面最为高效。]

本研究还分析了通用工作负载在苹果硅芯片上的行为。

- 一方面，**ANE 在 GEMM 中取得了强劲的性能，在 M1 芯片上达到了 2.5 TFLOPS，在 M4 Pro 上达到了 3.8 TFLOPS**。这一性能令人印象深刻，因为:mdnice-bold-italic[它达到了 GPU 峰值性能的 80%，同时在 M4 系统上比 GPU 少用了 80%的功耗。]
- 另一方面，**ANE 在更复杂的通用工作负载如 Jacobi 上表现出卓越的能效，达到了令人印象深刻的 49.8 GFlops/W 的效率**。然而，重要的是要注意 :mdnice-bold-italic[ANE 存在显著的内存限制，对于大数据量会导致运行时错误。]

::: block-1
聚焦于代际演进，观察到设备间存在不均衡的性能提升：虽然 CPU 和 GPU 经历了显著的加速，但 ANE 的改进幅度明显较小。**这暗示了由于 CPU 和 GPU 基线性能的实质性提升，性能差距正在缩小**。最终，ANE 被证明是一个有能力且高效的设备，适用于通用算法，特别是那些涉及小数据处理的算法。
:::

作为未来工作，我们建议将 Multigrid 方法在多设备异构环境中进行协同划分：

- 将最细网格（计算密集、数据量小）交由 Apple Neural Engine（ANE）高效执行，
- 而较粗网格（数据量大、需高带宽内存访问）则由 GPU 主导处理。

**这一划分策略源于实证分析——约 50% 的总执行时间消耗在最细网格层级，而 ANE 在小规模张量运算和低精度推理任务中展现出显著能效优势**；相比之下，:mdnice-bold-italic[GPU 更擅长处理高吞吐、大内存带宽需求的粗网格迭代。]

通过在运行时依据网格层级、数据尺寸与设备算力特征动态调度任务，可实现负载均衡与端到端加速。该方案不仅适配 Apple Silicon 平台的硬件拓扑，也为其他异构加速架构（如 GPU + NPU）提供了可迁移的算法划分范式。

## 七、参考文献

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/25.webp)

![文章插图](/articles/2026-02-14-yolov11-gemm-apple-silicon-m1-cb1fa47a/images/26.webp)

::: related-title
相关推荐
:::

::: related-links
- [Apple Silicon LLM 推理全方位深度剖析：与 NVIDIA GPU 从性能到成本的终极对决！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890883&idx=1&sn=d05b1dbc8df195d7a60dfc6d2bee4d6f&scene=21#wechat_redirect)
- [生产级 Apple Silicon LLM 推理框架深入分析对比：MLX, MLC-LLM, Ollama, llama.cpp](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894128&idx=1&sn=1af2f122be249b52eb4a764227b28c83&scene=21#wechat_redirect)
- [苹果 M5 芯片发布，AI 性能狂飙 4 倍！GPU 架构重构 + 16 核神经引擎 + 10 核 CPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893129&idx=2&sn=3a2db8a1d0c83334ff424c4b7cfea9c5&scene=21#wechat_redirect)

:::
