> 对神经网络高效处理的需求推动了硬件加速器的开发。专用硬件的广泛采用突显了对硬件-软件协同设计以及特定领域优化的更敏捷设计流程的需求。在本文中，我们介绍了 CFU Playground——一个用于在 FPGA 上为嵌入式机器学习（TinyML）加速设计和评估机器学习（ML）加速器的全栈开源框架。

我们的工具为用户提供更开放的端到端流程，以便**在 FPGA 和未来系统研究中进行硬件-软件协同设计**。该全栈框架使**用户能够访问实验性和定制的架构**，这些架构是**为嵌入式 ML 定制和协同优化的**。我们快速的部署-配置文件-**优化反馈循环使 ML 硬件和软件开发人员能够以相对较小的定制投资获得显著的回报**。

使用 CFU Playground 的设计和评估循环，我们展示了 **55 倍到 75 倍的速度提升**。软 CPU 与加速器的结合为两个组件之间开辟了一个新的丰富的设计空间，我们**通过 Google Vizier（一个开源的黑盒优化服务）以自动化的方式进行探索优化参数**。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/01.webp)

- 文章：https://arxiv.org/abs/2201.01863
- 组织：Harvard University、Google、Purdue University、Stanford University

## 本文目录

- 一、引言
- 二、背景和动机
  - A. 微型机器学习（TinyML）
  - B. 对敏捷和全栈研究框架的需求
  - C. 在 FPGA 上进行设计空间探索
- 三、CFU Playground 概览
  - A. 定制功能单元（CFU）
  - B. 门级逻辑
  - C. 硬件
  - D. 软件
  - E. 测试、模拟和基准测试
  - F. 设计空间探索
- 四、评估
  - A. 开发生命周期
  - B. 案例研究
  - C. CPU 与 CFU 的自动化设计空间探索
- 五、先前相关工作
- 六、结论与未来工作
- 参考文献

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/02.webp)

- https://google.github.io/CFU-Playground/workshops/hpca-2023#overview

## 一、引言

> **微型机器学习** （TinyML）是 ML 算法和低成本嵌入式系统交叉领域的一个快速发展的新领域。它能够**在超低功耗下实现设备上的传感器数据分析** （如视觉、音频、IMU 等）。在传感器附近处理数据可以扩展出新的各种始终在线的 ML 使用案例，这些案例在保持带宽、延迟和能源的同时提高了响应能力并维护了隐私[1]。

鉴于在这些**嵌入式平台上运行 ML 所需的能源效率**，**定制处理器支持和此类系统的硬件加速器可能提供所需的解决方案**。然而，ML 领域仍处于**起步阶段并且变化迅速**。因此，对于低成本嵌入式 ML 系统来说，提前投入巨大的非经常性工程（NRE）成本是不受欢迎的，尤其是。构建 ASIC 既昂贵又耗时。

此外，**由于嵌入式系统通常是特定任务的**，因此有机会**避免通用 ML 加速器**，而是**探索特定任务和模型的 ML 加速方法**。这种设置提出了对**敏捷设计空间探索工具的需求**，以使我们能够适应 ML 和硬件不断变化的格局。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/03.webp)

为了实现**对特定领域性能优化的全面硬件-软件协同设计和评估**，我们引入 CFU Playground。它是**一个用于以敏捷方式迭代（部署 → 配置文件 → 优化）探索轻量级加速器设计空间的全栈开源框架** （图 1）。该框架的独特之处在于，它**将各种开源软件** （TensorFlow Lite Micro/TFLM、GCC）、**开源 RTL 生成 IP 和工具包** （LiteX、VexRiscv、Migen、Amaranth）**以及开源 FPGA 工具** （用于综合、布局和布线的 yosys、nextpnr、F4PGA/SymbiFlow 等）**结合在一起**。

通过使用**整个堆栈的开源**，我们**使最终用户能够定制和协同优化硬件和软件**，从而产生**不受潜在许可限制约束的专用解决方案**，并且**不特定于某个 FPGA、板卡或供应商**。CFU Playground 以相对较小的定制硬件投资带来了巨大的回报，适用于低容量应用的长尾领域。**CFU Playground 的另一个新奇之处在于其设计定制功能单元（CFUs）的能力**，用于不同的 ML 操作。**CFUs 代表了一个新的设计空间，平衡了加速与灵活性，并减少了与分立加速器相关的开销**。

我们通过硬件在**环评估** 过程展示的**全栈解决方案开箱即用**。它还考虑了在**整个堆栈中可能产生的端到端瓶颈（软件开销、预处理等）**，但这些在**单独设计时常常被忽略**。从一个初始的、非定制的解决方案开始，用户可以逐步专门化个别（硬件或软件）组件以提高其应用的性能。由于 CFUs 的轻量级特性，可以快速进行开发和更改，因为针对嵌入式 ML 平台的编译和部署非常迅速。

我们使用该框架展示了如何设计 CFUs，以扩展基于 FPGA 的 RISC-V 核心，并且完全集成到其余的系统软件堆栈中。CFUs 适用于 ML 推理的主要原因是，通常存在几个小但关键的热点。利用 FPGA 的位级灵活性的小型定制硬件可以帮助加速大部分执行时间。紧密结合的 CFU 允许我们将复杂性、设置和外层循环保留在软件中，同时高效地解决数据通路中的核心计算瓶颈。

此外，正如我们所展示的，**CFUs 允许我们逐步扩展单元，直到它几乎变成一个完整的 ML 加速器**。通过我们的敏捷设计流程，我们展示了如何通过循环展开、SIMD 乘积累加和流水线等优化组合，**将 MobileNetV2 的卷积操作加速 55 倍。这仅用了五周时间，由一位兼职的高级工程师完成。**

::: block-1
总而言之，我们的贡献如下：

- 一个**开箱即用的全栈框架**，该框架在整个堆栈中**完全集成了开源工具**，以促进丰富社区驱动的生态系统研究和开发。
- 一种**敏捷** 方法，用于**迭代设计和评估紧密耦合的定制加速器**，适用于资源受限、延迟受限的 TinyML 应用程序。
- 我们展示了 **CFU、CPU 和内存系统之间新颖的模型特定资源分配权衡**，这些权衡使得在资源受限的 FPGA 平台上实现最佳的机器学习性能成为可能，适用于两种重要的用例。
- 我们将 **CFU Playground 与 Google 的开源黑盒优化器 Vizier （https://github.com/google/vizier）捆绑** 在一起，以**实现 CPU 与 CFU 配对的自动化设计空间探索**。
:::

## 二、背景和动机

### A. 微型机器学习（TinyML）

> 微型机器学习（TinyML）是将机器学习（ML）算法部署到低成本、低功耗且资源受限的微控制器（MCU）系统上的技术。TinyML 能够实现设备上的机器学习，而且仅需传统 ML 系统所需计算资源的一小部分。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/04.webp)

**表 I 对 TinyML 与传统的大型 ML（例如云端和移动系统）进行了比较**，展示了 **TinyML 在计算能力、内存、存储、功耗和成本方面的数量级差异**。设备上运行的 ML 模型可用于智能的、设备上的传感器分析，从而实现各种始终在线的 ML 应用场景。**尽管 TinyML 有许多优势，但 MCU 硬件的异构性和有限的资源也带来了新的挑战**。

MCU 通常只有十到几百 KB 的 SRAM 和一到两 MB 的 Flash 存储。这严重限制了设备上 ML 模型的大小。计算能力、内存和存储往往是紧密集成的，限制了根据各种需求灵活调整资源的能力。尽管这些系统已经能够完成诸如关键词识别、视觉唤醒词和异常检测等基本 ML 任务，但为了在保持低功耗运行点的同时支持更先进的应用场景，需要专门的硬件支持。

### B. 对敏捷和全栈研究框架的需求

> TinyML 带来了许多挑战，我们认为通过开源生态系统能够解决这些问题，从而开展系统研究和开发。**在微控制器类硬件上进行 ML 加速是一个新兴领域**，因此，**TinyML 部署工具和运行时** 环境**通常是特定供应商的**，这导致了**跨硬件平台的软件碎片化**。

此外，许多 TinyML 工具并非公开可用。这种缺乏访问权限的情况使得隔离和比较加速器性能变得困难。此外，**硬件定制是一个迭代过程**，特别是对于 ML 开发来说，**算法和模型** 本身通常会**经历反复的改进**。

::: block-1
例如，在 **ML 团队和处理器实现团队之间经常会有这样的交流** ——“如果我们把**这个层量化为 4 位会怎么样？** ”——以实现一个既符合性能目标又满足嵌入式系统的资源限制的解决方案。
:::

因此，对于**敏捷设计** 空间探索工具的需求是明确且迫切的，尽管先前已经有一些工作（我们在后文第五部分会对此再详细讨论）。**虽然模拟器是有用的** 并且对于功能正确性有帮助，**但获得准确的性能估计是困难的**。**准确建模系统的所有特性** （例如多个时钟、异步接口、不同带宽和延迟的内存混合等）**可能导致模拟速度变慢**。在进行硬件在环运行时，用户知道所收集的测量数据不会忽略任何关键的性能因素。

此外，如果成功加速了计算，系统通常会受到内存或 I/O 的限制。因此，从全栈系统视角对 ML 加速器的性能进行建模和评估也是重要的。使用**模拟器对总线延迟** （包括争用和仲裁）**进行建模是不精确的**。**全系统的周期精确模拟** 对于敏捷设计来说**太慢了**。

### C. 在 FPGA 上进行设计空间探索

> FPGA 为定制微架构的复杂设计空间探索打开了大门，这些微架构超越了 ML 加速器本身，还包括 CPU。**利用 FPGA 平台，可以定制处理器** 以高效地执行应用程序的计算。ML 计算往往是规律且重复的，同一操作序列会重复执行数百万次。

我们可以**专注于改进这些“热点”计算** 的执行，同时**使用标准指令来执行其余操作**。利用 FPGA 的位级灵活性，为这些热点设计的小型定制硬件可以带来巨大的改进。

**FPGA 平台还允许将加速单元紧密耦合到 CPU 流水线中**，通过**添加新的定制指令来补充 CPU 的标准功能**，从而轻松调用。这与将 ML 加速器视为独立处理单元的做法不同，后者会遭受“AI 税” [4]，[5]。除了易于编程外，**紧密耦合对于 TinyML 所需的低延迟应用程序也是有益的** [6]。此外，FPGA 的位级灵活性允许高效地实现小数据尺寸和非标准数据表示的操作，甚至包括数据类型的打包、解包和转换。

**FPGA 平台的另一个好处是**，除了计算效率外，在嵌入式 ML 应用程序中，**安全性** 往往至关重要，因为它们始终处于开启状态。软 CPU 提供了透明性，因此可以确切地知道其中的内容，而无需盲目信任芯片的供应链。另一个针对供应链不确定性的保护措施是，如果原始 FPGA 目标变得不可用，由于您的“处理器”，包括 CPU 和定制功能单元，只是可以部署在任何具有足够资源的 FPGA 上的 Verilog 代码，因此只需最少的工作就可以切换到另一个设备或供应商。

## 三、CFU Playground 概览

> 对于实验性 TinyML 和系统研究来说，像 CFU Playground 这样易于使用且敏捷的工具至关重要，它结合了一系列门级逻辑、软件和硬件组件，如图 1 所示。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/03.webp)

在本节中，我们将介绍框架的设计以及这些主要组件的详细信息。

### A. 定制功能单元（CFU）

> CFU 是添加到硬件中的小型定制逻辑，用于扩展 CPU 的数据通路，以加速由开发人员确定的离散函数。**定制功能单元（CFU）可以通过多种方式提升程序热点的执行效率**。CFU 可以对常量操作数进行专有操作，在每个周期内通过并行计算独立操作来执行多个操作，或在一个周期内融合连续操作，或在多个周期内流水线化计算。

灵活可配置的存储也允许数据在本地存储和复用，从而减少不必要的数据传输。用户还**可以迭代添加和执行这些操作的组合，以逐步提升性能**。**CFU 遵循 RISC-V 的 R 格式**，从寄存器文件接收两个操作数并写回一个结果。**图 2 展示了 CFU 与 CPU 的其他部分的关系**。CPU 和 CFU 之间的边界是逻辑上的。当前实现将设计展平，并一起进行优化、布局和布线。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/05.webp)

### B. 门级逻辑

> CFU Playground 将 CFU 集成到基于 FPGA 的系统芯片（SoC）中，以捕捉加速 ML 模型的全栈系统效应。**见图 3，其门级逻辑建立在 LiteX 框架 [8] 之上**。LiteX 提供了一个方便高效的基础设施，用于创建 FPGA 软核和 SoC。

对于要在 CFU Playground 中使用的任何板卡，必须首先有一个 LiteX 描述。大多数流行的商业可用板卡已经在众包的 LiteX 板卡库中有了描述。对于私人原型板卡，可以在本地创建并维护描述。

**CFU Playground 用户可以访问 LiteX 提供的全部选项，用于定制 SoC**。例如，在支持 L2 缓存的板卡上，只需在项目 Makefile 中添加以下一行，即可更改 L2 缓存大小：

```
export EXTRA_LITEX_ARGS='--l2-size=16384'
```

要使该更改适用于使用该板卡的所有项目，在 Python 中添加一行即可：

```
args.l2_size = 16384
```

**CFU Playground 中使用的软核是 VexRiscv**，它是用 SpinalHDL [9] 实现的 RISC-V CPU 的开源版本。VexRiscv 在软 CPU 竞赛中获胜，并且针对 FPGA 进行了优化。VexRiscv 的设计高度可配置，可以轻松地添加或移除许多不同的功能，如流水线阶段、缓存和浮点单元。**这种定制能力非常适合探索 CPU 与 CFU 的设计空间**。目前，VexRiscv 是唯一支持特定 CFU 接口的软 CPU。未来的工作将把 CFU 接口添加到其他 RISC-V CPU 上。不过，这仍然为开发人员提供了广泛的 VexRiscv 设计选择。

### C. 硬件

> CFU Playground 的门级逻辑可以适应多种 FPGA。**表 II 列出了 CFU Playground 测试和支持的几种嵌入式 FPGA 板卡**。其门级逻辑可以适应小至 Fomu [10] 的板卡，用于 TinyML 原型设计，Fomu 体积为 1 cm²，可放入 USB 接口中。然而，**当资源更丰富时，可以构建更强大的软 CPU 和 CFU，并且随着内存的增加，可以运行更大的模型**。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/06.webp)

因此，该系统本质上是可扩展的。对于评估硬件的任何资源的最低要求都很难明确，因为它们在一定程度上是可互换的。例如，CPU 可以完全用查找表（LUT）构建，但如果可以使用块 RAM 作为其寄存器文件，并用 DSP 块作为其乘法器，则所需的 LUT 将大大减少。

很难对评估硬件的任何资源的最低要求给出确切的定义，因为它们在某种程度上可以相互替代。例如，CPU 可以完全用 LUT（查找表）构建，但如果能用块 RAM 作为其寄存器文件，并用 DSP（数字信号处理）块作为其乘法器，那么所需的 LUT 数量就会少得多。

CFU Playground 目前支持 Xilinx 7 系列以及 Lattice iCE40、ECP5 和 CrossLink FPGA。带有 iCE40UP5k FPGA 的 Fomu 是接近最小可用板卡的示例。它具有 5280 个逻辑单元、128 千字节的大容量片上 RAM、30 个 512 字节的块 RAM 和 8 个 16 位 × 16 位的 DSP/乘法器块，并且可以放入 USB 插槽中。CFU 的 Verilog 代码可以通过开源工具链（例如 F4PGA/SymbiFlow [11]）或供应商提供的 FPGA 工具链编译为位流。我们注意到，CFU 可以用用户选择的任何可以生成 Verilog 的硬件描述语言（例如 Chisel、Bluespec、Amaranth 等）编写。

### D. 软件

> 为了调用 CFU 硬件，必须在 CPU 的指令集中添加自定义指令。然而，编译器的职责并不是为指令寻找用途，而只是在用户请求时生成这些指令。这与在 TensorFlow Lite for Microcontrollers（例如 CMSIS-NN [12]）中的每个硬件优化内核中使用的流程相同，并且我们重视它，因为它有利于未来丰富开源生态系统的研发。

因此，我们可以采用现成的 RISC-V GCC 工具链，并结合我们提供的宏，该宏扩展为 “asm” 指令以在必要位置生成编码后的自定义指令。该宏接受 4 个参数，并返回一个结果：

```
q = cfu_op(funct7, funct3, a, b);
```

该宏直接生成 32 位编码值，因此甚至不需要修改汇编器。“funct7” 和 “funct3” 分别是 7 位和 3 位字段，用于指定自定义指令的操作码。它们必须是编译时常量表达式。“a” 和 “b” 是用作指令操作数的 C/C++ 32 位整数变量，返回一个 32 位结果。该宏从编译器获取有关操作数（a、b 和 q 在此示例中）的寄存器位置的信息，以生成正确的寄存器源和目的位字段，因此不需要额外的寄存器到寄存器复制。为了提高可读性，用户可以定义一个宏，用于每个自定义指令，基于基础 cfu_op() 宏，如下所示：

```
#define single_popcount(a) cfu_op(1,1,(a),0)#define bit_reverse(a) cfu_op(1,2,(a),0)#define simd_add(a, b) cfu_op(1,3,(a),(b))
```

与任何内联汇编指令一样，这些自定义指令宏可以与常规 C 代码混合使用。

当反汇编时，工具链对自定义指令一无所知，甚至不知道哪些位指定寄存器，因此它只是将 32 位写为十六进制值。CFU Playground 将十六进制转换为可读的汇编代码，包括源寄存器和目的寄存器。

以下是一个自定义指令宏与常规 C 代码混合使用的示例：

```
int *x; int b; ...t1 = *x;t2 = cfu_op(0, 0, t1, b);t3 = cfu_op(1, 0, t2, b);*x = t3;
```

以下是上述示例编译并反汇编后的结果，展示了自定义指令如何在没有额外开销的情况下使用：

```
400001a0: 00812783 lw a5,8(sp)400001a4: 00d7878b cfu[0,0] a5, a5, a3400001a8: 00d7978b cfu[1,0] a5, a5, a3400001ac: 00f12423 sw a5,8(sp)
```

调用自定义操作是用户的职责。自定义指令宏可以与常规 C 代码混合使用，类似于任何其他 C/C++ 操作。TensorFlow Lite for Microcontrollers（TFLite Micro/TFLM）[13] 是 CFU Playground 用于部署神经网络的推理框架。用户必须提供一个优化内核，使用新的自定义指令来实现运行时性能提升。

由于 CFU Playground 和 TFLite Micro 使用源文件覆盖机制，因此无需修改推理框架。只需在一个单独的文件中实现优化内核，构建过程就会拾取新实现。通常，只需要针对一两个 TFLite 操作类型进行加速，因此只需要修改和优化实现这些操作类型的内核。此外，TFLite Micro 提供了灵活的部署方式，与特定供应商的框架相比，其内存和延迟开销相对较小。因此，如果用户创建了新的指令来加速 CONV_2D 操作类型，他们将开始复制 TFLite 库文件到当前项目目录下的位置：

```
./src/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h
```

然后他们将修改它以使用自定义指令。

TFLite 库内核是通用的，可以处理任何合法的 TF 操作参数化，以及任何合法的输入或输出张量形状。但是，用户可以选择仅用其修改后的优化内核处理特定的参数化。在这种情况下，修改后的内核版本通常在入口处包含一个检查；如果参数超出支持的范围，则将执行委托给通用内核。在部署时，如果模型仅包含受支持的参数化，则可以移除通用版本。

### E. 测试、模拟和基准测试

> 菜单驱动的软件包含来自 TFLite Micro 库的内核级单元测试。它还包含具有设定输入和预期输出的全推理黄金测试，用于每个提供的模型。可以根据需要添加其他模型，我们将在评估部分（第四部分）中进行演示。

为了支持调试和测试，用户可以使用高级 C 编程语言编写软件仿真，以功能等效的方式模拟他们的 CFU，但速度当然要慢得多，可以替换为真实的 CFU。软件仿真函数有多种用途。例如，针对 FPGA 板的随机或定向的 CFU 级单元测试可以将相同的输入序列馈送到真实的 CFU 和软件仿真中，并期望看到相同的输出序列以测试正确性。当使用真实 CFU 的执行似乎不正确时，可以用软件仿真替换 CFU，以确定问题出在使用自定义指令的代码中还是用户定义的 CFU 的实现中。

此外，运行在 FPGA 板上的应用程序可以访问 `printf()` 和终端输出，因此在执行期间打印变量值可以作为另一种调试方法。一些板支持调试桥，以连接调试器到运行中的系统。在这种情况下，必须使用启用了调试功能的 VexRiscv 变体，而这样的变体是现成可用的。

CFU Playground 还包含用于运行实际自定义指令的简单测试，与软件仿真进行对比，具有随机、穷举或定向的输入。这些测试是用 C 编写的，在板的 SoC 上运行。对于使用 Amaranth HDL（以前称为 nMigen [14]）编写的复杂 CFU，提供了一个基于 Python 的开放源码 RTL 设计工具包的单元测试设施。对于 Amaranth 的单元测试，以及下面描述的其他测试方法，可能不需要 Verilog 级别的仿真测试平台，但用户当然可以创建自己的测试平台。

**CFU Playground 还支持 Renode 模拟器** [15]，该模拟器**可以模拟物理硬件系统**。虽然 CFU Playground 建议遵循硬件在环流程并在物理板上运行，但 Renode 模拟器可以用于在没有这些板的情况下测试其他板上的 CFU。Renode 执行 CPU 的 ISA 模拟，结合 CFU 的周期精确 Verilog 模拟，从而测试 CFU 实现的功能正确性。它还模拟 RAM、ROM 和 UART。Renode 模拟器还**允许我们捕获来自 CFU 操作的波形**，这对于追踪用户定义的 CFU 的硬件设计中的错误非常有用。

此外，为了推动该领域的发展，人们已经努力对这些极度资源受限的 TinyML 系统进行基准测试，以推动行业标准化和创新 [3]。CFU Playground 配套提供了 MLPerf Tiny [2] 深度学习工作负载，用于基准测试目的。

在我们的敏捷设计流程中，有两个有趣的用例可以帮助支持这一基准测试生态系统和研究社区。第一种情况是当用户有一个特定的想法来加速现有的模型时。能够在构建 ASIC 加速器之前快速原型化和基准测试这一想法，可以节省构建 ASIC 加速器所需的昂贵和耗时的过程，从而实现对新硬件架构的探索。第二种用例涉及解决硬件彩票问题 [16]，并可以促进对新模型架构的探索。

**CNN 能够迅速发展的一个原因是它们可以方便地映射到现有硬件** （例如 GPU）。然而，如果用户有一个适用于 TinyML 的新模型架构的想法，而该架构不适合现有硬件，**CFU Playground 可以用来快速启动硬件，以便在与已经具有成熟硬件支持的现有模型架构进行公平基准测试之前，对新模型架构进行测试**。这可以帮助证明在适当的硬件支持下，他们的模型是高效的。

### F. 设计空间探索

> 我们为 CFU Playground 配套了 Google 内部 Vizier 框架的开源版本 Vizier [17]，这是一个基于黑盒优化服务的框架。通过此服务，我们的框架能够对 CPU 与 CFU 的组合进行自动设计空间探索（DSE）。

DSE 参数可以包括分支预测器类型（静态、动态、动态目标）、定制功能单元（SIMD、MAC 等）、I- 和 Dcache 大小、乘法器、除法器、移位器等。用户可以轻松地通过 Python 将这些参数提供给 Vizier 的搜索空间，使用 API：

```
problem.search_space.select_root() .add_categorical_param( name='prediction', feasible_values=['none', 'static', 'dynamic', 'dynamic_target'] )
```

Vizier 的服务根据用户希望优化的内容（例如资源或延迟）返回不同的配置进行探索。CFU Playground 中使用的开源综合工具 Yosys [18] 计算 CPU-CFU 设计所需的 FPGA 逻辑单元资源，这些资源随后传递给 Vizier。同样，框架中包含的周期精确模拟器 Verilator [19] 用于确定 Vizier 在云端大规模运行实验时的工作负载延迟。延迟也可以通过我们的硬件在环功能直接在板上测量。

CFU Playground 的 DSE 功能允许用户根据其工作负载调整 CPU，并分析 CFU 和 CPU 组件之间的权衡，这在资源受限环境中极为宝贵。Vizier 的系统搜索对于在 CFU Playground 启用的庞大而多样的设计空间中，在合理的时间内进行探索是至关重要的。我们提供了一个 vizier_dse.py 脚本，可以开箱即用，用于运行此设计空间探索循环（图 1）。

## 四、评估

> 在本节中，我们首先介绍 CFU Playground 中使用的设计和评估方法（第 IV-A 节）。在概述了在实验设计中促进指导性优化的方法后，我们通过两个案例研究来评估框架的易用性（第 IV-B 节）。

最后，使用案例研究中的加速器设计，我们使用自动化评估 CPU 和 CFU 之间的权衡，并突出 CFU Playground 所启用的丰富设计空间（第 IV-C 节）。

### A. 开发生命周期

> CFU Playground 启用了一个部署 → 配置文件 → 优化的设计和评估循环，从而实现对资源受限 ML 系统的迭代、指导性优化。该工具链的构建使得开发人员可以在堆栈的任何一层轻松运行他们的设计，在非常细粒度的级别上测量其性能，实现自定义优化，并重复这一过程。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/07.webp)

图 4 展示了这一评估工作流程，并突出了在部署堆栈的每一层中 CFU Playground 的各个组件所扮演的角色。整个堆栈跨越了我们所描述的软件、门级逻辑和硬件组件。

#### 软件优化

CFU Playground 使开发人员能够配置文件模型以了解基线性能。例如，TFLM 的内置配置文件可以显示每个 TFLM 操作（即 CONV_2D、DEPTHWISE_CONV 等）的名称和运行时间。然后可以使用周期计数器来配置文件 C 代码的各个部分，以识别热点。

基于配置文件信息，开发人员可以创建一个自定义的 TFLM 操作，例如缓存频繁使用的变量和展开循环，通过软件优化就可以相对快速地实现加速。

#### 硬件加速

在简单的软件优化之后，开发人员可以利用资源监视器识别硬件瓶颈，从而找到 CFU 加速的机会。例如，这样的优化可能包括将循环不变变量移入 CFU、制作 SIMD 指令版本以及制作相应的自定义指令。

#### 自动化协同设计

在开发人员完成软件和硬件优化后，仍有进一步专门化的空间，通过探索 CPU + CFU 配置的设计空间。这允许在可配置的处理器和加速器之间进行协同设计，以进一步将用户的最终设计定制到其特定任务。为了帮助评估软 CPU + CFU 配置之间的资源-性能权衡，开发人员可以使用框架的设计空间探索基础设施，自动调整其嵌入式 ML 系统以满足约束条件。

### B. 案例研究

为了展示 CFU Playground 在实际应用中对最终用户和实验设计的实用性，我们通过两个常见的 TinyML 应用案例展示了迭代开发方法：图像分类（IC）和关键词识别（KWS）。IC 示例展示了 CFU Playground 如何轻松启用硬件-软件的迭代改进。KWS 示例展示了如何在资源极度受限的环境中共同优化 CPU 和 CFU。我们重点关注性能优化，但这些方法也可以扩展到与能源相关的优化。所有这些示例和基础设施都是公开可用的。以下部分提到的两个基线是在基本的 VexRiscv 配置变体上获得的（即，没有 CFU 加速）。

#### 1. 在 Arty 上加速图像分类

我们介绍 CFU Playground 如何轻松实现部署 → 配置文件 → 优化，并将最耗时的 TFLite 操作的速度提升 55 倍，将操作时间从每次推理的 5.5 秒缩短到 0.10 秒。我们从软件优化开始，然后转向硬件，展示了我们的工具轻松启用的协同设计。目标是加速主导运行时间的操作类型，使其变得微不足道，以展示我们迭代方法的用途，理论上也可以应用于所有其他操作。

**部署**：我们针对 MobileNetV2（MNV2）模型，这是一个高效的图像分类模型，并在 Arty A7-35T 板上优化其性能，该板配备 Xilinx XC7A35T FPGA 和 256 MB 外部 DDR3 内存。该模型被量化为 8 位整数（int8），这是我们使用的 TFLite Micro 推理框架所支持的。

**配置文件**：在 Arty 板上对 MNV2 的配置文件显示，未加速的基线应用程序大约消耗 900 M 周期。其执行时间的约 95% 分布在三种不同类型的卷积上：1x1 2D 卷积（63%）、深度卷积（22.5%）和 3x3 2D 卷积（11%）。由于大部分周期被 1x1 CONV_2D 层消耗，我们的主要焦点是加速这个特定的操作符。

在 1x1 卷积的计算中，对于每个 x、y 空间坐标，输入向量（输入张量在 x、y 处的列，长度由输入通道数确定）乘以一个矩阵以产生输出向量（输出张量在 x、y 处的列，长度等于输出通道数）。该矩阵包含滤波器权重，大小为输入通道 × 输出通道，并且对每个 x、y 都相同。对每个输出元素应用后处理（应用偏置、激活函数和重新量化）。

**优化**：图 5 展示了我们通过每次优化迭代的速度提升和资源使用情况。我们从软件优化开始，以尽量减少硬件更改，直到需要时才转向 CFU 硬件支持。CFU 使用 Amaranth HDL 实现。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/08.webp)

##### 软件优化和专门化

我们首先为 1x1 情况创建了一个新的 CONV_2D 内核，即当 filter_width 和 filter_height 都等于 1 时。在通用内核中添加了一个检查，当这种情况发生时分支到专门内核。

然后，在专门内核中，可以假设 filter_width 和 filter_height 为 1，并且可以删除两层循环以及其他对这些参数的使用替换为常量 1。例如，还**可以移除 padding 超界检查。这些专门化以及其他优化（如循环展开）将执行时间减少了 49%，提供了接近 2 倍的速度提升** （图 5 中的 SW）。

##### CFU 优化

我们的第一个自定义指令加速了每个输出元素的后处理。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/09.webp)

**图 6 展示了 CFU 的微架构。每个输出通道的偏置、乘数和移位量值存储在 CFU 中。** 这大约节省了每个输出 55 个周期，将速度提升至 2.3 倍（图 5 中的 CFU）。接下来，将滤波器值移入 CFU（将 CFU 用作小型 scratchpad 内存）导致了小幅度的速度提升——每个乘积累加（MAC）大约节省 2 个周期（图 5 中的 CFU hold filt）。然而，将输入值移入 CFU 每个 MAC 大约花费 2 个周期。由于 CFU 按字存储值而不是按字节存储，CPU 必须执行位移和符号扩展以使用从 CFU 检索的值。这抵消了减少内存访问和更好的数据缓存行为带来的速度提升（图 5 中的 CFU hold inp）。

然而，在下一步中，当我们**构建了一个 4x4 乘积累加（MAC4）指令，该指令对从 CFU 缓冲区检索的打包滤波器和输入进行操作时，累积速度提升达到了 9.8 倍** （图 5 中的 CFU MAC4）。然后我们**将 MAC4 指令更改为直接从先前构建的缓冲区中提取输入参数，并将整个内积累加循环移入 CFU**。这些更改使我们达到了每个 MAC 少于一个周期，累积**速度提升为 26 倍** （图 5 中的 MAC4Run1）。

最后，我们将积累结果直接连接到 CFU 中的后处理，无需 CPU 干预，达到了 31.1 倍的速度提升（图 5 中的 Incl postproc）。此时的配置文件显示，**逐个计算和写回 8 位输出通道值并未有效利用内存带宽，因此我们重构了 CFU 以将四个这样的输出打包成一个 32 位字，供 CPU 写回内存** （图 5 中的 Macc4Run4）。最后，我们对 CFU 进行了流水线化，以在加载输入的同时进行计算，从而隐藏输入加载开销（图 5 中的 Overlap）。我们在后续部分将这个完整的大型 CFU 设计称为 CFU1。

**总结**：总体而言，**针对 MNV2 的设计在 1x1 CONV_2D 上实现了比原始实现快 55 倍的速度提升**。在整个迭代开发过程中，我们从未接近耗尽任何 FPGA 资源（如图 5 所示）。资源使用在中途达到峰值，当时不同的处理步骤分别在 CFU 上实现。随着处理变得更加集成在 CFU 上，来回移动值的路径被移除，导致整体资源使用减少。

#### 2. 在 Fomu 上加速关键词识别

> 关键词识别（KWS）是一种普遍存在的始终开启的 TinyML 应用案例，因此非常适合加速。这个示例具体展示了如何使用 CFU Playground 的部署 → 配置文件 → 优化循环，**通过针对模型的优化和自定义 CFU，将 MLPerf Tiny KWS 定量（int8）模型的速度提升 75 倍**。

目标是从比 ARM Cortex-M CPU 的 CMSIS-NN 手动优化内核慢 75 倍的基线开始，使我们的周期计数与这类优化内核相当。

**部署**：我们将系统部署到微型的 Fomu FPGA 板上，其大小约为一枚便士，并且可以放入 USB 插槽中。它结合了一个 iCE40UP5k FPGA（具有 5280 个逻辑单元和 128 kB 片上 RAM）和一个 2 MB 闪存。

**配置文件**：最小的 VexRiscv 配置（无缓存、无硬件乘法、无分支预测或向前转发）无法适应 Fomu。为了将 VexRiscv 挤入 FPGA，我们需要移除 LiteX SoC 中的一些功能（即硬件计时器和复位寄存器）并重新分配 VexRiscv 核心中的逻辑单元使用，移除硬件错误检查（例如对未对齐地址的检查）。通过简单的配置选项可以实现这些修改，不需要详细的 CPU 知识。此外，CFU Playground 编译的二进制映像无法适应 128 kB 的内存，因为其中大部分内存被 TFLite Micro 用于工作数据。我们修改了链接器脚本，将代码（.text 部分）和只读数据（.rodata 部分——主要是 ML 模型的权重）放入闪存中。KWS 应用最终适应了，但运行需要 2.5 分钟，我们最终将其优化到 2 秒内。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/10.webp)

**内存系统优化**：配置文件显示闪存 ROM 访问速度比预期慢。这指出了 SPI 闪存接口的一些潜在改进。首先，我们将 ROM 接口从串行外设接口（SPI）升级到四通道 SPI，显著提高了 ROM 读取带宽，比基线速度提升了 3.04 倍（图 7 中的 QuadSPI）。接下来，我们将关键部分从 ROM 移动到 SRAM。由于 SRAM 容量严重受限（128 kB），我们通过配置文件确定了主要瓶颈，包括 CONV_2D、Depthwise_CONV_2D 的代码和模型权重。这一优化使总速度提升了 7.84 倍（图 7 中的 SRAM Ops 和 Model）。然后我们移除了不必要的控制/状态寄存器和 SoC 功能，以调试为目的，为更大的 I-Cache 腾出空间，减少了平均指令获取时间。这使性能提升了 8.3 倍（图 7 中的 Larger I-Cache）。

**CFU 优化**：由于之前的资源节省，我们现在有了空间添加单周期乘法，而不是使用我们的迭代约 30 个周期的乘法。这使用了 Fomu 的八个 DSP 瓷砖中的四个，并将性能提升了 15.35 倍（图 7 中的 Fast Mult）。节省的空间还使我们能够添加一个四路并行 MAC CFU 以解决卷积瓶颈（图 7 中的 MAC Conv），这需要剩下的四个 DSP 瓷砖。深度卷积是第二个运行时间贡献者。它有不同的内存访问模式，因此可以使用我们为卷积构建的四路乘积累加。理想情况下，我们可以为深度卷积构建一个单独的 CFU 门级逻辑，但没有剩余资源来扩展 CFU。因此，我们利用四路乘积累加 CFU 中的一个通道来实现累积速度提升 32.10 倍。在实现 SIMD MAC 操作后，仍有逻辑单元剩余（尽管没有 DSP 瓷砖可用）。我们利用这些逻辑单元在 CFU 中添加了额外的功能，以在卷积和深度卷积操作内部深处执行累积后处理（包括饱和乘法、舍入除法和输出钳位），速度比之前快 14 倍。这一更改使速度提升了 37.64 倍（图 7 中的 Post Proc）。

**软件优化**：在软件方面，我们对操作符进行了专门化处理。TensorFlow Lite 中的卷积操作符非常通用。因此，我们向编译器提供了有关常量和不变量的信息（我们的 filter_width 始终为 3，我们的 depth_multiplier 始终为 1 等）。这使编译器能够生成更高效的汇编代码，减少分支数量（因为我们的 MCU 级 CPU 没有空间放置预测器，分支操作非常昂贵）。

**总结**：KWS 的性能从基线提升了 75 倍。单次推理的时间从 2.5 分钟减少到不到 2 秒。ASIC 实现会比这快一个数量级，但这超出了本文的范围。只有 3 倍的速度提升直接归功于小型 CFU。我们在下一节中将这个 CFU 设计称为 CFU2。其余的 25 倍提升来自对 CPU、软件、内存访问以及系统接口和驱动程序的优化。最终优化的 Fomu KWS 结果在归一化时，与使用手工优化的 CMSISNN 内核利用 M4 SIMD 指令的更复杂的 Cortex-M4 的 MLPerf Tiny 结果 [2] 大致相当。我们在这个阶段停止了工作，但可以继续使用工具进行改进。这说明了 CFU Playground 的迭代设计循环在实时系统上运行的价值，它便于硬件-软件协同设计。

### C. CPU 与 CFU 的自动化设计空间探索

> 在构建了第 IV-B 节中提到的两个用例的 CFU（即 CFU1 和 CFU2）之后，我们使用 CFU Playground 内置的设计空间探索（DSE）功能，探索了 MNV2 工作负载的不同 CPU + CFU 配置的设计空间。当考虑所有软 CPU 的可配置架构参数与 CFU 相结合时，设计空间包含了大约 93,000 个不同的设计点。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/11.webp)

图 8 展示了 Vizier 执行的 DSE。为了扩展 Vizier 可以测试的不同设计点的数量，我们在云端使用模拟运行实验。图 8 中有三条不同的 Pareto 曲线。绿色曲线代表仅 CPU，蓝色曲线代表 CPU 与大型复杂 CFU1 的组合，红色曲线代表 CPU 与更小更简单的 CFU2 的组合。带星号的点突出了设计空间中整体 Pareto 最优的点。这些结果展示了 CFU 设计如何创造更丰富的设计空间，从而获得更优的配置。例如，在图中我们可以看到，红色曲线（CPU + CFU2）上存在一些设计点，它们使用的资源比绿色曲线（仅 CPU）上的一些较大设计点更少，但性能更好。

## 五、先前相关工作

> 近年来，基于 FPGA 的机器学习加速器因其相比 GPU 的效率、比 ASIC 更轻量级的投资以及整体灵活性而受到关注，因为它们可以重新配置 [29]。因此，许多工具应运而生，支持机器学习的 FPGA 设计。

然而，直接对其他工具进行 quantitative（定量的）比较具有挑战性，因为**我们认为没有一种“一刀切”的解决方案来设计定制的机器学习硬件加速器**。**最佳的工作流程取决于特定任务**，由开发人员决定。因此，我们从不同维度对 CFU Playground 与先前工作进行 qualitative（定性的）比较。**CFU Playground 在先前技术的基础上增加了灵活性**，尤其适用于资源受限的嵌入式系统，因为在这些系统中，离散加速器并不总是可行的。

![文章插图](/articles/2025-04-25-55-tinyml-fpga-cfu-playground-f3e76d92/images/12.webp)

**已经有许多工具和工作流程用于加速机器学习的定制硬件**。**表 III 比较了 CFU Playground 与一些现有框架所支持的功能**。更全面广泛的 FPGA 现有工作调查可在相关工作 [30]–[32] 中找到。人们已经探索了使用应用特定指令集处理器（ASIPs）为深度神经网络（DNN）加速的专用 ISA（指令集架构）[33]，[34]。这些方法开发了固定、通用的 ISA 以加速更多样化和通用的 DNN，而**我们的方法则是针对嵌入式系统进行极端的模型特定定制**，这些嵌入式系统通常是特定任务的，不太适用于通用计算。此外，**ASIP 加速器的工具是商业化的** [35]，[36]，**需要昂贵的前期投入**。这些**工具的使用开销远大于 CFU Playground 的开箱即用解决方案**，后者已经集成了完整的机器学习推理堆栈和 SoC，可以立即采用和重现。

::: block-1
此外，通常不将加速器与 CPU 紧密耦合，大多数工作流程遵循这种设计模式 [22]–[25]，[27]，[28]。由于这种解耦，**许多流程在集成到其余系统和堆栈之前独立设计加速器**。这可能**导致由于缺乏对计算堆栈或系统级其他地方产生的效应的考虑而导致设计失败**，例如 offloading（卸载）的成本、调度等。独立于系统设计加速器不允许与处理器进行协同设计，以探索定制 CPU 的权衡。例如，Centrifuge [21] 并未像 CFU Playground 那样将 CPU 包含在公开的硬件设计空间中。
:::

**Chipyard** [20] 确实支持紧密耦合加速器的端到端集成，但**并未探索 CPU 和加速器之间的设计空间权衡**，或提供自动参数探索，正如我们的解决方案所做的那样。Gemmini 简要提到了加速器和 CPU 之间的设计权衡，但其使用架构模板生成 systolic array（ systolic array 是一种特定类型的硬件架构，用于高效计算矩阵运算 [37]）加速器，与软件堆栈和集成 SoC 一起捕获系统级效应 [23]。然而，与大多数加速器工作流程类似，Gemmini 的模板旨在优化和运行通用矩阵乘法内核。

**hls4ml** [24] 是一个**完整的将不同模型部署到 FPGA 的工具**，但其**加速器旨在加速大型内核或整个模型**，而不是我们方法中选择的单个 ML 操作。CFU 提供了比这些方法更细的粒度，可以逐步设计为加速模型特定操作，同时支持通常会在设备上部署的其余系统堆栈。

最后，大多数其他设计流程使用固定的模板和专用编译器自动生成加速器的硬件实现。这些限制较大，不如 CFU Playground 的工程师主导方法和使用 RISC-V 灵活。此外，许多流程往往不是模型不可知的，而是限于特定类别的机器学习模型，例如 FINN [28]。

## 六、结论与未来工作

> **微型机器学习** （TinyML）需要**依赖硬件-软件协同设计的定制架构**。为此，CFU Playground **为用户提供了用于实验性和模型特定加速的开源全栈框架**。它使开发人员能够**快速迭代设计和评估** 流程，通过探索 CPU 与紧密耦合的 CFU 之间的设计空间，获得显著的速度提升。这对于无法承受离散加速器的功率和面积限制的资源极度受限平台特别有帮助。

CFU Playground **已经得到行业采用和支持**，并将持续保持相关性。我们将继续发展 CFU Playground，**增加开源的 ASIC 流程**、**更多的模型架构** 支持、为紧密耦合的 CFU 引擎提供**直接内存访问**，以及更多**具备 CFU 接口的开源 RISC-V CPU（软核和硬核）**。研究使用高级综合（HLS）指令来识别执行热点以自动生成 CFU 并插入自定义指令调用，可能会提供进一步的自动化。未来的工作还将涉及研究功耗和能效优化的空间。

## 参考文献

[1] T. Yang, Y.-H. Chen, J. Emer, and V. Sze, “A method to estimate the energy consumption of deep neural networks,” in Asilomar conference on signals, systems, and computers, 2017.

[2] C. Banbury, V. J. Reddi, P. Torelli, J. Holleman, N. Jeffries, C. Kiraly, P. Montino, D. Kanter, S. Ahmed, and D. Pau, “Mlperf tiny benchmark,” NeurIPS Track on Datasets and Benchmarks, 2021.

[3] C. Banbury, , V. J. Reddi, M. Lam, W. Fu, A. Fazel, J. Holleman, X. Huang, R. Hurtado, D. Kanter, A. Lokhmotov et al., “Benchmarking tinyml systems: Challenges and direction,” arXiv, 2020.

[4] M. Buch, Z. Azad, A. Joshi, and V. J. Reddi, “Ai tax in mobile socs: End-to-end performance analysis of machine learning in smartphones,” in 2021 ISPASS, 2021.

[5] D. Richins, D. Doshi, M. Blackmore, A. T. Nair, N. Pathapati, A. Patel, B. Daguman, D. Dobrijalowski, R. Illikkal, and K. Long, “Ai tax: The hidden cost of ai data center applications,” ACM TOCS, 2021.

[6] T. Hanawa, Y. Kodama, T. Boku, and M. Sato, “Tightly coupled accelerators architecture for minimizing communication latency among accelerators,” in IPDPS, 2013.

[7] J. Gray, “Introducing composable custom extensions and custom function units for risc-v,” 2022. [Online]. Available: https://fpga.org/2022/04/01/composable-custom-extensions-and-custom-\ protect@normalcr\relaxfunction-units-for-risc-v/

[8] F. K., S. Bourdeauducq, J. L. Lann, and H. Badier, “Litex: an open-source soc builder and library based on migen python DSL,” CoRR, 2020.

[9] C. Pappon, “Vexriscv,” 2017. [Online]. Available: https://github.com/SpinalHDL/VexRiscv.

[10] Im-tomu, “Fomu-hardware,” 2018. [Online]. Available: https://github.com/im-tomu/fomu-hardware.

[11] SymbiFlow. [Online]. Available: https://symbiflow.github.io/.

[12] L. Lai, N. Suda, and V. Chandra, “Cmsis-nn: Efficient neural network kernels for arm cortex-m cpus,” arXiv preprint arXiv:1801.06601, 2018.

[13] R. David, J. Duke, A. Jain, V. J. Reddi, N. Jeffries, J. Li, N. Kreeger, I. Nappier, M. Natraj, and S. Regev, “Tensorflow lite micro,” arXiv, 2020.

[14] M-Labs, “nmigen,” 2018. [Online]. Available: https://github.com/.

[15] Antmicro, “Renode,” 2018. [Online]. Available: https://renode.io/.

[16] S. Hooker, “The hardware lottery,” CoRR, vol. abs/2009.06489, 2020. [Online]. Available: https://arxiv.org/abs/2009.06489.

[17] X. Song, S. Perel, C. Lee, G. Kochanski, and D. Golovin, “Open source vizier,” in AutoML-Conf Systems, 2022.

[18] D. Shah, E. Hung, C. Wolf, S. Bazanski, D. E. Gisselquist, and M. Milanovic, “Yosys+nextpnr: An open source framework from verilog to bitstream for commercial fpgas,” IEEE 27th FCCM, 2019.

[19] Verilator, 2003. [Online]. Available: https://github.com/verilator.

[20] A. Amid, D. Biancolin, A. Gonzalez, D. Grubb, S. Karandikar, H. Liew, A. Magyar, H. Mao, A. Ou, N. Pemberton et al., “Chipyard: Integrated design, simulation, and implementation framework for custom socs,” IEEE Micro, vol. 40, no. 4, pp. 10–21, 2020.

[21] Q. Huang, C. Yarp, S. Karandikar, N. Pemberton, B. Brock, L. Ma, G. Dai, R. Quitt, K. Asanovic, and J. Wawrzynek, “Centrifuge: Evaluating full-system hls-generated heterogenous-accelerator socs using fpga-acceleration,” in 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2019, pp. 1–8.

[22] P. Mantovani, D. Giri, G. Di Guglielmo, L. Piccolboni, J. Zuckerman, E. G. Cota, M. Petracca, C. Pilato, and L. P. Carloni, “Agile soc development with open esp,” in Proceedings of the 39th International Conference on Computer-Aided Design, 2020, pp. 1–9.

[23] H. Genc, S. Kim, A. Amid, A. Haj-Ali, V. Iyer, P. Prakash, J. Zhao, D. Grubb, H. Liew, and H. a. Mao, “Gemmini: Enabling systematic deep-learning architecture evaluation via full-stack integration,” in DAC, 2021.

[24] F. Fahim, B. Hawks, C. Herwig, J. Hirschauer, S. Jindariani, N. Tran, L. P. Carloni, G. D. Guglielmo, P. C. Harris, J. D. Krupa, D. S. Rankin, M. B. Valentin, J. D. Hester, Y. Luo, J. Mamish, S. Orgrenci-Memik, T. Aarrestad, H. Javed, V. Loncar, M. Pierini, A. A. Pol, S. Summers, J. M. Duarte, S. Hauck, S. Hsu, J. Ngadiuba, M. Liu, D. Hoang, E. Kreinar, and Z. Wu, “hls4ml: An open-source codesign workﬂow to empower scientiﬁc low-power machine learning devices,” CoRR, 2021.

[25] Y. Wang, J. Xu, Y. Han, H. Li, and X. Li, “Deepburning: Automatic generation of fpga-based learning accelerators for the neural network family,” in DAC, 2016.

[26] H. Sharma, J. Park, E. Amaro, B. Thwaites, P. Kotha, A. Gupta, J. K. Kim, A. Mishra, and H. Esmaeilzadeh, “Dnnweaver: From high-level deep network models to fpga acceleration,” in Workshop on Cognitive Architectures, 2016.

[27] X. Zhang, J. Wang, C. Zhu, Y. Lin, J. Xiong, W.-m. Hwu, and D. Chen, “Dnnbuilder: an automated tool for building high-performance dnn hardware accelerators for fpgas,” in ICCAD, 2018.

[28] Y. Umuroglu, N. J. Fraser, G. Gambardella, M. Blott, P. Leong, M. Jahre, and K. Vissers, “Finn: A framework for fast, scalable binarized neural network inference,” in International Symposium on FPGA, 2017.

[29] Y. Ma, Y. Cao, S. Vrudhula, and J.-s. Seo, “Optimizing loop operation and dataﬂow in fpga acceleration of deep convolutional neural networks,” in Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, ser. FPGA ’17. New York, NY, USA: Association for Computing Machinery, 2017, p. 45–54. [Online]. Available: https://doi.org/10.1145/3020078.3021736.

[30] A. Shawahna, S. M. Sait, and A. El-Maleh, “Fpga-based accelerators of deep learning networks for learning and classiﬁcation: A review,” IEEE Access, 2018.

[31] S. Venieris, A. Kouris, and C.-S. Bouganis, “Toolﬂows for mapping convolutional neural networks on fpgas: A survey and future directions,” arXiv, 2018.

[32] K. Guo, S. Zeng, J. Yu, Y. Wang, and H. Yang, “A survey of fpga-based neural network inference accelerators,” ACM TRETS, 2019.

[33] E. Yang, T. Jia, D. Brooks, and G.-Y. Wei, “Flexacc: A programmable accelerator with application-speciﬁc isa for ﬂexible dnn inference,” in IEEE 32nd ASAP, 2021.

[34] S. Liu, Z. Du, J. Tao, D. Han, T. Luo, Y. Xie, Y. Chen, and T. Chen, “Cambricon: An instruction set architecture for neural networks,” in 2016 ISCA, 2016.

[35] Synopsys, “Asip designer,” 2017. [Online]. Available: https://www.synopsys.com/designware-ip/processor-solutions/asips-tools.html.

[36] Cadence, “Tensilica processor ip,” 1997. [Online]. Available: https://www.cadence.com/en_US/home/tools/ip/tensilica-ip.html.
