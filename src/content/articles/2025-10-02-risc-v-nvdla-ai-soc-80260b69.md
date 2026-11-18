关键词：系统级芯片（SoC）、RISC-V、NVIDIA 深度学习加速器（NVDLA）、硬件加速器、深度学习、现场可编程门阵列（FPGA）

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/01.webp)

- Bare-Metal RISC-V + NVDLA SoC for Efficient Deep Learning Inference
- https://arxiv.org/abs/2508.16095
- https://github.com/vineetbitsp/riscv-nvdla-sw
- 本文 7025 字，阅读需 23 分钟，播客 18 分钟

相关推荐

- [谈谈TVM与领域专用架构：DSA优化思路、TVM落地挑战、硬件、NVDLA架构思考](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887071&idx=1&sn=c06f737da907fd03cd5233685049b50e&scene=21#wechat_redirect)
- [卷积提速 55 倍！为 TinyML 打造的 FPGA 全栈开源框架 CFU Playground，敏捷探索轻量加速器设计空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888011&idx=1&sn=3ea5446297b91cdd15378baa2ed83125&scene=21#wechat_redirect)
- [16KB RAM跑MobileNet？分块多阶段融合方案msf-CNN：探索 TinyML 精度与延迟极致灵活性！适用广泛硬件！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888703&idx=1&sn=a7806a7e47b81b9b09c8e40b3339d3fe&scene=21#wechat_redirect)

> 本文提出了**一种新颖的系统级芯** 片（SoC，将多个功能模块集成在单块芯片上的集成电路）架构，**通过硬件与软件优化相结合，为边缘计算应用加速复杂深度学习模型。**

该硬件架构将开源的 NVIDIA 深度学习加速器（NVDLA，NVIDIA 推出的开源深度学习推理引擎）与 Codasip® 公司一款名为 µRISC V 的 32 位、4 级流水线 RISC-V 核心紧密耦合。

::: block-1
为了在**软件层面** 分担模型加速任务，我们的工具流会**生成裸机** （不依赖操作系统，直接在硬件上运行）**应用代码** （汇编语言编写），从而克服了以往探索类似架构的研究中存在的复杂操作系统开销问题。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/02.webp)

::: image-caption
图 2 展示了所提片上系统的架构设计
:::

这种紧密耦合的架构与裸机流程提升了执行速度和存储效率，使其适用于边缘计算解决方案。
:::

我们在 AMD 公司的 ZCU102 现场可编程门阵列（FPGA，一种可根据需求重新配置硬件逻辑的芯片）开发板上，**采用 NVDLA-small（NVDLA 的小型配置版本）配置对该架构进行评估，并使用 LeNet-5、ResNet-18 和 ResNet-50 模型对工具流进行测试。** 结果表明，在 100 MHz 的系统时钟频率下，这些模型的推理时间分别可达到 4.8 毫秒、16.2 毫秒和 1.1 秒。

## 本文目录

- 一、引言
- 二、相关工作
- 三、我们提出的系统
- 四、方法学
  - A. 硬件开发流程
  - B. 软件开发流程（Software Development Flow）
- 五、评估与测试
  - 未来工作
- 六、结论
- 参考文献

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/03.webp)

## 一、引言

人工智能（AI）工作负载日益增长的计算需求以及边缘设备的局限性，推动了对专用硬件加速器的需求。开源硬件（硬件设计方案公开可获取、允许修改和再分发的硬件）的兴起，促成了诸如 NVIDIA 深度学习加速器（NVDLA）[1]-[4]这类加速器的开发。

> NVDLA 是一种可扩展、可配置的开源推理引擎（用于执行深度学习模型推理任务的硬件/软件模块），适用于边缘人工智能（Edge AI，在边缘设备上部署的人工智能技术）。正如多项研究所述，将 NVDLA 与 RISC-V 集成，为深度学习加速提供了一种极具吸引力的解决方案。

**以往的研究** [5]-[7]探讨了 RISC-V 与 NVDLA 的集成，以提升深度学习推理效率和灵活性。然而，这些研究主要**侧重于基于仿真的实现，而非基于现场可编程门阵列（FPGA）等实际硬件的部署**。

- 文献[8]提出了**一种基于 FPGA 的原型系统，该系统集成了多个 NVDLA 实例和一个 RISC-V 核心，但未提供资源利用率和集成方法的细节**。此外，这些研究[5]-[8]**依赖基于 Linux 的内核** （操作系统的核心组件，负责管理硬件资源和任务调度）来执行神经网络模型，这需要 NVDLA 驱动程序（用于使操作系统与 NVDLA 硬件通信的软件），并会产生显著的软件开销（软件运行过程中消耗的系统资源）。
- **仅有少数研究[10]-[12]展示了在系统级芯片（SoC）中，将 NVDLA 与现有处理器核心集成的 FPGA 实现**，这些实现通常采用如 PetaLinux[10]（一种针对嵌入式系统优化的 Linux 发行版）等基于 Linux 的环境。
- 其他研究[13]-[16]则探索了 NVDLA 在各类研究应用中的使用，但并未关注该加速器与 RISC-V 核心的集成。依赖 Linux 内核执行深度学习工作负载会带来额外的性能和存储开销，使得这些解决方案不太适用于资源受限（如计算能力、存储容量有限）的边缘设备。

> 本文提出了一种基于开源 NVDLA 和 RISC-V 的 SoC 设计，该 SoC 以神经网络模型为输入，不依赖 Linux 内核，而是通过 RISC-V 汇编代码在 NVDLA 上执行模型。此外，我们通过在 FPGA 上运行神经网络模型，对该 SoC 进行了实物验证。

我们没有使用由 Linux 内核管理的驱动程序栈（一系列协同工作的驱动程序集合），而是利用配置文件（轨迹文件，记录硬件控制指令序列的文件）直接配置 NVDLA 的寄存器（硬件中用于暂存数据和控制信息的存储单元），这些配置文件充当执行控制序列。

NVDLA 官方发布版本提供了用于基本硬件测试（如完整性检查、卷积层测试和池化层测试，均为深度学习模型中的常见操作和测试类型）的预生成配置文件。然而，目前尚无关于这些文件如何生成，或如何为任意神经网络创建此类文件的指南。**本研究通过提出一种为任意基于 Caffe（一种深度学习框架）的神经网络模型生成配置文件的方法，** 填补了这一空白。**这些配置文件随后会被转换为 RISC-V 汇编代码**，从而实现对 NVDLA 的直接硬件配置。本研究的主要贡献包括：

::: block-1
- 设计了基于 NVDLA 和 RISC-V 的 SoC 架构，并在 FPGA 上实现该架构
- 实现了为任意基于 Caffe 的神经网络模型自动生成配置文件和提取权重的功能
- **设计了紧密耦合的硬件架构和基于裸机汇编的执行方式**，无需依赖 Linux 内核和额外存储
:::

在系统设计方面，我们将 NVDLA 与 Codasip 公司的 µRISC V 核心集成，并在 AMD ZCU102 FPGA 开发板上实现了该设计。我们使用 LeNet-5、ResNet-18 和 ResNet-50 神经网络模型对该系统进行了验证。

## 二、相关工作

已有多项研究探索了 NVDLA 在不同计算环境中的集成与评估，这些研究主要侧重于基于仿真的方法和基于 Linux 内核的 FPGA 实现。

G**em5-NVDLA[6]是一款用于分析设计权衡** （在硬件设计中对性能、资源消耗等指标进行取舍）**和在仿真环境中评估 NVDLA 性能的实用工具**。然而，该研究的范围存在局限，因为它:mdnice-bold-italic[不支持 NVDLA 的小型配置] （nv small）[4]。冈萨雷斯（Gonzalez）和洪（Hong）[7]在 Chipyard 框架（一种开源的芯片设计与验证框架）内对 NVDLA 和 Gemmini（另一种深度学习加速器）进行了对比研究，评估了两者在处理深度学习工作负载方面各自的优势。尽管该研究具有启发性，但它:mdnice-bold-italic[依赖特定框架（即 Chipyard），这限制了其仅适用于 Chipyard 用户。]

法尔希奇（Farshchi）等人[5]**使用 FireSim（一种周期精确的仿真平台，能够精确模拟硬件每一个时钟周期的运行状态）研究了 NVDLA 与基于 RISC-V 的 SoC 的集成，并在目标检测任务中评估了系统性能**。然而，他们的研究:mdnice-bold-italic[仅限于基于仿真的分析，并未解决基于 FPGA 的部署或定制专用集成电路] （ASIC，为特定应用设计的专用芯片）设计所面临的实际挑战。值得注意的是，由于 FireSim 的限制，:mdnice-bold-italic[他们的仿真假设 NVDLA 的工作频率为 3.2 吉赫兹（GHz）——这一频率与中央处理器（CPU）时钟频率相当，显然不切实际]；而在实际的 FPGA 实现中，NVDLA 的工作频率通常低于 100 兆赫兹（MHz）[8]。

**吉里（Giri）等人[8]提出了一种用于敏捷异构 SoC** （包含多种不同类型处理器/加速器的 SoC）**设计的开源嵌入式系统平台，并展示了基于 FPGA 的原型系统——该系统集成了多个 NVDLA 实例和 Ariane RISC-V 64 位处理器核心** （一种开源的 64 位 RISC-V 处理器核心）。然而，他们的研究:mdnice-bold-italic[未提供关于 FPGA 资源利用率或集成方法的细节。]

> 据我们所知，以往所有研究[5]-[12]都需要依赖 Linux 内核来配置和运行 NVDLA。与之相反，**本研究采用裸机汇编编程（不依赖操作系统，直接通过汇编语言编写代码控制硬件），为特定神经网络直接配置 NVDLA 的寄存器**。此外，我们的:mdnice-bold-italic[实现同时支持 NVDLA 的小型配置（nv small）和完整配置（nv full）。]

## 三、我们提出的系统

本节首先概述所提片上系统（SoC，System-on-Chip，一种将多个功能模块集成在单块芯片上的系统架构）的架构，随后在后续章节中详细介绍软硬件开发流程。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/04.webp)

::: image-caption
图 1 所提系统及软件开发流程。该工作流可将训练完成的神经网络模型转换为 RISC-V 汇编代码及对应的权重文件。由于此过程具有模型特异性（即针对不同模型需单独执行一次），因此会利用 NVDLA 的虚拟平台（VP，Virtual Platform，一种用于模拟硬件运行行为的软件环境），结合第四章第二节所述的软件开发方法，以离线方式执行
:::

流程说明：

::: block-1
- 从`.caffemodel`文件开始，经 NVDLA 编译器处理后，在 NVDLA 虚拟平台（VP）上运行；
- 同时需针对 NVDLA 小型配置（nv small NVDLA，NVDLA 的轻量级版本，资源占用较低，适合边缘设备）更新虚拟平台；
- 运行过程中生成虚拟平台日志（VP logs）；基于日志进行权重提取（Weight extraction）；
- 最终生成 RISC-V 汇编代码（RISC-V assembly code）、测试文件（Test file，.bin 格式）与权重文件（Weight file）；
- 这些文件将用于片上系统（System-on-chip）中的 RISC-V 核与 NVDLA 加速器协同工作。
:::

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/02.webp)

::: image-caption
图 2 展示了所提片上系统的架构设计
:::

所提出的片上系统的架构设计，通过系统总线（system bus）、仲裁器（arbiter，用于协调多个设备对共享资源的访问，避免冲突）与定制化 NVDLA 封装模块（NVDLA wrapper，用于适配不同硬件模块的接口协议，确保数据正常传输），将 NVDLA 加速器与 µRISC-V 核集成。

其中，系统总线包含内部解码器（internal decoder，用于解析地址信息，将指令或数据路由至目标设备）与仲裁逻辑（arbitration logic），可实现 µRISC-V 核与两个内存映射从设备（memory-mapped slave devices）——NVDLA 引擎与基于 DRAM 的数据源存储器（DRAM-based data memory）之间的通信。由于数据源存储器为共享资源，仲裁器需管理 µRISC-V 核与 NVDLA 之间的潜在访问冲突。

:mdnice-bold-italic[NVDLA 封装模块包含加速器硬件、接口桥接器（interface bridges，用于实现不同总线协议间的转换）与数据宽度转换器（data width converter），以解决 µRISC-V 核与 NVDLA 之间的接口不匹配问题]。具体而言，AXI 数据宽度转换器（AXI 是一种高性能片上总线协议，常用于高带宽数据传输场景）可将 NVDLA 的 64 位数据骨干（DBB，Data Backbone，NVDLA 内部用于高速传输数据的核心通路）接口连接至 32 位数据源存储器。µRISC-V 核采用 AHB-Lite 接口（一种轻量级高性能总线协议，适用于低复杂度、高实时性的访问场景）访问程序存储器（program memory）与数据源存储器。

与 NVDLA 的配置空间总线（CSB，Configuration Space Bus，用于配置 NVDLA 内部寄存器的专用总线）通信时，需通过 AHB-Lite 到 APB 桥接器（APB 是一种低功耗、简单的总线协议，常用于外设配置），且复用了 NVDLA 软件包中提供的现有 APB-to-CSB 适配器。该 AHB-APB 桥接器基于开源 ARM 设计，为此次集成提供了支持。此外，AHB-Lite 到 AXI 桥接器可实现 µRISC-V 核与 AXI 兼容型数据源存储器之间的连接。系统总线解码器为每个从设备（NVDLA 与 DRAM）分配唯一的地址空间，确保高效的内存映射通信（memory-mapped communication，一种将硬件设备寄存器映射到内存地址空间，通过读写内存地址实现设备控制的方式）。

## 四、方法学

### A. 硬件开发流程

1. **NVDLA 硬件生成（NVDLA Hardware Generation）**：采用 NVDLA 官方 GitHub 仓库[17]中的参数化 Verilog 代码，通过文档[4]中概述的硬件树构建流程，生成目标硬件配置。
2. **系统集成（System Integration）**：利用 Codasip Studio（Codasip 公司推出的 RISC-V 处理器设计与集成工具）将 NVDLA 与 RISC-V 处理器集成。开发了定制化封装模块，该模块包含 NVDLA 核、接口桥接器与数据宽度转换器，以确保各模块间的无缝兼容。通过 Codasip Studio 的测试平台结构（testbench construct，用于验证硬件功能的仿真环境框架），将系统总线、仲裁器与存储器互连，生成可综合的 RTL 代码，随后将其导入 Vivado 设计套件见图 2。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/02.webp)

::: image-caption
图 2 展示了所提片上系统的架构设计
:::

> 系统总线解码器为两个从设备分配专用地址空间：

- NVDLA：地址范围 `0x0 -- 0xFFFFF`，覆盖 NVDLA 的所有配置寄存器地址；
- DRAM：地址范围 `0x100000 -- 0x200FFFFF`，可访问 512 MB 的 DRAM 数据源存储器。

这种内存映射设计使 RISC-V 处理器可通过标准加载（load）与存储（store）指令（RISC-V 指令集中用于读写内存数据的基础指令）编程 NVDLA，实现配置寄存器的写入与状态读取，无需设计定制化 RISC-V 指令。

仲裁器组件协调 NVDLA（通过其 DBB 接口）与 µRISC-V 核（通过其 AHB 接口）对 DRAM 的访问，确保访问互斥（mutual exclusion，同一时间仅允许一个设备访问 DRAM，避免数据冲突）与高效的内存利用率。:mdnice-bold-italic[这种紧密耦合的硬件接口支持基于裸机汇编（bare-metal assembly，不依赖操作系统，直接操作硬件寄存器的编程方式）的神经网络执行。]

1. **仿真与 FPGA 原型验证（Simulation and FPGA Prototyping）**：在 Vivado 中，使用前一步生成的 RTL 代码及用于桥接器与转换器的 Vivado IP 核（IP 核，Intellectual Property Core，预先设计好的可复用硬件功能模块）进行行为仿真（behavioral simulation，模拟硬件在实际工作中的行为，验证功能正确性），同时将软件二进制文件与神经网络权重加载到存储器中。仿真成功后，对设计进行综合（synthesis，将 RTL 代码转换为门级网表，即描述电路中逻辑门连接关系的文件）并部署到 FPGA 板上，利用板载 DDR 存储器（一种高速同步动态随机存取存储器，用于存储输入数据与模型权重）存储输入信息与权重。通过执行多种深度神经网络（DNN）模型，评估系统性能。

### B. 软件开发流程（Software Development Flow）

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/04.webp)

::: image-caption
图 1 所提系统及软件开发流程。该工作流可将训练完成的神经网络模型转换为 RISC-V 汇编代码及对应的权重文件。由于此过程具有模型特异性（即针对不同模型需单独执行一次），因此会利用 NVDLA 的虚拟平台（VP，Virtual Platform，一种用于模拟硬件运行行为的软件环境），结合第四章第二节所述的软件开发方法，以离线方式执行
:::

该软件流程可从 Caffe 模型生成 RISC-V 机器码（machine code，可被 RISC-V 处理器直接执行的二进制指令）并提取神经网络权重见图 1。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/05.webp)

::: image-caption
图 3 NVDLA 虚拟平台的内部结构，包含 CPU、路由器（router）、只读存储器（romem）、CSB 适配器（csb adaptor）、数据处理通路适配器（dpp adaptor）、SRAM 适配器（sram adaptor）、NVDLA 核（nvdla_core）与中断适配器（irq adaptor），各模块通过内部通路协同工作，模拟 NVDLA 的硬件行为
:::

> GitHub 仓库（https://github.com/vineetbitsp/riscv-nvdla-sw）提供了 Python 脚本、Linux 命令与详细说明，用于通过以下步骤生成裸机 RISC-V 软件：

1. **在虚拟平台上执行（Execution on Virtual Platform）**：使用 NVDLA 编译器编译 Caffe 模型，并在 NVDLA 的虚拟平台（VP）上执行。该虚拟平台通过 QEMU（一种开源虚拟机监控器，可模拟不同架构的处理器）与 SystemC（一种用于系统级设计的 C++库，支持软硬件协同仿真），提供周期精确的协同仿真环境[4]。执行过程中，记录接口级事务（interface-level transactions，即 CSB 与 DBB 总线的读写操作）（见图 3）。
2. **配置文件生成（Configuration File Generation）**：通过 Python 脚本处理虚拟平台日志文件，提取包含关键字“nvdla.csb_adaptor”的行。每个行条目代表一次寄存器事务（register transaction，对 NVDLA 配置寄存器的读写操作），根据 iswrite 标志（用于区分读写操作的标识位）分为读操作或写操作：

生成的命令序列构成配置文件，随后将其转换为 RISC-V 汇编代码。使用 Codasip Studio 中的 RISC-V 核 SDK，将汇编代码编译为机器码，并加载到程序存储器中待执行。

1. **权重提取（Weight Extraction）**：为提取神经网络权重，Python 脚本过滤包含关键字“nvdla.dbb_adaptor”的虚拟平台日志条目。每个条目对应一次数据事务（data transaction，对存储器的数据读写操作）：

最后，删除权重文件中的重复地址条目——保留首次出现的条目（因其为原始权重数据，后续重复条目可能为冗余或更新值，无需保留）。

## 五、评估与测试

> 本节介绍我们对提出的片上系统（SoC）的性能评估，包括其 FPGA 实现以及基于标准深度学习模型的测试。

首先，利用 NVDLA 官方 GitHub 仓库[17]中提供的标准测试轨迹（如完整性测试、卷积测试和内存测试），**通过行为仿真** （behavioral simulation，指通过模拟电路的功能行为验证设计正确性，不涉及具体时序细节）**完成了功能验证**。这些测试轨迹被转换为 RISC-V 汇编代码，用于验证集成 SoC 设计的正确性。

为支持**更大规模的模型**，片上系统通过 **DDR4 内存接口生成器** （MIG DDR4，Memory Interface Generator，用于实现 FPGA 与 DDR4 内存之间的接口连接）**连接了外部 DRAM**，并且通过 ZCU102 开发板上 Zynq UltraScale+多核片上系统（MPSoC，Multi-Processor System on Chip）的 ARM 核心对该 DRAM 进行初始化。此配置使得可编程逻辑部分能够访问 512 MB 的 DDR4 内存。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/06.webp)

::: image-caption
图4. 测试我们的片上系统（SoC）的设置（整个系统设置的Vivado框图设计）
:::

> 该系统架构在 AMD Vivado 中实现，其高层接口示意图如图 4 所示。

**Zynq 核心会将权重文件和输入图像一同初始化到 DRAM 中**。在任意时刻，DRAM 通过 AXI 智能连接器（AXI SmartConnect，用于实现多主设备与从设备间的灵活连接，此处充当多路选择器）要么连接到 Zynq 核心，要么连接到所设计的 SoC。此外，由于 SoC 工作频率为 300 MHz 而 DDR4 工作频率为 100 MHz，为解决频率不匹配问题，在 SoC 与 MIG DDR4 之间加入了 AXI 互连器（AXI Interconnect）。完整的模块设计经过综合后部署到了 FPGA 上。

**所设计的 SoC 已通过标准深度学习模型（包括 LeNet-5、ResNet-18 和 ResNet-50）的成功测试**。在执行过程中，DRAM 会预加载`.bin` 格式的权重文件和图像文件；而 RISC-V 程序内存（采用 FPGA 块 RAM 实现）则加载由`.mem` 格式配置文件生成的机器码。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/07.webp)

::: image-caption
表I FPGA资源利用率（AMD的ZCU102评估板）
:::

表 I 展示了完整系统配置、所提 SoC 及其主要组件在 FPGA（AMD ZCU102 评估板）上的资源占用情况；

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/08.webp)

::: image-caption
表II 包含 NV 小型 NVDLA 的我们的 SOC 评估（FPGA 实现结果）
:::

表 II 则报告了系统时钟频率为 100 MHz 时的执行时间。如表所示，该执行速度优于此前基于 64 位 RISC-V 平台集成 NVDLA 的研究工作[8]。

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/09.webp)

::: image-caption
表III 我们包含NV完整NVDLA的SOC评估（仿真结果）
:::

表 III 给出了 NVDLA 的 nv_full 配置的仿真结果，包括总周期数以及 100 MHz 频率下的处理时间。**尽管 nv_full 配置的性能优于 nv_small，但该配置的设计规模极大，无法适配大多数 FPGA** （包括本研究中使用的 ZCU102 FPGA 开发板）。在综合过程中观察到，对于该设备，:mdnice-bold-italic[nv_full 配置的查找表（LUT，Field-Programmable Gate Array 中的基本逻辑单元）占用率严重超标]。

nv_small 配置仅支持 INT8（8 位整数）精度，而 nv_full 配置还额外支持 FP16（半精度浮点）精度。表 III 中包含的模型均采用 FP16 精度计算，其计算时间已在表中列出。:mdnice-bold-italic[通过 ResNet-50 在两种配置下的性能对比可见，nv_full 配置的速度显著更快，这是因为它集成了更多的乘加（MAC，Multiply-Accumulate，深度学习计算中的基本运算单元）单元。]

本研究存在一个局限性：当前 nv_small 配置仅支持有限的模型集，这主要是由于缺少 INT8 校准表（用于将高精度模型量化为 INT8 精度时的校准依据）。未来的工作将解决这一局限，以扩展模型支持范围。

### 未来工作

未来的开发将聚焦于扩展 nv_small 配置的模型支持范围，使其能够兼容更多深度学习模型（如 MobileNet、GoogleNet 和 AlexNet）。有两个潜在的研究方向：

::: block-1
1. 生成 NVDLA 编译器所需的 **INT8 校准表** ——目前 NVDLA 未提供该表，但 NVDLA GitHub 文档[17]中对其有部分描述。
2. **集成 ONNC 编译器** [18]，使其能够从 ONNX（Open Neural Network Exchange，一种开源神经网络模型格式）模型生成可在 NVDLA 上加载的文件，从而通过在 NVDLA 虚拟平台（VP）上执行实现更广泛的部署。
:::

## 六、结论

> 本研究设计了一种定制化片上系统（SoC），该系统将 NVDLA 加速器与 RISC-V 处理器相集成，并在 FPGA 平台上完成了实现与验证。

当前设计采用 nv_small 配置，同时具备支持 nv_full 配置的灵活性——只需修改相关参数（如将 AXI 接口宽度从 64 位调整为 512 位）即可。该 SoC 无需依赖 Linux 内核即可运行，能够实现轻量级独立执行模式，**非常适合对延迟要求低且资源受限的边缘人工智能（AI）应用场景。**

FPGA 综合结果表明，**该设计在中低端 FPGA 设备上具有可行性。**

## 参考文献

![文章插图](/articles/2025-10-02-risc-v-nvdla-ai-soc-80260b69/images/10.webp)
