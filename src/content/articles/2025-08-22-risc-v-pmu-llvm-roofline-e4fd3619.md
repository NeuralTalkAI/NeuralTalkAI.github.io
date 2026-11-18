关键词：LLVM、性能、Roofline 模型、RISC-V、PMU、T-Head、SpacemiT、SiFive

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/01.webp)

- Dissecting RISC-V Performance: Practical PMU Profiling and Hardware-Agnostic Roofline Analysis on Emerging Platforms
- https://arxiv.org/abs/2507.22451
- https://github.com/alexbatashev/miniperf
- 本文 9271 字，阅读需 31 分钟，播客 8 分钟如下

推荐文章

- [浅谈C/C++性能优化背后方法论：Top-down Micro-architecture Analysis](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886603&idx=1&sn=5a003bbe593e571325eded84f4117632&scene=21#wechat_redirect)
- [KPerfIR：扩展 Triton 面向 GPU 的开源编译器中心性能工具，以 8.2% 开销换 24.1% FA3内核性能提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890015&idx=1&sn=cafc8d3ec7baebab7417835090e56242&scene=21#wechat_redirect)
- [OSDI25：顺序性能优化系统化框架：三大原则、八大方法、SysGPT 建议模型（建议准确率超 GPT-4 64.6%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889973&idx=1&sn=2b4e5716d031232793a084eba78f48c8&scene=21#wechat_redirect)

> 随着 RISC-V 架构（一种开源的指令集架构）在嵌入式和高性能领域的普及，开发者在性能优化方面面临着持续的挑战，这些**挑战源于工具的碎片化、硬件功能的不成熟以及平台特定的缺陷**。

本文提供了一种实用的方法，**用于在 RISC-V 系统上提取可操作的性能洞察**，即使在受限或不可靠的硬件条件下也是如此。

::: block-1
我们提出了一种:mdnice-bold-italic[绕过 RISC-V 实现中硬件漏洞的方法，实现了可靠的事件采样]。对于内存-计算瓶颈分析，我们**引入了编译器驱动的 Roofline 工具** （一种用于分析程序计算和内存瓶颈的性能模型工具），:mdnice-bold-italic[该工具不依赖硬件 PMU] （性能监控单元，用于监控处理器性能事件的硬件组件），而是**利用基于 LLVM 的插桩技术** （通过在代码中插入额外代码来收集信息的技术）:mdnice-bold-italic[直接从应用程序的 IR（中间表示，编译器处理代码时的一种中间形式）中推导操作强度和吞吐量指标]。
:::

我们的开源工具链自动化了这些解决方法，将 PMU 数据校正和编译器引导的 Roofline 构建整合到一个工作流中。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/02.webp)

## 关键问题

### 问题 1：论文提出的 PMU 采样 workaround 主要针对 SpacemiT X60，而 RISC-V 生态存在 SiFive U74、T-Head C910 等多种差异显著的硬件实现，这些方法对其他硬件的适配性如何？是否需要为每种硬件单独开发特定 workaround，这是否会削弱其在碎片化生态中的实用价值？

关于 PMU 采样方案对其他 :mdnice-bold-italic[RISC-V 硬件的适配性，作者指出 RISC-V 硬件 PMU 能力存在显著异质性（如 SiFive U74 缺乏溢出中断支持、T-Head C910 依赖厂商内核修改）]。

当前 workaround 主要针对 SpacemiT X60 的非标准计数器特性，但 **miniperf 通过直接读取 CPU 识别寄存器而非依赖标准 perf 事件发现机制，为适配其他硬件提供了基础。**

不过，作者也承认需针对不同硬件的具体限制开发特定适配（如处理不同厂商的非标准计数器或权限机制），但工具链的模块化设计使其可扩展，并未否定其在碎片化生态中的实用价值，反而强调其为新兴平台提供了早期分析能力。

### 问题 2. 编译器驱动的 Roofline 分析依赖 LLVM IR 获取指标，论文中其结果与 Intel Advisor 存在差异，且存在 instrumentation overhead。在包含大量外部库调用、动态代码生成或复杂内存依赖的真实应用中，该方法计算的操作强度、吞吐量等核心指标与硬件实际表现的偏差会多大？是否可能导致错误的性能瓶颈判断？

:mdnice-bold-italic[编译器驱动的 Roofline 分析在真实应用中存在一定偏差，但仍具实用价值]。

> 对于外部库调用，论文明确其为**局限性**：无法 instrument 外部函数内部操作，会导致内存/计算指标低估。instrumentation 带来的 overhead 可能影响绝对性能测量，但通过基线与 instrumented 两阶段执行可部分抵消。

:mdnice-bold-italic[编译器优化可能改变 IR 与硬件行为的关联，但其通过在优化 pipeline 后期应用 pass 减轻了这一影响]。尽管存在偏差，论文通过与基准值对比（如 matmul kernel 测量值 34.06 GFLOP/s 接近自报的 33.0 GFLOP/s）表明，**在 PMU 不可靠时，该方法仍能提供足够准确的瓶颈判断依据。**

### 问题 3. 集成工具链 miniperf 当前主要聚焦单核心分析，而 RISC-V 正向多核心、异构计算（如集成加速器）发展。现有工具在处理多核心协同、异构组件交互时，如何有效关联 PMU 数据与编译器指标？其设计架构是否能支撑未来对并行编程模型（如 OpenMP）的扩展需求？

针对多核心与异构计算的扩展，**论文当前聚焦单核心分析，但已规划相关方向**。:mdnice-bold-italic[miniperf 的现有架构将 PMU 采样与编译器分析分离，为整合多核心数据提供了模块化基础。]

未来计划通过集成 OMPT、XPTI 等接口，关联低级别 PMU/编译器指标与高级并行结构，以支持多核心协同分析。**尽管当前未实现，但设计路径明确**，旨在适应 RISC-V 向多核心、异构计算的发展。

## 本文目录

- 关键问题
  - 问题 1：论文提出的 PMU 采样 workaround 主要针对 SpacemiT X60，而 RISC-V 生态存在 SiFive U74、T-Head C910 等多种差异显著的硬件实现，这些方法对其他硬件的适配性如何？是否需要为每种硬件单独开发特定 workaround，这是否会削弱其在碎片化生态中的实用价值？
  - 问题 2. 编译器驱动的 Roofline 分析依赖 LLVM IR 获取指标，论文中其结果与 Intel Advisor 存在差异，且存在 instrumentation overhead。在包含大量外部库调用、动态代码生成或复杂内存依赖的真实应用中，该方法计算的操作强度、吞吐量等核心指标与硬件实际表现的偏差会多大？是否可能导致错误的性能瓶颈判断？
  - 问题 3. 集成工具链 miniperf 当前主要聚焦单核心分析，而 RISC-V 正向多核心、异构计算（如集成加速器）发展。现有工具在处理多核心协同、异构组件交互时，如何有效关联 PMU 数据与编译器指标？其设计架构是否能支撑未来对并行编程模型（如 OpenMP）的扩展需求？
- 一、引言
- 二、相关工作
  - 2.1 性能监控单元计数器
  - 2.2 性能建模与 Roofline 分析
- 三、在 RISC-V 硬件上访问 PMU 计数器
  - 3.1 架构支持
  - 3.2 Linux 中的软件支持
  - 3.3 硬件支持与当前限制
- 四、基于编译器的 Roofline 分析方法
  - 4.1 LLVM 背景知识
  - 4.2 编译器插桩细节
  - 4.3 运行时分析和 Roofline 模型构建
  - 4.4 优势和局限性
- 五、评估
  - 5.1 热点分析
  - 5.2 Roofline 分析
  - 5.3 x86 平台性能分析
  - 5.4 RISC-V 平台性能分析
- 六、结论
- 参考文献

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/03.webp)

## 一、引言

RISC-V 指令集架构（ISA，用于定义处理器能执行的指令的规范）正在计算领域迅速获得关注，从深度嵌入式系统到高性能计算集群都有其身影。其开源特性、模块化以及定制潜力带来了显著优势。然而，与任何新兴硬件生态系统一样，在多样化的 RISC-V 实现上实现最佳性能，对软件开发者和系统架构师来说都面临着相当大的障碍。

:mdnice-bold-italic[虽然性能分析对于优化至关重要，但当前的 RISC-V 生态系统常常受到工具碎片化、硬件功能不成熟以及平台特定缺陷的影响，特别是在性能监控单元（PMUs）方面]。与具有成熟分析工具和相对一致的 PMU 行为的成熟架构不同，**面向 RISC-V 的开发者经常遇到不可靠的硬件计数器、不完善的内核支持，或者直接导致标准分析技术（如事件采样）无法使用的硬件漏洞**。这种缺乏 robust、可靠的性能可观测性的情况，使有效的瓶颈分析和优化工作变得复杂，可能会限制 RISC-V 平台的采用和性能潜力。

> 本文通过提出一种实用的方法来解决这些挑战，:mdnice-bold-italic[该方法用于在 RISC-V 系统上提取可操作的性能洞察，即使在受限或不可靠的硬件条件下也是如此]。我们专注于实用的解决方法和**硬件无关的技术，以降低有效性能分析的门槛**。我们的方法:mdnice-bold-italic[结合了稳健利用可用 PMU 功能的技术，以及一种新颖的、不直接依赖硬件计数器的编译器驱动的性能建模方法。]

本工作的主要贡献有三个方面：

::: block-1
1. 一种实用的 **PMU 采样解决方法**：我们识别并展示了一种技术，用于在特定的 RISC-V 硬件（SpacemiT X60）上实现关键指标（如计算 IPC 所需的周期和指令）的可靠事件采样——在这些硬件上，由于硬件限制，标准机制无法工作。该技术利用了 Linux `perf_event` 子系统（Linux 中用于访问性能监控单元的内核子系统）中的观测到的交互。
2. **硬件无关的 Roofline 分析**：我们引入了一种使用基于 LLVM 的插桩技术的编译器驱动的 Roofline 建模方法。这种方法直接从应用程序的中间表示（IR）中推导操作强度和吞吐量指标，消除了传统 Roofline 工具通常所需的对硬件 PMU 计数器的依赖，并确保了在多样化或受限的 RISC-V 硬件上的适用性。
3. **一个集成的开源工具链**：我们提供了一个开源工具集，用于自动化这些技术，将 PMU 采样解决方法（miniperf）和编译器引导的 Roofline 构建整合到一个用于 RISC-V 性能分析的实用工作流中。
:::

## 二、相关工作

由于微架构复杂性不断增加、工作负载多样化以及硬件实现之间的差异，优化现代处理器上的应用性能面临着重大挑战。这种复杂性使得识别性能瓶颈成为一项艰巨的任务。研究人员和业界已经开发了各种方法和工具来帮助开发人员完成这一过程。

### 2.1 性能监控单元计数器

> **性能监控单元（PMUs，用于监控硬件执行过程中各种事件的硬件组件）是理解硬件执行特性的基石**。现代高端处理器通常会暴露数百个性能事件，但解读这些海量数据以精确定位实际瓶颈仍然具有挑战性。

:mdnice-bold-italic[为解决这一问题，英特尔实验室的研究人员开创了自顶向下分析方法] （一种分层识别复杂乱序处理器中瓶颈的方法），**旨在使用最少的特定性能事件来分层识别复杂乱序处理器中的瓶颈** [7]。这种方法简化了分析过程，降低了与微架构细节相关的陡峭学习曲线，并已被英特尔 VTune 等工具在实际生产中采用。

**SiFive 的研究人员尝试为其硬件构建 TMA（自顶向下微架构分析）方法的近似实现** [6]。尽管他们的一些结果适用于许多现有的 RISC-V 实现，但这项工作主要针对 SiFive 特定硬件。

在我们的工作中，我们将展示一些我们用于填补 RISC-V 架构中一种更易获取的实现空白的技术。**虽然与 TMA 的全部功能不符，但它为未来的研究提供了坚实的基础。**

### 2.2 性能建模与 Roofline 分析

> 除了直接的 PMU 计数器分析外，**性能模型还能提供对应用程序限制的直观洞察**。特别是 Roofline 模型（一种通过算术强度和硬件性能上限来判断应用是受内存限制还是计算限制的模型），因其能直观指导应用程序在特定系统上是受内存限制还是计算限制而受到关注[4]。

构建这些模型通常需要对系统能力（峰值性能、内存带宽）进行基准测试，并测量应用程序的性能特征（吞吐量、运算强度），这通常严重依赖硬件 PMU 计数器。人们普遍认识到，需要能跨不同架构自动构建模型和分析应用程序的工具。

**缓存感知 Roofline 模型（CARM）工具就提供了这种自动化能力**。CARM 包括:mdnice-bold-italic[用于评估目标硬件关键性能特征的微基准测试]，以及:mdnice-bold-italic[支持通过动态二进制插桩来提取应用程序的算术强度和内存使用情况] [5]。

尽管我们发现这种方法在成熟平台上效果很好，但在新兴平台上的初始启用需要大量的工具开发投入。**我们的工作与之不同，提出了一种与硬件无关的 Roofline 方法**。:mdnice-bold-italic[通过利用编译器插桩（特别是 LLVM IR），我们无需直接依赖 PMU 计数器就能推导出运算强度和性能指标]，即使在监控能力有限或存在故障的硬件上也能提供可行的分析路径。

## 三、在 RISC-V 硬件上访问 PMU 计数器

> **利用硬件性能监控单元（PMUs）对于深入的性能分析至关重要**。在 Linux 系统上，访问 PMU 的标准接口是 perf 工具及其底层内核子系统 `perf_event`（Linux 内核中用于处理性能事件监控的子系统）。

RISC-V 指令集架构:mdnice-bold-italic[通过 Sscofpmf 扩展（Supervisor and Counter Overflow/Filtering for Performance Monitoring Facility，即用于性能监控的 supervisor 模式计数器溢出/过滤功能扩展）提供了标准化的性能监控框架，支持周期/指令计数器和硬件事件采样]。然而，实际访问这些功能需要 Linux 内核、OpenSBI（开放源代码的 Supervisor Binary Interface，RISC-V 架构中用于管理层级交互的固件接口）固件和 perf 等用户空间工具之间的协同工作[1]。

### 3.1 架构支持

RISC-V 特权规范定义了一组用于性能监控的标准控制和状态寄存器（CSR）。核心寄存器包括：

::: block-1
- mcycle：机器周期计数器（统计处理器时钟周期）
- minstret：机器指令完成计数器（统计已完成的指令数）
- mhpmcounter[3-31]：机器硬件性能监控计数器。这些是通用计数器，可用于跟踪各种微架构事件。可用的 mhpmcounter 寄存器数量由具体实现定义。
- mhpmevent[3-31]：机器硬件性能监控事件选择器。每个选择器对应一个 mhpmcounter 寄存器，并通过编程特定的事件代码（由硬件供应商定义）来选择计数器应跟踪的事件。
- mcountinhibit：一种控制寄存器，用于全局或单独启用或禁用 mcycle、minstret 和 mhpmcounter 寄存器。
:::

尽管提供了大量通用寄存器，但通过 mhpmevent 寄存器配置这些计数器可测量的特定事件并未标准化；它们明确被定义为平台特定或实现特定的。虽然未来的规范版本可能会为常见的 ISA 级或微架构事件（如缓存未命中或特定指令类型）引入标准化，但当前已批准的版本将除周期和已完成指令之外的事件定义留给硬件实现者。

### 3.2 Linux 中的软件支持

perf 用户空间工具依赖于内核的 `perf_event` 子系统。当 perf 启动监控（例如 perf stat、perf record）时，它使用 `perf_event_open()`系统调用。这个系统调用请求内核配置特定的性能事件（硬件或软件）以进行计数或采样。然后，内核的架构特定 PMU 驱动程序负责：

::: block-1
- 硬件编程：配置 PMU 控制寄存器以计数请求的事件。
- 计数器管理：启用、禁用、读取和重置硬件计数器。
- 溢出处理：如果请求采样，则配置 PMU 在计数器溢出时生成中断。中断处理程序收集上下文信息（程序计数器、寄存器、调用栈）并记录样本。
- 数据传输：使计数器值或收集的样本可被用户空间的 perf 工具访问，通常通过映射到应用程序地址空间的高效环形缓冲区实现。
:::

运行在 supervisor 模式下的 Linux 内核通常缺乏直接配置或访问机器级性能监控单元（PMU）寄存器的必要权限，例如事件配置寄存器（mhpmevent#）或计数器禁止寄存器（mcountinhibit）。

为了弥合这一权限差距，需要与 OpenSBI 等机器模式实体进行通信。这种方法利用专用的 OpenSBI 硬件性能监控（HPM）扩展，该扩展定义了可通过标准 RISC-V 环境调用（ecall）机制调用的特定函数。通过这个 SBI 扩展，内核驱动程序可以请求 OpenSBI 代表其执行针对机器级 PMU 寄存器的特权读写操作。

此外，为了优化性能监控并减少开销，内核可以利用 SBI 调用来配置 mcounteren 寄存器，从而允许从 supervisor 模式直接读取 HPM 计数器，避免为计数器读取进行重复的 SBI 调用。

> 图 1 展示了 RISC-V 架构与系统软件层之间的关系。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/04.webp)

### 3.3 硬件支持与当前限制

虽然前面的章节概述了通过 Linux 的 perf 子系统和 SBI 接口访问 PMU 计数器的理论框架，但这些机制的实际实现在不同的 RISC-V 硬件上存在显著差异。上述符合规范的接口代表了一种理想情况，但当前 RISC-V 实现的实际情况往往与这种理想情况存在偏差。:mdnice-bold-italic[随着 RISC-V 生态系统的不断发展，硬件供应商对 PMU 功能的实现在完整性和对规范的符合性方面存在差异。这些不一致直接影响了性能分析工具和方法的有效性。]

> 为了全面理解 RISC-V 生态系统中的性能分析挑战，我们对研究人员和更广泛的开发社区可访问的商用 RISC-V 平台进行了系统评估。

**我们的研究重点关注三个具有代表性的内核，它们涵盖了不同的微架构方法和功能集**：:mdnice-bold-italic[SiFive U74、平头哥（T-Head）C910 和 SpacemiT X60]。选择这些内核是基于它们在消费级开发板（分别为 VisionFive II、荔枝派 4A、香蕉派 F3 和 Milk-V Jupyter）中的可用性，这使得我们的发现与广泛的潜在用户相关。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/05.webp)

> 我们的比较分析总结在表 1 中，考察了影响性能分析能力的几个关键维度：

::: block-1
1. 微架构组织：我们评估了每个内核是否实现了顺序或乱序执行模型，这对性能特征和性能分析的复杂性都有显著影响。
2. 向量扩展支持：评估了 RISC-V 向量（RVV）扩展的存在和版本，因为这项能力对于高性能计算工作负载至关重要，并引入了额外的性能监控考虑因素。
3. PMU 计数器能力：我们特别考察了对计数器溢出中断的支持，这对于基于采样的剖析技术至关重要。这一功能在不同实现中的支持程度差异很大，从完全不支持到完全支持不等。
4. 上游内核支持：评估了在主流 Linux 仓库中的集成程度，因为这会影响性能监控工具的可访问性和稳定性。
:::

我们的分析揭示了这些平台之间的显著异质性。

::: block-1
- **平头哥 C910 提供了最先进的微架构**，具有乱序执行和全面的 PMU 支持，但严重依赖供应商特定的内核修改。
- **SpacemiT X60** 虽然支持最新的 RVV 1.0 规范，但仅通过非标准计数器提供有限的 PMU 采样能力。
- :mdnice-bold-italic[SiFive U74 尽管在 Linux 上游集成方面表现更好，但缺乏向量扩展和溢出中断支持，严重限制了传统的性能分析方法。]
:::

X60 上的 PMU 采样 尽管 SpacemiT X60 内核在标准 mcycle 和 minstret 计数器上缺乏对溢出中断的硬件支持，但对供应商提供的内核源代码的检查显示，存在三个支持采样的非标准计数器：`u_mode_cycle`、`m_mode_cycle` 和 `s_mode_cycle`。这些计数器分别跟踪用户模式、机器模式和 supervisor 模式下花费的周期，但遗憾的是没有相应的指令计数器。

在 SpacemiT X60 系统上对 `perf_event_open()` 系统调用进行实验时，我们观察到将这些支持采样的计数器之一配置为领导者组会导致 mcycles 和 minstret 在该组内被同时采样，由领导者的溢出频率触发。

> 为了利用这种观察到的行为，我们开发了 miniperf¹，这是一个包装标准 `perf_event_open()` API 的工具。与标准 perf 工具不同，它会自动对计数器进行分组，并选择合适的支持采样的领导者。

这种技术使得能够收集关键性能指标，例如每周期指令数（IPC），尽管有文档表明必要的底层计数器缺乏直接的采样支持 ²。此外，:mdnice-bold-italic[miniperf 采用了一种不同的硬件兼容性方法；它不使用标准的 perf 事件发现机制，而是仅依赖 CPU 识别寄存器。这种直接的硬件识别能够更稳健地管理受支持的功能和平台特定的解决方法。]

## 四、基于编译器的 Roofline 分析方法

### 4.1 LLVM 背景知识

在深入探讨我们的具体实现之前，先了解一些构成我们方法基础的关键 LLVM 概念会有所帮助。LLVM 是一个编译器基础设施，提供了一系列模块化的编译器和工具链技术。其中两个概念与我们的工作尤为相关：

::: block-1
- **LLVM 中间表示（IR）** 是一种与平台无关的代码表示，位于编译过程中的源代码和机器码之间[3]。LLVM IR 在整个编译流程中作为一种通用格式，使得优化可以独立于源语言或目标架构而应用。这种 IR 采用类 RISC 指令集，但包含更高级别的信息，如类型、显式控制流图和内存操作。:mdnice-bold-italic[对于性能分析而言，LLVM IR 提供了一个理想的观察点，因为它保留了足够的高层结构以识别循环和函数，明确暴露内存操作和算术指令，并且独立于目标架构的特定指令集。]
- **LLVM Pass** 是在代码通过编译流程时对其进行的转换或分析[3]。Pass 可以分析代码或修改代码。Pass 基础设施允许在不修改编译器核心的情况下进行模块化的编译器扩展。常见的 Pass 包括循环分析、死代码消除和指令组合。我们的方法实现了一个自定义的插桩 Pass，它利用现有的分析 Pass 来识别和插桩性能关键区域。
:::

### 4.2 编译器插桩细节

我们的方法实现了一个 Clang 编译器插件，该插件添加插桩（在代码中插入额外代码以收集信息）来收集 Roofline 分析所需的指标，而不依赖于 PMU 计数器。

插桩过程包括几个关键步骤：

::: block-1
1. 循环嵌套识别：我们的 Pass 首先遍历程序中的每个函数，利用 LLVM 的循环分析基础设施来识别循环嵌套。循环嵌套对于性能分析尤为重要，因为它们通常包含主导执行时间的计算核心。
2. 区域提取：对于每个识别出的循环嵌套，我们使用 LLVM 的 RegionInfoAnalysis 来确保该区域具有单入口和单出口点（SESE）。这种特性对于干净地提取和插桩至关重要。然后，CodeExtractor 工具将该区域提取为一个单独的函数。
3. 函数复制：提取的函数被克隆以创建两个版本：原始（未修改）函数和收集性能指标的插桩版本。
4. 调用点修改：原始调用点被修改，加入根据环境变量选择两个函数版本的逻辑。这允许在运行时控制哪些区域被插桩。调用点还被特殊的通知函数包裹，这些函数标记被监控区域的开始和结束：LoopHandle *LH = mperf_roofline_internal_notify_loop_begin(LI);LoopInfo LI{line=42, filename="foo.c", func_name="bar"};if (mperf_roofline_internal_is_instrumented_profiling()) bar_loop0_instrumented(args..., LH);else bar_loop0_outlined(args...);mperf_roofline_internal_notify_loop_end(LH);
5. 指标收集：在插桩的函数版本中，我们在基本块级别插入代码，以统计加载/存储到内存的字节数、整数算术操作和浮点算术操作。
:::

这些统计数据在执行过程中累积，并在程序终止时通过回调函数报告给我们的 miniperf 工具。

### 4.3 运行时分析和 Roofline 模型构建

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/06.webp)

> 实际的 Roofline 模型构建通过两阶段执行方法完成（如图 2 所示）：

1. 基线执行：程序在插桩禁用的情况下运行，以建立基线性能。
2. 插桩执行：程序再次运行，对目标区域启用插桩。

我们的工具协调这些执行并关联结果，计算：

::: block-1
- 插桩和非插桩运行之间的执行时间差异
- 从加载/存储计数导出的内存流量（字节/秒）
- 计算吞吐量（操作/秒）
- 算术强度（操作/字节）
:::

然后，可以将这些指标与硬件能力（峰值计算性能和内存带宽）相对比，以构建 Roofline 模型。:mdnice-bold-italic[由于我们的方法从 LLVM IR 而不是硬件事件导出这些指标，因此该分析在不同的 RISC-V 实现中保持一致。]

### 4.4 优势和局限性

> 这种基于编译器的方法为性能分析提供了几个**优势**：

::: block-1
- 硬件独立性：**通过在 IR 级别操作，我们的方法可以在不同的硬件实现上工作，无论 PMU 支持或特定扩展如何**。:mdnice-bold-italic[添加一个新平台只需为新目标编写一个良好的 LLVM 后端，这是常规编译器开发周期的一部分。]
- 细粒度分析：我们可以针对特定代码区域，而不是整个程序进行分析。
- 指标一致性：指标源自程序行为而非硬件特定事件，提供了跨平台的一致视图。
:::

然而，也存在一些**明显的局限性**：

::: block-1
- 外部函数调用：包含对外部函数（如库调用）的循环无法被完全插桩，因为我们无法跟踪这些函数内部的操作。这尤其影响依赖外部库的代码。
- 运行时开销：插桩会增加显著的开销，使得绝对性能测量不太准确。我们通过两阶段执行方法来缓解这一问题。
- 编译器优化：插桩后的优化可能会改变 IR 级别指标与实际硬件行为之间的关系。我们通过在优化流程的后期应用我们的 Pass 来解决这一问题。
- 确定性执行：该方法假设程序在多次运行中具有确定性行为，这可能不适用于所有应用程序。
:::

尽管存在这些局限性，我们的方法仍能为 RISC-V 平台上的应用程序性能特性提供有价值的见解，在传统基于 PMU 的分析不可用或不可靠的情况下尤其如此。:mdnice-bold-italic[这种方法的硬件无关性 使其在快速发展的 RISC-V 生态系统中具有重要价值。]

## 五、评估

### 5.1 热点分析

> 识别 CPU 热点——处理器花费大部分时间的代码段——是性能优化的关键步骤。虽然存在各种分析技术，但 Brendan Gregg 发明的火焰图（Flame Graphs）为理解 CPU 使用模式提供了直观且强大的可视化方式。

#### 火焰图

**火焰图是对被分析软件的可视化，通过在一段时间内对堆栈跟踪进行采样创建**。火焰图的 x 轴表示堆栈配置文件的数量，堆栈帧按字母顺序排序以最大化合并，而非表示时间的流逝。y 轴表示堆栈深度，从下到上递增。图中的每个矩形对应一个函数（一个堆栈帧）。矩形的宽度与该函数在采样堆栈中出现的频率直接成正比——更宽的帧表示该函数在 CPU 上运行的时间更长，或者是在 CPU 上运行的调用堆栈的一部分。帧的上边缘表示当前在 CPU 上执行的函数，而其下方的帧表示其父调用者（其祖先）。这种层次化的可视化使开发人员能够快速定位最耗时的代码路径[2]。

**火焰图的主要价值在于其能够提供对 CPU 使用情况的快速、全面且深入的视图**。它们将可能令人难以承受的大量分析数据浓缩为单个可读图像，使识别消耗最多 CPU 资源的调用路径变得更加容易。这对于将优化工作导向最有影响的代码区域非常宝贵。

#### miniperf

> 我们的工具 miniperf 有助于使用 CPU 周期或已退役指令作为采样指标生成火焰图。

虽然基于周期的火焰图直接表示 CPU 时间，但基于已退役指令的火焰图提供了一个有价值的代理指标，**在不同平台上特别有用，可用于识别未被充分优化且值得程序员或编译器开发人员关注的代码段**。

例如，考虑一个预期会被向量化的函数；如果已退役指令的火焰图显示该函数的帧明显更宽（例如，处理的标量指令是等效向量化版本的 8 倍），则强烈表明向量化方案不佳或完全缺乏向量化。

火焰图使这种比较分析在视觉上直观——通常就像比较两张图像一样简单。:mdnice-bold-italic[至关重要的是，miniperf 能够对周期和已退役指令进行采样，即使在硬件支持有限或非标准的平台上] （如在 SpacemiT X60 上展示的解决方法），这使开发人员能够在成熟硬件平台发布之前进行这些重要的优化和分析，加速软件就绪。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/07.webp)

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/08.webp)

我们的工具针对 sqlite3 基准测试（取自 LLVM 测试套件）的结果，如火焰图（图 3）和热点摘要（表 2）所示，清楚地展示了当前的性能状况。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/09.webp)

虽然 x86 平台在给定任务中可能执行更多指令，但其微架构效率（IPC 达到 3.38）显著超过 SpacemiT X60 核心（IPC 为 0.86）。**这凸显了 RISC-V 系统上存在相当大的性能差距和优化机会。**

### 5.2 Roofline 分析

> 除了直接识别热点外，理解处理器的计算能力与其内存子系统性能之间的相互作用对于全面优化至关重要。

- Roofline 模型为此提供了一个富有洞察力的可视化框架，提供了一种简单的方法来表征应用程序性能并识别主要瓶颈。
- Roofline 模型的主要优势包括其能够直观地描述应用程序是受内存带宽限制还是受处理器峰值计算吞吐量限制，从而有效地指导优化工作。它将实现的性能（如 GFLOP/s）与算术强度（每访问 1 字节内存所执行的操作数）相关联，提供由硬件能力定义的明确上限。

为了展示我们基于编译器的 Roofline 分析的实际应用，我们将使用以下代码核心进行详细检查：

```
for (int ii = 0; ii < n; ii += TILE_SIZE) {  for (int jj = 0; jj < n; jj += TILE_SIZE)    for (int kk = 0; kk < n; kk += TILE_SIZE)      for (int i = ii; i < ii + TILE_SIZE && i < n; i++)        for (int j = jj; j < jj + TILE_SIZE && j < n; j++) {          float sum = C[i * n + j];          for (int k = kk; k < kk + TILE_SIZE && k < n; k++)            C[i * n + j] = sum;            sum += A[i * n + k] * B[k * n + j];        }}
```

对于我们的 Roofline 模型实验评估，所有基准测试都使用 Clang 19 编译，并带有 -O3 优化标志。

### 5.3 x86 平台性能分析

对于 x86 平台，我们通过 -mavx2 标志专门启用了 AVX2 指令，而对于 RISC-V，我们使用 `-march=rv64gcv` 目标 RV64GCV 配置文件。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/10.webp)

> 图 4 展示了在 x86 平台上对一个代表性核心的比较 Roofline 分析，对比了我们的 miniperf 工具与 Intel Advisor 获得的结果。

我们在 miniperf 中实现的基于编译器的插桩方法报告该核心的性能为 34.06 GFLOP/s。该数值与基准测试自行报告的 33.0 GFLOP/s **性能非常接近，表明与应用程序自身的测量结果具有良好的一致性**。相比之下，Intel Advisor 报告的性能更高，为 47.72 GFLOP/s。

这种差异主要源于不同的方法：

- miniperf 在 LLVM IR 级别直接对生成的代码进行插桩以统计操作，
- 而 Intel Advisor 通常依赖于硬件性能计数器的动态分析。

:mdnice-bold-italic[miniperf 的测量结果与基准测试自行报告的值之间的轻微差异可归因于我们低级循环插桩的固有开销]；任何围绕插桩循环的用户代码测量都必然包含这种插桩开销，尽管如观察到的那样，这种影响并不显著。

对于所呈现的算术强度计算，**我们目前专注于暴露给 L1 缓存的操作**。更全面的内存层次结构分析需要深入的、特定于平台的知识，这在 RISC-V 硬件多样化且不断发展的背景下具有挑战性。还值得注意的是，对于 x86 Roofline 图（图 4），性能上限（屋顶）直接采用自 Intel Advisor，我们的 miniperf 导出的数据点被映射到这个已建立的模型上。

![文章插图](/articles/2025-08-22-risc-v-pmu-llvm-roofline-e4fd3619/images/10.webp)

### 5.4 RISC-V 平台性能分析

> 转向 SpacemiT X60 RISC-V 平台，由于目前难以获得所有内存级别的精确、详细硬件规格，我们结合已建立的基准测试和理论峰值来构建其 Roofline 模型。

对于内存带宽上限，我们使用了 Olaf Bernstein 的 memset 基准测试结果[3]，该结果表明该核心的峰值性能约为 3.16 字节/周期。在 1.6 GHz 的标称频率下，这转化为约 4.7 GB/s 的理论 DRAM 内存带宽（3.16 字节/周期 ×1.6 GHz）。

:mdnice-bold-italic[对于计算上限，我们考虑了理论上可实现的峰值：假设每周期退役 2 条指令（这是顺序核心的常见设计目标），以及 8 元素单精度浮点向量能力] （根据具有 256 位 VLEN 的 RVV 1.0），这产生的峰值为 $2 \text{ IPC} \times 8 \text{ SP FLOP/vector instruction} \times 1.6 \text{ GHz} = 25.6 \text{ GFLOP/s}$（假设每周期一条向量 FMA 或等效指令）。

在这些理论上限下，我们的核心在 SpacemiT X60 上实现了 **1.58 GFLOP/s 的性能**。**这一结果性能显著低于理论计算和内存带宽能力，表明存在相当大的改进空间**。这凸显了编译器开发人员有机会增强针对该特定核心的代码生成和向量化，性能工程师有机会进一步调整应用程序，以及硬件供应商有机会改进微架构效率或提供更清晰的性能表征。:mdnice-bold-italic[即使在详细的硬件计数器分析面临挑战时，miniperf 能够提供这些与硬件无关的操作指标，这为新兴平台上的此类性能特性和优化潜力提供了独特的见解。]

## 六、结论

> 我们的评估成功展示了我们所提出方法的实际应用，展示了在有限的 RISC-V 硬件上进行有效的 PMU 分析，以及我们基于编译器的、与硬件无关的 Roofline 分析的实用性。

通过使用火焰图进行热点分析和比较 Roofline 建模，我们的 miniperf 工具有效地突出了关键性能特性和大量优化机会。

最终，这些评估证实，:mdnice-bold-italic[即使在缺乏成熟硬件支持的情况下，我们的方法也能为开发人员在复杂且不断发展的 RISC-V 性能领域提供有价值的、可操作的见解。]

## 参考文献

1. Domingos, J.M., Tomas, P., Sousa, L.: Supporting RISC-V performance counters through performance analysis tools for linux (perf). arXiv preprint arXiv:2112.11767 (2021)
2. Gregg, B.: Flame Graphs. https://www.brendangregg.com/flamegraphs.html, accessed: 2025-04-16
3. Lattner, C., Adve, V.: LLVM: A compilation framework for lifelong program analysis & transformation. In: International symposium on code generation and optimization, 2004. CGO 2004. pp. 75–86. IEEE (2004)
4. Lo, Y.J., Williams, S., Van Straalen, B., Ligocki, T.J., Cordery, M.J., Wright, N.J., Hall, M.W., Oliker, L.: Roofline model toolkit: A practical tool for architectural and program analysis. In: High Performance Computing Systems. Performance Modeling, Benchmarking, and Simulation: 5th International Workshop, PMBS 2014, New Orleans, LA, USA, November 16, 2014. Revised Selected Papers 5. pp. 129–148. Springer (2015)
5. Morgado, J., Sousa, L., Ilic, A.: CARM Tool: Cache-Aware Roofline Model Automatic Benchmarking and Application Analysis. In: 2024 IEEE International Symposium on Workload Characterization (IISWC). pp. 68–81. IEEE (2024)
6. Mou, C.Y., Hsiao, C.C., Chou, J.: Top-Down Microarchitecture Analysis Approximation Based on Performance Counter Architecture for SiFive RISC-V Processors. In: SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis. pp. 1666–1675. IEEE (2024)
7. Yasin, A.: A top-down method for performance analysis and counters architecture. In: 2014 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). pp. 35–44. IEEE (2014)
