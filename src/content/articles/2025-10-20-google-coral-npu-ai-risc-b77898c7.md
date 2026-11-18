关建词：Coral NPU、**边缘AI**、**全栈开源平台**、:mdnice-bold-italic[RISC-V]、超低功耗、硬件强化隐私

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/01.webp)

- https://research.google/blog/coral-npu-a-full-stack-platform-for-edge-ai
- https://developers.google.com/coral
- **https://github.com/google-coral/coralnpu**
- 5000 字，阅读需 18 分钟，:mdnice-bold-italic[播客 14 分钟]

相关推荐

- [Triton 内核在 RISC-V 平台基于 MLIR 的编译与优化探索](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890238&idx=1&sn=cf15b6e26b823d3d5a3921f852c0cf43&scene=21#wechat_redirect)
- [卷积提速 55 倍！为 TinyML 打造的 FPGA 全栈开源框架 CFU Playground，敏捷探索轻量加速器设计空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888011&idx=1&sn=3ea5446297b91cdd15378baa2ed83125&scene=21#wechat_redirect)
- [为 TinyAI 打造 RISC-V 架构高性能 GPU 方案 e-GPU：28mW功耗内实现 3.1×能耗降低！整体方案开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890988&idx=1&sn=dfefb10e95f72035e4dd940638d460d3&scene=21#wechat_redirect)

本文将介绍 Coral 神经网络处理器，这是一个**全栈、开源** 的平台，旨在解决**核心性能、设备碎片化以及隐私保护** 三大挑战——这些挑战目前正限制着低功耗边缘设备和可穿戴设备实现功能强大、持续运行的人工智能（AI）应用。

::: block-1
生成式人工智能（Generative AI，指能够生成文本、图像、音频等内容的 AI 技术，如 ChatGPT、DALL·E 等）:mdnice-bold-italic[从根本上重塑了我们对技术的期待。]

- 我们已经见证了大规模云原生模型在创作、推理和辅助等方面展现出的惊人能力。
- 然而，下一次重大技术飞跃不仅在于让云模型变得更强大，更在于将其智能直接嵌入我们身边的个人环境中。
:::

要让人工智能真正发挥辅助作用——比如主动帮助我们规划日常、实时翻译对话，或理解我们所处的物理环境——它:mdnice-bold-italic[必须运行在我们佩戴和携带的设备上。]

> 这就带来了一个核心挑战：**如何将环境感知型人工智能** （ambient AI，指能够感知周围环境并主动提供服务的 AI）**嵌入受电池容量限制的边缘设备**，使其摆脱对云的依赖，从而实现真正私密、全天可用的辅助体验。

要实现从云端到个人设备的跨越，我们必须解决三个关键问题：

| 类别 | 说明 |
| --- | --- |
| 性能差距 | 复杂、先进的机器学习（ML，人工智能的一个分支，专注于让计算机通过数据学习并改进）模型对计算能力的需求极高，远超边缘设备有限的功耗、散热和内存预算。 |
| 碎片化成本 | 为各类专有处理器（由不同厂商开发、仅适用于特定设备的处理器）编译和优化机器学习模型难度大、成本高，导致设备间性能难以保持一致。 |
| 用户信任缺失 | 要真正发挥作用，个人人工智能必须将个人数据和环境信息的隐私与安全放在首位。 |

> 今天，我们正式推出Coral NPU[1]。这一全栈平台基于我们此前在Coral[2]项目中的成果构建，为硬件设计师和机器学习开发者提供所需工具，助力他们打造下一代私密、高效的边缘人工智能设备。

:mdnice-bold-italic[Coral NPU 由谷歌研究院（Google Research）和谷歌深度思维（Google DeepMind）联合设计，是一种“AI 优先”的硬件架构，旨在支持下一代超低功耗、持续运行的边缘人工智能应用]。

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/02.webp)

它提供统一的开发者体验，让环境感知等应用的部署更加简便；专为可穿戴设备设计，可在最小化电池消耗的同时实现全天 AI 功能，同时也可根据更高性能的使用场景进行配置。

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/03.webp)

::: image-caption
交流加群请在 NeuralTalk 公众号后台回复：加群
:::

## 关键问题

### 问题1：Coral NPU 如何突破现有架构壁垒，平衡厂商差异化与生态统一？

> Google提到Coral NPU以“解决边缘AI碎片化问题”为核心目标之一，且依赖RISC-V架构的开放性与开源工具链（如IREE、MLIR）实现跨硬件一致性，但当前边缘AI领域已存在ARM架构生态、各厂商专用NPU（如高通Hexagon、苹果Neural Engine）等成熟体系，其**如何突破现有架构生态的既有壁垒**，确保主流芯片厂商、开发者愿意放弃既有技术路径转向适配？若仅靠“开源”，能否平衡厂商对硬件差异化竞争的需求与生态统一的目标？

Coral NPU主要通过**“全栈价值绑定+生态先发合作+差异化技术定位”** 应对现有架构壁垒，并尝试平衡厂商需求与生态统一：

- **突破壁垒层面**：
- **平衡差异化与统一层面**：Google 提到其架构“开放可扩展”——SoC设计者可修改基础设计或直接使用预配置NPU，既保留厂商在硬件参数（如算力升级、多传感器集成）上的差异化空间，又通过统一的软件工具链和RISC-V指令集标准，确保不同厂商产品能兼容同一套开发者生态，避免因硬件差异导致的软件碎片化。

### 问题2：矩阵执行单元延迟发布，Coral NPU 会偏离指标且削弱开发者信心吗？

> Google明确Coral NPU的矩阵执行单元（负责核心神经网络MAC运算）仍处于开发中，需“今年晚些时候才在GitHub发布”，但同时强调其“目标是成为首个支持可穿戴设备运行LLM的开放低功耗NPU”。那么在核心计算单元缺失的现阶段，Coral NPU原型或早期版本在运行Transformer类模型（LLM基础架构）时，是否会依赖标量/向量单元“代偿”，导致实际功耗、算力效率偏离“超低功耗”“512 GOPS”的宣称指标？这一关键组件的延迟发布，是否会削弱开发者早期适配信心，反而加剧边缘AI生态的短期碎片化？

矩阵执行单元延迟发布的影响**存在“短期妥协但长期可控”的特点**，核心逻辑如下：

- **关于“代偿与指标偏离”**：**Google明确Coral NPU基础设计已实现“数毫瓦功耗下512 GOPS性能”**，这一指标应基于现有标量核心（RISC-V前端，负责数据流转与传统CPU功能）和向量执行单元（RVV v1.0 SIMD，支持大规模数据并行运算）达成——对于当前边缘AI主流场景（如关键词 spotting、简单图像检测），向量单元的SIMD能力可覆盖多数轻量化模型的运算需求，暂无需矩阵单元“代偿”；而**针对Transformer类LLM，也提到“正与Gemma团队协作优化”**，说明现阶段或仅针对小参数LLM（如Gemma-2B及以下）进行适配，标量+向量单元的组合或可满足基础推理需求，暂不会大幅偏离“超低功耗”目标。
- **关于“开发者信心”**：Google强调已同步发布“文档和工具”（https://developers.google.com/coral），开发者可基于现有架构组件（标量+向量单元）先行开发轻量化边缘AI应用（如情境感知、音频转录），并非完全依赖未发布的矩阵单元；同时，Synaptics的量产产品落地已提供“实际硬件验证案例”，可降低开发者对架构成熟度的顾虑，:mdnice-bold-italic[短期虽可能限制复杂模型（如大尺寸LLM）的适配，但通过“分阶段开放+量产案例支撑”，可减少对生态建设的冲击。]

## 本文目录

- 关键问题
  - 问题1：Coral NPU 如何突破现有架构壁垒，平衡厂商差异化与生态统一？
  - 问题2：矩阵执行单元延迟发布，Coral NPU 会偏离指标且削弱开发者信心吗？
- Coral 神经网络处理器：“AI 优先”的架构
- Coral NPU 特性
- 统一的开发者体验
- 目标应用场景
- 硬件级隐私保护
- 构建生态系统
- 解决边缘计算的核心难题

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/04.webp)

::: image-caption
交流加群请在 NeuralTalk 公众号后台回复：加群
:::

## Coral 神经网络处理器：“AI 优先”的架构

> 为低功耗边缘设备开发应用的开发者面临一个根本性的权衡：**选择通用中央处理器还是专用加速器？**

- 通用 **CPU** 具备关键的**灵活性和广泛的软件支持**，但缺乏针对高要求机器学习工作负载的领域专用架构，因此**性能较低且功耗不经济；**
- 相反，**专用** 加速器虽然机器学习**效率高，但灵活性差**、编程难度大，且不适合处理通用任务。

这种硬件层面的问题因高度碎片化的软件生态系统而进一步加剧。**由于 CPU 和机器学习模块（处理 AI 计算的硬件单元）采用截然不同的编程模型，开发者往往被迫使用专有编译器和复杂的命令缓冲区** （用于存储硬件执行指令的数据区域）。这:mdnice-bold-italic[不仅提高了学习门槛，还难以结合不同计算单元的独特优势]。因此，行业目前缺乏一种成熟的低功耗架构，无法轻松、高效地支持多种机器学习开发框架。

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/05.webp)

::: image-caption
overview arch alpha
:::

> Coral NPU 架构通过颠覆传统芯片设计理念，直接解决了这一问题。

它将机器学习矩阵引擎（专门处理 AI 中矩阵运算的核心组件，矩阵运算为 AI 模型的主要计算任务）置于比标量计算更优先的地位，从芯片层面优化架构以适配 AI 需求，打造出一个专为提升效率、实现设备端推理而设计的平台。

作为一套完整的参考神经网络处理器架构[3]，**Coral NPU 为下一代高能效、机器学习优化的系统级芯片（SoC）[4]提供了基础模块**。该架构基于一组符合RISC-V ISA[5]构建，旨在最小化功耗，非常适合持续运行的环境感知应用。其:mdnice-bold-italic[基础设计的运算性能可达 512 GOPS]，而:mdnice-bold-italic[功耗仅为几毫瓦（mW）]，能够为边缘设备、可听设备（如无线耳机）、增强现实（AR）眼镜和智能手表提供强大的设备端 AI 能力。

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/06.webp)

::: image-caption
Coral NPU 生态系统全景图，展示面向系统级芯片（SoC）设计师和机器学习（ML）开发者的端到端技术栈
:::

基于 RISC-V 的**开放式可扩展架构**，允许开发者根据需求修改和扩展的硬件设计，**为系统级芯片（SoC）设计师提供了灵活性** ——他们既可以修改基础设计，也可将其用作预配置的NPU。Coral NPU 架构包含以下组件：

| 模块名称 | 详细说明 |
| --- | --- |
| *标量核心* | 一款轻量级、支持 C 语言编程的 RISC-V 前端（硬件的输入接口部分），负责管理数据流向后端核心（硬件的运算执行部分）的过程，采用简单的**“运行至完成”（run-to-completion）模型** （一种硬件执行模式，指令一旦开始执行便会持续到完成，无需中断），以实现超低功耗和处理传统 CPU 功能。 |
| *向量执行单元* | 一款功能完善的单指令多数据**协处理器** （协助主处理器完成特定任务的辅助处理器），符合 RISC-V 向量指令集（RVV）v1.0 标准，能够对大规模数据集执行并行操作。 |
| *矩阵执行单元* | 一款高效的量化外积乘积累加（MAC，AI 模型中最核心的计算操作，用于实现权重与输入数据的乘法和累加）引擎[6]，专为加速神经网络基础运算而设计。注：矩阵执行单元仍在开发中，将于今年晚些时候在 GitHub[7] 上发布。 |

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/07.webp)

::: image-caption
可视化展示从传统设计到 Coral NPU 架构的转变
:::

## Coral NPU 特性

Coral NPU 提供以下顶级特性集：

- RV32IMF_Zve32x RISC-V 指令集（具体为 rv32imf_zve32x_zicsr_zifencei_zbb）
- 面向应用程序和操作系统内核的 32 位地址空间
- 四级处理器，按序分发，乱序退出
- 四通道标量、双通道向量分发
- 128 位 SIMD、256 位（未来）流水线
- 8 KB ITCM 存储器（用于指令的紧密耦合存储器）
- 32 KB DTCM 存储器（用于数据的紧密耦合存储器）
- 两种存储器均为单周期延迟静态随机存取存储器（SRAM），比高速缓冲存储器（cache memory）效率更高
- AXI4总线接口，同时具备管理器和从属器功能，用于与外部存储器交互，并允许外部中央处理器（CPU）配置珊瑚神经网络处理器（Coral NPU）

## 统一的开发者体验

> Coral NPU 架构是一个简洁、支持 C 语言编程的目标平台，可与IREE[8]（一种用于高性能机器学习部署的编译器框架）和TFLM[9]（TensorFlow Lite for Microcontrollers，适用于微控制器的轻量级 TensorFlow 框架）等现代编译器无缝集成。

这使得它能够轻松支持TensorFlow[10]、JAX（谷歌开发的机器学习框架，适用于高性能数值计算）[11]和PyTorch[12]等机器学习框架。

Coral NPU 整合了一套全面的软件工具链，包括专为 TensorFlow 设计的 TFLM 编译器等专用解决方案，以及通用的MLIR[13]编译器、C 语言编译器、自定义内核和模拟器。这为开发者提供了灵活的开发路径。

::: block-1
例如，来自 JAX 等框架的模型：

- 首先会通过StableHLO[14]（一种稳定的高阶优化器表示，用于不同 ML 框架间的模型转换）方言（dialect，MLIR 中用于表示特定领域代码的格式）导入为 MLIR 格式；
- 随后，该中间文件会被输入 IREE 编译器，编译器通过硬件专用插件识别 Coral NPU 的架构；
- 接下来，编译器会执行“渐进式下探”（progressive lowering）——这是一个关键的优化步骤，代码会通过一系列方言逐步转换，不断接近硬件的原生指令；
- 优化完成后，工具链会生成最终的精简二进制文件（计算机可直接执行的文件格式），可在边缘设备上高效运行。
:::

这套基于**行业标准** 的开发者工具有助于简化机器学习模型的编程流程，并**确保在不同硬件目标平台上实现一致的开发体验**。

![文章插图](/articles/2025-10-20-google-coral-npu-ai-risc-b77898c7/images/08.webp)

::: image-caption
Coral NPU 编译器工具链，展示从机器学习（ML）模型创建、优化、编译到设备端部署的完整流程
:::

:mdnice-bold-italic[Coral NPU 的协同设计过程聚焦于两个关键领域]：

::: block-1
- 第一，该架构能高效加速当前设备端视觉和音频应用中主流的编码器架构；
- 第二，我们正与Gemma（谷歌开发的开源轻量级大语言模型）[15]团队紧密合作，**优化 Coral NPU 以支持小型 Transformer 模型**，确保该加速器架构能够支持下一代边缘生成式人工智能应用。
:::

这种双重聚焦意味着，:mdnice-bold-italic[Coral NPU 有望成为首个开源、基于标准、低功耗的NPU，专为在可穿戴设备上部署大语言模型而设计]。对开发者而言，这提供了一条经过验证的统一路径，可**在最小功耗下实现当前及未来模型的最高性能部署。**

## 目标应用场景

Coral NPU 旨在支持**超低功耗、持续运行** 的**边缘人工智能应用**，:mdnice-bold-italic[尤其专注于环境感知系统]。 其**核心目标是在可穿戴设备、智能手机和物联网** 设备上实现全天 AI 体验，同时最大限度减少电池消耗。

潜在应用场景包括：

::: block-1
- 情境感知：检测用户活动（如行走、跑步）、距离感应或环境状态（如室内/室外、移动中），以启用“免打扰”模式或其他情境感知功能。
- 音频处理：语音检测、关键词识别（如唤醒词“Hey Siri”）、实时翻译、转录以及基于音频的无障碍功能。
- 图像处理：人体和物体检测、面部识别、手势识别以及低功耗视觉搜索。
- 用户交互：通过手势、音频信号或其他传感器驱动的输入方式实现设备控制。
:::

## 硬件级隐私保护

> Coral NPU 的核心原则之一是通过硬件级安全保障构建用户信任。

我们的架构正被设计为支持CHERI（能力硬件增强 RISC 指令集，一种用于提升计算机内存安全的技术）[16]等新兴技术，该技术可提供细粒度的内存级安全和可扩展的软件隔离。

::: block-1
- **内存级安全**：确保对内存访问进行精确控制，防止非法访问
- **可扩展的软件隔离**：隔离保证了将软件拆分为独立安全区域，防止一个区域的漏洞影响其他区域
:::

通过这种方式，我们希望能够将敏感 AI 模型和个人数据隔离在硬件级安全沙箱中，降低基于内存的攻击风险。

## 构建生态系统

> 开源硬件项目的成功离不开强大的合作伙伴支持。为此，我们正与新突思科技（Synaptics）[17]开展合作——新突思科技是我们的**首个战略芯片合作伙伴**，也是嵌入式计算、无线连接和物联网多模态传感领域的领先企业。

今日，在新突思科技的技术日活动上，该公司发布了全新的Astra™ SL2610[18]系列AI 原生物联网处理器[19]。该产品系列搭载其Torq™ NPU[20]子系统，这是业内**首个基于 Coral NPU 架构的量产级实现方案**。该NPU的设计支持 Transformer 模型，并兼容动态算子，使开发者能够为消费级和工业级物联网系统构建面向未来的边缘人工智能应用。

这一合作体现了我们对 “统一开发者体验” 的承诺。新突思科技的 Torq™[21] 边缘人工智能平台基于开源编译器和运行时构建，而这些编译器和运行时又以 **IREE 和 MLIR** 为基础。**此次合作是朝着构建智能、情境感知设备共享开源标准迈出的重要一步。**

## 解决边缘计算的核心难题

> 通过 Coral NPU，我们正在为个人人工智能（AI）的未来构建基础层。

我们的目标是通过提供一个**行业通用的开源安全平台，培育充满活力的生态系统**。这将助力开发者和芯片厂商**摆脱当前碎片化** 的格局，围绕边缘计算的共享标准展开协作，进而加速创新进程。

参考资料

[1] https://developers.google.com/coral: *https://developers.google.com/coral*

[2] https://gweb-coral-full.uc.r.appspot.com/: *https://gweb-coral-full.uc.r.appspot.com/*

[3] coral/guides/architecture: *https://developers.google.com/coral/guides/architecture*

[4] SoC: *https://en.wikipedia.org/wiki/System_on_a_chip*

[5] riscv.org/specifications: *https://riscv.org/specifications/ratified/*

[6] MAC: *https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation*

[7] github.com/google-coral/coralnpu: *https://github.com/google-coral/coralnpu*

[8] IREE: *https://iree.dev/*

[9] TensorFlow Lite for Microcontrollers: *https://github.com/tensorflow/tflite-micro*

[10] TensorFlow: *https://www.tensorflow.org/*

[11] JAX: *https://docs.jax.dev/en/latest/*

[12] PyTorch: *https://pytorch.org/*

[13] mlir.llvm.org: *https://mlir.llvm.org/*

[14] StableHLO: *https://openxla.org/stablehlo*

[15] Gemma: *https://deepmind.google/models/gemma/*

[16] CHERI: *https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/*

[17] synaptics: *https://www.synaptics.com/*

[18] Astra™ SL2610: *https://www.synaptics.com/sl2610-product-line*

[19] AI 原生物联网处理器: *https://www.synaptics.com/sl2610-product-line*

[20] torq-github: *https://www.synaptics.com/torq-github*

[21] torq-github: *https://www.synaptics.com/torq-github*
