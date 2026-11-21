关键词：**效率**、:mdnice-bold-italic[调度编排]、**RISC-V**、**AI 推理**、SWaP-C

> *异构计算系统正越来越依赖轻量化的**控制平面核心** ——负责调度指挥的小核心，来完成加速器与可重构逻辑阵列之间的数据搬运、同步与任务调度。RISC-V 已成为被广泛采用的:mdnice-bold-italic[编排底层架构，用于统一的调度指挥指令集架构，覆盖 FPGA 软核与硬件级加速器类核心。]*

传统指标如 FLOPS（每秒浮点运算次数）、TOPS/W（每瓦每秒万亿次运算）、单运算能耗等，主要用于表征算术运算能力，却对编排效率的反映十分有限[1]，**而编排行为往往直接决定系统的实际持续吞吐率。随着异构系统向更紧密的集成方向发展，控制平面瓶颈依然存在，这一局限性表现得愈发突出。**

![文章插图](/articles/2026-03-22-riscbench-fpga-fpga-risc-v-ac266bbb/images/01.webp)

- **RISCBench: Benchmarking RISC-V Orchestration Efficiency in FPGA and FPGA-Like Computing Engines**
- https://arxiv.org/pdf/2603.06580
- www.riscbench.com
- 1500 字，阅读 5 分钟，**播客 5 分钟**

相关推荐

- **[基于 FPGA 的稀疏注意力加速器 FAST-Prefill：相比 NVIDIA A5000实现 2.5 倍TTFT加速与 4.5 倍能效提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898021&idx=1&sn=e821816258c60284f6f128d1530ef252&scene=21#wechat_redirect)**
- **[MICRO'25 StreamTensor：迭代张量驱动的 LLM 数据流加速器，从PyTorch到FPGA全流程优化实现 0.76x FPGA 延迟与 1.99x GPU 能效突破](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896673&idx=1&sn=c4bb2d42f2cdefa9f557bb4d2ddfa208&scene=21#wechat_redirect)**
- **[卷积提速 55 倍！为 TinyML 打造的 FPGA 全栈开源框架 CFU Playground，敏捷探索轻量加速器设计空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888011&idx=1&sn=3ea5446297b91cdd15378baa2ed83125&scene=21#wechat_redirect)**

> 来自 Flapmax、Altera 和佛罗里达大学的研究团队提出 RISCBench，一套轻量化基准测试套件与开放方法论，旨在量化编排效率。

该方法论的核心组件是**持续瞬时吞吐率（Sustained Instantaneous Throughput, SIT）** 指标，即衡量系统长时间稳定高效运行的能力，而非瞬间峰值，它累计高效执行阶段对实际持续吞吐率的贡献。

::: block-1
与面向峰值性能的描述指标不同，**SIT 反映在真实编排场景下，协同调度、任务规划与数据放置如何影响高效执行的持续性**。

通过聚焦持续运行表现而非瞬时峰值速率，RISCBench 与 SIT 能够暴露传统峰值指标无法捕捉的编排低效问题，从而实现对异构 FPGA 与加速器级计算系统更有意义的评估。
:::

![文章插图](/articles/2026-03-22-riscbench-fpga-fpga-risc-v-ac266bbb/images/02.webp)

## 实验概览

> RISCBench 基于 FPGA 上的 Nios-V 软核 RISC-V 核心完成原型开发，以验证其在可重构环境中的集成能力[2]，并在搭载大量专用硬件级 RISC-V 控制核心的加速器级平台上完成评估[3]。

研究采用驻留在片上 SRAM 的分块矩阵核，以最小化外部内存干扰并暴露真实编排行为。执行轨迹如图 1 显示，执行初期吞吐率接近聚合峰值，随后因同步开销增加与内存驻留状态切换累积而出现性能下降。

![文章插图](/articles/2026-03-22-riscbench-fpga-fpga-risc-v-ac266bbb/images/03.webp)

::: image-caption
图 1：展示执行时间与归一化吞吐率的关系：执行初期吞吐率接近聚合峰值，进入高效执行阶段后，因同步开销累积、内存驻留状态切换及带宽竞争，吞吐率逐步下降。全片上 SRAM 驻留时达峰值性能，协同调度复杂度上升会持续拉低持续性能。该曲线直观暴露编排瓶颈对实际吞吐的制约，证明传统峰值指标无法反映真实运行效率，SIT 可有效捕捉高效执行累积贡献，精准衡量 RISC‑V 编排效率，为异构计算系统评估提供关键依据。
:::

全片上驻留条件下可观测到峰值吞吐率，而随着协同调度与带宽竞争加剧，持续性能逐步降低。**上述观测表明，随着异构集成向更紧耦合的加速器架构演进，编排行为对实际吞吐率的影响愈发关键**。

::: block-1
SIT 提供独立于平台的持续执行行为描述指标，可与峰值指标互补，适用于包括 AI 推理在内的各类应用。为支持可复现研究与广泛应用，RISCBench 已通过 www.riscbench.com 开源发布，欢迎芯片架构、EDA 研究与系统领域同仁共同贡献。
:::

注：但 NeuralTalk 看了下目前还没发布。

![文章插图](/articles/2026-03-22-riscbench-fpga-fpga-risc-v-ac266bbb/images/04.webp)

::: image-caption
www.riscbench.com
:::

## 参考文献

- [1] Norman P. Jouppi, Cliff Young, Nishant Patil, et al. 2017. In-Datacenter Performance Analysis of a Tensor Processing Unit. ISCA ’17. ACM, New York, NY, USA, 1–12. https://doi.org/10.1145/3079856.3080246
- [2] Xiangyi Yi, Mengming Li, Joren Dumoulin, et al. 2025. OpenGeMM: Lightweight RISC-V Controlled Efficient GEMM Accelerator Generator with Tightly-Coupled Memory. ASPDAC ’25. New York, NY, USA, 1055–1061. https://doi.org/10.1145/3658617.3697652
- [3] Tenstorrent. 2024. Wormhole: RISC-V Based Scalable AI Accelerator Architecture. https://tenstorrent.com/en/hardware/wormhole
