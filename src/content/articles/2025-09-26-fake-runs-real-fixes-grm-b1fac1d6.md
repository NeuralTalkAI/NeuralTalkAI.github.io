关键词：xPU-Shark、ML 加速器、微架构模拟、机器码级性能分析、DMA 优化、LLM 性能调优

> 生成式 AI 预计 2032 年达万亿级市场，仅微软 Bing AI 推理基础设施成本就达 40 亿美元，**小幅性能损耗便会引发巨额损失**，其性能优化直接关系到大规模部署的成本效益。

但现有工具存在明显局限：PMU 依赖型工具粒度粗，无法捕捉机器码与硬件的交互；PC 采样难以定位 stalls 根源且需硬件支持；二进制插桩需重编译，不适用于生产环境。

在硬件设计与机器学习加速器性能优化领域，GRM 是 黄金参考模型（Golden Reference Model） 的缩写，它是一种在硬件开发阶段用于验证微架构行为的权威模拟器。通常由芯片厂商在硅前验证（即芯片量产前）开发。

GRM 通过软件精确模拟硬件的指令执行逻辑、流水线行为、缓存命中机制、DMA 调度规则等核心微架构特性，确保硬件设计符合预期规范。例如，在 ML 加速器开发中，GRM 可模拟矩阵运算单元（MXU）、向量内存（VMEM）等组件的协同工作模式，验证设计的正确性。

![文章插图](/articles/2025-09-26-fake-runs-real-fixes-grm-b1fac1d6/images/01.webp)

- Fake Runs, Real Fixes – Analyzing xPU Performance Through Simulation
- https://arxiv.org/abs/2503.14781
- 本文 14552 字，阅读需 50 分钟，播客 20 分钟
