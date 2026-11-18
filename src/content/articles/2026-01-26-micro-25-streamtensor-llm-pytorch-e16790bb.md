关键词：:mdnice-bold-italic[StreamTensor]、**数据流加速器**、LLM 推理、:mdnice-bold-italic[迭代张量]、**编译优化**

> 一种编译器框架如何通过创新的类型系统和分层设计空间探索，自动化生成高性能、低能耗的数据流加速器

在大型语言模型（LLM）推理任务中，**内存访问** 已成为制约性能的瓶颈。传统的冯·诺依曼架构在执行自回归解码时，频繁的中间结果写回与读取导致严重的“内存墙”问题。

::: block-1
数据流架构因其**片上数据流动** 和**计算与通信重叠** 的特性，被视为突破内存瓶颈的关键路径。然而，**如何高效、自动化地将深度学习模型映射到数据流硬件**，并解决内核间关联、外部内存访问、流式内核融合与 FIFO 大小确定等一系列复杂问题，一直是学术界与工业界面临的挑战。
:::

![文章插图](/articles/2026-01-26-micro-25-streamtensor-llm-pytorch-e16790bb/images/01.webp)

- StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs
- https://arxiv.org/abs/2509.13694
- 1 万字，阅读 30 分钟，**播客 19 分钟**

相关推荐

- [MLIR-AIR：AMD 基于空间分区与显式同步原语的开源编译器栈，融合通信-计算重叠，实现矩阵乘法 78.7%计算效率](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893191&idx=1&sn=fba9af0b8819b0766c12b58bb2adfef9&scene=21#wechat_redirect)
- [开源端到端编译器工具链：从 PyTorch 到 FPGA 硬件描述，性能达Vitis HLS 的 2.21 倍！媲美闭源工业级！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894338&idx=1&sn=fe91ec511b9061c65c4545e06e9f16c7&scene=21#wechat_redirect)
- [GEMM 95% FPU 利用率！基于 MLIR 面向 RISC-V 指令集扩展的加速Micro-kernels的多层编译器后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889407&idx=1&sn=f5640780406954ae498d2c98d162d454&scene=21#wechat_redirect)

> 来自伊利诺伊大学厄巴纳-香槟分校与 Inspirit IoT 公司的研究团队提出了 **StreamTensor** ——一个**从 PyTorch 模型自动生成优化数据流加速器的完整编译器框架**。

该工作已被 MICRO 2025 接收，并在 FPGA 平台上验证了其在多项 LLM 推理任务中超越现有 FPGA 加速器与 GPU 的性能表现。

![文章插图](/articles/2026-01-26-micro-25-streamtensor-llm-pytorch-e16790bb/images/02.webp)

::: image-caption
图 3 | 本文提出的数据流加速器设计范式。该范式核心是引入 StreamTensor 编译器，实现从算子到数据流架构的自动化转换。编译器可自动完成核融合、FIFO 尺寸优化等关键步骤，无需人工干预数据流调度与组件配置，大幅提升了数据流加速器的设计效率与可扩展性。
:::
