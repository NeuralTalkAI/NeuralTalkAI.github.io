关键词：**张量编译器**、Rust、异构计算、:mdnice-bold-italic[中间表示]、**分布式运行时**

> 生产级机器学习栈往往将图编译与内核执行拆分在**不同层级与不同语言中，使得后端行为、部署保证与性能降级路径难以进行端到端的推导与验证。**

![文章插图](/articles/2026-10-24-mit-rlx-rust-mcu-tpu-b9f940a6/images/01.webp)

过去十年，机器学习框架沿着前端易用性与后端性能的二元路径演进，Python 凭借灵活的动态特性成为主流宿主语言，而内核执行则下沉到 C/C++甚至硬件厂商专属的闭源库中。:mdnice-bold-italic[分层架构让 AI 应用开发门槛持续降低，却也在系统层面埋下了碎片化的隐患：跨语言调用带来固定开销，不同硬件后端需要独立适配，算子不支持时的静默降级让性能表现不可预测，端到端的正确性验证变得异常困难。]

![文章插图](/articles/2026-10-24-mit-rlx-rust-mcu-tpu-b9f940a6/images/02.webp)

- **RLX: A Unified Multi-Backend Tensor Compiler and Distributed Runtime in Rust**
- 组织机构：麻省理工学院（Massachusetts Institute of Technology）
- https://arxiv.org/pdf/2609.37916
- 开源代码：https://github.com/MIT-RLX/rlx
- 文档：https://docs.rs/rlx
- 配套实验代码仓库：https://github.com/MIT‑RLX/rlx‑paper，但我看这个仓库还没开源

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 vLLM，SGLang 吞吐！cuTile Rust 实现近零开销安全内核，GEMM 达 96% cuBLAS 性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902639&idx=1&sn=1b7eb41a0df5119f59af2413a7e8bc71&scene=21#wechat_redirect)**
- **[NVlabs/cuda-oxide：把 Rust 的类型系统接入 CUDA SIMT 世界的实验性编译器](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900944&idx=1&sn=471671b6081812dd4bae00a8dea74c32&scene=21#wechat_redirect)**
- **[NVlabs/cutile-rs：用 Rust 安全高效地编写 GPU 内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898633&idx=1&sn=44b1711f524e9ad14285527c9add5f18&scene=21#wechat_redirect)**

:::

> RLX 的核心目标就是打破这种分层割裂的状态。**它以单一 Rust 代码库同时承载图编译器与内核运行时的双重角色，围绕一套原语级的三层中间表示构建全编译流水线**，通过透明的调度契约将每个算子明确映射到原生执行、通用 IR 降级或重写路径，在无法完成合法化时直接终止编译并给出诊断信息，从根本上杜绝静默降级。

![文章插图](/articles/2026-10-24-mit-rlx-rust-mcu-tpu-b9f940a6/images/03.webp)

::: image-caption
图 1 RLX 架构图。前端与权重格式加载器追踪至原语级 IR，IR 分 HIR/MIR/LIR 三层逐级降级，同一套 IR 覆盖 14 类运行时目标与 2 类专用代码生成路径，适配全主流平台。 该图直观呈现了 RLX “单 IR 贯穿全链路” 的核心设计思想。三层 IR 共享同一算子集合，层间无语义鸿沟，所有优化能力可被全后端复用；透明调度机制独立于硬件层，从架构上杜绝静默降级；分布式能力以运行时扩展实现，不侵入核心 IR 设计，整体架构简洁且扩展性强。
:::

这套统一的 IR 可**覆盖 14 类运行时设备与 2 类专用代码生成路径，支撑从 Cortex-M 微控制器到云端 TPU 的全场景硬件，同时支持多精度量化与分布式并行扩展**，甚至将能力边界延伸到科学计算与 3D 高斯泼溅等非神经网络领域。

从实测结果来看，RLX 的性能表现已经超越多数主流框架。

- 在 all-MiniLM-L6-v2 模型的推理测试中，**RLX-Metal 后端在所有批量大小下均取得最优延迟，批量 32 时仅需 16.6 毫秒，较 PyTorch-MPS 快出 37.8%；**
- 在 MNIST 训练任务中，图融合后的 MLP 模型吞吐量达到 946487 张/秒，超过 NumPy+BLAS 的 787349 张/秒，同时保持与参考实现 100%的精度对齐。

![文章插图](/articles/2026-10-24-mit-rlx-rust-mcu-tpu-b9f940a6/images/04.webp)

::: image-caption
图 2：all-MiniLM-L6-v2 多引擎推理延迟对比图，该实验以主流的 Transformer Embedding 模型为测试对象，在同一 Apple Silicon 平台上横向对比了 5 种 RLX 后端与 PyTorch、ONNX Runtime 等业界主流方案的推理延迟。其核心价值体现在两点：一是验证跨后端可移植性，同一份编译完成的计算图无需修改，即可在 CPU、Metal、MLX、Vulkan 等不同硬件后端运行，完美体现了 “一次建模、全硬件部署” 的设计目标；二是证明性能领先性，RLX-Metal 在所有批量大小下均取得最低延迟，批量 32 时相比 PyTorch-MPS 性能提升约 38%，CPU 路径也显著优于 PyTorch-CPU 与 ONNX Runtime，直观反映了整图算子融合、Rust 原生低开销调度带来的性能增益。同时 ONNX-CoreML 的性能短板也反向印证了碎片化图拆分的性能损耗，凸显了 RLX 整图编译架构的技术优势。
:::

![文章插图](/articles/2026-10-24-mit-rlx-rust-mcu-tpu-b9f940a6/images/05.webp)

::: image-caption
图 3 展示了在 Apple M4 Pro 平台上，基于 LeNet CNN 模型、SGD 优化器、批量大小 128、训练 2 轮的 MNIST 训练吞吐量对比结果，涵盖 RLX 的多种后端实现以及 PyTorch、TensorFlow、JAX、MLX、tinygrad 等十余种主流框架与引擎。结果显示 RLX-Metal 以 1.8 秒的总训练耗时、67476 张 / 秒的吞吐量位列第一，其图融合版本的 CPU 路径性能达到标量实现的 13.5 倍，跻身 XLA/MLX 所在的第一梯队，完整测试集中 RLX 图融合 MLP 更是实现了 946487 张 / 秒的全局最高吞吐量，超越了经过长期优化的 NumPy+BLAS 方案。所有参与对比的方案最终准确率均落在 95%~98% 区间，且训练完成的模型可在全部 RLX 后端上无修改部署并保持一致准确率，直观印证了 RLX 统一三层 IR 架构下训练与推理的一体化设计 —— 反向传播完全通过 IR 变换生成，无需独立训练引擎，算子融合可同时作用于前向与反向计算流程，既实现了训练性能的显著提升，也保障了跨硬件的可移植性与数值一致性，证明 RLX 并非单纯的推理引擎，而是具备完整训练能力的全栈张量编译与运行时系统。
:::

## 本文目录

- 一、异构 AI 栈的碎片化困局：统一编译运行时的必要性
- 二、RLX 架构全景：三层 IR 构建的全栈统一编译体系
- 三、透明内核调度：终结静默降级的硬核契约
- 四、全场景硬件覆盖：从 Cortex-M 到 TPU 的 16 类后端实现
- 五、从推理到训练：统一 IR 打通全链路能力
- 六、性能实测：RLX 与主流框架的横向对比
- 七、总结与展望：Rust 原生 AI 基础设施的未来

![文章插图](/articles/2026-10-24-mit-rlx-rust-mcu-tpu-b9f940a6/images/06.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[当 Docker 不再安全：腾讯云 CubeSandbox 如何用 Rust+KVM 重塑 AI Agent 沙箱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900018&idx=1&sn=e1c1aa844446bb6c16ff56bd61aff9ef&scene=21#wechat_redirect)**
- **[刚刚，SpaceXAI 开源 Grok Build：一个 Rust AI Agent Harness 的系统级拆解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903721&idx=1&sn=20b3a6235e2c8a3bae1df8e27de39301&scene=21#wechat_redirect)**
- **[GPU故障不用重启，从 47 秒宕机到 1.5 秒自愈！解决LLM Agent长会话丢失的容错运行时方案 Concordia：PTX/SASS 底层插桩+JIT 专用检查点](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903054&idx=1&sn=4e9a63f3716b34a5e54b21d8141c8c8d&scene=21#wechat_redirect)**

:::
