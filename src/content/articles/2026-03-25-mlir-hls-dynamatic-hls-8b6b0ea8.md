关键词：**MLIR**、HLS、**高级综合**、:mdnice-bold-italic[Dynamatic]、**编译器基础设施**、数据流电路

> LLVM 是当前高级综合（HLS）工具的主流底层框架，但其中间表示（IR）难以定制化表达电路语义，而 MLIR 承诺通过自定义方言解决该问题。

![文章插图](/articles/2026-03-25-mlir-hls-dynamatic-hls-8b6b0ea8/images/01.webp)

- **Is It a Good Idea to Build an HLS Tool on Top of MLIR? Experience from Building the Dynamatic HLS Compiler**
- https://arxiv.org/pdf/2603.19856
- **代码：github.com/EPFL-LAP/dynamatic**
- 文档：https://epfl-lap.github.io/dynamatic/
- 4500 字，阅读 20 分钟，**播客 14 分钟**

相关推荐

- [hls4ml：面向 FPGA AI 部署的开源模块化编译器，实现微秒级延迟、极致资源效率，支持 PyTorch/Keras/ONNX，适配 AMD/Intel/ASIC](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897821&idx=1&sn=6f4033c3aa72e8c73d802f921164d134&scene=21#wechat_redirect)
- [Voyager：基于 HLS 的端到端 DNN 加速器设计框架，56%面积优化，61%延迟降低，99.8%利用率！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896823&idx=1&sn=87423b6a26f0643ecb1d370cc6fd2810&scene=21#wechat_redirect)
- [开源端到端编译器工具链：从 PyTorch 到 FPGA 硬件描述，性能达Vitis HLS 的 2.21 倍！媲美闭源工业级！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894338&idx=1&sn=fe91ec511b9061c65c4545e06e9f16c7&scene=21#wechat_redirect)

> 本文基于动态调度 HLS 编译器 Dynamatic 的长期开发实践，系统评估在 MLIR 之上构建 HLS 工具的可行性。

研究确认 MLIR 具备模块化架构、降低定制 IR 开发开销、适配学术团队开发等优势，可支撑数据流电路的表示与优化。**但团队在工程落地中发现 MLIR 存在四大核心局限**：

1. MLIR 数值无法为图边添加属性，难以建模内存依赖与控制流边信息；
2. 块参数替代 $\phi$ 节点的设计，增加软件转硬件时多路选择器的转换难度；
3. 第三方 MLIR-HLS 项目因 LLVM 版本不兼容，存在严重的集成碎片化问题；
4. 现有 C 语言前端优化能力不足，导致项目仍需依赖 LLVM 并采用复杂变通方案。

::: block-1
这些问题并非仅存在于 HLS 领域，对通用 MLIR 编译器开发同样具有参考价值。本文总结工程实践经验，为 MLIR 社区优化与未来 HLS 工具基础设施选型提供关键依据。
:::

![文章插图](/articles/2026-03-25-mlir-hls-dynamatic-hls-8b6b0ea8/images/02.webp)

::: image-caption
Processor Architecture Laboratory
:::
