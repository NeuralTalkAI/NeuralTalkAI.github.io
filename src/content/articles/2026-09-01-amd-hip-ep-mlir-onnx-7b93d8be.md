关键词：ONNX Runtime、**MLIR**、ROCm、**Execution Provider**、GPU 推理

> 在深度学习推理系统里，Execution Provider 往往被理解成“把算子派发到某个硬件后端”的适配层。hip-ep 展示了另一种更激进的形态：它把 EP 做成了一条真正的编译执行链。

![文章插图](/articles/2026-09-01-amd-hip-ep-mlir-onnx-7b93d8be/images/01.webp)

- :mdnice-bold-italic[ROCm hip-ep: ROCm Execution Provider for ONNX Runtime on AMD GPUs. Compiles ONNX models through MLIR to LLVM bitcode, JIT-loads them at session creation, and dispatches to hipDNN, MIOpen, hipBLASLt, and custom HIP kernels]
- https://github.com/ROCm/hip-ep

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 把 CDNA5 的"底牌"摊开了：832 页 ISA 手册全公开，MI455X 的每个细节都写在里头](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904248&idx=1&sn=b2b81e0229f104f7dd3d3a9cb3d31dcf&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**

:::

> :mdnice-bold-italic[hip‑ep 是面向 AMD GPU、基于 MLIR 实现的 ONNX Runtime 执行 Execution Provider，打破传统 EP 仅作为算子派发适配层的定位，将 EP 升级为完整模型编译执行链，打通 ONNX Runtime 与 ROCm 推理生态。] 模型经由 MorphiZen 图改写框架接入，通过 MLIR 编译管线逐级将 ONNX 算子方言 lowering 至 HIP 方言、memref 内存表示，最终输出可移植 LLVM 位码，在推理会话创建阶段进程内 JIT 加载执行，也支持可选编译为.so/.dll 原生库文件，底层调用 hipDNN、hipBLASLt 以及自定义 HIP 内核完成计算。

项目完整覆盖从源码构建、算子转换、内存管理到运行时调度全链路工程问题。

编译端依靠 CompilerDriver 统筹整套 Pass 流水线，流水线执行顺序直接决定系统正确性；算子转换采用 Rewrite Pattern 模式，原生支持大模型 GQA、KV‑Cache、量化算子，同时实现常量外置，解决大模型权重体积与加载瓶颈。

内存层面设计按需扩容 GPU 内存池减少频繁显存分配释放，输出张量通过 EP 回调机制对接 ONNX Runtime 张量体系。运行时 RuntimeState 统一管理会话生命周期内全部 GPU 资源，提供 LLVM bitcode JIT 与原生库两种交付模式。同时配置静态插件插槽、IR 调试环境变量，搭配分层测试体系，支持无 GPU 模拟环境开展验证测试。

::: block-1
hip‑ep 填补了 ONNX 模型到 AMD GPU 可执行程序之间的关键缺口，:mdnice-highlight[证明面向动态 shape 大模型推理，Execution Provider 应当具备编译器能力]，为异构硬件推理框架设计提供重要参考。
:::

## 本文目录

- 一、快速上手：先跑起来，再理解它
- 二、项目全貌：它其实是一个嵌入式 AI 编译器
- 三、编译主线：从 ONNX 方言到 LLVM 位码
- 四、ONNX 到 HIP：算子转换不是逐个翻译这么简单
- 五、内存：动态 shape 下的 GPU 池化与输出分配
- 六、运行时：会话状态是模型执行的载体
- 七、插件机制：给编译管线预留可控扩展点
- 八、工具与测试：它如何证明自己可运行
- 总结：把 EP 从适配层推进到编译层

![文章插图](/articles/2026-09-01-amd-hip-ep-mlir-onnx-7b93d8be/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 把 CDNA5 的"底牌"摊开了：832 页 ISA 手册全公开，MI455X 的每个细节都写在里头](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904248&idx=1&sn=b2b81e0229f104f7dd3d3a9cb3d31dcf&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**

:::
