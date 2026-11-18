关键词：Triton、MTIA-2i、**编译器后端**、:mdnice-bold-italic[TorchInductor]、**kernel 开发效率**

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/01.webp)

- 论文标题：**Triton for MTIA: Bridging the Programming Model Gaps for Custom AI Accelerators**
- 组织机构：Meta Platforms
- 论文链接：https://arxiv.org/pdf/2608.00325
- 开源代码地址：未公开（Meta内部生产项目）

::: related-title
**相关推荐**
:::

- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**
- **[vLLM Triton 注意力后端深度解析：跨 NVIDIA、AMD 的 GPU 架构创新优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898509&idx=1&sn=341fceedfbc28e118212eb85ae5b1807&scene=21#wechat_redirect)**
- **[从算子库到编译生成：Hexagon-MLIR 通过 MLIR 实现 PyTorch 与 Triton 在 Hexagon NPU 上的统一优化与 3.95 倍多线程加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898220&idx=1&sn=8da7c546a9ac4f04b378e2a316c35de7&scene=21#wechat_redirect)**

> 我们能不能在不牺牲开发者效率的前提下，支持当今数据中心里部署的所有 ML 加速器？:mdnice-bold-italic[Meta 用自家第二代推理芯片 MTIA-2i 给出了一个生产级答案——把 GPU 生态里最流行的 kernel 编程语言 Triton，完整地搬到一块编程模型与 GPU 截然不同的自研 ASIC 上]。

Triton 写出的 GEMM kernel 稳定达到架构 roofline 的 80% 以上，与专家手写 C++ 互有胜负；一个季度之内，生产环境中使用 Triton 的模型类型翻了三倍，覆盖约 60 种模型 50% 的层和 47% 的非 GEMM 执行时间。

::: block-1
这是 Triton 第一次在非 GPU 架构上完成生产规模验证，:mdnice-highlight[高级 DSL 加编译器，完全可以成为自研芯片跨越软件生态鸿沟的那座桥。]
:::

## 本文目录

- 一、GPU 的 kernel 语言，为什么要搬到自研芯片上
- 二、MTIA-2i 是一块什么样的芯片
- 三、从零造一个 Triton 编译器
- 四、性能从哪里来：编译器优化的三板斧
- 五、Inductor 定制与语言扩展：自动化与掌控力两头都要
- 六、实测数据：性能、覆盖率与生产渗透
- 七、移植经验与 DSL 战略：被放弃的 KNYFE 与留下的启示

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[完全由 Kimi K3 写出的 MiniTriton，碾压 Torch 与原生 Triton？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903963&idx=1&sn=63acd066cbada9c93dd89cb384e82169&scene=21#wechat_redirect)**
- **[零侵入替换训练热路径！Liger-Kernel：面向 LLM 训练的高性能 Triton Kernel 与 API 全解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903793&idx=1&sn=15c6caf35822547f67412d670bb7b60e&scene=21#wechat_redirect)**
- **[Triton 算子如何越过 PTX 直抵 Blackwell：CUDA TileIR 后端的诞生，Triton-to-tile-IR](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903681&idx=1&sn=a30c96b2bfbc7a32d9745ca6d303ddfd&scene=21#wechat_redirect)**
- **[从 196 到 547 TFLOPS！Triton-TLE：告别手动 barrier，用生产消费模型实现 FlashMLA 90% 的性能！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901317&idx=1&sn=0cdf1ca561fe14ccb56ef909b34bf661&scene=21#wechat_redirect)**

:::
