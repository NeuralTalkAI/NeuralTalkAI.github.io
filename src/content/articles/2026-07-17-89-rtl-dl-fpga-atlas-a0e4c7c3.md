关键词：**FPGA 深度学习加速**、:mdnice-bold-italic[高层次综合]、GEMM 抽象、专用计算硬核、**端到端编译流**

> *论文中有一句直击行业痛点的判断：**现有 FPGA 深度学习编译框架，始终无法填补高层模型描述与定制片内硬核之间的可编程性鸿沟**。过去几年，FPGA 厂商与架构研究者纷纷在传统 FPGA 布线中加入深度学习专用硬核，:mdnice-bold-italic[从 Intel 的 AI Tensor Block 到学术圈的 Tensor Slices、模拟存内计算引擎，这些专用单元的算力密度、能效比都远超传统软逻辑 DSP，是 FPGA 应对大模型推理算力需求的核心抓手。]*

但尴尬的现实是，**这些硬核的算力绝大多数开发者根本用不起来**。要调用这些硬核，:mdnice-bold-italic[开发者必须精通底层微架构，手动完成计算分块、RTL 实例化、控制状态机编写、数据通路调度]，每一层网络都要重复一遍工作，一个完整模型的开发动辄需要数天甚至数周，门槛极高。

![文章插图](/articles/2026-07-17-89-rtl-dl-fpga-atlas-a0e4c7c3/images/01.webp)

::: image-caption
图 1：在深度学习优化的 FPGA 上，基线 hls4ml 流程（左）仅将深度学习工作负载映射到软逻辑资源上，导致专用硬模块闲置。本文提出的 ATLAS 流程（右）可实现工作负载到深度学习硬模块的自动映射。该图清晰点明了研究的核心问题与解决方案：现有 hls4ml 等高层综合框架仅能映射到 CLB、DSP、BRAM 等通用软逻辑资源，无法利用 FPGA 内集成的 DL 专用硬模块，造成硬件资源浪费与效率瓶颈。ATLAS 通过新增 GEMM IP 生成环节，在传统 hls4ml 流程与 FPGA 硬件之间架起桥梁，自动将矩阵乘法类计算分流到 DL 硬模块执行。这种设计既保留了高层综合的高生产力优势，又能充分发挥专用硬模块的性能与能效优势，解决了 DL 优化 FPGA 的可编程性痛点，为新型 FPGA 架构的软件栈适配提供了通用思路。
:::

![文章插图](/articles/2026-07-17-89-rtl-dl-fpga-atlas-a0e4c7c3/images/02.webp)

- **ATLAS: Automated HLS for DL-Optimized FPGAs**
- https://arxiv.org/pdf/2607.07643

::: related-title
**相关推荐**
:::

::: related-links
- **[RISCBench：面向 FPGA 及类 FPGA 计算引擎的 RISC-V 编排效率基准测试](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898844&idx=1&sn=a9d00d2702d21c4120d7ba18050dc665&scene=21#wechat_redirect)**
- **[hls4ml：面向 FPGA AI 部署的开源模块化编译器，实现微秒级延迟、极致资源效率，支持 PyTorch/Keras/ONNX，适配 AMD/Intel/ASIC](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897821&idx=1&sn=6f4033c3aa72e8c73d802f921164d134&scene=21#wechat_redirect)**
- **[开源端到端编译器工具链：从 PyTorch 到 FPGA 硬件描述，性能达Vitis HLS 的 2.21 倍！媲美闭源工业级！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894338&idx=1&sn=fe91ec511b9061c65c4545e06e9f16c7&scene=21#wechat_redirect)**

:::

> 来自亚利桑那州立大学的团队提出的 ATLAS，正是为了解决这个核心矛盾。这项工作的关键洞察在于，:mdnice-bold-italic[深度学习的核心计算层都可以统一降维为通用矩阵乘法（GEMM），而所有 DL 专用硬核本质上都是固定尺寸的 GEMM 加速器。只要以 GEMM 作为统一抽象层，将编译流拆分为架构无关的前端与架构感知的后端，就能实现从高层 Keras 模型到底层硬核的全自动化映射]，完全不需要手动编写 RTL。

![文章插图](/articles/2026-07-17-89-rtl-dl-fpga-atlas-a0e4c7c3/images/03.webp)

::: image-caption
图 2：ATLAS 端到端编译流程。架构无关的 hls4ml-GEMM 前端，结合架构感知的 IP 生成器，以 GEMM 为抽象层，实现了高层 Keras 模型到深度学习优化 FPGA 的映射。该两段式解耦流程是 ATLAS 通用性的核心支撑。前端 hls4ml-GEMM 仅关注深度学习模型的计算特性，将全连接、卷积、注意力层统一转换为架构无关的 GEMM 函数调用，不涉及任何硬件细节；后端 GEMM IP 生成器则针对目标 FPGA 的硬模块架构，自动生成 RTL 封装、控制逻辑与调度元数据。两者通过标准化的 GEMM 规范对接，既降低了编译器的开发复杂度，也为后续扩展新的层类型、新的硬件架构提供了极高灵活性，前端无需修改即可适配新硬模块，后端无需改动即可支持新 DL 模型。
:::

实验结果显示：

- 在单层网络测试中，**ATLAS 的计算面积效率达到手写 RTL 基线的 89%，** 比纯软逻辑的 hls4ml 基线高出 24%；
- 在完整的 MLP、CNN、Transformer 模型测试中，**几何平均效率达到手写 RTL 的 63%，比软逻辑基线高出 42%。**

![文章插图](/articles/2026-07-17-89-rtl-dl-fpga-atlas-a0e4c7c3/images/04.webp)

::: image-caption
图 5：各完整设计的计算面积效率（每 1000 个 CLB 等效资源的 GMAC/s），以 hls4ml 基线为基准做归一化。本文提出的流程在三类模型上均优于软逻辑基线，在以全连接层为主的 MLP 与 CNN 模型上接近 RTL 基线水平。该图展示了端到端完整模型的效率表现，验证了 ATLAS 流程在复杂网络下的可用性。MLP 与 CNN 模型以权重固定模式的 GEMM 为主，自动生成的接口开销占比低，效率分别达到 RTL 基线的 89% 与 86%，与逐层测试结果一致；Transformer 模型因包含双动态模式的注意力计算，需要额外的流控与双输入缓冲逻辑，软逻辑开销显著增加，效率为 RTL 的 32%，但仍达到软逻辑 hls4ml 的 2 倍。整体几何平均效率比 hls4ml 高 42%，达到 RTL 的 63%，证明该流程能够有效支撑完整 DL 模型的自动化硬件实现，兼顾生产力与硬件效率。
:::

::: block-1
而开发时间从原本的 7-8 天压缩到 2-3 小时，和纯软逻辑 HLS 开发的工作量完全一致。:mdnice-highlight[这项工作第一次真正打通了从高层深度学习框架到定制 FPGA 硬核的端到端自动化路径，无论是对商用 FPGA 工具链的演进，还是对学术圈的新型架构探索，都有极高的参考价值。]
:::

## 本文目录

- 一、FPGA 深度学习加速的编程困局：硬核的算力与易用性鸿沟
- 二、ATLAS 的核心设计思想：用 GEMM 作为软硬件的统一抽象层
- 三、前端 hls4ml-GEMM：把各类 DL 层统一降维成架构无关的 GEMM 调用
- 四、后端 GEMM IP 生成器：架构感知的自动化 RTL 封装与调度适配
- 五、实验验证：自动化流到底离手写 RTL 还有多大差距
- 总结与展望

![文章插图](/articles/2026-07-17-89-rtl-dl-fpga-atlas-a0e4c7c3/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[FPGA上的LLM推理加速1.23倍至3.83倍：SkipOPU的token级动态分配与跨层KV复用技术](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899343&idx=1&sn=b73b0f49c9a7314bc2efa3b2f0733d21&scene=21#wechat_redirect)**
- **[FPGA'26 首款超低成本嵌入式 FPGA 平台 Hummingbird+：让 30B MoE大模型在 150 美元硬件上流畅运行！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898613&idx=1&sn=75ccf31f231bcf171dfbd0b9c868d3e4&scene=21#wechat_redirect)**
- **[基于 FPGA 的稀疏注意力加速器 FAST-Prefill：相比 NVIDIA A5000实现 2.5 倍TTFT加速与 4.5 倍能效提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898021&idx=1&sn=e821816258c60284f6f128d1530ef252&scene=21#wechat_redirect)**

:::
