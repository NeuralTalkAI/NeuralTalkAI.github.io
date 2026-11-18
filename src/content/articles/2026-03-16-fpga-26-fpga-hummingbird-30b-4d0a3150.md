关键词：**FPGA 加速器**、:mdnice-bold-italic[混合专家模型(MoE)、] 边缘部署、**低成本推理**、GEMV 优化

> 150 美元 BOM 成本、18 token/s 解码速度，FPGA 在 LLM 边缘部署领域终于翻身了

在深度学习硬件加速领域，FPGA 一直扮演着一个颇为尴尬的角色。它既不像 GPU 那样凭借极高的算力密度统治训练市场，也不像 ASIC 那样在特定场景下实现极致的能效比。**长久以来，FPGA 主要活跃在三个领域：架构原型验证、芯片验证平台，以及少数特定场景的最终部署平台。**

![文章插图](/articles/2026-03-16-fpga-26-fpga-hummingbird-30b-4d0a3150/images/01.webp)

::: image-caption
表 5：与 SOTA FPGA 的大语言模型加速器对比表。该表对比了本研究与 12 项现有研究的 FPGA 加速器，从平台、模型、压缩方式、性能等多维度量化分析，本研究在入门级 FPGA 上部署了最大的 30B MoE-LLM，且同时支持预填充与解码加速，解码性能居嵌入式 FPGA 加速器首位，凸显了技术突破。
:::

然而，随着大语言模型的兴起，这个局面正在发生微妙的变化。LLM 推理呈现出明显的**内存受限特性** ——端到端的性能瓶颈更多来自于内存带宽而非计算吞吐量。这使得原本在算力上处于劣势的 FPGA，突然与 GPU、NPU 站在了相对平等的起跑线上。

::: block-1
但问题依然存在：现有的 FPGA-based LLM 加速器研究大多使用昂贵的高端 FPGA 器件（如 Alveo U280、VCU128 等），这些动辄数千美元的平台对于边缘产品来说成本过高。**真正的边缘产品对成本极度敏感，如何在低成本的嵌入式 FPGA 上实现可用的 LLM 推理性能，成为一个亟待解决的难题。**
:::

![文章插图](/articles/2026-03-16-fpga-26-fpga-hummingbird-30b-4d0a3150/images/02.webp)

- **Hummingbird+: Advancing FPGA-based LLM Deployment from Research Prototype to Edge Product**
- https://dl.acm.org/doi/epdf/10.1145/3748173.3779189
- 9000 字，阅读 35 分钟，**播客 20 分钟**

相关推荐

- [树莓派跑 10B+参数 MoE！EdgeMoE：首款端侧稀疏 LLM 高性能引擎且代码开源！5.4×推理加速与3.2×内存节省！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889845&idx=1&sn=690674e0b64195ed9071603fd24c4ec5&scene=21#wechat_redirect)
- [EdgeLLM：数据并行、端到端编译器与混合精度的 CPU-FPGA 加速架构](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888046&idx=1&sn=59307cdc16ca341e201680d6a02f8db0&scene=21#wechat_redirect)
- [低功耗高性能！TeLLMe：首个 FPGA 三值 LLM 加速器！三值矩阵乘法查找表和预填充注意力协同优化计算、内存和调度！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888617&idx=1&sn=4ec623ed4d887d2cb7eebf5003057441&scene=21#wechat_redirect)

> 正是在这样的背景下，来自中国科学院自动化研究所的研究团队提出了**Hummingbird+，** 首次展示了使用低成本 FPGA 作为 LLM 最终部署介质的可行性。

他们的定制 PCB 平台基于 Zynq UltraScale XCZU2CG/3EG SoC，配备了 24GB 内存，大规模生产时的物料成本预计低于 150 美元。

本文提出的 Hummingbird+ 方案实测视频 Demo

在这个平台上，他们成功部署了 GPTQ 4-bit 量化后的 Qwen3-30B-A3B 混合专家模型，实现了超过 18 token/s 的解码速度和 50 token/s 的预填充速度。

![文章插图](/articles/2026-03-16-fpga-26-fpga-hummingbird-30b-4d0a3150/images/03.webp)

::: image-caption
图 1 | Hummingbird+针对混合专家（MoE）大语言模型部署打造的端到端解决方案架构图，直观呈现了该方案从硬件载体到计算架构的完整设计体系，是整个研究从FPGA原型验证走向边缘产品落地的核心视觉化体现。该架构以定制化PCB为硬件基础，搭载入门级Zynq UltraScale XCZU2CG/3EG SoC，集成24GB大容量内存并实现量产BOM成本低于150美元，突破了传统FPGA平台高成本、内存受限的问题。同时融合了经深度优化的加速器架构，将定制化AXI基础设施、多令牌处理器，以及轻量化的GEMV和标量引擎深度耦合，实现硬件资源与MoE模型推理需求的精准匹配，首次完成了FPGA平台上中大规模MoE-LLM的工程化部署设计，为边缘场景的LLM硬件加速提供了完整的解决方案参考。
:::
