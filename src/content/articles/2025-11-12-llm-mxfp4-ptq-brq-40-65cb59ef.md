关键词：大语言模型、**MXFP4格式**、:mdnice-bold-italic[训练后量化]、**块旋转量化**、低精度量化、模型部署

![文章插图](/articles/2025-11-12-llm-mxfp4-ptq-brq-40-65cb59ef/images/01.webp)

- Block Rotation is All You Need for MXFP4 Quantization
- https://arxiv.org/pdf/2511.04214
- 3000 字，阅读 10 分钟，**播客 19 分钟**

相关推荐

- [端侧 NPU 的 LLM 测试时计算扩展：硬件感知块量化与 LUT 优化实现 19.0×GEMM与 2.2×Softmax 加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892974&idx=1&sn=ef25e0315fc32464d4a9739fce435928&scene=21#wechat_redirect)
- [为 1B/3B LLM 实现近全精度与 3 倍推理加速方案 Tequila：突破死区陷阱的无陷阱三元量化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893058&idx=1&sn=2375641177b6de213718724f79a20fc5&scene=21#wechat_redirect)
- [W4A8 GEMM实现LLM 4.9倍加速！LiquidGEMM让反量化、权重加载和矩阵乘流水重叠，较TRT-LLM提升1.6倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891799&idx=1&sn=bb787e1c28c27320740a22759cad7812&scene=21#wechat_redirect)

> 大语言模型（LLMs）的规模化带来高昂的存储与计算成本，训练后量化（PTQ）是高效部署的关键方案，但 W4A4（4 位权重+4 位激活）量化的准确性仍存挑战。

新兴的 MXFP4 格式具备广泛软硬件支持，适配长尾分布且兼容低精度训练与推理，却**缺乏针对性 PTQ 方法指导**。

::: block-1
本文构建了 MXFP4 格式的 W4A4 量化基准，系统评估三类 PTQ 方法后发现，GPTQ 表现稳定，而主流旋转基方法因与 MXFP4 的 PoT（2 的幂）块缩放机制存在本质冲突，导致性能大幅下降。**冲突根源在于全局旋转会放大常规块的小值，而 MXFP4 对大值的重构能力有限，进而累积量化误差。**
:::

为此，本文提出块旋转量化（BRQ）策略，在 MXFP4 块内独立执行旋转，**既保留离群值抑制效果，又避免跨块误差传播。**

![文章插图](/articles/2025-11-12-llm-mxfp4-ptq-brq-40-65cb59ef/images/02.webp)

::: image-caption
图7：块旋转的原理：全局旋转会将离群值分散到所有通道中，导致常规块的尺度增大，并加剧量化误差。块级旋转会在局部重新分配离群值，在减轻离群值影响的同时保持常规块的尺度不变，从而将量化误差降至最低
:::

实验表明，BRQ 在 LLaMA、Mistral、Qwen 等多尺度模型上显著提升量化精度，**且推理延迟较全局旋转方法降低 40%，** 为低精度浮点硬件上的 LLM 部署提供了实用方案与理论支撑。

![文章插图](/articles/2025-11-12-llm-mxfp4-ptq-brq-40-65cb59ef/images/03.webp)
