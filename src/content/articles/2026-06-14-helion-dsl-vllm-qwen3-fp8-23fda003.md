关键词：**Helion**、vLLM、**FP8量化**、推理内核、PyTorch

> 本文围绕**Helion 内核集成 vLLM** 开展技术研究与性能评测，主要验证该内核 DSL 在 Qwen3 模型 FP8 量化推理场景下的落地效果，测试硬件为 NVIDIA H100 与 B200 GPU。

Helion 是一款 PyTorch 原生、硬件无关的 GPU 内核领域专用语言，采用 tile 编程模型，既保留 PyTorch 简易的开发体验，又支持内存布局、tile 策略等底层细节调控，还搭载强大的提前自动调优能力，可自动匹配硬件与 workload 的最优内核配置。

::: block-1
**本次实验针对 vLLM 无张量并行推理场景，分别适配逐 Token、逐分组两种动态 FP8 激活量化模式**，替换推理链路中量化、归一化、矩阵乘、算子融合等绝大部分前向内核，:mdnice-bold-italic[并对接 DeepGEMM、CUTLASS 两类主流 GEMM 后端。]
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[vLLM 冷启动延迟的系统性拆解与深度优化指南](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902194&idx=1&sn=3d17f7b85eeef0daac4735440cdebb66&scene=21#wechat_redirect)**
- **[vLLM Triton 注意力后端深度解析：跨 NVIDIA、AMD 的 GPU 架构创新优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898509&idx=1&sn=341fceedfbc28e118212eb85ae5b1807&scene=21#wechat_redirect)**
- **[“历史性的项目，错过了遗憾终生”：清华特奖游凯超与 vLLM 的三次蜕变与技术远征](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896591&idx=1&sn=689f8a0351d8ea9ed0c2c0ebd2f89ca2&scene=21#wechat_redirect)**

:::
