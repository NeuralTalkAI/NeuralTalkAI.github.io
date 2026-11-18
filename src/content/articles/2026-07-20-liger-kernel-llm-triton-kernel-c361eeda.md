关键词：**Triton 内核**、:mdnice-bold-italic[大模型训练]、算子融合、**显存优化**、:mdnice-bold-italic[后训练损失]

> 本文对应 ICML 2025 上发表《Liger Kernel: Efficient Triton Kernels for LLM Training》一文的代码讲解，该工作是一套从代码底层出发、专为大语言模型训练打造的开源高性能 Triton 内核库。

不同于上层训练框架的间接优化，**Liger-Kernel 直接从算子代码实现层面切入核心痛点：基于 Triton 高级 GPU 编程语言重构 LLM 训练全链路的高频算子**，既规避了原生 CUDA 内核极高的开发与维护门槛，又:mdnice-bold-italic[通过代码层面的多算子融合、分块流式计算、显存访问路径优化，从根源上消解大模型训练中频繁跨层显存读写带来的带宽瓶颈与算力空耗。]

![文章插图](/articles/2026-07-20-liger-kernel-llm-triton-kernel-c361eeda/images/01.webp)

- **Liger Kernel: Efficient Triton Kernels for LLM Training**
- 代码：https://github.com/linkedin/Liger-Kernel
- 论文：Liger-Kernel: Efficient Triton Kernels for LLM Training
- 链接：https://openreview.net/pdf?id=36SjAIT42G

::: related-title
**相关推荐**
:::

- **[高性能算子的“包管理时代”：HF kernels-community 打通模型、硬件与 Hub，重塑底层加速分发链路！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901232&idx=1&sn=a493721d68f2578e2585b11effaaa871&scene=21#wechat_redirect)**
- **[NVIDIA 开源 GPU 编程教学仓库 accelerated-computing-hub，从单线程 kernel 到协作归约](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900741&idx=1&sn=8a88431b7a8d1d46aa3b03f9889bc151&scene=21#wechat_redirect)**
- **[NVIDIA 开源 cuDNN Frontend：用图 API 把底层内核能力封装成可复用、可调优的 AI 算子构建系统](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901178&idx=1&sn=6b7dd90b31e401357660a4a02cdd1262&scene=21#wechat_redirect)**

> **在代码架构设计上，项目遵循模块化、低侵入的工程原则**，所有优化内核均以独立组件形式封装，无需改动模型核心逻辑，仅需数行代码即可完成接入，:mdnice-bold-italic[原生兼容 HuggingFace Transformers、Flash Attention、FSDP、DeepSpeed 等主流训练生态]，完整覆盖预训练、SFT 微调、DPO/ORPO 对齐等全场景的 RMSNorm、RoPE、SwiGLU、交叉熵损失等核心算子。

在原始最初的论文基准测试显示，其在 LLaMA 3、Qwen2等主流大模型上可实现最高 42.8% 的训练吞吐量提升与 56.8% 的显存占用缩减（*按最新代码，数据可能会有变化，以下文中图片和数据为准*），且完全保障训练收敛性与数值正确性。

![文章插图](/articles/2026-07-20-liger-kernel-llm-triton-kernel-c361eeda/images/02.webp)

::: image-caption
不同序列长度下，Hugging Face 原生实现与 Liger Kernel 的 token 吞吐性能。在 1024、2048 序列长度时，Liger Kernel 吞吐量显著高于 Hugging Face；当序列长度达到 4096 及以上时，原版 Hugging Face 出现 OOM 显存溢出无法运行，而 Liger Kernel 仍可正常推理并保持可观吞吐，兼顾更高推理速度与显存效率，有效支持超长上下文场景。
:::

![文章插图](/articles/2026-07-20-liger-kernel-llm-triton-kernel-c361eeda/images/03.webp)

::: image-caption
不同序列长度下 Hugging Face 原生实现与 Liger Kernel 的峰值显存占用。在 1024、2048 序列长度时，Liger Kernel 峰值显存占用远低于原版 Hugging Face，显存优化效果显著。当序列长度达到 4096 及以上时，原版 Hugging Face 发生显存溢出（OOM）无法运行，Liger Kernel 则可持续运行，虽显存占用随上下文变长逐步增加，但避免了溢出问题，大幅提升超长上下文任务的显存可用性。
:::

![文章插图](/articles/2026-07-20-liger-kernel-llm-triton-kernel-c361eeda/images/04.webp)

::: image-caption
柱状图对比六种后训练损失函数下，Hugging Face 原生实现与 Liger 的峰值显存占用。可以看到，在 ORPO、DPO、CPO、SimPO、KTO、JSD 全部方法中，Liger 的峰值显存占用均远低于原版 Hugging Face，显存优化效果十分突出。其中 KTO 场景差距尤为明显，整体大幅降低微调训练显存开销，减轻硬件压力，便于大模型后训练任务部署运行。
:::

::: block-1
作为兼具学术严谨性与工程落地性的开源项目，Liger-Kernel 既为 LLM 训练提供了开箱即用的性能优化方案，也为开发者自定义内核开发提供了可复用的代码范式。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 开源 cuDNN Frontend：用图 API 把底层内核能力封装成可复用、可调优的 AI 算子构建系统](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901178&idx=1&sn=6b7dd90b31e401357660a4a02cdd1262&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**
- **[面向边缘 NPU Kernel 的 MLIR 编译优化：向量化、多线程和双缓冲的量化评估与性能挖掘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898779&idx=1&sn=84a57101de07ace63adb93d03e7db1b7&scene=21#wechat_redirect)**

:::
