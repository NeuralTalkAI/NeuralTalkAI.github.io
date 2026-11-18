> 我们**所研究的优化方法累计实现了高达 589%的性能提升**，并已将相关内核与框架作为开源项目贡献（ibm.biz/vllm-ibm-triton-lib）。最终，我们开发的高度优化内核已成为 vLLM 中 AMD 部署的默认注意力后端。【第八章 见解与未来工作】这一节内容也很有点意思 XD

关键词：Triton、**Attention Kernel**、**Portability**、Large Language Models、**Autotuning**

![文章插图](/articles/2025-12-12-triton-gpu-attention-5-9-56effc1e/images/01.webp)

- **The Anatomy of a Triton Attention Kernel**
- :mdnice-bold-italic[副标题：How to achieve cross-platform state-of-the-art LLM attention using only Triton]
- https://arxiv.org/pdf/2511.11581
- https://github.com/foundation-model-stack/vllm-triton-backend/tree/main/ibm-triton-lib
- 1.5 万字，阅读 60 分钟，**播客 20 分钟**

相关推荐

- **[vLLM TPU：一款支持在 TPU 上运行 PyTorch 与 JAX 的全新统一后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893481&idx=1&sn=58eca26a08502cb6820bdb3c111092bc&scene=21#wechat_redirect)**
- **[端到端加速4.5倍！端侧SoC动态稀疏注意力 shadowAttn：关键token稀疏计算架构且兼容主流框架释放NPU性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891725&idx=1&sn=1c315204fb9f1edc0b26e679d8561983&scene=21#wechat_redirect)**
- **[注意力块 4 倍加速！Zen-Attention：面向AMD NPU基于硬件感知的动态注意力层折叠编译框架，端到端32%提速!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891471&idx=1&sn=7b3c57d36ba86b27a277a997b771478f&scene=21#wechat_redirect)**

> 本文聚焦于解决大语言模型（LLM）推理中硬件依赖与性能优化的核心矛盾，提出一种基于 Triton 领域特定语言的跨平台分页注意力内核方案。该方案旨在突破“硬件 lottery”限制，**实现无需底层手动调优、却能在 NVIDIA 和 AMD GPU 上达到顶尖性能的 LLM 推理部署。**

![文章插图](/articles/2025-12-12-triton-gpu-attention-5-9-56effc1e/images/02.webp)

::: image-caption
图5. vllm-triton-backend中的调优工作流程
:::

现有 LLM 推理框架如 vLLM，依赖大量平台专属闭源库，导致部署复杂且硬件适配成本高。**本文通过 Triton 的即时编译特性，结合算法优化、参数自动调优与系统级整合，将通用 Triton 注意力内核的性能从顶尖水平的 19.7%提升至 105.9%。** *

::: block-1
核心贡献包括：

- **设计功能完备** 且**性能顶尖** 的:mdnice-bold-italic[【跨平台】分页注意力内核；]
- **总结算法优化、编程模型选择** 及**系统级权衡** 等:mdnice-bold-italic[【可复用经验】；]
- **开源内核与微基准测试套件**，并将其集成到 vLLM 框架，成为 :mdnice-bold-italic[AMD GPU 的默认注意力内核。]
:::

![文章插图](/articles/2025-12-12-triton-gpu-attention-5-9-56effc1e/images/03.webp)

::: image-caption
图2. vLLM中triton_attn后端的概述
:::

关键技术优化涵盖 Q 块结构设计、并行分块 Softmax、可调节块大小及静态启动网格等，同时通过:mdnice-bold-italic[微基准测试驱动的启发式自动调优]，**平衡了调优开销与跨平台适配性。**

实验表明，该内核在 **NVIDIA H100 和 AMD MI300 GPU** 上，对 Llama-3.1-8B 模型的推理性能:mdnice-bold-italic[媲美甚至超越 FlashAttention 等专用库，最长序列场景下提速达 5.9 倍]，为 LLM 推理的:mdnice-bold-italic[硬件无关化部署] 提供了高效解决方案。

![文章插图](/articles/2025-12-12-triton-gpu-attention-5-9-56effc1e/images/04.webp)
