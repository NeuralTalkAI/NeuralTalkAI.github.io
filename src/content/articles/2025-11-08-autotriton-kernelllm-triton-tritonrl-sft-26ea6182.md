关键词：**Triton Kernel Generation**、Hierarchical Reward Reinforcement Learning、:mdnice-bold-italic[Knowledge Distillation-based Supervised Fine-Tuning]、**KernelBench**

![文章插图](/articles/2025-11-08-autotriton-kernelllm-triton-tritonrl-sft-26ea6182/images/01.webp)

- TritonRL: Training LLMs to Think and Code Triton Without Cheating
- https://arxiv.org/pdf/2510.17891
- **代码与数据集即将发布**
- 11000 字，阅读 40 分钟，**播客 10 分钟**

相关推荐

- **[AutoTriton：通过强化学习实现自动Triton编程！8B参数媲美Claude-4 与 DeepSeek-R1！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890530&idx=1&sn=2f4b9743242cd200c0bae8e8f8c68e67&scene=21#wechat_redirect)**
- **[模块化 Triton GPU 内核代码生成 Agent 架构 GEAK：生成正确率达 63%，速度提升 2.59 倍！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890465&idx=1&sn=a73eec7d10391ef7371018529fb2405c&scene=21#wechat_redirect)**
- **[基于 LLM 跨硬件GEMM优化框架 QiMeng-GEMM：自动搜索元提示组合下特定规模显著优于cuBLAS、OpenBLAS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889763&idx=1&sn=21f80712d58b5241461c6a1fe4098951&scene=21#wechat_redirect)**

> 随着大型语言模型（Large Language Models, LLMs）的快速发展，对自动化、高性能系统 GPU 核函数的需求日益凸显，这类核函数已成为推动开发与部署效率提升的关键支撑。

本文提出**TritonRL** ——一款**专为 Triton 核函数生成设计的领域专用 LLM**，其训练依托于一种新颖的训练框架，能够实现稳健且自动化的核函数合成。

::: block-1
与通用编程语言不同，Triton 核函数生成面临数据稀缺、评估标准不完善等独特挑战，易受“奖励作弊”（指模型利用评估漏洞获取高奖励，而非真正满足任务核心目标，如直接调用高级库而非生成自定义核函数）影响。
:::

![文章插图](/articles/2025-11-08-autotriton-kernelllm-triton-tritonrl-sft-26ea6182/images/02.webp)

::: image-caption
图1：TritonRL的组件和工作流程。
:::

为解决这些问题，我们的方法通过端到端流程实现：

- 首先，在精心筛选的数据集上通过**监督微调** （Supervised Fine-Tuning, SFT）提炼 Triton 领域专属知识；
- 随后，借助**强化学习** （Reinforcement Learning, RL）进一步提升代码质量——该 RL 框架采用稳健、可验证的奖励机制与分层奖励分配策略。

我们的 RL 框架能有效检测奖励作弊行为，并通过细粒度验证与分层奖励分解，同时引导模型的推理轨迹与代码生成过程，最终使模型能够生成可真正替代现有模块的高质量 Triton 核函数。

通过稳健且细粒度的评估，我们在 KernelBench 基准测试（用于评估 LLM 生成 GPU 核函数能力的标准测试集）上的实验表明，**TritonRL 在正确性与加速比方面均达到当前最佳水平，性能超越所有其他 Triton 专用模型**，充分证明了我们基于 RL 的训练范式的有效性。

![文章插图](/articles/2025-11-08-autotriton-kernelllm-triton-tritonrl-sft-26ea6182/images/03.webp)
