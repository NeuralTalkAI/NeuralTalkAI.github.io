关键词：LLM、**内核生成**、:mdnice-bold-italic[自动化优化]、**智能体**、基准测试

> 现代人工智能系统的性能从根本上受制于底层内核的质量，这类内核可将高层算法语义转化为底层硬件操作。要实现接近最优性能的内核，需要研发人员具备专家级的硬件架构知识与编程模型认知，这使得内核工程成为一项至关重要但耗时冗长且不具备可扩展性的工作。

![文章插图](/articles/2026-01-27-kernel-llm-nvidia-amd-npu-89616777/images/01.webp)

- :mdnice-bold-italic[Towards Automated Kernel Generation in the Era of LLMs]
- https://arxiv.org/pdf/2601.15727
- **awesome 仓库：https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation**
- 1 万字，阅读 30 分钟，**播客 22 分钟**

相关推荐

- [HotChips 2025：从 PyTorch 到 CUDA：大模型时代 GPU 内核自动生成的技术跃迁，Tri Dao最新演讲](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891347&idx=1&sn=173b8638e00da3610dd90ba7d5a4b7f6&scene=21#wechat_redirect)
- [超越AutoTriton/KernelLLM的 Triton 内核生成模型 TritonRL：融合SFT蒸馏与RL分层奖励设计](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893752&idx=1&sn=e5fdd6a30c934972634fa013748e8333&scene=21#wechat_redirect)
- [CUDA Kernel 自动生成新范式！多轮RL+反馈迭代方案 Kevin 让生成速度超o4-mini 41%、正确性领先44%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891083&idx=1&sn=511eaf94b8c3948457d107d8e8853641&scene=21#wechat_redirect)

> 近年来，大语言模型（LLMs）以及基于大语言模型的智能体取得了长足发展，为内核生成与优化的自动化开辟了全新可能。

- :mdnice-bold-italic[大语言模型能够有效凝练那些难以形式化表达的专家级内核知识]
- 而:mdnice-bold-italic[智能体系统则通过将内核开发构建为迭代式、反馈驱动的循环流程]，进一步实现了可扩展的优化。

![文章插图](/articles/2026-01-27-kernel-llm-nvidia-amd-npu-89616777/images/02.webp)
