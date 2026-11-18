关键词：:mdnice-bold-italic[CUDA 内核生成]、**智能体强化学习**、性能优化、KernelBench、**技能增强环境**

> 注：本文部分结论有误导，请在留言区直接开喷，见留言知乎链接。在深度学习基础设施的底层，有一个被称为“黑魔法”的领域——CUDA 内核开发。当我们在 PyTorch 中写下几行简洁的代码时，很少有人意识到，这些高层操作最终会被编译成成百上千个在 NVIDIA GPU 上执行的底层内核程序。:mdnice-bold-italic[这些内核程序的编写和优化，决定了我们训练模型的速度、能效，甚至是可行性。]

然而，CUDA 编程的门槛高得令人望而却步。开发者不仅要理解 GPU 的微架构特征如共享内存、寄存器文件、张量核心，还要精通性能剖析工具，手动处理内存访问模式、线程块调度、指令级并行等一系列底层细节。**即便对于经验丰富的工程师，优化一个内核也可能需要数周时间。**

::: block-1
大语言模型在通用编程任务上已经展现出媲美人类的能力，但在 CUDA 内核生成这个专业领域，它们甚至无法击败 PyTorch 内置的 `torch.compile` 编译器。这个尴尬的现实引发了学术界的思考：**如何让 AI 真正掌握 CUDA 优化这门“黑魔法”？**
:::

![文章插图](/articles/2026-03-03-kernelbench-seed-cuda-agent-cuda-8ec0b3c6/images/01.webp)

- **CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation**
- https://arxiv.org/pdf/2602.24286v1
- **代码 & 数据：https://cuda-agent.github.io**
- 1 万字，阅读 30 分钟，播客 17 分钟

相关推荐

- [跨平台 Kernel 自动生成综述：基于 LLM 的监督微调+强化学习，适配 NVIDIA/AMD/华为 NPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896690&idx=1&sn=c57e7dc9b3f7aa251d2815304613d94c&scene=21#wechat_redirect)
- [鹏城联合华为昇腾、中山大学提出AscendKernelGen：NPU 算子闭环生成与评估，复杂L2 Kernel编译成功率95.5%，正确率64.3](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896566&idx=1&sn=d920e9f8feed38e4734d18942d83897a&scene=21#wechat_redirect)
- [决战性能之巅！MLSys 2026 CUDA 内核优化大赛开启，用代码挑战 B200 GPU 极限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896612&idx=1&sn=4e667f93479948119cf7688933ca966c&scene=21#wechat_redirect)

> 2026 年 2 月，清华大学 AIR 团队和字节跳动 Seed 团队联合发表了一篇重磅论文《CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation》，给出了他们的答案。

在这篇文章中，我们将深入解读这项工作，看他们如何通过大规模智能体强化学习系统，让 AI 学会了“手撕”CUDA 内核，并在多个难度级别上超越了最先进的开源模型和闭源模型。

![文章插图](/articles/2026-03-03-kernelbench-seed-cuda-agent-cuda-8ec0b3c6/images/02.webp)

::: image-caption
统计图从正确率、快进率、加速比三类核心指标，对比了 GLM 4.6、Kimi K2、Gemini 3 Pro、Claude Opus 4.5 与 CUDA Agent（ours）的 CUDA 内核生成性能。在快进率（相比 torch.compile）和加速比指标上，CUDA Agent 表现断层领先，快进率接近 97%，加速比达 2.11×，远超其他模型；Claude Opus 4.5 和 Gemini 3 Pro 虽正确率达 90% 以上，但快进率仅 66%-69%，优化能力有限；GLM 4.6 和 Kimi K2 在各项指标上均明显落后。图表直观印证了 CUDA Agent 通过专用强化学习训练，在 CUDA 内核优化能力上实现对通用大模型的超越，凸显了其在硬件感知优化上的核心优势。
:::
