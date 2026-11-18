关键词：**记忆增强智能体**、记忆重构、:mdnice-bold-italic[组相对策略优化]、序列决策、**分布外泛化**

> **人类的记忆是重构性的，而非复制性的**。这是认知心理学领域流传半个世纪的经典结论，如今正在成为破解大语言模型智能体记忆困境的关键钥匙。

**当前主流的记忆增强智能体普遍遵循检索即重放的逻辑**，将语义匹配的历史经验直接注入上下文，却:mdnice-bold-italic[忽略了经验产生时的环境状态与决策时刻的真实状态之间的鸿沟，最终频繁出现经验越丰富、决策越跑偏的负迁移现象。]

![文章插图](/articles/2026-08-03-memharness-grpo-gpt-4o-7b-f36a1fb6/images/01.webp)

- **MemHarness: Memory Is Reconstructed, Not Replayed**
- https://arxiv.org/pdf/2607.28272
- https://github.com/KnowledgeXLab/MemHarness

::: related-title
**相关推荐**
:::

::: related-links
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**

:::

> 这项来自上海人工智能实验室联合浙江大学、复旦大学等机构的研究，**提出了名为 MemHarness 的智能体框架，首次将认知科学中的重构性记忆机制完整引入智能体决策流程**。

该框架在记忆检索与行动生成之间插入显式的批判与重构环节，由同一套策略模型完成状态对比、经验适配与行动输出，整套能力通过 GRPO 强化学习端到端训练自发涌现，无需任何重构相关的人工标注。

实验结果显示，**仅使用 7B 参数** 的骨干模型，MemHarness 就在：

- :mdnice-bold-italic[ALFWorld 基准上达到 85.2%的平均成功率，超出纯 RL 基线 8.8 个百分点，超出 Gemini-2.5-Pro 达 23.1 个百分点]；
- 在 WebShop 基准上达到 75.6%的成功率，:mdnice-bold-italic[超出纯 RL 基线 9.5 个百分点，超出 Gemini-2.5-Pro 达 39.7 个百分点。]

更重要的是，**在分布外场景下其优势进一步扩大**，同时重构训练目标本身还能隐式提升模型的基础推理能力，为记忆增强智能体的工程落地扫清了核心障碍。

## 本文目录

- 一、智能体记忆的困局：静态重放为何频频帮倒忙
- 二、认知科学的跨界启发：记忆的本质是重构而非复制
- 三、MemHarness 框架详解：一套统一策略完成检索、批判与行动
- 四、训练范式：GRPO 如何驱动重构能力无监督涌现
- 五、基准测试：7B 模型如何超越闭源大模型与纯 RL 基线
- 六、机制深挖：重构模块为什么是性能跃迁的核心
- 七、行业视角：记忆重构范式对 AI 工程落地的深层价值

![文章插图](/articles/2026-08-03-memharness-grpo-gpt-4o-7b-f36a1fb6/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**
- **[20 万 Star 的 Superpowers 技能框架，让编程智能体用TDD测试文档、用子智能体驱动开发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901506&idx=1&sn=0b27159dd6aa1892e6ef14a527765d5b&scene=21#wechat_redirect)**

:::
