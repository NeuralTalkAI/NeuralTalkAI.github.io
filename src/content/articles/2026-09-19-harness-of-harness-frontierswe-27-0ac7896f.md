关键词：**自主软件开发**、编码智能体、**Agent Harness**、迭代增量开发、证据驱动

> 编码 Harness 负责组织模型的一次运行，而 Harness of Harness (简称 HoH 😅) 负责组织 Harness 在整个长程开发中的全程参与，这句话浓缩了全文的野心👀。

![文章插图](/articles/2026-09-19-harness-of-harness-frontierswe-27-0ac7896f/images/01.webp)

- **Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement**
- 组织机构：上海人工智能实验室（Shanghai Artificial Intelligence Laboratory）
- 论文链接：https://arxiv.org/abs/2609.01481v1
- 开源代码：https://github.com/Flesymeb/HarnessOfHarness
- 项目主页：https://flesymeb.github.io/HarnessOfHarness/

::: related-title
**相关推荐**
:::

::: related-links
- **[AI 递归自进化下，量化前沿实验室真实研发节奏与算力分配，解读Anthropic 三大 AI 研发度量体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905979&idx=1&sn=33d553605fb9915585470a14b1d98dad&scene=21#wechat_redirect)**
- **[NVIDIA 开源 SoL-Pi 用递归自进化把 Agent Harness 成本压低 50%–54%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905961&idx=1&sn=9ca43c638ab7ca5cc429ae6b076a9173&scene=21#wechat_redirect)**
- **[从盲搜到自进化，MicroEvo 让 LLM 在十万亿的 CPU 微架构设计空间探索中积累经验、逃离局部最优](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905180&idx=1&sn=a09f367a0cf70c82cd47805e347de448&scene=21#wechat_redirect)**

:::

> 今天的编码智能体已从函数补全进化到仓库级 Issue 修复，却大多仍活在人在环中（human-in-the-loop）模式里：:mdnice-bold-italic[人定任务、人盯过程、人审代码、人在报错时把 Agent 重新拉起。这篇工作瞄准更激进的自主软件开发（autonomous software development）：只给一份高层需求，Agent 从零开始，无人干预地交付完整、可运行、可部署的软件系统。]

:mdnice-bold-italic[长程开发失败，根源不在 Agent 跑得不够久，核心问题是它无法在超长轨迹上维持连贯、有效的进展。]

::: related-title
**相关推荐**
:::

::: related-links
- **[从 EDA 软件到硬件 RTL：NVIDIA 仓库级代码自进化智能体框架 HORIZON，重构大模型芯片设计演化范式](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903526&idx=1&sn=52728aef42e16a67281889d8c21d54fe&scene=21#wechat_redirect)**
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[NVIDIA Research：LLM 的 RTL 硬件编码上限，被预训练阶段的知识储备牢牢锁死](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903332&idx=1&sn=3935b6554883f39d136eec5188c7eeb6&scene=21#wechat_redirect)**

:::
