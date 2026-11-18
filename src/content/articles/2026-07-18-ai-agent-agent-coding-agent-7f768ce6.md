关键词：**智能体工程**、上下文工程、工具调用、:mdnice-bold-italic[异步调度]、**自我进化**

> 过去一年，AI Agent 这个词被用得太多，也被解释得太轻。

很多介绍停留在“让大模型自己规划、自己调用工具、自己完成任务”的层面，听起来像魔法，落到工程现场却会立刻遇到一连串硬问题：

- 上下文应该怎么组织？
- 工具结果要不要全部塞回模型？
- 长期任务如何不中断用户交互？
- 一百多个工具全量注入会不会把模型压垮？代码生成 Agent 怎样避免把自然语言幻觉变成真实文件破坏？

![文章插图](/articles/2026-07-18-ai-agent-agent-coding-agent-7f768ce6/images/01.webp)

- :mdnice-bold-italic[《深入理解 AI Agent：设计原理与工程实践》（李博杰 著）开源主仓库：全书正文、编译版 PDF 与按章配套代码]
- https://github.com/bojieli/ai-agent-book/tags
- https://github.com/bojieli/ai-agent-book

::: related-title
**相关推荐**
:::

::: related-links
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**

:::

> bojieli/ai-agent-book[1] 的不只给出一套漂亮的 Agent 概念图，还把《深入理解 AI Agent：设计原理与工程实践》的全书正文、PDF、配图生成脚本和按章实验代码放在同一个仓库里。

README 明确把全书框架压缩为一个核心公式：**Agent = LLM + 上下文 + 工具**。但仓库真正有意思的地方在于，它**把这个公式拆成十章工程问题**：:mdnice-bold-italic[基础范式、上下文、记忆与 RAG、工具与 MCP、Coding Agent、评估、后训练、自我进化、多模态交互、多 Agent 协作。]

![文章插图](/articles/2026-07-18-ai-agent-agent-coding-agent-7f768ce6/images/02.webp)

::: image-caption
本书提到的大模型三种学习方式，按学习速度由慢到快排布：训练时的后训练，修改权重、永久通用但成本高更新慢；推理时的上下文学习，靠注意力临时适配，受窗口限制；运行时的外部化学习，依托知识库与工具，持久可更新、可靠可验证，三者分别给出对应实例，底部横轴标注学习速度从数周慢至毫秒级快。
:::

::: block-1
下面我们就:mdnice-highlight[沿着这本书的目录结构，看看一个严肃的 Agent 系统究竟需要哪些部件]，以及这些部件如何组合成可复现、可调试、可扩展的工程体系。
:::

## 本文目录

- 一、项目总览：一本书、十章实验与一条工程主线
- 二、核心公式落地：上下文是操作系统，工具是外设，模型是调度者
- 三、工具系统：从 MCP 到异步运行时，Agent 的难点在“行动管理”
- 四、Coding Agent：把代码生成从“写一段程序”升级为工具化工作循环
- 五、自我进化：少注入工具，多让 Agent 主动发现与复用
- 六、多 Agent 协作：隔离上下文、分工执行与中心协调
- 七、工程价值：这座仓库真正沉淀的是 Agent 系统的“可解释骨架”

![文章插图](/articles/2026-07-18-ai-agent-agent-coding-agent-7f768ce6/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[刚刚，SpaceXAI 开源 Grok Build：一个 Rust AI Agent Harness 的系统级拆解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903721&idx=1&sn=20b3a6235e2c8a3bae1df8e27de39301&scene=21#wechat_redirect)**
- **[下一代自进化 Agent 的真正瓶颈不是算法，是系统底座！蚂蚁联合港科大、清华揭开企业级智能体 RL 架构三大支柱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903671&idx=1&sn=ec3dbd910ce685748846b9053ca0d134&scene=21#wechat_redirect)**
- **[伯克利开源 CHIA：进化 Agent 打通 AI 芯片全链路，单框架实现仿真对齐/RTL 生成/时序优化五大工业级硬件自动化，最高 1.97 倍性能提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903506&idx=1&sn=27650a6cbe9467c5d3d0608e8839b15d&scene=21#wechat_redirect)**

:::
