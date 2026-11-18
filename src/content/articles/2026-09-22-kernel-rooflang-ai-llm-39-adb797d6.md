关键词：**RoofLang**、:mdnice-bold-italic[推理架构搜索]、Roofline建模、**AI Agent**、MoE推理

> 被 profiling 绑定的反馈，会把搜索困死在现有软件栈的能力与性能边界内，让那些本质上更优的架构永远没有机会被发现。

过去一年，KernelBench、KDA-Pilot、KernelPro 等“AI 写 kernel”的工作把 Agent 推上算子优化一线：给 profile、给 benchmark、给编译器报错，就能反复迭代出媲美手写的 CUDA kernel。:mdnice-bold-italic[但这类闭环本质上是“profiling-bound”，反馈来自真实软件栈，Agent 能改的只有软件栈允许它改的东西。一个理论上更优的架构，如果当前框架不支持、通信库缺算子、kernel 没优化好，Agent 连它的存在都探测不到。]

::: block-1
核心问题由此清晰：:mdnice-highlight[如何让 AI 【越过具体软件栈】，在“推理系统架构”层自主完成设计、评估与迭代？]
:::

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/01.webp)

- **RoofLang: Enabling AI-Driven Architecting of LLM Inference Systems**
- 组织机构：上海 Xingyunzhili 人工智能研究院；微软研究院
- 论文链接：https://arxiv.org/abs/2609.12551
- 开源代码：github.com/yzygitzh/rooflang
- 项目主页：yzygitzh.github.io/rooflang

::: related-title
**相关推荐**
:::

::: related-links
- **[MINISA：面向下一代可重构推理加速器的最小指令集架构，把 96.9% 指令停顿打到接近零](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901107&idx=1&sn=2b04a2a2493ac8107f13e7467b430cff&scene=21#wechat_redirect)**
- **[OpenAI 核心研究员 Noam Brown 深度拆解 Agentic AI、推理与多智能体的技术真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905895&idx=1&sn=6a5d81ee886d6992b05eb8061b077599&scene=21#wechat_redirect)**
- **[ArchAgent: 智能体驱动的计算机体系结构自动发现，实现 5.3%多核性能突破与 3-5 倍加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898414&idx=1&sn=d58a7e936f83b566db9a0f5d91e6266c&scene=21#wechat_redirect)**

:::

> 答案是 RoofLang，**一个面向 LLM 推理系统的 DSL，三要素为：与框架无关的计算图通用负载表示、一组把“架构动作”约束成数学可验证等价变换的语义保持（semantics-preserving）图改写原语、一个基于 Roofline 的离散事件仿真器**，可不跑任何真实 kernel 就评估架构好坏，:mdnice-bold-italic[三者拼成 AI 可访问的自主架构设计闭环。]

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/02.webp)

::: image-caption
图 2：RoofLang 的整体系统架构图，完整展示这套面向大模型推理系统设计的领域专用语言工作流程。该图从输入计算图开始，依次呈现语义保序变换、硬件资源映射放置，结合硬件图执行仿真，最终输出追踪日志的全链路。它直观说明 RoofLang 不依赖真实硬件剖面，支持 AI 驱动的推理架构搜索，是整篇论文的核心框架可视化，帮助读者理解如何脱离现有软件栈约束，探索全新 LLM 推理系统方案。
:::

在 H200、GH200、B300、GB300 四代硬件，DeepSeek V4 Flash、DeepSeek V4 Pro、GLM-5.3、Kimi K3 四个 2026 年旗舰 MoE 模型，**64K 与 1M 上下文的大规模扫描中，作者发现 DeepSeek V4 系列的峰值解码吞吐可达其他模型的 3.5–39.5 倍。**

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/03.webp)

::: image-caption
表 7：RoofLang 搜索得到的最优推理架构在 64 张 GB300 显卡、64K 上下文条件下，解码阶段内存开销不同模型对比表。表格量化对比各模型 KV‑Cache 占用、最大批处理规模、单 token 内存流量。实验数据证明 DeepSeek V4 系列凭借紧凑 KV‑Cache 设计，实现远超其他竞品的解码吞吐，解释了性能优势并非单纯源于参数量，为论文核心结论提供关键定量支撑，也是评估不同模型推理内存压力的核心实验结果。
:::

随后他们让一个基于 Claude Code、以 DeepSeek V4 Pro 为主模型的优化 Agent 在 RoofLang 中自主探索，**Agent 发现三种人工基线没有采用的新架构，在 NVIDIA B300 上把 DeepSeek V4 Pro 的吞吐与交互性同时提升 6.23%–50.1%。**

架构层面的红利不仅真实存在，而且已经大到足以被 AI 自动挖掘。

## 本文目录

- 一、Profiling 闭环的天花板：AI 优化需要一张“架构图纸”
- 二、三要素拼成的闭环：RoofLang 如何让 AI“看懂并改写”架构
- 三、两张图与一套原语：Agent 的“世界模型”和“动作集”
- 四、不跑一行 kernel 怎么判快慢：Roofline 离散事件仿真器
- 五、四大模型 × 四代硬件：仿真扫描挖出的模型级架构洞察
- 六、Agent 自主架构师：Claude Code 在 B300 上找到的三种新架构
- 定位、边界与启示：架构层与实现层的显式分工

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[不止算子优化！ncnn Vulkan int8 量化推理后端：全链路架构设计与系统性优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902586&idx=1&sn=14f256d165dccfb6c4ef866fd144ab6b&scene=21#wechat_redirect)**
- **[MINISA：面向下一代可重构推理加速器的最小指令集架构，把 96.9% 指令停顿打到接近零](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901107&idx=1&sn=2b04a2a2493ac8107f13e7467b430cff&scene=21#wechat_redirect)**
- **[比NVIDIA Thor‑U快3.8倍，却少用一个缓存层级：理想汽车M100编排式数据流架构对AI推理原有设计信条的根本性质疑](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900845&idx=1&sn=0044def693c7c0eb4d03f91096349b9d&scene=21#wechat_redirect)**
- **[OpenAI 核心研究员 Noam Brown 深度拆解 Agentic AI、推理与多智能体的技术真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905895&idx=1&sn=6a5d81ee886d6992b05eb8061b077599&scene=21#wechat_redirect)**
- **[AI 递归自进化下，量化前沿实验室真实研发节奏与算力分配，解读Anthropic 三大 AI 研发度量体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905979&idx=1&sn=33d553605fb9915585470a14b1d98dad&scene=21#wechat_redirect)**

:::
