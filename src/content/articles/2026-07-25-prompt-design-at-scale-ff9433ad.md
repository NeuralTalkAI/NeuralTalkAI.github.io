关键词：**提示词工程**、:mdnice-bold-italic[指令遵循]、长上下文、**幻觉**、拒答率

> 在日常的大模型应用开发里，有两个极其高频的设计决策，至今几乎没有严谨的对照实验给出定论：

- 一是**提示词的呈现格式**：给模型的指令和参考资料，用 Markdown、纯文本、连贯散文还是表格来写，效果最好？
- 二是**规模边界**：单轮对话里最多能给模型塞多少条并行指令？上下文窗口拉到多长时，模型的记忆力和诚实度会开始崩塌？

::: block-1
格式之争在技术社区吵了一轮又一轮，而“规模边界”这个更本质的问题，却常常被默认“模型能hold住”。

这篇《Prompt Design at Scale: How Format, Instruction Count, and Context Length Shape Instruction Adherence and Hallucination in Large Language Models》用两组严格受控的实验，首次把**提示词格式、指令数量、上下文长度** 三个变量放在同一份无污染合成语料上交叉验证：:mdnice-highlight[同一份底层内容渲染成四种格式，沿着“指令数量”和“上下文长度”两条轴逐步加码，直到模型能力出现断崖式下跌，完整观测整个衰减过程。]
:::

![文章插图](/articles/2026-07-25-prompt-design-at-scale-ff9433ad/images/01.webp)

- :mdnice-bold-italic[Prompt Design at Scale: How Format, Instruction Count, and Context Length Shape Instruction Adherence and Hallucination in Large Language Models]
- https://arxiv.org/pdf/2607.19257

::: related-title
**相关推荐**
:::

::: related-links
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**
- **[刚刚，SpaceXAI 开源 Grok Build：一个 Rust AI Agent Harness 的系统级拆解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903721&idx=1&sn=20b3a6235e2c8a3bae1df8e27de39301&scene=21#wechat_redirect)**
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**

:::

> 核心结论一句话概括：**一种格式好不好用，完全取决于它承载了多少条指令、放在多长的上下文里**，脱离规模谈“最优格式”没有任何意义。

10条指令时，Markdown和纯文本表现几乎没有差别；100条时两者差距急剧拉开。

2000 token上下文里，表格和段落的召回效果一致；拉到50万token时，两者表现天差地别。

- **实验一（指令数量维度）**：当并行规则增加到80条时，五款被测模型的**完美响应率全部归零** ——完美响应率指的是所有规则都被严格满足的回答占比。:mdnice-bold-italic[这一结果在四种格式（Markdown、纯文本、散文、表格）、两种指令放置位置（系统提示/用户轮次）下完全一致]。*注：纯文本即剥离所有Markdown标记后的内容；散文指把实体的键值对描述整合成连贯段落、不强制换行。*
- **实验二（上下文长度维度）**：:mdnice-bold-italic[召回准确率在64k~128k token以内基本维持满分，超过这个区间后，不同格式的表现开始分化崩塌]。真正随上下文长度飙升的是**直接拒答率**，从0%一路涨到79%~90%。

> 基于实验结果，论文提炼了五条可直接落地的提示词设计指导原则：

| 指导要点 | 具体说明 |
| --- | --- |
| 无通用最优格式 | 最佳格式与模型强相关，不存在跨模型通用的答案。例如Qwen 35B明确偏好纯文本，换另一款模型则可能完全相反。 |
| 指令放置位置影响显著 | 同一份指令放在系统提示或用户消息中，遵守率波动最高可达8.7个百分点（N=160时），且影响方向无法提前预判，必须针对目标模型实测。 |
| 单轮指令设40条为警戒线 | 所有模型在指令数达到40条后性能进入陡坡，80条时几乎完全触底。超限时应拆分为多轮交互、工具调用或后置校验，而非继续打磨单条提示词。 |
| 长上下文优先监控拒答率，而非幻觉 | 实验中事实捏造率为0，谄媚率低于8.3%，但拒答率最高达89.6%。只监控“答错”会忽略长上下文下最高发的故障：模型直接选择不回答。 |
| 成本校正仅在准确率有差距时有意义 | 若不同格式准确率接近，选token开销最低的即可；当准确率出现明显差距时，22%~37%的token开销差足以反转性价比排名，需要单独测算。 |

## 本文目录

- 一、实验一：80条规则下，五款模型全部触底归零
- 二、实验二：512k上下文阶梯上，幻觉没来，拒答先来了
- 三、没有常胜的提示词格式，还要算一笔token成本账
- 最后：五条可落地的实践建议

![文章插图](/articles/2026-07-25-prompt-design-at-scale-ff9433ad/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**
- **[从 EDA 软件到硬件 RTL：NVIDIA 仓库级代码自进化智能体框架 HORIZON，重构大模型芯片设计演化范式](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903526&idx=1&sn=52728aef42e16a67281889d8c21d54fe&scene=21#wechat_redirect)**

:::
