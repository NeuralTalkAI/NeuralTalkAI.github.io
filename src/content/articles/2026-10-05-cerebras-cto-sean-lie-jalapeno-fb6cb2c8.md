关键词：Cerebras、**晶圆级芯片**、:mdnice-bold-italic[超快推理]、CS5、Sean Lie

> 如果把大模型推理比作一场竞速，Sean Lie 大概就是那个把引擎做得最大、也最不怕别人说他疯的人。:mdnice-bold-italic[他是 Cerebras 的联合创始人兼 CTO，也是那整块晶圆级芯片（wafer-scale）背后的首席架构师——当年几乎所有人都对他们说"你们疯了，这东西造不出来"，而如今 Cerebras 不仅把晶圆级芯片做成了量产产品，还刚刚完成了史上规模最大的半导体 IPO。]

![文章插图](/articles/2026-10-05-cerebras-cto-sean-lie-jalapeno-fb6cb2c8/images/01.webp)

::: image-caption
Sean Lie 是 Cerebras Systems 联合创始人兼 CTO、首席硬件架构师。他拥有麻省理工学院电子工程与计算机科学学士及硕士（MEng）学位，研究方向为计算机体系结构，专长软硬件协同设计，持有 16 项相关专利。职业生涯早期在 AMD 高级架构团队工作五年；2008 年加入微服务器创企 SeaMicro，任 IO 虚拟化 fabric ASIC 首席硬件架构师。2012 年 SeaMicro 被 AMD 以约 3.3 亿美元收购后，他获任 AMD Fellow（院士）及首席数据中心架构师。2015 年底他与 Andrew Feldman 等五人再度聚首创立 Cerebras，主导晶圆级引擎（WSE）芯片的硬件设计，2022 年 4 月起出任 CTO。2026 年 5 月 Cerebras 登陆纳斯达克，创下史上最大半导体 IPO 纪录。
:::

- **The Inference Frontier: from 100 to 10,000 tokens per second — Sean Lie, Cerebras CTO**
- https://www.youtube.com/watch?v=3uSI8q_RN-o

::: related-title
**相关推荐**
:::

::: related-links
- **[Cerebras 创始人与 Black Forest Labs CEO 访谈：算力爆发、智能主权、AI 基础设施、递归学习、生成式内容的下一站](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905833&idx=1&sn=d2e2df7d334b498eae0892d52ec81ba7&scene=21#wechat_redirect)**
- **[NPU 设计的系统级艺术：权衡、适配与协同优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897198&idx=1&sn=4013a7097a200009e9439429e7ea61b9&scene=21#wechat_redirect)**
- **[Meta AI 基础设施的十年演进、下一代基础设施蓝图与 AI 自研芯片之路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896790&idx=1&sn=a13b82d8c267fcdbd6f3e20c53a1429b&scene=21#wechat_redirect)**

:::

> 这期 Latent Space 播客录制于 Hot Chips 大会次日，地点就在 Cerebras 总部。Sean Lie 带着大会余温，系统性地讲了下面四件事。

::: related-links
- 刚发布的 **CS4 系统如何把推理速度再翻一倍、跑出 GPT-OSS 超过 4,400 tokens/s 的演示**；
- 基于 Nexus 模块化平台的 CS5 预告——**中型模型冲 10,000 tokens/s、Kimi、DeepSeek 这类前沿模型冲 5,000 tokens/s**；
- 与 OpenAI 的深度绑定——对方不仅是最大客户，还用 Cerebras 跑自己最关键的应急响应和研究任务，**双方甚至进入了模型与芯片协同设计的阶段**；
- 以及他对行业热点的一系列不客气判断：OpenAI 自研芯片 **Jalapeno 的真正意义不在性能而在"AI 优先"的设计方法论，Groq 被英伟达收编后只敢拿 31B 小模型跑分暴露了非晶圆级 SRAM 架构的容量天花板**，Etched 至今没有 benchmark 只能算"很厉害的图形设计"。

:::

::: block-1
他还透露 :mdnice-highlight[Cerebras 两年前就启动了 3D DRAM 堆叠项目，并判断开源模型市场"95% 来自中国"是美国必须上升到国家战略层面应对的问题]。用他的话说：过去被认为很快的 100–200 tokens/s，正在变成新的"批处理模式"。
:::

## 本文目录

::: related-links
- 一、Hot Chips 次日：硬件的黄金时代
- 二、CS4：把晶圆级计算推向超大规模
- 三、从技术演示到解决真实问题
- 四、CS5 预告：10,000 TPS 不是终点
- 五、与 OpenAI 的关系：产能、内部使用与共同设计
- 六、行业热点点评
- 七、工作负载观：把数据中心当成一颗芯片来设计
- 八、中美竞争：开源模型市场 95% 是中国的
- 九、速度的尽头，是新能力的起点

:::

![文章插图](/articles/2026-10-05-cerebras-cto-sean-lie-jalapeno-fb6cb2c8/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[Jeff Dean 深度访谈：推理硬件特化、AI 自我进化、创始人品味修炼、餐巾纸计算、1% 创业法则](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905136&idx=1&sn=aca21aa6380f31f7e3eeb091d8b2702a&scene=21#wechat_redirect)**
- **[从 DeepMind研究到产品落地，Google Cloud工程副总裁深度访谈：十万 TPU 集群、Gemini研产全链路、从预训练到端侧部署、下一代AI基础设施](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904953&idx=1&sn=f37d34a9a0dcd0e989dac8f0e81b527b&scene=21#wechat_redirect)**
- **[SemiAnalysis 创始人深度访谈：AI 算力基建的下一个十年，半导体研究方法论，AI 软硬件协同的全产业链棋局](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904797&idx=1&sn=9261153c86d9b591bf51ec67f2603821&scene=21#wechat_redirect)**

:::
