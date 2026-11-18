关键词：**Jeff Dean**、Google DeepMind、Discovery Loop、**Anthropic 自研芯片**、:mdnice-bold-italic[AI 基础设施]

> 2026 年 8 月 5 日，AI 行业在同一天爆出两则指向同一趋势的重磅消息。

其一是谷歌宣布 2023 年合并 Google Brain 与 DeepMind 以来最大规模的 AI 权力重排：Google DeepMind CEO Demis Hassabis 交出日常运营，转任 GDM 董事长兼 Alphabet 首席科学家，继续执掌 Isomorphic Labs；CTO Koray Kavukcuoglu 升任高级副总裁，直接向 Pichai 汇报，统管 Gemini 模型研发、前沿研究及 Gemini 应用与开发者团队。同日，**效力谷歌近 27 年的首席科学家 Jeff Dean 离职**，与 Sanjay Ghemawat、Oriol Vinyals、Quoc Le :mdnice-bold-italic[创办公益型公司 Discovery Loop，用 AI 自动化"提出实验—执行—评估—迭代"的科学发现闭环]，Khosla Ventures 与 Radical Ventures 领投，Alphabet 参投并提供首年算力。

其二是 **Anthropic 首次公开确认组建内部芯片设计团队，为 Claude 开发定制半导体**，以 32 万至 48.5 万美元年薪招聘芯片工程师，此前已与三星洽谈 2 纳米代工、从 OpenAI 挖来芯片核心成员 Clive Chan。

两则消息共同揭示：AI 竞争正从模型层沉向组织与芯片两大底层。

## 本文目录

- 一、谷歌 AI 权力重排：Hassabis 退居幕后，Jeff Dean 带队出走
  - 1.1 谷歌：一次官宣，两场地震
  - 1.2 Hassabis：从经营者退回战略家
  - 1.3 Koray 接棒：Gemini 的权力集中
  - 1.4 Jeff Dean：27 年后的告别
  - 1.5 Discovery Loop：一支"梦之队"要做什么
  - 1.6 融资与一场"体面的分手"
- 二、Anthropic 官宣造芯：Claude 的"芯"征程
  - 2.1 Anthropic：同一天官宣，把手伸向芯片
  - 2.2 招聘细节：48.5 万美元年薪与八大领域
  - 2.3 "多芯片策略"的延伸，而非替代
  - 2.4 产业链布局：三星代工与关键挖角
- 三、行业格局与共同逻辑：AI 竞争沉向底层
  - 3.1 行业格局：OpenAI 已先行一步
  - 3.2 从"买芯片"到"造芯片"的必然
  - 3.3 两条新闻，一个共同方向

![文章插图](/articles/2026-08-06-jeff-dean-anthropic-ai-e56cb589/images/01.webp)

## 一、谷歌 AI 权力重排：Hassabis 退居幕后，Jeff Dean 带队出走

### 1.1 谷歌：一次官宣，两场地震

当地时间 8 月 5 日，谷歌及母公司 **Alphabet 宣布对 AI 权力中心进行系统性重排**。谷歌 CEO Sundar Pichai 在Google 官方博客[1]发布的致员工信。Pichai 在信中先晒出了底气：Gemini 模型在开发者和企业中需求旺盛，Gemini 应用月活用户已达 9.5 亿以上，Gemma 模型下载量突破 9 亿。

### 1.2 Hassabis：从经营者退回战略家

**Demis Hassabis 不再担任 Google DeepMind CEO**、不再负责日常运营，:mdnice-bold-italic[转任 GDM 董事长和 Alphabet 首席科学家，同时继续担任 AI 制药公司 Isomorphic Labs 的 CEO]。他在内部信中解释：**他一生都在追求 AGI，而现在 AGI 已经"近在眼前"，** 因此要交出日常经营，专注 AGI 战略、全球政策和长期科研方向，同时为 GDM 的模型和研究团队提供建议。

他明确表示未来会进一步投入 Isomorphic Labs——他一直将改善人类健康视为 AI 最重要的应用，希望"帮助治愈癌症这样的疾病"，以此证明 AI 能为现实世界带来"明确而无可争议的价值"。从组织架构看，已是 Google DeepMind 实际管理权的转移。

### 1.3 Koray 接棒：Gemini 的权力集中

**接替 Hassabis 掌管 GDM 日常业务的是 CTO、谷歌首席 AI 架构师 Koray Kavukcuoglu**。他升任 GDM 高级副总裁，直接向 Pichai 汇报，负责 Gemini 模型开发、前沿 AI 研究，以及 Gemini 应用和开发者团队——相当于把此前相对分散的模型研发与产品落地权力集中到一名负责人手中。

谷歌没有再设新的 GDM CEO，而是以"高级副总裁"头衔承接运营工作。:mdnice-bold-italic[Koray 是 DeepMind 早期成员，在公司超过 13 年，组建了其深度学习团队，主导过 DQN、WaveNet 等突破。Hassabis 透露，Gemini 模型"已经由 Koray 和相关负责人管理了一段时间"，包括 Gemini 4 在内的新模型正在取得进展。]

### 1.4 Jeff Dean：27 年后的告别

> 相比 Hassabis 的职位变化，Jeff Dean 的离开对谷歌技术体系的冲击更直接。

Dean 于 1999 年以谷歌第 30 号员工左右的身份加入公司，与 Sanjay Ghemawat 长期合作，参与设计了 Google File System、MapReduce、BigTable 等大规模计算基础设施，并在 2011 年前后共同推动 Google Brain 成立。2023 年 Google Brain 与 DeepMind 合并后，他出任谷歌首席科学家，负责 Google Research 和 Google DeepMind 的科研方向，并成为 Gemini 的技术联合负责人。

Pichai 在信中确认：经过"不可思议的 27 年"，:mdnice-bold-italic[Dean 与谷歌高级研究员 Sanjay Ghemawat 将创办一家独立的公益型公司（public benefit corporation），目标是"加速机器学习、科学和工程领域的发现"。谷歌将以创始投资者和云合作伙伴的身份继续合作，并围绕机器学习系统和基础设施建立研究合作框架。]

### 1.5 Discovery Loop：一支"梦之队"要做什么

> 完整的创业阵容在 Wired 的独家专访[2]提到：四位联创是 Jeff Dean、Sanjay Ghemawat、Oriol Vinyals（此前担任 DeepMind 研究副总裁、Gemini 技术负责人）和 Quoc Le（Google Brain 联合创始人之一，长期从事神经网络、AutoML 和大模型研究）。

**四人曾参与 Google Search、TensorFlow、TPU、Pathways、AlphaFold、AlphaStar、AlphaCode、Gemini、混合专家模型和序列到序列模型等项目，流失的人才覆盖从芯片、基础设施到模型和产品的多个关键层级**。路透社指出，Dean、Vinyals 等人此前一直深度参与 Gemini 模型开发。:mdnice-bold-italic[换句话说，谷歌失去的不只是一位首席科学家，仔细看，是一支横跨分布式系统、基础模型、机器学习架构和 Gemini 研发的核心团队。]

::: block-1
Discovery Loop 注册为公共利益公司，:mdnice-highlight[核心设想是用 AI 自动完成科学研究和工程研发的完整实验循环：提出实验、执行实验、评估结果，再根据结果继续迭代]。
:::

路线图分两步：

- 首先从机器学习研究入手，通过自动化实验改进自己的模型和技术栈——Quoc Le 说"说不定我们会发现一种不同的 Transformer 架构"；
- 再把这套"核心智能"扩展到芯片设计、药物研发、材料科学、清洁能源和网络安全等领域。

**其目标是让系统并行运行成千上万个实验，缩短传统科研依赖人工顺序迭代的时间**。Vinyals 点出了当前模型的短板："我们显然会非常关注模型如何产生值得尝试的新想法，这目前还不是它们的强项。"

### 1.6 融资与一场"体面的分手"

> Discovery Loop 首轮融资由 Radical Ventures 和 Khosla Ventures 共同领投，Alphabet、Lightspeed、Kleiner Perkins 等机构参与，金额与估值未披露。Vinod Khosla 的评价是："人类一直在用 AI 做研究，而不是让 AI 成为研究者。Discovery Loop 的根本之处，是 AI 就是研究者。"

据《连线》报道，**Pichai 曾通过多次会谈尝试挽留四位创始人，最终未能成功——他们离开的原因之一，是希望摆脱大型组织在推动激进技术变化时存在的惯性**。谷歌则同意投资，并在公司成立第一年提供算力支持。双方没有彻底切割：Alphabet 成为创始投资者，Google Cloud 成为云合作伙伴。

最有戏剧性的细节是：7 月 25 日，Dean 在旧金山 YC Startup School 对约 6000 名创业者演讲，被问到创业方向建议时，他认真描述了"自动化科学方法"的愿景——当时台下无人知道，他刚刚组建好的秘密创业公司干的就是这件事🤣。直到 7 月底，他还以谷歌首席科学家身份公开分析推理硬件与能耗趋势，离职谈判与公开履职同步进行，分手过程相当职业。

## 二、Anthropic 官宣造芯：Claude 的"芯"征程

### 2.1 Anthropic：同一天官宣，把手伸向芯片

> 同样是 8 月 5 日，Anthropic 首次公开确认正在组建内部芯片设计团队，为其 Claude 系列模型开发定制半导体。**发言人表示，公司将"协同设计硬件与模型"，使 Claude 在满足客户所需的大规模应用中"能够运行得更快、更高效"。**

### 2.2 招聘细节：48.5 万美元年薪与八大领域

Anthropic 在官方招聘平台 Greenhouse 发布了"Silicon Engineer"职位，年薪范围 32 万至 48.5 万美元。:mdnice-bold-italic[招聘启事写道："该角色适合那些拥有芯片出货经验、对时间表有实际认识、并能够在没有庞大组织支持的情况下自如做出重大决策的人。"招聘覆盖八大技术领域：前端设计、预硅验证、物理设计、可测试性设计、模拟与混合信号、工艺与代工、CAD 与设计基础设施、封装及信号/电源完整性。]

### 2.3 "多芯片策略"的延伸，而非替代

> Anthropic 发言人特别强调，**自研芯片并非要替代现有供应商，而是"多芯片策略"（multi-chip approach）的深化** ——来自亚马逊 AWS、谷歌、英伟达和 AMD 的硬件仍将是其算力扩张的核心。:mdnice-bold-italic[这与公司 4 月 6 日的官方公告一致：当时 Anthropic 宣布与谷歌和 Broadcom 签署多吉瓦级下一代 TPU 容量协议，并明确表示"我们在多种 AI 硬件上训练和运行 Claude——AWS Trainium、谷歌 TPU 和英伟达 GPU] ——这意味着我们可以将工作负载匹配到最适合它们的芯片上"。

芯片计划背后是需求的爆发：

- 2026 年 4 月公司披露年收入运行率突破 300 亿美元，较 2025 年底的约 90 亿美元增长超 3 倍；
- 年消费超 100 万美元的企业客户数量在不到两个月内从 500 家翻倍至 1000 家以上。

为支撑增长，Anthropic 已构建多云计算设施：与 AWS 合作 Project Rainier（约 50 万颗 Trainium2 芯片，计划扩展至 100 万颗），与谷歌合作部署多达 100 万颗 TPU，还通过 SpaceX 的 Colossus 集群获取英伟达 GPU 算力。

### 2.4 产业链布局：三星代工与关键挖角

> 制造端，Anthropic 已提前落子。

2026 年 7 月，The Information[3] 报道过，其正与三星电子就潜在代工合作谈判，计划采用三星 2 纳米工艺及先进封装技术。

这有迹可循：5 月 Anthropic 完成 650 亿美元 Series H 融资时，将三星、SK 海力士和美光列为"战略基础设施合作伙伴"，三者中三星是唯一具备逻辑芯片代工能力的企业。不过 Anthropic 方面回应称项目仍处早期阶段，尚未进入详细设计、原型制造或量产阶段。

人才端，6 月 :mdnice-bold-italic[Anthropic 从 OpenAI 挖来其芯片项目的早期核心成员 Clive Chan。Chan 曾在特斯拉工作约 2.5 年，参与 Dojo 超算的自研 ML 训练芯片设计，2024 年 1 月加入 OpenAI 后成为其芯片项目第二位硬件工程师，深度参与与 Broadcom 的合作]。他在 X 平台宣布跳槽时称被 Anthropic 的"人才、价值观和雄心"打动，其 LinkedIn 职位描述为"perplexity per picojoule"（每皮焦耳的困惑度），暗示工作将聚焦推理能效优化。

更早的 4 月，路透社已报道 Anthropic 在探索自研芯片的可能性。

## 三、行业格局与共同逻辑：AI 竞争沉向底层

### 3.1 行业格局：OpenAI 已先行一步

> Anthropic 并非首家自研芯片的 AI 实验室。

2026 年 6 月 24 日，OpenAI 与 Broadcom 联合发布名为"Jalapeño"的定制推理芯片——这是 OpenAI 首款自研 AI 处理器，一款专为 LLM 推理优化的 ASIC，从设计到流片仅用时 9 个月，OpenAI 称其自有 AI 模型参与了设计加速。:mdnice-bold-italic[早期测试显示，该芯片每瓦性能"显著优于当前业界最优水平"，推理成本较英伟达 GPU 降低约 50%。OpenAI 计划与微软等数据中心伙伴在 2026 年底前开始吉瓦级部署，并制定了多代产品路线图]。Broadcom CEO Hock Tan 直言："每个模型开发者最终都会走向自研芯片。"

其他科技巨头早已布局：谷歌 DeepMind 长期依赖自研 TPU，已迭代至第七代 Ironwood；Meta 持续推进 MTIA 系列加速器；亚马逊 AWS 的 Trainium 和 Inferentia 已大规模部署，Project Rainier 集群专为 Anthropic 训练需求建造；微软 Maia AI 芯片于 2026 年 1 月商业化。

### 3.2 从"买芯片"到"造芯片"的必然

> AI 实验室自研芯片的驱动力有三层。

- 成本层面：推理是 AI 公司经常性支出的大头，每降低一美分的单 token 成本，在数十亿次日请求中都会产生巨额复利效应——OpenAI 估计 Jalapeño 的 50% 成本降幅到 2028 年可转化为每年 5 亿至 10 亿美元的节省。
- 性能层面：通用 GPU 并非为 LLM 工作负载量身定制，存在大量数据搬运和内存带宽瓶颈，自研芯片可针对模型架构、内核和服务系统深度优化，实现更高的实际利用率。
- 供应链层面：先进制程产能（尤其是台积电 CoWoS 先进封装）极度紧张，依赖单一供应商存在断供风险，多源策略是分散风险的必要手段。

与 OpenAI 选择 Broadcom + 台积电不同，Anthropic 似乎在走一条差异化路径：制造端倾向三星 2 纳米工艺，可能获得更灵活的产能安排；人才端从 OpenAI 挖角具备全栈经验的工程师，而非从零组建；策略端强调"多芯片"而非"全自研"，保留与 AWS、谷歌、英伟达的深度合作。这种"渐进式自研"既保住了现有供应链的稳定性，又为未来深度定制预留了空间。

### 3.3 两条新闻，一个共同方向

> 把 8 月 5 日的两则官宣放在一起，指向同一个判断：**模型层的竞争已无法单独分出胜负，战场正在向更底层沉降**。

谷歌动的是"人和组织"——Hassabis 退回战略层、Koray 集中产品权力、Dean 团队索性离开大公司去自动化科学本身；Anthropic 动的是"硅"——把算力命脉从供应商手里部分收回。

值得注意的是，2023 年 4 月为应对 ChatGPT 搭建的"Hassabis 掌运营、Dean 掌科研"双核心架构（Google 合并公告[4]），至此彻底拆解；而谷歌的人才流失也有连续性——6 月 Transformer 论文作者 Noam Shazeer 转投 OpenAI、诺奖得主 John Jumper 转投 Anthropic，曾导致 Alphabet 股价单日下跌约 5%，Transformer 论文八位作者更是已全部离开谷歌。

谷歌这次选择了一种新的处理范式：**不切割，而是投资——把人才流失转化成战略布局，类似于当年微软吸纳 Inflection、谷歌投资 Anthropic 的逻辑**。而 Anthropic 在这个时间点官宣造芯，既是对外释放信号（:mdnice-bold-italic[其 965 亿美元估值需要更扎实的技术护城河]），也是对内凝聚共识（48.5 万美元的价码正在硅谷芯片人才市场掀起抢人大战）。

参考资料

[1] 官方博客: *https://blog.google/company-news/inside-google/message-ceo/next-chapter-ai-momentum/*

[2] Wired 的独家专访: *https://www.wired.com/story/jeff-dean-google-discovery-loop-startup/*

[3] The Information: *https://www.theinformation.com/articles/anthropic-is-in-talks-with-samsung-to-make-its-custom-ai-chips*

[4] 合并公告: *https://deepmind.google/blog/announcing-google-deepmind/*

::: related-title
**相关推荐**
:::

::: related-links
- **[Jeff Dean：智能体时代，定义问题比写代码更重要](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898377&idx=1&sn=e9768c03cb4c808ee223363ba78df1c8&scene=21#wechat_redirect)**
- **[Anthropic 秘密洽谈“第四极”AI推理芯片供应链，战略押注英国芯片初创公司 Fractile，2027 年或成 AI 推理成本分水岭](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900761&idx=1&sn=efb950773323936deb3f1e84946b6d9a&scene=21#wechat_redirect)**
- **[微软发布 Maia 200 AI 加速器，重塑 AI 基础设施推理经济性](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896707&idx=1&sn=17bdbfa1b3bf8e8da8878ba35ba6605e&scene=21#wechat_redirect)**

:::
