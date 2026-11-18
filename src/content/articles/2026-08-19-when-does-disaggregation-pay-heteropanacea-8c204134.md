关键词：**Agentic 推理**、:mdnice-bold-italic[PDAF 异构拆解]、HeteroPanacea、**推理系统仿真**、定制 NPU

> **智能体推理正在成为大语言模型推理的主流场景**，模型需要主动参与多轮交互并具备工具调用能力。这一转变为底层推理系统带来了更复杂的工作负载：:mdnice-bold-italic[预填充与解码等服务阶段表现出截然不同的行为特征，对计算与内存带宽能力的需求存在显著差异。]

单一同构 GPU 系统已经难以高效支撑智能体推理，这推动了行业向具备拆解服务能力的异构系统转型，例如 NVIDIA 即将推出的 Vera Rubin 平台，采用 GPU 搭配 Groq LPU 的异构架构。**但异构系统中每个组件的最优硬件形态，目前仍缺乏系统的探索与量化结论。**

![文章插图](/articles/2026-08-19-when-does-disaggregation-pay-heteropanacea-8c204134/images/01.webp)

- **When Does Disaggregation Pay? Simulating Prefill–Decode–Attention–FFN Specialization for Agentic LLM Inference**
- 组织机构：帝国理工学院、剑桥大学
- 论文链接：https://arxiv.org/pdf/2608.03741
- 代码地址：论文承诺接收后全量开源，暂未发布

::: related-title
**相关推荐**
:::

::: related-links
- **[异构智能体之间的“读心术”：稠密 KV 缓存高效通信实现 2-3 倍算力优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902336&idx=1&sn=d667567b5407191944b276e13a16ba7d&scene=21#wechat_redirect)**
- **[端侧推理近 3 倍提速！清华北航提出基于闪存-DRAM、CPU-NPU 异构的投机解码系统 LEVER](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902088&idx=1&sn=997b7db5bc318377694f17bbb4b3fe5b&scene=21#wechat_redirect)**
- **[骁龙 8 Gen3 预填充 CPU 性能反超 NPU：移动端大模型异构推理能力重估](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902141&idx=1&sn=8a9d63e50dd13be07200aee6313a34a0&scene=21#wechat_redirect)**

:::

> 针对这一空白，:mdnice-bold-italic[研究团队提出了名为 HeteroPanacea 的拆解服务仿真框架，能够从三个维度实现系统级仿真：分阶段量化、自动化的设备内与设备间并行调度、PDAF NPU 架构异构]。三者结合形成了面向未来异构智能体服务系统的跨栈仿真框架。

![文章插图](/articles/2026-08-19-when-does-disaggregation-pay-heteropanacea-8c204134/images/02.webp)

::: image-caption
图 4：HeteroPanacea 架构总览。TP 为张量并行，PP 为流水线并行，DP 为数据并行，D2D 为设备间互联，N2N 为节点间互联。底层是单设备 PLENA NPU 性能模型，支撑计算能力与内存配置的参数化搜索；中间层是并行调度模型，支持张量、流水线、数据、专家四种并行策略的独立配置，同时通过 D2D 与 N2N 两级互联模型建模不同场景下的通信开销；上层是系统服务层，将预填充与解码服务解耦，分别对应不同的硬件池与调度策略。此外，框架还集成了 MASE 精度评估器，支持分阶段的量化配置搜索，实现精度与性能的联合优化。这种分层设计的优势在于，每个维度都可以独立调整与扩展，既可以评估不同 NPU 架构的收益，也可以验证不同并行策略、量化方案的效果，为异构解耦系统的全维度设计空间探索提供了统一的仿真平台。
:::

研究证实了预填充-解码拆解的收益：在当前 GPU 环境下，**相比传统服务模式，服务吞吐量最高可提升 75%。** 在采用:mdnice-bold-italic[定制 NPU 的假设下，四路预填充-解码-注意力-前馈网络拆解是跨模型提升吞吐量最稳定的方案，最高可带来 2.06 倍的吞吐量增益]。

研究还通过一系列消融实验，揭示了**模型架构与拆解收益之间的内在关联。**

## 本文目录

- 一、Agentic 推理正在重构推理工作负载
- 二、拆解的演进：从两阶段到四阶段的异构范式
- 三、HeteroPanacea：跨栈异构推理仿真框架的设计内核
- 四、定制 NPU 场景：四级拆解的收益边界与临界点
- 五、商用 GPU 场景：现实约束下的务实选择
- 六、模型架构对拆解收益的影响：消融实验发现
- 总结与展望

![文章插图](/articles/2026-08-19-when-does-disaggregation-pay-heteropanacea-8c204134/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[字节 veRL 团队开源大规模通用 Agent 框架 Uni-Agent，用 model-tool-env 三层解耦统一智能体的推理与训练](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901399&idx=1&sn=2d0c9ae166e40644de7841aaa5663280&scene=21#wechat_redirect)**
- **[HeRo: 异构端侧SoC的智能RAG自适应编排框架，实现高达10.94倍加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898756&idx=1&sn=d80b41b1f38300c1d92fcc493716b8d1&scene=21#wechat_redirect)**
- **[240 万芯片实现 99%系统正常运行时间，谷歌提出完全异步分布式训练框架 Decoupled DiLoCo，兼容异构与地理分布](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901453&idx=1&sn=a6cb1e189d6bcf0d5bc42052f88d83ae&scene=21#wechat_redirect)**

:::
