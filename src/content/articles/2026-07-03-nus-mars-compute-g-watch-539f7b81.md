关键词：G-Watch、Intra-kernel Tracing、Agent 驱动优化、异构 AI 硬件、GPU 内核优化、小batch优化

> 在 AI 算力竞速的当下，硬件的理论峰值早已不是性能瓶颈——真正的天花板，藏在 GPU 内核每一个执行阶段的缝隙里。

传统性能分析工具只能给出 Kernel 级的端到端延迟，却无法剖开数十微秒的执行过程：访存占比多少、计算是否饱和、流水线气泡藏在何处、哪一组线程拖慢了整体节奏。**内核优化至今仍高度依赖专家经验与反复试错，效率与扩展性都已逼近极限。**

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/01.webp)

- :mdnice-bold-italic[模型与芯片之间的优化层。MarsCompute 是面向异构人工智能硬件的自适应优化层，可在英伟达、超威及混合加速集群环境下持续优化延迟、运行效率与小批量处理性能。]
- https://mars-compute.com/
- https://g-watch.github.io/

::: related-title
**相关推荐**
:::

::: related-links
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**

:::

> 2026 年 6 月，Mars Compute 正式开源 G-Watch，为 GPU 内核装上了“性能黑匣子”。这套由新加坡国立大学何丙胜教授团队打造的工具链，首次将内核级追踪能力覆盖 CUDA、Triton、TileLang 等多类 DSL，同时以标准化的 Agent 技能接口，让 AI Agent 能够自主完成“画像-假设-优化-验证”的完整闭环。

在 Flash Attention 全系列的生产级验证中，它在 A100 上实现最高 50.8%的性能提升，**即便在趋近优化极限的 Blackwell 架构上仍能稳定挖掘出 3%的额外增益。**

::: block-1
接下来，我们将从 GPU 内核优化的行业痛点切入，全面解析 G-Watch 的核心能力、设计逻辑与技术路径，以及其背后 Mars Compute 面向异构 AI 硬件的完整优化版图。
:::

## 本文目录

- 一、GPU 内核优化为什么难
- 二、G-Watch 是什么
- 三、杀手锏：Intra-kernel Tracing (Iket)
- 四、多 DSL 一等公民
- 五、Agent 驱动的优化闭环
- 六、Roadmap 技术路线
- 总结

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**

:::
