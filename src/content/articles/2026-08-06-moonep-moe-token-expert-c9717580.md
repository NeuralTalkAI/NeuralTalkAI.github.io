关键词：MoonEP、:mdnice-bold-italic[专家并行负载均衡]、**MoE 动态调度**、权重预取、**Token 路由**

> **说明** ：**本文只介绍MoonEP[1] 的基本思路和接口抽象，不展开 kernel 实现**。对于代码细节，我倾向于使用 AI 辅助阅读源码来理解。:mdnice-bold-italic[本文主要讨论三个问题：为什么 MoonEP 要同时调度 token 和 expert、这样做要付出什么成本，以及它适合哪些场景。]

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/01.webp)

- **从 MoonEP 出发：MoE 负载均衡，搬 Token 还是搬 Expert？**
- https://zhuanlan.zhihu.com/p/2068137220662473580
- 本文作者 Leo z，https://www.zhihu.com/people/yin-yin-yue-yue-yy-97

::: related-title
**相关推荐**
:::

::: related-links
- **[MoE吞吐达94%理想上限！北大联合小红书提出UltraEP：破解大规模专家并行的负载失衡难题](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903917&idx=1&sn=ec7e38d0846e941c1a01280fb70257ed&scene=21#wechat_redirect)**
- **[NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903850&idx=1&sn=e7418452bcd7df13a7917f2463127104&scene=21#wechat_redirect)**
- **[把 MoE 权重做成“分页内存”！UIUC 提出 PagedWeight，72%显存节省破解 KV 缓存与权重的显存死局](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903812&idx=1&sn=8884a16353e79af577a4e1bbeba1c872&scene=21#wechat_redirect)**

:::

> 本文以“MoE 负载均衡应调度 Token 还是 Expert”为核心问题，**针对传统专家并行（EP）中极端路由偏斜导致的计算与通信瓶颈展开分析**。传统方案如 EPLB+LPLB 采用阶段性调整 Expert 副本拓扑、运行时动态分配 Token 的模式，无法应对单批次突发热点。

作者详细阐释 **MoonEP 的核心设计**：在每次层调度前生成统一通信计划，通过全局汇总与本地落位的两级规划，同时决定 Token 分发路径与临时 Expert 权重预取方案，确保各 Rank 处理等量 Token 任务，维持静态通信缓冲区与稳定的计算负载。文中建立成本模型量化权重预取与通信收益的平衡关系，得出单份 Expert 权重约对应 1536 个 Token 的通信节省平衡点。

**后文也进一步对比三种不同时间粒度的调度方案**，结合 SIGCOMM 2026 相关研究分析各自适配场景，:mdnice-bold-italic[并以 Kimi 系列模型为例，说明 Expert 数量增多、单 Expert 体积减小的演进趋势将释放动态调度优化空间，指出 MoonEP 更适配大批次训练与高吞吐 Prefill 场景。]

## 本文目录

- 一、一个极端例子：所有 Token 都路由到 Expert 0
- 二、MoonEP 之前：阶段性调整 Expert，运行时调度 Token
- 三、MoonEP 的核心：Plan 同时决定 Token 去向和权重预取
- 四、成本分析：一份 Expert 权重，对应多少 Token？
- 五、场景决定设计：不同时间尺度的取舍
- 六、最后的补充：从 Kimi K2.6 到 Kimi K3
- 小结

![文章插图](/articles/2026-08-06-moonep-moe-token-expert-c9717580/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[面向大规模专家并行 MoE 推理的故障容错方案 EEP，让 671B MoE 单节点故障停机从 348 秒压缩至 19 秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901916&idx=1&sn=bb370fd7a527a3da6a08327163854740&scene=21#wechat_redirect)**
- **[重新思考 FP4 预训练！收缩偏置才是 E2M1 FP4 训练不稳定的根源，UFP4 均匀格式将 124B MoE 损失降解降低 20%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902714&idx=1&sn=f461543fe701a86413ad7da6ccd7156a&scene=21#wechat_redirect)**
- **[前Google TPU架构师访谈：从批大小经济学到MoE物理布局，从流水线并行陷阱到RL对训练策略颠覆，从API定价技术到神经网络与密码学趋同进化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901219&idx=1&sn=95ebfde4d930adbfd424999bc41188e6&scene=21#wechat_redirect)**

:::
