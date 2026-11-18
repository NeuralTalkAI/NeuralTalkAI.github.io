关键词：**混合专家模型（MoE）**、专家并行训练、:mdnice-bold-italic[动态负载均衡]、最优传输、拓扑感知副本放置

> MoE 已经成为前沿大模型的默认架构，**而新一代模型的方向更明确：专家越来越多、激活越来越稀疏**。DeepSeek-V4-Pro 每层路由专家从 V3 的 256 个增加到 384 个，每 token 激活的专家数反而从 8 个降到 6 个。

**这种结构上的收益，代价是把复杂度转移到了训练系统**。:mdnice-bold-italic[每张卡要处理多少 token 完全取决于路由结果，而专家越多越细，路由的长尾就越难预测]。一旦某几个专家成为热点，token 就会集中到承载它们的 rank 上，拉长整步耗时，其他 rank 即使提前算完也只能等待。

所以 MoE 训练慢，症结往往不在"卡不够快"或者"网络不够宽"，而在于负载能否被及时调平。

::: block-1
业界很早就想到:mdnice-highlight[一个直接的办法：把热点专家的权重临时复制一份到某个空闲 rank 上，让空闲卡帮忙分担一部分 token 计算]。FasterMoE、Megatron 里的 Echo、DeepSeek 的 LPLB，还有最新 ICML 2026 中的工作 LLEP，都是这个思路的代表。

这个思路在单机范围内很自然，机内卡间走 NVLink，把权重挪到哪张空闲卡上，代价差别不大。**但新一代模型的规模已经让 EP 域必须跨机：几百个专家的权重要靠专家并行摊开，NVIDIA 给 V4-Pro 的参考训练配置就需要 EP64、跨 64 节点共 512 张 H100**。一旦 EP 域跨出单机，"复制到哪张空闲卡"就不再等价，机内是 NVLink，跨机要走 InfiniBand，搬运一份热点专家权重的代价可以差出一个量级。
:::

问题在于：**这些方案大多只盯着一件事——"把负载摊得越平越好"，默认空闲容量是同质资源，一张空闲卡和另一张空闲卡没什么区别**。结果就是，专家权重有没有被搬到别的节点去，成了通信开销的主导因素——两个负载均衡质量几乎一样的方案，实际开销可能天差地别。:mdnice-bold-italic[换句话说，副本放置不能只看"哪里空"，还要看"搬过去的代价有多高"。]

![文章插图](/articles/2026-08-18-loongforge-taot-moe-1-43-a5a0afd5/images/01.webp)

- **TAOT: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in MoE Training**
- 技术报告地址：https://arxiv.org/abs/2608.03676
- 项目地址：https://github.com/baidu-baige/LoongForge

::: related-title
**相关推荐**
:::

::: related-links
- **[MoE吞吐达94%理想上限！北大联合小红书提出UltraEP：破解大规模专家并行的负载失衡难题](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903917&idx=1&sn=ec7e38d0846e941c1a01280fb70257ed&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**
- **[MoonEP 核心解析：MoE 负载均衡为何要同时调度 Token 与 Expert](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904416&idx=1&sn=0475814cae5e39760b048aa25fd1320d&scene=21#wechat_redirect)**

:::

> 最近，百度百舸 LoongForge 团队在 arxiv 上公开了一篇论文，名字叫《TAOT: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in MoE Training》（arXiv:2608.03676）。:mdnice-bold-italic[该论文提出了 TAOT（Topology-Aware Optimal Transport），它首次把"削峰收益"和"跨节点搬权重的代价"同时纳入优化目标，用最优传输（Optimal Transport）来决定每个热点专家的副本应该落在哪张卡上]：在优化中直接表达优先机内、必要时跨机的权衡，更适合 micro-batch 级动态热点的多节点 EP 场景。

## 本文目录

::: related-links
- 为什么只看负载均衡还不够
- 三阶段规划：从"往哪流"到"谁发多少"
- 在线规划不能成为新瓶颈：把几百次 kernel 融成几次
- 光规划完还不够，通信开销还得藏起来
- 实测结果：训练更快，通信更省
- 补上 MoE 均衡中被低估的一环
- 从论文方法，到框架里的可用能力

:::

::: related-title
**相关推荐**
:::

::: related-links
- **[加速GR00T N1.6具身VLA模型训练！LoongForge 开源框架实现三层全链路优化，吞吐提升 2.3 倍、训练周期缩短 56.6%！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902397&idx=1&sn=6d29e8546b82848b68611e96573e30cf&scene=21#wechat_redirect)**
- **[适配 Ada 中端卡的 MegaKernel 在线广告方案 Ada-MK：共享内存直接减半，端到延迟最高降 50%，击穿小批量毫秒级推理瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902893&idx=1&sn=5f53dd32ac726cea553f1c3743d8d2a1&scene=21#wechat_redirect)**
- **[Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel，支撑万卡模型训练！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904378&idx=1&sn=83e25209c3e1ffc683b97a6f8646f250&scene=21#wechat_redirect)**

:::
