关键词：**RDMA**、:mdnice-bold-italic[ByteExpress]、参数服务器、**分布式推理训练**、拥塞控制、零拷贝

> “要在生产环境中兑现 RDMA 的收益，远不止替换 TCP 这么简单。”RDMA 凭借硬件传输卸载、内核旁路、直接内存访问等特性，理论上可以大幅降低分布式机器学习系统的通信开销。

但真实生产环境中，可变尺寸的消息、海量并发的连接、突发的扇入流量、常态化的服务故障，都对软件层提出了硬件卸载无法覆盖的需求。:mdnice-bold-italic[字节跳动在数据中心内部署 RoCE RDMA 已有多年，但要将其深度集成到参数服务器架构的推理与训练系统中，同时保障可扩展性与服务可用性，远非简单替换传输协议就能完成。]

![文章插图](/articles/2026-10-26-byteexpress-rdma-74-41cd3aa2/images/01.webp)

- **Deploying RDMA for Distributed Inference and Training at Scale**
- 组织机构：复旦大学、字节跳动、密歇根大学
- 论文链接：https://arxiv.org/abs/2610.12815v2

::: related-title
**相关推荐**
:::

::: related-links
- **[字节跳动 DITRON：三层分块编译击穿分布式通信瓶颈，让 Kernel 性能超越专家手写库，推理端到端提速最高 30%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905469&idx=1&sn=e891c4c9281f2bb4c472f0456becb5ff&scene=21#wechat_redirect)**
- **[字节 veRL 团队开源大规模通用 Agent 框架 Uni-Agent，用 model-tool-env 三层解耦统一智能体的推理与训练](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901399&idx=1&sn=2d0c9ae166e40644de7841aaa5663280&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**

:::

> 本文介绍的 ByteExpress，是字节跳动基于硬件 RDMA 构建的软件通信层，已经在生产环境支撑推荐业务超过两年，覆盖头条、抖音、西瓜视频等核心产品。

论文总结了**三大落地经验，分别对应内存管理、资源扩展与服务可用性三个维度**，通过跨层协同设计，在不改变上层 RPC 接口的前提下，实现了显著的性能提升：

- 生产推理负载下，模型参数拉取平均延迟降低 35.7%，P99 延迟降低 43.7%，总 CPU 使用率降低 10.69%；
- 训练负载下，模型参数推送 P99 延迟最高降低 74.0%，观测周期内未出现 PFC 暂停。

下面先给出本文最核心的两张图：一张呈现 ByteExpress 的方法架构，一张呈现训练场景的性能提升。

![文章插图](/articles/2026-10-26-byteexpress-rdma-74-41cd3aa2/images/02.webp)

::: image-caption
图 2：通信协议栈：Archon 提供 RPC 能力；ByteExpress 管理缓冲区与基于硬件 RDMA 的消息传输。该栈清晰呈现了系统的分层职责划分，是理解 ByteExpress 设计理念的核心。最上层 Archon 是字节自研的 RPC 框架，保留了业务熟悉的 RPC 接口与超时、重试、副本选择等高可用能力，避免业务改造；中间层 ByteExpress 是核心适配层，通过类 socket 的端点抽象对接 Archon，向下将消息转换为 RDMA verbs 操作，负责缓冲管理、QP 多路复用、流量调度等核心逻辑；最底层是 RNIC 硬件，提供传输卸载能力。这种分层设计实现了业务逻辑与 RDMA 硬件的解耦，既保留了 RPC 层的服务治理能力，又充分释放了 RDMA 的硬件性能，同时也让 TCP 降级成为可能，是保障生产部署可用性的关键架构设计。
:::

![文章插图](/articles/2026-10-26-byteexpress-rdma-74-41cd3aa2/images/03.webp)

::: image-caption
表 7：负载最高的训练节点的各项指标。该表对比了 TCP、纯硬件 RDMA、硬件 RDMA + 软件窗口三种方案在训练场景下的核心指标，核心结论是纯硬件 RDMA 反而会劣化尾延迟，只有配合软件发送窗口才能实现显著性能提升。无软件窗口时，RDMA 的 P99 拉取、推送延迟均高于 TCP，还伴随频繁的 PFC 暂停与 NAK 重传，根源在于训练 incast 流量超出了 DCQCN 的响应速度，导致网络拥塞与链路反压。加入软件窗口后，P99 推送延迟相对 TCP 下降 74%，同时彻底消除了观测周期内的 PFC 与丢包重传。这一结果是论文最关键的生产验证结论，证明 RDMA 在大规模训练场景的落地，必须补充应用层的流量管控机制。
:::

## 本文目录

- 一、为什么 RDMA 落地不是“换个协议”这么简单
- 二、ByteExpress 整体架构：RPC 与硬件 RDMA 之间的中间层
- 三、高性能通信引擎：零拷贝的核心是内存所有权跨层协同
- 四、规模化扩展：连接、接收容量与突发流量的分层管控
- 五、服务可用性：传输可靠不等于服务不中断
- 六、生产环境实测：推理与训练的双重性能收益
- 七、落地经验总结与行业启示

![文章插图](/articles/2026-10-26-byteexpress-rdma-74-41cd3aa2/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[Meta 把推荐模型的真实复杂度推进到 Blackwell 时代：Ads Model Kernel Library 重写注意力与融合算子](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903625&idx=1&sn=802bb1e749f5f8b53f1c9678dd401fb9&scene=21#wechat_redirect)**
- **[最高 4.7 倍加速！UCSD、Meta 提出 AutoOverlap：以块级细粒度重叠融合计算与通信，突破多 GPU 通信瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897745&idx=1&sn=c925fdf089e3022438cce891ccbc440b&scene=21#wechat_redirect)**
- **[Meta AI 基础设施的十年演进、下一代基础设施蓝图与 AI 自研芯片之路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896790&idx=1&sn=a13b82d8c267fcdbd6f3e20c53a1429b&scene=21#wechat_redirect)**

:::
