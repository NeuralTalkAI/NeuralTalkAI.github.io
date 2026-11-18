关键词：**异构系统设计**、:mdnice-bold-italic[设计空间探索]、LLM 智能体、**性能建模**、AI+HPC 融合

ArchAgent 这个名字用的人还挺多，不过还是几个月前的这篇更有趣一些

ArchAgent: 智能体驱动的计算机体系结构自动发现，实现 5.3%多核性能突破与 3-5 倍加速

，而且不久前出了 v2 。

> 现代 AI 与 HPC 系统的硬件栈正在沿着计算、内存、互联三个维度快速碎片化，**没有任何单一架构能够同时覆盖大模型训练、推理服务与 AI 耦合 HPC 负载的最优性能**，而由此产生的组合式设计空间早已超出人工探索的能力边界。

![文章插图](/articles/2026-11-03-archagent-scale-up-scale-out-69be0277/images/01.webp)

- **ArchAgent: Agentic AI for Heterogeneous System Design Across Chip, Scale-Up, and Scale-Out Domains**
- 组织机构：美国阿贡国家实验室、伊利诺伊大学厄巴纳-香槟分校
- https://arxiv.org/pdf/2610.23724

::: related-title
**相关推荐**
:::

::: related-links
- **[ArchAgent: 智能体驱动的计算机体系结构自动发现，实现 5.3%多核性能突破与 3-5 倍加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898414&idx=1&sn=d58a7e936f83b566db9a0f5d91e6266c&scene=21#wechat_redirect)**
- **[端侧推理近 3 倍提速！清华北航提出基于闪存-DRAM、CPU-NPU 异构的投机解码系统 LEVER](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902088&idx=1&sn=997b7db5bc318377694f17bbb4b3fe5b&scene=21#wechat_redirect)**
- **[现代异构边缘 GPU 上的 AI 模型调度技术系统综述](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894254&idx=1&sn=6f1a0637b07902f14f027232292c3225&scene=21#wechat_redirect)**
- **[超越英伟达 B200 19%计算密度：7nm 四 Chiplet RISC-V 开源架构——面向 AI/HPC 的代际进化路线图](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894155&idx=1&sn=a148162fc540c7a1a4677e210c439dc4&scene=21#wechat_redirect)**

:::

> 阿贡国家实验室与伊利诺伊大学厄巴纳-香槟分校联合提出的 ArchAgent 框架，给出了一条全新的解题路径：**通过分层多智能体架构将系统设计拆解为芯片、单机扩展、集群扩展三个层级的子问题**，由具备领域知识的 LLM 智能体发挥架构先验优势导航设计空间，同时**以确定性分析评估器形成闭环，从根源上避免 LLM 生成性能数据的幻觉风险。**

![文章插图](/articles/2026-11-03-archagent-scale-up-scale-out-69be0277/images/02.webp)

::: image-caption
图 1：ArchAgent 框架概览。顶层规划器负责全局设计的生成与迭代，三个专项子智能体分别从芯片架构、单域互连、跨域网络三个维度注入领域先验，提前筛除不符合架构常识的无效设计。底层的确定性评估器是整个框架的可信底座，所有性能指标均来自解析成本模型，完全不由大模型生成，从结构上杜绝了性能幻觉问题。同时评估器返回的瓶颈归因结果会作为反馈输入下一轮规划，形成 “设计 - 评审 - 评估 - 优化” 的完整闭环。这种分工模式既释放了大模型的创造性推理能力，又通过专业化约束与定量验证保证了结果的专业性与可审计性。
:::

在 1MW 功率约束的混合 AI+HPC 负载下，ArchAgent 仅用 18 次评估便超越爬山法 100 次迭代的最优结果，最终找到的异构设计相比同构 H100 GPU 基准实现了 1.93 倍的加权几何平均性能提升。

![文章插图](/articles/2026-11-03-archagent-scale-up-scale-out-69be0277/images/03.webp)

::: image-caption
图 2：等功耗约束下，智能体发现的最优设计相对于同构基准系统的各工作负载加速比（综合加权得分 1.93）。该柱状图直观呈现了异构设计在不同负载上的收益分布，揭示了混合负载下异构架构的价值逻辑。AI 训练与服务负载分别获得 1.72 倍与 1.67 倍加速，核心收益来自 B200 相对于基准 H100 的代际性能提升，以及更合理的并行度配置；HPCG 共轭梯度负载获得 4.76 倍的最高加速，说明异构设计针对性优化了延迟敏感的全局规约操作，补全了同构 GPU 系统的短板；三维 FFT 负载仅获得 1.37 倍加速，是整个系统的性能瓶颈，根源在于 FFT 受限于全局二分带宽，现有互连架构难以实现大幅突破。整体所有负载均实现正向加速，验证了智能体在多目标约束下的均衡设计能力，没有出现明显的性能短板。
:::

![文章插图](/articles/2026-11-03-archagent-scale-up-scale-out-69be0277/images/04.webp)

::: image-caption
图 3：最优得分随评估次数的变化曲线（100 次评估预算）：ArchAgent（gpt-oss-120b）对比爬山法与随机搜索。该收敛曲线量化了 ArchAgent 的搜索效率优势，直观体现了领域先验对设计空间探索的增益。ArchAgent 仅用 18 次评估就超过了爬山法 100 次评估的最终得分 1.65，约 36 次评估便收敛到 1.93 的最优得分；爬山法依赖局部变异迭代，收敛速度慢且最终性能上限更低；随机搜索的收敛效率最低，48 次评估才突破 1.5 分。效率差异的核心原因是，ArchAgent 具备体系结构先验知识，早期设计就已经引入多分区异构、工作负载角色分配等结构性优化，而传统黑盒搜索需要从零探索这些架构模式，大量评估消耗在无效的局部参数调整上。该结果证明大模型代理可以显著降低复杂异构系统的设计探索成本。
:::

::: block-1
这项工作的核心价值不仅是找到了更优的系统配置，更在于:mdnice-highlight[证明了 Agentic AI 能够在高度复杂的系统级设计任务中，结合领域知识进行结构化推理，其搜索效率与设计质量均显著优于传统黑盒优化方法]，为下一代异构计算系统的设计自动化提供了可复用的范式。
:::

## 本文目录

- 一、硬件碎片化时代：异构系统设计的三重困境
- 二、ArchAgent 整体框架：分层智能体与确定性评估的双轮驱动
- 三、智能体层级设计：规划者与领域专家的分工协同
- 四、全栈分析评估模型：从芯片屋顶线到集群通信的精准建模
- 五、实验深度解析：异构设计为什么能带来近两倍性能提升
- 六、瓶颈洞察：下一代 AI+HPC 系统的三大演进方向
- 七、局限与未来：从分析模型到仿真闭环的进化路径

![文章插图](/articles/2026-11-03-archagent-scale-up-scale-out-69be0277/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[240 万芯片实现 99%系统正常运行时间，谷歌提出完全异步分布式训练框架 Decoupled DiLoCo，兼容异构与地理分布](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901453&idx=1&sn=a6cb1e189d6bcf0d5bc42052f88d83ae&scene=21#wechat_redirect)**
- **[面积减少15.31% 性能提升11.28%！AutoPPA：让 LLM 从海量RTL代码归纳最优规则，实现PPA最优设计全面碾压人类专家与SOTA](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900218&idx=1&sn=57868d55b82aa27c2f7d54e278883a06&scene=21#wechat_redirect)**
- **[骁龙 8 Gen3 预填充 CPU 性能反超 NPU：移动端大模型异构推理能力重估](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902141&idx=1&sn=8a9d63e50dd13be07200aee6313a34a0&scene=21#wechat_redirect)**
- **[整套软硬件协同系统设计！攻克内存墙(带宽和容量)：极长上下文推理的优化路径！实现 A100 2.24 倍吞吐量的推理加速器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892472&idx=1&sn=a65b145aced249a6066b19964f402ccb&scene=21#wechat_redirect)**

:::

::: related-links
- **[芯片软硬件全自动化设计！计算所处理器国重实验室提出 QiMeng 方案：从特定领域处理器芯片大模型，到编译器工具链、高性能库！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888640&idx=1&sn=22f8a7efccb43f1b0bdbf4186529b167&scene=21#wechat_redirect)**

:::
