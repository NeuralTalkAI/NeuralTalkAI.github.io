关键词：**GPU 内核优化**、自进化智能体、**黑盒架构反推**、:mdnice-bold-italic[硬件知识库]、**算子性能提升**

性能上，ForgeKernel 在 recurrent KDA prefill 上仅比 CAKE 快 0.4%，基本打平。成本则无法直接换算：CAKE 报告 token 消耗，ForgeKernel 报告 GPU 与 agent 时间。由于起点、模型和计量方式均不同，现有数据不能证明 ForgeKernel 的优化成本更低。

> “**每一个智能体优化循环都会撞上自己制造的天花板**：它在最初几次尝试中就固定了内核的架构，剩下的运行时间都在这个架构之下做局部调优，且每次运行都从零开始，丢弃所有测量得到的经验。”

前沿编码智能体如今已经能写出媲美专家水平的 GPU 内核，哪怕是通用的智能体循环，也能在大量算子上超过手工调优的参考实现。**但这种优化存在天然的能力边界：架构决策过早收敛，后续所有调优都无法突破顶层设计的上限；同时每次运行都是无状态的，实测得到的硬件经验无法沉淀，下一次优化还要重复试错。**

![文章插图](/articles/2026-10-28-cake-kda-forgekernel-agent-h100-ae070871/images/01.webp)

::: image-caption
表 4：每个 ForgeKernel 优化战役的实测成本。架构 Agent 列统计架构 Agent 的重介入次数；活跃小时为 Agent 耗时与 GPU 耗时之和；墙钟时间包含共享集群的排队等待时间。 该表量化了 ForgeKernel 体系的时间与计算成本。整体来看，简单算子如 DSA 分数重计算仅需 3.1 活跃小时，复杂算子如循环 KDA 前向则需要 97.8 活跃小时，GPU 耗时达 34 小时。架构 Agent 的重介入次数与算子复杂度正相关，2D 卷积前向多达 19 次介入，说明其架构探索空间更大。对比传统专家调优通常需要数天甚至数周的人工投入，ForgeKernel 的自动化流程已大幅缩短高性能内核的开发周期。同时，随着知识库的持续沉淀，后续算子的优化成本会逐步降低，因为可复用的成熟优化方案越来越多。
:::

![文章插图](/articles/2026-10-28-cake-kda-forgekernel-agent-h100-ae070871/images/02.webp)

- **FORGEKERNEL: SELF-EVOLVING KERNELS CATCHING AND SURPASSING EXPERTS AND AGENTS**
- 投稿状态：ICLR 2027 双盲评审中
- 论文链接：https://openreview.net/forum?id=EtV3f5BPD4

::: related-title
**相关推荐**
:::

::: related-links
- **[陈天奇团队提出 CAKE：编译器与智能体协同进化，可进化 IR 让 KDA 提速 2.05 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904925&idx=1&sn=ab085dc1e9e6c73917c8853a398775d7&scene=21#wechat_redirect)**
- **[MLSys 三赛道前三、奥赛四项满分背后大杀器：多智能体编码框架 Humanize 的"裁判工程"（Judgement Engineering）](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906854&idx=1&sn=9ef26460d72547889ed619ce42711323&scene=21#wechat_redirect)**
- **[KDA²：Kernel Design Agents 优化 Kimi Delta Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906834&idx=1&sn=53fcb953cfd29bf2cffbb8d2631ba4b5&scene=21#wechat_redirect)**

:::

> ForgeKernel 正是为打破这一天花板而生的自进化内核智能体系统。**它不需要读取参考内核的源代码，仅通过运行、计时和剖析封闭的二进制程序，就能还原其架构设计，并转化为架构优先的优化课程，引导编码智能体逐步落地。** 每当测量发现瓶颈来自架构而非调优时，系统就会重新打开架构设计环节。每一个通过验证的内核都会被封存，成为下一轮优化的参考基准，:mdnice-bold-italic[让优化持续突破已知最优实现；每一个经过硬件探针验证的发现都会进入知识库，在不同算子之间共享复用。]

![文章插图](/articles/2026-10-28-cake-kda-forgekernel-agent-h100-ae070871/images/03.webp)

::: image-caption
图 2：ForgeKernel 系统架构。架构 Agent 将参考内核的硬件可观测行为转化为里程碑序列；编码 Agent 逐行实现决策项，每一步都在同一个 GPU 任务中与参考基准做配对计时。门控结果分为三类：重试当前决策行、重新分析并修订优化课程、交付最终内核。 这张图完整呈现了双 Agent 嵌套循环的工作流。架构 Agent 通过 NCU、时间线探针等工具逆向分析密封二进制参考内核，输出从 M0 架构决策到 Mk 参数调优的分阶段里程碑课程，实现 “架构优先” 的优化路径。编码 Agent 逐里程碑实现内核代码，每一步都经过正确性校验、同任务配对计时和硬件探针验证。当优化陷入平台期时，外层架构 Agent 会重新介入，通过探针定位瓶颈，必要时重启 M0 级别的架构重构，而非在固定架构下无效调参，从机制上避免了传统 Agent 的架构早停问题。
:::

在 Hopper 和 Blackwell 架构的 27 个算子上，ForgeKernel 追平或超越了 22 个专家库内核中的 18 个，**覆盖 FlashAttention-4、cuDNN、DeepGEMM 等主流实现；同时超越三个内核生成系统的全部 7 个已发表内核**，:mdnice-bold-italic[最高性能提升 170.6%，几何均值提升 39.4%。]

![文章插图](/articles/2026-10-28-cake-kda-forgekernel-agent-h100-ae070871/images/04.webp)

::: image-caption
图 3：ForgeKernel 与专家库的性能对比。每个点代表一个算子、输入形状与 GPU 的组合，取每组最优优化战役的结果，按参考内核所属的库分组。增益为优化停止后确认的、相对于对应参考库的加速比；竖线代表性能持平。输入形状详见表 3。 这是论文最核心的主实验结果，覆盖 Hopper H100 与 Blackwell B200 两代架构，包含注意力、GEMM、卷积、DSA 稀疏链四大类共 22 组测试。结果显示 ForgeKernel 在 18 组上追平或超越 FlashAttention-4、cuDNN、DeepGEMM 等业界顶级专家实现，几何平均提升 1.6%，最高在 3D 卷积前向达到 7.6% 的增益。尤其关键的是，所有参考内核均以密封二进制形式提供，ForgeKernel 完全通过硬件行为逆向复现并优化，即使是发布仅 15 天的 DeepSeek-V4.1 新算子也能逼近到 1.3% 的性能差距。
:::

![文章插图](/articles/2026-10-28-cake-kda-forgekernel-agent-h100-ae070871/images/05.webp)

::: image-caption
图 4：ForgeKernel 与当前最优内核生成系统的性能对比（Hopper 平台对比 K-Search，Blackwell 平台对比 Kernel Design Agents 与 CAKE），每组取最优战役结果；坐标轴含义同图 3，但不同 GPU 采用独立刻度。 这组实验直接对比 ForgeKernel 与三类主流 AI 内核生成方案的性能差异，覆盖 7 个公开的 SOTA Agent 生成内核。结果显示 ForgeKernel 在全部 7 个对比项上均实现性能反超，几何平均加速比达 39.4%，最高在 TriMul 算子上相对 K-Search 实现 170.6% 的巨大提升。差距的核心来源是现有系统大多停留在单轮架构下的参数搜索或进化优化，而 ForgeKernel 可以通过架构重分析打破初始架构限制，同时跨算子知识库沉淀的硬件优化经验进一步放大了优势，体现了自进化架构相对于传统单轮生成方案的代际领先。
:::

::: block-1
针对发布仅 15 天的 DeepSeek-V4.1 稀疏 MLA 预填充和融合注意力内核，性能差距仅为 1.3%和 3.7%。:mdnice-bold-italic[仅更换剖析工具，就能在昇腾平台的 8 个 CANN-Bench 四级算子上全部领先最优公开提交，几何均值提升 33.1%。通过将最优内核提升为参考基准，通用智能体循环的性能还能进一步提升最多 9.9%。]
:::

## 本文目录

- 一、内核优化智能体的现实天花板：通用 Agent 为何跑不到硬件极限
- 二、ForgeKernel 整体架构：双智能体+三机制的自进化体系
- 三、参考内核反推：从硬件行为中还原架构设计的黑盒分析法
- 四、嵌套式优化循环：架构可调，让优化跳出局部最优陷阱
- 五、知识库锻造与复用：把硬件经验变成跨算子的可迁移资产
- 六、自进化迭代：世代式自我超越，逼近硬件性能屋顶线
- 七、实验结果与行业思考：自进化范式的价值、边界与未来

![文章插图](/articles/2026-10-28-cake-kda-forgekernel-agent-h100-ae070871/images/06.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[算子 编译器天塌了吗。。NVIDIA 提出 AVO：当 AI Agent 成为进化算子，探索突破 Blackwell 性能极限，超越 FlashAttention‑4 最高10.5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899076&idx=1&sn=64a6532410f9d22f65043945582d4f1a&scene=21#wechat_redirect)**
- **[AKO：Agentic Kernel Optimization，使用 Claude Code 在几小时内优化出超越 FlashInfer 专家实现的 Kernel](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901871&idx=1&sn=8d7495711c82a26ed7d8bf0a79cb6d26&scene=21#wechat_redirect)**
- **[跑掉四五千万人民币等值 Token 之后，两位英伟达工程师重新理解了 Agent：代码不是护城河，时间才是](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906785&idx=1&sn=28a960b4c341e8bf26543827b92694a9&scene=21#wechat_redirect)**
- **[34.93 倍加速！MLSys 26 DSA 智能体赛道冠军方案，一套规则手册与三个子智能体替代人类内核工程师](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901644&idx=1&sn=ddd845b9bc6fd48c0c73a0ce535f45e0&scene=21#wechat_redirect)**
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**

:::
