关键词：**递归自我改进（RSI）**、:mdnice-bold-italic[Agent Harness]、Token 效率、**自动研究循环**、编码智能体

> 对于研究单一问题的专业研究人员而言，SoL-Pi 相比原生 Codex 和 Claude Code 部署工具每小时可节省 8.75 至 13.50 美元，相比 Pi 则每小时可节省 4.36 至 5.71 美元。

![文章插图](/articles/2026-09-17-nvidia-sol-pi-agent-harness-497d7341/images/01.webp)

- **SoL-Pi: Scaling Auto-Research Loops for Efficient Agent Harnesses**
- https://nvlabs.github.io/SoL-Pi/
- 代码：https://github.com/NVlabs/SoL-Pi

::: related-title
**相关推荐**
:::

::: related-links
- **[从盲搜到自进化，MicroEvo 让 LLM 在十万亿的 CPU 微架构设计空间探索中积累经验、逃离局部最优](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905180&idx=1&sn=a09f367a0cf70c82cd47805e347de448&scene=21#wechat_redirect)**
- **[下一代自进化 Agent 的真正瓶颈不是算法，是系统底座！蚂蚁联合港科大、清华揭开企业级智能体 RL 架构三大支柱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903671&idx=1&sn=ec3dbd910ce685748846b9053ca0d134&scene=21#wechat_redirect)**
- **[从 EDA 软件到硬件 RTL：NVIDIA 仓库级代码自进化智能体框架 HORIZON，重构大模型芯片设计演化范式](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903526&idx=1&sn=52728aef42e16a67281889d8c21d54fe&scene=21#wechat_redirect)**

:::

> NVIDIA Labs 提出的 **SoL-Pi，是一个以 token 效率为目标、由递归自我改进（RSI）自动研究循环搜索出来的编码智能体 Harness**。系统先从公开数据与公开软件环境构建 535 个可执行训练环境，再把 152 条改进提案分发为彼此独立的机制谱系：:mdnice-bold-italic[每条提案依次走过轨迹 rollout、Map–Reduce 分析、提案、Ralph 循环实现、独立审查者审查、轨迹内验证与完全隔离的留出验证；留出评估一旦失败即淘汰候选，且不回流为修复信号。]

编排方式历经编译式工作流、代码编排、一次性技能循环三代演化，最终只有动作融合、在线上下文压缩、ObservationPack 与证据保真归约器四个机制幸存。

- 在可持续运行两到十二小时的长程基准 EdgeBench 上，SoL-Pi 保留约 94% 的 Pi 平均得分，**token 用量减少 45%–49%，成本较 Pi 降低约三分之一、较模型原生 Harness 降低 50%–54%；**
- 在 Anthropic 内核优化实测中，二十个 worker 的集群比原版 Pi 集群周期数少 17.5%、成本低 26.8%，而单纯堆智能体数量反而更贵更慢。

::: block-1
作者主张：:mdnice-bold-italic[token 效率目标天然抗基准过拟合，广度优先是跳出局部盆地的关键，人类仍负责设定先验、把关预算与重构代码，而效率本身可以递归——更省的 Harness 会让下一代 Harness 的搜索更便宜。]
:::

## 本文目录

- 一、问题根源：长周期 Agent 的 Token 浪费困境
- 二、核心方法：可扩展的自动研究循环流水线
- 三、演进之路：三代工作流架构的迭代与取舍
- 四、四大存活机制：从 152 个提案中筛选出的效率密码
- 五、实验验证：EdgeBench 上的能效实测结果
- 六、自动研究的边界与未来方向

::: related-title
**相关推荐**
:::

::: related-links
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[Claude 实现、Codex 审查、人类决策领航：Humanize 项目用 RLCR 编排长程闭环迭代任务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901161&idx=1&sn=88251ef01180c0f214ca6517d40beda1&scene=21#wechat_redirect)**
- **[Harness Intelligence 可训练可迁移](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905040&idx=1&sn=5717e702fb124dbc86415ab96fd2baec&scene=21#wechat_redirect)**
- **[JIT-Agent：重新定义 Agent 能力边界，Harness 智能成为第三增长曲线](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905032&idx=1&sn=c6ec05be0a2f7040510364185f40ae43&scene=21#wechat_redirect)**

:::
