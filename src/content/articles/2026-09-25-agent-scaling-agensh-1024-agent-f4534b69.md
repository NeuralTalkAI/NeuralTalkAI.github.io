关键词：**多智能体系统**、自组织协作、:mdnice-bold-italic[Agent编排]、**规模化智能**

> 多智能体系统可以通过并发执行来降低复杂任务的延迟。本文作者进行了一场极端实验：断网、6 小时、只给一个不能反编译的二进制程序，让 AI 从零复刻 FFmpeg、PHP 这类上万文件、数百万行代码的成熟系统，这是单个 Agent 几乎不可能完成的任务。

::: block-1
单个 Agent 再强，也只有一个上下文窗口、一条动作流、一条记忆流，**面对百万行目标只能串行硬熬**。Codex sub-agent、Claude Code agent teams、Copilot fleet、Kimi Agent Swarm 都用同一个套路：一个主管 Agent 拆任务、派活、收口。论文一针见血：:mdnice-highlight[Worker 可以无限加，主管的管理带宽却有上限，系统规模最终被这一个"项目经理"锁死。]
:::

![文章插图](/articles/2026-09-25-agent-scaling-agensh-1024-agent-f4534b69/images/01.webp)

- 论文标题：**Agensh: Scaling Organizational Intelligence to 1,024 Agents**
- 组织机构：Microsoft Research（微软研究院）
- 论文链接：https://arxiv.org/abs/2609.26781
- 开源代码：https://github.com/microsoft/Agensh（论文给出的代码地址，本文发稿时仓库页面尚未公开；
- 项目主页 https://agens-harness.github.io/project/ 已上线，并提供 1,024 Agent 轨迹回放 Demo）

::: related-title
**相关推荐**
:::

::: related-links
- **[系统视角下的测试时扩展(Test-Time Scaling)：Are We Scaling the Right Thing?](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892664&idx=1&sn=b2a542ccb8252de8f6a2d563b79cbfcd&scene=21#wechat_redirect)**
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**
- **[大模型 Scaling Law 的发展历程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887535&idx=1&sn=b7deccfd6753760b21d6bece2962b676&scene=21#wechat_redirect)**

:::

> 微软提出的 Agensh 方案干脆:mdnice-bold-italic[把主管删掉：让上千个对等 Worker 各自循环执行收集上下文、认领任务、动手实现、验证结果、合并进度五步，靠共享工作区、消息接口、共享上下文三件轻量设施协调，谁也不指挥谁。]

![文章插图](/articles/2026-09-25-agent-scaling-agensh-1024-agent-f4534b69/images/02.webp)

::: image-caption
图 1：在 6 小时预算、无互联网接入条件下从零构建 pandoc，Agent 规模从 1 个扩展到 1,024 个；最终测试通过率从 1 个 Agent 的 33.89% 升至 1,024 个 Agent 的 55.06%。随组织成长，涌现的自组织协作从同伴协调逐步扩展到多 Worker 整合、标准化工作流与专业化角色。多个采样点能看出收益并非均匀释放，而与协作形态的跃迁同步：每当组织跨过一个规模阈值、长出新的协调模式，曲线就获得新的斜率。右侧四级阶梯把这种组织形态学量化绑定到分数上。从体系结构视角看，这相当于集群中的串行段，即最终集成与测试一致性，被逐步拆解后逼近新的 Amdahl 天花板；55.06% 之后斜率再次放缓，提示千人规模仍有未被并行化的关键路径，扩容红利不会无限延续。
:::

实验结果在 ProgramBench 最难五项任务上，Agent 从 1 加到 128，平均通过率从 19.31% 升到 28.78%，相对提升约 49%；pandoc 上一路加到 1,024，部署在 16 个节点、每节点 64 个 Agent，**通过率从 33.89% 推到 55.06%，相对提升约 62%。**

速度差距同样明显：30% 通过率下，单个 Agent 前两小时始终没做到，8 个则需要要 90 分钟、32 个要 60 分钟，而128 个 Agent 仅 30 分钟就越过。:mdnice-bold-italic[更耐人寻味的是，没人分配岗位，评审人、整合工、专项专家却随规模自己长了出来。论文据此断言：Agent 数量，是与模型参数并列的全新 scaling 维度。]

![文章插图](/articles/2026-09-25-agent-scaling-agensh-1024-agent-f4534b69/images/03.webp)

::: image-caption
图 5：6 小时预算下，在五个最难 ProgramBench 任务上从 1 扩展到 128 个 Agent；最终测试通过率均值从 1 个 Agent 的 19.31% 升至 128 个 Agent 的 28.78%。 六张子图共享横轴，即 Agent 数的 log2 刻度，把组织规模当作独立变量做了一次受控 scaling 实验：模型、底座、预算全部锁死，只改并发 Worker 数量。均值子图给出干净的单调上行，净增 9.47 个百分点、相对提升约 49%，但五个分项揭示了均值掩盖的异质性：pandoc、ctags 收益巨大，PHP-src 平缓饱和，FFmpeg 高位微退，gromacs 前三档几乎不动。这说明 Agent 数量 scaling 与任务结构强相关，模块边界清晰、可并行面大的任务最受益，强耦合、需要全局一致性的任务则早早触及 Amdahl 天花板。它也提醒读者，49% 是统计意义上的趋势，而非对每个任务的保证，扩容前必须先判断任务类型。
:::

![文章插图](/articles/2026-09-25-agent-scaling-agensh-1024-agent-f4534b69/images/04.webp)

::: image-caption
图 6：6 小时运行的前 2 小时内，1 到 128 个 Agent 在五个最难任务上的测试通过率随时间变化；更大的组织通常更早达到更高通过率。这组图把纵轴从最终分数换成时间，展示 Agent 数量的第二重收益，即延迟压缩。六条子图中 128 Agent 曲线几乎都在最上方且最早爬升，单 Agent 曲线最早停滞。它揭示了一个与质量收益同样重要的事实：扩容买到的不仅是更高终值，还有到达任意给定分数线的更短时间，这对有硬截止时间的任务具有直接工程价值。曲线间距在不同任务上差异显著，PHP-src、pandoc 的延迟收益最大，gromacs 则四线纠缠，再次印证任务结构决定 scaling 效率。结合图 5 看，Agent 数量同时作用于性能与时延两个维度，才构成完整的组织 scaling 曲线，二者必须分开度量又共同决策。
:::

## 本文目录

- 一、多 Agent 的并发梦，卡在了一个"项目经理"身上
- 二、拆掉中心：把协调下沉为基础设施，请特权 Agent 退场
- 三、五步合作循环：每个 Worker 都是自己的项目经理
- 四、组织基础设施三件套：工作区、消息板与共享上下文
- 五、工程实现：组织长在单 Agent 之上，循环写在提示词里
- 六、上千 Agent 实测：最终分数更高，到达分数的时间更短
- 七、规模之上涌现的"组织形态"：协作模式如何随规模生长

![文章插图](/articles/2026-09-25-agent-scaling-agensh-1024-agent-f4534b69/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[系统视角下的测试时扩展(Test-Time Scaling)：Are We Scaling the Right Thing?](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892664&idx=1&sn=b2a542ccb8252de8f6a2d563b79cbfcd&scene=21#wechat_redirect)**
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**
- **[大模型 Scaling Law 的发展历程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887535&idx=1&sn=b7deccfd6753760b21d6bece2962b676&scene=21#wechat_redirect)**
- **[速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)**
- **[NVIDIA 开源 SoL-Pi 用递归自进化把 Agent Harness 成本压低 50%–54%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905961&idx=1&sn=9ca43c638ab7ca5cc429ae6b076a9173&scene=21#wechat_redirect)**
- **[OpenAI 核心研究员 Noam Brown 深度拆解 Agentic AI、推理与多智能体的技术真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905895&idx=1&sn=6a5d81ee886d6992b05eb8061b077599&scene=21#wechat_redirect)**

:::
