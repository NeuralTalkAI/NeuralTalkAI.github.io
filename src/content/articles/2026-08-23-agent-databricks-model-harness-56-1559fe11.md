关键词：:mdnice-bold-italic[智能路由]、Omnigent、**AI 编码成本**、Unity AI Gateway、**效率前沿**

> *抱一丝，最近先清一波前阵子偷懒没发的文章。2026 年，AI 编码工具在带来效率跃升的同时，也把成本曲线推向了不可持续的方向。Databricks 分析内部 163 个编码会话后发现，:mdnice-bold-italic[82% 的会话默认跑在最贵的模型上，琐碎任务消耗了最大头的开支]。**关键结论看第四章。** 👀*

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/01.webp)

- **Smart Routing in Unity AI Gateway: Match frontier quality with 30%+ lower cost per task**
- https://www.databricks.com/blog/smart-routing-unity-ai-gateway-match-frontier-quality-30-lower-cost-task

::: related-title
**相关推荐**
:::

::: related-links
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**

:::

::: related-links
- **[LLM负责思考，SLM负责记忆。LightMem小模型天团重塑 Agent 记忆，延迟仅 83 毫秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900501&idx=1&sn=2311904700b9bfd0c8da1f96e2e98a4b&scene=21#wechat_redirect)**
- **[《深入理解 AI Agent》开源！把 Agent 系统设计全景，拆成十个动手实验：从核心公式、Coding Agent 到自我进化与多 Agent 协作](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903768&idx=1&sn=fe2f78bbf4926e215d4c6420c2f90287&scene=21#wechat_redirect)**

:::

> 针对这一问题，Databricks 在 Unity AI Gateway 中上线了 :mdnice-bold-italic[Smart Routing：会话开始时用一个低成本分类模型评估任务复杂度，按任务级别把会话固定路由到能完成它的最便宜模型，保住缓存命中率的同时实现成本优化]。

内部基准上，Smart Router 以 2.13 美元/任务解出 92.3%，比 Claude Opus 5 便宜 35% 且通过率高 1.3 个百分点；公开基准上追平 Opus 5 的表现，成本不到一半。

::: block-1
:mdnice-highlight[路由的执行层是开源元 harness Omnigent，它把 Claude Code、Codex 等外壳变成可互换组件，模型和 harness 同时路由，子智能体还能进一步细分成本]。支撑这一切的是一套建在**数百万行真实代码上的内部基准** ——任务来自真实 PR、封掉 git 历史防作弊、用测试执行而非 LLM 评分。
:::

综合 Stripe、Uber 等公司的实践，:mdnice-bold-italic[Databricks 归纳出四根成本杠杆：迁移低成本模型（省 50%+）、智能路由（省 30%）、渐进式支出控制（省 10%）、上下文优化（省 10%），构成了从评测到网关再到客户端的完整闭环。]

| 杠杆 | 做法 | 节省成本幅度 |
| --- | --- | --- |
| 低成本模型 | 迁移到开源和更高效模型 | ~50%+ |
| 智能路由 | 动态路由模型和 harness | ~30% |
| 支出控制 | 可见性、警告、预算触发线 | ~10% |
| 上下文优化 | 压缩、工具裁剪、缓存调优 | ~10% |

## 本文目录

- **交流加群请在 NeuralTalk 公众号后台回复：加群**一、钱花在哪了：82% 的会话在用最贵的模型干最便宜的活
- 二、Smart Routing：在任务开始前就定好用哪个模型
  - 2.1 为什么选"任务级路由"而不是"请求级路由"
  - 2.2 路由器怎么判断任务难度
  - 2.3 效果：内部省 35%，公开基准省 56%
- 三、Omnigent：让路由决策跨 harness 生效
  - 3.1 Omnigent 本身的三个支柱
- 四、底座：一套建在数百万行真实代码上的基准
- 五、四根成本杠杆：一份跨公司调研的清单
  - 5.1 杠杆一：盯紧"效率前沿"，不是"智能前沿"
  - 5.2 杠杆二：动态路由
  - 5.3 杠杆三：可见性、触发线和渐进式干预
  - 5.4 杠杆四：削 token 开销
- 最后：成本管理是工程问题，不是采购问题

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/02.webp)

## **交流加群请在 NeuralTalk 公众号后台回复：加群**一、钱花在哪了：82% 的会话在用最贵的模型干最便宜的活

> Databricks 就智能路由（Smart Routing）发布文章《Smart Routing in Unity AI Gateway: Match frontier quality with 30%+ lower cost per task》[1]里先放了一张模型发布时间线图。

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/03.webp)

::: image-caption
2026 年前沿模型发布时间线：截至 8 月 13 日共 33 个新模型，平均约每 5 天发布一个。底部曲线为滚动 30 天发布数量，5 月和 8 月出现明显峰值（来源：Databricks）
:::

从 2026 年 1 月到 8 月 13 日，OpenAI、Anthropic、Google DeepMind、SpaceXAI 和开源阵营合计发布了 33 个新模型，前沿模型的平均发布节奏是每 5 天一个。

模型多了，开发者的行为反而变懒了。面对选择过载，**多数人的做法是：把最强的模型、最高的推理档位设成默认**，然后不再管它。

::: block-1
Databricks 分析了自己内部 163 个编码会话的路由前数据，结论相当刺眼：**82% 的会话（134/163）直接跑在 Claude Opus 4.8 上**，完全没有任何模型选择行为。按任务复杂度拆分后，:mdnice-highlight[琐碎任务在最贵模型上花了 553 美元，标准任务花了 470 美元，而真正"用对了模型"的会话只花了 18 美元。]
:::

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/04.webp)

::: image-caption
模型错配分析：163 个会话中 82% 默认使用 claude-opus-4-8，琐碎任务在昂贵模型上的花费为 553 美元，标准任务 470 美元，而模型使用得当的会话仅 18 美元（来源：Databricks）
:::

上面这张图解释了为什么"让每个任务匹配合适的模型"能省 50% 以上——前提是有人（或有系统）真的去匹配。**靠开发者自觉不现实，所以 Databricks 把这件事做成了基础设施。**

## 二、Smart Routing：在任务开始前就定好用哪个模型

> Smart Routing[2] 是 Unity AI Gateway 里新上线的成本控制能力，目前处于 Beta 阶段。它的核心思路一句话就能说完：:mdnice-bold-italic[自动评估每个编码任务的复杂度，把任务路由到能完成它的最便宜模型，整个过程发生在 Claude Code 和 Codex 这些开发者已经在用的工具内部，用户无感知。]

### 2.1 为什么选"任务级路由"而不是"请求级路由"

模型路由有两条技术路线：

1. **请求级路由**：每条请求单独判断复杂度，发给不同模型。问题在于 agentic 编码的成本大头是缓存命中率——连续轮次必须打到同一个模型（目前还得是同一推理档位）才能保住缓存。逐请求路由会把缓存打散，省下的模型差价可能还不够赔缓存失效的钱。
2. **任务级路由**：会话开始时评估一次任务复杂度，选定模型和 harness 后贯穿整个会话。缓存命中率保住了，还留了后续优化空间，比如发生上下文压缩事件后重新评估升级或降级。

Databricks 选了后者。这是:mdnice-bold-italic[对 agentic 工作负载成本结构的一个清醒判断：缓存经济性优先于路由粒度。]

### 2.2 路由器怎么判断任务难度

> 整个路由只用一套策略，分两步：

- **第一步是分类。用一个便宜、低延迟的小模型读任务描述，提取几个语义字段**：改动落在系统哪个部分、prompt 里带了什么代码证据（片段、堆栈、还是什么都没有）、失败表现是什么、修复看起来有多局部、属于什么类型的项目。从这些字段推出任务类型族和语言族。提取器故意做得小而快——如果用前沿模型做分类，等于给每个请求（包括本来要省钱的简单请求）都加税。
- **第二步是定档。默认落在中等规格模型上，根据标签往两个方向移动**：任务需要前沿级能力和知识就升级到贵模型，不需要就降级到便宜模型。一套策略撬动一整套模型池。

::: block-1
值得注意的是，路由决策只用了任务开始时可获得的信息——:mdnice-bold-italic[描述和元数据，没有答案、没有测试用例、没有仓库信息。Databricks 自己也承认这里还有很大空间：一个"全知"的路由器能以更低的成本击败所有单一模型，会话中途重新评估复杂度也是待做的事。]
:::

### 2.3 效果：内部省 35%，公开基准省 56%

> 在 Databricks 自己的内部编码基准（未对外公开，各实验室没有接触过）上，Smart Router 解出 92.3% 的任务，单任务成本 2.13 美元——比 Claude Opus 5 便宜 35%，通过率还高 1.3 个百分点。在公开编码基准套件上，Smart Router 追平 Opus 5 的表现，成本不到一半（节省 56%）。

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/05.webp)

::: image-caption
Smart Router 与单一智能体的对比：左图为 Databricks 内部编码基准，Smart Router（红星）以 2.13 美元/任务解出 92.3%，比 Opus 5 便宜 35% 且通过率高 1.3 个百分点；右图为公开基准套件，67.2% 通过率与 Opus 5 持平，成本 0.54 美元/任务，节省 56%（来源：Databricks）
:::

跑赢单一最佳模型的原因不复杂：大部分收益来自把简单任务交给便宜模型。内部任务分布里，翻转一个 flag、改单个文件、修一个边界清晰的 bug 这类活占大头，它们用不上最贵的模型。

## 三、Omnigent：让路由决策跨 harness 生效

路由决策做出来了，还得有执行层。模型要搭配 harness（智能体外壳）才能干活，而 Databricks 在内部基准测试[3]中发现，同一个模型换一个 harness，成本和质量差异可以超过两倍。只路由模型、不路由 harness，等于只做了一半的优化。

Omnigent[4] 是 Databricks 今年 6 月开源（Apache 2.0）的"元 harness"（meta-harness），由联合创始人、Spark 发明人 Matei Zaharia 主导。它坐在 Claude Code、Codex、Pi 这些单个 harness 之上，把它们变成可互换的组件。Smart Routing 在 Omnigent 里分两层落地：

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/06.webp)

::: image-caption
Smart Routing 与 Omnigent 的配合：左侧为传统模式，开发者手动在 Claude Code 和 Codex 之间切换；右侧为 Omnigent 元 harness 模式，Smart Routing 自动为每个任务选择 harness 和模型组合（来源：Databricks）
:::

第一，开发者直接把 Smart Routing 选为 harness，Omnigent 就同时为每个任务挑 harness 和模型，模型路由由 Unity AI Gateway 的 Smart Routing 驱动。

第二，所有子智能体（sub-agent）的启动也走 Smart Routing API，每个子智能体可以用不同的 harness 和模型。这一点很关键：用户的初始 prompt 往往描述不全、难度难判，而子智能体带着新信息、新缓存和明确指令重新开一局，路由判断更准。于是单个任务内部可以出现精细的分工——大规模代码库摘要交给便宜模型，架构设计留给贵模型——省得更多。

根据官方文档[5]，目前的使用门槛是：账户管理员开启 Smart Routing 预览；路由只在 `system.ai` 前缀的模型服务之间选择；用户必须对全部候选模型有 EXECUTE 权限，否则请求直接报错；跨 harness 路由需要 Omnigent v0.8.0 以上版本。配置方式很薄，Omnigent 下两条命令：

```
omni claude --smart-routing --server <workspace-url>omni codex --smart-routing --server <workspace-url>
```

### 3.1 Omnigent 本身的三个支柱

抛开路由，Omnigent 解决的是另一个普遍痛点：工程师同时开着四五个 agent，互相之间靠复制粘贴传话，每个 harness 只懂自己的会话。它把这类问题归成三根柱子：

- **组合（Composition）**：不改代码就能组合多个模型、harness 和技术路线，Claude Code、Codex、Pi 和自定义 YAML agent 之间一行配置切换。
- **控制（Control）**：有状态的上下文策略跑在 harness 层而不是 prompt 里。比如 agent 从 npm 装了新包之后，下次 git push 必须人工批准；或者每花掉 100 美元就暂停并询问是否继续。prompt 层面的约束在长会话里会被上下文稀释，基础设施层的策略不会。
- **协作（Collaboration）**：会话通过 URL 实时共享，同事可以进来看文件、评论、甚至直接指挥 agent。同一个会话可以同时出现在终端、网页、手机端。

架构上是 runner 加 server：runner 把任意 agent 包进带沙箱的会话，对外暴露统一 API；server 负责策略和共享，把每个会话同时暴露给终端、App 和 Web API。

## 四、底座：一套建在数百万行真实代码上的基准

> 上面所有"哪个模型划算"的结论，都依赖一个前提：你得先知道哪个模型在你的代码库上真的好使。Databricks 为此专门建了内部编码基准《Benchmarking Coding Agents on Databricks' Multi-Million Line Codebase》[6]，方法上有几处值得抄作业。

任务来自工程师真实完成并合并的 PR，覆盖 Python、Go、TypeScript、Scala 等语言，任务和解法都经过人工复核。候选任务由脚本和 AI 生成，但每个样本都人工评估过——有些原始 PR 里的测试锚定了精确字符串匹配，对非确定性输出不公平，就手工重写成评估行为。为了防止 agent 顺着 git 历史"抄答案"，团队还封掉了提交历史。

**评分不用 LLM judge**。agent 声明完成后，checkpoint 代码、打入预留测试、跑测试定通过与否。:mdnice-bold-italic[理由很直接：LLM 评分奖励"听起来对"，测试执行只认"真的对"。]

> 这套基准跑出四个结论：

1. 编码任务的帕累托前沿（同等成本下质量最好）上同时站着 OpenAI、Anthropic 和开源模型——**只有混合使用多家工具才能拿到前沿性能。**
2. 开源模型，尤其 **GLM 5.2，已经能处理最高难度档的任务。**
3. 模型的 token 单价和端到端任务成本是两回事，:mdnice-bold-italic[大模型 token 效率更高，总账可能更便宜。]
4. **harness 对成本和质量影响巨大**，一些简单 harness（比如 Pi）在他们的工作负载上表现最好。

## 五、四根成本杠杆：一份跨公司调研的清单

> Smart Routing 只是 Databricks 成本体系里的一环。

8 月 7 日的成本管理文章《Managing AI Coding Costs at Scale》[7]:mdnice-bold-italic[综合了 Databricks 自己和 Stripe、Coinbase、Uber、Ramp 几家公司的实践，把 AI 编码成本管理归成四根杠杆，并给出了各自的节省幅度（方向性数字，来自对开发团队的非正式调研）：]

| 杠杆 | 做法 | 节省幅度 |
| --- | --- | --- |
| 低成本模型 | 迁移到开源和更高效模型 | ~50%+ |
| 智能路由 | 动态路由模型和 harness | ~30% |
| 支出控制 | 可见性、警告、预算触发线 | ~10% |
| 上下文优化 | 压缩、工具裁剪、缓存调优 | ~10% |

### 5.1 杠杆一：盯紧"效率前沿"，不是"智能前沿"

> 前沿实验室卷的是智能前沿——峰值智能，能解数学新题、挖安全漏洞。:mdnice-bold-italic[但企业规模化部署时，更相关的是效率前沿：在给定智能水平上价格最好的那组模型。日常编码大多不需要证明定理，重要的是"达标质量"的最低成本。] 效率前沿的推进速度比智能前沿快得多，几乎每周都有单位价格智能更优的新模型出来。

**吃到这根杠杆的前提是有内部评测**。公开基准对真实编码表现的指示性很差，所以各家公司都建了自己的自动化评估。评测也经常给出负面结果：

- Stripe 发现 Opus 4.7 相比 4.6 质量没有实质提升但成本上升，直接没让它上线；
- Databricks 在 Opus 5.0 对比 4.8 时也看到类似的成本回退。

::: block-1
他们的**内部基准则确认了 GLM 系列的价格竞争力，据此把 GLM 推给了内部开发者** ——:mdnice-highlight[截至 8 月 6 日，Databricks 给核心软件开发推荐的三个模型是 GLM 5.2、Opus 4.8 和 GPT-5.6 Sol。]
:::

![文章插图](/articles/2026-08-23-agent-databricks-model-harness-56-1559fe11/images/07.webp)

::: image-caption
编码模型效率前沿（2026 年 8 月 6 日实测）：左图横轴为单任务平均成本、纵轴为平均通过率，GLM 5.2 以约 1 美元/任务达到 74% 通过率，位于前沿上；右图为 Databricks 当期推荐模型：GLM 5.2、Opus 4.8、GPT-5.6 Sol（来源：Databricks）
:::

### 5.2 杠杆二：动态路由

> 即前文详述的 Smart Routing，内部实测平均任务成本降 30% 以上，质量大致持平工作集里最贵的模型。

Databrick 还梳理了行业里三类路由形态：请求级路由（Cursor Router、OpenRouter AutoRouter、Ramp 的 Router 都在做）、任务级路由（Omnigent 这类元 harness）、以及升级/委派模式——:mdnice-bold-italic[Claude 的 Advisor Tool 让便宜模型当家、贵模型做顾问，Cognition 的 Devin Fusion 则反过来，贵模型跑主循环、把杂活外包给便宜模型。]

### 5.3 杠杆三：可见性、触发线和渐进式干预

> **这里有个反直觉的共识：所有受访公司都把硬性断供当作最后手段**。原因有两个——任务中途断供会把上下文全毁掉，开发者得冷启动重来；而且花钱最多的人往往也是产出最高的人，一刀切等于掐死效率最高的一批用户。

实际做法是四级递进：

1. 先给开发者实时的花费看板和省钱提示；
2. 再上可自行确认解除的软性支出闸门，更高额度需要管理层审批；
3. 触发闸门后降级到便宜模型而非停权；
4. 彻底暂停访问只作为最后手段，而且是开启对话的起点，不是终点。

### 5.4 杠杆四：削 token 开销

一句"修个 bug"，烧掉的 token 大头在上下文收集、工具调用、代码库搜索和系统信息上，用户真正写的那几句话占比很小。

::: block-1
Databricks 的对应手段：:mdnice-bold-italic[更频繁地做上下文压缩、选 token 效率更高的 harness、审计返回冗长输出的工具、鼓励开发者把任务范围切小、调优 prompt 缓存参数。单是把 harness 和缓存设置调一遍，生成 token 数和成本就降了将近 50%，质量无损失。]
:::

## 最后：成本管理是工程问题，不是采购问题

> 整体来看，Databricks 的整套逻辑是闭环的：:mdnice-bold-italic[先用真实代码库上的内部基准搞清楚每个模型、harness 的真实性价比；再用 AI Gateway 模式（模型代理、预算执行、配置管理、会话追踪日志）把策略落到基础设施层；然后 Smart Routing 按任务复杂度自动分派，Omnigent 把这个能力扩展到 harness 维度；最后用渐进式支出控制和 token 开销优化兜住长尾。]

其中一个判断值得单独记住：82% 的会话默认跑在最贵模型上，而真正花对钱的会话只占总花费的零头。**AI 编码的成本问题，一大半是"默认配置"问题**。:mdnice-bold-italic[把选择从人手里拿走、交给一个只用任务描述就能工作的路由器，省下的 35% 到 56% 基本是白捡的。]

这套东西里的关键组件——Omnigent 和 Unity AI Gateway——已经开源或免费开放。最后，手上有编码 agent 账单的团队，不妨先从建一个自己仓库的小基准开始。

*注：其实也有不少同类的开源的产品如 LiteLLM（网关 + 路由）：https://github.com/BerriAI/litellm，还有 New‑API（多模型网关、多租户）：https://github.com/QuantumNous/new‑api）*

参考资料

[1] 智能路由发布文章: *https://www.databricks.com/blog/smart-routing-unity-ai-gateway-match-frontier-quality-30-lower-cost-task*

[2] Smart Routing: *https://www.databricks.com/blog/smart-routing-unity-ai-gateway-match-frontier-quality-30-lower-cost-task*

[3] 内部基准测试: *https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase*

[4] Omnigent: *https://www.databricks.com/blog/introducing-omnigent-meta-harness-combine-control-and-share-your-agents*

[5] 官方文档: *https://docs.databricks.com/aws/en/ai-gateway/smart-routing*

[6] 内部编码基准: *https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase*

[7] 成本管理文章: *https://www.databricks.com/blog/managing-ai-coding-costs-scale*

::: related-title
**相关推荐**
:::

::: related-links
- **[Harness 更新 ≠ Harness 获益：解耦自进化 LLM 智能体的两种核心能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903450&idx=1&sn=7f6aa340cb4ddb4802e22dce96ed9fc1&scene=21#wechat_redirect)**
- **[Better Harnesses, Smaller LLM！CMU 用自动化 Harness 让小模型 Agent 成本骤降 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904003&idx=1&sn=189a00e193c3f392b054d8927cc3f9b3&scene=21#wechat_redirect)**
- **[Meta‑Harness重新定义LLM系统调优：TerminalBench‑2登顶Haiku 4.5，数学推理提升4.7点，文本分类7.7点涨点且4倍省token](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899387&idx=1&sn=04fefc5d359ac347b7572cbc36193254&scene=21#wechat_redirect)**

:::
