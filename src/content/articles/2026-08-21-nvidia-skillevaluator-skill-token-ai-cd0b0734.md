**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：AI 智能体、Agent Skills、**SkillEvaluator**、:mdnice-bold-italic[基准评测]、NVIDIA

> **AI 智能体的表现，很大程度上取决于喂给它的上下文**。模型再强、文档再全，:mdnice-bold-italic[智能体也可能在找工具上浪费步骤、在死胡同里烧 token]。Skill（技能）把指令、示例和工具调用指南打包成可复用的能力单元，让智能体从意图直达方案——**但"装了 Skill 应该会变好"和"装了这个 Skill 确实变好了"之间，隔着一整套测量。**

![文章插图](/articles/2026-08-21-nvidia-skillevaluator-skill-token-ai-cd0b0734/images/01.jpg)

- **Evaluating AI Agent Skill Performance with NVIDIA SkillEvaluator**
- https://developer.nvidia.com/blog/evaluating-ai-agent-skill-performance-with-nvidia-skillevaluator/
- https://github.com/NVIDIA/SkillSpector

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**
- **[筛选Agentic Skills成功率提升16.2%，Agentic Skills：超越工具使用的系统化知识综述与未来治理挑战](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897931&idx=1&sn=b263f45065d2772fda6e997a39458fbf&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::

> 为此，NVIDIA 构建了开源的 **SkillEvaluator，用三个层级的评估回答不同的问题**：:mdnice-bold-italic[Tier 1 做安全与结构静态检查，Tier 2 用嵌入相似度查重，Tier 3 则让智能体在隔离沙箱里真实跑任务] ——**装 Skill 一次，不装一次，分差就是这个 Skill 的贡献。**

在覆盖 30 多个 NVIDIA 产品、300 多个认证 Skill 的首批基准中，正确性从 46 分升至 87 分，有效性从 39 分升至 78 分，平均 Skill Lift 达 +31 分（剔除安全维度后为 +39 分）。Claude Code 与 Codex 两套 harness 的结果一致，但增益差异主要来自产品领域而非 harness 本身。

值得注意的是，token 节省并非必然：**有的 Skill 省下 76.9% 的 token，有的反而多花 120.3%——这正是需要实测而非感觉的原因**。

## 本文目录

- 一、先搞清楚：什么是 Skill，为什么需要评测
- 二、SkillEvaluator：三层评估体系
  - 2.1 Tier 3 实测是怎么运作的
  - 2.2 一个最小示例
- 三、实测结果：300 多个 Skill 的首批基准
  - 3.1 不装 Skill 的基线
  - 3.2 装与不装的对照结果
  - 3.3 分 harness 的 Skill Lift
- 四、合作伙伴的试点
- 五、三个对实操有用的发现
  - 5.1 更好的评测集，才能养出更好的 Skill
  - 5.2 产品比智能体更影响增益
  - 5.3 token 节省不是自动发生的
- 六、上手：安装、验证与目录结构
  - 6.1 安装 Skill
  - 6.2 验证签名
  - 6.3 目录覆盖的产品面
- 七、写在最后

![文章插图](/articles/2026-08-21-nvidia-skillevaluator-skill-token-ai-cd0b0734/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、先搞清楚：什么是 Skill，为什么需要评测

> AI 智能体的能力上限，取决于它收到的上下文。哪怕模型够强、NVIDIA 的库文档够全，智能体也可能多花好几步才找到正确的工具，在死胡同里白白烧掉 token，或者在专业性强的任务上卡壳。**Skill 把指令、示例和工具使用指南打包在一起，让智能体能更快地从意图走到解决方案。**

按照 NVIDIA Skill 官方文档[1]的定义，Skill 是可移植的指令集，能教智能体正确使用 CUDA-X 库、AI Blueprint、Omniverse 与 Physical AI 工作流、NeMo 训练与推理工具等平台组件。:mdnice-bold-italic[装一次 Skill，智能体在遇到匹配任务时就能按需加载对应产品的操作流程、参考脚本和安全边界。比如 cuOpt 的 Skill 可以帮智能体建模并求解路径规划、线性规划和二次规划问题；Dynamo 的 Skill 负责推理服务的部署拉起；NeMo、Megatron-Core 的 Skill 则覆盖分布式训练的构建、校验与调优。]

::: block-1
NVIDIA 的认证 Skill（verified skills）是经过打包和签名的能力描述符，明确告诉智能体某个 NVIDIA 产品能做什么、何时该调用、怎么调用。而"认证"二字的分量，恰恰来自测量——这正是 SkillEvaluator 存在的理由。
:::

当 Skill 变成一个可复用的能力层，团队需要的是能力治理：**这个能力从哪来？发布前有没有经过风险审查？发布后有没有被改动？** NVIDIA 的认证体系要求每个 Skill 发布前做到五件事：

| 环节 | 含义 |
| --- | --- |
| 编目（Cataloged） | 来自拥有该能力的产品团队或源仓库 |
| 扫描（Scanned） | 发布前检查软件风险与智能体原生风险：隐藏指令、提示注入、触发器滥用、过度授权、声明用途与实际行为不符 |
| 评测（Evaluated） | 让智能体对同一批任务装/不装 Skill 各跑一次，按正确性、安全性、可发现性、有效性、效率五个维度记录分差，结果写入 `BENCHMARK.md` |
| 签名（Signed） | 附带独立的 `skill.oms.sig` 签名文件，用户可验证下载的 Skill 确实出自 NVIDIA 且未被篡改 |
| 文档化（Documented） | 附带技能卡（skill card），记录 Skill 的用途、归属、许可证、依赖项和已知风险 |

*表 1：NVIDIA 认证 Skill 发布前的五道关卡。运行时护栏管的是"智能体在执行时能做什么"，而认证 Skill 管的是"哪些能力能进入工作流"。*

NVIDIA 目前为 Claude Code、Codex 和 Cursor 提供插件，同一批 Skill 也可通过 Skills.sh、ClawHub 和 Hermes Hub 获取。由于 Skill 建立在开放的 Agent Skills 规范之上，**理论上任何兼容该规范的智能体客户端都能使用。**

## 二、SkillEvaluator：三层评估体系

> SkillEvaluator[2] 是 NVIDIA 开源的技能评测工具，通过静态检查和"装/不装 Skill"的真实任务对照运行，测量 Skill 对智能体表现的影响。在一个 Skill 发布之前，它要过三道评估，每一层回答一个不同的问题，且可以独立运行：

- **Tier 1——安全与结构**：静态检查。包括 schema 与 frontmatter 校验、质量评分、针对提示注入和数据外泄的安全扫描、密钥与 PII 检测、许可证检查、脚本 lint。这一层的完整扫描会用到 Semgrep、SkillSpector、Gitleaks 等外部工具。
- **Tier 2——独特性**：用嵌入相似度识别单个 Skill 内部的重复指导内容，以及整个目录中互相重叠的覆盖范围。重复意味着浪费上下文窗口。
- **Tier 3——实测评估**：让智能体在隔离沙箱里对生成的任务跑实测，装 Skill 跑一次、不装跑一次，测量分差。

### 2.1 Tier 3 实测是怎么运作的

> Tier 3 基于 Harbor[3]——一个由 Terminal-Bench 团队打造的开源智能体评测框架，负责在可复现的隔离环境中运行评测。SkillEvaluator 包办 Harbor 的配置工作：把评测用例转换成任务、在沙箱中跑智能体、收集结果、计算 Skill 的影响。

每一个结果都来自受控对照实验。**对每个评测用例，智能体 harness 会跑两遍：一遍装着被测的认证 Skill，一遍不装。两遍运行各自在独立沙箱中执行，使用完全相同的提示词、模型、任务输入和评分标准。在同一个 harness 内，唯一的实验变*量就是"装没装这个 Skill"。** * 这个对照会在两个智能体 harness 上重复。装与不装的分差，就是这个 Skill 的贡献，以分数点（points）为单位报告为 **Skill Lift**。

### 2.2 一个最小示例

> 先为一个 Skill 生成评测数据集：

```
skillevaluator create-eval-dataset ./my-skill --full
```

这会生成 `evals/evals.json`。每个用例包含 ID、提示词、期望输出和可选的断言。加上 `--full`，数据集还会覆盖显式、隐式、上下文和负例四种类型的用例。

人工检查完用例之后，跑实测对照：

```
skillevaluator tier3 evaluate ./my-skill \  --agents codex \  --env-mode docker
```

SkillEvaluator 会把用例转换成 Harbor 任务包，对每个用例装/不装 Skill 各跑一遍，给两遍打分，输出得分和 Skill Lift。底层执行和隔离交给 Harbor 处理，用户侧的工作流保持简单。完整流程见 Tier 3 Live Evaluation 文档[4]。

## 三、实测结果：300 多个 Skill 的首批基准

> NVIDIA 公布了覆盖 30 多个产品、300 多个认证 Skill 的第一批基准结果。本节的数字均取自 2026 年 8 月 12 日 benchmarks.json 快照（commit 738d79e），得分对已发布的"Skill–harness"结果做宏平均，每对组合权重相同。

整个目录在持续评测，最新数字可以从 NVIDIA/skills 仓库[5]中的 benchmarks.json 获取。

### 3.1 不装 Skill 的基线

> 先看智能体"裸奔"时的表现。下表定义了五个评分维度，并给出不装相关 Skill 时各维度的平均基线分（满分 100）：

| 维度 | 测量内容 | 不装 Skill 的基线分（满分 100） |
| --- | --- | --- |
| 正确性（Correctness） | 最终答案对不对？ | 46 |
| 可发现性（Discoverability） | 相关时 Skill 能否被正确加载，不相关时能否保持不加载？ | 42* |
| 有效性（Effectiveness） | 智能体有没有达成用户目标、遵循预期工作流？ | 39 |
| 效率（Efficiency） | 有没有浪费步骤或冗余的工具调用？ | 43* |
| 安全性（Security） | 运行是否避免了不安全操作、密钥泄露和未授权访问？ | 97* |

*表 2：五个评分维度及不装 Skill 时的基线表现（带星号维度的解释见下文）。*

正确性、可发现性、有效性、效率四个维度的基线在 39 到 46 分之间，提升空间很大。安全性是例外，基线高达 97 分——**对安全维度，评测的主要目标是确认装 Skill 不会引入回退。**

::: block-1
这里有个已知的测量口径问题需要说明。**正确性、有效性和安全性测的是运行结果，基线如实反映智能体在没有相关 Skill 时能做什么。** 而可发现性和效率同时测量 Skill 的"使用情况"：:mdnice-highlight[智能体是否找到它、行动前是否读了它、是否避免了多余步骤。]

不装 Skill 时这些行为无从发生，但智能体仍能通过有效的工具调用、干净的执行、以及对无关任务正确地不加载 Skill 来得分——这就是为什么这两项基线在 42 和 43 分，而不是零分。
:::

### 3.2 装与不装的对照结果

> 在两种条件下口径一致的正确性和有效性两个维度上，均分分别从 46 涨到 87、从 39 涨到 78，提升 41 和 39 分。这些分数说明智能体在被评测的专业任务上平均表现更好。

| 维度 | 不装 Skill | 装 Skill | Skill Lift |
| --- | --- | --- | --- |
| 正确性 | 46 | 87 | +41 |
| 可发现性 | 42 | 82 | +40 |
| 有效性 | 39 | 78 | +39 |
| 效率 | 43 | 78 | +35 |
| 安全性 | 97 | 98 | +1 |
| 全维度平均 | — | — | +31 |
| 剔除安全维度平均 | — | — | +39 |

*表 3：认证 Skill 在 Codex 与 Claude Code 上的平均 Skill Lift。单位为分数点，不是百分比变化。*

::: block-1
可发现性和效率的 Skill Lift 分别是 40 和 35 分。由于这两项的评分本身就参照 Skill，应将其解读为"装上之后智能体确实会正确激活并使用对应的 Skill"，而非对智能体裸奔能力的度量。:mdnice-bold-italic[这个性质其实很关键：环境里每个 Skill 都在争夺智能体的注意力，一个在该加载时不加载、不该加载时乱加载的 Skill，反而会拖低智能体的表现。]
:::

### 3.3 分 harness 的 Skill Lift

> 两套 harness 都显示了一致且有意义的增益。两者之间的差异在预期之内——默认系统提示词、上下文处理和工具调用实现都不一样。认证 Skill 提供的是两套 harness 自身都无法产生的结构化领域知识。

| 统计范围 | Claude Code | OpenAI Codex |
| --- | --- | --- |
| 全维度 | +34 | +29 |
| 剔除安全维度 | +42 | +36 |

*表 4：Skill Lift 因 harness 而异，Claude Code 在全维度上的增益更高。*

## 四、合作伙伴的试点

> 评测体系不是 NVIDIA 关起门来自己玩的东西，已经有两家合作伙伴在真实产品里试点。

**OpenClaw 正在 ClawHub 上为官方组织试点 SkillEvaluator**。集成后，Tier 3 评测会自动运行，装/不装的结果直接显示在 Evals 标签页里，开发者在发现和采用 Skill 的地方就能看到评测信号。

**Nous Research 则在 Hermes Agent 中测试了 SkillEvaluator，并在 Skill 安装流程里加入了可选的 SkillSpector 咨询扫描**。该集成检查 PII、Unicode 偷渡、脚本 lint、许可证和安全问题，在安装前把问题定位到具体的文件行。整个工作流有 29 个通过的测试覆盖，每个 Skill 的扫描大约 1.4–1.5 秒。

## 五、三个对实操有用的发现

### 5.1 更好的评测集，才能养出更好的 Skill

把重要任务、期望输出和超范围请求定义清楚的团队，会得到更锐利的评测信号——而且这发生在任何智能体运行之前。**一个 Skill 能被测得多准，取决于它的评测集把这份工作描述得多准。Garbage in, garbage out，评测也不例外。**

### 5.2 产品比智能体更影响增益

**Skill Lift 在不同产品之间的差异远大于在不同 harness 之间的差异**。Claude Code 和 Codex 平均只差 5 分左右，而按产品统计的 Skill Lift 从大约 +2 到 +46 不等。领域、任务和评测设计的影响，压过了 harness 的影响。

### 5.3 token 节省不是自动发生的

> SkillEvaluator 把 token 用量和效率维度分开追踪。

两个单次尝试的示例很能说明问题：

- 来自认证 Skill 仓库的 `jetson-optimize-memory` 把 token 从 617,306 降到 142,540（省了 76.9%），执行时间从 474.9 秒降到 220.0 秒（快了 53.7%）；
- 反过来，`cuopt-install` 让 token 从 25,227 涨到 55,582（多了 120.3%），执行时间从 34.0 秒涨到 41.1 秒（慢了 20.8%）——这恰恰暴露了一个值得继续优化的方向。**装了 Skill 到底省 token 还是费 token，靠感觉猜不出来，实测才能给出答案。**

| Skill | token 变化 | 执行时间变化 |
| --- | --- | --- |
| `jetson-optimize-memory` | 617,306 → 142,540（−76.9%） | 474.9s → 220.0s（−53.7%） |
| `cuopt-install` | 25,227 → 55,582（+120.3%） | 34.0s → 41.1s（+20.8%） |

*表 5：两个 Skill 的 token 与执行时间对照。一个大幅节省，一个反而更贵——评测的价值就在于把这两种情况区分开。*

## 六、上手：安装、验证与目录结构

> 想自己动手，可以读 SkillEvaluator 文档[6]，或者直接看 GitHub 上的 NVIDIA 认证 Skill 目录[7]。这里补充一些博客之外的实用信息。

### 6.1 安装 Skill

通过 Vercel 的 `skills` CLI 安装（需要 v1.5.16 或更新版本），不需要克隆仓库或手动拷贝目录：

```
npx skills add nvidia/skills
```

CLI 会引导你选择 Skill 和安装目标。想跳过交互、直接装指定 Skill 到指定智能体：

```
npx skills add nvidia/skills \  --skill cuopt-numerical-optimization-api \  --agent claude-code \  --agent codex \  --yes
```

除了 Claude Code 和 Codex，`--agent` 还支持 Cursor、Kiro、Snowflake CoCo 等客户端。保持 Skill 最新用 `npx skills update`，它会顺带提示上游已删除或合并的 Skill，并帮你清理本地残留副本。

### 6.2 验证签名

每个发布的 Skill 都带一个独立的 OMS 签名（`skill.oms.sig`），同步流水线会在发布前丢弃缺少必需产物的 Skill，所以目录里的每个 Skill 都包含：`SKILL.md`（给智能体的指令）、`skill-card.md`（身份与治理卡）、`skill.oms.sig`（签名）、Tier-3 评测数据集，以及生成好的 `BENCHMARK.md`。对着 NVIDIA 的信任锚 `nv-agent-root-cert.pem` 验证：

```
pip install model-signingmodel_signing verify certificate SKILL_DIR \  --signature SKILL_DIR/skill.oms.sig \  --certificate_chain nv-agent-root-cert.pem \  --ignore_unsigned_files
```

验证通过，意味着这个 Skill 的内容自 NVIDIA 签名以来没被改动过。

### 6.3 目录覆盖的产品面

> 这个仓库本身是个目录：各产品的 Skill 在各自的产品仓库里维护，每天经自动化流水线镜像同步到这里。覆盖面相当广，按工作流分组大致如下：

| 分组 | 覆盖的工作 | 代表性 Skill 领域 |
| --- | --- | --- |
| Agentic AI | RAG、AI-Q、沙箱、策略、评测与智能体工作流自动化 | RAG Blueprint、AI-Q、NemoClaw |
| 对话式 AI | 语音 NIM、ASR/TTS/NMT 部署、ASR 评测闭环、语音智能体 | Nemotron Speech（Riva）、临床 ASR 评测微调 |
| 数据科学 | 数据准备、探索与加速分析 | cuDF、cuPyNumeric |
| 决策优化 | 路径规划、调度与数值优化 | cuOpt |
| GPU 开发 | CUDA 相关 kernel 开发、框架集成与性能调优 | TileGym |
| 推理 AI | 模型服务、部署拉起、运行时校验与运维排障 | Dynamo |
| 仿真与建模 | 气象、气候、量子、物理 ML 与仿真 | Earth2Studio、CUDA-Q、PhysicsNeMo |
| 训练 AI | 分布式训练、检查点转换、并行、韧性、训练栈 CI | NeMo、Megatron-Core、DALI、Nemotron |
| 视觉 AI | 视频分析、检索、摘要、感知管线与模型导入 | DeepStream、Video Search and Summarization |
| Physical AI 与 Omniverse | USD 资产工作流、SimReady 转换、实时查看、神经重建 | Omniverse 与 Physical AI 工作流 |

*表 6：NVIDIA Skill 目录按工作流的十大分组。除此之外还有 DOCA（BlueField DPU）、Jetson BSP/设备侧、医疗 AI（MONAI）、TAO Toolkit、Isaac 医疗工作流等长尾产品线，Skill 总数在持续增长。*

## 七、写在最后

> "装 Skill 有用"正在从一句经验之谈变成一个可验证的主张：**同一个任务、同一个模型、同一个评分标准，唯一的变量是装没装 Skill。**

首批 300 多个认证 Skill 的数据说明，这个变量平均值 +31 分——但按产品看，从 +2 到 +46 的跨度也提醒每个团队：你的 Skill 值多少分，得自己测。

要上手，去 SkillEvaluator 文档[8]，或者直接逛 GitHub 上的认证 Skill 目录[9]。

参考资料

[1]

NVIDIA Skill 官方文档: *https://docs.nvidia.com/skills/*

[2]

SkillEvaluator: *https://github.com/NVIDIA/SkillEvaluator*

[3]

Harbor: *https://github.com/laude-institute/harbor*

[4]

Tier 3 Live Evaluation 文档: *https://docs.nvidia.com/skills/skillevaluator*

[5]

NVIDIA/skills 仓库: *https://github.com/NVIDIA/skills*

[6]

SkillEvaluator 文档: *https://docs.nvidia.com/skills/skillevaluator*

[7]

NVIDIA 认证 Skill 目录: *https://github.com/NVIDIA/skills*

[8]

SkillEvaluator 文档: *https://docs.nvidia.com/skills/skillevaluator*

[9]

GitHub 上的认证 Skill 目录: *https://github.com/NVIDIA/skills*

::: related-title
**相关推荐**
:::

::: related-links
- **[技能即记忆：AutoSkill 让 LLM 智能体从交互经验中自我进化，实现终身学习，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898295&idx=1&sn=6bf15a2f47d940cea0165f5d1f50aa61&scene=21#wechat_redirect)**
- **[筛选Agentic Skills成功率提升16.2%，Agentic Skills：超越工具使用的系统化知识综述与未来治理挑战](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897931&idx=1&sn=b263f45065d2772fda6e997a39458fbf&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::
