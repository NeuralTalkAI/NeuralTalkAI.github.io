关键词：**多智能体协作**、:mdnice-bold-italic[RLCR 迭代审查（Ralph-Loop with Codex Review）]、AI 编程治理、Claude 插件、**代码质量闭环**

> 2025 年，AI 编程助手已不再新鲜。今天早上看到 OpenAI Codex 登录 ChatGPT 手机 APP，如今 Claude 能写代码，Codex 能做审查，Gemini 能查资料——可当你把一整个功能模块交给 AI 去实现时，谁来确保它不会跑偏？

谁来确认它写的代码质量达标？更要命的是，当人类开发者只需要点一个"确认"按钮就能让 AI 开始干活，**人类自己是否真正理解了即将被执行的计划**？

![文章插图](/articles/2026-05-15-claude-codex-humanize-rlcr-298625c6/images/01.webp)

- **PolyArch/humanize：From Automated Idea Factory to Realization**
- :mdnice-bold-italic[humanize 作为一款 Claude Code 插件，可支持具备独立AI评审能力的迭代式开发。依托持续反馈闭环，让开发更有把握、更可靠。]
- https://github.com/PolyArch/humanize
- 4000 字，阅读 18 分钟，**播客 22 分钟**

相关推荐

- [TokenSpeed 把 LLM 推理拆成可控流水线，为智能体编码时代重写 LLM 推理引擎的控制平面](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900928&idx=1&sn=a7a2425adc6733e807494b3ce7dbcc82&scene=21#wechat_redirect)
- [大模型修硬件Bug，北大提出HWE-Bench，揭示智能体芯片设计的真实能力与致命短板](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900686&idx=1&sn=6716a831e98f3d1d5c9ce05cedb06138&scene=21#wechat_redirect)
- [暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)

> Humanize 项目正是为了回答这些问题而生。

![文章插图](/articles/2026-05-15-claude-codex-humanize-rlcr-298625c6/images/02.webp)

::: image-caption
这张流程图展示了一套由多AI协作驱动的迭代式开发工作流。流程以“plan.md”中的项目计划为起点，由Claude作为执行方，基于T1、T2、T3三类任务推进工作。流程中设置了Codex的双重校验环节：首先Codex会询问“是否完成”，若工作未完成，则通过橙色反馈流让Claude继续推进；若已完成，则进入质量校验环节，Codex检查工作质量，发现Bug时通过紫色反馈流要求Claude修正，直至无问题后，流程走向“Done”。整个流程形成了开发-校验-迭代的闭环，通过AI间的协作与反馈，保障任务完整且高质量交付。
:::

::: block-1
它不是又一个代码生成工具，而是一套完整的 AI 编程治理框架：让 Claude 写代码、让 Codex 独立审查、让人类做架构师——三方角色在一个精密的迭代闭环中各司其职，直到代码质量和目标对齐同时达标才允许退出循环。
:::

![文章插图](/articles/2026-05-15-claude-codex-humanize-rlcr-298625c6/images/03.webp)

::: image-caption
“Humanize Loop Monitor”的终端监控界面截图，直观呈现了AI驱动开发流程的实时状态。界面顶部显示会话启动时间，当前迭代进度为第5轮（共42轮），使用模型为`gpt-5.3-codex (xhigh)`，未开启每轮自动推送。当前状态显示为“Complete”，Codex提问功能已开启；进度栏标注7项验收标准（ACs）已全部完成，20项任务中1项进行中、19项已完成，同时存在1个待处理问题；Git仓库状态为“clean”，无未提交变更。界面通过多色文本区分关键信息，清晰展示了迭代开发的进度与待解决问题。
:::

## 本文目录

- 快速上手
- 一、架构总览与设计哲学
  - 1.1 目录结构一览
  - 1.2 多智能体角色分工
- 二、从灵感到计划：gen-idea 与 gen-plan 的多阶段流水线
  - 2.1 gen-idea：定向多样性探索
  - 2.2 gen-plan：Claude 与 Codex 的辩论式收敛
- 三、RLCR 循环：实现与审查的迭代闭环
  - 3.1 Hook 系统：无处不在的拦截器
  - 3.2 两阶段审查流程
  - 3.3 ask-codex.sh：一次性咨询的执行引擎
- 四、防止人类沦为橡皮图章：计划理解测验
- 五、Goal Tracker 与 BitLesson：对抗目标漂移与知识遗忘
  - 5.1 Goal Tracker：不可变目标锚点
  - 5.2 BitLesson：项目级经验教训库
- 六、配置系统：三层合并与优雅降级
- 七、rlcr-stop-gate：从 Hook 到独立可调用的门控抽象
- 总结与思考

![文章插图](/articles/2026-05-15-claude-codex-humanize-rlcr-298625c6/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> 安装只需三步，在 Claude Code 中执行：

```
# 添加 PolyArch 市场源/plugin marketplace add PolyArch/humanize# 安装 humanize 插件/plugin install humanize@PolyArch
```

使用核心流程：

```
# 1. 从一句话生成想法草案（可选）/humanize:gen-idea "add undo/redo to the editor"# 2. 从草案生成实施计划/humanize:gen-plan --input draft.md --output docs/plan.md# 3. 启动迭代开发循环/humanize:start-rlcr-loop docs/plan.md
```

需要预装 Codex CLI[1] 作为独立审查引擎。更多安装选项参考 `docs/install-for-claude.md`[2]，配置层级与环境变量参考 `docs/usage.md`[3]。

## 一、架构总览与设计哲学

> Humanize 的核心理念可以用四句话概括：**Iteration over Perfection** （迭代胜于完美）、**One Build + One Review** （一个实现者加一个独立审查者）、**Ralph Loop with Swarm Mode** （持续循环加可选并行）、**Begin with the End in Mind** （先确认你理解了计划再开始执行）。

整个项目的代码库几乎全部由 Shell 脚本构成（占比 99.5%），辅以少量 Python。这不是偶然——作为 Claude Code 的插件，它需要直接与终端环境交互，Shell 是最天然的胶水语言。

### 1.1 目录结构一览

```
humanize/├── commands/          # 用户可调用的斜杠命令定义（Markdown 格式的命令规范）├── scripts/           # 核心执行引擎（Shell 脚本）├── hooks/             # Claude Code 生命周期钩子（拦截读/写/执行/停止）├── agents/            # 子智能体 Prompt 定义（合规检查、测验出题等）├── skills/            # 技能注册目录（ask-codex、ask-gemini 等）├── prompt-template/   # 各阶段的 Prompt 模板├── config/            # 默认配置与 Codex hooks 配置└── templates/         # BitLesson 等文档模板
```

这里最值得注意的设计是 **commands 与 scripts 的分离**。`commands/` 下的 `.md` 文件并非文档，而是 **Claude Code 插件的命令定义规范** ——它们用 YAML front-matter 声明允许的工具列表，用 Markdown 正文描述 Claude 应该如何理解并执行这个命令。这本质上是一种 **用自然语言编写的程序**，由 AI 解释执行。

### 1.2 多智能体角色分工

```
┌─────────────┐    实现代码     ┌─────────────┐│   Claude     │ ──────────────→ │  代码仓库    ││ （实现者）    │ ←────────────── │             │└──────┬───── ┘    审查反馈     └─────────────┘       │ 写摘要                        ↑       ↓                              │ codex review┌─────────────┐                 ┌─────────────┐│   Codex      │ ───审查摘要───→ │   Codex      ││ （摘要审查）  │                │ （代码审查）  │└─────────────┘                 └─────────────┘
```

Claude 负责写代码，Codex 负责两轮独立审查（先审摘要、再审代码），人类负责最终决策。三方权责清晰，没有任何一方能独断全局。

## 二、从灵感到计划：gen-idea 与 gen-plan 的多阶段流水线

### 2.1 gen-idea：定向多样性探索

> 当用户只有一句话的模糊想法时，`gen-idea` 命令会启动一个 **定向多样性（Directed-Diversity）** 探索流程。这不是简单地让 AI 扩写你的想法，而是一种结构化的创意发散：

```
<!-- 来源：commands/gen-idea.md --><!-- Phase 2: Direction Generation 核心规则 -->Hard constraint: **orthogonality**.Two near-duplicate directions defeat the directed-diversity premise.
```

流程分五个阶段：输入解析 → IO 校验 → 方向生成 → **并行探索** （Swarm） → 综合输出。其中最精妙的是 Phase 3，它会为每个方向分别启动一个独立的 Explore 子智能体，这些子智能体 **只读不写**，在仓库中寻找客观证据：

```
<!-- 来源：commands/gen-idea.md Phase 3 -->> Explore this direction within the current repo. Gather OBJECTIVE EVIDENCE:>> - Specific repo paths with existing patterns worth extending.> - Prior art or precedent in the codebase or adjacent tooling.>   ...>   If no concrete evidence exists, report the literal string>   `exploratory, no concrete precedent` once.>   Fabrication of references is PROHIBITED.
```

这段设计极其务实——它强制 AI 不准编造代码引用。如果没有找到证据，就老老实实承认"纯探索性方向，无具体先例"。最终由主控 Claude 根据证据密度、模式契合度、实现表面积三个维度选出最强方向作为主方案。

### 2.2 gen-plan：Claude 与 Codex 的辩论式收敛

> `gen-plan` 是整个 Humanize 中最复杂的命令，它将草案转化为一份结构化的实施计划。核心亮点是 **双 Codex 多轮收敛机制**：

```
Phase 3: Codex 第一轮分析（发现风险、缺失需求、技术差距）    ↓Phase 4: Claude 基于 Codex 分析生成候选计划 v1    ↓Phase 5: 第二个 Codex 反复挑战候选计划（最多 3 轮）    ↓Phase 6: 未解决分歧交由人类裁决    ↓Phase 7: 输出最终计划
```

收敛判定逻辑非常清晰：

```
<!-- 来源：commands/gen-plan.md Phase 5 -->Repeat convergence rounds until:- No REQUIRED_CHANGES remain and no high-impact DISAGREE remains- Two consecutive rounds produce no material plan changes- Maximum 3 rounds reachedSet: PLAN_CONVERGENCE_STATUS=converged | partially_converged
```

每轮 Codex 审查必须输出结构化的 `AGREE / DISAGREE / REQUIRED_CHANGES / UNRESOLVED` 字段，Claude 再据此修订。这不是走过场的审查——它构建了一个 **逐轮收敛矩阵**，追踪每个争议点的演变历史。

## 三、RLCR 循环：实现与审查的迭代闭环

> RLCR（Ralph-Loop with Codex Review）是 Humanize 的核心引擎。理解它需要先理解一个关键概念：**什么是一个 Round**。

```
<!-- 来源：commands/start-rlcr-loop.md -->**One round = the agent believes the entire plan is finished.**A round is NOT one task, one milestone, one stage, or one layer.
```

一个 Round 不是完成一个小任务，而是 Claude **认为整个计划已经全部完成** 后，写下工作摘要并尝试退出循环。这时，hooks 系统拦截退出并触发审查。

### 3.1 Hook 系统：无处不在的拦截器

> Humanize 通过 Claude Code 的 Hook 机制，对 AI 的几乎所有行为进行拦截：

```
// 来源：hooks/hooks.json{  "PreToolUse": [    { "matcher": "Write",  "command": "loop-write-validator.sh" },    { "matcher": "Edit",   "command": "loop-edit-validator.sh" },    { "matcher": "Read",   "command": "loop-read-validator.sh" },    { "matcher": "Bash",   "command": "loop-bash-validator.sh" }  ],  "Stop": [    { "command": "loop-codex-stop-hook.sh", "timeout": 7200 }  ]}
```

注意 `Stop` 钩子的超时时间——**7200 秒（2 小时）**。这意味着当 Claude 尝试退出循环时，Codex 有最长 2 小时来进行深度代码审查。这个钩子脚本 `loop-codex-stop-hook.sh` 是整个仓库中最大的单文件（87KB），它实现了两阶段审查的完整逻辑。

### 3.2 两阶段审查流程

```
Implementation Phase（实现阶段）:  Claude 按任务标签路由执行：    coding → Claude 直接实现    analyze → 调用 /humanize:ask-codex  → Claude 写摘要 → 尝试退出  → Stop Hook 拦截 → Codex 审查摘要  → 发现问题 → 反馈 → Claude 继续改  → 输出 "COMPLETE" → 进入 Review PhaseReview Phase（代码审查阶段）:  → codex review --base <branch> 执行代码审查  → 按 [P0-P9] 严重度标记问题  → 有问题 → 继续修复  → 无问题 → 循环结束
```

### 3.3 ask-codex.sh：一次性咨询的执行引擎

> `ask-codex.sh` 是 Codex 调用的底层包装器，它体现了 Humanize 对 **可观测性** 的执着追求：

```
# 来源：scripts/ask-codex.sh（简化核心逻辑）# 每次调用都生成唯一 ID，建立完整的输入/输出/元数据追踪UNIQUE_ID="${TIMESTAMP}-$$-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"SKILL_DIR="$PROJECT_ROOT/.humanize/skill/$UNIQUE_ID"# 保存输入cat > "$SKILL_DIR/input.md" << EOF# Ask Codex Input## Question$QUESTION## Configuration- Model: $CODEX_MODEL- Effort: $CODEX_EFFORTEOF# 执行并计时START_TIME=$(date +%s)printf'%s'"$QUESTION" | run_with_timeout "$CODEX_TIMEOUT" \    codex exec"${CODEX_EXEC_ARGS[@]}" - \    > "$CODEX_STDOUT_FILE" 2> "$CODEX_STDERR_FILE" || CODEX_EXIT_CODE=$?# 保存元数据（包括退出码、耗时、状态）cat > "$SKILL_DIR/metadata.md" << EOF---tool: codexexit_code: 0duration: ${DURATION}sstatus: success---EOF
```

每一次 Codex 调用都会留下 `input.md`、`output.md`、`metadata.md` 三件套，加上缓存目录里的 `codex-run.cmd`（完整命令）、`codex-run.out`（原始输出）、`codex-run.log`（错误日志）。当循环跑了 20 轮之后，你可以完整回溯每一次 AI 交互的全部细节。

## 四、防止人类沦为橡皮图章：计划理解测验

> 这可能是 Humanize 中最具人文关怀的设计。在 RLCR 循环启动前，系统会 **考试** ——对人类考试：

```
<!-- 来源：commands/start-rlcr-loop.md -->Before running the setup script, verify the user genuinelyunderstands what the plan will do. This is an advisory check --it never blocks the loop, but catches "wishful thinking" userswho blindly accept generated plans.
```

考试流程：

1. **出题智能体** （`plan-understanding-quiz`，使用 Opus 模型）分析计划，生成 2 道四选一选择题
2. 系统向人类提问
3. 如果两题都答对，放行
4. 如果答错，展示计划摘要和正确答案，然后让人类选择：继续还是回去重新审阅计划

```
<!-- 来源：agents/plan-understanding-quiz.md -->Good question characteristics:- Test understanding of HOW things will be done, not just WHAT- A user who carefully read the plan should pick the correct answer- A user who just skimmed the title or blindly accepted  a generated plan would likely pick wrong
```

这个设计直指 AI 编程时代的一个核心隐忧：当 AI 越来越能干，人类开发者是否会变成只负责点"确认"的橡皮图章？

::: block-1
Humanize 用测验机制强制人类保持对技术决策的理解力。而且它是 **advisory 而非 blocking** ——答错也可以继续，但至少你知道自己可能没完全理解正在执行的计划。
:::

## 五、Goal Tracker 与 BitLesson：对抗目标漂移与知识遗忘

### 5.1 Goal Tracker：不可变目标锚点

长时间运行的迭代循环最大的风险是 **目标漂移** ——第 15 轮的实现可能已经偏离了第 1 轮设定的目标。Humanize 用 Goal Tracker 解决这个问题：

```
<!-- 来源：commands/start-rlcr-loop.md -->IMMUTABLE SECTION: Ultimate Goal and Acceptance Criteria(set in Round 0, never changed)MUTABLE SECTION: Active Tasks, Completed Items,Deferred Items, Plan Evolution Log
```

目标和验收标准一旦在 Round 0 设定，就 **永远不可修改**。如果实现过程中发现计划需要变更，必须记录在 Plan Evolution Log 中并附上理由。每 5 轮（可配置），Codex 会进行一次 **全面目标对齐审计**。

### 5.2 BitLesson：项目级经验教训库

每个项目维护一个 `.humanize/bitlesson.md` 文件，记录从迭代中学到的具体经验。每轮开始时 AI 必须先阅读 BitLesson，对每个子任务运行 `bitlesson-selector` 选择相关经验，并在摘要中报告应用情况。

::: block-1
这本质上是一个 **项目级强化学习机制** ——不是在模型权重层面，而是在 Prompt 层面，让 AI 从过去的错误中学习。
:::

## 六、配置系统：三层合并与优雅降级

> Humanize 的配置体系体现了工程成熟度：

```
// 来源：config/default_config.json{  "codex_model": "gpt-5.5",  "codex_effort": "high",  "bitlesson_model": "haiku",  "agent_teams": false,  "alternative_plan_language": "",  "gen_plan_mode": "discussion"}
```

配置按三层合并：**插件默认** → **用户全局** （`~/.config/humanize/config.json`） → **项目级** （`.humanize/config.json`），后者覆盖前者。格式错误的可选配置 **只警告不阻断**，体现了"优雅降级"原则。

值得一提的是 `gen_plan_mode` 有两种模式：`discussion`（默认，走完整的 Claude-Codex 辩论收敛流程）和 `direct`（跳过收敛直出计划，适合简单任务）。

## 七、rlcr-stop-gate：从 Hook 到独立可调用的门控抽象

> `rlcr-stop-gate.sh` 是一个精巧的设计——它将 Stop Hook 的逻辑包装成独立可调用的脚本，让 Skill 工作流也能复用同一套退出门控逻辑：

```
# 来源：scripts/rlcr-stop-gate.sh（核心决策逻辑）# 构造与真实 Stop 事件相同 schema 的 JSON 输入HOOK_INPUT=$(jq -n \    --arg cwd "$PROJECT_ROOT" \    '{        hook_event_name: "Stop",        stop_hook_active: false,        cwd: $cwd    }')# 调用真实的 Stop Hook 脚本HOOK_OUTPUT="$(printf '%s' "$HOOK_INPUT" | \    CLAUDE_PROJECT_DIR="$PROJECT_ROOT" "$HOOK_SCRIPT")"# 解析决策：block（阻止退出，exit 10）或 allow（放行，exit 0）DECISION="$(printf '%s' "$HOOK_OUTPUT" | jq -r '.decision // empty')"if [[ "$DECISION" == "block" ]]; then    exit 10  # 循环继续fiexit 0       # 允许退出
```

这种"让 Hook 逻辑同时服务于被动拦截和主动查询"的抽象，避免了逻辑分叉和重复实现。

## 总结与思考

> Humanize 项目的技术野心不在于让 AI 写出更多代码，而在于构建一套 **AI 编程时代的软件工程治理体系**。它回答了几个关键问题：

- **谁来审查 AI 写的代码？** → 另一个独立的 AI（Codex），通过结构化的两阶段审查。
- **如何防止 AI 偏离目标？** → 不可变的 Goal Tracker + 周期性全面审计。
- **如何防止人类变成橡皮图章？** → 启动前的理解力测验。
- **如何从错误中学习？** → 项目级 BitLesson 经验库。

从代码实现的角度看，99.5% Shell 脚本的技术选型虽然看似"原始"，但恰恰是最合理的：它依赖最少、与终端环境最亲和、调试最直观。每个脚本都有完善的参数校验、超时处理、错误码体系和可追溯的日志链——这是严肃的生产级工程，不是玩具项目。

::: block-1
当我们谈论"AI 编程的未来"时，多数人想的是更强的模型、更快的代码生成。但 Humanize 指出了一个更本质的方向：**不是让 AI 更强，而是让 AI 的协作更可治理**。这或许才是从 Automated Idea Factory 到 Realization 真正需要跨越的那道鸿沟。
:::

参考资料

[1] Codex CLI: *https://github.com/openai/codex*

[2] `docs/install-for-claude.md`: *https://github.com/PolyArch/humanize/blob/main/docs/install-for-claude.md*

[3] `docs/usage.md`: *https://github.com/PolyArch/humanize/blob/main/docs/usage.md*
