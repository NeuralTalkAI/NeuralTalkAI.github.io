关键词：**AI 编程智能体**、:mdnice-bold-italic[技能框架]、**测试驱动开发**、子智能体驱动开发、软件工程方法论

> 你有没有遇到过这种情况：让 AI 帮你写一个"简单的"Todo 应用，它二话不说就吐出三百行代码——没有设计文档，没有测试，变量名随心所欲，更别提什么架构思考了。你说"加个功能"，它把之前的代码推倒重来。你说"修个 Bug"，它在症状上打补丁，引入两个新 Bug。

这不是 AI 不够聪明的问题，而是**缺乏工程纪律** 的问题。人类程序员经过多年训练才学会"先设计后编码""先写测试后实现""先找根因后修复"，但 AI 智能体默认没有这套规矩。

![文章插图](/articles/2026-05-21-20-star-superpowers-tdd-74093e95/images/01.webp)

- **An agentic skills framework & software development methodology that works.**
- https://github.com/obra/superpowers
- 3500 字，阅读 17 分钟，**播客 14 分钟**

相关推荐

- [Claude 实现、Codex 审查、人类决策领航：Humanize 项目用 RLCR 编排长程闭环迭代任务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901161&idx=1&sn=88251ef01180c0f214ca6517d40beda1&scene=21#wechat_redirect)
- [把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)
- [TokenSpeed 把 LLM 推理拆成可控流水线，为智能体编码时代重写 LLM 推理引擎的控制平面](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900928&idx=1&sn=a7a2425adc6733e807494b3ce7dbcc82&scene=21#wechat_redirect)

> **Superpowers** 正是要解决这个问题：它不是又一个代码生成工具，而是一套**强制 AI 智能体遵循软件工程最佳实践的技能框架与开发方法论**。它用"技能"取代"提示词"，用流程门禁取代随意发挥，让 AI 智能体从"能写代码"进化为"能做工程"。

## 本文目录

- 快速上手
- 一、架构设计
  - 1.1 项目定位：不是工具，是方法论
  - 1.2 目录结构与技术组成
  - 1.3 插件机制：技能如何被加载
- 二、核心工作流：从创意到交付的七步流水线
  - 2.1 brainstorming：拒绝"上来就写代码"
  - 2.2 writing-plans：为"没有品味的热心初级工程师"写计划
  - 2.3 subagent-driven-development：子智能体流水线
- 三、TDD 的极端执行：铁律与反合理化
  - 3.1 测试驱动开发的"铁律"
  - 3.2 反合理化表：堵住 AI 的"借口"
- 四、技能的元层设计：writing-skills
  - 4.1 用 TDD 写文档
  - 4.2 CSO：面向 AI 搜索的优化
- 五、系统化调试：四阶段根因分析
- 总结与启示

![文章插图](/articles/2026-05-21-20-star-superpowers-tdd-74093e95/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> Superpowers 支持 Claude Code、Codex CLI、Gemini CLI、Cursor 等多种 AI 编程环境。以最常用的 Claude Code 为例，一行命令即可安装：

```
/plugin install superpowers@claude-plugins-official
```

安装后无需任何额外配置。当你在对话中提出"Let's build X"时，Superpowers 会**自动触发** 头脑风暴技能，引导智能体先做设计再写代码。更多安装方式（Codex、Gemini CLI、Cursor 等），参考 README 安装指南[1]。

## 一、架构设计

### 1.1 项目定位：不是工具，是方法论

> Superpowers 的定位非常独特——它自称是"一套完整的软件开发方法论"（a complete software development methodology）。传统意义上的代码生成工具关注"写什么代码"，而 Superpowers 关注的是"**AI 智能体该如何做工程** "。

项目的核心哲学可以用四句话概括：

- **测试驱动开发** （Test-Driven Development）——先写测试，永远如此
- **系统化优于随意** （Systematic over ad-hoc）——流程优于猜测
- **降低复杂性** （Complexity reduction）——简洁是首要目标
- **证据优于声称** （Evidence over claims）——验证后才能宣告成功

### 1.2 目录结构与技术组成

```
superpowers/├── .claude-plugin/       # Claude Code 插件元数据├── .codex-plugin/        # Codex 插件适配├── .cursor-plugin/       # Cursor 插件适配├── .opencode/            # OpenCode 插件适配├── skills/               # ★ 核心：所有技能定义│   ├── brainstorming/│   ├── subagent-driven-development/│   ├── test-driven-development/│   ├── writing-plans/│   ├── systematic-debugging/│   ├── using-git-worktrees/│   └── ...（共14个技能）├── scripts/              # 版本管理与插件同步脚本├── tests/                # 自动化技能测试└── hooks/                # Git 钩子
```

有趣的是，项目 66% 的代码是 Shell 脚本——这些脚本主要用于插件同步、版本管理和测试执行，而非"核心逻辑"。**真正的核心是 `skills/` 目录下的 Markdown 文件**。

::: block-1
没错，Superpowers 的"代码"本质上是一组精心编写的**结构化指令文档**，它们通过插件系统注入到 AI 智能体的上下文中，从而改变智能体的行为模式。
:::

### 1.3 插件机制：技能如何被加载

```
{  "name": "superpowers",  "description": "Core skills library for Claude Code: TDD, debugging, collaboration patterns, and proven techniques",  "version": "5.1.0",  "author": { "name": "Jesse Vincent" },  "license": "MIT"}
```

当 AI 智能体启动会话时，插件系统加载 `using-superpowers` 这个"引导技能"，它会注入一条铁律：**在任何响应之前，先检查是否有适用的技能**。

## 二、核心工作流：从创意到交付的七步流水线

Superpowers 定义了一条完整的软件开发流水线，每一步都由对应的技能强制执行：

```
brainstorming → using-git-worktrees → writing-plans → subagent-driven-development → test-driven-development → requesting-code-review → finishing-a-development-branch
```

这不是"建议"，而是**强制流程**。让我们逐一拆解关键环节。

### 2.1 brainstorming：拒绝"上来就写代码"

> 这是整条流水线的入口。当用户说"让我们做一个 X"时，智能体**不会** 立刻开始写代码，而是进入苏格拉底式的设计对话：

```
<HARD-GATE>Do NOT invoke any implementation skill, write any code, scaffold any project,or take any implementation action until you have presented a designand the user has approved it.</HARD-GATE>
```

这个 `<HARD-GATE>` 标签是 Superpowers 的一个核心设计模式：**用明确的门禁阻止智能体跳过流程**。

::: block-1
brainstorming 技能要求智能体完成一个 9 步清单：探索项目上下文 → 提出澄清问题（每次只问一个）→ 提出 2-3 种方案 → 分段呈现设计 → 撰写设计文档 → 自审规格 → 用户审核 → 转交给计划编写。
:::

特别值得注意的是其"反模式"声明：

> *Every project goes through this process. A todo list, a single-function utility, a config change — all of them. "Simple" projects are where unexamined assumptions cause the most wasted work.*

即使是一个"简单的"配置修改，也必须经过设计流程。这种"偏执"看似过度，但正是对 AI 智能体"自以为理解了需求就开始写"这一致命倾向的精确对治。

### 2.2 writing-plans：为"没有品味的热心初级工程师"写计划

> 计划编写技能的描述极其生动：

```
Write comprehensive implementation plans assuming the engineer has zero contextfor our codebase and questionable taste.
```

每个任务被拆解为 2-5 分钟的"一口大小"步骤，每一步都必须包含：

- **精确的文件路径** （不允许"在相关文件中"这种模糊表达）
- **完整的代码** （不允许"类似 Task N"的交叉引用）
- **精确的运行命令与预期输出**
- **TDD 的 Red-Green 循环**

更关键的是，计划中**严禁占位符**：

```
Every step must contain the actual content an engineer needs. These are plan failures:- "TBD", "TODO", "implement later"- "Add appropriate error handling"- "Write tests for the above" (without actual test code)- "Similar to Task N"
```

### 2.3 subagent-driven-development：子智能体流水线

> 这是 Superpowers 最具创新性的技能。它将实现工作分解为**逐任务的子智能体流水线**：

```
每个任务:  Implementer子智能体(实现+自审+提交)    ↓  Spec Reviewer子智能体(规格合规审查)    ↓ (不合规则返回Implementer修复)  Code Quality Reviewer子智能体(代码质量审查)    ↓ (不合格则返回Implementer修复)  标记任务完成 → 下一个任务
```

核心设计原则是**每个任务启用全新子智能体**，避免上下文污染。控制器（主智能体）负责一次性读取计划、提取所有任务的完整文本，然后逐个分发给子智能体。子智能体不需要自己去读计划文件——控制器已经准备好了所有它需要的信息。

```
**Core principle:** Fresh subagent per task + two-stage review(spec then quality) = high quality, fast iteration
```

两阶段审查的顺序是**不可调换的**：必须先通过规格合规审查（"你实现了规格要求的所有功能吗？有没有多实现了什么？"），然后才能进入代码质量审查。Red Flags 列表中明确写道：

> *Start code quality review before spec compliance is ✅ (wrong order)*

## 三、TDD 的极端执行：铁律与反合理化

### 3.1 测试驱动开发的"铁律"

> Superpowers 对 TDD 的执行达到了近乎"极端"的程度：

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRSTWrite code before the test? Delete it. Start over.**No exceptions:**- Don't keep it as "reference"- Don't "adapt" it while writing tests- Don't look at it- Delete means delete
```

如果智能体在写测试之前写了实现代码，必须**删除实现代码**，不能保留为"参考"，不能"边写测试边适配"，甚至不能"看一眼"。这不是建议，是铁律。

### 3.2 反合理化表：堵住 AI 的"借口"

> Superpowers 深谙 AI 智能体的心理——它们非常擅长为跳过流程找合理化借口。因此每个纪律性技能都配备了一张**合理化借口对照表**：

| 借口 | 现实 |
| --- | --- |
| "太简单了不需要测试" | 简单代码也会出错。测试只要 30 秒。 |
| "我先写代码，之后再补测试" | 后补的测试立即通过，证明不了任何事。 |
| "删掉 X 小时的工作太浪费了" | 沉没成本谬误。保留不可信的代码才是真正的浪费。 |
| "TDD 太教条了，实用主义意味着灵活" | TDD 本身就是实用主义。 |

这种设计源自一个深刻的洞察：**AI 智能体不是不知道规则，而是会在压力下合理化绕过规则**。Superpowers 不仅定义规则，还预判并封堵了智能体可能使用的每一种逃避话术。

## 四、技能的元层设计：writing-skills

### 4.1 用 TDD 写文档

> Superpowers 有一个极其精妙的"元技能"——`writing-skills`，它将 TDD 方法论应用到了技能文档本身的编写：

```
Writing skills IS Test-Driven Development applied to process documentation.
```

| TDD 概念 | 技能创建 |
| --- | --- |
| 测试用例 | 给子智能体的压力场景 |
| 生产代码 | 技能文档（SKILL.md） |
| 测试失败（RED） | 没有技能时智能体违反规则 |
| 测试通过（GREEN） | 有技能时智能体遵守规则 |
| 重构 | 封堵新发现的借口漏洞 |

这意味着：创建一个新技能时，你必须先**在没有这个技能的情况下测试智能体行为**，观察它在压力下如何失败、用了什么借口，然后才能针对这些失败模式编写技能文档。

### 4.2 CSO：面向 AI 搜索的优化

> 技能的 YAML frontmatter 中的 `description` 字段有一条反直觉的规则：**只描述触发条件，不要概括技能的工作流程**。

```
# ❌ 错误：概括了工作流，AI会跟着description走而跳过正文description:Usewhenexecutingplans-dispatchessubagentpertaskwithcodereview# ✅ 正确：只描述触发条件description:Usewhenexecutingimplementationplanswithindependenttasks
```

原因在于：测试发现，如果 description 包含了工作流摘要，AI 智能体会**直接按照 description 执行而跳过阅读完整技能内容**。这个细节体现了 Superpowers 团队对 AI 行为模式的深度理解。

## 五、系统化调试：四阶段根因分析

> `systematic-debugging` 技能定义了一个四阶段调试流程：

```
Phase 1: 根因调查 → Phase 2: 模式分析 → Phase 3: 假设与验证 → Phase 4: 实现修复
```

其中最具洞察力的规则是**三次修复失败规则**：

```
- If < 3: ReturntoPhase1, re-analyzewithnewinformation-If ≥ 3:STOPandquestionthearchitecture-DON'TattemptFix #4withoutarchitecturaldiscussion
```

如果连续三次修复尝试都失败了，这不是一个"假设错误"的问题，而是一个**架构错误** 的信号。必须停下来与人类伙伴讨论根本性的设计问题。

## 总结与启示

> Superpowers 的核心创新在于：它将"软件工程最佳实践"从人类程序员的隐性知识，转化为了 AI 智能体的**可执行指令系统**。这些指令不是温柔的"建议"，而是带有门禁、反合理化表和红旗检查的**强制流程**。

从技术实现上看，整个系统的"代码"本质上是 Markdown 文档——但这些文档经过了如同软件工程般的严格测试（用子智能体做压力测试）、版本管理和跨平台适配。它用文档的形式实现了编译器的功能：将高层意图（"构建一个 X"）编译为严格的工程流程。

::: block-1
这个项目给我们的启示或许是：**AI 智能体的能力瓶颈不在智能，而在纪律**。正如 Superpowers 的 README 所言——"mandatory workflows, not suggestions"。:mdnice-bold-italic[当我们学会给 AI 套上工程纪律的缰绳，它才能真正成为可信赖的"超级力量"。]
:::

参考资料

[1] README 安装指南: *https://github.com/obra/superpowers#installation*
