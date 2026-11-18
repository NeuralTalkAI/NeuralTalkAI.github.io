关键词：HWE-Bench、**硬件 Bug 修复**、:mdnice-bold-italic[LLM Agent]、:mdnice-bold-italic[仓库级基准]、异构计算、**RTL 验证**

> “硬件任务暴露了软件基准所压缩的性能差异——在 SWE-bench 上，所有模型挤在 73%到 81%的窄带内，而在 HWE-Bench 上，同样的模型从 47.7%散落至 70.7%，差距从不足 8%骤然拉大到超过 23%。”

2023 年，SWE-bench 横空出世，让大语言模型在真实软件工程任务上的能力有了可量化、可复现的标尺。

::: block-1
两年过去，当 GPT-5.4 和 Claude Opus 在软件 Bug 修复上已逼近 80%的解决率时，一个令人不安的问题始终悬而未决：**这些在 Python 代码库里游刃有余的 AI 智能体，面对 Verilog 的并行语义、Chisel 的元编程抽象、以及芯片验证所需的跨层级协同，究竟能走多远？**
:::

![文章插图](/articles/2026-05-02-bug-hwe-bench-dc36f0e0/images/01.webp)

- **HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks**
- https://arxiv.org/pdf/2604.14709
- 1.4 万字，阅读 50 分钟，**播客 20 分钟**

::: related-title
**相关推荐**
:::

::: related-links
- **[Agentic AI狂卷芯片设计：Cadence 结构化心智模型让 Altera 验证效率快10倍 ！Tenstorrent直呼快4倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897410&idx=1&sn=378a387251f87c6a23c03a9240e68370&scene=21#wechat_redirect)**
- **[像写软件一样造芯片：Google XLS 让软件工程师写Verilog，用编译器思维颠覆硬件开发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899137&idx=1&sn=c98294c41e5453f33c67ee1b3c826838&scene=21#wechat_redirect)**
- **[Meta AI 基础设施的十年演进、下一代基础设施蓝图与 AI 自研芯片之路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896790&idx=1&sn=a13b82d8c267fcdbd6f3e20c53a1429b&scene=21#wechat_redirect)**

:::

> 北京大学梁云团队给出的答案，比多数人预期的更残酷，也更有启发性。:mdnice-bold-italic[他们发布的 HWE-Bench，从 OpenTitan、XiangShan、Rocket Chip 等六个明星开源芯片项目中，萃取 417 个真实历史 Bug 修复任务，构建起硬件设计领域首个仓库级、执行闭环的评测基准。每一个任务都不是孤立的模块级习题，而是嵌入完整项目仓库的工程难题] ——AI 智能体需要像人类工程师一样，阅读 Bug 报告、定位故障源、跨越多层设计工件（RTL 源码、验证组件、IP 配置、构建脚本）协调修改，并通过项目的原生仿真流程才算过关。

这篇论文的核心洞察力在于：**硬件调试的难度并不由代码量的绝对值决定，而是由项目范围（Project Scope）和 Bug 类型分布（Bug-Type Distribution）共同塑造。** 一个 4.7 万行代码的 Rocket Chip，其修复难度与 150 万行的 OpenTitan 并驾齐驱，因为前者的 Bug 高度集中于配置与集成类错误——这类错误的表征往往指向与根因完全不同的模块，这正是当前 LLM 最脆弱的能力边界。

![文章插图](/articles/2026-05-02-bug-hwe-bench-dc36f0e0/images/02.webp)

::: image-caption
图 1：HWE-Bench 中一个任务实例的概述。该图完整呈现 HWE-Bench 单任务执行闭环：LLM 智能体获取硬件仓库与缺陷报告，通过文件读写、Shell 执行、仿真工具生成补丁，在容器化 EDA 环境中经项目原生仿真验证。此设计抛弃组件级隔离评估，要求智能体遍历完整硬件仓库、调用原生验证流程，高度贴合真实硬件调试场景。实验中该闭环保障验证可复现，也暴露智能体需具备跨文件检索、硬件时序语义理解能力，为硬件缺陷修复代理提供标准化评估范式。
:::

> 实验数据描绘出一幅层次分明的能力图谱。

- **最强智能体 GPT-5.4 xhigh 拿下 70.7%的整体修复率**，在 Ibex、CVA6 等中小核心上突破 90%，但在 OpenTitan 和 XiangShan 这类 SoC 级项目上骤降至 65%以下。
- 更值得警惕的是，**417 个任务中有 45 个成为所有模型的“死亡地带”，无一攻克。**
- :mdnice-bold-italic[失败分析直指当前 AI 硬件能力的三大命门：故障定位偏差（把症状当病因）、硬件语义推理缺失（修得了语法改不对状态机）、以及跨工件协同断裂（修完 RTL 忘了更新 HJSON 配置描述）。]

![文章插图](/articles/2026-05-02-bug-hwe-bench-dc36f0e0/images/03.webp)

::: image-caption
图 5：同一组模型在三个基准测试的整体解决率。该图对比 7 款大模型在软硬件基准的解决率，硬件任务中模型性能分化远大于软件基准，闭源模型显著领先开源模型。软件基准中模型表现高度集中，而 HWE-Bench 拉开超 23% 差距，说明硬件任务更易暴露模型能力差异。且无模型在 HWE-Bench 达饱和性能，验证其作为挑战性基准的价值，也指明硬件感知代理需重点提升硬件语义与跨组件推理能力。
:::

![文章插图](/articles/2026-05-02-bug-hwe-bench-dc36f0e0/images/04.webp)

::: image-caption
表 3：展示 7 款大模型在 6 个仓库的解决率与文件级精度，GPT-5.4 整体解决率 70.7% 居首，闭源模型优于开源模型；小核项目解决率超 90%，SoC 项目低于 65%。Claude 系列精度最高，GPT-5.4 精度低但解决率高，源于其系统级跨组件修改。数据证实项目规模、缺陷类型决定难度，而非代码量，也揭示不同模型调试策略差异，为硬件代理优化提供数据支撑。
:::

::: block-1
HWE-Bench 的意义远不止于一张排行榜。

它为 AI 辅助硬件设计（Electronic Design Automation，即电子设计自动化）的研究确立了一个真正贴近工程现实的战场，也让业界第一次看清：:mdnice-bold-italic[从“会写 Verilog”到“能修芯片 Bug”，中间横亘的不是算力，而是 对并发硬件语义的深度理解与跨抽象层级的因果推理能力。]
:::

## 本文目录

- 一、从模块生成到仓库级调试：硬件评测的范式跃迁
- 二、HWE-Bench 的构建哲学：自动化为矛，审慎为盾
- 三、任务全景：417 个 Bug、六种芯片、五类错误的真实战场
- 四、七模型四框架正面对决：性能数据背后的残酷真相
- 五、失败的解剖学：三种致命失败与四类模型性格
- 六、相关工作：硬件自动化评测的前世今生
- 七、结论与展望

![文章插图](/articles/2026-05-02-bug-hwe-bench-dc36f0e0/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[Agentic AI狂卷芯片设计：Cadence 结构化心智模型让 Altera 验证效率快10倍 ！Tenstorrent直呼快4倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897410&idx=1&sn=378a387251f87c6a23c03a9240e68370&scene=21#wechat_redirect)**
- **[像写软件一样造芯片：Google XLS 让软件工程师写Verilog，用编译器思维颠覆硬件开发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899137&idx=1&sn=c98294c41e5453f33c67ee1b3c826838&scene=21#wechat_redirect)**
- **[Meta AI 基础设施的十年演进、下一代基础设施蓝图与 AI 自研芯片之路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896790&idx=1&sn=a13b82d8c267fcdbd6f3e20c53a1429b&scene=21#wechat_redirect)**

:::
