关键词：**判断工程**、:mdnice-bold-italic[多智能体编码]、Humanize、**联合采样**、机械门控

> “智能体编程（Agentic coding）降低了代码生成的成本，但实现可靠、准确的代码补全依然困难：负责编写代码的智能体，往往难以准确判断自身是否已真正完成任务。”这是作者开篇的核心论断，也点出了当前智能体编码领域最隐蔽的痛点。

**智能体编码让代码生成变得廉价，但可靠的完成度判断始终是难题。** 现有设计大多将完成度决策权交给执行任务的智能体，:mdnice-bold-italic[自精炼循环、Ralph 循环等模式本质上都是自我验收，模型的系统性盲区永远无法被自身发现。]

![文章插图](/articles/2026-10-08-mlsys-humanize-judgement-engineering-5e86f7ce/images/01.webp)

- **Humanize: Judgement Engineering for Agentic Coding**
- 组织机构：NVIDIA、UCLA、MIT、清华大学
- https://arxiv.org/pdf/2610.08900
- Project: https://humanfia.ai/
- Humanize: https://github.com/PolyArch/humanize
- Humanize2: https://github.com/humanfia/humanize
- 文档：https://docs.humanfia.ai/humanize/

::: related-title
**相关推荐**
:::

::: related-links
- **[Claude 实现、Codex 审查、人类决策领航：Humanize 项目用 RLCR 编排长程闭环迭代任务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901161&idx=1&sn=88251ef01180c0f214ca6517d40beda1&scene=21#wechat_redirect)**
- **[KDA²：Kernel Design Agents 优化 Kimi Delta Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906834&idx=1&sn=53fcb953cfd29bf2cffbb8d2631ba4b5&scene=21#wechat_redirect)**
- **[跑掉四五千万人民币等值 Token 之后，两位英伟达工程师重新理解了 Agent：代码不是护城河，时间才是](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906785&idx=1&sn=28a960b4c341e8bf26543827b92694a9&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**

:::

> **Humanize 提出了一套以判断工程为核心的多智能体编排工作流，在规划、实现、评审、学习的边界上设置明确的机械执行决策点。** 人类审批计划契约，构建智能体分轮实现，:mdnice-bold-italic[另一评审智能体独立决定是否完成，确定性钩子负责工作流路由，72 道机械门控强制约束边界。]

这套体系在真实场景中展现出了极强的落地能力：

- 108 天迭代 68 个版本，收获 1468 个 GitHub 星；
- 完成 567 个文件的 gem5 构建系统迁移；
- 在 MLSys 2026 FlashInfer 竞赛三个赛道全部进入前三；
- 拿下 IOI、IMO、IPhO 等多项奥赛满分与金牌；
- PutnamBench 取得 672/672 的满分成绩，登顶 Lean-Eval 排行榜。

::: block-1
本文将深入拆解 Humanize 的设计哲学、技术实现与落地经验，:mdnice-highlight[探讨判断工程如何成为智能体编码的下一个核心方向。]
:::

## 本文目录

- 一、AI 编码的核心困境：生成很便宜，验收很昂贵
- 二、判断工程：把验收权从构建者手里独立出来
- 三、系统实现：72 道机械门控下的 RLCR 循环
- 四、联合采样的数学本质：双模型马尔可夫链
- 五、落地验证：从奥赛金牌到大型工程迁移
- 六、118 份真实复盘：五个落地设计教训
- 七、总结与展望

![文章插图](/articles/2026-10-08-mlsys-humanize-judgement-engineering-5e86f7ce/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**
- **[陈天奇领衔 MLC 社区推出 TIRx Harness：面向智能体 GPU 编程的开源编译支撑框架（Compiler Harness）](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906450&idx=1&sn=9620af3802c9d3536f08680baf337cf1&scene=21#wechat_redirect)**
- **[TIRx：TVM新一代硬件内核DSL，FP8/NVFP4/FlashAttention4实测性能逼近行业最优实现，底层统一支撑人工、Agent、MegaKernel三类高性能内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902815&idx=1&sn=e440508ea6c74f9393cf6d5406e56ed8&scene=21#wechat_redirect)**
- **[陈天奇团队提出 CAKE：编译器与智能体协同进化，可进化 IR 让 KDA 提速 2.05 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904925&idx=1&sn=ab085dc1e9e6c73917c8853a398775d7&scene=21#wechat_redirect)**

:::
