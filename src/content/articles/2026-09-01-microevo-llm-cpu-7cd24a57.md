关键词：**微架构设计空间探索**、大语言模型、蒙特卡洛树搜索、**多目标优化**、:mdnice-bold-italic[Pareto 前沿]

> 微架构设计空间探索（DSE）始终是高端芯片设计的核心瓶颈：百亿级参数组合构成的超高维离散空间，搭配周期级仿真带来的高昂PPA（性能/功耗/面积）评估成本，让传统盲目搜索算法长期困于局部最优、仿真预算浪费与帕累托收敛乏力的三重困境。

![文章插图](/articles/2026-09-01-microevo-llm-cpu-7cd24a57/images/01.webp)

- 论文标题：**MicroEvo: Knowledge-Guided LLM Sampling for Efficient Microarchitecture Design Space Exploration**
- 组织机构：东南大学、国家 EDA 技术创新中心、NVIDIA、复旦大学、南京邮电大学、中国科学院计算技术研究所、北京大学
- 论文链接：https://arxiv.org/pdf/2608.06183
- 开源代码：https://github.com/GEAR-SEU/MicroEvo-ICCAD-26

::: related-title
**相关推荐**
:::

::: related-links
- **[MICRO'23 闭环硬件优化：DOSA 如何用可微分建模革新 DNN 加速器设计空间探索](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896412&idx=1&sn=48dfdb800add7fb71b9cffd7aefbc4e5&scene=21#wechat_redirect)**
- **[卷积提速 55 倍！为 TinyML 打造的 FPGA 全栈开源框架 CFU Playground，敏捷探索轻量加速器设计空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888011&idx=1&sn=3ea5446297b91cdd15378baa2ed83125&scene=21#wechat_redirect)**
- **[ISCA'24 揭示SRAM与计算资源最优配比！DSE创新方法Orojenesis：张量算法的可达数据移动与操作强度界限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891559&idx=1&sn=50b6d88425d3829e823a33a38b915801&scene=21#wechat_redirect)**

:::

> :mdnice-bold-italic[来自东南大学、国家 EDA 技术创新中心、NVIDIA、复旦大学、南京邮电大学、中国科学院计算技术研究所、北京大学的研究团队，提出的 MicroEvo 框架彻底重构了这一范式，首次将大语言模型（LLM）的架构先验知识与蒙特卡洛树搜索（MCTS）的结构化进化能力深度融合]，打造出知识引导的自进化探索闭环。

![文章插图](/articles/2026-09-01-microevo-llm-cpu-7cd24a57/images/02.webp)

::: image-caption
图 5：MicroEvo 框架概览。这张系统框图完整呈现了 MicroEvo 的四大核心模块：LLM 驱动的初始化与进化算子、Pareto-UCT 选择策略、主动知识积累（AKA）、状态感知指令（SAD）。整体流程从 LLM 生成差异化根节点开始，经 Pareto-UCT 选出精英节点，由知识调谐器和模式探索器两个 LLM 算子生成新设计，PPA 评估后更新搜索树结构；同时 AKA 从评估结果中提取可复用的优化知识存入记忆库，SAD 根据实时搜索进度动态切换探索 / 利用模式。整个闭环形成了 “生成 - 评估 - 学习 - 适配” 的自进化循环，区别于传统方法的盲搜索或纯数据驱动拟合范式。
:::

其核心突破在于四点：

- **Pareto-UCT 选点策略** ——在巨量的设计版图里挑选勘探点位，该过程既评估每个候选点能拓展出多大的“优质设计新疆域”，又主动避开扎堆重复的区域以保障探索多样性，:mdnice-bold-italic[让树形搜索算法从底层适配“性能、功耗、面积多目标兼顾”的需求。]
- **双模态 LLM 进化算子** ——本质是搭载了两套优化引擎：一套做“精细打磨”，在已有优质设计的基础上调校参数、雕琢细节；另一套做“跳脱创新”，跳出当前设计范式去尝试全新的架构思路，:mdnice-bold-italic[同时兼顾了局部优化的精度和全局探索的广度。]
- **主动知识积累机制** ——每完成一次仿真评估，都不会让昂贵的计算结果白白流失，:mdnice-bold-italic[评估后会主动从中提炼出“哪些参数搭配效率最高”“调整哪部分收益最显著”这类可复用的经验规律，存入“优化知识库”反复调用，让搜索过程越迭代越聪明。]
- **状态感知动态指令** ——系统会实时监测优化进度，:mdnice-bold-italic[一旦发现长期没有实质突破、陷入局部瓶颈，就自动从“深耕细作”模式切换到“大胆探路”模式]，从根源上避免了大模型驱动方法容易“过早困在局部最优里出不来”的通病。

::: block-1
实验结果：在 RISC-V 乱序核上，MicroEvo 较经典 NSGA-II 实现帕累托超体积最高 36.2% 提升，**搜索效率达 10.6 倍**；在工业级香山昆明湖核心上验证了强扩展性，**能效超越手动专家设计 8.6%，** 而 LLM 推理开销仅占总运行时的 7.31%，完全具备工程落地性。
:::

## 本文目录

- 一、问题的本质：微架构 DSE 为什么这么难
- 二、MicroEvo 总体框架：LLM+MCTS 的进化式搜索
- 三、Pareto-UCT：多目标下的节点选择策略
- 四、知识驱动的进化算子与主动知识积累
- 五、状态感知指令：让搜索学会自我调节
- 六、实验验证：从 RISC-V 核到香山处理器
- 七、总结与展望

![文章插图](/articles/2026-09-01-microevo-llm-cpu-7cd24a57/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[MICRO'23 闭环硬件优化：DOSA 如何用可微分建模革新 DNN 加速器设计空间探索](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896412&idx=1&sn=48dfdb800add7fb71b9cffd7aefbc4e5&scene=21#wechat_redirect)**
- **[卷积提速 55 倍！为 TinyML 打造的 FPGA 全栈开源框架 CFU Playground，敏捷探索轻量加速器设计空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888011&idx=1&sn=3ea5446297b91cdd15378baa2ed83125&scene=21#wechat_redirect)**
- **[ISCA'24 揭示SRAM与计算资源最优配比！DSE创新方法Orojenesis：张量算法的可达数据移动与操作强度界限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891559&idx=1&sn=50b6d88425d3829e823a33a38b915801&scene=21#wechat_redirect)**

:::
