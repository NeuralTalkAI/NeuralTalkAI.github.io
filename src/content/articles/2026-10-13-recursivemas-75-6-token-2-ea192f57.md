关键词：**递归多智能体**、:mdnice-bold-italic[隐空间协作]、RecursiveLink、:mdnice-bold-italic[内外循环训练]、**推理效率**

> 智能体协作本身能否通过递归方式实现规模化（Scaling）？

随着大语言模型单模型能力逐渐逼近瓶颈，多智能体系统成为突破能力边界的核心方向，**但传统多智能体始终受限于文本交互的底层瓶颈：每一次信息传递都需要经过编解码流程，带来巨大的延迟与 token 开销**，训练时更会造成梯度消失，难以实现端到端系统优化。

![文章插图](/articles/2026-10-13-recursivemas-75-6-token-2-ea192f57/images/01.webp)

- **Recursive Multi-Agent Systems: Scaling Agent Collaboration through Recurrent Depth in Latent Space**
- 组织机构：斯坦福大学、伊利诺伊大学厄巴纳-香槟分校、NVIDIA、麻省理工学院
- https://arxiv.org/pdf/2604.25917
- 项目页面：https://recursivemas.github.io/
- 在线演示：https://vishalmysore.github.io/recursiveMASDemo/
- **开源代码：https://github.com/RecursiveMAS/RecursiveMAS**

::: related-title
**相关推荐**
:::

::: related-links
- **[颠覆模型压缩与复用逻辑！基于 1100+个模型的“隐藏共性”：无论初始化、任务或领域，都会收敛到架构特定的层级共享低维子空间](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895577&idx=1&sn=cf89783edaf46f750bce3c37bc70c56d&scene=21#wechat_redirect)**
- **[多智能体系统的双层并行推理框架 TIPEX：GAIA 精度最高提 13%，时延压缩超 40%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905315&idx=1&sn=aa3244df5de0267b4a56e5aaee050a0e&scene=21#wechat_redirect)**
- **[穿透 Python/C++/CUDA 三层语义！用多智能体做深度学习框架代码静态分析，挖出 PyTorch 31 个真实 Bug](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904620&idx=1&sn=fce6da9484068bedcc1701af439b5a4c&scene=21#wechat_redirect)**

:::

> 递归语言模型的出现为深度推理提供了新的思路，:mdnice-bold-italic[通过复用 Transformer 层在隐空间迭代优化，单模型的推理深度得到显著提升。但这种递归范式始终局限于单个模型内部，无法利用多智能体的能力互补优势。]

**RecursiveMAS 的核心洞察，就是将递归缩放的范式从单模型内部推广到整个多智能体系统，把整个系统视为一个统一的隐空间递归计算过程** *，通过轻量的 RecursiveLink 模块连接异构智能体，形成闭环协作。

![文章插图](/articles/2026-10-13-recursivemas-75-6-token-2-ea192f57/images/02.webp)

::: image-caption
这张图展示RecursiveLink的两种结构：Inner内部模块与Outer外部模块。Inner针对同一个智能体，将模型最后一层嵌入，送入两层Linear+GELU的MLP，通过残差连接，输出该智能体输入层嵌入，实现同主体内部表征映射。Outer用于不同智能体间，把Agent $A_i$的嵌入输入同款MLP，额外增加一条线性支路做残差，输出Agent $A_j$的嵌入，完成跨智能体的表征递归链接，以此建立不同智能体表征之间的映射关系。
:::

实验结果显示，在数学、科学、医学、代码生成、搜索问答五大领域的 9 个基准测试中，RecursiveMAS 相比先进基线平均提升 8.3%的准确率，同时实现 1.2 倍到 2.4 倍的端到端推理加速，减少 34.6%到 75.6%的 token 使用量。

![文章插图](/articles/2026-10-13-recursivemas-75-6-token-2-ea192f57/images/03.webp)

::: image-caption
这张图展示 RecursiveMAS 多智能体系统的缩放规律与不同协作模式实验结果。上半部分 Sequential-Style（轻量串行模式），在 MATH500、AIME2025、GPQA-D、代码生成四项任务中，横轴为训练递归轮次，纵轴是推理递归轮次，随着训练与推理递归轮数增加，模型准确率整体稳步提升，体现递归轮数带来性能增益。下半部分对比三类协作范式：Mixture 混合专家、Deliberation 审议式、Distillation 蒸馏式，在 AIME2026、GPQA-D 等多个评测集上，RecursiveMAS 均优于普通领域专家、Reflector、Tool-Caller 等基线模型；蒸馏模式还标注了 1.4~1.6 倍加速，证明该架构既能提升任务精度，又具备推理加速优势。
:::

这项工作为多智能体系统的规模化提供了全新的技术路径，从底层打破了文本交互的效率天花板。

## 本文目录

- 一、多智能体系统的缩放困境：从文本交互到隐空间革命
- 二、RecursiveMAS 核心架构：把多智能体系统做成一个递归计算环
- 三、RecursiveLink 设计：两层残差投影，打通异构模型的隐空间通道
- 四、内外循环训练范式：让整个系统像单模型一样端到端优化
- 五、理论验证：为什么隐空间递归比文本交互更高效？
- 六、实验结果：横跨 9 大基准的三重性能跃升
- 七、总结与展望：系统级递归的产业意义与未来方向

::: related-title
**相关推荐**
:::

::: related-links
- **[ArchAgent: 智能体驱动的计算机体系结构自动发现，实现 5.3%多核性能突破与 3-5 倍加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898414&idx=1&sn=d58a7e936f83b566db9a0f5d91e6266c&scene=21#wechat_redirect)**

:::

::: related-links
- **[AKG Kernel Agent：多智能体联手，实现跨 DSL 多硬件平台 Kernel 的“一键生成”与迁移](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895857&idx=1&sn=3bce0710d38ccd3a772869ac46a24c83&scene=21#wechat_redirect)**
- **[MLSys 三赛道前三、奥赛四项满分背后大杀器：多智能体编码框架 Humanize 的"裁判工程"（Judgement Engineering）](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906854&idx=1&sn=9ef26460d72547889ed619ce42711323&scene=21#wechat_redirect)**
- **[多智能体 RL 任务较 A100 最高加速 2457 倍：北大 MCHA 用内存中心三层架构，把主存访问占比从 96% 压到 5.4%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906174&idx=1&sn=1c0ca865396936065adf9c8b6a551ada&scene=21#wechat_redirect)**
- **[OpenAI 核心研究员 Noam Brown 深度拆解 Agentic AI、推理与多智能体的技术真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905895&idx=1&sn=6a5d81ee886d6992b05eb8061b077599&scene=21#wechat_redirect)**

:::
