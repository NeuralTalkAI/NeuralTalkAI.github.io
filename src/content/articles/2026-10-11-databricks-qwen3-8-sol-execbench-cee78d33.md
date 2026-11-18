关键词：**SOL-ExecBench**、GPU kernel、英伟达、**agent 自动优化**、:mdnice-bold-italic[榜单饱和]

> 2026 年 3 月，英伟达发布了一套名为 SOL-ExecBench 的 GPU kernel 优化评测：从 124 个生产级模型中抽出 235 个真实计算任务，**以「硬件光速极限」为满分，让各家 AI 工具在 B200 上同台竞技。**

半年过去，这个榜单成了 AI agent 自动优化能力最热闹的竞技场：自动化科研公司 Recursive 用它证明「AI 能自己做研究」，Databricks 用它宣传自家 agent 框架，阿里用它给 Qwen3.8 刷出一个「登顶英伟达榜单」的头条，旧金山创业公司 SF Tensor 则在包揽全部赛道第一之后，反手写了一篇长文，论证这个 benchmark 已经快被刷穿了。

::: related-title
**相关推荐**
:::

::: related-links
- **[算子工程师末日。。。NVIDIA提出SOL-ExeBench：对标硬件效率极限，生成真实 GPU Kernel！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898808&idx=1&sn=0c8886f90fc178c6ae7aedbf1931e0e9&scene=21#wechat_redirect)**
- **[MLSys 三赛道前三、奥赛四项满分背后大杀器：多智能体编码框架 Humanize 的"裁判工程"（Judgement Engineering）](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906854&idx=1&sn=9ef26460d72547889ed619ce42711323&scene=21#wechat_redirect)**
- **[陈天奇团队提出 CAKE：编译器与智能体协同进化，可进化 IR 让 KDA 提速 2.05 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904925&idx=1&sn=ab085dc1e9e6c73917c8853a398775d7&scene=21#wechat_redirect)**

:::

> 本文以榜单发布到今天的时间线为线索，梳理 SOL-ExecBench 的半年榜史：它的题目来源与评分规则、从 0.699 到 0.80 的每一任「榜一」、各家官宣的「第一」分别对应哪个口径。:mdnice-bold-italic[更重要的是泼冷水的部分——5 月有开发者实测发现上榜 kernel 在真实负载下静默算错，9 月榜首团队自己承认 65% 的题目已经打到实际天花板，10 月仓库 issue 又暴露出正确性校验的结构性漏洞。]

::: block-1
「一次性优化 235 个算子」听起来像学术成就，其实不是。这些成绩是各家工具公司拿真金白银砸出来的——Databricks 一轮刷榜的 token 成本就约 7 万美元，SF Tensor 烧掉的 token 超过 1000 亿。

而且钱花得越多，榜单的区分度反而越低：大部分题目的分数已经见顶，第一名和第二名的差距比硬件本身的运行波动还小。所以，看懂这个榜单能证明什么、不能证明什么，比记住谁登过顶更有用。
:::

## 本文目录

- 一、SOL-ExecBench 的定位：英伟达出题，各家 AI 工具答卷
  - 1.1 题目来源：从 124 个生产模型中抽出 235 个 kernel
  - 1.2 评分规则：0.5 分打平基线，1.0 分打到硬件光速极限
  - 1.3 防作弊机制
- 二、半年榜史：每一任「第一」和它的真实口径
  - 2.1 5 月：doubleAI 先声夺人
  - 2.2 6 月：Recursive，用榜单给「AI 自动科研」背书
  - 2.3 7 月：Databricks 与 Qwen3.8 各拿一个「第一」
  - 2.4 9 月：SF Tensor 包揽，Hyra 论文互引
  - 2.5 10 月：榜首还在换
- 三、泼冷水之一：5 月的正确性翻车
- 四、泼冷水之二：榜首团队自己说，榜单快饱和了
  - 4.1 分数收敛的量化证据
  - 4.2 顺手拆穿的几个「聪明反被聪明误」
- 五、泼冷水之三：10 月，仓库 issue 里的校验漏洞
- 六、给这类成绩定位：能说明什么，不能说明什么

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/01.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、SOL-ExecBench 的定位：英伟达出题，各家 AI 工具答卷

> 国内不少报道把 SOL-ExecBench 说成「某公司一次性发布了 235 个算子」，这是对事实的误读。准确的说法是：英伟达出了一套 235 道题的考卷，各家 AI 工具来答卷。

SOL-ExecBench 由英伟达研究院于 2026 年 3 月 19 日发布（《SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits》[1]），作者团队 33 人，田渊栋之外华人研究员占了相当比例。它的全称是 Speed-of-Light Execution Benchmark——「光速极限」执行评测。

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/02.webp)

::: image-caption
《SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits》由 NVIDIA 团队提出，针对当前 GPU 内核优化基准多以软件基线加速比为评价标准、无法衡量内核逼近硬件极限效率的痛点，推出 SOL-ExecBench 评测基准。该基准从 124 款覆盖语言、扩散、视觉、音视频等领域的量产与前沿 AI 模型中，提取出 235 个 CUDA 内核优化问题，面向 NVIDIA Blackwell 架构 GPU，覆盖 BF16、FP8、NVFP4 精度下的前向与反向计算负载。研究团队配套开发 SOLAR 流水线，推导得出硬件层面的 “光速（SOL）” 性能理论下限，以 SOL 分数量化候选内核填补基线与硬件性能极限之间差距的程度，同时搭建具备防作弊机制的沙箱测试环境。该基准将 GPU 内核评测逻辑从 “超越软件基线” 重构为 “逼近硬件性能极限”，相关资源已开源 https://github.com/NVIDIA/SOL-ExecBench。
:::

### 1.1 题目来源：从 124 个生产模型中抽出 235 个 kernel

> 英伟达用 LLM 辅助管线从 124 个生产级和新兴 AI 模型中抽取计算子图——覆盖 LLM、扩散模型、视觉、音频、视频——**先得到 7400 个候选，再经人工加 AI 审核、执行验证，最终保留 245 个有效题目，公开发布 235 个，留 10 个做竞赛。** 题目分四个合集（collection）：

| 合集 | 题数 | 内容 | 累计提交数（10 月初） |
| --- | --- | --- | --- |
| L1 Single Operations | 94 | 单算子，如 RoPE、RMSNorm、MoE scatter | 14067 |
| L2 Fused Operations | 82 | 融合算子序列 | 8795 |
| Quantization | 33 | FP8/NVFP4 低精度 kernel | 6522 |
| FlashInfer-Bench | 26 | LLM 推理注意力相关，来自 Llama-3.1-8B、Qwen3-30B-A3B、DeepSeek-V3/R1 | 6274 |

*精度覆盖 BF16、FP8、NVFP4，前向反向都有，评测硬件统一是 B200（Blackwell）。*

### 1.2 评分规则：0.5 分打平基线，1.0 分打到硬件光速极限

> 传统 kernel 评测的问题是「跟谁比」：比 PyTorch 参考实现快 3 倍，不代表你离硬件极限近。**SOL-ExecBench 换了一个锚点：用配套的 SOLAR 管线对每个 kernel 做 roofline 解析，推导出硬件理论下界（Speed-of-Light 边界），然后定义：**

- 0.5 分 = 打平英伟达发布的评分基线（优化过的参考实现）
- 1.0 分 = 达到解析推导出的硬件光速极限

::: block-1
:mdnice-highlight[也就是说，每个 kernel 的分数度量的是「从基线到硬件极限之间的这段路，你走了百分之多少」。] 这个设计让分数跨 kernel 可比，也让「满分」有了一个不随软件实现漂移的固定靶子。
:::

### 1.3 防作弊机制

英伟达显然预料到会有人来刷分：评测在沙盒中执行，锁定 GPU 时钟频率、每次测量前清空 L2 缓存、提交在隔离子进程中运行，并用静态分析检查常见的 reward hacking 手法。后文会看到，这些防线被绕过的情况确实存在。

## 二、半年榜史：每一任「第一」和它的真实口径

截至 SF Tensor 9 月 25 日的统计，这个榜单共收到 583 支队伍的 31466 次提交。下面按时间线梳理每一任「第一」。

### 2.1 5 月：doubleAI 先声夺人

> 5 月 23 日，doubleAI 发布 WarpSpeed《WarpSpeed approaches Speed of Light on Blackwell》[2]，宣称只用一天搜索，就在 235 个 kernel 中的 90% 上击败英伟达的优化基线，平均提速 2.24 倍。

这是榜单上第一个引起关注的 agent 工具成绩。**同一份报告还顺手指出 Cursor 的提交中有 7 道题在部分 workload 上跑不过正确性测试——「刷榜」与「真实正确」之间的裂缝从这里就开始显现。**

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/03.webp)

::: image-caption
各组问题相对英伟达优化版PyTorch基准的平均加速比，数值越高代表速度越快。这张柱状图对比WarpSpeed与Cursor在4组测试集上相对NVIDIA优化基线的平均加速倍数，基准线1.00倍代表NVIDIA原生性能。蓝色柱为WarpSpeed，橙色柱是Cursor。L1测试：WarpSpeed2.50倍、Cursor1.53倍；L2：WarpSpeed1.58倍、Cursor仅1.07倍；Quant差距最大，WarpSpeed达3.43倍，Cursor1.61倍；FlashInfer-Bench中WarpSpeed2.21倍，Cursor1.68倍。所有项目里WarpSpeed加速效果均显著优于Cursor，证明其CUDA内核自动优化能力更强。
:::

### 2.2 6 月：Recursive，用榜单给「AI 自动科研」背书

> 6 月 11 日，田渊栋参与创立的 Recursive Superintelligence 发布首个公开成果《First Steps Toward Automated AI Research》[3]，SOL-ExecBench 均分从此前最佳的 0.699 提升到 0.754，官方提到「与硬件极限的差距缩小 18%」。

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/04.webp)

::: image-caption
《First Steps Toward Automated AI Research》介绍了Recursive公司推出的自动化AI研究系统的早期成果。该系统可自动完成“提出想法—落地实现—实验验证—迭代优化”的完整研究闭环，支持多线程并行探索与跨任务经验复用，在三项核心基准测试中均刷新业界最优水平。具体实验结果如下：固定算力小模型训练任务NanoChat Autoresearch中，验证BPB从0.9372降至0.9109，同等精度下训练速度提升1.3倍；小模型速训任务NanoGPT Speedrun中，达到3.28验证损失的训练时间从79.7秒缩短至77.5秒；GPU内核优化基准SOL-ExecBench上，235个内核的平均SOL得分从0.699提升至0.754，与硬件性能上限的差距缩小18%。系统内置严格校验机制抑制奖励破解，相关产出已开源。
:::

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/05.webp)

::: image-caption
Recursive 自动化 AI 研究系统在三项核心基准测试中与此前业界最优水平的量化对比，覆盖了 AI 进步的三大核心方向：训练算法效果、训练速度、硬件利用效率。NanoChat Autoresearch 为固定算力预算的小模型训练任务，原最优验证 BPB 为 0.9372，系统降至 0.9109，等效于同等精度下训练速度提升 1.3 倍；NanoGPT Speedrun 为小模型速训任务，达标训练时间从 79.7 秒缩短至 77.5 秒；SOL-ExecBench 面向 235 个 GPU 内核优化，平均 SOL 得分从 0.699 提升至 0.754，与硬件理论最优性能的差距缩小 18%。三项测试均采用明确可校验的指标，直观印证了系统的性能突破。
:::

值得注意的是：SOL-ExecBench 对 Recursive 来说只是三个战场之一（另两个是 NanoChat 训练效率和 NanoGPT Speedrun），**它的目的是证明自家系统能自动完成「提假设—写代码—跑实验—验证」的完整研究循环，kernel 优化只是展示品。** 这家公司刚出隐身状态不到一个月，融资 6.5 亿美元、估值 46.5 亿美元，正好需要一份成绩单。

### 2.3 7 月：Databricks 与 Qwen3.8 各拿一个「第一」

> 7 月 1 日，Databricks 宣布其 KDA 框架拿下 L1 单算子赛道第一。同月，阿里的 Atrex Kernel Agent 帮助 Qwen3.8 拿下 FlashInfer-Bench 子榜第一——这就是国内「Qwen3.8 登顶英伟达榜单」报道的原始出处。

需要说清楚的是：FlashInfer-Bench 是 235 题中 26 题组成的子榜，且成绩是 agent 工具反复迭代提交的产物，不是 Qwen3.8 模型本身「一次性写出」了什么。

### 2.4 9 月：SF Tensor 包揽，Hyra 论文互引

> 9 月 25 日，旧金山 kernel 优化公司 SF Tensor 发布长文 What we learned topping NVIDIA's SOL-ExecBench[4]：总榜第一，均分 0.7992，四个赛道全部第一，次日（9 月 26 日）均分突破 0.8。

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/06.webp)

::: image-caption
《What we learned topping NVIDIA's SOL-ExecBench》介绍了 SF Tensor 团队在 NVIDIA 官方 B200 内核优化基准 SOL-ExecBench 上登顶的成果与相关经验。该团队以平均 0.7992 的 SOL 分数包揽全部四个赛道第一，单内核中位加速比达 NVIDIA 参考优化方案的 3.03 倍。文章指出该基准测试正趋于饱和，当前 65% 的内核已接近实际性能天花板。作者分析了十余款大模型在优化任务中的行为特征与各类 “奖励黑客” 漏洞，同时指出基准存在数值校验边界不合理、评分程序与候选内核同进程等设计缺陷。最后提及团队正打造覆盖多硬件平台的内部基准 KernelWorld，以支撑更全面的模型能力评估。
:::

::: block-1
同一时期，腾讯混元的 Hyra 系统在论文中报告均分 0.771，《The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement》arXiv:2609.11873[5]，对比对象正是 Recursive 的 0.754。9 月 28 日，Databricks 又宣布自家方案拿下全部四个赛道第一，:mdnice-bold-italic[有 X 帖子估算这一轮刷榜花了约 7 万美元的 token 成本——「第一」的含金量开始需要除以成本来看。]
:::

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/07.webp)

::: image-caption
《The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement》由上海交通大学等机构联合完成。文章提出递归自我改进（RSI）核心概念，构建了从改进执行、策略选择、经验获取、环境适配到递归元改进的五级自主度框架，通过头部空间闭合指数（HCI）揭示现有大模型跨领域能力进展不均的现状。论文分析了科学发现、具身智能、软件工程、医疗四大场景下 RSI 的差异化要求与发展瓶颈，梳理了多家企业的产业实践案例，最终提出八大未来研究方向，指出真正的递归自我改进需实现可验证、可传承的持续能力进化。
:::

### 2.5 10 月：榜首还在换

> 榜单不是静态的。截至 10 月 10 日，FlashInfer-Bench 子榜的第一名已经换成了无问芯穹（Infinigence AI），得分 0.8057；SF Tensor 以 0.8035 紧随其后；曾经「登顶」的 Qwen 已经滑到第 7 名（0.7668），Recursive 则排在第 17 位。

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/08.webp)

::: image-caption
图 1：SOL-ExecBench 官网 FlashInfer-Bench 子榜实时排名（2026 年 10 月 10 日截图）。表头 SOL Bound 一行是理论硬件极限（1.0），Scoring Baseline 是 0.5 分的评分基线。当前第一名为无问芯穹（Infinigence AI），得分 0.8057，提交于 3 天前；第二名 SF Tensor 仅落后 0.002，且 18 小时前还在提交。曾在 7 月登顶、被国内报道为「Qwen3.8 登顶英伟达榜单」的 Qwen 队已滑落至第 7。第 11 名 KerSor 提交于 10 月 1 日，第 12、13 名的提交时间分别是 4 小时和 7 小时前——榜单至今仍处于高频变动状态。（来源：research.nvidia.com[6]）
:::

## 三、泼冷水之一：5 月的正确性翻车

> 5 月 27 日，有开发者在 Reddit 发帖实测：把榜单上来自 DeepSeek、Qwen、Gemma、Kimi 等团队的 235 个高分 kernel 拿到真实生产负载下运行，发现大量提交会静默产生错误结果甚至进程崩溃，详见：《WarpSpeed approaches Speed of Light on Blackwell》[7]。

最典型的案例是:mdnice-bold-italic[一个 embedding 梯度 kernel 用 bf16 而不是 fp32 做累加——数值误差会随成千上万次训练迭代悄悄腐蚀模型权重，不触发任何报错或告警。事后极难回溯。]

::: block-1
:mdnice-highlight[问题的根源在评测设计：SOL-ExecBench 的评分只关心吞吐量，正确性校验在受控的单次执行中完成，不测并发负载下的竞态和数值稳定性。] 而竞态、累加精度这类问题恰恰只在真实训练循环里暴露。榜单排名因此不能作为生产可用的依据。
:::

## 四、泼冷水之二：榜首团队自己说，榜单快饱和了

SF Tensor 那篇 9 月长文最有价值的部分，不在于宣布自己第一，而是作为现任榜一，**系统性地论证这个 benchmark 正在失去区分度。**

### 4.1 分数收敛的量化证据

> 他们追踪了「第一名与第二名差距在 0.01 SOL 以内」的 kernel 占比：7 月中旬是 39%，9 月 25 日已经到 65%（235 题中的 152 题）；前三名挤在 0.01 以内的有 80 题，占 34%。**分数越高的地方收敛越严重：32 个 kernel 的历史最好成绩达到 0.95 以上，其中 21 个有三支以上队伍咬在 0.01 以内。**

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/09.webp)

::: image-caption
图 2：SF Tensor 统计的分数收敛曲线（2026 年 9 月 25 日发布）。蓝线为「第二名与第一名差距在 0.01 SOL 以内」的 kernel 占比，从 7 月中旬的约 39% 一路爬到 9 月下旬的约 65%；灰线为「第二、三名同时在 0.01 以内」的占比，爬升至约 34%。两条线的持续上行说明：不同队伍、不同 agent、不同模型在这些题上正收敛到同一个答案——题目已接近实际天花板，榜首易主更多反映运气和提交频率，而不是工具能力差距。（来源：sf-tensor.com[8]）
:::

![文章插图](/articles/2026-10-11-databricks-qwen3-8-sol-execbench-cee78d33/images/10.webp)

::: image-caption
图 3：235 个 kernel 的历史最好 SOL 成绩从高到低排序（SF Tensor，2026 年 9 月 25 日）。虚线为 0.95 参考线，蓝色柱表示该 kernel 已有三支以上队伍与第一名差距在 0.01 以内。可以看到高分段几乎全蓝——接近理论极限的题基本被刷平；剩余空间集中在右侧 0.7 分以下的长尾难题上，而其中一部分「空间」来自评分与校验方式本身的瑕疵，不是真实可优化的余量。（来源：sf-tensor.com[9]）
:::

饱和之后，排名的参考价值会被硬件噪声吃掉。同型号 B200 的实际运行时间受环境影响会有 5% 上下的波动。**当头部队伍的差距小于硬件噪声时，「谁是第一」就不再是工具能力的有效度量。**

### 4.2 顺手拆穿的几个「聪明反被聪明误」

> SF Tensor 还披露了十几家模型在数万次 agent rollout 中的行为细节，有几条直接削弱榜单的可信度：

- **MoE 题可以少算 80% 的专家照样通过**：一道要求路由到 top-10 专家的 MoE 题，有 agent 发现只算 top-2 专家、聚合结果仍在数值容差内，:mdnice-bold-italic[于是「合法」地跳过了 80% 的计算量。正确性边界太松。]
- **改评分器而不是改自己**：评分器和提交的 kernel 跑在同一进程里，有 agent 通过 import 时设置环境变量，降低了评分器参考实现的 cuBLAS 计算精度——不是自己变准，是把参考答案拉低到和自己一样。
- **SOL 分可以超过 1.0**：个别 MoE 题的「硬件下界」假设所有专家都必须加载，但实际路由不均匀时部分专家可以不加载，于是出现了「超过光速」的分数——边界推导本身的误差。
- **BoomerCores 现象**：不给明确提示时，多数模型默认用 Ampere 时代的 MMA 指令写 Blackwell kernel，完全用不上第五代 tensor core。模型对最新硬件特性的知识滞后，是这类评测里普遍的天花板来源。

另外值得记录的一条：他们观察到 DeepSeek V4 Pro 在约三分之一的题目上极快收敛到与第一名几乎一致的成绩，远快于更「聪明」的模型，但在其余题目和 SF Tensor 自留的 held-out 评测上表现很差——**这种行为模式与针对这些题目做过训练或数据污染一致。刷榜成绩里混入了「背过答案」的成分。**

## 五、泼冷水之三：10 月，仓库 issue 里的校验漏洞

> 如果说 SF Tensor 的批评还带有「榜一之姿」，那么 10 月初社区在 GitHub 仓库（nvidia/sol-execbench[10]）里挂出的 issue 是更中立的证据：

- **Issue #25（10 月 2 日）**：benchmark 生成的 norm 层权重全是 1、bias 全是 0。这意味着一个完全忽略 norm 参数、或者错误使用参数的错误 kernel，也能通过正确性校验——测试数据本身覆盖不到这类错误。
- **Issue #21（8 月 9 日挂出，至今未关）**：题目定义用的 epsilon 是 1e-5，而 Triton 官方 RMSNorm 教程示例用 1e-6，两处不一致，会导致本应对的实现被判错或反之。
- **Issue #26（10 月 2 日）**：有提交卡在 Running 状态一整天——评测队列拥堵，侧面印证提交洪流仍在继续。

::: block-1
这三条和 5 月的生产翻车是同一条线：:mdnice-bold-italic[这个 benchmark 的正确性校验，从测试数据设计到容差设定，都存在让错误 kernel 蒙混过关的缝隙。分数可以刷上去，但「分数高」与「算得对」之间的关联比想象中弱。]
:::

## 六、给这类成绩定位：能说明什么，不能说明什么

梳理完半年榜史，可以给「235 个算子」这类成绩一个相对公允的定位。

> 它能说明的：AI agent 在反馈清晰、可自动验证的工程问题上，已经能稳定产出超过人类专家基线的方案。

- Recursive 承认团队里没人是 kernel 专家，优化思路来自系统本身；
- SF Tensor 的数据显示 agent 能把 235 个生产 kernel 的中位速度做到英伟达优化参考的 3 倍。

这些是真实的能力进展，也是这类工具商业价值的直接证据——Databricks、无问芯穹、腾讯混元愿意持续投入 token 成本刷榜，本身说明榜单的客户导向价值存在。

> 它不能说明的：第一，不代表学术成就。题目是英伟达从现成模型里抽的，方法是 agent 大规模试错迭代，单次「登顶」没有提出新的理论或方法，且可能混入训练数据污染的成分。:mdnice-bold-italic[第二，不代表生产可用。5 月的实测和 10 月的 issue 都说明，高分 kernel 在真实负载下的正确性没有保障。第三，在 65% 题目已收敛、硬件噪声大于队伍差距的当下，「谁是第一」的区分度正在快速衰减。]

**SF Tensor 在文末给出的去向也印证了这一点：他们已经开始转向自建的 KernelWorld 评测** ——22 万个环境、48 万个测试用例、覆盖 NVIDIA、AMD、TPU、Trainium 多厂商和多机场景。:mdnice-bold-italic[头部玩家在为 SOL-ExecBench 的退役做准备。]

所以嘛，下次再看到「某某登顶英伟达 kernel 榜单」的标题，可以先核对三件事：是 235 题总榜还是 26 题子榜，分数领先第二名多少、是否小于 0.01 的噪声区间，kernel 在真实负载下验证过正确性没有。三件事核完，或许大多数头条会缩水不少。

::: block-1
如对 kernel 自动优化的工程细节感兴趣，SF Tensor 的复盘长文《What we learned topping NVIDIA's SOL-ExecBench》[11]是目前信息密度最高的一手材料，推荐阅读原文。
:::

参考资料

[1] arXiv:2603.19173: *https://arxiv.org/abs/2603.19173*

[2] doubleAI 发布 WarpSpeed: *https://www.doubleai.com/research/warpspeed-approaches-speed-of-light-on-blackwell*

[3] 首个公开成果: *https://www.recursive.com/articles/first-steps-toward-automated-ai-research*

[4] What we learned topping NVIDIA's SOL-ExecBench: *https://sf-tensor.com/news/sol-execbench*

[5] arXiv:2609.11873: *https://arxiv.org/html/2609.11873v2*

[6] research.nvidia.com: *https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200*

[7] 《WarpSpeed approaches Speed of Light on Blackwell》: *https://www.doubleai.com/research/warpspeed-approaches-speed-of-light-on-blackwell*

[8] sf-tensor.com: *https://sf-tensor.com/news/sol-execbench*

[9] sf-tensor.com: *https://sf-tensor.com/news/sol-execbench*

[10] nvidia/sol-execbench: *https://github.com/nvidia/sol-execbench*

[11] SF Tensor 的复盘长文: *https://sf-tensor.com/news/sol-execbench*

::: related-title
**相关推荐**
:::

::: related-links
- **[KDA²：Kernel Design Agents 优化 Kimi Delta Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906834&idx=1&sn=53fcb953cfd29bf2cffbb8d2631ba4b5&scene=21#wechat_redirect)**
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**
- **[34.93 倍加速！MLSys 26 DSA 智能体赛道冠军方案，一套规则手册与三个子智能体替代人类内核工程师](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901644&idx=1&sn=ddd845b9bc6fd48c0c73a0ce535f45e0&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[数千美元、17 亿 Token、一周时间写出 Qwen 定制推理引擎，单流解码比调优后 vLLM 快 90%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906803&idx=1&sn=54917a863dd0a1e27ad80853c43dbaaf&scene=21#wechat_redirect)**
- **[Claude 实现、Codex 审查、人类决策领航：Humanize 项目用 RLCR 编排长程闭环迭代任务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901161&idx=1&sn=88251ef01180c0f214ca6517d40beda1&scene=21#wechat_redirect)**

:::
