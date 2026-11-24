**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**RoofLang**、:mdnice-bold-italic[推理架构搜索]、Roofline建模、**AI Agent**、MoE推理

> 被 profiling 绑定的反馈，会把搜索困死在现有软件栈的能力与性能边界内，让那些本质上更优的架构永远没有机会被发现。

过去一年，KernelBench、KDA-Pilot、KernelPro 等“AI 写 kernel”的工作把 Agent 推上算子优化一线：给 profile、给 benchmark、给编译器报错，就能反复迭代出媲美手写的 CUDA kernel。:mdnice-bold-italic[但这类闭环本质上是“profiling-bound”，反馈来自真实软件栈，Agent 能改的只有软件栈允许它改的东西。一个理论上更优的架构，如果当前框架不支持、通信库缺算子、kernel 没优化好，Agent 连它的存在都探测不到。]

::: block-1
核心问题由此清晰：:mdnice-highlight[如何让 AI 【越过具体软件栈】，在“推理系统架构”层自主完成设计、评估与迭代？]
:::

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/01.jpg)

- **RoofLang: Enabling AI-Driven Architecting of LLM Inference Systems**
- 组织机构：上海 Xingyunzhili 人工智能研究院；微软研究院
- 论文链接：https://arxiv.org/abs/2609.12551
- 开源代码：github.com/yzygitzh/rooflang
- 项目主页：yzygitzh.github.io/rooflang

::: related-title
**相关推荐**
:::

::: related-links
- **[MINISA：面向下一代可重构推理加速器的最小指令集架构，把 96.9% 指令停顿打到接近零](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901107&idx=1&sn=2b04a2a2493ac8107f13e7467b430cff&scene=21#wechat_redirect)**
- **[OpenAI 核心研究员 Noam Brown 深度拆解 Agentic AI、推理与多智能体的技术真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905895&idx=1&sn=6a5d81ee886d6992b05eb8061b077599&scene=21#wechat_redirect)**
- **[ArchAgent: 智能体驱动的计算机体系结构自动发现，实现 5.3%多核性能突破与 3-5 倍加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898414&idx=1&sn=d58a7e936f83b566db9a0f5d91e6266c&scene=21#wechat_redirect)**

:::

> 答案是 RoofLang，**一个面向 LLM 推理系统的 DSL，三要素为：与框架无关的计算图通用负载表示、一组把“架构动作”约束成数学可验证等价变换的语义保持（semantics-preserving）图改写原语、一个基于 Roofline 的离散事件仿真器**，可不跑任何真实 kernel 就评估架构好坏，:mdnice-bold-italic[三者拼成 AI 可访问的自主架构设计闭环。]

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/02.jpg)

::: image-caption
图 2：RoofLang 的整体系统架构图，完整展示这套面向大模型推理系统设计的领域专用语言工作流程。该图从输入计算图开始，依次呈现语义保序变换、硬件资源映射放置，结合硬件图执行仿真，最终输出追踪日志的全链路。它直观说明 RoofLang 不依赖真实硬件剖面，支持 AI 驱动的推理架构搜索，是整篇论文的核心框架可视化，帮助读者理解如何脱离现有软件栈约束，探索全新 LLM 推理系统方案。
:::

在 H200、GH200、B300、GB300 四代硬件，DeepSeek V4 Flash、DeepSeek V4 Pro、GLM-5.3、Kimi K3 四个 2026 年旗舰 MoE 模型，**64K 与 1M 上下文的大规模扫描中，作者发现 DeepSeek V4 系列的峰值解码吞吐可达其他模型的 3.5–39.5 倍。**

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/03.jpg)

::: image-caption
表 7：RoofLang 搜索得到的最优推理架构在 64 张 GB300 显卡、64K 上下文条件下，解码阶段内存开销不同模型对比表。表格量化对比各模型 KV‑Cache 占用、最大批处理规模、单 token 内存流量。实验数据证明 DeepSeek V4 系列凭借紧凑 KV‑Cache 设计，实现远超其他竞品的解码吞吐，解释了性能优势并非单纯源于参数量，为论文核心结论提供关键定量支撑，也是评估不同模型推理内存压力的核心实验结果。
:::

随后他们让一个基于 Claude Code、以 DeepSeek V4 Pro 为主模型的优化 Agent 在 RoofLang 中自主探索，**Agent 发现三种人工基线没有采用的新架构，在 NVIDIA B300 上把 DeepSeek V4 Pro 的吞吐与交互性同时提升 6.23%–50.1%。**

架构层面的红利不仅真实存在，而且已经大到足以被 AI 自动挖掘。

## 本文目录

- 一、Profiling 闭环的天花板：AI 优化需要一张“架构图纸”
- 二、三要素拼成的闭环：RoofLang 如何让 AI“看懂并改写”架构
- 三、两张图与一套原语：Agent 的“世界模型”和“动作集”
  - 3.1 计算图：只描述计算事实，不承诺任何框架
  - 3.2 硬件图：把机房拓扑写成一张图
  - 3.3 放置原语：把计算和数据钉到硬件上
  - 3.4 语义保持变换：架构创新的“合法动作表”
- 四、不跑一行 kernel 怎么判快慢：Roofline 离散事件仿真器
  - 4.1 算力屋顶
  - 4.2 带宽屋顶
  - 4.3 网络
  - 4.4 RoofLang 骨架
- 五、四大模型 × 四代硬件：仿真扫描挖出的模型级架构洞察
  - 5.1 实验设置：把 2026 年旗舰 MoE 放进同一把标尺
  - 5.2 Pareto 前沿：吞吐与交互性的跷跷板
  - 5.3 39.5 倍解码差距：公式比参数更能解释胜负
- 六、Agent 自主架构师：Claude Code 在 B300 上找到的三种新架构
  - 6.1 发现1：成本与头尾感知的连续 PP 划分
  - 6.2 发现2：DP-CP 路由专家切分
  - 6.3 发现3：本地专家复制
- 定位、边界与启示：架构层与实现层的显式分工

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/04.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、Profiling 闭环的天花板：AI 优化需要一张“架构图纸”

> 今天绝大多数“AI 做系统优化”的工作，都在回答同一个问题：给定这套软件栈，怎么把手上的 kernel 调得更快？**RoofLang 想问的则是另一个问题：如果软件栈本身可以重画，什么架构才是最快的？**

这两个问题的差别，是工匠与建筑师的差别。

当下的 kernel agent 工作流高度一致：拿 profiler 轨迹和 benchmark 分数，改 CUDA 代码，编译、跑分、再迭代，SGLang、KDA-Pilot、KernelPro 莫不如此。这类闭环具体可靠，但**搜索空间从第一天起就被现有 kernel、编译器、框架、硬件四重耦合锁死。当某个架构所需特性在当前栈中缺失、或对应实现没被优化好，它就成了 Agent 视野之外的“不可达区域”**，:mdnice-bold-italic[哪怕理论上本来更优，也连被比较的机会都没有]。

作者把困境归因于缺三样东西：

- 一是不绑定任何框架的通用负载表示，让计算结构可被独立描述；
- 二是可验证的“变异空间”，让 Agent 的每一步修改都能被证明语义正确，不必跑出错误结果再靠人排查；
- 三是与实现解耦的评估器，在没有 kernel、框架甚至真机时也能判出两个架构谁快、瓶颈在哪。

::: block-1
RoofLang 正为补齐这三块拼图而生，它与 kernel agent 分工：:mdnice-highlight[先依据硬件规格与 Roofline 关系，在“实现质量污染对比”之前搜索图组织、算子放置、切分与通信拓扑，筛出内在占优的候选]；再交框架层工具做具体实现。
:::

## 二、三要素拼成的闭环：RoofLang 如何让 AI“看懂并改写”架构

> RoofLang 的名字里，Roof 指向 Roofline 仿真评估，Lang 指向它作为 DSL 的图表达与图改写能力。名字的两半恰好对应架构闭环的两端：一端是可被形式化操作的架构描述，一端是可被快速计算的架构评分。下图 1 是这个闭环全貌：

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/05.webp)

::: image-caption
图 1：RoofLang 支撑的 AI 驱动自主架构设计闭环：① 以 RoofLang 提供经过验证的 LLM 推理负载与初始架构；② AI 生成新架构；③ 用 RoofLang Simulation 评估新架构，结果回流至②。闭环的关键在于②只能在“可验证动作空间”内行动、③给出的是与实现无关的架构级反馈，这使得 Agent 的每一步探索既不会破坏计算语义，又不会被软件栈的性能噪声带偏。
:::

- 第一步，人（或官方基线实现）用 RoofLang 描述经验证的 LLM 推理负载与初始架构；
- 第二步，AI 在合法动作空间内生成新架构；
- 第三步，仿真器评估并产出 trace 回流给 AI，支撑下一轮改写。循环没有预设终止点，由 Agent 自行决定何时收敛。

> 图 2 进一步展开了 DSL 内部的数据流。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/06.webp)

::: image-caption
图 2：RoofLang DSL 总览：计算图经语义保持变换、放置、再变换后，与硬件图共同进入仿真，输出仿真 trace。注意放置与变换可以交替进行——先放置再改写，等价于先决定“放在哪”再决定“怎么切”，这让 PD 分离、KV 分级存储、各类并行切分等真实架构策略都能被原语组合显式编码。
:::

外部供给的计算图（Compute Graph）先经语义保持变换改写；放置（Placement）原语把算子分到计算设备、把张量分到存储设备，给图打上硬件标签；之后可继续变换以显式刻画通信实现；最终带标签的计算图与硬件图（Hardware Graph）一起送入仿真，产出仿真 trace。

理论上没有新的概念，计算图表示在 TensorFlow 等框架中早已普及，Unity 等验证式图优化器也证明了图代换能在改变并行方式的同时保持语义，Roofline 则提供经典的解析性能边界语言。RoofLang 首次把三者组装成 AI 可访问的闭环：**图是“世界模型”，原语是“动作集”，仿真 trace 是“奖励信号”，强化学习式的探索范式由此被搬进系统架构设计。**

::: block-1
整体看下来，:mdnice-highlight[RoofLang 本质上是把 LLM 推理架构设计变成了一个可形式化、可验证、可快速评估的搜索问题]。通用表示解决“看得懂”，可验证变异解决“改不错”，独立评估器解决“判得准”，三者缺一不可。
:::

## 三、两张图与一套原语：Agent 的“世界模型”和“动作集”

> 有了闭环，还需要一套具体的“语言”把它写下来。RoofLang 的语法对象只有两类图，外加一小组把图改写成等价图的原语。

下面将逐个拆解这些构件：**计算图描述负载，硬件图描述平台，放置原语与语义保持变换原语则是 Agent 手中全部的合法动作。**

### 3.1 计算图：只描述计算事实，不承诺任何框架

> 计算图是 RoofLang 的输入，也是整个闭环的事实基座。

每个节点代表一个操作，如 attention、GEMM，显式声明输入输出张量及形状、数据类型、FLOPs、数据搬运量等解析属性。节点分两类：无副作用节点只能通过输出被观测，有副作用节点可能修改外部状态；每条边表示算子间的数据依赖。

论文强调，计算图由外部提供，**提供者必须保证它能忠实、正确地表示负载**，这是所有“语义保持”承诺成立的前提，也是局限性中坦承仍需人工核验的环节。

### 3.2 硬件图：把机房拓扑写成一张图

> 计算图描述“要算什么”，硬件图描述“能在什么上算”。

硬件图的节点分两类：

- 计算设备（CPU、加速器、NIC、交换机）带各精度峰值 FLOPs；
- 内存设备（DRAM、HBM、SSD）带明确容量。

边是连接设备的 fabric（PCIe、以太网），带带宽与延迟。

有了这张图，**“NVLink 域多大、跨节点走 InfiniBand、HBM 容量多少”都成为仿真器可直接读取的硬约束**，架构决策就有了和计算图对称的硬件侧形式化描述。

### 3.3 放置原语：把计算和数据钉到硬件上

> 放置（placement）回答“算子在哪跑、张量在哪存”，把计算图落到硬件图上。

放置由两个原语完成，如表 1。放置本身也是语义保证的：把算子换到另一块设备、把张量换个地方存，计算结果不变，改变的只有性能。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/07.webp)

::: image-caption
表 1：放置原语：operator.set_compute 将算子分配到计算设备，并可指定其执行所在的 stream/queue；tensor.set_memory 指定张量存储的内存设备。虽然只有两个原语，却足以表达设备映射与数据存放位置的全部决策，例如 PD 分离就是用这两个原语把 prefill 与 decode 的计算、以及两者之间交换的状态分别放到不同设备上。
:::

### 3.4 语义保持变换：架构创新的“合法动作表”

> 如果说放置原语决定“东西放哪”，变换原语决定“计算以什么结构存在”。表 2 列出了全部语义保持变换原语。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/08.webp)

::: image-caption
表 2：语义保持变换原语：增删控制边、增删 identity 算子、fuse_kernels 子图融合、split_kernel 按 m/k/n 维切分 GEMM（分别对应 scatter–GEMMs–gather、scatter–GEMMs–reduce、broadcast–GEMMs–gather）、dup/dedup_kernel 的算子复制与去重，以及四类 gather/reduce 与 scatter/broadcast 通信子图的合并。这张表就是 Agent 的“合法动作表”：split_kernel 的三种切分正好对应权重固定、按 token 切、按输出切三类并行；dup_kernel 对应算子复制带来的广播消除；通信合并原语把朴素的 gather-scatter 重写成 all-gather、reduce-scatter、all-reduce 等更高效的集合通信——人工架构师的经验被沉淀成一条条带前置条件的数学规则。
:::

**表中每个原语都是一次预定义的、结构化受限的图重写，其前置条件保证改写后模型的可观测行为不变**，换句话说，Agent 可以随意组合这些动作，却从数学上不可能把模型“算错”。

这些原语组合后可显式编码三类主流架构策略。

| 架构策略 | 描述 |
| --- | --- |
| PD 分离 | 用两个放置原语把 prefill、decode 分到不同设备，并显式放置交换的状态。 |
| KV cache 生命周期 | dup/dedup_kernel 控制重算与缓存粒度，insert/remove_identity 充当存储层级边界，表达 ToNvme、ToDRAM、ToHbm 分级。 |
| 切分 | split_kernel 配合放置与通信合并，还原 CP、DP、EP、PP 等典型并行策略。 |

人类架构师做切分，靠的是“这样切数学等价、通信划算”的隐性经验；**RoofLang 把它外化成两部分：等价性由重写前置条件强制保证，划算与否交给仿真 trace 量化回答**。:mdnice-bold-italic[Agent 不必记住并行策略大全，只需在合法动作空间探索、让评估结果筛选策略，这正是后文三个自主发现的机制基础。]

::: block-1
回顾一下，RoofLang 的动作空间刻意做得又小又硬：放置只有 2 个原语，变换只有十余条带前置条件的重写规则。:mdnice-highlight[小到 Agent 可以枚举组合，硬到每一步都不可能算错，架构创新由此从“灵光一现”变成“可搜索的有限动作序列”。]
:::

## 四、不跑一行 kernel 怎么判快慢：Roofline 离散事件仿真器

> 评估器是整个闭环里最反直觉的部分😅：没有 kernel 执行、没有框架开销，只有解析属性、硬件规格和一台事件驱动的“时钟”。

仿真器对两张已标记的图执行拓扑感知的离散事件仿真（DES）：

- 每个 kernel 仅在其所有前驱节点按拓扑序完成、且所需资源就绪后才被释放；
- 其开始与结束事件按资源优先级入队。资源占用具有持续性，实际执行时间由事件调度动态决定。张量内存随算子的进入与退出事件动态增减；
- 仿真器据此实时追踪各类内存的峰值存活字节数，一旦超出容量即触发 OOM 报警。“最大可支持批量大小”由此自然导出，无需人工预设或经验假设。

### 4.1 算力屋顶

> 对放置在设备 $d$ 上的算子 $K$，RoofLang 分别建模计算、本地内存、网络三类成本。计算成本对算子涉及的各种数据类型 $q$ 求和，$F$ 表示该精度下的浮点运算量，$P$ 表示设备在该精度的峰值算力：

$$
T_{\text{compute}}(K,d)=\sum_{q}\frac{F_{K,q}}{P_{d,q}} \tag{1}
$$

:mdnice-bold-italic[这就是最朴素的“算力屋顶”，GEMM 哪怕零搬运，也快不过 FLOPs 除以峰值 FLOP/s。以 B300 为例，其 FP4 算力 13,500 TFLOPS，$1.35×10^{16}$ 次运算的耗时下界就是 1 毫秒。]

### 4.2 带宽屋顶

> 本地内存成本覆盖算子从内存层级读取和写回的字节，$R_K$、$W_K$ 分别是读集合与写集合，$B$ 为字节数，$\beta$ 为对应层级带宽，读字节中还按建模比例区分输入、权重和输出：

$$
T_{\text{memory}}(K,d)=\sum_{l\in R_K}\frac{B_{K,l}^{\text{read}}}{\beta_{d,l}^{\text{read}}}+\sum_{t\in W_K}\frac{B_{K,t}^{\text{write}}}{\beta_{d,t}^{\text{write}}} \tag{2}
$$

:mdnice-bold-italic[这是“带宽屋顶”：解码每 token 计算极少，却要读取本批次的激活参数与 KV，耗时由 HBM 带宽决定。B300 的 HBM 带宽 7,750 GB/s，读取 7,750 GB 数据的访存下界就是约 1 秒。]

### 4.3 网络

> 网络成本含远端张量访问与集合通信，按硬件图有向 fabric 路由，采用经典 $\alpha + \beta$ 模型：$\alpha$ 为链路延迟，远端访问取各链路 $\frac{B}{\beta}$ 的最大值，集合通信用通信量 $V$ 除以瓶颈有效带宽 $B_{eff}$：

$$
T_{\text{network}}(K)=\alpha_K+\begin{cases}\max\limits_{l\in R_K}\dfrac{B_{K,l}}{\beta_l},&\text{远端张量访问}\\[6pt]\dfrac{V_K}{B_{\text{eff},K}},&\text{集合通信}\end{cases} \tag{3}
$$

其中 $\alpha$ 取参与 rank 间最大路径延迟（传播图加权直径）；$\beta$ 是按拓扑估计的聚合带宽，会因 fabric 争用折损，多算子共享资源时仿真时间按容量加权比例延展。

### 4.4 RoofLang 骨架

> 最后，RoofLang 假设计算、本地访存、网络三者理想重叠，算子耗时取三者最大值：

$$
T_K=\max\{T_{\text{compute}},T_{\text{memory}},T_{\text{network}}\} \tag{4}
$$

“取最大值”正是 Roofline 的骨架：**真实瓶颈由三类屋顶中最低的一块决定，架构优化的本质就是转移瓶颈**。:mdnice-bold-italic[仿真还假设计算与通信细粒度完美重叠、MoE 路由 token 理想均衡，这些明确写出的前提意味着仿真给的是“理论资源极限下的内在效率”]。

::: block-1
总之，这台仿真器要回答的是“这个架构在给定硬件定律下最多能跑多快、瓶颈卡在谁身上”，而非“这套代码实际跑多快”。:mdnice-highlight[它牺牲实现保真度，换来毫秒级、可复现、与代码质量无关的架构级判分]，恰好匹配架构搜索阶段的决策需求。
:::

## 五、四大模型 × 四代硬件：仿真扫描挖出的模型级架构洞察

> 仿真器到底准不准、能挖出什么？作者用四个 2026 年旗舰 MoE 模型与四代 NVIDIA GPU，做了一次覆盖 prefill/decode、64K 与 1M 上下文的大规模扫描。

本节先交代模型、硬件与搜索空间的设置，再看 Pareto 前沿的形状，最后用一张内存流量表解释 39.5 倍解码差距的来源。

### 5.1 实验设置：把 2026 年旗舰 MoE 放进同一把标尺

> 评估选了四个代表性模型与四代 NVIDIA 硬件，参数规模从 284B 一路跨到 2.8T。下文表 3、表 4 给出模型结构与精度配置：

- 四个模型总参从 284B 跨到 2.8T、激活参数 13B–104B，专家数从 256 到 896 不等，注意力结构涵盖 CSA+HCA、DSA+MLA、KDA+MLA。
- 精度上，V4 系 KV 主缓存用 FP8、索引缓存用 FP4，FFN 走 FP4，Kimi K3 的 KDA 状态保留 BF16。*H200 与 GH200 不原生支持 FP4，仿真按“FP4 与 FP8 等速”的保守假设处理，这等于让新旧硬件在同一条起跑线上比较架构，避免把“不支持 FP4”误算成架构劣势。*

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/09.webp)

::: image-caption
表 3：RoofLang 评估所用模型：四个模型的参数量（总参/激活）、层数、注意力结构与 FFN 专家配置。专家规模从 256 一路到 896、单 token 路由 4–16 个专家不等，MoE 路由通信量差异巨大，这为后文“互联敏感度”和“激活参数字节决定解码右边界”埋下伏笔。
:::

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/10.webp)

::: image-caption
表 4：评估中的模型精度配置：各模型 KV、Attention、FFN 三部分分别使用 FP8/FP4/BF16 的组合。FP4 KV 索引缓存与 FP4 FFN 是 V4 系压缩单 token 字节数的关键手段，而 Kimi K3 的 KDA 状态仍是 BF16，精度策略直接影响后文的每 token 内存流量。
:::

- 硬件如表 5：H200、GH200、B300、GB300，scale-up 假设 NVLink、scale-out 假设 InfiniBand。B300/GB300 原生 FP4 算力达 13,500/15,000 TFLOPS、HBM 均为 288GB；关键对照是 GB300 的 NVLink 域覆盖 72 卡而 B300 仅 8 卡。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/11.webp)

::: image-caption
表 5：评估硬件平台：H200、GH200、B300、GB300 在 FP4/FP8/BF16/TF32 各精度的峰值算力、HBM 容量与带宽、NVLink 带宽及最大域规模、InfiniBand 带宽。GB300 相对 B300 算力只高约 11%，NVLink 域却从 8 卡扩到 72 卡——论文借此把“算力进步”与“互联域进步”两个变量分开观察。
:::

- 搜索空间如表 6：prefill 与单步 decode 两阶段，64K/1M 上下文，8/64 卡，并行维度 DP、CP、PP、EP 满足 EP×PP = DP×CP×PP = GPU 总数，批量从 1 开始倍增直到首次 OOM。作者用 CP 而非 TP，因顺序切分更适合长上下文，且避免 TP 的 KV 复制。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/12.webp)

::: image-caption
表 6：工作负载配置搜索空间：阶段、上下文长度、GPU 数量、并行维度约束与批量扫描方式。批量上限由仿真器的容量事件自然判定（OOM 即停），并行组合满足严格的乘积约束，这保证扫出的每一个点都对应一张合法、可部署的架构配置。
:::

### 5.2 Pareto 前沿：吞吐与交互性的跷跷板

> 作者把每个配置画成“吞吐–交互性”平面上的一个点，所有最优配置连成一条 Pareto 前沿。

下面图 3、图 4 分别是 prefill 和 decode 的 Pareto 前沿：图表纵轴 tokens/s/GPU 为吞吐，横轴 tokens/s/user 为单用户交互性，每行一个模型、每列一种上下文，子图含 8/64 卡与四种硬件曲线。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/13.webp)

::: image-caption
图 3：Prefill 阶段的吞吐–交互性 Pareto 前沿，四行依次为 DeepSeek V4 Flash、GLM-5.3、DeepSeek V4 Pro、Kimi K3，列为 64K/1M 上下文，曲线区分 H200、GH200、B300、GB300 与 8/64 卡。所有曲线都呈典型跷跷板形状：左侧大 batch、高算术强度点受算力屋顶约束；GB300（紫色菱形）凭借更大 scale-up 域在多数配置下把前沿整体外推，而 Kimi K3 在 1M 下 8 卡几乎跑不动、只能依赖 64 卡，直观显示超大专家模型对互联与容量的依赖。
:::

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/14.webp)

::: image-caption
图 4：Decode 阶段的吞吐–交互性 Pareto 前沿，布局同图 3。Decode 右边界（小 batch、高交互性端）几乎垂直下坠，论文指出其由“每步激活参数字节数除以 HBM 带宽”锁定，这是带宽屋顶的直接体现；V4 Flash 的纵轴量级可达 100K tokens/s/GPU，而 Kimi K3 仅数千，数量级差距一目了然。
:::

> 从上面两张图可以看出：

- prefill 左端点是大 batch、高算术强度配置，算力受限，吞吐由模型 FLOPs 除以 GPU FLOP 率决定；
- decode 右端点是小 batch、交互性配置，带宽受限，由每步激活参数字节数除以 HBM 带宽决定。

一条曲线把“要吞吐还是要延迟”的经典权衡完整摊开。

### 5.3 39.5 倍解码差距：公式比参数更能解释胜负

> 最反直觉的发现是：总参数量并不能解释解码吞吐的巨大差距，KV 缓存设计才是。

在 64 张 GB300 上，64K 与 1M 上下文下：

- DeepSeek V4 Flash 的峰值解码吞吐分别是 GLM-5.3 的 9.6 倍和 16.7 倍、Kimi K3 的 25.2 倍和 39.5 倍；
- V4 Pro 对 GLM-5.3 为 3.5 倍和 5.3 倍，对 Kimi K3 为 9.3 倍和 12.4 倍。

> 作者用一个极简公式解释这种差距。设 $P$ 为编码后的模型权重大小，$B$ 为批量，$T_{KV}$ 为每请求 KV 缓存读取量，则每解码一个 token 的内存开销为：

$$
M_{\text{decode}}=\frac{P+B\,T_{\text{KV}}}{B} \tag{5}
$$

也就是说，每解码一个 token，每个用户都要摊一份固定“过路费”——激活权重 $P$ 除以批量，加上自己那份 KV 读取 $T_{KV}$。**批量越大权重摊得越薄，KV footprint 越小能塞下的批量越大**。紧凑 KV 设计由此形成正反馈，:mdnice-bold-italic[在带宽受限区把吞吐差距层层放大]。

> 表 7、表 8 用 64 张 GB300 上的仿真数据给出决定性证据。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/15.webp)

::: image-caption
表 7：64K 上下文、64 张 GB300 下每解码 token 的内存开销：表头是各模型的每请求参数流量、KV footprint、KV 流量、峰值批量与每 token 内存流量。V4 Flash 的峰值批量是 GLM-5.3 的 16 倍、每 token 流量仅为其约十二分之一，紧凑 KV 设计与 FP4/FP8 精度策略把带宽屋顶抬到了别人够不着的高度。
:::

64K 上下文时，V4 Flash 每请求 KV footprint 仅 0.192 GiB、峰值批量高达 65,536、每解码 token 流量 0.035 GiB；GLM-5.3 对应为 2.906 GiB、4,096、0.419 GiB；Kimi K3 为 1.065 GiB、8,192、1.237 GiB。

![文章插图](/articles/2026-09-22-kernel-rooflang-ai-llm-39-adb797d6/images/16.webp)

::: image-caption
表 8：1M 上下文、64 张 GB300 下每解码 token 的内存开销，列含义同表 7。上下文拉长 16 倍后 KV 流量成为绝对主角：GLM-5.3 的 KV footprint 达 46,500 GiB、Kimi K3 每请求 KV 流量 13,721 GiB，而 V4 Flash 凭借 MLA 类压缩与 FP4 索引缓存把 KV 流量压到 0.441 GiB，39.5 倍吞吐差距在这张表里可以逐项算出来。
:::

1M 长上下文把差距进一步放大：V4 Flash 仍能维持 4,096 批量、每 token 0.450 GiB；GLM-5.3 批量跌到 256、每 token 5.423 GiB；Kimi K3 虽有 512 批量，每 token 流量却高达 16.469 GiB，是 V4 Flash 的约 37 倍。

互联敏感度方面，仿真显示更大的 scale-up 网络域能加速 MoE token 路由，在路由流量主导时收益最明显；prefill 以计算为主，路由通信在合理批量下可被隐藏；decode 中 CP 下的 KV 流量由聚合集合通信承载，随 KV 增长、attention 耗时上升，路由占比反而下降，改进空间随之收窄。

::: block-1
总之，这组实验给模型设计与服务系统同时上了一课：:mdnice-highlight[解码吞吐的胜负手是“每 token 字节数 × 可支撑批量”，而非总参数量]。RoofLang 让这条过去只能靠真机 profiling 隐约感知的规律，在架构图纸阶段就被定量、可复算地呈现出来。
:::

## 六、Agent 自主架构师：Claude Code 在 B300 上找到的三种新架构

> 大规模扫描回答“已知架构谁好”，自主闭环还要回答“能不能发现人没试过的架构”。

优化 Agent 极简却完整：Claude Code 以非交互、无轮次上限方式持久运行，按 DeepSeek 官方集成指南配置 V4 Pro 为主模型、V4 Flash 为轻量子模型；:mdnice-bold-italic[Agent 拿到 RoofLang 仓库、V4 Pro 实现与基线结果，持续寻找新的放置与图变换，依据仿真反馈推进 Pareto 前沿，直到自行停止。论文报告了三个代表性发现。]

### 6.1 发现1：成本与头尾感知的连续 PP 划分

> 仅按层数均分会让承载 embedding 的首阶段或输出头末阶段落在关键路径上。

Agent 从 trace 标定各阶段成本，枚举不改层序与 PP 度数的合法连续划分：在 256 张 B300、batch 4,096、EP=16、PP=16 的 V4 Pro 64K 解码场景中，**它把均衡划分 (4×13, 3×3) 替换为 (3, 4×14, 2)，为头尾额外工作量留出预算，最大阶段跨度被压缩，吞吐与交互性同时提升 20.3%。**

### 6.2 发现2：DP-CP 路由专家切分

> Agent 沿既有的“DP 主、CP 次”token 分片，用 split_kernel 把每个路由专家的上、下投影切开，每个 DP×CP 分片各持一份副本；同维 Gather–Scatter 对由此可被 rank 本地数据边替换，去掉冗余的本地 token 物化，同时保留路由关系与专家归属。

切分会带来额外权重读取，Agent 因此自设门槛：每份副本至少摊到 1,024 个路由 token，且 EP 不超过 8。**在 8 张 B300、batch 512 的 V4 Pro 8K prefill 上，巨大的 prefill token 量摊薄了权重流量，移除本地 Gather–Scatter 后两项指标同时提升 6.23%。**

### 6.3 发现3：本地专家复制

> Agent 在一个 EP=16 阶段的两个节点间复制选中的专家，让路由 token 的 dispatch 与 combine 走节点本地 NVLink，而非跨节点 InfiniBand。

变换保持专家计算不变，代价是额外权重读取与 HBM 容量占用，因此只用于通信受限场景。在 16 张 B300、batch 1、CP=EP=16 的 V4 Pro 1M prefill 中，**巨大的路由 token 分片使跨节点通信成为主导，节点本地路由直接削掉瓶颈，两项指标同时提升 50.1%。**

::: block-1
三个案例放在一起，可以看出：:mdnice-highlight[Agent 的发现全部是“带条件的架构动作”，什么场景启用、代价是什么、在哪个阈值下划算，都由仿真反馈反推出来]。它做的事情已经接近一名资深推理架构师的日常：盯 trace、算账、在通信与计算/容量之间做交换，只是试错速度和枚举广度远超人类。
:::

## 定位、边界与启示：架构层与实现层的显式分工

> RoofLang 把自己定位为**“实现之前”的架构筛选器。**

相比 ASTRA-sim、Vidur、KernelSight-LM 等建模工作（分布式仿真、校准到前沿集群的离散事件模型、kernel-roofline 加学习效率因子），它们在保真度与通用性间交换，且无法建立候选架构间的语义等价；**RoofLang 继承 Unity“图代换可验证”的传统，把评估器搬到 LLM 推理并保持实现无关**。它与 KDA-Pilot、KernelPro、OptiFlow 等真机闭环构成上下游：:mdnice-bold-italic[真机反馈具体但搜索受限于现有栈，RoofLang 先在理论资源极限下筛出更优图组织，再交框架层落地。]

该方案也有三条局限。

| 局限 | 描述 |
| --- | --- |
| 初始输入验证 | 目前假设初始两张图正确、离不开人工核验，未来要从官方基线实现自动构造计算图并用执行 trace 反校。 |
| 建模保真度 | 仿真假设 MoE 理想均衡，省略 KV cache 追加、投机解码与 Kimi K3 多模态，后续需在不削弱变异空间与等价契约的前提下用实测校准。 |
| 可扩展性 | Kimi K3 在 128 GPU 下单次仿真耗时数小时、占数百 GB 内存，计划通过复用未变子图、剪枝与并行/近似评估解决。 |

> 当然这背后也有三层启示。模型设计者应看到，解码时代的竞争力越来越由 KV 压缩、激活精度、专家拓扑这些“字节级”指标定义。

$$
M_{\text{decode}}=\frac{P+B\,T_{\text{KV}}}{B} \tag{5}
$$

服务团队会发现，PD 分离、KV 分级、并行切分、专家放置这些散落决策第一次有了统一、可机器搜索的表述，架构评审可从“谁经验多”转向“谁的 Pareto 前沿更靠外”。Agent 工具开发者则得到一种更稳的范式：先定义可验证动作空间与实现无关的评估器，再让 AI 自由探索，因为可验证性保证了下限。

::: block-1
回顾全文，:mdnice-highlight[RoofLang 是希望 LLM 推理系统架构设计具备递归自我改进的入口——AI 提出架构、仿真给出定律级反馈、图变换保证不出错]。当 39.5 倍的模型级差距和 50.1% 的架构级红利都能在图纸阶段被算出来，“架构师”这个角色与“AI 架构师”这个工具，或许都到了重新定义的时候。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[不止算子优化！ncnn Vulkan int8 量化推理后端：全链路架构设计与系统性优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902586&idx=1&sn=14f256d165dccfb6c4ef866fd144ab6b&scene=21#wechat_redirect)**
- **[MINISA：面向下一代可重构推理加速器的最小指令集架构，把 96.9% 指令停顿打到接近零](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901107&idx=1&sn=2b04a2a2493ac8107f13e7467b430cff&scene=21#wechat_redirect)**
- **[比NVIDIA Thor‑U快3.8倍，却少用一个缓存层级：理想汽车M100编排式数据流架构对AI推理原有设计信条的根本性质疑](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900845&idx=1&sn=0044def693c7c0eb4d03f91096349b9d&scene=21#wechat_redirect)**
- **[OpenAI 核心研究员 Noam Brown 深度拆解 Agentic AI、推理与多智能体的技术真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905895&idx=1&sn=6a5d81ee886d6992b05eb8061b077599&scene=21#wechat_redirect)**
- **[AI 递归自进化下，量化前沿实验室真实研发节奏与算力分配，解读Anthropic 三大 AI 研发度量体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905979&idx=1&sn=33d553605fb9915585470a14b1d98dad&scene=21#wechat_redirect)**

:::
