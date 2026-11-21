关键词：**异构内存**、:mdnice-bold-italic[NPU]、**Agentic LLM**、Prefill、Decode、**设计空间搜索**

> *这篇论文最值得抓住的一句话是：**为下一代推理加速器识别合适的内存架构**，需要在一个巨大且快速演进的设计空间中导航，而:mdnice-bold-italic[工作负载特征、NPU 设计维度和内存系统设计之间的相互作用【仍然没有】得到充分探索。]*

过去讨论 AI 芯片，大家很容易先看 TOPS、TFLOPS、矩阵阵列规模和先进封装带来的带宽数字。但进入 agentic LLM 时代，这种视角已经不够了。

::: block-1
一个会调用工具、浏览网页、操作软件、写代码和长链推理的智能体，:mdnice-bold-italic[不再像传统聊天机器人那样有相对稳定的输入输出长度]。**它的上下文会在工具调用中快速膨胀，KV cache 可能变成数百 GB 级别，prefill 和 decode 阶段也呈现完全不同的访存压力。**
:::

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/01.webp)

- **MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs**
- https://arxiv.org/pdf/2604.16007
- 1.4 万字，阅读 50 分钟，**播客 22 分钟**

相关推荐

- [面向 dLLM 采样的高效 NPU 设计：d-PLENA 实现 NVIDIA RTX A6000 2.53倍 GPU 级加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897187&idx=1&sn=70d22f3ca4e3a92a4b486f7b04ffe0f9&scene=21#wechat_redirect)
- [上交、清华提出面向 LLM 推理的多核 NPU 创新策略：全栈多维度优化实现1.32x-6.03x超 SOTA 加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896365&idx=1&sn=4207226468f19a59045f666315481b10&scene=21#wechat_redirect)
- [一张表搞定 NPU 大模型推理！USTC/微软/清华联合提出 T-MAN 方案：解码提速 3.1 倍、能效比领先 84%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896176&idx=1&sn=d31c6a12cb47cc4b60f8548d2d23647c&scene=21#wechat_redirect)

> MemExplorer 这篇论文提出的核心判断是：:mdnice-bold-italic[未来 LLM 推理系统不是单一 GPU 或单一 NPU 的局部优化问题，而是异构设备、异构内存和软件执行策略共同构成的系统级设计问题。]

prefill 更需要高带宽和高吞吐，decode 更需要容量、KV cache 承载能力和更好的功耗效率。把这两类阶段都塞进同一种 HBM 架构里，必然会浪费其中一边的资源。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/02.webp)

::: image-caption
图 4：MemExplorer框架概述。扩展后的PLENA仿真器用于验证系统模型的准确性。该图完整展示了MemExplorer的核心架构，由系统模型、数据移动模型、工作负载特化与系统协同探索四大核心模块构成，形成“建模-仿真-探索-验证”的闭环设计流程。系统模型整合内存与计算解析模型，可快速预估性能与功耗；数据移动模型定义数据流、数据放置与带宽分配策略；工作负载特化模块区分预填充/解码场景建模；协同探索采用多目标贝叶斯优化遍历庞大设计空间。框架关键创新是用扩展PLENA周期级仿真器交叉验证解析模型精度，兼顾探索效率与结果可靠性，为异构NPU内存架构的快速设计空间探索提供了系统化解决方案，解决了传统仿真效率低、解析模型不准的核心矛盾。
:::

论文的实验结论相当直接：在相同功耗预算下：

- MemExplorer 面向 agentic workload 相比 baseline NPU 最高实现 2.3 倍能效提升；
- 在 prefill-only 场景下相比 H100 最高达到 3.23 倍能效；
- 在 decode 场景下，在等性能目标下相比 baseline NPU 和 H100 分别最高实现 1.93 倍和 2.72 倍功耗效率提升。

这些数字背后的本质不是某一个神奇内存技术胜出，而是**系统能自动判断什么时候需要 3D-stacked SRAM，什么时候需要 HBF，什么时候 LPDDR 比更高带宽的内存更合算。**

::: block-1
MemExplorer 的价值不在于宣称某种内存一定最好，而在于:mdnice-bold-italic[把内存技术、NPU 计算阵列、量化精度、数据流和负载阶段放进同一个可搜索空间，让系统自己找到 prefill 和 decode 各自的最优折中。]
:::

## 本文目录

- 一、问题背景：Agentic LLM 正在改变推理系统的内存假设
- 二、MemExplorer 的核心思想：把内存系统变成可综合、可搜索的设计对象
  - 2.1 MemExplorer 面向异构 NPU 的三层设计目标
  - 2.2 异构内存设计空间
  - 2.3 可连接内存的 stack 数量上界
- 三、内存层级模型：不只算容量，还要算数据如何逐层流动
  - 3.1 有效带宽公式
  - 3.2 单层传输延迟
  - 3.3 内存功耗模型
- 四、两类新兴内存：3D-stacked SRAM 与 HBF 分别解决什么问题
  - 4.1 3D-stacked SRAM
  - 4.2 HBF
- 五、软件策略：数据流、片上存储优先级和片外带宽分配
  - 5.1 软件策略
- 六、搜索方法：用多目标贝叶斯优化逼近 Pareto 前沿
- 七、实验结果：Prefill 要高带宽片上内存，Decode 要大容量低功耗层级
- 八、扩展到新模型：Diffusion LLM 与大规模 MoE
- 九、相关工作：本文相对已有研究的差异
  - 9.1 第一类是内存技术本身
  - 9.2 第二类是内存模拟和软硬件协同
  - 9.3 第三类是 prefill-decode disaggregation
- 十、结论与展望
  - 10.1 结论总结
  - 10.2 进阶分析
  - 11.3 未来工作

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/03.webp)

## 一、问题背景：Agentic LLM 正在改变推理系统的内存假设

> 传统 LLM 服务更多面对的是聊天式负载，输入输出长度相对可预测；agentic LLM 则不同，它要和外部环境交互，**工具调用和长链推理会带来快速增长的上下文与高度动态的访存模式。**

论文从 agentic workload 切入，这是非常关键的选择。:mdnice-bold-italic[因为只有在这类负载下，内存系统问题才会从“带宽够不够”升级为“容量、带宽、层级、功耗、阶段差异如何协同”]。作者举了几类典型场景：computer-use agents、autonomous coding agents、web-use agents。这些任务有一个共同点：模型不是简单地回答一句话，而是在外部系统中反复观察、推理、调用工具和生成行动。

> 这会导致两个后果。

- **第一，token 规模急剧增加**。论文图 1a 展示了 chatbot、WUA、CUA 之间的 token 处理差异，:mdnice-bold-italic[agentic workloads 在 prefill 阶段尤其显著地增加了 token 处理需求。]

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/04.webp)

::: image-caption
图 1a：智能体工作负载在预填充阶段的token处理需求显著更大。智能体工作负载与传统对话式LLM推理差异显著，其预填充阶段需处理超长输入序列，token处理量远超常规场景。
:::

- **第二，KV cache 和 activation 的容量压力变大**。论文指出，具有 agentic 能力的 LLM 已经扩展到百万级上下文窗口，仅 KV cache 就经常超过 500GB。这意味着 80GB HBM 级别的单卡内存容量，在很多长上下文智能体场景下已经不再是宽裕配置。

> 更麻烦的是，prefill 和 decode 不是同一种计算。

- prefill 处理长输入序列，通常计算密度高，activation 和 weight 的吞吐压力明显；
- decode 逐 token 生成，KV cache 访问和 batch size 受容量限制的问题更突出。

用一句硬件语言说：prefill 更像吞吐型矩阵计算问题，decode 更像容量受限的访存与复用问题。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/05.webp)

::: image-caption
图 1c：智能体工作负载大幅增加激活值和KV缓存的内存容量需求，同时带宽利用率适度上升，且预填充和解码阶段表现出截然不同的特征。智能体工作负载的长上下文特性使激活值与KV缓存占用空间暴涨，对内存容量提出严苛要求。同时，预填充阶段批量处理数据，带宽利用率高且波动大；解码阶段逐token生成，带宽需求低但持续稳定，两阶段带宽特征差异明显。这揭示了单一内存架构无法同时适配两阶段需求的核心矛盾，也证明预填充侧重带宽、解码侧重容量的异构设计思路具备合理性，为MemExplorer分阶段优化内存架构提供了关键数据支撑。
:::

> 这也是为什么业界开始采用异构系统。论文提到 NVIDIA Vera Rubin 平台，把不同设备组合到一个互联系统中：

- Rubin GPU 使用 HBM4，提供每节点 288GB 容量和最高 22TB/s 带宽；
- Groq 3 LPX 则使用聚合片上 SRAM 的数据流处理器，提供 128GB 片上 SRAM 和 40PB/s 聚合带宽。

::: block-1
它们不是同一种内存架构，却服务于同一个 LLM serving 任务。AWS 与 Cerebras 的合作也体现了类似趋势：**未来 LLM serving 不太可能只依赖统一设备，而会走向多种设备、多种内存体系并存。**
:::

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/06.webp)

::: image-caption
图 2：专为纯预填充和纯解码工作负载优化的NPU设计。黑色边缘表示计算单元与内存之间所有可能的数据移动，图中列出的参数定义了NPU系统的设计空间，直观展示了预填充与解码场景下NPU内存架构的差异化设计逻辑，核心差异体现在片上存储、片外内存组合及关键参数配置上。预填充加速器搭载高密度HBM、大容量3D堆叠SRAM，侧重高带宽以支撑批量数据处理；解码加速器舍弃高带宽HBM，采用大尺寸片上SRAM与HBF组合，优先保障大容量以容纳KV缓存。同时，PE阵列维度、向量长度、内存堆叠数量等参数构成完整设计空间，MemExplorer正是围绕这些参数开展协同优化。这种分场景硬件设计思路，契合论文提出的预填充-解码异构解耦理念，是提升智能体推理能效的核心设计方向。
:::

::: block-1
**Agentic LLM 让内存系统从“加速器附属模块”变成“系统设计主变量”**：容量决定能不能跑大上下文，带宽决定 prefill 能不能喂饱计算，功耗决定 decode 能不能以合理成本扩展。
:::

## 二、MemExplorer 的核心思想：把内存系统变成可综合、可搜索的设计对象

> MemExplorer 不是简单的模拟器，而是一个 **memory system synthesizer**。它的任务不是评估某个固定 NPU，而是**自动决定异构 NPU 系统中的内存层级、内存技术和 NPU 设计参数。**

### 2.1 MemExplorer 面向异构 NPU 的三层设计目标

论文提出 MemExplorer，目标是面向异构 NPU 系统做端到端架构探索。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/07.webp)

::: image-caption
图 4：MemExplorer框架概述。扩展后的PLENA仿真器用于验证系统模型的准确性。
:::

它的设计目标可以拆成三层：

- 第一，统一抽象多种内存技术；
- 第二，构建异构、解耦式 serving 的探索空间；
- 第三，把算法感知的软件策略与硬件设计联合搜索。

### 2.2 异构内存设计空间

> 这件事难在什么地方？不同内存技术的物理约束差异极大。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/08.webp)

::: image-caption
图 1b：异构内存设计空间。轮廓块清晰划分了不同技术在容量、带宽和访问延迟参数上的区域。图中覆盖SRAM、HBM、LPDDR、GDDR及HBF等主流与新兴内存技术，清晰呈现不同技术在容量、带宽、延迟三维度的权衡关系。SRAM聚焦超高带宽低容量，HBM平衡带宽与容量，HBF主打超大容量中等带宽，各技术形成互补生态。这一分布说明无单一内存技术可同时满足智能体推理的高带宽、大容量、低功耗需求，也奠定了MemExplorer探索异构分层内存架构的技术基础，为多技术组合优化提供了空间依据。
:::

- 片上 SRAM 延迟极低、带宽极高，但容量和泄漏功耗受面积制约；
- HBM 带宽高但容量受堆叠和封装边界限制；
- LPDDR 功耗友好但带宽相对有限；
- GDDR 更偏成本和带宽折中；
- HBF 容量非常大，但延迟高、读写能耗和背景功耗也更高。

若只用“容量”和“带宽”两个数字比较，就会忽略 die shoreline、PHY 占用、背景功耗、读写能耗、封装和热约束。

::: block-1
**MemExplorer 的统一建模方式，是把每类内存技术抽象成一组紧凑参数**，包括延迟、容量、带宽、shoreline 占用、背景功耗、读能耗、写能耗等。
:::

论文表 1 给出了 SRAM、HBM3E、HBM4、LPDDR5X、LPDDR6、GDDR6、GDDR7、3D-stacked SRAM 和 HBF 等设备参数。这个表不是简单罗列规格，而是为后续搜索提供统一坐标系。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/09.webp)

::: image-caption
表 1：不同内存设备的技术参数。带∗标记的值为实验数据，带†标记的值为根据缩放因子计算得出的值。该表系统梳理了片上SRAM、主流片外内存（HBM3E/4、LPDDR5X/6、GDDR6/7）及新兴内存（3D堆叠SRAM、HBF）的核心参数，涵盖延迟、容量、带宽、功耗、读写能耗等关键指标。数据清晰呈现各技术的权衡：SRAM/3D堆叠SRAM延迟仅数纳秒、带宽极高，适合片上高频访问；HBM平衡带宽与容量，适配中高端推理；LPDDR低功耗、GDDR低成本，适配特定场景；HBF容量达384GB，填补HBM与SSD间的容量空白，但延迟达微秒级。该参数表是MemExplorer构建统一内存模型、开展异构设计探索的基础，为多技术组合优化提供了精准的数据支撑。
:::

### 2.3 可连接内存的 stack 数量上界

论文对 off-chip memory stack count 的讨论尤其重要。HBM、LPDDR、GDDR、HBF 等片外内存都要消耗 die shoreline，也就是计算芯片边缘可用于高速 PHY 接口的物理长度。当前光刻 reticle limit 给 die 尺寸带来上界，论文采用约 $26mm \times 33mm$ 的曝光场约束，因此**即使功耗无限，能挂多少片外内存堆栈也不是任意的。**

> 论文用如下公式描述可连接内存 stack 数量的上界：

$$
N_{stack} \lesssim \left\lfloor \frac{L_{mem}}{L_{PHY} + L_{margin}} \right\rfloor
$$

其中，$N_{stack}$ 表示可连接的片外内存堆栈数量，$L_{mem}$ 表示为内存接口预留的 die 边缘长度，$L_{PHY}$ 表示单个内存 stack 对应的 PHY 接口占用长度，$L_{margin}$ 表示接口之间需要保留的物理间隔。**这个公式背后的直觉很简单：芯片边缘就像一条有限长度的“码头”，HBM、LPDDR、HBF 这些船再好，也必须有泊位才能接上。**

::: block-1
MemExplorer 的第一层贡献，是把多种内存技术从“规格表比较”提升为“带物理约束、功耗模型和层级行为的统一设计变量”。
:::

## 三、内存层级模型：不只算容量，还要算数据如何逐层流动

> 对 LLM 推理来说，数据访问通常是大块连续搬运，尤其是权重、activation 和 KV cache。因此论文**把每次读写建模为一次大粒度事务，并记录计算单元消耗的带宽。**

MemExplorer 的内存层级模型围绕一个核心问题：**当数据分布在多个层级的内存中时，从远端内存到计算单元的传输延迟和带宽瓶颈如何估计**。论文把 level 0 定义为计算单元，level $L$ 定义为最远端内存。数据从深层向浅层逐级移动，每个边界 $i$ 都有有效带宽、固定传输延迟、数据量和驻留比例。

### 3.1 有效带宽公式

论文给出有效带宽公式：

$$
B^{eff}_i = B^{peak}_i - B^{eff}_{i+1}
$$

其中，$B^{eff}_i$ 表示从 level $i+1$ 到 level $i$ 的有效带宽，$B^{peak}_i$ 表示该层级边界的峰值带宽，$B^{eff}_{i+1}$ 表示更深层向当前层传输已经消耗的带宽。

**这个公式表达的是双缓冲场景下的竞争关系**：当前层既要从更深层接收数据，又可能向更浅层继续输送数据，**所以真正可用带宽不是峰值，而是扣除下游占用后的剩余。**

### 3.2 单层传输延迟

单层传输延迟则写作：

$$
\tau_i(x,\alpha_i)=\lambda_i+\frac{\alpha_i x}{B^{eff}_i}
$$

其中，$\tau_i$ 表示在第 $i$ 层加载数据的延迟，$x$ 表示总数据量，$\alpha_i$ 表示存储在当前层的数据比例，$\lambda_i$ 是跨越该层级边界的固定延迟。通俗说，:mdnice-bold-italic[如果一部分数据已经在近端内存中，那么只需要用当前层带宽搬这一部分；剩余部分要递归地从更深层搬上来。]

::: block-1
这个模型的意义在于，它不是简单地说“HBM 有 2TB/s，HBF 有 1TB/s”，而是估计当权重、KV cache、activation 被分散放在不同层级时，哪些传输能被双缓冲隐藏，哪些会暴露成 stall 时间。

对 NPU 来说，真正影响吞吐的不是内存峰值带宽，而是计算和搬运能否重叠。
:::

### 3.3 内存功耗模型

论文还给出内存功耗模型：

$$
P(C,BW_{read},BW_{write})=p_{bg}\cdot C+e_{read}\cdot BW_{read}+e_{write}\cdot BW_{write}
$$

其中，$C$ 表示总内存容量，$BW_{read}$ 表示读带宽，$BW_{write}$ 表示写带宽，$p_{bg}$ 表示单位容量背景功耗，$e_{read}$ 和 $e_{write}$ 分别表示每 bit 读写能耗。:mdnice-bold-italic[这个公式揭示了 HBF 这类大容量内存的隐性代价：即使带宽够用，容量越大，背景功耗也会持续存在；如果读写能耗高，频繁访问会进一步拉高功耗。]

::: block-1
MemExplorer 的内存模型抓住了 LLM 推理中最关键的系统事实：**性能不是由单层峰值带宽决定，而是由多层内存之间的数据驻留比例、双缓冲重叠能力和读写功耗共同决定。**
:::

## 四、两类新兴内存：3D-stacked SRAM 与 HBF 分别解决什么问题

论文把 3D-stacked SRAM 和 High Bandwidth Flash 作为两个代表性新兴内存技术。它们分别位于设计空间的两个极端：一个追求极高片上带宽和低延迟，一个追求超大容量。

### 4.1 3D-stacked SRAM

> 其解决的是片上容量不足的问题。传统 2D SRAM 受面积和泄漏限制，在 7nm 以下扩展速度放缓。3D-stacked SRAM 通过垂直堆叠多层 SRAM die，在不扩大 compute die 平面面积的前提下提高有效容量。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/10.webp)

::: image-caption
表 1：不同内存设备的技术参数。带∗标记的值为实验数据，带†标记的值为根据缩放因子计算得出的值。
:::

论文表 1 中给出的单层 3D-stacked SRAM 约 1GB、8TB/s，并保持很低的访问延迟和每 bit 能耗。

::: block-1
它**特别适合 prefill 阶段**，因为 prefill 中 activation 工作集大且复用频繁，如果能把更多 activation 留在片上，就可以减少对 HBM 的压力。
:::

### 4.2 HBF

> HBF 则解决的是容量断层问题。HBM 容量一般是几十 GB 级，而 SSD 容量巨大但延迟和接口形态不适合作为推理内存。HBF 把 NAND Flash、DRAM buffer 和高带宽 PHY 垂直集成，单栈可提供约 384GB 容量和 1TB/s 带宽，容量密度远高于 DRAM 类方案。

但 HBF 的代价也很明显：背景功耗约为 HBM3E 的 4 倍，读写能耗约为 2 倍，访问延迟达到微秒级，比 SRAM 慢三个数量级。

::: block-1
因此 HBF 不适合作为主要带宽源，更适合作为最外层容量 tier，**用于长上下文场景下权重存储和 KV cache 溢出。**
:::

**这里有一个很值得注意的判断：HBF 并不是“更好的 HBM”，也不是“更快的 SSD”这么简单。它在 MemExplorer 里的角色是容量补足层**。当 decode 阶段需要更大 batch 以提升阵列利用率时，HBF 可以用容量换吞吐；但如果负载是高频随机访问或带宽主导，HBF 的能耗和延迟就可能成为负担。

::: block-1
3D-stacked SRAM 和 HBF 代表了两条完全不同的路线：前者把热点数据拉近计算，后者把冷或半热的大容量状态留在系统内存层级中。MemExplorer 的意义在于能判断二者何时该出现、出现多少、放在哪一层。
:::

## 五、软件策略：数据流、片上存储优先级和片外带宽分配

MemExplorer 不仅搜索硬件，还把软件执行策略纳入同一个设计空间。这里的关键是数据到底留在哪里、从哪里流过来、带宽优先给谁。

### 5.1 软件策略

> 论文把软件策略分成三类：Dataflow Strategy、On-Chip Storage Priority 和 Off-Chip Bandwidth Priority。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/11.webp)

::: image-caption
图 3：异构内存层级中软件控制的数据流策略、带宽分配与存储调度示例，该图清晰呈现MemExplorer软件层协同优化内存访问的核心策略，涵盖片外带宽优先级、片上存储优先级及数据流策略三大维度。左侧为片外带宽优先模式，聚焦矩阵计算带宽分配；右侧为片上存储优先模式，优先将激活值、权重或KV缓存驻留片上。权重、激活值、KV缓存三类数据在不同层级内存间的流动路径，体现了软件调度对数据复用率的调控作用。论文通过消融实验验证，权重驻留数据流+激活值优先片上存储的组合，能最大化数据复用、减少片外访问，这一策略可降低内存功耗、提升计算效率，是MemExplorer算法感知协同设计的关键组成部分。
:::

- Dataflow Strategy 包括 weight stationary、input stationary 和 output stationary。:mdnice-bold-italic[所谓 stationary，就是某类数据尽量留在片上，让其他数据从内存流过来。对 NPU 阵列而言，选择哪类数据 stationary，会改变权重、activation 和输出在内存层级中的复用方式。]
- **On-Chip Storage Priority 抽象的是编译器级存储调度：片上空间有限时，优先存权重、activation、KV cache 还是平均分配**。这个抽象看似简单，但它把复杂编译优化转成可搜索参数。对 prefill 来说，论文实验显示 activation-prioritized on-chip storage 结合 weight-stationary 数据流效果很好，因为它能最大化数据复用并降低 off-chip traffic。
- **Off-Chip Bandwidth Priority 处理的是多个数据类型同时流动时带宽如何分配**。很多 NPU 可以用不同 buffer 并发流式传输权重、activation 或 KV cache，但片外带宽总量有限，不可能每条路径都满速。:mdnice-bold-italic[论文为控制设计空间复杂度，采用固定分配策略，例如 matrix priority 时给 matrix data 分配 75%带宽，vector data 分配 25%带宽。]

上图 3 展示了软件控制的数据流策略、带宽分配和存储调度示例。:mdnice-bold-italic[图中展示了 on-chip、off-chip、systolic array、HBM 和 SRAM 之间的数据移动关系。它说明 MemExplorer 不是只看硬件规格，而是把数据流和存储优先级作为影响性能和能效的一等公民。]

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/12.webp)

::: image-caption
表 4：软件策略消融实验。基于表6中P1配置、批量大小为1的软件策略消融实验。权重驻留（WS）数据流+激活值优先片上存储，通过最大化数据复用、减少片外流量实现最优能效；结合双缓冲技术和权重优先带宽分配，实现计算与内存传输的高效重叠。该表4量化分析了片上存储优先级、数据流策略、带宽分配优先级三大软件策略对能效的影响。基准配置（均衡存储、输出驻留、均衡带宽）为参照，S1-S4为不同策略组合。结果显示，权重驻留数据流+激活值优先片上存储+权重优先带宽分配（S4）能效最优，达基准的2.31倍；原因是该组合最大化权重数据复用、优先缓存激活值，大幅减少片外内存访问，同时合理分配带宽，避免资源浪费。而输入驻留数据流（S5）因激活值频繁传输，能效反而低于基准。该消融实验验证了软件调度策略对异构内存系统能效的关键影响，为MemExplorer的算法感知协同设计提供了最优软件策略组合。
:::

从表 4 的软件策略消融看，Base 到 S3 的 Token/J 从 1.00 倍提升到 2.31 倍，关键组合是 activation 优先片上存储、weight-stationary 执行、weight 优先带宽。S4 反而降到 0.59 倍，**说明软件策略不是“越偏向某类数据越好”，而要和负载阶段、内存层级、带宽分配共同匹配。**

::: block-1
这篇论文最工程化的部分在于，它**没有把“编译器怎么调度数据”当成后处理，而是把数据流、片上驻留和带宽分配直接放入硬件架构搜索。**
:::

## 六、搜索方法：用多目标贝叶斯优化逼近 Pareto 前沿

> MemExplorer 面对的是约 $10^6$ 级别的设计空间，穷举搜索不可行。因此论文采用 Multi-Objective Bayesian Optimization，用较少评估次数逼近吞吐和功耗之间的 Pareto 前沿。

论文的搜索空间包括 PE array 维度、VLEN、3D-stacked SRAM 层数、传统 SRAM 开关、HBM/HBF/GDDR/LPDDR 类型和 stack 数量、activation/KV/weight 量化精度、片上存储优先级、dataflow 策略、片外带宽优先级等。表 2 给出了完整空间，例如 PE array 可选 $128 \times 128$、$64 \times 256$、$32 \times 512$、$16 \times 1024$，VLEN 可选 128 到 2048，HBM 可选 HBM3E 或 HBM4 并配置 1、2、4、8 个 stack。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/13.webp)

::: image-caption
表 2：设计空间参数，包含计算、内存配置及软件策略。片上存储指分配额外片上内存存储高频复用数据（权重、激活值、KV缓存），减少片外流量；数据流策略指不同GEMM执行策略（权重驻留WS、输入驻留IS、输出驻留OS），决定内存层级间数据复用模式；带宽分配指重新分配有限片外带宽，最大化带宽利用率。该表全面定义了MemExplorer探索的百万级设计空间，涵盖计算、内存、软件三大维度的核心参数。计算维度包含PE阵列维度、向量长度；内存维度覆盖3D堆叠SRAM层数、HBM/HBF/GDDR/LPDDR的类型与堆叠数；软件维度包含量化精度、片上存储优先级、数据流策略、带宽分配优先级。各参数均提供离散化候选值，兼顾设计可行性与探索效率。MemExplorer将硬件架构与软件策略纳入同一协同设计空间，区别于传统固定内存配置的优化思路，通过多目标优化算法遍历参数组合，锁定适配智能体工作负载的最优软硬件协同方案，是其实现高效异构内存设计的核心前提。
:::

搜索目标是同时最大化吞吐、最小化功耗，并满足总系统成本约束。设 $x \in X$ 为一个设计配置，$f(x)$ 表示吞吐、功耗等多个目标，:mdnice-bold-italic[目标是找到接近真实 Pareto frontier 的设计点。 Pareto 最优的含义是：不存在另一个配置能在不变差任何目标的情况下改善至少一个目标。]

论文采用 GP 作为代理模型，用 EHVI 作为采集函数。初始化阶段用 Sobol 准随机序列采样 20 个配置；迭代阶段不断拟合 GP、最大化 EHVI、评估新配置，直到 100 次评估预算耗尽。GP 给出预测均值和不确定性，EHVI 衡量一个候选点如果被评估，可能给当前 Pareto 集合带来多少超体积改善。

论文定义超体积为：

$$
HV(P_t,r)=Vol(\{y\in R^M|\exists x\in P_t:f(x)\preceq y\preceq r\})
$$

其中，$P_t$ 是当前 Pareto 集合，$r$ 是参考点，$HV$ 表示当前 Pareto 集合支配的目标空间体积。EHVI 则是：

$$
\alpha_{EHVI}(x)=E[HV(P_t \cup \{x\},r)-HV(P_t,r)|D_t]
$$

其中，$\alpha_{EHVI}(x)$ 表示候选设计 $x$ 的期望超体积提升，$D_t$ 表示已经评估过的数据集。通俗说，EHVI 会优先选择那些既可能很强、又能扩展 Pareto 边界的设计点。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/14.webp)

::: image-caption
图 6：设计空间探索过程中超体积（HV）收敛性，基于10个随机种子取均值（阴影区域表示±1标准差）。对比四种方法：本文方法（GP+EHVI）拟合独立高斯过程代理模型，通过最大化解析期望超体积改进值选择下一候选；NSGA-II是基于非支配排序和拥挤距离的种群进化算法；MO-TPE是基于Optuna实现的多目标树结构Parzen估计器；随机搜索为均匀随机采样。前20步采用相同随机初始化，保证对比公平性。该图验证了MemExplorer采用的多目标贝叶斯优化（GP+EHVI）在设计空间探索中的高效性。超体积（HV）是衡量帕累托前沿质量的核心指标，数值越高代表优化方案性能-功耗权衡越优。对比NSGA-II、MO-TPE和随机搜索，GP+EHVI收敛速度最快，在有限评估次数内快速逼近最优帕累托前沿，且标准差更小、稳定性更强。这是因为高斯过程代理模型能精准拟合设计参数与性能、功耗的非线性关系，EHVI采集函数可平衡探索与利用，高效筛选高潜力配置。该结果证明MemExplorer的探索算法能高效遍历百万级设计空间，快速锁定异构内存系统的最优配置，解决了传统遍历搜索效率低下的难题。
:::

::: block-1
MemExplorer 把 NPU 内存设计问题从经验调参变成多目标优化问题：不是找一个单点最优，而是**在吞吐、功耗和容量约束之间寻找可解释的 Pareto 前沿。**
:::

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/15.webp)

::: image-caption
图 5：内存读写功耗测量。左侧读能耗（$e_{read}$）路径展示从内存单元读取数据并传输至Plena计算核的端到端能耗；右侧写能耗（$e_{write}$）路径体现从Plena计算核传输数据并写入内存单元的能耗。该图拆解了内存读写能耗的物理路径，为MemExplorer精准构建内存功耗模型提供了实测依据。读能耗涵盖内存单元读取、片上传输、接口通信等环节，写能耗则包含数据接收、信号转换、单元写入等过程，两路径的能耗损耗节点差异直接影响整体功耗计算精度。论文中内存功耗模型将静态背景功耗、读/写单位能耗作为核心参数，正是基于此类实测数据。精准的读写能耗建模，让MemExplorer在设计空间探索中能准确评估不同内存配置的功耗代价，实现性能与功耗的平衡优化，为异构内存架构的能效评估奠定了基础。
:::

## 七、实验结果：Prefill 要高带宽片上内存，Decode 要大容量低功耗层级

> 实验部分最重要的结论是：prefill 和 decode 的最优内存结构确实不同，而且这种不同足以支撑分离搜索和异构设备设计。

论文在 Llama-3.3-70B、Qwen3、LLaDA 等模型上评估，并使用 BFCL 和 OSWorld 作为 agentic workload。**为了模拟真实 agentic 行为，作者使用 Llama-3.3-70B 运行 BFCL-Web Search Base 和 OSWorld LibreOffice**，得到两个代表性 token 配置：BFCL-Web Search Base 为 114K prompt 和 5K generation，OSWorld LibreOffice 为 90K prompt 和 8K generation。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/16.webp)

::: image-caption
表 3：BFCL智能体工作负载Web-Search-Base子任务的位宽消融实验。8/8/8配置精度与全精度基准相当，同时将内存带宽和存储需求减半，后续实验均采用该配置。验证了权重/激活值/KV缓存量化位宽对智能体任务性能、带宽及存储需求的影响。实验基于Qwen3-32B模型，全精度（16/16/16）虽性能最优，但带宽与存储占用极高；8/8/8量化配置在保持任务成功率的同时，带宽和存储需求减半，实现性能与资源开销的平衡；4/4/4量化因精度过低，任务成功率大幅下降，无法满足智能体推理需求。该消融实验确定了后续实验的标准量化配置，证明适度量化是降低内存压力、提升推理效率的有效手段，也为MemExplorer探索量化与内存架构的协同优化提供了关键依据，平衡了模型精度与硬件资源约束。
:::

量化方面，表 3 显示在 Qwen3-32B 的 BFCL Web-Search-Base 任务上，8/8/8 配置在成功率上与 16/16/16 基线相当，甚至从 0.33 到 0.36 略高，同时把峰值带宽和存储需求减半，从 174.4GB 降到 87.2GB。**4/4/4 虽然进一步降低存储到 43.6GB，但成功率降到 0.17。因此后续实验采用 8/8/8 作为折中点。**

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/17.webp)

::: image-caption
表 5：固定软件策略为表6中P1配置的内存层级消融实验。增加片上SRAM容量可容纳更多激活值和权重，降低片外带宽需求，适配低带宽、高能效的片外内存技术。该表5对比了不同内存层级组合在功耗、批量大小、吞吐量、能效上的差异。基准配置为单层SRAM+HBM3E，批量仅1、能效最低；H1配置增加3D堆叠SRAM，批量不变但能效提升2.62倍；H2配置叠加LPDDR5X，批量提升至8、能效达基准3.06倍；H3配置加入HBF，批量达32，但功耗飙升、能效回落。实验证明，3D堆叠SRAM是降低片外带宽依赖的核心，多层异构内存（SRAM+HBM+LPDDR）可平衡容量、带宽与功耗；而HBF虽提供超大容量，但高功耗限制其能效，适合作为容量补充层级。该结果为MemExplorer设计分层异构内存架构提供了层级组合的最优思路。
:::

**内存层级消融表 5 显示，增加 3D-SRAM 可以显著提升 Token/J**。

- Base 使用 1 个片上 SRAM 和 4 个 HBM3E，Token/J 为 1.00 倍；
- H1 使用 3 层 3D-SRAM 和 4 个 HBM3E，Token/J 达到 2.62 倍；
- H2 进一步加入 LPDDR5X，最大 batch 提高到 8，Token/J 达到 3.06 倍；
- H3 加入 HBF 和 LPDDR5X 后最大 batch 到 32，但功耗升到 718.96W，Token/J 降到 1.55 倍。

这说明容量不是越大越好，HBF 如果带来过高背景功耗，可能削弱能效。

> 表 6 是核心结果。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/18.webp)

::: image-caption
表 6：OSWorld任务（输入90K token、输出8K token）设计空间探索筛选出的预填充和解码优化帕累托前沿样本，该表列出了预填充和解码场景下的最优异构内存与计算配置，严格遵循700W功耗约束。预填充最优配置P1采用2048×256 PE阵列、3层3D堆叠SRAM、HBM4+HBF组合，吞吐量达基准6.71倍，适配长序列批量处理；能效优先配置P2降低计算规模、增加LPDDR层级，兼顾中高吞吐量与低功耗。解码最优配置D1精简计算单元、保留单层SRAM、HBM3E+HBF组合，吞吐量提升1.44倍；高能效配置D2增加3D堆叠SRAM与LPDDR层级，批量达32、能效最优。两类场景配置差异显著，预填充侧重高带宽片上存储与HBM，解码侧重大容量层级与精简计算，验证了分阶段异构设计的必要性与有效性。
:::

Prefill 优化中：

- P1 配置采用 $2048 \times 256$ PE array、VLEN 2048、3 层 3D-stacked SRAM、2 个 HBM4、1 个 HBF，软件策略为 activation storage、weight-stationary、matrix bandwidth priority，平均功耗 632.3W，TDP 697.1W，batch 为 16，TPS 达到 Base 的 6.71 倍。
- P2 更偏能效和容量，使用 2 层 3D-stacked SRAM、2 个 HBM4、两层 LPDDR5X，batch 为 16，TPS 为 4.93 倍。

Decode 优化中：

- D1 使用 $2048 \times 64$ PE array、VLEN 1024、1 个 SRAM、2 个 HBM3E、1 个 HBF，batch 为 16，TPS 为 1.44 倍；
- D2 使用 $1024 \times 64$、1 层 3D-stacked SRAM、2 个 HBM4、2 个 HBF 和两层 LPDDR5X，batch 为 32，TPS 为 2.19 倍。

**可以看到 decode 并不追求最大矩阵阵列，而是倾向更小计算资源和更大内存容量，以提高批处理和功耗效率。**

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/19.webp)

::: image-caption
图 7：从帕累托前沿中筛选出满足700W功耗约束、最大化每焦耳token数的样本。该图呈现了OSWorld任务下，预填充和解码优化配置在吞吐量（TPS）与能效（Token/J）维度的帕累托最优解集。所有样本均受700W功耗上限约束，与H100 GPU功耗持平，保证对比公平性。预填充优化样本集中在高吞吐量、中高能效区域，适配长序列批量处理；解码优化样本侧重高能效，兼顾中等吞吐量，契合KV缓存密集型场景。样本分布清晰体现两阶段优化目标的差异，MemExplorer通过多目标优化，筛选出兼顾性能与能效的差异化配置，证明异构内存架构能突破同构HBM架构的能效瓶颈，为智能体推理提供更优的硬件设计方案。
:::

上图展示了 prefill 与 decode 分别形成不同 Pareto frontier，并选出 P1、P2、D1、D2 等代表配置。**该图说明 MemExplorer 不是输出唯一答案，而是让系统设计者在性能优先和能效优先之间选择。**

> 图 8 进一步对比 A100、H100、Base 和 P1+D1 组合。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/20.webp)

::: image-caption
图 8：不同配置下的吞吐量与能效对比。P1配置延迟更低，但因计算资源少于A100和H100，延迟仍高于两者；D1配置吞吐量更高，提升解码效率；P2和D2配置优先保障能效，每焦耳token数显著提升；P1+D1组合配置实现均衡设计，预填充和解码阶段兼具低延迟与高吞吐量。预填充-解码解耦建模中，通信通道采用NVLink模拟，与LLMCompass方法一致。P1（预填充最优）、D1（解码最优）、P2（能效优先）、D2（能效优先）四类定制化配置，在吞吐量和能效上均显著优于基准NPU和H100。其中P1+D1组合实现预填充高吞吐、解码高能效的均衡表现，适配全流程智能体推理。对比GPU，MemExplorer异构NPU虽在原始延迟上略有差距，但能效提升2-3倍，且可通过解耦设计灵活匹配两阶段需求。该结果验证了异构内存架构与分阶段优化策略的有效性，为下一代智能体推理加速器提供了优于传统GPU的设计范式。
:::

论文指出 P1 配置能实现较低 TTFT，虽然由于计算资源减少仍高于 A100 和 H100；D1 配置提供更高 TPS，改善 decode 吞吐；P2 和 D2 更重视能效，tokens-per-joule 显著更高；P1+D1 组合则在 prefill 和 decode 之间取得平衡。

::: block-1
实验结果支撑了论文主张：:mdnice-bold-italic[prefill 最吃高带宽片上存储和计算吞吐，decode 最吃容量和功耗效率。 把二者拆开设计，不是工程复杂化，而是匹配负载本质。]
:::

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/21.webp)

::: image-caption
图 9：预填充和解码分阶段异构优化的性能上限对比。对比表6中的P1（紫色）和D1（绿色）配置。预填充密集型工作负载中，注意力层与前馈网络层瓶颈不同，可独立优化；解码序列较短，解码优化收益有限。解码密集型工作负载中，可将解码拆分为前期（前50%token）和后期（后50%token），两阶段内存与计算特征不同，存在额外优化空间。该图进一步探索智能体推理的极致异构优化潜力，突破传统预填充-解码两阶段划分，深入层级与解码阶段细分优化。预填充密集场景下，注意力层受带宽瓶颈制约、前馈网络层受计算瓶颈制约，独立优化可针对性分配内存与计算资源；解码密集场景下，前期解码KV缓存增长快、内存压力大，后期解码访问模式稳定、带宽需求低，分阶段优化可精准匹配资源需求。MemExplorer通过这种精细化异构设计，进一步挖掘性能与能效提升空间，证明智能体推理系统的异构化可从设备级延伸至层级、阶段级，为极致优化提供了新方向。
:::

## 八、扩展到新模型：Diffusion LLM 与大规模 MoE

> 论文没有只停留在自回归 Transformer，还进一步测试了 diffusion language model 和超大规模稀疏 MoE，说明 MemExplorer 的搜索框架可以迁移到新兴模型结构。

Diffusion Language Model 不是逐 token 自回归生成，而是通过对整个序列反复迭代去噪生成 token。推理时模型需要多次处理完整序列，因此 full-sequence computation 和 memory access 会重复发生。相比自回归模型，dLLM 的 activation footprint 更大，因为中间 activation 随序列长度增长，并且要在多个 diffusion step 中复用。因此它更依赖片上内存容量和带宽。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/22.webp)

::: image-caption
表 7：扩散语言模型LLaDA-8B不同内存配置的性能与能效对比，以基准配置为归一化参照。将所有激活值和部分KV缓存驻留片上可显著提升性能，因此预填充和解码倾向采用相似内存设计。针对扩散语言模型（dLLM）迭代去噪、全序列重复计算的特性，对比不同内存配置的表现。dLLM激活值占用空间大、需多次复用，对片上存储要求高于自回归LLM。基准配置为单层SRAM+HBM3E，批量128、能效为1；预填充优化配置采用2层3D堆叠SRAM+HBM3E，批量降至64但能效提升1.65倍；解码优化配置3层3D堆叠SRAM+HBM3E，能效提升1.33倍。实验证明，3D堆叠SRAM对dLLM性能提升至关重要，可减少重复的片外激活值传输；且因dLLM两阶段均依赖激活值复用，内存配置趋同，为MemExplorer适配新型AI模型提供了设计参考。
:::

> 表 7 以 LLaDA-8B 为例：

- Prefill Optimized 使用 2 层 3D-Stack 和 2 个 HBM3E，Token/J 达到 1.65 倍；
- Decode Optimized 使用 3 层 3D-Stack 和 2 个 HBM3E，Token/J 为 1.33 倍。

::: block-1
作者指出由于 dLLM 的 prefill 和 decode 都更倾向存储 activation，两个阶段会收敛到相似内存设计。**这一点很有启发：prefill/decode 异构不是永恒规则，而是由模型结构决定。换模型后，最优分工可能变化。**
:::

MoE 实验使用 Qwen3.5-397B-A17B，总参数 397B，但每个 token 只激活 17B 参数。**这个模型权重存储约 370GB，代表极大规模稀疏推理负载。对这种模型，HBF 频繁进入最优配置，因为它高容量、带宽中高，适合存放不频繁访问的 expert weights**。同时，MoE 中间 activation 也很大，3D-stacked SRAM 能减少 off-chip traffic 并提升复用。

> 表 8 显示，在 Qwen3.5-397B-A17B 上：

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/23.webp)

::: image-caption
表 8：大规模MoE模型Qwen3.5-397B-A17B不同内存配置的性能与能效对比，以PLENA+HBF×2基准配置为归一化参照。该表聚焦千亿级稀疏MoE模型，该模型总参数397B、单token激活17B，需超大容量存储专家权重，同时高带宽支撑专家加载。基准配置为3D堆叠SRAM+HBF×2，批量32、能效为1；预填充优化配置4层3D堆叠SRAM+HBF×2，批量达128、能效提升3.52倍，片上大容量存储大幅减少专家权重的片外访问；解码优化配置单层SRAM+HBF×1+LPDDR5X×8，批量64、能效提升1.13倍，LPDDR补充容量、降低功耗。结果显示，HBF是存储海量专家权重的最优选择，3D堆叠SRAM是降低稀疏访问带宽压力的关键，为超大规模稀疏模型的内存架构设计提供了核心方案。
:::

- Prefill Optimized 使用 4 层 3D-SRAM 和 2 个 HBF，batch size 从 32 提升到 128，Token/J 达到 3.52 倍；
- Decode Optimized 则使用 1 个 HBF 和两层 LPDDR5X，Token/J 为 1.13 倍。

这里可以看到 HBF 的价值非常明确：**对于权重巨大的稀疏模型，它不是锦上添花，而是让系统能容纳模型状态的关键容量层。**

::: block-1
MemExplorer 的更大意义在于面向模型演进。自回归 LLM、dLLM、MoE 对内存的压力完全不同，固定架构很难长期适配，而可搜索的异构内存框架更接近未来 AI Infra 的真实需求。
:::

## 九、相关工作：本文相对已有研究的差异

> 相关工作可以分成三类：内存技术、面向内存技术的软硬件协同设计，以及异构 NPU 的 PD disaggregation 服务。

### 9.1 第一类是内存技术本身

- HBM 已经广泛用于 NVIDIA V100 到 B200 以及未来 Vera Rubin 系统；
- GDDR 提供更具成本效率的选择；
- LPDDR 强调能效，已经用于 NVIDIA Grace 等数据中心平台；
- HBF 试图填补高带宽内存和传统存储之间的容量空缺；
- 3D-stacked SRAM 则通过垂直堆叠扩大片上容量。

**论文的立场不是押注某一种技术，而是承认它们各有容量、带宽、功耗和物理约束差异。**

### 9.2 第二类是内存模拟和软硬件协同

Ramulator、DRAMSys 等工具能做 DRAM 级别详细周期模拟，但主要面向传统 DRAM 技术，不覆盖 3D-stacked SRAM 或 HBF，也不直接建模 LLM 推理端到端负载和加速器执行特征。

- H3 提出 HBM-HBF 混合架构，证明分层片外内存对 LLM 推理有益，但没有建模片上内存层级和详细 traffic 模式。
- Li 等工作关注 3D-DRAM LLM serving 加速器，但聚焦特定技术，而不是统一探索多种内存技术。

### 9.3 第三类是 prefill-decode disaggregation

> DistServe、Splitwise、Mooncake 等工作将计算密集的 prefill 与 memory-bound 的 decode 分离，以减少资源干扰。

RPU 和 PLENA 等 NPU 设计探索了异构 NPU 配置，但更多关注计算异构，内存配置相对固定。**MemExplorer 的差异在于，它把计算异构和内存异构放进同一个搜索框架，尤其强调 prefill 与 decode 的内存需求差异。**

::: block-1
MemExplorer 相对已有工作的关键优势，是**它不是只做内存技术评估、也不是只做 PD 分离服务，而是把多层内存技术、NPU 计算参数和软件数据流策略联合起来做系统级搜索。**
:::

## 十、结论与展望

### 10.1 结论总结

> 论文结论明确指出，LLM 系统中不断增加的硬件加速器异构性，带来了内存系统架构协同设计的新挑战。MemExplorer 试图自动识别异构 NPU 系统中的高效内存架构。

本文的核心贡献可以概括为三点。

- 第一，提出统一内存抽象，把 SRAM、HBM、LPDDR、GDDR、HBF、3D-stacked SRAM 等技术放到同一个层级模型和功耗模型中比较。
- 第二，把 NPU 计算阵列、VLEN、内存 stack 配置、量化精度、数据流、片上存储和片外带宽分配联合搜索，而不是单独调硬件或软件。
- 第三，通过面向 agentic workload 的 prefill/decode 分阶段搜索，证明二者需要不同内存结构：prefill 受益于大容量高带宽 3D-stacked SRAM，decode 更受益于大容量、较低带宽但能效更好的片外内存层级。

实验层面：

- MemExplorer 在相同功耗预算下对 agentic workloads **相比 baseline NPU 最高实现 2.3 倍能效提升**；
- prefill-only 设置下相比 H100 最高达到 3.23 倍；
- decode 设置中在等性能目标下相比 baseline NPU 和 H100 最高分别提升 1.93 倍和 2.72 倍功耗效率。
- 对 dLLM 和 MoE 的扩展实验也说明，该框架能处理不同模型结构的内存需求差异。

### 10.2 进阶分析

> 这篇论文解决的是一个非常真实的问题，但它更像是架构空间探索框架，而不是一个已经闭环到真实芯片部署的完整系统。其方法从根本上指出了“固定内存架构无法适配 agentic 推理阶段差异”这一问题，但对实际部署来说，仍然依赖若干假设。

![文章插图](/articles/2026-05-21-h100-3-23-memexplorer-agentic-37001fba/images/24.webp)

::: image-caption
表 9：Llama-3.3-70B预填充阶段Transformer块性能验证，输入序列长度4096时，Transformer块的性能验证结果。该表通过与周期级模拟器对比，验证MemExplorer解析模型的准确性。PLENA模拟器为基准，单块Transformer模拟耗时814.14ms、运行4.15分钟；PLENA原生解析模型误差19.14%、耗时3ms；MemExplorer优化模型误差仅10.20%、耗时24ms。MemExplorer在保持高准确率的同时，兼顾了计算效率，原因是其统一内存模型精准刻画了异构层级的延迟、带宽与功耗，简化了模拟器的冗余细节，适配大规模设计空间探索需求。该验证证明MemExplorer解析模型可高效、精准评估异构NPU性能，为快速遍历百万级设计空间、锁定最优配置提供了可靠的评估工具，解决了周期级模拟器效率低下无法支撑大规模探索的难题。
:::

- 第一，论文的核心结果依赖分析模型和扩展模拟器。虽然表 9 显示作者的模型相比 PLENA emulator 误差从 19.14%降低到 10.20%，运行时间从分钟级降低到毫秒级，**但 10%左右误差对芯片架构选择并不总是可忽略**。尤其当多个 Pareto 点性能接近时，模型误差可能改变排序。
- 第二，HBF 等新兴内存技术的参数来自论文和公开资料推导，其中部分值是按 scaling factor 计算。HBF 在真实封装、可靠性、寿命、写放大、控制器复杂度、软件栈支持上的问题，并没有在本文中充分展开。**因此 HBF 在搜索结果中频繁出现，并不等价于它在近期工程上就一定可落地。**
- 第三，论文没有建模多设备共享内存系统，也没有捕获 C2C 互连和跨设备通信开销。可是 prefill-decode disaggregation 真正部署时，KV cache 迁移、调度延迟、互连拥塞和多租户干扰都可能影响端到端收益。论文使用类似 LLMCompass 的 NVLink 通信建模，但这还不是完整数据中心级服务模型。
- 第四，论文当前精度设置在 prefill 和 decode 阶段保持一致，不支持跨阶段 mixed precision。可现实中 prefill、decode、KV cache、attention 和 FFN 可能有不同精度容忍度。如果引入混合精度，搜索空间会更大，也可能改变最优内存层级选择。

### 11.3 未来工作

> 原文计划主要有三点。

- 第一，**验证框架在真实 GPU 和加速器平台上的表现**，而不仅限于分析模型和模拟器。
- 第二，**把框架扩展到具有共享内存层级的 multi-core compute systems**，并纳入 C2C 互连和跨设备通信开销。
- 第三，把 mixed-precision execution 和更多 emerging memory technologies 纳入探索。

> 从 NeuralTalk 视角看，MemExplorer 还可以往三个更有野心的方向发展。

- 第一，**和 LLM serving runtime 结合**，让系统根据实时 workload trace 动态选择 prefill/decode 分配策略，而不是只做离线架构搜索。
- 第二，**把成本模型纳入目标函数**，例如 HBM、HBF、3D SRAM 的封装成本、良率和供应链约束，这样搜索结果更接近数据中心采购决策。
- 第三，**把编译器和内存系统协同推进到更细粒度**，例如让 kernel fusion、KV cache layout、attention 分块策略直接反馈到内存层级搜索中。

::: block-1
MemExplorer 最重要的启示是：未来 AI Infra 的核心优化对象不再是单颗芯片的峰值算力，**而是模型阶段、内存层级、互连、数据流和功耗预算共同决定的系统级 Pareto 前沿。**
:::
