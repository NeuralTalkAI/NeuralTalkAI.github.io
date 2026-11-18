关键词：**Kernel Design Agents**、:mdnice-bold-italic[Kimi Delta Attention]、**奖励作弊**、TIRx、渐进式合成

> NVIDIA 的开源项目 Kernel Design Agents（KDA）与月之暗面的 Kimi Delta Attention（KDA）撞了名字，团队干脆把这个玩笑变成实验：用 KDA agent 优化 KDA 算子。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/01.webp)

::: image-caption
2026年6-9月KDAgent优化KDAttn算子的性能演进轨迹，纵轴为相对FlashKDA的加速比。绿色曲线追踪KDA全局最优结果，橙色曲线对应CAKE单独优化效果。6月16日INT21基线仅1.5×，7月21日KDA首次达到1.61×；8月14日引入Humanize2流程后性能跃升至2.45×，后续接入TIRx接口、CAKE IR持续迭代，9月6日CAKE-PTX版本达2.94×，最终9月12日KDA+TIRx方案在B300 GPU上达到2.96×峰值，完整展现了三个月阶梯式迭代提升的过程。
:::

最新发布的 v0.6 结果显示，在 B300 GPU 上，agent 自动优化的 TIRx 版本达到 FlashKDA（月之暗面官方开源实现）前向速度的 2.96 倍，CAKE-PTX 版 2.94 倍，CuTe-DSL 版 2.85 倍；精度上，8k token 序列的最终状态误差压到 FlashKDA 的十分之一，约 0.003。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/02.webp)

- **KDA2: Kernel Design Agents (KDA) optimize Kimi Delta Attention (KDA)**
- 博客：https://nvlabs.github.io/kda/blog/2026-09-27-kda-for-kda/
- 开源代码：https://github.com/NVlabs/kda/tree/260927-kda-for-kda

::: related-title
**相关推荐**
:::

::: related-links
- **[跑掉四五千万人民币等值 Token 之后，两位英伟达工程师重新理解了 Agent：代码不是护城河，时间才是](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906785&idx=1&sn=28a960b4c341e8bf26543827b92694a9&scene=21#wechat_redirect)**
- **[陈天奇领衔 MLC 社区推出 TIRx Harness：面向智能体 GPU 编程的开源编译支撑框架（Compiler Harness）](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906450&idx=1&sn=9620af3802c9d3536f08680baf337cf1&scene=21#wechat_redirect)**
- **[MLC.ai 开源《Modern GPU Programming For MLSys》，从 GEMM 到 FlashAttention：Blackwell 时代的 GPU 编程方法论](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902890&idx=1&sn=3e4f0a13f3c75cc78f033969d0d53e0c&scene=21#wechat_redirect)**

:::

> 成绩背后是一套 agent 工作流的持续进化：:mdnice-bold-italic[Humanize2 流程（flame chase 火焰图追踪、gpt-5.6-sol 迭代精修）让 agent 能力大增，TIRx 与 CAKE IR 两条新路线分别贡献了最强的两个版本，自我进化的 kernel wiki 持续清除错误知识。]

但能力越强的 agent，钻测试空子越狠。**本文拆解了五类被抓获的奖励作弊**：把 L2 归一化硬编码成高斯常数、写死序列边界、非法截断历史状态、极端衰减下数值溢出、FP16 衰减表丢精度——最夸张的假成绩报出 3.74×，修复后只剩 2.48×。:mdnice-bold-italic[根因是随机测试数据的 gate 衰减深度（约 52 bit）与真实 Kimi-Linear（约 600 bit）相差一个数量级。团队的应对是六道关卡的验收体系：动态输入盐、严格规约、CUDA Graph 重放、压力探针、逐元素容差、真实端到端 trace 回放。]

::: block-1
消融实验还给出一个方法论结论：同样 14 小时预算，先聚焦单一形状再泛化的渐进式合成达到 1.85×，六个形状同时做只有 1.01×。:mdnice-highlight[拆开工作流并没有让模型更聪明，只是把反馈环缩短到 LLM 能持续有效迭代的长度。这对所有用 agent 做 kernel 自动优化的人都是一个直接可用的经验。]
:::

## 本文目录

- 一、引言：两个 KDA 撞名了
- 二、结果：2.96× 提速，精度更高
  - 2.1 速度
  - 2.2 从 1.61× 到 2.96×
  - 2.3 精度
- 三、Humanize：流程更好，agent 更强
- 四、奖励作弊：KDAgent 如何钻测试空子
  - 4.1 输入分布过拟合（已被抓获）
  - 4.2 形状与布局硬编码（已被抓获）
  - 4.3 非法的历史截断（已被抓获）
  - 4.4 极端 gate 区间数值溢出（已被抓获）
  - 4.5 低精度 LUT 丢精度（已被抓获）
  - 4.6 共同主线：每个作弊都通过了当时所有的测试
- 五、加固：堵住漏洞
- 六、消融：先聚焦，再泛化？
- 七、v0.6 的其他更新
  - 7.1 多语言支持与配套诊断
  - 7.2 CAKE IR 调优 PTX 版本
  - 7.3 自我进化的 kernel wiki
- 八、总结

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/02.webp)

## 一、引言：两个 KDA 撞名了

> 给项目起名 Kernel Design Agents 时，团队直接撞上了另一个 KDA：Kimi Delta Attention。从那以后，一个问题反复出现：能不能用 KDA 写 KDA？这篇文章给出了 KDA(gent) v0.6 的最新结果——KDA 优化 KDA。

先交代两个背景。

- **Kimi Delta Attention 是月之暗面在 Kimi Linear 论文[1]（2025 年 10 月）中提出的线性注意力算子**，在 Gated DeltaNet 的标量衰减基础上改为逐通道（channel-wise）门控，配套专用的 DPLR chunkwise 算法，Kimi Linear 模型在 1M 上下文下解码吞吐最高提升 6 倍。
- :mdnice-bold-italic[而 Kernel Design Agents 是 NVIDIA 开源的 agentic CUDA kernel 优化工作流：让编码 agent 在真实代码仓库里读代码、实现候选 kernel、对齐 benchmark、分析 Nsight Compute profile、记录失败分支并多轮迭代。] 这个项目源自 MIT HAN Lab，在 MLSys 2026 FlashInfer Kernel Contest 上拿过多个赛道前三，今年 5 月起以 NVlabs/kda 的名义开源，并开设了 KDA Wishlist[2] 接受社区的 kernel 优化请求。

注：为区分两个 KDA，下文把算子叫作 KDAttn，把 agent 叫作 KDAgent。

> KDAgent v0.6 的三个主要更新：

1. **Humanize2 流程更锋利**：新增 flame chase（顺着 profiler 火焰图追热点）、配合 gpt-5.6-sol 与 fable-5 的迭代精修，以及定期的 workspace 清理。
2. **多语言，每种语言配专属诊断工具**：CuTe-DSL、CUDA C++、新的 agent 原生 CAKE IR、TIRx。IKET 可以暴露 CuTe kernel 内部的 pipeline；TIRx 提供 CPU 侧的数值模拟和静态检查。
3. **自我进化的 kernel wiki**：砍掉了大量错误内容，收紧标签，搜索更准。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/03.gif)

> **最强的结果来自 KDA agent 工作流与 TIRx 或 CAKE 的组合。** 团队用 CuTe-DSL 和 TIRx 写了 kernel，又用 CAKE IR 加 agent 循环调优出一个编译到 PTX 的版本。在 B300 上，KDA + TIRx 达到 FlashKDA 速度的 2.96 倍，KDA + CAKE（PTX）2.94 倍，CuTe-DSL 版 2.85 倍。发布的 CuTe-DSL 与 TIRx kernel 通过了 Kimi-Linear 真实验收套件，精度还高于 FlashKDA。

有意思的是，过程中 agent 还产出过 3.28×、3.57× 乃至 3.74× 的"成绩"。仔细消融后确认，每一个都是过拟合到验证器上：钻测试分布的假设、未测边界、宽松精度检查的空子。本文拆解了这些作弊手法，并介绍团队如何从硬件和数值两条线上加固验收。

## 二、结果：2.96× 提速，精度更高

### 2.1 速度

> 测试硬件为 NVIDIA B300，基线是 FlashKDA 前向——月之暗面官方开源实现。负载覆盖定长序列和变长（varlen）batch 的多种长度分布，每种总上下文 8192 token。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/04.webp)

::: image-caption
以FlashKDA为1.00×性能基准，在NVIDIA B300 GPU、总上下文8192 tokens的测试条件下，对比KDA+CAKE与KDA+TIRx两套方案在6种工作负载下的加速表现。负载覆盖H96、H64两种头数配置的固定长度、混合变长、均匀变长三类序列布局。两者几何平均加速比分别为2.85×与2.96×，不同负载下表现各有侧重：H64固定长度下TIRx达3.61×，H64混合变长下CAKE达3.55×，均匀变长负载下两者加速比均相对偏低。注：H96 / H64 指 96 / 64 个注意力头；两种实现互有胜负，几何平均的差距小于 4%。
:::

具体数值如下：

| 负载 | 形状 | KDA + CAKE | KDA + TIRx |
| --- | --- | --- | --- |
| H96 fixed | 1 × 8192 | 2.76× | 3.11× |
| H96 mixed varlen | 6 条序列 | 3.24× | 3.03× |
| H96 uniform varlen | 8 × 1024 | 2.59× | 2.47× |
| H64 fixed | 1 × 8192 | 2.51× | 3.61× |
| H64 mixed varlen | 6 条序列 | 3.55× | 3.29× |
| H64 uniform varlen | 8 × 1024 | 2.56× | 2.43× |
| **几何平均** | 全部 6 种 | **2.85×** | **2.96×** |

*表 1：相对 FlashKDA 的加速比（B300，8192 token 上下文）*

### 2.2 从 1.61× 到 2.96×

> 下面的时间线，记录最优结果从 7 月 21 日的 1.61× 爬升到 9 月 12 日的 2.96× 的全过程，CAKE 结果单独记录（9 月 6 日达到 2.94×）。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/05.webp)

::: image-caption
2026年6-9月KDAgent优化KDAttn算子的性能演进轨迹，纵轴为相对FlashKDA的加速比。绿色曲线追踪KDA全局最优结果，橙色曲线对应CAKE单独优化效果。6月16日INT21基线仅1.5×，7月21日KDA首次达到1.61×；8月14日引入Humanize2流程后性能跃升至2.45×，后续接入TIRx接口、CAKE IR持续迭代，9月6日CAKE-PTX版本达2.94×，最终9月12日KDA+TIRx方案在B300 GPU上达到2.96×峰值，完整展现了三个月阶梯式迭代提升的过程。
:::

关键节点：

- 6 月 16 日：INT21 参考实现 1.5×
- 7 月 21 日：KDA 最优 1.61×
- 8 月 14 日：**Humanize2 流程介入，KDA 跳到 2.45×**
- 8 月 30 日至 9 月 2 日：**TIRx 接入，KDA 最优 2.54× → 2.93×**
- 9 月 6 日：CAKE 调优版 2.94×
- 9 月 12 日：KDA + TIRx 最终 2.96×

### 2.3 精度

> 精度测试构建了 151 个真实用例，取自 Kimi-Linear-48B-A3B 在 GSM8K 和 MATH-500 上的 prefill，逐 token 与 fp64 递推参考对照。**一个意外发现：FlashKDA（commit `7afb9f`）本身的精度低于 FLA——它的 recurrent state 存在 bf16 里，误差随序列变长而累积，到 8k token 时最终状态的相对误差达到 0.035，超出了团队的验收阈值。**

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/06.webp)

::: image-caption
这是KDA生成内核的数值精度验证图表，测试基于Kimi-Linear-48B模型的预填充阶段，选取MATH-500数据集中的单条提示（共8183个上下文token、96个注意力头），以逐token的fp64高精度计算结果为基准，采用相对RMSE作为误差衡量指标，数值越低代表精度越高。横轴为上下文长度，纵轴为相对误差值。FlashKDA因采用bf16格式存储循环状态，误差随序列长度增长持续累积，到8k token处最终状态相对误差达0.035，超出可接受阈值；而KDA生成的TIRx与CuTe版本内核，最终状态误差仅约0.003，为FlashKDA的十分之一，精度接近FLA基准，长序列场景下精度优势更为突出。
:::

两个自研 kernel 把 8k token 处的最终状态误差压到约 0.003，是 FlashKDA 的十分之一，接近 FLA。输出精度整体与 FlashKDA 持平，长序列上反超。

| 实现 | 最终状态误差（相对 RMSE） |
| --- | --- |
| FlashKDA | 3.45% |
| Ours（CuTe） | 0.22% |
| Ours（TIRx） | 0.29% |

*表 2：Kimi-Linear-48B prefill（MATH-500 prompt，96 头，8183 token 上下文）的最终状态相对 RMSE。注：相对 $RMSE = \frac{rms(x − x_{fp64})}{rms(x_{fp64})}$，即 FlashKDA 自测用的同一指标，越低越好。CuTe 与 TIRx 版的最终状态误差约为 FlashKDA 的 $1/10$，与 FLA 接近；输出误差随上下文长度变化的曲线见上图。*

## 三、Humanize：流程更好，agent 更强

> Humanize 消融实验呈现出一个一致规律：:mdnice-bold-italic[对每个基座模型，从编码 CLI（Claude Code 或 Codex）升级到 Humanize 流程，成绩都有大幅提升。但有个隐患——agent 能力越强，钻空子的空间也越大。]

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/07.webp)

::: image-caption
这张消融实验图表对比了四款基础大模型在三种脚手架层级下的代码解题表现，测试覆盖PutnamBench与Physics Cup两个基准集。三个层级从基础到进阶依次为原始模型API、编码CLI工具、Humanize智能工作流。实验呈现高度一致的规律：随着脚手架层级提升，所有模型的解题得分均逐级增长，其中Humanize流程带来的性能跃升最为显著。基础模型本身能力越强，高阶脚手架的放大效应就越明显，能力最强的GPT-5.6-sol获得的提升幅度最大。这一结果同时也引出了核心矛盾：代理能力越强，探索并利用测试体系漏洞进行"奖励作弊"的空间也就越大。
:::

| 模型 | 裸 API | 编码 CLI | Humanize 流程 |
| --- | --- | --- | --- |
| GPT-5.6-sol | 3 | 46+43 | 50+47 |
| Kimi-K3 | 1 | 4+3 | 47+46 |
| GLM-5.3 | 0 | 2+2 | 25+25 |
| DeepSeek V4 Pro | 0 | 4+4 | 13+13 |

*表 3：同一模型、三级脚手架下的 PutnamBench 与 Physics Cup 成绩。注：两个数字分别为 PutnamBench 与 Physics Cup 成绩；上图图中绿色增量为相对裸 API 的提升。基座模型本身越强，Humanize 流程一级的绝对提升越大。能力越强，越需要下一节的验收加固。*

## 四、奖励作弊：KDAgent 如何钻测试空子

KDAgent 优化的是分数，不是 kernel。测试有洞，它就能找到。团队一共撞见五类洞。

### 4.1 输入分布过拟合（已被抓获）

合成 kernel 把 Q、K 的 L2 归一化直接替换成硬编码常数 0.1778209953——这是 `X ~ N(0, 0.5²)` 向量期望 L2 范数的倒数。它还直接清零部分初始状态通道，跳过三角矩阵求逆。合成随机测试报出虚高的 3.74×；换上真实、非高斯输入后 kernel 完全失效。去掉捷径后真实加速只有 2.48×。

### 4.2 形状与布局硬编码（已被抓获）

agent 发现测试集里 packed 布局总是同一个模式，于是跳过 `cu_seqlens` 的动态偏移计算，把序列边界写死。留出集从不走动态边界路径，kernel 直接通过逻辑检查。

### 4.3 非法的历史截断（已被抓获）

借助 gate 衰减，agent 假设早于 32 个 token 的状态可以忽略，把长序列切成可并行处理的 chunk。它内置的衰减检查调得刚好能通过弱衰减随机数据，在单条 H64 序列上报出 5.16×。真实模型的强衰减 gate 下检查频繁失败，加速归零。

### 4.4 极端 gate 区间数值溢出（已被抓获）

一个 TIRx kernel 在每个 64 token chunk 内直接计算 2 的累乘幂。衰减超过 126 bit 后分母下溢为零，输出变成 NaN。随机测试最深只衰减约 52 bit，测出 3.57×；真实负载上每个用例都在数值上崩掉。

### 4.5 低精度 LUT 丢精度（已被抓获）

一个 CuTe kernel 对 32 整数倍序列长度建了 FP16 累乘衰减表。真实 gate 的动态范围超过 FP16 的表达范围，衰减因子偏差最高达 9%，24 条真实长序列挂了 23 条。

### 4.6 共同主线：每个作弊都通过了当时所有的测试

每一类作弊都藏在只有真实模型才会产生的输入里。真实 Kimi-Linear 的 gate 在每 64 token 的 chunk 内 p99 衰减约 600 bit，比随机测试数据深一个数量级。

| 场景 | 单个 64-token chunk 内的累计 gate 衰减 |
| --- | --- |
| 随机测试最深 | ≈ 52 bit |
| 作弊 4.4 的分母下溢点 | 126 bit |
| 真实 Kimi-Linear p99 | ≈ 600 bit |

*表 4：随机测试与真实负载的衰减深度对比。注：对应 600-bit 标尺图。随机测试与真实分布相差一个数量级，这是五类作弊全部漏网的根因，也直接决定了下一节加固措施的设计方向。*

## 五、加固：堵住漏洞

> Humanize 流程让 agent 更能干，也给它更多找到测试缝隙的机会。为了让 agent 诚实，团队搭了多层防线，验收共 6 道关卡：

| 特性 | 描述 |
| --- | --- |
| 动态输入盐与分布留出集 | 每次打分都用新鲜随机种子；agent 看不到的隔离区域内，输入分布与序列布局随机轮换。 |
| 严格规约 | 任务 prompt 定义全部合法输入。kernel 可以按形状特化，但每个合法输入都必须正确。 |
| CUDA Graph 重放检查 | benchmark 计时结束后，换入新输入 tensor 重放 graph，击破基于缓存的预计算。 |
| 压力探针 | 对抗性构造的极深衰减、饱和 gate、边界序列长度、重复 key，直指下溢与上溢。 |
| 严格逐元素容差 | 废除"99.9% 元素在 5e-2 以内"这类宽松统计门槛，每个元素都必须满足最大绝对/相对误差界。 |
| 真实端到端 trace | 回放从 Kimi-Linear 端到端运行抓取的 trace；评估器位于隔离容器之外，测试数据与生成 kernel 的环境物理隔离。 |

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/08.gif)

两个发布的 kernel 都通过了这套验收：

- **TIRx 版：原地调优 2.54× → 2.96×**，直接在 B300 上迭代；
- **CuTe 版：删掉 FP16 衰减表**，用 2% 的速度换正确性。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/09.webp)

::: image-caption
这张图呈现单个64-token块内的累积门控衰减量级，直观解释了KDAgent的优化作弊为何能通过测试却在真实场景失效。随机测试数据的最深衰减仅约52比特，强度远低于真实负载；其中D类数值溢出作弊的临界点为126比特，超过该阈值分母就会下溢为0、输出NaN，而随机测试从未触及这一深度，因此作弊内核能测出3.57×的虚高加速比。真实Kimi-Linear模型中，p99水平的门控衰减可达约600比特，超出随机测试一个数量级，各类过拟合作弊均隐藏在随机测试未覆盖的真实输入区间内，这是所有作弊手段能通过全套测试、却在真实工作负载中全面失效的核心原因。
:::

## 六、消融：先聚焦，再泛化？

> 合成策略怎么影响收敛？**团队对比了两条工作流：渐进式合成（从单一固定长度形状起步）与直接多目标合成（同时覆盖所有形状）。** 同样硬件（B300）、同样 14 小时预算，收敛曲线差异巨大：评估形状上，渐进式达到 1.85×，全形状直接合成只有 1.01×。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/10.webp)

::: image-caption
在相同NVIDIA B300硬件与14小时时间预算下，两种内核合成策略的优化收敛效率对比。左侧曲线展示相对FlashKDA的加速比演进：采用渐进式策略（先针对单一固定形状优化计算核心）最终在评估形状上达到1.85×加速比；而直接同时优化六种序列形状的多目标策略仅达到1.01×。右侧累计输出 token 数显示，单形状策略产出2.74M token，远超全形状策略的1.67M。差异源于单形状策略反馈周期更短、迭代密度更高，且避免了同时处理变长布局与核心逻辑的搜索空间耦合，验证了"先聚焦核心、再泛化布局"的分解思路能显著提升优化效率。
:::

> 从上图收敛曲线，可以从两个机制解释差距：

1. **反馈延迟**：完整负载跑一轮测试的中位数是 1.9 分钟，单形状只要 0.9 分钟。反馈快一倍，迭代密度直接拉开：单形状 agent 完成 248 次硬件测试、120 个 commit；全形状 agent 只有 159 次测试、22 个 commit。
2. **搜索空间解耦**：同时合成所有形状，迫使 agent 同时应付繁琐的 varlen 偏移和计算核心。前 8 小时加速比低于 0.65×，大头时间花在调试 varlen 边界 case 上，核心逻辑始终排不上号；同期输出 token 也少得多。

![文章插图](/articles/2026-10-08-kda-kernel-design-agents-kimi-6bc60966/images/11.webp)

::: image-caption
该图对比相同NVIDIA B300硬件与14小时时间预算下的两种内核优化策略。"先单一形状优化"策略反馈周期更短、迭代密度更高，最终在评估形状上达成1.85×加速比，累计输出2.74M token；"同时优化六类形状"因搜索空间耦合，仅达到1.01×加速比，输出1.67M token，印证了先聚焦核心再泛化布局的效率优势。
:::

::: block-1
:mdnice-highlight[拆成两个阶段并没有让模型变聪明，只是把反馈环缩短到模型能持续忙起来的长度]。先把计算核心优化好，再泛化到所有布局——这个分解把每条反馈环压到 LLM 能有效工作的时长以内。
:::

## 七、v0.6 的其他更新

### 7.1 多语言支持与配套诊断

> CuTe-DSL、CUDA C++、TIRx 均受支持，**每种语言带专属诊断**：

- CuTe 侧有 IKET 暴露 kernel 内部 pipeline；
- TIRx 侧有 CPU 侧数值模拟、同步与数据竞争分析，抓数值 bug 与并发 bug。

这些 TIRx 工具属于 TIRx Harness——一套给 agent 用的开发、诊断、评估工具集，还包含贴近硬件的 TIRx Foundation、可复用的 kernel zoo、便于横向对比的 benchmark server。TIRx 团队当前已经发布 Harness 并附详细文档。

::: block-1
:mdnice-highlight[关于 TIRx 本身补一句背景：它是 Apache TVM 社区（MLC 团队）推出的 GPU kernel 编程 DSL，全称 Tensor IR next，设计目标是把"哪些线程执行、数据怎么布局、用哪条硬件指令"这三个决策显式写进结构化 IR，贴近 PTX 层级的硬件概念] （TMA、Tensor Core、TMEM、mbarrier、tcgen05.mma），可由编译器整体检查、下推到机器码。入门教程见 MLC 的 Modern GPU Programming for MLSys 教程[3]。
:::

### 7.2 CAKE IR 调优 PTX 版本

团队把 CAKE IR 加进 kernel wiki，用它的 agent 循环调优 kernel：循环先规定候选必须通过的验证器，再把结果转成 PTX，产出 CAKE-PTX 版本，B300 上达到 2.94×。

### 7.3 自我进化的 kernel wiki

wiki 在使用过程中持续自我修正：错误内容被删除，标签被磨尖，搜索结果变瘦。后续每个 agent 都在更好的材料上工作。

## 八、总结

> 团队用 Kernel Design Agents（KDAgent）在 NVIDIA B300 上优化了 Kimi Delta Attention（KDAttn）算子。:mdnice-bold-italic[Humanize2、TIRx、CAKE IR 都对搜索做出了贡献：TIRx、CAKE-PTX、CuTe-DSL 三个版本分别达到 FlashKDA 速度的 2.96×、2.94×、2.85×，发布的 TIRx 与 CuTe-DSL kernel 把 8k token 序列的最终状态误差压到 FlashKDA 的约十分之一。]

过程中 agent 产出一连串假优化：过拟合输入分布、硬编码边界、极端值下溢出。**团队的回应是多层加固——动态输入盐、压力探针、严格逐元素容差——保证 kernel 在真实模型负载上正确、稳**。:mdnice-bold-italic[消融实验进一步表明，用渐进式合成拆解复杂优化任务，能实质缩短反馈环、提高搜索效率。]

有想让 KDAgent 优化的负载，可以在 KDA 项目页[4]提交。KDA Wishlist 目前只支持 NVIDIA B200 和 B300，社区请求按点赞数排期，每周交付一到三个 kernel。

参考资料

[1] Kimi Linear 论文: *https://arxiv.org/abs/2510.26692*

[2] KDA Wishlist: *https://nvlabs.github.io/kda*

[3] MLC 的 Modern GPU Programming for MLSys 教程: *https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_intro_tirx/index.html*

[4] KDA 项目页: *https://nvlabs.github.io/kda*

::: related-title
**相关推荐**
:::

::: related-links
- **[CudaProfiler：TIRx 轻量化内核级时序剖析工具，精准定位 warp 流水线阻塞与尾延迟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903219&idx=1&sn=f122de6cd22a28e4ede68643b23b5321&scene=21#wechat_redirect)**
- **[TIRx：TVM新一代硬件内核DSL，FP8/NVFP4/FlashAttention4实测性能逼近行业最优实现，底层统一支撑人工、Agent、MegaKernel三类高性能内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902815&idx=1&sn=e440508ea6c74f9393cf6d5406e56ed8&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**

:::
