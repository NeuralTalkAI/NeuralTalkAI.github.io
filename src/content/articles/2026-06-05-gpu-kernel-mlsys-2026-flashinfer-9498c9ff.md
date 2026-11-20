**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

> 科普：文章在公众号A标原创后，其他号转载需A许可，但A大多并非原作者，著作权归投稿创作者。【投稿】目的是扩大影响力，不愿被单一公众号绑定，故而投稿文稿【不标注原创】，望周知！本文来自投稿。

关键词：**GPU Kernel 自动生成**、代码智能体、FlashInfer、:mdnice-bold-italic[MLSys2026]、**Harness Engineering**

> 本文基于 MLSys2026 FlashInfer AI Kernel 生成竞赛实践，探究大语言模型驱动 GPU CUDA/Triton 内核自动化生成技术，团队提交的 **Agent-Assisted 人机辅助方案获 Track A（Fused MoE）、Track C（Gated Delta Net）两项季军，Full-Agent 全自主方案获 Track C 亚军**。业内技术路线分为 LLM 微调训练与智能体迭代搜索两类，团队落地两套差异化架构：

- 人机协同方案依托 Harness 工程搭建四层闭环评测链路，由人类把控优化方向与晋级规则；
- 全自主方案基于 LoongFlow 实现规划 - 执行 - 评测 - 总结 - 存储全自动搜索。

::: block-1
B200 GPU 实测显示，**人机协同方案全部算子性能优于 FlashInfer 基线，DSA Attention 最高提速 29.68 倍；全自主方案仅部分算子实现有效提速，MoE FP8、GDN Decode 等算子性能落后基准**。优化轨迹表明性能呈跳跃式提升，而非稳步增长。文章最后从模型级优化闭环、标准化实验校验、负载知识库搭建三方面提出未来研究方向。
:::

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/01.jpg)

- **基于代码智能体的 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结**
- 作者 yue shui，https://www.zhihu.com/people/yue-shui-ai
- 作者主页：https://github.com/syhya

::: related-title
**相关推荐**
:::

::: related-links
- **[从"切多大、留什么、按什么顺序跑"，重新定义编译器后端计算图调度：MLSys 2026 系统赛 A 赛道冠军方案浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901589&idx=1&sn=8f35478cdabeab5cffeba7d704a30009&scene=21#wechat_redirect)**
- **[AKO：Agentic Kernel Optimization，使用 Claude Code 在几小时内优化出超越 FlashInfer 专家实现的 Kernel](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901871&idx=1&sn=8d7495711c82a26ed7d8bf0a79cb6d26&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::

> 我最近参加了 MLSys 2026 - NVIDIA Track: FlashInfer AI Kernel Generation Contest（FlashInfer Contest, 2026a[1]）。这篇博客并不是一篇关于 CUDA kernel 优化技巧的教程，**我本身并不是 GPU 算子开发专家（**；参加这次比赛的主要目的，是想借助一个高度可验证、反馈明确的任务环境，研究如何让 Coding Agent 在持续闭环中产出高质量的 GPU kernel。

完整材料分为两份报告：

- **Harness Engineering for LLM-Driven GPU Kernel Generation** （Shui et al., 2026[2]）：https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf
- **Full-Agent Kernel Generation for FlashInfer** （Ma et al., 2026[3]）：https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf
- 代码公开在 mlsys26-flashinfer-contest[4]：https://github.com/syhya/mlsys26-flashinfer-contest

## 本文目录

- 一、研究背景
  - 1.1 KernelBench
  - 1.2 FlashInfer-Bench
  - 1.3 本文方法
- 二、方案架构
  - 2.1 Agent-Assisted
  - 2.2 Full-Agent
- 三、实验结果
  - 3.1 团队名次与性能数据
  - 3.2 Agent-Assisted 优化轨迹
  - 3.3 Full-Agent 优化轨迹
- 四、未来工作
  - 4.1 模型级优化闭环
  - 4.2 实验管理与独立验证器
  - 4.3 负载特化与可检索记忆
- 总结

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/02.jpg)

交流加群请在

NeuralTalk 公众号

后台回复：

加群

## 一、研究背景

> 大语言模型生成 GPU 内核的难点，不只是写出一段看起来合理的 CUDA/Triton 代码。候选实现必须同时满足语义正确、能够编译、覆盖目标输入形状，并且在真实 GPU 上比已有实现更快。

### 1.1 KernelBench

**KernelBench** （Ouyang et al., 2025[5]）构建了一个评估框架，让 LLM 读取原始 PyTorch 参考实现，生成自定义内核，然后同时评估编译、正确性和运行性能。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/03.png)

::: image-caption
图1：KernelBench评测流程。模型接收一段PyTorch运算任务，编写自定义内核实现代码，随后从功能正确性与运行时延两个维度完成评测。左侧原始 Model 为基准 PyTorch 模块，forward 函数依赖原生 PyTorch 算子；依托任务指令与原模型结构，生成嵌入自定义 Inline CUDA 内核的新 ModelNew，新模型混合原生算子与自研 CUDA Kernel。优化后分两步验证，正确性校验通过多组随机输入，对比原 Model 与 ModelNew 前向输出是否一致；性能测试则多次压测，将定制 CUDA 实现同 PyTorch 原生 Eager、Torch.Compile 两种运行模式做耗时对标，兼顾数值正确性与运行加速效果，完成算子优化闭环验证。
:::

### 1.2 FlashInfer-Bench

**FlashInfer-Bench** （Xing et al., 2026[6]）进一步把这个问题放到大模型推理服务的真实负载分布中，强调执行轨迹、评测、候选实现和部署之间的闭环。它不是单独的微基准测试，而是要求候选内核在真实推理分布、统一轨迹描述格式和正确性评测下被检验。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/04.png)

::: image-caption
图2：FlashInfer-Bench架构。依托LLM智能体与人类专家协同产出内核方案，汇总内核定义、算子方案、跑分数据存入FlashInfer Trace，落地生成FlashInfer-Bench实测数据集；数据集提炼真实业务负载接入LLM推理引擎，经由`flashinfer_bench.apply()`自动筛选最优内核部署上线，线上真实负载再反向回流扩充数据集。跑分结果同步汇总至性能榜单，持续反哺内核迭代，构建从方案生成、基准测试到工程落地的数据闭环，持续迭代优化推理内核性能。
:::

### 1.3 本文方法

从方法分类看，本文参考综述 **Towards Automated Kernel Generation in the Era of LLMs** （Yu et al., 2026[7]）的整理，将相关工作概括为 LLM4Kernel 和 Agent4Kernel 两类路线。

#### 路线1：LLM4Kernel

> **LLM4Kernel 基于高质量领域语料，结合 CPT、SFT 和 RL 等训练技术**，持续优化模型能力，使其更擅长理解内核开发场景并生成高质量内核代码。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/05.png)

::: image-caption
图3：LLM4Kernel通过数据构建、有监督微调、强化学习与领域适配，预训练基座大模型依托指令数据集开展 SFT 监督微调，还可基于偏好数据集选配 RL 强化学习优化，最终产出 Kernel 专用指令大模型。推理阶段依托内核任务描述，由模型批量生成候选 CUDA 内核代码，产出内容分两路校验：一方面接入基准数据集做性能 Benchmark 评测，另一方面执行单元测试核验计算正确性，通过双维度验证筛选可用优化内核，形成从模型微调、代码生成到功能与性能核验的完整自动化管线。
:::

这条路线的优势是可以把内核知识内化到模型参数中，但它通常需要高质量训练数据、稳定的奖励设计和较高训练成本。

#### 路线2：Agent4Kernel

> **Agent4Kernel** 则更强调迭代搜索、外部记忆、多智能体编排和自动化评测。

**我的参赛方案更接近这条路线：不训练一个新模型，而是设计一个让现有代码智能体能持续试错、记录经验并优化实验脚手架。**

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/06.png)

::: image-caption
图4：Agent4Kernel围绕内核优化，重点聚焦迭代精调、进化搜索、外部存储器、硬件性能剖析以及多智能体协同调度。顶层包含迭代精修、进化算法两类学习机制，持续迭代优化内核方案；主体采用单主 Agent 搭配 Coder、Tester、Profiler、Planner 组成多智能体协作集群。Agent 可双向检索外部存储的代码案例库与结构化知识，智能体生成内核送入硬件性能剖析模块，硬件侧输出硬件规格、性能指标与剖析反馈回传系统；经测试调优后的最优内核再落回外部知识库沉淀，形成算法迭代、多角色分工、硬件反馈、知识库沉淀的全闭环自动化内核优化链路。
:::

这种思路也和 Meta 提出的 **KernelEvolve** （Liao et al., 2025[8]）这样的工业系统相呼应。:mdnice-bold-italic[KernelEvolve 强调持久化知识库、检索增强的提示构造、跨硬件编程抽象]，以及面向真实生产算子的持续优化。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/07.jpg)

::: image-caption
图5：KernelEvolve系统总览。该系统融合持久化存储、检索机制、进化搜索与硬件适配评估，实现智能内核编码在量产算子场景下的规模化落地。架构图分为上下两层，完整呈现 Triton Kernel 自动化生成全链路：下层从用户 Prompt 启动树形搜索，经多类大模型完成内核生成，依次通过 TritonBench 精度核验、多平台工具性能评测，更新持久知识库后由 Prompt 合成器输出最终内核；上层依托自演化状态机驱动树搜索分支，区分正误内核节点，代码生成器对接 MTIA、NVIDIA、AMD 三类硬件分析工具，评测数据汇入动态提示词模块，调度内外多系列 LLM，评测结果、内核源码与硬件知识统一存入元数据仓库与持久知识库，实现评测反馈、知识沉淀与迭代生成的闭环。
:::

::: block-1
GPU 内核智能体的难点是让它在复杂硬件、复杂负载和严格验证约束下保留有效经验，并把失败反馈压缩成下一轮可执行的搜索约束。
:::

## 二、方案架构

> LLM CUDA 队伍分别提交了 Agent-Assisted 和 Full-Agent 两套路线。它们的区别不在于是否使用 LLM，而**在于人类是否持续介入搜索过程。**

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/08.png)

::: image-caption
该表格从人类介入、搜索方式、状态管理三个维度，对比 Agent-Assisted 辅助智能体与 Full-Agent 全自主智能体两种内核优化模式。辅助方案中人类全程深度参与，负责策略设计、优化筛选，依托经验与性能结论做局部定向优化，项目状态依靠人工文档归档维护；全自主模式仅需人类下发初始任务、约束条件与工具，智能体自主循环完成规划、执行、评测、沉淀的全链路长程搜索
:::

### 2.1 Agent-Assisted

> Self-Evolving Agents[9] 中讨论过 **Harness Engineering** （Lopopolo, 2026[10]；Rajasekaran, 2026[11]）的核心范式：人类负责设计约束、构建反馈机制并定义评估标准，Agent 在受控环境内迭代生成更高质量的代码。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/09.jpg)

::: image-caption
图6 面向B200内核优化的智能体辅助闭环测试框架。该工作流依托算子定义、负载分布、参考基准、性能采样信号以及明确的择优迭代策略完成智能体的基础配置。 Shui et al., 2026
:::

> Agent-Assisted harness 主要分成四层：

- **Grounding inputs**：算子定义、参考实现和 workload JSON 等必要上下文输入。
- **Shape discovery**：从 workload 中根据 batch size 和 sequence length 等参数进行分组，并从每组抽取代表性维度，使 agent 可以快速评估和迭代候选，而不需要每次都做全量验证。
- **Closed-loop optimization**：在 *baseline → profile → diagnose → generate → evaluate → archive* 这条循环里生成候选，验证代码能否编译、结果是否正确，并评估性能；同时利用 Torch Profiler 和 NVIDIA Nsight Compute (NCU) 等工具分析算子瓶颈。
- **Outputs**：归档相关代码和性能指标等文件，后续提供给 agent 继续迭代。

人类编写优化 skills[12]、构建评测脚手架[13]。这些做法与 **Agent Skills** （OpenAI, 2026a[14]）和 **Subagents** （OpenAI, 2026b[15]）在上下文复用、工具封装和并行探索上的工程动机相似；在本文场景中，它们也帮助把搜索过程约束在可验证的闭环里。

### 2.2 Full-Agent

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/10.jpg)

::: image-caption
图7 改进版LoongFlow全智能体架构。智能体依次循环执行规划、代码生成、效果评估、总结归纳与数据库更新流程，将失效的候选方案转化为可供后续迭代检索的上下文信息。Ma et al., 2026
:::

Full-Agent 路线使用框架 **LoongFlow** （Wan et al., 2025[16]）的计划-执行-总结范式，其类似 **OpenEvolve** （Sharma, 2025[17]）这类演化搜索系统。它把一次内核搜索拆成规划、执行、评测、总结和存储五步，并把每个候选的来源、性能结果和失败原因等摘要写入持久数据库。

## 三、实验结果

### 3.1 团队名次与性能数据

> 官方公布的 Top-3 的 leaderboard 中，我们团队结果如下：

| 官方赛道 | 方法 | 排名 |
| --- | --- | --- |
| Track A Fused MoE | Agent-Assisted | 3rd |
| Track C Gated Delta Net | Agent-Assisted | 3rd |
| Track C Gated Delta Net | Full-Agent | 2nd |

下面的数据来自 Modal 平台 B200 GPU 上的本地评测，结果只适合作参考，因为开发环境的时钟频率不能完全锁定，最终评测运行在 bare-metal 机器上。

评测方法参考 **FlashInfer** （Ye et al., 2025[18]）和 **FlashInfer-Bench** （Xing et al., 2026[19]）的正确性门控评测设定。下表采用论文中的 matched comparison 口径：延迟为平均毫秒，PyTorch 加速使用对应的 PyTorch reference mean，FlashInfer 加速使用官方提供的 FlashInfer baseline。简化地说，表中的加速比为：

$$
\mathrm{Speedup} = \frac{\mathrm{mean\ baseline\ latency}}{\mathrm{mean\ solution\ latency}}.
$$

> 该表用于分析本地保留产物的优化幅度，并不是官方逐算子或逐赛道 contest score：

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/11.png)

::: image-caption
该表格统计DSA Attention、DSA Indexer等五类算子在Agent-Assisted、Full-Agent、FlashInfer基线三种方案的运行延迟与加速倍率，加速分别对标PyTorch原生与FlashInfer基准。整体上Agent-Assisted辅助优化方案全算子延迟最低、提速最优，DSA Indexer相对PyTorch最高提速494.13倍；Full-Agent全自主智能体性能普遍弱于辅助方案，MoE FP8场景甚至慢于FlashInfer基准，仅部分算子优于PyTorch。FlashInfer作为性能基准，多数场景性能介于两种智能体方案之间，印证人工参与引导的Agent-Assisted在内核优化上收益更突出。
:::

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/12.png)

::: image-caption
图8：该柱状图展示Agent-Assisted方案相较FlashInfer基线在五类大模型算子的加速倍数，结果基于本地Modal B200设备运行的平均时延测算得出，纵轴为性能提速倍率。DSA Attention优化收益最高，达到29.68倍；DSA Top-k索引算子次之，提速18.05倍；GDN Prefill取得13.70倍加速；MoE FP8、GDN Decode提升相对平缓，分别为1.62倍与1.12倍。整体可见注意力与索引类算子优化空间最大，稀疏MoE、解码阶段算子受硬件与访存约束优化幅度偏低，人工辅助智能体针对访存密集型算子的优化收益最为显著。Shui et al., 2026
:::

::: block-1
- Agent-Assisted 在五个算子上全部优于 FlashInfer baseline，加速比从 `1.12×`（GDN Decode）到 `29.68×`（DSA Attention）。
- Full-Agent 在 DSA Attention、DSA Indexer 和 GDN Prefill 上也找到有效候选，但在 MoE FP8 与 GDN Decode 上低于 baseline。
:::

### 3.2 Agent-Assisted 优化轨迹

> 下图展示了 Agent-Assisted 优化轨迹，相较于给定的FlashInfer基准方案所保留的加速比变化曲线。**曲线呈现出较长的平稳平台段与间断式跃升特征，并非平滑单调上升。**

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/13.png)

::: image-caption
图9：五张子折线图，分别展示MoE FP8、DSA Top-k、DSA稀疏Attention、GDN Decode、GDN Prefill五类算子随迭代版本提升、相对FlashInfer基线的加速变化，虚线为1倍性能基准，圆圈标注关键优化拐点、红星是最优留存版本。各算子依托分段Wmma、专家打包、CuTe预处理、分块调度等专项优化，性能随迭代持续抬升；DSA稀疏Attention最终提速29.68倍为全组最优，DSA Top-k、GDN Prefill依次达到18.05倍、13.70倍，MoE与GDN Decode优化上限偏低，仅小幅超越基线，直观体现不同算子的优化空间与关键内核优化手段带来的性能跃升。Shui et al., 2026
:::

::: block-1
从轨迹可以看到，**性能提升并不是平滑发生的，而是长期平台期之后出现少数几次明显跃迁**。:mdnice-bold-italic[有效的 Agent-Assisted kernel 优化不是单纯依赖提示词，而是依赖可测量的系统闭环]：把算子约束、评测脚手架、性能分析反馈和历史轨迹组织成可复用流程，再让智能体在其中生成、验证和保留候选。**这个过程需要人类持续设计和维护 harness。**
:::

### 3.3 Full-Agent 优化轨迹

> 下图是基于 LoongFlow 追踪日志得到的全智能体优化变化曲线。灰色圆点代表通过正确性校验的候选方案，实线为迭代过程中的当前最优结果，虚线代表FlashInfer基准方案。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/14.png)

::: image-caption
图10：该组五张子图展示Full-Agent全自主智能体迭代优化五类算子时，相对FlashInfer基线的性能变化，横轴为迭代轮次，虚线是1倍基准，实线为迭代最优性能、灰点是单次有效候选、红星标记最优迭代点。DSA Attention优化效果最优，29轮迭代最高提速14.54倍；DSA Indexer经16轮迭代达到3.81倍；GDN Prefill小幅超出基线至1.03倍；MoE FP8、GDN Decode全程未超越基准，最优仅分别为0.32倍、0.83倍。整体可见全自主模式在注意力类算子优化收效可观，但MoE、解码算子优化受限，多数候选方案性能波动大、大量候选劣于基线。 Ma et al., 2026
:::

::: block-1
Full-Agent 轨迹来自动搜索日志。它也能在部分算子上找到有效候选，例如 DSA Attention 达到 `14.54×`，但整体仍慢于 Agent-Assisted，并且 MoE FP8 和 GDN Decode 甚至低于 FlashInfer baseline。

**这个差距说明，完全自动化的智能体搜索仍然困难**。:mdnice-bold-italic[人类提供的高质量参考实现和持续积累的轨迹记忆，往往比让智能体从零开始探索更有效率]。未来系统需要**把控制状态和历史记忆纳入 harness，同时保持严格的最终验证。**
:::

## 四、未来工作

### 4.1 模型级优化闭环

> 结合 **AutoKernel** （Jaber et al., 2026[20]）的思路，把单算子优化扩展到模型级 `profile → extract → optimize → verify` 流程。

![文章插图](/articles/2026-06-05-gpu-kernel-mlsys-2026-flashinfer-9498c9ff/images/15.png)

::: image-caption
AutoKernel 整体架构图。A 阶段对模型做性能剖析并提取性能瓶颈算子；B 阶段执行自主优化循环：智能体修改 kernel.py 内核代码，基准测试模块通过五级校验规则完成正确性核验，调度器判定保留、回滚代码或切换至下一个待优化内核；C 阶段从端到端维度验证优化后模型的正确性与加速收益。三段分层解耦设计把算子定位、代码迭代、全量验证拆分，兼顾优化效率与工程落地可靠性，也是其能同时兼容 Triton、CUDA 双后端的底层保障。
:::

::: block-1
系统应先用性能剖析器找出模型里的 GPU 瓶颈，再抽取独立 Triton/CUDA 内核，并用 Amdahl's law 决定下一轮优先优化哪个内核。
:::

### 4.2 实验管理与独立验证器

参考官方公开的比赛 writeups（FlashInfer Contest, 2026b[21]），未来的实验脚手架需要固定基准测试、正确性检查、输入形状扫描、数值稳定性、确定性检查、屋顶线分析（Roofline Analysis）、保留/回滚决策和产物结构约束，并使用独立验证器复核候选实现。

### 4.3 负载特化与可检索记忆

**高分方案的共同点不是盲目尝试更多 kernel，而是先理解负载分布，再选择适合该分布的实现策略**。未来可以把常见负载画像、可复用优化模板、成功候选和失败原因都结构化记录下来，让智能体在生成代码前先检索相似场景，明确适用输入形状、已知瓶颈和不应重复尝试的方向。

## 总结

> 本次 MLSys2026 FlashInfer 比赛的结论是：全自主代码智能体远未达到替代人类的水平，GPU Kernel 自动化的胜负手从来不是 LLM 本身，而是 Harness 工程与闭环反馈系统。

团队凭借人机协同方案拿下两项季军、全自主方案拿下一项亚军，实测数据清晰暴露了**纯 AI 搜索的致命短板**：全自主 Agent 仅能在简单算子上实现提速，在 MoE、GDN Decode 等复杂场景下性能甚至不如官方基线，而人机协同方案凭借人类对优化方向的把控和可验证的评测闭环，实现了全算子超越，最高提速达 29.68 倍。

::: block-1
这证明 **Kernel 自动化的核心瓶颈不是代码生成能力，而是如何将人类经验转化为可复用的搜索约束，并构建能沉淀失败教训的持久化记忆系统。**
:::

参考资料

[1]

FlashInfer Contest, 2026a: *https://mlsys26.flashinfer.ai/*

[2]

Shui et al., 2026: *https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf*

[3]

Ma et al., 2026: *https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf*

[4]

mlsys26-flashinfer-contest: *https://github.com/syhya/mlsys26-flashinfer-contest*

[5]

Ouyang et al., 2025: *https://arxiv.org/abs/2502.10517*

[6]

Xing et al., 2026: *https://arxiv.org/abs/2601.00227*

[7]

Yu et al., 2026: *https://arxiv.org/abs/2601.15727*

[8]

Liao et al., 2025: *https://arxiv.org/abs/2512.23236*

[9]

Self-Evolving Agents: */zh/posts/2026-02-20-self-evolving-agents/*

[10]

Lopopolo, 2026: *https://openai.com/index/harness-engineering/*

[11]

Rajasekaran, 2026: *https://www.anthropic.com/engineering/harness-design-long-running-apps*

[12]

skills: *https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/skills*

[13]

评测脚手架: *https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/scripts*

[14]

OpenAI, 2026a: *https://developers.openai.com/codex/skills*

[15]

OpenAI, 2026b: *https://developers.openai.com/codex/concepts/subagents*

[16]

Wan et al., 2025: *https://arxiv.org/abs/2512.24077*

[17]

Sharma, 2025: *https://github.com/algorithmicsuperintelligence/openevolve*

[18]

Ye et al., 2025: *https://arxiv.org/abs/2501.01005*

[19]

Xing et al., 2026: *https://arxiv.org/abs/2601.00227*

[20]

Jaber et al., 2026: *https://arxiv.org/abs/2603.21331*

[21]

FlashInfer Contest, 2026b: *https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups*

::: related-title
**相关推荐**
:::

::: related-links
- **[从"切多大、留什么、按什么顺序跑"，重新定义编译器后端计算图调度：MLSys 2026 系统赛 A 赛道冠军方案浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901589&idx=1&sn=8f35478cdabeab5cffeba7d704a30009&scene=21#wechat_redirect)**
- **[AKO：Agentic Kernel Optimization，使用 Claude Code 在几小时内优化出超越 FlashInfer 专家实现的 Kernel](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901871&idx=1&sn=8d7495711c82a26ed7d8bf0a79cb6d26&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::
