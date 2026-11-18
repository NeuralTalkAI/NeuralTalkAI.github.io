关键词：**AKO**、:mdnice-bold-italic[GPU内核优化]、Agentic、FlashInfer、**自动化Kernel生成**

> 在MLSys2026英伟达内核竞赛B200硬件环境下，**AKO4X优化的五项内核性能全部超越MIT获奖KDA方案** 与FlashInfer人工专家实现，DSA稀疏注意力相对基线最高提速30.71倍；**轻量化AKO4ALL可在一小时内完成多款推理算子优化并突破专家基准。**

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/01.webp)

::: image-caption
相对于 FlashInfer 专家实现在 5 个 MLSys-2026 竞赛内核（NVIDIA B200）上的几何平均加速比 —— AKO4X 在 DSA 稀疏注意力上最高超越专家 30.71×，并在全部五项上都领先于 MIT 的 KDA。
:::

受赛事地域参赛资格规则约束，项目团队未能入围获奖名单，因此全量开放源码与实测数据。文章拆解Agent高效优化内核的核心诱因，客观指出当前自动化优化存在跑分作弊、真实工业环境适配困难等短板，并对Agent驱动Kernel研发的行业未来做出展望。

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/02.webp)

- **AKO：Agentic Kernel Optimization，使用 Claude Code 在几小时内优化出超越 FlashInfer 专家实现的 Kernel**
- 作者：sha7dow，https://zhuanlan.zhihu.com/p/2044554843948308001

相关推荐

- [算子 编译器天塌了吗。。NVIDIA 提出 AVO：当 AI Agent 成为进化算子，探索突破 Blackwell 性能极限，超越 FlashAttention‑4 最高10.5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899076&idx=1&sn=64a6532410f9d22f65043945582d4f1a&scene=21#wechat_redirect)
- [MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)
- [算子工程师末日。。。NVIDIA提出SOL-ExeBench：对标硬件效率极限，生成真实 GPU Kernel！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898808&idx=1&sn=0c8886f90fc178c6ae7aedbf1931e0e9&scene=21#wechat_redirect)

> 我们开源了完整的 AKO: Agentic Kernel Optimization，包括：

- 项目主页：https://tongminglaic.github.io/AKO/
- AKO4ALL：https://github.com/TongmingLAIC/AKO4ALL
- AKO4X：https://github.com/TongmingLAIC/AKO4X

## 本文目录

- 优化结果
- 简单介绍
- 一些思考

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/03.webp)

## 优化结果

> 在 MLSys 2026 Nvidia Track（FlashInfer AI Kernel Generation Contest）[1]里，MIT 的方案 Kernel Design Agents（KDA）[2]在 Full Agent 赛道的最终结果是：

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/04.webp)

::: image-caption
MLSys2026 英伟达 Kernel 生成赛事里 MIT-KDA 参赛方案的官方排名，赛事划分为 MoE、DSA、GDN 三个独立赛道，其中 KDA 在 MoE 赛道斩获第一名，DSA 赛道取得第二名，GDN 赛道位列第三名。这份官方榜单是 AKO4X 性能对比的基准参照，在相同 B200 硬件与评测规则下，AKO4X 优化后的五项内核实测性能全部优于该 MIT 获奖方案，尤其 DSA 稀疏注意力算子相较 FlashInfer 专家基线提速 30.71 倍，全方位实现对官方获奖 Agent 方案的性能反超，也侧面印证 AKO 架构在自动化 GPU 内核优化领域的突出实力。
:::

但实际我们 AKO4X 优化的 5 个 kernel（3 个 Track 一共 5 个 kernel），在我们公开的同一套 B200 评测口径下（官方环境），性能全部显著超过了 KDA 提交的 kernel：

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/05.webp)

::: image-caption
AKO4X与MIT获奖方案KDA性能比较，比较也参考FlashInfer性能作为基线。在DSA稀疏注意力上，AKO4X相对基线提速30.71倍，远超KDA的3.78倍；GDN prefill、MoE fp8、GDN decode三项算子中，AKO4X提速倍数全部优于KDA，后两项KDA甚至跑不赢专家基线；余下DSA top-k索引算子，AKO4X相较KDA提速1.27倍。全维度数据表明AKO4X自动化优化效果全面超越MIT顶尖Agent方案与人工专家实现。
:::

> *注意这里的 baseline 不是 naive PyTorch baseline，而是 FlashInfer 高性能算子库里的专家实现。*

所以从技术结果看，我们本来应该排在 KDA 前面。但比较遗憾的是，由于 **“we also removed (after careful review and discussion) teams with members based in restricted regions from the winner candidates.”** 的资格限制（比赛奖品的显卡有出口管控），**我们作为中国团队最终没有进入官方 winner candidates**。

我们尊重比赛规则和合规要求，但也希望把技术结果、代码、数据和完整优化轨迹都公开出来，一方面想向大家展示我们这几个月的工作和成果，另一方面也是希望能加速这个方向的发展。

除了比赛的这 5 个 kernel，我们还在 B200 上测试了 FlashInfer-Bench 中的其他几个 kernel（数量不多，因为 B200 太贵了 😢），**除了 GEMM 外都在几个小时甚至一个小时内超越了 FlashInfer 中的专家实现**，具体结果可以见上面的项目主页。

详见更多优化轨迹和结果[3]。

## 简单介绍

> 和两个多月前我们在为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？[4]的回答中说的一样，AKO 不是一个新模型，也不是一个复杂的搜索算法。它就是一个任务特定的 harness：一个给 coding agent 使用的 kernel 优化环境。

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/06.webp)

::: image-caption
为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？[5]内容节选，这段文字介绍AKO4ALL在NVIDIA推出的SOL-ExecBench基准榜单实现榜首，仅需1至2小时迭代生成的算子内核，大多性能优于英伟达官方多Agent优化基线（分值0.5）；研发受硬件限制，在A100调优后提交B200平台测评，优异成果依托Claude Opus大模型强悍的代码优化能力。
:::

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/07.webp)

::: image-caption
AKO4ALL在SOL-ExecBench基准集上的SOL Score性能对比结果，横轴罗列13项GPU算子，纵向划分为L1、L2、Fib三组测试场景，虚线SOL Bound=1.0代表硬件理论性能上限，橙色柱是英伟达多Agent优化基线（分值0.5基准线）、灰色为原生PyTorch参考实现、青绿色是AKO4ALL优化成绩。整体来看AKO4ALL绝大多数算子得分显著优于另外两组，除030 attn_proj_residual（0.45）略低于英伟达基线外，剩余算子全部超越NVIDIA多Agent方案与原生PyTorch实现，其中006 mm_rope_pos_calc以0.85取得全组最高分，逼近硬件理论上限，直观印证Agent驱动的AKO4ALL内核优化能力全面优于官方多Agent方案与原生实现。
:::

它做的事情很简单：给 coding agent 提供 benchmark、profiler、迭代记录、候选 kernel 管理、反作弊检查等工具，然后让 coding agent 自己读代码、改 kernel、跑 benchmark、看 profiling、换 DSL、继续迭代，每一步做什么都由它自己决定。

::: block-1
我们的设计理念是：

- 尽可能少地限制 agent 本身的能力 （对比 fixed pipeline 方法）；
- 提供工具、信息，使 agent 能够在 kernel 优化任务下更好地发挥能力
:::

> 我们开源了两个工具：

- **AKO4ALL[6]：这个实际上在两个月前的回答中已经开源了，这次我们只是对它进行了一次重构**。AKO4ALL 现在是一个非常轻量的 SKILL。安装后把 coding agent 指到你的 kernel（+ benchmark / reference）上，就可以开始迭代优化。:mdnice-bold-italic[它更适合快速优化一个 kernel 或者你已经有自己 benchmark / 快速适配新硬件等场景。]

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/08.webp)

::: image-caption
该柱状图展示轻量化工具AKO4ALL在一小时内优化四项主流推理算子、对标FlashInfer专家手写内核的提速表现，图中虚线1.0倍为人工实现基准。四项算子全部实现性能超越：GQA decode提速1.36倍、MLA decode提速1.21倍、MLA prefill提速1.50倍（四项里提升幅度最高）、RMSNorm提速1.12倍。数据印证AKO4ALL依托Claude编码大模型，仅用短时迭代就完成自动化内核优化，轻量化框架无需复杂配置，即可批量产出优于行业专家手工开发的高性能GPU算子，凸显Agent驱动内核优化在落地效率与性能层面的突出优势。
:::

- **AKO4X[7]：更完整的 campaign-grade 优化框架**。它会为每轮优化创建隔离环境，维护每个 operator 的 archive，把历史 variant、lessons、dead-ends 记录下来，让后续轮次基于前面的经验继续优化。:mdnice-bold-italic[它更适合固定 operator 上的多轮、可复现、可审计优化，比如这次 FlashInfer-Bench / MLSys contest 这种场景。]

![文章插图](/articles/2026-06-02-ako-agentic-kernel-optimization-claude-a6222b9b/images/09.webp)

::: image-caption
B200显卡环境下MLSys2026竞赛四项GPU内核的加速性能对比，虚线代表FlashInfer专家手写算子基准1.0倍，蓝色柱为AKO4X优化结果，紫色柱是MIT获奖方案KDA。四项测试里AKO4X全部突破专家基线，GDN decode、MoE两项中KDA性能甚至不及人工基准，分别只有0.81倍、0.59倍；GDN prefill场景AKO4X提速2.30倍，高于KDA的1.53倍；DSA稀疏注意力性能差距悬殊，AKO4X达到30.71倍提速，KDA仅3.78倍，直观体现AKO依托Agent的自动化内核优化实力全面领跑竞品与人工专家实现。
:::

::: block-1
比较有意思的是，**AKO4ALL 虽然比 AKO4X 轻很多，但效果仍然非常好。** 我们在 FlashInfer-Bench 上用 AKO4ALL 测了 4 个 inference operators：GQA decode、MLA decode、MLA prefill、RMSNorm。**它在不到一小时、单次 prompt 的设置下，四个都超过了 FlashInfer expert kernel**，其中一个的结果还超越了 AKO4X 花费更多时间的优化结果。不过，这里也可能部分受益于 AKO4ALL 我们测试时使用了更新的 Claude Opus 4.8（AKO4X 测试用的是 Opus 4.7）。
:::

详细内容同样见我们的项目主页和代码仓库。

- 项目主页：https://tongminglaic.github.io/AKO/
- AKO4ALL：https://github.com/TongmingLAIC/AKO4ALL
- AKO4X：https://github.com/TongmingLAIC/AKO4X

## 一些思考

> 今年 3 月 24 号，在为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？[8]的回答中，我们开源了初版的 AKO4ALL，并且展示了不错的效果。第二天 NVIDIA 发的 AVO: Agentic Variation Operators for Autonomous Evolutionary Search[9] 也做了和我们几乎一样的事情（使用 Agentic 的方式去做 kernel 优化，而不是之前很多固定 pipeline 的方法），并且做得更加系统，**实验也更加完善，不过并没有开源。**

实际上从那时开始，大家逐渐发现 Agentic 的方式做 kernel 优化的效果非常好。我们当时也正在参加 MLSys 2026 的比赛，所以虽然当时我们已经有了 AKO4X 的初版，但是并没有开源。

> 为什么 coding agent 做 kernel 优化的效果这么好呢？我们目前的思考是：

- **第一，最重要的当然还是模型本身**。比如 Opus 在读 profiling、理解 kernel bottleneck、改 Triton/CUDA/TileLang 这些方面已经很强。
- **第二，kernel 优化这个任务本身非常适合 coding agent 做闭环迭代**：反馈快、指标明确、改动范围相对小、边界也相对清楚。我们的 harness 不需要发明很多复杂机制，只要把 benchmark、profiler、日志、候选管理这些东西放到 agent 手边，再增加一些辅助机制比如允许它用更快的 subset workload 先拿到大致信号，就能让它比较快速稳定地逐步优化。同时 kernel 优化本身也需要更丰富的上下文、动态地决定下一步做什么，这也是 coding agent 的优势。
- **第三，这次比赛的 workload 集合是固定的，而且模型可以直接访问到**。虽然 FlashInfer-Bench 里的 workloads 来自真实工业推理场景，不是随便挑几个 shape；:mdnice-bold-italic[但从优化角度看，这仍然给了 agent 做 workload-specific optimization 的空间。而 agent 本来就很擅长根据具体输入分布写特化代码]。比如我们 DSA Sparse Attention 的最终提交里，就混合了 Triton 和 TileLang，并且针对 T = 1、T = 2、T > 2 分别写了不同 kernel。当然能写大量的 workload-specific 代码本身也是 LLM 相比于专家手写最大的优势之一吧，因为 coding agent 是不知疲倦的（

> 那是不是 kernel 优化已经被 agent 解决了？那肯定远远没有。

现在大部分 LLM-driven kernel optimization 仍然是在一个或少数几个 kernel 上单独做优化和评测，而且**通常满足上面这些条件：目标明确、benchmark 固定、反馈快、改动边界小。但真实工业场景要复杂得多。**

**即使是在 FlashInfer-Bench 这种已经很规范的 benchmark 里，reward hacking 也会非常严重**。我们在比赛过程中就遇到过至少几类 agent 利用 benchmark 漏洞、CUDA graph 行为或计时/缓存机制得到“假加速”的情况。它们看起来通过了 correctness，也能让 measured latency 变好，但并不是一个真实、可移植、可部署的 kernel 优化。:mdnice-bold-italic[实际上我们也并不能保证我们展示的最终优化结果一定有实际意义，因为总得来说我们只是在 FlashInfer-Bench 官方的规则下去优化（]

::: block-1
真实场景下显然要麻烦得多：很多时候即使是非常简单的端到端测试也很难给出干净、低噪声、可归因的信号；性能问题也不一定只在一个 kernel 文件里，可能牵涉更多更复杂的系统。
:::

所以我们并不觉得 “agent 已经解决 kernel 优化” 了。更准确的说法是：**在一个边界清楚、反馈明确、workload 可访问的环境里，强 coding agent 已经可以非常有效地做 kernel 优化，甚至在一些真实高性能库的 expert kernel 上继续推进性能。** （当然满足了这些条件，不只是 kernel，也许在其他领域也都可以利用 agent 进行类似的优化了）

也许随着模型能力继续增强，未来真的会出现这样的流程：新硬件出来后，agent 能自动读 profiling、理解架构、在真实工业场景下生成并验证一批高质量 kernels，很快把常见 workloads 调到接近专家水平，甚至超过专家。

希望大家以后都不用痛苦地手写 kernel 了。

我们也希望 AKO 能往这个方向推进一点点。

> 欢迎大家使用、Star ✨、提 Issue，也欢迎一起讨论！

- 项目主页：https://tongminglaic.github.io/AKO/
- AKO4ALL：https://github.com/TongmingLAIC/AKO4ALL
- AKO4X：https://github.com/TongmingLAIC/AKO4X

参考资料

[1] MLSys 2026 Nvidia Track（FlashInfer AI Kernel Generation Contest）: *https://mlsys26.flashinfer.ai/*

[2] Kernel Design Agents（KDA）: *https://github.com/mit-han-lab/kernel-design-agents*

[3] 更多优化轨迹和结果: *https://github.com/TongmingLAIC/AKO4X/tree/main/reference*

[4] 为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？: *https://www.zhihu.com/question/1998102573333385508/answer/2019598125745059025*

[5] 为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？: *https://www.zhihu.com/question/1998102573333385508/answer/2019598125745059025*

[6] AKO4ALL: *https://github.com/TongmingLAIC/AKO4ALL*

[7] AKO4X: *https://github.com/TongmingLAIC/AKO4X*

[8] 为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？: *https://www.zhihu.com/question/1998102573333385508/answer/2019598125745059025*

[9] AVO: Agentic Variation Operators for Autonomous Evolutionary Search: *https://arxiv.org/abs/2603.24517*
