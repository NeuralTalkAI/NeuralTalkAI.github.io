关键词：NaiveRT、**递归式自我改进（RSI）**、:mdnice-bold-italic[推理运行时]、稀疏注意力、**大模型推理优化**

> 当前大模型推理正在**从单纯追求模型能力，逐渐转向对整个计算系统效率的极致挖掘。尤其在强化学习训练中**，大量 Rollout 会让推理成为重要瓶颈，而传统的大 Batch 执行又容易受到长尾任务拖累。:mdnice-bold-italic[NaiveRT 的思路是从投机解码、GPU Kernel、内存搬运、跨卡通信以及执行调度等多个层面进行端到端优化]。*注：该优化未改动模型本身的结构。*

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/01.webp)

::: image-caption
图 1 | NaiveRT 与 SGLang 在 8 卡上的投机解码性能。左侧显示 NaiveRT 单流解码峰值可达 2122 tokens/s。右侧对比一轮投机推理（草稿生成、验证、采样与提交）的时延：SGLang 耗时 12.3ms，NaiveRT 仅 3.4ms，时延降低 72.4%，SGLang 完成一轮的时间里 NaiveRT 可完成约 3.6 轮。测试统计不含 prefill 阶段，在指定采样参数下测得该峰值性能。
:::

- **2122 Tokens/s 巅峰性能！从 MegaKernel 到 PDL：NaiveRT 将 RL Rollout 性能推向极限！**
- 模型权重即将开源：https://huggingface.co/NaiveAI，含 Naive-N0.5-Flash 的 W8A8 checkpoint 及配套 DFlash 草稿模型。
- 代码将于 10 月 12 日前开源：https://github.com/naiveai-labs/NaiveRT，将包含运行时、融合 kernel，以及基准测试脚本

::: related-title
**相关推荐**
:::

::: related-links
- **[拆解 megakernel 性能神话：普通 CUDA 内核靠 PDL 调度实现 97.4%持久化吞吐](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905121&idx=1&sn=f93dceb62089296ffcabc24c964e94c3&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**
- **[Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel，支撑万卡模型训练！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904378&idx=1&sn=83e25209c3e1ffc683b97a6f8646f250&scene=21#wechat_redirect)**

:::

> NaiveRT 核心技术包括 **MegaKernel Fusion、PDL、TMA 数据搬运、Shared Memory 优化以及 Tensor Core 映射优化等**。与此同时，NaiveRT :mdnice-bold-italic[还采用研究员与 AI Model 深度协同的研发模式]：研究员制定方向、约束和验收标准，AI Model 负责代码实现、性能分析、实验验证和失败回滚。

::: block-1
整个核心优化过程历时 6 天，记录了 151 轮优化实验，并通过大量端到端测试筛选有效方案。:mdnice-highlight[最终，NaiveRT 将单序列解码峰值推至 2,122 tokens/s。这一结果展示了一种“AI4AI Infra”的新型研发范式。]
:::

## 本文目录

::: related-links
- 一、AI 原生研发：从模型到系统的全链路探索
- 二、NaiveRT：面向 RL 场景的极致推理运行时
- 三、研发过程：6 天 151 轮实验的 AI 驱动迭代
- 四、性能结果与跨领域验证
- 结语：NaiveRT 的全链路优化与 AI4AI Infra 研发闭环

:::

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/02.webp)

::: image-caption
**交流加群请在 NeuralTalk 公众号后台回复：加群**
:::

## 一、AI 原生研发：从模型到系统的全链路探索

### 1.1 以 AI 为中心的研发范式

> NaiveAI 自成立起便摒弃传统以人为中心的研发体系，让 AI 模型直接参与 AI 研发的全流程：写代码、跑实验、监控进展、分析结果，再据此完成迭代。**人类研究员的精力集中在三件事上：确定方向，设定约束与评价标准，以及作出关键决策**，:mdnice-bold-italic[人的价值主要体现在经验、洞察与判断上。]

为了支撑这套体系，**NaiveAI 建立了以 AI 模型为中心的研发基础设施，系统每周支持近千万个沙箱运行，峰值并发达 10 万个**，统一的控制平台负责大规模算力、环境、工具、权限与安全管理。:mdnice-bold-italic[这套研发模式的最终目标是实现递归式自我改进，让每一代模型能够承担更多的研发工作，逐步形成研发闭环。]

### 1.2 基座模型：Naive-N0.5-Flash

::: block-1
为了充分释放 Naive-N0.5-Flash 模型轻量化架构的性能潜力，**NaiveAI 团队针对性打造了专属推理运行时 NaiveRT**。:mdnice-highlight[Naive-N0.5-Flash 是一款面向编程与 AI 研发的 MoE 模型，总参数量为 309B，激活参数量为 15.5B，原生支持 1M 上下文]，且不含全局注意力机制。
:::

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/03.webp)

::: image-caption
图 2 | Naive-N0.5-Flash大模型在 7 项代码与智能 Agent 评测基准上的横向对比结果，对比包含 Opus、GPT、DeepSeek、GLM、Kimi 等多款主流大模型。评测包含 DeepSWE v1.1、Terminal-Bench 2.1、SWE-bench Pro、ALE-CLI、ProgramBench、NL2Repo-Bench、FrontierSWE v1 共 7 项代码基准，覆盖长周期工程任务、代码仓库生成、程序重建等场景。Naive-N0.5-Flash 在 NL2Repo-Bench 取得第一名，在 ALE-CLI、SWE-bench Pro、FrontierSWE v1 也有亮眼表现；在 DeepSWE v1.1 处于中上水平，Terminal-Bench 2.1、ProgramBench 上弱于部分竞品。图表图例区分模型参数量规模，整体展现该模型在长周期代码智能体、代码仓库开发任务上强劲的竞争力。
:::

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/04.webp)

::: image-caption
图 3 | Naive-N0.5-Flash 模型在 AI 研发相关任务上的评测结果，聚焦模型后训练、机器学习工程、论文复现以及系统底层优化等 AI 科研任务，横向对比 Naive-N0.5-Flash 和 GPT、GLM、Kimi、Gemini 等主流大模型的表现。测评覆盖 PostTrainBench、MLE-bench-30、PaperBench 三大科研基准，还包含 SOL-ExecBench、NanoChat AutoResearch、NanoGPT SpeedRun 等训练与 GPU 内核优化测试。Naive-N0.5-Flash 在 MLE-bench-30、PaperBench 两项任务上显著领先其他模型，在 GPU 内核优化、模型训练效率与效果上，也优于未公开的第三方对照模型，仅在 PostTrainBench 上略低于 GPT-5.6-Sol 与 GLM-5.3，整体体现出极强的机器学习工程、论文复现和底层系统调优能力。
:::

> 在架构上，**Naive-N0.5-Flash 将滑动窗口注意力机制（SWA）与采用 GQA4 的轻量化 DeepSeek 稀疏注意力机制（DSA）相结合**，:mdnice-bold-italic[主要采用每 5 层 SWA 搭配 1 层 DSA 的布局。]

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/05.webp)

::: image-caption
图 4 | Naive-N0.5-Flash 的混合 SWA-DSA 架构。左侧 A 部分为网络堆叠结构，采用 5:1 配比的 SWA 滑动窗口注意力块与 DSA 稀疏注意力块，输入 token 先做嵌入，循环堆叠混合模块，经过归一化后送入语言模型头输出预测；右侧 B 部分是 DSA 注意力模块的详细流程，隐状态输入先经过 RMSNorm，主注意力分支完成 Q、K、V 投影与 RoPE 位置编码，保存完整历史 KV 缓存。黄色区域为 Lightning 索引器，具有 16 个 query 头，从全部历史 key 中筛选相关性最高的 top-2048 位置，仅提取选中的 KV 参与注意力计算，以此削减长序列算力开销。筛选后的 Q 与选中 K 做注意力运算，加权融合 V，输出结果残差连接后送入 FFN 层。该架构舍弃全局注意力，结合滑动窗口与稀疏检索，在支持百万级上下文的同时控制推理成本。
:::

::: block-1
如上图所示，整个网络由 8 个模块组成，每个模块 6 层；标准模块的前 5 层为 SWA，最后 1 层为 DSA，第 1 个模块的第 1 层也设为 DSA。SWA 的窗口大小为 128 tokens，DSA 则选取 top-2048 tokens 参与主干网络的注意力计算。
:::

**通过将少数全局注意力层替换为 DSA 稀疏注意力，模型在百万 token 级上下文下的解码开销大幅降低**。配合优化后的推理系统 NaiveRT，:mdnice-bold-italic[模型极速模式下推理速度最高可达 2000 tokens/s。]

## 二、NaiveRT：面向 RL 场景的极致推理运行时

### 2.1 单流速度：强化学习的核心瓶颈

> 为了充分发挥 Naive-N0.5-Flash 轻量化架构在大规模强化学习 rollout 中的潜力，**NaiveAI 构建了 NaiveRT——** :mdnice-bold-italic[一个专门优化单流解码速度、保证采样结果逐位确定的推理运行时。]

**在上下文可达 1M tokens 的强化学习任务中，单流解码速度是制约效率的核心因素**。一条 rollout 的上下文通过两种方式增长：

- 一是读入的 tokens，包括观测、工具返回与历史轮次等，速度为 5000–10000 tokens/s；
- 二是生成的 tokens，在常规解码下速度仅为 50–100 tokens/s。

生成的 tokens 只占约 20–40%，却占据了模型侧几乎全部的实际耗时，一条接近 1M tokens 的 rollout，读入需要数分钟，解码却需要数小时。

::: block-1
:mdnice-highlight[这些长 rollout 会成为拖尾任务：在同步管线中，少数缓慢的 rollout 会让整个 batch 等待]；在支持部分 rollout 的异步管线中，一条轨迹可以跨越多个策略版本，但最多只允许滞后 K 步；无法在这一限制内完成的轨迹会被整体丢弃，此前已生成的所有片段也随之作废。所以，**只有提高单流解码速度，才能从根本上缩短 rollout 本身的耗时。**
:::

> 除此之外，比特级确定性是强化学习场景的第二项核心要求。

Rollout 直接用于梯度计算，如果采样时使用的 logprobs 与训练端计算的 logprobs 不一致，训练便会在不知不觉中变成离策略训练。**NaiveRT 在所有优化中都严格满足确定性约束**：给定随机种子后，同一序列以相同种子解码两次，得到的 tokens、logits 与 KV cache 均逐位一致。

### 2.2 DSA 层：融合 29 个 Kernel 为一个 MegaKernel

> **NaiveRT 的核心优化思路是 MegaKernel 融合**。传统 GPU 推理往往把一个复杂计算拆成很多 Kernel。:mdnice-bold-italic[以 DSA 层为例，一个模块每一步原本需要执行 29 个 Kernel]，其中包括 Q/K/V Projection、RoPE、KV Cache Writeback、Indexer GEMM、Top-K、Sparse Gathering、O Projection 等。

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/06.webp)

::: image-caption
图 5 | 对比 SGLang 与 NaiveRT 在单个 DSA 层、单 GPU、单次解码步骤下的内核调用情况。SGLang 处理一层 DSA 需要执行 29 个内核，包含 25 项计算与数据搬运操作，外加 4 个索引器上下文并行内核；而 NaiveRT 通过内核融合，将整层 DSA 封装为单个 naivert_dsa_mega_kernel，由 148 个 CTA 协同执行，一次性完成输入 RMSNorm、QKV 投影、RoPE、KV 缓存写入、索引打分、稀疏注意力合并与输出投影等全部计算，还吸收了输入归一化与 TP8 通信，模型数学逻辑保持不变，仅重构调度与数据流。下方罗列了 SGLang 对应的 25 个计算数据搬运内核清单，直观体现 NaiveRT 通过大幅减少内核启动次数，消除多次内核调用带来的开销，优化推理执行架构。
:::

传统实现中，每个步骤对应 Kernel 基本是独立的。每一个 Kernel 都需要经历一定的调度过程，并且 Kernel 之间往往存在数据依赖。上一个 Kernel 将结果写入 GPU Memory，下一个 Kernel 再读取这些数据。

**因此，一个复杂的推理流程实际上包含两类成本。** 第一类是计算本身。第二类则是围绕计算产生的系统成本，包括 Kernel Launch、同步、调度以及中间结果的显存读写。

::: block-1
:mdnice-highlight[NaiveRT 的核心思路，就是尽可能消除第二类成本。在 DSA 层中，原本的 29 个 Kernel 被整合进一个 Cooperative MegaKernel]，连前置 RMSNorm 和结尾的 TP8 通信也一并纳入，减少 Kernel 间调度、同步和中间数据搬运。这种设计重新定义了这些计算阶段之间的数据流。
:::

### 2.3 148 个 CTA 如何协同工作

> **融合之后，原本跨 Kernel 的数据交换被重新组织到同一个 GPU Grid 内，由 148 个 CTA 协同完成**。CTA 可以简单理解为 GPU 上的一组线程工作单元。一个 MegaKernel 可以包含大量 CTA：

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/07.webp)

::: image-caption
图 6 | CUDA 软件线程模型与 GPU 硬件架构对比。左侧是软件抽象：一个 CUDA kernel 对应网格 grid，网格由多个线程块组成；每个线程块包含若干 CUDA 线程，并配有块内共享内存，所有线程共享全局内存。右侧是底层硬件：GPU 由多个流式多处理器 SM 构成，每个 SM 内置多个 CUDA 核心与 L1 缓存，所有 SM 共用 GPU 显存。线程块会被调度到 SM 上执行，软件线程映射到硬件 CUDA 核心，软件抽象屏蔽硬件细节，方便大规模并行编程。CTA 全称 Cooperative Thread Array，即协作线程数组，它就是 CUDA 里的线程块 thread block，同一个 CTA 内所有线程部署在同一个 SM 上运行，可使用片上共享内存交换数据，还能调用`__syncthreads()`实现块内线程同步；不同 CTA 相互独立，无法依靠共享内存通信，CTA 是 GPU 硬件调度的基本单元之一。
:::

> 这些 CTA 不再被严格分割成互相独立的 Kernel，而是在同一个 Grid 中按照既定的数据依赖关系协同工作。这就产生了一个重要变化。

原来的数据流可能是：

```
Kernel A   ↓写入显存   ↓Kernel B   ↓读取显存
```

融合后，则可以变成：

```
CTA / Stage A      ↓Grid 内同步与数据可见      ↓CTA / Stage B      ↓下一阶段
```

这意味着 **GPU 同步从原来的 Kernel 边界重新转移到了 MegaKernel 内部。此外，原本跨 kernel 的数据交换，现在在融合后的执行路径内完成**，AI 模型逐一处理各阶段的数据可见性与同步依赖，确保计算正确性。

::: block-1
在此基础上，优化进一步深入 kernel 内部。以 QKV 投影的权重搬运为例：

- 融合后每个 CTA 各自计算一份 RMSNorm，省去一次全局同步；
- 124 个负责 QKV 投影的 CTA 则先各自发起一次 128 KiB 的 TMA 搬运，再计算 RMSNorm，直到投影实际用到权重时，才在 barrier 处等待其就绪。

:mdnice-highlight[权重搬运与归一化计算重叠进行，仅此一项改动，就让整网每步耗时减少了 21–23 μs。]
:::

类似的优化还包括 K/V 暂存，按 split 将选中的 K/V 行预先搬入共享内存再执行计算，**使该层耗时从 57.0 μs 降至 49.5 μs**；以及 Tensor Core 重映射，将 8 个 query 头占据 MMA 的 n8 维度，省去补齐头带来的无效计算。

### 2.4 PDL 调度：让 GPU 自己推进执行

> MegaKernel 解决了很多 Kernel 边界问题，但 NaiveRT 进一步使用 PDL，也就是 Programmatic Dependent Launch。传统执行方式通常需要 Host 端参与：

```
CPU ↓启动 Kernel A ↓GPU 执行 ↓CPU 得知 A 完成 ↓启动 Kernel B ↓GPU 执行
```

**如果这样的流程循环很多次，CPU 就会成为不必要的中间调度者**。PDL 的思路则是让 GPU 根据程序中的依赖关系自主推进后续计算。

```
GPU：Kernel A   ↓检测依赖满足   ↓Kernel B   ↓Kernel C   ↓下一轮
```

甚至当上游 Kernel 还在进行最后一部分工作时，只要下游所需的数据已经准备好，下游 Kernel 就可以提前启动。因此可能形成这样的重叠：

```
A：████████████████████B：        █████████████████
```

B 不需要等待 A 的最后一点工作完全结束就可以开始。这进一步压缩了 GPU 的空闲时间。

通过 PDL，尾部 Kernel 提交 token 后，可以直接发起下一轮。最终，多轮循环无需 CPU 介入，Host 端主要负责将结果流式输出。

## 三、研发过程：6 天 151 轮实验的 AI 驱动迭代

### 3.1 三阶段优化路径

> NaiveRT 采用了与 Naive-N0.5-Flash 相同的 AI 原生研发方式，**研究员负责制定技术方向、质量约束与验收标准，AI 模型承担大量实现与优化工作，** :mdnice-bold-italic[包括分析整网性能画像、实现候选方案、进行数值验证与多卡测试、分析实验结果、决定哪些改动保留、修改或回滚。]

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/08.webp)

::: image-caption
图 7 | NaiveRT 在六天人工引导 + AI 驱动的优化工作，一共开展 151 项优化试验，最终 63 项优化方案被采纳。整个优化分为三大阶段：阶段一是整网工程优化，进行 43 次试验，落地 28 项改动；阶段二基于真实模型权重校验执行流程，完成 45 次试验，落地 15 项改动；阶段三针对 W8A8 量化内核做精细调优，共 63 次试验，落地 20 项改动。其余试验验证失败或被回滚，还有部分备选方案仅做原型验证，整套优化持续降低完整验证前向推理时延。
:::

研发分为三个阶段推进，与上图对应：

| 阶段 | 描述 |
| --- | --- |
| 第一阶段是整网工程实现，共 43 轮实验，28 轮被采纳。 | 该阶段在随机权重上打通融合后的整网执行路径，**重构路由与数据搬运，完成了基础的 MegaKernel 融合架构搭建。** |
| 第二阶段是切换至真实 checkpoint 优化，共 45 轮实验，15 轮被采纳。 | 基于真实模型权重优化 Tensor Core 与 TP 执行，**去除冗余操作，并实现跨 GPU 执行重叠**。 |
| 第三阶段是 **W8A8 kernel 精细优化**，共 63 轮实验，20 轮被采纳。 | 深入关键 kernel 内部，:mdnice-bold-italic[精细调整加载顺序、计算布局与收尾逻辑]，进一步压榨性能。 |

> 下面我们逐一看看每阶段的优化：

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/09.webp)

::: image-caption
图 8 | 左图整网优化在随机权重、prompt8 的基准条件下开展 25 轮试验，将时延由 6.995ms 下降至 3.173ms。左图中曲线随迭代逐步下降，标注了四项核心优化：基于 Tensor Core 实现 8 行 QKV 计算；融合注意力、输出投影与张量并行集合通信；通过屏障裁剪与 SWA 加载减少同步开销；优化专家选择器，合并两次内核启动。黄色实线代表成功采纳的优化路径，灰色点为未通过验证方案，直观呈现全网层面基础优化带来的时延持续降低。右图路由与数据搬运优化基于 prompt200 基准，共 15 轮试验，时延从 3.875ms 降至 3.171ms。该部分聚焦数据传输与调度，包含三项关键改进：重构专家选择器，合并两次内核调用；使用批量异步收集操作实现稀疏 KV 预加载；采用可编程调度，提前启动存在依赖关系的任务。曲线前期时延稳定，引入优化后出现明显下降。该模块主要消除 GPU 间数据搬运、任务调度的额外开销，降低数据流转带来的推理延迟。
:::

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/10.webp)

::: image-caption
图 9 | 左图张量核与张量并行执行优化，14 轮试验把时延从 4.632ms 降至 3.789ms。核心优化点有三处：利用 Tensor Core 的 LM 头，复用专家输出激活值；张量并行规约计算改为单次计算再分发；在 128 行张量核内核中实现成对专家矩阵运算。曲线随优化迭代持续走低，重点优化张量并行通信与 Tensor Core 硬件计算单元利用率，减少重复计算与通信等待耗时，充分释放 GPU 硬件算力，降低并行推理时延。右图消除冗余计算子图，31 轮试验将时延从 3.985ms 降至 3.608ms。核心优化包含三点：仅计算有效计算块，跳过空专家输出；在路由阶段一次性完成 FP8 量化并复用输入；专门定制计算线程组，精简参与 MMA 计算的集合。图中曲线持续缓慢下降，菱形标记为未落地原型方案。该优化聚焦裁剪无效计算，减少重复量化、无用激活读写等冗余操作，减少不必要算力消耗，进一步压缩推理耗时。
:::

![文章插图](/articles/2026-09-28-2122-tokens-s-megakernel-pdl-735ec1e3/images/11.webp)

::: image-caption
图 10 | 左图跨 GPU 执行重叠采用真实 checkpoint、上下文约 3.6K、修正精度路径，共 4 轮试验，时延从 3.607ms 降至 3.466ms。右图内存与内核尾部优化才是真实 prefill、context 200、W8A8、PDL on。主要两项优化：基于就绪依赖条件启动任务，搭配选择性 PDL 边；使用并发对等轮询，同时保证规约计算顺序。曲线经过两次关键优化后趋于平稳。该优化目标是让不同 GPU 的计算、通信过程互相重叠，掩盖跨设备通信等待时间，在保证计算结果正确性前提下，提升多卡并行执行效率，降低多卡推理时延。右图展示内存与内核尾部优化子图开启 PDL，共 63 轮试验，时延从 3.441ms 降至 3.131ms。包含四项关键优化：DSA 张量核重映射，去除头部填充计算；优化 SWA 权重暂存，把大量拷贝转为分块 TMA 传输；清除死路由写入，只保留需要使用的激活；专门定制输出投影尾段，优化线程与校验逻辑。曲线随大量细粒度内核调优持续下降，属于底层内核、内存访问层面的精细打磨，减少内存读写开销与内核尾部冗余指令。
:::

### 3.2 端到端优先的优化原则

> 在整个优化过程中，AI 模型始终依据候选优化在真实模型链路中的端到端表现作出取舍。下面是实际中遇到的三个案例：

- 一项跳过索引、直接按位置读取 K/V 的优化，**在 200 tokens 短上下文时能节省约 3 μs，但到了 2200 tokens 上下文，反而慢了约 3 μs**，因此未被采用。
- **TMA 预取优化则相反**，:mdnice-bold-italic[单算子 warm-cache 微基准略慢，但整网每步却能快 21–23 μs]，最终被采纳。
- 对于 MoE 部分的融合优化，候选融合路径经过 7 轮实现，**每一版数值都正确，但端到端性能却全部下降。** :mdnice-bold-italic[原因是原有 kernel 边界的开销大部分已被 PDL 重叠，而融合又引入了额外同步]，研究员最终叫停了这一方向，MoE kernel 继续通过 PDL 衔接。

端到端优化中，正确性是所有优化的前提。任何降低延迟的改动合入前，整网 logits 与 KV cache 都必须通过严格的逐位比对，同时完成端到端性能测量。

## 四、性能结果与跨领域验证

### 4.1 NaiveRT 的性能表现

::: block-1
:mdnice-highlight[最终，NaiveRT 在 8 卡上的单流解码峰值达到 2122 tokens/s]。这一峰值取自 41 个 HTML/SVG 生成请求中表现最好的一秒时间窗口，测试关闭思考模式，temperature 为 0.4、top-p 为 0.95，且不计入预填充。
:::

完整一轮投机解码（包含 draft、verify、sampling 与 commit），**SGLang 一轮耗时 12.3 ms，NaiveRT 为 3.4 ms，时延降低 72.4%，约相当于 3.6 倍差距**。:mdnice-bold-italic[这一结果来自整条推理链路中多项技术的协同：轻量化稀疏注意力架构、kernel 融合与 DFlash 草稿生成、完全由 GPU 驱动的调度，以及通过轻量原语实现的通信重叠。]

### 4.2 AutoWM：RSI 模式的跨领域印证

> NaiveRT 并非 AI 原生研发模式的唯一成果。**在此之前，一位 NaiveAI 研究员曾将构建世界模型的任务交给 Naive-N0.5-Flash**，设定目标、算力预算与评测协议后，由模型持续推进方案设计、模型训练、评测与结果分析。

经过累计 400 小时的研究与 15 轮主要实验，最终得到的世界模型 AutoWM，在 WorldArena-1 Track 1 的公开评测中取得 77.43 分，高于当时榜单上已公开的最高成绩 73.64 分。

::: block-1
从 AutoWM 到 NaiveRT，从模型训练到推理系统优化，**AI 原生研发模式展现出了跨领域的迁移能力。** :mdnice-highlight[当目标、算力预算与评测协议明确时，同样的研发方式可以快速迁移到新的领域，持续产出前沿成果。]
:::

## 结语：NaiveRT 的全链路优化与 AI4AI Infra 研发闭环

> 从表面上看，NaiveRT 是一个 GPU 推理运行时。但把整个研发过程串起来后可以发现，它真正值得关注的地方有两个。

- 第一个，是对 GPU 执行模型的重新组织。

与往常针对某个 Kernel 优化不同，NaiveRT 从整个推理链路出发，将 DSA 层的 29 个 Kernel 融合进一个 Cooperative Mega-Kernel，让 148 个 CTA 在统一执行空间中协作，再通过 PDL、TMA、Shared Memory 和 Tensor Core Mapping 等技术重新安排数据流和计算流。**它优化的是整个 GPU 上的时间关系。**

- 第二个，则是 AI Model 开始真正进入 AI 基础设施研发的核心环节。

**研究员负责定义方向、边界和评价标准，AI Model 负责大量具体的实现、实验、测量、验证和回滚**。在 6 天时间里完成 151 轮优化实验，并通过数千次端到端运行验证，从大量失败方案中筛选出有效路径，:mdnice-bold-italic[本质上是一种高密度的工程搜索过程。]

### 超越算力：NaiveRT 构建 AI4AI Infra 研发闭环

> 从 DSA 到投机解码，从 Mega-Kernel 到 PDL，从 TMA 到 Shared Memory，再到 AI Model 驱动的高频实验闭环，NaiveRT 展现的是一种越来越明显的趋势：

未来的大模型基础设施竞争，**不再只是“谁拥有更大的模型”或者“谁拥有更多的 GPU”，而会越来越取决于谁能够更高效地利用每一个 GPU 周期、每一次内存访问、每一次 Kernel 调度，以及每一轮研发实验。**

::: block-1
而当 AI 本身开始参与构建这些基础设施时，AI 优化 AI 的闭环也就真正开始形成。**NaiveRT 的 2,122 tokens/s，最终只是这个闭环产生的一个数字化结果**。:mdnice-highlight[更值得关注的，是数字背后的方法：让研究员负责判断什么值得做，让 AI Model 负责快速尝试怎么做，再用真实端到端数据和严格的 bitwise correctness 作为最终裁判]。这或许才是这套系统最值得被记录下来的地方。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[较 SGLang 1.21 倍提速！ForgeMegakernel 开启 AI 推理 MegaKernel 的自动化锻造时代](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905880&idx=1&sn=d33d843035f0dfc0a9af3f02f2811a94&scene=21#wechat_redirect)**
- **[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903011&idx=1&sn=7872d080a5475a44bba4cfabe7717b56&scene=21#wechat_redirect)**
- **[适配 Ada 中端卡的 MegaKernel 在线广告方案 Ada-MK：共享内存直接减半，端到延迟最高降 50%，击穿小批量毫秒级推理瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902893&idx=1&sn=5f53dd32ac726cea553f1c3743d8d2a1&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**
- **[10.3 倍加速！康奈尔大学 Perseus：终结多节点 MoE MegaKernel 的隐藏序列化噩梦，让代理传输反超 GPU-direct 1.2 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901207&idx=1&sn=4e68da37ab9ecd3a8d20e16451ea6ebe&scene=21#wechat_redirect)**

:::
