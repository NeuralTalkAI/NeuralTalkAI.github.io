关键词：Fast-dVLM、**视觉语言模型**、离散扩散、块扩散、:mdnice-bold-italic[KV Cache]、**自投机解码**、SGLang、**FP8 量化**

> 这篇论文真正要解决的问题，不是让视觉语言模型回答得更聪明一点，而是让它在真实部署中不再被逐 token 自回归解码拖住。

过去两年，视觉语言模型已经从“看图问答”工具变成了物理 AI、机器人、自动驾驶、文档理解、图表分析和多模态智能体的通用感知推理模块。

::: block-1
模型能力提升很快，但:mdnice-bold-italic[推理机制却仍然高度依赖自回归解码：每次只生成一个 token，生成下一个 token 前必须读入已有上下文、加载模型参数、执行一轮前向]。这套机制在云端大 batch 服务中还可以靠批处理摊薄成本，**但在机器人和车端这类 batch size 为 1 的物理 AI 场景中，问题会被急剧放大。**
:::

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/01.webp)

- **Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM**
- https://arxiv.org/abs/2604.06832
- 项目主页：https://nvlabs.github.io/Fast-dLLM/fast_dvlm
- 代码仓库：https://github.com/NVlabs/Fast-dLLM
- 8000 字，阅读 40 分钟，**播客 29 分钟**

相关推荐

- [Blackwell 架构下的高速扩散模型：基于 Diffusers 与 TorchAO 的 MXFP8 及 NVFP4 应用](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899996&idx=1&sn=c9541fb4324c46b5e2708ea19ca2bbf7&scene=21#wechat_redirect)
- [FP4 Explore, BF16 Train：扩散强化学习的效率革命——4.64 倍加速背后的算法-硬件协同设计。假量化之名，行解耦之实](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899811&idx=1&sn=5f3323ca5e7c6f6cc65f814377d4b767&scene=21#wechat_redirect)
- [无需训练，提速一个数量级！上交与华为小艺提出 LoPA 前瞻解码：让扩散大模型单步生成 10 个 Token，吞吐量超 1000 token/秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895908&idx=1&sn=2f25d8f315aac5920a775210a8b6d00b&scene=21#wechat_redirect)

> 论文指出，单请求 VLM 推理的瓶颈往往不是算力不够，而是内存带宽被自回归解码反复消耗。每生成一个 token，模型都要重新调动大规模参数，却只利用了硬件很小一部分并行能力。于是，一个自然问题出现了：:mdnice-bold-italic[既然 VLM 的生成部分可以看作语言模型生成，能否把文本领域正在兴起的离散扩散和块级并行生成引入 VLM，从一次只吐一个 token，变成一个 block 内多个 token 并行去噪？]

Fast-dVLM 的回答是肯定的。它以 Qwen2.5-VL-3B 为自回归基座，将其直接转换成 **block-diffusion VLM，支持 KV cache 兼容的块级并行解码，并进一步加入自投机块解码、SGLang serving 集成和 SmoothQuant W8A8 FP8 量化**。论文报告：

- 在 11 个多模态 benchmark 上，Fast-dVLM 的短答任务平均质量可以匹配 AR 基线；
- 在 MMMU-Pro-V 的长链式推理场景中，最终系统路径相对自回归基线达到 6.18× 端到端推理加速。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/02.webp)

::: image-caption
表 1：短答案基准性能对比（第一部分）。表格对比AR与扩散类VLM在8个短答案多模态基准的性能，Fast-dVLM（MDM/推测解码）在GQA、POPE等任务上超越AR基线，平均精度逼近Qwen2.5-VL-3B。在同期扩散VLM中，Fast-dVLM在11个短答案基准里拿下8项最优，远超LaViDa、Dimple等模型。这表明块扩散架构可在短答案任务中实现精度无损，同时具备显著的推理加速潜力。
:::

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/03.webp)

::: image-caption
表 2：基准性能对比（第二部分）。表格补充长答案基准MMMU-Pro-V与Tokens/NFE指标，Fast-dVLM推测解码平均精度74.0，与AR基线完全持平，Tokens/NFE达2.63×，MDM解码为1.95×。长文本推理中，推测解码将精度差距缩小至1.7分，证明块扩散结合自推测解码，能有效缓解并行去噪带来的长文本连贯性问题，兼顾速度与长文本性能。
:::

::: block-1
Fast-dVLM 的关键价值，不是简单把扩散模型套到 VLM 上，而是找到了一个可落地的折中点：块与块之间仍然自回归以保留 KV 缓存，块内部通过离散扩散并行生成以释放硬件并行度。
:::

## 本文目录

- 一、为什么 VLM 推理需要换一种生成范式
- 二、核心问题：把扩散语言模型扩展到 VLM 为什么难
- 三、直接转换还是两阶段转换：论文最重要的经验结论
  - 3.1 两阶段路径
  - 3.2 直接路径
- 四、训练设计：Fast-dVLM 如何让块扩散适配多模态输入
  - 4.1 Causal context attention：保留 AR 模型的因果结构
  - 4.2 Block-size annealing：不要一上来就让模型处理大块噪声
  - 4.3 Auto-truncation mask：防止短回答跨越下一轮 prompt
  - 4.4 Vision-efficient concatenation：视觉 token 不要复制两份
  - 4.5 训练目标：扩散损失与因果 LM 损失共同约束
- 五、推理机制：从 MDM 到自投机块解码
  - 5.1 Causal context decoding：每个 block 先由一个 AR token 启动
  - 5.2 MDM decoding 的阈值权衡
  - 5.3 Self-speculative block decoding：同一个模型既 draft 又 verify
  - 5.4 SGLang 与 FP8：系统优化把算法收益兑现到端到端
- 六、实验结果：质量是否真的保住了
  - 6.1 短答任务：平均质量匹配 AR 基线
  - 6.2 长答任务：MMMU-Pro-V 仍是难点
  - 6.3 消融：哪些组件最关键
- 七、把 Fast-dVLM 的核心机制压成一条链
- 八、相关工作：Fast-dVLM 站在扩散 LLM、扩散 VLM 和投机解码的交汇处
  - 8.1 离散扩散语言模型
  - 8.2 扩散视觉语言模型
  - 8.3 投机解码与扩散投机解码
- 九、结论与展望
  - 9.1 结论总结
  - 9.2 进阶分析
  - 9.3 未来工作

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、为什么 VLM 推理需要换一种生成范式

> 自回归解码的低效，在大模型推理里并不是新问题；但在物理 AI 和边缘 VLM 场景中，它从性能问题变成了部署瓶颈。

视觉语言模型的主流推理方式仍然是 autoregressive decoding。设输出 token 序列为 $\mathbf{x}=(x_1,\dots,x_L)$，自回归模型按如下方式分解概率：

$$
P_{\theta}(\mathbf{x})=\prod_i P_{\theta} (x_i | x_{<{i}})
$$

这意味着生成过程具有严格的串行依赖：必须先得到 $x_i$，才能生成 $x_{i+1}$。

- 在纯文本聊天里，这已经会限制吞吐；
- 在 VLM 里，问题更复杂，因为前缀中还包含视觉 token、图像编码器输出、多轮对话上下文和结构化提示。

::: block-1
论文特别强调了 physical AI 场景。机器人、自动驾驶车辆、具身智能体通常不是像云服务那样一次处理大量请求，**而是单设备、单观察流、batch size 为 1 地做实时推理**。这个 regime 下，自回归解码呈现典型 memory-bandwidth-bound 特征：:mdnice-bold-italic[每一个 token 都要触发一次模型参数读取和前向计算，但由于 batch 太小，Tensor Core 并行度很难吃满。]
:::

> 扩散式语言模型提供了一个替代方向。与自回归不同，masked diffusion 会随机 mask 一部分 token，让模型学习从噪声状态 $\mathbf{x}_t$ 恢复 clean token $\mathbf{x}_0$：

$$
p_{\theta}(\mathbf{x}_0\mid \mathbf{x}_t)
$$

如果多个 token 可以在同一轮前向中同时去噪，那么吞吐就不再被“一个 token 一次前向”严格锁死。**问题在于，普通全序列扩散很难使用增量 KV cache，尤其在长上下文、多轮对话和 VLM 视觉前缀下，直接全局双向注意力会让服务系统退回低效状态。**

Fast-dVLM 选择的是 block-wise discrete diffusion：

- 序列被分成 block，block 之间仍按顺序生成，已完成 block 可以复用 KV cache；
- block 内部 token 则并行去噪。

这种结构保留了自回归的增量缓存优势，又部分释放了扩散模型的并行生成能力。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/05.webp)

::: image-caption
图 1：论文总览。该图从速度 - 精度权衡、多基准测试、端到端推理加速三个维度呈现 Fast-dVLM 核心效果。速度 - 精度维度中，Fast-dVLM 在 MMMU-Pro-V 任务上精度比肩自回归（AR）基线且实现大幅提速；多基准测试里，其基于 Qwen2.5-VL-3B 基座在各多模态任务上精度近乎无损；推理加速结合 SGLang 与 FP8 量化，最终实现超 6 倍端到端提速。这验证块扩散架构可突破 AR 解码吞吐量瓶颈，完美适配机器人、自动驾驶等单批次边缘物理 AI 部署场景。
:::

::: block-1
从系统角度看，Fast-dVLM 的目标不是“替代所有 AR VLM”，而是**在 batch size 1、长输出、边缘或物理 AI 场景中，把原本 memory-bound 的逐 token 生成，尽量转化为更能利用硬件并行度的 block-level 生成。**
:::

## 二、核心问题：把扩散语言模型扩展到 VLM 为什么难

> 文本扩散模型已经证明了块级并行生成的潜力，但 VLM 不是简单在文本前面拼一段图像 token 就完事。

论文把挑战拆得很清楚。Fast-dVLM 基于 Fast-dLLM v2，但从 text-only dLLM 到 VLM 至少要解决四类问题。

- **第一，转换策略问题**。应当先把 LLM backbone 转成 diffusion LLM，再接视觉模块做多模态微调；还是直接把已经预训练好的 AR VLM 一步转换成 diffusion VLM？:mdnice-bold-italic[这不是训练流程偏好，而是决定模型是否能保留已有多模态对齐能力的关键。]
- **第二，多轮边界问题**。VLM 数据里有大量短回答，例如选择题只回答一个字母。如果 block size 固定，而回答长度不是 block size 的倍数，最后一个 denoising block 很可能越过当前回答边界，看到下一轮 prompt token。这样训练时就会出现未来信息泄漏。
- **第三，训练效率问题**。Fast-dLLM v2 采用 noisy stream 与 clean stream 双流设计。:mdnice-bold-italic[如果把视觉 token 同时复制进 noisy 和 clean 两条流，会浪费显存和计算。更重要的是，视觉 embedding 本身并不被 mask，不需要在 noisy stream 中重复出现。]
- **第四，因果兼容问题**。许多 block diffusion 方法在历史上下文上使用 block-level context attention，但这样会破坏预训练 AR 模型原本的 token-level causal structure，也会让同一模型无法自然执行 AR verification，从而影响自投机解码。

::: block-1
Fast-dVLM 的方法部分，本质上就是围绕这四个问题给出工程化可训练方案：直接转换、causal context attention、auto-truncation mask、vision-efficient concatenation，再加上自投机 block decoding 和 SGLang serving。
:::

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/06.webp)

::: image-caption
表 3：训练 recipe 消融实验。表格验证因果上下文、自动截断、块大小退火等组件的作用，移除因果上下文精度暴跌22.5%，是最核心组件；移除退火机制，长文本推理精度降32.5%；移除自动截断，MMMU精度降14.4%。四大训练配方协同工作，因果注意力保留AR推理能力，退火提升大去噪跨度稳定性，自动截断避免多轮对话信息泄露。
:::

## 三、直接转换还是两阶段转换：论文最重要的经验结论

> Fast-dVLM 最有价值的训练结论是：不要先把文本 LLM 转成扩散模型再重建多模态能力，直接从已经对齐好的 AR VLM 转换更有效。

论文系统比较了两种 AR-to-diffusion conversion 路径。

### 3.1 两阶段路径

两阶段路径先从 Qwen2.5-Instruct-3B 这类 AR LLM 出发，用文本数据按 Fast-dLLM v2 recipe 训练出 diffusion LLM。随后再接上视觉编码器和 MLP projector，并在多模态数据上联合微调，得到 diffusion VLM。

::: block-1
这条路线看起来更“模块化”：先解决语言扩散，再解决视觉对齐。但它有一个严重问题：第二阶段实际是在让一个 text-only diffusion LLM 重新学习视觉语言对齐。对于有限训练预算来说，这非常低效。
:::

### 3.2 直接路径

直接路径从 Qwen2.5-VL-3B 这样的完整 AR VLM 出发，在多模态数据上一次性转换成 block-diffusion VLM。它的优势是**继承已有 VLM 预训练中的视觉-语言对齐能力，训练只需要适配生成范式**，而不是重建多模态语义桥。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/07.webp)

::: image-caption
图 2：两种转换策略。图中对比两阶段转换与直接转换路径，两阶段先对 LLM 做纯文本扩散微调，再接入视觉编码器做多模态微调；直接转换则一步将完整预训练 AR VLM 转为块扩散模型。论文实验证实，同等训练预算下直接转换效率远高于两阶段，因直接复用预训练好的视觉 - 文本对齐能力，无需从零重建多模态关联。最终确定直接转换为 Fast-dVLM 的核心技术路线，既简化流程又提升性能。
:::

实验结果非常明确。在 10 个 benchmark 上，直接路径平均分 73.3，两阶段路径平均分 60.2；直接路径在全部 10 个 benchmark 上都超过两阶段路径。差距最大的任务包括 DocVQA、ChartQA 和 AI2D，论文分别报告了 +31.5、+21.4、+18.1 的优势。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/08.webp)

::: image-caption
图 4：直接转换与两阶段转换在10个基准上的对比。雷达图直观显示直接转换平均得分73.3，两阶段仅60.2，在DocVQA、ChartQA、AI2D等推理密集型任务上差距超18分。两组实验的训练数据、算力预算完全一致，直接转换因继承预训练VLM的多模态对齐先验，无需重新学习视觉文本关联。这充分证明直接转换的训练效率优势，是Fast-dVLM选型的核心依据。
:::

这组结果说明，**VLM 的多模态对齐能力不是一个可以轻易在短训练中重建的附属模块**。:mdnice-bold-italic[对于扩散化 VLM 来说，最有效路径不是“从文本扩散模型出发接视觉”，而是“从已经多模态对齐的 AR VLM 出发改生成机制”。]

::: block-1
直接转换路径的胜出，是 Fast-dVLM 全篇最重要的训练经验：扩散目标可以后接，但多模态对齐最好不要从零重学。
:::

论文也保持了谨慎。作者并没有断言两阶段路径的最终上限一定更低，而是假设两者可能有类似 ceiling，差异主要体现在训练预算利用效率上。在相同或可比训练预算下，直接路径显著更划算，因此成为默认 recipe。

## 四、训练设计：Fast-dVLM 如何让块扩散适配多模态输入

> Fast-dVLM 的训练方案不是一个单点技巧，而是一组相互配合的约束：哪些 token 可被 mask，哪些上下文必须因果，哪些视觉 token 不应重复。

设完整输入为：

$$
\mathbf{x}=(\mathbf{v},\mathbf{w})
$$

其中 $\mathbf{v}$ 是视觉 token embedding，$\mathbf{w}$ 是文本 token embedding。Fast-dVLM 只腐蚀 response text token，构造 noisy stream $\mathbf{w}^{t}$，并与 clean stream 拼接为：

$$
[\mathbf{w}^{t};\mathbf{x}]
$$

注意这里 noisy stream 只含文本位置，不重复视觉 token。这一点对应后文的 vision-efficient concatenation。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/09.webp)

::: image-caption
图 3：Fast-dVLM的训练架构与注意力掩码（块大小$\hat{B}=2$）。该图展示双流（噪声+干净）训练架构与三类注意力掩码规则，视觉token仅保留在干净流，噪声流仅含文本token。$M_{N2N}$实现块内双向并行去噪，$M_{N2C}$让噪声token关注前文干净视觉与文本信息，$M_{C2C}$保持干净流因果注意力。此设计无精度损失，却降低15.0%峰值内存、减少14.2%训练时间，还保留AR解码能力支撑自推测验证，是多模态块扩散的关键架构创新。
:::

### 4.1 Causal context attention：保留 AR 模型的因果结构

> 训练 attention mask 包含三部分。

1. $\mathcal{M}_{N2N}$：noisy tokens 在同一 block 内双向注意力，用于并行去噪。
2. $\mathcal{M}_{N2C}$：noisy tokens 可以看见前面 clean blocks，包括视觉 token。
3. $\mathcal{M}_{C2C}$：clean tokens 继续使用 token-level causal attention。

Fast-dVLM 与一些 block diffusion 方法不同，它没有在所有前置上下文上使用 block-level context attention，而是保留 causal attention。**这对两个目标都关键：一是保护 AR VLM 预训练得到的顺序推理表征，二是让同一模型仍能执行 AR decoding，用于 self-speculative verification。**

::: block-1
消融结果也证明了这一点。去掉 causal context，改成对前置上下文使用 block-level bidirectional attention，会导致平均分从 57.3 降到 44.4，下降 22.5%。

在 MMMU-Pro-V 上相对下降 58.9%，在 SeedBench2+ 上下降 39.5%。**这说明 causal context 不是可有可无的兼容项，而是保持推理能力的核心。**
:::

### 4.2 Block-size annealing：不要一上来就让模型处理大块噪声

> Fast-dVLM 采用 block size curriculum。候选 block size 为：

$$
S=\{2^1,2^2,\dots,B_d\}
$$

训练进度为 $u\in[0,1]$ 时，当前 block size 为：

$$
\hat{B}=S[\min(\lfloor u\cdot |S|\rfloor, |S|-1)]
$$

直觉上，模型先学习小块、细粒度 denoising，再逐渐处理更长 span 的 corruption。目标 block size 是 $B_d=32$。

消融显示，如果直接用目标 block size 32 训练，不做 annealing，平均分下降 4.4%，其中 MMMU-Pro-V 下降 32.5%。长推理任务对稳定去噪尤其敏感，因此直接大块训练会明显伤害 chain-of-thought 质量。

### 4.3 Auto-truncation mask：防止短回答跨越下一轮 prompt

> 多轮 VLM 数据中，回答长度常常不是 block size 的整数倍。如果最后一个 block 不截断，就可能跨过 response 边界进入下一轮 prompt。由于 block 内 noisy tokens 可以通过 $\mathcal{M}_{N2N}$ 双向注意力互相看见，这会造成 future prompt leakage。

Fast-dVLM 的 auto-truncation attention mask 会在每个 response 的最后一个 block 自动截断到 response boundary，防止跨轮信息泄漏。去掉该机制后，平均分下降 3.7%，MMMU 下降 14.4%。这说明**短答案和多轮边界不是数据清洗小问题，而是扩散 VLM 训练中必须显式处理的注意力语义问题。**

### 4.4 Vision-efficient concatenation：视觉 token 不要复制两份

视觉 embedding 在训练中不被 corruption，因此 noisy stream 和 clean stream 中的视觉信息本质相同。Fast-dVLM 将视觉 token 只放在 clean stream 中，让 noisy text token 通过 $\mathcal{M}_{N2C}$ 访问视觉上下文。

论文报告，在 Qwen2.5-VL-3B、H100、context length 2048 条件下，**这个无损设计将 peak memory 降低 15.0%，训练时间降低 14.2%。** 这不是改变模型能力的技巧，而是对多模态 diffusion 双流结构的一次必要工程修剪。

### 4.5 训练目标：扩散损失与因果 LM 损失共同约束

Fast-dVLM 同时训练 noisy stream 的 diffusion branch 与 clean stream 的 causal LM branch。设语言模型头为 $W$，noisy 和 clean stream 的 hidden states 分别为 $\mathbf{H}^{(t)}$ 与 $\mathbf{H}^{(0)}$，总损失为：

$$
\mathcal{L}=\alpha\,\mathrm{CE}(W\mathbf{H}^{(t)},\mathbf{y})+\beta\,\mathrm{CE}(W\mathbf{H}^{(0)},\mathbf{y})
$$

论文默认 $\alpha=\beta=0.5$。这个目标很有代表性：一边让模型学会 block 内并行 denoising，另一边保留 AR generation capability。后者直接服务于后续自投机解码的 verify step。

## 五、推理机制：从 MDM 到自投机块解码

> Fast-dVLM 的推理加速不是单一来源，而是由块扩散、多 token 接受、系统 serving 和量化逐层叠加出来的。

### 5.1 Causal context decoding：每个 block 先由一个 AR token 启动

Fast-dVLM 的 block 解码沿用 KV-cache-compatible 的顺序。每个 block 开始时，模型先用一次 AR step 从 cached causal context 中生成第一个 token；剩余 $\hat{B}-1$ 个位置填入 [MASK]，再通过迭代去噪生成。

这样设计让训练时的 causal context attention 与推理时上下文对齐，也让已完成 block 的 KV cache 可以继续被后续 block 复用。

### 5.2 MDM decoding 的阈值权衡

> MDM decoding 中，置信度阈值 $\tau$ 控制每轮可以揭示多少 token。阈值越高越保守，质量更稳但并行度低；阈值越低越激进，吞吐更高但质量下降。

论文在 MMMU-Pro CoT 上扫描 $\tau$：

1. $\tau=1.0$ 时，每步只揭示一个 token，准确率 21.6。
2. **$\tau=0.9$ 时，吞吐接近翻倍到 1.95 tokens/step，同时准确率仍为 21.4。**
3. $\tau=0.4$ 时，吞吐提升到 2.90 tokens/step，但准确率降到 18.5。

因此论文默认采用 $\tau=0.9$，作为速度和质量之间的平衡点。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/10.webp)

::: image-caption
图 5：阈值$\tau$对MMMU-Pro CoT精度（左轴）和每步token数（右轴）的影响。阈值$\tau$调控MDM解码的并行度与精度平衡，$\tau=1.0$时每步仅生成1个token，精度21.6；$\tau=0.9$时每步token数近翻倍，精度仍维持21.4；$\tau=0.4$时并行度最大化但精度跌至18.5。论文选定$\tau=0.9$为默认参数，实现速度与精度的最优配比，验证了置信度感知并行解码在块扩散VLM中的实用性。
:::

::: block-1
上图展示了阈值 $\tau$ 对 MMMU-Pro CoT 准确率与 tokens per step 的影响——MDM decoding 的核心 trade-off：降低阈值可以增加并行 token 数，但会牺牲生成质量。
:::

### 5.3 Self-speculative block decoding：同一个模型既 draft 又 verify

> Fast-dVLM 的 self-speculative decoding 利用同一套权重支持两种 attention mode：

1. diffusion mode：块内双向注意力，一次 draft 多个 token。
2. causal mode：自回归注意力，逐 token 验证 draft 是否与 AR distribution 一致。

论文提出两种变体。

**线性 speculative decoding 每个 block 用两次 forward：**

1. draft step：追加 $B-1$ 个 mask token，用双向注意力并行预测整个 block。
2. verify step：用 causal attention 重新评估填好的 block，从左到右比较 draft 与 AR prediction，接受最长匹配前缀，并裁剪 KV cache。

::: block-1
这个方案每个 block 需要 2 次 NFE，理论最高加速约为 $B/2$。
:::

**二次 speculative decoding 则把 verification 和 next-block proposal 融合到一次 forward。** 它把 $B$ 个 draft token 展开成 $B$ 组，每组 $B+1$ 个 token，总输入长度为：

$$
B(B+1)
$$

::: block-1
它的 NFE 更少，但计算量是 $O(B^2)$，并且 attention mask 不标准，当前 kernel 优化不够。因此实验里，quadratic variant 的 Tokens/NFE 更高，但 wall-clock TPS 反而低于 linear variant。
:::

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/11.webp)

::: image-caption
图 6：不同块大小下推测解码的吞吐量对比。图中对比线性与二次推测解码的Tokens/NFE和TPS指标，Tokens/NFE随块大小增大单调提升；线性解码在块大小16时TPS达峰值112.7，二次解码因$O(\hat{B}^2)$计算开销，实际时钟速度始终更低。线性方案更适配大块大小，二次方案仅适合中等块大小，为推理部署的块大小选择提供了明确的工程指导。
:::

从上图可以看出：Tokens/NFE 随 block size 增大而提升，但 wall-clock TPS 在 block size 16 附近达到峰值；二次方案的理论 NFE 优势被 $O(B^2)$ 输入成本抵消。

::: block-1
最终，**论文默认使用 linear variant。它在 block size 16 时达到 112.7 TPS，相比 AR baseline 的 56.7 TPS 为 1.98× wall-clock 加速**，MMMU-Pro-V 分数也从 MDM 的 21.4 恢复到 24.6，更接近 AR baseline 的 26.3。
:::

### 5.4 SGLang 与 FP8：系统优化把算法收益兑现到端到端

> 论文没有停在算法层，而是将 Fast-dVLM 集成到 SGLang 中**。关键改动是扩展 scheduler，以支持交替的 bidirectional-draft attention 和 causal-verify attention，并共享同一 paged KV cache**。:mdnice-bold-italic[这让 Fast-dVLM 可以使用 SGLang 的 optimized kernels 和 CUDA graph。]

推理加速路径是逐层叠加的：

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/12.webp)

::: image-caption
表 4 展示 MMMU-Pro-V上的推理的逐层优化的加速效果，MDM解码提速1.45×，叠加推测解码达1.98×，集成SGLang后5.63×，结合FP8量化最终实现6.18×提速。所有测试在单H100、单批次下完成，贴合物理AI单请求部署场景。结论是系统级优化（SGLang）、量化与算法级（推测解码）协同，最大化块扩散VLM的推理加速收益。
:::

这里需要注意：6.18× 不是单纯来自扩散解码本身，而是“扩散解码 + 自投机 + serving 系统 + FP8 量化”的端到端组合结果。**论文在 abstract 中说 over 6×，图 1 和表 4 给出的具体数值是 6.18×。**

::: block-1
Fast-dVLM 真正的系统意义在于：它没有只追求 Tokens/NFE 这种算法指标，而是把 SGLang 和 FP8 纳入路径，**证明块扩散 VLM 可以转化为真实 wall-clock 吞吐收益。**
:::

## 六、实验结果：质量是否真的保住了

> Fast-dVLM 的实验结论可以概括为：短答任务基本保住质量，长链式推理仍有差距，但自投机解码能显著缩小这个差距。

论文以 Qwen2.5-VL-3B 为 AR baseline，评估 11 个 VLM benchmark。短答任务包括 AI2D、ChartQA、DocVQA、GQA、MMBench、MMMU、POPE、RealWorldQA、SEEDBench2+、TextVQA；长答任务为 MMMU-Pro-V。

### 6.1 短答任务：平均质量匹配 AR 基线

> Qwen2.5-VL-3B AR baseline 的短答平均分是 74.0。Fast-dVLM 的结果为：

1. MDM decoding：平均分 73.3，Tokens/NFE 为 1.95×。
2. speculative decoding：平均分 74.0，Tokens/NFE 为 2.63×。

也就是说，**在短答任务上，Fast-dVLM speculative decoding 精确匹配了 AR baseline 的平均质量**，同时每次 forward 平均产生更多 token。它还:mdnice-bold-italic[在 GQA、POPE、RealWorldQA 等任务上达到或超过基线。论文解释认为，block 内双向上下文可能对整体视觉推理有帮助。]

::: block-1
与其他 diffusion VLM 相比，Fast-dVLM 在 11 个短答 benchmark 中有 8 个达到最佳 diffusion 模型结果，**整体显著强于 LaViDa、Dimple、LLaDA-V 等已有扩散 VLM 基线。**
:::

### 6.2 长答任务：MMMU-Pro-V 仍是难点

> MMMU-Pro-V 要求多步 chain-of-thought reasoning，是对块扩散生成最不友好的场景之一，因为它更依赖长距离顺序一致性。

论文给出的结果是：

1. AR baseline：26.3。
2. Fast-dVLM MDM：21.4，落后 4.9 分。
3. Fast-dVLM speculative：24.6，只落后 1.7 分。

::: block-1
这说明普通 MDM 并行去噪在长推理上仍有结构性劣势，但自投机 verify 能显著修正生成轨迹，把质量拉回接近 AR。论文也承**认剩余差距可能需要更大规模训练数据和更长 annealing schedule 进一步缩小。**
:::

### 6.3 消融：哪些组件最关键

> 论文在 ShareGPT-4V 数据上做了训练 recipe 消融，结果很清晰。

1. 去掉 causal context：平均分从 57.3 降到 44.4，下降 22.5%，是最大损伤。
2. 去掉 block-size annealing：平均分降到 54.8，下降 4.4%，MMMU-Pro-V 下降 32.5%。
3. 去掉 auto-truncation：平均分降到 55.2，下降 3.7%，MMMU 下降 14.4%。

::: block-1
这说明 Fast-dVLM 的成功并不是“只要直接转换就够了”。直接转换提供了强初始化，但 causal context、annealing 和 turn-aware truncation 是让模型真正稳定工作的关键条件。
:::

## 七、把 Fast-dVLM 的核心机制压成一条链

> Fast-dVLM 的技术链条可以简化为：直接转换保留多模态对齐，块扩散释放并行度，因果上下文保留 AR 能力，自投机解码把质量拉回，SGLang 和 FP8 把吞吐落到系统指标上。

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/13.webp)

::: image-caption
图 1：Fast-dVLM概述。该图从速度-精度权衡、多基准测试、端到端推理加速三个维度呈现Fast-dVLM核心效果。速度-精度维度中，Fast-dVLM在MMMU-Pro-V任务上精度比肩自回归（AR）基线且实现大幅提速；多基准测试里，其基于Qwen2.5-VL-3B基座在各多模态任务上精度近乎无损；推理加速结合SGLang与FP8量化，最终实现超6倍端到端提速。这验证块扩散架构可突破AR解码吞吐量瓶颈，完美适配机器人、自动驾驶等单批次边缘物理AI部署场景。
:::

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/14.webp)

::: image-caption
图 3：Fast-dVLM的训练架构与注意力掩码（块大小$\hat{B}=2$）。该图最重要的信息是视觉 token 只在 clean stream 出现，noisy token 通过 clean context 访问视觉信息；同时 clean stream 保持因果注意力。
:::

![文章插图](/articles/2026-05-13-fast-dvlm-batch-vlm-sglang-bba7517f/images/15.webp)

::: image-caption
图 5：阈值$\tau$对MMMU-Pro CoT精度（左轴）和每步token数（右轴）的影响。阈值$\tau$调控MDM解码的并行度与精度平衡，$\tau=1.0$时每步仅生成1个token，精度21.6；$\tau=0.9$时每步token数近翻倍，精度仍维持21.4；$\tau=0.4$时并行度最大化但精度跌至18.5。论文选定$\tau=0.9$为默认参数，实现速度与精度的最优配比，验证了置信度感知并行解码在块扩散VLM中的实用性。
:::

上图是 MDM decoding 的阈值曲线。$\tau=0.9$ 是论文采用的默认点，因为它在几乎不损害准确率的情况下，将 tokens/step 提高到 1.95。

> Fast-dVLM 的核心公式并不复杂，但每个公式都对应一个系统设计。

自回归分解说明传统生成为什么串行：

$$
P_{\theta}(\mathbf{x})=\prod_i P_{\theta} (x_i | x_{<{i}})
$$

Block-size annealing 说明模型如何从小块逐步过渡到目标 block size：

$$
\hat{B}=S[\min(\lfloor u\cdot |S|\rfloor, |S|-1)]
$$

双分支训练目标说明为什么模型既能做 diffusion denoising，又能保留 AR verification 能力：

$$
\mathcal{L}=\alpha\,\mathrm{CE}(W\mathbf{H}^{(t)},\mathbf{y})+\beta\,\mathrm{CE}(W\mathbf{H}^{(0)},\mathbf{y})
$$

线性自投机解码的理论上限说明了 block size 与加速之间的关系：

$$
\text{speedup}_{\max}\approx \frac{B}{2}
$$

二次自投机解码的成本则解释了为什么 Tokens/NFE 更高并不必然转化为 wall-clock TPS：

$$
\text{input cost}=O(B^2)
$$

::: block-1
这些公式共同指向一个结论：Fast-dVLM 的核心不是单纯追求每次 forward 预测更多 token，而是**在模型质量、KV cache、attention mask、serving kernel 和量化效率之间做平衡。**
:::

## 八、相关工作：Fast-dVLM 站在扩散 LLM、扩散 VLM 和投机解码的交汇处

> Fast-dVLM 不是第一篇扩散 VLM 工作，但它的定位更偏系统化：既要块级 KV 缓存，又要 VLM 质量，还要 serving 级吞吐。

### 8.1 离散扩散语言模型

离散扩散 LLM 已经从早期连续 latent 形式，发展到 masked diffusion objective。LLaDA、Dream 等模型证明扩散语言模型可以扩展到 7B–8B 参数并接近 AR baseline。:mdnice-bold-italic[Fast-dLLM 和 Fast-dLLM v2 进一步把 block-wise caching 和 AR-to-diffusion adaptation 做成更高效 recipe。]

Fast-dVLM 继承的正是这条线：**块与块之间保持顺序，块内部并行去噪，用这种方式突破全序列双向扩散无法增量缓存的限制。**

### 8.2 扩散视觉语言模型

LLaDA-V、LaViDa、MMaDA、Dimple 等工作都在尝试把离散扩散引入多模态理解。但论文指出，这些方法大多依赖 full-sequence diffusion，没有 block structure，因此难以对已完成 response block 做 incremental KV caching，也没有充分处理 multi-turn dialogue 中的 turn-aware attention masking。

Fast-dVLM 的差异在于，它明确把 VLM 扩散化和服务系统兼容性绑定在一起：**没有 KV-cache-compatible block diffusion，就很难在真实推理中兑现速度。**

### 8.3 投机解码与扩散投机解码

经典 speculative decoding 依赖 draft model 和 verify model。:mdnice-bold-italic[扩散 LLM 的特殊之处在于，它天然具备多 token prediction 能力，因此可以让同一个模型用 diffusion mode draft，再用 causal mode verify]。**Fast-dVLM 把这种 self-speculation 引入多模态扩散 VLM**，是论文声称的首个将 block-level self-speculation 集成到 multimodal diffusion VLM 并带有系统 serving 支持的工作。

::: block-1
这一点非常关键：如果没有 causal branch 和 causal attention 保留，模型就很难同时当 drafter 和 verifier。因此训练阶段的 AR loss，并不是为了附带保留一个老功能，而是为推理阶段的自投机路径埋下接口。
:::

## 九、结论与展望

> Fast-dVLM 说明，VLM 推理提速不能只靠更快 kernel，也不能只靠更小模型；生成范式本身也可以被重新设计。

### 9.1 结论总结

Fast-dVLM 的贡献可以总结为四点。

1. 它提出了从完整 AR VLM 直接转换为 block-diffusion VLM 的训练路线，并通过对比证明直接路径比两阶段路径更高效。
2. 它把 causal context attention、auto-truncation masking、block-size annealing、vision-efficient concatenation 组合成一套适合 VLM 的扩散训练 recipe。
3. 它通过 self-speculative block decoding，让同一模型在 diffusion mode 中 draft，在 causal mode 中 verify，显著改善长输出质量与吞吐。
4. 它将模型接入 SGLang 并结合 FP8 量化，使算法层的 Tokens/NFE 优势转化为 6.18× 的端到端 wall-clock speedup。

::: block-1
在短答任务上，Fast-dVLM speculative decoding 的平均分达到 74.0，与 Qwen2.5-VL-3B AR baseline 持平；:mdnice-bold-italic[在 MMMU-Pro-V 长推理上，它仍落后 1.7 分，但相比 MDM 的 4.9 分差距已经明显缩小]。**这是一个现实而可信的结果：并行生成确实有质量代价，但通过自投机验证可以显著缓解。**
:::

### 9.2 进阶分析

> 这篇论文最强的地方，是它没有把“扩散 VLM”停留在概念层，而是**完整打通了训练、解码、系统和量化**。但它也有几个边界需要明确。

- 第一，6.18× 是在 MMMU-Pro-V、单 H100、batch size 1、结合 SGLang 与 FP8 量化后的端到端结果，不应被泛化为所有 VLM 任务上的固定倍数。**短输出任务里，块扩散的优势可能没有长输出任务那么明显。**
- 第二，长链式推理仍然是 block diffusion 的难点。**Fast-dVLM speculative 在 MMMU-Pro-V 上仍低于 AR baseline**，这说明对于高度顺序依赖的 reasoning，AR verification 只能部分补偿并行 denoising 的结构差异。
- 第三，论文主要基于 Qwen2.5-VL-3B 规模验证。直接转换是否能同样稳定扩展到更大 VLM，例如 7B、14B 或更强推理模型，还需要后续实证。
- 第四，SGLang 对该范式的支持是重要工程贡献，但也意味着未来不同 serving backend、不同 attention kernel、不同硬件对最终加速比会有显著影响。Fast-dVLM 的算法优势需要系统栈共同配合才能完全释放。

::: block-1
Fast-dVLM 不是“扩散模型必然取代自回归”的证据，而是证明了一条更具体的路线：**在单请求、长输出、物理 AI 式 VLM 推理中，块扩散加自投机解码可以成为比纯自回归更高效的生成范式。**
:::

### 9.3 未来工作

> 从原文结果出发，未来值得继续推进的方向至少有五个。

1. 扩展模型规模。**验证直接转换 recipe 在更大 VLM 上是否仍然训练高效，尤其是强推理模型和更长上下文模型。**
2. 改进长链式推理。针对 MMMU-Pro-V 这类任务，探索更长 annealing、更高质量 CoT 数据、更强 verifier 或混合 AR/diffusion 策略。
3. 优化 quadratic speculation kernel。论文显示二次方案 Tokens/NFE 更高但 TPS 更低，**说明它需要专门 attention kernel 才能兑现理论优势。**
4. 推广到更多物理 AI 场景。论文附录给出自动驾驶和机器人案例，但真正部署仍需研究延迟抖动、流式输入、多帧视觉上下文和安全约束。
5. 联合模型压缩与扩散解码。FP8 已经带来系统收益，未来还可以结合稀疏化、低秩适配、视觉 token 压缩和动态 block size 策略。

::: block-1
最终看，Fast-dVLM 的意义不只是“又一个 VLM 加速方法”。它更像是一个信号：:mdnice-bold-italic[当多模态模型进入物理世界，逐 token 自回归不再是唯一默认答案。下一阶段的 VLM 推理系统，很可能会在自回归、扩散、投机解码和 serving runtime 之间寻找新的组合形态。]
:::
