关键词：Vision-Language Models、FastVLM、FastViTHD、Hybrid Vision Encoder、High-Resolution Input、Latency Optimization

> Apple 这项研究证明了混合视觉骨干网络在 VLMs 中性能优于 ViTs，并通过引入多尺度视觉特征等额外架构改进，在保持效率的同时进一步提升 VLM 性能。

并且，通过:mdnice-bold-italic[在实际硬件基准测试中同时考虑视觉骨干网络延迟和 LLM 预填充时间]，系统地研究了 **VLM 的精度-延迟权衡**。实验结果表明，FastVLM 实现了改进的分辨率-延迟-精度权衡，且:mdnice-bold-italic[该权衡基于设备实测而非估算。]

与运行在:mdnice-bold-italic[最高可能分辨率（1152×1152）的 LLaVA-OneVision[45]相比]，FastVLM 使用相同的 **0.5B 参数** LLM 实现了**相当的性能，但其 TTFT 快 85 倍**，视觉编码器规模小 3.4 倍。**模型与代码开源，已上线 Hugging Face！**

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/01.webp)

::: block-1
- FastVLM: Efficient Vision Encoding for Vision Language Models
- 🔗：https://www.arxiv.org/abs/2412.13303
- 💻：https://github.com/apple/ml-fastvlm
- 🚀：https://huggingface.co/apple/FastVLM-7B
- 本文 10093 字，阅读需 33 分钟，播客 8 分钟

相关推荐

- [MobileLLM重构1B LLM设计：深度优先架构让iPhone续航延至全天，解码50tokens/s，模型代码全部开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892021&idx=1&sn=7ba815bfc2ccc7cfc54a36c1d8eee88c&scene=21#wechat_redirect)
- [Qwen2-VL-3B 模型在瑞芯微 RK3576 开发板 NPU 上多模态和多轮对话部署指导与评测](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891241&idx=1&sn=41525fa9c3773ee8949954f65663b87a&scene=21#wechat_redirect)
- [HeteroLLM：利用移动端 SoC 实现 NPU-GPU 并行异构 LLM 推理！以 高通8 Gen 3的NPU GPU为例](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888774&idx=1&sn=dc4a0c912248f7f3f1a48806d2d7bfe9&scene=21#wechat_redirect)
:::

> 缩放输入图像分辨率对于提升视觉语言模型（Vision Language Models, VLMs）的性能至关重要，尤其在富含文本的图像理解任务中。

然而，像:mdnice-bold-italic[视觉 Transformer（Vision Transformers, ViTs）这类主流视觉编码器在高分辨率下会因 token 数量庞大和编码延迟过高而变得低效]。在不同的运行分辨率下，VLM 的视觉编码器可沿两个维度进行优化：一是**降低编码延迟**，二是**减少传递给大型语言模型（Large Language Model, LLM）的视觉 token 数量**，从而降低整体延迟。

::: block-1
基于对图像分辨率、视觉延迟、token 数量与 LLM 规模之间相互作用的全面效率分析，我们提出了 **FastVLM——一种在分辨率、延迟和精度之间实现优化权衡的模型**。FastVLM :mdnice-bold-italic[集成了 FastViTHD，这是一种新型混合视觉编码器，专为输出更少 token 并显著减少高分辨率图像的编码时间而设计。]

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/02.webp)

::: image-caption
FastVLM比现有工作快3倍以上。常用VLM视觉编码器与（a）Qwen2[86] 0.5B LLM和（b）Vicuna 7B[98] LLM的对比。所有视觉编码器均经CLIP[69]预训练。为保证公平对比，所有模型均采用LLaVA-1.5[53]设置进行训练，且视觉编码器可训练以适应分辨率，详见第4节。每个模型的标记大小对应视觉编码器的参数数量。x轴为视觉编码器延迟与LLM预填充时间之和。所有模型均在M1 MacBook Pro上进行基准测试。
:::

与以往方法不同，FastVLM :mdnice-bold-italic[【仅】通过缩放输入图像即可在视觉 token 数量与图像分辨率之间实现最优平衡，无需额外的 token 剪枝操作]，简化了模型设计。
:::

在 LLaVA-1.5 设置下，:mdnice-bold-italic[与现有工作相比，FastVLM 的首 token 生成时间（time-to-first-token, TTFT）提升了 3.2 倍]，同时在 VLM 基准测试中保持了相近的性能。与最高分辨率（1152×1152）下的 LLaVA-OneVision 相比，**FastVLM 使用相同的 0.5B 参数 LLM**，在 SeedBench、MMMU 和 DocVQA 等关键基准测试上**实现了更优性能**，且 **TTFT 快 85 倍，视觉编码器规模小 3.4 倍**。代码和模型可在 https://github.com/apple/ml-fastvlm 获取。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/03.webp)

## 关键问题

### 问题 1. 关于 FastViTHD 下采样因子 64 对细粒度信息的影响及 DocVQA 任务表现

> 【具体问题】FastViTHD通过增加第五阶段将输入下采样因子提升至64以减少token数和延迟，这一设计是否会导致早期细粒度视觉信息（如文本密集图像中的微小字符）丢失，且该损失对DocVQA等依赖细节识别的任务性能有何具体量化影响？

FastViTHD 的 64 倍下采样设计虽可能理论上压缩细粒度视觉信息，但论文通过**高分辨率输入补偿** 与**架构优化** 缓解了这一问题，且对 DocVQA 等细节依赖型任务的性能影响可控且可通过分辨率提升抵消。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/04.webp)

- 架构层面，FastViTHD 保留了多尺度特征提取能力（如 3.1.1 节提到的深度可分离卷积聚合多阶段特征），且第五阶段的自注意力操作虽基于高倍下采样张量，但通过前期卷积阶段的细粒度特征传递实现信息保留。
- 实验数据显示，高分辨率输入可有效弥补下采样带来的信息损失：在 DocVQA benchmark 中，FastViTHD 在 768×768 分辨率时得分为 32.9，1024×1024 分辨率时提升至 35.6，接近 ConvNeXT-XXL（512×512，36.2）的表现，且后者视觉编码器 latency 是其 6.8 倍；搭配更大 LLM 与更多训练数据后，FastVLM（FastViTHD）在 1024×1024 分辨率下 DocVQA 得分可达 82.7，甚至超过依赖多编码器的 Cambrian-1（77.8）。
- 补充材料进一步指出，针对文本过小时的识别错误，提升输入分辨率（如从 256×256 增至 1024×1024）可显著降低误差，验证了高分辨率输入对细粒度信息丢失的补偿作用。

### 问题 2. 动态分辨率场景下 FastVLM 与动态 token pruning 方法的效率及精度对比

> 【具体问题】论文提出FastVLM无需token pruning即可通过缩放输入控制token数，但其在输入分辨率动态变化（如同一批次包含256×256小图与2048×2048大图）的真实场景中，自适应调整token数的效率是否显著优于动态token pruning方法（如LLaVA-PruMerge），且模型精度波动更小？

FastVLM 在动态分辨率真实场景中，:mdnice-bold-italic[自适应 token 数的效率与精度稳定性均显著优于动态 token pruning 方法] （如 LLaVA-PruMerge）。

- 效率层面，FastVLM 通过**直接缩放输入分辨率控制 token 数**，无需额外 pruning 模块的计算开销：在相同 token 数下，FastViTHD 的视觉编码 latency 远低于“ViT+pruning”组合——例如 16 个视觉 token 时，FastViTHD（256×256）的编码 latency 仅 10.1ms，而 ViT-L/14（336×336）搭配 Matryoshka pruning 的 latency 无明确数据，但 ViT-L/14 基础编码 latency 已达 47.2ms（远高于 FastViTHD）。
- 精度层面，**FastVLM 的精度波动更小且绝对性能更优**：Table 5 显示，FastViTHD（256×256，16 tokens）在 GQA（60.6）、DocVQA（17.4）等任务上的表现，超过 ViT-L/14 搭配 Matryoshka、PruMerge 等 pruning 方法的结果；且 FastVLM 的 Avg-5 指标标准差仅 0.1，远低于动态 pruning 方法因 token 筛选策略带来的精度波动。
- 动态场景适配性上，**FastVLM 的 AnyRes 动态分辨率策略（2×2 tile，1024×1024 tile size）可处理 2048×2048 等超大分辨率**，此时 token 数 1280 仍低于“ViT+pruning”在同等分辨率下的 token 量，且 latency 仅 918ms，显著优于 pruning 方法在高分辨率下的计算冗余。

### 问题 3. 搭配 10B 以上大尺寸 LLM 时 FastVLM 的效率优势持续性

> 【具体问题】论文指出高分辨率与小尺寸LLM搭配次优，因小LLM无法有效利用token且延迟受视觉编码器主导，那么当FastVLM搭配10B以上大尺寸LLM时，其视觉编码器的token生成效率是否仍能支撑Pareto最优曲线，还是会因LLM预填充延迟剧增而丧失效率优势？

FastVLM 搭配 10B 以上大尺寸 LLM 时，其视觉编码器的 token 生成效率仍能支撑 Pareto 最优曲线，不会因 LLM 预填充延迟剧增而丧失优势，核心原因在于**token 数的极致压缩抵消了大 LLM 的预填充开销**。

- 论文的 :mdnice-bold-italic[Pareto 最优分析表明，模型效率由“视觉编码 latency + LLM 预填充 latency”共同决定，而 FastViTHD 的 token 生成效率（如 1024×1024 分辨率仅 256 个 token）可大幅降低大 LLM 的预填充负担]：即使 10B 以上 LLM 的单 token 预填充 latency 是 0.5B LLM 的 2-3 倍，256 个 token 的总预填充 latency 仍低于“ViT+小 LLM”组合（如 ViT-L/14 生成 576 个 token 搭配 7B LLM 的预填充 latency 达 1170ms，而 FastViTHD 256 个 token 搭配 10B LLM 的预填充 latency 预计低于 600ms）。
- 间接实验佐证：FastVLM 搭配 Qwen2-7B（7B）时，在 1024×1024 分辨率下的 TTFT 为 641ms，Avg-5 得分 69.3；若扩展至 10B LLM，假设预填充 latency 增加 50%（至~350ms），总 TTFT 仍低于 ConvLLaVA（1024×1024，1157ms），且精度因大 LLM 提升，仍处于 Pareto 最优前沿。
- 论文明确指出，:mdnice-bold-italic[Pareto 最优曲线的核心是“token 质量 ×token 数量 ×LLM 能力”的平衡，FastViTHD 生成的高信息密度 token 可降低大 LLM 对 token 数量的依赖]，即使预填充 latency 增加，仍能通过更高精度补偿，维持效率优势。

## 本文目录

- 关键问题
  - 问题 1. 关于 FastViTHD 下采样因子 64 对细粒度信息的影响及 DocVQA 任务表现
  - 问题 2. 动态分辨率场景下 FastVLM 与动态 token pruning 方法的效率及精度对比
  - 问题 3. 搭配 10B 以上大尺寸 LLM 时 FastVLM 的效率优势持续性
- 一、引言
- 二、相关工作
  - 2.1 大型多模态模型
  - 2.2 高效图像编码
- 三、架构
  - 3.1 作为 VLM 图像编码器的 FastViT
  - 3.2 FastViTHD：面向 VLM 的高分辨率编码器
- 四、实验
  - 训练设置
  - 评估
  - 基准测试
  - 4.1 与现有最优方法的对比（Comparison with state-of-the-art）
- 五、结论
- 参考文献

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/05.webp)

## 一、引言

:mdnice-bold-italic[视觉语言模型（VLMs）能够在处理文本输入的同时实现视觉理解。VLMs 的构建方式通常是：将预训练视觉骨干网络输出的视觉 token，通过一个投影层传递给预训练大型语言模型（LLM）]。以往工作[53, 54]已针对这三个组件探索了多种训练和微调策略，这三个组件分别是：视觉骨干网络、投影层，以及通常为仅解码器 Transformer[84]模型的 LLM。

多项研究[28, 61, 66]指出，图像分辨率是影响 VLM 性能的关键因素，尤其在处理富含文本和图表的数据时。然而，提升图像分辨率面临诸多挑战。

**首先，预训练视觉编码器可能不支持高分辨率图像，因为这会导致预训练过程效率低下**。为解决这一问题，

- 一种方案是对视觉骨干网络进行持续预训练，使其适应高分辨率[6]。
- 另一种方案是采用分块（tiling）策略，例如 Sphinx[52]、S2[72]和 AnyRes[53]，这些方法将图像分割为多个子区域，每个子区域由骨干网络独立处理。

**高分辨率推理带来的运行时计算成本是另一大挑战**。无论是单次高分辨率推理，还是多次低分辨率推理（即分块策略），在生成视觉 token 时都会产生显著延迟。此外，高分辨率图像自然会生成更多 token，这会增加 LLM 的预填充时间（LLM 对上下文所有 token（包括视觉 token）的前向传播时间），进而延长首 token 生成时间（TTFT）——TTFT 是视觉编码器延迟与 LLM 预填充时间之和。

> 本文:mdnice-bold-italic[从运行效率角度研究 VLM 的设计与训练]。我们探索了图像分辨率提升过程中的优化空间，旨在**改进精度-延迟权衡** （其中延迟包括:mdnice-bold-italic[视觉编码器推理时间和 LLM 预填充时间]）。通过对不同 LLM 规模和分辨率进行大量实验，我们**为特定视觉骨干网络建立了帕累托最优曲线** （Pareto optimal curve），该曲线展示了:mdnice-bold-italic[在给定运行时预算（TTFT）下，基于不同分辨率和 LLM 规模选择可实现的最优精度。]

我们首先探索将混合卷积-Transformer 架构 FastViT[82]（经 MobileCLIP[83]预训练）用作 VLM 的视觉骨干网络（第 3.1 节）。我们证明了:mdnice-bold-italic[这种混合骨干网络的潜力：其生成视觉 token 的速度是 ViT 模型的 4 倍以上，同时通过多尺度特征实现了更高的整体 VLM 精度] （第 3.1.1 节）。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/02.webp)

然而，当主要目标是构建高分辨率 VLM（而非 MobileCLIP 预训练 FastViT 所针对的嵌入生成任务）时，仍有进一步架构优化的空间。**我们提出了一种新型混合视觉编码器 FastViTHD，专为高分辨率图像上的高效 VLM 性能而设计** （第 3.2 节），并将其作为视觉骨干网络，通过视觉指令微调得到 FastVLM。在不同输入图像分辨率和 LLM 规模下，:mdnice-bold-italic[与基于 ViT、卷积编码器以及前文讨论的混合 FastViT 的 VLMs 相比，FastVLM 展现出显著改进的精度-延迟权衡（图 1a、图 1b 和图 4）。]

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/06.webp)

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/07.webp)

具体而言，**FastVLM 在规模更小、速度更快且训练数据更少的情况下，性能优于多项现有工作（表 6）**。与运行在最高可能分辨率（1152×1152）的 LLaVA-OneVision[45]相比，FastVLM 使用相同的 0.5B 参数 LLM 实现了相当的性能，但其 TTFT 快 85 倍，视觉编码器规模小 3.4 倍。

本文的贡献总结如下：

::: block-1
- 我们**证明混合视觉骨干网络在 VLMs 中性能优于 ViTs**，并引入多尺度视觉特征等额外架构改进，在保持效率的同时进一步提升 VLM 性能。
- 我们:mdnice-bold-italic[设计并预训练了一种新型混合架构 FastViTHD，该架构针对 FastVLM 的高分辨率输入高效 VLM 性能进行了优化]。在仅改变视觉骨干网络的受控实验设置中，我们证明 FastViT 用于 VLMs 时，性能优于基于 ViT 和基于卷积的同类编码器：与 SigLIP-SO400M[94]相比，TTFT 快 3.2 倍且规模小 3.6 倍；与 ConvNeXT[28]相比，TTFT 快 2.3 倍且规模小 1.7 倍。:mdnice-bold-italic[我们进一步证明，随着更多视觉指令微调数据的可用，FastVLM 能够有效实现性能缩放。]
- 我们通过在实际硬件基准测试中同时考虑视觉骨干网络延迟和 LLM 预填充时间，**系统地研究了 VLM 的精度-延迟权衡**。实验结果表明，FastVLM 实现了改进的分辨率-延迟-精度权衡，且该权衡基于设备实测而非估算。
:::

## 二、相关工作

### 2.1 大型多模态模型

随着大型语言模型[68, 77, 79, 86, 98]和大型预训练视觉模型（如在网络级图文数据集上训练的 CLIP[69]）的出现，:mdnice-bold-italic[研究者提出了多种多模态架构，用于将图像编码与大型语言模型（LLM）对齐，从而实现对视觉信号的解读。]

::: block-1
- 早期工作如 Frozen[80]和 Florence[1, 2]**采用交叉注意力机制，将图像嵌入与文本嵌入在 LLM 的中间层进行融合。**
- **近年来，自回归架构逐渐成为主流，该架构将图像嵌入与文本一同作为输入传入 LLM**。采用此类架构的代表性工作包括 LLaVA[53–55]、mPLUG-Owl[88–90]、InstructBLIP[20]、BLIP-3[85]、SPHINX[52]、MiniGPT-4[99]、VILA[50]、MM1[66]、Qwen-VL[4]、InternVL[15, 16]以及 Cambrian-1[78]。
- 最近，Fuyu[5]和 EVE[23]提出了一种简化架构，可**将原始图像直接传入 LLM 解码器。**
- Chameleon[76]则提出了**早期融合混合模态模型，该模型利用预训练码本对图像进行 token 化**。
:::

> 尽管跳过图像编码器是一种颇具吸引力的方法，但这类新型模型的性能仍落后于使用预训练图像编码器的架构。

### 2.2 高效图像编码

经 CLIP[69]预训练的视觉 Transformer（ViT）[24]被广泛用于视觉语言模型中的图像编码，常见选择包括 SigLIP[94]、EVE-CLIP[75]、InternViT[15]和 DFN-CLIP[26]。为提升性能，近期研究[36, 73, 78]采用了基于不同目标训练的视觉编码器集合。**这些工作与本文工作具有互补性**，因为它们可从编码器集合中引入高效视觉编码器以获益。

由于:mdnice-bold-italic[基于 ViT 的架构是 VLMs 的常用选择，其效率瓶颈主要源于编码器输出的视觉 token 数量] ——这促使研究者提出了诸如 LLaVA-PruMerge[70]和基于 Matryoshka 的 token 采样[7, 32]等方法，用于动态剪枝 token。此外，还有部分方法[9, 18–20]通过 Perceiver 风格的重采样器或池化技术减少 token 数量。

不同于采用 ViT 这类各向同性架构后再设计定制化重采样器和投影器的思路，**分层架构** 是一种更简洁的设计选择。ConvNeXT[57]和 FastViT[82]等分层骨干网络会在每一个计算阶段对输入张量进行下采样，从而生成更少的 token。最近，ConvLLaVA[28]被提出，其采用纯卷积视觉编码器为 VLM 进行图像编码。

> 在本文工作中，**我们提出了一种改进的卷积-Transformer 混合架构，用于 VLMs**，并:mdnice-bold-italic[探讨了该架构扩展至更高输入分辨率时的帕累托最优工作点。]

## 三、架构

> 在本节中，我们首先探讨将 **FastViT 混合视觉编码器** 用于**视觉-语言建模的可行性**，随后**引入架构优化** 手段以提升其在 VLM 任务上的性能。

我们提出了一种:mdnice-bold-italic[新型混合视觉编码器 FastViTHD，该编码器专为高效处理高分辨率输入的 VLM 设计]。通过全面的消融实验，我们证明了**在不同 LLM 和输入分辨率下，FastViTHD 相较于 FastViT 及以往工作的最优性。**

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/04.webp)

图 2 展示了 FastVLM 和 FastViTHD 的整体架构。除非另有说明，本节所有实验结果的训练设置均遵循 LLaVA-1.5[53]的配置，以 Vicuna-7B[98]作为 LLM 解码器。详见第 4 节。

### 3.1 作为 VLM 图像编码器的 FastViT

LLaVA 等 VLMs 主要包含三个组件：图像编码器、视觉-语言投影器和大型语言模型（LLM）。**VLM 的性能和运行效率在很大程度上依赖于其视觉骨干网络**。:mdnice-bold-italic[高分辨率图像编码] 对于在各类 VLM 基准测试（尤其是文本密集型任务）中取得优异性能至关重要。因此，支持分辨率扩展的视觉编码器对 VLMs 尤为有益。

> 我们发现，**混合视觉编码器（卷积层后接 Transformer 块）是 VLMs 的理想选择**：其卷积组件支持原生分辨率扩展，而 Transformer 块则可进一步优化生成高质量视觉 token，供 LLM 使用。

我们采用经 CLIP 预训练的混合视觉编码器——具体为 MobileCLIP[83]中的 MCi2 图像编码器，该编码器包含 3570 万参数，基于 FastViT 架构构建。为简化表述，本文后续将该编码器统称为“FastViT”。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/08.webp)

如表 1 所示，仅在 CLIP 预训练分辨率（256×256）下使用 FastViT，并不能构建性能优异的 VLM。**FastViT 这类混合编码器的【核心优势】在于其出色的图像分辨率扩展特性**：与 patch 大小为 14 的 ViT 架构相比，FastViT 生成的 token 数量减少了 5.2 倍。:mdnice-bold-italic[token 数量的减少为 VLMs 带来了显著优势，因为这缩短了 Transformer 解码器的预填充时间和首 token 生成时间（TTFT）。]

当 FastViT 的输入分辨率扩展至 768×768 时，其生成的视觉 token 数量与输入分辨率为 336×336 的 ViT-L/14 相同，但**在 VLM 基准测试中表现更优**。即便两种架构生成的 token 数量一致，在 TextVQA 和 DocVQA 等文本密集型基准测试中，FastViT 的性能优势仍更为显著。此外，由于采用了高效的卷积层，即便在更高分辨率下保持相同 token 数量，FastViT 的图像编码速度也远快于 ViT-L/14。

#### 3.1.1 多尺度特征

典型的卷积架构和混合架构会将计算过程划分为 4 个不同阶段，阶段之间采用下采样操作。:mdnice-bold-italic[尽管 VLM 主要依赖倒数第二层的特征，但网络早期阶段的特征可提取不同粒度的信息。聚合多尺度信息能够对倒数第二层的高层特征起到补充作用。]

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/04.webp)

$多$尺度特征提取的架构如图 2 所示。**我们对两种从不同阶段池化特征的设计（即平均池化（AvgPooling）和二维深度可分离卷积（2D Depthwise Convolutions））进行了消融实验**。:mdnice-bold-italic[从表 2 可见，采用深度可分离卷积的设计性能更优。]

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/09.webp)

### 3.2 FastViTHD：面向 VLM 的高分辨率编码器

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/10.webp)

尽管经过架构优化的 FastViT 作为图像编码器已表现出色（其规模仅为 ViT-L/14 的 1/8.7），但已有研究[15, 47]表明，扩大图像编码器的规模可提升其泛化能力。

混合架构[17, 92]通常会在 4 阶段设计中扩展自注意力层的数量和宽度，但这种方式存在缺陷。由图 3 可见，如以往工作那样，:mdnice-bold-italic[仅通过增加 FastViT 第 3、4 阶段的自注意力层数量进行扩展，其性能并非最优，且速度慢于 ConvNeXt-L。]

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/04.webp)

:mdnice-bold-italic[为解决这一问题，我们引入了一个额外的下采样阶段，确保自注意力在经过 32 倍下采样的张量上运行（而非 ViTamin 等近期模型采用的 16 倍下采样），如图 2 所示]。我们的设计不仅降低了图像编码延迟，还为计算密集型的 LLM 解码器生成了数量减少 4 倍的 token，从而缩短了首 token 生成时间（TTFT）。该架构示意图如图 2 所示，我们将其命名为 FastViTHD。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/04.webp)

> FastViTHD 的架构包含 5 个阶段（如图 2 所示）：前 3 个阶段采用 RepMixer[82]块，后 2 个阶段采用多头自注意力[24]块。

各阶段的深度（层数）为[2, 12, 24, 4, 2]，各阶段的嵌入维度为[96, 192, 384, 768, 1536]。ConvFFN 层的 MLP 扩展比设为 4.0。该模型包含 1.251 亿参数，**虽比 MobileCLIP 中最大的 FastViT 变体大 3.5 倍，但仍小于主流的 ViT 替代模型。**

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/11.webp)

在将 FastViTHD 用于 FastVLM 训练之前，我们遵循[83]的 CLIP 预训练设置，使用 DataCompDR-1B 数据集对其进行预训练。表 3 显示，尽管 FastViTHD 的规模仅为 ViT-L/14 的 1/2.4、速度是其 6.9 倍，但在 27]中提及的 38 项多模态零样本任务上，其平均性能与 ViT-L/14 相当。与为 VLMs 设计的混合 Transformer 架构 ViTamin[12]相比，FastViTHD 的检索平均性能更优，且规模仅为前者的 1/2.7、速度是其 5.6 倍。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/12.webp)

如表 4 所示，在经过 LLaVA-1.5 训练后，我们将 FastViTHD 与其他经 CLIP 预训练的分层骨干网络（ConvNeXt-L 和 ConvNeXt-XXL）在 VLM 任务上进行了对比。当生成相同数量的视觉 token 时，FastViTHD 的性能与 ConvNeXt-XXL 相当，但规模仅为后者的 1/6.8、速度是其 3.3 倍。例如，输入分辨率为 768×768 的 FastViTHD 生成 144 个 token，其平均 5 项基准性能（Avg-5）为 62.8，与输入分辨率为 512×512、生成 256 个 token 的 ConvNeXt-XXL（Avg-5=63.9）接近，但 FastViTHD 的编码延迟仅为 122.6ms，远低于 ConvNeXt-XXL 的 397.1ms。当 FastViTHD 的输入分辨率提升至 1024×1024 时，其生成 256 个 token，Avg-5 性能达到 63.9，与 ConvNeXt-XXL 持平，但编码延迟仅为后者的 1/1.7。

#### 3.2.1 视觉编码器与语言解码器的相互作用

> VLM 中的准确率-延迟权衡受多个因素影响。

- 一方面，:mdnice-bold-italic[VLM 的整体性能取决于：（1）输入图像分辨率；（2）视觉 token 的数量和质量；（3）LLM 的能力。]
- 另一方面，:mdnice-bold-italic[VLM 的总延迟（首 token 生成时间）由以下两部分决定：（1）视觉编码器的延迟；（2）LLM 的预填充时间——预填充时间受视觉编码器生成的 token 数量和 LLM 规模的共同影响。]

由于 VLM 的优化空间复杂，关于视觉编码器最优性的论断需在不同的（分辨率，LLM）组合下验证。在此，我们**通过实验证明了 FastViTHD 相对于 FastViT 的最优性：对于每种视觉编码器，我们分别搭配 Qwen2[86]系列的三种 LLM（0.5B/1.5B/7B），并测试不同的输入图像分辨率**。对于每一组（分辨率，LLM）组合，我们均采用 LLaVA-1.5[53]的预训练和视觉指令微调流程，并在多个任务上评估模型性能，结果如图 4 所示。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/06.webp)

首先，我们观察到：对于某一视觉编码器，其帕累托最优曲线（图 4 中高亮部分）代表在给定运行预算（TTFT）下可实现的最高性能，该曲线由不同规模的 LLM 组合构成。具体而言，:mdnice-bold-italic[将高分辨率与小型 LLM 搭配并非最优选择——因为小型 LLM 无法有效利用大量 token，且 TTFT 会受视觉编码器延迟的主导（见图 5）。]

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/13.webp)

其次，图 4 中 FastViTHD 的帕累托最优曲线显著优于 FastViT：

- 在给定运行预算下，考虑所有可能的（分辨率，LLM）组合，FastViTHD 的性能显著更优（在平均 5 项指标上提升超过 2.5 个百分点）；
- 同样，FastViTHD 达到目标 VLM 性能的速度最高可提升 3 倍。需注意的是，在前面的章节中我们已证明，基于 FastViT 的 VLM 已比基于 ViT 的 VLM 有显著提升，而 FastViTHD 在此基础上仍实现了性能的大幅改进。

#### 3.2.2 静态与动态输入分辨率

> 扩展输入分辨率主要有两种方式：直接调整模型的输入分辨率，或对图像进行分块（tiling）并将编码器分辨率设为块大小。

以往工作[52, 54]提出的分块推理（AnyRes）可使 ViT 模型处理高分辨率图像。由于 FastViTHD 设计用于高效处理高输入分辨率的推理任务，**我们分析了两种策略在不同分辨率下的最优工作点。**

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/14.webp)

由图 6 可见，:mdnice-bold-italic[将模型输入分辨率直接设为目标分辨率可实现最佳的准确率-延迟权衡]；仅在 1536×1536 等极高分辨率下（受内存带宽限制），动态分辨率（分块）才具有优势。:mdnice-bold-italic[若需使用动态分辨率，采用分块数量更少的设置可获得更优的准确率-延迟权衡]。

#### 3.2.3 与 token 剪枝和下采样方法的对比

我们进一步将不同分辨率下 FastViTHD 的性能与主流的 token 剪枝方法进行了对比。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/15.webp)

> 由表 5 可见，:mdnice-bold-italic[与在 ViT 等各向同性架构上使用 token 剪枝方法相比，采用分层骨干网络的 VLMs 可实现更优的准确率-延迟权衡]。

通过在较低输入分辨率下训练 VLM，FastViTHD 可将视觉 token 数量降至 16 个，且性能优于近期的 token 剪枝方法。有趣的是，即便像[7, 32, 33, 87]中提出的最有效的 token 剪枝方法，其性能也不及在 256×256 低输入分辨率下训练的 FastViTHD。

## 四、实验

### 训练设置

对于第 3 节中呈现的所有消融实验，除非另有说明，否则我们均遵循 LLaVA-1.5 [53]中描述的两阶段训练设置，并使用 Vicuna-7B [98]作为 LLM 解码器。

- **在第一阶段，仅训练投影层（projector）**：使用 LLaVA-558K 对齐数据集训练 1 个 epoch，批大小（batch size）设为 256，学习率（learning rate）设为(10^{-3})。此阶段的输入图像分辨率与骨干网络预训练分辨率保持一致（例如，FastViT 为 256×256，FastViTHD 为 224×224）。
- **在第二阶段，我们使用 LLaVA-665K 监督微调数据集，训练 1 个 epoch 并对所有模块（即视觉编码器、投影层和 LLM）进行调优**。:mdnice-bold-italic[此阶段的输入图像分辨率设置为目标分辨率。]

> 在第 4 节中，我们展示了使用不同 LLM 解码器的实验结果，主要包括 Qwen2-0.5B/1.5B/7B 模型家族[86]（对话版本）和 Vicuna-7B 模型[98]。

我们在两种训练设置下报告结果：

::: block-1
- 第一种是 LLaVA-1.5 中引入的两阶段训练设置；
- 第二种则遵循文献[43, 66]中的当前趋势，采用三阶段训练 VLM，即阶段 1（Stage 1）训练连接器（connector）、阶段 1.5（Stage 1.5）进行分辨率缩放、阶段 2（Stage 2）进行视觉指令微调。
:::

在此设置中，阶段 1 的输入图像分辨率设为骨干网络预训练分辨率，后续两个阶段则调整为目标分辨率。在两种设置中，仅在阶段 1 冻结视觉编码器和 LLM，其余阶段均对所有模块进行微调。

:mdnice-bold-italic[为实现最佳设置，我们进一步使用文献[30]中包含思维链推理（chain-of-thought reasoning）的高质量指令微调数据集，对阶段 2 训练后的模型进行额外微调，并将此过程称为阶段 3（Stage 3）]。我们公开发布了阶段 2 训练得到的 R4、R12 和 R41 checkpoint，以及阶段 3 训练得到的 R5、R13 和 R42 checkpoint，相关资源已纳入我们的开源代码库。

> 本文中报告的所有 FastVLM 模型均在单个节点上训练，该节点配备 8 块 NVIDIA H100-80GB GPU。

- VLM 的阶段 1 训练速度较快，使用 Qwen2-7B 解码器时约需 30 分钟。
- 阶段 1.5 和阶段 2 的训练时长取决于输入分辨率：对于 1024×1024 的输入分辨率，阶段 1.5 需 77 小时，阶段 2 需 8 小时。

所报告的实际时长对应以下阶段使用的数据集规模：阶段 1.5 使用 1500 万样本，阶段 2 使用 110 万样本。

### 评估

**我们在主流基准测试集上对模型进行评估**，包括 GQA [34]、ScienceQA [59]、TextVQA [74]、POPE [48]、LLaVA-in-the-wild [54]、VQAv2 [29]、MMVet [91]、MMMU [93]、DocVQA [65]和 SeedBench [42]。

- 其中，GQA、ScienceQA、TextVQA、POPE 和 LLaVA-in-the-wild 基准测试集采用 LLaVA [54]的官方评估方法；
- 其余评估则使用 lmms-eval [96]库 v0.2.2 版本。

所有评估均采用默认设置，对于依赖 GPT 作为评估器（judge）的任务，lmms-eval 默认使用 GPT-0613 版本。

对于第 3 节中呈现的消融实验，我们报告模型在 GQA、TextVQA、POPE、DocVQA 和 SeedBench 上的性能：GQA 和 SeedBench 为通用知识基准测试集，DocVQA 和 TextVQA 为文本密集型评估任务，POPE 为幻觉（hallucination）基准测试集。**这些基准测试集兼具多样性，且便于消融实验快速评估；最重要的是，它们在不同初始化条件和概率解码设置下的方差较低**。

我们报告了不同初始化条件下所有评估任务的方差：所选 5 个指标的标准差均小于 0.5。我们将这 5 个基准测试集的平均值称为“Avg-5”，并将其作为分析中的可靠指标。Avg-5 的经验标准差估计值为 0.1。

### 基准测试

所有模型均在**搭载 M1 Max 芯片和 32GB 内存的 MacBook Pro 上进行基准测试**。使用 coremltools v7.2 工具将视觉编码器转换为 Core ML 包文件，并在 XCode 15.4（版本号 15F31d）的神经引擎（neural engine）上进行基准测试。LLM 则在 MacBook Pro 的 GPU 上使用 MLX [31]进行基准测试：

::: block-1
- 首先通过 mlx-lm.convert 工具将模型从 Hugging Face 平台转换为 MLX 格式，并将张量转换为 FP16 精度；
- 然后使用 mlx-lm.cache-prompt 工具[31]估计预填充延迟。
:::

首次令牌生成时间（Time-To-First-Token, TTFT）的计算方式为：特定分辨率下的图像编码延迟与对应视觉令牌的 LLM 预填充延迟之和。

### 4.1 与现有最优方法的对比（Comparison with state-of-the-art）

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/07.webp)

在表 6 中，我们将 FastVLM 与近期发表的方法进行对比。不同研究的训练设置差异较大，因此我们对每个方法均报告其 LLM 解码器、预训练数据集和指令微调数据集的规模，以确保对比的公平性。

#### 分层骨干网络（Hierarchical Backbones）

将 FastVLM（R20）与 ConvLLaVA [28]（R18）对比（两者使用相同 LLM 且训练数据规模相近）：我们的模型在 TextVQA 上性能提升 8.4%，在 DocVQA 上性能提升 12.5%，同时速度快 22%。

在**更高分辨率下，性能差距进一步扩大**：FastVLM（R28、R29）在多种基准测试集上均实现更优性能，且在使用相同 LLM 解码器的情况下，速度是 ConvLLaVA（R26）的 2 倍。

#### 数据集扩展

通过加入包含 1500 万样本、用于分辨率缩放的中间预训练阶段来扩展预训练数据集后，FastVLM（R21）在多种基准测试集上均达到或超过 MM1 [66]（R38）的性能。**值得注意的是，FastVLM 生成的视觉令牌数量仅为 MM1 的 1/5，却能实现这一性能**。

在输入分辨率为 1024×1024 且使用规模为 1250 万的更大指令微调数据集时，FastVLM（R41）在包括 TextVQA、DocVQA 等对输入分辨率和视觉令牌数量敏感的文本密集型评估任务在内的多种基准测试集上，均优于 MM1（R38）和 LLaVA-NeXT（R39）。若使用 AnyRes 进一步提升输入分辨率，性能差距会更大。

#### 多视觉编码器

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/07.webp)

近期，MiniGemini [49]和 Cambrian-1 [78]提出了采用多视觉编码器的模型。在表 6 中，我们将使用单个视觉编码器的 FastVLM（R40）与采用多编码器且视觉指令微调数据规模相近的方法进行对比。在 Cambrian-1 [78]（R44）中，视觉编码对总首次令牌生成时间（约 5 秒）的贡献是 LLM 预填充的 3.2 倍（详细拆分见附表 10）。

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/16.webp)

FastVLM（R40）在使用相近规模视觉指令微调数据集的情况下，性能优于 Cambrian-1（R44），且速度是其 7.9 倍。将指令微调数据集规模扩展至 1250 万后，FastVLM（R41）即使在对视觉令牌数量敏感的文本密集型评估任务上（见附表 11），也能以仅为 Cambrian-1（R44）2.3 倍的视觉令牌数量实现更优性能。

#### 解码器的影响

如先前研究[44]所示，**VLM 性能还依赖于 LLM 的质量**。

- 将 LLM 从 Vicuna-7B（R21、R29）替换为 Qwen2 [77, 86]模型（R22、R30）后，模型在所有基准测试集上的性能均有提升，尤其在 MMVet、LLaVA-in-the-wild 和 MMMU 基准测试集上的提升更为显著。
- :mdnice-bold-italic[当使用 Qwen2-0.5B 作为 LLM 解码器时，FastVLM（R4）的性能优于 LLaVA-OneVision [45]（R2），且速度是其 85 倍。]

> 这一结果凸显了我们所提视觉编码器的优势：两款模型使用相同 LLM 解码器，而 FastViTHD 的规模仅为 SigLIP-SO400M [94]的 1/3.4。

#### 同期研究

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/17.webp)

在表 7 中，我们**将 FastVLM 与 SmolVLM2 [62]、FlorenceVL [11]等同期研究进行对比。**

::: block-1
- FastVLM（R7）在 ChartQA、TextVQA 等文本密集型评估任务上与 SmolVLM2（R3）表现相当，且在 DocVQA 和 InfoVQA 基准测试集上性能更优，而其使用的视觉令牌数量仅为 SmolVLM2（R3）的 1/8.2。
- 此外，FastVLM（R7）在 TextVQA、DocVQA 等基准测试集上的性能优于 FlorenceVL（R6），同时视觉令牌数量仅为后者的 1/2.3，视觉编码器规模仅为后者的 1/6.2。
- 在 AI2D、ScienceQA 等知识类评估任务上，FastVLM（R7）的性能优于所有对比模型。
- 在最小规模模型对比中，FastVLM（R2）在多种基准测试集上的性能均优于规模相近的 SmolVLM2 模型（R1），且视觉令牌数量仅为后者的 1/4.3。
:::

## 五、结论

> 在本研究中，我们提出了 **FastVLM——该模型利用 FastViTHD 视觉骨干网络实现对高分辨率输入的高效编码**。:mdnice-bold-italic[FastViTHD 采用混合架构，在增强型图文数据上进行预训练]，能够在仅损失极小精度的前提下大幅减少视觉令牌的输出数量。

FastVLM 在多种 VLM 基准测试集上均实现了与现有方法相当的性能，同时在首次令牌生成时间（TTFT）和视觉编码器参数数量两方面均提升了效率。在 M1 MacBook Pro 上进行的严格基准测试表明，与现有方法相比，FastVLM 实现了**最先进的分辨率-延迟-准确率权衡性能。**

## 参考文献

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/18.webp)

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/19.webp)

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/20.webp)

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/21.webp)

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/22.webp)

![文章插图](/articles/2025-09-16-85-ttft-ai-fastvlm-fastvithd-bcfdd2b7/images/23.webp)

交个朋友
