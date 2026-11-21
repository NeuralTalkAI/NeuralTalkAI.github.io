关键词：**MXFP8**、NVFP4、**强化学习（RL）**、:mdnice-bold-italic[低精度量化]、Blackwell

> *本文介绍了 LMSYS 团队**在 Miles 框架中落地的两套 Blackwell 原生低精度 RL 方案**：端到端 MXFP8 与逐 token NVFP4，:mdnice-bold-italic[目标是在不牺牲训练稳定性的前提下吃满 Blackwell Tensor Core 的 FP8/FP4 算力。]*

**低精度 RL 的核心难点在于"精度契约"——rollout、训练、checkpoint 转换与实时权重更新必须使用完全一致的量化方式**，否则采样策略与学习策略会逐渐分歧。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/01.webp)

- **Towards Blackwell-Native 8-bit and 4-bit RL: End-to-End MXFP8 and NVFP4 RL in Miles**
- https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl

::: related-title
**相关推荐**
:::

::: related-links
- **[Blackwell 架构下的高速扩散模型：基于 Diffusers 与 TorchAO 的 MXFP8 及 NVFP4 应用](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899996&idx=1&sn=c9541fb4324c46b5e2708ea19ca2bbf7&scene=21#wechat_redirect)**

:::

::: related-links
- **[MXFP4近无损LLM训练：MXFP4 GEMM结合RHT与SR，比FP8快2倍计算且6.7B GPT模型近无损](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891301&idx=1&sn=07fa64e075a9cf2fa5d6cced89b8b108&scene=21#wechat_redirect)**
- **[突破NVFP4量化性能瓶颈！MIT与NVIDIA提出Four Over Six开源方案：近BF16困惑度与&lt;2%推理开销兼得！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895052&idx=1&sn=7bd638a89f18e3a5c66c5bf67cc22aeb&scene=21#wechat_redirect)**
- **[面向 LLM 的 MXFP4 专属 PTQ 方案：BRQ 策略实现跨模型量化精度提升与 40% 推理提速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893841&idx=1&sn=84e6bf385dc318254943c9d16cc4a861&scene=21#wechat_redirect)**

:::

> 为此团队**横跨 SGLang、TransformerEngine、FlashInfer、Megatron 和 cuDNN frontend 五个项目做了全栈改造**：

::: related-links
- MXFP8 方案让 rollout、前向、权重梯度和数据梯度 GEMM 全部运行在 MXFP8 上；
- NVFP4 方案更激进，但只量化占模型体量约 97.8% 的 MoE 专家，并首创在线逐 token FP32 激活缩放，把量化误差局部化到单个 token，消除了静态标定的 batch 依赖——这一点对序列长度不断变化的 RL 尤为关键。

:::

反向传播提供高精度与反量化两种 BF16 模式，避开了预训练方案中的随机舍入与 Hadamard 变换。此外，团队:mdnice-bold-italic[建立了 FlashInfer 与 TransformerEngine 之间的位精确量化器契约，并通过按层、按名称的细粒度控制让末尾 15% 层与 MLA 投影等敏感张量保持 BF16。]

在 8x B200 的 Qwen3-30B-A3B 消融实验中，**五种低精度配置的奖励曲线与 BF16 高度重合，rollout 耗时明显下降**。:mdnice-bold-italic[同一套方案还支持在线 NVFP4 serving，SGLang 中仅需加一个参数即可启用，并已在 humans& 的大规模异步 RL 生产中投入使用。🐂🐸]

::: block-1
本文所述的方案设计与绝大部分实现由 humans&[1] 的 Ziang Li 完成。感谢以下合作者提供的工程支持、集成协助与评审：

- SGLang 团队与 Miles 团队。
- NVIDIA DevTech Compute 团队（Siyuan Fu、Yigong Qin、Zhongbo Zhu）、TransformerEngine 团队与 FlashInfer 团队。

同样感谢 Cursor 团队提出逐 token NVFP4 激活缩放的思路。
:::

## 本文目录

- 一、引言
- 二、为什么是 Blackwell 原生方案？
- 三、格式背景
  - 3.1 MXFP8
  - 3.2 NVFP4
- 四、方案一：端到端 MXFP8 RL
  - 4.1 训练
  - 4.2 Rollout
- 五、方案二：逐 token NVFP4 RL
  - 5.1 为什么不直接使用 NVFP4 预训练方案？
  - 5.2 逐 token 激活缩放
  - 5.3 高精度反向与反量化反向
  - 5.4 反向开销与显存
- 六、bit级精确的量化器契约
- 七、细粒度精度控制
  - 7.1 层精度选择
  - 7.2 案例研究：MXFP8 MLA
- 八、实验结果：8x B200 上的 Qwen3-30B-A3B 方案消融
  - 8.1 训练-推理不匹配
  - 8.2 奖励
  - 8.3 性能
- 九、在线 NVFP4 模型 serving
- 十、未来工作
  - 10.1 去掉额外的 BF16 权重副本
  - 10.2 偶发梯度尖峰
  - 10.3 重构权重更新接口
- 十一、在 Miles 中试用 NVFP4 方案

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/02.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、引言

> 在低精度 RL 中，rollout、训练、checkpoint 转换和实时权重更新必须遵守同一份精度契约，否则采样器与训练器的策略会发生分歧。

Miles 与 SGLang RL 生态已经纳入了多种低精度方案：

- 《Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL》[2]表明，**在训练与采样中同时使用 FP8，相比"BF16 训练 + FP8 rollout"能降低不匹配；**
- 《Squeezing 1TB Model Rollout into a Single H200: INT4 QAT RL End-to-End Practice》[3]则表明，**训练中使用伪量化（fake quantization）配合 W4A16 rollout 可以让 INT4 落地实用。**

::: block-1
:mdnice-highlight[我们将这条路线扩展到 Blackwell 原生格式：在 Miles 中实现了 MXFP8 与 NVFP4 方案，并向 SGLang、TransformerEngine、FlashInfer、Megatron 和 cuDNN frontend 贡献了支撑组件]。公开路线图见 miles#615：[Roadmap] Blackwell MXFP8 and NVFP4 RL training[4]。
:::

主要贡献：

1. **端到端 MXFP8 RL**：rollout、前向传播、权重梯度 GEMM 和数据梯度 GEMM 全部使用 MXFP8。
2. **面向 MoE 专家权重的逐 token NVFP4 RL**：采用在线逐 token 激活缩放。
3. **细粒度精度控制**：可以让选定的张量（如末尾若干层）一致地保持 BF16。
4. **TransformerEngine 与 FlashInfer 之间的位精确（bit-exact）量化器契约**：保证权重更新不会引入本可避免的训练-推理不匹配。

## 二、为什么是 Blackwell 原生方案？

> 此前的低精度方案并非围绕 MXFP8 或 NVFP4 设计。Miles 现有路径遵循 DeepSeek-V3[5] 风格的块缩放 FP8 方案：**权重使用 128x128 块缩放，激活使用 1x128 tile 缩放，scale 针对每个 tile 或块在线计算**。这是 Hopper 时代的强力方案，:mdnice-bold-italic[但在 Blackwell 上，其 FP32 scale 仍然是在 Tensor Core 路径之外通过软件施加的，而非经由原生微缩放（microscaling）硬件。]

INT4 QAT 解决的是另一个问题。训练用伪量化让模型适应 INT4 权重，rollout 则用 W4A16。**虽然省显存，但计算路径实际上仍是"BF16 激活 + 反量化的 INT4 权重"。** 下表将 NVIDIA HGX 平台[6]的稠密 Tensor Core 规格归一化为单 GPU 吞吐量：B200 与 B300 取自 8-GPU HGX 系统，Rubin 取自 HGX Rubin NVL8 表格。

| GPU | BF16 稠密 Tensor Core | FP8 稠密 Tensor Core | FP4 稠密 Tensor Core |
| --- | --- | --- | --- |
| B200 | 2.25 PFLOPS | 4.5 PFLOPS | 9 PFLOPS |
| B300 | 2.25 PFLOPS | 4.5 PFLOPS | 13.5 PFLOPS |
| Rubin GPU (NVL8) | 4 PFLOPS | 17.5 PFLOPS | 35 PFLOPS |

对 RL 系统而言，精度契约横跨：

- SGLang rollout。
- Megatron 与 TransformerEngine 训练。
- Hugging Face checkpoint 转换。
- Megatron 到 Hugging Face 的实时权重导出。
- 细粒度的高精度例外。

## 三、格式背景

### 3.1 MXFP8

> MXFP8 是一种微缩放 FP8 格式。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/03.webp)

::: image-caption
MXFP8对每连续32个元素使用一个E8M0缩放因子，实现细粒度量化以及紧凑的缩放因子表达。
:::

TransformerEngine 的 MXFP8 文档[7]将其描述为一种 Blackwell 原生的块级缩放方案：每 32 个连续的 E4M3 值共享一个局部 E8M0 scale，块是一维的。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/04.webp)

::: image-caption
MXFP8按行与按列量化布局。
:::

由于 E8M0 scale 表示 2 的幂，解码后的 scale 通常向上取整，以保证块内最大值不被截断。

### 3.2 NVFP4

> NVFP4 是 Blackwell 的原生 FP4 格式。如 NVIDIA 的 Introducing NVFP4 for Efficient and Accurate Low-Precision Inference[8]所述，它以 FP4 E2M1 存储数值，每 16 个值组成的块共享一个 FP8 E4M3 scale。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/05.webp)

::: image-caption
Blackwell 架构支持的三种主要 4 位浮点格式 FP4 (E2M1)、MXFP4 与 NVFP4 开展多维度对比，罗列了格式结构、硬件加速缩放、内存占用以及精度表现等关键特性。FP4 依靠软件缩放因子，MXFP4 每 32 个数值使用基于 2 的幂次的共享缩放，而 NVFP4 为每 16 个数值的微块配置 FP8 缩放，具备硬件加速缩放能力。相较另外两种格式，NVFP4 在大模型场景下精度损失风险更低，在保留硬件加速优势的同时，实现内存压缩与推理精度之间更优的权衡。
:::

由于 E4M3 比 UE8M0 分辨率更细，其 scale 通常舍入到最近可表示值。标准的 NVFP4 方案还会为更大的张量范围再加一个 FP32 scale，构成两级层级：

- 一个粗粒度 FP32 scale，把张量或 token 映射进 NVFP4 可表示范围。
- 一个细粒度 E4M3 scale，适配每个 1x16 块。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/06.gif)

::: image-caption
NVIDIA 的 NVFP4（E2M1）4 位浮点格式及其双级缩放机制，该格式包含 1 位符号、2 位指数、1 位尾数。数据划分为每 16 个值为一个数据块，采用两级缩放：块级缩放中，每个 16 值数据块配备一个 FP8 (E4M3) 缩放因子，实现块内数值校正；在此基础上，整个张量再使用 FP32 (E8M23) 缩放因子做全局张量缩放。这套分层缩放方案，用 4 比特存储权重，同时借助块、张量两级缩放补偿精度损失，在降低显存占用、提升算力的同时，尽量减少 AI 量化带来的精度下降，适配大模型训练与推理场景。
:::

FP32 scale 可以选取不同的张量范围。这个选择属于方案层面的决策，而非格式本身的属性——它对 RL 尤其重要。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/07.gif)

::: image-caption
该图直观对比了MXFP4与NVFP4的分块缩放结构。MXFP4采用大小为32个元素的数据块，块内共用一个粗粒度的2的幂次缩放系数；NVFP4则将块规模缩减至16个数值，每个微块配备独立的FP8缩放因子。图中展示重建公式$x = x_q \times s$，4位编码值$x_q$通过块内缩放因子$s$还原原始数值。更小的微块粒度让NVFP4能够更好适配张量内部局部动态范围，减少因统一缩放带来的量化偏差，更有利于保留权重与激活中细微但关键的数值差异。
:::

## 四、方案一：端到端 MXFP8 RL

> MXFP8 方案是此前端到端 FP8 工作最直接的 Blackwell 原生扩展。Rollout、前向传播、权重梯度 GEMM 和数据梯度 GEMM 全部使用 MXFP8，而选定的张量则通过下文介绍的精度控制规则保持 BF16。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/08.webp)

::: image-caption
端到端 MXFP8 RL 方案。BF16 激活与权重分别按行方向（row-wise，1x32 块）和列方向（col-wise，1x32 块）量化为 MXFP8；Fprop 使用 MXFP8 Tensor Core（FP32 累加）；Wgrad 与 Dgrad 使用 BF16 Tensor Core（FP32 累加）；优化器状态与主权重保持 FP32；MXFP8 权重直接送往 rollout。
:::

### 4.1 训练

> TransformerEngine 与 Megatron 将 MXFP8 实现为一条性能优化的一等公民 Blackwell 训练路径，包括 deepseek-v3-gb200-optimization.md[9] 中描述的 GB200 DeepSeek-V3 优化。在我们的 Miles 集成中，这条路径是端到端 MXFP8 RL 的训练侧基础。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/09.webp)

::: image-caption
DeepSeek‑V3 在 GB200 NVL72 硬件上基于 MXFP8 精度的全链路性能迭代过程，以 494.46 TFLOPS/GPU 作为优化基线，依次记录了内核融合、DeepEP、部分 CUDA Graph、CPU 绑定调优、HybridEP 通信库、Norm‑MXFP8 量化融合等关键改造。每一行对应一组调度器与功能开关组合，直观呈现各优化手段带来的算力增益，最终性能达到 970.01 TFLOPS/GPU，相较基线实现 1.96 倍提升，也体现出内核、内存、通信、CPU 侧协同调优对 MoE 大模型 FP8 预训练的重要价值。
:::

与 DeepSeek-V3 FP8 方案[10]的一个区别在于反向激活的表示方式。DeepSeek-V3 以 1x128 FP8 tile 存储前向激活，并在反向 GEMM 之前将其转换到反向所需方向。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/10.webp)

::: image-caption
DeepSeek‑V3 FP8 混合精度线性算子完整数据流框图，覆盖前向传播 Fprop、激活反向 Dgrad、权重梯度 Wgrad 三类 GEMM 计算。前向输入 BF16 转为 FP8 送入 Fprop，累加使用 FP32，输出转回 BF16；反向阶段，输出梯度经 FP8 转换后进入 Dgrad，权重参与计算，Dgrad 内部累加保持 FP32，输出 BF16 输入梯度回传上层；Wgrad 复用前向 FP8 缓存激活，FP32 累加生成 FP32 权重梯度，结合主权重与优化器状态完成权重更新，整套链路借助 FP8 压缩激活，有效降低训练显存开销。
:::

**这种做法 FP8 存储量更少，但在反向 GEMM 之前引入了一次反量化加再量化**。TransformerEngine 的 MXFP8 文档指出，行方向 1x32 块与列方向 32x1 块在数值上并不相同，必须各自独立地从全精度数据量化而来。因此 TransformerEngine 在量化时会同时物化行方向和列方向两份量化副本。:mdnice-bold-italic[这占用更多显存，但避免了额外的一次再量化步骤，减少了反向路径上的附加量化误差。]

这是 RL 中典型的系统权衡。我们采用 TransformerEngine 路径，以维持单一的端到端 MXFP8 契约，不引入新的不匹配来源。

### 4.2 Rollout

> 在 rollout 侧，SGLang 使用来自 FlashInfer 和 Triton 的 Blackwell MXFP8 内核。

我们实现并提交到上游的 FlashInfer 与 SGLang 中的 rollout 路径的 PR 有：

flashinfer#2581：Implement `cutlass_fused_moe` mxfp8[11]

sglang#17449：Add mxfp8 support for online quantization, Triton dense linear, and CUTLASS MoE[12]

sglang#19537：[FlashInfer v0.6.4] [RL] Integrate FlashInfer mxfp8 gemm, MoE, and routed MoE[13]

sglang#21576：[FlashInver v0.6.7] Integrate flashinfer_trtllm mxfp8 gemm[14]

sglang#28459：[RL] Fix FlashInfer TRTLLM MXFP8 dense weight layout[15]

**几乎所有主要 GEMM 都可以量化为 MXFP8，包括注意力投影和 MoE 专家**。主要例外是被显式控制的高精度层，例如下文所述的 BF16 MLA 投影。

## 五、方案二：逐 token NVFP4 RL

> NVFP4 比 MXFP8 更激进，因此我们选择性地用它。我们量化 MoE 专家，因为它们主导了模型体量与 rollout 显存流量；模型其余部分除非显式配置，否则保持 BF16。

例如，DeepSeek-V3[16] 总参数量约为 671B，其 MoE 专家占：

```
(61 - 3) * (256 + 1) * 3 * 7168 * 2048 / 1e9 = 656.5B 参数
```

**即约占模型的 97.8%。因此，瞄准 MoE 专家可以捕获绝大部分显存收益**，而不必强迫每一层都进入最激进的精度格式。

### 5.1 为什么不直接使用 NVFP4 预训练方案？

> **原始的 Pretraining Large Language Models with NVFP4(NVIDIA, 2509.25149v2)[17]方案面向大规模预训练设计：目标是在海量 token 上保持粗粒度优化方向，同时使用 FP4 GEMM**。它将 FP4 线性层 GEMM 与若干稳定器结合：选定层保持更高精度、权重缩放在前向与反向间保持一致、训练路径使用随机舍入（SR）和随机 Hadamard 变换（RHT）。论文中，SR 施加于梯度以减少量化偏差、产生无偏量化梯度；RHT 用于打散大数值的块级离群点，尤其是权重梯度 GEMM 的输入。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/11.webp)

::: image-caption
原始 NVFP4 预训练方案。激活量化为 NVFP4 进入 Fprop；DGRAD/WGRAD 使用 NVFP4 GEMM；反向传播中，BF16 激活梯度经带随机舍入 SR 的量化、转置与哈达玛变换，再量化，分别执行 WGRAD 权重梯度、DGRAD 输入梯度的 NVFP4 矩阵运算。权重梯度回传给优化器更新 FP32 主权重，输入梯度转换回 BF16 传递至上一层，以 4 比特计算降低开销，依靠 FP32 主权重、随机舍入保障训练精度。
:::

> 这是一个很好的起点，但 RL 的失效面不同：

| 场景 | 数值状况 | 核心结论 |
| --- | --- | --- |
| 预训练 | 梯度信号稳定，权重更新幅度大，模型权重自适应性强，激活与梯度对精度敏感且动态范围高。 | 保持粗粒度优化方向与收敛性。 |
| RL | 梯度噪声大，奖励方差高，有效更新小而脆弱。 | 让量化噪声低于真实更新信号；否则量化噪声会覆写脆弱的能力，导致性能崩塌。 |

NVFP4 RL 方案并未照搬预训练方案的全部组件。我们的目标是：MoE 专家权重量化、逐 token 激活缩放、一致的精度控制，以及可选原始操作数或反量化操作数的 BF16 反向 GEMM。

### 5.2 逐 token 激活缩放

> NVFP4 的两级层级很强大，但 FP32 激活 scale 的作用范围必须谨慎选择。正如 Cursor Composer 2 技术报告[18]所讨论的，:mdnice-bold-italic[逐张量 NVFP4 缩放会让训练依赖 batch 构成，而 token 间共享 scale 会把未来 token 的信息泄漏进过去 token 的表示中。如果一个 token 与其他 token 共享 scale，它的量化表示就取决于 batch 的组成。这对 RL 尤其成问题，因为 rollout 调度和序列长度都在变化。]

因此我们的方案在线地为每个 token 计算一个 FP32 激活 scale。这把激活离群点局部化到单个 token，消除了静态激活标定（calibration）带来的伪影，并让 SGLang rollout 与 Megatron 训练使用相同的激活 scale 作用范围。

在 rollout 侧，逐 token FP32 scale 的计算被融合进 FlashInfer 的激活量化内核路径：同一次调用在输出打包 FP4 激活和 E4M3 块 scale 的同时，也返回逐 token FP32 scale。因此，逐 token 激活缩放不需要单独的标定 scale 计算 pass。

**训练-推理一致性还要求并行方式匹配**。如果 FP32 scale 是在专家张量并行（ETP）分区内逐 token 计算的，SGLang 与 Megatron 应使用相同的 ETP 尺寸，否则双方看到的张量分区不同，计算出的 scale 也会不同。

**SwiGLU MoE 层还带来另一项关键契约**。SGLang 与 Megatron 通常把 gate 和 up 投影融合为一个 GEMM，因此即使 Hugging Face checkpoint 把它们分开存储，两个张量在转换和实时权重更新时也必须共享同一个 FP32 scale。:mdnice-bold-italic[Miles 通过在 NVFP4 导出路径中将 gate/up 成对量化来强制这一点。]

> 我们在整个技术栈上实现逐 token NVFP4 方案，并提交到了上游：

TransformerEngine 训练方案：TransformerEngine#2931：Implement row-scaled NVFP4 fprop recipe[19]

cuDNN frontend 训练内核：cudnn-frontend#251：Add row-scale support to grouped GEMM quant[20]

FlashInfer rollout 内核：flashinfer#3027：[feat] Trtllm-gen Per-token Nvfp4 MoE[21]

SGLang 集成与权重更新修复：sglang#22918：[FlashInfer v0.6.11] [RL] Support FlashInfer per-token NVFP4 MoE[22]、sglang#22204：[RL] Refactor NVFP4 shuffling/swizzling to in-place replacement[23]

### 5.3 高精度反向与反量化反向

> 在高精度反向的 NVFP4 变体中，前向传播与 rollout 对 MoE 专家使用 NVFP4，而反向 GEMM 使用原始的 BF16 操作数。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/12.webp)

::: image-caption
高精度反向的 NVFP4 方案。前向使用 NVFP4 激活（行缩放，1x16 块）与 NVFP4 权重（张量级缩放，1x16 块）；Wgrad 与 Dgrad 均保持 BF16 操作数、BF16 Tensor Core 与 FP32 累加；优化器状态与主权重为 FP32。
:::

反量化反向是第二种可选模式。反向 GEMM 仍以 BF16 运行，但消费的是"前向实际产生的低精度操作数反量化回 BF16"的值，而非原始 BF16 值。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/13.webp)

::: image-caption
NVFP4 RL 带反量化反向传播的训练流程图，BF16 激活与权重分别量化为 1×16 分块、行缩放的 NVFP4 激活和张量缩放的 NVFP4 权重，送入 Fprop 模块，通过 NVFP4 张量核完成前向运算。NVFP4 激活、权重会被反量化回 BF16，分别输入 Wgrad 与 Dgrad，二者均采用 BF16 张量核与 FP32 累加器做梯度计算。Wgrad 输出传入保存 FP32 主权重的优化器状态模块，更新后回写 BF16 权重，BF16 激活送往下一层，BF16 梯度传递至上一层，反向全程使用反量化后的 BF16 参与运算。
:::

两种模式都避免了低精度反向 GEMM，因此这些配置不使用原始 NVFP4 预训练方案中的 RHT 或随机舍入。它们用反向吞吐换取更高精度的计算——但 RL 往往受 rollout 限制（rollout-bound），长上下文注意力与通信又进一步摊薄了端到端影响。

同样的反向模式选择也适用于 MXFP8：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/14.webp)

::: image-caption
MXFP8 混合精度训练的数据流，BF16 激活、BF16 权重分别量化为行方向 1×32 分块的 MXFP8 激活与 MXFP8 权重，输入 Fprop 模块，借助 MXFP8 张量核、FP32 累加器完成前向计算。反向传播中 Wgrad 权重梯度、Dgrad 输入梯度均直接保留 BF16 精度，依靠 BF16 张量核、FP32 累加器运算。Wgrad 输出送入保存 FP32 主权重的优化器状态模块，更新后回写 BF16 权重，BF16 激活送往下一层，BF16 梯度传递至上一层。
:::

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/15.webp)

::: image-caption
MXFP8 带反量化反向传播的 RL 训练流程图，BF16 激活与权重分别量化为行向 1×32 分块的 MXFP8 激活与 MXFP8 权重，送入 Fprop，由 MXFP8 张量核配合 FP32 累加器执行前向计算。MXFP8 激活、权重分别反量化恢复为 BF16，再输入 Wgrad 和 Dgrad，二者均使用 BF16 张量核与 FP32 累加器运算。Wgrad 输出流向存储 FP32 主权重的优化器状态模块，更新后回写 BF16 权重，BF16 激活送往下一层，BF16 梯度传递至上一层。
:::

我们实现并给上游提交了 `NVTE_BACKWARD_OVERRIDE`，作为 TransformerEngine 中用于选择高精度或反量化反向操作数的可复用接口（TransformerEngine#2644：Add `NVTE_BACKWARD_OVERRIDE=high_precision|dequantized`[24]），随后通过 Miles 方案配置暴露了两种模式。姊妹篇 humans&[25] 文章覆盖了算法动机与更多反向模式消融。

### 5.4 反向开销与显存

> 反量化操作会在训练过程中额外引入一步反量化计算。我们与 NVIDIA 合作，在 TransformerEngine #2865：MXFP8/NVFP4 反量化内核优化[26] 中显著降低了该操作的计算开销。

高精度反向与反量化反向相比 Transformer Engine 默认的低精度反向路径，可进一步降低峰值显存占用。两种模式均无需生成并保存前文“MXFP8 训练”小节中所述、专用于低精度反向 GEMM 的第二份列方向量化副本。

以下显存数据是我们在 TransformerEngine#2644[27] 中验证反向模式实现时测得的。`alloc` 列报告已分配显存，`resrv` 列报告预留显存，单位均为 MB。

MXFP8 线性层显存，`dtype=torch.bfloat16`，`input_shape=(2048, 2048)`，`out_features=8192`：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/16.webp)

NVFP4 线性层显存，`dtype=torch.bfloat16`，`input_shape=(2048, 2048)`，`out_features=8192`：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/17.webp)

## 六、bit级精确的量化器契约

> 在 RL 中，量化不匹配会随权重更新不断累积。:mdnice-bold-italic[如果训练侧与 rollout 侧对同一 Tensor 的量化方式不同，采样与学习所用的策略就不再是同一个低精度模型。因此我们需要 RL 技术栈两侧之间的显式契约。]

我们将 FlashInfer 与 TransformerEngine 的量化器对齐到同一套 MXFP8 与 NVFP4 位级契约。我们的 FlashInfer 单元测试针对 TransformerEngine 风格的参考实现检查字节级完全一致，覆盖随机数据、量化边界数据、全零张量和最大值张量，对应 `init_data = ["random", "boundary", "zeros", "maxes"]`。该量化器对齐已在 flashinfer#3387：Make cute dsl mxfp8/nvfp4 quantizer bitwise exact[28] 中实现并上游。

:mdnice-bold-italic[纯 serving 与 RL 之间有一个实际区别。对纯 serving 负载，FlashInfer 出于性能考虑可能在 FP4 量化路径的某些部分使用快速数学（fast math）。这是合理的 serving 默认行为，但 RL 权重更新需要与训练侧量化器精确一致。] 本方案中，我们设置：

```
FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1
```

每一个接触 rollout 权重的后端，要么精确实现这套量化契约，要么把近似行为做成显式可选（opt-in）。

## 七、细粒度精度控制

> 实践中，单一的全局精度开关对低精度 RL 是不够的。有些张量应保持 BF16，但挑选它们只是问题的一半：同一决策必须在 Hugging Face checkpoint 转换、Megatron 训练、SGLang rollout 和实时权重导出四个环节中一致执行。

我们在 Miles 中通过按数量与按名称的 BF16 例外实现了张量级精度控制，覆盖 checkpoint 转换、训练、rollout 与实时导出，参见下面 PR：

miles#614：[feat] Expand mxfp8 quantization to support fine-grained layer precision config[29]

miles#1054：Use TE NVFP4 reference quantizer and add bitwise exact tests[30]

miles#1261：NVFP4 RL[31]

我们还实现了 SGLang 对由此产生的混合精度 checkpoint 的支持：

sglang#18742：[RL] Support per-layer mixed FP8/BF16 serving for FP8 checkpoints[32]

sglang#20214：[FlashInfer v0.6.6][RL] Support fp8-last-n-bf16 RL for `flashinfer_trtllm_routed` moe backend[33]

具体来说，转换使用 `--num-layers-at-start-in-bf16` 与 `--num-layers-at-end-in-bf16`；Megatron 训练将这些数量与 `--first-last-layers-bf16` 组合；SGLang 则直接 serve 混合精度 checkpoint。

### 7.1 层精度选择

> 按照 NVIDIA NVFP4 预训练论文[34]的建议，我们将少量末尾层保持在高精度。实验中，将最后 15% 的层保持 BF16 能显著降低训练-推理不匹配并稳定梯度。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/18.webp)

::: image-caption
末尾层保持 BF16 的效果——保持最后 2/4/8 层 BF16 对训练-推理不匹配的影响对比。横轴是训练步数，范围 0‑2000，纵轴代表对数概率绝对差值，范围 0‑0.05。图中共六条曲线，灰色 bf16‑bf16 曲线整体数值最低，在 0.01 附近小幅波动；其余五条 MXFP8 相关曲线数值更高，集中在 0.02‑0.03 区间，蓝色 mxfp8‑mxfp8 处于最高位置，粉色、紫色曲线依次靠下，红色 mxfp8‑mxfp8‑last8 整体处于 MXFP8 组内最低，各曲线均伴随噪声震荡，随训练步数增加，多数 MXFP8 曲线呈缓慢抬升趋势，灰色曲线略有下降。
:::

而在我们的实验中，把靠前的层开成 BF16 并不能带来有意义的训练-推理不匹配改善：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/19.webp)

::: image-caption
首层 BF16 与末层 BF16 的对比——末层 BF16 有效，首层 BF16 收益不明显。横轴为训练步数，范围 0‑1400，纵轴是对数概率绝对差值，区间 0‑0.05。图中有四条曲线，浅灰色 ds48b‑bf16‑r3 整体数值最低，在 0.01 附近震荡。深灰色 ds48b‑mxfp8‑r3 处于最高位置，大致在 0.025‑0.03 区间波动。红色与蓝色曲线数值相近，集中于 0.02 附近，两条曲线走势贴近，存在高频噪声抖动。随着训练步数推进，深灰色曲线整体呈小幅下降后略有回升，红、蓝曲线整体平稳，浅灰色基准曲线波动很小，各线条均附带浅色噪声包络。
:::

**将共享专家（shared experts）保持高精度也能降低训练-推理不匹配，且性能与显存开销很小**。

- 路由专家是稀疏激活的，其输出经过高精度的加权归约；
- 而共享专家始终激活，其精度误差会影响通过该块的每一个 token。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/20.webp)

::: image-caption
共享专家高精度消融——共享专家保持高精度对降低训练-推理不匹配的作用。Train‑Rollout Logprob Diff，横轴为训练步数，范围 0‑700，纵轴代表对数概率绝对差值，取值区间 0‑0.05。图中共三条曲线，浅灰色 ds48b‑bf16‑r3 数值最低，在 0.01 附近小幅震荡。红色 ds48b‑mxfp8‑r3 整体处于最高位，在 0.027‑0.03 区间波动。蓝色 ds48b‑mxfp8‑r3‑se‑bf16 居于二者之间，维持在 0.02‑0.024 附近。每条主曲线外侧都伴随浅色噪声包络，红色曲线存在小幅起伏，蓝色曲线走势相对平稳，浅灰色基准曲线波动微弱，整体没有明显的持续上升或下降趋势。
:::

### 7.2 案例研究：MXFP8 MLA

> 对 MLA 模型，`kv_b_proj` 是一个重要的 MXFP8 特例。吸收（absorbed）与非吸收 MLA 模式可能使用不同的收缩轴（contraction axis），而 MXFP8 使用一维微缩放块。

改变收缩轴会改变哪些元素共享同一个 scale。同样的顾虑也适用于其他一维格式，包括 NVFP4。**原始 DeepSeek-V3[35] FP8 方案没有这个问题，因为它的权重 scale 块是 128x128 的二维块而非一维块**。我们将这些投影张量保持 BF16，以避免隐藏的再量化并保持训练-推理一致性。

```
configs:  bf16:    transformer_engine_config_type: "TEQuantizationParams"    training_recipe: {}matchers:  mla_kv_up_proj_bf16:    type: "glob"    enabled: true    pattern: "*.self_attention.linear_kv_up_proj"    config: "bf16"  absorbed_k_up_proj_bf16:    type: "glob"    enabled: true    pattern: "*.self_attention.linear_k_up_proj"    config: "bf16"  absorbed_v_up_proj_bf16:    type: "glob"    enabled: true    pattern: "*.self_attention.linear_v_up_proj"    config: "bf16"
```

配套的按名称转换期与 rollout 期参数为：

```
--extra-high-precision-layers-hf .kv_b_proj.--extra-high-precision-layers-megatron .linear_kv_up_proj .linear_k_up_proj .linear_v_up_proj
```

对 DeepSeek-V3[36] 风格的 MLA，这个 BF16 例外很小：一个形状为 `32768 x 512` 的 `kv_b_proj` 张量，以 BF16 存储、跨 61 层，约占 1.90625 GB。

## 八、实验结果：8x B200 上的 Qwen3-30B-A3B 方案消融

> 为保证对比一致，所有实验均在 8x B200 上使用默认 Miles 设置进行同步 Qwen3-30B-A3B RL。固定负载为在 dapo-math-17k 上进行 GRPO 风格训练，每个 prompt 采样 8 条 rollout 样本，最大响应长度 8192 token。

这只是一个方案消融设置，并非完整调优的训练或 serving 基准。KL 路径开启用于诊断，但其系数为 0.0，因此 KL 在本消融中不构成优化惩罚。

硬件划分为 4 卡 rollout + 4 卡训练。所有低精度方案均满足：

- 启用 MoE rollout 路由回放（routing replay）。
- 最后 15% 的层保持 BF16。
- 低精度权重使用 0 权重衰减以保证稳定。
- SGLang rollout 使用 BF16 KV cache，低精度 MoE 路径使用 FlashInfer TRTLLM 路由 MoE 后端。

我们对比六种配置：

1. BF16 训练 + BF16 rollout。
2. 端到端 MXFP8 训练 + MXFP8 rollout。
3. MXFP8 rollout 与前向 + 高精度反向。
4. MXFP8 rollout 与前向 + 反量化反向。
5. 逐 token NVFP4 MoE rollout 与前向 + 高精度反向。
6. 逐 token NVFP4 MoE rollout 与前向 + 反量化反向。

### 8.1 训练-推理不匹配

> 如预期，两种低精度格式的训练-推理不匹配都高于 BF16，而每种格式内部两种反向选择的表现相近。数值在本消融的合理范围内。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/21.webp)

::: image-caption
各反向模式下训练侧与 rollout 侧 logprob 差异对比。横轴为训练步数，范围 0‑360，纵轴代表对数概率绝对差值，区间 0‑0.05。图中包含六条曲线，每条主线附带浅色噪声包络。红色 bf16_train_bf16_infer 整体数值最低，缓慢小幅下行，维持在 0.01 附近。三条 MXFP8 相关曲线处在中间区间，稳定在 0.02 上下小幅震荡。绿色与深绿色两条 NVFP4 曲线位于最上方，数值约 0.03‑0.035，波动幅度更大。各组曲线整体无明显单向升降趋势，全程保持分层排布，伴随高频抖动。
:::

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/22.webp)

::: image-caption
各反向模式下 KL 损失对比。NVFP4 初始参考 KL 高于 BF16 与 MXFP8——Miles 的该诊断指标是相对于 Megatron BF16 参考模型计算的，因此它同时包含了低精度策略与 BF16 参考之间的固有差异，不应被读作独立的优化惩罚项。
:::

### 8.2 奖励

> 尽管诊断性不匹配更高，五条低精度奖励曲线都与 BF16 奖励曲线紧密贴合。

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/23.webp)

::: image-caption
各反向模式下原始奖励曲线对比——这是本方案消融的关键结果：在 Qwen3-30B-A3B B200 设置下，Blackwell 原生低精度在保持可观察学习曲线的同时提升了 rollout 效率。
:::

### 8.3 性能

> MXFP8 与 NVFP4 相对 BF16 均降低了 rollout 耗时：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/24.webp)

::: image-caption
各反向模式下 rollout 耗时对比。横轴为 Rollout Step，范围 0‑360，纵轴为推演耗时，单位秒。图中有六条曲线，每条主线附带浅色噪声包络。红色 bf16_train_bf16_infer 始终处于最上方，耗时整体最高；三条 MXFP8 系列曲线居于中间；绿色与深绿色 NVFP4 曲线处于最下方，耗时最低。全部曲线整体呈现下降趋势，随步数增加各组耗时逐步降低，曲线之间保持明显分层。各曲线均伴随较大噪声抖动，红色基准曲线波动尤为突出，量化组别相比 bf16 基准展现出更短的推演时间。
:::

对 NVFP4 rollout，FlashInfer 直接在激活量化内核路径内计算在线逐 token FP32 scale，而非作为单独的前处理步骤。因此所报告的 rollout 性能已包含在线 scale 计算的开销。

> 训练侧，MXFP8 各变体快于 BF16，而 NVFP4 反向覆盖（backward-override）各变体在本次测量的实现中更慢：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/25.webp)

::: image-caption
各反向模式下训练耗时对比。横轴为 Rollout Step，范围 0‑360，纵轴为训练耗时，单位秒。图中有六条曲线，每条主曲线附带浅色噪声包络。初始步数各曲线数值差异较大，随后快速收敛分层。深绿色与绿色 NVFP4 曲线整体耗时最高，红色 bf16 基准处于中间区间，三条 MXFP8 系列曲线耗时更低。整体上各组训练耗时随步数推进呈下降趋势，后期趋于平稳，全程伴随明显噪声抖动。不同量化方案之间存在稳定的时间差距，MXFP8 组别相比其他配置拥有更短的训练耗时。
:::

训练侧的差距来自本消融所用的实现，而非 FP4 Tensor Core 的固有限制。我们的 TransformerEngine 路径将逐 token FP32 缩放作为单独的 PyTorch 算子执行（TransformerEngine#2931：Implement row-scaled NVFP4 fprop recipe[37]），而非缩放融合进内核 epilogue 的原生逐 token NVFP4 GEMM 路径。我们已实现并上传了融合版 cuDNN frontend 内核（cudnn-frontend#251：Add row-scale support to grouped GEMM quant[38]），TransformerEngine 侧的集成仍待完成。反量化反向则额外增加了上述反量化步骤。由于该 RL 负载以 rollout 为主，即使训练路径尚未完全加速，rollout 加速仍然很有意义。

除本消融外，humans&[39] 已在生产中使用同一方案家族与组件，开展大规模、长上下文、多智能体异步 RL 研究。

## 九、在线 NVFP4 模型 serving

> 同一套逐 token NVFP4 方案也可用于以在线训练后量化（PTQ）方式 serve FP8/BF16 模型 checkpoint：在保持良好模型质量的同时没有标定伪影，计算性能也与传统 NVFP4 serving 持平。

在线逐 token 激活缩放将激活离群点局部化到每个 token，并免去了从标定 pass 中携带静态激活 FP32 scale 的需要。在 FlashInfer 中，FP32 scale 计算被融合进激活量化内核路径，因此在线缩放不需要单独的前处理内核。

**PTQ 模型质量还可以通过 Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling 技术进一步提升，且 decode 延迟零额外暴露**，详见姊妹篇 humans&[40] 文章。

::: block-1
我们在 sglang#26083：Implement online nvfp4 quantization[41] 中实现并开源了这条 SGLang 在线 NVFP4 serving 路径。在 SGLang 中使用该特性，只需在任何现有 BF16/FP8 MoE 模型 serving 命令中添加 `--quantization nvfp4_online`。
:::

## 十、未来工作

### 10.1 去掉额外的 BF16 权重副本

> 尽管 rollout 与训练执行同一套低精度方案，Megatron 仍会额外保存一份 BF16 权重副本。这增加了显存占用，限制了低精度路径的实际显存收益。

Megatron 已有 `--fp8-param-gather` 与 `--fp4-param-gather`，但 Blackwell 原生路径仍在成熟中。Megatron-Bridge 跟踪 issue Megatron-Bridge#3801：[bug] Investigate convergence of performance features with Qwen3.5 VL as proxy model[42] 反映了稳健低精度参数收集所需的后续工作。NVFP4 的 `--fp4-param-gather` 尚不支持本方案使用的一维 1x16 权重布局。

### 10.2 偶发梯度尖峰

> 高精度反向的 NVFP4 变体仍可能出现偶发的梯度尖峰：

![文章插图](/articles/2026-08-04-blackwell-8-bit-4-bit-ba83f7e2/images/26.webp)

::: image-caption
反量化反向传播能够降低本次消融实验中最严重的尖峰，但无法将其完全消除。包括4/6方案以及符合链式法则的反向传播方案在内的更多高级技术，已在配套的 humans& 帖子中进行探讨。上图横轴为 Train Step，范围 0‑360，纵轴为梯度范数 Gradient Norm。图中共三条曲线，分别为红色 bf16 基准、绿色 nvfp4 高精度、深绿色 nvfp4 反量化。三条曲线整体交织震荡，大部分时间集中在 0.03‑0.06 区间，频繁出现波峰波谷。曲线存在多处极端尖峰与尖谷，绿色 nvfp4‑high‑precision 在约 300 步出现全图最高尖峰，数值接近 0.125；红色基准在 70 步附近出现明显低谷。全程没有稳定单向升降趋势，不同配置梯度范数波动模式相近，量化方案与 bf16 基准之间未出现持续性大幅偏移。
:::

### 10.3 重构权重更新接口

低延迟 FlashInfer 后端通常需要 padding、swizzling、shuffling 等后端特定的权重布局。这些变换对 serving 很自然，但会让 RL 实时权重更新和 RDMA 变复杂，因为训练侧通常持有另一套规范张量布局。

Miles 与 SGLang 中的相关工作旨在保留高性能 serving 布局的同时，让每一种权重变换显式化、可验证，并减少对后端私有细节的依赖。

## 十一、在 Miles 中试用 NVFP4 方案

> 以下环境变量可复现逐 token NVFP4 高精度反向设置：

```
NVTE_NVFP4_ROW_SCALED_ACTIVATION=1NVTE_BACKWARD_OVERRIDE=high_precisionNVTE_NVFP4_DISABLE_2D_QUANTIZATION=1NVTE_NVFP4_DISABLE_RHT=1NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1
```

将 `NVTE_BACKWARD_OVERRIDE=dequantized` 即可切换为反量化反向变体，其余方案不变。

对 Miles 启动脚本，该方案将这些环境变量与 `--fp4-format e2m1`、`--fp4-recipe nvfp4`、与 checkpoint 转换时相同的 BF16 首末层控制，以及如下 TransformerEngine 精度配置搭配使用：

```
configs:  nvfp4:    transformer_engine_config_type: "TEQuantizationParams"    training_recipe:      fp4_quantization_recipe: "nvfp4"  bf16:    transformer_engine_config_type: "TEQuantizationParams"    training_recipe: {}matchers:  routed_experts_fc1_nvfp4:    type: "glob"    enabled: true    pattern: "*.mlp.experts.linear_fc1"    config: "nvfp4"  routed_experts_fc2_nvfp4:    type: "glob"    enabled: true    pattern: "*.mlp.experts.linear_fc2"    config: "nvfp4"  default_bf16:    type: "glob"    enabled: true    pattern: "*"    config: "bf16"
```

参考资料

[1] humans&: *https://humansand.ai/*

[2] 《Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL》: *https://www.lmsys.org/blog/2025-11-25-fp8-rl*

[3] 《Squeezing 1TB Model Rollout into a Single H200: INT4 QAT RL End-to-End Practice》: *https://www.lmsys.org/blog/2026-01-26-int4-qat*

[4] miles#615：[Roadmap] Blackwell MXFP8 and NVFP4 RL training: *https://github.com/radixark/miles/issues/615*

[5] DeepSeek-V3: *https://arxiv.org/html/2412.19437v2*

[6] HGX 平台: *https://www.nvidia.com/en-us/data-center/hgx/*

[7] MXFP8 文档: *https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html*

[8] NVFP4 介绍: *https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/*

[9] deepseek-v3-gb200-optimization.md: *https://github.com/NVIDIA/Megatron-LM/blob/eb0783b6d35607ef1953eaca60b37b886b1a25d0/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-optimization.md*

[10] DeepSeek-V3 FP8 方案: *https://arxiv.org/html/2412.19437v2*

[11] flashinfer#2581：Implement `cutlass_fused_moe` mxfp8: *https://github.com/flashinfer-ai/flashinfer/pull/2581*

[12] sglang#17449：Add mxfp8 support for online quantization, Triton dense linear, and CUTLASS MoE: *https://github.com/sgl-project/sglang/pull/17449*

[13] sglang#19537：[FlashInfer v0.6.4] [RL] Integrate FlashInfer mxfp8 gemm, MoE, and routed MoE: *https://github.com/sgl-project/sglang/pull/19537*

[14] sglang#21576：[FlashInver v0.6.7] Integrate flashinfer_trtllm mxfp8 gemm: *https://github.com/sgl-project/sglang/pull/21576*

[15] sglang#28459：[RL] Fix FlashInfer TRTLLM MXFP8 dense weight layout: *https://github.com/sgl-project/sglang/pull/28459*

[16] DeepSeek-V3: *https://arxiv.org/html/2412.19437v2*

[17] NVFP4 预训练方案: *https://arxiv.org/html/2509.25149v2*

[18] Cursor Composer 2 技术报告: *https://arxiv.org/html/2603.24477v2*

[19] TransformerEngine#2931：Implement row-scaled NVFP4 fprop recipe: *https://github.com/NVIDIA/TransformerEngine/pull/2931*

[20] cudnn-frontend#251：Add row-scale support to grouped GEMM quant: *https://github.com/NVIDIA/cudnn-frontend/pull/251*

[21] flashinfer#3027：[feat] Trtllm-gen Per-token Nvfp4 MoE: *https://github.com/flashinfer-ai/flashinfer/pull/3027*

[22] sglang#22918：[FlashInfer v0.6.11] [RL] Support FlashInfer per-token NVFP4 MoE: *https://github.com/sgl-project/sglang/pull/22918*

[23] sglang#22204：[RL] Refactor NVFP4 shuffling/swizzling to in-place replacement: *https://github.com/sgl-project/sglang/pull/22204*

[24] TransformerEngine#2644：Add `NVTE_BACKWARD_OVERRIDE=high_precision|dequantized`: *https://github.com/NVIDIA/TransformerEngine/pull/2644*

[25] humans&: *https://humansand.ai/blog/nvfp4-rl*

[26] TransformerEngine#2865：Optimizations for MXFP8/NVFP4 dequantize kernels: *https://github.com/NVIDIA/TransformerEngine/pull/2865*

[27] TransformerEngine#2644: *https://github.com/NVIDIA/TransformerEngine/pull/2644#issuecomment-4026583299*

[28] random, boundary, zeros, maxes]`。该量化器对齐已在 [flashinfer#3387：Make cute dsl mxfp8/nvfp4 quantizer bitwise exact: *https://github.com/flashinfer-ai/flashinfer/pull/3387*

[29] miles#614：[feat] Expand mxfp8 quantization to support fine-grained layer precision config: *https://github.com/radixark/miles/pull/614*

[30] miles#1054：Use TE NVFP4 reference quantizer and add bitwise exact tests: *https://github.com/radixark/miles/pull/1054*

[31] miles#1261：NVFP4 RL: *https://github.com/radixark/miles/pull/1261*

[32] sglang#18742：[RL] Support per-layer mixed FP8/BF16 serving for FP8 checkpoints: *https://github.com/sgl-project/sglang/pull/18742*

[33] sglang#20214：[FlashInfer v0.6.6][RL] Support fp8-last-n-bf16 RL for `flashinfer_trtllm_routed` moe backend: *https://github.com/sgl-project/sglang/pull/20214*

[34] NVIDIA NVFP4 预训练论文: *https://arxiv.org/html/2509.25149v2*

[35] DeepSeek-V3: *https://arxiv.org/html/2412.19437v2*

[36] DeepSeek-V3: *https://arxiv.org/html/2412.19437v2*

[37] TransformerEngine#2931：Implement row-scaled NVFP4 fprop recipe: *https://github.com/NVIDIA/TransformerEngine/pull/2931*

[38] cudnn-frontend#251：Add row-scale support to grouped GEMM quant: *https://github.com/NVIDIA/cudnn-frontend/pull/251*

[39] humans&: *https://humansand.ai/*

[40] humans&: *https://humansand.ai/blog/nvfp4-rl*

[41] sglang#26083：Implement online nvfp4 quantization: *https://github.com/sgl-project/sglang/pull/26083*

[42] Megatron-Bridge#3801：[bug] Investigate convergence of performance features with Qwen3.5 VL as proxy model: *https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/3801*

::: related-title
**相关推荐**
:::

::: related-links
- **[Blackwell 架构下的高速扩散模型：基于 Diffusers 与 TorchAO 的 MXFP8 及 NVFP4 应用](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899996&idx=1&sn=c9541fb4324c46b5e2708ea19ca2bbf7&scene=21#wechat_redirect)**
- **[BF16 vs NVFP4精度鸿沟收窄术：4-bit LLM 训练方案 CHON 用二阶误差补偿实现0.58% Loss Gap](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898732&idx=1&sn=de240d2ff2823166c1d281fd94b01717&scene=21#wechat_redirect)**
- **[MXFP4近无损LLM训练：MXFP4 GEMM结合RHT与SR，比FP8快2倍计算且6.7B GPT模型近无损](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891301&idx=1&sn=07fa64e075a9cf2fa5d6cced89b8b108&scene=21#wechat_redirect)**
- **[突破NVFP4量化性能瓶颈！MIT与NVIDIA提出Four Over Six开源方案：近BF16困惑度与&lt;2%推理开销兼得！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895052&idx=1&sn=7bd638a89f18e3a5c66c5bf67cc22aeb&scene=21#wechat_redirect)**
- **[面向 LLM 的 MXFP4 专属 PTQ 方案：BRQ 策略实现跨模型量化精度提升与 40% 推理提速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893841&idx=1&sn=84e6bf385dc318254943c9d16cc4a861&scene=21#wechat_redirect)**

:::
