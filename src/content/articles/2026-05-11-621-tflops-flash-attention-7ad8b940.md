关键词：:mdnice-bold-italic[内核广播优化]、**推荐系统推理**、Flash Attention、**GPU 内核融合**、协同设计

> 推荐系统推理与大语言模型推理有一个很不一样、却长期被忽视的特点：:mdnice-bold-italic[同一个请求里的大量候选内容，往往共享同一份用户上下文。传统实现里，这意味着用户嵌入、用户序列等“请求级共享特征”必须反复复制，才能与候选批次对齐并送入交互层计算。这个复制动作看起来只是“广播”，但在工业规模部署里，它实际会持续吞噬显存带宽、抬高 IO 成本，并让延迟随着候选数线性恶化。]

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/01.webp)

- **In-Kernel Broadcast Optimization: Co-Designing Kernels for RecSys Inference**
- https://pytorch.org/blog/in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/
- 代码：https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/ikbo
- 1.5 万字，阅读 56 分钟，**播客 17 分钟**

相关推荐

- [小红书LarC：应用大规模深度学习的分钟级实时推荐系统](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886683&idx=1&sn=6562f5e7c79c04ab076cf19c7a67eabb&scene=21#wechat_redirect)
- [EdgeRec: 手机淘宝端上推荐的问题与挑战](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885842&idx=1&sn=ce61f3215ecc0ea89b1db793ea2597fe&scene=21#wechat_redirect)
- [Meta AI 基础设施的十年演进、下一代基础设施蓝图与 AI 自研芯片之路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896790&idx=1&sn=a13b82d8c267fcdbd6f3e20c53a1429b&scene=21#wechat_redirect)

> Meta 在这篇文章里提出 In-Kernel Broadcast Optimization，也就是 IKBO。它的核心判断非常直接：广播首先是数据布局问题，不是计算本身不可避免的一部分。

既然如此，就不该在内核外提前把共享张量物化出来，而**应让内核直接接受“用户批次”和“候选批次”这两个天然不一致的输入维度，在计算过程中按索引完成广播语义**。围绕这个想法，文章从内核、编译、推理运行时三个层面给出协同设计方案，并用 Linear Compression 与 Flash Attention 两个案例，:mdnice-bold-italic[展示如何把一次看似普通的广播消除，做成能支撑生产级推荐系统扩展性的基础能力。]

## 本文目录

- 关键问题
  - 问题一：IKBO 是否以控制流发散与同步开销取代了显存带宽瓶颈？
  - 问题二：这种强耦合设计是否以牺牲可组合性为代价，成为模型迭代的瓶颈？
- 一、内核内广播优化：消除内存与计算冗余
  - 1.1 消除内存与计算冗余
  - 1.2 内核优化类型
  - 1.3 端到端系统设计
  - 1.4 与其他方法的比较
- 二、内核深入案例一：IKBO Linear Compression
  - 2.1 矩阵乘法分解
  - 2.2 内存布局优化
  - 2.3 候选 GEMM 的内核内广播融合
  - 2.4 基于 TLX 的 warp-specialized 多阶段融合
- 三、内核深入案例二：IKBO Flash Attention
  - 3.1 IKBO Flash Attention 在推荐系统边界条件下解决 IO 受限问题
  - 3.2 在 TLX 上把现代内核技术（FA3、FA4）与 IKBO 结合
  - 3.3 通过模型协同设计融合 Self + Target Attention
- 四、基准测试与结果汇总
- 五、结论与未来方向
- 参考文献
- 附录
  - 附录 1：Benchmark 设置
  - 附录 2：算术强度分析
  - 附录 3：第 2.1 节的详细结果分析
  - 附录 4：瓶颈分析方法
  - 附录 5：第 2.2 节的详细结果分析
  - 附录 6：第 2.3 节的详细结果分析
  - 附录 7：Release-Acquire 同步协议
  - 附录 8：TLX 与 Triton 的 NCU Profiling 指标
  - 附录 9：普通 Flash Attention 与 IKBO Flash Attention 的 roofline 分析
  - 附录 10：IKBO TLX FA3 的 SMEM 占用
  - 附录 11：在推荐系统边界条件下，对 IKBO FA、CuTeDSL FA4 Hopper 与 TLX FA3 Hopper 的 benchmark
  - 附录 12：instruction cache miss 会在 consumer-2 warpgroup 上造成显著延迟

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/01.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 关键问题

### 问题一：IKBO 是否以控制流发散与同步开销取代了显存带宽瓶颈？

> IKBO 通过消除显式广播来减少显存占用和带宽压力，但这是否以一种新的瓶颈——即“控制流发散与同步开销”——取代了旧的瓶颈？当候选集与用户的比例极度不均或动态变化时，内核内部的复杂分支逻辑和跨 warp 同步是否会成为新的性能悬崖？

作者暴露的证据表明，这种代价确实存在但可被有效管理。在“奇数候选处理”部分，当一个用户的候选数为奇数时，某个 warpgroup 必须进入空闲状态并“排干”缓冲区以防止死锁。这直接证明了控制流分歧的存在。

**关键在于量级对比**。作者明确指出，在真实的约 70:1 候选-用户比下，这个空闲路径的触发概率“小于 0.7%”，开销可以忽略。而它所消除的显存复制开销则是系统性的、随候选数线性增长的——例如原方案中需将用户特征复制约 70 次，占用 0.87 GB 中间显存流量。

::: block-1
更深层的设计哲学是：**TLX 的 warp 特化并非规避控制流分歧，而是将其显式化、可控化**。通过将 CTA 划分为生产者与消费者组，并用命名屏障精确管理同步，IKBO 将不可预测的分支变成了结构化的流水线:mdnice-bold-italic[。NCU 数据显示，尽管存在同步，DRAM 吞吐量仍从 39% 提升至 52%，L2 瓶颈利用率增至 84%。]

**综合来看，IKBO 并非消除了瓶颈，而是用可控的同步开销换取了显存带宽的显著解放，这在批量大的推荐推理场景下是净收益。**
:::

### 问题二：这种强耦合设计是否以牺牲可组合性为代价，成为模型迭代的瓶颈？

> :mdnice-bold-italic[IKBO 声称通过“Kernel-模型-系统”协同设计实现了显著加速，并使模型设计摆脱了约束。但这种灵活性是否以牺牲“局部性”和“可组合性”的软件工程原则为代价？] 当需要调试精度问题或快速更换模型结构时，这种高度耦合的优化是否反而成为迭代速度的瓶颈？

作者展示的恰恰是相反的设计思路：**通过系统架构而非约定俗成来保护模块化。**

关键证据在作者提到：“训练时不改变模型代码”和“推断时自动将标准算子替换为 IKBO 等价实现”。这意味着模型作者无需感知 IKBO 的存在，仍使用标准 PyTorch 算子编写模型。编译器和推断运行时会自动解析批次继承关系并进行替换。这不是强耦合，而是将复杂性封装在了底层。

关于精度问题，在附录 3 中正面回应了分解 GEMM 的数值差异，称其“满足精度要求”，表明团队对此有意识且认为风险可控。至于算子语义被破坏的质疑，IKBO 并未创造新算子，只是改变了已有通用算子的内部实现。研究人员替换激活函数时，无需修改任何代码，内核团队独立维护 IKBO 变体库即可。

::: block-1
**最大的风险在于对编译器自动化的依赖**。作者承认，跨多个生产模型解析批次谱系“需要系统化的自动化”。如果这个自动化层出错，确实可能静默污染结果。但这正:mdnice-bold-italic[说明文章所描述的生产落实是成熟的——IKBO 已在 GPU 和 MTIA 上部署，服务“LLM 规模的模型”，证明这套自动化已足够可靠。 结论是：IKBO 通过将广播语义降为运行时/编译器职责，在获得性能的同时成功保护了上层模型的模块化。]
:::

## 一、内核内广播优化：消除内存与计算冗余

> 本文介绍 In-Kernel Broadcast Optimization（IKBO）：**一种面向推荐模型推理的内核—模型—系统协同设计方法，用来消除冗余的用户嵌入广播**。在生产级推荐系统中，对于给定请求，用户嵌入在所有候选之间都是相同的，但标准做法仍要求显式复制，造成随候选数量扩大的内存带宽与计算浪费。**IKBO 的核心洞察是：广播是数据布局问题，而不是计算必需**。每个 IKBO 内核都以其天然且不匹配的用户输入和候选输入批大小作为输入，并在内核内部处理广播，因此不会有任何复制后的张量被物化出来。本文通过两个内核深入案例来展示这一方法：Linear Compression 与 Flash Attention。

IKBO 已部署到 Meta 的推荐系统推理栈中，从早期排序到晚期排序模型，覆盖 GPU 与 MTIA（Meta Training and Inference Accelerator）。在协同设计后的模型上，它可将计算密集型网络延迟最多降低三分之二。它也是 Meta Adaptive Ranking Model[1] 所依赖的、以请求为中心且推理高效框架的可扩展性骨干，用于在生产中服务 LLM 规模的模型。

在 H100 SXM5 上，我们的 IKBO 线性压缩内核经过四个渐进式协同设计阶段：矩阵乘法分解、内存对齐、广播融合，以及通过 TLX (Triton Low-Level Extensions)[2] 实现的 warp-specialized 多阶段融合，获得约 4 倍加速。对于 Flash Attention，IKBO 相较未做协同设计的 CuTeDSL FA4-Hopper，实现了 2.4×/6.4× 的吞吐提升（仅注意力内核 / 注意力内核加广播开销），达到 621 BF16 TFLOPs。

与系统级广播或网络拆分这类“绕开复制”的方法不同，IKBO 在计算原语层面直接消除复制，以近似独立成本实现高密度交互质量。

注：本工作完成于作者在 Meta 任职期间。

### 1.1 消除内存与计算冗余

> 当用户打开信息流时，推荐系统必须对数百到数千个候选条目打分，以决定向其展示什么。模型输入可分为两类：

- 一类是用户特征，例如浏览历史、画像、上下文，它们在一个请求中的每个候选上都完全相同；
- 另一类是候选特征，例如物品 ID、类别、互动统计，它们对每个候选都不同。

两类特征都会经过 embedding lookup 以及后续处理，形成嵌入表示。在模型中的不同位置，交互层（例如线性投影、特征交叉、目标注意力）会把用户嵌入与候选嵌入结合起来。我们将一个请求中对所有候选共享的嵌入称为 Request-Only（RO），将每候选独有的嵌入称为 Non-Request-Only（NRO）。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/02.webp)

::: image-caption
图1. 一个极度简化的推荐系统推理数据流。仅请求（RO）的用户嵌入在进入交互层之前，必须被广播（复制）以匹配非仅请求（NRO）的候选物品批次维度。IKBO通过在每个内核内部处理广播，消除了这种物化。该图清晰地对比了传统方法与IKBO方案。传统方法中，广播是一个外部的、显式的数据准备步骤，而IKBO则将其内化为计算过程的一部分。这不仅是执行位置的改变，更意味着计算图的结构得到简化，运行时不再需要为庞大的复制张量分配和传输内存。
:::

交互层要求张量具有匹配的批维度。在一个由约 15 个用户服务、却包含 1024 个候选的批次中，RO 嵌入必须先被广播，即复制约 70 次，才能在任何交互发生之前与 NRO 批大小对齐（图 1）。随着模型架构从 DLRM [1] 和 DCN [2] 演进到 HSTU [3] 与 X 的 Phoenix [4] 这类序列模型，用户—候选交互不断增强。:mdnice-bold-italic[但更丰富的交互也意味着更高代价：用户特征必须广播到所有候选上。在推理中，当批大小处于 10 到 10000+ 时，这种复制开销会带来显著的计算和内存成本，并随候选数线性增长。]

**广播是数据布局问题，而不是计算必需。用这一视角重新看待模型与推理系统，就能在每一层发现优化空间**：推理运行时消除系统级广播，仅用户相关的模型层在更小的用户批大小上运行，而同时混合处理两类数据的内核则被重新设计为在内部处理广播，因此不会有任何复制后的张量被物化出来。IKBO 已部署到 Meta 的推荐系统推理栈中，从早期排序到晚期排序，覆盖 GPU 与 MTIA，在协同设计模型上可将计算密集型网络延迟最多降低三分之二。

::: block-1
本文聚焦内核层，通过两个深入案例展开：Linear Compression 与 Flash Attention。
:::

### 1.2 内核优化类型

- Type I — 可分解操作。通过数学重构，可让 Request-Only（RO）部分在小批大小下独立计算，并仅在最后与 Non-Request-Only（NRO）部分合并。这同时节省内存带宽与计算量。
- Type II — 纯内存优化。在内核内部处理 RO-NRO 广播，避免冗余数据移动，使内核远离 IO 受限状态。

### 1.3 端到端系统设计

> 部署 IKBO 会触及基础设施栈的三层：

1. 内核：自定义 GPU 内核，接受不匹配的 RO/NRO 批大小，并在内部处理广播（见后文第 2、3 节）。
2. 编译规格：ML 编译器需要每个算子的动态形状范围，以便选择匹配形状的内核。只有一个批大小时，这很简单；**当存在两个（用户与候选）甚至更多批大小时，要在生产模型中可靠判断每个算子究竟使用哪一个批维度——尤其在交互使批来源变得模糊时——就需要系统化自动化方案。**
3. 推理：运行时把 candidate-to-user 映射传入模型，而不是先物化广播结果。

这些内核通过两种方式进入模型：

1. 直接采用：模型作者将 IKBO 内核直接集成进模型定义。当训练时 candidate-to-user 比值大于 1，同样的内核也能降低训练成本。
2. 推理时变换：一个 pass 会在推理时自动把标准算子替换为 IKBO 等价算子，无需修改模型代码。

最终效果是：广播从推理的每一个阶段消失，而对模型架构没有约束，对基础设施的变更也仅限于推理运行时的映射接口。

### 1.4 与其他方法的比较

> 已有方法更多是在绕开广播，而非真正消除它。

1. 系统级广播会在 GPU dispatch 前先物化复制后的张量——实现简单，但代价高，且成本随候选数线性增长。
2. 网络拆分（ROO）[5] 将模型分成 RO 和 NRO 子网络，能减少冗余工作，但会限制用户—候选交互发生的位置，而且在较小 RO 批大小下仍会引入额外成本。

这两种方式都保留了广播这一“被物化的张量”。IKBO 则在计算原语层面直接消除它：节省量会随 candidate-to-user 比值扩大，任意交互模式都无需承担广播成本，而融合内核中的完整 NRO 批维度还能提供 GPU occupancy。

IKBO 已部署在 GPU 与 MTIA 加速器上。**本文聚焦 H100 GPU 内核设计，以说明其核心优化原则。**

## 二、内核深入案例一：IKBO Linear Compression

> Linear Compress Embedding（LCE）通过学习到的投影 `(M, K) @ (B, K, N) → (B, M, N)` 来压缩输入嵌入 `(B, K, N)`，并被广泛应用于 Meta 的推荐模型中，例如 Wukong [6]。下面我们按四个渐进优化阶段展开。

### 2.1 矩阵乘法分解

基线 LCE 会在所有 B 个候选上执行一次 batched matmul。输入嵌入在 K 维上拼接了用户部分与候选部分——但对同一用户的所有候选来说，用户嵌入是相同的。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/03.webp)

::: image-caption
图 2. LCE分解：基线批量矩阵乘法（左上），沿K维度的嵌入分离与用户去重（右上），两个独立的GEMM操作，最后对压缩输出进行广播加法（下）。该图展示了从“复制后计算”到“计算后轻量合并”的转变。第一阶段分离用户和候选物品的嵌入，立刻将用户侧的矩阵乘尺寸降低了数十倍。第二步将广播从完整的、高维的嵌入向量推迟到了低维的、压缩后的结果上，极大地降低了广播操作的数据量。
:::

把广播推迟到矩阵乘法之后。由于 W 与批次无关，我们利用线性性质进行分解：沿 K 维将用户嵌入块与候选嵌入块分开，对重复用户嵌入去重，然后在各自天然批大小上分别执行两个独立 GEMM。**这样，用户嵌入不再在 matmul 之前被复制，而只需要广播较小的压缩结果，见图 2**。在大约 70 的 candidate-to-user 比值下（这个是个典型设置直），用户批大小从 `B=1024` 缩小到 `B_user ≈ 15`，使用户侧计算量降低 70 倍。该分解使用标准 PyTorch 实现。

::: block-1
结果：1.944 ms → 1.389 ms（下降 28.5%，benchmark 设置见附录 1）。原始 batched GEMM 的算术强度约为 356 FLOPs/Byte，低于 H100 的约 495 FLOPs/Byte 机器平衡点（推导见附录 2），**分解后的两个 GEMM 也仍受内存限制，因此加速来自内存成本下降**。去重将内存成本削减了一半以上——因为用户侧 GEMM（`B_user ≈ 15`，而非 `B = 1024`）几乎可以忽略。
:::

注意，这里的分解只是把广播移到了 matmul 之后：不再在 GEMM 之前复制完整 K 维输入嵌入，而只广播较小的压缩结果，因此便宜得多。第 2.3 节会进一步通过内核内广播融合，把这部分剩余广播也完全消掉。

当前瓶颈是 L1/TEX pipeline 利用率（84%），而不是 DRAM 利用率——这种失衡很可疑，我们将在下一节详细分析。更细的 profiling 结果见附录 3。

### 2.2 内存布局优化

> 对分解后 GEMM 的详细结果分析显示出一种失衡：L1/TEX 达到了峰值的 84%，而 DRAM 只有 19%，**说明内存加载宽度不必要地过窄**。SASS 证实：每条 `cp.async` 仅复制 4 字节，而不是一次 128-bit 加载。

```
LDGSTS.E.LTC128B P0, [R203],      [R38.64]       // 4 bytesLDGSTS.E.LTC128B P1, [R203+0x4],  [R38.64+0x4]   // 4 bytes  (总共 ×4，也只有 16B)
```

`cp.async` 的宽度受源指针天然对齐约束。矩阵 A 为 `(M, K)` 的 row-major，步幅是 `K × 2` 字节，因此当 K 不是 8 的倍数时，这个 stride 就会破坏 128-bit 对齐。

::: block-1
模型—内核协同设计洞察：内存对齐本身是经典 GPU 优化问题，但分解让它变成了模型—内核协同设计挑战。K 来自多个 embedding tensor 的 `torch.cat`，而这些 tensor 的大小取决于许多模型配置因素。**做完分解后，几乎无法再靠人工微调这些因素，来保证分解后的 embedding 仍保持完美倍数对齐，因此需要系统化方案。**
:::

解决方案：将每个分解后的 K 通过补零 pad 到下一个 8 的倍数。**我们证明这在前向与反向传播中都与原式数学等价（见下方证明 1），并且借助 ML 编译器的内存规划器，这只相当于一次低成本常量拷贝。**

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/04.webp)

::: image-caption
公式描述了矩阵乘法中的填充（Padding）技术，其目的是将矩阵维度调整为硬件友好的倍数（此处为8的倍数），以优化计算效率，同时保证前向与反向传播的数学正确性。在设置中，矩阵乘法 $C = AB$ 里，$A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{\text{batch} \times K \times N}$，$C \in \mathbb{R}^{\text{batch} \times M \times N}$。填充时，为 $A$ 补 $p$ 个零列、为每个批次的 $B^{(b)}$ 补 $p$ 个零行，使新维度 $K' = K+p$ 成为8的倍数。前向传播中，$A_{\text{pad}}B_{\text{pad}}^{(b)}$ 的结果与原矩阵乘法 $AB^{(b)}$ 完全一致，零填充不影响计算结果。反向传播时，损失对 $A_{\text{pad}}$ 的梯度仅前 $K$ 列有效，其余零列梯度为0；损失对 $B_{\text{pad}}^{(b)}$ 的梯度仅前 $K$ 行有效，其余零行梯度为0，二者均能正确还原原矩阵的梯度。这种填充方法在不破坏模型数学逻辑的前提下，通过适配硬件计算单元的对齐要求，大幅提升了矩阵乘法在GPU/加速器上的执行效率，是高性能算子优化中兼顾正确性与性能的典型设计。
:::

> Proof 1：对 K 进行零填充，在前向与反向传播中都保持精确数值等价。

::: block-1
结果：1.389 ms → 0.798 ms（下降 42.5%）。**Padding 后，CUTLASS 可以选择基于 TMA 的内核，从而彻底绕过 L1/TEX（sectors 从 351M 变为 0），并将 GEMM 延迟从 0.984 ms 降到 0.400 ms**。GEMM 解决后，未融合的广播与加法（0.398 ms）现在占到总延迟的一半——下一节继续处理。详细分析见附录 5。
:::

### 2.3 候选 GEMM 的内核内广播融合

> 未融合的广播与加法是内存受限的：先把 candidate GEMM 结果写回 HBM，再读回它以及用户结果，相加，再写一次。我们把广播融合进 candidate GEMM 的 epilogue，从而消除这一过程（图 3）。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/05.webp)

::: image-caption
图 3. 核内广播融合：GEMM尾声通过索引查找加载预先计算好的用户结果，并在寄存器内完成加法。该图揭示了 Kernel 融合的核心机制，即在每个计算瓦片完成后，立即在寄存器层面完成加法，而不是将完整矩阵写回显存后再进行。这避免了中间结果在整个HBM上的往返，是实现节省0.87 GB DRAM传输量的关键。
:::

每个 tile 完成累积后，epilogue 会查找用户索引，加载预先计算好的用户结果，在寄存器中相加，再直接写出最终结果——中间张量不会被物化。我们用 Triton 内核实现它：本质上是一个标准 batched GEMM，加上自定义的 post-accumulation epilogue block。

::: block-1
结果：0.798 ms → 0.580 ms（下降 27.4%）。**融合消除了 0.87 GB 的中间 DRAM 流量**，因此获得延迟收益。然而 occupancy 只有 6.25%（每个 scheduler 仅 1 个 warp），使所有 stall 都被完整暴露。除了有 42% 以上周期等待 global load，还有 20% 周期在等 WGMMA——这些 stall 无法被 epilogue 隐藏；而且没有持久化时，也不存在下一个 tile 的加载可与之重叠。**这是个棘手权衡：为了喂饱 tensor core，需要较大 tile 与较深 pipeline，但它们会占掉大部分 shared memory 预算，导致几乎没有余量通过提升 occupancy 来隐藏延迟**。详见附录 6。
:::

### 2.4 基于 TLX 的 warp-specialized 多阶段融合

> TLX（Triton Low-level Language Extensions）[4] 在保留 Triton Python DSL 与 autotuning 基础设施的同时，暴露了 Hopper 的 warp specialization、TMA、mbarrier 与 named barrier。

借助 TLX，我们用 warp specialization 来解决 2.3 节中的 occupancy 限制——**不是通过增加更多 warp，而是通过功能分工来隐藏延迟。**

2.1–2.3 节把原始 LCE 分解成了两个独立计算：用户 GEMM（阶段 1）与带融合 broadcast-add epilogue 的候选 GEMM（阶段 2）。我们先优化占主导瓶颈的阶段 2 中的延迟隐藏，再把两个阶段融合成单一 persistent kernel。

#### 阶段内延迟重叠

> candidate IKBO 内核是内存受限的——设计目标是让内存流水线持续被喂满。Triton 的软件流水线（2.3 节）已经让 Load 与 WGMMA 重叠，但 epilogue 仍然是串行的——它会阻塞后续 Load，并暴露 WGMMA 等待 stall。:mdnice-bold-italic[我们通过把每个 CTA 划分为专门的 warp group 来解决：一个专用 producer 持续发出 TMA load（重叠 1，类似 Triton 原有软件流水线），两个 consumer 采用 ping-pong tile 方式，使一方的 epilogue 与另一方的 WGMMA 重叠（重叠 2）]。配合 persistence，tile 会连续流动，不会出现跨 tile 空隙。见图 4。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/06.webp)

::: image-caption
图 4. 候选IKBO内核结构，展示了两种阶段内延迟重叠以及warp group的角色分配。该图展示了从一维流水线到二维乒乓流水线的升级。该设计巧妙地解决了2.3节中发现的占用率与流水线深度的矛盾。通过warp特化，它不再依赖增加更多线程块来提高占用率，而是在单个CTA内部创建了三个专用且协作的硬件调度单元，实现了对延迟的精细掩盖。
:::

#### 多阶段融合

我们把用户 IKBO（阶段 1）与候选 IKBO（阶段 2）融合为一个 mega-kernel，以减少 wave quantization、消除 kernel launch 开销，并提高 L2 cache 利用率。较高的 candidate-to-user 比值会放大阶段 1 的 wave quantization 问题。由于 candidate GEMM 在 epilogue 之前并不依赖用户结果，我们可以并发调度两个阶段。

> 这种并发调度又解锁了两个跨阶段重叠，使总重叠数达到四个，见图 5。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/07.webp)

::: image-caption
图 5. 并发阶段调度：没有用户 tile 的SM立即进入阶段2，与阶段1的部分波重叠。多阶段融合后的所有四种延迟重叠，展示了阶段内（#1，#2）和跨阶段（#3，#4）的重叠机会。SM 0-49，50-131仅为示例数字。该图揭示了跨阶段重叠如何从根本上解决了因批次维度不匹配导致的计算资源闲置问题。它打破了传统串行执行两个算子的壁垒，让系统全局地调度两个计算任务，这对于提升整个端到端流水线的硬件利用率至关重要。
:::

#### Warp Group 专门化与同步设置

> :mdnice-bold-italic[为了实现这四种重叠，每个 CTA 被划分为一个 producer 和两个 consumer warp group。关键点在于，两个阶段共享同一套 circular buffer 与 `mbarrier` 基础设施] ——在阶段边界处不会清空 pipeline，也不需要重新初始化 barrier。最后一个 user K-block 与第一个 candidate K-block 会同时处于不同 buffer slot 中。见图 6。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/08.webp)

::: image-caption
图 6. 每个CTA的warp组设置以及三种同步机制。这张图展示了 GPU 算子中典型的生产者 - 消费者协同优化架构，以线程块（CTA）为单位，每个 CTA 包含 1 个生产者线程束组和 2 个消费者线程束组，通过共享内存（SMEM）中的环形缓冲区实现数据的分阶段读写与流水线并行，提升 Tensor Core 利用率。生产者负责分阶段加载数据并写入缓冲区，消费者则以乒乓模式或协作模式读取数据并执行计算，通过命名屏障（mbarrier）实现同 CTA 内的同步。跨 CTA 的依赖则通过全局内存（GMEM）中的依赖计数器，结合原子加法和轮询机制完成同步，确保中间结果和最终结果的正确共享。这种设计通过解耦数据加载与计算、重叠流水线阶段，有效隐藏访存延迟，提升算子整体吞吐与硬件利用率。
:::

#### 双向阶段交替 tile 调度

> 当两个阶段的 tile 数都不能被 SM 数整除时，朴素的单向 dispatch 会导致负载不均。我们在两个阶段之间反转 tile 分配方向：阶段 1 从 `pid` 开始，阶段 2 从 `NUM_SM - 1 - pid` 开始，见图 7。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/09.webp)

::: image-caption
图 7. 单向（左）与双向的阶段交替调度（右），平衡了部分波中每个SM的工作负载。该图对比了两种GPU调度策略：左侧的朴素单向调度因两阶段任务都按升序分配瓦片，导致部分SM（如SM 0/1）承担过多工作，而SM 131在Stage 1无任务，Stage 2仅处理1个瓦片，负载严重不均，出现“Bad balance”；右侧的双向交替调度则让Stage 1按升序分配瓦片，Stage 2按降序分配瓦片，使每个SM在两阶段中各承担1个瓦片，Wave 1阶段所有SM都恰好处理2个瓦片，实现了“Good balance”。这种交替调度通过反转任务分配顺序，避免了热点SM的过载与空闲，让各SM的计算负载更均匀，从而提升了整体硬件利用率与算子性能。
:::

#### tile 粒度的跨 CTA 同步

> 用户 tile 与候选 tile 可能运行在不同 CTA 上，因此需要跨 CTA 同步——但设备级 barrier 会把所有工作串行化，完全破坏这些重叠。我们采用每个 tile 粒度的三步 release-acquire 协议：

1. 每个 warp group 只让一个线程用 `ld.relaxed` 轮询 tile flag，以最小化内存流量。
2. 一旦 flag 被置位，用单条 `ld.acquire` 建立 happens-before 关系。
3. 用 named barrier 把就绪状态广播给该 warp group 的全部 128 个线程。

这避免了轮询时昂贵的 fence，也让处理不同 user tile 的 candidate CTA 能完全独立推进。详见附录 7。

#### 结果

> 所有优化叠加后，延迟从 0.580 ms 降到 0.482 ms（下降 16.9%）。清晰的 intra-warp Proton tracer[5] 时间线表明，这四种重叠在实际运行中都得到了实现。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/10.webp)

::: image-caption
图 8. 两个CTA的Proton性能分析时间线，四种重叠用不同颜色标出。内存流水线持续保持满载。这张图是 IKBO 算子的 GPU timeline，展示了四层流水线并行优化：Overlap #1 通过同一 SM 内的数据加载（Load）与矩阵乘（WG MMA）重叠，隐藏访存延迟；Overlap #2 在同一阶段的 WG MMA 与后处理（Epilogue）间实现并行；Overlap #3 则让不同阶段的计算在同一 SM 上流水执行；Overlap #4 更进一步，让用户特征与候选特征的计算跨 SM 并行。通过生产者 - 消费者线程束的协作与乒乓缓冲机制，算子实现了数据加载、计算与结果处理的全流程重叠，充分利用了 GPU 的多线程与多 SM 并行能力，有效提升了硬件利用率与整体吞吐。
:::

**主要收益来自重叠 2：ping-pong consumer 在每个 tile 上都隐藏了 WGMMA 与 epilogue stall，直接解决了 2.3 节中的主要浪费周期**。重叠 1（Load↔WGMMA）继承自 Triton 现有软件流水线。重叠 3 与 4 则隐藏了 user-to-candidate 阶段切换时的空闲时间。见图 8。

::: block-1
NCU 证实：occupancy 从 6.25% 提升到 18.75%（3 个 warp group 对比 1 个），DRAM 吞吐从 39% 提升到 52%，而作为瓶颈的 L2 吞吐从 74% 提升到 84% 的峰值。这不仅仅是 occupancy 提升：跨四种重叠的积极延迟隐藏让内存流水线持续饱和，才推动 L2 超过 80%。更详细的 NCU 指标见附录 8。
:::

我们在不同批大小与不同 candidate-to-user 比值下进行了基准测试，默认设置为 batch=1024、ratio=70。见图 9。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/11.webp)

::: image-caption
图 9. 跨批次大小（左，比率 $= 70$）和候选-用户比率（右，批次 $= 1024$）的累积IKBO加速效果。这张柱状图展示了 IKBO 线性压缩算子（LCE）在多阶段优化下，相比 PyTorch Eager 基线的吞吐加速比表现。左图在用户 - 候选比例固定为 70 的条件下，测试不同批量大小（512~3072）的性能；右图则在批量大小固定为 1024 时，测试不同用户 - 候选比例（2~1000）的性能。从分解（Stage1）、K 维对齐（Stage2）、内核融合（Stage3）到 TLX 线程束专用优化（Stage4），每一步优化都叠加了前序成果，最终实现约 4 倍的整体加速。数据显示，随着批量增大或用户 - 候选比例升高，优化收益愈发显著，在高比例场景下 Stage4 的加速比稳定突破 4 倍，证明 IKBO 的优化设计能有效适配推荐系统典型负载，大幅提升算子吞吐性能。
:::

IKBO 融合在不同场景下都表现出稳定收益：在不同批大小与不同 candidate-to-user 比值下，都能获得约 4 倍加速。**即使在较低比值下，这个内核仍有可观提升。**

## 三、内核深入案例二：IKBO Flash Attention

> 随着推荐模型不断扩展以捕获更丰富的用户序列行为，序列架构——包括注意力——已经成为关键计算瓶颈：在序列长度为 1K 时，其大约占据 40% 的推理延迟。因此，我们将重点放在结合推荐系统独特 batching 语义进行协同设计的、IKBO-aware Flash Attention 上。

受 Transformer 与 Set Transformer [7, 8] 启发，推荐系统中广泛采用两类基本的用户历史交互模块：

- target attention（类似 cross-attention），用于捕获预测候选与用户历史行为之间的关系。
- self-attention，用于建模用户历史序列内部的依赖。

由于用户历史是 RO 特征，而 target 则作用于独立的 candidate（NRO）批维度，**这种结构上的不对称性给 IKBO 提供了提升模型可扩展性与计算效率的机会**。我们主要优化 target attention；而只需少量协同设计，就可以在 3.3 节中把 self-attention 也融合进 IKBO target attention。由于我们的模型是 encoder-driven，因此使用完整注意力而非 causal masking。

::: block-1
最终的、利用端到端协同设计优化后的 target attention 版本，**相比未做协同设计的 CuTeDSL FA4-Hopper，可实现 2.4×/6.4× 的吞吐** （仅注意力内核 / 注意力内核加广播成本），并分别降低 0.320 ms / 1.232 ms 延迟（如后文表 2）。
:::

### 3.1 IKBO Flash Attention 在推荐系统边界条件下解决 IO 受限问题

> IKBO 把 K/V 广播融合进注意力内核中，并借助推理运行时提供的 candidate-user 映射张量保持数学等价性，从而处理非均匀的 candidate-to-user 比值。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/12.webp)

::: image-caption
图 10：带 candidate-user 广播的传统 SDPA（左）与融合后的 IKBO target attention（右）。对比直观展示了IKBO方案如何重塑数据流。传统路径中，KV矩阵显式膨胀了数十倍，这对内存容量和带宽是巨大考验。IKBO路径则通过一个映射张量，将索引操作直接注入内核，实现了“虚拟广播”。这种转变的核心在于，它将高带宽需求的数据（KV）的访问次数降低了数十倍，而代价仅仅是增加一次轻量级的索引计算。
:::

上图 10 对比了两种方式：

- 传统 SDPA 路径在注意力前把 K 和 V 广播到完整 candidate 批大小，
- 而 IKBO 路径则完全消除了这一步物化——每个候选在运行时按需索引其所属用户的 K/V。

#### 通过 IKBO 协同设计，把 IO 受限转为计算受限

> 在推荐系统边界条件下，相比用户浏览历史，target attention 只使用相对较少的 candidate embedding 来表示候选属性。对标准注意力做 roofline 分析可知，其算术强度约为 60 FLOPs/Byte，远低于 H100（SXM5 HBM2e 版本）约 495 FLOPs/Byte 的峰值（附录 2），因此即使是标准 Flash Attention 也严重受 IO 限制。**IKBO 通过让共享同一用户上下文的多个候选摊销 K/V 内存访问，把算术强度从约 60 FLOPs/Byte 提升到约 833 FLOPs/Byte（当 `B_candidate : B_user = 70:1` 时），将内核明确推入计算受限区间。**

为了最大化这一收益，我们将 threadblock launch grid 重新排序，让 `batch_size_candidate` 先于 `num_heads`。这样可确保那些处理不同候选、却共享同一用户 K/V 的 threadblock 被并发调度，从而提高 L2 cache 复用。

| Grid dimension | Flash attention (SDPA) | IKBO target attention |
| --- | --- | --- |
| x | num_q_seq_block | num_q_seq_block |
| y | num_heads | batch_size_candidate |
| z | batch_size_candidate | num_heads |

*表 1：launch grid 配置对比展示了：SDPA 把 `num_heads` 放在 `grid.y` 以优先优化 GQA；IKBO 则交换了 head 和 candidate 维度，把 `batch_size_candidate` 放到 `grid.y`，以便在不同候选间高效共享 K/V。*

表 2 比较了我们的 IKBO Triton 实现（FA2 逻辑 + IKBO）与 Hopper 上最先进的 Flash Attention 实现（未做 IKBO 协同设计）。吞吐与 IO 只测量注意力部分；Key 与 Value 的广播延迟甚至比注意力本身还高。

| Kernel | Throughput (TFLOPs/s) | IO (GB/s) | Latency (ms) |
| --- | --- | --- | --- |
| Triton IKBO FA2 | 425 | 487 | 0.321（broadcast fused） |
| TLX FA3 | 245 | 2152 | 0.561 + 0.912（broadcast K&V） |
| CuTeDSL FA4 Hopper | 250 | 2193 | 0.550 + 0.912（broadcast K&V） |
| TLX IKBO FA3 persistence generalized | 594 | 681 | 0.230（broadcast fused） |

*表 2：推荐系统边界条件下的注意力内核对比（`B_candidate = 2048`，`B_u = 32`，candidate-to-user 比值均匀）。在未做协同设计的情况下，即使是前沿 Hopper 实现也依然受 IO 限制。*

### 3.2 在 TLX 上把现代内核技术（FA3、FA4）与 IKBO 结合

> 当 IKBO 把内核从 IO 受限转为计算受限后，自然的下一步就是采用 Flash Attention 3（FA3 [10]）与 Flash Attention 4（FA4 [11]）在 Hopper 上最先进的计算优化——尤其是 warp specialization 与流水线技术。然而，我们在 query embedding 数量上的边界条件（`q_seq = 32` 或 `64`）使得难以直接采用 FA3 的 ping-pong 或 cooperative warp specialization。

Hopper 上的 warp specialization 需要异步 WGMMA 指令，而这要求 `BLOCK_M ≥ 64`。同时，为了尽量减少两个 consumer warp group 之间的 bubbles，还需要两个 consumer warp group。为满足这些约束，我们定制了内核，让单个 threadblock 同时发射 `B_candidate = i` 与 `B_candidate = i + 1`，共享同一个 `B_user`。下面讨论中，我们假设所有用户都对偶数个候选进行排序，且 `q_seq = 64`；奇数候选的处理随后说明。

#### IKBO FA3 内核的性能提升

> 从 FA3 的方案出发——阶段内流水线、warpgroup specialization 与 ping-pong 调度——最初的 TLX IKBO FA3 内核性能与 FA2 基线大致相当（图 12，蓝线对红线，详见附录 11），吞吐几乎持平。

为了诊断瓶颈，我们使用 Proton tracer[6] 以 GPU cycle 作为延迟单位，对 intra-warp pipeline 进行了可视化（图 11）。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/13.webp)

::: image-caption
图 11：基于 Proton 的 TLX IKBO FA3 内核 intra-warp profiling。图中给出了每个 warp group 的代表 warp：warp 0（producer）、warp 4（consumer 1）与 warp 8（consumer 2）。softmax_PV_overlap 与纯 softmax 区域被单独标出，以识别 tensor core bubbles。(A) persistence 前对 B 的放大视图；(B) persistence 前，两波次；(C) persistence 后，两波次。该图通过对比生动地展示了持久化的效果。持久化前（B），两个波次之间明显的黑色间隙就是跨CTA的等待时间。持久化后（C），间隙消失，生产者warp（warp 0）的连续忙碌状态表明流水线几乎没有中断。这不仅是性能的提升，更是对硬件利用率的极致挖掘。
:::

> 下表 3 汇总了 persistence 优化前后通过 Proton tracer 测得的主要瓶颈。

| Bottleneck | Before | After | Key change |
| --- | --- | --- | --- |
| Tensor Core Bubbles（每个 wave 的第一个 QKT，蓝色） | ~1,300 cycles（其中 400 cycles 来自 warp scheduler 切换） | ~1,300 cycles | 无变化 |
| Tensor Core Bubbles（每个 wave 的最后一个 PV，蓝色） | ~2,000 cycles | ~300 cycles | Async TMA store + reciprocal overlap with last PV |
| Cross-CTA Stalls（橙色） | ~14,000 cycles | 消除 | Persistence 完全消除 CTA re-launch |
| Init Buffers & Barriers（绿色） | ~1,600 cycles/wave | ~1,600 cycles（仅首个 wave） | Persistence 将共享 buffer 与 barrier 的成本摊到多个 wave |
| Wait 1st Q/K Load（深紫色） | 2,100~4,000 cycles/wave（长度随 HBM 带宽竞争变化） | ~2,000 cycles（仅首个 wave） | 跨 wave 流水线；producer 提前约 3K cycles 预取 |

*表 3：persistence 及优化前后的关键瓶颈。*

::: block-1
**关键结论是：在这种较小 query sequence 长度下，主导瓶颈是 cross-CTA stall，而不是 tensor core 利用率**。要获得提升，persistence 是必要条件。引入 persistence 后，其 profiling 结果与延迟变化见图 11C 与表 3。
:::

#### 针对 HBM2e 的特定优化

> 随后我们又针对 H100 SXM5 的 HBM2e 带宽约束对 persistent kernel 做了进一步调优，用更多 shared memory capacity 去换取更少的 load/store 阻塞（表 4）。

| Customized optimization/fix | Benefit |
| --- | --- |
| 将 O 的 SMEM buffer 与 Q/V 解耦，并配合流水化 TMA async store | O 不再与 Q/V 共享 SMEM，使 TMA async store 可与下一 wave 计算重叠，把 store blocking 从每 wave 1,300 cycles 缩短到 400 cycles |
| 分离 Q₀ 与 Q₁ buffer | 降低每个 Q 的加载时间，使一个 consumer group 可更早启动——当 wave 数远大于 K/V sequence iteration 数时尤其有利（推荐系统中很常见） |
| 修复 instruction cache miss | 将被 peeled-out 的最后一次迭代代码路径并回主循环，消除大量 warp-specialized 指令导致的 icache thrashing（附录 12） |

*表 4：面向 HBM2e H100 SXM5 的定制优化。这些优化在推荐系统边界条件下仍能装入可用的 SMEM 预算（附录 10）。*

> 我们还实现了 persistent V2，它从 K 序列末尾向前迭代（与 FA3/FA4-Hopper 方案一致），以简化 masking 逻辑。两种 persistent 变体都应用了表 4 中的优化。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/14.webp)

::: image-caption
图 12：不同序列长度下的 IKBO 实现吞吐（`B_candidate = 2048`；`B_candidate : B_user = 64`；`num_head = 2`；`d_head = 128`）。实际推荐系统序列长度通常低于 4K [3]；更长序列仅用于与 LLM 场景对比。generalized 版本支持每用户候选数不为偶数的情况，其中每用户奇数候选概率为 50%。
:::

上图 12 显示，在较短序列长度（512–4096）下，TLX FA3 persistent kernel 优于其他候选方案；当长度超过 8K 时，两种 persistent 变体趋于一致。

#### 将 IKBO FA3 泛化到任意候选批大小排序场景

> 我们的 IKBO FA3 内核为了满足 WGMMA 的 `BLOCK_M ≥ 64` 要求，每个 CTA 会同时处理两个 candidate batch。当某个用户只有奇数个候选时，其中一个 consumer warpgroup 就没有成对伙伴。对此我们加入了 idle 逻辑（详见图 13 左以及算法 1）：

- idle 的 warpgroup 通过 mbarrier signaling 排空 K/V buffer，防止 producer 死锁。
- active 的 warpgroup 关闭 ping-pong 同步（因为其配对方不会再到达 named barrier）。

在约 70:1 的 candidate-to-user 比值下，idle 路径触发概率低于 0.7%，开销可以忽略不计（图 12，IKBO TLX FA3 generalized）。同样思路还可推广到 `q_seq_len = 32`，此时每个 CTA 绑定四个 candidate batch，并使用相应的 idle 与 masking 逻辑。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/15.webp)

::: image-caption
图 13：泛化 target attention 的 CTA 分配（左）以及 self + target attention 融合（右）。每个 CTA 分配两个共享同一用户 K/V 的 consumer warp group。当候选数为奇数时，第 2 个 consumer 进入 idle 并排空 barrier。
:::

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/16.webp)

::: image-caption
算法 1：带奇数候选处理的 IKBO Attention 前向过程。该算法是IKBO Attention前向传播中针对奇/偶候选的优化流程，以CTA为单位，输入候选-用户批次映射张量M与起始索引i：首先加载相邻候选的用户批次标识 $B_{\text{user0}}$、$B_{\text{user1}}$，判断是否为同一用户（`same_user`）；生产者warpgroup优先通过TMA加载 $Q_0$、$K_0$，若同用户则额外加载 $Q_1$，并流水线加载后续 $K_j$、$V_l$；消费者warpgroup1处理候选 $i$，同用户时启用命名屏障实现乒乓计算，否则禁用屏障；消费者warpgroup2处理候选 $i+1$，同用户时启用屏障计算，否则跳过计算并等待流水线`mbar`同步。该设计通过预判断用户连续性，优化TMA加载与屏障同步，减少分支开销，实现候选并行处理，提升注意力前向的流水线效率。
:::

### 3.3 通过模型协同设计融合 Self + Target Attention

> 前面几节集中优化的是 target（cross）attention。一个自然问题是：能否把 self-attention 折叠进同一个内核？

:mdnice-bold-italic[关键洞察在于，这两种注意力共享同一份 key-value 来源——用户序列。唯一差异在于 query：self-attention 的 query 来自用户侧，而 target-attention 的 query 来自候选侧]。通过让二者共享 K/V 投影，我们就能在单次 launch 中实现直接的横向内核融合。图 13（右）展示了融合后的 CTA 布局：前一部分 CTA 处理 self-attention 的 query block，剩余 CTA 处理 target-attention 的 candidate pair——它们都从同一条流水化 K/V 流中读取数据。

类似的协同设计思想也曾在 XAI Phoenix——X 开源推荐系统 [4]——中出现。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/17.webp)

::: image-caption
图 13：泛化 target attention 的 CTA 分配（左）以及 self + target attention 融合（右）。每个 CTA 分配两个共享同一用户 K/V 的 consumer warp group。当候选数为奇数时，第 2 个 consumer 进入 idle 并排空 barrier。
:::

我们实现了一个原型 fused kernel，用于量化融合收益，且不计入 K/V projection 节省（图 13 右）：

- `seq_len = 512`：提升 6.6%（514 对 482 TFLOPs/s）
- `seq_len = 1024`：提升 4.1%（581 对 558 TFLOPs/s）
- `seq_len = 2048`：提升 0.3%（612 对 610 TFLOPs/s）——此时 self-attention 已让 SM 饱和

::: block-1
短序列下的收益来自内核融合本身：更少的 launch 开销、共享 buffer 分配节省、跨内核流水线机会，以及缓解 wave quantization——这些也是 megakernel 技术 [12] 在 LLM 推理中所针对的问题。在生产环境中，共享 K/V projection 还会进一步节省线性投影成本，类似于 KV cache 复用。
:::

## 四、基准测试与结果汇总

> 下面汇总本文给出的内核级 benchmark 以及端到端部署结果。以下所有内核 benchmark 均在 H100 SXM5 上完成（细节见附录 1）。

- Linear Compression（第 2 节）：四个渐进式协同设计阶段——矩阵乘法分解、内存对齐、广播融合，以及通过 TLX 进行的 warp-specialized 多阶段融合——在代表性设置下带来累计约 4× 加速（1.944 ms → 0.482 ms）。这种收益在不同批大小与 candidate-to-user 比值下都保持稳健（图 9）。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/18.webp)

::: image-caption
图 9. 跨批次大小（左，比率 $= 70$）和候选-用户比率（右，批次 $= 1024$）的累积IKBO加速效果。这张柱状图展示了 IKBO 线性压缩算子（LCE）在多阶段优化下，相比 PyTorch Eager 基线的吞吐加速比表现。左图在用户 - 候选比例固定为 70 的条件下，测试不同批量大小（512~3072）的性能；右图则在批量大小固定为 1024 时，测试不同用户 - 候选比例（2~1000）的性能。从分解（Stage1）、K 维对齐（Stage2）、内核融合（Stage3）到 TLX 线程束专用优化（Stage4），每一步优化都叠加了前序成果，最终实现约 4 倍的整体加速。数据显示，随着批量增大或用户 - 候选比例升高，优化收益愈发显著，在高比例场景下 Stage4 的加速比稳定突破 4 倍，证明 IKBO 的优化设计能有效适配推荐系统典型负载，大幅提升算子吞吐性能。
:::

- Flash Attention（第 3 节）：IKBO 将 target attention 从 IO 受限（约 60 FLOPs/Byte）转为计算受限（约 833 FLOPs/Byte），相较未做协同设计的 CuTeDSL FA4-Hopper，可实现 2.4×/6.4× 吞吐（仅内核 / 内核加广播），并达到 621 BF16 TFLOPs。
- 端到端部署：IKBO 已广泛部署在 Meta 推荐系统推理栈中，从早期排序到晚期排序，覆盖 GPU 与 MTIA 加速器；在协同设计模型上，计算密集型网络延迟最多可降低三分之二。IKBO 还在约 `10,000:1` 到 `10:1` 的 candidate-to-user 广播比值范围内得到验证，证明了其在不同工作负载下的数值稳定性与可扩展性。

## 五、结论与未来方向

> IKBO 说明：广播——长期以来被视为用户—候选交互中不可避免的成本——可以通过内核—模型—系统协同设计，在计算原语层面被消除。**通过把广播语义直接编码进内核，复制张量不再被物化，而收益会自然随 candidate-to-user 比值放大。**

尽管本文中的内核实现是面向 NVIDIA Hopper、基于 Triton 与 TLX 的，但**其核心思想——用索引驱动的内核内查找取代物化广播——并不依赖特定硬件厂商**。下一步的自然方向包括：把 IKBO 内核适配到 CuTeDSL（以获得更先进的 NVIDIA 后端支持），以及补全 AMD CK 支持。

::: block-1
除了本文讨论的双层 user-candidate 层级外，:mdnice-bold-italic[一些推荐场景还会涉及更深层级，例如 user → ads vendor → ads item，也就是一个用户看到多个商家，每个商家又提供多个条目。这会引入两个嵌套的广播关系，而且比值彼此独立、且不均匀。IKBO 也能优雅处理这种结构]，将其用于多层级工作负载，是进一步降低生产推荐系统中物化开销的自然方向。
:::

## 参考文献

[1] Naumov, M., et al. “Deep Learning Recommendation Model for Personalization and Recommendation Systems,” arXiv:1906.00091, 2019.

[2] Wang, R., et al. “Deep & Cross Network for Ad Click Predictions,” ADKDD, 2017.

[3] Zhai, J., et al. “Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,” ICML, 2024.

[4] xAI. “Phoenix: Recommendation System,” GitHub, 2026. https://github.com/xai-org/x-algorithm

[5] Guo, L., et al. “Request-Only Optimization for Recommendation Systems,” arXiv:2508.05640, 2025.

[6] Zhang, B., et al. “Wukong: Towards a Scaling Law for Large-Scale Recommendation,” ICML, 2024.

[7] Vaswani, A., et al. “Attention Is All You Need,” NeurIPS, 2017.

[8] Lee, J., et al. “Set Transformer: A Framework for Attention-based Permutation-Invariant Input,” ICML, 2019.

[9] Dao, T. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning,” ICLR, 2024.

[10] Shah, J., et al. “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision,” NeurIPS, 2024.

[11] Zadouri, T., et al. “FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling,” arXiv:2603.05451, 2026.

[12] Spector, B., et al. “Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B,” Hazy Research Blog, 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

## 附录

### 附录 1：Benchmark 设置

所有实验均在单张 NVIDIA H100 SXM5 GPU（700W TDP，96GB HBM2e）上完成，软件栈如下：

- CUDA: 12.4
- PyTorch: 2.11.0a0+fb（内部构建）
- Triton: facebookexperimental/triton@`4059e79bf`（#831[7]）

### 附录 2：算术强度分析

#### 2.1 H100 SXM5（700W TDP，96GB HBM2E）的机器平衡点

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/19.webp)

::: image-caption
计算 H100 GPU 硬件算术强度（AI，FLOPs/Byte 的公式，是判断算子性能瓶颈的核心基准。分子989.5 TFLOPs/sec是 H100 FP16 张量核心的理论峰值浮点算力，分母2 TB/sec是其显存的理论峰值带宽，二者相除得到硬件算术强度为495 FLOPs/Byte，代表每传输 1 字节数据，硬件理论上可完成 495 次浮点运算。 该指标定义了硬件的 “算力 - 访存平衡线”：若算子的算术强度低于此值，说明算子访存需求相对更高，易受带宽限制；若高于此值，则算力会成为瓶颈，可针对性优化访存或计算调度，提升硬件利用率。
:::

#### 2.2 基线 LCE 的算术强度

对于 FP16 下的 batched matmul `(M, K) @ (B, K, N) → (B, M, N)`，当 `B=1024, M=433, K=2044, N=256` 时：

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/20.webp)

::: image-caption
矩阵乘法（GEMM）算子的算术强度（AI，FLOPs/Byte）计算公式，用于量化算子的计算访存比，判断其性能瓶颈类型。分子`2·B·M·K·N`是GEMM的总浮点运算量，其中`2`表示乘加操作的2次FLOP/元素，`B`为批大小，`M/K/N`为矩阵维度；分母`(M·K + B·K·N + B·M·N)×2`是算子的总访存字节数，三项分别对应输入矩阵A、输入矩阵B、输出矩阵C的读写数据量，`×2`是FP16数据的字节宽度。计算结果`356 FLOPs/Byte`表示算子每访问1字节数据可执行356次浮点运算，该值越高，算子越偏向计算密集型，越能充分利用GPU算力，反之则易受内存带宽限制，此指标是性能优化中判断算子瓶颈、指导优化方向的核心依据。
:::

### 附录 3：第 2.1 节的详细结果分析

> 设置：H100 SXM5（附录 1），PyTorch eager mode（无 kernel fusion），inference。形状取自代表性配置。

| Version | Total (ms) | Kernels | Latency (ms) | DRAM (GB) | L1/TEX Sectors (M) | Compute (GFLOPs)* | Bottleneck† |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1.944 | 1 CUTLASS GEMM | 1.944 | 1.31 | 798 | 460 | L1/TEX (89%) |
| Decomposition | 1.389 | 2 CUTLASS GEMM（user + candidate matmul） | 0.984 | 0.68 | 351 | 200 | L1/TEX (84%) |
|  |  | 1 ATen Gather + 1 ATen add | 0.405 | 0.87 | 36 | 0.11 | DRAM (92%) |

* 执行的总 FLOPs，不是吞吐。† 瓶颈由 NCU Speed of Light 分析识别；方法见附录 4。

去重消除了 98% 以上的用户侧工作量（batch 1024 → 约 15），将 L1/TEX sectors 从 798M 降到 351M，并把 GEMM 延迟从 1.944 ms 降到 0.984 ms。GEMM 后的广播与加法耗时 0.405 ms（DRAM 受限），因此净节省为 0.555 ms。

::: block-1
精度说明：基线在单个 FP32/TF32 reduction 中累加所有 K 乘积。分解后会分别累加 `K_user` 与 `K_cand`，再在 BF16/FP16 中把部分结果相加。**训练使用同样分解，因此端到端数值是一致的。若推理需要与原始实现完全一致，第 2.4 节中的 fused kernel 可在最终求和时使用 FP32。**
:::

### 附录 4：瓶颈分析方法

> 在 roofline 分析之后，我们使用 NCU 的 Speed of Light 分析来识别硬件子系统瓶颈。瓶颈被定义为：相对其峰值持续吞吐率利用率最高的子系统。针对 2.1 节分析，我们监控三类指标：

- Compute：峰值 SM pipeline 利用率，由 NCU 直接报告（`Compute (SM) Throughput`）。它衡量最活跃执行流水线（GEMM 中通常是 tensor core）相对峰值指令速率的繁忙程度。
- L1/TEX 利用率：由 L1/TEX 单元需要处理的总 sector 数推导，其中 `num_L1_tex_sectors` 对应 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 与 `_st.sum` 计数器，`SM_active_cycles` 对应 `sm__cycles_active.avg`，`num_SM` 为 132，而 `num_sustained_peak_sectors_per_sm_per_cycle` 在 H100 上为 2.0。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/21.webp)

::: image-caption
GPU L1纹理缓存（TEX）利用率的性能分析公式，核心是量化算子对纹理缓存资源的实际利用效率。公式中，分子`num_L1_tex_sectors / num_SM / SM_active_cycles`将总纹理缓存访问扇区数，按流多处理器（SM）数量与SM活跃周期数平摊，得到每个SM在每个活跃周期内的平均纹理访问速率；分母`num_sustained_peak_sectors_per_sm_per_cycle`是硬件每个SM、每个周期能支持的纹理缓存访问峰值。两者的比值乘以100%，即得到L1纹理缓存的利用率百分比，可直观判断算子是否受限于纹理缓存带宽，或是否存在资源浪费，为访存优化提供依据。
:::

- DRAM 利用率：由总 HBM 读写字节数推导，其中 `dram_bytes_read_and_write` 对应 `dram__bytes_read.sum` 与 `dram__bytes_write.sum` 计数器。测试服务器上峰值带宽为 2TB/s。

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/22.webp)

::: image-caption
DRAM带宽利用率（DRAM%）标准公式，核心是衡量程序运行时对内存带宽的实际利用程度。公式中，`dram_bytes_read_and_write`代表程序运行期间DRAM的总读写字节数，`duration`为程序运行时长，二者相除可得到实际DRAM传输带宽（单位时间内的读写速率）；分母`peak_bandwidth`是硬件标称的DRAM理论峰值带宽。用实际带宽除以峰值带宽，再乘以100，就得到百分比形式的带宽利用率。 这个指标是性能优化的关键参考：高利用率说明程序的访存模式高效，充分利用了硬件带宽；低利用率则可能暗示内存访问存在瓶颈（如缓存失效、访存碎片化），需针对性优化（如调整数据布局、提升缓存命中率），以减少内存带宽浪费，提升整体性能。
:::

### 附录 5：第 2.2 节的详细结果分析

> 结果：1.389 ms → 0.798 ms（下降 42.5%）。

| Version | Total Latency (ms) | Kernels | Latency (ms) | DRAM Traffic (GB) | Compute (GFLOPs) | L1/TEX Sectors (M) | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Decomposition（unpadded） | 1.386 | 2 CUTLASS GEMM – user & candidate matmul | 0.984 | 0.68 | 200 | 351 | L1/TEX (84%) |
|  |  | 1 ATen Gather – broadcast / 1 ATen Elementwise – add | 0.402 | 0.87 | 0.11 | 36 | DRAM (92%) |
| Decomposition（padded K） | 0.798 | 2 CUTLASS GEMM – user & candidate matmul | 0.400 | 0.69 | 200 | 0 | Balanced |
|  |  | 1 ATen Gather – broadcast / 1 ATen Elementwise – add | 0.398 | 0.87 | 0.11 | 36 | DRAM (92%) |

大幅加速主要来自两个因素：

- TMA：**矩阵对齐后，CUTLASS 会选择基于 TMA 的内核，完全绕过 L1/TEX** （sectors → 0）。未 padding 的内核还会让矩阵 `B` 也被错误惩罚：即便 `B` 的 `N` 已对齐，它仍然只能用 4-byte load，而不能使用 128-bit load。
- Bank conflict：**未 padding 的内核走的是 sm80 MMA 路径，其 swizzle 模式无法防止 4-byte `cp.async` 写入带来的 shared memory bank conflict**。padding 后的内核不存在这个问题。

### 附录 6：第 2.3 节的详细结果分析

> 结果：延迟 0.798 ms → 0.580 ms（下降 27.4%）。

| Version | Total Latency (ms) | Kernels | Latency (ms) | DRAM Traffic (GB) |
| --- | --- | --- | --- | --- |
| Decomposition（padded K） | 0.798 | 2 CUTLASS GEMM – user & candidate matmul | 0.400 | 0.68 |
|  |  | 1 ATen Gather – broadcast / 1 ATen Elementwise – add | 0.398 | 0.87 |
| iKBO Fusion | 0.580 | user GEMM & candidate iKBO kernel | 0.580 | 0.68 |

如预期，0.87 GB 的中间 DRAM 流量被完全消除。**NCU profiling 也揭示了进一步优化空间：occupancy 仅 6.25%，每个 scheduler 只有 1 个 warp，PC sampling 显示只有 23% 的周期真正用于有效工作：**

| Stall Reason | Percentage | Kernel 中主要对应内容 |
| --- | --- | --- |
| Stall long scoreboard | 41.8% | Global memory loads |
| Selected (executing) | 23.1% | 有效工作——实际发出的指令 |
| Stall wait | 20.1% | 等待 WGMMA |
| Stall barrier | 5.7% | 软件流水线阶段之间的 `bar.sync` |

::: block-1
每个 scheduler 只有 1 个 warp 时，所有 stall 都会完整暴露，因为没有其他 warp 可切换。若通过减小 pipeline 深度来提升 occupancy，又会牺牲 K-loop 的延迟隐藏。**这正是该内核的困难所在：为了保持 tensor core 吞吐，需要更大 tile 与更深 pipeline，但它们消耗掉大部分 shared memory 预算，导致几乎没有空间再通过 occupancy 来隐藏延迟。**
:::

### 附录 7：Release-Acquire 同步协议

> Producer（user CTA）：当 user tile 被写入 global memory 后，CTA 以 release 语义设置每个 tile 的 flag，保证数据在 flag 写入前已经可见：

```
tl.atomic_add(user_tile_flag_ptr, 1, sem="release", scope="gpu")
```

Consumer（candidate CTA）：每个 warp group 中仅一个线程使用 `ld.relaxed` 轮询 flag，以减少 spin 时内存流量。flag 变化后，再用一次 `ld.acquire` 建立 happens-before 边界，然后用 named barrier 把就绪状态广播给该 warp group 中全部 128 个线程：

```
if tlx.thread_id(axis=0) % 128 == 0:  # 1 thread per warp group (4 warps)    ready = tl.inline_asm_elementwise(        "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l",        [user_tile_flag_ptr], dtype=tl.int32, is_pure=False, pack=1)    while ready == 0:        ready = tl.inline_asm_elementwise(            "nanosleep.u32 50; ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l",            [user_tile_flag_ptr], dtype=tl.int32, is_pure=False, pack=1)    tl.inline_asm_elementwise(        "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l",        [user_tile_flag_ptr], dtype=tl.int32, is_pure=False, pack=1)tlx.named_barrier_wait(12, 128)
```

### 附录 8：TLX 与 Triton 的 NCU Profiling 指标

| Metric | Triton | TLX | Notes |
| --- | --- | --- | --- |
| Theoretical Occupancy | 6.25% | 18.75% | 每个 CTA 3 个 warp group 对比 1 个 |
| DRAM Throughput (`dram__cycles_active.avg.pct_of_peak_sustained_elapsed`) | 38.51% | 52.39% | 连续 TMA load 带来更高利用率 |
| L2 Cache Throughput (`lts__throughput.avg.pct_of_peak_sustained_elapsed`) | 73.69% | 83.86% | 瓶颈所在；TLX 更接近峰值 |

### 附录 9：普通 Flash Attention 与 IKBO Flash Attention 的 roofline 分析

> 在 FP16/BF16 精度下，设 `user_seq_len = 1024`、`n_seed = 64`、`B_candidate : B_user = 70:1`：

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/23.webp)

::: image-caption
通过算术强度（AI，FLOPs/Byte，即计算操作数与访存字节数的比值），量化对比了普通注意力算子、IKBO优化后算子与H100 GPU硬件的算力瓶颈。普通注意力算子的算术强度仅为60 FLOPs/Byte，远低于H100的硬件算术强度上限（495 FLOPs/Byte），说明算子受限于访存带宽，属于访存瓶颈型任务，无法充分利用GPU算力；而IKBO优化通过减少用户序列的访存总量（分母中用户序列访存量除以候选数`num_cand_user`），将算术强度提升至833 FLOPs/Byte，超过了H100的硬件上限，说明优化后算子转变为计算瓶颈型任务，理论上可打满GPU算力，大幅提升硬件利用率与实际性能。
:::

### 附录 10：IKBO TLX FA3 的 SMEM 占用

| SMEM buffer | Counts | Block dim | Total size |
| --- | --- | --- | --- |
| Query | 2（每个 consumer group 1 个） | `64 * 128`（2 Bytes） | 32KB |
| Key | 2 | `128 * 128`（2 Bytes） | 64KB |
| Value | 2 | `128 * 128`（2 Bytes） | 64KB |
| Output | 2（每个 consumer group 1 个） | `64 * 128`（2 Bytes） | 32KB |
| Total |  |  | 192KB |

### 附录 11：在推荐系统边界条件下，对 IKBO FA、CuTeDSL FA4 Hopper 与 TLX FA3 Hopper 的 benchmark

> IKBO 内核本质上是在启用 user-candidate 交互映射逻辑，其 IO 与计算模式与 GQA 相近。benchmark 时，对 IKBO 内核使用稳定的 `B_candidate : B_user = 64:1`，并为 CuTeDSL FA4 Hopper 的 GQA 版本构造相似的计算模式（`Q_seq_len = 128`，以保证 2-consumer warpgroup 正常工作）。另外还需说明：IKBO 内核仍需额外消费 candidate-user mapping tensor，以处理实时排序中每用户候选数变化的问题。

| Kernel type | Throughput (TFLOPs/s) | IO (GB/s) |
| --- | --- | --- |
| Triton IKBO FA2 | 425 | 519 |
| TLX IKBO FA3 | 418 | 510 |
| TLX IKBO FA3 persistent | 592 | 723 |
| TLX IKBO FA3 persistent V2（reverse k,v order） | 537 | 655 |
| CuTeDSL FA4 Hopper GQA | 518 | 633 |
| TLX FA3 GQA | 576 | 703 |

IKBO FA 与开源 GQA kernel 的 benchmark。IKBO kernel 的 Q、K、V 形状顺序为 `[Batch size, num head, seq, d_head]`：`Q_ikbo [2048, 2, 64, 128]`，`K/V_ikbo [32, 2, 1024, 128]`。GQA kernel 的 Q、K、V 形状为：`Q_gqa [1024, 2, 128, 128]`，`K/V_gqa [32, 2, 1024, 128]`。

| Kernel type | Throughput (TFLOPs/s) | IO (GB/s) |
| --- | --- | --- |
| Triton IKBO FA2 | 449 | 329 |
| TLX IKBO FA3 | 470 | 345 |
| TLX IKBO FA3 persistent | 621 | 455 |
| TLX IKBO FA3 persistent V2（reverse k,v order） | 587 | 430 |
| CuTeDSL FA4 Hopper GQA | 608 | 445 |
| TLX FA3 GQA | 628 | 460 |

IKBO FA 与开源 GQA kernel 的 benchmark。`Q_ikbo [2048, 2, 64, 128]`，`K/V_ikbo [32, 2, 2048, 128]`。GQA kernel 的 `Q_gqa [1024, 2, 128, 128]`，`K/V_gqa [32, 2, 2048, 128]`。

注：由于标准 Flash Attention 内核不包含 IKBO 逻辑，我们使用具有相近 IO 成本与 FLOPs 消耗的 GQA 配置来模拟 cuteDSL 版本的吞吐结果。

### 附录 12：instruction cache miss 会在 consumer-2 warpgroup 上造成显著延迟

![文章插图](/articles/2026-05-11-621-tflops-flash-attention-7ad8b940/images/24.webp)

::: image-caption
图 A1 注意力前向算子（`attn_fwd`）的GPU性能剖析时间线图，展示了Core 0上CTA76中多个warp（线程束）的指令执行序列。图中橙色框标注了指令缓存缺失（Instruction cache miss）导致的性能问题：右侧区域的warp指令流出现了明显的执行停顿与气泡，例如warp 0的内存加载指令（`issue_v_load`/`issue_k_load`）出现间隙，warp 4/8的`QKT`、`softmax`等核心计算操作也被打断。当指令缓存未命中时，GPU需从更高层级内存读取指令，导致流水线停滞、warp调度被阻塞，原本紧凑的指令流产生长延迟，直接降低了SM（流多处理器）的指令级并行效率与利用率，尤其在循环密集的算子中，这类缺失会显著拖慢整体执行性能。
:::

instruction cache miss 修复前后结果：

```
Before instruction cache miss fix:    ---------------------------------------------------- ----------- ------------    Metric Name                                          Metric Unit Metric Value    ---------------------------------------------------- ----------- ------------    gcc__cache_requests_type_instruction.sum                              319,394    gcc__cache_requests_type_instruction_lookup_miss.sum                    7,234    sm__icc_requests.sum                                       cycle    6,049,376    sm__icc_requests_lookup_hit.sum                            cycle    5,438,421    sm__icc_requests_lookup_miss.sum                           cycle      610,955    ---------------------------------------------------- ----------- ------------After instruction cache miss fix:    ---------------------------------------------------- ----------- ------------    Metric Name                                          Metric Unit Metric Value    ---------------------------------------------------- ----------- ------------    gcc__cache_requests_type_instruction.sum                               33,008    gcc__cache_requests_type_instruction_lookup_miss.sum                      769    sm__icc_requests.sum                                       cycle      792,437    sm__icc_requests_lookup_hit.sum                            cycle      722,244    sm__icc_requests_lookup_miss.sum                           cycle       70,193    ---------------------------------------------------- ----------- ------------
```

参考资料

[1] Meta Adaptive Ranking Model: *https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/*

[2] TLX (Triton Low-Level Extensions): *https://github.com/facebookexperimental/triton/tree/tlx*

[3] https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/ikbo: *https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/ikbo*

[4] TLX（Triton Low-level Language Extensions）: *https://github.com/facebookexperimental/triton/tree/tlx*

[5] Proton tracer: *https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials/intra_kernel*

[6] Proton tracer: *https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials/intra_kernel*

[7] #831: *https://github.com/facebookexperimental/triton/pull/831*
