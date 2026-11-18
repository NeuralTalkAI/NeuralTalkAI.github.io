> 本文介绍来自华为**诺亚方舟实验室、清华大学和香港中文大学** 联合在**大语言模型量化上的最新工作 FlatQuant** (Fast and Learnable Affine Transformation)。
>
> FlatQuant 通过为每个线性层适配轻量的可学习的仿射变换，**有****效平滑 LLM 离群值，得到更加平坦的权重和激活值分布，**有效降低量化损失。相比此前的量化方法 [1][2]，本方法**首次** 在 **LLaMA-3-70B 上达到 W4A4 <1% 的精度损失**，并可带来**最高 2.3x prefill 和 1.7x decoding 加速比**。
>
> **注：本文做了大量修改，原文：** **https://zhuanlan.zhihu.com/p/899259762**，欢迎转载！

- 论文：**arxiv.org/abs/2410.09426**
- 代码：**github.com/ruikangliu/FlatQuant**

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/01.webp)

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/02.webp)

**1. 大语言模型 (LLM) **W4A4** 量化问题**

**模型量化** 是大语言模型 (LLM) 推理加速的常用技术，可以通过**将权重和激活值同时压缩到低比特** 来有效降低访存开销，并**利用峰值算力更高的 INT4/8 Tensor Core 完成矩阵运算**，从而带来实际的推理加速比。

然而，**目前的 W4A4 （权重4位，激活值4位）量化模型相比全精度模型还存在着较大的量化损失**，难以在实际应用中使用，也就难以利用峰值算力最高的 INT4 Tensor Core 加速 LLM 的实际推理部署。我们发现，**量化前权重和激活值分布的平坦度 (flatness) 是影响 LLM 量化误差的关键因素。**

直观来看，**分布越平坦，离群值就越少****，量化时的精度也就越高**。已有方法大多使用 pre-quantization transformations，通过在量化前对权重和激活值做等价变换得到更平坦的分布来降低量化误差，**常用的变换主要有 Per-channel Scaling [1] 和 Hadamard 变换 [2]。**

**FlatQuant 推动 W4A4 LLM 部署**

**然而，我们发现这些变换并不是最优的**，为此**我们提出 FlatQuant (Fast and Learnable Affine Transformation)，为每个线性层学习一个最优的仿射变换来有效缓解权重和激活值上的离群值**，从而得到**平坦的权重和激活值分布****，** 有效提升了量化精度。此外，针对推理中的在线变换，我们进行了算子融合进一步降低访存开销，使得在线变换仅带来极小的推理开销。

实验表明，**FlatQuant 在 W4A4 的设置下极大地减少了量化模型的精度损失**，甚至在**部分模型上达到了接近无损的效果 (e.g. LLaMA-3-70B)**，**轻量的在线变换**也使得 FlatQuant 能达到 **2.3x 的 prefill 和 1.7x 的 decoding 加速比**。我们希望 FlatQuant 能进一步推动 W4A4 LLM 的实际部署，从而更加有效地降低 LLM 的推理成本。

**2. 探索平坦分布与量化损失的优化路径**

## **The Flatness for Quantization**

LLM 的权重和激活值上存在较多的离群值，特别是激活值上常常存在离群值通道 (outlier channels)，导致 LLM 难以量化。目前针对 LLM WA 量化的方法大多在量化前**对权重和激活值做等价变换来用其他通道吸收离群值**，从而**得到更加平坦的分布**以降低量化损失。例如：

- Per-channel Scaling [1] 对应的等价变换为![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/03.webp)，通过 scaling 将激活值上的离群值转移到权重的相同通道上，使得激活值分布更加平坦；
- Hadamard 变换 ![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/04.webp) 则是通过给权重和激活值同时做 Hadamard 变换来将离群值重新分配到权重/激活值的其他通道上。

**为降低理解的难度，NeuralTalk 在此** **举一个例子****。**

权重和激活值分布可以看作两个斜坡，变换就类似于用铲子搬土，土不会凭空增加或者减少，所以**目标是通过把两个坡中高处**（离群值）**的土填到低处**（非离群值通道），**从而把这两个坡填平**。

- **Per-channel Scaling** 就相当于只能把一个坡上的土填到另一个坡的**相同位置上**，比较**局限**。
- **Hadamard 变换** 相当于在每个坡的内部把高处的土**填到自身的低处**，但**不能在两个坡之间转移土**。并且由于不同坡的形状不同，**相同的 Hadamard 变换（坡内搬土方式）不一定适用于所有**土坡。

相比之下，FlatQuant 方法可以被看作是一种**更加精细和智能的“搬土”策略**。在这个方法中，我们**不再局限于只在单个斜坡内部移动土，也不只是在两个斜坡的相同位置上进行土的转移**。相反，FlatQuant 允许我们**对每个斜坡进行定制化的调整**，这意味着我们可以**针对每个斜坡的独特形状和需求，设计出最佳的“搬土”方案**。

这就相当于**为模型的每一层学习一个特定的仿射变换，以得到更平坦的分布**，并且可以**自适应地平衡权重和激活值**的量化难度。

**2.1 平坦分布的追求与挑战**

在下图 1 中，我们画出了 LLM 的不同权重和激活值在变换前后的分布情况，**理想情况下，我们希望能利用所有通道吸收离群值**，使得变换后的分布呈现**一条平坦的水平线**。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/05.webp)

****图 1：等价变换前后模型的权重和激活值分布，具体来说，**** **按通道幅值** （即 Frobenius 范数）**降序排列的 LLaMA-3-8B 和 LLaMA-3-70B 的权重和输入的分布情况。**

注：在 Transformer 层中， Wo 和 Xo 分别表示自注意力层输出投影层的权重矩阵和输入。Wg 和 Xg 分别表示前馈网络中门控线性层的权重和输入。更多的可视化内容可以在文章附录 D 中找到。四个图分别是：(a) LLaMA-3-8B 的第10层Transformer 的 Wo 。(b) LLaMA-3-8B 的第10层Transformer的 Xo 。(c) LLaMA-3-70B 的第30层Transformer的 Wg 。(d) LLaMA-3-70B 的第30层 Transformer 的 Xg 。

但如上图 1 所示，我们发现**已有的等价变换得到的分布仍然可能是不平坦的**：

- **Per-channel Scaling**，**离群值仍然被限制在了权重和激活值的相同通道上，非离群值通道得不到有效利用**，因此不管是权重还是激活值，变换后的分布都非常陡峭，呈现出非常明显的离群值通道。
- **Hadamard 变换** **对所有权重和激活值都施加相同的变换**，而不同层的权重和激活值分布是不同的，这意味着 Hadamard 变换**并不是对于每个层的最优解**，例如图 1(a)(b) 中，LLaMA-3-8B 的权重和激活值经过 Hadamard 变换后仍然比较陡峭，特别是激活值上的离群值无法得到有效平滑。此外，**Hadamard 变换作为一种正交变换不会改变向量的模长**，而 **LLM 激活值上大量的离群值会导致激活值模长显著大于权重**，这导致正交变换后的激活值量化难度也会显著高于权重，无法像 Per-channel Scaling 一样灵活地平衡权重和激活值上的量化难度。

相比之下，**FlatQuant 通过给每一层针对性地学习仿射变换，不仅可以得到平坦的分布，还可以自适应地平衡权重和激活值的量化难度**。

**2.2 不同等价变换下的量化损失平面**

下面的图 2 中，我们画出了不同变换后 LLM 的量化损失平面，可以发现，per-channel scaling 和 Hadamard 变换都无法很好处理具有 massive outlier [3] 的关键词元 (pivot token)，导致在首词元上具有非常大的量化误差，已有研究表明**关键词元上的量化误差会比较严重地影响模型的量化精度** [4]。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/06.webp)

**图 2：不同等价变换下的量化损失平面。明显看出 FlatQuant 方法的 MSE 更小。**

相比之下，**FlatQuant 则可以显著降低关键词元上的量化损失，并有效抑制量化误差的逐层传播，带来更加平坦的量化损失平面**。

**2.2 方法概述**

**轻量仿射变换**

FlatQuant 通过**轻量的仿射变换** 平滑权重和激活值上的离群值，需要为每个线性层学习最优的仿射变换![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/07.webp)：

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/08.webp)

学到![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/09.webp)后，变换![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/10.webp)可以融到权重中不会带来额外推理开销，但![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/11.webp)必须作为在线变换，这会使得线性层的存储和计算开销翻倍，这显然是不现实的。

**Kronecker Decomposition**

为了解决上述问题，我们对![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/09.webp)使用 Kronecker 分解![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/12.webp)，其中![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/13.webp)，选取![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/14.webp)并且![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/15.webp)尽可能接近即可，例如![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/16.webp)时，有![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/17.webp)。这样在线变换![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/18.webp)可以表示为![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/19.webp)，相当于把![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/20.webp)reshape 为![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/21.webp)然后左右分别乘上![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/22.webp)和![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/23.webp)两个小矩阵，相比直接使用稠密变换矩阵可以将额外的内存和计算开销分别降低至原来的![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/24.webp)和![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/25.webp)。

**Per-channel Scaling**

Kronecker 分解本质上还是对![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/09.webp)的 rank-1 近似，我们进一步使用 learnable Per-channel Scaling 提升 Kronecker 分解的表征能力。Per-channel Scaling 可以融到前序的 LN/线性层中不会带来额外推理开销。

**Learnable Clipping Thresholds**

我们对变换后的权重和激活值进一步采用了 learnable clipping 来更好地消除离群值。

**以上就是关键方****法，分步来说：**

1. 轻量仿射变换：通过学习每个线性层的最优仿射变换来**平滑离群值**。
2. Kronecker 分解：将大的变换矩阵分解为小矩阵，**减少存储和计算开销**。
3. Per-channel Scaling：为每个通道提供独立的缩放因子，**增加变换的灵活性**。
4. Learnable Clipping Thresholds：通过可学习的裁剪阈值**进一步减少离群值的影响**。

**优化过程**

损失函数采用 Layer-wise MSE loss：

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/26.webp)

其中，![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/27.webp)和![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/28.webp)分别为 FP16 和量化后的 Transformer block，![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/29.webp)包括仿射变换、Per-channel Scaling 以及 clipping 参数，其中仿射变换初始化为随机仿射变换。整个训练过程非常轻量，7B 模型仅需单卡大约 1h 即可完成。

****模型架构****

如图下图 3 所示，FlatQuant 在**单个 Transformer 内会引入 5 种不同的在线变换**，**对于 LLaMA-2-7B，这些在线变换在序列长度 2K 时的 FLOPs 仅为 FP16 模型的 2.61%**，对在线变换中两个小矩阵乘以及量化操作的算子融合还可以帮助进一步降低 FlatQuant 的额外推理开销。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/30.webp)

**图 3：FlatQuant 模型架构图**

另外注意到，在 QuaRot [2] 和 SpinQuant [5] 中，为了降低在线推理开销，MHA / MLP 输入处的正交变换会被融合到前序线性层里，但由于残差连接的限制，不同 Transformer block 中的 MHA / MLP 都必须共享输入处的正交变换，这不仅限制了变换的灵活性，还使得在优化变换矩阵时必须采用端到端优化，需要较大的训练开销。

相比之下，FlatQuant 不仅可以**对每个线性层都学得最适配的等价变换**，还可以**逐层优化，仅需单卡**即可完成对 **70B 模型的量化**。

## **3. 实验结果**

量化设置. 实验中，我们保持了与 QuaRot [2] 相同的量化设置，权重和激活值分别**采用 per-channel 和 per-token 对称量化**，**KV cache 量化采用 group-wise 非对称量化 (g128)**，校准集为来自 WkiText-2 数据集的 128 条样本。

## **3.1 量化精度**

我们测试了 W4A4 下量化模型的 PPL 和 QA 任务上的精度结果，从表 1 和表 2 中可以看到，**FlatQuant 在使用 RTN 作为 weight quantizer 时精度就已经能比较****明显地超过 QuaRot 和 SpinQuant 使用 GPTQ 的效果**。

- 对于较大的 13B/70B 模型，QA 精度损失均在 1% 左右。
- 更小的 7B/8B 模型的精度损失也维持在了 2% 左右。
- FlatQuant **对于更难量化的 LLaMA-3 模型提升尤为明显**， 例如 **LLaMA-3-70B 的 QA 任务上 FlatQuant 相比 SpinQuant 有超过 7% 的精度提升**，同时**与全精度模型的精度差距保持在 1% 以内**。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/31.webp)

**表 1：W4A4 PPL 实验结果**

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/32.webp)

**表 2：W4A4 zero-shot QA 任务实验结果**

## **3.2 端到端加速比**

我们在 RTX3090 上测试了 FlatQuant 的**prefill/decoding 端到端加速比**。如图 4 所示，**FlatQuant 最高能带来 2.30x 的 prefill 和 1.76x 的 decoding 加速比，推理速度超过了 QuaRot，相比 INT4 也仅有极小的加速比损失**。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/33.webp)

**图 4：端到端加速比**

## **3.3 更多实验**

**(1) 消融实验.** 从表 3 中可以看到，**在 RTN 量化的基础上加入 LT**(Learnable Transformation) 就**已经能极大地提升量化模型精度**，进一步**加入 PS (Per-channel Scaling) 和 LCT (Learnable Clipping Thresholds) 还能进一步提升**模型精度。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/34.webp)

**表 3：LLaMA-3-8B 消融实验**

**(2) 权重量化.** FlatQuant 在权重量化上也**能与 SOTA 的 uniform 量化方法达到相当的精度**。

**(3) Train One and Get More.** FlatQuant 中 W4A4 量化设置下**学到的变换矩阵可以直接用在其他量化设置下**，这使得我们能更加便利地在不同量化设置下使用 FlatQuant。

![文章插图](/articles/2024-10-26-llm-4-bit-flatquant-1e5d7f64/images/35.webp)

**表 5, 6： 更多量化设置**

**4. 总结**

现有的量化方法在 W4A4 下量化损失大难落地。量化前权重和激活值分布的平坦度显著影响量化误差。 FlatQuant **通过为每个线性层适配轻量的可学习的仿射变换，平滑权重和激活值上的离群值，得到更平坦的分布解决该问题**。

在 **LLaMA-3-70B 模型上实现小于1%的量化损失**。**部分模型接近无损效果**。**性能上 FlatQuant 带来高达 2.3 倍的 prefill 加速和 1.7 倍的 decoding 加速**。

总的来说， **FlatQuant 是一种创新的量化方法，它通过学习最优的仿射变换来提高 LLM 量化精度，保持高加速比的同时，显著降低量化损失**。这项工作对于推动大型语言模型在实际应用中的部署具有重要意义。

## **5. 参考文献**

- [1] Xiao, Guangxuan, et al. “**Smoothquant: Accurate and efficient post-training quantization for large language models.** ” International Conference on Machine Learning. PMLR, 2023.
- [2] Ashkboos, Saleh, et al. "**Quarot: Outlier-free 4-bit inference in rotated llms.** " arXiv preprint arXiv:2404.00456 (2024).
- [3] Sun, Mingjie, et al. **"Massive Activations in Large Language Models."** arXiv preprint arXiv:2402.17762 (2024).
- [4] Liu, Ruikang, et al. **"IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact."** arXiv preprint arXiv:2403.01241(2024).
- [5] Liu, Zechun, et al. **"SpinQuant--LLM quantization with learned rotations."** arXiv preprint arXiv:2405.16406(2024).

## **相关文章**

- [让量化大模型无损，IntactKV：保持关键词元无损，轻松提升量化精度！](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887239&idx=1&sn=b976d340b7ba0b1b0b120874b1763377&chksm=b2904d6585e7c473a8a3980798c6600456127d7985327efed2ec8aafa1ca4716a1512e6c05c6&scene=21#wechat_redirect)
- [走向ARM CPU 1-bit推理的极致道路](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885109&idx=1&sn=0f32a578c71b66d6ceffb7952859c575&chksm=b29045d785e7ccc185ad120904c2e4e14987b473e2e2aa6817c7ebd66c75b3c5a359ad58d8c8&scene=21#wechat_redirect)
- [MLCEngine: 全平台大语言模型推理引擎](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887268&idx=1&sn=0f7b0f591ed129eace8d036662b44b11&chksm=b2904d4685e7c450ee0b3763c447258f87d56c894073608d0c12cc40c081f7ff144ace5e6812&scene=21#wechat_redirect)
- [SIMD 指令集与数据并行程序](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887352&idx=1&sn=c579010b3cb554095be51ebc6d95748c&chksm=b2904d1a85e7c40c0c7d239b03a73abae86c573382803e35c538b22bb487c2c307dd39b2d7b7&scene=21#wechat_redirect)
- [低比特离线量化EasyQuant: 比TensorFlow更友好的离线量化方式](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885024&idx=1&sn=fc2f5ea0db0a61eb9e7f72aea46d0688&chksm=b290440285e7cd14ae098380e7cecfb35b36aea42739e3fed49c56c856e02c2a57c7ff7a1b14&scene=21#wechat_redirect)
- [IEEE 754 Floating NaN](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887217&idx=1&sn=ac3c93ac58b1f2eddd074ee3d5874951&chksm=b2904d9385e7c485882347eebf7602540a9aa3849ebb07e706fb166df66e4cabd7fcba46f830&scene=21#wechat_redirect)
- [爱奇艺在DCN、EDVR等 4K 超分模型上的 TensorRT 10倍加速实践](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886743&idx=1&sn=987497f322f403029d0e6e8b08a61b49&chksm=b2904f7585e7c6639491bc5696596f34accda01fa5f9ee015101752be0dac34c4f2676980711&scene=21#wechat_redirect)
- [AI、ML加速器发展趋势综述](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886740&idx=1&sn=4e66cc0346e914168191425145cc8c4a&chksm=b2904f7685e7c660f2bfce6bb3738aeffd4a030dfb273b068b0ec30137cecf3f1e79ece0970d&scene=21#wechat_redirect)
- 一篇关于深度学习编译器架构的综述论文

**- 查看往期，【阅读原文】 -**
