**— 机器之心编译 —**

Composable Kernel（CK）库旨在提供一套在 **AMD GPU 上算子融合的后端方案**，该研究希望未来能够**移植到 AMD 的所有 GPU 上，并且最终也可以被移植到 AMD CPU 上，该项目已开源**。

此外，**CK库与Meta AITemplate的深度合作** 大幅提升了AI模型在AMD GPU的端到端性能。

AITemplate 是一个能把 AI 模型转换成高性能 C++ GPU 模板代码的 Python 框架。该框架在设计上专注于性能和简化系统。AITemplate 系统一共分为两层：前端部分进行图优化，后端部分针对目标 GPU 生成 C++ 模板代码。AITemplate 不依赖任何额外的库或 Runtime，如 cuBLAS、cudnn、rocBLAS、MIOpen、TensorRT、MIGraphX 等。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/01.webp)

**图1 BERT类模型AMD MI250性能 vs. FasterTransformer A100性能比较**

图优化在降低 AI 模型的训练和推理使用的时间和资源方面起着重要作用。**图优化的一个重要功能是模型中将可以融合的算子进行融合，通过降低内存占用和减少数据在低速内存中的搬运来提高计算效率**。然而，实现一套能够**提供各种算子融合的后端方案** 难度很大，导致在实际硬件上 AI 模型能够使用的算子融合非常有限。

Composable Kernel (CK)库旨在提供一套**在 AMD GPU 上的算子融合的后端方案**。CK 使用**通用编程语言 HIP C++**，完全**开源**。其**设计理念** 包括：

- **高性能****& 高生产力**：CK 的核心是一组精心设计，高度优化，**可复用的基础模块**。CK 库内所有的算子都是**通过组合这些基础模块实现的**。复用这些基础模块大大缩短开发后端算法的周期，同时还能保证高性能。
- **精通当前的 AI 问题**，快速适应未来的 AI 问题：CK 旨在提供一套完整的 AI 算子后端方案，这让复杂的算子融合成为可能，因为这样让整个后端都可以用 CK 实现，而**不需依赖外部算子库**。CK 的**可复用基础模块足以实现常见 AI 模型**（机器视觉，自然语言处理，等等）所需的各种算子及其融合。当**新出现的 AI 模型需要新的算子时，CK 也将会提供所需的基础模块****（复用已有模块需要对已有模块的API很熟悉，对新算子也需要人力跟进）**。
- AI 系统专家的简单但强大的工具：**CK 所有的算子都是用 HIP C++ 模版实现的**。AI 系统专家可以通过实例化模版来定制这些算子的属性，比如数据类型，元操作类型，张量存储格式，等等。这通常只需要几行代码**（听起来HIP C++的扩展和表达能力很强）**。
- 友好的 HIP C++ 界面：HPC 算法开发者一直在推动着 AI 计算加速的前沿。CK 的一个重要设计理念就是要让 HPC 算法开发者更容易对 AI 加速作出贡献。因此 **CK 所有核心模块都是用 HIP C++ 实现**，而不是 Intermediate Representation （IR）。HPC 算法开发者直接**以他们熟悉的编写 C++ 代码的形式编写算法，而无需像基于 IR 的算子库那样，以通过编写针对某种特定算法的 Compiler Pass 来实现**。这样做可以大大提高算法的迭代速度。
- 可移植性：今天使**用 CK 作为后端的图优化**将能够移植到**未来 AMD 的所有的 GPU** 上，并且最终**也可以被移植到****AMD CPU****上** 【2】。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/02.webp)

- CK 源代码：https://github.com/ROCmSoftwarePlatform/composable_kernel

**核心概念**

CK 引入了两个概念：**Tensor Coordinate Transformation 和 Tile Tensor Operator ，二者共同组成了 CK 的可复用的基础模块。**

1. **开创性的引入“张量坐标变换”** （Tensor Coordinate Transformation）降低 AI 算子的编写复杂度。该研究开创性地定义了**一组可复用的 Tensor Coordinate Transformation 基础模块**，并且用它们**把复杂的 AI 算子**（比如卷积，group normalization reduction，Depth2Space，等等）**以数学严谨的方式重新表达成了最基础的 AI 算子**（GEMM，2D reduction，tensor transfer，等等）。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/03.webp)

**图1 使用 CK 的 Tensor Coordinate Transformation 基础模块将 convolution 算子表达成 GEMM 算子**

这项技术可以让为基础 AI 算子编写的算法直接被用到所有与之对应的复杂的 AI 算子上，而无需重写算法。

2. **基于 Tile 的编程范式**：开发算子融合的后端算法可以被看成先**将每一个融合前的算子（独立算子）拆解成许多 “小块” 的数据操作**，然后再把这些**“小块” 操作组合成融合的算子**。每一个这样的 “小块” 操作都对应一个原始的独立算子，但是被操作的数据只是原始张量的一部分（tile），因此这样的 “小块” 操作被称为 Tile Tensor Operator。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/04.webp)

**图2 C** **K 的组成（下：可** **复用的基础模块；上：独立算子与融合算子）**

CK 库包含一组针对 Tile Tensor Operator 的高度优化的实现，CK 里**所有的 AI 独立算子和融合算子都是用它们实现的**。目前，这些**Tile Tensor Operators 包括 Tile GEMM，Tile Reduction 和 Tile Tensor Transfer**。**每一个 Tile Tensor Operator 都有针对 GPU thread block，warp 和 thread 的实现**。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/05.webp)

**图** **3 Composable** **Kernel Overview**

**代码结构**

**CK 库结构分为四层，从下到上分别是**：**Templated Tile Operator**，**Templated Kernel and Invoker**，**Instantiated Kernel and Invoker** 和 **Client API**【3】。每一层对应不同的开发者：

- **AI 系统专家**：“我需要一个**后端方案**提供高性能的独立和融合算子让我可以**直接使用** ”。这个例子【4】里用的 **Client API 和 Instantiated Kernel and Invoker** 提供了预先实例化并编译好的对象，以满足这类开发者的需求。
- **AI 系统专家**：“我为一个开源的 AI 框架做最先进的**图优化工作**。我需要一个能够为图优化所需的所有融合算子提供高性能 kernel 的后端方案。同时我也需要定制这些 kernel，所以像 “要么接受，要么弃用” 的黑盒解决方案不能满足我的需求”。**Templated Kernel and Invoker 层**能满足这类开发者。比如这个例子【5】中开发者可以自己**使用 Templated Kernel and Invoker 层实例化出所需的 FP16 的 GEMM + Add + Add + FastGeLU 的 kernel**。
- **HPC 算法专家**：“我的团队为公司内部不断迭代的 AI 模型开发高性能后端算法。我们**团队中有 HPC 算法专家**，但我们仍然希望可以通过复用和改进硬件供应商提供的高度优化的源代码来提高我们的生产力，并且让我们的代码可以被移植到未来的硬件构架上。我们**希望可以****不用通过****与硬件提供商****分享我们的代码****来做到这点**”。**Templated Tile Operator** 层可以帮助到这一类开发者。比如这个代码【6】中开发者**使用 Templated Tile Operator 来实现 GEMM 的优化管线**。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/06.webp)

**图4 CK 库四层结构**

**基于 AITemplate + CK 的端到端模型推理**

Meta 的 AITemplate 【7】（AIT）是一个统一 AMD 和 Nvidia GPU 的 AI 推理系统。**AITemplate 使用 CK 作为其 AMD GPU 上的后端**，它使用的是 CK 的 Templated Kernel and Invoker 层。

AITemplate + CK 在 AMD Instinct™ MI250 上取得了多个重要 AI 模型最先进的推理性能。CK 里大多数先进的融合算子的定义，都是在 AITemplate 团队的远见下推动的。**许多融合算子的算法也是由 CK 和 AITemplate 团队共同设计**。

本文比较了几个端到端模型在 AMD Instinct MI250 和同级别产品【8】的性能表现。本文中所有 AMD Instinct MI250 的 AI 模型的性能数据都是用 AITemplate【9】 + CK【10】取得的。

**实验**

**ResNet-50**

下图显示了 AMD Instinct MI250 上的 AIT + CK 与 A100-PCIe-40GB 和 A100-DGX-80GB 上的 TensorRT v8.5.0.12 【11】（TRT）的性能比较。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/07.webp)

结果显示 **AMD Instinct MI250 上的 AIT + CK 取得了相比于 A100-PCIe-40GB 上的 TRT 1.08 倍的加速**。

**BERT**

**一个基于 CK 实现的 Batched GEMM + Softmax + GEMM 融合算子模版，可以** **完全消除掉****中间结果在 GPU 计算单元（Compute Unit）与 HBM 之间的****搬运**。通过使用这个融合算子模版，attention layer 许多原本是**带宽瓶颈（bandwidth bound）的问题变成了计算瓶颈（compute bound）的问题**，这样可以更好发挥 GPU 的计算能力。这个 CK 的**实现深受 FlashAttention 【12】的启发，并比原始的 FlashAttention 的实现减少了更多的数据搬运**。

下图显示了 AMD Instinct MI250 上的 AIT + CK 与 A100-PCIe-40GB 和 A100-DGX-80GB 上的 FasterTransformer v5.1.1 bug fix 【13】（FT）的 Bert Base 模型（uncased)的性能比较。当 Sequence 是 4096 时，FT 在 A100-PCIe-40GB 和 A100-DGX-80GB 上会在 Batch 32 时 GPU 内存溢出。因此，在 Sequence 是 4096 时，本文只显示 Batch 16 的结果。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/08.webp)

结果显示 **AMD Instinct MI250 上的 AIT + CK 取得了相比于 A100-PCIe-40GB 上的 FT 3.28 倍，以及相比于 A100-DGX-80GB 上的 FT 2.91 倍的加速**。

**Vision Transformer （VIT）**

下图显示了 AMD Instinct MI250 上的 AIT + CK 与 A100-PCIe-40GB 和 A100-DGX-80GB 上的 TensorRT v8.5.0.12（TRT）的 Vision Transformer Base （224x224 图片）的性能比较。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/09.webp)

结果显示 **AMD Instinct MI250 上的 AIT + CK 取得了相比于 A100-PCIe-40GB 上的 TRT 1.8 倍，以及相比于 A100-DGX-80GB 上的 TRT 1.4 倍的加速**。

**Stable Diffusion**

端到端的 Stable Diffusion

下表显示 AIT + CK 在 AMD Instinct MI250 上 Stable Diffusion 端到端（Batch 1，2，4， 6）的性能数据。当 Batch 是 1 时，在 MI250 上只有一个 GCD 被使用，而在 Batch 2，4，6 时，两个 GCD 都被使用了。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/10.webp)

Stable Diffusion 中的 UNet

不过本文还没有关于使用 TensorRT 运行 Stable Diffusion 端到端模型的公开的信息。但这篇文章“Make stable diffusion 25% faster using TensorRT” 【14】说明了怎么使用 TensorRT 加速 Stable Diffusion 中的 UNet 模型。**UNet 是 Stable Diffusion 中最重要最花时间的部分**，因此**UNet 的性能大致反应了 Stable Diffusion 的性能**。

下图显示了 AMD Instinct MI250 上的 AIT + CK 与 A100-PCIe-40GB 和 A100-DGX-80GB 上的 TensorRT v8.5.0.12（TRT）的 UNet 的性能比较。

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/11.webp)

结果显示 **AMD Instinct MI250 上的 AIT + CK 取得了相比于 A100-PCIe-40GB 上的 TRT 2.45 倍，以及相比于 A100-DGX-80GB 上的 TRT 2.03 倍的加速**。

**算子开发伪代码**

![文章插图](/articles/2022-11-14-amd-composable-kernel-tile-transformer-6a6b85ec/images/12.webp)

**图5 Transform a forward convolution operator into a GEMM operator**

**原文链接**

- https://community.amd.com/t5/instinct-accelerators/amd-composable-kernel-library-efficient-fused-kernels-for-ai/ba-p/553224

**相关文章**

- [谈谈AI编译软件栈、软硬件协同、TVM和MLIR的看法](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886637&idx=1&sn=477371ef67890c91b565b383493959b6&chksm=b2904fcf85e7c6d9fd752119c1487b08a2e9c8c7d847067f6b481467a862ad0508100470a792&scene=21#wechat_redirect)
- [一篇关于深度学习编译器架构的综述论文](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884994&idx=1&sn=2415d682852bd3d29f9f522e4ac2e8d9&chksm=b290442085e7cd367068d267e2633e376e17a5bb2f47c09488a762cdec8e8bf800e327a98937&scene=21#wechat_redirect)
- [AI框架图层IR的优势、特殊需求和挑战](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885305&idx=1&sn=bbca62d8de3794156ad35254a7c471e6&chksm=b290451b85e7cc0d4744486d176d5a071141c0e7643b2932df57f07b59f3e070df8914a0e1ea&scene=21#wechat_redirect)
- [Paddle Lite底层backend的kernel选择策略](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884913&idx=1&sn=66acdf5b2db0aabb58f0d34ce638ce8a&chksm=b290449385e7cd85ebf4056dcea410b370b7085c6c2582731af7b6dac6f54b70947c7f2b3014&scene=21#wechat_redirect)
- [AI 硬件公司接入框架与 AI 框架算子层级的思考](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886093&idx=1&sn=bc56c3a0843a39dbd66b0b513f5803b9&chksm=b29041ef85e7c8f9c484a5a2e5646d8a8822b37b98fce0d6d57fab0e1ea12b5d9c7b09541336&scene=21#wechat_redirect)
- [AI编译优化---访存密集算子优化](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885230&idx=1&sn=8c81fcf79d10decaf6fea7968d3203bf&chksm=b290454c85e7cc5a96aae15580d105c3603ecc8226feb33af9473a5e0a7d506a6c267095437d&scene=21#wechat_redirect)
- [AI编译优化--计算密集算子优化](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885202&idx=1&sn=7ee2c929b55b21f5981d6ae3186d15f2&chksm=b290457085e7cc66372ca7b84ce0bf0279da704ad3f416e2bd8f281d22070133ae13302c1418&scene=21#wechat_redirect)
- [AI编译优化--业务实践](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885245&idx=1&sn=f19f72507a44bed1b5d964ce499978f8&chksm=b290455f85e7cc493af5a5a7c6a5e97172c33eef647dbdef8a2192ae7ecd3eb89c1dd740e639&scene=21#wechat_redirect)
- [AI编译优化--总纲](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885158&idx=1&sn=c5b2f5399d133f1b4389cbf034fabf06&chksm=b290458485e7cc9201483b2411c2dffd481c4fee4510dcf2b54de9bd30a6a641cb595c9be114&scene=21#wechat_redirect)
- [新一代深度学习编译技术变革和展望](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886255&idx=1&sn=1b55f4fe1335260a6086082577ec8951&chksm=b290414d85e7c85bbe0b14e41e3b36b6266bb51f1a4dee67d9d4eef0f797e2eb231e1de406f4&scene=21#wechat_redirect)
- [bottom-up多层规约【图融合】策略：解决buffer/loop级融合、“内存墙”突破融合、“并行墙”突破融合](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886599&idx=1&sn=d7713e4100c06655538cfd3ee651ba21&chksm=b2904fe585e7c6f3d75740828bc815ec6a030df3419ba621da87a031f39cc6d57f9c87ced665&scene=21#wechat_redirect)

> 点击【阅读原文】，查看往期文章
