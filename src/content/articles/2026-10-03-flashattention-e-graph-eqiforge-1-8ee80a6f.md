关键词：**相等饱和**、:mdnice-bold-italic[张量程序编译]、EqiForge、**算子融合**、GPU 内核优化

> 高效的 GPU 张量程序实现往往需要高层代数表达与底层执行策略的联合优化。这是本文作者点明的核心判断，也是当前张量编译领域的核心痛点。**随着算子间变换组合不断叠加，优化搜索空间呈爆炸式增长，代数重写与执行优化的分层处理模式，很难支撑跨层级的联合优化规模化落地。**

![文章插图](/articles/2026-10-03-flashattention-e-graph-eqiforge-1-8ee80a6f/images/01.webp)

- **Unleashing the Power of Equality Saturation for Tensor Program Superoptimization**
- 组织机构：浙江大学
- 论文链接：https://arxiv.org/pdf/2609.12330

::: related-title
**相关推荐**
:::

::: related-links
- **[端到端 LLM 编译器 nncase：基于 e-graph 的异构存储架构高性能统一编译框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895279&idx=1&sn=cfe132ed7791edfd649561561681c72d&scene=21#wechat_redirect)**
- **[超优化驱动的数值计算库自动学习系统，实现数学原语高效挖掘与最高 2.2× 性能增益](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901822&idx=1&sn=dcdd94c36d76c146d74b9c86bb4ff13e&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)**
- **[平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)**

:::

> 本研究提出的 :mdnice-bold-italic[EqiForge，是一款基于相等饱和技术的张量程序超优化器。它通过统一的中间表示，将高层张量表达式与分块计算纳入同一套表达体系，依靠可组合的相等规则，直接从张量表达式推导出 FlashAttention 风格的融合内核实现]。同时通过早期压缩技术在候选程序构建完成前剪枝冗余分支，结合子图组合策略将搜索能力扩展到更大规模的计算图。

实验结果显示，在各类张量程序基准测试中，**EqiForge 相对于各配置下的最快基线，取得了 1.32 倍的几何平均加速比，最高加速可达 2.74 倍**。

![文章插图](/articles/2026-10-03-flashattention-e-graph-eqiforge-1-8ee80a6f/images/02.webp)

::: image-caption
图 7：多样张量程序的性能（以 EQUIFORGE 为基准归一化）。数值越高性能越好，标签显示相对于最快有效基线的几何平均加速比。× 表示搜索超时或数值无效结果。该图对比了 EQUIFORGE 与 PyTorch、TVM、Mirage、Trinity 等基线在归一化、MLP、LoRA、注意力等多种张量程序上的性能。结果显示 EQUIFORGE 在 21 种配置中的 18 种上超越了最快基线，几何平均加速比达 1.32 倍，最高在 RMSNorm-MLP 场景下达到 2.74 倍。其性能优势主要来自代数重写与算子融合的联合优化，例如将归一化因子移到矩阵乘法之后，实现多层算子的全融合，减少全局内存访问。仅在 LayerNorm-GEMM 场景下略逊于基线，原因是分核策略带来了额外的启动开销与中间存储。整体结果验证了等式饱和超优化在多样张量负载下的普适性能提升。
:::

其生成的注意力内核，**在解码阶段最高比 FlashAttention 快 1.87 倍，预填充阶段性能接近 FlashAttention 水平。**

![文章插图](/articles/2026-10-03-flashattention-e-graph-eqiforge-1-8ee80a6f/images/03.webp)

::: image-caption
图 8：多头注意力在解码与预填充阶段的性能（以 EQUIFORGE 为基准归一化）。数值越高性能越好，标签显示相对于最快有效基线的几何平均加速比。该图聚焦多头注意力场景，对比了 EQUIFORGE 与 FlashAttention、Trinity、Neptune 等专用注意力实现的性能。解码阶段 EQUIFORGE 全面超越所有基线，最高加速比达 1.87 倍，优势来自其通过通道分块与重复计算换并行的策略，在小批量解码场景下获得了更高的并行度；预填充阶段 EQUIFORGE 与 FlashAttention 性能接近，仅有 3%-6% 的差距，差距主要来自 Triton 代码生成的调度优化空间，而非算法结构本身。结果表明 EQUIFORGE 不仅能自动推导出与人工优化的 FlashAttention 相当的分块注意力实现，还能针对解码场景探索出更优的并行策略，体现了超优化的策略探索能力。
:::

在多种 Transformer 层的优化中，**EqiForge 发现的新实现大幅超越 torch.compile**，其中 :mdnice-bold-italic[QK 归一化 MLA 加速 3.16 倍，mHC 加速更是达到 5.84 倍。]

![文章插图](/articles/2026-10-03-flashattention-e-graph-eqiforge-1-8ee80a6f/images/04.webp)

::: image-caption
图 9：EQUIFORGE 发现的层结构：(a) QK 归一化多头潜注意力，(b) 流形约束超连接层。每个蓝色虚线框包围一个内核。SK₂₀表示 20 次 Sinkhorn 迭代。图 (a) 展示逻辑数据流：缓存的逆 RMS 键统计量 ρ 在重写前对键进行缩放，在重写后对 logits 进行缩放；辅助状态输出与分块部分归约已省略。该图呈现了 EQUIFORGE 在两种 Transformer 变体层上探索出的最优结构，分别代表了代数重构与跨组件融合两类典型优化方向。QK 归一化 MLA 的优化侧重高层数学形式的重写，通过调整计算顺序大幅减少冗余计算与内存访问；mHC 的优化侧重执行层面的跨边界融合，通过算子合并消除中间内存流量。两个案例分别实现了 3.16 倍与 5.84 倍的加速，远超传统算子级优化的收益，证明了等式饱和超优化不仅能优化底层执行策略，还能探索高层算法层面的重构，为 Transformer 架构的性能优化提供了自动化的探索工具。
:::

::: block-1
:mdnice-highlight[这项工作的核心洞察在于，将代数重构与执行优化放到同一个相等饱和搜索空间中，通过类型化统一 IR、可组合规则体系、早期压缩三大设计]，既保留了联合优化的完整机会，又有效控制了搜索复杂度，为张量编译的超优化落地提供了完整可行的技术路径。
:::

## 本文目录

- 一、张量编译的核心困境：代数与执行的断层
- 二、EqiForge 整体架构：相等饱和的全流程设计
- 三、统一 IR：打通代数与执行的表达基础
- 四、相等规则体系：构建可组合的优化空间
- 五、可扩展的程序提取：从搜索空间到最优实现
- 六、实验评估：全场景性能验证与案例分析
- 七、总结与行业启示

![文章插图](/articles/2026-10-03-flashattention-e-graph-eqiforge-1-8ee80a6f/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[ASPLOS'26 高频语义等价指令挖掘，实现2.6倍加速、93%面积节省！用E-Graph反合一为AI芯片打造乐高式指令！寻找高频数据级并行指令模式](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899424&idx=1&sn=e2a2f9e94eeeefe1858087e094837b18&scene=21#wechat_redirect)**
- **[CGO'25 杀死阶段顺序问题：基于 MLIR 的等式饱和持久化 e-graph，让等价信息贯穿编译全程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897696&idx=1&sn=d963ddf62be45438f2c7c6804545af64&scene=21#wechat_redirect)**
- **[端到端 LLM 编译器 nncase：基于 e-graph 的异构存储架构高性能统一编译框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895279&idx=1&sn=cfe132ed7791edfd649561561681c72d&scene=21#wechat_redirect)**
- **[超优化驱动的数值计算库自动学习系统，实现数学原语高效挖掘与最高 2.2× 性能增益](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901822&idx=1&sn=dcdd94c36d76c146d74b9c86bb4ff13e&scene=21#wechat_redirect)**
- **[性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)**
- **[从形式化推理到 GPU 加速！Lean4 + TileLang 张量程序超优化器，让 Attention 快 4.08 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901134&idx=1&sn=f689cf4d266599f5873320f98ae75d4b&scene=21#wechat_redirect)**

:::
