关键词：DeepSeek、**昇腾 950**、:mdnice-bold-italic[TileLang]、**稀疏注意力**、专家并行

> 2026 年 9 月 30 日，DeepSeek 通过官方渠道宣布正式开源面向华为昇腾算力平台的基础设施组件，:mdnice-bold-italic[涵盖 TileLang 高级语言编译工具、计算库与分布式通信库。这批组件与此前面向英伟达平台的开源组件一一对应，标志着 DeepSeek 在英伟达生态之外，系统性地把整套训练与推理软件栈移植到了国产算力平台上。]

DeepSeek 在开源公告中阐述了这一路线的底层逻辑：建立新一代**自主可控的 GPU 软件生态，首要的事情是建立一个通用的、编程简单的、同时能达到硬件性能上限的高级语言**。

::: block-1
TileLang 正是在这样的背景下诞生的。相对于 CUDA，TileLang 的编程更简单，能提高开发效率、简化代码逻辑；:mdnice-highlight[相对于其它同类高级语言，TileLang 的编程模型可以充分发挥芯片特性，达到硬件性能上限。]

这条路线首先在英伟达成熟平台上得到验证，如今已承载了 DeepSeek V4 系列模型训练中大部分算子的实现。**本次开源的 TileLang 昇腾版本对昇腾 Ascend C 底层指令进行封装，提供高级语言的编程方式，同时不损失硬件性能**。:mdnice-highlight[据 DeepSeek 披露，其训练中用到的每一个 TileLang 算子，在昇腾平台上都有对应的高性能实现。]
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**
- **[算子进化框架 AgenticCANN 用六层知识体系，打通低语料硬件昇腾 910B NPU 的算子生成](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904464&idx=1&sn=623e8a0de0165ee11d5687218a6be035&scene=21#wechat_redirect)**

:::

本次开源还包括昇腾平台的核心计算与通信组件：DeepGEMM Ascend 加速通用矩阵运算，DeepEP Ascend 提供大规模跨设备通信，TileKernels 提供常规向量计算和访存算子，FlashMLA 提供稀疏注意力算子，DeepSelect 实现高效的数据筛选。**DeepSeek 称，在多项关键测试用例中，这些组件的计算与通信性能已接近硬件上限。**

值得注意的是双方的合作深度。在面向昇腾平台的研发过程中，**华为团队给予了大力支持，双方共同推进基于昇腾 950 的 128 卡超节点方案，并对计算与通信进行了深度优化。**

## 本文目录

- 一、TileLang 昇腾后端：封装 Ascend C，性能不打折
  - 1.1 架构与编译管线
  - 1.2 与 Torch NPU 的性能对比
- 二、DeepGEMM Ascend：矩阵计算摸到硬件天花板
  - 2.1 稠密 GEMM：硬件利用率最高 99.8%
  - 2.2 推理场景 FP8 GEMM
  - 2.3 MoE 分组 GEMM 与 MQA Logits
  - 2.4 MegaMoE 与 mHC Prenorm
- 三、DeepEP Ascend：超节点上的专家并行通信
  - 3.1 通信带宽实测
- 四、FlashMLA：稀疏注意力正式登陆昇腾
  - 4.1 昇腾稀疏注意力算子
  - 4.2 算子背后的优化技术
- 五、TileKernels 与 DeepSelect：向量算子与数据筛选
  - 5.1 TileKernels：一套算子，两个后端
  - 5.2 DeepSelect：为稀疏注意力定制的 TopK
- 生态观察

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/01.webp)

## 一、TileLang 昇腾后端：封装 Ascend C，性能不打折

TileLang 昇腾后端是这次开源的基石，仓库地址为 TileLang[1]，昇腾相关代码与文档位于 tilelang/ascend 目录[2]下。

### 1.1 架构与编译管线

> 昇腾后端复用 TileLang 的共享前端，在此基础上叠加昇腾专用的 lowering、调度、同步与代码生成，最终面向昇腾 950 NPU（dav-3510）的 Cube 核与 Vector 核生成可执行程序。

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/02.webp)

::: image-caption
图 1：TileLang 昇腾 950 后端的整体架构。用户以 Ascend Dialect 编写 kernel，经共享前端完成 DSL 解析与 IR 构建后，进入昇腾编译管线，依次完成分形布局（Fractal Layout）推断、自动调度（AutoSchedule）、自动同步，再经 Cube/DMA lowering、FP4 存储打包等昇腾专用 pass，最终生成 Ascend C 设备代码，由毕昇编译器编译并在昇腾 950 NPU 上执行。
:::

从编程体验看，昇腾后端把 Ascend C 开发中最繁琐的部分接管了过去。**开发者只需声明 buffer、拷贝、tile 尺寸与循环结构，TPipe/TQue 的建立、DMA 调用、线程映射、barrier 放置、flag 的分配与配对等全部由编译器自动完成**。昇腾平台允许在单个 kernel 中混合 SIMT 与 SIMD 两种向量编程模式，Cube 核负责矩阵乘，Vector 核负责逐元素计算，编译器默认处理核间任务分配、调度与同步。

开发方式上，昇腾 950 kernel 通常采用持久化（persistent）执行模型，由固定数量的 block 迭代处理所有 work tile。

### 1.2 与 Torch NPU 的性能对比

> 官方在昇腾 950DT 上将 TileLang 与 Torch NPU 进行了对比评测，覆盖 BF16 GEMM、FP8 量化转换与 GQA 反向传播三个算子，每个算子四种形状。计算密集型算子以 TFLOP/s 计量，访存密集型算子以有效带宽 GB/s 计量。

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/03.webp)

::: image-caption
图 2：TileLang 与 Torch NPU 在昇腾 950DT 上的对比实测。左图 GEMM（BF16 输入、FP32 输出）：TileLang 在 1K³、4K³、8K³、4K×7K×2K 四种形状下分别达到 222.0、425.8、431.4、425.0 TFLOP/s，对应 Torch NPU 为 197.2、413.0、426.3、405.0 TFLOP/s。中图 FP8 Cast（FP32 转 E4M3FN，group 128）：TileLang 有效带宽达 2316.4 至 3294.7 GB/s，而 Torch NPU 仅为 402.8 至 637.4 GB/s，差距约 4.8 至 6.8 倍。右图 GQA Backward（BF16、batch 1、单 KV head、head dim 128）：TileLang 在四种序列长度与 Q:KV 比例下达到 374.0 至 422.4 TFLOP/s，全面高于 Torch NPU 的 339.6 至 397.4 TFLOP/s。
:::

这组数据说明两个问题。

- 其一，**在 GEMM 这类厂商库已深度优化的算子上，TileLang 能稳定持平或小幅超越 Torch NPU**，印证了其高级语言抽象不损失性能的设计目标。
- 其二，**在 FP8 量化转换这类相对冷门、厂商库覆盖不足的算子上，TileLang 的优势被放大到数倍**，:mdnice-bold-italic[这正是自研 kernel 工具链的价值所在。]

## 二、DeepGEMM Ascend：矩阵计算摸到硬件天花板

> DeepGEMM Ascend[3] 是 DeepGEMM[4] 在昇腾平台的移植版，于 2026 年 9 月 30 日首次发布，支持昇腾 950 设备。:mdnice-bold-italic[它与 DeepGEMM 完全 API 兼容，支持 BF16、FP8、FP4 GEMM、MQA logits 与 MegaMoE 算子，用户更换平台后无需修改调用代码。]

实现上，**DeepGEMM Ascend 对昇腾的矩阵乘加原语（MAD）做了轻量抽象，隐藏了分形矩阵布局、对齐约束、地址计算等底层细节，同时广泛采用昇腾特有的优化手段，包括稀疏数据加载与基于协程的流水线。** 官方给出的评价是：代码轻量，且在多种矩阵形状上能达到硬件极限性能。以下数据均在昇腾 950DT（CANN 9.20）上冷 L2 实测，形状覆盖 DeepSeek 模型系列的推理与训练负载。

### 2.1 稠密 GEMM：硬件利用率最高 99.8%

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/04.webp)

::: image-caption
表 1：DeepGEMM Ascend 稠密 GEMM 实测。在 DeepSeek 模型典型的大形状下，四种数据类型的算力利用率均在 98% 以上，BF16 更是达到 99.8%，几乎不存在软件层的性能损耗。（数据来源：DeepGEMM-Ascend 官方仓库[5]）
:::

### 2.2 推理场景 FP8 GEMM

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/05.webp)

::: image-caption
表 2：FP8 GEMM 在推理（M=128，对应小批量 decode）与训练/大批量（M=4096）两类工况下的表现。小 M 时算子转为带宽 bound，最高有效带宽 2456 GB/s；大 M 时算力稳定在 790 至 861 TFLOPS 区间。（数据来源：DeepGEMM-Ascend 官方仓库[6]）
:::

### 2.3 MoE 分组 GEMM 与 MQA Logits

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/06.webp)

::: image-caption
表 3：面向 DeepSeek MoE 专家的 M-Grouped GEMM（FP8×FP4 输入、BF16 输出）。在 4 与 8 专家分组下算力均保持在 818 至 862 TFLOPS，接近 FP8×FP4 稠密 GEMM 的硬件上限，说明分组调度引入的开销极小。（数据来源：DeepGEMM-Ascend 官方仓库[7]）
:::

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/07.webp)

::: image-caption
表 4：MQA Logits 算子，服务于 DeepSeek 稀疏注意力中的 Lightning Indexer 打分环节。该 kernel 的瓶颈不在计算而在 FIX 管线，实测将 FIX 管线打到了 99% 的利用率。（数据来源：DeepGEMM-Ascend 官方仓库[8]）
:::

### 2.4 MegaMoE 与 mHC Prenorm

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/08.webp)

::: image-caption
表 5：MegaMoE 融合算子实测，单 kernel 融合 EP dispatch、两个分组 GEMM、SwiGLU 与 combine，在 EP8、top-k=6 并带一个共享专家的配置上以 8 卡均值统计。大规模配置（384 专家、16384 token）下算力达 846.3 TFLOPS，同时维持 103.3 GB/s 的通信带宽，体现了计算通信融合的成熟度。（数据来源：DeepGEMM-Ascend 官方仓库[9]）
:::

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/09.webp)

::: image-caption
表 6：HC Prenorm GEMM，服务于 DeepSeek mHC（Manifold-Constrained Hyper-Connections）模块。该算子为写带宽 bound，M=8192 时有效带宽 3463 GB/s，几乎打满 HBM 写带宽。其 kernel 由 TileLang 实现，也是 TileLang 昇腾后端能力的直接佐证。（数据来源：DeepGEMM-Ascend 官方仓库[10]）
:::

## 三、DeepEP Ascend：超节点上的专家并行通信

> DeepEP Ascend[11] 是面向昇腾 NPU 的高性能通信库，提供 MoE dispatch/combine 所需的专家并行 all-to-all 操作，支持 FP8 dispatch 与延迟 epilogue，另有流水线并行、Bucket 集合通信与 Engram 远端内存访问等原语处于开发中。

其公开 buffer API 与英伟达版 DeepEP[12] 对齐，Ascend C kernel 基于 HCCL/HCOMM、UBMEM 与 URMA 实现，并通过 DeepJIT[13] 在运行时编译。

### 3.1 通信带宽实测

> 测试在昇腾 950DT（CANN 9.2.0）上进行，使用超节点外部 Clos 网络，配置为每 rank 16384 token 容量、隐藏维度 7168、256 专家 top-6 路由、32 个 AI 核。Dispatch 采用 FP8，combine 采用 BF16。

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/10.webp)

::: image-caption
表 7：DeepEP Ascend 不同 EP 规模下的 all-to-all 带宽区间（覆盖所有 rank）。EP32 以内 dispatch 持续带宽达到物理载荷带宽上限的约 90% 至 95%；EP64 以上与 combine 仍在优化中，combine 额外受本地归约开销与 URMA 的 HBM 争用影响，官方称华为后续固件更新将缓解这一争用。（数据来源：DeepEP-Ascend 官方仓库[14]）
:::

有一个使用层面的细节需要留意：上述成绩采集自华为提供给 DeepSeek 的 PoC 版 HDK 外加手工配置，并非公开版本。华为面向 Atlas 850E 的 Q3 商用 HDK 预计 2026 年 10 月中旬发布，届时可通过 华为 Atlas 850E 软件下载页[15]申请获取，其中包含跑满带宽所需的配置，官方建议以其作为公开部署基线。

## 四、FlashMLA：稀疏注意力正式登陆昇腾

> FlashMLA[16] 是 DeepSeek 的高性能注意力算子库，支撑 DeepSeek-V4.1[17] 在英伟达 GPU 与昇腾 NPU 上的推理。其 9 月 30 日更新包含三部分内容。

### 4.1 昇腾稀疏注意力算子

本次发布了昇腾 950 上的稀疏注意力 prefill 与 decoding 算子：

- prefill 最高 410 TFlops，**达到硬件理论峰值的 95%；**
- decoding 最高 360 TFlops**，达到硬件理论峰值的 83%。**

同时，融合 Q-norm、Q-RoPE、注意力、O-RoPE 与 FP8 转换的 fused kernel 在 decoding 场景下性能提升了约 10% 至 15%。随算子一同发布的还有一篇技术报告《Ascend Sparse Attention Forward 算法与优化技术简析》，系统介绍了算法与优化细节。

这次更新同时是一次破坏性变更：新版移除了对英伟达 Hopper 架构与 DeepSeek V3、V3.2、V4.0 等前代模型的支持，并更改了 FP8/FP4 KV cache 格式，与旧版本不兼容。需要运行旧模型或旧 KV cache 格式的用户需切换到 指定的历史 commit[18]。

### 4.2 算子背后的优化技术

> 从官方技术报告看，昇腾版稀疏注意力的设计围绕一个目标展开：让 Cube 核的矩阵乘单元时刻打满。几个关键决策颇具代表性。

**其一是 1C2V 的核间配比**。昇腾 950 每个 AI 核包含一个 Cube 核与两个 Vector 核，理想情况下若能为每个 Cube 核只配一个 Vector 核，剩下的 Vector 核便可跑其他任务。但实测发现此路不通：单个 Vector 核最多只能挂起 16 个 MTE2 拷贝请求，decoding 时反量化速度跟不上 Cube 计算，且 FixPipe 搬运带宽不足。最终方案是为每个 Cube 核配两个 Vector 核，分摊拷贝发射、反量化、搬运与 softmax 的开销。

**其二是 KV 的 layout 转换路径**。稀疏注意力按 token 粒度从显存 gather KV，无法在搬运途中完成 ND 到 NZ 的 layout 转换，而 AI 核内部的访存合并又要求源与目的都连续。最终采用的路径是先把 KV 拷入 UB，用 SIMD 向量函数完成 ND2NZ 转换，再送入 L1。配套的高速 ND2NZ 实现通过对齐加载、跨步写回、指针自增与一次处理一对 256B 元素等技巧，打满了 UB 的读写吞吐。

**其三是 skip-scale**。只要 online softmax 当前使用的 max 与真实 max 的差距不超过阈值（实现中取 6），就跳过本轮输出的 rescale 与累加。这一技术在昇腾上收益更大，因为 NPU 上做 rescale 需要经 FixPipe 把数据从 L0C 搬到 UB 再向量累加，代价高昂。配套的跨核信号传递通过 SS buffer 与 cross core flag 完成。此外，工程上还处理了指令缓存容量有限（Cube 核 32K、Vector 核 16K）带来的 icache miss 问题，以及把每个 tensor 切两份分置两个 L1 bank 以打满带宽的配平技巧。

## 五、TileKernels 与 DeepSelect：向量算子与数据筛选

### 5.1 TileKernels：一套算子，两个后端

> TileKernels[19] 是基于 TileLang[20] 实现的高性能算子库，包含数十个深度优化的 kernel，覆盖大模型训练与推理中的混合专家路由、Engram 门控、FP8/FP4 量化（含融合 SwiGLU 的量化算子）、流形超连接（mHC）、RoPE 与随机数生成等操作。**官方称大多数算子性能接近硬件的计算吞吐或内存带宽上限，且全部算子已在内部训练与推理任务中实战使用。**

其 9 月 30 日的更新是新增昇腾后端：继英伟达路径之后，同一套 Python API 现在可以在英伟达 GPU（SM90/SM100）与昇腾 950 NPU 上运行，后端在运行时自动选择。对使用者而言，这意味着上层代码不需要感知底层芯片的差异。

### 5.2 DeepSelect：为稀疏注意力定制的 TopK

> DeepSelect[21] 是为 DeepSeek 稀疏注意力（DSA[22]，应用于 V3.2、V4、V4.1）与采样器定制的高性能 TopK 算子，支持英伟达 CUDA 与昇腾双平台。

与原生 torch.topk 相比，整体可实现 2 至 20 倍加速。它针对两类工况做了专门优化：

- 一是 Lightning Indexer 场景（bfloat16 输入、topk 最大 4096、重点优化 DeepSeek V4 使用的 topk=512）
- 二是采样场景（float32 输入、约 128K 词表）。9 月 30 日的更新正式加入了昇腾 NPU 支持

![文章插图](/articles/2026-09-30-deepseek-tilelang-6ad30a1c/images/11.webp)

::: image-caption
图 3：DeepSelect 与 torch.topk 在昇腾平台上的对比（bfloat16、topk=512，按 batch 6、512、4096 分三组子图，横轴为词表大小）。TopK 不涉及浮点运算，因此以有效内存带宽计量。大 batch 下 DeepSelect 随词表增大持续爬升，batch 4096、词表 1M 时达到约 3.5 TB/s，而 torch.topk 在各配置下均停留在 0.15 TB/s 左右，两者差距最高约 20 倍。（图片来源：DeepSelect 官方仓库[23]）
:::

## 生态观察

> 把这次开源放进更大的图景里，可以看到三条清晰的线索。

第一条是完整性。**DeepSeek 开源的是从编程语言（TileLang）、矩阵计算（DeepGEMM Ascend）、通信（DeepEP Ascend）、注意力（FlashMLA）到配套算子（TileKernels、DeepSelect）的全栈组件，且每一件都与英伟达版 API 对齐**。对昇腾平台的开发者而言，拿到的是一套可以直接复用的生产级基础设施。

第二条是性能口径。从稠密 GEMM 的 99.8% 硬件利用率，到 EP32 以内 90% 至 95% 的通信带宽利用率，再到稀疏注意力 prefill 95% 的峰值占比，**这批组件给出的都是相对硬件理论上限的占比数据**。这说明软件栈已不再是昇腾硬件性能发挥的瓶颈。

第三条是示范意义。DeepSeek 在公告中明确表示**，希望 TileLang 昇腾版本能为更多 AI 芯片建立高可用软件生态起到示范作用**。一门高级语言同时支撑英伟达与昇腾两大平台并被生产环境验证，这一范式若能成立，后续其他国产芯片接入同一套生态的成本将大幅降低。而双方围绕昇腾 950 的 128 卡超节点所开展的联合优化，也意味着这套软件栈的下一个战场是更大规模的集群形态。

附本次开源项目链接：

- TileLang[24]
- DeepGEMM Ascend[25]
- DeepEP Ascend[26]
- TileKernels[27]
- FlashMLA[28]
- DeepSelect[29]

参考资料

[1] TileLang: *https://github.com/tile-ai/tilelang*

[2] tilelang/ascend 目录: *https://github.com/tile-ai/tilelang/blob/main/tilelang/ascend/README.md*

[3] DeepGEMM Ascend: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[4] DeepGEMM: *https://github.com/deepseek-ai/DeepGEMM*

[5] DeepGEMM-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[6] DeepGEMM-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[7] DeepGEMM-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[8] DeepGEMM-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[9] DeepGEMM-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[10] DeepGEMM-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[11] DeepEP Ascend: *https://github.com/deepseek-ai/DeepEP-Ascend*

[12] DeepEP: *https://github.com/deepseek-ai/DeepEP*

[13] DeepJIT: *https://github.com/deepseek-ai/DeepJIT*

[14] DeepEP-Ascend 官方仓库: *https://github.com/deepseek-ai/DeepEP-Ascend*

[15] 华为 Atlas 850E 软件下载页: *https://support.huawei.com/enterprise/zh/atlas-computing/atlas-850e-pid-265102853/software*

[16] FlashMLA: *https://github.com/deepseek-ai/FlashMLA*

[17] DeepSeek-V4.1: *https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash*

[18] 指定的历史 commit: *https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7*

[19] TileKernels: *https://github.com/deepseek-ai/TileKernels*

[20] TileLang: *https://github.com/tile-ai/tilelang*

[21] DeepSelect: *https://github.com/deepseek-ai/DeepSelect*

[22] DSA: *https://github.com/deepseek-ai/DeepSeek-V3.2-Exp*

[23] DeepSelect 官方仓库: *https://github.com/deepseek-ai/DeepSelect*

[24] TileLang: *https://github.com/tile-ai/tilelang*

[25] DeepGEMM Ascend: *https://github.com/deepseek-ai/DeepGEMM-Ascend*

[26] DeepEP Ascend: *https://github.com/deepseek-ai/DeepEP-Ascend*

[27] TileKernels: *https://github.com/deepseek-ai/TileKernels*

[28] FlashMLA: *https://github.com/deepseek-ai/FlashMLA*

[29] DeepSelect: *https://github.com/deepseek-ai/DeepSelect*

::: related-title
**相关推荐**
:::

::: related-links
- **[16 万颗华为昇腾 950DT！DeepSeek 内蒙古建全球最大国产 AI 算力集群](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905491&idx=1&sn=22f1ee0495b7fd923c81101a29f55835&scene=21#wechat_redirect)**
- **[用 TileLang 在 AMD Radeon 上榨干 GPU 性能：RDNA4 GEMM 118.5 TFLOPS 持平 rocBLAS，Dequant达 93 TFLOPS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901949&idx=1&sn=4520024a563e4c5821d9dcf98d83c8ac&scene=21#wechat_redirect)**
- **[从 CUDA 迁移到昇腾 910 部署千亿 MoE 血泪实录！附国产卡移植指南](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905761&idx=1&sn=45d3b4f00af42d486ba27df7847fa840&scene=21#wechat_redirect)**

:::
