关键词：GPU 内核自动生成、领域专用语言 (DSL)、大语言模型 (LLM)、性能优化、Softmax、Gemm

> 十年前，我们写 CUDA；五年前，我们用 Triton；今天，我们让 LLM 自己写 Triton。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/01.webp)

- DSL for GPU Kernels & Automatic Kernel Authoring with LLMs
- Tri Dao，https://tridao.me
- https://github.com/tridao
- 本文 5444 字，阅读需 18 分钟，播客 9 分钟如下

> 当大模型参数突破万亿、训练数据量呈指数级增长，GPU硬件算力的提升逐渐触达瓶颈，“如何在软件层面榨干硬件每一分性能”成为行业破局的关键。

过去，手写CUDA虽能逼近硬件峰值性能，却需付出“周级别”的开发成本；通用框架如PyTorch虽提升了生产力，却要牺牲10%-30%的性能。**这种“生产力与效率”的矛盾，在大模型时代愈发尖锐。**

::: block-1
Tri Dao作为AI系统领域的权威研究者，在其最新演讲“DSL for GPU Kernels & Automatic Kernel Authoring with LLMs”中，给出了一套颠覆性解决方案——**以领域专用语言（DSL）为桥梁，衔接“研究效率”与“硬件效率”，并借助大语言模型（LLM）实现GPU内核的自动生成。**

从Softmax算子的5层并行归约优化，到FlashAttention-4在Blackwell架构上的性能突破，再到LLM生成内核的“数据-评测-算法”三位一体体系，**他系统梳理了GPU内核开发从“人工手写”到“智能生成”的技术跃迁路径。**
:::

本文将基于Tri Dao的演讲内容，拆解DSL如何平衡性能与开发成本、LLM在 kernel 生成中的能力边界与突破方向，并给出从原型验证到极致优化的落地策略，为开发者呈现大模型时代GPU内核技术的全貌与未来趋势。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/02.webp)

## 本文目录

- 一、为什么现在必须重新思考 GPU Kernel？
- 二、DSL：把 “研究效率” 和 “硬件效率” 同时拉满
- 三、Softmax 的 5 层并行归约解剖
  - 1. Thread 级归约
  - 2. Warp 级归约
  - 3. Thread-block 级归约
  - 4. Cluster 级归约（H100 新增特性）
  - 5. Grid 级归约
- 四、Gemm：当硬件让 “矩阵乘法” 不再性感
- 五、FlashAttention-4：算法、指令、流水线的三重协同
- 六、LLM 写 Kernel：数据、评测、算法的三位一体
  - 6.1 数据集：构建 “PyTorch-Triton” 配对的黄金样本
  - 6.2 评测 ——KernelBench：给 LLM 的 “内核考试” 定标准
  - 6.3 算法：从有监督微调（SFT）到多轮强化学习（RL）
- 七、当前 LLM 的能力边界
- 八、面向 LLM 的下一代 DSL 长什么样？
  - 8.1 LLM 仅需编写的 “算法核心代码”（5 行左右）
  - 8.2 编译器自动填充的 “硬件细节代码”
- 九、落地建议：三阶段策略
- 结语：软件 3.0 的黎明

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/03.webp)

## 一、为什么现在必须重新思考 GPU Kernel？

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/04.webp)

::: image-caption
Scaling Laws：计算、算法、数据提升能力
:::

Tri Dao 在演讲开头给出了一个简洁的公式：**算力（Compute）↑ × 算法（Better Algorithms）↑ × 数据量（More Data）↑ = 模型能力（More Capabilities）↑**，这正是大模型领域核心的缩放定律（Scaling Laws），揭示了性能提升的底层逻辑。但随着硬件迭代，GPU 算力（FLOPS）与美元效率（FLOPS/Dollar）的提升逐渐减速，单纯依赖硬件升级的 “红利期” 正在消退 —— 这意味着我们必须在软件层面榨干硬件的每一滴性能，重新审视 GPU 内核的开发模式。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/05.webp)

::: image-caption
GPU 算力（FLOPS）与美元效率（FLOPS/Dollar）的提升逐渐减速
:::

过去十年，PyTorch 这类通用框架通过封装底层细节，把开发生产力推到极致，但其 “通用性” 也带来了性能损耗，通常会牺牲 10%–30% 的硬件峰值性能；而手写 CUDA 虽然能让性能逼近 100% 峰值，却需要开发者掌握复杂的硬件架构知识，开发周期长达 “周级别”，且难以适配不同 GPU 型号。这种 “生产力” 与 “硬件效率” 的矛盾，在大模型参数突破万亿、训练数据量指数级增长的今天，变得愈发尖锐。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/06.webp)

::: image-caption
本文第一部分 Kernels for AI：AI 内核、DSL、Softmax 等示例
:::

## 二、DSL：把 “研究效率” 和 “硬件效率” 同时拉满

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/07.webp)

::: image-caption
为何用 DSL：提升研究效率与硬件效率
:::

演讲中，Tri Dao 明确提出用**领域专用语言（DSL）** 作为破解 “生产力 - 效率” 矛盾的核心方案，其核心价值在于实现 “双向适配”：

::: block-1
- 向上，保留高层语义抽象，避免开发者陷入硬件细节，同时让 LLM 更容易理解和生成内核代码；
- 向下，精准暴露 GPU 的关键硬件细节（如线程层次、内存层级），使编译器能将代码映射到接近手写 CUDA 的性能水平。
:::

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/08.webp)

::: image-caption
PyTorch PT2：后端集成架构与组件，详见：https://pytorch.org/get-started/pytorch-2-x/
:::

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/09.webp)

::: image-caption
Triton 与 CUDA 对比：内存、调度等差异，详见：https://openai.com/index/triton/
:::

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/10.webp)

::: image-caption
Cute-DSL：嵌入 Python 的 Cutlass C++，详见：https://www.nvidia.com/en-us/on-demand/session/gtc25-s74639/
:::

为了直观展现 DSL 的优势，Tri Dao 以 Softmax 算子为案例 —— 这个看似简单的算子，因涉及全局归约和内存访问优化，实则是 GPU 性能调优的 “试金石”。通过对比不同实现方案的核心指标，DSL 的价值被清晰呈现：

| 方案 | 代码量 | 性能 (H100 bf16, M=32K) | 开发心智负担 |
| --- | --- | --- | --- |
| Torch.compile | 1 行（仅需添加装饰器） | ≈ 1× baseline（ baseline 为 PyTorch 原生实现） | 0（无额外学习成本） |
| Triton | 30 行左右 | 1.8×（相对 baseline 加速） | 中（需理解 Tile 划分） |
| Cute-DSL | 150 行左右 | 2.3×（相对 baseline 加速） | 高（需掌握内存层级） |

> 值得注意的是，DSL 的优势还体现在 “LLM 友好性” 上：:mdnice-bold-italic[良好的抽象设计让 LLM 能快速抓住内核逻辑的核心，减少因硬件细节混乱导致的生成错误 —— 这为后续 “LLM 自动生成内核” 奠定了基础。]

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/11.webp)

::: image-caption
其他 DSL：ThunderKittens、Mojo、TileLang
:::

## 三、Softmax 的 5 层并行归约解剖

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/12.webp)

::: image-caption
Cute-DSL：4 级线程与内存层次结构
:::

> Cute-DSL 的核心竞争力之一，在于其能**显式控制 GPU 的 5 级并行层次**，让开发者精准优化每一步计算与内存访问，这也是其在 Softmax 算子上实现 2.3× 加速的关键。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/13.webp)

::: image-caption
Softmax：Cute-DSL 异步内存复制
:::

具体来说，这 5 层并行归约的逻辑的如下：

### 1. Thread 级归约

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/14.webp)

::: image-caption
Softmax：Cute-DSL 线程级归约
:::

在单个线程的寄存器内完成局部数据的归约，通过`x.reduce(cute.ReductionOp.MAX, init_val=float("-inf"))`直接操作寄存器数据，避免内存访问开销；

### 2. Warp 级归约

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/15.webp)

::: image-caption
Softmax：Cute-DSL warp 级归约
:::

利用 GPU 的 Warp（32 线程一组）特性，通过`shuffle_sync_bfly`（蝶式交换）指令实现线程间数据交换，无需共享内存（SMEM），彻底消除同步延迟；

### 3. Thread-block 级归约

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/16.webp)

::: image-caption
Softmax：Cute-DSL 线程块级归约
:::

先让每个 Warp 将归约结果写入共享内存，再由指定线程读取共享内存数据，发起第二次 Warp 归约，实现 Block 内的全局结果聚合；

### 4. Cluster 级归约（H100 新增特性）

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/17.webp)

::: image-caption
Softmax：Cute-DSL 集群级归约
:::

借助 H100 的 “分布式共享内存”（Distributed Shared Memory），将多个 Thread-block 的共享内存打通，实现跨 Block 的归约，适合超大规模数据场景；

### 5. Grid 级归约

当输入行宽远超单 Grid 处理能力时，启动第二次 Kernel，对多个 Cluster 的归约结果做最终聚合，完成全局归约。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/18.webp)

::: image-caption
Softmax：从线程到集群的归约流程
:::

Tri Dao 在演讲中特别强调了一个 “反直觉” 的优化经验：当 Softmax 的行宽（N）≤ 4K 时，Cluster 级归约反而会因跨 Block 同步拖慢性能；此时若将`tv_layout`（线程 - 数据映射布局）设计为 “一个 Warp 负责一行数据”，可直接砍掉 Block 级和 Cluster 级同步，性能反而能再提升 15%–20%，最终逼近硬件极限。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/19.webp)

::: image-caption
Softmax 性能：Torch、Triton、Cute-DSL 对比。https://github.com/Dao-AILab/quack/blob/main/quack/softmax.py
:::

## 四、Gemm：当硬件让 “矩阵乘法” 不再性感

> 矩阵乘法（Gemm）是 AI 模型的 “计算核心”，过去十年，手写 Gemm 优化一直是 GPU 开发者的 “核心技能”。但 Tri Dao 在演讲中指出：:mdnice-bold-italic[随着硬件架构升级（如 Hopper 的 TMA、Blackwell 的 Persistent MMA），纯 Gemm 的性能已经 “卷无可卷”—— 甚至 DSL 实现都能追上专业优化库的水平。]

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/20.webp)

::: image-caption
Hopper 的 Gemm 性能：cuBLAS 与 Cute-DSL 对比。https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py，https://github.com/Dao-AILab/quack/blob/main/quack/dense_gemm_sm90.py
:::

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/21.webp)

::: image-caption
Hopper Gemm：PingPong 架构与流水线
:::

在 Hopper 架构的 H100 上：

::: block-1
- cuBLAS（NVIDIA 官方优化库）的 bf16 Gemm 性能已逼近 800 TFLOPS；
- 而 Cute-DSL 通过 “Ping-Pong Epilogue” 技术（双缓冲存储 + 异步 TMA 内存预取 + 计算 - 访存重叠），实现了与 cuBLAS 持平甚至略超的性能 —— 这在 5 年前几乎是不可想象的，因为当时手写 CUDA 都难以追上 cuBLAS 的优化水平。
:::

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/22.webp)

::: image-caption
Blackwell 的 Gemm 性能：cuBLAS 与 Cute-DSL。https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py，https://github.com/Dao-AILab/quack/blob/main/quack/dense_gemm_sm100.py
:::

到了 Blackwell 架构的 B200 上，同样的 “追赶剧本” 再次上演：Cute-DSL 的 bf16 Gemm 性能达到 1620 TFLOPS，而 cuBLAS 为 1600 TFLOPS，两者差距缩小到误差范围（±2%），意味着 DSL 在纯 Gemm 场景下已实现 “性能平权”。

> Tri Dao 由此提出一个关键结论：**纯 Gemm 已经不是性能竞争的核心，真正的战场是 “Gemm+X” 场景** （X 指 Swiglu、LayerNorm、激活函数等算子）。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/23.webp)

::: image-caption
Gemm+Swiglu：Hopper 上性能对比
:::

以 Gemm+Swiglu 为例，Cute-DSL 通过 “Epilogue Fusion”（尾端融合）技术，将 Gemm 的输出直接送入 Swiglu 计算，省去了一次全局内存（GMEM）写回操作，最终实现 7%–15% 的性能提升 —— 这正是通用库难以覆盖的 “定制化优化场景”，也是 DSL 的核心价值所在。

## 五、FlashAttention-4：算法、指令、流水线的三重协同

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/24.webp)

::: image-caption
FlashAttention4：Blackwell 上的流水线设计。https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha，https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py
:::

> 注意力机制（Attention）是大模型的 “灵魂”，而 FlashAttention 系列一直是注意力优化的标杆。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/25.webp)

::: image-caption
FlashAttention4：关键算法改进点
:::

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/26.webp)

::: image-caption
FlashAttention4 vs cuDNN：B200 性能。https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py
:::

在 Blackwell 架构上，FlashAttention-4（FA4）通过 “算法创新 + 硬件指令 + 流水线优化” 的三重协同，再次刷新了性能极限 —— 这也是 Tri Dao 演讲中 “软件 - 硬件协同优化” 的典型案例。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/27.webp)

::: image-caption
性能与投入权衡：各工具对比表。https://semianalysis.com/wp-content/uploads/2025/03/Blackwell-Programming-for-the-Masses-With-OpenAI-Triton-Phil-Tillet.pdf
:::

Blackwell 架构的硬件特性带来了一个关键矛盾：Tensor Core（矩阵计算单元）的峰值性能达到 8192 FLOPS / 周期，而 MUFU.EX2（指数计算单元）仅有 16 OPS / 周期，两者性能差距高达 512 倍，这意味着 Softmax 中的指数计算会成为注意力机制的性能瓶颈。

为解决这一矛盾，FA4 做了三件关键优化：

::: block-1
1. **在线 Softmax 算法**：将传统 Softmax 的 “计算 max → 计算 exp → 求和 → 归一化” 四趟扫描，压缩为 “边读取数据边计算（max, sum (exp)）” 的一趟扫描，直接减少 90% 的输出 rescaling 开销，大幅降低内存访问次数；
2. **软件模拟指数计算**：利用 MUFU.EX2 指令的高精度特性，通过指令组合模拟出符合需求的指数函数，将指数计算的吞吐翻倍，缓解 MUFU 单元的性能瓶颈；
3. **2-CTA MMA 设计**：在反向传播阶段，将原本需要写入 L2 缓存的原子加操作，迁移到共享内存中完成，减少 50% 的 L2 带宽占用，进一步降低延迟。
:::

这些优化的效果极为显著：在 32K 长序列场景下，FA4 的性能达到 1620 TFLOPS，而 cuDNN（NVIDIA 官方深度学习库）仅为 900 TFLOPS 左右，FA4 比 cuDNN 快 1.8×，彻底解决了长序列注意力的性能痛点。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/28.webp)

::: image-caption
工具推荐：优先 Torch compile，再 Triton 等
:::

## 六、LLM 写 Kernel：数据、评测、算法的三位一体

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/29.webp)

::: image-caption
本文第二部分 AI for Kernels：注意力优化、LLM 生成内核等
:::

> 让 LLM 自动生成高效 GPU 内核，是 Tri Dao 演讲的核心目标之一。要实现这一目标，需要解决三个关键问题：**高质量数据、科学评测标准、高效训练算法** —— 三者共同构成 “LLM 内核生成” 的技术基石。

### 6.1 数据集：构建 “PyTorch-Triton” 配对的黄金样本

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/30.webp)

::: image-caption
数据集：PyTorch 与 Triton 内核对（10K-100K）：https://huggingface.co/datasets/GPUMODE/pytorch_scrape_inductor_data
:::

> LLM 的生成能力高度依赖训练数据的质量与规模。Tri Dao 团队采用的数据集主要来自两部分：

- **GPU-MODE 开源数据集**：包含 18K 组（PyTorch 原生代码，Triton 优化代码）配对样本，覆盖 Elementwise、Reduction、Gemm 等常见算子，数据格式为 Parquet，可直接用于 LLM 的有监督训练；
- **手写 Triton 内核清洗集**：从开源仓库中筛选出 2K 个手写 Triton 内核，经过许可证合规性检查（确保可商用）和性能验证（排除低效率实现），作为 “高质量范例” 补充到数据集中。

Tri Dao 特别强调：数据集的核心挑战不是 “数量”，而是 “质量”—— 低效率或错误的内核代码会严重误导 LLM 的学习，因此必须通过人工筛选和性能基准测试，确保每个样本的 “正确性” 和 “高效性”。

### 6.2 评测 ——KernelBench：给 LLM 的 “内核考试” 定标准

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/31.webp)

::: image-caption
KernelBench：评估 LLM 生成内核的正确性
:::

> 为客观评估 LLM 生成内核的能力，Tri Dao 团队提出了**KernelBench 评测体系**，包含 “正确性” 和 “性能” 两个核心关卡，模拟真实工业场景的需求：

- **正确性关卡**：给定 PyTorch 原生算子代码，让 LLM 生成对应的 Triton/CUDA 内核；随后用 100 组随机输入数据测试，要求生成内核的输出与 PyTorch 原生代码的输出 “bit-wise 一致”（无精度损失），否则判定为失败；
- **性能关卡**：在 A100/H100 两种主流 GPU 上，对通过正确性测试的内核进行 10 次性能测试，取中位数加速比；只有当加速比 > 1.0（即比 PyTorch 原生实现更快）时，才算通过性能测试。

这种 “先正确、后高效” 的评测逻辑，确保了 LLM 生成的内核不仅能 “跑通”，还能 “跑快”，符合工业界的实际需求。

### 6.3 算法：从有监督微调（SFT）到多轮强化学习（RL）

> 为提升 LLM 的内核生成能力，Tri Dao 团队尝试了两种核心训练算法，且都取得了超出预期的效果：

#### **有监督微调（SFT）**

基于上述数据集训练的 8B 参数模型 KernelLLM，在 KernelBench 的 “pass@1”（一次生成即通过测试）指标上，击败了 671B 参数的 DeepSeek-R1 模型 —— 这表明 “领域高质量数据” 比 “模型参数量” 更能提升特定任务的性能

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/32.webp)

::: image-caption
算法：KernelLLM 监督微调结果：https://huggingface.co/facebook/KernelLLM
:::

#### **多轮强化学习（Multi-Turn RL）**

32B 参数模型 Kevin-32B 通过 “8 步环境反馈 + 思维遮蔽（Thought Masking）” 机制，在训练过程中不断修正生成错误：每生成一轮内核，就用 KernelBench 测试并反馈 “错误点”（如内存访问越界、性能不达标），LLM 基于反馈调整生成逻辑。最终，Kevin-32B 的 “pass@16”（16 次生成内通过测试）指标从 23% 提升到 91%，且生成的内核性能达到人类手写基线的 1.56×。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/33.webp)

::: image-caption
算法：Kevin-32B 的强化学习结果
:::

## 七、当前 LLM 的能力边界

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/34.webp)

::: image-caption
LLM 生成内核：简单内核佳，复杂待提升
:::

> 尽管 LLM 在 kernel 生成上取得了显著进展，但 Tri Dao 在演讲中也坦诚地指出了其**能力边界** —— 通过 KernelBench 的测试结果，可清晰划分 LLM 的 “擅长” 与 “不擅长” 领域：

| 任务难度 | 能否正确生成 | 能否生成最优性能 |
| --- | --- | --- |
| 逐元素算子（如 Add、Relu）/ 基础归约（如 Sum、Max） | ✅ | ✅ |
| 中阶内存受限任务（如 TMA 异步访存、双缓冲） | ❌ | ❌ |
| 高阶计算受限任务（如 Tensor Core 流水线、Cluster 级同步） | ❌ | ❌ |

Tri Dao 进一步分析了背后的核心原因：LLM 擅长捕捉 “算法骨架”（如 Softmax 的归约逻辑、Gemm 的矩阵乘法流程），但对 “硬件细节”（如寄存器分配、共享内存大小、流水线深度）的记忆和把控能力极弱 —— 这些细节需要精确到 “字节” 和 “周期”，超出了 LLM 的上下文理解范围；而编译器恰好相反，擅长优化硬件细节，却难以自主设计算法逻辑。

由此，Tri Dao 提出一个关键洞察：**未来的最优解不是 “让 LLM 替代编译器”，而是 “LLM 负责高层算法设计 + 编译器负责底层硬件优化” 的协同模式**，需要设计新的 DSL 抽象来衔接两者。

## 八、面向 LLM 的下一代 DSL 长什么样？

> 基于 “LLM - 编译器协同” 的思路，Tri Dao 在演讲中给出了 “CuTe DSL 2.0” 的设计草图 —— 核心是 “最大限度简化 LLM 的输入，将复杂细节交给编译器”。具体来说，下一代 DSL 的代码分为两部分：

### 8.1 LLM 仅需编写的 “算法核心代码”（5 行左右）

```
# LLM 只需定义核心逻辑：循环、计算、流水线控制for k_block in range(k_block_cnt):   ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)  # 等待缓冲就绪   # 矩阵乘法计算，指定累加模式   cute.gemm(tiled_mma, tCtACC, tCrA[kphase_coord], tCrB[kphase_coord], accumulate=True)   ab_pipeline.consumer_release(ab_consumer_state)  # 释放缓冲，触发下一轮
```

### 8.2 编译器自动填充的 “硬件细节代码”

::: block-1
- 流水线深度（如双缓冲 / 三缓冲）；
- 共享内存（SMEM）的大小与布局；
- 寄存器双缓冲的分配策略；
- TMA（Tensor Memory Accelerator）的异步访存参数；
- 循环展开（Unroll）的次数。
:::

这些细节由编译器通过 “静态分析” 自动确定 —— 编译器会根据输入数据大小（如 Gemm 的 M/N/K 值）、GPU 型号（如 H100/B200），动态选择最优参数，无需 LLM 干预。这种设计既发挥了 LLM 的 “算法思维” 优势，又利用了编译器的 “硬件优化” 能力，实现了 “1+1>2” 的效果。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/35.webp)

::: image-caption
LLM 擅长算法推理：优化 Softmax 示例：https://chatgpt.com/share/68ab90b9-fd28-8012-aa20-5990499898a7
:::

## 九、落地建议：三阶段策略

针对不同的业务场景，Tri Dao 给出了 “从快速验证到极致优化” 的三阶段落地策略，帮助开发者平衡 “开发成本” 与 “性能收益”：

| 场景 | 首选方案 | 投入时间 | 预期性能（相对硬件峰值） | 核心优势 |
| --- | --- | --- | --- | --- |
| 原型验证 / 快速训练 | Torch.compile | 小时级 | 70%–90% | 仅需添加`@torch.compile`装饰器，零学习成本 |
| 推理性能瓶颈优化 | Triton | 天–周级 | 80%–95% | 支持 Tile 划分、自动内存合并，平衡效率与复杂度 |
| 极致推理 / 大规模训练 | Cute-DSL | 周–月级 | 100%（逼近硬件峰值） | 显式控制硬件层次，适配 Tensor Core 等高级特性 |

Tri Dao 特别强调：“不要一开始就陷入 Cute-DSL 的细节优化 —— 先用 1 行`torch.compile`试跑，90% 的场景下性能已足够；只有当`torch.compile`无法满足性能需求时，再用 Triton 或 Cute-DSL 做针对性优化。” 这种 “由粗到细” 的策略，能最大限度降低开发成本，同时保证业务性能达标。

## 结语：软件 3.0 的黎明

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/36.webp)

::: image-caption
需新抽象：LLM 与编译器能力互补
:::

> 十年前，我们靠手写 CUDA 挖掘 GPU 性能，开发者需要精通硬件架构；五年前，我们用 Triton 简化开发，通过 DSL 平衡效率与成本；今天，我们让 LLM 自动生成 Triton 代码，将开发者从 “硬件细节” 中解放出来，专注于 “算法逻辑”。

![文章插图](/articles/2025-08-27-hotchips-2025-pytorch-cuda-gpu-a2524691/images/37.webp)

::: image-caption
训练 LLM 写内核：需专家数据、抽象等
:::

Tri Dao 在演讲的最后指出：当 DSL 成功实现 “算法思维” 与 “硬件细节” 的解耦，GPU 内核开发正在从 “工程化” 转向 “Prompt 化”—— 大模型不仅在吞噬应用层，也在重塑系统层。这或许就是 “软件 3.0” 的黎明：开发者用自然语言描述需求，LLM 生成核心代码，编译器优化硬件细节，最终实现 “想法即产品” 的高效开发模式。

而这一切的起点，正是 DSL 与 LLM 的协同 —— 这也是 Tri Dao 想告诉我们的 “下一个十年的技术方向”。

交流加群请在 NeuralTalk 公众号后台回复：交个朋友
