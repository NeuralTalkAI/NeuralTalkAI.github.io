关键词：**GEMM**、:mdnice-bold-italic[Neural Processing Unit]、**Hardware Acceleration**、Deep Learning、**XDNA Architecture**

> 在 AI 算力竞赛白热化的今天，从云端数据中心到边缘终端，专用硬件加速器已成为深度学习（DL）应用落地的核心支撑。而深度学习 workload 的核心——通用矩阵乘法（GEMM），其性能直接决定了模型推理与训练的效率。

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/01.webp)

- **Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen™ AI NPUs**
- https://arxiv.org/pdf/2512.13282
- 6000 字，阅读 20 分钟，**播客 24 分钟**

相关推荐

- **[AMD Composable Kernel: 基于 Tile 的编程范式与张量坐标变换，大幅提升端到端Transformer性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886669&idx=1&sn=6ac0368282305a37ad508fcf0e9a65e0&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[7.46倍性能提升！结果重用GEMM加速器Transitive Array，LLaMA模型实现3.97倍加速与1.65倍能耗降低](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890742&idx=1&sn=981ff2fb5277b9aa18d08e0c875b2909&scene=21#wechat_redirect)**

> AMD 推出的 Ryzen AI 处理器搭载 XDNA 架构 NPU，凭借高能效比成为边缘 AI 加速的热门选择，但由于缺乏 FPGA fabric支持，其 GEMM 优化面临独特挑战。

近期，来自德州大学奥斯汀分校与 AMD 的团队在论文《Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen™AI NPUs》中，提出了一套**跨 XDNA/XDNA2 两代 NPU 的系统性 GEMM 优化方案**，实现了 int8 精度下 6.76 TOPS（XDNA）和 38.05 TOPS（XDNA2）的业界顶尖性能。

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/02.webp)

::: image-caption
图3：所提出的 GEMM 多级分块方案（a），以及XDNA（b）和XDNA2（c）上的GEMM映射策略。其中（a）包含四级分块：内层为AIE API支持的r×s×t分块，单核层为适配L1内存的m_ct×k_ct×n_ct分块，多核层为覆盖整个NPU阵列的(m_ct·m_rows)×k_mt×(n_ct·n_cols)分块，最外层为输入输出矩阵的M×K×N分块；（b）中XDNA采用4×4对称映射，（c）中XDNA2采用4×8非对称映射，均通过广播实现数据复用，并借助内存Tile暂存输入块以提升DRAM连续访问效率
:::

本文将深入拆解该工作的核心创新、技术细节与实验成果，带大家看清 AMD NPU 性能突破的底层逻辑。

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/03.webp)

## 本文目录

- 一、背景与核心挑战
  - 1. GEMM：深度学习的“算力基石”
  - 2. AMD NPU 的架构特性与优化难点
- 二、相关工作：现有优化方案的局限
- 三、系统性优化方法论与实现细节
  - 创新点 1：计算-内存平衡的解析建模优化
  - 创新点 2：基于多维张量寻址的无预分片 GEMM 实现
  - 创新点 3：NPU-DRAM 数据移动的 BD 重配置优化
- 四、实验结果：性能突破与验证
  - 1. 单核心性能验证
  - 2. NPU 阵列性能突破
  - 3. 多尺寸 GEMM 性能稳定性
- 五、总结与展望

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/04.webp)

## 一、背景与核心挑战

### 1. GEMM：深度学习的“算力基石”

深度学习模型（如 CNN、Transformer）的核心计算的本质是大量矩阵乘法与累加操作，GEMM 操作占比超过 90%。因此，GEMM 的优化效率直接决定了整个 DL 任务的运行速度。

### 2. AMD NPU 的架构特性与优化难点

AMD Ryzen AI NPU 基于 XDNA 架构，分为 XDNA（20 核，4×5 阵列）和 XDNA2（32 核，4×8 阵列）两代，架构如下图所示：

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/05.webp)

::: image-caption
图 1：锐龙 AI（Ryzen™AI）神经网络处理单元（NPU）的架构图：该架构呈模块化可扩展设计，核心是 2D 计算 tile（CompTiles）阵列（含带 SIMD datapath 的 VLIW 核心与 L1 内存），下方设一行存储 tile（MemTiles，提供 L2 内存），最下方是接口 tile（ShimTiles）以连接 DRAM，依赖 DMA 引擎与命令处理器实现数据移动和任务调度
:::

NPU 采用模块化架构，通过 DMA 引擎实现 L1→L2→DRAM 的数据移动，命令处理器负责任务调度与硬件配置。两代 NPU 均支持 int8、bf16 精度，XDNA2 额外增加 bfp16 硬件支持，理论峰值算力提升至 50 TOPS（XDNA 为 10 TOPS）。

与传统 Versal 平台（含 FPGA fabric）不同，:mdnice-bold-italic[AMD NPU 缺乏 FPGA 带来的额外存储层次与灵活配置能力，导致 GEMM 优化面临三大核心挑战：]

::: block-1
- 数据移动瓶颈：DRAM 与 NPU 间的数据传输效率直接限制性能，需最大化连续内存访问；
- 计算与内存的平衡：单核心算力提升可能导致内存访问成为瓶颈，反之亦然；
- 跨代兼容性：需一套方案适配 XDNA/XDNA2 两代架构，同时保证高性能。
:::

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/06.webp)

::: image-caption
图2：AMD锐龙AI NPU中跨内存层级的数据移动：输入缓冲区A（a）与输出缓冲区C（b）。其中（a）展示输入缓冲区A从主内存（DRAM）出发，经ShimTile的内存映射到流（MM2S）通道传输至L2内存Tile（MemTile），再进一步传输到L1内存，供计算Tile（CompTile）的核心使用；（b）展示输出缓冲区C从L1内存传输至L2 MemTile，再经流到内存映射（S2MM）通道写入DRAM，整个过程由硬件锁单元实现DMA与核心、DRAM间的数据同步
:::

## 二、相关工作：现有优化方案的局限

> GEMM 优化已在多款硬件平台得到广泛研究，但针对 AMD NPU 的方案仍存在明显不足，具体对比如下：

| 相关工作 | 目标平台 | 核心思路 | 关键局限 |
| --- | --- | --- | --- |
| MaxEVA | Versal（含 FPGA） | 分区 GEMM 到多个 AIE 核心，20% 核心用于加法树归约 | 依赖 FPGA fabric，核心利用率仅 80%，无法迁移到 AMD NPU |
| GAMA | Versal（含 FPGA） | 利用级联接口传输部分累加结果 | 级联阻塞导致 7% 性能损失，不适配无 FPGA 架构 |
| 基于 XDNA 的基础实现 | XDNA NPU | 任务型编程模型，简单映射 GEMM 到核心 | 性能低下（int8 仅 5.04 TOPS，bf16 1.95 TOPS），未优化数据移动 |

::: block-1
本文工作的**核心突破：在无 FPGA 支持的情况下，通过系统级优化实现跨两代 NPU 的高性能 GEMM**，:mdnice-bold-italic[同时解决数据移动、计算-内存平衡、跨代兼容三大问题。]
:::

## 三、系统性优化方法论与实现细节

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/02.webp)

::: image-caption
图3：所提出的 GEMM 多级分块方案（a），以及XDNA（b）和XDNA2（c）上的GEMM映射策略。其中（a）包含四级分块：内层为AIE API支持的r×s×t分块，单核层为适配L1内存的m_ct×k_ct×n_ct分块，多核层为覆盖整个NPU阵列的(m_ct·m_rows)×k_mt×(n_ct·n_cols)分块，最外层为输入输出矩阵的M×K×N分块；（b）中XDNA采用4×4对称映射，（c）中XDNA2采用4×8非对称映射，均通过广播实现数据复用，并借助内存Tile暂存输入块以提升DRAM连续访问效率
:::

> 本文的创新并非孤立的技术技巧，而是一套贯穿“**建模-优化-实现** ”的完整框架，其核心包含以下三大创新点：

### 创新点 1：计算-内存平衡的解析建模优化

团队发现，:mdnice-bold-italic[GEMM 性能受计算能力与内存带宽的双重约束]，且二者呈**反比关系**：增大计算核心的 tile 尺寸（提升算力利用率）会导致内存访问次数增加（内存瓶颈加剧），反之亦然。因此，**关键是找到“计算时间 ≈ 内存时间”的平衡 point。**

#### 1.1 单核心优化模型

通过解析建模定义核心约束，优化 $m_{ct}$（核心 M 维度 tile 大小）、$k_{ct}$（核心 K 维度 tile 大小）、$n_{ct}$（核心 N 维度 tile 大小）三个参数：

- 计算周期：$C_{comp} = \frac{m_{ct} \cdot k_{ct} \cdot n_{ct}}{eff \cdot peak_{MACs}}$$eff$ 为核心效率，$peak_{MACs}$ 为核心峰值算力
- 数据传输周期（A 矩阵）：$CA_{comm} = \frac{m_{ct} \cdot k_{ct} \cdot ty(A)}{DMA_{BW}}$$ty(A)$ 为 A 矩阵数据类型字节数（int8=1，bf16=2），$DMA_{BW}$ 为 DMA 带宽
- 内存约束：$2 \cdot m_{ct} \cdot k_{ct} \cdot ty(A) + 2 \cdot k_{ct} \cdot n_{ct} \cdot ty(B) + m_{ct} \cdot n_{ct} \cdot ty(C) \leq 63KB$L1 内存共 64KB，预留 1KB 栈空间。这里对输入矩阵 A 和 B 的 tile 都乘以 2 是因为 L1 缓存采用双缓冲（Double-Buffering） 设计。

::: block-1
**优化目标**：通过整数规划（IP）同时最大化 MAC 数量（$m_{ct} \cdot k_{ct} \cdot n_{ct}$）和最小化输出 tile 大小（$m_{ct} \cdot n_{ct}$），在满足内存和带宽约束下寻找最优解，提升计算效率并减少存储冲突。

要特别说明的是，这两个目标在 IP 求解中是联合考虑的，并非“先最大化再最小化”，而是通过参数约束和双目标优化找到一个均衡解。
:::

#### 1.2 系统级优化模型

扩展单核心优化模型至整个 NPU 阵列时，需结合 DRAM 带宽约束建立系统级性能模型，核心逻辑与参数定义需严格匹配硬件架构特性：

##### 1. 关键参数与 DRAM 访问量计算

- **DRAM 访问量（A 矩阵）**：$A_{mem} = \frac{M \cdot K \cdot N \cdot ty(A)}{n_{ct} \cdot n_{cols}}$

| 技术参数 | XDNA | XDNA2 |
| --- | --- | --- |
| NPU 代际 | XDNA | XDNA2 |
| CompTile 总数（核心数） | 20 | 32 |
| 原始 2D 阵列（rows×cols） | 4×5 | 4×8 |
| 实际 GEMM 映射阵列（rows×cols） | 4×4 | 4×8 |
| $m_{rows}$（行数） | 4 | 4 |
| $n_{cols}$（列数） | 4 | 8 |
| 映射调整原因 | 由于XDNA架构的最后一列（第5列）物理上不存在ShimTile（接口Tile），无法实现该列核心与DRAM的数据交换，因此GEMM映射时仅使用前4列，形成逻辑上的4×4对称阵列 | 全阵列可用，无 ShimTile 限制 |

- **DRAM 访问量（B 矩阵）**：$B_{mem} = \frac{M \cdot K \cdot N \cdot ty(B)}{m_{ct} \cdot m_{rows}}$
- **DRAM 访问量（C 矩阵，写操作）**：$C_{mem} = M \cdot N \cdot ty(C)$

##### 2. 核心时间模型

- **总计算时间**：$T_{comp} = \frac{2 \cdot M \cdot K \cdot N}{eff \cdot peak_{TOPS}}$
- **总内存时间**：$T_{mem} = \frac{A_{mem} + B_{mem} + C_{mem}}{DRAM_{BW}}$其中 $DRAM_{BW}$ 为 NPU 可利用的有效 DRAM 带宽，作者实测 XDNA 约 15 GB/s，XDNA2 约 50 GB/s。

##### 3. 优化流程

::: block-1
1. 基于单核心优化结果设定初始参数（$m_{ct}, k_{ct}, n_{ct}, eff$），确保初始状态为内存瓶颈（$T_{comp} < T_{mem}$）；
2. 每次迭代中，固定 $k_{ct}$ 按 $s$ 的倍数减小（$s$ 为 AIE API 支持的 tile 尺寸参数）；
3. 针对当前 $k_{ct}$，通过整数规划（IP）最大化 $m_{ct} \cdot n_{ct}$，同时满足 L1 内存容量、DMA 带宽约束；
4. 实测当前参数的 GEMM 性能，若性能持续提升则继续迭代；当性能首次下降时，前一次迭代的参数即为最优平衡解（此时 $T_{comp}$ 开始大于 $T_{mem}$，进入计算瓶颈）。
:::

### 创新点 2：基于多维张量寻址的无预分片 GEMM 实现

> 传统 GEMM 优化需在 DRAM 中对矩阵进行预分片（tiling），才能适配 NPU 核心的输入格式，导致与 DL 张量库（如 GGML）集成复杂。

**本文利用 AMD NPU 的 DMA 多维寻址特性**，实现**实时（on-the-fly）张量转换**，矩阵在 DRAM 中保持标准行/列主序，无需预处理。具体转换流程如下图所示：

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/07.webp)

::: image-caption
图 4 | 矩阵A的动态 DMA 变换。该图展示矩阵 A 从 DRAM 传输到 NPU 过程中的多阶段 DMA 变换：ShimTile 通过 3D 寻址将行优先的 $m_{ct}×K$ 块拆为 $m_{ct}×k_{mt}$ 块，MemTile 经 3D+4D 寻址进一步拆为 $m_{ct}×k_{ct}$ 块，最终 CompTile 通过 3D 寻址转为核心所需的 tiled 布局，确保 DRAM 中矩阵保持标准行优先格式
:::

观察上图可以发现：

::: block-1
1. ShimTile（3D 寻址）将 DRAM 中 $m_{ct} \times K$ 的行主序矩阵拆分为 $m_{ct} \times k_{mt}$ 子块；
2. MemTile（4D 寻址）进一步拆分为 $m_{ct} \times k_{ct}$ 子块；
3. CompTile（3D 寻址）最终转换为核心所需的 $r \times s \times t$ tile 格式。
:::

整个过程无需预分片，无缝兼容标准张量布局。其中需要强调的技术细节：

- MemTile 支持 4D 寻址，CompTile/ShimTile 支持 3D 寻址，通过多级转换满足核心输入要求；
- 针对 int8/bf16 等小精度数据，利用 AIE 核心的 `shuffle` 指令实现细粒度数据重排，适配列主序存储的矩阵 B。

### 创新点 3：NPU-DRAM 数据移动的 BD 重配置优化

> NPU 通过 ShimTile 与 DRAM 通信，**每个 ShimTile 仅支持 16 个缓冲描述符（Buffer Descriptor, BD）**，而复杂 GEMM 任务在细粒度分块下需定义的 DMA 传输远多于16个，直接顺序重配置 BD 会导致 DMA 通道空闲，造成性能瓶颈。

:mdnice-bold-italic[缓冲描述符（BD）是硬件指令，用于告知 DMA 引擎一次数据传输的所有细节，包括源/目标地址、数据量、同步锁等]。在 GEMM 的复杂数据流中，为每个输入/输出 tile 的传输都需要一个独立的 BD。当 GEMM 维度（`M, K, N`）很大时，所需的 BD 数量很容易超过每个 ShimTile 16 个的硬件上限。

::: block-1
本文提出 **“BD 重配置与 DMA 传输重叠”** 的优化方案，核心思想是：**在 DMA 持续传输数据的同时，动态复用和重写已完成的 BD，避免因等待 BD 配置而导致的流水线停滞**。
:::

其操作流程如下图所示，并可通过以下步骤细化理解：

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/08.webp)

::: image-caption
图 5 | 最外层（第四层）GEMM 分块层级的简化视图
:::

> 优化方案分步解析

1. **初始化与队列填充**：在 GEMM 计算开始前，向每个 ShimTile 的任务队列提交 **15 个 BD**，上图中 ShimTile 0 的 `BD_A0, BD_B0, BD_C0, BD_A4, BD_B4, BD_C1...`），占满 16 个槽位中的 15 个，预留 1 个冗余。这些 BD 描述了首批需要传输的矩阵 `A`, `B` 的输入块和矩阵 `C` 的输出块。
2. **重叠执行与关键同步**：DMA 引擎立即开始处理队列前部的 BD，将数据在 DRAM 与 NPU 间移动，同时 NPU 核心开始计算。:mdnice-bold-italic[关键的优化点在于同步逻辑：命令处理器仅需顺序等待 输出矩阵 C 的 BD 完成] （如 `BD_C0`）。一旦某个 `C` 的 BD 完成，意味着生成该输出块所对应的所有输入块（如 `BD_A0`, `BD_B0`）的传输也必然已完成。此时，这一组相关的 BD（两个输入，一个输出）即可被安全回收。
3. **动态重配置与稳态维持**：上述三个 BD 被回收后，**立即在原地被重写（reconfigure）** 为下一组待传输的数据块描述符，并重新插入任务队列尾部。例如，`BD_A0`, `BD_B0`, `BD_C0` 被更新为 `BD_A8`, `BD_B8`, `BD_C2` 的描述信息。此过程与仍在进行的其他 DMA 传输和核心计算**完全重叠**。
4. **达成高效稳态**：通过这种机制，系统可以维持一个**持续的流水线**。在稳态下，任务队列中始终有接近满载的 BD（约 15 个）等待执行或正在执行，确保了 DMA 引擎的带宽被持续、充分地利用，消除了因 BD 数量限制带来的空闲等待。

## 四、实验结果：性能突破与验证

> 实验基于两款 mini PC 搭建：XDNA（Minisforum UM790 Pro，Ryzen 9 7940HS）和 XDNA2（ASRock 4×4 Box，Ryzen AI 7 350），均配置 DDR5-5600 DRAM。

### 1. 单核心性能验证

单核心优化后的吞吐量与内存利用率，如下表是单核心 GEMM 性能结果所示：

| 设备 | 精度（输入-输出） | 内核尺寸 $m_{ct} \times k_{ct} \times n_{ct}$ | 吞吐量（MACs/cycle） | L1 核心内存占用（KB） |
| --- | --- | --- | --- | --- |
| XDNA | int8-int8 | 64 × 232 × 64 | 233.0 | 62.0（97%） |
| XDNA | int8-int32 | 48 × 280 × 48 | 192.0 | 61.5（96%） |
| XDNA | bf16-bf16 | 64 × 104 × 64 | 112.6 | 60.0（94%） |
| XDNA2 | int8-int8 | 64 × 232 × 64 | 450.6 | 62.0（97%） |
| XDNA2 | int8-int32 | 48 × 280 × 48 | 384.0 | 61.5（96%） |
| XDNA2 | bf16-bf16 | 48 × 152 × 48 | 158.1 | 61.5（96%） |

**上表展示了两代架构的 L1 内存利用率高达 94%-97%，接近饱和**；XDNA2 单核心吞吐量是 XDNA 的 1.8-2.0 倍，侧面体现架构升级带来的算力提升。

### 2. NPU 阵列性能突破

跨两代 NPU 的阵列级性能（矩阵 B 为列主序）如下表所示（选取最优平衡 kernel，数据整合自表 2 和表 3，具体见下文）：

| 设备 | 精度（输入-输出） | 内核尺寸 $m_{ct} \times k_{ct} \times n_{ct}$ | 峰值计算 TOPS | 实际 NPU TOPS | GEMM 尺寸 $M \times K \times N$ |
| --- | --- | --- | --- | --- | --- |
| XDNA | int8-int8 | 112 × 112 × 112 | 6.80 | 6.52 | 4032 × 4032 × 4032 |
| XDNA | bf16-bf16 | 96 × 56 × 96 | 3.19 | 3.12 | 4224 × 4032 × 4224 |
| XDNA2 | int8-int8 | 144 × 72 × 144 | 39.52 | 37.35 | 4032 × 4320 × 4608 |
| XDNA2 | bf16-bf16 | 112 × 48 × 96 | 15.81 | 14.52 | 4032 × 4224 × 4608 |

1. XDNA int8 实际性能 6.52 TOPS（最高 6.76 TOPS），比此前工作提升 34%；
2. XDNA2 int8 实际性能 37.35 TOPS（最高 38.05 TOPS），接近理论峰值的 95%；
3. bf16 精度下，XDNA 提升 62%，XDNA2 达到 14.71 TOPS，满足高精度 DL 任务需求。

更多详细数据：

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/09.webp)

::: image-caption
表 2：XDNA 在不同数据类型下（矩阵 B 为列优先存储）的两个最优方案评估。表格涵盖输入输出精度、核大小（𝑚ct×𝑘ct×𝑛ct）、参数乘积（𝑚ct・𝑛ct）等关键信息，最优方案平衡计算与内存开销，如 int8-int8 类型中 112×112×112 核在 4032×4032×4032 矩阵规模下，实际 NPU 性能达 6.52 TOPS，有效提升 GEMM 效率
:::

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/10.webp)

::: image-caption
表3：XDNA2在不同数据类型下（矩阵B为列优先存储）的两个最优方案评估。表格涵盖输入输出精度、核大小（𝑚ct×𝑘ct×𝑛ct）、参数乘积（𝑚ct·𝑛ct）等指标，最优方案平衡计算与内存，如int8-int8类型中144×72×144核在4032×4320×4608矩阵规模下，实际NPU性能达37.35 TOPS，还借助NPU多维张量寻址提升DRAM带宽利用率，优化GEMM效率
:::

### 3. 多尺寸 GEMM 性能稳定性

> 通过 **roofline 曲线，验证不同 GEMM 尺寸下的性能稳定性**，横轴为算术强度（OPs/Byte），纵轴为性能（TOPS）。

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/11.webp)

::: image-caption
图 7：XDNA 不同精度下的 roofline 性能曲线：（a）int8-int8精度，（b）int8-int16精度，（c）bf16-bf16精度
:::

观察上图不同精度下 roofline 性能曲线：

- 当算术强度超过 1000 后，性能趋于稳定（计算-内存平衡）；
- 列主序存储的矩阵 B 比行主序平均性能高 4.4%-4.8%，验证了连续内存访问的重要性。

![文章插图](/articles/2026-01-01-amd-ryzen-ai-npu-gemm-ef7288aa/images/12.webp)

::: image-caption
图 8：XDNA2 不同精度下的 roofline 性能曲线：（a）int8-int8精度，（b）int8-int16精度，（c）bf16-bf16精度
:::

观察上图，发现：

- XDNA2 相比 XDNA 性能稳定性更优，列主序 B 矩阵比行主序高 8.7%-25.2%；
- int8 精度下性能波动仅 5%，远超行主序的 19%，**体现了多维张量寻址与 BD 优化的协同效果。**

## 五、总结与展望

> 本文通过三大核心创新，构建了跨 AMD XDNA/XDNA2 两代 NPU 的 GEMM 优化框架：

1. **计算-内存平衡的解析建模**：通过迭代优化找到性能最优平衡点，解决算力与带宽的反比约束；
2. **多维张量寻址的无预分片实现**：兼容标准张量布局，降低与 DL 库的集成复杂度；
3. **BD 重配置与 DMA 重叠**：突破 ShimTile BD 数量限制，最大化 NPU-DRAM 数据传输效率。

实验证明，该方案实现了业界顶尖的 GEMM 性能，为 Ryzen AI NPU 部署深度学习任务提供了关键技术支撑。:mdnice-bold-italic[未来工作可进一步探索 bfp16 精度的原生优化，以及将方法论扩展到矩阵-向量乘法（GEMV）等特殊场景。]

对于 AI 硬件开发者与研究者而言，本文的价值不仅在于性能突破，更在于**提供了一套“硬件特性-解析建模-系统优化”的完整思路** ——在专用加速器日益复杂的今天，只有深度结合硬件架构的系统性优化，才能真正释放算力潜力。
