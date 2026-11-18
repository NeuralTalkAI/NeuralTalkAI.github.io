**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：:mdnice-bold-italic[ThunderKittens]、**Vera Rubin NVL72**、GEMM、**NVFP4**、FP8

> 谁说天文学家小猫，不可爱呢🐱！Together AI 内核团队完成 ThunderKittens 算子库对 NVIDIA Vera Rubin NVL72 新平台适配，针对 NVFP4、FP8 精度 GEMM 矩阵乘法做深度优化。:mdnice-bold-italic[Vera Rubin 相比 Blackwell 架构 B200，算力、显存带宽、SM 数量大幅跃升，但直接移植 B200 原有内核，算力仅达到理论屋顶值 42%‑44%，存在明显数据供给瓶颈。]

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/01.webp)

- **To Infinity and Beyond: ThunderKittens Now on NVIDIA Vera Rubin NVL72!**
- https://www.together.ai/blog/to-infinity-and-beyond-thunderkittens-now-on-nvidia-vera-rubin-nvl72

::: related-title
**相关推荐**
:::

::: related-links
- **[雷霆小猫！从 20 到 500 TFLOP/s！ThunderKittens 2.0：打造更快的 GPU Kernels](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897675&idx=1&sn=880cf866507740797f20ae9d94bee3af&scene=21#wechat_redirect)**
- **[3.9 倍 RL rollout 吞吐提升！GT/CMU提出推理系统ThunderAgent，用“程序感知”调度解决KV缓存颠簸](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897457&idx=1&sn=c4cd21e8c80ba7ac0848710f39c0730c&scene=21#wechat_redirect)**
- **[Intelligence per Watt：斯坦福与 Together AI 实测 20+本地模型，88.7%的 AI 查询根本不需要上云](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905803&idx=1&sn=c912ba223bcd3de1233131540dd7b28f&scene=21#wechat_redirect)**

:::

> 作者拆解 Rubin 五大硬件新能力：64 字节 K 步 MMA 指令、扩容至 576 列的 Tensor Memory、最高 328KiB 超大共享内存、B 端收集器缓冲、A 操作数提前释放指令。

**团队基于这些硬件特性迭代内核**：启用双倍 K 步指令；采用 2×1 分块策略复用 B 矩阵数据削减访存流量；借助扩容共享内存加深流水线；搭配 B 端收集器、A 操作数提前释放、L2 缓存优化等手段调优。

::: block-1
:mdnice-bold-italic[在 16k 方阵 NVFP4 GEMM 任务上，配置 5 级共享内存流水线实现 22.239 PFLOPS 性能，最终优化后的 ThunderKittens 内核性能对标甚至超越 cuBLAS、CuTe‑DSL]。测试基于 CUDA13.4 与工程样卡，团队表示后续驱动版本还会进一步释放硬件潜力，同时也披露团队正在招聘内核开发人才。
:::

## 本文目录

- 一、起点：NVIDIA HGX B200 上的 GEMM
- 二、NVIDIA Vera Rubin 平台有哪些新特性？
  - 2.1 张量核心的 K 维度步长翻倍
  - 2.2 张量内存扩展至 576 列
  - 2.3 共享内存提升至 328 KiB
  - 2.4 B 侧收集器
  - 2.5 A 矩阵提前释放
- 三、构建 GEMM 内核
  - 3.1 拓宽指令步长
  - 3.2 减少读取字节数
  - 3.3 加深流水线
  - 3.4 最终优化细节
  - 3.5 性能结果
- 结语

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/02.webp)

::: image-caption
三只卡通小猫机器人带着望远镜和发光球体，身处 “薇拉・鲁宾天文台”，上方是猫形状的星座。
:::

## 一、起点：NVIDIA HGX B200 上的 GEMM

> 在介绍 Vera Rubin 的升级之前，我们先快速回顾一下 Blackwell 架构 GPU 上的 GEMM 实现，作为铺垫。

NVIDIA Blackwell 架构的第五代张量核心从根本上改变了 GEMM 的编程模型。Hopper 架构中的 `wgmma` 指令由整个线程束组统一发射，而 Blackwell 的 `tcgen05` 指令可由单一线程发射，只需一个小规模的生产者线程束即可驱动张量核心。累加器也从寄存器移至张量内存中，操作数可直接从共享内存读取，使得单次矩阵乘累加（MMA）运算可以跨两个流多处理器（SM）中的两个协作线程阵列（CTA）。

为了在 Blackwell 架构上达到有竞争力的性能，我们的 GEMM 内核做了如下优化：

- **启动线程块集群**，让每对 CTA 可通过张量内存加速器（TMA）多播共享操作数，将高带宽显存（HBM）的内存流量减半。
- **在集群内对线程束进行功能拆分**：加载线程束通过 TMA 将矩阵 A、B 载入共享内存，单个 MMA 线程束驱动张量核心运算，消费者线程束组将完成的累加器从张量内存写回 HBM。
- **采用持续运行模式**：当前计算块的输入正在流入时，上一个块的输出仍在写出，实现流水线重叠。

通过这些优化，我们取得了如下性能结果：

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/03.webp)

::: image-caption
HGX B200 上的 NVFP4 GEMM：ThunderKittens、cuBLASLt 与 CuTeDSL 在不同矩阵尺寸下的 TFLOPS 性能对比。横轴是矩阵尺寸 M=N=K，纵轴为算力 TFLOPS，对比 ThunderKittens、cuBLASLt、CuTeDSL 三种方案。矩阵规模较小时三者算力偏低，随着矩阵尺寸增大，算力整体逐步提升。在 M=N=K=16384 时，cuBLASLt 达到峰值 5385 TFLOPS，性能领先；继续增大到 32768，三者性能略有回落。整体来看，大矩阵场景下三种方案算力差距不大，小矩阵时 ThunderKittens 优势明显。
:::

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/04.webp)

::: image-caption
HGX B200 平台上 FP8 (E4M3) 精度 GEMM 运算性能，横轴为矩阵尺寸 M=N=K，纵轴算力单位 TFLOPS，对比 ThunderKittens、cuBLASLt、CuTeDSL 三种实现方案。矩阵规模较小时三者算力偏低，随着矩阵尺寸增大性能持续上升，在 M=N=K=16384 时 ThunderKittens 达到峰值 3422 TFLOPS，性能最优。矩阵尺寸继续增大至 32768，三种方案算力均出现回落。在全部测试规格下 ThunderKittens 算力基本领先，小矩阵场景下三种方案性能差距尤为明显。
:::

::: block-1
关于这些内核及其优化细节，可阅读我们此前的 Together 博客文章《ThunderKittens Now Optimized for NVIDIA Blackwell GPUs》[1] 或 ThunderKittens 2.0 版本发布[2] 了解更多。
:::

由于 Vera Rubin 保留了 Blackwell 的编程模型，我们原有的 GEMM 内核可以直接运行。**但直接在 Vera Rubin 上运行时我们发现，NVFP4 和 FP8 内核分别仅能达到理论峰值的 42.1% 和 44.4%——还有很大的优化空间！**

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/05.webp)

::: image-caption
Vera Rubin 平台上 TK B200 的 NVFP4 与 FP8 (E4M3) GEMM 算力，横轴为矩阵尺寸 M=N=K，纵轴单位 TFLOPS。紫色柱代表 NVFP4 GEMM，棕色柱为 FP8 (E4M3) GEMM。随着矩阵规模增大，两者算力先持续上升，在 M=N=K=16384 时 NVFP4 达到峰值 14741 TFLOPS，FP8 (E4M3) 同步达到 7763 TFLOPS。矩阵尺寸增大至 32768 时，两种精度算力均回落。所有测试规格下 NVFP4 性能均高于 FP8，大矩阵场景下性能差距更加明显。
:::

本文分为两部分：

- 首先介绍 Vera Rubin 中对 GEMM 至关重要的新特性，以及如何在 ThunderKittens 中使用它们；
- 随后我们将这些特性逐步整合到现有的 Blackwell NVFP4 内核中，将性能提升至 22 PFLOPS 以上，达到与 cuBLAS 和 CuTe DSL 相当的水平。

::: block-1
我们发现核心问题在于：:mdnice-bold-italic[尽管 Vera Rubin 的张量核心消费操作数的速度翻倍，但原有 Blackwell 内核的供给速度跟不上。要达到计算峰值，我们需要让每个分块充分复用片上已有的数据。]
:::

## 二、NVIDIA Vera Rubin 平台有哪些新特性？

> 对比官方规格参数，我们可以看到从 Blackwell 到 Vera Rubin 的各项提升：

| 指标 | NVIDIA HGX B200 | NVIDIA Vera Rubin NVL72 |
| --- | --- | --- |
| NVFP4 张量核心性能 | 9 PFLOPS / 每 GPU | 35 PFLOPS / 每 GPU |
| FP8 张量核心性能 | 4.5 PFLOPS / 每 GPU | 17.5 PFLOPS / 每 GPU |
| FP16 / BF16 张量核心性能 | 2.25 PFLOPS / 每 GPU | 4 PFLOPS / 每 GPU |
| 显存带宽 | 8 TB/s / 每 GPU | 22 TB/s / 每 GPU |
| 流多处理器（SM）数量 | 148 个 / 每 GPU | 224 个 / 每 GPU |
| 峰值功耗 | 1000W / 每 GPU | 2300W / 每 GPU |

对于编写高性能 GEMM 而言，我们重点关注以下几项新特性。

### 2.1 张量核心的 K 维度步长翻倍

> 回顾一下，`tcgen05.mma` 指令在 M×N×K 的分块上计算 `C = A@B + C`，每一步沿 K 维度消费固定字节数的数据。在 Blackwell 上，这一步长是 32 字节；:mdnice-bold-italic[而在 Vera Rubin 上，可提升至 64 字节。MMA 运算本身的周期数不变，因此 K 维度步长翻倍意味着同一条指令窗口内可以完成两倍的工作量。]

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/06.webp)

::: image-caption
Blackwell 的 32-byte K 步与 Rubin 的 64-byte K 步 MMA 矩阵分块。左侧 Blackwell 采用 32B 粒度，A、B 矩阵分块尺寸均为 32 字节，与 M×N 的 C 输出块配合完成矩阵乘。右侧 Rubin 升级为 64B K 步，A、B 分块尺寸扩大至 64 字节，单轮 MMA 可处理更长 K 维度。更大的 K 步粒度减少 K 方向循环迭代次数，降低指令开销与共享内存同步频次，提升单次 MMA 计算的数据吞吐量，充分发挥 Rubin 硬件更高的矩阵计算理论峰值性能。
:::

在 ThunderKittens 中，我们通过为现有 mma 操作新增一个模板参数来支持该特性：

```
mma_ABt    (...);   // Blackwell 默认：32 字节 K 步长mma_ABt<64>(...);   // Vera Rubin：64 字节 K 步长
```

### 2.2 张量内存扩展至 576 列

> Blackwell 首次引入了张量内存的概念——一个 128 通道 × 512 列 × 32 bit 的存储空间，张量核心可直接对其读写。在 Vera Rubin 中，该空间扩展至 576 列，额外增加了 32 KiB 的张量内存可用空间。

*注意：这些额外的列仅可通过 `.exclusive` 限定符访问，这是 PTX 9.4 新增的特性，用于确保每个 SM 上仅有一个活跃的张量内存分配。非独占分配的容量上限仍为 512 列，且必须是 2 的幂。*

在 ThunderKittens 中，用户可通过张量内存分配器的模板参数指定独占分配，来使用这部分空间：

```
template<int _nblocks_per_sm, int _ncta, bool _managed = true, bool _exclusive = false>struct tensor_allocator { .... }tensor_allocator<1, C::CLUSTER_SIZE, false> tm; // Blackwell 默认：512 列tensor_allocator<1, C::CLUSTER_SIZE, false, true>  tm; // Rubin：最高 576 列
```

### 2.3 共享内存提升至 328 KiB

> Hopper 和 Blackwell 架构提供 228 KiB 的共享内存，而 Vera Rubin 引入了超大共享内存模式，可动态提升至 328 KiB。这是一个主机端的配置项，可通过如下方式调用：

```
CUfunction function = nullptr;cudaGetFuncBySymbol(&function,reinterpret_cast<const void*>(kernel));cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,                   CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
```

### 2.4 B 侧收集器

> **Blackwell 引入了收集器缓冲区的概念——一个小型的 MMA 暂存缓冲区，可以暂存一个 A 分块，这样下一条指令可直接从缓冲区读取，无需再从共享内存加载**。Vera Rubin 将该功能扩展到了 B 分块，通过 `.collector::b::*` 实现。

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/07.webp)

::: image-caption
两个 MMA 矩阵运算的数据复用与缓冲区管理机制，B 矩阵仅读取一次（FILL）即可重复使用（LASTUSE），无需二次读取。具体来说，MMA1执行$A_0\times B_0$计算，完成后丢弃$A_0$，通过FILL操作将$B_0$写入collector buffer缓存。随后MMA2执行$A_1\times B_0$，直接读取缓存中的$B_0$，这是$B_0$的最后一次使用，运算结束后释放该缓存。$B_0$仅在MMA1阶段加载一次，MMA2不再重复读取，依靠缓冲区复用B矩阵分块，减少重复访存，降低内存带宽压力，提升GEMM内核的执行效率。
:::

为了利用该特性，我们为每个 MMA 的操作数标注四个标签之一，描述其对收集器缓冲区的操作：

- **FILL**：从共享内存读取操作数并锁存到缓冲区
- **USE**：从缓冲区读取操作数
- **LASTUSE**：从缓冲区读取并释放该缓冲区
- **DISCARD**：默认值，不使用缓冲区锁存

这些标签是复用的权限说明，而非保证。**也就是说，即使标注了复用权限，张量核心仍可能重新加载矩阵。**

现在两侧操作数都可以驻留在收集器缓冲区中，我们可以尝试新的计算模式。例如在 2×2 的分块中，启用双侧收集后，4 次 MMA 运算的操作数读取次数从 8 次减少到 5 次。

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/08.webp)

::: image-caption
2x2 MMA 矩阵乘法的分块流水线调度方案，在 512 端口周期与 512 计算周期之间取得平衡。该方案将矩阵切分为 A、B 面板，计算 C 的分块 C [0][0]、C [0][1]、C [1][0]、C [1][1]。计算 C [0][0] 时加载 A0、B0；完成后丢弃 B0，保留 A0 用于 C [0][1]，同步填充 B1。C [0][1] 计算结束后保留 B1，加载 A1 计算 C [1][1]；最后复用 A1，丢弃 B1 完成 C [1][0]。通过复用 A、B 分块数据，减少重复加载，实现数据复用与流水线重叠，降低共享内存带宽开销，提升 MMA GEMM 内核整体执行效率。
:::

在 ThunderKittens 中，我们通过如下方式暴露该功能：

```
mma2_ABt_chunk<64, false, false, collector::FILL,collector::DISCARD>(C[0][0], a0, b0, ...);mma2_ABt_chunk<64, false, false, collector::LASTUSE, collector::FILL   >(C[0][1], a0, b1, ...);mma2_ABt_chunk<64, false, false, collector::FILL,    collector::LASTUSE>(C[1][1], a1, b1, ...);mma2_ABt_chunk<64, false, false, collector::LASTUSE, collector::DISCARD>(C[1][0], a1, b0, ...);
```

### 2.5 A 矩阵提前释放

> `tcgen05.commit` 指令会在 MMA 运算完成后，通过内存屏障发出信号，通知生产者可以复用该阶段的槽位。

:mdnice-bold-italic[PTX 9.4 新增了 `tcgen05.commit.sync_restrict::shared::read::mma::a` 指令，允许我们提前发出 A 分块的完成信号。无需等待 MMA 全部完成，只要 MMA 读完共享内存中的 A 操作数，就可以触发屏障，让 TMA 加载器提前开始下一阶段的内存搬运。]

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/09.webp)

::: image-caption
该时序图展示GPU MMA矩阵运算的共享内存同步机制，展示 A 矩阵提前释放屏障在 MMA 读完 A 后即触发，早于 MMA 完全结束的时间点。具体来说，左侧蓝色阶段为读取矩阵A的数据，完成后触发`.sync_restrict::shared::read::mma::a`同步约束。同步完成后，紫色区块执行MMA运算的剩余计算流程，这一整段标记为earlier。同步操作释放A对应的共享内存槽位，下方绿色框表示A的存储槽变为空闲状态。该机制保证A数据读取完成后才启动后续MMA计算，同时及时释放共享内存资源，实现访存与计算的流水线调度，提升GEMM内核的硬件资源利用率。
:::

ThunderKittens 引入了新的提交类型来支持该特性：

```
tensor_commit<2> (inputs_finished[stage], mask); // MMA 结束时触发tensor_aread_commit<2>(A_finished[slot], mask); // MMA 读完 A 时触发
```

## 三、构建 GEMM 内核

现在我们有了新特性，也有了现成的 Blackwell GEMM 内核。接下来我们将这些特性逐步整合到现有的 Blackwell 内核中，并解释在 Vera Rubin 上每一步优化的必要性。

### 3.1 拓宽指令步长

> 最直观的性能瓶颈在于，我们仍在使用 Blackwell 的 32 字节 K 步长。在 Vera Rubin 上，这种编码的指令集性能上限约为 16.8 PFLOPS，:mdnice-bold-italic[而我们的 Blackwell 版 NVFP4 GEMM 原生就能达到 14.7 PFLOPS（达到上限的 88%）。要进一步提升性能，必须将 MMA 的 K 维度处理字节数翻倍。]

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/10.webp)

::: image-caption
GEMM 计算中 32-byte K 步与 64-byte K 步的理论算力上限与实际性能。红色标注的 32-byte K 步理论峰值为 16.8 PFLOPS，实测算力可达 14741 TFLOPS，硬件利用率达到 88%。绿色标注的 64-byte K 步拥有更高理论算力上限，为 35 PFLOPS。可视化色块直观呈现二者算力差距，说明 32-byte K 步方案已实现较高硬件占用，但受限于 K 步粒度，理论上限远低于 64-byte K 步，采用更大 K 步可进一步释放硬件潜在计算能力。
:::

::: block-1
然而，直接在原有 Blackwell 内核中打开宽步长后，我们发现性能仅略有提升，远达不到理论上的翻倍效果。:mdnice-bold-italic[虽然宽步长 MMA 让张量核心消费操作数的速度翻倍，但并未解决操作数供给速度的问题。要让双倍 K 步长发挥作用，我们需要两个手段来保证张量核心持续满载：减少数据搬运量，以及加深流水线让数据传输充分重叠。]
:::

### 3.2 减少读取字节数

> 为了减少数据读取量，我们在同一对 CTA 上沿 M 维度堆叠第二个输出分块。由于两个累加器仅在 M 维度不同，因此可以共享同一个 B 数据块。

我们最初的 Blackwell NVFP4 内核采用 1×1 的分块格式，意味着覆盖 M512×N256 的输出需要两对 CTA，各自独立加载自己的 B 矩阵副本。**改用 2×1 格式后，只需加载一次 B 即可覆盖同样的输出区域，减少了操作数流量。**

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/11.webp)

::: image-caption
GEMM 矩阵分块计算的 2×1 分块布局示意图，两个 M 分块（A0、A1）共享一个 B 数据块，生成 512×256 的累加器区域。左侧蓝色区域划分为 A0 与 A1 两个 A 矩阵子块；上方红色区域是 B chunk，代表 B 矩阵分块；右侧紫色区域包含 C0、C1 两组 M256×N256 累加器。C0 与 C1 上下堆叠，共同完成 M512 维度的计算。该分块策略将大矩阵拆分为多个子块，借助两组独立累加器并行计算，充分利用硬件计算资源，是 NVFP4 矩阵乘法中用于提升计算吞吐的分块设计方案。
:::

这种 2×1 分块格式在 Blackwell NVFP4 内核中很难实现，因为当时张量内存上限为 256 KiB。:mdnice-bold-italic[两个 M256×N256 的累加器已经占用了 512 列，意味着分块级 MMA 没有空间存储 A、B 的缩放因子。程序员可以通过让收尾线程束仅加载部分累加器列，再通知 MMA 空闲状态，从而让下一个 K 分块的 MMA 提前开始，但这会引入少量无法隐藏的延迟。幸运的是，Vera Rubin 多出的 64 列张量内存让我们可以直接存储缩放因子，无需这种迂回操作。]

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/12.webp)

::: image-caption
该图对比 NVIDIA Blackwell 与 Rubin 硬件的矩阵计算列资源。Blackwell 拥有 512 列，Rubin 提升至 576 列。图中紫色区域为 2×M256×N256 累加器，占用 560 列；右侧蓝色 16 列存放 A 矩阵缩放系数，红色 32 列存放 B 矩阵缩放系数，合计占用 560 列。红色虚线为资源边界，Blackwell 的 512 列不足以放下这套累加器与缩放因子组合，而 Rubin 的 576 列资源余量充足，可原生支持 NVFP4 带缩放的 GEMM 计算。
:::

### 3.3 加深流水线

> 分块格式的优化减少了操作数流量，但并未缩短单次读取的耗时。下一个挑战是让张量核心持续获得数据供给。:mdnice-bold-italic[Vera Rubin 更大的共享内存让我们可以构建更深的流水线，提前缓存更多分块，给数据传输留出更多完成时间。对 16k 见方的 NVFP4 和 FP8 GEMM 遍历环形流水线深度后，得到如下结果：]

**NVFP4 16k 见方 GEMM：**

| 共享内存分块级数 | 所需共享内存 | 实测 TFLOPS |
| --- | --- | --- |
| 3 | 202 KiB | 17,054 |
| 4 | 258 KiB | 20,595 |
| 5 | 314 KiB | 22,239 |

**FP8 (E4M3) 16k 见方 GEMM：**

| 配置 | 所需共享内存 | 实测 TFLOPS |
| --- | --- | --- |
| 4 | 209 KiB | 10,895 |
| 5 | 257 KiB | 11,995 |
| 6 | 305 KiB | 11,288 |

虽然最大的性能提升来自最后一步优化，但需要注意的是，**这些收益建立在前面所有优化的基础上**。下图是分别调整 K 步长、共享内存流水线和分块策略后的性能对比。

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/13.webp)

::: image-caption
NVFP4 Square 16384 GEMM 性能折线图，横轴为每个 CTA 占用的共享内存大小（KiB），纵轴是算力 TFLOPS，对比三种访存配置方案。橙色 2×1,64-byte 方案算力显著高于另外两组，算力随共享内存增加持续上升，d5 点达到最高值。蓝色 1×1,64-byte 与紫色 1×1,32-byte 性能明显偏低，曲线平缓，算力提升幅度很小。整体可见，采用 2×1,64-byte 访存策略能充分发挥 NVFP4 矩阵运算能力，共享内存容量提升对该方案性能增益效果突出。
:::

### 3.4 最终优化细节

> 为了进一步提升内核性能，我们还调整了几个最终参数：

1. **内核配置调优**：我们针对不同工作负载进一步调优内核以追求极致性能。在 CTA 对的尺寸上，我们发现 Blackwell 内核原有的 1×1 分块格式在小规模矩阵上表现最优；而对于更大尺寸的矩阵，我们采用 2×1 CTA 对分块，并搭配 2、4 或 8 个 CTA 的集群规模。此外，在所有尺寸下，我们都调整了分块的遍历顺序，以提升内存局部性和性能。
2. **B 侧收集器**：由于采用 2×1 分块格式，我们可以利用 B 侧收集器。通过在一个 MMA 上标注“FILL”，下一个 MMA 上标注“LASTUSE”，可以将 B 矩阵的读取次数从两次减少到一次。实测带来约 1-3% 的性能提升。
3. **使用 `sync_restrict::shared::read::mma::a`：** 对于 64k 和 128k 见方的更大规模 NVFP4 GEMM，我们观察到 A 矩阵提前释放分别带来了 13.5% 和 22.1% 的加速。我们发现该指令在大尺寸场景下效果显著——此时 A 分块会和其他资源竞争缓存驻留，导致其行数据在两次复用之间被驱逐，加载器不得不等待数据。而提前释放机制让我们可以充分利用 A 矩阵的复用收益。在小尺寸下，A 分块不会离开二级缓存，因此无需该优化读取速度就足够快。为了使用该指令，我们修改了传统的环形流水线逻辑。在普通 GEMM 中，A 和 B 分块属于同一个环形队列，在同一个屏障下同步推进。但要让 A 提前释放生效，需要将两个分块解耦，让 A 的加载可以独立运行。A 的提前释放需要 A 的槽位在更早的信号下就被释放，因此我们为 A 单独设置了环形队列和对应的就绪/完成屏障对。
4. **二级缓存驱逐提示**：我们为 A 操作数标记 `EVICT_LAST`，以提升后续任务对二级缓存的复用效率。这种复用收益来自跨任务的层面，而非单个集群内部，能带来零点几个百分点的提升。

### 3.5 性能结果

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/14.webp)

::: image-caption
柱状图「Vera Rubin 上的 NVFP4 GEMM」：ThunderKittens、cuBLASLt 与 CuTeDSL 在不同矩阵尺寸下的最终 TFLOPS 对比。横轴为矩阵尺寸 M=N=K，纵轴单位 TFLOPS，对比 ThunderKittens、cuBLASLt、CuTeDSL 三种实现方案。随着矩阵规模增大，三者算力整体持续上升，在 M=N=K=16384 时 ThunderKittens 达到峰值 22490 TFLOPS，性能最优。矩阵尺寸提升至 32768 时，三者算力小幅回落。在全部测试规格中 ThunderKittens 性能基本领先，cuBLASLt 紧随其后，CuTeDSL 算力相对更低。
:::

![文章插图](/articles/2026-09-18-22pflops-thunderkittens-vera-rubin-nvl72-b9362f6f/images/15.webp)

::: image-caption
柱状图「Vera Rubin 上的 FP8 GEMM」：ThunderKittens、cuBLASLt 与 CuTeDSL 在不同矩阵尺寸下的最终 TFLOPS 对比。横轴为矩阵尺寸 M=N=K，纵轴是算力 TFLOPS，对比 ThunderKittens、cuBLASLt、CuTeDSL 三种方案。矩阵规模较小时三者算力偏低，随矩阵增大性能持续上升，在 M=N=K=16384 时达到峰值，ThunderKittens 为 12117 TFLOPS，整体性能最优。矩阵尺寸继续增大至 32768，三种方案算力均出现下降。在全部测试规格下 ThunderKittens 算力基本领先，cuBLASLt 紧随其后，CuTeDSL 性能相对偏低。
:::

*注：以上所有测试均基于 NVIDIA CUDA 13.4，在资格样卡（QS）上运行。我们预计随着 Vera Rubin 软件栈的迭代，所有基准的性能还会继续提升。*

## 结语

希望本文的内容对大家有所帮助，我们也期待所有人都能尽快上手体验这些新特性。除了查找表 GEMM、硬件原生宏内核，以及新的引擎优化之外，还有很多有趣的技术细节值得分享。后续我们会带来更多内容！

参考资料

[1] Together 博客文章《ThunderKittens Now Optimized for NVIDIA Blackwell GPUs》: *https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus*

[2] ThunderKittens 2.0: Even Faster Kernels for Your GPUs: *https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2*

::: related-title
**相关推荐**
:::

::: related-links
- **[无源码改写汇编优化 GPU Kernel！AMD 联合南方科技大学提出 AsmEvo，让已编译 GPU 内核最高提速 3.88 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905090&idx=1&sn=53df28ba322bdc929e8df59b9f391f98&scene=21#wechat_redirect)**
- **[跨平台 Kernel 自动生成综述：基于 LLM 的监督微调+强化学习，适配 NVIDIA/AMD/华为 NPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896690&idx=1&sn=c57e7dc9b3f7aa251d2815304613d94c&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**

:::
