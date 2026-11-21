**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**Helion**、:mdnice-bold-italic[TPU]、**Pallas**、软件流水线、**自动调优**

> "**Helion 是 PyTorch 用于编写性能可移植机器学习内核的高级 DSL**。我们与 Google 合作构建了 **TPU 后端，将 Helion 内核编译到 Pallas**。"PyTorch 官方博客在 2026 年 7 月发布的这篇文章提到：:mdnice-bold-italic[深度学习内核编写，正在从"一套硬件一套代码"走向"一次编写、多硬件运行"。]

这背后的产业背景十分清晰。Google 最新的 TPU v7（Ironwood）在 BF16 算力与 HBM 带宽这两项现代机器学习负载最关键的硬件指标上，已经**与 NVIDIA B200 高度可比，且总体拥有成本可能更低。**

::: block-1
:mdnice-highlight[这意味着 TPU 正在从 Google 的内部武器，变成大规模训练与推理的现实选项]。但硬件可用只是第一步，**软件生态才是真正的门槛**：想为 TPU 写高性能内核，过去必须精通 Pallas 这门低级 DSL，学习曲线陡峭、代码复杂度高。
:::

Helion 给出的答案是，让开发者写熟悉的 PyTorch 风格代码，由编译器生成优化后的 TPU 代码，再配合自动调优器搜索最优配置。

- 在 **Flash Attention** 负载上，Helion 生成的内核在 :mdnice-bold-italic[TPU v7 上达到 838 TFLOPs，约为单个张量核 MFU 的 79%；]
- 在 **16 个内核的更广基准** 中，:mdnice-bold-italic[相对 TorchTPU eager 模式取得 1.55 倍几何平均加速，相对 torch.compile（XLA）也有 1.12 倍。]

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/01.jpg)

## 本文目录

- 一、TPU 内核为什么难写：一台"顺序机器"的编程鸿沟
- 二、Helion 的定位：用 PyTorch 语法写性能可移植内核
- 三、从 add 内核看懂 Helion 的 Pallas 代码生成
- 四、Flash Attention 实战：两种流水线策略的取舍
  - 4.1 默认方案 emit_pipeline：嵌套流水线及其气泡
  - 4.2 备选方案 unroll：用 VMEM 容量换零气泡
- 五、性能数字：838 TFLOPs、4.45 倍加速与 1.55 倍几何平均
- 行业意义与未来路线

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、TPU 内核为什么难写：一台"顺序机器"的编程鸿沟

> TPU 与 GPU 最本质的区别，藏在执行模型里：**TPU 是一台顺序执行的机器，依靠超宽的向量寄存器与计算单元取胜**；GPU 则依靠大规模并行执行与专用张量单元双轮驱动。

- 在线程模型上，**TPU（Pallas 编程模型下）是顺序执行、少数几个"大工人"干活**；GPU（CUDA 编程模型下）是 SIMT 并行加张量核、成千上万"小工人"协作。
- 在内存层次上，差异更为深刻：**TPU 采用显式内存空间，区分持久内存与暂存内存，想做流水线必须显式发起异步内存拷贝**；GPU 则依赖硬件管理的隐式缓存，程序员大多数时候可以把内存调度交给硬件。

> 这条差异直接决定了内核编写者的思维负担。在 TPU 上，内核作者必须深入理解内存层次，精确编排数据何时、以何种方式从片外 HBM 加载到片上 VMEM，并且让这些 HBM 与 VMEM 之间的传输，与矩阵单元（MXU）和向量计算单元上的浮点运算充分重叠。换句话说，**TPU 把流水线的责任从硬件转移到了程序员肩上。**

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/03.png)

::: image-caption
图 1：TPU 计算与内存层次结构，自上而下依次为 MXU 与向量计算单元、VREGs（最快、片上）、VMEM（快速、片上）、HBM（大容量、片外）。这张图也揭示了 TPU 编程的核心矛盾：计算单元只直接消费 VREGs 与 VMEM 中的数据，而模型数据栖身于 HBM，二者之间的搬运必须由内核显式编排。一个高性能 Pallas 内核的本质，就是让这条搬运链路与计算在时间轴上严丝合缝地重叠，任何空隙都是算力的浪费。
:::

值得注意的是，尽管架构哲学迥异，当代 TPU 与 GPU 的原始性能已经高度可比。:mdnice-bold-italic[TPU7x 与 NVIDIA B200 在 BF16 计算 TFLOPS 和 HBM 带宽上非常接近。硬件指标趋同，意味着竞争的胜负手转移到软件栈：谁能把硬件峰值更稳定地释放出来，谁就能在成本敏感的算力市场占据主动]。这正是 Helion 切入的位置。

## 二、Helion 的定位：用 PyTorch 语法写性能可移植内核

> Helion 要做的事，是让开发者用熟悉的 PyTorch 风格代码编写内核，由编译器负责生成针对目标硬件优化的实现，TPU 后端的目标语言是 Pallas。

作为 PyTorch 生态中面向性能可移植内核编写的高级 DSL，Helion 的 TPU 后端是与 Google 合作构建的。**Helion TPU 瞄准了三类用户场景，这三类场景实际上勾勒出了它的产品定位。**

| 场景 | 核心价值与适用场景 |
| --- | --- |
| 性能关键场景 | 当负载需要自动调优来探索配置空间时，Helion 的自动调优器可系统性地搜索分块大小、流水线策略等参数组合，替代人工试错，显著提升性能优化效率与可靠性。 |
| 非 Pallas 专家 | 希望快速上手 TPU 内核编写的团队，无需深入掌握 Pallas 的底层细节与复杂 API，可直接使用熟悉的 PyTorch 语法表达计算意图，大幅降低开发门槛与上手时间。 |
| 跨硬件用户 | 偏好在 TPU 与 GPU 之间复用同一套内核代码的组织，Helion 提供统一抽象与可移植性保障，支持“一次编写、双端运行”，减少硬件适配成本与维护负担。 |

这三类场景的共性，是**把"硬件专家知识"从内核开发的必要前提，变成编译器内建的能力**。这个思路与主流内核 DSL 在 GPU 上的成功路径一脉相承：:mdnice-bold-italic[保留高级语言表达，把硬件细节交给编译器。如今 TPU 后端编译到 Pallas，Helion 在朝着"硬件异构内核编写"的愿景演进]，这也是标题中 Towards Hardware Heterogeneous Kernel Authoring 的含义所在。

::: block-1
总而言之，:mdnice-highlight[Helion 的价值主张是双重的：向下屏蔽 Pallas 的复杂性，向上保留 PyTorch 的表达习惯，横向打通 GPU 与 TPU 的内核维护]，其本质是把硬件专业知识沉淀进编译器与自动调优器。
:::

## 三、从 add 内核看懂 Helion 的 Pallas 代码生成

> 要榨干 TPU 性能，核心命题只有一个：最大化软件流水线，让内存传输与计算尽一切可能重叠。Helion 的 Pallas 代码生成围绕这个命题构建了一套三层策略。

这套策略可以拆解为：

- **外层循环采用 Pallas 提供的流水线化设备调用**，即 `pallas_call` 与 `emit_pipeline`；
- **内层循环在两种方案之间自动调优选择**，一种是 Pallas 提供的设备侧流水线循环 emit_pipeline，另一种是在 VMEM 容量允许时把所有数据预取进 VMEM 的 unroll 方案；
- 流水线缓冲区的大小也交给自动调优器决定。

下面我们用最简单的逐元素加法内核展示了这套机制的运作方式。Helion 源码中，开发者只需写一个 `hl.tile` 循环，在循环体内对分块做加法。

```
@helion.kerneldef add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:    out = torch.empty_like(x)    for tile in hl.tile(out.size()):        out[tile] = x[tile] + y[tile]    return out
```

编译器将其翻译为两个函数：一个是主机侧的启动器，负责对输入分块并以流水线方式调用设备函数；另一个是设备函数，只操作驻留在 VMEM 中的分块。

```
def _helion_add(x, y, out):    out[:] = x[:] + y[:]def add(x: torch.Tensor, y: torch.Tensor):    _BLOCK_SIZE_0 = <autotuner-selected value>    out = torch.empty(...)    out = launcher( # wraps around pallas_call        _helion_add,         ((x.shape[0] + _BLOCK_SIZE_0 - 1) // _BLOCK_SIZE_0,), # grid size        x, y, out,         _block_spec_info=[_BLOCK_SIZE_0, ...], ...    )    return out
```

> 上述生成的代码里藏着三个关键设计决策。

- 第一，Helion 源码中的 `hl.tile` 循环在主机侧变成一个网格。启动器包装了 `pallas_call`，每个分块调用一次设备函数，且调用自动流水线化——当前分块在计算时，下一个分块的数据正在从 HBM 加载进 VMEM。
- 第二，设备函数收到的是 VMEM 引用而非 HBM 指针，所以内核 `_helion_add` 本体就是一句纯粹的加法，内存搬运的复杂性被完全隔离在启动器一侧。
- 第三，分块与缓冲区尺寸 `_BLOCK_SIZE_0` 由自动调优器选定，通过探索不同取值，为目标硬件找到内存传输与计算的最佳重叠点。

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/04.png)

::: image-caption
图 2：add 内核流水线执行时序，内存时间轴上依次执行加载分块 0 至 3，计算时间轴上相应执行分块 0 至 3 的加法，二者错开半个节拍持续重叠。这张图直观地展示了软件流水线的收益形态：除了流水线启动和排空的短暂首尾阶段，内存通道与计算单元始终都有活干。这正是 TPU 编程中最难手工做好的部分，而 Helion 用户写下的只是一个普通的分块循环。
:::

回顾第一章的内存层次图，可以把这里发生的事理解为一个注水模型：HBM 是水库，VMEM 是蓄水池，计算单元是水轮机。拙劣的内核让水轮机空转等水，优秀的内核则在发电的同时预热下一段水流。**Helion 生成的启动器代码，本质上就是一个自动化的水闸调度系统，而调度参数由自动调优器通过实测确定，无需开发者拍脑袋。**

## 四、Flash Attention 实战：两种流水线策略的取舍

> 加法是教学示例，Flash Attention 才是试金石。作为现代语言模型最关键的算子之一，它的生产级实现遵循分块计算模式，避免实例化完整的注意力矩阵，其内核结构中嵌套的内层循环，恰是流水线设计真正的分水岭。

Helion 版本的 Flash Attention 源码结构并不复杂：**外层对批量、注意力头与查询序列做分块，内层循环沿整个 K、V 序列分块遍历，循环体内完成 QK 矩阵乘、在线 softmax、与 V 的矩阵乘并更新累加器**。下面示例以规模为批量 8、头数 32、序列长 8192、头维度 256 展示。:mdnice-bold-italic[与 add 示例相比，多出来的这条内层循环如何流水线化，直接决定了内核性能的上限。]

```
B, H, S, D = 8, 32, 8192, 256 # batch, head, sequence length, head dimension@helion.kerneldef attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:    out = torch.empty(...)    for tile_b, tile_q in hl.tile(B * H, S):        this_q = q[tile_b, tile_q, :]        acc = ...        for tile_kv in hl.tile(S):            this_k = k[tile_b, tile_kv, :]            this_v = v[tile_b, tile_kv, :]            <qk matmul, online softmax, v matmul, update acc>        out[tile_b, tile_q] = acc    return out
```

Helion 编译器在自动调优时对两种翻译策略做选择，开关是自动调优配置中的 pallas_loop_type。

### 4.1 默认方案 emit_pipeline：嵌套流水线及其气泡

> 默认配置下，Helion 借助 Pallas 的设备侧 emit_pipeline 接口对内层循环体做流水线，方式与主机侧用 pallas_call 流水线化设备函数调用如出一辙。

```
def _helion_attention(q_VMEM, k_HBM, v_HBM, out_VMEM):    acc = ...    this_q = q_VMEM[:, :, :]    def _inner_pipeline_body(k_VMEM, v_VMEM):        this_k = k_VMEM[:, :, :]        this_v = v_VMEM[:, :, :]        <matmul, online softmax, matmul, update acc>    pallas.tpu.emit_pipeline(_inner_pipeline_body, k_HBM, v_HBM, _block_spec_info=[BLOCK_SIZE_KV ,...], ... )    out_VMEM = accdef attention(q: torch.Tensor, k: torch.Tensor, v:torch.Tensor):    out = torch.empty(...)    out = launcher( # wraps around pallas_call        _helion_attention,         q, k, v, out,         _block_spec_info=[BLOCK_SIZE_Q ,...], ...    )    return out
```

生成的内核呈现嵌套流水线结构：

- 外层流水线上，主机用 pallas_call 调用设备函数，Q 的 HBM 引用被分块，每次调用收到一个 Q 的 VMEM 分块，而 K 和 V 以 HBM 引用直接传入；
- 内层流水线上，设备函数内部用 emit_pipeline 调用内层循环体，后者接收 K、V 的 VMEM 分块。

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/05.png)

::: image-caption
图 3：emit_pipeline 方案下 Flash Attention 的流水线时序，内存轴上 Q0 之后依次加载 KV0、KV1、KV2，随后 Q1 与又一轮 KV0、KV1、KV2，计算轴上相应执行各组 Q 与 KV 分块的计算。当然，这张图暴露了该方案的结构性弱点：每当新的 Q 分块到来，整套 KV 分块都要从 HBM 重新加载，而在抓取第 0 个 KV 分块的窗口内，计算单元无活可干，形成清晰可见的气泡。其根源在于 KV 数据没有被跨 Q 分块复用。
:::

### 4.2 备选方案 unroll：用 VMEM 容量换零气泡

> 针对上述气泡，Helion **提供了 pallas_loop_type 取 unroll 的备选配置。该方案把内层循环翻译成一个普通的 Python for 循环** ——:mdnice-bold-italic[之所以叫 unroll，是因为 Pallas 设备函数会被 JAX 的 JIT 追踪，Python 循环在追踪期实际上被展开成一串扁平的操作序列]。

```
def _helion_attention(q_VMEM, k_VMEM_FULL, v_VMEM_FULL, out_VMEM):    acc = ...    this_q = q_VMEM[:, :, :]    for offset in range(0, k_VMEM_FULL.size(1) , BLOCK_SIZE_KV):        this_k = k_VMEM_FULL[:, pallas.dslice(offset, BLOCK_SIZE_KV), :]        this_v = v_VMEM_FULL[:, pallas.dslice(offset, BLOCK_SIZE_KV), :]        <matmul, online softmax, matmul, update acc>    out_VMEM = accdef attention(q: torch.Tensor, k: torch.Tensor, v:torch.Tensor):    out = torch.empty(...)    out = launcher( # wraps around pallas_call        _helion_attention,         q, k, v, out,         _block_spec_info=[BLOCK_SIZE_Q, None, None], ...    )    return out
```

这个版本有两个相辅相成的设计要点。

1. K 和 V 被整体预取。主机侧把 K、V 的分块规格设为 None，指示 `pallas_call` 将它们完整加载进 VMEM，且完整的 VMEM 引用跨所有设备函数调用持续存在。
2. 内层循环改为本地切片。每次迭代用 `pallas.dslice` 从已驻留的 VMEM 缓冲区中切出当前 KV 分块，内层循环期间不发生任何 HBM 流量。

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/06.png)

::: image-caption
图 4：unroll 方案流水线时序，内存轴上先一次性加载完整 KV，之后仅需依次加载 Q0、Q1 等查询分块，计算轴上各组 Q 与 KV 分块的计算首尾相接。与图 3 对照可以一眼看出差异：计算流水线上的气泡彻底消失，Q0 与 Q1 的计算批次之间不再隔着 KV 重载的空窗。代价同样直观——内存轴前端那次完整 KV 加载意味着 VMEM 必须装得下整个 K 与 V 序列。
:::

代价可以用一个简单式子刻画。设序列长度为 $S$，头维度为 $D$，每个元素占 $b$ 字节，则单个批量、单个注意力头分片下，完整驻留 K 与 V 所需的 VMEM 为：

$$
M_{KV} = 2 \cdot S \cdot D \cdot b
$$

代入示例参数 $S = 8192$、$D = 256$、$b = 2$（BF16），得到 $M_{KV} = 2 \times 8192 \times 256 \times 2$ 字节，即 8 MB。序列长度翻倍到 32768 时，需求线性膨胀到 32 MB——注意它随序列长度线性增长，与分块大小无关。

这解释了给出的对照数据：在批量 8、头数 32、头维度 256 的负载上，序列长 8k 时 unroll 跑出 892 TFLOPs，显著高于 `emit_pipeline` 的 653 TFLOPs；但序列长 32k 时，unroll 直接显存溢出，`emit_pipeline` 则以 695 TFLOPs 成为唯一可行解。

::: block-1
于是 Helion 自动调优的价值闭环浮现出来：**短序列时，它选择 unroll，充分利用可用 VMEM 生成零气泡的流水线代码；长序列时，它回退到 emit_pipeline，后者可以扩展到任意上下文长度**。这个按输入形状切换代码生成策略的能力，正是 Helion 即便面对 Tokamax 这类高度优化的实现也能占据上风的根源。
:::

## 五、性能数字：838 TFLOPs、4.45 倍加速与 1.55 倍几何平均

> Flash Attention 负载上，Helion 生成的内核在 TPU v7 上达到 838 TFLOPs，约为单个张量核 MFU 的 79%。

把 79% MFU 翻译一下：模型算力利用率衡量实际算力占理论峰值的比重，79% 意味着每 5 份理论算力中有近 4 份被真正用于有效计算。**由 838 TFLOPs 除以 0.79 可以反推，单个张量核的 BF16 峰值约在 1060 TFLOPs 量级**。对于一条由高级 DSL 自动生成的内核而言，这个利用率已经跨入了手写优化内核的领地。

跨序列长度的对比图进一步说明了自动调优的实战意义，也划出了它的边界。作者绘制了该注意力内核与 Pallas 参考实现、Tokamax、纯 JAX 在不同序列长度下的前向速度曲线：

- **序列长度 8192 及以内，Helion 全程领先**，并在 8192 处冲上 892 TFLOPs 的峰值；
- **序列加长到 16384 与 32768 后，Helion 回退至 emit_pipeline**，成绩维持在 695 TFLOPs 上下，被持续爬升的 Tokamax 反超。

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/07.png)

::: image-caption
图 5：Helion 注意力内核与 Pallas 参考实现（JAX 仓库）、Tokamax、纯 JAX（无 Pallas）在不同序列长度下的前向速度对比，测试条件为非因果前向、批量 8、头数 32、头维度 256、BF16。这条曲线的关键信息有两层：其一，Helion 在 8192 及以内全面领先并于 8192 处达到 892 TFLOPs 峰值，这是 unroll 策略吃满 VMEM 后的零气泡红利；其二，16384 与 32768 处 Helion 回落至 695 TFLOPs 附近，被 Tokamax 的 726 与 748 TFLOPs 反超，说明回退到 emit_pipeline 后 Helion 的长序列流水线效率仍有提升空间，这也呼应了博客"进一步性能改进"的在研方向。
:::

作者提到，:mdnice-bold-italic[自动调优器按输入长度切换循环与流水线策略的能力，正是 Helion 即便面对 Tokamax 这类高度优化实现也能占据上风的根源——这句话在 8k 以内的区间完全成立，在更长序列上则更多是"保持第一梯队"而非"保持第一"。]

> 视野放宽到 16 个内核的基准测试，作者对比了 Helion、TorchTPU eager 模式与 torch.compile（XLA 后端）三方。

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/08.png)

::: image-caption
图 6：Helion 与 TorchTPU eager、torch.compile 在 16 个内核上的性能对比图表。全表呈明显的两极分布：attention、rms_norm 反向、bmm 等融合或复杂模式内核上 Helion 大幅领先，而 layer_norm、rms_norm、cross_entropy 等标准算子上 XLA 仍占优。这张图实际上是一份编译器能力地图，标出了 Helion 自动调优当前的射程与 XLA 成熟优化的腹地，也为用户选择工具链提供了直接依据。
:::

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/09.jpg)

> 整体结论是 **Helion 相对 eager 取得 1.55 倍几何平均加速，相对 compiled 取得 1.12 倍**。几个代表性数据点值得展开。

- 注意力内核在形状为批量 8、头数 32、序列长 8192、头维度 256 的负载上，eager 模式 87.77 毫秒，compile 88.28 毫秒，Helion 仅 19.72 毫秒，加速比分别达 4.45 倍与 4.48 倍。
- rms_norm 反向传播、bmm、epilogue_subtiling 相对 eager 分别加速 2.71 倍、2.10 倍与 2.04 倍。

同样值得注意的是硬币的另一面：在 matmul、layer_norm、rms_norm、cross_entropy 等**标准算子上，Helion 相对 torch.compile 的加速比在 0.68 倍到 0.96 倍之间，均落于下风**。

::: block-1
作者对此的解释：Helion 的最大优势体现在那些采用融合或特定优化模式、而 XLA 难以自动识别的内核上，Flash Attention 是典型代表；对于 matmul 和 layer_norm 等标准操作，XLA 编译器已能生成高质量代码，Helion 的性能与之相当。:mdnice-highlight[这组数据明确了 Helion 当前的适用边界：它专精于复杂融合内核的优化，尚无法全面替代 XLA 对标准算子的优化能力。]
:::

## 行业意义与未来路线

> Helion on TPU 的意义超出单个后端的范畴：它是 PyTorch 内核编写范式向多硬件时代迁移的一次实弹演习。

回看整条技术链，Helion 的 TPU 后端把内核编译到 Pallas，而其跨硬件定位让偏好同一套内核的组织得以在 TPU 与 GPU 之间自由迁移。**同一份 PyTorch 风格的内核源码，由此成为多硬件后端的共同输入**。对于需要在异构算力间灵活调度的机构而言，这承诺了一种诱人的工程形态：**内核资产一次沉淀，硬件选型保持开放**。Helion 的三类目标场景——性能关键、非 Pallas 专家、跨硬件用户——恰好对应了算力多元化时代最典型的三类痛点。

当然，愿景与现实之间还有明确的 ToDo，包括：

- 扩大内核覆盖面，让更多 Helion 示例在 TPU 上跑通；
- 进一步的性能改进；
- 对不规则与稀疏操作的更好支持；
- 以及分布式 TPU 计算支持。

此外还有一个现实的落地约束：:mdnice-bold-italic[Helion 的 TPU 后端依赖 TorchTPU，后者预计在今年晚些时候才会公开发布。也就是说，当前外界能立即上手的是 Helion 的开源仓库、文档、TPU 示例与性能看板，完整的 TPU 体验尚需等待 TorchTPU 就位。]

::: block-1
**从更长的视角看，这项工作验证了 DSL 加自动调优路线在新硬件上的可迁移性**。Flash Attention 案例中，真正困难的工程判断——VMEM 够不够装下整个 KV、哪种流水线结构更优、缓冲区开多大——全部被编码为可搜索的离散选择，交给实测数据裁决。:mdnice-highlight[当硬件继续分化，这类"把硬件直觉编译进搜索空间"的方法论，可能比任何单一内核优化都更有复利价值。]
:::

- 原文博客：Helion on TPU: Towards Hardware Heterogeneous Kernel Authoring
- Helion GitHub 仓库：https://github.com/pytorch/helion
- Helion 文档：https://helionlang.com/
- Helion TPU 示例：https://github.com/pytorch/helion/tree/main/examples
- 性能看板：https://helionlang.com/dashboard/

::: related-title
**相关推荐**
:::

::: related-links
- **[Helion DSL 赋能 vLLM 实现跨硬件高效推理：Qwen3 模型 FP8 推理加速实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902433&idx=1&sn=5e65e917235f732c27c1cd121a2d6d24&scene=21#wechat_redirect)**
- **[Google 第八代 TPU：面向智能体时代的训练与推理芯片，原生支持 JAX、MaxText、PyTorch、SGLang 与 vLLM](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900266&idx=1&sn=dd3219b311dfca2bba5566c5acb42fd5&scene=21#wechat_redirect)**
- **[用GPU原型验证，用TPU训练和推理！TorchTPU：谷歌 Meta 联手让 PyTorch 原生跑 TPU，3 行代码从 GPU 切换到 TPU 的极致易用性！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899873&idx=1&sn=f61bc039e138133b982b29e612d0a1fb&scene=21#wechat_redirect)**

:::
