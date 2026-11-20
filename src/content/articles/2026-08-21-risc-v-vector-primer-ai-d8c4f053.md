**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**RISC-V**、RVV、:mdnice-bold-italic[向量处理器]、矩阵计算、**编译器优化**

> 很多人第一次读 RISC-V Vector Extension，都会被一串缩写挡在门外：VLEN、SEW、LMUL、VL、VLMAX、vtype、vstart、mask。它们看似只是规格书术语，实际牵动三层系统：**编译器如何生成循环，处理器如何组织寄存器文件，AI 与矩阵计算如何获得吞吐。**

![文章插图](/articles/2026-08-21-risc-v-vector-primer-ai-d8c4f053/images/01.jpg)

- **An implementation-focused guide to the RISC-V Vector Extension (RVV 1.0) and the emerging Matrix Extension** —:mdnice-bold-italic[written for architects, compiler engineers, and embedded/edge AI developers.]
- https://github.com/simplex-micro/riscv-vector-primer

::: related-title
**相关推荐**
:::

::: related-links
- **[培养下一代芯片架构师！65 名学生、33 个项目、5 次流片的端到端硅验证实践，Croc 领域专用 RISC-V 平台的教育规模化实验](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903235&idx=1&sn=bd826c155b8d5ebc1f6434b8cb97bb18&scene=21#wechat_redirect)**
- **[用 Python 写一个开源 RISC-V 向量编译器：MLIR 与 xDSL 的 GEMM 代码生成，较 OpenBLAS 最高提升 35%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899103&idx=1&sn=1729342f5820122c5fad0a641be01943&scene=21#wechat_redirect)**
- **[开放指令集的 AI 答卷：PyTorch 登陆 RISC-V 架构的技术长征与生态破局](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900048&idx=1&sn=59a68af5c7a3333e2f816a50c9b6ee43&scene=21#wechat_redirect)**

:::

> `RISC-V Vector Primer` 这个项目，**把 RVV 从一份抽象 ISA 文档，翻译成可被工程师操作的心智模型**。

读完这个仓库，你会发现 RVV 的核心野心是让软件摆脱固定硬件宽度，让同一套二进制能够在 128、256、512 甚至更宽的实现上自然伸缩。

## 本文目录

- 一、直接阅读这本“规格书翻译器”
- 二、这本 Primer 解决的核心矛盾：向量宽度不该写死在 ISA 里
- 三、VL/SEW/LMUL：把循环交给硬件协商
- 四、从 ISA 到硅片：512-bit VPU 的真实代价
- 五、指令与编译器：内存形态决定性能
- 六、从向量到矩阵：RVV 留下的下一层抽象
- 结语

![文章插图](/articles/2026-08-21-risc-v-vector-primer-ai-d8c4f053/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、直接阅读这本“规格书翻译器”

> `simplex-micro/riscv-vector-primer` 是一本面向架构师、编译器工程师、嵌入式与边缘 AI 开发者的开放技术书。它没有传统项目里的构建系统、测试脚本和源码目录，**核心内容集中在 Markdown 章节与配图中。仓库 README 明确建议直接浏览章节文件，并从第一章开始阅读。**

```
# 来源：README.mdgit clone https://github.com/simplex-micro/riscv-vector-primer.gitcd riscv-vector-primer# 从目录页或第一章开始open README.mdopen chapter-01.md
```

在线阅读时，建议从 `README.md`[1] 进入，再按章节顺序阅读。

- `chapter-01.md` 建立向量与 SIMD 的心智模型；
- `chapter-02.md` 进入 512-bit RVV 核的硬件实现；
- `chapter-03.md` 解释 SEW、LMUL、VL、vtype、掩码等基础；
- `chapter-04.md` 展开指令分类；
- `chapter-05.md` 讨论矩阵乘与性能；
- `chapter-06.md` 把 RVV 推向矩阵扩展。

仓库描述特别提醒 “Read chapters directly from this repo”，所以建议以 GitHub 仓库章节为准。整个项目的顶层结构非常克制：

```
// 来源：simplex-micro/riscv-vector-primerREADME.md       项目介绍、作者、引用方式、章节说明index.md        章节索引chapter-01.md   RVV 概念、SIMD 对比、strip mining、chainingchapter-02.md   RVV 核实现、512-bit VRF、lane、port pressurechapter-03.md   SEW、LMUL、VL、vtype、掩码、vstartchapter-04.md   load/store、compute、mask、permutation 指令族chapter-05.md   GEMM、MAC pipeline、性能分析、确定性执行chapter-06.md   从一维向量到二维矩阵 tile 与 Matrix Extension
```

这套组织方式很适合技术深读：第一章解决“为什么”，第二章解决“怎么造”，第三、四章解决“怎么编程”，第五、六章回答“为什么 AI 与矩阵计算会把 RVV 推向新的抽象”。

## 二、这本 Primer 解决的核心矛盾：向量宽度不该写死在 ISA 里

> 传统 SIMD 常把硬件宽度直接写进软件假设里。MMX、SSE、AVX、AVX-512 每一次扩展都带来新寄存器宽度、新编码、新指令组合。编译器需要判断目标机器支持哪一档宽度，运行库也常常准备多套路径。硬件升级以后，软件想吃满新宽度，往往要重新编译、分发新版本，或者在运行时选择不同代码路径。

RVV 的思路更锋利：

- ISA 描述“对一组元素执行操作”，硬件实际能一次容纳多少元素，由实现决定；
- 软件通过 `vsetvl` 系列指令动态询问并设置当前可处理元素数。

这样，硬件宽度成为实现细节，程序表达保留可移植性。

```
// 来源：chapter-01.mdfor (int i = 0; i < 64; i++) {    y[i] += a * x[i];}// RVV 心智模型：// 1. 一次向量 load 读取 x 的一段// 2. 一次向量 multiply 处理多个元素// 3. 一次向量 add 合并 y// 4. 一次向量 store 写回// 数百条动态标量指令，被少量向量指令压缩
```

这段例子是整本书的入口。同一件事，标量代码让前端不断取指、译码、分支；向量代码把循环体压缩成数据流描述。前端负担下降，后端流水线获得更长、更规则的工作。

::: block-1
Primer 最重要的贡献，是把“向量处理器”从“大号 SIMD”的误解中拉出来。SIMD 更像一把固定宽度的尺子，一次量多少由尺子长度决定；:mdnice-bold-italic[RVV 更像一套测量协议，硬件告诉软件本轮能处理多长，软件按这个长度推进循环。这个差异决定了 RVV 面向的是硬件世代演进，而非某个单点宽度。]
:::

## 三、VL/SEW/LMUL：把循环交给硬件协商

> RVV 的可伸缩性来自几个关键量：

- **SEW**：单个元素宽度，比如 8、16、32、64 bit。
- **VLEN**：单个向量寄存器的硬件宽度，由芯片实现固定。
- **LMUL**：把多个向量寄存器逻辑拼成一组，牺牲寄存器数量换取更长向量。
- **VL**：当前指令真正参与计算的元素数。
- **VLMAX**：当前 SEW 与 LMUL 下，一组向量寄存器最多容纳多少元素。

核心公式非常朴素：

$$
VLMAX = (VLEN / SEW) × LMUL
$$

这个公式把硬件容量、数据类型、寄存器分组绑到一起。:mdnice-bold-italic[编译器真正关心的是：本轮循环还能处理多少元素？硬件回答一个 `vl`，程序处理一段，再进入下一段。这就是 strip mining 的基础。]

```
# 来源：chapter-03.md# vsetvli 同时设置 VL 与 vtype# e32 表示 SEW=32，m1 表示 LMUL=1vsetvli t0, a0, e32, m1, ta, ma# 若 a0 > VLMAX，硬件令 t0 = VL = VLMAX# 若 a0 <= VLMAX，硬件令 t0 = VL = a0
```

这条指令体现了 RVV 的优雅之处：循环不用硬编码“512 位能放 16 个 int32”，也不用知道目标机器实际 VLEN。**编译器把剩余元素数放进 `a0`，硬件给出这次最合适的 `VL`。**

::: block-1这套机制也让“尾巴处理”变得自然。现实数组长度很少刚好等于 VLMAX。传统 SIMD 常出现“主循环 + 尾循环”的代码形态：先处理能整除向量宽度的大块，最后用标量或掩码处理剩余元素。**RVV 的 `VL` 让最后一轮自动变短，循环结构仍然统一。**
:::

掩码与 `vstart` 则补上不规则控制。`v0` 中每个 bit 对应一个元素，决定它是否参与操作；`vstart` 记录从哪个元素继续执行，用于异常恢复。向量 load 如果在第 4 个元素触发 page fault，硬件可以记录 `vstart=3`，异常处理后继续执行。RVV 给系统软件留下了可恢复的坐标，也给编译器留下了表达不规则数据的空间。

## 四、从 ISA 到硅片：512-bit VPU 的真实代价

> 第二章把概念落到硅片：典型 RVV 核由标量 CPU 加向量处理单元 VPU 构成。CPU 负责取指、提交、CSR 状态等控制逻辑；VPU 拥有向量寄存器文件、向量整数单元、浮点单元、掩码单元、置换单元和向量 load/store 单元。

这种分工很像“主厨与灶台”：CPU 决定菜谱、步骤和火候参数，VPU 负责持续吞吐数据。但二者无法彻底分离，因为向量指令使用标量寄存器提供基地址、标量操作数和配置参数，内存相关性也要在系统层面保持正确。

512-bit VRF 听起来很美，落到版图就是大量位线、端口、电容与时序压力。项目反复强调，向量处理器的难点并非“放几个宽 ALU”这么简单。每个功能单元都想读写 VRF，每增加一个端口，面积、功耗、时序都会恶化。

解决思路是 lane。以 512-bit VLEN、4 lanes 为例，每个 lane 处理 128 bit 切片，寄存器文件也按 lane 切分。局部连线变短，功能单元靠近自己的数据，时钟频率更容易收敛。

```
// 来源：chapter-02.mdVLEN = 512 bitlanes = 4per_lane_slice = 128 bit每个 lane 包含：- 本地 VRF 切片- ALU / multiplier / divider / FPU- 128-bit load/store 通路设计目标：让大部分数据移动停留在 lane 内部，减少长距离布线。
```

从软件看，`v1` 是一个向量寄存器；从硬件看，它可能被切成多个 lane 的物理片段。RVV 的抽象屏蔽了这种复杂性：软件拿到统一寄存器模型，芯片内部保留实现自由。

**另一个关键机制是 chaining**。向量指令可以被拆成多个元素组或 micro-op。:mdnice-bold-italic[当前一条指令的第一批结果刚出来，后一条依赖指令就能立刻消费这批结果，无需等待整个向量全部写回 VRF]。好比一条生产线：VLOAD 装上第一批零件，VMUL 立刻加工，VADD 紧跟着装配。每一批数据都在流水线上前进，功能单元持续忙碌。对于长向量，这种重叠可以显著隐藏延迟。

Primer 在这里给出了一个很现实的架构判断：**高性能 RVV 实现需要调度、端口规划、旁路路径和异常语义共同配合**。向量性能来自流水线、寄存器文件、内存带宽和编译器循环形态的合奏。

## 五、指令与编译器：内存形态决定性能

> `chapter-04.md` 把 RVV 指令分成四类：memory、compute、mask、permutation。它们看起来像指令手册目录，背后却是编译器优化的成本模型。

内存模式尤其关键。RVV 支持 unit-stride、constant-stride、indexed gather/scatter、segment load/store。它们表面只是寻址模式，实际决定缓存效率、bank conflict 和 load/store 单元复杂度。

- **unit-stride**：连续访问，最容易跑满带宽。
- **strided**：固定步长，适合列访问、结构化间隔访问。
- **indexed**：任意偏移，表达能力强，硬件冲突更多。
- **segment**：把 AoS 数据一次拆成 SoA，非常适合 RGB、复数、结构数组。

项目在第五章进一步指出：矩阵乘中，strided load 与 unit-stride load 加 permutation 是一组经典取舍。前者代码直接，后者可能用额外 shuffle 换更少 cache line 流量。

置换指令则是“把不规则性从内存搬进寄存器”的工具。`vrgather`、`vslideup`、`vslidedown`、`vcompress` 让程序可以先用连续 load 高效取数，再在 VRF 内重排。对编译器来说，这是一个重要策略：如果随机内存访问昂贵，那就把布局调整放到寄存器文件里完成。

```
# 来源：chapter-04.mdvrgather.vv vd, vs2, vs1, vm      # vd[i] = vs2[vs1[i]]vrgatherei16.vv vd, vs2, vs1, vm  # 16-bit indexvrgather.vx vd, vs2, rs1, vm      # 所有元素取同一个索引vrgather.vi vd, vs2, uimm, vm     # 立即数索引# 常见策略：# 1. 用 unit-stride load 连续取数# 2. 用 vrgather 在 VRF 内完成重排
```

::: block-1这类指令让 RVV 具备处理真实数据布局的能力。:mdnice-bold-italic[许多性能问题本质上是数据没有以合适形态抵达计算单元。RVV 给了编译器两条路：直接表达复杂访问，或者用连续访问加寄存器内重排来换取更稳定的带宽。]
:::

## 六、从向量到矩阵：RVV 留下的下一层抽象

> 第五章用单精度 GEMM 和低精度 MAC pipeline 展示 RVV 的强项：可伸缩向量长度、链式执行、混合精度、显式内存模式。到了第六章，作者把问题推向更本质的一层：矩阵天然是二维结构，RVV 的向量寄存器是一维抽象。用一维向量表达二维 tile，软件要负责切片、分组、加载、置换、归约、推进指针。

**这说明工作负载结构发生了变化**。矩阵乘的内核拥有高复用、高算术密度、规则二维数据流。继续用向量表达当然可行，只是指令开销、寄存器压力和数据布局管理会越来越显眼。

第六章提出的路线很清晰：RVV 继续承担通用数据并行、控制流、不规则访问、数据搬运；Matrix Extension 引入 tile 作为一等架构对象，让二维矩阵块常驻在 tile register file 中，由 PE array 执行大规模乘加。

```
// 来源：chapter-06.mdRVV 角色：- 控制流- 数据搬运- layout transform- 不规则与混合工作负载Matrix Extension 角色：- tile register file- 二维 PE array- tile-level matrix multiply-accumulate- 降低指令开销，提高数据复用
```

这条路线符合现代处理器趋势：标量负责控制，向量负责通用数据并行，矩阵单元负责高度规则的密集计算。三者组合成层次化计算结构，服务 AI、DSP、图像和科学计算。

## 结语

> `RISC-V Vector Primer` 的核心价值把 RVV 从“术语集合”讲成了“系统协约”：

- 软件用 `VL`、mask、strip mining 写可移植循环；编译器围绕 `vsetvl`、LMUL、SEW、load/store 模式建立成本模型；
- 芯片架构师面对 VRF 端口爆炸、lane 切分、chaining、异常恢复与矩阵阵列取舍。

它真正揭示的，**是一种面向未来硬件的伸缩原则**：:mdnice-bold-italic[ISA 定义稳定契约，让软件描述元素级并行，让硬件选择寄存器宽度、lane 数量、端口数量、流水线深度与矩阵阵列规模]。RVV 的意义，也正在这份自由里。

参考资料

[1] `README.md`: *https://github.com/simplex-micro/riscv-vector-primer/blob/main/README.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[RISCBench：面向 FPGA 及类 FPGA 计算引擎的 RISC-V 编排效率基准测试](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898844&idx=1&sn=a9d00d2702d21c4120d7ba18050dc665&scene=21#wechat_redirect)**
- **[217 倍加速与 137GOP/s 峰值算力！数字存内计算与 RISC-V 向量架构的流水线级创新集成，突破边缘 AI 计算极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897291&idx=1&sn=2c7fb897d75c4f6b9bc04659b802735c&scene=21#wechat_redirect)**
- **[1 核抵 8 核！解耦 RISC-V GPGPU 的控制流和数据访问：Vortex GPGPU 从 0.35 到 1.63 GFLOP/s/mm² 的性能飞跃](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896391&idx=1&sn=83dedbba5262bbea3b8fcd0b5d6b0659&scene=21#wechat_redirect)**

:::
