关键词：:mdnice-bold-italic[NVIDIA SASS]、Blackwell SM120、**张量核心**、:mdnice-bold-italic[模式库]、**逆向工程**

> 每一份用 `nvcc` 编出来的 cubin，最终都会被 GPU 翻译成一种没有官方手册的本地汇编——SASS。CUDA 程序员看得到 PTX，**能读懂 Nsight 给的 warning，却几乎可能没有办法在底层去回答**：":mdnice-bold-italic[为什么 ptxas 把我的 `a  b + c` 拆成两条 FADD 而不是一条 FFMA？"、"为什么这两个 `LDG` 共用了同一个 scoreboard？"]

自 2018 年 Citadel 的 Jia 等人系统逆向 Volta 与 Turing 之后，Ampere、Hopper、Blackwell 三代消费/数据中心架构基本【鲜有】同等深度的公开工作。

![文章插图](/articles/2026-06-07-sass-king-gpu-blackwell-29-280bed77/images/01.webp)

- **SASS King 是一项针对英伟达SASS的系统性逆向工程项目**。SASS是编译后的CUDA二进制文件中所使用的原生GPU指令集。*该项目以Blackwell架构的消费级硬件SM120/SM120a为起点，后续将逐步拓展，打造一套完整的跨架构指令集架构与模式库。*
- :mdnice-bold-italic[该项目的目标注重实际应用：助力内核工程师查看SASS转储文件、识别编译器特征模式、定位影响性能的程序结构，并将二进制代码与源码层面的优化策略建立关联。]
- 项目地址：https://github.com/florianmattana/sass-king

::: related-title
相关推荐
:::

::: related-links
- [暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)
- [逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)
- [逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894524&idx=1&sn=0282673a4d170d4c5a14a563ad311d78&scene=21#wechat_redirect)

:::

::: block-1
SASS King[1] 正是想接住这条断掉的链子——它不是又一款反汇编器，而是一个**证据驱动** 的研究语料库，把消费级 Blackwell（SM120 / SM120a，RTX 5070 Ti / 5090）的 SASS 重新拆开、命名、归档，最终凝结为 **29 条可在任何新 dump 上复用的模式签名**。
:::

## 本文目录

::: related-links
- 快速上手与使用
- 一、为什么 SASS 还需要一支"考古队"
- 二、证据流水线：把"猜测"和"事实"严格隔离
- 三、读懂第一段 SASS：从 vector_add 到全套阅读模板
- 四、张量核心的 SASS 真相：HMMA / QMMA / OMMA 家族图谱
- 五、编译器的"心思"：寄存器与统一寄存器流的工程含义
- 六、29 条模式的工程价值与下一站

:::

![文章插图](/articles/2026-06-07-sass-king-gpu-blackwell-29-280bed77/images/02.webp)

## 快速上手与使用

> 整个项目的"可执行部分"只是一组受控 CUDA 微核，因此上手只需要 `nvcc` + `cuobjdump`：

```
# 取自 docs/START_HERE.mdcd corpus/basics/01_vector_addnvcc -arch=sm_120 kernel1.cu -o vector_addcuobjdump --dump-sass vector_add > sm_120.sass
```

> 读 dump 时按"四件套"走：

先看 `docs/START_HERE.md`[2] 学会项目的阅读姿势；再用 `patterns/README.md`[3] 作为审计入口去匹配 29 条 `PATTERN-NN` 页面；如果想追根溯源，每条模式末尾都标了 `Source: knowledge/FINDINGS.md`，那是项目的"全量证据日志"；想看具体 Micro Kernel 长什么样，则进入 `corpus/tensor_cores/`[4] 这一章一章读 `conclusionNN.md`。

::: block-1
如果你手上有 A100、H100、4090 等非 RTX 5070 Ti 硬件，按 `CONTRIBUTING.md`[5] 提交一份带完整元数据的 `cuobjdump` dump，作者会把分析跟进做完——这就是这个项目目前对外的"协作 API"。
:::

## 一、为什么 SASS 还需要一支"考古队"

> SASS 是 NVIDIA 真正发到 SM 上执行的本地指令集，但 NVIDIA 从未公开它的指令表、控制码字段和调度模型。CUDA 用户能看到的只是 PTX——一种"虚拟汇编"，再往下就是 `cuobjdump` 喷出来的、几乎没有上下文的助记符。

Jia 等人在 2018 年那篇 Dissecting the NVIDIA Volta GPU Architecture[6] 之后，公开研究几乎停滞，而这期间 NVIDIA 已经迭代了三代架构、引入了 `cp.async`、warpgroup MMA、TMA、块缩放 FP4/FP6/FP8 等一整套新指令族。SASS King 的目标因此非常具体：**先把 SM120 / SM120a 这一片"消费级 Blackwell"的指令集与编译器决策搞清楚，再向其他架构横向复用**。

项目作者把这件事的工程哲学浓缩成一条流水线，路线图直接画在 README 里：

```
corpus/      controlled kernels and raw SASS evidenceknowledge/   project-wide findings, instruction notes, and encoding notespatterns/    reusable Phase 3 audit signaturesproduction/  Phase 4 real-kernel audits
```

`corpus/` 产出原始证据，`knowledge/` 把"够稳定可以反复引用的事实"沉淀下来，`patterns/` 再把"反复出现的结构"提炼为带有签名、变体、反模式、置信度、未决项的审计页，`production/` 才是真刀真枪用这些模式去解读 FlashAttention、CUTLASS 这类生产级 kernel——这就是项目目前正在迈入的 Phase 4。

## 二、证据流水线：把"猜测"和"事实"严格隔离

> 读完整个仓库后，最值得 ML/系统工程师抄走的其实不是任何一条具体 SASS 结论，而是这条流水线背后的**证据纪律**。:mdnice-bold-italic[它解决的是一切逆向工程都会踩到的同一颗雷：你以为你在描述硬件，其实你在转述自己昨天的假设。]

### 2.1 五种声明标签

每一条 claim 都必须挂一个标签，否则不被接受：

```
[OBS] 直接观察 —— 来自本地 SASS dump、ptxas 日志、运行输出或 profile[INF] 推理结论 —— 必须给出从 [OBS] 到 [INF] 的证据链[HYP] 待验证假设 —— 必须写明需要哪种受控实验来证实或证伪[RES] 假设/未决项已被后续证据回答[GAP] 当前证据无法回答的开放问题，明确记录而不是隐藏
```

这条规则强到什么程度？

::: block-1
作者明确说"不能因为外部某篇论文这么写，就把一个 `[HYP]` 升级为 `[OBS]`"——**外部权威可以支持解释，但只有本地 dump 才算第一手证据**。:mdnice-bold-italic[这种近乎偏执的纪律，让整个仓库可以被像数据库一样查询：你看到的每一句话，都能反推到一段编译命令、一段 SASS 切片，或者一段 NCU 报告。]
:::

### 2.2 控制变量法

> 每一个微核都遵循同一种实验设计：**两份 kernel 只差一个变量**，dtype、操作数顺序、unroll 因子、内存布局或编译目标——任选其一。

这样产生的 SASS diff，几乎可以直接"指认"是 ptxas 的哪一个决策导致了差异，而不是混杂在多重变化里的噪声。`corpus/basics/01_vector_add` 一直到 `corpus/tensor_cores/25_stsm_epilogue`，25 个 chapter 全部按这套规则展开。

## 三、读懂第一段 SASS：从 vector_add 到全套阅读模板

> 要理解 SASS King 的"显微镜"如何工作，最好的入口就是它的 0 号样本——一段最朴素的 `c[i] = a[i] + b[i]`。

```
__global__ void vector_add(const float* a, const float* b,                           float* c, int n) {    int i = blockIdx.x * blockDim.x + threadIdx.x;    if (i < n) {        c[i] = a[i] + b[i];    }}
```

编译成 SM120 之后，整段 dump 大致长这样（已精简到关键 20 条）：

```
/*0000*/  LDC       R1, c[0x0][0x37c]            // SP init，每个 kernel 都有/*0010*/  S2R       R0, SR_TID.X                  // per-thread: threadIdx.x/*0020*/  S2UR      UR4, SR_CTAID.X               // uniform:    blockIdx.x/*0030*/  LDCU      UR5, c[0x0][0x3a0]            // uniform:    n/*0040*/  LDC       R11, c[0x0][0x360]            // per-thread: blockDim.x/*0050*/  IMAD      R11, R11, UR4, R0             // i = blockDim*blockIdx + tid/*0060*/  ISETP.GE.AND P0, PT, R11, UR5, PT       // P0 = (i >= n)/*0070*/  @P0 EXIT                                // 越界线程直接退出.../*00d0*/  LDG.E     R2, desc[UR4][R2.64]   SBS=4  // 加载 a[i]/*00f0*/  LDG.E     R5, desc[UR4][R4.64]   SBS=4  // 加载 b[i]，复用同一 SB/*0110*/  FADD      R9, R2, R5             wait={SB4}/*0120*/  STG.E     desc[UR4][R6.64], R9/*0130*/  EXIT
```

短短 20 条指令，已经把一个 GPU kernel 的"看不见的基础设施"全暴露了：

- **两套寄存器文件并行**：跨 warp 一致的值（`blockIdx`、`n`）走 `UR` uniform 寄存器，每线程独立的值（`threadIdx`、`i`）走普通 `R`。`S2UR` / `LDCU` / `UMOV` / `ULEA` 这一族指令，对应的是项目 PATTERN-19 uniform-register dataflow[7] 所追踪的"跨文件数据流"。
- **谓词执行而非分支跳转**：越界处理通过 `@P0 EXIT` 完成，没有 `BRA`，更没有 `BSSY/BSYNC` 这种重型的发散/再汇聚机制——这是"零成本发散"。
- **scoreboard 的两种用法**：四条指针加载各占一个独立 scoreboard（`SBS=0..3`），让依赖它们的下游可以独立 ready；而两条会被同一个 `FADD` 消费的 `LDG`，反而被 ptxas 故意合并到 `SB4`，再让 FADD 一次 `wait={SB4}` 就同步两个加载。SASS King 把这条机制升级成了 PATTERN-18 Scoreboard grouping and variable-latency waits[8]，并明确点破一个常见误读："**不要因为两条 LDG 用同一个 SB，就以为它们对独立的下游是独立 ready 的** "。
- **kernel 体的 70% 都是"管道"：** 在 conclusion 里作者算过账：:mdnice-bold-italic[20 条指令里只有 1 条是真正的算术（FADD），剩下全是地址、scoreboard、控制流和守护——这正是 memory-bound kernel 的指纹。]

::: block-1
这一份 vector_add 的拆解直接成为整个项目的"阅读模板"：跳过 prologue → 找到第一处 `@P EXIT` → 识别从常量内存里加载出来的参数指针 → 定位算术段 → 看 STG → 其余都是 plumbing。
:::

## 四、张量核心的 SASS 真相：HMMA / QMMA / OMMA 家族图谱

> 如果说 vector_add 是基础语法，那么 chapter 13–25 （对应 `corpus/tensor_cores/` 目录下编号为 13_* 到 25_* 的 13 个子目录）这一套针对张量核心的逆向，才是 SASS King 的核心贡献。

Blackwell 引入了一整族此前 PTX 文档之外的 SASS 新指令，作者用 13 个 chapter 把它们逐一锚定。

### 4.1 累加器链：让"持续累加"在汇编里现形

`HMMA`、`QMMA`、`OMMA` 是 SM120 上的三大 MMA 家族（分别对应 FP16/BF16 基线、Blackwell 低精度密集 FP8/FP6/FP4、以及块缩放 FP4 峰值路径）。Chapter 13 的关键发现，是给出了一个**判定 GEMM tile 是否在做"in-place 累加"的** 稳定签名：

```
# PATTERN-02: HMMA accumulator chain[OBS] 13d:  HMMA R16, R12, R10, R16   // D 与 C 都落在 R16，原地累加[OBS] 13e:  HMMA R16, R4,  R2,  R16   // 同一累加器跨 16/32/64 条 HMMA 链[OBS] B 操作数除最后一条外都带 .reuse：R10.reuse / R2.reuse[OBS] 链上首条控制码  0x084ff60000001810      中段控制码     0x080ff60000001810      末条控制码     0x000ff00000001810
```

它说明的事情非常具体：当连续 HMMA 把 D 和 C 落在同一个寄存器基（这里是 `R16`），就意味着后一条 HMMA 把前一条的累加器作为 C 输入读回再写回，这就是 GEMM 主循环里"持续往同一片寄存器叠加 A×B 的乘积"。

作者还在 chapter 13 用 `clock64()` 微基准跑出**串行依赖链下大约 35 cycles/HMMA** 的延迟数字——并且**非常克制地** 强调这不是吞吐数字，也不是 FP16 累加器或 BF16 输入的延迟：留下一条 `[HYP]` 等待后续验证。这种"测准了什么、没测什么"都白纸黑字写清楚的态度，是这个项目最值钱的地方。

### 4.2 LDSM：让共享内存与张量核心零拷贝交接

> 要喂 MMA，先得把 shared memory 里的 tile 翻译成寄存器里的 fragment。SM120 给的指令是 `LDSM`，对应 PTX 里的 `ldmatrix.sync.aligned`。Chapter 17 把它拆得很细：

```
# PATTERN-05: LDSM fragment loading[OBS] 助记符家族  LDSM.16.M[T]88[.<N>]   // .16=b16，M88/MT88=矩阵形状/转置，N∈{ ,2,4}=宽度[OBS] 低 opcode 字节 0x3b；宽度与转置不在 opcode，而在控制码[RES] 控制码 bit 8-9 = 宽度（00=x1, 01=x2, 10=x4）[RES] 控制码 bit 14  = 转置标志[OBS] 实战组合:        LDSM.16.M88.2 R10, [R7+UR5]        LDSM.16.M88.4 R12, [R6+UR4]        HMMA.16816.F32 R12, R12, R10, RZ   // 零 NOP 直接交接[OBS] LDSM 链微基准  ≈ 33 cycles / LDSM（依赖链延迟，非吞吐）
```

这里有几个细节值得圈出：

- 第一，**操作数 A/B 的角色不靠助记符判定**，必须看下游 MMA 怎么用——这是审计准则里典型的"不要从语法形状跳到语义"。
- 第二，宽度和转置都藏在控制码里，这一类"opcode 不变、控制码变"的指令在 NVIDIA 文档里几乎完全缺席，但对反汇编工具来说极其重要。
- 第三，从 LDSM 到 HMMA 之间没有显式 NOP，靠的是 scoreboard wait mask `0xff` 完成同步——这是 `[RES]` 标签下的"已解决问题"。

### 4.3 cp.async 流水：LDGSTS / LDGDEPBAR / DEPBAR 三件套

> 真正的 GEMM/Attention kernel 不会让张量核心等内存——它们会用 `cp.async` 把下一块 tile 拷到 shared memory，同时让张量核心吃上一块。这条路径在 SASS 上是三件套：

```
# PATTERN-06: Async global-to-shared tile pipeline[OBS] PTX  cp.async.ca.shared.global.L2::128B      → SASS  LDGSTS.E.LTC128B.128 [R_smem], desc[UR_desc][R_gmem.64][OBS] PTX  cp.async.commit_group   → SASS  LDGDEPBAR        (无操作数)[OBS] PTX  cp.async.wait_group N   → SASS  DEPBAR.LE SB0, N[RES] DEPBAR.LE 的 N 编码在控制码 bit 38-39:      00 → N=0,  01 → N=1,  10 → N=2[INF] 未编译的 11 预测对应 N=3；当前 corpus 未覆盖（留为 [GAP]）
```

这一段最有教科书意味的，是它**把一个 PTX 指令族在 SASS 上的落点彻底拆开**：异步搬运 → commit group → wait group 各对应一条新指令，并且用一个 3-stage pipeline 微核刻意去枚举 `DEPBAR.LE SB0, N` 的 N=0/1/2，反推出控制码字段的 2 个 bit。剩下的 `11` 对应 `N=3` 是用模型预测出来的、并被诚实地标成 `[GAP]`——这就是 SASS King 区别于"看图说话"型逆向的核心：**预测必须被记为预测，不能伪装成观察。**

## 五、编译器的"心思"：寄存器与统一寄存器流的工程含义

29 条模式里有一类不直接对应硬件指令，而是描述 ptxas 的**决策风格**。理解它们，就是理解为什么同一段 CUDA 改一个 `__noinline__` 或一个 `-maxrregcount`，输出 SASS 会"突然"变样。

### 5.1 寄存器外溢：什么时候 ptxas 会"认输"

> PATTERN-09 的整理来自 chapter 12 的一组"压力实验"。最强反直觉的发现是：在 SM120 上，**ptxas 在很大压力区间里宁可重排活跃区间，也不情愿真的吐到 local memory**。

```
[OBS] 即使没有 spill，每个 kernel 也会有  LDC R1, c[0x0][0x37c]   // SP init[INF] 真正的"有 spill"的强信号是：紧接着出现额外的负向 IADD      12i:  IADD R1, R1, -0x128      12k:  IADD R1, R1, -0x100[OBS] 标量溢出  STL/LDL [R1+offset], R   ← 这才是 per-thread local[OBS] LDL.LU R, [R1+offset]              ← .LU 疑似 last-use 提示（[HYP] 待解码）[OBS] 12i 在 32 累加器+ -maxrregcount=24 下 170+ 对 STL/LDL，570 条指令[RES] SM120 ptxas 不会低于每线程 24 寄存器；-maxrregcount=16 会被自动抬到 24
```

这条模式给一线 kernel 工程师的工程含义非常直接：**仅凭加 `-maxrregcount` 不能证明你 spill 了** ——必须在 dump 里同时看到栈帧分配（`IADD R1, R1, -N`）和成对的 `STL/LDL`。

它还顺手澄清了三个常见反模式："`LDC R1, c[0x0][0x37c]` 不是 spill"、"`__noinline__` 调用不一定意味着 spill"、"`STS/LDS` 是 shared memory 而不是 local"——这些都是审计现场最容易混淆的细节。

### 5.2 uniform 寄存器：每束 warp 共享的"标量寄存器"

> PATTERN-19 描述的是另一种被严重低估的现象：ptxas 会自动对值做"warp 内一致性分析"，把跨线程一致的值搬进一个独立的 `UR` 寄存器文件，再让 per-thread 指令以混合操作数（`IMAD R, R, UR, R`）的形式去消费。共享内存基址、全局描述符、kernel 参数都常常住在 `UR` 里。

```
[OBS] S2UR UR4, SR_CTAID.X           // blockIdx.x 进入 uniform[OBS] LDCU.64 UR4, c[0x0][0x358]     // 全局描述符进入 uniform[OBS] UMOV UR4, 0x400                // 共享内存声明的"招牌常量"（[GAP] 含义未解码）[OBS] R2UR UR7, R1                   // 反向跨文件：per-thread → uniform（栈/local 寻址）[OBS] CS2UR UR6, SR_CLOCKLO          // 微基准里用 uniform 形式读时钟[OBS] MOV R9, UR7  ← REDUX.SUM 之后把 uniform 结果再带回 per-thread
```

这条模式真正的价值，是给一线性能调优者一个清晰的"问号清单"：

- 每一次跨文件的 `R2UR` / `MOV R, UR`，都意味着 ptxas 在为你做一次寄存器文件迁移；
- 每一次混合操作数，都意味着这条指令同时读两个文件，可能触发额外的操作数槽限制；
- 而 `UMOV 0x400` 这种至今没人解释清楚的"招牌常量"，被作者老老实实留在 `[GAP]` 里，等待后续 chapter 解决。

## 六、29 条模式的工程价值与下一站

> 读到这里，可以回头看看 `patterns/README.md`[9] 列出的 29 条模式：从张量核心计算（`02/03/04/10/21`）、矩阵内存与 epilogue（`05/06/07/28`）、控制流（`08/14/16/26/29`）、内存与寄存器（`09/11/17/19/20/23/27`），到算术与调度（`12/18/22/24`），再到 warp 协同（`01/25`）和审计方法本身（`13/15`）。

它们各自都带有同样七段式结构：朴素英语解释 / SASS 签名 / 变体 / 解读边界 / 反模式 / 未决项 / 置信度——这套结构让任何一条模式都可以**直接被另一份 production dump 引用**，而不会丢掉"哪些是观测、哪些是推理、哪些尚不知"。

这正是 SASS King 与一般 GPU 优化文章最大的不同。它**不卖性能数字、不附带 benchmark 截图、不写"How to make your GEMM 2× faster"——它把"如何在 SASS 层面安全地说出一件事"这件事本身做成了产品**。从 v0.2 的 RELEASE_NOTES 也能看出作者的边界感：

```
Not included yet:- standalone SASS disassembler;- standalone SASS assembler;- Ghidra plugin;- one-command audit CLI;- completed cross-architecture replay;- production-library audit suite.
```

不做反汇编器、不做汇编器、不做 Ghidra 插件、不急着写 audit CLI——作者宁可先用手工方式把 FlashAttention / CUTLASS / xFormers / Transformer Engine 这些库的真实 kernel 用现有 29 条模式跑一遍，等手工报告格式稳定，再考虑自动化。

这是典型的"先把刻度尺做准，再去量大象"的工程节奏，也是 Phase 4 的方向。Phase 5 才是 cubin-to-report 的流水线，Phase 6 是把整套方法重放到 SM80（A100）、SM86（3090）、SM89（4090）、SM90a（H100）、SM100a（B200）、SM120（5070 Ti / 5090）之间——一旦完成，公开世界将第一次拥有一份跨代、可比较、可追踪证据的 NVIDIA SASS 知识库。

::: block-1
GPU 早已是这一代 AI 基础设施的硅基底，但它的最底一层指令集，在过去七年里几乎处于"事实加密"状态。SASS King 没有用炫技的方式去破解这层加密，而是选择了一条更慢、更费力、却更可持续的路径：**用一份 `[OBS]/[INF]/[HYP]/[RES]/[GAP]` 的声明纪律，把每一条 SASS 观察都钉死在可复现的微核之上；再用 29 条精炼到不能再精炼的模式，把"读懂一段汇编"这件原本要靠多年口耳相传的手艺，凝成了一份可被任何工程师引用的公开文档。**
:::

> 在大模型时代抢算力的喧嚣里，这种"愿意把工具刀磨利、并把磨刀法写下来"的工程美学，正是真正稀缺的东西——而这，也许才是 SASS King 这个项目最值得被认真对待的本质意义。

参考资料

[1] SASS King: *https://github.com/florianmattana/sass-king*

[2] `docs/START_HERE.md`: *https://github.com/florianmattana/sass-king/blob/main/docs/START_HERE.md*

[3] `patterns/README.md`: *https://github.com/florianmattana/sass-king/blob/main/patterns/README.md*

[4] `corpus/tensor_cores/`: *https://github.com/florianmattana/sass-king/tree/main/corpus/tensor_cores*

[5] `CONTRIBUTING.md`: *https://github.com/florianmattana/sass-king/blob/main/CONTRIBUTING.md*

[6] Dissecting the NVIDIA Volta GPU Architecture: *https://arxiv.org/abs/1804.06826*

[7] PATTERN-19 uniform-register dataflow: *https://github.com/florianmattana/sass-king/blob/main/patterns/19-uniform-register-dataflow-and-cross-file-transfers.md*

[8] PATTERN-18 Scoreboard grouping and variable-latency waits: *https://github.com/florianmattana/sass-king/blob/main/patterns/18-scoreboard-grouping-and-variable-latency-waits.md*

[9] `patterns/README.md`: *https://github.com/florianmattana/sass-king/blob/main/patterns/README.md*

::: related-title
相关推荐
:::

::: related-links
- [暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)
- [逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)
- [逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894524&idx=1&sn=0282673a4d170d4c5a14a563ad311d78&scene=21#wechat_redirect)

:::
