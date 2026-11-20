**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：Triton、MTIA-2i、**编译器后端**、:mdnice-bold-italic[TorchInductor]、**kernel 开发效率**

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/01.jpg)

- 论文标题：**Triton for MTIA: Bridging the Programming Model Gaps for Custom AI Accelerators**
- 组织机构：Meta Platforms
- 论文链接：https://arxiv.org/pdf/2608.00325
- 开源代码地址：未公开（Meta内部生产项目）

::: related-title
**相关推荐**
:::

- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**
- **[vLLM Triton 注意力后端深度解析：跨 NVIDIA、AMD 的 GPU 架构创新优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898509&idx=1&sn=341fceedfbc28e118212eb85ae5b1807&scene=21#wechat_redirect)**
- **[从算子库到编译生成：Hexagon-MLIR 通过 MLIR 实现 PyTorch 与 Triton 在 Hexagon NPU 上的统一优化与 3.95 倍多线程加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898220&idx=1&sn=8da7c546a9ac4f04b378e2a316c35de7&scene=21#wechat_redirect)**

> 我们能不能在不牺牲开发者效率的前提下，支持当今数据中心里部署的所有 ML 加速器？:mdnice-bold-italic[Meta 用自家第二代推理芯片 MTIA-2i 给出了一个生产级答案——把 GPU 生态里最流行的 kernel 编程语言 Triton，完整地搬到一块编程模型与 GPU 截然不同的自研 ASIC 上]。

Triton 写出的 GEMM kernel 稳定达到架构 roofline 的 80% 以上，与专家手写 C++ 互有胜负；一个季度之内，生产环境中使用 Triton 的模型类型翻了三倍，覆盖约 60 种模型 50% 的层和 47% 的非 GEMM 执行时间。

::: block-1这是 Triton 第一次在非 GPU 架构上完成生产规模验证，:mdnice-highlight[高级 DSL 加编译器，完全可以成为自研芯片跨越软件生态鸿沟的那座桥。]
:::

## 本文目录

- 一、GPU 的 kernel 语言，为什么要搬到自研芯片上
- 二、MTIA-2i 是一块什么样的芯片
- 三、从零造一个 Triton 编译器
- 四、性能从哪里来：编译器优化的三板斧
  - 4.1 软件流水线：让 RISC-V 核心跑在 FFU 前面
  - 4.2 RISC-V 向量核：长尾算子的收容所
  - 4.3 负载均衡：一个 zigzag 换来 80% 加速
- 五、Inductor 定制与语言扩展：自动化与掌控力两头都要
  - 5.1 Inductor 侧的定制1：生成可分析的访存代码
  - 5.2 Inductor 侧的定制2：扩大 Triton Templates 的适用范围
  - 5.3 Inductor 侧的定制3：资源建模
  - 5.4 语言扩展
- 六、实测数据：性能、覆盖率与生产渗透
  - 6.1 kernel 级性能
  - 7.2 模型级测试
  - 7.3 生产模型渗透程度
- 七、移植经验与 DSL 战略：被放弃的 KNYFE 与留下的启示
  - 7.1 GPU kernel 移植
  - 7.2 Meta 为什么放弃了自研 DSL

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、GPU 的 kernel 语言，为什么要搬到自研芯片上

> 今天的数据中心正在同时经受两股膨胀：模型越来越多样，芯片也越来越多样，而夹在中间的 kernel 开发者成了最贵的瓶颈。

- 模型侧的膨胀可以用数字衡量。PyTorch 的核心算子库 ATen 包含近 200 个算子，完整版超过 540 个。LLM、推荐系统、自动驾驶视频处理，每类负载都在要求新的算子组合。
- 硬件侧的膨胀同样惊人：Google TPU、Amazon Trainium、Microsoft MAIA、Groq、Cerebras，再加上 Meta 自己的 MTIA 系列。Meta 的数据中心里同时跑着多代 NVIDIA GPU、AMD GPU 和自研 ASIC。

问题在于，**自研芯片为特定负载做了大量裁剪，编程模型也随之偏离 GPU 范式**。想要榨干性能，开发者必须吃透底层的内存管理与命令调度细节，为每颗芯片手工优化 kernel，:mdnice-bold-italic[而这些优化经验跨代、跨芯片几乎无法复用。模型迭代要求快，硬件适配注定慢，这对矛盾就是论文要解决的核心问题。]

在 GPU 世界，这个矛盾已经被 DSL 部分化解了。Triton 用 Python 风格的语法隐藏底层硬件细节，让开发者写出可移植的高性能 kernel，FlashAttention-2 这类复杂 kernel 就是用 Triton 快速迭代出来的。**但 Triton 从设计之初就只支持 GPU 风格的编程模型，它能否适配架构迥异的定制加速器，此前从未被验证过。**

> 论文的贡献可以归纳为三件事：

1. 从零构建了面向 MTIA-2i 的 Triton 编译器后端；
2. 定制了 TorchInductor 的代码生成以匹配芯片约束；
3. 并引入了一组最小语言扩展，把 MTIA 特有的硬件能力暴露给 kernel 作者。

最终，手写与 Inductor 自动生成的 Triton kernel 双双进入生产环境。

::: block-1整体来说，模型多样性与硬件多样性同时膨胀，kernel 开发速度成为自研芯片落地的真正瓶颈。:mdnice-highlight[Triton 在 GPU 上已证明 DSL 能化解这一矛盾，Meta 这篇论文要验证的是：同样的解法能否跨编程模型复用。]
:::

## 二、MTIA-2i 是一块什么样的芯片

> 理解这项工作的难度，先要理解 :mdnice-bold-italic[MTIA-2i 与 GPU 的三个本质差异：谁来执行计算、谁来管理内存、谁来隐藏延迟。]

MTIA-2i 是 Meta 第二代自研推理芯片。**整体架构是一个 8×8 的处理单元（PE）阵列**，通过定制的片上网络（NoC）互连，阵列四周通过交叉开关（XBAR）连接片上内存块和片外 DDR 控制器，颗粒为 LPDDR5。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/03.jpg)

::: image-caption图 1：MTIA-2i 高层架构图。64 个 PE 组成 8×8 网格，四边各有 XBAR 连接片上 MEM 与 DDR Ctrl，还配有控制核心、解压缩模块与主机接口。与 GPU 的 SM 加统一缓存层次不同，这种阵列式布局意味着数据在哪个 PE 上、走哪条 NoC 路径，都会直接影响性能——这也是后文 zigzag 负载均衡和跨 PE 归约扩展存在的根本原因。
:::

- 每个 PE 内部是典型的异构设计：两个 RISC-V 核心（一个标量核、一个支持向量扩展的向量核），外加一组固定功能单元（FFU）。
- FFU 各司其专职：点积引擎（DPE）负责 GEMM；SIMD 引擎（SE）负责量化、非线性、归约等向量操作；
- 归约引擎（RE）累加 DPE 结果并可转发给相邻 PE；
- 内存布局单元（MLU）处理转置、拼接、reshape；
- Fabric 接口（FI）负责本地内存、NoC 与片外内存之间的 DMA；
- 命令处理器（CP）统一管理 FFU 的执行、依赖检查与调度。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/04.jpg)

::: image-caption图 2：PE 内部组织图，两个 RISC-V 核心通过 PE Interconnect 与命令处理器、各 FFU 相连，LS Mem 与寄存器堆位于核心与 FFU 之间。这张图的看点在于计算的去中心化：RISC-V 核心不直接干重活，它们的主要职责是向 FFU 异步发射命令，真正的计算由 DPE、SE 这些专用单元以数据块为粒度完成。
:::

**最特殊的是片上内存的管理方式**。每个 PE 配有一块快速的本地内存（LS），但 LS 里的数据以循环缓冲（Circular Buffer，CB）的抽象来组织。每个 CB 带一对读写指针，依赖关系由指针位置强制保证：如果一次读的地址越过了写指针，说明数据还没写好，读操作会被自动阻塞，直到写指针前移。这让同一块 CB 可以像 FIFO 一样同时入队和出队，FFU 直接与 CB 交互，生产者消费者天然流水。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/05.jpg)

::: image-caption图 3：CB 抽象示意图。32B 粒度的环形缓冲区内，读指针与写指针之间的区域即当前可读数据。这个设计与 GPU 可寻址的共享内存有本质区别：共享内存里你想读哪就读哪，越界了是你自己的责任；CB 里越界读会被硬件级阻塞，正确性由指针协议保证，但内存布局与指针推进的时机也因此成了编译器必须精确求解的问题。
:::

> 由此可以归纳出 MTIA-2i 与 GPU 的三道编程模型鸿沟。

| 维度 | 描述 |
| --- | --- |
| 执行模型 | GPU 是 SIMT——线程束内线程锁步执行，每个线程处理一个元素；MTIA-2i 是 RISC-V 核心异步向 FFU 发射命令，FFU 以数据块为单位工作。 |
| 内存管理 | GPU 有硬件托管的缓存层次加软件可寻址的共享内存；MTIA-2i 是显式的 FIFO 式 CB，DMA 传输要 kernel 自己管。 |
| 延迟隐藏 | GPU 靠硬件调度器在线程束间切换；MTIA-2i 全靠软件流水线。这三点差异，决定了开源的 GPU 版 Triton 编译器无法直接复用。 |

::: block-1
:mdnice-highlight[MTIA-2i 的编程模型与 GPU 存在执行模型、内存管理、延迟隐藏三重本质差异：异步命令加固定功能单元、FIFO 式 CB 内存、软件流水线]。这正是"Triton 能否跨架构"这个问题值得一篇论文来回答的原因。
:::

## 三、从零造一个 Triton 编译器

> 既然 GPU 版编译器没法改一改就用，**Meta 的选择是基于 MLIR 从头构建一套编译器，但坚决复用上游 Triton 的语言前端。**

整个编译流程分四级，如图 4 所示。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/06.jpg)

::: image-caption图 4：Triton-MTIA 编译器高层流程图。从 Triton kernel 到 LLVMIR/C++ 的四级流水线中，每一级都对应明确的职责划分。这套分层设计的深层价值在第 8 节才完全显现：Meta 后来做的另一个低级 DSL 直接复用了这套编译器的 dialect 与 pass 子集，分层清晰让一套编译器底座可以服务多个抽象层级的前端。
:::

- **前端直接复用上游 Triton 解析器**，把 kernel 解析为 TritonIR（ttir），并扩展以支持 MTIA 的语言扩展。:mdnice-bold-italic[直接构建在上游解析器与 MLIR dialect 之上]，保证了编译器能跟上 Triton 语言的演进，不会被语言层面的升级甩下。
- **中端做通用 lowering**，把操作映射到架构；
- 后端处理架构相关的细节与片上内存管理；
- 最终生成 LLVM IR 或可读的 C++，交给 Clang 完成最后的编译链接。

> 编译器的总体策略是"贪心"：凡能用 FFU 加速的操作，一律优先映射到 FFU——块级计算用 DPE、SE、MLU、RE，数据搬运用 FI 里的 DMA 引擎；FFU 干不了的长尾操作，才落到 RISC-V 核心。

**中端最棘手的问题是内存访问的降级**。MTIA 上 DMA 的带宽利用率远高于标量 load/store，但 DMA 只支持"结构化"访问——能用基地址、维度、各维步长表达的多维访问。而 Triton 语言有三种表达内存访问的方式：tensor-of-pointers、pointer-of-tensor、tensor descriptor。其中最灵活的 tensor-of-pointers 可以表达任意 scatter/gather，恰恰也是使用最广泛的方式，哪怕访问模式本身是结构化的，很多 kernel 也习惯用它来写。编译器借助 triton-shared 开源的 IR 分析 pass，分析指针算术并把能结构化的部分改写为显式结构化形式，转化不了的才降级为 RISC-V 核心的标量访存。计算操作的映射同理：FFU 性能最优但只覆盖主流算子，RISC-V 核心支持的算子与数据类型更全但性能较低，中端根据架构约束与输入 IR 做决策。

**后端的核心任务是张量 bufferization**。Triton 语言只建模片外内存与片上张量，不区分片上存储的层次，所以 LS 的分配与复用完全是编译器的责任。编译器分析每个张量的数据使用模式，为其分配 CB 并管理读写指针。

代码生成阶段有个值得玩味的工程决策。编译器支持 LLVM IR 与 C++ 两种输出格式，性能几乎一致，:mdnice-bold-italic[但 MTIA 的专家 kernel 开发者更爱用 C++ 路径：生成的人类可读 C++ 能映射回原始 Triton 源码，并且允许开发者手工修改后再走最终编译链接]。这相当于给编译器留了一个"人工修正接口"，把编译器调试和 kernel 调试的循环都缩短了。

::: block-1整体看下来，其:mdnice-highlight[编译器的三层设计哲学：前端全复用以保语言兼容，中端贪心映射 FFU 并以结构化分析抢救 DMA，后端把 Triton 不管的片上内存与 CB 指针全部接管]。C++ 代码生成路径则是少见的以人为中心的编译器设计。
:::

## 四、性能从哪里来：编译器优化的三板斧

映射正确只是及格线，追平手写 C++ 靠的是三类优化：软件流水线、RISC-V 向量代码生成、跨 PE 负载均衡。

### 4.1 软件流水线：让 RISC-V 核心跑在 FFU 前面

> **MTIA 的硬件设计天然为软件流水线而生**：RISC-V 核心异步发射命令，依赖由 CB 指针在 FFU 层面强制执行，**核心可以尽量超前发射，把访存与计算的流水线占满**。:mdnice-bold-italic[编译器要做的，是把这份硬件善意转化为实实在在的吞吐。]

**第一项优化是有效形状传播**。Triton 操作作用于静态形状的块（block size），当张量尺寸不是块尺寸的整数倍时，kernel 用掩码（mask）保护越界访问。编译器从掩码反推出有效形状并沿 def-use 链传播，带来三重收益：确认无需填充时可以跳过缓冲区初始化；只在 $\min(row\_size, BLOCK\_SIZE)$ 个元素上做实际计算；以及完全免去掩码的实体化——既省片上内存，又省运行指令。

**第二项优化针对指令发射瓶颈**。:mdnice-bold-italic[每个 PE 有两个 RISC-V 核心，都能向 FFU 发命令，这个设计的本意就是避免单核发令成为瓶颈]。编译器静态分析 IR，把命令在两个核心间分配，生成双核发射的代码，并处理好两核共享 CB 时的同步；追求极致的 kernel 作者还可以通过语言提示显式指定操作的归属核。

**第三项是贪心 CB 尺寸分配**。Triton 张量全部静态定尺寸，编译器可以精确算出每个张量的容量需求，然后贪心选择最小倍数的分配方案，让所有分配恰好塞进 LS。:mdnice-bold-italic[kernel 通常带内层循环，这份余量正是跨迭代流水访存与计算的空间来源]。与之配套的第四项是 CB 指针管理：面对任意控制流，编译器把操作按对缓冲区的读写关系分组成静态区间，在区间边界插入指针推进，保证任意循环嵌套下的正确性。

### 4.2 RISC-V 向量核：长尾算子的收容所

> FFU 覆盖不了的**长尾操作，落到支持 RISC-V 向量扩展（RVV）的核上执行**。向量化分两步：先把操作切成目标向量尺寸的分块，再把 MLIR 向量操作映射到 RVV intrinsics。过程中有两个细节优化值得一提。

- 其一，上游 Triton 解析器总把张量-标量运算降级成张量-张量运算——先把标量广播成同形张量再算，凭空多占时间和存储；**编译器把这类操作直接降级为自定义 mtiaarith dialect 的向量-标量运算，绕开广播的实体化。**
- 其二，**相邻向量操作之间需要反复在 LS 与向量寄存器间往返，编译器通过融合向量化循环消掉冗余传输**。论文用 $y = (x_1 + x_2)^2$ 举例：融合前，加法循环把和写回 LS、乘法循环再读出来；融合后，读一次、算两步、写一次，中间结果的往返被完全消除。

### 4.3 负载均衡：一个 zigzag 换来 80% 加速

> Triton 的 PID 网格可以是任意大小，MTIA 的 PE 阵列却固定为 8×8，PID 到 PE 的映射因此成了性能变量。**编译器提供线性、轮询、zigzag 三种常见策略**。:mdnice-bold-italic[收益最直观的例子是带因果掩码的 attention：每个 PID 负责注意力矩阵的一行，行号越大工作量越大，线性映射会让小 PID 的 PE 提前闲置，zigzag 则把大小 PID 交错搭配]。**在生产负载上，zigzag 相对线性映射实测最高提速 80%。**

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/07.jpg)

::: image-caption图 5：zigzag 与线性负载分配对比。8 个 PID、2 个 PE 的玩具例子中，线性映射下 PE0 分到 PID 1–4、PE1 分到 5–8，后者总工作量明显更重；zigzag 让 PE0 拿 1、4、5、8，PE1 拿 2、3、6、7，两边工作量拉平。这个小学算术级别的技巧能在生产 attention 上换来 80% 提速，说明跨 PE 负载均衡在这类阵列架构上是第一优先级的性能问题。
:::

更进一步，MTIA 的 2D Mesh NoC 提供行广播、列归约等拓扑感知的通信原语，要用好它们，kernel 作者必须直接控制 PID 到 PE 的映射。编译器因此把映射策略开放为用户可指定的选项，这是"抽象与性能平衡"这条主线在负载均衡上的体现。

::: block-1总之，:mdnice-highlight[三类优化各管一段：软件流水线压榨单 PE 内的访存计算重叠，RVV 向量化收容长尾算子，zigzag 映射解决跨 PE 负载倾斜]。其中负载均衡的单点收益最大——一个映射策略的改变就能带来 80% 的 kernel 级提速。
:::

## 五、Inductor 定制与语言扩展：自动化与掌控力两头都要

> 手写 kernel 之外，**Meta 还要让 PyTorch 模型自动落到 MTIA 上** ——这要求 TorchInductor 生成的 Triton 代码恰好是编译器"看得懂"的形状。

### 5.1 Inductor 侧的定制1：生成可分析的访存代码

Inductor 常用的标量索引方式是 $f(x) = s_1 \cdot \lfloor x / d \rfloor + s_2 \cdot (x \bmod d)$，其中 $x$ 是标量索引，$d \in \mathbb{Z}^2$ 是被访问二维区域的形状，$s \in \mathbb{Z}^2$ 是对应的步长。

::: block-1代入一个具体例子：要访问一个行主序张量中形状为 $d = (4, 8)$ 的块、行步长 $s_1 = 8$、列步长 $s_2 = 1$，那么 $x = 13$ 对应的偏移就是 $8 \times \lfloor 13/8 \rfloor + 1 \times (13 \bmod 8) = 8 + 5 = 13$。带取模和除法的指针算术，编译器难以证明其结构性，DMA 降级就会失败。
:::

**Meta 的做法是把 kernel 的迭代空间按 $d$ 平铺，生成的指针算术里不再出现取模与除法**，Inductor 进而可以发出 `tl.make_block_ptr` 或 `tl.make_tensor_descriptor`，这两类访问保证能被映射到 DMA。

### 5.2 Inductor 侧的定制2：扩大 Triton Templates 的适用范围

> GPU 上模板主要用于 GEMM 融合和 attention，:mdnice-bold-italic[MTIA 上 Meta 把它推广到 layernorm 等复杂算子，模板由架构优化过的 kernel 宏构成，并能与多种激活函数自动融合，消掉中间张量的读写。]

模板还负责把 N 维张量操作映射为 2D Triton kernel，用占位符、可变输入和布尔标志跳过不必要的代码生成。

### 5.3 Inductor 侧的定制3：资源建模

MTIA-2i 每个 PE 只有 15 个 CB 和 384 KB LS，与 GPU SM 的寄存器堆和本地内存量级差异巨大，Inductor 后端会估算每个 kernel 的资源占用，据此调整融合决策与块尺寸，防止在存储层次间溢出。

### 5.4 语言扩展

> 语言扩展则走另一条路线——**把编译器推断不出来的硬件能力直接交给 kernel 作者**。论文给出两组动机：:mdnice-bold-italic[暴露 MTIA 架构特性，以及增强语言表达力。]

前者包括 `tl.mtia.get_pe_id` 与 `tl.mtia.get_pe_grid_size`，让 kernel 实例查询自己落在哪个 PE 上，配合 2D Mesh 拓扑做跨 PE 协同；以及 `tl.mtia.re_reduce`，显式调用归约引擎的硬件归约网络完成跨 PE 归约，注意只有归约网络末端的 PE 拿到完整结果。

libdevice 机制则让用户直达 SIMD 引擎，配合编译器指令决定调用降到 FFU 还是 RISC-V，未修改的 GPU Triton kernel 因此能无缝跑在 MTIA 上并自动吃到硬件优化。

表达力方面最典型的是 `tl.extract_slice`。:mdnice-bold-italic[原生 Triton 的内存与计算操作都以完整张量为粒度，不支持子张量访问——在有硬件缓存的 GPU 上这很合理，反复访问的区域会被缓存；但 MTIA 的 LS 是软件托管的，数据复用必须显式表达]。对于嵌套循环内层反复使用大张量切片的场景，kernel 作者可以先用一次 `tl.load` 把张量搬进 LS，再用 `tl.extract_slice` 按偏移取切片，编译器在验证安全后把拷贝改写为视图。

此外，借助 ctypes.Structure 的自定义类型支持，:mdnice-bold-italic[Triton kernel 可以直接使用 TensorView、CoreId 这类 MTIA C++ 软件栈里的数据结构，打通了 Triton 与既有 C++ kernel 生态之间的数据通道。]

::: block-1整体看下来，:mdnice-highlight[Inductor 定制解决"自动生成的代码能被编译器优化"的问题，语言扩展解决"编译器推不出的意图由人来表达"的问题]。一收一放之间，是全文反复出现的母题：**抽象层级与性能掌控力的平衡**。
:::

## 六、实测数据：性能、覆盖率与生产渗透

所有测量都在 MTIA-2i 硅片上完成，配置取自真实生产模型，对照组是架构 roofline 或专家手写 C++。

### 6.1 kernel 级性能

> GEMM 测试采用生产推荐模型与生成式模型的形状与精度配置，Triton 版本使用了语言扩展，并按形状与数据类型实现了多种平铺与负载分配变体，由顶层调度器在运行时按静态策略分发。结果是全部配置稳定超过 roofline 的 80%，与专家手写 C++ 互有胜负、整体持平。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/08.jpg)

::: image-caption图 6：GEMM kernel 性能对比。浅蓝 Triton、深蓝 C++，横轴为各 GEMM 配置。Triton 全程站在 80% 线以上，且没有明显短板配置。考虑到 GEMM 是 FFU 映射、双核发射、贪心 CB 分配、语言扩展全部叠满的算子，这组图实际上是对整套编译器与扩展体系的综合验收。
:::

**融合 FlashAttention 的表现更有戏剧性**。以算子拆分后走优化 kernel 库的方案为基线，推荐模型配置下归一化加速比约 1.2–1.35 倍，生成式模型配置下则冲到约 5–6 倍，序列越长优势越大。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/09.jpg)

::: image-caption图 7：attention 性能归一化对比。左图为推荐模型配置，加速比稳定在 1.2 以上；右图为生成式模型配置，加速比从约 1 倍一路爬升到 6 倍以上。融合 kernel 的价值随序列长度放大：长序列下拆分方案的中间结果读写开销占比更高，融合省掉的正是这部分访存。
:::

**长尾 kernel 方面**，论文评测了 17 个生产模型在用的 kernel，按算子聚合取几何平均、归一化到专家 C++。整体与基线持平，且:mdnice-bold-italic[图 8 中存在明显离群点——那些此前没有优化过的场景，Triton 快速补位后大幅反超。]

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/10.jpg)

::: image-caption图 8：长尾 kernel 性能对比。多数算子的归一化性能围绕 1.0 波动，个别达到 1.5 以上，也有少数落在 0.65 附近。这张图的价值不在证明 Triton 更快，而在证明 Triton 不慢：对于长尾算子这种专家时间最稀缺的领域，能用几天而非几周拿出性能持平的 kernel，本身就是胜利。
:::

> 再看 Inductor 自动化链路。Meta 从 10 个内部模型中提取 PyTorch 算子用法，按算子、数据类型、形状、非张量参数的每种唯一组合复现 PyTorch 测试，统计编译通过率。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/11.jpg)

::: image-caption表 1：Inductor 编译流程的算子级覆盖统计。推理与训练的成功率分别为 94.5% 与 94.7%，失败主要来自 Triton error（约 4.5%–4.6%），Inductor error 不足 0.1%。论文指出剩余失败大多归因于 MTIA 硬件限制——某些测试用例与 FFU 约束天然不兼容。换言之，接近 95% 的通过率里，编译器本身能继续压榨的空间已经不大，天花板在硬件。
:::

自动融合的效果用生产模型中常见的 layernorm-sigmoid 组合验证：Inductor 通过 Triton Templates 自动生成的融合 kernel，与手写融合基线在各张量形状下性能相当。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/12.jpg)

::: image-caption图 9：自动融合 layernorm-sigmoid 与手写基线的归一化加速比。各形状下加速比在 1.1–1.35 之间，全部站在 1.0 以上。自动融合能打赢手写基线，一部分功劳要记给模板的预优化宏与形状特化——通用手写 kernel 需要兼顾多种形状，Inductor 生成的 kernel 只需要服务一种。
:::

### 7.2 模型级测试

> 模型级测试用一个推荐模型的训练过程：Inductor 生成的 kernel 占 graph 模式执行时间的 22%（GEMM 与 layernorm 走手工优化 kernel，不计入），:mdnice-bold-italic[总体比被替换的手写 kernel 快 31%；算上 eager 模式部分，端到端 QPS 提升 4%。融合是性能来源的关键词：平均融合深度 3.9 个算子，最深达 12 个，数据搬运与类型转换算子大量被并入逐元素算子。]

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/13.jpg)

::: image-caption图 10：推荐模型训练的子图性能。右轴的平均融合深度折线与加速比柱形同向抬升：融合最深的子图加速约 30%，融合浅的子图也有 5% 左右。即使融合深度浅，形状特化的 kernel 也常在运行时追平通用手写 kernel。
:::

### 7.3 生产模型渗透程度

> 最有说服力的是生产渗透曲线。从 Q1'25 到 Q2'25，三个产品组中部署 Triton 的模型类型总数翻了三倍。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/14.jpg)

::: image-caption图 11：Triton 部署的模型类型数量。PG1 从约 14 种增至约 18 种，PG2 与 PG3 则从 2 种左右分别跃升到 20 种以上。PG2、PG3 起步基数低但增速凶猛，说明 Triton 的价值在首批标杆模型跑通之后迅速外溢——这正是 DSL 复用性带来的扩散曲线。
:::

> 与之对应，Triton 在生产模型中的层覆盖率与非 GEMM 运行时占比双双约翻倍，达到摘要中的 50% 与 47%。

![文章插图](/articles/2026-08-11-triton-asic-meta-mtia-2i-65d7cd64/images/15.jpg)

::: image-caption图 12：Triton 的生产足迹。层覆盖率从约 28% 升至约 50%，非 GEMM 运行时占比从约 20% 升至约 47%。运行时占比低于层覆盖率这个细节值得注意：被 Triton 覆盖的层平均而言比 GEMM 层更轻，这也符合直觉——先被吃掉的是数量多但单个体量小的长尾部分。
:::

::: block-1总之，:mdnice-highlight[性能层面，Triton 在 GEMM 上追平专家 C++、在长序列 attention 上拿到 6 倍加速；自动化层面，Inductor 算子通过率近 95%、模型级提速 31%；生产层面，一个季度内模型数翻三倍、覆盖 60 种模型的半数层]。三组数据共同坐实了生产规模这四个字，相当亮眼了👀。
:::

## 七、移植经验与 DSL 战略：被放弃的 KNYFE 与留下的启示

论文最后一章是两个战略级复盘：GPU kernel 移植到底有多顺滑，以及 Meta 为什么放弃了自研 DSL。

### 7.1 GPU kernel 移植

> 先看移植实验。Meta 把 16 个 GPU Triton kernel 搬到 MTIA，复杂度从张量布局变换到融合 attention 不等。其中 11 个零修改直接跑通且性能够用。

**最有意思的坑来自 Triton 在 GPU 上的未定义行为**：`tl.load` 的非块指针形式接受可选的 mask 与 other 参数，不指定 other 时，某些 GPU 后端会把被掩掉的元素初始化为零，而 MTIA 保持未初始化。:mdnice-bold-italic[依赖这个隐式行为的 kernel 在 MTIA 上轻则数值错误，重则越界访存——因为索引本身可能是从未初始化值里读出来的。修复后的 kernel 在 GPU 上同样可移植]。

剩下 5 个需要重写的 kernel 分三类：两个用了取模回绕的指针算术、无法降级为 DMA，改动很小；一个栽在混合精度处理的数值差异上，需要显式类型转换；还有两个复杂度太高，编译器优化不够，只能按 MTIA 的方式重写——显式数据复用、定制负载映射。

11/16 的直接移植率，加上修复后仍可回流的特性，让 Triton 成为快速承接既有 kernel 库、加速新模型落地的杠杆。

### 7.2 Meta 为什么放弃了自研 DSL

> 再看被放弃的方案。在 Triton 之前，Meta 自研过低级 DSL KNYFE，直接把 CB、DMA、片上内存等架构细节暴露给 kernel 作者，性能可以追平专家 C++，但陌生语法的门槛高到只有最资深的专家能用。

**性能与可及性不可兼得，Meta 最终弃船**。这个教训与行业趋势形成对照：近来 DSL 开始往回找性能，路线有两条——CuTeDSL、Gluon 这类架构定制的新 DSL，或者 TLX 这样给 Triton 挂低级扩展。

**Triton-MTIA 的扩展走的正是 TLX 式路线，但暴露的操作与控制按 MTIA 裁剪**：:mdnice-bold-italic[基础 Triton 保持通用与可移植，性能敏感处按需使用 MTIA 专属扩展]。Meta 还另做了一个直接暴露架构特性的低级 DSL，它不维护独立后端，直接复用 Triton-MTIA 编译器的 dialect 与 pass 子集。**一套编译器底座，多个抽象层级的前端** ——这是分层编译架构在工程组织上的红利。

::: block-1
KNYFE 的教训是性能换不来采用率，Triton 的胜利是生态与抽象层级的胜利。给所有自研芯片团队的启示浓缩成一句话：:mdnice-highlight[与其造一门新语言，不如把编译器接到已有的生态上，再用最小扩展找回架构特性。]
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[完全由 Kimi K3 写出的 MiniTriton，碾压 Torch 与原生 Triton？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903963&idx=1&sn=63acd066cbada9c93dd89cb384e82169&scene=21#wechat_redirect)**
- **[零侵入替换训练热路径！Liger-Kernel：面向 LLM 训练的高性能 Triton Kernel 与 API 全解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903793&idx=1&sn=15c6caf35822547f67412d670bb7b60e&scene=21#wechat_redirect)**
- **[Triton 算子如何越过 PTX 直抵 Blackwell：CUDA TileIR 后端的诞生，Triton-to-tile-IR](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903681&idx=1&sn=a30c96b2bfbc7a32d9745ca6d303ddfd&scene=21#wechat_redirect)**
- **[从 196 到 547 TFLOPS！Triton-TLE：告别手动 barrier，用生产消费模型实现 FlashMLA 90% 的性能！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901317&idx=1&sn=0cdf1ca561fe14ccb56ef909b34bf661&scene=21#wechat_redirect)**

:::
