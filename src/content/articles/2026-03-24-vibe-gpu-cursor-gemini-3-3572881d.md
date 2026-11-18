关键词：**Vibe-GPU**、Gemini 3 Pro、**SIMT Stack**、**RTL 代码**、:mdnice-bold-italic[硬件架构]

> 如今，大模型在硬件开发领域已实现突破性进展——Gemini 3 Pro （现在已是 3.1）不仅对 GPU 微架构理解专业、堪称 RTL 专家，更能像专业工程师般精准定位微架构层级 bug、对齐汇编与硬件调试输出。

其 Vibe Coding 模式可完成 GPU 硬件、软件栈及测试用例的全流程开发，而 SIMT Stack 的 Split-Swap-Merge 机制更以创新设计适配分支嵌套的线程调度逻辑。正如编程的核心是逻辑构建，这类工具正将人从代码编写中解放，推动编程抽象层级整体提升。

::: block-1
也正因如此，作为研究计算机体系结构的从业者，作者萌生了测试 Gemini 3 Pro 底层打灰能力的想法，想看看 Cursor + Gemini 3 Pro 能否直接写出一个能跑的“GPU”。
:::

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/01.webp)

- **Vibe-GPU: 用 Cursor+Gemini 3 Pro 写一个“GPU”**
- 作者：知返，https://zhuanlan.zhihu.com/p/1975609507276136889
- 作者主页：pkuzhou.github.io
- 项目：github.com/PKUZHOU/Vibe-GPU
- 6000 字，阅读 18 分钟，**播客 22 分钟**

::: related-title
相关推荐
:::

::: related-links
- [我已经一年多没写过一行代码了，算是 vibe coding 老手了](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897135&idx=1&sn=1ec81e600f5000e6fce34f76c8268be0&scene=21#wechat_redirect)
- [逼近 Opus 模型能力极限！烧掉2万刀，16个Claude暴力“肝”出10万行编译器，竟真能跑Linux内核！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897026&idx=1&sn=f1d236dd579687a588596631ac493b85&scene=21#wechat_redirect)
- [微基准视角下 NVIDIA Blackwell 架构深度解析：FP4/FP6赋能LLM 2.5倍推理，36.3TFLOPS FP64重塑科学计算](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896057&idx=1&sn=e3b882b56e8516ff1163d4cc300a25dd&scene=21#wechat_redirect)

:::

> 最近听说刚发布的 Gemini 3 模型很牛，各种 Fancy 的前端 UI 程序手到擒来，比如可以写各种游戏：

- 如何评价谷歌正式发布的 Gemini 3？哪些信息值得关注？
- https://www.zhihu.com/question/1974260216498500253/answer/1974394037051086213
- 该回答摘要：Gemini 3 Pro经测评被认定为当前世界最强AI模型，在3D游戏编程测试中表现优于Claude Sonnet 4.5、GPT 5.1等模型，多模态图像识别能力碾压竞品，数学逻辑也稳压无工具GPT5、Grok4，综合能力顶尖，仅写作能力未达预期，整体表现符合期待。

但是本牛马是研究计算机体系结构的，而且也不会做奥数，看完这些也只能觉得好像生成的东西花花绿绿的是挺不错的，但是达不到“Amazing”的地步。

::: block-1
恰好最近业余时间在重新学习研究 GPU 架构相关的知识，索性测试一下 Gemini 3 的底层打灰能力，看看 Cursor + Gemini 3 Pro 能不能直接写出一个能跑的“GPU”出来。
:::

最终项目代码已经传上 Github，感兴趣的同学可以玩一玩。

- PKUZHOU/Vibe-GPU: Vibe Coding A GPGPU via Cursor + Gemini 3 Pro
- github.com/PKUZHOU/Vibe-GPU

> 先说 Vibe Coding 完的结论：

- Gemini 3 pro 对 GPU 微架构的了解很专业，而且知道怎么去实现一些 GPU 核心功能，比如 SIMT Stack，Warp Scheduler 等等
- 可以认为是 RTL 专家了，对如何写硬件代码，如何进行测试和问题定位都很专业。这和一两年前的大模型给我的体感完全不一样。早期大模型对硬件代码的掌握基本上是高中生水平
- 对软硬件接口的定义很清晰，掌握如何基于一个自定义的硬件去写上层的软件栈的能力
- 仍然会有一些问题需要解决，比如上下文长度过长之后会”说胡话“，需要手动暂停模型输出

> Code is cheap, show me the talk !

## 本文目录

- 一、明确 GPU 设计需求
  - 1.1 硬件架构和 RTL 代码层面
  - 1.2 软件框架和测试
  - 1.3 文档
- 二、硬件架构设计和 RTL 代码
- 三、配套软件栈
- 四、性能评估
- 五、后记
- Supplementary Materials
  - 1. 栈表项结构 (Stack Entry)
  - 2. 核心流程：Split - Swap - Merge
  - 3. 硬件实现逻辑 (sm_core.sv)
  - 嵌套支持 (Why Stack?)

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、明确 GPU 设计需求

> 首先我要写出我的大概需求：

### 1.1 硬件架构和 RTL 代码层面

- GPU 支持四个 SM Core。每个 SM 可以最多并发跑 4 个 Warp，每个 Warp 8 个线程，一共 128 个线程。
- SM 内支持 Warp 调度
- 支持基于 SIMT Stack 的分支跳转的优化：
- 支持 FP8 乘法
- 每个 SM Core 实现流水线微架构，处理好各种数据冒险

### 1.2 软件框架和测试

- 生成配套的“编译器/汇编器”工具，支持类似 CUDA 的前端
- 基于 verilator 生成 simulation 框架
- 自动生成“CUDA”测试用例，涵盖矩阵向量乘法，分支跳转测试，FC 层等，并且保证完成通过测试

### 1.3 文档

- 生成软硬件框图、README 文档等

注意，以上的需求其实不是我一次性提的，很多是半路拍脑袋加上的，比如加一个 FP8 单元，比如改造成流水线架构等。不过 Gemini 还是很聪明的，交互过程中感觉我给的需求都理解的非常到位。经过大概三个小时的交互（大部分时间都在等输出），最后 Gemini 给我做了如下一套东西：

## 二、硬件架构设计和 RTL 代码

> 总体架构图（也是自动生成的），严格上按我说的来做了。四个 SM Core 共享了一块 Unified Shared Memory。目前没有外接 DRAM。

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/03.webp)

::: image-caption
架构框图
:::

自动生成的 ISA，里面除了 FP8 Add/Mul 是我要求加的。一共 32 个寄存器，每个寄存器 32bit

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/04.webp)

::: image-caption
计算指令集
:::

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/05.webp)

::: image-caption
访存控制指令集
:::

除了通常就会有的计算/控制类指令集，还有一些和 GPU 相关的指令集，比如获取 Warp ID, SM ID，Thread ID 等等。

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/06.webp)

::: image-caption
GPU 相关指令集
:::

根据这些指令集，自动生成了 RTL 实现，整体项目文件结构如下：

```
.├── apps/               # Application Kernels (Python)│   ├── app_cuda_matmul.py   # Matrix Multiplication│   ├── app_cuda_fc.py       # Fully Connected Layer│   └── ...├── docs/               # Documentation & Images├── rtl/                # SystemVerilog Source Code│   ├── sm_core.sv      # Streaming Multiprocessor│   ├── simt_stack.sv   # Divergence Stack│   ├── l1_cache.sv     # L1 Cache│   └── ...├── sim/                # Verilator Simulation Environment├── tests/              # Test Suite│   ├── test_suite.py   # Automated Test Runner│   └── ...└── tools/              # Compiler & Assembler    ├── vibe_cuda.py    # Compiler Frontend    ├── assembler.py    # Assembler    └── isa.py          # ISA Definitions
```

可以随便打开一个 RTL 文件品鉴一下，比如 ALU 单元：

```
`include "defines.svh"module alu (    inputlogic [5:0] opcode,    inputlogic [`DATA_WIDTH-1:0] op_a,    inputlogic [`DATA_WIDTH-1:0] op_b,    inputlogic [`DATA_WIDTH-1:0] imm,    inputlogic [2:0] thread_id, // Used for TID instruction (3 bits for 8 threads)    inputlogic [`WARP_ID_WIDTH-1:0] warp_id, // New: Warp ID    inputlogic [1:0] sm_id,     // Used for SMID instruction    // CSR Inputs    inputlogic [`DATA_WIDTH-1:0] csr_cycle,    inputlogic [`DATA_WIDTH-1:0] csr_instret,    outputlogic [`DATA_WIDTH-1:0] result);    logic [7:0] fp8_add_res;    logic [7:0] fp8_mul_res;    // Instantiate FP8 Unit    // Only using lower 8 bits of operands    fp8_unit u_fp8_unit (        .a(op_a[7:0]),        .b(op_b[7:0]),        .add_res(fp8_add_res),        .mul_res(fp8_mul_res)    );    always_combbegin        case (opcode)            `OP_ADD:  result = op_a + op_b;            `OP_SUB:  result = op_a - op_b;            `OP_LDI:  result = imm; // Immediate is already sign extended in decode            `OP_AND:  result = op_a & op_b;            `OP_OR:   result = op_a | op_b;            `OP_MOV:  result = op_a;            `OP_MUL:  result = op_a * op_b;            `OP_SLL:  result = op_a << op_b[4:0]; // Shift Left            `OP_SRL:  result = op_a >> op_b[4:0]; // Shift Right Logical            `OP_FADD: result = {24'b0, fp8_add_res}; // Zero-extend FP8 result            `OP_FMUL: result = {24'b0, fp8_mul_res}; // Zero-extend FP8 result            `OP_TID:  result = {29'b0, thread_id};            `OP_SMID: result = {30'b0, sm_id};            `OP_WARPID: result = {30'b0, warp_id};            `OP_CSR: begin                if (op_a == 0) result = csr_cycle;                elseif (op_a == 1) result = csr_instret;                else result = 0;            end            // Address Calculation for Memory Ops (Reg + Imm)            `OP_LDW:  result = op_a + imm;            `OP_STW:  result = op_a + imm;            default:  result = 32'b0;        endcase    endendmodule
```

虽然逻辑上确实不复杂，但是代码写的比较干净利落。调用了自己生成的 FP8 运算单元实例，ALU 计算产生的 FP8 的计算结果给补齐到了 32bit。这里我追问他为什么这么做，解释的也很专业很有自己的设计想法：

> 在 rtl/alu.sv 中，FP8 运算结果被 Zero-Extend 到 32-bit 是为了保持数据通路的一致性，并简化寄存器文件的设计。

1. 统一的寄存器宽度 (32-bit)在 Vibe GPU 的架构中，所有通用寄存器 (R0 - R31) 和数据总线都是 32 位宽 (DATA_WIDTH = 32)。
2. 为什么不把 4 个 FP8 打包 (Pack) 进一个寄存器？虽然将 4 个 FP8 打包进 32 位寄存器（SIMD within a register）可以提高存储密度，但这会显著增加硬件复杂度：
3. 软件/编译器视角在 Python 编译器 (`vibe_cuda.py`) 看来，FP8 只是一个数值类型。

其他模块我就不说了，感兴趣的可以继续在 Github 仓库里面品鉴 Gemini 的代码：github.com/PKUZHOU/Vibe-GPU

## 三、配套软件栈

> 在硬件基本上生成完之后，我要求 Gemini 再生成一个对应的软件栈，支持用户用高层次语言编程，而不是写机器码。我给的需求是“生成类似 CUDA 的前端，但是用 Python 来写”。

Gemini 自动补全了软件栈：Level 1 是对接 ISA 的汇编，Level2 是我要求的“CUDA”前端。

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/07.webp)

::: image-caption
类 CUDA 软件栈
:::

```
@cuda.jit(block_dim=32)def matmul_kernel(a, b, c):    # Grid: 128 threads (4 blocks of 32 threads)    # C = A * B, where A is 16x16, B is 16x8, C is 16x8    # Each thread calculates one element of C.    # Calculate global ID    gid = cuda.blockIdx.x    bdim = cuda.blockDim.x    gid = gid * bdim    tid = cuda.threadIdx.x    gid = gid + tid    # Map 1D global_id to 2D (row, col) for C matrix (16x8)    # WIDTH_B = 8    row = gid >> 3# gid / 8    col = gid & 7# gid % 8    acc = 0    # K loop (inner loop for matrix multiplication)    for k in range(16): # K is 16        # Load A[row][k]        # idx_a = row * 16 + k        tmp_a_idx = row << 4        tmp_a_idx = tmp_a_idx + k        val_a = a[tmp_a_idx]        # Load B[k][col]        # idx_b = k * 8 + col        tmp_b_idx = k << 3        tmp_b_idx = tmp_b_idx + col        val_b = b[tmp_b_idx]        # acc += val_a * val_b        tmp_prod = val_a * val_b        acc = acc + tmp_prod    # Store C[row][col]    c[gid] = acc
```

上面的代码是自动生成的，基于“类 CUDA”编程语言实现的矩阵乘法 kernel。写法上确实和 CUDA 程序是一致的，每个线程负责一个 output element 的计算。

在 `vibe_cuda.py` 里面，生成了一个 python 写的“cuda”解释器，并完成寄存器分配、线程分配映射等功能，最后把前端的代码转换到 ASM。Gemini 写了 Assembler 和 Disassembler ，这个相对简单。

```
class Assembler:    def __init__(self, base_data_addr=1024):        self.insts = []        self.labels = {} # name -> index        self.patches = [] # (index, label_name, opcode, rd, rs1, rs2)        self.data_seg = [] # list of bytes/words        self.base_data_addr = base_data_addr        self.data_offset = 0    def get_current_index(self):        return len(self.insts)    # --- Instructions ---    def nop(self):        self.insts.append(make_inst(OP_NOP, 0, 0, 0, 0))    def add(self, rd, rs1, rs2):        self.insts.append(make_inst(OP_ADD, rd, rs1, rs2, 0))    def sub(self, rd, rs1, rs2):        self.insts.append(make_inst(OP_SUB, rd, rs1, rs2, 0))    def mul(self, rd, rs1, rs2):        self.insts.append(make_inst(OP_MUL, rd, rs1, rs2, 0)).... 省略很多行
```

可以对生成的 Matmul 程序进行反汇编得到 GPU 的汇编程序。

```
0000: LDI R1, 10230004: LDI R2, 10008: ADD R1, R1, R2000C: LDI R2, 10230010: LDI R3, 650014: ADD R2, R2, R30018: LDI R3, 1023001C: LDI R4, 10230020: ADD R3, R3, R40024: LDI R4, 66/省略很多行0158: HALT
```

接着我让 Gemini 生成了一系列其他的测试用例，并且完成整体的端到端测试。

运行 `test_suite.py` 来跑一遍七个测试用例。测试时会调用 verilator 仿真器来对 RTL 代码进行仿真，并且验证 memory 内最终输出值与是否一致

```
python tests/test_suite.py--- Testing Debug Kernel (Gid) ---[EXEC] python3 apps/app_cuda_debug.py[EXEC] cd sim && make clean && make run[PASS] Debug Test Passed--- Testing Load Kernel (Copy A to C) ---[EXEC] python3 apps/app_cuda_load_test.py[EXEC] cd sim && make clean && make run[PASS] Load Test Passed--- Testing Loop Kernel (acc += 1 for 16 times) ---[EXEC] python3 apps/app_cuda_loop_test.py[EXEC] cd sim && make clean && make run[PASS] Loop Test Passed--- Testing Divergence Kernel (Odd=11, Even=12) ---[EXEC] python3 apps/app_cuda_divergence.py[EXEC] cd sim && make clean && make run[PASS] Divergence Test Passed--- Testing VecMul Kernel (2 * 3 = 6) ---[EXEC] python3 apps/app_cuda_vecmul_test.py[EXEC] cd sim && make clean && make run[PASS] VecMul Test Passed--- Testing Matrix Multiplication (Random Data) ---[EXEC] python3 apps/app_cuda_matmul.py[EXEC] cd sim && make clean && make run[PASS] MatMul Test Passed!--- Testing Fully Connected Layer ---[EXEC] python3 apps/app_cuda_fc.py[EXEC] cd sim && make clean && make run[PASS] FC Test PassedSummary: 7/7 Tests Passed
```

**如何通过这些测试用例其实是最花时间的地方。** Gemini 花了很长时间去检视和修改 RTL 代码和测试用例，最终才保证 7 个测试用例都通过了。**这其实是模拟 RTL 开发过程中的“硬件验证”过程。** 在实际芯片设计开发中，也是芯片验证最花人力和时间。观察 Gemini 在 debug 过程中的输出，我有以下几点感悟：

- **Gemini 很知道如何 Debug RTL 代码** ，会在 RTL 的代码里面自动加上各种 debug 输出，比如插入 Display ： $display("Time %0t: Mem Write SM%0d T%0d Addr %0d Data %0d",$time, i, j, mem_addr_l1[i][j], mem_wdata_l1[i][j]); 这种 debug 控制语句。而且**在 debug 过程中写了反汇编工具，直接对着汇编出来的代码，一行一行地和 debug 输出对齐** ，这个和人自己去 debug 模式完全一样了。
- **似乎学会 Best-effort 工作模式了** ，如果在一轮对话中没有办法把 bug 完全搞定，他会输出目前的进展发现和可能的猜测，然后结束对话。可以通过给提示或者“根据你的发现再试一下”这种 prompt，让大模型再次进行 debug。基本上两轮都能把 BUG 定位出来。我猜这个设定也许是 Cursor 或者 Gemini 怕一次对话过长影响最终的效果？
- 最终发现的 bug 偏微架构层级，比如冒险没处理好，分支逻辑没处理好等等。**由于 RTL 代码写的不对等等低级错误几乎没有出现**
- 如果对话积累的太长，有一定几率出现输出崩溃的情况，但是不是很常见

当然，我作为 Human 可以手动打开波形图来观察更细的信号，不过目前大模型应该很难直接看波形图，而且也确实没用到。

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/08.webp)

::: image-caption
Vibe-GPU matmul kernel 波形图
:::

## 四、性能评估

> 最后，我让 gemini 自己生成了一个 benchmark 程序来测试 GPU 的性能指标，包括 IPC 和虚拟的 GOPS 性能指标（100MHz)

```
----------------------------------------Performance Results (Thread 0)----------------------------------------Start Cycle: 158End Cycle:   272Delta Cycle: 114----------------------------------------Start Inst:  158End Inst:    202Delta Inst:  44----------------------------------------IPC (Instruction Throughput): 0.3860----------------------------------------Performance Metrics (1 Block, 32 Threads)Total Ops Executed: 6144Ops Per Cycle:      53.8947Est. GOPS @ 100MHz: 5.3895 GOPSChecksum: 22
```

![文章插图](/articles/2026-03-24-vibe-gpu-cursor-gemini-3-3572881d/images/09.webp)

Gemini 自己对这个指标的评价似乎还不错，也知道哪里是性能瓶颈。理论上可以继续深入进行架构优化。

## 五、后记

> 整个过程中我只写 prompt，没有参与调整过一行代码。Gemini 给人的体感是冷静且专业，找问题稳准快。

当然目前看 Vibe-GPU 也就达到了教学工具水平，主要是因为还没有精力去一步步地细化，增加更多的功能和指令（比如 Tensor Core，线程同步，TMA ）等等。**其实 vibe coding 到这里我已经相信只要我需求提的好，Gemini 肯定能帮我挨个写出来了。** 感兴趣且有时间的同学可以试一下，到底能不能 prompt 出一个“中高级难度”的 GPU 出来。

另外很想测试一下 Gemini 对 EDA 工具的使用，比如用 vivado 的命令行工具，完整跑一遍 FPGA 综合和上板测试流程。但是家里没有这个环境，只能作罢。

最后吐槽一下，生成这个工具基本上把我 Cursor 这个月的额度已经用完了，再用似乎要继续充钱。**目前大模型 Token 还是贵啊，特别是强力模型。降本增效任重道远。**

## Supplementary Materials

> 为了展示 Gemini 对微架构和对应实现的理解深度，我把它在实现 SIMT Stack 功能时，给我提供的讲解贴到后面，感兴趣的可以仔细读一读：

**以下为模型输出：**

因为所有线程共享一个 PC（程序计数器），当遇到分支（有的线程要走 if，有的要走 else）时，硬件不能同时执行两边。必须串行化执行：先让一部分线程执行 if，把另一部分“挂起”存入栈中；等 if 跑完了，再把栈里的线程“唤醒”执行 else，最后把两部分合并（Re-converge）。

我将用代码图示为你讲解 Vibe GPU 中实现的 SSM (Split-Swap-Merge) 机制。

### 1. 栈表项结构 (Stack Entry)

> 在 rtl/simt_stack.sv 中，每一行栈存储了三个关键信息：

```
[ Type | Mask | PC ]
```

- Type (1 bit):
- Mask (8 bits): 这一层被挂起的线程掩码。
- PC (32 bits): 这些线程该从哪里开始（或继续）执行。

### 2. 核心流程：Split - Swap - Merge

> 假设我们有 8 个线程 (T0-T7)，初始掩码全是 1 (11111111)。

代码如下：

```
if (tid % 2 == 0):  # 偶数线程 (0,2,4,6)    # Block Aelse:               # 奇数线程 (1,3,5,7)    # Block B# End (Re-converge)
```

#### 阶段 1: Split (遇到分支)

> 遇到 BEQ/BNE 指令。

- 判断: 偶数线程满足条件 (Taken)，奇数线程不满足 (Not Taken)。
- 动作: 立即执行 Taken 的路径。将 Not Taken 的路径 Push 到栈上。

```
当前状态 (Current Execution):PC: Block A StartMask: 0 1 0 1 0 1 0 1 (只剩偶数线程活跃)------------------------------------------------------| Type |   Mask (8b)     |       PC (Target)         ||------|-----------------|---------------------------||  DIV | 1 0 1 0 1 0 1 0 | Block B Start (奇数线程)  |  <-- Top------------------------------------------------------
```

#### 阶段 2: Swap (Block A 结束，遇到第一个 JOIN)

> 偶数线程跑完了 Block A，遇到了编译器插入的 JOIN 指令。

- 检查栈顶: 发现是 DIV 类型（说明栈里的是还没跑的另一半）。
- 动作 (Swap):

```
当前状态 (Current Execution):PC: Block B StartMask: 1 0 1 0 1 0 1 0 (切换到奇数线程活跃)------------------------------------------------------| Type |   Mask (8b)     |       PC (Target)         ||------|-----------------|---------------------------|| CONV | 0 1 0 1 0 1 0 1 | End Address (偶数线程)    |  <-- Top------------------------------------------------------
```

#### 阶段 3: Merge (Block B 结束，遇到第二个 JOIN)

> 奇数线程跑完了 Block B，再次遇到了 JOIN 指令。

- 检查栈顶: 发现是 CONV 类型（说明栈里的是之前跑完等着合并的）。
- 动作 (Merge):

```
当前状态 (Current Execution):PC: End Address (继续执行)Mask: 1 1 1 1 1 1 1 1 (所有线程恢复活跃)------------------------------------------------------|      |                 |                           ||      |    (Empty)      |                           |  <-- Top------------------------------------------------------
```

### 3. 硬件实现逻辑 (sm_core.sv)

> 结合我们之前的 bug fix，硬件逻辑如下：

```
// rtl/sm_core.sv 伪代码简化if (instruction == OP_JOIN) begin    if (!stack_empty) begin        // 情况 A: 栈顶是 DIV (Swap 操作)        if (stack_pop_type == 0) begin            Pop();                  // 取出奇数线程            Push(Type=CONV, ...);   // 存入偶数线程 (当前掩码)            Next_Mask = Pop_Mask;   // 切换执行权            Next_PC   = Pop_PC;        end        // 情况 B: 栈顶是 CONV (Merge 操作)        elsebegin            Pop();                  // 取出偶数线程            Next_Mask = Current_Mask | Pop_Mask; // 合并！            Next_PC   = PC + 4;     // 继续下一条指令        end    endend
```

### 嵌套支持 (Why Stack?)

为什么叫 Stack 而不是 Register？因为 if 里面可能还有 if。

如果 Block A 里面又有一个分支，硬件会再次 Push 一个 DIV 条目压在当前的 DIV 之上。

栈的特性保证了后进先出 (LIFO)，这完美对应了代码块的嵌套结构：必须先处理完最内层的 if/else，才能退回到外层继续处理。

::: related-title
相关推荐
:::

::: related-links
- [我已经一年多没写过一行代码了，算是 vibe coding 老手了](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897135&idx=1&sn=1ec81e600f5000e6fce34f76c8268be0&scene=21#wechat_redirect)
- [逼近 Opus 模型能力极限！烧掉2万刀，16个Claude暴力“肝”出10万行编译器，竟真能跑Linux内核！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897026&idx=1&sn=f1d236dd579687a588596631ac493b85&scene=21#wechat_redirect)
- [微基准视角下 NVIDIA Blackwell 架构深度解析：FP4/FP6赋能LLM 2.5倍推理，36.3TFLOPS FP64重塑科学计算](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896057&idx=1&sn=e3b882b56e8516ff1163d4cc300a25dd&scene=21#wechat_redirect)

:::
