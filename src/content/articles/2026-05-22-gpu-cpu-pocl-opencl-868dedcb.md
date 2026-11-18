关键词：**PoCL**、OpenCL、**LLVM Pass**、工作项循环、:mdnice-bold-italic[并行区域]

> OpenCL 的灵魂是 SPMD：成千上万个工作项（work-item）在同一段内核里齐头并进，遇到 `barrier()` 就整齐对齐。

这套模型在 GPU 上由硬件原生承载，可一旦目标变成 x86_64、ARM64、RISC-V 这类只会一条线程跑到底的 CPU，问题就来了——**你怎么让一颗 CPU 核同时假装自己是 256 个工作项，并且在屏障处"全员到齐"？**

![文章插图](/articles/2026-05-22-gpu-cpu-pocl-opencl-868dedcb/images/01.webp)

- 便携式计算语言（PoCL）
- PoCL是符合OpenCL 3.0标准的实现版本，可适配中央处理器与零级图形处理器目标设备，也能够便捷适配各类新型设备。
- 文档：https://portablecl.org/docs/html/
- **代码：https://github.com/pocl/pocl**

相关推荐

- [从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)
- [NVlabs/cuda-oxide：把 Rust 的类型系统接入 CUDA SIMT 世界的实验性编译器](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900944&idx=1&sn=471671b6081812dd4bae00a8dea74c32&scene=21#wechat_redirect)
- [剖析RISC-V不同平台性能：PMU剖析与基于LLVM硬件无关Roofline，分析SiFive、平头哥C910和进迭时空性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891030&idx=1&sn=54a3b4ba00d5f77b2dbffbf7b8355e72&scene=21#wechat_redirect)

> 这正是 pocl/pocl[1] 必须解决的核心难题。

作为目前唯一在 CPU[2] 与 Level Zero GPU[3] 上都通过 Khronos 官方一致性认证的开源 OpenCL 3.0 实现，**PoCL 给出的答案是：在 LLVM IR 这一层，把工作项的"并行"通过编译器 Pass 折叠成"循环"，把屏障转化成循环的边界**。

::: block-1
本文将穿过 60 多万行代码，钻进 `lib/llvmopencl/` 这个 PoCL 真正的"心脏"，看它如何把一份 GPU 思维写下的内核，变成 CPU 也能高效跑起来的机器码。
:::

## 本文目录

- 快速上手与使用
- 一、架构总览与设计哲学
  - 1.1 三层结构：Host Runtime / Kernel Compiler / Device Driver
  - 1.2 为什么需要专门的 Kernel Compiler
- 二、Kernel Compiler：让屏障变成循环边界
  - 2.1 核心抽象：`Barrier` 与 `ParallelRegion`
  - 2.2 屏障规范化：`CanonicalizeBarriers` 与 `ImplicitLoopBarriers`
  - 2.3 选择策略：WorkitemHandlerChooser
  - 2.4 主角登场：WorkitemLoops 如何包出循环
  - 2.5 跨区域变量的"上下文数组"
  - 2.6 PHIsToAllocas、IsolateRegions 等辅助工
  - 2.7 终点：Workgroup 函数包装
- 三、Host Runtime：从 API 到设备命令
  - 3.1 ICD 与 Platform Dispatch
  - 3.2 命令队列与事件机制
  - 3.3 程序构建：从源码/SPIR-V 到 LLVM bitcode
- 四、Device Driver：统一接口下的多后端
  - 4.1 `pocl_devices_ops` 函数表
  - 4.2 CPU 驱动：把 Workgroup 当函数指针调用
  - 4.3 GPU 驱动：编译期目标切换
- 五、几个值得回味的工程细节
  - 5.1 不变量驱动的 Pass 设计
  - 5.2 与 LLVM 上游生态的紧耦合
  - 5.3 一致性的代价
- 总结：PoCL 教会我们什么

![文章插图](/articles/2026-05-22-gpu-cpu-pocl-opencl-868dedcb/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手与使用

> 下面给出在 Ubuntu 上构建并运行 PoCL CPU 驱动的最精简流程，更细致的依赖、交叉编译、CUDA / Level Zero / Vulkan 后端构建说明请参考 官方安装指南[4] 与仓库 README.md[5]。

```
# 1. 安装依赖（LLVM/Clang 必须为最新发布版本）sudo apt install -y cmake ninja-build python3 \    libclang-dev libclang-cpp-dev libllvm-dev \    zlib1g-dev libtinfo-dev libhwloc-dev ocl-icd-libopencl1 ocl-icd-opencl-dev# 2. 拉取并构建（默认开启 CPU 驱动 + ICD）git clone https://github.com/pocl/pocl.git && cd poclcmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Releasecmake --build build -j# 3. 不安装也可直接使用：把构建目录里的 ICD 注册给 OpenCL Loaderexport OCL_ICD_VENDORS=$PWD/build/ocl-vendorsclinfo | head    # 应能看到 "Portable Computing Language" 平台
```

若仅想体验，发行版的二进制包同样可用（如 `apt install pocl-opencl-icd`、`brew install pocl`、`mamba install pocl-cuda`）。

CTS 一致性测试、SPIR-V 支持等可参考 构建文档[6] 中 `ENABLE_CONFORMANCE`、`ENABLE_SPIRV` 等选项。

## 一、架构总览与设计哲学

### 1.1 三层结构：Host Runtime / Kernel Compiler / Device Driver

> 打开仓库根目录，PoCL 的物理结构其实很克制，主要是三块：

- `lib/CL/`：**Host Runtime**。每一个 `clXxx.c` 文件对应一个 OpenCL API 入口，例如 `clEnqueueNDRangeKernel.c`[7]、`clCreateBuffer.c`[8]。:mdnice-bold-italic[它们负责参数检查、引用计数、命令打包，最终把工作丢给设备驱动]。共享数据结构集中在体量惊人的 `pocl_cl.h`[9] 里。
- `lib/llvmopencl/`：**Kernel Compiler**。这才是 PoCL 区别于 "调一调 LLVM" 类项目的灵魂。:mdnice-bold-italic[它由几十个自定义 LLVM Function/Module Pass 组成，把 Clang 产出的 OpenCL 内核 IR]，转写成"一个工作组就是一个普通函数"的形式，使其能以串行循环 + SIMD 在 CPU 上执行。
- `lib/CL/devices/`：**Device Drivers**。CPU、CUDA、Level Zero、Vulkan、Remote、OpenASIP 等后端在此目录下，通过统一的 `pocl_devices_ops` 函数表接入 Runtime。

> 整套流程可以画成一条"两端窄、中间宽"的链路：

```
clBuildProgram ─► Clang(前端) ─► LLVM IR ─► PoCL Pass 流水线 ─► 设备 ISA / SPIR-V ─► clEnqueueNDRangeKernel                                            (lib/llvmopencl)              (lib/CL/devices/*)
```

### 1.2 为什么需要专门的 Kernel Compiler

OpenCL 程序员写 `kernel void k(...) { ... barrier(CLK_LOCAL_MEM_FENCE); ... }` 时，假设有 *N = local_size_x × y × z* 个工作项同时执行，并在 `barrier()` 处对齐。

- GPU 通过 warp/wave 调度天然满足这个语义；
- 但 CPU 下若朴素地为每个工作项启一个线程，N 动辄上千，调度开销将彻底碾压计算。

::: block-1
:mdnice-bold-italic[PoCL 的关键洞察是：屏障切分了内核的控制流——两个屏障之间的代码段对所有工作项都是"互不干涉的并行执行"，这恰恰可以折叠成一个 `for (lid=0; lid<local_size; ++lid)` 的串行/可向量化循环]。这个洞察由 Pekka Jääskeläinen 等人在 2015 年的论文中提出，PoCL 是其参考实现。
:::

下文我们就沿着 IR 进入 Pass 流水线一路向下。

## 二、Kernel Compiler：让屏障变成循环边界

### 2.1 核心抽象：`Barrier` 与 `ParallelRegion`

> PoCL 把 OpenCL 的 `barrier()` 调用先标准化成一个特殊的 LLVM `CallInst`——`pocl.barrier`。

它由 `Barrier`[10] 这个继承自 `llvm::CallInst` 的薄封装表达：

```
// 来源：lib/llvmopencl/Barrier.hclass Barrier :public llvm::CallInst {public:// 在 BB 起始/末尾插入屏障；若已存在则复用static Barrier *createAtStart(llvm::BasicBlock *BB) {    return create(BB->getFirstInsertionPt());  }static Barrier *createAtEnd(llvm::BasicBlock *BB) {    return create(BB->getTerminator()->getIterator());  }// 判定一个循环里是否包含屏障——非常关键的查询static bool isLoopWithBarrier(llvm::Loop &L) {    for (auto *BB : L.blocks())      for (auto &I : *BB)        if (llvm::isa<Barrier>(&I)) returntrue;    returnfalse;  }// 利用 LLVM RTTI 让 isa<Barrier>() 工作static bool classof(const llvm::CallInst *C) {    return C->getCalledFunction() &&           C->getCalledFunction()->getName() == BARRIER_FUNCTION_NAME;  }};
```

有了 `Barrier`，PoCL 就可以做一个图论操作：把控制流图（CFG）按照"路径上必经一个屏障"的关系切成若干 **ParallelRegion** ——这是 PoCL 编译器最核心的概念，定义在 `ParallelRegion.h`[11]。

::: block-1
一个 ParallelRegion 是一段在所有工作项中无屏障穿越的连通子图，其入口前、出口后都是屏障基本块。一旦切分出来，**对同一 ParallelRegion，所有工作项的执行就彼此独立** ——这就是后面把它套上工作项循环的合法性基础。
:::

### 2.2 屏障规范化：`CanonicalizeBarriers` 与 `ImplicitLoopBarriers`

> 为了让 ParallelRegion 切分干净利落，PoCL 先对 IR 做一系列"形整"。最重要的一组 Pass：

- `CanonicalizeBarriers`[12]：保证每个屏障独占一个基本块，且基本块只含屏障一条指令。这把屏障变成纯粹的"图边界"。
- `ImplicitLoopBarriers`[13]：若一个循环内部含屏障，按 OpenCL 语义所有工作项必须在循环每次迭代都同步。该 Pass 在循环前后插入隐式屏障，避免后续 ParallelRegion 切分错把循环当成可独立并行的区域。
- `ImplicitConditionalBarriers`[14]：处理"条件分支两侧路径不一致"的屏障，把它们提升到分支前/汇合点，避免发散导致的死锁。
- `LoopBarriers`[15]：保证带屏障的循环 header 与 latch 都各有独立屏障基本块。

这些"看起来很啰嗦"的预处理 Pass 共同维护一个不变量：**所有发散的屏障必须等价地存在于所有工作项的执行路径上** ——否则 SPMD 语义无法折叠为循环。

### 2.3 选择策略：WorkitemHandlerChooser

> 切完 ParallelRegion 后，下一个问题是：怎么把"在所有工作项上重复执行 ParallelRegion"实现出来？

PoCL 在 `WorkitemHandlerChooser.h`[16] 里定义了两条道路：

```
// 来源：lib/llvmopencl/WorkitemHandlerChooser.henum class WorkitemHandlerType { LOOPS, CBS, INVALID };class WorkitemHandlerChooser    : public llvm::AnalysisInfoMixin<WorkitemHandlerChooser> {public:  using Result = WorkitemHandlerResult;     // 仅承载枚举  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);};WorkitemHandlerType ChooseWorkitemHandler(llvm::Function &F);
```

- **LOOPS**：以 `WorkitemLoops`[17] Pass 在每个 ParallelRegion 外套上 1 ～ 3 维 `for` 循环（对应 X/Y/Z 维度的 local_id），是默认与最成熟的方案。
- **CBS** （Continuation-Based Synchronization）：由 `SubCFGFormation`[18] 实现，用 sub-CFG 复制 + 状态机切换，对带不规则控制流的内核更友好，便于 LLVM 的循环向量化器（LoopVectorize）发挥。

`Chooser` 作为 LLVM Analysis 不修改 IR，只是把决策结果挂在分析管理器上，供后续 Pass 取用。

### 2.4 主角登场：WorkitemLoops 如何包出循环

> `WorkitemLoops`[19] 的实现类 `WorkitemLoopsImpl` 持有 LLVM 的 DominatorTree、LoopInfo、PostDominatorTree 与 PoCL 自家的 `VariableUniformityAnalysis`。

它的核心动作是：对每个 ParallelRegion，调用 `createLoopAround` 在 X/Y/Z 维度上各包一层循环。

```
// 来源：lib/llvmopencl/WorkitemLoops.ccclass WorkitemLoopsImpl :public pocl::WorkitemHandler {// 关键状态  llvm::DominatorTree &DT;  llvm::LoopInfo &LI;  llvm::PostDominatorTree &PDT;  VariableUniformityAnalysisResult &VUA;          // 哪些变量对所有 WI 同值  ParallelRegion::ParallelRegionVector OriginalParallelRegions; // 切分结果  StrInstructionMap ContextArrays;                // 跨区域变量的"上下文数组"std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;  llvm::Value *LocalIdXFirstVar;                  // 第一次迭代起点（用于 0,0,0 已 peeled）// 围绕 Region 在 Dim 维度上构造 [EntryBB, ExitBB] 之间的 for 循环std::pair<llvm::BasicBlock *, llvm::BasicBlock *>  createLoopAround(ParallelRegion &Region,                   llvm::BasicBlock *EntryBB,                   llvm::BasicBlock *ExitBB,                   bool PeeledFirst, int Dim,                   bool AddIncBlock = true,                   llvm::Value *DynamicLocalSize = nullptr);};
```

`createLoopAround` 内部生成的 IR 模板（节选其在源文件中的英文注释翻译）大致如下：

```
; 来源：lib/llvmopencl/WorkitemLoops.cc 中 createLoopAround 注释模板for.init:  store i32 0, i32* %_local_id_x        ; PeeledFirst=false 时常规起步  br label %for.condfor.cond:  %lid = load i32, i32* %_local_id_x  %cmp = icmp ult i32 %lid, %local_size_x  br i1 %cmp, label %region.entry, label %for.endregion.entry:  ... ; ParallelRegion 原本的代码  br label %for.incfor.inc:  %next = add i32 %lid, 1  store i32 %next, i32* %_local_id_x  br label %for.condfor.end:                                ; 即外层 ExitBB / 屏障所在 BB
```

这段循环在 Y、Z 维度上再套两层，就把一个工作组的全部 N = X·Y·Z 次工作项执行串成了循环。:mdnice-bold-italic[LLVM 在拿到这种"标准 induction variable 循环"后，可顺势进行 LoopVectorize、SLP 向量化、Loop Unroll 等通用优化——这正是 PoCL 让 OpenCL 内核在 CPU 上跑出像样性能的关键。]

### 2.5 跨区域变量的"上下文数组"

> 光把循环包好还不够。考虑一段简化代码：

```
int x = compute();barrier(CLK_LOCAL_MEM_FENCE);use(x);
```

`x` 在屏障前定义、屏障后使用——这意味着：屏障切出了两个 ParallelRegion，而**不同工作项的 `x` 必须各自保留** 到第二个 Region 才能用。

PoCL 的解法是把这种"跨区域活变量"提升成一个长度等于 `local_size_x*y*z` 的 alloca 数组（即代码里的 `ContextArrays`）：在第一个 Region 的循环内按 local_id 索引存入，在第二个 Region 的循环内按 local_id 索引取出。

`WorkitemLoopsImpl::fixMultiRegionVariables` 就是干这件事的；它得益于上一节预先建立的支配/后支配树以及 `VariableUniformityAnalysis`——后者识别出"对所有工作项同值的变量"（如循环不变量、kernel 参数），从而避免对它们做不必要的上下文存储，节省栈与时间。

### 2.6 PHIsToAllocas、IsolateRegions 等辅助工

> 为了让上下文存储正确实现，PoCL 还有几个"看似冷门、实则不可或缺"的 Pass：

- `PHIsToAllocas`[20]：把 PHI 节点降级为内存读写，便于在循环外部统一以 alloca 数组存活变量。
- `IsolateRegions`[21]：在 Region 边界插入空基本块，避免不同 Region 的边相互干扰。
- `AllocasToEntry`[22]：把所有 alloca 集中到入口块，方便后续 mem2reg。
- `BarrierTailReplication`[23]：处理屏障后的"尾部"代码在多个前驱中的复制，确保屏障的多入口语义被正确建模。

这些 Pass 共同把 LLVM IR 揉成一种"对工作项循环友好"的形态。

### 2.7 终点：Workgroup 函数包装

> 经过上述变换，每个 OpenCL kernel `K` 实际被编译成一个等价的 C 函数 `_pocl_kernel_K_workgroup`，签名形如：

```
// 来源：lib/llvmopencl/Workgroup.h（概念签名）void _pocl_kernel_K_workgroup(uint8_t **args,                              void *pocl_context,    // group_id, local_size, ...                              uint32_t group_x,                              uint32_t group_y,                              uint32_t group_z);
```

`Workgroup.cc`[24] 负责生成这层包装：把 OpenCL 的 kernel 参数从一个统一的指针数组里"拆包"，再调用包了工作项循环的内核体。设备驱动只需在 host 侧串起来 `for (group_x...) for (group_y...) for (group_z...) workgroup(...)`，剩下的工作项级并行已经在编译期被折叠完毕。

## 三、Host Runtime：从 API 到设备命令

### 3.1 ICD 与 Platform Dispatch

> OpenCL 应用通常通过 ICD Loader 调用厂商实现。

PoCL 在 `clIcdGetFunctionAddressForPlatformKHR.c`[25]、`clIcdGetPlatformIDsKHR.c`[26] 中实现 ICD 入口；`pocl_intfn.h`[27] 把所有 `clXxx` 入口收拢为可被 dispatch 表索引的内部函数指针。

### 3.2 命令队列与事件机制

> OpenCL 是异步语义：`clEnqueueXxx` 把命令塞进队列，并通过 `cl_event` 串接依赖。

PoCL 在 `pocl_util.c`[28]（73 KB，几乎是 Runtime 的"瑞士军刀"）和 `pocl_mem_management.c`[29] 中实现：

- `pocl_create_event`、`pocl_create_command`：构造命令节点与事件；
- 事件状态机：`SUBMITTED → READY → RUNNING → COMPLETE`，跨设备时还要处理 migration；
- 调度时把命令入队到设备驱动的 `push_command` 钩子。

`clEnqueueNDRangeKernel.c` 仅做参数检查并转交给 `pocl_ndrange_kernel.c`，后者完成 work-group 切分、work_dim 校正、SVM/USM 指针登记，最后构造一条 `CL_COMMAND_NDRANGE_KERNEL` 命令送入设备队列。

### 3.3 程序构建：从源码/SPIR-V 到 LLVM bitcode

`clBuildProgram` 看似只是 4 行包装，真正的重活在 `pocl_build.c`[30] 与 `pocl_llvm_build.cc`[31]（41 KB）。

后者直接调用 Clang 的 C++ API 完成 OpenCL → LLVM IR 的前端编译，再用 `pocl_llvm_spirv.cc`[32] 处理 SPIR-V 输入输出。**生成的 bitcode 会经 `pocl_cache.c`[33] 进行内容哈希缓存**，避免重复构建——这也是 PoCL 在大型应用首次启动后能"预热"出明显加速的原因。

## 四、Device Driver：统一接口下的多后端

### 4.1 `pocl_devices_ops` 函数表

> 每个驱动通过实现一组同构的回调挂入 Runtime：内存分配、命令调度、内核运行、镜像访问等。

- CPU 驱动在 `lib/CL/devices/pthread/`、`lib/CL/devices/basic/` 等子目录；
- GPU 后端如 `lib/CL/devices/level0/`、`lib/CL/devices/cuda/`。

这种"瘦驱动 + 厚 Runtime"的设计让 OpenCL 一致性测试（CTS）只需在 Runtime 层修一次 bug，所有驱动自动受益。

### 4.2 CPU 驱动：把 Workgroup 当函数指针调用

> CPU 驱动对每个 NDRange 命令在 host 层做三层 group 循环，每次直接 `workgroup_fn(args, &ctx, gx, gy, gz)`。

由于 `workgroup_fn` 内部已经是工作项循环 + LLVM 向量化后的本地代码，**整体性能在标准 BLAS、卷积、reduce 等 kernel 上能逼近手写 OpenMP 实现**。

CPU 驱动还借助 hwloc[34] 自动识别拓扑，把 group 调度到合适的核上，并通过 `pocl_threads.c`[35] 的轻量线程池减少调度抖动。

### 4.3 GPU 驱动：编译期目标切换

对 CUDA 和 Level Zero，`lib/llvmopencl/` 中的同一套 Pass 流水线照样工作，只是在 `Workgroup.cc`[36] 与 SPIR-V/PTX 生成阶段切换 LLVM target——**这种"前端 + 中端通用、后端按设备分流"的策略，正是 PoCL 能用一套代码覆盖如此多硬件的根本原因。**

## 五、几个值得回味的工程细节

### 5.1 不变量驱动的 Pass 设计

回头梳理 `lib/llvmopencl/` 下的 Pass，我们能看到一种朴素却高效的分层：先用预处理 Pass 维护语义不变量（屏障在每个工作项上必经、循环含屏障必同步），再让真正改写代码的 Pass（WorkitemLoops / SubCFGFormation）在干净的 IR 上工作。它的好处是单 Pass 复杂度可控、可独立测试——这对一个 60+ 万行、跨多 LLVM 大版本（README 同时支持 LLVM 18~22）的项目至关重要。

### 5.2 与 LLVM 上游生态的紧耦合

PoCL 大量直接 `#include <llvm/...>`，紧贴 LLVM 内部 API。这是双刃剑：得以使用最新优化（如 `MinLegalVecSize`[37] 调向量化阈值、`SanitizeUBofDivRem`[38] 兜底未定义行为），但每年也要适配 LLVM 大版本变更（README 中 CUDA/LLVM 21 的 Clang Bug 兼容说明就是一例）。

### 5.3 一致性的代价

README 提供的 CTS 通过率表格告诉我们：CPU/x86_64 驱动 100% 通过 OpenCL 3.0 CTS，CPU/RISCV 也>99%。这背后是大量 corner case 修复，例如：图像采样器的初始化（`HandleSamplerInitialization.cc`[39]）、半精度 emulation（`pocl_cl_half_util.c`[40]）、SVM 偏移修正（`SVMOffset.cc`[41]）。这些"边角料"，恰恰是 PoCL 能挂上 Khronos Conformant Products 名单的关键。

## 总结：PoCL 教会我们什么

> 如果只能把 PoCL 抽象成一句话，那便是：**用 LLVM IR 这一通用基板，把 OpenCL 的 SPMD 语义转译为 CPU 可高效执行的循环**。在这背后，是一个清晰、几乎可以作为编译课程教材的 Pass 序列：

```
1. 标准化屏障       ↓       2. 切分 ParallelRegion       ↓3. 选择处理策略（LOOPS/CBS）       ↓4. 套上工作项循环       ↓5. 跨区域变量上下文化       ↓6. 包装为 workgroup 函数       ↓7. 多后端生成代码
```

而 Host Runtime（`lib/CL/`）则用一套统一的 ICD/事件/命令缓冲机制，把 OpenCL API 与 Device Driver 解耦，让 CPU、CUDA、Level Zero、Vulkan、Remote 等差异巨大的后端能在同一份测试套件下被验证为"符合标准"。

- 对编译器从业者，PoCL 是研究 SPMD-to-SIMD 转换、屏障语义降级、LLVM Pass 工程化的活教材；
- 对系统软件作者，它演示了"瘦驱动 + 厚 Runtime"如何在保证可移植性的同时不牺牲一致性；
- 对硬件加速器初创团队，它甚至是一条捷径——只要写好一个驱动接入 `pocl_devices_ops`，就能立刻拥有一个完整的 OpenCL 3.0 实现。

> 这或许才是 PoCL 在异构计算碎片化年代依然不可替代的根本意义。

参考资料

[1] pocl/pocl: *https://github.com/pocl/pocl*

[2] CPU: *https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_450*

[3] Level Zero GPU: *https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_453*

[4] 官方安装指南: *http://portablecl.org/docs/html/install.html*

[5] README.md: *https://github.com/pocl/pocl/blob/main/README.md*

[6] 构建文档: *http://portablecl.org/docs/html/install.html*

[7] `clEnqueueNDRangeKernel.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/clEnqueueNDRangeKernel.c*

[8] `clCreateBuffer.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/clCreateBuffer.c*

[9] `pocl_cl.h`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_cl.h*

[10] `Barrier`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/Barrier.h*

[11] `ParallelRegion.h`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/ParallelRegion.h*

[12] `CanonicalizeBarriers`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/CanonicalizeBarriers.cc*

[13] `ImplicitLoopBarriers`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/ImplicitLoopBarriers.cc*

[14] `ImplicitConditionalBarriers`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/ImplicitConditionalBarriers.cc*

[15] `LoopBarriers`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/LoopBarriers.cc*

[16] `WorkitemHandlerChooser.h`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/WorkitemHandlerChooser.h*

[17] `WorkitemLoops`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/WorkitemLoops.cc*

[18] `SubCFGFormation`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/SubCFGFormation.cc*

[19] `WorkitemLoops`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/WorkitemLoops.cc*

[20] `PHIsToAllocas`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/PHIsToAllocas.cc*

[21] `IsolateRegions`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/IsolateRegions.cc*

[22] `AllocasToEntry`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/AllocasToEntry.cc*

[23] `BarrierTailReplication`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/BarrierTailReplication.cc*

[24] `Workgroup.cc`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/Workgroup.cc*

[25] `clIcdGetFunctionAddressForPlatformKHR.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/clIcdGetFunctionAddressForPlatformKHR.c*

[26] `clIcdGetPlatformIDsKHR.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/clIcdGetPlatformIDsKHR.c*

[27] `pocl_intfn.h`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_intfn.h*

[28] `pocl_util.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_util.c*

[29] `pocl_mem_management.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_mem_management.c*

[30] `pocl_build.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_build.c*

[31] `pocl_llvm_build.cc`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_llvm_build.cc*

[32] `pocl_llvm_spirv.cc`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_llvm_spirv.cc*

[33] `pocl_cache.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_cache.c*

[34] hwloc: *https://www.open-mpi.org/projects/hwloc/*

[35] `pocl_threads.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_threads.c*

[36] `Workgroup.cc`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/Workgroup.cc*

[37] `MinLegalVecSize`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/MinLegalVecSize.cc*

[38] `SanitizeUBofDivRem`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/SanitizeUBofDivRem.cc*

[39] `HandleSamplerInitialization.cc`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/HandleSamplerInitialization.cc*

[40] `pocl_cl_half_util.c`: *https://github.com/pocl/pocl/blob/main/lib/CL/pocl_cl_half_util.c*

[41] `SVMOffset.cc`: *https://github.com/pocl/pocl/blob/main/lib/llvmopencl/SVMOffset.cc*
