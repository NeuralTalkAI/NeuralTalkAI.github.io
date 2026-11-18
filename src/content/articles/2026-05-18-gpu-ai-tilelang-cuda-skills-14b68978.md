关键词：:mdnice-bold-italic[AI 编码智能体]、**TileLang**、CUDA 内核优化、**Agent Skills**、GPU 算子开发

> GPU 内核开发是深度学习基础设施中最晦涩也最关键的一环。一个高性能的 GEMM 算子，需要开发者同时理解硬件内存层级、Tensor Core 指令映射、流水线调度和数值精度——这些知识散落在文档、论坛帖子和工程师的脑海中，难以系统传递。

而如今，AI 编码智能体（如 Claude Code 中的 Agent）正在接管越来越多的编程任务，**但面对 GPU 编程，它们往往缺乏结构化的领域知识。**

![文章插图](/articles/2026-05-18-gpu-ai-tilelang-cuda-skills-14b68978/images/01.webp)

- **Skills for writing tilelang and debugging with CUDA toolkits.**
- https://github.com/sablin39/tilelang-cuda-skills
- 4000 字，阅读 20 分钟，**播客 23 分钟**

相关推荐

- [暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)
- [暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)
- [暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)

> `tilelang-cuda-skills` 项目正是为解决这一断层而生：**它将 GPU 算子开发的完整工作流——从编写第一行 TileLang 代码，到调试、性能剖析、自动调优，再到前反向传播测试——编码为六份结构化的"Agent Skill"文档**，让 AI 智能体具备一位资深 GPU 工程师的决策框架。这不是一个普通的教程集合，而是一次将专家经验系统化注入 AI 的知识工程实验。

![文章插图](/articles/2026-05-18-gpu-ai-tilelang-cuda-skills-14b68978/images/02.webp)

::: image-caption
TileLang Skills：由 Claude Opus 4.6 借助 SKILL 创建插件在 Claude Code 中编写的 TileLang 相关技能。该内容依据 TileLang 0.1.9 版本的文档与示例开发完成，并已搭载 CUDA 13.1 和 PyTorch 2.11，在 RTX PRO 6000 BlackWell 架构显卡（算力架构 sm_120）上完成实测验证。这张表格系统梳理了TileLang GPU编程的五大核心 Skill，清晰定义了每项 Skill 的能力边界与关键技术点：从从零编写适配通用矩阵乘（GEMM）、元素级运算等模式的 TileLang 内核，到通过打印调试、自动差分调试（AutoDD）等工具诊断数值漂移、竞态条件等错误；再到利用基准测试与 NVIDIA Nsight Compute（NCU）分析流水线利用率、 warp 阻塞等性能瓶颈，以及通过分块大小调优、流水线优化等手段提升内核效率，最后覆盖前向/反向传播的全流程验证。这套技能体系完整覆盖了 TileLang 开发的全生命周期，为高效开发、调试、优化 GPU 算子提供了标准化的能力框架。
:::

![文章插图](/articles/2026-05-18-gpu-ai-tilelang-cuda-skills-14b68978/images/03.webp)

::: image-caption
CUDA Skills：适用于调试、分析性能以及优化GPU内核的通用CUDA开发技术，不受任何特定框架限制。该内容源自technillogue/ptx-isa-markdown项目，内含整理爬取的PTX指令集架构9.1版本、CUDA运行时应用程序接口13.1版本以及CUDA驱动应用程序接口13.1版本相关文档（共计640余个Markdown文件），可支持基于文本检索工具快速查阅相关内容。
:::

## 本文目录

- 快速上手
- 一、项目定位与架构哲学
  - 1.1 为什么 AI 需要"技能"才能写好 GPU 代码
  - 1.2 六技能闭环架构
- 二、编写技能：从模板到内核的结构化生成
  - 2.1 TileLang 内核的"两层嵌套"骨架
  - 2.2 三层内存模型与 API 模式速查
- 三、调试技能：一棵精密的故障诊断决策树
  - 3.1 四类故障分类法
  - 3.2 "结果错误"的快速三连检
- 四、性能分析与优化：从测量到决策的闭环
  - 4.1 性能分析的"先验证后测量"铁律
  - 4.2 优化技能的"单变量控制法"
- 五、前反向传播测试：可微分算子的完整工程范式
  - 5.1 五阶段架构
  - 5.2 为什么不能用 gradcheck
- 六、通用 CUDA 技能：框架无关的底层武器库
  - 6.1 核心哲学四句箴言
  - 6.2 本地化文档搜索体系
- 总结与启示

![文章插图](/articles/2026-05-18-gpu-ai-tilelang-cuda-skills-14b68978/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> 该项目是一组 Agent Skill 文件，专为 Claude Code 的`skill-creator`插件设计，无需安装即可使用。核心用法是将技能文件配置到你的 AI 编码环境中：

```
# 克隆仓库git clone https://github.com/sablin39/tilelang-cuda-skills.git# 技能文件位于 skills/ 目录下，按需引用即可# TileLang 技能：skills/tilelang/ 下五个子目录# CUDA 通用技能：skills/cuda_skill/SKILL.md
```

若需实际运行 TileLang 内核，需确保环境中已安装 TileLang（基于`v0.1.9`验证）：

```
pip install tilelangpython -c "import tilelang; print(tilelang.__version__)"
```

更多 TileLang 安装与 GPU 兼容性细节，参考 TileLang 官方仓库[1]。该技能体系在 RTX PRO 6000 Blackwell GPU（sm_120）上经过验证，具体硬件适配情况参考项目 README[2]。

## 一、项目定位与架构哲学

### 1.1 为什么 AI 需要"技能"才能写好 GPU 代码

> 大语言模型在通用编程任务上已表现出色，但 GPU 内核开发有三个独特难点：

- **硬件感知性**：必须理解 shared memory、register file、Tensor Core 等物理约束
- **调试不透明性**：错误往往表现为"输出全是 NaN"而非明确的报错信息
- **优化反直觉性**：直觉上"更好"的代码可能因 bank conflict 而更慢）

::: block-1
`tilelang-cuda-skills`的设计哲学是：**不教模型"理解"GPU，而是给它一套决策树**。每个 Skill 文件本质上是一棵结构化的 if-else 知识树——当遇到 X 症状，执行 Y 动作。这种设计让 Agent 无需真正理解硬件原理，只需按图索骥就能做出正确决策。
:::

### 1.2 六技能闭环架构

> 项目包含六份核心 Skill，形成一条完整的开发流水线：

```
writing → debugging → profiling → optimizing                                      ↓                              testing-fwd-bwd (用于可微分算子)
```

外加一份独立于框架的通用 CUDA 技能（`cuda-programming`），覆盖`compute-sanitizer`、`cuda-gdb`、`ncu/nsys`等底层工具链。六份文档总计约 870KB，堪称一部压缩的 GPU 开发百科全书。

## 二、编写技能：从模板到内核的结构化生成

### 2.1 TileLang 内核的"两层嵌套"骨架

> 编写技能（`writing-tilelang-kernels`）的核心贡献是定义了一套**标准化的内核骨架**。每个 TileLang 内核都遵循"外层函数定义编译期参数、内层函数定义 GPU 逻辑"的两层结构：

```
# 来源：skills/tilelang/writing-tilelang-kernels/SKILL.md@tilelang.jit(out_idx=[-1])  # 最后一个参数是输出def my_kernel(M, N, K, block_M, block_N, block_K,              dtype=T.float16, accum_dtype=T.float32):    # 外层：编译期参数（问题规模、分块大小、数据类型）    @T.prim_func    def kernel(        A: T.Tensor((M, K), dtype),        B: T.Tensor((K, N), dtype),        C: T.Tensor((M, N), dtype),    ):        # 内层：实际GPU逻辑        with T.Kernel(T.ceildiv(N, block_N),                      T.ceildiv(M, block_M), threads=128) as (bx, by):            # 1. 分配片上内存            A_shared = T.alloc_shared((block_M, block_K), dtype)            B_shared = T.alloc_shared((block_K, block_N), dtype)            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)            # 2. 初始化累加器            T.clear(C_local)            # 3. 流水线计算            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):                T.copy(A[by * block_M, ko * block_K], A_shared)                T.copy(B[ko * block_K, bx * block_N], B_shared)                T.gemm(A_shared, B_shared, C_local)            # 4. 写回全局内存            T.copy(C_local, C[by * block_M, bx * block_N])    return kernel
```

这段代码是整个技能体系的"Hello World"，但它浓缩了 GPU 编程的四个核心步骤：**内存分配 → 初始化 → 流水线计算 → 写回**。Agent 只需识别目标运算类型（GEMM、逐元素、规约等），从模板库中选择匹配的骨架，填入具体的维度和计算逻辑即可。

### 2.2 三层内存模型与 API 模式速查

> 技能文件明确定义了 TileLang 的三级内存语义，这是 Agent 决策"数据放在哪里"的核心依据：

| 内存层级 | API | 可见性 | 典型用途 |
| --- | --- | --- | --- |
| Shared Memory | `T.alloc_shared` | 线程块内所有线程 | 暂存全局内存的分块数据 |
| Fragment（寄存器） | `T.alloc_fragment` | 由编译器自动分配给各线程 | 累加器、中间结果 |
| Local | `T.alloc_local` | 单个线程私有 | 标量累加器 |

配合`T.copy`（数据搬运）、`T.gemm`（Tensor Core 矩阵乘）、`T.Pipelined`（软件流水线）等原语，Agent 可以像搭积木一样组装出完整的 GPU 内核。

## 三、调试技能：一棵精密的故障诊断决策树

### 3.1 四类故障分类法

> 调试技能（`debugging-tilelang-programs`）最精妙的设计是将所有可能的故障分为四个互斥类别，每类对应完全不同的排查路径：

```
A: 编译/代码生成失败 → Python traceback、TVM错误、nvcc错误B: 运行时崩溃       → CUDA错误、非法内存访问、TMA描述符错误C: 结果错误         → 输出全是垃圾、全零、NaND: 数值漂移         → 结果接近但超出容差
```

这种分类法的价值在于：**Agent 不需要理解错误的根因，只需根据症状定位到类别，然后按该类别的检查清单逐项排查**。

### 3.2 "结果错误"的快速三连检

> 类别 C（结果错误）是最常见也最棘手的情况。技能文件提供了一个"先做这三件事"的快速诊断方案：

```
# 来源：skills/tilelang/debugging-tilelang-programs/SKILL.md# 检查1：是否遗漏了 T.clear？C_local = T.alloc_fragment((block_M, block_N), accum_dtype)T.clear(C_local)  # ← 漏掉这一行 = 累加器含垃圾值# 检查2：是否遗漏了写回？T.copy(C_local, C[by * block_M, bx * block_N])  # ← 漏掉 = 输出未更新# 检查3：T.gemm 的形状是否正确？# A 必须是 (BM,BK), B 必须是 (BK,BN), C 必须是 (BM,BN)
```

这三个检查覆盖了实践中**排名前三的 TileLang Bug 类型**。技能文件还贴心地提供了`T.print`调试输出的解读方法——`index=k`对应二维缓冲区的`buffer[k // N, k % N]`坐标，以及 AutoDD 自动最小化复现工具的使用方法。

## 四、性能分析与优化：从测量到决策的闭环

### 4.1 性能分析的"先验证后测量"铁律

> 性能分析技能（`profiling-tilelang-programs`）开篇就立下规矩：**永远先验证正确性，再谈性能**。这不是空话——技能文件将其编码为一个强制性的前置步骤：

```
# 来源：skills/tilelang/profiling-tilelang-programs/SKILL.mdprofiler = kernel.get_profiler(    tensor_supply_type=tilelang.TensorSupplyType.Normal)profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)  # 先过正确性tl_latency = profiler.do_bench(warmup=25, rep=100, return_mode="median")
```

技能文件还提供了完整的性能指标计算公式（TFLOPS = `2*M*N*K / latency_ms * 1e-9`）和基于`ncu`的瓶颈诊断流程：比较 SM 吞吐率与 DRAM 吞吐率，判断内核是计算受限还是访存受限，再进一步检查 Tensor Core 管线与 FMA 管线的利用率。

### 4.2 优化技能的"单变量控制法"

> 优化技能（`optimizing-tilelang-programs`）的方法论极为务实——**每次只改一个参数**：

```
| 配置                                           | 延迟(ms) | TFLOPS | 备注          || baseline: bM=128 bN=128 bK=32 stages=2 thr=128 | ...      | ...    |               || 改 bK=64                                        | ...      | ...    | +X% ← 保留    || 改 thr=256                                       | ...      | ...    | +Y% 对比基线   || 改 stages=3                                      | ...      | ...    | -Z% ← 回退    |
```

当手动调参过于繁琐时，技能文件推荐使用 TileLang 的`AutoTuner` API 进行自动搜索：

```
# 来源：skills/tilelang/optimizing-tilelang-programs/SKILL.mdfrom tilelang.autotuner import AutoTunerconfigs = [    {"block_M": 64,  "block_N": 64,  "block_K": 32, "num_stages": 2, "threads": 128},    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},]autotuner = (    AutoTuner.from_kernel(my_kernel, configs)    .set_compile_args(out_idx=[-1], target="auto")    .set_profile_args(supply_type=tl.TensorSupplyType.Integer,                      ref_prog=ref_program, warmup=3, rep=20))result = autotuner.run()# result.config 即为最优配置, result.kernel 为编译好的最优内核
```

注意技能文件特别警告了一个实际的 Bug：**不要使用`@tilelang.autotune`装饰器搭配`ref_prog`参数**，因为存在缓存序列化问题，应始终使用编程式 API。这类"踩坑经验"正是 Agent Skill 相比普通文档的核心价值。

## 五、前反向传播测试：可微分算子的完整工程范式

### 5.1 五阶段架构

> 前反向传播测试技能（`testing-fwd-bwd-kernels`）定义了可微分 GPU 算子的标准架构，以 Flash Attention 为例：

```
前向内核(output + lse) → 反向预处理(Delta) → 主反向内核(dQ/dK/dV) → 反向后处理(float32→fp16) → autograd.Function封装
```

其中，最关键的设计决策是**反向内核不使用`out_idx`**——梯度缓冲区由调用方预分配并作为可变参数传入，而非由内核内部分配返回。原因在于：

```
# 来源：skills/tilelang/testing-fwd-bwd-kernels/SKILL.md# dQ 必须用 torch.zeros 预分配（atomicAdd需要零初始化）dq = torch.zeros([BATCH, H, N_CTX, D_HEAD],                 dtype=torch.float32, device=q.device)  # 必须float32！dk = torch.empty_like(k)   # 直接写入，可用emptydv = torch.empty_like(v)
```

### 5.2 为什么不能用 gradcheck

> 技能文件中一个特别有价值的结论是：**不要使用`torch.autograd.gradcheck`测试 TileLang 内核的梯度**。

经过系统实验验证，在 fp16/bf16/fp32 及混合精度下，gradcheck 对交换梯度、2 倍缩放梯度、梯度置零等三类 Bug 的检出率为零。原因是`fast_mode=True`只检查随机 Jacobian 投影，而低精度前向传播所需的宽松容差掩盖了错误。正确做法是使用`compare_backward`模式——分别通过自定义路径和 PyTorch 参考路径运行前向+反向，然后逐个比较梯度：

```
# 来源：skills/tilelang/testing-fwd-bwd-kernels/SKILL.md# 关键：运行完自定义backward后，必须clone梯度并清空.gradO.backward(dO, retain_graph=True)dQ, Q.grad = Q.grad.clone(), None  # clone后立即清空！# 再运行参考backward，否则梯度会叠加
```

技能文件还提供了余弦相似度诊断法：`cos_sim≈0`表示完全算错了矩阵（通常是转置问题），`cos_sim≈1`但幅值不对表示缩放因子有误，`cos_sim≈-1`表示符号翻转。

## 六、通用 CUDA 技能：框架无关的底层武器库

### 6.1 核心哲学四句箴言

> CUDA 技能（`cuda-programming`）开篇定义了四条核心原则，堪称 GPU 开发的"武德"：

1. **先测量再猜测** ——GPU 性能反直觉，profile 永远第一步
2. **小步隔离修改** ——CUDA Bug 会复合，一次只改一处
3. **printf 是最强工具** ——当调试器失灵时，device 代码里的 printf 揭示真相
4. **有时候，盯着 diff 看** ——最小化 working 和 broken 代码的差异，Bug 就在其中

### 6.2 本地化文档搜索体系

该技能附带了完整的离线文档体系：PTX ISA 9.1（405 个文件，2.3MB）、CUDA Runtime API 13.1（107 个文件）和 CUDA Driver API 13.1（128 个文件），配合 grep 搜索指南，使 Agent 在无网络环境下也能查阅指令级参考资料。这种"把手册直接喂给 Agent"的做法，避免了大模型在底层 API 细节上的幻觉问题。

## 总结与启示

> `tilelang-cuda-skills`项目的真正创新不在于其技术内容本身——这些知识对资深 GPU 工程师而言并不陌生——而在于它完成了一次**知识表示的范式转换**：将分散在工程师脑海中的隐性知识，转化为 AI Agent 可直接消费的结构化决策树。

每份 Skill 文件都遵循统一的模式：**症状表 → 分类决策 → 具体操作步骤 → 常见陷阱表 → 升级路径**。这种设计使得 Agent 无需具备对 GPU 硬件的"理解"，只需具备模式匹配和步骤执行的能力，就能完成从内核编写到性能调优的完整工作流。

::: block-1
从更宏观的视角看，这个项目代表了 AI 辅助编程的一个重要方向：**不是让模型变得更聪明，而是让知识变得更可被模型消费**。

当我们把专家经验编码为结构化的 Skill 文件时，我们:mdnice-bold-italic[实际上在构建一种新型的"知识中间件"——它连接了人类专家的领域智慧与 AI Agent 的执行能力]。这或许才是"AI 编程"真正的基础设施。
:::

参考资料

[1] TileLang 官方仓库: *https://github.com/tile-ai/tilelang*

[2] README: *https://github.com/sablin39/tilelang-cuda-skills/blob/main/README.md*
