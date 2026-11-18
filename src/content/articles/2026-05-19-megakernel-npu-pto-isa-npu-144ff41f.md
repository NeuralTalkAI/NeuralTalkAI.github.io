关键词：Gated DeltaNet、**Megakernel 融合**、:mdnice-bold-italic[PTO-ISA]、**昇腾 NPU**、:mdnice-bold-italic[线性注意力推理加速]

> 在大语言模型推理的赛道上，注意力机制的计算效率直接决定了系统吞吐量的天花板。Gated **DeltaNet（GDN）作为一种新兴的线性注意力变体，通过分块递推（chunk-wise recurrence）的方式巧妙绕开了标准 Attention 的二次复杂度瓶颈。**

::: block-1
**然而，"算法层面的线性"并不自动转化为"硬件层面的高效** "——当七个紧密耦合的计算阶段被拆分为七次独立的 kernel launch 时，:mdnice-bold-italic[Python 层的调度开销、核间同步延迟、以及反复的 Global Memory 读写，足以将理论上的加速比吞噬殆尽。]
:::

![文章插图](/articles/2026-05-19-megakernel-npu-pto-isa-npu-144ff41f/images/01.webp)

- **Megakernel for Gated DeltaNet, optimized by PTO-ISA**
- https://github.com/huawei-csl/megagdn-pto
- 7000 字，阅读 38 分钟，**播客 37 分钟**

相关推荐

- [“几乎不可编译”到15.9倍加速！AscendCraft：面向 NPU 的内核生成方法，DSL引导转译弥合NPU内核生成鸿沟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898824&idx=1&sn=6de0d761ed4fa4421805471e80f4d12c&scene=21#wechat_redirect)
- [鹏城联合华为昇腾、中山大学提出AscendKernelGen：NPU 算子闭环生成与评估，复杂L2 Kernel编译成功率95.5%，正确率64.3](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896566&idx=1&sn=d920e9f8feed38e4734d18942d83897a&scene=21#wechat_redirect)
- [SGEMM-cube：昇腾910A用FP16 Cube单元模拟 FP32 GEMM 实现 22 位尾数精度恢复，77%理论峰值！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890912&idx=1&sn=9bd135756aa33bd195fa42099bd6e840&scene=21#wechat_redirect)
- [Grid Sample比昇腾NPU厂商库快2.4倍！微信HPC团队硬件感知优化加速多尺度Deformable Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888739&idx=1&sn=888cdb64f175ad90da1a0215e4879a19&scene=21#wechat_redirect)

> 华为 CSL 团队的 MegaGDN-PTO 项目，直接使用 PTO-ISA——一套面向昇腾 AI Core 的底层指令集抽象——手写了一个将全部七个阶段融合在单次 kernel launch 中的 Megakernel，**在 Atlas 910B 芯片上实现了对 Triton-Ascend 基线 1.5~3 倍的性能碾压，同时保持模型精度完全无损。**

![文章插图](/articles/2026-05-19-megakernel-npu-pto-isa-npu-144ff41f/images/02.webp)

::: image-caption
这张柱状图展示了三种实现方式在相同配置（16条序列、8192 tokens长度）下的单阶段核延迟：原生PTO实现总耗时约24ms，各阶段耗时均衡；Triton（BT=64）实现耗时最高，达52ms，`chunk h`与`chunk o`阶段耗时占比最大；Triton（BT=128）实现总耗时约22ms，整体性能优于BT=64，但部分阶段（如`scaled dot KKT`、`solve tril`、`chunk o`）出现“Triton stage failed”的报错标记，说明部分算子实现存在兼容性问题。整体来看，原生PTO的实现最稳定，Triton优化在BT=128时虽有性能潜力，但算子稳定性不足，而BT=64的Triton实现则出现了明显的性能倒退。
:::

![文章插图](/articles/2026-05-19-megakernel-npu-pto-isa-npu-144ff41f/images/03.webp)

::: image-caption
这张双栏柱状图对比了PTO megakernel与Triton baseline在不同模型上的精度表现。左图Wikitext困惑度（越低越好）显示，两者在Qwen3.5-0.8B/9B、Qwen3.6-27B/35B系列模型上数值几乎完全重合，困惑度随模型规模增大而显著下降；右图MMLU准确率（越高越好）同样表现出高度一致性，PTO megakernel与Triton baseline在各模型上的准确率差值均在1%以内，且随模型规模提升而稳步上升。整体来看，PTO megakernel在大幅优化计算效率的同时，实现了与Triton baseline完全一致的模型精度，验证了其无精度损失的优化效果。
:::

![文章插图](/articles/2026-05-19-megakernel-npu-pto-isa-npu-144ff41f/images/04.webp)

::: image-caption
这张图表展示了PTO megakernel与Triton baseline在Qwen36-35B模型Prefill阶段的性能对比。在512到65536 tokens的提示长度范围内，PTO megakernel实现了1.1-1.25倍的加速比，峰值在4096 tokens处；对应的TTFT延迟上，PTO在各长度下均低于Triton，尤其在65536 tokens时延迟差距显著拉大；吞吐量方面，PTO的token生成速度全程领先，在16384 tokens时达到约16000 tokens/s的峰值，整体性能优势随提示长度增加而愈发明显。整体来看，PTO megakernel在Prefill阶段全面优于Triton baseline，尤其在长序列场景下表现出更强的性能优势。
:::

![文章插图](/articles/2026-05-19-megakernel-npu-pto-isa-npu-144ff41f/images/05.webp)

::: image-caption
这组三行子图对比了PTO megakernel与Triton baseline在Qwen36-35B模型Prefill阶段的性能差异。上半部分的加速比曲线显示，PTO实现了1.1-1.18倍的稳定加速优势；中间的TTFT延迟曲线表明，PTO的首包响应延迟全程低于Triton，尤其在32768 tokens时差距明显；下方的吞吐量曲线显示，PTO的token生成速度始终领先，在16384 tokens时达到约19000 tokens/s的峰值。整体来看，PTO megakernel在Prefill阶段全面优于Triton baseline，尤其在中长序列场景下，延迟更低、吞吐量更高，展现出稳定的性能优势。
:::

## 本文目录

- 快速上手
- 一、架构总览与设计哲学
  - 1.1 GDN 的七阶段计算流水线
  - 1.2 分阶段执行的性能瓶颈
  - 1.3 Megakernel 的解法
  - 1.4 项目代码组织
- 二、NPU 硬件编程模型：Vec、Cube 与显式流水线
  - 2.1 AI Core 内部架构
  - 2.2 内存层次与 Tile 抽象
  - 2.3 同步原语一览
- 三、编译系统：Bisheng JIT 的编译期特化
  - 3.1 编译期常量注入策略
  - 3.2 增量编译与缓存
  - 3.3 硬件自适应
- 四、核心内核逐级剖析
  - 4.1 chunk_cumsum：Vec-only 的前缀和
  - 4.2 scaled_dot_kkt：Cube 与 Vec 的双核协作
  - 4.3 wy_fast：WY 分解的 Vec+Cube 协作
  - 4.4 solve_tril：CubeCore 递推三角求逆
  - 4.5 Megakernel 融合入口：七阶段一体化
  - 4.6 异构屏障 SyncAllImpl 的三步握手
- 五、Python 接口层：从编译到调用的完整链路
  - 5.1 ctypes 零拷贝调用
  - 5.2 GQA（Grouped Query Attention）原生支持
  - 5.3 Variable-length sequence 支持
- 六、vLLM 集成：运行时 Monkey-Patch 架构
  - 6.1 非侵入式钩子注入
  - 6.2 为何使用 monkey-patch 而非 fork
- 七、性能分析与工程取舍
  - 7.1 加速的三层来源
  - 7.2 首次编译开销
  - 7.3 混合精度保障
  - 7.4 调试与可观测性
- 八、技术洞察与总结

![文章插图](/articles/2026-05-19-megakernel-npu-pto-isa-npu-144ff41f/images/06.webp)

## 快速上手

> 项目采用"插件式"设计，可独立测试内核性能，也可集成至 vLLM-Ascend 端到端推理。

```
# 克隆（含PTO-ISA子模块）git clone --recursive https://github.com/huawei-csl/megagdn-pto.gitcd megagdn-pto# 安装Python接口（需已有CANN + torch-npu环境）pip install -e '.[eval,plot]'# 精度验证python tests/test_single_kernels.py --H-list 16,32,48,64# 性能基准python benchmarks/kernel/bench_gdn_kernels.py \    --device npu:0 --n-seq 16 --l-seg 8192 --H-list 16,32,48,64
```

环境要求：昇腾 910B + CANN 8.5+ + torch-npu。推荐使用 vllm-ascend Docker 镜像[1] 作为开箱即用的环境。若仅测试 PTO 内核无需 vLLM，可使用更轻量的 CANN Docker 镜像[2]。端到端 vLLM 评测需先执行 `python vllm_patch/install_hook.py` 注入运行时钩子，更多模型权重准备与评测流程参考 `README.md`[3]。

## 一、架构总览与设计哲学

### 1.1 GDN 的七阶段计算流水线

> Gated DeltaNet 将输入序列按固定长度 C（默认 128 token）分块，在每个 chunk 内部执行如下七阶段计算：

```
cumsum → transpose → scaled_dot_kkt → solve_tril → wy_fast → chunk_h → chunk_o
```

每个阶段的数学含义如下：

- **cumsum**：对 gate logits 沿时间维做 chunk 内前缀和。下游可通过 `exp(g_sum[i] - g_sum[j])` 快速计算从 token j 到 token i 的累积门控衰减系数。
- **transpose**：将 `[T, H]` 的 BSND 布局转置为 `[H, T]` 的按 head 连续布局，使后续内核可按 head 做连续内存访问。
- **scaled_dot_kkt**：计算 `A = mask(K@K^T · gating_coeff)`，其中 gating 系数为 `exp(clamp(g[i]+log(β[i])-g[j], max=0))`，叠加下三角因果掩码。这是 chunk 内的"注意力矩阵"。
- **solve_tril**：对下三角矩阵 A 做递推求逆 `A^{-1}`，用于后续 WY 分解的"去耦合"。
- **wy_fast**：基于 `A^{-1}` 计算 WY 分解的两个分支：`U = (A·β_2d) @ V`，`W = (A·(exp(g)·β)_2d) @ K`。
- **chunk_h**：递推更新隐状态矩阵 S（形如 `[H, D, D]`），同时计算去干扰值 `v_new`。
- **chunk_o**：将 chunk 内 Q·K^T 注意力输出与跨 chunk 的状态贡献 Q·S 相加，得到最终输出 O。

### 1.2 分阶段执行的性能瓶颈

> 在分阶段模式下，每一步都需要：

1. Python 层通过 ctypes 调用 `lib.call_kernel()` → NPU 运行时调度 → AI Core 启动
2. 阶段输出写回 Global Memory（HBM）
3. 下一阶段从 HBM 重新读取

对于典型推理场景（128 token × 16 head），单阶段计算仅需数十微秒，但每次 kernel launch 本身就有几十到上百微秒的开销。七次 launch 累积的调度与 HBM 读写延迟，可能占总耗时的 30%~50%。

### 1.3 Megakernel 的解法

Megakernel 的策略直截了当：**将全部七阶段的 device 代码编译进同一个 `.so`，执行一次 launch，阶段之间通过核内 FFTS 同步替代 host-device 往返**。中间结果在理想情况下可部分驻留在 UB（片上 SRAM），大幅减少 HBM 访问。

### 1.4 项目代码组织

```
megagdn-pto/├── kernels/pto/            # C++ PTO 内核源码│   ├── chunk_cumsum.cpp    # 前缀和（Vec-only，~430行）│   ├── scaled_dot_kkt.cpp  # 门控 KKT（Cube+Vec，~700行）│   ├── tri_inverse*.cpp    # 三角求逆（CubeCore，~40行实现）│   ├── wy_fast.cpp         # WY 分解（Vec+Cube，~1000行）│   ├── chunk_h.cpp         # 状态递推（Cube+Vec，~1000行）│   ├── chunk_o.cpp         # 输出计算（Cube+Vec，~1000行）│   └── mega_kernel.cpp     # 七合一融合入口（~500行胶水）├── megagdn_pto/            # Python 接口层│   ├── compile.py          # Bisheng JIT 编译管理│   ├── kernel_libs.py      # 分阶段加载与运行│   ├── mega_kernel.py      # 融合 kernel Python 入口│   └── fast_inverse.py     # 三角求逆 Python 封装├── vllm_patch/             # vLLM 运行时 monkey-patch└── third_party/pto-isa/    # PTO-ISA 头文件（git submodule）
```

## 二、NPU 硬件编程模型：Vec、Cube 与显式流水线

### 2.1 AI Core 内部架构

> 理解这份代码的前提是理解昇腾 910B 的 AI Core 架构。每个 AI Core 类比 GPU 的一个 SM（Streaming Multiprocessor），但内部结构截然不同：

| 硬件单元 | 功能 | GPU 类比 | 操作的内存层 |
| --- | --- | --- | --- |
| **Vec** | SIMD 向量运算（add/mul/exp/log/cvt） | CUDA Core | UB（片上 SRAM，~256KB） |
| **Cube** | 矩阵乘法（fp16 输入，fp32 累积） | Tensor Core | L0A/L0B → L0C（寄存器文件） |
| **MTE2** | DMA 加载：GM → UB 或 GM → L1 | 全局内存 Load 单元 | GM → L1/UB |
| **MTE3** | DMA 存储：UB → GM 或 L0C → GM | 全局内存 Store 单元 | L0C/UB → GM |
| **MTE1** | L1 → L0A/L0B 搬运 | — | L1 → L0 |

::: block-1
关键特性：**这五条流水线物理并行、异步执行**。程序员必须通过 `set_flag/wait_flag` 显式管理它们之间的数据依赖——这就像在一个没有硬件缓存一致性的系统上手动管理 memory fence。
:::

更特殊的是，**Cube 和 Vec 运行在不同的物理核上**。它们之间的通信必须经过 Global Memory + 跨核信号（FFTS），这比核内的 pipe 同步昂贵得多。

### 2.2 内存层次与 Tile 抽象

> PTO-ISA 的核心编程模型围绕 `Tile` 模板展开。一个 Tile 就是一块带有形状、布局、对齐信息的内存区域描述符：

```
// 来源：kernels/pto/chunk_cumsum.cpp// UB 上的二维 Tile：ChunkSize 行 × HTC 列，float 类型，行主序// 512 字节对齐是硬件 DMA 要求template <typename T, int R, int C, int RV = R, int CV = C,          PadValue P = PadValue::Null>using UbND = Tile<TileType::Vec, T, R, C, BLayout::RowMajor,                  RV, CV, SLayout::NoneBox, 512, P>;
```

类比 PyTorch：`UbND<float, 128, 16>` 相当于 `torch.empty(128, 16, dtype=float32)`，但分配在每个 AI Core 的片上 SRAM 中。地址由程序员通过 `TASSIGN(tile, byte_offset)` 手动管理——这是一个完全静态的内存池，没有 malloc，没有垃圾回收，一切在编译时确定。

`GlobalTensor` 则是对 HBM 中张量的"视图"：

```
// 来源：kernels/pto/chunk_cumsum.cpp// 等价于 torch.as_strided(ptr, size=[valid, NumHeads], stride=[NumHeads, 1])using GmShape  = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;using GmStride = Stride<1, 1, 1, NumHeads, 1>;using GmFloat  = GlobalTensor<float, GmShape, GmStride>;
```

### 2.3 同步原语一览

> 在这份代码中反复出现的同步模式可以归纳为三类：

**核内 pipe 间同步**：`set_flag(SRC_PIPE, DST_PIPE, event)` + `wait_flag(...)`

- 例如：MTE2 完成 DMA 加载后，Vec 才能读取 UB 数据
- 类比：producer-consumer 之间的 semaphore

**核内全 pipe barrier**：`pipe_barrier(PIPE_ALL)` 或 `pipe_barrier(PIPE_V)`

- 确保某条 pipe 的所有待处理操作排空
- 类比：`__syncthreads()` 的轻量版

**跨核同步**：`ffts_cross_core_sync()` + `wait_flag_dev(flag_id)`

- 用于 Cube 核与 Vec 核之间的握手
- 类比：MPI 中两个 rank 之间的 barrier

## 三、编译系统：Bisheng JIT 的编译期特化

### 3.1 编译期常量注入策略

> 性能敏感参数——head 数量 H、key head 数量 Hg、head 维度 D、chunk 大小 C——全部通过编译器 `-D` 宏注入。这意味着：

- 所有循环边界在编译时已知，编译器可做完整展开
- UB 地址在编译时静态计算，零运行时地址算术
- 模板实例化产出的代码是**该参数组合的最优形式**

```
# 来源：megagdn_pto/compile.pydef _common_flags(*, num_heads, key_heads, hidden_size, chunk_size):    flags = [        "-fPIC", "-shared", "-xcce", "-O2", "-std=gnu++17",        "--cce-aicore-arch=dav-c220",        "-mllvm", "-cce-aicore-stack-size=0x8000",        f"-DGDN_H={num_heads}",    # 值 head 数        f"-DGDN_HG={key_heads}",   # 键 head 数（GQA）        f"-DGDN_D={hidden_size}",  # head 维度        f"-DGDN_C={chunk_size}",   # chunk 大小    ]    return flags
```

每种参数组合产出独立的 `.so`，文件名编码完整签名：`mega_kernel_H16_Hg16_D128_C128.so`。

### 3.2 增量编译与缓存

```
# 来源：megagdn_pto/compile.py@lru_cache(maxsize=None)def compile_mega_kernel(*, num_heads, key_heads, hidden_size, chunk_size, cpp_mtime_ns=0):    """cpp_mtime_ns 作为 cache key：源文件修改时自动触发重编译"""    lib_path = os.path.join(_COMPILED_DIR,        f"mega_kernel_H{num_heads}_Hg{kh}_D{hidden_size}_C{chunk_size}.so")    _run_bisheng(["bisheng", *flags, cpp_path, "-o", lib_path], timeout=600)    return lib_path
```

将 `cpp_mtime_ns`（源文件修改时间戳的纳秒值）作为 `lru_cache` 的 key，实现了"源码一改就重编译"的增量构建语义。无需 Makefile，无需 CMake——Python 的 functools 就是构建系统。

### 3.3 硬件自适应

```
# 来源：megagdn_pto/compile.py_npu_dev = os.environ.get("GDN_NPU_DEVICE", "npu:0")try:    BLOCK_DIM: int = int(        getattr(torch.npu.get_device_properties(_npu_dev), "cube_core_num", 20)    )except (RuntimeError, AssertionError):    BLOCK_DIM = 24  # 安全回退值
```

`BLOCK_DIM`（启动的 AI Core 数量）从硬件属性动态获取。910B2 有 20 个 Cube Core，910B4 有 24 个。这个值决定了 kernel launch 时的并行度——类似 CUDA 的 grid size。

## 四、核心内核逐级剖析

### 4.1 chunk_cumsum：Vec-only 的前缀和

> 这是最简单的阶段，纯 SIMD 运算，不涉及矩阵引擎。数学操作：对每个 chunk 的每个 head 独立做前缀和 `g_sum[t,h] = Σ_{i=0}^{t} g[i,h]`。

```
// 来源：kernels/pto/chunk_cumsum.cpp（核心循环简化）// 1. DMA: 从 GM 加载一个 chunk 的 gate 值到 UBTLOAD(g_load, g_gm);  // 异步！在 MTE2 流水线上执行// 2. 同步：等待 DMA 完成set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);// 3. 前缀和计算：所有 H 个 head 在 SIMD 宽度内并行UbND<float, 1, HTC> acc_ub;  // 累加器，宽度覆盖所有 headTASSIGN(acc_ub, AccUbAddr);TMOV(acc_ub, g_row_0);       // acc = g[0, :]  初始化pipe_barrier(PIPE_V);for (int32_t i = 1; i < valid; ++i) {    UbND<float, 1, HTC> g_row_i;    TASSIGN(g_row_i, GUbAddr + i * RowBytes);    TADD(acc_ub, acc_ub, g_row_i);  // acc += g[i, :]  全 head 并行    pipe_barrier(PIPE_V);    UbND<float, 1, HTC> s_row_i;    TASSIGN(s_row_i, SUbAddr + i * RowBytes);    TMOV(s_row_i, acc_ub);          // g_sum[i, :] = acc    pipe_barrier(PIPE_V);}// 4. 同步 + DMA: 写回结果set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);TSTORE(gs_gm, s_store);
```

**工作分配** 采用类似 CUDA grid-stride loop 的模式：每个 AI Core 按 round-robin 处理不同的 chunk。代码中的 `HTC = ((NumHeads + 7) / 8) * 8` 将 head 数向上对齐到 8 的倍数（32 字节对齐），确保每个 SIMD "字"被完整填充。

::: block-1
值得注意的是，这段代码中的 `pipe_barrier(PIPE_V)` 并非多余——Vec 引擎是流水线化的，前一条指令的结果可能尚未写入 UB 时后一条指令就开始读取，必须显式 barrier 保序。
:::

### 4.2 scaled_dot_kkt：Cube 与 Vec 的双核协作

> 这是复杂度最高的单阶段内核。它同时使用 Cube 引擎做矩阵乘法和 Vec 引擎做门控系数计算，两者运行在**不同的物理核** 上，通过 workspace（GM）和 FFTS 信号协作。

**Cube 侧（矩阵乘法 K@K^T）：**

```
// 来源：kernels/pto/scaled_dot_kkt.cpp（Cube 相关，简化）#if defined(__DAV_C220_CUBE__)// 加载 K [C×D] 从 GM → L1（NZ fractal 格式）TLOAD(_l1, _gm);if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);  // 尾部补零// 转置技巧：TRESHAPE 是零开销的布局重解释// NZ → ZN 等价于逻辑转置，不移动任何数据！L1MatZN<half, HiddenSize, ChunkSize> _bzn;TRESHAPE(_bzn, k_l1);  // K^T "免费"获得// L1 → L0A/L0B 搬运（MTE1 流水线）TEXTRACT(_l0a, k_l1, 0, 0);   // 左操作数 KTEXTRACT(_l0b, _bzn, 0, 0);   // 右操作数 K^T// 矩阵乘：K @ K^T → L0C 累加器（fp16×fp16→fp32）TMATMUL(a_l0, _l0a, _l0b);// L0C → workspace GM（自动 fp32→fp16 截断）TSTORE(_gm, _l0);// 跨核信号：通知 Vec "KK^T 已就绪"ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (slot << 8));#endif
```

**Vec 侧（门控系数与掩码应用）：**

```
// 来源：kernels/pto/scaled_dot_kkt.cpp（Vec 相关，简化）#if defined(__DAV_C220_VEC__)// 等待 Cube 完成 KK^T 写入wait_flag_dev(slot);// 计算门控系数：coeff[i,j] = exp(min(g[i]+log(β[i]) - g[j], 0))TCVT(beta_ub, beta_ub_half, RoundMode::CAST_NONE);  // fp16→fp32TLOG(beta_ub, beta_ub);                               // log(β)TADD(g_v_ub, g_v_ub, beta_ub);                       // g + log(β)// 二维广播：g_v → 行方向，g → 列方向TROWEXPAND(g_r_2d_ub, g_r_ub_temp);  // g_r_2d[i,j] = g_v[i]TCOLEXPAND(g_c_2d_ub, g_c_ub);       // g_c_2d[i,j] = g[j]// 差值 + clamp + expTSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);   // diff = g_v[i] - g[j]TMINS(coeff_ub, coeff_ub, 0.0f);          // clamp to ≤ 0TEXP(coeff_ub, coeff_ub);                  // exp → (0, 1]// 最终：A = KK^T × coeff × causal_maskTCVT(a_ub, a_ub_half, RoundMode::CAST_NONE);  // 加载 KK^T 到 fp32TMUL(a_ub, a_ub, coeff_ub);                    // 应用门控TMUL(a_ub, a_ub, msk_ub);                      // 应用因果掩码// 信号 Cube：workspace slot 可复用ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((2 + slot) << 8));#endif
```

- **双缓冲（Double Buffering）**：Cube 和 Vec 通过两个 workspace slot 交替工作。当 Vec 处理 slot 0 的 KK^T 时，Cube 可以并行计算 slot 1 的下一个 chunk。这种经典的流水线重叠技巧有效隐藏了跨核通信延迟。
- **Sub-block 并行**：每个 AI Core 的 Vec 引擎有 2 个 sub-block（vid=0, vid=1），分别处理 C×C 矩阵的上半部和下半部，将单核 Vec 吞吐翻倍。

### 4.3 wy_fast：WY 分解的 Vec+Cube 协作

> WY 分解需要两次矩阵乘法（`U = A2 @ V`, `W = A1 @ K`），但 A2 和 A1 本身需要先通过逐元素运算从原始 A 矩阵派生。项目的设计是：

- **Vec 核**：加载 A 和参数，计算 `A2 = A * β_2d`（beta 缩放）和 `A1 = A * (exp(g)*β)_2d`（门控+beta 缩放），写入 workspace
- **Cube 核**：从 workspace 加载 A2/A1，从 GM 加载 K/V，执行两次 GEMM，输出 U/W

```
// 来源：kernels/pto/wy_fast.cpp（Vec 侧核心逻辑简化）// 广播 beta 到 2DTCOLEXPAND(beta_2d_ub, beta_r_ub);  // beta_2d[i,j] = beta[j]// A2 = A * beta_2d（用于 U = A2 @ V）TMUL(a2_ub, a1_ub, beta_2d_ub);// 存储 A2 到 workspace，通知 CubeTSTORE(workspace_a2_global, a2_ub_half);ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));// A1 = A * (exp(g) * beta)_2d（用于 W = A1 @ K）TEXP(g_ub, g_ub);              // exp(g)TMUL(g_ub, g_ub, beta_ub);    // exp(g) * betaTCOLEXPAND(g_2d_ub, g_r_ub);  // 广播到 2DTMUL(a1_ub, a1_ub, g_2d_ub);  // A1 = A * weight_2dTSTORE(workspace_a1_global, a1_ub_half);ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
```

Cube 侧使用了一个精心编写的 `gemm_v0` 模板函数，将 K 维度按 128 切片做分块矩阵乘法：

```
// 来源：kernels/pto/wy_fast.cpp（Cube 侧 GEMM helper 简化）template <...>AICORE void gemm_v0(A, B, C, clear) {    constexpr uint32_t kL0Size = 128;  // L0 寄存器文件一次容纳 128 列    for (uint32_t kIdx = 0; kIdx < kSplit; ++kIdx) {        TEXTRACT(l0a, A, 0, kIdx * kL0Size);  // L1 → L0A        TEXTRACT(l0b, B, kIdx * kL0Size, 0);  // L1 → L0B        if (kIdx == 0 && clear)            TMATMUL(C, l0a, l0b);              // C = A_slice @ B_slice        else            TMATMUL_ACC(C, C, l0a, l0b);       // C += A_slice @ B_slice    }}
```

### 4.4 solve_tril：CubeCore 递推三角求逆

> 三角矩阵求逆（128×128 下三角）是整个流水线中算法最精巧的环节。项目在 CubeCore 上使用**递推展开** 策略：将 128×128 矩阵递归分割为子块，利用分块三角逆公式在 Cube 上高效执行。

```
// 来源：kernels/pto/mega_kernel.cppAICORE void mega_solve_tril(...) {    // 根据矩阵总数动态选择展开因子    // 负载轻时少展开（节省 L0 寄存器），负载重时多展开（提升并行度）    if (num_matrices <= get_block_num())        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 1, true, half>(...);    else if (num_matrices <= 2u * get_block_num())        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 2, true, half>(...);    else        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 4, true, half>(...);}
```

Python 侧的封装展示了调用约定：

```
# 来源：megagdn_pto/fast_inverse.pydef launch_tri_inverse_kernel(tensor_out, tensor_in, minus_identity, ...):    lib = _tri_inverse_cdll()    # is_lower 标志通过位域打包进 num_bsnd_heads 参数    heads_with_flag = (num_bsnd_heads & 0xFFFF) | (0x10000 if is_lower else 0)    lib.call_kernel(eff_bd, stream_ptr,        _vp(tensor_out), _vp(tensor_in), _vp(minus_identity),        matrix_size, num_matrices, heads_with_flag, _vp(cu_seqlens))
```

### 4.5 Megakernel 融合入口：七阶段一体化

> Megakernel 的核心技巧在 `mega_kernel.cpp`——它通过 `#include` + namespace 隔离将所有独立内核的实现代码"内联"进同一编译单元：

```
// 来源：kernels/pto/mega_kernel.cpp// 通过宏重定向避免 call_kernel 符号冲突#define call_kernel _mk_unused_gv_ck_cumsumnamespace mk_cumsum { #include "chunk_cumsum.cpp" }#undef call_kernel#define call_kernel _mk_unused_gv_ck_kktnamespace mk_kkt { #include "scaled_dot_kkt.cpp" }#undef call_kernel// ... 对所有阶段重复此模式
```

然后在统一的 `launch_mega_kernel` 入口中顺序调用：

```
// 来源：kernels/pto/mega_kernel.cppextern"C"__global__ AICORE void launch_mega_kernel(...) {    set_ffts_base_addr(ffts_addr);    constexprint32_t H = GDN_H, HG = GDN_HG, D = GDN_D, C = GDN_C;    // Stage 1: 前缀和    mk_cumsum::cumsum_kernel<H, C>(g_in, g_sum, cu_seqlens, ...);    SyncAllImpl<false>();  // 全核屏障    // Stage 2: 转置 [T,H] → [H,T]    mega_transpose_TH_to_HT<float, H>(g_sum, g_t, total_tokens);    mega_transpose_TH_to_HT<half, H>(beta, beta_t, total_tokens);    SyncAllImpl<false>();    // Stage 3: 门控 KKT    mk_kkt::kkt_kernel<H, HG, D, C>(k, beta_t, g_t, mask, ws, A, ...);    SyncAllImpl<false>();    // Stage 4: 三角求逆    mega_solve_tril(A_inv, A, minus_id, C, num_matrices, H, cu_seqlens, 1);    SyncAllImpl<false>();    // Stage 5: WY 分解    mk_wy::wy_fast_kernel<H, HG, D, C>(k, v, beta_t, g_t, A_inv, ..., w, u, ...);    SyncAllImpl<false>();    // Stage 6: 状态递推    mk_h::chunk_h_kernel<H, HG, D, C>(k, w, u, g_t, s, v_new, fs, ...);    SyncAllImpl<false>();    // Stage 7: 输出    mk_o::chunk_o_kernel<H, HG, D, C>(q, k, v_new, s, g_t, mask, ..., o, ...);}
```

### 4.6 异构屏障 SyncAllImpl 的三步握手

> 阶段间的 `SyncAllImpl<false>()` 是整个 Megakernel 的"粘合剂"——它确保所有 AI Core 的 Vec 和 Cube 都完成当前阶段后才进入下一阶段：

```
// 来源：kernels/pto/mega_kernel.cpptemplate <bool isAIVOnly>AICORE inline void SyncAllImpl() {    pipe_barrier(PIPE_ALL);  // Step 0: 本核内所有流水线排空    if constexpr (isAIVOnly) {        // 纯 Vec 场景：所有 Vec 核互相等待        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));        wait_flag_dev(SYNC_AIV_ONLY_ALL);    } else {        // Vec + Cube 异构场景：三步握手        #if defined(__DAV_C220_CUBE__)        wait_flag_dev(SYNC_AIV_FLAG);     // Step 1: Cube 等 Vec 完成        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));        wait_flag_dev(SYNC_AIC_FLAG);     // Step 2: Cube 核间自同步        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));                                          // Step 3: Cube 通知 Vec 可以继续        #elif defined(__DAV_C220_VEC__)        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));                                          // Step 1: Vec 通知 Cube 自己已完成        wait_flag_dev(SYNC_AIC_AIV_FLAG); // Step 3: Vec 等 Cube 确认        #endif    }}
```

这个三步握手保证：Vec 完成 → 通知 Cube → Cube 完成并确认 → 通知 Vec → 双方同时进入下一阶段。没有这个机制，一个阶段的 Cube 可能在 Vec 尚未写完中间结果时就开始读取。

## 五、Python 接口层：从编译到调用的完整链路

### 5.1 ctypes 零拷贝调用

> Python 层通过 ctypes 直接加载编译产物并调用 kernel 入口。所有 tensor 数据通过指针传递，不发生任何拷贝：

```
# 来源：megagdn_pto/mega_kernel.pydef run_mega_kernel(q, k, v, g_in, beta, cu_seqlens, *, stream, ...):    # 预分配全部中间缓冲区    g_sum  = torch.empty(1, T, H, device=dev, dtype=torch.float32)    A      = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)    s      = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)    # ... 还有 workspace（供各阶段的 Cube 和 Vec 使用）    kkt_ws = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)    h_ws   = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)    # 一次调用完成全部七阶段    lib = _load_mega_kernel(num_heads=H, key_heads=kh, hidden_size=D, chunk_size=C)    lib.call_kernel(        bd, stream,                                    # block_dim, stream        _vp(q), _vp(k), _vp(v), _vp(g_in), _vp(beta),  # 输入        _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu_seqlens),        _vp(o_out),                                    # 输出        _vp(g_sum), _vp(g_t), _vp(beta_t),            # 中间结果        _vp(A), _vp(A_inv_f32), _vp(A_inv),        _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),        _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),        _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),  # workspace        N_seq, T, T, num_matrices,                     # 标量元数据    )    return o_out * scale  # scale = head_dim ** -0.5
```

`_vp(tensor)` 的实现极为简洁：

```
# 来源：megagdn_pto/kernel_libs.pydef _vp(t: torch.Tensor | None) -> ctypes.c_void_p:    if t is None:        return ctypes.c_void_p()    return ctypes.c_void_p(t.data_ptr())
```

### 5.2 GQA（Grouped Query Attention）原生支持

> 当 `key_heads < num_heads`（如 Qwen3 的 MoE 模型），Q/K 使用 Hg 个 head 而 V/gates 使用 H 个 head。项目通过编译期的 `GDN_H` 和 `GDN_HG` 两个宏分别控制，内核中使用 `GROUP = H / Hg` 计算 GQA 分组：

```
// 来源：kernels/pto/scaled_dot_kkt.cppconstexpr int32_t GROUP = NumHeads / NumKeyHeads;// K 的 head 索引 = value head 索引 / GROUPint32_t head_g = head_idx / GROUP;
```

### 5.3 Variable-length sequence 支持

全部内核原生支持 `cu_seqlens`（cumulative sequence lengths）编码的变长序列打包格式（packed varlen），与 vLLM 的内部表示完全一致，无需 padding。

## 六、vLLM 集成：运行时 Monkey-Patch 架构

### 6.1 非侵入式钩子注入

> 项目通过 `vllm_patch/install_hook.py` 对已安装的 vllm-ascend 包做最小限度的源码修改，在 worker 初始化路径中注入一个环境变量驱动的钩子：

```
# 来源：vllm_patch/install_hook.py（注入到 vllm_ascend 的代码片段）_pto_dir = _pto_os.environ.get("VLLM_PTO_PATCH_DIR")if _pto_dir and _pto_os.path.isdir(_pto_dir):    _pto_sys.path.insert(0, _pto_dir)    from apply import apply_pto_patch    apply_pto_patch()  # 将 chunk_gated_delta_rule 替换为 PTO 实现
```

这意味着：

- **不设 `VLLM_PTO_PATCH_DIR`：** 使用原始 Triton kernel，完全向后兼容
- **设置环境变量**：透明替换为 PTO Megakernel，获得 1.5~3x 加速

### 6.2 为何使用 monkey-patch 而非 fork

Qwen 模型在 vllm-ascend 中的 GDN 调用路径涉及多个文件的 import chain。项目将 `from vllm...ops import chunk_gated_delta_rule` 改为动态查找 `_vllm_fla_ops.chunk_gated_delta_rule`，使得 `apply_pto_patch()` 能在运行时替换该函数引用。这种设计允许项目跟随上游 vllm-ascend 版本升级而无需维护 fork 分支。

## 七、性能分析与工程取舍

### 7.1 加速的三层来源

| 层面 | 机制 | 估算收益 |
| --- | --- | --- |
| Launch 消除 | 7 次 →1 次 kernel dispatch | ~30% 延迟节省 |
| GM 流量减少 | 中间结果不反复穿越 HBM | ~20-40% 带宽节省 |
| 编译期特化 | 循环展开 + 静态地址 + 无分支 | ~10-30% 指令效率提升 |

综合效果在不同 head 配置下达到 1.5x~3x 的端到端加速。README 中展示的 prefill benchmark 显示，在 Qwen3.6-35B-A3B (W8A8) 上，PTO Megakernel 相比 Triton 基线带来约 15% 的端到端 prefill 吞吐提升。

### 7.2 首次编译开销

每组 (H, Hg, D, C) 参数需要独立编译一次 Megakernel。`compile_mega_kernel` 设置了 600 秒超时，实际编译时间取决于代码量（Megakernel 包含 ~200K 行展开后的 C++）。对于生产推理场景，模型参数固定，编译仅发生一次并永久缓存。

### 7.3 混合精度保障

> 全链路采用 fp16 输入 → fp32 中间累积 → fp16 输出 的混合精度策略。特别是：

- Cube GEMM 天然以 fp32 累积（硬件特性）
- 三角求逆在 fp32 空间完成递推以避免数值不稳定
- 门控系数 `exp(...)` 在 fp32 空间计算后再转回 fp16

端到端 lm-eval 评测（覆盖 0.8B~35B 模型）确认了替换后所有评分指标无损。

### 7.4 调试与可观测性

项目提供了分阶段执行模式（`kernel_libs.py` + `fast_inverse.py`），可以单独运行每个阶段并检查中间结果。此外，`mega_kernel.cpp` 中预埋了多个条件编译断点：

```
#ifdef MEGA_STOP_AFTER_CUMSUM    pipe_barrier(PIPE_ALL);    return;  // 只执行到 cumsum 就返回#endif
```

通过编译时添加 `-DMEGA_STOP_AFTER_KKT` 等标志，可以逐阶段验证 Megakernel 的中间结果与分阶段版本的一致性。

## 八、技术洞察与总结

> MegaGDN-PTO 项目的核心贡献可以从三个维度理解：

- **算法层面**：完整实现了 Gated DeltaNet 的六阶段 chunk 计算（外加转置阶段），覆盖 GQA 分组、变长序列打包等生产级需求。
- **系统层面**：通过 Megakernel 融合消除了七次 launch 开销，通过编译期参数化消除了运行时分支，通过双缓冲和 sub-block 并行最大化了硬件利用率。
- **工程层面**：设计了一套从 JIT 编译到 ctypes 调用到 vLLM monkey-patch 的完整工具链，使得底层内核优化能够以"插件"形式无缝接入上层推理框架。

从更宏观的视角看，这个项目揭示了当前 NPU 算子优化的一个重要趋势：**当自动编译器（如 Triton-Ascend）的性能无法满足极致需求时，手写底层内核仍然是最后的杀手锏**。PTO-ISA 在裸金属硬件之上提供了一层恰到好处的抽象——Tile 描述符和显式 pipe 同步——让开发者能够精确控制每一次 DMA 搬运、每一条 SIMD 指令的执行时机，代价是需要像三十年前的汇编程序员一样思考每一个时钟周期的去处。

::: block-1
对于关注昇腾生态的开发者而言，这个项目不仅是一份高性能 GDN 实现，更是:mdnice-bold-italic[目前公开可获取的、最为详尽的 PTO-ISA 编程范例——它完整展示了 Vec-only 内核（cumsum）、Cube-only 内核（tri_inverse）、以及 Vec+Cube 双核协作内核（kkt/wy/chunk_h/chunk_o）三种典型模式的工程实践，堪称昇腾 NPU 底层编程的"教科书级"参考。]
:::

参考资料

[1] vllm-ascend Docker 镜像: *https://quay.io/repository/ascend/vllm-ascend?tab=tags*

[2] CANN Docker 镜像: *https://quay.io/repository/ascend/cann?tab=tags*

[3] `README.md`: *https://github.com/huawei-csl/megagdn-pto/blob/main/README.md*
