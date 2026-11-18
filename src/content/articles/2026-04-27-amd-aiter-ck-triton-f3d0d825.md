关键词：AITER、**Fused MoE**、JIT 编译、**GEMM 自动调优**、:mdnice-bold-italic[ROCm 算子加速]

> 在大模型推理的战场上，算子性能是一切的地基。当 Transformer 的参数量突破千亿、MoE 架构成为标配，单一 kernel 实现早已无法满足不同形状、不同精度、不同硬件代际的需求。

![文章插图](/articles/2026-04-27-amd-aiter-ck-triton-f3d0d825/images/01.webp)

- **AITER（面向ROCm的 AI 张量引擎）是 AMD 的高性能 AI 算子库**，为基于 ROCm 的推理与训练负载提供经过优化的GPU核心运算程序。该库整合了各类可直接投入生产使用的算子，框架开发人员可将其直接集成至自身技术架构体系中。
- https://github.com/ROCm/aiter
- https://rocm.github.io/aiter/
- 4000 子，阅读 15 分钟，**播客 13 分钟**

相关推荐

- [AMD GPU 设备数学库分层编译架构浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899885&idx=1&sn=c4a604edeff740435f04ea5b6f90bf7c&scene=21#wechat_redirect)
- [对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)
- [DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)

> AMD 推出的 AITER（AI Tensor Engine for ROCm）正是瞄准这一痛点——它不是一个简单的算子集合，而是一个**融合了汇编手写内核（ASM）、Composable Kernel（CK）代码生成、Triton JIT 以及自研 FlyDSL 四种后端** 的统一调度框架。

![文章插图](/articles/2026-04-27-amd-aiter-ck-triton-f3d0d825/images/02.webp)

::: image-caption
AITER 是 AMD 的集中式代码库，为 AI 工作负载加速提供各类高性能算子，同时也是统一接收所有自定义算子级需求的优质平台，能够满足不同定制化需求。开发者可专注于算子开发，而客户则能将该算子集合集成到自身的私有、公有或其他各类框架中。
:::

更关键的是，它内建了从"未调优形状发现"到"在线自动 Tuning"再到"CSV 配置热加载"的完整闭环，让每一个 GEMM 和 MoE 调用都能在运行时命中最优内核。

::: block-1
本文将从架构设计、JIT 编译基础设施、Fused MoE 核心流水线、GEMM 自动调优四个维度，带你深入 AITER 的源码腹地。
:::

该框架集成详情如下表：

| 框架 | 集成方式 | 状态 | 所用算子 |
| --- | --- | --- | --- |
| vLLM | ROCm 平台默认注意力后端 | 生产可用 | 多头注意力、多头线性注意力、分页注意力、混合专家融合、通用矩阵乘法、均方根归一化、旋转位置编码+键值缓存 |
| SGLang | ROCm 容器镜像默认集成 | 生产可用 | 注意力机制、混合专家融合、分块缩放矩阵乘法、全规约通信、均方根归一化 |
| ATOM | 基于 AITER 原生构建 | 持续开发中 | 全部 AITER 算子（注意力、混合专家、采样、通信算子） |
| JAX | 依托 XLA 外部函数接口桥接，无 PyTorch 依赖 | 实验阶段 | 多头注意力/融合多头注意力、均方根归一化、BF16 精度矩阵乘法 |
| 多家客户自研推理引擎 | 内核级深度集成 | 生产可用 | 注意力、混合专家、通用矩阵乘法、量化算子 |

关键算子性能如下，详情可以参考：ATOM/benchmark-dashboard[1]：

| 算子 | 加速比 |
| --- | --- |
| 多头线性注意力解码内核（MLA decode kernel） | 最高 **17 倍** |
| 多头注意力预填充内核（MHA prefill kernel） | 最高 **14 倍** |
| 分块缩放融合混合专家（Block-scaled Fused MoE） | 最高 **3 倍** |
| 分块缩放通用矩阵乘法（Block-scaled GEMM） | 最高 **2 倍** |
| DeepSeek-R1 端到端推理（SGLang） | 6,484 → 13,704 令牌/秒（**2.1 倍**） |
| JAX-AITER 注意力算子（MI350 平台） | 中位数加速 **4.39 倍** |

![文章插图](/articles/2026-04-27-amd-aiter-ck-triton-f3d0d825/images/03.webp)

::: image-caption
ATOM项目的性能基准测试仪表盘，基于ROCm 7.2.2运行在AMD Radeon（MI355X）显卡上。仪表盘显示峰值总吞吐量达58.6K tok/s，较基准提升3.1%，最低单Token延迟（TPOT）为4.10ms。测试覆盖24个模型、449种配置，存在28项性能波动与4项精度问题。下方柱状图对比了DeepSeek-R1-0528模型及其MTP3变体在不同并发下的吞吐量，最高分别达到24.9K和28.9K tok/s，直观展现了ATOM对大模型推理的优化效果。详见：https://rocm.github.io/ATOM/benchmark-dashboard/
:::

当前支持的 AMD GPU 型号如下：

| 显卡型号 GPU | 架构 Architecture | 支持状态 Status |
| --- | --- | --- |
| AMD Instinct MI300X | gfx942 (CDNA3) | 完全支持 |
| AMD Instinct MI325X | gfx942 (CDNA3) | 完全支持 |
| AMD Instinct MI350 | gfx950 (CDNA4) | 支持 |
| AMD Instinct MI355X | gfx950 (CDNA4) | 支持 |

## 本文目录

- 快速上手
- 一、架构总览与设计哲学
  - 1.1 一个入口，四种后端
  - 1.2 核心数据流
- 二、JIT 编译基础设施：从源码到 .so 的全自动流水线
  - 2.1 compile_ops 装饰器：声明即编译
  - 2.2 build_module：多进程安全的编译引擎
  - 2.3 torch_compile_guard：与 torch.compile 无缝对接
- 三、Fused MoE：Token 排序与多阶段 GEMM 的精密编排
  - 3.1 Token 排序：为 GPU 并行铺路
  - 3.2 一阶段 vs 两阶段：动态决策引擎
  - 3.3 Block Size 自适应：榨干每一个 CU
  - 3.4 K-Split：低 Token 数场景下的 CU 利用率救星
- 四、GEMM 自动调优：从形状发现到最优内核的闭环
  - 4.1 多后端路由表
  - 4.2 Skinny GEMM：小 M 场景的秘密武器
  - 4.3 在线调优与离线调优的双轨机制
- 五、量化生态：从 FP4 到 INT8 的全精度覆盖
  - 5.1 量化类型的精细分级
  - 5.2 MXFP4 量化的融合排序
- 六、有趣的地方
  - 6.1 LRU 缓存的大量使用
  - 6.2 FlyDSL 回退机制
  - 6.3 ctypes FFI：绕过 PyTorch 的无 Torch 调用路径
- 结语

![文章插图](/articles/2026-04-27-amd-aiter-ck-triton-f3d0d825/images/04.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> AITER 的安装非常直接，只需三行命令：

```
git clone --recursive https://github.com/ROCm/aiter.gitcd aiterpython3 setup.py develop
```

如需 FlyDSL 混合精度 MoE 支持，可额外安装：

```
pip install --pre flydsl
```

更多安装选项（Triton 通信、Iris 库等），参考 `README.md`。安装完成后即可在 Python 中 `import aiter` 使用全部算子，例如运行内置测试：

```
python3 op_tests/test_layernorm2d.py
```

## 一、架构总览与设计哲学

### 1.1 一个入口，四种后端

> AITER 最核心的设计理念可以用一句话概括：**用 Python 层统一 API 屏蔽底层内核异构性，通过配置驱动在运行时选择最优后端**。

从 `aiter/__init__.py` 的导入结构可以清晰看到这种分层：

```
# 来源：aiter/__init__.pyfrom .ops.gemm_op_a8w8 import *    # INT8/FP8 量化 GEMMfrom .ops.gemm_op_a16w16 import *  # BF16/FP16 GEMMfrom .ops.gemm_op_a4w4 import *    # FP4 量化 GEMMfrom .ops.moe_op import *          # MoE 算子from .ops.attention import *       # MHA/MLA/PAfrom .ops.activation import *      # SiLU/GeLU 等激活from .ops.rope import *            # RoPE 位置编码
```

每个 `ops` 模块对外暴露统一的函数签名，内部则根据**硬件架构（gfx942/gfx950）、矩阵形状（M/N/K）、量化类型（per_Token/per_1x128/per_1x32）和预调优配置（CSV 文件）** 动态路由到不同的后端实现。这四种后端分别是：

- **ASM 汇编内核**：手写的 GCN/RDNA 汇编，极致性能但形状受限
- **CK（Composable Kernel）**：AMD 开源的模板化 C++ 代码生成框架
- **Triton**：基于 OpenAI Triton 的 Python DSL 内核
- **FlyDSL**：AMD 自研的领域特定语言，专攻混合精度 MoE

### 1.2 核心数据流

> 一次典型的 Fused MoE 推理调用，数据流如下：

```
用户调用 fused_moe()    → 量化类型推断 & 形状对齐    → get_2stage_cfgs() 查询 CSV 配置    → moe_sorting() Token 按专家排序    → 1-stage 或 2-stage GEMM 执行    → 输出 moe_buf
```

这条流水线的每一环都蕴含精巧的工程决策，我们逐一展开。

## 二、JIT 编译基础设施：从源码到 .so 的全自动流水线

### 2.1 compile_ops 装饰器：声明即编译

> AITER 的 JIT 系统是整个项目的骨架。

开发者只需用 `@compile_ops` 装饰一个**仅含类型签名的空函数**，框架就能自动完成"查找预编译 .so → 未找到则触发 JIT 编译 → 加载模块 → 调用 C++ 算子"的全流程。

```
# 来源：aiter/ops/gemm_op_a8w8.py@compile_ops(    "module_gemm_a8w8", fc_name="gemm_a8w8",    gen_fake=gen_gemm_a8w8_ck_fake_tensors)def gemm_a8w8_ck(    XQ: torch.Tensor, WQ: torch.Tensor,    x_scale: torch.Tensor, w_scale: torch.Tensor,    Out: torch.Tensor, bias: Optional[torch.Tensor] = None,    splitK: int = 0,) -> torch.Tensor: ...
```

注意函数体是 `...`（Ellipsis）——它根本不需要 Python 实现。`compile_ops` 内部的 `wrapper` 函数会：

1. 尝试 `get_module(md_name)` 加载已编译模块
2. 若抛出 `ModuleNotFoundError`，调用 `build_module()` 启动编译
3. 编译完成后通过 `getattr(module, loadName)` 获取 C++ 函数指针

### 2.2 build_module：多进程安全的编译引擎

> `build_module` 是 JIT 系统的重型核心。它需要处理几个关键难题：

**多进程编译锁**：在多 GPU 训练场景中，多个进程可能同时触发同一模块的编译。AITER 使用文件锁（FileBaton）机制来保证只有一个进程执行编译：

```
# 来源：aiter/jit/core.pydef mp_lock(lockPath, MainFunc, FinalFunc=None, WaitFunc=None):    baton = FileBaton(lockPath)    if baton.try_acquire():        try:            ret = MainFunc()        finally:            if FinalFunc is not None:                FinalFunc()            baton.release()    else:        baton.wait()  # 其他进程等待
```

**编译标志自适应**：框架会自动检测 HIP 编译器版本，逐项添加优化标志。例如，对于 ROCm 6.2+ 会启用 `amdgpu-early-inline-all` 和 `amdgpu-function-calls=false`；对于 gfx950 架构会自动添加 FP4 类型支持标志：

```
# 来源：aiter/jit/core.py（build_module 内部）if hip_version > Version("6.2.41132"):    flags_hip += [        "-mllvm -amdgpu-early-inline-all=true",        "-mllvm -amdgpu-function-calls=false",    ]if get_gfx() == "gfx950" and int(os.getenv("AITER_FP4x2", "1")) > 0:    flags_hip += ["-D__Float4_e2m1fn_x2"]
```

### 2.3 torch_compile_guard：与 torch.compile 无缝对接

AITER 的每个算子都通过 `torch_compile_guard` 装饰器注册为 `torch.ops.aiter.*` 自定义算子，使其能被 `torch.compile` 的计算图捕获和优化。装饰器会自动推断算子 schema、生成 FakeTensor（用于 symbolic tracing）、并处理 CUDA/CPU 双端分发：

```
# 来源：aiter/jit/utils/torch_guard.pyaiter_lib.define(op_schema, tags=tags)aiter_lib.impl(f"aiter::{loadName}", custom_func, dispatch_key="CUDA")aiter_lib.impl(f"aiter::{loadName}", custom_func, dispatch_key="CPU")aiter_lib._register_fake(f"{loadName}", fake_func)
```

这意味着用户可以直接在 `torch.compile(model)` 中使用 AITER 算子，编译器会正确处理形状推导和内存分配。

## 三、Fused MoE：Token 排序与多阶段 GEMM 的精密编排

Fused MoE 是 AITER 中最复杂也最关键的算子，整个 `fused_moe.py` 长达近 2000 行。它需要将 "门控路由选择 → Token 重排序 → 专家 GEMM 计算 → 加权聚合" 这条 MoE 流水线融合为尽可能少的 kernel launch。

### 3.1 Token 排序：为 GPU 并行铺路

MoE 推理的第一步是将 Token 按被分配的专家 ID 重新排序，使得同一专家处理的 Token 在内存中连续，从而让后续 GEMM 能以密集矩阵乘法的形式高效执行。

```
# 来源：aiter/fused_moe.pydef _moe_sorting_impl(...):    M, topk = topk_ids.shape    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd    fwd_fn(topk_ids, topk_weights, sorted_ids, sorted_weights, ...)
```

这里有一个精妙的**padding 策略**：`max_num_tokens_padded` 不仅包含所有 Token×TopK 的展开，还为每个专家额外预留 `block_size` 的对齐空间。这保证了后续 Tile-based GEMM 不会因为尾部 Token 不足一个 Block 而需要特殊处理。

### 3.2 一阶段 vs 两阶段：动态决策引擎

> AITER 为 MoE 提供了两种执行路径：

- **1-stage**：一个 kernel 同时完成 Gate+Up 投影、激活函数和 Down 投影
- **2-stage**：拆成两个 kernel——Stage1 做 Gate+Up+激活，Stage2 做 Down 投影

选择哪种路径由 `get_2stage_cfgs()` 函数决定，它是整个 MoE 调度的"大脑"。决策依据包括：

```
# 来源：aiter/fused_moe.py（get_2stage_cfgs 内部）# 一阶段路径的启用条件因量化类型而异if q_type == QuantType.per_1x128:    run_1stage = token > 32 and (inter_dim % 128 == 0)elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:    run_1stage = token > 16 or inter_dim % 128 != 0elif q_type != QuantType.per_1x32:    run_1stage = token < 256
```

这背后的直觉是：

- 当 Token 数较少（decode 场景）时，1-stage 能减少 kernel launch 开销；
- 当 Token 数较大（prefill 场景）时，2-stage 允许中间结果在两次 GEMM 之间做更精细的量化（如 MXFP4 动态量化），从而获得更好的计算密度。

### 3.3 Block Size 自适应：榨干每一个 CU

`get_block_size_M` 函数展示了一种极致的硬件感知调度策略：

```
# 来源：aiter/fused_moe.py@functools.lru_cache(maxsize=2048)def get_block_size_M(token, topk, expert, inter_dim):    cu_num = get_cu_num()          # GPU Compute Unit 数量    tileN = 128    tgN = (inter_dim + tileN - 1) // tileN    support_list = [32, 64, 128]    tmp = []    for el in support_list:        max_num_tokens = token * topk + expert * el - topk        tg_num = tgN * (max_num_tokens + el - 1) // el  # 总 Tile Group 数        rnd = (tg_num + cu_num - 1) // cu_num            # 每 CU 需执行的轮数        empty = cu_num - tg_num % cu_num                  # 空闲 CU 数        tmp.append((rnd, empty, el))    return sorted(tmp, key=lambda x: x[:2])[0][-1]
```

它遍历候选的 Block Size（32/64/128），计算每种配置下的**总 Tile 数 → CU 利用率 → 空闲 CU 数**，选出"执行轮数最少、空闲 CU 最少"的那个。这相当于在 Python 层做了一次简化的 GPU occupancy 分析。

### 3.4 K-Split：低 Token 数场景下的 CU 利用率救星

在 decode 场景（Token 数极少，如 1-8），MoE 的 GEMM 矩阵 M 维度很小，导致 Tile 总数不足以填满所有 CU。`get_ksplit` 通过在 K 维度上拆分计算来人为制造更多并行度：

```
# 来源：aiter/fused_moe.pydef get_ksplit(token, topk, expert, inter_dim, model_dim):    if token * topk > expert:        return 0  # Token 足够多，不需要 split    cu_num = get_cu_num()    tg_num = tgN * tgM    if tg_num >= cu_num:        return 0  # CU 已经满了    split_max = (cu_num + tg_num - 1) // tg_num    for i in reversed(range(2, split_max + 1)):        if (model_dim % i == 0) and ((model_dim // i) % 256 == 0):            return i    return 0
```

这就像把一个大蛋糕（K 维度的累加）切成多份让更多人（CU）同时处理，最后再合并——一种经典的 split-K 并行策略。

## 四、GEMM 自动调优：从形状发现到最优内核的闭环

### 4.1 多后端路由表

> AITER 的 BF16/FP16 GEMM 支持五种后端，通过一个简洁的路由表切换：

```
# 来源：aiter/tuned_gemm.pysolMap = {    "torch":     torch_gemm,     # PyTorch F.linear 兜底    "hipblaslt": hipb_gemm,      # hipBLASLt 库    "skinny":    skinny_gemm,    # 超窄矩阵专用汇编    "asm":       asm_gemm,       # 手写汇编 kernel    "triton":    triton_gemm,    # Triton JIT kernel}
```

路由决策完全由 CSV 配置文件驱动。`get_GEMM_A16W16_config` 函数会用 `(cu_num, M, N, K, dtype, otype, ...)` 作为索引查表：

```
# 来源：aiter/tuned_gemm.pyconfig = cfg.get(    (cu_num, padded_M, N, K, bias, str(dtype), str(otype), scaleAB, bpreshuffle),    None,)
```

如果精确形状未命中，还会尝试两种 padding 策略（`get_padded_m` 的 `gl=0` 和 `gl=1`）进行模糊匹配。

### 4.2 Skinny GEMM：小 M 场景的秘密武器

> 当 M 极小（1-16）且 N 不超过 CU 数的 1-2 倍时，AITER 会绕过通用 GEMM 库，直接使用手写的 Split-K 汇编内核：

```
# 来源：aiter/tuned_gemm.py（默认路由逻辑）if (    ((M == 1 and N <= 2 * cu_num) or (M > 1 and M <= 4 and N <= cu_num))    and K <= 9216    or (M > 4 and M <= 8 and N <= cu_num) and K <= 5120):    default_config["libtype"] = "skinny"    default_config["solidx"] = 2
```

这些内核针对 MI300X 的 304 个 CU 特别优化，在 decode 场景（M=1）下通常比通用库快 2-3 倍。

### 4.3 在线调优与离线调优的双轨机制

> AITER 的 MoE 层支持**在线调优**：当运行时遇到 CSV 中没有的形状组合，且环境变量 `AITER_ONLINE_TUNE=1` 时，框架会自动触发一次调优流程：

```
# 来源：aiter/fused_moe.py（get_2stage_cfgs 内部）if cfg is None and os.environ.get("AITER_ONLINE_TUNE", "0") == "1":    lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{keys}")    mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
```

调优脚本 `gemm_moe_tune.py` 会被直接调用，结果写回 CSV。这意味着**第一次推理时可能会有几分钟的调优延迟，但后续所有推理都能享受最优配置**。

对于离线调优，`gradlib` 工具提供了完整的工作流：先用 `AITER_TUNE_GEMM=1` 捕获形状，再用 `gemm_tuner.py` 批量调优，最后将结果 CSV 放回 `aiter/configs/` 目录。

## 五、量化生态：从 FP4 到 INT8 的全精度覆盖

### 5.1 量化类型的精细分级

AITER 的 MoE 支持的量化粒度从粗到细排列：

| 量化类型 | 粒度 | 典型场景 |
| --- | --- | --- |
| `per_Tensor` | 整个张量一个 scale | 训练后量化 |
| `per_Token` | 每行一个 scale | FP8/INT8 推理 |
| `per_1x128` | 每 128 列一个 scale | FP8 Block Scale |
| `per_1x32` | 每 32 列一个 scale | MXFP4 微缩浮点 |

这套体系在 `fused_moe_1stage_dict` 中以查找表形式固化，每种 `(activation, quant_type, dtype, q_dtype_a, q_dtype_w, isG1U1)` 组合映射到一个专用 kernel：

```
# 来源：aiter/fused_moe.pyfused_moe_1stage_dict = {    "gfx942": {        (ActivationType.Silu, QuantType.No,       dtypes.bf16, dtypes.bf16, dtypes.bf16, False, False): aiter.fmoe,        (ActivationType.Silu, QuantType.per_1x128, dtypes.bf16, dtypes.fp8,  dtypes.fp8,  True,  False): aiter.fmoe_g1u1,        # ... 共 11 种组合    },    "gfx950": {        # ... 7 种组合，包括 gfx950 特有的 fp4x2 和 blockscale    }}
```

### 5.2 MXFP4 量化的融合排序

对于 MXFP4（per_1x32）量化，AITER 实现了一个特别巧妙的优化——**将动态量化和 Token 排序融合为一个 kernel**：

```
# 来源：aiter/fused_moe.py（fused_moe_2stages 内部）a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(    hidden_states,    sorted_ids=sorted_ids,    num_valid_ids=num_valid_ids,    token_num=token_num,    topk=topk,    block_size=block_size_M,    num_rows=num_local_tokens,)
```

这避免了"先量化、再排序"带来的两次全局内存读写，对显存带宽受限的场景意义重大。

## 六、有趣的地方

### 6.1 LRU 缓存的大量使用

AITER 在性能关键路径上大量使用 `@functools.lru_cache`，包括 `get_inter_dim`、`get_block_size_M`、`get_2stage_cfgs`、`get_GEMM_A16W16_config` 等。由于推理阶段的矩阵形状是有限且重复的，这些缓存几乎能保证每种形状只做一次决策计算，后续全部走 O(1) 查表。

### 6.2 FlyDSL 回退机制

当调优配置指定了 FlyDSL kernel 但运行环境中未安装 FlyDSL 时，AITER 不会直接报错，而是查找 CSV 中标记为 `flydsl_fallback` 的备选配置：

```
# 来源：aiter/fused_moe.pyif cfg is not None and not is_flydsl_available():    kn1 = str(cfg.get("kernelName1", ""))    if kn1.startswith("flydsl_"):        fallback = fallback_cfgs.get(keys)        if fallback is not None:            cfg = fallback
```

这种**优雅降级** 设计让 AITER 在不同部署环境中都能稳定工作。

### 6.3 ctypes FFI：绕过 PyTorch 的无 Torch 调用路径

对于某些对延迟极度敏感的算子（如 ASM GEMM），AITER 提供了 `ffi_type="ctypes"` 选项，通过 ctypes 直接调用 C ABI 的 .so，完全绕过 PyTorch 的 pybind11 开销。`_ctypes_call` 函数会根据 Python 类型标注自动生成 ctypes 参数映射，并在每次调用时将 `torch.Tensor` 转换为轻量的 `aiter_tensor_t` 结构体。

## 结语

> AITER 展示了一个高性能 AI 算子库应有的样子：它不追求单点极致，而是构建了一套**配置驱动、多后端协同、自动调优** 的系统化工程。

在大模型推理日益成为产业刚需的今天，这种"让每块 GPU 都跑在最优点"的基础设施能力，正是 AMD ROCm 生态最需要的底层支撑。对于从事 GPU 算子开发或大模型推理优化的工程师而言，AITER 的 JIT 编译框架设计、MoE 多阶段调度策略和自动调优闭环，都是极具参考价值的工程实践。

参考资料

[1] ATOM/benchmark-dashboard: *https://rocm.github.io/ATOM/benchmark-dashboard/*

相关推荐

- [AMD GPU 设备数学库分层编译架构浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899885&idx=1&sn=c4a604edeff740435f04ea5b6f90bf7c&scene=21#wechat_redirect)
- [对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)
- [DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)
