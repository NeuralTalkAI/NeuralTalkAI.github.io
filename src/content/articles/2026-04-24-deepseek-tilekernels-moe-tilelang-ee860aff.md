关键词：**TileKernels**、TileLang、**MoE 路由**、:mdnice-bold-italic[低精度量化]、算子融合

> 在大模型训练与推理的工程实践中，算子性能往往是"最后一公里"的决定性因素。

DeepSeek 于 2026 年 4 月开源的 TileKernels 项目，用一种令人意想不到的方式回答了这个问题——:mdnice-bold-italic[不写一行 CUDA C++，仅凭 Python 领域专用语言 TileLang，就将 MoE 路由、多精度量化（FP8/FP4/E5M6）、SwiGLU 融合、Engram 门控、Manifold HyperConnection 等大模型关键路径上的算子，逼近甚至触及 GPU 的计算与带宽理论上限]。

![文章插图](/articles/2026-04-24-deepseek-tilekernels-moe-tilelang-ee860aff/images/01.webp)

- **deepseek-ai/TileKernels: A kernel library written in tilelang**
- https://github.com/deepseek-ai/TileKernels
- 4000 字，阅读 20 分钟，**播客 19 分钟**

相关推荐

- [DeepSeek Engram 里没有记忆，就像 MoE 里没有专家](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899832&idx=1&sn=30597017d95df44a62c0ad6f39f30588&scene=21#wechat_redirect)
- [突破 Agentic LLM 推理的存储 I/O 瓶颈！北大、清华联合 DeepSeek 提出 DualPath方案：双路径 KV 缓存加载](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897897&idx=1&sn=02c65e5a3b49b5dff327680df39ea39b&scene=21#wechat_redirect)
- [DeepSeek 针对HC训练不稳定、扩展性受限及内存开销问题的Manifold-Constrained Hyper-Connections解决方案](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895419&idx=1&sn=4bd4822fb5c1ed32937087a4cbde37bf&scene=21#wechat_redirect)

> 更关键的是，这些算子并非实验室玩具——它们已在 DeepSeek 内部的训练与推理流水线中实际部署。这不禁让人追问：当"用 Python 写 GPU 算子"不再是性能妥协，而是逼近极限的工程选择时，大模型基础设施的开发范式是否正在被重新书写？

细心的读者可能已经注意到，TileLang 官方组织 `tile-ai` 下也有一个类似的算子库 TileOPs[1]。它们是什么关系？

::: block-1
三者的关系链是：**TileLang（DSL 编译器）→ TileOPs（官方通用算子库）/ TileKernels（DeepSeek 专用算子库）**。
:::

- TileOPs 是 TileLang 团队自建的"官方示范库"，定位类似 PyTorch 生态中的 `torchvision`——提供 GEMM、elementwise 等**通用基础算子**，强调 Spec-driven 的规范化设计（每个算子都有 `ops_manifest.yaml` 声明签名、workload 和 roofline 公式），面向社区开发者和 AI Agent 的自动化构建场景。
- TileKernels 则完全不同。它是 DeepSeek 作为 TileLang 的**重度用户**，将自家模型中最核心、最性能敏感的算子实现后开源。这些算子带有鲜明的"DeepSeek 特色"——MoE 路由全链路、Engram 门控、Manifold HyperConnection、SwiGLU+FP8 融合量化——**全部是 DeepSeek 模型架构的独有组件**，实战优先，kernel 即接口，无中间抽象层。

| 项目 | **tile-ai/TileOPs** | **deepseek-ai/TileKernels** |
| --- | --- | --- |
| 定位 | 通用算子，Spec-driven | DeepSeek 模型专用，实战优先 |
| 算子类型 | GEMM、elementwise 等基础算子 | MoE 路由、Engram 门控、mHC、融合量化 |
| 设计理念 | 分 Op/Kernel 两层，manifest 声明式 | Kernel 即接口，已在生产部署 |

::: block-1
简言之：`tile-ai` 是造发动机的，`deepseek-ai` 是造整车的——TileKernels 相当于 DeepSeek 把自研的涡轮增压器开源了，反过来也验证了 TileLang 引擎的生产级能力。
:::

## 本文目录

- 快速上手
- 一、架构总览与设计哲学
  - 1.1 项目结构与模块划分
  - 1.2 核心设计思想：声明式 Tile 编程
- 二、MoE 路由算子：从 Top-K 选择到融合展开
  - 2.1 Top-K 门控：重复取最大值的巧妙实现
  - 2.2 融合展开：一次 kernel 搬运 Token 与 Scaling Factor
- 三、量化算子：逐 Token FP8 Casting 与 SwiGLU 融合
  - 3.1 量化基础设施：统一的 Config 抽象
  - 3.2 逐 Token 量化 Kernel 的分块策略
  - 3.3 SwiGLU 与量化的极致融合
- 四、Engram 门控：高度优化的异步流水线
  - 4.1 双 Pass 异步流水线
  - 4.2 跨 Pass 的缓冲区复用
  - 4.3 反向 Kernel 的极致工程
- 五、Manifold HyperConnection：GPU 上的 Sinkhorn 归一化
- 六、转置算子：Bank Conflict 消除的教科书示范
- 七、总结与展望

![文章插图](/articles/2026-04-24-deepseek-tilekernels-moe-tilelang-ee860aff/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> TileKernels 要求 Python ≥ 3.10、PyTorch ≥ 2.10、TileLang ≥ 0.1.9，以及 NVIDIA SM90/SM100 架构 GPU（如 H100/B200）与 CUDA Toolkit ≥ 13.1。

```
# 安装发布版pip install tile-kernels# 或安装开发版（含测试依赖）pip install -e ".[dev]"
```

安装完成后即可在 Python 中直接调用：

```
import torchfrom tile_kernels.moe import topk_gatefrom tile_kernels.quant import per_token_castfrom tile_kernels.transpose import transpose# MoE Top-K 门控选择scores = torch.randn(1024, 256, dtype=torch.float32, device='cuda')topk_idx = topk_gate(scores, num_topk=8)# 逐 Token FP8 量化x = torch.randn(1024, 4096, dtype=torch.bfloat16, device='cuda')out, out_sf = per_token_cast(x, fmt='e4m3', num_per_channels=128)# 高性能转置mat = torch.randn(2048, 4096, dtype=torch.bfloat16, device='cuda')mat_t = transpose(mat)
```

运行测试与 benchmark：

```
pytest tests/transpose/test_transpose.py -n 4                    # 正确性pytest tests/transpose/test_transpose.py --run-benchmark          # 正确性 + 性能
```

更多 MoE、量化、Engram 等模块使用细节，参考 `README.md`。

## 一、架构总览与设计哲学

### 1.1 项目结构与模块划分

TileKernels 的代码组织非常干净，核心代码全部位于 `tile_kernels/` 目录下：

```
tile_kernels/├── moe/        # MoE 路由：Top-K 门控、Token-Expert 映射、融合展开/归约├── quant/      # 多精度量化：FP8/FP4/E5M6 casting，含 SwiGLU 融合├── transpose/  # 高性能批量转置├── engram/     # Engram 门控：融合 RMSNorm 的前向/反向├── mhc/        # Manifold HyperConnection：Sinkhorn 归一化、Mix 分裂├── modeling/   # PyTorch autograd.Function 封装层├── torch/      # PyTorch 参考实现（用于正确性对照）└── testing/    # 测试与 benchmark 工具
```

### 1.2 核心设计思想：声明式 Tile 编程

> TileKernels 完全构建于 TileLang[2] 之上。

TileLang 是一套 Python 嵌入式 DSL，其核心抽象是 **Tile（数据块）** 和 **Fragment（寄存器片段）**。开发者用声明式语法描述数据在 shared memory、register file 之间的搬运与计算，由编译器完成线程映射、向量化、流水线插入等底层优化。

从全局配置文件可以看到这种"硬件感知"的设计理念：

```
# 来源：tile_kernels/config.py@functools.lru_cache(maxsize=None)def get_device_num_sms() -> int:    prop = torch.cuda.get_device_properties(torch.cuda.current_device())    return prop.multi_processor_countdef get_num_sms() -> int:    global _num_sms    if _num_sms == 0:        return get_device_num_sms()    return _num_sms
```

SM 数量被用来动态决定 persistent kernel 的 block 数量、共享内存预算分配等关键参数——这使得同一份代码能自动适配不同规格的 GPU。

## 二、MoE 路由算子：从 Top-K 选择到融合展开

> MoE（Mixture of Experts）是 DeepSeek 模型的核心架构组件。`tile_kernels/moe/` 包含了一套完整的路由算子链：**Top-K 门控 → 分组计数 → Token-Expert 映射 → 融合展开/归约 → 权重归一化**。

### 2.1 Top-K 门控：重复取最大值的巧妙实现

Top-K 门控的任务是从 `num_experts` 个专家中为每个 token 选出得分最高的 `num_topk` 个。TileKernels 采用了一种直觉上非常简洁的策略：**重复 K 次"取最大值 → 标记为负无穷"操作**。

```
# 来源：tile_kernels/moe/topk_gate_kernel.py@T.prim_funcdef topk_gate_kernel(    scores: T.Tensor[(num_tokens, num_experts), T.float32],    topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],):    with T.Kernel(num_tokens, threads=num_threads) as pid:        scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)        idx_reducer = T.alloc_reducer((1,), T.int32, 'min', replication='all')        # 加载分数，越界位置填负无穷        for i in T.Parallel(num_aligned_experts):            if i < num_experts:                scores_fragment[i] = scores[pid, i]            else:                scores_fragment[i] = -T.infinity(T.float32)        # 重复 K 次：找最大值 → 取最小索引（平局稳定）→ 置负无穷        for k in T.unroll(num_topk):            T.reduce_max(scores_fragment, amax_fragment)            T.fill(idx_reducer, T.max_value(T.int32))            for i in T.Parallel(num_aligned_experts):                if scores_fragment[i] == amax_fragment[0]:                    idx_reducer[0] = T.min(idx_reducer[0], idx_fragment[i])            T.finalize_reducer(idx_reducer)            topk_idx_shared[k] = idx_reducer[0]            # 已选中的专家置为负无穷            for i in T.Parallel(num_aligned_experts):                if idx_fragment[i] == idx_reducer[0]:                    scores_fragment[i] = -T.infinity(T.float32)
```

这里有两个精妙之处：

- 第一，`T.unroll(num_topk)` 将循环完全展开，消除了分支预测开销；
- 第二，当出现分数相同的"平局"时，使用 `T.alloc_reducer('min')` 确保总是选择**索引最小的专家**，保证结果稳定可复现。

整个算子仅使用一个 warp（32 线程），对应 `threads=32`，在专家数不超过几百时能完美装入寄存器文件。

### 2.2 融合展开：一次 kernel 搬运 Token 与 Scaling Factor

选出 Top-K 专家后，需要将每个 token 的激活值按路由结果"展开"到对应专家的槽位中。`expand_to_fused_kernel` 实现了这一操作，并支持同时搬运量化后的 scaling factor：

```
# 来源：tile_kernels/moe/expand_to_fused_kernel.pyfor k in T.serial(num_topk):    T.assume(pos_local[k] < num_expanded_tokens)    if pos_local[k] >= 0:        for i in T.Parallel(hidden_aligned):            expanded_x[pos_local[k], i] = x_fragment[i]        if num_per_channels is not None:            for i in T.Parallel(hidden_sf_aligned):                if use_tma_aligned_col_major_sf:                    expanded_x_sf[i, pos_local[k]] = x_sf_fragment[i]                else:                    expanded_x_sf[pos_local[k], i] = x_sf_fragment[i]
```

`T.assume()` 是 TileLang 提供的编译器提示——它告诉后端"这个条件恒成立"，从而让编译器消除边界检查。同时，代码通过 `T.Kernel(T.max(num_tokens, num_expanded_tokens))` 将"填零无效位"和"复制有效数据"合并到同一批 block 中，避免了额外的 kernel launch。

## 三、量化算子：逐 Token FP8 Casting 与 SwiGLU 融合

### 3.1 量化基础设施：统一的 Config 抽象

量化模块的设计核心是 `CastInputConfig` 和 `CastOutputConfig` 两个 dataclass，它们统一描述了输入/输出的数据类型、scaling block 尺寸、是否使用 TMA 对齐的列主序 SF、是否使用 packed UE8M0 格式等所有变体：

```
# 来源：tile_kernels/quant/common.py@dataclass(frozen=True)class CastOutputConfig(BaseCastConfig):    round_sf: bool = False    custom_clamp_min_value: Optional[float] = None    @property    def clamp_min_value(self) -> float:        if self.custom_clamp_min_value isnotNone:            return self.custom_clamp_min_value        elif self.dtype == T.float8_e4m3fn:            return1e-4        elif self.dtype == T.float4_e2m1fn:            return T.max_value(self.dtype) * (2**-126)
```

`clamp_min_value` 确保 scaling factor 不会过小导致量化值溢出。对于 FP4（E2M1），这个下界被精确设置为 `max_value * 2^(-126)`，恰好是 FP32 denorm 的边界，体现了对浮点表示的深入理解。

### 3.2 逐 Token 量化 Kernel 的分块策略

`per_token_cast_kernel` 是量化模块中最核心的算子。它将一个 `[num_tokens, hidden]` 的 BF16/FP32 矩阵就地转为 FP8/FP4，同时产出 per-group 的 scaling factor。

```
# 来源：tile_kernels/quant/per_token_cast_kernel.pywith T.Kernel(T.ceildiv(num_tokens, block_m), T.ceildiv(hidden, block_k),              threads=num_threads) as (pid_token, pid_hidden):    x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)    T.annotate_layout({        x_fragment: T.Fragment(            (block_m, block_k),            forward_fn=x_layout_fn,        )    })    # 1. 加载数据到寄存器    T.copy(x[pid_token * block_m, pid_hidden * block_k], x_fragment, disable_tma=True)    # 2. Reduce 求 absmax    amax_fragment = T.alloc_fragment((block_m, num_groups), in_config.dtype)    x_fragment_reshaped = T.reshape(x_fragment, [block_m, num_groups, num_per_channels])    T.reduce_absmax(x_fragment_reshaped, amax_fragment, dim=2)    # 3. 计算 SF 并存储    for i, j in T.Parallel(block_m, num_groups):        sf, sf_inv = get_sf_and_inv(amax, out_config)        store_sf(out_sf, sf, m_idx, k_idx, out_config)        sf_inv_fragment[i, j] = sf_inv    # 4. 乘以 SF 逆并写出    for i, j in T.Parallel(block_m, block_k):        out_shared[i, j] = x_fragment[i, j] * sf_inv_fragment[i, j // num_per_channels]
```

这里的 `T.annotate_layout` 自定义了寄存器文件中 fragment 到线程的映射方式 `x_layout_fn`，目的是让 128 个线程中每个线程负责连续 32 个元素的向量化加载。`T.reshape` 在编译期将 fragment 重新解释为三维视图，使得 `T.reduce_absmax(dim=2)` 能直接按 `num_per_channels` 粒度归约——这一切都在寄存器内完成，零额外内存访问。

### 3.3 SwiGLU 与量化的极致融合

大模型 FFN 层常用 SwiGLU 激活函数。朴素实现需要三次 kernel launch（SwiGLU → 求 SF → 量化），每次都要读写一遍全量数据。TileKernels 将这三步融合进一个 kernel：

```
# 来源：tile_kernels/quant/swiglu_forward_and_per_token_cast_kernel.py# SwiGLU + clamp + 可选权重乘法val_l = T.float32(xl_fragment[i, j])val_r = T.float32(xr_fragment[i, j])if use_clamp:    val_l = T.min(val_l, swiglu_clamp_value)    val_r = T.max(T.min(val_r, swiglu_clamp_value), -swiglu_clamp_value)if with_weight:    val = val_l / (1 + T.exp(-val_l)) * val_r * topk_weights_fragment[i]else:    val = val_l / (1 + T.exp(-val_l)) * val_r# 紧接着就地做 per-group absmax → SF → 量化写出T.reduce_absmax(x_fragment_reshaped, sf_inv_fragment, dim=2)
```

注意 `val_l / (1 + T.exp(-val_l))` 正是 SiLU（Swish）的数学定义。整个 SwiGLU 运算、可选的 clamp 计数、Top-K 权重乘法、absmax 归约、SF 计算和 FP8 cast 全部在同一组寄存器中流水完成。对于需要统计 clamp 次数的训练场景，还通过 `T.alloc_reducer('sum')` 和 `T.atomic_add` 实现了跨 block 的计数聚合，使用 persistent kernel 策略避免了额外 launch。

## 四、Engram 门控：高度优化的异步流水线

> Engram 门控是 TileKernels 中工程复杂度最高的算子，其前向 kernel `engram_gate_fwd_kernel` 展示了教科书级别的 GPU 流水线设计。

### 4.1 双 Pass 异步流水线

前向过程分为两个 pass：**Pass 1 计算 gate score（归约密集型）** 和 **Pass 2 输出 `x + gate * v`（访存密集型）**。它们通过 **cp.async 异步拷贝** 实现流水重叠：

```
# 来源：tile_kernels/engram/engram_gate_kernel.py# Pass 1: 通过 cp.async pipeline 双缓冲加载 x 和 kfor i_b in T.Serial(1, num_blk):    phase = i_b % 2    T.async_copy(hidden_states[i_s, pid_h, i_b * blk_d:(i_b+1) * blk_d],                 x_smem[i_b * blk_d:(i_b+1) * blk_d])    T.async_copy(k[i_s, pid_h, i_b * blk_d:(i_b+1) * blk_d],                 kv_smem[phase, :])    T.ptx_wait_group(2)  # 等待最多 2 个 async group 完成    # 计算 rstd_x, rstd_k, gate_score（点积 + 加权）    for i_k in T.serial(vec_size):        rstd_x_local[0] += x_local[i_k] * x_local[i_k]        rstd_k_local[0] += k_local[i_k] * k_local[i_k]        gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]
```

关键细节在于 `T.ptx_wait_group(2)` 的使用——它直接映射到 PTX 的 `cp.async.wait_group` 指令，允许计算流同时保持最多 2 个异步拷贝 in-flight，从而实现数据搬运与乘加运算的完全重叠。

### 4.2 跨 Pass 的缓冲区复用

在 Pass 1 的尾声和 Pass 2 的开头，代码复用了 `kv_smem` 来预取 `v` 向量：

```
# Pass 1 末尾：复用释放的 kv_smem bank 预取 v[0]T.async_copy(v[i_s, 0:blk_d], kv_smem[v_start_phase, :])# Pass 2: 利用 x_smem（仍有效）和 kv_smem（已装 v）写输出for i_k in T.vectorized(vec_size):    output[i_s, pid_h, sub_base + thread_idx * vec_size + i_k] = \        x_local[i_k] + gate_score_reducer[0] * v_local[i_k]
```

这种"零额外共享内存"的缓冲区交替策略，将 shared memory 占用压缩到最小，从而让每个 SM 能同时驻留更多 block，提升占用率。

### 4.3 反向 Kernel 的极致工程

反向 kernel `engram_gate_bwd_kernel` 使用 8 个 warp（256 线程），每 2 个 warp 协作处理一个 head（`hc_mult=4`）。最引人注目的是 **grad_w 的寄存器累加策略**：每个 warp pair 在寄存器中维护完整的 `hidden_size / threads_per_head` 个 grad_w 元素，跨所有 token 累加后才一次性写回全局内存，将写带宽需求降至最低。

## 五、Manifold HyperConnection：GPU 上的 Sinkhorn 归一化

`tile_kernels/mhc/sinkhorn_kernel.py` 实现了 Sinkhorn 归一化的前向和反向传播。Sinkhorn 归一化本质是交替做行归一化和列归一化，可理解为"让一个矩阵同时成为双随机矩阵"。

```
# 来源：tile_kernels/mhc/sinkhorn_kernel.py# 初始 softmax + epsT.reduce_max(comb_frag, row_max, dim=2)for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):    comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])T.reduce_sum(comb_frag, row_sum, dim=2)for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):    comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps# 交替归一化 repeat 次for _ in T.serial(repeat - 1):    T.reduce_sum(comb_frag, row_sum, dim=2)  # 行归一化    ...    T.reduce_sum(comb_frag, col_sum, dim=1)  # 列归一化    ...
```

反向 kernel 的亮点在于：它将前向过程中每一步的中间结果存储在 shared memory 的 `xs` 和 `sums` 数组，然后以逆序遍历这些"检查点"计算梯度。这种**全寄存器 + shared memory 的重计算策略** 避免了将中间状态写回 HBM，在 `hidden_size` 较小（如 4~8）时将整个前向 + 反向完全保持在片上。

## 六、转置算子：Bank Conflict 消除的教科书示范

看似简单的矩阵转置，要写到极致也大有学问：

```
# 来源：tile_kernels/transpose/batched_transpose_kernel.py# 读入寄存器并转置for j in T.unroll(block_k):    for k in T.vectorized(block_k):        tmp_row[k] = x[pid_batch, pid_x * block_x + i * block_k + j,                       pid_y * block_y + col * block_k + k]    for k in T.unroll(block_k):        tmp[k, j] = tmp_row[k]# 写入 shared memory，附加 swizzle 消除 bank conflictfor j in T.unroll(block_k):    swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k    for k in T.vectorized(block_k):        out_shared[col * block_k + swizzle_j, i * block_k + k] = tmp[swizzle_j, k]
```

`swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k` 是经典的 **XOR swizzle** 变体——通过将线程 ID 混入列索引，让同一 warp 内的不同线程访问不同 bank，将 bank conflict 降为零。同时，shared memory 被声明为 `(block_y, block_x + block_k)` 而非 `(block_y, block_x)`，额外的 `block_k` 列 padding 进一步消除了跨行的 bank 冲突。

## 七、总结与展望

> TileKernels 的核心贡献可以归纳为三点：

1. **用 Python DSL 达到手写 CUDA 级别的性能**。通过 TileLang 的 `T.alloc_fragment`、`T.async_copy`、`T.reduce_absmax` 等高级原语，开发者可以精确控制数据在 register → shared memory → global memory 三级存储间的搬运方式，同时将线程映射、向量化等机械工作交给编译器。
2. **算子融合的系统性实践**。从 SwiGLU + 量化融合，到 Engram 门控中 RMSNorm + 点积 + Sigmoid 的单 kernel 完成，:mdnice-bold-italic[TileKernels 展示了如何通过"把更多计算塞进一个 kernel"来减少 HBM 访问次数，榨取 GPU 的 compute/memory 比。]
3. **生产级的工程质量**。SM 数量自适应、persistent kernel、`T.assume` 编译器提示、`TK_PRINT_KERNEL_SOURCE` 调试开关、完整的 pytest benchmark 插件——:mdnice-bold-italic[这些细节表明 TileKernels 不是学术 demo，而是经过生产磨炼的基础设施。]

项目地址：https://github.com/deepseek-ai/TileKernels[3]，MIT 协议开源。对于正在打造大模型训练/推理系统的团队而言，这套代码库既是可直接使用的高性能算子集合，也是学习 TileLang 编程范式与 GPU 算子优化技巧的绝佳教材。

参考资料

[1] TileOPs: *https://github.com/tile-ai/TileOPs*

[2] TileLang: *https://github.com/tile-ai/tilelang*

[3] deepseek-ai/TileKernels: *https://github.com/deepseek-ai/TileKernels*
