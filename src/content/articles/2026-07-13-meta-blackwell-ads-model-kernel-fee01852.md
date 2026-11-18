关键词：**推荐系统**、GPU 内核、**GDPA**、:mdnice-bold-italic[CuTe DSL、Triton TLX]

> 大多数人谈注意力优化，脑海里浮现的是大语言模型里的长序列、FlashAttention、显存占用和吞吐率。

但在广告与推荐模型里，问题常常更别扭：序列长度参差不齐，头维配置不统一，MHA、GQA、MQA 混在一起，注意力里的“softmax”还可能被 GELU、ReLU、SiLU 这类激活替代。**标准算子能跑，却不一定跑得好；模板内核能覆盖，却未必贴合生产流量。**

![文章插图](/articles/2026-07-13-meta-blackwell-ads-model-kernel-fee01852/images/01.webp)

- High-performance GPU kernels for Ads and Recsys model training, independently implemented and optimized for real-world workloads and model-specific input characteristics.
- https://github.com/facebookresearch/ads_model_kernel_library

::: related-title
**相关推荐**
:::

::: related-links
- **[消除推荐系统广播税，从内存带宽到计算瓶颈的蜕变，揭秘 621 TFLOPS 的 Flash Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900976&idx=1&sn=0f29d856ded3ed90bc6c1c4ce3152d0f&scene=21#wechat_redirect)**
- **[FlashNormAttention 融合注意力与归一化到 GEMM，最高可掩盖 90%归一化时延](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903614&idx=1&sn=d4c7a5e4345d25b1925979e0793d531a&scene=21#wechat_redirect)**
- **[适配 Ada 中端卡的 MegaKernel 在线广告方案 Ada-MK：共享内存直接减半，端到延迟最高降 50%，击穿小批量毫秒级推理瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902893&idx=1&sn=5f53dd32ac726cea553f1c3743d8d2a1&scene=21#wechat_redirect)**

:::

> `facebookresearch/ads_model_kernel_library`的价值就在这里：它把真实推荐训练负载里的碎片化、变长、稀疏、融合需求，一层层压进 Blackwell/Hopper GPU 的内核调度、张量内存、在线归一化和自定义激活路径中。

![文章插图](/articles/2026-07-13-meta-blackwell-ads-model-kernel-fee01852/images/02.webp)

::: image-caption
这张表格列出4个适配NVIDIA Blackwell（SM100）架构的GPU内核项目：GDPA是广义点积注意力内核；TLX Block Attention是Triton TLX块注意力内核；TLX Multi-CTA Norm Fusion是融合矩阵乘与归一化的TLX内核；TLX GDPA Megakernel是TLX版广义点积注意力超大内核，并标注了各自存放路径，均面向Blackwell SM100硬件做深度优化，用于大模型AI计算加速。
:::

## 本文目录

::: related-links
- 一、快速上手：先跑起来，再看内核为什么快
- 二、项目全景：四个内核项目围绕“真实推荐负载”展开
- 三、GDPA 的核心抽象：Attention 不再等于 Softmax Attention
- 四、前向执行路径：从 PyTorch 张量到 CuTe DSL JIT 内核
- 五、变长序列与 tile 调度：推荐系统真正难的地方
- 六、在线 Softmax 与自定义激活：数值稳定性和表达能力的折中
- 七、TLX 旁支：Block Attention、Norm Fusion 与 Megakernel
- 八、为什么它重要：性能来自“贴着负载写内核”

:::

![文章插图](/articles/2026-07-13-meta-blackwell-ads-model-kernel-fee01852/images/03.webp)

## 一、快速上手：先跑起来，再看内核为什么快

> 这个仓库并不是一个单一 Python 包，是一个公开内核集合。根目录 README 将项目分成四块：`gdpa/`、`block_attention/`、`multi_cta_norm_fusion/`、`gdpa_megakernel/`。其中最值得优先阅读和试跑的是`gdpa/`，也就是 Generalized Dot Product Attention。

仓库声明的基础要求包括：Python 3.10+、PyTorch 2.0+、CUDA 12.0+，并要求 NVIDIA Hopper（SM90）或 Blackwell（SM100）GPU；CuTe DSL 路径依赖`nvidia-cutlass-dsl`，TLX 相关项目依赖`fbtriton==3.6.1`。

最精简的安装方式可以这样开始：

```
git clone https://github.com/facebookresearch/ads_model_kernel_library.gitcd ads_model_kernel_library# GDPA / CuTe DSL 路径pip install "nvidia-cutlass-dsl>=4.1.0" torch einops# TLX 项目路径pip install fbtriton==3.6.1
```

如果要跑 GDPA OSS benchmark，README 给出的依赖更接近真实基准环境：

```
conda create -n gdpa-bench python=3.12 -yconda activate gdpa-benchpip install -r gdpa/scripts/requirements.txtpython gdpa/scripts/benchmark.py bench-fwd
```

这里需要注意两点。

- 第一，GDPA 的完整 benchmark 面向 Blackwell GPU，README 中明确提到 B200/B100/GB200 与 CUDA 12.8+。
- 第二，部分对比基线涉及 FlashAttention v4、CUTLASS Blackwell FMHA 和带`tl.async_task`能力的 Triton TLX，因此如果只是阅读或验证核心接口，可以先从`gdpa/src/interface.py`入手；如果要复现实验，则参考`gdpa/README.md`里的“Cross-kernel benchmark”步骤。

TLX 两个子项目也提供了独立环境文件。例如：

```
cd gdpa_megakernelconda env create -f environment.ymlconda activate gdpa-megakernelTRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 ADS_MKL_DISABLE_AUTOTUNE=1 \  python -m unittest discover -s tests -p "test_*.py"
```

`multi_cta_norm_fusion/`类似：

```
cd multi_cta_norm_fusionconda env create -f environment.ymlconda activate multi-cta-norm-fusionpython -m unittest discover -s tests -p "test_*.py"
```

如果只是想快速知道某个模块如何被调用，README 里给出了最短入口：

```
from gdpa.src.interface import flash_attn_func, flash_attn_varlen_funcout, lse = flash_attn_func(q, k, v, softmax_scale=scale, causal=False)out, lse = flash_attn_varlen_func(    q, k, v,    cu_seqlens_q=cu_q,    cu_seqlens_k=cu_k,    max_seqlen_q=max_q,    max_seqlen_k=max_k,    activation="fast_gelu",)
```

更完整的 benchmark 依赖、FlashAttention v4 安装和 Triton TLX 补丁流程，可以参考项目文档：gdpa/README.md。

## 二、项目全景：四个内核项目围绕“真实推荐负载”展开

> 从根目录看，这个仓库很克制：

```
ads_model_kernel_library/  gdpa/                    CuTe DSL与Triton版Generalized Dot Product Attention  block_attention/          Triton TLX block attention，可变长块注意力  multi_cta_norm_fusion/    TLX fused matmul + RMSNorm / LayerNorm  gdpa_megakernel/          TLX版GDPA megakernel  README.md  LICENSE
```

它的语言组成几乎全是 Python，但这不意味着它是普通 Python 项目。这里的 Python 更像“GPU 内核生成语言”：

- 一部分通过 NVIDIA CuTe DSL 描述 Blackwell/Hopper 上的矩阵搬运、MMA、同步和张量内存；
- 另一部分通过 Triton TLX 使用`tl.async_task`、cluster CTA、remote shared memory 等机制，表达更底层的并行流水线。

> 根 README 中列出的四个项目分别对应四类工作：

| 项目 | 作用 | 主要路径 | 目标架构 |
| --- | --- | --- | --- |
| GDPA | Generalized Dot Product Attention | `gdpa/` | Blackwell 为主，Triton 路径覆盖 SM80/SM90/SM100 |
| TLX Block Attention | 变长 block attention | `block_attention/` | Blackwell |
| TLX Multi-CTA Norm Fusion | matmul 与 RMSNorm/LayerNorm 融合 | `multi_cta_norm_fusion/` | Blackwell |
| TLX GDPA Megakernel | TLX 版 GDPA 大内核 | `gdpa_megakernel/` | Blackwell |

最核心的是`gdpa/`。README 对 GDPA 的定义很明确：它是标准 dot-product attention 的一种推广，关键变化是把 softmax 替换为不同激活函数。这句话看似简单，实际会牵动整个内核设计。因为 FlashAttention 原本依赖 softmax 的在线归一化性质：每个 Q 块扫描多个 K/V 块时，只要维护 row-wise max 和 sum，就能稳定地计算概率矩阵而不物化完整注意力矩阵。一旦“归一化函数”变成 GELU、ReLU²、FastSiLU 等，内核不能只照搬 SDPA 模板，而要重新设计前向、反向、激活、梯度、量化和调度。

::: block-1
这也是仓库名字中“Ads Model Kernel”的含义：广告推荐系统里的模型结构、输入分布和生产约束，反过来塑造了内核。
:::

## 三、GDPA 的核心抽象：Attention 不再等于 Softmax Attention

> 标准 Scaled Dot Product Attention 大致是：

```
S = QKᵀ * scaleP = softmax(S)O = PV
```

GDPA 把中间的`softmax`推广为更多激活或变换：

```
S = QKᵀ * scaleP = activation(S)O = PV
```

- 如果`activation=softmax`，它退化为传统注意力；
- 如果`activation=fast_gelu`、`relu`、`identity`，它就成了更贴合推荐模型实验需求的广义点积注意力。

> 代码中的公共入口在`gdpa/src/interface.py`，固定长度和变长接口分别是`flash_attn_func`与`flash_attn_varlen_func`。

```
// 来源：gdpa/src/interface.pydef flash_attn_func(    q: torch.Tensor,    k: torch.Tensor,    v: torch.Tensor,    softmax_scale: Optional[float] = None,    causal: bool = False,    window_size: Tuple[Optional[int], Optional[int]] = (None, None),    learnable_sink: Optional[torch.Tensor] = None,    softcap: float = 0.0,    pack_gqa: Optional[bool] = None,    prefer_persistent: bool = False,):    return FlashAttnFunc.apply(        q, k, v, softmax_scale, causal, window_size,        learnable_sink, softcap, pack_gqa, prefer_persistent,    )
```

变长接口更能体现推荐场景：

```
// 来源：gdpa/src/interface.pydef flash_attn_varlen_func(    q: torch.Tensor,    k: torch.Tensor,    v: torch.Tensor,    cu_seqlens_q: Optional[torch.Tensor] = None,    cu_seqlens_k: Optional[torch.Tensor] = None,    max_seqlen_q: Optional[int] = None,    max_seqlen_k: Optional[int] = None,    causal: bool = False,    window_size: Tuple[Optional[int], Optional[int]] = (None, None),    activation: str = "fast_gelu",    tile_to_batch_q: Optional[torch.Tensor] = None,    tile_to_head_q: Optional[torch.Tensor] = None,    tile_to_block_q: Optional[torch.Tensor] = None,    sfq: Optional[torch.Tensor] = None,    sfk: Optional[torch.Tensor] = None,    sfv: Optional[torch.Tensor] = None,):    return cutedsl_generalized_dot_product_attention(        q, k, v,        cu_seqlens_q, cu_seqlens_k,        max_seqlen_q=max_seqlen_q,        max_seqlen_k=max_seqlen_k,        causal=causal,        window_size_left=window_size[0],        window_size_right=window_size[1],        activation=activation,        tile_to_batch_q=tile_to_batch_q,        tile_to_head_q=tile_to_head_q,        tile_to_block_q=tile_to_block_q,        sfq=sfq, sfk=sfk, sfv=sfv,    )
```

这里有几个关键点：

- 第一，输入布局同时支持普通 batched 张量和 jagged/varlen 张量。后者通过`cu_seqlens_q`、`cu_seqlens_k`描述每个样本在拼接大张量中的边界。
- 第二，`activation`是显式参数，默认是`fast_gelu`，这和传统 FlashAttention 默认 softmax 的心智模型不同。
- 第三，接口里预留了`tile_to_batch_q/head/block`这类调度映射。这说明变长负载不只是“传入 offset 就行”，还要把 tile 怎样分配给 CTA、怎样在 SM 之间均衡，前置成内核可消费的数据结构。
- 第四，`sfq/sfk/sfv`和`cu_seqlens_sf_q/k`说明它已经把 MXFP8 block scaling 纳入接口设计，而不是事后加一层量化包装。

## 四、前向执行路径：从 PyTorch 张量到 CuTe DSL JIT 内核

> `_flash_attn_fwd`是整个 CuTe DSL 路径的主干。它做的事情不是单纯调用一个 CUDA 函数，而是完成一系列“运行时配置到编译期特化”的转换：

1. 检查输入是否连续、dtype 是否合法、head 维是否对齐；
2. 推导 batch、seqlen、total token、page table 等维度；
3. 处理 causal 与 sliding window 语义；
4. 根据 GPU compute capability 调整 block size 和 GQA packing；
5. 将 PyTorch Tensor 通过 DLPack 转换成 CuTe Tensor；
6. 构造`compile_key`，按形状、dtype、mask、激活、blockscaled 等配置缓存 JIT 结果；
7. 首次调用时`cute.compile`，之后直接复用已编译内核。

```
// 来源：gdpa/src/interface.pyq, k, v = [maybe_contiguous(t) for t in (q, k, v)]batch_size, seqlen_q, total_q, seqlen_k, num_pages, page_size = (    _extract_sequence_dimensions(q, k, cu_seqlens_q, cu_seqlens_k, page_table))num_head, head_dim = q.shape[-2:]num_head_kv = k.shape[-2]head_dim_v = v.shape[-1]_validate_flash_attn_inputs(    q, k, v,    batch_size, seqlen_k,    num_head, num_head_kv,    head_dim, head_dim_v,    num_pages, page_size,    cu_seqlens_q, cu_seqlens_k,    seqused_q, seqused_k,    page_table, learnable_sink,)if softmax_scale isNone:    softmax_scale = 1.0 / math.sqrt(head_dim)qhead_per_kvhead = num_head // num_head_kvif pack_gqa isNone:    pack_gqa = qhead_per_kvhead > 1
```

这段代码体现了一个工程事实：高性能内核不仅仅要“算得快”这么简单，而是要在入口处把所有会影响内核形态的参数整理干净。比如`num_head % num_head_kv == 0`不仅是数学约束，也是 GQA/MQA 映射能否成立的前提；`head_dim % alignment == 0`直接决定向量化 load/store 能不能安全使用。

之后进入编译缓存：

```
// 来源：gdpa/src/interface.pycompile_key = (    dtype,    head_dim,    head_dim_v,    qhead_per_kvhead,    causal,    softcap isnotNone,    lse isNone,    cu_seqlens_q isNone,    cu_seqlens_k isNone,    seqused_q isNone,    seqused_k isNone,    page_table isnotNone,    window_size_left isnotNone,    window_size_right isnotNone,    learnable_sink isnotNone,    m_block_size,    n_block_size,    num_threads,    pack_gqa,    compute_capability,    activation,    blockscaled,    sf_vec_size,    cu_seqlens_sf_q isnotNone,    cu_seqlens_sf_k isnotNone,    prefer_persistent,)
```

`compile_key`越长，越说明内核越“特化”。这里每个布尔值都可能影响生成代码：是否有变长、是否有 paged KV、是否有 local mask、是否需要 LSE、是否 blockscaled、是否启用 persistent scheduler。

换句话说，GDPA 不是用一个巨大通用 kernel 硬扛所有情况，而是借 CuTe DSL 在运行时 JIT 出适配当前负载的内核。

## 五、变长序列与 tile 调度：推荐系统真正难的地方

> 推荐模型的输入常常是 jagged tensor：每个样本的历史行为长度不同，广告候选数量不同，交叉特征也可能不同。如果把所有样本 padding 到最大长度，GPU 看起来更规整，但会浪费大量计算。GDPA 选择直接面对变长。

最底层的序列信息抽象在`SeqlenInfoQK`中：

```
// 来源：gdpa/src/seqlen_info.pyclass SeqlenInfoQK:    def __init__(        self,        batch_idx: cutlass.Int32,        seqlen_q_static: cutlass.Int32,        seqlen_k_static: cutlass.Int32,        mCuSeqlensQ: cute.Tensor | None = None,        mCuSeqlensK: cute.Tensor | None = None,        mSeqUsedQ: cute.Tensor | None = None,        mSeqUsedK: cute.Tensor | None = None,        mCuSeqlensSFQ: cute.Tensor | None = None,        mCuSeqlensSFK: cute.Tensor | None = None,    ) -> None:        self.offset_q = (            0if cutlass.const_expr(mCuSeqlensQ isNone)            else mCuSeqlensQ[batch_idx]        )        self.offset_k = (            0if cutlass.const_expr(mCuSeqlensK isNone)            else mCuSeqlensK[batch_idx]        )        if cutlass.const_expr(mSeqUsedQ isnotNone):            self.seqlen_q = mSeqUsedQ[batch_idx]        else:            self.seqlen_q = (                seqlen_q_static                if cutlass.const_expr(mCuSeqlensQ isNone)                else mCuSeqlensQ[batch_idx + 1] - self.offset_q            )
```

这段代码的价值在于：每个 tile 一开始只读一次全局内存中的 offset 和 length，后面所有 mask、block 范围、SF offset 都复用这些信息。对 GPU 内核来说，少一次无规律 global memory 读取，可能就是一次流水线停顿的减少。

有了长度，下一步是判断当前 Q 块应该扫描哪些 K/V 块。`BlockInfo.get_n_block_min_max`负责处理 causal、sliding window 以及 GQA packing：

```
// 来源：gdpa/src/block_info.py@cute.jitdef get_n_block_min_max(    self, seqlen_info: SeqlenInfoQK, m_block: Int32) -> Tuple[Int32, Int32]:    n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)    if const_expr(        self.is_causal or (self.is_local and self.window_size_right isnotNone)    ):        m_idx_max = (m_block + 1) * self.tile_m        if const_expr(self.qhead_per_kvhead_packgqa > 1):            m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q        n_idx_right = (            n_idx if const_expr(self.is_causal)            else n_idx + self.window_size_right        )        n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))    n_block_min = 0    if const_expr(self.is_local and self.window_size_left isnotNone):        m_idx_min = m_block * self.tile_m        n_idx_left = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q        n_idx_left = n_idx_left - self.window_size_left        n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)    return n_block_min, n_block_max
```

可以把它理解成“内核里的交通管制”：当前 Q tile 是一辆车，K/V tile 是一段段道路。causal mask 禁止它开到未来；sliding window 限制它只能在左右窗口内行驶；GQA packing 会改变 Q 头与 KV 头的对应比例。这个函数的目标是尽早排除完全被 mask 掉的 tile，避免把 Tensor Core 时间浪费在必然无效的矩阵块上。

更难的是调度。固定长度时，`SingleTileScheduler`把 grid 映射成`(block, head, batch)`即可；变长时，如果某些样本很长、某些样本很短，简单三维 grid 会导致 SM 之间工作量严重不均。代码里提供了多种 scheduler：`StaticPersistentTileScheduler`、`SingleTileLPTScheduler`、`SingleTileVarlenScheduler`、`SingleTileVarlenSchedulerPersistent`等。

其中`precompute_for_varline_load_balancing`很有代表性：它先估算每个 tile 的工作量，再按降序排序，并用 zigzag 方式分发到 SM，避免“重活”集中到一批 SM 上。

```
// 来源：gdpa/src/interface.pymain_seqlens = torch.diff(main_cu_seqlens)tiles_per_example = (main_seqlens + main_tile_size - 1) // main_tile_sizeworkload_seqlens = torch.diff(workload_cu_seqlens)workloads_per_example = (    workload_seqlens + workload_tile_size - 1) // workload_tile_size# Workload for each tileworkload_expanded = torch.repeat_interleave(    workloads_per_example, num_head * tiles_per_example)# Sort by workload in descending ordersorted_indices = torch.argsort(workload_expanded, descending=True)# Dispatch to SMs using zigzag patternnum_sm = sm_counttotal_tiles = len(sorted_indices)tile_indices = torch.arange(total_tiles)cycle = tile_indices // num_smpos_in_cycle = tile_indices % num_smzigzag_indices = (    torch.where(cycle % 2 == 0, pos_in_cycle, num_sm - 1 - pos_in_cycle)    + cycle * num_sm)sorted_indices = sorted_indices[zigzag_indices]tile_to_batch = batch_expanded[sorted_indices]tile_to_head = head_expanded[sorted_indices]tile_to_block = tile_local_indices[sorted_indices]
```

::: block-1
:mdnice-bold-italic[这段代码很好地说明了“推荐系统内核优化”的本质：不是单个 GEMM 越大越好，而是大量不规则小任务如何均匀喂给 GPU]。zigzag 调度像发扑克牌，先把大牌排序，再正反交替分给不同 SM，避免某个 SM 一直拿到最重的牌。
:::

## 六、在线 Softmax 与自定义激活：数值稳定性和表达能力的折中

> 虽然 GDPA 强调“Generalized”，但 softmax 路径依然是基准。`gdpa/src/softmax.py`实现了在线 softmax：扫描 K/V 块时不物化完整`S`矩阵，而是维护每一行的`row_max`和`row_sum`。这正是 FlashAttention 节省显存的核心。

```
// 来源：gdpa/src/softmax.py@cute.jitdef online_softmax(    self,    acc_S: cute.Tensor,    is_first: cutlass.Constexpr[bool] = False,    check_inf: cutlass.Constexpr[bool] = True,) -> cute.Tensor:    acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)    row_scale = cute.make_fragment_like(self.row_max, Float32)    for r in cutlass.range(cute.size(self.row_max), unroll_full=True):        acc_S_row = acc_S_mn[r, None].load()        row_max_cur = self._compute_row_max(            acc_S_row,            init_val=self.row_max[r] if cutlass.const_expr(not is_first) elseNone,        )        row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)        if cutlass.const_expr(is_first):            acc_S_row_exp = utils.exp2f(                acc_S_row * self.scale_log2 - row_max_cur * self.scale_log2            )            acc_S_row_sum = self._compute_row_sum(acc_S_row_exp)            row_scale[r] = 1.0        else:            row_scale[r] = utils.exp2f(                (self.row_max[r] - row_max_cur) * self.scale_log2            )            acc_S_row_exp = utils.exp2f(                acc_S_row * self.scale_log2 - row_max_cur * self.scale_log2            )            acc_S_row_sum = self._compute_row_sum(                acc_S_row_exp, init_val=self.row_sum[r] * row_scale[r]            )        self.row_max[r] = row_max_cur        self.row_sum[r] = acc_S_row_sum        acc_S_mn[r, None].store(acc_S_row_exp)    return row_scale
```

这里的关键是`row_scale`。当新的 K/V 块带来更大的 row max 时，之前累积的输出`O`必须按比例缩放，否则不同块之间的 softmax 分母不一致。它像是在“边读边修正账本”：每读一页数据，就更新最大值、更新分母，并把旧账按新尺度折算。

**但 GDPA 更进一步，把激活函数作为一等公民**。`activation_utils.py`里不仅实现 ReLU、GELU、FastGELU 等，还大量使用 packed F32x2、inline PTX、FP8 转换等低层技巧。以 ReLU + MXFP8 block scaling 为例，代码注释已经点明优化点：用 3-input max 减少指令，用`cvt.rn.satfinite.relu.e4m3x2.f32`把 ReLU 融合进 FP8 转换，用直接 uint32 store 避免 PRMT。

```
// 来源：gdpa/src/activation_utils.py@cute.jitdef relu_and_convert_blockscaled(    self,    acc_S_row: cute.Tensor,    acc_S_row_converted: cute.Tensor,) -> tuple[Uint8, Uint8, Uint8, Uint8]:    """Apply ReLU, compute per-32-block scales, and quantize for MXFP8.    Optimized version:    - Uses 3-input max pattern for AMAX    - Uses cvt.rn.satfinite.relu.e4m3x2.f32 for fused ReLU+FP8 conversion    - No abs() needed for AMAX since ReLU zeroes negatives    - Direct uint32 stores to avoid PRMT instructions    """
```

GELU 路径则不能直接用`.relu`修饰符，因为 GELU 可能产生负值，所以它先应用 GELU，再对绝对值求 amax，最后 scale 并转换成 FP8。这个差异很小，却非常底层：激活函数的数学性质决定了能否使用某条硬件转换指令，进而影响内核指令数。

这也是 GDPA 真正“广义”的地方。它不是在 PyTorch 层面写：

```
P = torch.nn.functional.gelu(Q @ K.T)
```

而是在 GPU tile 内部、在 MMA accumulator 仍然热乎的时候，就把激活、梯度、量化、存储全部安排好。少一次写回、少一次读入、少一次中间张量，就少一段显存带宽压力。

## 七、TLX 旁支：Block Attention、Norm Fusion 与 Megakernel

> 除了 CuTe DSL 版 GDPA，仓库还包含三个 TLX 项目。它们看似分散，实际都在回答同一个问题：**Blackwell 时代，如何把更长的计算链条留在 GPU 内核内部。**

`block_attention/`提供 Triton TLX block attention。公共入口很薄：

```
// 来源：block_attention/triton/tlx_block_attention.py@torch.fx.wrapdef block_attention_api(    query: torch.Tensor,    key: torch.Tensor,    value: torch.Tensor,    offsets: torch.Tensor,    cpu_offsets: torch.Tensor,) -> torch.Tensor:    return tlx_block_attention(query, key, value, offsets, cpu_offsets)[0]
```

它的前面会构造`TensorDescriptor`，设置`SMEM_BUFFER`、`TMEM_BUFFER`，再启动`_block_attention` kernel。这个模块的重点不是 API 复杂，而是把变长 block attention 的 Q/K/V/O 描述符、offset 和 block 划分交给 TLX kernel，让内部异步流水线处理数据搬运与计算。

> `multi_cta_norm_fusion/`则聚焦另一个常见瓶颈：matmul 之后接 RMSNorm 或 LayerNorm。传统图执行会把 matmul 结果写回显存，再由 norm kernel 读回来做归一化。

如果能把这两步融合，显存流量会显著下降。它的 RMSNorm 计算支持多 CTA 协同归约：

```
// 来源：multi_cta_norm_fusion/src/tlx_matmul_rmsnorm.py@triton.jitdef _compute_rms_norm(    x: Any,    cluster_cta_rank: Any,    reduction_buf: Any,    barrier: Any,    phase: Any,    BLOCK_SIZE_M: tl.constexpr,    N: tl.constexpr,    NUM_REDUCTION_CTAS: tl.constexpr,) -> Any:    local_partial_sum = tl.sum(x * x, axis=1, keep_dims=True).to(tl.float32)    for i in tl.static_range(NUM_REDUCTION_CTAS):        if cluster_cta_rank != i:            tlx.async_remote_shmem_store(                dst=reduction_buf[cluster_cta_rank],                src=local_partial_sum,                remote_cta_rank=i,                barrier=barrier,            )        else:            tlx.local_store(reduction_buf[cluster_cta_rank], local_partial_sum)    if NUM_REDUCTION_CTAS:        tlx.barrier_wait(barrier, phase=phase)    final_square_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.bfloat16)    for i in tl.static_range(NUM_REDUCTION_CTAS):        final_square_sum += tlx.local_load(reduction_buf[i])    return x * libdevice.rsqrt(final_square_sum / N + 1e-5)
```

这段代码体现了 TLX 的独特味道：CTA 之间通过 remote shared memory 交换局部归约结果，并用 barrier 同步。它不只是“一个 CTA 算一块”，而在 cluster 级别让多个 CTA 合作完成 norm。这对大 N 维归一化尤其重要，因为单 CTA 承载全部归约可能受寄存器、shared memory 或并行度限制。

> `gdpa_megakernel/`则是 TLX 版 GDPA 大内核，`tlx_gdpa_megakernel.py`体量接近 36 万字节，是仓库中最大的单文件之一。:mdnice-bold-italic[Megakernel 的意图通常是减少 kernel launch 与中间内存往返，把更多阶段融合到一个大内核中。它也配套`tlx_math.py`、`vararg_kernel.py`、`register_helpers.py`，说明其内部同样需要定制数学函数、可变参数 kernel 包装和寄存器辅助函数。]

这三个 TLX 模块与 CuTe DSL GDPA 并不是互相替代，而是从不同角度探索：

- CuTe DSL 路径更像精雕细刻的 Blackwell attention 内核；
- TLX Block Attention 强调块级变长注意力；
- Multi-CTA Norm Fusion 强调 matmul 后处理融合；
- GDPA Megakernel 尝试把 GDPA 路径压缩进更大的 TLX 执行单元。

## 八、为什么它重要：性能来自“贴着负载写内核”

> `ads_model_kernel_library`最值得学习的地方，不是某一个函数，也不是某条 PTX 指令，而是它的优化姿态：**它没有假设推荐模型负载会变得规整，而是承认复杂性，并把复杂性逐层编码进内核。**

- **第一层是接口复杂性**。`flash_attn_varlen_func`把`cu_seqlens`、`seqused`、`page_table`、`tile_to_batch`、`activation`、`sfq/sfk/sfv`都暴露出来。这些参数看起来繁琐，但它们让调用者能明确表达真实数据形态。
- **第二层是调度复杂性**。变长序列不是靠 padding 解决，而是通过`SeqlenInfoQK`、`BlockInfo`、varlen scheduler 和 zigzag load balancing 减少无效 tile、均衡 SM 负载。这是推荐系统比标准 Transformer benchmark 更棘手的地方。
- **第三层是数学复杂性**。GDPA 不把 softmax 当成唯一选择，而是把 GELU、ReLU、FastGELU 等激活放进 attention 中间态。由于不同激活对数值范围、梯度和量化路径的要求不同，内核必须为它们定制转换与反向。
- **第四层是硬件复杂性**。代码里到处可见对 Blackwell SM100、Tensor Memory、TMA、packed F32x2、E4M3、E8M0、remote shared memory、barrier 的利用。性能不是来自“Python 写得优雅”，而是来自 Python DSL 背后生成的 GPU 指令序列足够贴近硬件。

README 中提到，在某些真实生产流量设置下，CuteDSL GDPA 相对 FA4 可达到最高 3.5 倍前向和 1.6 倍反向加速。这个数字不应被理解为“所有 attention 都能快这么多”，**而应理解为：当负载特征与内核假设高度吻合时，针对性内核能远远超过通用模板。**

> 总结来看，这个仓库给技术读者的启发有三点。

- 其一，推荐系统的算子优化已经不只是 embedding lookup 或稀疏特征工程，attention 类结构正在进入广告推荐模型的核心路径。
- 其二，真实负载的性能瓶颈往往藏在变长、mask、head mapping 和后处理融合里，而不是单纯的矩阵乘法峰值。
- 其三，未来高性能 AI 系统工程会越来越依赖这种“算法—编译器 DSL—GPU 微架构”三者合一的写法：用 Python 表达接口，用 DSL 生成内核，用硬件特性兑现吞吐。

::: block-1
`ads_model_kernel_library`不是一个适合初学者直接拿来改模型的高层库，它更像一份公开的工程剖面：**告诉我们 Meta Ads AI 如何把推荐模型训练中的真实不规则性拆开，再用 CuTe DSL 与 Triton TLX 重新缝进 GPU**。对做 AI 系统、推荐训练、编译器后端或 GPU 内核的人来说，它的价值正在于此——不是展示一个干净的抽象世界，反倒是展示性能从混乱现实中被一点点榨出来的过程。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[消除推荐系统广播税，从内存带宽到计算瓶颈的蜕变，揭秘 621 TFLOPS 的 Flash Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900976&idx=1&sn=0f29d856ded3ed90bc6c1c4ce3152d0f&scene=21#wechat_redirect)**
- **[FlashNormAttention 融合注意力与归一化到 GEMM，最高可掩盖 90%归一化时延](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903614&idx=1&sn=d4c7a5e4345d25b1925979e0793d531a&scene=21#wechat_redirect)**
- **[适配 Ada 中端卡的 MegaKernel 在线广告方案 Ada-MK：共享内存直接减半，端到延迟最高降 50%，击穿小批量毫秒级推理瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902893&idx=1&sn=5f53dd32ac726cea553f1c3743d8d2a1&scene=21#wechat_redirect)**

:::
