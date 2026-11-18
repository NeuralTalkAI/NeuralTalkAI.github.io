关键词：**ROCm**、AITER、:mdnice-bold-italic[分页 KV 缓存]、分段编译、**推理服务**

> 当我们谈论大模型推理优化时，最容易被看见的是单个算子：GEMM 快了多少，Attention 快了多少，MoE 路由是否融合。

![文章插图](/articles/2026-05-04-aiter-p-d-amd-gpu-7b1550d5/images/01.webp)

::: image-caption
这张图展示了DeepSeek R1 0528模型在FP8精度下，不同GPU与推理框架的单卡Token吞吐量和端到端延迟的权衡关系，横轴为延迟（秒），纵轴为吞吐量（tok/s/gpu），整体呈现吞吐量随延迟升高而提升的趋势，体现大模型推理中“速度-延迟”的典型性能权衡。其中，AMD MI355X搭配ATOM框架的表现最优，延迟约26秒时吞吐量突破1100 tok/s/gpu，同系列SGLang版本次之；NVIDIA B200（TRT）的吞吐量在相近延迟下显著更低，直观展现了不同硬件与优化栈在该模型推理性能上的明显差异。更多数据，去这里看：https://inferencex.semianalysis.com/
:::

**但真正决定线上服务体验的，往往不是某个 kernel 的峰值**，而是:mdnice-bold-italic[一条请求从 HTTP 入口进入、被分词、调度、写入 KV cache、进入 GPU graph、经过采样再流式返回的完整链路。]

![文章插图](/articles/2026-05-04-aiter-p-d-amd-gpu-7b1550d5/images/02.webp)

- ATOM (AiTer Optimized Model) is **a lightweight vLLM-like implementation, focusing on integration and optimization based on AITER.**
- 代码：https://github.com/ROCm/ATOM
- 文档：https://rocm.github.io/ATOM/docs
- 5000 字，阅读 22 分钟，**播客 27 分钟**

相关推荐

- [AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)
- [AMD MI300X GPU 计算、内存、通信性能分析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894062&idx=1&sn=9ad5eaaccdb2c7fa64db9cc1eaeb6e62&scene=21#wechat_redirect)
- [AMD GPU 设备数学库分层编译架构浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899885&idx=1&sn=c4a604edeff740435f04ea5b6f90bf7c&scene=21#wechat_redirect)

> ATOM（AiTer Optimized Model）正是这样一个项目：它不是庞大到包罗万象的训练/推理框架，而是一个:mdnice-bold-italic[轻量 vLLM-like 推理引擎，用 AMD ROCm 平台上的 AITER 内核，把模型执行、分页 KV、编译图捕获、多 GPU 并行、OpenAI 兼容服务串成一个可运行、可压测、可剖析的系统。]

它的价值不只在“能跑”，更在于把 AMD GPU 上的大模型推理优化路径具体化为工程结构。

**下面的表格列出了 ATOM 目前适配的主要模型家族**、HuggingFace 架构与密集/混合专家（MoE）类型，并附上关键备注，方便你快速了解兼容范围和注意事项。

| Model Family | HF Architecture | Dense/MoE | Notes |
| --- | --- | --- | --- |
| Llama[1] | `LlamaForCausalLM` | Dense | Llama 2, Llama 3, Llama 3.1 |
| Qwen3[2] | `Qwen3ForCausalLM` | Dense |  |
| Qwen3-MoE[3] | `Qwen3MoeForCausalLM` | MoE | 128 experts, top-8 routing |
| Qwen3-Next[4] | `Qwen3NextForCausalLM` | MoE | Hybrid full attention + Gated DeltaNet |
| DeepSeek V2/V3[5] | `DeepseekV3ForCausalLM` | MoE | MLA attention, MTP speculative decoding |
| Mixtral[6] | `MixtralForCausalLM` | MoE | 8 experts, top-2 routing |
| GLM-4-MoE[7] | `Glm4MoeForCausalLM` | MoE |  |
| GLM-5[8] | `GlmMoeDsaForCausalLM` | MoE | MLA attention, similar to DeepSeek V3.2. See recipe[9] |
| GPT-OSS[10] | `GptOssForCausalLM` | MoE | Sliding window + attention sinks |
| Kimi-K2[11] | via `--trust-remote-code` | MoE | See recipe[12] |
| MiMo-V2-Flash[13] | `MiMoV2FlashForCausalLM` | MoE | Hybrid full + SWA attention, 3-layer MTP. See recipe[14] |

## 本文目录

- 一、快速上手：先把 ATOM 跑起来
- 二、项目定位：不是“模型库”，而是 ROCm 推理链路样板间
  - 2.1 ATOM 的核心目标
  - 2.2 整体组件关系
- 三、请求生命周期：一条 prompt 如何穿过 ATOM
  - 3.1 从 API 到 Sequence
  - 3.2 Sequence 是请求在系统里的身份证
- 四、调度器：prefill-first 与连续 batching 的取舍
  - 4.1 为什么先 prefill
  - 4.2 Decode 阶段：KV 空间不足就抢占
- 五、分页 KV Cache：把显存切成可复用的街区
  - 5.1 BlockManager 的本质
  - 5.2 Prefix caching：用 xxhash64 识别相同前缀
- 六、算子层：AITER 是 ATOM 的性能地基
  - 6.1 Linear：按量化类型分派 GEMM
  - 6.2 Attention：把复杂动态逻辑封装成可切分 custom op
- 七、编译与 CUDA Graph：把 decode 的 CPU 开销压到最低
  - 7.1 四档编译级别
  - 7.2 为什么 prefill eager，decode graph replay
- 八、MoE、量化与多卡：ATOM 面向大模型的真正压力区
  - 8.1 MoE 路径：路由、专家并行与 MORI 通信
  - 8.2 量化不是单一格式，而是一组执行合同
- 九、P/D 分离：把 prefill 与 decode 拆到不同 GPU 节点
- 十、如何评价 ATOM：轻量外壳下的系统工程密度
- 结语：ATOM 的意义在于把“ROCm 能跑大模型”推进到“ROCm 能服务大模型”

![文章插图](/articles/2026-05-04-aiter-p-d-amd-gpu-7b1550d5/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、快速上手：先把 ATOM 跑起来

> ATOM 的官方 README 给出的最短路径是直接使用 nightly Docker 镜像。

它默认要求 AMD GPU、ROCm 支持与 Docker 环境。**最推荐的方式是拉取预构建镜像**，里面已包含 AITER 与 ATOM。来自 `README.md` 的推荐镜像启动方式：

```
docker pull rocm/atom-dev:latestdocker run -it --network=host \  --device=/dev/kfd \  --device=/dev/dri \  --group-add video \  --cap-add=SYS_PTRACE \  --security-opt seccomp=unconfined \  -v $HOME:/home/$USER \  -v /mnt:/mnt \  -v /data:/data \  --shm-size=16G \  --ulimit memlock=-1 \  --ulimit stack=67108864 \  rocm/atom-dev:latest
```

如果希望从基础 ROCm PyTorch 镜像构建，README 中给出了另一条路径：先进入 `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0`，再安装 AITER 与 ATOM。

```
pip install amd-aitergit clone https://github.com/ROCm/ATOM.git && pip install ./ATOM
```

最小推理示例则通过 Python module 方式启动：

```
python -m atom.examples.simple_inference --model meta-llama/Meta-Llama-3-8B --kv_cache_dtype fp8
```

如果读者更关心服务化部署，ATOM 提供 OpenAI-compatible server，直接暴露 `/v1/chat/completions` 与 `/v1/completions` 类接口：

```
# Single GPUpython -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8# Multi-GPU with tensor parallelismpython -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8# With MTP speculative decodingpython -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8 \  --method mtp --num-speculative-tokens 3
```

更完整的架构、配置、模型支持、调度、编译、多卡与 benchmark 说明，可参考项目文档入口：rocm.github.io/ATOM/docs[15]。

## 二、项目定位：不是“模型库”，而是 ROCm 推理链路样板间

### 2.1 ATOM 的核心目标

README 对 ATOM 的定义很清晰：它是一个 :mdnice-bold-italic[lightweight vLLM-like implementation，重点是基于 AITER[16] 做集成与优化。换句话说，ATOM 的重心不是重新发明 Transformer，而是回答一个更系统的问题：]

> 如果 AMD 已经有一组高性能 ROCm kernel，怎样把它们组织成一个真实可服务的大模型推理引擎？

这决定了 ATOM 的架构关注点非常工程化：

- **算子层**：Linear、Attention、MoE、Norm、Sampling 等重计算路径交给 AITER。
- **模型层**：支持 Llama、Qwen3、DeepSeek、Mixtral、GLM、GPT-OSS、Kimi、MiMo 等架构。
- **执行层**：使用 Scheduler、BlockManager、ModelRunner 组织连续 batching 与 KV cache。
- **编译层**：提供 0 到 3 四档编译级别，生产默认是 piecewise `torch.compile` + CUDA graph。
- **服务层**：暴露 OpenAI-compatible API，并提供 benchmark、profiling、accuracy validation。
- **分布式层**：支持 TP、DP、EP，以及基于 MORI 的 MoE all-to-all 与 P/D 分离 KV 传输。

::: block-1
因此，ATOM 更像一间“ROCm 推理系统样板间”：它把 AMD GPU 上的高性能内核、编译策略和线上推理机制放在同一个可验证框架里。
:::

### 2.2 整体组件关系

> 官方架构文档中给出的主链路如下：`LLMEngine` 面向用户，`CoreManager` 做多进程编排，`EngineCore` 持有调度器与模型执行器，`ModelRunner` 真正驱动每个 GPU 上的模型 forward。

可以把它想象成一家餐厅：

- `LLMEngine` 是前台，接收用户请求；
- `InputOutputProcessor` 是点餐员，负责分词、反分词与统计；
- `Scheduler` 是后厨调度，决定先做哪些菜；
- `BlockManager` 是仓储管理员，管理 KV cache 货架；
- `ModelRunner` 是厨师，调用 AITER kernel 完成真正计算；
- `CoreManager` 和 ZMQ 像传菜系统，把请求送进不同进程。

## 三、请求生命周期：一条 prompt 如何穿过 ATOM

### 3.1 从 API 到 Sequence

> ATOM 的用户入口在 `atom/model_engine/llm_engine.py`。

`LLMEngine.generate()` 会先调用 `add_request()`，把 prompt 和 sampling 参数交给 `InputOutputProcessor.preprocess()`，生成内部的 `Sequence` 对象，再交给 `CoreManager`。

```
# atom/model_engine/llm_engine.pydef generate(    self,    prompts: list[str],    sampling_params: SamplingParams | list[SamplingParams],) -> list[str]:    # Reset round-robin counter to ensure consistent DP not core dump    self.core_mgr._rr_counter = 0    self.add_request(prompts, sampling_params)    outputs = {}    whilenot self.is_finished() and (        self.core_mgr.is_alive() or self.core_mgr.is_rest()    ):        seqs = self.step()        outs = self.io_processor.postprocess(seqs)        outputs.update(outs)    outputs = [outputs[seq_id] for seq_id in sorted(outputs)]    return outputs
```

这段代码非常朴素，但体现了 ATOM 的核心控制模式：用户线程不是直接跑模型，而是持续从 engine core 拉取完成的 `Sequence`，再做 detokenize 与统计。

### 3.2 Sequence 是请求在系统里的身份证

`Sequence` 保存了 prompt token、生成 token、KV block table、状态、采样参数、到达时间、首 token 时间、离开时间等信息。它既是调度单位，也是性能统计单位。

ATOM 的请求状态大致是：

```
WAITING -> RUNNING(PREFILL) -> RUNNING(DECODE) -> FINISHED
```

如果 KV cache 空间不够，运行中的请求还可能被抢占，回到等待队列重新 prefill。这看似浪费，但在高并发场景中，它让系统有机会在有限 KV 空间下维持整体吞吐。

## 四、调度器：prefill-first 与连续 batching 的取舍

### 4.1 为什么先 prefill

> 大模型推理分两段：prefill 处理整段输入 prompt，decode 一次通常只生成一个或少量 token。prefill 计算量大、形状变化大；decode 频繁、延迟敏感、适合 CUDA graph 重放。

ATOM 的 `Scheduler.schedule()` 采用 prefill-first：只要有可以调度的等待请求，就先组成 prefill batch；没有 prefill 时再调度 decode。

核心代码来自 `atom/model_engine/scheduler.py`：

```
# atom/model_engine/scheduler.pydef schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]]:    """Select the next batch of sequences for a forward pass.    Tries prefill first; if no new prefills are ready, falls back to    decoding already-running sequences.    """    scheduled_seqs = {}    num_seqs_prefill = 0    num_batched_tokens = 0    skipped_waiting_requests: deque[Sequence] = deque()    num_scheduled_tokens: list[int] = []    scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}    ifnot self.running andnot self.waiting:        returnNone    # --- Prefill scheduling ---    while self.waiting and num_seqs_prefill < self.max_num_seqs:        seq = self.waiting.popleft()        num_new_tokens = seq.num_tokens - seq.num_cached_tokens        if (            num_batched_tokens + num_new_tokens > self.max_num_batched_tokens            ornot self.block_manager.can_allocate(seq)        ):            self.waiting.appendleft(seq)            break        self.block_manager.allocate(seq)        num_seqs_prefill += 1        num_new_tokens = seq.num_tokens - seq.num_cached_tokens        if self.cache_stats:            self.cache_stats.update(seq.num_cached_tokens, seq.num_tokens)        num_batched_tokens += num_new_tokens        seq.status = SequenceStatus.RUNNING        seq.type = SequenceType.PREFILL        self.running.append(seq)        scheduled_seqs[seq.id] = seq        num_scheduled_tokens.append(num_new_tokens)
```

这段逻辑的关键在两个预算：

- `max_num_seqs`：一个 batch 中最多多少条序列；
- `max_num_batched_tokens`：一个 step 中最多处理多少 token。

prefill-first 的好处是能尽快让新请求进入系统，降低 TTFT；代价是可能暂时打断 decode 的连续性。因此 ATOM 还提供 `scheduler_delay_factor`，允许等待队列攒一小段时间，以换取更好的 batch 效率。

### 4.2 Decode 阶段：KV 空间不足就抢占

> 当没有 prefill 被调度时，调度器进入 decode 阶段。它会检查每条 running sequence 是否还能追加 KV block，如果不够，就抢占队尾请求。

```
# atom/model_engine/scheduler.py# --- Decode scheduling ---num_seqs_decode = 0num_new_tokens = self.mtp_k + 1while self.running and num_seqs_decode < self.max_num_seqs:    seq = self.running.popleft()    whilenot self.block_manager.can_append(seq, num_new_tokens):        if self.running:            self.preempt(self.running.pop())        else:            self.preempt(seq)            break    else:        if seq.spec_token_ids.size > 0:            scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids        num_seqs_decode += 1        ifnot getattr(seq, "is_first_decode", False):            self.block_manager.may_append(seq, num_new_tokens)        scheduled_seqs[seq.id] = seq        seq.type = SequenceType.DECODE        num_scheduled_tokens.append(num_new_tokens)        seq.is_first_decode = False
```

这里的 `num_new_tokens = self.mtp_k + 1` 很有意思：如果启用 MTP speculative decoding，一步 decode 不只是一个 target token，还要为 draft token 预留 KV 空间。调度器并不“理解”模型内部如何验证 draft，但它必须提前为可能写入的 token 准备缓存位置。

## 五、分页 KV Cache：把显存切成可复用的街区

### 5.1 BlockManager 的本质

大模型推理的显存瓶颈常常不是权重，而是 KV cache。ATOM 用 `BlockManager` 管理固定大小的 KV block，默认 block size 在文档中为 16 token。每个 `Sequence` 维护一个 `block_table`，指向自己占用的 block。

`Block` 数据结构非常小：

```
# atom/model_engine/block_manager.pyclass Block:    def __init__(self, block_id):        self.block_id = block_id        self.ref_count = 0        self.hash = -1        self.token_ids = []    def update(self, hash: int, token_ids: list[int]):        self.hash = hash        self.token_ids = token_ids    def reset(self):        self.ref_count = 1        self.hash = -1        self.token_ids = []
```

可以把 KV cache 想象成酒店房间：请求来了就分配房间，结束就退房；如果两个客人拥有相同的前缀 prompt，它们甚至可以共享前面几间房间。这里的 `ref_count` 就是共享房间的住户数。

### 5.2 Prefix caching：用 xxhash64 识别相同前缀

ATOM 的 prefix caching 使用 `xxhash64`，并采用链式 hash：当前 block 的 hash 会包含前一个 block 的 hash。这样即使两个 block 的 token 内容相同，只要前文不同，hash 也不同，避免错误共享上下文。

```
# atom/model_engine/block_manager.py@classmethoddef compute_hash(cls, token_ids: list[int], prefix: int = -1):    h = xxhash.xxh64()    if prefix != -1:        h.update(prefix.to_bytes(8, "little"))    h.update(np.array(token_ids).tobytes())    return h.intdigest()
```

这不是简单的“把 token 列表哈希一下”。链式 hash 保证了 KV block 的语义与它所在的上下文绑定。对于多轮对话、批量相似 prompt、RAG 模板复用等场景，这种机制可以减少重复 prefill，提高吞吐。

## 六、算子层：AITER 是 ATOM 的性能地基

### 6.1 Linear：按量化类型分派 GEMM

ATOM 的 `atom/model_ops/linear.py` 是理解其算子策略的关键文件。它把不同量化格式映射到不同 AITER GEMM 实现：

- 无量化：`tgemm.mm`
- per-tensor FP8：带 scale 的 `tgemm.mm`
- per-token INT8：`gemm_a8w8`
- per-token FP8：`gemm_a8w8_bpreshuffle`
- per-1x128 FP8：`gemm_a8w8_blockscale_bpreshuffle`
- per-1x32 MXFP4：`gemm_a4w4`

核心 dispatch 如下：

```
# atom/model_ops/linear.py@mark_tracedef forward(    self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None, otype=dtypes.bf16) -> torch.Tensor:    if self.quant_type.value == QuantType.No.value:        y = tgemm.mm(            x,            self.weight,            self.bias,            otype=otype,        )    else:        if x_scale isNone:            quant_func = self.quant_func            if self.quant_type.value == QuantType.per_1x128.value:                quant_func = functools_partial(                    self.quant_func, transpose_scale=True                )            if self.quant_type.value != QuantType.per_1x32.value:                x, x_scale = quant_func(                    x,                    quant_dtype=self.params_dtype,                    scale=getattr(self, "input_scale", None),                )        if self.quant_type.value == QuantType.per_Tensor.value:            y = tgemm.mm(                x,                self.weight,                self.bias,                otype=otype,                scale_a=x_scale,                scale_b=self.weight_scale,            )        elif self.quant_type.value == QuantType.per_Token.value:            if self.params_dtype == dtypes.i8:                y = gemm_a8w8(                    x,                    self.weight,                    x_scale,                    self.weight_scale,                    self.bias,                    dtype=otype,                )            else:                y = gemm_a8w8_bpreshuffle(                    x,                    self.weight,                    x_scale,                    self.weight_scale,                    dtype=otype,                )        elif self.quant_type.value == QuantType.per_1x128.value:            y = gemm_a8w8_blockscale_preshuffle_impl(                x,                self.weight,                x_scale,                self.weight_scale,                dtype=otype,                prefix=self.prefix,            )        elif self.quant_type.value == QuantType.per_1x32.value:            y = gemm_a4w4_quant(                x,                x_scale,                self.weight,                otype,                self.weight_scale.data,                self.params_dtype,                getattr(self, "input_scale", None),                self.output_size,            )    if self.tp_dim == 1and self.tp_size > 1and self.reduce_results:        y = get_tp_group().all_reduce(y, ca_fp8_quant=False)    return y
```

这里有两个重要设计点：

第一，ATOM 不把量化当成“模型加载时的一次性转换”，而是在 layer forward 中保留 scale、weight layout、kernel dispatch 的明确路径。第二，Tensor Parallel 逻辑也在 Linear wrapper 内部处理，例如 row parallel 输出需要 all-reduce，这让模型文件可以保持相对简洁。

### 6.2 Attention：把复杂动态逻辑封装成可切分 custom op

Attention 是推理系统里最难被编译器“顺滑处理”的部分，因为它依赖动态 sequence length、block table、KV cache、prefill/decode 分支。ATOM 的做法不是强行让 Dynamo/Inductor 看懂所有细节，而是把 attention 注册成 custom op 和 splitting op。

```
# atom/model_ops/base_attention.py# Dynamo will not try to inspect any of the internal operations for prefill or decode# This way, although attention operation is complicated,# we can still capture the model's computation graph as a full-graph@mark_spliting_op(is_custom=True, gen_fake=fake_, mutates_args=[])def unified_attention_with_output_base(    q: torch.Tensor,    q_scale: Optional[torch.Tensor],    k: torch.Tensor,    v: torch.Tensor,    positions: torch.Tensor,    layer_name: str,    use_mla: bool,    qkv: torch.Tensor,) -> torch.Tensor:    atom_config = get_current_atom_config()    self = atom_config.compilation_config.static_forward_context[layer_name]    if use_mla:        return self.impl.forward(            layer=self,            query=q,            k_nope=k,            k_rope=v,            positions=positions,            q_scale=q_scale,        )    else:        return self.impl.forward(            layer=self,            query=q,            key=k,            value=v,            position=positions,            q_scale=q_scale,            qkv=qkv,        )
```

这段代码揭示了 ATOM 的编译哲学：**不是所有东西都交给编译器优化；该黑盒的地方就黑盒，该切图的地方就切图。**

Attention 内部再根据 MHA/MLA、prefill/decode、Triton/ASM、KV cache dtype 等条件调用不同 AITER kernel。对于 DeepSeek 一类 MLA 模型，ATOM 还支持压缩 KV 表示与专门的 MLA decode/prefill kernel。

## 七、编译与 CUDA Graph：把 decode 的 CPU 开销压到最低

### 7.1 四档编译级别

ATOM 文档定义了 0 到 3 四个 compilation level：

- Level 0：纯 eager；
- Level 1：Dynamo as-is，后端为 eager；
- Level 2：Dynamo + Inductor；
- Level 3：Piecewise compilation + CUDA graph，生产默认。

Level 3 的核心是把模型图在 attention 等 splitting op 处分开，能编译的子图交给 Inductor，复杂动态 attention 留给专门 kernel。然后对 decode batch 捕获 CUDA graph，运行时重放。

这就像修高速公路：不是把每条乡间小路都改造成高速，而是识别出车流稳定、形状固定的主干路段，铺上最顺的路面；容易拥堵、入口复杂的路口，则保留专门调度。

### 7.2 为什么 prefill eager，decode graph replay

官方编译文档说明：prefill 因为长度变化大，通常走 eager；decode batch 形状更稳定，适合预捕获 CUDA graph。`ModelRunner.run_model()` 中的逻辑可概括为：

```
# docs/compilation_cudagraph_guide.mddef run_model(self, input_ids):    forward_context = get_forward_context()    context = forward_context.context    bs = context.batch_size    is_prefill = context.is_prefill    positions = context.positions    if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:        # Eager path: prefills, enforce_eager mode, or oversized batches        hidden_states = self.model(input_ids, positions)    else:        # Graph replay path: decode batches within captured range        graph_bs = context.graph_bs        max_q_len = forward_context.attn_metadata.max_seqlen_q        graph_key = (graph_bs, max_q_len)        self.graphs[graph_key].replay()        num_tokens = context.batch_size * max_q_len        hidden_states = self.forward_vars["outputs"][:num_tokens]    return self.model.compute_logits(hidden_states), hidden_states
```

这段伪代码说明了一个现实主义原则：**prefill 追求吞吐与灵活性，decode 追求稳定低延迟。**

在在线服务中，每生成一个 token 都要经过一轮模型执行。如果每轮都由 Python 重新发射大量 kernel，CPU launch overhead 会被放大。CUDA graph replay 的意义，就是把这部分调度开销压下去。

## 八、MoE、量化与多卡：ATOM 面向大模型的真正压力区

### 8.1 MoE 路径：路由、专家并行与 MORI 通信

ATOM 支持 DeepSeek、Mixtral、Qwen3-MoE、GLM、GPT-OSS 等 MoE 模型。MoE 的难点不只是专家 GEMM，还包括 token routing、top-k、专家间通信、输出 combine。项目文档显示，`FusedMoE` 会根据量化配置选择不同方法，例如 FP8、MXFP4、unquantized，并可结合 MORI 做 expert parallel all-to-all。

在大模型推理里，MoE 像一个大型分拣中心：每个 token 根据 router 分数被送去少数专家，专家计算完再汇总。如果分拣、运输、计算三个阶段不能流水化，理论 FLOPs 再高也难以转化成吞吐。ATOM 引入 MORI all-to-all 与 Two-Batch Overlap 的意义，就在于把通信从“等待项”变成可被重叠隐藏的流水线阶段。

### 8.2 量化不是单一格式，而是一组执行合同

> README 明确列出 ATOM 支持 FP8、MXFP4、INT8、INT4，并可从 HuggingFace config 自动检测。源码层面看，量化格式会影响：

- 参数 dtype；
- scale tensor shape；
- weight 是否需要 preshuffle；
- activation 是否动态 quant；
- GEMM kernel 选择；
- TP shard 加载方式；
- MoE expert kernel 路径。

这意味着 ATOM 的量化不是“把权重压小”这么简单，而是贯穿加载、布局、计算、通信的执行合同。不同格式背后对应不同 kernel 与 layout，错误的 layout 甚至会抵消量化收益。

## 九、P/D 分离：把 prefill 与 decode 拆到不同 GPU 节点

ATOM 在 2026/03 的 README news 中强调支持 Prefill/Decode disaggregation。其 `atom/kv_transfer/disaggregation/README.md` 说明了机制：prefill 节点计算 KV cache，然后通过 MORI-IO 的 RDMA 能力把 KV 传给 decode 节点，decode 节点跳过 prefill 直接生成 token。

简化流程如下：

```
Client -> Proxy          -> Prefill Node: 计算 KV cache          -> Proxy: 接收 block metadata          -> Decode Node: RDMA 读取 KV 并生成 token          -> Proxy -> Client
```

这背后的系统动机非常明确：prefill 和 decode 的资源画像不同。prefill 更像大批量矩阵计算，decode 更像低延迟小步迭代。把它们拆到不同节点，就可以分别优化吞吐与延迟，也便于在服务集群里独立扩缩容。

## 十、如何评价 ATOM：轻量外壳下的系统工程密度

> ATOM 的代码 99% 以上是 Python，但这并不意味着它“只是 Python 框架”。恰恰相反，它的 Python 层承担的是高性能推理系统中最难统一的部分：调度、状态、缓存、编译边界、分布式通信、服务接口与性能验证。

它的核心价值可以总结为四点：

1. **把 AITER kernel 系统化**单个 kernel 再快，如果没有模型 wrapper、调度器和服务入口，也很难落地。ATOM 把 AITER 放进完整推理链路。
2. **把 vLLM-like 架构移植到 ROCm 语境**Scheduler、paged KV、continuous batching、OpenAI API 都是成熟推理系统的关键构件，ATOM 将它们与 AMD GPU 优化结合。
3. **把编译器与手写 kernel 边界处理得务实**Attention 等动态复杂路径通过 custom op 切出，Linear、MLP 等稳定路径交给编译和 CUDA graph。这是工程上更稳的选择。
4. **把大模型服务的复杂场景前置**MoE、MTP speculative decoding、FP8/MXFP4、TP/DP/EP、P/D 分离、benchmark dashboard，都说明 ATOM 面向的不是 demo，而是线上推理压力场。

当然，它也有边界。ATOM 强依赖 ROCm、AMD GPU 与 AITER；首次运行可能有编译耗时；部分高级能力如 P/D 分离、MORI、EP 通信依赖特定硬件和网络环境。对普通单卡用户来说，它的最佳入门方式是先跑 README 中的 simple inference 与 OpenAI server；对系统工程读者来说，更值得读的是 `atom/model_engine/`、`atom/model_ops/`、`docs/compilation_cudagraph_guide.md` 与 `docs/scheduling_kv_cache_guide.md`。

## 结语：ATOM 的意义在于把“ROCm 能跑大模型”推进到“ROCm 能服务大模型”

> 如果只看项目名，ATOM 像是一个模型优化仓库；如果读完代码，会发现它真正关心的是端到端推理服务的工程闭环。

它把 AITER 的底层 kernel、分页 KV cache 的显存管理、prefill-first 的动态调度、piecewise compile 的图优化、CUDA graph 的低延迟重放、多 GPU 并行与 OpenAI-compatible server 拼接在一起。

::: block-1
这类项目的意义不在于“替代所有框架”，而在于清晰展示一条路线：AMD ROCm 生态中的大模型推理，不应停留在单算子 benchmark，而要**走向可部署、可观测、可压测、可扩展的系统形态**。ATOM 正是在这条路上，把抽象的硬件能力翻译成具体工程结构的一个关键样本。
:::

参考资料

[1] Llama: *https://huggingface.co/meta-llama*

[2] Qwen3: *https://huggingface.co/Qwen*

[3] Qwen3-MoE: *https://huggingface.co/Qwen*

[4] Qwen3-Next: *https://huggingface.co/Qwen*

[5] DeepSeek V2/V3: *https://huggingface.co/deepseek-ai*

[6] Mixtral: *https://huggingface.co/mistralai/Mixtral-8x7B-v0.1*

[7] GLM-4-MoE: *https://huggingface.co/THUDM*

[8] GLM-5: *https://huggingface.co/zai-org/GLM-5-FP8*

[9] recipe: *https://github.com/ROCm/ATOM/blob/main/recipes/Kimi-K2-Thinking.md*

[10] GPT-OSS: *https://huggingface.co/openai*

[11] Kimi-K2: *https://huggingface.co/moonshotai/Kimi-K2-Thinking*

[12] recipe: *https://github.com/ROCm/ATOM/blob/main/recipes/Kimi-K2-Thinking.md*

[13] MiMo-V2-Flash: *https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash*

[14] Hybrid full + SWA attention, 3-layer MTP. See recipe: *https://github.com/ROCm/ATOM/blob/main/recipes/MiMo-V2-Flash.md*

[15] rocm.github.io/ATOM/docs: *https://rocm.github.io/ATOM/docs*

[16] AITER: *https://github.com/ROCm/aiter*
