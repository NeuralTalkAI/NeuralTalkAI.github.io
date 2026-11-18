关键词：:mdnice-bold-italic[端侧推理]、**ARM 优化**、模型量化、**MoE 卸载**、图执行引擎

> 大模型本地运行的难点，常被简化成一句话：内存不够，算力不够。

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/01.webp)

- **Palm on-device acceleration capabilities.**
- https://github.com/TencentYoutuResearch/Palm-Infra

::: related-title
**相关推荐**
:::

::: related-links
- **[不止算子优化！ncnn Vulkan int8 量化推理后端：全链路架构设计与系统性优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902586&idx=1&sn=14f256d165dccfb6c4ef866fd144ab6b&scene=21#wechat_redirect)**
- **[NVIDIA 开源 cuDNN Frontend：用图 API 把底层内核能力封装成可复用、可调优的 AI 算子构建系统](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901178&idx=1&sn=6b7dd90b31e401357660a4a02cdd1262&scene=21#wechat_redirect)**
- **[让端侧AI不再难用！谷歌开源 LiteRT-CLI：用一行命令重写 TFLite 模型工作流，从 HuggingFace 到 NPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903206&idx=1&sn=419b84c1b6a835de19f43cdcf6d37e11&scene=21#wechat_redirect)**

:::

> **Palm-Infra 里的 mollm 给出了一个更工程化的答案**：把模型转换成可直接加载的单文件包，把 Python 侧复杂的图构建留在离线阶段，把 C++ 运行时收敛成一次线性图遍历，再用 ARM NEON、FP16FML、dot-product、i8mm、W4/W8 权重量化和 SSD 专家缓存去抠每一毫秒。

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/02.webp)

::: image-caption
当前已支持的模型见上图，测试最充分的运行路径是 w4g128：它占用内存最少，且具有 mollm 中最快的 decode 速度。若纯 W4 的质量不足，可使用 w4mixg128，它会保留部分敏感 tensor 为 W8。
:::

**它最有冲击力的地方，是让 48GB Apple Silicon Mac 跑起 W4 量化的 Qwen3.5-122B-A10B MoE**：稠密权重常驻内存，被路由到的专家从 SSD 异步读入，再通过共享缓存和跨层预取隐藏 I/O 延迟。

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/03.webp)

::: image-caption
在 16GiB 专家缓存容量限制下，对比 MoE 模型 SSD 卸载推理的两套缓存调度方案。相比旧逐层均分缓存、无预测策略，全新共享缓存搭配跨层预取机制将专家缓存命中率从 74.5% 提升至 89.3%，解码吞吐由 10.89t/s 上涨至 13.54t/s；尽管 SSD 总读取量小幅增加，但路由预测预取机制掩盖磁盘 IO 延迟。数据证明全局共享缓存与异步预取优化思路，能够显著改善低内存设备运行超大稀疏 MoE 模型的推理速度。
:::

> 下面是更多性能数据

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/04.webp)

::: image-caption
Apple M5 Pro、4 线程环境下 molli 与 llama.cpp 的推理吞吐量对比测试，覆盖 FP16、W8、W4 多种量化精度，对 Qwen、RWKV、Yuan 等多款模型，分别统计预填充 Prompt 与 token 解码吞吐速率。蓝色 molli 在绝大多数模型的解码环节性能优势明显；灰色 llama.cpp 在部分量化预填充场景具备竞争力。不同模型、量化规格下两者性能差距波动较大，高参数量模型测试中出现内存溢出，体现两套推理框架在苹果硬件平台有着不同适配表现。
:::

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/05.webp)

::: image-caption
FP16 无损精度下 mollm 与 llama.cpp 的推理性能对照，测试包含多款主流稠密大模型。多数被测模型上 mollm 在预填充、解码两个阶段吞吐速率全面领先；仅 Qwen3.5-4B 出现预填充速度不及 llama.cpp 的情况，但解码生成性能依旧具备优势。
:::

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/06.webp)

::: image-caption
W8 量化方案 mollm 与 llama.cpp Q8_0 量化在多款稠密大模型上预填充、解码吞吐性能。中小参数量模型测试场景中，llama.cpp 预填充阶段速度普遍更优；面向交互式对话核心的解码环节，mollm 在全部被测模型上均拥有更快推理速度。数据直观体现两款推理引擎优化方向差异：llama.cpp 更侧重批量 prompt 预计算场景，mollm 针对持续生成 token 的交互式解码深度调优，更加适配在线对话等真实业务场景。
:::

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/07.webp)

::: image-caption
W4 量化下 mollm 与 llama.cpp 在多款稠密、MoE 模型上预填充与解码吞吐对比。中小参数量稠密模型测试中，llama.cpp 预填充速度更占优势，但 mollm 解码性能全面领先；针对 30B、35B 级别 MoE 混合专家模型，mollm 在预填充、解码两阶段均实现性能反超。面对 122B 超大 MoE 模型，llama.cpp 直接触发内存溢出，mollm 依托 SSD 权重卸载方案可在 48GB Mac 正常推理，凸显其面向稀疏大模型的内存调度与异步 IO 卸载优化优势。
:::

::: block-1
下面我们沿着“模型转换—图执行—内核调度—SSD 卸载—交互服务”的链路，拆开这个轻量端侧推理引擎的关键设计。
:::

## 本文目录

- 一、快速上手：从模型目录到本地聊天
- 二、整体架构：把复杂性压到离线转换阶段
  - 2.1 顶层目录透露的分工
  - 2.2 构建系统中的硬件分叉
- 三、模型转换：把 Hugging Face 目录变成运行时友好的包
  - 3.1 自动识别模型族
  - 3.2 prefill chunk 是转换与运行时之间的契约
- 四、图格式：C++只执行，不在现场猜形状
  - 4.1 Graph 是线性拓扑序节点列表
  - 4.2 OpType 覆盖 Transformer、MoE、RWKV 和 recurrent 路径
  - 4.3 动态维度用 DimExpr 烘进图
- 五、执行器：一次线性遍历背后的内存管理
  - 5.1 ExecContext 记录后端、线程池、动态 shape 与释放队列
  - 5.2 execute_graph 只做执行与释放
- 六、LLMEngine：prefill、decode 与 KV cache 的生命周期
  - 6.1 引擎配置直接暴露端侧关键旋钮
  - 6.2 KV cache 把 metadata 放进 buffer 头部
  - 6.3 prefill 按 graph_seq_len 分块
  - 6.4 decode 是单 token 快路径
- 七、内核层：围绕 ARM decode 重新打磨矩阵乘法
  - 7.1 matmul 按精度和平台分派
  - 7.2 lm_head 对 N 维切分
- 八、122B MoE：SSD 专家缓存是项目最锋利的部分
  - 8.1 只读被路由到的专家
  - 8.2 request_many 让 I/O 和计算重叠
  - 8.3 共享全局池和 Least-Stale 驱逐
- 九、Metal 后端：实验性，但路线清楚
- 十、交互层：REPL 和本地服务都围绕 KV 复用
  - 10.1 REPL 中的精确前缀缓存
  - 10.2 本地 HTTP 服务的边界也很清晰
- 十一、性能画像：decode 强，prefill 还有空间
- 十二、从工程视角看 Palm-Infra 的核心价值

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/08.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：从模型目录到本地聊天

> Palm-Infra 当前核心项目是 `mollm`，一个面向 ARM CPU 的轻量 C++ LLM 推理引擎，并带有实验性 Apple Metal 后端。

最短使用路径来自项目 README：先构建运行时，再把 Hugging Face 模型目录转换成 `.mollm` 单文件包，随后用 `mollm_chat` 启动交互。

```
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Releasecmake --build build_i8mm -j# W4 转换需要 C++ 辅助量化工具cmake --build build_i8mm --target mollm-quantize# 转换 Hugging Face 模型目录为单个 .mollm 包python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128# 启动本地聊天./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

依赖很克制：macOS/Apple Silicon 或 ARM Linux，CMake、Ninja 或 Make，Python 3，以及转换阶段需要的 `numpy`、`safetensors`。

量化模式有四类：

- `fp16` 是最直接的基线；
- `w8pc` 是 int8 weight-only；
- `w4g128` 是项目中最强调的性能路径，包更小、decode 更快；
- `w4mixg128` 则把部分敏感 tensor 保持在 W8，换取质量稳定性。

> 如果想用本地 HTTP 接口，项目也提供 OpenAI 兼容的最小服务：

```
./build_i8mm/mollm_server \    --package qwen35_4b_w4g128.mollm \    --host 127.0.0.1 --port 8080 --threads 4
```

服务端支持 `GET /v1/models` 与 `POST /v1/chat/completions`，包括 SSE streaming。它定位为本机 serving baseline，当前生成路径为确定性 `temperature=0`，请求串行执行，并保留一个精确 token-prefix KV cache。更完整的接口示例可参考项目中的 SERVER.md[1]。

## 二、整体架构：把复杂性压到离线转换阶段

### 2.1 顶层目录透露的分工

> mollm 的目录划分非常清楚：

```
engine/     LLMEngine、tokenizer、生成生命周期、Metal backendgraph/      图格式、图加载、执行器、BufferPool、mmap package loadingkernels/    matmul、attention、MoE、norm、rope、RWKV、threading、tracemodels/     Python 转换器和各模型族图构建器examples/   mollm_chat、mollm_server、mollm_bench、mollm_ppltools/      mollm-quantize 权重量化工具tests/      单元、压力、端到端与 benchmark 测试
```

这套结构背后的思路很像“离线编译 + 轻量运行时”。Python 负责识别模型类型、读取权重、构建图、写出 `.mollm` 包；C++ 运行时只做加载、调度和 kernel 执行。**这样一来，端侧运行时不需要携带庞大的框架依赖，也不需要在推理现场做复杂图优化。**

项目 README 明确写到：`.mollm` 单文件包含 graph、weights、tokenizer、chat template。对用户来说，它像一个可携带的模型镜像；对运行时来说，它是一次顺序加载和按需读取的对象集合。

### 2.2 构建系统中的硬件分叉

CMake 配置展示了 mollm 的目标平台：ARM64 是主场，Apple Metal 是可选实验后端，x86_64 只保留 scalar fallback。构建系统会探测 ARM i8mm 支持，满足条件时启用 `-march=armv8.6-a+i8mm+fp16+fp16fml`，这也解释了它在 W8/W4 decode 上的重点优化方向。

```
# 来源：CMakeLists.txtif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")    message(STATUS "Target: ARM64 (NEON enabled)")    target_compile_definitions(mollm_lib PRIVATE __ARM_NEON=1)    set(MOLLM_ARM_I8MM_FLAGS "-march=armv8.6-a+i8mm+fp16+fp16fml")    check_cxx_source_compiles("${_mollm_i8mm_probe_source}" MOLLM_ARM_I8MM_COMPILES)    if(MOLLM_ARM_I8MM_COMPILES AND MOLLM_ARM_I8MM_RUNS)        message(STATUS "ARM64 i8mm target enabled (${MOLLM_ARM_I8MM_FLAGS})")        target_compile_options(mollm_lib PRIVATE ${MOLLM_ARM_I8MM_FLAGS})    else()        message(STATUS "ARM64 i8mm target not available; using DOTPROD/NEON kernels")    endif()endif()
```

这里的设计取向很务实：**在可用硬件上吃满新指令，在不可用机器上退回 NEON/DOTPROD**。工程层面，它没有把“高性能”和“可运行”绑定死，让同一套代码能覆盖 Apple Silicon 与 ARM Linux。

## 三、模型转换：把 Hugging Face 目录变成运行时友好的包

### 3.1 自动识别模型族

> 入口脚本 `models/converter.py` 是转换链路的总调度器。

它读取 `config.json` 中的 `model_type`，再分发到对应转换模块。当前支持 Qwen3、Qwen3 MoE、Qwen3.5、Qwen3.5 MoE、Youtu-LLM MLA；RWKV7 则通过 `.pth` 路径单独处理。

```
# 来源：models/converter.pySUPPORTED_MODELS = {    "qwen3":       ("qwen3",      "convert_qwen3"),    "qwen3_moe":   ("qwen3_moe",  "convert_qwen3_moe"),    "qwen3_5":     ("qwen35",     "convert_qwen35"),    "qwen3_5_moe": ("qwen35_moe", "convert_qwen35_moe"),    "youtu":       ("mla",        "convert_mla"),}
```

这段代码看起来简单，价值却很关键。

推理引擎最怕“运行时才发现模型结构差异”。**mollm 把模型族差异放到离线转换阶段处理：不同模型的 attention 形态、MoE 参数、MLA 结构、RWKV recurrent state，都在 Python 图构建器里落成统一的 C++ 图格式**。到了执行阶段，运行时面对的是 `OpType` 与 tensor，而非一堆模型专用 Python 逻辑。

### 3.2 prefill chunk 是转换与运行时之间的契约

> 转换器默认 `DEFAULT_PREFILL_SEQ_LEN = 256`。

README 也提到：prefill 图内部以 256 token 为 chunk size 构建，但 CPU runtime 支持 dynamic prefill，短 prompt 不会被强行 pad 到 256，除非显式开启 `--static-padded`。

这点非常重要。端侧推理中，prefill 常常是延迟大头。如果每次短问题都补齐到固定 256 token，用户感知到的首 token 延迟会被浪费的 padding 放大。:mdnice-bold-italic[mollm 的做法是：图里保留符号维度表达，运行时注入真实长度；必要时再进入静态 padding 模式，给未来 NPU 或固定 shape 后端留出口。]

## 四、图格式：C++只执行，不在现场猜形状

### 4.1 Graph 是线性拓扑序节点列表

> `graph/graph.h` 里对 Graph 的注释非常直白：Graph 是一个线性、拓扑排序后的节点列表。执行器只走一遍列表，把每个节点派发到对应 kernel。**图构建与优化全部在 Python 完成。**

```
// 来源：graph/graph.h// A Graph is a linear (topologically-sorted) list of nodes. Execution walks// the list once, dispatching each node to its kernel. All graph construction// and optimisation happens in Python; the C++ side only loads and executes.//// Dynamic shape support:// 每个 GraphNode 携带 dim_expr[4]，运行时只负责把 SEQ/MUL/ADD// 根据 runtime_seq_len 求值，不做完整 shape inference。
```

这是一种对端侧运行时很友好的设计。它避开了通用深度学习框架的重型 dispatcher，也减少了 runtime shape inference 的复杂度。你可以把 Graph 理解为一张“已经排好队的工单”：**C++ worker 不需要讨论任务依赖，只要按顺序取工单、调用内核、释放临时内存。**

### 4.2 OpType 覆盖 Transformer、MoE、RWKV 和 recurrent 路径

> `OpType` 里能看到 mollm 的模型覆盖范围：常规 `MATMUL`、`RMS_NORM`、`ROTARY_EMBED`、`SDPA`，还有 `SDPA_MLA`、`GATED_DELTANET`、`MOE`、`SHORTCONV`、`RWKV7` 等。它没有追求 ONNX 那种大而全的 op 体系，反而**围绕当前支持模型裁出一组够用、可优化的算子。**

```
// 来源：graph/graph.henum class OpType : uint32_t {    INPUT = 0,    CONSTANT = 1,    MATMUL = 10,    RMS_NORM = 20,    ROTARY_EMBED = 40,    SDPA = 50,    SDPA_MLA = 51,    RESHAPE = 60,    PERMUTE = 61,    ADD = 70,    SWIGLU = 75,    GATED_DELTANET_DECODE  = 110,    GATED_DELTANET_PREFILL = 111,    MOE = 120,    SHORTCONV = 140,    RWKV7 = 150,};
```

这种“模型驱动的最小算子集”有一个直接好处：每个算子都值得手写优化。对端侧 LLM 推理来说，抽象过厚会损失可控性；抽象太薄又会让图构建难以维护。mollm 在中间取了一个工程平衡点。

### 4.3 动态维度用 DimExpr 烘进图

> GraphNode 保存 `out_shape[4]` 与 `dim_expr[4]`。静态维度直接读 `out_shape`，动态维度通过 `SEQ`、`MUL`、`ADD` 表达。运行时只要把 `runtime_seq_len` 注入执行上下文，再调用 `inject_runtime_shapes()` 即可。

```
// 来源：graph/graph.hstruct DimExpr {    int8_t  kind  = 0;   // 0=CONST, 1=SEQ, 2=MUL, 3=ADD, 4=BATCH    int32_t coeff = 0;   // MUL 时为 multiplier，ADD 时为常数项    bool is_static() const { return kind == 0; }    bool is_dynamic() const { return kind != 0; }};
```

这像是在离线编译时给每个 tensor 贴好“尺寸公式”。运行时不解方程，只代入数值。对 C++ 推理引擎而言，这能降低错误面，也让动态 prefill 的实现更可控。

## 五、执行器：一次线性遍历背后的内存管理

### 5.1 ExecContext 记录后端、线程池、动态 shape 与释放队列

> `graph/execute.h` 中的 `ExecContext` 是执行器的运行环境，里面包含 Graph、BufferPool、ThreadPool、Backend，以及 profile、trace、workspace reuse、动态 shape 注入字段和 `release_queue`。

```
// 来源：graph/execute.hstruct ExecContext {    Graph*        graph   = nullptr;    BufferPool*   pool    = nullptr;    ThreadPool*   thread_pool = nullptr;    Backend*      backend = nullptr;    int  runtime_seq_len = -1;    bool static_padded   = false;    int  padded_seq_len  = -1;    // release_queue[i] = node i 执行后可释放的节点列表    std::vector<std::vector<uint32_t>> release_queue;    std::vector<NodeProfileStat> profile_stats;};
```

执行前，`prepare_execution()` 会构建保守的 last-use release queue。执行时，中间 tensor 从 BufferPool 申请，最后一次使用后归还。

对移动端推理来说，这类生命周期管理和 kernel 优化同样重要：算得快但中间内存乱涨，依然会卡在内存墙上。

### 5.2 execute_graph 只做执行与释放

> `execute_graph()` 的注释强调“pure executor, no shape inference”。它会根据 workspace 复用策略清理非 INPUT/CONSTANT tensor，特别注意 view-like op：`RESHAPE`、`PERMUTE`、`SLICE` 可能借用上游存储，不能把借来的指针当成 owner 释放。

```
// 来源：graph/execute.cppvoid execute_graph(ExecContext& ctx) {    auto& nodes  = ctx.graph->nodes;    auto& tensors = ctx.graph->runtime.tensors;    auto* pool   = ctx.pool;    const int seq_val = ctx.static_padded ? ctx.padded_seq_len : ctx.runtime_seq_len;    const bool has_dynamic = (seq_val > 0);    // 执行前清理非 INPUT/CONSTANT 状态。    // view op 借用上游数据，不能释放真实 owner。    // materialized tensor 默认释放；静态图可选择复用 workspace。}
```

这段逻辑代表了 mollm 的另一个主题：**它愿意为“少分配、少拷贝、可复用”写很多细节代码**。端侧推理里的性能瓶颈常常来自看不见的小分配、临时拷贝和 cache miss，而非单个矩阵乘法本身。

## 六、LLMEngine：prefill、decode 与 KV cache 的生命周期

### 6.1 引擎配置直接暴露端侧关键旋钮

> `engine/engine.h` 中的 `EngineConfig` 很能说明项目重点：包路径、设备、上下文长度、线程数、权重加载方式、MoE SSD cache、I/O worker、跨层预取、dense weight lock、trace 路径、采样参数都在这里。

```
// 来源：engine/engine.hstruct EngineConfig {    std::string package_path;    Device device = Device::CPU;    int n_ctx = 4096;    int num_threads = 4;    WeightLoadingMode weight_loading = WeightLoadingMode::RESIDENT;    size_t moe_ssd_cache_bytes = 0;    int moe_ssd_io_workers = 8;    bool moe_ssd_cross_layer_prefetch = true;    bool moe_ssd_global_cache = true;    bool lock_dense_weights = true;    std::string trace_path;};
```

**其中 `moe_ssd_global_cache`、`moe_ssd_cross_layer_prefetch`、`lock_dense_weights` 是 122B MoE 能跑进 48GB Mac 的关键旋钮**。:mdnice-bold-italic[稠密权重被锁住，专家权重按需从 SSD 读]，缓存从传统逐层配额改成全局池，预取尝试把下一层可能需要的专家提前搬进来。

### 6.2 KV cache 把 metadata 放进 buffer 头部

> mollm 的 KV cache buffer 前 64 字节放 `CacheMetadata`，后面才是 key/value 数据。这种布局让 kernel 拿到 tensor 时，也能知道当前有效序列长度、最大长度、KV head 数和 head dim。

```
// 来源：engine/engine.hstruct CacheMetadata {    uint64_t current_seq_len = 0;    uint64_t max_seq_len     = 0;    uint64_t num_kv_heads    = 0;    uint64_t head_dim        = 0;    uint64_t v_head_dim      = 0;    uint64_t reserved[3]     = {0, 0, 0};};inline void* cache_data(void* data) {    return static_cast<char*>(data) + CacheMetadata::SIZE;}
```

这是一种低开销的“数据自描述”方式。它避免在每次 attention 调用时额外传一堆上下文参数，也让 cache tensor 在 prefill graph 与 decode graph 之间共享同一块物理存储。

### 6.3 prefill 按 graph_seq_len 分块

> prefill **输入可能超过图构建时的 chunk 长度。mollm 的做法是循环切块**，每个 chunk 调一次 `prefill_chunk()`，并逐步推进 `past_len_`。

```
// 来源：engine/engine.cppint LLMEngine::prefill(const std::vector<int>& token_ids) {    int n = (int)token_ids.size();    if (n == 0) return -1;    int graph_seq_len = 1;    // 从 hidden INPUT 的 out_shape 中取图的 chunk 长度    int offset = 0;    int last_token = -1;    while (offset < n) {        if (past_len_ >= cfg_.n_ctx) return -1;        int chunk_size = std::min(n - offset, graph_seq_len);        std::vector<int> chunk(token_ids.begin() + offset,                               token_ids.begin() + offset + chunk_size);        last_token = prefill_chunk(chunk, past_len_);        if (last_token < 0) return -1;        offset += chunk_size;    }    return last_token;}
```

这段实现简单，却解决了两个实际问题：图可以按固定 chunk 构建，运行时仍能处理长 prompt；短 prompt 又可以通过动态 shape 路径避免无意义 padding。对聊天场景而言，这直接影响首 token 时间。

### 6.4 decode 是单 token 快路径

> decode 阶段每次只处理一个 token：embedding、生成 RoPE cache、构造 causal mask、跑 decode graph、更新 `past_len_` 与 cache metadata、最后通过 lm_head 采样下一个 token。

```
// 来源：engine/engine.cppint LLMEngine::decode(int token_id) {    Tensor h = embed({token_id});    h.shape[1] = 1;    h.compute_strides();    Tensor cos, sin;    generate_rope_cache(1, past_len_, cos, sin);    Tensor mask = build_causal_mask(1, past_len_);    Tensor out = run_graph(graph_decode_, exec_ctx_decode_, h, mask, cos, sin);    past_len_++;    for (auto& cp : caches_) {        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;    }    return run_lmhead(out);}
```

README 中的性能表也印证了这个取向：mollm 在 W4 decode 上很强，多个模型的 decode token/s 高于 llama.cpp 对照。项目自己也承认 dense model 的 prefill 仍是主要优化目标，这种表述很诚实。

## 七、内核层：围绕 ARM decode 重新打磨矩阵乘法

### 7.1 matmul 按精度和平台分派

> `kernels/matmul.cpp` 是仓库里最大的文件之一，承担 FP32、FP16、W8、W4 等路径。

一个典型分派逻辑是：有 NEON 时走 NEON kernel，没有时走 scalar fallback。FP16 路径中，权重是 `__fp16*`，激活与输出是 `float*`，这对应推理中常见的“累加保持 FP32”策略。

```
// 来源：kernels/matmul.cppstatic void matmul_fp16_range(const float* A, const __fp16* B, float* C,                              int M, int N, int K,                              int lda, int K_weight, int ldc,                              int m_begin, int m_end) {#if HAS_NEON    matmul_fp16_neon_8x8_range(A, B, C, M, N, K,                               lda, K_weight, ldc, m_begin, m_end);#else    for (int m = m_begin; m < m_end; m++) {        float* c_row = C + m * ldc;        for (int n = 0; n < N; n++) {            float sum = 0.f;            for (int k = 0; k < K; k++) {                sum += A[k + m * lda] * (float)B[n * K_weight + k];            }            c_row[n] = sum;        }    }#endif}
```

在 decode 场景里，很多计算退化成 GEMV 或小 M GEMM，常规大矩阵乘库未必最合适。mollm 围绕“单 token、多输出通道、量化权重”的形态写专门路径，解释了它在 decode 上的优势。

### 7.2 lm_head 对 N 维切分

> `matmul_fp32_range_n` 的注释提到：当 `N >> M`，例如 `lm_head` 中 `M=1`、`N=vocab_size`，按输出维 N 切分更合理。

这类细节说明项目不是只堆 NEON intrinsic，而是从 LLM 推理的具体张量形状出发调整并行策略。

```
// 来源：kernels/matmul.cpp// 当 N >> M 时按输出维切分，例如 lm_head: M=1, N=vocab_size。static void matmul_fp32_range_n(const float* A, const float* B, float* C,                                int M, int N, int K,                                int lda, int K_weight, int ldc,                                int n_begin, int n_end) {#if HAS_NEON    matmul_fp32_neon_8x8_range(A, B + n_begin * K_weight, C + n_begin,                               M, n_end - n_begin, K,                               lda, K_weight, ldc, 0, M);#else    matmul_fp32_scalar_range(A, B + n_begin * K_weight, C + n_begin,                             M, n_end - n_begin, K,                             lda, K_weight, ldc, 0, M);#endif}
```

这就是端侧推理优化的“颗粒度”：不是简单喊多线程，而是判断哪一维值得切、哪一块值得常驻、哪个临时 buffer 可以复用。

## 八、122B MoE：SSD 专家缓存是项目最锋利的部分

### 8.1 只读被路由到的专家

> README 中最吸引人的数据是：**Qwen3.5-122B-A10B W4 能在 48GB Apple Silicon Mac 上运行**。:mdnice-bold-italic[方法是把始终需要的 dense weights 锁在 RAM，只从包文件中读取被路由到的 MoE expert pair]。RAM cache 保存近期使用过的 SSD-backed expert pair，并受 `--ssd-cache-mb` 控制。

`kernels/moe_ssd.h` 的类注释把这个机制说得很清楚：

```
// 来源：kernels/moe_ssd.h// CPU-side, demand-paged storage for the two matrices belonging to a routed// MoE expert. Converter emits byte-accurate slices in `moe_expert_storage`.// This class reads only selected slices with pread(2) and retains recently// used expert pairs in an LRU cache.struct MoeSsdTensorSpec {    std::string weight_ref;    int layer = -1;    int num_experts = 0;    int rows = 0;    int cols = 0;    Precision precision = Precision::FP16;    uint64_t data_offset = 0;    uint64_t data_bytes = 0;    uint64_t scales_offset = 0;    uint64_t scales_bytes = 0;};
```

MoE 的稀疏激活特性给了这个设计空间：每个 token 只会路由到少数专家。mollm 把专家矩阵切成可定位的 byte slice，运行时根据 router 结果读取对应专家。这样，122B 的“总参数量”不再等价于“每轮都要常驻内存的参数量”。

### 8.2 request_many 让 I/O 和计算重叠

> `MoeSsdCache` 提供 `request_many()` 与 `acquire()` 两阶段接口。

`request_many()` 在 router top-k 后立即提交缺失专家读取任务，`acquire()` 到真正需要某个 expert pair 时再等待。两者之间，运行时可以继续做 shared expert 或 cache hit expert 的计算，从而隐藏 SSD 延迟。

```
// 来源：kernels/moe_ssd.h// router top-k 后立即排队，让磁盘读取与已命中专家/共享专家计算重叠bool request_many(const MoeSsdTensorSource* gate_up,                  const MoeSsdTensorSource* down,                  const std::vector<int>& experts);// speculative reads 优先级更低，服务于跨层预测bool prefetch_many(const MoeSsdTensorSource* gate_up,                   const MoeSsdTensorSource* down,                   const std::vector<int>& experts);
```

具体实现里，`request_many_impl()` 会去重专家 id，查找缓存命中；没命中的 entry 被 reserve 后进入 I/O 队列。真实请求会更新 active layer 和 miss 统计，speculative 请求则进入跨层预取统计。

```
// 来源：kernels/moe_ssd.cppbool MoeSsdCache::request_many_impl(const MoeSsdTensorSource* gate_up,                                    const MoeSsdTensorSource* down,                                    const std::vector<int>& experts,                                    bool speculative) {    std::vector<uint8_t> seen((size_t)gate_up->spec.num_experts, 0);    std::vector<Entry*> queued_entries;    std::lock_guard<std::mutex> lock(mutex_);    if (!speculative) active_layer_ = gate_up->spec.layer;    for (int expert : experts) {        if (expert < 0 || expert >= gate_up->spec.num_experts || seen[(size_t)expert]) continue;        seen[(size_t)expert] = 1;        Entry* entry = find_entry_locked(gate_up, down, expert);        if (entry) {            entry->used_at = ++clock_;            entry->forward_epoch = forward_epoch_;            continue;        }        entry = reserve_entry_locked(gate_up, down, expert, speculative);        if (!entry) continue;        if (speculative) ++cross_layer_experts_;        else ++misses_;        queued_entries.push_back(entry);    }}
```

这套机制的性能结果很直观：README 给出的 16GiB cache 下，旧的逐层均分 cache、无预测 decode 为 10.89 t/s；默认共享 cache + 跨层预取达到 13.54 t/s，expert-cache hit rate 从 74.5% 提升到 89.3%。SSD 读取量略增，交互 decode 速度却提升，因为更多 I/O 被藏在计算后面。

### 8.3 共享全局池和 Least-Stale 驱逐

传统逐层 cache 配额容易浪费：有些层暂时不需要很多专家，另一些层的路由窗口却很紧。:mdnice-bold-italic[mollm 默认使用全局 RAM pool，配合 Least-Stale eviction 保护当前与未来层的专家。这个策略贴近 MoE decode 的时间结构：当前层马上要用，下一层很快要用，刚过期且未来短期用不到的 entry 更适合被驱逐。]

这也是 mollm 价值最高的地方：它把 SSD offload 做成一套可分析系统，把缓存、预取、驱逐、trace 和 benchmark 都放了进去。

## 九、Metal 后端：实验性，但路线清楚

> **mollm 的主战场仍是 CPU，README 也把 CPU runtime 称为 portable baseline**。不过仓库中有 `engine/metal_backend.mm`、`graph/metal_pool.mm`、`kernels/metal/mollm.metal`，CMake 可通过 `MOLLM_METAL` 打开 Apple Metal 后端。

Metal kernel 里已经出现 Metal 4 tensor API 的 W8A8 GEMM 路径。注释强调 cooperative tensor register layout 在不同编译器间有可移植性风险，因此使用 tensor `.load/.store` 路径。

```
// 来源：kernels/metal/mollm.metal// W8A8 int8xint8->int32 GEMM:// C[M,N] = A_i8[M,K] * W_i8[N,K]^T,// 然后反量化为 fp32 输出。kernel void gemm_w8a8_i8a_i8b_f32c(    device const int8_t*  A       [[buffer(0)]],    device const int8_t*  B       [[buffer(1)]]) {    // 省略：tensor API 加载、int32 累加、scale 反量化}
```

这条路线的意义在于未来空间：Apple Silicon 的 GPU/NPU 能力在增强，Metal 4 tensor API 让量化 GEMM 有机会走更高吞吐路径。**不过项目 Roadmap 也承认，Metal 的量化 prefill、MoE prefill、CPU/GPU 同步开销仍需优化。**

## 十、交互层：REPL 和本地服务都围绕 KV 复用

### 10.1 REPL 中的精确前缀缓存

`examples/main.cpp` 的多轮对话逻辑并不复杂，却很谨慎。

每轮用户输入后，程序把完整 history 重新套 chat template 得到 token 序列，再与 `cached_token_ids` 比较公共前缀。如果前缀不匹配，就 reset engine，避免把新 prompt 追加到错误 KV cache 后面。

```
// 来源：examples/main.cppbool run_turn_multi(LLMEngine& engine, const Tokenizer& tokenizer,                    std::vector<ChatMessage>& history,                    std::vector<int>& cached_token_ids,                    const std::string& user_input, int max_new_tokens) {    history.push_back({"user", user_input});    std::vector<int> full_prompt_ids = tokenizer.apply_chat(history);    size_t prefix = common_prefix_len(cached_token_ids, full_prompt_ids);    if (prefix < cached_token_ids.size()) {        engine.reset();        cached_token_ids.clear();        prefix = 0;    }    std::vector<int> prompt_delta(full_prompt_ids.begin() + (ptrdiff_t)prefix,                                  full_prompt_ids.end());    return generate_greedy(engine, tokenizer, prompt_delta, max_new_tokens,                           tokenizer.eos_id(), result, error, on_token,                           /*reset_context=*/false);}
```

这里的“精确”二字很关键。**文本看起来一样，不代表 token 一定一样**；history 重新渲染后若发生差异，KV cache 就不能复用。**mollm 选择 token equality 作为缓存正确性的边界，性能让位于正确性。**

### 10.2 本地 HTTP 服务的边界也很清晰

> `SERVER.md` 说明服务端是 dependency-free 的本地 baseline，支持 OpenAI compatible chat completions 和 SSE streaming。它串行请求，保留一个 engine KV/GDN state 与上一请求的精确 token prefix。若下一个请求扩展该 prefix，只 prefill suffix；若 divergent，则 reset。

这套策略适合单用户本机体验，也给未来多 session cache 留了路线：engine state snapshot/restore、容量统计、LRU 驱逐、隔离不同会话。项目发布清单中特别提醒：在认证、TLS、请求限制、取消、并发、多 session cache 隔离完成前，不应描述为 production-ready。

## 十一、性能画像：decode 强，prefill 还有空间

> README 的基准数据有几个清晰结论：

- FP16：mollm 在 Qwen3.5-0.8B、Youtu-LLM-2B、RWKV7-1.5B 上 prefill 与 decode 都表现强；Qwen3.5-4B 上 llama.cpp prefill 更快，mollm decode 更快。
- W8：mollm 多数情况下 decode 更快，prefill 常被 llama.cpp 领先。
- W4：mollm decode 优势更明显，尤其是小模型和 RWKV mixed W4。
- MoE W4：Qwen3.6-35B-A3B、Qwen3-30B-A3B 的 prefill/decode 对照都很有竞争力；Qwen3.5-122B-A10B 在 48GB Mac 上通过 SSD offload 跑起来，llama.cpp CPU baseline OOM。

::: block-1
**这说明 mollm 当前最成熟的定位是：端侧、ARM、低 batch、本地交互、decode 优先。它适合“用户在本机与模型持续对话”的场景**。:mdnice-highlight[prefill 优化、Metal 量化路径和 serving 级 prefix cache 是它下一阶段的自然增长点。]
:::

项目还提供 `--trace <path.json>` 输出 Chrome Trace / Perfetto 时间线，**可以观察 prefill/decode、逐层 MoE routing、expert compute、cache request/acquire、SSD worker 合并后的 `pread`。** 这让性能问题从“猜测”变成“看时间线”：哪一层 I/O 没盖住，哪个 worker 队列太浅，哪个 expert cache 命中率不足，都能被定位。

![文章插图](/articles/2026-07-23-48gb-mac-122b-palm-infra-fbd98ab2/images/09.webp)

::: image-caption
该时序图呈现 MoE 大模型 decode 解码阶段运行剖面，横轴为微秒级时间刻度。顶层主线运行 decode 任务，内部 MoE 模块执行过程中频繁出现 acquire 同步锁等待，多次打断计算流水线；下方多条线程并行发起 pread 预读 IO，并发加载专家权重。尽管 IO 预读与模型计算实现并发调度，但频繁锁竞争造成计算间断，持续大量 IO 加载带来显著开销，暴露出 MoE 推理时专家权重加载存在 IO 密集瓶颈，同步阻塞损耗算力，制约解码吞吐，是 MOE 推理优化的关键方向。
:::

## 十二、从工程视角看 Palm-Infra 的核心价值

> mollm 的价值不在于发明某个单点算法，而在于把端侧 LLM 推理所需的多个层面接起来：

1. **格式统一**：Hugging Face 目录进入 Python 转换器，输出 `.mollm` 单文件包。
2. **运行时轻量**：C++ 执行线性图，图构建和 shape 传播在离线阶段完成。
3. **内存可控**：BufferPool、last-use release、workspace reuse、KV metadata header 降低运行时抖动。
4. **硬件贴合**：ARM NEON、FP16FML、dot-product、i8mm 与 W4/W8 packed layout 服务于 decode 热路径。
5. **超内存模型可运行**：MoE expert 通过 SSD demand paging、共享缓存和跨层预取进入端侧交互。
6. **可观测**：Perfetto trace 把 I/O、路由、计算、cache 行为放到同一条时间线上。

如果把传统推理框架比作一辆功能齐全的重卡，mollm 更像一辆为山路改装过的小车：载重有限，但底盘、轮胎、油门响应都围绕具体道路调过。它没有覆盖所有模型，也没有把 HTTP server 包装成完整生产系统；**它把精力集中在“ARM 本地跑大模型”这件事上，并且把 122B MoE 的门槛从数据中心级内存拉到了高配个人 Mac。**

::: block-1
:mdnice-highlight[更深一层看，mollm 体现的是端侧 AI Infra 的一个趋势：模型能力继续变大，设备内存增长有限，推理系统必须学会让 RAM、SSD、CPU SIMD、GPU tensor core、离线转换器和运行时缓存共同工作]。单纯压缩权重不够，单纯 mmap 也不够；真正可用的系统需要知道哪些权重总会用、哪些专家马上会用、哪些读取可以提前发、哪些 cache entry 该被保护。
:::

Palm-Infra 目前还很年轻，README 中也列出了明确路线：优化 dense model prefill、改进 Metal 性能、实现 serving workload 的完整 prefix cache、扩展更多模型系列和视觉模型支持。

**即便如此，mollm 已经提供了一个很有参考价值的样板：端侧大模型推理可以用一套小而硬的系统，把模型格式、执行图、内核、缓存、I/O 和观测工具打通**。它的突破点不只在“跑得快”，更在于让开发者看见一条可复现、可调试、可继续优化的本地大模型工程路径。

参考资料

[1] SERVER.md: *https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/SERVER.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[端侧推理近 3 倍提速！清华北航提出基于闪存-DRAM、CPU-NPU 异构的投机解码系统 LEVER](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902088&idx=1&sn=997b7db5bc318377694f17bbb4b3fe5b&scene=21#wechat_redirect)**
- **[端侧 LLM 最优架构被彻底颠覆：MoE 全占优、宽而浅才是硬件协同终极解](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898569&idx=1&sn=91014b936dcb46c92c8b33a5e81b74b8&scene=21#wechat_redirect)**
- **[端侧LLM设计美学，手机CPU也能起飞！Meta 提出 MobileLLM-Flash：以350M~1.4B参数实现延迟与精度的帕累托最优](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898874&idx=1&sn=5018e1722ad311779e1e69b7162d3863&scene=21#wechat_redirect)**

:::
