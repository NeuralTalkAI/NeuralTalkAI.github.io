关键词：**Luminal 编译器**、HLIR、ShapeTracker、**egraph 饱和搜索**、CUDA Graph

> 先前我们推送过不少 Mega Kernel 的文章，今天来说这篇：**[无需手动构建MegaKernels！Luminal 编译生成 MegaKernels：解决 GPU SM 负载不均，消除内核启动开销与内存气泡，适配任意架构！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896119&idx=1&sn=80dce399f083de55a8cded74e2a54c3d&scene=21#wechat_redirect)** 本文作者郑启航剖析了这款开源编译器 Luminal，结合其在 H200 上运行 gemma-3-4b 的实测，梳理其 IR 设计与搜索机制。

编译流程分六步：前端用 `GraphTensor` 描述计算，通过 `ShapeTracker` 记录 layout 信息，从而消除大量显式形状操作，最终生成仅含 20 个 primop 的高层 IR HLIR。全图按 `graph_break` 切成 chunk，结构相同的 chunk 合并为 group，以 group 为单位进行 egraph saturation 等价性搜索，生成包括 CUDA kernel、cuBLAS 调用在内的候选方案，再通过实测时延筛选最优实现。提取出的低级 IR LLIR 经模板参数替换和 NVRTC 即时编译生成 GPU 可执行码，并由 Runtime 构建 CUDA Graph 执行推理。

::: block-1
实测显示，Luminal 的 fp32 **推理吞吐远逊于 vLLM，且当前并未实现宣传中“自动导出 FlashAttention”的融合**。作者指出其缺少内存层级描述与 tiling 优化，认为其宣传目标与实际进展存在落差，质疑其“编译器”定位。
:::

![文章插图](/articles/2026-04-27-flashattention-luminal-f5533bda/images/01.webp)

- **Luminal 编译器浅析**
- 作者：郑启航，https://zhuanlan.zhihu.com/p/2031495391997317887
- 个人主页：zhen8838.github.io
- 4000 字，阅读 20 分钟，**播客 21 分钟**

相关推荐

- **[无需手动构建MegaKernels！Luminal 编译生成 MegaKernels：解决 GPU SM 负载不均，消除内核启动开销与内存气泡，适配任意架构！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896119&idx=1&sn=80dce399f083de55a8cded74e2a54c3d&scene=21#wechat_redirect)**
- **[端到端 LLM 编译器 nncase：基于 e-graph 的异构存储架构高性能统一编译框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895279&idx=1&sn=cfe132ed7791edfd649561561681c72d&scene=21#wechat_redirect)**
- **[ASPLOS'26 高频语义等价指令挖掘，实现2.6倍加速、93%面积节省！用E-Graph反合一为AI芯片打造乐高式指令！寻找高频数据级并行指令模式](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899424&idx=1&sn=e2a2f9e94eeeefe1858087e094837b18&scene=21#wechat_redirect)**
- **[CGO'25 杀死阶段顺序问题：基于 MLIR 的等式饱和持久化 e-graph，让等价信息贯穿编译全程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897696&idx=1&sn=d963ddf62be45438f2c7c6804545af64&scene=21#wechat_redirect)**

> 我去年就关注到 Luminal 编译器。它宣称通过全自动编译可以达到 80% 的峰值性能，并且能够搜索出 FlashAttention，之后还获得了投资。最近我跑了它的 `gemma-3-4b` 示例，顺便梳理了它的 IR 设计和搜索机制。

## 本文目录

- 一、整体概览
- 二、HLIR
  - 2.1 ShapeTracker
- 三、Partition / Group
- 四、Egglog saturation
  - 4.1 单个 HLIR primop 匹配 kernel op
  - 4.2 多个 HLIR primop 对应高效库调用
  - 4.3 Batch 和 shape 展平
  - 4.4 In-place 候选 + aliasing 检查
  - 4.5 Saturation
- 五、Extraction
- 六、LLIR
  - 6.1 Gemma 3 4B 的 LLIR
- 七、Code Generation
  - 7.1 模板 + 参数
  - 7.2 库调用
- 八、Runtime
  - 8.1 Load 阶段
  - 8.2 Execute 阶段
- 总结

![文章插图](/articles/2026-04-27-flashattention-luminal-f5533bda/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、整体概览

> Luminal 的编译流程大致分为六步：

1. **Frontend**：用户通过 `GraphTensor` API 编写算子（如 `matmul`、`softmax`）。前端的 `expand_dim`/`permute` 等操作仅修改张量附带的 `ShapeTracker` 元数据，不会生成新 op，因此最终 HLIR 图中的 op 数量远少于用户表达式的节点数。
2. **HLIR**：前端操作最终凝聚成一张由 20 个 primop 组成的张量 DAG，这就是 Luminal 自己的高层 IR。
3. **Partition / Group**：按前端插入的 `graph_break` 把整张 HLIR 切成若干 chunk，再把结构相同的 chunk 合并成 unique group。后面几步都以 group 为单位推进。
4. **Egglog saturation**：将每个 group 序列化为 egglog 程序，执行等价关系饱和。4B 模型在单核 CPU 上约需 30 分钟，是编译开销的主要部分。
5. **Extraction / LLIR**：从饱和后的 egraph 中先提取候选方案，然后 lower 到 LLIR。
6. **Codegen / Runtime**：每个 LLIR 节点先 Codegen 成 CUDA kernel（或 cuBLAS 调用），再由 Runtime 将它们串进 CUDA Graph 执行推理。整体更像 JIT：kernel 编译在 Codegen 阶段即时发生，buffer 分配在 Runtime 启动时完成，而不是提前全编译好的 AOT。

## 二、HLIR

> HLIR 是 Luminal 的高层张量 IR，只有 20 个 primop，代表最小的原子运算。一个 Gemma 3 4B 模型的 HLIR 大约包含 5000 个 primop。

20 个 primop 分七类：

| 分类 | Ops |
| --- | --- |
| I/O | Input, Output, Constant |
| DType / Range | Cast, Iota |
| Unary | Exp2, Log2, Sin, Recip, Sqrt |
| Binary | Add, Mul, Mod, LessThan |
| Reduction | SumReduce, MaxReduce, Softmax |
| Indexing | Gather, Scatter |
| Fallback | CustomOpKind |

以 matmul 为例：`a: [M, K] @ b: [K, N] -> [M, N]`。HLIR 里没有显式的 `for k` 循环，对应的前端代码如下：

```
// src/frontend/matmul.rslet mul = self.expand_dim(1, n) * rhs.permute((1, 0)).expand_dim(0, m);let ret = mul.sum(2);
```

代入具体 shape 后，构造出的 HLIR 只有 5 个节点。

![文章插图](/articles/2026-04-27-flashattention-luminal-f5533bda/images/03.webp)

它的设计很类似 Jittor，通过扩展 layout 的方式表示循环区域。观察上面的 `Mul` 和 `SumReduce` 节点：input 节点的 rank 是 2 维，但 `Mul` 使用 `dims=[2, 4, 3]`，两个输入的 `strides` 分别是 `[(z*3), 0, z]` 和 `[0, z, (z*4)]`（其中 `z` 是 `sizeof(dtype)`）。stride 里的 `0` 就是 `expand_dim` 制造出来的 broadcast 维度。没有单独的 Shape 操作 op，基本都由 `ShapeTracker` 来表达。

值得注意的是，`Softmax` 并没有被拆分为 `Exp2 + SumReduce + Div`，这应该是为了便于后续的 rewrite 和 pattern match 而做出的妥协。

### 2.1 ShapeTracker

> `ShapeTracker` 的主要作用是替代显式的 `Expand`/`Reshape`/`Permute` 等 op。可以理解为：它先记录 Layout 信息，再在后续计算中应用，从而表达这些形状操作。其工作流程大致是：

1. 每个 `GraphTensor` 都挂着一个 `ShapeTracker`，里面记录当前的 `dims`、`strides`、`offset`、`mask` 等影响访问顺序的信息。
2. `expand_dim`、`permute`、`reshape`、`slice` 这类前端函数只修改这个 `ShapeTracker`，不会往 HLIR 图里插入新节点。
3. 等到真正创建计算 op（`Mul`、`Add`、`SumReduce`）时，当前 `ShapeTracker` 会被读出，固化进这个 op 的输入签名。

因此上面的例子中，HLIR 包含的是一个带着 `shape`/`stride` 信息的 `Mul`，而不是 `Expand -> Permute -> Mul`。

具体到几种常见操作：

- `expand_dim`：往 `dims` 里插入一维，对应的 stride 置为 `0`，表示广播。
- `permute`：重排 `dims` 和 `strides`，表示只是改变了观察顺序，没有搬移数据。
- `reshape`/`slice`：更新 `dims`、`offset`、`mask` 等视图信息，仍然不新建 HLIR op。

## 三、Partition / Group

> HLIR 构建全图后，直接对整图做 egg 搜索代价过高，尤其对于 Transformer 这种结构高度重复的模型，也没有必要。因此这一步做两件事：

1. **Partition**：把整张 HLIR 切成若干 **chunk**，即“一整块子图，内部一起搜索/编译”。切分点由前端显式指定（`graph_break`），典型地放在 transformer 每层边界或 KV cache 更新处这种天然分界。
2. **Group**：再把结构完全一致的 chunk 合并成同一个 **group**。每个 group 只做一次 egraph 搜索，结果供所有 member chunk 共用。

以 Gemma 3 4B 在 H200 上为例，规模数据如下：

| 层级 | 数量 | 说明 |
| --- | --- | --- |
| chunk | 35 | 整张图切出 35 块，每块约 140 个 HLIR op |
| group | 5 | 35 块按结构去重后剩下 5 类模板 |

这 5 个 group 对应模型结构分别为：

- **1 个 decoder layer group**：34 层 decoder layer 全部共享这一套模板，这是去重收益的主要来源。
- **1 个 embedding group**：处理 token lookup 这一块。
- **1 个 final norm + logits group**：处理模型最后的输出头。
- **2 个辅助 group**：对应 prefill / decode 入口、RoPE / mask 等不属于主干 decoder layer 的块。

## 四、Egglog saturation

这一步使用 egraph saturation 技术，对 HLIR 做等价变换与优化，产生大量等价实现候选。搜索主要完成四件事：

### 4.1 单个 HLIR primop 匹配 kernel op

每个 HLIR op（`Add`、`Mul`、`SumReduce`、`Exp2`……）都有对应的 `kernel_rewrite<HLIR, Kernel>` 规则，将其扩展为 dialect 级 `KernelOp`（CUDA `KernelAdd`、Metal 对应 op 等）。17 个 HLIR 计算 op 各有一条这样的 rewrite（`crates/luminal_cuda_lite/src/kernel/hlir.rs`）。这一步把纯 HLIR op 变成“可以真正执行的候选”。

最小的那类 rewrite，在代码里其实就是一个通用 helper：

```
pub fn kernel_rewrite<H: Default + EgglogOp, L: Default + EgglogOp>() -> Rule {    ...    rule(union(hlir_op.clone(), llir_op)).fact(eq(dt, dtype(hlir_op)))}
```

它的做法很直接：看到一个 HLIR op，就把它和对应的 `KernelOp` union 到同一个 eclass 里。例如 `Mul` 可以 rewrite 成 `KernelMul`，`Add` 可以 rewrite 成 `KernelAdd`。

### 4.2 多个 HLIR primop 对应高效库调用

像 `Mul + SumReduce` 这种 pattern（即 matmul）会被单独识别，并 lower 到 cuBLAS / cuBLASLt 的 sgemm 变体。规则命名如：`cublas sgemm row-major x column-major`、`cublaslt batched column-major x row-major`（`crates/luminal_cuda_lite/src/host/cublas/` + `cublaslt/`）。同一 pattern 根据 shape / stride 可以匹配到不同的库变体。

这类高阶 pattern 的 rewrite 大致长这样：

```
(rewrite  (Op (SumReduce ...) (ICons (Op (Mul ...) ...) (INil)))  (Op (CuBlasSGemm ...) ...)  :name "cublas sgemm row-major × column-major")
```

### 4.3 Batch 和 shape 展平

基于 Layout 做一些化简，涉及 `src/egglog_utils/matmul_flattening/*.egg`（三条规则）：

- `batch_merge_a_contig.egg` / `batch_merge_b_contig.egg`：把“batch × matmul，其中一侧 contiguous，另一侧 broadcast”展平成 2D matmul。
- `squeeze.egg`：压缩无效维度。

### 4.4 In-place 候选 + aliasing 检查

`Scatter` 会被 rewrite 成 `ScatterNoCopy(ConsumedBuffer(dest), ...)`。这里的 `ConsumedBuffer` 并不是具体操作，而是搜索阶段的所有权标记。因为 egraph 中的节点可能出现环状依赖，且难以收集 user 数量，所以 `ConsumedBuffer` 的作用就是把 usage analysis 显式纳入搜索空间：如果 `dest` 这个 buffer 从此不再被他人读取，就可以原地写回。

后面的 `cleanup` / `base_cleanup` ruleset 就是在检查这一点：

- 如果 `dest` 后面**没有** 别的 reader，就保留 `ConsumedBuffer(dest)`，最终允许走 `ScatterNoCopy`，即原地写。
- 如果 `dest` 后面**还有** 别的 reader，就把这个候选删除，退回普通 `Scatter`。

### 4.5 Saturation

Luminal 并没有把所有 rewrite rules 混在一起执行，而是分为 4 个 ruleset 分阶段搜索，以缩小每轮 rewrite 的匹配空间，降低编译代价：

- `expr`：主 rewrite，HLIR 对应 kernel 候选、batch matmul 展平、ConsumedBuffer 注入等全在此。
- `dtype_prop`：侧函数 `(function dtype (IR) DType :merge new)` 沿 dataflow 传播 dtype 的规则。
- `cleanup`：如果 `dest` 被别的 op 读到，删除 `ConsumedBuffer`，级联清除 ScatterNoCopy 候选。
- `base_cleanup`：独立 ruleset 放在最后，专门处理 `(union ?cb ?dest)` 这类不可逆操作，必须等前面都饱和后才安全。代码里有 admitted TODO 承认这是脆弱点。

实际执行顺序为：

```
(repeat 10 (saturate expr) (saturate dtype_prop) (run))(saturate expr)(saturate cleanup)(saturate base_cleanup)
```

在我的实验中（Gemma 3 4B，H200，34 层 transformer），34 层被切为 35 个 chunk，合并为 5 个结构等价的 group，每个 group 的 egraph saturation 产生约 5076 个 enode、3633 个 eclass，单 CPU 核耗时 30 分钟。

## 五、Extraction

> saturation 后，Luminal 直接通过实测时延来获取真正的开销：

1. **随机选择**：对每个 eclass 随机选一个 enode，lower 成 LLIR，用 NVRTC 编译，实际执行并测量延迟（默认 10 次取平均），同时检查结果是否有 NaN。编译失败或出现 NaN 就换一套，最多重试 100 次，全部失败会直接 panic（`src/graph.rs:653`）。
2. **变异**：以当前最快的候选作为种子（默认保留 1 套），每代生成 30 个变异：在存在多个可选 enode 的 eclass 中，随机选取几个替换为其他选择，并通过哈希去重避免重复测量。
3. **评估**：每个变异同样 lower + 编译 + 执行 + 测量。跑得比种子快就顶替种子。
4. **预算**：**每个 group** 最多评估 `options.limit` 个候选（Gemma 3 4B 有 5 个 group，`GEMMA_SEARCH_GRAPHS=3` 即每 group 3 个候选，全模型共 5 × 3 = 15 次 NVRTC + profile）。官方默认 500，搜索时间会很久。

这种方法用实测取代了传统分析建模中难以精确预测的成本模型，但只适用于稳定的硬件环境，在设计阶段无法使用。

## 六、LLIR

> 代码里将 LLIR 定义为：

```
pub type LLIRGraph = StableGraph<LLIROp, ()>;
```

`StableGraph` 可以简单理解为一个节点编号稳定的图容器。`LLIROp` 是节点内容，边表示依赖关系。dump 出来的内容大致如下：

```
LLIROp(DialectOp(KernelMul { out_shape: [4, s, 256], ... }))LLIROp(DialectOp(KernelSumReduce { out_shape: [s], ... }))LLIROp(DialectOp(CuBlasLt { m: 1024, n: s, k: 2560, ... }))
```

其中每个节点直接对应一个具体的执行单元，例如：

- CUDA kernel 源码（之后由 NVRTC 实时编译成 GPU 可执行码）
- Metal kernel（Apple backend）
- host 上的库调用（cuBLAS、cuBLASLt 等现成的 sgemm）

同时包含供 Runtime 使用的元信息：

- 输出 buffer 的大小（符号表达式，支持动态 shape）
- 读写字节数、计算 FLOPs
- 输出是否复用某个输入 buffer（in-place 写）等

在 LLIR 这一层，可以认为节点间均通过 **global memory** 传递数据，而 `shared memory`、`register` 等更细粒度的层次不会反映在 LLIR 中。

### 6.1 Gemma 3 4B 的 LLIR

> Gemma 3 4B 编译生成的 LLIR 约 7250 个节点，其中：

```
KernelMul          2043        KernelGather        205KernelAdd           810        KernelSin            68KernelIota          648        KernelScatter        66KernelCast          438        KernelLessThan       63KernelConstant      409        KernelExp2           35KernelRecip         378        KernelExp            35KernelSumReduce     375        KernelMaxReduce      34KernelSqrt          205        KernelSigmoid        32                               KernelScatterNoCopy   2
```

elementwise 操作占绝大多数。仅有的 2 个 `KernelScatterNoCopy` 全部用于 KV cache 的原地写。这就是前面 `ConsumedBuffer` 机制的效果：只有当 buffer 没有多个使用者时，egglog 才会保留普通 `Scatter` 的 `ScatterNoCopy` 版本。

## 七、Code Generation

LLIR 本身只是一段数据，GPU 无法直接执行。Luminal 的处理方式是：

### 7.1 模板 + 参数

每种 kernel op 自己维护一份 C++ kernel 模板，Codegen 时把节点里的 shape、stride、dtype 等参数填入，生成一段具体的 CUDA 源码，然后将这段源码交给 NVRTC 做 JIT 编译，得到 GPU 真正能执行的 kernel。没有 loop-level IR、schedule pass 或 tiling，模板长什么样，kernel 就长什么样。

比如一个 `KernelAdd` 节点，codegen 实际做的就是模板替换，最后拼成一段完整的源码：

```
extern "C" {__global__ void add_k(float *C, const float *A, const float *B, const int* dyn_dims) {  long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;  if (const_z >= /* n_elements */) return;  C[/* out_idx */] = A[/* a_idx */] + B[/* b_idx */];}}
```

为避免同样的 kernel 反复编译，Luminal 会按生成的源码做进程内缓存，当两个节点最终拼出的源码完全相同时，直接复用已编译好的 function。

### 7.2 库调用

当然，并非所有 LLIR 节点都需要生成源码。前面搜索部分提到 `Mul + SumReduce` 被 rewrite 为 `matmul`，最终对应的是 cuBLAS / cuBLASLt 入口的 wrapper。这类节点的 Codegen 只是选择一个合适的库入口，将 stride / leading dimension 填成 cuBLAS 可支持的格式，执行时直接调用 `cublasSgemm` 这类 host 函数。

## 八、Runtime

> 由于代码已在上一阶段编译完毕，交给 Runtime 的是一组互相独立的 kernel 和库调用。Runtime 的工作主要分为两个阶段：

- `load_llir`：先把每个 group 的 LLIR 装配好，设置 input / output 指针，分配中间 buffer，并捕获成 CUDA Graph。
- `execute`：每步推理按 chunk 顺序 replay 对应的 CUDA Graph，取出输出。

### 8.1 Load 阶段

> Load 阶段首先读取每个 group 的 LLIR，然后：

**1. 分配 buffer**：Runtime 遍历 LLIR 中每个节点，根据节点的输出大小表达式，代入当前 `dyn_map`（如 `M=1024, N=4096`）计算所需字节数。再与已有 buffer 比较：若容量足够则复用，否则调用 `cudaMalloc` 分配新 buffer。

节点还可能携带输出可复用某个输入 buffer 地址的标记。此时 Runtime 直接将 output pointer 指向该 input pointer，不再额外分配。前面提到的 `KernelScatterNoCopy`（KV cache 原地写）正是这样处理的。

**2. 打包 CUDA Graph**：每个 group 单独处理，Runtime 按 LLIR 顺序排好该 group 内的 kernel，调用 CUDA Graph API 将整段 launch 序列捕获成一张图。在 Gemma 3 4B 上我总共构建了 **5 张 CUDA Graph** （每个 group 一张），每张图内部封装 12 ～ 180 个 kernel。执行时一次 `cuGraphLaunch` 即可发出整段序列，减少 launch overhead。

### 8.2 Execute 阶段

此阶段只需：

1. 提供输入数据指针。
2. 按 chunk 顺序发起各 chunk 所属 group 的 CUDA Graph。
3. 如有需要，将 output buffer 读回 host。

## 总结

> 先把我最后跑出来的数据放在一起：

| 框架 | dtype | TTFT | TPOT | TPS |
| --- | --- | --- | --- | --- |
| vLLM | bf16 | — | 3.71 ms | 269 |
| vLLM | fp32 | — | 5.81 ms | 172 |
| Luminal main | fp32 | 202 ms | 37.42 ms | 26.7 |
| Luminal fusion | fp32 | 250 ms | 48.13 ms | 20.8 |

再对比 Luminal 官方在 `README.md` 中的宣传：

- 性能上，它写的是 **Q8 Llama 3 8B 在 H100 上能到 ~80% theoretical max performance**
- 技术上，它写的是这套搜索 **可以自动导出 FlashAttention**

然而，从我这次 Gemma 3 4B fp32 在 H200 上的实测来看，这两点都很难成立：

- 实际的 TPS 与 vLLM 差距还很大
- 当前代码里也没有真正把 attention 融合成 FlashAttention 的路径，它的 egraph saturation 中并没有任何 rule 能跨过 `Softmax` Op 然后 rewrite 得到 FlashAttention。

我的几点看法是：

1. 缺少对 Bufferization / Memory Hierarchy 的描述
2. 缺少 fusion / tiling / scheduling 的优化
3. 早期宣传自动生成 FlashAttention，但实际上输入与规则都是精心设计好的[1]。并且之前的 IR 设计与现在大相径庭，原先带有 `LoopOut, Let` 等，可以表示复杂程序，但规则难写、搜索空间更大。现在又退回到类似 linalg 的纯算子 IR，这就很难支持之前宣传的自动生成 FlashAttention 了。

::: block-1
至少到本文撰写时，最新的 PR 依然在做 elementwise fusion[2]。以这样的开发进展，显然很难匹配其宣传目标和投资规模[3]，我怀疑这是在挂“编译器”的羊头，卖手动优化的狗肉。
:::

参考资料

[1] flash_attention_demo/src/code.lisp: *https://github.com/luminal-ai/luminal/blob/0ccd344a69226205f1992f43f0dc3ef590bd56b2/flash_attention_demo/src/code.lisp*

[2] luminal-ai/luminal/pull/274: *https://github.com/luminal-ai/luminal/pull/274*

[3] Luminal raises $5.3 million to build a better GPU code framework: *https://techcrunch.com/2025/11/17/luminal-raises-5-3-million-to-build-a-better-gpu-code-framework/*
