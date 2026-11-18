**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**HPC-Ops**、SGLang、:mdnice-bold-italic[大模型推理]、算子优化、MoE

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/01.webp)

- **HPC-Ops × SGLang: High-Performance Attention, Router GEMM, and MoE Kernels from Tencent Hunyuan**
- https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang

::: related-title
**相关推荐**
:::

::: related-links
- **[超越 EAGLE3 与 DFlash！Domino 轻量级因果修正实现 SGLang 下最高 5.8× 吞吐量加速，突破推测解码质量与成本权衡](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901783&idx=1&sn=6a45c2b4f1ef05983a0bb787096cfd1c&scene=21#wechat_redirect)**
- **[1 亿美元种子轮！SGLang 核心团队创立 RadixArk，估值 4 亿美元打造开放 AI 基础设施](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900811&idx=1&sn=c0aa61f4f8ebc3fb7fc5b1e23cab5ac9&scene=21#wechat_redirect)**
- **[把 H20 榨干到底：腾讯混元 AI Infra 团队的 HPC-Ops 如何用数百行 CuTe 改写 LLM 推理底座](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901966&idx=1&sn=f85b7106b836fdb0c1761428cd699824&scene=21#wechat_redirect)**

:::

> 近日，**腾讯混元 AI 基础设施团队联合 SGLang 团队宣布，将生产级 LLM 推理算子库 HPC-Ops 的三大核心算子** ——动态调度 Attention、精度感知 Router GEMM 与融合式 MoE——:mdnice-bold-italic[正式集成至 SGLang 主分支，面向 NVIDIA Hopper 架构 GPU 优化，已在 Qwen3、Hy3、LongCat 等主流模型上完成验证。]

针对 MoE 推理的三大性能瓶颈，HPC-Ops[1] 分别给出定制化优化方案：

- **Attention 模块以持久化内核动态分配 KV 计算任务，替代传统静态拆分策略**，解决混合长度批次的负载不均与尾部延迟问题，同时融合 QK-Norm、RoPE、量化与 KV 缓存写入流程，消除冗余访存与内核启动开销；
- **Router GEMM 将 FP32 路由权重拆解为高低位两个 BF16 分量，在 BF16 张量核心上完成计算**，既获得高吞吐量，又将数值误差控制在接近 FP32 的水平，保障路由决策的准确性；
- **MoE 模块构建端到端融合流水线，省去独立 token gather 步骤**，通过任务映射均衡小尺寸专家 GEMM 的负载，并借助链式启动压缩阶段间隙，大幅降低调度与中间访存开销。

实测数据显示，在 H20 显卡上，动态 Attention :mdnice-bold-italic[相对 FlashInfer 与 FlashAttention 的最优结果平均提速 2.25 倍]；Router GEMM :mdnice-bold-italic[相对 FP32 cuBLAS 提速 1.30–3.22 倍]，最大误差仅为 TF32 的约 1/36；MoE 在 Hy3 模型上:mdnice-bold-italic[相对基线平均提速 1.08–1.21 倍。]

::: block-1
:mdnice-highlight[8 卡 H20 端到端部署中，Hy3-FP8 的 TPOT 最高降低 48.8%，TTFT 最高降低 9.0%；LongCat 场景输入吞吐量提升 5.5%–6.1%。H200 验证同样展现出一致的性能收益，且数值精度与模型输出一致性均通过校验]。本次合入将腾讯大规模生产验证的推理优化带入开源社区，双方后续将持续扩展算子能力与适配范围。
:::

## 本文目录

- 一、核心亮点
- 二、MoE 模型服务的三大性能热点路径
  - 2.1 注意力：混合长度解码的负载均衡
  - 2.2 路由 GEMM：平衡路由精度与吞吐量
  - 2.3 MoE：降低小型专家 GEMM 的周边开销
- 三、HPC-Ops 算子与 SGLang 的集成
- 四、使用指南
  - 4.1 安装
  - 4.2 注意力与 MoE 算子启用
  - 4.3 路由 GEMM 算子启用
- 五、性能评估
  - 5.1 H20 算子基准测试
  - 5.2 H200 算子验证
  - 5.3 端到端性能表现
  - 5.4 H200 服务端验证
- 六、后续规划

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/02.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、核心亮点

- **注意力算子**：在 H20 显卡上，:mdnice-bold-italic[HPC-Ops 动态调度方案的性能是其静态拆分 KV 调度方案的 2.95 倍]；在所有测试场景中，:mdnice-bold-italic[平均性能优于 FlashInfer 与 FlashAttention 中的最优结果 2.25 倍]。在 H200 上游验证中，集成的 Hy3-FP8 路径搭配 FP8 KV 缓存，:mdnice-bold-italic[输出吞吐量较 FlashAttention 提升 3.7%–5.9%。]
- **路由 GEMM**：在 H20 显卡上，HPC-Ops :mdnice-bold-italic[性能是 FP32 精度 cuBLAS 的 1.30–3.22 倍]；相对于 FP32 cuBLAS 的最大绝对误差仅为 0.00177，远低于 TF32 cuBLAS 的 0.06464。在上游 H200 LongCat-Flash 算子验证中，其性能是现有 FP32 路径的 4.31 倍。
- **MoE 算子**：在 H20 显卡的 Hy3 模型测试中，:mdnice-bold-italic[相较于 SGLang 与 vLLM 基线中的最优结果，HPC-Ops 在 TP8/EP1 配置下批次平均提速 1.08 倍，在 TP1/EP8 配置下平均提速 1.21 倍]。在上游 Qwen3/H200 算子基准测试中，8 令牌场景下性能:mdnice-bold-italic[最高达到 Triton 实现的 4.21 倍。]
- **端到端服务性能**：在 8×H20 配置下运行 Hy3-FP8，同时启用 HPC-Ops 注意力与 MoE 算子后，批次大小 4–64 场景下 TPOT 降低 15.1%–48.8%，批次大小 4–16 场景下首令牌时间（TTFT）降低 3.3%–6.0%。在 8×H20 配置下运行 LongCat-Flash-Lite-FP8，启用 HPC-Ops 路由 GEMM 后，批次大小 4–64 场景下输入吞吐量提升 5.5%–6.1%。

## 二、MoE 模型服务的三大性能热点路径

> **生产环境中的 MoE 模型服务，其负载特征往往与孤立算子基准测试中的均匀工作负载截然不同**。在同一条延迟敏感的执行路径中，同时存在长度不一的注意力计算、精度敏感的路由运算，以及稀疏的专家执行过程；长上下文、多轮对话与智能体类工作负载进一步拉大了活跃 KV 长度的分布差异。因此，:mdnice-bold-italic[服务性能不仅取决于原始矩阵乘法的吞吐量，还取决于负载均衡效果、数值精度保持能力与额外开销控制水平。]

这些约束集中体现在 MoE 模型服务的三个关键性能阶段：

| 阶段问题 | 具体表现 |
| --- | --- |
| 解码阶段中，注意力计算量随每个请求的活跃 KV 长度线性增长，混合长度的批次就成了负载均衡难题 | 注意力计算量与每个请求当前活跃的 KV 缓存长度呈线性关系；不同请求的 KV 长度差异大时，批次内各请求计算负载不均，难以高效调度 |
| 路由 GEMM 生成用于 top-k 专家选择的得分，数值上的微小波动就可能改变专家选择结果 | 路由层通过矩阵乘法输出专家得分；浮点计算精度、舍入误差或输入微小扰动，可能导致 top-k 排序结果变化，进而引发专家分配不稳定 |
| 被选中的专家处理的是规模小且分布不均的令牌组，导致元数据构建、令牌搬运、中间结果存储与内核启动等开销，甚至能与专家 GEMM 本身的计算耗时相当 | 每个专家实际处理的令牌数少且方差大；需动态构建路由元数据、跨设备/内存搬运令牌、分配临时存储空间、频繁启动小规模 GEMM 内核；这些系统开销累积后接近甚至超过专家前向计算本身耗时 |

:mdnice-bold-italic[HPC-Ops 为每个阶段定制了专属算子：为注意力设计了负载感知的调度机制，为路由 GEMM 设计了精度感知的数值方案，为 MoE 设计了融合流水线——省去独立的令牌收集步骤，降低启动开销与中间数据访存量。] 本次上游合入通过 SGLang 原生的后端与调度接口，将这些算子与 SGLang 服务运行时深度整合。下文将详细介绍每个算子的设计思路。

### 2.1 注意力：混合长度解码的负载均衡

> 解码阶段，每个新生成的令牌都需要访问请求的完整 KV 缓存，因此注意力计算量与序列的活跃长度正相关。:mdnice-bold-italic[一个缓存了 16K 令牌的请求，其 KV 计算量大约是 1K 缓存请求的 16 倍]。**在生产环境中，提示词与输出长度差异极大，而连续批处理机制会将生成阶段不同的请求放入同一次内核启动**，因此批次中经常同时存在短 KV 缓存与数万令牌长度的长序列。

静态 KV 拆分调度会按照 KV 头、请求与 KV 分块的固定网格分配计算任务，整个批次共用同一种分区策略。**静态拆分调度通常采用两种方案，但在混合长度批次下表现都不理想：**

1. 固定拆分数量：长请求对应的分块计算量会远大于短请求，处理短请求的线程块簇（CTA）早早完成任务，少数运行时间长的 CTA 决定了整个内核的尾部延迟。
2. 固定分块大小：计算网格必须为最长的请求预留足够的拆分份数，短请求对应的分块要么为空、要么计算量极小，却依然占用调度槽位。

前者导致负载不均，后者则调度了实际不存在的计算任务，两者都会造成计算资源浪费。

#### 2.1.1 基于活跃 KV 计算量的动态调度

> HPC-Ops 摒弃了按请求静态拆分的方案，**采用持久化内核，根据批次的实际长度分布，在各 CTA 之间动态均衡 KV 计算分块。**

对于每个解码批次，分配内核会根据活跃 KV 长度构建全局任务映射：将所有序列切分为统一的 64 令牌分块，统计所有头与所有请求的总分块数，再除以持久化 CTA 的总数，得到每个 CTA 的分块任务配额。

::: block-1
分配内核会逐个填满每个 CTA 的任务槽，达到配额后再分配给下一个 CTA；长序列会按长度比例分配到多个 CTA 上，短序列则仅占用实际存在的分块数。同时设置了最小计算量下限，避免总计算量过小时过度拆分，保证后续合并步骤的开销可控。任务映射在每个解码步骤中根据设备端的序列长度生成一次，在所有 Transformer 层间复用，摊薄了生成开销。
:::

**执行阶段，每个 CTA 依次处理分配给自己的任务队列**。对于每个任务描述符，CTA 会对一个或多个连续的 KV 分块执行注意力计算，输出带有对数和指数（log-sum-exp）统计量的部分结果；常驻的 CTA 处理完一个描述符后直接继续下一个，直到任务队列为空。:mdnice-bold-italic[由于单个 CTA 只生成某个请求的部分结果，最终的合并内核会读取每个请求、每个头的实际分块数量，在全局 softmax 的正确归一化规则下合并所有部分结果。各 CTA 的任务规模基本持平，因此几乎同时完成计算，彻底消除了少数超长请求导致的内核尾部延迟。]

#### 2.1.2 融合式注意力前置处理

针对 Hy3 FP8 模型，HPC-Ops 融合了 QKV 投影后的注意力前置处理流程：在 RoPE 编码前执行 QK 归一化，输出带有逐令牌、逐头缩放因子的 FP8 精度 Q 矩阵，并将 K、V 直接写入分页式 FP8 缓存。量化后的 Q 矩阵与其缩放因子会直接传递给主注意力内核，避免了重复量化。这条融合路径省去了中间张量的存储，消除了预填充与解码阶段对应的 HBM 往返读写与独立内核启动开销。

### 2.2 路由 GEMM：平衡路由精度与吞吐量

> 路由计算的精度直接影响 MoE 模型的最终效果。每一层 MoE 中，路由层将隐藏状态投影为各专家的得分，基于得分的 top-k 选择决定了哪些专家会被执行。第 k 名与第 k+1 名专家的得分差异往往很小，因此投影运算的数值精度直接决定了专家选择的正确性。

为了保证路由精度，部分生产模型即使隐藏状态采用 BF16 精度，路由权重仍保留 FP32 格式。将权重转为 BF16 可以利用 BF16 张量核心的高吞吐量，但会丢失尾数低位信息，足以改变 top-k 的选择结果。而完整的 FP32 GEMM 虽然能保留全部权重精度，张量核心的计算吞吐量却更低。

#### 2.2.1 精度感知的 BF16 实现方案

**HPC-Ops 通过将 FP32 权重拆解为两个 BF16 分量解决了这一矛盾**。直接截断 FP32 权重得到高位 BF16 分量 $W_{\mathrm{high}}$，再将残差缩放 256 倍得到低位 BF16 分量 $(W - W_{\mathrm{high}}) \times 256$。原始权重可近似为 $W \approx W_{\mathrm{high}} + W_{\mathrm{low}} / 256$，因此矩阵乘法可拆解为两次 BF16 GEMM，对结果做缩放修正后，即可恢复尾数低位的贡献。

单个内核即可完成两次 BF16 乘法运算：从共享内存中读取一次激活分块，在 FP32 寄存器中累加两个部分结果，在输出阶段执行 1/256 缩放，最终将 FP32 精度的路由得分写入全局内存。该方案在 BF16 张量核心上执行主体计算的同时，能达到接近完整 FP32 GEMM 的数值精度。

框架层面，SGLang 会**在模型加载时缓存拆解后的权重对，在所有请求与 CUDA 图重放中复用**。调度器会根据矩阵形状自动选择 HPC-Ops 内核或默认路径：:mdnice-bold-italic[当矩阵尺寸小于实测的交叉点时，单次 FP32 运算的速度更快，因为两次乘法的开销超过了张量核心带来的收益。]

### 2.3 MoE：降低小型专家 GEMM 的周边开销

> 解码阶段，MoE 层的每个专家仅能收到少量令牌，对应的专家 GEMM 尺寸小、访存受限，GPU 流多处理器（SM）的利用率很低。**负载不均进一步加剧了这一问题**：路由到每个专家的令牌数不仅专家间差异大，还会随解码步动态变化，很难将这些大小不一的小型计算块均匀分配到所有 SM 上。

除了专家 GEMM 本身，周边操作也带来了大量开销。传统 MoE 执行路径是多个独立内核的串联：路由计算、令牌收集到各专家缓冲区、Gate-Up 矩阵乘法、激活与量化、Down 矩阵乘法、top-k 加权归并回令牌原位置。其中收集步骤会在 HBM 中实例化完整的令牌张量后才开始矩阵运算，后续每个阶段也都有独立的内核启动开销与中间结果的 HBM 往返读写。当 GEMM 本身计算量很小时，这些周边开销会占据整个阶段相当比例的耗时。

#### 2.3.1 面向延迟的融合式 MoE 流水线

> 针对小批量推理场景，HPC-Ops 的 MoE 后端以任务映射驱动的持久化专家 GEMM 为核心，将路由与索引预处理、Gate-Up 运算、激活与重量化、Down 运算，以及 top-k 加权归并整合为一条低延迟流水线。

1. **路由与索引构建**：基于选中的 top-k 专家 ID，通过共享内存计数将令牌-专家分配整理为连续的各专家输出区间，降低全局原子操作压力，同时生成持久化专家 GEMM 直接消费的路由索引与分块任务映射。
2. **Gate-Up 与激活运算**：Gate-Up GEMM 通过路由索引直接读取原始令牌，省去了独立的收集步骤与额外的 HBM 访存。SiLU 激活与乘法、FP8 重量化被融合为单个内核，其输出直接供 Down GEMM 读取。
3. **优先占优，无线程束专用化**：单个线程束组同时负责数据搬运与矩阵计算，而非拆分独立的生产组与消费组。这提升了 CTA 的驻留数量，将访存延迟隐藏从 CTA 内的软件流水线转为跨 CTA 的硬件调度。持久化计算网格消费这些任务映射，将大小不均的小型专家计算块均匀分摊到所有 SM 上。
4. **PDL 链式启动阶段**：程序化依赖启动（Programmatic Dependent Launch）机制让下游内核的启动与上一阶段的尾部执行重叠，压缩了 Gate-Up、激活、Down 运算与最终 top-k 加权归并（将专家输出恢复为令牌原顺序）之间的执行间隙。

这些优化共同降低了关键路径上的中间数据访存与内核启动开销。

## 三、HPC-Ops 算子与 SGLang 的集成

> 通过 SGLang 原生的后端与调度接口，HPC-Ops 可以直接操作服务运行时的现有状态，同时保持为独立维护的算子库。

- 注意力算子直接消费分页式 KV 存储与设备端活跃序列元数据，无需额外的格式转换；
- 路由 GEMM 在不同请求与 CUDA 图重放间复用预处理后的权重与工作空间；
- MoE 算子直接沿用 SGLang 的专家 ID 与分区规则，无需重新映射。

这些集成方案在适配 SGLang 现有执行模型的同时，完整保留了各算子的最优数据路径。三款集成算子的核心信息汇总如下：

| HPC-Ops 算子 | 优化方向 | 精度 | 上游 PR 编号 |
| --- | --- | --- | --- |
| 注意力（Attention） | 混合长度解码的负载均衡；融合 QK 归一化、RoPE、量化与 KV 写入前置流程 | 激活值 BF16；KV 缓存 BF16 或 FP8 E4M3 | #30540[2]、#32304[3] |
| 路由 GEMM（Router GEMM） | 精度感知的路由投影，在 BF16 张量核心上计算的同时保留 FP32 权重信息 | 激活 BF16 × 权重 FP32 → 得分 FP32 | #30247[4]、#31943[5] |
| MoE | 降低小型、不均专家 GEMM 的执行开销 | 隐藏状态 BF16；专家权重 FP8 E4M3 | #30541[6] |

## 四、使用指南

> 本指南介绍如何在 SGLang 中使用 HPC-Ops[7] 的注意力、路由 GEMM 与 MoE 算子。

### 4.1 安装

从源码安装 HPC-Ops 的命令如下：

```
git clone https://github.com/Tencent/hpc-ops.gitcd hpc-opsmake wheelpython3 -m pip install dist/*.whl
```

HPC-Ops 已预装在 SGLang 官方的 x86_64 开发镜像中（`lmsysorg/sglang:dev`，CUDA 12.9 版本为`lmsysorg/sglang:dev-cu12`），使用这些镜像无需单独安装。

### 4.2 注意力与 MoE 算子启用

> 注意力与 MoE 在 SGLang 中是独立的后端选项，在 Qwen3、Hy3 等兼容模型上可单独启用，也可同时开启。以下示例同时启用两个 HPC-Ops 后端，并开启 FP8 KV 缓存的注意力路径：

```
python3 -m sglang.launch_server \  --model tencent/Hy3-FP8 \  --tp-size 8 \  --attention-backend hpc_ops \  --kv-cache-dtype fp8_e4m3 \  --page-size 64 \  --moe-runner-backend hpc_ops
```

若使用 BF16 KV 缓存，去掉`--kv-cache-dtype fp8_e4m3`参数即可。若只需使用某一款 HPC-Ops 算子，仅指定对应的后端参数即可。

### 4.3 路由 GEMM 算子启用

> 在 SGLang 中，HPC-Ops 路由 GEMM 既能保留 FP32 路由权重的低位信息，又能在 BF16 张量核心上执行矩阵运算。

该集成路径已在 LongCat-Flash Chat 与 Lite 版本上完成验证，对于支持的模型与路由形状会自动启用。安装 HPC-Ops 后，标准的 LongCat-Flash 启动命令即可自动使用该算子：

```
python3 -m sglang.launch_server \  --model meituan-longcat/LongCat-Flash-Lite-FP8
```

## 五、性能评估

HPC-Ops 后端目前支持 NVIDIA Hopper 架构 GPU，在 H20 显卡上优化效果最佳。以下评估涵盖 H20 上的算子基准测试、8×H20 上的 SGLang 端到端服务性能，以及 SGLang 上游 PR 中公布的 H200 测试结果。

### 5.1 H20 算子基准测试

#### 5.1.1 注意力算子

> **注意力调度的核心优势体现在混合长度解码场景——同批次内请求的 KV 缓存长度差异极大**。我们测试了从均匀到高度倾斜的多种 FP8 KV 缓存解码场景，表格中`A×B`表示 A 个请求、每个请求的 KV 长度为 B。为单独验证调度机制的效果，:mdnice-bold-italic[我们对比了 HPC-Ops 动态调度与其自身的静态拆分 KV 方案，同时以 FlashInfer 与 FlashAttention 作为额外基线。]

**动态相对静态的收益随负载倾斜程度提升而增大**：均匀的 64×0.5K 批次中两者持平，在 1×128K + 31×4K 的混合场景中达到 2.95 倍。:mdnice-bold-italic[在全部六个测试场景中，动态调度的平均性能比每个场景下 FlashInfer 与 FlashAttention 的最优结果快 2.25 倍。]

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/03.webp)

::: image-caption
表 1：H20 上不同 KV 长度分布的解码延迟（数值越低越好）以及动态相对静态的加速倍数。测试涵盖均匀长度与长短文本混合的复杂请求，在短序列场景下动态与静态 HPC‑Ops 耗时持平；存在长文本的混合场景中，动态 HPC‑Ops 优势逐步放大，加速倍数最高达 2.95 倍，耗时远低于其余方案，可见动态 HPC‑Ops 在长短混杂的真实业务负载中，能显著降低解码时延，性能表现更优。
:::

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/04.webp)

::: image-caption
图 1：活跃 KV 计算量越倾斜，动态调度的优化效果越显著，数值越低越好。测试覆盖均匀 KV 长度与长短请求混合场景，短请求下动态与静态 HPC‑Ops 性能基本持平；当场景混入长 KV 序列，动态 HPC‑Ops 时延优势持续扩大，最高实现 2.95 倍加速，整体时延显著优于另外两种推理方案，验证动态调度在长短混杂负载下的性能优势。
:::

#### 5.1.2 路由 GEMM

> 我们首先在通用尺寸 $K = 4096, N = 192$ 下对路由 GEMM 进行了全面测试。**在所有测试的 M 维度取值中，HPC-Ops 性能是 FP32 cuBLAS 的 1.30–3.22 倍，是 TF32 cuBLAS 的 1.25–1.78 倍**。以 FP32 cuBLAS 为数值基准，HPC-Ops 的最大绝对误差不超过 0.00177，而 TF32 cuBLAS 的最大误差为 0.06464。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/05.webp)

::: image-caption
表 2：H20 上 K=4096、N=192 时 BF16×FP32 路由 GEMM 延迟（数值越低越好），单位为微秒，同时给出 HPC‑Ops 相对二者的加速比。随着 M 增大，三者耗时均同步上升，而 HPC‑Ops 耗时始终低于 cuBLAS。相较 FP32 cuBLAS，加速比区间 1.30‑3.22 倍；对比 TF32 cuBLAS，加速比可达 1.25‑1.78 倍。无论小 M 还是大 M 场景，HPC‑Ops 均展现出更优性能。
:::

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/06.webp)

::: image-caption
图 3：左为 LongCat-Flash Chat 版本、右为 Lite 版本的路由 GEMM 延迟，测试范围为 SGLang 调度区间。数值越低越好。随 M 维度增大，两套方案时延均上升，HPC‑Ops（蓝色）时延始终低于 SGLang 默认（灰色）。小 M 下加速比较低，M 增大后加速效果显著，Chat 最高 2.83 倍，Lite 最高 2.46 倍，证明 HPC‑Ops 在大 M 矩阵运算场景下具备突出性能收益。
:::

#### 5.1.3 MoE 算子

> MoE 测试基于 Hy3 模型尺寸，**分别在 TP8/EP1 与 TP1/EP8 两种配置下测试完整融合运算的性能，对比基线包括 SGLang、vLLM Triton 与 vLLM CUTLASS。** 以每行中三个基线的最低延迟为基准，HPC-Ops :mdnice-bold-italic[在 TP8/EP1 配置下的批次平均提速为 1.08 倍，TP1/EP8 配置下为 1.21 倍，在低延迟解码常见的中小批次尺寸下收益最大。]

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/07.webp)

::: image-caption
表 4：H20 上 TP8/EP1 配置下 Hy3 MoE 延迟（数值越低越好），HPC‑Ops、SGLang、vLLM Triton、vLLM CUTLASS 的算子时延，单位为微秒，最后一列为 HPC‑Ops 相对其余最优方案的加速比。随 Batch 增大，各方案时延同步上涨，绝大多数 Batch 规模下 HPC‑Ops 耗时最低。在中等 Batch 区间，加速效果更突出，最高可达 1.18 倍；小批量与超大批量场景加速逐步收敛至 1.00‑1.03 倍。整体来看，HPC‑Ops 在各批次下性能均有优势，在中等并发场景收益最为明显。
:::

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/08.webp)

::: image-caption
表 5：H20 上 TP1/EP8 配置下 Hy3 MoE 延迟（数值越低越好），单位微秒，最后一列是 HPC‑Ops 对比其余方案最优值的加速比。随着 Batch 增大，各方案时延整体上升。中小 Batch 场景 HPC‑Ops 优势明显，Batch=64 时加速最高达 1.60 倍；Batch 增大到 512 后性能基本持平甚至略低。整体上 HPC‑Ops 在中小并发批次时延更低，更适配中小批量推理业务场景。
:::

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/09.webp)

::: image-caption
图 4：Hy3 模型在 TP8/EP1 与 TP1/EP8 两种配置下的 MoE 延迟对比，数值越低越好。随着 Batch 增大，所有方案时延均上升，两种并行设置下 HPC‑Ops（蓝色斜纹柱）在绝大多数 batch 规模时延都低于其他实现。中小批次场景性能优势尤为突出，大批次下各方案差距缩小，验证 HPC‑Ops 在 MoE 推理中，不同并行策略下均可带来时延优化。
:::

### 5.2 H200 算子验证

> 上游 PR 同时包含了 H200 上的测试结果，证明性能收益在 Hopper 架构 GPU 上具备通用性。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/10.webp)

::: image-caption
表 6：SGLang 上游 PR 中公布的算子验证结果，分别为 FP8 Attention、BF16 Attention、Router GEMM 与 MoE 算子，列明测试负载、对比基线与测试结果。动态调度 Attention 相比静态 split‑KV，吞吐量提升 2%‑3%，TTFT、E2E、TPOT 时延均下降；Router GEMM 内核加速 1.56‑4.31 倍；MoE 算子对比 SGLang Triton 融合专家实现，加速区间 0.89‑4.21 倍。整体体现 HPC‑Ops 在吞吐量、首字时延、算子速度上均有正向收益，部分极端场景性能存在小幅回落。
:::

### 5.3 端到端性能表现

端到端评估在 8×NVIDIA H20 GPU 上开展，以 SGLang 对应默认实现为基线。针对 Hy3-FP8（TP8 配置、FP8 KV 缓存），我们测试了同时启用 HPC-Ops 注意力与 MoE 的联合服务效果；针对 LongCat-Flash-Lite-FP8，单独测试了路由 GEMM 的收益。下文同时汇总了上游 PR 中公布的 H200 服务验证结果。

#### Hy3-FP8：注意力+MoE 联合优化

> 在 8K 输入、4K 输出的配置下，批次大小为 1 时 HPC-Ops 将 TPOT 降低 3.3%；批次大小 4–64 范围内，优化幅度提升至 15.1%–48.8%。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/11.webp)

::: image-caption
表 7：启用 FP8 KV 缓存与 HPC-Ops 注意力+MoE 后，Hy3-FP8 的 TPOT 表现（数值越低越好），单位为毫秒，最后一列为 HPC‑Ops 带来的性能提升比例。随着 Batch 增大，两者时延均逐步增加。Batch=1 时提升仅 3.3%；当 Batch 提升至 32 时优化效果达到峰值 48.8%；Batch 继续增至 64，提升回落至 41.5%。可见 HPC‑Ops 在中等批量场景收益最为显著，可大幅降低推理时延，小 batch 场景收益相对有限。
:::

> 输入长度为 8K 时，HPC-Ops 在批次大小 1–16 范围内将 TTFT 优化了 3.3%–9.0%。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/12.webp)

::: image-caption
表 8：8K 输入下，Hy3-FP8 搭配 FP8 KV 缓存的 TTFT 表现（优化幅度为正代表延迟降低），SGLang 默认版本与 HPC‑Ops 的推理时延，单位为毫秒，最后一列为性能提升比例。随 Batch 增大，两者时延均显著上涨。Batch 为 1 时优化效果最好，时延降低 9.0%；Batch 为 4、8 时提升分别为 4.9%、6.0%；Batch 到 16 时提升回落至 3.3%。可见在长文本负载下 HPC‑Ops 全量 Batch 均带来正向收益，但整体提升幅度相比短文本场景更小，小批量收益相对更突出。
:::

> 我们还在批次大小 16 的条件下，测试了 2K 到 8K 的不同输入长度（关闭分块预填充与前缀缓存），HPC-Ops 在三种输入长度下将 TTFT 优化了 2.3%–8.9%。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/13.webp)

::: image-caption
表 9：批次大小 16 时，不同输入长度下 Hy3-FP8 搭配 FP8 KV 缓存的 TTFT 表现（优化幅度为正代表延迟降低）。随着输入序列从 2K 增长至 8K，整体推理时延显著变大。输入长度 2K 时 HPC‑Ops 优化收益最高，性能提升 8.9%；4K、8K 长输入场景提升分别为 2.3% 与 3.3%。可以看出 HPC‑Ops 对中等长度输入优化效果更好，输入越长，相对性能增益有所收窄，但依然保持正向收益。
:::

#### LongCat-Flash-Lite-FP8：路由 GEMM 优化

> 路由 GEMM 单独测试采用 1024 令牌输入、128 令牌输出的配置。批次大小为 1 时输入吞吐量基本持平，优化幅度 0.5%；批次大小 4–64 范围内，输入吞吐量提升 5.5%–6.1%。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/14.webp)

::: image-caption
表 10：启用 HPC-Ops 路由 GEMM 后，LongCat-Flash-Lite-FP8 的输入吞吐量（数值越高越好）。随着 Batch 增大，两者 token 吞吐整体上升并逐步趋于饱和。Batch=1 时提升仅 0.5%；Batch 为 4 时提升达到 6.1%；Batch 从 8 至 64 区间，性能稳定维持在 5.5%‑5.7%。可见 HPC‑Ops Router GEMM 在小 batch 收益有限，在中高并发场景可以稳定带来约 5.5% 以上的吞吐增益。
:::

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/15.webp)

::: image-caption
图 5：端到端 SGLang 性能结果。左上、右上、左下为 Hy3-FP8 搭配 FP8 KV 缓存、同时启用 HPC-Ops 注意力+MoE 的结果，数值越低越好；右下为 HPC-Ops 路由 GEMM 的单独测试结果，数值越高越好。在不同 Batch、输入长度下，HPC‑Ops 均实现时延下降，部分场景降幅接近 48.8%。右下角为 LongCat‑Flash‑Lite‑FP8 的 Router GEMM 吞吐测试，数值越高越好，中高 Batch 下吞吐提升稳定在 5.5%‑6.1%。整体可见 HPC‑Ops 对首字时延、生成时延、算子吞吐均有优化效果。
:::

### 5.4 H200 服务端验证

> 上游 PR 同时在 H200 的 SGLang 服务链路中评估了集成算子，在 H20 主力调优目标之外，完成了模型级的集成一致性校验。

![文章插图](/articles/2026-08-09-hpc-ops-sglang-1aaf93c0/images/16.webp)

::: image-caption
表 11：SGLang 上游 PR 中公布的模型级服务验证结果。Attention 算子以 FlashAttention 为基线，在 Hy3‑FP8 服务负载下，输出吞吐量提升 3.7‑5.9%。Router GEMM 对比 SGLang 默认实现，在 LongCat‑Flash Lite 预填充场景，输入吞吐量提升 2.8‑5.4%。MoE 算子分别测试 Qwen3、Hy3‑FP8，Qwen3 吞吐持平至提升 2.7%；Hy3 波动区间‑4.2%~+6.3%。整体上多数场景获得正向吞吐收益，部分 MoE 负载下存在性能小幅下降。
:::

> 上游集成同时通过了数值精度与模型效果的一致性校验：

- 注意力算子在 BF16 与 FP8 精度下全部测试通过，Hy3 FP8 的贪心解码输出与 BF16 路径逐令牌完全一致；
- 路由 GEMM 通过了与 FP32 基准的对比，贪心输出结果保持一致；Qwen3 模型的 HPC-Ops MoE 路径与 Triton 实现相对 FP32 的误差水平相当，余弦相似度达 0.99974，最大相对误差为 0.024。完整配置与各场景结果可查看上游 PR。

## 六、后续规划

本次工作是 HPC-Ops 与 SGLang 社区深度合作的一部分。双方将继续与 SGLang 维护者、贡献者协作，**持续优化与扩展现有算子，并将更多成熟的 HPC-Ops 能力逐步上游合入**。欢迎社区提交反馈、问题与基准测试结果，期待与大家共同推动开源高性能大语言模型推理技术的发展。

参考资料

[1] HPC-Ops: *https://github.com/Tencent/hpc-ops*

[2] #30540: *https://github.com/sgl-project/sglang/pull/30540*

[3] #32304: *https://github.com/sgl-project/sglang/pull/32304*

[4] #30247: *https://github.com/sgl-project/sglang/pull/30247*

[5] #31943: *https://github.com/sgl-project/sglang/pull/31943*

[6] #30541: *https://github.com/sgl-project/sglang/pull/30541*

[7] HPC-Ops: *https://github.com/Tencent/hpc-ops*

::: related-title
**相关推荐**
:::

::: related-links
- **[SGLang 中的流水线并行：扩展至百万 Token 上下文及更长远景规划](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896233&idx=1&sn=27be78c67c4a9c7d3fa2cca23d8348bc&scene=21#wechat_redirect)**
- **[比真 GPU 硬件快5-17 倍的时间跳跃仿真框架 REVATI！实现 vLLM/SGLang 跨框架无 GPU 加速仿真，预测误差低于 5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896002&idx=1&sn=7b81c53ef9a19c14aab1b6bb1d0a5361&scene=21#wechat_redirect)**
- **[训练即部署！SpecForge：加速 SGLang 推测性解码训练，让16GB GPU支持70B模型训练，加速比达2.18倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892155&idx=1&sn=5c2c834d047a75a4da3e1b52ff14a30d&scene=21#wechat_redirect)**

:::
