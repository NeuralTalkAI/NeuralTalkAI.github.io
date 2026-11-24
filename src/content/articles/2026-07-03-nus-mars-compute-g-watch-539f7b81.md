**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：G-Watch、Intra-kernel Tracing、Agent 驱动优化、异构 AI 硬件、GPU 内核优化、小batch优化

> 在 AI 算力竞速的当下，硬件的理论峰值早已不是性能瓶颈——真正的天花板，藏在 GPU 内核每一个执行阶段的缝隙里。

传统性能分析工具只能给出 Kernel 级的端到端延迟，却无法剖开数十微秒的执行过程：访存占比多少、计算是否饱和、流水线气泡藏在何处、哪一组线程拖慢了整体节奏。**内核优化至今仍高度依赖专家经验与反复试错，效率与扩展性都已逼近极限。**

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/01.webp)

- :mdnice-bold-italic[模型与芯片之间的优化层。MarsCompute 是面向异构人工智能硬件的自适应优化层，可在英伟达、超威及混合加速集群环境下持续优化延迟、运行效率与小批量处理性能。]
- https://mars-compute.com/
- https://g-watch.github.io/

::: related-title
**相关推荐**
:::

::: related-links
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**

:::

> 2026 年 6 月，Mars Compute 正式开源 G-Watch，为 GPU 内核装上了“性能黑匣子”。这套由新加坡国立大学何丙胜教授团队打造的工具链，首次将内核级追踪能力覆盖 CUDA、Triton、TileLang 等多类 DSL，同时以标准化的 Agent 技能接口，让 AI Agent 能够自主完成“画像-假设-优化-验证”的完整闭环。

在 Flash Attention 全系列的生产级验证中，它在 A100 上实现最高 50.8%的性能提升，**即便在趋近优化极限的 Blackwell 架构上仍能稳定挖掘出 3%的额外增益。**

::: block-1
接下来，我们将从 GPU 内核优化的行业痛点切入，全面解析 G-Watch 的核心能力、设计逻辑与技术路径，以及其背后 Mars Compute 面向异构 AI 硬件的完整优化版图。
:::

## 本文目录

- 一、GPU 内核优化为什么难
- 二、G-Watch 是什么
  - Mars Compute 是谁
  - Mars Compute 产品架构
  - 产品矩阵：G-Watch 只是一角
  - 研究方向
- 三、杀手锏：Intra-kernel Tracing (Iket)
  - 3.1 使用流程
  - 3.2 Human View vs Agent View：双视图设计
- 四、多 DSL 一等公民
- 五、Agent 驱动的优化闭环
  - 实战验证：Flash Attention 全系列优化
- 六、Roadmap 技术路线
- 总结

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/02.webp)

交流加群请在

NeuralTalk 公众号

后台回复：

加群

## 一、GPU 内核优化为什么难

> 现有 profiler 告诉你 kernel 跑了 50μs，却不告诉你这 50μs 里 load、compute、epilogue 怎么分配，气泡藏在哪。

GPU 内核优化至今仍是一门手艺活。工程师盯着 profiler 数据，靠经验猜测瓶颈在哪——是访存 bound 还是计算 bound？pipeline bubble 有多大？哪个 warp 在拖后腿？这些问题在聚合指标里找不到答案。

::: block-1
更关键的是，**现有的 profiling 工具只能给出 kernel 级别的端到端延迟**，无法呈现 kernel 内部各阶段的时序关系。你优化了一个 kernel，延迟从 50μs 降到 45μs，但你不知道这 5μs 省在了哪个阶段，也不知道是否引入了新的气泡。
:::

2026 年 6 月，Mars Compute 开源了 G-Watch，试图改变这一局面。

## 二、G-Watch 是什么

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/03.webp)

::: image-caption
https://g-watch.github.io/en/blog/humanize/introducing-g-watch/
:::

> 一套工具链 + 一套 Agent 技能 = 让 coding agent 获得"看懂 GPU 内核"的能力。

G-Watch 是一套面向**Agent 驱动的 GPU/TPU 内核优化** 的开源工具链，由 Mars Compute 开发和维护。

Agentic Optimization of FlashAttention with G-Watch

### Mars Compute 是谁

> Mars Compute 定位于 **"模型与芯片之间的自适应优化层"（The Optimization Layer Between Models and Silicon）**。

公司由新加坡国立大学（NUS）何丙胜教授（Prof. Bingsheng He）[1] 领导，团队由编译器、内核和 AI 系统领域的专家组成。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/04.webp)

::: image-caption
https://scholar.google.com.sg/citations?user=RogYLKYAAAAJ&hl=en
:::

Mars 项目的历史可追溯到 2007 年——CUDA 诞生同年——何教授在 GPU 加速计算方向的开创性工作。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/05.webp)

::: image-caption
《Efficient gather and scatter operations on graphics processors》为 2007 年 ACM/IEEE 超算会议论文，聚焦 GPU 环境下 gather、scatter 两类基础数据并行操作。二者原生实现因随机访存，显存带宽利用率低下、访存延迟无法隐藏，且 GPU 存储层级对开发者不透明。研究设计多轮执行方案优化数据局部性，并搭建性能模型剖析运行特性，该论文累计被引 211 次，是 GPU 通用计算访存优化的经典参考。
:::

### Mars Compute 产品架构

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/06.webp)

::: image-caption
Mars Compute 的产品架构分为三层
:::

1. **Architecture Discovery** — 自动映射目标硬件的拓扑结构、内存层级、互联行为和执行约束
2. **Optimization Planning Engine** — 构建优化图，生成跨 kernel、runtime policy 和调度路径的无冲突优化计划
3. **Execution & Continuous Adaptation** — 在生产环境中安全应用优化，持续度量效果，随工作负载和硬件演进自适应调整

::: block-1
**其目标用户是超大规模云厂商、AI 基础设施平台团队和 GPU/加速器厂商** ——绝非面向个人开发者或 hobby 项目，而是**聚焦于"基础设施关键级"的 AI 性能问题。**
:::

### 产品矩阵：G-Watch 只是一角

> Mars Compute 目前有三个层次的产品，G-Watch 是其中唯一开源的部分：

| 产品 | 定位 | 形态 |
| --- | --- | --- |
| **Mars Optimization Brain** | 核心引擎 | 闭源，设计伙伴阶段 |
| **mars-compute CLI** | 优化执行入口 | 内部工具 |
| **G-Watch** | 可观测性 + Agent 技能 | 开源（2026.06） |

表格说明：

- **Mars Optimization Brain** 是整个平台的“大脑”，采用三层自适应优化系统：自动发现硬件拓扑 → 构建 conflict-aware 优化图 → 安全执行并持续迭代。
- **mars-compute CLI** 是 Blog 中用于 FA2/FA3/FA4 优化的命令行工具，通过 profiling-guided 分析识别性能瓶颈、生成优化方案并验证效果。
- **G-Watch** 将内部工具链中的 tracing、profiling 与二进制分析能力开源，同时作为 Agent 技能的统一接入层，支撑可观测性与自动化协同。

::: block-1
三者的关系是：:mdnice-bold-italic[G-Watch 负责"看见"，mars-compute CLI 负责"分析 + 执行"，Optimization Brain 负责"持续学习 + 自适应进化"。]
:::

FA 系列博客中使用的`mars-compute` 工具 3 月就已跑通优化闭环，而 G-Watch 6 月才开源——说明 Mars Compute 先有内部产品，再将其中的可观测性层剥离出来开源，作为开发者生态入口。

### 研究方向

> 从产品架构和网站公开信息可以梳理出 Mars Compute 的几条研究主线：

1. **跨架构自适应优化** — 核心命题。将 NVIDIA / AMD / TPU 等异构硬件的拓扑差异抽象为统一的 optimization graph，让同一套优化逻辑跨平台生效。官网特别强调 AMD MI300 的"性能解锁层"定位，暗示对 ROCm 生态的深入投入。
2. **小批次推理延迟优化** — 官网多次提及 "small-batch latency" 和 "production serving paths"，指向 LLM 推理场景中的真实痛点：大 batch 吞吐容易做，:mdnice-bold-italic[小 batch 延迟才是线上服务质量的关键。]
3. **Agent-driven 自动优化** — G-Watch 是这个方向的开源载体。不只是 tool-calling（Agent 调 profiler），而是让 Agent 理解 profiler 输出、生成可验证的假设、单步修改并闭环验证。FA4 案例中 softmax exp2 单元压力分析就是这种推理能力的例证。
4. **编译器协同设计** — 从 TileLang PTX 导出方案、SASS 级分析到 CuTeDSL 集成，团队显然在与编译器链路深度协作，而非仅在 runtime 层面打补丁。
5. **持续学习系统** — Optimization Brain 的 "self-evolving" 定位暗示了长期愿景：非一次性调优，而是系统随新硬件、新驱动、新 serving pattern 的引入自动适应和进化。

> G-Watch 是 Mars Compute 开源生态的核心组件，也是外部开发者接触这套技术体系的入口。其工具链围绕三大能力构建：

| 模块 | 能力范围 | 当前状态 |
| --- | --- | --- |
| Intra-kernel Tracing (Iket) | 单 kernel 内部阶段级时间线分析 | 已发布，支持 CUDA / Triton / TileLang / CuTeDSL |
| Intra-kernel Profiling | Counter-based 性能计数器 + PC Sampling 热点定位 | 文档占位，开发中 |
| Binary Analysis | SASS/AMDGPU 二进制分析：控制流、数据流、指令调度、寄存器活性 | 文档占位，开发中 |

**硬件兼容性**：NVIDIA GPU、AMD GPU (ROCm)、Google TPU**Agent 兼容性**：Claude Code、Codex、Gemini CLI

安装方式：

```
pip install gwatchnpx skills add mars-compute-ai/G-Watch -g
```

第一条命令安装 Python 工具包（profiling + 程序分析），第二条命令将 G-Watch 的 Agent 技能分发到所有 coding agent 的全局技能库。

## 三、杀手锏：Intra-kernel Tracing (Iket)

> 在 kernel 源码里标记几个 scope，运行一次，就能看到每个 warp 在每个阶段花了多少时间——这在以前只有芯片设计级的仿真工具才能做到。

Iket 是 G-Watch 目前最完整、也最具差异化的功能模块。

::: block-1
传统 profiler 只能给出 kernel 级别的聚合指标。Iket 把视角拉进单个 kernel 调用内部，按阶段画出 per-thread 的 Gantt 时间线，精确暴露 pipeline bubble、同步开销和 warp-role 调度问题——这是目前开源工具中唯一具备此能力的方案。
:::

### 3.1 使用流程

开发者在 kernel 代码中插入 device-side scope markers，标记 load、compute、epilogue 等阶段：

```
#include "gwatch/cuda/trace.hpp"__global__ void my_kernel(...) {    for (int k = 0; k < K; k += BK) {        GWATCH_CUDA_KERNEL_SCOPE_START(10);  // load        // 数据搬移到 shared memory        __syncthreads();        GWATCH_CUDA_KERNEL_SCOPE_END(10);        GWATCH_CUDA_KERNEL_SCOPE_START(20);  // compute        // Tensor Core 计算        GWATCH_CUDA_KERNEL_SCOPE_END(20);    }    GWATCH_CUDA_KERNEL_SCOPE_START(30);      // epilogue    GWATCH_CUDA_KERNEL_SCOPE_END(30);}
```

标记完成后运行 kernel，G-Watch 通过 CUPTI hook + PTX instrumentation 自动捕获每个 scope 的起止时间戳，输出 trace 数据。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/07.webp)

::: image-caption
上图是FA4 Intra-Kernel Tracing。gwatch 工具解析 fa4_trace.gwatch CUDA 性能追踪文件的时序分析界面，日志生成于 2026 年 3 月 7 日，仅追踪单个 CUDA 内核。视图横轴为高精度时间戳，纵轴排布 TID 0 至 288 多组 GPU 线程块，各色块区分 softmax、校正、数据加载、MMA 矩阵运算等算子阶段。TID0-224 线程执行时序高度统一，依次走完 softmax 计算、校正收尾、等待、重缩放流程；仅 TID256、288 线程执行路径完全分化，以重缩放、TMA 数据搬运为主，末尾附带校正段，直观呈现出 GPU 线程块间显著的负载不均衡问题。详见：FA4 Forward Kernel Optimization Report
:::

### 3.2 Human View vs Agent View：双视图设计

> G-Watch 从架构层面区分了两类消费者：

- **Human View**：交互式 HTML 报告，包含时间线视频回放、per-scope 统计分布、pipeline 重合度分析，适合工程师诊断
- **Agent View**：`gwatch show` 输出的紧凑文本，包含 min / p50 / mean / p95 / max 延迟、bubble ratio、per-thread ASCII 时间线、straggler 线程列表，适合 AI Agent 读取

```
gwatch show trace.json --stats       # scope 统计摘要gwatch show trace.json --bubbles     # pipeline 气泡 + 最长气泡排行gwatch show trace.json --timeline    # per-thread ASCII 甘特图gwatch show trace.json --outliers    # straggler 线程定位gwatch show trace.json --json        # 结构化 JSON，供程序消费
```

这种设计的意义在于：它不简单地把 profiler 输出"翻译"成文本塞给 LLM，而是从 Agent 的视角重新设计了信息架构——删掉视觉装饰、保留量化指标、提供可编程的`--json` 出口。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/08.webp)

使用gwatch show将内核内追踪信息渲染为大语言模型智能体可读取的精简文本。图上展示的是CUDA 内核 main_kernel 的性能分析报告，单轮执行采用 96×1×1 网格、1024 线程块，总计 98304 线程，整体耗时 88.99μs。内核执行分为 load_x、load_y、mul、poly、store 五大阶段，其中 load_y 总耗时占比最高。流水线统计显示线程平均生命周期 70.37μs，气泡停顿占比 38.2%，多条 warp 气泡耗时达到线程生命周期的 52%；下方时序图直观展现各线程访存、计算阶段割裂，存在大量空白气泡区间，说明访存延迟带来严重阻塞，是该内核核心性能瓶颈。

## 四、多 DSL 一等公民

> 同一个 scope marker API，同一个 visualize 工作流。CUDA、Triton、TileLang、CuTeDSL 没有二等公民。

G-Watch 的一个关键设计决策是**DSL 无关**。不同 DSL 各有原生 marker 实现，但底层 tracing 引擎和上层可视化完全复用：

| DSL | Scope 标记方式 | 底层 instrumentation |
| --- | --- | --- |
| CUDA C++ | `GWATCH_CUDA_KERNEL_SCOPE_START / END(id)` 宏 | CUPTI + PTX |
| Triton | `gw_trace.scope_start / end(stream, id)` | Triton IR → PTX |
| TileLang | `T.evaluate(gw_trace.scope_start / end(id))` | 需显式导出 PTX + 设置`TILELANG_CACHE_PATH` |
| CuTeDSL | `cute_trace.scope_start / end(id)` | CuTe → PTX |

其中 TileLang 的处理最能体现团队对底层编译链路的理解：TileLang 编译产物 cubin 不含嵌入式 PTX（纯 SASS），常规 CUPTI module capture 会失败。

::: block-1
G-Watch 的处理方案是让用户显式调用`kernel.export_ptx()` 并设置`TILELANG_CACHE_PATH` 环境变量，tracing 引擎从缓存中加载 PTX 完成 instrumentation。这种适配不是 hack，反而对 TileLang 编译模型深思熟虑后的设计选择。
:::

## 五、Agent 驱动的优化闭环

> 并非表面性地"帮我把这个 kernel 跑快一点"，而是有基准地 profile → hypothesize → apply → verify，每一步都有 on-disk 证据支撑。

G-Watch 区别于一般 profiler 的地方在于，它提供了一套**Agent 技能**，定义了严格的推理-优化闭环：

```
Profile → Hypothesize → Apply one minimal change → Verify → 循环
```

关键约束是**每次只改一处** （apply one minimal change），且验证环节同时检查三项：

1. **正确性** — 数值输出是否与 baseline 一致
2. **性能** — 延迟 / 吞吐是否实际改善
3. **硬件行为** — profiler 数据是否印证了优化假设（例如：如果假设是"减少 shared memory bank conflict"，那优化后 SMEM 吞吐是否确实提升）

这种 evidence-driven 的设计理念，让 G-Watch 区别于简单的 prompt-to-code 脚本——它不"碰运气"，反倒构建了一套可交付生产级优化的推理框架。

### 实战验证：Flash Attention 全系列优化

> Mars Compute 已经在 Flash Attention 这一最关键的 LLM 推理 kernel 上验证了其优化能力。团队使用 `mars-compute` 工具（G-Watch 的前身/内核）对三个版本的 Flash Attention 进行了 profiling-guided 优化，全部交出正向结果：

| Kernel | GPU | 配置 | 最佳提升 | 典型提升 |
| --- | --- | --- | --- | --- |
| FA4 Fwd | Blackwell (sm100a) | MHA/GQA, causal/non-causal, bf16/fp16, 1k–32k | +3.1% | **+1.3% avg TFLOPS** |
| FA3 Fwd (Causal) | H200 PCIe (141GB) | HeadDim=128 | +4.2% | 1–3% |
| FA2 Fwd (Causal) | A100 PCIe (80GB) | HeadDim=64 | **+50.8%** | 25–35% |

三个案例有三个看点：

- **FA4 on Blackwell**：在已高度优化的 FA4 forward kernel 上，通过分析 softmax warp group 的硬件 exp2 单元压力，调整软件模拟参数，获得平均 1.3% TFLOPS 提升。改动量极小——两个 Python 文件，纯 CuTe DSL，无需重编译 C++，bf16/fp16 精度无损。优化已合并入 FA4 主仓库（commit 6017fa7）。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/09.webp)

::: image-caption
FA4 Forward Kernel优化效果对比，该图由 8 组柱状子图构成，对比基准 Original（紫色）与 Mars 优化方案（黄色）在 MHA、GQA 两类注意力、bf16/fp16 精度、因果 / 非因果掩码组合下，序列长度 1k 至 32k 的 TFLOPS 算力表现，柱顶标注 Mars 相对原始版本的性能变动比例。Mars 在绝大多数测试场景实现算力正向提升，涨幅集中 0.2%~3.1%，仅 GQA bf16 因果模式 8k 序列出现 - 2% 小幅下降；非因果场景优化增益普遍高于因果场景，8k 至 32k 长序列区间性能提升稳定，验证 Mars 对注意力算子算力具备持续正向优化效果。详见：FA4 Forward Kernel Optimization Report
:::

- **FA3 on H200**：在接近硬件峰值吞吐的 H200 上仍稳定获得 1–4% 增益，证明工具能在"油水已尽"的场景下找到余量。优化已提交为 FA3 官方 PR。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/10.webp)

::: image-caption
热力图量化了 HeadDim=128 的因果型 FA3 前向计算在 141GB H200 PCIe GPU 上，优化方案相对原始实现的性能提升百分比，横轴为 2048、4096、8192 三种序列长度，纵轴覆盖 1 至 32 的批量尺寸，色块色彩深浅对应提速幅度。绝大多数测试组合均实现正向性能增益，仅批量 4、序列长度 4096 场景出现 - 0.32% 的微小性能回落；其中批量 1、序列 8192 工况提速最高达 4.19%，小批量搭配长序列的优化收益更突出，随批量数值增大，整体性能提升幅度呈逐步走低的趋势。详见：Flash Attention Kernel Optimization Results with mars-compute
:::

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/11.webp)

::: image-caption
头维度 128 的因果型 FA3 前向计算在 141GB H200 PCIe GPU 上，原始实现与优化方案的 TFLOPS 算力对比，分为序列长度 2048、4096、8192 三组子图，横轴覆盖 1 至 32 的批量尺寸，蓝色柱为原始版本、红色柱代表优化方案。全部测试条件里优化方案算力均高于原版，固定序列长度时，批量越小两者算力差距越突出，批量提升后差值逐步收窄；不同序列长度下整体算力区间平稳，长序列场景优化依旧持续带来算力增益，直观体现这套优化手段可有效提升 FlashAttention3 前向运算的硬件利用效率。详见：Flash Attention Kernel Optimization Results with mars-compute
:::

- **FA2 on A100**：在相对老旧的 A100 上获得 25–51% 的大幅提升，最高达 50.8%，说明自动化优化对"未充分调优的 kernel"效果尤为显著。

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/12.webp)

::: image-caption
该热力图展示 HeadDim=64 的因果型 FA2 前向计算在 80GB A100 PCIe GPU 上，优化方案相对原始实现的性能提升百分比，横轴为 512 至 8192 序列长度，纵轴覆盖 1 至 64 批量尺寸，色块颜色越深代表提速幅度越高。多数测试组合实现明显正向性能增益，批量 16、序列长度 1024 场景提升峰值达 50.8%；仅少量工况出现性能倒退，如批量 2、8 搭配短序列 512/1024；序列长度 1024 搭配中大规模批次优化收益最为突出，序列拉长后整体性能提升幅度小幅回落。详见：Flash Attention Kernel Optimization Results with mars-compute
:::

![文章插图](/articles/2026-07-03-nus-mars-compute-g-watch-539f7b81/images/13.webp)

::: image-caption
该分组柱状图对比 HeadDim=64 的因果型 FA2 前向计算在 80GB A100 PCIe GPU 上，原始实现（蓝色）与优化方案（红色）的 TFLOPS 算力，分为序列长度 512、1024、2048、4096、8192 五组子图，横轴批量尺寸覆盖 1 至 64。多数工况下优化版本算力更高，仅少量极小批量短序列场景优化算力略低于原版；序列长度 1024 搭配中高批次时两者算力差距最明显，随序列拉长至 4096、8192，二者差值逐步收窄，整体算力随批量增大同步上涨，直观验证优化对算力的提升效果。详见：Flash Attention Kernel Optimization Results with mars-compute
:::

---

这些案例的核心价值在于：每次优化都不依赖工程师手工调参，而是**mars-compute 自动完成 profile → hypothesize → apply → verify 闭环**。FA4 的案例尤其说明问题——在已被世界顶级团队调优的 kernel 上，自动优化仍能稳定贡献 1–3% 的性能增益。

将上述方法论进一步抽象，就是 G-Watch 的系统性价值：

::: block-1
把 profiler 输出从人类可读的图表变成 Agent 可编程消费的结构化数据，再配合 evidence-driven 的优化闭环，让 AI Agent 具备了"看懂 GPU 内核、有理有据地优化 GPU 内核"的能力。FA2 上 51% 的增益已经证明：这条路走得通。
:::

## 六、Roadmap 技术路线

> G-Watch 目前处于早期阶段（2026 年 6 月首次发布），网站 34 个页面中 19 个标注为"Content coming soon"。但从完整的文档骨架可以清晰看到其技术路线：

**Intra-kernel Profiling** （规划中）：

- Counter-based Profiling — occupancy、pipe / tensor-core 利用率、显存吞吐、峰值算力达成率
- PC Sampling — 指令级热点 + warp stall 根因（访存等待、依赖、divergence、throttle），映射到源码行和 SASS

**Binary Analysis** （规划中，13 个子模块）：

- Control Flow 分析：Control Flow Graph、Control Divergence Graph、Warp Specialization、Software Pipeline
- Data Flow 分析：Data Flow Graph、SMEM Usage、DMEM / SMEM Access Pattern
- Hardware Execution 分析：Instruction Scheduling、RMEM Liveness、RMEM Cache

::: block-1
这些能力一旦落地，G-Watch 将从"tracing 工具"演进为 GPU 内核分析领域功能最全面的开源平台。
:::

## 总结

> 几个值得关注的判断：

1. **时机精准** — AI coding agent 在 2026 年大规模进入工程流程，但 GPU kernel 优化领域几乎还没有面向 Agent 设计的工具，G-Watch 填补了空白。
2. **公司底蕴扎实** — Mars Compute 由 NUS 何丙胜教授领导，团队在 GPU 加速计算领域有近 20 年积累（始于 2007 年 CUDA 元年）。产品定位清晰——不做 hobby 工具，:mdnice-bold-italic[聚焦超大规模云厂商和 AI 基础设施团队的生产级需求。] 👀
3. **实战验证充分** — 在 Flash Attention FA2/FA3/FA4 全系列上的优化结果（A100 上 +51%、H200 上 +4%、Blackwell 上 +3%）说明工具不是 PPT 产品，而是已在生产级 kernel 上跑出可复现的增益。
4. **技术深度罕见** — 从 PTX instrumentation、CUPTI hook、SASS 解码到 TileLang PTX 导出的边界 case 处理，:mdnice-bold-italic[文档透露出团队对 GPU 底层执行模型有实战级理解]。FA4 案例中的 softmax exp2 单元压力分析更是将这种深度展现得淋漓尽致。
5. **仍在早期** — G-Watch 的核心差异化能力 Iket 已可用的同时，完整功能的 Binary Analysis 和 Profiling 模块仍在开发中。:mdnice-bold-italic[Mars Compute 的整体产品也处于"设计伙伴阶段"（Design Partner Program），距离全功能平台还有显著距离。]

对于关注 AI Infra 和 GPU 编程工具的读者，G-Watch 和背后的 Mars Compute 都值得持续追踪。

> **Mars Compute**: mars-compute.com[2]**G-Watch 仓库**: github.com/mars-compute-ai/G-Watch[3] 撤回一个开源仓库👀**G-Watch 文档**: g-watch.github.io[4]**安装**: `pip install gwatch` && `npx skills add mars-compute-ai/G-Watch -g`

参考资料

[1]

何丙胜教授（Prof. Bingsheng He）:*https://www.comp.nus.edu.sg/~hebs/*

[2]

mars-compute.com:*https://mars-compute.com/*

[3]

github.com/mars-compute-ai/G-Watch:*https://github.com/mars-compute-ai/G-Watch*

[4]

g-watch.github.io:*https://g-watch.github.io/*

::: related-title
**相关推荐**
:::

::: related-links
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**

:::
