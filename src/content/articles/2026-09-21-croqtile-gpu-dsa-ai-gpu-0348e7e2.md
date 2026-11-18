关键词：内核编程、**分块计算**、:mdnice-bold-italic[符号推导]、**异步流水**、智能调优

> Croqtile 是一个面向 GPU/DSA 场景的 AI 原生高性能计算 DSL，近期已正式开源。它的核心主张，在与 TileLang、Helion、Triton、CUDA、CuteDSL 对比中，可以用三组量化结果概括：

- 每个算子**内核生成仅需约 500 Token** （CUDA/CuTe 约 2200、CUTLASS 约 4000）；
- 在暴露 **warp 级控制的 DSL 中 pass@1 最高**，动态形状下仅下降 4 个百分点；
- 353 项编译时检查覆盖 8 类常见 GPU 内核 Bug。

![文章插图](/articles/2026-09-21-croqtile-gpu-dsa-ai-gpu-0348e7e2/images/01.webp)

::: image-caption
图 1：Token 预算总览。Croqtile 将结构意图直接编码，每个内核仅需约 500 Token，而 CUDA/CuTe 约需 2200 Token，CUTLASS 约需 4000 Token。节省效果会叠加：更少的 Token 意味着 Agent 可以在相同上下文窗口内容纳更多优化历史、性能分析数据和优化规则。从(a)"静态形状"和(b)"动态形状"两个维度，系统证明了 Croqtile 的核心设计卖点——AI 原生的语法密度。其提出"Syntax-Context Co-Design"原则，而这组柱状图正是该原则的量化验证。三要素堆叠结构（源码 + profile 反馈 + 技能提示词）特别值得关注：它说明 Token 预算并非只看源码行数，需要考虑 AI Agent 完成一次完整内核编写任务所需的全部上下文。Croqtile 在三个维度上均保持最低，说明其优势是系统性的——不仅源码更短，而且因为编译器能返回结构化诊断，Agent 所需的试探性反馈轮次也更少。对 AI Agent 而言，Token 预算直接等于单次上下文内能尝试的优化迭代次数，2-3 倍的 Token 节省意味着 Agent 有更多空间进行多轮调优，这对性能敏感的 GPU 内核开发是决定性的。
:::

![文章插图](/articles/2026-09-21-croqtile-gpu-dsa-ai-gpu-0348e7e2/images/02.webp)

- **Croqtile, a C++ embedded DSL for high-performance kernel programming**, :mdnice-bold-italic[featuring easy-to-use syntax with zero-cost abstractions, comprehensive compile-time safety checks, first-class dynamic shapes and symbolic dimensions, and an AI-tuning-friendly design from the ground up.]
- 开源仓库：https://github.com/LancerLab/croqtile
- 官方网站：https://lancerlab.github.io/croqtile-website/en/
- 入门教程：https://lancerlab.github.io/croqtile-tutorial/tutorial/
- 语言文档：https://lancerlab.github.io/croqtile-tutorial/documentation/

::: related-title
**相关推荐**
:::

::: related-links
- **[CuTeDSL 4.7 把 NVVM IR 摆上了台面：用 Python 写 GPU kernel，也能像 CUDA 一样指挥硬件](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904352&idx=1&sn=655d4dde49461853e38ffb7649f5ce8e&scene=21#wechat_redirect)**
- **[TIRx：TVM新一代硬件内核DSL，FP8/NVFP4/FlashAttention4实测性能逼近行业最优实现，底层统一支撑人工、Agent、MegaKernel三类高性能内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902815&idx=1&sn=e440508ea6c74f9393cf6d5406e56ed8&scene=21#wechat_redirect)**
- **[算子 编译器天塌了吗。。NVIDIA 提出 AVO：当 AI Agent 成为进化算子，探索突破 Blackwell 性能极限，超越 FlashAttention‑4 最高10.5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899076&idx=1&sn=64a6532410f9d22f65043945582d4f1a&scene=21#wechat_redirect)**

:::

> **研发与开源状态。** 该项目由 LancerLab 团队全职历时一年半独立研发，**核心代码与架构设计为手写自研，仅边缘辅助功能借助 AI 工具完成**。目前主体代码已在 GitHub 开源，:mdnice-bold-italic[配套的 Agent 自动调优工具、标准化 Benchmark 测试套件及全套对比实验资料仍处于内部测试阶段，后续将逐步开源。]

**硬件覆盖。** 当前代码中可直接确认的目标后端包括：NVIDIA（通过 CuTe 路径，支持 SM 70+，Hopper 上可使用 TMA、WGMMA）、AMD（HIP 路径，RDNA 2+）、以及可移植 C++ CPU 后端。据说，还支持多款未开源的 DSA 加速架构。

**研究成果。** 依托该语言开展的科研工作已有四篇论文被 SC、ATC、CGO 收录（两篇 SC、一篇 ATC、一篇 CGO），其中一篇 SC 论文获最佳论文提名，其所有论文将于年底正式见刊。

**生态工具。** 开源仓库中可直接运行的编译器为 `choreo`（多 pass 流水线）与 `copp`（预处理器）；此外还有 CoIR 工具和 `tools/co-py` Python 绑定入口。官方教程语言文档、浏览器 Playground（WASM 编译器）和 AI 驱动的 Tuner 框架构成外围生态，但 Tuner 的完整代码尚未随主体开源。

![文章插图](/articles/2026-09-21-croqtile-gpu-dsa-ai-gpu-0348e7e2/images/03.webp)

::: image-caption
图 2：AI Agent 内核编写准确率总览。Croqtile 的紧凑语法和即时编译时反馈使 Agent 无需反复试错和调试即可一次写对。在所有暴露 warp 级控制的 DSL 中，Croqtile 的 pass@1 最高；动态形状下，Croqtile 仅下降 4 个百分点（至 82.1%），而 Triton 降至 48%，TileLang 降至 52%，Helion 降至 45%。Croqtile 的优势是"模型无关"的——这一点由后续表 1 的跨模型实验进一步证实：在 Opus 4.6 Max 和 MiniMax M2.5 两种能力差异巨大的模型上，Croqtile 均保持领先。这说明 Croqtile 的设计是从语法和编译器两个层面降低了所有 AI Agent 的认知负荷。动态场景下的衰减幅度（仅 4pp）是最有说服力的数据：它证明 Croqtile 的编译时安全机制是真正将动态形状的复杂性内化到了编译流程中。对生产环境而言，这意味着从静态到动态工作负载的迁移几乎不损失 Agent 的自动化编写能力。
:::

![文章插图](/articles/2026-09-21-croqtile-gpu-dsa-ai-gpu-0348e7e2/images/04.webp)

::: image-caption
表 1：跨模型性能优势。该优势在不同能力层级的模型上均成立：在中等容量模型上优势更为扩大。对比 Opus 4.6 Max 和 MiniMax M2.5 两款模型，在 Croqtile、Triton、TileLang、CUDA 四种 DSL 下的内核编写成功率。
:::

## 本文目录

- 一、快速上手：从一个能校验结果的内核开始
- 二、AI 友好设计：同时缩短代码和修改链条
  - 2.1 两个协同设计：语法密度与反馈闭环
  - 2.2 代码站点：改动一项功能需要碰几处？

![文章插图](/articles/2026-09-21-croqtile-gpu-dsa-ai-gpu-0348e7e2/images/05.webp)

## 一、快速上手：从一个能校验结果的内核开始

> 基础环境按照 README 准备：支持 C++17 的编译器、CMake 3.18+、Ninja，以及 Flex/Bison。CUDA/CuTe 路径需要 CUDA Toolkit 12.0+，体验 TMA、WGMMA 等 Hopper 特性需要对应硬件。

首次构建会按配置下载相关依赖。建议先运行仓库中的加法测试，它包含完整宿主程序和结果校验。

```
# 来源：`README.md`；将文档中的输入文件替换为仓库已有的 `tests/gpu/end2end/add.co`。git clone https://github.com/LancerLab/croqtile.gitcd croqtilemake# 仅生成目标源码，便于观察编译结果./choreo -t cute -es tests/gpu/end2end/add.co# 生成编译与运行脚本./choreo -t cute -gs tests/gpu/end2end/add.co -o run.shbash run.sh --execute# 全量测试入口make test
```

成功运行加法程序后，应看到 `Test span_as Passed` 和 `Test local_dma Passed`。若 CUDA 安装位置不在默认目录，需要配置相应环境；测试还依赖 `FileCheck`、`bc`，并可能因硬件条件跳过部分用例。

当前 Makefile 默认构建进入 `build-release/`，除了 `choreo`、`copp`，还生成 CoIR 工具。具体配置参考构建与测试指南[1]，学习路径则可沿官方教程从数据搬运、并行映射一路进入流水线。

## 二、AI 友好设计：同时缩短代码和修改链条

### 2.1 两个协同设计：语法密度与反馈闭环

> Croqtile 区别于传统 DSL 的核心不在于"语法更短"，而在于它从两个方向同时压缩 Agent 的工作负担：

**语法-上下文协同设计（Syntax-Context Co-Design）。** 一轮自动调优需要在上下文窗口中同时容纳：当前内核源码、硬件约束、上一轮 profiling 结果、失败原因和历史尝试。:mdnice-bold-italic[传统 DSL 中，重复的描述符、地址计算和同步样板会持续占用上下文。Croqtile 把 tile 形状、流水线深度、warp 角色、DMA/TMA 搬运等结构意图直接编码为语法，而非通过模板参数或宏展开间接表达]。这样做的目标是提高每个 Token 承载的有效语义——Agent 看到一行 `paralell p by 1 : group-4`，就同时知道流水线深度、缓冲槽数量和 warp 分组方式，不需要再从分散的代码位置拼凑这些信息。

**编译器-执行环境协同设计（Compiler-Harness Co-Design）。** 编译器不只是"生成代码"，还有 Agent 的反馈回路。每次编译在数秒内完成，且是返回结构化诊断：违反了哪条约束、符号推导路径是什么、建议如何修复。Agent 因此直接根据机器可读的反馈做一次性修正。这一点是理解下图 1 中"Profile + feedback"段为何同样最低的关键——反馈轮次少，意味着 profiling 和试探性编译的 Token 消耗也少。

![文章插图](/articles/2026-09-21-croqtile-gpu-dsa-ai-gpu-0348e7e2/images/01.webp)

::: image-caption
官方上下文预算对比图
:::

### 2.2 代码站点：改动一项功能需要碰几处？

> 另一个指标同样重要：实现或调整一项功能，需要同时触碰多少处代码？

::: related-title
**相关推荐**
:::

::: related-links
- **[CuTeDSL 4.7 把 NVVM IR 摆上了台面：用 Python 写 GPU kernel，也能像 CUDA 一样指挥硬件](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904352&idx=1&sn=655d4dde49461853e38ffb7649f5ce8e&scene=21#wechat_redirect)**
- **[TIRx：TVM新一代硬件内核DSL，FP8/NVFP4/FlashAttention4实测性能逼近行业最优实现，底层统一支撑人工、Agent、MegaKernel三类高性能内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902815&idx=1&sn=e440508ea6c74f9393cf6d5406e56ed8&scene=21#wechat_redirect)**
- **[算子 编译器天塌了吗。。NVIDIA 提出 AVO：当 AI Agent 成为进化算子，探索突破 Blackwell 性能极限，超越 FlashAttention‑4 最高10.5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899076&idx=1&sn=64a6532410f9d22f65043945582d4f1a&scene=21#wechat_redirect)**
- **[颠覆传统 EDA 流程！AI 设计的 Redwood 芯片反向优化下一代硬件架构，能效超 Jetson 3.4 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905246&idx=1&sn=edf9b23c44cca0bcf127fd98bcff45b0&scene=21#wechat_redirect)**
- **[速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)**

:::
