关键词：**推理引擎**、:mdnice-bold-italic[智能体集群]、**自我进化基础设施**、CUDA kernel、MiniMax

> INT21 发布 Inference Engine Factory 的首批工程样例：由自我进化智能体集群端到端构建的视频、音乐、语音生成推理栈。

**其背景是推理工程师人才瓶颈——新模型每周涌现，每个模型都需要 GPU 与框架优化专家，而 AI 生成代码普遍存在技术债问题** （GitClear 数据显示重复代码增加约 10 倍，超 15% 的 AI 提交引入长期潜伏的问题）。:mdnice-bold-italic[INT21 的解法是让智能体集群从第一性原理出发编排整个推理栈：CUDA/PTX kernel、megakernel、Rust CUDA 驱动与 serving 集成，不依赖 PyTorch 或任何推理框架，专家只负责设定目标与约束]。

![文章插图](/articles/2026-08-20-pytorch-int21-sglang-vllm-78e8ebab/images/01.webp)

- INT21 — Addressing the Inference Bottleneck
- https://int21.ai/insights/addressing-the-inference-bottleneck/
- 在线 Demo：https://demo.int21.ai/

::: related-title
**相关推荐**
:::

::: related-links
- **[INT21 与 PTX Kernel 工厂：告别底层调优难题，智能系统让英伟达 GPU 跑得更快](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902539&idx=1&sn=d0f6cee3c8dc45fa3870ff53f3233b1d&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**
- **[Triton 算子如何越过 PTX 直抵 Blackwell：CUDA TileIR 后端的诞生，Triton-to-tile-IR](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903681&idx=1&sn=a30c96b2bfbc7a32d9745ca6d303ddfd&scene=21#wechat_redirect)**

:::

> 底层模型为 MiniMax H3、MiniMax Music 3 和 Fish Audio S2 Pro。**基准测试显示 INT21 ES Engine 全面领先：**

::: related-links
- H3 视频生成 10.80 秒对 SGLang 的 12.03 秒；
- Music 3 延迟 4.24 秒对 4.91 秒；
- S2 Pro 实时率 0.252 对 vLLM 的 0.326，且代码无需人工审查。

:::

::: block-1
每个引擎约 **2 万行 Rust 加 1 万行 CUDA/PTX 代码，直接跑在 CUDA 驱动上，无额外依赖**。:mdnice-highlight[这是继 PTX Kernel Factory 之后，"自我进化基础设施"从单点 kernel 走向完整生产系统的第二次品类验证。] INT21 的智能体集群（agent swarms）已经为视频生成、音乐生成和语音合成构建了完整的推理栈，这些工程样例现已上线（https://demo.int21.ai/）。你可以直接上手生成自己的视频、音乐和 TTS 内容，无需 API key。
:::

## 本文目录

::: related-links
- 一、瓶颈在哪里
- 二、推理引擎工厂（Inference Engine Factory）做了什么
- 三、代码质量：拒绝 AI 泔水（AI Slop）
- 四、这对自我进化基础设施意味着什么

:::

## 一、瓶颈在哪里

> 新 AI 模型几乎每周都在发布，模型数量的增速远超能运维它们的人数。:mdnice-bold-italic[每往推理栈里加一个模型，就需要一位在 GPU 与框架优化上有深厚功底的工程师，而这类专家越来越难找。对 token 工厂和推理服务商来说，人才约束是 AI 基础设施规模化最大的瓶颈之一。]

核心问题是：**AI 智能体能否解决这个困境？** 这又引出第二个问题：**智能体生成的基础设施，交付出来是干净的，还是藏着看不见的技术债？**

GitClear 分析了 2.11 亿行代码后发现：重复代码块增加了约 10 倍，而重构比例从 24.1% 跌到 9.5%。:mdnice-bold-italic[在生产仓库中，超过 15% 的 AI 生成提交会引入往往潜伏数月才被发现的问题。比如 GitHub Copilot 在 Hysteria 2 中引入的命令注入漏洞，三周后才被抓获]。AI 生成的代码确实能更快交付，但技术债会隐形积累，侵蚀人们对所建系统的信任。

## 二、推理引擎工厂（Inference Engine Factory）做了什么

> INT21 的解法，是迈出"消除手工打造推理引擎"的第一步。**其自我进化的智能体集群端到端地编排整个推理栈：CUDA 与 PTX kernel、megakernel、Rust 版 CUDA 驱动、以及 serving 集成**。整套栈从第一性原理出发设计与优化，不依赖 PyTorch，不依赖 kernel 库抽象，也不依赖任何推理框架。

![文章插图](/articles/2026-08-20-pytorch-int21-sglang-vllm-78e8ebab/images/02.webp)

::: image-caption
图 1：传统推理栈 vs INT21 智能体集群生成栈。传统推理栈需要在每一层（推理框架、AI 框架、GPU 库、kernel 编译器、集成与运行时）都配备专家团队，层层人工实现，最终落在 CUDA 驱动和 NVIDIA GPU 上；而 INT21 的集群生成栈中，专家只负责设定目标、约束与架构，自我进化智能体集群直接产出为你的模型和场景定制的 Rust + CUBIN 二进制，跑在 CUDA 驱动之上。中间三层人工环节被整体压缩掉了。
:::

视频、音乐、语音生成属于高度复杂的推理负载：多阶段、多尺度的网络架构，对 kernel 和运行时都有很高要求。证明智能体能为这类负载构建干净的生产级栈，等于验证了它们能应对更广的基础设施优化挑战。当一段 768p 视频可以即时生成并返回，这就是自我进化基础设施的实际运转。

![文章插图](/articles/2026-08-20-pytorch-int21-sglang-vllm-78e8ebab/images/03.webp)

::: image-caption
图 2：MiniMax Music 3 的架构图，展示了 INT21 音乐栈所服务的模型有多复杂——底层是结构化描述（Structured Caption）与歌词（Lyrics）两类输入条件，经 Global LLM 编码后进入 Hybrid LM 层：Local LLM 逐帧生成音频 codebook token（C₁ 到 C₇），并输出 hidden state 与 stop token；上层经 fusion 融合后进入 Flow-Matching 与 Flow VAE Decoder 完成波形合成。来源：MiniMax。
:::

底层模型分别是：在 Artificial Analysis 视频生成榜单上包揽前三的 MiniMax H3、用于音乐生成的 MiniMax Music 3，以及用于 TTS 的 Fish Audio S2 Pro。智能体编排的推理栈在不需要人类专家调优的情况下，跑出了顶级性能。

这些是 INT21 自我进化基础设施中 Inference Engine Factory 的首批工程样例，建立在公司第一个产品 PTX Kernel Factory 之上。**PTX Kernel Factory 证明了智能体能理解并构建底层 GPU 指令；Inference Engine Factory 则证明智能体能编排整个生产系统。**

Kernel 是边界清晰的孤立问题；推理栈是系统，各层之间存在复杂交互，一个架构决策会在层与层之间产生涟漪。这恰恰是人们预期 AI 生成代码会崩掉、需要人工收拾烂摊子的地方。

## 三、代码质量：拒绝 AI 泔水（AI Slop）

> INT21 的视频、音乐、语音生成栈表明：**自我进化智能体集群构建的代码无需人工代码审查，并且在相同并发设置下的低延迟推理中跑赢了最先进的推理方案**。INT21 在 MiniMax H3、MiniMax Music 3、Fish Audio S2 Pro 三个模型上做了测试，以展示工厂的覆盖面。

### 3.1 MiniMax H3 —— 8x B200

基准测试使用官方 BF16 checkpoint，合并 Turbo LoRA，四步采样，生成 15 秒 H.264/AAC 媒体。结果为四条不同 prompt 的平均推理耗时。

| 引擎 | 延迟（秒，越低越好） |
| --- | --- |
| **INT21 ES Engine** | **10.80** |
| SGLang | 12.03 |
| vLLM | 无官方 8x B200 支持 |
| TensorRT-LLM | 无官方支持 |

### 3.2 MiniMax Music 3 —— 2x H100

基准测试为两个 session 共 14 个请求，全部产出等效的合法 32 kHz 立体声 WAV 文件，时长恰好 10 秒。

**亮点：** 协作式 megakernel（cooperative megakernel）。

| 引擎 | 延迟（秒，越低越好） |
| --- | --- |
| **INT21 ES Engine** | **4.24** |
| SGLang | 4.91 |
| vLLM | 无官方支持 |
| TensorRT-LLM | 无官方支持 |

### 3.3 Fish Audio S2 Pro —— 1x H100

基准测试使用完整的 Seed-TTS-Eval 英文测试集，共 1,088 条样本。

**亮点：** 慢速与快速自回归 megakernel，外加一个用于 codec 运算的 316 节点 CUDA Graph。

| 引擎 | 实时率 RTF（越低越好） |
| --- | --- |
| **INT21 ES Engine** | **0.252** |
| SGLang | 0.301 |
| vLLM | 0.326 |
| TensorRT-LLM | 无官方支持 |

所有 INT21 工程样例引擎都直接构建在 CUDA 驱动之上，无需额外依赖即可部署，连 CUDA Runtime 容器都不需要。每个引擎约包含 2 万行 Rust 代码和 1 万行 CUDA/PTX kernel 代码。

## 四、这对自我进化基础设施意味着什么

> 推理工程师瓶颈是实打实的：每个新模型、每个新硬件目标、每项架构优化，都要消耗稀缺的专家时间。:mdnice-bold-italic[INT21 的视频、音乐、语音栈表明，这些约束不该拖住基础设施。智能体可以编排 kernel 之外的整个生产系统，并交付从第一天起就正确的代码。]

这就是自我进化基础设施的形态：人类设定目标、定义测试、授权发布；优化工作本身自主完成，比人工调优更快也更好。

对在 AI 上竞争的基础设施团队来说，优势属于能跑起最强自主优化闭环的一方。INT21 这次发布的，是 Inference Engine Factory 的工程样例，它证明自我进化基础设施是可行的：算力改进基础设施的速度和质量，可以超过单纯依赖人类专家。

![文章插图](/articles/2026-08-20-pytorch-int21-sglang-vllm-78e8ebab/images/04.webp)

在线 demo[1] 已开放，可即时生成视频、音乐和语音。视频、音乐、TTS 的生成上限分别为 15 秒、3 分钟和 30 秒，每位用户每 24 小时有两次免费生成机会。

*INT21 特别感谢 NVIDIA Inception 计划[2]为 SwarmOS 的开发赞助了 8 张 H100 GPU。*

参考资料

[1] 在线 demo: *https://demo.int21.ai/*

[2] NVIDIA Inception 计划: *https://www.nvidia.com/en-us/startups/*

::: related-title
**相关推荐**
:::

::: related-links
- **[INT21 与 PTX Kernel 工厂：告别底层调优难题，智能系统让英伟达 GPU 跑得更快](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902539&idx=1&sn=d0f6cee3c8dc45fa3870ff53f3233b1d&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**
- **[Triton 算子如何越过 PTX 直抵 Blackwell：CUDA TileIR 后端的诞生，Triton-to-tile-IR](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903681&idx=1&sn=a30c96b2bfbc7a32d9745ca6d303ddfd&scene=21#wechat_redirect)**

:::
