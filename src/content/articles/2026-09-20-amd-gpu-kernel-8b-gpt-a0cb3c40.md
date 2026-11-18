关键词：AMDKernelVault、**智能体内核优化**、HIP 内核、Triton 内核、**ROCm 生态**

> 本文这项工作说明了：没有数据，就没有好的模型；没有好的模型，就只能依赖人工和通用大模型，效率上不去，生态差距就越拉越大。此外也有三点洞察：

- **内核优化的核心是架构原生的优化范式**，只有经过原生硬件编译、执行、实测的数据，才能训练出真正懂架构的模型。:mdnice-bold-italic[语法翻译永远是二流方案，原生数据才是根本]。
- **垂直领域小模型+专用语料，效果可以超越通用大模型**。内核生成是高度垂直的领域，:mdnice-bold-italic[通用大模型再强，知识面再广，也不如经过专用语料深度微调的小模型。]
- **Agent 闭环既是数据生产工具，也是最终的部署形态**。训练的时候用 Agent 流水线生产数据，:mdnice-bold-italic[部署的时候用同样的 Agent 闭环做优化，训练和部署的范式完全统一]，模型学的就是部署时要用的能力，没有落差。

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/01.webp)

::: image-caption
表6：与同类内核生成系统的定位对比。对比Kevin、AutoTriton、GEAK、AMDKernelVault四个系统。GEAK本身不训练模型，推理阶段反复调用前沿大模型，成本高；AMDKernelVault直接训练单个8B模型，一个模型同时承担生成器、反思器、优化器，全部在本地AMD GPU运行。奖励信号同时融合编译、数值正确性、性能延迟。
:::

::: block-1
现有的基于大模型的内核智能体大多以 CUDA/NVIDIA 为中心。这种一边倒的产业格局，使得 AMD 的 ROCm 生态长期缺乏大规模经过原生硬件执行验证的内核数据，:mdnice-highlight[无法训练出适配 AMD 架构的专用内核大模型，只能依赖通用大模型做语法层面的翻译式优化，效果不佳且部署成本高、灵活性差。]
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/02.webp)

- **AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization**
- 组织机构：Advanced Micro Devices, Inc.（AMD）
- 论文链接：https://arxiv.org/pdf/2609.12471
- :mdnice-bold-italic[开源代码地址：https://github.com/AMD-AGI/hip_kernel_llm_lab]
- :mdnice-bold-italic[开源数据集地址：https://huggingface.co/datasets/amd/AIG-Datasets]

::: related-title
**相关推荐**
:::

::: related-links
- **[模块化 Triton GPU 内核代码生成 Agent 架构 GEAK：生成正确率达 63%，速度提升 2.59 倍！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890465&idx=1&sn=a73eec7d10391ef7371018529fb2405c&scene=21#wechat_redirect)**
- **[AutoTriton：通过强化学习实现自动Triton编程！8B参数媲美Claude-4 与 DeepSeek-R1！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890530&idx=1&sn=2f4b9743242cd200c0bae8e8f8c68e67&scene=21#wechat_redirect)**
- **[超越AutoTriton/KernelLLM的 Triton 内核生成模型 TritonRL：融合SFT蒸馏与RL分层奖励设计](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893752&idx=1&sn=e5fdd6a30c934972634fa013748e8333&scene=21#wechat_redirect)**

:::

> 为此，AMD 推出 **AMDKernelVault，这是一套面向 CDNA 架构 GPU 的开源 HIP 与 Triton 内核语料库，同时配套了完整的 Agent 化数据生产流水线与模型训练框架**。通过生成-评估-反思的智能体闭环，项目产出了 :mdnice-bold-italic[6.4 万+经过 ROCm 编译、正确性验证与延迟实测的 HIP 内核样本，以及近 4 万个 Triton 内核样本。]

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/03.webp)

::: image-caption
图 1：AMDKernelVault整体流水线架构。展示HIPKernelGen与TritonKernelGen两套Agent数据生成流水线。输入是PyTorch参考实现，Agent循环执行：生成候选HIP/Triton代码 → ROCm编译 → AMD真机运行校验数值正确性 → 采集延迟性能数据 → 根据报错做反思修正，循环迭代直到产出合格样本。合格样本存入数据集，同时记录编译状态、正确性、延迟元数据。这套流水线是数据集的源头，和普通文本爬取最大区别：每一条样本都经过AMD硬件真机验证。该流水线可以大规模自动制造训练数据，不需要人工手写大量kernel，解决AMD生态缺少实测代码样本的痛点。
:::

基于这套语料训练的 **Qwen3-8B 模型**，多项正确性指标:mdnice-bold-italic[超越 GPT-5、Claude Sonnet 4 等千亿级前沿大模型：]

- 在 PyTorch-to-HIP 任务上达到 34.0% Pass@1
- 在 TritonBench-G 上达到 33.2% Corr@3
- 在 ROCmBench 上达到 41.94% Corr@3

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/04.webp)

::: image-caption
表 4：固定 3 轮 GEAK OptimAgent-v2 评估协议下 TritonBench-G 测试结果。本表是论文最核心的横向对比，所有模型使用完全相同的 Agent 循环预算，变量仅为模型权重。训练后的 Qwen3-8B 在 Corr@3 达到 33.2%，超过 GPT-5、Gemini、Claude，同时带来 1.46 倍平均加速。这组结果直接证明，在 AMD 硬件上，用 AMDKernelVault 数据集训练的小模型，在 Triton 算子生成任务上超越通用闭源大模型。单纯的通用大模型缺少 AMD GPU 硬件知识，很容易写出语法能过、但在 AMD 上数值错误或访存低效的代码。该表同时给出编译准确率与最终加速，兼顾正确性和性能两个内核生成任务的核心指标，不是只看能不能编译。
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/05.webp)

::: image-caption
表 5：固定 3 轮 GEAK OptimAgent-v2 评估协议下 ROCmBench 测试结果。ROCmBench 是 AMD 专属算子测试集，更贴近 ROCm 真实生产场景。训练后的 Qwen3-8B 取得最高 Corr@3（41.94%），优于所有对比模型。Claude 编译成功率更高，但结果正确性弱于本文模型。这一结果说明，AMDKernelVault 学到是适配 AMD MI 系列硬件的算子逻辑。通用大模型容易直接套用 CUDA 经验，在 AMD 特有 LDS、wavefront 调度场景出错。
:::

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/06.webp)

::: image-caption
图 3：ROCmBench 上 Pass@K 对比曲线（31 个内核任务）。本文模型在全部 K 取值下都取得最高通过率，证明在内核生成任务上拥有更优的样本效率。Pass@K 曲线衡量：生成 K 个候选 kernel，至少一个正确的概率。训练后的 Qwen3-8B 整条曲线高于所有对比模型。它的核心价值是体现样本效率：不需要大量采样候选代码，就能产出正确内核。传统 Agent 方案要生成大量候选再筛选，推理成本高；本工作把反思能力内化进模型权重，单次生成有效概率更高。曲线在 K 增大后增速放缓，也暴露局限：高难度算子无论采样多少候选，成功概率都难以显著提升，对应硬件深度调优能力存在天花板。
:::

::: block-1
这项工作不仅:mdnice-highlight[从根源上极大扩充了 AMD 生态的内核训练数据，更证明了垂直领域【专用小模型】可以在【特定任务】上反超通用大模型]，为 ROCm 生态的 AI 驱动内核优化建立了完整的基础设施闭环。
:::

## 本文目录

- 一、CUDA 垄断下的 AMD 内核优化困局
- 二、AMDKernelVault：十万级原生验证内核语料库全解析
- 三、Agent 驱动的数据生产流水线：生成-评估-反思闭环
- 四、小模型也能打：SFT+执行感知 RL 的训练范式
- 五、实验结果：8B 模型如何反超前沿大模型
- 六、技术本质与边界：这项工作到底改变了什么
- ROCm 生态的新起点

![文章插图](/articles/2026-09-20-amd-gpu-kernel-8b-gpt-a0cb3c40/images/07.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 开源 hip-ep：用 MLIR 贯通 ONNX Runtime 与 ROCm，落地 LLVM bitcode 编译](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905194&idx=1&sn=2f1918c0623bc9f0918a730bcb28a992&scene=21#wechat_redirect)**
- **[无源码改写汇编优化 GPU Kernel！AMD 联合南方科技大学提出 AsmEvo，让已编译 GPU 内核最高提速 3.88 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905090&idx=1&sn=53df28ba322bdc929e8df59b9f391f98&scene=21#wechat_redirect)**
- **[34.93 倍加速！MLSys 26 DSA 智能体赛道冠军方案，一套规则手册与三个子智能体替代人类内核工程师](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901644&idx=1&sn=ddd845b9bc6fd48c0c73a0ce535f45e0&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[Triton 首次生产级落地自研 ASIC：Meta MTIA-2i 上 60 种模型、50% 的层与 47% 非 GEMM 时间被攻占](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904593&idx=1&sn=48f27e651841785d00dcf73aaf58a55f&scene=21#wechat_redirect)**

:::
