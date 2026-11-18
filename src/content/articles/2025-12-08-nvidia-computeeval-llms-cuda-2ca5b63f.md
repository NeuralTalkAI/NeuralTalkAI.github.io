关键词：**CUDA 代码生成**、NVIDIA、**CUDA 代码数据集**、AI 辅助编程

> 大语言模型（LLMs）正深刻改变开发者的编码方式——无论是资深工程师还是入门开发者，如今的顶尖模型已能流畅生成 Python 脚本、React 网站代码等。

- 代码：https://github.com/nvidia/compute-eval
- 数据：https://huggingface.co/datasets/nvidia/compute-eval
- 2500 字，阅读需 8 分钟，**播客 10 分钟**

相关推荐

- **[MultiKernelBench：首个覆盖英伟达GPU、华为NPU、谷歌TPU的Kernel生成基准，涵盖14类算子285项测试](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891504&idx=1&sn=d1558faf582d556f3ac231f985efc057&scene=21#wechat_redirect)**
- **[针对AMD AI Engine空间架构的可扩展BLAS库AIEBLAS：基于自动代码生成的创新实现方案！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891849&idx=1&sn=6f99fa04d5077256647c2d43d33cccce&scene=21#wechat_redirect)**
- **[迁移 CUDA 代码到其它硬件！HPCTransCompile：用 LLM 对 CUDA 代码转义！用AI编译器生成高质量代码对](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889060&idx=1&sn=f73025e12265f3ee62a373c5f3b03177&scene=21#wechat_redirect)**

- 更多相关推荐见文末 -

但在高性能 GPU 编程领域，一个关键问题始终存在：**如何判断 LLMs 能否驾驭 CUDA 编程的复杂性（如并行逻辑、内存管理、线程同步）？**

::: block-1
为解决这一问题，NVIDIA 推出了**ComputeEval** ——一款专注于 LLM CUDA 代码生成能力的开源评估框架，并在短短半年内完成了从初始版本到重大升级的迭代，持续推动 AI 辅助 GPU 开发的标准化进程。
:::

## 本文目录

- 一、ComputeEval：建立 CUDA 编程评估基准
  - 1. 初始版本的核心构成
  - 2. 初始模型评估：基础能力达标，复杂场景存短板
- 二、ComputeEval 难度升级：深挖 LLM 的现代 CUDA 能力
  - 1. 新版本的核心突破
  - 2. 新评估结果：分数下降 ≠ 能力退化，而是基准更贴合实际
- 三、未来规划：扩展 CUDA-X 生态，拥抱社区协作
- 结语

![文章插图](/articles/2025-12-08-nvidia-computeeval-llms-cuda-2ca5b63f/images/01.webp)

## 一、ComputeEval：建立 CUDA 编程评估基准

> 2025 年 4 月，NVIDIA 首次发布 ComputeEval，旨在填补 CUDA 领域缺乏专用评估工具的空白。

该框架以“**社区驱动、真实场景** ”为核心，灵感源自 HumanEval 等通用代码基准。

::: block-1
HumanEval 是 OpenAI 发布的代码生成评测集，共 164 道手写 Python 题，含函数签名、自然语言描述与单元测试，覆盖算法、字符串、数据结构等场景，难度由浅入深，以 Pass@k 指标衡量模型能否一次或多次采样中生成通过全部测试的代码，被广泛用于检验大模型的编程能力。
:::

不过，ComputeEval 则针对 CUDA 编程的特殊性（精度、并行性、性能要求极高）做了深度定制。

### 1. 初始版本的核心构成

ComputeEval 2025.1 版本通过两大组件实现对 LLM 的全面评估：

- **128 个手工打造的 CUDA 问题**：覆盖 GPU 编程的核心场景，从内核（kernel）启动、线程管理，到内存布局优化、共享内存利用，均基于真实开发需求设计，避免脱离实际的“纸面测试”。
- **沙盒化功能正确性测试**：提供专用代码环境，可安全执行 LLM 生成的 CUDA 代码，并自动验证其是否符合预期功能——这一设计解决了 CUDA 代码直接运行可能导致的设备风险，确保评估安全性。

```
# ComputeEval中的每个问题都存储为data目录下的一个文件夹，其中包含：CUDA-0/├── problem-spec.yaml      # Problem metadata and configuration├── context/               # Files visible to the tested model/system (headers, helpers)│   ├── include/│   │   └── kernel.h       # Interface contract to implement│   └── helpers/│       └── helpers.cu     # Optional helper utilities├── solution/              # Reference implementation (not shown to tested model/system)│   └── solution.cu└── test/                  # Test harness (not shown to tested model/system)    └── test/        └── test_main.cu
```

> 开发者可通过 NVIDIA 的compute-eval GitHub 仓库[1]获取框架代码，或在 Hugging Face[2] 平台访问数据集，快速开展自定义评估。

`problem-spec.yaml` 文件定义了每个问题的元数据和配置：

```
task_id: "CUDA/0"                     # Unique identifier (generally matches directory name)date: "2024-12-19"                    # Problem creation dateproblem_type: cuda_cpp                # Type: cuda_cpp or cuda_pythonprompt: "Implement a CUDA kernel..."  # Problem description shown to model# Build and test configurationbuild_command: "nvcc -I include -o test.out solution.cu test/*.cu"test_command: "./test.out"timeout_seconds: 30.0# Requirementsmin_cuda_toolkit: "12.0"             # Minimum CUDA version requiredarch_list: []                        # Optional: specific GPU architectures# Optional metadatametadata:  difficulty: medium                 # Problem difficulty level  tags: [kernels, memory]            # Classification tags  releases: [2025-1, 2025-2]         # Which releases include this problem  do_not_release: false              # Internal-only flag to skip CIsource_references: null              # Optional: required API calls/symbols to verify                                     #  - string: single item must be present                                     #  - list of strings: all must be present                                     #  - {any: [...]} at least one must be present                                     #  - {all: [...]} all must be present                                     #  - {all: [...], any: [...]} combines both
```

有关该数据集的更多信息，请参见 DATASET_CARD.md[3]。

### 2. 初始模型评估：基础能力达标，复杂场景存短板

为建立行业基准，NVIDIA 对当时 4 款顶尖 LLM 进行了测试，核心指标为“pass@1”（单次生成代码通过测试的概率）和“pass@3”（三次生成中至少一次通过的概率），结果如下：

| Model | pass@1 | pass@3 |
| --- | --- | --- |
| OpenAI o3-mini | 0.61 | 0.74 |
| Anthropic Claude Sonnet 3.7 | 0.54 | 0.60 |
| Llama 3.1 405b | 0.40 | 0.55 |
| Google Gemini 2.0 Flash Thinking | 0.37 | 0.52 |

*表 1：ComputeEval 2025.1 版本对顶尖 LLM 的评估结果（来源：2025 年 4 月《Announcing ComputeEval, an Open Source Framework for Evaluating LLMs on CUDA》[4]）*

从结果可见，**OpenAI o3-mini** 表现最佳，单次生成正确率超 60%，三次生成正确率接近 75%；但:mdnice-bold-italic[所有模型均暴露共性问题：仅能应对基础 CUDA 任务，在复杂并行逻辑（如多线程同步）、性能优化（如共享内存冲突避免）上频繁失效]，甚至无法遵守在其他编程语言中已能完成的基础指令——这**表明 LLMs 在 CUDA 编程领域仍有巨大提升空间。**

## 二、ComputeEval 难度升级：深挖 LLM 的现代 CUDA 能力

> 为更贴近工业界对高性能 GPU 编程的需求，2025 年 11 月 NVIDIA 推出 ComputeEval 2025.2 版本，通过“**扩规模、提难度、贴场景** ”三大升级，进一步挤压 LLM 的能力边界。

### 1. 新版本的核心突破

相比初始版本，2025.2 的升级聚焦于**“现代 CUDA 特性”** 和**“系统级编程能力”**：

::: block-1
- **问题规模翻倍**：新增 100+个 CUDA 挑战，总题量达 232 个，覆盖 CUDA Compute Core Libraries（CCCL），并融入动态模拟等真实应用场景（如物理引擎、AI 训练加速）。
- **难度显著提升**：要求 LLM 掌握进阶技术，包括 :mdnice-bold-italic[Tensor Cores 调用、高级共享内存模式（动态分配、冲突规避）、线程束级原语] （warp-level primitives），以及 :mdnice-bold-italic[CUDA Graphs、流（Streams）、事件（Events）的系统级编排] ——这些能力是实现 GPU 高效异步计算的关键，也是工业界 HPC（高性能计算）场景的核心需求。
:::

### 2. 新评估结果：分数下降 ≠ 能力退化，而是基准更贴合实际

NVIDIA 对 13 款主流 LLM（含新增的开源模型如 gpt-oss 系列、Llama 4 Maverick）进行了测试，重点对比“pass@1”指标在 2025.1 与 2025.2 版本中的变化：

| Model | ComputeEval 2025.2（232 题）pass@1 | ComputeEval 2025.1（128 题）pass@1 |
| --- | --- | --- |
| GPT-5 (medium) | 0.5819 | 0.61 |
| Claude Sonnet 4.0 | 0.5517 | 0.64 |
| gpt-oss-20B (high) | 0.5474 | N/A |
| gpt-oss-120b (high) | 0.5302 | N/A |
| Claude Opus 4.0 | 0.5216 | N/A |
| DeepSeek-R1 | 0.4397 | 0.55 |
| gpt-oss-120b (medium) | 0.4224 | N/A |
| gpt-oss-20b (medium) | 0.4224 | N/A |
| gpt-oss-120b (low) | 0.4052 | N/A |
| DeepSeek-V3.1 | 0.3750 | 0.44 |
| Llama 4 Maverick 17B 128E | 0.3448 | 0.47 |
| Llama 3.1 405B | 0.3405 | 0.40 |
| gpt-oss-20B (low) | 0.3319 | 0.41 |

*表 2：LLM 在 ComputeEval 2025.1 与 2025.2 版本的 pass@1 对比（来源：2025 年 11 月《Benchmarking LLMs on AI-Generated CUDA Code with ComputeEval 2025.2》[5]）*

所有模型的“pass@1”分数均出现下降（如 GPT-5 从 0.61 降至 0.5819，Claude Sonnet 从 0.64 降至 0.5517），但这并非 LLM 能力退化，而是**基准测试更贴近真实工业需求** 的体现：:mdnice-bold-italic[新版本暴露了 LLMs 在“系统级协调”（如多流异步控制）、“性能优化”（如 Tensor Cores 高效利用）上的显著短板] ——即使是表现最佳的 GPT-5（medium）和 Claude Sonnet 4.0，也难以完全驾驭现代 CUDA 的复杂性。

值得注意的是，**开源模型（如 gpt-oss 系列）首次加入评估，其高性能版本（high 配置）表现接近闭源模型**，显示开源社区在 CUDA 编程领域的潜力，但低配置版本差距明显，反映模型规模对 CUDA 能力的影响。

## 三、未来规划：扩展 CUDA-X 生态，拥抱社区协作

> ComputeEval 的核心目标【不仅】是“评估工具”，更是“推动 AI 辅助 CUDA 开发的标准”。

NVIDIA 明确了两大未来方向：

1. **扩展 CUDA-X 库覆盖**：计划将评估范围延伸至 cuBLAS（线性代数）、CUTLASS（矩阵乘法模板）、cuDNN（深度学习加速）、RAPIDS（数据科学）等 CUDA-X 核心库，:mdnice-bold-italic[进一步贴近 AI 训练、科学计算等核心场景，评估 LLM 调用专业 API 的准确性与参数配置能力。]
2. **深化社区协作**：**邀请 HPC 专家、学生、开发者参与框架迭代** ——既可将自定义 CUDA 问题通过 GitHub Pull Request 提交至数据集，也可在 GitHub Issues 反馈评估逻辑，共同完善基准测试的全面性与公正性。

## 结语

> 从 2025.1 的“建立基准”到 2025.2 的“难度升级”，ComputeEval 始终以“真实场景、社区驱动”为核心，逐步揭开 LLMs 在 CUDA 编程领域的能力边界。

也有用户在 Github 提问[6]：**提问者确认这一“ correctness-first ”设计是否官方初衷**，并建议后续版本像 KernelBench、robust-kbench 那样加入对 kernel 优化水平与执行速度的评估维度，使 benchmark 能同时检验代码质量与性能优劣，帮助社区全面比较模型生成代码的实用性。

![文章插图](/articles/2025-12-08-nvidia-computeeval-llms-cuda-2ca5b63f/images/02.webp)

仓库作者回答：compute-eval **当前“正确性优先”是刻意为之，先确保模型能生成可编译且结果正确的 CUDA 代码**；性能评估已在路线图，:mdnice-bold-italic[下一版将加入 kernel 执行速度与优化质量指标，成为兼顾正确与高效的综合基准，但暂无明确时间表。]

尽管当前 LLMs 仍难以完全驾驭复杂 GPU 编程，但 ComputeEval 的存在为开发者提供了可信赖的评估工具，也为 LLM 研发者指明了优化方向——未来，随着更多 CUDA-X 库的融入和社区贡献的积累，AI 辅助高性能 GPU 开发的时代或将加速到来。

参考资料

[1] https://github.com/nvidia/compute-eval: *https://github.com/nvidia/compute-eval*

[2] https://huggingface.co/datasets/nvidia/compute-eval: *https://huggingface.co/datasets/nvidia/compute-eval*

[3] DATASET_CARD: *https://github.com/NVIDIA/compute-eval/blob/main/DATASET_CARD.md*

[4] 《Announcing ComputeEval, an Open Source Framework for Evaluating LLMs on CUDA》: *https://developer.nvidia.com/blog/announcing-computeeval-an-open-source-framework-for-evaluating-llms-on-cuda/*

[5] 《Benchmarking LLMs on AI-Generated CUDA Code with ComputeEval 2025.2》: *https://developer.nvidia.com/blog/benchmarking-llms-on-ai-generated-cuda-code-with-computeeval-2025-2/*

[6] compute-eval: correctness-focused benchmark, not performance-oriented?: *https://github.com/NVIDIA/compute-eval/issues/7*

更多推荐

- [NVIDIA 技术博客：基于 DeepSeek-R1 与推理计算扩展的 GPU Kernel 自动生成](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893770&idx=1&sn=ea8ab799f7c0f09aee4ea9c3cc1ed492&scene=21#wechat_redirect)
- [模块化 Triton GPU 内核代码生成 Agent 架构 GEAK：生成正确率达 63%，速度提升 2.59 倍！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890465&idx=1&sn=a73eec7d10391ef7371018529fb2405c&scene=21#wechat_redirect)
- [超越AutoTriton/KernelLLM的 Triton 内核生成模型 TritonRL：融合SFT蒸馏与RL分层奖励设计](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893752&idx=1&sn=e5fdd6a30c934972634fa013748e8333&scene=21#wechat_redirect)

加群
