关键词：大型语言模型 (LLM)、**能效分析**、延迟分析、**性能评测工具**、缓存分析、模型部署

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/01.webp)

- ELANA: A Simple Energy and Latency Analyzer for LLMs
- https://arxiv.org/pdf/2512.09946
- **https://github.com/enyac-group/Elana**
- 4800 字，阅读 16 分钟，**播客 3 分钟**

相关推荐

- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**
- **[剖析RISC-V不同平台性能：PMU剖析与基于LLVM硬件无关Roofline，分析SiFive、平头哥C910和进迭时空性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891030&idx=1&sn=54a3b4ba00d5f77b2dbffbf7b8355e72&scene=21#wechat_redirect)**
- **[代码开源！OSDI'25 通过类 eBPF 探测实现可编程细粒度 GPU 内核分析工具 Neutrino：跨厂商GPU剖析工具！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892532&idx=1&sn=446017b09eaec1e92fb6d77ed8301908&scene=21#wechat_redirect)**

> 大型语言模型（LLM）在各类硬件平台**部署时，延迟与能耗是核心约束**，而:mdnice-bold-italic[现有评测工具缺乏统一、轻量化的 LLM 专用框架]。为此，美国得克萨斯大学奥斯汀分校团队开源了轻量型学术友好型**评测工具 ELANA，用于 LLM 的能效与延迟分析**。

ELANA **支持多 GPU 与边缘 GPU 平台，兼容 Hugging Face 所有公开模型及 API，可轻松适配压缩或低比特模型**。

其核心功能包括：

::: block-1
- 评测模型大小、KV/SSM 缓存大小，精准测量预填充延迟（TTFT）、生成延迟（TPOT）、端到端延迟（TTLT）三类关键延迟指标
- 每令牌、每提示、每请求的能耗数据
- 提供细粒度内核级分析（支持 Perfetto 可视化，Perfetto 是一款 Google 开发的开源系统级性能分析工具）。
:::

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/02.webp)

::: image-caption
图 1b：Perfetto 工具上的细粒度核级分析结果示例
:::

相较于通用工具 Zeus，**ELANA 无需修改代码，输出更精细，专注于 LLM 标准化推理场景**。实验在 A6000 云 GPU 及 Jetson 系列边缘设备上验证了其有效性，覆盖 Llama-3、Qwen-2.5 等主流模型。该工具为 LLM 研究与部署提供了统一基准，助力开发者定位性能瓶颈，推动高效 LLM 的研发与落地。

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/03.webp)

## 本文目录

- 一、引言
- 二、主要功能与分析结果
  - 2.1 Hugging Face 接口
  - 2.2 模型大小分析
  - 2.3 延迟分析
  - 2.4 能耗分析
  - 2.5 细粒度内核级分析
- 三、安装和使用
  - 3.1 环境搭建与安装
  - 3.2 核心功能使用
  - 3.3 补充说明
- 四、结论
- 参考文献

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/04.webp)

## 一、引言

> 如今，众多新兴应用均由大型语言模型（LLMs）提供支持。然而，部署参数规模达数十亿（例如 1000 亿）的模型时，在满足所需的推理延迟、内存占用和能耗成本方面面临着巨大挑战。

**已有大量研究致力于探索量化** [Xiao et al., 2023, Lin et al., 2024a,b, Chiang et al., 2025a]和**压缩技术** [Wang et al., 2025, Lin et al., 2025, Chiang et al., 2025b]，以降低模型部署时的推理延迟和模型大小。但这些研究方向主要**侧重于算法设计，在评估过程中却忽略了能耗问题**。此外，现有的分析基准测试与结果在很大程度上依赖于研究人员各自开发的工具，目前仍**缺乏一个统一且公平的分析框架**。

近期已有研究关注机器学习系统的能耗特征，包括跨云服务器与边缘设备，对训练和推理过程中的能耗进行测量[You et al., 2023, Tschand et al., 2025]、基准测试[Chung et al., 2025, Samsi et al., 2023, Krupp et al., 2025]及优化[You et al., 2023, Chung et al., 2024]。

- 其中，Zeus 工具[You et al., 2023]是一款通用的 GPU 能耗/时间分析器，可用于分析任意 Python 代码块。它还包含一个命令行界面（CLI），能够报告监控进程运行期间 GPU 消耗的总能耗。

::: block-1
尽管如此，**目前仍缺少一款轻量级的命令行分析器** ——该分析器需专为标准化 LLM 推理场景设计，且能为机器学习开发者提供细粒度延迟分析、能耗测量与核级（kernel-level）分析功能的组合。
:::

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/05.webp)

::: image-caption
ELANA是一款轻量级、适合学术研究的分析工具，用于在多GPU和边缘GPU平台上分析大型语言模型的模型大小、预填充延迟（TTFT）、生成延迟（TPOT）和端到端延迟（TTLT）。它提供简单的命令行界面和可选的能耗日志记录功能，非常适合高效大型语言模型的研究工作。
:::

为填补这一空白，我们发布了 ELANA——一款用于 LLM 基准测试的精简分析框架。ELANA 提供了对学术研究友好的界面，可用于测量评估 LLM 性能的关键指标：

::: block-1
- 包括模型大小、键值缓存（KV 缓存）占用空间、预填充延迟（首 token 生成时间，Time-to-First-Token, TTFT）、生成延迟（每输出 token 时间，Time-per-Output-Token, TPOT），
- 跨多 GPU 和边缘 GPU 设备的请求端到端推理延迟（末 token 生成时间，Time-to-Last-Token, TTLT）。
- 我们的工具还**具备能耗分析功能，可测量“每 token 焦耳数（J/Token）”“每提示词焦耳数（J/Prompt）”和“每请求焦耳数（J/Request）”等指标。**
- 该工具**支持 Hugging Face 平台上所有可用模型，并提供极简的命令行界面。**
- 此外，ELANA 能与 Hugging Face API 无缝集成，且**可轻松扩展以处理压缩模型或低比特精度模型**，使其成为高效 LLM 研究与原型开发的实用解决方案。
:::

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/06.webp)

::: image-caption
表 1：我们的 ELANA 与 Zeus 分析框架[You et al., 2023]的对比
:::

## 二、主要功能与分析结果

> 本节将介绍 ELANA 的主要功能及其分析结果。为演示 ELANA 的使用方法，我们在 **A6000 GPU（云端）** 以及 **Jetson AGX Thor 128GB** 和 **Orin Nano 8GB** 设备（边缘端）上，对多个模型进行了分析，包括 **Llama-3.1-8B** [Meta, 2024]、**Qwen-2.5-7B** [Hui et al., 2024]和**混合模型 Nemotron-H-8B** [Blakeman et al., 2025]。对于 Orin Nano，我们分析的是小型语言模型，如 Llama-3.2-1B、Qwen2.5-1.5B。

### 2.1 Hugging Face 接口

ELANA 设计为使用主流的 Hugging Face 接口来构建待分析的模型，如下列代码块所示。因此，ELANA 能够支持并分析 Hugging Face 平台上发布的模型。

```
def _build_model_and_tokenizer(self):    tokenizer = AutoTokenizer.from_pretrained("model_hf_repo")  # 从Hugging Face仓库加载分词器    model = AutoModelForCausalLM.from_pretrained("model_hf_repo")  # 从Hugging Face仓库加载因果语言模型
```

这一设计选择使得研究人员只需修改少量代码，即可将 ELANA 与新兴的模型架构及新开发的压缩算法集成。例如：

```
def _build_model_and_tokenizer(self):    tokenizer = MyLocalTokenizer.from_pretrained("local_model_path")  # 本地分词器（自定义本地分词器）    model = MyLocalModel.from_pretrained("local_model_path")  # 本地模型（自定义本地模型）
```

总而言之，我们旨在为未来的研究人员提供一款简单且统一的工具，用于评估其新架构和新算法，无需自行开发分析工具。

### 2.2 模型大小分析

我们采用大多数存储厂商所使用的国际单位制（SI，十进制，base-10）定义作为分析和报告模型大小与缓存大小的默认单位（即 $1\ \text{GB} = 1000^3$ 字节）。ELANA 还提供二进制单位（GiB，其中 $1\ \text{GiB}=1024^3$ 字节）作为可选的内存报告单位，该单位在 Linux 及其他操作系统中较为常用。

:mdnice-bold-italic[参数与缓冲区大小：ELANA 会报告模型的总参数大小（包括可训练权重和不可训练权重），以帮助用户了解所部署模型的内存占用情况]。除参数外，ELANA 还会分析辅助缓冲区（如位置嵌入向量、量化层等）的大小。这一功能使从业者能够比较不同的压缩算法，并**识别对内存占用贡献最大的组件** ——这在内存受限的边缘设备上部署 LLM 时尤为重要。

**键值缓存（KV 缓存）与状态空间模型缓存（SSM 缓存）大小**：在自回归生成过程中，LLM 会维护中间状态，例如 Transformer 模型中的键值缓存（KV 缓存）或状态空间模型（SSMs）中的递归状态缓存。这些缓存通常是内存消耗的主要来源，尤其是在长序列生成或多请求（即大批次大小）服务场景下。ELANA 能够:mdnice-bold-italic[估算基于注意力机制的模型的 KV 缓存大小，以及基于状态空间模型（SSM）架构的状态缓存大小]，从而帮助用户评估不同服务负载下的内存需求。

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/07.webp)

::: image-caption
表 2 ：模型与缓存大小分析结果。展示了我们对 Llama-3.1-8B、Qwen-2.5-7B 和 Nemotron-H-8B 在不同负载下的参数大小与 KV 缓存大小的分析结果，所有结果均以 GB 为单位
:::

### 2.3 延迟分析

:mdnice-bold-italic[首 token 生成时间（TTFT，预填充阶段）：TTFT 用于测量预填充阶段（即处理输入提示词的阶段，模型在生成第一个输出 token 之前会先处理整个输入提示词）的延迟]。该指标反映了初始前向传播的延迟，在聊天助手、长上下文摘要等交互式应用中尤为重要。ELANA :mdnice-bold-italic[通过隔离预填充阶段，并报告多次运行的原始延迟与平均统计数据，实现对 TTFT 的精准测量]。我们使用随机生成的输入提示词对模型进行预填充，并分析 TTFT 的延迟。由于实际应用中输入提示词的长度各不相同，因此在模型推理的预填充阶段，我们不缓存 CUDA 图（CUDA graphs，一种用于优化重复计算任务的 CUDA 特性，可减少内核启动开销）。

:mdnice-bold-italic[每输出 token 时间（TPOT，生成阶段延迟）：TPOT 用于衡量自回归生成（即解码）过程中生成每个 token 的平均解码延迟]。由于解码阶段本质上是串行的，因此 TPOT 是评估模型在连续 token 生成场景下效率的关键指标。ELANA :mdnice-bold-italic[通过记录 token 间的生成间隔，并在整个输出序列上取平均值来计算 TPOT]。在分析 TPOT 之前，我们会使用用户指定提示词长度的随机生成输入来预填充 KV 缓存。:mdnice-bold-italic[为最大化生成过程的吞吐量，我们参考 TensorRT-LLM[NVIDIA, 2023]和 SGLang[Zheng et al., 2024]的做法，对生成阶段的 CUDA 图进行缓存。]

:mdnice-bold-italic[末 token 生成时间（TTLT，端到端延迟）：TTLT 用于测量推理过程的完整端到端延迟，即从接收输入提示词到生成最后一个输出 token 的总时间]。该指标结合了预填充延迟与解码延迟，:mdnice-bold-italic[能够全面反映处理请求的运行时性能]。ELANA 在报告 TTLT 的同时，还会将其分解为 TTFT 和 TPOT 两部分，使从业者能够分析性能瓶颈，并理解不同优化手段对整体推理体验的影响。我们使用随机生成的输入提示词分析 TTLT，并测量不同提示词长度和生成长度下，处理一批请求的端到端延迟。

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/08.webp)

::: image-caption
表 3：A6000 GPU 上的延迟与能耗分析结果。我们在 A6000 GPU 上对各类负载进行了分析，展示了平均能耗（单位：焦耳，J），其中，除“每请求焦耳数（J/Req.）”为 20 次运行的平均值外，其余指标均为 100 次运行的平均值。预填充 token 数与生成 token 数统一表示为 $L=T_{p}+T_{g}$（其中 $T_p$ 为预填充 token 数，$T_g$ 为生成 token 数）。延迟与能耗的单位分别为毫秒（ms）和焦耳（J）
:::

表 3 报告了在 A6000 GPU 上所有负载的平均延迟（单位：毫秒，ms），其中除 TTLT 为 20 次运行的平均值外，其余指标均为 100 次运行的平均值。我们还在 Jetson AGX Thor（128 GB 统一内存）和 Orin Nano（8 GB 统一内存）这两款边缘 GPU 上对 ELANA 进行了测试，不同负载在 Jetson 系列 GPU 上的分析结果如表 4 所示。

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/09.webp)

::: image-caption
表 4：Jetson 系列设备上的延迟与能耗分析结果。我们在 Jetson AGX Thor 128G 和 Orin Nano 8G 设备上对各类负载进行了分析。预填充 token 数与生成 token 数统一表示为 $L=T_{p}+T_{g}$（其中 $T_p$ 为预填充 token 数，$T_g$ 为生成 token 数）。延迟与能耗的单位分别为毫秒（ms）和焦耳（J）
:::

表 4 还报告了实验模型在 Jetson AGX Thor 和 Orin Nano 设备上的能耗成本。

### 2.4 能耗分析

> 在能耗分析方面，我们通过 NVIDIA 的 NVML 接口（NVIDIA Management Library，NVIDIA 管理库，用于监控和管理 NVIDIA GPU 设备），使用 pynvml 库（NVML 的 Python 绑定库）查询目标 GPU 的瞬时功率。

在 Jetson 设备上，我们通过 jetson-stats 工具包中的 jtop 工具（一款用于 Jetson 系列设备的系统监控工具，可读取板载功率传感器数据）获取系统级芯片（SoC）上 GPU 的功率。

**我们每 0.1 秒对功率使用情况进行一次采样，并记录所有测量数据**。在延迟分析过程中，会有一个独立进程并行运行以收集功率数据，我们会计算对应测量窗口内的平均功率。将该平均功率与测得的延迟相结合，即可得到能耗数据。

::: block-1
- 我们针对 TTFT 报告“每提示词焦耳数（J/Prompt）”
- 针对 TPOT 报告“每 token 焦耳数（J/Token）”
- 针对 TTLT 报告“每请求焦耳数（J/Request）”
:::

在多 GPU 场景下，我们会将所有参与计算的 GPU 的平均功率求和，以计算最终的能耗指标。

### 2.5 细粒度内核级分析

:mdnice-bold-italic[PyTorch 分析器（PyTorch Profiler）：为支持细粒度分析，ELANA 可选择性集成 PyTorch 分析器 ∗ 和整体轨迹分析（HTA）†，以捕获低层级执行轨迹、算子运行时间和核级统计数据]。

::: block-1
- PyTorch 分析器——全新且改进的性能工具：https://docs.pytorch.org/docs/stable/profiler.html
- 整体轨迹分析（Holistic Trace Analysis）：https://github.com/facebookresearch/HolisticTraceAnalysis
:::

细粒度内核级分析的功能使用户能够诊断性能瓶颈、分析 GPU 利用率，并检查核融合、量化或模型压缩技术的影响。分析结果可导出至 Perfetto‡等工具进行可视化，以便进一步分析——这提供了超越高层级延迟指标的、更详细的运行时行为视图。

::: block-1
- Perfetto——系统分析、应用追踪与轨迹分析工具：https://ui.perfetto.dev/
:::

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/10.webp)

::: image-caption
图 1a：Perfetto 工具上的细粒度核级分析结果示例
:::

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/02.webp)

::: image-caption
图 1b：Perfetto 工具上的细粒度核级分析结果示例
:::

## 三、安装和使用

### 3.1 环境搭建与安装

```
# 1. 克隆仓库git clone https://github.com/hychiang-git/Elana.gitcd Elana# 2. 创建conda环境（可选）conda create -n elana-env python==3.12  # 创建环境并指定Python版本conda activate elana-env  # 激活环境# 3. 若用virtualenv（替代conda）# python3 -m venv elana-env  # 创建虚拟环境# source elana-env/bin/activate  # 激活环境（Linux/macOS）# pip install --upgrade pip  # 升级pip# 4. 安装Elana及依赖pip install .
```

### 3.2 核心功能使用

```
# 1. 分析预填充延迟（TTFT）+ 能耗elana meta-llama/Llama-3.2-3B-Instruct --ttft --energy# 2. 分析生成延迟（TPOT）+ 能耗 + 缓存图elana meta-llama/Llama-3.2-3B-Instruct --tpot --energy --cache_graph# 3. 分析端到端延迟（TTLT）+ 能耗 + 缓存图elana meta-llama/Llama-3.2-3B-Instruct --ttlt --energy --cache_graph# 4. 分析模型大小elana meta-llama/Llama-3.2-3B-Instruct --size# 5. 带Torch Profiler分析（生成trace文件，用于Perfetto可视化）elana meta-llama/Llama-3.2-3B-Instruct --tpot --energy --cache_graph --torch_profile# 6. 查看所有用法elana --help
```

### 3.3 补充说明

1. **Perfetto可视化**：运行带`--torch_profile`的命令后，trace文件会生成在`torch_profile/{model_name}/`目录下，可上传至 Perfetto：https://ui.perfetto.dev/ 进一步查看内核级详情。
2. **模型替换**：命令中的模型可替换为任意HuggingFace `AutoModelForCausalLM` 类型模型。

## 四、结论

> ELANA 是一个轻量级、易用且可扩展的框架，用于评估跨云端和边缘 GPU 的大型语言模型（LLMs）效率。

通过提供对模型大小、键值缓存（KV 缓存）内存、延迟特征以及可选能耗的全面分析，ELANA 能使从业者和研究人员都更深入地了解现代 LLMs 的性能瓶颈。其与 Hugging Face API 的兼容性，以及对新开发模型或压缩算法的适配性，进一步使 ELANA 成为推动高效模型设计与部署研究的实用工具。

我们希望 ELANA 能成为可复现基准测试的基础，推动不同模型与系统间的公平比较，并加速下一代资源高效型 LLMs 的开发。

## 参考文献

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/11.webp)

![文章插图](/articles/2025-12-14-llm-elana-token-kv-c38ba86f/images/12.webp)
