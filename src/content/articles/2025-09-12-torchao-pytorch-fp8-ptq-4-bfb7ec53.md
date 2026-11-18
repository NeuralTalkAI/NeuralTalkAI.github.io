关键词：TorchAO、模型量化、稀疏性（Sparsity）、端到端模型优化、大语言模型、PyTorch 原生（PyTorch-Native）

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/01.webp)

- TorchAO: PyTorch-Native Training-to-Serving Model Optimization
- https://arxiv.org/abs/2507.16099
- https://github.com/pytorch/ao/
- 本文 6134 字，阅读需 20 分钟，播客 7 分钟

相关推荐

- [面向昇腾 NPU 的 PyTorch 与 MindSpore 框架性能评估：适配开源软件栈 Vs. 内部软件栈](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887921&idx=1&sn=37a10e5a7745d17d4b3d76469fa142cf&scene=21#wechat_redirect)
- [端到端加速4.5倍！端侧SoC动态稀疏注意力 shadowAttn：关键token稀疏计算架构且兼容主流框架释放NPU性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891725&idx=1&sn=1c315204fb9f1edc0b26e679d8561983&scene=21#wechat_redirect)
- [ACL'25 TTFT 最高降 81.3%！MobiLoRA：端侧 LoRA-LLM 推理框架！基于上下文感知和KV缓存优化加速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890383&idx=1&sn=d106a2482a5c13e1d66e1ddf37a3c698&scene=21#wechat_redirect)

> 我们提出 TorchAO——一个 PyTorch 原生的模型优化框架，该框架**利用量化与稀疏化技术**，为 AI 模型提供**端到端的训练到部署工作流。**

TorchAO 支持多种主流模型优化技术，包括 FP8 量化训练、量化感知训练（QAT）、训练后量化（PTQ）以及 **2:4 稀疏化**；同时，该框架还利用一种新颖的张量子类抽象，来表示多种广泛使用的、与后端无关的低精度数据类型，包括 INT4、INT8、FP8、**MXFP4、MXFP6 和 MXFP8**。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/02.webp)

::: block-1
在**模型优化流程的每一步** 中，TorchAO 都**与更广泛的生态系统紧密集成**：从预训练（TorchTitan (Liang et al., 2024)）到微调（TorchTune (torchtune, 2025)、Axolotl (Axolotl, 2025)），再到部署（HuggingFace (Wolf et al., 2019)、vLLM (Kwon et al., 2023)、SGLang (Zheng et al.)、ExecuTorch (executorch, 2025)），**将原本碎片化的技术领域整合为一个统一的工作流。**
:::

近期发布的量化版 Llama 3.2 1B/3B 模型（MetaAI, 2025b）和 LlamaGuard3-8B 模型（Inan et al., 2023）均基于 TorchAO 实现；该框架已开源，地址为：https://github.com/pytorch/ao/。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/03.webp)

## 本文目录

- 一、引言
  - 1.1 预训练（Pre-training）
  - 1.2 微调（Fine-tuning）
  - 1.3 部署（Serving）
- 二、工作流：面向服务器 GPU 的 FP8 优化
  - 2.1 FP8 训练
  - 2.2 训练后优化
  - 2.3 模型部署
- 三、工作流：面向移动设备的 QAT 优化
  - 3.1 量化感知训练
  - 3.2 下转到边缘设备
- 四、实验评估
  - 4.1 FP8 训练
  - 4.2 训练后量化（PTQ）
  - 4.3 量化感知训练（QAT）
- 五、结论
- 参考文献

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/04.webp)

## 一、引言

在内容生成、文本摘要、聊天机器人、代码生成等众多应用场景中，大语言模型（LLMs）一直处于领先地位。然而，这些能力通常需要庞大的基础设施支持，顶级性能模型的情况便可见一斑，例如 Qwen3（2350 亿参数）(Qwen3, 2025)、DeepSeek-v3（6710 亿参数）(Guo et al., 2025)、Llama 3.1（4050 亿参数）(Grattafiori et al., 2024)以及 Llama 4 Behemoth（2 万亿参数）(MetaAI, 2025a)。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/02.webp)

这些模型的计算成本和内存占用，在 LLM 流程的每一步（从训练、微调到部署）都带来了巨大挑战。例如，训练 Llama 3.1 在 1.6 万块 H100 GPU 上耗费了 3084 万 GPU 小时(Grattafiori et al., 2024)；即便以模型原始的 BF16 精度进行部署，仅加载模型就需要至少 800GB 的总内存，这已超出了配备 8 块 H100 GPU 的单台服务器的内存限制。即便在 10-80 亿参数这一较小规模下，减小模型体积对于在移动设备、边缘设备等资源受限环境中部署模型也至关重要。

> 然而，现有的 :mdnice-bold-italic[LLM 优化流程高度碎片化。]

例如，研究人员可能会利用 Transformer Engine（NVIDIA, 2025）中的 FP8/BF16 混合精度支持来预训练模型，将预训练模型加载到 Unsloth（Han & Han, 2025）或 Axolotl（Axolotl, 2025）中进行进一步微调，通过 bitsandbytes（Dettmers et al., 2022）实现量化，最终使用 llama.cpp（GGML, 2025）进行模型部署。**在每个步骤中，用户可能需要手动转换模型格式** （例如，从 HuggingFace 的 safetensors 格式转换为 llama.cpp 中的 GGUF 格式），而且当前步骤使用的量化方案可能与之前步骤存在细微差异，导致方案不一致。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/02.webp)

> 我们提出的 TorchAO 是一个 **PyTorch 原生的模型优化框架**，它**借助量化和稀疏化技术，为 AI 模型提供端到端的“训练到部署”工作流**。在模型优化流程的每一步（图 1），TorchAO 都与更广泛的生态系统紧密集成：

### 1.1 预训练（Pre-training）

TorchAO 的 FP8 训练支持（§2.1）可与 PyTorch 的各类特性原生结合，包括 torch.compile、autograd、FSDP2 以及张量并行支持；将其整合到 TorchTitan 的预训练流程中（Liang et al., 2024），在模型质量几乎无损失的前提下，可实现吞吐量提升（4050 亿参数规模下约提升 1.5 倍）（Wright et al., 2024）。

### 1.2 微调（Fine-tuning）

> 量化感知训练（QAT）是一种主流技术，通过在训练过程中模拟量化数值特性来缓解量化精度损失。

- TorchAO 的 :mdnice-bold-italic[QAT 支持可恢复量化精度损失的 96%（Or et al., 2024），并且能与 LoRA（Hu et al., 2022）结合使用] ——**相较于常规 QAT，这种组合可将训练吞吐量提升 1.89 倍** （Or, 2024；MetaAI, 2025b）。
- 此外，TorchAO 还**为 QLoRA（Dettmers et al., 2023）提供 NF4 数据类型**，以进一步降低训练过程中的资源需求。

TorchTune（torchtune, 2025）已将 TorchAO 的 QAT、NF4 以及 FP8 训练支持原生整合到其微调方案中。Axolotl（Axolotl, 2025）也提供了利用 TorchAO QAT 的微调工作流，其中包括一种将 QAT 与直接偏好优化（DPO）（Rafailov et al., 2023）相结合的方案。

### 1.3 部署（Serving）

> TorchAO 提供灵活支持，可兼容多种与后端无关的训练后量化（PTQ）方案，以便在部署前对模型进行优化。

- 对于服务器后端，TorchAO 的 PTQ 支持可作为可选量化转换步骤，在 SGLang（PyTorch et al., 2024）或 vLLM（vLLM, 2025）部署前应用。
- 对于边缘设备和移动设备后端，用户可通过 ExecuTorch（ExecuTorch, 2025；MetaAI, 2025b）将经 TorchAO 量化的模型转换为适配格式——**ExecuTorch 提供轻量级运行时和静态内存规划**，可降低性能开销和功耗；同时，用户可借助我们**自定义的量化 ARM CPU 和 Metal 内核** （torchao, 2025a），在设备端部署并运行模型。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/05.webp)

**TorchAO 还与 HuggingFace Transformers（Wolf et al., 2019）和 Diffusers（von Platen et al.）这两个主流机器学习模型定义框架紧密集成** （上面代码清单 1）。更具体地说，TorchAO 已作为量化后端之一原生整合到 HuggingFace 中：:mdnice-bold-italic[用户从 HuggingFace Hub 加载模型时，可选择应用训练后量化]。我们的整合还支持通过 HuggingFace 原生 API（如 save_pretrained、load_pretrained 和 push_to_hub）实现序列化，从而帮助用户无缝保存、加载量化模型，以便后续在目标部署框架上进行推理或生成任务。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/06.webp)

此外，**用户也可直接调用 TorchAO 的单行 API 对模型进行优化（图 2）**。例如，若用户希望使用自定义训练循环，或采用尚未与 TorchAO 整合的训练框架进行 FP8 预训练或 QAT 微调，便可通过这种方式操作。更多详细用法见下图。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/07.webp)

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/08.webp)

在后续章节中，我们将介绍两个端到端的示例工作流：它们将采用 TorchAO 中不同的模型优化技术，针对不同后端进行适配。

::: block-1
- :mdnice-bold-italic[第一个工作流以 FP8 训练和推理为核心，目标后端为服务器 GPU] （§2）；
- **第二个工作流将借助 QAT 缓解量化精度损失，并将模型转换为适配 XNNPACK** （Google, 2025）后端的格式——XNNPACK 提供经过优化的 ARM 内核，被:mdnice-bold-italic[主流安卓和 iOS 手机] 所采用（§3）。
:::

## 二、工作流：面向服务器 GPU 的 FP8 优化

> 第一个端到端工作流利用 TorchAO 的 FP8 训练支持，分别通过 TorchTitan 和 TorchTune 对模型进行预训练与微调，随后采用相同的 FP8 配置在 vLLM 中部署训练后的模型。

通常情况下，预训练是一个耗时的过程，需在 C4（allenai, 2025）等大型通用数据集上进行训练；而微调则是一个独立步骤，用于将预训练模型适配到更具领域针对性的任务中。**在本示例工作流中，两个训练步骤均借助 TorchAO 的“带张量级缩放的动态 FP8 训练”来加速训练过程**。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/09.webp)

在工作流的每个步骤中，用户可选择将预训练或微调后的检查点（checkpoint）上传至 HuggingFace Hub，以便社区中的其他研究者继续对模型进行微调，或直接将其用于推理（参见代码清单 2）。

### 2.1 FP8 训练

> TorchAO 的 FP8 训练会将激活值、权重和梯度动态转换为 FP8 格式，并利用专用的 GEMM（通用矩阵乘法）核，以充分发挥底层 GPU 中 FP8 张量核心（NVIDIA, 2023）的性能。对于那些大部分 GEMM 操作规模足够大的模型训练而言，该技术尤为实用——在这类场景中，**使用 FP8 张量核心带来的加速效果会超过动态量化的开销**，如下图。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/10.webp)

我们支持三种 FP8 训练缩放（Scaling）方案，每种方案在吞吐量与准确率之间有着不同的权衡：张量级（tensorwise）、行级（rowwise）以及“带高精度梯度权重的行级”（rowwise with high precision grad weight）。

::: block-1
- **tensorwise（张量级）**：这是默认方案，在性能与精度之间实现了合理的权衡。它为每个张量计算单个缩放因子。该技术的开销最低、性能最高，但对异常值更为敏感——因为张量中任意位置的单个异常值都会影响整个张量的缩放。这可能导致更高的量化误差，因为更多数值可能下溢至 0。在使用 FSDP（完全共享数据并行）进行训练时，tensorwise 缩放还支持一项额外优化——`enable fp8 all gather`，该优化会在 FSDP 中使用 FP8 执行全聚集（all-gathers）操作，以降低通信开销。
- **rowwise（行级）**：此方案以牺牲少量性能为代价换取了更高的精度。它会沿着左侧 GEMM（通用矩阵乘法）操作数的逻辑行、以及右侧 GEMM 操作数的逻辑列计算缩放因子。为张量更细粒度的切片计算缩放因子，可降低对异常值的敏感性，从而提升精度，但与 tensorwise 相比会产生性能损耗。
- **rowwise gw hp（行级高 precision 梯度权重）**：此方案与 rowwise 类似，但会将 $\frac{\partial L}{\partial W}$（损失函数 $L$ 对权重 $W$ 的偏导数）的计算保留在 bfloat16（脑浮点数 16 位）精度下。:mdnice-bold-italic[实验表明，$\frac{\partial L}{\partial W}$ 对低精度更为敏感，将其保留在更高精度下有助于提升模型精度]。在大多数 GEMM 操作规模小于 $M==N==K=13k$ 的模型中，该方案还能实现比 rowwise 更高的加速比。
:::

结合`torch.compile`使用时：

- 在 512 块 H100 GPU 上训练 4050 亿参数规模的模型时，张量级缩放实现了高达**1.5 倍的训练吞吐量提升** （Wright et al., 2024）；
- 而在 1920 块 H200 GPU 上训练 700 亿参数规模的模型时，行级缩放实现了高达**1.43 倍的吞吐量提升** （Wright et al., 2025）——上述提升均以 BF16 训练为基准。

| 量化方式 | 输出 token 吞吐量（tok/s） | 每个输出 token 耗时（ms） | token 间延迟（ms） |
| --- | --- | --- | --- |
| 无量化（BF16） | 103.6（+0%） | 9.50（+0%） | 9.47（+0%） |
| FP8 张量级 | 132.8（+28.2%） | 7.48（-21.2%） | 7.47（-21.1%） |

**表 1**：在 vLLM 上部署 FP8 预训练与微调后的 Llama3.1-8B 模型。与原始精度（BF16）相比，以 FP8 格式部署该模型时，吞吐量提升 28%，延迟降低 21%。两次实验中，客户端均使用 ShareGPT 数据集，且提示词（prompt）数量均为 1。

TorchAO 的 FP8 训练借助张量子类抽象（PyTorch, 2025），实现了与 PyTorch 自动微分（autograd）和 PyTorch 分布式训练的兼容性。我们的 FP8 训练支持已集成到 TorchTitan 和 TorchTune 中，因此只需极少的配置，即可将上述所有缩放方案用于预训练和微调（参见代码清单 2）。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/09.webp)

为进一步提升吞吐量，用户可借助异步张量并行（Wang et al., 2024b）——这是一种基于编译器的方法，通过重叠张量并行中的计算与通信来实现加速。具体而言，可使用 SymmetricMemory API（Wang et al., 2024a），采用基于复制引擎（copy-engine）的通信而非基于流多处理器（SM）的通信，并对计算过程进行微流水线（micro-pipelining）处理。

TorchAO 的 FP8 张量级和行级方案可与异步张量并行兼容，根据模型大小、激活检查点策略等因素，可额外提升高达 17%的训练吞吐量（torchtitan, 2025）。

### 2.2 训练后优化

> 与 FP8 训练类似，TorchAO 中的训练后量化（PTQ）和稀疏性支持均借助 PyTorch 的张量子类抽象，实现了与其他 PyTorch 特性的原生兼容性以及无缝序列化支持。

TorchAO 的 :mdnice-bold-italic[PTQ 支持多种主流数据类型，包括 INT4、INT8、FP8、MXFP4、MXFP6 和 MXFP8，且可下转到不同后端] （如 CUDA、ARM CPU）的高效核中运行。此外，:mdnice-bold-italic[TorchAO 的 PTQ 还与 GemLite（gemlite, 2025）集成，可借助专用的 Triton 核] （triton, 2025），在不同批次大小和张量并行规模下将推理速度进一步提升 1.1-2 倍（PyTorch et al., 2024）。

TorchAO 的**稀疏性支持通过利用现代 NVIDIA GPU 提供的稀疏矩阵乘法硬件支持** （Mishra et al., 2021）来加速推理。我们的基准测试表明，在 ViT 模型上，与非稀疏基线相比，**稀疏性可实现高达 1.3 倍的速度提升，且模型相对准确率保持在 91%-100%之间**。

TorchAO 为不同的稀疏性技术提供了 API 和核，包括稀疏 Marlin 2:4（Frantar et al., 2025）、2:4 稀疏性、块稀疏性、“INT8 动态量化+2:4 稀疏性”以及“行级 FP8+2:4 稀疏性”（适用于权重和激活值）（Haziza et al., 2025）。

有关量化和稀疏性的详细基准测试数据，请参考相应的 README 文档（torchao, 2025c;d）。

### 2.3 模型部署

TorchAO 的 PTQ 支持已作为可选的量化转换模块，与 vLLM（Kwon et al., 2023）、SGLang（Zheng et al.）等主流模型部署框架紧密集成，成为其量化后端之一。具体而言，我们的 FP8 推理支持采用与 FP8 训练相同的配置，确保工作流中不同步骤间具有一致的端到端数值特性。在 Llama3.1-8B 模型上的初步部署基准测试显示：与原始精度（BF16）相比，以 FP8 格式部署时，吞吐量提升 28%，延迟降低 21%（参见表 1）。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/11.webp)

```
torchtitan/run_train.sh  # 使用TorchTitan进行FP8预训练--training.compile--model.converters="float8"# 使用TorchTune进行FP8微调tune run --nnodes 1 --nproc_per_node 8full_finetune_distributed --config llama3.1/8B_fullenable_fp8_training="true"# 将FP8模型上传至HuggingFace Hub（可选，代码未展示）# 通过vLLM部署FP8模型vllm serve <HF_USER_ID>/Llama-3.1-8B-Instruct-fp8--tokenizer meta-llama/Llama-3.1-8B
```

**代码清单 2**：FP8 端到端示例流程。用户借助 TorchAO 的 FP8 训练支持，分别通过 TorchTitan 和 TorchTune 对模型进行预训练与微调，随后（可选）将训练后的模型上传至 HuggingFace Hub，并通过 vLLM 进行部署。为简洁起见，已省略详细的 FP8 训练选项。

## 三、工作流：面向移动设备的 QAT 优化

> 第二个端到端工作流利用 TorchAO 的量化感知训练（QAT）支持，通过 TorchTune 对模型进行微调，随后通过 ExecuTorch 将模型下转到 XNNPACK 后端（Google, 2025）——该后端提供经过优化的 ARM 核，被大多数主流安卓（Android）和 iOS 手机所采用，从而实现模型在移动电话、智能眼镜等可穿戴设备上的部署。

### 3.1 量化感知训练

量化感知训练（QAT）指在模型中插入“伪量化操作”（fake quantization operations），这些操作会在训练过程中模拟量化过程。这一机制能让模型学习对量化误差具有鲁棒性，从而在最终进行训练后量化时提升模型准确率。TorchAO 提供简洁且灵活的 QAT API，支持用户指定不同的量化方案（例如，INT8 动态激活值搭配 INT4 权重），并提供相应的 PTQ 配置以确保端到端数值一致性（详见下面清单 7）。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/08.webp)

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/12.webp)

> TorchAO 的 QAT 支持已集成到 TorchTune 和 Axolotl 的微调工作流中，可轻松对计划进行量化的模型进行微调（参见代码清单 3）。

在微调过程中，所有“伪量化操作”仍在高精度（如 BF16）下执行，仅模拟低精度（如 INT4）的数值特性。最终得到的 QAT 检查点与原始检查点具有完全相同的模型结构，因此可作为即插即用的替代品——其训练后量化准确率更高，且推理速度与原始模型完全一致。此外，TorchTune 还提供了一种将 QAT 与 LoRA（Hu et al., 2022）结合的配方（recipe），以降低额外伪量化操作的开销，**与普通 QAT 相比，训练吞吐量提升 1.89 倍** （Or, 2024; MetaAI, 2025b）。

近期发布的量化版 Llama 3.2 1B/3B（MetaAI, 2025b）和 LlamaGuard3-8B（Inan et al., 2023）模型，均借助 TorchAO 的 QAT 支持，缓解了面向 ARM CPU 后端的 INT4 检查点的量化性能损失。**与原始 BF16 检查点相比，这些量化模型的推理速度提升 2-4 倍，模型大小减少 56%，内存占用减少 41%，** 同时在各类推理任务中仍保持着具有竞争力的性能。

### 3.2 下转到边缘设备

通过 TorchAO 的 QAT 或 PTQ API **量化后的模型，可通过 ExecuTorch 轻松下转到边缘后端** ——ExecuTorch 提供的转换功能经过适配，能原生匹配 TorchAO 的量化模式（参见代码清单 3）。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/12.webp)

> 这使得这些模型能高效部署到各类边缘后端，包括 Android、iOS 和 CoreML。

此外，TorchAO 还**为量化线性操作和嵌入操作提供了“低于 8 位”（sub-8-bit）的 ARM CPU 和 Metal 核**，这些核可用于：即时模型执行（eager model execution）、结合`torch.compile`使用、结合 PyTorch AOTInductor 使用，或用于导出后的模型中。有关如何将 TorchAO 量化模型导出到移动后端运行的详细说明，请参考 ExecuTorch 的 README 文档（executorch, 2024）。

## 四、实验评估

> 在本节中，我们:mdnice-bold-italic[将对 TorchAO 的 FP8 训练、训练后量化（PTQ）以及量化感知训练（QAT）支持进行评估]。所有实验均在 1-8 块 H100 GPU 上执行，每块 GPU 配备 96GB HBM3 内存。

### 4.1 FP8 训练

我们使用 TorchTitan 在 8 块 H100 GPU 上对 Llama3-8B 模型在 C4 数据集[allenai, 2025]上进行了 100 步训练基准测试。所有实验的批大小（batch size）均为 1，序列长度（sequence length）为 8192，启用了`torch.compile`，并采用了按操作选择的激活检查点（selective activation checkpointing, SAC）策略。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/13.webp)

在该工作负载下，**张量级缩放（tensorwise scaling）结合 FP8 全归约（all-gather）操作** 相较于 BF16 基准版本**实现了 1.25 倍的速度提升**，且峰值内存占用相当（表 3），同时损失曲线几乎完全一致如下图。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/14.webp)

### 4.2 训练后量化（PTQ）

我们在多种 PTQ 设置下对 Llama3.1-8B 模型进行了量化，并在单块 H100 GPU 上以批大小 1、启用`torch.compile`的配置，在 hellaswag 和 wikitext 任务上对量化后的模型进行了评估。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/15.webp)

结果显示，PTQ 将模型大小减少了 2-4 倍，推理吞吐量提升了最高 2 倍，同时在所有量化设置下，模型在 hellaswag :mdnice-bold-italic[准确率和 wikitext 单词困惑度（word perplexity）指标上与 BF16 基准模型基本保持一致（表 4）。]

### 4.3 量化感知训练（QAT）

**在 4 位（4-bits）等更低精度场景下**，仅通过 PTQ 带来的量化性能损失会更为显著，此时 **QAT 的作用会更加突出**。我们通过 TorchTune 在 4 块 H100 GPU 上对 Llama3-8B、Llama3.1-8B 和 Llama3.2-3B 模型在 OpenAssistant 对话数据集（OASST1）[OpenAssistant, 2025]上进行了 1000 步微调，以此评估 QAT 的效果。

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/16.webp)

所有实验的批大小为 8，学习率为 2e-5，权重量化组大小（weight quantization group size）为 32，并启用了激活检查点。在该工作负载下，**QAT 能够恢复量化导致的性能损失**，其中在 hellaswag 准确率上的恢复率最高达 69.8%，在 wikitext 单词困惑度上的恢复率最高达 82.8%（表 2）。

## 五、结论

在本文中，我们介绍了 TorchAO——一个 PyTorch 原生的模型优化框架，该框架深度集成于大型语言模型（LLMs）的预训练、微调与推理服务全生命周期的每一个环节。TorchAO 支持多种主流模型优化技术，包括 FP8 训练、量化感知训练（QAT）、训练后量化（PTQ）以及 2:4 稀疏化，并可适配多种后端，涵盖服务器 CPU/GPU 以及移动设备/边缘设备。我们欢迎社区开发者通过以下地址为项目贡献力量：https://github.com/pytorch/ao/。

## 参考文献

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/17.webp)

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/18.webp)

![文章插图](/articles/2025-09-12-torchao-pytorch-fp8-ptq-4-bfb7ec53/images/19.webp)
