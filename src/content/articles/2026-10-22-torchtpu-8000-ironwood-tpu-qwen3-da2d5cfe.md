关键词：**TorchTPU**、PyTorch、TPU、**vLLM**、:mdnice-bold-italic[TorchTitan]

> 10 月 21 日，谷歌云 VP Mark Lohmeyer 与 Meta 工程副总裁 Bill Jia 联合署名，在谷歌云官方博客上发布了《TorchTPU is now open source: Run native PyTorch on TPUs》[1]。谷歌与 Meta PyTorch 团队共同开发的原生 TPU 软件栈，让 TPU 成为与 GPU、CPU 并列的 PyTorch 原生设备选项。

围绕它，谷歌同步打通了 vLLM、SGLang、llm-d、TorchTitan、VeRL、Ray、Hugging Face 与 LiteRT 一整条链路：

- 训练侧在 8000 颗 Ironwood TPU 上完成验证、Qwen3-8B 最高 52% MFU；
- 推理侧有客户把日生成 token 数从 14.7 亿提到 59.7 亿。![文章插图](/articles/2026-10-22-torchtpu-8000-ironwood-tpu-qwen3-da2d5cfe/images/01.webp)

- **TorchTPU is now open source: Run native PyTorch on TPUs**
- 博客：https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus
- 文档：https://torchtpu.readthedocs.io/

::: related-title
**相关推荐**
:::

::: related-links
- **[Google 第八代 TPU：面向智能体时代的训练与推理芯片，原生支持 JAX、MaxText、PyTorch、SGLang 与 vLLM](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900266&idx=1&sn=dd3219b311dfca2bba5566c5acb42fd5&scene=21#wechat_redirect)**
- **[用GPU原型验证，用TPU训练和推理！TorchTPU：谷歌 Meta 联手让 PyTorch 原生跑 TPU，3 行代码从 GPU 切换到 TPU 的极致易用性！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899873&idx=1&sn=f61bc039e138133b982b29e612d0a1fb&scene=21#wechat_redirect)**
- **[开源数千页逆向文档，覆盖 NVIDIA GPU、Google TPU、AWS Trainium 三大加速器软件栈的逆向工程全景](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905331&idx=1&sn=c10f37227ebe1fb16fdb248d7eb71333&scene=21#wechat_redirect)**

:::

> 同一天，开源仓库 google-pytorch/torch_tpu[2] 在 GitHub 公开，采用 Apache 2.0 许可证，配套文档站 torchtpu.readthedocs.io[3] 同步上线。

## 本文目录

::: related-links
- 一、开篇：TPU 成为 PyTorch 的原生选项
- 二、在 TPU 上原生运行 PyTorch 代码
- 三、推理优化：vLLM、SGLang 与 llm-d 接入 TPU
- 四、大规模训练：TorchTitan、VeRL 与 Ray
- 五、部署灵活性与 LiteRT、TPU agent suite
- 六、在谷歌云上开始使用 PyTorch
- 七、写在最后

:::

## 一、开篇：TPU 成为 PyTorch 的原生选项

> 在 PyTorch Conference 2026 上，谷歌宣布了一系列新产品与集成，帮助开发者在谷歌云或自有基础设施上高效运行 PyTorch。其中最重要的一项：TorchTPU——谷歌 TPU 的原生 PyTorch 软件栈——已经作为开源项目放上 GitHub[4]。

有了 TorchTPU，TPU 与 GPU、CPU 并列，成为 PyTorch 的原生设备选项。**开发者现在可以使用自己已经在依赖的工具在 TPU 上训练、服务和部署模型，包括 vLLM、SGLang、llm-d、TorchTitan、Ray、TorchRec、Hugging Face 和 LiteRT。** 最终，这让客户可以在 TPU、GPU、CPU 这组覆盖不同负载优化方向的计算选项之间自由选择，整个产品组合共享一致的开放软件与运维模型。

![文章插图](/articles/2026-10-22-torchtpu-8000-ironwood-tpu-qwen3-da2d5cfe/images/02.webp)

::: image-caption
图 1：谷歌博客中「Introducing TorchTPU」的发布横幅，PyTorch 与 Google Cloud TPU 两个徽标并列，直观传达了这次合作的定位——TPU 作为 PyTorch 的原生后端登场，由谷歌与 Meta 的 PyTorch 团队共同开发。这张图的发布语境是 PyTorch Conference 2026，同日开源的还有 vLLM、SGLang 的 TorchTPU 后端仓库，以及 llm-d 1.0。（图片来源：Google Cloud Blog[5]）
:::

## 二、在 TPU 上原生运行 PyTorch 代码

> **TorchTPU[6] 与 Meta 的 PyTorch 团队共同开发**，让开发者用尽量少的代码改动、用熟悉的 API 在 TPU 上运行现有的 PyTorch 工作负载。

开发者可以**先用 eager mode 调试模型，再通过 torch.compile 过渡到生产级性能，优化 TPU 利用率并降低内存开销**。通过开箱支持标准 PyTorch API、高级诊断能力和原生分布式原语，TorchTPU 从第一天起就能提供高性能与运维效率。

::: block-1
Meta PyTorch 工程总监 Jana van Greunen 表示：

在 Meta，推进 AI 前沿需要巨大的算力规模和深度的硬件多样性。:mdnice-highlight[我们不能被单一硬件架构束缚。TorchTPU 是向前迈出的重要一步——它让我们能与核心机器学习系统团队灵活地协同开发硬件，同时在关键生产场景中支撑大语言模型、广告和推荐系统等主要工作负载。] 它让我们能把 PyTorch 工作负载部署到谷歌 TPU 上，直接向工程团队反馈使用情况，构建所有人都需要的生态灵活性。
:::

谷歌还与 Hugging Face 合作，让 Hugging Face Hub 上的热门开源权重模型原生运行在 TorchTPU 上。在 Transformers v5 中，把模型部署到 TPU 只需要传入 `device_map="tpu"`，示例写法如下：

```
from transformers import AutoModelForCausalLM, AutoTokenizermodel_id = "Qwen/Qwen3.5-9B"model = AutoModelForCausalLM.from_pretrained(    model_id,    device_map="tpu",  # 模型将直接加载到 TPU 上)tokenizer = AutoTokenizer.from_pretrained(model_id)inputs = tokenizer("The future of AI is", return_tensors="pt").to("tpu")out = model.generate(**inputs, max_new_tokens=64)print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Diffusers 库也原生运行在 TPU 上，支持 eager mode、torch.compile 和张量并行，可以把大型生成模型扩展到多颗加速器。为保证可靠性，双方还搭建了专门的 CI/CD 流水线，在 TPU 基础设施上对 Hugging Face Transformers 套件做持续的每日构建测试。

::: block-1
Hugging Face 产品 VP Jeff Boudier 表示：

Hugging Face 致力于成为一个开放、中立的平台，让 AI 开发者可以选择最适合自己的模型、框架和硬件。:mdnice-bold-italic[硬件多样性是让开源 AI 惠及所有人的关键。我们与谷歌合作，把原生 TorchTPU 支持带进 Transformers 等核心库，正是这一承诺的落地。] 我们一起让社区用熟悉的 PyTorch 工作流，在谷歌云 TPU 上高效运行 Hub 上的热门模型。给开发者更多选择，整个开源生态都会成长。
:::

### 2.1 仓库里的信息：功能、安装与示例

> 从开源仓库 README[7] 可以补全博客没有展开的细节。TorchTPU 项目提供了面向谷歌 TPU 的 PyTorch 后端，主要功能包括：

- 面向 TPU 的 PyTorch ATen kernel 实现；
- 针对 XLA 计算优化执行的编译缓存；
- 分布式训练支持（数据并行、张量并行）；
- 多种模型示例（Llama、Qwen、ResNet、DLRM、minGPT、UNet）；
- 与 `torch.compile()` 集成的图模式执行；
- 调试与基准测试工具。

安装方面，Python 版本支持策略与 PyTorch 的发布兼容矩阵[8]一致，当前支持 Python 3.11 及以上。安装最新的公开发布版（PyPI 上的 0.2.0[9]）：

```
pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

需要注意，不带 `--extra-index-url https://download.pytorch.org/whl/cpu` 参数安装会默认拉取 CUDA 版本的 torch，要么按上面方式传参，要么单独安装 CPU 版本。从源码构建则使用 Bazel（项目用 Bazel 做构建与依赖管理，推荐通过 Bazelisk 安装）：

```
bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe
```

其中 `--config=no_rbe` 用于在没有 Remote Build Execution 凭据时去掉远程缓存与远程执行标志，且必须放在 `--config=wheel_common` 之后——两者都设置远程缓存相关标志，后写的配置生效。

文档入口有两个：用户指南在 torchtpu.readthedocs.io[10]，是交互式 notebook 的静态版本，notebook 源文件在仓库 docs 目录下，格式为 Marimo（类似 Jupyter），用 `marimo edit file.py` 运行，需要在 TPU VM 上操作，最省事的是开一台 v6e GCE 实例。此外仓库还提供了公共 API 列表[11]和贡献指南（需签署谷歌 CLA）。

## 三、推理优化：vLLM、SGLang 与 llm-d 接入 TPU

**随着推理模型和 Agent 模型规模扩大，推理成本与延迟成为生产环境的主要瓶颈。** 谷歌把 TorchTPU 集成进广泛使用的开源推理引擎，并配合 Kubernetes 原生的集群路由能力，在 TPU 和 GPU 之间调度，帮助开发者优化每一美元换来的吞吐量。

::: block-1
这套基础让开发者可以把负载扩展到多样硬件上，同时尽量减少针对特定平台调优的摩擦。谷歌与多个开源社区和行业伙伴合作，下面分别看三个方向：:mdnice-highlight[用 vLLM 做高吞吐推理、用 SGLang 做结构化 Agent 执行、用 llm-d 做大规模分布式集群编排。]
:::

### 3.1 vLLM：高吞吐服务与内存效率

> 谷歌与 Inferact 合作开发的 vLLM[12] TorchTPU 原生 PyTorch 后端，让开发者可以直接在 TPU 上部署开源权重模型，具备生产就绪的服务特性和定制的 Pallas kernel。

这套集成在 TPU 硬件上支持连续批处理、量化和多芯片并行，同时保留上游的 worker 模型、调度器和 API 服务器，在不要求改动现有推理栈的前提下改善成本性能比。

::: block-1
Inferact CEO 兼联合创始人 Simon Mo 表示：

**Inferact 的使命一直是让高性能推理在整个 AI 生态中可用、可靠、低成本。我们很高兴参与 TorchTPU 的发布，把 vLLM 的一流支持带到 TPU 上。** 通过把 PyTorch 优化后的 TPU 能力直接桥接进 vLLM，开发者可以立即在 TPU 上运行生产负载，获得他们期待的高吞吐、低延迟和内存效率。
:::

### 3.2 SGLang：加速 Agent 与结构化负载

> 谷歌与 RadixArk 合作开发的 SGLang[13] 原生 PyTorch 后端，在 TPU 上加速高度结构化和 Agent 类负载。:mdnice-bold-italic[SGLang 使用 RadixAttention 在不同请求间复用 KV cache，重复的提示词前缀可以完全跳过重新计算。SGLang-TorchTPU 保留了上游的 worker 模型、调度器和 API 服务器，并使用一个分离式传输后端，同时充当 HiCache 的 I/O 后端，为复杂的多步生成优化延迟与吞吐。]

::: block-1
RadixArk CTO 兼联合创始人 Banghua Zhu 表示：

在 RadixArk，随着我们在各种芯片上铺开 SGLang 支持，TorchTPU 大幅加快了我们的开发速度。它给了我们一条无痛的原生 TPU 加速路径，把可观的效率提升和灵活的算力选择直接传递给我们的客户。
:::

### 3.3 llm-d：分布式 LLM 推理

> 为了在大规模集群上运行推理引擎，谷歌与红帽、IBM 以及超过 500 名贡献者、25 家机构组成的社区共同发布 llm-d 1.0。**作为云原生计算基金会（CNCF）的 Sandbox 项目，llm-d 连接了两个开源社区：一边是 PyTorch，vLLM 和 SGLang 在这里服务模型；另一边是 Kubernetes，分布式系统在这里运行和扩展。**

llm-d 的路由器使用前缀缓存感知路由，把每个请求发给已经持有其上下文的副本，并把计算密集的 prefill 与内存密集的 decode 拆开处理。llm-d 1.0 让这些能力达到生产就绪：多租户流量控制、路由器高可用、稳定的 v1 API，以及面向 Agent 的会话感知路由。在 Google Kubernetes Engine（GKE）上，开发者可以通过统一的 llm-d 端点在 GPU 和 TPU 之间选择，为每种负载挑合适的硬件。

::: block-1
Zenteiq.ai 创始人 Sashikumaar Ganesan 教授表示：

在 Zenteiq，我们使用由 llm-d 驱动的 GKE Inference Gateway 来加速主权科学基础模型的合成数据生成。**通过负载感知和前缀缓存感知路由，我们的整体推理吞吐量提升了 4 倍以上，每 1000 颗 TPU v6e 加速器的日生成 token 量从 14.7 亿提升到 59.7 亿。** llm-d 的开源基础与 GKE 集成，让我们能高效地规模化服务多个模型家族，不用自己开发和维护定制的推理调度器。
:::

## 四、大规模训练：TorchTitan、VeRL 与 Ray

把 PyTorch 训练扩展到数千颗加速器，需要容错、对编译友好的分布式框架。谷歌已经把 TorchTPU 集成进 PyTorch 社区的训练、强化学习和编排核心框架。

### 4.1 TorchTitan：大规模分布式高性能训练

> TorchTitan[14] 是 PyTorch 原生的可扩展分布式训练参考实现，与 TorchTPU 完成了生产级集成。通过对接 PyTorch 的 torch.compile、TorchTitan 对编译友好的 SimpleFSDP、专用 Pallas kernel 和图级优化，开发者可以在 TPU 上高效地构建和训练模型。

TorchTitan 为多类模型提供预训练配方：大型稠密模型（Llama 3 70B/405B、Qwen 3 8B/32B、MuseGlimmer 30B）、混合专家模型（Qwen 3 235B）、扩散 Transformer（FLUX.1 12B），**并已在最多 8000 颗 Ironwood TPU（TPU v7x）上完成验证**。预训练之外，TorchTitan 还作为 VeRL 等后训练流水线中的可扩展训练器，承担 GRPO 和 RL 对齐任务。

在苛刻的开放架构上测试大规模效率时，**TorchTPU 经由 TorchTitan 的训练路径在 Qwen 3 8B 上实现了最高 52% 的模型浮点运算利用率（MFU），并在最多 8192 颗 Ironwood TPU 上展现出接近线性的扩展能力与稳定性。**

### 4.2 VeRL：可扩展的后训练与强化学习（预览）

> RL 是前沿实验室推进模型能力的基础技术，也是企业做领域定制的重要手段，但编排分布式 RL 负载仍是现代 AI 基础设施中最复杂的挑战之一。VeRL[15] 已经快速成为最受欢迎的 RL 框架之一，支持从可验证的数学和代码奖励到大规模多轮 Agent 工具调用的各种场景。

为了给这个活跃的社区扩展硬件选择，**谷歌向 VeRL 上游贡献了 TPU 后端和一份端到端参考配方。** 该方案基于训练器与采样器分离的架构以最大化训练效率，其中的 GRPO 参考配方运行在数学推理数据集上，展示了开发者如何在保留硬件无关的模型与算法代码的前提下，把 VeRL 负载部署到 TPU 上。

### 4.3 Ray：编排异构算力

谷歌还为 Ray[16] 加入了原生 TorchTPU 支持，可以在同一套工作流中跨 CPU、GPU 和 TPU 编排 PyTorch 的数据处理、训练、RL 和服务。只需几行资源配置，就能把异构 AI 流水线的每个阶段调度到最合适的加速器上。

::: block-1
Anyscale 首席架构师 Edward Oakes 表示：

:mdnice-bold-italic[在 Ray Summit 上我们谈到 PARK 栈——PyTorch、AI 模型、Ray 和 Kubernetes，因为这就是今天团队构建生产级 AI 的方式。TorchTPU 把谷歌 TPU 直接嵌入了这个行业标准栈。我们还与谷歌团队紧密合作，让 TPU 成为 Ray 中的一等加速器。] 配合原生 PyTorch 支持，团队现在可以在 TPU 上无缝地开发、扩展和监控自己的工作负载。
:::

## 五、部署灵活性与 LiteRT、TPU agent suite

真正的可移植性要求在不同硬件形态之间统一部署工具，包括边缘设备和云端加速器。

[让端侧AI不再难用！谷歌开源 LiteRT-CLI：用一行命令重写 TFLite 模型工作流，从 HuggingFace 到 NPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903206&idx=1&sn=419b84c1b6a835de19f43cdcf6d37e11&scene=21#wechat_redirect)

### 5.1 LiteRT 加入 PyTorch 生态

> 许多 PyTorch 模型需要运行在手机、笔记本、浏览器和嵌入式设备上。LiteRT[17]（原名 TensorFlow Lite，谷歌的高性能端侧 AI 推理框架）正式加入 PyTorch 生态。

通过 LiteRT-Torch[18]，一次 `litert-torch.convert()` 调用就能把 PyTorch 模型转换过来。它构建在 `torch.export()` 和 Core ATen 算子之上，模型代码仍留在 PyTorch 体系内。

对大语言模型，LiteRT-Torch 可以把 Hugging Face Transformer 模型导出到 LiteRT-LM[19] 运行。转换后的模型可以跑在 Android、iOS、macOS、Windows、Linux、Web 和物联网设备上，硬件加速覆盖 CPU（经 XNNPack）、GPU（经开源的 ML Drift[20]），以及通过统一 API 适配主要芯片平台的 NPU。

### 5.2 TPU agent suite 进入早期预览

> 为推动 PyTorch 在 TPU 上的采用，谷歌宣布 TPU agent suite[21] 在 GitHub 上预览发布。**该工具通过插件与多种代码编排工具和模型组合集成，帮助自动化完成大量 PyTorch 模型在 TPU 上的上手部署、问题诊断、用户问答和推理性能调优。**

![文章插图](/articles/2026-10-22-torchtpu-8000-ironwood-tpu-qwen3-da2d5cfe/images/03.webp)

::: image-caption
图 2：「Introducing the TPU agent suite」横幅，PyTorch 与 Google Cloud 徽标并列。这套工具的定位是把 TPU 使用中的琐碎环节交给编码 agent：部署上手、故障诊断、答疑和推理性能调优，通过插件接入现有的代码编排工具链，目前处于早期预览阶段，仓库已在 GitHub 公开。（图片来源：Google Cloud Blog[22]）
:::

## 六、在谷歌云上开始使用 PyTorch

> 这一系列发布共同强化了谷歌的承诺：让 TPU 软件生态成为对 PyTorch 社区开放、友好的环境。**通过把原生 TPU 支持上游化到 PyTorch 技术栈——TorchTPU、Hugging Face、vLLM、SGLang、llm-d、TorchTitan、VeRL 和 Ray** ——并把 LiteRT 贡献给 PyTorch 生态，谷歌给了开发者按自己的方式构建、适配和扩展 AI 的灵活性。

想今天在 TPU 上跑起 PyTorch，可以访问 Google Cloud 文档[23]开始。

## 七、写在最后

> 这次发布的看点不只在 TorchTPU 本身。

**训练侧的 52% MFU 和 8192 颗 TPU 近线性扩展、推理侧 Zenteiq 实测的 4 倍吞吐**，都是需要第三方复现验证的厂商自报数字；但它把 TPU 从「PyTorch 生态的旁路」变成了与 GPU、CPU 并列的原生设备选项，且 vLLM、SGLang、llm-d、TorchTitan、VeRL、Ray 的后端全部上游开源。

对已经在 PyTorch 上的团队来说，试一试 TPU 的成本确实降到了一行 `device_map="tpu"` 的量级。迁移是否真的发生，最终取决于真实负载上每个 token 的成本。

参考资料

[1] 《TorchTPU is now open source: Run native PyTorch on TPUs》: *https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus*

[2] google-pytorch/torch_tpu: *https://github.com/google-pytorch/torch_tpu*

[3] torchtpu.readthedocs.io: *https://torchtpu.readthedocs.io/*

[4] GitHub: *https://github.com/google-pytorch/torch_tpu*

[5] Google Cloud Blog: *https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus*

[6] TorchTPU: *https://github.com/google-pytorch/torch_tpu*

[7] 开源仓库 README: *https://github.com/google-pytorch/torch_tpu/blob/main/README.md*

[8] PyTorch 的发布兼容矩阵: *https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix*

[9] PyPI 上的 0.2.0: *https://pypi.org/project/torch-tpu/0.2.0/*

[10] torchtpu.readthedocs.io: *https://torchtpu.readthedocs.io/*

[11] 公共 API 列表: *https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md*

[12] vLLM: *https://github.com/vllm-project/vllm-torchtpu*

[13] SGLang: *https://github.com/sgl-project/sglang-torchtpu/*

[14] TorchTitan: *https://github.com/google-pytorch/torchtitan*

[15] VeRL: *https://github.com/verl-project/verl/tree/main/examples/tpu/grpo*

[16] Ray: *https://github.com/ray-project/ray*

[17] LiteRT: *https://github.com/google-ai-edge/litert*

[18] LiteRT-Torch: *https://github.com/google-ai-edge/litert-torch*

[19] LiteRT-LM: *https://github.com/google-ai-edge/LiteRT-LM*

[20] ML Drift: *https://developers.googleblog.com/ml-drift-next-gen-gpu-aiml-inference-at-the-edge/*

[21] TPU agent suite: *https://github.com/google-pytorch/tpu-agent-suite*

[22] Google Cloud Blog: *https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus*

[23] Google Cloud 文档: *https://cloud.google.com/tpu/docs/pytorch-on-tpu*

::: related-title
**相关推荐**
:::

::: related-links
- **[vLLM TPU：一款支持在 TPU 上运行 PyTorch 与 JAX 的全新统一后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893481&idx=1&sn=58eca26a08502cb6820bdb3c111092bc&scene=21#wechat_redirect)**
- **[从 DeepMind研究到产品落地，Google Cloud工程副总裁深度访谈：十万 TPU 集群、Gemini研产全链路、从预训练到端侧部署、下一代AI基础设施](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904953&idx=1&sn=f37d34a9a0dcd0e989dac8f0e81b527b&scene=21#wechat_redirect)**
- **[前Google TPU架构师访谈：从批大小经济学到MoE物理布局，从流水线并行陷阱到RL对训练策略颠覆，从API定价技术到神经网络与密码学趋同进化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901219&idx=1&sn=95ebfde4d930adbfd424999bc41188e6&scene=21#wechat_redirect)**
- **[不写 Pallas 也能榨干 TPU：Helion 自动调优让 Flash Attention 在 TPU v7 上跑出 838 TFLOPs](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904098&idx=1&sn=72a96a46d50946d949a2fac7c602e6f2&scene=21#wechat_redirect)**
- **[性能较 TPUv5e 提升 4.83 倍！单脉动阵列内实现 FlashAttention 全流程且开源！面积开销仅 10%!](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890052&idx=1&sn=36b6413d652d086eb7fa34e59ed83999&scene=21#wechat_redirect)**

:::
