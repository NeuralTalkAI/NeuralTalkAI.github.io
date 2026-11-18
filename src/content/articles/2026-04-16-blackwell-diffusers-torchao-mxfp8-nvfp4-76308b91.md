关键词：**Blackwell 架构**、MXFP8、**NVFP4**、:mdnice-bold-italic[扩散模型]、模型量化

> 当前图像与视频生成扩散模型性能持续突破，但高显存与算力需求成为规模化部署核心瓶颈，**量化技术是实现高效推理的关键手段。**

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/01.webp)

- **Faster Diffusion on Blackwell: MXFP8 and NVFP4 with Diffusers and TorchAO**
- https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
- https://github.com/sayakpaul/diffusers-blackwell-quants
- 2500 字，阅读 9 分钟，**播客 19 分钟**

相关推荐

- [TorchAO：PyTorch原生端到到框架，打通训练到移动端全链路！FP8/PTQ缩容4倍/延迟降21%！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891983&idx=1&sn=f03133d7a88c10a28d6e83c1ec26acb1&scene=21#wechat_redirect)
- [Triton内核生成正确率从43%至97%！Makora 用可验证奖励 RL 让 GPT-5 写出比 Torch Inductor 更快的 Triton 内核](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897767&idx=1&sn=7f4588d9aa4512f348abed3023d06ed8&scene=21#wechat_redirect)
- [MXFP4近无损LLM训练：MXFP4 GEMM结合RHT与SR，比FP8快2倍计算且6.7B GPT模型近无损](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891301&idx=1&sn=07fa64e075a9cf2fa5d6cced89b8b108&scene=21#wechat_redirect)

> 本文聚焦 NVIDIA Blackwell 架构（B200 GPU）原生支持的 MXFP8 与 NVFP4 微尺度浮点格式，:mdnice-bold-italic[区别于传统全局张量量化，该技术将张量划分为微小数据块并共享高精度缩放系数，在降低比特位宽的同时保留动态范围与生成精度。]

研究依托 Diffusers 与 TorchAO 工具链，在 Flux.1-Dev、LTX-2、QwenImage 三大主流生成模型上完成可复现实验，**实现 MXFP8 最高 1.26 倍、NVFP4 最高 1.68 倍的端到端推理加速，显存占用显著下降。**

::: block-1
团队通过选择性量化规避小层量化损耗、借助 CUDA Graphs 降低 CPU 开销、采用 LPIPS 感知指标评估生成质量，构建速度与视觉效果的最优平衡，完整开放实验代码与工程配置，为 Blackwell 架构下扩散模型的高效工业化部署提供标准化解决方案。
:::

## 本文目录

- 一、概述
- 二、MXFP8 与 NVFP4 技术背景
- 三、基于 Diffusers 与 TorchAO 的基础使用
  - 3.1 环境前提
  - 3.2 基础使用代码
  - 3.3 量化方案选择
- 四、基准测试结果
  - 4.1 Flux.1-Dev 性能与显存
  - 4.3 LTX-2 与 QwenImage 结果
- 五、技术优化要点
  - 5.1 选择性量化优化精度与性能
  - 全量量化与选择性量化对 Flux-1.dev 的影响
  - 5.2 CUDA Graphs 降低 CPU 开销
  - 5.3 LPIPS 感知精度评估
- 结论
- 相关资源

## 一、概述

> 用于图像与视频生成的扩散模型人气持续攀升，可生成超逼真视觉内容。但其应用往往**受限于极高的内存与计算需求，量化技术是实现这类模型高效部署的核心方案。**

本文展示了可复现的端到端推理加速效果：在 NVIDIA B200 显卡上，基于 diffusers[1] 与 torchao[2]，MXFP8 可实现最高**1.26 倍** 加速，NVFP4 最高**1.68 倍** 加速，覆盖 Flux.1-Dev[3]、QwenImage[4]、LTX-2[5] 三大模型。同时阐述了选择性量化、CUDA Graphs 的使用方式，以及以 LPIPS 指标迭代优化模型精度与性能的方法。本文实验复现代码见：https://github.com/sayakpaul/diffusers-blackwell-quants[6]。

## 二、MXFP8 与 NVFP4 技术背景

> MXFP8 与 NVFP4 是 NVIDIA Blackwell 架构（如 B200 GPU）原生支持的**微尺度量化格式**。与传统量化对整个张量统一缩放不同，**微尺度量化将元素划分为小块（如 16 或 32 个数值），每个小块共享一个高精度缩放因子**。

该方式可大幅降低比特位宽，同时保留数据动态范围与模型精度。

- MXFP8（OCP 微尺度 FP8）：开放计算项目（OCP）推出的 8 位行业标准格式（E4M3/E5M2），分块大小为 32，采用 8 位缩放。在速度与画质间取得最优平衡，相比 BF16 推理更快且几乎无视觉质量损失（LPIPS 值更低），小批量场景下延迟最低。
- NVFP4（NVIDIA FP4）：Blackwell 张量核心专属加速的 4 位浮点格式（E2M1），分块大小为 16，采用 FP8 缩放因子。理论吞吐量最高、内存占用最小（约为 BF16 的 3.5 分之一），适合大批量、计算密集型任务。

更多细节可参考这篇 NVIDIA 技术博客《3 Ways NVFP4 Accelerates AI Training and Inference》[7]。

## 三、基于 Diffusers 与 TorchAO 的基础使用

### 3.1 环境前提

NVFP4 要求显卡 CUDA 算力至少为 10.0，需确保硬件符合要求。本文基准测试均在 B200 机型（B200 DGX）上完成。

虚拟环境可使用 conda 配置：

```
conda create -n nvfp4 python=3.11 -yconda activate nvfp4pip install --pre torch --index-urlhttps://download.pytorch.org/whl/nightly/cu130pip install --pre torchao --index-urlhttps://download.pytorch.org/whl/nightly/cu130pip install --pre mslk --index-urlhttps://download.pytorch.org/whl/nightly/cu130pip install diffusers transformers accelerate sentencepiece protobuf av imageio-ffmpeg
```

撰写本文时，PyTorch、TorchAO、MSLK 的夜间版版本分别为`2.12.0.dev20260315+cu130`、`0.17.0.dev20260316+cu130`、`2026.3.15+cu130`。

部分模型需要在 Hugging Face Hub 完成认证，未登录的用户需先执行`hf auth login`。

### 3.2 基础使用代码

> TorchAO 的 NVFP4 量化配置已原生集成至 Diffusers，使用方式简洁：

```
from diffusers import DiffusionPipeline, TorchAoConfig, PipelineQuantizationConfigimport torchfrom torchao.prototype.mx_formats.inference_workflow import (    NVFP4DynamicActivationNVFP4WeightConfig,)config = NVFP4DynamicActivationNVFP4WeightConfig(    use_dynamic_per_tensor_scale=True, use_triton_kernel=True,)pipe_quant_config = PipelineQuantizationConfig(    quant_mapping={"transformer": TorchAoConfig(config)})pipe = DiffusionPipeline.from_pretrained(    "black-forest-labs/FLUX.1-dev",    torch_dtype=torch.bfloat16,    quantization_config=pipe_quant_config).to("cuda")pipe.transformer.compile_repeated_blocks(fullgraph=True)pipe_call_kwargs = {    "prompt": "A cat holding a sign that says hello world",    "height": 1024,    "width": 1024,    "guidance_scale": 3.5,    "num_inference_steps": 28,    "max_sequence_length": 512,    "num_images_per_prompt": 1,    "generator": torch.manual_seed(0),}result = pipe(**pipe_call_kwargs)image = result.images[0]image.save("my_image.png")
```

上述代码会对模型中所有`torch.nn.Linear`层进行量化。

本文全程采用`fullgraph=True`局部编译，可大幅缩短编译时间，效果与全模型编译几乎一致。局部编译详情见这篇 PyTorch 技术博客《torch.compile and Diffusers: A Hands-On Guide to Peak Performance》[8]。

### 3.3 量化方案选择

以下代码展示 TorchAO 配置 MXFP8 与 NVFP4 推理的方式：

```
# MXFP8quant_config = MXDynamicActivationMXWeightConfig(    activation_dtype=torch.float8_e4m3fn,    weight_dtype=torch.float8_e4m3fn,    kernel_preference=KernelPreference.AUTO,)# NVFP4quant_config = NVFP4DynamicActivationNVFP4WeightConfig(    use_dynamic_per_tensor_scale=True,    use_triton_kernel=True,)
```

## 四、基准测试结果

### 4.1 Flux.1-Dev 性能与显存

测试参数：

```
{    "prompt": "A cat holding a sign that says hello world",    "height": 1024,    "width": 1024,    "guidance_scale": 3.5,    "num_inference_steps": 28,    "max_sequence_length": 512,}
```

首先，我们展示不同配置与不同测试基准下的延迟及峰值内存占用情况：使用 MXFP8 时加速比最高可达 **1.26 倍**，使用 NVFP4 时加速比最高可达 **1.59 倍**。 需要注意的是，这些结果均采用**选择性量化**，即不对部分特定层进行量化处理。我们将在本文后续内容中详细讨论选择性量化。

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/02.webp)

::: image-caption
NVIDIA B200、选择性量化、基于区域编译的 torch.compile；批量大小 batch_size=1 时采用 torch.compile(..., mode='reduce-overhead')。量化模式“None”表示不进行量化。
:::

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/03.webp)

为进行更全面的精度评估，我们计算了 bfloat16 图像（基准）与 MXFP8|NVFP4 图像（实验）之间的平均 LPIPS 得分，并在 Drawbench 数据集的所有提示词上取平均值：

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/04.webp)

NVIDIA B200、选择性量化、基于区域编译的 torch.compile。

LPIPS 得分为 0 表示“图像完全一致”，且 LPIPS 得分越低，代表感知相似度越高。我们用于计算平均 LPIPS 得分的代码见此处。有关使用 LPIPS 进行精度评估的更多细节，请参阅本文后续的 LPIPS 部分。

### 4.3 LTX-2 与 QwenImage 结果

> LTX-2 启用 VAE 分块以控制内存，视频生成场景下 NVFP4 最高加速**1.68 倍**，为全测试模型最优；QwenImage 对量化更敏感，MXFP8 精度低于 Flux.1-Dev。

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/05.webp)

::: image-caption
英伟达 B200、选择性量化、基于区域编译的 torch.compile。量化模式“None”表示不进行量化。
:::

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/06.webp)

::: image-caption
英伟达 B200、选择性量化、支持区域编译的 torch.compile 编译优化，批次大小 batch_size=1 时采用 torch.compile(..., mode='reduce-overhead')。量化模式“None”表示不进行量化。
:::

针对测试提示生成的MXFP8与NVFP4图像均接近bfloat16基准水平，其中NVFP4相较于MXFP8呈现出略大的差异：

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/07.webp)

在下述表格中，我们给出了与 Flux.1-Dev 类似的 LPIPS 评分。

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/08.webp)

注：在实验中我们发现，QwenImage 对量化的敏感度高于 Flux.1-Dev；QwenImage 在 MXFP8 量化下的平均 LPIPS 得分为 0.34（而 Flux.1-Dev 在 MXFP8 下仅为 0.11），**这一结果印证了该结论。通过更激进的选择性量化或更先进的数值算法（如 GPTQ、QAT 等）进一步降低 QwenImage 的平均 LPIPS 得分，将作为未来研究方向。**

## 五、技术优化要点

### 5.1 选择性量化优化精度与性能

> 本文采用两条核心规则跳过部分层量化，平衡速度与精度：

1.若`torch.nn.Linear`层维度过小（`min(M, K, N) < 1024`），跳过量化——避免矩阵乘法加速收益低于量化开销，详情见这篇 PyTorch 文档《Microbenchmarks and roofline model》[9]。此外，这篇文章《Debugging Weights and Activations with quant_logger》[10]，介绍如何使用 torchao 工具查看模型中的权重形状与激活值形状。请注意：即便权重规模较大，若激活值形状过小，量化处理也可能无法带来收益；

2.对嵌入层、归一化层等精度敏感层，直接跳过量化。若要在你的模型中应用此方法，你可以打印模型（`print(model)`）并手动查看其完全限定名称（FQN），然后根据你对模型架构的了解，跳过那些你认为可能会影响精度的完全限定名称。

> 我们针对各模型所采用的具体启发式策略如下：

- Flux-1.dev[11]
- QwenImage[12]
- LTX-2[13]

为量化选择性量化的效果，我们对比了纯 Bfloat16 图像与采用 NVFP4、MXFP8 生成图像之间的性能、显存占用以及平均 LPIPS[14]（基于 AlexNet）。

### 全量量化与选择性量化对 Flux-1.dev 的影响

| 量化模式 | LPIPS | 延迟（秒） | 显存（GB） |
| --- | --- | --- | --- |
| MXFP8 + 全量量化 | 0.138128 | 1.774 | 26.84 |
| MXFP8 + 选择性量化 | 0.107562 | 1.746 | 26.90 |
| NVFP4 + 全量量化 | 0.479679 | 2.112 | 21.25 |
| NVFP4 + 选择性量化 | 0.438337 | 2.076 | 21.33 |

注：LPIPS 值越低越好，约 0.1 时通常意味着图像几乎无法区分。LPIPS 计算代码可在diffusers-blackwell-quants/compute_lpips.py[15]获取。

从上述结果可以看出，**对部分层不进行量化（即“选择性量化”），能在延迟、峰值显存占用与 LPIPS 之间取得最优平衡**。因此，本文中另外两个模型均采用选择性量化方案。

我们通过简单的启发式方法确定选择性量化策略。目前存在更先进的选择性量化方法，例如层级敏感度分析[16]。

::: block-1
需要说明的是，在迭代优化选择性量化方案时，我们发现 TorchAO 中将张量量化为 NVFP4 的算子存在性能瓶颈。我们在此 PR：hook up mslk's to_nvfp4 kernel to torchao's inference nvfp4 workflows[17] 中通过将 `to_nvfp4` 算子升级为 MSLK[18]，提升了 NVFP4 的性能。
:::

### 5.2 CUDA Graphs 降低 CPU 开销

> 我们发现，在批量大小（batch size）为 1 等较小规模下使用NVFP4时，CPU开销会对延迟优化产生**不可忽视** 的影响。

为大幅降低此类开销，我们采用了“reduce-overhead”编译模式，该模式可启用CUDA Graph。以下为启用CUDA Graph前后的性能分析轨迹对比。

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/09.webp)

为了让`torch.compile(..., mode='reduce-overhead')`能与diffusers库中的逐块编译方式完美兼容，我们需要将每个Transformer模块封装在一个会克隆输入张量的函数中。相关PR详见：enable torch.compile with cuda graphs[19]，该优化在批量大小为1时，使QwenImage + nvfp4实现了**1.81倍** 的速度提升。

### 5.3 LPIPS 感知精度评估

我们使用 LPIPS[20] 指标来对比量化模型生成的图像与基准（bfloat16）模型生成图像的相似度。伪代码如下：

```
# 实际代码位于：https://github.com/sayakpaul/diffusers-blackwell-quants/blob/main/compute_lpips.pylpips_scores = []for text_prompt in dataset:    generator = torch.Generator(device=device).manual_seed(seed)    kwargs = {"prompt": prompt, "generator": generator, ...}    image_baseline = pipe_bf16(**kwargs)    image_quantized = pipe_quantized(**kwargs)    lpips_score = calculate_lpips_score(image_baseline, image_quantized)    lpips_scores.append(lpips_score)lpips_mean = lpips_scores.sum() / len(lpips_scores)
```

> 下面以 LPIPS 作为核心画质指标，对比基准 BF16 与量化模型的生成结果，实现客观可量化的精度评估，替代主观肉眼判断。

下方图像均由FLUX.1-dev生成。

- 左侧图像为基准结果（bfloat16精度）
- 右侧图像是将模型中所有torch.nn.Linear层采用MXFP8量化后生成的结果

LPIPS 得分基于右侧图像（实验结果）与左侧图像（基准结果）的对比计算得出。

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/10.webp)

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/11.webp)

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/12.webp)

下文我们将提供一组类似对比，仅将右侧图像替换为NVFP4格式图像。

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/13.webp)

![文章插图](/articles/2026-04-16-blackwell-diffusers-torchao-mxfp8-nvfp4-76308b91/images/14.webp)

## 结论

本文系统验证了 NVFP4 与 MXFP4 量化在主流图像、视频扩散模型上的性能，给出速度、显存、画质平衡的工程方案，解决大模型部署核心痛点，相关方案可直接用于 Blackwell 架构下生成模型的高效部署。

## 相关资源

- 代码仓库[21]
- TorchAO 文档：MXFP8[22]、NVFP4[23]
- Diffusers 与 TorchAO 集成方案：链接[24]

参考资料

[1] huggingface/diffusers: *https://github.com/huggingface/diffusers*

[2] pytorch/ao: *https://github.com/pytorch/ao*

[3] black-forest-labs/FLUX.1-dev: *https://huggingface.co/black-forest-labs/FLUX.1-dev*

[4] Qwen/Qwen-Image: *https://huggingface.co/Qwen/Qwen-Image*

[5] Lightricks/LTX-2: *https://huggingface.co/Lightricks/LTX-2*

[6] diffusers-blackwell-quants: *https://github.com/sayakpaul/diffusers-blackwell-quants*

[7] 3 Ways NVFP4 Accelerates AI Training and Inference: *https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/*

[8] torch.compile and Diffusers: A Hands-On Guide to Peak Performance: *https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/*

[9] microbenchmarks-and-roofline-model: *https://docs.pytorch.org/ao/main/workflows/inference.html#microbenchmarks-and-roofline-model*

[10] Debugging Weights and Activations with quant_logger: *https://docs.pytorch.org/ao/main/eager_tutorials/debugging_weights_and_activations.html*

[11] https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/benchmark.py#L190-L201: *https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/benchmark.py#L190-L201*

[12] https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L137: *https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L137*

[13] https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L160: *https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L160*

[14] PerceptualSimilarity: *https://github.com/richzhang/PerceptualSimilarity*

[15] diffusers-blackwell-quants/compute_lpips.py: *https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/compute_lpips.py*

[16] Layer Sensitivity Estimation : *https://huggingface.co/blog/badaoui/sensitivity-aware-mixed-precision-quantizer-v1#layer-sensitivity-estimation*

[17] hook up mslk's to_nvfp4 kernel to torchao's inference nvfp4 workflows: *https://github.com/pytorch/ao/pull/4031*

[18] meta-pytorch/MSLK: *https://github.com/meta-pytorch/MSLK*

[19] diffusers-blackwell-quants/pull/1: *https://github.com/sayakpaul/diffusers-blackwell-quants/pull/1*

[20] PerceptualSimilarity: *https://github.com/richzhang/PerceptualSimilarity*

[21] sayakpaul/diffusers-blackwell-quants: *https://github.com/sayakpaul/diffusers-blackwell-quants*

[22] mx_formats.MXDynamicActivationMXWeightConfig: *https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig.html*

[23] mx_formats.NVFP4DynamicActivationNVFP4WeightConfig: *https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig.html*

[24] Diffusers x TorchAO integration: *https://huggingface.co/docs/diffusers/main/en/quantization/torchao*
