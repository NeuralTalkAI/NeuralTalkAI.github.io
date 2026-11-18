关键词：内核融合、**归一化优化**、Multi-CTA Norm、**Lazy Pre-Norm**、:mdnice-bold-italic[FlashNormAttention]

> 本文针对 LLM 与广告推荐模型中广泛使用的 **LayerNorm、RMSNorm 性能痛点提出全套内核融合优化方案**。归一化（Normalization）算子是典型内存受限操作，无法调用 TensorCore，在推荐模型训练中占 20%总时延、大模型占 10%，传统算子融合因 GEMM 与归一化分块逻辑冲突，仅小维度场景有效。

作者推出三大核心创新：

- Lazy Pre-Norm 借助数学恒等变换破解预归一化循环依赖，将归一化计算后置至矩阵乘收尾阶段；
- Multi-CTA Norm 依靠多 CTA 集群与分布式共享内存协同归约，实现通用后归一化融合，最高可掩盖 90%归一化时延；
- FlashNormAttention 一体化融合前后归一化与 GDPA 注意力，构建低 IO MegaKernel。

::: block-1
作者设计前向、反向差异化融合策略，规避反向计算冗余，基于 NVIDIA B200 GPU 实测，:mdnice-bold-italic[矩阵场景归一化延迟掩盖率最高 91.61%，注意力模块正向总时延最高降低 35.39%、反向近 20%，大幅削减 HBM 读写开销，整套优化代码开源，为生成式广告模型、大语言模型提供通用高性能内核优化路径。]
:::

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/01.webp)

- **Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels**
- https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#
- multi_cta_norm_fusion 代码地址：https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion
- gdpa_megakernel 代码地址：https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel

::: related-title
**相关推荐**
:::

::: related-links
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**
- **[Helion DSL 赋能 vLLM 实现跨硬件高效推理：Qwen3 模型 FP8 推理加速实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902433&idx=1&sn=5e65e917235f732c27c1cd121a2d6d24&scene=21#wechat_redirect)**
- **[借助 torch.compile 实现 Normalization 顶尖性能表现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899700&idx=1&sn=dad00ee6111ffb73275bae4c98c8107f&scene=21#wechat_redirect)**

:::

> 本研究主要基于两种核领域特定语言（DSL）实现：

::: related-links
- TLX[1]：Triton DSL 的扩展集，提供底层、硬件感知的 GPU 执行控制支持
- Helion[2]：高层 DSL，在开发效率、可移植性和全面自动调优方面表现出色

:::

基准测试采用 bfloat16 数据类型，在 Meta 数据中心的 NVIDIA B200 GPU 上进行，功率限制为 750W。

::: block-1
我们感谢 Tri Dao、Markus Hoehnerbach、Jay Shah、Ted Zadouri、Vijay Thakkar、Wentao Guo 在Flash Attention[3] 和Quack[4] 方面的开源工作——这些工作为本文讨论的许多优化技术奠定了基础并提供了灵感。我们感谢 PyTorch 和 Triton 团队对Helion[5] 和TLX[6] 的开发和维护——它们使本文提出的思路能够高效、顺利地探索和实现。
:::

## 本文目录

- 引言
- 一、归一化融合的挑战
- 二、延迟预归一化：一种将预归一化与线性层融合的新颖技术
- 三、多 CTA 归一化：将后归一化与线性层作为后缀融合
- 四、FlashNormAttention：将预归一化和后归一化同时融合到 FlashAttention 变体核中
- 结语

![文章插图](/articles/2026-07-13-flashnormattention-gemm-90-4add979f/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**
- **[Helion DSL 赋能 vLLM 实现跨硬件高效推理：Qwen3 模型 FP8 推理加速实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902433&idx=1&sn=5e65e917235f732c27c1cd121a2d6d24&scene=21#wechat_redirect)**
- **[借助 torch.compile 实现 Normalization 顶尖性能表现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899700&idx=1&sn=dad00ee6111ffb73275bae4c98c8107f&scene=21#wechat_redirect)**

:::
