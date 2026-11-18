关键词：**Helion**、:mdnice-bold-italic[TPU]、**Pallas**、软件流水线、**自动调优**

> "**Helion 是 PyTorch 用于编写性能可移植机器学习内核的高级 DSL**。我们与 Google 合作构建了 **TPU 后端，将 Helion 内核编译到 Pallas**。"PyTorch 官方博客在 2026 年 7 月发布的这篇文章提到：:mdnice-bold-italic[深度学习内核编写，正在从"一套硬件一套代码"走向"一次编写、多硬件运行"。]

这背后的产业背景十分清晰。Google 最新的 TPU v7（Ironwood）在 BF16 算力与 HBM 带宽这两项现代机器学习负载最关键的硬件指标上，已经**与 NVIDIA B200 高度可比，且总体拥有成本可能更低。**

::: block-1
:mdnice-highlight[这意味着 TPU 正在从 Google 的内部武器，变成大规模训练与推理的现实选项]。但硬件可用只是第一步，**软件生态才是真正的门槛**：想为 TPU 写高性能内核，过去必须精通 Pallas 这门低级 DSL，学习曲线陡峭、代码复杂度高。
:::

Helion 给出的答案是，让开发者写熟悉的 PyTorch 风格代码，由编译器生成优化后的 TPU 代码，再配合自动调优器搜索最优配置。

- 在 **Flash Attention** 负载上，Helion 生成的内核在 :mdnice-bold-italic[TPU v7 上达到 838 TFLOPs，约为单个张量核 MFU 的 79%；]
- 在 **16 个内核的更广基准** 中，:mdnice-bold-italic[相对 TorchTPU eager 模式取得 1.55 倍几何平均加速，相对 torch.compile（XLA）也有 1.12 倍。]

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/01.webp)

## 本文目录

- 一、TPU 内核为什么难写：一台"顺序机器"的编程鸿沟
- 二、Helion 的定位：用 PyTorch 语法写性能可移植内核
- 三、从 add 内核看懂 Helion 的 Pallas 代码生成
- 四、Flash Attention 实战：两种流水线策略的取舍
- 五、性能数字：838 TFLOPs、4.45 倍加速与 1.55 倍几何平均
- 行业意义与未来路线

![文章插图](/articles/2026-07-29-pallas-tpu-helion-flash-attention-75fade11/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[Helion DSL 赋能 vLLM 实现跨硬件高效推理：Qwen3 模型 FP8 推理加速实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902433&idx=1&sn=5e65e917235f732c27c1cd121a2d6d24&scene=21#wechat_redirect)**
- **[Google 第八代 TPU：面向智能体时代的训练与推理芯片，原生支持 JAX、MaxText、PyTorch、SGLang 与 vLLM](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900266&idx=1&sn=dd3219b311dfca2bba5566c5acb42fd5&scene=21#wechat_redirect)**
- **[用GPU原型验证，用TPU训练和推理！TorchTPU：谷歌 Meta 联手让 PyTorch 原生跑 TPU，3 行代码从 GPU 切换到 TPU 的极致易用性！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899873&idx=1&sn=f61bc039e138133b982b29e612d0a1fb&scene=21#wechat_redirect)**

:::
