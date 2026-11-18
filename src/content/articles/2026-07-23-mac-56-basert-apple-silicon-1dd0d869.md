关键词：BaseRT、**Apple Silicon**、:mdnice-bold-italic[Metal]、LLM 推理、**统一内存**

> "框架抽象本身就是这个平台上一个有分量的开销来源"。

当整个行业都在讨论端侧大模型该跑多大的参数、用几比特量化时，来自墨尔本 Base Compute 的三位研究者把问题往下钻了一层：**同一台 Mac、同一个模型、同样的量化位宽，仅仅是换一套推理运行时，速度能差出多少？** 他们给出的答案相当惊人。

![文章插图](/articles/2026-07-23-mac-56-basert-apple-silicon-1dd0d869/images/01.webp)

- **BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal**
- https://arxiv.org/pdf/2607.00501
- https://github.com/basecompute/baseRT

::: related-title
**相关推荐**
:::

::: related-links
- **[RIGEL：逆向拆解 Apple M4 Max GPU 的 Metal 4.1 张量计算通路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902692&idx=1&sn=6c546a2fd3f9fc3960e81c9b98e151f8&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[苹果 M5 芯片发布，AI 性能狂飙 4 倍！GPU 架构重构 + 16 核神经引擎 + 10 核 CPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893129&idx=2&sn=3a2db8a1d0c83334ff424c4b7cfea9c5&scene=21#wechat_redirect)**

:::

> BaseRT 是一个直接:mdnice-bold-italic[构建在 Apple Metal GPU API 之上的 C++ 推理运行时，不依赖 MLX、PyTorch、CoreML 或任何中间数组框架，靠芯片级内核融合、面向统一内存的数据布局和自研的 GPU 调度逻辑]，把框架方案"留在桌面上的性能"捡了回来。

在 M4 Pro 上，它的解码吞吐最高达到 **llama.cpp 的 1.56 倍、MLX 的 1.35 倍**；在 MoE 模型的预填充阶段，领先幅度更是拉到 1.81 倍。

![文章插图](/articles/2026-07-23-mac-56-basert-apple-silicon-1dd0d869/images/02.webp)

::: image-caption
图 1：BaseRT 在 Apple Silicon 上交出最佳吞吐。左图为解码吞吐（tok/s），在 M4 Pro 上测试的 Q4 模型中，BaseRT 相对 llama.cpp 提升 1.15 至 1.56 倍，相对 MLX 最高提升 1.35 倍；右图为预填充吞吐随提示词长度（128 至 2048 token）的变化，BaseRT 在 4 bit 量化的 Qwen3-30B-A3B 上全程领先，相对 MLX 最高提升 1.78 倍。值得注意的是左右两图的赢家逻辑并不相同：解码阶段小模型的优势最大，因为此时固定调度开销占延迟的大头；预填充阶段则是 MoE 模型的优势最大，因为稠密 GEMM 会把各家运行时都顶到 GPU 算力上限，差距只能在调度效率上拉开。这张图基本浓缩了全文的核心结论。
:::

::: block-1
从 0.6B 到 30B 参数，BaseRT 在几乎所有测试配置上拿下了 Apple Silicon 平台有公开记录以来的最高吞吐，并且已在 GitHub[1] 上公开。**这篇论文用实测数据证明了一件事：Apple Silicon 的推理能力被现有运行时系统性地低估了**，而低估的来源恰恰是大家习以为常的"框架"本身。
:::

## 本文目录

- 一、为什么非要在你自己的 Mac 上跑大模型
- 二、框架的隐性税：现有运行时慢在哪
- 三、BaseRT 的解题思路：把运行时削成一层薄纸
- 四、Decode 实测：小模型最高提速 56%，优势跨越芯片代际
- 五、Prefill 与 uzu 对决：MoE 才是真正的分水岭
- 六、BaseRT 的边界在哪
- 结语：端侧推理的地基之争

![文章插图](/articles/2026-07-23-mac-56-basert-apple-silicon-1dd0d869/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[评估 YOLOv11/GEMM 等应用在 Apple Silicon M1 与 M4 Pro 神经引擎上的潜力：GEMM性能达3.8 TFLOPS，能效超GPU 80%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897336&idx=1&sn=faef0c904b6cd45e50a35faf4ecf2372&scene=21#wechat_redirect)**
- **[生产级 Apple Silicon LLM 推理框架深入分析对比：MLX, MLC-LLM, Ollama, llama.cpp](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894128&idx=1&sn=1af2f122be249b52eb4a764227b28c83&scene=21#wechat_redirect)**
- **[Apple Silicon LLM 推理全方位深度剖析：与 NVIDIA GPU 从性能到成本的终极对决！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890883&idx=1&sn=d05b1dbc8df195d7a60dfc6d2bee4d6f&scene=21#wechat_redirect)**

:::
