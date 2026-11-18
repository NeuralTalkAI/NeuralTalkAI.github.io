关键词：YANchor-4B、**恒定内存**、KV Cache、:mdnice-bold-italic[记忆锚点]、**长上下文推理**

> 长序列推理的核心矛盾在于：全历史注意力虽能保留完整上下文信息，但显存与计算成本随序列长度持续膨胀；循环/状态空间模型虽具备线性时间、常数显存的效率优势，却易在迭代压缩中丢失精准的细节关联，难以支撑复杂长程推理。

![文章插图](/articles/2026-10-20-yanchor-4b-o-n-o-8fb9aedc/images/01.webp)

- **YANchor-4B: Effective Long-Horizon Reasoning in O(N) Time with O(1) Memory**
- https://arxiv.org/pdf/2610.10118
- 代码：https://github.com/RocoreMatrix/YANchor
- 模型权重：https://huggingface.co/rockmatrix-ai/EdgeRock

::: related-title
**相关推荐**
:::

::: related-links
- **[记忆是被重构的而非重放的！浙大提出 MemHarness，用 GRPO 炼出超越 GPT-4o 的 7B 智能体](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904205&idx=1&sn=d511f9fea5385a822ac1844841af6596&scene=21#wechat_redirect)**
- **[LLM负责思考，SLM负责记忆。LightMem小模型天团重塑 Agent 记忆，延迟仅 83 毫秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900501&idx=1&sn=2311904700b9bfd0c8da1f96e2e98a4b&scene=21#wechat_redirect)**
- **[让手机CPU,GPU,NPU协同！浙大端侧AI“超级记忆”AME硬件感知引擎：7倍索引加速，6倍插入提速，隐私与速度兼得！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895124&idx=1&sn=5c073c54ccc26b1797e79a12862ac496&scene=21#wechat_redirect)**

:::

> 针对这一行业痛点，Rocore Matrix （岩芯矩阵，岩山科技旗下）团队推出:mdnice-bold-italic[通用循环语言模型YANchor-4B，以“锚点记忆”为核心创新，在$O(N)$生成时间、$O(1)$历史显存的约束下实现了媲美全注意力架构的长程推理能力。]

:mdnice-bold-italic[该模型基于Qwen3.5-4B主干改造，将全局注意力替换为1024窗口局部注意力与独立记忆分支，与Gated DeltaNet循环层形成“全局压缩上下文+近期细节保留+锚点精准检索”三路互补的历史表示机制]。经过记忆适配、联合适配、监督后训练、结果导向强化学习四阶段训练，模型可自主完成关键信息的写入、筛选与跨距检索。

![文章插图](/articles/2026-10-20-yanchor-4b-o-n-o-8fb9aedc/images/02.webp)

::: image-caption
图 3：YANchor-4B 架构，循环层与局部注意力、记忆模块交错排布。(a) 主干是 3 个 Gated DeltaNet 循环层加 1 个"局部注意力+记忆"层的组合，重复 8 次，图像 token 和文本 token 进同一套主干；(b) 滑窗注意力管最近 1,024 个 token，独立记忆分支把滑出窗口的重要信息写成 key-value 记录，经打分和 Top-K 留存进记忆库，查询时按组检索回来。两路输出汇入同一条残差流，后续层同时看到近期细节和远处记忆。
:::

实验结果展现出量级式提升：**AIME 2024-2026平均pass@1达82.93%，HMMT 2026达63.64%，** 较13.3B参数的最强同架构基线提升超4倍；24项通用基准中领先20项，**平均分超出最强基线22.35分，同时支持多模态能力。**

![文章插图](/articles/2026-10-20-yanchor-4b-o-n-o-8fb9aedc/images/03.webp)

::: image-caption
图 2：单H100推理性能三维对比。该组图从单序列解码速度、批量生成吞吐量、峰值显存占用三个维度，在单张H100 80GB上对比YANchor-4B与主流Transformer基线的推理效率。左图：128K 输入 128K 输出，YANchor 解码稳定在 213 tokens/s，三个基线随生成推进降速。中图：32K 输出的端到端吞吐，YANchor 约 12.4K tokens/s，是 Qwen3.5-4B 的 4.12–6.58 倍。右图：512 条序列并发时 YANchor 峰值显存 67.71 GiB 且基本不随上下文增长，三个基线在 2K–8K 上下文先后触发缓存抢占，外推到 64K 需要 541–1,352 GiB。
:::

::: block-1
**单张H100上**，其批量长生成吞吐量达Transformer基线的4-6倍，512并发64K上下文峰值显存仅67.7GiB。:mdnice-highlight[长记忆测试准确率达94.14%，禁用记忆分支后骤降至0.39%，充分验证了锚点机制的核心作用。]
:::

## 本文目录

- 一、长上下文推理的两难：全注意力贵，循环模型忘
- 二、改造 Qwen3.5-4B：372M 参数加一个记忆分支
- 三、记忆锚点机制：每条序列固定 40 MiB
- 四、四段训练：R 阶段把记忆准确率从 79.4% 推到 96.1%
- 五、能力评测：AIME 82.93%，领先恒定状态阵营 22.35 分
- 六、H100 实测：512 条 64K 序列同驻，峰值 67.71 GiB
- 七、记忆消融：关掉记忆分支，准确率从 94.14% 掉到 0.39%

![文章插图](/articles/2026-10-20-yanchor-4b-o-n-o-8fb9aedc/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[LLM负责思考，SLM负责记忆。LightMem小模型天团重塑 Agent 记忆，延迟仅 83 毫秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900501&idx=1&sn=2311904700b9bfd0c8da1f96e2e98a4b&scene=21#wechat_redirect)**
- **[NVIDIA, UCB提出Kernel Blaster：记忆增强上下文 RL 让 CUDA 内核几何平均1.43×-2.50×加速！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897594&idx=1&sn=7b2a26eda203fb62c3c7f6aa75e8d046&scene=21#wechat_redirect)**
- **[让手机CPU,GPU,NPU协同！浙大端侧AI“超级记忆”AME硬件感知引擎：7倍索引加速，6倍插入提速，隐私与速度兼得！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895124&idx=1&sn=5c073c54ccc26b1797e79a12862ac496&scene=21#wechat_redirect)**

:::
