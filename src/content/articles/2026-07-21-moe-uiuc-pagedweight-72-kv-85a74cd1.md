关键词：MoE 大模型、动态权重量化、KV 缓存管理、:mdnice-bold-italic[推理服务优化]、**分页式显存调度**

> 论文中有一句核心判断精准点出了当前 MoE 部署的核心矛盾：With MoE, the loaded model size can occupy more than 60% of the GPU memory。:mdnice-bold-italic[MoE 架构通过稀疏激活大幅降低了推理计算量，成为长上下文、复杂推理任务的主流选择，但总参数量的膨胀让模型权重成为 GPU 显存的核心占用者，与随上下文线性增长的 KV 缓存形成了直接的零和博弈。]

传统静态量化方案只能在推理前固定权重精度，无法适配动态变化的显存压力：短上下文时显存有余量却无法提升精度，长上下文时 KV 缓存挤占空间又无法进一步压缩权重，最终只能通过卸载权重到 CPU 引发性能暴跌。

![文章插图](/articles/2026-07-21-moe-uiuc-pagedweight-72-kv-85a74cd1/images/01.webp)

- **PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization**
- https://arxiv.org/pdf/2607.16184

::: related-title
**相关推荐**
:::

::: related-links
- **[重新思考 FP4 预训练！收缩偏置才是 E2M1 FP4 训练不稳定的根源，UFP4 均匀格式将 124B MoE 损失降解降低 20%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902714&idx=1&sn=f461543fe701a86413ad7da6ccd7156a&scene=21#wechat_redirect)**
- **[面向大规模专家并行 MoE 推理的故障容错方案 EEP，让 671B MoE 单节点故障停机从 348 秒压缩至 19 秒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901916&idx=1&sn=bb370fd7a527a3da6a08327163854740&scene=21#wechat_redirect)**
- **[前Google TPU架构师访谈：从批大小经济学到MoE物理布局，从流水线并行陷阱到RL对训练策略颠覆，从API定价技术到神经网络与密码学趋同进化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901219&idx=1&sn=95ebfde4d930adbfd424999bc41188e6&scene=21#wechat_redirect)**

:::

> PagedWeight 的核心洞察是**借鉴 PagedAttention 的分页思想，将 MoE 专家权重拆分为可独立调度的分页单元，运行时根据 KV 缓存压力动态调整权重精度**，精准选择对质量影响最小的权重降精度释放显存。

实验数据显示:

- PagedWeight 可以在保持 FP16 等效精度的前提下实现最高 72.0%的 GPU 显存节省，推理吞吐量提升 1.94 倍；
- 在相同显存预算下，模型质量比现有量化方法最高提升 39.3%，吞吐量损失最多仅 4.1%。

::: block-1
这项工作跳出了单点压缩的传统思路，从系统级显存调度的角度重新定义了 MoE 推理的资源分配模式，对长上下文 MoE 服务的成本优化有直接落地价值。
:::

## 本文目录

- 一、MoE 推理的显存困局：权重与 KV 缓存的零和博弈
- 二、PagedWeight 的核心洞察：分页式权重的动态表示
- 三、质量感知规划器：三层信号实现精准量化决策
- 四、异步调度与内核优化：隐藏动态调整的性能开销
- 五、多维度实验验证：全面领先的质量-显存-吞吐平衡
- 总结与未来展望

![文章插图](/articles/2026-07-21-moe-uiuc-pagedweight-72-kv-85a74cd1/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[GPU开销直降64%、智能体交互降 62%！易于 Agent 开发的精简 MoE 训练系统 PithTrain，吞吐对标 Megatron-LM！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902508&idx=1&sn=fa5498a13991cc26d253302cda567d66&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**

:::
