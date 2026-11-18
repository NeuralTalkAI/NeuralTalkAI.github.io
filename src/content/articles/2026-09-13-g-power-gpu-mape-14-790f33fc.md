关键词：**GPU 功耗建模**、:mdnice-bold-italic[架构级仿真]、**知识迁移**、注意力聚合、特征分组

> GPU 已经成为大规模并行计算的核心算力底座，从机器学习训练到科学计算仿真，每一次算力跃升背后都伴随着功耗墙的逼近。:mdnice-bold-italic[一张消费级的 RTX 3090 峰值功耗就能达到 350W，数据中心级 GPU 的散热和供电成本更是占据 TCO 的大头]。**功耗效率已经成为现代 GPU 设计中与性能同等重要的核心指标**。

然而，GPU 功耗优化的前提是快速、准确的功耗评估。:mdnice-bold-italic[传统的 VLSI 功耗仿真流程需要走完 RTL 仿真、逻辑综合、物理设计、功耗分析四个阶段，单次迭代可能耗时数天甚至数周，完全无法支撑架构探索阶段的快速反馈。架构级功耗模型因此应运而生——它通过统计程序运行时的架构事件，用解析公式或机器学习模型快速估算功耗，将评估时间从天数级压缩到秒级。]

::: block-1
但问题在于，架构级模型的训练需要高质量的功耗标签。对于目标 GPU 而言，真实负载的功耗标签收集极其耗时，研究者只能退而求其次，用仅含数百条指令的微基准测试来标定模型。**这些短小、简单的微基准与包含数百万条指令的真实负载之间存在显著的数据分布差异**，直接导致现有模型在真实场景下精度崩塌。:mdnice-highlight[论文实测显示，被广泛使用的 AccelWattch 在 Ampere 和 Ada 架构 GPU 上的平均绝对百分比误差（MAPE）超过 30%，相关系数 R 不足 0.6——这样的精度根本无法指导架构级的功耗优化决策。]
:::

![文章插图](/articles/2026-09-13-g-power-gpu-mape-14-790f33fc/images/01.webp)

- 论文标题：**G-Power: Architecture-level GPU Power Modeling with Aggregated Knowledge Foundations from Known GPUs**
- 组织机构：香港科技大学、上海交通大学、UniVista
- 论文链接：https://arxiv.org/abs/2608.06870

::: related-title
**相关推荐**
:::

::: related-links
- **[把 LLM 扩缩容拆到算子级！莱斯大学与微软开源 OpScale 方案，最高省 36.3% GPU、28% 功耗](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905555&idx=1&sn=991cb393953176f0ac8b56361129e314&scene=21#wechat_redirect)**
- **[从GEMM、FA再到分布式vLLM：TileSight 用三层 tile 抽象击穿 GPU 性能建模黑盒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904280&idx=1&sn=91a08cdc1338df361894a581f5ee9fb8&scene=21#wechat_redirect)**
- **[不用ML，跨代际，高保真的GPU能耗建模方案Wattchmen：用微基准测试集让每一行 SASS 指令能耗显形](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900251&idx=1&sn=cbf8080416682bc93398bcd28c54607e&scene=21#wechat_redirect)**

:::

> G-Power 的核心洞察直指这一痛点：**既然目标芯片缺乏真实负载的训练数据，那就从已经量产、可以方便测量的其他已知 GPU 芯片上"借"数据**。:mdnice-bold-italic[这些已知芯片同时拥有微基准和真实负载的完整功耗数据，可以训练出涵盖丰富程序行为特征的基础模型]。再通过注意力机制启发的聚合策略，根据目标芯片与各已知芯片的架构相似度，**有选择地融合这些基础模型的知识，最后在目标芯片的少量微基准数据上做参数高效的微调。**

![文章插图](/articles/2026-09-13-g-power-gpu-mape-14-790f33fc/images/02.webp)

::: image-caption
图 2：G-Power三阶段框架全景图。阶段1（Pre-Training with Additional Known Chips）为每个已知GPU芯片分别训练基础模型，数据集包含微基准和真实负载；阶段2（Attention-Inspired Aggregation）从已知芯片的微基准数据生成Key向量，从目标芯片的微基准数据生成Query向量，计算相似度后剪枝并加权聚合基础模型；阶段3（Fine-Tuning on Our Target GPU）通过两级特征分组将架构事件分组，冻结组内参数比例，仅训练每组的缩放旋钮α。这张图是理解G-Power整体数据流的核心：数据从已知芯片的全量数据集流入，经过基础模型训练和相似度聚合，最终在目标芯片的微基准数据上完成适配。
:::

实验在四款现代 NVIDIA GPU 上验证了这一思路：G-Power 平均 MAPE 仅 14%，相关系数 R 达到 0.88，分别**比 AccelWattch 降低 22%和提升 0.36。**

![文章插图](/articles/2026-09-13-g-power-gpu-mape-14-790f33fc/images/03.webp)

::: image-caption
图 5：G-Power与两个基线在四种测试场景下的精度对比，左图为MAPE（越低越好），右图为相关系数R（越高越好）。在所有四颗GPU上，G-Power的MAPE均显著低于AccelWattch和McPAT-Calib，相关系数R均显著更高。尤其在RTX 5880 Ada上，AccelWattch的MAPE高达70.4%，而G-Power仅约13%；在RTX 4090上，AccelWattch MAPE为35.8%，G-Power约28%。平均来看，G-Power的MAPE为14%，R为0.88，分别比AccelWattch降低22%和提升0.36。
:::

## 本文目录

- 一、问题根源：微基准训练数据与真实负载的分布鸿沟
- 二、G-Power 总体架构：三阶段知识迁移框架
- 三、核心技术细节深度拆解
- 四、实验结果与精度分析
- 五、领域影响与未来方向

![文章插图](/articles/2026-09-13-g-power-gpu-mape-14-790f33fc/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**
- **[基于 Roofline 建模的端侧大语言模型硬件协同设计缩放律](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898362&idx=1&sn=40049cc0a977a5c9d200ae68f41a0448&scene=21#wechat_redirect)**
- **[LLM推理极限建模：Bandwidth, Compute, Sync and Capacity are All You Need](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890183&idx=1&sn=2fc6e4e2e0c705123df18ad6d9f0dcb4&scene=21#wechat_redirect)**

:::
