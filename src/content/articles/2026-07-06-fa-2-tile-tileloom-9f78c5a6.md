关键词：**空间数据流加速器**、:mdnice-bold-italic[MLIR 编译框架]、 **tile 编程语言**、自动数据流规划、Tenstorrent

![文章插图](/articles/2026-07-06-fa-2-tile-tileloom-9f78c5a6/images/01.webp)

- **TileLoom: Automatic Dataflow Planning for Tile-Based Languages on Spatial Dataflow Accelerators**
- 论文：https://arxiv.org/pdf/2512.22168v2
- 代码：https://github.com/ecolab-nus/loom-dataflow

::: related-title
**相关推荐**
:::

::: related-links
- **[GEMM/FA 平均性能超厂商库 1.03×/1.91×！面向空间数据流架构的端到端 Tile-Based 编译器 TL](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896903&idx=1&sn=78559ef90da4c8638475faac4265f65a&scene=21#wechat_redirect)**
- **[碾压GH200注意力实现：FlatAttention在Tile数据流加速器上，计算场景利用率达86%，访存场景带宽利用率达 78%，注意力内核平均加速 1.9倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899603&idx=1&sn=59da0ff7b8208a308553b463fc36a817&scene=21#wechat_redirect)**
- **[解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)**

:::

> 论文中提到，**尽管空间数据流加速器在性能、能效与成本效率上具备巨大潜力，但有限的可编程性仍是其大规模普及的核心障碍**。这一判断精准点出了:mdnice-bold-italic[当前 AI 专用芯片领域的普遍困境：硬件架构已经走在了软件的前面，大量算力因为缺乏高效的编译工具而无法被普通开发者利用。]

TileLoom 这项工作的核心价值，就是为空间数据流加速器提供了一套通用的自动化编译方案。它基于 MLIR 构建了端到端的编译栈，能够将 Triton 这类面向 tile 的高级编程语言，自动映射到空间数据流架构上，自动完成时空映射、数据复用规划与通信调度，不再需要专家手工为每个算子编写硬件专用的优化代码。

![文章插图](/articles/2026-07-06-fa-2-tile-tileloom-9f78c5a6/images/02.webp)

::: image-caption
图 2：TileLoom 整体框架总览 。该图完整展示三层端到端编译流水线，串联 tile 级 DSL 输入到硬件可执行程序全链路，分为前端、数据流规划、后端代码生成三大核心模块。前端负责 Triton/Helion 内核分块形状自动调优，输出硬件无关的标准化 MLIR；核心数据流规划模块依托自定义 df 硬件描述方言，完成时空分块映射、跨核数据复用规划，再通过性能模型筛选最优候选映射；后端对接 Tenstorrent 底层 TT-Metalium API 生成硬件指令。框架创新引入 top-k 静态筛选+硬件二次调优的双层候选选择机制，弥补硬件建模存在的精度误差。三层解耦架构实现强硬件无关性，一套编译流程可适配二维网格、一维环形等多种空间数据流加速器，解决厂商专属库绑定硬件、难以快速拓展新型算子的行业痛点。
:::

在两代 Tenstorrent 硬件上的实验结果显示，TileLoom 编译生成的代码，**在 FlashAttention 算子上实现了接近 2 倍的性能提升，在 GEMM 这类已经被厂商深度优化的算子上也能达到持平甚至反超的水平**，对于厂商没有提供融合实现的算子，更是能带来十几倍到几十倍的性能飞跃。这一成果不仅证明了自动化数据流规划的可行性，也为空间数据流架构的软件生态建设提供了新的思路。

![文章插图](/articles/2026-07-06-fa-2-tile-tileloom-9f78c5a6/images/03.webp)

::: image-caption
表 2：TileLoom 相对 TTNN 在矩阵乘法、FlashAttention、Flash 解码、Mamba 分块扫描线性注意力算子上的性能倍数 。FlashAttention 具备充足跨核空间复用，两款硬件均接近 2 倍加速；Flash Decode 并行维度稀缺，仅达到厂商库 84%-87%性能；标准 GEMM 在 Wormhole 算力受限场景略弱，Blackhole 带宽瓶颈场景反超至 1.10 倍；Mamba 因厂商无原生融合算子，实现 16~27 倍巨大加速。该表清晰区分四类典型场景：复用充足的注意力算子、并行受限的解码算子、成熟高度优化的矩阵算子、无原生融合的新型线性算子，完整论证 TileLoom 对新型算子、带宽受限新一代硬件的独特价值，同时客观承认成熟算子场景下与手工调优库存在小幅性能差距。
:::

接下来我们将从背景痛点、架构设计、核心技术、实验验证等多个维度，对这项工作进行全面深入的解读。

## 本文目录

- 一、空间数据流架构的算力红利与可编程性鸿沟
- 二、TileLoom 整体设计：把硬件调度权完全交给编译器
- 三、时空映射：将逻辑 tile 网格铺展到物理核心阵列
- 四、数据复用规划：榨干片上网络与分布式缓存的每一分价值
- 五、统一硬件抽象：用一套 IR 描述所有空间数据流架构
- 六、性能建模：编译期精准预测映射的实际执行耗时
- 七、实验验证：两代 Tenstorrent 硬件上的真实表现
- 八、定位与对比：TileLoom 在编译技术版图中的位置
- 九、总结与展望

![文章插图](/articles/2026-07-06-fa-2-tile-tileloom-9f78c5a6/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[GEMM/FA 平均性能超厂商库 1.03×/1.91×！面向空间数据流架构的端到端 Tile-Based 编译器 TL](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896903&idx=1&sn=78559ef90da4c8638475faac4265f65a&scene=21#wechat_redirect)**
- **[碾压GH200注意力实现：FlatAttention在Tile数据流加速器上，计算场景利用率达86%，访存场景带宽利用率达 78%，注意力内核平均加速 1.9倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899603&idx=1&sn=59da0ff7b8208a308553b463fc36a817&scene=21#wechat_redirect)**
- **[解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)**

:::
