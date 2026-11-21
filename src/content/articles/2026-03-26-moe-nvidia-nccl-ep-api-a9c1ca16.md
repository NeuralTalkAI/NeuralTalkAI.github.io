关键词： MoE（Mixture-of-Experts）、NCCL、**GPU 通信**、**Device-Initiated Communication**、:mdnice-bold-italic[大模型推理]

> *在通往通用人工智能的道路上，模型的规模正以前所未有的速度扩张。当稠密的 Transformer 模型在计算和参数效率上触及天花板时，混合专家（Mixture-of-Experts, MoE）架构以其“**加人加卡不加计算** ”的特性，成为了各大 AI 公司竞相追逐的“圣杯”。:mdnice-bold-italic[从 DeepSeek-V3 到 Mixtral，MoE 通过将模型参数解耦为稀疏的“专家”网络，在保持推理成本可控的前提下，实现了模型容量的指数级增长。]*

然而，硬币的另一面是，MoE 架构带来了一个全新的、极其棘手的通信挑战。假设你有一万个“专家”（子网络）分布在数千张 GPU 上，每个输入 token 需要根据实时计算出的路由规则，动态地、精确地被“分发”到其选中的几个专家那里，处理完后再将结果“收集”回来。:mdnice-bold-italic[这个过程不再是传统的大规模同步通信（如 AllReduce），而是变成了一个细粒度、动态、全对全（All-to-All）的“调兵遣将”。]

::: block-1
面对这一挑战，业界涌现出了 DeepEP、Hybrid-EP 等一系列优秀的设备端发起（Device-Initiated）通信库。它们如同为 MoE 量身定做的“特快专递”，通过让 GPU 直接发起远程直接内存访问（RDMA），绕开 CPU，极大地降低了通信延迟。

![文章插图](/articles/2026-03-26-moe-nvidia-nccl-ep-api-a9c1ca16/images/01.webp)

::: image-caption
表 I：MoE 常用通信库：在统一 API、模式支持、编程接口及目标负载等维度的差异。分析表明，当前生态呈现碎片化：DeepEP 和 pplx-kernels 虽性能卓越，但针对低延迟（LL）和高吞吐（HT）模式提供分离的 API；Hybrid-EP 仅专注于训练场景。相比之下，NCCL EP 是唯一同时提供统一 C/Python API、双模式支持并完全构建于 NCCL 生态之上的解决方案。这种统一性极大简化了框架集成，避免了用户在不同库间切换的复杂性，标志着 MoE 通信向标准化、平台化迈出了关键一步。
:::

**但问题也随之而来：生态碎片化**。每个库都有自己独特的 API，有的为推理优化，有的为训练优化，有的依赖 NVSHMEM，有的依赖 IBGDA。**开发者和框架维护者不得不在多个“方言”之间疲于奔命**，这无疑增加了部署的复杂性和维护成本。
:::

![文章插图](/articles/2026-03-26-moe-nvidia-nccl-ep-api-a9c1ca16/images/02.webp)

- **NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL**
- https://arxiv.org/pdf/2603.13606
- 8000 字，阅读 25 分钟，**播客 19 分钟**

相关推荐

- **[突破 GPU 通信瓶颈：NCCL 协议创新与 25-95% 带宽利用率的量化研究](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890497&idx=1&sn=54b6b60452029a0a580b02184abfa308&scene=21#wechat_redirect)**
- **[最高 4.7 倍加速！UCSD、Meta 提出 AutoOverlap：以块级细粒度重叠融合计算与通信，突破多 GPU 通信瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897745&idx=1&sn=c925fdf089e3022438cce891ccbc440b&scene=21#wechat_redirect)**
- **[性能超传统库达 1.79×！AMD 提出 Triton 原生多 GPU 通信库 Iris：以 tile 级抽象实现 GEMM+All-Scatter 近峰值带宽](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895934&idx=1&sn=d02bb97055c0f7586e8a3d71e0b3b4cb&scene=21#wechat_redirect)**

> 正是在这样的背景下，NVIDIA 推出了 NCCL EP (Expert Parallelism)。它不是另起炉灶，而是选择了一条“大一统”的道路——将所有 MoE 通信能力，原生地、统一地集成进 NCCL（NVIDIA Collective Communications Library，NVIDIA 集体通信库）。

::: block-1
正如论文中所言：

“By building MoE communication natively within NCCL, NCCL EP provides a supported path for expert parallelism on current and emerging NVIDIA platforms.”

通过在 NCCL 内部原生构建 MoE 通信，NCCL EP 为当前和未来的 NVIDIA 平台上的专家并行提供了一条受支持的路径。
:::

:mdnice-bold-italic[NCCL EP 的核心理念是：提供一套统一的`ncclEpDispatch`和`ncclEpCombine` API，在底层根据工作负载（训练/推理）自动选择最优的通信算法（低延迟模式或高吞吐模式），并全面拥抱 NCCL 的设备端 API（Device API）]。这意味着，开发者再也不用在多个库之间切换，只需一套 API，就能在 NVIDIA 的硬件上获得极致的 MoE 通信性能。

![文章插图](/articles/2026-03-26-moe-nvidia-nccl-ep-api-a9c1ca16/images/03.webp)

::: image-caption
图 1：NCCL EP 架构：低延迟内核改编自 DeepEP [8]，高吞吐内核改编自 Hybrid-EP [10]，两者都使用 NCCL GIN（GPU 发起的网络）进行节点间 RDMA 通信，同时保留其原生 NVLink 实现用于节点内通信。该图揭示了 NCCL EP 的核心架构思想：集百家之长，统一于 NCCL 框架之下。它并未重新发明内核算法，而是将经过业界验证的 DeepEP（低延迟）和 Hybrid-EP（高吞吐）内核设计进行适配，将其底层的通信原语（如 IBGDA）替换为 NCCL Device API 中的 GIN 和 LSA。这样，NCCL EP 既继承了这些前沿内核在各自领域的性能优势，又统一了通信后端，从而能够自动利用 NCCL 的拓扑感知能力（如选择 NVLink 或 PCIe）和生态系统兼容性。这种“站在巨人肩膀上”的架构，确保了性能与可维护性的双重胜利。
:::
