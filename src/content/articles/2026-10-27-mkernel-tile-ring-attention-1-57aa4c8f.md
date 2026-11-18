关键词：mKernel、融合内核、**计算通信重叠**、:mdnice-bold-italic[分布式训练]、**自适应 SM 划分**、RDMA 通信

> 当前大模型分布式系统最核心的痛点，是随着模型规模与集群规模的持续扩张，通信开销在端到端训练中的占比不断攀升，:mdnice-bold-italic[在生产级 MoE 训练中，通信时间可占前向传播的 43.6%、端到端训练时间的 32%，部分场景下接近一半的执行时间都消耗在设备间数据传输上。]

算力与网络的发展失衡正在加剧这一矛盾：新一代 GB300 NVL72 整机柜可提供 720 PFLOP/s 的 FP8 算力，但单张 GPU 跨机柜的出口带宽仅为 400-800 Gb/s，**算力增速远远超过网络带宽的增长速度。如果不能更高效地隐藏通信开销，分布式系统的算力利用率将持续走低，集群扩展的边际效益也会越来越差。**

::: block-1
内核级融合被认为是解决通信瓶颈的重要技术方向，通过在内核内部以 tile 粒度重叠计算与通信，可以大幅提升通信隐藏的粒度与效率。:mdnice-highlight[但现有融合内核方案大多局限于单个 NVLink 域内，跨节点通信始终需要独立的集合通信阶段完成，无法在内核内部与计算形成流水重叠，成为了融合性能进一步提升的核心瓶颈。]
:::

![文章插图](/articles/2026-10-27-mkernel-tile-ring-attention-1-57aa4c8f/images/01.webp)

- **mKernel: Fast Multi-GPU, Multi-Node Fused Kernels**
- 组织机构：加州大学伯克利分校、加州大学戴维斯分校、加州大学洛杉矶分校、布加勒斯特理工大学
- 论文链接：https://arxiv.org/pdf/2609.13585
- 开源代码：https://github.com/uccl-project/mKernel
- 博客：https://uccl-project.github.io/posts/mkernel/

::: related-title
**相关推荐**
:::

::: related-links
- **[GraphPP：基于计算图的流水线并行，零气泡+极致重叠，训练提速 70%+](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900542&idx=1&sn=ffd9887d1aebc574dccf2f93c054c6a1&scene=21#wechat_redirect)**
- **[最高 4.7 倍加速！UCSD、Meta 提出 AutoOverlap：以块级细粒度重叠融合计算与通信，突破多 GPU 通信瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897745&idx=1&sn=c925fdf089e3022438cce891ccbc440b&scene=21#wechat_redirect)**
- **[MLIR-AIR：AMD 基于空间分区与显式同步原语的开源编译器栈，融合通信-计算重叠，实现矩阵乘法 78.7%计算效率](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893191&idx=1&sn=fba9af0b8819b0766c12b58bb2adfef9&scene=21#wechat_redirect)**
- **[W4A8 GEMM实现LLM 4.9倍加速！LiquidGEMM让反量化、权重加载和矩阵乘流水重叠，较TRT-LLM提升1.6倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891799&idx=1&sn=bb787e1c28c27320740a22759cad7812&scene=21#wechat_redirect)**

:::

> mKernel 的出现正是为了突破这一瓶颈。作为一套多 GPU、多节点融合内核库，mKernel 在单个持久内核内部实现了计算、节点内 NVLink 通信、跨节点 RDMA 通信的三层 tile 级重叠，通过 SM 专业化分工、分层数据移动、跨平台主机辅助通信、运行时自适应 SM 划分四大核心设计，将融合内核的边界从单节点拓展到了多节点。

![文章插图](/articles/2026-10-27-mkernel-tile-ring-attention-1-57aa4c8f/images/02.webp)

::: image-caption
图 2：mKernel 架构图（双节点；每个节点仅详细展示一张 GPU）。每张 GPU 运行一个持久内核，其线程块被分配不同的功能角色。①计算线程块将完成的 tile 写入 GPU 内存并设置就绪标志。②节点内通信线程块通过 NVLink 与其余七张 GPU 交换 tile，利用 NVSwitch 实现广播与归约操作。③跨节点发送线程块将传输命令排入主机内存的队列。④主机代理线程将成批的命令提交给网卡。⑤网卡将数据传输到 1 号节点的对等 GPU；根据传输协议不同，由网卡或接收端代理通知数据到达。⑥接收线程块检测到到达标志后消费数据。可选的控制器读取进度计数器并发布目标通信块数量（虚线所示）。
:::

**在双节点 16 卡 H200 集群的测试中，mKernel 实现了 GEMM+AllReduce 最高 1.72 倍、Ring Attention 最高 1.88 倍的性能提升**，同时兼容 InfiniBand 与 AWS EFA 两种主流网络架构，且无需针对每种输入尺寸做静态调优。

![文章插图](/articles/2026-10-27-mkernel-tile-ring-attention-1-57aa4c8f/images/03.webp)

::: image-caption
5：AWS EFA 测试平台的吞吐量结果（2 节点 ×8 张 H200；单位为单 GPU TFLOPS，数值越高越好）。橙色柱代表 mKernel。测试所用的 DeepEP+DeepGEMM 实现不支持 EFA；其柱体为 ConnectX-7 平台的测量值，仅作参考。 该组实验验证了 mKernel 在公有云 EFA 网络环境下的通用性能优势，覆盖三大并行模式的五类内核。相比 cuBLAS/FlashAttention+NCCL 的非融合基线，mKernel 在所有场景下都取得了性能提升，其中 MoE Dispatch+GEMM 加速比可达 3.1-4.5 倍，Ring Attention 最高 1.78 倍，GEMM+AllReduce 最高 1.53 倍。相比其他融合内核方案如 Triton-distributed，mKernel 也普遍领先，尤其在小问题规模下优势更明显，因为细粒度融合能更有效抵消短序列下通信占比高的问题。DeepEP+DeepGEMM 即便在性能更强的 InfiniBand 平台上，也不如 EFA 平台上的 mKernel，进一步体现了 mKernel 融合设计的性能优势。
:::

![文章插图](/articles/2026-10-27-mkernel-tile-ring-attention-1-57aa4c8f/images/04.webp)

::: image-caption
6：ConnectX-7 InfiniBand 测试平台的吞吐量结果（2 节点 ×8 张 H200；单位为单 GPU TFLOPS）。红色叉号代表对应基线在该问题规模下运行失败。 该组实验是 mKernel 在高性能 InfiniBand 集群上的核心性能验证，也是论文最优性能结果的来源。相比非融合基线，GEMM+AllReduce 最高加速 1.72 倍，Ring Attention 最高加速 1.88 倍，AllGather+GEMM 最高 1.34 倍，全面领先各类基线方案。相比 Triton-distributed、Mercury、MagiAttention 等同类型融合方案，mKernel 在所有规模下都保持性能领先，且在长序列等基线无法运行的场景下仍能正常执行。InfiniBand 平台下性能提升略高于 EFA 平台，因为更低的网络延迟让 tile 级流水的优势能更充分发挥。该组结果充分证明了多节点融合内核架构的性能价值，能够有效缓解分布式大模型的通信瓶颈。
:::

## 本文目录

- 一、分布式训练的通信困局：为什么内核级融合走到了跨节点这一步
- 二、mKernel 整体架构：单内核内实现三层流水的 tile 级重叠
- 三、关键技术拆解：自适应 SM 划分与跨平台通信层的核心巧思
- 四、五类融合内核实现：覆盖三大并行范式的全场景支持
- 五、实验性能全面评测：双网络环境下的性能验证
- 六、行业启示与未来展望：融合内核的下一个阶段

![文章插图](/articles/2026-10-27-mkernel-tile-ring-attention-1-57aa4c8f/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[10.3 倍加速！康奈尔大学 Perseus：终结多节点 MoE MegaKernel 的隐藏序列化噩梦，让代理传输反超 GPU-direct 1.2 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901207&idx=1&sn=4e68da37ab9ecd3a8d20e16451ea6ebe&scene=21#wechat_redirect)**
- **[Weave：面向计算通信重叠的 MoE MegaKernel 细粒度动态 SM 调度，较 DeepEP 1.95 倍层加速，1.12 倍端到端提速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906220&idx=1&sn=177db891e7fc7087f94090009ef44dc4&scene=21#wechat_redirect)**
- **[mlc-ai/Pith-Train用1万行代码，把生产级MoE训练栈的4D并行、DualPipeV重叠调度与FP8算子，压缩成可读、可改、可验证的Python工程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900698&idx=1&sn=3fee9786434b20fc247af5e3d7643b69&scene=21#wechat_redirect)**

:::
