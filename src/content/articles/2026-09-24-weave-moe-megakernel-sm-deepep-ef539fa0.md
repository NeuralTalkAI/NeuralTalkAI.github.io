关键词：MoE 推理、**计算通信重叠**、专家并行、**SM 动态调度**、:mdnice-bold-italic[持久MegaKernel]

> Weave 在全部 18 种配置下都实现了最低的端到端延迟：相对 DeepEP 为 1.12 倍，相对 Comet 为 1.13 倍，相对 TD 为 1.28 倍，相对 SGLang 为 1.50 倍，相对 PK 为 1.70 倍。

“分发与聚合操作可占据 MoE 层延迟的 40%”，这是分布式 MoE 推理领域公认的核心痛点。随着大模型向 MoE 架构演进，**专家并行成为分布式部署的标准范式，但跨 GPU 的 token 通信开销也随之成为制约推理效率的核心瓶颈。**

现有系统普遍通过计算通信重叠技术掩盖通信延迟，即将 GPU 的流多处理器（SM）划分为通信组和计算组，让两类任务并行执行。:mdnice-bold-italic[但现有方案要么在编译时固定 SM 分区比例，要么仅按序列长度做粗粒度切换，都无法适配每层路由结果带来的动态负载变化，最终导致 GPU 资源在空间和时间两个维度上的双重浪费：]

::: block-1
- :mdnice-highlight[空间上，SM 比例与实际通信计算量不匹配，要么带宽闲置要么算力空转；]
- :mdnice-highlight[时间上，数据依赖产生的流水线气泡让大量 SM 处于空闲状态。]
:::

![文章插图](/articles/2026-09-24-weave-moe-megakernel-sm-deepep-ef539fa0/images/01.webp)

::: image-caption
图 2：四种 MoE 执行模式对比。(a) 无重叠（TD）：所有 SM 串行执行分发、GEMM 和聚合操作；(b) 静态 SM 划分（DeepEP）：通信与计算 SM 数量编译期固定，存在空闲气泡；(c) 粗粒度动态（Comet）：SM 数量按迭代调整；(d) 细粒度动态（Weave）：SM 划分按层、按 GPU 动态调整。四张子图从时间维度完整呈现了 MoE 推理调度技术的演进路径，从完全串行到静态划分，再到粗粒度动态，最终到 Weave 的细粒度动态调度。演进的核心逻辑是不断缩小调度粒度，让 SM 分配更贴合实际的动态负载，同时不断压缩流水线空闲气泡，提升计算与通信的重叠度。Weave 的方案突破了现有方案的粒度限制，首次实现了层级、GPU 级的动态 SM 调度，代表了当前 MoE 重叠技术的最优方向。
:::

![文章插图](/articles/2026-09-24-weave-moe-megakernel-sm-deepep-ef539fa0/images/02.webp)

- **Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap**
- 组织机构：上海交通大学、新加坡国立大学、阿里巴巴集团
- 论文链接：https://arxiv.org/pdf/2609.21483
- :mdnice-bold-italic[Our implementation will be open-sourced upon acceptance.]

::: related-title
**相关推荐**
:::

::: related-links
- **[较 SGLang 1.21 倍提速！ForgeMegakernel 开启 AI 推理 MegaKernel 的自动化锻造时代](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905880&idx=1&sn=d33d843035f0dfc0a9af3f02f2811a94&scene=21#wechat_redirect)**
- **[拆解 megakernel 性能神话：普通 CUDA 内核靠 PDL 调度实现 97.4%持久化吞吐](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905121&idx=1&sn=f93dceb62089296ffcabc24c964e94c3&scene=21#wechat_redirect)**
- **[Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel，支撑万卡模型训练！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904378&idx=1&sn=83e25209c3e1ffc683b97a6f8646f250&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**

:::

> 由上海交通大学、新加坡国立大学、阿里巴巴集团联合提出的 Weave 系统，是首个**基于每层每 GPU 的路由结果，在运行时执行细粒度动态 SM 调度的 MoE 计算通信重叠系统。**

![文章插图](/articles/2026-09-24-weave-moe-megakernel-sm-deepep-ef539fa0/images/03.webp)

::: image-caption
图 7：Weave 系统架构概览。系统接收路由结果（软件输入）与通信带宽 / 计算吞吐量曲线（硬件输入），在巨型内核内部从两个维度进行运行时决策：空间调度器（如何在通信与计算间划分 SM）和时间调度器（两类工作单元如何协调执行），输出应用于当前层的优化执行计划。该架构的核心创新在于将调度逻辑内嵌到持久化 MegaKernel 中，实现了层内在线调度。传统方案的调度决策在主机端离线完成，开销大且粒度粗；而 Weave 将轻量成本模型放在 GPU 内核序言阶段，利用路由完成后的负载信息，微秒级完成调度决策。这种端到端的内核内调度，避免了主机与 GPU 的交互开销，也能支撑更细粒度的动态调整，是实现层级、GPU 级调度的关键架构支撑。
:::

其核心洞察在于：**一旦路由阶段完成，每层的通信量与计算量就完全确定，此时可以通过一个运行在持久MegaKernel内的轻量代价模型，同时优化空间上的 SM 分区比例和时间上的任务执行顺序。**

在 4 张 H100 SXM GPU 的测试环境下，Weave 在 6 个主流 MoE 模型上实现了 2.89 倍的 MoE 层几何平均加速，以及 1.33 倍的端到端几何平均加速，**全面超越五大现有 SOTA 系统。**

![文章插图](/articles/2026-09-24-weave-moe-megakernel-sm-deepep-ef539fa0/images/04.webp)

::: image-caption
图 10：4×H100 SXM、专家并行度 4、BF16 精度下的 MoE 层延迟与端到端延迟。第一行：六个模型、三种序列长度（2k/4k/8k）下的 MoE 单层延迟（y 轴对数刻度）；第二行：相同配置下的端到端延迟拆解，灰色条为注意力层耗时，上方彩色条为各系统的 MoE 层耗时。该图全面展示了 Weave 的性能优势。在所有 18 种配置下，Weave 的 MoE 层延迟都是最低的，几何平均加速比达到 2.89 倍，端到端几何平均加速比 1.33 倍。MoE 层的加速收益在端到端中有所稀释，因为注意力层耗时随序列长度平方增长，长序列下注意力占比更高。但即便如此，Weave 依然在所有配置下保持端到端性能领先，证明了细粒度调度优化在实际推理场景中的实用价值。
:::

![文章插图](/articles/2026-09-24-weave-moe-megakernel-sm-deepep-ef539fa0/images/05.webp)

::: image-caption
图 11（总述）：MoE 前向窗口的聚合 GPU 指标（DSv2-Lite、序列长度 2048、4×H100、专家并行度 4）。柱形为平均值，虚线顶端空心圆为峰值。(a) SM 活跃率（%）。柱形为平均值，虚线顶端空心圆为峰值，Weave 的 SM 平均活跃率达到 91%，远高于其他基线系统；(b) NVLink 利用率（%）。柱形为平均值，Weave 的 NVLink 平均利用率为 7.0%，峰值利用率远高于其他系统；(c) 张量核心 HMMA 利用率；(d) 计算 - 通信重叠率。四个维度的指标全面验证了 Weave 的资源利用优势。无论是计算侧的 SM 活跃率、张量核心利用率，还是通信侧的 NVLink 利用率，以及两者的重叠率，Weave 都全面领先现有方案。这说明 Weave 的空间 + 时间联合调度，同时优化了计算资源和通信资源的利用效率，并且通过提升重叠度让两者并行发挥作用，最终转化为显著的性能提升。
:::

## 本文目录

- 一、MoE 分布式推理的通信困局：计算通信重叠的天花板在哪
- 二、Weave 核心洞察：路由完成后，所有工作量都是确定的
- 三、空间调度器：按真实负载动态切分 SM 资源
- 四、时间调度器：用流水线与气泡窃取消除空闲间隙
- 五、统一代价模型与 Megakernel 工程实现
- 六、实验评测：全面碾压五大 SOTA 基线
- 七、总结与展望

![文章插图](/articles/2026-09-24-weave-moe-megakernel-sm-deepep-ef539fa0/images/06.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[较 SGLang 1.21 倍提速！ForgeMegakernel 开启 AI 推理 MegaKernel 的自动化锻造时代](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905880&idx=1&sn=d33d843035f0dfc0a9af3f02f2811a94&scene=21#wechat_redirect)**
- **[拆解 megakernel 性能神话：普通 CUDA 内核靠 PDL 调度实现 97.4%持久化吞吐](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905121&idx=1&sn=f93dceb62089296ffcabc24c964e94c3&scene=21#wechat_redirect)**
- **[Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel，支撑万卡模型训练！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904378&idx=1&sn=83e25209c3e1ffc683b97a6f8646f250&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**
- **[适配 Ada 中端卡的 MegaKernel 在线广告方案 Ada-MK：共享内存直接减半，端到延迟最高降 50%，击穿小批量毫秒级推理瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902893&idx=1&sn=5f53dd32ac726cea553f1c3743d8d2a1&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**
- **[10.3 倍加速！康奈尔大学 Perseus：终结多节点 MoE MegaKernel 的隐藏序列化噩梦，让代理传输反超 GPU-direct 1.2 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901207&idx=1&sn=4e68da37ab9ecd3a8d20e16451ea6ebe&scene=21#wechat_redirect)**
- **[从 MegaKernel 视角重新定义 NPU 上的线性注意力极限！PTO-ISA 驱动的昇腾 NPU Gated DeltaNet MegaKernel 浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901380&idx=1&sn=4ae6987a6ba87626737c46cb4be9458e&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**
- **[超越 vLLM 与 SGLang！Event Tensor：以动态 MegaKernel 消除重编译，解锁GPU核间通信-计算重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900086&idx=1&sn=fb8be688929f686853cc12c815e759ae&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)**

:::
