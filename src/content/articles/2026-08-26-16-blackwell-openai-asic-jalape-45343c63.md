关键词：**OpenAI**、Jalapeño、:mdnice-bold-italic[推理芯片]、**能效比**、软硬协同

> OpenAI 在 Hot Chips 上正式公布首款自研推理芯片 Jalapeño，SemiAnalysis 受邀进入实验室用 InferenceX 套件实地验证。

这颗与博通合作、从零设计的 ASIC 于 2024 年中立项，**约 16 个月完成流片，首代产品即在多款主流开源模型上击败 NVIDIA、AMD、Google 的全部在售芯片。**

::: block-1
**核心成绩是每兆瓦 token 吞吐量全面领先**：:mdnice-highlight[在不开多 token 预测、不做投机解码、不做 PD 分离的"裸奔"状态下，STP 成绩仍越过 Vera Rubin 的 MTP 公开数据]；DeepSeek R1 单用户并发下超 700 token/s，Kimi K2.5 上达次优芯片的 9 倍。
:::

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/01.webp)

::: image-caption
OpenAI Jalapeño 与 H100、H200、GB200、GB300、MI355X 等多款 AI 芯片，统计 FP16、FP8、FP4 算力、HBM 容量、显存带宽与 TDP 功耗，Jalapeño 不支持 FP16，FP8 算力 3.4PFLOPS、FP4 算力 13.4PFLOPS，搭载 216GiB HBM，显存带宽达到 15.4TB/s，TDP 为 700W，在 700W 功耗下拥有极高的显存带宽，适合大模型推理，同时表格也列出 Rubin、MI450X 等更高规格芯片作为参照。
:::

- **OpenAI Jalapeño: Better Than Nvidia Blackwell**
- https://newsletter.semianalysis.com/p/openai-jalapeno-better-than-nvidia

::: related-title
**相关推荐**
:::

::: related-links
- **[“这不只是 GPU 的问题，而是关乎所有层面”，Brendan Gregg 从 Intel 到 OpenAI 的性能工程新征程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897068&idx=1&sn=a17d384d80cb85f3e3b8e8e42a0f78d4&scene=21#wechat_redirect)**
- **[Anthropic 挖来谷歌 TPU 创始人，大模型公司集体转无晶圆厂芯片公司](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904745&idx=1&sn=45ff740c3c0c43f38b93fbf7f5afa4ac&scene=21#wechat_redirect)**
- **[从清华到 OpenAI：一位 AI Infra 工程师的成长与思考——对话翁家翌](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896307&idx=1&sn=61a45f13fdab2f25a3b0d70944f03d2e&scene=21#wechat_redirect)**

:::

> 硬件上，Jalapeño 采用台积电 N3P 工艺、单光罩 die 实现 13.4 PFLOPs MXFP4 算力，TDP 仅 700W，搭配 15.4TB/s 的 HBM4，**每瓦 HBM 带宽与每瓦 FLOPs 双双登顶**。

:mdnice-bold-italic[架构核心是消除固定延迟：slice 化 HBM 直连、乱序核心加 L1 缓存、小维度脉动阵列]，配合自研 Gluon 语言与 Codex 自动写 kernel，两周内吞吐翻倍、8 天从 TP8 扩到 TP32。机架层面由 Katsu 主机托盘、Vindaloo ASIC 托盘与 Chana 交换托盘组成，单柜 128 颗芯片，scale-up 可扩至 2,048 颗。

SemiAnalysis 同时给出保留意见：成绩均来自 8k1k 单轮负载，缺乏 AgentX 多轮长上下文验证；**与 Blackwell 对比并不公平，真正的对手 Rubin 已在出货**。:mdnice-bold-italic[B0 stepping 可再提 25% 能效，量产 2027 年爬坡，CUDA 护城河面临真实威胁。]

## 本文目录

- 两个关键问题：通用性边界与能效优势的成色
  - 问题一："通用推理"的边界在哪：单轮基准优势不等于全场景通用
  - 问题二：能效优势会不会被竞品软件优化抹平：架构红利大于价差
- 一、定位澄清：Jalapeño 是通用推理芯片，而非 OpenAI 模型专用
  - 1.1 成绩背后的三点保留意见
- 二、性能分析：为什么"每兆瓦 token 数"是唯一重要的指标
- 三、硬件规格与架构：700W 功耗下如何做到能效第一
  - 3.1 芯片迭代节奏：A0 跑出实测成绩，B0 能效再提 25%
  - 3.2 HBM4 显存配置：15.4TB/s 带宽超过所有 HBM3E 产品
  - 3.3 与 Rubin 的时间线对比：流片晚一个月，进度反而更快
  - 3.4 核心架构解析：slice 化内存层级与乱序核心设计
- 四、软件栈：从零搭建却跑得比 CUDA 还快
  - 4.1 像写汇编一样写 kernel，Codex 接管了人
  - 4.2 Gluon 语言与 Linear Layouts：kernel 编程的底层抽象
  - 4.3 软件迭代速度：两周吞吐翻倍，8 天从 TP8 扩到 TP32
  - 4.4 不做 PD 分离的理由：利用率、局部性与流量的动态变化
- 五、机架与组网：单柜 128 颗芯片，scale-up 扩展至 2,048 颗
  - 5.1 机架组成：Katsu 主机托盘、Vindaloo ASIC 托盘与 Chana 交换托盘
  - 5.2 双层 scale-up 网络：本地域铜缆全互联，全局域铜光混合
  - 5.3 部署计划：2027 年量产爬坡，为下一代模型预留选择权
- 结语：首代即领先的底气来自全栈，悬念留给长上下文

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/02.webp)

## 两个关键问题：通用性边界与能效优势的成色

### 问题一："通用推理"的边界在哪：单轮基准优势不等于全场景通用

> 文中反复强调 Jalapeño 是"通用推理芯片"而非 OpenAI 模型专用，但**目前所有亮眼能效成绩均来自 8K 上下文的单轮推理场景，且架构上刻意放弃了 Prefill/Decode 解耦这一业界公认的提效手段**。在多轮 Agent、长上下文正在成为主流生产负载的当下，:mdnice-bold-italic[这种统一池化架构是否会在真实复杂流量下出现能效大幅倒退？所谓的"通用"是否只是相对于 OpenAI 自身当前负载的通用，而非全行业场景的通用？]

从测试边界与架构逻辑看，统一池化架构在真实复杂生产负载下的能效优势会打折扣，"通用"也有明确场景边界。

当前所有亮眼数据均基于 8K 上下文的单轮推理基准，作者明确承认尚未完成 AgentX 多轮长上下文测试；真实 Agent 负载会深度考验路由调度、前缀缓存管理、KV 搬运开销等系统能力，这些恰恰是单轮基准无法覆盖的。:mdnice-bold-italic[统一架构虽避免了解耦方案的资源碎片化，但单芯片同时承载 prefill 与 decode，理论峰值利用率天然弱于专用拆分池。]

**它的"通用"是相对于强专用 ASIC 的通用**：支持小矩阵尺寸避免性能悬崖，能跑非 AI 负载，不绑定 OpenAI 自家模型。:mdnice-bold-italic[但这种通用仍局限在推理场景，不覆盖训练；且目前适配的模型均非业界最大尺寸，更大模型、复杂算子的适配效率尚未验证]。它解决的是"推理场景内的通用性"，而非 GPU 级别的全场景通用。

### 问题二：能效优势会不会被竞品软件优化抹平：架构红利大于价差

> 目前 Jalapeño 的性能对比，是用自身单 Token 预测的结果，对标英伟达、AMD 开启 MTP 推测解码后的最优配置仍实现领先，且:mdnice-bold-italic[很大程度依赖 Codex 驱动的软硬全栈定制优化。当 Rubin 等竞品完成软件成熟度爬坡、补齐全栈优化后，Jalapeño 的代际能效优势是否会被大幅收窄？OpenAI 自研芯片真正击穿的是英伟达的硬件技术壁垒，还是只是凭借自用场景绕开了英伟达的高溢价，本质上仍是定点优化的工程胜利而非架构革命？]

竞品软件成熟后会收窄部分差距，但 Jalapeño 的底层架构优势不会被完全抹平；其胜利既是成本结构的胜利，更是架构取舍与软硬协同的工程突破，而非单纯绕开英伟达溢价。

**当前对比确实存在成熟度差**：

- Rubin 流片仅早一个月，但软件栈完成度更低，英伟达未开放第三方全量基准测试；
- 随着 CUDA 生态持续优化，GPU 的 MTP、解耦调度等特性会进一步释放性能，缩小能效差。

但 GPU 背负向后兼容的历史包袱，无法像 Jalapeño 一样采用简化内存层级、片上分片直连 HBM、乱序核心消除固定延迟等激进设计，这些底层差异是软件优化无法彻底弥补的。

TCO 优势确实包含博通毛利更低的因素，但并非核心：:mdnice-bold-italic[Meta、微软自研 ASIC 起步更早却未达同等水平，证明成本不是唯一变量。Jalapeño 真正的壁垒是从零开始的软硬全栈协同——用 Codex 自动生成最优内核、用 Gluon 语言打通布局抽象，这套能力是英伟达通用生态无法直接复制的。]

## 一、定位澄清：Jalapeño 是通用推理芯片，而非 OpenAI 模型专用

> 外界都在说 OpenAI 的芯片是为 OpenAI 自家模型专用的，这是错的。OpenAI 做的是一颗面向 AI 推理的通用芯片。

项目的时间线堪称疯狂，也证明"用 AI 加速芯片设计"的说法是真的。抛开速度不谈，OpenAI 砸了大钱，做了务实的设计决策，团队实力又极强，这个结果并不令人意外。

光看规格，它立刻就是一个有力的竞争者：

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/03.webp)

::: image-caption
Jalapeño 与各家旗舰加速器的规格对比表（来源：SemiAnalysis）
:::

HBM4 的使用让它足以对标 NVIDIA 和 AMD 的旗舰 GPU：

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/04.webp)

::: image-caption
Jalapeño 采用 HBM4，显存规格与 NVIDIA、AMD 旗舰 GPU 同级（来源：OpenAI）。Jalapeño 的芯片版图，采用先进多芯粒封装方案，中央为 Compute Die 计算裸片，承担 AI 矩阵运算核心任务；上方是 IO Chiplet 芯粒，集成 PCIe 等各类外部接口；芯片左右两侧共集成 6 颗 HBM4 高带宽内存，通过封装互联直连计算裸片，以此实现极高显存带宽。整套设计将计算、IO、高速显存集成于单封装内，面向大模型推理场景，满足高带宽、大容量显存的业务需求。
:::

很多媒体报道这颗芯片时，都在转述 OpenAI 几句随口的说法，称芯片会以其他芯片做不到的方式针对自家模型优化。这是错的。**Jalapeño 是一颗通用推理芯片，能跑各种模型、各种负载，包括我们的 InferenceX 基准** ——我们是和 OpenAI 工程师一起在实验室里跑的。:mdnice-bold-italic[作为玩笑，OpenAI 甚至给我们展示了在芯片上跑 Doom，移植只用了 Codex 提示词就搞定了。]

> **下面是头条级的 perf/W（性能功耗比）结果**：每全口径公用事业 MW 的 token 吞吐量。:mdnice-bold-italic[Jalapeño 碾压了所有其他芯片。而且这些结果是在没有多 token 预测（MTP）的情况下取得的，而图表上其他芯片都是各自的最佳配置，全部开了 MTP。]

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/05.webp)

::: image-caption
核心结果：Jalapeño 的 token 吞吐量/MW 大幅领先所有竞品，且未使用 MTP（来源：SemiAnalysis）。对比多款 AI 芯片每兆瓦功耗下 token 吞吐量与交互度的关系。横轴代表用户交互度，纵轴是每兆瓦算力的 token 输出。随着交互度升高，所有芯片单位功耗吞吐量整体下滑。Jalapeño 芯片在低交互场景性能领先，MI355X 曲线异常标注存在精度问题。
:::

**Jalapeño 没有对曲线上的任何特定工作点做调优，却在几乎所有场景下的 perf/W 上都击败了 Blackwell**。它在低延迟和高吞吐场景中都很出色。更公平的对比是对标单 token 预测（STP）结果——它把所有竞争对手都甩在身后。:mdnice-bold-italic[在低并发场景下，Jalapeño 展现出惊人的交互性：在 DeepSeek R1 模型上，并发数为 1 时单用户超过 700 token/s。]

::: block-1
:mdnice-highlight[难以置信的是，这一切都建立在单 token 预测之上，没有投机解码（speculative decoding），也没有 prefill-decode 分离]。除 DeepSeek R1 之外，我们还看到了其他模型，包括 Kimi-K2.5 和 GPT-OSS，大约跑到 1,400 token/s/用户。对所有模型，我们确认了 Jalapeño 的 GSM8k 评测结果与 NVIDIA 芯片持平。
:::

### 1.1 成绩背后的三点保留意见

第一，所有数字均由 OpenAI 提供。我们在实验室现场验证了 InferenceX 的运行，但没有跑完整套 InferenceX[1] 基准，也没有看到 AgentX[2] 的结果。AgentX 是我们比较芯片性能时更偏好的套件，因为其数据集的长上下文、多轮特性反映了真实生产工作流的缓存行为。在 8k1k 上表现好的框架，到了 AgentX 上可能表现更差，因为真实生产负载会压测路由、prefix cache 机制、缓存管理、offload 基础设施等组件——单轮 8k1k 测不到这些。

第二，我们认为与 Blackwell 的对比并不完整，也不太公平。J:mdnice-bold-italic[alapeño 真正的竞争对手是 Rubin 这类同样使用 HBM4 的芯片。Vera Rubin 系统眼下正在开始向客户发货，而 OpenAI 手里的 Jalapeño 还只有工程样品。]

::: block-1
所以性能上真正该比的是 Rubin 而非 Blackwell；某种意义上，我们本来也预期 Jalapeño 这样的定制芯片会跑赢 Blackwell。Vera Rubin NVL72 的 perf/MW 是 GB200 NVL72 的 5.4 倍，我们在上月分析 NVIDIA 与 CoreWeave 发布性能声明的文章里讨论过[3]。下文我们会把 Jalapeño 与 Vera Rubin 今年 7 月的性能数据放在一起比。
:::

第三，被测模型不在开放 frontier 上。NVIDIA 和 AMD 已经在 AgentX 上发布了 DeepSeek V4 Pro、Kimi K3 等更大模型的结果。模型越大、发布越近，在新芯片上 bring-up 就越复杂。当然，OpenAI 已经在 Jalapeño 上跑起来的这些模型也算不上小。

## 二、性能分析：为什么"每兆瓦 token 数"是唯一重要的指标

> OpenAI 的设计目标是 perf/W。原因很简单：**OpenAI 目前受限于数据中心电力，而不是预算或机房面积，因此每 MW 的 token 数至关重要**。在 Computex 2026 上，黄仁勋说 perf/W、可靠性和长寿命是未来 GPU 的核心特性。原话是："如果你有 1 GW 电力，那么每瓦吞吐量就是收入。"他还提到，仅仅因为芯片更便宜就选错架构是没有意义的。

NVIDIA 在 Hot Chips 2026 的 Vera 演讲中展示同一张收入图时也强调了这一点："今天的数据中心是受电力限制的。"电力决定收入。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/06.webp)

::: image-caption
Computex 2026 主题演讲：数据中心受电力限制，每瓦吞吐量即收入（来源：Computex 2026 keynote）。黄绿色曲线代表 AI 工厂营收，呈现快速拉升后趋稳的增长态势。蓝色曲线代表系统负载表现。图中标注两个关键指标：TTFI（首 token 生成时延）向左代表时延降低，提升用户响应速度；Tokens/Watt 代表每瓦电能产出 token 数量，向上代表能效提升。该图表达更低推理时延、更高算力能效，能够直接拉动 AI 工厂业务收入增长，体现能效与响应速度对 AI 商业化收益的关键作用。
:::

运营商没法说加电就加电，因为加 GPU 和加电网容量走的是完全不同的时间尺度。数据中心的电力包络受制于公用事业并网、基础设施、制冷能力和 UPS/备用发电设计。电网延迟一再超过硬件和建设进度，这驱动了 BtM（behind-the-meter，表后电源）容量的需求：建在数据中心现场的燃气轮机和自备发电机。这部分容量绕开公用事业的电表，不从公共电网取电。运营商可以靠它直接给设施供电，无需等待并网和电网升级——这正是 xAI 的 Colossus 2 重度依赖 BtM 的原因，它实际的电网接入远远滞后。更多内容见我们的 Energy Model[4]。

> 如在 X 帖子中所写，tok/s/MW 可以化简为每焦耳 token 数，因为 1 瓦就是每秒 1 焦耳。这让 tok/s/MW 能代表一个系统将能源转化为 token 的效率。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/07.webp)

::: image-caption
tok/s/MW 本质上等于 tokens/J，是系统能效的直接度量（来源：SemiAnalysis）。通过单位换算对比 B300 芯片与人类大脑的语言输出能效，把芯片算力换算成每卡路里热量可输出 token 数量，算出 B300 约 30.8 token / 卡路里；而人脑说话可达约 690 token / 卡路里，人类语言输出的能效大约是 B300 芯片的 22 倍。下方 AI 推理性能曲线图则展示不同交互度下每兆瓦功耗的 token 吞吐量，直观体现 AI 硬件在能量效率上和人脑仍存在巨大差距。
:::

> 在这个维度上，即使与 Rubin 相比，Jalapeño 也赢了。OpenAI 的 Jalapeño 以 STP 取得的每 MW 输出 token 吞吐量，超过了 NVIDIA 与 CoreWeave 今年 7 月公布的 Vera Rubin MTP 结果，更远超 GB200 的 2025 年 MTP 结果[5]。

正如我们在 Vera Rubin 文章里提到的，VR 当时对比的是 2025 年的 GB200 结果，因为那是类似的早期 bring-up 阶段，与 2025 年的 GB200 对比可以保持软件成熟度一致。沿用这个逻辑，我们把 Vera Rubin 最新的 2026 年 7 月结果、GB200 的 2025 年结果和今天的 Jalapeño 结果放在一起比。这个对比是成立的：这些是公开渠道能拿到的 Rubin 最好数字，而且 OpenAI 的流片时间晚于 Rubin。**OpenAI 和 Rubin 两边都还不成熟，性能还会继续爬升。**

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/08.webp)

::: image-caption
Jalapeño（STP）对比 Vera Rubin（MTP）与 GB200 的每 MW token 吞吐曲线（来源：OpenAI, SemiAnalysis）。对比 VR200、GB200、Jalapeño 三款芯片。横轴为用户交互度，纵轴代表每兆瓦功耗可输出的 token 吞吐量。交互度升高，三款芯片的单位功耗 token 输出均快速下降。低交互负载下，Jalapeño 与 VR200 能效表现领先，GB200 基线水平更低。
:::

补充一组 CoreWeave 官方博客里的原始数据，便于理解这条对比线的量级：CoreWeave 在 2026 年 6 月完成业界首次 Vera Rubin NVL72 整机 bring-up，7 月公布首个实测硅片性能——在 DeepSeek R1 上、相同交互性目标（约 150 TPS/用户）下，Vera Rubin NVL72 达到约 800,000 TPS/MW，是 GB200 NVL72（约 80,000 TPS/MW，2025 年基线）的 10 倍。**需要注意的是，CoreWeave 这组数字是在开启全部主流推理优化的情况下测得的**，包括大规模专家并行、NVFP4 精度、多 token 预测和 prefill-decode 分离，软件栈为 TensorRT-LLM 加 Dynamo。:mdnice-bold-italic[换句话说，Jalapeño 是在"什么都没开"的 STP 状态下，越过了对手"全副武装"的成绩线。]

**在 perf/TCO 上，Vera Rubin 和 Jalapeño 打成平手，每美元产出的输出 token 数几乎相同**。但如前所述，Jalapeño 的结果没用投机解码，而 Vera Rubin 用了。:mdnice-bold-italic[投机解码能为每 token 成本带来 3 倍以上的改善。等 Jalapeño 实装投机解码，它给出的 token 成本会低得多]。当然，这个 TCO 优势有一部分来自把 NVIDIA 的高毛利换成博通较低（但仍然很高）的毛利。**不过这不是全部原因。举个反例：Meta 和微软的 AI ASIC 项目搞了更久却没能落地，说明成本只是等式的一部分**。Jalapeño 的完整 TCO 拆解见 SemiAnalysis AI Cloud TCO 模型[6]。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/09.webp)

::: image-caption
perf/TCO 维度上 Vera Rubin 与 Jalapeño 基本持平，但 Jalapeño 尚未启用投机解码（来源：OpenAI, SemiAnalysis）。上图表示每 1 美元成本可输出的 token 数量，横轴为用户交互度，纵轴代表每美元产出 token。图中对比 VR200 与 Jalapeño 两款芯片，低交互负载时 VR200 每美元产出 token 更高，成本效益更优；随着交互度上升，两条曲线同步快速下滑，性能差距缩小。
:::

架构上，OpenAI 选择不把 prefill 和 decode（PD）拆分到不同的芯片池。:mdnice-bold-italic[draft 模型和主模型共享同样的芯片和互联，这种设计哲学用一部分理论效率换取实际运营上的便利。动机在于负载结构会随时间变化：例如随着模型经历三个时代（知识、推理和智能体，输入 token、cache 写、cache 读、输出 token 之间的比例已经发生了显著变化。] 因此，一开始就固定采购异构的 prefill 硅片和 decode 硅片，长期看会造成低效。OpenAI 在这个架构里选择同构池，并努力让芯片在所有任务上都表现出色。

它确实做到了在 Kimi K2.5（Cursor Composer 2.5 的基座模型）上，Jalapeño 达到接近 700 token/s/用户，在 100 token/s/用户的同交互性水平上是次优芯片的 9 倍以上。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/10.webp)

::: image-caption
Kimi K2.5 上的吞吐-交互性曲线：Jalapeño 约为次优芯片的 9 倍（来源：OpenAI, SemiAnalysis）。横轴代表用户交互度，纵轴为每兆瓦功耗产出的 token 吞吐量。多款芯片曲线随交互度提升，单位功耗性能普遍快速下降。Jalapeño 芯片在全部负载区间能效遥遥领先，GB200 紧随其后，其余芯片性能更低。
:::

在 GPT-OSS 上又是一场屠杀。Jalapeño 同等交互性下的每 MW 吞吐量接近 GB200 最高吞吐点的两倍，是 GB200 并发 1 工况点的 50 倍以上。Jalapeño 的高并发数据点使用了 EP8。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/11.webp)

::: image-caption
GPT-OSS 上的对比：Jalapeño 同交互性吞吐/MW 接近 GB200 峰值的两倍（来源：OpenAI, SemiAnalysis）。横轴为用户交互度，纵轴代表每兆瓦功耗的 token 吞吐量。图中对比多款 AI 芯片，Jalapeño 芯片在全区间能效大幅领先。所有芯片均呈现交互度越高，单位功耗吞吐量快速衰减的趋势。
:::

这些结果很亮眼，但我们还是要挑刺：它们都只是 8k1k，一个容易调优得多的负载，而且还没有 AgentX[7] 的运行结果。如我们在 AgentX[8] 文章里所说，多轮、长上下文负载会对 serving 栈的更多环节形成压力，比如路由和 prefix cache。想在智能体负载上表现出色，还需要大量优化工作。

## 三、硬件规格与架构：700W 功耗下如何做到能效第一

### 3.1 芯片迭代节奏：A0 跑出实测成绩，B0 能效再提 25%

> 所有这些结果都是在 Jalapeño 的 A0 stepping 上拿到的，彼时项目才启动 9 个月。**但 B0 stepping 已经进了晶圆厂，能在 A0 硅片基础上带来约 25% 的每瓦性能提升**。具体来说，B0 在单个光罩尺寸的计算 die 上实现 13.4 PFLOPs 的 MXFP4 算力，采用台积电 N3P 工艺制造。相比之下，Rubin 单个计算 die 的 dense NVFP4 算力为 17.5 PFLOPs，die 尺寸相近，工艺节点相同。

**考虑到 Jalapeño 的 TDP 只有 700W，而 Rubin 每个计算 die 是 900–1,150W，这个差距就要重新评估了**。Jalapeño 面向推理而非训练，OpenAI 不需要靠推高 TDP 去堆 FLOPs，这可以理解。无论如何，上述数据说明 Jalapeño 的峰值理论算力相当可观。

与其他加速器直接对比，**Jalapeño 拥有最高的每瓦 HBM 带宽和最高的每瓦 FLOPs**，可与 1,800W 的 Rubin Max-Q 配置相提并论：

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/12.webp)

::: image-caption
各家加速器每瓦 HBM 带宽与每瓦 FLOPs 对比：Jalapeño 双双登顶（来源：SemiAnalysis）。Jalapeno 功耗仅 700W，FP4 算力 13.4PFLOPS，HBM 带宽 15.4TB/s，216GiB 显存。它的每瓦 HBM 带宽 22、每瓦算力 19.1，大幅优于 H100、GB200 等竞品，能效表现突出。Rubin、MI450X 算力更高，但 TDP 功耗也显著拉高。
:::

封装外 I/O 由一颗 N3E 工艺的 I/O chiplet 提供，带 32 条 800G SerDes 通道用于计算互联：其中 24 条（600GB/s）用于机架内的本地 scale-up，8 条（200GB/s）用于全局 scale-up，即 2,048 XPU 的多机架域。系统 I/O 走 PCIe Gen 5，连接 x86 主机 CPU。

### 3.2 HBM4 显存配置：15.4TB/s 带宽超过所有 HBM3E 产品

> Jalapeño 将配备 HBM4，这让它成为继 NVIDIA 和 AMD 之后较早的一批采用者，**甚至跑在了 TPU 和 Trainium 这些老牌项目前面**。:mdnice-bold-italic[Jalapeño 的核心架构原则之一就是榨干 HBM 带宽——既然如此，不用最好的 HBM 就等于违背自己的设计目标]。最终**每封装 15.4TB/s 的显存带宽，超过了所有正在出货的 HBM3E 加速器**。15.4TB/s 意味着其 HBM4 能达到 10Gbps 的引脚速率，略优于 NVIDIA 在 Rubin 上从 HBM4 拿到的 9.6Gbps。HBM 大概率由三星供应。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/13.webp)

::: image-caption
Jalapeño 芯片完整研发时间线，2024 年 10 月启动架构概念设计，次年 2 月完成初始 RTL，4 月完成 Codex 命令行版本，7 月 RTL 冻结，从初始 RTL 到流片仅耗时 9 个月，2025 年 11 月完成流片。后续 2026 年 2 月实现桌面端 Codex，5 月拿到首批硅片、芯片跑通 Codex，7 月芯片可运行 ChatGPT。该时间线体现这款 AI 芯片从架构构思到大模型可实际运行，整体开发节奏很快。
:::

### 3.3 与 Rubin 的时间线对比：流片晚一个月，进度反而更快

> OpenAI 于 2025 年 11 月完成 Jalapeño 流片——更准确地说，这是整个 CoWoS 设计的流片，而不只是顶层 die 的硅片。**从流片算起 9 个月内、在实际硅片上只做了 3 个月 bring-up 的情况下，OpenAI 已经交出了非常好的结果**。考虑到团队是从零开始搭软件栈，这一点尤其惊人。

作为对照，**Rubin 的 CoWoS 流片完成于 2025 年 10 月，比 Jalapeño 还早一个月**，但目前我们看到的早期结果只有 CoreWeave 的工程样品。NVIDIA 没有像 OpenAI 这样开放第三方实测并发布基准，说明他们的芯片软件还不够成熟。:mdnice-bold-italic[鉴于 OpenAI 在自家硅片上 bring-up 新模型的速度，CUDA 护城河可能真要保不住了]。

两边都远未优化到位，但总体上 Jalapeño 交出的数字更好。我们不认为 NVIDIA 硬件不行，更准确的解读是：**Jalapeño 的软件 bring-up 进度比 NVIDIA 快**。这就是软硬件协同设计的力量，也是一支精干的前沿实验室 ASIC 团队最能胜过老牌商用芯片厂商的地方。反直觉的是，:mdnice-bold-italic[从零开始反而帮了 OpenAI：它可以做干净利落的架构决策，不用操心向后兼容或旧版软件]。

OpenAI 目前手握 Jalapeño 工程样品，量产计划是在 2027 年内逐步爬坡，大部分产出安排在明年年底。关于单位出货量与 ASP 的更多细节，见 SemiAnalysis Accelerator Model[9]。可以说，OpenAI Jalapeño 是一颗货真价实的大批量 ASIC。

和 Rubin 的时间线摆在一起，Jalapeño 快得惊人：尽管对手起步更早，成绩却被反超了。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/14.webp)

::: image-caption
Jalapeño 与 Rubin 的开发时间线对比：起步晚、速度快、成绩反超（来源：SemiAnalysis）。Vera Rubin 芯片项目时间线，其 2024 年 6 月对外公布产品路线图；2025 年 3 月正式披露 Vera Rubin，5 月拿到 HBM4 内存样品，8 月确认完成流片。2026 年 1 月发布配套硬件平台，2 月产出首批芯片样品，3 月 7 款芯片进入生产，5 月全面量产爬坡，7 月 NVL72 正式上线落地。
:::

### 3.4 核心架构解析：slice 化内存层级与乱序核心设计

> 深入架构层面：这颗芯片的矩阵引擎使用 MXFP 数值格式和权重静止（weight stationary）脉动阵列，思路与 TPU 类似。但与 TPU 不同的是，**它支持更小的 shape/维度，因此没有大脉动阵列在不规整 matmul 形状上暴露的性能悬崖**。

芯片还配有 64 位标量核心和 FP32/INT32 向量核心。OpenAI 在托盘（tray）层面做了冗余设计，并在核心与通道层面内建了良率收割（yield harvesting）。他们称 :mdnice-bold-italic[AI 辅助芯片设计带来了 SIMD 面积 8% 的缩减和矩阵引擎面积 10% 的缩减]；虽然没有明确具体的工艺/电压/温度（PVT）条件，但也提到 AI 辅助模块在时序和功耗上都优于初版模块。

**Jalapeño 架构设计的重心是消除 KV cache 和权重的内存搬移，以及固定延迟与开销**，目标是即使在小 batch 或小 shape 下，也能比其他加速器更接近原始峰值算力/带宽。

核心和 HBM 被划分为 slice，每个核心 slice 对自己的那片 HBM 有低延迟的本地视图，slice 之间的同步走一条高带宽的专用集合通信网络。:mdnice-bold-italic[这套极简内存层级给了 Jalapeño 相对 GPU 的巨大潜在优势：GPU 的内存访问必须穿过复杂的内存系统，产生的大延迟只能靠更大的 shape 去摊薄或隐藏]。

这个选择之所以可行，是因为只要权重和 KV 摆放得当，核心间的同步就可以限制在少量已知的高带宽通信上，比如可以与计算重叠的张量并行通信。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/15.webp)

::: image-caption
OpenAI Jalapeño 芯片架构框图。芯片划分为 64 组 Core slice 计算切片，每组切片直连独立 HBM slice 显存，通过 local view 做本地内存管理。内部设有两套片上网络：Collective network为高带宽低延迟网络，专门负责 AI 聚合通信；General NoC通用性更强，带宽更低、延迟更高，处理普通数据流。外部通过 Scale‑up Ethernet bridge 以太网桥接，支持多芯片横向扩展。该分层架构兼顾本地访存性能、片内高速集合通信，同时保留对外扩展能力。
:::

芯片上还有一张额外的通用 NoC，用于一般通信和访问 scale-up 网络。总体而言，**相比 NVIDIA 和 Google，OpenAI 用简化的 NoC 和内存子系统省下了大量功耗**，也换来了可观的性能增益。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/16.webp)

::: image-caption
Jalapeno 芯片内核子模块示意图。图中 CORE0‑CORE2 每个内核集成 tensor 张量单元、SIMD 向量单元、scalar 标量单元，搭配 L1 缓存与 fast local view 高速本地内存视图，实现本地高速数据访问。下层为SPECIALIZED COLLECTIVES专用集合通信单元，负责多个物理内核之间协同交互，适配大模型的集合运算。最底层是GLOBAL MEMORY VIEW全局内存视图，可访问全部存储空间，但访问速度更慢。
:::

在核心层面，OpenAI 拿出的是一个带 L1 缓存的乱序（OoO）核心。这与其他加速器的主流做法大相径庭——它们清一色使用软件管理的 scratchpad，通常搭配异步 DMA。**OpenAI 的论点是：乱序核心加缓存让 Jalapeño 能避开 barrier 延迟这类固定开销；在 GPU 上，这类开销必须靠提高每核心工作量来隐藏或摊薄，想逼近原始峰值带宽/算力就更难。**

代价也随之而来：**Jalapeño 要依赖良好的预取来保证内存请求及时到达，这比软件管理 scratchpad 更难预测、更难分析**。但是，把 Codex 放进一个能拿到详细 trace 的好 harness 里，针对给定 shape 寻找最优 kernel 和最优预取策略，大概率只需要很少的人工干预。我们认为这正是 OpenAI 能如此快速 bring-up DeepSeek R1、Kimi K2.5 和 GPT-OSS 的原因。

这些核心还支持"小"矩阵维度，这（取决于到底有多小）应该能让它在不同模型和 batch 维度下更通用，对矩阵维度对齐、padding 开销和 tiling 低效不那么敏感。举个例子，TPU、Trainium 和 Etched 的芯片都有非常大的脉动阵列，可能需要很大的 batch 或恰好整除的模型维度，才能避开 tiling 低效。

**通过消除系统中的固定延迟，OpenAI 让整条帕累托曲线上的所有区域都尽可能贴近 roofline 性能**。理论上，这可以在多个工作点上带来相对 GPU 的优势：

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/17.webp)

::: image-caption
上图衡量了每兆瓦功耗输出 token 吞吐量与交互性（单用户 token/s）的关系，基于 DeepSeek R1 671B FP4 模型推理。横轴为交互性，数值越高代表单用户生成速度越快；纵轴是每兆瓦能输出的 token 数，代表机房整体能效。绿色曲线为 VR200 NVL72，紫色两条曲线代表 OpenAI Jalapeño，8‑17 版本性能偏低，8‑22 更新版本大幅提升，整体能效逼近 VR200。曲线呈下降趋势：交互性越高，整机每兆瓦吞吐量越低。
:::

但附带一个警告：即使理论上限存在，真实 kernel 要兑现它可能更困难。所以实际的打法是：

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/18.webp)

::: image-caption
InferenceX 仿真结果，测试 OpenAI Jalapeño 芯片跑 DeepSeek R1 671B FP4 大模型。横轴Interactivity是单用户 token 生成速度，纵轴是每兆瓦功耗的 token 输出吞吐量，代表机房能效。曲线上标记 TP 张量并行度、C 并发用户数。并发数 C 越高，整体每兆瓦吞吐越高，但单用户速度下降；C=1 时单用户速度最快，集群整体能效跌到很低。
:::

从 OpenAI 团队在 Jalapeño 上 bring-up InferenceX 负载的惊人速度来看，我们对这条路线持乐观态度。

::: block-1
:mdnice-highlight[如果 Jalapeño 成功，它将释放一个强烈信号：行业对编程模型和"完美的通用编译器"的执念，会被前沿 AI 模型证伪。]
:::

## 四、软件栈：从零搭建却跑得比 CUDA 还快

### 4.1 像写汇编一样写 kernel，Codex 接管了人

> OpenAI 写 Jalapeño kernel 的方式像写汇编。**每个 kernel 都是手工调优的代码，有些长达约 3,000 行**，配有正确性检查和自研的 sanitizer。早期 kernel 工作是人在回路，而非全自动，但随着一个规模化升级的内部版 Codex 上线，情况变了——OpenAI 还打算把这个内部版 Codex 卖给企业客户。内部 serving 引擎叫"Teacup"。

::: block-1
有意思的是，**OpenAI 内部本来没有 MLA kernel 的实现，直到他们用 InferenceX 跑 DeepSeek 基准时才有**。:mdnice-highlight[Codex 能在没有任何 kernel 工程师介入的情况下，快速写出功能正确且高效的 kernel]，展示了这条软件管线的进化能力。
:::

### 4.2 Gluon 语言与 Linear Layouts：kernel 编程的底层抽象

> OpenAI 用 Gluon 给 Jalapeño 编程。**Gluon 是 OpenAI 的 kernel 编程语言，构建在 Triton 之上，保留 Triton 的 SPMD（单程序多数据）编程模型，同时暴露底层编程抽象**。例如对 NVIDIA GPU，它提供映射到 PTX 指令的 API，包括 MMA 指令、TMA 指令、mbarrier 机制等等。

**Gluon 最独特的抽象是 layout**：笼统地说，layout 定义硬件资源（如 warp 9 的第 5 个寄存器）与张量元素（如第 6 行第 7 列的元素）之间的映射。Gluon 的 layout 抽象基于 Linear Layouts[10]，这是 OpenAI 发明的一种 layout 代数，对"layout 是什么"做了数学形式化，并提供了操作 layout 的工具，从而支持可证明正确的 layout 转换和最优的内存 swizzling。

::: block-1
补充一下 Linear Layouts 这篇论文的核心思路（arXiv: 2505.23819，一作 Keren Zhou，作者列表里有 Triton 之父 Phil Tillet 和 Jeff Niu 等 GPU 编译器老手）：它用 $\mathbb{F}_2$（二元域）上的线性代数来建模张量 layout，把 layout 表示为作用在硬件表示的比特位上的二进制矩阵。**传统编译器对 layout 的处理是逐个 case 硬编码的，layout 转换会出现组合爆炸；线性化之后，layout 定义变成通用的，layout 到 layout 的转换也有了通用算法，直接消灭了二次爆炸问题**。:mdnice-highlight[这项工作已集成进 Triton，既优化了单个算子和完整 kernel，也顺带修掉了 Triton 旧 layout 系统的若干 bug。Gluon 在 Jalapeño 上的 TensorInfo 抽象，正是这套数学的落地。]
:::

在 Jalapeño 的编程模型上，每个 Gluon 程序映射到一个持久线程（persistent thread）。我们认为:mdnice-bold-italic[这暗示 Jalapeño 适合持久 kernel 的编程模式：每个程序在多个 tile 上执行，由程序员而非硬件调度器分配工作。OpenAI 提到的 TensorInfo 是一个显式编码 layout 的抽象，大概率就是为 Jalapeño 设计的一组 layout，由 Linear Layouts 驱动。] 此外，每个核心提供数据预取和解耦的乱序单元——例如，用户可以对一个预取数据写等待操作，由信号量（semaphore）锁定。

:mdnice-bold-italic[命运的奇特转折在于：GPT 5.6 Sol 这类目前跑在 NVIDIA GPU 上的 OpenAI 模型，被用来设计一颗对 CUDA 护城河构成真实威胁的芯片——NVIDIA 自己的 GPU 正在实时帮助迎接它的潜在继任者]。

### 4.3 软件迭代速度：两周吞吐翻倍，8 天从 TP8 扩到 TP32

> 跨时间对比也能看出 Jalapeño 的进化速度：**不到两周时间，在某些交互性水平下吞吐量提升超过 2 倍**。我们从 Jalapeño 团队拿到的每个 tarball 里都装着一个奇妙世界。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/19.webp)

::: image-caption
Jalapeño（哈拉佩诺）既是墨西哥青辣椒，也是 OpenAI 自研 AI 芯片的代号。圈内有个经典玩梗传统：把 DOOM 跑在各种奇奇怪怪硬件上。
:::

提升的不只是 kernel 性能：**短短 8 天内，Jalapeño 团队在原有 TP8 配置的基础上启用了 TP32**，越过单系统边界，让大模型跑上了完整机架级配置。这个开发节奏相当惊人。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/20.webp)

::: image-caption
对比大模型推理的两种集群部署模式，业务负载分为预填充密集、均衡、解码密集三种场景，上方的可弹性调度集群将硬件统一为一个资源池，可根据业务动态分配预填充与解码资源，不会出现资源浪费和请求拒绝的情况，而下方固定配比集群把硬件划分为独立的预填充池和解码池，负载波动时容易出现硬件空闲，或是直接拒绝用户请求，以此体现 OpenAI Jalapeño 芯片支持资源灵活切分，相比传统固定配比 GPU 集群能够实现更高的资源利用率。
:::

为了在真实硬件运行之前验证性能，OpenAI 还有一个模拟器"chilisim"，与实测硬件的误差在 5% 以内，使用定宽 trace 总线。A0 上的 trace 能力有限，但在 B0 上大幅增强，大概率吸收了 A0 硅片实际运行的反馈。工程师们已经演示过 Codex CLI 跑一个内部模型（绰号"Raiku"或"5.3 Codex Spark"），**TPOT 做到 1.2ms**。

团队还展示了直接在芯片上运行的 Codex 编写的 demo：36 FPS 的 Doom、一个 FP32 流体动力学模拟，以及一个"Liquid Light"鼠标拖拽可视化。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/21.webp)

::: image-caption
对比大模型推理下两种集群架构，Attention 为注意力模块，FFN 为前馈网络，业务分为短上下文、均衡、长上下文三类负载，上方可弹性调度集群使用统一硬件资源池，能够根据上下文长度动态分配 Attention 与 FFN 算力，适配不同业务负载，不会产生资源闲置或拒绝请求，下方固定配比集群将 Attention、FFN 硬件资源硬性拆分，业务负载变化时，会出现硬件空闲，或者直接拒绝请求，以此说明 Jalapeño 芯片算力资源可灵活调度，面对长短上下文混合业务，相比传统固定拆分集群资源利用率更高。
:::

在模型侧，OpenAI 内部的 megakernel 路线（绰号"gigakernel"）围绕单一 megakernel 构建，在设备上循环执行以降低 CPU 开销和启动时间。团队还在 test-time compute 策略上越走越远，内部特别关注的一个方向是：如何连贯地用好 100 万个 rollout。

### 4.4 不做 PD 分离的理由：利用率、局部性与流量的动态变化

> 前面提到，OpenAI 没有在这些芯片上做 prefill-decode 分离（PDD）。这让我们意外，因为 NVIDIA 和 AMD 的 GPU 性能从 PDD 中获益显著，即便在同构硬件上也是如此。我们来挖一下 Jalapeño 团队为什么这么选。

**PDD 在负载冻结时看起来很诱人**。Prefill 和 decode 对硬件的压力不同，把两个阶段各自分给单独调优的池子，可以在某个选定的输入/输出比上提升效率。但生产流量不会停在那个比例上：输入输出序列长度、并发、缓存命中率、投机接受率、延迟目标，全天都在动。

一旦设备被切成 prefill 池和 decode 池，prefill 需求过多会让 decode 芯片空转、请求排队；decode 需求过多则反过来。运营商必须持续预测正确的配比，在两边都预留冗余容量，还要不断再平衡一个理想比例永远在移动的系统。

:mdnice-bold-italic[在统一系统里，某些资源在特定阶段可能利用不足，但每台设备都能服务下一个请求；在分离系统里，一整颗芯片可能仅仅因为"分错了池"而闲置。局部利用率好看了，全局利用率可能很差]。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/22.webp)

::: image-caption
统一系统与分离系统的利用率对比：分离会在流量结构变化时造成整芯片闲置（来源：SemiAnalysis）。业务分为批量请求、均衡、高速请求三类，Fast 代表低延迟高速推理业务，Slow batched 代表高吞吐批量任务，上方弹性调度集群共用统一硬件资源池，可动态分配算力，灵活适配不同业务，不会出现资源浪费与请求拒绝；下方固定配比集群把硬件划分为高速机架与普通 GPU，负载发生变化时容易出现硬件空闲或者请求被拒绝的情况，体现 Jalapeño 芯片硬件资源可灵活调度，在混合业务场景中相比传统固定硬件划分集群，整体资源利用率更高。
:::

**分离还会破坏局部性**。Prefill worker 产生的大块 KV cache 是 decode worker 马上要的，系统必须先把这份状态跨网络搬过去，生成才能继续。这增加了带宽消耗、同步、排队，还多出一个故障域。成本还随输入序列长度上升，因为 KV cache 在长大。当然，避免搬 KV 主要是个功耗和延迟优化：愿意搬一部分 KV，可以换来更高的硬件利用率，代价是一些功耗和单请求延迟。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/23.webp)

::: image-caption
对比推测解码架构下两种集群模式，Draft 代表草稿模型快速生成候选 token，Oracle 为大模型校验模块，业务分为 Oracle 密集、设计基准点、Draft 密集三类负载，上方弹性集群使用统一共享设备池，算力可动态分配给草稿或校验任务，适配各类负载；下方固定配比集群将 Draft 与 Oracle 硬件池硬性隔离，业务负载偏移时会出现硬件空闲或者 SLO 溢出，无法满足服务时延指标，以此体现 Jalapeño 的资源可灵活调度，相比硬件池固定拆分的方案，在推测解码业务中能获得更高的硬件利用率。
:::

一个可互换（fungible）的 fleet 能在延迟敏感型请求和吞吐导向型 batch 之间腾挪容量，而固定配比的拆分在流量结构变化时总会搁浅一部分硬件。此外，上下文长度会改变 attention 和 FFN 工作量的比例，任何固定的硬件配比只在设计点附近才高效。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/24.webp)

::: image-caption
Jalapeño 整机柜硬件布局图，展示 CPU 机柜与 ASIC 计算机柜两套机架结构，左侧机柜以 2U CPU 托盘为主，搭载 Turin x86 处理器与 1.5TB RDIMM 内存，搭配电源、OOB 管理交换机、ToR 以太网交换机；右侧为 ASIC 计算机柜，大量部署 1U 计算托盘，每个托盘内置 8 颗 Jalapeño AI 芯片，同时配备多组电源、全局与局部 Scale‑Up 交换机、管理交换机，两种机柜都包含机架加固件，整套系统通过 CPU 服务器配合自研 ASIC 芯片、多层高速交换机，共同组成 Jalapeño 大模型推理整机柜基础设施。
:::

同样的约束也适用于投机解码。Draft 模型必须以极低延迟把候选 token 喂给 verifier；把两者拆到专用池里，会把一个紧耦合的解码循环变成分布式协议，额外的通信和协调可能吃掉 draft 省下的延迟。**把两个模型放在同样的设备和低延迟互联上，才能保住让投机解码值得做的局部性**。

当然，在需求足够大、足够稳定、足够可预测的地方，分离仍然能赢，特别是当传统 GPU 需要很大的阶段专属 batch 才能达到好吞吐的时候。**但它不是免费的午餐**。

## 五、机架与组网：单柜 128 颗芯片，scale-up 扩展至 2,048 颗

### 5.1 机架组成：Katsu 主机托盘、Vindaloo ASIC 托盘与 Chana 交换托盘

> Jalapeño 系统在机架单元层面由一个 CPU 主机柜和一个 ASIC 机柜组成。主机柜容纳 16 个名为"Katsu"的主机 CPU 托盘，每个 Katsu 对应右侧 16 个 ASIC 托盘"Vindaloo"中的一个。每个主机装两颗 Turin 级 AMD EPYC CPU、1.5TB DRAM，每机架配 2 块 E1.S 和 2 块 M.2 SSD。每个托盘还配有 400G（2×200G）前端网络。每个 Katsu 托盘通过 8 根外置 PCIe DAC 线缆与对应的 Vindaloo 托盘相连，线缆在机架前方横向走线。系统级设计是与 Celestica 合作完成的。

ASIC 机柜由 16 个 Vindaloo 托盘和 8 个 scale-up 交换托盘（6 个本地 + 2 个全局）组成，交换托盘名为"Chana"。**每个 Vindaloo 托盘装 8 颗 Jalapeño ASIC，每机柜合计 128 颗**。ASIC 通过铜缆背板连接到各 Chana 交换托盘，方式类似 NVIDIA 的 Oberon。Scale-up 拓扑分两层：机架内 128 颗 ASIC 的本地域，以及最多连接 16 个机架、2,048 颗 ASIC 的全局域。带宽和拓扑的细节下文展开。

供电方面，边车（sidecar）主机柜的预留功率约 50kW（生产环境 31kW），ASIC 机柜 130kW，**两机柜系统合计约 160kW**，功耗上基本相当于一个双倍宽度的 GB300 机架。

![文章插图](/articles/2026-08-26-16-blackwell-openai-asic-jalape-45343c63/images/25.webp)

::: image-caption
Jalapeño 整机柜集群互联拓扑，展示多机架组网架构，单机架内部包含 AMD Turin CPU 托盘与 Jalapeño ASIC 计算托盘，通过铜背板与本地 Tomahawk‑6 交换机完成域内高速互联，CPU 与 ASIC 之间通过 AEC/ACC 链路通信，机架之间依靠 1.6T 高速链路跨柜互连，底层多台交换机组成 Rail 交换矩阵，实现跨机架的大规模算力互通，整套网络分为机架本地域与跨机架交换网络，为大量 Jalapeño 芯片提供高带宽互联，支撑大模型分布式推理运行。
:::

### 5.2 双层 scale-up 网络：本地域铜缆全互联，全局域铜光混合

> OpenAI 最多可以把 2,048 颗 Jalapeño XPU 连进同一张 scale-up 网络。每个机架有 8 个 Chana 交换托盘：中间 6 个用于本地域，各装一颗 102.4T Tomahawk 6 交换 ASIC；位于本地交换机顶部和底部的 2 个用于全局域，我们认为每个交换托盘可能由 2 颗 102.4T Tomahawk 6 组成，单托盘合计 204.8T。

在本地域，**128 颗 Jalapeño 每颗拥有 4.8Tb/s 的每 XPU 单向带宽**，以全互联（all-to-all）方式连接 6 颗 102.4Tb/s Tomahawk 6 ASIC。折算成连接器，每 XPU 有 48 对差分对（DP）公母连接器，整机架本地 scale-up 共使用 6,144 DP 的无源铜缆。

全局域方面，16 个机架共 2,048 颗 XPU 通过铜背板、204.8T TH6 电交换、1.6T 光模块和光路交换（OCS）的组合互联。**每颗 XPU 的全局链路单向带宽为 1.6Tb/s**，对应 XPU 与全局交换机之间背板的 16 对差分对连接器。每个全局交换托盘（含 2 颗 ASIC）的输出带宽在背板与前面板光口之间分配。

本地域与全局域合计，每机架背板连接器数量为每 XPU 64 DP，整机架共 8,192 DP 的无源铜缆。

全局域采用纯 rail 架构，共 8 条 rail。我们认为 OpenAI 通过每个机架安装的光路交换（OCS）来调度全局域的光链路。每颗 XPU 的 1.6Tb/s 全局带宽经铜背板到达全局交换托盘，再经前面板的 1.6T 光模块出交换托盘，进入无源光交换，最后出机架。:mdnice-bold-italic[这把 scale-up 的规模扩展到了 2,048 颗 XPU，即 16 个各 128 颗 XPU 的机架]。

### 5.3 部署计划：2027 年量产爬坡，为下一代模型预留选择权

**由于 scale-up 网络只占系统总成本的约 10%，这种灵活性为未来的 10–20 万亿参数模型或 200–400 万 token 上下文窗口买下了宝贵的选择权**。部署方面，OpenAI 正在与 neocloud 合作，并在明年 1 月前与数据中心伙伴一起收集可靠性数据，同时优化到货上架（dock-to-rack）的部署时间。

## 结语：首代即领先的底气来自全栈，悬念留给长上下文

> Jalapeño 的意义在于它把两件被认为很难的事同时做成了：第一代芯片就做到能效领先，从零搭的软件栈跑得比老牌厂商还快。

:mdnice-bold-italic[支撑这个结果的是一条清晰的逻辑链——电力成为数据中心最紧的约束，每瓦 token 数直接等于收入，于是 OpenAI 把架构、显存、互联、kernel 语言、模型团队全部攥在自己手里，围绕 perf/W 做端到端取舍]。简化内存层级、乱序核心、slice 直连 HBM 这些选择，都是背着兼容包袱的 GPU 难以复制的红利。

当然，悬念同样明显：目前的成绩全部来自 8k1k 单轮负载，AgentX 长上下文多轮验证还没交卷；投机解码、PD 分离这些业界标配优化一个都还没上。它们既是风险，也是余量——每实装一项，账面数字还会再涨一截。B0 stepping 已在产，2027 年量产爬坡，届时 CUDA 护城河是否真被撕开缺口，市场会给出答案。

参考资料

[1] InferenceX: *https://inferencex.semianalysis.com/*

[2] AgentX: *https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat*

[3] 我们在上月分析 NVIDIA 与 CoreWeave 发布性能声明的文章里讨论过: *https://newsletter.semianalysis.com/p/vera-rubin-nvl72-vs-gb200-nvl72-inference*

[4] Energy Model: *https://semianalysis.com/energy-model/*

[5] NVIDIA 与 CoreWeave 今年 7 月公布的 Vera Rubin MTP 结果，更远超 GB200 的 2025 年 MTP 结果: *https://www.coreweave.com/blog/nvidia-vera-rubin-nvl72-on-coreweave-10x-more-tokens-per-megawatt-than-blackwell*

[6] SemiAnalysis AI Cloud TCO 模型: *https://semianalysis.com/ai-cloud-tco-model/*

[7] AgentX: *https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat*

[8] AgentX: *https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat*

[9] 关于单位出货量与 ASP 的更多细节，见 SemiAnalysis Accelerator Model: *https://semianalysis.com/accelerator-hbm-model/*

[10] Linear Layouts: *https://arxiv.org/abs/2505.23819*

::: related-title
**相关推荐**
:::

::: related-links
- **[“这不只是 GPU 的问题，而是关乎所有层面”，Brendan Gregg 从 Intel 到 OpenAI 的性能工程新征程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897068&idx=1&sn=a17d384d80cb85f3e3b8e8e42a0f78d4&scene=21#wechat_redirect)**
- **[Anthropic 挖来谷歌 TPU 创始人，大模型公司集体转无晶圆厂芯片公司](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904745&idx=1&sn=45ff740c3c0c43f38b93fbf7f5afa4ac&scene=21#wechat_redirect)**
- **[从清华到 OpenAI：一位 AI Infra 工程师的成长与思考——对话翁家翌](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896307&idx=1&sn=61a45f13fdab2f25a3b0d70944f03d2e&scene=21#wechat_redirect)**

:::
