关键词：**ByteRobust**、**LLM 训练**、:mdnice-bold-italic[大规模 GPU 集群]、故障容错、:mdnice-bold-italic[有效训练时间比（ETTR）]、快速恢复

> ByteRobust 已部署在包含超过 20 万块 GPU 的生产级 GPU 平台上，在 9600 块 GPU 上进行的为期三个月的训练任务中，有效训练时间占比（ETTR）达到 97%，刷新了训练稳健性的当前最佳水平。

![文章插图](/articles/2025-10-29-20-gpu-llm-byterobust-97-d60a7f73/images/01.webp)

- Robust LLM Training Infrastructure at ByteDance
- https://arxiv.org/pdf/2509.16293v3
- 20000 字，阅读 80 分钟，:mdnice-bold-italic[播客 21 分钟（豆包说：“这是我们公司的工作”:）]

相关推荐

- :mdnice-bold-italic[[LLM 推理系统综述：从内存管理、系统架构到严峻挑战](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892359&idx=1&sn=8c14f38e11cc07e5daa5c298e1f99c22&scene=21#wechat_redirect)]
- :mdnice-bold-italic[[PerfTracker：万卡 GPU 集群超大规模训练性能问题定位利器！100万GPU 定位仅 3 分钟，7 分钟端到端性能诊断](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888997&idx=1&sn=68ef138e5f39ad09aa0d548bf61f38d2&scene=21#wechat_redirect)]
- :mdnice-bold-italic[[节省 GPU 51% Capacity 和 26% 能耗，CMU, Meta提出LithOS：迈向高效 GPU 机器学习操作系统](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888297&idx=1&sn=c87e81667f2e91731e2628a5abb53923&scene=21#wechat_redirect)]

> 随着大型语言模型（LLM）训练规模扩展至数万 GPU，CUDA 错误、NaN 值、任务挂起等故障频发，现有依赖日志分析或集群压力测试的故障处理方案，**常导致数小时至数天的恢复耗时，严重制约有效训练时间比（ETTR）**。为此，字节跳动提出**ByteRobust** ——一款专为 LLM 稳健训练设计的**大规模 GPU 基础设施管理系统**，旨在:mdnice-bold-italic[通过高效故障诊断、容错与快速恢复，实现持续高效的训练。]

ByteRobust 采用“控制平面+数据平面”架构：

::: block-1
- 控制平面含 Robust Controller（统筹故障缓解策略）与 Runtime Analyzer（聚合栈轨迹定位隐式故障）；
- 数据平面通过 Robust Agent 集成监控、诊断、checkpoint 管理等模块，提供实时可观测性与快速回滚能力。
:::

![文章插图](/articles/2025-10-29-20-gpu-llm-byterobust-97-d60a7f73/images/02.webp)

::: image-caption
图4. ByteRobust的架构
:::

其核心技术包括三方面：

::: block-1
- 一是**自动容错框架**，结合轻量级实时检查（秒级监测网络、GPU、主机状态）、分层停机诊断（针对性测试定位故障）与双阶段重放（固定 TP/PP 规模、调整 DP 分组定位静默数据损坏（SDC））；
- 二是**数据驱动过驱逐**，通过解析训练进程树、聚合栈轨迹，识别异常进程组并隔离可疑机器，解决任务挂起、MFU 下降等隐式故障定位难题；
- 三是**受控快速恢复**，采用原地热更新避免环境重部署、热备用机器预验证以减少调度等待、跨并行组 checkpoint 备份消除远程存储依赖，大幅降低恢复开销。
:::

ByteRobust 已部署于超 20 万台 GPU 的生产集群，实验显示：

- 9600 GPU 上的 3 个月训练任务 ETTR 达 97%；
- 热备用与热更新机制分别实现 10.87 倍、11.04 倍恢复加速；
- 每步 checkpoint 开销低于 0.9%。

> 该系统有效解决了大规模 LLM 训练中故障频发、隐式故障难定位、恢复开销大等问题，为超大规模 LLM 高效持续训练提供关键支撑。

![文章插图](/articles/2025-10-29-20-gpu-llm-byterobust-97-d60a7f73/images/03.webp)
