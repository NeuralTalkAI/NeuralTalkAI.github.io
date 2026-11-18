关键词：**NCCL 扩展**、专家并行、:mdnice-bold-italic[零拷贝重分片]、GPU 发起通信、**MoE 系统**

> 大模型系统的瓶颈，越来越多地藏在“看似普通”的数据搬运里。

MoE 模型每一步都要把 token 送到对应专家，再把专家输出按原顺序合回来；强化学习与解耦式推理系统又常常需要把训练侧的权重布局，转换成推理侧的张量并行、专家并行或流水并行布局。传统做法依赖框架层调度、CPU 参与、临时缓冲区与多次通信拼接，复杂度和延迟都会被放大。

![文章插图](/articles/2026-07-21-nvidia-nccl-extensions-moe-mesh-772bc239/images/01.webp)

- **Communication patterns for AI, built on top of NCCL device and host APIs**
- https://github.com/NVIDIA/nccl-extensions

::: related-title
**相关推荐**
:::

::: related-links
- **[打破 MoE 通信“方言”乱局！NVIDIA 提出 NCCL EP，用一套 API 实现 &lt;10% 性能差距，统一训练与推理](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899046&idx=1&sn=319df3dc40e4c5fd85c245319ca62009&scene=21#wechat_redirect)**
- **[突破 GPU 通信瓶颈：NCCL 协议创新与 25-95% 带宽利用率的量化研究](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890497&idx=1&sn=54b6b60452029a0a580b02184abfa308&scene=21#wechat_redirect)**
- **[NVIDIA CCCL (CUDA Core Compute Libraries) 核心架构深度分析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899677&idx=1&sn=7cbf73b7430dba1208d735f80a832ca1&scene=21#wechat_redirect)**

:::

> NVIDIA/nccl-extensions 的意义在：**它把这些 AI 工作负载中反复出现的通信模式，沉到 NCCL 的设备 API、GIN、LSA 和 Window 机制之上**，让 GPU 直接参与编排、搬运与同步。

本文将沿着代码结构拆解这个仓库的两条主线：面向 MoE token shuffle 的`nccl_ep`，以及面向训练到推理权重 rollout 的`nccl_m2n`。

## 本文目录

- 一、快速上手：先把库编出来，再跑通最小路径
- 二、项目总览：两个 AI 通信痛点，两套 NCCL 扩展原语
- 三、NCCL EP：把 MoE 的 token 旅行拆成 dispatch 与 combine
- 四、EP 的 Host 侧工程：严密 API、环境变量与 JIT 专用化
- 五、NCCL M2N：把训练侧 Mesh 重分片到推理侧 Mesh
- 六、从计划到 Kernel：RING、DIRECT 与 Window 缓存
- 七、正确性、可调参性与可演进 API
- 八、读懂这个仓库的核心：让 GPU 通信原语贴近 AI 工作负载

![文章插图](/articles/2026-07-21-nvidia-nccl-extensions-moe-mesh-772bc239/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[打破 MoE 通信“方言”乱局！NVIDIA 提出 NCCL EP，用一套 API 实现 &lt;10% 性能差距，统一训练与推理](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899046&idx=1&sn=319df3dc40e4c5fd85c245319ca62009&scene=21#wechat_redirect)**
- **[突破 GPU 通信瓶颈：NCCL 协议创新与 25-95% 带宽利用率的量化研究](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890497&idx=1&sn=54b6b60452029a0a580b02184abfa308&scene=21#wechat_redirect)**
- **[NVIDIA CCCL (CUDA Core Compute Libraries) 核心架构深度分析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899677&idx=1&sn=7cbf73b7430dba1208d735f80a832ca1&scene=21#wechat_redirect)**

:::
