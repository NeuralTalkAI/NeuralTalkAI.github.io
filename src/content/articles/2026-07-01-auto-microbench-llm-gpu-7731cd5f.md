关键词：**GPU 微基准**、LLM 智能体、**SASS 验证**、:mdnice-bold-italic[Nsight Compute]、**模拟器校准**

> GPU 模拟器最怕什么？不是没有模型，而是模型里的硬件参数“看似合理，却悄悄错了”。一个 FMA 吞吐、一次 L2 命中延迟、一条 Tensor Core 指令的真实执行路径，只要偏一点，仿真结果就会在复杂应用中层层放大。

![文章插图](/articles/2026-07-01-auto-microbench-llm-gpu-7731cd5f/images/01.webp)

- Code Implementation for OSCAR 2026 Workshop: **Toward Sustainable GPU Modeling: Autonomous Micro-benchmarking with LLM Agents**
- https://github.com/abdelfattah-lab/Auto-Microbench

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::

> 传统做法依赖专家手写微基准、人工读 SASS、查白皮书、调 Nsight Compute 计数器，既慢又难复现。

Auto-MicroBench 的野心正是在这里：它绝非简单让大模型生成 CUDA，反而:mdnice-bold-italic[给 LLM 套上一套“实验室制度”——调度器决定测什么，知识库告诉怎么测，固定 harness 负责编译、验证、统计与归档，最终把一次次试错沉淀成 Accel-Sim 与 GPGPU-Sim 可用的硬件配置碎片。]

## 本文目录

- 一、快速上手：用最短路径跑起自动微基准
- 二、项目到底解决了什么问题
- 三、整体架构：调度器、样品、仪器与报告器
- 四、关键闭环：LLM 只负责改样品，harness 负责当裁判
- 五、为什么 SASS 验证是项目的灵魂
- 六、测量模板：把微基准经验固化成可复制骨架
- 七、统计与状态机：为什么一个数字不够
- 八、知识库与模板：给 LLM 装上“实验记忆”
- 九、结果如何变成模拟器配置
- 十、这个项目最值得学习的设计思想
- 十一、局限与未来空间
- 结语：Auto-MicroBench 的真正价值是“可信自动化”

![文章插图](/articles/2026-07-01-auto-microbench-llm-gpu-7731cd5f/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::
