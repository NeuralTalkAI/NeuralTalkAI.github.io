关键词：**TIRx Harness**、:mdnice-bold-italic[智能体 GPU 编程]、**编译支撑框架**、GPU 内核优化、MLC 社区

> 我们打造了TIRx Harness[1]这一编译支撑框架，**它融合了精简稳定的编译基础、知识库、工具集与基准测试服务器**，可帮助智能体开发出正确、高效的 GPU 内核。在测试负载下，:mdnice-bold-italic[Kimi Delta 注意力（KDA）内核的前向传播几何平均提速较 FlashKDA 达 2.94 倍，反向传播较 Flash 线性注意力（FLA）达 6.84 倍。]

![文章插图](/articles/2026-09-30-mlc-tirx-harness-gpu-compiler-ba3fdfea/images/01.webp)

::: image-caption
面向 GPU 内核开发的 TIRx Harness 智能编译框架。左侧 Agent 工作流依托大模型，从 TIRx Harness 知识库读取内核与硬件资料，调用领域编译器分析内核、获取诊断信息；TIRx 底层提供稳定可扩展 IR，基准测试服务器在 GPU 上运行测试，校验正确性并使用 NCU、IKET 做性能剖析。智能体持续收集迭代数据，进入自优化环节，沉淀演化记录与优化内核，形成闭环，持续迭代产出更高质量的 GPU 计算内核。
:::

::: related-title
相关推荐
:::

::: related-links
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::
