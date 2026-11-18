关键词：:mdnice-bold-italic[Blackwell]、**TIRx**、GEMM、**FlashAttention**、异步协作

> 如果说过去的 GPU 优化还像“给流水线加速”，那么 Blackwell 时代的 GPU 编程更像“给一支交响乐团编排乐谱”。TMA 负责搬运，Tensor Core 负责计算，TMEM 负责中转，mbarrier 负责协调，warpgroup 和 CTA 负责分工，而 TIRx 则把这一切翻译成可运行、可验证、可推导的代码。

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/01.webp)

- **Modern GPU Programming For MLSys**
- 本书循序渐进讲解现代GPU内核编程：认识GPU硬件→掌握GPU编程方法→编写顶尖内核程序。全书以 Blackwell GPU为核心研究对象，详细介绍其存储层次、张量存储、张量核心、异步数据传输引擎、线程束组与计算簇等关键组件。书中依托TIRx（新一代张量中间表示语言）展开讲解，这是一门可在中间表示层级编写GPU内核的Python领域专用语言。
- https://github.com/mlc-ai/modern-gpu-programming-for-mlsys
- 📖 Read it online: https://mlc.ai/modern-gpu-programming-for-mlsys/

::: related-title
**相关推荐**
:::

::: related-links
- **[暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)**
- **[NVIDIA 开源 GPU 编程教学仓库 accelerated-computing-hub，从单线程 kernel 到协作归约](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900741&idx=1&sn=8a88431b7a8d1d46aa3b03f9889bc151&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::

> `modern-gpu-programming-for-mlsys` 并不是一本只讲技巧的书，它更像一张地图：**它先告诉你现代 GPU 为什么变复杂，再带你一步步走进 GEMM、FlashAttention、布局系统和异步调度的核心腹地**。读懂它，:mdnice-bold-italic[意味着你不仅能写出更快的 kernel，还能真正理解“快”是如何被系统性设计出来的。]

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/02.webp)

::: image-caption
这本《Modern GPU Programming For MLSys》书中有大量的交互式例子，比方上图。这张图展示 GPU 内存重排单元 Swizzle Atoms 的对比效果，swizzle atom 是内存中执行地址重排的连续存储块，当前采用 128B 格式、f16 数据，基础原子块为 8 行 8 列，重排规则为重排列号 = 逻辑列号异或行号。交互式的教程，让你能更好地理解很多看似晦涩复杂的概念。
:::

## 本文目录

::: related-links
- 一、这不是“GPU 技巧合集”，而是一套现代算子工程学
- 二、仓库的核心目标：把现代 GPU 的“复杂协作”讲清楚
- 三、Part I：先把“硬件世界”讲明白，才谈得上优化
- 四、Part II：TIRx 的价值，不只是“能写”，而是“能表达硬件语义”
- 五、Part III：GEMM 是试金石，也是方法论的落地点
- 六、Part IV：FlashAttention 是“把所有前面学到的东西一次性用上”
- 七、这本书最难得的地方：它把“工程细节”讲成了“可迁移的方法论”
- 八、为什么这本书值得 AI 系统工程师认真读
- 结语：现代 GPU 编程的终点，不是更快，而是更懂

:::

![文章插图](/articles/2026-06-24-mlc-ai-modern-gpu-programming-15602cce/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)**
- **[NVIDIA 开源 GPU 编程教学仓库 accelerated-computing-hub，从单线程 kernel 到协作归约](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900741&idx=1&sn=8a88431b7a8d1d46aa3b03f9889bc151&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::
