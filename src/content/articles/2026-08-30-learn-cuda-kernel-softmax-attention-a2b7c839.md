关键词：CUDA、**PyTorch扩展**、矩阵乘、**TensorCore**、:mdnice-bold-italic[FlashAttention]

> GPU 编程最劝退人的地方是“看不到因果”：改了一个访存方式，性能变了；换了一个 tile 形状，寄存器爆了；用了 Tensor Core，结果还不如 cuBLAS。

![文章插图](/articles/2026-08-30-learn-cuda-kernel-softmax-attention-a2b7c839/images/01.webp)

- Learn CUDA with PyTorch
- https://github.com/gau-nernst/learn-cuda

::: related-title
**相关推荐**
:::

::: related-links
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**
- **[MLC.ai 开源《Modern GPU Programming For MLSys》，从 GEMM 到 FlashAttention：Blackwell 时代的 GPU 编程方法论](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902890&idx=1&sn=3e4f0a13f3c75cc78f033969d0d53e0c&scene=21#wechat_redirect)**
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**

:::

> `gau-nernst/learn-cuda` 的可贵之处，在于它**把这些抽象问题拆成一组能直接运行、能对照 PyTorch、能用 Nsight Compute 继续追踪的实验。**

从 `01_vector_addition/add.cu` 的线程索引，到 `02b_matmul_sm80` 的 `ldmatrix`、`mma`、`cp.async`，再到 `07_attention` 中:mdnice-bold-italic[在线 softmax 与寄存器累积的 FlashAttention 路线，这个仓库像一条“从 CUDA 入门到 LLM 推理内核”的实战栈道。]

![文章插图](/articles/2026-08-30-learn-cuda-kernel-softmax-attention-a2b7c839/images/02.webp)

::: block-1
读完代码，你会发现:mdnice-highlight[性能优化并非只靠神秘参数，更多时候是在数据搬运、计算分块、同步边界和硬件指令之间做清晰取舍。]
:::

## 本文目录

- 一、快速上手：先把 Kernel 跑起来
- 二、项目全景：一条从入门 Kernel 到 LLM Megakernel 的路线
- 三、矩阵乘主线：性能优化从访存层级开始
- 四、Tensor Core 主线：从 C++ 循环进入 PTX 指令世界
- 五、Reduction 与 Softmax：从求和到 Attention 的前置训练
- 六、Attention：FlashAttention 的核心在“边算边归一化”
- 七、低比特与 LLM：项目正在走向推理内核
- 八、为什么这个仓库值得读：它保留了优化过程
- 结语：从代码实验到硬件直觉

![文章插图](/articles/2026-08-30-learn-cuda-kernel-softmax-attention-a2b7c839/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[陈天奇团队提出 CAKE：编译器与智能体协同进化，可进化 IR 让 KDA 提速 2.05 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904925&idx=1&sn=ab085dc1e9e6c73917c8853a398775d7&scene=21#wechat_redirect)**
- **[暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)**

:::
