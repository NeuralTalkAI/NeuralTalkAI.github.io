关键词：**张量核**、:mdnice-bold-italic[精度模拟]、内核融合、**矩阵乘法**、:mdnice-bold-italic[Ozaki 方案]、高性能计算

> **现代 GPU 将越来越多的硅片预算投入低精度矩阵乘法单元**，正在持续拉大科学计算工作负载的精度-吞吐差距。这是当前不可否认的行业背景，也是:mdnice-bold-italic[整个高精度计算领域共同面对的结构性挑战。]

从 Hopper 到 Blackwell，NVIDIA GPU 的 **INT8 与 FP64 峰值吞吐比从 30 倍跃升至 112 倍，一代架构之间差距近乎翻倍**。大模型的市场红利推动硬件持续向低精度倾斜，:mdnice-bold-italic[而计算化学、气候模拟、有限元分析等科学计算领域无法牺牲精度，正面临“硬件算力暴涨、可用算力停滞”的尴尬处境。]

::: block-1
Ozaki 提出的两套精度模拟方案，为破解这一困境提供了数学基础：通过:mdnice-bold-italic[将高精度运算拆解为多次低精度整数矩阵乘法，再重组还原高精度结果，理论上可以借用海量低精度算力服务高精度场景。但现有实现普遍受困于全局内存往返开销，中间结果反复落盘，数据移动取代计算成为核心瓶颈，张量核的理论算力始终无法充分释放。]
:::

![文章插图](/articles/2026-08-10-int8-emugemm-blackwell-gemm-4-375dda89/images/01.webp)

- **EmuGEMM: Fused Tensor Core Kernels for Precision Emulation in Matrix Multiplication**
- https://arxiv.org/pdf/2606.25453
- **代码开源**：https://zenodo.org/doi/10.5281/zenodo.19712923

::: related-title
**相关推荐**
:::

::: related-links
- **[FlashNormAttention 融合注意力与归一化到 GEMM，最高可掩盖 90%归一化时延](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903614&idx=1&sn=d4c7a5e4345d25b1925979e0793d531a&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[1979 TFLOPS巅峰表现！ETH实现Tile-Based PE加速器的自动化GEMM部署，碾压英伟达GH200专家优化库！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895104&idx=1&sn=ca0706d1203faf880462921ce62196ed&scene=21#wechat_redirect)**

:::

> ETH Zurich 团队提出的 EmuGEMM，核心洞察在于：**内核融合是精度模拟方案释放全部潜力的必要前提**。:mdnice-bold-italic[针对 Ozaki 两套方案的不同计算结构，研究团队分别设计了融合内核，将中间数据全程约束在片上存储中]，彻底消除冗余的全局内存流量。

底层实现基于 DeepGEMM，实测结果显示：

- **基于 Scheme I 的 EmuGEMM-I 在 Hopper 上** 可维持 1639 Top/s 的 INT8 吞吐，:mdnice-bold-italic[达到硬件峰值的 83%，Blackwell 上达到 3654 Top/s，占峰值的 81%；] 大矩阵场景下，同等精度时吞吐分别超过 cuBLAS TF32 的 1.4 倍与 1.7 倍。
- **基于 Scheme II 的 EmuGEMM-II 扩展至复数运算**，:mdnice-bold-italic[性能分别超出 cuBLAS ZGEMM 的 2.3 倍与 5.5 倍]；FP64 精度下，在 Blackwell 上达到原生 FP64 性能的 4.6 倍。

## 本文目录

- 一、高精度计算的算力困境：GPU 正在“偏科”
- 二、Ozaki 精度模拟：用低精度单元拼出高精度结果
- 三、EmuGEMM-I：Scheme I 的全融合内核设计
- 四、EmuGEMM-II：Scheme II 与复数运算的融合优化
- 五、实测性能：跨架构的性能与精度全面对比
- 六、权衡与边界：这项工作的适用场景与局限
- 七、总结与展望

![文章插图](/articles/2026-08-10-int8-emugemm-blackwell-gemm-4-375dda89/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[评估 YOLOv11/GEMM 等应用在 Apple Silicon M1 与 M4 Pro 神经引擎上的潜力：GEMM性能达3.8 TFLOPS，能效超GPU 80%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897336&idx=1&sn=faef0c904b6cd45e50a35faf4ecf2372&scene=21#wechat_redirect)**
- **[SOLE：Softmax/LayerNorm软硬件协同优化，36倍加速与2.8+倍面效和能效比的低精度Transformer架构](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893805&idx=1&sn=b82015bdc4d1d0bee906f5471e88de62&scene=21#wechat_redirect)**
- **[微秒级决策，近最优性能！AMD 提出 tritonBLAS：告别耗时参数搜索，用“数学”算出最快矩阵乘法](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896258&idx=1&sn=fcd4a9bd2843b8eac7ebd0f84092336a&scene=21#wechat_redirect)**
- **[打破库依赖与 93% 峰值效率！Intel 提出 MLIR 驱动的编译器自动生成 NanoKernel 实现高性能矩阵乘法内核！性能媲美手工优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895855&idx=1&sn=23bb841f0a281cb91e82b9e37fbddd8c&scene=21#wechat_redirect)**

:::
