关键词：AsmEvo、GPU 内核优化、**汇编级优化**、:mdnice-bold-italic[功能等价验证]、**后编译优化**

> “GPU Kernel **一旦被编译为 AMDGPU 代码对象，尚存的优化空间便无法再通过源程序层面挖掘** —— 而在大量实际部署场景中，源码本身就完全不可获取。”这句话点出了当前 GPU 内核优化的核心痛点：:mdnice-bold-italic[编译完成的二进制代码就像一个黑盒，既看不到内部的优化空间，也很难在不破坏功能的前提下修改。]

高性能机器学习系统越来越依赖 GPU 内核，而这些内核的可编辑源码往往不可获取、由工具自动生成，或者与最终机器码距离过远，导致剩余的优化机会无法暴露。

::: block-1
**现有的 LLM 内核优化器与自动调优器主要在 CUDA、Triton、HIP 或张量程序源码层面操作**，并且需要依赖参考实现进行验证。:mdnice-highlight[而这项研究针对更严格的场景：优化已经编译完成的 AMDGPU 代码对象，此时部署的二进制本身就是唯一的行为参照。]
:::

![文章插图](/articles/2026-08-31-gpu-kernel-amd-asmevo-gpu-d9d9e096/images/01.webp)

- **AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification**
- Advanced Micro Devices, Inc. (AMD)、南方科技大学
- https://arxiv.org/pdf/2608.20711

::: related-title
**相关推荐**
:::

::: related-links
- **[一次编译，多平台运行！GPU 二进制文件兼容性在NVIDIA、AMD、Intel 和 Tenstorrent上的探索！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889722&idx=1&sn=ee787fcb0d896221430035a780fb5dff&scene=21#wechat_redirect)**
- **[ZLUDA：让 CUDA 程序在非英伟达 GPU 上“零修改”运行的开源兼容层——现状、原理与 2025 最新进展](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889691&idx=1&sn=ef6cd986f95c897029ba4b2981231cbe&scene=21#wechat_redirect)**
- **[汇编级 NVIDIA 与 AMD GPU 代码转换新 SOTA！跨架构转换方案 CASS，远超商业转换基线，数据代码全开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889091&idx=1&sn=a50fd9383f13b3c1ab7b6d3393369123&scene=21#wechat_redirect)**

:::

> 来自 AMD 和南方科技大学的研究团队提出了 :mdnice-bold-italic[AsmEvo，一套面向 AMD GPU 内核的智能体汇编级优化器。给定一个 AMDGPU 代码对象，AsmEvo 先重建出可重新汇编的表示，通过长视野智能体提出低层级修改，重建出保留 ABI 的优化对象，并且只有在与原始对象在相同启动条件下通过差分验证后，才会接受候选优化结果]。

AsmEvo 整合了代码对象恢复、元数据感知重建、性能分析引导的热点窗口编辑、正确性门控计时与保守原地补丁回退等技术。

![文章插图](/articles/2026-08-31-gpu-kernel-amd-asmevo-gpu-d9d9e096/images/02.webp)

::: image-caption
图 1：AsmEvo 架构概览：包含可信代码对象恢复、感知元数据且保留 ABI 的重建，以及以原始内核 $K_0$ 为基准的 “正确性优先于性能” 评估流程。模块化 LLM 驱动通过多起点搜索、经验证的组合优化和防停滞重定向来提出修改方案，只有通过验证且实现延迟降低的候选内核才会进入优化谱系。
:::

实验结果显示，在 MI308X 显卡上，AsmEvo 优化了 30 个选中的 KernelBench 内核中的 29 个，几何平均提速达到 1.35 倍，最高提速 3.88 倍。

![文章插图](/articles/2026-08-31-gpu-kernel-amd-asmevo-gpu-d9d9e096/images/03.webp)

::: image-caption
表 2：默认 AsmEvo 配置下的 KernelBench 性能表现。加速比均相对于原始编译代码对象测算，且仅在通过功能等价性验证后报告。97% 的优化成功率印证了汇编级优化的普遍可行性，两级测试的性能差异也体现出编译优化程度与底层优化空间的负相关规律。L1 单算子未经过深度融合优化，因此优化收益最高；L2 融合算子经过编译期流水线优化后空间收窄，但仍有九成以上内核可获得提升，说明成熟编译产物仍存在可挖掘的底层性能余量。
:::

在 MI300X 的生产工作负载上，:mdnice-bold-italic[它优化了所有评估的 AITer 二进制文件与 vLLM/SGLang Triton 汇编内核，分别达到 1.09 倍/1.31 倍和 1.18 倍/1.34 倍的几何平均/最高提速，同时完整保留功能等价性]。这项工作打破了编译完成即优化终点的固有认知，为部署后的 GPU 内核提供了新的性能提升路径。

![文章插图](/articles/2026-08-31-gpu-kernel-amd-asmevo-gpu-d9d9e096/images/04.webp)

::: image-caption
表 3：MI300X 上生产级二进制内核的单内核验证加速比。仅包含二进制级别的 AITer 结果与汇编级别的 Triton 结果；两类内核的家族几何平均加速比分别为：AITer 1.09 倍、Triton JIT 1.18 倍。Triton JIT HSACO 文件采集自 vLLM 与 SGLang 框架，采用捕获的生产调度负载进行评估。两类生产内核全部实现性能提升，说明汇编级优化对不同来源的编译产物都具有普适性。Triton JIT 内核的平均优化收益高于 AITer，核心原因是即时编译产物的底层调度优化程度低于 AMD 官方打磨的产品级代码，保留了更多优化空间。混合专家类算子的优化收益普遍更高，这类算子访存模式复杂、分支调度多，编译期难以覆盖所有底层调度场景，恰好是汇编级优化的优势场景；而融合注意力算子优化空间极小，说明这类核心算子的编译优化已经非常充分。
:::

## 本文目录

- 一、GPU 内核优化的“最后一公里”困境
- 二、AsmEvo 整体架构：严守正确性的 Agent 优化流水线
- 三、核心技术拆解：从二进制恢复到差分验证的全链路
- 四、Agent 搜索机制：长视野多起点的进化式优化
- 五、实验结果：基准与生产环境的双重验证
- 六、定位与意义：补全编译器优化的最后一块拼图
- 总结与展望

![文章插图](/articles/2026-08-31-gpu-kernel-amd-asmevo-gpu-d9d9e096/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[pyptx：用 Python 写 PTX，B200 1240 TFLOPS，H100 815 TFLOPS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900467&idx=1&sn=984c3c1b70cf02c5ceb1c5ba0b05f55b&scene=21#wechat_redirect)**

:::
