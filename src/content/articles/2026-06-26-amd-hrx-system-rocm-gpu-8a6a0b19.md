关键词：HRX、**HIP 兼容层**、:mdnice-bold-italic[IREE 运行时]、**时间线信号量**、透明拦截

> 当你在 AMD GPU 上跑一个 PyTorch 模型，链接的其实是 `/opt/rocm/lib/libamdhip64.so`——一个十几年演进、夹带历史包袱的 HIP 运行时。**它能跑，但延迟不够低、对 NPU/CPU 异构不够友好、对调试也不够透明。**

![文章插图](/articles/2026-06-26-amd-hrx-system-rocm-gpu-8a6a0b19/images/01.webp)

- :mdnice-bold-italic[HRX System：HRX 系统由一组精简运行时组件构成，针对当前 ROCm 内置的 HIP 提供一套替代实现方案。该系统搭建通用底层支撑，可与 AMD 旗下 GPU、NPU 及 CPU 产品实现低延迟、高性能的适配集成。]
- https://github.com/ROCm/hrx-system

::: related-title
**相关推荐**
:::

::: related-links
- **[用 TileLang 在 AMD Radeon 上榨干 GPU 性能：RDNA4 GEMM 118.5 TFLOPS 持平 rocBLAS，Dequant达 93 TFLOPS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901949&idx=1&sn=4520024a563e4c5821d9dcf98d83c8ac&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**

:::

> ROCm 团队在 GitHub 低调地开源了一个名为 `ROCm/hrx-system`[1] 的项目，README 第一行就用 `[!CAUTION]` 标注它"**不是 ROCm 官方栈的一部分** "，却又毫不掩饰地表示:mdnice-bold-italic[要"为 AMD 的 GPU、NPU、CPU 提供一个低延迟、高性能的公共底座] "。**HRX——HIP Runtime Extended——究竟是一次实验性的另起炉灶，还是 ROCm 下一代运行时的雏形？**

HRX 是 ROCm 团队基于 IREE 重构的极简 HIP 替代运行时，通过统一时间线信号量、流式调度与 dlopen 透明拦截层，让现有 HIP 应用免重编即切换后端，兼顾低延迟与跨架构可观测性，重塑 AMD GPU 软件栈。

::: block-1
本文将从公开仓库切入，沿着 `libhrx/include/hrx_runtime.h` 的 C ABI、`libhrx/src/libhrx/runtime.c` 的 IREE 集成，再到 `libhrx/src/passthrough/` 的 `LD_PRELOAD` 拦截层，**拆解它如何在一个 `libhrx.so` 里同时塞下"新运行时"和"老 HIP 兼容层"两层皮。**
:::

## 本文目录

- 快速上手、使用
- 一、架构总览与设计哲学
- 二、核心 C ABI：hrx_runtime.h 的极简美学
- 三、运行时实现剖析：runtime.c 的核心调度
- 四、HIP 兼容层与透明拦截：libhrx 的另一面
- 五、测试体系：从 CTS 到 YAML 执行清单
- HRX 的意义：把 HIP 重塑成一组可替换的层

![文章插图](/articles/2026-06-26-amd-hrx-system-rocm-gpu-8a6a0b19/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[用 TileLang 在 AMD Radeon 上榨干 GPU 性能：RDNA4 GEMM 118.5 TFLOPS 持平 rocBLAS，Dequant达 93 TFLOPS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901949&idx=1&sn=4520024a563e4c5821d9dcf98d83c8ac&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**

:::
