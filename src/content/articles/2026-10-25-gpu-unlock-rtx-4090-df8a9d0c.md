关键词：**GPU Unlock**、:mdnice-bold-italic[RTX 4090、RTX 3090]、BF16、**大模型推理**

> 能用上 AI Agent 后，REA 等逆向项目自然受到关注，许多人开始逆向低成本或早期 GPU，尝试提升 AI 性能。

最近几天，几个非官方 GPU Unlock 项目集中出现。**实现在 NVIDIA 驱动接管显卡前修改 SM speed-select，让部分消费级显卡提高 FP16、BF16、TF32 的 Tensor Core 运算速率**。

::: block-1
已经公开的结果里，:mdnice-highlight[RTX 3090 Ti 的 BF16 GEMM 提升 79.9%、TF32 提升 94.0%；RTX 4090 的 FP16/FP32 指令吞吐最高测到 2.002 倍]；4070 Ti SUPER 在 240 W 下也把 FP16、BF16、TF32 提到约 2 倍。
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/01.webp)

::: image-caption
图 1：支持矩阵按公开实测数量和复现范围排序。Linux 系统的 NVIDIA RTX 30 系显卡用户报告最多；Ada 同卡对照正在补齐；RTX 4090 的公开实测只覆盖指定设备 ID；Windows UEFI 组合绑定具体主板和 BIOS。RTX 5060 Ti 暂无同类项目。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892934&idx=1&sn=5ec357f95854bc9b6ebaa827012de874&scene=21#wechat_redirect)**

:::
