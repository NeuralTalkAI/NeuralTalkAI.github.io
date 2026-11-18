关键词：FlashTile、**CUDA Tile IR**、:mdnice-bold-italic[TileLang]、NVRTC、**编译流水线**

::: block-1
FlashTile 整个设计哲学是：前端做精确的语义下降和轻量规范化，把重型硬件优化委托给 TileLang，同时在前后端之间做好"桥接优化"（block size 适配、shared memory 限制、PTX 后处理等），确保端到端的正确性和性能。
:::

- :mdnice-bold-italic[FlashTile is a CUDA Tile IR compiler that is compatible with NVIDIA's tileiras, targeting SM70 through SM121 NVIDIA GPUs.]
- 代码：https://github.com/serdes21/flashtile
- 4000 字，阅读 10 分钟，**播客 17 分钟**

相关推荐

- [GEMM/FA 平均性能超厂商库 1.03×/1.91×！面向空间数据流架构的端到端 Tile-Based 编译器 TL](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896903&idx=1&sn=78559ef90da4c8638475faac4265f65a&scene=21#wechat_redirect)
- [性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)
- [从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)

> FlashTile 是一个使用 Rust 语言实现的、用于 NVIDIA GPU 的 **CUDA Tile IR 编译器**。

其核心目标是将由 cutile-python 前端生成的 `.cutile` 字节码文件，编译为可在 GPU 上直接执行的 `.cubin` 二进制文件，:mdnice-bold-italic[旨在作为 NVIDIA 私有编译器 `tileiras` 的兼容替代品。]

![文章插图](/articles/2026-02-08-nvidia-tileiras-flashtile-cuda-tile-f6efbe8c/images/01.webp)

::: image-caption
FlashTile 的正确性验证矩阵，展示了不同 NVIDIA 显卡在 Linux 与 Windows 系统下的内核测试结果。它以通过率、通过 / 跳过 / 失败的数量直观呈现，其中 RTX 5060 在双系统下均取得 91.9% 的最高通过率，而更早的 RTX 3060 在 Linux 下通过率为 82.4%，清晰反映出硬件架构与系统环境对验证结果的影响。
:::

::: block-1
本文将从其整体架构、核心编译流水线、关键技术实现及优化策略等方面，进行解读。最后一部分会结合 README 讲解其安装、使用等！
:::
