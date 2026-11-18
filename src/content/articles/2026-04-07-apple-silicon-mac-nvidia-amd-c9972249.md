关键词：AI、**Apple**、GPU、硬件、**推理**、开源

> 专为 AI 设计，游戏玩家再次失望

![文章插图](/articles/2026-04-07-apple-silicon-mac-nvidia-amd-c9972249/images/01.webp)

- **Apple approves drivers that let AMD and Nvidia eGPUs run on Mac — software designed for AI, though, and not built for gaming**
- https://www.tomshardware.com/pc-components/gpu-drivers/apple-approves-drivers-that-let-amd-and-nvidia-egpus-run-on-mac-software-designed-for-ai-though-and-not-built-for-gaming

> 2026 年 3 月 31 日，Tiny Corp 创始人 George Hotz 宣布：苹果正式批准了 TinyGPU 驱动扩展。这意味着 Apple Silicon Mac 用户现在可以通过 Thunderbolt 或 USB4 接口，直接运行外接的 NVIDIA 和 AMD 显卡，**无需关闭系统完整性保护（SIP）**。

经过一年多的努力，Tiny Corp 团队开发了自定义的用户态 GPU 驱动，并最终获得了苹果的官方认可。这是自苹果转向自研芯片以来，macOS 首次通过官方机制支持第三方独立显卡。

![文章插图](/articles/2026-04-07-apple-silicon-mac-nvidia-amd-c9972249/images/02.webp)

## 为什么这件事值得关注？

2020 年底，苹果推出自研 M 系列芯片，同时也**放弃了所有 eGPU 支持** ——这项功能原本在 Intel Mac 上配合 AMD 显卡使用。进入 Apple Silicon 时代后，NVIDIA 从未获得官方 macOS 驱动支持，AMD 的驱动也被彻底中断。

对于运行大型 AI 模型的研究人员和开发者来说，这意味着：苹果硬件虽然性能不俗，却只能依赖集成的 GPU 和神经网络引擎（Neural Engine）。TinyGPU 彻底改变了这一局面。

## TinyGPU 是什么？

TinyGPU 是一个基于 **tinygrad 神经网络框架** 的 macOS 应用。它安装一个经过苹果批准的 **DriverKit 扩展**，使得 Mac 能够通过 Thunderbolt 或 USB4 端口，与外部 AMD（RDNA3+）和 NVIDIA（Ampere+）显卡通信。

- **无需内核扩展**
- **无需绕过 SIP**
- 只需在“系统设置”中打开一个驱动开关即可

> 💡 Tiny Corp 在 X 上调侃：“现在安装驱动简单到连 Qwen（一个 AI 模型）都能完成——装完它还能跑自己。”

## 从零到苹果批准：一段不简单的历程

- **2025 年 5 月**：团队首次在 Apple Silicon Mac 上通过 USB3 驱动 AMD GPU，使用重刷固件的 ASM2464PD 适配器（ADT-UT3G）。当时需要自定义用户态驱动和修改固件。
- **2025 年 10 月**：在 MacBook Pro M3 Max 上通过 USB4 成功运行 NVIDIA RTX GPU，这是 ARM 架构 Mac 首次与 NVIDIA 独立显卡配对。不过，当时仍需禁用 SIP。
- **2026 年 3 月**：苹果正式批准驱动，整个流程被整合为干净的一键安装。

![文章插图](/articles/2026-04-07-apple-silicon-mac-nvidia-amd-c9972249/images/03.webp)

## 如何使用？

1. 通过 Thunderbolt 或 USB4 连接外置显卡扩展坞
2. 运行 TinyGPU 安装脚本（自动下载 TinyGPU.app）
3. macOS 提示安装驱动扩展 → 打开“系统设置”并开启 TinyGPU
4. 安装 GPU 编译器：
5. 运行推理命令：`DEV={AMD|NV} python3 tinygrad/apps/11m.py`

系统要求：macOS 12.1（Monterey）或更高版本、Thunderbolt 或 USB4 端口、AMD RDNA3+ 或 NVIDIA Ampere+ 显卡。

## 性能表现

> 根据 Tiny Corp 公布的基准测试数据：

- **设备**：Mac mini（M4 芯片）+ Radeon RX 7900 XTX（通过 Thunderbolt/USB4 连接）
- **模型**：Qwen 3.5 27B（270 亿参数的大语言模型）
- **推理速度**：**18.5 tokens/s**

虽然无法与原生 PCIe 带宽相媲美，但对于交互式推理来说已经非常实用，且性能远超苹果集成 GPU 在同等规模模型上的表现。

## ⚠️ 别高兴太早：不是给游戏玩家准备的

Tom's Hardware 在报道中明确指出：**这个驱动是为 AI 大语言模型设计的，不是为游戏打造的**。

尽管苹果近年来多次表态“认真对待游戏”，但这次驱动的官方批准仍然绕开了游戏场景。评论区里的玩家们一如既往地表达了失望：

> “苹果一直告诉我们‘这次请相信我们’，然后新驱动又说‘不关心游戏’。这种信任的裂痕从未消失。” —— ezst036

如果你期待用外接 RTX 4090 在 Mac 上流畅运行《赛博朋克 2077》，目前来看并不现实。TinyGPU 的目标用户是 AI 研究者和开发者，他们现在可以用 Apple Silicon 机器搭配已有的 NVIDIA/AMD 显卡进行本地推理，从而避免昂贵的云计算成本。

## 关于 Tiny Corp

> Tiny Corp 是 **tinybox** AI 加速器的制造商。该公司曾因驱动问题与 AMD 发生争执，最终 AMD CEO 苏姿丰亲自介入才得以解决。目前 Tiny Corp 在售的产品包括：

- **red v2**：四块 AMD 9070XT，售价 12,000 美元
- **green v2**：四块 RTX Pro 6000 Blackwell，售价 65,000 美元
- **exabox** （计划 2027 年）：720 块 RDNA5 ATO XL GPU，约 1 exaflop 算力，售价约 1000 万美元

有趣的是，由于 AI 智能体（如 OpenClaw）的火爆，高配 Mac 一度供不应求。苹果甚至取消了 Mac Studio 的 512GB 统一内存选项，并将 256GB 版本价格提高了 400 美元。

## 这对 Mac 上的 AI 意味着什么？

TinyGPU 填补了困扰 Mac AI 社区多年的空白。结合 tinygrad 框架（已支持多种后端的训练和推理），Mac 现在可以成为一个真正异构 AI 开发环境中的有效节点。

> 更重要的是，**苹果愿意通过 DriverKit 框架允许第三方 GPU 计算驱动** ——即便是对 NVIDIA 硬件。这是否会为 tinygrad 运行时之外更广泛的 GPU 支持打开大门，还有待观察。但无论如何，这是一个实实在在的进步。
