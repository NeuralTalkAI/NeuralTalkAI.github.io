技术关键词：移动神经超采样、参数预测网络、Vulkan ML扩展、10 TOPs/Watt能效、27 GOPs运算量、 temporal super sampling、UNet架构

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/01.png)

::: image-captionhttps://developer.arm.com/mobile-graphics-and-gaming/neural-graphics
:::

- Neural Graphics Development Kit: Bring AI-Enhanced Graphics to Mobile & Neural Graphics Development Kit
- https://developer.arm.com/mobile-graphics-and-gaming/neural-graphics
- 本文 1832 字，阅读需 5 分钟，播客 10 分钟如下

> 在移动图形技术迈向更高画质与能效平衡的关键节点，Arm 推出的 Neural Technology 重新定义了 GPU 与 AI 的关系。

开发者可在更低 GPU 开销下实现桌面级画质——只需借助神经超分技术。

交流加群请在 NeuralTalk 公众号后台回复：交个朋友

## 一、核心定位：让 AI 成为 GPU 的"原生基因"

AI —— 不再是"外挂式"支持，而是将**专用神经网络处理单元（NPU）直接集成进 GPU**，形成统一的 AI 渲染管线。

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/02.png)

::: image-caption
Arm表示，神经加速器将在未来的Arm GPU中发挥关键作用
:::

其**核心目标明确**：让手机、掌机、车载设备、AR 眼镜等移动终端，在不牺牲续航的前提下，流畅运行 PC 级画质内容。时间线已清晰规划：

::: block-1
- 2025 年 Q3：全套开发工具开源，开发者可提前上手
- 2026 年 Q2：首批集成神经加速器的 Mali-G9xx/Immortalis-G9xx 芯片出货
- 2026 年 Q4：AI 插帧、AI 降噪等进阶功能通过 OTA 推送
:::

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/03.png)

::: image-captionhttps://developer.arm.com/mobile-graphics-and-gaming/neural-graphics
:::

## 二、核心技术：Neural Super Sampling（NSS）的突破

> 作为首发核心功能，NSS（神经超分）通过 AI 技术重构图像细节，彻底革新了移动终端的画质提升方式。

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/04.png)

::: image-caption
NSS 方案整体概览
:::

### 技术原理：从"数学插值"到"智能学习"

传统超分（如双线性插值）依赖固定数学公式，而 NSS 基于**轻量级卷积网络**，通过学习游戏场景中的纹理、光影模式，智能重建丢失细节。其训练数据全部来自游戏内容，可实现 2×/4× 超分，且保持画面时间连贯性（避免帧间闪烁）。

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/05.png)

::: image-caption
NSS 模型结构在 Model Explorer 的示意图
:::

### 实战性能：效率与画质的双重突破

| 关键指标 | 数据 | 说明 |
| --- | --- | --- |
| 分辨率提升 | 540p→1080p（2×） | GPU 原生渲染低分辨率，AI 重建高分辨率 |
| 延迟 | ≤4ms/帧 | 满足实时渲染需求（持续性能模式下测试） |
| 算力消耗 | 27GOPs/帧（含前后处理） | 其中核心网络仅 10GOPs，远低于硬件预算 |
| 功耗效率 | 10TOPS/瓦 | 远超传统计算着色器的 AI 推理效率 |
| 画质表现 | 85%用户盲测无法区分与原生 1080p | 感知损失极小 |

更重要的是，NSS 完全**本地运行**，无需依赖云端，即使离线或网络中断，玩家仍能享受高画质——这与云游戏的服务器端增强形成本质区别。

### 对比传统方案：全面超越

- 与 Arm 2024 年推出的 ASR（传统启发式超分）相比：

## 三、2026 年进阶功能：构建完整 AI 图形管线

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/06.png)

> NSS 只是起点，Arm 计划通过两项核心技术，为移动端打造与桌面 DLSS/FSR 抗衡的完整 AI 图形能力：

### 1. Neural Frame Rate Upscaling（AI 插帧）

- 原理：类似 NVIDIA DLSS 3，通过 AI 生成中间帧
- 效果：GPU 渲染 30fps，显示 60fps，负载几乎不变
- 价值：在不增加功耗的前提下，实现高刷新率流畅体验

### 2. Neural Super Sampling + Denoising（AI 降噪）

- 定位：移动端实时光追的"关键拼图"
- 突破：每像素仅需 1/4 光线采样，通过 AI 降噪即可还原无损画质
- 组合优势：结合 NSS，可将 540p 光追画面无损提升至 1080p，且帧率不下降

## 四、开发者工具链：开源生态加速落地

> 为让开发者提前布局，Arm 推出完全开放的 Neural Graphics Development Kit，核心组件包括：

### SDK 组件一览表

| 组件名称 | 描述 | 使用场景 | GitHub 链接 |
| --- | --- | --- | --- |
| **Model Converter** （模型转换器） | 将 TOSA IR（Tensor Operator Set Architecture 中间表示）转换为 SPIR-V 图，并打包为 `.vgf` 文件以供运行时执行。 | 用于资源管线中的模型部署。 | arm/ai-ml-sdk-model-converter |
| **VGF Library** （VGF 库） | 轻量级运行时解码器，用于加载/使用包含图、常量与着色器的 `.vgf` 文件。 | 集成到游戏引擎中，直接加载并使用图。 | arm/ai-ml-sdk-vgf-library |
| **Scenario Runner** （场景运行器） | 基于 JSON 的场景描述，以声明式方式执行机器学习任务。 | 快速原型验证与效果测试。 | arm/ai-ml-sdk-scenario-runner |
| **Emulation Layer for Vulkan** （Vulkan 仿真层） | 在缺少原生 Vulkan ML 扩展的设备上，通过计算着色器仿真数据图和张量扩展的 Vulkan 层。 | 用于测试，确保在没有硬件 ML 扩展的设备上也能运行。 | arm/ai-ml-emulation-layer-for-vulkan |

::: block-1
- https://github.com/arm/ai-ml-sdk-model-converter
- https://github.com/arm/ai-ml-sdk-vgf-library
- https://github.com/arm/ai-ml-sdk-model-converter
- https://github.com/arm/ai-ml-emulation-layer-for-vulkan
- https://github.com/arm/ai-ml-sdk-scenario-runner
:::

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/07.png)

::: image-captionhttps://developer.arm.com/mobile-graphics-and-gaming/neural-graphics
:::

## 五、生态支持：从引擎到游戏的全链路响应

行业头部企业已积极跟进，推动技术落地：

- **Epic Games**：将 NSS 集成进 Unreal Engine，让开发者提前一年接入 AI 渲染管线
- **腾讯游戏**：计划在下一代《PUBG Mobile》《Honor of Kings》首发支持 NSS
- **网易游戏**：将其纳入 Messiah 引擎，目标实现全场景 1080p 60fps
- **Sumo Digital**：评估《Sackboy: A Big Adventure》移动版画质提升至主机级
- **Traverse Research**：将技术引入商用降噪 SDK，扩展至更多图形场景

## 六、用户价值：移动端体验的全面升级

从玩家视角，NSS 及后续技术将带来革命性体验提升：

| 场景 | 旧体验 | NSS 后体验 |
| --- | --- | --- |
| MOBA 手游 | 720p 30fps | 1080p 60fps，功耗持平 |
| 开放世界 | 540p 25fps | 1080p 30fps，续航增加 20% |
| 光追 Demo | 480p 20fps（无降噪） | 1080p 30fps+AI 降噪，画质接近主机 |
| 云游戏串流 | 720p 60fps（高码率） | 540p 低码率上行 → 本地 1080p 超分，省 40%带宽 |

![文章插图](/articles/2025-08-21-arm-neural-technology-gpu-ai-77b2e25a/images/08.png)

::: image-caption
Arm 正在为 Arm GPU 添加专用神经加速器，以便将人工智能驱动的图形技术引入移动设备，并为未来的设备端人工智能奠定基础
:::

## 七、未来展望：AI 重构移动图形的边界

Arm 的野心不止于"追赶桌面"，而是通过 AI 技术实现移动端图形的"跨越式进化"：

::: block-1
- **硬件层面**：2026 年 GPU 首次集成专用神经加速器，模糊 GPU 与 AI 加速器的界限
- **软件层面**：开源工具链让 AI 渲染成为游戏引擎原生能力，而非"插件式附加"
- **应用层面**：技术将从游戏扩展至 AR/VR 场景重建、移动摄影降噪、自适应 UI 渲染等多元领域
:::

> 正如 Arm 所强调："AI 要塑造移动图形的未来，就必须'烘焙'进芯片"。

随着 NSS 及后续技术的落地，"**手机开光追** "或将从愿景变为常态，移动端图形体验正迎来十年未有的变革。

## 参考文献

1. Arm Newsroom, "Arm Neural Technology Delivers Smarter, Sharper, More Efficient Mobile Graphics for Developers", 2025-08-13.
2. CNX-Software, "Arm neural technology to add AI acceleration to Arm GPUs...", 2025-08-13.
3. The FPS Review, "Arm Announces AI-Driven Neural Super Sampling (NSS) Graphics Upscaler...", 2025-08-14.

交流加群请在 NeuralTalk 公众号后台回复：交个朋友
