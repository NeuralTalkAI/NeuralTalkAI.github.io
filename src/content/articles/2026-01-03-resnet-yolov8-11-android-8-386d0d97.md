关键词：:mdnice-bold-italic[Android]、异构计算、**模型量化**、ResNet、YOLO、**高通骁龙**

> 当你用手机的“拍照识物”功能时，是否遇到过画面卡顿？当直播 APP 的实时美颜延迟明显时，你是否好奇问题出在哪？这些场景的背后，都是**移动 AI 推理** 的核心挑战：如何在手机有限的硬件资源下，让 AI 模型（如图像分类、目标检测）既跑得快（低延迟），又算得准（高精度），还不费电（低能耗）。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/01.webp)

- Hardware optimization on Android for inference of AI models
- https://arxiv.org/pdf/2511.13453
- 4000 字，阅读 13 分钟，**播客 16 分钟**

相关推荐

- [Arm Neural Technology 全景解读：下一代移动 GPU 的 AI 超分、降噪与帧生成功能一次看懂](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891009&idx=1&sn=234c5e1cd7509dd053041847a0cce196&scene=21#wechat_redirect)
- [八款神经网络加速器 μNPU 全面评估：从 Arm 到 RISC-V 架构，从超低功耗到高性能 SoC，评估性能、功耗和内存开销](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888534&idx=1&sn=1e79a785385a632dd6a9f33014104d2f&scene=21#wechat_redirect)
- [让手机CPU,GPU,NPU协同！浙大端侧AI“超级记忆”AME硬件感知引擎：7倍索引加速，6倍插入提速，隐私与速度兼得！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895124&idx=1&sn=5c073c54ccc26b1797e79a12862ac496&scene=21#wechat_redirect)

近日，西班牙马德里康普顿斯大学的研究团队在 arXiv 上发表了题为《Hardware optimization on Android for inference of AI models》的论文，针对 Android 设备的异构硬件（CPU/GPU/NPU），**系统研究了模型量化方案与硬件加速的协同优化，为移动 AI 推理提供了可落地的“精度-速度”平衡方案**。本文将拆解这篇论文的核心方法、创新点与关键发现，并结合原论文图表详细解读。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/02.webp)

::: image-caption
表 4 | 缩写对应。这些缩写用于简化后续实验结果表述，所有对比均以 “CPU-SC 执行 FP32 模型” 为基线，该基线用于计算其他配置的速度提升倍数和精度损失。例如，CPU-MC 指利用设备 8 个 CPU 核心的多线程模式，FINT8 指模型操作、激活值、权重均为 INT8 精度，确保实验数据表述清晰统一
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/03.webp)

::: image-caption
表 15 | 各模型不同量化方案下 CPU 多线程（CPU-MC）相对单线程（CPU-SC）的速度提升。表格显示 CPU 多线程速度提升受模型类型和量化方案影响：INT8 量化下提升最优（10-15 倍），FP32 仅 2-3 倍；ResNet 随模型增大，INT8 加速倍数下降（ResNet50 3.4 倍→ResNet152 13.4 倍，此处原文可能存在排版误差，核心趋势为 YOLO 大模型加速更高），YOLO 则相反，大模型（如 YOLOv8x）INT8 加速达 10 倍以上，因 YOLO 大模型并行性更好，更适配 CPU 不对称多核架构，而 ResNet 大模型或受内存带宽限制
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/04.webp)

::: image-caption
表 14 | 各模型不同量化方案下 NPU 执行的速度提升（相对 CPU-SC FP32 基线）。NPU 是所有设备中加速效果最显著的，且随模型增大加速倍数越高（YOLOv8x INT8 达 298 倍，ResNet18 INT8 达 129.6 倍），因 NPU 专为 AI 张量操作优化。但 DYN（动态量化）加速极弱（如 ResNet18 仅 2.1 倍），因 NPU 不支持动态精度转换，需 CPU 协助处理精度转换，数据在 NPU 与 CPU 间传输的开销抵消了加速效果，仅静态量化（如 INT8、FP16）能发挥 NPU 优势
:::

## 本文目录

- 一、研究背景：移动 AI 的“两难困境”与本文目标
- 二、实验环境：硬件与软件的“地基”
  - 2.1 硬件平台：三星 Galaxy Tab S9 + 骁龙 8 Gen2
  - 2.2 软件工具链：LiteRT 框架与模型转换 pipeline
- 三、核心工作方法：如何科学对比“精度-速度”？
  - 3.1 评估指标
  - 3.2 关键控制细节
- 四、颠覆性实验结果：NPU+INT8 成最优解？
  - 4.1 ResNet（图像分类）：INT8 量化在 NPU 上“精度损失小，速度快”
  - 4.2 YOLO（目标检测）：FP16 更适合，YOLO11 精度优于 v8
  - 4.3 NPU 加速比：最大 298 倍，模型越大加速越明显
- 五、创新性解读：这篇论文的 3 个核心突破
  - 1. 首次系统梳理 Android 异构硬件的量化适配规则
  - 2. 发现“模型大小-量化损失”的负相关规律
  - 3. 用 Pareto 前沿实现“多目标优化”
- 六、相关工作：本文与现有研究的区别
- 七、结论与未来方向
  - 核心结论
  - 未来工作
- 参考文献

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/05.webp)

## 一、研究背景：移动 AI 的“两难困境”与本文目标

> 随着 AI 在手机端的普及（虚拟助手、图像处理、实时检测等），两个问题愈发突出：

1. **计算成本高**：AI 推理（模型“干活”的阶段）占 AI 总能耗的 60%（Google 数据），且 Meta 等公司的 AI 能耗年增长率超 30%；
2. **硬件适配难**：Android 设备的 CPU/GPU/NPU（神经网络处理器）构成“异构架构”，如何让模型在不同硬件上高效运行，同时减少精度损失，是行业痛点。

基于此，论文的**核心目标** 的明确：

::: block-1
- 针对两类主流 AI 模型：ResNet 家族（图像分类）、YOLO 家族（目标检测，含 YOLOv8/YOLO11）；
- 测试 Android 异构硬件：CPU（单/多线程）、GPU（FP32/FP16 模式）、NPU（高通 Hexagon）；
- 对比 6 种量化方案：从高精度的 FP32 到低精度的 INT8、动态量化（DYN）；
- :mdnice-bold-italic[最终找到“最小精度损失”与“最大推理加速”的最优配置。]
:::

## 二、实验环境：硬件与软件的“地基”

要理解实验结果，必须先明确实验的“硬件平台”与“软件工具链”——这是所有优化的基础。

### 2.1 硬件平台：三星 Galaxy Tab S9 + 骁龙 8 Gen2

> 论文选择了市售 Android 平板三星 Galaxy Tab S9，其核心是**高通骁龙 8 Gen2（SM8550-AC）** SoC，包含三大计算单元：CPU、GPU、NPU，具体参数如下：

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/06.webp)

::: image-caption
表 1 | Kryo CPU 配置（包含核心 ID、核心类型及最大频率）。该 CPU 来自三星 Galaxy Tab S9 搭载的骁龙 8 Gen 2（SM8550-AC）SoC
:::

::: block-1
- **CPU**：采用 big.LITTLE 架构3 个小核为 ARM Cortex-A510（ARMv9 系列首代高效核），5 个大核含 2 个 A710、2 个 A715 和 1 个 A715，其中 X3 核频率最高，达 3.36GHz，是性能最强的核心，这种架构旨在平衡性能与能效；
- **GPU**：高通 Adreno 740，支持 FP32（单精度）和 FP16（半精度），含 12 个着色器单元，最高频率 719MHz；
- **NPU**：高通 Hexagon 处理器，:mdnice-bold-italic[支持 INT8/FP16 计算，核心技术是“Micro Tile Inferencing”] （将模型层拆分，由张量/标量/向量单元并行处理，减少数据传输延迟）。
:::

### 2.2 软件工具链：LiteRT 框架与模型转换 pipeline

> 论文采用 Google 的**LiteRT（原 TensorFlow Lite）** 作为推理框架，但由于 LiteRT 直接转换 PyTorch 模型的功能尚不成熟（早期版本支持有限），设计了“PyTorch→ONNX→TFLite”的转换 pipeline，具体如下：

#### 模型准备与实验流程

> workflow 框图，核心步骤如下：

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/07.webp)

::: image-caption
图 1 | 实验工作流与模型准备流程。该流程是实验的核心环节，先从官方仓库获取模型：ResNet 家族来自 PyTorch 的 torchvision 模块，YOLO 家族来自 Ultralytics 官网。接着分两步转换模型，先通过 PyTorch 的导出工具生成中间格式 ONNX，再用 onnx2tf 工具将 ONNX 转为 LiteRT 支持的.tflite 格式，转换时可指定动态量化、INT8 量化等方案。最后对转换后的模型进行分类（ResNet）和检测（YOLO）精度评估，确保模型满足实验分析需求
:::

1. **模型来源**：ResNet 来自 PyTorch torchvision 库，YOLO 来自 Ultralytics 官网；
2. **第一步转换**：用 PyTorch 的`torch.onnx.export`将模型转为 ONNX 格式（中间格式，兼容多框架）；
3. **第二步转换**：用`onnx2tf`工具将 ONNX 转为 TFLite（.tflite）格式，同时指定量化方案；
4. **推理执行**：开发 Android APP，在不同硬件（CPU/GPU/NPU）上运行.tflite 模型，记录推理时间与精度。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/08.webp)

::: image-caption
表 2 | 量化参考。这些量化方案由 onnx2tf 工具支持，用于将 PyTorch 模型转为 TFLite 格式时配置精度。其中静态量化（如 int_quant、full_int_quant）需离线将权重转为目标类型（如 INT8），动态量化（dynamic_range_quant）则在运行时量化激活值。需注意，LiteRT 框架对 INT16 缺乏优化的操作实现，实际性能不佳，后续实验中部分 INT16 量化方案被排除
:::

> 量化的核心是“用低精度数据（如 INT8）替代高精度（FP32）”，减少计算量与内存占用。其中：

- 静态量化（如 INT8/FINT8）：离线将权重转为低精度，运行时直接使用；
- 动态量化（DYN）：仅在运行时将激活量化为 INT8，权重需临时转回 FP32 计算，有额外开销。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/09.webp)

::: image-caption
表 3 | LiteRT 版本要求。该表格源于软件兼容性限制：CPU 和 GPU 采用 LiteRT 2.0.2（即 LiteRT Next），因其 API 能更好地控制这两类硬件，提升实验可控性；而 NPU 需用 1.4.0 版本，因 2.0.2 版本的 NPU 模型运行管线存在问题（如模型无法正常执行）。这种版本拆分设计，是为了全面基准测试 CPU、GPU、NPU 三类加速器的性能，确保各设备都能稳定工作
:::

## 三、核心工作方法：如何科学对比“精度-速度”？

> 论文的方法学核心是“控制变量法”，即固定一个变量（如模型），改变另一个变量（如硬件/量化），通过标准化指标评估性能。

### 3.1 评估指标

- **推理速度**：平均推理时间（单位：ms），越低越好；
- **加速比**：基准（CPU-SC FP32）推理时间 / 目标配置推理时间，越高越好；
- **精度指标**：

### 3.2 关键控制细节

1. **基准设定**：将“CPU 单线程（CPU-SC）+ FP32”作为所有对比的基准，确保数据可比；
2. **模型规模覆盖**：ResNet 测试 18/34/50/101/152（从小到大），YOLO 测试 v8n/s/m/l/x、11n/s/m/l/x（n=小，x=大）；
3. **统计可靠性**：每个配置运行 100 次，取平均推理时间，减少随机误差。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/02.webp)

::: image-caption
表 4 | 缩写对应。这些缩写用于简化后续实验结果表述，所有对比均以 “CPU-SC 执行 FP32 模型” 为基线，该基线用于计算其他配置的速度提升倍数和精度损失。例如，CPU-MC 指利用设备 8 个 CPU 核心的多线程模式，FINT8 指模型操作、激活值、权重均为 INT8 精度，确保实验数据表述清晰统一
:::

## 四、颠覆性实验结果：NPU+INT8 成最优解？

> 论文通过大量实验，得出了多个反常识且实用的结论，以下结合关键图表拆解。

### 4.1 ResNet（图像分类）：INT8 量化在 NPU 上“精度损失小，速度快”

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/10.webp)

::: image-caption
表 5 | ResNet18 平均推理时间。表格清晰展现不同配置下 ResNet18 的推理延迟：INT16 和 FINT16 量化 latency 极高（如 NPU 上 INT16 达 882.30ms），因 LiteRT 缺乏 INT16 优化，后续实验排除这两类量化；NPU 性能最优，INT8 下仅 0.61ms，比 CPU-SC 的 FP32（79.06ms）快超 129 倍；CPU-MC 比 CPU-SC 提速明显，如 FP32 下从 79.06ms 降至 26.34ms，体现多线程优势
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/11.webp)

::: image-caption
表 6 | ResNet18 Top1 准确率（包含不同量化方案及设备）。除 FINT8 外，其他配置准确率均接近 69%：FP32/FP16 准确率约 68.8%，INT8 略降至 65.87%-65.96%，DYN 仅微降 0.1% 左右；而 FINT8 准确率仅 0.08%-0.09%，完全无法用于实际场景，故后续排除。此外，同一量化方案在不同设备上准确率差异极小，说明设备类型对 ResNet18 准确率影响甚微
:::

**关键发现**：

::: block-1
1. **NPU 速度碾压**：NPU+INT8 的推理时间仅 0.61ms，是 CPU-SC FP32（79.06ms）的**129.6 倍加速** （原论文 Table XIV）；
2. **FINT8 废弃原因**：虽然 FINT8 推理时间最短，但准确率仅 0.08%（完全失效），因全 INT8 量化破坏了模型特征提取；
3. **GPU16 更优**：GPU 在 FP16（GPU16）模式下比 FP32（GPU32）快 2 倍左右（如 FP32 13.68ms vs FP16 5.54ms），且精度无损失。
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/12.webp)

::: image-caption
图 2 | ResNet50 跨设备与量化方案的推理速度。纵轴为推理延迟（毫秒），横轴为设备（CPU 单线程 / 多线程、GPU32 位 / 16 位精度、NPU），曲线对应 FP32、FP16、INT8、DYN 四种量化。NPU 在 INT8 量化下表现最优，比 CPU 单线程快超 120 倍；但 DYN 量化在 NPU 上延迟显著升高，因 DYN 含混合精度层，NPU 不支持动态数据类型转换，需依赖 CPU 处理精度转换，数据在两设备间传输的开销抵消了加速效果，这与文中 NPU 适配预计算数据的特性一致
:::

::: block-1
**注**：动态量化（DYN）在 NPU 上“翻车”——推理时间 37ms，远高于 INT8 的 0.61ms。:mdnice-bold-italic[原因是 NPU 不支持混合精度（DYN 的激活与权重精度不同），需频繁在 NPU 与 CPU 间传输数据，产生额外开销。]
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/13.webp)

::: image-caption
表 7 | ResNet50 多线程与单线程执行时间（单位：毫秒）及速度提升。表格显示 CPU 多线程（CPU-MC）相比单线程（CPU-SC）有明显提速，但未达 8 核理论最大速度（8 倍），因骁龙 8 Gen 2 的 CPU 为不对称多核架构，核心性能差异限制了扩展性。其中 INT8 量化下提速最显著（3.4 倍），FP32 提速最少（2.1 倍），说明低精度量化更适配 CPU 多线程并行处理，能更好发挥多核优势
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/14.webp)

::: image-caption
表 8 | ResNet50 GPU FP32 与 FP16 执行时间（单位：毫秒）。表格表明 GPU 性能主要受精度模式（FP32/FP16）影响，GPU16 比 GPU32 快近 2 倍（如 FP32 量化下 GPU32 为 14.9ms，GPU16 为 8.1ms）；而同一精度模式下，不同量化方案（FP32、FP16、INT8、DYN）的 latency 差异极小（如 GPU32 下各量化 latency 在 14.78-15.17ms 间）。这是因 GPU 性能受内存带宽和算术复杂度限制，量化对其影响远小于精度模式
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/15.webp)

::: image-caption
表 9 | ResNet50 Top1 准确率（包含不同量化方案及设备）。ResNet50 在各设备和量化方案下准确率均较高：FP32 与 FP16 准确率几乎一致（约 79.06%-79.10%），INT8 仅微降 0.3%-0.4%（78.68%-78.77%），DYN 接近 FP32（78.90%-78.94%）。且同一量化方案在不同设备上准确率差异极小，说明 ResNet50 对硬件适配性好，设备类型不会显著影响其推理精度，量化是精度变化的核心因素
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/16.webp)

::: image-caption
表 10 | 各 ResNet 模型不同量化方案相对 FP32 的 Top1 精度损失（单位：%）。表格揭示量化对 ResNet 精度的影响规律：INT8 精度损失随模型复杂度增加而显著降低，ResNet18（小模型）损失 2.94%，ResNet101（大模型）仅 0.01%，因大模型深层特征提取能力强，能减轻低精度量化的信息损失；而 DYN（动态量化）损失始终极小（0.00%-0.19%），几乎不影响精度，是 ResNet 系列平衡速度与精度的优选方案
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/17.webp)

::: image-caption
表 11 | 各 ResNet 模型从 PyTorch 转为 LiteRT 框架的 Top1 精度损失（单位：%）。该表格显示模型框架转换存在固有精度损失，损失范围在 0.83%-1.77% 之间，其中 ResNet50 损失最大（1.77%），ResNet34 损失最小（0.83%）。这是因 PyTorch 与 LiteRT 的操作实现细节存在差异（如算子计算逻辑、数值处理方式），无法实现 1:1 完美转换，因此在评估 LiteRT 模型精度时，需将该转换损失纳入整体精度 degradation 分析
:::

### 4.2 YOLO（目标检测）：FP16 更适合，YOLO11 精度优于 v8

> 目标检测比分类更复杂（需定位+分类），量化对精度的影响更大。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/18.webp)

::: image-caption
表 12 | 各 YOLOv8 模型不同量化方案相对 FP32 的 mAP 损失。与 ResNet 不同，YOLOv8（目标检测模型）对 INT8 量化敏感，平均 mAP 损失约 6.5（YOLOv8l 达 7.5），因目标检测需同时完成目标定位与分类，对特征精度要求更高，低精度量化易丢失关键信息；而 DYN（动态量化）损失极小（0.0-0.1），几乎不影响检测效果，说明 YOLOv8 更适合采用动态量化，而非 INT8 量化
:::

**关键发现**：

::: block-1
- **INT8 量化对 YOLO“伤害大”**：平均 mAP 损失 6.5 个百分点（如 YOLOv8l 从原精度 50%降至 42.5%），因检测需更精细的特征；
- 动态量化（DYN）精度损失极小（<0.1），但速度比 INT8 慢（NPU 不支持），适合对精度敏感的场景。
:::

#### YOLO11 vs YOLOv8 精度对比

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/19.webp)

::: image-caption
图 3 | YOLOv8n 跨设备与量化方案的推理速度。纵轴为推理延迟（毫秒），横轴为不同设备，涵盖四种量化方案。YOLOv8n 推理延迟整体高于 ResNet50，因目标检测需同时完成定位与分类，任务复杂度更高。所有设备中 NPU 速度最快，GPU 次之，CPU 单线程最慢；同一设备下，INT8 等低精度量化通常比 FP32 快，但提升幅度因设备而异，印证了文中 “加速器可大幅提升 YOLO 模型推理速度” 的结论
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/20.webp)

::: image-caption
图 5 | YOLO11 与 YOLOv8 的精度对比。纵轴为 FP32 精度下的 mAP（目标检测精度指标），横轴为模型尺寸（n、s、m、l、x）。所有尺寸的 YOLO11 模型 mAP 均高于同尺寸 YOLOv8，即使 YOLO11 从 PyTorch 转为 LiteRT 框架有 0.2-0.4 mAP 的固有损失，仍保持精度优势。其中小尺寸模型（n、s）的精度差距更明显，大尺寸模型优势略有收窄，证明 YOLO11 在目标检测精度上的迭代提升
:::

::: block-1
**注**：YOLO11 作为 v8 的升级版，在所有型号上精度均更高（如 YOLO11n mAP 比 v8n 高 2%），但推理时间略增（YOLO11n CPU-SC FP32 为 120ms，v8n 为 110ms），体现“精度-速度”的权衡。
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/21.webp)

::: image-caption
图 4 | YOLO11n 跨设备与量化方案的推理速度。纵轴为推理延迟（毫秒），横轴为设备类型，包含四种量化方案。作为 YOLOv8 的升级款，YOLO11n 在 CPU 上推理略快于 YOLOv8n，GPU 速度接近，但 NPU 速度稍慢，且 DYN 量化完全无法在 NPU 运行。这是因 YOLO11 新增复杂层，NPU 对其适配性下降，动态量化所需的精度转换更难支持，体现了新架构对硬件兼容性的影响
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/22.webp)

::: image-caption
表 13 | 各 YOLO11 模型不同量化方案相对 FP32 的 mAP 损失。YOLO11 作为 YOLOv8 的升级版本，INT8 量化的 mAP 损失（平均 7.2，YOLO11l 达 8.7）比 YOLOv8 略高，因 YOLO11 架构更复杂且新增层，低精度量化对其特征表达影响更大；DYN 损失仍极小（0.1-0.2）。此外，YOLO11 从 PyTorch 转为 LiteRT 时，有 0.2-0.4 mAP 损失（因新层转换难度大），但 FP32 精度仍优于 YOLOv8，整体检测性能更优
:::

### 4.3 NPU 加速比：最大 298 倍，模型越大加速越明显

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/04.webp)

::: image-caption
表 14 | 各模型不同量化方案下 NPU 执行的速度提升（相对 CPU-SC FP32 基线）。NPU 是所有设备中加速效果最显著的，且随模型增大加速倍数越高（YOLOv8x INT8 达 298 倍，ResNet18 INT8 达 129.6 倍），因 NPU 专为 AI 张量操作优化。但 DYN（动态量化）加速极弱（如 ResNet18 仅 2.1 倍），因 NPU 不支持动态精度转换，需 CPU 协助处理精度转换，数据在 NPU 与 CPU 间传输的开销抵消了加速效果，仅静态量化（如 INT8、FP16）能发挥 NPU 优势
:::

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/11.webp)

::: image-caption
表 6 | ResNet18 Top1 准确率（包含不同量化方案及设备）。除 FINT8 外，其他配置准确率均接近 69%：FP32/FP16 准确率约 68.8%，INT8 略降至 65.87%-65.96%，DYN 仅微降 0.1% 左右；而 FINT8 准确率仅 0.08%-0.09%，完全无法用于实际场景，故后续排除。此外，同一量化方案在不同设备上准确率差异极小，说明设备类型对 ResNet18 准确率影响甚微
:::

**关键发现**：

::: block-1
1. **INT8 加速比最高**：YOLOv8x INT8 在 NPU 上实现 298 倍加速，意味着原本 CPU 需要 298ms 的推理，NPU 仅需 1ms；
2. **模型越大，加速越明显**：ResNet152（大模型）INT8 加速比 164.1x，远高于 ResNet18 的 129.6x，因 NPU 擅长并行处理大规模张量运算；
3. **DYN 加速比失效**：仅 1.6-2.1x，与 CPU-MC 相当，印证 NPU 不支持混合精度的结论。
:::

## 五、创新性解读：这篇论文的 3 个核心突破

> 不同于“泛泛谈移动 AI 优化”的研究，本文的创新点具体且可落地：

### 1. 首次系统梳理 Android 异构硬件的量化适配规则

论文明确了“**哪种硬件该用哪种量化方案** ”：

::: block-1
- NPU：优先 INT8（加速比最高，ResNet 精度损失<3%，大模型损失<0.5%）；
- GPU：优先 FP16（加速比 2x，精度无损失，INT8 无额外收益）；
- CPU：优先 DYN（精度损失<0.2%，多线程加速比 3.4x，INT8 需谨慎）。
:::

这一规则解决了开发者的“选择困难症”——无需反复测试，可直接根据硬件选型量化方案。

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/23.webp)

::: image-caption
表 16 | 各模型不同量化方案下 GPU16 相对 CPU-SC FP32 基线的速度提升。GPU16 对 YOLO 大模型加速效果更显著（YOLOv8x 达 38.6 倍），ResNet 各模型加速稳定（15-20 倍），因 YOLO 大模型包含更多可并行的张量操作，更能发挥 GPU 的并行计算优势。且不同量化方案对 GPU16 加速影响小（如 YOLOv8x 各量化加速在 34.8-38.6 倍间），因 GPU 性能受内存带宽和算术复杂度限制，量化对其并行效率影响有限，FP16 精度模式已能最大化 GPU 性能
:::

### 2. 发现“模型大小-量化损失”的负相关规律

论文通过 ResNet 家族实验发现：

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/16.webp)

::: image-caption
表 10 | 各 ResNet 模型不同量化方案相对 FP32 的 Top1 精度损失（单位：%）。表格揭示量化对 ResNet 精度的影响规律：INT8 精度损失随模型复杂度增加而显著降低，ResNet18（小模型）损失 2.94%，ResNet101（大模型）仅 0.01%，因大模型深层特征提取能力强，能减轻低精度量化的信息损失；而 DYN（动态量化）损失始终极小（0.00%-0.19%），几乎不影响精度，是 ResNet 系列平衡速度与精度的优选方案
:::

::: block-1
**规律**：模型越大（层数越多），INT8 量化的精度损失越小。:mdnice-bold-italic[原因是大模型的“特征冗余度更高”，低精度量化对关键特征的破坏更小]。这一发现颠覆了“小模型更适合量化”的传统认知，为大模型在移动端落地提供了依据。
:::

### 3. 用 Pareto 前沿实现“多目标优化”

> 论文通过 Pareto 前沿分析（仅保留“精度不低、速度不慢”的非劣解），为不同场景提供最优配置：

#### ResNet 的 Pareto 前沿

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/24.webp)

::: image-caption
图 6 | ResNet 的帕累托前沿（Pareto Front ResNet）。纵轴为推理延迟（毫秒），横轴为 Top1 精度（%），实心点代表 FP16 量化，空心点代表 INT8 量化，涵盖 ResNet18 至 ResNet152 各型号。帕累托前沿展示了精度与延迟的最优权衡，ResNet 所有型号的 INT8 量化点均更靠近前沿：虽 INT8 比 FP16 精度略低（如 ResNet18 低约 3%），但延迟大幅降低（NPU 上 INT8 延迟仅为 FP16 的一半左右），完全抵消精度损失，符合文中 “ResNet 选 INT8 最优” 的结论
:::

::: block-1
**结论**：**ResNet 所有型号的 INT8 配置均在 Pareto 前沿上** ——如 ResNet50 INT8 在 NPU 上，精度 78.77%（仅损失 0.41%），推理时间 0.6ms，是“精度-速度”的最优解。
:::

#### YOLO 的 Pareto 前沿

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/25.webp)

::: image-caption
图 7 | YOLO 的帕累托前沿（Pareto Front YOLO）。纵轴为推理延迟（毫秒），横轴为 mAP（%），实心点分别是 YOLOv8 和 YOLO11 的 FP16 量化，空心点是两者的 INT8 量化，包含各尺寸模型。与 ResNet 不同，YOLO 的 INT8 量化点因精度损失大（平均 6.5-7.2 mAP），远离最优前沿；而 FP16 量化点更靠近前沿，尤其是 YOLO11 小尺寸模型（n、s），虽延迟比 YOLOv8 稍高，但 mAP 更高，成为目标检测任务中速度与精度权衡的优选
:::

::: block-1
**结论**：**YOLO 需优先选 FP16** ——如 YOLO11s FP16 在 NPU 上，mAP 48%（比 v8s 高 1.5%），推理时间 8ms，是目标检测的最优配置。
:::

## 六、相关工作：本文与现有研究的区别

> 为了凸显创新性，需对比现有研究的不足：

| 相关工作 | 研究重点 | 本文的改进点 |
| --- | --- | --- |
| MLPerf Inference [10] | 通用 ML 推理基准（跨平台） | 聚焦 Android 异构硬件，补充 NPU 量化数据 |
| TensorFlow Lite 文档 | 框架使用指南 | 解决版本适配 bug，提供具体量化-硬件匹配规则 |
| 早期移动量化研究 | 仅关注 CPU/GPU，小模型 | 扩展到 NPU，验证大模型量化优势 |
| YOLOv8 官方优化 | 模型结构优化 | 补充 Android 硬件加速数据，对比 YOLO11 |

简言之，现有研究要么是“通用基准”，要么是“框架指南”，而本文是针对 Android 设备（骁龙 8 Gen2）、ResNet/YOLO 家族的“量化-硬件-精度”协同优化的系统性研究。

## 七、结论与未来方向

### 核心结论

1. **NPU 是移动 AI 的核心加速器**：加速比最高 298 倍，是低延迟场景的必选硬件；
2. **量化方案需“因模型而异”**：ResNet 用 INT8，YOLO 用 FP16，DYN 仅适合 CPU；
3. **YOLO11 优于 v8**：小型号精度更高，适合手机端实时检测。

### 未来工作

::: block-1
1. 加入**功耗分析**：当前仅测 latency，需补充不同配置的能耗数据，实现“精度- latency-功耗”三目标优化；
2. 修复 NPU 动态量化兼容性：解决混合精度数据传输开销问题；
3. 扩展模型类型：测试 NLP（如 BERT）、图像分割（如 U-Net）模型，验证结论通用性。
:::

## 参考文献

![文章插图](/articles/2026-01-03-resnet-yolov8-11-android-8-386d0d97/images/26.webp)
