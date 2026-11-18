关键词：**GPU 数据库**、AI 编译器、内核优化、硬件规格、JSON 数据集

> 翻查厂商文档犹如考古，剖析性能瓶颈恰似验尸。

做 GPU 内核开发时，真正折磨人通常来说真不是某一行 CUDA 或 shader 代码，却是那些分散在厂商白皮书、规格网页、论坛表格里的硬件事实：这张卡支持哪个 CUDA Compute Capability？每个 SM 有多少共享内存？显存带宽是多少？线程块上限如何影响 kernel launch？

![文章插图](/articles/2026-07-04-nvidia-amd-intel-gpu-2824-cd9fa4ff/images/01.webp)

- :mdnice-bold-italic[RightNow GPU Database：Comprehensive GPU specifications database with 2,824 GPUs across NVIDIA, AMD, and Intel]
- https://github.com/RightNow-AI/RightNow-GPU-Database

::: related-title
**相关推荐**
:::

::: related-links
- **[暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**
- **[Auto-MicroBench：把 LLM 变成可验证的 GPU 微基准工程师，自动校准模拟器的硬件真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903091&idx=1&sn=00b927c466aaca4a983e55ac52b68fe6&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**

:::

> 当 AI 系统开始自动生成、改写甚至调优 GPU kernel 时，这些问题就不再是“查一下文档”的小事，而会变成编译器决策、代码生成边界和性能模型的输入。

RightNow GPU Database 的价值正在于此：它没有试图写一个复杂框架，却把**NVIDIA、AMD、Intel 共 2,824 款 GPU 的规格压缩成一套可直接读取、可版本化、可嵌入工具链的 JSON 硬件事实库。**

## 本文目录

- 一、快速上手：把 GPU 规格当作接口直接读取
- 二、项目定位：它不是应用，而是 GPU 硬件事实层
- 三、Schema 设计：用最多 55 个字段描述一张 GPU
- 四、空值策略：为什么“不写 null”是一种工程选择
- 五、厂商差异：同一 Schema 下保留不同 GPU 家族的语义
- 六、对 AI 编译器和内核生成的意义：让模型生成代码前先读硬件
- 七、性能字段的正确读法：峰值不是实测，但能提供上界
- 八、数据源与可信边界：开源数据库也需要知道从哪里来
- 九、设计上的克制：没有 API 服务反而是一种优势
- 十、从数据到产品：它能支撑哪些真实场景
- 十一、值得改进的地方：从数据集走向硬件知识库
- 结语：真正的突破，是让硬件知识进入软件闭环

## 一、快速上手：把 GPU 规格当作接口直接读取

> 这个仓库没有复杂的安装流程，也没有构建脚本。它更像一个“数据依赖包”：**核心资产都放在 `data/` 目录中，使用者可以直接通过 GitHub Raw URL、克隆仓库，或将 JSON 文件集成进自己的工具链。**

README 中给出的最小使用方式是直接读取厂商维度的 JSON 文件：

```
// 来源：README.md Direct URL: https://raw.githubusercontent.com/RightNowAI/gpu-database/main/data/nvidia/all.json
```

如果你基于当前仓库地址访问，也可以使用如下形式：

```
# 来源：README.md 与仓库目录结构整理git clone https://github.com/RightNow-AI/RightNow-GPU-Database.gitcd RightNow-GPU-Database# 查看三个厂商的数据文件ls data/nvidia/all.json data/amd/all.json data/intel/all.json data/all-gpus.json
```

JavaScript 里可以把它当作远程 JSON 数据源：

```
// 来源：README.mdconst gpus = await fetch(  'https://raw.githubusercontent.com/RightNowAI/gpu-database/main/data/nvidia/all.json').then(r => r.json());const rtx4090 = gpus.find(g => g.name === 'GeForce RTX 4090');console.log(rtx4090.cuda);           // "8.9"console.log(rtx4090.sharedMemPerSM); // 102400console.log(rtx4090.coresPerSM);     // 128
```

Python 中则可以直接请求并查询：

```
# 来源：README.mdimport requestsgpus = requests.get(    'https://raw.githubusercontent.com/RightNowAI/gpu-database/main/data/nvidia/all.json').json()rtx4090 = next(g for g in gpus if g['name'] == 'GeForce RTX 4090')print(rtx4090['cuda'])            # 8.9print(rtx4090['maxThreadsPerSM']) # 1536
```

更多具体字段可以参考仓库中的`README.md`[1]，完整数据则位于`data/`[2]。

## 二、项目定位：它不是应用，而是 GPU 硬件事实层

> RightNow GPU Database 的仓库结构非常克制。顶层主要包含：

```
// 来源：仓库顶层目录与 data/ 子目录 . ├── README.md ├── LICENSE ├── package.json └── data ├── all-gpus.json ├── nvidia/all.json ├── amd/all.json └── intel/all.json
```

`package.json` 也印证了它的定位：项目名是`gpu-database`，描述为 “Comprehensive GPU specifications database for AI and GPU kernel development”，关键词包括`gpu`、`nvidia`、`cuda`、`amd`、`intel`、`machine-learning`、`deep-learning`。

::: block-1
也就是说，它不是运行时服务，也不是npm 工具包，当然更不是一个前端应用，而是👀（欲扬先抑🤮）一份面向 AI 与 GPU kernel 开发的数据资产。
:::

```
{  "name": "gpu-database","version": "1.0.0","description": "Comprehensive GPU specifications database for AI and GPU kernel development","author": "RightNow AI <hello@rightnowai.co>","license": "Apache-2.0","keywords": [    "gpu",    "nvidia",    "cuda",    "amd",    "intel",    "specifications",    "database",    "machine-learning",    "deep-learning"  ]}
```

这类项目的关键不在“代码复杂度”，而在“**数据结构是否足够适合被机器消费** ”。对人类读者来说，显卡规格表是网页；:mdnice-bold-italic[对编译器、推荐系统、AI 代码编辑器来说，显卡规格必须是结构化字段：可过滤、可比较、可推导、可参与约束判断。]

README 明确写出该库覆盖 NVIDIA、AMD、Intel 三家，共 2,824 款 GPU。其中 NVIDIA 1,286 款，AMD 1,292 款，Intel 180 款，另有一个聚合文件`data/all-gpus.json`。

| Vendor | GPUs | File |
| --- | --- | --- |
| NVIDIA | 1,286 | `data/nvidia/all.json`[3] |
| AMD | 1,292 | `data/amd/all.json`[4] |
| Intel | 180 | `data/intel/all.json`[5] |
| **All** | 2,824 | `data/all-gpus.json`[6] |

> 换句话说，它的“架构”不是传统意义上的模块调用关系，而是一个面向硬件知识的分层索引：

```
// 来源：data/ 目录结构 data/all-gpus.json # 跨厂商聚合入口，适合全局检索、排序、筛选 data/nvidia/all.json # NVIDIA GPU 数据，包含 CUDA/SM/warp 等字段 data/amd/all.json # AMD GPU 数据，覆盖架构、显存、接口、性能等字段 data/intel/all.json # Intel GPU 数据，包含 EU、subslice 等 Intel 特色字段
```

这个设计很直接，但对工具链非常友好：

- 如果你只关心 CUDA kernel，可以读取 NVIDIA 文件；如果你要做跨厂商硬件推荐，则读取`all-gpus.json`；
- 如果你在分析 Intel Arc 或集显演进，则读取 Intel 数据即可。

## 三、Schema 设计：用最多 55 个字段描述一张 GPU

> README 中说明，每个 GPU 条目最多包含 55 个字段，并强调“Only populated fields are included — no nulls or zeros.” 这句话看似简单，实际是整个数据模型最重要的设计选择之一。

首先看 RTX 4090 示例：

```
{  "name": "GeForce RTX 4090","vendor": "nvidia","manufacturer": "NVIDIA","gpuName": "AD102","architecture": "Ada Lovelace","generation": "GeForce 40","foundry": "TSMC","processSize": 5,"transistors": 76.3,"transistorDensity": 125.3,"dieSize": 609.0,"releaseDate": "2022-09-20","baseClock": 2235.0,"boostClock": 2520.0,"memoryClock": 1313.0,"memorySize": 24.0,"memoryType": "GDDR6X","memoryBus": 384,"memoryBandwidth": 1010.0,"shaders": 16384,"sms": 128,"tensorCores": 512,"rtCores": 128,"coresPerSM": 128,"fp16": 82.58,"fp32": 82.58,"fp64": 1.29,"cuda": "8.9","warpSize": 32,"maxThreadsPerBlock": 1024,"maxThreadsPerSM": 1536,"sharedMemPerSM": 102400,"registersPerSM": 65536}
```

这份 Schema 大致可以分为七组：

| 字段组 | 字段列表 | 说明 |
| --- | --- | --- |
| 身份字段 | `id`、`name`、`vendor`、`manufacturer`、`gpuName`、`architecture`、`generation` | 解决“这是谁”的问题，标识 GPU 的基本身份与代际归属。 |
| 制造工艺字段 | `foundry`、`processSize`、`transistors`、`transistorDensity`、`dieSize`、`chipPackage`、`releaseDate` | 反映芯片物理实现与演进脉络，支撑代际对比与制程分析。 |
| 频率字段 | `baseClock`、`boostClock`、`memoryClock` | 构成理论性能计算的基础输入，影响时钟驱动型吞吐估算。 |
| 显存字段 | `memorySize`、`memoryType`、`memoryBus`、`memoryBandwidth` | 对 AI 推理/训练至关重要：容量决定模型与 KV cache 加载能力，带宽制约 memory-bound kernel 性能上限。 |
| 执行资源字段 | `shaders`、`tmus`、`rops`、`sms`、`tensorCores`、`rtCores`、`coresPerSM` | 渲染侧重 TMU/ROP；AI 计算更关注 SM、Tensor Core 与 shader 规模。 |
| 性能字段 | `pixelRate`、`textureRate`、`fp16`、`fp32`、`fp64` | 可用于 roofline model 粗粒度建模，但仅为理论峰值，非实际任务吞吐。 |
| 编程模型字段 | `directX`、`openGL`、`vulkan`、`openCL`、`cuda`、`shaderModel`、`warpSize`、`maxThreadsPerBlock`、`maxThreadsPerSM`、`maxBlocksPerSM`、`sharedMemPerSM`、`registersPerSM` | 支撑 AI 编译器与 kernel 生成，是软硬件协同优化的核心元数据。 |

尤其是下面几个字段：

```
{  "cuda": "8.9","warpSize": 32,"maxThreadsPerBlock": 1024,"maxThreadsPerSM": 1536,"maxBlocksPerSM": 24,"sharedMemPerSM": 102400,"registersPerSM": 65536}
```

这些是 kernel launch 和编译优化的硬边界。例如：

- `maxThreadsPerBlock` 会限制 block size
- `sharedMemPerSM` 会限制 tile 大小
- `registersPerSM` 会影响 occupancy
- `warpSize` 会影响线程组织和分支发散成本

如果一个 AI 代码助手要生成 CUDA kernel，它必须知道这些边界，否则很容易写出“语法正确但性能灾难”甚至无法 launch 的代码。

## 四、空值策略：为什么“不写 null”是一种工程选择

> README 中的 “Only populated fields are included — no nulls or zeros” 很值得展开。

在很多数据库设计中，字段统一、缺失填`null` 是常见做法。但在这个仓库中，GPU 之间跨度极大：**从 1998 年 Intel i740，到现代 NVIDIA Ada、Intel Arc，硬件概念本身就发生了变化**。早期显卡没有 Tensor Core，没有现代意义上的 CUDA compute capability，也没有 Vulkan；Intel 集显常用`executionUnits`、`subslices`，NVIDIA 则更关注`sms`、`warpSize`、`tensorCores`。

::: block-1
如果强行给所有 GPU 填满同一套字段，就会出现大量无意义的`null`。而“不存在字段”比“字段为 null”更接近真实语义：不是数值未知，而是这个概念对该硬件可能并不适用，或者数据源没有提供可靠值。
:::

Intel 早期 i740 的数据就体现了这种稀疏模型：

```
{  "id": "intel-i740","name": "i740","vendor": "intel","manufacturer": "Intel","gpuName": "Auburn","architecture": "i740","generation": "Graphics","foundry": "Intel","processSize": 150,"releaseDate": "1998-02-12","busInterface": "AGP 2x","baseClock": 66.0,"boostClock": 66.0,"memoryClock": 100.0,"memorySize": 0.004,"memoryType": "SDR","memoryBus": 64,"memoryBandwidth": 0.8,"tmus": 1,"rops": 1,"directX": "5.0","openGL": "1.1","url": "https://www.techpowerup.com/gpu-specs/i740.c1288"}
```

这个对象没有`cuda`，没有`fp16`，没有`tensorCores`，也没有`sharedMemPerSM`。这不是缺陷，而是数据模型对历史硬件差异的尊重。

对使用者来说，正确读取方式不应该假设字段一定存在，而应采用能力探测式逻辑：

```
// 来源：基于 data/*.json Schema 的简化伪代码function supportsCudaKernel(gpu) {// 字段不存在，不等同于 false；但对 CUDA kernel 生成而言，可视为不可用return gpu.vendor === "nvidia" && typeof gpu.cuda === "string";}function canUseLargeSharedMemoryTile(gpu, requiredBytes) {// sharedMemPerSM 缺失时，不应贸然生成依赖共享内存容量的优化版本returntypeof gpu.sharedMemPerSM === "number" &&         gpu.sharedMemPerSM >= requiredBytes;}
```

**这是一种非常适合 AI 工具链的模式：先读硬件事实，再决定代码生成路径**。并非“写完 kernel 再祈祷它能跑”，而要明确“在生成 kernel 之前就知道边界”。

## 五、厂商差异：同一 Schema 下保留不同 GPU 家族的语义

> 跨厂商 GPU 数据库最大的难点，不只是堆字段，也**要统一结构并在不同厂商特性之间保持平衡。**

- NVIDIA 的关键语义往往围绕 CUDA、SM、warp、Tensor Core 展开。RTX 4090 示例里，`cuda: "8.9"`、`sms: 128`、`warpSize: 32`、`sharedMemPerSM: 102400` 等字段直接服务于 CUDA kernel 编写。
- Intel 数据则体现出另一套结构语言：`executionUnits` 和`subslices`。例如 Intel Arc A770：

```
{  "id": "intel-arc-a770","name": "Arc A770","vendor": "intel","manufacturer": "Intel","gpuName": "DG2-512","architecture": "Xe-HPG","generation": "Alchemist(Arc 7)","foundry": "TSMC","processSize": 6,"transistors": 21.7,"transistorDensity": 53.4,"dieSize": 406.0,"releaseDate": "2022-10-12","busInterface": "PCIe 4.0 x16","baseClock": 2100.0,"boostClock": 2400.0,"memoryClock": 2000.0,"memorySize": 16.0,"memoryType": "GDDR6","memoryBus": 256,"memoryBandwidth": 512.0,"shaders": 4096,"tmus": 256,"rops": 128,"rtCores": 32,"l2Cache": 16.0,"tdp": 225,"fp16": 39.32,"fp32": 19.66,"fp64": 2.46,"directX": "12.2","openGL": "4.6","vulkan": "1.4","openCL": "3.0","shaderModel": "6.6","executionUnits": 512,"subslices": 32}
```

这段数据很有代表性：它既能放进统一字段体系，比如显存、频率、工艺、API 支持、理论算力；又保留了 Intel 自己的执行单元组织方式。对数据库来说，这避免了“为了统一而抹平差异”；对使用者来说，则意味着可以先做通用分析，再做厂商特化处理。

> 一个合理的抽象方式是将字段分成“跨厂商通用层”和“厂商特化层”：

```
// 来源：基于 README.md 与 data/intel/all.json 的简化类型抽象type CommonGpuFacts = {  id: string;  name: string;  vendor: "nvidia" | "amd" | "intel";  architecture?: string;  processSize?: number;  memorySize?: number;  memoryBandwidth?: number;  fp16?: number;  fp32?: number;  fp64?: number;  openCL?: string;  vulkan?: string;};type NvidiaKernelFacts = {  cuda?: string;  sms?: number;  warpSize?: number;  sharedMemPerSM?: number;  registersPerSM?: number;  tensorCores?: number;};type IntelKernelFacts = {  executionUnits?: number;  subslices?: number;};
```

注意这里全部使用可选字段。这可不是 TypeScript 风格偏好，而是对数据真实形态的映射。因为这个数据库不是围绕单一厂商、单一时代设计的，而是**横跨近三十年 GPU 演进史。**

## 六、对 AI 编译器和内核生成的意义：让模型生成代码前先读硬件

> RightNow 在 README 中提到该项目是 RightNow 的一部分，而 RightNow 是面向 GPU kernel development 的 AI-powered code editor。这个背景非常重要。因为如果只是给人查显卡参数，一个网页表格就够了；**但如果要服务 AI 代码编辑器，JSON 数据库就变成了编译器前端的一部分。**

设想一个 AI 系统要为用户生成矩阵乘法 kernel。它至少需要回答几个问题：

- 目标 GPU 的 shared memory 是否足够支持某个 tile size？
- 每个 block 应该使用多少线程，是否超过`maxThreadsPerBlock`？
- 是否可以使用 Tensor Core？
- CUDA compute capability 是否支持某些指令或内存访问模式？
- 显存带宽和 FP16/FP32 峰值大致暗示该 kernel 更可能 compute-bound 还是 memory-bound？

::: block-1
这些判断都不能只靠大语言模型“记忆”。模型可能记错硬件规格，也可能混淆不同显卡版本。**更可靠的方式是：把 GPU 名称映射到数据库条目，再让代码生成器在硬件约束内行动。**
:::

例如一个简化的 kernel 配置选择流程可以这样写：

```
# 来源：基于 README.md Schema 的简化伪代码def choose_cuda_kernel_config(gpu, dtype):    # 1. 确认这是可用于 CUDA 路径的 NVIDIA GPU    if gpu.get("vendor") != "nvidia"or"cuda"notin gpu:        raise ValueError("目标 GPU 不适合走 CUDA kernel 生成路径")    # 2. 使用硬件边界约束 block size    max_threads = gpu.get("maxThreadsPerBlock", 1024)    warp_size = gpu.get("warpSize", 32)    # 3. 根据共享内存容量选择 tile    shared_mem = gpu.get("sharedMemPerSM", 0)    if shared_mem >= 100 * 1024:        tile_m, tile_n = 128, 128    else:        tile_m, tile_n = 64, 64    # 4. 根据 Tensor Core 可用性选择实现路线    use_tensor_core = dtype in ("fp16", "bf16") and gpu.get("tensorCores", 0) > 0    return {        "block_threads": min(256, max_threads),        "warp_size": warp_size,        "tile": (tile_m, tile_n),        "use_tensor_core": use_tensor_core,    }
```

这段伪代码不追求完整，却说明了数据库的核心意义：它让“硬件规格”从自然语言知识变成了程序分支条件。AI 生成 kernel 时，数据库相当于一张交通规则表；没有它，模型只能凭经验开车。

## 七、性能字段的正确读法：峰值不是实测，但能提供上界

> 数据库中包含 `fp16`、`fp32`、`fp64`、`memoryBandwidth`、`pixelRate`、`textureRate` 等字段**。对 AI 系统而言，最常用的是浮点吞吐和显存带宽。它们可以帮助建立一个粗略的 roofline 判断：**

```
# 来源：基于 data/*.json 性能字段的简化伪代码def roofline_hint(gpu, arithmetic_intensity):    """    arithmetic_intensity: 每读取 1 Byte 数据执行多少 FLOPs    返回粗略瓶颈判断，不代表真实 benchmark。    """    fp32_peak = gpu.get("fp32")              # 通常以 TFLOPS 表示    bandwidth = gpu.get("memoryBandwidth")  # 通常以 GB/s 表示    if fp32_peak isNoneor bandwidth isNone:        return"缺少足够字段，不能判断"    # 简化：把 TFLOPS 与 GB/s 转成相对比较所需量纲    compute_bound_threshold = fp32_peak * 1000 / bandwidth    if arithmetic_intensity >= compute_bound_threshold:        return"更可能受计算峰值限制"    return"更可能受显存带宽限制"
```

这里必须强调一个严谨点：:mdnice-bold-italic[数据库中的`fp32` 等字段不能直接等价于你的模型推理吞吐，也不能替代 benchmark。真实性能还受 kernel 实现、访存模式、缓存命中、调度、驱动、功耗墙、精度路径等因素影响。但它们非常适合作为“上界”和“先验”。]

这就像地图上的高速限速：它不能告诉你今天实际会开多快，但能告诉你这条路理论上不可能无限快。对编译器和 AI 代码助手来说，这种上界信息足够有用。

## 八、数据源与可信边界：开源数据库也需要知道从哪里来

README 中明确说明数据来源：

```
// 来源：README.md## SourceData sourced from [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/ "TechPowerUp GPU Database") via [dbgpu](https://github.com/painebenjamin/dbgpu "dbgpu").## License[Apache 2.0](LICENSE "Apache 2.0")
```

这句话决定了使用者应如何看待数据可信度。它不是厂商官方 SDK，也不是运行时自动探测结果，**是来自 TechPowerUp GPU Database，并经由`dbgpu` 整理后形成的数据集**。因此，适合做：

- 硬件信息查询；
- AI 代码生成前的能力约束；
- GPU 代际分析；
- 粗粒度性能建模；
- 选型推荐；
- 文档、编辑器、插件中的硬件知识补全。

但如果用于生产级调度、计费、云资源隔离、驱动兼容性判定，仍应结合运行时探测。例如 NVIDIA 可以通过 CUDA Runtime API 查询实际设备属性，Linux 环境可以结合`nvidia-smi`、ROCm SMI、Level Zero 等工具。静态数据库解决的是“先验事实”，运行时 API 解决的是“当前机器状态”。

最稳妥的工程方式，是把二者结合：

```
# 来源：基于仓库静态 Schema 的工程化使用伪代码def merge_static_and_runtime_facts(static_gpu, runtime_gpu):    """    static_gpu: 来自 RightNow GPU Database 的静态规格    runtime_gpu: 来自本机驱动/API 的实时探测结果    """    facts = dict(static_gpu)    # 运行时信息优先覆盖会受环境影响的字段    for key in ["driverVersion", "availableMemory", "multiprocessorCount"]:        if key in runtime_gpu:            facts[key] = runtime_gpu[key]    # 静态数据库仍保留架构、工艺、理论峰值、API 支持等背景信息    return facts
```

这样，数据库不需要承担它不该承担的责任，却能在工具链中发挥最大作用。

## 九、设计上的克制：没有 API 服务反而是一种优势

> 很多数据库项目会很自然地走向“提供一个服务端 API”。RightNow GPU Database 没有这么做，至少从当前仓库结构看，它就是 JSON 文件加 README。这种克制有明显好处。

- **第一，零运行时依赖**。用户不需要部署数据库、不需要鉴权、不需要连接服务，只需要读取 JSON。
- **第二，易于版本化**。硬件数据随 Git 提交演进，用户可以固定 commit，避免上游变化影响构建。
- **第三，适合嵌入**。编辑器插件、编译器前端、CLI 工具、Web 应用都可以直接打包 JSON。
- **第四，透明**。所有数据都在仓库里，字段结构肉眼可查，不存在黑盒 API。

当然，这也带来一些限制：没有内置索引、没有查询语言、没有数据校验脚本、没有自动生成类型定义。如果使用者面对的是高频查询场景，最好在本地构建索引。例如按`name`、`id`、`vendor`、`architecture` 建立 Map：

```
// 来源：基于 data/all-gpus.json 的推荐使用方式伪代码function buildGpuIndex(gpus) {return {    byId: newMap(gpus.map(g => [g.id, g])),    byName: newMap(gpus.map(g => [g.name.toLowerCase(), g])),    byVendor: gpus.reduce((acc, gpu) => {      (acc[gpu.vendor] ||= []).push(gpu);      return acc;    }, {})  };}function findGpu(index, name) {return index.byName.get(name.toLowerCase());}
```

对一个 AI 编辑器来说，这样的索引可以常驻内存。当用户输入“为 RTX 4090 写一个 FP16 matmul kernel”时，编辑器可以先查出`GeForce RTX 4090` 的字段，再把硬件约束注入生成上下文。

## 十、从数据到产品：它能支撑哪些真实场景

> RightNow GPU Database 的价值可以落到几个具体场景。

- 第一个场景是 AI kernel 代码生成。模型生成 CUDA、OpenCL、Vulkan compute shader 或 SYCL 代码前，可以先检索目标 GPU 能力，避免生成不适配的实现。
- 第二个场景是硬件选型。比如用户要找显存大于 16GB、FP16 算力较高、功耗不超过某个范围的 GPU，就可以直接对 JSON 做过滤。# 来源：基于 data/all-gpus.json 字段的筛选伪代码defrecommend_for_inference(gpus):return [ g for g in gpusif g.get("memorySize", 0) >= 16and g.get("fp16", 0) >= 30and g.get("memoryBandwidth", 0) >= 400 ]
- 第三个场景是技术文章与可视化。因为数据包含`releaseDate`、`processSize`、`transistors`、`dieSize`，可以画出 GPU 工艺演进、晶体管密度变化、显存带宽增长曲线。
- 第四个场景是编译器参数搜索。自动调优器可以根据`sharedMemPerSM`、`registersPerSM`、`maxThreadsPerSM` 缩小搜索空间，而不是盲目枚举所有 tile/block 组合。
- 第五个场景是教学。对刚入门 GPU 编程的工程师来说，这个库把抽象概念和具体产品连接起来：SM 不是课本上的词，显存带宽不是营销数字，warp size 不是孤立常量，它们共同决定 kernel 的性能边界。

## 十一、值得改进的地方：从数据集走向硬件知识库

> 从当前仓库看，它已经完成了最重要的一步：把大量 GPU 规格统一成 JSON。但是如果它要进一步成为 AI 编译器和底层开发工具的核心依赖，还可以补充几个方向。

| 建议项 | 说明 |
| --- | --- |
| **提供 JSON Schema** | 在仓库中增加正式的`schema.json` 文件，定义字段类型、必选性、枚举等，支持下游自动校验及生成 TypeScript/Python 类型定义；当前 README 仅含人类可读示例，缺乏机器可读结构。 |
| **增加字段单位说明** | 明确标注各数值字段单位，如`memorySize`（GB）、`memoryBandwidth`（GB/s）、`processSize`（nm）、`sharedMemPerSM`（Byte）等，避免依赖 README 示例推断，提升文档严谨性与可用性。 |
| **增加数据校验脚本** | 提供脚本（如 Python/Shell）校验数据一致性：`vendor` 限于`nvidia`/`amd`/`intel`；`releaseDate` 统一为`YYYY-MM-DD` 格式；数值字段确保为`number` 类型。 |
| **增加派生索引文件** | 生成辅助索引文件，如`architectures.json`（按架构归类）、`cuda-compute-capabilities.json`（按计算能力分组）、`by-vendor/nvidia.json` 等，便于按需加载子集，提升查询效率。 |
| **补充运行时探测映射** | 提供映射表（如`device-name-mapping.json`），将`nvidia-smi` 输出名、CUDA`device.getName()` 结果与数据库中的`name`、`id` 显式对齐，解决实际设备名与规格库命名不一致问题。 |

这些不是对项目当前形态的否定。相反，正因为底层数据已经足够清晰，才有继续生长为“硬件知识基础设施”的空间。

## 结语：真正的突破，是让硬件知识进入软件闭环

> RightNow GPU Database 最值得关注的地方，是它把 GPU 规格从“人读的资料”变成了“程序可依赖的事实”。这件事在 AI Agent、AI 编译器时代尤其重要。

过去，工程师写 kernel，遇到问题自己查文档、看白皮书、跑 benchmark。如今，AI 代码助手和自动调优器会参与越来越多底层代码生成。

- 如果这些系统没有稳定的硬件知识输入，它们就只能依赖模型参数中的模糊记忆；
- 而一旦接入结构化数据库，它们就能把每一次生成都约束在真实硬件边界内。

::: block-1
这个仓库的简洁也说明了一个常被忽视的道理：基础设施不一定总是庞大的服务、复杂的框架或炫目的模型。有时候，**一份组织良好、字段清晰、可版本化的 JSON 数据集，就能成为上层智能系统的地基。**
:::

RightNow GPU Database 做的正是这件事：把 NVIDIA、AMD、Intel 三家 2,824 款 GPU 的架构、制造、显存、API、性能和 kernel 开发约束汇集到同一张事实地图上。

- 对普通开发者，它是显卡规格查询表；对 AI kernel 工具，它是生成代码前必须阅读的硬件说明书；
- 对未来的编译器系统，它则可能是一层轻量但关键的硬件语义接口。

参考资料

[1] `README.md`:*https://github.com/RightNow-AI/RightNow-GPU-Database/blob/main/README.md*

[2] `data/`:*https://github.com/RightNow-AI/RightNow-GPU-Database/tree/main/data*

[3] `data/nvidia/all.json`:*data/nvidia/all.json*

[4] `data/amd/all.json`:*data/amd/all.json*

[5] `data/intel/all.json`:*data/intel/all.json*

[6] `data/all-gpus.json`:*data/all-gpus.json*

::: related-title
**相关推荐**
:::

::: related-links
- **[暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**
- **[Auto-MicroBench：把 LLM 变成可验证的 GPU 微基准工程师，自动校准模拟器的硬件真相](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903091&idx=1&sn=00b927c466aaca4a983e55ac52b68fe6&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**

:::
