关键词：NVIDIA、**开源 GPU 内核模块**、Linux、:mdnice-bold-italic[GPU 驱动]、**开源实现**、硬件支持

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/01.webp)

- NVIDIA Linux open GPU kernel module source
- https://github.com/NVIDIA/open-gpu-kernel-modules
- https://us.download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/kernel_open.html
- 约 4000 字，阅读 13 分钟，**播客 11 分钟**

相关推荐

- [NVIDIA 技术博客：CUDA 全局内存的合并访问，如何最大化 GPU 带宽](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892826&idx=2&sn=3c703707de4d194bfd297ddc849b095d&scene=21#wechat_redirect)
- [通过微基准测试剖析 NVIDIA Blackwell 架构](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892765&idx=1&sn=cb3750799286bd416436960c4bf97ebb&scene=21#wechat_redirect)
- [PD 多路复用：SGLang 借助 NVIDIA GreenContext 释放高有效吞吐量的大语言模型服务能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892879&idx=1&sn=fdd05fbc29cd1001cd1479034c4b0930&scene=21#wechat_redirect)

> 在 Linux 系统 GPU 驱动领域，NVIDIA 的开源 GPU 内核模块项目无疑是里程碑式的突破。它不仅为开发者提供了透明化的技术实现参考，更通过模块化设计与跨平台兼容能力，让 NVIDIA GPU 在 Linux 环境下的功能拓展与性能优化有了更广阔的空间。

本文将基于 NVIDIA 开源 GPU 内核模块的官方技术文档，从核心架构、构建系统、功能模块到技术细节，学习这一复杂而强大的驱动框架。

## 本文目录

- 零、快速上手：构建、安装与架构支持
  - 构建与安装步骤
  - 支持的目标 CPU 架构
- 一、核心架构：分层设计与模块协同
  - 架构核心组件
  - 核心设计理念
- 二、构建系统：适配多元环境的编译框架
  - 核心构建组件
  - 构建流程解析
- 三、核心内核模块：功能分工与层级依赖
  - 四大模块功能解析
  - 模块协同机制
- 四、GPU 管理：对象模型与生命周期控制
  - OBJGPU 核心对象
  - 生命周期与状态管理
  - 子设备接口
- 五、平台与硬件集成：跨环境兼容的关键保障
  - 芯片集与 PCIe 管理
  - DMA 缓冲共享
- 六、显示与互联：高清显示与高速互联的技术实现
  - DisplayPort 技术实现
  - NVLink 与 NVSwitch 架构
- 七、代码生成与命名：大规模代码库的一致性保障
  - 代码生成机制
  - 命名规范
- 关键问题
  - 问题1: GSP固件加载的安全性验证机制是否足够健壮?
  - 问题2: RPC通信机制的错误恢复策略是否会导致系统状态不一致?
- 总结：开源驱动的价值与意义

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/02.webp)

## 零、快速上手：构建、安装与架构支持

> 作为深入理解 NVIDIA 开源 GPU 内核模块的基础，将先介绍模块的快速构建、安装流程，以及支持的目标 CPU 架构，为后续技术解析提供实操前提。

### 构建与安装步骤

模块构建依赖 Linux 系统环境，执行以下命令即可完成编译：

```
make modules -j$(nproc)
```

安装前需卸载已有的 NVIDIA 内核模块，随后以 root 权限执行安装命令：

```
make modules_install -j$(nproc)
```

需注意，此处构建的内核模块需搭配对应 580.95.05 版本驱动发布的 GSP 固件及用户空间组件。可通过 `.run` 格式驱动文件安装，使用 `--no-kernel-modules` 选项避免重复安装内核模块，示例如下：

```
sh ./NVIDIA-Linux-[...].run --no-kernel-modules
```

### 支持的目标 CPU 架构

当前内核模块支持 x86_64 与 aarch64 两种架构。若需交叉编译，需在 `make` 命令行中指定以下变量：

- `TARGET_ARCH`：取值为 aarch64 或 x86_64
- 编译工具链变量：CC、LD、AR、CXX、OBJCOPY

交叉编译示例（在 x86_64 环境为 aarch64 架构编译）：

```
make modules -j$(nproc)         \    TARGET_ARCH=aarch64         \    CC=aarch64-linux-gnu-gcc    \    LD=aarch64-linux-gnu-ld     \    AR=aarch64-linux-gnu-ar     \    CXX=aarch64-linux-gnu-g++   \    OBJCOPY=aarch64-linux-gnu-objcopy
```

## 一、核心架构：分层设计与模块协同

> NVIDIA 开源 GPU 内核模块采用分层架构，清晰划分了用户空间、内核空间与硬件层的交互边界，确保各组件各司其职又高效协同。

### 架构核心组件

- **硬件层**：以 NVIDIA GPU 硬件为基础，涵盖 Turing 及后续架构的 GeForce RTX 系列、专业 RTX、Quadro 等全系列 GPU，支持 x86_64 与 aarch64 架构。
- **内核空间**：四大核心内核模块构成驱动主体，通过层级依赖关系实现功能联动，同时集成资源管理、内存管理等关键子系统。
- **用户空间**：包含应用程序与 `libGL.so`、`libcuda.so` 等库文件，通过标准化接口与内核模块通信，获取 GPU 硬件加速、显示控制等能力。

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/03.webp)

::: image-caption
NVIDIA GPU 驱动架构图各组件的交互流程，分用户空间（应用、库、GSP 固件）、内核空间（主驱动、统一内存、显示、DRM 驱动）、硬件（NVIDIA GPU）三层
:::

### 核心设计理念

分层架构的核心优势在于解耦与扩展性。

内核模块作为硬件与用户空间的中间层，屏蔽了不同 GPU 型号的硬件差异，为上层应用提供统一的调用接口；而模块化设计则让功能迭代更灵活，例如新增显示协议支持时，仅需扩展对应模块即可，无需改动核心驱动逻辑。

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/04.webp)

::: image-caption
NVIDIA 开放 GPU 内核模块由多个相互连接的组件构成，涵盖了从用户空间应用程序到硬件接口的各个层面。其架构采用分层方法，责任划分清晰
:::

## 二、构建系统：适配多元环境的编译框架

> 驱动的跨内核版本兼容性离不开强大的构建系统，NVIDIA 开源模块基于 Linux 内核的 Kbuild 基础设施，打造了一套自动化、自适应的编译解决方案。

### 核心构建组件

- **Kbuild 控制器**：以 `kernel-open/Kbuild` 为核心，定义编译参数、编译器标志与模块依赖关系，协调各模块的编译流程，是构建系统的"指挥中心"。
- **兼容性测试框架**：`conftest.sh` 脚本通过 200 余项测试，探测内核 API 特性、头文件存在性等关键信息，生成适配当前内核环境的兼容性头文件，确保驱动在 Linux 4.15 及以上版本均可正常编译。
- **版本管理机制**：通过 `version.mk` 与 `nvBldVer.h` 实现版本信息统一管理，确保内核模块与用户空间组件、GSP 固件的版本一致性。

### 构建流程解析

构建过程遵循"配置-测试-编译-安装"的标准化流程：先通过 `conftest.sh` 完成内核环境探测与配置生成，再由 Kbuild 控制器执行模块化编译，最后通过 `make modules_install` 完成安装。

整个流程支持多线程编译与增量构建，大幅提升开发与部署效率。

## 三、核心内核模块：功能分工与层级依赖

> 四大核心内核模块是驱动功能的载体，它们以层级依赖的方式加载，共同实现 GPU 的完整功能支持。

### 四大模块功能解析

1. **nvidia.ko（主驱动模块）**：驱动的核心基础，负责 GPU 初始化、资源管理、内存分配与 PCIe 总线交互，是所有其他模块的依赖基础。
2. **nvidia-modeset.ko（显示管理模块）**：专注于显示控制，实现模式设置、分辨率调整、DisplayPort/HDMI 等接口管理，对接内核模式设置（KMS）子系统。
3. **nvidia-drm.ko（DRM 接口模块）**：实现与 Linux DRM（直接渲染管理器）子系统的深度集成，支持原子模式设置，适配现代 Linux 桌面环境。
4. **nvidia-uvm.ko（统一内存模块）**：提供 CPU 与 GPU 之间的统一虚拟内存支持，实现内存页自动迁移，为 CUDA 应用提供高效的内存管理能力。

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/05.webp)

### 模块协同机制

模块加载遵循严格的层级顺序：

- `nvidia.ko` 首先加载，为后续模块提供核心资源
- `nvidia-modeset.ko` 依赖 `nvidia.ko` 加载
- `nvidia-drm.ko` 又依赖 nvidia-modeset.ko
- 而 `nvidia-uvm.ko` 直接依赖主驱动模块

这种依赖关系确保了资源的有序分配与功能的正常联动。

## 四、GPU 管理：对象模型与生命周期控制

> GPU 的硬件抽象与生命周期管理是驱动的核心能力之一，NVIDIA 通过 OBJGPU 对象模型与子设备接口，实现了对 GPU 硬件的精细化控制。

### OBJGPU 核心对象

OBJGPU 是物理 GPU 在驱动中的抽象表示，包含芯片信息、总线信息、设备状态等完整数据，是所有 GPU 操作的协调中心。

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/06.webp)

它通过指针关联多个子对象，涵盖内存管理、显示控制、总线交互等专项功能，形成"核心对象+子系统"的管理架构。

### 生命周期与状态管理

GPU 的生命周期分为初始化、运行、卸载、销毁四个阶段，通过 `gpuStatePreInit()`、`gpuStateLoad()` 等接口实现状态跃迁。

![文章插图](/articles/2025-10-28-linux-gpu-nvidia-gpu-05146aad/images/07.webp)

驱动会实时跟踪 GPU 的电源状态、错误状态等关键信息，当出现故障时，通过 `gpuSetStateResetRequired()` 等接口触发重置或恢复操作，保障系统稳定性。

### 子设备接口

Subdevice 类为多 GPU 环境提供了精细化的设备控制能力，支持 GPU 功耗设置、ECC 状态查询、NVLink 能力获取等专项操作，让客户端应用可直接与特定 GPU 进行交互。

## 五、平台与硬件集成：跨环境兼容的关键保障

> 驱动的广泛兼容性依赖于完善的平台集成机制，NVIDIA 通过芯片集管理、PCIe 配置与 DMA 缓冲共享等技术，实现了对不同硬件平台的深度适配。

### 芯片集与 PCIe 管理

OBJCL（核心逻辑对象）作为平台集成的核心，负责芯片集检测、PCIe 配置空间访问与 ASPM（主动状态电源管理）控制。驱动内置了 Intel、AMD、NVIDIA 等主流芯片集的适配逻辑，通过平台特定工作区解决兼容性问题，确保在工作站、嵌入式等不同设备上的稳定运行。

### DMA 缓冲共享

通过 Linux dma-buf 框架，驱动支持 GPU 内存与其他设备的共享访问。`nv_dma_buf_export()`、`nv_dma_buf_map()`等接口实现了内存的导出、映射与释放，为视频编解码、图形渲染等跨设备协作场景提供了高效的数据传输通道。

## 六、显示与互联：高清显示与高速互联的技术实现

> 显示输出与 GPU 互联是 GPU 的核心应用场景，NVIDIA 开源模块通过 DisplayPort、NVLink 与 NVSwitch 技术，构建了完整的显示与互联解决方案。

### DisplayPort 技术实现

以 ConnectorImpl 类为核心，支持单流（SST）与多流（MST）传输模式，实现设备自动发现、EDID 读取、链路速率协商等功能。驱动内置了丰富的兼容性工作区，可解决不同显示器的 EDID 异常、DSC（显示流压缩）兼容等问题，支持高清显示与 HDCP 加密。

### NVLink 与 NVSwitch 架构

NVLink 作为 GPU 间高速互联技术，通过 nvlink 核心库实现设备注册、链路训练与状态管理，支持 GPU 间的高带宽数据传输。NVSwitch 则提供了复杂的交换矩阵，LR10 型号支持 36 条链路与 4 条通道，可构建大规模 GPU 集群的互联拓扑，满足 AI 训练、高性能计算等场景的需求。

## 七、代码生成与命名：大规模代码库的一致性保障

> NVIDIA 开源 GPU 内核模块代码量庞大，通过标准化的代码生成机制与命名规范，确保了代码库的可维护性与一致性。

### 代码生成机制

基于 NVOC（NVIDIA 面向对象 C 语言）系统，自动生成类定义、方法表与虚拟函数表，减少重复代码编写。build 过程中，`conftest.sh` 与 `version.mk` 会生成兼容性头文件与版本信息，确保代码对不同内核环境的适配。

### 命名规范

遵循严格的命名规则：GPU 对象以"OBJGPU"为前缀，HAL 函数以"hal"开头，生成文件采用"g_*_nvoc.h"格式，核心头文件以"nv"为前缀。统一的命名规则让开发者可快速识别代码功能，降低维护成本。

## 关键问题

### 问题1: GSP固件加载的安全性验证机制是否足够健壮?

从代码实现来看，GSP固件加载过程涉及多个签名验证路径但存在以下潜在问题:

1. **条件分支的复杂性**: 代码根据`kgspIsDebugModeEnabled_HAL`和`PDB_PROP_CONFCOMPUTE_CC_FEATURE_ENABLED`选择不同的固件镜像，这种多路径选择增加了攻击面。
2. **签名验证的时序**: 在`kfspPrepareAndSendCotPayload`中，固件镜像、哈希、签名和公钥是分别提取和验证的，如果在这些步骤之间存在TOCTOU(Time-of-check to time-of-use)漏洞,攻击者可能利用竞态条件替换固件。
3. **内存保护标志**: 代码使用`MEMDESC_FLAGS_ALLOC_IN_UNPROTECTED_MEMORY`标志，这意味着GSP-FMC镜像放置在未受保护的系统内存中，在SEV环境下可能被恶意访问。

> 当前的多层HAL(Hardware Abstraction Layer)架构是否引入了不必要的复杂性，从而削弱了安全验证的可靠性?

关于TOCTOU漏洞的担忧问题：代码中的固件加载流程实际上**已经考虑了TOCTOU问题**。在`kfspPrepareAndSendCotPayload`中，固件镜像、哈希、签名和公钥的提取是通过`bindataArchiveGetStorage`和`bindataWriteToBuffer`完成的，这些操作是**原子性地将数据复制到内存描述符** 中。

关键保护机制:

1. **一次性复制**: 固件镜像在`memdescMap`后立即通过`bindataWriteToBuffer`写入,然后`memdescUnmap` (src/nvidia/src/kernel/gpu/fsp/arch/hopper/kern_fsp_gh100.c:1042-1052)
2. **物理地址锁定**: COT payload中存储的是`memdescGetPhysAddr`返回的物理地址 (src/nvidia/src/kernel/gpu/fsp/arch/hopper/kern_fsp_gh100.c:1054)，一旦设置就不会改变。

关于未保护内存的担忧上，`MEMDESC_FLAGS_ALLOC_IN_UNPROTECTED_MEMORY`标志的使用是**有意为之且必要的** (src/nvidia/src/kernel/gpu/fsp/arch/hopper/kern_fsp_gh100.c:1001-1005)。其中，注释明确说明:"在启用SEV的系统上，GSP-FMC镜像必须能被FSP(GPU内部单元)访问，因此放置在未保护的系统内存中"。

这不是安全漏洞，因为FSP会验证签名和哈希 (src/nvidia/src/kernel/gpu/fsp/arch/hopper/kern_fsp_gh100.c:1056-1067)，即使攻击者修改内存，签名验证也会失败。

---

### 问题2: RPC通信机制的错误恢复策略是否会导致系统状态不一致?

GSP的RPC通信是CPU-RM和GSP-RM之间的关键接口，但错误处理存在以下隐患：

1. **超时计数器的单调递增**：代码中提到`_kgspRpcIncrementTimeoutCount`，但没有看到超时计数器的重置机制。若RPC超时累积，可能导致系统过早进入错误状态。
2. **致命错误的传播**：`kgspServiceFatalHwError`和`kgspRcAndNotifyAllChannels_IMPL`会通知所有通道，但若某些通道已在处理其他错误，这种全局通知可能导致状态机混乱。
3. **事件处理的原子性**：在`_kgspProcessEccNotifier`中，ECC错误处理涉及内存黑名单操作，但如果在`heapStorePendingBlackList`执行过程中发生RPC超时或GSP重置，可能导致内存管理状态不一致。
4. **多种启动模式的状态转换**：GSP支持`KGSP_BOOT_MODE_NORMAL`、`KGSP_BOOT_MODE_SR_RESUME`、`KGSP_BOOT_MODE_GC6_EXIT`等多种启动模式，但代码中没有明确的状态机来管理这些模式之间的转换，尤其在错误恢复场景下。

> 当GSP-RM发生致命错误需要重启时，CPU-RM如何保证所有待处理的RPC请求都被正确清理，以及如何防止重启后的GSP-RM接收到过时的RPC消息？

#### 超时计数器的重置机制

代码**确实有超时计数器重置机制** (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:2382)。在`_kgspRpcRecvPoll`中，当RPC成功完成时，`pRpc->timeoutCount = 0`会重置计数器。

超时处理流程:

1. 检测超时后递增计数器 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:2328-2332)
2. 达到阈值(`RPC_TIMEOUT_GPU_RESET_THRESHOLD`)后标记GPU重置 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:2339-2345)
3. 成功的RPC会重置计数器 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:2382)

#### GSP重启时的RPC清理

代码通过**RPC sanity check机制** 确保不会处理过时的RPC (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:276-322)。

关键保护:

1. **致命错误标志**: 当`pKernelGsp->bFatalError`为真时，所有新RPC都会被拒绝并返回`NV_ERR_RESET_REQUIRED` (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:279-290)
2. **递归轮询保护**: `bPollingForRpcResponse`标志防止递归RPC调用 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:2242-2243)
3. **意外事件处理**: 超时后到达的响应会被识别为"意外事件"并记录，但不会导致状态混乱 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:1598-1610)

#### 启动模式的状态管理

虽然代码中没有显式的状态机图，但启动模式通过**boot arguments传递** (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:3638)，并且在`_kgspBootGspRm`中有重试逻辑 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:3919-3945)。重试机制会在失败时重置线程状态超时 (src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:3922-3924)，确保每次尝试都有完整的超时窗口。

## 总结：开源驱动的价值与意义

> NVIDIA 开源 GPU 内核模块不仅是技术上的突破，更体现了开源协作的理念。

- 通过开放驱动源码，NVIDIA 为开发者提供了深入了解 GPU 工作原理的窗口，促进了 Linux 生态中 GPU 应用的创新；
- 模块化设计与跨平台兼容能力，让驱动可适配从桌面到数据中心的各类场景；
- 而完善的文档与标准化的代码规范，也为社区参与开发与维护提供了便利。

未来，随着 GPU 技术的不断演进，这一开源驱动框架将持续迭代，为 AI、图形渲染、高性能计算等领域提供更强大的底层支持，推动 Linux 平台 GPU 应用生态的持续发展。

更多推荐

- [打破 NVIDIA 封闭壁垒！这篇 16 年前的论文，如何用 PTX 仿真库解锁 GPU 通用计算？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893026&idx=1&sn=b9d66b6bcb97cf5d6c2eb878bc0e47cf&scene=21#wechat_redirect)
- [一次编译，多平台运行！GPU 二进制文件兼容性在NVIDIA、AMD、Intel 和 Tenstorrent上的探索！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889722&idx=1&sn=ee787fcb0d896221430035a780fb5dff&scene=21#wechat_redirect)
- [汇编级 NVIDIA 与 AMD GPU 代码转换新 SOTA！跨架构转换方案 CASS，远超商业转换基线，数据代码全开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889091&idx=1&sn=a50fd9383f13b3c1ab7b6d3393369123&scene=21#wechat_redirect)
