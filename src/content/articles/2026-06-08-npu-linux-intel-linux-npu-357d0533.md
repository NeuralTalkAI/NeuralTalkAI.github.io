关键词：**Intel NPU**、Level Zero、**ivpu 内核驱动**、:mdnice-bold-italic[命令缓冲]、模型缓存

> 自从 Meteor Lake 把 NPU（Neural Processing Unit）这块异构加速器塞进消费级 CPU 之后，"AI PC" 这个词就开始频繁出现在各种发布会上。但很少有人去看一眼：:mdnice-bold-italic[当你在笔记本上跑一个 OpenVINO 推理时，从 Python 调用到 NPU 上电运算，中间到底经过了多少层代码？这个过程里，是谁在生成命令缓冲、谁在做模型缓存、谁负责把 ELF 形态的 graph 翻译成 NPU 能吃进去的指令？]

![文章插图](/articles/2026-06-08-npu-linux-intel-linux-npu-357d0533/images/01.webp)

- **Intel® NPU (Neural Processing Unit) Driver**
- :mdnice-bold-italic[英特尔®NPU设备是集成在英特尔客户端处理器中的AI推理加速器，搭载起始于英特尔®酷睿Ultra系列处理器（原研发代号Meteor Lake），能够以高能效完成人工神经网络相关运算任务。]
- 项目代码：https://github.com/intel/linux-npu-driver
- 文档：https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 A100：拆解摩尔线程开源的 GPU 驱动级基准测试 gpu-compute-driver-bench](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901875&idx=1&sn=3db0fb835b96a96286fc1e65ef6492e2&scene=21#wechat_redirect)**
- **[赋能 Linux GPU 生态：NVIDIA 开源 GPU 内核模块的架构、功能与技术实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893566&idx=1&sn=1cadc8c260f80856125b710ce6bc4249&scene=21#wechat_redirect)**
- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**

:::

> Intel 把答案完整地放在了 `intel/linux-npu-driver`[1] 这个仓库里——一份以 91.8% C++、6% C 写就的、连接 oneAPI Level Zero 用户态接口与 `intel_vpu` 内核模块的工业级驱动。

本文以源码为线索，带你穿越四层抽象，看看**一次 NPU 推理是如何被"组装、调度、提交、回收"的。**

## 本文目录

- 快速上手：30 秒跑通驱动
- 一、为什么需要这样一份驱动？
- 二、四层抽象：从 Python 到固件
  - 2.1 一次 NPU 推理的"七步走"
- 三、入口与生命周期：`Driver` 单例怎么初始化
- 四、命令系统：`VPUCommand` 与 `VPUJob` 的两级抽象
  - 4.1 `VPUCommand`：每条命令的最小单元
  - 4.2 `VPUJob`：把命令组装成可提交的批次
  - 4.3 `VPUCommandBuffer`：硬件可见的 64 字节对齐结构
- 五、推理执行：`VPUInferenceExecute` 与 ELF 解析
  - 5.1 Mutable Command List：不重录命令也能换输入
- 六、磁盘缓存：让二次编译只剩 hash 命中
  - 6.1 Hash Key：保证语义一致
  - 6.2 LRU 淘汰：用 `st_atim` 而不是 `mtime`
- 七、命令队列与 fence：最终把 Job 投到内核
- 八、跨代硬件的优雅抽象
- 九、结语：当一颗 NPU 真正"被 Linux 看见"

![文章插图](/articles/2026-06-08-npu-linux-intel-linux-npu-357d0533/images/02.webp)

## 快速上手：30 秒跑通驱动

> 下面这套命令源自仓库的 `docs/overview.md`[2]，是把驱动从源码编译到加载的最短路径。具体平台支持、Wildcat Lake / Panther Lake / Lunar Lake 的差异、Ubuntu 24.04 的 PPA snapshot 包用法，请参考 docs/overview.md[3] 与 release 页面[4]。

```
# 1. 安装基本依赖（Ubuntu 22.04+）sudo apt updatesudo apt install -y build-essential git git-lfs cmake python3# 2. 拉源码与子模块git clone https://github.com/intel/linux-npu-driver.gitcd linux-npu-drivergit submodule update --init --recursive# 3. 编译并安装用户态驱动 + 固件cmake -B build -S .cmake --build build --parallel $(nproc)sudo cmake --install build# 4. 重新加载内核模块以匹配新固件sudo rmmod intel_vpusudo modprobe intel_vpu# 5. 自检设备节点是否就绪ls /dev/accel/accel0   # 期望存在该节点./build/bin/npu-umd-test   # 跑功能测试，期望仅 ConfigurationCheck 因缺 yaml 而失败
```

如需同时构建 `libnpu_driver_compiler.so`（即启用 Compiler-in-Driver），请添加编译选项 `-DENABLE_NPU_COMPILER_BUILD=ON`，并确保已安装 OpenVINO 的构建依赖。

关于具体构建步骤，以及处理如 `-fcf-protection=full` 不被识别等边界情况，请参考以下两处文档：

- `compiler/npu_compiler.cmake`[5]
- docs/overview.md[6] 中的 **"Building a driver together with the compiler"** 小节。

> 下面是两个典型使用场景：

- **想用 Perfetto 给驱动打 trace、或开启 `ZE_INTEL_NPU_LOGPERFETTO=system` 抓系统调用栈**，看 `docs/UMD_overview.md`[7]；
- **想跑功能压测、配置 YAML、跑 ImageNet 精度比对**，看 `validation/umd-test/configs/README.md`[8]。

## 一、为什么需要这样一份驱动？

> NPU 不是显卡，也不是普通的 PCIe 加速卡。:mdnice-bold-italic[它是从 Meteor Lake 开始集成进 Intel 客户端 CPU 的"AI 协处理器"，在 Linux 内核里被一个叫 `intel_vpu`（旧名 VPU - Versatile Processing Unit）的内核模块管理，对外暴露 `/dev/accel/accel0` 节点。]

换句话说，**你想让 Python 端的 OpenVINO 调起 NPU，就必须有一条软件链路把 oneAPI Level Zero 的 API 调用** ——比如 `zeCommandQueueExecuteCommandLists`——一路翻译成 NPU 固件能识别的命令缓冲（command buffer），再通过 DRM ioctl `DRM_IOCTL_IVPU_SUBMIT` 投喂给硬件。

这条链路就是 `linux-npu-driver` 干的事。它不仅要做协议翻译，还要解决三个工程难题：

::: block-1
- **跨代硬件兼容**：MTL/LNL/PTL/ARL/WCL 多代 NPU 共用一套驱动，在 `umd/vpu_driver/source/device/` 下并存 `vpu_37xx`、`vpu_40xx`、`vpu_50xx` 三套 HW family 实现。
- **编译开销摊销**：OpenVINO 的 IR（XML+BIN）首次进入驱动会被 Compiler-in-Driver 编成 NPU 可执行的 ELF blob，编译动辄数秒甚至十数秒，必须做磁盘缓存。
- **低延迟调度**：推理是个高频小任务场景，命令构造、内存映射、fence 等待这些路径上的每一微秒都要算清楚。
:::

## 二、四层抽象：从 Python 到固件

> 仓库 `docs/overview.md`[9] 用 ASCII art 给出了完整软件栈，简化后大致如下：

```
[ 用户应用 / OpenVINO Python ]            │[ libze_loader.so  oneAPI Level-Zero 加载器 ]            │[ libze_intel_npu.so  ── 用户态驱动 (UMD) ]        ↕[ libnpu_driver_compiler.so  Compiler-in-Driver ]            │[ /dev/accel/accel0  ── intel_vpu.ko 内核模块 (KMD) ]            │[ NPU Firmware (intel-fw-npu) + 硬件 ]
```

这里仓库自身仅负责中间的 UMD + Compiler 集成 + 固件分发：

- `umd/level_zero_driver/`：实现 Level Zero spec 的 NPU 后端，把 `ze_command_queue_handle_t` 这类抽象句柄映射到 NPU 概念。
- `umd/vpu_driver/`：硬件无关的内部驱动核心，处理命令构造、buffer object、设备能力探测。
- `compiler/`：CMake 拉取并构建 OpenVINO 仓库中的 NPU Compiler。
- `firmware/`：固件二进制文件。
- `validation/`：超过万行的功能测试，含 `npu-umd-test`、`npu-kmd-test`、Mutable Cmd List 测试等。

### 2.1 一次 NPU 推理的"七步走"

> 把这四层贯穿起来，一次推理大致经过七个步骤：

```
用户                         UMD / KMD                NPU │ ① zeInit            → Driver::driverInit() │ ② zeGraphCreate     → DiskCache → 命中? │                        └─ MISS: Compiler 编译 │ ③ zeCmdListCreate   → new VPUJob │ ④ AppendGraphExec   → VPUInferenceExecute │ ⑤ zeCmdListClose    → VPUJob::closeCommands() │                        切分为多个 CmdBuffer │ ⑥ ExecCmdLists      → vpuQueue->submit() │                        └─ ioctl SUBMIT  ─────► 执行 │ ⑦ Synchronize       → waitForCompletion ◄──── 完成 ▼                                                ▼
```

1. 用户调用 `zeInit/zeDriverGet` → UMD 探测 `/dev/accel`，初始化 `Driver` 单例。
2. 用户调用 `zeGraphCreate` → UMD 计算 hash 命中 `DiskCache`，未命中则交给 Compiler 把 IR 编译成 ELF blob。
3. 用户调用 `zeCommandListCreate` → UMD 创建 `VPUJob` 容器。
4. 用户 `zeAppendGraphExecute` → UMD 构造 `VPUInferenceExecute` 命令并 `appendCommand`。
5. 用户 `zeCommandListClose` → `VPUJob::closeCommands()` 把命令打包进 `VPUCommandBuffer`。
6. 用户 `zeCommandQueueExecuteCommandLists` → UMD 通过 `VPUDeviceQueue::submit()` 走 ioctl 进内核。
7. 用户 `zeCommandQueueSynchronize` → UMD 在 fence 上等待 KMD 报告 job 完成。

> 下面我们就沿着这条链路逐一拆解最关键的三个抽象。

## 三、入口与生命周期：`Driver` 单例怎么初始化

整个 UMD 的入口在 `umd/level_zero_driver/source/driver.cpp`[10]。

它做的事很直白——读环境变量、初始化日志、探测 OS、扫描 NPU 设备、缓存初始化——但用了 `std::call_once` 来保证多线程下只跑一次。

```
// 来源：umd/level_zero_driver/source/driver.cpp#L98-L124void Driver::driverInit(ze_init_flags_t flags) {    std::call_once(this->initDriverOnce, [&]( "&") {        initializeEnvVariables();                        // 读 ZE_AFFINITY_MASK 等        if (osInfc == nullptr) {            osInfc = VPU::getOsInstance();               // 拿到 Linux/Android 的 OS 抽象            if (osInfc == nullptr) {                LOG_E("Failed to initialize (OS interface is null)");                return;            }        }        diskCache = std::make_unique<DiskCache>(*osInfc);    // 初始化磁盘 blob 缓存        auto vpuDevices = VPU::DeviceFactory::createDevices(osInfc);  // 扫描 /dev/accel        if (!vpuDevices.empty()) {            pGlobalDriverHandle = DriverHandle::create(std::move(vpuDevices));            if (pGlobalDriverHandle == nullptr) {                LOG_W("Failed to initialize (GlobalDriverHandle is null)");                return;            }            initStatus = ZE_RESULT_SUCCESS;              // 至此驱动就绪        }        displayComponentVersions();    });}
```

这里有几个细节值得指出：

- **OS 接口被抽象**。`VPU::getOsInstance()` 返回一个 `OsInterface`，封装了 `osiCreateDirectories`、`osiOpenWithSharedLock` 等文件操作。这让磁盘缓存模块完全无需关心是 Linux 还是 Android（Android 路径走 `/data/data/<progname>/ze_intel_npu_cache`）。
- **多卡的预留**。`vpuDevices` 是 vector，意味着架构上支持多 NPU 设备——尽管目前消费级 CPU 都只有一个。
- **失败但不抛异常**。Level Zero spec 里这条路径不允许抛异常，所以失败后 `initStatus` 保持错误码，后续 `getInitStatus()` 会把它返给上层。

## 四、命令系统：`VPUCommand` 与 `VPUJob` 的两级抽象

> 下面是两级抽象向的示意图，具体后文介绍：

```
ze_command_list_t          ze_command_queue_t       │                          │       ▼                          ▼┌──────────────┐         ┌─────────────────┐│   VPUJob     │ submit  │ VPUDeviceQueue  ││  commands[]  ├────────►│  → ioctl 进 KMD ││  cmdBuffers[]│         └─────────────────┘└──────┬───────┘       │ closeCommands() 按 Synchronize 切分       ▼┌─────────────────────────────────────────┐│ VPUCommand (基类, std::any command)      ││  ├─ VPUInferenceExecute   推理 (可 update)││  ├─ VPUCopyCommand        DMA 拷贝       ││  ├─ VPUFillCommand        内存填充       ││  ├─ VPUBarrierCommand     栅栏 (Sync)    ││  ├─ VPUEventCommand       signal / wait ││  ├─ VPUTimestampCommand   时间戳         ││  └─ VPUQueryCommand       Metric        │└─────────────────────────────────────────┘
```

### 4.1 `VPUCommand`：每条命令的最小单元

NPU 固件吃的是一条条带头部的二进制命令（`vpu_cmd_header_t`），UMD 把它们抽象成 `VPUCommand` 基类。子类包括 `VPUInferenceExecute`、`VPUCopyCommand`、`VPUFillCommand`、`VPUBarrierCommand`、`VPUTimestampCommand` 等等，对应一一 Level Zero 中的不同 append API。

```
// 来源：umd/vpu_driver/source/command/command.hpp#L35-L100class VPUCommand {public:    enumclass ScheduleType {        Generic,        // 普通命令，可与后续命令同批        Synchronize,    // 同步命令，强制结束当前 command buffer    };    size_t getCommitSize() const {        constvpu_cmd_header_t *hdr = this->getHeader();        if (!hdr) return0;        returnstatic_cast<size_t>(hdr->size);  // 命令在最终 buffer 中占多大    }    // 关键能力：MutableCommandList 扩展用，允许在 close 后改写参数    using ArgumentUpdatesMap =        std::unordered_map<uint32_t, ArgumentUpdates>;    virtual bool setUpdates(const ArgumentUpdatesMap &updatesMap) {        LOG_E("Command with type %#x does not support changing arguments",              getCommandType());        returnfalse;    }    virtual bool update(VPUCommandBuffer *commandBuffer) { returntrue; }protected:    virtual const vpu_cmd_header_t *getHeader() const { returnnullptr; }    std::any command = {};                  // 存放具体命令结构体    bool cmdNeedsUpdate = false;    // ...省略 buffer object 关联与描述符相关方法};
```

注意 `std::any command` 这一笔——它让 `VPUCommand` 不必为每种命令派生一个独立成员，而是把固件 ABI 里的 `vpu_cmd_inference_execute_t`、`vpu_cmd_copy_buffer_t` 等结构体直接塞进 `any`，由子类 `getHeader()` 用 `std::any_cast` 取出。

这种"类型擦除 + 派生覆写"的组合，比维护一个庞大的 union 干净得多。

### 4.2 `VPUJob`：把命令组装成可提交的批次

`VPUJob` 是 `VPUCommand` 的容器，对应 Level Zero 里的 `ze_command_list_handle_t`。它的核心责任是 `closeCommands()`：根据"同步命令切分"原则，把命令切分到一个或多个 `VPUCommandBuffer` 中。

```
// 来源：umd/vpu_driver/source/command/job.cpp#L33-L65bool VPUJob::closeCommands() {    if (ctx == nullptr) {        LOG_E("VPUDeviceContext is nullptr");        returnfalse;    }    if (isClosed()) returntrue;    if (cmdBuffers.size()) {        LOG_E("Failed to close the job because of dirty state");        returnfalse;    }    LOG(VPU_JOB, "Schedule commands, number of commands %lu", commands.size());    std::shared_ptr<VPUBufferObject> lastEventBo;    for (auto it = commands.begin(); it != commands.end();) {        // 按"同步命令"为界，切出 [it, next) 这一段        auto next = scheduleCommands(it);        // 为这一段分配并填充一个独立的 VPUCommandBuffer        if (!createCommandBuffer(it, next, lastEventBo,                                 next == commands.end() ? true : false)) {            returnfalse;        }        it = next;    }    closed = true;    returntrue;}
```

`scheduleCommands` 的策略简单但有效：遇到 `isSynchronizeCommand()` 就把当前段截断，下一条命令开新 buffer。这背后是硬件约束——某些命令必须独立成 buffer 以便 fence 同步。

而 `createCommandBuffer` 会在多 buffer 间织入一个隐式的"前一个 buffer 的 self-signal → 后一个 buffer 的 head-wait"链条，确保 NPU 严格顺序执行：

```
// 来源：umd/vpu_driver/source/command/job.cpp#L108-L129bool VPUJob::createCommandBuffer(...) {    auto cmdBuffer = VPUCommandBuffer::allocateCommandBuffer(ctx, begin, end);    if (cmdBuffer == nullptr) returnfalse;    if (lastEventBo) {        cmdBuffer->addWaitAtHead(lastEventBo);    // 等待上一段完成    }    if (!last) {        lastEventBo = cmdBuffer->getBuffer();        cmdBuffer->addSelfSignalAtTail();         // 自己结尾打信号给下一段    }    cmdBuffers.emplace_back(std::move(cmdBuffer));    returntrue;}
```

### 4.3 `VPUCommandBuffer`：硬件可见的 64 字节对齐结构

> 最终送给固件的不是 C++ 对象，而是一段连续内存。

```
低地址                                          高地址┌──────────────────────────────────┐ ◄─ 64B ★│ contextSaveArea (抢占现场)         │├──────────────────────────────────┤ ◄─ 64B ★│ fenceValue + reserved[7]         │├──────────────────────────────────┤ ◄─ 64B ★│ vpu_cmd_buffer_header (固件入口)  ││ internalSync[2]  (wait + signal) │├──────────────────────────────────┤│ commandList[0]  ── 柔性数组       ││   inference_execute / copy /     ││   timestamp / ... 顺序排列        │├──────────────────────────────────┤ ◄─ 64B│ Descriptors (扩展描述符)          │└──────────────────────────────────┘★ static_assert(offset % 64 == 0) 编译期约束
```

VPUCommandBuffer 它的布局在 `umd/vpu_driver/source/command/command_buffer.hpp`[11] 里被锁死成下面这个 `CommandHeader` 结构。每个字段都精确对齐到 64 字节（NPU 固件 cache line 大小）：

```
// 来源：umd/vpu_driver/source/command/command_buffer.hpp#L113-L130struct CommandHeader {    uint8_t  contextSaveArea[VPU_CONTEXT_SAVE_AREA_SIZE]; // 上下文保存区    VPUEventCommand::KMDEventDataType fenceValue;         // 内部 fence 值    uint64_t reserved[7];    vpu_cmd_buffer_header header;                         // 命令缓冲头    vpu_cmd_fence_t internalSync[2];                      // 同步命令对（wait+signal）    uint8_t commandList[0];                               // 命令流（柔性数组）    /* 末尾还会拼接 descriptors，需 64 字节对齐 */};static_assert(offsetof(CommandHeader, header) % 64 == 0, "...");static_assert(offsetof(CommandHeader, contextSaveArea) % 64 == 0, "...");static_assert(offsetof(CommandHeader, fenceValue) % 64 == 0, "...");
```

`static_assert` 在编译期就保证了所有关键偏移必须 64 字节对齐——这种"对齐合同写进类型系统"的写法，是高性能驱动里非常优雅的一笔，免得运行时偶然踩雷。

## 五、推理执行：`VPUInferenceExecute` 与 ELF 解析

> 最有"AI 味道"的命令当属 **`VPUInferenceExecute`，它对应 `zeAppendGraphExecute`，本质是告诉 NPU："用这个编译好的 ELF blob 做一次推理**，输入在这几个 buffer，输出写到那几个 buffer"。

```
// 来源：umd/vpu_driver/source/command/inference_execute.cpp#L42-L64VPUInferenceExecute::VPUInferenceExecute(...) {    vpu_cmd_inference_execute_t cmd = {};    cmd.header.type = VPU_CMD_INFERENCE_EXECUTE;    cmd.header.size = sizeof(vpu_cmd_inference_execute_t);    cmd.inference_id = inferenceId;    // 把 host-parsed inference 这个 ELF 解析后的 buffer 地址写进命令    cmd.host_mapped_inference.address = bos.at(0)->getVPUAddr();    cmd.host_mapped_inference.width   = safe_cast<uint32_t>(bos.at(0)->getAllocSize());    command.emplace<vpu_cmd_inference_execute_t>(cmd);    appendAssociateBufferObject(bos);            // 关联模型 BO    userArgIndex = getAssociateBufferObjects().size();    appendAssociateBufferObject(userBos);        // 关联用户输入/输出 BO    userArgHandles.reserve(userBos.size());    for (constauto &bo : userBos)        userArgHandles.emplace_back(bo->getHandle());    LOG(VPU_CMD, "VPUInferenceExecute command created - hpi: %p, inference_id: %lu",        hpi.get(), inferenceId);}
```

`hpi` 全称 `HostParsedInference`，是 ELF 解析器（`L0::ElfParser`）从编译产物中解析出来、进一步在 host 侧做过预处理的对象。所谓"NPU 上跑模型"，在这里被精炼成了**两个 64 位地址**：模型 ELF 的 VPU 地址 + 用户 buffer 的 VPU 地址。

### 5.1 Mutable Command List：不重录命令也能换输入

> 支持 `update()` 的命令最酷的是它们让 zeMutableCommandList[12] 这条扩展能够生效：

```
// 来源：umd/vpu_driver/source/command/inference_execute.cpp#L139-L166bool VPUInferenceExecute::update(VPUCommandBuffer *commandBuffer) {    if (!cmdNeedsUpdate) returntrue;    cmdNeedsUpdate = false;    // 用新输入/输出参数重新走一次 ELF 参数应用    std::vector<std::shared_ptr<VPUBufferObject>> userArgs;    if (!parser->applyInputOutputs(hpi, inputs, outputs,                                   inputStrides, outputStrides,                                   profilingQuery, userArgs)) {        returnfalse;    }    // 把新 BO handle 替换进已构造的命令缓冲，KMD 提交时就会拿到新地址    std::vector<uint32_t> newHandles;    for (constauto &bo : userArgs) newHandles.emplace_back(bo->getHandle());    commandBuffer->replaceBufferHandles(userArgHandles, newHandles);    userArgHandles = std::move(newHandles);    eraseAssociatedBufferObjects(userArgIndex);    appendAssociateBufferObject(std::move(userArgs));    returntrue;}
```

这意味着：你可以录一次命令、close 一次、然后反复换输入张量提交——而不必每次都重建 command list。对于推理服务这种高 QPS 场景，这一手能省下大量"录命令"的 CPU 开销。

## 六、磁盘缓存：让二次编译只剩 hash 命中

> NPU Compiler 把 OpenVINO IR 编成 ELF 是个昂贵操作。`linux-npu-driver` 的解决办法朴素却扎实——按内容 hash 写到 `~/.cache/ze_intel_npu_cache/`，下次直接 mmap 出来用。

### 6.1 Hash Key：保证语义一致

```
// 来源：umd/level_zero_driver/source/ext/disk_cache.cpp#L95-L132DiskCache::Key DiskCache::computeKey(const ze_graph_desc_2_t &desc,                                     const vcl_compiler_properties_t *vclProp) {    if (cachePath.empty()) return {};    HashCity hash;                              // 用 Google CityHash    constexpruint32_t driverVersion = DRIVER_VERSION;    hash.updateConfigurationHash(/*driver 版本*/);    if (vclProp) {        hash.updateConfigurationHash(/*compiler id*/);        hash.updateConfigurationHash(/*compiler version*/);        hash.updateConfigurationHash(/*supportedOpsets*/);    }    hash.updateConfigurationHash(/*desc.format*/);    if (desc.pBuildFlags)        hash.updateConfigurationHash(/*编译 flags 字符串*/);    hash.updateConfigurationHash(/*desc.flags*/);    // 模型本体也参与 hash；若调用方给了显式的 input hash 则优先用之    constze_structure_type_graph_ext_t *type =        static_cast<constze_structure_type_graph_ext_t *>(desc.pNext);    if (type != nullptr && *type == ZE_STRUCTURE_TYPE_GRAPH_INPUT_HASH) {        constze_graph_input_hash_t *inputHash = ...;        return hash.final(/*input hash 指针*/, sizeof(inputHash->hash));    }    return hash.final(desc.pInput, desc.inputSize);}
```

注意这里的工程考量：**驱动版本、compiler 版本、build flags、模型字节** 全部进 hash。任何一个变化都会让 key 变，避免老缓存被新驱动错读。`desc.pNext` 这条分支允许调用方（比如 OpenVINO Plugin）传入预先算好的 model hash，省去对 IR 文件再 hash 一次的开销。

### 6.2 LRU 淘汰：用 `st_atim` 而不是 `mtime`

```
// 来源：umd/level_zero_driver/source/ext/disk_cache.cpp#L181-L211static size_t removeLeastUsedFiles(VPU::OsInterface &osInfc,                                   std::filesystem::path &cachePath, size_t expSize) {    std::map<time_t, std::string> sortedFiles;    osInfc.osiScanDir(cachePath, [&sortedFiles](constchar *name, struct stat &stat "&sortedFiles") {        // 用 access time（最后访问时间）排序，而非 mtime        sortedFiles.emplace(stat.st_atim.tv_sec, name);    });    size_t removedSize = 0;    for (auto &[timestamp, filename] : sortedFiles) {        auto filePath = cachePath / filename;        // 删除前必须先拿到独占锁，避免并发场景下吃别人的缓存        auto file = osInfc.osiOpenWithExclusiveLock(filePath, false);        if (!file) continue;        auto fileSize = file->size();        if (!osInfc.osiFileRemove(filePath)) continue;        removedSize += fileSize;        if (removedSize >= expSize) break;    }    return removedSize;}
```

两个细节藏着真功夫：

- **用 `st_atim` 做 LRU**。不像 mtime 反映写入，atime 反映"最后被读"的时间，这才是 LRU 真正想要的。当然这要求挂载点没禁掉 atime 更新。
- **删除时拿独占锁**。`osiOpenWithExclusiveLock` 是 flock 风格的文件锁，确保不会把其它进程正在读的缓存文件删掉。

整个写入路径也很对称：`setBlob`[13] 会先做容量检查，必要时调用 `removeLeastUsedFiles` 回收空间，然后写入 blob 加 CityHash 校验和；下次 `getBlob` 读出来时会校验末尾哈希，一旦不一致则立即删除文件——损坏的缓存比没缓存还危险。

## 七、命令队列与 fence：最终把 Job 投到内核

> 绕回到 Level Zero 入口侧，`CommandQueue::executeCommandLists` 是用户调用 `zeCommandQueueExecuteCommandLists` 的最终落点：

```
zeCommandQueueExecuteCommandLists(...)              │              ▼  CommandQueue::executeCommandLists()   ├─ 检查 cmdList 已 close   └─ preemptionCacheAcquire (按需)              │              ▼  for each cmdList:   ┌──────────────────────────────────┐   │ job->updateOnSubmit()  (Mutable) │   │ job->addPreemptionBuffer(...)    │   │ vpuQueue->submit(job)            │   │  └─ ioctl DRM_IOCTL_IVPU_SUBMIT  │   │      └─► intel_vpu.ko → 固件执行  │   └──────────────────────────────────┘              │       ┌──────┴──────┐       ▼             ▼   hFence ≠ NULL   hFence == NULL   fence 跟踪 jobs queue 跟踪 jobs              │              ▼   SYNC 模式? → 立即 synchronize(MAX)   否则       → 返回，留待后续 sync
```

> 该过程核心代码如下：

```
// 来源：umd/level_zero_driver/source/cmdqueue.cpp#L228-L286std::vector<std::shared_ptr<VPU::VPUJob>> jobs;jobs.reserve(nCommandLists);for (auto i = 0u; i < nCommandLists; i++) {    auto cmdList = CommandList::fromHandle(phCommandLists[i]);    if (!cmdList->getNumCommands()) continue;        // 跳过空 list    std::shared_ptr<VPU::VPUJob> job = cmdList->getJob();    if (!job->updateOnSubmit()) {                    // 触发所有命令的 update 钩子        return ZE_RESULT_ERROR_UNKNOWN;    }    if (preemptionBuffer) {        job->addPreemptionBuffer(preemptionBuffer);  // 抢占上下文    }    if (!vpuQueue->submit(job.get())) {              // 真·下发到 KMD        if (errno == -EBADFD) return ZE_RESULT_ERROR_DEVICE_LOST;        return ZE_RESULT_ERROR_UNKNOWN;    }    jobs.emplace_back(std::move(job));}if (hFence != nullptr) {    auto fence = Fence::fromHandle(hFence);    fence->setTrackedJobs(std::move(jobs));          // 关联到用户 fence} else {    std::copy(jobs.begin(), jobs.end(), std::back_inserter(trackedJobs));}if (queueMode == CommandQueueMode::SYNCHRONOUS)    return synchronize(std::numeric_limits<uint64_t>::max());
```

这段代码体现了若干工业化的精细：

- **抢占 buffer 按需挂载**：只有 NPU 支持上下文抢占（preemption）时，每个 job 才会附带一段 context save area，让多 queue 互相打断时能保存现场。
- **fence 双轨**：用户给了 `hFence` 就把 job 挂到 fence 上；没给就挂到 queue 自己的 `trackedJobs`，等下一次 `synchronize()` 统一回收。
- **同步队列即时阻塞**：`SYNCHRONOUS` 模式下提交完直接 `synchronize`，阻塞等所有 job 完成，方便单线程脚本式场景。

`vpuQueue->submit()` 内部会构造 `drm_ivpu_submit` 结构并走 `DRM_IOCTL_IVPU_SUBMIT` 进内核——这是用户态与 `intel_vpu.ko` 唯一的 fast-path 接口。其余 `mmap`/`bo_create`/`bo_wait` 等也都通过同样的 DRM ioctl 通道。

## 八、跨代硬件的优雅抽象

> 最后值得一提的是 `umd/vpu_driver/source/device/` 下的多代硬件实现。

```
┌───────────────────────────────┐│  VPUDevice (基类)              ││  + init / getHwInfo / getBDF  ││  + virtual createDeviceCtx()  ││  # jsmApiVersion              ││  # mappedInferenceVersion ◄── 与固件协商└───────────────┬───────────────┘                │ 继承   ┌────────────┼────────────┐   ▼            ▼            ▼┌─────────┐ ┌─────────┐ ┌─────────┐│vpu_37xx │ │vpu_40xx │ │vpu_50xx ││ MTL     │ │ LNL/ARL │ │ PTL/WCL ││ 第1代   │ │ 第2代   │ │ 第3代   │└─────────┘ └─────────┘ └─────────┘          ▲          │ DeviceFactory 按 PCI ID          │ 探测并实例化对应 family
```

仓库里同时维护着 **vpu_37xx** （Meteor Lake）、**vpu_40xx** （Lunar Lake / Arrow Lake）、**vpu_50xx** （Panther Lake / Wildcat Lake）三套 family 子目录。共用部分被抽到 `vpu_device.hpp`：

```
// 来源：umd/vpu_driver/source/device/vpu_device.hpp#L24-L58class VPUDevice {public:    bool init();    VPUDevice(std::string devPath, OsInterface &osInfc);    virtual ~VPUDevice() = default;    const VPUHwInfo &getHwInfo() const;    const std::vector<GroupInfo> &getMetricGroupsInfo();    bool getCapMetricStreamer() const;    bool getCapCmdQueueCreation() const;    virtual std::unique_ptr<VPUDeviceContext> createDeviceContext();    int getBDF(uint32_t *domain, uint32_t *bus, uint32_t *dev, uint32_t *func);    bool getActiveTime(uint64_t &activeTimeUs);    bool isConnected();    std::string jsmApiVersion;            // JSM = Job Submission Manager    std::string jsmCmdApiVersion;    std::string mappedInferenceVersion;   // NN runtime 与固件协议的版本// ...};
```

`createDeviceContext()` 是 virtual，由各代 family 重载，决定该代设备用哪种命令上下文实现；`jsmApiVersion`/`mappedInferenceVersion` 这些字符串则用于运行时检查："compiler 编出来的 ELF 期望 `MAPPED_INFERENCE_VERSION = 6.1.0`，但当前固件是 `7.0.0`，所以请你升级固件"——这就是 docs/overview.md 中那条著名的"Firmware version mismatch"提示[14] 背后的实际判断逻辑。

## 九、结语：当一颗 NPU 真正"被 Linux 看见"

> 回到文章开头那个问题——从 Python 一行 `infer()` 到 NPU 上电运算，中间到底隔着多少层代码？

读完之后我想，答案其实不是一个数字，而是一种姿态。

我们一路看下来：

- `Driver::driverInit` 用 `std::call_once` 把全局初始化锁死
- `VPUCommand` 用 `std::any` 抹掉命令类型差异
- `VPUJob::closeCommands` 按同步语义把命令流切成多个 buffer
- `CommandHeader` 用 `static_assert` 把 64 字节对齐写进类型系统
- `DiskCache` 用 `st_atim` 而非 mtime 做 LRU
- `createCommandBuffer` 用一前一后的 fence 把多 buffer 串成一条隐式 DAG
- ……

> 每一处单看都不算炫技，但叠在一起，组成的是一种"我必须把这件事做对"的工程姿态。

而这种姿态，恰恰是过去二十年 Linux GPU 驱动（i915、amdgpu、nouveau）一路打磨出来的范式，被 Intel 几乎"等比例放大"地搬到了 NPU 这块新硬件上——**用户态/内核态分层、DRM ioctl 通道、命令缓冲 + fence 的异步模型、host/shared/device 三类 BO、可缓存的离线编译产物**。:mdnice-bold-italic[NPU 在硅片层面是新东西，但在软件抽象层面，它被以"老熟人"的姿势接入了 Linux 生态。] 这件事，是 `intel/linux-npu-driver` 这份代码最容易被低估、却也最重要的价值。

::: block-1
它意味着什么呢？意味着当你在一台 Core Ultra 笔记本上 `pip install openvino` 然后跑一行推理，你调用的并不是某个 2024 年才横空出世的新栈，而是**一条已经在 Linux 内核与图形子系统里磨了二十年的加速器软件范式**。

:mdnice-bold-italic[AI PC 听起来是发布会上的新词，但真正把它撑起来的，是 DRM 子系统、是 Mesa 时代留下的 BO 模型、是 oneAPI Level Zero 的统一接口、是 Google 在 2011 年随手开源的 CityHash……] 这些零件各自演化、各自成熟，最后被 `linux-npu-driver` 用一万多行 C++ 像榫卯一样咬合起来，让 NPU "悄无声息"地变成了 `/dev/accel/accel0` —— 一个普通到不能再普通的 Linux 设备节点。
:::

所以这趟源码之旅最大的收获，也许并不是哪几个 API 的具体用法，而是看到了一件更朴素的事：

> 异构硬件的"平民化"，从来不靠某一次惊天动地的发明，而靠一份份这样"愿意把对齐写进 static_assert、愿意用 atime 而不是 mtime"的驱动代码，一寸一寸地把硬件从 datasheet 里拉到 `/dev` 之下、拉到 Python `import` 之下、最终拉到普通开发者的指尖。

**这条路 GPU 走了二十年。NPU 才刚刚出发**。但当你看到 `umd/vpu_driver/source/device/` 下 `vpu_37xx`、`vpu_40xx`、`vpu_50xx` 三代实现已经并排站好，看到 `docs/overview.md` 的 changelog 里 Wildcat Lake 已经悄悄到位、Panther Lake 已经稳定支持——你会意识到，这次的速度，比当年 GPU 那段路要快得多。

如果你的 CPU 恰好是 Core Ultra 系列，那台笔记本的硅片里，此刻就有一颗正在等待被你调度的 NPU。不妨打开 `intel/linux-npu-driver`，从 `umd/level_zero_driver/source/driver.cpp` 那个 `std::call_once` 开始读起——你会发现，所谓 AI PC，并不在发布会的 PPT 里，而在这些静静运行在 `/dev/accel/accel0` 背后、一行一行被认真写下来的 C++ 代码里。

> **硬件造出来，只是故事的开始；让它"被操作系统看见、被框架调用、被开发者随手用"，才是故事真正动人的部分。** `linux-npu-driver` 正在替这个时代，把这后半段故事写下去。

参考资料

[1] `intel/linux-npu-driver`: *https://github.com/intel/linux-npu-driver*

[2] `docs/overview.md`: *https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md*

[3] docs/overview.md: *https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md*

[4] release 页面: *https://github.com/intel/linux-npu-driver/releases*

[5] `compiler/npu_compiler.cmake`: *https://github.com/intel/linux-npu-driver/blob/main/compiler/npu_compiler.cmake*

[6] docs/overview.md: *https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md*

[7] `docs/UMD_overview.md`: *https://github.com/intel/linux-npu-driver/blob/main/docs/UMD_overview.md*

[8] `validation/umd-test/configs/README.md`: *https://github.com/intel/linux-npu-driver/blob/main/validation/umd-test/configs/README.md*

[9] `docs/overview.md`: *https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md*

[10] `umd/level_zero_driver/source/driver.cpp`: *https://github.com/intel/linux-npu-driver/blob/main/umd/level_zero_driver/source/driver.cpp*

[11] `umd/vpu_driver/source/command/command_buffer.hpp`: *https://github.com/intel/linux-npu-driver/blob/main/umd/vpu_driver/source/command/command_buffer.hpp*

[12] zeMutableCommandList: *https://spec.oneapi.io/level-zero/latest/core/EXT_Exp_MutableCommandList.html*

[13] `setBlob`: *https://github.com/intel/linux-npu-driver/blob/main/umd/level_zero_driver/source/ext/disk_cache.cpp#L213-L258*

[14] Firmware version mismatch提示: *https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 A100：拆解摩尔线程开源的 GPU 驱动级基准测试 gpu-compute-driver-bench](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901875&idx=1&sn=3db0fb835b96a96286fc1e65ef6492e2&scene=21#wechat_redirect)**
- **[赋能 Linux GPU 生态：NVIDIA 开源 GPU 内核模块的架构、功能与技术实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893566&idx=1&sn=1cadc8c260f80856125b710ce6bc4249&scene=21#wechat_redirect)**
- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**

:::
