关键词：gpu_ext、**eBPF**、:mdnice-bold-italic[GPU资源管理]、**跨层策略运行时**、SIMT感知优化、:mdnice-bold-italic[多租户GPU调度]

> 如今，GPU 早已不是“游戏专属”，而是 AI 时代的“算力心脏”。从 LLM 推理（如 ChatGPT 背后的模型）、DNN 训练（如 ResNet 训练），到向量搜索（如 Faiss）、GNN 分析，**不同 workload 对 GPU 资源的需求天差地别**：有的怕延迟（推理），有的要算力（训练），有的爱“随机访问”（向量搜索）。

但传统 GPU 管理却陷入了“一刀切”的僵局——:mdnice-bold-italic[默认的 UVM（统一虚拟内存）放置策略、FIFO 调度策略，在多样 workload 面前频频“掉链子”]。比如：

- LLM 推理的 KV-cache 频繁换入换出时，UVM 的 LRU 算法会导致严重的内存颠簸；
- 多租户共享 GPU 时，推理任务的低延迟需求和训练任务的高吞吐量需求更是“水火不容”。

::: block-1
**难道就没有一种方式，既能灵活适配不同 workload，又不用修改应用、不用冒险改内核？**
:::

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/01.webp)

- :mdnice-bold-italic[gpu_ext: Extensible OS Policies for GPUs via eBPF]
- https://arxiv.org/pdf/2512.12615
- 5000 字，阅读 16 分钟，**播客 47 分钟**

相关推荐

- **[代码开源！OSDI'25 通过类 eBPF 探测实现可编程细粒度 GPU 内核分析工具 Neutrino：跨厂商GPU剖析工具！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892532&idx=1&sn=446017b09eaec1e92fb6d77ed8301908&scene=21#wechat_redirect)**
- **[eGPU：让eBPF为GPU可观测性与可编程性赋能！首个动态PTX注入式eBPF-GPU框架，较NVBit显著降低内存观测开销](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892100&idx=1&sn=e1995e2908172d914fb38f4d7577268f&scene=21#wechat_redirect)**
- **[节省 GPU 51% Capacity 和 26% 能耗，CMU, Meta提出LithOS：迈向高效 GPU 机器学习操作系统](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888297&idx=1&sn=c87e81667f2e91731e2628a5abb53923&scene=21#wechat_redirect)**

> 来自 eunomia-bpf、加州大学圣克鲁兹分校等团队的研究人员，在论文《gpu_ext: Extensible OS Policies for GPUs via eBPF》中给出了答案——**基于 eBPF 的跨层 GPU 策略运行时**，:mdnice-bold-italic[让 GPU 驱动和设备层变成“可编程的 OS 子系统”，吞吐量最高提升 4.8 倍，尾延迟降低 2 倍。]

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/02.webp)

::: image-caption
图3：gpu_ext的高层架构。gpu_ext 通过三个核心组件实现跨层管理：① 用户空间控制平面（编译 eBPF 策略、导出指标）；② 主机驱动层（提供安全钩子、验证 eBPF 代码）；③GPU 设备层（JIT 编译 eBPF 到 PTX、注入 trampoline 触发策略）。共享的 eBPF 映射（Maps）则实现主机-设备的状态同步
:::

## 本文目录

- 一、为什么需要 gpu_ext？——现有方案的四大“坑”
  - 1. 用户空间 Runtime：灵活但“短视”
  - 2. 内核级修改：强大但“笨重”
  - 3. 二进制工具：精细但“危险”
  - 4. CPU eBPF：通用但“管不到”
  - 本质原因
- 二、gpu_ext 如何破局？——三大设计原则+核心架构
  - 1. 破解三大技术难题
  - 2. 高层架构：跨层 eBPF Runtime 全景
  - 3. 三大核心接口：让 GPU“可编程”
- 三、gpu_ext 的创新点：这三个设计绝了！
  - 1. 跨主机-设备的统一 eBPF 运行时
  - 2. SIMT 感知的验证与优化
  - 3. 低开销+高可编程性的平衡
- 四、相关工作对比：gpu_ext 到底强在哪？
- 五、实验说话：性能提升有多猛？
  - 1. 单租户优化：吞吐量飙升 4.8 倍
  - 2. 多租户协调：双向收益，不偏科
  - 3. 低开销：观测工具比 NVBit 快 10 倍
- 六、未来展望：从 NVIDIA 到多厂商，从单机到集群
- 总结：GPU 管理进入“可编程时代”

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/03.webp)

## 一、为什么需要 gpu_ext？——现有方案的四大“坑”

> 在 gpu_ext 出现之前，工程师们为了优化 GPU 管理，尝试了多种方案，但都逃不开“trade-off”的困境：

### 1. 用户空间 Runtime：灵活但“短视”

像 vLLM、PyTorch 这样的用户空间框架，虽然能自定义内存或调度策略（如 vLLM 的 PagedAttention），但有三个致命问题：

::: block-1
- **缺全局视野**：:mdnice-bold-italic[只能管自己进程内的资源，多租户共享时无法协调] （比如 A 进程的缓存把 B 进程的热数据挤掉）；
- **无硬件控制权**：:mdnice-bold-italic[没法精细控制] GPU 的时间片、页表等底层机制；
- **需改应用代码**：不同框架的策略不通用，换个框架就得重写。
:::

### 2. 内核级修改：强大但“笨重”

有些方案（如 TimeGraph、GPREEMPT）直接修改 GPU 驱动内核，虽然能控制硬件状态，但：

::: block-1
- **复杂度爆炸**：GPU 驱动代码庞大，修改后难维护、易引入 BUG；
- **不安全**：一旦改崩，整个系统可能宕机；
- **无法动态更新**：改完策略要重启驱动，线上服务没法用。
:::

### 3. 二进制工具：精细但“危险”

NVBit 这类工具能注入代码到 GPU 二进制中，但：

::: block-1
- **无安全保障**：注入的代码可能导致死锁；
- **只观测不控制**：主要用于离线 profiling，没法实时调整内存或调度；
- **开销高**：比如 NVBit 的观测开销高达 85%，线上用不起。
:::

### 4. CPU eBPF：通用但“管不到”

eBPF 在 CPU 侧早已是“瑞士军刀”（如调度优化、网络过滤），:mdnice-bold-italic[但直接用到 GPU 上就行不通]：

::: block-1
- **看不见设备端**：CPU eBPF 只能跟踪驱动命令，没法知道 GPU 内部的 warp 发散、缓存命中情况；
- **语义不匹配**：CPU eBPF 是单线程逻辑，GPU 是 SIMT（单指令多线程）架构，直接用会导致效率暴跌。
:::

### 本质原因

上面这四大方案的困境，本质是“**策略与机制没有彻底分离** ”——:mdnice-bold-italic[GPU 的底层管理机制（如页表、调度队列）和上层策略（如何时预取、如何分配时间片）绑死在了一起]。

::: block-1
而 **gpu_ext 的核心思路，就是用 eBPF 把“策略”抽离出来**，让底层机制保持稳定，上层策略灵活可编程。
:::

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/04.webp)

::: image-caption
图1 展示了不同GPU工作负载的内存访问（页面错误）模式差异，是作者论证“一刀切”资源管理策略存在缺陷的关键依据。其中，faiss Build呈现顺序扫描模式，faiss Query为随机访问；llama.cpp MoE的预填充阶段有周期性顺序特征，解码阶段则转为稀疏随机访问；PyTorch DNN表现为周期性块访问。这些迥异的访问模式表明，单一静态策略无法适配所有场景，也凸显了gpu_ext打造可编程跨层策略的必要性
:::

## 二、gpu_ext 如何破局？——三大设计原则+核心架构

gpu_ext 的目标很明确：**让 GPU 管理像 CPU eBPF 一样，安全、动态、可编程，同时适配 GPU 的 SIMT 特性**。为了实现这个目标，它遵循三大设计原则：

| 设计原则 | 核心含义 |
| --- | --- |
| OS 级透明动态可编程 | 策略在 OS 层生效，不用改应用/驱动，运行时能动态更新（比如白天用低延迟策略，晚上用高吞吐量策略） |
| 跨层主机-设备抽象 | 主机（CPU 侧）和设备（GPU 侧）用统一的 eBPF 接口，避免“主机调主机的，GPU 调 GPU 的”脱节 |
| SIMT 感知安全高效 | 策略要适配 GPU 的 SIMT 架构，确保 warp 执行一致（不发散），同时控制资源开销 |

### 1. 破解三大技术难题

> 要实现这三大原则，gpu_ext 需要解决三个关键技术挑战：

#### 技术难题1：如何平衡“表达力”和“安全性”？

既要让 eBPF 策略能控制 GPU 底层机制（如预取、时间片），又要防止策略搞崩系统。

gpu_ext 的解法是：**暴露“窄而稳定”的接口** ——:mdnice-bold-italic[不直接给策略访问页表、中断的权限，而是封装成标准化的钩子] （如内存的 activate/evict、调度的 task_init），:mdnice-bold-italic[策略只能通过这些钩子做决策，底层安全由驱动保证。]

#### 技术难题2：如何解决 CPU eBPF 与 GPU SIMT 的语义冲突？

CPU eBPF 是单线程逻辑，GPU 是 32 个线程组成 warp 并行执行。如果每个线程都跑 eBPF 策略，会导致 warp 发散（部分线程走分支 A，部分走分支 B），效率暴跌。gpu_ext 的解法是：

::: block-1
- **SIMT 感知验证**：静态检查 eBPF 代码，确保分支条件、循环边界等是“warp 统一”的（比如所有线程用同一个内存地址做判断）；
- **warp leader 执行**：每个 warp 只让一个“leader”线程跑 eBPF 策略，结果广播给其他线程，避免重复计算。
:::

#### 技术难题3：如何高效同步主机-设备的共享状态？

主机（DRAM）和 GPU（显存）的延迟差了两个数量级，直接同步会卡死。gpu_ext 的解法是**分层跨层映射**：

::: block-1
- 逻辑上是一个统一的 KV 存储（比如记录每个内存块的访问频率）；
- 物理上分片存储：热数据放 GPU 共享内存（低延迟），冷数据放主机 DRAM（大容量）；
- 异步快照同步：不用实时同步，在 GPU 内核执行完后批量合并状态，兼顾一致性和效率。
:::

### 2. 高层架构：跨层 eBPF Runtime 全景

> gpu_ext 的架构贯穿主机和设备，主要分三部分：

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/02.webp)

::: image-caption
图3：gpu_ext的高层架构。gpu_ext 通过三个核心组件实现跨层管理：① 用户空间控制平面（编译 eBPF 策略、导出指标）；② 主机驱动层（提供安全钩子、验证 eBPF 代码）；③GPU 设备层（JIT 编译 eBPF 到 PTX、注入 trampoline 触发策略）。共享的 eBPF 映射（Maps）则实现主机-设备的状态同步
:::

#### 架构组件1：用户空间控制平面

用 clang/libbpf 编译 eBPF 策略代码，支持：

- 动态部署策略（不用重启应用）；
- 调整策略参数（如预取阈值、时间片长度）；
- 导出监控指标（如每个内存块的命中率、租户延迟）。

#### 架构组件2：主机驱动层

在 NVIDIA 开源 GPU 驱动中嵌入轻量级 instrumentation（约 100 行代码），暴露两类钩子：

- 内存钩子：activate（内存块激活）、evict_prepare（准备驱逐）等；
- 调度钩子：task_init（队列创建）、task_destroy（队列销毁）等。 同时，扩展 Linux eBPF 验证器，确保 eBPF 代码安全。

#### 架构组件3：GPU 设备层

- **动态二进制重写**：拦截 CUDA API，在 GPU 内核中注入“trampoline”（跳板），触发设备端 eBPF 策略（不用重编译应用）；
- **JIT 编译**：把 eBPF 字节码翻译成 GPU 能执行的 PTX 指令；
- **warp 级优化**：每个 warp 只跑一次 eBPF 策略，结果广播，降低开销。

### 3. 三大核心接口：让 GPU“可编程”

gpu_ext 定义了三类 eBPF 程序类型，覆盖 GPU 管理的核心场景：

#### 接口1：内存接口（BPF_PROG_TYPE_GPU_MEM）

把 GPU 内存管理变成“可编程缓存”，主机和设备端协同工作：

- 主机侧钩子：比如`gpu_prefetch`决定何时预取数据，`gpu_evict_prepare`决定哪些数据优先被驱逐；
- 设备侧钩子：比如`access`监测 warp 的内存访问，触发主机预取；
- 抽象粒度：用 2MB UVM 块或 4KB 页，隐藏不同 GPU 的硬件差异。

举个例子：LLM 推理时，设备侧发现 warp 在访问 KV-cache 的第 10 块，就通过`gdev_mem_prefetch`告诉主机“赶紧把第 11 块预取过来”，避免缺页等待。

#### 接口2：调度接口（BPF_PROG_TYPE_GPU_SCHED）

分主机和设备两层控制：

::: block-1
- 主机侧：控制 GPU 队列的优先级、时间片，比如给推理任务分配更长时间片；支持主动抢占（`bpf_gpu_sched_preempt`）；
- 设备侧：实现工作窃取调度器，比如`should_try_steal`判断是否从负载高的 SM“偷”任务，解决图 2 中的 SM 负载不平衡问题。
:::

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/05.webp)

::: image-caption
图2 通过eBPF追踪直观呈现GPU线程调度不平衡，是论证传统调度机制缺陷的关键依据。（a）SM负载分布存在极端差异，SM 15承载382个线程，而SM 6仅3个，失衡倍数达127倍；（b）warp活动热图进一步显示，SM 0将工作集中在高编号warp，其他SM的warp槽位却处于未充分利用状态。这种调度失衡会严重浪费GPU算力，也凸显了gpu_ext设计灵活调度策略的必要性。
:::

#### 接口3：设备端 Instrumentation（BPF_PROG_TYPE_GPU_DEV）

在 GPU 内核中注入 eBPF 代码，实现细粒度观测或控制。比如：

::: block-1
- 观测 warp 的执行时间，定位性能瓶颈；
- 动态调整计算逻辑（如根据缓存命中率调整数据分片）。
:::

## 三、gpu_ext 的创新点：这三个设计绝了！

> gpu_ext 能在众多方案中脱颖而出，关键在于**三个“突破性”设计**：

### 1. 跨主机-设备的统一 eBPF 运行时

这是 gpu_ext 最核心的创新——**第一次把 eBPF 从 CPU 侧延伸到 GPU 侧，并且用统一的接口：**

- 策略开发者不用学新语言，用熟悉的 eBPF 就能写 GPU 策略；
- 主机和设备共享状态（通过跨层映射），避免“策略割裂”（比如主机预取的 data，正好是设备需要的）。

### 2. SIMT 感知的验证与优化

针对 GPU 的 SIMT 架构做了深度适配，解决了 CPU eBPF 的“水土不服”：

- **静态验证**：确保 eBPF 代码不会导致 warp 发散（比如禁止非统一的原子操作）；
- **warp 级执行**：每个 warp 只跑一次策略，把 eBPF 的 scalar 逻辑高效适配到 SIMT 架构，设备侧策略开销降低 60%-80%（对比传统注入方案）。

### 3. 低开销+高可编程性的平衡

gpu_ext 做到了“鱼和熊掌兼得”：

- **低开销**：主机侧钩子开销<0.2%，设备侧观测开销仅 3%-14%（对比 NVBit 的 85%-93%）；
- **高可编程**：策略代码量极少（16-925 行），还能组合复用（比如把“LRU eviction”和“stride prefetch”组合成新策略）。

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/06.webp)

::: image-caption
表1：gpu_ext策略代码量。gpu_ext 的核心策略仅需几十到几百行代码，且能跨框架复用（比如“Stride Prefetch”策略，既能用于 LLM 推理，也能用于 GNN 训练）
:::

## 四、相关工作对比：gpu_ext 到底强在哪？

为了更直观地看出 gpu_ext 的优势，我们把它和主流方案做个对比：

| 方案类型 | 代表工作 | 可编程性 | 跨租户协调 | 硬件控制 | 动态更新 | 开销 | 需改应用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 用户空间 Runtime | vLLM、PyTorch | 中 | 无 | 弱 | 是 | 低 | 是 |
| 内核级修改 | TimeGraph、GPREEMPT | 低 | 有 | 强 | 否 | 中 | 否 |
| 二进制工具 | NVBit | 低 | 无 | 弱 | 是 | 高 | 否 |
| CPU eBPF | sched_ext | 高 | 有 | 无（GPU） | 是 | 低 | 否 |
| **gpu_ext** | - | 高 | 有 | 强 | 是 | 低 | 否 |

gpu_ext 在“可编程性”“跨租户协调”“硬件控制”“动态更新”“低开销”“无需改应用”六个维度全部拉满，解决了现有方案的核心痛点。

## 五、实验说话：性能提升有多猛？

> 研究人员在 Linux 6.15 + NVIDIA RTX 5090/P40 GPU 上做了大量实验，覆盖 LLM 推理、GNN 训练、向量搜索等真实 workload，结果非常亮眼：

### 1. 单租户优化：吞吐量飙升 4.8 倍

#### （1）LLM 推理：GPT-OSS-120B MoE

这个模型有 59GB 参数，远超 RTX 5090 的 32GB 显存，需要 CPU-GPU 内存分层。对比“框架级 CPU offloading”，gpu_ext 的 eBPF 预取策略：

- 解码阶段（内存敏感）吞吐量提升 4.8 倍；
- 预填充阶段（计算敏感）性能接近框架方案，整体端到端延迟大幅降低。

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/07.webp)

::: image-caption
图5：GPT-OSS-120B的吞吐量。横轴是不同配置（框架 offloading、默认 UVM、gpu_ext 等），纵轴是吞吐量（tokens/s）。gpu_ext 在解码阶段的吞吐量是框架 offloading 的 4.8 倍，核心原因是 eBPF 策略能精准预取专家权重，避免 PCIe 传输瓶颈
:::

#### （2）Qwen-30B MoE：KV-cache 优化

100 个并发请求下，KV-cache 总大小达 60K tokens，默认 UVM 会频繁颠簸。gpu_ext 的“KV-aware 预取”策略：

- 首 token 时间（TTFT）降低 2 倍；
- 解码吞吐量提升 1.3 倍，接近专门的 KV-cache 优化框架 LMCache，且不用改应用。

#### （3）GNN 训练：15M 节点无 OOM

在 15M 节点的 GCN 训练中（2.17 倍内存超配）：

- 原生 GPU 分配会 OOM（内存不够）；
- 用户空间预取需要改代码，能提速 5.5 倍；
- gpu_ext 不用改代码，提速 2.65 倍，还能和用户空间策略叠加，再提 1.44 倍。

### 2. 多租户协调：双向收益，不偏科

实验中，让“高优先级 LLM 推理”和“低优先级 GNN 训练”共享 GPU：

- 推理任务的尾延迟（TPOT）降低 45%；
- 训练任务的速度提升 28%；
- 对比默认 UVM 的“互相抢资源”，gpu_ext 通过租户感知的内存/调度策略，实现了“双赢”。

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/08.webp)

::: image-caption
图11：双租户共置：llama.cpp推理（LC）+图神经网络（GNN）训练（BE）。gpu_ext实现了相互改进：LC延迟优于单租户性能，而BE训练提升了28%。租户感知策略减少了竞争，而非以牺牲一个工作负载为代价来优先处理另一个工作负载
:::

### 3. 低开销：观测工具比 NVBit 快 10 倍

gpu_ext 的设备端观测工具（如记录线程执行时间的 threadhist）：

- 开销仅 3%-14%；
- 对比 NVBit 的 85%-93%，性能提升 10 倍以上，线上场景也能用。

![文章插图](/articles/2025-12-25-gpu-ext-ebpf-gpu-os-6096ca28/images/09.webp)

::: image-caption
表2：观测工具开销对比。pu_ext 的观测工具（如 kernelretsnoop、threadhist）开销远低于 NVBit，其中 threadhist 仅 3%开销，而 NVBit 需要 87%
:::

## 六、未来展望：从 NVIDIA 到多厂商，从单机到集群

目前 gpu_ext 主要支持 NVIDIA GPU（基于开源驱动），但研究团队已经规划了未来方向：

::: block-1
- **多厂商适配**：基于 Linux 的 HMM（异构内存管理）和 DRM 调度器，适配 AMD ROCm、Intel Level Zero；
- **集群级扩展**：把单机的策略扩展到 GPU 集群，解决跨节点的资源协调问题；
- **更细粒度优化**：结合 AI workload 的语义（如 LLM 的 layer 结构），做更智能的预取或调度。
:::

## 总结：GPU 管理进入“可编程时代”

> gpu_ext 的价值，不仅在于“吞吐量提升 4.8 倍”这样的性能指标，更在于它为 GPU 管理提供了一种“新范式”——**用 eBPF 打通主机-设备层，让策略与机制彻底分离**。

从此，:mdnice-bold-italic[优化 GPU 管理不用再“二选一”：不用为了灵活改应用，不用为了安全牺牲性能]。对于 AI 基础设施团队来说，这意味着可以根据业务需求快速迭代策略（:mdnice-bold-italic[比如白天跑推理用低延迟策略，晚上跑训练用高吞吐量策略]），而不用动底层驱动或应用代码。

随着 AI workload 越来越多样，gpu_ext 这样的“可编程管理”方案，或许会成为未来 GPU 集群的“标配”。

更多推荐

- [30.9% 能耗节省！亚秒级 GPU 频率实时决策框架 AGFT：实现LLM推理实时功耗与性能的平衡术！低侵入且不降服务质量！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891636&idx=1&sn=9d16c36f68880175ee491d5b411434e9&scene=21#wechat_redirect)
- [赋能 Linux GPU 生态：NVIDIA 开源 GPU 内核模块的架构、功能与技术实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893566&idx=1&sn=1cadc8c260f80856125b710ce6bc4249&scene=21#wechat_redirect)
- [eGPU：让eBPF为GPU可观测性与可编程性赋能！首个动态PTX注入式eBPF-GPU框架，较NVBit显著降低内存观测开销](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892100&idx=1&sn=e1995e2908172d914fb38f4d7577268f&scene=21#wechat_redirect)

加群
