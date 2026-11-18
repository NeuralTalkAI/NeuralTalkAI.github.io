关键词：**NVLink**、MNNVL、**基准测试**、:mdnice-bold-italic[GB200]、CUDA、**集群验收**

> nvloom 是 NVIDIA 开源的 MNNVL（Multi-Node NVLink）fabric 基准测试工具，由一个 C++ 库和一个命令行工具 nvloom_cli 组成，代码量约 5600 行。

它解决的问题比方说吧：当 NVLink 域从单机 8 卡扩展到 72 卡甚至更大规模时，传统的逐对带宽测试以 $O(n^2)$ 时间增长，测一轮 72 卡 pairwise 要执行 5112 次串行测量，工程上无法接受。

![文章插图](/articles/2026-08-05-72-gb200-nvlink-nvidia-mnnvl-6d88b1ea/images/01.webp)

- **nvloom is a set of tools designed to scalably test MNNVL fabrics.**
- https://github.com/NVIDIA/nvloom
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html

::: related-title
**相关推荐**
:::

::: related-links
- **[CudaProfiler：TIRx 轻量化内核级时序剖析工具，精准定位 warp 流水线阻塞与尾延迟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903219&idx=1&sn=f122de6cd22a28e4ede68643b23b5321&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[约束求解赋能 GPU 模糊测试工具 GPU-Fuzz：挖掘深度学习框架 GPU 高危内存错误](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899663&idx=1&sn=41831f6311c438513d5c1732a2e67989&scene=21#wechat_redirect)**

:::

> nvloom 的应对是**把测试组织成九套 test suite：gpu-to-rack 用随机抽样加中位数把 pairwise 压成线性时间，fabric-stress 让所有 GPU 同时收发以打满 fabric**，:mdnice-bold-italic[bisect 模式 3 秒完成一次冒烟测试]。在测量精度上，它用自旋 kernel 让几十张 GPU 在同一时刻开始拷贝，用负载维持算法消除快慢拷贝之间的空窗。

🥱 下面我们先给出工具定位与上手路径，再拆解自旋 kernel 同步与负载维持两个核心机制的源码实现，帮助运维与性能工程师快速把 nvloom 接入自己的集群验收与回归流程。

## 本文目录

- 一、nvloom 是什么
  - 1.1 功能定位
  - 1.2 设计初衷与行业痛点
  - 1.3 核心亮点
  - 1.4 测试维度：拷贝参数与内存类型
- 二、四套高频使用场景
  - 2.1 上线前冒烟测试
  - 2.2 大域全面体检
  - 2.3 复现特定故障流量
  - 2.4 结果可视化对比
- 三、代码剖析
  - 3.1 机制一：自旋 kernel 同步——让 72 张卡在同一拍起跑
  - 3.2 机制二：负载维持——快慢拷贝之间不留空窗

## 一、nvloom 是什么

### 1.1 功能定位

> nvloom 是**面向大规模 MNNVL fabric 的可扩展基准测试工具**，服务于需要:mdnice-bold-italic[验收、排障和回归测试多节点 NVLink 域] 的集群运维与性能工程师。

MNNVL 指跨节点的 NVLink 互联域，典型代表是 GB200 NVL72——18 个节点、72 张 GPU 通过 NVLink Switch 组成一个统一内存域。**nvloom 的任务是对这个域的每一条链路、每一种流量模式做可量化的带宽与延迟测量。**

### 1.2 设计初衷与行业痛点

NVIDIA 此前有 nvbandwidth 做 GPU 间带宽测试[1]，更多参考：performance_and_tuning[2]，但它的 pairwise 模式执行时间为 $O(n^2)$：n 张 GPU 的单向测试需要 $n\times(n-1)$ 次串行测量。72 卡域就是 5112 次，域再扩大就无法在合理时间内完成。

::: block-1
nvloom 保留了 pairwise 套件用于兼容 nvbandwidth 的全部用例，同时提供线性时间的替代方案：gpu-to-rack 对每张 GPU 只在每个机架随机抽 3 个对端测带宽并取中位数。:mdnice-highlight[如果某张 GPU 链路异常，它自己那一行的所有测量都会偏低，而作为别人样本时只影响一次抽样，被中位数过滤。单点故障能被定位，又不污染整体数据。]
:::

### 1.3 核心亮点

| 描述 | 内容 |
| --- | --- |
| 九套预置 suite 覆盖主流流量模式。 | pairwise、fabric-stress、all-to-one、rack-aware-all-to-one、multicast、egm、gpu-to-rack、rack-to-rack、egm-gpu-to-rack，按 -s 套件名或 -t 用例名直接调用。 |
| 一条命令 3 秒完成冒烟。 | bisect 模式把 GPU 分成两组同时互写，72 卡域上含启动时间约 3 秒跑完，适合上线前快速确认 fabric 可用。 |
| CSV 文件定义任意流量。 | 不改代码、不重新编译，用一张 CSV 描述源、目的、内存类型、读写方向、执行引擎，同 measurement_id 的行并发执行。 |
| SM/CE/TMA/CFT 四种拷贝路径。 | 拷贝可由 Copy Engine（cuMemcpyAsync）、SM 自定义 kernel、TMA 或 CUDA 13.3 的 Compute Fabric Transport 执行，分别压测不同硬件通路。 |
| 结果自带正确性校验。 | 每次测量后核对目的缓冲区数据是否与源端写入的伪随机模式一致，带宽数字之外还验证传对了。 |

### 1.4 测试维度：拷贝参数与内存类型

> 每个测试用例由三个维度组合而成。

- 拷贝引擎：SM kernel 压测计算通路，CE（cuMemcpyAsync）压测拷贝引擎通路，二者可在同一 GPU 上并发而互不干扰。
- 拷贝方向：write 由源 GPU 发起写、read 由目的 GPU 发起读，两条方向的 NVLink 表现分别测量；
- 双向用例同时跑两个方向并报告带宽之和。

测试的内存类型覆盖三种，**device** 是 VRAM 里的多节点可寻址内存；**EGM** 是落在 CPU 内存上、GPU 可跨节点访问的扩展显存；**multicast** 是同时驻留在所有 GPU VRAM 中的组播内存，写一次即广播到全域，配套 multimem.st、multimem.red 等 PTX 指令的专用 kernel。三个维度自由组合，构成了几十个用例。

![文章插图](/articles/2026-08-05-72-gb200-nvlink-nvidia-mnnvl-6d88b1ea/images/02.webp)

::: image-caption
multicast_one_to_all 流量模式。源 GPU 向组播内存写一次（实线箭头），数据由硬件广播到全域（虚线）。报告值是实线拷贝的带宽，fabric 上的实际流量是其 $NUM\_GPU-1$ 倍。
:::

## 二、四套高频使用场景

### 2.1 上线前冒烟测试

> 集群扩容或固件升级后，需要在几分钟内确认整个 MNNVL 域连通。这时候可以按照下面操作：

操作：用 MPI 启动 nvloom_cli，每 GPU 一个进程（72 卡域即 18 节点各 4 进程），执行 `nvloom_cli -t bisect_device_to_device_write_sm`。

预期效果：约 3 秒完成。所有 GPU 同时向远端 GPU 写入，任何断链或降速都会立刻暴露在该行的带宽数字上。

![文章插图](/articles/2026-08-05-72-gb200-nvlink-nvidia-mnnvl-6d88b1ea/images/03.webp)

::: image-caption
bisect 流量模式示意。N 张 GPU 被分成两组，GPU X 与 GPU $(X+N/2) \bmod N$ 配对互写，单次测量即覆盖全部链路，这是 3 秒冒烟的数学基础。
:::

### 2.2 大域全面体检

> 当新集群验收，需要接近 pairwise 的覆盖度但耗时可接受。

操作：执行 `nvloom_cli -s gpu-to-rack rack-to-rack fabric-stress`。gpu-to-rack 定位慢卡，rack-to-rack 测机架间聚合带宽，fabric-stress 全员同时收发打满 fabric。

预期效果：线性时间拿到每张 GPU 对每个机架的中位带宽； fabric-stress 给出 fabric 饱和下的真实吞吐，直接对照硬件标称值。

![文章插图](/articles/2026-08-05-72-gb200-nvlink-nvidia-mnnvl-6d88b1ea/images/04.webp)

::: image-caption
fabric-stress 的 fully-random 流量模式。每张 GPU 随机选一个非自身对端并发拷贝，一张 GPU 可同时服务多个对端，用最少的协调制造最接近真实负载的 fabric 压力。
:::

![文章插图](/articles/2026-08-05-72-gb200-nvlink-nvidia-mnnvl-6d88b1ea/images/05.webp)

::: image-caption
fabric-stress 的 random-permutation ring 模式。GPU 随机排列后首尾相连成环，第 n 张 GPU 是第 $n-1$ 张的写入目的地、同时是第 $n+1$ 张的写入来源，每张卡恰好承担一进一出两份流量，适合观察均匀负载下的链路一致性。
:::

### 2.3 复现特定故障流量

> 监控发现某机架在 all-to-one 模式下掉带宽，预置套件里没有完全匹配的模式。

操作：编写 CSV 输入文件，每行一条拷贝（源 GPU、目的 GPU、内存类型、ce/sm、buffer 大小），需要并发的拷贝标相同 measurement_id，执行下面：

```
nvloom_cli --csvInputFile pattern.csv --csvOutputFile result.csv
```

预期效果：输出文件在末列追加每条拷贝的实测带宽（GB/s）或延迟（ns），输出文件可直接作为下一次运行的输入，形成回归基线。

### 2.4 结果可视化对比

> 拿到 pairwise 或 gpu-to-rack 的文本输出，要快速看出矩阵里的异常点，或对比两次跑分。

操作：把 nvloom_cli 输出管道给 `plot_heatmaps.py`，用 `--heatmap_lower_limit` 和 `--heatmap_upper_limit` 固定色阶范围。

预期效果：每个用例生成一张热力图，固定色阶后两次运行的图可直接并排比较；加 `--histogram` 可切换为分布直方图。

![文章插图](/articles/2026-08-05-72-gb200-nvlink-nvidia-mnnvl-6d88b1ea/images/06.webp)

::: image-caption
表 1：九套 suite 速查表。时间复杂度决定适用规模：$O(n^2)$ 套件用于小域精确测量，$O(n)$ 与 $O(1)$ 套件用于大域快速覆盖。bisect 等流量模式隶属于 fabric-stress 等套件，以 `-t` 用例名单独调用，不构成独立 suite。
:::

## 三、代码剖析

### 3.1 机制一：自旋 kernel 同步——让 72 张卡在同一拍起跑

> 并发带宽测试的前提是"同时"：如果各 GPU 的拷贝先后起跑，测到的就不是 fabric 的饱和能力。主机侧 barrier 只同步 CPU，GPU 流上的工作入队有先后，nvloom 需要一个 GPU 侧的发令枪。

**方案是两级自旋 kernel（spin kernel）**。每个执行进程在自己的流上、正式拷贝之前插入一个单线程 kernel，它自旋等待一块跨节点可见的标志位；MPI barrier 确认所有进程就绪后，由第一个拷贝的执行进程（copies[0].executingMPIrank，不一定是 rank 0）的主机线程写标志，所有 GPU kernel 几乎同时放行，排在后面的拷贝随即开始。代码注释明确说这不符合 CUDA 编程模型😅，但对基准测试是有效工具。

```
// `doBenchmark` 的完整流程，library/nvloom.cppbenchmarkState.fillBuffers();          // 源/目的缓冲区写入可校验的伪随机模式MPI_Barrier(MPI_COMM_WORLD);           // 所有进程缓冲区就绪benchmarkState.doWarmupCopies();       // 预热，不计入测量benchmarkState.blockStreams();         // 每条流插入自旋 kernel，挂住后续工作benchmarkState.scheduleWorkPreBlock(); // 正式拷贝入队，排在自旋 kernel 之后MPI_Barrier(MPI_COMM_WORLD);           // 所有进程的流都已挂起等待benchmarkState.releaseStreams();       // 主机写标志位：发令枪响
```

`blockStreams` 把每个流的自旋 kernel 都指向同一块 blockingVar，但只在 copies[0] 执行进程的主机侧留一个可写的副本，`releaseStreams` 翻转它即全员放行。

```
// 自旋 kernel 本体，library/kernels.cuif (latch1) {                // 仅 copies[0] 的执行进程传入 latch1    while (!*latch1) { ... } // 自旋等待主机放行，带 clock64 超时防死锁    *latch2 = 1;             // 通过 MNNVL 可见内存写 device 侧标志}while (!*latch2) { ... }     // 所有进程自旋等待 device 标志
```

这里两级设计的原因是说，**主机写 device 内存的延迟不可控，让一个进程先等主机信号、再通过 NVLink 可见的 device 内存广播，比所有进程直接等主机写更整齐**。latch1 仅 copies[0] 的执行进程持有，其余进程传 nullptr 直接跳过第一阶段。

代价是 GPU 空转，且妨碍 Nsight 等 profiler 工作，所以库提供 `--enableProfiling` 关闭自旋 kernel，同时保留相关内存分配，尽量不改变被测行为。

### 3.2 机制二：负载维持——快慢拷贝之间不留空窗

> 并发测量多条拷贝时，各条拷贝几乎不会同时结束。先结束的 GPU 一旦空闲，fabric 负载下降，后结束的拷贝会测出偏高的带宽，数据失真。

**nvloom 的对策是负载维持（load sustainment），某条流的正式拷贝结束后，立刻往这条流补发同样的拷贝，每次补一发，用两个 CUDA event 双缓冲追踪补发工作的进度，直到所有进程都完成正式测量**。额外拷贝不记录 event，不计入带宽结果。

```
// 核心循环（library/nvloom.cpp，节选并注释）while (true) {    // 轮询各流：正式拷贝（end event）完成的，计入 done    if (isEventDone(filteredEndEvents[i])) { filteredStreamDone[i] = true; pendingStreams--; }    if (pendingStreams == 0)  { MPI_Ibarrier(...); pendingStreams = -1; } // 本进程完成，通知其他进程    if (pendingStreams == -1) { MPI_Test(...); if (flag) break; }         // 全员完成，收工    // 已完成的流：双缓冲补发一发拷贝，维持 fabric 负载    if (isEventDone(filteredLoadSustainmentEvents[i][eventId])) {        doMemcpyInSpinKernel(filteredCopies[i], filteredStreams[i], 1);        cuEventRecord(filteredLoadSustainmentEvents[i][eventId], ...);    }}
```

::: block-1
设计权衡有三处。第一，每次只补一发而非批量补发，补发的颗粒度越小，全体完成时浪费的额外流量越少。第二，退出条件用 MPI_Ibarrier 加 MPI_Test 轮询而非阻塞 barrier，主机在等待期间还能继续给流补货。第三，延迟测量和组播归约不支持补发——这两类操作补发会干扰测量本身，所以会有如 `isLoadSustainmentSupported` 用排除清单显式剔除它们（它们是谁？当然是包含 TMA 与 CFT 的归约变体、multimem.ld_reduce、latency），混合场景下只给支持的操作维持负载。
:::

上手也就三步：

- 环境：CUDA 13.0、580 驱动、MPI、CMake 3.20；编译库 `cd library && cmake . && make`，产出 `libnvloom.so`
- 编译 CLI：仓库根目录 `cmake . && make -j`，产出 `nvloom_cli`，仓库附带的 `Dockerfile` 可作容器化起点
- 运行：一进程一 GPU，用 MPI 按 GPU 总数启动；`nvloom_cli -s gpu-to-rack rack-to-rack fabric-stress` 跑大域体检，`nvloom_cli -l` 列出全部用例

两点边界：项目处于早期开发阶段，接口可能变动（2.1.0 刚加入 CFT 与 MPS 支持）；初始化时 `checkSystemConsistency` 会跨进程比对 fabric、EGM、组播支持与驱动版本，不一致直接中止——驱动版本不齐的集群会被拒绝运行，而不是给出不可比的数据。

> 工具已经开源，手上正好有 GB200 NVL72 或更大 MNNVL 域的话，拉下仓库先跑一条 bisect，估计 3 秒就能看到 fabric 的底细。大家上手实验吧。🥱

参考资料

[1] nvbandwidth 做 GPU 间带宽测试: *https://github.com/NVIDIA/nvbandwidth*

[2] performance_and_tuning: *https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html*

::: related-title
**相关推荐**
:::

::: related-links
- **[CudaProfiler：TIRx 轻量化内核级时序剖析工具，精准定位 warp 流水线阻塞与尾延迟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903219&idx=1&sn=f122de6cd22a28e4ede68643b23b5321&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[约束求解赋能 GPU 模糊测试工具 GPU-Fuzz：挖掘深度学习框架 GPU 高危内存错误](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899663&idx=1&sn=41831f6311c438513d5c1732a2e67989&scene=21#wechat_redirect)**

:::
