关键词：:mdnice-bold-italic[CUDA 共享内存]、NVIDIA GPU、**存储体（bank）**、:mdnice-bold-italic[bank 冲突]、**微基准测试**、访问模式

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/01.webp)

- Notes About Nvidia GPU Shared Memory Banks[1]
- https://feldmann.nyc/blog/smem-microbenchmarks
- 8000 字，阅读 24 分钟，**播客 12 分钟**

相关推荐

- [NVIDIA 技术博客：在 CUDA C/C++ 中使用共享内存](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893635&idx=1&sn=f2a2964d653114410949d0a38869e451&scene=21#wechat_redirect)
- [NVIDIA 技术博客：CUDA 全局内存的合并访问，如何最大化 GPU 带宽](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892826&idx=2&sn=3c703707de4d194bfd297ddc849b095d&scene=21#wechat_redirect)
- [从GPU谈异构，这是ARM处理器架构师的趣谈](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885856&idx=1&sn=fe738c91a44d69ddc2e7ea3e1c2ef9f6&scene=21#wechat_redirect)

> 本文以 NVIDIA A100 GPU 为测试平台，通过微基准测试深入探究 CUDA 共享内存的性能特征与访问优化逻辑。

作为 GPU 流式多处理器的高速暂存内存，**共享内存的效能易受存储体架构与访问模式影响**，存储体冲突（bank conflict）是制约其吞吐量的关键瓶颈。测试结果显示：

- 无冲突访问时 32 个线程通道对应 32 个存储体，10 万次迭代仅耗时 0.57 毫秒，达每存储体每时钟周期 1 次加载的峰值性能；
- 存储体冲突会导致访问串行化，最差性能下降 32 倍（耗时 18.2 毫秒）；
- 多播模式（含全线程束广播、任意分组多播）可维持无冲突访问速率；
- 宽向量加载通过硬件调度规避冲突，2 宽向量多播加载能提升数据吞吐量，但 4 宽向量加载未实现同比优化。

::: block-1
:mdnice-bold-italic[研究量化了不同访问模式的性能差异，揭示了共享内存优化的核心逻辑]，为编写高性能 CUDA 内核提供了关键实践依据，明确了**无冲突访问**、**合理利用多播** 与**向量加载** 的优化方向。
:::

## 本文目录

- 一、线程束（Warps）与存储体（Banks）
  - 1.1 线程束（Warp）
  - 1.2 共享内存（Shared Memory）
- 二、无冲突访问（Conflict-Free Accesses）
  - 这里为什么 8 个 warp 没有并行，0.57 毫秒被 8 个 warp 串行？
  - 关键补充：“并行” 的粒度误解 —— 是 “线程级并行”，而非 “warp 级并行”
- 三、存储体冲突（Bank Conflicts）
  - 3.1 同一线程束在不同存储体上发生冲突
  - 3.2 不同线程束在不同存储体上发生冲突
- 四、多播（Multicasts）：warp 内所有线程访问相同地址
- 五、宽向量加载（Wide Vectorized Loads）
- 六、共享向量加载（Shared Vector Loads）
- 关键问题
  - 问题1：不同线程束非重叠存储体冲突下，访问无法重叠的根源及多线程束优化上限影响
  - 问题2：4 宽向量多播加载吞吐量未达预期的核心瓶颈与共享内存带宽突破路径
- 结论

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/02.webp)

## 一、线程束（Warps）与存储体（Banks）

> NVIDIA GPU 和 CUDA 编程模型采用了一种名为 SIMT（单指令多线程）的执行模型。SIMT 扩展了计算机体系结构的弗林分类法[2]，**该分类法根据指令流和数据流的数量描述了四类体系结构**。

弗林分类法中的四类之一 ——SIMD（单指令多数据）常被用于描述像 GPU 这样的体系结构。

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/03.webp)

::: image-caption
弗林分类法[3]根据计算机体系结构能够同时处理的指令流（进程）和数据流的数量对其进行分类，将其分为四类：SISD、SIMD、MISD和MIMD。
:::

::: block-1
但 **SIMD 和 SIMT 之间存在一个细微却重要的区别**。在 SIMD 体系结构中，每条指令对多个数据元素并行执行相同的操作。

- SIMD 通常通过配备向量寄存器和执行单元的处理器来实现；
- 一个标量线程会发出以 SIMD 方式执行的向量指令。

而:mdnice-bold-italic[在 SIMT 体系结构中，【并非】由单个线程发出应用于数据向量的向量指令，而是由【多个线程】向【任意数据】发出通用指令。]，这里没有单个线程发起 “向量指令”，而是由 32 个（如 CUDA 的线程束）独立线程组成 “线程组”，硬件会给这个线程组分配同一条 “通用指令”。
:::

### 1.1 线程束（Warp）

> 编织中有两个术语 “warp” 和 “weft”，即经纱和纬纱，先说后者——纬纱是指填充纱或纬线，而 NVIDIA 则借用了前者 “warp” 这一术语[4]，也是Lindholm 等人介绍 SIMT 的原始论文中的说法：“warp这一术语源于编织，这是最早的并行线程技术”。（NVIDIA 在其《CUDA C++ 编程指南》中继续使用这一引语。）

因为在编织中，它指的是织梭穿过的一组纵向线，经纱也被称为端纱，它们构成了织物的基本结构。

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/04.webp)

::: image-caption
在机织物中，经纱沿着直纹被拉伸。通常需要将这些线排列成相互平行的状态，并让它们沿着织物的布边垂直延伸。经纱沿面料的直纹方向排列。它提供了纵向的强度和稳定性。此外，经纱能更好地承受张力和拉伸，尤其是在织造过程
:::

CUDA 线程按线程束（warp）组织。每个线程束由 32 个线程组成，这些线程同步执行——它们“同时”执行相同指令。

::: block-1
目的是通过 “批量调度” 减少硬件调度开销，提升并行效率。
:::

- Warp（线程束）是一组被一同调度并并行执行的线程。
- 一个线程束中的所有线程都被调度到单个流式多处理器（SM）上。
- **一个 SM 通常执行多个线程束**，至少会执行来自同一个协同线程数组（也称为线程块）的所有线程束。
- 线程束是 GPU 上典型的执行单元。
- 在正常执行过程中，**一个线程束的所有线程会并行执行相同的指令**，这就是所谓的 “单指令多线程”（SIMT）模型。

线程调度程序周期性地从一条 warp 切换到另一条 warp，以达到 SM 计算资源使用的最大化。

::: block-1
具体来说，如果一个线程束的下一条指令因缺少操作数而延迟，就称该线程束处于停滞状态。[5]

:mdnice-bold-italic[当多个线程束被调度到单个流式多处理器（SM）上时，线程束调度器会选择另一个线程束来执行，而不是等待某条指令的结果返回。这种隐藏延迟的方式正是 GPU 实现高吞吐量的原因]，它能确保在执行过程中所有核心始终有工作可做。
:::

因此，最大限度地增加调度到每个 SM 上的线程束数量往往是有益的，这样可以确保 SM 总有一个符合条件的线程束可以运行。

- 线程束被分配指令的周期所占的比例称为发射效率[6]。发射效率衡量的是 warp 调度器通过从符合条件的 warp 发射指令，使执行管道保持忙碌状态的有效程度。
- 线程束调度中的并发程度称为占用率[7]。

### 1.2 共享内存（Shared Memory）

共享内存分为 32 个存储体（bank）。每个存储体是静态随机存取存储器（SRAM）的一个分片，每个存储体每个时钟周期能够处理一个内存请求，那么可加载或存储 4 字节数据（bank 宽度通常是 4 字节）。**这种存储体系统的存在有一个根本原因：硬件并行性。**

::: block-1
warp 内的 32 个线程和 32 个 bank，这两个数字均为 32 并非巧合。**系统设计初衷是：若线程束中的每个线程通道都从“自己对应的”存储体加载数据，那么 32 次加载操作可在单个时钟周期内完成。**
:::

共享内存地址通过函数 `bank_id := (byte_address / 4) % 32` 映射到存储体。这意味着，若我们定义一个 float 类型的共享内存数组：

```
__shared__ float s[64];
```

其元素会按轮询方式（round-robin，依次循环分配的方式）分布在各个存储体中，具体如下：

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/05.webp)

::: image-caption
在共享内存中的 64 个 float 数据：上部分是完整的 64 个数两行，下面是首尾部标识了 bank 编号的局部数据
:::

## 二、无冲突访问（Conflict-Free Accesses）

> 作为基准测试，我们首先设计一个微基准测试程序，让每个线程通道从共享内存的不同存储体加载数据，验证无冲突时的峰值性能：

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/06.webp)

::: image-caption
无 bank conflict：一个 warp 内的 32 个线程对应 32 个不同的 bank_id
:::

```
__global__ void conflict_free() {    __shared__ float s[8][32];    int warp_id = threadIdx.y;    int lane_id = threadIdx.x;    float* ptr = &s[warp_id][lane_id];    int addr = (int)ptr & 0xFFFF;    // 将共享内存地址转为整数并取低 16 位，    // 仅保留共享内存内的偏移（共享内存地址空间较小，    // 低 16 位足够标识），避免全局地址干扰    for (int j = 0; j < num_iters; j++) {        asm volatile ("ld.volatile.shared.f32 %0, [%1];"                        : "=f"(r1)                        : "r"(addr));        // 内联汇编避免编译器优化对访问行为的干扰        // ld.volatile.shared.f32：        //     表示 “volatile 修饰的共享内存单精度浮点数加载”，        //     volatile确保每次循环都真实访问共享内存（而非复用寄存器缓存）；        // 输入输出约束"=f"(r1)和"r"(addr)：        //     指定将加载结果存入浮点寄存器r1，        //     加载地址为addr，确保指令按预期执行。    }}// 启动配置：conflict_free<1, dim3(32, 8)>();// 即 1 个线程块，线程块内线程维度为(x=32, y=8)// 共 32×8=256 个线程，对应 8 个 warp
```

该微基准测试包含一个线程块（thread block），内含 8 个线程束。每个线程束中，每个线程通道都会对自己对应的共享内存存储体执行对齐良好的加载操作。这应该是共享内存的“光速”使用场景。

在 A100 GPU 上，当`num_iter = 100,000`时，该内核执行耗时约 0.57 毫秒。我们有 8 个线程束，每个线程束向每个存储体发起 100,000 次加载操作，因此每个存储体的加载次数为 800,000 次。

::: block-1
（800,000 次/存储体）÷（0.57 毫秒）= 1.4 吉次加载/存储体/秒（gigaloads/bank/second）。
:::

这与 A100 的峰值频率 1.4GHz 相等。因此，本质上我们实现了预期的“每个存储体每时钟周期 1 次加载”。

### 这里为什么 8 个 warp 没有并行，0.57 毫秒被 8 个 warp 串行？

> 8 个 warp 未并行、0.57 毫秒被串行占用的根本原因，是共享内存 32 个存储体的总带宽仅能支撑 “1 个 warp 的无冲突并行访问” 。

- 首先，Block 是软件层面 “可独立调度到 SM 上的最小单位”。`conflict_free<1, dim3(32, 8)>();` 表示是 1 个线程块，同一 Block 的所有线程必须在同一个 SM 上执行，不能跨 SM 拆分，一个 SM 可同时承载多个 Block（只要资源足够）。
- 一个 SM 的硬件资源是有限的（如寄存器数量、Shared Memory 大小、能处理的 Block 数量上限等），**Block 能否被分配到 SM 上，取决于 Block 所需资源是否不超过 SM 的剩余可用资源**。只有当资源满足时，Block 才能在 SM 上 “驻留” 并等待执行。

::: block-1
Warp 周期性切换与资源利用率优化。线程调度程序会周期性地从当前执行的 Warp 切换到其他就绪 Warp[8]（切换延迟极低），:mdnice-bold-italic[如当某个 Warp 因等待共享内存访问结果而暂停时，调度程序会立即切换到另一个就绪 Warp 执行计算，避免 CUDA Core 空闲]，以掩盖内存访问延迟或指令执行延迟，确保 SM 的计算资源始终处于忙碌状态。这是 GPU 具备延迟隐藏能力的关键所在。[9]
:::

### 关键补充：“并行” 的粒度误解 —— 是 “线程级并行”，而非 “warp 级并行”

需区分 GPU 并行的两个不同粒度：

- **线程级并行（Thread-Level Parallelism）**：:mdnice-bold-italic[单个 warp 内的 32 个线程是并行的（通过 32 个存储体同时执行）]，这是 “无冲突访问” 场景下性能高效的核心。
- **warp 级并行（Warp-Level Parallelism）**：:mdnice-bold-italic[多个 warp 的并行需依赖 “非共享资源竞争” 场景] （如计算密集型任务中，warp 间争抢 CUDA Core）。但在共享内存访问场景中，因**存储体带宽是 “稀缺共享资源”，多个 warp 无法并行，只能串行** 调度。

0.57 毫秒的总耗时，正体现了 “**warp 级串行、线程级并行** ” 的混合执行模式 ——8 个 warp 必须按序占用存储体，而每个 warp 内部的线程则能并行完成访问。

## 三、存储体冲突（Bank Conflicts）

### 3.1 同一线程束在不同存储体上发生冲突

> 但如果 32 个线程通道都尝试从同一个存储体加载不同值呢？

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/07.webp)

::: image-caption
all-conflicts
:::

```
__global__ void all_conflicts() {    __shared__ float s[8][32];    int warp_id = threadIdx.y;    int lane_id = threadIdx.x;    float* ptr = &s[lane_id][0];    int addr = (int)ptr & 0xFFFF;    for (int j = 0; j < num_iters; j++) {        asm volatile ("ld.volatile.shared.f32 %0, [%1];"                        : "=f"(r1)                        : "r"(addr));    }}// 启动配置：conflict_free<1,dim3(32, 8)>();// 即 1 个线程块，线程块内线程维度为(x=32, y=8)// 共 32×8=256 个线程，对应 8 个 warp
```

运行该代码（同样`num_iters = 100,000`）耗时 18.2 毫秒——**几乎是无冲突示例的 32 倍**！这是因为在每个线程束中，32 个线程通道都尝试从存储体 0 加载不同值，这些加载操作必须串行执行。

本示例中，所有线程束的所有线程通道都仅从存储体 0 加载数据。**存储体 0 每时钟周期只能处理 1 次加载，因此自然比无冲突版本慢 32 倍**。

### 3.2 不同线程束在不同存储体上发生冲突

> 但如果**不同【线程束】** 在不同存储体上发生冲突呢？比如线程束 0 的所有线程通道尝试从存储体 0 加载，线程束 1 的所有线程通道尝试从存储体 1 加载？

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/08.webp)

::: image-caption
8 个线程束，每个线程束内 32-way 冲突，并且这样的 warp 8 个，依次冲突 bank0 到 bank7
:::

代码中只需将：`float* ptr = &s[lane_id][0];` 改为 `float* ptr = &s[lane_id][warp_id];`

结果显示，每次迭代仍耗时 18.2 毫秒。

::: block-1
尽管每个线程束在不同存储体上发生冲突，但**共享内存系统似乎无法重叠不同线程束的访问操作**。

这有点令人遗憾，但合乎情理——若要支持该功能，硬件实现将变得复杂得多，且成本极高。
:::

## 四、多播（Multicasts）：warp 内所有线程访问相同地址

> 如果一个线程束的所有线程通道都从同一个存储体加载相同地址的数据呢？

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/09.webp)

::: image-caption
线程束内的 32 个线程访问同一个地址，那必然是同一个 bank
:::

在微基准测试代码中，只需将`ptr`设置为`&s[warp_id][0];`

在 A100 上运行（`num_iters = 100,000`）**耗时 0.57 毫秒，与无冲突加载耗时相同**。这是因为:mdnice-bold-italic[硬件只需从存储体 0 加载一次数据，然后将加载的值广播到所有 32 个线程通道]。共享内存执行的加载操作更少（仅 1 个存储体在工作），但每个线程仍能实现每时钟周期 1 次加载。

线程束中所有线程通道加载相同值是最简单的情况。

> 现在考虑多个线程束的情况，**每个线程束部分线程加载相同地址**，即：:mdnice-bold-italic[如果我们在每个线程束中随机挑选几组线程通道，让每组从特定存储体加载相同值呢？]

我们可以通过简单的哈希函数设置`ptr`来实现：

```
// 2654435761u是一个经典的无符号整数哈希常数（源于斐波那契哈希// 或黄金比例哈希），特点是与 2 的幂次（如 2^32）互质，能将输入// 的 lane_id 均匀映射到不同数值，避免结果集中在某一范围，保证后续// 分组的随机性。// 右移 16 位（32 位无符号整数的一半），目的是截取哈希结果的高 16// 位，进一步打乱 lane_id 的原始顺序，增强分组的随机性，同时将结果// 范围缩小（避免后续取模时数值过大）。unsigned int hash = (lane_id * 2654435761u) >> 16;volatile float* ptr = &sh[warp_id][hash % 32];
```

这会形成如下复杂的加载模式：

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/10.webp)

::: image-caption
每个线程束部分线程加载相同地址
:::

在该模式下，部分存储体（如存储体 1）没有加载请求，而其他存储体（如存储体 0）有多个加载请求。需注意的是，这里不存在存储体冲突！只是部分值需要多播到少数线程。实际上，有些值（如`s[15]`）仅由单个线程加载。

::: block-1
令人印象深刻的是，该基准测试执行 100,000 次/线程加载仍耗时 0.57 毫秒。**GPU 的共享内存系统能够以与无冲突加载和全线程束广播相同的速度——执行任意多播操作！** 非常出色。
:::

## 五、宽向量加载（Wide Vectorized Loads）

> 到目前为止，所有微基准测试均仅涉及 32 位加载。这些加载与共享内存存储体架构适配良好，因为每个存储体的宽度也是 32 位。

但 **NVIDIA GPU 也支持更宽的“向量”加载类型**。这类加载指令（如`ld.shared.v4`）可:mdnice-bold-italic[从共享内存将多个连续的 32 位值加载到寄存器中。]

假设我们使用 4 宽向量加载（4-wide vector loads）将 128 个连续的 float 值加载到寄存器中。在此方案下，线程通道`i`将在单个指令中加载`s[4*i]`、`s[4*i+1]`、`s[4*i+2]`和`s[4*i+3]`。

代码如下：

```
__global__ void vectorized_loads() {    __shared__ float sh[8][128];    int warp_id = threadIdx.y;    int lane_id = threadIdx.x;    float4* ptr = reinterpret_cast<float4*>(&sh[warp_id][lane_id * 4]);    int addr = (int)ptr & 0xFFFF;    float4 r;    for (int j = 0; j < num_iters; j++) {        asm volatile ("ld.volatile.shared.v4.f32 {%0,%1,%2,%3}, [%4];"                        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)                        : "r"(addr));    }}
```

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/11.webp)

::: image-caption
仅显示 32 个线程通道中的前 16 个：每个线程加载 4 个 float
:::

我们立即发现一个潜在问题：:mdnice-bold-italic[线程通道 0 和线程通道 8 都从同一组 4 个存储体加载数据]。更糟的是，线程通道 16 和线程通道 24（示意图中未显示）也会从这同一组 4 个存储体加载数据！我们之前已经了解到存储体冲突对性能的负面影响极大，因此这似乎会导致性能灾难。

运行该代码（`num_iter = 100,000`）耗时 2.27 毫秒，是无冲突加载的 4 倍。但每次迭代加载的数据量也是 4 倍（512 字节 vs 128 字节）。因此，仍能实现每线程通道每时钟周期 4 字节加载的吞吐量。

**尽管存在潜在的存储体冲突，但通过精心的硬件加载调度，仍能实现这一“光速”（即无冲突）吞吐量**。:mdnice-bold-italic[每个线程通道需要从 4 个不同的存储体加载数据，但无需按固定顺序加载这 4 个存储体的数据。] 例如：线程通道 0 可在 4 个连续时钟周期内依次加载`s[0]`、`s[1]`、`s[2]`、`s[3]`；而线程通道 8 可在相同的 4 个连续时钟周期内依次加载`s[33]`、`s[34]`、`s[35]`、`s[32]`。

::: block-1
通过**交错访问** 不同存储体，每个线程束可在 4 个时钟周期内加载 512 字节数据，且*无存储体冲突*。

注：我们也可以通过软件实现这一效果！本质是用软件逻辑模拟硬件的冲突规避调度，避免直接使用向量指令可能带来的冲突问题。

```
// 给 8 个线程一组分配 “起始偏移”// 分组逻辑：lane_id / 8将 32 个线程分为 4 组（每组 8 个线程），组号分别为 0（lane 0~7）、1（lane 8~15）、2（lane 16~23）、3（lane 24~31）。// 起始偏移分配：%4确保每组的start值为 0、1、2、3（4 个不同值），即 4 个线程组分别从 “第 0、1、2、3 个存储体” 开始访问。int start = (lane_id / 8) % 4;for (int j = 0; j < num_iters; j++) {    // 完整循环 i 的 4 次 load，确保第 start 组的 8 个线程    // 不会加载相邻的数据对应的同一 bank    for (int i = 0; i < 4; i++) {        int offset = (i + start) % 4;        asm volatile ("ld.volatile.shared.f32 %0, [%1];"                      : "=f"(r)                      : "r"(addr + offset * 4));    }}
```
:::

该代码强制每组 8 个线程通道按循环移位顺序执行 4 次加载操作。其性能与`ld.shared.v4`微基准测试相同。但显然这种写法更繁琐，因此我们不妨直接使用向量加载:)

## 六、共享向量加载（Shared Vector Loads）

> 我们还可以通过向量加载提升共享内存带宽。具体来说，每对相邻的线程通道可从共享内存加载相同的 float 类型 2 向量（2-vector）。

![文章插图](/articles/2025-11-02-bank-conflict-a100-gpu-be1a1ce3/images/12.webp)

```
__global__ void multicast_pairs(float* result) {    __shared__ float sh[8][32];    int warp_id = threadIdx.y;    int lane_id = threadIdx.x;    float* ptr = &sh[warp_id][(lane_id / 2) * 2];z    int addr = (int)ptr & 0xFFFF;    float2 r;    for (int j = 0; j < num_iters; j++) {        asm volatile ("ld.volatile.shared.v2.f32 {%0,%1}, [%2];"                    : "=f"(r.x), "=f"(r.y)                    : "r"(addr));    }}
```

与无冲突加载类似，该代码在`num_iter = 100,000`时的执行耗时也为 0.57 毫秒。

- 从共享内存存储体的角度来看，每个存储体每时钟周期处理 1 次加载操作，与无冲突加载示例相同。
- 但从线程通道的角度来看，每个线程通道每时钟周期可获取 2 次加载的数据量——包括`r.x`和`r.y`。每个值都会多播到加载它的两个线程通道，且不会产生任何性能损失。

::: block-1
这一功能非常强大！尽管每个存储体的实际吞吐量未变，但通过利用这种向量多播，线程能够加载更多数据。
:::

我们也可以尝试使用 4 宽向量加载实现类似效果：

```
__global__ void multicast_quads(float* result) {    __shared__ float sh[8][32];    int warp_id = threadIdx.y;    int lane_id = threadIdx.x;    float* ptr = &sh[warp_id][(lane_id / 4) * 4];z    int addr = (int)ptr & 0xFFFF;    float4 r;    for (int j = 0; j < num_iters; j++) {        asm volatile ("ld.volatile.shared.v4.f32 {%0,%1,%2,%3}, [%4];"                        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)                        : "r"(addr));    }}
```

但在这种情况下，我们观察到性能下降了 2 倍（约 1.14 毫秒）。尽管每条指令加载 4 个值，仍能实现每线程每时钟周期 2 次加载的吞吐量（与 2 宽向量加载示例相同），但相比 2 宽向量加载，并未获得额外的吞吐量提升。

> **作者尚不清楚确切原因，可能是寄存器文件写端口不足**？这无疑是后续将深入研究的问题。

## 关键问题

### 问题1：不同线程束非重叠存储体冲突下，访问无法重叠的根源及多线程束优化上限影响

> 不同线程束在互不重叠的存储体上发生冲突时，A100 GPU 共享内存系统为何仍无法重叠这些访问以提升吞吐量？这一硬件设计限制是否存在底层架构权衡，且会如何影响多线程束高并发场景下的内核优化上限？

A100 GPU 共享内存系统**无法重叠不同线程束的存储体冲突访问**，**核心原因是硬件实现的复杂性与成本权衡**。作者表示，即便不同线程束冲突分布在互不重叠的存储体，仍需 18.2 毫秒（与全冲突耗时一致），这是因为**共享内存的存储体访问调度机制未设计跨线程束的并行冲突处理逻辑** ——若支持该功能，需额外增加硬件调度单元与存储体接口，导致芯片面积、功耗显著上升，不符合 GPU“高性能与成本/功耗平衡”的架构设计原则。

**这一限制直接降低了多线程束高并发场景的优化上限**：即使通过算法将冲突分散到不同存储体，也无法利用线程束并行性抵消冲突损耗，只能通过严格避免单个线程束内的存储体冲突（如按存储体映射规则设计数据访问模式），才能充分释放共享内存吞吐量。

---

### 问题2：4 宽向量多播加载吞吐量未达预期的核心瓶颈与共享内存带宽突破路径

> 4 宽向量多播加载未实现 2 宽向量加载的同等吞吐量提升（仅达每线程每时钟周期 2 次加载），其核心瓶颈是寄存器文件写端口不足还是存储体多播调度机制限制？在复杂数据复用场景中，如何突破该限制以实现共享内存带宽的进一步挖掘？

4 宽向量多播加载未达 2 宽向量的同等吞吐量提升，原文推测核心瓶颈是寄存器文件写端口不足——4 宽向量加载需同时向寄存器写入 4 个数据，而寄存器文件的写端口数量有限，无法满足并行写入需求，导致实际吞吐量未随向量宽度同比提升（仅维持每线程每时钟周期 2 次加载），而非存储体多播调度机制限制（2 宽向量已验证多播调度可高效工作）。

突破该限制的关键方向的是：1. 沿用 2 宽向量加载方案，通过合理数据复用最大化现有带宽；2. 优化数据分组策略，减少单次向量加载的寄存器写入压力；3. 结合软件调度（如原文提到的循环移位加载），拆分 4 宽向量为多次低宽度加载，避免寄存器端口拥堵，从而在不依赖硬件改进的情况下，进一步挖掘共享内存带宽潜力。

## 结论

> 本文通过针对 NVIDIA A100 GPU 的系列微基准测试，明确了 CUDA 共享内存的核心性能特征与优化关键。

测试证实，无冲突访问可实现每存储体每时钟周期 1 次加载的峰值效能（10 万次迭代耗时 0.57ms），而存储体冲突会导致性能暴跌 32 倍。多播模式（含任意分组多播）与 2 宽向量加载能在无性能损耗前提下提升数据吞吐量，宽向量加载可通过硬件调度规避冲突。4 宽向量加载受限于寄存器文件写端口等瓶颈，未实现同比优化。

::: block-1
研究量化了不同访问模式的性能差异，为高性能 CUDA 内核开发提供了明确指引：**规避单线程束内存储体冲突、合理利用多播与 2 宽向量加载，是释放共享内存带宽潜力的核心路径。**
:::

参考资料

[1] smem-microbenchmarks | feldmann.nyc: *https://feldmann.nyc/blog/smem-microbenchmarks*

[2] 弗林分类法: *http://en.wikipedia.org/wiki/Flynn%27s_taxonomy*

[3] flynns-taxonomy: *https://www.geeksforgeeks.org/computer-organization-architecture/computer-architecture-flynns-taxonomy/*

[4] https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp: *https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp*

[5] gpu-glossary/device-software/warp: *https://modal.com/gpu-glossary/device-software/warp*

[6] issue-efficiency: *https://modal.com/gpu-glossary/perf/issue-efficiency*

[7] https://modal.com/gpu-glossary/perf/occupancy: *https://modal.com/gpu-glossary/perf/occupancy*

[8] NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf: *https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf*

[9] warp-scheduler: *https://modal.com/gpu-glossary/device-hardware/warp-scheduler*

更多推荐

- [华为端引擎Bolt GPU性能优化：让上帝帮忙掷骰子](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885381&idx=1&sn=cab9666a15cc0012732b537c52d9ca89&scene=21#wechat_redirect)
- [移动端GPU矩阵乘优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885554&idx=1&sn=7f3078d27555c151e05aa967b5de68b8&scene=21#wechat_redirect)
- [华为诺亚方舟实验室 Bolt v1.2.1 在 QualComm GPU 上的优化总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886197&idx=1&sn=5e99d9a8fc07d16047e4924b72830bf4&scene=21#wechat_redirect)
