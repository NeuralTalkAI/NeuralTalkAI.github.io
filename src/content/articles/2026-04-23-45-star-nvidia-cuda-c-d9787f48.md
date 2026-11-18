关键词：CUDA 内核优化、**AI 智能体自主循环**、:mdnice-bold-italic[Claude Code 插件]、基准驱动试验、GPU 性能调优

> GPU 内核优化是高性能计算领域最考验功力的手艺活——向量化加载、共享内存分块、寄存器压力、访存合并……每一项都需要工程师对硬件架构的深刻理解和大量 profiling 迭代。

![文章插图](/articles/2026-04-23-45-star-nvidia-cuda-c-d9787f48/images/01.webp)

::: image-caption
作者信息：Bryce Adelstein Lelbach 是 C++ 领域极具影响力的核心推动者。他担任ISO C++库演进工作组主席、美国编程语言标准委员会主席，任职于NVIDIA，主导CUDA C++核心库开发。同时，他也是CppCon、C++Now等顶级C++大会的程序主席，r/cpp论坛版主与include_cpp活动组织者，深度参与C++标准化进程与社区生态建设，推动着现代C++的演进与普及。
:::

::: block-1
然而，这个过程有一个显著特征：它本质上是一个"假设 → 修改 → 编译 → 验证 → 测量 → 保留或回滚"的循环，而**循环恰恰是机器最擅长的事情。**
:::

![文章插图](/articles/2026-04-23-45-star-nvidia-cuda-c-d9787f48/images/02.webp)

- **brycelelbach/autocuda: Skills and tools for automatically writing and optimizing CUDA kernels**
- 开源代码：https://github.com/brycelelbach/autocuda
- 4000 字，阅读 15 分钟，**播客 18 分钟**

相关推荐

- [暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)
- [算子 编译器天塌了吗。。NVIDIA 提出 AVO：当 AI Agent 成为进化算子，探索突破 Blackwell 性能极限，超越 FlashAttention‑4 最高10.5%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899076&idx=1&sn=64a6532410f9d22f65043945582d4f1a&scene=21#wechat_redirect)
- [算子工程师末日。。。NVIDIA提出SOL-ExeBench：对标硬件效率极限，生成真实 GPU Kernel！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898808&idx=1&sn=0c8886f90fc178c6ae7aedbf1931e0e9&scene=21#wechat_redirect)

> autocuda 正是抓住了这一本质：它把 CUDA 内核优化包装成一个让 AI 智能体无人值守运行的试验闭环——**你给它一个 kernel 和一个固定不变的 benchmark，它自己改代码、跑测试、量性能、决定去留，然后继续下一轮，永不停歇。**

这不是一个代码生成工具，而是一个真正意义上的**自主优化系统**。让我们拆开它的每一层设计，看看这套机制究竟如何运转。

## 本文目录

- 快速上手
- 一、架构总览与设计哲学
  - 1.1 项目目录结构
- 二、两大核心技能：discover 与 optimize
  - 2.1 discover：项目探索与环境标定
  - 2.2 optimize：永不停歇的试验循环
- 三、示例内核的代码剖析
  - 3.1 memcpy：最简单的带宽测试
  - 3.2 matmul：经典的计算密集型挑战
  - 3.3 stencil 与 sigmoid：访存模式的两个极端
- 四、构建系统与可视化基础设施
  - 4.1 CMake 构建：FetchContent 自动拉取 nvbench
  - 4.2 实时 Dashboard：零依赖的试验监控
- 五、日志与试验管理的工程细节
  - 5.1 CSV 日志的 schema 契约
  - 5.2 Git 分支即试验记录
  - 5.3 上下文压缩的容错处理
- 六、bootstrap 一键部署
- 七、总结与展望

![文章插图](/articles/2026-04-23-45-star-nvidia-cuda-c-d9787f48/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> autocuda 作为 Claude Code 的插件运行，依赖一块 NVIDIA GPU、CMake 3.30+ 和 CUDA 工具链。以下是最精简的上手步骤：

```
# 1. 安装插件（需先安装 Claude Code）/plugin marketplace add brycelelbach/autocuda/plugin install autocuda@brycelelbach-autocuda# 2. 克隆项目并构建示例内核git clone https://github.com/brycelelbach/autocuda.gitcd autocudacmake -B build -S . -DCMAKE_BUILD_TYPE=Releasecmake --build build --parallel# 3. 验证基准能跑通./build/bench_memcpy# 4. 在 Claude Code 中启动自主优化/autocuda:discover    # 探索项目结构，生成 project-layout.md/autocuda:optimize    # 启动无限试验循环
```

若需要一键无人值守部署（含 Claude Code 安装 + 插件注册），可直接执行：

```
curl -fsSL https://raw.githubusercontent.com/brycelelbach/autocuda/main/bootstrap.bash | bash
```

更多构建选项与环境细节，参考 `README.md`。

## 一、架构总览与设计哲学

> autocuda 的灵感来自 karpathy/autoresearch[1]，但目标从语言模型训练转向了 GPU 内核性能。整个项目的设计哲学可以浓缩为三句话：

1. **基准即真理** ——benchmark harness（`bench.cu`）和构建系统（`CMakeLists.txt`）是只读的，不可被 AI 修改，这保证了测量结果的一致性与可重复性。
2. **先发现再优化** ——通过 `discover` 和 `optimize` 两个独立技能，将"理解项目"和"优化代码"解耦，避免优化循环每次重新推导项目结构。
3. **永不停机** ——优化循环没有内部终止条件，只有人类可以中断它。失败的试验不是浪费，而是排除假设空间的数据。

### 1.1 项目目录结构

> 这里最精妙的设计在于 `bench.cu` 和 `kernel.cuh` 的分离。

```
.claude-plugin/marketplace.json    ← 插件市场清单plugins/autocuda/  skills/    discover/SKILL.md              ← 项目发现技能的完整 prompt    optimize/SKILL.md              ← 自主优化技能的完整 promptkernels/                           ← 四个示例 CUDA 工作负载  memcpy/  stencil/  matmul/  sigmoid/    bench.cu                       ← 只读的 nvbench 基准 harness    kernel.cuh                     ← AI 可以修改的内核源码CMakeLists.txt                     ← 构建系统（只读）dashboard.py                       ← 实时监控面板ci/plot_progress.py                ← 试验进度可视化bootstrap.bash                     ← 一键部署脚本
```

以 memcpy 为例，`bench.cu` 是固定的 nvbench harness，负责分配内存、启动 kernel、验证正确性、报告带宽；`kernel.cuh` 则是 AI 的"画布"，block size、向量化策略、循环结构全部可以改。

## 二、两大核心技能：discover 与 optimize

> autocuda 的核心不是 CUDA 代码本身，而是两份精心编排的 **SKILL.md** ——它们是写给 Claude 的行为指令，定义了 AI 智能体在每个阶段该做什么、怎么做、不该做什么。

### 2.1 discover：项目探索与环境标定

`discover` 技能的职责是把一个陌生的 CUDA 项目变成一份结构化的 `project-layout.md` 文件。它的工作流程如下：

1. **探索**：读 README、构建脚本、CI 配置，识别源文件、验证代码、基准代码、构建系统、依赖项
2. **环境设置**：检查 CUDA 工具链、安装依赖、配置构建、编译、运行验证和基准
3. **验证 profiler**：编译一个最小的 hello-world CUDA 程序，分别用 `nsys` 和 `ncu` 做 profiling，确保后续优化循环可以做性能剖析

这第三步尤为关键。`discover` 技能的 SKILL.md 中详细列出了 profiler 的常见失败模式和修复手段：

```
// 来源：plugins/autocuda/skills/discover/SKILL.md（节选）- `ERR_NVGPUCTRPERM` — re-run under `sudo -E ncu --set full /tmp/autocuda-hello`- `nsys`/`ncu` "command not found" — they usually live under `/usr/local/cuda*/bin`- Permission errors writing the profile report — pass `--output /tmp/<name>`- `ncu: error while loading shared libraries ...` — set `LD_LIBRARY_PATH`- `kernel.perf_event_paranoid` too strict — `sudo sysctl -w kernel.perf_event_paranoid=1`
```

最终输出的 `project-layout.md` 包含极其详细的结构化信息：可编辑文件列表、只读文件列表、精确的构建命令、每个 benchmark 的指标/单位/方向/轴/聚合方式、超时限制、GPU 环境信息、以及经过验证的 profiler 命令。这份文件是 `optimize` 技能的"操作手册"。

### 2.2 optimize：永不停歇的试验循环

`optimize` 技能是整个系统的灵魂。它读取 `project-layout.md`，然后进入一个**无限循环**：

```
读取源码和日志 → 提出假设 → 修改代码 → 编译 → 验证正确性 → 运行 benchmark→ 提取指标 → 计算聚合 → 记录日志 → 改进则提交/退步则回滚 → 回到第一步
```

> 这个循环有几个值得深挖的设计细节：

**假设驱动而非清单驱动**。SKILL.md 明确要求 AI 不能按固定的优化技巧清单逐一尝试，而必须从当前代码、指标类型和试验历史中推断瓶颈，形成假设并验证：

```
// 来源：plugins/autocuda/skills/optimize/SKILL.md（节选）Do not follow a fixed checklist of optimization tricks. Infer what to try nextfrom the evidence:- The current source code — what is the actual bottleneck?- The metric — bandwidth-bound, compute-bound, and latency-bound workloads  need fundamentally different strategies.- The trial history — what has been tried, what worked, what failed?
```

**强制不停机机制**。SKILL.md 用了大量篇幅禁止 AI 以各种方式"停下来"：不能问用户是否继续、不能宣布完成、不能写总结报告、不能声称优化已穷尽、不能因为连续回归就停止。甚至贴心地写道："如果你发现自己正在打'这就是我能做到的极限'这种话😭，**删掉它**，然后开始下一次试验。"🤣

**卡住时的系统化突围算法**。当连续 3 次以上试验回归时，SKILL.md 给出了明确的 7 步脱困流程：立即 profiling → 重读全部源码 → 反转策略方向 → 枚举未尝试的优化家族 → 组合近似成功的试验 → 切换 benchmark → 回到循环。

## 三、示例内核的代码剖析

### 3.1 memcpy：最简单的带宽测试

> memcpy kernel 是整个项目中最基础的工作负载，目标是测量设备到设备的内存拷贝带宽：

```
// 来源：kernels/memcpy/kernel.cuhstaticconstexprint BLOCK_SIZE = 256;template<typename T>using restrict_span = cuda::restrict_mdspan<T, cuda::std::dextents<std::size_t, 1>>;template<typename T>__global__ void kernel(restrict_span<const T> src,                       restrict_span<T>       dst){    conststd::size_t num_elements = src.extent(0);    conststd::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x                               + threadIdx.x;    conststd::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;    for (std::size_t i = tid; i < num_elements; i += stride) {        dst[i] = src[i];    }}
```

几个值得注意的点：

- 使用了 CUDA 的 `restrict_mdspan`（来自 `<cuda/mdspan>`），这是 C++23 mdspan 的 CUDA 扩展，`__restrict__` 语义告诉编译器 src 和 dst 不会别名，有利于生成更激进的优化代码。
- 基线实现极其朴素——逐元素拷贝，没有向量化加载（`float4`）、没有异步拷贝（`cp.async`）。这正是留给 AI 的优化空间。
- 支持 5 种数据类型：`int8_t`、`__half`、`float`、`double`、`cuDoubleComplex`，通过显式模板实例化保证跨类型的泛化。

对应的 benchmark harness 则展示了测量框架的精度：

```
// 来源：kernels/memcpy/bench.cu（节选）staticconstexprstd::size_t DATA_BYTES = 2ULL * 1024 * 1024 * 1024; // 2 GiBstate.add_global_memory_reads<T>(num_elements, "DataSize");state.add_global_memory_writes<T>(num_elements, "DataSize");state.exec([src_span, dst_span, num_elements](nvbench::launch& launch) {    constint grid = compute_grid_size(num_elements);    kernel<T><<<grid, BLOCK_SIZE, 0, launch.get_stream()>>>(src_span, dst_span);});// 正确性校验thrust::host_vector<T> h_src(src);thrust::host_vector<T> h_dst(dst);if (!std::equal(h_src.begin(), h_src.end(), h_dst.begin()))    throwstd::runtime_error("Correctness check failed: dst != src");
```

2 GiB 的数据量确保测量的是持续带宽而非缓存效应，nvbench 自动管理预热和多次采样。`bench.cu` 还特别处理了 Nsight Compute 场景——检测到在 `ncu` 下运行时，只执行一次并跳过计时，避免 profiling 开销干扰结果。

### 3.2 matmul：经典的计算密集型挑战

> 矩阵乘法的基线实现是教科书级别的朴素版本：

```
// 来源：kernels/matmul/kernel.cuhstatic constexpr int BLOCK_DIM = 16;template<typename T>__global__ void kernel(restrict_span_2d<const T> A,                       restrict_span_2d<const T> B,                       restrict_span_2d<T>       C){    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;    if (row >= M || col >= N) return;    T sum = T(0);    for (int k = 0; k < K; ++k)        sum += A(row, k) * B(k, col);    C(row, col) = sum;}
```

这是一个 $O(M \times N \times K)$ 的朴素实现——没有共享内存分块、没有寄存器级 tiling、没有 Tensor Core。对于 AI 智能体来说，优化空间巨大：

- 共享内存 tiling 可以将全局内存访问量从 $O(M \times N \times K)$ 降至 $O(M \times N \times \sqrt{K})$；
- 更进一步可以尝试 CUTLASS 风格的多级 tiling、WMMA API 利用 Tensor Core、甚至直接调用 cuBLAS。

SKILL.md 也明确鼓励使用库："**强烈优先使用优化过的库而非手写 CUDA kernel** "。

### 3.3 stencil 与 sigmoid：访存模式的两个极端

> stencil（五点热方程）展示了二维网格上的邻域访问模式：

```
// 来源：kernels/stencil/kernel.cuh（核心计算）T center = u_in(j, i);T left   = u_in(j, i - 1);T right  = u_in(j, i + 1);T up     = u_in(j - 1, i);T down   = u_in(j + 1, i);u_out(j, i) = center + alpha * (left + right + up + down - T(4) * center);
```

这里的优化方向包括共享内存 halo 区域缓存、读写分离的 ping-pong 缓冲、以及通过调整 block 形状改善 L2 cache 局部性。

sigmoid 则是纯粹的逐元素操作，带宽受限，优化重点在于向量化加载和 `__half2` 批量计算：

```
// 来源：kernels/sigmoid/kernel.cuhfor (std::size_t i = idx; i < n; i += stride) {    T x = input[i];    output[i] = T(1) / (T(1) + exp(-x));}
```

::: block-1
四个 kernel 覆盖了 GPU 计算的两大核心场景——带宽受限（memcpy、stencil、sigmoid）和计算受限（matmul），迫使 AI 不能用一招打天下。
:::

## 四、构建系统与可视化基础设施

### 4.1 CMake 构建：FetchContent 自动拉取 nvbench

```
// 来源：CMakeLists.txtfunction(add_kernel_bench name)    set(dir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${name})    add_executable(bench_${name} ${dir}/bench.cu)    target_link_libraries(bench_${name} PRIVATE nvbench::main)    target_include_directories(bench_${name} PRIVATE ${dir})endfunction()add_kernel_bench(memcpy)add_kernel_bench(stencil)add_kernel_bench(matmul)add_kernel_bench(sigmoid)
```

`add_kernel_bench` 函数将每个 kernel 目录封装为独立的可执行目标，nvbench 通过 `FetchContent` 自动从 GitHub 拉取。这种设计让新增 kernel 只需一行 `add_kernel_bench(xxx)` 和对应的目录。

### 4.2 实时 Dashboard：零依赖的试验监控

> `dashboard.py` 是一个纯标准库实现的 HTTP 服务器（无第三方依赖），读取 `experiments/*.csv` 日志并渲染为实时刷新的 HTML 页面：

```
# 来源：dashboard.py（核心渲染逻辑节选）def compute_speedup(metric, baseline, lower_better):    if lower_better:        return baseline / metric    return metric / baseline
```

Dashboard 会自动推断每个 benchmark 的优化方向（值越大越好还是越小越好），方法是查看 `improved` 状态行的平均值是否低于 baseline。它为每个 benchmark 绘制独立的加速比曲线，所有曲线共享 x 轴（墙钟时间）和 y 轴（1.0 = baseline），并用颜色编码试验状态：绿色=改进、橙色=回归、红色=错误、灰色=baseline。

::: block-1
页面每 10 秒自动刷新，且通过 `snapshot/apply` 机制保留用户展开/折叠的 `<details>` 状态，避免刷新时丢失交互状态。这个细节体现了工程品味——优化可能跑几个小时，你需要一个不打扰你的监控界面。
:::

## 五、日志与试验管理的工程细节

### 5.1 CSV 日志的 schema 契约

`optimize` 技能要求在第一行 baseline 写入后"锁定 schema"——列顺序、精度、命名约定在整个运行期间不可变更。这是一个看似不起眼但极其重要的设计：AI 智能体可能经历数百次试验和多次上下文压缩（context compaction），schema 漂移会导致 dashboard 解析崩溃。

### 5.2 Git 分支即试验记录

每次启动优化会创建 `experiment/<tag>` 分支，每次"改进"都是一次 git commit，回归则 revert。这意味着：

- `git log` 就是赢家变更的清晰历史
- `git diff experiment/<tag>~3..experiment/<tag>` 可以看最近 3 次改进的累积变化
- 任何人都可以 code review AI 的优化决策

### 5.3 上下文压缩的容错处理

长时间运行不可避免会触发 LLM 的上下文压缩。SKILL.md 为此设计了专门的恢复协议：检测到压缩后，重新读取日志的头 5 行和尾 20 行，写入一行 `compaction` 标记，然后继续循环。这保证了即使 AI 的记忆被截断，行为仍然一致。

## 六、bootstrap 一键部署

`bootstrap.bash` 脚本展示了生产级的自动化部署思维：

```
# 来源：bootstrap.bash（核心流程）main() {    install_claude          # 幂等安装 Claude Code    write_settings          # 写入 ~/.claude/settings.json（备份旧文件）    update_bashrc           # 追加环境变量（通过标记行保证幂等）    ensure_claude_on_path   # 确保当前脚本能调用 claude    install_plugin          # 注册市场并安装插件}
```

脚本设置了 `bypassPermissions` 模式和 `CLAUDE_CODE_SANDBOXED=1` 环境变量，让 AI 智能体可以无需人工确认地执行命令——这是"无人值守"的基础设施保障。同时通过 `BASHRC_MARKER_BEGIN/END` 标记保证多次运行不会重复追加。

## 七、总结与展望

> autocuda 的真正创新不在于它的 CUDA 代码有多精巧——事实上，它提供的基线 kernel 故意写得朴素。它的价值在于**将 GPU 内核优化这个高度专业化的迭代过程，系统性地编码为一个 AI 智能体可以自主执行的闭环协议**。

从工程角度看，这个项目展示了 AI coding agent 的一种非常具体的应用模式：不是一次性生成代码，而是在一个有严格正确性约束（validation）和可量化目标函数（benchmark metric）的搜索空间中，做持续的假设-实验-学习循环。这种模式天然适合 GPU 内核优化，但同样可以推广到编译器 pass 调优、数据库查询优化、网络协议参数调整等任何有明确度量指标的性能工程领域。

项目仅诞生两周，star 数已过 45，代码量精简但设计完备。如果你有一块 GPU 和一个想加速的 kernel，不妨让 Claude 跑一晚上试试——也许你醒来时，性能已经提升了。

参考资料

[1] karpathy/autoresearch: *https://github.com/karpathy/autoresearch*
