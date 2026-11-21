**投稿/自荐/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：摩尔线程、**MUSA 驱动**、基准测试、:mdnice-bold-italic[Celero 框架]、**CUDA 对照评测**

> *当我们谈论一块 GPU "快不快"，惯性反应往往是去看 TFLOPS、HBM 带宽，或跑一遍 GEMM、ResNet。**可真正决定上层框架体验的，往往是那些藏在驱动深处的细节**：`cudaMalloc(1B)` 要花多少微秒？:mdnice-bold-italic[空 kernel 的 launch latency 是 5μs 还是 15μs？]*

两路 stream 通过 event 串行起来比单 stream 多了多少同步开销？这些"边角料"在大模型训练里被放大成性能黑洞。

![文章插图](/articles/2026-06-03-a100-gpu-gpu-compute-driver-2812978f/images/01.webp)

- :mdnice-bold-italic[GPU Compute Driver Bench 是一套全维度性能评测套件，立足实际开发与工程落地场景，专门用于评测 GPU 计算驱动的综合性能。套件同时兼容摩尔线程 MUSA 驱动与各类 CUDA 适配型 GPU 驱动，可开展结果公允、可复现的跨平台基准测试。]
- 项目代码：https://github.com/MooreThreads/gpu-compute-driver-bench
- 5000 字，阅读 19 分钟，**播客 26 分钟**

相关推荐

- [赋能 Linux GPU 生态：NVIDIA 开源 GPU 内核模块的架构、功能与技术实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893566&idx=1&sn=1cadc8c260f80856125b710ce6bc4249&scene=21#wechat_redirect)
- [NVIDIA ComputeEval：从基准建立到难度升级，全面评估 LLMs 的 CUDA 代码生成能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894186&idx=1&sn=4bcf0b3458bf4a638fc3d18efc85e67c&scene=21#wechat_redirect)
- [暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)

> 摩尔线程去年年底开源的 `gpu-compute-driver-bench`[1] 正是为了把这些黑洞照亮——它不测峰值算力，而专门拿"驱动"这一层做对照实验，让 MUSA 与 CUDA 在同一套测试框架下被公平地横向打分。本文将从代码层面拆解它的设计哲学、关键实现以及自动评分机制。

## 本文目录

- 快速上手
- 一、架构设计
  - 1.1 为什么是"驱动"基准而不是"算力"基准
  - 1.2 顶层目录与角色
  - 1.3 一条贯穿全栈的"对照实验"思路
- 二、底座 common：被改造过的 Celero
  - 2.1 借 Celero 之骨，添驱动测试之肉
  - 2.2 测试夹具的统一形态
- 三、调度子系统：驱动级"暗时间"如何被精确测量
  - 3.1 用驱动 API 测最纯的 launch latency
  - 3.2 与之相对的 `kernelLaunchThroughputMode`
  - 3.3 双 stream 同步开销的对照实验：`efficiencyOfSync.cu`
  - 3.4 GPU 命令处理器的 "kernelGap" 体检
- 四、自动评分系统：是如何把"跑数"变成"一个分数"的
  - 4.1 三套件、加权基线分
  - 4.2 评分核心：log10 相对比率法
  - 4.3 表格对齐与逐元素累加
- 五、跨平台兼容：一份代码，三种二进制
- 六、可观察性与正确性：藏在细节里的"较真"
  - 6.1 警告与剔除：错位数据零容忍
  - 6.2 异常值的就地处理
  - 6.3 直方图：把分布也画出来
- 七、它告诉了我们什么

![文章插图](/articles/2026-06-03-a100-gpu-gpu-compute-driver-2812978f/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> 仓库自带 `install.sh` 一键脚本，依赖少、流程清爽。以 CUDA 平台为例（MUSA 平台把 `-n` 换成 `-m` 即可）：

```
# 1. 基础系统依赖sudo apt install libcpuid-dev libeigen3-dev hwloc \                 libblas-dev libopenblas-base libopenblas-dev# 2. 编译（NVCC 路径；MUSA 用 -m，MetaX MACA 用 -x）./install.sh -n -j16# 3. 运行单个 case，结果落到 CSV./build/schedule/kernelLaunchLatencyMode -t latency.csv# 4. 一键跑完一整套（memoryOp / mulStreams / graphAndSchedule）cd scripts/python3 autorun.py --suits graphAndSchedule# 5. 对比 baseline 自动评分python3 calculateScoreOfSuit.py --base ../baseline/ --test ../result/
```

- 跨平台移植的细节（比如 musify-text 的 mapping 配置、MUSA→CUDA 的源码转换）可参考 `scripts/porting2cuda.sh`[2] 与 `scripts/porting2musa.sh`[3]；
- 架构识别与 `mp_xx` 自动选择逻辑见根目录 `CMakeLists.txt`[4] 中的 `MTGPU_ARCH_MAPPING` 段落；
- 评分系统的更多细节可参考 `TestSuitConfig.json`[5] 与 `scripts/calculateScoreOfSuit.py`[6]。

## 一、架构设计

### 1.1 为什么是"驱动"基准而不是"算力"基准

> 业界已有一大批 GPU benchmark：`mixbench`、`gpu-burn`、`cuda-samples/bandwidthTest`、MLPerf……它们大多盯着**硬件能力** ——浮点峰值、显存带宽、端到端模型吞吐。:mdnice-bold-italic[但深度学习框架在算子之间真正消耗的"碎片时间"，是另一类成本：]

- **API 路径成本**：一次 `cuLaunchKernel` 从用户态走到 GSP/MMU 再到 SM 调度，要经过运行时、驱动、固件多层；
- **资源管理成本**：`cudaMalloc` / `cudaFree`、event/stream 创建销毁、module load；
- **协同/同步成本**：跨 stream 依赖、graph 优化、P2P 拷贝、IPC 句柄。

这些点在 GEMM benchmark 里被均摊到不可见，但在 LLM 训练里被一万个小算子放大成显著的"气泡"。`gpu-compute-driver-bench` 的核心定位，就是**精确度量驱动层在这些"碎片场景"中的开销**。

### 1.2 顶层目录与角色

> 仓库根目录的子模块边界非常清晰：

| 目录 | 角色 |
| --- | --- |
| `common/` | 基于 Celero[7] 改造的基准测试底座，提供 fixture/timer/result table 等基础设施 |
| `schedule/` | kernel launch、graph、event、stream 调度类 case |
| `memory/` | malloc/free、1D/2D/3D copy、pinned、registered、IPC 等内存类 case |
| `multicards/` | P2P、跨卡 set/launch |
| `resource/` | event/stream 句柄管理 |
| `scripts/` | autorun、自动评分、可视化、MUSA↔CUDA 源码转换 |
| `baseline/` | 标定基线（默认 cudaA100） |
| `TestSuitConfig.json` | 三大套件与每个 case 的权重配置 |

### 1.3 一条贯穿全栈的"对照实验"思路

> 整个项目最关键的设计哲学是：**不直接报告"我们多快"，而是报告"我们相对基线的相对值"。**

`baseline/` 目录预置了 cudaA100 基线表，`calculateScoreOfSuit.py` 会把当前测试结果按列对齐做相对比值，再映射成分数（详见 第四章[8]）。这种做法回避了"绝对性能不可比"的根本困境——不同代际、不同架构的 GPU 直接比 `μs` 是没有意义的，比"**相对于 A100 是多少倍** "才有工程价值。

## 二、底座 common：被改造过的 Celero

### 2.1 借 Celero 之骨，添驱动测试之肉

> Celero 是 C++ 微基准库，提供 `BASELINE` / `BENCHMARK` 宏、Fixture 生命周期与统计聚合。

本仓库把它静态库化（`benchmark_common`），并叠加了一些 GPU 专属的扩展——最直接的就是 `UserDefinedMeasurement` 的子类化：每个 case 都会声明若干 `UDMGPUTime`、`UDMBandWidth`、`UDMThroughPut` 用作"自定义列"，最终被写入 CSV 表头与每一行数据。

注意到代码里头一个不起眼的小约定：表头以 `*` 开头的列会被打分脚本识别为"焦点列"（focus column），见 `find_focus_columns`：

```
# 来源：scripts/calculateScoreOfSuit.pydef find_focus_columns(header_row):    focus_columns = []    for index, column_name in enumerate(header_row):        if column_name.startswith('*'):            focus_columns.append(index)    return focus_columns
```

这意味着 case 作者通过给 measurement 起一个 `*TP(s^-1)`、`*Bmall` 这样的名字，就能把"这个指标参与评分"这件事写在代码里。**命名即语义**，是这个基准框架里非常 lightweight 的一处巧思。

### 2.2 测试夹具的统一形态

> 每个 case 文件几乎都是同一种骨架：继承 `TestFixture`、暴露 `getExperimentValues()` 定义参数扫描空间、在 `setUp`/`onExperimentStart` 里准备 stream/event，最后用 `BASELINE_F` 注册测试体。以 malloc benchmark 为例：

```
// 来源：memory/MallocRateFrom1Bto4GB.cppclass MallocFixture :public TestFixture {public:    // 参数空间：1B、2B、4B … 一路扫到 4GB    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {        std::vector<TestFixture::ExperimentValue> problemSpace;        for (int64_t elements = 1LL << 0; elements <= (1LL << 32); elements *= 2) {            problemSpace.push_back(elements);        }        return problemSpace;    }    std::shared_ptr<UDMBandWidth> uband1{new UDMBandWidth("*Bmall")}; // 焦点列    std::shared_ptr<UDMGPUTime>   utime1{new UDMGPUTime("tmall")};    std::shared_ptr<UDMBandWidth> uband2{new UDMBandWidth("*Mfree")};    std::shared_ptr<UDMGPUTime>   utime2{new UDMGPUTime("tfree")};};BASELINE_F(musa_malloc, normal_mem, MallocFixture, /*samples*/3, /*iter*/1) {    // 1. warm up，避开首次分配冷启动    void* dTemp;    checkMusaErrors(musaMalloc(&dTemp, 4));    checkMusaErrors(musaFree(dTemp));    // 2. 测 musaMalloc：CPU 计时器卡住一次调用    CPerfCounter timer; timer.Restart();    checkMusaErrors(musaMalloc(&deviceMemory, this->mallocSize));    timer.Stop();    float ms = timer.GetElapsedSeconds() * 1000.f;    this->utime1->addValue(ms);    this->uband1->addValue(static_cast<double>(mallocSize) / (ms * 1000));    // 3. 测 musaFree    timer.Restart();    checkMusaErrors(musaFree(deviceMemory));    timer.Stop();    ms = timer.GetElapsedSeconds() * 1000.f;    this->utime2->addValue(ms);    this->uband2->addValue(static_cast<double>(mallocSize) / (ms * 1000));}
```

这段代码看着平淡，却把驱动测试的几个"潜规则"全藏在里面了：

1. **必须 warm up**。`musaMalloc(4)` 这一行的本质是触发驱动 lazy init、显存池预热——否则首次调用会把驱动 fault-in 的成本也算进 1B 的延迟里，产生离谱的"小 size 显存带宽极低"假象。
2. **CPU 计时器而非 event**。malloc/free 是同步 API，没有 stream 概念，event 计时不适用，必须使用 `CPerfCounter`（位于 `common/include/timer_his.h`）做 host 侧 wall-clock。
3. **size 扫到 4GB**。`1LL << 32`，这是有意为之的——大模型动辄百 MB 级 KV cache 分配，驱动里的页表批处理路径与小分配走的可能不是同一条。

## 三、调度子系统：驱动级"暗时间"如何被精确测量

> `schedule/` 是这个仓库最具技术含量的部分。它实际上回答了一个问题：**当我连续提交一万个空 kernel，驱动到底慢在哪里？**

### 3.1 用驱动 API 测最纯的 launch latency

`kernelLaunchLatencyMode.cpp` 测的是"串行化的单次 kernel launch 延迟"。它特意走 **driver API** （`muLaunchKernel` 而非 runtime API 的 `<<<>>>` ），并在每次 launch 之后立刻 `musaStreamSynchronize`——这样测出来的就是"提交 + 入队 + 执行 + host 等待"的完整闭环时间，对应业务里串行 kernel 链路的真实开销：

```
// 来源：schedule/kernelLaunchLatencyMode.cppcheckMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));            // 预加载 ELF/PTXcheckMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));for (int i = 0; i < 100; ++i) {            // 100 次 warm-up，消除模块首次加载与 JIT 抖动    checkMuErrors(muLaunchKernel(function, 1,1,1, 1,1,1, 0, streamIdx, nullptr, nullptr));}checkMusaErrors(musaDeviceSynchronize());CPerfCounter timer; timer.Restart();for (int i = 0; i < launchTime; ++i) {    // launchTime = 1<<18，约 26 万次    checkMuErrors(muLaunchKernel(function, 1,1,1, 1,1,1, 0, streamIdx, nullptr, nullptr));    checkMusaErrors(musaStreamSynchronize(streamIdx));   // 关键：每次 launch 后立即同步}timer.Stop();this->utime1->addValue(milliseconds * 1000.f / launchTime);   // per(us)this->utp->addValue(launchTime / (milliseconds / 1000));      // *TP(s^-1)
```

值得注意的是 `codeObjectFile`：根据 `TEST_ON_NVIDIA` / `TEST_ON_CUBRIDGE` / 默认（MUSA）三种宏，分别加载 `.ptx`、`.mcfb`、`.elf` 三种二进制（预编译产物在 `schedule/elf/`[9] 下，由 `gen.sh` 生成）。这样三平台测到的"launch 延迟"才是同一颗等价的空 kernel，避免了 JIT 编译差异污染测量结果。

### 3.2 与之相对的 `kernelLaunchThroughputMode`

> 如果说 latency mode 是逼问 "**一次** launch 多慢"，那么吞吐模式则是"**一万次连发** 能跑多快"——同样的 launch，但仅在最后做一次 `Synchronize`，得到的是驱动 queue + GPU 命令处理器的稳态吞吐率。

两者结合，可以分离"驱动单次开销"与"驱动批量摊销能力"，这对 LLM 训练中的小算子 launch 优化是直接的体检指标。

### 3.3 双 stream 同步开销的对照实验：`efficiencyOfSync.cu`

> 这是整个仓库我最欣赏的一个设计：它不是直接测 `cudaEventRecord` / `cudaStreamWaitEvent` 的延迟，而是用**对照法** ——同样一串 N 个 kernel，跑两遍，一遍单 stream，一遍双 stream 交替 + event 串行，相减得到"纯同步开销"：

```
// 来源：schedule/efficiencyOfSync.cu// 路线 A：N 个 delay kernel 全部塞到 streams[0]timer.Start();for (uint64_t i = 0; i < n; ++i) {    delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);}checkMusaErrors(musaStreamSynchronize(nullptr));timer.Stop();float result1 = timer.GetElapsedSeconds() * 1e6f;   // 单 stream 总耗时// 路线 B：双 stream 交替提交，依赖通过 event 串成一条链timer.Reset(); timer.Start();for (uint64_t i = 0; i < n; ++i) {    if (i != 0) {        // 让本次 launch 等待上一个 stream 上记录的 event        checkMusaErrors(musaStreamWaitEvent(streams[i % 2], events[(i + 1) % 2], 0));    }    delay<<<1, 1, 0, streams[i % 2]>>>(flag + i, tickNum);    checkMusaErrors(musaEventRecord(events[i % 2], streams[i % 2]));}checkMusaErrors(musaStreamSynchronize(streams[(n - 1) % 2]));timer.Stop();float result2 = timer.GetElapsedSeconds() * 1e6f;// 单次 event 同步开销 ≈ (result2 - result1) / (N - 1)this->utime3->addValue((result2 - result1) / float(synchronizedConut - 1));
```

精妙之处有三：

1. **delay kernel 的 tick 数** （`tickNum = 1750`）刻意选得很小但又不为 0，保证 kernel 之间没有 occupancy 冲突，也不会被 GPU 优化掉；
2. **路线 A 和 B 提交的工作量完全一致**，差异只在"是否需要跨 stream 同步"，因此 `result2 - result1` 可以直接归因到 event 同步开销；
3. **代码里写了 `if (result2 - result1)` 的过滤**，因为多 stream 偶尔可能因并行得快于单 stream（"have no idea"——作者自嘲），这种异常值被显式丢弃，体现了对噪声的工程化处理。

### 3.4 GPU 命令处理器的 "kernelGap" 体检

> `kernelGap.cu` 用了一种极其讨巧的写法测"两个相邻 kernel 之间的最小间隔"：通过让每个 delay kernel 自己烧掉 `12500000 / N` 个 GPU clock，控制单个 kernel 的实际执行时间，再用 event 量总耗时：

```
// 来源：schedule/kernelGap.cuconst int tickNum = 12500000 / launchTimes;delay<<<1, 1>>>(flag, tickNum);                    // warmupcheckMusaErrors(musaEventRecord(start));for (uint64_t i = 0; i < launchTimes; ++i) {    delay<<<1, 1>>>(flag, tickNum);                // 提交 N 次，每次烧 (12.5M / N) 个 clock}checkMusaErrors(musaEventRecord(stop));checkMusaErrors(musaEventSynchronize(stop));checkMusaErrors(musaEventElapsedTime(&t, start, stop));
```

理想情况下总耗时应该约等于 "N × 单 kernel 执行时间"；实际多出来的那一截就是**命令处理器在两个 kernel 之间的气泡**。通过扫描 `launchTimes ∈ {1, 5, 10, 50, …, 10000}`，可以画出 gap 随并发密度变化的曲线——这是评估硬件命令流水线、驱动批处理逻辑的核心指标。

## 四、自动评分系统：是如何把"跑数"变成"一个分数"的

### 4.1 三套件、加权基线分

整个评分由 `TestSuitConfig.json`[10] 驱动，定义了三个套件：`graphAndSchedule`、`memoryOps`、`mulStreams`，每个套件设定 `baseline: "cudaA100"`、`baseScore: 1000`，并列出参与计分的 case 与其权重：

```
// 来源：TestSuitConfig.json（节选）{"suitName": "memoryOps","baseline": "cudaA100","baseScore": 1000,"testCases": [    { "caseName": "Copy1DAlignedRate", "scoreWeight": 1 },    { "caseName": "GpuReadAndWriteRate", "scoreWeight": 1 },    { "caseName": "MallocRateFrom1Bto4GB", "scoreWeight": 1 },    { "caseName": "IpcOpenMemHandle", "scoreWeight": 1 }    // ...  ]}
```

`parseWeightConfig()` 会把每个 case 的 `baseScore` 按权重再次切分：

```
# 来源：scripts/calculateScoreOfSuit.pytotalWeight = sum(case['scoreWeight'] for case in testCases)for case in testCases:    case['baseScore'] = (case['scoreWeight'] / totalWeight) * baseScore
```

### 4.2 评分核心：log10 相对比率法

> 真正的"算分"在 `calculateRadio` 一行：

```
# 来源：scripts/calculateScoreOfSuit.pydef calculateRadio(baseLine, test):    return 1 + math.log10(test / baseLine)
```

为什么是 `1 + log10(test/base)` 而不是简单的 `test / base`？这是评分系统里最值得品味的一处设计：

- 当 `test == base`，得分系数为 **1** （恰好拿到基础分）；
- 当 `test = 10 × base`（快十倍），系数为 **2** （不是 10，避免某一极端项把总分炸穿）；
- 当 `test = 0.1 × base`（慢十倍），系数为 **0**；再慢则为负——把"严重劣化"折扣得很狠但不会无限大。

`log10` 是一个**对数压缩 + 对称伸缩** 的算子，等价于：

> 每比基线快/慢一个数量级，得分上下浮动一个单位。

这与硬件评测里常用的"几何平均"思路一脉相承，能让多 case 总分对单一离群点更鲁棒。

### 4.3 表格对齐与逐元素累加

> 具体的算分循环非常直白：找到表头里以 `*` 开头的"焦点列"，按 (case 的基础分 ÷ 焦点列数 ÷ 数据行数) 把基础分均摊到每个 cell，然后逐 cell 累加 `scoreRadio × scoreOfPerElement`：

```
# 来源：scripts/calculateScoreOfSuit.pyfocusColumn = find_focus_columns(basicResutTable[0])scoreOfPerElement = float(basicScoreOfTest) / (len(focusColumn) * (len(basicResutTable) - 1))for rowIndex in range(1, len(testResultTableCopy)):    for colIndex in focusColumn:        basicValue = float(basicResutTable[rowIndex][colIndex])        testValue  = float(testResultTable[rowIndex][colIndex])        scoreRadio = calculateRadio(basicValue, testValue)        socreOfExecutableTestCase += (scoreRadio * scoreOfPerElement)
```

这种"扁平累加"+ "对数比率"的组合，让最终的 score 与 baseline 拉开数量级时仍能保持可读性。同时脚本会把所有 `scoreRadio` 喂给 matplotlib 画一张直方图（`Histogram.svg`），让用户一眼看出有没有"长尾性能洞"。

## 五、跨平台兼容：一份代码，三种二进制

> 这是另一处工程巧思——所有 case 文件全部用 MUSA API (`musa*`)、MUSA 驱动 API (`mu*`) 写成，对外却同时支持 **MUSA、NVIDIA CUDA、MetaX MACA** 三套后端。

秘密在 `common/include/helper_musa.h`（及 drvapi 版本）+ `scripts/porting2cuda.sh` 的 `musify-text` 工具：

```
# 来源：scripts/porting2cuda.sh（节选）${PORTING_PATH}/musify-text -d m2c \    -m ${PORTING_PATH}/extra.json \       ${PORTING_PATH}/general.json \       ${PORTING_PATH}/include.json \    --inplace `find ${test_path}/memory/ ${test_path}/common/ ... \               -name '*.cu' -o -name '*.h' -o -name '*.cpp' ...`
```

`musify-text` 走的是文本映射方案（mapping JSON 决定 `musaMalloc` ↔ `cudaMalloc` 这种 1:1 替换），相当于一次源码级 sed。再配合 CMake 里的三套分支：

```
# 来源：CMakeLists.txt（节选）if(ENABLE_CUBRIDGE)    project(${PROJECT_NAME} LANGUAGES CUDA C CXX)    add_definitions(-DTEST_ON_CUBRIDGE)elseif(ENABLE_NVCC)    project(${PROJECT_NAME} LANGUAGES CUDA C CXX)    add_definitions(-DTEST_ON_NVIDIA)    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)   # A100、Turing、Volta 通吃else()    project(${PROJECT_NAME} LANGUAGES C CXX)    include_directories(/usr/local/musa/include)    link_directories(/usr/local/musa/lib)endif()
```

得益于此，cudaA100 的 baseline CSV 可以由真实 A100 跑出来，而 MUSA GPU 跑出的 CSV 直接用同一份 baseline 折算分数，**不需要等价的"NVIDIA 版 case 代码"——避免了"对照测试因移植细节差异而失真"的常见坑。**

另外，CMake 里还有一段非常有趣的**自动架构识别** 逻辑：通过 `lspci -nn | grep -E '1ed5|0202'` 抓取 PCI ID，再到 `MTGPU_ARCH_MAPPING` 表里查 `mp_10 / mp_21 / mp_22 / mp_31 / mp_43` 等 MUSA GPU 架构号，设到 `--cuda-gpu-arch=` 编译选项中。这意味着用户在不同代摩尔线程 GPU 上 `./install.sh -m` 就能自适应编译，不需要手动指定架构。

## 六、可观察性与正确性：藏在细节里的"较真"

### 6.1 警告与剔除：错位数据零容忍

> `calculateScoreOfSuit.py` 在做对齐前先严格校验行列数：

```
# 来源：scripts/calculateScoreOfSuit.pyif len(basicTable) == 0 or len(basicTable) != len(testTable) \   or any(len(basicRow) != len(testRow) for basicRow, testRow in zip(basicTable, testTable)):    print("[Warning] the test file %s has mismatched rows or columns" % testFile)    testFiles.remove(testFile)
```

任何与 baseline 行列不一致的 CSV 都会被剔除而不是"凑合算"。在性能评测里，这种"宁缺毋滥"的取舍非常重要——一次错位会让 log 比率失真上百倍，污染整套件的总分。

### 6.2 异常值的就地处理

回看 `efficiencyOfSync.cu`，作者用了一行 `if (result2 - result1)` 把"双 stream 反而比单 stream 快"的诡异样本直接丢弃；malloc case 用 `SamplesCount = 3` + 单次迭代，把首次 malloc 的"驱动唤醒成本"通过 warm-up 而非平均化的方式剔除。诸如此类的小处理在仓库里随处可见，体现了一线性能工程师对噪声源的精准把控。

### 6.3 直方图：把分布也画出来

最终评分脚本不仅吐 `totalScore.csv`，还会调用 `drawHistogram` 把所有 cell 级的 `scoreRadio` 画成直方图：

```
# 来源：scripts/calculateScoreOfSuit.pydef drawHistogram(histogramData):    plt.figure(figsize=(9, 6), dpi=100)    plt.hist(histogramData, 25, color='w', edgecolor='k', hatch=r'ooo', label='frequency')    plt.savefig("./Histogram.svg")
```

一个分数告诉你"总体如何"，一个分布告诉你"是均衡好还是个别项偏科"——这两个视角对驱动迭代评审而言缺一不可。

比如总分 980（接近 A100 的 1000），但若直方图重心明显左移、长尾里有几个 `radio < 0` 的项，那就意味着大多数 case 强，但有少数 case 严重劣化——这恰恰是驱动 regression 排查最需要的信息密度。

## 七、它告诉了我们什么

> 把整个仓库的代码读完，会发现 `gpu-compute-driver-bench` 真正想做的事情可以浓缩成三句话：

1. **把"驱动体验"工程化、可测量、可对比。** 它不去比 TFLOPS，而是用一组上百个微基准把驱动 API 路径切成最小颗粒，每一刀都对应一类真实工作负载里的"暗时间"。
2. **以对照实验代替绝对测量。** 无论是 single-stream vs dual-stream、warm-up vs cold-start、driver API vs runtime API，几乎每个 case 都是一对（或多对）对照组的差值，最大限度消除环境噪声。
3. **用对数比率折算分数，对标 cudaA100。** 这是一个面向**国产 GPU vs NVIDIA** 横向比较的、刻意"克制"的评分体系——它不会让某个突出指标把分数刷爆，也不会让某个短板把分数砸穿。

> 从工程价值角度，它对三类人特别有用：

- **驱动/Runtime 开发者**：每个 PR 跑一遍，看 `Histogram.svg` 的形状有没有变差，可以替代靠人肉跑模型才能发现 regression 的工作流；
- **上层框架/编译器作者**：调度策略（要不要用 CUDA Graph？要不要 fuse 小 kernel？要不要复用 stream？）需要的真实开销数据，全在这里以 CSV 形态准备好；
- **采购与选型方**：在没有官方 spec 的情况下，用同一套 baseline 把不同代际、不同厂商的卡放在同一把尺子上做相对量化，避免被某个 marketing benchmark 误导。

::: block-1
最后还是要说一句题外话：在国产 GPU 仍处在追赶 NVIDIA 软件生态的今天，把 baseline 设成 `cudaA100`、把代码做成"一次源码、三处编译"，本身就是一种相当自信、相当开放的姿态——它把"我比 A100 差在哪里、好在哪里"亮在阳光下，让外部社区可以一起监督、一起优化。**这种"敢于被打分"，比任何 PPT 上的峰值算力都更接近工程师能信任的产品力。**
:::

如果你正在做 GPU 驱动、CUDA 兼容层、AI 编译器或 LLM 推理优化，强烈建议把这个仓库 clone 下来——哪怕只是阅读 `schedule/efficiencyOfSync.cu` 和 `scripts/calculateScoreOfSuit.py` 这两份文件，也足以让你对"如何严肃地测一颗 GPU 驱动"有全新的理解。

参考资料

[1] `gpu-compute-driver-bench`: *https://github.com/MooreThreads/gpu-compute-driver-bench*

[2] `scripts/porting2cuda.sh`: *https://github.com/MooreThreads/gpu-compute-driver-bench/blob/main/scripts/porting2cuda.sh*

[3] `scripts/porting2musa.sh`: *https://github.com/MooreThreads/gpu-compute-driver-bench/blob/main/scripts/porting2musa.sh*

[4] `CMakeLists.txt`: *https://github.com/MooreThreads/gpu-compute-driver-bench/blob/main/CMakeLists.txt*

[5] `TestSuitConfig.json`: *https://github.com/MooreThreads/gpu-compute-driver-bench/blob/main/TestSuitConfig.json*

[6] `scripts/calculateScoreOfSuit.py`: *https://github.com/MooreThreads/gpu-compute-driver-bench/blob/main/scripts/calculateScoreOfSuit.py*

[7] Celero: *https://github.com/DigitalInBlue/Celero*

[8] 第四章: *#四自动评分系统是如何把跑数变成一个分数的*

[9] `schedule/elf/`: *https://github.com/MooreThreads/gpu-compute-driver-bench/tree/main/schedule/elf*

[10] `TestSuitConfig.json`: *https://github.com/MooreThreads/gpu-compute-driver-bench/blob/main/TestSuitConfig.json*
