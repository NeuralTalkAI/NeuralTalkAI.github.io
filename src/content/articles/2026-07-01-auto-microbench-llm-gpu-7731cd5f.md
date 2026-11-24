**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**GPU 微基准**、LLM 智能体、**SASS 验证**、:mdnice-bold-italic[Nsight Compute]、**模拟器校准**

> GPU 模拟器最怕什么？不是没有模型，而是模型里的硬件参数“看似合理，却悄悄错了”。一个 FMA 吞吐、一次 L2 命中延迟、一条 Tensor Core 指令的真实执行路径，只要偏一点，仿真结果就会在复杂应用中层层放大。

![文章插图](/articles/2026-07-01-auto-microbench-llm-gpu-7731cd5f/images/01.webp)

- Code Implementation for OSCAR 2026 Workshop: **Toward Sustainable GPU Modeling: Autonomous Micro-benchmarking with LLM Agents**
- https://github.com/abdelfattah-lab/Auto-Microbench

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::

> 传统做法依赖专家手写微基准、人工读 SASS、查白皮书、调 Nsight Compute 计数器，既慢又难复现。

Auto-MicroBench 的野心正是在这里：它绝非简单让大模型生成 CUDA，反而:mdnice-bold-italic[给 LLM 套上一套“实验室制度”——调度器决定测什么，知识库告诉怎么测，固定 harness 负责编译、验证、统计与归档，最终把一次次试错沉淀成 Accel-Sim 与 GPGPU-Sim 可用的硬件配置碎片。]

## 本文目录

- 一、快速上手：用最短路径跑起自动微基准
  - 1.1 一次性安装与环境准备
  - 1.2 最小工作流
- 二、项目到底解决了什么问题
  - 2.1 GPU 模拟器需要的不是“峰值”，而是可复现的微观参数
  - 2.2 这不是普通 benchmark，而是带审计的实验系统
- 三、整体架构：调度器、样品、仪器与报告器
  - 3.1 顶层文件分工
  - 3.2 参数目录是“实验任务书”
  - 3.3 调度器是“实验排班系统”
- 四、关键闭环：LLM 只负责改样品，harness 负责当裁判
  - 4.1 `benchmark.cu`的契约很窄
  - 4.2 harness 做了哪些“裁判动作”
- 五、为什么 SASS 验证是项目的灵魂
  - 5.1 PTX 不是最终真相
  - 5.2 SASS 缺失时，harness 还会诊断“为什么缺失”
- 六、测量模板：把微基准经验固化成可复制骨架
  - 6.1 吞吐测量：独立寄存器链让 pipeline 吃饱
  - 6.2 延迟测量：反过来，必须制造串行依赖
  - 6.3 Cache 测量：pointer chasing 对抗预取和并行隐藏
- 七、统计与状态机：为什么一个数字不够
  - 7.1 中位数、CV 与 95%置信区间
  - 7.2 合理范围不是“真值”，而是防呆栏杆
- 八、知识库与模板：给 LLM 装上“实验记忆”
  - 8.1 `program.md`像智能体操作系统
  - 8.2 方法论知识库把经验拆成专题
- 九、结果如何变成模拟器配置
  - 9.1 TSV 是事实表，report 是解释层
  - 9.2 Accel-Sim/GPGPU-Sim 映射是谨慎的“配置片段”
- 十、这个项目最值得学习的设计思想
  - 10.1 把 LLM 关进窄接口，而不是放任它改全局
  - 10.2 用多源验证对抗“看起来对”的幻觉
  - 10.3 把失败沉淀为状态，而不是丢给上下文窗口
- 十一、局限与未来空间
  - 11.1 当前仍是“初始脚手架”，多架构还有重构空间
  - 11.2 Tensor Core 新指令族是最大挑战
  - 11.3 NCU 指标名不是稳定契约
- 结语：Auto-MicroBench 的真正价值是“可信自动化”

![文章插图](/articles/2026-07-01-auto-microbench-llm-gpu-7731cd5f/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：用最短路径跑起自动微基准

### 1.1 一次性安装与环境准备

> 项目核心 Python 依赖几乎为空，`pyproject.toml`中只把`matplotlib`、`numpy`、`pytest`作为可选依赖；真正的外部条件是 CUDA 工具链、`nvcc`、`cuobjdump`，以及可选的 Nsight Compute `ncu`。README 给出的最小安装路径如下：

```
git clone https://github.com/abdelfattah-lab/Auto-Microbenchcd Auto-Microbenchpython -m pip install -e .[viz]bash knowledge/download-ptx-isa.sh# 推荐：锁定GPU时钟，降低DVFS带来的统计噪声sudo nvidia-smi -lgc <base_clock_mhz>
```

随后启动你习惯的 coding agent，例如 Claude Code、Codex 或 Cursor Agent，并让它阅读`program.md`。一个典型启动命令是：

```
python orchestrate.py --arch sm_80 --max-priority 3
```

其中`--arch`需要按 GPU 架构选择：Ampere A100 用`sm_80`，Ada RTX 4090 用`sm_89`，Hopper 与 Blackwell 应使用带`a`后缀的架构值，如`sm_90a`、`sm_100a`、`sm_103a`、`sm_120a`。具体取值和原因可参考项目 README 中的 GPU architecture target 章节：https://github.com/abdelfattah-lab/Auto-Microbench#gpu-architecture-target-the---arch-flag[1]

### 1.2 最小工作流

> 如果只想理解它怎么跑，可以把 Auto-MicroBench 看成四条命令构成的闭环：

```
# 1. 选择下一个待测硬件参数python orchestrate.py --arch sm_80 --max-priority 3# 2. agent读取current_parameter.json，改写benchmark.cu# 3. 用固定harness编译、验证、运行、统计python harness.py --param <name> --unit <unit> --arch sm_80 \  --check-sass <expected_sass> \  --reasonable-min <min> --reasonable-max <max> \  --ncu --ncu-metrics <metrics># 4. 生成最终报告和模拟器配置python report.py --format all --output report/
```

运行过程中可以用下面命令观察进度：

```
watch -n 5 'python orchestrate.py --progress; echo; tail -10 results/measurements.tsv 2>/dev/null'
```

这套项目最值得注意的使用约束是：**agent 原则上只改`benchmark.cu`，不改调度器、不改 harness、不改知识库。** 这让 LLM 的创造性被限制在“实验样品”层，而实验仪器、验收规则和记录系统保持稳定。

## 二、项目到底解决了什么问题

### 2.1 GPU 模拟器需要的不是“峰值”，而是可复现的微观参数

> Accel-Sim、GPGPU-Sim 这类学术 GPU 模拟器需要大量底层参数：指令吞吐、指令延迟、L1/L2 容量与命中延迟、DRAM 带宽、共享内存 bank 行为、warp 调度特征、Tensor Core 吞吐等。**厂商白皮书通常给的是片段化指标，且不同 SKU、不同架构、不同指令族差异巨大。**

例如同样是 FP64，数据中心卡和消费级 Blackwell 可能完全不是一个吞吐级别；同样是 Tensor Core，Ampere 的`mma.sync.aligned`、Hopper 的`wgmma`、Blackwell 数据中心的`tcgen05.mma`背后是不同指令家族。

::: block-1
Auto-MicroBench 的核心**价值是把 GPU 测量拆成一个可重复的闭环**：

- 参数目录定义“要测什么”；
- 知识库定义“应该怎么测”；
- 模板定义“测量骨架”；
- harness 定义“验收标准”；
- report 把可信结果导出到模拟器配置。
:::

### 2.2 这不是普通 benchmark，而是带审计的实验系统

> 普通 benchmark 追求跑得快、数值好看；:mdnice-bold-italic[Auto-MicroBench 追求的是“这个数能不能被相信”。README 直接写明：win condition 是 measurement accuracy，而不是 speed。它对一次测量至少设置了几道门：]

| 验证类型 | 目的与方法 |
| --- | --- |
| **SASS 验证** | 确认 `ptxas` 编译器最终生成的目标机器指令（SASS）与预期一致，确保汇编层面无优化偏差或指令替换。 |
| **NCU 验证** | 利用 NVIDIA Nsight Compute（NCU）采集硬件计数器数据，验证实际执行的指令流水线（execution pipe）分布与理论分析或性能模型预测一致。 |
| **统计验证** | 执行多轮 benchmark trial（建议 ≥30 轮），计算结果的变异系数（CV）及 95% 置信区间，确保性能数据具备统计稳健性。 |
| **合理范围验证** | 调用 `reasonable_range` 辅助函数，基于经验阈值或硬件规格快速过滤明显异常值（如远超理论峰值或低于噪声基线的结果）。 |
| **known value 对比** | 当存在厂商提供参考值（如 cuBLAS/cuFFT 官方吞吐量、白皮书数据）时，计算实测值相对偏差（如 |实测−参考|/参考），设定容忍阈值（如 ±5%）进行自动判定。 |
| **归档与审计** | 验证通过（PASS）后，将生成该结果的完整源文件 `benchmark.cu` 自动归档至 `solved_kernels/` 目录，保留可复现性与审计线索。 |

这意味着 LLM 不再只是一个“写代码的人”，而是被放进一套实验规范里，像研究生一样写实验、跑实验、接受仪器检查、记录失败原因。

## 三、整体架构：调度器、样品、仪器与报告器

### 3.1 顶层文件分工

> 项目顶层很克制，核心文件包括：

- `parameter_catalog.py`：硬件参数目录；
- `orchestrate.py`：决定下一步测什么；
- `benchmark.cu`：agent 每轮改写的唯一 CUDA 工作区；
- `harness.py`：固定评测仪器，负责编译、SASS 检查、运行 trial、NCU 验证、统计和日志；
- `templates/`：不同测量技术的 CUDA 模板；
- `knowledge/`：方法论 playbook 与 PTX/CUDA/Nsight 参考；
- `validators/`：SASS、NCU、cache、occupancy 等辅助验证器；
- `report.py`与`configs/`：把结果导出为 Markdown、JSON、Accel-Sim/GPGPU-Sim 配置。

::: block-1
项目真正的运行形态可以用一句话概括：**orchestrator 发任务，agent 改`benchmark.cu`，harness 验收，results 记录，report 导出。**
:::

### 3.2 参数目录是“实验任务书”

> `parameter_catalog.py`中的`Parameter`是一份完整实验规格：**类别、单位、测量技术、优先级、已知值、SASS 期望、PTX 期望、NCU 指标、合理范围、架构特定 override** 都在这里。

```
# 来源：parameter_catalog.py@dataclassclass Parameter:    name: str    category: Category    description: str    unit: str    measurement_technique: Technique    priority: int = 3    known_value: Optional[float] = None    known_value_source: Optional[str] = None    ncu_validation_metrics: list[str] = field(default_factory=list)    dependencies: list[str] = field(default_factory=list)    notes: str = ""    shared_note_tags: list[str] = field(default_factory=list)    expected_sass: list[str] = field(default_factory=list)    forbidden_sass: list[str] = field(default_factory=list)    expected_sass_regex: bool = False    expected_ptx: list[str] = field(default_factory=list)    reasonable_range: Optional[tuple[float, float]] = None    arch_overrides: dict[str, dict] = field(default_factory=dict)    ncu_metric_alternatives: dict[str, list[str]] = field(default_factory=dict)
```

这个设计非常关键：它让“测量任务”机器可读。agent 不需要凭上下文猜测目标，而是读取`current_parameter.json`，知道这次要测`fma_f32_throughput`还是`l2_cache_latency`，应使用吞吐循环还是 pointer chase，应检查`FFMA`还是`LDG`。

### 3.3 调度器是“实验排班系统”

> `orchestrate.py`负责从目录中挑选下一个未测参数。它不仅看是否已经出现在`results/measurements.tsv`，还**支持优先级过滤、only 过滤、deferred backlog，以及依赖顺序。**

```
# 来源：orchestrate.pydef select_next_parameter(results_file=RESULTS_FILE,                          max_priority=None,                          only=None,                          include_deferred=False) -> Parameter | None:    unmeasured = get_unmeasured_parameters(results_file)    ifnot unmeasured:        returnNone    ifnot include_deferred:        deferred = get_deferred_names()        unmeasured = [p for p in unmeasured if p.name notin deferred]    if only isnotNone:        wanted = set(only)        unmeasured = [p for p in unmeasured if p.name in wanted]    if max_priority isnotNone:        unmeasured = [p for p in unmeasured if p.priority <= max_priority]    measured = get_measured_names(results_file)    ready = [p for p in unmeasured if all(dep in measured for dep in p.dependencies)]    ifnot ready:        ready = unmeasured    ready.sort(key=lambda p: (p.priority, p.name))    return ready[0]
```

这个选择逻辑看似朴素，却很适合自动化 agent：低数字优先级代表核心参数，依赖已满足者优先，失败太久的参数可以 defer，避免整个测量任务被一个难点阻塞。

## 四、关键闭环：LLM 只负责改样品，harness 负责当裁判

### 4.1 `benchmark.cu`的契约很窄

> `benchmark.cu`开头明确写着契约：输出一行`RESULT: <measured_value>`，可选输出`META: key=value`，成功退出 0，单次运行小于 30 秒。这个契约非常像 Unix 工具哲学：样品可以千变万化，仪器只认标准输出。

```
// 来源：benchmark.cu/* * CONTRACT (do not break): *   - Print exactly one line:   `RESULT: <measured_value>` *   - Optionally print:         `META: <key>=<value>` lines *   - Exit 0 on success, non-zero on failure *   - Total runtime per invocation < 30 seconds * * METHODOLOGY (cycle-level) *   - Time the kernel with `mov.u64 %0, %%clock64;` via inline PTX, *     NOT the `clock64()` builtin and NOT `cudaEvent`. */
```

这正是项目约束 LLM 的方式：LLM 可以重写内核，但不能随意改变评测系统；它必须把复杂实验包装成一行可解析结果。

### 4.2 harness 做了哪些“裁判动作”

> `harness.py`是**全项目最重的固定基础设施**。它完成六件事：

1. 用`nvcc -O3`按目标架构编译；
2. 用`cuobjdump --dump-sass`检查目标 SASS；
3. 多次 warmup 与 trial 运行；
4. 可选调用`ncu`读取硬件计数器；
5. 统计均值、中位数、标准差、CV、95% CI；
6. 写入 TSV，并在 PASS 时归档源码。

它的编译逻辑甚至专门处理 Hopper/Blackwell 的`a`后缀问题：如果`arch`以`a`结尾，使用`-gencode=arch=compute_120a,code=sm_120a`形式，而不是简单`-arch=sm_120a`。这是底层 CUDA 工具链经验的体现，因为某些 PTX target 必须显式带`a`才能启用架构特定指令。

```
# 来源：harness.pydef compile_benchmark(source="benchmark.cu",                      output="benchmark",                      arch="sm_80",                      extra_flags=None,                      timeout=60) -> tuple[bool, str]:    _warn_if_bare_arch(arch)    flags = list(extra_flags or [])    if arch.endswith("a"):        compute = arch.replace("sm_", "compute_")        arch_flag = f"-gencode=arch={compute},code={arch}"    else:        arch_flag = f"-arch={arch}"    cmd = ["nvcc", "-O3", arch_flag, "-lineinfo", "-o", output, source] + flags    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)    if result.returncode != 0:        returnFalse, result.stderr or result.stdout    returnTrue, ""
```

这里隐藏着一个重要判断：Auto-MicroBench 承认“目标架构字符串本身就是实验条件的一部分”。对于 Tensor Core、TMA、WGMMA、TCGEN05 这类新指令，少一个`a`后缀，测到的就可能是完全不同的路径。

## 五、为什么 SASS 验证是项目的灵魂

### 5.1 PTX 不是最终真相

> 很多 GPU 程序员知道 PTX 是中间表示，但在写微基准时仍容易犯错：以为写了`add.s32`就测到了 ADD，以为写了`mma.sync.aligned`就一定测到了 Tensor Core。Auto-MicroBench 不接受这种假设。**它要求用 SASS 确认 ptxas 最终发出了什么。**

`validators/sass_checker.py`就是一个很薄但很关键的模块：调用`cuobjdump --dump-sass`，统计必须出现和禁止出现的 pattern。

```
# 来源：validators/sass_checker.pydef check_patterns(sass: str,                   patterns: list[str],                   regex: bool = False) -> dict[str, int]:    """Return {pattern: count} for each pattern in the SASS dump."""    counts: dict[str, int] = {}    for pat in patterns:        if regex:            counts[pat] = len(re.findall(pat, sass))        else:            counts[pat] = sass.count(pat)    return counts
```

这段代码不复杂，但它让整个框架从“相信 LLM 写的 PTX”升级为“相信机器码证据”。对于微基准而言，这是生死线。

### 5.2 SASS 缺失时，harness 还会诊断“为什么缺失”

> `harness.py`不只是报`SASS_MISMATCH`。当 SASS count 为 0 时，它会 dump PTX，判断是 PTX 里有目标 mnemonic 但 SASS 没有，还是 PTX 本身就没生成出来。

- 前者称为`PTXAS_SILENT_DROP`，常见原因是修饰符、寄存器压力或约束类型；
- 后者是`BAD_INLINE_ASM`，说明 inline asm 没有按预期降低到 PTX。

**这对 LLM 非常重要：失败信息越结构化，下一轮修改越可能有效**。否则 agent 会陷入“改一点、跑一次、再猜”的随机游走。

## 六、测量模板：把微基准经验固化成可复制骨架

### 6.1 吞吐测量：独立寄存器链让 pipeline 吃饱

> 指令吞吐模板`templates/throughput_template.cu`的核心思想是：一个 block 落在一个 SM 上，每个线程维护多条独立寄存器链，内层循环充分 unroll，让目标指令密集出现。最终公式是：

```
inst_per_cycle = threads × lanes × iters / measured_cycles
```

```
// 来源：templates/throughput_template.cu#pragma unroll 1for (int it = 0; it < iters; it += UNROLL_ITERS) {    #pragma unroll    for (int u = 0; u < UNROLL_ITERS; ++u) {        #pragma unroll        for (int i = 0; i < LANES; ++i) {            asm volatile("fma.rn.f32 %0, %1, %2, %0;"                         : "+f"(r[i]) : "f"(a), "f"(b));        }    }}
```

这里的`LANES`不是 CUDA lane，而是每个线程里的独立寄存器流。比喻一下：

- 测延迟时是一辆车走单行道，前车不走后车不能动；
- 测吞吐时要开多条独立车道，让收费站满负荷工作。

只有这样测到的才是 pipeline issue/execute 能力，而不是依赖链延迟。

### 6.2 延迟测量：反过来，必须制造串行依赖

> `templates/latency_dep_chain.cu`采用单 warp、单线程依赖链。每一条指令的输出都喂给下一条，报告`cycles / iters`。这与吞吐模板的目标正好相反：吞吐要消除依赖，延迟要制造依赖。

```
// 来源：templates/latency_dep_chain.cu__threadfence_block();const uint64_t t0 = rdclk64();#pragma unroll 1for (int it = 0; it < iters; it += UNROLL_ITERS) {    #pragma unroll    for (int u = 0; u < UNROLL_ITERS; ++u) {        asm volatile("fma.rn.f32 %0, %1, %2, %0;"                     : "+f"(r) : "f"(a), "f"(b));    }}const uint64_t t1 = rdclk64();
```

这就是微基准设计中最容易被 LLM 混淆的一点：同样是测 FMA，吞吐和延迟的 kernel 形态完全不同。

- 如果用吞吐模板测延迟，会被 ILP 隐藏；
- 如果用延迟模板测吞吐，会低估 pipeline 能力。

### 6.3 Cache 测量：pointer chasing 对抗预取和并行隐藏

> Cache 层级测量使用随机环形链表。单线程追逐指针，每次 load 依赖前一次 load 结果，这能压制 GPU 通过 MSHR、并行线程或 coalescing 隐藏延迟的能力。

```
// 来源：templates/cache_pointer_chase.custatic void build_random_cycle(std::vector<int> &arr) {    const int n = static_cast<int>(arr.size());    std::vector<int> order(n);    for (int i = 0; i < n; ++i) order[i] = i;    std::mt19937 rng(0xC0FFEE);    std::shuffle(order.begin() + 1, order.end(), rng);    for (int i = 0; i < n - 1; ++i) {        arr[order[i]] = order[i + 1];    }    arr[order[n - 1]] = order[0];}
```

这段随机化很关键。如果是顺序访问，硬件预取器可能提前把下一条 cache line 拉进来，最后测出来的是“预取成功后的路径”，而不是目标 cache 层级真实延迟。**Auto-MicroBench 的知识库也明确提醒：cache size 要通过 working set sweep 观察阶梯，而不是单点测量。**

## 七、统计与状态机：为什么一个数字不够

### 7.1 中位数、CV 与 95%置信区间

> `harness.py`在`analyze_results()`中使用 MAD 过滤离群值，然后计算 mean、median、std、CV 和近似 95% CI。最终上报值用 median，这比 mean 更抗 GPU 测量中的偶发抖动。

```
# 来源：harness.pyfiltered = _filter_outliers_mad(values)mean = statistics.mean(filtered)median = statistics.median(filtered)std = statistics.stdev(filtered) if len(filtered) > 1else0.0cv = std / mean if mean != 0else float("inf")status = "PASS"note_parts: list[str] = []if cv > 0.05:    status = "HIGH_VARIANCE"    note_parts.append(f"CV={cv:.3f} exceeds 5% threshold.")elif cv > 0.02:    note_parts.append(f"CV={cv:.3f} above 2% target (still acceptable).")
```

这里体现了项目的科学态度：一次运行的`RESULT`只是样本，多次 trial 之后的分布才是证据。**CV 超过 5%直接标为`HIGH_VARIANCE`，意味着该 kernel 可能受 DVFS、工作量太短、调度干扰或计时窗口错误影响。**

### 7.2 合理范围不是“真值”，而是防呆栏杆

> 每个 Parameter 可带`reasonable_range`。这是跨现代 NVIDIA GPU 都应大致合理的宽范围。比如 FMA 延迟不该测成 0.4 cycles，DRAM 带宽不该测成几十 GB/s。超出范围会标为`OUT_OF_RANGE`，提示 agent 重审方法。

这种设计比硬编码单一 known value 更稳健。因为真实 GPU SKU 差异很大，尤其 FP64、Tensor Core 新格式、Blackwell 消费卡与数据中心卡之间不能简单套同一个数。合理范围负责抓荒谬结果，known value 负责在有参考时辅助判断。

## 八、知识库与模板：给 LLM 装上“实验记忆”

### 8.1 `program.md`像智能体操作系统

> `program.md`不是普通说明文档，而是 agent 的操作规程。它告诉 agent 每轮先跑`orchestrate.py`，读`current_parameter.json`，读匹配的`knowledge/common_knowledges/*.md`，必要时 grep PTX ISA 文档，然后复制模板改`benchmark.cu`。

失败时按固定诊断清单走：先看 PTX 是否存在，再查 inline asm 语法，再查 modifier gotcha，再查寄存器压力，再判断架构支持，最后做最小复现。

这实际上是在弥补 LLM 的两个弱点：

- 第一，LLM 容易“凭印象写 PTX”；
- 第二，LLM 失败后容易随机试错。

`program.md`把专家经验变成了流程约束。

### 8.2 方法论知识库把经验拆成专题

- `knowledge/common_knowledges/instruction_throughput.md`专门讲吞吐测量，并强调整数 ADD/MUL 会被 ptxas 因结合律折叠，需要 Fibonacci 式双寄存器递推。
- `cache_hierarchy.md`讲 pointer chase、随机 cycle、working set sweep、NCU cache hit 指标。还有 shared memory、memory bandwidth、register file、warp scheduling、common pitfalls 等。

这让 Auto-MicroBench 更像“自动实验室”而不是“代码生成器”。LLM 负责组合和执行，但实验方法的底线来自人类写好的 playbook。

## 九、结果如何变成模拟器配置

### 9.1 TSV 是事实表，report 是解释层

> 每次 harness 运行都会写入`results/measurements.tsv`，字段包括 parameter、value、unit、mean、median、std_dev、CV、CI、trials、status、ncu_validated、sass_validated、range verdict、notes、timestamp。`report.py`读取 TSV 后生成 Markdown 报告、JSON 和模拟器配置片段。

```
# 来源：report.pydef generate_json_export(results: list[dict], output_dir: str) -> str:    payload = {        "generated": datetime.now().isoformat(timespec="seconds"),        "parameters": {            r["parameter"]: {                "value": _to_float(r.get("value")),                "unit": r.get("unit", ""),                "mean": _to_float(r.get("mean")),                "median": _to_float(r.get("median")),                "std_dev": _to_float(r.get("std_dev")),                "cv": _to_float(r.get("CV")),                "status": r.get("status", ""),                "ncu_validated": r.get("ncu_validated", "False") == "True",                "sass_validated": r.get("sass_validated", "False") == "True",            }            for r in results        },    }
```

JSON 保留了结构化数据，Markdown 面向人审阅，而配置导出器面向模拟器集成。三者共同保证结果既能读、又能被程序消费。

### 9.2 Accel-Sim/GPGPU-Sim 映射是谨慎的“配置片段”

> `configs/accelsim_config.py`把部分参数映射到 Accel-Sim 选项，例如 warp size、shared memory size、L1/L2 latency、global memory bandwidth 等。它同时把无法明确映射的参数作为注释输出，提醒用户审阅。

```
# 来源：configs/accelsim_config.pyACCELSIM_MAPPING: dict[str, tuple[str, str]] = {    "warp_size":                ("-gpgpu_shader_core_pipeline_warp_size", "raw"),    "max_warps_per_sm":         ("-gpgpu_max_warps_per_shader", "raw"),    "registers_per_sm":         ("-gpgpu_shader_registers", "raw"),    "shared_memory_size":       ("-gpgpu_shmem_size", "raw_kb"),    "l1_cache_latency":         ("-gpgpu_l1_latency", "raw"),    "l2_cache_latency":         ("-gpgpu_l2_latency", "raw"),    "global_memory_latency":    ("-gpgpu_dram_latency", "raw"),    "global_memory_bandwidth":  ("-gpgpu_dram_bandwidth", "raw"),    "fma_f32_latency":          ("-ptx_opcode_latency_int", "raw"),    "fma_f64_latency":          ("-ptx_opcode_latency_dp", "raw"),}
```

这份映射故意保守。因为模拟器配置项往往不是一一对应的物理参数，例如 cache config 还涉及 associativity、sets、line size 等几何结构。Auto-MicroBench 输出的是“可合并的片段”和“审阅线索”，而不是宣称一键生成完美模拟器。

## 十、这个项目最值得学习的设计思想

### 10.1 把 LLM 关进窄接口，而不是放任它改全局

> **Auto-MicroBench 最漂亮的工程边界是：LLM 只改`benchmark.cu`。调度、验证、统计、报告、知识库都固定**。这让 LLM 的错误被隔离在单个实验样品中，harness 会把错误分类为`COMPILE_ERROR`、`SASS_MISMATCH`、`HIGH_VARIANCE`、`OUT_OF_RANGE`等，而不是污染整个系统。

这也是 AI 工程落地的重要启发：不要让 LLM 直接掌控所有层；给它一个可创造的窄面，再用确定性系统包围它。

### 10.2 用多源验证对抗“看起来对”的幻觉

GPU 微基准最危险的不是编译失败，而是编译成功、数字稳定、但测错对象。比如 PTX 写的是 INT4 MMA，SASS 却是 IMAD/SHF/IADD3 软件模拟；或者 FMA 指令数被 ptxas 折叠，结果快得离谱但 SASS pattern 仍有少量命中。Auto-MicroBench 用 SASS、NCU、统计、合理范围共同围堵这种“可信幻觉”。

这套思想同样适合其他 LLM 代码生成场景：只要任务存在不可见语义，就必须引入独立验证信号，而不能只看输出是否存在。

### 10.3 把失败沉淀为状态，而不是丢给上下文窗口

`orchestrate.py`支持 defer、annotate、retry 和 backlog，`harness.py`维护 history sidecar，PASS 后归档 solved kernel。这些机制的意义在于：实验失败不只是一段聊天记录，而是持久化的工程状态。下一个 agent、下一次 session、甚至另一个人接手时，都能看到之前为什么失败、尝试过什么、下一步该试什么。

::: block-1
对于长周期 AI 智能体任务，这是非常重要的模式：**把上下文从 prompt 中移到文件系统和结构化日志中。**
:::

## 十一、局限与未来空间

### 11.1 当前仍是“初始脚手架”，多架构还有重构空间

README 的 Status 部分明确说这是 initial scaffold，正在 A100、RTX 4090、RTX 5090、H100 等真实 GPU 上 battle-test，多架构 refactor 仍是后续工作。

`parameter_catalog.py`虽然已有`arch_overrides`机制，但 README 也提到早期思路是复制不同架构 catalog。**长期看，更理想的方式是让参数目录成为分层配置：公共测量规格、架构族 override、SKU override 分离。**

### 11.2 Tensor Core 新指令族是最大挑战

Blackwell 的 FP4/FP6、block scale、`tcgen05.mma`、CTA group cooperative mode 等，本身就是高度复杂的底层接口。项目已经通过`SHARED_NOTES`、模板和知识库做了大量防护，但这部分仍然依赖真实硬件反复验证。尤其是“PTX 兼容但硬件路径已被 drop”的情况，必须通过 SASS 和吞吐量共同识别。

### 11.3 NCU 指标名不是稳定契约

项目已经意识到 NCU metric names 会随架构和版本漂移，所以把`ncu_validation_metrics`定义为 hint，并提供`ncu_metric_alternatives`。这很务实：硬件计数器是强证据，但工具命名不是强契约。未来如果能自动从本机`ncu --query-metrics`中匹配候选指标，agent 的闭环会更稳。

## 结语：Auto-MicroBench 的真正价值是“可信自动化”

> :mdnice-bold-italic[Auto-MicroBench 表面上是一个 GPU 微基准框架，深层看则是一种 AI 系统架构范式：让 LLM 承担创造性但高风险的部分，让确定性程序承担验证、记录、调度和导出]。它把“写一个 CUDA benchmark”升级为“完成一次可审计的硬件实验”。

对 GPU 模拟器研究者来说，它的直接收益是更快得到一批可用的硬件参数，最终导出 Accel-Sim/GPGPU-Sim 配置片段。对 AI 工程师来说，它更大的启发是：:mdnice-bold-italic[LLM 智能体并不一定要靠更强模型才能可靠，很多时候，可靠性来自边界设计、任务分解、验证闭环和持久化状态。]

::: block-1
一句话总结：Auto-MicroBench 不是让大模型替你“猜 GPU”，而是让大模型在一套严格实验制度下，逐步把 GPU 黑盒拆成可测、可验、可复现的参数表。
:::

参考资料

[1]

https://github.com/abdelfattah-lab/Auto-Microbench#gpu-architecture-target-the---arch-flag: *https://github.com/abdelfattah-lab/Auto-Microbench#gpu-architecture-target-the---arch-flag*

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::
