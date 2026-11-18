关键词：GPU 算子优化、**Nsight Compute 性能剖析**、AI 编码智能体、:mdnice-bold-italic[多后端自动调优]、**策略记忆机制**

> 今天在群里看到一个关于 **NPU 算子悖论** 的讨论：*如果一个算子哥能在指令集难用、文档匮乏的国产 NPU 上写出媲美 NV 卡的性能，他理应比同等水平的 CUDA 算子哥强一个维度。但现实是，国产卡算子哥的工资往往不如互联网大厂。按此推导，他若去字节、Qwen、DeepSeek 写算子，工资可以翻倍；他若去 NV 或平头哥 PPU，脑力负担又能减半。*

这个“悖论”的背后，其实指向了一个行业沉疴：**GPU/NPU 算子调优，长久以来被视为一门无法量产、极其耗神的“手艺活”。**

工程师被绑在枯燥的循环里——对着 Nsight Compute 几十项晦涩的硬件计数器反复揣摩，手动推演瓶颈，小心翼翼地拨动 tile shape 或调整访存模式，然后跑一轮 benchmark，再看一轮报告，周而复始。:mdnice-bold-italic[最令人沮丧的是，上一个算子好不容易积攒的“屠龙之技”，几乎无法系统性地迁移到下一个算子上——每一次优化都像是推倒重来。]

这就解释了为什么国产 NPU 的“大神”价值被低估：**因为他们的生产力被手工作坊式的调试工具封印了。**

::: block-1
那么，如果存在这样一个系统：它能自动完成 :mdnice-bold-italic[“编译-验证-测速-剖析-提出假设-生成新版本-再验证”] 的完整闭环，更重要的是——**它还能记住哪些策略是有效的，哪些是死胡同，又会发生什么？**
:::

![文章插图](/articles/2026-04-15-36-star-ai-agent-nsight-5b49e858/images/01.webp)

::: image-caption
GPU 算子自动化优化与验证的全流程框架。流程始于 Preflight 环境检查，涵盖 GPU、Torch 等配置核验，接着进行正确性验证与基准测试，未通过则标记拒绝。通过后开展分层 NCU 剖析，结合知识库生成优化方案，迭代产出算子版本，并记录策略结果、记忆优化经验。若未达上限则循环优化，直至触发上限后选出最优版本，最终输出结构化产物与报告，全程支持 CUDA, CUTLASS, Triton 等三类后端，实现从检测到优化的闭环自动化流程。
:::

- :mdnice-bold-italic[A CUDA kernel optimization toolkit for validation, benchmarking, Nsight Compute profiling, bottleneck analysis, and iterative tuning. It helps improve custom GPU operators with reproducible workflows and evidence-based performance comparison.]
- https://github.com/KernelFlow-ops/cuda-optimized-skill
- 5000 字，阅读 18 分钟，**播客 15 分钟**

相关推荐

- [暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)
- [暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)
- [技能即记忆：AutoSkill 让 LLM 智能体从交互经验中自我进化，实现终身学习，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898295&idx=1&sn=6bf15a2f47d940cea0165f5d1f50aa61&scene=21#wechat_redirect)

> `cuda-optimized-skill` 正是这样一个项目。它并非又一个 GPU kernel 模板库，而是一个面向 **AI 编码智能体** （如 Claude）的"优化技能包"，把算子调优从人类专家的经验驱动，升级为数据驱动、可复现、有记忆的工程系统。

接下来，让我们先用最快的方式把它跑起来，然后深入它的代码，看看这套系统是如何被精心设计的。

## 本文目录

- 一、三分钟上手：从零开始跑通你的第一次优化
  - 1.1 环境准备
  - 1.2 最快体验：一条命令跑 benchmark
  - 1.3 完整体验：一条命令启动多轮迭代优化
  - 1.4 只检查环境，不跑 benchmark
  - 1.5 在 AI Agent 中使用（推荐姿势）
- 二、架构总览与设计哲学
  - 2.1 项目的定位：不是库，是"技能"
  - 2.2 三后端统一：一套流程，三种武器
- 三、Benchmark 引擎：统一的性能度量基座
  - 3.1 CUDA/CUTLASS 的签名解析与自动装配
  - 3.2 Triton 后端的约定式接口
  - 3.3 正确性验证：快了但错了，不算数
- 四、迭代编排引擎：optimize_loop.py 的精密流水线
  - 4.1 每轮迭代的强制流程
  - 4.2 两阶段 NCU 剖析策略
  - 4.3 Preflight：启动前的安全网
- 五、策略记忆机制：让 AI 不再重蹈覆辙
  - 5.1 问题：AI 的"遗忘"与"执念"
  - 5.2 指纹计算与分类判定
  - 5.3 两级记忆：当前 run + 全局
- 六、最优版本选择：严格的资格赛
- 七、实战效果：CUTLASS Softmax 优化案例
- 总结与技术价值

![文章插图](/articles/2026-04-15-36-star-ai-agent-nsight-5b49e858/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、三分钟上手：从零开始跑通你的第一次优化

技术解析固然重要，但"先跑起来"才是程序员的第一直觉。本章用最精简的步骤，带你在三种场景下快速体验这套系统的核心能力。

### 1.1 环境准备

> 你需要一台装有 NVIDIA GPU 的机器，并确保以下工具就绪：

```
# 基础依赖：Python 3.8+、PyTorch（CUDA 版本）、nvcc、ncupython -c "import torch; print(torch.cuda.is_available())"  # 应输出 Truenvcc --version   # CUDA/CUTLASS 后端需要ncu --version    # 所有后端的性能剖析需要
```

然后克隆项目：

```
git clone https://github.com/KernelFlow-ops/cuda-optimized-skill.gitcd cuda-optimized-skill
```

### 1.2 最快体验：一条命令跑 benchmark

如果你手上已有一个 CUDA kernel 文件（比如 `my_kernel.cu`，其中暴露了 `extern "C" void solve(...)` 接口），只需一行命令就能完成编译、运行和性能测量：

```
# 简单例子：只做 benchmark，无 reference 对比python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py my_kernel.cu \    --M=4096 --N=4096
```

脚本会自动检测 GPU 架构、用 `nvcc` 编译为共享库、分配 CUDA tensor 并运行 benchmark。如果你还有一个 Python reference 实现，加上 `--ref` 即可同时验证正确性：

```
# 正确性验证 + benchmark + speedup 对比python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py my_kernel.cu \    --ref=my_ref.py --M=4096 --N=4096
```

Triton 用户也只需一行：

```
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py my_triton_kernel.py \    --backend=triton --ref=my_ref.py --M=4096 --N=4096
```

::: block-1
**提示**：`--backend` 不填时，脚本会按文件后缀自动推断——`.py` 走 Triton，`.cu` 走 CUDA。
:::

### 1.3 完整体验：一条命令启动多轮迭代优化

> 真正的杀手锏是**多轮迭代优化**。一条命令即可启动"benchmark → NCU 剖析 → 策略记忆 → 下一版本"的完整闭环：

```
# CUDA 算子：3 轮迭代优化python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py my_kernel.cu \    --ref=my_ref.py --max-iterations=3 --M=4096 --N=4096
```

```
# CUTLASS 算子：3 轮迭代优化python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py my_cutlass.cu \    --backend=cutlass --ref=my_ref.py --max-iterations=3 --M=4096 --N=4096
```

```
# Triton 算子：3 轮迭代优化python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py my_triton.py \    --backend=triton --ref=my_ref.py --max-iterations=3 --M=4096 --N=4096
```

其中 `--max-iterations` 是**唯一必填参数** （除算子文件外）。执行完成后，你会在算子同目录下看到一个 `optimize_runs/run_YYYYMMDD_HHMMSS/` 目录，里面包含了每一轮的 benchmark 结果、NCU 报告、优化提案和最终摘要。

### 1.4 只检查环境，不跑 benchmark

> 不确定环境是否就绪？加一个 `--preflight-only` 就能做一次完整的环境检查：

```
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py my_kernel.cu \    --max-iterations=1 --preflight-only
```

这会生成 `preflight_check.md` 和 `preflight_check.json`，列出 GPU 型号、driver 版本、nvcc/ncu 路径、PyTorch CUDA 状态等一切信息——出了问题，一目了然。

### 1.5 在 AI Agent 中使用（推荐姿势）

> 本项目的设计初衷是作为 AI 编码智能体的"技能包"。如果你使用支持 Codex skill 的 Agent（如 Claude），最简单的方式是：

```
@optimized-skill 使用这个 skill 对 my_kernel.cu 进行优化，迭代次数为 3 次。
```

Agent 会自动调用 `optimize_loop.py`，阅读每一轮的 NCU 报告，生成下一版算子，并在策略记忆的约束下持续迭代，直到达到指定轮数或提前收敛。

## 二、架构总览与设计哲学

### 2.1 项目的定位：不是库，是"技能"

> 传统的 GPU 算子优化库（如 CUTLASS、Triton）提供的是**算子实现** 本身。而 `cuda-optimized-skill` 提供的是**优化过程的标准化编排能力** ——它假设上游有一个 AI 编码智能体（当前面向 Claude Codex），由智能体负责"理解代码、提出优化方案、生成新版本算子"，而**本项目则负责"跑 benchmark、抓 NCU 报告、判定策略成败、记住经验教训"。**

整个项目组织在 `skills/optimized-skill/` 目录下，由四个子模块构成：

```
skills/optimized-skill/├── kernel-benchmark/       # 统一 benchmark 入口├── ncu-rep-analyze/        # NCU 性能剖析与诊断├── operator-optimize-loop/ # 多轮迭代主编排引擎└── reference/              # 按后端组织的优化知识库    ├── cuda/    ├── cutlass/    └── triton/
```

::: block-1
这四个模块的关系可以用一句话概括：**knowledge（reference）驱动 proposal，benchmark + NCU 提供 evidence，optimize-loop 完成闭环编排，strategy-memory 沉淀经验**。
:::

### 2.2 三后端统一：一套流程，三种武器

> 项目同时支持三类 GPU 编程后端：

- **CUDA**：原生 `.cu` 文件，暴露 `extern "C" void solve(...)` 接口
- **CUTLASS**：同样是 `.cu` 文件，但使用 NVIDIA CUTLASS 模板库
- **Triton**：Python `.py` 文件，暴露 `setup()` + `run_kernel()` 接口

三者共享完全相同的评测-剖析-迭代流程，区别仅在于编译方式和优化策略来源。**这种设计的精妙之处在于：AI 智能体无需关心底层后端差异，只需按统一协议交付算子文件，优化引擎就能自动完成后续全部工作。**

## 三、Benchmark 引擎：统一的性能度量基座

### 3.1 CUDA/CUTLASS 的签名解析与自动装配

> benchmark 引擎（`benchmark.py`）是整个系统的"度量衡"。对于 CUDA/CUTLASS 后端，它做了一件非常巧妙的事情——**自动解析 C 函数签名，动态构造调用参数**。

```
// 来源：skills/optimized-skill/kernel-benchmark/scripts/benchmark.pydef parse_solve_signature(cu_file: str):    """从 extern "C" void solve(...) 中提取参数列表"""    with open(cu_file, "r", encoding="utf-8") as f:        content = f.read()    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'    match = re.search(pattern, content)    ifnot match:        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')    raw = match.group(1)    raw = re.sub(r"/\*.*?\*/", "", raw)       # 去注释    raw = re.sub(r"//[^\n]*", "", raw)    raw = " ".join(raw.split())    params = []    for token in raw.split(","):        # ... 逐参数匹配类型和名称        is_const = "const"in token        # 指针参数 → 自动分配 CUDA tensor        # 整型参数 → 从命令行维度参数中读取
```

这段代码通过正则表达式从 `.cu` 源码中提取 `solve(...)` 的参数签名，然后根据参数类型自动分配 GPU 内存。指针参数（如 `float*`）会自动映射为 PyTorch CUDA tensor，`const` 修饰的指针被视为输入，非 `const` 指针被视为输出。整型参数（如 `int M, int N`）则从命令行的 `--M=4096 --N=4096` 中获取。

::: block-1
这意味着：**只要你的 CUDA kernel 暴露了标准的 `solve(...)` 接口，benchmark 引擎就能零配置地为你构造完整的测试环境** ——包括输入张量的分配、随机初始化、输出张量的捕获。
:::

### 3.2 Triton 后端的约定式接口

> 对于 Triton 后端，benchmark 引擎则采用"约定大于配置"的策略：

```
// 来源：skills/optimized-skill/kernel-benchmark/scripts/benchmark.pydef _setup_triton(solution_file, dim_values, seed=None):    module = load_python_module(solution_file, "_triton_kernel_module")    # 要求模块必须定义 setup() 和 run_kernel()    prepared = module.setup(**setup_kwargs)    # setup() 返回 {"inputs": {...}, "outputs": ["out"]}    reference_inputs = prepared.get("inputs")    outputs = prepared.get("outputs")
```

Triton 模块需要实现两个函数：

- `setup()` 负责构造输入/输出张量并声明哪些是输出
- `run_kernel()` 负责执行 kernel

这种设计让 benchmark 引擎不需要了解 Triton 内核的任何实现细节，实现了真正的后端解耦。

### 3.3 正确性验证：快了但错了，不算数

> benchmark 引擎内建了一套严格的正确性验证流程。当用户提供 reference 实现时，引擎会：

1. 用相同的随机种子和输入数据，分别运行 kernel 和 reference
2. 逐张量对比输出，计算最大误差、平均误差、相对误差
3. **正确性失败时立即终止**，不会产出任何性能数据

```
// 来源：skills/optimized-skill/kernel-benchmark/scripts/benchmark.pydef _validate_outputs(kernel_tensors, ref_tensors, output_params, atol, rtol):    all_pass = True    for pname, ptype in output_params:        kt = kernel_tensors[pname].float()        rt = ref_tensors[pname].float()        match = torch.allclose(kt, rt, atol=atol, rtol=rtol)        if not match:            all_pass = False            # 报告第一个错误位置和具体数值差异            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)            bad_idx = diff_mask.nonzero(as_tuple=True)[0]
```

这个设计原则非常重要：**性能优化的前提是正确性**。在后面的迭代编排中，正确性失败的版本会被自动标记为 `rejected`，永远不会被选为最优版本。

## 四、迭代编排引擎：optimize_loop.py 的精密流水线

### 4.1 每轮迭代的强制流程

> 迭代编排引擎 `optimize_loop.py` 是整个系统的核心。它的设计目标是：**让每一轮迭代都可复现、有完整证据链、结果可追溯**。

```
// 来源：skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.pydef main() -> int:    # 1. 解析参数，推断后端    backend = infer_backend(solution_file, args.backend)    # 2. 创建/恢复 run 目录    run_dir = ensure_run_dir(solution_file, args.run_dir)    # 3. 初始化策略记忆    strategy_memory = ensure_strategy_memory(manifest, scope_key, global_memory_file)    # 4. Preflight 环境检查    preflight = collect_preflight(args, benchmark_script, solution_file, ref_file, backend)    # 5. 快照当前算子版本    snapshot_file = iter_dir / f"{solution_file.stem}_v{iteration}{solution_file.suffix}"    shutil.copy2(solution_file, snapshot_file)    # 6. 运行 benchmark    bench_res = run_command(bench_cmd, benchmark_stdout, benchmark_stderr)    # 7. 运行 targeted NCU（5 个关键 section）    # 8. 运行 full NCU（--set full）    # 9. 导入 NCU 报告为文本    # 10. 判定策略成败，更新记忆    # 11. 选出当前最优版本    # 12. 生成结构化 summary
```

每一轮迭代都会生成一个独立的 `iter_v{i}/` 目录，包含算子快照、benchmark 结果 JSON、NCU 报告（targeted 和 full）、诊断文本、迭代摘要等完整产物。我们在第一章中看到的那个 `optimize_runs/run_YYYYMMDD_HHMMSS/` 输出目录，正是由这段逻辑生成的。

### 4.2 两阶段 NCU 剖析策略

> NCU（Nsight Compute）剖析被精心设计为两阶段：

**第一阶段 targeted** ——只采集 5 个最关键的 section：

```
// 来源：skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.pyTARGETED_SECTIONS = [    "LaunchStats",    "Occupancy",    "SpeedOfLight",    "MemoryWorkloadAnalysis",    "SchedulerStats",]def build_targeted_ncu_cmd(args, bench_cmd, out_prefix):    cmd = [args.ncu_bin, "--target-processes", "all",           "--profile-from-start", "on",           "--launch-skip", str(args.launch_skip),           "--launch-count", str(args.launch_count)]    for section in TARGETED_SECTIONS:        cmd.extend(["--section", section])    cmd.extend(["-o", str(out_prefix), "-f"])    cmd.extend(bench_cmd)    return cmd
```

**第二阶段 full** ——使用 `--set full` 采集所有指标，提供完整的性能画像：

```
def build_full_ncu_cmd(args, bench_cmd, out_prefix):    cmd = [args.ncu_bin, "--target-processes", "all",           "--profile-from-start", "on",           "--launch-skip", str(args.launch_skip),           "--launch-count", str(args.launch_count),           "--set", "full"]    # ...
```

之所以分两阶段，是因为 `--set full` 开销巨大（可能比 targeted 慢 5~10 倍），而 targeted 的 5 个 section 已经覆盖了 90% 以上的常见瓶颈诊断场景。这种设计在效率和信息完备性之间取得了平衡。

### 4.3 Preflight：启动前的安全网

> 还记得第一章中提到的 `--preflight-only` 吗？它背后的实现是一套全面的环境预检机制：

```
// 来源：skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.pydef collect_preflight(args, benchmark_script, solution_file, ref_file, backend):    # 检查 GPU 型号、compute capability、driver version    torch_info = probe_torch_cuda(args.gpu)    # 检查 nvidia-smi    nvidia_smi_info = probe_nvidia_smi()    # CUDA/CUTLASS 检查 nvcc    if backend in {"cuda", "cutlass"}:        nvcc_info = probe_executable(args.nvcc_bin, "nvcc", ["--version"])    # 所有后端检查 ncu    if backend_supports_ncu(backend):        ncu_info = probe_executable(args.ncu_bin, "ncu", ["--version"])    # 交叉验证 --arch 与实际 GPU capability    if args.arch and gpu_info.get("sm") and args.arch != gpu_info["sm"]:        warnings.append(f"--arch={args.arch} does not match ...")
```

Preflight 检查甚至会交叉验证用户指定的 `--arch` 参数是否与实际 GPU 的 compute capability 匹配。所有结果都被序列化为 JSON 和 Markdown 两种格式——JSON 供程序消费，Markdown 供人类阅读。这种"双格式输出"贯穿了整个项目。

## 五、策略记忆机制：让 AI 不再重蹈覆辙

### 5.1 问题：AI 的"遗忘"与"执念"

> 在没有策略记忆的情况下，AI 智能体在多轮优化中容易犯两个错误：

1. **遗忘**：上一轮尝试过的失败策略，下一轮又重新提出
2. **执念**：某个曾经有效的优化方向，在不适用的场景下仍被反复推荐

策略记忆机制正是为了解决这两个问题。

### 5.2 指纹计算与分类判定

> 每一轮迭代的优化策略会被从 `optimization_proposal.md` 中提取为标签（strategy tags），然后计算一个唯一指纹：

```
// 来源：skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.pydef extract_strategy_tags(proposal_path: Path) -> list[str]:    """从 optimization_proposal.md 中提取 ## Strategy tags 下的标签列表"""    content = proposal_path.read_text(encoding="utf-8")    # 解析 Markdown 中 "## Strategy tags" section 下的 "- tag" 行    # ...def build_strategy_fingerprint(backend: str, tags: list[str]) -> str:    canonical = {"backend": backend, "tags": tags}    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
```

每个策略被分类为三种结果之一：

```
def classify_strategy_outcome(record, previous_record):    # benchmark 失败 → rejected    if record.get("benchmark_rc") != 0:        return ("rejected", "benchmark_failed")    # 正确性失败 → rejected    if bench.get("has_reference") and correctness.get("passed") is False:        return ("rejected", "correctness_failed")    # NCU 失败/缺失 → rejected    # 比上一轮更快 → positive    if current_median < previous_median:        return ("positive", "faster_than_previous")    # 否则 → negative    return ("negative", "slower_or_equal_to_previous")
```

### 5.3 两级记忆：当前 run + 全局

> 策略记忆存储在两个层级：

- **当前 run**：存在 `run_manifest.json` 的 `strategy_memory.current_run` 字段中
- **全局**：存在 `global_strategy_memory.json` 中，按 scope key 隔离

```
// 来源：skills/optimized-skill/operator-optimize-loop/strategy-memory/global_strategy_memory.json{"version": 1,"scopes": {    "cutlass__cutlass_softmax__cutlass_softmax_ref__auto_arch__339090bb90fe": {      "positive": {        "693f19f37331ba8b": {          "tags": ["register_cached_exp", "rowwise_softmax_shape_specialization",                   "thread_retile_128", "vectorized_float4_io"],          "last_outcome": "positive",          "last_reason": "faster_than_previous",          "evidence": {            "baseline_median_ms": 0.14894,            "current_median_ms": 0.10317          }        }      },      "negative": {        "f5f0e39777db3029": {          "tags": ["keep_vectorized_single_pass", "register_fragment_expansion",                   "retile_64_for_higher_ilp"],          "last_outcome": "negative",          "last_reason": "slower_or_equal_to_previous"        }      }    }  }}
```

下一轮迭代开始时，系统会自动合并两级记忆，生成约束：

```
// 来源：skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.pydef merge_strategy_constraints(run_memory, global_scope):    blocked = set(run_memory["negative"].keys()) | set(run_memory["rejected"].keys())    preferred = set(run_memory["positive"].keys())    # 再合并全局记忆    blocked.update(global_scope["negative"].keys())    preferred.update(global_scope["positive"].keys())    return {"blocked": sorted(blocked), "preferred": sorted(preferred)}
```

`blocked` 列表中的策略指纹会被显式禁止再次使用，`preferred` 列表中的会被优先融合。这等于给 AI 智能体装上了一个"集体记忆"——不仅记住当前任务的经验，还能跨任务复用。

## 六、最优版本选择：严格的资格赛

> 不是每个跑出数字的版本都有资格参与"最优版本"的评选。评选规则非常严格：

```
// 来源：skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.pydef choose_best_iteration(iterations):    candidates = []    for item in iterations:        # 条件 1：full NCU 报告必须存在        if ncu_required andnot item.get("full_report_exists"):            continue        # 条件 2：有 reference 时，正确性必须通过        if bench.get("has_reference") andnot correctness.get("passed"):            continue        # 条件 3：必须有有效的 median 和 average        if kernel.get("median_ms") isNoneor kernel.get("average_ms") isNone:            continue        candidates.append(item)    # 排序：median 最低 → average 最低 → 更早的版本优先    return min(candidates, key=lambda item: (        item["benchmark_result"]["kernel"]["median_ms"],        item["benchmark_result"]["kernel"]["average_ms"],        item["iteration"],    ))
```

> 注意这里的设计哲学：

1. **full NCU 报告是硬性要求** ——没有完整的性能剖析证据，即使数字再漂亮也不认可
2. **正确性是一票否决** ——快了但错了，直接淘汰
3. **排序优先看 median 而非 average** ——因为 median 对噪声更鲁棒

这与第一章中"一条命令跑通"的简洁体验形成了鲜明对比：**入口极简，标准极严**。

## 七、实战效果：CUTLASS Softmax 优化案例

> 项目仓库中记录了一个真实的优化案例：使用 CUTLASS 后端对 softmax 算子进行 4 轮迭代优化，对标 PyTorch 原生 `softmax`。实际执行命令正如我们在第一章中展示的那样精简：

```
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py \    example/cutlass_softmax.cu \    --backend=cutlass --ref=example/cutlass_softmax_ref.py \    --max-iterations=3 --rows=4096 --cols=1024
```

从 `global_strategy_memory.json` 中可以看到完整的优化轨迹：

| 版本 | 策略标签 | 结果 | Kernel median (ms) |
| --- | --- | --- | --- |
| v0 | baseline | positive | 0.14894 |
| v1 | vectorized_float4_io, thread_retile_128, register_cached_exp | positive | 0.10317 |
| v2 | retile_64_for_higher_ilp, register_fragment_expansion | **negative** | 0.14587 |
| v3 | approximate_exp_path, revert_to_128_tile, arithmetic_only_tuning | positive | 0.10245 |

v2 试图通过缩小 tile 到 64 来提高指令级并行度（ILP），:mdnice-bold-italic[结果反而导致性能回退——这个策略的指纹 `f5f0e39777db3029` 被记入 `negative`，后续不会再被采用。]

v3 则聪明地**回退到 128 tile** （因为 v1 的记忆显示它是有效的），同时尝试近似 exp 路径，最终将 kernel median 从 0.14894ms 降到 0.10245ms，**延迟下降 31.2%，相对 PyTorch reference 的加速比达到 2.04x**。

::: block-1
这个案例完美展示了策略记忆机制的价值：**v3 之所以能做出正确决策，正是因为它"记住了" v1 的成功和 v2 的失败。**
:::

## 总结与技术价值

> `cuda-optimized-skill` 的本质贡献，不在于某一个具体的 CUDA kernel 写得有多快，而在于它**重新定义了 GPU 算子优化的工作范式**：

1. **从人类经验驱动到数据证据驱动**：每一轮优化决策都基于 benchmark 数字和 NCU 报告中的具体硬件计数器，而非工程师的直觉。
2. **从一次性手工调优到可复现的自动化闭环**：正如第一章所展示的，:mdnice-bold-italic[一条命令即可启动完整的 preflight → benchmark → NCU → proposal → 下一版本的流水线]，每一步都有结构化产物。
3. **从"调完就忘"到有记忆的持续进化**：策略记忆机制让优化经验跨轮次、跨任务沉淀，AI 智能体越用越聪明。
4. **从单一后端到三后端统一编排**：CUDA、CUTLASS、Triton 共享一套评测和迭代框架，降低了多后端切换的认知负担。

在 AI 编码智能体快速发展的今天，这个项目指向了一个值得关注的方向：:mdnice-bold-italic[AI 不仅可以写代码，还可以像一个有经验的性能工程师那样，系统性地优化代码——前提是，你给它配上正确的"工具"和"记忆"。]

> 而你要做的，可能只是一条命令。
