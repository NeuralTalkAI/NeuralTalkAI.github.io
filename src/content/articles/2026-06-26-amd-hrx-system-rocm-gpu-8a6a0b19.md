**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：HRX、**HIP 兼容层**、:mdnice-bold-italic[IREE 运行时]、**时间线信号量**、透明拦截

> 当你在 AMD GPU 上跑一个 PyTorch 模型，链接的其实是 `/opt/rocm/lib/libamdhip64.so`——一个十几年演进、夹带历史包袱的 HIP 运行时。**它能跑，但延迟不够低、对 NPU/CPU 异构不够友好、对调试也不够透明。**

![文章插图](/articles/2026-06-26-amd-hrx-system-rocm-gpu-8a6a0b19/images/01.webp)

- :mdnice-bold-italic[HRX System：HRX 系统由一组精简运行时组件构成，针对当前 ROCm 内置的 HIP 提供一套替代实现方案。该系统搭建通用底层支撑，可与 AMD 旗下 GPU、NPU 及 CPU 产品实现低延迟、高性能的适配集成。]
- https://github.com/ROCm/hrx-system

::: related-title
**相关推荐**
:::

::: related-links
- **[用 TileLang 在 AMD Radeon 上榨干 GPU 性能：RDNA4 GEMM 118.5 TFLOPS 持平 rocBLAS，Dequant达 93 TFLOPS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901949&idx=1&sn=4520024a563e4c5821d9dcf98d83c8ac&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**

:::

> ROCm 团队在 GitHub 低调地开源了一个名为 `ROCm/hrx-system`[1] 的项目，README 第一行就用 `[!CAUTION]` 标注它"**不是 ROCm 官方栈的一部分** "，却又毫不掩饰地表示:mdnice-bold-italic[要"为 AMD 的 GPU、NPU、CPU 提供一个低延迟、高性能的公共底座] "。**HRX——HIP Runtime Extended——究竟是一次实验性的另起炉灶，还是 ROCm 下一代运行时的雏形？**

HRX 是 ROCm 团队基于 IREE 重构的极简 HIP 替代运行时，通过统一时间线信号量、流式调度与 dlopen 透明拦截层，让现有 HIP 应用免重编即切换后端，兼顾低延迟与跨架构可观测性，重塑 AMD GPU 软件栈。

::: block-1
本文将从公开仓库切入，沿着 `libhrx/include/hrx_runtime.h` 的 C ABI、`libhrx/src/libhrx/runtime.c` 的 IREE 集成，再到 `libhrx/src/passthrough/` 的 `LD_PRELOAD` 拦截层，**拆解它如何在一个 `libhrx.so` 里同时塞下"新运行时"和"老 HIP 兼容层"两层皮。**
:::

## 本文目录

- 快速上手、使用
- 一、架构总览与设计哲学
  - 1.1 三层叠放的运行时模型
  - 1.2 设计哲学：把"对象生命周期"做对
- 二、核心 C ABI：hrx_runtime.h 的极简美学
  - 2.1 状态码与 IREE 完全二进制兼容
  - 2.2 主机分配器：两词值类型 + 即时可用
  - 2.3 时间线信号量：HRX 的同步原语
  - 2.4 Stream 与 queue 双轨 API
  - 2.5 直接可执行体 vs VM 字节码
- 三、运行时实现剖析：runtime.c 的核心调度
  - 3.1 三个全局单例：把状态收敛到最少
  - 3.2 共享基础设施按需创建
  - 3.3 创建本地任务设备：CPU 后端的精髓
  - 3.4 GPU 设备初始化：识别"伪设备"是关键
  - 3.5 通过 gfxip 还原架构字串
  - 3.6 环境变量即配置：HRX_PROFILE_MODE 的设计
- 四、HIP 兼容层与透明拦截：libhrx 的另一面
  - 4.1 拦截层的三段式架构
  - 4.2 关键代码：函数表 + pthread_once 一次性初始化
  - 4.3 转发：从应用到 active 表
  - 4.4 现成两个拦截器：日志 & buffer 追踪
- 五、测试体系：从 CTS 到 YAML 执行清单
  - 5.1 CTS：分层覆盖运行时 API
  - 5.2 YAML 执行清单：声明式 e2e 测试
- HRX 的意义：把 HIP 重塑成一组可替换的层

![文章插图](/articles/2026-06-26-amd-hrx-system-rocm-gpu-8a6a0b19/images/02.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 快速上手、使用

> HRX 自身依赖 ROCm 的 HSA / AQL Profile / ROCm LLVM 工具链，构建脚本默认走 CMake + Ninja。最小可用流程如下（详细的 hermetic 构建、Bazel 路径与发布包用法参见 README[2]）：

```
# 1. 准备 Python 虚拟环境（仅 PyYAML 一个依赖）python -m venv .venvuv pip install --python .venv/bin/python -r requirements-dev.lock.txt# 2. 用 ROCm 的 clang/lld 配置cmake -S . -B build/hrx-system -GNinja \  -DCMAKE_PREFIX_PATH=/opt/rocm \  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \  -DCMAKE_BUILD_TYPE=RelWithDebInfo \  -DIREE_ROCM_TEST_TARGET_CHIP=gfx942cmake --build build/hrx-systemcmake --install build/hrx-system --prefix build/hrx-install --component HrxPublicDist
```

安装产物会得到 `lib/libhrx.so`、`hrx-info` 以及一个名为 `lib/libamdhip64.so` 的兼容层。**真正有意思的用法** 是直接把现成的 HIP 应用接管过来——不必重新编译：

```
export HRX_ROOT=/tmp/hrx-rootexport LD_LIBRARY_PATH="${HRX_ROOT}/lib:${HRX_ROOT}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"export HRX_GPU_DRIVER=amdgpuLD_PRELOAD="${HRX_ROOT}/lib/libamdhip64.so" ./my_hip_application
```

- 测试矩阵（含 CTS、execution test、benchmark）通过 CTest 驱动，可在 CPU 节点上通过 `-LE requires-gpu-` 排除 GPU 用例；
- 执行清单是 YAML，由 `build_tools/testing/` 提供的 Python runner[3] 解析。

::: block-1
这种"零重编 + 可拦截 + 可结构化测试"的组合，正是 HRX 区别于传统 HIP 运行时的第一印象。
:::

## 一、架构总览与设计哲学

> HRX 仓库的根目录非常克制，几乎只有四个有意义的子树：

```
libhrx/        # 公共 C ABI、C++ 包装、HIP 兼容、CTS、透传调试runtime/       # 核心运行时基础设施（与 IREE 共享的 CMake bootstrap）build_tools/   # Bazel↔CMake 互转、CI 打包、第三方 overlaydocs/          # 安装测试树等使用文档
```

这种"按发布物切目录"的布局，本身就传达了一个设计哲学：**HRX 不仅仅是一个大而全的 SDK，而是一组可被独立安装、独立替换的薄层组件**。README 用一句话把这层意图说得很清楚——`public install exports libhrx.so, hrx-info, public HRX headers …, and the HRX HIP compatibility layer as lib/libamdhip64.so`。也就是说，HRX 想做的是 ROCm 软件栈里"被替换的那一格"。

### 1.1 三层叠放的运行时模型

把 `libhrx/include/hrx_runtime.h` 和 `libhrx/src/libhrx/runtime.c` 对照着读，可以画出 HRX 运行时的三层结构：

```
┌────────────────────────────────────────────────────────────┐│  binding/hip/api.c        ← HIP 应用看到的 ABI（仿 ROCm）  │├────────────────────────────────────────────────────────────┤│  libhrx/ : hrx_runtime.h  ← HRX 原生 C ABI（低延迟新接口） │├────────────────────────────────────────────────────────────┤│  IREE HAL / VM            ← 真正干活的底层（amdgpu 驱动）  │└────────────────────────────────────────────────────────────┘
```

- 最上层 `binding/hip` 是 *为兼容性而生* 的胶水：把 `hipMalloc / hipMemcpyAsync / hipModuleLaunchKernel` 这类签名映射到中间层的 HRX 原生 API；中间层 `libhrx` 提供了一套带"流（stream）+ 时间线信号量（timeline semaphore）+ 命令缓冲"语义的现代抽象；
- 最下层则把所有真正的设备访问下沉到 IREE[4] 的 HAL（Hardware Abstraction Layer）与 VM（Virtual Machine）。

::: block-1
这种分层带来的最大好处是：**HIP 兼容层只是 HRX 的一个客户端**。如果未来需要新增 SYCL、Triton runtime、自定义低延迟 API，只需要在 `binding/` 下再加一个目录，无需改动核心。
:::

### 1.2 设计哲学：把"对象生命周期"做对

> 读 `hrx_runtime.h` 时最容易忽略的是注释里那些 retain/release 规则。HRX 几乎所有对象都被设计成 **引用计数 + 不透明指针**：

```
// 来源：libhrx/include/hrx_runtime.htypedefstruct hrx_device_s        *hrx_device_t;typedefstruct hrx_allocator_s     *hrx_allocator_t;typedefstruct hrx_semaphore_s     *hrx_semaphore_t;typedefstruct hrx_stream_s        *hrx_stream_t;typedefstruct hrx_buffer_s        *hrx_buffer_t;typedefstruct hrx_module_s        *hrx_module_t;typedefstruct hrx_fence_s         *hrx_fence_t;typedefstruct hrx_executable_s    *hrx_executable_t;
```

每种类型都有显式的 `*_retain()` / `*_release()`，并且头文件明确区分"借用引用"和"持有引用"——`hrx_device_allocator()` 返回的是借用引用，调用者不允许 release，除非自己先 retain。

对运行时来说，这种约定避免了 HIP 历史上"谁创建谁销毁"语义混乱的陷阱，也使得跨语言绑定（PyTorch / Python）写起来非常安全。

## 二、核心 C ABI：hrx_runtime.h 的极简美学

> `libhrx/include/hrx_runtime.h` 全文**不到一千行，却覆盖了一个完整 GPU 运行时所需的全部概念**。:mdnice-bold-italic[它的关键设计可以拆成五个簇。]

### 2.1 状态码与 IREE 完全二进制兼容

```
// 来源：libhrx/include/hrx_runtime.htypedefenumhrx_status_code_t {  HRX_STATUS_OK = 0,  HRX_STATUS_CANCELLED = 1,  HRX_STATUS_UNKNOWN = 2,  HRX_STATUS_INVALID_ARGUMENT = 3,  ...  HRX_STATUS_DATA_LOSS = 15,} hrx_status_code_t;// 不透明指针：NULL 即 OK，非 NULL 携带错误负载typedefstruct hrx_status_s *hrx_status_t;static inline bool hrx_status_is_ok(hrx_status_t s) { return s == NULL; }
```

头文件注释直说 `Values match iree_status_code_t. Verified by _Static_assert in implementation.`——**HRX 没有发明新的错误码，而是直接借 IREE 的编号。**

**这让"穿透层"几乎为零开销**：底层 IREE 抛出的 `iree_status_t` 可以经过一次指针类型 cast 就转成 `hrx_status_t`，避免冗长的 switch 翻译。

### 2.2 主机分配器：两词值类型 + 即时可用

```
// 来源：libhrx/include/hrx_runtime.htypedefstruct hrx_host_allocator_t {void *self;void *ctl;} hrx_host_allocator_t;// 在 libhrx.so 加载完成的瞬间即可用，无需任何 initialize()HRX_API externhrx_host_allocator_t hrx_host_allocator_system_value;static inline hrx_host_allocator_t hrx_host_allocator_system(void) {return hrx_host_allocator_system_value;}
```

这是一个非常**系统级** 的细节：HRX 的主机分配器是一个**两字值类型**，布局与 `iree_allocator_t` 完全兼容；并且通过 ELF 数据导出符号让它在加载时就可用。其底层在可用时会回落到 `mimalloc`，对追求 alloc/free 微秒级延迟的推理路径非常关键。

### 2.3 时间线信号量：HRX 的同步原语

> HRX 最具现代感的设计是把 timeline semaphore（来自 Vulkan/IREE）作为唯一的跨设备同步原语，而非沿用 HIP 的 event/stream wait 体系：

```
// 来源：libhrx/include/hrx_runtime.htypedef struct hrx_timeline_point_t {  hrx_semaphore_t semaphore;  uint64_t value;} hrx_timeline_point_t;HRX_API hrx_status_t hrx_semaphore_create(hrx_device_t device,                                          uint64_t initial_value,                                          hrx_semaphore_t *semaphore);HRX_API hrx_status_t hrx_semaphore_wait(hrx_semaphore_t sem,                                        uint64_t value, uint64_t timeout_ns);HRX_API hrx_status_t hrx_semaphore_signal(hrx_semaphore_t sem, uint64_t value);
```

时间线信号量比传统 binary semaphore 强在哪里？它把"事件"映射成单调递增的 64 位整数：生产者写 `value=N` 表示"我做完第 N 件事"，消费者 `wait(value=N)` 表示"我至少要等做完第 N 件事再走"。一条流可以在不创建新对象的情况下表达任意多个同步点。

HRX 进一步把流和时间线绑定，注释里有一段写得格外清楚：

> hrx_stream_t is the canonical owner of a stream timeline/semaphore inside libhrx.

也就是说，前端 stream 对象不要再去创建第二套时间线宇宙，而应该 retain `hrx_stream_t` 并通过 `hrx_stream_get_semaphore()` 共享。这是写跨层运行时时一个非常关键的工程纪律——避免"双时间线"导致的死锁与竞态。

### 2.4 Stream 与 queue 双轨 API

> HRX 同时提供了两种执行模型，给上层留出最大灵活度：

```
// 来源：libhrx/include/hrx_runtime.h// （A）Stream 模型：批量、命令缓冲式HRX_API hrx_status_t hrx_stream_fill_buffer(hrx_stream_t stream,                                            hrx_buffer_t buffer, size_t offset,                                            size_t size, const void *pattern,                                            size_t pattern_size);HRX_API hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream, ...);HRX_API hrx_status_t hrx_stream_dispatch(hrx_stream_t stream,                                         hrx_executable_t executable, ...);// （B）Queue 模型：单操作、显式 wait/signal 列表HRX_API hrx_status_t hrx_queue_dispatch(    hrx_device_t device, hrx_queue_affinity_t affinity,    const hrx_semaphore_list_t *wait_semaphores,    const hrx_semaphore_list_t *signal_semaphores,    hrx_executable_t executable, uint32_t export_ordinal,    const hrx_dispatch_config_t *config, ...);
```

- Stream 模型适合"我有一串紧接着的 op，请你帮我批起来再 flush"的场景（这是 HIP 用户的常态）；
- queue 模型则适合"这一发我自己管同步语义，请最低延迟地调度它"——例如一个推理 server 想要在不同请求之间共享细粒度调度时。

两套 API 共享底层时间线，避免重复造轮子。

### 2.5 直接可执行体 vs VM 字节码

> HRX 同时支持加载 `native executable`（如 HSACO）做直接 dispatch，也支持加载 VMFB 字节码模块通过 IREE VM 走"模块函数调用"：

```
// 来源：libhrx/include/hrx_runtime.hHRX_API hrx_status_t hrx_executable_load_data(hrx_device_t device,                                              const void *executable_data,                                              size_t executable_data_size,                                              const char *executable_format,                                              hrx_executable_t *executable);HRX_API hrx_status_t hrx_module_load_vmfb(hrx_device_t device,                                          const void *vmfb_data,                                          size_t vmfb_size,                                          hrx_module_t *module);HRX_API hrx_status_t hrx_function_invoke(hrx_module_t module,                                         hrx_function_t function,                                         hrx_value_list_t args,                                         hrx_value_list_t rets);
```

这是有意为之的"两条腿走路"：

- HIP 用户继续传二进制内核 + 自己组 launch；
- IREE 用户则可以把整张计算图编译成 VMFB 喂给 HRX 跑。

两者共享 buffer/fence/semaphore，无须额外的数据搬运。

## 三、运行时实现剖析：runtime.c 的核心调度

> `libhrx/src/libhrx/runtime.c` 是整个仓库最值得逐行读的文件——它把 IREE 的几大子系统（VM Instance、HAL Driver、Task Executor、Proactor Pool）按 HRX 的语义编排起来。

### 3.1 三个全局单例：把状态收敛到最少

```
// 来源：libhrx/src/libhrx/runtime.cstatic hrx_shared_state_t g_shared = {0};   // VM 实例 + proactor 池static hrx_gpu_state_t    g_gpu    = {0};   // 全部 GPU 设备static hrx_cpu_state_t    g_cpu    = {0};   // 全部 CPU 设备hrx_shared_state_t *hrx_get_shared_state(void) { return &g_shared; }hrx_gpu_state_t    *hrx_get_gpu_state(void)    { return &g_gpu; }hrx_cpu_state_t    *hrx_get_cpu_state(void)    { return &g_cpu; }
```

HRX 不像传统 HIP runtime 那样把"上下文"挂在 TLS 上，而是用三个进程级单例。`g_shared` 用**引用计数 `init_count`** 来支持"GPU 与 CPU 任意先后初始化、任意先后释放"——谁是最后一个 release 的，谁负责把 VM 实例和异步 I/O 池销毁。

### 3.2 共享基础设施按需创建

```
// 来源：libhrx/src/libhrx/runtime.chrx_status_t hrx_ensure_shared_state(void) {if (g_shared.shared_initialized) {  // 已初始化：只增加引用    g_shared.init_count++;    return hrx_ok_status();  }  g_shared.host_allocator = iree_allocator_system();// 1. 创建 IREE VM 实例并注册 HAL 类型iree_status_t status = iree_vm_instance_create(      IREE_VM_TYPE_CAPACITY_DEFAULT, g_shared.host_allocator,      &g_shared.vm_instance);if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);  status = iree_hal_module_register_all_types(g_shared.vm_instance);  ...// 2. 创建异步 I/O 所需的 proactor pool（local-task 设备依赖）uint32_t node_id = 0;  status = iree_async_proactor_pool_create(      /*node_count=*/1, &node_id,      iree_async_proactor_pool_options_default(),      g_shared.host_allocator, &g_shared.proactor_pool);  ...  g_shared.shared_initialized = true;  g_shared.init_count = 1;return hrx_ok_status();}
```

注意每一段失败路径都会逐级 release 已成功创建的对象——`iree_vm_instance_release(g_shared.vm_instance); g_shared.vm_instance = NULL;`。这种 *failure unwinding* 看似啰嗦，却是写 C 系统级代码必须的纪律，等同于 C++ 的 RAII。

### 3.3 创建本地任务设备：CPU 后端的精髓

CPU 后端走的是 IREE 的 `local-task` 驱动：

```
// 来源：libhrx/src/libhrx/runtime.cstatic hrx_status_t hrx_create_local_task_device(    int group_count, iree_task_executor_t **out_executor,    iree_hal_driver_t **out_driver, iree_hal_device_t **out_hal_device) {iree_allocator_t alloc = g_shared.host_allocator;// 1. Task topology + executor（线程池）iree_task_topology_t topology;  iree_task_topology_initialize_from_group_count(group_count, &topology);iree_task_executor_options_t exec_options;  iree_task_executor_options_initialize(&exec_options);// GPU 运行时的 TLS 可能拉高 pthread 最小栈，256KB 对 ASAN 也安全  exec_options.worker_stack_size = 256 * 1024;iree_task_executor_t *executor = NULL;iree_status_t status = iree_task_executor_create(      exec_options, &topology, alloc, &executor);// 2. 加载所有可用的 executable loader（VMVX、ELF...）iree_hal_executable_loader_t *loaders[8] = {NULL};iree_host_size_t loader_count = 0;  status = iree_hal_create_all_available_executable_loaders(      /*plugin_manager=*/NULL, IREE_ARRAYSIZE(loaders),      &loader_count, loaders, alloc);// 3. 堆分配器 + 组装 task driveriree_hal_allocator_t *device_allocator = NULL;  iree_hal_allocator_create_heap(iree_make_cstring_view("hrx"),                                 alloc, alloc, &device_allocator);iree_hal_task_device_params_t task_params;  iree_hal_task_device_params_initialize(&task_params);iree_hal_driver_t *driver = NULL;  status = iree_hal_task_driver_create(      iree_make_cstring_view("local-task"), &task_params,      /*queue_count=*/1, &executor, loader_count, loaders,      device_allocator, alloc, &driver);// 4. 由 driver 创建默认 device，绑定 proactor 池iree_hal_device_create_params_t device_params =      iree_hal_device_create_params_default();  device_params.proactor_pool = g_shared.proactor_pool;iree_hal_device_t *hal_device = NULL;  iree_hal_driver_create_default_device(driver, &device_params, alloc, &hal_device);  *out_driver = driver;  *out_hal_device = hal_device;return hrx_ok_status();}
```

这段代码值得细看的是 **`worker_stack_size = 256 * 1024;`** 这种来自实战的小常数：注释说 GPU runtime 的 TLS 可能让 pthread 最小栈从 16KB 起跳，256KB 既能跑 ASAN 也不至于太奢侈。这种"在抽象之中保留一道工程后门"的做法在 HRX 里很常见。

### 3.4 GPU 设备初始化：识别"伪设备"是关键

`hrx_gpu_initialize()` 里最有趣的是它如何处理 IREE AMDGPU 驱动报告的设备列表：

```
// 来源：libhrx/src/libhrx/runtime.c// IREE AMDGPU 会在 ordinal 0 报告一个空 path 的"伪设备"，代表// 所有可见 GPU 作为一个逻辑设备；后续 entry 才是真物理设备。// HRX 只向调用者暴露物理设备。int physical_count = 0;for (iree_host_size_t i = 0; i < device_info_count; ++i) {if (device_infos[i].path.size == 0) continue;  // 跳过聚合伪设备  physical_count++;}int created_count = 0;for (iree_host_size_t info_index = 0;     info_index < device_info_count && created_count < count; ++info_index) {if (device_infos[info_index].path.size == 0) continue;iree_hal_device_t *hal_device = NULL;  iree_status = iree_hal_driver_create_device_by_ordinal(      driver, info_index, /*param_count=*/0, /*params=*/NULL,      &create_params, alloc, &hal_device);  ...  hrx_device_s *dev = &g_gpu.devices[created_count];  dev->type = HRX_ACCELERATOR_GPU;  dev->ordinal = created_count;  dev->hal_device = hal_device;  dev->allocator.hal_allocator = iree_hal_device_allocator(hal_device);  ...  hrx_set_gpu_architecture_from_hal(hal_device, dev);   // 解析 gfxip → "gfx942"  hrx_device_profile_begin(dev, profile_sink);          // 可选 profiling  created_count++;}
```

这种"过滤底层骚扰条目"的逻辑，是写 wrapper 时的常见痛点——HRX 用一个跳过空 path 的简单规则，把"逻辑设备 + 物理设备"的混合列表整理成一个干净、连续编号的物理设备数组，对上层呈现的就是"GPU 0、GPU 1、GPU 2"。

### 3.5 通过 gfxip 还原架构字串

注释里写得很直白：

```
// 来源：libhrx/src/libhrx/runtime.cstatic void hrx_set_gpu_architecture_from_hal(iree_hal_device_t *hal_device,                                              hrx_device_s *dev) {int64_t gfxip = 0;iree_status_t status = iree_hal_device_query_i64(      hal_device, iree_make_cstring_view("hal.device"),      iree_make_cstring_view("gfxip"), &gfxip);if (!iree_status_is_ok(status)) {    snprintf(dev->architecture, sizeof(dev->architecture), "unknown");    return;  }int major    = (int)((gfxip >> 16) & 0xff);int minor    = (int)((gfxip >> 8)  & 0xff);int stepping = (int)( gfxip        & 0xff);snprintf(dev->architecture, sizeof(dev->architecture),           "gfx%d%d%d", major, minor, stepping);}
```

`gfxip` 编码成 `(major<<16) | (minor<<8) | stepping`——读出来再格式化成 `"gfx942"` 这种 ROCm 用户熟悉的字符串。HRX 顺手把这个细节封装在内部，调用者只用 `hrx_device_get_property(dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, …)` 就能拿到。

### 3.6 环境变量即配置：HRX_PROFILE_MODE 的设计

```
// 来源：libhrx/src/libhrx/runtime.cstatic iree_status_t hrx_get_profile_data_families(    iree_hal_device_profiling_data_families_t *out_data_families) {constchar *value = getenv("HRX_PROFILE_MODE");if (!value || !value[0] || strcmp(value, "queue") == 0) {    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;  } elseif (strcmp(value, "dispatch") == 0) {    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |                         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;  } elseif (strcmp(value, "executable") == 0) {    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;  } elseif (strcmp(value, "all") == 0) {    *out_data_families = ...;  // 全部位或  } else {    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                            "unsupported HRX_PROFILE_MODE '%s'", value);  }return iree_ok_status();}
```

`HRX_GPU_DRIVER`、`HRX_GPU_DEBUG`、`HRX_PROFILE_FILE`、`HRX_PROFILE_MODE`、`HRX_TEST_TMPDIR`……运行时几乎所有运行期可调的开关都通过环境变量提供。这是一种典型的 *为生产部署而生* 的设计：不需要重新编译，不需要新 API，运维只改 `systemd` unit 里的 `Environment=` 就能切换 profiling 颗粒度。

## 四、HIP 兼容层与透明拦截：libhrx 的另一面

如果说 `libhrx.so` + 原生 ABI 是"未来"，那 `lib/libamdhip64.so` 这层"假身份"就是 HRX 通往现实世界的钥匙。HRX 在 `libhrx/src/passthrough/` 提供了一整套用于"截胡"HIP 调用的工具。

### 4.1 拦截层的三段式架构

`libhrx/src/passthrough/README.md` 里画了一张极简却精准的架构图：

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐│ Application │───▸│ libhip_intercept │───▸│ Real libamdhip64 │└─────────────┘    └────────┬─────────┘    └──────────────────┘                            │ (可选)                            ▼                   ┌──────────────────┐                   │   Interceptor    │                   │  (logging, etc.) │                   └──────────────────┘
```

应用以为自己链接的还是 `libamdhip64.so`，实际链接的是 `libhip_intercept.so`；后者在首次使用时通过 `dlopen` 加载真正的 HIP 库，再可选地加载一个用户提供的拦截器 `.so` 来包装函数表。这套机制和 OpenGL 圈子里的 `apitrace`、`renderdoc` 同源，但 HRX 把它做成了 ROCm 一等公民。

### 4.2 关键代码：函数表 + pthread_once 一次性初始化

```
// 来源：libhrx/src/passthrough/passthrough.cstaticpthread_once_t g_init_once = PTHREAD_ONCE_INIT;staticvoid *g_backend_lib = NULL;staticvoid *g_interceptor_lib = NULL;statichip_function_table_t  g_real_table   = {0};statichip_function_table_t *g_active_table = NULL;static void passthrough_init(void) {constchar *backend_path = getenv("HIP_PASSTHROUGH_BACKEND_LIB");if (!backend_path) { /* 报错并 abort，强制用户显式配置 */ }  g_backend_lib = dlopen(backend_path, RTLD_NOW | RTLD_LOCAL);  load_hip_symbols(g_backend_lib, &g_real_table);  // 用 dlsym 填充函数表  g_active_table = &g_real_table;                  // 默认直通constchar *interceptor_path = getenv("HIP_PASSTHROUGH_INTERCEPTOR");if (interceptor_path && *interceptor_path) {    g_interceptor_lib = dlopen(interceptor_path, RTLD_NOW | RTLD_LOCAL);    pfn_hip_interceptor_init init_fn = (pfn_hip_interceptor_init)        dlsym(g_interceptor_lib, "hip_interceptor_init");    if (init_fn) {      hip_function_table_t *t = init_fn(&g_real_table);      if (t) g_active_table = t;   // 拦截器返回非 NULL 表示接管    }  }}
```

整个初始化由 `pthread_once` 保护，零分配、零等待、O(符号数)，第一次 HIP 调用进来时才付出代价。**`RTLD_LOCAL` 是有意为之的**——避免拦截器和真 HIP 库的符号互相污染，确保我们 `dlsym` 出来的指针是真 HIP 的，而不是被拦截器替换的版本。

### 4.3 转发：从应用到 active 表

每个被导出的 HIP 函数实现都长这副样子（这正是为什么 `stubs_generated.c` 高达 22 万行——它代表了 HIP 完整 ABI 的所有 stub）：

```
// 来源：libhrx/src/passthrough/passthrough.chipError_t hipMalloc(void **ptr, size_t size) {  ensure_initialized();              // 触发 pthread_onceif (!g_active_table->hipMalloc) return1;return g_active_table->hipMalloc(ptr, size);}hipError_t hipModuleLaunchKernel(hipFunction_t f,    unsigned gridX, unsigned gridY, unsigned gridZ,    unsigned blockX, unsigned blockY, unsigned blockZ,    unsigned sharedMemBytes, hipStream_t stream,    void **kernelParams, void **extra) {  ensure_initialized();if (!g_active_table->hipModuleLaunchKernel) return1;return g_active_table->hipModuleLaunchKernel(      f, gridX, gridY, gridZ, blockX, blockY, blockZ,      sharedMemBytes, stream, kernelParams, extra);}
```

把"函数指针表 + 显式转发"作为唯一通路，意味着拦截器只需要替换表中那几个它关心的指针：

```
// 来源：libhrx/src/passthrough/README.md（示例约定）hip_function_table_t* hip_interceptor_init(hip_function_table_t* real) {    memcpy(&g_wrapper, real, sizeof(g_wrapper));    g_wrapper.hipMalloc = my_wrap_hipMalloc;   // 只替换关心的    g_wrapper.hipFree   = my_wrap_hipFree;    return &g_wrapper;}
```

这个设计的精妙之处在于"开发者编写拦截器的成本与所拦截函数数量成正比"，而不是 HIP API 总数。对比 LD_PRELOAD 一把梭再 `dlsym(RTLD_NEXT)` 的传统做法，HRX 把这些公式化的样板代码全部抽出来由 `libhip_intercept.so` 统一承担。

### 4.4 现成两个拦截器：日志 & buffer 追踪

> 仓库自带：

| 拦截器 | 作用 | 控制变量 |
| --- | --- | --- |
| `libhip_logging.so` | 给每次 HIP 调用打时间戳日志 | `HIP_LOG_FILE` / `HIP_LOG_LEVEL` |
| `libhip_buffer_tracer.so` | 跟踪 `hipMalloc/Free`，在 kernel launch 前后 dump buffer 内容或 FNV-1a 哈希 | `HIP_TRACE_DUMP` / `HIP_TRACE_KERNEL_FILTER` 等 |

`HIP_TRACE_DUMP=2` 只 dump 哈希、`HIP_TRACE_DUMP=1` dump 完整 hex——对调试"两次跑结果不一致"这种数值正确性问题极为有用。结合 `HIP_TRACE_KERNEL_FILTER=gemm`，你可以只追踪含 `gemm` 字串的内核，避免日志膨胀。

## 五、测试体系：从 CTS 到 YAML 执行清单

HRX 的测试基础设施值得单独一节，因为它体现了"运行时正确性如何被持续守护"。

### 5.1 CTS：分层覆盖运行时 API

`libhrx/cts/` 是 HRX 自带的 Conformance Test Suite，按 API 簇组织：lifecycle、status、device、allocator、memory、transfer、semaphore、stream、stream_ops、refcount、virtual_memory……README 表格明确列出"已测"和"待测"两栏，比如：

> `queue_dispatch / stream_dispatch` 还没测试，因为"需要编译好的 HSACO 或 VMFB kernel binary"。

这种诚实的进度披露在系统软件项目里其实并不常见，反而对外部贡献者非常友好——一眼能看到"哪些功能可以放心用，哪些还在路上"。

### 5.2 YAML 执行清单：声明式 e2e 测试

`build_tools/testing/README.md` 描述了一种用 YAML 描述工具调用的测试形式，runner 直接 `subprocess.run`，不走 shell：

```
# 来源：build_tools/testing/README.mdcases:-name:writethenrun    steps:      -write:          path:"{tmp}/input.txt"          text:"hello\n"      -run:          tool:fixture          args:["--input={tmp}/input.txt"]    stdout:      contains:        -"first literal"        -regex:"second .+ regex"      not_contains:        -"forbidden literal"    files:      -path:"{tmp}/module.vmfb"        non_empty:true
```

支持 `{srcdir}`、`{tmp}`、`{case}`、`{tool:name}` 等替换；`contains` 默认有序，可以加 `unordered:` 切换。这种"测试即声明"的风格让"新增一个回归用例"的成本极低——不需要写 Python、也不需要写 shell，只要修改一份 YAML。

## HRX 的意义：把 HIP 重塑成一组可替换的层

> 读完上述代码，再回头看 README 的那句 `This is not an official component within the ROCm stack`，会觉得意味深长。HRX 实际上在偷偷做几件事：

1. **把"HIP API 兼容性"和"运行时实现"解耦**：`binding/hip` 只是一层皮，可以随时换。
2. **把 IREE 的现代运行时语义引入 ROCm 用户视野**：timeline semaphore、VM 字节码、命令缓冲——这些是当下推理引擎共同的"通用骨架"。
3. **提供一条无侵入式的迁移路径**：通过 `LD_PRELOAD libamdhip64.so`，现有 HIP 应用可以零改动地切到 HRX；通过 `libhip_intercept.so`，又能把请求路由回真 ROCm 做对比、做 trace——这是任何运行时灰度发布的梦想场景。
4. **以可读的 C ABI 和精简的目录布局降低维护门槛**：相比传统 HIP 实现，HRX 的核心代码（`runtime.c` 加上几个 `*.c`）控制在万行级别，CI 与 Bazel↔CMake 工具链也以"可复用"为目标。

::: block-1
更具远景的解读是：当 AMD 把 NPU 和 CPU 也纳入"统一的低延迟运行时"愿景时，必然需要一个比传统 HIP 更现代、更可观测、更可替换的底座。HRX 把所有这些机会都放在了同一个 `libhrx.so` 里：

- 既能假装自己是 `libamdhip64.so`，也能露出 `hrx_*` 这套面向未来的 C ABI；
- 既能直接跑 HSACO，也能调度 VMFB 字节码；既能默默工作，也能在你设一个环境变量后把每一次 `hipMalloc` 的字节哈希打到日志里。
:::

对开发者而言，HRX 给出的不光只是一个全新框架，还是一组**可单独替换的层**。在 GPU 软件栈越发复杂的当下，"可替换"本身就是稀缺品。它意味着你下一次面对调度延迟、跨设备同步或调试可观测性的问题时，不必再把整个 HIP 翻一遍——你只需要找到 HRX 里那个对应的 `.c` 文件，重写一段，重新 `cmake --install`，应用甚至感知不到任何变化。

> 而这，恰恰是一个"基础设施级"项目最难得的特质。

参考资料

[1]

`ROCm/hrx-system`: *https://github.com/ROCm/hrx-system*

[2]

README: *https://github.com/ROCm/hrx-system/blob/main/README.md*

[3]

Python runner: *https://github.com/ROCm/hrx-system/blob/main/build_tools/testing/README.md*

[4]

IREE: *https://iree.dev/*

::: related-title
**相关推荐**
:::

::: related-links
- **[用 TileLang 在 AMD Radeon 上榨干 GPU 性能：RDNA4 GEMM 118.5 TFLOPS 持平 rocBLAS，Dequant达 93 TFLOPS](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901949&idx=1&sn=4520024a563e4c5821d9dcf98d83c8ac&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**

:::
