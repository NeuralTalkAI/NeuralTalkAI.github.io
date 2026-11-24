**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**模糊测试**、:mdnice-bold-italic[设备覆盖率]、**二进制插桩**、检测解耦、持久化执行

> 一个 CUDA 程序顺利返回，甚至打印了“测试通过”，显存访问就一定合法吗？当计算进入 GPU，主机端，也就是 CPU 一侧，能观察到的控制流往往只剩下内存申请、数据复制和内核启动。

实际决定结果的条件分支藏在设备代码里，越界访问也可能没有立即变成主机进程崩溃。**传统模糊测试因此同时遭遇两道障碍：不知道哪些输入值得继续探索，也无法仅凭退出状态判断错误是否发生。**

![文章插图](/articles/2026-10-26-nvidia-cufuzz-gpu-21d7868b/images/01.jpg)

- :mdnice-bold-italic[cuFuzz, a GPU-oriented coverage-guided fuzzer for userland CUDA applications. cuFuzz combines host-side and device-side coverage collection with sanitization to effectively discover bugs in CUDA programs.]
- https://github.com/NVlabs/cuFuzz

::: related-title
**相关推荐**
:::

::: related-links
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[约束求解赋能 GPU 模糊测试工具 GPU-Fuzz：挖掘深度学习框架 GPU 高危内存错误](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899663&idx=1&sn=41831f6311c438513d5c1732a2e67989&scene=21#wechat_redirect)**
- **[编译器测试新利器 MLIR-Forge：拼图解耦 IR 测试，让随机程序生成变得简单高效！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896753&idx=1&sn=cfa8a4376a322e46a41b38d668c7502f&scene=21#wechat_redirect)**

:::

> cuFuzz 把这两件事拆开处理：借助 NVBit 在设备二进制中插入覆盖率记录，将 GPU 执行信息合并到 AFL++ 的反馈位图；再通过 SAND，把选中的输入交给独立的错误检测通路。持久化模式进一步复用执行进程，摊薄 CUDA 初始化与插桩成本。

*补充一些信息：AFL++ 是 AFL（American Fuzzy Lop）的延续项目：它不断变异输入，并优先保留触发新代码路径的样本。覆盖率就是这类路径是否被执行的反馈，位图则用固定大小的字节数组记录这些反馈。*

*NVBit 是 NVIDIA 的动态二进制插桩工具，可以在 CUDA 设备指令执行前加入记录逻辑，无需逐个修改内核源码。SAND 是 AFL++ 中把覆盖率运行与 sanitizer（运行时错误检测器）分开执行的机制。*

![文章插图](/articles/2026-10-26-nvidia-cufuzz-gpu-21d7868b/images/02.jpg)

::: image-caption
上图是 cuFuzz 面向 GPU 库的模糊测试（Fuzzing）工作流程：首先初始种子送入基于 AFL++ 的模糊引擎，输入变异器生成新测试输入，传给模糊测试桩，调用被测库的 CPU 主机代码并触发 GPU 设备内核执行。分别通过 Clang、NVBit 采集主机与设备侧代码覆盖率，更新覆盖率位图；若发现新代码覆盖，该输入放回输入队列继续变异。测试输入同时送入启用消毒器的程序做错误检测，一旦捕获错误就记录到漏洞列表，无错误则丢弃样本，持续迭代挖掘 GPU 库缺陷。
:::

::: block-1
:mdnice-highlight[本文沿着输入、插桩、反馈、检查和状态重置这条链路，拆解其中最关键的实现]，同时说明覆盖率压缩、并发行为和公开评测材料的适用边界。
:::

## 本文目录

- 一、快速上手：先验证 GPU 反馈确实接通
- 二、架构总览：一份输入，分工明确的两条通路
  - 2.1 项目的核心落在哪里
  - 2.2 为什么覆盖率和检测结果缺一不可
- 三、设备插桩：把 GPU 基本块变成搜索路标
  - 3.1 在实际设备指令入口放置探针
  - 3.2 每个线程保存来路，活跃线程组由代表更新计数
- 四、反馈融合：把大规模执行计数压进一张小地图
  - 4.1 为什么要把设备计数压缩成标签
  - 4.2 如何把主机和设备反馈写进同一张位图
- 五、检测解耦：把昂贵检查花在选中的输入上
  - 5.1 谁值得进入检测器
  - 5.2 把设备错误翻译成 AFL 能识别的崩溃
- 六、持久化执行：复用进程，也必须划清输入边界
  - 6.1 两个空内核承担开始与结束通知
  - 6.2 历史状态残留，会制造虚假的第一条边
- 七、验证与边界：工具在运行，不等于反馈链路有效
  - 7.1 用正向证据排除“静默失效”
  - 7.2 覆盖率执行会改变被观察程序的行为
  - 7.3 示例可验证，论文评测仍有材料边界
- 结语：让反馈可用，比堆叠工具更重要

![文章插图](/articles/2026-10-26-nvidia-cufuzz-gpu-21d7868b/images/03.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：先验证 GPU 反馈确实接通

> 项目推荐 Docker，目前项目列出的已测试设备包括 A40、H100、B200，对应 `sm_86`、`sm_90`、`sm_100`。软件环境以 CUDA 13.4.1、NVBit 1.8、AFL++ 5.03c 为基准；宿主机还需准备 NVIDIA 驱动和 NVIDIA Container Toolkit。

```
# 来源：README.md；Git 获取方式结合 .gitmodules 调整git clone https://github.com/NVlabs/cuFuzz.gitcd cuFuzzgit submodule update --init --recursive  # A40 示例；其他设备替换 GPU_ARCH 和镜像标签sudo docker build --build-arg GPU_ARCH=sm_86 -t cufuzz:sm_86 .sudo docker run --gpus all -it --name cufuzz-demo cufuzz:sm_86 bash# 以下在容器内执行./verify_build.sh
```

*这里省略了将宿主机根目录挂入容器的参数，示例验证无需这种大范围挂载。容器也没有使用 `--rm`，退出后仍可保留检查结果。*

::: block-1
上面脚本示例中的 `verify_build.sh` 会编译示例、比较主机覆盖率与混合覆盖率、验证已知越界能够进入 AFL 的崩溃目录，再执行一轮以 200 次执行为目标的测试。这里的 oracle 指结果已知的验证用例：设备反馈必须出现，预先植入的越界也必须被发现。观察 `ORACLE_OK`、`SAND_OK` 及最终成功退出，比只看 AFL 界面是否出现更可靠；两个标记分别对应设备覆盖率检查和 SAND 崩溃保存检查通过。
:::

持续测试自有程序时，需要准备覆盖率版本、供检测器运行的原始版本，以及种子输入。种子输入是最初交给变异器的一组有效样本，测试桩（harness）负责把变异后的字节数据转换成被测程序能够接受的参数和 API 调用。完整命令见 README 的 Basic Mode[1]。

## 二、架构总览：一份输入，分工明确的两条通路

### 2.1 项目的核心落在哪里

> 语言统计中 Python、Shell 占比较高，但在线覆盖率收集的关键实现集中在 CUDA/C++ 与 C 文件中。

| 位置 | 主要职责 |
| --- | --- |
| `src/cufuzz_cov_nvbit/` | 设备插桩、边反馈记录、位图融合、资源清理 |
| `src/cufuzz_sand/` | 启动检测器，将指定错误退出码转成崩溃信号 |
| `Tools/AFLplusplus.patch` | 对接位图大小、检测策略、输入文件及预加载环境 |
| `targets/sampleApp/` | 带故意越界的普通与持久化示例 |
| `scripts/` | 实验启动、性能测量、覆盖率分析和错误归类 |

它的运行过程可以压缩成两条箭头：

- **输入变异 → 覆盖率目标 → 主机与设备反馈 → AFL++ 决定后续探索**
- **被选中的输入 → 独立检测目标 → 检测结果 → 崩溃样本保存**

AFL++ 继续承担变异、队列管理和调度；cuFuzz 补充 GPU 可观测性，并把设备错误接入原有反馈体系。

### 2.2 为什么覆盖率和检测结果缺一不可

> 下面示例中的 `vectorOp` 根据输入参数 `select` 选择 GPU 分支：取不同值时执行不同的向量运算，当 `select == 6` 时，代码访问 `B[i+1000]`。数组 `B` 只有 1000 个元素，因此这个分支会发生越界读取。

```
// 这段示例同时包含两种现象：输入能够改变设备执行路径，其中一条路径还含有内存错误。// 代码中的 `__global__` 表示该函数由主机启动、在 GPU 上执行；// `blockIdx`、`blockDim` 和 `threadIdx` 分别给出线程块编号、// 块内线程数和线程在块内的位置，三者组合出当前线程处理的数据下标 `i`。__global__ void vectorOp(const float *A, const float *B, float *C,                          int numElements, int select) {int i = blockDim.x * blockIdx.x + threadIdx.x;if (i < numElements) {   if (select == 1)      C[i] = A[i] + B[i];     // Add     elseif (select == 2)      C[i] = A[i] + 2*B[i];   // Double B and Add      //C[i] = A[i] + B[1003] + 10.0f;    elseif (select == 3)      C[i] = 2*A[i] + B[i];   // Double A and Add    elseif (select == 4)      C[i] = A[i] - B[i];     // Subtract     elseif (select == 5)      C[i] = 2*A[i] - B[i];   // Double A and subtract    elseif (select == 6)      C[i] = A[i] + B[i+1000];// Access OOB memory!    elseif (select == 7)      C[i] = A[i] - 2*B[i];   // Double B and subtract    else      C[i] = A[i] + B[i];     // Add            }}
```

> 这段代码把两个问题连在了一起：怎样让模糊测试器看见不同的 GPU 分支，又怎样确认某个分支中的内存访问已经越界。

- 先看覆盖率的作用。无论 `select` 取什么值，主机端都沿着读取输入、申请内存、复制数据、启动内核的流程执行，真正的分支选择发生在 GPU 内核内部。:mdnice-bold-italic[只收集主机覆盖率时，多个不同的 `select` 值可能留下相同的主机反馈，模糊测试器也就难以区分它们各自触发了哪条设备路径。加入设备覆盖率后，新触发的 GPU 分支才能转化为反馈，引导 AFL++ 保留并继续变异对应输入。]
- 再看检测器的作用。覆盖率只能说明某条路径已经执行，不能判断路径中的内存访问是否合法。即使 `select == 6` 的分支已经覆盖，越界读取也未必立即触发硬件异常，主机进程仍可能正常结束。:mdnice-bold-italic[示例主机端打印的 `Test PASSED` 也只是一条输出，没有验证全部计算结果或显存访问。因此，还需要 Compute Sanitizer 等检测器检查同一份输入，确认执行过程中是否发生非法访问。]

上面示例代码[2]由此对应项目的两个目标：**让搜索看见设备行为，让检查识别非法行为。**

## 三、设备插桩：把 GPU 基本块变成搜索路标

### 3.1 在实际设备指令入口放置探针

> **`instrument_function_if_needed` 是 cuFuzz 覆盖率工具中负责给设备代码插入探针的函数。** NVBit 捕获到内核启动事件后，cuFuzz 会在内核实际执行前调用该函数，并传入 CUDA 上下文 `ctx` 和待启动的内核 `func`。

`instrument_function_if_needed` 的处理过程如下：

::: block-1
首先，它获取待启动内核以及该内核调用的设备函数，并逐一检查 `kernels` 登记表。已经登记的函数会直接跳过，避免重复插桩。

对于尚未处理的函数，它读取控制流图（CFG）。控制流图用节点和有向边描述指令之间的执行与跳转关系，图中的基本块则是一段顺序执行的指令，通常从入口进入、从出口离开。基本块的入口因此适合作为覆盖率探针的插入位置。

随后，它遍历这些基本块，在每个基本块的第一条指令之前插入 `record_coverage_edge_count` 调用，并把设备计数数组、线程上一基本块状态数组的地址作为参数传给探针。同时，函数地址、基本块编号和指令信息会被分别保存到 `kernels` 和 `bbs` 中，用于识别已处理函数及后续分析。

到这里，插桩准备已经完成。`instrument_function_if_needed` 本身不统计覆盖率；内核真正运行时，插入的 `record_coverage_edge_count` 才会随设备代码执行，将边覆盖事件累加到设备侧的 `exec_cov_bb` 数组。随后，cuFuzz 再对这些计数进行量化，并与主机覆盖率合并，形成 AFL++ 使用的反馈位图。
:::

> 下面保留核心语句，省略外围循环：

```
// 来源：src/cufuzz_cov_nvbit/cufuzz_cov.cu（节选）Instr *i = bb->instrs[0];nvbit_insert_call(i, "record_coverage_edge_count", IPOINT_BEFORE);// 结合函数名哈希和基本块序号生成标识XXH64_hash_t fname_hash = XXH64(name.c_str(), name.length(), HASH_CONST);unsigned int bb_id_rand = (int)(((fname_hash << 5) + fname_hash) + bb_id);nvbit_add_call_arg_const_val32(i, bb_id_rand);nvbit_add_call_arg_const_val64(i, (uint64_t)exec_cov_bb);nvbit_add_call_arg_const_val64(i, (uint64_t)prev_cov_bb);
```

这段代码对应上述插桩过程。`IPOINT_BEFORE` 指定在原基本块的第一条指令之前执行探针。`XXH64` 使用 `HASH_CONST` 作为种子，将函数名散列为 64 位值；代码再把这个哈希值与基本块序号 `bb_id` 组合，生成当前基本块的标识 `bb_id_rand`。这里生成的是基本块标识，后续 `record_coverage_edge_count` 才会将它映射到具体的位图槽位。最后三条 `nvbit_add_call_arg_*` 语句依次传入基本块标识、设备计数数组地址和上一基本块状态数组地址。

::: block-1
**整套插桩作用于 NVBit 能够解析的设备二进制，因此无需为每个内核手工修改源码，覆盖范围则取决于 NVBit 能否解析相应设备函数。** 主机侧覆盖率使用另一条收集路径：示例构建时通过 AFL++ 提供的 `afl-clang-fast++` 编译器包装器，在主机代码编译阶段加入覆盖率记录。
:::

到这里，每个基本块都有了标识，但仅仅知道进入过哪些基本块，还不足以区分执行路径。例如，一次执行沿 A→B 到达 B，另一次执行沿 C→B 到达 B，两者都经过了 B，来路却不同。接下来插入的 `record_coverage_edge_count`，就是要把这种前后连接关系也记录下来。

### 3.2 每个线程保存来路，活跃线程组由代表更新计数

> 上一节插入的 `record_coverage_edge_count` 会在每次进入基本块时执行。这里要记录的“边”，指线程从上一个基本块跳转到当前基本块的执行关系。函数不会保存完整路径，而是把这种前后关系映射到覆盖率位图中的一个数组位置，并更新该位置的计数。为了减少大量 GPU 线程同时更新计数器带来的开销，代码还会从当前活跃线程中选出一个代表，由它执行计数更新。

理解代表线程的选择过程，需要先了解 CUDA 的线程组织方式。CUDA 通常以 warp 为一组调度线程，lane 表示线程在 warp 内的位置。发生分支时，同一 warp 中可能只有部分线程仍然活跃。代码通过 `__activemask()` 取得当前活跃线程的范围，再由 `__ballot_sync()` 汇总成 `active_mask` 位掩码，最后用 `__ffs()` 找到编号最小的活跃 lane，作为本次计数更新的代表。

```
// 来源：src/cufuzz_cov_nvbit/inject_funcs.cu（节选，省略 tid 计算）constint active_mask = __ballot_sync(__activemask(), 1);constint laneid = get_laneid();constint first_laneid = __ffs(active_mask) - 1;uint64_t* prev_cov_bb = (uint64_t*)p_prev_cov_bb;int new_bb_id = (MAP_SIZE/2)             + ((prev_cov_bb[tid] ^ bb_id) % (MAP_SIZE/2));prev_cov_bb[tid] = bb_id >> 1;// 只有当前活跃掩码中的首个 lane 执行原子加if (first_laneid == laneid) {    uint32_t* exec_cov_bb = (uint32_t*)p_exec_cov_bb;    atomicAdd((unsignedint*)&exec_cov_bb[new_bb_id], 1);}
```

选出代表线程后，函数开始计算当前边应该写入位图的哪个位置。每个活跃线程都根据自己的全局索引 `tid`，从 `prev_cov_bb[tid]` 读取上一基本块的状态，再与当前基本块标识 `bb_id` 异或。对异或结果取模并加上偏移量后，就得到了 `new_bb_id`。虽然这个变量的名字像是“新的基本块编号”，但它实际表示本次边覆盖记录在 `exec_cov_bb` 数组中的下标。

随后，每个活跃线程都会更新自己的 `prev_cov_bb[tid]`，为下一次记录保存来路。计数更新则只由代表线程执行：`atomicAdd` 将 `exec_cov_bb[new_bb_id]` 增加 1。原子加可以避免多个执行单元同时修改同一计数器时互相覆盖；每组活跃线程只由代表执行一次，也减少了原子操作的数量。

::: block-1
这里的计数需要谨慎解读：每次只有代表线程对自己计算出的 `new_bb_id` 执行加 1，其他活跃线程计算出的数组位置不会分别更新。因此，该计数既不是所有 CUDA 线程经过某条边的总次数，也可能遗漏同一活跃线程组内的不同来路。源码虽然通过 `__popc(active_mask)` 计算了活跃线程数量，但这个数量并未参与原子加。
:::

上面倒数第二个代码块的最后两行的位运算决定了边如何被压缩进位图。`prev_cov_bb[tid] = bb_id >> 1` 将当前基本块标识右移一位后保存，下一次再与新的基本块标识异或，用来帮助区分 A→B 和 B→A 这类方向不同的边。取模则把大量可能的边压缩到有限数量的数组位置中，再加上 `MAP_SIZE/2` 的偏移，使设备覆盖率使用 65536 项位图的后半区，即下标 32768 至 65535。**这种压缩节省了存储空间，也让不同的边可能映射到同一个数组位置，部分路径差异由此无法区分。**

## 四、反馈融合：把大规模执行计数压进一张小地图

### 4.1 为什么要把设备计数压缩成标签

> 前面提到，GPU 探针会把边覆盖事件累加到 `exec_cov_bb` 中。它是 cuFuzz 分配的设备覆盖率计数数组，每个数组元素保存一个位图位置对应的 32 位计数。**AFL++ 使用的 `trace_bits` 则是另一种结构：它是一张字节数组形式的覆盖率位图，每个位置只占用 8 位。主机侧和设备侧的反馈要合并，就必须先把 `exec_cov_bb` 中的 32 位计数压缩成单字节。**

直接截取低八位会带来问题。例如，原始计数达到 256 时，低八位会变成 0，一次实际发生过的覆盖可能因此被写成“没有命中”。**cuFuzz 使用 `counterToByte` 函数处理这个转换。它接收一个原始计数，按照计数所在的区间输出一个 0～7 的标签：**

| 原始执行计数（次） | 压缩后的标签值 |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 ～ 511 | 3 |
| 512 ～ 4095 | 4 |
| 4096 ～ 16383 | 5 |
| 16384 ～ 65535 | 6 |
| 65536 及以上 | 7 |

这张表要按“原始计数落入哪个区间”来读。例如，某个位置的原始计数是 600，位于 512～4095 这一档，`counterToByte` 就返回 4。这里的 4 只是该计数区间的标签，不能理解为执行了 4 次，也不表示发现了 4 条边。

`quantize_device_map` 负责对整张设备计数数组执行这项转换。它遍历 `exec_cov_bb` 的每个元素，调用 `counterToByte` 得到标签，再把标签写入 `exec_cov_bb_quantized` 的相同位置。后者是 cuFuzz 准备的单字节中间数组，专门保存压缩后的设备覆盖率，方便接下来与 AFL++ 的字节位图合并。

分桶保留了执行规模的粗粒度变化，也放弃了同一区间内的细节。3 次和 500 次都会得到标签 3，512 次和 4000 次都会得到标签 4，这些输入在设备计数上已经不能仅凭标签区分。

标签与主机覆盖率合并后，还会经过 AFL++ 自己的计数分类。在当前固定的 AFL++ 版本中，字节值 2 和 3 都归为 2，4～7 都归为 4。因此，表中的八个标签描述的是 cuFuzz 的量化结果，不能直接当成 AFL++ 最终用于新颖性判断的八种独立计数特征。

### 4.2 如何把主机和设备反馈写进同一张位图

> 经过量化，设备覆盖率已经有了单字节表示，但它仍保存在 cuFuzz 自己的 `exec_cov_bb_quantized` 数组中。AFL++ 判断输入是否带来新覆盖时，读取的是 `trace_bits`。因此，接下来要做的事情很明确：找到 AFL++ 的这张位图，把量化后的设备反馈合并进去。

这里还隔着一道进程边界。AFL++ 的模糊测试进程创建并管理共享位图，NVBit 工具则随被测程序运行，两个进程不能直接读取对方的普通内存。Linux 共享内存用来打通这条通路。AFL++ 通过环境变量 `__AFL_SHM_ID` 传出共享内存的编号，NVBit 工具读取编号后调用 `shmat`，把这段内存映射到自己的地址空间。映射完成后，工具中的 `trace_bits` 指针就能访问 AFL++ 使用的那张位图。

设备侧数组还涉及主机与 GPU 之间的数据交换。`exec_cov_bb`、`exec_cov_bb_quantized` 等数组通过 `cudaMallocManaged` 分配统一内存，主机和设备可以使用同一指针访问。GPU 执行探针时更新原始计数，主机随后读取这些计数，完成量化和合并。

准备工作完成后，`merge_coverage_byte_quant` 开始逐位置合并数据。它读取 `trace_bits` 中的主机字节，再读取 `exec_cov_bb_quantized` 中同一位置的设备标签，将两者相加后写入临时主机数组 `merged_cov`。所有位置处理完后，代码再把 `merged_cov` 整体复制回 `trace_bits`。

::: block-1
例如，某个位置的主机字节为 1，设备侧原始计数为 600。原始计数先被量化为标签 4，合并时计算 1＋4，最终向该位置写入 5。设备执行信息由此进入 AFL++ 后续读取的覆盖率反馈。相关实现集中在量化与合并函数[3]。
:::

合并结果仍然只有一个字节，所以代码只保留相加结果的低八位。如果主机字节和设备标签之和为 256，写入结果就会回绕为 0。为了避免这次已有覆盖被误写成“没有命中”，代码会把这种“原始和非零、低八位却为零”的结果改成 1。**这项处理只能防止命中直接消失，其他回绕造成的计数差异仍可能丢失。**

> 两种运行模式在不同时间执行上述合并。普通模式会等到最后一个 CUDA 上下文终止时提交设备反馈。CUDA 上下文保存驱动管理 GPU 执行所需的状态，可以理解为程序使用 GPU 的运行环境。持久化模式会复用进程和 CUDA 上下文，不能一直等到上下文退出，因此需要在每份输入处理结束时，通过专门的轮次结束通知触发量化与合并。

最后还有两条边界需要注意。设备覆盖率使用位图后半区，并不能保证主机覆盖率只写入前半区；如果主机插桩也写入后半区，两侧数据仍会在相同下标上相加。与此同时，`CUFUZZ_MAP_SIZE` 调整的是 AFL++ 侧的位图参数，NVBit 工具内部的 `MAP_SIZE` 仍固定为 65536，也就是整个位图包含 65536 个字节位置。修改环境变量不会同步改变设备侧数组的大小和循环遍历范围。

## 五、检测解耦：把昂贵检查花在选中的输入上

### 5.1 谁值得进入检测器

> 通过 `-w` 接入的包装器可以调用 Compute Sanitizer 的 `memcheck`、`initcheck`、`racecheck`，也提供 ASan 接口。`memcheck` 检查非法显存访问，`initcheck` 检查未初始化设备全局内存的读取，`racecheck` 检查共享内存中的数据竞争；ASan（AddressSanitizer）主要检查主机侧的越界、释放后使用等内存错误。

在覆盖率目标正常结束的输入处理路径中，SAND 根据策略决定是否送检。结合固定版本的 AFL++ 实现[4]，四种模式应这样理解：

- `all_trace`：cuFuzz 补丁增加的无覆盖率筛选送检分支。
- `simplify_trace`：去掉命中次数信息，对命中与否构成的位图做签名去重。
- `unique_trace`：对计数分类后的位图做签名去重。
- `coverage_increase`：由 AFL 的新颖性判定决定是否送检，包括某些计数特征变化。

README 推荐 `simplify_trace`。这里所谓“轨迹”是位图抽象，并未保存完整有序指令序列。

**相同反馈也可能对应不同数据值和不同错误结果**。因此，基于反馈去重送检属于性能与检测机会之间的取舍，不能推出“不漏报”的普遍保证。

### 5.2 把设备错误翻译成 AFL 能识别的崩溃

> 包装器执行检测器，等待其退出；当 Compute Sanitizer 返回约定的 99，包装器调用 `abort()`：

```
// 来源：src/cufuzz_sand/wrapper_san.c（Compute Sanitizer 分支节选）int status;waitpid(pid, &status, 0);if (WIFEXITED(status)) {    int exit_code = WEXITSTATUS(status);    if (exit_code == 99) {        // 将检测器报告转换为主机进程崩溃信号        abort();    }}
```

`waitpid` 让包装器等待检测子进程结束，`WIFEXITED` 判断它是否正常退出，`WEXITSTATUS` 再取出退出码。`abort()` 会主动触发异常终止信号，AFL 因而能按崩溃处理并保存当前输入。**这段代码很短，却连接了两套错误模型：检测器报告的设备违规，变成 AFL 可以保存的崩溃样本。**

此外，AFL++ 补丁 AFLplusplus.patch [5]还处理了两个容易被忽略的问题：

- 在检测进程启动前移除覆盖率预加载环境，避免 NVBit 跟随进入独立检查通路。
- 显式写入 `@@` 输入文件。`@@` 是 AFL 命令行中的输入文件占位符，运行时会被当前样本路径替换；这一步避免主通路使用共享内存输入时，检测程序读不到当前样本。

系统能否真正工作，往往就取决于这些进程边界上的细节。

## 六、持久化执行：复用进程，也必须划清输入边界

### 6.1 两个空内核承担开始与结束通知

> 反复创建进程，会重复付出 CUDA 初始化和设备插桩等成本。示例通过 `__AFL_LOOP(100)` 在一个进程中处理多份输入，再用空内核通知 NVBit 每轮边界。

`__AFL_LOOP(100)` 是 AFL++ 持久化模式的循环宏，让当前进程最多连续处理 100 份输入，实际次数仍受 AFL 调度控制。下方为流程伪代码：

```
// 来源：targets/sampleApp/sampleApp_persistent.cu（流程伪代码）while (__AFL_LOOP(100)) {    cufuzz_notification_kernel<<<1, 1>>>(1234);    cudaDeviceSynchronize();    // 读取输入，执行本轮 CUDA 工作，释放本轮资源    RunOneInput();    cufuzz_notification_kernel<<<1, 1>>>(5678);    cudaDeviceSynchronize();}
```

伪代码中的 `<<<1, 1>>>` 表示只启动一个线程块、块内只有一个线程，`cudaDeviceSynchronize()` 则等待设备工作完成，确保重置或合并发生在正确的轮次边界。1234 和 5678 是双方约定的魔数，也就是仅用于识别协议事件的固定常量；空内核本身不计算业务数据，只负责让 NVBit 在内核启动回调中看见通知。工具检查通知内核名称、启动维度及魔数：

- 收到 1234：清空主机反馈位图，并重置每线程上一基本块状态。
- 收到 5678：量化设备反馈、与主机反馈合并，再清空设备计数，准备下一轮。

这套协议让“进程生命周期”与“输入生命周期”分开，从而允许进程复用而反馈仍按输入结算。

### 6.2 历史状态残留，会制造虚假的第一条边

> 假设上一输入最后停在 A，下一输入首先进入 B。如果 `prev_cov_bb` 未清零，记录器就会把 A→B 当成下一输入的起始边。

于是，同一输入的反馈可能随着前一个输入改变，影响稳定性与去重。当前版本的 `reset_prev_cov_bb()` 在轮次开始处同步设备并清零数组；CHANGELOG[6]明确记录了这项修复。

重置也有成本：数组包含 3200 万个 `uint64_t`，仅这一项约占 256 MB，即约 244 MiB，每轮需要清零。

此外，示例仍在循环内申请和释放数组，并写输入文件、调用 `sync()` 请求将系统缓冲区中的数据写回存储。持久化包装器也不能据名称推断为“检测器常驻复用”：当前实现仍会启动检测器执行目标。

所以，持久化提供了摊薄启动成本的机制，具体吞吐收益必须结合目标与测量结果判断。

## 七、验证与边界：工具在运行，不等于反馈链路有效

### 7.1 用正向证据排除“静默失效”

> 项目代码中验证脚本 verify_build.sh[7] 的设计很有借鉴价值。它要求：

1. NVBit 日志中确实出现内核启动。
2. 设备量化位图中确实存在非零字节。
3. 混合反馈确实区别于无 NVBit 的主机对照组。
4. 已知越界输入确实产生至少一个保存的崩溃。

验证脚本还用 `afl-showmap` 单独运行一份输入并导出覆盖率位图，用它做主机对照与 NVBit 对照，无需先跑完整模糊测试。缺少任一条件，脚本失败退出。这比“命令返回零”更接近可观测性的端到端验证。**比如 `nvdisasm` 不在路径中，就可能导致看起来运行正常、实际上没有设备反馈的测试。** 其中，`nvdisasm` 是 CUDA 二进制反汇编工具，NVBit 分析设备指令时依赖它。

### 7.2 覆盖率执行会改变被观察程序的行为

> grid 是一次内核启动所包含的全部线程块，stream 是设备上按顺序执行工作的队列，不同 stream 之间可以重叠执行。工具默认启用 `SERIALIZE_GRIDS`，在所处理的内核启动回调退出阶段调用 `cudaDeviceSynchronize()`，等待当前设备工作完成。这有利于反馈收集，却可能改变原有跨内核、跨流重叠执行。

因此，覆盖率通路中的时序表现不能直接代表原始程序；设备边反馈也无法替代对全部并发交错的探索。独立检测通路的价值，正体现在它能使用不同观察机制检查同一输入。

### 7.3 示例可验证，论文评测仍有材料边界

> Python 脚本负责实验外围工作：`benchmark_script.py` 使用 hyperfine 测时。hyperfine 是命令行基准测试工具，会预热并多次运行同一命令，再汇总耗时统计；脚本这里设置为预热 2 次、测量 10 次。`analyze-crashes.py` 根据检测器日志提取错误签名，绘图脚本展示覆盖率随时间或执行次数的变化。

但当前公开树没有 `data/` 目录，而 `build_workloads.bash`[8]依赖其中的 harness、种子和配置。部分分析脚本还引用当前构建不生成的旧覆盖率库路径。

因此，仓库具备明确的示例验证入口，但不能据此宣称论文全部实验可直接一键复现。错误签名数量、位图槽位数量，也分别需要与真实独立缺陷数量、真实控制流边数量区分。

## 结语：让反馈可用，比堆叠工具更重要

> cuFuzz 的核心贡献，可以归纳为三层连接：把设备二进制行为连接到覆盖率，把覆盖率连接到输入筛选，再把检测器结果连接到崩溃样本保存。

它没有消除模糊测试的全部盲区。哈希会压缩信息，选择性检查会舍弃部分机会，持久化会引入状态管理责任，插桩也会影响执行时序。但这些取舍在源码中都有具体落点，能够被检查、解释和测量。

::: block-1
对于 CUDA 算子、图像处理库和 AI 系统底层组件，这提供了一条务实路径：**先让隐藏的执行差异成为搜索信号，再让专业检测器判断这些差异是否触碰安全边界。** 当 GPU 从计算黑箱变成能够反馈内部行为的测试对象，缺陷搜索才真正获得了方向。
:::

参考资料

[1]

README 的 Basic Mode: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/README.md#L242-L291*

[2]

这段示例代码: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/targets/sampleApp/sampleApp.cu#L33-L56*

[3]

量化与合并函数: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/src/cufuzz_cov_nvbit/cufuzz_cov.cu#L279-L331*

[4]

AFL++ 实现: *https://github.com/AFLplusplus/AFLplusplus/blob/dbaf11913c1b2702dee5b4d3dcfffd52f1defe50/src/afl-fuzz-bitmap.c#L991-L1042*

[5]

AFL++ 补丁: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/Tools/AFLplusplus.patch*

[6]

CHANGELOG: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/CHANGELOG.md#L25-L31*

[7]

验证脚本: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/verify_build.sh#L104-L187*

[8]

`build_workloads.bash`: *https://github.com/NVlabs/cuFuzz/blob/bb3ba27f3dab775738c8f8a9ec9a78969dd95312/scripts/build_workloads.bash#L116-L169*

::: related-title
**相关推荐**
:::

::: related-links
- **[PROPILOT：面向 AI 编译器的张量不变量编译测试新范式，削减 49%冗余测试用例，摆脱 LLM 测试幻觉！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902734&idx=1&sn=9867a1dc58b8d7025c8a0a9e4a932300&scene=21#wechat_redirect)**
- **[72 卡 GB200 的 NVLink 互联怎么测？NVIDIA 开源 MNNVL fabric 基准测试工具 nvloom](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904392&idx=1&sn=98f83575f3df05231538d9d678d82993&scene=21#wechat_redirect)**
- **[对标 A100：拆解摩尔线程开源的 GPU 驱动级基准测试 gpu-compute-driver-bench](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901875&idx=1&sn=3db0fb835b96a96286fc1e65ef6492e2&scene=21#wechat_redirect)**

:::
