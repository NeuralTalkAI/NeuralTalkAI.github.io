**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：ONNX Runtime、**MLIR**、ROCm、**Execution Provider**、GPU 推理

> 在深度学习推理系统里，Execution Provider 往往被理解成“把算子派发到某个硬件后端”的适配层。hip-ep 展示了另一种更激进的形态：它把 EP 做成了一条真正的编译执行链。

![文章插图](/articles/2026-09-01-amd-hip-ep-mlir-onnx-7b93d8be/images/01.jpg)

- :mdnice-bold-italic[ROCm hip-ep: ROCm Execution Provider for ONNX Runtime on AMD GPUs. Compiles ONNX models through MLIR to LLVM bitcode, JIT-loads them at session creation, and dispatches to hipDNN, MIOpen, hipBLASLt, and custom HIP kernels]
- https://github.com/ROCm/hip-ep

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 把 CDNA5 的"底牌"摊开了：832 页 ISA 手册全公开，MI455X 的每个细节都写在里头](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904248&idx=1&sn=b2b81e0229f104f7dd3d3a9cb3d31dcf&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**

:::

> :mdnice-bold-italic[hip‑ep 是面向 AMD GPU、基于 MLIR 实现的 ONNX Runtime 执行 Execution Provider，打破传统 EP 仅作为算子派发适配层的定位，将 EP 升级为完整模型编译执行链，打通 ONNX Runtime 与 ROCm 推理生态。] 模型经由 MorphiZen 图改写框架接入，通过 MLIR 编译管线逐级将 ONNX 算子方言 lowering 至 HIP 方言、memref 内存表示，最终输出可移植 LLVM 位码，在推理会话创建阶段进程内 JIT 加载执行，也支持可选编译为.so/.dll 原生库文件，底层调用 hipDNN、hipBLASLt 以及自定义 HIP 内核完成计算。

项目完整覆盖从源码构建、算子转换、内存管理到运行时调度全链路工程问题。

编译端依靠 CompilerDriver 统筹整套 Pass 流水线，流水线执行顺序直接决定系统正确性；算子转换采用 Rewrite Pattern 模式，原生支持大模型 GQA、KV‑Cache、量化算子，同时实现常量外置，解决大模型权重体积与加载瓶颈。

内存层面设计按需扩容 GPU 内存池减少频繁显存分配释放，输出张量通过 EP 回调机制对接 ONNX Runtime 张量体系。运行时 RuntimeState 统一管理会话生命周期内全部 GPU 资源，提供 LLVM bitcode JIT 与原生库两种交付模式。同时配置静态插件插槽、IR 调试环境变量，搭配分层测试体系，支持无 GPU 模拟环境开展验证测试。

::: block-1
hip‑ep 填补了 ONNX 模型到 AMD GPU 可执行程序之间的关键缺口，:mdnice-highlight[证明面向动态 shape 大模型推理，Execution Provider 应当具备编译器能力]，为异构硬件推理框架设计提供重要参考。
:::

## 本文目录

- 一、快速上手：先跑起来，再理解它
  - 1.1 Linux 原生构建
  - 1.2 设置运行环境并执行模型
- 二、项目全貌：它其实是一个嵌入式 AI 编译器
  - 2.1 顶层目录反映的职责边界
  - 2.2 构建系统的工程取向
- 三、编译主线：从 ONNX 方言到 LLVM 位码
  - 3.1 CompilerDriver 是总指挥
  - 3.2 Pipeline 的顺序是正确性的核心
- 四、ONNX 到 HIP：算子转换不是逐个翻译这么简单
  - 4.1 ConvertOnnxToHip 的核心策略
  - 4.2 常量处理体现了大模型意识
- 五、内存：动态 shape 下的 GPU 池化与输出分配
  - 5.1 临时 buffer 进入 grow-on-demand 池
  - 5.2 输出由 EP 回调分配
- 六、运行时：会话状态是模型执行的载体
  - 6.1 RuntimeState 管住所有跨调用资源
  - 6.2 bitcode JIT 与 native artifact 的取舍
- 七、插件机制：给编译管线预留可控扩展点
  - 7.1 静态插件避免平台差异
  - 7.2 环境变量提供研究级逃生门
- 八、工具与测试：它如何证明自己可运行
  - 8.1 命令行工具覆盖开发闭环
  - 8.2 测试分层覆盖编译与运行
- 总结：把 EP 从适配层推进到编译层

![文章插图](/articles/2026-09-01-amd-hip-ep-mlir-onnx-7b93d8be/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：先跑起来，再理解它

### 1.1 Linux 原生构建

> 项目 README 将 Linux 入口分成三类：源码原生构建、Docker 构建、下载预构建包。开发者最短路径如下，更多环境细节可参考 `docs/quick_start_linux.md`[1]。

```
git clone https://github.com/ROCm/hip-ep.gitcd hip-ep# 自动检测 AMD GPU 架构；若无 GPU，会退到 mock runtimepython3 build.py
```

构建后默认产物位于工作区相邻目录的 `install/` 下，核心工具包括：

```
install/bin/hip-onnx-runnerinstall/bin/hip-compilerinstall/bin/hip-mlir-optinstall/bin/hip-testinstall/lib/libhipgpu.so
```

若希望隔离环境，可走 Docker：

```
./docker/run.sh image./docker/run.sh build# 指定 GPU 架构示例HIP_ARCHITECTURES=gfx942 ./docker/run.sh build
```

### 1.2 设置运行环境并执行模型

> 源码构建版本依赖 ONNX Runtime 前缀与 TheRock ROCm SDK 路径；预构建包会打包更多运行库，但 TheRock 仍需在运行时可见。最简示意如下：

```
export ROOT="$WORKSPACE/install"export THEROCK_DIST="$WORKSPACE/therock-dist"export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"export LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"export PATH="$ROOT/bin:$PATH"$ROOT/bin/hip-onnx-runner --help$ROOT/bin/hip-onnx-runner -m /path/to/model.onnx -i gen_inputs
```

若是 LLM 模型，随机输入可能越过词表范围，项目建议先用输入生成脚本生成 `gen_inputs/`，再传给 runner。完整命令与 benchmark 参数见 `docs/quick_start_linux.md`[2]。

## 二、项目全貌：它其实是一个嵌入式 AI 编译器

### 2.1 顶层目录反映的职责边界

> hip-ep 的顶层结构很清晰：`lib/` 是编译器与运行时主体，`include/hip/` 暴露方言、pass、C API 与 ABI 头文件，`backend-mlir-compiler/` 承接 MorphiZen/ONNX Runtime 集成，`morphizen/` 是图改写框架，`tools/` 提供命令行入口，`schemas/` 定义 FlatBuffers 元数据，`docs/` 写下许多架构决策。

可以把它视作四层：

```
ONNX Runtime / MorphiZen        │        ▼ONNX MLIR 入口与图级改写        │        ▼HIP 方言 + MLIR Pass Pipeline        │        ▼LLVM bitcode / native artifact + Runtime ABI        │        ▼hipDNN / hipBLASLt / custom HIP kernels / AMD GPU
```

这条链路的关键在于：ONNX 图不再只是一组待派发的节点，它**会被整体分析、重写、bufferize、规划内存，并生成带 C ABI 的模型级执行入口。**

### 2.2 构建系统的工程取向

> `CMakeLists.txt` 透露了项目的现实约束：Linux 使用 CMake HIP 支持，Windows 走自定义 hipcc 调用以兼容 MSVC；依赖统一由 `cmake/deps.cmake` 处理，包括 LLVM/MLIR/LLD、flatbuffers、protobuf、ONNX Runtime、TheRock ROCm SDK。`build.py` 则提供跨平台驱动，负责子模块检查、工具链检查、GPU 架构检测、CMake configure/build/install，以及 GPU-free 测试。

```
# 来源：build.pydef detect_hip_arch():    # Linux 从 /sys/class/kfd 读取 gfx_target_version    # Windows 通过驱动提供的 HIP runtime 读取 gcnArchName    return _detect_hip_arch_windows() if IS_WINDOWS else _detect_hip_arch_linux()def generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock):    cmd = ["cmake", "-S", str(REPO), "-B", str(build_dir), "-G", args.cmake_generator]    cmd += [        f"-DCMAKE_BUILD_TYPE={args.config}",        f"-DCMAKE_INSTALL_PREFIX={args.install_dir}",        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",    ]    if mock:        cmd.append("-DBUILD_MOCK_RUNTIME=ON")    else:        cmd.append("-DBUILD_MOCK_RUNTIME=OFF")        cmd.append(f"-DHIP_ARCHITECTURES={hip_arch}")    run_subprocess(cmd)
```

这段代码很有代表性：hip-ep 没有假设开发机已经装好完整 LLVM，也没有强制每台机器都有 GPU。**它允许 mock runtime，使 MLIR pass、插件注册、输出分配等逻辑在无 GPU 环境中接受测试。**

## 三、编译主线：从 ONNX 方言到 LLVM 位码

### 3.1 CompilerDriver 是总指挥

> :mdnice-bold-italic[`lib/Compiler/CompilerDriver.cpp` 是整个编译过程的主入口。它做了几件非常关键的事：注册所有 pass，触发静态插件注册，加载 HIP 方言与 LLVM dialect translation，解析 MLIR，运行 pass pipeline，再翻译成 LLVM IR]。输出路径有两种：默认生成 LLVM bitcode；若选择 native 模式，则合并 runtime bitcode，编译成 object，再链接成 `.so/.dll`。

```
// 来源：lib/Compiler/CompilerDriver.cppbool CompilerDriver::compileImpl(ModuleOp module,                                 const std::string &output_path,                                 const CompilationOptionsT &options,                                 std::string &error_message) {  if (!runMLIRPasses(module, options, error_message))    return false;  LLVMContext llvmContext;  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message);  if (!llvmModule)    return false;  const bool native = (options.output_mode == OutputMode::NATIVE);  if (native)    linkRuntime(llvmModule.get(), error_message);  optimizeLLVMIR(llvmModule.get(), options.opt_level);  if (!native) {    return emitLlvmIr(llvmModule.get(), output_path, error_message);  }  compileToObject(llvmModule.get(), obj_path, error_message);  linkToDLL(obj_path, output_path, libraries, library_paths,            export_symbols, error_message);  return true;}
```

**这里的“模型编译”已经超出传统 EP 的范畴**：它生成的是带 `inference_init`、`inference_compute`、`inference_cleanup` 等入口的模型程序。:mdnice-bold-italic[ONNX Runtime 创建 session 时，EP 可以把这个模型程序装载起来，在后续 Compute 中直接调用生成的入口。]

### 3.2 Pipeline 的顺序是正确性的核心

> hip-ep 的 pass 顺序写在 `lib/Dialect/Transforms/Pipelines.cpp`，文档 `docs/pipeline_pass_menu.md` 也给出稳定的文本名称。**默认管线分为 ONNX→HIP 与 HIP→LLVM 两段。**

```
// 来源：lib/Dialect/Transforms/Pipelines.cppvoid buildOnnxToHipPipeline(OpPassManager &pm,                            const OnnxToHipPipelineOptions &options,                            morphizen::FileSystem *fs) {  pm.addPass(createSimplifyOnnxPass());  addPluginPassesForSlot(pm, PipelineSlot::AfterSimplifyOnnx);  pm.addPass(createHipAddContextArgPass());  pm.addPass(createOnnxLoopOutlinePass());  pm.addPass(createOnnxIfOutlinePass());  pm.addPass(createInferLoopBodyShapesPass());  pm.addPass(createConvertOnnxToHipPass());  addPluginPassesForSlot(pm, PipelineSlot::AfterConvertOnnxToHip);  buildOnnxToHipPipelineTail(pm, options, fs);}void buildHipToLLVMPipeline(OpPassManager &pm,                            const HipToLLVMPipelineOptions &options) {  pm.addNestedPass<func::FuncOp>(createRelaxMultiDynExpandShapePass());  addPluginPassesForSlot(pm, PipelineSlot::BeforeConvertHipToLLVM);  pm.addPass(memref::createExpandStridedMetadataPass());  pm.addPass(createLowerAffinePass());  pm.addPass(createAssignOpStateSlotsPass());  pm.addPass(createGenerateOpStateInitPass());  pm.addPass(createConvertHipToLLVMPass());  pm.addPass(createGenerateInterfacePass(compOpts));}
```

> 这段顺序背后有大量工程经验：

- 常量外置必须在 shape 推断之后，以免过早丢失可检查的 DenseElementsAttr。
- one-shot bufferize 前要解决 DPS init alias，否则类似 GQA present key/value 会共享错误 buffer。
- 输出分配要早于 pool-allocs，让 graph output 走 EP callback，避免进入临时 GPU 池。
- `expand-strided-metadata` 与 `lower-affine` 要在 HIP→LLVM 后段处理 memref view/reshape 的 stride 计算。

换句话说，pipeline 这条装配线的背后积累，考虑了任意工序错位都带来的 silent CPU fallback、错误别名、设备内存访问故障或 LLVM 翻译失败。

## 四、ONNX 到 HIP：算子转换不是逐个翻译这么简单

### 4.1 ConvertOnnxToHip 的核心策略

> `lib/Conversion/OnnxToHip/OnnxToHip.cpp` 说明了 hip-ep 的前端立场：它通过 MLIR 通用 `Operation` API 按名字匹配 ONNX op，不依赖 onnx-mlir 头文件和库。**转换目标是 HIP dialect 的 tensor DPS 形式，后续再统一 bufferize 到 memref。**

```
// 来源：lib/Conversion/OnnxToHip/OnnxToHip.cppstatic LogicalResult convertComputeOps(func::FuncOp funcOp, MLIRContext *ctx) {  RewritePatternSet patterns(ctx);  populateMatMulConversionPatterns(patterns, ctx);  populateConvConversionPatterns(patterns, ctx);  populateGqaConversionPatterns(patterns, ctx);  populateMultiHeadAttentionConversionPatterns(patterns, ctx);  populateMatMulNBitsConversionPatterns(patterns, ctx);  populateQMoEConversionPatterns(patterns, ctx);  populateReshapeConversionPatterns(patterns, ctx);  populateWhereConversionPatterns(patterns, ctx);  // 还包括 elementwise、reduce、gather、scatter、pool、resize 等大量算子  GreedyRewriteConfig config;  config.setStrictness(GreedyRewriteStrictness::ExistingOps);  return applyPatternsGreedily(funcOp, std::move(patterns), config);}
```

项目支持的算子族覆盖 transformer、vision、conv、量化与多模态模型常见路径：MatMul/Gemm、GQA、rotary embedding、KV-cache、norm、activation、gather/scatter、resize、control flow 等。代码把转换过程拆成一组 rewrite pattern，便于新算子独立扩展。

### 4.2 常量处理体现了大模型意识

> 大模型权重巨大，直接把常量塞进模型 artifact 会带来体积、加载时间与内存峰值问题。hip-ep 先把 `onnx.Constant` 降成策略中立的 `hip.constant carrier`，随后由 `hip-externalize-constants` 决定外置策略。运行时再按 metadata 将常量上传到 GPU。

这套设计有三个好处：

- 第一，shape 推断阶段仍能看到必要的常量值；
- 第二，大权重可以放入 sidecar constants file；
- 第三，运行时支持 bulk 与 per-entry 两种加载路径，能够对接 FileRef、Splat、MemSource 等来源。

## 五、内存：动态 shape 下的 GPU 池化与输出分配

### 5.1 临时 buffer 进入 grow-on-demand 池

> GPU 推理中，动态 shape 的临时张量大小会随输入变化。:mdnice-bold-italic[hip-ep 通过 `hip-pool-allocs` 把生命周期不重叠的临时分配压进一个或多个 pool domain。运行时按需扩容，且不主动缩小。]

```
// 来源：lib/Runtime/hipdnn_ep_runtime_state.cppvoid *hipdnn_ep_get_pool_base(RuntimeState *state,                              int domain_id,                              size_t needed_size) {  ensure_pool_domains(state, domain_id + 1);  // 0 字节请求保留 1 字节，保证 memref.view 有非空 base  if (needed_size == 0)    needed_size = 1;  if (needed_size > state->pool_size[domain_id]) {    if (state->pool_base[domain_id]) {      hipStreamSynchronize(state->stream); // 避免释放仍被异步 kernel 使用的旧池      hipFree(state->pool_base[domain_id]);    }    hipMalloc(&state->pool_base[domain_id], needed_size);    state->pool_size[domain_id] = needed_size;  }  return state->pool_base[domain_id];}
```

这段代码击中了推理系统的关键痛点：**避免热路径上频繁分配释放，兼顾动态 shape 的峰值增长**。多个 domain 的设计还能让不同支配域的 allocation 不互相牵制，减少过度膨胀。

### 5.2 输出由 EP 回调分配

> 图输出与临时 buffer 的所有权完全不同。临时 buffer 可以归 session pool，最终输出要回到 ONNX Runtime 的张量管理体系。hip-ep 通过 `hip.alloc_output` 与运行时回调建立 ABI：EP 先设置 allocator，生成的 `main_graph` 在知道 runtime shape 后调用 `hipdnn_ep_alloc_output`。

```
// 来源：lib/Runtime/output_allocator.cppextern "C" HIPDNN_EP_RT_EXPORT voidhipdnn_ep_set_output_allocator(RuntimeState *state,                               const hipdnn_output_allocator_t *allocator) {  if (!state) return;  state->output_allocator = allocator ? *allocator : hipdnn_output_allocator_t{};}extern "C" void *hipdnn_ep_alloc_output(RuntimeState *state,                                        int64_t out_idx,                                        const int64_t *shape,                                        int64_t rank,                                        int64_t elem_size) {  if (!state || !state->output_allocator.allocate)    return nullptr;  return state->output_allocator.allocate(      state->output_allocator.self, out_idx, shape, rank, elem_size);}
```

这相当于让编译后的图在内部完成“确定输出 shape → 申请输出 tensor → 写入 GPU 数据”的闭环。对动态输出尤其重要，因为编译期无法提前固定所有输出维度。

## 六、运行时：会话状态是模型执行的载体

### 6.1 RuntimeState 管住所有跨调用资源

> `lib/Runtime/hipdnn_ep_runtime_state.cpp` 中的 `RuntimeState` 初始化包括 HIP device、stream、hipBLASLt handle、device error flag、常量 blob、workspace、pool、host scratch、op state slots、profiling state 等。好比一个“模型会话对象”，负责跨多次 `inference_compute` 保持资源复用。

常量加载流程尤其值得注意：

- 先根据 FlatBuffers metadata 计算所有常量总大小；
- 在 GPU 上分配一个连续 blob；
- 将 `gpu_constants[i]` 修正到 blob 内部偏移；
- 若 metadata 中有 per-entry source，逐项加载；否则直接 bulk load `constants.bin`。

这种连续 blob + 指针数组的组合，既保留了常量索引访问的简单性，也减少了大量小 `hipMalloc` 带来的开销。

### 6.2 bitcode JIT 与 native artifact 的取舍

> README 明确说默认 artifact 是 OS-portable LLVM bitcode，会在 session 创建时由进程内 JIT 装载；native `.dll/.so` 属于显式请求路径。两者的取舍很清楚：

- LLVM bitcode 更利于跨 OS 打包与推迟链接，EP 可在本机环境中完成最终装载；
- native artifact 自包含程度更高，适合需要系统 loader 直接加载的场景；
- 两条路径都共享同一套 MLIR pipeline 与 Runtime ABI。

这也是 hip-ep 区别于单纯 kernel library 的地方：它把“模型”当成可编译单元，而非每个算子临时派发。

## 七、插件机制：给编译管线预留可控扩展点

### 7.1 静态插件避免平台差异

> `CMakeLists.txt` 与 `cmake/HipEpPlugins.cmake` 描述了插件策略：编译器插件静态链接进 host，在 configure 时选择 `HIPDNN_EP_COMPILER_PLUGINS`。这降低了运行时动态加载在 Windows/Linux、符号导出、LLVM ABI 上的复杂性。

文档中定义了多个 pipeline slot：`AfterSimplifyOnnx`、`AfterConvertOnnxToHip`、`BeforeBufferization`、`AfterPoolAllocs`、`BeforeConvertHipToLLVM`、`AfterGenerateInterface` 等。插件可以选择语义稳定的位置插入 pass，而无需重写完整 pipeline。

### 7.2 环境变量提供研究级逃生门

> `CompilerDriver::runMLIRPasses` 支持 `HIPDNN_EP_PIPELINE` 覆盖默认 pipeline。文档也提醒，最稳妥的方式是组合注册好的 pipeline 名称：

```
export HIPDNN_EP_PIPELINE="onnx-to-hip-pipeline, hip-to-llvm-pipeline"
```

此外还有 `HIPDNN_EP_IR_DUMP_PATH`、`HIPDNN_EP_IR_DUMP_TREE`、`HIPDNN_EP_STRICT` 等调试入口。对编译器开发者来说，这些开关极有价值：可以把每个 pass 之后的 IR 保存下来，进一步定位哪个阶段让某个 ONNX op 残留、哪个 bufferization 决策导致内存形态异常。

## 八、工具与测试：它如何证明自己可运行

### 8.1 命令行工具覆盖开发闭环

> README 列出的工具形成一条完整开发链：

- `hip-mlir-opt`：运行和检查 HIP/MLIR pass；
- `hip-compiler`：将 ONNX MLIR 编译成 LLVM bitcode 或 native artifact；
- `hip-onnx-runner`：把 ONNX 模型通过安装好的 EP 跑起来；
- `hip-inspect`：检查 artifact 内的 metadata；
- `hip-test`：通过 EP 同一路径加载并执行编译产物。

这些工具让开发者能从 IR、artifact、runtime 三个角度观察系统，而不用每次都进入完整 ONNX Runtime session。

### 8.2 测试分层覆盖编译与运行

`test/README.md` 将测试分为 LIT、E2E、runtime unit、numeric 等类别。LIT 主要验证 MLIR pass 与 FileCheck 模式，E2E 关注模型编译与执行，runtime unit 可在无 GPU 环境检查输出 allocator 等 ABI 逻辑。`build.py` 默认会跑 GPU-free 的 LIT 与部分 ctest，避免每次开发都绑定真实 GPU。

这种测试结构很适合编译器项目：pass 层用文本 IR 保证可重复，runtime 层用 mock 类型隔离硬件依赖，模型层再补真实端到端行为。

## 总结：把 EP 从适配层推进到编译层

> hip-ep 的意义在于：它把 ONNX Runtime 的 Execution Provider 做成了“模型级编译器 + 会话级运行时”。:mdnice-bold-italic[传统 EP 常以算子 kernel 注册为中心，遇到动态图、大权重、多后端、多形态 artifact 时，系统复杂度会散落到各处。hip-ep 选择用 MLIR 把复杂度集中到可组合、可检查、可扩展的编译管线中。]

它的技术路线可以概括为四句话：

- 第一，前端接住 ONNX 图，通过 MorphiZen 与 MLIR rewrite pattern 把高层模型语义转成 HIP dialect。
- 第二，中端用 shape inference、constant externalization、bufferization、pool planning 等 pass，把动态图压成运行时可执行的内存计划。
- 第三，后端生成 LLVM IR，并按 bitcode JIT 或 native artifact 两条路径交付。
- 第四，运行时通过稳定 C ABI 管理常量、输出分配、workspace、pool、stream 与 vendor backend 调用。

从工程成熟度看，:mdnice-bold-italic[这个仓库已经处理了许多“论文式编译器”较少触及的问题：Windows/MSVC 与 HIP 的混合构建、TheRock SDK 自动获取、ONNX Runtime 官方包对接、host-mapped scratch、GPU 池扩容时的异步安全、常量 sidecar、插件插槽、IR dump、strict 模式、mock runtime 测试]。它的代码量分布也印证了这一点：C++ 是主体，MLIR 和 HIP 占据重要比例，Python 承担构建、模型准备与工具脚本。

如果说 ONNX 是模型交换格式，ROCm 是 AMD GPU 的底层能力集合，那么 hip-ep 正在填补中间最难的一段：如何把复杂模型图变成面向 AMD GPU 的可执行程序，并让它自然嵌入 ONNX Runtime 的 session 生命周期。对开发者来说，:mdnice-bold-italic[它值得研究的地方在整条链路的设计：编译期做什么、运行时保留什么、ABI 如何稳定、动态 shape 怎样落到内存计划、插件如何在不破坏主线的前提下加入优化。]

::: block-1
这也是 hip-ep 最有启发性的地方：:mdnice-bold-italic[当推理模型越来越大，shape 越来越动态，后端越来越异构，Execution Provider 需要拥有编译器的判断力。hip-ep 给出的答案，是把 ONNX 图一路降到 LLVM bitcode，再在会话期把它变成 AMD GPU 上的执行计划。]
:::

参考资料

[1]

`docs/quick_start_linux.md`: *https://github.com/ROCm/hip-ep/blob/main/docs/quick_start_linux.md*

[2]

`docs/quick_start_linux.md`: *https://github.com/ROCm/hip-ep/blob/main/docs/quick_start_linux.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 把 CDNA5 的"底牌"摊开了：832 页 ISA 手册全公开，MI455X 的每个细节都写在里头](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904248&idx=1&sn=b2b81e0229f104f7dd3d3a9cb3d31dcf&scene=21#wechat_redirect)**
- **[首个跨 NVIDIA, AMD, Intel, Apple 的 16 代微架构 GPU ISA 研究，硬件不变原语系统性分析与通用 ISA 构想](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902613&idx=1&sn=e73f53c29947b54a31b61c7f6f84660d&scene=21#wechat_redirect)**
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**

:::
