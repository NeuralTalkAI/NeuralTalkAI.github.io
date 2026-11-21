关键词：**Triton 后端**、:mdnice-bold-italic[CUDA TileIR]、Blackwell GPU、**MLIR 方言转换**、无序内存模型

> *当 Triton 的 `tl.dot`、`tl.load` 遇上 NVIDIA Blackwell 上的 TMA、2CTA-MMA、CGA 调度，:mdnice-bold-italic[传统"Triton → TTGIR → LLVM/PTX → SASS"的下降路径开始显出疲态——它要在 SIMT 抽象上手工调度 warp、stage、register。]*

![文章插图](/articles/2026-07-15-triton-ptx-blackwell-cuda-tileir-ec23b636/images/01.webp)

- **Triton-to-tile-IR：Incubator repo for CUDA-TileIR backend**
- https://github.com/triton-lang/Triton-to-tile-IR

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 vLLM，SGLang 吞吐！cuTile Rust 实现近零开销安全内核，GEMM 达 96% cuBLAS 性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902639&idx=1&sn=1b7eb41a0df5119f59af2413a7e8bc71&scene=21#wechat_redirect)**
- **[从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)**
- **[NVIDIA 技术博客：cuTile.jl 将 CUDA 基于分块的编程模型引入 Julia 语言](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898470&idx=1&sn=13df29073e4622bf1ac3f5dd8741f097&scene=21#wechat_redirect)**

:::

> `triton-lang/Triton-to-tile-IR` 是 OpenAI Triton 官方的一处"孵化器"：它打开了一条新通道——`TTIR → CUDA TileIR (cuda_tile 方言) → cubin`，**把流水线编排、warp 分配、SMEM/TMEM 分配统统交给 NVIDIA 的 `tileiras` 编译器。**

本文将沿着 `TritonToTileIRPass.cpp`、`AutoGenMemoryToken.cpp` 和 Python 编译器入口 `compiler.py`，看 Triton 团队如何用一组优雅的 MLIR 重写模式，把声明式的 tile 语义重新交还给硬件。

## 本文目录

- 快速上手、使用
- 一、架构总览与设计哲学
  - 1.1 为什么需要 CUDA TileIR 后端
  - 1.2 仓库结构一览
  - 1.3 三层下降流水线
- 二、TritonToTileIR：把声明式张量交还给硬件
  - 2.1 方言转换的总体框架
  - 2.2 TMA 描述符：把 host API 翻译成 device view
  - 2.3 GEMM 与混合精度：把 `tt.dot` 拆成多次 MMA
  - 2.4 控制流：scf.while → cuda_tile.loop
  - 2.5 模式批量注册：约 50 个一行算子映射
- 三、AutoGenMemoryToken：无序内存模型的"语义补丁"
  - 3.1 问题的起源
- Known functional issuesCUDA Tile IR now supports only an unordered memory model,where global memory access operations are not ordered bydefault.If explicit memory access ordering is required, memorytoken semantics are required.
  - 3.2 序列识别：用 hash 给访存分组
  - 3.3 跨控制流的 token 传递
  - 3.4 启发式剪枝：跳过单访存与只读序列
- 四、RewriteAssumeWithCudaTile：把 LLVM 假设上提为 tile 语义
- 五、Python 编译入口与运行时
  - 5.1 TileIROptions：八个核心调优旋钮
  - 5.2 make_tileir：组合 Pass 形成完整管线
  - 5.3 make_cubin：和 tileiras 二进制对话
- 六、性能数据与未来路径
  - 6.1 实测数据：Persistent Matmul 提速 21%+
  - 6.2 未支持算子：当前的"功能棱角"
  - 6.3 路线图与启示

![文章插图](/articles/2026-07-15-triton-ptx-blackwell-cuda-tileir-ec23b636/images/02.webp)

## 快速上手、使用

> 仓库提供两条安装路径，最推荐的"side-by-side"模式可让 nvtriton（带 TileIR 后端的 Triton）与上游 oait Triton 并存，通过环境变量切换。所需前置：Python 3.12/3.13、PyTorch、CUDA 13.1+、Blackwell GPU。

```
# 1. 下载预编译 wheel（来自 release 页面）#    nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl# 2. 安装到隔离目录（不污染默认 triton）NVTRITON_DIR=$VIRTUAL_ENV/opt/nvtritonmkdir -p $NVTRITON_DIRpip install --no-cache-dir --no-deps --target $NVTRITON_DIR \    ./nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl# 3. 通过环境变量激活 TileIR 后端export PYTHONPATH=$NVTRITON_DIRexport ENABLE_TILE=1python my_script.py# 4. 验证后端是否激活python -c "from triton.runtime.driver import driver; print(type(driver.active).__name__)"# 期望输出: TileIRDriver
```

源码编译用 `pip install -e .` 即可，详见 INSTALL.md[1]。

- Helion 用户若要切到 TileIR 后端，需额外设 `HELION_BACKEND=tileir`，调优建议详见 HelionPerformanceTuningGuide.md[2]；
- 纯 Triton 内核的 `occupancy`/`num_ctas` 调优请参考 PerformanceTuningTips.md[3]。

## 一、架构总览与设计哲学

### 1.1 为什么需要 CUDA TileIR 后端

> 要理解这个仓库的价值，必须先看清 Triton 现有下降路径的天花板。

Triton 把高层 Python kernel 一路降到 TTGIR（Triton GPU IR），TTGIR 仍是 **SIMT 视图** ——它显式描述 warp、线程、共享内存布局，再走 LLVM PTX 后端生成机器码。在 Hopper、Blackwell 这样引入 **TMA、CGA（Cooperative Grid Array）、2CTA MMA、TMEM** 的架构上，:mdnice-bold-italic[这套 SIMT-first 的中间表示需要在编译器侧手工组合上面的硬件特性，代码越来越像汇编🤣。]

::: block-1
:mdnice-highlight[NVIDIA 内部的 CUDA TileIR（`cuda_tile` MLIR 方言）则把 tile 视为一等公民]：一个 `tile<MxNxf16>` 不是 32 个 thread 各持一份，而是一个 **CGA 级** 的对象，由 tileiras 编译器统一决定它落在哪些 SM、用多少 warp 完成、走 TMA 还是普通 LDG。
:::

`README.md` 点明了仓库的定位：

```
# Triton CUDA Tile IR BackendThis incubator repo adds the CUDA Tile IR backend to Triton.Users can enable the CUDA Tile IR backend by setting theenvironment variable `ENABLE_TILE=1`.
```

也就是说：**一行 `export ENABLE_TILE=1`，整条编译管线就从"TTIR→TTGIR→LLVM/PTX"切到"TTIR→TileIR→cubin"。** 后者跳过了 TTGIR 这一 SIMT 层，把调度权交给硬件原生的 IR。

### 1.2 仓库结构一览

仓库延续 Triton 的目录布局，新后端集中在 `third_party/tileir/` 子树下：

```
third_party/tileir/├── triton_tileir.cc            # PyBind11 入口，把 C++ Pass 暴露给 Python├── backend/                    # Python 编译器和 driver（与 third_party/nvidia 对齐）│   ├── compiler.py             # TileIRBackend：make_ttir / make_tileir / make_cubin│   ├── driver.py               # TileIRDriver：cuLaunchKernelEx 启动逻辑│   └── conf.py                 # 环境变量读取（TILEIR_ENABLE_APPROX 等）├── include/                    # MLIR Pass 头文件（TableGen 生成）├── lib/│   ├── TritonToTileIR/│   │   └── TritonToTileIRPass.cpp   # 主战场：~2880 行的 OpConversion 模式集│   └── Transform/│       ├── AutoGenMemoryToken.cpp   # 无序内存模型自动加 token│       ├── LiftTTCFToSCF.cpp        # cf → scf 升提（操作在 tt.func 上）│       └── RewriteAssumeWithCudaTile.cpp  # llvm.assume → cuda_tile.assume├── PerformanceTuningTips.md    # 调优指南└── README.md
```

仓库其余部分基本保持 Triton 原貌；新增的"魔法"集中在 `third_party/tileir/` 与一组对 `compiler.py`、`driver.py`、`jit.py` 的小幅 patch（`ENABLE_TILE=1` 触发的 fallback 路径）。

### 1.3 三层下降流水线

> 整个后端的核心由 `compiler.py` 中三段静态方法定义清楚：

```
# third_party/tileir/backend/compiler.pydef add_stages(self, stages, options, language):    assert language == Language.TRITON, "Only TRITON language is supported for now"    capability = int(self._parse_arch(options.arch))    stages["ttir"] = lambda src, metadata: self.make_ttir(        src, metadata, options, capability    )    stages["tileir"] = lambda src, metadata: self.make_tileir(        src, metadata, options, capability    )    stages["cubin"] = lambda src, metadata: self.make_cubin(        src, metadata, options, capability    )
```

三个阶段构成完整的 lowering 链：

1. **`make_ttir`：** 与 NVIDIA 后端共享的标准 Triton 优化（inliner、combine、canonicalizer、CSE、LICM、symbol-DCE）。
2. **`make_tileir`：** 本仓库的灵魂——把 TTIR 转成 `cuda_tile` 方言，下面会详细拆解。
3. **`make_cubin`：** 调用 `bin/tileiras` 二进制将 cuda_tile bytecode 编成 cubin。

值得注意，仓库 **完全跳过了 TTGIR 阶段** ——TTIR 直降到 TileIR。SIMT 层不复存在。

## 二、TritonToTileIR：把声明式张量交还给硬件

`third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cpp` 是仓库技术含量最高的文件🤣，3000 行左右的 C++ 里塞满了 60+ 个 `OpConversionPattern`，每一个都对应一类 Triton 算子到 `cuda_tile` 算子的语义映射。

### 2.1 方言转换的总体框架

主 Pass `ConvertTritonToCudaTile` 走的是 MLIR 的 **Dialect Conversion** 标准范式：定义 `ConversionTarget`（哪些方言合法）、`TypeConverter`（类型映射）、`RewritePatternSet`（重写规则），再 `applyFullConversion` 一气呵成。

```
// 来源：third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cppclass CudaTileConversionTarget :public ConversionTarget {public:  CudaTileConversionTarget(MLIRContext &context,                           CudaTileTypeConverter &typeConverter)      : ConversionTarget(context) {    // 只允许 cuda_tile 方言存活，其他全部非法    addLegalDialect<cuda_tile::CudaTileDialect>();    addIllegalDialect<scf::SCFDialect, cf::ControlFlowDialect,                      mlir::gpu::GPUDialect, triton::TritonDialect,                      ub::UBDialect>();    addLegalOp<ub::PoisonOp>();    // barrierOp will be removed in AutoGenMemoryTokenPass    addLegalOp<mlir::gpu::BarrierOp>();   // 后续 Pass 会消去    addLegalOp<arith::IndexCastOp>();    addLegalOp<UnrealizedConversionCastOp>();  // 类型转换"桥"  }};
```

这个目标声明非常激进：**只有 `cuda_tile` 方言是合法的**，其余 dialect（包括上游 `scf`、`cf`、`triton`、`ub`）必须在 Pass 结束前消失。少数几个例外（`gpu::BarrierOp`、`arith::IndexCastOp`）是为下游 Pass 留的"过渡券"。

> `ConvertTritonToCudaTile::runOnOperation` 方法揭示了完整的执行流程：

```
// 来源：third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cppvoid runOnOperation() override {  MLIRContext *context = &getContext();  ModuleOp mod_buildin = getOperation();  CudaTileTypeConverter typeConverter;// 1) 把 builtin.module 包进 cuda_tile.module 容器OpBuilder builder(context);auto mod = cuda_tile::ModuleOp::create(builder, mod_buildin.getLoc(),                                         "cuda_tile_module");// ... 把 mod_buildin 的 body 整体克隆进 cuda_tile::ModuleOp ...// 2) 处理 host TMA 描述符：把 tt.tensordesc 函数参数展开成 ptr+shape+stride  convertTmaDescriptorOps(mod.getOperation(), typeConverter);// 3) 轴信息分析：用 triton 的 AxisInfo 推出 divisibility/contiguityModuleAxisInfoAnalysis axisInfo(mod_buildin);  checkDivisibilityForDescriptorOps(mod_buildin, axisInfo);  convertAxisAttributes(mod_buildin, axisInfo, typeConverter);// 4) 收集 load/store 的 num_stages 提示（来自父 forOp）  getNumStages(mod.getOperation(), this->numStagesMap);// 5) 预处理：tt.map_elementwise 展开成 tensor 级算子if (failed(mlir::triton::expandMapElementwiseOps(mod.getOperation())))    return signalPassFailure();// 6) 主菜：注册所有 OpConversionPattern，applyFullConversionCudaTileConversionTarget target(*context, typeConverter);RewritePatternSet patterns(context);  populateTTirToCudaTileConversionPatternsAndLegality(      typeConverter, patterns, target, this->approxModifier,      this->flushToZeroModifier, this->numStagesMap, this->computeCapability,      this->numCTAInCGA, this->simtNumWarpsInCTA, this->occupancy,      this->numStages);if (failed(applyFullConversion(mod, target, std::move(patterns), config)))    return signalPassFailure();// 7) 清理残留的 unrealized_conversion_cast  reconcileUnrealizedCasts(castOps, &remainingCastOps);}
```

### 2.2 TMA 描述符：把 host API 翻译成 device view

Triton 在 host 端通过 `tt.make_tensor_desc` 创建 TMA 描述符，但 cuda_tile 的 TMA 实现 **只有 device 版本**。这种语义鸿沟由 `convertTmaDescriptorOps` 和 `ConvertMakeTensorDescOp` 联手填补——后者负责设备侧的 lowering：

```
// 来源：third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cpp// TMA 对齐铁律：指针、stride 必须 16 字节对齐staticconstexprint64_t kTMAAlignment = 16;constexprunsignedlonglong kMaxStride = (1LL << 40) - 1;constexprunsignedlonglong kMaxShape  = (1LL << 32) - 1;LogicalResult matchAndRewrite(triton::MakeTensorDescOp op, ...) {// ① 把每个动态 shape/stride 包装成 scalar tile，附加 range 与 div-by 假设for (auto v : op.getShape())    wrappedDynShapes.push_back(        wrapIntoScalarTile(rewriter, v, /*attachAlignment=*/0, kMaxShape));// ② 处理 stride（最后一维必须为 1，其余必须 16-byte 对齐）for (size_t i = 0; i < rank; ++i) {    if (i == rank - 1) {      // 末维硬约束：必须是 constexpr 1      if (!constValOpt.has_value() || constValOpt.value() != 1)        return rewriter.notifyMatchFailure(loc, "last stride must be 1");    } else {      // 中间维：附加 16-byte 对齐 + 2^40 上界假设      wrappedDynStrides.push_back(wrapIntoScalarTile(          rewriter, stride, /*attachAlignment=*/align_byte, kMaxStride));    }  }// ③ 指针附加 16-byte 对齐 assumeauto ptrWithDivBy = cuda_tile::AssumeOp::create(      rewriter, loc, ptrOp,      cuda_tile::DivByAttr::get(ctx, kTMAAlignment, std::nullopt, std::nullopt))      .getResult();// ④ 构造 cuda_tile.make_tensor_view + make_partition_viewauto makeTensorViewOp = cuda_tile::MakeTensorViewOp::create(...);auto partViewOp = cuda_tile::MakePartitionViewOp::create(      rewriter, loc, tilePartViewTy, makeTensorViewOp);  rewriter.replaceOp(op, partViewOp);}
```

这里的关键点是 **"假设而非检查"：** 所有对齐、上界都通过 `cuda_tile::AssumeOp` 注入 IR，下游 tileiras 编译器据此做最激进的优化。如果用户传错指针，结果是 UB——但相对地，编译器获得了和手写 CUDA 等价的特化能力。

### 2.3 GEMM 与混合精度：把 `tt.dot` 拆成多次 MMA

> `ConvertDotOp` 是性能上最重的一段重写。

Triton 的 `tt.dot` 支持 `tf32`、`tf32x3`、`bf16x3`、`bf16x6` 等多种"非 IEEE"模式，本质都是 **用低精度 MMA 多次累加来逼近 fp32 精度**。仓库用一段优雅的"split-then-mma"模式实现：

```
// 来源：third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cpp// 把 f32 input 切成 N 段低精度（用 splitF32 lambda），再分别做 MMA 累加auto splitF32 = [&](Value input, unsigned N, FloatType dstFloatTy "&") {  llvm::SmallVector<Value, 3> splitInputs;  FloatType f32Ty = rewriter.getF32Type();for (unsigned i = 0; i < N; ++i) {    Value dstValue = convertFloat(input, dstFloatTy);   // f32 → tf32 / bf16    if (i != N - 1) {      Value dstF32Value = convertFloat(dstValue, f32Ty); // 回升 f32 算残差      input = sub(input, dstF32Value);                   // input -= hi/mid    }    splitInputs.push_back(dstValue);  }return splitInputs;};// 非 IEEE 模式：mixed precisionif (isF32(op.getA()) && isF32(op.getB()) &&    op.getInputPrecision() != InputPrecision::IEEE) {// 根据精度模式选 split 段数switch (op.getInputPrecision()) {    case InputPrecision::TF32:   computeTy=TF32; nSplits=1; break;    case InputPrecision::TF32x3: computeTy=TF32; nSplits=2; break;    case InputPrecision::BF16x3: computeTy=BF16; nSplits=2; break;    case InputPrecision::BF16x6: computeTy=BF16; nSplits=3; break;  }constauto lhs_parts = splitF32(adaptor.getA(), nSplits, computeTy);constauto rhs_parts = splitF32(adaptor.getB(), nSplits, computeTy);if (nSplits == 1) {    // TF32：单次 MMA 即可    auto result = mma(lhs_parts[hi], rhs_parts[hi], adaptor.getC());  } else {    // 多次累加：跨段交叉乘加，最后再做一次 hi×hi 抵冲    auto result = zeroLike(adaptor.getC());    if (nSplits > 2) {  // BF16x6      result = mma(lhs_parts[mid], rhs_parts[mid], result);      result = mma(lhs_parts[lo],  rhs_parts[hi],  result);      result = mma(lhs_parts[hi],  rhs_parts[lo],  result);    }    if (nSplits > 1) {      result = mma(lhs_parts[mid], rhs_parts[hi],  result);      result = mma(lhs_parts[hi],  rhs_parts[mid], result);      result = replaceNansWithZeros(result);  // 处理溢出 NaN    }    result = mma(lhs_parts[hi], rhs_parts[hi], result);    result = add(result, adaptor.getC());  }}
```

这段代码诠释了 **"精度即算力"** 的思想：通过把 1 次 f32 MMA 换成 6 次 bf16 MMA，可以在保留 ~24 bit 尾数精度的同时吃满 Blackwell 上 TFlops 翻倍的 bf16 张量核。Triton 用户只要写 `tl.dot(a, b, allow_tf32=False, input_precision='bf16x6')`，剩下的拆分逻辑全在这里。

### 2.4 控制流：scf.while → cuda_tile.loop

cuda_tile 没有等价的 while 语义，于是 `ConvertWhileOp` 必须用 if + continue/break 重新装配出 while 形态：

```
// 来源：third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cpp// 改写规则示意：//   scf.while  {before} do {after}// 改成：//   cuda_tile.loop iter_values(...) {//     <before>//     cuda_tile.if %cond {//        <after>; cuda_tile.continue <yield>//     }//     cuda_tile.break <cond_args>//   }class ConvertWhileOp :public OpConversionPattern<scf::WhileOp> {LogicalResult matchAndRewrite(scf::WhileOp whileOp, ...) const override {    auto newLoopOp = cuda_tile::LoopOp::create(...);    Block *newLoopBlock = rewriter.createBlock(&newLoopOp.getRegion(), ...);    // before-region 内联到 loop block    rewriter.inlineBlockBefore(beforeBlock, newLoopBlock, ...);    // 用 if 把 after-region 包起来，then-branch 走 continue，else-branch 走 break    auto ifOp = cuda_tile::IfOp::create(rewriter, ...);    rewriter.inlineBlockBefore(afterBlock, ifOp.getThenBlock(), ...);    cuda_tile::ContinueOp::create(rewriter, ..., yieldOpCastArgs);    cuda_tile::BreakOp::create(rewriter, ..., afterBlockCastArgs);   // else    cuda_tile::BreakOp::create(rewriter, ..., afterBlockCastArgs);   // loop end  }};
```

逻辑上等价下面，该结构本质是一个无限循环，通过 `break` 提前退出，只是用 IR 语法重排。

```
do {    before;    if (cond) {        after;        continue;    }    break;} while (true);
```

### 2.5 模式批量注册：约 50 个一行算子映射

> `populateTTirToCudaTileConversionPatternsAndLegality`是这场转换战的"指挥部"。绝大多数算子只需要一个泛型模板 `ConvertGenericOp<SrcOp, DstOp, Signedness, IntegerUpCast>`：

```
// 来源：third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cpppatterns.add<// arith 算术  ConvertGenericOp<arith::AddFOp, cuda_tile::AddFOp, Signedness::None, IntegerUpCast::None>,  ConvertGenericOp<arith::AddIOp, cuda_tile::AddIOp, Signedness::None, IntegerUpCast::To_I16>,// ... 30+ arith ops ...// math 数学  ConvertGenericOp<math::ExpOp,  cuda_tile::ExpOp,  ...>,  ConvertGenericOp<math::Log2Op, cuda_tile::Log2Op, ...>,// ... 15+ math ops ...// triton 专用  ConvertGenericOp<triton::AddPtrOp,    cuda_tile::OffsetOp,    ...>,  ConvertGenericOp<triton::MulhiUIOp,   cuda_tile::MulhiIOp,    ...>,  ConvertGenericOp<triton::PreciseSqrtOp, cuda_tile::SqrtOp,    ...>>(typeConverter, context, approx, flushToZero);
```

注意 `IntegerUpCast::To_I16` 这种细粒度参数——cuda_tile 不支持 i8 的某些运算，必须先升到 i16；这种工程级细节在仓库里随处可见。

## 三、AutoGenMemoryToken：无序内存模型的"语义补丁"

### 3.1 问题的起源

`README.md` 直白地承认：

```
## Known functional issuesCUDA Tile IR now supports only an unordered memory model,where global memory access operations are not ordered bydefault.If explicit memory access ordering is required, memorytoken semantics are required.
```

cuda_tile 的全局内存访问 **默认无序** ——这是为了释放编译器的调度自由度。但 Triton 用户写代码时是按 **顺序内存模型** 思考的。一个简单的"先写后读同一地址"模式，在 cuda_tile 里如果不加 token，编译器有权完全乱序执行。

::: block-1
`AutoGenMemoryToken.cpp`（这个 Pass 名字是 `add_auto_gen_memtoken`）就是这道难题的工程答案：**自动识别可能存在数据依赖的访存序列，给它们串上 token，让 tileiras 维持原本的顺序语义**。
:::

### 3.2 序列识别：用 hash 给访存分组

> Pass 把所有访存 op 按"访问的数据"分组（每组叫一个 sequence，分配一个 `SeqId`）。:mdnice-bold-italic[难点是怎么判断"访问的是同一片数据"——对于指针类访存，用指针 Value 的 hash；对于 view 类访存，用 `view + indices` 联合 hash：]

```
// 来源：third_party/tileir/lib/Transform/AutoGenMemoryToken.cppSeqId getMemOpSeqId(Operation *op) {// 若 kernel 有 debug_barrier，全部归到同一序列（最保守）if (currentBlockMemSeqs.hasBarrierOp)    return BARRIER_SEQ_ID;  Value val = Value();  SmallVector<Value> indexes;if (auto loadOp = dyn_cast<cuda_tile::LoadPtrTkoOp>(op)) {    val = loadOp.getSource();    if (loadOp.getToken())      currentBlockMemSeqs.hasMemToken = true;  // 用户已显式加了 token  } elseif (auto viewOp = dyn_cast<cuda_tile::LoadViewTkoOp>(op)) {    val = viewOp.getView();    indexes = viewOp.getIndex();      // view ops 还要把 index 一起 hash    // ...  }// ... 其他 6 种访存 op ...// 把 ptr value + 所有 index value 的 mlir hash 合成  llvm::hash_code hash = mlir::hash_value(val);for (const Value &val : indexes)    hash = llvm::hash_combine(hash, mlir::hash_value(val));return hash;}
```

文件顶部的注释（21-56 行）坦率列出 3 个 **不能处理** 的场景：ptr-op 与 view-op 访问同一数据、不同地址但范围重叠、用户传入两个指向同一内存的指针。这是无序内存模型必然要付出的代价。

### 3.3 跨控制流的 token 传递

> token 不只是在线性代码里串一遍那么简单——遇到 `if`、`for`、`while` 时，token 需要 **穿过控制流边界**。

Pass 用三个并行的 handler 处理 `cuda_tile::IfOp` / `cuda_tile::ForOp` / `cuda_tile::LoopOp`，逻辑大同小异：递归处理子 block，回来时把更新过的 token 当作新的 yield/init 参数注入到容器 op 的返回类型与入口参数：

```
// 来源：third_party/tileir/lib/Transform/AutoGenMemoryToken.cppSeqTokens handleForOpTokens(cuda_tile::ForOp forOp, SeqTokens tokens,                            IRRewriter &rewriter) {// 进入循环体，递归收集每个序列的最终 token  SeqTokens forTokens =      addMemTokenForBlock(forOp.getBody(), tokens, rewriter, &localTermOps);  SeqIdVec sids = tokens.aggregate(tokenList);if (!sids.empty()) {    // 1) terminator 末尾加上 token，作为循环回边    updateTermOpWithToken(forOp.getBody()->getTerminator(), forTokens, sids);    // 2) 把当前 token 当作新的 init args 喂给新建的 forOp    SmallVector<Value> newInitArgs = llvm::to_vector(forOp.getInitValues());    for (constauto &sid : sids)      newInitArgs.push_back(tokens[sid]);    auto newForOp = cuda_tile::ForOp::create(        rewriter, forOp.getLoc(), forOp.getLowerBound(),        forOp.getUpperBound(), forOp.getStep(), newInitArgs);    // 3) 在 loop body 内用 block argument 替换 token 引用    for (size_t i = 0; i < sids.size(); i++) {      auto arg = newBlock->getArgument(originNumArgs + i);      tokens[sids[i]].replaceAllUsesExcept(arg, newForOp);      tokens.update(sids[i], newForOp.getResult(forOp.getNumResults() + i));    }  }}
```

### 3.4 启发式剪枝：跳过单访存与只读序列

最后 Pass 给自己加了两道剪枝（`runOnOperation`）：

```
// 来源：third_party/tileir/lib/Transform/AutoGenMemoryToken.cpp// 仅当 sequence 含 >1 个访存且至少一个是 write 时才插入 tokenfor (auto &p : currentBlockMemSeqs.srcToMemSeqInfoMap) {  if (p.second.memOpCounter <= 1 || !p.second.writeMemOpCounter)    p.second.ignored = true;  else    willModify = true;}if (!willModify) {  LLVM_DEBUG(llvm::errs() << "[AutoGenMemoryTokenPass] will not modify IR\n");  return WalkResult::skip();}
```

只有 **同一数据被访问 >1 次且至少有一次写** 的序列才需要 token——纯只读和单点访问完全可以无序执行，token 是纯开销。这条规则把保序成本压到了最小。

## 四、RewriteAssumeWithCudaTile：把 LLVM 假设上提为 tile 语义

Triton 用户常写 `tl.assume(offset % BLOCK == 0)` 来辅助编译器优化。前端会把它降成 `llvm.intr.assume`，但 cuda_tile 完全不认识 LLVM 方言——这就是 `RewriteAssumeWithCudaTile.cpp` 的用武之地。它专门匹配 `assume(a % k == 0)` 这种"divisibility"模式：

```
// 来源：third_party/tileir/lib/Transform/RewriteAssumeWithCudaTile.cpp// 匹配模式：//   %rem = arith.remsi %a, %k        ; a 可以是 i32 也可以是 tt.ptr_to_int 结果//   %eq  = arith.cmpi  eq, %rem, 0//   llvm.intr.assume %eq// 改写成：//   %tile_a = unrealized_conv (a) → tile<i32>//   %assume = cuda_tile.assume div_by<k>, %tile_a : tile<i32>//   ... 用 %assume 替换所有 dominator 下游的 %aLogicalResult RewriteArithAssumeImpl(LLVM::AssumeOp assumeOp,                                     PatternRewriter &rewriter) {// Step 1-5: 把 cmpi-eq / cst-0 / remsi / cst-k 的 use-def 链验证一遍auto cmpOp = condValue.getDefiningOp<arith::CmpIOp>();if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq) return failure();// ... 一连串模式匹配 ...int64_t divisor = divisorAttr.getValue().getSExtValue();// Step 6: 处理指针 vs 整数两种来源auto ptrToIntOp = intOrPtrToInt.getDefiningOp<triton::PtrToIntOp>();if (ptrToIntOp) {    // 指针：cast 回 cuda_tile.ptr，加 div_by attr    auto cudaTilePtr = ...    auto assumeCudaTileOp = cuda_tile::AssumeOp::create(        rewriter, loc, cudaTilePtr,        cuda_tile::DivByAttr::get(ctx, divisor, std::nullopt, std::nullopt));  } else {    // 整数：直接 cast 成 scalar tile<i32> 并 assume    auto cudaTileInt = ...    auto assumeCudaTileOp = cuda_tile::AssumeOp::create(rewriter, loc, cudaTileInt, divByAttr);  }}
```

仔细看就会发现，这个 Pass 把"用户的标量整数假设"提升成了"tile 级假设"——通过插入两个 `UnrealizedConversionCastOp` 形成 `int ↔ tile<int>` 的桥梁。dominator 分析（`domInfo.dominates`）保证只替换语义可达的 use，避免引入 cyclic dep。这种"用 cast 当通行证"的技巧在仓库里多次复用，是 MLIR 增量式 lowering 的经典写法。

## 五、Python 编译入口与运行时

### 5.1 TileIROptions：八个核心调优旋钮

`compiler.py` 顶部的 `TileIROptions` dataclass 是后端对外暴露的配置面板：

```
# 来源：third_party/tileir/backend/compiler.py@dataclass(frozen=True)class TileIROptions:    backend_name: str = "tileir"    num_ctas: int = 1          # 关键：2CTA-MMA on Blackwell    num_stages: int = 3        # cost hint，不再是硬指令    opt_level: int = 3    occupancy: int = 1         # TileIR 专属：1-32，控制 register usage    enable_fp_fusion: bool = True    # 兼容字段（cuda_tile 不用，但保留以兼容现有 Triton 调度）    num_warps: int = 4         # 在 TileIR 不可调，固定 4    # ...    enable_autogen_alias_mem_token: bool = True    @property    def enable_ftz(self):        return TileIREnvConf.enable_ftz()   # 环境变量动态读    @property    def enable_approx(self):        return TileIREnvConf.enable_approx()
```

这里几条设计哲学值得品味：

- **`num_warps` 不可调**：tileiras 自己决定 warp 数，把"几个 warp 一起处理 tile"这件事抽出来。
- **`num_stages` 从硬指令变 cost hint**：上游 PTX 后端的 num_stages 是严格的流水深度，TileIR 把它当作"我希望多深"的偏好，编译器全局优化时可能调整。
- **`occupancy` 是新一等公民**：1-32 的整数，表达"每个 SM 同时跑多少 block"的期望，autotuner 默认搜 {1, 2, 4, 8}。

### 5.2 make_tileir：组合 Pass 形成完整管线

> `make_tileir` 静态方法是 TileIR 后端真正的"组装车间"：

```
# 来源：third_party/tileir/backend/compiler.py@staticmethoddef make_tileir(mod, metadata, opt: TileIROptions, capability):    pm = ir.pass_manager(mod.context)    pm.enable_debug()    # 1. cf → scf：让控制流先升一格，方便后续转 cuda_tile.if/for    tileir.passes.add_lift_tt_cf_to_scf(pm)    # 2. llvm.assume → cuda_tile.assume    tileir.passes.add_assume_to_tileir(pm)    # 3. 主菜：所有 OpConversion 模式    tileir.passes.add_triton_to_cudatile(        pm, opt.enable_approx, opt.enable_ftz, capability,        metadata["num_ctas"], metadata["num_warps"],        opt.occupancy, metadata["num_stages"],    )    # 4. 自动加 memory token    tileir.passes.add_auto_gen_memtoken(pm, opt.enable_autogen_alias_mem_token)    # 5. 内联 & FMA 融合    passes.common.add_inliner(pm)    if opt.enable_fp_fusion:        tileir.passes.add_fma_fusion(pm)    tileir.passes.add_strip_debuginfo(pm)    pm.run(mod, "make_tileir")    # 6. 强校验：模块里不能再有非 cuda_tile 方言的 op    ifnot tileir.only_contain_legal_dialects(mod):        raise RuntimeError("Triton ttir to tileir ir failed.")
```

整条管线 6 步，每一步都在前文剖析过。最后那行 `only_contain_legal_dialects` 是一道防御性检查——保证转换后的 IR 是纯净的 cuda_tile，否则 tileiras 会立刻报错。

### 5.3 make_cubin：和 tileiras 二进制对话

> `make_cubin`（`call_tileiras` 静态方法）实际上是个子进程封装：把 cuda_tile bytecode 写到临时文件，喂给 `tileiras --gpu-name=sm_X --opt-level=3`，再读回 cubin。最巧的是它对 `tileiras` 错误的细分处理：

```
# 来源：third_party/tileir/backend/compiler.pytry:    subprocess.run(tileiras_cmd, check=True, ..., env=tileiras_env)except subprocess.CalledProcessError as e:    log = open(flog.name).read()    # 把 "uses too much shared data" 翻译成结构化异常 OutOfResources    if"uses too much shared data"in log:        match = re.search(r"0x([0-9a-fA-F]+) bytes, 0x([0-9a-fA-F]+) max", log)        if match:            raise OutOfResources(used_smem, max_smem, "shared memory")    if"allocated tmem out of resource"in log:        raise OutOfResources(used_tmem, max_tmem, "tensor memory")    # SIGSEGV (-11) 等致命信号：抛 TileirasError，让 autotuner 跳过该 config    raise TileirasError(...)
```

这种"把字符串日志变结构化异常"的小心思直接服务于自动调优——`OutOfResources` 让 Helion 的 autotuner 自动剪枝不可行的 block_size 组合，整体调优效率显著提升。

## 六、性能数据与未来路径

### 6.1 实测数据：Persistent Matmul 提速 21%+

`PerformanceTuningTips.md` 给出了 B200 (1000W) 上 16 个 K 值（512~8192）下、8 个 matmul 变体的全套数据。摘要一组对比：

| K=4096 | NVIDIA PTX backend | CUDA TileIR Backend | 加速比 |
| --- | --- | --- | --- |
| `matmul_kernel_descriptor_persistent` | 524.3 TFLOPS | **632.6 TFLOPS** | +20.7% |
| `matmul_kernel_tma_persistent` | 509.2 TFLOPS | **635.6 TFLOPS** | +24.8% |
| `matmul_kernel_tma_persistent_ws` | 534.0 TFLOPS | **633.8 TFLOPS** | +18.7% |
| `matmul_kernel_tma_ws` | 495.3 TFLOPS | **599.5 TFLOPS** | +21.0% |

TileIR 后端在 Persistent + TMA 这类"对调度敏感"的 kernel 上一致领先，最高 +24%。

::: block-1
原因不难理解：:mdnice-bold-italic[tileiras 在 CGA 级看到整个 grid，能自动决定 cluster scheduling、warp specialization、stage 数；而 PTX 后端是 thread-block 视角，跨 CTA 协作要靠 Triton 手动编排。]
:::

### 6.2 未支持算子：当前的"功能棱角"

`README.md` 列出了 20+ 项尚未支持或不完善的特性，包括 `tt.dot_scaled`、TMA scatter/gather/reduce、`atomic_rmw(bf16)`、i64 索引等。

ConvertExternElementwiseOp 部分那一长串 `__nv_xxx` 分支昭示着 libdevice 函数还在逐个补齐——比如 `erf` 是直接用 Horner 多项式手写的 7 阶逼近，`asin`/`acos` 用 `atan2 + sqrt` 重写。

这些"工程菜单"既反映了 cuda_tile 方言当前的边界，也明白告知用户哪些 kernel 需要 fallback。

### 6.3 路线图与启示

> 仓库注释里多次提到 "next CUDA release"——`num_warps` 暴露、小 GEMM 性能、`tensor-of-pointer` 路径优化都被 NVIDIA 排到了未来版本。短期内 Triton 用户最好的实践是：

1. **GEMM/Attention** → 强制 `indexing="tensor_descriptor"`（TMA 路径），加 `occupancy=1~2`。
2. **Elementwise/Reduction/Norm** → `indexing="pointer"`，加 `occupancy=4`。
3. **大 GEMM** → 试 `num_ctas=2` 启用 2CTA-MMA。
4. **加 `tl.assume(offset % BLOCK == 0)`——** 让 `RewriteAssumeWithCudaTile.cpp` 帮你把信息传到 tileiras。

更深远的启示是：**当硬件抽象从 SIMT 走向 CGA-Tile，编译器中间表示必须同步上升一个层次**。

::: block-1
:mdnice-bold-italic[Triton 自身的 TTGIR 是这场迁移的"过渡形态"——它仍然描述 warp/thread；NVIDIA 的 cuda_tile 则是终态——一个 tile 就是一个对象，调度细节回到硬件原生 IR 手中。]

Triton 用 `third_party/tileir` 这样一个增量后端，不抢占任何已有用户、不破坏 PTX 路径，又把通往下一代调度抽象的桥架了起来。
:::

> 关键参考：仓库主页[4] · TritonToTileIRPass.cpp[5]AutoGenMemoryToken.cpp[6] · Performance Tuning Tips[7] · Helion Tuning Guide[8]

参考资料

[1] INSTALL.md: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/INSTALL.md*

[2] HelionPerformanceTuningGuide.md: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/HelionPerformanceTuningGuide.md*

[3] PerformanceTuningTips.md: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/third_party/tileir/PerformanceTuningTips.md*

[4] 仓库主页: *https://github.com/triton-lang/Triton-to-tile-IR*

[5] TritonToTileIRPass.cpp: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/third_party/tileir/lib/TritonToTileIR/TritonToTileIRPass.cpp*

[6] AutoGenMemoryToken.cpp: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/third_party/tileir/lib/Transform/AutoGenMemoryToken.cpp*

[7] Performance Tuning Tips: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/third_party/tileir/PerformanceTuningTips.md*

[8] Helion Tuning Guide: *https://github.com/triton-lang/Triton-to-tile-IR/blob/main/HelionPerformanceTuningGuide.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 vLLM，SGLang 吞吐！cuTile Rust 实现近零开销安全内核，GEMM 达 96% cuBLAS 性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902639&idx=1&sn=1b7eb41a0df5119f59af2413a7e8bc71&scene=21#wechat_redirect)**
- **[从 SIMT 到 Tile：CUDA 13.1 新特性实现编译优化提速 50%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894108&idx=1&sn=d5e3b846e0f6dcbfdafd8bc3d6ee9588&scene=21#wechat_redirect)**
- **[NVIDIA 技术博客：cuTile.jl 将 CUDA 基于分块的编程模型引入 Julia 语言](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898470&idx=1&sn=13df29073e4622bf1ac3f5dd8741f097&scene=21#wechat_redirect)**

:::
