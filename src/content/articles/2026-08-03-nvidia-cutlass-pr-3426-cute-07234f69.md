**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**CuTe IR**、MLIR Dialect、cutegen、**cute-to-base**、:mdnice-bold-italic[NVVM Lowering]

![文章插图](/articles/2026-08-03-nvidia-cutlass-pr-3426-cute-07234f69/images/01.jpg)

- Cutlass compiler: Introducing CuTe IR MLIR dialect- #3426
- https://github.com/NVIDIA/cutlass/pull/3426

::: related-title
**相关推荐**
:::

::: related-links
- **[无 GPU 也能玩转 CuTe！tensor-layouts：一份纯 Python 的 CuTe 代数实现，让 GPU 张量布局变成可读、可玩、可视](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903658&idx=1&sn=dbba694188a91680d9149ca9a1596276&scene=21#wechat_redirect)**
- **[对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)**
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**

:::

> 本次 Pull Request 在 CUTLASS 仓库中:mdnice-bold-italic[新增独立的 `cutlass_compiler` 编译器栈，将 CuTe 布局代数以 MLIR 方言形式系统化表达，并建立从 CuTe 高层代数、Base 伪方言边界到 LLVM/NVVM 目标代码的分层 lowering 架构。]

该变更一次性地新增了编译基础设施的架构性：

- `cute` 方言负责建模形状、步长、坐标、布局、tile、swizzle 与 composed layout 等代数对象；
- `base` 层作为由上游 MLIR 方言构成的伪边界，隔离 CuTe 语义并承接后续 LLVM/NVVM 转换。

实现层面，PR 新增 TableGen 类型、属性与操作定义，接入 `cutegen` 作为类型推导与代数计算后端，实现 `cute-fold-static`、`cute-expand-ops`、`cute-to-base`、`base-prepare`、`one-shot-convert-to-llvm`、`attach-nvvm-target` 与 `emit-gpu-binary` 等关键 Pass，并提供 `cute-opt`、`base-opt`、`cutlass-compiler` 三类工具。

::: block-1其长期价值在于将原先隐式存在于模板元编程中的 CuTe 代数显式化为可验证、可测试、可优化的 IR 体系，:mdnice-highlight[为后续 DSL、JIT、GPU kernel pipeline 与 MLIR 生态集成提供基础。]
:::

## 本文目录

- 一、CuTe 代数进入 MLIR 的整体变更图景
  - 1.1 从模板抽象到 IR 抽象的转变
  - 1.2 变更规模与工程性质
  - 1.3 架构角度
- 二、架构与设计影响
  - 2.1 Cute/Base 双层边界降低耦合
  - 2.2 cutegen 成为类型推导与代数计算后端
  - 2.3 静态信息与动态叶的稀疏运行时表示
- 三、关键代码实现
  - 3.1 Cute Pass 定义：从高层代数展开到 base 边界
  - 3.2 类型转换：CuTe 类型到 sparse-flat LLVM 表示
  - 3.3 Cute 到 Base 的完整转换目标
- 四、测试、工具链与工程可维护性
  - 4.1 三类工具各司其职
  - 4.2 Base Pipeline 承接 LLVM/NVVM Lowering
  - 4.3 矩阵化测试覆盖
- 总结：整体判断与后续关注点
  - 1. 主要收益
  - 2. 主要风险
  - 3. 最终评价

![文章插图](/articles/2026-08-03-nvidia-cutlass-pr-3426-cute-07234f69/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、CuTe 代数进入 MLIR 的整体变更图景

### 1.1 从模板抽象到 IR 抽象的转变

> 该 PR 的核心是在 CUTLASS 中新增一套面向编译器的 CuTe 表示层。

`cutlass_compiler/README.md` 明确将 `cute` 方言定位为「CuTe layout algebra and NVIDIA GPU kernel programming」的 MLIR 表达，并将 `base` 定义为非独立 dialect、但语义边界清晰的上游 MLIR dialect 集合。

这意味着 CuTe 中原本依赖 C++ 类型系统、模板推导与编译期计算表达的 shape、stride、layout、tile、composition、partitioning 等概念，被重新映射为 MLIR 类型、属性和操作。此后编译器可在 IR 层面观察、转换和验证 CuTe 代数，而不只依赖 C++ 模板实例化结果。

### 1.2 变更规模与工程性质

PR 当前状态为 open，包含 1 个提交：

- 新增约 **124,603 行代码👀、475 个文件，无代码删除**。

文件规模表明它是完整子系统导入，覆盖：CMake 构建脚本、文档、方言 ODS 定义、C++ Pass、cutegen 库、opt 工具、LIT 测试、gtest 单元测试、集成测试与 Pipeline 测试。

该变更风险在于新增编译器栈的可维护性、构建复杂度，以及后续与主线 CUTLASS 的集成边界。

### 1.3 架构角度

> 从架构角度看，该 PR 为 CUTLASS 引入了**可编译的 CuTe IR 中间层**。

短期来看，CuTe lowering pipeline 的可测试性为开发者提供了快速验证和迭代优化的路径，显著提升了底层算子适配与调试效率。

中期则通过构建可组合的 MLIR Pass 体系，实现模块化、可复用的编译流程设计，支持灵活插入、替换与协同优化各类变换逻辑，增强框架的扩展性与维护性。

::: block-1
:mdnice-highlight[长期目标是推动 CUTLASS 从一个以手写模板为核心的高性能 GPU 算子库，演进为一个兼具 DSL 支持、自动优化能力与多后端目标代码生成能力的通用编译基础设施]，使其不仅能服务传统 CUDA 高性能计算场景，还可**支撑新兴领域如 AI 编译器、领域专用加速器及异构计算平台的深度集成与定制化开发**，从而在编译器栈中承担更核心的中间表示与优化枢纽角色。
:::

## 二、架构与设计影响

### 2.1 Cute/Base 双层边界降低耦合

- `cute` 层：负责完整表达 CuTe 代数；
- `base` 层：承接上游标准 MLIR 方言，严格禁止 Cute 语义泄漏。

`base-opt` 不会注册 cute dialect，一旦 base 层代码意外出现 `!cute.*` 类型或 `cute.*` 操作，会在解析阶段直接失败。 该设计把「CuTe 语义是否完成 lowering」从人为约定升级为工具层面硬约束。

分层架构分工：

1. `cute-opt`：专注方言级转换、调试与单元测试；
2. `base-opt`：校验 lowering 结果彻底脱离 Cute 高层语义；
3. `cutlass-compiler`：面向用户的端到端入口，串联两层完整能力。

### 2.2 cutegen 成为类型推导与代数计算后端

PR 引入 `cutlass_compiler/cutegen`，以 C++ 库承载全套 CuTe 代数逻辑：代数对象构造、字符串解析、静态/动态叶分析、layout 组合、coalesce、tiling 乘除等核心运算。

MLIR 方言不重复实现领域代数，通过 `cutegen` 提供的 `from_string`、`to_string`、`get_llvm_type`、`build_value`、`collect_scalar_values` 接口完成类型与数值转换。

分工逻辑：

- MLIR：IR 结构校验、Pass 调度、转换管线；
- cutegen：CuTe 领域数学运算。

代价：MLIR 方言正确性高度依赖 cutegen 语义稳定，需要充足测试覆盖。

### 2.3 静态信息与动态叶的稀疏运行时表示

编译器栈核心优化思路：**静态 CuTe 信息存放于类型/属性，运行时仅保留动态叶。** 完全静态的 CuTe 类型可 lowering 为空 LLVM struct；动态叶采用 sparse-flat 形式映射为 LLVM struct 字段。 优势：缩减运行时 IR 体积，大量布局计算前移至编译期执行。

## 三、关键代码实现

### 3.1 Cute Pass 定义：从高层代数展开到 base 边界

文件路径：`cutlass_compiler/cute_ir/include/cute_ir/Conversion/CuteToBase/Passes.td`

```
def CuteExpandOps : Pass<"cute-expand-ops"> {  let summary = "Expand selected cute ops via dialect conversion patterns";  let dependentDialects = [    "::mlir::arith::ArithDialect",    "::mlir::math::MathDialect",    "::mlir::scf::SCFDialect",  ];}
```

**改动意图** 将 lowering 拆分为两阶段：先把复杂 CuTe 代数操作展开为构造器、算术、控制流；再彻底将剩余 Cute 类型与操作降至 base 方言集合。

**实现逻辑**

- `cute-expand-ops`：局部代数展开，保留 `cute.static`、`cute.get_scalars` 与构造类操作；
- `cute-to-base`：完整方言转换，输出 IR 不再包含 Cute 方言。

**上下游影响** 上游 Cute IR 可直接使用高层代数算子；下游 base / LLVM/NVVM Pass 只需处理 arith/scf/ub/LLVM/gpu 等标准 MLIR 方言，大幅简化后端开发。

### 3.2 类型转换：CuTe 类型到 sparse-flat LLVM 表示

文件路径：`cutlass_compiler/cute_ir/lib/Conversion/CuteToBase/CuteTypeConverter.cpp`

```
void configCuteToBaseTypeConverter(TypeConverter &converter) {  converter.addConversion([](Type ty) { return ty; });  converter.addConversion(convertRecvarType<ShapeType>);  converter.addConversion(convertRecvarType<StrideType>);  converter.addConversion(convertRecvarType<IntTupleType>);}
```

**改动意图** 解决 CuTe 原生类型无法直接进入 LLVM/base lowering 的问题，同时保证非 Cute 类型透明透传。

**实现逻辑** shape、stride、coord、tile、layout、composed_layout 通过 cutegen 映射为 LLVM struct；swizzle 仅为编译期对象，运行时映射为空 struct；动态叶由 materializer 在边界处处理。

**上下游影响** 函数签名、返回值、构造器、访问器复用同一套 TypeConverter；后续 `base-prepare`、`reconcile-unrealized-casts` 可统一清理类型转换残留。

### 3.3 Cute 到 Base 的完整转换目标

文件路径：`cutlass_compiler/cute_ir/lib/Conversion/CuteToBase/CuteToBase.cpp`

```
ConversionTarget target(context);target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, ub::UBDialect,                       LLVM::LLVMDialect, gpu::GPUDialect>();target.addIllegalDialect<CuteDialect>();target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
```

**改动意图** 强制 `cute-to-base` 输出彻底清除 Cute 方言操作，阻止高层语义泄漏进 base/LLVM 后端。

**实现逻辑** 将 `CuteDialect` 整体标记为非法方言；动态校验 `func.func`、`func.return`、`func.call` 签名合法性；为核心构造、静态值、标量提取、打印操作注册专属转换模式。

**上下游影响** 调用方可将该 Pass 视作 Cute 语义的正式边界；未来新增 Cute Op 若缺少 lowering Pattern，转换流程会直接报错，提升系统迭代安全性。

## 四、测试、工具链与工程可维护性

### 4.1 三类工具各司其职

PR 新增三套独立工具，职责隔离：

1. `cute-opt`：注册 Cute 方言与 Cute 相关 Pass，用于方言层调试与单元测试；
2. `base-opt`：**不注册 Cute 方言**，验证 base 层不存在 Cute 语义；
3. `cutlass-compiler`：加载完整编译管线，面向端到端编译场景。

相比单一工具，分层工具能够精准定位故障：解析失败、代数展开失败、base lowering 失败、NVVM 二进制生成失败分属不同阶段，便于排障。

### 4.2 Base Pipeline 承接 LLVM/NVVM Lowering

- `base-prepare`：组合 canonicalize、CSE、LICM、unrealized cast 清理，进入 `gpu.module` 内部执行优化；
- `one-shot-convert-to-llvm`：优先处理 `gpu.module` 下沉至 LLVM/NVVM，再处理 host module，保留 GPU 模块外壳用于生成目标文件；
- `attach-nvvm-target` + `emit-gpu-binary`：封装上游 MLIR GPU/NVVM Pass，默认目标 sm_80，支持输出 PTX / cubin / fatbin / LLVM bitcode。

### 4.3 矩阵化测试覆盖

> 整套测试体系覆盖：

- 类型/属性 parse-print 往返测试；
- 构造器、访问器、布局代数、尺寸索引、算术运算、分块与划分算子；
- 静态折叠生效、动态分支禁止折叠的行为校验；
- base prepare、LLVM 转换、NVVM Target 挂载、GPU 二进制生成；
- Host JIT 集成、Device 编译管线、结构化 LIT 测试；
- C++ gtest：类型唯一化、DataLayout Interface、对象构造行为验证。

匹配庞大新增代码体量的测试矩阵，是该 PR 具备合入潜力的核心支撑。

## 总结：整体判断与后续关注点

### 1. 主要收益

> 该 PR 最大价值：将 CuTe 代数显式建模为 MLIR IR。布局、分块、组合、swizzle、动态叶传播逻辑均可被 Pass 管线分析、变换与优化。

静态常量通过 `cute.static` 与类型属性完成编译期折叠，动态部分采用稀疏存储，降低 IR 冗余与运行时开销。 长期来看，这套 IR 相比 C++ 模板实例，更容易对接 Python DSL、自动调优流水线。

### 2. 主要风险

1. **规模风险**：单 PR 新增 475 个文件、12 万余行代码，评审负担巨大🤣；
2. **语义一致性风险**：MLIR 方言层与 cutegen 代数库需要长期保持语义对齐，否则类型推导、打印、lowering 行为会出现分叉；
3. **构建复杂度风险**：独立 LLVM/MLIR、bundled LLVM、GoogleTest、NVPTX、CUDA 运行环境等多条路径，需要持续稳定的 CI 覆盖。

### 3. 最终评价

> 总体而言，该 PR 是 CUTLASS 走向编译器化的关键基础设施提交。**它搭建起「Cute IR → cutegen → Base lowering → LLVM/NVVM 后端 → 工具链 → 完整测试矩阵」的闭环。**

若后续完成评审单元拆分、稳定 CI 配置、明确与现有 CUTLASS 主线的集成策略，该变更将拥有极高架构价值与长期演进潜力。

::: related-title
**相关推荐**
:::

::: related-links
- **[无 GPU 也能玩转 CuTe！tensor-layouts：一份纯 Python 的 CuTe 代数实现，让 GPU 张量布局变成可读、可玩、可视](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903658&idx=1&sn=dbba694188a91680d9149ca9a1596276&scene=21#wechat_redirect)**
- **[对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)**
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**

:::
