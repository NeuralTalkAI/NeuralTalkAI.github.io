关键词：MLIR、**昇腾编译**、方言设计、**Pass流水线**、:mdnice-bold-italic[算子降级]

> 很多人第一次接触 AI 编译器项目时，往往会先问两个问题：**它支持哪些算子，它最终能生成什么结果**。

这当然重要，但如果只盯着结果看，常常会错过真正决定一个编译系统上限的东西——**它是否拥有一套足够稳定、足够清晰、又足够贴近硬件的中间表示体系。**

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/01.webp)

::: image-caption
昇腾 BiSheng 编译器的整体接入与编译流程，整体分为三层结构。顶层为 AI 框架层，涵盖 MindSpore、PyTorch、TensorFlow 等主流深度学习框架；中层是生态开发语言层，包含 Triton、Tilelang、FlagTree/TLE、DLCompiler 等多种开发工具，两类上层输入均可统一向下接入底层 BiSheng 编译器。编译器内部以 AscendNPU IR 为核心中间表示，内置 HFusion Dialect 与 HIVM Dialect 两类专用方言完成算子优化与硬件适配，再通过转换模块将 AscendNPU IR 转为 LLVM IR，依托成熟 LLVM 体系完成后续编译，整套架构实现多框架、多开发语言统一接入，依托分层中间表示完成面向昇腾 NPU 的标准化编译链路，打通上层 AI 开发到底层硬件编译的完整通路。
:::

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/02.webp)

- **AscendNPU-IR: MLIR for Ascend, delivering comprehensive support**
- 代码：https://github.com/Ascend/AscendNPU-IR
- 分享等资源：https://ascendnpu-ir.gitcode.com/en/reference/talk_and_course.html

::: related-title
**相关推荐**
:::

::: related-links
- **[cann-learning-hub：八章教程、二十五篇博客、两大SKILL，撑起昇腾 NPU 开发者的完整知识图谱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902138&idx=1&sn=878d817c8462758c2c6e188739ad57c3&scene=21#wechat_redirect)**
- **[“几乎不可编译”到15.9倍加速！AscendCraft：面向 NPU 的内核生成方法，DSL引导转译弥合NPU内核生成鸿沟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898824&idx=1&sn=6de0d761ed4fa4421805471e80f4d12c&scene=21#wechat_redirect)**
- **[鹏城联合华为昇腾、中山大学提出AscendKernelGen：NPU 算子闭环生成与评估，复杂L2 Kernel编译成功率95.5%，正确率64.3](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896566&idx=1&sn=d920e9f8feed38e4734d18942d83897a&scene=21#wechat_redirect)**

:::

> `AscendNPU-IR` 恰恰属于后者。**它的价值，是试图把过去大量分散在经验、工具链、后端黑盒里的硬件知识，沉淀为一套分层的 IR 结构**，:mdnice-bold-italic[让高层算子语义、融合策略、数据搬运、同步约束和底层执行动作在同一条编译链中被显式表达、逐层收敛、持续优化]。理解这件事，才能真正看懂这个仓库为什么值得被认真阅读。

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/03.webp)

::: image-caption
这张示意图分为左右两大模块，左侧完整展示昇腾 CANN 软件栈分层架构，自上而下依次为 Ops、HCCL、AAL、GE 组件，Kernel 编程层、粉色高亮的 BiSheng 编译器、Runtime 运行时与底层 Driver 驱动；右侧单独拆解 BiSheng 编译器工作链路，上层是用户接口层，包含 Python 侧 Triton、各类第三方框架，以及 C/C++ 侧 Ascend C 三类开发入口，多路径统一转换至核心中间表示 AscendNPU-IR，再经由 Compile 编译环节完成编译处理，清晰体现 BiSheng 编译器作为 CANN 核心编译模块，承接多语言、多框架开发输入，通过专属 IR 完成统一编译流程的整体逻辑，直观呈现昇腾 NPU 完整开发编译软件栈层级与数据流。
:::

## 本文目录

::: related-links
- 一、快速上手：先用起来，再理解它为什么这样设计
- 二、先别急着看细节，先搞清楚这个仓库到底想解决什么问题
- 三、仓库结构其实已经把它的设计哲学写在了目录里
- 四、真正的骨架在这里：方言注册暴露了项目的分层核心
- 五、这条编译链最关键的思想，是把 lowering 做成一个分阶段收缩过程
- 六、HIVM 是整个项目最值得关注的硬件着陆层
- 七、HACC 的价值在于把“编译元信息”变成了 IR 内部语义
- 八、Scope 与 InlineScope：它处理的是优化边界
- 九、CanonicalizeModule：真正成熟的工程，往往体现在这些不起眼的小 Pass 上
- 十、Python 编译入口是生态落地的重要接口
- 十一、从 CMake 开关也能看出，这个项目是面向真实演进环境设计的
- 十二、Pass 管理与调试设施，体现的是“可观测性”意识
- 十三、把前面的线索合在一起看，AscendNPU-IR 的核心能力其实非常清晰
- 十四、当然，这样的系统也天然面临挑战
- 结语：它真正做成的，是一种面向 AI 硬件时代的编译器组织方式

:::

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/03.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：先用起来，再理解它为什么这样设计

> 如果只是想最快体验这个项目，其实仓库已经给出了相当简洁的入口。它并没有把全部复杂度暴露给用户，而是先提供了一个可以直接上手的 Python 编译接口。这个接口的存在很重要，因为它意味着 `AscendNPU-IR` 并不是只为编译器工程师准备的底层库，它同时也在为上层框架、自动化流水线和调试工具提供统一调用面。

最短安装命令如下：

```
pip install ascendnpu-ir
```

安装之后，可以直接在 Python 中调用编译接口。README 和 wheel 文档里给出的最核心用法，本质上就是把一段 MLIR 文本交给编译器，然后指定输出文件和若干编译选项。最小可感知的体验大致如下：

```
import ascendnpuirmlir_str = """module {  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>, %arg1: memref<16xi16, #hivm.address_space<gm>>, %arg2: memref<16xi16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc<function_kind DEVICE>} {    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>, memref<16xi16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)    return  }}"""res = ascendnpuir.compile(    mlir_str,    output_path="example.o",    option=["-enable-hivm-compile=true"],)print(res)
```

如果需要从源码构建，则 README 中给出的最短路径是：

```
cd build-tools./build.sh./build_wheel.sh
```

如果还想查看更完整的安装说明、文档构建和 E2E 示例，可以继续参考仓库中的 README、`docs` 文档站，以及 `bishengir/test/Integration` 目录下的例子。

::: block-1
之所以先从这个入口讲起，是因为它直接揭示了一个核心事实：项目对外暴露的是一个“输入 MLIR、输出 Ascend 目标产物”的编译能力，但这个简单入口背后是一整条分层中间表示和 Pass 组织起来的复杂流水线。理解这层落差，是读懂这个项目的第一步。
:::

## 二、先别急着看细节，先搞清楚这个仓库到底想解决什么问题

> 从仓库 README 的介绍来看，`AscendNPU-IR` 被定义为一个构建在 MLIR 之上的 AscendNPU Intermediate Representation。表面上看，这似乎只是“为昇腾做了一套中间表示”。但如果把它说得更准确一些，**它真正试图解决的问题其实是：怎样在 Ascend 的编译过程中，保留足够多的高层语义，让融合、调度、地址空间、同步和性能优化拥有稳定承载面，同时又能在适当时机把这些抽象逐步收敛为底层硬件动作。**

这件事听起来抽象，但本质上非常现实。AI 编译器最难的地方，是“翻译时丢掉了什么信息”。如果你过早把高层计算结构打平成底层指令，就会丢掉融合空间、调度弹性和结构化分析能力；可如果你始终停留在非常抽象的 linalg 或 tensor 表达层，又无法自然承接硬件真实存在的片上存储、DMA、同步、对齐和流水化需求。**所以一个成熟的编译中间层，必须在“抽象足够高”和“硬件足够近”之间，找到一条可以逐层过渡的路线。**

**这正是 `AscendNPU-IR` 的核心方法：它不把编译过程理解成一次性翻译，而把它拆成多个有边界、有职责的语义层**。每一层保留特定信息，每一轮 Pass 只做一部分语义收缩，从而把原本容易混在一起的问题分开处理。这样一来，高层可以关注算子结构与融合，中层可以关注目标相关转换，底层则专注于真正贴近硬件的数据运动与执行组织。这个分层思想，几乎贯穿了整个仓库。

## 三、仓库结构其实已经把它的设计哲学写在了目录里

> 仓库 README 里对主目录结构有一个非常简练的概览。**最关键的部分都集中在 `bishengir` 目录中，它下面又分成 `include`、`lib`、`tools`、`test`、`python` 等多个子模块**。对于编译器项目来说，这种布局本身就是一种态度：它把声明、实现、工具入口、测试和语言绑定分别组织起来，说明项目一开始就朝“长期维护的基础设施”方向设计。

- `include` 更像系统的接口层，放着各种 dialect、conversion、interfaces 和 pass 的声明；
- `lib` 则是所有关键行为的实现所在，真正的方言逻辑、转换逻辑和变换逻辑都在这里；
- `tools` 是对外可执行工具；`python` 负责更友好的上层调用方式；
- `test` 则告诉你这个项目是有一整套可验证的行为边界。

对于读代码的人来说，正确的阅读姿势是先接受这样一个事实：`AscendNPU-IR` 是一个完整的编译中间层工程。

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/04.webp)

::: image-caption
昇腾编译体系两大核心分层抽象模块，上层为 HFusion 多维融合抽象，依托 Linalg 基础构建三层处理逻辑：Transformation Layer 完成 arith、math、torch 等多类算子格式向 HFusion 的转换；Preprocessing 负责数据类型规范化、算子化简与拆解；Fusion Layer 实现维度传播、展平、自动调度与主机分块等算子融合优化。下层是面向昇腾硬件高层抽象的 HIVM，依次包含三层硬件映射编译：Core Mapping Compilation 管理核间通信、同步并行与显存分块；On-Chip Memory Mapping 完成片上存储空间、格式推导与地址分配；Processing Unit Mapping 实现多计算单元流水线同步与向量化优化，整套分层架构自上而下完成算子融合优化到昇腾 NPU 硬件资源精细化映射的完整编译流程。
:::

如果一定要用一个容易理解的比喻来形容它，那么可以把它想象成一条多工位装配线。**高层算子语义像是初始设计图，HFusion 更像还保留结构意图的半成品工艺单，HIVM 则像真正能指挥流水线工位动作的工艺指令，而 Pass 是流水线中一个个负责切割、组装、搬运和校准的工站**。最终工具链要做的，是沿着这条已经结构化的工艺链，把程序一步步锻造成硬件可执行产物。

## 四、真正的骨架在这里：方言注册暴露了项目的分层核心

> 一个 MLIR 项目最值得先看的，不是某个花哨的优化 Pass，而是它到底注册了哪些 dialect。因为 dialect 决定的是这个系统允许自己“思考什么”。`AscendNPU-IR` 在这方面信息非常集中，相关入口就在 `bishengir/include/bishengir/InitAllDialects.h`。

```
// 来源：bishengir/include/bishengir/InitAllDialects.hnamespace bishengir {inline void registerAllDialects(mlir::DialectRegistry &registry) {  registry.insert<mlir::annotation::AnnotationDialect,                  mlir::hacc::HACCDialect,                  mlir::hfusion::HFusionDialect,                  mlir::hivm::HIVMDialect,                  mlir::mathExt::MathExtDialect,                  mlir::scope::ScopeDialect,                  mlir::symbol::SymbolDialect,                  bishengir::memref_ext::MemRefExtDialect>();#if BISHENGIR_ENABLE_TORCH_CONVERSIONS  registry.insert<mlir::torch::Torch::TorchDialect,                  mlir::torch::TorchConversion::TorchConversionDialect,                  mlir::torch::TMTensor::TMTensorDialect>();#endif#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)  mlir::hivm::registerBufferizableOpInterfaceExternalModels(registry);  mlir::annotation::registerBufferizableOpInterfaceExternalModels(registry);  mlir::hfusion::registerTilingInterfaceExternalModels(registry);  mlir::hfusion::registerBufferizableOpInterfaceExternalModels(registry);  mlir::hfusion::registerDecomposeInterfaceExternalModels(registry);  mlir::hivm::registerTilingInterfaceExternalModels(registry);  mlir::scope::registerBufferizableOpInterfaceExternalModels(registry);  bishengir::tensor::registerTilingInterfaceExternalModels(registry);  bishengir::bufferization::registerTilingInterfaceExternalModels(registry);#endif}
```

这段代码非常值得慢慢看。首先，它清楚地说明这个项目并不是只有一个所谓“Ascend dialect”，而是有一组相互配合的语义层：`HACC`、`HFusion`、`HIVM`、`Scope`、`Symbol`、`MemRefExt`、`Annotation` 等等。它们彼此不是替代关系，而是分工关系。`HACC` 更像编译入口和函数类别等元信息的承载层，`HFusion` 更强调融合和中高层算子表达，`HIVM` 明显贴近底层执行语义，`Scope` 管的是区域和边界，`Symbol` 负责动态符号语义，`MemRefExt` 与 `Annotation` 则在数据布局和附加语义上补齐支撑能力。

更关键的是，这里不只是注册 dialect，还注册了大量 interface external models，例如 `BufferizableOpInterface`、`TilingInterface`、`DecomposeInterface`。这说明项目并不是把优化规则写死在某几个 Pass 里，而是主动把自己的方言纳入 MLIR 更通用的优化框架中。:mdnice-bold-italic[换句话说，它是在尽可能复用 MLIR 的结构化能力。这种做法的意义非常大，因为一个编译器项目真正能否长期演化，往往不取决于你写了多少自定义代码，而取决于你是否把系统接到了足够稳定、足够抽象的公共接口层上。]

这里还有一个非常重要的信号：Torch 相关 dialect 的条件注册。它说明 `AscendNPU-IR` 并不想把自己限制在孤立的中间层位置，而是在为更高层生态接入留口子。对于 AI 编译系统来说，这几乎决定了它是“内部组件”还是“平台级组件”。能接前端生态，才能真正变成一条通路，而不是一个封闭技巧集合。

## 五、这条编译链最关键的思想，是把 lowering 做成一个分阶段收缩过程

> 理解 `AscendNPU-IR` 的第二个关键切口，是看它到底如何定义 conversion pass。相关信息集中在 `bishengir/include/bishengir/Conversion/Passes.td` 中。这个文件乍看像配置，实际上几乎就是整条编译链的路线图。

```
// 来源：bishengir/include/bishengir/Conversion/Passes.tddef ConvertArithToAffine : Pass<"convert-arith-to-affine"> {  let summary = "Convert arith ops to affine ops";}def ConvertArithToHFusion : Pass<"convert-arith-to-hfusion"> {  let summary = "Convert arith ops to HFusion ops";}def ConvertMathToHFusion : Pass<"convert-math-to-hfusion"> {  let summary = "Convert math ops to HFusion ops";}def ConvertLinalgToHFusion : Pass<"convert-linalg-to-hfusion"> {  let summary = "Convert linalg ops to HFusion ops";}def ConvertGPUToHFusion : Pass<"convert-gpu-to-hfusion"> {  let summary = "Convert GPU ops to HFusion ops";}def ConvertHFusionToHIVM : Pass<"convert-hfusion-to-hivm"> {  let summary = "Convert HFusion ops to HIVM ops";}def ConvertHIVMToStandard : Pass<"convert-hivm-to-std", "ModuleOp"> {  let summary = "Convert the operations from the hivm dialect into the Standard dialect";}def ConvertTensorToHFusion : Pass<"convert-tensor-to-hfusion"> {  let summary = "Convert tensor ops to Linalg/HFusion ops";}def ConvertTensorToHIVM : Pass<"convert-tensor-to-hivm"> {  let summary = "Convert tensor ops to HIVM ops";}
```

这一组名字已经把项目的主线写得非常清楚了。它允许多种起点——`Tensor`、`Arith`、`Math`、`Linalg`、`GPU`——向中层语义汇聚，然后再由 `HFusion` 进一步收敛到 `HIVM`。

::: block-1
这意味着项目把 lowering 理解为一个“逐步失去抽象、逐步获得确定性”的过程。高层 IR 在开始阶段拥有更强结构信息，越往后走，结构会减少，但硬件相关细节会增加。最终，程序是开始明确“在哪个地址空间做、以怎样的数据搬运顺序做、需要什么同步、如何映射到底层执行单元”。
:::

这条路线的高明之处，在于它避免了两个常见极端。

- 第一个极端是过早贴近硬件，导致高层优化空间还没用完就被榨干；
- 第二个极端是永远停留在很抽象的中间层，到了真正要映射硬件时信息不足。

:mdnice-bold-italic[`AscendNPU-IR` 显然不接受这两个极端，它通过 `HFusion` 和 `HIVM` 之间的分工，给自己搭了一段缓冲带。前者负责保留较高层的融合表达，后者负责落到更接近 Ascend 硬件执行的语义上。] 于是，编译器不必在“抽象”和“落地”之间一次性跳跃，而是可以分阶段压缩语义。

如果把这条链条说得再形象一点，那么 `HFusion` 更像在描述“这组操作应该作为一个更大的融合计算整体来看”，而 `HIVM` 则开始回答“那它们在昇腾上应该如何搬、如何算、如何同步”。这两层之间的区分，恰恰是现代 AI 编译器的难点所在，而 `AscendNPU-IR` 最值得肯定的地方，就是它没有回避这个难点，而是明确地把它建成了系统的一部分。

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/05.webp)

::: image-caption
IR 中间表示与 Toolchain 工具链两大板块，清晰呈现昇腾 BiSheng 编译器的底层编译转化链路。左侧 IR 层顶层为 AscendNPU IR，内部封装 HFusion Dialect 算子融合方言与 HIVM Dialect 硬件映射方言；下方设有转换模块，负责将 AscendNPU IR 转换为标准 LLVM IR，搭建起昇腾专属中间表示与通用底层 IR 的桥梁。左侧 IR 体系分别对应右侧两类编译工具链：完整 AscendNPU IR 流向 bishengir-compile 工具，聚焦上层算子融合编译；硬件相关 HIVM Dialect 链路对接 hivmc 工具，专门处理硬件层编译优化。整张图直观区分中间表示分层与配套编译工具，展现从双层定制 Dialect、IR 转换到两类专用编译工具的完整映射关系，梳理出昇腾 NPU IR 体系对应的工具链分工逻辑。
:::

## 六、HIVM 是整个项目最值得关注的硬件着陆层

> 如果说 `HFusion` 还保留着较强的高层算法意味，那么 `HIVM` 的角色就完全不同了。它是整个仓库里最接近“硬件动作语义”的一层。从 Python 示例就能看出，**到了 `HIVM` 层，程序已经不再只说“我要做加法”，而是开始说“我要从 GM 把数据搬到 UB，在 UB 上做向量加法，再把结果写回 GM”。这已经超越了抽象算法视角，而是非常明确的存储层次和执行动作视角。**

从项目中的文档聚合文件也能看出，`HIVM` 的设计是完整而系统的，并非零散 op 的堆积。

```
// 来源：bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDoc.tdinclude "bishengir/Dialect/HIVM/IR/HIVMBase.td"include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.td"include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.td"include "bishengir/Dialect/HIVM/IR/HIVMOps.td"include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td"include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.td"// Attributesinclude "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
```

这一小段包含的信息很多。`DMAOps` 说明数据搬运在这套语义里是一等公民，而不是附属动作；`SynchronizationOps` 表明同步是 IR 里可以分析和变换的显式语义；`VectorOps` 则说明向量计算被独立建模；`MacroOps` 又暗示存在更高粒度的组合级动作表达。:mdnice-bold-italic[对于 NPU 来说，这种设计极其关键，因为许多决定性能的因素根本是搬运是否合理、同步是否最少、片上 buffer 是否复用、地址空间是否合适。如果没有一个像 `HIVM` 这样的中间层承载这些信息，那么优化只能在后端黑盒里完成，调试和演化都会极其痛苦。]

也正因为如此，`HIVM` 的存在不是“方言数量增加了一个”这么简单，它实际上决定了这个编译系统有没有可能把硬件知识真正变成结构化资产。一个高水平的 AI 编译器，不只是要能把高层程序翻译到硬件上，更要能让工程师在中间层上“看见”性能问题在哪里、依赖关系如何变化、数据到底怎样流动。`HIVM` 就是在提供这种可见性。

## 七、HACC 的价值在于把“编译元信息”变成了 IR 内部语义

> 与 `HIVM` 相比，`HACC` 不那么像执行层，反而更像一层编译组织语义。它的作用是把函数属于 Host 还是 Device、是不是入口、是否要求始终内联这类信息显式编码出来。很多时候，这类信息如果处理不好，会变成系统最脆弱的隐式约定——靠文件位置、命名风格或外部脚本来识别。但在 `AscendNPU-IR` 里，它们被做成了正式的属性定义。

```
// 来源：bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.tddef HACC_HOST : I32EnumAttrCase<"HOST", 1>;def HACC_DEVICE : I32EnumAttrCase<"DEVICE", 2>;def HACC_FuncTypeEnum : HACC_I32Enum<"HACCFuncType", "HACC Function Category", [  HACC_HOST,  HACC_DEVICE]> {}def HACC_FuncTypeAttr : HACC_Attr<"HACCFuncType", "function_kind"> {  let parameters = (ins EnumParameter<HACC_FuncTypeEnum>:$function_kind);  let assemblyFormat = "`<` params `>`";}def HACC_ToLLVMIRTranslateAttr : HACC_I32Enum<"HACCToLLVMIRTranslateAttr", "", [  I32EnumAttrCase<"ENTRY", 0, "hacc.entry">,  I32EnumAttrCase<"MIX_ENTRY", 1, "hacc.mix_entry">,  I32EnumAttrCase<"ALWAYS_INLINE", 2, "hacc.always_inline">,]> {  let cppNamespace = "::mlir::hacc";}
```

这类设计看似平常，其实非常重要。因为编译器中的很多关键决策都依赖这些“元信息”：

- 一个函数是不是设备入口，直接影响调用边界和代码生成组织；
- Host/Device 的类别会影响后续归属和 lowering 路径；
- `always_inline` 则会改变优化器如何对待边界。

::: block-1
:mdnice-bold-italic[把这些信息做成可打印、可检查、可匹配、可序列化的 IR 属性，本质上是在把原本容易散落在外部约定里的知识收回到系统核心中]。这样，Pass 就不用靠猜，也不用依赖脆弱的命名协议，而是可以直接在结构化 IR 上工作。**这种“把编译知识语义化”的思路，是 MLIR 真正厉害的地方，而 `AscendNPU-IR` 显然在有意识地沿着这条路线走。**
:::

## 八、Scope 与 InlineScope：它处理的是优化边界

> 接下来值得看的，是一个看起来不算“宏大”，但很能体现项目味道的实现：`Scope` dialect 对应的 `InlineScope` Pass。:mdnice-bold-italic[很多人理解“内联”时，只会想到减少函数调用开销，但在复杂 IR 系统里，内联真正重要的地方常常是：它在控制哪些结构边界该保留，哪些该打平，以便后续优化看见更大的上下文。]

`AscendNPU-IR` 中这部分代码相当直接。相关核心逻辑在 `bishengir/lib/Dialect/Scope/Transforms/InlineScope.cpp`。

```
// 来源：bishengir/lib/Dialect/Scope/Transforms/InlineScope.cpptemplate <typename OpTy>class ExtractOpsFromBodyPattern :public OpRewritePattern<OpTy> {public:using OpRewritePattern<OpTy>::OpRewritePattern;LogicalResult matchAndRewrite(OpTy regionOp,                                PatternRewriter &rewriter) const override {    if (regionOp.getNoInline())      return failure();    assert(regionOp->getNumRegions() == 1 &&           regionOp.getRegion().hasOneBlock() &&           "only handle case of ScopeOp with single block");    Region &region = regionOp.getRegion();    Block &block = region.front();    auto opsToMove = llvm::make_range(block.begin(), std::prev(block.end()));    for (Operation &op : llvm::make_early_inc_range(opsToMove)) {      rewriter.moveOpBefore(&op, regionOp);    }    for (auto [res, opr] : llvm::zip_equal(             regionOp.getResults(),             regionOp.getRegion().front().getTerminator()->getOperands())) {      rewriter.replaceAllUsesWith(res, opr);    }    rewriter.eraseOp(regionOp);    return success();  }};
```

这段代码的核心意思并不复杂：如果一个 `scope` 没有被标记为禁止内联，那么就把它的 body 中除 terminator 之外的操作全部挪到 `scope` 外部，再把 `scope` 的结果替换为 terminator 产出的真实值，最后删除 `scope` 本身。**也就是说，`scope` 在这里更像一种阶段性组织结构，而不是不可触碰的语义实体。编译器前期可以借它来组织某些局部结构，但当后续优化需要更大视野时，这个边界就会被主动拆掉。**

这件事的意义比“少一个 op”大得多。因为很多优化，比如融合、缓冲区复用、同步消除、调度重排，本质上都要求编译器尽可能看见更大的 producer-consumer 邻接关系和更完整的数据流。如果结构边界过早固化，优化空间就会被人为切断。:mdnice-bold-italic[反过来说，如果从一开始完全没有边界，IR 的组织性又会很差，不利于某些阶段的局部推理。因此，`Scope` 的存在体现的是一种非常成熟的策略：边界不是绝对的，而是服务于不同编译阶段的。合适的时候建立边界，合适的时候再拆除边界，这才是高质量 IR 系统对“结构”的真正理解。]

同一个文件中，`InlineScopePass` 还把这件事放进了更标准的 PassManager 流程里，而不是靠一次 ad-hoc 改写完成。

```
// 来源：bishengir/lib/Dialect/Scope/Transforms/InlineScope.cppvoid InlineScopePass::runOnOperation() {auto moduleOp = getOperation();PassManager pm(moduleOp->getContext());if (forceInline) {    moduleOp.walk([](scope::ScopeOp op) { op.setNoInline(false); });  }  pm.addPass(std::make_unique<ExtractScopeBodyPass>());  pm.addPass(createInlinerPass());if (failed(pm.run(moduleOp)))    return signalPassFailure();}
```

这意味着 `AscendNPU-IR` 是先把自定义区域抽平，再借助 MLIR 标准 `inliner` 做后续处理。换句话说，**它把定制需求和上游公共基础设施很好地拼接在了一起**。这种做法的价值，让系统更容易保持一致性。

## 九、CanonicalizeModule：真正成熟的工程，往往体现在这些不起眼的小 Pass 上

> 大型编译器项目里，最容易被忽视的一类代码，就是那些看起来不“炫技”的 canonicalization Pass。它们不负责做深奥的性能优化，却负责把 IR 维持在干净、统一、可预测的形态。

这类工作如果做得不好，后面的 Pass 就会不停面对结构上的例外情况。`AscendNPU-IR` 中的 `CanonicalizeModule` 就是这样一个非常典型、也非常说明工程成熟度的例子。

```
// 来源：bishengir/lib/Transforms/CanonicalizeModule.cppvoid runOnOperation() override {  ModuleOp moduleOp = getOperation();RewritePatternSet patterns(&getContext());  populateCanonicalizePatterns(patterns);if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))    signalPassFailure();  Region &outerRegion = moduleOp->getRegion(0);if (outerRegion.empty())    return;  Block &outerBlock = outerRegion.front();if (outerBlock.getOperations().size() != 1)    return;  Operation &firstOp = outerBlock.front();  ModuleOp innerModule = dyn_cast<ModuleOp>(firstOp);if (!innerModule)    return;  forwardModuleAttributes(innerModule, moduleOp);  Region &innerRegion = innerModule->getRegion(0);if (innerRegion.empty()) {    innerModule.erase();    return;  }  Block &innerBlock = innerRegion.front();  outerBlock.getOperations().splice(outerBlock.begin(),                                    innerBlock.getOperations());  innerModule.erase();  runOnOperation();}
```

这段代码做的事很简单：如果顶层 module 里面只嵌套了一个内层 module，那么就把内层内容提到外面，把属性向外转发，然后继续递归拍平。:mdnice-bold-italic[这个逻辑听起来平凡，但在真实系统里极其重要。因为多阶段转换后，IR 结构经常会出现“合法但不理想”的状态，比如多包了一层 module、属性挂在了错误层级、某些 Pass 默认只看顶层导致行为不一致。假如这些结构性脏活没人管，后面的优化和 codegen 就会不断增加补丁式分支，系统会越来越难维护。]

::: block-1
所以，`CanonicalizeModule` 的意义不在于它有多聪明，而在于它体现了一个成熟工程的底层共识：:mdnice-bold-italic[编译链的稳定性，很多时候是被“结构越来越不整齐”慢慢拖垮的。一个项目如果愿意把这种清理型 Pass 认真写出来，通常说明它已经不满足于“跑通样例”，而是在认真维护一个可长期演进的 IR 生态。]
:::

## 十、Python 编译入口是生态落地的重要接口

> 前面已经提到，`AscendNPU-IR` 提供了 Python 包装。这里值得进一步看一眼它的实现，因为这能帮助我们理解项目如何看待“被调用”这件事。

在 `bishengir/python/wheel/ascendnpuir/compiler.py` 中，项目并没有把 Python 绑定做成一层非常厚的抽象，而是做了一个非常务实的包装：找到 `bishengir-compile` 二进制，检查必要的后端工具是否存在，然后把 MLIR 文本通过 stdin 交给编译器。

先看定位编译器二进制的逻辑：

```
# 来源：bishengir/python/wheel/ascendnpuir/compiler.pydef _get_compiler_path() -> Path:    package_dir = Path(__file__).parent    binary_name = "bishengir-compile"    binary_path = package_dir / "bin" / binary_name    if binary_path.exists():        return binary_path    binary_path = package_dir / binary_name    if binary_path.exists():        return binary_path    try:        result = subprocess.run(            ["which", "bishengir-compile"],            capture_output=True,            text=True,            check=False        )        if result.returncode == 0:            return Path(result.stdout.strip().split('\n')[0])    except Exception:        pass    raise FileNotFoundError(        f"Could not find bishengir-compile binary. "        f"Expected location: {binary_path}"    )
```

这里的设计相当朴素，但很现实。它先尝试 wheel 自带 `bin`，再尝试旧目录结构，最后退回到系统 PATH。也就是说，项目显然考虑了多种分发形态：打包好的 wheel、开发态部署以及系统级安装。这说明 Python 接口是真正考虑了被不同环境使用的场景。

再看真正的 `compile` 函数：

```
# 来源：bishengir/python/wheel/ascendnpuir/compiler.pydef compile(    input: str,    output_path: Optional[Union[str, Path]] = None,    option: Optional[List[str]] = None) -> CompileResult:    ifnot input ornot input.strip():        raise ValueError("Input MLIR string cannot be empty")    compiler_path = _get_compiler_path()    ifnot _check_hivmc_available():        raise FileNotFoundError(            "hivmc binary not found in PATH. "            "bishengir-compile requires hivmc to generate binary kernel files. "            "Please ensure hivmc is installed and accessible in your PATH."        )    cmd = [str(compiler_path)]    cmd.append("-")    if output_path:        cmd.extend(["-o", str(output_path)])    if option:        cmd.extend(option)    result = subprocess.run(        cmd,        input=input,        capture_output=True,        text=True,        check=False    )    if output_path:        final_output_path = str(output_path)    else:        final_output_path = "a.out"    return CompileResult(        returncode=result.returncode,        stdout=result.stdout,        stderr=result.stderr,        output_path=final_output_path    )
```

这一段最值得注意的地方有三个。

- 第一，它直接把 MLIR 文本经 stdin 喂给编译器，避免上层调用者必须落盘临时文件。这对于 notebook、自动化服务、前端框架集成都非常友好。
- 第二，它显式检查 `hivmc`，说明 `bishengir-compile` 是处在一条更完整工具链的关键节点上。
- 第三，它返回的是结构化 `CompileResult`，而不是简单打印日志，这说明项目认真考虑了程序化集成场景。换句话说，Python 这一层是在把底层编译能力转化为更高层系统容易消费的 API 资产。

## 十一、从 CMake 开关也能看出，这个项目是面向真实演进环境设计的

> 很多人读代码时会忽略构建系统，其实对编译器项目来说，`CMakeLists.txt` 往往非常能说明问题。`AscendNPU-IR` 的顶层构建脚本里，有几个选项尤其值得注意。

```
# 来源：CMakeLists.txtoption(LLVM_MAJOR_VERSION_21_COMPATIBLE "NPUIR build with LLVM 21 or later" OFF)if(LLVM_MAJOR_VERSION_21_COMPATIBLE)  add_definitions(-D__LLVM_MAJOR_VERSION_21_COMPATIBLE__)endif()option(LLVM_MAJOR_VERSION_22_COMPATIBLE "NPUIR build with LLVM 22" OFF)if(LLVM_MAJOR_VERSION_22_COMPATIBLE)  add_definitions(-D__LLVM_MAJOR_VERSION_21_COMPATIBLE__)  add_definitions(-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__)endif()option(LLVM_MAJOR_VERSION_20_COMPATIBLE "NPUIR build with LLVM 20" OFF)if(LLVM_MAJOR_VERSION_20_COMPATIBLE)  add_definitions(-D__LLVM_MAJOR_VERSION_20_COMPATIBLE__)endif()
```

这一段表明，项目是明确**把 LLVM/MLIR 大版本兼容视为一等现实问题的**。对 MLIR 生态稍有经验的人都知道，这个领域的 API 变化频率并不低，工业项目如果不正视版本适配，后续维护成本会非常大。这里显式提供 `LLVM 20/21/22` 的兼容开关，说明项目设计者一开始就把“跨版本演进”当成必须处理的问题，而不是临时补丁。

再看另外几个选项：

```
# 来源：CMakeLists.txtoption(BISHENGIR_BUILD_TEMPLATE  "Build the BiShengIR template." OFF)option(BISHENGIR_BUILD_STANDALONE_IR_ONLY  "Experimental feature. If set, only build IR definitions." OFF)option(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES  "Experimental feature. If set, enable A5 unpublished features." OFF)
```

这几个选项也很有意思。

- `BISHENGIR_BUILD_STANDALONE_IR_ONLY` 暗示项目的 IR 层有能力被裁剪出来单独构建，这说明其设计并没有死绑所有后续组件；
- `BISHENGIR_BUILD_TEMPLATE` 暗示这里不仅有 IR 和 Pass，还有模板型构建资源或产物组织能力；
- :mdnice-bold-italic[`BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES` 更是直接暴露出一个现实：这个项目是在开源与部分未公开硬件能力演进之间寻找平衡。这种信息非常宝贵，因为它让我们知道，`AscendNPU-IR` 的语义边界和工程取舍，很大概率来自真实硬件和真实产品约束，而不是纸面设计。]

![文章插图](/articles/2026-06-30-ascendnpu-ir-mlir-e93632a2/images/06.webp)

::: image-caption
AscendNPU IR（A5）完整编译分层体系，顶层提供 HFusion、TritonGPU 两条上层算子输入路径，二者统一下沉至核心 HIVM 硬件中间表示层。HIVM 顶层为 CV Core Mapping 编译模块，负责硬件核心基础映射；下层拆分为 HIVM-SIMD 与 HIVM-SIMT 两大优化分支：SIMD 侧包含 VF 融合、向量化及通用优化，面向向量计算单元做算子调度优化；SIMT 侧布局格式优化、共享内存分配与其他优化，适配并行线程存储调度。整套 HIVM 完成硬件层精细化优化后，统一向下转换为 LLVM IR，完整呈现从上层算子 IR 到昇腾硬件专属优化层，最终落地标准 LLVM 中间表示的编译链路，清晰区分两种硬件并行模型的差异化优化逻辑。
:::

## 十二、Pass 管理与调试设施，体现的是“可观测性”意识

> **大型编译系统真正难的，不只是 Pass 多，而是 Pass 太多之后，工程师不知道哪里出了问题**。`AscendNPU-IR`:mdnice-bold-italic[在 `PassManagerOptions.cpp` 中专门加了一层 IR 打印控制，这虽然只是个小部件，却能看出项目很重视调试可观测性。]

```
// 来源：bishengir/lib/Pass/PassManagerOptions.cppstatic std::set<std::string> kAllowedPassList = {    "hivm-inject-sync", "hivm-inject-block-sync", "hivm-graph-sync-solver"};struct PassManagerOptions {  PassNameCLParser printBefore{"bishengir-print-ir-before",                               "Print IR before specified passes"};  PassNameCLParser printAfter{"bishengir-print-ir-after",                              "Print IR after specified passes"};};
```

这背后体现的是一个非常务实的取舍。它对白名单中的特定 Pass 提供重点观测。:mdnice-bold-italic[为什么偏偏是同步注入和图同步求解相关 Pass？因为在 NPU 编译中，这类 Pass 通常最容易同时影响正确性和性能：同步加少了会错，加多了会慢，放错位置则可能又错又慢。因此，能在这些节点前后观测 IR，几乎就是性能分析和正确性调试的生命线。]

::: block-1
这也再次说明，`AscendNPU-IR` 的工程意识并不只停留在“定义方言”和“写 lowering 规则”上，它也在关心另一个常被忽略但极其关键的问题：出了问题之后，开发者有没有办法看清问题是怎么产生的。一个编译系统没有可观测性，再强的优化能力最终都会因为难以维护而失去价值。
:::

## 十三、把前面的线索合在一起看，AscendNPU-IR 的核心能力其实非常清晰

> 读到这里，基本可以把 `AscendNPU-IR` 的核心能力归纳为一条完整叙事了。

- 它是**一组围绕 Ascend 编译问题进行职责划分的 dialect 集群**；
- 其次，它不把 lowering 理解成一次性坠落，而是拆成高层表达、中层融合与低层硬件语义逐步收敛的过程；
- 再次，它在 `HIVM` 层显式承载数据搬运、同步、向量计算和地址空间等硬件关键知识，从而为性能优化提供真实抓手；
- 与此同时，它通过 `HACC`、`Scope`、`Annotation`、`Symbol` 等语义层，把入口属性、区域边界、附加标注和动态符号等信息纳入同一系统；
- 最后，:mdnice-bold-italic[它没有把自己封闭成一个只适合底层专家使用的组件，而是通过 Python wheel、Torch 相关接入、构建裁剪和调试设施，努力把这套能力变成可集成、可观测、可演进的平台型基础设施。]

::: block-1
如果一定要概括它的“根本意义”，可以说**“把 Ascend 编译中原本隐式的硬件知识和优化知识，逐渐语义化了”**。这件事极其重要。因为 AI 硬件生态变化快、算子形态变化快、前端框架变化也快。如果一个系统靠的是后端秘术和经验手工，短期也许能跑通，但长期一定会难以演化。:mdnice-bold-italic[只有当这些知识被提炼成中间表示、接口和 Pass 组织之后，它们才真正成为可继承、可复用、可升级的资产。]
:::

## 十四、当然，这样的系统也天然面临挑战

> 要公允地看待 `AscendNPU-IR`，也必须承认这类项目的挑战同样不小。

- **首先，方言一旦变多，边界维护就会变得困难**。什么语义应该放在 `HFusion`，什么又必须推迟到 `HIVM`；哪些属性属于 `HACC`，哪些该归 `Annotation`；哪些变换由 interface 驱动，哪些必须写成具体 Pass，这些都是需要长期维持纪律的事情。一旦边界变得模糊，系统就会逐渐出现语义重复、职责漂移和 Pass 互相踩踏的问题。
- **其次，lowering 时机永远是这类编译器最难的平衡项。抽象暴露得太早，会损失高层优化空间；暴露得太晚，则可能导致硬件映射信息不足**。`AscendNPU-IR` 已经通过 `HFusion` 和 `HIVM` 的分层给出了自己的答案，但真正难的地方在于，这个答案必须随着硬件代际、上游生态和模型形态变化而不断被重新校准。
- **再次，基于 MLIR 的工业项目几乎必然要长期承担版本兼容成本**。顶层构建文件已经很坦诚地把这个问题摆在桌面上。:mdnice-bold-italic[享受上游 MLIR 带来的抽象能力和生态红利，就必须接受它快速演化带来的维护负担。这种负担没有捷径，只能靠持续的工程纪律来消化。]

::: block-1
但也正因为这些挑战存在，`AscendNPU-IR` 当前展示出的分层设计、接口意识和工程化细节，才显得更有价值。它至少说明项目不是在逃避这些问题，而是在试图用系统化方法管理它们。
:::

## 结语：它真正做成的，是一种面向 AI 硬件时代的编译器组织方式

> 如果只从名字理解，`AscendNPU-IR` 似乎只是“昇腾的一个中间表示项目”。但当你把 README、方言注册、Pass 定义、`HIVM` 语义层、`Scope` 和 `CanonicalizeModule` 这些细节连起来看，就会发现它远不止于此。:mdnice-bold-italic[它真正完成的工作，是把原本容易散落在后端黑盒、工具链约定和人工经验里的知识，逐步拉回到一条可分析、可变换、可观测、可演化的 MLIR 编译链里。它让“高层算子如何变成硬件动作”这件事，不再只是终端工具给出的黑箱结果，而开始成为一条工程师能够理解、插手、优化和长期维护的语义路径。]

这也是为什么我会说**，`AscendNPU-IR` 最重要的地方，是“它把昇腾编译里的硬件知识和优化知识，沉淀成了中间层资产”**。在 AI 编译器这个领域，这种工作往往比单点性能技巧更重要。因为单点技巧会过时，某个后端规则会变化，某一代硬件特性会替换，但一套能不断吸收新知识、重新组织语义边界的中间层架构，却可以持续生长。

::: block-1
从这个意义上说，`AscendNPU-IR` 的价值并不只是服务一个仓库、一种工具或一条构建脚本，它代表的是一种更长期的判断：:mdnice-bold-italic[未来的 AI 硬件编译优化，不应该继续依赖不可解释的经验黑盒，而应该尽可能前移到结构化、分层化、可编排的中间表示系统中。而这，正是这个项目最值得认真对待的地方。]
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[cann-learning-hub：八章教程、二十五篇博客、两大SKILL，撑起昇腾 NPU 开发者的完整知识图谱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902138&idx=1&sn=878d817c8462758c2c6e188739ad57c3&scene=21#wechat_redirect)**
- **[“几乎不可编译”到15.9倍加速！AscendCraft：面向 NPU 的内核生成方法，DSL引导转译弥合NPU内核生成鸿沟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898824&idx=1&sn=6de0d761ed4fa4421805471e80f4d12c&scene=21#wechat_redirect)**
- **[鹏城联合华为昇腾、中山大学提出AscendKernelGen：NPU 算子闭环生成与评估，复杂L2 Kernel编译成功率95.5%，正确率64.3](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896566&idx=1&sn=d920e9f8feed38e4734d18942d83897a&scene=21#wechat_redirect)**

:::
