关键词：Google XLS、**高层次综合 (HLS)、软硬协同 (Co-design)、** DSLX语言、:mdnice-bold-italic[编译器架构]、即时编译 (JIT)

> 摩尔定律逼近物理极限，算力焦虑正席卷整个科技圈。既然通用处理器的“免费午餐”快吃完了，未来的出路在哪里？答案指向**专用硬件与软硬协同设计（Co-design）**。

但是，现实很骨感。懂底层硬件逻辑的大佬往往被传统的 EDA 工具链和繁琐的 Verilog 缠身，而掌握高级抽象逻辑的软件工程师，一看 Verilog 就头大。**有没有一种可能，让开发者像写现代软件代码一样，优雅、安全且高效地“编译”出底层硬件电路？**

![文章插图](/articles/2026-03-29-google-xls-verilog-fbda400b/images/01.webp)

- **XLS implements a High Level Synthesis toolchain that produces synthesizable designs (Verilog and SystemVerilog) from flexible, high-level descriptions of functionality. It is Apache 2 licensed.**
- 项目地址：https://google.github.io/xls/
- 代码：https://github.com/google/xls
- 论坛：https://github.com/google/xls/discussions
- 3000 字，阅读 10 分钟，**播客 13 分钟**

相关推荐

- [HotChips 2025 从摩尔定律到巨型内核：GPU 上机器学习系统优化的十年跃迁，Zhihao Jia 演讲解读](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891453&idx=1&sn=632b9abf4170c20af52b481ca8ec24e7&scene=21#wechat_redirect)
- [Next Decade AI Performance, Power, Area with Co-optimization](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889567&idx=1&sn=ea902cf74f98dccae87bf919610206ac&scene=21#wechat_redirect)
- [hls4ml：面向 FPGA AI 部署的开源模块化编译器，实现微秒级延迟、极致资源效率，支持 PyTorch/Keras/ONNX，适配 AMD/Intel/ASIC](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897821&idx=1&sn=6f4033c3aa72e8c73d802f921164d134&scene=21#wechat_redirect)

> 最近发现一个 Google 闷声开源的硬核项目——**XLS (Accelerated HW Synthesis)。** 它的终极目标，就是成为后摩尔定律时代（EoML）的通用开发 SDK！

::: block-1
**XLS (Accelerated HW Synthesis)** 是 Google 开源的一个处于快速开发阶段的**高层次综合（High Level Synthesis, HLS）工具链**。它的核心愿景是成为“摩尔定律终结（End of Moore's Law, EoML）时代”的软件开发工具包（SDK）。
:::

![文章插图](/articles/2026-03-29-google-xls-verilog-fbda400b/images/02.webp)

::: image-caption
这张图展示了Google XLS硬件综合工具链的完整端到端工作流：工具支持以DSLX硬件领域专用语言或C++作为输入，将二者统一编译为XLS中间表示（XLS IR），该中间表示还可导出为text.ir文件；XLS IR会进入优化流水线生成优化后的XLS IR，随后经调度（Scheduling）与代码生成（Codegen）环节，输出(System)Verilog网表，用于后续的硬件仿真（Simulation）与综合（Synthesis）；同时，优化前后的XLS IR还可对接多种辅助工具，包括DSLX解释器、IR解释器、基于LLVM JIT的快速功能仿真（可输出x86-64代码），以及全栈模糊测试、逻辑等价性验证、可视化等工具，以此保障硬件设计的正确性与性能，实现了从高级语言到硬件网表的高效开发与全流程验证。
:::

为了应对未来硬件专用化（Specialization）的趋势，软件和硬件工程师必须跨越领域的边界，进行更多的**软硬协同设计（Co-design）**。:mdnice-bold-italic[XLS 试图通过自动化、发挥软件工程师的作用以及利用机器算力来加速整个硬件开发的过程。]

## 本文目录

- 一、核心设计理念
- 二、核心架构与工具栈
  - A. 前端语言层：DSLX
  - B. 中间表示层：XLS IR
  - C. 优化与调度
  - D. 代码生成：Codegen 与 VAST
- 三、周边生态与核心工具
- 总结

![文章插图](/articles/2026-03-29-google-xls-verilog-fbda400b/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、核心设计理念

> XLS 允许开发者使用一种灵活的、高层次的功能描述语言来编写代码，最终能够将其自动综合为可综合的硬件设计（Verilog 和 SystemVerilog）。它的主要特点有：

- **软件风格的硬件开发**：使用 XLS 开发的硬件 IP 可以同时作为高效的“宿主软件（Host software）”运行。这意味着:mdnice-bold-italic[你可以用软件的习惯来写硬件，并且代码不仅可以生成硬件（Verilog），还能在软件环境或模拟器中以 Native 速度运行。]
- **严格的功能一致性**：XLS :mdnice-bold-italic[工具链提供了形式化验证（Formal Verification）工具，确保其作为软件运行时的表现]，与生成的硬件电路块在功能上是完全一致的。
- **纯函数与并发进程（Procs）**：XLS 支持纯连线 I/O 接口的（可选流水线的）函数（Functions），也支持有状态的并发进程（Concurrent processes / `proc`s），用于表达随时间推移的归纳和更复杂的通信接口。

## 二、核心架构与工具栈

XLS 的内部架构呈现为一个层次分明的编译栈（Stack）。从上层语言到底层硬件生成的链路如下：

### A. 前端语言层：DSLX

> 为了更契合硬件设计（描述数据流，而不是传统冯·诺依曼架构的指令），Google 设计了一种名为 **DSLX** 的领域特定语言（DSL）。

- 它在语法上**模仿了 Rust**，是一种不可变的（Immutable）、基于表达式的数据流语言。
- 包含了面向硬件的特性：比如支持任意位宽的整数、完全固定大小的对象、以及完全可分析的调用图。

下面是 DSLX 语法演示：一个真实存在的简单硬件加法器示例，它非常接近 Rust：

```
// xls/examples/tiny_adder.x// https://github.com/google/xls/blob/3c09a745f08de7ad06429f8044ab5b3b854550a2/xls/examples/tiny_adder.x# ![feature(type_inference_v2)]// 返回加法结果，类型拓宽以反映进位位 (carry bit)。fn tiny_adder(x: u1, y: u1) -> u2 {  let lhs = x as u2;  let rhs = y as u2;  lhs + rhs}#[test]fn test_exhaustive() {  assert_eq(tiny_adder(u1:0, u1:0), u2:0b00);  assert_eq(tiny_adder(u1:0, u1:1), u2:0b01);  assert_eq(tiny_adder(u1:1, u1:1), u2:0b10);  assert_eq(tiny_adder(u1:1, u1:0), u2:0b01);}
```

这段代码在软件里能直接被作为单元测试运行（`#[test]`），但在 XLS 栈中，它会被解析为底层硬件网表。

### B. 中间表示层：XLS IR

> DSLX 代码首先会被转换为 XLS 的中间表示（IR）。XLS IR 是对底层逻辑的一个非常纯粹的、静态单赋值（SSA）形式的图结构。编译器的大量优化工作都在这个层级进行。

在 `xls/ir/node.h` 中，定义了 IR 中最基础的节点概念。在硬件网表中，每一个操作（如 AND、OR、MUX）都是一个 `Node`，它们彼此之间通过操作数（`operands_`）和使用者（`users_`）相互连接。

```
// xls/ir/node.h#L83-L122// xls/ir/node.h url=https://github.com/google/xls/blob/3c09a745f08de7ad06429f8044ab5b3b854550a2/xls/ir/node.h#L83-L122// 高层 IR 中表示一个节点（代表一个表达式）的抽象类型class Node {public:virtual ~Node() = default;// 访问者模式：遍历该节点absl::Status Accept(DfsVisitor* visitor);Type* GetType() const { return type_; }Op op() const { return op_; }FunctionBase* function_base() const { return function_base_; }// 返回此节点使用的操作数序列（输入端口）absl::Span<Node* const> operands() const { return operands_; }// ... (省略部分方法)protected:  FunctionBase* function_base_;int64_t id_;  Op op_;  Type* type_;  SourceInfo loc_;std::unique_ptr<std::string> name_;  // 如果已分配名称则非空// 绝大多数节点具有 <= 2 个操作数，因此我们使用 InlineVector 尽量在栈上分配  absl::InlinedVector<Node*, 2> operands_;// 按 node_id 排序的该节点的使用者集合（输出端口去向）  absl::InlinedVector<Node*, 2> users_;};
```

从上述代码可以看出，XLS IR 是用 C++ 严谨地构建起来的，为了追求高性能，连存放拓扑结构的向量都使用了 `absl::InlinedVector` 来优化内存和缓存命中率。

### C. 优化与调度

- **Passes** （位于 `xls/passes/`）: 和 LLVM 类似，IR 被生成后，XLS 会执行一系列的 Pass 来对 IR 进行优化（例如死代码消除、常量折叠等）。
- **Scheduling** （位于 `xls/scheduling/`）: 硬件不同于软件的一点是“时序”。`scheduling` 模块中的算法负责决定每个 IR 操作应该在时钟周期的哪一个流水线阶段（Pipeline Stage）执行。

### D. 代码生成：Codegen 与 VAST

> 经过调度优化的 IR，最终要被翻译成电子自动化设计（EDA）工具能认识的语言（Verilog）。 XLS 构建了一个名为 **VAST（Verilog AST）** 的抽象语法树。

通过一些名为 `Generator` 的组件（比如 `PipelineGenerator` 负责流水线，`SequentialGenerator` 负责状态机 FSMs），XLS 会把 IR 转化为 VAST，并最终输出 `.v` 或 `.sv` 文件，这部分的实现位于 `xls/codegen/`。

## 三、周边生态与核心工具

> 除了上述主干链路，XLS 在架构设计上充分考虑了工程化、验证和仿真：

1. **JIT 编译器 (`xls/jit`)：** 基于 LLVM 的即时编译器（JIT）可以将 XLS IR 编译成本地机器码运行。这让 DSLX 的验证与执行能够达到 Native 速度。
2. **全栈 Fuzzer (`xls/fuzzer`)：** 硬件的 BUG 代价巨大，XLS 开发了一个多进程的模糊测试器。:mdnice-bold-italic[它随机生成 DSL 级别的程序，并交叉对比 DSL 解释器、IR 解释器、JIT、以及最终生成的 Verilog 在仿真器中的结果]。只要出现结果不一致，就能立即捕获工具链的问题。
3. **SMT 形式化验证 (`xls/solvers`)：** 使用著名的 Z3 求解器引擎。XLS 可以将 IR 转化为 SMT 逻辑表达式，用于进行形式化证明，比如检查 XLS IR 与 Netlist 之间是否存在逻辑等价性 / Logical Equivalence Check。
4. **C++ 前端 (`xls/contrib/xlscc`)：** 如果你不想学 DSLX，:mdnice-bold-italic[团队还开发了一个实验性的 C++ 前端，它能解析特定的 C++ 语法并直接输出成 XLS IR]。这极大地方便了原有 C++ HLS 代码库的迁移。

## 总结

> `google/xls` 的核心野心是将现代软件编译器的设计理念（强类型DSL -> 静态单赋值IR -> 多重Pass优化 -> AST生成器）完整地平移到硬件电路生成（Hardware Synthesis）中。

通过 DSLX 提供类 Rust 的安全开发体验，通过 `Node` 树管理底层的逻辑门元操作，并提供 JIT、Fuzzing 和 SMT Solvers 组成全方位的测试保证。:mdnice-bold-italic[对于希望在后摩尔时代通过软硬协同榨取算力极限的开发者而言，XLS 是一个非常具有前瞻性的基础设施箱。]

纵观整个 `google/xls` 项目，我们可以清晰地感受到 Google 在基础设施布局上的巨大野心。

::: block-1
它并不是简单地在造“另一个 HLS（高层次综合）工具”，而是**真正把现代软件工程的降维打击带到了硬件圈**。从带有 Rust 范儿的强类型安全前端（DSLX），到严谨的 SSA 形式中间表示（XLS IR），再到 JIT 即时编译、全栈自动化 Fuzzing 以及 SMT 形式化验证……XLS 几乎把现代编译器最精华的武器库全套搬了过来。
:::

**它带来的最大启示是：未来的硬件开发，必然走向敏捷化。** 在 AI 和专用加速芯片（DSA）全面爆发的今天，让懂业务逻辑的软件工程师能够安全地跨界、直接定义和生成硬件，是一件极具想象力的事情。:mdnice-bold-italic[XLS 虽然目前仍处于实验和快速迭代阶段，但它已经为我们勾勒出了“软件定义硬件”的清晰轮廓。]
