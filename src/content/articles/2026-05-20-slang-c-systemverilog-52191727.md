关键词：**SystemVerilog 编译器**、:mdnice-bold-italic[抽象语法树]、BumpAllocator 内存池、**延迟展开**、访问者模式

> 在芯片设计领域，SystemVerilog 既是硬件描述语言也是验证语言，其语法复杂度堪比 C++。长期以来，EDA 行业依赖闭源商业工具进行编译与分析，开源社区缺乏一个真正完整、高效的前端实现。

![文章插图](/articles/2026-05-20-slang-c-systemverilog-52191727/images/01.webp)

- **SystemVerilog compiler and language services**
- slang 是一个软件库，为 SystemVerilog 代码的词法分析、语法分析、类型检查和精化提供各类组件。它附带一个可执行工具，能够对任意 SystemVerilog 项目进行编译和静态分析，同时也旨在用作综合工具、模拟器、代码检查器、代码编辑器和重构工具的前端。
- 代码：https://github.com/MikePopoloski/slang
- 文档：https://sv-lang.com/
- 4000 字，阅读 20 分钟，**播客 18 分钟**

相关推荐

- [nvlabs/FVRuleLearner：把 LLM 写错的 SystemVerilog Assertion 沉淀成规则，让形式验证经验可学习、可检索、可复用](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900862&idx=1&sn=811ff41089435ba9d19053bd67480cc5&scene=21#wechat_redirect)
- [斯坦福、CMU 提出 VeriCoder：基于 LLM 的 RTL 代码生成，VerilogEval 上功能正确性 SOTA](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888243&idx=1&sn=76d2a20cfb1cfecc0a2c86d72ddfeaf7&scene=21#wechat_redirect)
- [像写软件一样造芯片：Google XLS 让软件工程师写Verilog，用编译器思维颠覆硬件开发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899137&idx=1&sn=c98294c41e5453f33c67ee1b3c826838&scene=21#wechat_redirect)

> slang 项目横空出世，以单人之力实现了对 IEEE 1800 标准的完整覆盖，在 ChipsAlliance 开源测试套件中跑出了最高合规率与最快速度。它不仅是一个编译器，更是一个可嵌入的库——你可以用它做语法检查、代码生成、LSP 语言服务器，甚至作为综合工具的前端。

::: block-1
本文将深入 slang 的源码架构，揭示其在词法分析、语法树构建、符号表管理与 AST 遍历等关键环节的精妙设计。
:::

## 本文目录

- 快速上手
- 一、架构总览
  - 1.1 模块化分层
  - 1.2 核心设计原则
- 二、词法分析：Lexer 的极致性能
  - 2.1 指针追逐式扫描
  - 2.2 BumpAllocator——编译器的内存引擎
- 三、语法树：从文本到结构的无损映射
  - 3.1 SyntaxTree——CST 的所有权容器
  - 3.2 Parser 的递归下降策略
- 四、符号系统：Symbol 与 Scope 的精妙编排
  - 4.1 Symbol——万物之基
  - 4.2 SymbolKind 的宏魔法
  - 4.3 Scope——延迟展开的作用域
- 五、表达式系统与常量求值
  - 5.1 Expression——类型驱动的绑定
  - 5.2 40+种表达式类型
- 六、AST 遍历：编译时多态的 Visitor 模式
  - 6.1 ASTVisitor——CRTP 驱动的零开销遍历
  - 6.2 makeVisitor——Lambda 组合的便捷 API
  - 6.3 Symbol 的 dispatch 实现
- 七、诊断系统：像 clang 一样的错误报告
  - 7.1 结构化诊断
- 八、工具链生态
- 总结

![文章插图](/articles/2026-05-20-slang-c-systemverilog-52191727/images/02.webp)

## 快速上手

> slang 支持 Linux、macOS 和 Windows，可直接下载预编译二进制，也可从源码构建：

```
# 从源码构建git clone https://github.com/MikePopoloski/slang.gitcd slangcmake -B buildcmake --build build -j# 编译完成后直接对你的SystemVerilog代码进行检查./build/bin/slang your_design.sv# 或通过Python绑定使用pip install pyslang
```

- 更多编译选项参考 Building[1]
- 命令行使用详见 User Manual[2]
- 库开发接口参考 Developer Guide[3]

你也可以直接在 在线 Playground[4] 体验解析与类型检查。

## 一、架构总览

### 1.1 模块化分层

> slang 的代码组织严格遵循编译器分层原则，每个阶段对应独立的目录：

| 层级 | 目录 | 职责 |
| --- | --- | --- |
| 文本管理 | `text/` | 源文件加载、SourceLocation、字符编码 |
| 词法分析 | `parsing/` | Lexer、Preprocessor、Token |
| 语法树 | `syntax/` | CST（具体语法树）节点、SyntaxTree |
| 语义分析 | `ast/` | Symbol、Scope、Expression、Compilation |
| 诊断系统 | `diagnostics/` | DiagCode、Diagnostic、格式化输出 |
| 基础设施 | `util/` | BumpAllocator、SmallVector、ThreadPool |
| 驱动层 | `driver/` | 命令行解析、文件加载编排 |

这种分层设计使得每一层都可以独立使用。例如你只需要预处理功能，直接调用`parsing`层即可，无需触及 AST。

### 1.2 核心设计原则

> slang 的设计哲学可以概括为四点：

- **容错优先**：即使源码有语法错误也要产出尽可能完整的树
- **零拷贝**：Token 直接指向源文件内存
- **延迟求值**：Scope 的成员只在首次访问时展开
- **arena 分配**：所有 AST 节点通过 BumpAllocator 统一管理生命周期

## 二、词法分析：Lexer 的极致性能

### 2.1 指针追逐式扫描

> slang 的 Lexer 采用最直接的字符指针遍历方式，没有使用状态机表驱动，而是用`peek()`/`advance()`组合实现零开销的字符扫描：

```
// 来源：include/slang/parsing/Lexer.hclass Lexer {private:    constchar* originalBegin;    constchar* sourceBuffer;  // 当前扫描位置    constchar* sourceEnd;    constchar* marker;        // 当前lexeme起始位置    // 核心指针操作——极简且零开销    void mark() { marker = sourceBuffer; }    void advance() { sourceBuffer++; }    char peek() const { return *sourceBuffer; }    bool reallyAtEnd() const { return sourceBuffer >= sourceEnd - 1; }    std::string_view lexeme() const {        returnstd::string_view(marker, lexemeLength());    }};
```

这种设计使得每个 Token 的生成只需要几次指针比较和移动，没有任何虚函数调用或动态分派开销。

### 2.2 BumpAllocator——编译器的内存引擎

> 所有 Token、Trivia 以及后续的语法树节点都通过 BumpAllocator 分配。这个分配器的核心思想极其简单：只分配，不释放单个对象，最终一次性释放整块内存：

```
// 来源：include/slang/util/BumpAllocator.hclass BumpAllocator {public:    // O(1)分配：只是移动指针    byte* allocate(size_t size, size_t alignment) {        byte* base = alignPtr(head->current, alignment);        byte* next = base + size;        if (next > endPtr)            return allocateSlow(size, alignment);  // 分配新段        head->current = next;        return base;    }    // 类型安全的构造——要求T必须平凡析构    template<typename T, typename... Args>    T* emplace(Args&&... args) {        static_assert(std::is_trivially_destructible_v<T>);        returnnew (allocate(sizeof(T), alignof(T))) T(std::forward<Args>(args)...);    }};
```

这是编译器领域的经典技巧：AST 节点一旦创建就不会被单独销毁，因此完全不需要`delete`。当编译结束时，整个 allocator 连同所有节点一起释放，避免了逐个析构的开销。这就像给编译器装了一个"内存快进键"。

## 三、语法树：从文本到结构的无损映射

### 3.1 SyntaxTree——CST 的所有权容器

> slang 与大多数编译器不同，它保留了完整的**具体语法树** （CST），而非直接构建 AST。CST 保留了所有空白、注释、括号等信息，使得源代码可以完美 round-trip：

```
// 来源：include/slang/syntax/SyntaxTree.hclass SyntaxTree {    SyntaxNode* rootNode;           // 语法树根节点    BumpAllocator alloc;            // 拥有所有节点的内存    Diagnostics diagnosticsBuffer;  // 解析期间产生的诊断    SourceManager& sourceMan;       // 源文件管理    // 最简单的入口——从文本创建语法树    static std::shared_ptr<SyntaxTree> fromText(        std::string_view text,        std::string_view name = "source");};
```

SyntaxTree 对象持有 BumpAllocator 的所有权，这意味着只要 SyntaxTree 存活，所有语法节点的内存就是安全的。这是一种**RAII 粒度的内存管理策略**。

### 3.2 Parser 的递归下降策略

slang 的 Parser 继承自 ParserBase，采用经典的递归下降方式。关键的设计决策是：即使遇到语法错误也要继续解析，尽可能产出完整的树。这使得 slang 可以用于 IDE 场景下的实时语法高亮与补全。

## 四、符号系统：Symbol 与 Scope 的精妙编排

### 4.1 Symbol——万物之基

slang 的所有语义实体都继承自`Symbol`基类。通过一个枚举`SymbolKind`来标识具体类型，配合模板`as<T>()`实现安全的向下转型：

```
// 来源：include/slang/ast/Symbol.hclass Symbol {public:    SymbolKind kind;          // 符号类型标识    std::string_view name;    // 符号名（指向源文件内存，零拷贝）    SourceLocation location;  // 源码位置    // 安全向下转型    template<typename T>    decltype(auto) as() {        SLANG_ASSERT(T::isKind(kind));        return *static_cast<T*>(this);    }    // 尝试转型，失败返回nullptr    template<typename T>    decltype(auto) as_if() {        if (!T::isKind(kind)) returnstatic_cast<T*>(nullptr);        returnstatic_cast<T*>(this);    }private:    const Scope* parentScope = nullptr;    // 所属作用域    const Symbol* nextInScope = nullptr;   // 同级链表的下一个    const syntax::SyntaxNode* originatingSyntax = nullptr;};
```

注意`name`是`string_view`——它直接引用源文件 buffer 中的字符串，没有任何拷贝。`nextInScope`形成一条侵入式链表，将同一 Scope 内的所有成员串联起来，避免了额外的容器开销。

### 4.2 SymbolKind 的宏魔法

slang 用 X-macro 技巧定义了超过 100 种符号类型，从基本类型（`ScalarType`、`PackedArrayType`）到复杂结构（`Instance`、`Subroutine`、`ClassType`），完整覆盖了 SystemVerilog 的所有语义构造：

```
// 来源：include/slang/ast/Symbol.h（简化）#define SYMBOLKIND(x) \    x(Root) x(Definition) x(CompilationUnit) \    x(Instance) x(InstanceBody) x(Package) \    x(Variable) x(Net) x(Subroutine) \    x(ClassType) x(EnumType) x(Parameter) \    /* ...总计100+种... */SLANG_ENUM(SymbolKind, SYMBOLKIND)
```

这种设计使得新增一种符号类型只需在宏列表中加一行，所有 dispatch 逻辑自动更新。

### 4.3 Scope——延迟展开的作用域

Scope 是 slang 中最关键的设计之一。它代表一个可以包含子符号的命名空间（模块、函数、begin-end 块等）。核心创新在于**延迟展开** （lazy elaboration）：

```
// 来源：include/slang/ast/Scope.hclass Scope {    mutableconst Symbol* firstMember = nullptr;    mutableconst Symbol* lastMember = nullptr;    mutablebool needsElaboration = false;    // 任何成员访问前，先确保已经展开    void ensureElaborated() const {        if (needsElaboration)            elaborate();  // 按需展开    }public:    // 获取成员列表——触发延迟展开    std::ranges::subrange<iterator> members() const {        ensureElaborated();        return {firstMember, nullptr};    }    // 名称查找    const Symbol* find(std::string_view name) const;};
```

这意味着如果你只解析了文件但从未访问某个模块的内部成员，slang 不会浪费时间去解析其内部符号。这对大型设计尤其重要——你可能有数千个模块但只实例化了其中几十个。

## 五、表达式系统与常量求值

### 5.1 Expression——类型驱动的绑定

Expression 的设计与 Symbol 类似，采用 kind+dispatch 模式。但表达式额外持有一个类型指针，实现了完整的类型推导：

```
// 来源：include/slang/ast/Expression.hclass Expression {public:    ExpressionKind kind;    not_null<const Type*> type;  // 表达式的类型，永不为空    const ExpressionSyntax* syntax = nullptr;    // 从语法节点绑定表达式（类型检查入口）    static const Expression& bind(        const ExpressionSyntax& syntax,        const ASTContext& context);    // 常量求值    ConstantValue eval(EvalContext& context) const;private:    mutableconst ConstantValue* constant = nullptr; // 缓存求值结果};
```

`constant`字段是一个缓存：一旦表达式被求值为常量，结果就被缓存下来。后续对同一表达式的常量求值直接返回缓存值，避免重复计算。

### 5.2 40+种表达式类型

slang 支持 SystemVerilog 的全部表达式种类——从简单的整数字面量到流式拼接操作：

```
#define EXPRESSION(x) \    x(IntegerLiteral) x(RealLiteral) x(StringLiteral) \    x(UnaryOp) x(BinaryOp) x(ConditionalOp) \    x(Assignment) x(Concatenation) x(Streaming) \    x(Call) x(MemberAccess) x(ElementSelect) \    /* ...总计40+种... */
```

## 六、AST 遍历：编译时多态的 Visitor 模式

### 6.1 ASTVisitor——CRTP 驱动的零开销遍历

slang 的 Visitor 实现堪称 C++模板编程的教科书案例。它基于 CRTP（Curiously Recurring Template Pattern），在编译时就确定了所有分派逻辑，运行时没有虚函数开销：

```
// 来源：include/slang/ast/ASTVisitor.htemplate<typename TDerived, VisitFlags Flags = VisitFlags::Symbols>class ASTVisitor {public:    template<typename T>    void visit(const T& t) {        // 编译期检查：如果派生类有handle(T)方法就调用它        ifconstexpr (requires { (DERIVED).handle(t); })            (DERIVED).handle(t);        else            visitDefault(t);  // 否则递归访问子节点    }    template<typename T>    void visitDefault(const T& t) {        // 如果T是Scope，递归访问所有成员        if constexpr (std::is_base_of_v<Scope, T>) {            for (auto& member : t.members())                member.visit(DERIVED);        }    }};
```

### 6.2 makeVisitor——Lambda 组合的便捷 API

对于简单场景，slang 提供了一个优雅的`makeVisitor`工具，允许你用 lambda 组合出一个完整的 Visitor：

```
// 来源：include/slang/ast/ASTVisitor.htemplate<typename... Functions>auto makeVisitor(Functions... funcs) {    struct Result :public Functions...,                   public ASTVisitor<Result, VisitFlags::AllGood> {        Result(Functions... funcs) : Functions(std::move(funcs))... {}        using Functions::operator()...;    };    return Result(std::move(funcs)...);}// 使用示例：统计所有二元表达式int count = 0;auto visitor = makeVisitor(    [&](auto& visitor, const BinaryExpression& expr "&") {        count++;        visitor.visitDefault(expr);    });compilation.getRoot().visit(visitor);
```

这种设计将 C++17 的 fold expression 与 CRTP 完美结合，既保持了类型安全，又提供了极致的易用性。

### 6.3 Symbol 的 dispatch 实现

Symbol 的`visit`方法通过一个巨大的 switch 完成从`SymbolKind`到具体类型的分派：

```
// 来源：include/slang/ast/ASTVisitor.h（简化）template<typename TVisitor, typename... Args>decltype(auto) Symbol::visit(TVisitor&& visitor, Args&&... args) const {    switch (kind) {        case SymbolKind::Root:            return visitor.visit(*static_cast<const RootSymbol*>(this), args...);        case SymbolKind::Instance:            return visitor.visit(*static_cast<const InstanceSymbol*>(this), args...);        // ...100+个case...    }}
```

虽然代码看似冗长，但编译器会将其优化为一张跳转表，性能极优。

## 七、诊断系统：像 clang 一样的错误报告

### 7.1 结构化诊断

slang 的诊断系统借鉴了 clang 的设计，每条诊断由`DiagCode`标识，携带位置信息、参数和附注：

```
// 来源：include/slang/diagnostics/Diagnostics.hclass Diagnostic {public:    DiagCode code;                    // 诊断码    SourceLocation location;          // 源码位置    std::vector<Arg> args;            // 格式化参数    std::vector<SourceRange> ranges;  // 高亮范围    std::vector<Diagnostic> notes;    // 附加说明};
```

诊断子系统覆盖了编译器的每一个阶段（Lexer、Parser、Types、Lookup 等），并支持按 severity 过滤和 severity 重映射——这意味着用户可以将某些 warning 提升为 error，或忽略不关心的检查。

## 八、工具链生态

> slang 不仅是一个库，还附带了多个实用工具：

- **slang（driver）**：主编译器工具，支持完整的编译+静态检查
- **slang-hier**：层次结构可视化，显示模块实例树和参数值
- **slang-tidy**：类似 clang-tidy 的 lint 工具
- **slang-rewriter**：基于 CST 的代码重写工具
- **pyslang**：Python 绑定，支持脚本化分析

## 总结

> slang 用约 95%的 C++代码实现了一个完整的 SystemVerilog 编译器前端，其架构有几个值得学习的核心决策：

1. **BumpAllocator 统治一切**：编译器天然适合 arena 分配——创建大量短生命周期对象，最后统一释放。
2. **延迟展开避免无效功**：Scope 的 lazy elaboration 确保只有真正被使用的代码才会被完整分析。
3. **CRTP Visitor 替代虚函数**：编译期多态让 AST 遍历达到手写 switch 的性能。
4. **CST+AST 双层结构**：保留完整的具体语法树使得代码重构、格式化等工具成为可能。
5. **容错解析**：即使面对不完整代码也能产出有用的语法树，这是 IDE 场景的硬需求。

对于从事 EDA 工具开发、编译器工程或需要处理复杂 DSL 的工程师来说，slang 是一个绝佳的学习样本——它证明了一个人用现代 C++也能构建出超越商业工具合规性的编译器前端。

参考资料

[1] Building: *https://sv-lang.com/building.html*

[2] User Manual: *https://sv-lang.com/user-manual.html*

[3] Developer Guide: *https://sv-lang.com/developer-guide.html*

[4] 在线 Playground: *https://sv-lang.com/explore/*
