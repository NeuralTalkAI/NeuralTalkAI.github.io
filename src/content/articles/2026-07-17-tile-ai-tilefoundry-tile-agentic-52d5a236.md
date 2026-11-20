**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**Python DSL**、:mdnice-bold-italic[张量编译器]、**HIR/TIR**、CUDA 代码生成、运行时链接

> 高性能 GPU 程序开发长期有一个尖锐矛盾：算法研究者希望用接近数学表达的方式描述张量计算，系统工程师却必须在寄存器、共享内存、线程块、launch ABI 和模板代码之间反复打磨。

![文章插图](/articles/2026-07-17-tile-ai-tilefoundry-tile-agentic-52d5a236/images/01.jpg)

- **TileFoundry is a tile-based, agentic platform for automatic high-performance program generation across hardware.**
- https://github.com/tile-ai/TileFoundry

::: related-title
**相关推荐**
:::

::: related-links
- **[速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)**
- **[TileRT v0.1.3 重磅发布：超低延迟 LLM 推理再提速，GLM-5 支持上线，在线 Demo 即刻体验！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897357&idx=1&sn=5ca8dfaeb17f34d4ac452018efd9fffb&scene=21#wechat_redirect)**
- **[性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)**

:::

> 越接近硬件，性能越可控；越接近业务，表达越高效。

**TileFoundry 试图在二者之间架起一座“可编译的桥”**：用户用 Python DSL 描述张量意图，系统把它解析为高层 HIR，再经 Pass 降到 TIR，最终拆分生成 CUDA device module 与 CPU host wrapper，并通过运行时加载为可直接调用的 Python 对象。

::: block-1
它当前还是 Alpha 阶段，但代码已经展示出:mdnice-highlight[一个严肃编译器项目的骨架：明确的 IR 分层、注册式 Op Schema、PassManager、代码生成模板、运行时 ABI，以及围绕 spec 的工程纪律。]
:::

## 本文目录

- 一、快速上手：先把编译器跑起来
- 二、架构总览：一条从 Python 函数到 CUDA `.so` 的流水线
- 三、DSL 前端：用 Python 语法承载张量意图
- 四、IR 设计：用 HIR/TIR 分层隔离“意图”和“实现”
- 五、Pass 与 lowering：从高层张量图走向目标程序
- 六、代码生成与链接：split pipeline 的关键价值
- 七、运行时：让生成出来的 `.so` 像 Python 函数一样调用
- 八、工程纪律：spec、dump、测试与可演进性
- 结语：TileFoundry 的本质是重塑内核生产方式

![文章插图](/articles/2026-07-17-tile-ai-tilefoundry-tile-agentic-52d5a236/images/02.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：先把编译器跑起来

> TileFoundry 的 README 很克制，只给出一句定位：**它是一个 tile-based、agentic 的跨硬件高性能程序自动生成平台**。真正可操作的开发说明主要在 `docs/develop.md`[1]。结合项目依赖和开发文档，最短路径如下：

```
# 1. 准备一个带 PyTorch + CUDA 的 conda 环境conda create --clone <torch_env> -n tilefoundry-devconda activate tilefoundry-dev# 2. 初始化 CUTLASS 子模块git submodule update --init --recursive# 3. 以 editable 方式安装，并带上测试依赖python -m pip install -e '.[test]'# 4. 运行测试pytest tests/
```

如果只是修改 IR、Pass 或类型系统，可以先跑更窄的测试集合：

```
pytest tests/passes tests/ir tests/ir_types -q
```

如果关注代码生成与运行时，可以参考：

```
pytest tests/codegen tests/runtime -qpytest tests/e2e -q
```

需要特别注意：项目文档明确说明，完整 `pytest tests/` 包含 unit、codegen、nvcc 编译与 GPU end-to-end 测试；没有单独的 `--gpu` opt-in。也就是说，**没有 CUDA 设备或没有 `nvcc` 的机器会失败，而不是悄悄跳过**。这种设计体现了项目对“端到端正确性”的态度：内核生成系统不能只验证文本，最终必须验证可编译、可运行。

项目依赖在 `pyproject.toml` 中也很直观：`isl-python` 用于整数集合/多面体相关表达能力，`jinja2` 用于代码模板生成，`apache-tvm-ffi` 用于运行时调用桥接，`torch>=2.7` 用于张量运行与 dtype 映射，`graphviz` 用于可视化**，`transformers` 则暗示未来 agentic 自动生成能力的方向。当前代码主体更像一个编译器与运行时底座，agentic 更像平台愿景而非本文重点。**

## 二、架构总览：一条从 Python 函数到 CUDA `.so` 的流水线

> TileFoundry 的主流程可以概括为：

```
Python DSL   ↓ @func / @prim_func / @moduleParser   ↓HIR：高层张量表达、Mesh、GridRegion、函数组合   ↓ HirToTirPassTIR：更接近目标后端的程序表示   ↓ BufferizePassTIR with buffer plan   ↓ Codegen registryCUDA device module + CPU host module   ↓ linkershared library   ↓ runtime loaderRuntimeModule，可被 Python 直接调用
```

仓库顶层结构也印证了这种分层：

```
src/tilefoundry/  script.py        @func / @prim_func 装饰器入口  module.py        @module 收集多个 DSL 函数  compile.py       lower / build / compile / jit 总入口  parser/          Python AST → HIR/TIR  ir/              core / hir / tir / types / target  passes/          PassManager 与 lowering / bufferize  codegen/         cuda / cpu emitter 与 linker  runtime/         RuntimeModule、loader、ABI 封装  inspection/      IR 打印、DOT、viewer  dump/            Pass dump 机制include/tilefoundry/  runtime.h        C++/CUDA runtime umbrella headerdocs/  spec/            公共契约  SPEC-RULES.md    spec 书写规则  develop.md       开发流程tests/  parser / ir / passes / codegen / runtime / e2e 等
```

最核心的用户入口在 `compile.py`。它把编译行为拆成三个公共动词：`lower`、`build`、`compile`，另外提供 `jit` 作为带缓存的快捷方式。

```
# 来源：src/tilefoundry/compile.pydef _build_default_pipeline() -> PassManager:    pm = PassManager()    pm.add(HirToTirPass())    pm.add(BufferizePass())    return pm
```

这个默认 pipeline 非常短：先 HIR 到 TIR，再 bufferize。短并不意味着简单，而是说明项目当前阶段把“主干路径”压得很清楚：先把 DSL 表达稳定映射到中间表示，再把内存与后端拆分逐步增强。

`lower` 要求输入必须是 `Module`，并且当前只支持 `target="cuda"`；`build` 则要求 lowered module 已经带有 target metadata。`compile(mod)` 实际就是 `build(lower(mod))`。这种分层很像传统编译器中的 frontend、middle-end、backend，只是它把 Python DSL、CUDA JIT 和 Python RuntimeModule 打通了。

```
# 来源：src/tilefoundry/compile.pydef compile(    mod: Module,    /,    *,    target: str = "cuda",) -> "RuntimeModule":    """``build(lower(mod, target=target))`` — full compile shortcut."""    lowered = lower(mod, target=target)    return build(lowered)
```

从工程设计上看，这个入口有两个重要信号：

- 第一，TileFoundry 不是简单的代码模板拼接器。它坚持 `Module → Pass → LinkableModule → RuntimeModule` 的编译器结构。
- 第二，它也不是“纯 IR 研究玩具”。`build` 真正会进入 CUDA/CPU split pipeline，生成、编译、链接并加载共享库。

## 三、DSL 前端：用 Python 语法承载张量意图

> TileFoundry 最吸引人的地方，是用 Python 函数作为 DSL 外壳。用户通过 `@func`、`@prim_func` 把 Python 函数源代码解析为 IR 节点。注意，这里的装饰器不是运行原 Python 函数，而是读取源代码、解析 AST、构造 IR。

```
# 来源：src/tilefoundry/script.pydef func(fn=None, *, topologies=(), target=None):    """Decorator: parse an ``@func``-decorated function into a ``hir.Function``."""    resolved_target = resolve_target(target) if target isnotNoneelseNone    def _wrap(fn_inner):        extra_closure = _definition_namespace()        ir = parse_func(            fn_inner, topologies=topologies, target=resolved_target,            extra_closure=extra_closure,        )        verify_function(ir)        return ir    if fn isnotNone:        return _wrap(fn)    return _wrap
```

这段代码揭示了一个关键事实：`@func` 装饰后的名字绑定到 `hir.Function`，而不是原 Python callable。**换句话说，Python 在这里扮演的是“宿主语言”，真正的程序语义在 IR 中。**

`_definition_namespace()` 还做了一件很实用的事：它会从定义帧中收集已经解析好的 sibling `@func` / `@prim_func`，使得一个 DSL 函数可以调用上方定义的另一个 DSL 函数。这种设计让 DSL 代码仍然保留 Python 的组织习惯，但避免了任意 Python 对象混入 IR。

当多个函数需要作为一个编译单元时，`@module` 会收集 class body 中的 DSL 函数，并指定 entry。

```
# 来源：src/tilefoundry/module.pydef _wrap(cls_inner):    functions = []    for name, value in vars(cls_inner).items():        if name.startswith("__") and name.endswith("__"):            continue        ifnot isinstance(value, (HirFunction, PrimFunction)):            raise TypeError(                f"@module {cls_inner.__name__!r}: member {name!r} is a "                f"{type(value).__name__}, not an @func / @prim_func result"            )        if getattr(value, "specializations", ()):            continue        functions.append(value)    ...    return Module(name=cls_inner.__name__, functions=tuple(functions), entry=entry)
```

这个约束非常严：`@module` 的 class body 只能包含 DSL 函数，不能夹杂普通 Python 成员。这样做牺牲了一点灵活性，但换来的是编译单元的纯净性和可验证性。

Parser 层真正承担 AST 到 IR 的翻译。基础表达式 visitor 支持常量、变量、二元运算、张量切片、Op 调用、函数调用等。最典型的是 `call_to_op_call`：它把 `tf.add(...)`、`T.copy(...)` 或 bare name 调用解析为注册表里的 OpSchema，再构造 IR `Call`。

```
# 来源：src/tilefoundry/parser/base.pydef call_to_op_call(self, node: ast.Call) -> Expr:    """Resolve ``foo(...)`` to a ``Call`` on an hir Op."""    if isinstance(node.func, ast.Name):        name = node.func.id    elif isinstance(node.func, ast.Attribute):        name = ast.unparse(node.func)    else:        raise VerifyError("only Name / Attribute callees supported in V1")    callee_func = self._resolve_function_target(node.func)    if callee_func isnotNone:        return self._build_function_call(callee_func, node, name)    schema = self._resolve_call_target(node.func)    if schema isNone:        raise VerifyError(f"unknown Op name {name!r}")
```

可以把 Parser 想象成“海关”：它允许 Python 语法进入，但只放行编译器能理解的那部分。普通 Python 控制流、任意对象、不可静态求值的表达式，都不能随意穿过边界。这正是 DSL 与一般 Python 库的区别。

## 四、IR 设计：用 HIR/TIR 分层隔离“意图”和“实现”

TileFoundry 的 IR 分为 core、HIR、TIR、types、target 等层次。`Module` 是顶层编译单元，它包含函数列表、entry 名称、topologies 和 metadata。

```
# 来源：src/tilefoundry/ir/core/module.py@dataclass(frozen=True)class Module:    """Frozen container of functions + the name of the public entry function."""    name: str    functions: tuple[ModuleFunction, ...]    entry: str    topologies: tuple[Topology, ...] = field(default_factory=tuple)    metadata: dict[str, object] = field(default_factory=dict)    def __post_init__(self) -> None:        for fn in self.functions:            seal = getattr(fn, "seal", None)            if callable(seal):                seal()
```

`Module` 构造时会 seal 函数，禁止之后继续添加 specialization variant。这是一个非常编译器化的设计：authoring 阶段可以有可变状态，进入编译单元后必须冻结，否则缓存、验证、lowering 都会变得不可靠。

HIR 的 `Function` 则代表高层张量函数。它是一个 `Expr`，包含参数、body、return type、topologies、specializations、variants 和 target。

```
# 来源：src/tilefoundry/ir/hir/function.py@dataclass(frozen=True)class Function(Expr):    """HIR function container: a pure-SSA ``Expr`` whose value type is its callable signature."""    name: str    params: tuple[Var, ...]    body: Expr | None    return_type: Type    topologies: tuple[Topology, ...] = field(default_factory=tuple)    specializations: tuple[Pattern, ...] = field(default_factory=tuple)    variants: tuple["Function", ...] = field(default_factory=tuple)    target: Target = field(default_factory=CudaTarget)
```

这里有一个有趣细节：`body=None` 不是错误，而是 dispatch prototype。**也就是说，一个 `@func` 可以只声明签名和 envelope，再通过 `@base.specialize(pattern)` 添加 shape specialization**。`script.py` 中的 `_specialize` 会把 variant 解析为同名函数，并挂到 base 的 `variants` 上。这为按形状分派不同实现预留了机制。

HIR parser 对函数体的限制很明确：函数体必须以 `return` 结束；assignment 只是把名字绑定到 Expr；HIR 是 “SSA-as-DAG”，不会生成传统 Python 局部变量语义。循环则被解析为 `GridRegionExpr`，用于表达 tile/range 形式的区域计算，而不是直接执行 Python 循环。

```
# 来源：src/tilefoundry/parser/hir_parser.py# Function body: assignment statements only update the symtab; hir is# SSA-as-DAG (§8.6), so variable sharing is expressed by the tail Expr# referencing the same Expr object via env lookup.def visit_body(self, stmts: list[ast.stmt]) -> Expr:    for stmt in stmts:        for sub in ast.walk(stmt):            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):                raise VerifyError(                    "hir: nested function definition not allowed in an "                    "@tilefoundry.func body"                )    return self._visit_chain(stmts, 0)
```

从语言设计角度看，这是一种典型的“受限 Python”：**语法像 Python，语义不是 Python**。这样做的好处是用户学习成本低，编译器仍能掌控语义边界。

Op 系统则通过 `OpSchema` 注册表提供统一入口。`tf` dialect 对应 HIR 张量操作，`T` dialect 对应 TIR 操作。注册表允许同名多 schema，也允许 alias schema 插到 real Op 前面，实现 surface API 与 IR class 的解耦。

```
# 来源：src/tilefoundry/ir/core/op_registry.pydef _register_schema(schema: "OpSchema") -> None:    """Append an OpSchema to the list-per-name overload registry."""    key = (schema.dialect, schema.name)    bucket = _schemas_by_dialect_name.setdefault(key, [])    if schema in bucket:        return    bucket.append(schema)
```

这套机制让 DSL 层可以写 `tf.add`、`tf.matmul` 这样的表面 API，而内部可以逐步迁移到更统一的 kinded Op，例如把多个二元算子归并为 `Binary(kind=...)`。对一个处于快速演进期的编译器来说，这种“外部 API 稳定、内部 IR 可重构”的能力非常重要。

## 五、Pass 与 lowering：从高层张量图走向目标程序

> TileFoundry 的 **PassManager 目前是线性 pipeline**：按顺序运行 pass，并在每个 pass 后对发生身份变化的函数重新 typecheck 或 verify。同时它会在 DumpScope 中记录 pass 前后的 IR。

```
# 来源：src/tilefoundry/passes/pass_manager.pydef run(self, module: Module) -> Module:    self._check_requires()    for seq, p in enumerate(self.passes):        prev = module        with DumpScope(f"{seq:02d}_{p.name}"):            dump("before.txt", repr(prev), DumpFlags.PASS_IR)            module = p.run(module)            self._post_pass_recheck(prev, module)            dump("after.txt", repr(module), DumpFlags.PASS_IR)    return module
```

这段代码虽短，却体现出成熟编译器工程中的几个习惯：

- Pass 有依赖检查：后一个 pass 可以声明必须先运行哪个 pass。
- Pass 后自动重检：避免某个 transform 破坏 IR 不变量。
- Pass dump 是内建能力：调试 lowering 时可以看到每一步前后差异。
- 没有 change-set tracking 时，采用 whole-function fallback，先保证正确性，再优化成本。

> 默认 pipeline 中的第一个核心 pass 是 `HirToTirPass`。

这个文件很大，说明真正的语义 lowering 主要发生在那里：HIR 中的张量、mesh、grid region、dispatch 等抽象，需要转为更接近目标代码的 TIR 结构。

> 第二个 pass 是 `BufferizePass`。

当前实现是 MVP 策略：收集 logical buffer lifetime，但物理 placement 仍采用“一逻辑 buffer 一物理 allocation”的保守方案。

```
# 来源：src/tilefoundry/passes/transforms/bufferize.pyclass BufferScheduler:    """Decide a ``Placement`` for each ``BufferEntry``.    MVP policy: every logical buffer gets its own physical allocation.    """    def schedule(        self, entries: tuple[BufferEntry, ...]    ) -> tuple[Placement, ...]:        return tuple(            Placement(var=e.var, pool_id=e.var, offset=0) for e in entries        )
```

这很值得注意。很多编译器项目一开始就试图把 buffer reuse、shared memory packing、liveness analysis 做到复杂而精巧，结果主线难以稳定。TileFoundry 这里先把接口边界切出来：`LifetimeCollector` 负责收集生命周期，`BufferScheduler` 负责 placement。当前策略保守，但未来可以替换为真正的复用调度器，而不必改变 pass 外部契约。

::: block-1
换句话说，它先造“插槽”，再逐步插入更聪明的调度策略。这是编译器架构中非常重要的可演进性。
:::

## 六、代码生成与链接：split pipeline 的关键价值

> TileFoundry 的 backend 并不是把所有东西塞进一个 CUDA 源文件，而**采用 split pipeline**：CUDA device functions 生成 `.cu` translation unit，CPU host entry 生成 host module，最后链接为一个 host-callable `.so`。

这个设计的关键，是把 CUDA 类型和 launch 细节封装在 device translation unit 内，通过 C ABI shim 暴露给 host。

`compile.py` 中 `_build_split_runtime_module` 的流程很清楚：

```
# 来源：src/tilefoundry/compile.pylinked = insert_default_host_entry(mod)groups = group_functions_by_target(linked)cuda_group = groups.get("cuda", ())if not cuda_group:    raise ValueError(        f"tilefoundry.build: module {linked.name!r} has no CUDA device functions"    )cpu_entry = linked.entry_function()device_module = get_emitter("cuda")(cuda_group)host_module = get_emitter("cpu")(cpu_entry, linked)
```

这里先插入默认 host entry，再按 target 分组，然后分别调用 CUDA emitter 与 CPU emitter。这个分离带来三个好处：

- 第一，device 端可以专注生成 kernel 与 launch shim，不必处理 Python/FFI 可见 ABI。
- 第二，host 端可以负责 shape scalar 推导、输出参数管理、dispatch entry 等逻辑。
- 第三，最终运行时只面对一个共享库入口，用户体验仍然是 `rm(a)` 这样的 Python 调用。

CUDA module emitter 的代码更直观。它会为每个 kernel 生成 `__global__` kernel 以及一个 `extern "C"` launch shim。shim 接收普通 C ABI 参数，包括 grid、block、dynamic shared memory 和 stream。

```
# 来源：src/tilefoundry/codegen/cuda/module.pydef shim_symbol(kernel_name: str) -> str:    """The ``extern "C"`` launch-shim symbol for *kernel_name*."""    return "tilefoundry_" + kernel_name.replace("$", "__") + "_launch"_LAUNCH_ABI_PARAMS = (    "int grid_x, int grid_y, int grid_z, "    "int block_x, int block_y, int block_z, "    "int dynamic_smem, void* stream")
```

这就是底层工程里常见但极其关键的“边界清洁”：host 不直接拼 CUDA launch 语法，device module 自己知道怎么 `<<<grid, block, smem, stream>>>`，host 只调用 C ABI shim。

CUDA emitter 还检查同一 module 中所有 kernel 的 launch topology 是否一致。如果多个 kernel 的 grid/block 不一致，它会抛错，而不是默默生成不可靠的 program shape specialization。

```
# 来源：src/tilefoundry/codegen/cuda/module.pybase = all_fields[0]for f in all_fields[1:]:    if (f.grid, f.block) != (base.grid, base.block):        raise ValueError(            "emit_cuda_module: kernels disagree on launch topology — "            f"{base.kernel_name!r} has (grid={base.grid}, block={base.block}) "            f"but {f.kernel_name!r} has (grid={f.grid}, block={f.block})"        )
```

这类检查看似“啰嗦”，实则是高性能代码生成系统的生命线。因为 GPU 程序的错误往往不是立即崩溃，而是性能下降、越界、结果偶发错误。越靠近后端，越要把不变量显式化。

## 七、运行时：让生成出来的 `.so` 像 Python 函数一样调用

> 编译器的终点不只是生成代码文本，更是生成可用的程序。

TileFoundry 的运行时入口是 `RuntimeModule` 和 `RuntimeFunction`。`RuntimeFunction.__call__` 支持两种模式：

- auto-alloc：只传输入张量，运行时自动分配输出；
- pre-alloc：传输入和输出张量，调用后返回输出。

```
# 来源：src/tilefoundry/runtime/module.py@dataclass(frozen=True)class RuntimeFunction:    """Callable wrapper for a single compiled function."""    type: CallableType    _entry: object  # tvm_ffi callable    def __call__(self, *args):        if len(args) == len(self.type.params):            return self._call_pre_alloc(*args)        elif len(args) == self.type.input_count:            return self._call_auto_alloc(*args)        else:            raise TypeError(                f"{self.type.name}: expected "                f"{self.type.input_count} inputs (auto-alloc) or "                f"{len(self.type.params)} inputs+outputs (pre-alloc), "                f"got {len(args)}"            )
```

auto-alloc 的实现会从输入 torch.Tensor 推断 device，再按 ABI 中记录的 dtype 和 shape 创建输出 tensor。

```
# 来源：src/tilefoundry/runtime/module.pydef _call_auto_alloc(self, *args):    import torch    device = None    for a in args:        if isinstance(a, torch.Tensor):            device = a.device            break    if device isNone:        raise TypeError(            f"{self.type.name}: cannot infer device for auto-alloc; "            f"no torch.Tensor in inputs"        )    outs = []    for p in self.type.output_params:        dtype = _torch_dtype(p.dtype)        outs.append(torch.empty(p.shape, dtype=dtype, device=device))    all_args = tuple(args) + tuple(outs)    self._entry(*all_args)    return outs[0] if len(outs) == 1else tuple(outs)
```

这层封装的意义很大。对用户来说，编译产物不是“某个临时目录里的 `.cu` 和 `.so`”，而是一个可调用对象。对系统来说，ABI 信息、kernel metadata、launch config 都被结构化保存下来，后续可以用于调试、profile、缓存和可视化。

`jit` 入口还提供基于 canonical text 的缓存。它不是用 Python 对象身份做 key，而是把 entry function 的脚本化文本、topology 声明、target 和 options 串起来做 sha256。这个选择很关键：对象身份会随进程和定义方式变化，canonical text 更接近“程序语义”。

```
# 来源：src/tilefoundry/compile.pycanonical_text = _canonical_module_text(mod)payload = canonical_text + "\0" + target + "\0" + options.canonical_text()key = hashlib.sha256(payload.encode("utf-8")).hexdigest()if key not in _jit_cache:    _jit_cache[key] = compile(mod, target=target)return _jit_cache[key]
```

这说明 TileFoundry 不只是“把函数编译一次”，而是在构建一个可反复使用、可缓存、可追踪的内核生成系统。

## 八、工程纪律：spec、dump、测试与可演进性

> 一个编译器项目最怕什么？不是一开始功能少，反倒是**契约散落在注释、PR、issue、聊天记录和测试里，最后没人知道哪个行为是正式设计**。:mdnice-bold-italic[TileFoundry 的 `docs/SPEC-RULES.md` 对此有明确态度：spec 是公共契约的 single source of truth。]

**文档要求：如果同一 public construct 的另一种实现需要知道某个事实，那么这个事实应该进入 `docs/spec/*.md`，** 而不是藏在代码注释或 plan 文件里。代码注释只描述局部机制，不承担长期契约。这种纪律对编译器尤其重要，因为 IR、Pass、Parser、Codegen 之间有大量隐式不变量，一旦契约不集中，系统很快会变成“只能靠原作者记忆维护”。

测试目录也覆盖了项目主链路：parser、IR、types、passes、codegen、runtime、e2e。`docs/develop.md` 明确指出，完整测试不会静默跳过 GPU 端到端覆盖。对于一个生成 CUDA 的系统来说，这是正确的强硬：只比较生成文本远远不够，真正的正确性必须跨过 nvcc、链接、加载和执行。

项目还内建了 dump 机制。PassManager 会在每个 pass 的 DumpScope 里输出 before/after IR。调试编译器时，这比普通日志更重要，因为错误往往发生在某个 transform 之后：输入 IR 合法，输出 IR 看起来相似，但某个 type、shape、mesh 或 buffer lifetime 已经被破坏。可观察性是编译器工程的基础设施。

::: block-1
**从当前代码看，TileFoundry 仍处在快速成长阶段**：README 极简，Bufferize 还是 MVP，目标后端当前主要是 CUDA，CPU codegen 更像 host wrapper 角色，**agentic 自动生成能力尚未在主链路中成为显性模块**。但这并不削弱它的价值。恰恰相反，:mdnice-highlight[它的核心骨架已经比较清晰：前端收敛 Python DSL，IR 承载语义，Pass 保持验证，Backend 拆分 host/device，Runtime 提供 Python 调用闭环。]
:::

## 结语：TileFoundry 的本质是重塑内核生产方式

> 如果只把 TileFoundry 看成“一个生成 CUDA 的 Python 项目”，就低估了它。**它真正要解决的问题，是高性能内核生产中的知识断层：算法意图、张量布局、tile 策略、线程拓扑、内存分配、kernel launch、Python 调用**，这些知识过去分散在不同工程师、不同文件和不同经验中。:mdnice-bold-italic[TileFoundry 试图把它们纳入一个可解析、可验证、可 lowering、可生成、可运行的系统。]

- 它的 DSL 前端让用户用接近 Python 的方式表达计算；
- HIR/TIR 分层让高层意图和底层实现逐步解耦；
- OpSchema 注册表让 API 演进不必绑死内部 IR；
- PassManager 和 spec 机制让变换过程有纪律；
- split codegen 把 CUDA device 与 CPU host ABI 清晰分离；
- RuntimeModule 则把最终产物重新包装成 Python 可调用对象。

::: block-1
**这条路不会简单**。真正高性能的自动程序生成，还需要更强的 schedule 搜索、更复杂的 buffer reuse、更丰富的目标后端、更成熟的 shape dispatch，以及也许来自 agent 的自动策略生成。但从现有代码看，TileFoundry 选择了一条正确的路线：不先追求“神奇自动优化”，而:mdnice-highlight[先建立一个严肃的编译器底座。只有当语义、IR、Pass、代码生成和运行时都能闭环，自动化才有可靠的落点。]
:::

因此，TileFoundry 当前最值得关注的突破，绝非能替代多少手写 CUDA，是它:mdnice-bold-italic[把“手写内核的隐性经验”转化为“编译系统的显性结构”。这正是高性能程序生成走向规模化、可验证和可演进的前提。]

参考资料

[1]

`docs/develop.md`: *https://github.com/tile-ai/TileFoundry/blob/main/docs/develop.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)**
- **[TileRT v0.1.3 重磅发布：超低延迟 LLM 推理再提速，GLM-5 支持上线，在线 Demo 即刻体验！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897357&idx=1&sn=5ca8dfaeb17f34d4ac452018efd9fffb&scene=21#wechat_redirect)**
- **[性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)**

:::
