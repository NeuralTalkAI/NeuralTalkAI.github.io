关键词：**智能体友好 UX**、:mdnice-bold-italic[JSON 信封契约]、稳定 Row Key、**ProfileSource 抽象**、静默失败守护

> 当 Claude 或 Cursor 这样的**编码智能体被丢进一个充斥着 `.nsys-rep` 与 `.ncu-rep` 的 GPU 性能调优工程时**，它会遇到一堵令人难堪的墙：:mdnice-bold-italic[Nsight GUI 不可脚本化、`nsys stats` 输出的是给人眼读的纯文本表格、`jq` 与 SQLite 之间的胶水又得每个团队自己手搓。]

智能体擅长生成代码，却拙于点鼠标；它擅长解析 JSON，却容易在自由文本里幻觉数字。

- **Agent-friendly GPU profile-query CLI。** :mdnice-bold-italic[设计目标为：代码智能体（或 Shell 脚本）不用打开任何 GUI，就能完成 GPU 性能剖析数据的解析与逻辑处理。]
- 本工具是适配智能体的性能剖析查询命令行套件：默认输出 JSON 数据，在适用场景下可按需导出 CSV 或规整表格视图；采用单次调用单次执行的运行模式。
- 项目代码：https://github.com/lucifer1004/veloq
- 项目主页：https://lucifer1004.github.io/VeloQ/

::: related-title
**相关推荐**
:::

::: related-links
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**

:::

> VeloQ（velo-query）正是为弥合这道沟而生的工具——它不替代 `nsys` 与 `ncu`，而是把它们的输出"重新打包"成一份**版本化的 JSON 契约**，让智能体可以像调用纯函数那样查询 GPU 行为。

![文章插图](/articles/2026-06-05-nsight-agent-veloq-llm-gpu-7f5bd1d2/images/01.webp)

::: image-caption
作为面向智能体GPU性能解析而设计的VeloQ工具，它不替代Nsight、ncu采集数据，重在优化结果面向程序与Agent的数据交互。产品从四大核心特性落地：采用带版本的JSON规范封装返回结构，固定数据契约，结果统一收纳于data.rows；输出规整结构化行数据而非全量原始日志，精简LLM的Token消耗；支持无状态单行调用，可管道对接jq实现脚本化处理，主键设计方便多轮采样对比；同时统一适配Nsight多类组件与PyTorch追踪数据，多数据源归一为相同输出格式，极大简化智能体自动读取、解析硬件性能指标的链路。
:::

![文章插图](/articles/2026-06-05-nsight-agent-veloq-llm-gpu-7f5bd1d2/images/02.webp)

::: image-caption
该图展示VeloQ单行查询的使用范式，整体遵循query→jq→answer链路，无需会话、GUI界面与数据爬虫操作。左侧是单条Shell查询指令，依托veloq读取nsys性能追踪文件，限定筛选内核类型、按总耗时降序并取TOP3结果，再经由jq做数据格式重塑，将纳秒耗时换算为毫秒、精简输出字段；右侧为最终结构化JSON结果，清晰罗列内核名称与对应耗时。这套方案输出规整结构化数据，大幅精简冗余文本，有效缩减LLM解析时的Token开销，适配智能体自动化批量解析GPU性能剖面数据。
:::

![文章插图](/articles/2026-06-05-nsight-agent-veloq-llm-gpu-7f5bd1d2/images/03.webp)

::: image-caption
对比Nsight GUI、原始nsys/ncu文本、SQLite+jq与VeloQ四类GPU性能剖析方案，从脚本化调用、LLM Token开销、数据契约、多捕获对比、单次查询部署五项维度评判优劣。前三类方案各有短板：Nsight无法脚本自动化，原始文本全量冗余导致Token浪费，SQLite需自定义数据表结构；VeloQ在全部五项指标均达标，依托规整行结构压缩上下文开销、带版本数据规范与稳定行主键，适配智能体程序化重复查询。文末标注适用边界：人工交互式时序排查优先Nsight GUI，自动化、Agent驱动的批量性能查询更适合VeloQ。
:::

![文章插图](/articles/2026-06-05-nsight-agent-veloq-llm-gpu-7f5bd1d2/images/04.webp)

::: image-caption
VeloQ统一命令架构，遵循一套工具、统一返回封装、多类子命令的设计思路，把Nsight Systems、Nsight Compute与元操作三类指令做分层收纳。Nsight Systems子命令包含stats、correlate、hardware、timeline等功能，用于系统级时序、硬件指标、调用关联分析；Nsight Compute面向内核，提供launches、warp-stalls、反汇编disasm等算子细粒度性能查询；meta元命令负责版本、数据源管理、脚本配方与工具自更新。命令区分顶层nsys系列与veloq ncu子空间，所有指令输出复用VeloQ标准化JSON封装，方便Agent批量调用、脚本串联与结构化数据解析。
:::

本文将拆开 VeloQ 的工程内里，看看一个真正:mdnice-bold-italic[以"智能体体验（Agent UX）"为第一性原理设计的 CLI，到底在做哪些不显眼但极其关键的事。]

## 本文目录

- 快速上手、使用
- 一、为智能体而生：从 GUI 到 JSON 信封的范式跃迁
  - 1.1 三种"智能体逃不掉的痛"
  - 1.2 一个二进制、三种轨迹源
- 二、稳定的有形契约：版本化信封与可追溯 Key
  - 2.1 信封的五元组
  - 2.2 跨轨迹 diff 的钥匙：稳定 Row Key
- 三、可插拔的 ProfileSource：让"智能体友好"从架构传染
  - 3.1 一个 trait 钉死所有契约
  - 3.2 detect 与 info：让智能体无需先验知识
- 四、缄默之防：guardrails、warnings 与 applied_scope 的反共谋
  - 4.1 让"零行结果"不再撒谎
  - 4.2 applied_scope 的"反语义漂移"哲学
- 五、诱导式发现：next_steps 与 recipes 让智能体自举工作流
  - 5.1 NextStep——"下一步给我一句完整命令"
  - 5.2 Recipes——把"老司机直觉"编译进二进制
- 六、单一可信源：JsonSchema 同时驱动 --help、wire 与 schema 端点
- 结语：Agent-friendly，不是功能标签，而是一种新的软件设计原则

![文章插图](/articles/2026-06-05-nsight-agent-veloq-llm-gpu-7f5bd1d2/images/05.webp)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 快速上手、使用

> VeloQ 默认 JSON 输出、单二进制、零状态，安装后即可对 `.nsys-rep`、`.ncu-rep`、`.pt.trace.json` 任一种轨迹一击直查：

```
# Linux/macOS：脚本同时安装 veloq 二进制与 Claude Code 技能包curl -fsSL https://raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh | bash# 或仅装二进制cargo binstall veloq# 第一条命令：探针式 info，自动检测源、报告能力位与可用 recipeveloq info path/to/trace.nsys-rep# 取 GPU 内核 Top10（默认 JSON），同时打印 trace_span 用于 per-second 归一veloq stats path/to/trace.nsys-rep --type kernel --limit 10# NCU 报告：列出 launch 后钻取 SASS / 源码行veloq ncu launches path/to/report.ncu-rep --kernel '*gemm*'veloq ncu source-metrics path/to/report.ncu-rep --row-id launch:0
```

更多 verb 与跨轨迹 diff、recipe 自举、`self-update` 等使用细节请参考 README.md[1] 与 AGENTS.md[2]；如需多 rank PyTorch trace 目录的聚合查询，参考 `veloq pytorch collectives --help`。

## 一、为智能体而生：从 GUI 到 JSON 信封的范式跃迁

### 1.1 三种"智能体逃不掉的痛"

> 智能体在面对 GPU profile 时通常只有三条路：打开 Nsight GUI（不可脚本化）、把 `nsys stats` 的整页文本灌进 context（token 浪费、且没有稳定 schema）、自己拼 SQLite 与 jq 一锅煮（每次都得重写 schema）。

VeloQ README 用一张表把这场冲突摊得很清楚：相对于"原始 nsys/ncu 文本灌 context"或"手写 SQLite + jq"，VeloQ 锁定了三个对智能体特别要紧的轴——**稳定的类型化契约**、**Token 经济性**、**可脚本化**。它的设计宣言一句话：Pure CLI in / JSON contract out; no GUI required.

### 1.2 一个二进制、三种轨迹源

> VeloQ 把 Nsight Systems（时间轴）、Nsight Compute（内核报告）和 PyTorch/Kineto（Chrome trace）三种异构 profile 通过同一个二进制、同一个信封暴露给智能体。

`crates/veloq/src/main.rs` 仅做一件事：注册 `ProfileSource` 实现并把 clap 子树嫁接到顶层。

```
// 来源：crates/veloq/src/main.rslet sources: Vec<Box<dyn ProfileSource>> = vec![    Box::new(NsysSource),    Box::new(NcuSource),    Box::new(PytorchSource),];let parser = build_parser(&sources);// ...let code = match dispatch(&sources, &matches, fmt) {    Ok(c) => c,    Err(err) => {        // 顶层 dispatch 失败：写一份不带 source/verb 的 CLI 级错误信封，        // 让智能体至少有一份 JSON 文档可解析，永远不会撞见裸 panic。        emit_cli_error(&err, fmt);        1    }};
```

注意三个细节：

- 默认源 `nsys` 的 verb 被**拔到顶层** （`veloq stats` 与 `veloq nsys stats` 等价）以减少 token；
- 任何 dispatch 失败都被翻译成结构化错误信封；
- 以及 `mimalloc` 仅在二进制层面接管全局分配器，库 crate 保持 allocator-agnostic——所有这些"非必要但贴心"的工程决策都指向同一个用户：那个看不见、不会抱怨、但只认 JSON 的 AI 智能体。

## 二、稳定的有形契约：版本化信封与可追溯 Key

### 2.1 信封的五元组

`Envelope<T>` 是 VeloQ 在 stdout 上写出的唯一成功载荷。它由 `schema`（信封自身版本）、`source`（含 `kind` 与独立的 `version`）、`command`（源 verb 用 `<source>.<verb>` 形式，meta verb 仅用 `<verb>`）、可选的 `trace` 与 `trace_span`、可选的 `meta` 元数据块、以及具体的 `data` 组成。错误亦被同一形状（`EnvelopeError`）覆盖，仅把 `data` 替换为 `error`。

```
// 来源：crates/veloq-core/src/envelope.rspubconst ENVELOPE_VERSION: &str = "v1";#[derive(Debug, Serialize)]pubstruct Envelope<T: Serialize> {    pub schema: &'staticstr,           // 信封形状版本，破坏性变更才 bump    pub source: SourceRef,              // { kind: "nsys" | "ncu" | "pytorch", version }    pub command: String,                // 例如 "nsys.stats"    #[serde(skip_serializing_if = "Option::is_none")]    pub trace: Option<EnvelopeTraceRef>,    #[serde(skip_serializing_if = "Option::is_none")]    pub trace_span: Option<TraceSpan>,  // (origin_ns, span_ns)，用于 per-second 归一    #[serde(skip_serializing_if = "meta_is_absent")]    pub meta: Option<ResponseMeta>,     // applied_scope / next_steps / warnings    pub data: T,                        // 强类型响应，由 verb 决定}
```

精妙之处在于"两套独立版本号"：信封形状（`schema`）与每个源的载荷形状（`source.version`）解耦。给 `nsys.stats` 加一个字段只需要 bump `nsys.version`，而不会牵动 `ncu.version`，更不会触动信封自身。这是给智能体写出 `if env.source.kind == "nsys" && env.source.version == "v1"` 这类版本判断的前提——契约必须能被精细地协商。

### 2.2 跨轨迹 diff 的钥匙：稳定 Row Key

> 每个 list-shaped verb 的响应都遵循 `data: { count, total_matched, rows: Vec<Row>, auxiliary? }` 形态，并且每一行强制带一个 `pub key: String`。

这个 key 不是行号，而是由该行的**身份轴** 编码而来——例如 `kernel:1234`、`bucket|0..1000000`、`launch:0|counter:sm__cycles_active.sum`。`AGENTS.md` 里给出了 30 多个 verb 的 key 格式表，并由 `wire_format_smoke::every_primary_rows_item_carries_key` 测试在结构上强制每一种 Response 都有 key 字段。

**为什么要花这么大力气？** 因为智能体跨两次 capture 做回归对比时，它需要的不是"第 7 行 vs 第 7 行"，而是"同一身份的事件 vs 同一身份的事件"。有了稳定 key，智能体可以直接 `INDEX(.data.rows; .key)` 然后在 jq 里做差分——不用反序列化复杂结构、不用拼名字字符串、不用担心顺序变化。这把"跨轨迹可 diff"从"靠经验"升级为"协议保证"。

## 三、可插拔的 ProfileSource：让"智能体友好"从架构传染

### 3.1 一个 trait 钉死所有契约

> 新增一种轨迹源（比如 ROCm rocprof 或 Habana profile）在分发层的全部代价，仅是实现 `ProfileSource` trait 加一行 `Box::new` 注册——source 本体的实现工作另算：

```
// 来源：crates/veloq-core/src/source.rspubtrait ProfileSource: Send + Sync {    fn kind(&self) -> &'staticstr;            // 进入 envelope.source.kind    fn version(&self) -> &'staticstr;         // 独立版本号    fn detect(&self, trace: &Path) -> bool;    // 不打开文件的轻量启发    fn compute_trace_span(&self, _: &Path) -> Option<TraceSpan> { None }    fn cli(&self) -> clap::Command;            // 该源贡献的 clap 子树    /// 三态返回：    ///   Ok(0) → 成功，已写成功信封    ///   Ok(1) → 失败，但已写带 verb/trace 上下文的错误信封    ///   Err  → 顶层未处理失败，由 main 写 CLI 级错误信封    fn run(&self, m: &clap::ArgMatches, fmt: OutputFormat) -> Result<i32>;}
```

注意 `run` 那个不寻常的"三态返回"。把"已自报错"和"失控异常"分开，让源 crate 在能掌握 verb 上下文时尽量自己写一份带 `source.command.trace` 的错误信封——智能体收到的报错从此不再是"哪里炸了不知道"，而是"`nsys.stats` 在 `/tmp/x.nsys-rep` 上炸了，链路如下"。这正是写给智能体读的错误。

### 3.2 detect 与 info：让智能体无需先验知识

> 智能体拿到一个文件路径时，并不知道它是哪种 profile。`veloq info <trace>` 把这一步抽掉：它依次问每个注册源 `detect()`，挑出认领者后再决定是给冷态 `.nsys-rep` 推一个 `prep` 提示，还是直接展开完整 trace map。

```
// 来源：crates/veloq/src/meta/info.rslet detected = sources.iter().find(|s| s.detect(&trace)).map(|s| s.kind());// ...} else if extension.as_deref() == Some("nsys-rep") {    // 冷 .nsys-rep：避免在 info 里付多秒级导出代价，    // 改用 next_steps 把"该跑 prep"明确告诉智能体。    meta_next_steps.push(NextStep {        hint: "Run `veloq prep` to export the parquet sidecar; \               a follow-up `info` then surfaces devices, processes, \               and NVTX top paths.".to_string(),        command: format!("veloq prep {}", trace.display()),    });}
```

智能体不再需要维护"什么后缀该走哪个工具"的查表逻辑，也不会因为冷启动突然挂上几秒——这两件事看似琐碎，却分别对应了"智能体免预设知识"和"延迟可预测"两条 UX 红线。

## 四、缄默之防：guardrails、warnings 与 applied_scope 的反共谋

### 4.1 让"零行结果"不再撒谎

> 智能体最容易踩的坑是把 `data.rows: []` 当作"没有这种事件"——而事实上，可能只是一个手滑的 `--device 7` 把所有行过滤光了。VeloQ 在 `veloq-core/src/guards.rs` 里把这种"沉默谎言"显式化：

```
// 来源：crates/veloq-core/src/guards.rspubfn check_empty_result(row_count: usize, scope: &AppliedScope) -> Option<Warning> {    if row_count > 0 { returnNone; }    letmut filters: Vec<&'staticstr> = Vec::new();    if scope.device.is_some()        { filters.push("--device"); }    if scope.stream.is_some()        { filters.push("--stream"); }    if scope.kind.is_some()          { filters.push("--type"); }    if scope.nvtx_pattern.is_some()  { filters.push("--nvtx"); }    if scope.time_window_ns.is_some(){ filters.push("--from/--to"); }    if filters.is_empty() { returnNone; } // 没显式过滤 → 零行就是真零行    Some(Warning {        severity: WarningSeverity::Warn,        code: WarningCode::EmptyWithScope,        message: format!("scope filter ({}) excluded every row. \                          Drop a filter or widen the query …", filters.join(", ")),    })}
```

同理，`check_time_window` 在 `--from/--to` 的窗口宽度小于 trace 全长 1% 时抛出 `narrow-window` 警告——直接帮智能体捕捉"忘加 `ms` 单位"这类高频幻觉。所有警告都是**和成功响应一起返回**，并不替代结果，只是在 `meta.warnings[]` 里贴一张"小心这答案可能不是你想的那个意思"的便签。

### 4.2 applied_scope 的"反语义漂移"哲学

> 每个 list verb 的成功响应都被强制要求带上 `meta.applied_scope`，把智能体刚刚施加了哪些过滤器原样回放。`AppliedScope` 不仅记录用户输入，还记录解析器**反推出来** 的隐含值，比如 `--device 0` 在 `TARGET_INFO_CUDA_CONTEXT_INFO` 表上桥接出来的 `native_pid`：

```
// 来源：crates/veloq-core/src/meta.rs#[derive(Debug, Default, Clone, Serialize, schemars::JsonSchema)]pubstruct AppliedScope {    #[serde(skip_serializing_if = "Option::is_none")]pub device: Option<i32>,    #[serde(skip_serializing_if = "Option::is_none")]pub stream: Option<i64>,    #[serde(skip_serializing_if = "Option::is_none")]pub native_pid: Option<i64>, // 解析器反推    #[serde(skip_serializing_if = "Option::is_none")]pub kind: Option<String>,    #[serde(skip_serializing_if = "Option::is_none")]pub nvtx_pattern: Option<String>,    #[serde(skip_serializing_if = "Option::is_none")]pub time_window_ns: Option<(i64, i64)>,    /// `--all-devices` 等显式聚合走这里，与 None 字段语义不同。    #[serde(skip_serializing_if = "Vec::is_empty", default)]pub aggregated_over: Vec<String>,}
```

这一字段消除了"智能体猜不到 CLI 真正做了什么"的歧义：哪怕智能体在 prompt 里漏写了一个 flag，回放也会让它当场看见。配合"多设备无 scope 时直接拒绝执行"的 ADR，VeloQ 把"歧义查询要不要冒着错答风险继续跑"的决策权完整地交还给智能体。

## 五、诱导式发现：next_steps 与 recipes 让智能体自举工作流

### 5.1 NextStep——"下一步给我一句完整命令"

`ResponseMeta.next_steps` 最多三条，每条都被强制成 `{ hint, command }` 结构，且 `command` 必须是一条 shell-safe、可直接执行的 `veloq …` 调用串。模板作者要为内嵌的 NVTX 名、文件名、`row_id` 自行做 shell 引号转义，并且**禁止注入 jq、管道或非 veloq 命令**。这不是装饰性的"提示文字"，而是智能体可以直接复制粘贴进下一次 tool call 的二级燃料。

### 5.2 Recipes——把"老司机直觉"编译进二进制

VeloQ 的另一招是把"针对常见问题的 canonical 工作流"编码成 `registry.toml`，由 `build.rs` 在编译期校验后烧进二进制：

```
# 来源：crates/veloq-core/src/recipes/registry.toml[[recipe]]id = "cold-kernel-hotspot"title = "Find the hottest kernel, then queue an NCU rerun"body = """# 按 demangled name 聚合内核，按总 wall-time 排序veloq stats <trace> --type kernel --group-by demangled --sort total:desc --limit 10# 用 search 拿到 row_id；ncu-command 直接生成 rerun 脚本veloq search <trace> --type kernel --name '<top-kernel>' --limit 1veloq nsys ncu-command <trace> kernel:<row-id> --print"""keywords = ["kernel", "hotspot", "ncu", "rerun"]related_verbs = ["stats", "search", "ncu-command"]trace_shape = ["has_kernels"]
```

更妙的是 `info` 会读取 trace 的能力位（`has_kernels`、`has_nvtx`、`multi_device` 等）反向匹配出 `applicable_recipes`，把"这条 trace 上能跑哪些 recipe"作为 JSON 字段直接喂给智能体。智能体因此不再需要"先掌握 GPU 性能调优心法"——它只需要会读 `applicable_recipes`，工作流就能自举展开。

## 六、单一可信源：JsonSchema 同时驱动 --help、wire 与 schema 端点

> VeloQ 几乎所有响应结构体都同时 `derive(Serialize, JsonSchema)`，于是产生了一个有趣的副产物：`wire_format` 模块把 schemars 生成的 JSON Schema 投影成紧凑的 TS-shorthand，再喂给 clap 的 `long_about`。

这意味着 `veloq stats --help`、`veloq schema stats` 与实际 wire 字段**永不漂移** ——它们都是同一个 Rust struct 的不同投影。

```
// 来源：crates/veloq-core/src/wire_format.rs//// 渲染示例：//   { count: int, total_matched: int, rows: StatsRow[] }////   StatsRow://     { type: kernel|memcpy|memset|sync|graph,//       name: string, short_name: string?, device_id: int?,//       count: int, total_ns: int, avg_ns: int, … }pub fn wire_format_for<T: JsonSchema>() -> WireFormat {    let schema = schema_for!(T);    project_root(&schema.into())}
```

对智能体而言，这把"读文档"与"读响应"塞回一个心智模型：在线 `--help` 即合同，离线 `schema` 即合同，stdout 上的 JSON 也是合同。三处一致，永不撒谎。

再加上 NSys 端那个 `EventRef` 标签枚举（`#[serde(tag = "type")]`，定义于 `crates/nsys/veloq-nsys-query/src/event_ref.rs`）给 `kernel`/`memcpy`/`memset`/`nvtx` 四种行直接挂上 `grid`/`block`/`bytes`/`copy_kind` 等 headline 字段，智能体在 `search` 一次后就能拿到大多数判断所需的上下文，省掉一次冗余的 `inspect` round-trip——这种**反 round-trip 设计** 是 token 经济的另一只手。

## 结语：Agent-friendly，不是功能标签，而是一种新的软件设计原则

> VeloQ 最精彩的地方，在于它没有把“agent-friendly”当成一句营销口号，而是真正把它落实成了一整套设计原则：**输出必须稳定、字段必须可协商、身份必须可对齐、错误必须可归因、下一步必须可执行。**

这背后其实指向一个更大的变化：:mdnice-bold-italic[软件工具的第一用户，正在从“直接操作界面的工程师”，逐步扩展为“代替工程师执行任务的智能体”。]

::: block-1
一旦接受这个前提，很多过去理所当然的设计都要被重新审视——自由文本不再友好，GUI 不再天然高效，依赖经验补全语义的交互也不再可靠。:mdnice-bold-italic[真正优秀的工具，必须把“给人看”升级成“既给人看，也给机器推理”。]
:::

所以，VeloQ 的意义不只是让 GPU profile 更容易查，而是把 profiling 这件事第一次明确地改写成：一套能被智能体稳定调用、组合、比较与继承的协议化能力。

> 它真正证明的是：在 AI 时代，**好的 UX 不只是 Human-friendly，更必须是 Agent-friendly**；而谁先把这件事想明白，谁就更接近下一代开发工具的形态。

参考资料

[1] README.md: *https://github.com/lucifer1004/VeloQ/blob/main/README.md*

[2] AGENTS.md: *https://github.com/lucifer1004/VeloQ/blob/main/AGENTS.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**

:::
