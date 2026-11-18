关键词：**智能路由引擎**、流式网关执行器、:mdnice-bold-italic[Token 精确计费]、**Rust 高性能服务**、分布式节点调度

> 在大模型百花齐放的今天，企业和开发者面临一个共同的技术痛点：如何在 OpenAI、Claude、Gemini、DeepSeek 等众多 Provider 之间实现统一接入、智能调度、精确计费与故障自愈？

市面上的 API 网关方案要么用 Python/Node.js 编写性能堪忧，要么缺乏流式处理的生命周期管理，**更鲜有人将个人 PC 算力纳入调度版图。**

![文章插图](/articles/2026-05-14-rust-pc-ai-keycompute-19676d8d/images/01.webp)

- **keycompute/keycompute：KeyCompute 是新一代高性能，易拓展，开箱即用的 AI Token 算力服务平台。**
- https://github.com/keycompute/keycompute
- 7000 字，阅读 20 分钟，**播客 24 分钟**

相关推荐

- [Agent Skill 框架释放小语言模型潜能，12B 模型技能选择准确率逼近 90%，算力成本降低 50%！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897865&idx=1&sn=c9271c5cbd6f373209864a1dd4af92a7&scene=21#wechat_redirect)
- [构建面向 LLM 的下一代记忆管理系统，开源项目 Hypatia：为大模型装上“结构化外脑”](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900194&idx=1&sn=e79ae72e71b12737574b362096704177&scene=21#wechat_redirect)
- [暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)

> KeyCompute 以 94.7% 的 Rust 代码比例，**从零打造了一套覆盖"路由决策 → 流式执行 →Token 计费 → 节点调度"全链路的 AI 算力服务平台**。

::: block-1
本文将深入其核心架构，剖析其如何在类型安全与零成本抽象的 Rust 哲学下，解决 AI Token 服务的全栈工程挑战。
:::

![文章插图](/articles/2026-05-14-rust-pc-ai-keycompute-19676d8d/images/02.webp)

::: image-caption
KeyCompute 多模型支持，通过标准 OpenAI API 格式访问所有大模型，开箱即用
:::

## 本文目录

- 快速上手
- 一、架构总览
  - 1.1 服务启动的七阶段生命周期
- 二、LLM Gateway：唯一执行层的流式生命周期管理
  - 2.1 执行器的 Retry-Fallback 链
  - 2.2 流式事件处理管道
  - 2.3 tiktoken 精确 Token 计数
- 三、两层路由引擎与健康状态反馈
  - 3.1 Layer1：Provider 智能排序
  - 3.2 Layer2：租户账号池与冷却机制
  - 3.3 Node 路由：个人 PC 的独立通道
  - 3.4 健康状态的反馈闭环
- 四、后置计费：不可变主账本设计
- 五、Node Gateway：让个人 PC 成为算力节点
- 六、Provider 适配器的 Trait 抽象
- 总结
  - 1. 类型安全贯穿全栈
  - 2. 零成本抽象的异步模型
  - 3. 可观测性作为一等公民
  - 4. 加密存储与安全设计

![文章插图](/articles/2026-05-14-rust-pc-ai-keycompute-19676d8d/images/03.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> 对于想要快速体验 KeyCompute 的读者，只需三步即可在本地拉起完整的 AI 算力服务——包含 PostgreSQL 数据库、Redis 缓存、后端 API 服务与前端管理面板。整个过程依赖 Docker Compose 一键编排，无需手动安装 Rust 工具链：

```
# 克隆项目git clone https://github.com/keycompute/keycompute.gitcd keycompute# 配置环境变量cp .env.example .env# 编辑 .env 填入数据库密码、JWT 密钥等（详见文件内注释）# 一键启动所有服务（PostgreSQL + Redis + Server + Web）docker compose up -d# 验证服务状态docker compose ps
```

启动后访问 `http://localhost:8080`，默认管理员账号 `admin@keycompute.local`，密码在 `.env` 中配置。更多开发环境搭建、前端独立启动等详细步骤，参考 `README.md`。

服务就绪后，调用 AI 模型只需标准 OpenAI 格式——这意味着你现有的所有 OpenAI SDK 客户端代码无需任何改动，只需将 `base_url` 指向 KeyCompute 即可：

```
curl http://localhost:3000/v1/chat/completions \  -H "Authorization: Bearer sk-your-key" \  -H "Content-Type: application/json" \  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

如果你想将请求路由到个人 PC 上的 Ollama 本地模型，只需在模型名前加 `node:` 前缀：

```
curl http://localhost:3000/v1/chat/completions \  -H "Authorization: Bearer sk-your-key" \  -H "Content-Type: application/json" \  -d '{"model":"node:deepseek-chat","messages":[{"role":"user","content":"Hello!"}]}'
```

## 一、架构总览

> KeyCompute 采用 Rust Workspace monorepo 结构，将系统拆分为 18 个 crate；核心设计原则为"唯一执行层约束"、"后置计费"和"类型驱动"；服务启动经历七个严格有序的初始化阶段。

一个优秀的系统架构，往往能从目录结构中一目了然地读出设计者的意图。**KeyCompute 将系统拆分为 18 个高内聚低耦合的 crate，每个 crate 承担单一职责，彼此通过明确定义的 trait 和类型契约交互：**

```
keycompute/├── crates/│   ├── keycompute-server/       # Axum HTTP 服务入口│   ├── llm-gateway/             # 唯一执行层：retry/fallback/streaming│   ├── keycompute-routing/      # 两层智能路由引擎│   ├── keycompute-billing/      # 后置精确计费│   ├── keycompute-pricing/      # 定价引擎│   ├── node-gateway/            # 个人 PC 节点网关│   ├── keycompute-auth/         # 认证鉴权（JWT + Argon2 密码哈希）│   ├── keycompute-ratelimit/    # 分布式限流（Redis 后端可选）│   ├── keycompute-runtime/      # 运行时状态 + API Key 加密│   ├── keycompute-distribution/ # 二级分销引擎│   ├── keycompute-observability/# 可观测性（Prometheus + tracing）│   ├── keycompute-config/       # 配置加载（TOML + 环境变量）│   ├── keycompute-emailserver/  # 邮件服务（验证码 + 密码重置）│   ├── keycompute-payment/      # 支付（支付宝 + 微信）│   ├── llm-provider/            # Provider 适配器集合│   │   ├── keycompute-openai/   # OpenAI/兼容接口│   │   ├── keycompute-claude/   # Anthropic Claude│   │   ├── keycompute-gemini/   # Google Gemini│   │   ├── keycompute-deepseek/ # DeepSeek│   │   ├── keycompute-ollama/   # Ollama 本地模型│   │   └── keycompute-vllm/     # vLLM 自托管│   └── ...└── packages/                    # Dioxus 0.7 前端（Web + Desktop + Mobile）
```

这种拆分不仅是代码组织上的整洁，更是**编译隔离** 的实践——修改计费逻辑不会触发路由引擎的重编译，修改前端不会触发后端的任何重编译。在 Rust 这种编译时间较长的语言中，这是非常重要的工程优化。

> 其核心设计原则有三：

- **架构约束（Architectural Constraint）** ——只有 `llm-gateway` 能发起上游请求，路由引擎是只读的、计费模块是后置的，这种"谁能做什么"的边界通过 crate 可见性在编译期强制执行；
- **后置计费（Post-billing）** ——计费模块不参与路由决策、不预扣余额、不阻塞请求，仅在流式响应完全结束后执行一次精确结算；
- **类型驱动（Type-driven）** ——通过 `keycompute-types` crate 统一全系统的数据流类型（如 `RequestContext`、`ExecutionPlan`、`StreamEvent`），确保模块间的契约在编译期可验证。

### 1.1 服务启动的七阶段生命周期

> 理解一个系统最好的入口是它的 `main` 函数。KeyCompute 的主入口被精心组织为七个有序阶段，每个阶段的失败都会立即终止进程并打印精确错误，避免"半死不活"的服务状态：

```
// 来源：crates/keycompute-server/src/main.rs#[tokio::main]asyncfn main() -> anyhow::Result<()> {    // 阶段 1: 加载配置（环境变量 + TOML 文件，环境变量优先级更高）    let config = AppConfig::load()?;    config.validate()?;    // 阶段 2: 初始化可观测性（开发用 pretty-print，生产用 JSON 结构化日志）    init_observability();    // 阶段 3: 全局加密初始化（用于 API Key 的 AES 加密存储）    init_global_crypto(&config)?;    // 阶段 4: 数据库连接 + 自动迁移（sqlx 管理 schema 版本）    let db = Database::new(&db_config).await?;    db.migrate().await?;    // 阶段 5: 初始化默认系统管理员 + 系统设置    initialize_default_admin(&pool).await?;    // 阶段 6: 构建应用状态（路由、网关、计费等模块注入 AppState）    let app_state = AppState::with_pool_and_config(pool, state_config);    // 阶段 7: 启动 HTTP 服务器（带 SIGINT/SIGTERM 优雅关闭支持）    tokio::select! {        result = run(server_config, app_state) => { /* ... */ }        _ = shutdown => { info!("正在优雅关闭..."); }    }}
```

- **值得注意的是阶段 3 中的全局加密初始化——KeyCompute 使用 AES 对上游 Provider 的 API Key 进行加密存储**，这意味着即使数据库被拖库，攻击者也无法直接获取明文 Key。`KC__CRYPTO__SECRET_KEY` 一旦有数据写入后便不可更换，否则历史数据将无法解密——这是一个在安全性与可维护性之间经过深思熟虑的权衡。
- **阶段 7 的优雅关闭处理同样值得关注**：系统同时监听 `SIGINT`（Ctrl+C）和 `SIGTERM`（容器编排器发送），收到信号后不会粗暴杀死进程，而是等待正在处理的流式请求自然结束后再退出。

## 二、LLM Gateway：唯一执行层的流式生命周期管理

> `GatewayExecutor` 是唯一能发起上游 HTTP 请求的组件；通过 `tokio::spawn` + 有界 `mpsc::channel` 实现流式背压控制；使用 tiktoken-rs 的 `o200k_base` 分词器实现与 OpenAI 完全一致的 Token 精确计数。

:mdnice-bold-italic[`llm-gateway` 是整个系统最核心的 crate。如果把 KeyCompute 比作一个操作系统，那么 `GatewayExecutor` 就是它的"系统调用层"——:mdnice-bold-italic[所有对外部世界（上游 LLM Provider）的访问都必须经过这里。] 这种"唯一出口"的设计看似增加了约束，实则大大简化了系统的安全审计、故障排查和性能监控。]

Gateway 的核心职责可以概括为四个词：**执行、重试、降级、计量**。

### 2.1 执行器的 Retry-Fallback 链

> 当路由引擎生成了一个 `ExecutionPlan`（包含一个 primary target 和若干 fallback targets）后，执行器负责按顺序尝试，直到成功或全部失败：

```
// 来源：crates/llm-gateway/src/executor.rspubasyncfn execute(    &self,    ctx: Arc<RequestContext>,    plan: ExecutionPlan,       // 包含 primary + fallback_chain    account_states: Arc<AccountStateStore>,    provider_health: Option<Arc<ProviderHealthStore>>,) -> Result<mpsc::Receiver<StreamEvent>> {    let (tx, rx) = mpsc::channel(100);    // 关键设计：在后台 tokio::spawn 中执行，立即返回 rx    // 避免有界 channel 背压阻塞——handler 必须先拿到 rx 才能消费    tokio::spawn(asyncmove {        runner.run_plan(ctx, plan, tx, account_states, provider_health).await;    });    Ok(rx)}
```

这里有一个精妙的工程决策值得展开讨论：为什么 `execute` 方法要**立即返回** `mpsc::Receiver`，而将实际的上游调用放在后台 task 中？

答案在于**流式背压（Backpressure）** 的处理。

::: block-1
考虑这样一个场景：channel 容量为 100，上游 Provider 高速推送 chunk 事件，如果 HTTP handler 还没有开始向客户端写入 SSE 数据（比如正在等待 `execute` 返回），那么 channel 会迅速填满，生产者 `tx.send()` 将 `.await` 挂起——形成死锁。

通过 `tokio::spawn` 将生产者放入独立 task，`execute` 可以立即返回 `rx`，handler 拿到 `rx` 后立即开始消费，生产与消费并行运转，死锁问题迎刃而解。
:::

在 `run_plan` 内部，Fallback 的逻辑简洁而有力：

```
// 来源：crates/llm-gateway/src/executor.rs（简化）asyncfn run_plan(&self, ctx: Arc<RequestContext>, plan: ExecutionPlan, ...) -> Result<()> {    // 构建 target 链：primary 在前，fallback 在后    letmut targets = vec![plan.primary];    targets.extend(plan.fallback_chain);    letmut last_error = None;    letmut is_primary = true;    for target in targets {        let target_start = Instant::now();        matchself.try_execute(&ctx, &target, tx.clone()).await {            Ok(()) => {                // 成功：记录健康状态（延迟 + 是否为 fallback）                let latency_ms = target_start.elapsed().as_millis() asu64;                health_store.record_success(provider, latency_ms);                if !is_primary { health_store.record_fallback(); }                returnOk(());            }            Err(e) => {                // 失败：记录失败事件，尝试下一个 target                health_store.record_failure(provider);                last_error = Some(e);            }        }        is_primary = false; // 之后的都是 fallback    }    Err(last_error.unwrap_or_else(|| KeyComputeError::RoutingFailed(ctx.model.clone())))}
```

这里的核心思想是：**失败不是终点，而是切换通道的信号**。只有当所有 target（包括 primary 和所有 fallback）都失败时，请求才真正失败。这使得系统具备了"N+M 冗余"的高可用特性。

### 2.2 流式事件处理管道

> 每个 target 的实际执行发生在 `try_execute` 中。这里展示了 KeyCompute 如何处理 SSE 流式事件的完整生命周期：

```
// 来源：crates/llm-gateway/src/executor.rs（简化）asyncfn try_execute(&self, ctx: &RequestContext, target: &ExecutionTarget, tx: mpsc::Sender<StreamEvent>) -> Result<()> {    // 1. 获取 Provider 适配器（通过 trait object 实现多态）    let provider_impl = self.providers.get(provider)?;    // 2. 选择 HTTP 传输层（优先使用内部 HTTP Proxy 的连接池）    let transport: Arc<dyn HttpTransport> = ifletSome(ref proxy) = self.http_proxy {        Arc::clone(proxy.default_client())    } else {        Arc::new(DefaultHttpTransport::new())    };    // 3. 发起流式请求，获得异步 Stream    letmut stream = provider_impl.stream_chat(transport.as_ref(), request).await?;    // 4. 逐事件处理——这是流式管道的核心循环    whileletSome(event) = stream.next().await {        match event? {            StreamEvent::Delta { content, finish_reason } => {                // 实时计算输出 token 数                let tokens = Self::estimate_tokens(&content);                ctx.add_output_tokens(tokens);                // 转发给客户端                tx.send(event).await?;            }            StreamEvent::Usage { input_tokens, output_tokens } => {                // Provider 报告的精确用量——覆盖本地估算值                ctx.set_input_tokens(input_tokens);            }            StreamEvent::Done => { tx.send(StreamEvent::Done).await?; break; }            StreamEvent::Error { message } => returnErr(ProviderError(message)),            _ => {}        }    }    Ok(())}
```

注意 `StreamEvent::Usage` 事件的处理策略：KeyCompute 在流式传输过程中通过 tiktoken 实时估算 token 数，但如果上游 Provider 在流结束时报告了精确用量（OpenAI 和 DeepSeek 都会这样做），则以 Provider 的数值为准。这是一种"乐观估算 + 权威修正"的双重保险机制。

### 2.3 tiktoken 精确 Token 计数

> Token 计数的准确性直接影响计费精度。KeyCompute 在这一点上不做任何妥协：

```
// 来源：crates/llm-gateway/src/executor.rsfn estimate_tokens(content: &str) -> u32 {    if content.is_empty() { return 0; }    // 使用 o200k_base tokenizer（GPT-4o/o1/o3 系列的官方分词器）    // singleton 模式：全局只加载一次词表，后续调用零开销    let bpe = tiktoken_rs::o200k_base_singleton();    bpe.encode_with_special_tokens(content).len() as u32}
```

为什么不用简单的"字符数 ÷ 4"估算？

因为对于中文、日文等非拉丁语系文本，这种粗糙估算的误差可达 50% 以上。KeyCompute 直接集成了 `tiktoken-rs` 库的 `o200k_base` 分词器（这正是 GPT-4o、o1、o3 等模型使用的官方词表），确保计费结果与 OpenAI 官方 API 返回值完全一致。`singleton` 模式保证词表只加载一次，后续每次调用仅执行 BPE 编码运算，性能开销极小。

## 三、两层路由引擎与健康状态反馈

> 路由引擎采用 Layer1（Provider 排序）+ Layer2（账号选择）的两层架构；通过成本×0.3 + 延迟×0.25 + 成功率×0.25 + 健康度×0.2 的加权评分公式进行 Provider 排序；账号选择支持优先级排序与冷却机制；Node 路由通过 `node:` 前缀触发独立路径。

如果说 Gateway 是"手"（负责执行），那么 `RoutingEngine` 就是"脑"（负责决策）。KeyCompute 的路由系统采用**模型级路由 + 账号池路由** 的两层架构，并且遵循一个核心架构约束——路由引擎是**只读无副作用** 的，它不修改任何状态，只基于当前快照做出最优决策：

```
// 来源：crates/keycompute-routing/src/lib.rs//! 路由引擎，双层路由，只读无副作用。//! 架构约束：只读 Pricing 和状态快照，不写任何状态。
```

### 3.1 Layer1：Provider 智能排序

> 第一层路由的目标是：给定一个模型名（如 `gpt-4o`），按照"综合性价比"对所有支持该模型的 Provider 进行排序。排序算法使用四维加权评分：

```
// 来源：crates/keycompute-routing/src/lib.rs/// 路由权重常量（硬编码，不可通过配置修改）const COST_WEIGHT: f64 = 0.3;      // 成本权重 30%const LATENCY_WEIGHT: f64 = 0.25;  // 延迟权重 25%const SUCCESS_WEIGHT: f64 = 0.25;  // 成功率权重 25%const HEALTH_WEIGHT: f64 = 0.2;    // 健康度权重 20%const UNHEALTHY_PENALTY: f64 = 100.0; // 不健康额外惩罚fn score_provider(&self, provider: &str, pricing: &PricingSnapshot) -> f64 {    let cost_score = self.calculate_cost_score(pricing);       // 成本越高，分数越高    let latency_score = /* ... */;  // 延迟越高，分数越高    let failure_score = /* ... */;  // 失败率越高，分数越高    let unhealthiness_score = /* ... */; // 越不健康，分数越高    // 加权平均 + 不健康惩罚（分数越低越优先）    let weighted_score = (COST_WEIGHT * cost_score        + LATENCY_WEIGHT * latency_score        + SUCCESS_WEIGHT * failure_score        + HEALTH_WEIGHT * unhealthiness_score) / 1.0;    weighted_score + unhealthy_penalty}
```

这个评分公式的设计体现了一个重要的工程哲学：**所有指标统一为"越高越差"，最终分数越低的 Provider 越优先**。这种统一方向性避免了混合正负指标时容易出现的逻辑错误。权重之和严格为 1.0（通过单元测试保证），确保评分的可解释性。

延迟评分采用分档制而非线性映射，更符合实际体感：

```
// 来源：crates/keycompute-routing/src/lib.rsfn calculate_latency_score(&self, latency_ms: u64) -> f64 {    if latency_ms == 0    { 50.0 }   // 无数据，给中间值    else if latency_ms < 100  { 10.0 }   // 优秀（<100ms）    else if latency_ms < 300  { 30.0 }   // 良好（100-300ms）    else if latency_ms < 1000 { 60.0 }   // 一般（300ms-1s）    else                      { 90.0 }   // 较差（>1s）}
```

### 3.2 Layer2：租户账号池与冷却机制

> 第二层路由在排序后的 Provider 中，为每个 Provider 选择最优的上游账号。选择逻辑考虑三个因素：**租户隔离** （不同租户使用不同的账号池）、**模型匹配** （账号必须支持请求的模型）、**冷却状态** （最近失败的账号暂时不可用）：

```
// 来源：crates/keycompute-routing/src/lib.rs（简化）asyncfn select_best_account(&self, provider: &str, accounts: Vec<Account>) -> Result<Option<ExecutionTarget>> {    // 按优先级降序排列    sorted_accounts.sort_by_key(|account| std::cmp::Reverse(account.priority));    for account in sorted_accounts {        // 跳过正在冷却中的账号（最近失败过，需要等待恢复）        ifself.account_states.is_cooling_down(&account.id) {            continue;        }        // 解密上游 API Key（AES 加密存储）        let upstream_api_key = Self::decrypt_upstream_api_key(&account.upstream_api_key_encrypted)?;        returnOk(Some(ExecutionTarget::new_provider(            provider.to_string(),            account.id,            account.endpoint,            upstream_api_key,        )));    }    Ok(None) // 所有账号都在冷却中}
```

"冷却"机制是一种**熔断（Circuit Breaker）** 的轻量级实现：当某个上游账号连续失败时（例如触发了 OpenAI 的速率限制），系统将其标记为冷却状态，一段时间内不再调度请求到该账号，给它恢复的机会。这避免了"明知会失败还反复重试"的无效开销。

### 3.3 Node 路由：个人 PC 的独立通道

> 当模型名以 `node:` 前缀开头时，路由引擎会进入一条完全独立的路径——不走 Provider 排序，而是直接查询是否有在线节点能处理该模型：

```
// 来源：crates/keycompute-routing/src/lib.rspubasyncfn route(&self, ctx: &RequestContext) -> Result<ExecutionPlan> {    // 检测 Node 路由前缀    ifletSome(actual_model) = ctx.model.strip_prefix("node:") {        let node_index = self.node_index.as_ref().ok_or_else(|| {            KeyComputeError::Internal("Node routing not configured".to_string())        })?;        // 查询数据库：是否存在 ready 状态的节点支持该模型        if node_index.has_ready_node(actual_model).await {            returnOk(ExecutionPlan {                primary: ExecutionTarget::Node { model: actual_model.to_string() },                fallback_chain: Vec::new(), // Node 路由不支持 fallback            });        } else {            returnErr(KeyComputeError::NoReadyNode(actual_model.to_string()));        }    }    // 无前缀：走标准的两层 Provider 路由    // ...}
```

Node 路由的一个重要设计决策是**不支持 fallback** ——如果没有可用节点，直接返回错误而不会回退到云端 Provider。:mdnice-bold-italic[这是有意为之的：用户显式使用 `node:` 前缀，说明他明确希望使用本地算力（可能出于隐私、成本或离线考虑），自动 fallback 到云端可能违背用户意图。]

### 3.4 健康状态的反馈闭环

> 路由引擎虽然不写状态，但它读取的健康数据来自执行层的反馈。这形成了一个优雅的闭环：

```
请求 → 路由决策（读取健康快照）→ 执行（成功/失败）→ 更新健康状态 → 影响下一次路由决策
```

```
// 来源：crates/llm-gateway/src/executor.rs（成功路径）let latency_ms = target_start.elapsed().as_millis() as u64;if let Some(ref health_store) = provider_health {    health_store.record_success(provider, latency_ms); // 记录成功 + 延迟    if !is_primary {        health_store.record_fallback(); // 统计 fallback 发生频率    }}// 失败路径health_store.record_failure(provider); // 累积失败次数
```

通过 `ProviderHealthStore` 和 `AccountStateStore` 的被动记录机制，路由引擎能实时感知各 Provider 的可用性和延迟水平，实现"越健康权重越高、越不稳定越少调度"的自适应效果。当一个 Provider 连续失败达到阈值时，其健康评分会急剧下降（叠加 100 分的 `UNHEALTHY_PENALTY`），被自然排到队列末尾，但不会被完全剔除——保留了"自我恢复"的可能性。

## 四、后置计费：不可变主账本设计

> 计费模块遵循"不参与路由、不预扣余额、不影响执行结果"的三不原则；使用 `rust_decimal::Decimal` 确保金额计算的绝对精度；流结束后一次性写入不可变的 `usage_logs` 主账本。

在许多算力平台的设计中，余额检查被放在请求入口处——"余额不足则拒绝服务"。这种看似合理的设计在高并发场景下会成为严重瓶颈：每个请求都需要一次数据库读取来查询余额，而且还需要处理并发扣款的竞态条件。

KeyCompute 选择了一条不同的路：

```
// 来源：crates/keycompute-billing/src/lib.rs//! 计费模块，仅在 stream 结束后执行。//! 架构约束：不参与路由，不预扣余额，不反向影响执行结果。
```

这三条约束的含义是：

1. **不参与路由** ——路由引擎不知道计费模块的存在，不会因为"这个 Provider 更便宜"而改变路由决策（成本已经是路由评分的一个维度，但那是在 Provider 定价层面，不是用户余额层面）；
2. **不预扣余额** ——请求发起时不检查也不冻结用户余额，避免了高并发下的锁竞争；
3. **不反向影响执行结果** ——即使计费失败（比如数据库短暂不可用），已经返回给用户的结果不会被撤回。

计费的精确度通过 `rust_decimal::Decimal` 类型保证：

```
// 来源：crates/keycompute-billing/src/lib.rs/// 计费公式：/// user_amount = (input_tokens/1000) × input_price + (output_tokens/1000) × output_pricepub fn compute_user_amount(    input_tokens: u32, output_tokens: u32,    input_price_per_1k: Decimal, output_price_per_1k: Decimal,) -> Decimal {    let input_cost = Decimal::from(input_tokens) / Decimal::from(1000) * input_price_per_1k;    let output_cost = Decimal::from(output_tokens) / Decimal::from(1000) * output_price_per_1k;    input_cost + output_cost}
```

为什么不用 `f64`？因为浮点数的精度问题在金融场景中是不可接受的。举一个典型的例子：`0.1 + 0.2` 在 `f64` 中并不等于 `0.3`，而是 `0.30000000000000004`。当这种微小误差在百万级请求中累积，最终账单可能与实际消耗产生显著偏差。`rust_decimal::Decimal` 使用 128 位定点表示，确保每一笔计费精确到小数点后 28 位。

`BillingTrigger` 是计费的触发点，它的调用时机非常明确——仅在流式响应完全结束后：

```
// 来源：crates/keycompute-billing/src/lib.rspubstruct BillingTrigger;impl BillingTrigger {    /// 触发计费结算    /// 输入: usage + pricing_snapshot + request metadata    /// 输出: 同步写入不可变 usage_logs 主账本    pubasyncfn trigger(        &self,        ctx: &RequestContext,        provider_name: &str,        account_id: uuid::Uuid,        status: &str, // "success" | "partial" | "upstream_error"        billing: &BillingService,    ) -> Result<NewUsageLog> {        billing.finalize(ctx, provider_name, account_id, status).await    }}
```

注意 `status` 参数支持三种状态：完全成功、部分完成（流中断但已产生内容）和上游错误。即使请求只完成了一半，已产生的 token 仍会被精确计费——这对于长对话场景尤其重要。

## 五、Node Gateway：让个人 PC 成为算力节点

> `node-gateway` 通过 pull-based 长轮询实现 NAT 穿透；节点注册需要共享 token 验证；后台 Sweeper 负责清理过期节点和补推滞留任务；`PostgresNodeIndex` 实现了路由引擎所需的 `NodeCapabilityIndex` trait。

`node-gateway` 是 KeyCompute 最具创新性的模块之一。在传统的 AI 算力平台中，计算资源必须部署在有公网 IP 的服务器上。但 KeyCompute 另辟蹊径——它允许任何一台安装了 Ollama 的个人 PC 通过 **pull-based 长轮询** 接入系统，无需公网 IP、无需端口映射、无需 VPN，只要能访问 KeyCompute 服务端即可：

```
// 来源：crates/node-gateway/src/lib.rspub mod config;       // 节点配置（心跳间隔、任务超时等）pub mod node_index;   // 节点能力索引（实现 NodeCapabilityIndex trait）pub mod redis;        // Redis 任务队列（BRPOP 实现长轮询）pub mod service;      // 业务逻辑（注册、心跳、任务分发、结果提交）pub mod store;        // 数据库持久化（节点状态、任务记录）pub mod sweeper;      // 后台维护（清理过期会话、补推滞留任务）
```

其工作流程可以用一个生活中的比喻来理解：想象一个快递代收点（KeyCompute 服务端），快递员（用户请求）把包裹（任务）放在柜子里，然后通知取件人（PC 节点）。取件人定期来柜子前查看（长轮询），发现有自己的包裹就取走处理，处理完后把结果送回柜子。如果取件人长时间不来（心跳超时），系统就把他的包裹重新分配给其他人。

具体流程为：

1. **节点注册**：PC 节点使用共享的 `registration_token` 向服务端注册，声明自己支持哪些模型
2. **心跳保活**：每 30 秒发送心跳，更新 `accepted_models`（可能因为用户在 Ollama 中新拉取了模型）
3. **长轮询领取任务**：节点向 Redis 队列发起 BRPOP，最多等待 30 秒
4. **本地执行**：节点调用本地 Ollama API 执行推理
5. **提交结果**：执行完成后将结果 POST 回服务端

从配置文件可以看出其完善的容错设计——每个参数都经过生产环境的打磨：

```
# 来源：config.example.toml[node_gateway]registration_token = "change-me-in-production"  # 节点注册凭证session_ttl_secs = 300          # 会话 5 分钟有效期heartbeat_interval_secs = 30    # 30 秒心跳间隔poll_timeout_secs = 30          # 长轮询最大等待时间task_deadline_secs = 120        # 任务必须在 2 分钟内完成complete_grace_secs = 60        # 超时后仍允许 1 分钟内提交结果node_failure_threshold = 3      # 连续 3 次失败则排除节点task_failure_threshold = 3      # 单个任务最多重试 3 次sweeper_heartbeat_ttl_secs = 600  # 10 分钟无心跳则标记离线sweeper_repush_interval_secs = 10  # 每 10 秒补推滞留在队列中的任务
```

`complete_grace_secs` 是一个特别贴心的设计：假设一个任务的 deadline 是 2 分钟，但节点在 1 分 59 秒时刚好完成计算，还没来得及提交结果——如果此时就判定超时并重新分发，会造成重复计算和结果冲突。60 秒的宽限期给了"慢半拍"的节点一个补交作业的机会。

`NodeCapabilityIndex` trait 是连接路由引擎和节点网关的桥梁：

```
// 来源：crates/keycompute-routing/src/lib.rs#[async_trait::async_trait]pub trait NodeCapabilityIndex: Send + Sync {    /// 检查是否存在 ready 节点可以处理指定模型    /// ready predicate:    /// - nodes.status = 'online'    /// - node_sessions.expires_at > NOW()    /// - node_sessions.revoked_at IS NULL    /// - nodes.capabilities_json->>'runtime' = 'ollama'    /// - node_sessions.accepted_models_json 包含目标模型    async fn has_ready_node(&self, model: &str) -> bool;}
```

`PostgresNodeIndex`（在 `node-gateway/src/node_index.rs` 中实现）通过一条 SQL 查询同时验证上述所有条件，确保只有真正健康且有能力的节点才会被路由到。注意它**不读取 Redis、不信任客户端自报的负载数据** ——所有判断依据都是服务端可验证的数据库记录，这是一个安全性优先的设计。

## 六、Provider 适配器的 Trait 抽象

> 通过 `ProviderAdapter` trait 统一所有 LLM Provider 的接口；`HttpTransport` trait 的依赖注入使测试可以完全 mock 网络层；新增 Provider 只需实现一个 trait + 创建一个独立 crate。

在面向对象的语言中，我们会用"接口"或"抽象类"来定义 Provider 的统一契约。在 Rust 中，这个角色由 trait 承担，并且做得更好——因为 Rust 的 trait 不仅定义了方法签名，还通过 `Send + Sync` 约束保证了并发安全性：

```
// 来源：crates/llm-gateway/src/executor.rs（从测试代码中可见 trait 完整用法）#[async_trait]impl ProviderAdapter for ManyChunksProvider {    fn name(&self) -> &'staticstr { "many-chunks" }    fn supported_models(&self) -> Vec<&'staticstr> { vec!["gpt-4o"] }    asyncfn stream_chat(        &self,        _transport: &dyn HttpTransport,  // 注入的传输层——可 mock        _request: UpstreamRequest,    ) -> Result<StreamBox> {        // 返回 Pin<Box<dyn Stream<Item = Result<StreamEvent>>>>        Ok(Box::pin(futures::stream::iter(events)))    }}
```

这种设计带来了三个重要好处：

**1. 零修改扩展**：新增一个 Provider（比如未来的 Meta Llama API）只需创建一个新 crate（如 `keycompute-llama`），实现 `ProviderAdapter` trait，然后在 Gateway 构建时注册即可。核心代码无需任何改动。

**2. 完全可测试**：注意 `stream_chat` 的第一个参数是 `&dyn HttpTransport` 而非具体的 HTTP 客户端。这意味着在单元测试中，你可以注入一个返回预设响应的 mock transport，完全不依赖网络：

```
// 测试中的 Provider：生成 150 个 chunk + Usage + Donestruct ManyChunksProvider { chunks: usize }// 测试验证：execute 必须在 50ms 内返回 receiver（不被背压阻塞）let mut rx = tokio::time::timeout(    Duration::from_millis(50),    executor.execute(ctx, plan, account_states, Some(provider_health)),).await.expect("execute should return receiver immediately");
```

**3. 内部 HTTP Proxy 支持**：通过 `HttpProxy` 层统一管理上游连接，可以实现多代理出口（按 Provider 路由到不同的 HTTP 代理）、连接池复用、请求追踪等高级功能，而 Provider 适配器完全不感知这些基础设施层的存在。

目前 KeyCompute 已实现的 Provider 适配器包括：

| 适配器 crate | 支持的 Provider | 备注 |
| --- | --- | --- |
| `keycompute-openai` | OpenAI (GPT-4/4o/5) | 同时兼容所有 OpenAI 格式的第三方 API |
| `keycompute-claude` | Anthropic Claude | Claude 4/3.7/3.5 系列 |
| `keycompute-gemini` | Google Gemini | Gemini 2.5/2.0/3 系列 |
| `keycompute-deepseek` | DeepSeek | V3/V4/R1 系列 |
| `keycompute-ollama` | Ollama | 本地模型（Llama/Qwen 等） |
| `keycompute-vllm` | vLLM | 自托管高性能推理引擎 |

## 总结

> 类型安全贯穿全栈、异步并发模型精心设计、可观测性作为一等公民内建、加密存储保护敏感数据。

回顾整个 KeyCompute 的代码库，有几个工程亮点值得特别提及：

### 1. 类型安全贯穿全栈

从 `SensitiveString`（防止 API Key 意外打印到日志——`Debug` trait 实现会输出 `***` 而非明文）到 `ExecutionPlan`（编译期保证路由结果的完整性——primary 是必须的，fallback_chain 是可选的），Rust 的类型系统被充分利用来在编译期消灭一整类运行时 bug。

### 2. 零成本抽象的异步模型

基于 tokio + axum 的全异步架构，`mpsc::channel(100)` 的有界背压控制、`tokio::spawn` 的任务隔离、`tokio::select!` 的优雅关闭——每一个并发原语的选择都不是随意的，而是针对具体场景（流式传输、信号处理、后台维护）精心匹配的。

### 3. 可观测性作为一等公民

Prometheus 指标导出、JSON 结构化日志（通过 `tracing` + `tracing-subscriber` 实现）、`/health` 健康端点——这些不是"上线后再补"的运维工具，而是第一天就内置的核心能力。从代码中随处可见的 `tracing::info!` 和 `tracing::warn!` 可以看出，可观测性是与业务逻辑同时编写的。

### 4. 加密存储与安全设计

上游 Provider 的 API Key 使用 AES 加密存储，用户密码使用 Argon2 哈希（当前最安全的密码哈希算法），JWT token 有明确的过期时间。`KC__CRYPTO__SECRET_KEY` 的"一旦使用不可更换"策略虽然在运维上增加了约束，但从安全角度看是正确的——它迫使运维人员在首次部署时就认真对待密钥管理。

::: block-1
:mdnice-bold-italic[KeyCompute 不仅是一个 API 网关，更是一个完整的"AI 算力操作系统"——从路由决策到流式传输，从精确计费到分布式节点调度，每一层都体现了对生产环境严苛要求的深刻理解]。94.7% 的 Rust 代码比例不是对"新潮语言"的追捧，而是对性能、安全性和正确性的工程承诺。

对于正在构建 AI 基础设施的团队而言，KeyCompute 的架构设计思路——特别是其"唯一执行层"约束、"后置计费"哲学和"两层路由"引擎——值得深入借鉴与学习。
:::

> 📌 项目地址：https://github.com/keycompute/keycompute📖 开源协议：MIT License🦀 最低 Rust 版本要求：1.92+
