关键词：**跨平台 AI 编程**、Express 桥接层、:mdnice-bold-italic[WebSocket 实时通信]、Cloudflare 隧道穿透、**Telegram 机器人桥接**

> 上周五，OpenAI 宣布 Codex 手机版在 ChatGPT App 中上线，安卓和 iOS 版都已开启 preview，面向包括免费版的所有用户，但是总感觉可玩儿性差点意思。
>
> 如果你在地铁上掏出手机，打开 Termux，输入一行 `npx codexapp`，三秒后浏览器里就跑起了 OpenAI Codex 的完整编程界面——这不是概念视频，而是 codex-mobile 项目正在做的事。

Codex 官方桌面应用被锁定在特定平台的 Electron 壳里，而 codex-mobile 用一层不到 300 行的 Express 中间件加上 Vue 3 前端，把桌面应用的 app-server 协议完整代理到浏览器。

![文章插图](/articles/2026-05-19-codex-linux-windows-android-termux-520fa834/images/01.webp)

- 🚀 **Run Codex Mobile Anywhere: Linux, Windows, or Termux on Android** 🚀
- https://github.com/friuns2/codex-mobile
- 4000 字，阅读 20 分钟，**播客 22 分钟**

相关推荐

- [Claude 实现、Codex 审查、人类决策领航：Humanize 项目用 RLCR 编排长程闭环迭代任务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901161&idx=1&sn=88251ef01180c0f214ca6517d40beda1&scene=21#wechat_redirect)
- [暴涨 45 个 Star！ptx-isa-markdown 语料金矿：借助 Claude Code 突破 NVIDIA 文档的检索壁垒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899150&idx=1&sn=6f293554cfda635bbba6c8ebfc3de823&scene=21#wechat_redirect)
- [TokenSpeed 把 LLM 推理拆成可控流水线，为智能体编码时代重写 LLM 推理引擎的控制平面](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900928&idx=1&sn=a7a2425adc6733e807494b3ce7dbcc82&scene=21#wechat_redirect)

> 更妙的是，它通过自动拉起 Cloudflare 快速隧道、生成 QR 码，甚至桥接 Telegram Bot，让这个 AI 编程助手可以从地球上任何一块屏幕访问。

![文章插图](/articles/2026-05-19-codex-linux-windows-android-termux-520fa834/images/02.webp)

::: image-caption
“Skills & Apps”插件管理界面，当前弹出了“Computer Use”插件的详情窗口。该插件由OpenAI开发，已安装，其核心功能是让Codex能够通过MCP服务器控制macOS上的桌面应用程序。窗口右下角提供了“Uninstall（卸载）”和“Disable（禁用）”选项，背景中可见Linear、Slack、Figma等其他集成插件，整体界面简洁，以卡片式布局呈现各类应用插件，用户可在此管理、查看Codex的扩展功能。
:::

![文章插图](/articles/2026-05-19-codex-linux-windows-android-termux-520fa834/images/03.webp)

::: image-caption
这张截图展示了 Chat 对话界面，左侧是线程列表，包含对话历史，底部输入框支持通过“/”唤起技能，下方可选择“gpt-5.3-codex”模型、启用技能插件并调整响应等级，整体界面专为开发场景设计，助力用户复刻Termux终端模拟器，实现AI辅助开发流程。
:::

![文章插图](/articles/2026-05-19-codex-linux-windows-android-termux-520fa834/images/04.webp)

::: image-caption
Mobile UI 截图，展示了OpenClaw社区的“Skills Hub”界面，是一个用于浏览、发现和管理技能插件的平台。页面顶部设有搜索栏，提示用户可搜索“flight、docker”等技能，标注社区共有8166个技能资源；下方“Installed (40)”分类中，以卡片形式列出了用户已安装或禁用的本地技能，如“add-ui-to-electron-app”“api-keys”“baoyu-post-to-x”“codex-app-parity”等，部分技能标注“Installed”表示启用状态，“Disabled”则代表已停用，整体界面简洁，聚焦开发任务，结合 Codex 模型与社区技能，为复刻终端工具提供代码生成与开发支持。
:::

![文章插图](/articles/2026-05-19-codex-linux-windows-android-termux-520fa834/images/05.webp)

::: image-caption
Mobile UI 截图，“New thread” 对话界面，核心任务为 “Let's build termux clone”，即开发一个 Termux 终端模拟器的复刻版本。界面顶部显示线程操作图标，中间展示当前项目目标；底部是输入框，提示可通过 “/” 唤起技能，下方配置栏显示当前选用 “gpt-5.3-codex” 模型，支持自定义技能、调整响应等级（当前为 “Low”），并提供语音输入与发送按钮。整体界面简洁，聚焦开发任务，结合 Codex 模型与社区技能，为复刻终端工具提供代码生成与开发支持。
:::

本文将深入这个项目的源码，从 CLI 启动引擎到 RPC 桥接层，从安全认证到实时通知机制，逐层拆解它的架构设计与工程智慧。

## 本文目录

- 快速上手
- 一、架构设计
  - 1.1 项目目录结构概览
- 二、CLI 引擎：从一条命令到完整运行时
  - 2.1 自动依赖安装与平台适配
  - 2.2 Cloudflare 隧道自动管理
  - 2.3 端口回退与优雅关闭
- 三、HTTP 服务器：Express 中间件编排的艺术
  - 3.1 WebSocket 附着机制
- 四、安全认证：从时序安全到多层信任
  - 4.1 时序安全的密码比较
  - 4.2 多层信任机制
- 五、浏览器端 RPC 客户端：WebSocket 优先的双通道设计
  - 5.1 WebSocket + SSE 双通道自动降级
  - 5.2 RPC 调用封装
- 六、Telegram 机器人桥接：从聊天到编程的跨平台通道
  - 6.1 长轮询与消息路由
  - 6.2 双向映射：Chat ↔ Thread
- 七、密码生成与文件安全
- 总结与思考

![文章插图](/articles/2026-05-19-codex-linux-windows-android-termux-520fa834/images/06.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> 只需确保 Node.js 18+ 环境，一条命令即可启动：

```
npx codexapp# 浏览器打开 http://localhost:5900
```

在 Android Termux 上：

```
pkg update && pkg upgrade -ypkg install nodejs -ynpx codexapp
```

启动后会自动检测并安装 Codex CLI，生成访问密码，并尝试建立 Cloudflare 隧道。使用 `--no-tunnel` 关闭隧道，`--no-login` 跳过自动登录。更多选项参考 README[1]。若需通过 iPhone/iPad 访问，推荐 Tailscale Serve 方案，详见 Tailscale 章节[2]。

## 一、架构设计

> codex-mobile 的核心架构是一个经典的三层模型：

```
┌─────────────────────────────┐│  Browser (Desktop/Mobile)   │  ← Vue 3 SPA + WebSocket 客户端└──────────────┬──────────────┘               │ HTTP / WebSocket┌──────────────▼──────────────┐│         codexapp            │  ← Express 中间件 + 认证 + 隧道│  (Express + Vue UI bridge)  │└──────────────┬──────────────┘               │ RPC / 进程桥接┌──────────────▼──────────────┐│      Codex App Server       │  ← OpenAI 官方 app-server 进程└─────────────────────────────┘
```

这种设计的**核心哲学是"不重写，只桥接** "。codex-mobile 并不试图重新实现 Codex 的 AI 推理逻辑，而是:mdnice-bold-italic[完整代理官方 app-server 的 RPC 协议，把原本只能在 Electron 内部调用的接口暴露为标准 HTTP + WebSocket 端点。这意味着 Codex 官方的任何功能升级，codex-mobile 可以近乎零成本地跟进。]

### 1.1 项目目录结构概览

```
src/├── cli/index.ts          # CLI 入口：端口监听、隧道管理、Codex 安装├── server/│   ├── httpServer.ts      # Express 应用组装：路由注册、WebSocket 附着│   ├── codexAppServerBridge.ts  # 核心桥接层（290KB 巨型文件）│   ├── authMiddleware.ts  # 密码认证与会话管理│   ├── telegramThreadBridge.ts  # Telegram Bot 双向消息桥│   └── ...├── api/│   ├── codexRpcClient.ts  # 浏览器端 RPC 客户端│   └── codexGateway.ts    # API 网关（123KB）├── App.vue               # Vue 主组件（212KB 单文件组件）└── main.ts               # Vue 应用入口
```

值得注意的是，`codexAppServerBridge.ts` 体量达 290KB，`App.vue` 达 212KB，`codexGateway.ts` 达 123KB——这不是代码臃肿，而是项目将大量核心逻辑（RPC 代理、协议翻译、UI 状态管理）集中在关键文件中的务实选择。

## 二、CLI 引擎：从一条命令到完整运行时

`src/cli/index.ts` 是整个项目的入口，使用 `commander` 库构建命令行界面。它的职责远超"启动一个服务器"，实际上承担了一套完整的**运行时引导流程**。

### 2.1 自动依赖安装与平台适配

> CLI 启动时的第一要务是确保 Codex CLI 可用。项目通过探测策略自动解决这个问题：

```
// 来源：src/commandResolution.tsexportfunction resolveCodexCommand(): string | null {const explicit = process.env.CODEXUI_CODEX_COMMAND?.trim()const packageCandidates = getPotentialNpmPrefixes().flatMap(getPotentialCodexExecutables)const fallbackCandidates = process.platform === 'win32'    ? [...packageCandidates, 'codex']    : ['codex', ...packageCandidates]for (const candidate of uniqueStrings([explicit, ...fallbackCandidates])) {    if (isRunnableCommand(candidate, ['--version'])) {      return candidate    }  }returnnull}
```

这段代码展示了一个精巧的**多路径探测链**：先检查环境变量中的显式配置，再遍历各种 npm 全局安装前缀下的可执行文件路径，最后尝试系统 PATH 中的 `codex` 命令。

Windows 和 Linux/macOS 上的搜索顺序刻意不同——Windows 优先搜索包目录（因为全局 npm 的 PATH 配置经常出问题），而 Unix 系统优先信任 PATH。

> 对于 Termux 环境，CLI 还有专门的判断逻辑：

```
// 来源：src/cli/index.tsfunction isTermuxRuntime(): boolean {  return Boolean(process.env.TERMUX_VERSION || process.env.PREFIX?.includes('/com.termux/'))}
```

在 Termux 上，它会优先安装社区维护的 `@mmmbuto/codex-cli-termux` 包，失败后再回退到官方 `@openai/codex`。这种梯级降级策略确保了在受限环境下的最大兼容性。

### 2.2 Cloudflare 隧道自动管理

> codex-mobile 最令人惊喜的功能之一是自动建立 Cloudflare 快速隧道，让本地服务可以从公网访问：

```
// 来源：src/cli/index.tsasyncfunction startCloudflaredTunnel(command: string, localPort: number): Promise<{  process: ReturnType<typeof spawn>  url: string}> {returnnewPromise((resolve, reject) => {    const child = spawn(command, ['tunnel', '--url', `http://localhost:${String(localPort)}`], {      stdio: ['ignore', 'pipe', 'pipe'],    })    const timeout = setTimeout(() => {      child.kill('SIGTERM')      reject(newError('Timed out waiting for cloudflared tunnel URL'))    }, 20000)    const handleData = (value: Buffer | string) => {      const text = String(value)      const parsedUrl = parseCloudflaredUrl(text)  // 正则匹配 *.trycloudflare.com      if (!parsedUrl) return      clearTimeout(timeout)      resolve({ process: child, url: parsedUrl })    }    child.stdout?.on('data', handleData)    child.stderr?.on('data', handleData)  })}
```

这里的设计非常精妙：**通过监听 cloudflared 子进程的 stdout 和 stderr，实时解析输出中的隧道 URL**。设置了 20 秒超时，如果在此期间未获取到 URL，则优雅终止子进程。隧道 URL 解析后，还会通过 `qrcode-terminal` 在终端生成 QR 码——掏出手机扫一扫就能访问，这对于 Termux 场景简直是刚需。

更智能的是，当检测到机器上存在 Tailscale IP 时，会自动禁用 Cloudflare 隧道：

```
// 来源：src/cli/index.tsconst effectiveTunnel = tunnelFlagExplicit ? opts.tunnel : hasDetectedTailscaleIp()
```

这体现了项目对用户网络环境的细致考量——如果你已经在用 Tailscale 组网，公网隧道就是多余的。

### 2.3 端口回退与优雅关闭

> 端口监听采用逐次递增的回退策略：

```
// 来源：src/cli/index.tsconst attempt = (port: number) => {  const onError = (error: NodeJS.ErrnoException) => {    if (error.code === 'EADDRINUSE' || error.code === 'EACCES') {      attempt(port + 1)  // 端口占用或权限不足时自动尝试下一个端口      return    }    reject(error)  }  server.once('error', onError)  server.once('listening', onListening)  server.listen(port, '0.0.0.0')  // 绑定所有网卡，支持局域网访问}
```

关闭流程同样考虑周全：先终止 cloudflared 子进程，再关闭 HTTP 服务器，最后释放桥接资源。设置了 5 秒强制退出定时器，防止挂起进程。

## 三、HTTP 服务器：Express 中间件编排的艺术

`src/server/httpServer.ts` 负责将所有功能模块组装成一个完整的 Express 应用。其中间件注册顺序极为讲究：

```
// 来源：src/server/httpServer.tsexportfunction createServer(options: ServerOptions = {}): ServerInstance {const app = express()const bridge = createCodexBridgeMiddleware()const authSession = options.password ? createAuthSession(options.password) : null// 1. 认证中间件（最先执行，拦截未授权请求）if (authSession) { app.use(authSession.middleware) }// 2. Codex API 桥接（所有 /codex-api/* 请求转发到 app-server）  app.use(bridge)// 3-7. 本地文件服务系列路由  app.get('/codex-local-image', ...)      // 本地图片代理  app.get('/codex-local-file', ...)       // 本地文件代理  app.get('/codex-local-directories', ...)// 目录列表 JSON  app.get('/codex-local-browse/*', ...)   // 目录浏览 HTML  app.get('/codex-local-edit/*', ...)     // 文件编辑器  app.put('/codex-local-edit/*', ...)     // 文件保存// 8. Vue 构建产物静态服务  app.use(express.static(distDir))// 9. SPA fallback（所有未匹配路由返回 index.html）  app.use((_req, res) => { res.sendFile(spaEntryFile) })}
```

这个顺序保证了：认证先行，API 请求优先处理（不经过静态文件逻辑），本地文件操作路由精确匹配，最后才是 SPA 兜底。这是一个教科书级的 Express 中间件编排。

### 3.1 WebSocket 附着机制

WebSocket 连接复用 HTTP 服务器的 `upgrade` 事件，避免了额外端口：

```
// 来源：src/server/httpServer.tsattachWebSocket: (server: HttpServer) => {const wss = new WebSocketServer({ noServer: true })  server.on('upgrade', (req, socket, head) => {    const url = new URL(req.url ?? '', 'http://localhost')    if (url.pathname !== '/codex-api/ws') return    // 认证检查：WebSocket 升级请求也需要通过密码验证    if (authSession && !authSession.isRequestAuthorized(req)) {      socket.write('HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n')      socket.destroy()      return    }    wss.handleUpgrade(req, socket, head, (ws) => {      wss.emit('connection', ws, req)    })  })  wss.on('connection', (ws) => {    ws.send(JSON.stringify({ method: 'ready', params: { ok: true } }))    const unsubscribe = bridge.subscribeNotifications((notification) => {      if (ws.readyState !== 1) return      ws.send(JSON.stringify(notification))    })    ws.on('close', unsubscribe)  })}
```

桥接层的通知通过 `subscribeNotifications` 订阅模式分发给所有 WebSocket 客户端。每个连接建立时先发一个 `ready` 消息确认握手成功，断开时自动取消订阅——没有内存泄漏的空间。

## 四、安全认证：从时序安全到多层信任

> `src/server/authMiddleware.ts` 实现了一套适合个人/小团队场景的认证体系，虽然不是企业级 OAuth，但细节处理非常到位。

### 4.1 时序安全的密码比较

```
// 来源：src/server/authMiddleware.tsfunction constantTimeCompare(a: string, b: string): boolean {  const bufA = Buffer.from(a)  const bufB = Buffer.from(b)  if (bufA.length !== bufB.length) return false  return timingSafeEqual(bufA, bufB)}
```

使用 `crypto.timingSafeEqual` 进行密码比较，防止时序攻击。这是一个很多小型项目会忽略的安全细节。

### 4.2 多层信任机制

> 认证逻辑并非一刀切，而是建立了分层信任模型：

```
// 来源：src/server/authMiddleware.tsfunction isAuthorizedByRequestLike(remoteAddress, hostHeader, cookieHeader, validTokens) {// 第一层：本地请求直接放行（localhost + Host 头双重检查）if (isLocalhostRemote(remote) && isLocalhostHost(hostHeader ?? '')) returntrue// 第二层：Tailscale 内网 IP 信任（100.64-127.x.x.x 段）if (isTrustedTailscaleRemote(remote)) returntrue// 第三层：Cookie 中的 session token 校验const token = cookies[TOKEN_COOKIE]returntypeof expiresAt === 'number' && expiresAt > Date.now()}
```

localhost 请求免密码、Tailscale 内网 IP 免密码、外部请求走 Cookie Session——三层信任各司其职，既保证了本地开发的零摩擦体验，又为远程访问提供了足够的安全性。

Session 持久化到 `~/.codex/webui-auth-sessions.json`，Token 上限 128 个，使用原子写入（先写临时文件再 rename）保证文件完整性，权限设置为 `0o600`（仅当前用户可读写）。

## 五、浏览器端 RPC 客户端：WebSocket 优先的双通道设计

`src/api/codexRpcClient.ts` 展示了浏览器端如何与服务器保持实时通信。

### 5.1 WebSocket + SSE 双通道自动降级

```
// 来源：src/api/codexRpcClient.ts（简化）const attachWebSocket = (attempt = 0) => {const socket = new WebSocket(`${protocol}//${window.location.host}/codex-api/ws`)let didOpen = false// 2.5 秒内未建立连接则降级到 SSElet fallbackTimer = window.setTimeout(() => {    if (didOpen || closed) return    socket.close()    attachSse()  // 降级到 Server-Sent Events  }, 2500)  socket.onopen = () => { didOpen = true; clearReconnectTimer() }  socket.onclose = () => {    if (!didOpen) { attachSse(); return }  // 首次连接失败直接降级    scheduleReconnect(() => attachWebSocket(attempt + 1), attempt)  // 指数退避重连  }}
```

这个设计堪称实时通信的最佳实践：优先使用 WebSocket（低延迟、双向通信），连接失败时自动降级到 SSE（兼容性更好），重连采用指数退避（1s、2s、4s...最大 10s），避免在网络波动时给服务器造成风暴。

### 5.2 RPC 调用封装

> 所有 Codex API 调用通过统一的 `rpcCall` 函数：

```
// 来源：src/api/codexRpcClient.tsexportasyncfunction rpcCall<T>(method: string, params?: unknown): Promise<T> {const response = await fetch('/codex-api/rpc', {    method: 'POST',    headers: { 'Content-Type': 'application/json' },    body: JSON.stringify({ method, params: params ?? null }),  })// ... 错误处理（网络错误、HTTP 错误、信封格式验证）const envelope = payload as RpcEnvelope<T>return envelope.result}
```

三层错误处理覆盖了网络异常、HTTP 状态码异常、响应格式异常，每层都携带结构化的错误上下文（方法名、状态码、错误类型），方便上层 UI 精准展示错误信息。

## 六、Telegram 机器人桥接：从聊天到编程的跨平台通道

`src/server/telegramThreadBridge.ts` 实现了一个完整的 Telegram Bot 桥接系统，让用户可以通过 Telegram 与 Codex 对话。

### 6.1 长轮询与消息路由

```
// 来源：src/server/telegramThreadBridge.tsprivateasync pollLoop(): Promise<void> {while (this.active) {    try {      const updates = awaitthis.getUpdates()  // Telegram Long Polling，超时 45 秒      for (const update of updates) {        this.nextUpdateOffset = Math.max(this.nextUpdateOffset, updateId + 1)        awaitthis.handleIncomingUpdate(update)  // 路由到命令处理器      }    } catch (error) {      awaitnewPromise((resolve) => setTimeout(resolve, 1500))  // 错误后冷却    }  }}
```

### 6.2 双向映射：Chat ↔ Thread

桥接器维护了一个双向映射表，将 Telegram Chat ID 与 Codex Thread ID 关联：

```
// 来源：src/server/telegramThreadBridge.tsprivate bindChatToThread(chatId: number, threadId: string): void {// 解绑旧的映射关系const previousThreadId = this.threadIdByChatId.get(chatId)if (previousThreadId && previousThreadId !== threadId) {    const previousSet = this.chatIdsByThreadId.get(previousThreadId)    previousSet?.delete(chatId)  }// 建立新映射（一个 Thread 可以映射多个 Chat）this.threadIdByChatId.set(chatId, threadId)const chatIds = this.chatIdsByThreadId.get(threadId) ?? new Set<number>()  chatIds.add(chatId)this.chatIdsByThreadId.set(threadId, chatIds)}
```

当 Codex app-server 推送 `turn/completed` 通知时，桥接器自动提取 assistant 回复并转发给所有关联的 Telegram 聊天：

```
// 来源：src/server/telegramThreadBridge.tsprivateasync handleNotification(notification): Promise<void> {if (notification.method !== 'turn/completed') returnconst threadId = this.extractThreadId(notification)const chatIds = this.chatIdsByThreadId.get(threadId)const assistantReply = awaitthis.readLatestAssistantMessage(threadId)for (const chatId of chatIds) {    awaitthis.sendTelegramMessage(chatId, assistantReply)  // HTML 格式发送，失败则降级纯文本  }}
```

Markdown 到 Telegram HTML 的转换也做了精心处理，**支持代码块、加粗、斜体、链接**，发送失败时自动降级为纯文本——这种韧性设计在真实网络环境中极为重要。

## 七、密码生成与文件安全

> 一个小而精妙的细节体现在密码生成模块中：

```
// 来源：src/server/password.tsimport { randomInt } from 'node:crypto'const CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789'export function generatePassword(): string {  return `${randomGroup(3)}-${randomGroup(3)}-${randomGroup(3)}`  // 输出格式如：a7k-x2m-9bf}
```

使用 `crypto.randomInt` 而非 `Math.random`，保证了密码的密码学安全性。`xxx-xxx-xxx` 的分组格式既方便人类记忆，又便于口述传递——这在 Termux + 手机浏览器的场景下非常实用。

密码文件写入时设置权限为 `0o600`，确保只有当前用户可以读取。

## 总结与思考

> codex-mobile 的工程价值在于它证明了一种务实的"解锁"思路：不是重写整个 AI 编程助手，而是在官方 app-server 之上架设一层薄薄的 Web 桥接层，就能将一个平台锁定的桌面应用变成一个真正跨平台的 Web 应用。

几个值得学习的工程决策：

1. **CLI 即运行时引导器**：自动安装依赖、探测环境、管理子进程，让 `npx codexapp` 一条命令背后是一整套零配置的引导链。
2. **双通道实时通信**：WebSocket 优先 + SSE 降级 + 指数退避重连，覆盖了从理想到恶劣的全部网络条件。
3. **分层信任认证**：localhost 免密 → Tailscale 免密 → Cookie Session → 密码登录页，按场景递进，既安全又不折腾用户。
4. **桥接而非重写**：290KB 的桥接层代码量不小，但它的价值在于零维护成本地跟随上游 Codex 的协议演进。

这个项目从工程哲学上给我们的启示是：**在 AI 工具链快速迭代的时代，最有效率的创新往往不是从零造轮子，而是在已有基础设施上架设正确的桥梁。**

::: block-1
当 OpenAI 的 Codex 只对特定平台开放桌面体验时，codex-mobile 用不到两万行 TypeScript/Vue 代码，**就让任何一台能运行 Node.js 的设备成为了 AI 编程助手的入口** ——这正是开源社区最擅长的事。
:::

参考资料

[1] README: *https://github.com/friuns2/codex-mobile#-quick-start*

[2] Tailscale 章节: *https://github.com/friuns2/codex-mobile#iphone--ipad-via-tailscale-serve*
