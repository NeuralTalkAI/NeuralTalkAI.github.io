关键词：AI 沙箱、微虚拟机、KVM 隔离、RustVMM、E2B 兼容

> 当你让大模型生成一段 Python 脚本并按下"执行"的那一刻，一个根本性的安全问题就已经摆上了桌面——这段未经审计的代码，将在哪里运行？

Docker 容器是多数团队的第一直觉，但共享内核的 Namespace 隔离早已被证明不堪一击：容器逃逸漏洞年年有，一旦 Agent 被诱导执行恶意代码，整台宿主机都可能沦陷。

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/01.webp)

- Cube Sandbox 是基于 RustVMM 与 KVM 构建的高性能、开箱即用的安全沙箱服务。它既支持单节点部署，也可轻松扩展至多节点集群。该服务兼容 E2B SDK，能够在 60 毫秒内创建具备完整服务能力的硬件隔离沙箱环境，同时内存开销保持在 5MB 以下。
- https://github.com/TencentCloud/CubeSandbox
- 4000 字，阅读 15 分钟，**播客 21 分钟**

> 传统虚拟机虽然安全，但分钟级的启动速度和动辄数百兆的内存开销让它在"每次对话创建一个新环境"的 AI 场景中完全失格。

有没有一种方案，能同时拥有虚拟机的内核级隔离和容器的轻量敏捷？腾讯云刚刚开源的 CubeSandbox，正是对这道难题的工程化回答。

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/02.webp)

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/03.webp)

## 本文目录

- 零、三步上手：从零到运行你的第一个安全沙箱
  - 0.1 环境要求
  - 0.2 第一步：一键安装服务
  - 0.3 第二步：创建沙箱模板
  - 0.4 第三步：用 E2B SDK 执行代码
- 一、架构总览与设计哲学
  - 1.1 请求的完整生命周期
- 二、CubeAPI：用 Rust 重写 API 网关
  - 2.1 路由注册与中间件栈
  - 2.2 异步运行时的精细控制
- 三、CubeShim：Rust 实现的 containerd Shim v2
  - 3.1 启动时的文件描述符预扩展技巧
  - 3.2 核心模块划分
- 四、CubeProxy：基于 OpenResty 的智能路由
- 五、Hypervisor 层：基于 Cloud Hypervisor 的深度定制
- 六、CubeVS 与网络安全模型
- 七、一键部署与生产实践
- 总结与展望

## 零、三步上手：从零到运行你的第一个安全沙箱

> 本节旨在让读者在 5 分钟内跑通完整链路。如需了解多节点集群部署、自定义模板、TLS 证书管理等进阶内容，请参阅 Quick Start 完整指南。

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/04.gif)

### 0.1 环境要求

- **x86_64 裸金属 Linux 服务器**，已开启 KVM（确认 `/dev/kvm` 存在）
- **Docker** 已安装并正在运行
- 具备外网访问能力（需拉取镜像和安装包）

> 💡 虚拟机（嵌套虚拟化）理论上可用但不建议用于生产；云服务器需确认实例类型支持 KVM 透传。详见 多节点部署文档[1]。

### 0.2 第一步：一键安装服务

```
curl -sL https://github.com/tencentcloud/CubeSandbox/raw/master/cube-sandbox/deploy/one-click/online-install.sh | bash
```

安装脚本会自动完成以下工作：启动 CubeAPI（端口 3000）、CubeMaster、Cubelet、CubeShim、CubeProxy（含 mkcert TLS 证书）、CoreDNS，以及 MySQL / Redis 依赖容器。

> ⚠️ 如果主网卡不是 `eth0`，需手动指定节点 IP：
>
> ```
> CUBE_SANDBOX_NODE_IP=<your-ip> bash <(curl -sL https://github.com/tencentcloud/CubeSandbox/raw/master/cube-sandbox/deploy/one-click/online-install.sh)
> ```

安装完成后，将 CLI 工具加入 PATH：

```
echo 'export PATH=/usr/local/services/cubetoolbox/CubeMaster/bin:$PATH' >> ~/.bashrcsource ~/.bashrc
```

### 0.3 第二步：创建沙箱模板

```
cubemastercli tpl create-from-image \  --image ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest \  --writable-layer-size 1G \  --expose-port 49999 \  --expose-port 49983 \  --probe 49999
```

等待模板构建完成（可通过 `cubemastercli tpl watch --job-id <job_id>` 跟踪），记录输出中的 **Template ID**。

关于自定义镜像构建模板的完整工作流，参见 Template Concepts[2]。

### 0.4 第三步：用 E2B SDK 执行代码

```
pip install e2b-code-interpreter
```

设置环境变量：

```
export E2B_API_URL="http://127.0.0.1:3000"export E2B_API_KEY="dummy"                        # 任意非空字符串即可export CUBE_TEMPLATE_ID="<第二步获得的模板ID>"export SSL_CERT_FILE="$(mkcert -CAROOT)/rootCA.pem"
```

| 变量 | 说明 |
| --- | --- |
| `E2B_API_URL` | 将 E2B SDK 指向本地 CubeAPI，而非 E2B 官方云服务 |
| `E2B_API_KEY` | SDK 内部有非空校验，本地部署填任意值即可 |
| `CUBE_TEMPLATE_ID` | 沙箱模板 ID，决定 VM 内运行什么环境 |
| `SSL_CERT_FILE` | mkcert 根证书路径，用于 SDK 访问沙箱的 HTTPS 连接 |

运行你的第一段安全隔离代码：

```
import osfrom e2b_code_interpreter import Sandbox  # 没错，直接用 E2B 官方 SDKwith Sandbox.create(template=os.environ["CUBE_TEMPLATE_ID"]) as sandbox:    # 这段代码运行在一个拥有独立内核的 KVM MicroVM 中    result = sandbox.run_code("print('Hello from Cube Sandbox!')")    print(result)    # 也可以执行 Shell 命令    cmd = sandbox.commands.run("uname -a")    print(cmd.stdout)  # 你会看到这是一个独立的 Guest 内核
```

更多使用场景（Shell 命令、文件读写、浏览器自动化、暂停/恢复、网络策略等），参见 examples/ 目录[3] 和 Example 教程[4]。

## 一、架构总览与设计哲学

> CubeSandbox 的核心设计思想可以概括为八个字：**分层解耦、极致裁剪**。

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/05.webp)

整个系统自上而下分为六大组件，形成一条从 HTTP 请求到 MicroVM 内进程执行的完整链路：

| 组件 | 语言 | 职责 |
| --- | --- | --- |
| **CubeAPI** | Rust (Axum) | E2B 兼容的 REST API 网关 |
| **CubeMaster** | Go | 集群编排调度器，管理资源与状态 |
| **CubeProxy** | OpenResty/Lua | 反向代理，路由请求至目标沙箱 |
| **Cubelet** | Go | 单节点沙箱生命周期管理 |
| **CubeVS** | eBPF/C | 内核态虚拟交换机，网络隔离 |
| **CubeRuntime** | Rust | Shim + Hypervisor + Agent 三位一体 |

这个架构有意模仿了 Kubernetes 的 Master-Node 范式（CubeMaster 对应 kube-apiserver + scheduler，Cubelet 对应 kubelet），但它**不依赖 Kubernetes** ——单机一键部署即可运行。

::: block-1
这是一个关键的产品决策：AI Agent 场景下的开发者往往不需要也不想维护一个 K8s 集群，只想"给我一个安全的沙箱"。
:::

### 1.1 请求的完整生命周期

> 一个典型的沙箱创建流程如下：

1. 用户通过 E2B SDK 发送 `POST /sandboxes` 请求
2. **CubeAPI** 接收请求，转发至 **CubeMaster**
3. CubeMaster 调度至合适的 **Cubelet** 节点
4. Cubelet 通过 containerd 调用 **CubeShim** （Shim v2 协议）
5. CubeShim 请求 **Hypervisor** 启动 KVM MicroVM
6. VM 内的 **cube-agent** 就绪，沙箱可用

整个链路端到端耗时 < 60ms——这得益于资源池预置和快照克隆技术，跳过了传统 VM 的引导加载流程。

## 二、CubeAPI：用 Rust 重写 API 网关

CubeAPI 是整个系统的"门面"，承担着一个极其特殊的使命：**让用户在不修改任何业务代码的前提下，从 E2B 云服务无缝迁移到自托管沙箱**。只需更改一个环境变量 `E2B_API_URL`，E2B 官方 Python SDK 的所有调用都会被 CubeAPI 透明接管。

### 2.1 路由注册与中间件栈

> CubeAPI 基于 Axum 框架构建，路由层设计非常清晰：

```
// 来源：CubeAPI/src/routes.rspubfn build_router(state: AppState) -> Router {    let sandbox_routes = Router::new()        .route("/sandboxes", get(sandboxes::list_sandboxes))        .route("/sandboxes", post(sandboxes::create_sandbox))        .route("/v2/sandboxes", get(sandboxes::list_sandboxes_v2))        .route("/sandboxes/:sandboxID", get(sandboxes::get_sandbox))        .route("/sandboxes/:sandboxID", delete(sandboxes::kill_sandbox))        .route("/sandboxes/:sandboxID/pause", post(sandboxes::pause_sandbox))        .route("/sandboxes/:sandboxID/resume", post(sandboxes::resume_sandbox))        .route("/sandboxes/:sandboxID/connect", post(sandboxes::connect_sandbox));    // 全局中间件：请求ID、链路追踪、超时、压缩、跨域    Router::new()        .route("/health", get(health::health))        .merge(sandbox_routes)        .with_state(state)        .layer(            ServiceBuilder::new()                .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))                .layer(TraceLayer::new_for_http())                .layer(TimeoutLayer::new(Duration::from_secs(30)))                .layer(CompressionLayer::new())                .layer(CorsLayer::permissive()),        )}
```

几个值得注意的设计决策：

- **认证可选化**：当配置了 `auth_callback_url` 时，才挂载认证和限流中间件。本地开发场景下 `E2B_API_KEY=dummy` 即可通过。
- **V1/V2 双版本路由并存**：`/sandboxes` 和 `/v2/sandboxes` 同时暴露，兼容 E2B SDK 的不同版本。
- **30 秒全局超时**：通过 `tower_http::TimeoutLayer` 统一兜底，防止下游阻塞拖垮整个网关。

### 2.2 异步运行时的精细控制

> 在 `main.rs` 中，CubeAPI 并没有使用 Tokio 的 `#[tokio::main]` 宏，而是手动构建 runtime：

```
// 来源：CubeAPI/src/main.rslet mut builder = tokio::runtime::Builder::new_multi_thread();if cfg.worker_threads > 0 {    builder.worker_threads(cfg.worker_threads);}let rt = builder    .enable_all()    .thread_name("cube-api-worker")    .build()?;rt.block_on(async_main(cfg, cli.debug))
```

这看似微小的选择实际上意味深长——在高并发沙箱创建场景下（同时 50 个请求），Tokio worker 线程数直接影响请求调度的并行度。手动构建允许运维通过 `--worker-threads` 参数或环境变量精确调优，这在生产环境中至关重要。

## 三、CubeShim：Rust 实现的 containerd Shim v2

> CubeShim 是连接容器编排世界与 MicroVM 世界的"翻译官"。

它实现了 containerd Shim v2 协议，让 containerd 以为自己在管理普通容器，**实际上底层每一个"容器"都是一个拥有独立内核的 KVM 虚拟机**。

```
containerd    │  (Shim v2 API)    ▼containerd-shim-cube-rs   ← CubeShim    │  (ttrpc / vsock)    ▼cube-agent                ← 虚拟机内的 guest agent    │    ▼container workload        ← 运行在 MicroVM 内部
```

### 3.1 启动时的文件描述符预扩展技巧

> CubeShim 的 `main.rs` 中有一段极其精妙的底层优化：

```
// 来源：CubeShim/shim/src/main.rsunsafe {    // Create dummy socket and dup it to large fd, so that we could trigger    // expand_files in kernel. And this tricky work will avoid later multi    // thread try to invoke expand_files at same time, there will be lock    // contention in expand_files.    let dummy = libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);    if dummy > 0 {        libc::dup2(dummy, 512);    }}
```

这段代码在进程启动时创建一个 Unix socket，然后用 `dup2` 将它复制到 fd 512。目的是什么？

Linux 内核维护的文件描述符表（`fdtable`）在首次访问高编号 fd 时会触发 `expand_files` 进行扩容，而这个操作需要持锁。如果在多线程运行时（Tokio 的 worker 线程）才首次触发扩容，就会产生锁竞争。**通过在单线程启动阶段提前触发扩容，彻底规避了运行时的锁争用。**

这是一个典型的"用空间换时间、用启动时间换运行时性能"的系统编程技巧。

### 3.2 核心模块划分

> CubeShim 的代码结构清晰地映射了它的职责边界：

```
// 来源：CubeShim/shim/src/lib.rspubmod common;      // 版本信息、公共常量pubmod container;   // 容器生命周期（Create/Start/Kill/Delete）pubmod cube;        // Cube 运行时特有逻辑pubmod hypervisor;  // Hypervisor 交互层（VM 创建/销毁）pubmod log;         // 日志基础设施pubmod sandbox;     // 沙箱抽象（一个沙箱 = 一个 VM + N 个容器）pubmod service;     // Shim v2 gRPC 服务实现pubmod snapshot;    // 快照相关（暂停/恢复/克隆）
```

`sandbox` 和 `container` 的分离体现了一个重要的架构洞察：

- 在 Kata Containers 等安全容器方案中，一个 VM（sandbox）可以承载多个容器；
- CubeShim 继承了这一模型，但在 AI Agent 场景下通常是一对一使用。

## 四、CubeProxy：基于 OpenResty 的智能路由

> CubeProxy 解决了一个关键的"最后一公里"问题：用户的 E2B SDK 如何访问到具体的沙箱服务端口？

答案是通过域名编码：`<port>-<sandbox_id>.<domain>` 格式的 Host 头会被 CubeProxy 解析并路由到对应 VM 的对应端口。

```
# 来源：CubeProxy/nginx.confserver {    listen 8080 ssl reuseport;    server_name _;    ssl_certificate     /usr/local/openresty/nginx/certs/cube.app+3.pem;    ssl_certificate_key /usr/local/openresty/nginx/certs/cube.app+3-key.pem;    location / {        proxy_http_version 1.1;        proxy_set_header Upgrade $http_upgrade;        proxy_set_header Connection "upgrade";        proxy_send_timeout 7206s;        proxy_read_timeout 7206s;        rewrite_by_lua_file lua/rewrite_phase.lua;        proxy_pass http://backend;    }}
```

几个关键配置值得解读：

- **`reuseport`：** 启用 SO_REUSEPORT，让多个 worker 进程各自拥有独立的 accept 队列，避免惊群效应，在高并发下显著提升吞吐。
- **WebSocket 支持**：`Upgrade` + `Connection "upgrade"` 的组合使得 CDP（Chrome DevTools Protocol）等长连接协议得以穿透代理——这对浏览器沙箱场景至关重要。
- **7206 秒超时**：约 2 小时，足够覆盖长时间运行的 Agent 任务，同时避免无限等待。
- **Lua 阶段注入**：`rewrite_phase.lua` 负责从 Host 头解析沙箱 ID 和端口号，`balancer_phase.lua` 通过 `lua_shared_dict` 共享内存查找后端 IP，整个路由决策在 Lua 层完成，不需要每次请求都查询 CubeMaster。

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/06.webp)

## 五、Hypervisor 层：基于 Cloud Hypervisor 的深度定制

> `hypervisor/` 目录是整个项目中体量最大的 Rust 代码区域，它 fork 自 Cloud Hypervisor 项目并进行了面向 AI 沙箱场景的定制。从子模块结构可以窥见其完整性：

- `vmm/`：虚拟机管理器核心，负责 VM 的创建、配置和运行循环
- `hypervisor/`：KVM 接口抽象层
- `virtio-devices/`：virtio 设备模拟（网卡、块设备、控制台等）
- `vm-migration/`：VM 状态的序列化与反序列化，是快照恢复的基础
- `block_util/`、`qcow/`、`vhdx/`：多种磁盘格式支持
- `arch/`：x86_64 架构特定代码（CPUID、MSR 配置等）

![文章插图](/articles/2026-04-16-docker-cubesandbox-rust-kvm-ai-8465bbb2/images/07.webp)

::: image-caption
不同规格沙箱 Cube Sandbox 自身内存消耗。*其中蓝色部分为沙箱规格，橙色部分为对应规格下消耗内存，随着规格扩大，内存消耗呈现少量增长
:::

CubeSandbox 能做到 < 5MB 的单实例内存开销，核心秘密在于两点：

1. **极致裁剪的 Guest OS**：guest image 通过专用 Dockerfile 生成，仅包含最小化的 init 进程（cube-agent）和必要的文件系统
2. **CoW（Copy-on-Write）内存复用**：基于快照克隆技术，多个沙箱实例共享同一份 guest 内核和 rootfs 的物理内存页，只有被修改的页面才会产生新的分配

## 六、CubeVS 与网络安全模型

> CubeVS（Cube Virtual Switch）是基于 eBPF 实现的内核态虚拟交换机，位于 `CubeNet/cubevs/` 目录。它在数据平面层面实现了三个核心能力：

1. **沙箱间严格网络隔离**：不同沙箱的网络命名空间彼此不可达，即使运行在同一台物理机上
2. **细粒度出站流量过滤**：可以按策略限制沙箱访问外部网络的范围，防止 Agent 被利用发起对内网的攻击
3. **内核态包转发**：不经过用户态协议栈，减少了上下文切换开销

相比传统的 iptables/nftables 方案，eBPF 程序直接 attach 到网络设备的 TC（Traffic Control）钩子上，在包进入协议栈之前就完成过滤和转发决策，性能损耗可以忽略不计。

## 七、一键部署与生产实践

> CubeSandbox 的部署体验同样值得称道。整个 `deploy/one-click/` 目录实现了一个自包含的发布包构建和安装流程：

```
# 在 KVM 支持的机器上三步启动curl -sL .../online-install.sh | bash            # 安装服务cubemastercli tpl create-from-image \             # 创建模板  --image sandbox-code:latest \  --writable-layer-size 1G --expose-port 49999export E2B_API_URL="http://127.0.0.1:3000"       # 配置SDK
```

安装脚本会自动处理 MySQL/Redis 依赖部署、TLS 证书生成（通过内置 mkcert）、CoreDNS 的 split-DNS 配置、以及所有组件的编排启动。对于多节点场景，计算节点只需设置 `ONE_CLICK_DEPLOY_ROLE=compute` 和控制面 IP 即可加入集群。

## 总结与展望

> CubeSandbox 代表了 AI 基础设施领域的一个重要趋势：**将虚拟化技术下沉为一种透明的安全原语**。开发者不需要理解 KVM、virtio、eBPF 这些底层技术，只需要像使用 Docker 一样调用 E2B SDK，就能获得硬件级的隔离保证。

从技术实现来看，它的几个关键选择值得学习：

- **Rust 重写关键路径** （CubeAPI、CubeShim、Hypervisor）：内存安全 + 零成本抽象，在系统编程场景下兼顾性能与可靠性
- **协议兼容而非重新发明**：100% 兼容 E2B SDK 接口，降低迁移成本到趋近于零
- **eBPF 做网络安全面**：在内核态完成策略执行，不引入额外的用户态组件和性能损耗
- **快照克隆实现亚百毫秒启动**：将虚拟机的启动问题转化为内存页的 CoW 复制问题

项目路线图上还有"事件级快照回滚"这一激动人心的特性——想象一下，Agent 在探索过程中可以在任意时间点 fork 出分支环境，就像 Git 的 branch 一样轻量。这将为 AI Agent 的树搜索、强化学习训练等场景打开全新的可能性。

> CubeSandbox 已在腾讯云内部经过大规模生产验证。对于任何正在为 AI Agent 寻找安全执行环境的团队，它都值得被认真审视。

参考资料

[1] 多节点部署文档: *https://github.com/TencentCloud/CubeSandbox/blob/master/docs/guide/multi-node-deploy.md*

[2] Template Concepts: *https://github.com/TencentCloud/CubeSandbox/blob/master/docs/guide/templates.md*

[3] examples/ 目录: *https://github.com/TencentCloud/CubeSandbox/tree/master/examples*

[4] Example 教程: *https://github.com/TencentCloud/CubeSandbox/blob/master/docs/guide/tutorials/examples.md*
