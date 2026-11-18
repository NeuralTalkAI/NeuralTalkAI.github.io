关键词：**Python DSL**、:mdnice-bold-italic[张量编译器]、**HIR/TIR**、CUDA 代码生成、运行时链接

> 高性能 GPU 程序开发长期有一个尖锐矛盾：算法研究者希望用接近数学表达的方式描述张量计算，系统工程师却必须在寄存器、共享内存、线程块、launch ABI 和模板代码之间反复打磨。

![文章插图](/articles/2026-07-17-tile-ai-tilefoundry-tile-agentic-52d5a236/images/01.webp)

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

![文章插图](/articles/2026-07-17-tile-ai-tilefoundry-tile-agentic-52d5a236/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)**
- **[TileRT v0.1.3 重磅发布：超低延迟 LLM 推理再提速，GLM-5 支持上线，在线 Demo 即刻体验！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897357&idx=1&sn=5ca8dfaeb17f34d4ac452018efd9fffb&scene=21#wechat_redirect)**
- **[性能远超 vLLM 和 SGLang！TileRT：编译器驱动下的 Tile-Based Runtime](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894277&idx=1&sn=119b555c403aab345333aa5071d641e8&scene=21#wechat_redirect)**

:::
