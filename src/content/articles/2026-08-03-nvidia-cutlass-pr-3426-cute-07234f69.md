关键词：**CuTe IR**、MLIR Dialect、cutegen、**cute-to-base**、:mdnice-bold-italic[NVVM Lowering]

![文章插图](/articles/2026-08-03-nvidia-cutlass-pr-3426-cute-07234f69/images/01.webp)

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

::: block-1
其长期价值在于将原先隐式存在于模板元编程中的 CuTe 代数显式化为可验证、可测试、可优化的 IR 体系，:mdnice-highlight[为后续 DSL、JIT、GPU kernel pipeline 与 MLIR 生态集成提供基础。]
:::

## 本文目录

- 一、CuTe 代数进入 MLIR 的整体变更图景
- 二、架构与设计影响
- 三、关键代码实现
- 四、测试、工具链与工程可维护性
- 总结：整体判断与后续关注点

![文章插图](/articles/2026-08-03-nvidia-cutlass-pr-3426-cute-07234f69/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[无 GPU 也能玩转 CuTe！tensor-layouts：一份纯 Python 的 CuTe 代数实现，让 GPU 张量布局变成可读、可玩、可视](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903658&idx=1&sn=dbba694188a91680d9149ca9a1596276&scene=21#wechat_redirect)**
- **[对标 NVIDIA CuTe：AMD FlyDSL 通过 MLIR 显式布局实现 Preshuffle GEMM 等 Kernel 高效优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897712&idx=1&sn=e538ede28ebec0de2eb29e8ca45a7cef&scene=21#wechat_redirect)**
- **[解锁 GPU GEMM 极致性能！PyTorch 集成 CuteDSL 后端，覆盖 BF16/FP8/FP4](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900155&idx=1&sn=a315649b04e8288899a6eac08ca47107&scene=21#wechat_redirect)**

:::
