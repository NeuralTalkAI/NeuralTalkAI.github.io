关键词：**张量编译**、MLIR、布局传播、**CUDA Tile**、:mdnice-bold-italic[Python DSL]

> *写一个矩阵乘、逐元素加法或融合激活函数并不难，困难在于**如何把这些高层张量关系，稳定地变成适配具体 GPU 架构、拥有合理分块和内存访问方式的内核**。开发者既希望:mdnice-bold-italic[保留 Python 级表达力，又希望编译器真正理解布局、步长、动态形状与迭代空间]；既需要生成可执行代码，也需要观察每一层 IR 如何变化。*

![文章插图](/articles/2026-08-20-nvidia-tensorir-compiler-cuda-tile-e5883518/images/01.webp)

- :mdnice-bold-italic[TensorIR Compiler: TensorIR is a lightweight NVIDIA-owned MLIR compiler frontend for expressing tensor computations and lowering them to NVIDIA CUDA Tile IR.]
- https://github.com/NVIDIA/tensor-ir

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 vLLM，SGLang 吞吐！cuTile Rust 实现近零开销安全内核，GEMM 达 96% cuBLAS 性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902639&idx=1&sn=1b7eb41a0df5119f59af2413a7e8bc71&scene=21#wechat_redirect)**
- **[Tile-AI 开源 TileFoundry：基于 Tile、面向跨硬件自动生成高性能程序的 Agentic 平台](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903753&idx=1&sn=3a2d3176bb846bb22f9edb192f6f26a7&scene=21#wechat_redirect)**
- **[取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)**

:::

> NVIDIA 开源的 TensorIR 正面回应了这组矛盾。它**以 MLIR 方言承载张量计算，利用布局传播和 Tile 分析形成代码生成决策，再下降到 CUDA Tile IR** 👀，并通过 C++、C API、Python DSL、DLPack 运行时和参考实现串起编译、执行、验证全过程。

这个项目规模不算庞大，边界却很有意味：它正在成为**高层张量表达与 NVIDIA 新一代 Tile 编译基础设施之间的一座专用桥梁。**

## 本文目录

- 一、快速上手：先让第一个 TensorIR 程序跑起来
- 二、TensorIR 究竟补上了哪一层
- 三、TensorIR 方言：先保存语义，再考虑硬件细节
- 四、布局传播：这条流水线才是项目的技术中轴
- 五、两条代码生成路径：默认布局传播，保留仿射映射
- 六、从 CUDA Tile IR 到可执行内核
- 七、Python DSL：用运算符重载把函数追踪成图
- 八、验证链路：编译成功只是第一关
- 九、工程设计中最值得关注的四个选择
- 十、项目当前边界与未来观察点
- 结语：TensorIR 重新划清编译边界

![文章插图](/articles/2026-08-20-nvidia-tensorir-compiler-cuda-tile-e5883518/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[对标 vLLM，SGLang 吞吐！cuTile Rust 实现近零开销安全内核，GEMM 达 96% cuBLAS 性能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902639&idx=1&sn=1b7eb41a0df5119f59af2413a7e8bc71&scene=21#wechat_redirect)**
- **[Triton 算子如何越过 PTX 直抵 Blackwell：CUDA TileIR 后端的诞生，Triton-to-tile-IR](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903681&idx=1&sn=a30c96b2bfbc7a32d9745ca6d303ddfd&scene=21#wechat_redirect)**
- **[NVIDIA 技术博客：cuTile.jl 将 CUDA 基于分块的编程模型引入 Julia 语言](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898470&idx=1&sn=13df29073e4622bf1ac3f5dd8741f097&scene=21#wechat_redirect)**

:::
