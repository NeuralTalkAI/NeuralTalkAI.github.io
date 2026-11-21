关键词：:mdnice-bold-italic[低精度 GEMM]、**AMD Zen**、JIT 微内核、量化后处理、**CPU 推理加速**

> *在深度学习系统里，最昂贵的计算常常藏在一个朴素公式背后：`C = alpha * A * B + beta * C`。看起来只是矩阵乘法，落到真实推理和训练场景，却会同时牵扯数据类型、内存布局、量化尺度、激活函数、线程调度、CPU 指令集，以及缓存层级。:mdnice-bold-italic[GPU 占据了很多聚光灯，可 CPU 端的高性能 GEMM 依旧是服务器推理、混合部署、前后处理、传统数值库集成中的关键拼图。]*

![文章插图](/articles/2026-09-17-amd-aocl-dlp-gemm-jit-cee760d9/images/01.webp)

- **A library of AMD Optimized Deep Learning Primitives**
- 代码：https://github.com/amd/aocl-dlp
- 文档：github.com/amd/aocl-dlp/wiki

::: related-title
**相关推荐**
:::

::: related-links
- **[当你的数学库比编译器更懂 CPU：AMD AOCL-LibM 微架构自适应分发机制与高精度逼近算法浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901631&idx=1&sn=1d91e391905fba69873c7957808f98a3&scene=21#wechat_redirect)**
- **[AMD HRX System：ROCm 的下一代低延迟 GPU 抽象](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902946&idx=1&sn=ffd3a29bcd54169e2ecd6d733318b09d&scene=21#wechat_redirect)**
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**

:::

> AMD 开源的 AOCL-DLP（Deep Learning Primitives）正是围绕这个拼图展开：:mdnice-bold-italic[它面向 AMD 处理器优化低精度 GEMM 与 batch GEMM，用 C/C++、AVX2、AVX512、AVX512_VNNI、AVX512_BF16、JIT 生成器和后处理融合，把通用接口下沉为贴近硬件的执行路径。]

## 本文目录

::: related-links
- 一、快速上手：先把库编起来、跑起来
- 二、项目定位：围绕 GEMM 构建的深度学习原语库
- 三、API 入口：薄薄一层头文件，背后连接多条执行链
- 四、一次 FP32 GEMM 调用在内部经历了什么
- 五、Kernel Dispatch Table：为高频查询优化的内核索引
- 六、JIT 生成器：把运行时形态变成机器码
- 七、Post-ops 融合：把“矩阵乘之后的一串小算子”留在缓存里
- 八、线程、分块与 5-loop 框架：CPU GEMM 的经典骨架
- 九、量化支持：低精度不是只换一个类型
- 十、工程化亮点与适用边界
- 结语：CPU 端深度学习性能的底层答案

:::

![文章插图](/articles/2026-09-17-amd-aocl-dlp-gemm-jit-cee760d9/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD 开源 hip-ep：用 MLIR 贯通 ONNX Runtime 与 ROCm，落地 LLVM bitcode 编译](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905194&idx=1&sn=2f1918c0623bc9f0918a730bcb28a992&scene=21#wechat_redirect)**
- **[AMD GPU 设备数学库分层编译架构浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899885&idx=1&sn=c4a604edeff740435f04ea5b6f90bf7c&scene=21#wechat_redirect)**
- **[超优化驱动的数值计算库自动学习系统，实现数学原语高效挖掘与最高 2.2× 性能增益](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901822&idx=1&sn=dcdd94c36d76c146d74b9c86bb4ff13e&scene=21#wechat_redirect)**
- **[从形式化推理到 GPU 加速！Lean4 + TileLang 张量程序超优化器，让 Attention 快 4.08 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901134&idx=1&sn=f689cf4d266599f5873320f98ae75d4b&scene=21#wechat_redirect)**
- **[对标 A100：拆解摩尔线程开源的 GPU 驱动级基准测试 gpu-compute-driver-bench](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901875&idx=1&sn=3db0fb835b96a96286fc1e65ef6492e2&scene=21#wechat_redirect)**

:::
