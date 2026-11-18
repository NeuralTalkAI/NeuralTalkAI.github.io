关键词：TileLang、**AMD Radeon**、:mdnice-bold-italic[RDNA架构]、**GPU性能优化**、ROCm

> 本文依托TileLang编程语言，选取RX 7900 XTX、Radeon AI PRO R9700、Strix Halo三款覆盖RDNA3/3.5/RDNA4的AMD硬件，完成10种主流GPU计算内核的落地与深度调优。

研发过程先后定位threads参数限制、warpSize硬编码、exp2向量化编译异常四类易静默出错的编译器缺陷，通过PR#2313[1]完成源码修复，解除BM取值、BLOCK_M等参数隐性约束。

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/01.webp)

- **用 TileLang 在 AMD Radeon 上榨干 GPU 性能**
- 作者：Jason Zhang，https://zhuanlan.zhihu.com/p/2045169991885096417
- 测试代码：https://github.com/zhangnju/tilelang_notebooks
- 1.2 万字，阅读 50 分钟，**播客 35 分钟**

::: related-title
**相关推荐**
:::

::: related-links
- **[从形式化推理到 GPU 加速！Lean4 + TileLang 张量程序超优化器，让 Attention 快 4.08 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901134&idx=1&sn=f689cf4d266599f5873320f98ae75d4b&scene=21#wechat_redirect)**
- **[无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**

:::

> 针对iGPU低统一带宽痛点，落地BN=1访存优化、2-pass Online注意力等专属方案；独立显卡则以高并行块尺寸提升硬件利用率。

为保证评测公允，修正自动微分、循环卷积等失真基准，规避基准虚高加速比问题。:mdnice-bold-italic[实测RDNA4显卡GEMM峰值118.5TFLOPS对标rocBLAS]，反量化算子93TFLOPS，:mdnice-bold-italic[iGPU场景部分算子相较PyTorch最高提速193%].，**文章同步总结出适配三代RDNA的内核参数调优规律。**

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/02.webp)

::: image-caption
三款AMD GPU上TileLang相对PyTorch的算子加速倍率，黑色虚线为1倍基准线，蓝、红、绿柱分别对应gfx1100独显、gfx1151核显、gfx1201专业卡。多数算子TileLang实现性能优于PyTorch，gfx1151在反向传播Bwd bwd达到全图最高2.9倍提速，gfx1201的Flash Attn提速2.4倍；Outer Add算子里gfx1100与gfx1201同为2.2倍、Mul+RELU的gfx1100达2.0倍。仅少量GEMM等场景个别硬件略低于基准，整体iGPU优化收益普遍突出，得益于BN=1连续访存等定制优化，独显与专业卡在注意力、逐元素算子上同样收获明显加速。
:::

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/03.webp)

::: image-caption
三款AMD GPU下TileLang相对Triton的算子加速倍数，黑色虚线为1倍性能基准，蓝、红、绿色柱依次代表gfx1100独显、gfx1151集成iGPU、gfx1201专业卡，测试前已针对各硬件单独调优Triton参数。整体上TileLang多数算子性能优于Triton，gfx1151在反量化Dequant算子取得2.4倍全图表最高加速，Copy算子里iGPU与R9700同为2.2倍提速；反向传播、Softmax、GEMM等算子同样具备可观增益，仅Reduce等个别算子小幅落后基准，iGPU凭借访存定制优化在各类算子中加速表现尤为突出。
:::

> 本文完整记录调优过程与踩坑细节，附核心代码与实测数据。

## 本文目录

- 一、背景：为什么要在 AMD GPU 上做这件事
- 二、先踩一遍 TileLang 的“静默陷阱”
- 三、iGPU 的内存哲学：慢带宽倒逼精细设计
- 四、公平对比：修正 PyTorch、Triton 错误基准
- 五、各 GPU 的调优配置规律
- 六、最终性能实测数据
- 七、调优系统性结论
- 八、全文总结

![文章插图](/articles/2026-06-03-tilelang-amd-radeon-gpu-rdna4-e6d45f9d/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[从形式化推理到 GPU 加速！Lean4 + TileLang 张量程序超优化器，让 Attention 快 4.08 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901134&idx=1&sn=f689cf4d266599f5873320f98ae75d4b&scene=21#wechat_redirect)**
- **[无妥协的性能：DeepSeek-V4揭示TileLang如何在零碎小算子中追平手写上限](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901047&idx=1&sn=80caa2f8f3e9b9ed5032bed0ea196f55&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**

:::
