关键词：RIGEL、**M4 Max GPU**、Metal 4.1、**张量计算逆向**、:mdnice-bold-italic[fp8 矩阵乘法]

> Metal 4.1 对外暴露了基于协作张量分片的 MPP `matmul2d` 张量计算通路，同时新增 fp8、fp4、MXFP4 等低精度格式，**但官方规范仅定义接口，刻意隐去硬件执行逻辑、加速能力、累加精度与分片布局等关键信息**。

![文章插图](/articles/2026-06-21-rigel-apple-m4-max-gpu-af82e08b/images/01.webp)

- **RIGEL: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU**
- https://arxiv.org/pdf/2606.12765

::: related-title
**相关推荐**
:::

::: related-links
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[微基准视角下 NVIDIA Blackwell 架构深度解析：FP4/FP6赋能LLM 2.5倍推理，36.3TFLOPS FP64重塑科学计算](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896057&idx=1&sn=e3b882b56e8516ff1163d4cc300a25dd&scene=21#wechat_redirect)**

:::

> 本文提出 RIGEL 微基准逆向测试框架，以 40 核 M4 Max（Mac Studio，macOS 27.0）为测试平台，:mdnice-bold-italic[通过校验码校验、全链路溯源的测试套件，结合吞吐上限、SIMD 指令对比、功耗归因三重信号交叉验证，还原出 11 项文档隐藏或与文档矛盾的硬件特性。]

核心结论：

- **M4 Max 无专用矩阵计算单元，`matmul2d` 完全运行于 GPU 着色器 SIMD 流水线；**
- **fp8（E4M3）矩阵乘法为纯软件模拟**，峰值吞吐仅为 fp16 的 0.94 倍，仅能缩减内存占用，无性能加速效果。

文中提到全套测试代码与数据集以 MIT 协议开源，所有结果可复现。但论文里没给出仓库链接。

## 本文目录

- 一、11 项核心发现总览
- 二、实验方法：四重校验保障可信度
- 三、算子调度路径：三重信号验证无专用矩阵单元
- 四、fp8 加速性判定：纯软件模拟，无性能收益
- 五、数值语义与隐藏编译约束
- 六、协作张量分片布局逆向
- 七、优化落地：后融合有效，注意力融合失效
- 八、局限性与开源说明
- 结语

![文章插图](/articles/2026-06-21-rigel-apple-m4-max-gpu-af82e08b/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[微基准视角下 NVIDIA Blackwell 架构深度解析：FP4/FP6赋能LLM 2.5倍推理，36.3TFLOPS FP64重塑科学计算](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896057&idx=1&sn=e3b882b56e8516ff1163d4cc300a25dd&scene=21#wechat_redirect)**

:::
