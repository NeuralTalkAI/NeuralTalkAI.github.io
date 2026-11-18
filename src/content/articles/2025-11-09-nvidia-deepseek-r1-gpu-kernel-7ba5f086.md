关键词：DeepSeek-R1、**GPU 内核生成**、:mdnice-bold-italic[推理时扩展]、注意力机制、**Hopper GPU**、大语言模型

![文章插图](/articles/2025-11-09-nvidia-deepseek-r1-gpu-kernel-7ba5f086/images/01.webp)

- Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling
- https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/
- https://build.nvidia.com/deepseek-ai/deepseek-r1
- 3000 字，阅读 10 分钟，**播客 7 分钟**

相关推荐

- [CUDA Kernel 自动生成新范式！多轮RL+反馈迭代方案 Kevin 让生成速度超o4-mini 41%、正确性领先44%](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891083&idx=1&sn=511eaf94b8c3948457d107d8e8853641&scene=21#wechat_redirect)
- [针对AMD AI Engine空间架构的可扩展BLAS库AIEBLAS：基于自动代码生成的创新实现方案！代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891849&idx=1&sn=6f99fa04d5077256647c2d43d33cccce&scene=21#wechat_redirect)
- [Act: 首个从张量加速器 ISA 描述自动生成编译器后端工具，实现 1.77 倍性能超越手工优化库与 311ms 超大内核编译](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893521&idx=1&sn=ccd14af9d24b0afc3fd62eef8edb4b0b&scene=21#wechat_redirect)

> 本文聚焦 NVIDIA 工程师的创新实践，提出利用 DeepSeek-R1 模型结合推理时扩展技术[1]，自动化生成优化 GPU 注意力内核的解决方案。

注意力机制是 LLM 的核心，但计算复杂度随序列长度呈二次增长，且存在**多种变体，传统手动生成优化内核耗时耗力、难度极高**。

::: block-1
该方案采用闭环工作流：通过手动提示初始化，DeepSeek-R1 生成内核代码，经 Hopper H100 GPU 上的验证器分析后生成新提示，反复迭代优化 15 分钟。
:::

![文章插图](/articles/2025-11-09-nvidia-deepseek-r1-gpu-kernel-7ba5f086/images/02.webp)

::: image-caption
图2. 在NVIDIA Hopper平台上结合DeepSeek-R1的推理时扩展流程，包含“提示词初始化（Prompt Initial）”“生成注意力内核（Generate attention kernel）”“DeepSeek-R1模型”“格式处理与提示词优化（Format Prompt、Modified Prompt Refine）”“GPU代码验证器（GPU Code Verifier）”“是否满足标准（Criteria Met?）”“优化内核（Optimized Kernels）”“Hopper GPU”等环节，流程逻辑为：从提示词初始化开始，经DeepSeek-R1生成内核代码后，由验证器判断是否达标，未达标则优化提示词后重新生成，达标则输出优化内核至Hopper GPU
:::

测试显示，其在 Stanford KernelBench 基准中，**Level-1 问题内核正确率达 100%，Level-2 达 96%，部分场景性能优于工程师手动优化版本，且在多种注意力变体下实现有效提速。**

推理时扩展技术通过分配额外计算资源迭代评估，为复杂问题解决提供新思路，但目前该技术仍处于早期研究阶段，需进一步优化以适配更广泛场景及不同架构 GPU。

![文章插图](/articles/2025-11-09-nvidia-deepseek-r1-gpu-kernel-7ba5f086/images/03.webp)
