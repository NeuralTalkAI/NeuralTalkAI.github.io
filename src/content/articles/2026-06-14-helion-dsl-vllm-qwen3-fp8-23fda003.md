**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**Helion**、vLLM、**FP8量化**、推理内核、PyTorch

> 本文围绕**Helion 内核集成 vLLM** 开展技术研究与性能评测，主要验证该内核 DSL 在 Qwen3 模型 FP8 量化推理场景下的落地效果，测试硬件为 NVIDIA H100 与 B200 GPU。

Helion 是一款 PyTorch 原生、硬件无关的 GPU 内核领域专用语言，采用 tile 编程模型，既保留 PyTorch 简易的开发体验，又支持内存布局、tile 策略等底层细节调控，还搭载强大的提前自动调优能力，可自动匹配硬件与 workload 的最优内核配置。

::: block-1
**本次实验针对 vLLM 无张量并行推理场景，分别适配逐 Token、逐分组两种动态 FP8 激活量化模式**，替换推理链路中量化、归一化、矩阵乘、算子融合等绝大部分前向内核，:mdnice-bold-italic[并对接 DeepGEMM、CUTLASS 两类主流 GEMM 后端。]
:::

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/01.png)

- **Portable vLLM Model Inference Kernels in Helion**
- https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/
- 9000 字，阅读 35 分钟，**播客 22 分钟**

::: related-title
**相关推荐**
:::

::: related-links
- **[vLLM 冷启动延迟的系统性拆解与深度优化指南](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902194&idx=1&sn=3d17f7b85eeef0daac4735440cdebb66&scene=21#wechat_redirect)**
- **[vLLM Triton 注意力后端深度解析：跨 NVIDIA、AMD 的 GPU 架构创新优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898509&idx=1&sn=341fceedfbc28e118212eb85ae5b1807&scene=21#wechat_redirect)**
- **[“历史性的项目，错过了遗憾终生”：清华特奖游凯超与 vLLM 的三次蜕变与技术远征](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896591&idx=1&sn=689f8a0351d8ea9ed0c2c0ebd2f89ca2&scene=21#wechat_redirect)**

:::

> 团队基于 Helion 开发多款全新融合内核，结合模型固定维度与 1~8192 区间的动态 Token 数完成全场景自动调优，并依靠运行时调度分发最优内核方案。

实测表明，**Helion 的非矩阵乘类算子性能全面超越 TorchInductor 与 vLLM 原生 CUDA 内核**；矩阵乘算子在 H100 上优于 CUTLASS，但因依赖的 Triton 代码生成对 B200 架构适配不足，性能出现回落。

端到端测试中，H100 平台最高实现 1.09 倍吞吐提升，B200 仅启用量化内核也可获得正向收益。同时实验也:mdnice-bold-italic[暴露出 Helion 全量调优耗时久、运行调度存在 CPU 开销、暂不兼容 UE8M0 量化格式等阶段性问题。]

**整体来看，Helion 有效降低了大模型推理内核的开发门槛，具备可观的应用价值。**

## 关键问题

### 问题一： 追求硬件极致性能是否会消解硬件无关定位

> 既然 Helion 以**PyTorch 原生、硬件无关、低开发门槛** 作为核心设计定位，而当前它在 Blackwell 架构 B200 GPU 上 GEMM 性能落后于 CUTLASS 的根源是依赖 Triton 代码生成，团队也计划引入 CuteDSL 等全新后端来优化该问题，那么**在接入多款差异化后端、针对新一代 GPU 做深度定制优化的过程中，Helion 要如何避免逐步丧失“硬件无关、开发简洁”的核心优势？**，:mdnice-bold-italic[若为追赶硬件性能不断叠加硬件专属适配逻辑，它和传统 CUDA、CUTLASS 这类硬件绑定的底层编程方案还有本质区别吗？]

不会消解 Helion 硬件无关的核心定位，反而会通过分层架构强化其“开发简洁、跨硬件可移植”的核心优势。Helion 从设计之初就采用清晰的上下分层架构：

- 上层是面向开发者的 PyTorch 风格分块编程 DSL，开发者只需关注计算逻辑与分块策略，无需感知底层硬件的指令集、内存层级等细节；
- 下层是代码生成后端层，硬件专属的架构适配、指令优化逻辑全部下沉在这一层，对上层开发者完全透明。

**引入 CuteDSL 等新后端、针对 Blackwell 等新架构做深度性能优化，所有硬件定制的复杂度都由 Helion 框架研发团队统一承接**，不会转嫁给业务侧的内核开发者。开发者依然可以用熟悉的 PyTorch 语法编写一份内核代码，通过不同后端自动适配多款硬件，这与 CUDA、CUTLASS 需要开发者手写硬件绑定逻辑、每个硬件平台重复开发的模式有本质区别。

::: block-1
**当前 B200 上 GEMM 性能落后的根源是 Triton 代码生成对 Blackwell 架构适配不足，而非 Helion 的编程模型本身存在缺陷**。整个优化过程只需迭代后端代码生成能力，上层的 Helion 内核实现无需任何改动，这恰恰验证了硬件抽象层的价值。:mdnice-bold-italic[多后端适配不会让 Helion 沦为硬件绑定的底层方案，反而会夯实“一次编写、多硬件高性能运行”的核心竞争力]，在保持低开发门槛的同时逐步逼近原生硬件的性能上限。
:::

### 问题二：三大落地硬伤是否削弱生产场景投入产出比

> **Helion 目前存在三大落地硬伤：内核全量自动调优耗时长达一整天😂、运行时调度存在 CPU 开销（小内核场景会抵消性能收益，强依赖 CUDA Graph）、暂不兼容主流 DeepGEMM 的 UE8M0 量化格式，无法适配 vLLM 分组量化的默认后端**。对比已经成熟稳定的 CUTLASS、原生 Triton 内核方案，:mdnice-bold-italic[在对延迟、吞吐、运维成本、兼容性要求极高的 LLM 线上生产场景中，Helion 的综合投入产出比体现在哪里？团队又准备通过哪些系统性方案彻底解决这三大硬伤，支撑它大规模商用部署？]

短期来看，自动调优耗时长、调度存在 CPU 开销、不兼容 DeepGEMM 格式这三大问题，确实会提升 Helion 的落地门槛，削弱部分场景的短期投入产出比，但从长期研发效率与优化空间来看，其综合产业价值依然显著。

- **针对自动调优耗时问题，全量 shape 的大规模调优可长达一天，但这是覆盖全模型、全尺寸的极端扫描场景**。:mdnice-bold-italic[实际生产环境中模型维度固定，只需针对对应隐藏层、中间层尺寸做一次调优，最优配置即可长期复用，一次性成本会被长期使用摊薄]。同时团队正在探索搜索空间剪枝、大模型引导的自动调优方案，可大幅压缩调优周期。
- 针对运行时调度开销，数十微秒的 CPU 开销确实会在小内核场景占据主导，启用 CUDA Graph 捕获重放即可完全规避该开销，而 vLLM 生产部署本身普遍依赖 CUDA Graph 做性能优化。同时 Helion 团队正在主动优化无 CUDA Graph 模式下的调度延迟，后续可覆盖更多延迟敏感的在线服务场景。
- **针对 DeepGEMM 的 UE8M0 格式不兼容问题，这只是阶段性的功能缺失，而非架构性缺陷。** 当前实验中强制切换 CUTLASS 后端已验证了量化融合内核的性能优势，后续扩展内核适配 UE8M0 格式的开发成本远低于从零开发 CUDA 内核，可快速补齐兼容性短板。

Helion 的核心价值是将内核开发验证周期压缩至单日，大幅降低内核融合、新量化方案的探索成本，长期研发效率收益远高于短期落地成本。

## 本文目录

- 关键问题
  - 问题一： 追求硬件极致性能是否会消解硬件无关定位
  - 问题二：三大落地硬伤是否削弱生产场景投入产出比
- 一、vLLM 与 Helion 简要背景
- 二、基于 Helion 内核的 vLLM 模型推理
  - 2.1 vLLM 前向传播融合逻辑
  - 2.2 Helion 内核实现
  - 2.3 vLLM 与 Helion 内核集成
  - 2.4 自动调优
  - 运行时调度
- 三、性能评测——内核层面
  - 3.1 结果分析
- 四、性能评测——模型端到端层面
  - 4.1 动态逐 Token 激活量化
  - 4.2 动态分组激活量化
- 四、相关资源
- 五、注意事项
  - 5.1 自动调优耗时问题
  - 5.2 运行时调度开销
- 总结

## 一、vLLM 与 Helion 简要背景

> vLLM[1] 是面向大语言模型（LLM）的高性能推理与部署框架。凭借出色的吞吐量表现、高效的键值缓存（KV-cache）管理、连续批处理架构，以及对推测解码、量化、分布式部署等高级推理特性的支持，vLLM 被广泛应用于大语言模型生产环境部署。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/02.jpg)

::: image-captionvLLM 整体系统架构。vLLM 采用集中式调度器 + 分布式 GPU 工作节点的架构，适配超大模型多卡部署场景。核心组件包含全局调度器、KV 缓存管理器、块表、GPU/CPU 块分配器以及模型分片模块。调度器统一管控全量请求与内存策略，块表负责逻辑块与物理块的映射，CPU 块分配器支持缓存交换以应对显存不足。该架构兼容主流的张量模型并行策略，解耦模型计算与内存管理。分布式工作节点仅负责计算，内存管控由中央模块统一执行，保障分页、缓存共享、请求抢占等机制在多卡环境中稳定运行。
:::

在内部实现中，vLLM 高度依赖自定义 GPU 内核、TorchInductor 融合机制，以及 CUTLASS、DeepGEMM 等优化后的 GEMM 后端，以此在各类硬件平台上实现超高推理效率。

> Helion[2] 是一套基于 PyTorch 开发、硬件无关的内核领域专用语言（DSL），采用分块编程模型编写高性能内核。相较于底层 CUDA 编程，Helion 沿用贴近 PyTorch 的语法体系，开发体验更友好，同时开发者仍可对内存布局、分块策略、内核调度进行底层管控。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/03.jpg)

::: image-caption创建全新 DSL 领域专用语言的核心动因，左侧点明痛点与优势：企业海量自定义算子易累积技术债务、跨硬件移植困难，而异构硬件已是行业趋势；该 DSL 可自动化 Triton 中张量索引、网格分块、参数管理等手动流程，相比 torch.compile 拥有更精细的底层管控，同时具备跨硬件性能移植、自动调优能力。右侧纵向抽象层级轴从底层硬件专属 PTX、CUDA 向上延伸，依次为 Triton、Helion、顶层硬件无关的 PyTorch，Helion DSL 居中，平衡高层抽象与底层算力调度需求。
:::

可以将其理解为**带分块能力的 PyTorch**。如果你掌握 PyTorch 或 Triton，就能快速上手 Helion。除了简洁的开发体验外，Helion 还配备强大的超前编译（AOT）自动调优组件，可遍历海量内核配置组合，针对特定负载与硬件目标自动筛选最优实现方案。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/04.jpg)

::: image-caption
Helion 的自动调优机制，单个 Helion 内核可依托配置映射生成大量 Triton 内核，分块语法会隐式构建包含块尺寸、迭代顺序等参数的搜索空间；右侧流程图展示完整流程：Helion 内核输出多组配置生成对应 Triton 算子，调优器对全部算子完成性能实测，遍历数千套参数组合后筛选出速度最优内核。整套自动调优仅需提前执行一次，调好的最优配置可直接部署，省去人工手动调试参数的繁琐。
:::

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/05.jpg)

::: image-caption
Helion DSL 的核心语法构件，分为三大模块：hl.tile 分块算子、PyTorch 标准算子兼容、控制流支持。hl.tile 能够切分迭代空间，外层分块映射 GPU 启动网格，内层转为设备端循环，分块尺寸、迭代顺序等参数可自动调优；框架原生兼容各类 PyTorch 算子，逐元素、规约算子经 TorchInductor 降级，矩阵乘、视图算子转为 Triton 底层接口，降低学习成本；顶层内核支持控制流，编译后生成多份 Helion IR FX 图，函数调用通过 make_fx 完成追踪，兼顾开发便捷性与底层硬件调度能力。
:::

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/06.png)

::: image-caption
Helion DSL 实现矩阵乘法的示例代码，先导入 Torch 与 Helion 语言库，通过 @helion.kernel 装饰器定义 GPU 内核函数。代码先读取输入张量维度并初始化输出张量，紫色框为核心分块计算逻辑：借助 hl.tile 对 m、n、k 三维度做嵌套分块循环，用 addmm 完成分块矩阵乘累加，最终将结果写入输出张量。hl.tile 会自动搜索最优分块尺寸、迭代顺序等参数，外层分块映射 GPU 计算网格，内层转化为设备循环，同时复用 PyTorch 算子写法，大幅简化高性能自定义矩阵算子的开发流程。
:::

## 二、基于 Helion 内核的 vLLM 模型推理

> 本次实验首先选用通义千问 3（Qwen3）系列模型，开启 FP8 激活量化，采用**无张量并行** 的推理模式开展测试。

- 实验目标：对比 Helion 内核与 vLLM 原生实现，验证前者能否提升推理性能。
- 实验方案：**将量化推理流程中几乎所有前向传播内核替换为 Helion 实现**，并分别在内核层面、端到端部署层面完成基准测试。

### 2.1 vLLM 前向传播融合逻辑

> 对于 Qwen3 系列模型，vLLM 未做内核融合的前向传播会按如下顺序执行内核：

1. input_norm（输入归一化）
2. fp8_quant（FP8 量化）
3. scaled_mm (qkv_proj)（缩放矩阵乘法，QKV 投影）
4. split_qkv（QKV 拆分）
5. q_norm（Q 归一化）
6. k_norm（K 归一化）
7. rope（旋转位置编码）
8. attention（注意力计算）
9. fp8_quant（FP8 量化）
10. scaled_mm (out_proj)（缩放矩阵乘法，输出投影）
11. post_attention_norm（注意力后归一化）
12. fp8_quant（FP8 量化）
13. scaled_mm (gate_up)（缩放矩阵乘法，门控-上升投影）
14. silu_and_mul（SiLU 激活与乘法运算）
15. fp8_quant（FP8 量化）
16. scaled_mm (down_proj)（缩放矩阵乘法，下降投影）

#### 动态逐 Token 激活量化

> 在启用 `torch.compile` 与 TorchInductor 融合优化后，执行流程变更为：

1. rms_norm + fp8_quant（均方根归一化 + FP8 量化）
2. scaled_mm (qkv_proj)（缩放矩阵乘法，QKV 投影）
3. split_qkv + q_norm + v_norm（QKV 拆分 + Q 归一化 + V 归一化）
4. rope（旋转位置编码）
5. attention（注意力计算）
6. fp8_quant（FP8 量化）
7. scaled_mm (out_proj)（缩放矩阵乘法，输出投影）
8. rms_norm + fp8_quant（均方根归一化 + FP8 量化）
9. scaled_mm (gate_up)（缩放矩阵乘法，门控-上升投影）
10. silu_and_mul + fp8_quant（SiLU 激活与乘法 + FP8 量化）
11. scaled_mm (down_proj)（缩放矩阵乘法，下降投影）

::: block-1补充说明：`scaled_mm` 与 `attention` 均已注册为 PyTorch 自定义算子[3]。由于 TorchInductor 无法解析这类自定义算子，二者会形成融合边界，阻碍编译器开展进一步内核融合。
:::

#### 动态分组激活量化

> 当开启动态分组激活量化，且 `scaled_mm_blockwise` 选用 DeepGEMM 作为后端时，执行流程再次变更：

1. rms_norm（均方根归一化）
2. fp8_quant (ue8m0)（UE8M0 格式 FP8 量化）
3. scaled_mm (qkv_proj, DeepGEMM)（缩放矩阵乘法，QKV 投影，DeepGEMM 后端）
4. split_qkv + q_norm + v_norm（QKV 拆分 + Q 归一化 + V 归一化）
5. rope（旋转位置编码）
6. attention（注意力计算）
7. fp8_quant (ue8m0)（UE8M0 格式 FP8 量化）
8. scaled_mm (out_proj, DeepGEMM)（缩放矩阵乘法，输出投影，DeepGEMM 后端）
9. rms_norm（均方根归一化）
10. fp8_quant (ue8m0)（UE8M0 格式 FP8 量化）
11. scaled_mm (gate_up, DeepGEMM)（缩放矩阵乘法，门控-上升投影，DeepGEMM 后端）
12. silu_and_mul（SiLU 激活与乘法运算）
13. fp8_quant (ue8m0)（UE8M0 格式 FP8 量化）
14. scaled_mm (down_proj, DeepGEMM)（缩放矩阵乘法，下降投影，DeepGEMM 后端）

::: block-1补充说明：DeepGEMM 内部采用 UE8M0 格式完成激活量化。在当前 vLLM 实现中，UE8M0 量化模式不支持 `fuse_act_quant`（激活量化融合）与 `fuse_norm_quant`（归一化-量化融合），因此无法实现这两类额外融合优化。
:::

若环境中无 DeepGEMM，系统会切换为基于 CUTLASS 的内核，此时执行流程与**动态逐 Token 激活量化** 场景基本一致。

### 2.2 Helion 内核实现

> 本次工作中，研究团队基于 Helion 实现了以下内核：

- dynamic_per_token_scaled_fp8_quant：动态逐 Token 缩放 FP8 量化
- rms_norm_dynamic_per_token_quant：均方根归一化 + 动态逐 Token 量化（融合内核）
- silu_and_mul_dynamic_per_token_quant：SiLU 激活乘法 + 动态逐 Token 量化（融合内核）
- fused_qk_norm_rope：Q/K 归一化 + 旋转位置编码（融合内核）
- per_token_group_fp8_quant：逐 Token 分组 FP8 量化
- rms_norm_per_block_quant：均方根归一化 + 分块量化（融合内核）
- silu_and_mul_per_block_quant：SiLU 激活乘法 + 分块量化（融合内核）
- scaled_mm：缩放矩阵乘法
- scaled_mm_blockwise：分块缩放矩阵乘法

其中 `scaled_mm` 与 `scaled_mm_blockwise` 内核参考了 vLLM 中原有的 Triton 实现：triton_scaled_mm[4]、w8a8_triton_block_scaled_mm[5]。`silu_and_mul_dynamic_per_token_quant` 为全新融合内核，将 `silu_and_mul` 与 `dynamic_per_token_quant` 合并为单次内核调用。 其余内核均为 vLLM 原有 `torch.ops._C` CUDA 内核的 Helion 重实现版本。

### 2.3 vLLM 与 Helion 内核集成

> 团队借助 vLLM Helion 内核集成框架[6] 完成内核接入，该框架提供以下能力：

- 自动调优组件
- 配置管理功能
- 内核注册能力
- 运行时调度能力

**为启用 Helion 内核，团队手动修改了 vLLM 的融合逻辑，将原有内核替换为对应的 Helion 融合内核**。完成融合后，两种量化模式下的前向传播执行流程如下：

#### 逐 Token 激活量化场景

1. rms_norm_dynamic_per_token_quant (Helion)
2. scaled_mm (Helion)
3. fused_qk_norm_rope (Helion)
4. attention（原生默认实现）
5. dynamic_per_token_scaled_fp8_quant (Helion)
6. scaled_mm (Helion)
7. rms_norm_dynamic_per_token_quant (Helion)
8. scaled_mm (Helion)
9. silu_and_mul_dynamic_per_token_quant (Helion)
10. scaled_mm (Helion)

#### 分组激活量化场景

1. rms_norm_per_block_quant (Helion)
2. scaled_mm_blockwise (Helion)
3. fused_qk_norm_rope (Helion)
4. attention（原生默认实现）
5. per_token_group_fp8_quant (Helion)
6. scaled_mm_blockwise (Helion)
7. rms_norm_per_block_quant (Helion)
8. scaled_mm_blockwise (Helion)
9. silu_and_mul_per_block_quant (Helion)
10. scaled_mm_blockwise (Helion)

### 2.4 自动调优

> 实验采用 Helion 默认的 LFBOTreeSearch[7] 算法，具体配置如下：

```
initial_population=FROM_RANDOM, copies=5, max_generations=20, similarity_penalty=1.0
```

为最大化性能收益，团队使用与模型编译期静态维度（如隐藏层维度、中间层维度）完全匹配的张量形状开展内核自动调优。**这也是 vLLM 结合 Helion 的核心优势：Helion 可针对海量张量形状完成调优、存储最优配置并在运行时调度，该能力同样适用于线上生产环境。**

针对动态维度 `num_tokens`（Token 数量），调优范围设置为 **1 ~ 8192** 之间的 2 的幂次数值。

以 `scaled_mm` 内核为例，其输入张量格式为 `[M, K] × [K, N]`，调优规则：

- M：取值范围 1 ~ 8192
- (K, N) 组合：对应各款 Qwen3 模型的投影层维度

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/07.png)

::: image-caption表 1：各 Qwen3 模型投影层 [K, N] 维度。该表格列出 Qwen3-1.7B、8B、32B 三款模型 Transformer 四大核心投影层的权重矩阵尺寸，涵盖注意力 qkv_proj、输出 out_proj、MLP gate_up 与 down_proj。随模型参数量提升，各层矩阵维度整体同步扩容；out_proj 均为同维度方阵，gate_up 升维、down_proj 降维构成 MLP 扩展收缩结构，qkv_proj 负责多头注意力映射。这些权重规格是前文 FP8 量化、算子自动调优、内核性能实测的核心基础参数，直接决定矩阵计算负载规模与优化适配方案。
:::

团队针对每一款被测硬件平台，对所有内核分别执行独立自动调优。

### 运行时调度

> 在程序运行阶段，Helion 集成框架[8] 会根据输入张量形状，调度匹配度最高的已调优内核配置。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/08.png)

::: image-caption该 issue 是 vLLM 项目集成 Helion 自定义内核的性能优化议题，明确了内核适配范围，梳理出线性计算、FP8 量化、归一化、集合通信等多类待优化内核，并关联对应开发任务。项目在 H100、B200 显卡上完成内核与端到端性能测试，多数内核相比原有方案实现显著加速，同时经评测验证模型精度基本无损。文中还给出完整的部署、测试复现指令，也记录了社区开发者参与交流、协作开发的相关动态。
:::

以 `scaled_mm` 为例，调度逻辑基于两个输入矩阵的维度 (M, K, N) 实现：会根据每一批请求的运行时 `num_tokens`，将 M 向上取整为最近的 2 的幂次。其余内核均采用同类调度策略。

## 三、性能评测——内核层面

> 内核级基准测试用于衡量单个 Helion 内核相对基线实现的加速比。基线选择规则如下：

- `scaled_mm`、`scaled_mm_blockwise`：以 CUTLASS 作为基线
- 其余算子：对比对象分为两类——开启 `torch.compile` 的 vLLM 原生实现、vLLM 内置 `torch.ops._C` CUDA 内核

选择依据：

1. vLLM 在逐 Token 量化场景下默认启用 `torch.compile`
2. 受性能问题影响[9]，分组量化场景下 vLLM 默认使用 `torch.ops._C` CUDA 实现

针对 `torch.compile` 基线，团队完全复用 vLLM 原生编译配置：

```
torch.compile(    native_torch_impl,    fullgraph=True,    dynamic=False,    backend="inductor",    options={        'enable_auto_functionalized_v2': False,        'size_asserts': False,        'alignment_asserts': False,        'scalar_asserts': False,        'combo_kernels': True,        'benchmark_combo_kernel': True    })
```

> 关键说明：开启 `'combo_kernels': True` 后，TorchInductor 可将多个独立内核融合为单次调用，对性能至关重要。

所有内核级测试均通过 `triton.testing.do_bench_cudagraph` 启用 CUDA Graph 模式，并执行充分预热与重复测试，消除调度开销、缓存冷启动、计时波动等干扰因素。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/09.png)

::: image-caption表 2：Helion 内核几何平均加速比汇总。该表格统计多款自定义算子在 H100、B200 显卡上，相对 torch.compile、torch.ops._C、CUTLASS 基线的吞吐加速倍数，N/A 代表无对应对照数据。各类 FP8 量化、归一化融合算子在两块硬件均实现正向提速，分块量化的 rms_norm、silu_and_mul 算子收益最高，相对原生 torch.ops._C 最高可达 2.325 倍，且 B200 多数算子加速幅度优于 H100；仅 scaled_mm 矩阵算子表现分化，H100 小幅提速，B200 对比 CUTLASS 基线出现性能下降，整体验证融合量化内核可显著优化大模型推理算子效率。
:::

### 3.1 结果分析

1. **非 GEMM 类内核**：Helion 表现优异，在 H100、B200 两款 GPU 上，性能全面超越 TorchInductor 生成内核与 vLLM 原生 CUDA 内核。
2. **GEMM 类内核（scaled_mm / scaled_mm_blockwise）**：表现存在差异

::: block-1
:mdnice-bold-italic[B200 平台性能短板的核心原因：Helion 的 GEMM 内核当前依赖 Triton 生成代码，而 Triton 针对 Blackwell 架构 GPU 的 GEMM 实现本身存在性能短板，并非 Helion 编程模型本身存在缺陷]。目前 Helion 团队正在研发 CuteDSL 后端，有望大幅提升 Blackwell 架构 GPU 上的 GEMM 性能。
:::

## 四、性能评测——模型端到端层面

> 端到端模型级测试用于直观体现 Helion 内核在实际部署中的性能收益。本次测试选用三款 Qwen3 模型：

- Qwen3-1.7B
- Qwen3-8B
- Qwen3-32B

所有模型级测试均启用 CUDA Graph 模式；Token 数量 `num_tokens` 取值为 1 ~ 8192 之间的 2 的幂次，覆盖全梯度负载。

测试负载构建：使用 vLLM 内置部署基准工具，输入数据为随机数据。

为消除前缀缓存带来的结果干扰，实验做了两项约束：

1. 禁用提示词随机打乱功能
2. 每轮基准测试前重启 vLLM 服务

服务启动与基准测试命令示例：

```
vllm serve --model $MODEL --max-num-seqs $BATCH_SIZE --tensor-parallel-size 1 --compilation-config '{"max_cudagraph_capture_size": 8192, "custom_ops": ["+quant_fp8"], "pass_config": {"fuse_norm_quant": true, "fuse_act_quant": true, "enable_qk_norm_rope_fusion": true}}'vllm bench serve \  --backend vllm \  --model $MODEL \  --endpoint /v1/completions \  --dataset-name random \  --num-prompts $NUM_PROMPTS \  --max-concurrency $BATCH_SIZE \  --input-len 512 \  --output-len 600 \  --num-warmups $NUM_WARMUPS \  --disable-shuffle
```

参数说明：`max_cudagraph_capture_size` 设置为 8192，与默认最大批处理 Token 数保持一致，确保所有执行链路均可被 CUDA graph 捕获。

整套测试在两款英伟达 GPU 上完成：**NVIDIA H100**、**NVIDIA B200**。

为精准定位性能提升来源，研究团队将 Helion 内核划分为三大类，分别单独测试与组合测试：

1. **fp8_quant**：FP8 量化内核及量化融合内核
2. **qk_norm_rope**：`fused_qk_norm_rope` 融合内核
3. **scaled_mm**：`scaled_mm` / `scaled_mm_blockwise` 矩阵乘法内核

### 4.1 动态逐 Token 激活量化

> 本次测试使用的模型权重：

- RedHatAI/Qwen3-1.7B-FP8-dynamic
- RedHatAI/Qwen3-8B-FP8-dynamic
- RedHatAI/Qwen3-32B-FP8-dynamic

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/10.png)

::: image-caption图 1：H100 开启逐 Token 激活量化后的整体吞吐量加速比（基线为 vLLM 默认配置）。图中左右两张折线图分别展示 H100 显卡下 Qwen3-1.7B、Qwen3-8B 动态 FP8 量化的吞吐加速表现，横轴为 Batch Size，纵轴是吞吐加速倍数，四条曲线对应 scaled_mm、qk_norm_rope、fp8_quant 单算子与全部算子叠加优化方案。1.7B 模型小批量下全算子优化峰值达 1.10 倍，单算子曲线波动明显；8B 模型整体提速幅度更小，全算子优化随批量增大稳步上涨至 1.05 倍。整体来看，叠加全部算子的加速效果普遍优于单一算子，两款模型在不同批次负载下的提速波动与峰值特性存在显著区别，多数场景下优化方案可带来正向吞吐增益。
:::

测试结果解读：

- Qwen3-1.7B：在 H100 上启用全部 Helion 内核后，端到端吞吐量提升约 **1.05 倍**；
- Qwen3-8B：批大小为 32 时性能提升最为显著，该现象与内核级测试结论吻合——`num_tokens=32` 是 Helion `scaled_mm` 内核的性能最优区间。

> 团队还针对**推测解码** 场景开展测试，该场景下解码阶段的有效 Token 数量恰好落在上述性能最优区间。

推测解码测试使用模型：

- RedHatAI/Qwen3-8B-speculator.eagle3
- RedHatAI/Qwen3-32B-speculator.eagle3

实测结果：启用全部 Helion 内核后，端到端吞吐量最高提升约 **1.09 倍**。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/11.png)

::: image-caption表 3：H100 平台、逐 Token 激活量化+推测解码场景端到端测试结果（括号内为推测 Token 接受率）TTFT：Time To First Token，首 Token 生成延迟；TPOT：Time Per Output Token，单输出 Token 生成延迟。该表格对比 Helion 方案与默认基线，在 Qwen3-8B、Qwen3-32B 模型、不同批次及投机预测 token 数下的推理指标，涵盖 TTFT、TPOT、总吞吐与对应加速倍数。Helion 可显著降低首包与单 token 生成时延，TTFT 加速最高达 1.33 倍，TPOT 稳定提升 1.05–1.08 倍；整体吞吐实现 1.06–1.09 倍提速。增加投机 token 数量后 TTFT 优化效果更突出，16 批次下 8B 模型吞吐绝对值更高，8 批次 32B 大模型也保持正向增益，证明 Helion 投机推理能同步优化时延与吞吐。
:::

#### B200 平台表现

> 在 NVIDIA B200 平台的端到端测试中，仅启用 **fp8_quant** 类内核：其余两类内核要么受 Blackwell 架构下 Triton GEMM 性能限制、表现弱于基线，要么在不同负载下性能收益不稳定。

即便仅启用量化相关内核，所有 Qwen3 模型的吞吐量仍实现明显提升。

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/12.png)

::: image-caption图 2：B200 开启逐 Token 激活量化后的整体吞吐量加速比（基线为 vLLM 默认配置）。该折线图展示 B200 显卡上单 Token FP8 量化算子对 Qwen3-1.7B、8B、32B 三款模型的吞吐加速效果，横轴为 1 至 64 的批次大小，纵轴代表总吞吐加速倍数。三款模型提速均为正向，变化趋势相近，均在 Batch Size=8 时迎来加速峰值，其中 1.7B 模型峰值最高，接近 1.07 倍；批量超过 8 后加速持续回落，到 64 时 1.7B 模型降幅最明显，8B、32B 曲线相对平缓，可见单 Token 量化能稳定提升推理吞吐，但不同参数量模型对大批次负载的适配表现存在差异。
:::

### 4.2 动态分组激活量化

> 本次测试使用的模型权重：

- Qwen/Qwen3-1.7B-FP8
- Qwen/Qwen3-8B-FP8
- Qwen/Qwen3-32B-FP8

![文章插图](/articles/2026-06-14-helion-dsl-vllm-qwen3-fp8-23fda003/images/13.png)

::: image-caption图 3：H100、B200 开启分组激活量化后的整体吞吐量加速比（基线：线性层后端替换为 CUTLASS 的 vLLM 配置）。图中左右两张折线图分别展示H100、B200显卡上Qwen3系列模型分组量化算子的吞吐加速表现，横轴为1至64的Batch Size，纵轴代表总吞吐加速倍数。左侧H100测试1.7B、8B模型，1.7B加速波动幅度更大，小批量时峰值达1.1倍，8B随批量增大提速稳步上涨；右侧B200新增32B模型，三款模型加速稳定维持在1.04至1.08区间，曲线波动更小，大批量下32B模型加速持续走高，两款硬件中FP8分组量化算子均为模型带来正向吞吐增益，B200平台运行更平稳。
:::

在分组激活量化场景下，H100、B200 平台的分块 FP8 GEMM 默认后端为 DeepGEMM。:mdnice-bold-italic[但当前 Helion 分组量化内核暂不兼容 DeepGEMM 依赖的 UE8M0 量化格式，因此本次实验强制 vLLM 将线性层后端切换为 CUTLASS。]

::: block-1重要说明：该场景下的基线**并非** vLLM 默认配置。但所有实验组与对照组均统一使用 CUTLASS 作为线性层后端，因此性能差异仅来源于非 GEMM 内核（FP8 量化、量化融合内核），对比结果具备参考价值。
:::

测试结果：仅启用 Helion 轻量化内核，全负载场景下端到端吞吐量即可实现约 **1.05 倍** 提升。

## 四、相关资源

为方便复现实验与二次开发，本文涉及的所有 Helion 内核实现代码、实验所用 vLLM 分支均汇总至下方 GitHub 讨论区：issue: Performance: Custom Helion Kernels[10]

## 五、注意事项

### 5.1 自动调优耗时问题

> 本次实验的主要工作量集中在内核自动调优环节。以 `scaled_mm` 这类大型内核为例：针对三款模型、总计 168 种不同输入形状[11]执行**完整调优，耗时可达一整天——Helion 会为每一种形状生成并测试数千组候选内核实现。**

现有研究表明[12]：无需对每一种形状做穷尽式调优与调度，适当减少专用配置分组数量，可在性能损失极小的前提下，大幅平衡调优耗时与运行时性能。目前 Helion 团队正在探索多种优化方案：缩减搜索空间、基于大模型引导自动调优等，进一步降低调优成本。

### 5.2 运行时调度开销

Helion 运行时调度逻辑会为每次内核调用引入数十微秒的 CPU 开销。对于轻量小型内核，该开销会成为端到端延迟的主要来源。**因此，启用 CUDA graph 捕获与回放是 Helion 发挥最优性能的必要条件**。Helion 团队也在持续优化，降低非 CUDA graph 模式下的调度延迟。

## 总结

> 本次 Helion 与 vLLM 的联合实践，完整验证了这款新型内核 DSL 在大语言模型推理领域的核心优势。:mdnice-bold-italic[相比 CUDA、CUTLASS 等传统底层开发方案，Helion 依托 PyTorch 语法体系，大幅简化了融合内核的编写、调试与验证流程，多数算子可在单日内完成落地，显著提升了内核迭代效率。在性能表现上，其针对 FP8 量化、归一化、多头注意力附属算子的优化效果尤为突出，不仅单内核测速结果亮眼，在常规推理、推测解码等真实业务场景中，也实现了端到端吞吐与延迟的双重优化。]

现阶段 Helion 仍存在多项亟待解决的短板：

- B200 Blackwell 架构下矩阵乘性能偏弱
- 大规模内核自动调优耗时长达一天
- 运行时调度存在 CPU 开销，且暂时无法兼容 DeepGEMM 所需的 UE8M0 量化格式

::: block-1
**对此研发团队已规划清晰优化路径**：优化 Triton 代码生成、引入 CuteDSL 新后端适配新一代 GPU；通过搜索空间剪枝、大模型引导调优缩短自动调优周期；同时持续精简调度逻辑、补齐量化格式兼容性。
:::

在大模型推理技术快速迭代的当下，兼顾**开发效率、跨硬件可移植性与运行性能** 是行业的核心诉求。Helion 创新的设计思路为 GPU 推理内核开发提供了新范式，随着后续优化持续落地，它有望在更多线上生产场景落地，成为推动 LLM 推理技术演进的重要工具。

参考资料

[1] vLLM: *https://docs.vllm.ai/en/latest/*

[2] Helion: *https://helionlang.com/index.html*

[3] PyTorch 自定义算子: *https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html*

[4] triton_scaled_mm: *https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py#L141*

[5] w8a8_triton_block_scaled_mm: *https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L835*

[6] vLLM Helion 内核集成框架: *https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L835*

[7] LFBOTreeSearch: *https://helionlang.com/api/autotuner.html#helion.autotuner.surrogate_pattern_search.LFBOTreeSearch*

[8] Helion 集成框架: *https://github.com/vllm-project/vllm/issues/32219*

[9] 受性能问题影响: *https://github.com/vllm-project/vllm/issues/25094*

[10] issue: *https://github.com/vllm-project/vllm/issues/32962*

[11] 168 种不同输入形状: *https://github.com/vllm-project/vllm/commit/5bc478ccee9bae4056aeae9953861fe587265e3f#diff-be77e79f35962c7bc20c44638613a5fdca7bb745b987888b4c63dd7557dd4207*

[12] 现有研究表明: *https://github.com/vllm-project/vllm/commit/5bc478ccee9bae4056aeae9953861fe587265e3f#diff-be77e79f35962c7bc20c44638613a5fdca7bb745b987888b4c63dd7557dd4207*

::: related-title
**相关推荐**
:::

::: related-links
- **[vLLM 冷启动延迟的系统性拆解与深度优化指南](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902194&idx=1&sn=3d17f7b85eeef0daac4735440cdebb66&scene=21#wechat_redirect)**
- **[vLLM Triton 注意力后端深度解析：跨 NVIDIA、AMD 的 GPU 架构创新优化](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898509&idx=1&sn=341fceedfbc28e118212eb85ae5b1807&scene=21#wechat_redirect)**
- **[“历史性的项目，错过了遗憾终生”：清华特奖游凯超与 vLLM 的三次蜕变与技术远征](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896591&idx=1&sn=689f8a0351d8ea9ed0c2c0ebd2f89ca2&scene=21#wechat_redirect)**

:::
