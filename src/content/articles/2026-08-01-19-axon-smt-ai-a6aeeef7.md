关键词：**合成式超优化**、程序合成、**SMT 等价性验证**、:mdnice-bold-italic[AI 加速器内核]、算子融合

> "为 AI 加速器编写高性能内核，需要在**切分、指令选择、数据布局和算子融合** 上具备深厚的专业知识——这给程序员带来了沉重的负担。"

UIUC 与 Amazon Web Services 的联合团队在论文《Axon: A Synthesizing Superoptimizer for Tensor Programs》开篇就提出了这一 AI 基础设施行业的普遍痛点。:mdnice-bold-italic[针对现有张量优化方法普遍依赖手工重写规则、正确性保障不足的问题，该团队提出 Axon——一款面向分块式 AI 加速器的综合型超级优化器。]

![文章插图](/articles/2026-08-01-19-axon-smt-ai-a6aeeef7/images/01.webp)

- Axon: A Synthesizing Superoptimizer for Tensor Programs
- https://arxiv.org/pdf/2606.26344

::: related-title
**相关推荐**
:::

::: related-links
- [NVIDIA 编译器超参优化器 CompileIQ：用遗传搜索把 PTXAS / NVCC 调成"玄学终结者"](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901797&idx=1&sn=7e1e00004c316be529b26fb11248b137&scene=21#wechat_redirect)
- [ASPLOS'26 高频语义等价指令挖掘，实现2.6倍加速、93%面积节省！用E-Graph反合一为AI芯片打造乐高式指令！寻找高频数据级并行指令模式](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899424&idx=1&sn=e2a2f9e94eeeefe1858087e094837b18&scene=21#wechat_redirect)
- [端到端 LLM 编译器 nncase：基于 e-graph 的异构存储架构高性能统一编译框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895279&idx=1&sn=cfe132ed7791edfd649561561681c72d&scene=21#wechat_redirect)

:::

> Axon 以类 NumPy 接口的高层张量程序为输入，:mdnice-bold-italic[基于程序综合技术自动生成目标硬件指令，通过穷尽搜索语义等价的程序变体，经硬件实测筛选最优内核。]

其核心技术包括：

- 基于算子传播的代数变换机制，通过 SMT 求解器在无界张量上验证语义等价性，无需人工编写重写规则；
- νGraph 统一中间表示，可同时维护代数重排、分块配置、指令选择、融合策略等所有合法变体，避免贪心优化错失全局最优；
- 覆盖符号分块、ISA 生成、双层融合的全栈综合管线。

在亚马逊 Trainium 平台的 20 项基准测试中，Axon 相较官方 Neuron 编译器实现:mdnice-bold-italic[单算子最高 3.7 倍、多算子内核最高 19 倍的性能提升]，优于手工优化 NKI 内核:mdnice-bold-italic[最高达 1.35 倍]；**相较前沿搜索式编译器 Mirage 几何平均提速 2%~10%，且额外支持 16 个 Mirage 无法生成的基准内核**，为 AI 加速器自动化高性能内核生成提供了可证明正确的新路径。

![文章插图](/articles/2026-08-01-19-axon-smt-ai-a6aeeef7/images/02.webp)

::: image-caption
图 1：Axon 工作流程总览。这张图是理解全文的地图：输入的 NumPy 式张量程序先被解析成张量计算图，经过图变换器产生多个语义等价的 νGraph 变体；随后切分通行证为每个算子标注符号化切分参数，ISA 合成器把算子下降为目标指令序列，ISA 融合变换器再做指令级重排与融合；最终代码发射器吐出少则数百、多则近三万个 NKI 候选程序，交给 Neuron 编译器编译并在 Trainium 上实测，最快者胜出。注意图中两个灰色输入框——张量算子语义与 ISA 语义加硬件约束，这是 Axon 全部"智能"的知识来源，除此之外没有任何手写优化规则。
:::

## 本文目录

- 一、AI 加速器的性能上限，需要专家知识
- 二、一个例子看懂 Axon：RMSNorm+MatMul
- 三、没有重写规则的代数变换：算子传播加上 SMT 证明
- 四、νGraph：把所有等价变体同时留住
- 五、符号化切分与草图合成：从张量算子到机器指令
- 六、两级融合与代码生成：把中间结果留在片上
- 七、实验：19 倍加速从哪来，代价是什么
- 结论与未来工作

![文章插图](/articles/2026-08-01-19-axon-smt-ai-a6aeeef7/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- [平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)
- [性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)
- [性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)

:::
