关键词：**Intra-kernel Tracing**、:mdnice-bold-italic[二进制插桩]、指令拼接、**性能剖析**、**高保真**、异构计算、Agent 驱动优化

今年七月份的时候就写过：

NUS 团队旗下 Mars Compute 重磅出品 G-Watch：重构 GPU 优化路径，赋能 Agent 多 DSL 内核级自动调优，Flash Attention 全系列验证增益

，恰巧前几天看到文章挂出来了。

> 论文中有一句核心判断直击行业痛点：细粒度的性能可见性，只有在不破坏被观测执行本身的前提下才有价值。

现代大语言模型的性能瓶颈越来越集中在 GPU 内核层面，为了提升硬件利用率，融合内核成为主流设计，将计算、访存、通信整合进单次内核启动，通过复杂的软件流水线掩盖延迟。

![文章插图](/articles/2026-11-02-gpu-neutrino-iket-xtrace-19-e7c776ce/images/01.webp)

::: image-caption
图 1：Neutrino 和 IKET 在三个代表性 kernel 上的两类追踪保真度损失来源。现有 SOTA trace 工具同时存在两类保真度损失 ——compile-time interference（图 a，二进制变了）和 runtime perturbation（图 b，变慢了）。横轴三个 kernel 覆盖了 NVIDIA 两代旗舰（H100 Hopper、B300 Blackwell）和两种 attention 实现（FA-3 warp-specialized、FlashMLA fused MoE 风格），采样点选得很有代表性。作者把问题拆成正交的两个维度，隐含的设计结论是：要同时解决这两个问题，必须在 binary level 动手 ——IR level 只能解决 runtime overhead 的一半但必然破坏 compile；non-inlined call 注入（NVBit 路线）虽然不动 compile，但 context save/restore 成本过高不实用。这张图实际上定义了 Xtrace 的设计空间边界：binary-level instruction splicing 是唯一能同时把两条曲线都压到右下角的路径。
:::

::: block-1
传统硬件计数器只能提供聚合统计数据，无法暴露纳秒级的指令重叠、流水线气泡与空闲周期，内核内追踪因此成为精细化性能优化的核心手段。:mdnice-highlight[但现有追踪工具普遍在编译前插入探针，既会干扰编译器的优化决策，导致追踪的二进制与实际运行版本差异巨大，又会与原内核竞争寄存器资源，引入显著的运行时开销，最终得到的是被扭曲的执行画像。]

![文章插图](/articles/2026-11-02-gpu-neutrino-iket-xtrace-19-e7c776ce/images/02.webp)

::: image-caption
表 1：按追踪方法与各厂商插桩层级划分的现有GPU内核内追踪工具对比。表格从源码级、IR级、预留式、非内联二进制级四个类别，列出了不同工具在NVIDIA和AMD平台的支持情况。可以看到，现有二进制级工具大多只支持NVIDIA，且多为非内联实现，开销极高；而Xtrace是首个同时支持NVIDIA和AMD多架构的二进制内联拼接方案。
:::
:::

![文章插图](/articles/2026-11-02-gpu-neutrino-iket-xtrace-19-e7c776ce/images/03.webp)

- **Xtrace: High-Fidelity GPU Intra-Kernel Tracing via Binary-Level Instruction Splicing**
- 组织机构：新加坡国立大学、Meta
- 论文链接：https://arxiv.org/abs/2609.28769
- 文档：https://g-watch.github.io

::: related-title
**相关推荐**
:::

::: related-links
- **[NUS 团队旗下 Mars Compute 重磅出品 G-Watch：重构 GPU 优化路径，赋能 Agent 多 DSL 内核级自动调优，Flash Attention 全系列验证增益](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903272&idx=1&sn=de48552599c4440975bafdd4d4904a46&scene=21#wechat_redirect)**

:::

::: related-links
- **[从GEMM、FA再到分布式vLLM：TileSight 用三层 tile 抽象击穿 GPU 性能建模黑盒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904280&idx=1&sn=91a08cdc1338df361894a581f5ee9fb8&scene=21#wechat_redirect)**
- **[CudaProfiler：TIRx 轻量化内核级时序剖析工具，精准定位 warp 流水线阻塞与尾延迟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903219&idx=1&sn=f122de6cd22a28e4ede68643b23b5321&scene=21#wechat_redirect)**

:::

> Xtrace 的核心洞察在于，**将探针插入时机后移到编译完成的二进制阶段，通过指令拼接技术直接将探针指令嵌入原有指令流。** 它通过寄存器活性分析复用插入点的死寄存器，结合编译器同款冒险表解决指令冲突，再通过最优调度压低探针带来的依赖等待，最终实现近乎零编译干扰和极低运行开销。

![文章插图](/articles/2026-11-02-gpu-neutrino-iket-xtrace-19-e7c776ce/images/04.webp)

::: image-caption
图 2：现有 IR 级 [35] 和 binary 级 [56,65,68] intra-kernel trace 工具与 Xtrace 的插桩机制对比。FE/BE 为前端 / 后端 GPU kernel 编译器。这张图是论文的技术路线图。四列分别展示四种插桩时机：IR 内插、编译期占位、函数调用跳转、binary 指令拼接。每列上方 "Compile w/o interference / Compile w/interference" 的标注是关键判断 —— 只有 (c)(d) 两列真正做到 compile 零干扰，但 (c) 被 context switch 成本杀死，只剩 (d) 可行。图里用图例区分 original kernel instructions、placeholder instructions、probe instructions，直观说明 Xtrace 的 probe 是 "缝合" 进原有指令流而非 "附加" 在外部。对体系结构研究者，这张图的深层含义是：GPU profiling 工具的演进方向正在从 "编译时插桩" 转向 "二进制级 re-link"，类似 CPU 上 Pin→DynamoRIO→JIT binary rewriting 的演化路径，但 GPU SIMT 和 register file 物理化让这条路上的 hazard problem 比 CPU 难得多。
:::

实验数据显示，在 H100、B300、MI300X 三款旗舰 GPU 上，Xtrace 保留了 94-98%的原内核指令，现有工具仅能保留 8-48%；运行时开销仅 0.9-2.8%，现有工具高达 3.8-75.6%。

![文章插图](/articles/2026-11-02-gpu-neutrino-iket-xtrace-19-e7c776ce/images/05.webp)

::: image-caption
图 10：文中 Case 2，Xtrace 对 FA-4 和闭源 cuDNN SDPA 的 trace 引导 agent 优化 FA-4。 整图展示了 Xtrace 最具商业价值的能力：binary-level tracing 让开源 kernel 能够 "学习" 闭源 vendor kernel 的调度策略。四个子图形成完整论证链：(a) cuDNN 怎么做→(b) FA-4 差在哪→(c) 优化后对齐→(d) 量化收益。这个 case 说明未来 kernel 优化范式会变成 "trace-guided co-design"：用最高保真度的 trace 把 closed-source SOTA kernel 的行为拆开看，再把这些 pattern 移植到开源实现上。Xtrace 在这里实际上扮演了 "kernel reverse-engineering + performance coaching" 的双重角色。
:::

在指导编码 Agent 优化 FlashAttention-3 的任务中，Xtrace 将达到目标性能的迭代次数减少 3.9 倍；**更关键的是，它可以直接追踪闭源的 cuDNN 内核**，以此为参考将开源 FlashAttention-4 的吞吐量提升 5.2-13.3%。

## 本文目录

- 一、GPU 内核优化的痛点：追踪保真度的双重困境
- 二、现有方案的局限：从源码到二进制的插桩路线对比
- 三、Xtrace 核心设计：指令拼接技术的三大挑战与解法
- 四、拼接算法详解：如何在二进制空间里做最优插桩
- 五、跨架构实现：NVIDIA 与 AMD 的差异化适配
- 六、实验评估：保真度与开销的全面碾压
- 七、实战价值：从内核优化到闭源逆向的能力拓展
- 结语

![文章插图](/articles/2026-11-02-gpu-neutrino-iket-xtrace-19-e7c776ce/images/06.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[把 Nsight 装进 Agent 的工具箱，VeloQ项目让 LLM 自己读懂 GPU 性能剖析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902049&idx=1&sn=982c280219b6fc0407c196a885cc6435&scene=21#wechat_redirect)**
- **[暴涨 36 个 Star！开源 AI Agent 用 Nsight Compute 报告做 GPU 调优决策，实现"正确性-剖析-策略记忆"闭环驱动](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899947&idx=1&sn=0f63b74acab0c238e6c9494d54184804&scene=21#wechat_redirect)**
- **[代码开源！OSDI'25 通过类 eBPF 探测实现可编程细粒度 GPU 内核分析工具 Neutrino：跨厂商GPU剖析工具！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892532&idx=1&sn=446017b09eaec1e92fb6d77ed8301908&scene=21#wechat_redirect)**
- **[eGPU：让eBPF为GPU可观测性与可编程性赋能！首个动态PTX注入式eBPF-GPU框架，较NVBit显著降低内存观测开销](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892100&idx=1&sn=e1995e2908172d914fb38f4d7577268f&scene=21#wechat_redirect)**

:::
