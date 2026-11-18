关键词：大语言模型推理、**MegaKernel**、:mdnice-bold-italic[GPU 内核合成]、**编码智能体**、内存带宽利用率

> 自回归模型的解码过程受内存带宽约束，每生成一个 token，所有权重和 KV 缓存字节都要穿过一次高带宽内存。

**解码 MegaKernel 是解决这一瓶颈的理想方案，它将整个解码步骤融合到单个内核启动中，消除大量启动开销与中间数据搬运**。:mdnice-bold-italic[但现有自动生成方法始终无法同时实现跨模型通用性与计算正确性保证。]

![文章插图](/articles/2026-09-16-sglang-1-21-forgemegakernel-ai-ed339666/images/01.webp)

- **ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels**
- 组织机构：清华大学、中国科学院大学、ModelBest Inc.
- 论文链接：https://arxiv.org/abs/2609.12379v1

::: related-title
**相关推荐**
:::

::: related-links
- **[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903011&idx=1&sn=7872d080a5475a44bba4cfabe7717b56&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)**
- **[超越 vLLM 与 SGLang！Event Tensor：以动态 MegaKernel 消除重编译，解锁GPU核间通信-计算重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900086&idx=1&sn=fb8be688929f686853cc12c815e759ae&scene=21#wechat_redirect)**

:::

> **ForgeMegakernel 利用编码智能体为每个模型生成定制化高性能解码 MegaKernel，核心由两部分构成**：包含 :mdnice-bold-italic[10 个渐进里程碑的通用知识库]，定义高性能巨内核的结构属性；:mdnice-bold-italic[独立中间态测试预言机]，推导中间状态并在生成过程中校验性能、误差与精度。

*中间态测试预言机就是一个完全独立的 “内核质检台 + 标准答案库”，专门用来校验编码智能体写出来的巨内核靠不靠谱。*

![文章插图](/articles/2026-09-16-sglang-1-21-forgemegakernel-ai-ed339666/images/02.webp)

::: image-caption
图 2：ForgeMegakernel 框架总览架构图，清晰展示了 ForgeMegakernel 的三大核心组件与工作流。C1 规定内核需达成的结构特性，每个里程碑对应一项结构要求；C2 推导所有校验判决所需的量化指标；C3 每轮迭代采用全新上下文的 Agent 进行代码编辑，最终完成内核部署。具体来说，C1 的里程碑知识库是优化的 “导航图”，从功能正确到性能优化再到稳定性，分层递进地定义了高性能内核的结构属性，且不包含模型特定参数，保证了跨模型的泛化性；C2 的中间态测试预言机是 “独立裁判”，所有校验指标都从模型与硬件配置预先推导，独立于生成 Agent，从根源上避免了自校验的漏洞；C3 的迭代生成框架是 “执行引擎”，每轮用全新上下文的 Agent 编辑内核，基于 GPU 实测结果反馈优化，同时引入审核机制防止违规修改。三者形成闭环，共同实现了从模型配置到可部署高性能内核的端到端自动生成。
:::

在 8 个模型家族、0.6B-13B 参数规模的 14 个代表性解码任务上，ForgeMegakernel 生成的巨内核**内存带宽利用率达到 50.5%-85.9%**，** 相对于 SGLang 0.5.18 几何平均提速 1.21 倍，相对于同配置巨内核编译器提速 1.54 倍。** 详见下图😅。

![文章插图](/articles/2026-09-16-sglang-1-21-forgemegakernel-ai-ed339666/images/03.webp)

::: image-caption
图 3：该图全面呈现了 ForgeMegakernel 在 14 个测试配置上的性能与精度表现，纵轴向上为单步骤模型带宽利用率 MBU，向下为相对 float64 前向传播的对数概率误差。性能方面，生成内核的 MBU 达到 50.5%-85.9%，全面超越 SGLang（36.5%-78.9%）与 MPK（20.6%-75.9%），几何平均相对 SGLang 加速 1.21×，相对 MPK 加速 1.54×；即使对比专家手工编写的 Llama-1B 巨型内核，生成内核也达到了 74.0% MBU，与手工优化的 74.6% 十分接近，证明自动生成的效果可以逼近顶尖手工优化水平。精度方面，所有配置下生成内核的对数概率误差都低于 SGLang，误差比值在 0.38-0.72 之间，中位数 0.66，说明 ForgeMegakernel 在提升性能的同时，计算精度反而优于原生推理引擎，这得益于测试预言机对中间计算精度的严格管控。
:::

集成到 SGLang 后，**在 GSM8K 测试集上所有巨内核解码速度均超过原生引擎，同时保持相当的回答精度**，详见下表😅。

![文章插图](/articles/2026-09-16-sglang-1-21-forgemegakernel-ai-ed339666/images/04.webp)

::: image-caption
表 2：SGLang 框架内的 GSM8K 测试结果：回答准确率与单 token 解码延迟。该表展示了 13 组生成的巨型内核在实际推理服务中的端到端表现，覆盖多个主流模型系列，batch 从 1 到 16，上下文长度从 128 到 4096。从延迟数据看，所有配置下 ForgeMegakernel 的延迟均低于 SGLang 基线，加速比在 1.03× 到 1.28× 之间，几何平均 1.11×，验证了巨型内核在真实服务链路中的性能收益。准确率方面，巨型内核与基线的差值均在 1.5 个百分点以内，且半数配置下准确率持平或更高，说明中间态测试预言机的正确性校验有效保障了计算精度，没有因为算子融合和执行结构优化引入精度损失，实现了性能与精度的平衡，具备实际部署价值。
:::

## 本文目录

- 一、为什么解码巨内核是 LLM 推理的必争之地
- 二、现有自动生成方案的两大死穴：通用性不足与正确性失控
- 三、核心设计一：10 级里程碑构成的通用知识库
- 四、核心设计二：独立中间态测试预言机
- 五、端到端生成框架：智能体迭代+硬件反馈的闭环
- 六、实验评测：性能、正确性与成本的全面验证
- 总结与展望

![文章插图](/articles/2026-09-16-sglang-1-21-forgemegakernel-ai-ed339666/images/05.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903011&idx=1&sn=7872d080a5475a44bba4cfabe7717b56&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)**
- **[超越 vLLM 与 SGLang！Event Tensor：以动态 MegaKernel 消除重编译，解锁GPU核间通信-计算重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900086&idx=1&sn=fb8be688929f686853cc12c815e759ae&scene=21#wechat_redirect)**
- **[拆解 megakernel 性能神话：普通 CUDA 内核靠 PDL 调度实现 97.4%持久化吞吐](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905121&idx=1&sn=f93dceb62089296ffcabc24c964e94c3&scene=21#wechat_redirect)**
- **[10.3 倍加速！康奈尔大学 Perseus：终结多节点 MoE MegaKernel 的隐藏序列化噩梦，让代理传输反超 GPU-direct 1.2 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901207&idx=1&sn=4e68da37ab9ecd3a8d20e16451ea6ebe&scene=21#wechat_redirect)**
- **[Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel，支撑万卡模型训练！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904378&idx=1&sn=83e25209c3e1ffc683b97a6f8646f250&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**
- **[适配 Ada 中端卡的 MegaKernel 在线广告方案 Ada-MK：共享内存直接减半，端到延迟最高降 50%，击穿小批量毫秒级推理瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902893&idx=1&sn=5f53dd32ac726cea553f1c3743d8d2a1&scene=21#wechat_redirect)**
- **[实测提速 16%：硬件光追下 MegaKernel 与波前路径追踪的 GPU 性能对照](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902818&idx=1&sn=fab8996b09f6a940673d007317e08657&scene=21#wechat_redirect)**

:::
