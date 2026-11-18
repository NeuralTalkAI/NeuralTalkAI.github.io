关键词：**编译器世界模型**、张量程序优化、**代价模型**、:mdnice-bold-italic[动作条件状态转移]、**TVM AutoScheduler**

> AutoTVM 到 Ansor，从 TenSet 到 TLP，:mdnice-bold-italic[评估一个候选调度（Schedule）的方式始终是把最终的 TensorIR 代码拍成一张"静态照片"喂给模型，至于这段代码是经过了哪些循环切分、融合、绑定才长成这样，评估器一概不知。]

![文章插图](/articles/2026-07-26-toward-compiler-world-models-tvm-c458f9fa/images/01.webp)

- **Toward Compiler World Models: Learning Latent ynamics for Efficient Tensor Program Search**
- https://arxiv.org/pdf/2606.09312

::: related-title
**相关推荐**
:::

::: related-links
- **[从"切多大、留什么、按什么顺序跑"，重新定义编译器后端计算图调度：MLSys 2026 系统赛 A 赛道冠军方案浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901589&idx=1&sn=8f35478cdabeab5cffeba7d704a30009&scene=21#wechat_redirect)**
- **[平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)**
- **[调度语言的过去、现在和未来：面向 CPU/GPU 集群、性能接近厂商库的统一调度技术创新](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891672&idx=1&sn=f4798f7db11e0bb73d24ee3826c6fb32&scene=21#wechat_redirect)**
- **[比 Ansor 平均快 2.7 倍！调优快70倍！基于分块表达式与搜索空间剪枝的 MBCI 算子链快速融合框架 MCFuser！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889535&idx=1&sn=4f85a584809f1533ef57644a39212741&scene=21#wechat_redirect)**

:::

> 中国科学院软件研究所与国科大的联合团队在《Toward Compiler World Models: Learning Latent ynamics for Efficient Tensor Program Search》这篇文章中:mdnice-bold-italic[提出了一个世界模型视角的解法：把编译器和硬件看成环境，把程序看成状态，把调度原语看成动作，在连续潜空间里逐步推演状态转移，用预测出的终态表示来给候选打分。]

![文章插图](/articles/2026-07-26-toward-compiler-world-models-tvm-c458f9fa/images/02.webp)

::: image-caption
图 3：本文框架总览。给定初始 TensorIR 状态与候选调度动作序列，框架通过该序列诱导的终态表示对候选进行评估。框架从初始程序状态出发，在表示空间中建模动作条件约束的状态演化，将预测得到的终态表示与动作、硬件特征结合，为候选调度打分，最终得分用于指导 TVM AutoScheduler 在线搜索循环中的候选选择。该框架拆解为三个解耦模块：TensorIR 状态编码器、多步状态转移模型、排序代价模型，三者分别承担表示学习、动力学建模、候选排序的职能。这种分解将原本端到端的预测任务拆分为三个可独立训练的子任务，降低了训练难度：编码器用对比学习预训练获得通用程序表示，转移模型用轨迹数据学习状态演化规律，排序模型用实测数据学习性能排序。模块解耦也带来了复用性，比如预训练编码器可迁移到其他编译器分析任务，转移模型也可适配不同硬件平台，仅需微调排序模型即可快速落地。
:::

实验结果相当硬：

- 同样 64 次硬件实测预算下，代表性子图延迟**相对 Ansor 在 RTX 4090 上几何平均提速 1.37 倍**、在**至强 6430 上提速 1.54 倍**；用 1024 次实测就在算术平均上:mdnice-bold-italic[小幅反超 Ansor-10240，几何平均差距收敛到 2.2% 以内，等于把样本效率抬高了约十倍；]
- 端到端推理**相对 PyTorch 与开启 cuDNN 的 PyTorch-opt 分别取得 4.61 倍与 3.67 倍几何平均加速**，:mdnice-bold-italic[峰值达 58.61 倍与 35.50 倍]。

![文章插图](/articles/2026-07-26-toward-compiler-world-models-tvm-c458f9fa/images/03.webp)

::: image-caption
图 8：GPU 代表性子图上，本文方法相对大预算 Ansor 的样本效率对比。(a) 为不同量级 Ansor 基线的几何均值比值；(b) 为本文 1024 次试验相对 Ansor-10240 的子图级比值，按加速比排序。整组实验回答了 “少样本下能达到多大预算的调优质量” 这一核心问题，结论是 10 倍样本效率提升：1024 次试验就能接近 Ansor 一万次试验的几何平均性能，且半数子图实现反超。这一结果极具工程价值，张量程序调优的主要成本来自硬件实测，样本效率提升直接转化为调优时间与算力成本的下降。同时也证明了隐状态动力学建模的有效性 —— 通过学习调度过程的状态演化规律，模型能更精准地预判候选性能，避免了静态评估的语法噪声干扰，从而在极少实测的情况下锁定优质调度方案。
:::

::: block-1
对于长期被"调优几小时、省毫秒"困扰的 AI Infra 工程师来说，这项工作指向了一条:mdnice-highlight[把编译器搜索成本打下来的现实路径。]
:::

## 本文目录

- 一、静态快照的原罪：代价模型为什么总是"看不懂"调度
- 二、三段式框架：编码、转移、排序各管一件事
- 三、状态编码器：给 TensorIR 造一个可比较的潜空间
- 四、动作条件状态转移：TransH 客串编译器动力学
- 五、排序即评估：LambdaRank 与在线搜索的闭环
- 六、实验解剖：十倍样本效率究竟从哪来
- 七、消融与边界：什么在起作用，什么还没解决

![文章插图](/articles/2026-07-26-toward-compiler-world-models-tvm-c458f9fa/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[从"切多大、留什么、按什么顺序跑"，重新定义编译器后端计算图调度：MLSys 2026 系统赛 A 赛道冠军方案浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901589&idx=1&sn=8f35478cdabeab5cffeba7d704a30009&scene=21#wechat_redirect)**
- **[平均性能超Triton/TVM 1.35倍！Neptune：整合调度-tile优化流水线的GPU张量编译器，突破注意力性能瓶颈！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893273&idx=1&sn=bda860ca237cd5fef6909c173178b271&scene=21#wechat_redirect)**
- **[调度语言的过去、现在和未来：面向 CPU/GPU 集群、性能接近厂商库的统一调度技术创新](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891672&idx=1&sn=f4798f7db11e0bb73d24ee3826c6fb32&scene=21#wechat_redirect)**
- **[比 Ansor 平均快 2.7 倍！调优快70倍！基于分块表达式与搜索空间剪枝的 MBCI 算子链快速融合框架 MCFuser！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889535&idx=1&sn=4f85a584809f1533ef57644a39212741&scene=21#wechat_redirect)**

:::
