关键词：**MegaKernel**、内核融合、**流式推理**、:mdnice-bold-italic[GPU 性能建模]、首包延迟优化

> “将流式模型融合为MegaKernel，核心问题在于确定融合的边界，而非判断是否需要融合。”这是本研究最具穿透力的核心结论。

MegaKernel（Megakernel）通过将多个算子融合为单个持久内核，消除内核启动开销与边界屏障，一度成为低延迟推理的标准优化路径，:mdnice-bold-italic[业界普遍认为融合范围越大性能越高。但流式推理模型通常包含多个运行速率、算力特征截然不同的阶段，单一MegaKernel无法兼顾所有阶段的最优配置，寄存器压力上升、线程块无法专业化等问题反而会抵消融合收益，甚至导致性能倒退。]

![文章插图](/articles/2026-11-01-megakernel-nvidia-3a7bdbd3/images/01.webp)

- **One Megakernel is Not Enough: Predicting Where to Cut for Streaming Inference**
- https://arxiv.org/pdf/2610.35336
- 组织机构：NVIDIA（班加罗尔、海得拉巴、浦那研发团队）

::: related-title
**相关推荐**
:::

::: related-links
- **[北大开源 PipeLink 把 MegaKernel 拆成可拼接组件：解码最快提速 2.16 倍，MoE 训练性能追平原生整块巨核！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447907574&idx=1&sn=0bdc690b1c8aefa149e48b7eb413ba29&scene=21#wechat_redirect)**
- **[PDL vs Megakernel：两类Kernel延迟优化方案的本质差异](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903286&idx=1&sn=a9cff3297f146d77532935b6056a9290&scene=21#wechat_redirect)**
- **[打平 CAKE、领先 KDA！ForgeKernel打破 Agent 优化天花板，覆盖 H100、B200 与昇腾 910C](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447907458&idx=1&sn=7bd293d0ff7dc3142eb8850bc14ad27f&scene=21#wechat_redirect)**
- **[拆解 megakernel 性能神话：普通 CUDA 内核靠 PDL 调度实现 97.4%持久化吞吐](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905121&idx=1&sn=f93dceb62089296ffcabc24c964e94c3&scene=21#wechat_redirect)**

:::

> 本研究提出一套边界规划框架，仅依靠常规图执行的性能剖面与独立算子基准测试数据，无需对候选设计进行端到端实测，即可预测模型中最优的 MegaKernel 切割位置与组合方式。

在 NH-TTS、Qwen3-TTS 与 Llama-1B 三个模型的 13 种批量大小场景中，该框架选中的设计均为实测最优方案；在 H100 NVL GPU 上，NH-TTS 模型采用 3 个专用 MegaKernel 的设计，**首音频延迟相比常规 TensorRT 流水线降低 4.05 倍，相比单 MegaKernel 设计降低 21.4%，同时实现延迟与吞吐的双重提升。**

![文章插图](/articles/2026-11-01-megakernel-nvidia-3a7bdbd3/images/02.webp)

::: image-caption
图 2：批次 32 下 NH-TTS 采用所选三 MegaKernel 设计的 TTFA 与 RTFx 指标。该图对比了常规执行、单 MegaKernel 、三 MegaKernel 三种方案的首音频延迟与实时因子 RTFx，同时呈现了吞吐量与延迟的双重收益。左侧延迟轴显示三 MegaKernel 方案 TTFA 仅 35.985ms，相比常规方案的 145.892ms 提速 4.05 倍，相比单 MegaKernel 的 45.776ms 提速 21.4%；右侧吞吐量轴显示三 MegaKernel 的 RTFx 达 269.430，同样优于单 MegaKernel 的 225.017 与常规方案的 210.533。这一结果打破了 “延迟与吞吐量不可兼得” 的固有认知，核心原因是动态 CTA 拆分与阶段专属优化同时提升了资源利用率与计算效率，证明合理的融合边界能够同时优化流式推理的首包延迟与整体吞吐。
:::

![文章插图](/articles/2026-11-01-megakernel-nvidia-3a7bdbd3/images/03.webp)

::: image-caption
图 4：已实现验证池内的预测选型结果。预测与实测裕度对比了同池内最优方案与次优方案的性能差距。该图汇总了三款模型共 13 个批次场景的选型结果，直观验证了边界规划器的准确率与决策裕度。NH-TTS 在全批次均选中多 MegaKernel 方案，次优方案与最优方案差距随批次增大从 14.9% 降至 4.2%；Qwen3-TTS 呈现明显的衰退特征，小批次选中 4 MegaKernel ，批次 48 以上转为常规执行最优，决策裕度在 4% 左右；Llama-1B 全批次选中 2 MegaKernel 方案。所有 13 个场景中规划器均准确命中实测最优方案，且次优方案的性能差距普遍在 3% 以上，远超测量误差范围，证明基于 tile 图与事件图的一阶建模能够可靠排序候选方案，无需端到端穷举实测即可锁定最优融合边界。
:::

该研究打破了“融合越多越好”的行业惯性认知，为流式推理场景的内核融合优化提供了可量化、可自动化的决策方法论。

## 本文目录

- 一、流式推理的性能困局：单核融合的天花板
- 二、Megakernel 组合的本质：切割点决定性能上限
- 三、边界规划器的核心技术栈：从区域图到事件图的三级建模
- 四、四大核心观察：重新理解 Megakernel 的性能规律
- 五、实验验证：13 个场景全命中最优的硬核结果
- 六、模型分类与行业启示：内核融合的未来走向
- 七、总结与展望

![文章插图](/articles/2026-11-01-megakernel-nvidia-3a7bdbd3/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[2122 Tokens/s 巅峰性能！从 MegaKernel 到 PDL：NaiveRT 将 RL Rollout 性能推向极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906316&idx=1&sn=a4035048c843aaace742978bd79b5704&scene=21#wechat_redirect)**
- **[较 SGLang 1.21 倍提速！ForgeMegakernel 开启 AI 推理 MegaKernel 的自动化锻造时代](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447905880&idx=1&sn=d33d843035f0dfc0a9af3f02f2811a94&scene=21#wechat_redirect)**
- **[Weave：面向计算通信重叠的 MoE MegaKernel 细粒度动态 SM 调度，较 DeepEP 1.95 倍层加速，1.12 倍端到端提速](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447906220&idx=1&sn=177db891e7fc7087f94090009ef44dc4&scene=21#wechat_redirect)**
- **[Cursor 开源了面向 GB300 NVL72 的生产级 MoE 训练 megakernel，支撑万卡模型训练！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904378&idx=1&sn=83e25209c3e1ffc683b97a6f8646f250&scene=21#wechat_redirect)**
- **[AutoMegaKernel 背后这家公司 RightNow AI 的产品矩阵、研究版图与技术哲学](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903030&idx=1&sn=d7de3a5385a1e87bcf5ecbc260431708&scene=21#wechat_redirect)**
- **[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903011&idx=1&sn=7872d080a5475a44bba4cfabe7717b56&scene=21#wechat_redirect)**
- **[实测提速 16%：硬件光追下 MegaKernel 与波前路径追踪的 GPU 性能对照](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902818&idx=1&sn=fab8996b09f6a940673d007317e08657&scene=21#wechat_redirect)**
- **[从 MegaKernel 视角重新定义 NPU 上的线性注意力极限！PTO-ISA 驱动的昇腾 NPU Gated DeltaNet MegaKernel 浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901380&idx=1&sn=4ae6987a6ba87626737c46cb4be9458e&scene=21#wechat_redirect)**
- **[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901263&idx=1&sn=91b18cdbc9fc8949c2277baf3de23826&scene=21#wechat_redirect)**
- **[10.3 倍加速！康奈尔大学 Perseus：终结多节点 MoE MegaKernel 的隐藏序列化噩梦，让代理传输反超 GPU-direct 1.2 倍！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901207&idx=1&sn=4e68da37ab9ecd3a8d20e16451ea6ebe&scene=21#wechat_redirect)**
- **[AMD 提出多芯粒 GPU 的 MegaKernel 方案 Fleet：通过 Chiplet 任务抽象将 L2 命中率从 12%提升至 54%，解码延迟降低 1.5 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900183&idx=1&sn=67b81e638e55f882448bf5b586e53167&scene=21#wechat_redirect)**
- **[超越 vLLM 与 SGLang！Event Tensor：以动态 MegaKernel 消除重编译，解锁GPU核间通信-计算重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900086&idx=1&sn=fb8be688929f686853cc12c815e759ae&scene=21#wechat_redirect)**
- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[性能超 Mirage、TVM、PyTorch！CMU 清华提出 Prism：符号化超优化终结张量程序枚举爆炸，在 LLM 核心负载上最高加速 2.2 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901362&idx=1&sn=6740959d78e8f65f2c0fdaf740fdde2f&scene=21#wechat_redirect)**

:::
