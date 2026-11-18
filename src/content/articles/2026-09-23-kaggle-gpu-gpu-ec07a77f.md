关键词：**图形性能指数**；逆向工程；:mdnice-bold-italic[结构化预测]；XGBoost；**特征工程**；Kaggle

> 本文完整复盘了 Kaggle 竞赛「GPU Performance Prediction Challenge」的参赛过程与夺冠方法。**该竞赛要求仅依据 GPU 硬件规格（流处理器数量、显存位宽、时钟频率等）预测主办方自定义的「图形性能指数」（Graphics Performance Index, GPI）**，以原始空间的均方根误差（RMSE）为评估指标。

**前七轮穷尽通用机器学习手段** 😅——LightGBM/XGBoost 选型、目标变换、特征工程、多 seed 集成与外部数据库增强——将公开榜分数从 0.20754 降至 0.18150，逼近通用方法的性能天花板，也参考了 TileSight。

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/01.webp)

- GPU Performance Prediction Challenge: Predict the Graphics Performance Index (GPI) from GPU hardware specifications including cores, clock speeds, memory, and bus width.
- https://www.kaggle.com/competitions/gpu-performance-prediction-challenge/leaderboard

::: related-title
**相关推荐**
:::

::: related-links
- **[从GEMM、FA再到分布式vLLM：TileSight 用三层 tile 抽象击穿 GPU 性能建模黑盒](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904280&idx=1&sn=91a08cdc1338df361894a581f5ee9fb8&scene=21#wechat_redirect)**
- **[解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**

:::

> :mdnice-bold-italic[随后通过观察训练集中同一 GPU 型号的「超频变体」]，**发现 GPI 是一个近乎确定性的函数，反向推导出隐藏公式** 🤣 $\mathrm{gpi} \approx C(\text{静态规格}) \cdot \mathrm{fp32}^{0.55}$，并据此构建「查表 + 回归」的两阶段结构化模型。这个过程中也发现竞赛数据来自我们先前分享的一篇文章的 GPU DataBase 开源项目，通过多字段规格指纹匹配（覆盖率 49.1%）。

[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)

公开榜分数最终降至 0.05543，位列第 1 名，较第二名（0.10781）领先约 48%；同时第二名本身亦为离群值，暗示前两名均掌握了隐藏公式😅。

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/02.webp)

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/03.webp)

::: image-caption
图 1：九轮迭代的公开榜分数轨迹，直观展示了这九轮的分数轨迹：前七轮在 0.18 附近反复震荡（CV 噪声淹没了真实增益），第八轮对公式的洞察一举将分数拉低近 70%，第九轮则因验证方式失误而小幅回退。图中不含 v2，因为该轮未提交。
:::

本文记录每一轮提交的方法考量与经验教训，其中「CV 提升不等于公开榜提升」「切分方式必须匹配测试集分布」两条认识最具普适价值，可为结构化回归竞赛提供方法论参考。

## 本文目录

- 一、引言
- 二、问题定义与数据
- 三、方法演进：九轮迭代的完整复盘
- 四、核心方法：两阶段结构化模型
- 五、实验与结果
- 六、讨论
- 七、结论
- 参考文献

![文章插图](/articles/2026-09-23-kaggle-gpu-gpu-ec07a77f/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[超越仿真，精准预测推理性能！上交与阿里提出SynPerf：一种混合分析与机器学习的 GPU 性能预测框架，实现 6.1%内核误差与 1.7 倍加速！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897849&idx=1&sn=0e780adf7b65d20826ef5e3b2330cda2&scene=21#wechat_redirect)**
- **[伯克利开源 CHIA：进化 Agent 打通 AI 芯片全链路，单框架实现仿真对齐/RTL 生成/时序优化五大工业级硬件自动化，最高 1.97 倍性能提升](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903506&idx=1&sn=27650a6cbe9467c5d3d0608e8839b15d&scene=21#wechat_redirect)**
- **[解锁下一代 AI 加速器性能潜力！普渡、Meta提出 AI 加速器复合操作数据流和显式集体通信建模方案 COMET](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896642&idx=1&sn=ed9b64a8a02e6db7a61a486e6fe64d65&scene=21#wechat_redirect)**
- **[Warp Specialization在 GEMM Kernel 中的性能建模与优化：仿真工具与 ≤21.5% 最大预测误差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893999&idx=1&sn=7aa3c3b7c1bf0a4327c0f248789de96f&scene=21#wechat_redirect)**
- **[ISCA'25 建模GPU LLM推理性能AMALI: 借助张量核心优化与warp分布，推理误差低于GCoM的GPU分析模型](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891884&idx=1&sn=ce8f9271c145efbebf087f4dd31ea97b&scene=21#wechat_redirect)**

:::
