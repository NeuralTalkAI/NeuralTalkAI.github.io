关键词：**端侧训练**、LLM 微调、:mdnice-bold-italic[移动 GPU 优化]、内存布局、**纹理内存**

> Transformer 模型在语言、视觉和多模态任务上展现出前所未有的能力，在设备端直接微调 Transformer 模型提供了一条保护隐私的个性化 AI 路径，但由于**训练过程中注意力机制里频繁的布局变换和严峻的内存约束，这条路径在移动 GPU 上依然效率低下**。

现有的:mdnice-bold-italic[移动端训练框架要么在前向和反向传播中复用同一种布局，导致反向传播时内存访问碎片化、GPU 利用率低下；要么依赖显式的布局转换，引入了巨大的转换开销。]

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/01.webp)

- **FBLayout: Optimizing Memory Layout for Efficient LLM Finetuning on Mobile GPUs**
- https://arxiv.org/pdf/2607.21624**相关推荐**

> FBLayout 的核心洞察在于：:mdnice-bold-italic[矩阵乘法在不同方向上做归约时，加载的其实是同一批 Tile 内的元素，差别仅仅在于聚合顺序。如果能让这些 Tile 元素在物理上对任意遍历方向都保持连续，那么单一布局就足以服务所有归约方向。]

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/02.webp)

::: image-caption
图 7：FBLayout系统总览。输入为联合前向-反向训练图，经过图分析、R-Tile布局设计、基于Tile的索引变换消除、激活引导的全局布局选择四个阶段，输出优化后的R-Tile布局训练图。
:::

在七款 Transformer 模型上，横跨 ARM Mali 和高通 Adreno 两类移动 GPU，**FBLayout 相比 MNN、TFLite 和 TVM 取得了 2.2 到 5.7 倍的加速**。在解码器-only 的大语言模型上，加速比更是达到 **3.9 到 5.7 倍**。

- 编译调优时间从 MNN 的 723 秒、TVM 的超过 4800 秒，**骤降至 6.1 秒。**
- **全局内存访问量降低 3.5 倍，缓存未命中减少 4.2 倍，能耗降低 3.5 到 6.3 倍。**

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/03.webp)

::: image-caption
图 13：FBLayout与其他框架在一加Ace5 Pro（骁龙8 Elite）上的训练速度对比。七个模型从左到右分别为Llama3.2-1B、Qwen2.5-1.5B、Gemma2-2B、Bert-L、VIT-L、Whisper-L、SD1.5。FBLayout在所有模型上均显著领先。
:::

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/04.webp)

::: image-caption
图 14：优化分解，包含训练速度和内存总访问次数。四个配置分别为MNN基线、仅R-Tile、R-Tile+Tile索引消除、完整FBLayout（含全局布局选择）。
:::

::: block-1
这些数字背后，是一套:mdnice-highlight[从算子分类、统一 Tile 布局、索引变换消除到全局布局传播的完整技术栈]，也是 MobiSys 2026 上首篇专门针对移动 GPU 大语言模型微调加速的系统工作。
:::

## 本文目录

- 一、问题的根源：前向-反向布局冲突
  - 1.1 推理与训练的性能鸿沟
  - 1.2 FB-Layout 冲突的本质
  - 1.3 移动 GPU 纹理内存的特殊性
  - 1.4 现有方案的两难
- 二、算子分类与问题形式化
  - 2.1 算子分类框架
  - 2.2 四个形式化问题
- 三、R-Tile：面向多维归约的统一布局
  - 3.1 Tile 不变性观察
  - 3.2 Tile 形状候选生成
  - 3.3 两级物理映射
- 四、基于 Tile 的索引变换消除
  - 4.1 从物理变换到逻辑重映射
  - 4.2 Tile 传播合法性检查
  - 4.3 索引变换消除
- 五、激活引导的全局布局选择
  - 5.1 算子对的布局决策
  - 5.2 连续布局冲突算子的优化
- 六、实验评估与深度分析
  - 6.1 实验设置
  - 6.2 端到端性能
  - 6.3 各优化组件贡献分解
  - 6.4 缓存效率分析
  - 6.5 R-Tile 微观分析
  - 6.6 Tile 索引消除效果
  - 6.7 编译时间与系统开销
  - 6.8 能耗与可移植性
- 结语

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/05.webp)

## 一、问题的根源：前向-反向布局冲突

> 推理时移动 GPU 相比 CPU 能提供约 6.5 倍的加速，**但在微调阶段，这个优势急剧萎缩甚至反转**。对于 0.5B 到 2B 参数的模型，GPU 上的微调延迟可能与 CPU 相当，甚至比 CPU 还差 43%。

### 1.1 推理与训练的性能鸿沟

理解 FBLayout 的价值，需要先看清端侧训练面临的独特困境。

- 推理阶段，数据流是单向的，每个张量通常只被消费一次，内存访问模式相对可预测。
- :mdnice-bold-italic[训练阶段完全不同：前向传播产生的中间激活值必须被保存下来，在反向传播时被重复使用以计算梯度。这导致训练的内存占用达到推理的 7 到 10 倍，即便使用 LoRA 等参数高效微调技术和激活重计算，内存压力依然巨大。]

::: block-1
更关键的是计算复杂度的跃升。在移动 CPU 上微调一个 1B 参数的模型，单步可能需要超过 50 秒。:mdnice-highlight[实际微调往往需要数千步，用户可能要等待数小时甚至数天才能让模型反映最新数据]。长时间训练还会触发热节流，导致吞吐量持续下降，甚至可能让训练无法完成。
:::

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/06.webp)

::: image-caption
图 1：端侧LLM推理与训练在移动平台（骁龙8 Elite）上的延迟对比。左图为推理，GPU相对CPU有显著优势；右图为微调，GPU优势消失甚至反超。这张图直观揭示了一个反直觉现象：移动GPU在训练场景下并不能自动延续推理时的加速优势。
:::

推理时 GPU 对 CPU 有碾压级优势，Llama 1B 上 GPU 仅需约 15ms 而 CPU 需要 100ms 以上。但到了微调阶段，Llama 1B 的 GPU 延迟达到 200ms 以上，Qwen 1.5B 更是超过 400ms，:mdnice-bold-italic[ViT 上 GPU 延迟接近 400ms 而 CPU 仅约 300ms。这个 43%的反超根源在于内存访问模式出了根本性问题，而非 GPU 算力不足。]

### 1.2 FB-Layout 冲突的本质

> **反向传播的数学原理决定了同一个张量在前向和反向中会以不同甚至冲突的模式被访问**。以矩阵乘法 $C_{m,n} = \sum_{k=1}^{K} A_{m,k} \cdot B_{k,n}$ 为例，前向传播沿 K 维度归约，而计算权重梯度 $\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$ 时沿 M 维度归约。:mdnice-bold-italic[链式法则天然引入转置操作，reshape 操作则在前向反向维度不匹配时隐式出现。]

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/07.webp)

::: image-caption
图 2：LoRA微调注意力模块中的前向-反向布局冲突示例。LoRA注入低秩适配器矩阵A和B来近似权重更新。反向传播引入了大量转置和reshape算子，与前向布局产生冲突。图中红色椭圆为前向激活，绿色星形为上游梯度，紫色方块为权重。
:::

> 上图 2 完整展示了注意力模块中 LoRA 微调的计算图。

- 前向传播中，$W_V$ 与输入 X 做 MatMul，$A_{LoRA}$ 与 X 做 MatMul 后经 $B_{LoRA}$ 再做一次 MatMul，结果相加后经 Softmax 产生注意力概率 P，再与 V 做 MatMul 得到输出。
- 反向传播时，为了计算 $dV/dP$、Softmax 梯度、$dA_{LoRA}/dB_{LoRA}$，需要对 $W_V$、$A_{LoRA}$、$B_{LoRA}$ 做转置，**还需要 reshape 操作来匹配多头注意力的维度要求。每一个转置都意味着内存布局的潜在冲突。**

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/08.webp)

::: image-caption
图 3：权重矩阵的FB布局冲突示例。(a)前向和反向MatMul沿不同维度归约；(b)共享同一种内存布局时，前向连续访问而反向跨步访问，导致缓存抖动。
:::

- 前向 MatMul 沿 $D_{out}$ 维度归约，行主序布局下访问是连续的，缓存命中率高。
- 反向 MatMul 沿 $D_{in}$ 维度归约，同样的行主序布局导致访问是跨步的，每次步长等于一行的长度。:mdnice-bold-italic[在移动 GPU 的 2.5D 纹理内存体系下，这种跨步访问的后果被急剧放大。]

### 1.3 移动 GPU 纹理内存的特殊性

> 移动 GPU 采用 2.5D 纹理内存系统，张量被映射到二维坐标空间`(width, height)`，每个坐标存储一个 4 元素 RGBA 向量。:mdnice-bold-italic[配套的纹理缓存专门为二维空间局部性优化，访问相邻坐标效率极高。这意味着在二维网格中连续的访问模式（比如分块遍历）能获得比跨步或分散访问高得多的带宽。]

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/09.webp)

::: image-caption
图 4：纹理内存空间局部性。(a)四种不同访问模式：列主序、行主序、Block-4分块、Block-8分块；(b)四种模式在不同线程数下的吞吐量。行主序和分块访问的吞吐量显著高于列主序。
:::

上图 4 中，行主序访问的吞吐量达到列主序的 1.6 倍，分块访问在高线程数下也保持优势。当线程数达到 64 时，列主序吞吐量仅约 1.0，而行主序和分块访问达到 1.2 到 1.4。**这个差距在大张量、长序列的 LLM 训练中会被放大到数倍**。:mdnice-bold-italic[前向-反向布局冲突直接破坏了这种局部性：为一个 pass 优化的布局通常违反另一个 pass 所需的二维连续访问，导致严重的缓存抖动、SIMD 单元利用率低下] （ALU 工作占比可低至 15%以下），以:mdnice-bold-italic[及频繁访问慢速全局内存造成的流水线停顿。]

### 1.4 现有方案的两难

现有移动框架采用两种策略应对这个冲突，两种都不够好。

> **第一种是布局复用，以 MNN 为代表**。:mdnice-bold-italic[前向传播的布局直接复用到反向传播，避免了转换开销，但迫使反向 kernel 使用极不高效的非连续内存访问]。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/10.webp)

::: image-caption
图 5：复用前向布局在反向传播中的性能开销（MNN）。(a)反向MatMul延迟：最优布局对比复用前向布局，序列长度从128到1024，复用前向布局的反向 MatMul 延迟达到最优布局的 5.31 倍；(b)各LLM训练模型的GPU ALU利用率，各模型反向传播的 ALU 工作占比普遍低于 20%，大量时间花在等待内存上。
:::

> **第二种是显式转换，以 TFLite 为代表**。:mdnice-bold-italic[插入显式的布局转换算子（transpose、reshape）为每个 kernel 提供最优布局。这保持了单 kernel 效率，但引入了难以承受的开销]。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/11.webp)

::: image-caption
图 6：显式布局转换在LLM训练中的开销分析。(a)运行时延迟分解，布局转换算子占用了高达 40.8%到 50.0%的运行时间；(b)内存利用率与布局转换分配比例，布局转换占用了 20%到 50%的内存分配，加剧了内存碎片化。
:::

::: block-1
:mdnice-highlight[总的来说，冲突的本质是：训练的双向数据流与移动 GPU 的 2.5D 纹理内存层次之间存在根本性不匹配]。**前向沿一个维度归约需要行方向连续，反向沿另一个维度归约需要列方向连续，而线性布局只能保证一个方向连续**。

现有方案要么牺牲一个 pass 的访问效率，要么在两个 pass 之间插入昂贵的物理转换，两者都将布局冲突视为不可避免的代价。
:::

## 二、算子分类与问题形式化

**优化训练图的内存布局，需要理解算子在前向和反向传播中与张量布局的交互方式**。:mdnice-bold-italic[与推理的单向数据流不同，训练引入了双向约束]，同一个张量在前向计算和梯度计算中以发散模式被访问。

### 2.1 算子分类框架

> :mdnice-bold-italic[FBLayout 提出了一个简洁但强大的算子分类框架，基于两个二元维度：前向张量依赖性（反向计算是否需要保留前向张量）和访问模式一致性（前向和反向是否表现出相同的内存访问模式）]。这两个维度定义了四类算子，每类有不同的优化需求。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/12.webp)

::: image-caption
表 1 按照这两个维度将算子分为四类。访问一致且无前向张量依赖的 FTI-C 类包括 Add、Subtract、Scale 等逐元素算子；访问一致但有前向张量依赖的 FTD-C 类包括 ReLU、Sigmoid、Tanh、Softmax、Dropout；访问发散但无前向张量依赖的 FTI-D 类包括 Transpose、Permute、Reshape 等变换算子；访问发散且有前向张量依赖的 FTD-D 类包括 MatMul、Conv2D、BatchNorm、LayerNorm，这是最关键的一类。
:::

- **第一类是前向张量无关且访问一致** （FTI-C），以逐元素加法为代表。前向 $C = A + B$，反向 $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}$，:mdnice-bold-italic[梯度直接来自上游，完全相同的逐元素访问模式不施加布局约束。]
- **第二类是前向张量依赖且访问一致** （FTD-C），以 ReLU 为代表。反向 $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot \mathbb{1}_{(X>0)}$ 需要前向激活值来判断掩码，但访问模式与前向完全一致，任何前向高效的布局对反向同样高效。
- **第三类是前向张量无关且访问发散** （FTI-D），以 transpose 和 reshape 为代表。前向置换 $(d_1, d_2, d_3) \to (d_2, d_3, d_1)$，反向需要逆置换 $(d_2, d_3, d_1) \to (d_1, d_2, d_3)$，创建发散访问，但不依赖前向张量值。
- **第四类是前向张量依赖且访问发散** （FTD-D），这是最关键的一类，以矩阵乘法为代表。反向 $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$ 需要保留前向张量 X，且前向沿 K 归约、反向沿 M 归约，创建了根本性的布局冲突。Conv2D、BatchNorm、LayerNorm 也属于此类。

::: block-1
这个分类直接圈定了三个核心挑战：:mdnice-highlight[解决 FTD-D 算子中的前向-反向布局冲突、消除 FTI-D 算子中的转换开销、在全局范围内选择布局以最小化转换成本]。FBLayout 的三个核心技术分别对应这三个挑战。
:::

### 2.2 四个形式化问题

> 论文将布局优化问题形式化为四个研究问题。

**Q1 是统一布局设计**：对于存在归约布局冲突的张量，是否存在一种统一布局 $L_u(t)$，能为前向和反向都提供接近最优的性能，同时避免显式转换的开销？

**Q2 是变换消除**：在什么条件下，变换操作序列可以被替换为代价可忽略的逻辑坐标映射？这需要确定何时物理数据变换可以折叠为轻量级索引计算，同时保持语义正确性和缓存效率。

**Q3 是全局布局优化**：有什么实用算法能以有界次优性解决真实训练图的全局布局分配问题？指数级搜索空间需要高效的启发式方法，考虑张量复用模式和融合机会。

**Q4 是 2.5D 内存映射**：如何在 2.5D 纹理内存上高效实现统一布局，同时遵守基于坐标的索引和 RGBA 通道组织？这需要开发缓存感知的映射策略。

| 问题 | 描述 |
| --- | --- |
| Q1 是统一布局设计 | 对于存在归约布局冲突的张量，是否存在一种统一布局 $L_u(t)$，能为前向和反向都提供接近最优的性能，同时避免显式转换的开销？ |
| Q2 是变换消除 | 在什么条件下，变换操作序列可以被替换为代价可忽略的逻辑坐标映射？这需要确定何时物理数据变换可以折叠为轻量级索引计算，同时保持语义正确性和缓存效率。 |
| Q3 是全局布局优化 | 有什么实用算法能以有界次优性解决真实训练图的全局布局分配问题？指数级搜索空间需要高效的启发式方法，考虑张量复用模式和融合机会。 |
| Q4 是 2.5D 内存映射 | 如何在 2.5D 纹理内存上高效实现统一布局，同时遵守基于坐标的索引和 RGBA 通道组织？这需要开发缓存感知的映射策略。 |

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/13.webp)

::: image-caption
图 7：FBLayout系统总览。输入为联合前向-反向训练图，经过图分析、R-Tile布局设计、基于Tile的索引变换消除、激活引导的全局布局选择四个阶段，输出优化后的R-Tile布局训练图。其中，图分析阶段利用算子分类识别优化目标和布局冲突；R-Tile 阶段设计统一的纹理感知 Tile 布局，解决前向-反向归约冲突；基于 Tile 的变换消除阶段跨相邻算子传播 Tile 结构，用轻量坐标重映射替换显式布局变换；全局布局选择阶段在整个计算图上平衡变换开销与执行效率。
:::

::: block-1
算子分类是 FBLayout 整个系统的基石。:mdnice-highlight[通过两个维度将所有算子归入四象限，FBLayout 精确识别出哪些算子是布局冲突的"震中"（FTD-D 的 MatMul），哪些是可以被消除的"噪声"（FTI-D 的 transpose/reshape），哪些是可以自适应的"柔性节点"（FTI-C/FTD-C）]。这种分类使得后续的布局决策有了明确的优先级和传播规则。
:::

## 三、R-Tile：面向多维归约的统一布局

归约操作在一个 Tile 上加载的是同一组元素，无论归约方向如何——只有聚合顺序不同。如果能让这些 Tile 元素在物理上对任意遍历方向都保持连续，单一布局就足以服务所有归约方向。

### 3.1 Tile 不变性观察

> R-Tile（Reduction-aware Tiling）的核心洞察来自一个简单但深刻的观察：矩阵乘法前向沿 K 归约，反向沿 N 归约（对转置后的权重），但两者加载的权重矩阵 Tile 是完全相同的。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/14.webp)

::: image-caption
图 9：矩阵乘法不同归约轴下的Tile不变性示例。不同颜色深浅表示不同Tile。(a)沿k轴归约；(b)W转置后沿n轴归约。尽管做了转置，访问的是完全相同的Tile（W0, W1, ...）作为原子单元，仅Tile内遍历方向不同。
:::

以图 9 为例，前向计算 $Y_{m \times n} = X_{m \times k} W_{k \times n}$ 时，权重 W 被划分为 $W_0, W_1, ..., W_7$ 等 4×4 Tile；反向计算 $Y_{h \times k} = X_{h \times n} W^T_{n \times k}$ 时，$W^T$ 被划分为对应的 Tile，每个 Tile 内的元素集合与前向完全一致，只是遍历方向从水平变为垂直。

::: block-1
:mdnice-bold-italic[这个 Tile 不变性意味着：我们不需要为每个归约方向设计不同布局，只需要保证 Tile 内部的元素在任意方向上都能被连续访问，并且 Tile 之间的遍历在二维纹理空间中保持空间局部性。] 传统线性布局做不到这一点——行主序只能保证行方向连续，列方向必然跨步。R-Tile 通过利用移动 GPU 纹理内存的二维特性解决了这个问题。
:::

### 3.2 Tile 形状候选生成

> R-Tile 的构建分两步：Tile 形状候选生成和物理放置优化。

Tile 形状直接影响内存访问效率、缓存利用率和计算吞吐量。传统方法如 TVM 依赖经验性能反馈在巨大搜索空间中寻找，耗时且不实用。**R-Tile 采用架构感知的剪枝策略，通过三个优先级递减的约束逐步过滤候选。**

- **C1 是计算架构对齐**。Tile 尺寸必须有效映射到 GPU 并行性。:mdnice-bold-italic[R-Tile 强制 warp 级对齐——Tile 维度必须是 warp 大小的整数倍（ARM Mali 为 32，高通 Adreno 为 64），确保并行线程充分利用 SIMD 执行单元]，没有空闲通道。
- **C2 是内存架构对齐**。移动纹理缓存容量有限且偏好特定访问模式。:mdnice-bold-italic[R-Tile 只保留能放入 L1 纹理缓存且与缓存行边界（通常 64 字节）对齐的 Tile，] 避免部分缓存行获取浪费带宽。
- **C3 是张量形状兼容**。为最小化填充开销，优先选择能整除张量维度的 Tile。无法精确整除时，保留填充率低于 10%的候选。

通过这三层约束，R-Tile 构建了一个紧凑的、架构对齐的候选空间，无需穷举搜索。

### 3.3 两级物理映射

> R-Tile 通过两级映射将张量数据放入二维纹理内存：Tile 间映射（inter-tile）控制 Tile 在二维纹理空间中的排列，影响 L2 缓存复用；Tile 内映射（intra-tile）控制每个 Tile 内部元素的排列，影响 L1 纹理缓存效率。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/15.webp)

::: image-caption
图 10：R-Tile布局组织。(a)Tile内内存打包保持空间连续性，小Tile用行主序，大Tile用2D分块；(b)Tile间按行主序排列形成全局张量布局，Tile数量多的维度放在水平方向。
:::

> Tile 内映射方面，移动 GPU 以矩形缓存 Tile 块 $(C_h \times C_w)$ 为单位获取纹理数据。一个 warp 读取一个元素时，硬件隐式加载整个缓存 Tile，效率取决于被加载元素在 eviction 前被使用的比例。R-Tile 覆盖的纹理缓存 Tile 数量为：

$$
N_{cache} = \left\lceil \frac{tile_h}{C_h} \right\rceil \left\lceil \frac{tile_w}{C_w} \right\rceil
$$

最小化 $N_{cache}$ 直接提升空间局部性。

- 当 $tile_w \leq C_w$ 时，R-Tile 以水平条带（行主序）存储元素，因为任何水平遍历都保持在单行的单个缓存行内。
- 当 $tile_w > C_w$ 时，水平遍历跨多个缓存行，R-Tile 切换到紧凑的 2D 分块，块尺寸通过离线基准测试选择，使 $N_{cache}$ 最小化。

Tile 间映射方面，移动 GPU 纹理缓存的二维块是矩形而非正方形——水平方向跨度大于垂直方向。

> 图 4 的实验证实水平跨步访问比垂直访问吞吐量高 1.6 倍。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/16.webp)

::: image-caption
图 4：纹理内存空间局部性。(a)四种不同访问模式：列主序、行主序、Block-4分块、Block-8分块；(b)四种模式在不同线程数下的吞吐量。行主序和分块访问的吞吐量显著高于列主序。
:::

R-Tile 因此采用简单启发式：按行主序排列 Tile，将需要更多 Tile 遍历的维度作为主（水平）维度，使沿该维度的连续 Tile 在纹理内存中水平相邻，与更宽的缓存块范围对齐。

> 图 8 直观对比了两种布局。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/17.webp)

::: image-caption
图 8：传统内存布局与R-Tile布局的对比。(a)传统行主序布局在2.5D纹理内存中，沿Dim1归约时连续访问，沿Dim2归约时垂直跨步访问浪费带宽；(b)R-Tile布局，沿Dim1归约时水平连续，沿Dim2归约时Z字形遍历仍在单个2D纹理缓存块内，带宽浪费极小。
:::

传统行主序下，沿 Dim2 归约需要垂直遍历，跨越缓存块边界，大量加载的数据未被使用就被 evict。R-Tile 下，沿 Dim2 归约时通过 Z 字形路径在单个 2D 纹理缓存块内完成遍历，每一条缓存线都被充分利用。:mdnice-bold-italic[这从根本上利用了纹理内存的二维特性——通过将归约维度重映射到水平轴，R-Tile 以一维线性内存无法支持的方式利用了缓存块的二维空间范围。]

对于被多个不同归约模式的算子消费的张量（比如同时被前向和反向使用的权重），R-Tile 选择使整体性能最大化的配置：

$$
R\text{-}Tile^* = \arg\max_{R\text{-}Tile} \sum_{op \in ReductionOps} \text{Performance}(R\text{-}Tile, op)
$$

其中 $\text{Performance} = \text{GFLOPS} \times \text{MemEfficiency}$，$\text{MemEfficiency} = \frac{\text{MinimalBytes}}{\text{ActualBytes}}$ 衡量内存流量效率。

::: block-1
:mdnice-highlight[R-Tile 的精髓在于：它不试图让物理内存同时满足两个方向的"一维连续"——这在线性内存中不可能；而是利用移动 GPU 纹理内存的二维特性，让任意归约方向的访问路径都落在 2D 缓存块内部。] Tile 不变性是理论基础，架构感知剪枝是工程保障，两级映射是实现手段。三者结合，使得单一物理布局无需转换即可同时高效服务前向和反向。
:::

## 四、基于 Tile 的索引变换消除

> Transformer 模型中的许多变换仅仅重排 Tile 而不改变 Tile 内部结构——如果一个变换只是重新排列 Tile 的位置同时保持每个 Tile 的元素布局，我们就可以用 Tile 粒度的轻量索引重映射替换物理数据移动。

### 4.1 从物理变换到逻辑重映射

布局变换算子（Transpose、Reshape 等 FTI-D 算子）在训练图中无处不在，在 R-Tile 布局下它们需要在分块和线性表示之间做昂贵转换。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/18.webp)

::: image-caption
图 11：基于消除的布局变换与R-Tile布局在计算图中的对比。(a)原始计算图；(b)优化前需要在R-Tile和线性布局之间插入两次Layout Convert；(c)优化后通过Tile传播与索引消除，两个MatMul直接以R-Tile对接。B为batch大小，D为隐藏维度，S为序列长度，H为注意力头数，HD为每头维度。
:::

图 11(b)显示，两个 MatMul 之间的 Reshape 和 Transpose 子图需要两个显式的 Layout Convert 节点来在 R-Tile 和线性布局之间切换，增加了不必要的数据移动开销。

> FBLayout 的关键洞察是：这些变换中的绝大多数只重排 Tile 而不改变 Tile 内部结构。

以多头注意力中典型的 reshape+transpose 序列为例：输入张量(B,S,D)被 reshape 为(B,S,H,HD)，再 transpose 为(B,H,S,HD)。由于 R-Tile 的 Tile 沿 HD 维度定义，reshape 将 D 分解为 H×HD 时每个 Tile 的内部元素不变，transpose 交换 S 和 H 轴时也只是改变 Tile 的位置，Tile 内部的元素排列完全保持。

这意味着整个变换序列可以折叠为一次 Tile 级别的坐标映射，无需任何物理数据移动。

### 4.2 Tile 传播合法性检查

> Tile 传播确定变换序列是否可以表达为 Tile 级坐标重映射而不破坏 R-Tile 结构。FBLayout 通过三个依赖检查来验证合法性。

- **第一个检查是 R-Tile 维度依赖**。利用为归约算子确定的 R-Tile 配置，如果矩阵乘法使用 $(tile_w, tile_h)$ 的 R-Tile，前置变换必须保持这些 Tile 维度。检查维度操作是否维持所需的 Tile 结构。
- **第二个检查是 Tile 边界保持**。对于 reshape 等拆分操作，拆分后的维度必须与 R-Tile 边界对齐。例如将 hidden=768 reshape 为(heads=12, head_dim=64)，若 R-Tile 的 $tile_w=64$，则拆分必须确保 tile 是 head_dim 的倍数以保持对齐。$tile_w=64$ 或 $tile_w=8$ 配合 head_dim=64 都能保持 Tile 边界，而错位则破坏 Tile 原子性。
- **第三个检查是 Tile 内连续性**。验证 transpose 和 permute 操作在 R-Tile 的最内层维度保持 stride-1，维持元素的连续排列。stride-1 丢失会使 Tile 元素分散到内存各处。

::: block-1
**当所有检查通过，Tile 在整个变换链中传播，内部结构保持完整，序列被折叠为单个 Tile 坐标映射**。如果验证失败，FBLayout 会尝试替代 R-Tile 配置或调整分块维度，最后才回退到显式布局转换:mdnice-highlight[。实践中验证失败几乎只发生在元素数量变化的步骤（比如分组查询注意力中的扩展），一旦新形状稳定，通常可以重新建立有效的 R-Tile 配置。]
:::

### 4.3 索引变换消除

> 一旦验证变换保持 Tile 连续性，FBLayout 通过将变换算子替换为 Tile 级索引映射来消除逐元素索引计算。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/19.webp)

::: image-caption
图 12：基于Tile的索引变换工作流。上半部分为Tile传播：输入(B,S,D)经Reshape到(B,S,H,HD)再经Transpose到(B,H,S,HD)，Tile内部结构始终保持；下半部分为索引消除：源R-Tile到目标R-Tile通过预计算的Tile级线性映射完成，无需物理数据移动。
:::

> 如图 12 所示，每个 Tile 被抽象为行主序编号的单个元素，实现 Tile 坐标翻译：

$$
T_{dst}[i] = f(T_{src}[j])
$$

其中 $f$ 将源 Tile 索引 $j$ 映射到目标 Tile 索引 $i$。这个函数在图构建时预计算，消除了运行时逐元素地址计算。

::: block-1
为进一步减少堆叠变换的开销，FBLayout 对 Tile 索引计算应用强度削减。直接使用线性表示计算会引入 GPU 上昂贵的取模和除法运算。FBLayout 分析索引依赖并应用数学化简规则——例如当 $C_a \mod C_b \equiv 0$ 时，$i \mod C_a \mod C_b$ 化简为 $i \mod C_b$，这在 R-Tile 布局中维度对齐的情况下非常常见。
:::

当多个变换堆叠时（比如注意力机制中 reshape 后接 transpose），FBLayout 在离线时组合多个 Tile 映射：

$$
T_{final} = f_3 \circ f_2 \circ f_1(T_{initial})
$$

并对组合后的函数应用强度削减。这将连续变换合并为单个化简后的 Tile 坐标函数，完全消除中间变换 kernel。

::: block-1
**基于 Tile 的索引变换消除的本质是"只搬坐标不搬数据"。** 传统框架中 transpose 和 reshape 虽然逻辑上只是重排视图，但在物理内存上要么真的搬数据（显式转换），要么让后续 kernel 承受跨步访问（布局复用）。

:mdnice-highlight[FBLayout 识别出 Tile 粒度的不变性，将物理数据移动转化为编译时预计算的 Tile 索引重映射，代价从 O(元素数)的内存操作降为 O(Tile 数)的整数运算]。**在 LLM 中这覆盖了超过 95%的变换操作。**
:::

## 五、激活引导的全局布局选择

训练图需要跨算子链的协调布局决策。核心优化问题涉及确定张量何时应采用 R-Tile 布局、何时应进行变换，在整个前向-反向图上平衡开销与效率。FBLayout 利用激活复用模式引导布局传播。

### 5.1 算子对的布局决策

> FBLayout 的算子分类为跨算子组合的系统性布局传播提供了基础。当两个算子通过共享张量连接时，它们的布局需求相互作用，产生指导优化策略的模式。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/20.webp)

::: image-caption
表 2 总结了算子对的布局决策规则。当生产者和消费者都是 FTI-C 时，直接融合；任一方为 FTD-C 时，布局由 FTD-C 主导；任一方为 FTI-D 时，通过索引消除处理；任一方为 FTD-D 时，R-Tile 布局强制传播到连接张量。唯一需要贪心搜索的场景是生产者和消费者都是 FTD-D（连续 MatMul 链），此时需要在多个候选 R-Tile 配置间权衡。
:::

> 表 2 揭示了三个关键传播模式。

- 第一，FTD-D 算子主导布局选择：当算子对中任一算子是 FTD-D 时，其 R-Tile 布局传播到连接的张量，因为解决前向-反向冲突有刚性要求。
- 第二，FTI-D 算子通过基于 Tile 的索引变换被消除，无需显式布局决策。
- 第三，FTI-C 和 FTD-C 算子灵活适应传播来的布局，因为无论 Tile 组织如何它们都保持空间局部性。

这种系统性传播将 FTD-D 算子识别为锚点，它们的 R-Tile 布局流经柔性算子并吸收变换算子。:mdnice-bold-italic[布局变换仅在必要时插入，比如图边界或不兼容 Tile 表示的算子处。关键在于，这种策略联合优化了两个 pass——基于需要跨 pass 复用的激活做出的布局决策自然地将 R-Tile 选择应用到两个阶段，没有激活复用的操作则从前向对应项继承布局。]

### 5.2 连续布局冲突算子的优化

> 最具挑战性的场景是多个 FTD-D 算子连续连接，见表 2 中黄色单元格。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/21.webp)

::: image-caption
表 2 总结了算子对的布局决策规则。
:::

比如注意力块中 $Q \times K^T$ 后接 $(QK^T) \times V$。每个操作可能基于其特定归约模式偏好不同的 R-Tile 配置。天真地在每个操作之间插入布局变换会引入难以承受的开销，而强制所有操作使用单一 R-Tile 配置可能降低某些操作的性能。

> FBLayout 观察到，Transformer 模型中连续操作的候选 R-Tile 配置在很大程度上重叠。这源于 Transformer 架构的相似性：

- 注意力块内 $Q \times K^T$ 和 $(QK^T) \times V$ 共享相似的 MNK 维度；
- FFN 层内连续的 MatMul 也展现维度规律性。

显著的维度差异主要出现在大块之间（比如注意力与 FFN 之间），那里布局变换已经被更长的计算序列摊销。

**为处理可能有冲突布局偏好的连续 FTD-D 序列，FBLayout 应用张量级感知的贪心搜索**：识别 FTD-D 链，收集候选 R-Tile 配置，按输入张量大小排序（更大的张量主导开销）。然后按顺序尝试每个配置，尝试将其统一应用到整个序列，仅在有益处时插入变换，选择总成本（计算+变换开销）最小的排列。:mdnice-bold-italic[例如在 QKV 注意力中，$Q \times K^T$ 输入更大，FBLayout 先尝试其偏好的 R-Tile 用于两个操作，仅在有益时为 $(QK^T) \times V$ 插入变换。]

::: block-1
总的来说，:mdnice-highlight[全局布局选择的智慧在于"让最痛的地方决定布局"。FTD-D 算子是布局冲突最剧烈的节点，它们的 R-Tile 需求是刚性约束]；其他算子要么可以被索引消除吸收，要么可以灵活适配。

通过将 FTD-D 算子作为锚点向外传播布局，FBLayout 避免了在每个算子边界做局部最优但全局次优的决策。连续 FTD-D 算子的贪心搜索则利用了 Transformer 架构的维度规律性，在极小的搜索成本下取得接近全局最优的结果。
:::

## 六、实验评估与深度分析

> **FBLayout 在 MNN 之上构建**，包含约 9.2K 行 C++和 OpenCL 代码，扩展了自动微分和所有评估模型的算子实现。:mdnice-bold-italic[评估横跨七款代表性 Transformer 模型、三款商用手机、两类移动 GPU 架构。]

### 6.1 实验设置

评估覆盖三大架构族：编码器-only（BERT-Large 340M、ViT-Large 304M）、解码器-only（Llama3.2-1B、Qwen2.5-1.5B、Gemma2-2B）、编码器-解码器（Whisper-Large 1.55B、Stable Diffusion v1.5 1.4B）。除 ViT 使用 adapter 微调外，其余模型均使用 LoRA 微调，插入 QKV 投影和 MLP 块。

- 基线框架为 MNN、TVM 和 TFLite。
- 目标设备包括一加 Ace5 Pro（骁龙 8 Elite，Adreno 830，16GB）、一加 Ace10 Pro（骁龙 8 Gen1，Adreno 730，12GB）、一加 Ace5 Ultra（天玑 9400+，Mali Immortalis-G925，16GB）。

> 由于 FBLayout 仅改善数据访问和计算效率，不改变模型架构、张量精度或训练目标，微调精度在各框架间完全一致，**因此所有比较聚焦于执行效率**。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/22.webp)

::: image-caption
表 3 列出了七款评估模型的详细配置。解码器-only 模型包括 Llama3.2-1B（10 亿参数，batch 1，序列长度 512）、Qwen2.5-1.5B（15 亿参数）、Gemma2-2B（20 亿参数）；编码器-only 模型包括 BERT-Large（3.4 亿参数，batch 4，序列长度 512）和 ViT-Large（3.04 亿参数，batch 8，输入 3×224×224）；编码器-解码器模型包括 Whisper-Large（15.5 亿参数，仅微调解码器约 5.9 亿参数，batch 2，300 帧）和 Stable Diffusion v1.5（14 亿参数，仅微调 UNet 约 8.6 亿参数，batch 2，512×512 分辨率）。
:::

### 6.2 端到端性能

> 图 13 报告了端到端训练吞吐量。FBLayout 在所有架构上都取得了一致的加速。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/23.webp)

::: image-caption
图 13：FBLayout与其他框架在一加Ace5 Pro（骁龙8 Elite）上的训练速度对比。七个模型从左到右分别为Llama3.2-1B、Qwen2.5-1.5B、Gemma2-2B、Bert-L、VIT-L、Whisper-L、SD1.5。FBLayout在所有模型上均显著领先。
:::

- 在解码器-only LLM 上相比 MNN 加速 3.9 到 4.1 倍，相比 TFLite 加速 4.3 到 4.9 倍，相比 TVM 加速 5.4 到 5.7 倍；
- 在编码器-only 模型上相比 MNN 加速 2.2 到 3.2 倍，相比 TFLite 加速 2.6 到 4.6 倍，相比 TVM 加速 2.9 到 5.2 倍；
- 在编码器-解码器模型上相比 MNN 加速 2.3 到 3.5 倍，相比 TFLite 加速 3.5 到 4.5 倍，相比 TVM 加速 4.0 到 4.7 倍。

加速比的差异源于 Transformer 架构的结构差异。

- 解码器-only 和编码器-解码器模型依赖密集的 QKV 交互、大投影矩阵和频繁的 reshape/transpose 操作，这些模式加剧了布局冲突，也增加了 MNN 和 TFLite 等后端中 kernel 偏好布局转换的开销。
- Stable Diffusion 混合卷积层和注意力层，进一步加剧了 TFLite 低效处理的布局转换。TVM 受益于算子融合，但缺乏面向归约的布局或移动 GPU 纹理感知映射，当归约或 reshape 算子主导时性能严重退化。

### 6.3 各优化组件贡献分解

> 图 14 逐步启用各组件。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/24.webp)

::: image-caption
图 14：优化分解，包含训练速度和内存总访问次数。四个配置分别为MNN基线、仅R-Tile、R-Tile+Tile索引消除、完整FBLayout（含全局布局选择）。
:::

- **仅启用 R-Tile 就带来平均 1.88 倍加速**，全局内存访问减少 2.50 倍，:mdnice-bold-italic[因为统一布局消除了前向-反向归约冲突并产生更连续的内存访问模式。]
- **加入基于 Tile 的索引消除再提供 1.45 倍提升**，内存访问再减少 1.86 倍，:mdnice-bold-italic[因为去除了物理布局转换并暴露了更多算子融合机会。]
- **加入全局布局选择再贡献 1.31 倍加速**，内存读取再减少 1.56 倍，:mdnice-bold-italic[因为一致的布局传播避免了内存密集区域不必要的 R-Tile 切换。]

三个优化提供互补收益，共同促成了显著的性能提升。

### 6.4 缓存效率分析

> 图 15 显示，FBLayout 平均发出 3.5 倍更少的全局内存访问，产生 4.2 倍更少的缓存未命中。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/25.webp)

::: image-caption
图 15：缓存效率分析。所有结果以FBLayout为基准归一化。(a)全局内存访问次数，FBLayout平均减少3.5倍；(b)缓存未命中次数，FBLayout平均减少4.2倍。解码器-only LLM 受益最大：FBLayout 相比所有基线一致减少 4 倍以上缓存未命中。大型 Transformer 工作负载由于大投影矩阵、多头注意力算子和 reshape 密集的执行模式，对移动 GPU 缓存造成巨大压力。
:::

基线框架应用不一致的布局和频繁的布局转换，遭受非连续和低局部性内存访问，触发过度缓存抖动。**FBLayout 通过 R-Tile 布局维持空间局部性、增加数据复用、减少矩阵和注意力密集操作期间的全局内存压力，缓解了这个瓶颈。**

### 6.5 R-Tile 微观分析

> 图 16 对一个主导性 MatMul 算子做了深入剖析。前向 M=512、K=2048、N=8192，反向 M=512、K=8192、N=2048，双向都触发归约操作。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/26.webp)

::: image-caption
图 16：R-Tile优化分解。基准MatMul算子：前向计算AB（M=512, K=2048, N=8192），反向复用同一权重矩阵计算$dA=dO·B^T$（M=512, K=8192, N=2048）。(a)kernel时间；(b)GPU ALU利用率；(c)相对R-Tile的内存访问次数。
:::

**R-Tile 在两个 pass 上平均比 MNN 快 3.5 倍、比 TFLite 快 4.6 倍，ALU 利用率超过 60%（MNN/TFLite 低于 21%）**。MNN 的反向 kernel 时间高达 100ms（R-Tile 仅约 15ms），TFLite 反向 kernel 时间更是高达约 120ms。:mdnice-bold-italic[内存访问方面，MNN 反向访问量是 R-Tile 的 53 倍，TFLite 反向访问量是 R-Tile 的 53 倍以上。]

这些数字清晰地说明：**R-Tile 的统一归约友好布局为 K 归约（前向）和 N 归约（反向）都保持了空间局部性**，无需布局切换。

### 6.6 Tile 索引消除效果

> 图 17 评估了 Tile 索引消除对布局相关开销的影响。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/27.webp)

::: image-caption
图 17：基于Tile的布局索引变换消除的优化分解。LT表示布局变换。(a)不同注意力配置下数据变换的kernel时间，C1=(512,32,2048)、C2=(512,16,1024)、C3=(512,8,1024)；(b)布局变换算子数量；(c)缓存未命中次数。
:::

- 在 MHA 工作负载中，基于 Tile 的实现一致优于 MNN 和 TFLite，布局变换延迟最多减少 2.8 倍和 5.1 倍。
- 显式布局变换算子数量最多减少 2.1 倍，缓存未命中平均减少 1.5 倍。

覆盖率方面，基于 Tile 的索引重映射成功处理了超过 86%的所有变换（LLM 中超过 95%），剩余显式转换占总运行时间不到 6%。这些回退发生在变换前后元素大小不同时（比如分组查询注意力中的扩展），阻止了直接的 Tile 级索引重映射。

### 6.7 编译时间与系统开销

> 表 4 对比了各框架在 Llama3.2-1B 训练图上的总编译时间。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/28.webp)

::: image-caption
MNN 需要 723 秒，TFLite 需要 862 秒，TVM 超过 4800 秒。即便关闭架构感知剪枝的 R-Tile 也需要 221 秒，而完整的 FBLayout 仅需 6.1 秒。
:::

编译时间直接影响可用性。FBLayout 将调优时间从 MNN 的 723 秒、TFLite 的 862 秒、TVM 的超过 4800 秒降至 6.1 秒。**这源于两个因素：R-Tile 的架构感知剪枝基于纹理内存约束过滤不可行的 Tile 候选，激活引导的全局布局规则通过强制端到端一致性大幅缩小布局搜索空间**。:mdnice-bold-italic[即使没有剪枝的 R-Tile 也需要 221 秒，说明架构约束剪枝贡献了超过 36 倍的编译加速。]

> 图 18 显示离线优化总开销在 2.6 秒到 6.1 秒之间，全部在训练开始前完成。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/29.webp)

::: image-caption
图 18：不同运行条件下各组件的系统开销。(a)不同模型的离线优化时间分解；(b)不同输入序列长度下的开销变化。
:::

对于动态输入形状，初始编译后（seq=128 时 6.1 秒），形状变化带来更低开销（seq=512/1024 时 3.5/3.6 秒），因为图处理被跳过，全局布局优化简化为轻量冲突检查。仅当变化维度位于归约冲突轴上时 R-Tile 优化和索引变换才重新运行，否则复用先前结果。

### 6.8 能耗与可移植性

> 图 19 显示 FBLayout 总能耗降低 3.5 到 6.3 倍。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/30.webp)

::: image-caption
图 19：5轮训练内的能耗对比，以FBLayout为基准归一化。设备为骁龙8 Elite。FBLayout相比MNN和TFLite减少3.5到6.3倍能耗。
:::

**能耗节省来自两个因素：更短的训练时间直接减少活跃能耗，更少的缓存未命中既避免了能耗昂贵的 DRAM 访问**，又消除了原本会延长执行并进一步增加能耗的流水线停顿。

> 图 21 显示 FBLayout 在联发科天玑 9400+和高通骁龙 8 Gen1 上都取得了一致加速。

![文章插图](/articles/2026-09-08-fblayout-tile-gpu-llm-mnn-574bdcc5/images/31.webp)

::: image-caption
图 21：Mali GPU和高通骁龙8 Gen1上的可移植性评估。(a)Adreno 730（骁龙8 Gen1）；(b)Mali Immortalis-G925（联发科天玑9400+）。FBLayout在两类GPU上均取得一致加速。
:::

在 Adreno 730 上加速 1 到 5 倍，在 Mali G925 上加速 1.5 到 7.3 倍。**FBLayout 能跨平台泛化，因为 R-Tile 针对移动 GPU 共享的通用纹理内存特性而非 GPU 特定功能，加上基于 Tile 的变换消除能适应各种架构**。适配新 GPU 需要一次性的、与模型无关的硬件特性刻画（约 21 到 26 秒），刻画 warp 大小、纹理内存 L1/L2 缓存大小和访问模式等参数，之后可跨所有模型复用。

::: block-1
实验数据展示出因果链：

- R-Tile 统一布局解决了最根本的前反向归约冲突，带来最大单块加速（1.88 倍）和最大内存访问削减（2.5 倍）；
- Tile 索引消除去除了物理变换开销，使能更多算子融合；
- 全局布局选择确保布局决策在整个图上协调一致。

三者叠加产生 2.2 到 5.7 倍端到端加速、3.5 倍内存访问减少、4.2 倍缓存未命中减少、6.1 秒编译时间和 3.5 到 6.3 倍能耗降低。
:::

## 结语

> FBLayout 作为加速移动 GPU 上 LLM 微调的工作，它识别出前向-反向布局冲突这个被长期忽视的系统性瓶颈，通过 R-Tile 统一布局、Tile 级索引变换消除、激活引导全局布局选择三项核心技术，在不改变模型精度的前提下取得了 2.2 到 5.7 倍的端到端加速。

这项工作揭示了一个重要趋势：当大模型从云端走向端侧，从推理走向训练，我们需要重新审视硬件与软件的接口设计。:mdnice-bold-italic[移动 GPU 的纹理内存是为图形渲染的二维空间局部性设计的，传统深度学习框架将其当作一维线性内存使用，在训练的双向数据流下必然低效。FBLayout 证明了，只有将张量布局与移动 GPU 的 2.5D 纹理内存层次协同设计，才能真正释放端侧训练的硬件潜力。]

::: related-title
**相关推荐**
:::

::: related-links
- **[移动端内存不够？阿里 MNN-LLM：移动端通用 LLM 推理引擎，DRAM-Flash 混合存储和模型量化，显著减少内存使用！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888799&idx=1&sn=507601a48d39a08b06557b7c95b2f12a&scene=21#wechat_redirect)**
- **[HeteroLLM：利用移动端 SoC 实现 NPU-GPU 并行异构 LLM 推理！以 高通8 Gen 3的NPU GPU为例](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888774&idx=1&sn=dc4a0c912248f7f3f1a48806d2d7bfe9&scene=21#wechat_redirect)**
- **[让端侧AI不再难用！谷歌开源 LiteRT-CLI：用一行命令重写 TFLite 模型工作流，从 HuggingFace 到 NPU](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903206&idx=1&sn=419b84c1b6a835de19f43cdcf6d37e11&scene=21#wechat_redirect)**

:::
