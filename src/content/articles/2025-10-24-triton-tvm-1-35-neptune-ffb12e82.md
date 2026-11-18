关键词：**Tensor Compilers**、:mdnice-bold-italic[ML Operator Fusion]、**GPU Parallelism Optimization**、:mdnice-bold-italic[Reduction Operators]、Data Locality Enhancement、Compilation Performance Benchmarking

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/01.webp)

- Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs
- https://arxiv.org/abs/2510.08726
- **https://github.com/uiuc-arc/neptune**
- 约 2700 字，阅读需 9 分钟，**播客 21 分钟**

相关推荐

- [超越 Triton？任意低精度 GPGPU 计算虚拟机，加速 LLM 服务](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888475&idx=1&sn=c452ca77513dd283a6850181735ecaa0&scene=21#wechat_redirect)
- [Tawa：面向现代 GPU 的自动 warp 特化创新方案，性能超cuBLAS 1.1 倍、Triton 1.2倍的编译器设计](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447893227&idx=1&sn=bccbe9189be51c038845680c54363072&scene=21#wechat_redirect)
- [谈谈TVM与领域专用架构：DSA优化思路、TVM落地挑战、硬件、NVDLA架构思考](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887071&idx=1&sn=c06f737da907fd03cd5233685049b50e&scene=21#wechat_redirect)

> 在深度学习部署中，**算子融合** 是提升性能的核心优化——把多个算子合并成一个内核，减少数据在显存和寄存器间的来回搬运，从而降低延迟。

但面对注意力机制这类包含复杂循环依赖的归约操作（比如矩阵乘法、Softmax 的组合），**传统张量编译器往往束手无策，只能依赖手动优化的内核** 如 FlashAttention。

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/02.webp)

::: image-caption
Neptune 的核心工作流程，分为输入、处理、输出三部分。输入是张量表达式（定义计算逻辑）和转换模板（指定高层优化规则）；处理环节含三步：模板引导优化器按模板做算子融合等高层优化，循环 - 张量块转换器将优化后的循环代码转为硬件友好的张量块代码，张量块优化器负责底层硬件适配，自动调优器筛选最优参数；最终输出可直接在 GPU 运行的内核模块，全程兼顾用户可控性与自动化效率，无需手动处理底层细节
:::

今天要解读的 :mdnice-bold-italic[Neptune 编译器，恰恰解决了这个痛点：它能自动融合带复杂依赖的归约算子，在 4 类 NVIDIA/AMD GPU 上实现平均 1.35 倍的性能提升]，甚至超越手动优化的主流方案。这篇文章就带大家拆解 Neptune 的核心创新和工作思路。

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/03.webp)

## 本文目录

- 一、核心痛点：为什么传统编译器搞不定注意力融合？
- 二、Neptune 的核心创新：打破依赖+代数修复
  - 创新 1：先进算子融合范式（两大核心算法）
  - 创新 2：调度与 tile 优化的无缝集成
- 三、实验结果：性能碾压主流方案
  - 1. 对比张量编译器（Triton/TVM/FlexAttention）
  - 2. 对比手动优化库（FlashAttention/CUTLASS/cuDNN）
  - 3. 可编程性优势
- 四、总结：Neptune 的价值与未来

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/04.webp)

::: image-caption
交流加群请在 NeuralTalk 公众号后台回复：加群
:::

## 一、核心痛点：为什么传统编译器搞不定注意力融合？

> 注意力机制的计算流程包含 4 个关键步骤：矩阵乘法（matmul）→ 逐元素除法 → Softmax → 第二次矩阵乘法。:mdnice-bold-italic[其中 3 个步骤都是归约操作] （需要循环累加/取极值），存在严重的**循环依赖** ——比如 Softmax 的指数计算依赖全局最大值，而最大值又要遍历所有元素才能确定。

传统编译器面临两个致命问题：

1. **依赖壁垒**：如刚提到的归约操作，规约操作循环依赖会阻止算子融合，强行融合会导致计算结果错误，比如提前读取未计算完成的最大值；
2. **优化割裂**：调度式编译器（如 TVM）擅长高层优化但搜索空间过大，tile 式编译器（如 Triton）擅长硬件适配但难以处理跨算子融合，两者无法协同。

为了更直观理解，看原论文的经典示例：

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/05.webp)

::: image-caption
图1 归约融合问题示例。(a)是原始程序（3 个独立循环：求行最大值 s_max、指数计算 s_exp、求和 s_sum）；(b)是 naive 融合的错误结果（s_exp 提前读取未完成的 s_max，导致数值错误）；(c)是 Neptune 的正确融合（通过添加修复项实现依赖兼容）；(d)-(f)是具体数值示例，蓝色箭头为正确依赖，橙色为错误依赖，绿色为修复后结果
:::

## 二、Neptune 的核心创新：打破依赖+代数修复

> Neptune 的核心思路是：**不回避依赖，而是主动打破依赖，再用代数修正表达式弥补错误**。这个思路看似“冒险”，却能在不损失正确性的前提下，实现高效融合。

### 创新 1：先进算子融合范式（两大核心算法）

Neptune 提出了**两种针对归约算子的融合算法**，分别适配**不同场景**：

#### 1. Rolling Update（滚动更新融合）—— 适合 Prefill 场景

Prefill 是大语言模型“填充上下文”的阶段（查询序列与键值序列长度相等），需要优先保证数据复用。算法核心步骤：

1. 数据流重组：内联所有中间算子，让归约操作直接关联目标算子；
2. 朴素融合：将所有归约循环强行融合到同一外层循环下（故意打破依赖）；
3. 模式匹配：识别归约算子的计算模式，如“最大值+指数+求和”的 Softmax 结构；
4. **求解修复函数**：:mdnice-bold-italic[通过符号计算推导“代数修复项”，修正朴素融合的错误]；
5. 缓存历史结果：保留归约操作的上一轮结果，供修复函数使用。

> :mdnice-bold-italic[修复函数的核心逻辑的是：用当前迭代的归约结果（如最新最大值），修正上一轮迭代的计算结果] （如之前的指数值）。

#### 2. Split-K Update（拆分 K 融合）—— 适合 Decoding 场景

Decoding 是模型生成文本的阶段（查询序列长度=1），需要优先保证并行性。算法核心思路：

1. 拆分计算：将归约操作拆分为“局部归约”和“全局归约”——局部归约并行计算部分结果，全局归约合并局部结果；
2. 延迟修复：**局部归约阶段不处理依赖（保持并行性），仅在全局归约阶段应用修复函数**；
3. 无依赖并行：局部归约相互独立，可充分利用 GPU 多核并行能力。

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/06.webp)

::: image-caption
Split-K融合示例。将图1(a)的原有的 4 次迭代拆分为 2 个局部块（j0=0/1），先并行计算局部最大值（max_l）和局部和（sum_l），再通过全局归约合并结果，修复项（exp(max_l - max_g)）确保最终结果正确
:::

### 创新 2：调度与 tile 优化的无缝集成

> Neptune 解决了传统编译器“优化割裂”的问题，设计了三层流水线：

1. 模板引导优化：用户提供高层优化模板如对 Softmax 应用 Rolling Update，编译器自动执行融合、循环变换；
2. 循环-tile 翻译：通过模式匹配识别可 tile 化的计算，将高层循环程序转换为 tileIR；
3. tile 优化：调用 Triton 编译器处理底层优化（如 TensorCore 适配、线程调度）。

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/07.webp)

::: image-caption
Neptune系统架构：输入为张量表达式（如注意力计算的数学定义）和优化模板，经过三层优化后生成可执行内核，自动调优器负责搜索最优参数
:::

## 三、实验结果：性能碾压主流方案

> Neptune 在 10 种注意力变体、4 类 GPU（NVIDIA A100/RTX 6000 Ada/RTX A5000 + AMD MI300）上进行了全面测试，核心结果如下：

### 1. 对比张量编译器（Triton/TVM/FlexAttention）

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/08.webp)

::: image-caption
320 个测试场景中，Neptune 在 284 个场景中性能最优，全局平均加速比 1.35x。AMD MI300 上提升最显著，因为传统编译器对 AMD GPU 的优化支持较弱
:::

### 2. 对比手动优化库（FlashAttention/CUTLASS/cuDNN）

**在 256 个测试场景中，Neptune 在 101 个场景中超越手动优化库**，全局平均性能达到手动优化库的 1.07x。其中：

- Decoding 场景优势明显（GQA/ALiBi 算子加速比 1.24x-5.71x）；
- 长序列场景下，Neptune 的融合优化比手动优化库更灵活，内存占用降低 47%。

![文章插图](/articles/2025-10-24-triton-tvm-1-35-neptune-ffb12e82/images/09.webp)

::: image-caption
性能对比示例。以 GQA (PF)算子为例，Neptune 在所有 GPU 和序列长度下均优于 Triton/TVM/FlexAttention，短序列（$2^{10}=1024$）优势尤为明显
:::

### 3. 可编程性优势

手动优化的 Triton 内核需要 650 行代码，而 Neptune 仅需 38 行张量表达式（定义计算）+28 行优化模板（定义调度），就能覆盖所有注意力变体，大幅降低开发成本。

## 四、总结：Neptune 的价值与未来

> Neptune 的核心贡献在于，**首次将“打破依赖+代数修复”的思路引入张量编译，让复杂归约算子的自动融合成为可能**。它不仅解决了注意力机制的性能瓶颈，更提供了一种通用的优化范式——:mdnice-bold-italic[未来任何包含复杂依赖的归约操作] （如 Transformer 的 FeedForward 层、卷积后的池化操作），**都可能通过这种思路实现自动优化。**

对于开发者而言，Neptune 的意义在于：**无需手动编写复杂的融合内核，只需通过简单模板就能获得接近甚至超越手动优化的性能，同时兼容多 GPU 架构**，降低深度学习部署的门槛。

目前 Neptune 已开源（https://github.com/uiuc-arc/neptune），感兴趣的同学可以直接体验。随着大模型向更长序列、更复杂结构发展，这类自动化、通用化的编译优化工具，必将成为深度学习部署的核心基础设施。
