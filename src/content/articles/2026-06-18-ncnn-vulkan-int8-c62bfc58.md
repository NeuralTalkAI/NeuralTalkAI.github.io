关键词：**ncnn**、:mdnice-bold-italic[Vulkan]、**int8**、推理后端、模型量化

> 本文详细介绍 ncnn Vulkan int8 推理后端的完整架构设计与落地优化，对应项目PR#6751 vulkan int8 kernels[1]，**指出该工作并非简单新增硬件dot指令，而是对现有Vulkan推理后端进行系统性架构补齐**。

项目以 CPU int8 语义对齐、跨显卡兼容、分级硬件降级为核心目标，重构推理三段式执行流程，拆分`create_pipeline_int8`等专属接口，将权重重排、维度补零等复杂布局变换前置至 Host 端，大幅简化 GPU 着色器运行时逻辑。

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/01.webp)

::: image-caption
是谁这么可爱啊？
:::

- **ncnn Vulkan int8 推理后端的架构设计与优化实现**
- 作者：nihui，https://zhuanlan.zhihu.com/p/2050925833053906453
- PR 链接：https://github.com/Tencent/ncnn/pull/6751

::: related-title
**相关推荐**
:::

::: related-links
- **[ncnn的设计理念和软件工程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885535&idx=1&sn=33b9039dc88c7dbd36e21fba81df1762&scene=21#wechat_redirect)**
- **[我把 ncnn 移植到 RISC-V 啦！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885118&idx=1&sn=653d0176375d47255518f9c9499f581f&scene=21#wechat_redirect)**
- **[Mobile GPU推理的选择：OpenGL vs. OpenCL](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886575&idx=1&sn=b8e5cc2d2ff57cf230a08a412f01910f&scene=21#wechat_redirect)**

:::

> :mdnice-bold-italic[作者搭建完整 int8 推理数据流，支持卷积层间直接传递 int8 张量，规避反复量化反量化带来的性能损耗；提出数据布局优化优先级高于硬件指令，依托 elempack pack4 打包方案优化访存效率，并针对普通卷积、深度卷积、全连接、矩阵乘法四大核心算子定制差异化实现方案]。文中解决 Winograd 变换中间值 int16 存储兼容问题，同时剖析 cooperative matrix 峰值算力与数据重排开销的矛盾，给出 int16 拆分为双 8bit 复用矩阵核心的精确计算方案。

作者梳理存储格式、舍入规则引发的数值精度bug，通过统一系数存储精度保证推理一致性；**实测表明MobileNet-int8性能可持平fp16，多数网络仍受调度、张量转换开销限制**。

::: block-1
文末总结核心工程思路：计算复杂度向Host侧转移、数值语义优先于硬件峰值算力，并指出 Winograd 并行度、矩阵核心自适应调度等后续优化方向。
:::

## 本文目录

- 导语
- 1. ncnn Vulkan backend 的基本边界
- 2. Vulkan int8 的数据流
- 3. layout 比指令更关键
- 4. 四个主算子的实现策略
  - Convolution
  - ConvolutionDepthWise
  - InnerProduct
  - Gemm
- 5. Shader 热路径：少做事，比多封装更重要
- 6. Winograd 与 int16 中间值
- 7. cooperative matrix：峰值算力和模型耗时之间隔着 layout
- 8. 正确性和 profiling 里暴露的问题

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/01.webp)

::: image-caption
是谁这么可爱啊？
:::

## 导语

> 这次 Vulkan int8 不是简单加几条`dot` 指令。**真正麻烦的是把 int8 tensor 接进 ncnn 现有的 Vulkan backend**：`VkMat` 、elempack 、packing layout、fp16 storage、shader specialization、pipeline cache、CPU 数值一致性，这些东西本来就互相影响。**int8 进来以后，每一个看起来局部正确的改动，都可能在另一个 GPU、另一个 shape hint、另一个 storage option 下变成 bug。**

我更愿意把这件事看成一个 backend 架构补齐，而不是单个 kernel 优化。算子包括`Convolution` 、`ConvolutionDepthWise` 、`InnerProduct` 、`Gemm` ，周边还要补`Padding` 、`Flatten` 、`Packing` 、`Quantize` 。目标也不是“能跑 int8 模型”这么低，而是：

- 结果要和 CPU int8 语义对齐；
- 不能只为某一张显卡写；
- 支持普通 Vulkan fallback；
- 有`integer dot product` 时用 packed int8 路径；
- 有`cooperative matrix` 时保留更高阶的优化空间；
- 代码风格要像 ncnn 原本的 Vulkan 代码，而不是另起一套 helper-heavy 框架。

下面按实现中真正有压力的地方展开。

## 1. ncnn Vulkan backend 的基本边界

> ncnn Vulkan 层通常按三段组织：

- `create_pipeline` ：根据 shape hint、Option、设备能力创建 pipeline；
- `upload_model` ：把模型参数上传到 GPU；
- `forward` ：运行时绑定`VkMat` ，发出`record_pipeline` 。

int8 也沿用这个边界，对主要算子补`create_pipeline_int8 / upload_model_int8 / forward_int8` 。这个形式看起来只是多了三个函数，但它解决了一个关键问题：fp 和 int8 的 storage decision 不应该混在一条越来越复杂的`forward` 里。

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/02.webp)

> `create_pipeline_int8` 里适合做这些事：

- 决定 direct / 1x1 / GEMM / Winograd / cooperative matrix；
- 决定 pack1 / pack4；
- 计算 shader 需要的 specialization constant；
- 在 host 上把 weight 重排成 shader 最舒服的 layout。

`upload_model_int8` 只负责上传已经准备好的数据。它不应该再做复杂 layout 变换。这个边界很重要，因为`upload_model` 从名字和 ncnn 现有风格看，就是 host 到 device 的 transfer；把重排逻辑塞进去，后面很容易出现同一份 weight 在不同 pipeline 下状态不一致的问题。

`forward_int8` 则应该尽量薄。运行时要做的事越少，越容易保证 pipeline 创建和 dispatch 对齐，也越不容易出现某个 shape hint 下没创建 pipeline、运行时却 dereference 空指针这种问题。

::: block-1
在代码组织上，`Convolution / ConvolutionDepthWise / InnerProduct / Gemm / Padding / Flatten` 的 int8 kernel 相关成员和函数用`#if NCNN_INT8` 包起来。`Packing` 、`Cast` 、`Quantize` 更像基础设施，可以无条件理解 int8 blob；否则主算子支持了 int8，图中间的 tensor 转换反而断掉。
:::

## 2. Vulkan int8 的数据流

> Vulkan int8 的基本路径是：

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/03.webp)

输入如果已经是 int8，就直接进 int8 kernel；如果是 fp32/fp16，则先走`Quantize` layer。kernel 内部用 int32 accumulate，最后有两种输出：

- 输出 fp32/fp16/bf16，给普通 fp 后续层使用；
- 通过 requantize 输出 int8，继续传给下一个 int8 convolution。

CPU int8 convolution 很重要的一点就是可以 requantize，让连续 convolution 之间传递 int8 tensor。**Vulkan 也必须有这个能力，否则每层都 int8 accumulate 后转回 fp，再下一层重新 quantize，模型级性能很容易被量化/反量化拖垮。**

这也是为什么`Padding` 要支持 int8。正确顺序不是 fp input 先 padding 再 quantize，而是：

```
fp input -> Quantize -> int8 Padding -> int8 Convolution
```

**这样和 CPU int8 路径更一致，convolution shader 也不需要藏一份隐式 padding 逻辑。** shader 里少一类边界判断，代码也更接近 fp Vulkan kernel。:mdnice-bold-italic[这里牺牲的是多一次 Padding dispatch，但换来的是职责更清楚，CPU/Vulkan 语义更容易对齐。]

## 3. layout 比指令更关键

> 刚开始做 int8 kernel 时，很容易把注意力放在`dotPacked4x8EXT` 或 cooperative matrix 上。但实际 profiling 下来，很多慢点并不是乘加吞吐不够，而是数据喂得不舒服。

ncnn 的 Vulkan tensor 通过 elempack 表示 channel packing：

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/04.webp)

pack4 下，四个 int8 lane 可以放进一个 32-bit word。shader 里按 packed int 读取，再走`integer dot product` ，这比标量`i8buffer_ld1` 一路拆 lane 要自然得多。

但是 packed load 只有在数据 layout 配合时才有意义。weight 原始布局通常是：

```
outch x inch x kernel
```

shader 热路径想要的是按 output tile、input tile、kernel tap 排好的连续数据。于是`create_pipeline_int8` 做 host 侧 reorder 和 zero padding：

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/05.webp)

这类重排代码看起来啰嗦，但动机很直接：weight 只在模型加载时处理一次，shader load 会在每个输出像素上执行。宁可 host 上多做一点，也不要让每个 invocation 反复算复杂 offset、判断尾部、临时 pack/unpack。

同样的思路也用在 bias、descale、top scale 上。把它们按 pack4 padding 好，shader 就可以用`vec4` /`sfpvec4` 或 packed 形式读取，少掉`base_ch + lane < outch` 这类保护。输出 store 仍然需要在边界处小心，但计算阶段可以尽量假设 host 已经准备了安全的 padding。

**这里有一个容易踩坑的地方：bias 和 descale 的精度**。int8 accumulate 后会乘 descale，再加 bias。fp16 storage 打开时，如果这些系数也被悄悄按 fp16 上传，结果可能和 CPU 漂。:mdnice-bold-italic[Convolution int8 最终把 bias 保持 fp32 上传，shader binding 也按 float/vec4 读，就是这类问题修出来的。]

**descale 这块单独值得说一下**。CPU int8 路径里，input scale、weight scale、top scale 的语义已经很明确。Vulkan shader 如果每次都在 hot path 里算：

```
1 / (bottom_scale * weight_scale)
```

不仅多 ALU，而且每个 shader 容易写出不同的零值处理和精度路径。**更干净的做法是在 host 侧预先算好`descale` ，按输出 channel 或 group 打包上传**。shader 里只做：

```
float sumfp32 = float(sum) * descale + bias;
```

如果需要 requantize，再乘`top_scale` 并走`float2int8` 。这也是为什么命名上统一用`descale` ，不要一会儿叫 scale，一会儿叫 inverse scale。量化代码最怕名字不准，后面 review 时很难判断一个系数到底是乘上去还是除下去。

有些 scale buffer 用`sfp` 在精度上可以接受，但 bias 和 descale 不应该随便下放到 fp16。尤其 Convolution / Winograd 这种 accumulate 后数值范围较大的路径，bias 读错 storage type 或 descale 被压低精度，都会表现成“只有某个 fp16 option + 某个 GPU + 某个 activation case 失败”的诡异问题。这里宁可多用一点带宽，也要先把 CPU 对齐守住。

## 4. 四个主算子的实现策略

### Convolution

> `Convolution_vulkan` 的 int8 路径主要分几类：

- 1x1s1d1；
- direct convolution；
- implicit-im2col / GEMM；
- 3x3s1d1 Winograd；
- cooperative matrix 版本的 1x1 / GEMM / Winograd GEMM。

**1x1 是最容易被 dot product 吃满的路径，但前提仍然是 weight 和 input 都按 packed int8 连续读取**。普通 convolution 的直接路径更复杂，kernel tap、stride、dilation、elempack、out_elempack 都会影响 load pattern。:mdnice-bold-italic[所以实现上尽量融合 pack1/pack4 情况，但不在 shader 中维持很多分散的 helper]。ncnn 原本的 fp shader 更偏直接展开，int8 shader 也应该接近这个风格。

**Winograd 是另一条线**。3x3s1d1 可以降低 GEMM 部分的乘法量，但 int8 input transform 后的数据范围会进 int16，不再是纯 int8。这个问题后面单独讲。

### ConvolutionDepthWise

> DepthWise 看起来计算量小，实际更容易被 memory 和 dispatch 开销控制。每个输出 channel 的工作很薄，shader 中多几次 branch 或多一次 pack/unpack，影响会比普通 convolution 明显。

这里的策略是**把 depthwise 和 grouped convolution 分清楚：**

- true depthwise：`group == channels == num_output` ；
- grouped convolution：每个 group 有多个 input/output channel。

grouped int8 shader 尽量融合 pack1、pack4、pack1to4、pack4to1，类似 convolution packed shader 的方向。weight、bias、descale 在 host 上按 group 内 output pack 补零，shader 不再到处检查尾部 channel。

::: block-1
**DepthWise 还有一个很实际的小优化：kernel tap 的 tail**。为了走 packed dot，容易把`maxk` 对齐到 4，然后`for (k += 4)` 一把处理四个 tap。

- 对 3x3 来说`maxk=9` ，直接对齐到 12 会白算 3 个 tap；
- 对 5x5 来说`maxk=25` ，也会有尾部浪费。

普通 convolution 里这点浪费可能被更大的 channel reduce 掩盖，DepthWise 每个 channel 的计算本来就薄，浪费会更明显。**因此更合理的是主循环处理`k + 3 < maxk` 的 packed dot，剩下 1/2/3 个 tap 走轻量 tail。这个优化不炫，但符合 DepthWise 的成本结构。**
:::

### InnerProduct

> InnerProduct 一开始慢得很明显，主要不是算术问题，而是 weight layout。naive 做法是一个 output 一个线程扫完整个 input，这对 GPU 不友好，也浪费了 pack4。

fp Vulkan InnerProduct 已经有`sum8 + reduce_sum8` 的思路：先做局部 sum，再 reduce。int8 也应当借这个结构，而不是重新发明一条完全不同的 shader。另一方面，batch/GEMM 形式的 InnerProduct 输入可能是 2D，需要走 gemm pipeline；普通 1D flatten 后的输入则走 sum/reduce。

还有一个小但重要的优化：`num_output % 4 == 0` 时直接输出 pack4。否则后面接`Softmax` 之前可能多一次 packing，单层看不大，模型级会积累。

这里不适合拆出很多`pack1to4 / pack4to1 / pack4` 的 shader 变体。变体多了以后，host 端 pipeline 选择和 shader code 都会膨胀，而且和现在 convolution packed int8 的融合方向不一致。更好的方式是让 shader 接受 pack1/pack4 输入，在权重 layout 上保证读取连续，在 store 处根据输出 elempack 处理差异。InnerProduct 的核心不是分更多 shader，而是把 weight 排成 reduce 时最容易读的形状。

### Gemm

> Gemm 是接口最复杂的一个，因为它不只是矩阵乘法：

- A/B 可能是 constant，也可能是 runtime blob；
- A/B 可能已经是 int8，也可能需要运行时 quantize；
- 有`alpha` 、`beta` ；
- 有 C broadcast；
- 还有 transpose 和 output transpose。

int8 Gemm 不能为了 shader layout 方便就改 CPU 语义。例如最终表达式应该保持：

```
alpha * A * B + beta * C
```

而不是把`alpha` 不小心套到 bias/C 上。类似这种 bug 很隐蔽，因为很多测试里`alpha=1` ，跑不出来。

constant A/B 的路径适合在 host 上预重排和补零；runtime A/B 则需要 quantize shader 生成 packed layout 和 descale。`cooperative matrix` 路径下还要按 coopmat M/N/K tile 排布，否则 matrix core 理论吞吐再高，也会被 load 和 shared memory 搬运抵消。

Gemm 的 runtime B 量化也比 A 麻烦。A 通常可以按 row 独立求 absmax 和 descale，B 如果是 runtime blob，还要考虑 transpose、constantB、broadcast C 和输出 transpose 的组合。这里为了保持 CPU 语义，宁可多几个量化辅助 pipeline，也不能把`C` 的 broadcast 或`alpha/beta` 语义塞进一个看起来更快但行为不同的公式里。Gemm 是基础算子，一旦语义偏了，后面很难从模型输出定位。

## 5. Shader 热路径：少做事，比多封装更重要

> int8 shader 里可以加很多 helper，把 load、pack、unpack、scale 都包起来。但 ncnn 的 shader 风格不是这样。现有 fp shader 往往很直接：load、算、store，局部展开，少抽象。这样读起来不一定“现代”，但很利于看 hot path，也容易推测编译器会生成什么。

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/06.webp)

这次 int8 shader 里比较关键的几条原则：

- **第一，尽量使用 packed load/store。`sint8vec4` 在很多路径里等价于 packed int8 word，用`i8buffer_ld4 / st4 / sm4` 这类方式更接近硬件需要的数据形态**。能一次拿四个 lane，就不要标量拿四次再 pack。
- **第二，int8 storage 和 packed fallback 都要能工作**。有`VK_KHR_8bit_storage` 和`VK_KHR_shader_float16_int8` 时，可以在 shader 里直接用 int8 storage 类型；没有 native int8 storage 时，就用 32-bit int 承载四个 int8 lane。对 hot path 来说，后者并不是低级 fallback，很多时候反而正好贴合`integer dot product` 的 packed 输入形式。
- **第三，rounding 要和 CPU 对齐**。GLSL 的`round()` 在半数处的行为不能直接拿来当 CPU`float2int8` 。因此 shader 侧统一使用`float2int8` /`float2int8vec4` ，先做 half-away-from-zero，再 clamp 到 int8 对称量化范围。
- **第四，int16 packed / int16 storage 是 Winograd 必需的基础设施**。int8 transform 后的中间值不能硬塞回 int8。支持 int16 storage 的设备走直接存储；否则用 int16 packed，把两个 int16 合进一个 32-bit word。这里补`i16buffer_ld/st` 系列不是为了抽象好看，而是为了让 shader 能像 fp16 packed 那样清楚地表达数据存取。
- **第五，shared memory 不是免费优化**。桌面 GPU 上 shared memory tiling 很容易有效，但移动 GPU 上 shared memory 资源很贵。int8 kernel 必须同时有 shared-memory 路径和 no-shared-memory 路径，不能假设某一种永远最好。
- **第六，tail 和边界尽量前移**。weight、bias、descale、top scale 能 host padding 就 host padding。shader 输出 store 处保留必要边界判断，但计算阶段尽量不让每个 lane 都背着`if (ch + i < outch)` 。
- **第七，fallback 路径也要像一等公民**。不是所有 GPU 都有`VK_KHR_shader_integer_dot_product` ，更不是所有 GPU 都有高效的 cooperative matrix。fallback 如果写得太差，兼容性只是名义上的。

::: block-1
这里有个风格问题也很关键：宏和 helper 只应该服务于 storage 差异，例如`i8buffer_ld4` 、`i16buffer_st2` 这类基础设施；不要把每个数学片段都包成 helper。ncnn 原有 fp shader 的可读性来自“热路径摊开在眼前”，int8 也应该保持这个感觉。
:::

## 6. Winograd 与 int16 中间值

> Winograd int8 的麻烦在于 transform。输入是 int8，weight 是 int8，但 Winograd transform 后的中间值可能到 int16 范围。

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/07.webp)

普通 dot-product 路径可以让 transform 输出 int16。设备支持`VK_KHR_8bit_storage` /`VK_KHR_shader_float16_int8` / int16 storage 时，可以直接走 int16 storage；老设备上则用 int16 packed，把两个 int16 放进一个 32-bit word。这个设计类似 fp16 packed 的思路：不要求所有设备都有 native int16 storage，但仍然可以节省显存和带宽。

Winograd output transform 也要小心整数除法和 rounding。负数除法、`round()` 、scale 为 0，这些地方如果不同 shader 各写一套，很容易出现 CPU/Vulkan 差一个 1 的漂移。对量化模型来说，这种小漂移会在某些网络结构里放大，而且很难从最终输出反推原因。

## 7. cooperative matrix：峰值算力和模型耗时之间隔着 layout

> `cooperative matrix` 是 int8 优化里很有吸引力的部分。vkpeak 里 int8 CM 的峰值可能比 fp16 高很多，NVIDIA/AMD 桌面卡上尤其明显。但这不代表把所有 GEMM 都换成 CM 就一定快。

CM 路径有额外成本：

- A/B 要按 coopmat tile 重排；
- K/N/M 维度要 padding；
- shared memory load 要符合 bank padding；
- 小 shape 下 transform 和 dispatch 开销占比高；
- Winograd 还要处理 int16 中间值。

对于普通 Gemm、1x1 convolution、convolution-GEMM，int8 CM 可以直接使用 int8 x int8 -> int32。Winograd 更特殊，因为 A 和 B 的 transform 结果可能是 int16。为了不损失精度，可以做 exact split：

![文章插图](/articles/2026-06-18-ncnn-vulkan-int8-c62bfc58/images/08.webp)

把 int16 拆成 low8 和 high8：

```
int16 = low8 + (high8 << 8)
```

如果 A 和 B 都是 int16，则：

```
A * B =  lowA * lowB  + (highA * lowB + lowA * highB) << 8  + (highA * highB) << 16
```

这样四次都是 int8 x int8 -> int32 cooperative matrix，最后在 int32 accumulator 中重建，结果是 exact 的。代价也很明确：更多 MMA、更多数据、更多 layout 压力。这个方案理论上能利用 int8 matrix core 的高峰值，但实际是否赚钱，要看 transform 占比、channel 数、tile 利用率和驱动实现。

**这也是这轮优化里最重要的教训之一：峰值算力只是上限，不是模型耗时**。VGG/ResNet 这类 3x3 重的模型在卷积 kernel 层面更可能吃到 int8 和 CM 的收益，但端到端仍不保证超过 fp16；MobileNet 这种 depthwise/pointwise/小 kernel 多的模型，很多时候瓶颈在 dispatch、layout、transform、后处理，而不是乘加吞吐。

## 8. 正确性和 profiling 里暴露的问题

> 这类 backend 工作，正确性问题经常不是大错，而是某个 storage option 下读错类型。

一个实际例子是 Convolution int8 bias。shader 如果把 bias binding 写成`sfp` ，在`use_fp16_storage` 或`use_fp16_packed` 打开时会按 half 读。但 host 侧 Winograd int8 bias 是 fp32 上传的，结果直接错。最后修法不是“这里特判 fp16”，而是统一让 Convolution int8 bias 按 float/vec4 读，host 用 fp32 option 上传。

descale 也类似。int8 accumulate 后的结果非常依赖 descale 精度，把 descale 悄悄压成 fp16 可能在某些模型上过不了 CPU 对齐。因此 Gemm、ConvolutionDepthWise 等路径里 descale 上传要显式避开 fp16/bf16 storage。

还有一类问题来自 shape hint。ncnn 的`create_pipeline` 会根据 shape hint 决定创建哪些 pipeline，`forward` 默认运行时 shape 和 hint 一致。这个前提成立时，代码可以很干净；如果为了“防御”到处创建所有 pipeline，反而会增加 shader compile 成本，也不像 ncnn 现有风格。真正需要保证的是 create 和 forward 的条件一致，不要 create 时认为不会用，forward 时又去 dispatch。

1D`Convolution` 是另一个兼容性点。:mdnice-bold-italic[历史上有些模型会用 1x1 的 1D Convolution 表达 InnerProduct，CPU float 路径保留了这种兼容行为。Vulkan int8 不应该继续扩展这个历史包袱；这类模型应该迁移到`InnerProduct`] 。但 CPU int8 路径也不能因为 Vulkan 不支持就直接打断旧模型，**比较合理的处理是保留 CPU fallback 行为，同时给出 warning，提醒用户不要继续依赖 1D Convolution 兼容路径**。:mdnice-bold-italic[这里体现的是另一个取舍：新 backend 不再扩大旧接口语义，但也不无意破坏非 Vulkan 用户。]

> 性能上，一开始 int8 全面慢于 fp16 并不奇怪。**量化、padding、packing、transform、descale、shader launch 都是真成本。后面逐步有收益，主要靠这些点：**

- 连续 convolution 之间保留 int8 tensor；
- pack4 输出减少后续 packing；
- host 侧重排 weight，让 shader load 连续；
- descale/bias/top scale padding 到 vec4；
- InnerProduct 参考 fp 的 sum/reduce，而不是 naive per-output；
- Winograd/GEMM/1x1 上选择性引入 cooperative matrix。

::: block-1
RTX3060 上的 benchncnn 结果反而更能说明这个问题：`mobilenet_int8` 可以做到和 fp16 基本持平，但其他 int8 模型整体仍慢于对应 fp16 模型。**也就是说，局部 kernel 的 layout、pack4、dot product、requantize 优化确实能拉回不少开销，但还没有把端到端模型都推到 fp16 前面**。Kabylake iGPU 这类没有硬件 int8 dot product 的设备则更直接，int8 fallback 比 fp16 慢是合理结果。:mdnice-bold-italic[这说明 int8 后端不能只看一张卡、一个峰值指标，也不能只看单个 kernel。]
:::

工程上最后留下的判断是：

- int8 Vulkan 不是单 kernel 优化，而是 backend 数据流；
- layout 比单条指令更关键；
- shader hot path 要简单，复杂度尽量前移到 host；
- CPU 语义必须先对齐，再谈性能；
- cooperative matrix 是重要方向，但不能盲目压过所有路径；
- 代码风格要融进 ncnn 现有 Vulkan backend，而不是把一套外部框架搬进来。

后续还有不少可以继续做的事。

- Winograd input/output transform 的并行度仍然偏低，特别是通道数不大时，transform 开销会吃掉 GEMM 的收益；
- cooperative matrix 的选择策略需要更多 GPU 实测，不适合只按扩展支持与否一刀切；
- mobile GPU 上 shared memory 压力还要继续压；
- benchmark model 的 shape hint 也应该继续补全。

遇到 driver 问题时，最好提炼成最小 Vulkan/SPIR-V 复现，而不是在 ncnn 里堆厂商 hack。

> 如果只总结一句话，就是：这件事不是“打开 int8 指令”，而是让 quantize、layout、packing、shader、pipeline 和 fallback 在 ncnn 的架构里同时成立。

Guided by nihui, written by GPT 5.5 xhigh and GPT image 1.5, cover image generated by GPT image 2

参考资料

[1] PR#6751 vulkan int8 kernels: *https://github.com/Tencent/ncnn/pull/6751*

::: related-title
**相关推荐**
:::

::: related-links
- **[ncnn的设计理念和软件工程](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885535&idx=1&sn=33b9039dc88c7dbd36e21fba81df1762&scene=21#wechat_redirect)**
- **[我把 ncnn 移植到 RISC-V 啦！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885118&idx=1&sn=653d0176375d47255518f9c9499f581f&scene=21#wechat_redirect)**
- **[Mobile GPU推理的选择：OpenGL vs. OpenCL](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886575&idx=1&sn=b8e5cc2d2ff57cf230a08a412f01910f&scene=21#wechat_redirect)**

:::
