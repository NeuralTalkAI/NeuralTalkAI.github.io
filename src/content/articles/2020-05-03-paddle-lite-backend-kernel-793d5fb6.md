> 导读
>
> ：
>
> Paddle Lite是Paddle Mobile和Anakin的推理框架继任者。
>
> 定位安卓/iOS移动端，以及X86端在内的多场景高性能预测
>
> ，兼容支持ONNX、TensorFlow、Caffe等模型的部署。
>
> 本文作为将描述Paddle Lite在模型转换过程（模型转换opt工具）中，
>
> 静态kernel选择的策略
>
> 以及一些思考。目录如下：

1. 用于kernel注册的Place：同一个kernel根据Place的不同可以有多种实现（注册）；
2. 用于模型执行时kernel选择的候选 `valid_places`；
3. 静态kernel选择（static kernel pick）：候选kernel们的place与用户valid_places的笛卡尔积；
4. 思考；
5. 补充。

# Paddle Lite（以下简称plite）底层kernel选择上会考虑候选Place，**Place由设备Target，精度Precision，数据排布（DataLayout）构成** （还有一个常被忽略的 `device_id`，用来区分多gpu的id。不过在选择策略中没出现可忽略）。

```
/* Place specifies the execution context of a Kernel or input/output for a * kernel. It is used to make the analysis of the MIR more clear and accurate. */struct LITE_API Place {  TargetType target{TARGET(kUnk)};  PrecisionType precision{PRECISION(kUnk)};  DataLayoutType layout{DATALAYOUT(kUnk)};  int16_t device{0};  // device ID  Place() = default;  Place(TargetType target,        PrecisionType precision = PRECISION(kFloat),        DataLayoutType layout = DATALAYOUT(kNCHW),        int16_t device = 0)      : target(target), precision(precision), layout(layout), device(device) {}}
```

## 1. 用于kernel注册的Place：同一个kernel根据Place的不同可以有多种实现（注册）

同一个kernel如conv2d，可能会有不同设备的实现如armcpu/opencl/x86/cuda等。在kernel注册时，需要指定kernel的Place信息。

kernel的Place用于kernel的注册，以区分唯一性。比方实现了一个arm cpu以NCHW数据排布并以fp32计算的conv2d kernel，那么其注册时候就会以conv2d，kARM，kFloat，kNCHW，def，用来区分这个kernel的唯一性。下面是conv2d的多种不冲突的kernel注册形式：

```
conv2d, kARM, kInt8, kNCHW, defconv2d, kARM, kFloat, kNCHW, defconv2d, kARM, kFloat, kNHWC, defconv2d, kOpenCL, kFloat, kNCHW, defconv2d, kOpenCL, kFloat, kImageDefault, def
```

PS：你问我def是干啥的，好像依稀记得def默认大概也是用来区分kernel注册时唯一性起名的一部分，作为补充。

## 2. 用于模型执行时kernel选择的候选valid_places

那模型执行的时候，遇到conv2d是选择opencl还是arm（cpu）来执行呢？就比方上面5个conv2d，模型执行时候选哪个？

这个涉及同一个op算子或称为layer层，在对应不同kernel注册的Place（上面5个conv2d kernel），和候选的执行 `valid_places`的比较打分排序。

用户需要（注：现在用户已经不需要了，有预设）给出候选的模型执行 `valid_places`。例如下面是arm cpu跑Float模型的预设 `valid_places`：

```
  std::vector<Place> valid_places({      Place{TARGET(kARM), PRECISION(kFloat)},  });
```

再例如，下面是opencl跑模型时的预设 `valid_places`：

```
  std::vector<Place> valid_places({      Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},      Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},      Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},      TARGET(kARM),  // enable kARM CPU kernel when no opencl kernel  });
```

用于模型执行候选kernel的 `valid_places`，其中的顺序也很重要，越靠前，同一op的候选kernel的选择，就越倾向valid_places顺序靠前的Place，即权重系数越大（见后文KernelGrade方法中的 `weight`计算）。后面会再说到。

## 3. 静态kernel选择（static kernel pick）：候选kernel们的place与用户valid_places的笛卡尔积

注：这里，代码即就走到与硬件直接相关的 `static_kernel_pick_pass`，前面已经完成了conv-bn/conv-relu/elem-act等graph级别的与硬件设备无关的大粒度的op融合策略。

和动态地对同一个op/layer，选择多种kernel实现试跑的方法不同。plite对同一个op/layer的不同kernel实现（绑定不同的Place），与模型选择的 `valid_places`，静态地选择要执行的kernel。那选择策略又是怎样的呢？

plite有一个基于二者Place做打分策略的方法，实现的代码对应下面两个文件：

1. `./lite/core/mir/static_kernel_pick_pass.cc`：**全图遍历选择kernel**。遍历模型中的计算节点（graph概念），并针对同一位置节点的不同候选kernel的place，与用户传入的 `valid_places`的候选place，两两打分（笛卡尔积），选择分数最高的kernel；
2. `./lite/core/mir/static_kernel_pick_pass.h`：**KernelGrade对kernel的不同Place打分**。针对确定的某个kernel，与用户传入的 `valid_places`候选依次打分。打分策略，主要基于Place信息中包含的设备（Target），精度（Precision），数据排布（DataLayout）等信息。

下面描述一下这两个过程：

### 3.1 全图遍历选择kernel

上面第一个过程代码化简如下，主要流程见下面注释：

```
  // lite/core/mir/static_kernel_pick_pass.cc  // 1. 依次遍历模型graph节点  for (auto& node : graph->mutable_nodes()) {    if (!node.IsStmt()) continue; // 跳过非计算节点    auto& instruct = node.AsStmt();    // 获取所有该节点的输入和输出的tensor精度，实现略    std::unordered_map<std::string, PrecisionType> in_precision_types;    std::unordered_map<std::string, PrecisionType> out_precision_types;    // 获取该层op的不同kernel候选实现：instruct.kernels()    //     比方该层是conv2d，那么instruct.kernels()方法，    //     就可获取到所有编译进去的conv2d的不同实现。    // 2. 依次(for)对不同kernel实现打分(KernelGrade)，    //      KernelGrade用来找出该kernel实现的最佳Place，    //      及最佳Place下的kernel得分。    std::vector<std::pair<float, std::unique_ptr<KernelBase>>> scored;    for (auto&& kernel : instruct.kernels()) {      float score = KernelGrade(instruct,                                *kernel,                                graph->valid_places(),                                in_precision_types,                                out_precision_types,                                instruct.op_info()->input_names(),                                instruct.op_info()->output_names());      // 3. 记录每种kernel实现在最佳Place下的最高分值      scored.emplace_back(score, std::move(kernel));    }    // 4. 对打分结果scored排序，clear清空候选kernel列表    //       重置候选kernel列表为分数最高的那一个kernel，    //       即最终选中要执行的Kernel    std::sort(scored.begin(), scored.end(), KernelScoreCmp);    instruct.kernels().clear();    instruct.kernels().emplace_back(std::move(scored.front().second));  }
```

### 3.2 KernelGrade：对kernel的不同Place打分

可以看到在**全图遍历选择kernel** 的过程中，KernelGrade起了至关重要的作用：该方法找出当前kernel下的最佳Place（方法内会**对用户传入的 `valid_places`遍历计算打分: `final_score=score*weight`**），及最佳Place下的该kernel得分。

公式中 `weight`就是 `valid_places`中的次序，越靠前的Place， `weight`越大。比方我们希望模型以CPU的lNCHW的layout来跑，其中的 `valid_places`第一个必须是Place{kARM, kFloat, kNCHW}，假设第二个是 `Place{kARM,kFloat,kNHWC}`，除了layout其他都和第一个Place一样，**那么，在两个Place都有对应kernel注册且实现过的前提下（候选kernel里二者都有），因NCHW是第一位，则NCHW对应的Place的weight就更大，包含NCHW的Place最终被选中为winner_place概率会大，包含NCHW的Place的kernel被选中的概率也会更大**。

kernel对place打分的过程，有5个阶段，我将代码简化如下：

```
  // lite/core/mir/static_kernel_pick_pass.h  size_t KernelGrade(      const mir::Node::Stmt& instruct,      const KernelBase& kernel,      const vector<Place>& valid_places,      const unordered_map<std::string, PrecisionType>& in_node_precisons,      const unordered_map<std::string, PrecisionType>& out_node_precisons) {    float final_score_for_winner_place{-1.};    const int kMax = numeric_limits<int>::max();    size_t place_size = valid_places.size();    for (size_t pidx = 0; pidx < place_size; ++pidx) {      const auto& place = valid_places[pidx];      float weight = static_cast<float>(place_size - pidx) / place_size;      size_t place_score{0};      if (place.target == kernel.target())        place_score += kMax / KernelPickFactor::Factor::TargetFirst;      if (place.precision == kernel.precision())        place_score += kMax / KernelPickFactor::Factor::PrecisionFirst;      if (place.layout == kernel.layout())        place_score += kMax / KernelPickFactor::Factor::DataLayoutFirst;      if ((in_node_precisons == kernel_registered_in_tensor_precisions) &&            out_node_precisons == kernel_registered_out_tensor_precisions))        place_score *= 2;      if (weight * place_score > final_score_for_winner_place) {        final_score_for_winner_place = weight * place_score;        winner_place = place;      }    }    return final_score_for_winner_place;  }
```

这5个阶段对应place的设备/place精度/place数据排布/输入输出精度检查/place排位系数，前3个在计算时有对应系数，来看看代码中的设定以及思考：

```
// /lite/core/types.h// 系数在实际计算中转为分母class KernelPickFactor { public:  using value_type = unsigned char;  enum class Factor : int {    // The following factors are sorted by priority.    TargetFirst = 1,    PrecisionFirst = 1 << 1,    DataLayoutFirst = 1 << 2,    DeviceFirst = 1 << 3,  };
```

1. 设备target（系数为1）：相比Place中的其他两个，设备系数排第一，因为来回的数据搬运开销极大。若模型中conv都是gpu计算，中间有些层的实现是cpu的，且在不考虑zero copy/ION等前提下，来回的数据拷贝带来的性能下降就很明显；
2. 精度precision（系数为1/4）：其实精度还有数据排布哪个排在第二位更好，我也不知道，以opencl来说，数据排布layout为cl::image可利用L1 cache（一般）性能比cl::buffer要好，精度fp16比fp32性能也要好不少，所以我很难说哪个系数更大好，就从opencl来说可能二者打分的系数可以一样吧。但这里代码写的是精度的重要性系数（比layout）更大。此外，arm cpu的int8比较特殊，详细可阅读代码；
3. 数据排布datalayout（系数为1/8）：同上。访存的优化是重要且必要的，cpu也有NHWC，opencl也有为了性能对乘法的两个矩阵做转置和重排；
4. kernel注册的输入输出的tensor精度与该graph中当前op的输入输出精度是否匹配。全部匹配就score *= 2。这个检查其实是看当前graph中的节点精度和kernel注册时tensor的精度是否一致，其实不只精度，似乎layout和target也可以做这个判断（目前没有）；
5. 分数乘以当前place在valid_places中的排位系数。这个前面已经说过，排在越靠前的place，对应kernel被选中的 概率就越大。

以上，便是kernel静态选择的整个过程。

## 4. 思考

其实可以看到：

1. plite的kernel选择前先做硬件无关的graph层op粒度的融合操作，与硬件无关；
2. 在之后，是硬件相关基于图的信息与硬件信息的静态kernel选择，选择的参考有Place{target, precision, layout}等，从而确定要执行的backends的对应kernel，其中没有参考如卷积核的大小，输入的大小等信息。或者说，该过程是模型维度和op具体信息无关的，选择的依据粒度较大；
3. `static_pick_kernel_pass`是模型转换为plite格式的过程中一个pass，在之后的pass里应该还有更大的操作空间。比方结合试跑，结合模型更细粒度的信息做一些更细粒度的kernel选择，或者加载很多硬件试跑后的性能数据等。

## 5. 补充

### 1. 细粒度的kernel选择比方cpu或者opencl的cl kernel是什么阶段选择的呢？

答：plite在模型转换过程（非运行期，离线）会做op大粒度基于用户偏好Place（设备，精度，数据排布）的lite kernel选择。模型和设备绑定。这也是模型转换工具opt所做的事情。

但细粒度如conv3x3s2p1具体要执行的kernel，会在运行期lite kernel第一次执行的时候基于op具体信息做选择。此外，如果是动态shape，即当前层op这次输入与下次不一样，也会做这个具体kernel的重新选择。

以opencl为例，选择具体要执行的cl kernel的时候，也是lite kernel里做，像是lws等与硬件相关的信息（和性能直接相关）也会在这个地方确定。要是想做针对opencl这一个backend的模型自动化调优需要在lite kernel这个粒度来做。而且也仅限当前kernel这个Place，前面我们说过place包含三个信息target/precision/layout，对于opencl有两种layout，kNCHW和kImageDefault，对应cl::Buffer和cl::Image2D。但前面说了lite kernel的layout已经在静态选kernel时确定了，一次只能调一种。

### 2. 这样模型对所有backend的kernel搜索就不好做

答：先说一下目前plite还不支持基于试跑的最佳kernel搜索。拿opencl来说，因为模型优化过程（opt）的staticpickkernel，与运行时具体cl kernel，有这两个不同粒度的kernel选择阶段。

虽然目前plite不支持kernel的搜索，不过可以想一下。如果要针对backend来做最佳性能的kernel搜索的话，就要做试跑，可能有2种方式：

1. 两个阶段打通，在opt阶段的本文所说的staticpickkernel的过程中，与具体的kernel选择绑定，即在这个过程也能拿到conv的kernel size，input shape等信息。但这样虽然两个阶段的kernel选择打通，但是二阶段的具体kernel判断需要再写一遍，维护上有一定成本；
2. 两阶段分开做kernel选择，即每个阶段相对于局部的最优，从而达到相对全局的(次)最优。Ps：本来我这里写了一堆，但是想了想意义不大。其实我们的目的是找一个模型在所有不同target/precision/layout的kernel实现上排列组合这个模型下的最佳性能。但其实本质上staticpickkernel已经考虑了backend不同带来的差异，端侧对性能的极致要求，可能不同backend下的kernel组合出的一个模型，也会带来性能不稳定，在端侧会非常不友好。而且还有拷贝带来的性能损耗。
