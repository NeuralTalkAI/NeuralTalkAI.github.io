关键词：**Ascend C 算子开发**、:mdnice-bold-italic[Tiling 分块策略]、Host-Kernel 双侧编程、**CANNBot 智能体**、:mdnice-bold-italic[NPU 异构计算]

> 做过 CUDA 开发的人大概都有类似的体验：第一次写 kernel 函数时，那种"我到底该怎么把数据搬进共享内存"的茫然。换到昇腾 NPU 上，这种茫然可能还会加上一层——"Ascend C 是什么？Tiling 又是什么？Host 侧和 Kernel 侧为什么要分开写？ "

![文章插图](/articles/2026-06-07-cann-learning-hub-skill-npu-446b1552/images/01.webp)

- :mdnice-bold-italic[CANN 学习中心以 Notebook 教程覆盖 Ascend C 算子开发与推理优化全栈技术，创新集成 CANNBot 智能体打通学练赛闭环，助 NPU 开发者零门槛迈向精通。]
- https://github.com/hicann/cann-learning-hub
- 本文 8000 字，阅读 50 分钟，播客 30 分钟

::: related-title
**相关推荐**
:::

::: related-links
- **[从 MegaKernel 视角重新定义 NPU 上的线性注意力极限！PTO-ISA 驱动的昇腾 NPU Gated DeltaNet MegaKernel 浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901380&idx=1&sn=4ae6987a6ba87626737c46cb4be9458e&scene=21#wechat_redirect)**
- **[鹏城联合华为昇腾、中山大学提出AscendKernelGen：NPU 算子闭环生成与评估，复杂L2 Kernel编译成功率95.5%，正确率64.3](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896566&idx=1&sn=d920e9f8feed38e4734d18942d83897a&scene=21#wechat_redirect)**
- **[SGEMM-cube：昇腾910A用FP16 Cube单元模拟 FP32 GEMM 实现 22 位尾数精度恢复，77%理论峰值！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890912&idx=1&sn=9bd135756aa33bd195fa42099bd6e840&scene=21#wechat_redirect)**

:::

> 华为 CANN 团队显然意识到了这个问题。2026 年 3 月，他们在 GitCode 上开源了 `cann-learning-hub`，一个定位为"官方学习中心"的代码仓库。但如果你以为它只是一堆文档加几个 Hello World，那就太小看它了。

这个仓库做了一件非常有野心的事情：**用一套 Notebook 交互式教程 + CANNBot 智能体 + 在线评测平台，把"学一个算子怎么写"到"参加算子竞赛拿到排名"的全链路打通了。** 八章 Ascend C 算子开发系列教程从基础概念讲到性能优化，二十余篇技术博客覆盖推理、训练、算子三大方向，两大 CANNBot 技能甚至可以帮你自动生成算子工程、一键提交竞赛代码。

::: block-1
更关键的是，所有教程都能在云端 Notebook 环境中直接运行——真正做到了"边看边写边调"。
:::

## 本文目录

- 快速上手
- 一、仓库全景：一个仓库里装了些什么
  - 1.1 目录结构与功能定位
  - 1.2 "学·练·赛"闭环生态
- 二、Ascend C 算子开发的双侧编程模型
  - 2.1 为什么要分 Host 侧和 Kernel 侧
  - 2.2 Kernel 侧：模板类与多数据类型支持
  - 2.3 Float16 精度的"升降级"策略
- 三、Tiling 策略：把大象装进冰箱的分块艺术
  - 3.1 什么是 Tiling，为什么它如此重要
  - 3.2 TilingKey：编译时决策的模板化编程
- 四、算子注册：连接 Host 与 Kernel 的"胶水层"
  - 4.1 OpDef 注册机制
  - 4.2 ValueDepend 机制
- 五、CANNBot 技能系统：AI 驱动的算子开发
  - 5.1 两大核心技能
  - 5.2 竞赛提交的安全设计
  - 5.3 泛化算子的设计哲学
- 六、编译构建：从源码到可部署安装包
  - 6.1 一键编译脚本
  - 6.2 工程生成的自动化
- 七、教程体系与技术博客：从入门到前沿
  - 7.1 八章 Ascend C 算子开发教程
  - 7.2 HiXL 单边通信教程
  - 7.3 二十五篇技术博客
- 八、社区共建与生态扩展
  - 8.1 社区贡献通道
  - 8.2 未来方向
- 写在最后

![文章插图](/articles/2026-06-07-cann-learning-hub-skill-npu-446b1552/images/02.webp)

## 快速上手

> 对于迫不及待想动手试试的开发者，最快的体验路径只需三步：

**1. 在线体验自定义算子（无需本地安装）：**

直接点击仓库 README 中 10 分钟体验自定义算子[1] 旁的"在线体验"链接，即可在云端 Notebook 环境中完成 Ascend C 算子的编写、编译和运行全流程。

**2. 本地克隆仓库：**

```
git clone https://github.com/hicann/cann-learning-hub.gitcd cann-learning-hub
```

**3. 使用 CANNBot 自动生成算子：**

```
cd cann-learning-hub && opencode# 在对话界面输入：帮我生成 erf 算子
```

- 更多环境配置细节参考 `docs/gitcode_env_experience_guide`[2]；
- 教程版本基于 CANN 8.5.0 验证，具体版本对应关系参考 release 仓库[3]。

## 一、仓库全景：一个仓库里装了些什么

### 1.1 目录结构与功能定位

打开 `cann-learning-hub` 的根目录，你会看到一个组织得相当清晰的六层结构：

```
├── quick_start/          # 快速入门（10 分钟级别的体验）├── tutorials/            # 系列教程（八章 Ascend C + HiXL + MC2）├── reference_practice/   # 参考实践（模型推理优化等端到端案例）├── blogs/                # 技术博客（25 篇，覆盖算子/推理/训练）├── skills/               # CANNBot 技能（智能体驱动的自动化工具）└── contrib/              # 社区贡献（第三方教程，如推荐系统）
```

这个结构设计的逻辑其实很直觉——它模拟了一个开发者的真实学习曲线。

- 你从 `quick_start` 开始，花十分钟跑通第一个自定义算子或第一个算子 API 调用；
- 然后进入 `tutorials` 系统学习，从基础概念一路推进到性能优化；
- 期间穿插阅读 `blogs` 里的前沿技术实践；
- 等到有一定基础了，用 `skills` 里的 CANNBot 工具辅助开发；
- 最后还可以通过 `contrib` 目录把自己的经验回馈社区。

### 1.2 "学·练·赛"闭环生态

> 但仓库本身只是冰山一角。`cann-learning-hub` 真正有意思的地方在于它不是孤立的，它是一个三方联动的生态系统：

| 环节 | 平台 | 作用 |
| --- | --- | --- |
| 📖 学 | cann-learning-hub（本仓库） | 教程 + 博客 + 实践 |
| 🏋️ 练 | CANNJudge + CANNLab | 在线刷题 + NPU 云环境 |
| 🏆 赛 | CANN 大赛专区 | 算子天梯赛、校园赛 |

你在仓库里学会的东西，可以立即到 CANNJudge[4] 上去做练习题验证——:mdnice-bold-italic[而且提交和评测的过程，甚至可以由仓库里的 CANNBot 技能自动完成。这种"学完就能练，练完就能比"的设计，在国内 AI 芯片的开发者生态中确实不多见。]

## 二、Ascend C 算子开发的双侧编程模型

> 这是整个仓库的技术核心。如果说学习 CANN 就像学一门新语言，那么 Ascend C 算子开发的八章系列教程就是它的语法教科书。

### 2.1 为什么要分 Host 侧和 Kernel 侧

在 NVIDIA 的 CUDA 编程模型中，你写一个 `__global__` 函数就是 kernel，CPU 端直接通过 `<<<grid, block>>>` 语法启动。但昇腾 NPU 的架构不太一样。CANN 的算子开发被明确拆分成两个世界：

- **Host 侧**：运行在 CPU 上，负责三件事——形状推导（InferShape）、数据类型推导（InferDataType）和 **Tiling 参数计算**。
- **Kernel 侧**：运行在 NPU 的 AI Core 上，根据 Host 侧传下来的 Tiling 参数，执行真正的计算逻辑。

这种分离不是随意的设计，而是由昇腾 NPU 的硬件特性决定的。

::: block-1
NPU 的 AI Core 没有操作系统，不支持动态内存分配，也没有 RTTI（运行时类型信息）。所有关于"数据怎么切分""用多少个核""每个核处理多少元素"的决策，都必须在 Host 端提前算好，然后以一个结构体的形式传给 Kernel。
:::

> 仓库中的 Host 侧代码模板清楚地展示了这个流程：

```
// 来源：skills/ascendc-ops-project/templates/host_template.cppnamespace optiling {static ge::graphStatus TilingFunc(gert::TilingContext* context){    // 1. 获取输入 shape    const gert::StorageShape* x_shape = context->GetInputShape(0);    auto shape_dims = x_shape->GetStorageShape();    int32_t rank = shape_dims.GetDimNum();    // 2. 计算总元素数    uint32_t totalLength = 1;    for (int i = 0; i < rank; i++) {        totalLength *= shape_dims.GetDim(i);    }    // 3. 获取数据类型    auto input_dtype = context->GetInputDesc(0)->GetDataType();    // 4. 设置 Tiling 数据（传给 Kernel）    tiling->totalLength = totalLength;    tiling->dtype = static_cast<int32_t>(input_dtype);    // 5. 设置核数    uint32_t coreNum = platform.GetCoreNum();    context->SetBlockDim(blockDim);    tiling->blockDim = blockDim;    return ge::GRAPH_SUCCESS;}}
```

注意这段代码里的 `TilingFunc`——它的返回值是一个 `ge::graphStatus`，本质上是图引擎（Graph Engine）的一部分。也就是说，**每一个 Ascend C 算子的 Tiling 计算，都是图编译过程中的一环**。这和 CUDA 的"写完 kernel 直接调用"在心智模型上有本质区别。

### 2.2 Kernel 侧：模板类与多数据类型支持

> Kernel 侧的代码同样有讲究。由于 NPU 不支持虚函数和运行时多态，算子对多种数据类型的支持只能通过 **C++ 模板** + **dtype 分发** 来实现。仓库模板给出了一个非常标准的写法：

```
// 来源：skills/ascendc-ops-project/templates/kernel_template.cpptemplate<typename T>class KernelOp {public:    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,                                uint32_t totalLength, uint32_t blockDim)    {        // 计算当前核的工作范围        blockIdx_ = AscendC::GetBlockIdx();        xGm.SetGlobalBuffer((__gm__ T *)x, totalLength);        yGm.SetGlobalBuffer((__gm__ T *)y, totalLength);    }    __aicore__ inline void Process()    {        // 多核切分：每个核算自己的那一段        uint32_t lengthPerCore = (totalLength + blockDim - 1) / blockDim;        uint32_t start = blockIdx_ * lengthPerCore;        uint32_t end = (start + lengthPerCore < totalLength) ?                       (start + lengthPerCore) : totalLength;        for (uint32_t i = start; i < end; i++) {            T value = xGm.GetValue(i);            T result = value;  // 替换为实际计算逻辑            yGm.SetValue(i, result);        }    }private:    AscendC::GlobalTensor<T> xGm;    AscendC::GlobalTensor<T> yGm;};
```

而在 Kernel 的入口函数中，通过 `dtype` 整数值来选择模板实例：

```
// 来源：skills/ascendc-ops-project/templates/kernel_template.cppextern "C" __global__ __aicore__ void op_kernel(    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling){    GET_TILING_DATA_WITH_STRUCT(OpTilingData, tilingData, tiling);    int32_t dtype = tilingData.dtype;    if (dtype == 0) {       // DT_FLOAT        KernelOp<float> op;        op.Init(x, y, tilingData.totalLength, tilingData.blockDim);        op.Process();    } else if (dtype == 1) { // DT_FLOAT16        KernelOp<half> op;        op.Init(x, y, tilingData.totalLength, tilingData.blockDim);        op.Process();    }    // ... 其他类型}
```

这里有一个值得注意的细节：`__aicore__` 修饰符标记了在 AI Core 上运行的函数，`__gm__` 标记了 Global Memory 地址空间。这些都是 Ascend C 的领域特定关键字，类似于 CUDA 中 `__device__` 和 `__shared__` 的角色。

### 2.3 Float16 精度的"升降级"策略

做过半精度计算的工程师都知道一个痛点：`half` 类型的数值范围和精度有限，直接在 `half` 上做累加或复杂运算，精度误差可能会爆炸。仓库模板中专门为 Float16 给出了一个特化版本：

```
// 来源：skills/ascendc-ops-project/templates/kernel_template.cpptemplate<>__aicore__ inline void KernelOp<half>::Process(){    // ...    for (uint32_t i = start; i < end; i++) {        half value_half = xGm.GetValue(i);        float value = (float)value_half;  // 升级为 float32 计算        float result = value;             // 在 float32 空间做运算        yGm.SetValue(i, (half)result);    // 降级写回 half    }}
```

说白了就是"读的时候升精度，算完再降回去"。这个策略在 SKILL.md 的规范文档中被明确列为最佳实践，并指出 NPU 的 AI Core 函数中 **不允许直接进行 half 精度运算** （会报 `half precision operation is not allowed` 错误）。

这一点和 GPU 不同——NVIDIA 的 Tensor Core 原生支持 FP16 运算，而昇腾的 AI Core 在标量计算路径上需要走 FP32。

## 三、Tiling 策略：把大象装进冰箱的分块艺术

如果说 Host-Kernel 分离是 Ascend C 的骨架，那么 **Tiling** 就是它的灵魂。

### 3.1 什么是 Tiling，为什么它如此重要

> 简单来说，Tiling 解决的是一个经典问题：**NPU 的片上缓存（Local Memory）放不下完整的输入数据，怎么办？** 答案是分块——把数据切成一块一块的"瓦片"（Tile），每次只搬一块到片上计算，算完搬出去，再搬下一块。

这和 GPU 编程中的 shared memory tiling 思路一致，但 Ascend C 把它做得更加显式和系统化。仓库提供的 Tiling 数据结构模板展示了它的核心设计：

```
// 来源：skills/ascendc-ops-project/templates/tiling_header_template.hconstexpr uint32_t MAX_DIM = 8;   // 最高支持 8 维张量constexpr uint32_t ALIGN_NUM = 8; // 对齐粒度struct ${OPERATOR_CLASS}TilingData {    uint32_t blockNum;           // 使用的核数    uint32_t totalElements;      // 输出总元素数    uint32_t inputTotalElements; // 输入总元素数（边界检查用）    uint32_t numPerCore;         // 每个核处理的元素数    uint32_t tailNumLastCore;    // 最后一个核的尾部元素数};
```

注意 `tailNumLastCore` 这个字段——当总元素数不能被核数整除时，最后一个核要处理的是"余数"。这种边界处理在并行计算中极其常见，但也极其容易出 bug。**教程中反复强调的"泛化设计"原则，很大程度上就是在教开发者怎么正确处理这些边角情况。**

### 3.2 TilingKey：编译时决策的模板化编程

仓库中还有一个更高级的 Tiling 机制—— `TilingKey` 模板化编程。它允许在编译时根据数据类型的不同组合，生成不同的 Kernel 二进制：

```
// 来源：skills/ascendc-ops-project/templates/tiling_key_template.hASCENDC_TPL_ARGS_DECL(${OPERATOR_CLASS},    // 数据类型模板参数    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16,                              ASCENDC_TPL_INPUT(0)),    ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_FLOAT, C_DT_FLOAT16,                              ASCENDC_TPL_INPUT(1)),    // 自定义 UINT 模板参数    ASCENDC_TPL_UINT_DECL(TILE_NUM, ASCENDC_TPL_8_BW,                          ASCENDC_TPL_UI_MIX, 2, 0, 2, 3, 5, 10),    // 布尔模板参数    ASCENDC_TPL_BOOL_DECL(IS_SPLIT, 0, 1),);
```

这个宏展开后，编译器会为每一种合法的参数组合生成一份优化过的二进制代码。相比运行时的 `if-else` 分发，这种方式可以 **减少 icache miss** ——因为每个特化版本的指令序列更短、更紧凑。技术博客中的《TilingKey 模板化编程》[5]一文对此有详细的性能数据分析。

## 四、算子注册：连接 Host 与 Kernel 的"胶水层"

### 4.1 OpDef 注册机制

> 一个 Ascend C 算子要真正跑起来，光有 Host 侧 Tiling 和 Kernel 侧计算逻辑还不够。它还需要在 CANN 的图引擎中"注册"自己，告诉框架：**我有哪些输入、哪些输出、支持什么数据类型、形状怎么推导。**

仓库模板中的注册代码是这样的：

```
// 来源：skills/ascendc-ops-project/templates/host_template.cppnamespace ops {class Op : public OpDef {public:    explicit Op(const char* name) : OpDef(name)    {        // 声明输入 x：支持 float16/float/int32/int8 四种类型        this->Input("x")            .ParamType(REQUIRED)            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,                       ge::DT_INT32, ge::DT_INT8})            .Format({ge::FORMAT_ND, ge::FORMAT_ND,                     ge::FORMAT_ND, ge::FORMAT_ND});        // 声明输出 y        this->Output("y")            .ParamType(REQUIRED)            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,                       ge::DT_INT32, ge::DT_INT8})            .Format({ge::FORMAT_ND, ge::FORMAT_ND,                     ge::FORMAT_ND, ge::FORMAT_ND});        // 绑定推导函数和 Tiling 函数        this->SetInferShape(ge::InferShape)             .SetInferDataType(ge::InferDataType);        this->AICore().SetTiling(optiling::TilingFunc);        this->AICore().AddConfig("ascend910b"); // 目标芯片    }};OP_ADD(Op); // 注册宏}
```

这段代码里有几个关键设计值得展开说：

- **第一**，`DataType` 列表的长度必须与所有输入输出一致。如果你的算子有两个输入 x 和 axis，x 支持 4 种类型，那 axis 也必须声明 4 种类型（即使它只用 `int32`，也要写 4 个 `ge::DT_INT32`）。这是 `msopgen` 工具的硬性要求。
- **第二**，`AddConfig("ascend910b")` 指定了目标芯片。当前教程基于 Atlas A2 系列（即 Ascend 910B），但仓库路线图中已经标注了 A5 系列的 SIMT 编程教程正在建设中。
- **第三**，`OP_ADD(Op)` 是一个注册宏，它的效果是在程序启动时将算子注册到全局算子表中，供图引擎在编译模型图时查找和调用。

### 4.2 ValueDepend 机制

SKILL.md 中专门用了一整节来讲 `ValueDepend`，这是一个容易踩坑的机制。当算子的某个输入既是张量，又需要在 Tiling 阶段读取其值时（比如 `axis` 参数），必须使用 `ValueDepend` 声明：

```
// 来源：skills/ascendc-ops-project/SKILL.md// 正确写法：ValueDepend 输入的 dtype 必须一致，且只能是 int64/float/boolthis->Input("axis")    .ParamType(REQUIRED)    .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})    .ValueDepend(REQUIRED, DependScope::TILING);
```

这里有一条反直觉的限制：**ValueDepend 输入的数据类型必须全部相同，且只能是 float、int64 或 bool 之一**。如果你尝试用 `int32`，编译器会直接报错。这是因为 Tiling 阶段的张量数据读取走的是一条特殊路径，只支持这三种类型的序列化。

## 五、CANNBot 技能系统：AI 驱动的算子开发

仓库中最让我意外的部分是 `skills/` 目录。它不是教程，而是两个可以被 AI 智能体调用的"技能模块"。

### 5.1 两大核心技能

> 仓库提供了两个 CANNBot 技能：

- **ascendc-ops-project**：Ascend C 算子工程化开发技能，提供从原型定义、工程生成到编译部署的全流程指导和代码模板。
- **cannjudge-submit**：CANNJudge 竞赛提交技能，自动完成登录、下载题目、提交代码、查询结果。

它们之间存在依赖关系：`cannjudge-submit` 负责与 CANNJudge 平台交互，`ascendc-ops-project` 负责提供算子实现的技术细节。两者配合，可以实现"告诉 AI 我要做哪个算子 →AI 自动生成代码 → 自动编译测试 → 自动提交评测 → 返回成绩"的全自动流程。

### 5.2 竞赛提交的安全设计

> CANNJudge 客户端的实现也颇有工程意识。仓库没有让用户直接在对话中输入明文密码，而是设计了一套 RSA 非对称加密机制：

```
# 来源：skills/cannjudge-submit/cannjudge_cli.pydef decrypt_password(ciphertext_b64: str,                     private_key_path: str = "private.pem") -> str:    """用 RSA 私钥解密密码密文。    返回的明文密码仅供内部调用登录 API，禁止输出到对话/日志/文件。"""    rsa_key = RSA.importKey(private_key_pem)    cipher = PKCS1_v1_5.new(rsa_key)    ciphertext = base64.b64decode(ciphertext_b64)    plaintext = cipher.decrypt(ciphertext, sentinel=None)    return plaintext.decode()
```

流程是这样的：在云开发环境中生成 RSA 密钥对 → 私钥留在服务器 → 公钥拷到个人电脑 → 用公钥加密密码得到密文 → 把密文粘贴给 CANNBot。解密后的明文密码只在内存中短暂存在，用于调用登录 API 后立即被 `del password` 清除。

代码注释中反复出现的"安全铁律"——**禁止在对话中输入明文密码、禁止将解密后的密码显示在对话中** ——体现了对安全边界的严格把控。

```
# 来源：skills/cannjudge-submit/cannjudge_cli.pydef login_with_ciphertext(self, email, ciphertext, private_key_path):    """RSA 密文登录，解密后的明文仅用于内部调用"""    password = decrypt_password(ciphertext, private_key_path)    result = self.login(email, password)    del password  # 立即清除明文    return result
```

### 5.3 泛化算子的设计哲学

> CANNBot 技能文档（SKILL.md）中提出了一个核心原则：**泛化设计**。它不鼓励为特定 shape 或特定 dtype 写死逻辑，而是要求算子具备五个维度的泛化能力：

| 泛化维度 | 要求 | 典型陷阱 |
| --- | --- | --- |
| Shape 泛化 | 支持任意维度和大小 | 硬编码 `dim=4` |
| Dtype 泛化 | 模板类支持多种数据类型 | 只处理 `float` |
| 属性泛化 | 处理所有属性值包括边界值 | 忽略 `axis=-1` 的负值语义 |
| 对齐泛化 | 处理对齐和非对齐情况 | 未使用 `DataCopyPad` |
| 边界泛化 | 处理空输入、单元素、极端值 | 除零未保护 |

这个设计原则直接影响了代码模板的写法。比如 Kernel 模板中的多核切分逻辑，使用了 `(totalLength + blockDim - 1) / blockDim` 这种上取整公式来确保最后一个核能正确处理余数元素——这就是"边界泛化"的典型体现。

## 六、编译构建：从源码到可部署安装包

### 6.1 一键编译脚本

仓库提供的 `build.sh.template` 封装了完整的 CMake 编译流程，开发者不需要手动记忆复杂的编译参数：

```
# 来源：skills/ascendc-ops-project/templates/build.sh.templateCANN_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}"# [1/3] 配置 CMakecmake -B "$BUILD_DIR" \    --preset="${BUILD_TYPE,,}" \    -DASCEND_CANN_PACKAGE_PATH="$CANN_PATH"# [2/3] 编译cmake --build "$BUILD_DIR" -j"$JOBS"# [3/3] 打包为 .run 安装包cmake --build "$BUILD_DIR" --target package
```

最终产出的是一个 `.run` 自解压安装包，通过 `./custom_opp_*.run --install-path=${ASCEND_OPP_PATH}` 即可部署到目标环境。这种"编译即打包"的设计，让算子的分发和部署变得非常标准化。

### 6.2 工程生成的自动化

整个工程的创建起点是 `msopgen` 工具，它根据算子原型 JSON 文件自动生成工程骨架：

```
msopgen gen -i op.json -c ai_core-ascend910b -lan cpp -out custom_op
```

生成的工程包含完整的 `op_host/`、`op_kernel/` 目录以及 CMakeLists.txt，开发者只需要在模板基础上填写自己的计算逻辑。这种"脚手架"式的开发体验，和前端领域的 `create-react-app` 异曲同工。

## 七、教程体系与技术博客：从入门到前沿

### 7.1 八章 Ascend C 算子开发教程

> 教程按难度分三级，覆盖了算子开发的完整知识栈：

| 难度 | 章节 | 核心内容 |
| --- | --- | --- |
| 初级 | 第一章：高性能算子开发语言 | AI 基础、CANN 架构、NPU 原理 |
| 初级 | 第二章：开发基础知识 | Hello World、编程范式、Add 算子实战 |
| 中级 | 第三章：Vector 算子开发 | 工程化开发、aclnn 调用、泛化 Tiling |
| 中级 | 第四章：矩阵算子开发 | 矩阵乘 API、高阶 Matmul |
| 中级 | 第五章：融合算子开发 | VV 融合、CV 融合 |
| 中级 | 第六章：开源算子仓 | UT 编写、社区贡献 |
| 高级 | 第七章：功能调试 | CPU 域孪生调试、NPU 上板调试 |
| 高级 | 第八章：性能优化 | msProf 工具、仿真分析 |

每一章都是可运行的 Jupyter Notebook，配合云端 NPU 环境可以直接在浏览器中执行。这种"文档即代码"的理念，确保了教程中的每一行代码都是经过验证的。

### 7.2 HiXL 单边通信教程

> 除了算子开发，仓库还包含了一套 4 章的 HiXL（Huawei Xfer Library）单边通信教程。HiXL 是昇腾的高性能设备间通信库，支持 D2D（Device-to-Device）和 H2H（Host-to-Host）传输，在大模型推理的 PD 分离（Prefill-Decode 分离）场景中至关重要。

教程覆盖了三种传输模式：**直传模式** （低延迟、适合小数据量）、**中转传输模式** （适合跨节点通信）和 **FabricMem 模式** （适合大规模 KV Cache 池化场景）。

### 7.3 二十五篇技术博客

> `blogs/` 目录下按"算子—推理—训练"三个维度组织了 25 篇技术博客，每一篇都是实际工程场景中的经验结晶。举几个有代表性的：

- **Overlap Scheduling 吞吐优化**：通过 CPU 调度与 NPU 执行的时间重叠，隐藏任务下发延迟，TPS（Tokens Per Second）提升约 70%。
- **SuperKernel 技术综述**：将整网重新编译为"大算子"，减少调度与显存访问开销，在已有优化基础上再提升 10%-20%。
- **Deepseek-R1 SuperPoD 推理优化**：全栈协同优化，实现 TTFT（首 Token 延迟）< 2s、TPOT（Token 间延迟）< 50ms。
- **CrossEntropyLoss 与 Zloss 融合**：MoE 场景下通过损失函数融合实现端到端 5.2% 的效率提升。

::: block-1
这些博客覆盖的技术深度远超教程本身，它们本质上是 **CANN 团队在产业实战中积累的技术备忘录**，对于有一定基础的开发者极具参考价值。
:::

## 八、社区共建与生态扩展

### 8.1 社区贡献通道

> `contrib/` 目录已经有了第一批外部贡献——来自 DataWhale 社区和 Torch-RecHub 的推荐系统实战教程，**覆盖了 CTR 预测（DeepFM）、序列兴趣建模（DIN）、双塔召回（DSSM）等推荐系统核心场景**。另外，SwanLab 社区共建的 LLM 微调实战教程也在评审中。

贡献流程相当标准化：Fork → 在 `contrib/tutorials/` 下创建目录 → 提交 PR → 等待审核合并。仓库对提交的 Notebook 要求使用 UTF-8 编码、避免硬编码路径、提交前需在本地验证可执行——这些看似基础的规范，其实是保证教程长期可维护的关键。

### 8.2 未来方向

> 从仓库 README 的路线图来看，CANN 学习中心的野心不止于算子开发。正在建设中的方向包括：

- **A5 系列 SIMT 编程**：兼容 A2/A3 的 SIMD 模型，同时支持新增的 SIMT 编程范式
- **PyPTO / PyASC / TileLang**：基于 Python 的算子开发框架，进一步降低门槛
- **大模型推理优化系列**：从 Qwen3-0.8B 到 8B，从单卡部署到 Profiling 瓶颈定位
- **图框架加速**：图编译与图模式应用的系统教程

## 写在最后

> `cann-learning-hub` 做对了一件很关键的事：**它没有把教程和工具割裂开，而是把它们编织成了一个有机的整体。**

- 教程中出现的代码模板，就是 CANNBot 技能中使用的模板；
- 技术博客中讨论的 TilingKey 优化，对应着教程第三章 3.5 节的编程实践；
- 竞赛提交工具读取的文件目录结构，和 `msopgen` 生成的工程结构完全一致。

从技术视角来看，整个仓库最核心的贡献是把 Ascend C 算子开发这件"看起来门槛很高"的事情，拆解成了可操作、可验证、可复用的标准流程：:mdnice-bold-italic[原型定义 → 工程生成 → Tiling 设计 → Host 实现 → Kernel 实现 → 编译安装 → 测试验证。每一步都有模板、有规范、有示例，甚至有 AI 来帮你兜底。]

::: block-1
对于正在考虑昇腾 NPU 开发的工程师来说，这个仓库可能是目前最低成本的入门路径。十分钟跑通一个算子，八章教程建立体系，刷题竞赛检验能力——这套组合拳打下来，至少比对着 API 文档"盲人摸象"要高效得多。
:::

当然，仓库也有明显的未完成感。**大模型推理和训练方向的教程大多还是"建设中"状态，A5 系列的 SIMT 编程教程也尚未发布**。但作为一个上线仅三个月的项目来说，它已经展现出了足够的技术深度和生态规划能力。接下来的关键，是看这个"学·练·赛"的飞轮能不能真正转起来。

参考资料

[1] 10 分钟体验自定义算子: *https://github.com/hicann/cann-learning-hub/tree/master/quick_start/first_custom_operator*

[2] `docs/gitcode_env_experience_guide`: *https://github.com/hicann/cann-learning-hub/tree/master/docs/gitcode_env_experience_guide*

[3] release 仓库: *https://gitcode.com/cann/release-management*

[4] CANNJudge: *https://cannjudge.cn*

[5] 《TilingKey 模板化编程》: *https://github.com/hicann/cann-learning-hub/blob/master/blogs/operator/tilingkey_template_programming*

::: related-title
**相关推荐**
:::

::: related-links
- **[从 MegaKernel 视角重新定义 NPU 上的线性注意力极限！PTO-ISA 驱动的昇腾 NPU Gated DeltaNet MegaKernel 浅析](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901380&idx=1&sn=4ae6987a6ba87626737c46cb4be9458e&scene=21#wechat_redirect)**
- **[鹏城联合华为昇腾、中山大学提出AscendKernelGen：NPU 算子闭环生成与评估，复杂L2 Kernel编译成功率95.5%，正确率64.3](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896566&idx=1&sn=d920e9f8feed38e4734d18942d83897a&scene=21#wechat_redirect)**
- **[SGEMM-cube：昇腾910A用FP16 Cube单元模拟 FP32 GEMM 实现 22 位尾数精度恢复，77%理论峰值！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890912&idx=1&sn=9bd135756aa33bd195fa42099bd6e840&scene=21#wechat_redirect)**

:::
