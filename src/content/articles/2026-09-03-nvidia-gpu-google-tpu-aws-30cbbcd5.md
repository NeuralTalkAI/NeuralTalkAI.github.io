**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**编译器逆向**、CUDA、TPU、AWS Neuron、**形式验证**

> 闭源 AI 加速器最深的壁垒，在从计算图到机器指令之间那条漫长的软件隧道里。

- NVIDIA CUDA 程序为何被拆成 host 与 device 两路？PTX 经过哪些优化才变成 SASS？
- Google TPU 编译器怎样估算矩阵单元、片上存储与互联的代价？
- AWS Trainium 又如何把 HLO、内核 DSL 和多种中间表示封装进可执行文件？

::: block-1
厂商文档通常只描述入口与结果，:mdnice-bold-italic[真正决定兼容性、正确性和性能的内部数据结构、调度模型及二进制协议仍藏在巨型 ELF 文件中。]
:::

![文章插图](/articles/2026-09-03-nvidia-gpu-google-tpu-aws-30cbbcd5/images/01.jpg)

- **Reverse engineering notes. Personal reference only. Everything here is a best-guess reconstruction.**
- 代码：https://gh.evko.io/crucible-notes/
- 在线文档：https://github.com/GrigoryEvko/crucible-notes

::: related-title
**相关推荐**
:::

::: related-links
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[横跨 NVIDIA, AMD, Intel 三十年 GPU 演进史！为 2824 款 GPU 架构、显存、算力与线程约束统一建模的 RightNow GPU Database](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447903298&idx=1&sn=cef94205aad2c37d0669a19aa53572a7&scene=21#wechat_redirect)**
- **[KernelWiki：Blackwell 内核调优的 Wikipedia，将2179个PR、48篇综述、20篇博客、11篇官方文档和7场竞赛，压缩成结构化技能](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901711&idx=1&sn=ded74b498ed6a51e8b99b82244cd2f0e&scene=21#wechat_redirect)**

:::

> crucible-notes[1] 选择了一条极具野心的路径：把 NVIDIA CUDA、Google TPU 和 AWS Neuron 三套加速器软件栈逐层拆开，:mdnice-bold-italic[用字符串交叉引用、调用图、虚表、数据表提取、差分实验、GPU 实测乃至 Lean 形式证明，将观察到的行为整理成可检索、可复核的工程文档，真正把二进制证据炼成可复现知识。]

它同时反复提醒读者：这些内容包含 AI 生成的逆向笔记，属于作者的 best-guess reconstruction，不能直接等同于厂商规格。也正因为保留了这种边界意识，这个仓库最值得研究的地方，除了“发现了什么”，更在于“如何把黑盒猜测逐步升级为带证据等级的知识”。

::: block-1
此外，跨厂商的逆向研究带来一个额外价值👀：:mdnice-bold-italic[读者可以看到不同编译器在表面命名之外的共同骨架—— 图优化、中间表示下降、目标代价模型、指令选择、寄存器或片上存储分配、依赖调度、二进制封装和运行时提交]。这些结构构成 AI 加速器软件栈的“比较解剖学”。
:::

![文章插图](/articles/2026-09-03-nvidia-gpu-google-tpu-aws-30cbbcd5/images/02.webp)

::: image-caption
crucible-notes 项目标识
:::

## 本文目录

- 一、一座加速器软件栈的“数字解剖馆”
  - 1.1 仓库真正研究的对象
  - 1.2 顶层目录透露出的研究哲学
  - 1.3 版本锚定是一切结论的地基
- 二、从 CUDA 源码到 GPU 机器码：仓库如何拼回整条编译链
  - 2.1 cudafe++：先把一份 CUDA 源码拆成两个世界
  - 2.2 cicc：商业前端、NVVM 桥与 LLVM 后端的混合体
  - 2.3 ptxas：从虚拟 ISA 跨越到真实硬件
  - 2.4 nvlink：链接器里藏着一套编译器
- 三、逆向工程的核心手艺：从匿名函数中建立证据链
  - 3.1 字符串是黑盒中的路标
  - 3.2 虚表与结构模板用来突破间接调用
  - 3.3 调用图把局部发现组织成系统
  - 3.4 解密脚本展示了从反汇编到可执行知识的跨越
- 四、从“看懂”到“能运行”：SASS 工具链如何闭合验证回路
  - 4.1 表驱动解码器把 128 位机器字还原成语义字段
  - 4.2 调度器把机器指令转换为带权依赖图
  - 4.3 Z3 将全程序重排变成约束优化
  - 4.4 GPU 是最终裁判，但裁决仍有范围
- 五、Lean 形式证明：让逆向语义越过“样本正确”的天花板
  - 5.1 从 Python 模型转向带宽度的 BitVec
  - 5.2 三种验证层次各自解决不同问题
- 六、项目真正的突破、局限与工程启示
  - 6.1 它重建的其实是一套“可执行文档”
  - 6.2 TPU 与 Neuron 让项目超越 CUDA 单点研究
  - 6.3 最需要警惕的是证据不均与内部漂移
  - 6.4 对编译器与 AI 系统研究者的三点启示

![文章插图](/articles/2026-09-03-nvidia-gpu-google-tpu-aws-30cbbcd5/images/03.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、一座加速器软件栈的“数字解剖馆”

### 1.1 仓库真正研究的对象

> 从 GitHub 的语言统计看，这似乎是一个以 Python 为主的项目：Python 占 75.3%，CSS 占 12.7%，C 占 7.2%，其余是 JavaScript、HTML 和 Lean。然而，这个数字容易制造错觉。仓库的主体并非传统意义上的 Python 应用，**其核心资产是数千页 Markdown 逆向文档、结构化 TSV 数据表、提取与验证脚本，以及少量可以直接运行的二进制工具。**

Markdown 通常不会主导 GitHub Linguist 的语言占比，所以 Python 的 75.3% 更准确地表示**“可执行研究工具主要用 Python 编写”**，并不代表项目本身是一套 Python 服务。CSS、HTML 和 JavaScript 负责 mdBook 站点及免责声明；**C 实现 fat binary 操作工具；Lean 则承担位级语义证明。**

仓库首页将研究对象分成三大阵营：

| 阵营 | 主要组件 | 研究边界 |
| --- | --- | --- |
| NVIDIA CUDA | `cudafe++`、`cicc`、`ptxas`、`nvlink`、`tileiras`、`fatbin` | 从 CUDA C++ 前端、NVVM/LLVM 优化、PTX 汇编，到 SASS 编码、设备链接和 fat binary 容器 |
| Google TPU | `libtpu` | PJRT 接口、编译流水线、LLO VLIW ISA、代价模型、调度、片上存储、互联和运行时 |
| AWS Neuron | `neuronx-cc`、`neuronx-runtime`、`neuronx-gpsimd` 等 | Trainium/Inferentia 编译器、NEFF、运行时、内核驱动、固件、分布式通信与自定义算子 |

表中信息来自仓库的 README 组件清单[2] 和 站点首页组件表[3]。

### 1.2 顶层目录透露出的研究哲学

> 仓库可以粗略归纳为四层：

| 层次 | 代表目录 | 作用 |
| --- | --- | --- |
| 组件知识库 | `cicc/`、`ptxas/`、`libtpu/`、`neuronx-cc/` 等 | 每个二进制或软件栈拥有独立 mdBook |
| 解码事实库 | `decoded/` | 保存解码器、调度表、指令表、寄存器表、ELF 属性表和验证数据 |
| 可执行工具 | `fatbin/`、`decoded/sass-tools/`、`decoded/sass-bitsem/` | 解析容器、解码指令、重排调度、验证语义 |
| 发布系统 | `.github/workflows/`、`site/` | 构建多个 mdBook，组装 GitHub Pages，并生成 canonical URL 与 sitemap |

这套结构很像一间数字考古实验室🤣：wiki 是整理后的展厅，`decoded/` 是标号归档的出土物，Python 和 C 工具是刷子、探针与测量仪，Lean 则像一台用于核验结构强度的证明设备。

::: block-1
**仓库的重点也因此从“给出某个神秘函数的名字”转向“形成一条可重复的证据链”**。:mdnice-highlight[输入是什么版本的二进制，使用什么地址或数据表定位，如何提取，怎样交叉验证，哪些结论仍属推断，这些内容都被视为成果的一部分。]
:::

### 1.3 版本锚定是一切结论的地基

> 逆向文档最怕版本漂移。同一函数在重新编译后可能换地址，结构体增加一个字段也会让后续偏移全部失效。项目为此建立了专门的 PROVENANCE.md[4]，记录目标二进制的版本、角色和 SHA-256。

| 二进制 | 版本 | 角色 |
| --- | --- | --- |
| `ptxas` | 13.0.88 | PTX 到 SASS 的汇编与优化 |
| `cicc` | 13.0.88 | NVVM IR 到 PTX 的 LLVM 系编译器 |
| `cudafe++` | 13.0.88 | CUDA C++ 前端与 host/device 分离 |
| `nvlink` | 13.0.88 | 设备代码链接 |
| `tileiras` | 13.1.80 | Tile IR 优化汇编 |
| `nvdisasm` | 13.1.115 | SASS 反汇编与控制流提取 |

更关键的是，文件还记录了官方下载地址、wheel 哈希、wheel 内部路径和解压命令。第三方可以取得字节完全相同的输入，再检查某个地址、字符串或数据表是否真实存在。对于一个没有官方源码可以引用的项目，这种 provenance 机制相当于实验科学中的样本编号。

## 二、从 CUDA 源码到 GPU 机器码：仓库如何拼回整条编译链

### 2.1 cudafe++：先把一份 CUDA 源码拆成两个世界

> CUDA 的 `__host__`、`__device__` 和 `__global__` 让同一翻译单元同时属于 CPU 与 GPU。`cudafe++` 的职责，是理解这些执行空间，将源码拆成 host 路径和 device 路径。

**仓库对 `cudafe++` 13.0 的分析将其定位为基于 EDG 6.6 的前端**。根据 cudafe++ 首页[5]，device 侧输出带类型和作用域连接的 EDG IL，交给 `cicc`；host 侧输出转换后的 `.int.c`，随后交给 GCC、Clang 或 MSVC。

可以把它看成海关分流系统：解析器先检查每个声明、调用与 lambda 的“护照”，再决定它属于 CPU 通道、GPU 通道，或同时具有两种身份。

具体来说，仓库给出的关键证据之一，是实体节点偏移 `+182` 处的执行空间位域：

| 位 | 掩码 | 项目推断的含义 |
| --- | --- | --- |
| 4—5 | `0x30` | 无限定、host、device、host-device |
| 6 | `0x40` | device/global 标记 |
| 7 | `0x80` | global kernel 标记 |

该表及其用途见 cudafe++ 执行空间分析[6]。同一个位域会被跨空间调用检查、device 代码筛选、host 输出和 kernel stub 生成反复读取。这说明它很可能是贯穿前端的核心语义状态，而非局部标志。

### 2.2 cicc：商业前端、NVVM 桥与 LLVM 后端的混合体

> `cicc` 的体量远大于 `cudafe++`。仓库记录的目标二进制共包含 80,562 个函数，其中 80,281 个成功被反编译，整体架构清晰划分为三个核心模块：

1. 基于 EDG 6.6 的 C++ 前端，负责语法解析与语义分析；
2. 自研的 NVVM 桥接层，实现 EDG AST 到 LLVM IR 的语义对齐与 CUDA 特性注入；
3. 经深度定制的 LLVM 后端，适配 NVIDIA GPU 的指令集、内存模型与调度约束。

值得注意的是，cicc 总览[7] 明确指出，该二进制中实际嵌入了两条**逻辑独立、物理分离** 的后端执行路径：:mdnice-bold-italic[一条专为 LibNVVM API 设计，供 `nvcc` 动态链接调用；另一条则面向 standalone 编译模式，支持离线生成 `.cubin`。] 尽管二者地址空间不同，却共享高度一致的组件结构——包括成对的 CLI 参数解析器、顶层编译入口函数、builtin 函数符号表，以及内嵌的 `libdevice` 实现。这种设计看起来冗余，实际是 NVIDIA 为兼顾兼容性与部署灵活性所作的工程权衡。

::: block-1
当然，**这种“双副本”结构为逆向工程提供了天然的交叉验证机会**。例如，若在路径 A 中识别出某函数承担 `__syncthreads()` builtin 查询职责，而在路径 B 中定位到控制流图高度相似、字符串常量（如 `"__syncthreads"`）与全局数据访问模式（如对 builtin 表的偏移寻址）完全对应的函数，则二者互为佐证，极大提升了函数语义判定的置信度。
:::

> 仓库重建出的主要流水线如下，详见 cicc 编译流水线[8]：

| 阶段 | 输入与输出 | 关键职责 |
| --- | --- | --- |
| IRGEN | EDG IL → LLVM IR | 类型映射、表达式和语句生成、CUDA builtin 降低 |
| LNK | LLVM 模块 → 链接后模块 | 链接内嵌 libdevice、验证 triple 与 IR 版本 |
| OPT | LLVM IR → 优化 IR | 全模块优化、逐函数优化、NVIDIA 自定义 pass |
| OPTIXIR | 可选输出 | 生成 OptiX IR |
| LLC | LLVM IR → PTX | SelectionDAG、指令选择、寄存器类与 PTX 发射 |

其中比较有价值的观察是“两阶段优化”：

- **第一阶段处理全模块决策**，包括内联和跨函数分析；
- **第二阶段按函数重新优化**，并允许借助线程池并发执行。这让 `cicc` 更接近一个带 GPU 专用 pass 的工业编译平台，而非简单的 CUDA 语法翻译器。

### 2.3 ptxas：从虚拟 ISA 跨越到真实硬件

> PTX 是 NVIDIA 面向软件的虚拟指令集，SASS 才是 GPU 真正执行的机器指令**。`ptxas` 正处于抽象与硬件之间最敏感的一层。**

仓库将 `ptxas` 划分成三个子系统，更多参考 ptxas 三大子系统[9]：

| 子系统 | 项目识别出的结构 |
| --- | --- |
| PTX 前端 | Flex DFA 扫描器、Bison LALR(1) 解析器、指令描述符和语义验证器 |
| Ori 优化器 | 基本块与指令 DAG、159 阶段 PhaseManager、寄存器分配与调度 |
| SASS 后端 | 指令选择、Mercury 编码、peephole、操作数展开和 ELF/CUBIN 输出 |

仓库进一步区分“注册阶段”和“默认执行阶段”：PhaseManager 构造 159 个 phase 对象，但默认调度表只派发前 157 个，尾部的 `DebuggerBreak` 与 `NOP` 不进入常规编译。若内部 knob 298 被设置，驱动器会改走一个 recipe DSL，构造自定义阶段序列，再复用同一派发循环。这条模型来自 ptxas driver 说明[10]。

> **这种高质量的逆向分析，其核心在于多源证据的交叉验证与协同解读**。当面对一个看似孤立的 `159` 分支的 `switch` 语句时，仅凭其结构，我们最多只能推测它可能承担对象创建职责——即“对象工厂”的雏形。但这一推测缺乏坚实依据，极易误判。

真正提升分析置信度的，是同步引入五类关键线索：

- **phase 名称表** 揭示了系统级状态流转逻辑；
- **工厂 `switch`** 提供了实例化入口的粗粒度分发视图；
- **虚表（vtable）布局** 暴露了对象行为契约与继承关系；
- **默认顺序数组** 则隐含了初始化优先级与依赖拓扑；
- 而**运行时调度入口** （如 `dispatch_phase()` 或 `init_all()`）则锚定了这些组件在真实执行流中的激活时机与上下文。

**这五类信息彼此印证、层层收敛**：例如，`switch` 中某分支调用的构造函数地址，若恰好在虚表首项出现，且该类名又出现在 phase 表的“`INIT`”阶段条目中，同时其初始化顺序索引与默认数组位置一致——此时，该分支作为特定 phase 下某类对象的工厂入口，便从假设升格为可复现、可验证的工程事实。:mdnice-bold-italic[这种证据链驱动的分析范式，不仅显著降低主观臆断风险，更使逆向成果具备接近原始设计文档的结构清晰度与语义完整性。]

### 2.4 nvlink：链接器里藏着一套编译器

> `nvlink` 接受 cubin、PTX、fatbin、NVVM IR、archive 和 host ELF。项目分析认为其约 95% 的体量属于嵌入式编译后端及指令编码设施，真正的链接器核心只占很小一部分。

这解释了一个看似反常的现象：:mdnice-bold-italic[设备链接器为什么会拥有指令选择、寄存器分配和调度代码？因为 LTO 或 PTX 输入需要在链接阶段完成 IR→PTX→SASS 的转换，`nvlink` 为此静态嵌入了大量 `ptxas` 后端。]

它的主干流程由文件类型检测开始，依次经过 section 合并、符号解析、共享内存布局、调用图可达性分析、数据布局、地址分配、重定位和 ELF 输出；SM100 及以后还会进入仓库称为 Mercury/FNLZR 的后处理路径。详见如下 nvlink 架构图[11]。

```
Input Files (cubin, PTX, fatbin, NVVM IR, archives, host ELF)  |  +-- [File type detection: 56-byte header read, magic number dispatch]  |  +-- PTX input?  --> embedded ptxas JIT: PTX -> SASS cubin  |  +-- NVVM IR / LTO IR?  --> dlopen libnvvm.so, compile IR -> PTX,  |                          then embedded ptxas: PTX -> SASS cubin  |  +-- fatbin?  --> extract members, recurse per-member  |  +-- archive (.a)?  --> iterate members, recurse  |  vMerge Phase (merge_elf: 89KB function, iterates all sections)  |  +-- Symbol resolution, weak symbol selection  +-- Section merging (.nv.global, .nv.shared, .nv.constant, .text)  +-- Debug section merging (DWARF line tables, .nv.info)  |  vShared Memory Layout (65KB function, overlap set analysis)  |  vEntry Property Computation (98KB function, register/barrier propagation)  |  vDead Code Elimination (callgraph reachability, removes unused functions)  |  vData Layout Optimization (constant dedup, overlapping data merge)  |  vLayout Phase (section ordering, address assignment)  |  vRelocation Phase (R_CUDA relocation application, UFT/UDT resolution)  |  vFinalization (final relocation patching, ELF section generation)  |  vOutput (ELF serialization -> cubin file)  |  +-- SM100+?  --> FNLZR post-link transform -> capsule mercury  +-- --gen-host-linker-script?  --> write SECTIONS { .nvFatBinSegment ... }
```

## 三、逆向工程的核心手艺：从匿名函数中建立证据链

### 3.1 字符串是黑盒中的路标

> 被 strip 的 ELF 文件缺失符号表，IDA 等反编译工具只能显示 `sub_446240` 这类无意义的地址化函数名，极大增加逆向分析难度。此时，程序中残留的**高稳定性字符串线索** ——如错误提示、计时标签、断言路径、命令行选项及 pass 名称——便成为定位关键逻辑的锚点，其可靠性远超易被优化或删除的调试符号。

以 NVIDIA 编译器工具链为例：`ptxas` 中大量使用形如 `Parse-time`、`DAGgen-time`、`OCG-time`、`ELF-time` 的格式化字符串，这些字符串不仅出现在日志输出中，更被嵌入计时器初始化与结束逻辑。:mdnice-bold-italic[通过交叉引用（XREF）回溯，可精准定位各编译阶段的起止边界，进而向上追踪至对应阶段的入口函数。类似地，`cicc` 中的 `instcombine`、`gvn`、`nvvm-memspace-opt` 等字符串，直接对应 LLVM Pass 注册结构中的 `PassInfo` 或 `PassRegistry` 调用点，为识别 pass 注册与执行流程提供明确路径。]

::: block-1
值得注意的是，`ptxas` 对部分内部 knob（配置开关）及 opcode 名称采用 ROT13 轻量混淆。尽管该编码无密码学强度，但解码后如 `MercuryUseActiveThreadCollectiveInsts`、`ScavDisableSpilling` 等名称，能清晰揭示模块职责与设计意图，有效辅助边界划分与功能归类。
:::

然而，必须保持方法论上的审慎：名称仅指示语义倾向，而非完整行为。例如 `DisableSpilling` 可确认其与寄存器溢出相关，:mdnice-bold-italic[但具体生效时机、默认值继承路径、是否影响 spill placement 或 reload 插入等副作用，仍需结合数据流追踪其读取点（如 `cl::opt<bool>` 构造与 `getValue()` 调用）及控制流上下文。官方方法论文档亦明确指出：解码名称是发现入口的起点，而非语义确认的终点。]

### 3.2 虚表与结构模板用来突破间接调用

> 静态调用图对直接 `call` 指令的分析非常有效，但面对虚函数调用和函数指针跳转时，却存在天然的局限性——这类调用目标在编译期无法确定，必须依赖运行时的虚表（vtable）或间接跳转逻辑:mdnice-bold-italic[。`ptxas` 的方法论文档指出，高达 39.6% 的函数在静态分析中找不到明确的 caller，其中绝大多数正是通过虚表动态分派的成员函数，凸显了传统静态分析在面向对象语境下的盲区。]

为突破这一瓶颈，项目构建了一套基于**结构特征识别** 的逆向分析框架：首先，通过一张涵盖 159 个编译 phase 的虚表映射表，精准定位每个 phase 对应的 `execute()`、`isNoOp()` 和 `getName()` 等关键虚函数入口；继而，针对底层 SASS 编码逻辑，提炼出约 4000 个 handler 函数共有的六步模板，实现批量识别与归类：

1. 写入 opcode ID，作为指令语义的顶层标识；
2. 从只读数据段加载 128 位格式描述符，提供操作数布局与编码约束；
3. 初始化寄存器类别映射（如 `R` 类、`C` 类、`P` 类），支撑后续语义解析；
4. 注册操作数描述，明确各字段的位宽、偏移与语义角色；
5. 完成 SASS 指令编码逻辑，生成合法机器码片段；
6. 提取打包后的位域，用于校验或反向映射。

该方法借鉴生物学中的形态分类思想：:mdnice-bold-italic[即便函数名被剥离（如编译器内联、符号剥离或模板实例化），其“骨骼结构”——即指令序列的控制流模式、数据加载顺序与寄存器使用范式——仍高度稳定。只要模板覆盖充分、特征提取鲁棒，就能先完成粗粒度分类（“属于哪一科”），再结合常量字面量、虚表索引偏移及调用上下文，逐步收敛至具体指令变体，显著提升虚函数与间接调用链的可追溯性。]

### 3.3 调用图把局部发现组织成系统

> 仓库中的 validate_callgraph.py[12] 首先解析二进制符号与重定位信息，构建完整的调用图索引——包括每个函数的 caller 集合、callee 集合、调用频次、内存地址及所属模块；在此基础上，进一步推导跨模块调用路径、识别潜在入口点（如未被静态调用但导出或被 PLT 引用的函数）、评估函数可达性、检测依赖环（尤其在模块间循环依赖场景）、定位桥接函数（即唯一连接两个模块子图的函数），并标记静态不可达但符号可见的疑似死代码。

这段脚本的实现包含 BFS 遍历、DFS 环检测、集合运算与哈希索引均为成熟技术。但**其核心意义在于将逆向分析中依赖经验与直觉的判断过程（例如“该函数很可能是入口”或“此处调用频繁，应重点关注”）转化为可复现、可验证、可版本化追踪的自动化流程**。

- 当研究者提出某函数是跨模块调用热点时，脚本可即时重算其 fan-in 数量并比对历史快照；
- 当断言某函数为入口点时，工具能严格验证其是否具备静态 caller，或是否仅通过动态机制（如 GOT/PLT、dlsym）被间接调用。

当然，该分析基于静态链接视图，无法捕获虚函数调用、函数指针间接跳转、运行时加载的回调等动态行为。因此，“静态不可达”仅表明该函数在当前符号与重定位上下文中无显式调用路径，**不等于** 运行时不可达。仓库在方法论文档中明确声明这一局限，并将“静态不可达 + 无导出符号 + 无 PLT/GOT 引用”作为死代码的**必要但不充分条件**，体现了对证据链完整性的审慎态度。

### 3.4 解密脚本展示了从反汇编到可执行知识的跨越

> `decoded/` 目录系统性地记录了四类经典数据隐藏方案，涵盖流密码、位置异或、ROT13 等基础编码机制，其核心解码逻辑集中体现在 `decode_pool.py`[13] 中——该脚本复现了一条从反汇编线索中逆向还原出的字节级解码链，完整还原了原始二进制中嵌入的隐写逻辑。

| 步骤 | 运算 |
| --- | --- |
| 状态推进 | 32 位线性同余生成器（LCG）：`seed = 1103515245 × seed + 12345`，提供确定性密钥流基础 |
| 字节选取 | 按序消费 LCG 输出的每个 32 位状态的低 8 位，形成逐字节密钥流 |
| 反馈 | 引入前一密文字节参与当前轮次运算，增强状态依赖性与非线性特征 |
| 代换 | 查表操作基于目标二进制自身携带的 256 字节 S-box（硬编码于 payload），确保解码逻辑与载体强绑定 |
| 输出 | S-box 查表结果与密钥流字节执行异或，生成最终明文字节 |

该项目不仅提供可复现的解码机制、预置常量及配套提取工具，更强调其技术本质属于不可版权化的事实性方法与数学过程。所有工具均面向用户**合法取得的二进制样本** 设计，其使用边界在 decoded/MANIFEST.md[14] 第 13–40 行中明确定义：禁止用于未经授权的逆向分析或规避版权保护措施，严格遵循合理使用原则。

## 四、从“看懂”到“能运行”：SASS 工具链如何闭合验证回路

### 4.1 表驱动解码器把 128 位机器字还原成语义字段

> `sass_decode.py`[15] 使用从 `nvdisasm` 架构表恢复的位域模型，对 128 位 SASS 指令进行表驱动解码。

| 位区间 | 字段 |
| --- | --- |
| `11:0` 与位 91 | opcode |
| `14:12`、15 | guard predicate 与取反 |
| `16:91` | 寄存器、立即数、常量 bank、modifier |
| `105:109` | stall 与 yield |
| `110:112` | 写 scoreboard |
| `113:115` | 读/释放 scoreboard |
| `116:121` | wait mask |
| `122:125` | reuse 或 batch 控制 |

来源：SASS 解码器字段说明[16]。

解码器并未硬编码“某 opcode 永远对应某字符串”。它先建立 canonical opcode 到 instruction class 的一级索引，再用 sub-form opcode 和固定常量位处理立即数形式及歧义。真实 cubin 的 `.text.*` section 则通过 `readelf` 读取偏移，按每 16 字节一条指令解码。

仓库报告该模型在多个架构、368 条真实指令上，对 opcode、目标寄存器、源寄存器和 guard predicate 达到 100% 匹配。这个数字属于项目自己的验证结果，适用范围限定在测试样本与指定 CUDA 版本，不能外推为“覆盖全部 SASS”。

### 4.2 调度器把机器指令转换为带权依赖图

> SASS 调度控制字决定指令需要停顿多久、写入哪个 scoreboard、等待哪些异步结果。项目在 `sass-tools/README.md`[17] 中把调度抽象成四层：

| 层 | 表示 |
| --- | --- |
| L0 | 原始 128 位机器字 |
| L1 | stall、yield、scoreboard、wait mask 等控制字段 |
| L2 | “设置 SB”“等待 SB”“停顿 k 周期”等语义事件 |
| L3 | 指令间带类型、带权重的依赖 DAG |

依赖边至少包括 RAW、WAR、WAW 和 CTRL。这里最容易犯错的是把 WAR 当作 RAW 的反向版本。RAW 表示消费者必须等待新值产生，权重来自结果延迟；WAR 表示后写入不能破坏先读取，通常只要求顺序或很小的 operand-read 延迟。两者对应不同的物理事件，自然不应共用对称延迟矩阵。

对固定延迟指令，调度器求解差分约束：消费者发射周期减去生产者发射周期必须大于等于依赖权重。对内存等可变延迟生产者，则分配虚拟 scoreboard，将生产者的完成事件与消费者的 wait bit 配对。

### 4.3 Z3 将全程序重排变成约束优化

> `sass_smt.py`[18] 为每条指令建立一个整数发射周期，并加入以下约束：

- 每个可重排基本块内的发射周期互异；
- RAW 延迟和 WAR/WAW/CTRL 顺序必须满足；
- terminator 固定在基本块末尾；
- barrier、控制指令和跨块状态相关的块保持原顺序；
- 指令不得跨越分支边界；
- 目标函数是最小化各块 makespan 之和。

值得注意的是，代码后来放弃直接使用 Z3 `Optimize`，转而对总 makespan 做有界 `check-sat` 和二分搜索。原因很工程化：带 `distinct` 与差分约束的优化问题可能让 Optimize 超时，而普通可满足性查询通常能快速回答。每次查询设置短超时，一旦无法继续收紧，就保留已找到的最佳可行模型。

这种策略体现了仓库的一贯风格：理论模型要服务于可运行工具，求出稍次但安全的解，往往优于追求全局最优却长期无结果。

### 4.4 GPU 是最终裁判，但裁决仍有范围

> 项目不会因 Z3 返回 `sat` 就直接宣称调度正确。它把重排后的控制字写回 cubin，再用相同输入分别运行原始版本 V1 与修改版本 V2，并比较输出。

仓库报告的部分实验结果如下（来源：调度器 GPU 实测表[19] 与 约束调度结果[20]）：

| 测试 | 项目记录的周期改善 |
| --- | --- |
| `amp_transc` | 2.70% |
| `amp_fpchain` | 4.70% |
| `amp_intchain` | 1.59% |
| `rc1_twochain` 重排 | 1.08% |

这些结果很有启发性，却需要准确解读。“多组输入上 bit-identical”是强有力的动态证据，仍不等价于对所有输入的数学证明；周期收益也与 GPU 型号、时钟、占用率、编译版本和测试 kernel 有关。仓库自己同样承认有限种子集的边界。它采用的正确姿态是：验证通过才接受，失败便回退到 `ptxas` 原始结果，并持续扩大测试面。

## 五、Lean 形式证明：让逆向语义越过“样本正确”的天花板

### 5.1 从 Python 模型转向带宽度的 BitVec

> 差分测试能发现大量错误，却无法穷举 32 位输入空间。`decoded/sass-bitsem/` 因此使用 Lean 4 将部分 SASS 位操作写成纯 `BitVec` 函数，让位宽进入类型系统。

以 `LOP3` 为例，三个输入位共同选择一个三位索引，索引指向 8 位 LUT 中的某一位。对 32 位操作数，每个位独立应用同一真值表。仓库在 `Logic.lean`[21] 中把八个 minterm 完整展开，并证明：

| LUT 常量 | 等价表达式 |
| --- | --- |
| `0xF0` | 选择输入 A |
| `0xCC` | 选择输入 B |
| `0xAA` | 选择输入 C |
| `0xC0` | `A AND B` |
| `0xFC` | `A OR B` |
| `0x96` | `A XOR B XOR C` |
| `0xE8` | 三输入多数函数 |

每个定理由 `bv_decide` 完成：目标被 bit-blast 为 SAT 问题，外部求解器产生证明证书，Lean 内核检查证明项。这样得到的结论覆盖全部输入，而非若干随机样本。

同一文件还证明了 funnel shift 的拆分恒等式和连续掩码构造。例如右移 8 位等价于低 32 位右移 8 位，再与高 32 位左移 24 位合并。这类定理可以成为解码器、模拟器和优化器之间的稳定语义契约。

### 5.2 三种验证层次各自解决不同问题

> 这个项目最值得借鉴的创新之处，在于构建了一条层次清晰、互为支撑的验证闭环：三类方法并非孤立使用，而是通过互补性设计，共同收敛于对硬件行为的高置信度建模。

| 层次 | 主要回答的问题 | 层间依赖与局限 |
| --- | --- | --- |
| 静态逆向 | 二进制中存在哪些结构、常量与控制流 | 易受间接调用、编译器优化（如内联）及语义模糊命名干扰；但为后续验证提供初始符号锚点与候选映射假设 |
| 动态差分 | 模型或修改后的程序在真实硬件上是否表现一致 | 覆盖受限于实际输入、执行路径与目标架构；却能为静态分析结果提供实证反馈，修正歧义假设 |
| 形式证明 | 两个位向量函数是否对全部输入恒等 | 依赖前提——即 Lean 中函数定义必须严格对应真实 SASS 指令语义；该对应关系本身无法自动建立，需静态与动态结果联合校准 |

值得注意的是，形式证明本身并不“自动认出”某段 Lean 代码等价于某条 SASS 指令；这一关键语义桥接，必须由反汇编表（提供指令字节到助记符的映射）、动态探针（捕获寄存器/内存状态变化）与硬件差分（比对微架构级行为）协同完成。当三者交叉验证使映射关系足够牢固时，Lean 才能可靠地推演模型内部的代数性质——例如指令组合的交换律、掩码操作的幂等性等。

由此形成的验证链自然递进：**静态逆向提出“可能是什么”，动态差分检验“在真实世界中是否如此”，形式证明则回答“若定义无误，其数学性质是否普适成立”**。:mdnice-bold-italic[三者环环相扣，既规避单一方法的盲区，又将经验观察升华为可验证的抽象结论。]

## 六、项目真正的突破、局限与工程启示

### 6.1 它重建的其实是一套“可执行文档”

> 如果只看页面数量，crucible-notes 很容易被理解为一部超大型逆向百科。更深层的价值在于，文档正逐步变成可执行规格：

- `decode_pool.py` 可以重建加密数据池；
- `sass_decode.py` 可以解析 128 位机器指令；
- `sass_scheduler.py` 可以把控制字还原成依赖 DAG；
- `sass_smt.py` 可以搜索更短的合法调度；
- `sass-bitsem` 可以证明位级恒等式；
- `fatbin` 工具可以拆包、筛选并重组 `.nv_fatbin`。

C 语言实现的 `fatbin_dump.c`[22] 会在输入中扫描 `0xBA55ED50` 容器 magic，验证版本和边界，再按 `entry_size + aligned_data_size` 前进。它体现的同样是“从发现格式到编写兼容实现”的路线。

项目文档声称工具支持 64、80、112 字节 entry header、ZSTD 数据以及 PTX、ELF、LTOIR 等类型，并报告在特定 `libcublasLt` 样本上完成了解包—重打包—再解包的字节匹配。编译入口则很朴素：`fatbin/Makefile`[23] 使用 GCC 和 `libzstd` 构建五个命令行工具。

这意味着仓库正在跨过一个重要门槛：知识不再只供人阅读，还能被程序执行、被样本检验、被后续实现复用。

### 6.2 TPU 与 Neuron 让项目超越 CUDA 单点研究

> `libtpu` wiki 将一个约 745 MB 的 PJRT 插件拆成 18 个部分，覆盖 PJRT 外层 C ABI、内部 Tpu C shim、六代 TPU、编译 pass、LLO ISA、代价模型、VLIW bundle、调度、SparseCore、DMA、运行时和互联。其组织顺序遵循数据依赖：先确定硬件参数，再研究指令、代价和调度。详见 libtpu 依赖链[24]。

AWS Neuron 部分则贯通 `neuronx-cc`、NEFF、`libnrt.so`、DKMS 内核驱动、固件、PJRT、collectives 和 GPSIMD。尤其是 GPSIMD wiki，试图区分 NeuronCore 内多个容易混淆的处理器与固件路径，并把某个 Vision-Q7 配置、寄存器文件、FLIX/VLIW 格式及运行时装载过程串在一起。

### 6.3 最需要警惕的是证据不均与内部漂移

> 项目首页明确写着“AI-GENERATED”“best-guess reconstruction”“not a reliable source”。

仓库中不同板块的证据强度并不均匀：

- 某些 CUDA 结论来自 strip 后二进制的静态分析；
- SASS 调度工具还结合了差分输出和真实 GPU 测量；
- Neuron runtime 部分可利用 GPL 内核源码、符号表或 DWARF；
- 部分 Neuron 横向 wiki 仍标记为 scaffolding；
- TPU 页面同时使用 Confirmed、Inferred、Open 等等级；
- README、站点首页与各 wiki 的页数偶有差异，说明仓库仍在高速更新。

::: block-1
`ptxas` 的 Corrections Log[25] 尤其值得阅读。它:mdnice-bold-italic[记录了 opcode 误标、寄存器类冲突、语义方向颠倒、虚构条目、错误地址和函数身份误判。这里透露出一个重要事实：即便拥有海量反编译材料，局部解释仍会出错；跨页面一致性检查、完整表扫描、权威名称表和动态验证，才是修正推断的有效机制。]
:::

因此，这个仓库适合被当作研究线索、实验平台和待审计规格，不宜直接充当安全关键系统或生产编译器的唯一事实来源。

### 6.4 对编译器与 AI 系统研究者的三点启示

- 第一，**逆向工程必须管理 provenance**。版本、哈希、文件偏移、虚拟地址、来源包与复现命令决定了一条结论能否被第三方重新检查。失去输入身份，再精确的函数地址也只是漂浮的数字。
- 第二，**要把人工发现变成结构化数据和可执行工具**。一张 opcode 表、一份 phase 顺序、一套寄存器类别，如果只写在文章里，很难持续校验；:mdnice-bold-italic[一旦变成 TSV、解析器或测试程序，后续版本差异便可机械比较。]
- 第三，**结论需要证据分级**。字符串直接命名、虚表结构吻合、调用图位置推断、行为差分一致、形式证明成立，代表完全不同的可信强度。把它们统称为“已经逆向出来”，反而会掩盖最关键的信息。

::: block-1
crucible-notes 的根本意义也在这里。:mdnice-bold-italic[它展示了一种研究闭源 AI 基础设施的完整范式：先把巨型二进制拆成可导航的结构，再把结构提炼成数据表和算法模型，继而用真实工具链、硬件实验与证明系统反复校准]。最终留下的成果是一套能够被阅读、运行、质疑、修订和继续实现的知识底座。
:::

当 AI 算力越来越依赖少数封闭编译栈，**理解这些系统怎样把模型变成机器行为**，已经不只是逆向爱好者的技术游戏。:mdnice-bold-italic[它关系到兼容实现、性能诊断、供应链审计、长期可维护性，以及我们是否有能力独立解释一段计算为何以这样的方式运行]。crucible-notes 尚未提供终局答案，它更像一张不断修订的航海图；而在闭源加速器的深水区，一张标明坐标、误差与未知海域的地图，本身就具有罕见的工程价值。

参考资料

[1]

crucible-notes: *https://github.com/GrigoryEvko/crucible-notes*

[2]

README 组件清单: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/README.md#L10-L25*

[3]

站点首页组件表: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/site/index.html#L307-L421*

[4]

PROVENANCE.md: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/PROVENANCE.md#L1-L21*

[5]

cudafe++ 首页: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/cudafe%2B%2B/wiki/src/index.md#L30-L60*

[6]

cudafe++ 执行空间分析: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/cudafe%2B%2B/wiki/src/index.md#L89-L108*

[7]

cicc 总览: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/cicc/wiki/src/index.md#L18-L39*

[8]

cicc 编译流水线: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/cicc/wiki/src/index.md#L71-L111*

[9]

ptxas 三大子系统: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/ptxas/wiki/src/index.md#L35-L62*

[10]

ptxas driver 说明: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/ptxas-driver/README.md#L26-L40*

[11]

nvlink 架构图: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/nvlink/wiki/src/index.md#L40-L114*

[12]

validate_callgraph.py: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/cicc/scripts/validate_callgraph.py#L46-L99*

[13]

`decode_pool.py`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/tools/decode_pool.py#L27-L45*

[14]

decoded/MANIFEST.md: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/MANIFEST.md#L13-L40*

[15]

`sass_decode.py`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-tools/sass_decode.py#L1-L30*

[16]

SASS 解码器字段说明: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-tools/sass_decode.py#L10-L25*

[17]

`sass-tools/README.md`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-tools/README.md#L154-L200*

[18]

`sass_smt.py`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-tools/sass_smt.py#L6-L44*

[19]

调度器 GPU 实测表: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-tools/README.md#L395-L410*

[20]

约束调度结果: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-tools/README.md#L474-L505*

[21]

`Logic.lean`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/decoded/sass-bitsem/SassBitsem/Logic.lean#L15-L48*

[22]

`fatbin_dump.c`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/fatbin/fatbin_dump.c#L42-L131*

[23]

`fatbin/Makefile`: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/fatbin/Makefile#L4-L39*

[24]

libtpu 依赖链: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/libtpu/wiki/src/index.md#L42-L65*

[25]

Corrections Log: *https://github.com/GrigoryEvko/crucible-notes/blob/618ac057085c4c36618a75289d57ed0a0193b0b4/ptxas/wiki/src/methodology.md#L816-L903*

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[击穿 NVIDIA SASS 逆向核心瓶颈！佐治亚理工 CuLifter：让 SASS 二进制可分析、可编译、可跨平台](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902934&idx=1&sn=f64fb74aba33a49e509b5de97bd4acee&scene=21#wechat_redirect)**
- **[RIGEL：逆向拆解 Apple M4 Max GPU 的 Metal 4.1 张量计算通路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902692&idx=1&sn=6c546a2fd3f9fc3960e81c9b98e151f8&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**

:::
