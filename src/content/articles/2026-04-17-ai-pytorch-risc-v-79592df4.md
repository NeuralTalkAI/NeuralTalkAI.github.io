关键词：:mdnice-bold-italic[RISC-V]、**PyTorch**、CI/CD、**RVV 向量化**、RISE 项目

> 在刚刚落幕的 PyTorch Conf EU 2026 上，Meta 软件工程师、RISE 项目技术指导委员会联合主席 Ludovic Henry，:mdnice-bold-italic[用一句“So... how do you start?”的质朴发问，撕开了新兴硬件生态最关键的命题：如何让 RISC-V 这一潜力架构，真正承接 PyTorch 这座 AI 框架基石？]

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/01.webp)

作为开源软件与新兴硬件交叉领域的深耕者、RISC-V 生态的关键推动者，Ludovic 始终聚焦 AI 软件栈在 RISC-V 上的性能突破与稳定性落地。

::: block-1
他带着近期重磅实践登台：**深度打磨 OpenBLAS、oneDNN 等原生依赖库以释放 RVV 向量扩展潜力，主导搭建 RISC-V 专属 PyTorch 测试基础设施，更系统性拆解 “编译、测试、发布” 三重技术壁垒** —— 不靠 “一等公民” 的口号，只交付可运行的代码、可复现的测试、可持续的发布流程。
:::

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/02.webp)

- **PyTorch on RISC-V: From Cross-Compilation to Native CI**
- https://github.com/riseproject-dev/riscv-runner
- 4000 字，阅读 20 分钟，**播客 20 分钟**

相关推荐

- **[从校园项目到行业挑战者：RISC-V 如何用开放重塑芯片世界？](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896324&idx=1&sn=1510ede4b184eb24ce39c4c5a9bcfc38&scene=21#wechat_redirect)**
- **[超越英伟达 B200 19%计算密度：7nm 四 Chiplet RISC-V 开源架构——面向 AI/HPC 的代际进化路线图](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894155&idx=1&sn=a148162fc540c7a1a4677e210c439dc4&scene=21#wechat_redirect)**
- **[1 核抵 8 核！解耦 RISC-V GPGPU 的控制流和数据访问：Vortex GPGPU 从 0.35 到 1.63 GFLOP/s/mm² 的性能飞跃](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896391&idx=1&sn=83dedbba5262bbea3b8fcd0b5d6b0659&scene=21#wechat_redirect)**

> 这场演讲不止是一次技术迁移的复盘：它直面 RISC-V 适配 PyTorch 的核心痛点，既解答了 RVV 向量扩展如何与动态图执行模型协同、算子优化如何上游化等深层技术问题，也破解了 NumPy/SciPy 等依赖库的“最后一公里”兼容难题，更给出了从交叉编译到原生硬件测试池的完整演进路线图。

这不仅填补了 RISC-V 在主流 AI 框架支持上的关键空白，更是一份开源硬件从“可用”走向“好用”的行业实操指南。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/03.webp)

::: image-caption
Ludovic Henry 将介绍解决问题的方案、踩坑经验与有效 / 无效路径，并坦诚 “尚未完全成功”，下方 “编译→测试→发布” 的流程也呼应了项目从适配到落地的完整工程路径，为这场硬核实践分享定下了务实、不浮夸的基调。
:::

## 本文目录

- 一、战略支点：为何 PyTorch 与 RISC-V 的相遇至关重要
  - 1.1 RISC-V 的生态位与 PyTorch 的试金石效应
  - 1.2 从云端到边缘的硬件现实
- 二、冰山之下：跨越编译鸿沟的工程艺术
  - 2.1 地基重构：从操作系统到底层运行时
  - 2.2 Python 世界的最后一公里
- 三、信任锚点：从“能编译”到“跑得稳”的 CI 进化
  - 3.1 RISE Runners：打破硬件获取壁垒
  - 3.2 漫长的绿化之旅
- 四、性能解耦与未来图景：走向原生加速
  - 4.1 硬件加速的向量化路径
  - 4.2 边缘侧的降维打击：ExecuTorch
  - 4.3 社区驱动的飞轮效应

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/04.webp)

::: image-caption
RISE 联盟成员墙，云集了 Meta、Google、高通、英伟达、联发科等科技巨头与国内达摩院、中兴等企业，直观展现 RISC-V 软件生态的阵营实力，印证 Ludovic 推动 PyTorch 适配并非单打独斗，而是全行业协同打通开源软件与新兴硬件的集体行动。
:::

## 一、战略支点：为何 PyTorch 与 RISC-V 的相遇至关重要

> 在深入具体的技术攻坚之前，Ludovic Henry 首先勾勒了 RISC-V 在当前计算版图中的位置以及 PyTorch 所承载的象征意义。这不是一次普通的软件适配，而是开源软件栈与开源指令集架构的一次顶峰对齐。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/05.webp)

::: image-caption
多年生态建设成果：已生成 linux-noble-riscv64-py3.12 编译产物（258MB），在原生 RISC-V 环境完成矩阵乘法测试，输出正确张量形状与数值。用真实运行结果证明，PyTorch 已能在 RISC-V 上基础执行 AI 计算，达成编译阶段核心目标。
:::

### 1.1 RISC-V 的生态位与 PyTorch 的试金石效应

> RISC-V 并非某家公司的私有财产，而是一个**开放标准的指令集架构**。其“免版税、开源、社区驱动”的特质，并已有数十亿颗芯片基于该架构出货。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/06.webp)

从嵌入式微控制器到数据中心级处理器，RISC-V 的触角正在迅速延伸。:mdnice-bold-italic[然而，一个架构能否在算力要求苛刻的 AI/ML 时代站稳脚跟，关键取决于其对主流框架的支持程度。]

::: block-1
**在这一背景下，PyTorch 的角色被提到了战略高度**。作为 RISE 项目的旗舰级项目，:mdnice-bold-italic[PyTorch 在 RISC-V 上的运行表现]，被定义为检验该平台是否“准备好应对 AI/ML 工作负载”的**试金石**。
:::

如果开发者无法在 RISC-V 服务器上流畅运行 PyTorch，那么所有关于开放硬件在数据中心替代 x86 或 ARM 的畅想都将是无源之水。Ludovic Henry 的演讲逻辑清晰地表明：RISE 社区的目标不仅是让代码跑通，而是要:mdnice-bold-italic[通过优化 PyTorch 这一关键负载，反向驱动 RISC-V 底层硬件特性（如向量扩展）的成熟与工具链的完善。]

### 1.2 从云端到边缘的硬件现实

> 为了证明这场迁移并非纸上谈兵，下图展示了当前的硬件基础——**Scaleway EM-RV1 云实例**。这是一款基于真实 RISC-V 裸金属服务器的商用云产品。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/07.webp)

::: image-caption
Scaleway EM-RV1 裸金属服务器已在 Scaleway 云端正式上线。该硬件是本次 PyTorch 适配、编译、测试的真实物理载体，提供原生 RISC-V 运行环境，支撑从交叉编译验证到原生 CI 测试的全流程工作。
:::

相比于模拟器（QEMU）上的交叉编译验证，真实硬件的存在让后续的 CI/CD 流程具备了物理意义上的可靠性。值得注意的是，虽然当前硬件已可用于基础构建，但 Ludovic Henry 也预告了下一代 **RVA23 标准硬件** 的到来，承诺提供 2-3 倍的性能跃升。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/08.webp)

::: image-caption
展望 RISE RISC-V Runner 升级：将采用下一代 RVA23 硬件，性能提升 2-3 倍，完整支持 RVV；推出 VM 化 Runner，对齐 GitHub 原生体验，完善 setup-python、ccache 等工具链，为 PyTorch、Kubernetes 等旗舰项目提供一站式集成，降低适配测试门槛。
:::

这种代际预告暗示了当前 RISC-V 软件栈正处在黎明前的“算力忍耐期”，开发者正在利用当前硬件完善生态，以待下一代向量化硬件爆发。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/09.webp)

::: image-caption
Ludovic Henry 此处预告 RISC-V 硬件升级：下一代硬件将完全兼容 RVA23，开启 RVV 向量扩展，性能提升 2-3 倍。新硬件让编译更高效，支持向量优化测试，为 PyTorch 在 RISC-V 上的性能提升、大规模测试提供硬件基础，加速生态成熟。
:::

## 二、冰山之下：跨越编译鸿沟的工程艺术

> “Step 1: Make it compile.”。

这句话听起来轻描淡写，但任何有过交叉编译或新架构 bring-up 经验的工程师都明白，这背后隐藏着一座巨大的依赖项冰山。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/10.webp)

::: image-caption
用 “依赖冰山” 可视化 PyTorch 编译复杂度：可见层是 PyTorch、VLLM、ExecuTorch；隐藏层包含 NumPy、SciPy、数学库、GCC/LLVM、Linux 内核、RISC-V 硬件等全栈依赖。只有打通全栈依赖，才能完成 PyTorch 基础编译，凸显第一步工作的系统性难度。
:::

Ludovic Henry 以抽丝剥茧的方式，展示了 RISE 社区过去两年在这一层面的艰苦作业。

### 2.1 地基重构：从操作系统到底层运行时

> 任何框架的运行都建立在稳健的操作系统与工具链之上。在 x86 生态中习以为常的 `pip install`，在 riscv64 架构下却是一场供应链的重塑。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/11.webp)

::: image-caption
PyTorch 编译底层基石：操作系统与工具链。RISE 推动 AlmaLinux 10、RockyLinux 10、manylinux 新增 riscv64 支持，完善 GCC、LLVM/Clang 在 RISC-V 上的适配。该基础层是上层所有软件运行的前提，无此支撑则 PyTorch 无法启动编译。
:::

上面截图拼贴揭示了一幅宏大图景：AlmaLinux、RockyLinux 开始为 riscv64 提供官方支持，许多 linux 与 musllinux 的 wheel 标准被扩展以纳入 riscv64。这不仅仅是重新编译一次内核，而是要让 **GCC/LLVM 能够为 RISC-V 生成正确且高效的代码**，特别是对后续性能至关重要的 RVV 向量扩展支持。

> Ludovic Henry 特别强调了数学库这一层的挑战。

在 AI 运算中，**OpenBLAS** 和 **oneDNN** 是 CPU 推理性能的核心加速库。如果没有针对 RISC-V Vector 扩展的手工汇编优化或 intrinsics 重写，PyTorch 在 CPU 上的计算图将退化为标量运算，性能将低至无法接受的水平。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/12.webp)

::: image-caption
AI 计算核心数学库适配：OpenBLAS 由 RISE 贡献 RVV 支持，SLEEF 提供向量化数学函数，oneDNN 适配是 RISE 重点任务。这些库是 PyTorch 矩阵运算、线性层的底层支撑，其 RISC-V 优化直接决定 AI 计算的正确性与性能，是编译环节的核心工作。
:::

上图确认了 RISE 对 OpenBLAS RVV 支持的贡献，并指出 oneDNN 的适配是当前工作的重中之重。对于具备芯片背景的读者而言，这暗示着社区正在为 RISC-V 定制一套高效的 GEMM 微内核，这是所有上层矩阵运算的物理基础。

### 2.2 Python 世界的最后一公里

> 当 C/C++ 底层库编译通过后，也进入了更为繁琐的“Python 最后一公里”。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/13.webp)

::: image-caption
Python 生态适配最后一公里：PyTorch 依赖的 NumPy、SciPy、ONNX、conda 等所有 wheel 包均需支持 riscv64。RISE 耗时两年完成全依赖栈适配，演示在 riscv64 架构 Ubuntu Docker 中成功安装 maturin，验证 Python 工具链在 RISC-V 上的可用性。
:::

NumPy、SciPy、ONNX、maturin——这些 PyTorch 生态的基石库必须逐个生成适配 riscv64 的原生 wheel 包。Ludovic Henry 在幻灯片中嵌入了一段 Docker 命令行录屏，展示了在 `linux/riscv64` 容器中成功安装 `maturin` 的瞬间。这段看似简单的录屏实则是巨大的里程碑：它意味着**RISC-V 平台已经具备了 Python 原生扩展的构建能力**。

::: block-1
**这一环节最棘手的问题在于解决 ABI 兼容性与二进制分发标准**。由于 RISC-V 的历史包袱较轻，社区得以在一开始就推动 `manylinux_2_31_riscv64` 这类较新的 ABI 标准，避免了 ARM 生态早期碎片化的覆辙。:mdnice-bold-italic[这使得开发者可以像使用 x86 机器一样，直接通过 pip 拉取预编译的轮子，而无需在本地耗尽数小时进行源码编译。]
:::

## 三、信任锚点：从“能编译”到“跑得稳”的 CI 进化

> 跨越了编译的千山万水之后，抛出了一个工程哲学层面的论断：**No CI = no official support**。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/14.webp)

::: image-caption
无 CI 则无官方支持。RISE 提供基于 Scaleway EM-RV1 裸金属的免费 GitHub Actions Runner，环境与 x86/arm64 一致、安全临时。项目仅需配置runs-on: ubuntu-24.04-riscv即可使用，为 RISC-V 项目提供标准化 CI 测试能力。
:::

对于一门成熟的软件工程学来说，软件能否在特定架构上运行，不取决于某次手动编译的成功，而取决于是否有一个持续验证、且结果公开的自动化测试体系。

### 3.1 RISE Runners：打破硬件获取壁垒

> 为了解决开发者手中没有 RISC-V 硬件的问题，RISE 项目推出了具有变革意义的 **RISE RISC-V Runners**。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/15.webp)

这是一种托管在 GitHub Actions 上的免费、裸金属、生命周期短暂的 CI 服务。其架构设计直接对标 GitHub 官方提供的 x86/ARM 运行器。通过简单的 `runs-on: ubuntu-24.04-riscv` 配置，任何开源项目都能获得真实的 RISC-V 算力验证。

这不仅降低了准入门槛，更重要的是建立了一套**标准化的验证流程**。下一代规划——RVA23 硬件与 VM 运行器、`setup-python` 等动作的开箱即用——显示 RISE 正在试图抹平 RISC-V 与主流架构在 DevOps 体验上的最后一丝差异。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/16.webp)

::: image-caption
RISC-V CI/CD 运行器的三大规划：借助 RVA23 硬件实现 2-3 倍性能提升、对齐 GitHub 运行器体验实现开箱即用，以及为 PyTorch 等旗舰项目提供深度集成，勾勒出 Ludovic 团队后续优化 PyTorch RISC-V 生态的技术路线。
:::

### 3.2 漫长的绿化之旅

> 有了硬件，接下来的任务便是所揭示的“Greening the Test Suite”。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/17.webp)

::: image-caption
PyTorch 测试套件优化：通过社区贡献者与承包商协同，系统性修复 RISC-V 下测试用例失败问题，提交 bug、修复补丁并上游合并。核心目标是实现 RISC-V 架构 PyTorch CI 全绿，满足官方支持的核心条件，保障版本迭代稳定性。
:::

PyTorch 拥有数十万个测试用例，在 x86 上通过率极高，**但在新架构上，隐含的未定义行为、端序假设、时间精度差异都会导致测试变红**。Ludovic Henry 描述了团队和外包人员如何系统性地排查失败用例、提交补丁并向上游反馈。这是一个枯燥但绝对必要的净化过程。

从芯片与编译器的角度看，**CI 变绿的过程实质上是软硬件接口规范对齐的过程**。例如，某些测试失败可能并非 PyTorch 代码有误，而是触发了 RISC-V 特定内存模型下的竞态条件，或是 GCC 针对 RVV 向量化生成的代码在边界对齐上存在问题。解决这些问题是反向打磨 RISC-V 软件栈鲁棒性的最佳途径。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/18.webp)

::: image-caption
PyTorch RISC-V 适配的关键抉择：用红色高亮 “Fork or upstream?” 的核心问题，明确团队将与 LF/PyTorch 官方协作，目标是将适配代码合入上游主分支，而非维护永久分支，体现了 Ludovic 团队推动生态兼容、共建社区标准的务实思路。
:::

## 四、性能解耦与未来图景：走向原生加速

在解决可用性问题后，自然过渡到性能与发布策略。

### 4.1 硬件加速的向量化路径

> 下面架构示意图直观展示了 RISC-V 硬件加速栈：**OpenBLAS + RVV** 与 **oneDNN**。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/19.webp)

::: image-caption
标量计算单周期单元素，OpenBLAS+RVV 已达通用最优性能（无微架构调优）；oneDNN+RVV 支持单周期 VLEN 元素，覆盖 GEMM、卷积、池化、归一化等算子，是 PyTorch 在 RISC-V 上性能提升的核心路径。
:::

Ludovic Henry 坦诚地指出，:mdnice-bold-italic[虽然 OpenBLAS 的 RVV 支持已接近最优（针对通用 RVV），但针对特定微架构的调优尚未开始]。这是典型的“先求有，再求好”策略。对于专业听众而言，oneDNN 作为 Intel 开源的深度神经网络加速库，其 pipeline 中的 RISC-V 适配工作预示着未来 PyTorch 在 RISC-V 上的卷积、RNN 等算子将能享受到现代 CPU 的缓存阻塞与指令重排优化。

### 4.2 边缘侧的降维打击：ExecuTorch

> 除了数据中心场景，下面还特意点名了 **ExecuTorch**。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/20.webp)

::: image-caption
边缘 AI 场景：ExecuTorch 是 PyTorch 边缘轻量化方案，RISC-V 在嵌入式领域优势显著，二者天然融合。边缘端是 RISC-V+PyTorch 生态的重要前沿，完善从数据中心到终端的全场景 AI 部署能力。
:::

PyTorch 在边缘侧的部署利器与 RISC-V 在嵌入式领域的天然优势在此形成了战略交汇。对于 AI 芯片从业者来说，这意味着未来大量的端侧 AI 推理芯片（基于 RISC-V IP 核）将拥有一个极其丝滑的模型部署工作流——模型在云端 PyTorch 训练，导出为 `.pte` 文件，直接运行在指甲盖大小的 RISC-V MCU 上。

### 4.3 社区驱动的飞轮效应

> 在演讲的尾声，Ludovic Henry 发出了具体的行动呼吁，并描绘了下图所展示的“良性循环”：**更多的库迁移 → 更丰富的 CI 验证 → 更稳定的平台体验 → 吸引更多开发者 → 发现更深层性能瓶颈 → 驱动硬件改进**。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/21.webp)

::: image-caption
Ludovic Henry 倡导社区协同，构建生态良性循环。呼吁开发者：将库移植到 riscv64、使用免费 RISE RISC-V Runner 搭建 CI、向上游提交补丁 / 报 bug / 参与测试。RISC-V 上 PyTorch 成熟是必然结果，社区参与将加速全栈落地。
:::

这份演讲最大的价值在于其**诚实性**。它没有宣称 RISC-V 已经超越了 x86，而是通过页码的递进，诚实地展示了底层代码迭代，到最终用户可见的 `pip install` 之间，到底隔着多么浩繁的工程细节。

::: block-1
RISC-V 在 AI 领域的崛起并非一道“是与否”的选择题，而是一个关于“何时到来”的时间函数。

![文章插图](/articles/2026-04-17-ai-pytorch-risc-v-79592df4/images/22.webp)

::: image-caption
RISC-V 成为主流 AI 架构不是 “会不会” 的问题，而是 “什么时候” 的问题
:::

正如演讲最后结语所言——**Not if, but when.** 当编译、测试、发布这三座大山被社区合力移除后，一个真正开放的、无专利壁垒的 AI 算力时代轮廓已然清晰可见。
:::
