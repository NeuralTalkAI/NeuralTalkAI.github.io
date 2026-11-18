关键词：**GPU 二进制提升**、:mdnice-bold-italic[SASS 逆向]、**类型恢复**、LLVM IR、:mdnice-bold-italic[SIMT SSA]

> 我们证明类型恢复是 GPU 二进制提升（GPU binary lifting）的核心难题。

这句话直接戳破当前 GPU 逆向领域最核心的底层矛盾，也是 CuLifter 整套技术体系的设计原点。**当下大模型、科学计算依赖 cuDNN、TensorRT 等闭源 CUDA 库，厂商仅下发编译完成的 SASS 机器码，无源码、无 PTX 中间文件，传统二进制分析手段完全失效**。

::: block-1
CPU 成熟的 McSema、rev.ng 等工具依托分离浮点/整数寄存器获取天然类型线索，:mdnice-bold-italic[但英伟达 SASS 使用单一统一寄存器堆，同一个物理寄存器可交替存储 32 位浮点数、32 位整数、指针，无任何硬件标记区分数据种类，直接阻断 LLVM IR 生成通路。]
:::

![文章插图](/articles/2026-06-26-nvidia-sass-culifter-sass-c659e77f/images/01.webp)

- **CuLifter: Lifting GPU Binaries to Typed IR**
- https://arxiv.org/pdf/2604.27486

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894524&idx=1&sn=0282673a4d170d4c5a14a563ad311d78&scene=21#wechat_redirect)**

:::

> 佐治亚理工团队提出的 CuLifter 完整流水线，其核心洞察是 **GPU 二进制提升本质是三重丢失结构的修复工程：数据类型、显式控制流、高层运算语义**。工具通过约束传播+冲突检测完成寄存器类型推理，适配 SIMT 谓词执行重构 SSA 形式，匹配多指令模式还原编译器折叠的高层算子。

![文章插图](/articles/2026-06-26-nvidia-sass-culifter-sass-c659e77f/images/02.webp)

::: image-caption
图 2：CuLifter 二进制提升流水线：SASS 经三阶段转换输出 LLVM IR，依次为控制流恢复、语义操作归一、类型恢复与 IR 生成。流水线顺序不可调换，控制流恢复作为首步，是构建 def-use 依赖图、SSA 单赋值形式的基础，缺少完整 CFG 会导致后续模式匹配、类型传播依赖断裂；中间语义归一化将硬件拆分的多指令原语还原高层运算，避免类型分析拆分子操作产生误判；末尾类型恢复在规整 SSIR 上完成定点迭代。前两阶段包含 SM75~SM120 架构专属解析逻辑，类型推理、LLVM 生成完全架构无关，新增 GPU 世代仅需扩充指令模板，无需重构核心分析算法，分层解耦架构是工具兼容多代 NVIDIA GPU 的关键工程设计。
:::

在覆盖线性代数、大模型推理、仿真等 10 大领域共 24437 个 GPU 函数测试中，整体有效提升率达到 99.98%；:mdnice-bold-italic[消融实验给出决定性结论：无论仅关闭类型约束传播、还是彻底移除整套类型恢复模块，CPU 仿真执行通过率都会直接从基线 73.8% 跌至 0，证明类型推理是整个逆向链路不可替代的核心基石。]

![文章插图](/articles/2026-06-26-nvidia-sass-culifter-sass-c659e77f/images/03.webp)

::: image-caption
表 6：各基准套件函数提升成功率统计表：全数据集 24437 个 GPU 函数仅 6 个提升失败，全部来自 CUTLASS 库 SM90 QGMMA 未实现张量指令，其余套件 100% 成功，整体 99.98% 提升率证明工具具备工业级可用性。cuBLAS、cuDNN 等闭源厂商大型库可完整解析，说明 CuLifter 适配量产高度优化 SASS 二进制。失败仅为新型张量指令缺失解析模板，不改动核心类型、控制流算法，仅需扩充指令规则即可修复，提升框架底层稳定。评测覆盖 ML、线性代数、科学计算等主流 GPU 负载，验证工具可用于逆向商用闭源 CUDA 库二进制。
:::

::: block-1
:mdnice-bold-italic[该工具原生支持 SM75、SM90、SM100、SM120 英伟达架构（兼容老旧 SM35/SM52），逆向输出标准 LLVM IR，可直接用于闭源内核访存缺陷分析、x86/ARM 跨平台移植]，补齐了 GPU 二进制分析长期缺失的底层基础设施。
:::

## 本文目录

- 一、行业痛点：SASS 二进制逆向长期无解的底层根源
- 二、CuLifter 核心洞察：GPU 逆向是三重丢失结构的修复工程
- 三、核心技术一：基于格约束传播的类型恢复算法
- 四、核心技术二：适配 SIMT 谓词执行的 ψ-SSA 重构
- 五、核心技术三：多指令模式聚合与隐式寄存器补全
- 六、CuLifter 完整三阶段流水线全拆解
- 七、大规模基准实验：99.98%提升率数据深度解析
- 八、工程落地两大核心下游应用场景
- 九、当前工具局限与未来拓展方向
- 十、领域深远产业与学术影响

![文章插图](/articles/2026-06-26-nvidia-sass-culifter-sass-c659e77f/images/04.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894524&idx=1&sn=0282673a4d170d4c5a14a563ad311d78&scene=21#wechat_redirect)**

:::
