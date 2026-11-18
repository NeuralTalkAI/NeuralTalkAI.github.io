关键词：**模糊测试**、:mdnice-bold-italic[设备覆盖率]、**二进制插桩**、检测解耦、持久化执行

> 一个 CUDA 程序顺利返回，甚至打印了“测试通过”，显存访问就一定合法吗？当计算进入 GPU，主机端，也就是 CPU 一侧，能观察到的控制流往往只剩下内存申请、数据复制和内核启动。

实际决定结果的条件分支藏在设备代码里，越界访问也可能没有立即变成主机进程崩溃。**传统模糊测试因此同时遭遇两道障碍：不知道哪些输入值得继续探索，也无法仅凭退出状态判断错误是否发生。**

![文章插图](/articles/2026-10-26-nvidia-cufuzz-gpu-21d7868b/images/01.webp)

- :mdnice-bold-italic[cuFuzz, a GPU-oriented coverage-guided fuzzer for userland CUDA applications. cuFuzz combines host-side and device-side coverage collection with sanitization to effectively discover bugs in CUDA programs.]
- https://github.com/NVlabs/cuFuzz

::: related-title
**相关推荐**
:::

::: related-links
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[约束求解赋能 GPU 模糊测试工具 GPU-Fuzz：挖掘深度学习框架 GPU 高危内存错误](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899663&idx=1&sn=41831f6311c438513d5c1732a2e67989&scene=21#wechat_redirect)**
- **[编译器测试新利器 MLIR-Forge：拼图解耦 IR 测试，让随机程序生成变得简单高效！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896753&idx=1&sn=cfa8a4376a322e46a41b38d668c7502f&scene=21#wechat_redirect)**

:::

> cuFuzz 把这两件事拆开处理：借助 NVBit 在设备二进制中插入覆盖率记录，将 GPU 执行信息合并到 AFL++ 的反馈位图；再通过 SAND，把选中的输入交给独立的错误检测通路。持久化模式进一步复用执行进程，摊薄 CUDA 初始化与插桩成本。

*补充一些信息：AFL++ 是 AFL（American Fuzzy Lop）的延续项目：它不断变异输入，并优先保留触发新代码路径的样本。覆盖率就是这类路径是否被执行的反馈，位图则用固定大小的字节数组记录这些反馈。*

*NVBit 是 NVIDIA 的动态二进制插桩工具，可以在 CUDA 设备指令执行前加入记录逻辑，无需逐个修改内核源码。SAND 是 AFL++ 中把覆盖率运行与 sanitizer（运行时错误检测器）分开执行的机制。*

![文章插图](/articles/2026-10-26-nvidia-cufuzz-gpu-21d7868b/images/02.webp)

::: image-caption
上图是 cuFuzz 面向 GPU 库的模糊测试（Fuzzing）工作流程：首先初始种子送入基于 AFL++ 的模糊引擎，输入变异器生成新测试输入，传给模糊测试桩，调用被测库的 CPU 主机代码并触发 GPU 设备内核执行。分别通过 Clang、NVBit 采集主机与设备侧代码覆盖率，更新覆盖率位图；若发现新代码覆盖，该输入放回输入队列继续变异。测试输入同时送入启用消毒器的程序做错误检测，一旦捕获错误就记录到漏洞列表，无错误则丢弃样本，持续迭代挖掘 GPU 库缺陷。
:::

::: block-1
:mdnice-highlight[本文沿着输入、插桩、反馈、检查和状态重置这条链路，拆解其中最关键的实现]，同时说明覆盖率压缩、并发行为和公开评测材料的适用边界。
:::

## 本文目录

- 一、快速上手：先验证 GPU 反馈确实接通
- 二、架构总览：一份输入，分工明确的两条通路
- 三、设备插桩：把 GPU 基本块变成搜索路标
- 四、反馈融合：把大规模执行计数压进一张小地图
- 五、检测解耦：把昂贵检查花在选中的输入上
- 六、持久化执行：复用进程，也必须划清输入边界
- 七、验证与边界：工具在运行，不等于反馈链路有效
- 结语：让反馈可用，比堆叠工具更重要

![文章插图](/articles/2026-10-26-nvidia-cufuzz-gpu-21d7868b/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[PROPILOT：面向 AI 编译器的张量不变量编译测试新范式，削减 49%冗余测试用例，摆脱 LLM 测试幻觉！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902734&idx=1&sn=9867a1dc58b8d7025c8a0a9e4a932300&scene=21#wechat_redirect)**
- **[72 卡 GB200 的 NVLink 互联怎么测？NVIDIA 开源 MNNVL fabric 基准测试工具 nvloom](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904392&idx=1&sn=98f83575f3df05231538d9d678d82993&scene=21#wechat_redirect)**
- **[对标 A100：拆解摩尔线程开源的 GPU 驱动级基准测试 gpu-compute-driver-bench](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901875&idx=1&sn=3db0fb835b96a96286fc1e65ef6492e2&scene=21#wechat_redirect)**

:::
