关键词：**编译器逆向**、CUDA、TPU、AWS Neuron、**形式验证**

> 闭源 AI 加速器最深的壁垒，在从计算图到机器指令之间那条漫长的软件隧道里。

- NVIDIA CUDA 程序为何被拆成 host 与 device 两路？PTX 经过哪些优化才变成 SASS？
- Google TPU 编译器怎样估算矩阵单元、片上存储与互联的代价？
- AWS Trainium 又如何把 HLO、内核 DSL 和多种中间表示封装进可执行文件？

::: block-1
厂商文档通常只描述入口与结果，:mdnice-bold-italic[真正决定兼容性、正确性和性能的内部数据结构、调度模型及二进制协议仍藏在巨型 ELF 文件中。]
:::

![文章插图](/articles/2026-09-03-nvidia-gpu-google-tpu-aws-30cbbcd5/images/01.webp)

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
- 二、从 CUDA 源码到 GPU 机器码：仓库如何拼回整条编译链
- 三、逆向工程的核心手艺：从匿名函数中建立证据链
- 四、从“看懂”到“能运行”：SASS 工具链如何闭合验证回路
- 五、Lean 形式证明：让逆向语义越过“样本正确”的天花板
- 六、项目真正的突破、局限与工程启示

![文章插图](/articles/2026-09-03-nvidia-gpu-google-tpu-aws-30cbbcd5/images/03.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[击穿 NVIDIA SASS 逆向核心瓶颈！佐治亚理工 CuLifter：让 SASS 二进制可分析、可编译、可跨平台](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902934&idx=1&sn=f64fb74aba33a49e509b5de97bd4acee&scene=21#wechat_redirect)**
- **[RIGEL：逆向拆解 Apple M4 Max GPU 的 Metal 4.1 张量计算通路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902692&idx=1&sn=6c546a2fd3f9fc3960e81c9b98e151f8&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**

:::
