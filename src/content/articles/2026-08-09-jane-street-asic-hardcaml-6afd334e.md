关键词：**ASIC 逆向工程**、GDS 版图、Hardcaml、**开源 PDK**、:mdnice-bold-italic[Jane Street]

![文章插图](/articles/2026-08-09-jane-street-asic-hardcaml-6afd334e/images/01.webp)

- **Can you reverse engineer an ASIC?**
- https://blog.janestreet.com/can-you-reverse-engineer-an-asic/
- https://github.com/janestreet/asic-puzzle-2026

**不相关推荐**

- **[hls4ml：面向 FPGA AI 部署的开源模块化编译器，实现微秒级延迟、极致资源效率，支持 PyTorch/Keras/ONNX，适配 AMD/Intel/ASIC](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897821&idx=1&sn=6f4033c3aa72e8c73d802f921164d134&scene=21#wechat_redirect)**
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[RIGEL：逆向拆解 Apple M4 Max GPU 的 Metal 4.1 张量计算通路](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902692&idx=1&sn=6c546a2fd3f9fc3960e81c9b98e151f8&scene=21#wechat_redirect)**

> Jane Street 最近放出一道 :mdnice-bold-italic[ASIC 逆向谜题：只给芯片版图（GDS 文件），没有任何标签，要求从几何图形中恢复网表、推断电路功能，最终找到让 `success` 信号拉高的输入字符串]。GDS 是代工厂实际用来造芯片的文件，里面包含了晶体管、金属层、通孔的全部几何信息。

谜题文件托管在 GitHub 仓库 `janestreet/asic-puzzle-2026` 中，附带一个二进制加法器的热身设计，以及开源工具链（KLayout[1]、Magic VLSI、Surfer[2]）和 SkyWater 130nm 开源 PDK。

提交截止 2026 年 9 月 4 日，Jane Street 还预告了年底的"设计你自己的芯片"竞赛。原 blog[3] 后半段重点介绍了他们内部使用的 Hardcaml——一个基于 OCaml 的硬件设计框架，2022 年在 ZPrize 密码学竞赛 FPGA 赛道拿过第一名。

## 本文目录

- 一、从芯片逆向说起
  - 1.1 谜题背景
- 二、GitHub 仓库内容
  - 2.1 核心文件
  - 2.2 热身设计
  - 2.3 工具链
- 三、解谜提示
- 四、Hardcaml：Jane Street 的硬件设计框架
- 结语

## 一、从芯片逆向说起

> 国外在芯片逆向这块其实有不少公开研究。一些团队会从物理版图反推网表，再还原出原始电路的功能。这类工作对理解竞品架构、验证设计思路都有价值。

Jane Street 这次直接把一颗 ASIC 的完整版图放出来让人解，算是把这类工作做成了一个公开的智力游戏。

### 1.1 谜题背景

> Jane Street 去年 2 月发布过一道神经网络谜题：Can you reverse engineer our neural network?[4]——给你一个完整的网络，要求推断它的功能。那道题的反响很好，所以他们今年又出了一道，方向换成了芯片。

::: block-1
GDS 文件就是芯片的"最终形态"。从 Verilog 源码综合成网表，再布局布线，最后输出 GDS——代工厂拿这个文件就能造出物理硅片。问题是 GDS 里没有任何标签，只有一堆多边形和金属层。你的任务是从这些几何图形中恢复出电路的网表，推断它到底在做什么，然后找到正确的输入让 `success` 信号拉高。
:::

![文章插图](/articles/2026-08-09-jane-street-asic-hardcaml-6afd334e/images/02.webp)

::: image-caption
上图一张数字电路仿真波形图，展示 clk 时钟、rst_n 复位、enable 使能、input 输入等信号，经过标注为 output generator 的输出生成模块，产生 out [0]‑out [7] 八路输出信号，右上角 success 信号标记测试成功。图中密集彩色线条代表信号随时间的电平跳变，橙色竖线是时钟边沿，可观察复位、使能、输入变化如何影响 8 位输出，用于验证该输出生成模块在时钟驱动下，对输入的响应逻辑是否符合预期，整体用于数字逻辑的功能仿真调试。
:::

这里有个值得琢磨的点：Jane Street 是一家做市商，核心资产是低延迟交易系统。他们的硬件团队日常做的是为交易系统定制 FPGA/ASIC 加速器，把延迟压到纳秒级。这类工作的常态就是"盯着一堆没有文档的波形、时序报告和版图，慢慢理清发生了什么"。**这道谜题模拟的正是这个场景——给你一堆没有标签的多边形，想办法让它"说话"。从这个角度看，谜题不只是游戏，也是他们工作文化的切片。**

## 二、GitHub 仓库内容

> 谜题文件在 janestreet/asic-puzzle-2026[5] 这个仓库里。

### 2.1 核心文件

- `puzzle.gds`：芯片版图本身，可以用 KLayout[6] 或 TinyTapeout[7] 的在线查看器打开
- `example_inputs.vcd`：一些输入信号的波形示例，但注意——这些不是正确答案，只是给你看看输入长什么样
- 一张带注释的版图截图，标出了 `clk`、`rst_n`、`enable`、`input`、`success` 和 `out[0..7]` 的位置，右侧有个黑框标着"output generator"——这块初始阶段可以忽略，但最后模拟答案时需要用到

### 2.2 热身设计

> 仓库里还有个热身设计：一个简单的二进制加法器，完整走了一遍从 Verilog 源码到最终 GDS 的流程。文件按步骤编号：

- `00_source.v`：原始 Verilog
- `01_netlist.v`：综合后的门级网表
- `02_netlist_with_power_rails.v`：加了电源轨的网表
- `03_post_place_and_route.def`：物理布局和布线
- `04_final.gds`：最终可制造版图，内部名称已移除

设计本身很简单：两个移位寄存器、一个加法器、一个比较器，当 `A + B == 496` 时输出 `success`。目的是让你熟悉"从代码到版图"的完整流程，以及工具链怎么玩。

::: block-1
这个热身展示了一个完整的正向映射：源码长什么样，对应的版图就长什么样。反过来，这个映射关系正是解谜时需要逆向的直觉训练。Jane Street 的硬件团队日常也在处理类似的"逆向"问题——面对一堆没有文档的遗留代码或第三方 IP，需要快速理解、修改、集成。热身设计的存在，暗示了他们想要的人不是"只会写 Verilog 的工程师"，而是**能在代码和物理实现之间自由切换的人**。
:::

### 2.3 工具链

工具方面都是开源的：KLayout、Magic VLSI 看版图，Surfer 看波形。设计用的开源 PDK，应该是 SkyWater 130nm 那套。

## 三、解谜提示

> Jane Street 给了几个提示：

- 电路的物理排布本身就在暗示功能，别只看逻辑，看布局
- 需要自己想个办法模拟底层电路来测试解法
- 每次尝试前记得切 `rst_n`
- 电路和仓库里藏了彩蛋，解完主谜题可以找找

::: block-1
"电路的物理排布暗示功能"这句话很关键。:mdnice-bold-italic[数字电路布局通常有规律：总线会排成平行的金属线，运算单元会聚集在一起，存储阵列呈现规则的网格。这些视觉模式是逆向的突破口。Jane Street 特别强调这一点，可能是因为他们的交易硬件里]，**布局本身就直接关联时序和延迟** ——一个运算单元放得太远，信号传播时间就会吃掉宝贵的纳秒。所以他们培养的工程师，天然会"用眼睛读版图"。
:::

提交截止是 **2026 年 9 月 4 日**。:mdnice-bold-italic[他们说不给提示，但欢迎发邮件问非提示类问题]。解完后可以提交答案和简要说明，最有趣的解法会在后续博文中展示。

另外他们提了一句：今年晚些时候会办一个更大的竞赛——不是又一个逆向工程，而是**设计你自己的芯片**，优秀作品会实际流片。这个可以留意一下。

![文章插图](/articles/2026-08-09-jane-street-asic-hardcaml-6afd334e/images/03.webp)

::: image-caption
上图展示芯片从硬件描述语言到成品芯片的完整流程。首先用 Verilog 这类硬件语言编写电路代码，经综合转化为由门电路组成的网表；接着通过布局布线，把网表转换成 GDS 版图文件，版图定义金属、多晶硅栅、扩散层等物理层细节，晶体管由多晶硅与扩散层交叉处形成；最后依据 GDS 完成晶圆制造与封装，最终产出实体芯片，实现从逻辑设计到物理硬件的落地。
:::

> 这个预告值得玩味。流片成本不低，为什么愿意花这个钱？

- 一种可能是他们想观察社区在开源 PDK 上的设计能力，测试 SkyWater 130nm 等开源流程的成熟度，为内部项目做技术储备。谜题是"低成本探路"，竞赛是"高投入验证"。
- 另一种可能是品牌延伸——Jane Street 在 OCaml 社区已经投入了大量开源资源，这次想在硬件领域复制同样的影响力。用技术影响力降低招聘成本，吸引主动投递，这是他们一贯的路数。

## 四、Hardcaml：Jane Street 的硬件设计框架

> 原文后半段介绍了 Hardcaml[8]，这是他们开源的硬件设计库。简单说几点核心的：

Hardcaml 是一个**嵌入在 OCaml 中的硬件描述语言 DSL**。Jane Street 公司内部有超过 8000 万行 OCaml 代码，:mdnice-bold-italic[从交易系统到基础设施全是 OCaml 写的。Hardcaml 让硬件团队能用同一门语言做 RTL 设计，减少了跨语言协作的摩擦。这种"统一技术栈"的策略，在大型工程团队里很常见——但 Jane Street 把它做到了极致，连编译器都自己 fork 了一个 OxCaml。]

![文章插图](/articles/2026-08-09-jane-street-asic-hardcaml-6afd334e/images/04.webp)

Hardcaml 最大的卖点是 OCaml 的类型系统。在 Verilog 里，位宽不匹配通常不会报错，而是静默截断或补零，这种 bug 很难抓。Hardcaml 在编译阶段就会检查：浮空信号、多驱动、位宽不匹配——这些错误直接拦在代码编译时，而不是等到仿真或上板才发现。对于做高频交易硬件的团队来说，这个特性也确实是"生死攸关"——交易系统的 FPGA/ASIC 一旦上线，bug 的代价是真金白银。

Hardcaml 不是玩具，早在2022 年 ZPrize 密码学竞赛中，Jane Street 团队用 Hardcaml 写的 MSM（多标量乘法）加速器在 FPGA 赛道拿了第一名，NTT（数论变换）赛道拿了第二名。ZPrize 参赛团队来自学术界和工业界，**Jane Street 拿第一，证明 Hardcaml 在性能上不输传统 HLS 或手写 Verilog**。但更有意思的是参赛动机：密码学加速跟高频交易没有直接关系，他们参加竞赛，更像是在**验证 Hardcaml 的极限能力**，同时获取社区反馈——这跟开源 Hardcaml 本身的逻辑一致，也跟这次 ASIC 谜题的思路一脉相承：用外部场景测试内部工具，用社区互动扩大技术影响力。

## 结语

> 原文提到了实习岗位、访问研究员计划和研究生奖学金。表面是发谜题，底层还是招人。不过咱们这儿不是帮他们招聘，就是看看这道题本身有什么门道，以及他们内部用的 Hardcaml 框架长什么样。

版图逆向这个方向，国内也在做。Jane Street 把它包装成公开谜题，算是个不错的参考。Hardcaml 这种"用高级语言写硬件"的思路，跟 Chisel、SpinalHDL 类似，但 OCaml 的类型系统确实给了它一些独特的优势。有兴趣的可以翻翻他们的 GitHub 仓库[9]和 2023 年那篇 Hardcaml: An OCaml Hardware Domain-Specific Language for Efficient and Robust Design[10]，再就是前文提到的比赛的文章 Hardcaml MSM: A High-Performance Split CPU-FPGA Multi-Scalar Multiplication Engine[11]。

::: block-1
Jane Street 说"不要把谜题文件直接喂给 AI"，但允许用 AI 写脚本。这个立场很微妙——一方面承认 AI 工具的价值，另一方面强调"人类理解"不可替代。**在高频交易硬件领域，AI 目前确实无法替代人类对时序、功耗、布局的直觉判断。**

这道谜题某种程度上是在捍卫"硬核技术人"的价值：是对底层的真正理解。:mdnice-highlight[这种立场，跟他们用 OCaml 而不是 Python、用 Hardcaml 而不是 HLS 的选择，底层逻辑是一致的——信任抽象，但不放弃控制]。
:::

参考资料

[1] KLayout: *https://www.klayout.de/*

[2] Surfer: *https://surfer-project.org/*

[3] 原 blog: *https://blog.janestreet.com/can-you-reverse-engineer-an-asic/*

[4] Can you reverse engineer our neural network?: *https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/*

[5] janestreet/asic-puzzle-2026: *https://github.com/janestreet/asic-puzzle-2026*

[6] KLayout: *https://www.klayout.de/*

[7] TinyTapeout: *https://gds-viewer.tinytapeout.com/*

[8] Hardcaml: *https://hardcaml.org/*

[9] GitHub 仓库: *ttps://github.com/janestreet/asic-puzzle-2026*

[10] Hardcaml: An OCaml Hardware Domain-Specific Language for Efficient and Robust Design: *https://arxiv.org/abs/2312.15035*

[11] Hardcaml MSM: A High-Performance Split CPU-FPGA Multi-Scalar Multiplication Engine: *https://dl.acm.org/doi/10.1145/3626202.3637577*
