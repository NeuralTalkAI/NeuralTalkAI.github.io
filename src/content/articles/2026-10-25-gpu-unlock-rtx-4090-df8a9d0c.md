**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：**GPU Unlock**、:mdnice-bold-italic[RTX 4090、RTX 3090]、BF16、**大模型推理**

> 能用上 AI Agent 后，REA 等逆向项目自然受到关注，许多人开始逆向低成本或早期 GPU，尝试提升 AI 性能。

最近几天，几个非官方 GPU Unlock 项目集中出现。**实现在 NVIDIA 驱动接管显卡前修改 SM speed-select，让部分消费级显卡提高 FP16、BF16、TF32 的 Tensor Core 运算速率**。

::: block-1
已经公开的结果里，:mdnice-highlight[RTX 3090 Ti 的 BF16 GEMM 提升 79.9%、TF32 提升 94.0%；RTX 4090 的 FP16/FP32 指令吞吐最高测到 2.002 倍]；4070 Ti SUPER 在 240 W 下也把 FP16、BF16、TF32 提到约 2 倍。
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/01.jpg)

::: image-caption
图 1：支持矩阵按公开实测数量和复现范围排序。Linux 系统的 NVIDIA RTX 30 系显卡用户报告最多；Ada 同卡对照正在补齐；RTX 4090 的公开实测只覆盖指定设备 ID；Windows UEFI 组合绑定具体主板和 BIOS。RTX 5060 Ti 暂无同类项目。
:::

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892934&idx=1&sn=5ec357f95854bc9b6ebaa827012de874&scene=21#wechat_redirect)**

:::

> 落到应用层，比方双 RTX 3080 的 vLLM 虽然 decode 基本不动，但 prefill 提升 33% 至 36%。从覆盖的型号来说：

- Zagitta/gpu-unlock[1] 这份面向 NVIDIA RTX 30 系显卡的仓库报告最多；
- JustAResearcher/gpu-unlock-ada[2] 只覆盖少数 Ada 型号；
- impacifist/Windows-GPU-UNLOCK[3] 绑定具体主板和 BIOS；RTX 5060 Ti 暂无可用同类项目。

这些非官方的工具，风险主要集中在启动链、Secure Boot、BitLocker 和回滚准备。因为这些都是这几天刚冒出来的非官方方案，**涉及显卡底层操作，需要关 Secure Boot、改启动参数，存在兼容性和稳定性风险，不建议在主力机上直接试，折腾前最好留好回滚入口。**

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/02.jpg)

::: image-caption
RTX 3090 Ti GEMM 实测：在 450 W、8192³ GEMM 下，BF16 从 87.50 到 157.41 TFLOPS，提升 79.9%；TF32 提升 94.0%。
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/03.jpg)

::: image-caption
JustAResearcher/gpu-unlock-ada 4070 Ti SUPER 实测：在 240 W 下，FP16、BF16、TF32 指令吞吐都接近 2 倍。
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/04.jpg)

::: image-caption
双 RTX 3080 20GB 在 220 W 下，vLLM prefill 提升 33% 至 36%，Strata 只有长上下文提升约 12% 至 13%；decode 未见稳定变化。
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/05.jpg)

::: image-caption
elias-gaeros/gpu-unlock 的 rtx4090 分支在寄存器内测试，FP16/FP32 从 165.6 到 331.6 TF/s，BF16/FP32 从 165.5 到 330.6 TF/s。它测的是指令发射速率，不等于整机应用速度；原始记录只有一张卡和一次成功冷启动。
:::

## 本文目录

- 一、解锁只改部分运算速率，显存和功耗墙没有变化
- 二、核心原理：先打开寄存器权限，再把 speed-select 调到全速档
  - 2.1 speed-select 决定部分 Tensor Core 指令能跑多快
  - 2.2 SS0、SS1 是速率开关，PLM 是开关前面的权限门
  - 2.3 直接写 SS0、SS1 会被忽略，V67 payload 负责把权限门打开
  - 2.4 Zagitta/gpu-unlock 把提权、写值和驱动交接放进同一次启动事务
  - 2.5 为什么能成功：硬件开关本来存在，签名启动链又能被复用
- 三、Zagitta/gpu-unlock：NVIDIA RTX 30 系显卡的用户报告最多
  - 3.1 RTX 3090 Ti 的 BF16 提升约 80%，TF32 提升约 94%
  - 3.2 大模型推理的主要收益在 prefill，decode 基本不变
  - 3.3 量化越重、非矩阵环节越多，端到端收益通常越小
- 四、JustAResearcher/gpu-unlock-ada：Ada 显卡的同卡对照开始补齐
  - 4.1 RTX 4090 同卡复测确认指令接近翻倍，GEMM 增幅明显收窄
  - 4.2 4070 Ti SUPER 的 FP16、BF16、TF32 都接近翻倍
- 五、elias-gaeros/gpu-unlock（rtx4090 分支）：数据最整齐，测试记录最少
- 六、目前只支持这些系统和卡系
  - 6.1 Zagitta/gpu-unlock：有成功报告的主要是 GA102、GA104、GA106
  - 6.2 JustAResearcher/gpu-unlock-ada：新增三组实验目标，范围仍按单卡收窄
  - 6.3 elias-gaeros/gpu-unlock（rtx4090 分支）：RTX 4090 是新增且有实测的目标
  - 6.4 impacifist/Windows-GPU-UNLOCK：已有三套具体到主板和 BIOS 的实测组合
  - 6.5 phiraml/gpu-unlock-blackwell：目前只到 RTX 5090，5060 Ti 仍无同类项目
- 七、潜在风险集中在启动链、恢复入口和后续维护
  - 7.1 Secure Boot、BitLocker 和启动参数都要处理
  - 7.2 启动顺序出错，可能静默失效，也可能让显卡暂时不可用
  - 7.3 睡眠、功耗和长期稳定性仍需测试
- 八、适合先测 BF16 和 prefill，不适合期待所有任务翻倍

## 一、解锁只改部分运算速率，显存和功耗墙没有变化

> 开源项目 Zagitta/gpu-unlock[4] 的 Linux 内核模块会在 NVIDIA 驱动绑定显卡之前运行，写入 SS0、SS1 speed-select，再执行 PCIe Function Level Reset（FLR）。它不刷 VBIOS，也不替换 NVIDIA 驱动。后续驱动升级时，不需要像驱动补丁那样重新修改驱动文件。

**这套操作影响的是 FP16、BF16 等路径的指令发射速率。** 它不会增加 CUDA Core、显存容量、显存带宽和功耗上限。elias-gaeros/gpu-unlock 的 rtx4090 分支对照数据也支持这一点：**FP16、BF16 的 FP32 累加吞吐接近翻倍，** INT8、dp4a 和 scalar FP32 基本保持原样。

::: block-1
因此，期待所有 AI 应用都获得两倍速度并不现实。**计算密集的矩阵乘法和 Attention 更可能受益；** 依赖显存带宽的逐 token decode，以及主要走整数 kernel 的量化推理，收益会小很多。
:::

## 二、核心原理：先打开寄存器权限，再把 speed-select 调到全速档

::: block-1
:mdnice-highlight[这套解锁可以拆成两件事。第一件事是找到显卡里控制 Tensor Core 运算速率的“档位开关”；第二件事是拿到修改开关所需的权限。] SS0、SS1 是开关，PLM 是挡在开关前面的权限检查，V67 payload 和签名 Booter 负责把这道权限打开。
:::

理解这三层关系后，后面的代码路线就比较清楚了。

### 2.1 speed-select 决定部分 Tensor Core 指令能跑多快

> NVIDIA GPU 内部有一组 **feature override（特性覆写）机制，可以调整某些硬件特性的工作状态。** 消费级显卡上，一部分 Tensor Core 指令路径受 speed-select（速率选择）控制。速率档位较低时，FP16、BF16、TF32 搭配 FP32 累加的指令发射速率会被压下来；档位放开后，同样的核心每个周期可以发射更多这类指令。

主机和显卡之间可以通过 PCIe 的 BAR0 窗口读写控制寄存器。BAR0 可以理解成显卡向系统暴露的一段控制面板，内核模块映射这段地址后，就能访问其中的硬件寄存器。CUDA、cuBLAS、vLLM 都运行在更上层，不参与这次修改；它们只在后续运行时感受到指令速率已经变化。

### 2.2 SS0、SS1 是速率开关，PLM 是开关前面的权限门

> Zagitta/gpu-unlock 的 gfw_eval.h[5] 把这套控制面板里的三个位置标了出来：`0x00823804` 对应 FEAT 权限掩码（PLM），`0x0082381c` 对应 SS0，`0x00823820` 对应 SS1。代码真正想改的是 SS0、SS1，PLM 决定主机有没有资格改。

先看 SS0。它里面放着八个 4 位字段，每个字段控制一组速率覆写。4 位里的最高位表示“启用覆写”，低三位表示覆写到哪个档位。十六进制数字 `8` 写成二进制是 `1000`，含义正好是“启用覆写，目标档位为 0”。因此，代码向 SS0 写入 `0x88888888`，就是一次性把八个字段都设成零值覆写；SS1 则写入 `0x8`，对另一个选择字段做同样处理。这里的“零”对应放开后的有效速率，含义是当前路径不再按较低档位运行。

elias-gaeros/gpu-unlock 的 RTX 4090 寄存器报告[6] 还用 `FEAT_READOUT_1` 做了状态读回：解锁后，其中九个 3 位 speed-select 字段全部变成零。这能帮助模块确认开关状态，但每个字段和具体算子路径的完整对应关系还没有公开。SS1 的独立性能作用也没有单独测出，不能把全部增益都归到 SS1 身上。

### 2.3 直接写 SS0、SS1 会被忽略，V67 payload 负责把权限门打开

> 既然最后只是写两个寄存器，为什么系统启动后直接 `writel()` 没有效果？RTX 4090 的实验记录显示，在签名 payload 运行前，主机对 PLM 和 speed-select 的写入会被忽略。换句话说，模块知道开关在哪里，也还没拿到打开开关的权限。

这些项目借用了显卡已有的签名启动链。

SEC2 是显卡内部的安全处理器，Booter 是经过 NVIDIA 签名的启动程序。正常启动时，SEC2 会认证并解密 Booter。V67 payload 是一份随启动过程送入的数据载荷，它会覆盖 Booter 运行时的特定栈槽，让后续返回流程走到 Booter 里已经存在的认证指令。最终完成受保护寄存器写入的仍是那段已经通过签名的 Booter 指令，主机没有向已认证镜像塞入新的可执行程序，也没有重新签名 NVIDIA 固件。

::: block-1
AD102 是 NVIDIA Ada Lovelace 架构的旗舰 GPU 芯片代号，RTX 4090 使用的就是这颗芯片。它的公开 payload 会依次尝试写 WPR_CFG、FBPA、WPR 和 FEAT，并把 mailbox 标记设成 `0x67b00001`。

mailbox 是主机和固件交换状态的通信位置，模块看到标记后再检查 SEC2 停止状态、PLM 读回结果和 speed-select 读回结果。确认权限已经打开，主机后续写 SS0、SS1 才会被硬件接受。**payload 依赖具体 Booter 的指令地址和栈布局，所以换芯片或换固件后不能直接套用。**
:::

### 2.4 Zagitta/gpu-unlock 把提权、写值和驱动交接放进同一次启动事务

> Zagitta/gpu-unlock 的 main.c[7] 负责选择目标显卡。`gpu_unlock_init()` 读取 `gpu_unlock.targets=`，筛选设备 ID，`unlock_one()` 检查显卡还没有绑定 NVIDIA 驱动。满足条件后，代码进入 `gpu_unlock_610_unlock()`。如果驱动已经接管显卡，这条入口会直接退出，这也是手动后装模块通常无效的原因。

真正的事务位于 Zagitta/gpu-unlock 的 unlock_610.c[8]。按代码调用顺序，可以把它读成四步：

1. 准备硬件和固件。代码保存 PCI 状态，映射 BAR0，分配 DMA 缓冲区，并加载固定版本的 GSP、Booter、启动描述和 V67 文件。
2. 建立受保护的固件运行环境。`unlock_transaction_fwsec()` 从显卡 ROM 提取 FWSEC-FRTS，在 GSP Falcon 上运行它，并建立 WPR2 固件保留区。Falcon 是显卡内部的微控制器，WPR2 是供固件使用的受保护内存区域。
3. 打开权限并写入速率开关。代码先发布 V67 payload，再由 `unlock_transaction_boot()` 启动 SEC2 上的签名 Booter；随后恢复原厂签名数据，调用 `write_selector()` 写入 SS0、SS1，并立即读回确认。
4. 清理中间状态并交接驱动。`unlock_transaction_reset()` 执行 PCIe Function Level Reset（FLR），确认 SS0、SS1 在复位后仍然保留，同时检查 WPR2 关闭、GSP 停止、DMA 受控，最后恢复 PCI 状态。

::: block-1
其中，第四步的 FLR 的作用是把提权过程中启动的固件和临时状态收束起来。:mdnice-bold-italic[speed-select 的修改在 FLR 后仍能保留，原版 NVIDIA 驱动随后从一个相对干净的状态初始化显卡。驱动文件没有被替换，CUDA 测试却能看到新的指令发射速率。]
:::

### 2.5 为什么能成功：硬件开关本来存在，签名启动链又能被复用

> 这条路线能跑通，依靠三个条件同时成立。

- 第一，显卡内部本来就有 speed-select 这套速率选择机制，代码没有增加新的运算单元。
- 第二，已有签名 Booter 的指令流程可以被 V67 数据复用，PLM 权限因此能被打开。
- 第三，SS0、SS1 的修改能穿过 FLR 保留下来，原版驱动初始化后仍按新档位运行。

**几个仓库的差异主要在入口和配套检查**。开源项目 elias-gaeros/gpu-unlock 的 rtx4090 分支把这条路线推进到 AD102：它加入 RTX 4090 的设备 ID，选用 AD102 对应的 Booter、启动镜像和 V67 payload，并记录 GA102 Booter 在 AD102 上无法打开 FEAT 权限的反例。这个反例说明，设备 ID 只负责让模块选中显卡，芯片对应的固件资产和 payload 才决定提权能否完成。

开源项目 JustAResearcher/gpu-unlock-ada 的 unlock_610.c[9] 继续沿用同一事务，把资产选择整理到 `unlock_assets()`：GA102 使用原厂 GA10x 组合，RTX 4090、4070 Ti SUPER 和 4060 Ti 16GB 进入 AD102 家族组合。它还加入固件 SHA-256 固定校验、payload 完成标记、四组 PLM 读回，以及 FLR 前后的有效速率检查。AD103、AD106 能进入实验流程，依靠的就是这些资产选择与状态检查；它们仍需精确 PCI 地址和单独实验参数，不能外推到其他 Ada 型号。

开源项目 impacifist/Windows-GPU-UNLOCK 的 app.c[10] 同样调用 `gpu_unlock_610_load_shared()` 和 `gpu_unlock_610_unlock()`。变化主要发生在 port.c[11]：它用 `EFI_PCI_IO_PROTOCOL` 实现 MMIO 读写、DMA 分配、总线主控开关和 FLR，并在 app.c 中加入指定显卡、PROM 哈希、显示路径和手动确认检查。Windows 版本更换了宿主执行环境与硬件适配层，speed-select 和签名固件事务仍沿用同一条技术路线。phiraml/gpu-unlock-blackwell 目前没有公开 payload 和完整解锁事务，因此还没有一条可以复现的 Blackwell 成功调用链。

## 三、Zagitta/gpu-unlock：NVIDIA RTX 30 系显卡的用户报告最多

开源项目 Zagitta/gpu-unlock[12] 是面向 NVIDIA RTX 30 系显卡的 Linux 内核模块，也是这一组项目里用户报告最多的仓库。它已经积累多张 Ampere 显卡的反馈，测试也从 GEMM 延伸到大模型 prefill、decode 和 ComfyUI。不同报告的控制严格程度不一样，**合并来看仍能提供目前最完整的应用侧图景。**

### 3.1 RTX 3090 Ti 的 BF16 提升约 80%，TF32 提升约 94%

> Zagitta/gpu-unlock 的 README 收录了一组 RTX 3090 Ti 测试：8192³ dense GEMM、450 W 功耗限制、FP32 累加，取五次测试中位数。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/06.jpg)

::: image-caption
图 2：RTX 3090 Ti 在 8192³ GEMM、450 W、FP32 累加和五次中位数。FP16 提升 35.4%，BF16 提升 79.9%，TF32 提升 94.0%，可直接比较不同精度受 speed-select 影响的程度。这组数据说明解锁确实改变了部分 Tensor Core 路径的有效吞吐。FP16 没有同步接近翻倍，也提醒读者不能用一个“2×”概括所有精度。FP16 从 87.46 提升到 118.39 TFLOPS，增幅 35.4%；BF16 从 87.50 提升到 157.41 TFLOPS，增幅 79.9%；TF32 从 42.58 提升到 82.61 TFLOPS，增幅 94.0%。（数据来源：开源项目 Zagitta/gpu-unlock[13]）
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/07.jpg)

::: image-caption
图 3：RTX 3090 Ti GEMM 实测，只呈现解锁前后差异。BF16 和 TF32 的柱高差距明显大于 FP16，提醒读者不能用一个“2×”概括所有精度；它仍是 GEMM 结果，不代表所有应用的端到端速度。（数据来源：开源项目 Zagitta/gpu-unlock[14]）
:::

### 3.2 大模型推理的主要收益在 prefill，decode 基本不变

> 真实应用里，开源项目 Zagitta/gpu-unlock 的 issue #15 对照测试[15] 比较有参考价值。测试使用两张改装 20 GB 的 RTX 3080，每卡限制在 220 W，保持脚本、输入、随机种子和推理参数一致，并检查缓存没有复用。

在 vLLM 0.30、27B W4A16 模型、双卡 tensor parallel 环境中，8K、32K、128K 单流输入的 **prefill 分别提升 33%、34%、36%，TTFT 缩短约 25% 至 26%。** 换成 Strata 后，8K prefill 只提升 0.8%，32K 和 128K 提升约 12% 至 13%。两种引擎的 decode 都没有出现超出运行波动的改善。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/08.jpg)

::: image-caption
图 4：双 RTX 3080 20GB 的 issue #15，下方是原厂 RTX 3090 的 issue #25。两张表保留 prefill、decode 和 TTFT 的关键结果，去掉了难以阅读的过宽原始列。（来源：Zagitta/gpu-unlock 的 #15、#25 测试）
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/09.jpg)

::: image-caption
图 5：大模型 prefill 实测，统计图只取 issue #15 的单流结果。vLLM 在 8K、32K、128K 下提升 33% 至 36%；Strata 只有 32K、128K 提升约 12% 至 13%。decode 没有稳定变化，所以没有画成收益柱。（来源：Zagitta/gpu-unlock 的 #15 测试）
:::

**另一张原厂 RTX 3090 在 vLLM、27B BF16 模型、约 2.9K prompt 下测得 prefill 提升 33%，TTFT 缩短 24%，** decode 只变化 1.2%。测试者还记录了 8 小时以上混合负载，包括持续 LLM 服务、图像生成和图生视频，期间没有 hang 和 Xid 错误。这是一段有价值的短期稳定性记录。

### 3.3 量化越重、非矩阵环节越多，端到端收益通常越小

> RTX 3080 的 Issue #14[16] 给出了更具体的应用差异。在两个 INT4 扩散模型上，ComfyUI 热运行平均耗时分别从 9.30 秒降到 7.98 秒、从 9.34 秒降到 8.17 秒，时间缩短 14.2% 和 12.5%，生成图片逐字节一致。同一报告中的 llama.cpp 量化模型，prompt 和 generation 基本都保持原速。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/10.jpg)

::: image-caption
图 6：issue #14 的两个 INT4 扩散模型在模型加载完成后的重复运行时间缩短 14.2% 和 12.5%，四组 llama.cpp 量化模型基本不变，收益取决于受影响矩阵 kernel 的占比。（数据来源：#14 RTX 3080 测试[17]）
:::

RTX 3090 Ti 的 Issue #1[18] 也显示，在一个 INT8 占比较高的 vLLM 分支中，BF16/FP16 kernel 只占约 29% GPU 时间，端到端收益约 5%，且接近运行波动。

这些结果放在一起，结论就清晰了：**先看自己的程序有多少时间花在受影响的 FP16、BF16、TF32 kernel 上，再谈收益。** 只看 GEMM 峰值，很容易高估最终体验。

## 四、JustAResearcher/gpu-unlock-ada：Ada 显卡的同卡对照开始补齐

> 开源项目 JustAResearcher/gpu-unlock-ada[19] 是面向 Ada 显卡的扩展仓库，在 Zagitta/gpu-unlock 的 Linux 启动流程上加入 AD102、AD103 等目标。

按公开实测数量和复现范围，它排在 Zagitta/gpu-unlock 之后，因为补上了 Zagitta/gpu-unlock 和早期 elias-gaeros/gpu-unlock rtx4090 分支都缺少的一类对照数据：同一张显卡、同一驱动、同一功耗限制下的前后测试。它的样本仍少，长期稳定性也还没有测试。

### 4.1 RTX 4090 同卡复测确认指令接近翻倍，GEMM 增幅明显收窄

这组 4090 复测保持同一驱动和 450 W 限制。**FP16/FP32 指令吞吐从 160.4 提升到 320.4 TF/s，BF16/FP32 从 159.7 提升到 320.2 TF/s。**

::: block-1
:mdnice-highlight[更有参考价值的是 cuBLAS GEMM：不同矩阵形状的 FP16 增幅为 12% 至 34.4%，BF16 在 1024³ 和 8192³ 提升约 16% 至 19.6%，4096³ 反而基本持平。] 该复测使用五个内部计时重复，桌面仍保持活动，也不能外推成整机应用提速。这组对照里的 stock 指显卡保持原厂默认状态、未执行解锁时测得的基线。
:::

### 4.2 4070 Ti SUPER 的 FP16、BF16、TF32 都接近翻倍

> JustAResearcher/gpu-unlock-ada 给出的 4070 Ti SUPER 数据同样强调精度差异。该卡在 240 W 下，FP16/FP32 从 95.0 提升到 189.9 TF/s，BF16/FP32 从 95.0 提升到 189.7 TF/s，TF32/FP32 从 47.49 提升到 94.51 TF/s；INT4 指令路径没有变化，一个 E4M3 FP8 CUTLASS GEMM 也只变化 0.3%。JustAResearcher/gpu-unlock-ada 的 4060 Ti 16GB 报告[20] 已记录一次成功冷启动和全速率寄存器读回，**但没有同卡锁定基线，暂时不能写成性能翻倍。**

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/11.jpg)

::: image-caption
图 7：4070 Ti SUPER 与 4090 的 cuBLAS GEMM 比值分开列出。前三条指令路径接近翻倍；4090 的 FP16 最高提升 1.344×，4096³ BF16 为 0.995×。（来源：JustAResearcher/gpu-unlock-ada 报告）
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/12.jpg)

::: image-caption
图 8：统计图只呈现 4070 Ti SUPER 的指令吞吐对照，避免与 4090 GEMM 表数据混排。FP16/FP32、BF16/FP32、TF32/FP32 分别约为 2.00×、2.00×、1.99×。（来源：JustAResearcher/gpu-unlock-ada 报告）
:::

## 五、elias-gaeros/gpu-unlock（rtx4090 分支）：数据最整齐，测试记录最少

> 开源项目 elias-gaeros/gpu-unlock（rtx4090 分支）[21] 是 Zagitta/gpu-unlock 的 RTX 4090 移植分支，:mdnice-bold-italic[目标是让 AD102 在 Linux 启动早期完成同一类 speed-select 修改。 它给出的结果很整齐：FP16 输入、FP32 累加从 165.6 提升到 331.6 TF/s，达到 2.002 倍；BF16 输入、FP32 累加从 165.5 提升到 330.6 TF/s，达到 1.998 倍。] 450 W 功耗限制和 NVIDIA 615.78.08 原版驱动保持不变。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/13.jpg)

::: image-caption
图 9：六组寄存器内指令测试。FP16/FP32、BF16/FP32 接近 2 倍，INT8、dp4a 和 scalar FP32 保持原速；计时循环没有显存读写，所以它衡量的是指令发射速率。（来源：该分支技术报告）
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/14.jpg)

::: image-caption
图 10：RTX 4090 分支指令实测，统计图单独保留变化最大的两组数据。FP16/FP32 从 165.6 到 331.6 TF/s，BF16/FP32 从 165.5 到 330.6 TF/s；（来源：该分支技术报告）
:::

## 六、目前只支持这些系统和卡系

> 支持范围已经在开头的图 1 汇总。这里再按项目拆开说，重点区分“代码允许尝试”和“已经有人跑过 benchmark”。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/15.jpg)

::: image-caption
图 1：支持矩阵按公开实测数量和复现范围排序。Linux 系统的 NVIDIA RTX 30 系显卡用户报告最多；Ada 同卡对照正在补齐；RTX 4090 的公开实测只覆盖指定设备 ID；Windows UEFI 组合绑定具体主板和 BIOS。RTX 5060 Ti 暂无同类项目。
:::

### 6.1 Zagitta/gpu-unlock：有成功报告的主要是 GA102、GA104、GA106

> Zagitta/gpu-unlock 面向 Arch、Debian/Ubuntu、Fedora/RHEL 等发行版提供打包路径，支持 mkinitcpio、initramfs-tools 和 dracut。模块接受 GA102、GA103、GA104、GA106，**但目前有成功报告的是 RTX 3090 Ti、RTX 3090、RTX 3080 10 GB、改装 RTX 3080 20 GB、RTX 3070 和 RTX 3060 12 GB。** GA103 暂无报告。

RTX 3080 Ti 20 GB OEM 是一个重要反例。#28 报告[22] 中，模块完成解锁，SS0 在驱动运行期间也保持目标值，BF16 却仍停留在半速。日志出现“unlocked”，不等于性能一定提升，最后仍需跑 benchmark。

### 6.2 JustAResearcher/gpu-unlock-ada：新增三组实验目标，范围仍按单卡收窄

> JustAResearcher/gpu-unlock-ada 继续**覆盖 RTX 4090，并把 RTX 4090 D 标为实验目标。它还接受两张明确点名的显卡：RTX 4070 Ti SUPER `10de:2705` 和 RTX 4060 Ti 16GB `10de:2805`。**

这两张卡都只有一张实体卡的记录。4070 Ti SUPER 已完成两次冷启动和两次普通热重启，实测卡上的 WPR2 状态确实恢复；4060 Ti 16GB 只有一次冷启动，尚缺同卡 stock benchmark。其他 Ada ID、专业卡和数据中心卡仍不会被接受。

### 6.3 elias-gaeros/gpu-unlock（rtx4090 分支）：RTX 4090 是新增且有实测的目标

> elias-gaeros/gpu-unlock 的 rtx4090 分支仍是 Linux 方案，在原有部分 GA102、GA104 设备 ID 的基础上，新增 RTX 4090 的 AD102 设备 ID `10de:2684`。

需要区分代码保留的入口和实际测试：这个分支的公开性能数据与启动记录只覆盖 RTX 4090，不能把保留的 Ampere ID 当作新的实测结果。RTX 4080、4070 等其他 Ada 型号没有被加入，RTX 50 系、AMD 和 Intel 显卡也不在这些方案的已有测试范围内。**首次尝试需要完全断电后再开机，普通热重启不能保证显卡回到可用状态。显示输出交接和多卡运行也没有纳入这次 4090 测试。**

### 6.4 impacifist/Windows-GPU-UNLOCK：已有三套具体到主板和 BIOS 的实测组合

> 开源项目 impacifist/Windows-GPU-UNLOCK[23] 是把同类解锁思路搬到 Windows 启动链的 UEFI 方案。它当前兼容性文档记录了两套组合：

- RTX 3090 Founders Edition，搭配 ASUS TUF GAMING B650M-PLUS WIFI、BIOS 3057、Ryzen 7 7800X3D，记录环境为 Windows 11 Home 26200 和 NVIDIA 617.14。
- MSI RTX 3070 Ti，搭配 ASUS TUF GAMING B660M-E D4、BIOS 4003、Core i7-12700F，记录环境为 Windows 11 Pro 22000 和 NVIDIA 616.92。

JustAResearcher/gpu-unlock-ada 另有一套 MSI RTX 4070 Ti Windows 参考移植[24]，对应 MSI Z490、BIOS A.E0、i9-10850K 和 NVIDIA 616.56。该记录在 4096³ cuBLAS GEMM 下测得 FP16 从 84.08 提升到 151.43 TFLOPS，BF16 从 84.03 提升到 148.25 TFLOPS，分别提升 80.1% 和 76.4%。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/16.jpg)

::: image-caption
图 11：RTX 3090 FE 的 Windows 情况，保留前后中位数和三次运行范围。FP16 后期中位数为 125.91 TFLOPS，BF16 为 128.70 TFLOPS。（来源：impacifist/Windows-GPU-UNLOCK 测试文档）
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/17.jpg)

::: image-caption
图 12：Windows MSI RTX 3070 Ti 性能表，只保留路径、实测值、锁定基线和变化四列。FP16、BF16 均约达到锁定基线的 2.03 倍，FP32 pedantic 作为对照保留。（来源：impacifist/Windows-GPU-UNLOCK 测试文档）
:::

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/18.jpg)

::: image-caption
图 13：Windows MSI RTX 4070 Ti 性能表。4096³ cuBLAS GEMM 下，FP16 从 84.08 提升到 151.43 TFLOPS，BF16 从 84.03 提升到 148.25 TFLOPS。（来源：JustAResearcher/gpu-unlock-ada 参考移植）
:::

这些数字仍受单机测试限制。**3090 文档明确说明，没有独立控制的未处理冷启动基线，不能据此证明所有主板都能复现**；游戏、ComfyUI、睡眠恢复和长期稳定性也没有完成测试。

### 6.5 phiraml/gpu-unlock-blackwell：目前只到 RTX 5090，5060 Ti 仍无同类项目

> 截至 2026 年 10 月 25 日，公开检索到的最接近项目是 开源项目 phiraml/gpu-unlock-blackwell[25]。它计划把同一思路移植到 Blackwell GB20x，目前从 RTX 5090（GB202）起步。README 明确记录了两点：公开 Blackwell payload 尚不存在，移植工作还没有开始。

![文章插图](/articles/2026-10-25-gpu-unlock-rtx-4090-df8a9d0c/images/19.jpg)

::: image-caption
图 14：phiraml/gpu-unlock-blackwell 的只测量 stock RTX 5090：FP16/FP32 为 FP16/FP16 的 0.676 倍，BF16/FP32 为 0.664 倍。它说明限制存在，尚无公开 payload 或 5060 Ti 解锁数据。
:::

phiraml/gpu-unlock-blackwell 的 RTX 5090 测试只确认限制存在：NVIDIA 610.88、575 W 功耗限制下，FP16/FP32 吞吐为 FP16/FP16 的 0.676 倍，BF16/FP32 为 0.664 倍。仓库没有把 RTX 5060 Ti（GB206）列为支持目标，也没有 5060 Ti 解锁前后的 benchmark。因此，RTX 50 系仍处在限制已被测量、解锁尚未公开的阶段。

## 七、潜在风险集中在启动链、恢复入口和后续维护

### 7.1 Secure Boot、BitLocker 和启动参数都要处理

**Zagitta/gpu-unlock 和 JustAResearcher/gpu-unlock-ada 的 Linux 成功报告普遍关闭 Secure Boot，** 因为模块属于未签名的树外内核模块，加载后还会出现 kernel tainted 提示。安装过程涉及 DKMS、initramfs 和内核命令行，还要通过 `gpu_unlock.targets=` 明确指定显卡。JustAResearcher/gpu-unlock-ada 中的 4070 Ti SUPER、4060 Ti 16GB 和 4090 D，还需要额外的 experimental 参数。只安装软件包、看到模块加载成功，都不能证明解锁已经执行。

**impacifist/Windows-GPU-UNLOCK 的前置工作更多。** 当前 v0.3.1 是 source-only pre-release，用户需要按自己的分区 GUID 构建 EFI，准备至少 1 GiB 的独立 GPT FAT32 EFI System Partition，保留原 Windows ESP 和恢复分区，并准备 Python、指定版本 clang/lld-link 及五个固件文件。测试文档还要求 Windows C 盘完全解密、关闭 BitLocker 和 Secure Boot。启用默认自动启动时，还会关闭 Windows Fast Startup。

### 7.2 启动顺序出错，可能静默失效，也可能让显卡暂时不可用

> 这些方案必须赶在 NVIDIA 驱动之前运行，启动顺序成为最常见的故障点。Zagitta/gpu-unlock 的 issue 报告过 initramfs 没有装入模块、mkinitcpio 配置导致模块未加载，以及非冷启动解锁超时后 NVIDIA 驱动暂时无法使用两张 RTX 3090 的情况。**相关问题可能与特定提交和主机环境有关，不能推定当前版本必然复现，但足以说明恢复入口必须提前准备。**

Zagitta/gpu-unlock 和 JustAResearcher/gpu-unlock-ada 的基本回滚方式是删除内核命令行里的 `gpu_unlock.targets=`，再更新启动配置并重启。impacifist/Windows-GPU-UNLOCK 测试前应确认能通过 F8 选择 Windows Boot Manager，并保留恢复脚本生成的状态和备份。**没有这些准备，就不适合直接动主力机。**

### 7.3 睡眠、功耗和长期稳定性仍需测试

Zagitta/gpu-unlock 已经为 suspend to RAM 增加重新解锁流程，恢复时间约增加一秒，hibernation 仍未覆盖。impacifist/Windows-GPU-UNLOCK 的文档则明确写着 S3 睡眠会使解锁失效，目前没有 resume 路径。elias-gaeros/gpu-unlock（rtx4090 分支）的显示交接、多卡和长期负载同样缺少测试。

功耗限制还会吃掉一部分峰值收益。一张 RTX 3090 在 350 W 下，FP16、BF16 的绝对吞吐约提升 1.6 倍；按 SM 频率归一化后接近 1.9 倍。解锁后每周期执行更多运算，频率可能因同一功耗墙而下降。想复现接近两倍的绝对数字，还要同时考虑温度、功耗和具体板卡。

## 八、适合先测 BF16 和 prefill，不适合期待所有任务翻倍

> 这套方案目前最值得测试的是 BF16/FP16 矩阵运算、扩散模型、部分训练任务，以及长上下文 prefill 占比较高的推理服务。主要使用 INT8、INT4 量化推理，或者最关心 decode 速度的用户，公开数据暂时没有给出同等幅度的收益。游戏和普通图形负载也缺少可靠测试。

按实测数量和复现范围看，**Zagitta/gpu-unlock 的用户报告相对丰富，JustAResearcher/gpu-unlock-ada 开始补齐同卡对照和更多 Ada 型号**，elias-gaeros/gpu-unlock（rtx4090 分支）展示了接近两倍的指令吞吐，impacifist/Windows-GPU-UNLOCK 则把启动交接推进到了具体到主板的实验阶段。

真正决定是否值得折腾的，是目标程序的 kernel 构成、机器能否承担启动失败后的恢复，以及用户愿不愿意自行完成稳定性测试。对主力生产机来说，先保留原厂启动路径，等更多独立复测，是更稳妥的选择。

参考资料

[1]

Zagitta/gpu-unlock: *https://github.com/Zagitta/gpu-unlock*

[2]

JustAResearcher/gpu-unlock-ada: *https://github.com/JustAResearcher/gpu-unlock-ada*

[3]

impacifist/Windows-GPU-UNLOCK: *https://github.com/impacifist/Windows-GPU-UNLOCK*

[4]

开源项目 Zagitta/gpu-unlock: *https://github.com/Zagitta/gpu-unlock*

[5]

Zagitta/gpu-unlock 的 gfw_eval.h: *https://github.com/Zagitta/gpu-unlock/blob/af28cbbfd3d922a659227ecfb7369adfd924cf03/kernel/gpu_unlock/gfw_eval.h*

[6]

elias-gaeros/gpu-unlock 的 RTX 4090 寄存器报告: *https://github.com/elias-gaeros/gpu-unlock/blob/4ec490d49c1cc292c0b4cd5ce5a801b5d2c396dc/rtx4090-register-report.md*

[7]

Zagitta/gpu-unlock 的 main.c: *https://github.com/Zagitta/gpu-unlock/blob/af28cbbfd3d922a659227ecfb7369adfd924cf03/kernel/gpu_unlock/main.c*

[8]

Zagitta/gpu-unlock 的 unlock_610.c: *https://github.com/Zagitta/gpu-unlock/blob/af28cbbfd3d922a659227ecfb7369adfd924cf03/kernel/gpu_unlock/unlock_610.c*

[9]

JustAResearcher/gpu-unlock-ada 的 unlock_610.c: *https://github.com/JustAResearcher/gpu-unlock-ada/blob/4e7fefcc4923f9e083d5324dc75a7d2f906af77f/kernel/gpu_unlock/unlock_610.c*

[10]

impacifist/Windows-GPU-UNLOCK 的 app.c: *https://github.com/impacifist/Windows-GPU-UNLOCK/blob/e3f03cd9389195adcabda4743d8fc2922b32498c/unlock/app.c*

[11]

port.c: *https://github.com/impacifist/Windows-GPU-UNLOCK/blob/e3f03cd9389195adcabda4743d8fc2922b32498c/unlock/port.c*

[12]

开源项目 Zagitta/gpu-unlock: *https://github.com/Zagitta/gpu-unlock*

[13]

开源项目 Zagitta/gpu-unlock: *https://github.com/Zagitta/gpu-unlock*

[14]

开源项目 Zagitta/gpu-unlock: *https://github.com/Zagitta/gpu-unlock*

[15]

开源项目 Zagitta/gpu-unlock 的 #15 对照测试: *https://github.com/Zagitta/gpu-unlock/issues/15*

[16]

#14 测试: *https://github.com/Zagitta/gpu-unlock/issues/14*

[17]

#14 RTX 3080 测试: *https://github.com/Zagitta/gpu-unlock/issues/14*

[18]

#1 测试: *https://github.com/Zagitta/gpu-unlock/issues/1*

[19]

开源项目 JustAResearcher/gpu-unlock-ada: *https://github.com/JustAResearcher/gpu-unlock-ada*

[20]

4060 Ti 16GB 报告: *https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/ada-2805-report.md*

[21]

开源项目 elias-gaeros/gpu-unlock（rtx4090 分支）: *https://github.com/elias-gaeros/gpu-unlock/tree/rtx4090*

[22]

#28 报告: *https://github.com/Zagitta/gpu-unlock/issues/28*

[23]

开源项目 impacifist/Windows-GPU-UNLOCK: *https://github.com/impacifist/Windows-GPU-UNLOCK*

[24]

MSI RTX 4070 Ti Windows 参考移植: *https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/ad104-windows-latence-app.md*

[25]

开源项目 phiraml/gpu-unlock-blackwell: *https://github.com/phiraml/gpu-unlock-blackwell*

::: related-title
**相关推荐**
:::

::: related-links
- **[SASS King 的逆向工程：用证据流水线重启 GPU 底层公开知识，把消费级 Blackwell 拆成 29 条可复用审计签名](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902119&idx=1&sn=58a34c2173c2dd10d138a83137f9523f&scene=21#wechat_redirect)**
- **[逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm "榨干"成 NVIDIA GPU ISA 手册](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901758&idx=1&sn=d316e5dc6c2e2d5e85568467472124d4&scene=21#wechat_redirect)**
- **[暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)**
- **[逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892934&idx=1&sn=5ec357f95854bc9b6ebaa827012de874&scene=21#wechat_redirect)**

:::
