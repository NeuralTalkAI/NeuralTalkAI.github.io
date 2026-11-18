关键词：**Anthropic**、IPO 招股书、**算力承诺**、TPU、:mdnice-bold-italic[AI 基础设施]

> 2026 年 6 月，Anthropic 向美国 SEC 秘密递交了 IPO 招股书（S-1 草稿）。9 月 28 日，路透社拿到了这份文件并连续发出多篇报道，随后 Fortune、eeNews、The Next Web、Quartz 等媒体基于路透的审阅做了二次拆解。

有一点需要先交代：招股书原文还没有公开，下面所有数字都来自路透社对文件的审阅[1]和各家媒体基于路透报道的整理，不同报道之间的数字可以对得上；等 Anthropic 正式公开递交，完整文件才会挂上 SEC EDGAR。**接下来只看其中一块——推理与训练基础设施，把散在各篇报道里的数字收拢到一处。**

![文章插图](/articles/2026-10-08-1-1-6-anthropic-ipo-4de1c74b/images/01.webp)

::: image-caption
图 1：Anthropic 在 Dreamforce 2026 大会上的展台，横幅写着 "Your business to the power of Claude"。路透记者 Carlos Barria 拍摄。这张照片被选为多家媒体招股书报道的题图，也对应了招股书披露的渠道结构——Anthropic 近半数营收经由 Amazon 与 Google Cloud 等合作平台触达企业客户，展会与云市场是其 to B 分发的主战场。（图片来源：eeNews Europe[2]，摄影：Reuters / Carlos Barria）
:::

## 本文目录

- 一、已经花掉的钱：73.3 亿美元，是营收的 1.6 倍
  - 1.1 2025 财年：营收 46 亿，算力花了 73.3 亿
  - 1.2 每收入 1 美元，算力花掉 1.6 美元
  - 1.3 现金储备与 2026 年上半年
- 二、未来十年的承诺：至少 5180 亿美元，约 80% 不可取消
  - 2.1 承诺的总盘子与条款
  - 2.2 六家伙伴的承诺明细
- 三、钱从哪来：Broadcom 的 420 亿美元供应商融资
  - 3.1 420 亿美元可转换票据
  - 3.2 供应商、出租方、债权人的三重身份
- 四、供给端：从租云到自建，TPU 成为主算力
  - 4.1 算力规模以吉瓦计
  - 4.2 从租云到自建数据中心
  - 4.3 芯片路线：TPU 为主，Nvidia 只做弹性补充
- 五、三个比例看清 Anthropic 的基础设施策略
  - 5.1 承诺与收入的比例
  - 5.2 刚性与柔性的比例
  - 5.3 渠道依赖

## 一、已经花掉的钱：73.3 亿美元，是营收的 1.6 倍

### 1.1 2025 财年：营收 46 亿，算力花了 73.3 亿

> 招股书披露的 2025 财年损益结构，按 Fortune 的整理[3]如下：营收约 46 亿美元（较 2024 年增长 12 倍），运营支出 126.5 亿美元，运营亏损 80.6 亿美元（2024 年为 29.8 亿），净亏损约 420 亿美元。净亏损中约 340 亿是会计处理项——随公司估值上升而重估的可转换融资工具价值变动，不是真实现金流出。

**真正花出去的大头在算力**。2025 年 Anthropic 在"计算与基础设施"上支出 73.3 亿美元，是 2024 年的 3 倍，占总运营支出的 58% 以上（数据来源：eeNews Europe[4]）。

### 1.2 每收入 1 美元，算力花掉 1.6 美元

::: block-1
把这个数字和营收放在一起看：73.3 亿 ÷ 45.9 亿 ≈ 1.6，也就是 Anthropic 每收入 1 美元，就要在算力上花掉约 1.6 美元（这个倍数是拿上面两个披露数字相除得出的，招股书里没有直接写）。
:::

需要说明的是，这 73.3 亿把训练和推理的花费混在了一起，招股书没有拆开两者；**不过算力支出一年内涨到三倍，推理侧的用量增长——Claude Code 用户暴涨、API 调用量上升——是主要驱动之一。**

### 1.3 现金储备与 2026 年上半年

现金层面，截至 2025 年 12 月底，Anthropic 账上有 202.8 亿美元现金及短期投资（The Next Web[5]）。2026 年上半年情况明显好转：Q2 单季营收 115 亿美元（Q1 为 47.3 亿），并连续第二个季度实现调整后运营盈利（Fortune[6]）。

## 二、未来十年的承诺：至少 5180 亿美元，约 80% 不可取消

### 2.1 承诺的总盘子与条款

招股书里体量最大的一组数字是未来承诺：Anthropic 与六家伙伴签署了长期基础设施协议，未来约十年合计至少 5180 亿美元，其中约 80% 不可取消、或无论是否用满承诺容量都必须付款（路透社报道，经 The Next Web 转述[7]）。

Anthropic 在文件中给出的理由是：先进 AI 的需求将超过供给，AI 发展"将主要受制于算力的可获得性"（"limited principally by the availability of compute"）。

### 2.2 六家伙伴的承诺明细

![文章插图](/articles/2026-10-08-1-1-6-anthropic-ipo-4de1c74b/images/02.webp)

::: image-caption
表 1：Anthropic 招股书披露的六家基础设施伙伴承诺明细。六项合计约 5182 亿美元，与招股书"至少 5180 亿"的总数吻合。数据来源：Reuters，经 TrendForce 转述[8]、Press Insider[9]、The Next Web[10]
:::

值得注意三点：

- 其一，**金额最大的单项是与 Broadcom 相关的设备租赁（1612 亿），对应 Anthropic 直接租用 TPU 硬件而非纯买云服务的转向；**
- 其二，Google 与 Amazon 的合同采用"照付不议"（take-or-pay）结构，承诺期内"无论是否使用"都须付款，招股书原话是"如果我们的实际支出不足，必须向 Google 补足差额"；
- 其三，六家中唯一灵活的是 xAI——845 亿是上限而非承诺，且可提前 90 天退出，这笔钱买的是基于 Nvidia GPU 的容量，相当于 Anthropic 给自己留的一个弹性仓位。

## 三、钱从哪来：Broadcom 的 420 亿美元供应商融资

招股书披露了一套"供应商融资"结构来支付这些承诺，核心是 Broadcom。

### 3.1 420 亿美元可转换票据

> 10 月 1 日路透社报道，Broadcom 同意向 Anthropic 提供最高 420 亿美元融资，形式为可转换票据（convertible notes），IPO 后可转换为 Anthropic 股票；这笔钱可覆盖 Anthropic 五年期、总额 1252 亿美元 TPU 租赁承诺的约三分之一。

Anthropic 已于 2026 年 4 月把一笔资金存入以 Broadcom 为受益人的受限账户，且表示预计 IPO 完成前不会售出任何票据（The Next Web[11]、Finimize[12]）。

### 3.2 供应商、出租方、债权人的三重身份

::: block-1
Anthropic 自己在招股书里把风险写明了：:mdnice-bold-italic[Broadcom 同时横跨芯片供应、设备租赁与资金出借三重角色，深入绑定 Anthropic 的算力路线图，这与 Amazon 等主要提供云基础设施和 Claude 分发渠道的伙伴有本质区别——供应商的定价和硬件分配决策，直接影响 Anthropic 能否拿到足够的算力] （Finimize[13]、The Next Web[14]）。
:::

芯片厂商借钱给客户买自己的芯片，循环融资正式写进了招股书。这套结构并非首创——Nvidia 过去几年一直在用资产负债表撬动 GPU 需求——**但 Broadcom 以"供应商 + 出租方 + 债权人"三重身份跟进，把供应商融资的规模推到了一个新量级。**

## 四、供给端：从租云到自建，TPU 成为主算力

把这些承诺翻译成物理量，能看到 Anthropic 算力结构的两个变化。

### 4.1 算力规模以吉瓦计

第一个变化是算力规模的单位从"美元"变成了"吉瓦"。

2026 年 4 月，Anthropic 宣布扩大与 Google、Broadcom 的合作，自 2027 年起获得约 3.5 GW 的下一代 TPU 算力（eeNews Europe[15]）。一座大型核电站反应堆的输出功率约 1 GW，**Anthropic 仅 TPU 这一条线，锁定的就是三座反应堆量级的新增电力容量。**

### 4.2 从租云到自建数据中心

第二个变化是部署模式从"纯公有云租用"转向"专用数据中心 + 直接租赁芯片"，把基础设施敞口搬上自己的资产负债表（eeNews Europe[16]）。

Anthropic 官方已宣布投资 500 亿美元建设美国本土 AI 基础设施，与数据中心开发商 Fluidstack 合作在得克萨斯州和纽约州等地建设定制数据中心（Anthropic 官方公告[17]）。

### 4.3 芯片路线：TPU 为主，Nvidia 只做弹性补充

**芯片路线也随之清晰：主力训练与推理算力押在 Google TPU 上（Google 与 Broadcom 联合设计，Anthropic 通过 Broadcom 租赁硬件），AWS Trainium 承担 Amazon 侧，xAI 合同提供 Nvidia GPU 的弹性补充，AMD 则是 200 亿美元量级的新增供给方。** 换句话说，招股书里没有一个数字指向 Anthropic 大规模直接采购 Nvidia GPU——这与 OpenAI 的 Stargate 路线形成鲜明对比。

## 五、三个比例看清 Anthropic 的基础设施策略

把上述数字压缩成三个比值，Anthropic 的基础设施策略可以看得比较清楚。

### 5.1 承诺与收入的比例

5180 亿美元摊到十年，每年约 500 亿；而按 2026 年上半年推算，Anthropic 的年化营收约 460 亿美元——每年的算力承诺比当前的年化营收还略高一点（这个比较是把两个披露数字直接相除得出的）。**也就是说，未来两三年的营收增长基本会被算力账单吃掉，盈利能不能改善，全看收入增速能不能继续跑赢成本。**

### 5.2 刚性与柔性的比例

约 80% 承诺不可取消，唯一的大额弹性仓位（xAI 的 845 亿）恰好是 Nvidia 算力——:mdnice-bold-italic[Anthropic 把"可退出权"留给了通用性最强、最易转售的算力形态，把"刚性承诺"押在了 TPU 这条自研芯片路线上。这是一次明确的架构级站队。]

### 5.3 渠道依赖

> 2025 年近半数营收经由 Amazon 与 Google Cloud 转售，而这两家同时是 Anthropic 最大的算力供应商和股东（SiliconANGLE[18]）。:mdnice-bold-italic[客户、渠道、供应商、股东四位一体，是这份招股书里比亏损数字更值得关注的结构性事实——Anthropic 增长得越快，流向云厂商的钱就越多，云厂商回流给它的算力和资本也越多。这个循环能转多久，取决于 Claude 的收入曲线能不能一直陡下去。]

这份泄露版招股书第一次把一家前沿模型公司的算力成本、合同条款和供应链关系变成了可以核查的数字。**等公开版 S-1 挂上 EDGAR，训练与推理成本各自占多少、毛利率怎么算、对各家伙伴的承诺按什么节奏付款，才是下一层值得挖的东西。**

说明🥱：以上数据均来自路透社对 Anthropic 秘密递交版 S-1 草稿的审阅及各家媒体转述，招股书原文尚未公开；文中标注为推算得出的比值，是依据披露数字计算的结果，不是招股书原文直接给出的数字。

参考资料

[1] 路透社对文件的审阅: *https://thenextweb.com/news/anthropic-ipo-prospectus-42bn-loss-518bn-compute*

[2] eeNews Europe: *https://www.eenewseurope.com/en/anthropic-ipo-518-billion-infrastructure-commitments/*

[3] Fortune 的整理: *https://fortune.com/2026/09/29/anthropic-ipo-s-1-prospectus-income-statement/*

[4] eeNews Europe: *https://www.eenewseurope.com/en/anthropic-ipo-518-billion-infrastructure-commitments/*

[5] The Next Web: *https://thenextweb.com/news/anthropic-ipo-prospectus-42bn-loss-518bn-compute*

[6] Fortune: *https://fortune.com/2026/09/29/anthropic-ipo-s-1-prospectus-income-statement/*

[7] 路透社报道，经 The Next Web 转述: *https://thenextweb.com/news/anthropic-ipo-prospectus-42bn-loss-518bn-compute*

[8] Reuters，经 TrendForce 转述: *https://www.trendforce.com/news/2026/09/30/news-anthropic-eyes-2t-ipo-valuation-518b-ai-buildout-spans-amazon-google-broadcom-and-more/*

[9] Press Insider: *https://pressinsider.com/technology/anthropic-flags-existential-risks-to-humanity-in-ipo-filing/*

[10] The Next Web: *https://thenextweb.com/news/anthropic-ipo-prospectus-42bn-loss-518bn-compute*

[11] The Next Web: *https://thenextweb.com/news/broadcom-anthropic-42bn-loan-tpu-lease-reuters*

[12] Finimize: *https://finimize.com/content/anthropics-ipo-filing-puts-broadcom-at-the-center*

[13] Finimize: *https://finimize.com/content/anthropics-ipo-filing-puts-broadcom-at-the-center*

[14] The Next Web: *https://thenextweb.com/news/broadcom-anthropic-42bn-loan-tpu-lease-reuters*

[15] eeNews Europe: *https://www.eenewseurope.com/en/anthropic-ipo-518-billion-infrastructure-commitments/*

[16] eeNews Europe: *https://www.eenewseurope.com/en/anthropic-ipo-518-billion-infrastructure-commitments/*

[17] Anthropic 官方公告: *https://www.anthropic.com/news/anthropic-invests-50-billion-in-american-ai-infrastructure*

[18] SiliconANGLE: *https://siliconangle.com/2026/09/29/leaked-anthropic-ipo-filing-reveals-8b-operating-loss-rapid-revenue-growth/*

::: related-title
**相关推荐**
:::
