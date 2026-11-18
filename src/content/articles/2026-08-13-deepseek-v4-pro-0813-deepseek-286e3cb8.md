关键词：**DeepSeek V4-Pro**、Agent、:mdnice-bold-italic[Harness]、开源模型、API 定价

> 2026 年 8 月 12 日，DeepSeek 官方 API 文档首页与定价页双双确认：`deepseek-v4-pro` 模型版本已更新为 DeepSeek-V4-Pro-0813，调用方式不变，Responses API 同步支持。:mdnice-bold-italic[V4-Pro 正式版已经静默上线，只是更新日志的正式公告尚未发布。官方版式的 Agent 评测表显示，0813 在 HLE、Terminal Bench 2.1、DeepSWE、DSBench 等基准上大幅超越 4 月预览版，并与 Opus-4.8、Fable 5 贴身竞争。]

四个月前预览版发布时，DeepSeek 官方把话说得很满：Agent 能力开源第一梯队、世界知识仅次于 Gemini-3.1-Pro、推理比肩顶级闭源模型。但"预览版"三个字始终悬在头上，直到 7 月 31 日 Flash 正式版惊喜发布，Pro 版依然缺席。

现在不用等了。官方文档已经改口，0813 上线了。

![文章插图](/articles/2026-08-13-deepseek-v4-pro-0813-deepseek-286e3cb8/images/01.webp)

::: image-caption
DeepSeek-V4-Pro 正式版 Agent 评测对比表：表中 DSBench-FullStack 与 DSBench-Hard 为 DeepSeek 内部测试集，外部无法复现，看官自行折算权重；其余基准中，V4-Flash-0731 列的九项分数与官方 7 月 31 日更新日志逐项吻合。
:::

## 一、官方文档实锤：deepseek-v4-pro 已切换至 0813

打开 DeepSeek API 文档首页[1]，首次调用页面的模型参数表下方有一条脚注，原文是：

> `deepseek-v4-flash` 模型版本已更新为 DeepSeek-V4-Flash-0731，`deepseek-v4-pro` 模型版本已更新为 DeepSeek-V4-Pro-0813，调用方法不变，使用 `deepseek-v4-flash`、`deepseek-v4-pro` 即可调用最新版本。

第二处实锤在 官方定价页[2]的模型细节表：`deepseek-v4-pro` 一列的模型版本明确标注 DeepSeek-V4-Pro-0813，Responses API 一栏为"支持"，并发限制 500。两个页面互相印证，版本切换不是笔误。

::: block-1
**DeepSeek 发模型向来是先改文档、切线上模型，再补公告**。更新日志页[3] 最新条目目前仍是 7 月 31 日的 Flash 正式版公测，Pro 版的公告条目应该会在一两天内补上——那条日志末尾就写了"DeepSeek-V4-Pro 正式版将会尽快发布"。
:::

定价页还有一条预告：计划近期整体上调 API 定价，预计涨幅较大。满血 Pro 上线叠加涨价预告，两件事大概率是配套的。

## 二、0813 到底强在哪

### 2.1 对预览版是碾压级提升

> 拿 V4-Pro-0813 对 V4-Pro-Preview：HLE（带工具）从 48.2 涨到 60.0，Terminal Bench 2.1 从 72.1 涨到 87.9，DeepSWE 从 12.8 翻到 62.7，DSBench-Hard 从 31.1 涨到 67.2。

![文章插图](/articles/2026-08-13-deepseek-v4-pro-0813-deepseek-286e3cb8/images/02.webp)

::: image-caption
V4-Pro-0813 与预览版、Flash-0731 的六项 Agent 基准对比。注：数据取自上文评测表，六个基准均为 Agent 相关任务，分数越高越好。
:::

这种幅度单靠后训练微调很难解释，更像训练策略整体重做。7 月 31 日官方日志里，Flash 正式版"结构尺寸不变、仅重做后训练"就在九项基准全面反超 Pro 预览版——Pro 正式版的提升路径大概率同源，幅度更大。

### 2.2 对标对象换成了 Opus-4.8 和 Fable 5

4 月预览版发布时，官方对标的还是 Opus 4.6，并坦承与思考模式存在差距。

新表格的对比列直接换成 Opus-4.8 与 Fable 5（w/ fallback）注 1：HLE 上 Fable 5 以 53.3/63.0 领先，DeepSWE 上 Fable 5 拿 70.0 对 0813 的 62.7，但 Toolathlon-Verified（77.9 对 74.1）、Terminal Bench 2.1（88.0 对 87.9）已经咬到同一水平线。DSBench-FullStack 一项，0813 的 71.1 与 Opus-4.8 的 71.6 几乎打平。

*注 1："w/ fallback"指 Fable 5 评测时启用了回退机制，具体含义与测试条件官方未披露，横向对比时请注意口径差异。*

也要泼点冷水：NL2Repo 一项，Opus-4.8 以 69.7 领先 0813 的 61.5。旗舰对决远没到"全面超越"的地步。

### 2.3 Flash 正式版不是终点，是下限

一个 284B 总参、13B 激活的 Flash-0731 都能在 Agent 基准上压着四个月前的 Pro 预览版打，0813 的下限已经被自家小兄弟抬高了。Pro 与 Flash 的差价摆在那里，正式版 Pro 必须在最难的任务上拉开身位，从表格看它做到了：DSBench-Hard 67.2 对 59.6，Cybergym 83.3 对 76.7。

## 三、价格与规格：开发者关心的硬信息

> 根据 官方定价页[4]与 [预览版发布文章：DeepSeek-V4 预览版：迈入百万上下文普惠时代](https://mp.weixin.qq.com/s?__biz=Mzk0OTYwNzc3NQ==&mid=2247485745&idx=1&sn=ef3bc11042fc329e89bc66a8938e8b2c&scene=21#wechat_redirect)，整理如下：

| 项目 | deepseek-v4-flash | deepseek-v4-pro |
| --- | --- | --- |
| 当前模型版本 | V4-Flash-0731 | V4-Pro-0813 |
| 参数规模 | 284B 总参 / 13B 激活 | 1.6T 总参 / 49B 激活 |
| 上下文长度 | 1M | 1M |
| 输入（缓存命中） | 0.02 元/百万 token | 0.025 元/百万 token |
| 输入（缓存未命中） | 1 元/百万 token | 3 元/百万 token |
| 输出 | 2 元/百万 token | 6 元/百万 token |
| 并发限制 | 2500 | 500 |
| Responses API | 支持 | 支持 |

两点补充。

- 其一，4 月预览版上线时 Pro 的定价是输入 12 元、输出 24 元，现在降到 3 元、6 元，与官方当时"算力到位后价格大幅下调"的承诺对得上；涨价预告针对的是现行价格。
- 其二，V4 全系采用 token 维度压缩结合 DSA 稀疏注意力（DeepSeek Sparse Attention），1M 上下文是官方服务标配，技术报告与开源权重见 DeepSeek V4 Hugging Face 页面[5]。

## 四、开发者现在该做什么

1. **直接切，接口不用动。** `deepseek-v4-pro` 原地升级，模型名不变，base_url 不变，OpenAI 与 Anthropic 两种接口格式都支持，Responses API 也已就绪（见 DeepSeek API 文档[6]）。
2. **先用 Flash-0731 跑一遍成本基线。** 它的输出价格只有 Pro 的三分之一，九项 Agent 基准全面超过 Pro 预览版。日常任务够用就不必上 Pro，尤其 Pro 的并发上限只有 500。
3. **盯住涨价公告和 Harness。** 官方配套编程工具 DeepSeek Harness 已在 7 月 31 日日志中预告"即将发布"，Flash 正式版的成绩正是用它跑出来的。:mdnice-bold-italic[Harness 的落地节奏和涨价方案，直接决定 0813 的实际使用成本。]

## 写在最后

> 从 4 月 24 日预览版到 7 月 31 日 Flash 正式版，再到今天 API 文档确认的 Pro-0813，DeepSeek 这一轮的节奏很清晰：:mdnice-bold-italic[先用预览版铺生态，再用正式版逐层兑现性能，顺手把价格砍到原来的四分之一。Flash 正式版已经验证了"重做后训练"的杀伤力，0813 的模型在 Agent 赛道与 Opus-4.8、Fable 5 的差距会被压缩到一个身位以内。]

最后，就是盯紧 DeepSeek 官方更新日志[7]，等待正式公告了。

参考资料

[1] DeepSeek API 文档首页: *https://api-docs.deepseek.com/zh-cn/*

[2] 官方定价页: *https://api-docs.deepseek.com/zh-cn/quick_start/pricing*

[3] 更新日志页: *https://api-docs.deepseek.com/zh-cn/updates*

[4] 官方定价页: *https://api-docs.deepseek.com/zh-cn/quick_start/pricing*

[5] DeepSeek V4 Hugging Face 页面: *https://huggingface.co/collections/deepseek-ai/deepseek-v4*

[6] DeepSeek API 文档: *https://api-docs.deepseek.com/zh-cn/*

[7] DeepSeek 官方更新日志: *https://api-docs.deepseek.com/zh-cn/updates*
