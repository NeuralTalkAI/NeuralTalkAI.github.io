关键词：AutoGLM、GUI 智能体、**自主智能体训练方法**、**中间接口设计**、:mdnice-bold-italic[Android 自主控制]

![文章插图](/articles/2026-01-04-autoglm-gui-agent-96-2-f62827ab/images/01.webp)

- **AutoGLM: Autonomous Foundation Agents for GUIs**
- https://arxiv.org/pdf/2411.00820
- https://xiao9905.github.io/AutoGLM
- https://github.com/zai-org/Open-AutoGLM
- 4000 字，阅读 13 分钟，**播客 23 分钟**

::: related-title
相关推荐
:::

::: related-links
- [阶跃星辰4B模型实现手机GUI自动化！端侧Agent隐私本地执行+跨设备兼容：GUI-MCP 协议加持！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895177&idx=1&sn=c7c289af95d4c30ed4e57c3cb79c28f8&scene=21#wechat_redirect)
- [突破GUI像素瓶颈！面向端侧Agent语义世界建模 MobileWorldBench！1.4M 数据样本驱动 7.4%性能跃升！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895146&idx=1&sn=c640f4707f8edf12ba941a2f276dcda4&scene=21#wechat_redirect)
- [350M小模型精度性能双超 ChatGPT等商业大模型！平衡模型规模、训练效率与任务精度的微调方案，大幅提升智能体工具调用能力！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894791&idx=1&sn=9ece1ecc5b9a4c46e091f4c828c2ecef&scene=21#wechat_redirect)

:::

> 订咖啡、订餐厅、查快递…你是否想过，有一天 AI 可以像真人一样在手机和网页上帮你完成这些任务？智谱 AI 与清华大学联合团队最近推出的 AutoGLM，正让这一想象成为现实。

你是否曾经想果，只要对手机📱说一句“帮我从最近的美团店订一杯半糖的冰椰子拿铁”，它就能自动完成打开 App、选择商品、填写地址、下单支付的全过程？或者对浏览器说“帮我在 OpenTable 上为我和父母预订 10 月 23 日晚上 7 点半 Megan's Kitchen 的位子，电话是 146xxxxxxxx”，它就能自动导航、填写表单并成功预订？

这不再是科幻电影中的场景，而是智谱 AI 与清华大学联合团队最新研究成果 AutoGLM 已经实现的能力。

![文章插图](/articles/2026-01-04-autoglm-gui-agent-96-2-f62827ab/images/02.webp)

::: image-caption
图 1 | AUTOGLM 在中文 Android 版美团 APP 上的真实手机使用演示，任务指令为 “从瑞幸咖啡点一杯半糖热椰拿铁”。查看更多视频。包括启动 APP（步骤 1）、点击搜索框（步骤 3）、确认店铺（步骤 5）、进入店铺（步骤 7）、提交订单（步骤 14）等关键操作。与 AndroidLab 的英文虚拟环境不同，此演示基于物理手机，还原用户真实操作逻辑 —— 例如需识别美团 APP 内 “瑞幸咖啡” 的店铺入口、“热饮” 分类及 “半糖” 甜度选项。该任务属于中文用户高频需求，AUTOGLM 的流畅执行验证了其对中文 APP 界面元素的理解能力，目前相关 Android 客户端仅对受邀用户开放内部测试
:::
