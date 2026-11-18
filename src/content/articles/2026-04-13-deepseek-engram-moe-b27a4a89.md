关键词：DeepSeek Engram、LLM、**外部记忆**、:mdnice-bold-italic[正则化]、**残差通路**

> 在LLM架构创新中，DeepSeek Engram以“reasoning与knowledge分离”为核心主张，凭借外部N-gram记忆表的知识托管、O(1)检索等特性，被寄予解决Transformer原生知识查找缺陷的厚望。**其声称通过百亿参数级外部记忆库实现精准记忆读取，一度成为记忆增强架构的热门方向。**

然而，:mdnice-bold-italic[本文作者团队的控制变量实验却颠覆了这一认知：将记忆表替换为高斯白噪声、统一共享向量，甚至仅保留单一bucket，模型性能仍显著优于纯训基线，且与原版效果相差无几]。研究揭示，**Engram的性能增益并非来自所谓的知识检索，而是源于门控结构与残差通路带来的优化特性**，外部记忆表实则仅扮演正则化“伪负载”角色。

::: block-1
这一发现不仅打破了对Engram记忆功能的固有认知，更为LLM架构优化提供了新思路——无需复杂的系统级调度与海量显存占用，仅通过结构优化即可实现性能提升，为降本增效的模型开发提供了重要启示。
:::

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/01.webp)

- **DeepSeek Engram里没有记忆，就像MoE里没有专家**
- 复现方法、实验 Checkpoint 与训练日志：https://im-yuka.notion.site/340306a9770680aa8c16ddee4916f49b
- 作者：栀染，https://zhuanlan.zhihu.com/p/2026419832371836848
- 2000 字，阅读 15 分钟，**播客 6 分钟**

相关推荐

- [Agent 知识工程实践：让 Coding Agent 像人一样学习、记忆和成长](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898504&idx=1&sn=a1cf34af9e395a3fb2431de79ea2a30e&scene=21#wechat_redirect)
- [技能即记忆：AutoSkill 让 LLM 智能体从交互经验中自我进化，实现终身学习，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898295&idx=1&sn=6bf15a2f47d940cea0165f5d1f50aa61&scene=21#wechat_redirect)
- [让手机CPU,GPU,NPU协同！浙大端侧AI“超级记忆”AME硬件感知引擎：7倍索引加速，6倍插入提速，隐私与速度兼得！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895124&idx=1&sn=5c073c54ccc26b1797e79a12862ac496&scene=21#wechat_redirect)

> 最近笔者所在的组在研究 reasoning 和 knowledge 分离的架构，Engram 作为一个著名的以此为核心 claim 的方法自然值得我们研究，但在我们深入研究 Engram 后，却得出了一些令人感叹的结论。

## 本文目录

- 一、缘起：完美的记忆增强架构
  - 1.1 一个朴素而致命的疑问
- 二、控制变量实验：四组模型的记忆真相测试
  - 2.1 令人沉默的实验结果
- 三、正则化的外衣：当记忆库变成梯度旁路
  - 3.1 微弱的“真记忆”增益
  - 3.2 背后的朴素数学直觉
- 四、它到底有没有记忆？答案是否定的
  - 4.1 直接内容干预实验的失败
  - 4.2 事实型提示词下的进一步验证
- 五、规模扩大后的结论固化：3B 参数模型的验证
  - 5.1 五条件消融的最终结论
- 六、工程启示：扔掉记忆库，拥抱随机噪声
- 结语：请礼貌地追问那个庞大的记忆表

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/02.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、缘起：完美的记忆增强架构

> 论文里说什么 Transformer 缺乏原生的 knowledge lookup 算子啦，什么把静态知识托管给外部 N-gram embedding 啦，什么 O(1) 检索解放前序计算层啦……听起来简直是完美的 memory-augmented 架构，是一个好文明！

### 1.1 一个朴素而致命的疑问

一个自然而然的问题是，那个动辄百亿参数、被大书特书富含海量世界知识的 external memory table，真的非要不可吗？里面的知识真的有用吗？

## 二、控制变量实验：四组模型的记忆真相测试

> 于是我们捣鼓出了一个令人啼笑皆非的实验。我们做了一个简单的控制变量：

1. Real: 原汁原味的 Engram，乖乖用训练好的庞大 memory table。
2. Randomize: 把那个巨大的 N-gram table 全部换成高斯白噪声（随机初始化，冻结住不更新）。
3. Uniform: 强制所有的 Hash 函数全部撞到同一个 bucket 里，等于所有 token 查表查出来的都是同一个共享向量。
4. Dense Baseline: 没有任何 Engram 分支的纯训苦逼 baseline。

### 2.1 令人沉默的实验结果

> 本来只是想随便测测 memory 的鲁棒性，希望没有 coding bug 的问题（悲。结果跑完一看，哈哈哈哈，一看一个不吭声(x

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/03.webp)

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/04.webp)

**目前结论是：Real > Uniform ≈ Randomize >>>>> Dense Baseline。**

## 三、正则化的外衣：当记忆库变成梯度旁路

你没看错。哪怕你往那个被寄予厚望的记忆库里塞满纯纯的随机垃圾（Randomize），或者让全天下的 token 都去喝同一口 Uniform，**它的效果依然能把纯训的 Dense baseline 吊起来打！** 哪怕我们把 table 的有效容量破坏得只剩下一个 bucket，模型依然觉得真香。

这就很有意思了。**这明明就是一个披着 memory 外衣的巨大正则化啊，孩子们！**

### 3.1 微弱的“真记忆”增益

> 可能有人会 argue 说：“哎呀，Real 还是比 Uniform 好一点点的嘛，说明知识还是有用的！这好嘛？”

是的，表里的内容确实贡献了一点点微弱的增益，但这就好比你以为自己吃的是十全大补丸（外部知识），其实真正让你强身健体的是每天按时喝水（多了一条残差通路）。**所谓的 Engram 增益，绝大部分根本不是来自于什么精准的细粒度记忆读取，而是来自于 architecture/pathway 本身带来的优化性质**。我们可能不需要记忆本身，只需要幻想记忆存在就可以。

### 3.2 背后的朴素数学直觉

> 这背后的数学原因可能极其朴素：Engram 那个看似花哨的`gating * memory_vector`结构，其实只是在原本的梯度流上开了一条额外的 bypass。

那个 context-aware gating 才是真 make sense 的东西，它根据当前的 hidden state 动态学到了如何调节特征，而外面的 memory table 扮演的不过是一个 dummy payload——哪怕这个 payload 是一坨随机噪声，只要加上这层 non-linear 的门控，就能给早期 layer 提供极好的梯度引导和宏观统计场，让模型更早收敛。

启发式的比喻来说，你多修了一条高速公路，至于上面跑的是运钞车（真实知识）还是垃圾车（随机噪声），对缓解主干道交通拥堵（梯度传播）的效果差不多。

## 四、它到底有没有记忆？答案是否定的

那么它到底有没有记忆的作用？答案是显然的，NO！

### 4.1 直接内容干预实验的失败

如果一个模块真在做较强意义上的 retrieval，那么我们对 memory 内容做针对性干预时，它应该在输出侧表现得比较听话。比如说，如果我们选那些理论上最有利的 bucket，故意把 donor 内容塞进去，那么模型至少应该在这些位置表现出更强的 donor steering、rank 改变、甚至 top-k 可见性提升。但结果并不支持这种想象。

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/05.webp)

即便是在最强 bucket 上，direct content manipulation 也没有产生那种强、稳定、可编程的 retrieval-like 控制！这很重要，因为它说明 Engram 里确实有 content signal，但这个 signal 的作用方式，并不像他们说的是外包语义记忆库。

### 4.2 事实型提示词下的进一步验证

有人可能会说：也许你们挑的测试还不够像记忆题。如果换成更加 factual 的 prompt，也许外部内容的作用就会更明显。

这个反驳是合理的，所以我们也做了这一组。结果依然没有出现我们原本期待中的 retrieval-like behavior：

## 五、规模扩大后的结论固化：3B 参数模型的验证

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/06.webp)

如果有人担心上面的现象只是小模型复现体上的 artifact，那么 3B 的结果就很重要。先看训练轨迹。在 3B 规模上，Engram 相比 strict S0 的优势不是只出现在最终点，而是在训练过程中始终存在。(除了早期的部分时刻)这表明 Engram 的性能优势是真实存在的。

### 5.1 五条件消融的最终结论

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/07.webp)

但更重要的是最终的五条件消融。

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/08.webp)

在 3B 上，我们看到 real 最好，但 off ≈ random ≈ uniform 仍然非常接近，而它们整体又明显优于 strict S0，随着规模上去，我们没有看到越来越像 retrieval 的趋势，反而看到了更强的 pathway-dominant pattern。这完全不同于我们对 Engram 的原始直觉。

## 六、工程启示：扔掉记忆库，拥抱随机噪声

> 换个角度想，其实这也是个天大的好消息。论文里花了大量篇幅去讲 algorithm-system co-design，讲怎么用 CPU host memory 做 100B table 的 offload，讲怎么做 PCIe 异步 prefetch 掩盖延迟……

既然我们发现 Uniform（全部路由到同一个 bucket）或者 Randomize 都能拿走绝大部分收益，那其实完全不用苦一苦 infra 的同学去开发什么复杂的系统级调度了！

**我们只需要在内存里保留一个参数，或者干脆每次前向都 torch.randn 一下，直接广播给所有的 layer！** 不用查表，不用跨 PCIe 通信，更不用搞什么 Zipfian 缓存分层，显存占用瞬间从几百 G 降到几个 Byte，性能还能暴打 baseline，简直是降本增效的工程奇迹！皆大欢喜！

## 结语：请礼貌地追问那个庞大的记忆表

> 行文匆匆，未免有不少纰漏，但实验现象和大致思路应该是大差不差的。

总之，你的 DeepSeek Engram 可能根本不需要那个巨大的 memory table，下次再有人跟你吹 LLM 挂载了多大的外部 N-gram 记忆库，读者朋友们可以礼貌地问一句：“哥，您这表里装的不会全是正则化偏置吧？”

![文章插图](/articles/2026-04-13-deepseek-engram-moe-b27a4a89/images/09.webp)
