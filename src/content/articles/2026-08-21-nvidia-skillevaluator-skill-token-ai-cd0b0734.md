关键词：AI 智能体、Agent Skills、**SkillEvaluator**、:mdnice-bold-italic[基准评测]、NVIDIA

> **AI 智能体的表现，很大程度上取决于喂给它的上下文**。模型再强、文档再全，:mdnice-bold-italic[智能体也可能在找工具上浪费步骤、在死胡同里烧 token]。Skill（技能）把指令、示例和工具调用指南打包成可复用的能力单元，让智能体从意图直达方案——**但"装了 Skill 应该会变好"和"装了这个 Skill 确实变好了"之间，隔着一整套测量。**

![文章插图](/articles/2026-08-21-nvidia-skillevaluator-skill-token-ai-cd0b0734/images/01.webp)

- **Evaluating AI Agent Skill Performance with NVIDIA SkillEvaluator**
- https://developer.nvidia.com/blog/evaluating-ai-agent-skill-performance-with-nvidia-skillevaluator/
- https://github.com/NVIDIA/SkillSpector

::: related-title
**相关推荐**
:::

::: related-links
- **[NVIDIA 官方 Agent Skills：为 AI 智能体装进"NVIDIA 官方知识"，重塑智能体 SKILL 的可信分发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902669&idx=1&sn=7553fda3b64e88da2f51242491a15b6f&scene=21#wechat_redirect)**
- **[筛选Agentic Skills成功率提升16.2%，Agentic Skills：超越工具使用的系统化知识综述与未来治理挑战](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897931&idx=1&sn=b263f45065d2772fda6e997a39458fbf&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::

> 为此，NVIDIA 构建了开源的 **SkillEvaluator，用三个层级的评估回答不同的问题**：:mdnice-bold-italic[Tier 1 做安全与结构静态检查，Tier 2 用嵌入相似度查重，Tier 3 则让智能体在隔离沙箱里真实跑任务] ——**装 Skill 一次，不装一次，分差就是这个 Skill 的贡献。**

在覆盖 30 多个 NVIDIA 产品、300 多个认证 Skill 的首批基准中，正确性从 46 分升至 87 分，有效性从 39 分升至 78 分，平均 Skill Lift 达 +31 分（剔除安全维度后为 +39 分）。Claude Code 与 Codex 两套 harness 的结果一致，但增益差异主要来自产品领域而非 harness 本身。

值得注意的是，token 节省并非必然：**有的 Skill 省下 76.9% 的 token，有的反而多花 120.3%——这正是需要实测而非感觉的原因**。

## 本文目录

- 一、先搞清楚：什么是 Skill，为什么需要评测
- 二、SkillEvaluator：三层评估体系
- 三、实测结果：300 多个 Skill 的首批基准
- 四、合作伙伴的试点
- 五、三个对实操有用的发现
- 六、上手：安装、验证与目录结构
- 七、写在最后

![文章插图](/articles/2026-08-21-nvidia-skillevaluator-skill-token-ai-cd0b0734/images/02.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[技能即记忆：AutoSkill 让 LLM 智能体从交互经验中自我进化，实现终身学习，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898295&idx=1&sn=6bf15a2f47d940cea0165f5d1f50aa61&scene=21#wechat_redirect)**
- **[筛选Agentic Skills成功率提升16.2%，Agentic Skills：超越工具使用的系统化知识综述与未来治理挑战](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897931&idx=1&sn=b263f45065d2772fda6e997a39458fbf&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**

:::
