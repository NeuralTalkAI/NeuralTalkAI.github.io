关键词：DeepSeek V4 Flash、**视觉微调**、:mdnice-bold-italic[NVFP4 量化]、MoonViT、**SGLang、vLLM**

> DeepSeek V4 Flash 发布以来一直被一件事卡着脖子：这个总参数 304B、每 token 只激活 13B 的 MoE 模型**只能读文字，看不见图**。:mdnice-bold-italic[浏览器智能体要读 UI 截图、自动化工具要识别界面元素，全都得绕道闭源多模态 API。]

8 月初，Hugging Face 上出现了社区仓库 `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`[1]，干的正是这件事——**给 V4 Flash 接上视觉前端，并用 NVFP4 量化压缩权重**。它被社区媒体评价为"补上了浏览器智能体不依赖闭源多模态 API 的缺口"。**同期还有 FlyCockpit 项目的 vLLM 路线**。

两条路线思路相似、实现不同，一起看下。

## 本文目录

- 一、底座：0731 是个什么样的模型
- 二、两条社区路线
- 三、它能做什么、不能做什么
- 总结

![文章插图](/articles/2026-08-10-deepseek-v4-flash-deepseek-v4-0da5fbf0/images/01.webp)

::: related-title
**相关推荐**
:::

::: related-links
- **[仅81个视觉Token就能推理：DeepSeek把坐标写进思考过程，弥合多模态模型的逻辑鸿沟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900649&idx=1&sn=b43d459bc03df111550c0dc767a3052a&scene=21#wechat_redirect)**
- **[训练-推理一致的高效多模态LLM训练：双模视觉Token修剪框架实现 4 倍加速与 99%性能保留，方案开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897010&idx=1&sn=90af12492cc6b0cccb3ef634d0d96076&scene=21#wechat_redirect)**
- **[最高16倍加速、精度近无损！阿里 VLCache 大幅提升视觉语言模型推理效率：只算 2%token，智能复用 98%缓存](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895822&idx=1&sn=affe73851d3f3abc853f0928c522d31f&scene=21#wechat_redirect)**
- **[Qwen2-VL-3B 模型在瑞芯微 RK3576 开发板 NPU 上多模态和多轮对话部署指导与评测](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891241&idx=1&sn=41525fa9c3773ee8949954f65663b87a&scene=21#wechat_redirect)**

:::
