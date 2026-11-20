**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：DeepSeek V4 Flash、**视觉微调**、:mdnice-bold-italic[NVFP4 量化]、MoonViT、**SGLang、vLLM**

> DeepSeek V4 Flash 发布以来一直被一件事卡着脖子：这个总参数 304B、每 token 只激活 13B 的 MoE 模型**只能读文字，看不见图**。:mdnice-bold-italic[浏览器智能体要读 UI 截图、自动化工具要识别界面元素，全都得绕道闭源多模态 API。]

8 月初，Hugging Face 上出现了社区仓库 `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`[1]，干的正是这件事——**给 V4 Flash 接上视觉前端，并用 NVFP4 量化压缩权重**。它被社区媒体评价为"补上了浏览器智能体不依赖闭源多模态 API 的缺口"。**同期还有 FlyCockpit 项目的 vLLM 路线**。

两条路线思路相似、实现不同，一起看下。

## 本文目录

- 一、底座：0731 是个什么样的模型
- 二、两条社区路线
  - 2.1 WebBrain：SGLang 打包版（即本次的模型仓库）
  - 2.2 FlyCockpit：2× DGX Spark 上的 vLLM playbook
  - 2.3 两条路线对比
- 三、它能做什么、不能做什么
  - 3.1 能力边界
  - 3.2 使用上的硬约束
- 总结

![文章插图](/articles/2026-08-10-deepseek-v4-flash-deepseek-v4-0da5fbf0/images/01.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、底座：0731 是个什么样的模型

> 两个视觉方案都建立在 `deepseek-ai/DeepSeek-V4-Flash-0731`[2] 之上，即 7 月 31 日的 V4 Flash 正式版。

架构与预览版一致：CSA + HCA 混合注意力、1M token 上下文，权重里直接捆绑了 DSpark 投机解码草稿头。提升全部来自后训练——Agent 类基准全面反超参数更大的 V4-Pro 预览版。

| 基准 | V4 Flash 预览版 | V4 Flash 0731 | V4-Pro 预览版 |
| --- | --- | --- | --- |
| Terminal Bench 2.1 | 61.8 | **82.7** | 72.1 |
| DeepSWE | 7.3 | **54.4** | 12.8 |
| Cybergym | 38.7 | **76.7** | 52.7 |
| Toolathlon-Verified | 49.7 | **70.3** | 55.9 |
| NL2Repo | 39.4 | **54.2** | 38.5 |

*表：DeepSeek 官方模型卡同口径数据，Code Agent 类任务使用 DeepSeek Harness 极简模式、max 档位，top_p 0.95、temperature 1.0。DeepSWE 涨幅最夸张——一次后训练翻了 7 倍多。*

![文章插图](/articles/2026-08-10-deepseek-v4-flash-deepseek-v4-0da5fbf0/images/02.png)

::: image-caption
DeepSeek-V4 架构示意。DeepSeek-V4 的 Transformer block 结构——输入 token 经 CSA/HCA 混合注意力与 DeepSeekMoE 前馈层，由预测头与 MTP 模块读出（来源：DeepSeek-V4 技术报告）。
:::

## 二、两条社区路线

给 0731 接眼睛👀（vision）的公开实现目前有两个：WebBrain 的 SGLang 打包版和 FlyCockpit 的 vLLM playbook。**两者都不重训大模型本体，只训练一个很小的"投影器"，** 把视觉特征映射进 0731 的嵌入空间。

### 2.1 WebBrain：SGLang 打包版（即本次的模型仓库）

> `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`[3] 是一个"开箱即用"取向的整包：内含 0731 的 NVFP4 文本权重、来自 Kimi K2.6 的 MoonViT 视觉塔、一个 40,119,040 参数（4096 维）的投影器，以及配套的 SGLang processor/model 代码。

它最有特色的部分是 DeepSeek 专用的输入路由桥：0731 的词表里没有图像 token，WebBrain 用一个词表外的 image sentinel（ID 129280）占位，在 prefill 阶段按图像位置循环替换成一组固定的 64 个 ID 的 palette，让图像特征能正确落进 DeepSeek 的嵌入流。投影器结构为 `pre_norm → 4608×4608 → GELU → 4608×4096`。

风险也很明确：包内自带的 `VISION_ADAPTER_MANIFEST.json` 白纸黑字写着 `gpu_validated_for_this_0731_package=false`——打包方自己声明这套组合尚未在完整 GPU 环境验证过，也没有公开任何视觉基准成绩。第三方审计项目的结论是"包装得很完整，但缺可复现的验证证据"。

### 2.2 FlyCockpit：2× DGX Spark 上的 vLLM playbook

> FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks[4] 走的是另一条路：面向本地部署的开源 playbook，配套开源视觉权重 `FlyCockpit/DeepSeek-V4-Flash-0731-vision`[5]。

它的视觉前端是一个约 865 MB 的冻结视觉塔（DeepEncoderV2）加一个 40 MB 的训练好的投影适配器（step 4800），通过 vLLM 插件 `dsv4_vision_vllm` 注册包装模型 `DeepseekV4VisionForCausalLM`。整个方案不动原始权重——模型目录用符号链接搭出来，只改 `config.json` 里的 `architectures` 字段。

硬件目标是两台 NVIDIA DGX Spark（GB10）做张量并行：官方 FP8 检查点约 167 GB，单台 Spark 的统一内存放不下，双机 TP=2 是支持的路径。冷启动约 6 分钟，实测生成速度约 40–50 tok/s，对外提供 OpenAI 兼容 API。

![文章插图](/articles/2026-08-10-deepseek-v4-flash-deepseek-v4-0da5fbf0/images/03.jpg)

::: image-caption
FlyCockpit 项目封面：两台 DGX Spark 驱动视觉版 DeepSeek-V4-Flash。FlyCockpit 项目的部署场景示意——两台 DGX Spark 跑起带视觉能力的 0731，"截图进、答案出，全在自己的硬件上"。
:::

这条路线还踩过一个有代表性的坑：上游插件最初会悄悄破坏 DSpark 投机解码——草稿模块照常跑，但接受率掉到 1%–15%，吞吐腰斩到约 20 tok/s。:mdnice-bold-italic[原因是视觉包装层挡住了主干网络，切断了 DSpark 草稿依赖的辅助隐状态流。修复办法是让包装层对主干透明（forward 里透传 `kwargs`、暴露 `lm_head`），接受率随即恢复到 50%–64%，平均接受长度约 2.0。]

### 2.3 两条路线对比

| 维度 | WebBrain（SGLang 整包） | FlyCockpit（vLLM playbook） |
| --- | --- | --- |
| 推理框架 | SGLang | vLLM |
| 视觉塔 | Kimi K2.6 MoonViT | DeepEncoderV2（约 865 MB） |
| 投影器 | 约 4011 万参数，4096 维 | 约 40 MB 适配器，step 4800 |
| 图像路由 | 词表外 sentinel + 64-ID palette 桥 | vLLM 插件注册包装模型 |
| 目标硬件 | 未明确（NVFP4 需 Blackwell 系 GPU） | 2× DGX Spark，TP=2 |
| 验证状态 | manifest 自标未经 GPU 验证，无公开基准 | 双机实测 40–50 tok/s，有部署记录 |

*表：两条路线同源不同路——都是"冻结主干 + 小投影器"，但框架、视觉塔和工程化程度差异明显。WebBrain 胜在整包即拿，FlyCockpit 胜在有完整实测。*

## 三、它能做什么、不能做什么

> 下面这些是来自 FlyCockpit 项目的信息了：

### 3.1 能力边界

| 任务 | 表现 |
| --- | --- |
| 描述截图与 UI | 强 |
| 读取界面布局与屏幕文字 | 强（实测转录接近满分） |
| 文档解析 | 强 |
| 日常照片 | 尚可，但描述泛泛 |
| click 级 computer-use 智能体 | **尚不成熟，上游明确声明未达生产水平** |

*表：FlyCockpit 项目的能力评估表（WebBrain 路线未公开对应数据）。合成 GUI 上的坐标定位可用，但迁移到真实网页会失效。一句话概括：截图专家，不是通用视觉模型。*

### 3.2 使用上的硬约束

| 约束项 | 具体规则 |
| --- | --- |
| 单次图片数 | 最多 8 张（含历史消息回放），第 9 张直接返回 HTTP 400 |
| 思考模式 | 图片请求必须显式传 `chat_template_kwargs: {"thinking": false}`，否则答案落进未闭合的 think 块、`content` 返回为空 |
| 图像 token 开销 | tiles=2 布局下每张图展开为 257 / 769 / 1281 个 token |
| 兼容底座 | 仅适配 0731 正式版，预览版及旧线量化版（含 NVIDIA 官方 `DeepSeek-V4-Flash-NVFP4`[6]）会因嵌入空间不同而输出乱码 |
| 实测速度 | 2× DGX Spark（TP=2）下约 40–50 tok/s，图片请求时 DSpark 接受率约 63% |

*表：以上规则来自 FlyCockpit 项目的部署记录（NVIDIA 开发者论坛原帖[7]）。开发者也注明其上下文管理仍有问题，"尚不适合生产环境"。*

## 总结

> V4 Flash 的 API 定价本就极低（输入 $0.14/M、输出$0.28/M），现在视觉这一环被社区补上后，:mdnice-bold-italic[一条"完全开源、本地部署、能看屏幕"的智能体推理栈已经成型：读截图、调工具、跑长上下文，全链路不需要碰任何闭源 API。]

当然，两条路线都还处在"能跑"而不是"能用"的阶段——WebBrain 的包自标未经 GPU 验证，FlyCockpit 的上下文管理有已知问题，click 级操作两边都不成熟。把截图识别类任务（UI 巡检、表单读取、界面回归测试）先交给它们，把关键操作留给成熟方案，是当前比较务实的用法。

参考资料

[1] `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`: *https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4*

[2] `deepseek-ai/DeepSeek-V4-Flash-0731`: *https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731*

[3] `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`: *https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4*

[4] FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks: *https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks*

[5] `FlyCockpit/DeepSeek-V4-Flash-0731-vision`: *https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision*

[6] `DeepSeek-V4-Flash-NVFP4`: *https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4*

[7] NVIDIA 开发者论坛原帖: *https://forums.developer.nvidia.com/t/deepseek-v4-flash-with-vision/379212*

::: related-title
**相关推荐**
:::

::: related-links
- **[仅81个视觉Token就能推理：DeepSeek把坐标写进思考过程，弥合多模态模型的逻辑鸿沟](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900649&idx=1&sn=b43d459bc03df111550c0dc767a3052a&scene=21#wechat_redirect)**
- **[训练-推理一致的高效多模态LLM训练：双模视觉Token修剪框架实现 4 倍加速与 99%性能保留，方案开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897010&idx=1&sn=90af12492cc6b0cccb3ef634d0d96076&scene=21#wechat_redirect)**
- **[最高16倍加速、精度近无损！阿里 VLCache 大幅提升视觉语言模型推理效率：只算 2%token，智能复用 98%缓存](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895822&idx=1&sn=affe73851d3f3abc853f0928c522d31f&scene=21#wechat_redirect)**
- **[Qwen2-VL-3B 模型在瑞芯微 RK3576 开发板 NPU 上多模态和多轮对话部署指导与评测](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891241&idx=1&sn=41525fa9c3773ee8949954f65663b87a&scene=21#wechat_redirect)**

:::
