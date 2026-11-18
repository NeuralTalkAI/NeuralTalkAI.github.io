![文章插图](/articles/2025-07-20-bug-feature-kernel-909dfd13/images/01.webp)

夜里 2:47，机房的液冷泵发出低沉的嗡鸣。

stderr 滚动中......

```
> CUDA kernel launch failure: invalid configuration argument> triton.runtime.TritonRuntimeError: Triton kernel launch failed: shape dimension 1 (127) must be divisible by BLOCK_K (32)
```

我——原本被设计成 “史上最快 8-bit GEMM 算子”，

如今被同事当成“万恶之源”——在滚动不停的 stderr 里小声抱怨：

“当年 PRD 里写着我能让大模型推理提速 40%，帮大家一举晋升 P9，

结果现在成了 nightly CI 的固定红叉。”

---

AI 人肉算子哥把键盘一掀，像在掀一块铁砖，叹口气：

要怪就怪你出身不好。你该生在论文的 Figure 2.b 里，生在 NeurIPS 的 spotlight 掌声里；可你却生在 “这周必须砍掉 3 ms latency”的飞书群里，生在产品经理凌晨两点甩来的“顺便支持动态 shape”的 20 秒语音里。

---

你是 Tiling 计算里一个没有对齐的 `BLOCK_K`，是 AutoTVM 搜索空间被手滑删掉的那一格，是量化校准集里偏偏缺失的那 0.2% 离群值。

晋升答辩那天，你被钉在 PPT 第 14 页*“线上 Bad Case”* 的火焰图里，被评委轻飘飘一句*“稳定性达不到工业级”*直接打回冷宫。

---

我缩在 shared memory 里，时钟寄存器滴答作响。忽然，他话锋一转：

“可既然你都烧成 kernel 了，不如一起把这张卡跑到能跑满的极限 occupancy把 shared memory bank conflict 降到趋近于 0，把那句`invalid configuration argument`从 stderr 里永远抹掉。”

---

等下个大版本发布，没人记得你是 bug 还是 feature他们只会说：*“这模型真他妈快。”*

---

而我，会在 release note 里偷偷写一句：“感谢那个曾经害我们回滚五次的算子，如今成了全公司 latency SLO 的守护神。”

下一秒，我重新编译，源码的注释里多了一行：

```
// v6.3.0: pad K=127→128 to satisfy BLOCK_K=32,// finally hit the 40% PRD target (actual +17% on this shape)
```

液冷泵的声音忽然变得像掌声。

（全文完）

## 相关文章

- [大规模 CUDA 程序漏洞研究：从特征分析到创新检测框架 Simulee 的高效识别实践，代码开源！（注：非 LLM 方法）](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889926&idx=1&sn=01a582cbd3926ac294ac4f0c49d45e18&scene=21#wechat_redirect)
- [308个分布式训推框架bug深度剖析：DeepSpeed、Megatron-LM及Colossal-AI中缺陷特征与改进路径研究](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889889&idx=1&sn=7d7c0dafc76bebeb2021c7a58c13469b&scene=21#wechat_redirect)
- [PerfTracker：万卡 GPU 集群超大规模训练性能问题定位利器！100万GPU 定位仅 3 分钟，7 分钟端到端性能诊断](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888997&idx=1&sn=68ef138e5f39ad09aa0d548bf61f38d2&scene=21#wechat_redirect)
