关键词：:mdnice-bold-italic[YOLO-Master]、**混合专家（MoE）**、实时目标检测、**动态路由**、ES-MoE

> 动态计算，按需分配：YOLO-Master 如何用 MoE 重新定义实时目标检测

实时目标检测（Real-Time Object Detection, RTOD）是计算机视觉领域的核心任务之一，从自动驾驶到工业质检，从安防监控到移动端应用，都离不开它的身影。YOLO 系列作为 RTOD 的佼佼者，凭借其极致的速度和不错的精度，几乎成为工业界的事实标准。

::: block-1
然而，传统的 YOLO 模型有一个固有缺陷：**静态密集计算**。无论输入图像是一片空旷的天空，还是一个拥挤的十字路口，:mdnice-bold-italic[模型都会分配相同的计算资源。这显然是一种浪费——简单场景不需要复杂的特征提取，而复杂场景却可能因为算力不足而漏检。]
:::

![文章插图](/articles/2026-02-19-yolo-master-yolo-moe-ai-2b0cd0a3/images/01.webp)

- YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection.
- **代码：https://github.com/Tencent/YOLO-Master**
- https://arxiv.org/pdf/2512.23273
- 6000 字，阅读 30 分钟，**播客 26 分钟**

相关推荐

- [YOLO26：十年架构演进回顾与关键性能改进综述](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897492&idx=1&sn=26e31c233a082014294239e0bb4762cf&scene=21#wechat_redirect)
- [推理＜2ms！Ultralytics最新YOLO26+树莓派+国产AI加速卡实现 500 FPS 端侧 AI 性能巅峰！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896506&idx=1&sn=69f0c1ba27e2c4945773f47ce9858cbb&scene=21#wechat_redirect)
- [ResNet/YOLOv8/11在Android骁龙 8 Gen2 异构硬件 CPU/GPU/NPU 量化-硬件-精度研究：NPU+INT8 量化竟能实现 298 倍加速！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895555&idx=1&sn=0cc7f40025edc564c11101b3ea9f6b24&scene=21#wechat_redirect)

> 人类视觉系统天生具有“自适应”能力：扫视空旷区域时，我们快速略过；聚焦复杂区域时，我们仔细分析。那么，能否让检测模型也学会这种“按需计算”呢？

2025 年底，腾讯优图实验室等机构的研究者给出了他们的答案——**YOLO-Master**。这项工作首次将混合专家（Mixture-of-Experts, MoE）深度集成到 YOLO 架构中，实现了**实例条件自适应计算**。简单来说，模型会根据每个样本的复杂度，动态激活不同的“专家”子网络，从而在精度和延迟之间达到更优的平衡。

![文章插图](/articles/2026-02-19-yolo-master-yolo-moe-ai-2b0cd0a3/images/02.webp)

::: image-caption
雷达图直观对比了 YOLOv10-N、YOLOv11-N、YOLOv12-N、YOLOv13-N 与 YOLO-Master-N 在 VOC、MS COCO、VisDrone、SKU-100K 和 KITTI 等数据集上的 mAP 性能。从各维度表现来看，YOLOv13-N 在多数指标上实现显著提升，尤其在 SKU-100K mAP50（90.3）和 KITTI mAP50-95（90.7）上表现突出，而 YOLO-Master-N 则在 VOC mAP50-95（62.1）等指标上展现出强劲竞争力，整体反映出新一代 YOLO 模型在多场景目标检测任务上的性能迭代优势。
:::
