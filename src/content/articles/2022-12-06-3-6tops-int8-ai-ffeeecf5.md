**在正（guang）文（gao）开始前（不是，先来看看两张表**

**下表统计一下目前千元内在售的 AI 主板（仅单主板，全套太贵）的 NPU 算力性能从高到低的排行（20221012）**

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/01.webp)

**下面是目前（20221014）市场上销量排行的典型 AI 开发产品。（已除去低成本 38 板摄像头和芯片原厂超昂贵开发板）**

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/02.webp)

---

好了，正（guang）文（gao）开始，可以重点关注一下支持的

算子列表和模型库（见文末）

大人！时代变了！！一款「3.6TOPS@INT8」视觉 AI 边缘 Linux 板卡入驻 SIPEED 开源社区啦！！！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/03.webp)

PART

01 **MAIX-III AXera-Pi 是什么？**

这是一款搭载「爱芯元智 AX620A」的视觉 AI 开发板，支持 400w 像素 AI ISP 夜景增强算法，在给用户带来了出色影像效果的同时，通过多元化的 AI 生态加持，使得在边缘设备部署模型更加快速和容易，比如

ResNet50、 YOLOv5s

、YOLOv7-Tiny、 SCRFD、 MonoFlex、HRNet、PP-HumanSeg 等模型已实现部署，未来会借助开源社区的力量把更多的“模型”进一步“落地”！这就是今天给大伙带来的新品「MAIX-III AXera-Pi」！（可简称 m3axpi ）

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/04.webp)

下文预计阅读时间1.什么是微光全彩？（阅读1分钟）2.什么是 AI 视觉？（阅读2分钟）3.我们带来了什么？（阅读3分钟）PART 02 什么是微光全彩？

先来看下面的这张图，这是传统去噪（左）和 AI 去噪（右）的效果对比，可以看到 AI 的图像处理效果在很多领域已经远超传统算法的极限了。

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/05.webp)

AI 微光全彩夜景增强效果

微光全彩是指在光线微弱无需补光就能呈现全彩画质，而

这款产品的摄像头在经过上述多种 AI ISP 图像算法增强后，就能够得到以下视频中「微光全彩」的效果。

以下视频均来自真实用户提供

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/06.webp)

400w像素有什么特别？

SIPEED 从 18 年 K210 30w 到 20 年 V831 200w 再到 22 年 AX620A 400w 像素增长，在这期间的画质有着明显的进步，不妨来看看这个成像进化的过程吧！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/07.webp)

（左上 30w 右上 200w 中下 400w）

**PART**

03 **什么是 AI 视觉？** 所谓的 AI 视觉是指将 AI 神经网络技术

应用到计算机视觉 CV 领域中， 如果说人工智能赋予计算机思考的能力，那么计算机视觉就是赋予发现、观察和理解的能力。

一直以来，传统计算机视觉

的泛化效果不够理想，识别过程中往往会受到环境光照干扰、目标的

大小与

场景变化、物体发生旋转倾斜等问题的影响。为了解决这些问题引入了 AI 神经网络模型，并

梳理出了计算机视觉的四大基本任务：分类、定位、检测、分割。

0 1 举个🌰子？

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/08.gif)

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/09.webp)

左图像分类（object classification)

给 AI 一张图，就能对这张图主体是猫、狗还是物体等进行分类并显示名称，如**图中的车**。

中图像定位（object localization）

给 AI 一张图，就可以用矩形框出这张图主体在图里的具体位置，如**图中被框起来的车**。

右图像检测（object detection）

可以同时对多个不同的物体进行分类和定位，将图里的事物分别是猫、狗、人脸或物品，通过矩形框出猫、狗、人脸或其他东西在图里的位置，如**图中被框起来的不同角度的两辆车**。

下图像分割（object segmentation）

当确定了物体的类别与定位，还可以进一步的把图像中需分割的目标从背景中分离出来，从下图来看类似一键抠图的效果。

接着我们来看看在 **服务器** 和 **开发板** 上的实际部署情况如何？

02 在服务器上的运行结果

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/08.gif)

如今在服务器部署上述 AI 模型已经不足为奇！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/10.webp)

03 在开发板的运行结果

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/08.gif)

在过去，因边缘设备算力普遍低于1TOPS，导致开发者往往需要开足马力才能实现勉强能用的分类、检测模型。而如今大幅提升的3.6TOPS算力能游刃有余的跑出与服务器近似的效果。

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/11.webp)

实测 YOLOV5-S 检测，在输入 640x640 分辨率的情况下只需要 12~22ms

（ 注：12ms 为不使用 AI-ISP 的情况）而 YOLOV7-Tiny 的则是8 ~ 11 ms。由此可见板端部署 Ai 模型推理速度快，兼容性不错、模型效果好，3.6T算力是真的香！

PART 04 **我们带来了什么？** 边缘 AI 社区

Sipeed 开源以边缘 AI 落地为核心目标，为社区用户提供 [MaixHub AI] 在线服务，提供 AI 开发指南、在线标注、模型在线训练、模型转换、模型部署、模型分享等在线内容。训练好的模型可在浏览器部署，也可部署到边缘设备，在线训练或本地训练好的 AI 模型可直接分享到模型库，未来也将持续引导社区里热情而又活跃的开发者们踊跃尝试和分享，为边缘AI生态贡献更多内容！

（左：在线标注 右：在线训练）

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/12.webp)

一键采集上传数据

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/13.webp)

体验一键部署模型

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/14.gif)

在线 AI 模型分享

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/15.webp)

多元 AI 生态（以下内容是合作的小伙伴们带来的）

****2022世界人工智能大会：爱芯元智发布“爱芯智眸®”**AI-ISP******

第五届世界人工智能大会（WAIC）于上海开幕。今年的大会以“智联世界 元生无界”为主题， 聚集了虚拟原生、数字孪生和空间计算等元宇宙技术，展现出人工智能+元宇宙领域的“硬核新科技、产业新赛道、未来新场景、治理新议题”的全新画卷。

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/16.webp)

**爱芯元智AX620A获百度飞桨【I】级兼容性认证**

飞桨（PaddlePaddle）以百度多年的深度学习技术研究和业务应用为基础，集深度学习核心训练和推理框架、基础模型库、端到端开发套件、丰富的工具组件于一体，是中国首个自主研发、功能丰富、开源开放的产业级深度学习平台。

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/17.webp)

**MAIX-III AXera-Pi & OpenMMLab**

OpenMMLab 诞生于 2018 年，是深度学习时代计算机视觉领域最全面、最具影响力的开源算法体系。旨在为学术和产业界提供一个可跨方向、结构精良、跨站性强、易复现的统一算法工具库。

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/18.webp)

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/19.webp)

更好的产品体验现在！体验Ai 视觉只需要【四步】！！！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/20.webp)

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/08.gif)

还有众多应用案例在等着你体验！~

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/21.webp)

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/22.webp)

以及大家关心的 Ai 模型部署指南

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/23.webp)

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/19.webp)

拥抱开源

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/24.webp)

如今在边缘 AI 嵌入式 Linux 芯片半导体领域获取

技术资料还存在一定的壁垒，个人玩家往往不具备这样的条件。

而 SIPEED 开源会在原厂授权的条件下，将会持续不断地开放和

更新源码、文档、案例等技术资料，让我们一同交流技术、一同进步、一同成长吧！！！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/25.webp)

**更多开发资料请点** **wiki.sipeed.com/m3axpi****（文末【阅读原文】）获取喔！**

新品上新活动大放送！~

复制 **ngv6dWYVkf1** 后前往淘宝购买！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/26.webp)

我们还为小伙伴们精心准备了以下**开箱测评送板子活动**！

![文章插图](/articles/2022-12-06-3-6tops-int8-ai-ffeeecf5/images/27.webp)

- **算子列表：https://pulsar-docs.readthedocs.io/zh_CN/latest/appendix/op_support_list.html**
- **模型库：https://maixhub.com/model/zoo**

更多开发资料请点wiki.sipeed.com/m3axpi

【阅读原文】获取喔！
