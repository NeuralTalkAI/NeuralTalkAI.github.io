拆箱大吉

最近新入手一枚AI开发板， 这是一款搭载「爱芯元智 AX620A」的视觉 AI 开发板，支持 400w 像素 AI ISP 夜景增强算法，在给用户带来了出色影像效果的同时，通过多元化的 AI 生态加持，使得在边缘设备部署模型更加快速和容易，比如ResNet50、

YOLOv5s

、 YOLOv6s、YOLOX、 YOLOv7-Tiny、 SCRFD、MonoFlex、HRNet、PP-HumanSeg 等模型已实现部署，未来会借助开源社区的力量把更多的“模型”进一步“落地”！

今天给大家测评「MAIX-III AXera-Pi」，最近还有优惠活动，详情请点击：**《** **[推荐一款国产「3.6TOPS@INT8」视觉 AI 开发板：易部署资料多，支持微光全彩，文末送开发板](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886798&idx=1&sn=02400bac1dba21c61cc003e24b21c714&chksm=b2904f2c85e7c63a707948032802825d5f3a6b75858c862796e62197853d9318589a0e77fd1d&scene=21#wechat_redirect)》**

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/01.webp)

套餐配置

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/02.webp)

**产品文档**

*https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/axpi.html*

附上一张官网图片，(*^▽^*)

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/03.webp)

安装步骤

**step-1->启动linux系统**

购买全功能套餐的同学们跳过装配步骤点击前往上电启动 AXera-Pi 设备登录 Linux 系统即可，小呆就是这些同学中的一员，小呆在测试的时候，直接从启动 AXera-Pi 设备登录 Linux 系统这步开始了。

step-2->登陆工具安装

系统推荐了两种工具， 小呆使用了MobaXterm， windows系统下，点点点一键安装完成

step-3->登陆到板子里

官方提供了两种方式登陆， USB-UART 串口 或 USB-OTG SSH 登录， Maix-III AXera-Pi 开发板的 Linux debian11 系统默认使用 root 用户登录。用户名为 root，密码为 root，目前板子接入电脑端上电启动后支持以下登录 Linux 系统方式。

小呆选择用【有线 串口 serial 登陆】，需要安装【CH340驱动】，按照官网描述步骤，一顿点点点操作后

完成。

如果使用【SSH】方式登陆，需要先完成网络配置， 连上网络。

step-4->网络配置

官方提供了三种配置方法，USB RNDIS（usb0）/有线以太网（eth0）/无线WIFI（wlan0）。小呆使用第一种方式，USB RNDIS（usb0）配置方法，需要安装【 rndis 驱动】，按照官网操作，又是一顿点点点操作，完成配置。

遇到的两个常见问题

写在最前面，遇到问题先查文档，官方已给出常见问题以及解决办法，链接如下：

https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/faq_axpi.html

问题一：运行后屏幕花屏

查看摄像头型号参数是否配置正确，摄像头选项“-c”需要与摄像头型号相配的参与，对应maix-lll板子，参数为“-c 2”。

问题二：屏幕图像倒立

物理旋转即可解决，小呆的板子是有外壳的，断电拆除过程中，拆下了链接线，误接反了，导致通电后连接不上板子，摄像头发烫，-_-||，-_-||， 立刻断电，还好板子质量上佳，降温后，正常使用，

提醒尽量不要拆摄像头连接线，如果拆除一定要注意方向，切勿接反。

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/04.webp)

摄像头功能展示

代码使用产品文档里【试试python编程目录下】第一个例子，文档链接如下：

https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/python_api.html

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/05.webp)

关灯模拟暗示环境，自动启动了夜间增强效果

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/06.webp)

换一组背景

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/07.webp)

推理结果展示

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/08.webp)

内置应用-IPC demo 测试结果展示文档链接

https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/basic_usag

e.html#

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/09.webp)

配置启动

- 命令启动，/opt/bin/IPCDemo/run.sh /opt/bin/IPCDemo/config/gc4653_config.json
- 浏览器访问 192.168.233.1:8080进入登陆页面
- ipad找了一段道路视频，模拟测试一下

结果展示

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/10.webp)

![文章插图](/articles/2022-12-23-ai-maix-iii-axera-pi-6936eb7a/images/11.webp)

相关文章

- **[推荐一款国产「3.6TOPS@INT8」视觉 AI 开发板：易部署资料多，支持微光全彩，文末送开发板](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886798&idx=1&sn=02400bac1dba21c61cc003e24b21c714&chksm=b2904f2c85e7c63a707948032802825d5f3a6b75858c862796e62197853d9318589a0e77fd1d&scene=21#wechat_redirect)**
- **[国产AI开发板评测活动：CSK6系视觉AI开发套件——NPU+DSP+MCU异构架构](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886510&idx=1&sn=65320678981138b69d25050968991ca9&chksm=b2904e4c85e7c75a3ea7cae410dea3d3fb2ccc39202f754a2f658b3454047be2c5991d9e1b26&scene=21#wechat_redirect)**
- [**xNN：支付宝端侧深度学习框架**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887094&idx=1&sn=7efae4ff5007a8db326560adf21ce82c&chksm=b2904c1485e7c5022c66b5f12a4d9a2eac1be64671fce5571e8628240930d12dc56fe792f416&scene=21#wechat_redirect)
- [**为 Linux 6.2 引入的AI加速器子系统**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447887080&idx=1&sn=1ac8444063b2d3e458e9a11a229dc604&chksm=b2904c0a85e7c51ca69ee667b541410d62f050c2da92e1220262a3132cfeb38cc2997eb64826&scene=21#wechat_redirect)
- [**端侧模型训练54分钟——完成声音网络模型调优**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886298&idx=1&sn=fa2497be977edbfc452cb4b993ad6287&chksm=b290413885e7c82e5464d681074a9f20402191b50be31b41516e3128d4cdbb211f4e443d2768&scene=21#wechat_redirect)
- [**网易有道开源EMLL：高性能端侧机器学习计算库，大幅提高计算性能**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885858&idx=1&sn=bed49eb1ab50320809760cd50615aab7&chksm=b29040c085e7c9d6172a02b0d1e1294820e60e58a10e5fb4991dbf1bd20d7ba1a61a02e58227&scene=21#wechat_redirect)[**深度学习端侧推理框架Bolt v1.0发布，领跑端侧AI性能与应用**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885344&idx=1&sn=cea1a671c5ca723c154c87d7f9eefe3b&chksm=b29042c285e7cbd4c3e5eff0c9560bce239d6361917d6e16e5a55b6feb2ca5f01538e0c661ed&scene=21#wechat_redirect)

- 【阅读原文】，查看历史文章 -
