关键词：**MLSys**、:mdnice-bold-italic[苗旭鹏课题组]、北京大学、**博士招生**、**实习生博后**

> 机器学习系统（MLSys）方向的科研 er 看过来！北京大学计算机学院苗旭鹏助理教授课题组开启招生，面向全球招募博士、硕士、博士后及实习生，聚焦机器学习系统核心研究，实验室资源丰富（特指 GPU）、国际合作紧密，期待志同道合的你加入～

![文章插图](/articles/2026-02-09-mlsys-mlsys-f73d4ec6/images/01.webp)

- 北大苗旭鹏课题组 MLSys 方向全球招生！博士/硕士/博后/实习生长期开放
- 👉 导师个人主页：https://hsword.github.io
- 📧 联系邮箱：xupeng.miao@pku.edu.cn
- 2000 字，阅读 10 分钟，**播客 7 分钟**

相关推荐

- **[性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)**
- **[从训练到部署的全链路优化：腾讯 AngelSlim 让大模型推理成本降低 47% 以上](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896146&idx=1&sn=8adb92abaa9c78fbc3f31a69c1b8218b&scene=21#wechat_redirect)**
- **[HotChips 2025 从摩尔定律到巨型内核：GPU 上机器学习系统优化的十年跃迁，Zhihao Jia 演讲解读](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891453&idx=1&sn=632b9abf4170c20af52b481ca8ec24e7&scene=21#wechat_redirect)**

> 最近正式从 Purdue 辞职回国啦，接下来我会长期在北大展开科研和招生工作，特来广而告之。

有朋友好奇我为什么回国，不同的人可能关注不同的因素，但对我个人来说，答案其实很简单。

回国，不一定需要什么复杂的理由。

在这个每天都在见证历史、全球科技竞争愈发激烈的时代，能够回到自己的祖国，在自己熟悉的土壤上做研究、带学生、一起探索未知，本身就是一件值得、也应该去做的事情。

- 十年前的夏天，北大 CS 是我 MLSys 研究的**起点**；
- 十年后的今天，我重新回到这里，希望带更多人把这条路走得更宽，更远。

::: block-1
从 2026 年开始，我会在北京大学计算机学院组建并带领一个专注于机器学习系统（MLSys）的研究团队，团队持续招收新成员（博/硕/博后/实习生）。如果你对这个方向感兴趣，也希望参与到顶级系统研究中来，或者有相关的合作想法，欢迎邮件联系我！
:::

## 本文目录

- 一、导师简介
- 二、研究成果与学术任职
  - 核心开源项目
  - 学术论文
  - 学术任职
  - 荣誉奖项
- 三、课题组招生详情
  - 招生岗位
  - 核心研究方向
  - 实验室优势
- 四、报名要求
- 五、申请方式
- 六、常见问题 FAQ
  - 博士项目申请
  - 本科/硕士学生申请
  - 邮件未收到回复

![文章插图](/articles/2026-02-09-mlsys-mlsys-f73d4ec6/images/02.webp)

## 一、导师简介

> 苗旭鹏（Xupeng Miao）博士，**北京大学计算机学院助理教授、研究员、博士生导师**，国家级高层次青年人才项目获得者。

苗旭鹏博士长期深耕机器学习系统（MLSys）领域，核心研究方向为机器学习系统、数据管理与分布式计算，致力于通过系统设计与优化，提升 AI 模型/算法在硬件设备上的执行效率，降低人工智能的应用成本与技术门槛。

::: block-1
在此之前，他在普渡大学[1]（Purdue University）普渡大学计算机系[2]担任Kevin C. and Suzanne L. Kahn New Frontiers 助理教授。他于2022年在北京大学[3]计算机学院[4]获得博士学位，导师是崔斌教授[5]，毕业后曾在卡内基梅隆大学（CMU）担任博士后研究员，与 Zhihao Jia[6] 和 Tianqi Chen[7] 教授及其研究团队一起工作。
:::

## 二、研究成果与学术任职

他的主要研究兴趣包括机器学习系统，数据管理和分布式计算。

### 核心开源项目

> 主导/参与多款知名开源机器学习系统的设计与研发：

| 项目 | 核心介绍 | 项目地址 |
| --- | --- | --- |
| Hetu，高效分布式深度学习系统 | 由北大DAIR Lab开发，支持数万亿参数深度学习模型训练，兼顾工业界高可用性与学术界创新性，具备多项先进特性 | https://github.com/PKU-DAIR/Hetu |
| Galvatron，自动并行训练框架 | 由北大河图团队发布，该分布式训练系统获 VLDB 2023 接收，支持 DP、SDP、TP、PP 四种并行方式，以决策树分解搜索空间、动态规划优化等技术，精准估计开销，兼容 PyTorch，仅需几行代码实现自动并行，在多 GPU 环境下比现有方案训练吞吐量显著提升 | https://github.com/PKU-DAIR/Hetu/tree/main/tools/Galvatron |
| FlexFlow Serve，大模型推理引擎 | 自动搜索高效并行化策略加速分布式DNN训练，可直接替代PyTorch和TensorFlow Keras，现有程序仅需少量修改即可运行 | https://github.com/flexflow/FlexFlow |
| Mirage，AI编译器 | 革新性GPU多层超级优化器，无需CUDA/Triton编程，能提升开发效率、优化性能、保障内核正确性，为不同GPU架构定制专属内核 | https://github.com/mirage-project/mirage |
| Angel，分布式机器学习平台 | 腾讯与北大联合研发，基于参数服务器理念，经大数据性能调优，高可用稳定，擅处理高维模型，适配多种算法，支持Yarn和Spark | https://github.com/Angel-ML/angel |

### 学术论文

在 OSDI/SOSP（操作系统）、ASPLOS/NSDI（分布式计算）、SIGMOD/VLDB（数据管理）等计算机系统领域国际顶级学术会议和期刊发表论文**50 余篇**，并在 ICML’24、SIGMOD’24、ASPLOS’25 等国际大会做 Tutorial 专题报告。

详见：https://hsword.github.io/publications/

### 学术任职

- MLSys’25、KDD’25 国际会议 Artifact Evaluation 共同主席
- 多个计算机系统领域国际顶级会议/期刊评审专家

### 荣誉奖项

苗旭鹏博士斩获多项国内外重磅学术与科研奖项，包括： 2022 ACM 中国优秀博士论文奖（全国奖）、VLDB 2022 最佳论文奖、ASPLOS 2024 杰出成果奖、ACL 2024 杰出论文奖、2024 世界人工智能大会（WAIC）云帆奖·璀璨明星、IEEE Micro Top Picks 2025 荣誉提名奖[8]、2025 NSF POSE Award[9]、2025 Amazon Research Award[10]、普渡大学荣誉命名助理教授职位等。

详见：https://hsword.github.io/honors/

## 三、课题组招生详情

### 招生岗位

> 实验室常年有博士和硕士研究生名额，与海外名校的顶级实验室（如CMU Catalyst）联系紧密，工业界合作广泛，**拥有丰富的 GPU 计算资源**，欢迎感兴趣的同学直接邮件联系。

1. **博士研究生**：计划招收 2027 年秋季入学博士研究生若干名
2. **访问/实习学生**：长期欢迎本科、硕士、博士阶段同学前来实验室访问、实习与合作
3. **博士后研究员**：长期开放招聘
4. **远程实习生**：支持远程合作，面向全球对研究方向感兴趣的科研 er

### 核心研究方向

聚焦**机器学习系统（System for ML）优化**，包括但不限于：

- 大规模 AI 模型训练系统
- 大模型推理系统
- AI 编译器研发

### 实验室优势

1. 常年保有博士、硕士研究生招生名额，科研培养体系完善
2. 与海外顶级实验室（如 CMU Catalyst）联系紧密，拥有丰富的国际合作资源
3. 工业界合作广泛，科研成果落地性强
4. 配备充足的 GPU 计算资源，为科研工作提供坚实硬件支撑

## 四、报名要求

1. 对机器学习系统（MLSys）研究具备基本认知，且拥有强烈的研究兴趣；
2. 具备扎实的编程能力，熟练掌握**C++和 CUDA**：:mdnice-bold-italic[有程序设计竞赛获奖经历、开源项目贡献；]
3. 拥有较强的自主学习能力和科研探索精神，能独立开展部分研究工作；
4. 具备良好的沟通表达能力和团队合作精神，能与课题组伙伴高效协作。

## 五、申请方式

课题组长期接收申请，**符合要求的同学可直接将个人简历（CV）、成绩单等材料发送至导师邮箱**：xupeng.miao@pku.edu.cn，邮件主题建议注明「姓名-申请 XX 岗位-MLSys 招生」。

> I am actively looking for strong and self-motivated Postdoc/Ph.D./Master students, postdocs, and (remote) interns interested in **building systems for machine learning** to join my group.

## 六、常见问题 FAQ

因日常邮件量较大，导师无法逐一回复所有咨询，以下常见问题可解答你的疑惑：

### 博士项目申请

无论是否提前联系导师，请按照北京大学计算机科学系官方网站的指示准备申请材料，申请博士项目，在申请中注明苗旭鹏博士为第一意向导师。

### 本科/硕士学生申请

课题组为有科研热情的本硕学生提供项目参与机会，可直接发送个人简历至导师邮箱；若暂无可匹配的研究项目，导师可能无法及时回复，敬请谅解。

### 邮件未收到回复

可能存在导师遗漏邮件的情况，若发送邮件两周后仍未收到回复，可再次发送邮件提醒；若二次提醒后仍无回复，大概率是导师已查阅邮件，但因咨询量过大无法逐一回应，感谢理解。

::: block-1
北大苗旭鹏课题组深耕机器学习系统前沿研究，为科研 er 提供优质的科研平台、丰富的资源支持和广阔的发展空间，期待对 MLSys 方向满怀热情的你加入，一起探索人工智能系统的无限可能！
:::

参考资料

[1] 普渡大学: *https://www.purdue.edu/*

[2] cs.purdue.edu: *https://www.cs.purdue.edu/*

[3] 北京大学: *https://pku.edu.cn/*

[4] 计算机学院: *https://cs.pku.edu.cn/*

[5] 崔斌教授: *https://cuibinpku.github.io/*

[6] Zhihao Jia: *https://www.cs.cmu.edu/~zhihaoj2*

[7] Tianqi Chen: *https://tqchen.com/*

[8] IEEE Micro Top Picks 2025 荣誉提名奖: *https://dl.acm.org/doi/10.1145/3620665.3640411*

[9] 2025 NSF POSE Award: *https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2449151*

[10] 2025 Amazon Research Award: *https://www.amazon.science/research-areas/latest-news/63-amazon-research-award-recipients-announced-spring-2025*

相关推荐

- **[重读《机器学习科研的十年》：未来最难的，还是保持初心和热情！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447892043&idx=1&sn=fb5fe29f30a2155db388a13e01d44f3e&scene=21#wechat_redirect)**
- **[HotChips 2025：从 PyTorch 到 CUDA：大模型时代 GPU 内核自动生成的技术跃迁，Tri Dao最新演讲](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891347&idx=1&sn=173b8638e00da3610dd90ba7d5a4b7f6&scene=21#wechat_redirect)**
- **[“历史性的项目，错过了遗憾终生”：清华特奖游凯超与 vLLM 的三次蜕变与技术远征](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896591&idx=1&sn=689f8a0351d8ea9ed0c2c0ebd2f89ca2&scene=21#wechat_redirect)**

加群
