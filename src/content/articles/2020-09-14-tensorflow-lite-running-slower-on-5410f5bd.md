> Written by Michael Larabel in Software on 10 September 2020.
>
> https://www.phoronix.com/scan.php?page=article&item=tflite-kernel-regress
>
> **注：这篇文章转载phoronix的，简单总结一句话：Linux kernel一年来的改动，影响了TensorFlow Lite的性能，改动的地方在负载均衡的地方（后面有该commit和代码提交地址），作者做了一些初步的分析。**

**The Linux 5.0 to 5.9 kernel benchmarking** (https://www.phoronix.com/vr.php?view=29483) **posted this week showed TensorFlow Lite running slower since the Linux 5.5 kernel**... On top of looking at the new Linux 5.9 regressions(https://www.phoronix.com/scan.php?page=news_item&px=Kernel-Bisecting-5.9-Ongoing), I also spent some time bisecting and figuring out what happened for TensorFlow Lite last year that has at least for the system under test caused it to run slower for all the kernel releases this year as shown in the aforelinked article.

So with the speedy AMD EPYC box and knowing the regression happened between Linux 5.4 and 5.5, it was on to bisecting the TensorFlow Lite performance drop. TensorFlow Lite is the open-source deep learning framework for on-device inference. TensorFlow Lite is an optimized version of TensorFlow catering to mobile and edge devices. This testing is focused on TensorFlow Lite with not yet evaluating the full TensorFlow performance or other deep learning workloads across the tested kernel versions with regards to this regression.

![文章插图](/articles/2020-09-14-tensorflow-lite-running-slower-on-5410f5bd/images/01.webp)

**图1 明显看出5.5和5.4中间这道性能分界线（InceptionV4和Inception ResNetV2）**

With there being a clear difference across the 5.5 threshold, the TensorFlow Lite test profile(https://openbenchmarking.org/test/pts/tensorflow-lite) with the Phoronix Test Suite made for straightforward bisecting of the Linux Git state during the 5.5 merge window and set off to bisecting with the sixty-four core AMD EPYC processor.

![文章插图](/articles/2020-09-14-tensorflow-lite-running-slower-on-5410f5bd/images/02.webp)

**图2 Inception V4模型在v5.5和v5.4及前几个commit上的性能情况**

![文章插图](/articles/2020-09-14-tensorflow-lite-running-slower-on-5410f5bd/images/03.webp)

**图3 MobileNet Quant **模型在v5.5和v5.4及前几个commit上的性能情况****

![文章插图](/articles/2020-09-14-tensorflow-lite-running-slower-on-5410f5bd/images/04.webp)

**图4 Inception ResNetV2 **模型在v5.5和v5.4及前几个commit上的性能情况****

The bisecting revealed...

The culprit was narrowed down to sched/fair: Remove meaningless imbalance calculation(https://git.kernel.org/pub/scm/linux/kernel/git/tip/tip.git/commit/?id=fcf0553db6f4c79387864f6e4ab4a891601f395e).

![文章插图](/articles/2020-09-14-tensorflow-lite-running-slower-on-5410f5bd/images/05.webp)

**图5 引入该问题的commit**

Unfortunately, while the commit title may appear mundane, this sched/fair work is part of a much larger set of changes introduced back for Linux 5.5. In fact, the patch is part of the work covered last year on Phoronix and titled Linux 5.5's Scheduler Sees A Load Balancing Rework For Better Perf But Risks Regressions(https://www.phoronix.com/scan.php?page=news_item&px=Linux-5.5-Scheduler). This scheduler balancing rework is what appears to be to blame for the lower TensorFlow Lite performance we are seeing on the kernels over the past year.

That rework was quite an intrusive change last year and in the kernels since there has obviously only been further scheduler changes. With post-5.5 kernels we have seen some improvements made but still not to the pre-5.5 levels from the testing we have been doing on some of the more powerful boxes. Thus for now there is no easy/straightforward solution given the complexity of the change, thus moving on to looking at other active kernel regressions...

If you appreciate the relentless, daily Linux benchmarking on Phoronix you can show your support by joining Phoronix Premium for ad-free access and multi-page articles on a single page. PayPal tips are also welcome. Back to benchmarking and looking at some of the other open issues.
