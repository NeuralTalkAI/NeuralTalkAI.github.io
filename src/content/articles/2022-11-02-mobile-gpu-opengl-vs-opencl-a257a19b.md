> 2020年8月TensorFlow Blog关于TensorFlow Lite的文章，就是**提到OpenCL后端比OpenGL后端性能好，但事实真的是如此么？**前几天我们刚发布有关[《**ShaderNN推理框架：来自OPPO的着色器深度学习推理引擎》**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886556&idx=1&sn=33dfc3d92786acc81f540bb01de11239&chksm=b2904e3e85e7c728a5ebeadd78b6c0ddfdff46ccbc74a6b5e4e375768f57cf030a8cc73cd4c5&scene=21#wechat_redirect)**的介绍，** [**其GPU Kernel实现是基于OpenGL的Compute Shader（计算着色器），且性能好于TensorFlow Lite的OpenCL**](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886556&idx=1&sn=33dfc3d92786acc81f540bb01de11239&chksm=b2904e3e85e7c728a5ebeadd78b6c0ddfdff46ccbc74a6b5e4e375768f57cf030a8cc73cd4c5&scene=21#wechat_redirect)，也有实测的benchmark数据。到底谁才是对的。
>
> 我们先来看看TensorFlow这篇Blog《Even Faster Mobile GPU Inference with OpenCL》讲了什么，然后再看看OpenGL和OpenCL选择的背后逻辑。

虽然TensorFlow Lite (TFLite) GPU团队不断提高现有OpenGL-based移动GPU推理引擎，但同时也调研了其他相关技术。其中一个便是**OpenCL-based Android移动GPU推理引擎**，其**提供了相比现有的OpenGL后端2倍的加速**。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/01.webp)

**图1 Adreno GPU的OpenCL概念内存分层（左），Adreno A5x系列架构（右）**

## 1. OpenCL的特点

由于历史原因，OpenGL本是为渲染矢量图形的API，计算Shader的加入也是在OpenGL ES 3.1之后，其向后兼容的API设计限制了充分发挥GPU的潜力。另一方面，OpenCL从一开始就被用于加速计算，当然适用于移动GPU推理。因此，**TensorFlow Lite团队通过调研OpenCL的在移动GPU上的特点，发现其有如下优点**。

### 1.1 性能分析

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/02.webp)

**图2 性能剖析工具可以找到如GPU的ALU利用率、L1/L2 cache命中等信息**

**优化OpenCL后端比OpenGL容易得多****，因为OpenCL提供性能分析工具**，且Adreno也很好地支持这些功能。通过性能剖析API，能够非常精确地测量每个内核调度的性能。上图是《高通骁龙移动平台OpenCL通用编程优化手册》提到有关性能剖析工具的小结，**可以方便获取APP性能数据，不限于register footprint、total instructions、GPU busy ratio、ALU utilization ratio、L1/L2 cache hit ratio** 等等。

### 1.2 可优化的工作组大小

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/03.webp)

**图3 与性能相关的工作组因素**

通过TFLite团队的实验，高通**Adreno GPU的性能对工作组大小非常敏感**。选择适配任务的工作组大小可以提高性能，否则会降低性能。然而，对于具有复杂内存访问模式的计算任务来说，选择合适的工作组大小并非易事。在OpenCL中上述性能分析功能的帮助下，实现了一个**针对工作组大小的优化器，其速度比平均速度提高了50%**。

### 1.3 16位精度浮点（FP16）

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/04.webp)

**图4 不同精度、编译选项下的性能**

**OpenCL支持FP16**，并要求加速器如GPU指定数据类型的可用性。作为官方规范的一部分，即使是一些较旧的GPU，例如2012年的Adreno 305，也可以使用FP16。这个计算精度是可选的，依赖GPU本身是否支持。

### 1.4 常量内存

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/05.webp)

**图5 OpenCL memory model in Adreno GPUs**

**OpenCL有常量内存（constant memory）** 的概念。高通添加了一种物理内存，其特性使其非常适合与OpenCL的constant memory一起使用。实验证明，对于某些特殊情况如在网络开始或结束时feature map比较小的层。**Adreno上的OpenCL能够通过与此物理常数存储器和上述原生FP16支持的协同作用，大大超过OpenGL的性能**。

## 2. 性能比较

下面展示了TFLite在CPU（大内核上的单线程）、使用现有OpenGL后端及使用新OpenCL后端的GPU上的性能，**目的是突出OpenCL性能优于OpenGL与CPU**。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/06.webp)

**图6 安卓平台MNASNet 1.3推理性能比较：CPU、OpenGL、OpenCL**

图2和图3分别是在MNASNet 1.3和SSD MobileNet v3（大型））上使用OpenCL的选定Android设备上的性能。**新OpenCL后端的速度大约是OpenGL后端的两倍**，但在Adreno设备上表现得特别好（用SD标注，即SnapDragon），因为使用了前面提到的Adreno性能分析器调整了工作组大小。此外，图2和图3之间的差异表明，OpenCL在**更大的网络上****表现得更好**。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/07.webp)

**图7 安卓平台SSD MobileNet v3( large)推理性能比较：CPU、OpenGL、OpenCL**

## 3. OpenCL后端支持：简单、高效

**使用OpenCL推理引擎的一个主要障碍** **是OpenCL不是标准Android发行版的一部分**。虽然主流Android供应商将OpenCL作为其系统库的一部分，但OpenCL可能不适用于某些用户。所以对于这些设备，**需要回到每个Android设备上都可用的OpenGL后端**。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/08.webp)

**图8 Buffer vs. image in Adreno GPUs**

为了简化开发人员的工作，TensorFlow Lite团队对TFLite GPU Delegate添加了一些修改：首先在运行时检查OpenCL的可用性。如果可用则使用OpenCL后端，因为**它比OpenGL后端快得多；如果它不可用或无法加载，将返回到现有的OpenGL后端**。事实上，OpenCL后端自2019年年中以来一直在TensorFlow存储库中，并通过TFLite GPU delegate v2无缝集成。

**以上便是TensorFlow Lite团队对OpenCL后端支持的背景原因，下面我们来看看OpenGL后端。**

## 4. OpenGL's Compute Shader

> Compute shaders are a new feature of OpenGL 4.3 and are exposed via the new GLARBcompute_shader OpenGL extension.
>
> Shader是GPU上的小程序，或说是着色器程序，通常**用类C语言的GLSL编写**，GLSL**专为图形设计**，**包含了对矢量和矩阵操作的功能**。这些程序针对图形管道或者说流水线的**每个特定部分运行**，通俗些，Shader无非是将输入处理后转换为输出的程序。
>
> 在Shader程序的实现上，以版本声明开始，然后是输入和输出变量、uniform（uniform是将数据从CPU上的应用程序传递到GPU上的着色器的另一种方式）及其main函数列表。每个Shader程序的入口点都在其main函数处。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/09.webp)

**图9 Compute Shader简介：通用计算shader、处理图形数据等**

Compute Shader可以利用OpenGL的图形处理器的强大运算能力。同OpenGL中的其他shader，他们是使用GLSL编写并且并行的按照分组的形式进行运行，通过该方式能同时处理大量数据。此外**，** **compute shader里还能访问纹理、访问storage buffer以及原子操作，compute shader可以与其他的compute shader进行同步并且他们之间还可以共享数据**，如此一来，**使用compute shader来进行通用计算的时候会变得更加的简单**。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/10.webp)

**图10 OpenGL4.3中的Compute Shader的执行流水线**

**compute shader不在OpenGL的管线里面**，其设计初衷是让开发人员能够足够灵活的去使用compute shader。《OpenGL SuperBible: Comprehensive Tutorial and Reference》第三章，“Following the Pipeline”中看到了关于使用compute shader来执行非图形渲染的工作。事实上，compute shader使用的是它自己的一套管线，它**与OpenGL的管线的其他部分是脱节的，即它没有固定的输入或者输出，没有任何固定管线模块的管理API，非常灵活，并且具有其它的shader不具备的能力**。

计算着色器本身通过work item定义了它操作的数据“空间”。OpenGL函数可用于定义同时启动计算操作执行的执行量。计算着色器没有用户定义的输入或其他着色器已知的任何输出。要将数据传递给计算着色器，着色器需要例如通过纹理访问、图像加载或着色器存储块访问来获取数据，以便将计算数据显式写入图像或着色器存储器块。

| Stage | Data Element |
| --- | --- |
| Vertex Shader | per vertex |
| Tessellation Control Shader | per vertex (in a patch) |
| Tessellation Evaluation Shader | per vertex (in a patch) |
| Geometry Shader | per primitive |
| Fragment Shader | per fragment |
| Compute Shader | per (abstract) "work item" |

**表1 Computer shader中的数据元素是work item**

**计算空间（Compute Space）是用户使用一个称为工作组（workgroup）的概念来定义计算着色器正在操作的空间，类似OpenCL**。工作组是用户可以（从主机应用程序）执行的最小数量的计算操作。如果工作组的空间是三维空间（“X”、“Y”、“Z”），用户可以将任意维度设置为1，以在一维或二维空间中执行计算。在下图中，每个绿色立方体都是一个工作组。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/11.webp)

**图11 global与local的组织**

虽然有X、Y、Z，但在执行工作组期间，顺序可能会任意变化，程序不应依赖于处理单个组的顺序。工作组可能包含许多计算着色器调用。着色器的调用量由工作组的局部大小定义，工作组也是三维的。上图显示了每个工作组如何在由红色立方体表示的本地空间调用中进行拆分。

例如，给定计算着色器的local大小为（128，1，1），并以工作组计数（16，8，64）执行着色器。着色器将被单独调用1048576次。这是工作组尺寸乘以计算着色器的局部大小的乘积：（ `128*1*1*16*8*64=1048576`）。每个调用都可以由一组唯一的输入唯一标识。虽然可以在特定工作组中的不同调用之间使用共享变量和特殊函数进行通信，但在不同工作组之间进行通信要可能会导致死锁问题。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/12.webp)

**图12 Compute Shader在图像处理上的应用：高斯模糊、更快的shared memory**

从编程的角度来说，**compute shader跟其他的shader都是一样的。它使用GLSL进行编写并表示为shader对象，然后链接成一个GPU程序**。当创建一个compute shader时，需要调用 `glCreateShader`并且传入 `GL_COMPUTE_SHADER`参数作为shader类型，这会得到一个shader对象的返回值，然后给shader对象里通过 `glShaderSource`传入shader的代码，然后通过 `glCompileShader`进行编译，并且通过 `glAttachShader`把它挂载到一个GPU程序上去。然后像是链接其他的GPU程序一样通过 `glLinkProgram`来链接这个GPU程序。

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/13.webp)

**图13 使用Compute Shader实现物理仿真**

需要注意的是，不能把compute shader和其他类型的shader相混淆。比如说，你无法把一个compute shader挂载到一个挂载了vertex shader或者fragment shader的GPU程序上去然后链接这个GPU程序。如果这么干了，链接这一步将会失败。也就是说，**一个GPU程序只能有compute shader或者是图形管线相关的shader，一个GPU程序无法混合这两种shader**，包含了compute shader的GPU程序也称为compute GPU程序。

## 5. Compute Shader vs. OpenCL

前面也提到了，Compute shader与其他OpenGL shader类型不同，计算着色器与计算机图形不直接相关，并**提供了更直接的底层硬件抽象**，类似于CUDA和OpenCL。在Stackoverflow上有个问题《What is the difference between OpenCL and OpenGL's compute shader》提到了，Compute shader提供了**可定制的工作组大小、共享内存、组内同步以及CUDA和OpenCL所熟知的所有内容**。二者主要区别：

1. Compute Shader使用GLSL编写而非OpenCL C。虽然语言的区别不说大，但在Compute shader下，还可以使用OpenCL无法使用的所有与图形相关的GLSL函数，例如高级纹理类型（例如立方体贴图数组）、高级过滤（例如mipmapping）、以及如4x4的小矩阵计算或几何函数这样的便利性。其实随着OpenCL也提供了几何函数计算等等，但是**本质是OpenCL没有OpenGL的函数接口，无法操作已有的纹理对象**（这里在考虑 `clCreateFromGLBuffer`这样互操作的扩展前提下而言）。
2. Computer Shader本身和其他Shader一样，是一个着色器对象。**那就意味着操作GL数据如buffer、texture、images会很平滑**，虽然OpenCL也提供Image数据类型，但前者更加直接，不需要中间转换。可能原本你用OpenGL做图形渲染中间有计算用的是OpenCL，但是**现在有Compute Shader你不需要用OpenCL了，因为用了OpenCL你还得配置环境等等乱七八糟一堆事**。

归根结底，**Compute shader虽然和CUDA、OpenCL一样都有通用计算能力，但其本身是推荐在现有OpenGL渲染等任务的应用程序中用的**。OpenGL的Computer Shader是对OpenGL没有通用计算能力shader的一种补充，**补充的是OpenGL**。**但当任务中既有图形图像渲染也有计算，此时就是计算着色器发光发热的时候。**

![文章插图](/articles/2022-11-02-mobile-gpu-opengl-vs-opencl-a257a19b/images/14.webp)

**图14 OpenGL与OpenCL的比较**

性能方面，在community.khronos.org论坛里也有人提到，**在某平台计算着色器比 OpenCL 慢约 20%。有人是这么回答的**：

OpenGL计算着色器和OpenCL内核背后可能有一个**完全不同的编译器**；并非OpenCL中存在的所有计算功能都在OpenGL中可用。更不用说，OpenCL中处理同步的方式与OpenGL有很大不同。它甚至可能**因硬件而异**。

在计算着色器（即shaderstoragebuffer或imageStore/imageLoad或imageStore/texelFetch）中**使用全局内存的策略是什么？** **所有这些都是选项**。就个人而言，对于只读数据，我更喜欢使用纹理提取，因为在某些硬件上，存储缓冲区或加载/存储图像的路径可能与R/W数据源不同。此外，存储缓冲区和加载/存储映像实现之间也可能存在差异，因为后者具有固定的元素大小，而前者根本没有元素的定义，因此特别是动态索引可能会导致这两种情况下的不同性能。另一件事是，存储缓冲区、图像缓冲区和纹理缓冲区访问线性内存，而其他图像和纹理通常访问平铺内存，因此性能也可能存在巨大差异。

> 综上所述，**OpenCL是否优于OpenGL计算着色器，或者OpenGL的计算着色器是否优于片段着色器，没有一个普遍的答案。这完全取决于硬件、驱动程序、着色器代码以及您想要解决的问题。**

根据从开发人员那里听到的信息，如果你想在已经使用OpenGL的图形应用程序中做一些计算工作，最好不要使用OpenCL GL互操作，因为互操作性能通常很差，与GPU生成或供应商无关。

二者同样算法类似代码实现，讲道理从本质上来说，OpenGL与OpenCL在同样线程数、精度、SRAM使用、数据排布等条件下，他们应该运行一样快。但**实际，非常依赖厂商对设备与驱动的优化在二者上的支持优化粒度**。

## 参考

1. **Even Faster Mobile GPU Inference with OpenCL | blog.tensorflow.org** https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html
2. What is the difference between OpenCL and OpenGL's compute shader | stackoverflow https://stackoverflow.com/questions/15868498/what-is-the-difference-between-opencl-and-opengls-compute-shader
3. Chapter 10. Compute Shaders| OpenGL SuperBible: Comprehensive Tutorial and Reference, Seventh Edition
4. OpenGL Compute Shaders vs OpenCL | community.khronos.org https://community.khronos.org/t/opengl-compute-shaders-vs-opencl/69114/11
5. OpenGL 4.3: Compute Shaders Details | Geeks3D https://www.geeks3d.com/20120806/opengl-4-3-compute-shaders-details/
6. Qualcomm® Snapdragon™ Mobile Platform OpenCL General Programming and Optimization

## 相关文章

- [华为诺亚方舟实验室 Bolt v1.2.1 在 QualComm GPU 上的优化总结](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886197&idx=1&sn=5e99d9a8fc07d16047e4924b72830bf4&chksm=b290419785e7c881bd0dfbadc29c69cc1004ade09225057463c4a35f3da4771d39831e627365&scene=21#wechat_redirect)
- [ShaderNN：来自OPPO的着色器深度学习推理引擎](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447886556&idx=1&sn=33dfc3d92786acc81f540bb01de11239&chksm=b2904e3e85e7c728a5ebeadd78b6c0ddfdff46ccbc74a6b5e4e375768f57cf030a8cc73cd4c5&scene=21#wechat_redirect)
- [华为端引擎Bolt GPU性能优化：让上帝帮忙掷骰子](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885381&idx=1&sn=cab9666a15cc0012732b537c52d9ca89&chksm=b29042a785e7cbb1eacb9a3f0a5b842973c1f3e1ba8084405223ae02c2f8daed8165c376503d&scene=21#wechat_redirect)
- [移动端GPU矩阵乘优化](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885554&idx=1&sn=7f3078d27555c151e05aa967b5de68b8&chksm=b290421085e7cb06432f9d4fe2e7009819a3cd322c4e7cb86fff7ceaeddd31c2ad451ed9d4a5&scene=21#wechat_redirect)
- [深入浅出 | 谈谈MNN GPU性能优化策略](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885606&idx=1&sn=4dbf5d6c52018485eea4076d4b61c1de&chksm=b29043c485e7cad256f73405a2fafecead5f36a99da7dc43ad658c4cf0fa7c5d44ab4f9da677&scene=21#wechat_redirect)
- [Arm Mali GPU 四大微架构概述](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447884956&idx=1&sn=b0fd51f51a5dfd47075d4bb0f545fda2&chksm=b290447e85e7cd68c4dd8db5461c138996eb25a0bdbe238c59b0abe5b26f50f4dde02642f163&scene=21#wechat_redirect)
- [从GPU谈异构，这是ARM处理器架构师的趣谈](http://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447885856&idx=1&sn=fe738c91a44d69ddc2e7ea3e1c2ef9f6&chksm=b29040c285e7c9d43b7afce201c8f5d548964faf808b3941bb899e044629c3e58621a75465a9&scene=21#wechat_redirect)

> 点击【阅读原文】，查看往期文章
