> 前阵子看到Tengine为OpenCV4.3版本贡献了[ARM CPU底层汇编代码，加速深度学习计算](https://mp.weixin.qq.com/s?__biz=MzUxOTEwNjM2Nw==&mid=2247484982&idx=1&sn=4b67991ba01bd9cdb3f869b1dd7beaa8&scene=21#wechat_redirect)。最近也看到Tengine的不少同学在做相关PR。可能有小伙伴不了解Tengine。根据ARM官网也有介绍Tengine，其介绍如下。Tengine 是OPEN AI LAB 针对前端智能设备开发的软件开发包，核心部分是一个轻量级，模块化，高性能的AI 推断引擎，并支持用DLA、GPU、xPU作为硬件加速计算资源异构加速。

话不多说，还是先试试怎么样。

首先找到github上Tengine主页：github.com/OAID/Tengine和文档页面，其实就是wiki。同时，再去release页面，可以看到有提供android的armv7和v8预先编译好的二进制代码包和源码，对应下面五个文件：

1. andoirdarm32ndk16api22prebuilt.tar：应该是针对Android armv7，基于NDK16预编译的库，查了下api22的Android API Level对应Android的版本是安卓5.1。下同；
2. andoirdarm64ndk16api22prebuilt.tar
3. convertmodelto_tm
4. Source code(zip)
5. Source code(tar.gz)

不知道其中的 `convert_model_to_tm`是干啥的，先不管。后来在wiki的Tengine快速上手指南(中文版)的模型转换小节看到，这个工具是模型转换工具， `tm`即Tengine模型格式tmfile。扯得有点远了，继续回过神来。

## 1. 交叉编译Tengine安卓库

虽然release页面有提供预编译好的库，但是还是想亲身感受一下Tengine的编译过程。

先把Tengine代码clone下来再说，顺带这个过程看看当前release页面，看到最新（这篇文章是2020年5月23日写的，之后可能会有新release，大家参考）的release版本是 v1.12.0，即 `04b6791`，接着刚刚克隆的代码接着操作：

```
git clone https://github.com/OAID/Tengine.git# 下面进入Tengine目录并切换到最新的<release-commit-id>cd Tenginegit checkout 04b6791
```

切到代码目录下，一眼就瞄到这个 `build.sh`，很是亲切，一顿无脑执行 `./build.sh`，根据屏幕输出的log信息，发现自动找到我在 `~/.bashrc`里设置的NDK环境变量，还是不要无脑执行，ctrl+c暂停掉后，打开瞧一瞧，哇，写的真好，对我来说真的通俗易懂！`build.sh`支持android的armv7、armv8、armv8.2、himix200、hisiv500、arm-linux的armv7和v8等等编译。

> 注：我这里的环境是基于自己平时开发的docker环境，已经装好了CMake、NDK等工具。本文大量参考Tengine这篇wiki的安卓指导: Tengine快速上手指南和Tengine使用教程。

因为这次上手android，我只编译android的v7和v8之后的 `exit`掉顺带修改编译线程数为30（我服务器cpu核数，大家可以根据自己的情况做修改），可能不到1分钟吧编译完成，得到两个目录 `build-android-aarch6`和 `build-android-armv7`。

瞅一眼项目代码的样子，这里我根据 `README.md`里提到的组成Tengine的5个模块，结合我蹩脚的翻译，再配合我的理解简单介绍一下：

- build-android-aarch64/：这个是刚编译的android-armv8的产物；
- build-android-armv7/：同上；
- build.sh*：编译脚本。通俗易懂看了就明白，可以无脑用起来；
- cmake/：项目CMakeLists.txt的辅助文件；
- core/：提供框架基本组件和功能；
- driver/：这个没看太明白，贴一下原文：the adapter of real H/W and provides service to device executor by HAL API. It is possible for single driver to create multiple devices；
- examples/：看了下有不少例子如 `mobilenet_ssd`、 `faster_rcnn`；
- executor/：实现了对graph和operator的执行，特别对多核的A72做了更深度的优化；
- operator/：定义算子如卷积、激活池化等操作
- pytengine/：我估计是Python API；
- serializer/：加载保存模型，这个模块也自定义扩展其功能来增加对其它模型的支持，据说已经支持NCNN模型，常规模型结构支持了Caffe/ONNX/Tensorflow/MXNet，前文在release页面也准备好了编译好的模型转换工具convertmodelto_tm；
- tests/：不同模块对应的单元测试，如conv等等；toolchains/：不同平台编译时不可或缺的文件如android、hisiv的toolchain。

## 2. Tengine的API

既然库很快编译好了，可以先看下将要跑模型所用到的调用代码，比较懒，懒得看文档了直接翻代码， `example`目录下有个 `classification.cpp`，摘录调用Tengine的API关键代码如下：

```
    // 下面的somenet应该就是类似    // 其他框架的Predictor或Executor的东西    tengine::Net somenet;    tengine::Tensor input_tensor;    tengine::Tensor output_tensor;    // 根据传入的Tengine的模型路径加载模型    somenet.load_model(NULL, "tengine", _model_file);    // 定义输入Tensor的宽度和高度，通道数默认为3    // 并传入输入数据    input_tensor.create(img_w, img_h, 3);    get_input_data(_image_file, (float* )input_tensor.data, img_h, img_w, _channel_mean, scale);    // 绑定输入    // 查了include下tengine_cpp_api.h的input_tensor    // input_tensor有多种实现，下面的实现    // 前两个参数分别为int node_index, int tensor_index    // 同时根据代码注释看出绑定输入即执行推理    somenet.input_tensor(0, 0, input_tensor);    // 获取计算结果    // 同样是查tengin_cpp_api.h的extract_tensor说明    // 下面实现的方法支持多输出获取    // 前两个参数分别是int node_index, int tensor_index    somenet.extract_tensor(0, 0, output_tensor);
```

看起来是没有显式释放 `somenet`的过程的，应该是资源自动释放，挺好！不需要用户来手动维护资源。

## 3. 编译产物

因为想跑一个分类模型，编译产物，以armv7为例也就是刚编译出的 `build-android-armv7`目录（如果是armv8则是 `build-android-aarch64`目录），重点内容在build下的install目录，其内容如下（因为内容较多，我这里只保留重要的）：

```
.|-- benchmark|   |-- bench_mobilenet|   `-- bench_sqz|-- examples|   |-- classification|   |-- faster_rcnn|   `--  mobilenet_ssd|-- include|   |-- cpu_device.h|   |-- net.h|   |-- tengine_c_api.h|   |-- tengine_c_compat.h|   |-- tengine_cpp_api.h|   `-- tengine_operations.h|-- lib|   |-- libc++_shared.so|   `-- libtengine.so`-- tests    `-- bin        |-- test_tm        `-- tm_mssd
```

因为是跑Caffe的MobileNetV1这个1000类别的分类模型，会用到编译产物里的如下几个文件（我也是比较懒，先操作出了问题后看文档，可能是大多开发者的通病）：

1. `install/example/classification`这个文件。同时也是在后面发现这个文件会动态链接 `libc++_shared.so`和 `libtengine.so`，即在后文安卓ADB shell环境执行前，需要将存放这两个动态库的文件导入到 `LD_LIBRARY_PATH`中，否则会报相关错误；
2. `install/lib/libc++_shared.so`，编译的是什么版本对应的就是v7或者v8的；3, `install/lib/libtengine.so`，同上；
3. `cat.jpg`，输入模型的图片，来自 `Tengine/tests/images/cat.jpg`；
4. `install/examples/synset_words.txt`，包含分类1000类的各个类别的名字，用于将分类结果的序号“翻译”为实际所指类别；
5. 还需要Tengine的专属的tmfile格式的模型：下面即将会讲到。

## 4. Caffe模型转换

将Caffe训练好的MobileNetV1分类模型拿出来，使用刚下载好的模型转换工具。做如下操作：

```
./convert_model_to_tm -f caffe -p ../models/MobileNet-Caffe/mobilenet_deploy.prototxt -m ../models/MobileNet-Caffe/mobilenet.caffemodel -o ./caffe_mobilenetv1.tmfileThe input file specified is using deprecated params: ../models/MobileNet-Caffe/mobilenet_deploy.prototxtPlease upgrade the input file by using caffe tools(upgrade_net_proto_text/upgrade_net_proto_binary).Create graph failederrno: 22# 发现是因为Caffe的模型格式<1.0造成的，# 需要用Caffe的工具对老模型升级Prototxt和Caffemodel。# 我下载了caffe的docker容器：docker pull bvlc/caffe:cpu# 在容器里做的模型升级，进入容器后作类似如下的操作# 升级prototxt$CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt# 升级caffemodel$CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel# 基于升级后的Caffe模型重新转换./convert_model_to_tm -f caffe -p ../models/MobileNet-Caffe/mobilenet_deploy.prototxt.new.prototxt -m ../models/MobileNet-Caffe/mobilenet.caffemodel.new.caffemodel -o ./caffe_mobilenetv1.tmfileCreate tengine model file done: ./caffe_mobilenetv1.tmfile
```

根据执行log最后一行可以看到转换模型成功！来，试试看。

> 后来发现，Tengine也提供了一些常见模型，是在百度云上的，即有原始模型也有转换好的Tengine的tmfile格式的模型文件

## 5. ADB Shell环境测试

在集成到APP前，先在ADB shell环境试一下，将刚编译好的库 + 模型依次上传到手机，我写了如下脚本：

```
# 假设当前在<Tengine>项目根目录下# 且转好的模型caffe_mobilenetv1.tmfile也在该目录下# serial_num是手机device的序列码，可以通过命令adb devices查看adb -s $serial_num shell mkdir -p /data/local/tmp/tengineadb -s $serial_num push ./tests/images/cat.jpg /data/local/tmp/tengine/adb -s $serial_num push ./build-android-armv7/examples/classification /data/local/tmp/tengine/adb -s $serial_num push ./build-android-armv7/install/lib/libtengine.so /data/local/tmp/tengine/adb -s $serial_num push ./build-android-armv7/install/lib/libc++_shared.so /data/local/tmp/tengine/adb -s $serial_num push ./caffe_mobilenetv1.tmfile /data/local/tmp/tengine/adb -s $serial_num push ./build-android-armv7/install/examples/synset_words.txt /data/local/tmp/tengine/adb -s $serial_num shell chmod +x /data/local/tmp/tengine/classificationadb -s $serial_num shell "export LD_LIBRARY_PATH=/data/local/tmp//tengine/;                          /data/local/tmp/tengine/classification \                          -m /data/local/tmp/tengine/caffe_mobilenetv1.tmfile \                          -i /data/local/tmp/tengine/cat.jpg \                          -l /data/local/tmp/tengine/synset_words.txt \                          -g 224,224 -s 0.017 -w 104.007,116.669,122.679 -r 100"# 其他参数，依据前文的classification.cpp代码里的pint_usage# 用法如下# [-m model_file]，模型路径# [-l label_file]，输出类别序号转类名映射表# [-i image_file]，输入图片路径，[-g img_h,img_w]，输入图片宽高# [-s scale] [-w mean[0],mean[1],mean[2]]，图片预处理操作相关的属性设置，我上面根据默认# [-r repeat_count]，循环的运行次数
```

最后的参数 `-r`表示跑100次，会打印出每次的执行时间。来试试看！

```
tengine library version: 1.12.0-githubModel name : /data/local/tmp/tengine/caffe_mobilenetv1.tmfiletengine model file : /data/local/tmp/tengine/caffe_mobilenetv1.tmfilelabel file : /data/local/tmp/tengine/synset_words.txtimage file : /data/local/tmp/tengine/cat.jpgimg_h, imag_w, scale, mean[3] : 224 224 0.017 104.007 116.669 122.679Cost 98.888 msRepeat [1] min 98.888 ms, max 98.888 ms, avg 98.888 ms0.2920 - "n02123159 tiger cat"0.1459 - "n02119022 red fox, Vulpes vulpes"0.1364 - "n02119789 kit fox, Vulpes macrotis"0.0806 - "n02113023 Pembroke, Pembroke Welsh corgi"0.0318 - "n02123045 tabby, tabby cat"--------------------------------------ALL TEST DONE
```

上面是执行的结果。可以看到将 `cat.jpg`成功分类为tiger cat，因为只跑了一次性能会有波动且第一次有加载耗时会比较慢，如果是做benchmark，多跑几次如10次20次先预热起来，之后的性能就稳定且很快了。

此外，据说Tengine有开源版和商业版，本次只是试验了开源版本，感觉上手非常容易，尤其是编译速度很快，模型转换也很方便，也提供了转好模型。也期待商业版本的性能！

## 6. 参考

1. OAID/Tengine: Tengine is a lite, high performance, modular inference engine for embedded device https://github.com/OAID/Tengine
2. Tengine快速上手指南(中文版) · OAID/Tengine Wiki https://github.com/OAID/Tengine/wiki/Tengine%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B%E6%8C%87%E5%8D%97(%E4%B8%AD%E6%96%87%E7%89%88)
3. Tengine源码编译 · OAID/Tengine Wiki https://github.com/OAID/Tengine/wiki/Tengine%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91
4. OPEN AI LAB发布嵌入式前端深度学习框架Tengine - 中文社区论区 - 中文社区 - Arm Community https://community.arm.com/cn/f/discussions/10381/open-ai-lab-tengine
5. Android API Level对应Android版本一览表移动开发 红壶吃猬队的博客-CSDN博客 https://blog.csdn.net/chenmiaoqin950606/article/details/78959683
