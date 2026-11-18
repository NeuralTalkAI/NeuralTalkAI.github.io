> 本文作者是Anastasia Stulova（简称A.S.），写于December 14, 2020，A.S.是OpenCL方面的专家。
>
> OpenCL分为两部分，host端代码和kernel代码，本文AS将会给大家介绍OpenCL C++在kernel代码方面的支持工作。

**The Mali Software Development Kit (SDK)****facilitates the development of OpenGL ES, Vulkan, or OpenCL applications for Mali-based platforms**.

- While OpenGL and Vulkan are predominantly used for graphics,
- OpenCL provides ability to execute applications from various domains on Mali GPUs, such as image processing, machine learning (ML), computer vision (CV) and many others.

The main concept behind OpenCL is to accelerate data parallel computation on a device with high compute power. If applications are suitable for the OpenCL acceleration, the increase in speed-up can be orders of magnitude compared to conventional execution.

Any OpenCL program has

1. **a host side** where regular computations are executed typically on a CPU
2. and **device side** where the compute intensive kernels are sent from the host for acceleration.

On the host side developers could use C, C++ and even Python from very early versions of the standard. **However, the kernel core accelerated on the device had to be written in** **OpenCL C derived from C99.**

Last year a new **kernel language C++ for OpenCL** was developed in upstream **LLVM**. This allows the use of **most C++17 features in the OpenCL kernel code** – more details about this are in the previous blog. **The kernels in C++ for OpenCL language could be** **compiled offline using open-source tools and the binaries****in** **SPIR-V intermediate format** **could be imported by OpenCL applications running** **with existing OpenCL drivers**. Now, we are pleased to announce that with the latest release of Mali SDK developers can benefit from the **online and offline compilation of kernel written in C++ for OpenCL** using the **clextcxxforopencl** extension. This has been recently published on the Khronos website.

**Arm is the first vendor** providing **support of this new extension** in its SDK. While this is great news for developers, this is still in the experimental phase. Some features, such as program scope objects with non-trivial conductors or destructors, are not yet supported at the time this blog was published.

## About C++ for OpenCL

**C++ for OpenCL** provides the **ability to use most of the modern C++ features from the C++17 standard** in **OpenCL kernels**. This improves the programmer productivity of applications with ever growing complexity running on GPUs. Another great benefit is that**C++ for OpenCL is backward compatible with OpenCL C**. This means existing applications can be migrated smoothly to C++ features and developers can continue to use familiar OpenCL programming concepts and tools. Overall, the kernel code written in C++ for OpenCL looks just like code written in OpenCL C with some extra C++ features available for convenience.

C++ for OpenCL kernels can leverage many C++ specific compiler optimizations leading to competitive performance on OpenCL devices. The community-defined language documentation is hosted on the OpenCL-Docs repository on GitHub along with other Khronos specifications for OpenCL. The latest published revision can be found here.

You can find out more about the C++ for OpenCL programming language from the slide deck PDF and video of the talk presented at IWOCL this year.

## Implementing kernels with C++ features

The following code is a snippet illustrating how to implement kernels with complex number arithmetic using C++ features. The full example can be found in Code Explorer.

```
// Define a class - Complex, that can perform complex number arithmetic// with various precision when different types for ‘T’ are used - double, float, half...template<typename T>class complex_t {  T m_re; // Real component.  T m_im; // Imaginary component.  public:    complex_t(T re, T im): m_re{re}, m_im{im}{};    complex_t operator*(complex_t &other) {      return {m_re * other.m_re - m_im * other.m_im,                    m_re * other.m_im + m_im * other.m_re};  }  int get_re() { return m_re;}  int get_im() { return m_im;}};// A kernel function to compute multiplication over complex numbers read from// the input buffer and to store the result into the output buffer.kernel void compute_helper(global float *in, global float *out) {    auto idx = get_global_id(0);    // Every work-item uses 4 consecutive items from the input buffer -    // two for each complex number.    auto offset = idx * 4;    complex_t num1{in[offset], in[offset + 1]};    complex_t num2{in[offset + 2], in[offset + 3]};    // Perform complex number multiplication.    complex_t res = num1 * num2;    // Every work-item writes 2 consecutive items to the output buffer.    out[idx * 2] = res.get_re();    out[idx * 2 + 1] = res.get_im();}
```

## About offline compilation

The offline compilation of kernels written in C++ for OpenCL has been available through open-source tooling from September 2019. This was when Clang 9.0 was released with the experimental support of C++ for OpenCL in **upstream llvm-project**. Kernels can **be compiled to SPIR-V format offline** following the flow explained in this blog on the Khronos website. **The SPIR-V binary can then be loaded by OpenCL applications** running on OpenCL 2.0 or later version drivers using regular **clCreateProgramWithIL** API call.

## Developers, try for yourself

We recommend developers experimenting with the C++ for OpenCL kernel language in their applications. If you have any feedback regarding the new kernel language, its support in tooling or Mali SDK in general, then we would like to hear from you.

## reference

- https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/mali-sdk-supporting-compilation-of-kernels
- IWOCL - Enhanced Support for OpenCL Open Source Tools and Resources https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/new-support-for-opencl-open-source-projects
- clextcxxforopencl https://www.khronos.org/registry/OpenCL/extensions/ext/clextcxxforopencl.html
- Khronosdotorg/CXXforOpenCL.pdf at main · KhronosGroup/Khronosdotorg https://github.com/KhronosGroup/Khronosdotorg/blob/main/api/opencl/assets/CXXforOpenCL.pdf
- iwocl-syclcon-2020-stulova-13-slides.pdf https://www.iwocl.org/wp-content/uploads/iwocl-syclcon-2020-stulova-13-slides.pdf
- https://youtu.be/SR5E3hKVh14

> 点击【阅读原文】，看往期文章
