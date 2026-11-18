关键词：Triton Kernels、RISC-V、MLIR、Compiling、Optimizing、AI Accelerators、Benchmark

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/01.webp)

- Compiling and Optimizing Triton Kernels onto RISC-V Targets based on MLIR
- https://static.sched.com/hosted_files/riscvsummit2024/0b/202410-RISC-V-NA-Summit-Compiling%20and%20Optimizing%20Triton%20Kernels%20Onto%20RISC-V%20Targets%20Based%20on%20MLIR.pdf
- 本文 1455 字，阅读需 6 分钟，播客 8 分钟如下

> 在软件从 1.0 时代迈向 2.0 时代的浪潮中，AI 领域的发展日新月异。当前，NVIDIA CUDA 凭借 90% 的市场份额在 AI 计算领域占据主导地位，同时存在着 PyTorch、TensorFlow、JAX、ONNX 等众多机器学习框架以及大量的 AI 加速器。

然而，这些 AI 加速器的底层内核多采用特定目标运行时 API 编写，每个 AI 加速器供应商都在构建自己的机器学习编译器和目标特定内核，缺乏统一的内核编程语言和机器学习编译器，这给 AI 技术的发展带来了诸多不便。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/02.webp)

::: image-caption
Expectation vs. Reality
:::

在此背景下，兆松科技开展了将 Triton 内核基于 MLIR 编译和优化到 RISC-V 目标平台的研究。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/03.webp)

## 本文目录

- Triton GPU 编程语言
- Triton 在 RISC-V 的性能表现
  - 在 RoPE 内核
  - Matmul 内核
  - Warp 内核
  - LayerNorm 内核
- 基于 IREE、MLIR 的 MAGIC 架构
  - MLIR
  - IREE
  - MAGIC 架构
- 结论
- 相关推荐

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/04.webp)

## Triton GPU 编程语言

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/05.webp)

::: image-caption
Triton VS CUDA VS C
:::

Triton 作为一种来自 OpenAI 的开源神经网络 GPU 编程语言，具有类 Python 的领域特定语言（DSL）特性，在 NVIDIA GPU 上性能与 CUDA 相当，支持即时编译（JIT）或预先编译（AOT），且是块级编程语言，相较于线程级的 OpenCL C/CUDA C++ 和全局级的 C/C++ 具有独特优势。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/06.webp)

::: image-caption
Why not CUDA C++/OpenCL C?
:::

它是一种目标无关的编程语言，PyTorch 2.0 中已有大量内核用 Triton 编写，初期实验也显示其在 RISC-V 平台有良好性能潜力，这使得它成为解决 RISC-V 平台缺乏统一内核编程语言问题的理想选择。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/07.webp)

::: image-caption
Matrix-Vector Multiplication in Triton
:::

## Triton 在 RISC-V 的性能表现

> 为验证 Triton 在 RISC-V 平台的表现，研究人员在 SpacemiT K1（RVA22，256bit RVV 1.0）目标平台上，对 LLM 领域的 RoPE、Matmul、Layernorm、Softmax 以及 CV 领域的 Resize、Warp、Correlation 等内核进行了基准测试。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/08.webp)

::: image-caption
Benchmark Triton kernels on RISC-V
:::

### 在 RoPE 内核

多线程 Triton 内核性能优于 RISC-V GCC 15%~18%；

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/09.webp)

::: image-caption
Triton kernel on RISC-V - RoPE
:::

### Matmul 内核

Triton 内核比 RISC-V GCC 性能好 14%~21%，比 RISC-V ZCC 好 33%~44%，且性能随线程数增加几乎线性扩展；

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/10.webp)

::: image-caption
Triton kernel on RISC-V - Matmul
:::

### Warp 内核

Triton 内核在不同线程数下均优于 ZCC 9%~20%，优于 GCC 56%~109%。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/11.webp)

::: image-caption
Triton kernel on RISC-V - Warp
:::

### LayerNorm 内核

不过，**在 Layernorm 内核的多线程测试中，Triton kernel 性能不及 ZCC 和 GCC**，这与 Triton kernel 生成额外的掩码加载指令有关。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/12.webp)

::: image-caption
Triton kernel on RISC-V - Layernorm
:::

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/13.webp)

::: image-caption
Triton kernel on RISC-V - Softmax
:::

---

:mdnice-bold-italic[总体而言，Triton 内核性能与 C 内核相当甚至更优，能被 ZCC 自动向量化以适配 RVV 目标]，还可通过 OpenMP 实现并行化，其编译路径为 MLIR + LLVM，采用 TritonCPU 编译器与 ZCC 编译器结合的方式。

## 基于 IREE、MLIR 的 MAGIC 架构

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/14.webp)

::: image-caption
Introducing “big compiler”
:::

### MLIR

> MLIR 在这一过程中发挥了关键作用。

传统编译器如 GCC、LLVM 主要将高级语言或计算图编译到同构处理器，而以 IREE 为代表的 “big compiler” 能将高级语言或计算图编译到异构平台，在编译器内部抽象运行时 API 和主机 - 设备通信。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/15.webp)

::: image-caption
Code generation in MLIR
:::

### IREE

IREE 作为端到端的机器学习模型编译栈，支持多种模型导入，利用 MLIR 核心方言，通过 Flow 进行目标无关的张量和计算图划分，Stream 实现目标无关的放置和调度算法，HAL 作为硬件抽象层，最终实现主机代码生成和设备代码生成，支持包括 RISC-V 在内的多种目标硬件。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/16.webp)

::: image-caption
Introduction to IREE
:::

### MAGIC 架构

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/17.webp)

> 兆松科技提出的 MAGIC 架构（MLIR AI Graph Intermediate Compiler）更是将 Triton 等编程语言的 IR 转换为 Tensor、Linalg/LinalgExt 等。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/18.webp)

::: image-caption
“Big compiler” – MAGIC architecture
:::

基于 IREE 的模型导入器，借助包含 Flow、Stream、HAL 的核心 IR，实现内核操作与机器学习计算图的融合，为 RISC-V 及其他 CPU、GPU 设备等提供支持。

![文章插图](/articles/2025-07-29-triton-risc-v-mlir-abd89a01/images/19.webp)

::: image-caption
“Big compiler” – MAGIC architecture
:::

## 结论

> 综上所述，将 Triton 内核基于 MLIR 编译和优化到 RISC-V 目标平台的研究取得了显著成果。

Triton 内核在 RISC-V 平台上展现出良好的性能潜力，MLIR 为编译和优化提供了强大的技术支撑。

未来，研究团队将进一步将 Triton 下放到核心 MLIR，并支持 Triton 内核与 MLIR 核心方言中的机器学习计算图进行融合，推动 RISC-V 平台在 AI 领域的更广泛应用，朝着实现统一的 AI 加速器内核编程语言和机器学习编译器的目标迈进。

## 相关推荐

- [在 RISC-V GPU 上运行 CUDA：基于 NVVM-SPIR-V 构建转换流水线与基准测试，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447890118&idx=1&sn=6be49934e8bf5806b444ab036716d44a&scene=21#wechat_redirect)
- [GEMM 95% FPU 利用率！基于 MLIR 面向 RISC-V 指令集扩展的加速Micro-kernels的多层编译器后端](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889407&idx=1&sn=f5640780406954ae498d2c98d162d454&scene=21#wechat_redirect)
- [为 Arm、x86、RISC-V 架构进行指令级模拟下的性能自动调优！跳过真机硬件限制，用模拟器加速自动调优！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889026&idx=1&sn=c4d677efd0d252f899e7ae78a916119e&scene=21#wechat_redirect)
- [评估 Tenstorrent RISC-V 矩阵乘法加速能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888358&idx=1&sn=0279f801d6b7ea4b25e36503f552fbd0&scene=21#wechat_redirect)
