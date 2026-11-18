> 三星提出了一种新颖的方案，用于在多个硬件 IP 核（如 GPU 和 NPU）上执行神经网络时消除 CPU 干预。将 CPU 中的**数据同步角色委托给 GPU 和 NPU**，使得 GPU 和 NPU 能够直接作为生产者或消费者无缝协作，而**无需 CPU 干预**。

实验结果表明，该方案**同时减少了执行时间和功耗**，因此可**以有效地用于移动设备上的神经网络执行**。

![文章插图](/articles/2025-04-17-ip-gpu-npu-cpu-ba3b96ea/images/01.webp)

## 本文目录

- 1. 介绍
- 2. 相关工作
  - A. 基于 GPU 委托的 NN 执行
  - B. 设备端入队
- 3. 无缝协作执行模型（SEAMLESS PROCEDUER CONSUMER EXECUTION MODEL）
  - A. 协调器（Coordinator）
  - B. 地址传播器（Address Propagator）
- 4. 实验比较
- 5. 结论
- 参考文献

![文章插图](/articles/2025-04-17-ip-gpu-npu-cpu-ba3b96ea/images/02.webp)

## 1. 介绍

> 近年来，移动设备上的神经网络（NN）应用数量不断增加。面部识别、照片/相机中的物体移除、超分辨率、稳定扩散（Stable Diffusion）和神经渲染等都是此类应用的例子。

**移动设备上的 NN 应用通常一起使用各种硬件 IP 核** （如 GPU、NPU、DSP）来通过将 CPU 工作负载迁移到 IP 核来极大地加速其性能。

![文章插图](/articles/2025-04-17-ip-gpu-npu-cpu-ba3b96ea/images/03.webp)

图 1（a）展示了以前工作场景的一个例子，其中**NPU 作为生产者工作，GPU 作为消费者工作**。

::: block-1
1. CPU 要求 NPU 运行一个操作符（算子，Operator）作为生产者。
2. NPU 在 dmabuf（dmabuf 是一种用于在不同硬件 IP 核之间共享数据的机制）上生成数据，这些数据也可以被 GPU 访问。
3. CPU 等待 NPU 的输出数据准备好。
4. 一旦准备好，CPU 要求 GPU 运行另一个操作符作为消费者来读取数据。
5. GPU 读取数据作为操作符执行的输入。
6. 然后 GPU 生成另一个数据并返回结果给 CPU。
:::

一般来说，**NN 执行由重复的生产者和消费者执行（PCE，Producer Consumer Execution）组成**。上述步骤的最后输出数据可以用作下一个 PCE 的输入。**但 CPU 干预总是需要来告诉消费者数据已经准备好**。在本文中，**我们称这种模型为以 CPU 为中心的生产者消费者执行模型（CPU centric PCE）**。

已经**有很多研究致力于优化 GPU 和 NPU 上 NN 的性能和功耗**。但**很少有研究从 NN 执行中去除 CPU 干预**，这对于**减少移动设备上的执行时间和功耗是非常重要** 的。

本文的其余部分组织如下：

- 第 2 节简要介绍了在**远程 IP 核加速时去除 CPU 干预的先前方案**。
- 第 3 节提出了我们的**新颖 NN 执行模型**，无缝 PCE。
- 第 4 节展示了**评估结果**，以证明我们提出的模型在移动设备上的 NN 执行中表现良好。

## 2. 相关工作

### A. 基于 GPU 委托的 NN 执行

这种方案建议在通过 **OpenCL API** （OpenCL API 是一种用于异构计算的编程接口）**在 GPU 上加速 NN 时去除 CPU 干预**。**主机程序的角色从 CPU 迁移到 GPU**，因此位于 GPU 中的 GPU 委托（GPU Delegator）就像主机程序一样工作。

为此，**需要一个离线编译器来分析 NN，然后创建一个网络描述符表**，用于**在设备队列上排队下一个 OpenCL 内核而无需 CPU 干预。这有效地去除了 CPU 干预**，但它**只在** CPU 仅与 GPU 交互时有效。

### B. 设备端入队

设备端入队（device-side enqueue）是 **Khronos 在 OpenCL 2.0 规范中提到的另一种方案**。使用**设备端入队的 OpenCL 内核执行不需要主机干预**，因为**父内核可以直接在设备端队列上排队其子内核**。

尽管这种方法有效地消除了主机干预的需要，但它**需要更新许多父内核以调用子内核**。块语法（block syntax）在父内核中被显式描述。由于内核之间的依赖关系必须提前定义，因此在一个层中使用的内核代码不能在 NN 的其他隐藏层中重用。此外，更新数百或数千个内核以实现设备端入队并非易事。

## 3. 无缝协作执行模型（SEAMLESS PROCEDUER CONSUMER EXECUTION MODEL）

> 在本节中，我们解释了在 GPU 和 NPU 上加速神经网络时去除 CPU 干预的新颖方案。GPU 和 NPU 可以直接协作而无需 CPU 干预，因此我们称其为**无缝 PCE 模型**。

下图 2（a）解释了当生产者和消费者在同一 IP 核上协作时的无缝 PCE 架构**，图 2（b）则展示了在不同 IP 核上协作的架构**。

![文章插图](/articles/2025-04-17-ip-gpu-npu-cpu-ba3b96ea/images/04.webp)

### A. 协调器（Coordinator）

如图 2（a）和（b）所示，在每个 IP 核（GPU 或 NPU）中添加了**一个 GPU 协调器或 NPU 协调器**。**协调器的角色是根据提交的命令类型注入相应的子命令**。

- W：将生产者中存储的虚拟地址（作为目标地址）写入 addr_in_ready 寄存器。
- R：轮询 addr_in_ready，直到它具有消费者中存储的虚拟地址（作为源地址）。

当从应用程序提交的命令类型是生产者（P）时，协调器会在 P 之后立即注入子命令 W，表示为[W<P]，即“先 P 然后 W”。另一方面，如果命令类型是消费者（C），协调器会在 C 之前立即注入子命令 R，表示为[C<R]，即“先 R 然后 C”。最终，命令[C<P]被更新为[[C<R]<[W<P]]。

协调器拥有一组名为 addr_in_ready 的寄存器数组，用于存储已准备好供消费者使用的内存地址。

### B. 地址传播器（Address Propagator）

图 2（b）展示了一个示例场景，其中**生产者在 NPU 上执行，消费者在 GPU 上执行**。

**地址传播器（addr_propagtr）是一个硬件模块，由地址映射表组成**，表项包括{PA（物理地址）, CPU_va（CPU 虚拟地址）, GPU_va（GPU 虚拟地址）, NPU_va（NPU 虚拟地址）, ISP_va（ISP 虚拟地址）}。每个虚拟地址（对于 CPU、GPU、NPU 和 ISP）具有相同的物理地址 PA。

**每个虚拟地址可以相互映射**。地址传播器的角色是**将源 addr_in_ready 中的一个虚拟地址映射到目标 IP 核的虚拟地址**，然后**将映射后的虚拟地址存储到目标** addr_in_ready 中。

一旦源 addr_in_ready 被更新，其映射的虚拟地址会立即传播到目标 addr_in_ready，以便消费者可以在虚拟地址上读取数据。再次说明，[[C<R]<[W<P]]的工作流程如下：

1. P（生产者）：生产者在 NPU_va（NPU 虚拟地址）上生成数据。
2. W（写入）：NPU_va 被记录在 NPU 的 addr_in_ready 中。映射的虚拟地址 GPU_va 自动传播到目标 addr_in_ready。
3. R（读取）：轮询直到 GPU 的 addr_in_ready 具有消费者等待的虚拟地址 GPU_va。
4. C（消费者）：一旦寄存器具有 GPU_va，消费者立即读取该虚拟地址上的数据。addr_in_ready 被解锁以供下次使用。

结果，生产者和消费者**不仅可以在单个 IP 核上**，**还可以在多个 IP 核上无缝协作，而无需 CPU 干预**。

## 4. 实验比较

> 我们**评估了性能和电流消耗**，以比较我们提出的方案（无缝 PCE）与以前的以 CPU 为中心的方案，即 CPU 干预发生的情况。实验比较是**在三星 Exynos 2500 和 UltraSAN（UltraSAN 是一种用于性能评估的模拟工具）模拟工具** 上进行的。

![文章插图](/articles/2025-04-17-ip-gpu-npu-cpu-ba3b96ea/images/05.webp)

根据图 3（a）中的比较结果，无缝 PCE 方案优于以 CPU 为中心的 PCE，即**执行时间减少了约 5%**。这是因为像 **NPU 和 GPU 这样的加速器不再需要等待 CPU**。

根据图 3（b），无缝 PCE 和以 CPU 为中心的 PCE 在 DRAM 和其他部分（包括 NPU 和 GPU）的电流消耗几乎相同。但当使用无缝 PCE 时，CPU 的电流消耗减少了 47%。因为 CPU 的空闲时间增加，CPU 关机的可能性也随之增加。

## 5. 结论

本文的贡献在于**提出了一种新颖的移动设备上的神经网络执行模型**，**有效地去除了 CPU 干预**。通过**将主机程序（host program）在 CPU 上的数据同步角色委托给远程 IP 核** （如 GPU 和 NPU），生产者和消费者风格的神经网络执行可以无缝进行，而无需 CPU 干预。

实验结果表明，我们的方案**将延迟减少了 5%，CPU 电流消耗减少了 47%**。因此，我们的** 方案可以作为移动设备上推理和设备端训练的神经网络执行模型的替代方案**。

## 参考文献

1. The kernel development community, “Buffer Sharing and Synchronization (dma-buf)”, https://docs.kernel.org/driver-api/dmabuf.html
2. Lee H. Aaftab M., “The OpenCL Specification: Version 2.0,” Khronos specification, July 2015.
3. J.Yoo, K.Oh et al, “Removing Host Interventions from GPU Accelerated Neural Network,” In Proc. Of 41st IEEE Int’ Conf. on Consumer Electronics, 2023.
4. W.H. Sanders, W.D. Obal II, M.A. Qureshi, F.K. Widjanarko, The UltraSAN modeling environment, Performance Evaluation, Vol. 24, Issues 1-2, pp89-115, 1995.
