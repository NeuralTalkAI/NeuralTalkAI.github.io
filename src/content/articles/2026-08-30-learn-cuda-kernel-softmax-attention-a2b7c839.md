**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

关键词：CUDA、**PyTorch扩展**、矩阵乘、**TensorCore**、:mdnice-bold-italic[FlashAttention]

> GPU 编程最劝退人的地方是“看不到因果”：改了一个访存方式，性能变了；换了一个 tile 形状，寄存器爆了；用了 Tensor Core，结果还不如 cuBLAS。

![文章插图](/articles/2026-08-30-learn-cuda-kernel-softmax-attention-a2b7c839/images/01.jpg)

- Learn CUDA with PyTorch
- https://github.com/gau-nernst/learn-cuda

::: related-title
**相关推荐**
:::

::: related-links
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**
- **[MLC.ai 开源《Modern GPU Programming For MLSys》，从 GEMM 到 FlashAttention：Blackwell 时代的 GPU 编程方法论](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902890&idx=1&sn=3e4f0a13f3c75cc78f033969d0d53e0c&scene=21#wechat_redirect)**
- **[全智能体与人机协同双方案复盘 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902034&idx=1&sn=3cf5398e3e5b0b79a77bb32b7e31ff37&scene=21#wechat_redirect)**

:::

> `gau-nernst/learn-cuda` 的可贵之处，在于它**把这些抽象问题拆成一组能直接运行、能对照 PyTorch、能用 Nsight Compute 继续追踪的实验。**

从 `01_vector_addition/add.cu` 的线程索引，到 `02b_matmul_sm80` 的 `ldmatrix`、`mma`、`cp.async`，再到 `07_attention` 中:mdnice-bold-italic[在线 softmax 与寄存器累积的 FlashAttention 路线，这个仓库像一条“从 CUDA 入门到 LLM 推理内核”的实战栈道。]

![文章插图](/articles/2026-08-30-learn-cuda-kernel-softmax-attention-a2b7c839/images/02.jpg)

::: block-1读完代码，你会发现:mdnice-highlight[性能优化并非只靠神秘参数，更多时候是在数据搬运、计算分块、同步边界和硬件指令之间做清晰取舍。]
:::

## 本文目录

- 一、快速上手：先把 Kernel 跑起来
- 二、项目全景：一条从入门 Kernel 到 LLM Megakernel 的路线
  - 2.1 目录就是课程大纲
  - 2.2 PyTorch 扩展是所有实验的胶水层
- 三、矩阵乘主线：性能优化从访存层级开始
  - 3.1 从一个线程算一个 C 元素到 block tiling
  - 3.2 SIMT 优化的三把刀：向量化、寄存器、布局
- 四、Tensor Core 主线：从 C++ 循环进入 PTX 指令世界
  - 4.1 ldmatrix、mma 与 cp.async 的分工
  - 4.2 Shared Memory Swizzle：为了躲开 bank conflict
  - 4.3 多阶段流水：让搬运和计算重叠
- 五、Reduction 与 Softmax：从求和到 Attention 的前置训练
  - 5.1 Sum 是 GPU 集体协作的基本功
  - 5.2 Online Softmax：把三次遍历压成两次
- 六、Attention：FlashAttention 的核心在“边算边归一化”
  - 6.1 数据流比公式更重要
  - 6.2 Attention v5 的硬件意识
- 七、低比特与 LLM：项目正在走向推理内核
  - 7.1 FP6：精度格式也是系统设计
  - 7.2 Row-scaled 与 block-scaled matmul
  - 7.3 P2P 与 Megakernel：从单算子到系统路径
- 八、为什么这个仓库值得读：它保留了优化过程
  - 8.1 它讲清楚了“性能来自哪里”
  - 8.2 它把正确性放在性能之前
- 结语：从代码实验到硬件直觉

![文章插图](/articles/2026-08-30-learn-cuda-kernel-softmax-attention-a2b7c839/images/03.jpg)

**交流加群请在 NeuralTalk 公众号后台回复：加群**

## 一、快速上手：先把 Kernel 跑起来

> README 没有提供完整环境脚本，项目的使用方式以每个子目录中的 `main.py` 或 `main.cu` 为入口。最短路径如下，更多 profiling 细节可参考仓库根目录的 README[1]。

```
git clone https://github.com/gau-nernst/learn-cuda.gitcd learn-cuda# 建议在已有 CUDA + PyTorch 环境中运行# 例如先进入最简单的向量加法实验cd 01_vector_additionpython main.py# 运行 SIMT 矩阵乘实验cd ../02a_matmul_simtpython main.py# 调试非法访存compute-sanitizer python main.py# 使用 Nsight Compute 分析 Kernel，README 建议编译时带 -lineinfoncu --set full --import-source on -o profile python main.py
```

这里的核心依赖是 PyTorch 的 `torch.utils.cpp_extension.load`：Python 负责动态编译 CUDA/C++ 扩展、构造输入张量、调用自定义 kernel，并用 `torch.testing.assert_close` 与 PyTorch 结果对齐。

::: block-1也就是说，这个仓库的学习姿势是**把 CUDA kernel 放进真实深度学习工作流里验证**，并非“单独写一堆 `.cu` 文件”。
:::

## 二、项目全景：一条从入门 Kernel 到 LLM Megakernel 的路线

### 2.1 目录就是课程大纲

> `learn-cuda` 顶层目录非常直白，每个编号目录对应一个主题：

```
01_vector_addition/          最小 PyTorch CUDA 扩展示例02a_matmul_simt/             CUDA Core 上的 FP32 矩阵乘优化02b_matmul_sm80/             SM80 Tensor Core：PTX、ldmatrix、mma、cp.async02c_matmul_sm120/            新架构矩阵乘实验02d_matmul_cdna3/            AMD CDNA3 方向的矩阵乘实验02e_matmul_sm100/            SM100、tcgen05、tmem 等前沿实验03_sum/                      reduction：shared memory、warp shuffle、cooperative groups04_softmax/                  naive softmax、online softmax、多 block 合并05_fp6/                      FP32/FP16/BF16 与 FP6 的转换和打包06_box_blur/                 2D block/thread 图像模糊07_attention/                FlashAttention 风格实现08_row_scaled_mm/            行缩放量化矩阵乘，INT8/FP8 输入09a_block_scaled_mm_sm120/   MXFP8 block-scaled matmul 实验10_p2p/                      多 GPU P2P、IPC handle、对称地址访问11_gemv/                     GEMV 与 persistent kernel12_megakernel/               面向 LLM decode 的融合内核探索
```

根目录的 README 把**项目定位为 “Learn CUDA with PyTorch”**。这句话很关键：它是直接站在 PyTorch 张量之上。:mdnice-bold-italic[这样做的好处是，输入、输出、正确性验证、benchmark、Profiler 都能借助成熟生态完成]，**学习者的注意力可以集中在 kernel 本身。**

### 2.2 PyTorch 扩展是所有实验的胶水层

> 最小例子来自 `01_vector_addition`。Python 侧用 `torch.utils.cpp_extension.load` 动态编译 `add.cu`，CUDA 侧暴露一个 `add` 函数给 Python 调用。

```
// 来源：01_vector_addition/add.cu__global__ void add_kernel(const float *input1, const float *input2, float *output, int size) {  const int idx = blockIdx.x * blockDim.x + threadIdx.x;  if (idx < size)    output[idx] = input1[idx] + input2[idx];}torch::Tensor add(torch::Tensor input1, torch::Tensor input2) {  CHECK_INPUT(input1);  CHECK_INPUT(input2);  int size = input1.numel();  torch::Tensor output = torch::empty(size, input1.options());  int n_threads = 256;  int n_blocks = (size + n_threads - 1) / n_threads;  add_kernel<<<n_blocks, n_threads>>>(      input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), size);  return output;}PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add, "Add two vectors"); }
```

这段代码是整个仓库的方法论缩影：先检查输入必须是 CUDA tensor 且连续，然后创建输出 tensor，再用 raw pointer 进入 CUDA 世界。`blockIdx.x * blockDim.x + threadIdx.x` 是每个 CUDA 初学者都会见到的索引公式，**但在这个仓库里，它很快会被推广成矩阵乘中的二维 tile、warp tile、MMA tile，以及 Attention 中的 Q/K/V 分块。**

## 三、矩阵乘主线：性能优化从访存层级开始

### 3.1 从一个线程算一个 C 元素到 block tiling

> `02a_matmul_simt/matmul.cu`**用一组版本展示 FP32 GEMM 如何逐步接近 cuBLAS**。README 给出的 4096 方阵结果很有代表性：naive v1 约为 cuBLAS 的 8.49%，v4 通过 2D thread tiling 到 53.48%，v6b 通过去边界检查、向量化访存和 shared memory 中转置 A 到 92.44%。

::: block-1关键变化可以用一句话概括：:mdnice-highlight[矩阵乘优化的第一目标，是让全局内存中的元素被搬进 shared memory 后复用尽可能多次；第二目标，是让每个线程在寄存器里攒更多局部结果，减少反复访存。]
:::

```
// 来源：02a_matmul_simt/matmul.cutemplate <int BLOCK_SIZE>__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {  const int tid_x = threadIdx.x;  const int tid_y = threadIdx.y;  const int offset_m = blockIdx.y * BLOCK_SIZE;  const int offset_n = blockIdx.x * BLOCK_SIZE;  __shared__ float A_shmem[BLOCK_SIZE][BLOCK_SIZE];  __shared__ float B_shmem[BLOCK_SIZE][BLOCK_SIZE];  float acc = 0.0f;  for (int offset_k = 0; offset_k < K; offset_k += BLOCK_SIZE) {    A_shmem[tid_y][tid_x] =        tid_y < (M - offset_m) && tid_x < (K - offset_k) ? A[(offset_m + tid_y) * K + offset_k + tid_x] : 0.0f;    B_shmem[tid_y][tid_x] =        tid_y < (K - offset_k) && tid_x < (N - offset_n) ? B[(offset_k + tid_y) * N + offset_n + tid_x] : 0.0f;    __syncthreads();    for (int k = 0; k < BLOCK_SIZE; k++)      acc += A_shmem[tid_y][k] * B_shmem[k][tid_x];    __syncthreads();  }  if (tid_y < (M - offset_m) && tid_x < (N - offset_n))    C[(offset_m + tid_y) * N + offset_n + tid_x] = acc;}
```

这段 v2 已经具备经典 GEMM 的骨架：沿 K 维循环，每次把 A/B 的一个小块搬进 shared memory，线程块内部同步，然后在片上 SRAM 中计算。**它的收益来自复用，代价是同步和 shared memory 容量。**

::: block-1
**后续 v3/v4/v5/v6 的优化，本质上是在这副骨架上继续切分**：block 级 tile 之下有 thread tile，thread tile 中用寄存器缓存 `A_reg`、`B_reg`，再进一步按 warp 组织输出区域。
:::

### 3.2 SIMT 优化的三把刀：向量化、寄存器、布局

> v6 系列做了两个很实用的动作。

- 第一，用 `float4` 做向量化 global load/store，减少指令数量并提高内存事务利用率。
- 第二，把 A 在 shared memory 中转置，让后续读取 A 的线程访问模式更友好。

代码里还通过 `assert(is_power_of_two(M/N/K))` 移除边界检查，让循环展开和地址计算更干净。

这里能看到作者并没有只追逐“理论最优”。:mdnice-bold-italic[例如 v6 注释里写到 vectorized write 更慢、修复 uncoalesced write 没有让 kernel 更快，这种记录非常有价值。GPU 优化经常会出现局部指标改善、整体时间无收益的情况，真实实验比单一规则更可靠。]

## 四、Tensor Core 主线：从 C++ 循环进入 PTX 指令世界

### 4.1 ldmatrix、mma 与 cp.async 的分工

> **`02b_matmul_sm80` 是项目的核心跃迁：它从普通 CUDA Core 上的 FMA，进入 SM80 Tensor Core。** 这里的矩阵乘输入是 BF16，A 为 row-major，B 为 column-major，最终用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 做累积。

在 `common.h` 中，作者**把底层 PTX 包装成几个小函数**：

- `cvta_shared` 把 generic pointer 转成 PTX shared state space 地址；
- `ldmatrix_x2/x4` 从 shared memory 按 Tensor Core 需要的布局装入寄存器；
- `mma_m16n8k16` 发出真正的矩阵乘累加指令；
- `cp_async` 负责 global 到 shared 的异步拷贝。

```
// 来源：02b_matmul_sm80/common.h__device__ inline uint32_t cvta_shared(const void *ptr) {  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));}__device__ inline void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"              : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])              : "r"(addr));}__device__ inline void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"              : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));}__device__ inline void cp_async(uint32_t dst, const void *src) {  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));}
```

**这几个函数很短，却是理解 Tensor Core kernel 的钥匙**。普通 CUDA 代码里，线程通常显式读标量、乘加、写回；Tensor Core kernel 中，warp 作为整体参与计算，每个 lane 持有矩阵碎片的一部分。

- `ldmatrix` 解决“怎样把 shared memory 中的 tile 分发到 32 个 lane 的寄存器”；
- `mma` 解决“怎样让这些寄存器片段参与硬件矩阵乘”；
- `cp.async` 解决“怎样在计算当前 tile 时提前搬运下一个 tile”。

### 4.2 Shared Memory Swizzle：为了躲开 bank conflict

> README 对 bank conflict 的解释相当细。:mdnice-bold-italic[shared memory 有 32 个 bank，`ldmatrix` 会按 8x8 tile 读取 16-bit 元素；如果行跨度让多个 row 落到相同 bank，就会形成严重冲突。作者提供了 padding 与 swizzled layout 两种思路，最后用 XOR 改列索引，使 8 个 16-byte word 分散到不同 bank。]

`common.h` 中的 `swizzle_better` 是这种思想的简化表达：

```
// 来源：02b_matmul_sm80/common.htemplate <int STRIDE>__device__ static uint32_t swizzle_better(uint32_t row, uint32_t col) {  if constexpr (STRIDE >= 128)    col ^= (row % 8) / std::max(128 / STRIDE, 1);  return row * STRIDE + col * 16;}
```

它看起来像一个小小的地址变换，背后其实是在**配合 `ldmatrix` 的取数粒度和 shared memory bank 编号**。可把它想象成电影院排座：原始布局让几排观众都挤在同一个入口，swizzle 则把不同排的人引导到不同入口，硬件执行时就少了排队。

### 4.3 多阶段流水：让搬运和计算重叠

> **`02b_matmul_sm80/matmul_v6.cu` 进一步把 `cp.async` 做成两阶段 pipeline**。循环不再是“加载 A/B、等待、计算”这样的串行节奏，:mdnice-bold-italic[本质是保持 `NUM_STAGES - 1` 个预取阶段在路上。当前 tile 用 Tensor Core 计算时，下一个 tile 已经从 global memory 往 shared memory 搬。]

README 中的性能结果显示，v6 在 5090 上达到 177.66 TFLOPS，约 84.80% peak；在 A100 上达到 220.77 TFLOPS，约 70.76% peak。

::: block-1
**这说明手写 kernel 已经进入和 cuBLAS、Triton 可比较的区域**。对学习者而言，意义很大：你能看到:mdnice-highlight[高性能 GEMM 是由 tile 切分、寄存器碎片、shared memory 布局、异步流水一步步叠出来。]
:::

## 五、Reduction 与 Softmax：从求和到 Attention 的前置训练

### 5.1 Sum 是 GPU 集体协作的基本功

> **`03_sum/sum.cu` 从一个线程负责一整行的 Kahan sum 开始，逐渐引入 shared memory reduction、thread coarsening、warp-level reduction、cooperative groups 和 `float4` 向量化读取**。这个目录在整个仓库中像一个“中场训练营”：矩阵乘关注数据复用，softmax 和 attention 则强依赖 reduction。

尤其值得注意的是 v4 对 warp 内规约的三种处理：`volatile` shared memory、`__syncwarp()`、`__shfl_down_sync()`。作者在注释中指出，仅靠 shared memory 写读，编译器可能缓存读取，导致直觉上“同时读写无竞争”的代码仍然出错。这个例子提醒我们，:mdnice-bold-italic[CUDA 正确性不只关乎线程是否同时写同一地址，也关乎编译器优化、同步语义和内存可见性。]

### 5.2 Online Softmax：把三次遍历压成两次

> `04_softmax/softmax.cu` 是连接 reduction 与 FlashAttention 的桥。naive softmax 做三件事：找每行最大值、求 `sum(exp(x - max))`、写归一化结果。**online softmax 则把最大值和分母合并更新，核心公式来自论文中的 online normalizer。**

```
// 来源：04_softmax/softmax.cu__device__ float2 online_normalizer(float2 a, float2 b) {  float new_max = max(a.x, b.x);  float normalizer = a.y * exp(a.x - new_max) + b.y * exp(b.x - new_max);  return {new_max, normalizer};}__global__ void softmax_online_kernel(const float *input, float *output, int M, int N) {  const int tid = threadIdx.x;  const int BLOCK_SIZE = blockDim.x;  const int row = blockIdx.x;  extern __shared__ float2 shmem[];  input += row * N;  output += row * N;  float2 normalizer = {-FLT_MAX, 1.0f};  auto f = [](float2 a, float b "") {    float new_max = fmaxf(a.x, b);    float new_normalizer = a.y * exp(a.x - new_max) + exp(b - new_max);    return float2{new_max, new_normalizer};  };  normalizer = thread_reduce(f, normalizer, input, N, BLOCK_SIZE, tid);  normalizer = block_reduce(online_normalizer, normalizer, BLOCK_SIZE, tid, shmem);  normalizer = warp_reduce(online_normalizer, normalizer, tid);  normalizer = block_broadcast(normalizer, tid, shmem);  write_softmax_result(input, output, N, normalizer.x, normalizer.y, BLOCK_SIZE, tid);}
```

这段代码非常适合科普 softmax 的数值稳定性。`max` 变化时，旧分母不能直接相加，必须乘上 `exp(old_max - new_max)` 做重标定。:mdnice-bold-italic[这个技巧在 Attention 中会再次出现：当 K/V 分块流式进入时，每个 Q 行的当前最大值和分母都要被动态维护，旧的输出累积也要随之 rescale。]

## 六、Attention：FlashAttention 的核心在“边算边归一化”

### 6.1 数据流比公式更重要

> `07_attention` 包含 `attention_v1.cu` 到 `attention_v5.cu` 多个版本。

v5 已经具备典型 FlashAttention 风格：Q 只加载一次，K/V 按块流入；先算 `S = Q @ K.T`，对 S 做 scale 与 online softmax，再把概率矩阵片段 P 与 V 相乘，累积到 O 的寄存器中。中间的 S/P 并没有完整写回 HBM，这正是 FlashAttention 节省显存流量的关键。

```
// 来源：07_attention/attention_v5.cu// 省略加载 Q/K/V 到 shared memory 与 ldmatrix 的细节// 1. S = Q @ K.Tfor (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)  for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)      mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],                   K_rmem[mma_id_kv][mma_id_d],                   S_rmem[mma_id_q][mma_id_kv]);// 2. 更新 rowmax，重缩放旧 Ofloat rescale[2];rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);// 3. exp(S - rowmax)，累积 rowsumexp，并打包成 Pregs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);// 4. O += P @ Vfor (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)  for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)      mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],                   V_rmem[mma_id_kv][mma_id_d],                   O_rmem[mma_id_q][mma_id_d]);
```

这段逻辑可以用流水线来理解：Q 像固定的查询窗口，K/V 像一页页翻过来的字典。每翻一页，就更新当前页的相关性分数、更新 softmax 的最大值和分母、把这一页贡献到输出 O。完整注意力矩阵没有落地，HBM 压力自然下降。

### 6.2 Attention v5 的硬件意识

> `attention_v5_kernel` 明确限制 `dim == 128`，设置 `BLOCK_Q = 64`、`BLOCK_KV = 64`、`NUM_WARPS = 4`。它还把 Q 的 shared memory 与 K/V 的 shared memory 区域复用：Q 只需加载一次，K 做 double buffer，V 使用 single buffer。:mdnice-bold-italic[代码中多处 `cp.async.commit_group`、`cp.async.wait_group 1` 和 `__syncthreads()`，体现出作者非常小心地管理“哪块 shared memory 还在被使用，哪块可以被下一轮覆盖”。]

这类代码的难点在于三层状态同时存在：

- 数学状态是 rowmax、rowsumexp、O 累积；
- 内存状态是 global/shared/register 的数据流；
- 执行状态是 warp、lane、MMA tile 的分工。

`learn-cuda` 通过逐版本文件把这些复杂度摊开，读者可以先在 `04_softmax` 看 online normalizer，再回到 `07_attention` 理解为何 O 要 rescale。

## 七、低比特与 LLM：项目正在走向推理内核

### 7.1 FP6：精度格式也是系统设计

- **`05_fp6/fp6.cu` 实现了 FP32/FP16/BF16 到 FP6 的转换，以及 unpacked、packed 两种表示**。packed 形式把 4 个 6-bit 数打成 3 个字节，CPU 与 CUDA 都提供转换路径。:mdnice-bold-italic[代码中特别处理 round-to-nearest-even、subnormal、overflow、NaN/Inf 等情况]，这说明低比特并非简单截断几位。

作者还提到，在 CUDA 上 dtype conversion kernel 往往 memory-bound，因此位操作版本和数值转换版本差异不一定明显；位操作分支多，还可能造成 warp divergence。这是很典型的底层权衡：更“聪明”的算术路径，如果让 warp 内线程走不同分支，吞吐可能反而受损。

### 7.2 Row-scaled 与 block-scaled matmul

- `08_row_scaled_mm` 针对 INT8/FP8 输入，在 MMA 累积后加载每行/每列 scale，把结果写为 BF16。其中的 epilogue 已经不再只是 `C = A @ B`，还要把量化 scale 融入输出阶段。
- `09a_block_scaled_mm_sm120` 则探索 MXFP8 block scaling，并显式引用 CuBLAS SF layout。值得提醒的是，当前 `mxfp8_mm_v3.cu` 中有一行输出 row 计算残缺，说明这个目录仍处于 WIP 状态，阅读时应把它视作实验草稿，而非稳定实现。

### 7.3 P2P 与 Megakernel：从单算子到系统路径

- **`10_p2p/p2p.cu` 涉及多 GPU 通信基础**：`cudaDeviceEnablePeerAccess`、`cudaIpcGetMemHandle`、`cudaIpcOpenMemHandle`，以及通过 heap base 做 symmetric pointer translation。它展示了:mdnice-bold-italic[从单卡 kernel 走向多卡系统时必须面对的地址空间问题。]
- **`12_megakernel` 则直接指向 LLM decode**。README 写明目标是 Qwen3 dense 小模型，先做 MLP 与 Attention，再从 Triton 过渡到 CUDA C++。其中 Qwen3-0.6B 的交互式 chat 入口为 `python chat.py`，:mdnice-bold-italic[README 中给出的 decode 粗测结果显示：HF eager 约 127/132 tok/s，vLLM 约 616/599 tok/s，项目实现约 838/807 tok/s]。作者也声明 measurement by `chat.py` “not rigorous”，这点很诚实：它:mdnice-bold-italic[证明方向有效，但不把非严格 benchmark 包装成最终结论。]

## 八、为什么这个仓库值得读：它保留了优化过程

### 8.1 它讲清楚了“性能来自哪里”

> 很多 CUDA 教程会直接给最终 kernel，读者看到的是一团高度优化后的代码**。`learn-cuda` 更像实验记录：同一个主题下保留 v1、v2、v3……每个版本只推进几个概念。**

- 矩阵乘从 naive 到 block tiling，再到 thread tiling、warp tiling、vectorized load、shared memory 转置；
- softmax 从三 pass 到 online；
- attention 从基础版本推进到 Tensor Core 与 online softmax 融合。

::: block-1
**这种写法让性能变化有了因果链**。你能把 README 中的表格和源码中的改动对应起来：:mdnice-highlight[为什么 shared memory 提高算术强度，为什么 bank conflict 会吃掉 Tensor Core 的收益，为什么 `cp.async` 需要 wait group 与 commit group 配合，为什么 register cache 提升复用却会降低 occupancy。]
:::

### 8.2 它把正确性放在性能之前

> 每个 Python 入口通常都会生成随机输入，用 PyTorch 原生算子做 reference，再用 `torch.testing.assert_close` 验证。这个习惯很重要。CUDA kernel 的错误可能表现为轻微数值偏差、偶发非法访存、未初始化输出、同步缺失，只有持续对照 reference，性能数据才有意义。

项目也提供了工具建议：

- 用 `compute-sanitizer` 查 illegal memory access
- 用 `ncu --set full --import-source on` 做 Nsight Compute 分析
- PyTorch 程序可用 `torch.profiler.profile()` 导出 trace 到 Perfetto。

换言之，它不只教你写 kernel，还**教你建立“写、测、验、剖析”的闭环。**

## 结语：从代码实验到硬件直觉

> `learn-cuda` 把 GPU 性能优化拆成一条可以亲手走完的路线：先理解线程、block、warp 与 PyTorch 扩展；再掌握 shared memory、寄存器、coalesced access 和 reduction；随后进入 Tensor Core 的 PTX 指令、shared memory swizzle、异步拷贝流水；最后把这些能力汇入 softmax、FlashAttention、量化矩阵乘、P2P 和 LLM megakernel。

如果把 GPU 看成一座工厂，global memory 是远处仓库，shared memory 是车间缓存，寄存器是工人手边的零件盒，Tensor Core 是专用加工机床。**高性能 kernel 的任务，就是让仓库少跑空车，让车间入口不堵，让工人手里的零件持续供给机床**。这个仓库最实在的地方，正是它用可运行代码把这套工厂调度过程摊在读者眼前。

::: block-1对想深入 AI 系统、编译器后端或推理引擎的工程师来说，这是一份带着实验痕迹的底层性能地图。沿着它读下去，你会逐渐拥有判断 kernel 性能瓶颈的直觉：:mdnice-highlight[看一段代码时，先问数据从哪里来、在哪里复用、由哪个 warp 计算、何时同步、输出前是否还能融合更多工作。这样的直觉，正是从“会写 CUDA”走向“能设计高性能算子”的分水岭。]
:::

参考资料

[1] README: *https://github.com/gau-nernst/learn-cuda/blob/main/README.md*

::: related-title
**相关推荐**
:::

::: related-links
- **[MLSys 26 NVIDIA Blackwell 内核竞赛纯智能体方案 Kernel Design Agents：MoE 赛道第一、DSA 赛道第二，均超 FlashInfer 性能基线！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901565&idx=1&sn=2b32f4d5649568dcd3fb34a6575f973b&scene=21#wechat_redirect)**
- **[陈天奇团队提出 CAKE：编译器与智能体协同进化，可进化 IR 让 KDA 提速 2.05 倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447904925&idx=1&sn=ab085dc1e9e6c73917c8853a398775d7&scene=21#wechat_redirect)**
- **[暴涨 45 个 Star！NVIDIA CUDA C++ 核心库 Lead 开源 AutoCUDA：永不停机的 GPU 内核优化智能体框架](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900280&idx=1&sn=5253c3d5faaeef74802a187e0878de2b&scene=21#wechat_redirect)**

:::
