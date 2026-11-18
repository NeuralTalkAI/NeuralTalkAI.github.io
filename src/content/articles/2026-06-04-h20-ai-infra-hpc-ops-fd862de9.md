关键词：:mdnice-bold-italic[高性能算子库]、**Hopper 架构**、**CuTe/CUTLASS**、FP8 推理、:mdnice-bold-italic[FusedMoE]

> 在大模型推理这条赛道上，"快一点"从来不是一句口号，而是直接关乎单卡 QPS、单 Token 成本以及集群规模的硬指标。

FlashInfer、TensorRT‑LLM、vLLM、DeepGEMM ——这些名字几乎构成了 2024–2026 年推理引擎的全部地标。

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/01.webp)

- **High Performance LLM Inference Operator Library**
- https://github.com/Tencent/hpc-ops
- 6000 字，阅读 35 分钟，**播客 25 分钟**

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**
- **[高性能算子的“包管理时代”：HF kernels-community 打通模型、硬件与 Hub，重塑底层加速分发链路！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901232&idx=1&sn=a493721d68f2578e2585b11effaaa871&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**

:::

> Tencent/hpc-ops[1]项目是腾讯混元 AI Infra 团队在 2026 年初开源的：宣称在 NVIDIA H20 上把 Attention 的 Decode 阶段拉到 2.22 倍加速，把 GroupGEMM 比 DeepGEMM 再快 1.88 倍。

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/02.webp)

::: image-caption
各算子实测最大加速比
:::

::: block-1
**短短数百行 CuTe + CUTLASS 代码，凭什么打穿这些工业级算子？**

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/03.webp)

::: image-caption
HPC-Ops 底层从NVIDIA GPU硬件出发，依托CUDA驱动层，向上搭载cute、multimem等底层基础组件，在此之上落地Attention、GEMM、通信规约等各类算子内核实现；再经由C++调用封装层，对外暴露PyTorch Python算子接口。配套测试体系包含单元测试、内存检测两类校验流程，整体接入CI自动化流水线，在代码迭代时自动完成全链路验证，从硬件、底层库、内核实现到上层API形成完整闭环，兼顾算子开发、封装与自动化质量管控。
:::

本文将顺着 `src/` 的目录走一遍，用最少的代码、最直白的中文，把它的设计哲学和关键工程手段理清楚。
:::

## 本文目录

::: related-links
- 快速上手、使用
- 一、架构总览与设计哲学
- 二、Hopper 的"三板斧"：TMA、WGMMA 与 Warp Specialization
- 三、GroupGEMM：把"成组矩阵乘"做成 Persistent Kernel
- 四、FusedMoE：把四个算子缝合成一个"DAG"
- 五、Attention：从 m64 到 warp specialization 的"按形选刀"
- 六、易被忽略但很关键的若干工程细节
- 七、它和 FlashInfer / TensorRT-LLM / DeepGEMM 的根本差异
- 写在最后：一份"现代 CUDA 教程"

:::

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/04.webp)

## 快速上手、使用

> HPC-Ops 是一个 PyTorch 自定义算子库，安装即用、API 与 vLLM/SGLang 风格一致。最精简的安装路径只需三条命令（前提：H20/SM90 GPU + CUDA 12.8+ + Python 3.8+，其它环境依赖见 requirements-dev.txt[2]）：

```
git clone https://github.com/Tencent/hpc-ops.gitcd hpc-opsmake wheel && python3 -m pip install dist/*.whl
```

调用方式与普通 PyTorch 算子无异：

```
import torch, hpcnum_tokens, num_group, n, k = 1024, 8, 4096, 4096x = torch.randn((num_tokens, k), device="cuda").to(torch.float8_e4m3fn)w = torch.randn((num_group, n, k), device="cuda").to(torch.float8_e4m3fn)scale = torch.full((num_group,), 1.0, device="cuda")num_tok = torch.full((num_group,), 8, dtype=torch.int32, device="cuda")cu_num_tok = torch.cumsum(    torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), num_tok]), 0).to(torch.int32)y = hpc.group_gemm_pertensor_fp8(x, w, num_tok, cu_num_tok, scale)
```

更多算子（Attention 的 Prefill/Decode、FusedMoE 的 blockwise/pertensor 两套量化路径、RoPE、RMSNorm、SiLU+Mul 融合等）的调用范式与张量约束，可直接参考 tests/[3] 下同名的单测文件，那里列出了每个算子对 shape、dtype、scale 的完整契约。

## 一、架构总览与设计哲学

### 1.1 一份"自包含"的极简仓库

> HPC-Ops 的目录哲学带着一种工程师式的克制：第三方库 `3rd/cutlass` 直接 vendored 进仓库而非用 git submodule，理由写在 3rd/README.md[4] 里**——"简化编译、保持自包含** "。整个仓库主干如下：

```
hpc/            # Python 层薄壳：act.py / attention.py / fuse_moe.py / group_gemm.py / rope.py …src/            # C++ / CUDA 实现  attention/    # prefill + decode（warp specialization、splitk、m64、smallm 多套实现）  group_gemm/   # GroupGEMM 的 pertensor / blockwise 两套 fp8 内核  fuse_moe/     # 把 GroupGEMM、激活、reduce 全部拼成一个 MoE  activation/   # SiLU、GELU、act_and_mul、量化融合  normalization/  rope/  utils/  C/    # 边角算子 + PyTorch 注册3rd/cutlass/    # vendored CUTLASS / CuTe
```

源代码总量并不大——这是它"清爽"的另一面：HPC-Ops 不打算做一个大而全的 op 库，而是只挑 **LLM 推理热点路径上的关键节点**，做到极致。

### 1.2 编译与发布：CMake + setup.py + torch.ops

> 构建链路看似平常，细节却体现了一个生产库的姿态。setup.py[5] 把构建完全委托给 CMake，并通过 `py_limited_api: cp39` 生成单一 ABI3 wheel——一份 `.whl` 在多 Python 版本下都能加载：

```
// 来源：setup.pyclass CMakeBuild(build_ext):    def build_extension(self, ext):        ...        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir)        subprocess.check_call(            ["cmake", "--build", ".", "--config", "Release", "-j16"], cwd=build_temp_dir        )        so_src_path = os.path.join(build_temp_dir, "_C.abi3.so")        so_dst_path = os.path.join(build_lib_dir, "hpc/_C.abi3.so")        shutil.copy(so_src_path, so_dst_path)
```

而 Python 入口 hpc/init.py[6] 也别有一番"老练"：

```
// 来源：hpc/__init__.pyso_files = list(Path(__file__).parent.glob("_C.*.so"))assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"torch.ops.load_library(so_files[0])           # 把算子注册到 torch.ops.hpc 命名空间__all__ = []_export_functions(_discover_modules())         # 遍历 hpc/*.py，把里面函数"提升"为顶层 API__version__   = torch.ops.hpc.version()__built_json__ = torch.ops.hpc.built_json()
```

两件事值得记住：

- 第一，**算子注册一律走 `torch.ops.hpc.*`，** 因此能直接被 `torch.compile`、`torch.export`、TorchScript 序列化，对接 vLLM / SGLang 时基本零成本；
- 第二，**Python 包装层不做调度，只做参数整理和默认值** ——所有 dispatch、tiling 选择都下沉到 C++/CUDA 端，避免了 Python 的 GIL 与解释开销。

### 1.3 设计哲学：少抽象、强模板、显式调优

> 如果用一句话概括 HPC-Ops 的"灵魂"，那就是：**不追求通用，但追求每一种典型形状下的最优**。

在 LLM 推理里，"典型形状"屈指可数——hidden=128 的 Attention head、tile_M ∈ {16, 32, 48, 64} 的 group GEMM、num_topk = 8 的 MoE……

::: block-1
于是它选择用 C++ 模板把这些维度全部编进类型系统：你看不到运行时 if/else 选 tile，看到的是 `kTileM/kTileN/kTileK/kStage/kSwizzle…` 一长串模板参数，编译期一刀切，运行期零分支。
:::

## 二、Hopper 的"三板斧"：TMA、WGMMA 与 Warp Specialization

要看懂 HPC-Ops，需要先认识 SM90（Hopper）赋予 CUDA 开发者的三件新武器。

### 2.1 TMA（Tensor Memory Accelerator）：让搬运变成"DMA"

- 在 Ampere 时代，从 Global Memory 把数据搬到 Shared Memory 仍由 warp 内的线程执行 `cp.async`。
- Hopper 引入 TMA：一条指令描述一块多维子张量的搬运，硬件 DMA 引擎自动完成，并通过 `mbarrier`（异步屏障）通知 warp"我搬完了"。

> HPC-Ops 借助 CuTe 的 `make_tma_copy` 把 TMA 描述符直接构造在 host 端：

```
// 来源：src/group_gemm/config.htemplate <typename TX, typename TW, typename TY>auto get_tma(TX x, TW w, TY y) {  auto tma_x = make_tma_copy(SM90_TMA_LOAD{},  x, take<0, 2>(SLayoutX{}));  auto tma_w = make_tma_copy(SM90_TMA_LOAD{},  w, take<0, 2>(SLayoutW{}));  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{});  return std::make_tuple(tma_x, tma_w, tma_y);}
```

`SLayoutX/W/Y` 都来自下面的 swizzle 选择器，确保搬到 Shared Memory 之后的数据排布能直接喂给 WGMMA 而无需任何"重排"。

### 2.2 WGMMA（Warpgroup MMA）：以 4 warp 为最小调度单元

> SM90 的张量核心一次以 **一个 warpgroup（4 warps = 128 threads）** 为单位发起 `wgmma` 指令，单条指令完成 64×N×32 的 fp8/bf16 矩阵乘。HPC-Ops 用一个 `mma_selector` 把所有 tile 形状的 wgmma 全部封装好：

```
// 来源：src/group_gemm/config.htemplate <int kTileM>static constexpr auto mma_selector() {if      constexpr (kTileM == 16) return cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};elseifconstexpr (kTileM == 32) return cute::SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};elseifconstexpr (kTileM == 48) return cute::SM90_64x48x32_F32E4M3E4M3_SS_TN<>{};elseifconstexpr (kTileM == 64) return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};  ... // 96, 128 类似}
```

四个候选 tile_M（16/32/48/64）并非随手挑：**它们恰好对应 MoE 解码场景下每个 expert 平均承接的 token 数**。运行时根据负载自动选 tile_M（见后文 2.5 节），:mdnice-bold-italic[让"小 batch + 多 expert"的 LLM 解码场景也尽可能跑满 Tensor Core。]

### 2.3 Warp Specialization：让 producer 与 consumer 解耦

> Hopper 之前的常见做法是 "all warps both load and compute"。WS（Warp Specialization）则把 warps 分两类：

- **producer warp** 专职发 TMA
- **consumer warp** （更确切说是 warpgroup）专职跑 WGMMA

两者通过 `mbarrier` 同步、通过 shared-memory 上的环形 buffer（`kStage` 控制深度）异步流水。在 `group_gemm_pertensor_fp8.cu` 里能看见这个味道：

```
// 来源：src/group_gemm/group_gemm_pertensor_fp8.cudim3 block(384);                  // 1 producer + 2 consumer warpgroup = 384 threaddim3 grid(get_sm_count());        // persistent kernel：每个 SM 一个 block
```

`384 = 32 × 12 = 4 warps（producer） + 8 warps（2 个 consumer warpgroup）`，再叠加 `kStage=8` 的环形 buffer，TMA 与 WGMMA 完全异步：当 producer 在加载 `stage k+1` 时，consumer 已经在算 `stage k`。

### 2.4 Swizzle：搬数据顺手把 bank conflict 解了

> SM90 的 Shared Memory 仍有 32 个 bank，跨 warpgroup 的 wgmma 一次会读 128B 大段，:mdnice-bold-italic[若不 swizzle，bank conflict 会让带宽掉一半。HPC-Ops 用统一的选择器把 swizzle 压进 SLayout 里：]

```
// 来源：src/group_gemm/config.htemplate <int kSwizzle, typename T, bool kKmajor = true>staticconstexprauto slayout_selector() {if      constexpr (kSwizzle == 128) return kKmajor ? cute::GMMA::Layout_K_SW128_Atom<T>{}                                                      : cute::GMMA::Layout_MN_SW128_Atom<T>{};elseifconstexpr (kSwizzle == 64)  ...elseifconstexpr (kSwizzle == 32)  ...else                                ...}
```

K‑major 的 X/W 用 `SW128`、N‑major 的 Y 用 `SW64`——这就是 GroupGEMM 配置里 `kSwizzleX=128, kSwizzleW=128, kSwizzleY=64` 的来历。

::: block-1
这种粒度的细节，正是 HPC-Ops 与"通用算子"的分水岭：它认死了一组 tile 形状，把每一个 16B 段的去向都摆好。
:::

### 2.5 运行时调优：唯一被允许的 if/else

> 绝大多数 dispatch 在编译期，但 tile_M 这种**与 batch 强相关** 的维度只能在运行时定。HPC-Ops 把它写得直白干脆：

```
// 来源：src/group_gemm/group_gemm_pertensor_fp8.cuif      (num_seq_per_group_avg <= 16) { constexpr int kTileM = 16; ... }else if (num_seq_per_group_avg <= 32) { constexpr int kTileM = 32; ... }else if (num_seq_per_group_avg <= 48) { constexpr int kTileM = 48; ... }else                                  { constexpr int kTileM = 64; ... }
```

四套模板特化产物全部静态实例化进 `.so`，运行时只是挑指针——零分支损耗、零代码分支预测污染。

## 三、GroupGEMM：把"成组矩阵乘"做成 Persistent Kernel

> GroupGEMM 在 MoE 推理中是核心瓶颈：每个 expert 收到的 token 数都不同，传统的"每 expert 一次 cublas 调用"会被 launch overhead 拖死。

HPC-Ops 的做法是把所有 group 揉成一个**持久化 kernel** ——`grid = SM 数`，每个 block 自己去"领工作"。

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/05.webp)

::: image-caption
该图由四张子图组成，从TP、EP两种并行策略，Prefill、Decode两大推理阶段，对比HPC两种实现与DeepGEMM两类分块方案的GroupGEMM耗时，纵坐标为延迟（微秒，数值越低性能越好）、横坐标是每组序列长度。随着每组序列尺寸增大，所有方案延迟同步上涨；全场景下蓝色HPC系列延迟始终低于灰色DeepGEMM实现，HPC blockwise优化表现最优。Prefill阶段序列取值256~4096、Decode阶段1~128，Decode整体耗时远低于Prefill，且HPC相较原生DeepGEMM在稀疏分组矩阵运算中具备稳定性能优势。
:::

### 3.1 工作分配：`tiles_ptr` 与 `cu_tiles_ptr`

> 在调用 GEMM 主 kernel 之前，HPC-Ops 先用一个小 kernel `update_grouped_tma` 计算每个 group 在 tile_M 切分下产生多少 tile，并写入 `tiles_ptr`，对应的前缀和写入 `cu_tiles_ptr`：

```
// 来源：src/group_gemm/group_gemm_pertensor_fp8.cuif (update_tma) {  kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y),                              kTileM, kGroupPerThread, kThreadPerBlock>      <<<num_group + 1, kThreadPerBlock, 0, stream>>>(          td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr,          (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,          (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);}
```

这一步还顺便把每个 group 单独的 TMA 描述符（基址、tensor shape 都因 group 而异）写进 `tma_xy` 数组里——以后主 kernel 拿到 tile id，先二分查 group，再用对应的 TMA 描述符发 load，**完全规避了运行时的 cudaMemcpy 或 stream switch**。

### 3.2 主 kernel：一次 launch 吃光所有 group

```
// 来源：src/group_gemm/group_gemm_pertensor_fp8.cuint num_tile_n = (n + kTileN - 1) / kTileN;cutlass::FastDivmod flat_divider(num_tile_n);    // 用乘加替代整数除法dim3 block(384);dim3 grid(get_sm_count());                       // persistent gridint shm_size = config.get_shm_size() + sizeof(int) * (num_group + 1);cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);kernel<<<grid, block, shm_size, stream>>>(    tma_w, tma_xy, (int*)seqlens_ptr, (float*)y_scale,    (int*)tiles_ptr, (int*)cu_tiles_ptr, num_group, m, n, k, flat_divider);
```

两个工程细节：

- **`cutlass::FastDivmod`：** 在 kernel 里要把"全局 tile id"反解成 `(group_id, tile_m, tile_n)`，会出现大量整数除法/取模。CUTLASS 的 FastDivmod 用乘‑移位替代，单条 IMAD 完成。
- **`cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`：** H20 上单个 SM 最多可分配 228 KB shared memory，但默认上限只有 48 KB，必须显式申请。HPC-Ops 把 X/W/Y 三块 buffer 加上一份 `seqlens` 索引表全部塞进 shared，再借 `kStage=8` 跑深流水。

### 3.3 IsLoopH：根据形状切换循环方向

> GroupGEMM 主 kernel 模板上还有个 `IsLoopH` 开关：

```
// 来源：src/group_gemm/group_gemm_pertensor_fp8.cuif (k <= 1024 || n <= 1024) {constexprbool IsLoopH = true;   // 形状窄时，外层循环 K 维以摊薄 storeauto kernel = kernels::group_gemm_pertensor_fp8_kernel<..., IsLoopH>;  ...} else {constexprbool IsLoopH = false;  // 形状大时，外层循环 N 维以摊薄 loadauto kernel = kernels::group_gemm_pertensor_fp8_kernel<..., IsLoopH>;  ...}
```

不同形状下，"K 在外"和"N 在外"对 LDS bank 的访问模式、对 WGMMA 累加器寄存器的压力完全不同。HPC-Ops 干脆生成两份 kernel，运行时挑一份——这与"通用 GEMM"的最大不同就是它**不在乎二进制大小，只在乎极致性能**。

## 四、FusedMoE：把四个算子缝合成一个"DAG"

> MoE 在推理里其实是一条小流水线：`count_and_gather → gate_up GEMM → SiLU+Mul+量化 → down GEMM → reduce`。任何一步落到 Python，都会被 launch overhead 与 PCIe 同步打断。

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/06.webp)

::: image-caption
本图由四张子图构成，在TP、EP两种分布式并行、Prefill与Decode两大推理场景下，对比HPC、TensorRT-LLM、vLLM三套方案的FusedMoE算子延迟，纵轴为耗时（微秒，数值越低性能越优）、横轴是单专家承载token数量。随单专家token数提升，三类实现的运算延迟同步走高；全场景里蓝色HPC柱状高度始终最低，性能全面优于TensorRT-LLM与vLLM，vLLM灰色柱耗时最高。Prefill阶段token跨度256~4096、Decode为16~128，Prefill整体运算延迟显著高于Decode，HPC在MoE融合算子上相较主流推理框架具备稳定提速效果。
:::

HPC-Ops 在 `fuse_moe.cu` 里把这条 DAG 完整地写到 host 函数里，所有算子共享同一组 buffer 与同一条 stream：

```
// 来源：src/fuse_moe/fuse_moe.cuvoid fuse_moe_pertensor_fp8_async(/* …超长参数表… */ cudaStream_t stream) {int total_num_seq = num_seq * num_topk;int num_seq_per_group_avg = total_num_seq / num_expert_total;// 0. 路由统计 + token 重排（按 expert 聚拢）+ TMA 描述符就地更新  count_and_gather_async(gate_up_input_ptr, ..., tiles_ptr, cu_tiles_ptr,                         num_seq, hidden_size, intermediate_size, num_topk,                         num_expert_local, rank_ep, num_seq_per_group_avg, stream);// 1. gate_up GroupGEMM（fp8）  group_gemm::group_gemm_pertensor_fp8_async(      gate_up_output_ptr, gate_up_input_ptr, gate_up_weight_ptr,      seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr,      gate_up_tmas_ptr, tiles_ptr, cu_tiles_ptr, num_expert_local,      total_num_seq, intermediate_size, hidden_size,      num_seq_per_group_avg, /*update_tma=*/false, stream);// 2. SiLU(act) * Mul + 在线量化为 fp8（写到 down_input）constint *valid_row_range_ptr = (int *)cu_seqlens_ptr + num_expert_local;  activation::act_mul_and_quant_async((T2 *)down_input_ptr,      (const T1 *)gate_up_output_ptr, (constfloat *)act_and_mul_scale_ptr,      valid_row_range_ptr, total_num_seq, intermediate_size,      use_bf16_mul, stream);// 3. down GroupGEMM（fp8）  group_gemm::group_gemm_pertensor_fp8_async(      down_output_ptr, down_input_ptr, down_weight_ptr,      seqlens_ptr, cu_seqlens_ptr, down_scale_ptr,      down_tmas_ptr, tiles_ptr, cu_tiles_ptr, num_expert_local,      total_num_seq, hidden_size, intermediate_size / 2,      num_seq_per_group_avg, false, stream);// 4. 按 topk_pos / topk_scale 把每个 token 的 top-k 输出加权求和回原位  reduce_async(output_ptr, down_output_ptr, topk_pos_ptr, topk_scale_ptr,               shared_output_ptr, total_num_seq, num_seq,               hidden_size, num_topk, stream);}
```

短短 40 行 host 代码，体现了三个工程要点：

1. **TMA 描述符复用**：只有 `count_and_gather` 阶段会 `update_tma`，后面两个 GroupGEMM 都传 `update_tma=false`，直接复用 step 0 写好的描述符数组。这意味着 host 侧再无任何 `cudaMemcpy`，全部寄居在 device global memory。
2. **激活与量化融合到一个 kernel**：`act_mul_and_quant_async` 既算 `silu(gate) * up`、又顺手在 register 里算 fp8 缩放，避免把 bf16 中间结果落回 HBM——这是省 HBM 带宽的关键。
3. **`shared_output_ptr` 直送 reduce**：reduce 阶段把 expert 输出 + shared expert 输出加权合并，少一次 ATen 加法 kernel。

而对应的 `fuse_moe_blockwise_fp8_async` 走另一条路径：把 weight 与 activation 都按 128×128 block 做更精细的量化，配上 `group_gemm_blockwise_fp8_async`。从代码组织看，它和 pertensor 路径**对称地共享 host 编排**，只是把 scale 从一个标量升级为一张二维表——:mdnice-bold-italic[这种"主路径不变、量化策略可插拔"的写法，正是它能同时跑 W8A8 与更细颗粒度量化的根本原因。]

## 五、Attention：从 m64 到 warp specialization 的"按形选刀"

> `src/attention/` 是整个仓库最热闹的一处🤣，光是 decode 就有四五份实现：

```
src/attention/decode/  m64_dim128.cu                 # 经典 64×128 Hopper kernel  m64_dim80.cu                  # 80 维 head（部分 MoE 模型）  smallm_dim128.cu              # 小 M 直接走 single-stage  smallm_splitk_dim128.cu       # split-K 切 KV，解决 decode 长 context  smallm_splitk_dim128_fp8.cu   # split-K + fp8 KV-cache  smallm_splitk_combine_kernels.cuh
```

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/07.webp)

::: image-caption
该流程图拆解Attention Decode阶段的分块优化逻辑，左侧是原始计算矩阵Q/K/P/V/Y，依靠TileAndLoop沿KV序列长度维度切分矩阵，得到分块后的算子任务；再对K轴二次分块，拆出两类计算任务，依托GPU MMA矩阵指令调度执行。常规任务将Q、K、P映射为MMA的A/B/C操作数，SwapAB任务互换矩阵排布，调换Q与K在MMA输入位置，适配硬件矩阵运算指令格式，通过分层分块贴合GPU硬件计算粒度，拆分大矩阵访存与计算负载，以此优化解码阶段Attention的算子执行效率。
:::

而 `prefill/` 则区分 `multi_stage` 与 `warp_spec`（warp specialization）两个家族，并且每种又分 `with_kvcache` / `without`、`fp8` / `bf16`：

```
src/attention/prefill/  multi_stage_dim128.cu                    # 通用 multi-stage 流水  multi_stage_with_kvcache_dim128.cu       # 续推（带 kv-cache）  warp_spec_dim128.cu                      # WS：producer/consumer 解耦  warp_spec_with_kvcache_dim128.cu  warp_spec_with_kvcache_fp8_dim128.cu     # FP8 KV-cache
```

设计意图非常清晰：**Decode 的瓶颈是访存（KV-Cache 大、batch 小、M 小），Prefill 的瓶颈是计算（M 长，需要深流水掩盖 TMA）**。所以 decode 路径把 `split-K + small-M` 玩到极致，让"一个 token 对几千个 KV"的访存问题通过多个 block 协同 reduce 解决；prefill 路径则把 warp specialization 上到最强档，让 WGMMA 永远不停。

::: block-1
文件命名本身就构成了一份"按形选刀"的索引：HPC-Ops 不试图写一个通吃的 attention kernel，而是把每种业务最常见的 head_dim、batch、context 长度都准备一把"专刀"，运行时调度器（在 `entry.cc` 中）依据请求形状选其中一把发出去——:mdnice-bold-italic[这与 FlashAttention‑3 用 schedule template 在 kernel 内动态切换的思路截然不同，而是把"分支"完全前移到了调度器，kernel 本身极简、寄存器压力低、占用 (occupancy) 高。]
:::

![文章插图](/articles/2026-06-04-h20-ai-infra-hpc-ops-fd862de9/images/08.webp)

::: image-caption
该图由四张子图组成，从BF16、FP8两种精度、Prefill与Decode两大推理阶段，横向对比HPC、FlashAttention2/3、FlashInfer、TensorRT-LLM的Attention算子延迟，纵坐标为耗时（单位微秒，数值越低性能越好），横轴依次排布不同序列长度与Batch组合。随着序列长度增大，全部方案延迟同步抬升；全测试用例中蓝色HPC柱状始终最低，性能全面优于其余基线方案。Decode整体延迟远低于Prefill，FP8量化相较BF16普遍进一步压缩运算耗时，HPC在长序列、大Batch场景下的性能领先优势尤为突出。
:::

## 六、易被忽略但很关键的若干工程细节

### 6.1 `tiles_ptr / cu_tiles_ptr`：device 侧的调度表

> 在 MoE 一类"动态形状"算子里，最难的并不是矩阵乘本身，而是**让 GPU 自己知道要算多少 tile**。

HPC-Ops 的解法是把"动态形状"提前在 device 上展开成一张 tile 表：每条 entry 描述一个 tile 属于哪个 group、覆盖哪段 token、读哪段 weight。所有后续 kernel 都拿同一份表，配合 `cu_tiles_ptr` 的前缀和实现 O(1) 索引。

::: block-1
这种"device side scheduling table"是 SM90 持久化 kernel 的标准范式，HPC-Ops 把它做得很薄，但工程价值极高。
:::

### 6.2 持久化 grid + `get_sm_count()`

注意 GroupGEMM 主 kernel 的 `grid = get_sm_count()`：H20 的 132 个 SM 一人发一个 block，block 内主动从 tile 表中"抢工作"。这种**persistent kernel + work-stealing** 模式既能消除 launch 抖动，又能让长尾形状（最后一组 expert 只有几个 token）不至于把整个 SM 阵列空转。

### 6.3 `cudaFuncSetAttribute` 的两次出现

Hopper 上动辄 100KB 以上的 shared memory 需求都要显式申请，否则会被 driver 默默限制在 48KB。HPC-Ops 在每个 launch 点都会先 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size)`，把这个常见的"占用率/正确性陷阱"摆在主路径上。

### 6.4 测试与 sanitizer：把"工程化"写在 Makefile 里

> Makefile[7] 里特意留了一个 `sanitizer` target：

```
sanitizer:$(PY_TEST) @rm -rf /dev/shm/tmp_hpc_* @for test in $^; do \   PYTORCH_NO_CUDA_MEMORY_CACHING=1 \   SANITIZER_CHECK=synccheck,memcheck,racecheck \   NV_SANITIZER_INJECTION_PORT_BASE=1111 \   python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \ done
```

`synccheck + memcheck + racecheck` 三件套，再加上关闭 PyTorch 自己的 CUDA 缓存——任何 TMA mbarrier 用错、shared-memory 数组越界、warp 间竞态，都会在这里现形。这种"愿意为正确性付出运行慢 10x 的回归代价"的态度，是它敢自称 *production-grade* 的真正底气。

## 七、它和 FlashInfer / TensorRT-LLM / DeepGEMM 的根本差异

> 读到这里，我们可以回头审视那张性能表：

- **Attention 2.22× 提速** 来自 decode 阶段的 `smallm_splitk_*`：把"长 KV-Cache、小 batch"这一推理痛点单独成 kernel，再用 split-K combine 把多个 SM 的部分和合起来。FlashAttention/FlashInfer 也支持 split-K，但其调度面向训练优先；:mdnice-bold-italic[HPC-Ops 则把所有 hyperparameter 都 fix 到推理常见形状上做编译期特化。]
- **GroupGEMM 1.88× 提速 DeepGEMM** 来自 persistent kernel + TMA 描述符复用 + tile_M 在 16/32/48/64 间精细切换。DeepGEMM 默认的 tile 选择更倾向"通用最优"，:mdnice-bold-italic[而 HPC-Ops 直接为 expert‑per‑token ≤ 64 这一狭窄区间做了 4 套特化。]
- **FusedMoE 1.49× 提速 TRT-LLM** 来自 `count_and_gather → GEMM → act_quant → GEMM → reduce` 整链路 stream 内零拷贝，以及 act_and_mul 与 fp8 量化的 register 级融合。

::: block-1
三者背后是一个共同的工程信念：**把 LLM 推理热路径上的每一个常量都焊死在编译期，把每一次 launch、每一次 H2D 同步都消灭在 stream 内部**。这个"少而精"的姿态，恰恰是当下推理引擎最稀缺的工程品味。
:::

## 写在最后：一份"现代 CUDA 教程"

> README 的最后一句话很值得玩味："**A Modern CUDA Tutorial: hands-on examples of building SOTA kernels with CuTe and CUTLASS in just hundreds of lines.** "这不是营销，而是事实——把 `group_gemm_pertensor_fp8.cu` 加上 `config.h`、`kernels.cuh` 三个文件全部读完，你已经掌握了 SM90 上写一个 SOTA fp8 GEMM 所需的全部要素：CuTe layout 思维、TMA 描述符构造、WGMMA SS_TN 形态、warp specialization、persistent grid、FastDivmod、dynamic shared memory 申请、运行时 tile_M dispatch。

对国内研发团队而言，HPC-Ops 的开源至少有三层意义：

- **可对接**：以 `torch.ops.hpc.*` 暴露，vLLM/SGLang 集成成本几乎为零；
- **可学习**：每一个 kernel 都不到几百行，读起来比 FlashInfer/TRT-LLM 庞大的代码库友好得多，是真正能"上手照抄"的 Hopper 入门样本；
- **可演进**：Roadmap 已经列出 Sparse Attention、4/8bit 混合量化、计算‑通信重叠 三个方向，意味着它会随长上下文与 EP 并行的发展持续迭代。

在"算子库越来越像编译器"的今天，HPC-Ops 反其道而行——**用模板把编译器该干的事一次做完，把 kernel 写得短、写得快、写得敢于发布**。

::: block-1
当你下一次怀疑"H20 是不是被低估了"，不妨先把这个仓库 clone 下来，跑一遍 `make test`，再回头看看自己的推理栈，还有多少 launch 没被消灭、还有多少 tile 没被特化。答案，往往就在那短短数百行 CuTe 代码里。
:::

参考资料

[1] Tencent/hpc-ops: *https://github.com/Tencent/hpc-ops*

[2] requirements-dev.txt: *https://github.com/Tencent/hpc-ops/blob/main/requirements-dev.txt*

[3] tests/: *https://github.com/Tencent/hpc-ops/tree/main/tests*

[4] 3rd/README.md: *https://github.com/Tencent/hpc-ops/blob/main/3rd/README.md*

[5] setup.py: *https://github.com/Tencent/hpc-ops/blob/main/setup.py*

[6] hpc/init.py: https://github.com/Tencent/hpc-ops/blob/main/hpc/:mdnice-bold-italic[init].py

[7] Makefile: *https://github.com/Tencent/hpc-ops/blob/main/Makefile*

::: related-title
**相关推荐**
:::

::: related-links
- **[AMD AITER 算子库：同时驾驭汇编、CK 与 Triton 三套内核引擎](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900415&idx=1&sn=33ced8712ed2f3f0f2d794a959c80868&scene=21#wechat_redirect)**
- **[高性能算子的“包管理时代”：HF kernels-community 打通模型、硬件与 Hub，重塑底层加速分发链路！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901232&idx=1&sn=a493721d68f2578e2585b11effaaa871&scene=21#wechat_redirect)**
- **[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900291&idx=1&sn=023e27c629ba44ef76aed57c2edac4f1&scene=21#wechat_redirect)**

:::
