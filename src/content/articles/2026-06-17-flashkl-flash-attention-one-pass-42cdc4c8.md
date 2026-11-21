**投稿/寻求报道/文章纠错：公众号后台 -> 联系我们**

> 【投稿】目的是扩大影响力，不愿被单一公众号绑定，故而投稿文稿【不标注原创】，望周知！

关键词：**FlashKL**、:mdnice-bold-italic[Attention KL Loss]、One-Pass、:mdnice-bold-italic[Tiling]、**稀疏注意力**

![文章插图](/articles/2026-06-17-flashkl-flash-attention-one-pass-42cdc4c8/images/01.png)

- **FlashKL：像 Flash Attention 一样 One-Pass 计算 Attention KL Loss**
- 作者：Eleven，https://zhuanlan.zhihu.com/p/2048512989352010720
- :mdnice-bold-italic[FlashKL代码仓库：github.com/XiaojuanTang/FlashKL]

::: related-title
**相关推荐**
:::

::: related-links
- **[消除推荐系统广播税，从内存带宽到计算瓶颈的蜕变，揭秘 621 TFLOPS 的 Flash Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900976&idx=1&sn=0f29d856ded3ed90bc6c1c4ce3152d0f&scene=21#wechat_redirect)**
- **[碾压GH200注意力实现：FlatAttention在Tile数据流加速器上，计算场景利用率达86%，访存场景带宽利用率达 78%，注意力内核平均加速 1.9倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899603&idx=1&sn=59da0ff7b8208a308553b463fc36a817&scene=21#wechat_redirect)**
- **[质疑“自动生成 FlashAttention”：浅析 Luminal 编译器的设计与现实落差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900430&idx=1&sn=e1847048208639c245dac0a1135d1700&scene=21#wechat_redirect)**

:::

> 最近在看长文训练，绕不开 DeepSeek 提出的 DSA（DeepSeek Sparse Attention）。其实除了文本训练，长视频模型也非常需要稀疏化操作，最近快手开源的 Keye-VL-2.0 就把 DSA 引入了多模态场景。**但真正上手实现的时候发现一个问题：DSA 的 warmup 阶段需要计算 KL loss，朴素实现要把完整的 $L \times L$ attention 矩阵存下来，序列长度一旦到 512K、1M，显存直接炸了**。

其实需要计算 attention 分布间 KL loss 的场景有很多，例如 attention 的压缩或剪枝、高效 attention 与标准 attention 的对齐、蒸馏时 teacher-student 逐层 attention 分布匹配等。如果有一种不用 materialize 完整 attention 矩阵就能算 KL loss 的方法，上述场景都能直接受益。

::: block-1
所以：有没有办法像 Flash Attention 那样，用 tiling + one pass 的方式，**一趟扫过去就把 KL loss 算完**，显存从 $O(L^2)$ 降到 $O(L)$，而且不损失效果呢？如果是两个 softmax 分布之间的 KL，社区已经有 one-pass 的 TileLang 实现（如tilelang-dsa[1]），但其算法需要先拿到 softmax 之前的 logits。
:::

然而在 DSA 的标准实现中，teacher attention 是多头平均后的分布：

$$
p_{t,:} = \frac{\sum_{h=1}^H A_h[t, :]}{\Vert \sum_{h=1}^H A_h[t, :] \Vert_1}
$$

显然 $\sum_h \text{Logits}^h_{t, :} \neq \text{Logits}^\prime_{t,:}$，想从平均后的分布倒推出等价 logits 非常复杂。因此他们折中采用 per-head KL loss，但这样 indexer 就没办法用更少的 head，违背了[Lightning Indexer]( "Lightning Indexer")的设计初衷。

为此，我提出了 FlashKL 的方式，像 Flash Attention 一样 one-pass 计算 attention KL loss。它不仅适用于通用的 attention KL loss 计算，在 DSA warmup 这类涉及多头平均的场景中也能完美适配。

![文章插图](/articles/2026-06-17-flashkl-flash-attention-one-pass-42cdc4c8/images/02.png)

::: image-caption
FlashKL github.com/XiaojuanTang/FlashKL
:::

::: block-1
本文代码已开源，包含 PyTorch 参考实现与完整的正确性验证，后续将提供基于 TileLang 的 fused kernel 实现。
:::

## 本文目录

- DSA
- 1. KL loss 公式推导
  - 1.1 符号约定
  - 1.2 Teacher 分布 $\bar{p}_{t,s}$
  - 1.3 Indexer 分布 $\hat{p}_{t,s}$
  - 1.4 简化推导
- 2. 梯度计算
  - 2.1 对 indexer logits $I_{t,s}$
  - 2.2 对 indexer 参数 $q^I, k^I, w$
- 3. FlashKL 算法
  - 3.1 核心思路
  - 3.2 定义统计量
  - 3.3 更新公式
  - 3.4 Forward Pass 伪代码
  - 3.5 Backward Pass 伪代码
- 4. 为什么 teacher entropy 不可以 one-pass 计算呢？是否可以 tiling 计算呢？
  - 4.1 单头情况（$H = 1$）
  - 4.2 多头情况（$H > 1$）
- 小结
- References

![文章插图](/articles/2026-06-17-flashkl-flash-attention-one-pass-42cdc4c8/images/03.jpg)

## DSA

![文章插图](/articles/2026-06-17-flashkl-flash-attention-one-pass-42cdc4c8/images/04.png)

> 回顾一下，DSA（DeepSeek Sparse Attention）采用两阶段训练：

第一阶段通过 warmup 训练一个轻量级的 Lightning Indexer，使其学会预测 teacher attention 的分布，从而在推理时仅对 top-k 重要的 key 位置做完整 attention，大幅降低长序列计算量。Warmup 的目标函数是 indexer 输出分布 $\hat p$ 与 teacher attention 多头平均分布 $\bar p$ 之间的 KL 散度。如前所述，朴素实现需要 $O(L^2)$ 显存，现有 one-pass 方案又无法处理先 softmax 后多头平均的情况。FlashKL 通过一个简单的数学变换就能巧妙解决这个问题，下面来看具体做法。

## 1. KL loss 公式推导

### 1.1 符号约定

| 符号 | 含义 | 维度 |
| --- | --- | --- |
| $L$ | 序列长度 | scalar |
| $H$ | 主 attention 的 head 数 | scalar |
| $H_I$ | indexer 的 head 数 | scalar |
| $d, d_I$ | 主 / indexer 的 head 维度 | scalar |
| $x_t$ | 第 $t$ 个 token 的 hidden state | $\mathbb{R}^D$ |
| $t$ | query 索引（外层 sum 跑遍 $t = 1\dots L$） | — |
| $s$ | key 索引（内层 sum 跑遍 $s$） | — |

### 1.2 Teacher 分布 $\bar{p}_{t,s}$

主 attention 第 $h$ 个 head 在 query $t$ 上的 logits 和 attention score：

$$
\alpha^{(h)}_{t,s} = q^{(h)\top}_t k^{(h)}_s / \sqrt{d} \quad\longrightarrow\quad p^{(h)}_{t,s} = \mathrm{softmax}_s(\alpha^{(h)}_{t,:}) = \frac{\exp(\alpha^{(h)}_{t,s})}{\sum_{s'} \exp(\alpha^{(h)}_{t,s'})} \tag{1}
$$

跨 head 求和再 L1 归一化：

$$
\boxed{\;\bar{p}_{t,s} = \frac{1}{H}\sum_{h=1}^{H} p^{(h)}_{t,s}\;} \tag{2}
$$

### 1.3 Indexer 分布 $\hat{p}_{t,s}$

Indexer 的 raw score：

$$
I_{t,s} = \sum_{j=1}^{H_I} w_{t,j} \cdot \mathrm{ReLU}\!\left( q^{I,(j)\top}_t k^I_s \right) \tag{3}
$$

再套一层 softmax 得到 student 分布：

$$
\boxed{\;\hat{p}_{t,s} = \mathrm{softmax}_s(I_{t,:})_s = \frac{\exp(I_{t,s})}{\sum_{s'} \exp(I_{t,s'})}\;} \tag{4}
$$

### 1.4 简化推导

KL 散度展开：

$$
\mathrm{KL}(\bar{p} \| \hat{p}) = \sum_s \bar{p}_{t,s} \log \frac{\bar{p}_{t,s}}{\hat{p}_{t,s}} = \sum_s \bar{p}_{t,s} \log \bar{p}_{t,s} - \sum_s \bar{p}_{t,s} \log \hat{p}_{t,s}
$$

整个 loss：

$$
\mathcal{L}^I = \sum_t \mathrm{KL}(\bar{p}_{t,:} \| \hat{p}_{t,:}) = \underbrace{\sum_t \sum_s \bar{p}_{t,s} \log \bar{p}_{t,s}}_{(A)\ \text{teacher entropy 项}} - \underbrace{\sum_t \sum_s \bar{p}_{t,s} \log \hat{p}_{t,s}}_{(B)\ \text{cross-entropy 项}} \tag{5}
$$

由于 teacher model 参数冻结，所以优化的 loss 可以去掉 $(A)$ 项。loss 可写成

$$
\mathcal{L}^I_{\text{trainable}} = -\sum_t \sum_s \bar{p}_{t,s} \log \hat{p}_{t,s}
$$

因为：

$$
\hat{p}_{t,s} = \frac{\exp(I_{t,s})}{\sum_{s'}\exp(I_{t,s'})}
$$

代入化简：

$$
\mathcal{L}^I_{\text{trainable}} = -\sum_t \sum_s \bar{p}_{t,s}\, I_{t,s} \;+\; \sum_t \log \sum_{s'} \exp(I_{t,s'}) \cdot \underbrace{\sum_s \bar{p}_{t,s}}_{=\,1}
$$

logsumexp 项可以提到 $s$ 求和外面，又 $\sum_s \bar{p}_{t,s} = 1$，

$$
\boxed{\;\mathcal{L}^I = -\sum_t \underbrace{\sum_s \bar{p}_{t,s}\, I_{t,s}}_{\text{加权点积}} \;+\; \sum_t \underbrace{\log \sum_{s'} \exp(I_{t,s'})}_{\mathrm{logsumexp}_{\,s}(I_{t,:})} \;+\; \text{const}\;} \tag{6}
$$

把 $\bar{p} = \frac{1}{H}\sum_h p^h$ 代入加权点积那项：

$$
\sum_s \bar{p}_{t,s}\, I_{t,s} = \frac{1}{H}\sum_h \sum_s p^h_{t,s}\, I_{t,s} \tag{7}
$$

每个 head 可以独立算 $\sum_s p^h_{t,s}\, I_{t,s}$，最后进行平均即可。事实上，逐 tile 扫描时 $p^{(h)}_{t,s}$ 的归一化常数会随新 tile 不断更新，但 $I_{t,s} = \sum_{j=1}^{H_I} w_{t,j} \cdot \mathrm{ReLU}\!\left( q^{I,(j)\top}_t k^I_s \right)$ 的计算只和当前 $t, s$ 有关，所以相当于 $\sum_s p^{(h)}_{t,s}\, I_{t,s}$ 对 $p^{(h)}$ 是线性的，也就是说当归一化常数因新 tile 更新时，只需对之前的累积值乘一个 rescale 系数 $e^{m_{\text{old}} - m_{\text{new}}}$ 即可修正，这正是 Flash Attention 的 tiling 核心技巧。因此 one-pass tiling 就能得到精确结果。

::: block-1
这可能也解释了 DSA 在设计时 indexer 为什么在 logit 阶段做 head 加权求和而不是先 softmax 再加权：前者产出的 $I_{t,s}$ 是和其他 $s$ 无关的 logit，可以直接参与上述线性累积；后者会引入额外的 softmax 归一化，破坏 one-pass 所需的线性结构。
:::

## 2. 梯度计算

### 2.1 对 indexer logits $I_{t,s}$

$$
\frac{\partial \mathcal{L}}{\partial I_{t,s}} = -\bar p_{t,s} + \frac{\exp(I_{t,s})}{\sum_{s'}\exp(I_{t,s'})} = \hat p_{t,s} - \bar p_{t,s} \tag{8}
$$

梯度就是 indexer 预测的分布和 teacher 平均分布的差。

展开多头：

$$
\frac{\partial \mathcal{L}}{\partial I_{t,s}} = \hat p_{t,s} - \frac{1}{H}\sum_h p^h_{t,s}
$$

### 2.2 对 indexer 参数 $q^I, k^I, w$

记 $g_{t,s} = \hat p_{t,s} - \bar p_{t,s}$。由公式 (3)，$I_{t,s} = \sum_{j=1}^{H_I} w_{t,j} \cdot \mathrm{ReLU}(z^{(j)}_{t,s})$，其中 $z^{(j)}_{t,s} = {q^{I,(j)}_t}^\top k^I_s$。

对 pre-activation $z^{(j)}_{t,s}$，ReLU 引入一个 mask：

$$
\frac{\partial \mathcal{L}}{\partial z^{(j)}_{t,s}} = g_{t,s} \cdot w_{t,j} \cdot \mathbb{1}[z^{(j)}_{t,s} > 0] \tag{9}
$$

记 $\tilde G^{(j)}_{t,s} = g_{t,s} \cdot w_{t,j} \cdot \mathbb{1}[z^{(j)}_{t,s} > 0]$，则参数梯度为：

$$
dQ^{I,(j)} = \tilde G^{(j)} \cdot K^I, \qquad dK^I = \sum_{j=1}^{H_I} {\tilde G^{(j)}}^\top \cdot Q^{I,(j)} \tag{10}
$$

$$
dw_{t,j} = \sum_s g_{t,s} \cdot \mathrm{ReLU}(z^{(j)}_{t,s}) \tag{11}
$$

因此，我们只需要 tiling 地去恢复出来 $\hat p_{t,s}$ 和 $\bar p_{t,s}$ 就可以了，比 Flash-Attention bwd 过程其实还要简单点。

## 3. FlashKL 算法

### 3.1 核心思路

> 先回忆一下 Flash Attention 是怎么做 tiling 的（可以参考文章Flash Attention原理详解(含代码讲解)[2]，讲解有很多）：它要算 $O_t = \sum_s \mathrm{softmax}_s(QK^\top) \cdot V_s$，也就是对 key 维度做 softmax 再对 $V_s$ 加权求和。它的做法是把 key 轴切成小 tile，每次只加载一个 tile 到 SRAM，维护三个统计量来逐 tile 更新：其中 max 和 sum-of-exp 来自 online softmax 算法，而加权累积值（$\sum p \cdot V$）利用同样的 rescale 系数 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正之前的累积，不需要把整个 $L \times L$ 矩阵存到 HBM 上。

我们这里要算的东西结构几乎一样：

- 加权点积 $\sum_s p^h_{t,s}\, I_{t,s}$：跟 Flash Attention 的 $\sum_s p_{t,s}\, V_s$ 一回事，都是只和当前 $t, s$ 有关的量。
- logsumexp $\mathrm{LSE}_s(I_{t,:})$：天然支持 online 累积。

以下是详细的 tiling 算法过程，如果对 FA 很熟悉了，可以跳过。

### 3.2 定义统计量

先把第一项重写成 per-head 形式：

$$
\sum_s \bar{p}_{t,s}\, I_{t,s} = \frac{1}{H}\sum_{h=1}^{H} \frac{\sum_s e^{\alpha^{(h)}_{t,s}} I_{t,s}}{\sum_s e^{\alpha^{(h)}_{t,s}}}
$$

定义两个 per-head 累加量：

$$
N^{(h)}_t = \sum_s e^{\alpha^{(h)}_{t,s}} I_{t,s} \quad, \qquad D^{(h)}_t = \sum_s e^{\alpha^{(h)}_{t,s}} \tag{12}
$$

那么：

$$
\sum_s \bar{p}_{t,s}\, I_{t,s} = \frac{1}{H}\sum_h \frac{N^{(h)}_t}{D^{(h)}_t}
$$

$N^{(h)}_t$ 和 $D^{(h)}_t$ 都是对 $s$ 的求和，可以**逐 block 累加**。但 $e^{\alpha}$ 会数值溢出，因此需要维护一个 running max 来保证数值稳定性。

定义当前已扫描 block 的 max：

$$
M^{(h)}_t = \max_{s \in \text{seen}} \alpha^{(h)}_{t,s}, \qquad m_t = \max_{s \in \text{seen}} I_{t,s}
$$

把累加量改成相对于当前 max 的 shifted form：

$$
\widetilde N^{(h)}_t = \sum_{s \in \text{seen}} e^{\alpha^{(h)}_{t,s} - M^{(h)}_t} \cdot I_{t,s}, \qquad \widetilde D^{(h)}_t = \sum_{s \in \text{seen}} e^{\alpha^{(h)}_{t,s} - M^{(h)}_t}, \qquad \widetilde \ell_t = \sum_{s \in \text{seen}} e^{I_{t,s} - m_t} \tag{13}
$$

比值 $\widetilde N / \widetilde D$ 等于真实的 $N / D$，因为 $e^{-M}$ 上下抵消：

$$
\frac{\widetilde N^{(h)}_t}{\widetilde D^{(h)}_t} = \frac{\sum_s e^{\alpha - M} I}{\sum_s e^{\alpha - M}} = \frac{e^{-M}\sum_s e^{\alpha} I}{e^{-M}\sum_s e^{\alpha}} = \frac{N^{(h)}_t}{D^{(h)}_t}
$$

同理 student 端的 logsumexp，也可以采用最大值来防止数值溢出：$\log\sum_s e^{I_{t,s}} = m_t + \log\widetilde\ell_t$。

### 3.3 更新公式

有了上面的统计量，下面推导逐 block 的更新规则。设已经处理完 key 索引 $s \in [0, S)$，现在要把 key block $[s, s+B_k)$ 加入累加。

#### Teacher 端

算 block 内 row max，并决定新 max：

$$
M^{(h)}_{\text{blk}} = \max_{s \in \text{blk}} \alpha^{(h)}_{t,s}, \qquad M^{(h)}_{\text{new}} = \max\!\big(M^{(h)}_{\text{old}},\; M^{(h)}_{\text{blk}}\big)
$$

重缩放因子：

$$
\rho^{(h)} = e^{M^{(h)}_{\text{old}} - M^{(h)}_{\text{new}}}
$$

累加器更新——老的累积量乘 rescale 系数统一到新 max 下，再加上 block 内的贡献：

$$
\boxed{\;\widetilde D^{(h)}_{\text{new}} = \rho^{(h)} \cdot \widetilde D^{(h)}_{\text{old}} + \sum_{s \in \text{blk}} e^{\alpha^{(h)}_{t,s} - M^{(h)}_{\text{new}}}\;} \tag{14}
$$

$$
\boxed{\;\widetilde N^{(h)}_{\text{new}} = \rho^{(h)} \cdot \widetilde N^{(h)}_{\text{old}} + \sum_{s \in \text{blk}} e^{\alpha^{(h)}_{t,s} - M^{(h)}_{\text{new}}} \cdot I_{t,s}\;} \tag{15}
$$

#### Indexer 端

$$
m_{\text{blk}} = \max_{s \in \text{blk}} I_{t,s}, \quad m_{\text{new}} = \max(m_{\text{old}}, m_{\text{blk}}), \quad \beta = e^{m_{\text{old}} - m_{\text{new}}}
$$

$$
\boxed{\;\widetilde \ell_{\text{new}} = \beta \cdot \widetilde \ell_{\text{old}} + \sum_{s \in \text{blk}} e^{I_{t,s} - m_{\text{new}}}\;} \tag{16}
$$

#### Final reduction

扫完所有 key block 后，汇总得到每个 query 位置的 loss：

$$
\boxed{\;\mathcal{L}^I_t = -\frac{1}{H}\sum_h \frac{\widetilde N^{(h)}_t}{\widetilde D^{(h)}_t} + \Bigl( m_t + \log \widetilde \ell_t \Bigr)\;} \tag{17}
$$

有了上面的更新公式，可以直接写出逐 tile 扫描的 forward 和 backward 伪代码。

### 3.4 Forward Pass 伪代码

```
# 算法：FlashKL Forward（Tiled）# 输入：Q_tea[H,L,d], K_tea[H,L,d], Q_idx[H_I,L,d_I], K_idx[L,d_I], W[L,H_I]# 输出：loss# 初始化M[h,t] = -inf,  D_tilde[h,t] = 0,  N_tilde[h,t] = 0   m[t] = -inf,  ell_tilde[t] = 0for j in1..ceil(L/B):                           # 逐 tile 扫描 key 轴    K_tea_tile = K_tea[:, j-th tile, :]          # [H, B, d]    K_idx_tile = K_idx[j-th tile, :]             # [B, d_I]    for t in1..L:                               # 实际按 query tile 并行        # Indexer 端        I_tile = 0                                        # [B]        for ji in1..H_I:            z_tile[ji] = K_idx_tile @ Q_idx[ji,t]         # [B]            I_tile    += W[t,ji] * ReLU(z_tile[ji])        m_new      = max(m[t], max(I_tile))        beta       = exp(m[t] - m_new)        ell_tilde[t] = ell_tilde[t] * beta + sum(exp(I_tile - m_new))        m[t]       = m_new        # Teacher 端        for h in1..H:            alpha_tile = K_tea_tile[h] @ Q_tea[h,t] / sqrt(d) # [B]            M_new      = max(M[h,t], max(alpha_tile))            rho        = exp(M[h,t] - M_new)            e_tea      = exp(alpha_tile - M_new)               # [B]            D_tilde[h,t] = D_tilde[h,t] * rho + sum(e_tea)            N_tilde[h,t] = N_tilde[h,t] * rho + e_tea @ I_tile            M[h,t]    = M_new# 汇总loss = sum_t( -1/H * sum_h(N_tilde[h,t] / D_tilde[h,t]) + m[t] + log(ell_tilde[t]) )save M, D_tilde, m, ell_tildereturn loss
```

### 3.5 Backward Pass 伪代码

> Backward 的核心思想跟 Flash Attention 一样：不存 $p$ 和 $\hat p$，用 forward 保存的统计量逐 tile 重算。

```
# 算法：FlashKL Backward（Tiled, Recomputation）# 输入：Q_tea[H,L,d], K_tea[H,L,d]#       Q_idx[H_I,L,d_I], K_idx[L,d_I], W[L,H_I]#       M[H,L], D_tilde[H,L], m[L], ell_tilde[L]   -- forward 保存的统计量# 输出：dQ_idx[H_I,L,d_I], dK_idx[L,d_I], dW[L,H_I]# 初始化dQ_idx = 0,  dK_idx = 0,  dW = 0for j in1..ceil(L/B):                              # 逐 tile 扫描 key 轴    K_tea_tile = K_tea[:, j-th tile, :]              # [H, B, d]    K_idx_tile = K_idx[j-th tile, :]                 # [B, d_I]    dK_idx_tile = 0                                  # [B, d_I]    for t in1..L:                                   # 实际按 query tile 并行        # Indexer 端        I_tile = 0                                              # [B]        for ji in1..H_I:            z_tile[ji] = K_idx_tile @ Q_idx[ji,t]               # [B]            I_tile    += W[t,ji] * ReLU(z_tile[ji])        p_hat_tile = exp(I_tile - m[t]) / ell_tilde[t]          # [B]        # Teacher 端        p_bar_tile = 0                                           # [B]        for h in1..H:            alpha_tile  = K_tea_tile[h] @ Q_tea[h,t] / sqrt(d)  # [B]            p_bar_tile += exp(alpha_tile - M[h,t]) / D_tilde[h,t]        p_bar_tile /= H        # 梯度：g = p_hat - p_bar        g_tile = p_hat_tile - p_bar_tile                         # [B]        # 累积参数梯度        for ji in1..H_I:            mask         = (z_tile[ji] > 0)                      # [B]            g_tilde      = g_tile * W[t,ji] * mask               # [B]            dQ_idx[ji,t] += K_idx_tile.T @ g_tilde               # [d_I]            dK_idx_tile  += outer(g_tile, Q_idx[ji,t])          # [B, d_I]            dW[t,ji]     += g_tile @ ReLU(z_tile[ji])            # scalar    dK_idx[j-th tile] += dK_idx_tilereturn dQ_idx, dK_idx, dW
```

## 4. 为什么 teacher entropy 不可以 one-pass 计算呢？是否可以 tiling 计算呢？

事实上如果是单头情况（$H = 1$）这个是可以实现的，但是当多头的时候，采用 tiling 方式，必须要 two-pass。

### 4.1 单头情况（$H = 1$）

> 当 $H=1$ 时 $\bar{p} = p^{(1)}$，teacher entropy 可以利用 $\log \mathrm{softmax} = \alpha - \mathrm{LSE}(\alpha)$ 展开：

$$
\sum_s p^{(1)}_{t,s} \log p^{(1)}_{t,s} = \sum_s p^{(1)}_{t,s} \left( \alpha^{(1)}_{t,s} - \mathrm{LSE}(\alpha^{(1)}_{t,:}) \right) = \underbrace{\sum_s p^{(1)}_{t,s}\, \alpha^{(1)}_{t,s}}_{\text{softmax 加权求和}} - \underbrace{\mathrm{LSE}(\alpha^{(1)}_{t,:})}_{\text{logsumexp}}
$$

第一项跟 Flash Attention 算 $\sum_s p_{t,s} V_s$ 完全一样（只是把 $V_s$ 换成标量 $\alpha^{(1)}_{t,s}$），逐 tile 累积即可；第二项（LSE）本身就是 online softmax 统计量的副产品。所以一趟扫过就够了。

### 4.2 多头情况（$H > 1$）

> 多头 teacher entropy 为：

$$
\sum_s \bar{p}_{t,s} \log \bar{p}_{t,s} = \sum_s \left(\frac{1}{H}\sum_h p^{(h)}_{t,s}\right) \log\!\left(\frac{1}{H}\sum_h p^{(h)}_{t,s}\right)
$$

$\log$ 套着跨 head 的 sum，这就没办法像单头情况一样，可以把公式展开，那么在逐 tile 扫描时，每个 $p^{(h)}_{t,s}$ 会随着 tile 的到来不断更新。Flash Attention 的 tiling 技巧能处理线性累积（如 $\sum p \cdot V$，max 更新时只需对之前的累积值乘 $e^{m_{\text{old}} - m_{\text{new}}}$ 即可修正），但 $f(x) = x \log x$ 是非线性的，rescale 时之前累积的 $\sum \bar{p} \log \bar{p}$ 无法简单地乘一个系数修正。但是如果硬要用 tile 的方式去实现，也可以扫两遍：

> 第一次：逐 tile 扫描 key 轴，对每个 teacher head 维护 online softmax 的归一化统计量：

$$
m^h_t = \max_s \alpha^{(h)}_{t,s}\;(\text{running max}), \quad \ell^h_t = \sum_{s} \exp\!\left(\alpha^{(h)}_{t,s} - m^h_t\right)\;(\text{running sum-of-exp})
$$

这一趟结束后，$(m^h_t, \ell^h_t)$ 收敛到最终值。

> 第二次重算 logits ：

$$
\text{logit}\; \alpha^{(h)}_{t,s} = q^{(h)\top}_t k^{(h)}_s / \sqrt{d} \quad\longrightarrow\quad A^{(h)}_{t,s} = \frac{\exp(\alpha^{(h)}_{t,s} - m^h_t)}{\ell^h_t} \quad\longrightarrow\quad \bar{p}_{t,s} = \frac{1}{H}\sum_h A^{(h)}_{t,s}
$$

然后累积：$\text{entropy}_t \mathrel{+}= \bar{p}_{t,s} \log \bar{p}_{t,s}$。

::: block-1
总之，即便保留 teacher entropy 也可以通过 tiling 计算，但多头情况下需要两趟扫描。既然这一项本身不影响梯度，用更快更省的 one-pass 方式显然是更好的选择。
:::

## 小结

> FlashKL 通过将 KL loss 展开，在 teacher model 参数冻结的前提下去掉与优化无关的 teacher entropy 常数项，使得剩余的每一项都天然适配 Flash Attention 的 online softmax tiling 框架，显存从 $O(L^2)$ 降到 $O(L)$，one-pass 完成 fwd/bwd，长序列训练不再受 KL loss 的显存瓶颈制约。

文中的伪代码只作为算法示意，至于将算法落地为高性能 fused kernel（流水线调度、shared memory 优化等），还需要更专业的 infra 工程，期待和社区多多交流进步。

PS：本文部分公式推导和代码与 Claude Code 共同完成，感谢 cc；感谢伟大师兄孟繁续[3] 的讨论和帮助。

*随手记录，如有疑问和不足，欢迎交流 :)*

![文章插图](/articles/2026-06-17-flashkl-flash-attention-one-pass-42cdc4c8/images/05.png)

- 本文作者：Eleven
- 作者主页：https://www.zhihu.com/people/ru-guo-yan-jing-hui-shuo-hua-de-hua

## References

1. DeepSeek-DSA[4] .
2. Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness[5] .
3. TileLang-DSA: TileLang implementation of DSA KL loss[6] .

参考资料

[1]

tilelang-dsa: *https://github.com/lemyx/tilelang-dsa*

[2]

Flash Attention原理详解(含代码讲解): *https://zhuanlan.zhihu.com/p/676655352*

[3]

孟繁续: *https://www.zhihu.com/people/meng-fan-xu-4*

[4]

DeepSeek-DSA: *https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf*

[5]

Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness: *https://arxiv.org/abs/2205.14135*

[6]

TileLang-DSA: TileLang implementation of DSA KL loss: *https://github.com/lemyx/tilelang-dsa*

::: related-title
**相关推荐**
:::

::: related-links
- **[消除推荐系统广播税，从内存带宽到计算瓶颈的蜕变，揭秘 621 TFLOPS 的 Flash Attention](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900976&idx=1&sn=0f29d856ded3ed90bc6c1c4ce3152d0f&scene=21#wechat_redirect)**
- **[碾压GH200注意力实现：FlatAttention在Tile数据流加速器上，计算场景利用率达86%，访存场景带宽利用率达 78%，注意力内核平均加速 1.9倍](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899603&idx=1&sn=59da0ff7b8208a308553b463fc36a817&scene=21#wechat_redirect)**
- **[质疑“自动生成 FlashAttention”：浅析 Luminal 编译器的设计与现实落差](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900430&idx=1&sn=e1847048208639c245dac0a1135d1700&scene=21#wechat_redirect)**

:::
