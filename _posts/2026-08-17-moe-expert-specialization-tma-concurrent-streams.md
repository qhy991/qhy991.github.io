---
layout: post
title: "一个 MoE Kernel 为什么不够用：从 Expert Specialization、TMA Tile 到多 Stream 并发"
date: 2026-08-17 00:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [MoE, CUDA, CUTLASS, GPU Kernel, Hopper, TMA, Performance]
reading_time: 35
cover_image: /assets/blog-moe-expert-specialization.png
excerpt: "从 Per-Expert Kernel 选择、Problem Size Mask 与真实 TMA 数据量出发，解释为什么 MoE 优化最终还需要多 Stream 并发来回收 Persistent Kernel 的尾部资源。"
---

> 本文基于 Anonymous 发布的 Expert Specialization 优化实践进行重新组织和教学化解读。核心方案、Kernel 配置和实验现象来自原文；部分公式推导、图示和类比用于帮助理解。公开转载时请保留原作者与出处。

很多人第一次看到这项工作时，可能会以为它只是在调整 CUTLASS 的 `TileShape` 和 `ClusterShape`。

实际上，这项优化包含三个层层递进的问题：

1. **选择问题**：一个统一的 MoE Kernel，无法同时适配不同 Expert 的计算规模；
2. **数据搬运问题**：即使知道某个 Expert 是 Memory Bound，也不能只凭“Memory Kernel”这个名字选择配置，还要计算真实的 TMA 数据量；
3. **GPU 调度问题**：把一个大 Kernel 拆成三个专用 Kernel 后，单个任务池变小，Persistent Kernel 的尾部负载不均衡反而会变严重。

最终形成的完整方案是：

$$
\boxed{
\begin{gathered}
\text{Per-Expert Kernel Specialization}
\\+
\text{Problem Size Mask}
\\+
\text{Concurrent Streams}
\end{gathered}
}
$$

直观地说就是：

> 根据每个 Expert 实际收到的 Token 数量，给它选择合适的 GEMM Kernel；避免 CPU 回读动态任务数；最后让多个专用 Kernel 并发执行，回收闲置的 SM。

---

## 1. 先理解 MoE：它不是一个大 GEMM，而是一组大小不同的小 GEMM

### 1.1 什么是 Expert

MoE，也就是 Mixture of Experts，会在模型的 FFN 层中放置很多个 Expert。

每个 Expert 本质上都包含自己的一组权重。对于每个输入 Token，Router 会选择其中 Top-K 个 Expert 执行计算。

例如：

* 原始输入有 256 个 Token；
* Top-K 等于 8；
* 那么会产生：

$$
256\times 8=2048
$$

个 Token-Expert 分配关系。

这里的 2048 并不是 2048 个互不相关的新 Token，而是 256 个 Token 分别被发送给 8 个 Expert。

假设模型有 256 个 Expert，那么每个 Expert 最终收到的 Token 数量可能完全不同：

$$
M_0,M_1,M_2,\ldots,M_{255}
$$

例如：

```text
Expert 0:   2 tokens
Expert 1:  11 tokens
Expert 2:   0 tokens
Expert 3:  67 tokens
...
Expert 255: 5 tokens
```

每个 Expert 都需要执行自己的矩阵乘法：

$$
C_e[M_e,N]=A_e[M_e,K]B_e[K,N]
$$

其中：

* $e$ 表示 Expert 编号；
* $M_e$ 表示该 Expert 收到的 Token 数；
* $K$ 是输入通道维度；
* $N$ 是输出通道维度。

通常情况下，不同 Expert 的 $N$ 和 $K$ 相同，主要变化的是 $M_e$。

![同一批 Routed Token 在不同 Expert 上形成完全不同的 M 维度](/assets/blog-moe-token-skew.svg)

*图 1：全局 Routed Token 数只说明总工作量；真正决定单个 GEMM 形状的是每个 Expert 的 $M_e$。*

---

### 1.2 为什么使用 Group GEMM

如果有 256 个 Expert，最直接的做法是启动 256 个独立 GEMM Kernel。

但这样会产生大量 Kernel Launch 开销，也很难充分利用 GPU。

因此，CUTLASS MoE 通常使用 Group GEMM：

```text
Group 0:   [M0, K] × [K, N]
Group 1:   [M1, K] × [K, N]
Group 2:   [M2, K] × [K, N]
...
Group 255: [M255, K] × [K, N]
```

Group GEMM 把这些相互独立、形状不完全相同的 GEMM 放到一个 Persistent Kernel 中统一调度。

所以，理解这项优化的第一个关键点是：

> MoE GEMM 不是一个很大的规则矩阵乘法，而是很多个 $M_e$ 不同的小矩阵乘法。

---

## 2. 原始 CUTLASS MoE 的问题：用总 Token 数选择一个全局 Kernel

原始 CUTLASS MoE 中存在类似下面的判断：

```cpp
if (a.size(0) <= 2048) {
    // 使用适合小 M 的 Swap A/B Kernel
} else {
    // 使用更加通用的 Kernel
}
```

当总 Routed Token 数不超过 2048 时，选择适合小 $M$ 的 Kernel；超过 2048 时，选择更加通用、更加偏向充分利用计算单元的 Kernel。

这个策略的问题在于：

$$
\text{Kernel选择依据}
=
\sum_e M_e
$$

但真正决定某个 Expert 应该使用什么 Kernel 的是：

$$
\text{Expert }e\text{ 的计算规模}
=
M_e
$$

总量和局部形状并不是同一件事。

---

### 2.1 第一种失效：总 Token 很多，但每个 Expert 都很小

以 DeepSeek R1 为例：

* Top-K 为 8；
* Expert 数量为 256；
* 原始 Token 数为 256。

Routed Token 总数为：

$$
256\times8=2048
$$

如果原始 Token 数再增加一点，CUTLASS 就会切换到通用 Kernel。

但即使这些 Routed Token 完全均匀地分配到 256 个 Expert，每个 Expert 平均也只有：

$$
\frac{2048}{256}=8
$$

个 Token。

也就是说，CUTLASS 看到的是：

```text
总任务数很大：超过 2048
```

但单个 Expert 看到的实际上是：

```text
我的 GEMM 只有 M≈8
```

对一个 $M=8$ 的 GEMM 使用面向大 $M$ 的通用 Kernel，往往会产生大量 Padding 和无效计算。

---

### 2.2 第二种失效：总 Token 很少，但少数 Expert 很大

Token 路由通常并不均匀。

假设总 Routed Token 数没有超过 2048，因此系统选择了小 $M$ Kernel。但是大部分 Token 都集中到了几个热门 Expert：

```text
Expert 0:  120 tokens
Expert 1:   96 tokens
Expert 2:   83 tokens
其余 Expert: 0～5 tokens
```

此时，对于大部分 Expert，小 $M$ Kernel 是合适的；但对于 $M=120$ 的热门 Expert，小 $M$ Kernel 可能无法充分发挥硬件计算能力。

因此，一个全局判断会同时产生两类错误：

| 情况               | 全局判断           | 单个 Expert 的真实情况 |
| ---------------- | -------------- | --------------- |
| 总 Token 很多，但分布均匀 | 使用大 $M$ Kernel | 每个 Expert 仍然很小  |
| 总 Token 不多，但分布倾斜 | 使用小 $M$ Kernel | 少数 Expert 已经很大  |

原文将这一问题概括为：单一 Kernel 无法同时满足不同 Expert 的计算需求。

---

## 3. Expert Specialization：让每个 Expert 自己选择 Kernel

解决思路非常直接：

> 不再为整轮 MoE 只选择一个 Kernel，而是准备多个配置，并根据每个 Expert 的 $M_e$ 单独选择。

例如准备三个 Kernel：

```text
Low-M Kernel
Middle-M Kernel
High-M Kernel
```

运行时根据每个 Expert 收到的 Token 数分类：

```cpp
if (M_e <= threshold_low) {
    Expert e -> Low-M Kernel
} else if (M_e <= threshold_high) {
    Expert e -> Middle-M Kernel
} else {
    Expert e -> High-M Kernel
}
```

逻辑流程可以表示为：

```text
原始 Token
    │
    ▼
Top-K Router
    │
    ▼
统计每个 Expert 的 Token 数 M_e
    │
    ├── 小 M ──> Low-M Kernel
    │
    ├── 中 M ──> Middle-M Kernel
    │
    └── 大 M ──> High-M Kernel
```

这就是 Expert Specialization。

![通过固定 group count 和零 Problem Size Mask 在 GPU 内完成 Expert Specialization](/assets/blog-moe-specialization-mask.svg)

*图 2：三个 Kernel 都保留 256 个 group；不属于当前配置的 Expert 以零尺寸跳过，因此无需把动态 bucket 数量回读 CPU。*

---

## 4. 工程难点：每个 Kernel 到底有多少个 Group

理论上，我们可以把 Expert 分成三个数组：

```text
low_experts    = [0, 2, 5, ...]
middle_experts = [1, 4, 9, ...]
high_experts   = [3, 7, ...]
```

然后分别启动三个 Group GEMM：

```cpp
launch_low(low_experts.size());
launch_middle(middle_experts.size());
launch_high(high_experts.size());
```

但这里有一个严重问题：

> 每个 Bucket 中有多少个 Expert，是 GPU 完成 Token 路由和统计之后才知道的。

如果 CPU 想获得这个数量，就需要：

1. GPU 计算每个 Bucket 的 Expert 数；
2. 将数量从 GPU 拷贝回 CPU；
3. CPU 等待结果；
4. CPU 再启动对应 Kernel。

这会引入：

* Device-to-Host memcpy；
* CPU/GPU 同步；
* Host 调度延迟；
* Python 或框架层额外开销。

对于微秒级 MoE Kernel，这种同步代价通常不可接受。

---

## 5. Problem Size Mask：保持 group_count 不变

作者采用了一个非常巧妙的办法：

> 三个 Kernel 的 `group_count` 都保持为 Expert 总数，例如 256；不属于某个 Kernel 的 Expert，在该 Kernel 中传入全零 Problem Size。

例如，Expert 3 应该由 High-M Kernel 处理：

```text
Low-M Kernel:
    Expert 3 problem = (0, 0, 0)

Middle-M Kernel:
    Expert 3 problem = (0, 0, 0)

High-M Kernel:
    Expert 3 problem = (M3, N, K)
```

伪代码如下：

```cpp
for (int e = 0; e < num_experts; ++e) {
    int m = tokens_per_expert[e];

    if (select_kernel(m) == LOW) {
        low_problem[e] = {m, n, k};
    } else {
        low_problem[e] = {0, 0, 0};
    }

    if (select_kernel(m) == MIDDLE) {
        middle_problem[e] = {m, n, k};
    } else {
        middle_problem[e] = {0, 0, 0};
    }

    if (select_kernel(m) == HIGH) {
        high_problem[e] = {m, n, k};
    } else {
        high_problem[e] = {0, 0, 0};
    }
}
```

随后三个 Kernel 都按照 256 个 Group 启动：

```cpp
launch(low_kernel,    group_count = 256);
launch(middle_kernel, group_count = 256);
launch(high_kernel,   group_count = 256);
```

每个 Expert 只在一个 Kernel 中拥有非零 Problem Size，其余两个 Kernel 会跳过这个 Expert。

这样就把问题从：

$$
\text{动态 group 数量}
$$

转换成：

$$
\text{固定 group 数量}+\text{动态 mask}
$$

从而避免了 D2H 拷贝和 CPU 同步。原作者将这套方案命名为 Expert Specialization。

---

## 6. 选择 Kernel 前，必须先理解 Compute Bound 和 Memory Bound

Expert Specialization 并不是简单地按照 $M_e$ 划分三段。

首先需要判断：某个 Expert 的 GEMM 到底受什么限制？

* 如果大部分时间在等待数据从显存搬进来，就是 **Memory Bound**；
* 如果数据供应已经足够快，主要时间花在矩阵乘法上，就是 **Compute Bound**。

这可以通过 Roofline Model 中的算术强度来分析。

---

### 6.1 GEMM 的算术强度

对于：

$$
C[M,N]=A[M,K]B[K,N]
$$

计算量为：

$$
F=2MNK
$$

文中考虑的是 FP8 Blockwise GEMM，可以近似认为：

* $A$ 为 FP8，每个元素 1 Byte；
* $B$ 为 FP8，每个元素 1 Byte；
* 输出 $C$ 为 FP16/BF16，每个元素 2 Byte。

忽略 Scale Factor、元数据和重复缓存流量后，理想数据量为：

$$
Q=MK+KN+2MN
$$

于是理想算术强度为：

$$
I(M,N,K)
=
\frac{2MNK}
{MK+KN+2MN}
$$

单位是：

$$
\mathrm{FLOP/Byte}
$$

---

### 6.2 什么是 Ridge Point

假设 GPU 的有效计算吞吐为 $P$，有效显存带宽为 $B$，那么硬件的 Ridge Point 可以写成：

$$
R=\frac{P}{B}
$$

如果：

$$
I<R
$$

说明每搬一个字节的数据，计算量还不够高，Kernel 更容易是 Memory Bound。

如果：

$$
I>R
$$

说明数据搬运不再是主要瓶颈，Kernel 更容易进入 Compute Bound。

原文在 H20 FP8 Blockwise 场景中使用了约：

$$
R=70\ \mathrm{FLOP/Byte}
$$

作为经验 Ridge Point。

这个值低于直接根据硬件规格计算的理论值，主要是因为：

* FP8 Blockwise 反量化还需要 CUDA Core 参与；
* Scale Factor 读取会增加额外数据流量；
* 实际可达到的计算吞吐低于纯 Tensor Core 峰值。

---

## 7. 为什么 Gate+Up 约 40 个 Token 才进入 Compute Bound

文中对 DeepSeek R1 TP=8 的 Gate+Up Layer 使用：

$$
M=40,\qquad N=512,\qquad K=7168
$$

代入公式：

$$
I=
\frac{2\times40\times512\times7168}
{40\times7168+7168\times512+2\times40\times512}
$$

得到：

$$
I\approx73.44\ \mathrm{FLOP/Byte}
$$

已经略高于经验 Ridge Point 70。

因此，Gate+Up Layer 大约需要达到 40 个 Token，才可能进入 Compute Bound。

可以把数据量拆开看：

| 数据     |                                 大小 |
| ------ | ---------------------------------: |
| 激活 $A$ |      $40\times7168=286{,}720$ Byte |
| 权重 $B$ | $7168\times512=3{,}670{,}016$ Byte |
| 输出 $C$ | $2\times40\times512=40{,}960$ Byte |

此时最大的部分是权重矩阵 $B$。

当 $M$ 很小时，大量时间都花在把权重搬进来，Tensor Core 并没有足够多的 Token 可以计算。

---

## 8. 为什么 Down Layer 需要约 50 个 Token

文中对 Down Layer 使用：

$$
M=50,\qquad N=7168,\qquad K=256
$$

于是：

$$
I=
\frac{2\times50\times7168\times256}
{50\times256+256\times7168+2\times50\times7168}
$$

得到：

$$
I\approx71.55\ \mathrm{FLOP/Byte}
$$

因此，Down Layer 大约需要 50 个 Token 才可能进入 Compute Bound。

这里的数据量为：

| 数据     |                                   大小 |
| ------ | -----------------------------------: |
| 激活 $A$ |          $50\times256=12{,}800$ Byte |
| 权重 $B$ |   $256\times7168=1{,}835{,}008$ Byte |
| 输出 $C$ | $2\times50\times7168=716{,}800$ Byte |

与 Gate+Up 不同，Down Layer 的输出写回非常大，因此需要更多 Token 才能达到相同的算术强度。

---

### 8.1 一个方便理解的近似

在小 $M$ 场景中，权重读取 $KN$ 往往占主导，因此：

$$
I
\approx
\frac{2MNK}{KN}
=
2M
$$

如果 Ridge Point 为 70，那么一个粗略估计是：

$$
M_{\mathrm{ridge}}
\approx
\frac{70}{2}
=
35
$$

Gate+Up 的实际阈值约为 40，与这个近似比较接近。

Down Layer 因为输出写回占比更高，所以阈值进一步上升到了约 50。

这也说明了为什么 $M_e$ 是非常自然的 Kernel 选择变量：

> Token 越多，同一份权重就能服务越多次计算，权重搬运成本被更多 Token 分摊，算术强度随之上升。

---

## 9. Hopper GEMM 中必须理解的几个概念

在继续看 Kernel 配置之前，需要先认识下面几个术语。

| 概念                   | 含义                                             |
| -------------------- | ---------------------------------------------- |
| SM                   | GPU 上执行 Thread Block 的计算单元                     |
| CTA                  | Cooperative Thread Array，也就是 CUDA Thread Block |
| Tile                 | 一个 CTA 或 Cluster 一次处理的矩阵分块                     |
| Shared Memory        | SM 内部的高速缓存，用于暂存 TMA 搬入的数据                      |
| TMA                  | Hopper 上负责异步搬运多维张量的数据传输单元                      |
| Thread Block Cluster | 由多个 CTA 组成的协作单元                                |
| Multicast            | 一次数据搬运同时发送给 Cluster 中多个 CTA                    |
| Mainloop             | 沿 K 维不断加载数据并执行矩阵乘法的主循环                         |

---

### 9.1 TileShape 是什么

例如：

```cpp
MmaTileShape = Shape<_256, _32, _128>;
```

表示主计算 Tile 的三个方向大致为：

$$
T_M=256,\qquad T_N=32,\qquad T_K=128
$$

可以将其理解为：Kernel 每次在矩阵的 $M,N,K$ 方向上处理多大的分块。

---

### 9.2 ClusterShape 是什么

例如：

```cpp
ClusterShape = Shape<_2, _1, _1>;
```

表示两个 CTA 沿第一个维度组成一个 Cluster。

Cluster 级别的 Tile 可以近似看作 TileShape 与 ClusterShape 逐元素相乘：

$$
\langle256,32,128\rangle
\odot
\langle2,1,1\rangle
=
\langle512,32,128\rangle
$$

也就是说，一个 Cluster 在一次计算中整体覆盖：

$$
512\times32\times128
$$

的逻辑分块。

Cluster 的价值之一，是让多个 CTA 共享某个输入操作数。

例如两个 CTA 计算不同的输出通道，但使用相同的一组 Token 激活，那么这份激活就可以通过 TMA Multicast 同时发送给两个 CTA，而不必重复从更高层缓存中完整读取。

---

### 9.3 TMA 为什么重要

Hopper 上的 Group GEMM 会使用 TMA 将数据从 Global Memory/L2 搬到 Shared Memory。

对于 Memory Bound Kernel，一个非常重要的经验原则是：

> 尽量使用更少的 TMA 指令，每次搬运更大的有效数据块。

因为每条 TMA 指令本身都有固定的发起、描述符处理和流水线同步开销。

同样搬运 128 KB 数据：

```text
方法 A：128 次 × 1 KB
方法 B：  8 次 × 16 KB
```

通常方法 B 更容易有效利用带宽。

但 Tile 也不能无限增大，因为过大的 Tile 会导致：

* Shared Memory 占用上升；
* 单个 SM 可同时驻留的 CTA 数量下降；
* Padding 和无效计算增加；
* Persistent Scheduler 的任务粒度变粗。

因此，Kernel 设计本质上是在做权衡。

---

### 9.4 Cooperative 和 Pingpong 是什么

Hopper CUTLASS GEMM 通常使用 Warp Specialization：

* Producer Warp Group 负责发起 TMA、搬运数据；
* Consumer Warp Group 负责执行 WGMMA。

两类常见调度方式是 Cooperative 和 Pingpong。

可以做一个直观理解：

#### Cooperative

多个 Consumer 更偏向协作处理较大的 Tile。

它通常允许使用更大的数据分块，因此有利于：

* 减少 TMA 指令数量；
* 提高单次搬运的数据量；
* 改善 Memory Bound 场景中的带宽利用率。

#### Pingpong

不同 Consumer Warp Group 交替处理计算阶段，以隐藏流水线延迟。

它通常更适合：

* K 维循环较长；
* 计算量充足；
* Tensor Core 是主要瓶颈的场景。

这里的重点不是认为某一种调度永远更好，而是：

> Cooperative 和 Pingpong 适合的瓶颈不同，必须结合 Problem Shape 和数据路径选择。

---

## 10. H20 上的 Low-M Kernel：为 Memory Bound 设计

H20 上的 Low-M 配置为：

```cpp
MmaTileShape = Shape<_256, _32, _128>;
ClusterShape = Shape<_2, _1, _1>;
Schedule     = Cooperative;
```

Cluster Tile 为：

$$
\langle512,32,128\rangle
$$

这个 Kernel 还有一个关键特征：Swap A/B。

---

### 10.1 什么是 Swap A/B

原始 GEMM 为：

$$
C[M,N]=A[M,K]B[K,N]
$$

其转置形式为：

$$
C^T[N,M]=B^T[N,K]A^T[K,M]
$$

从数学上看，两者表示的是同一项计算。

当 $M$ 很小时，如果直接使用一个较大的 $M$-Tile，例如 128，可能会出现：

```text
实际 M = 8
Tile M = 128
```

大量 MMA 位置都在处理无效 Padding。

Swap A/B 后，原来的小 Token 维 $M$ 被映射到 Tile 的较小方向，例如 32；而较大的输出通道方向被映射到 256。

于是逻辑上从：

```text
Token 方向使用很大的 Tile
```

变成：

```text
输出通道方向使用大 Tile
Token 方向只使用 32 的小 Tile
```

这样既能减少 Token 方向的无效计算，又能在权重方向上进行较大的 TMA 搬运。

---

### 10.2 `<2,1,1>` Cluster 如何复用数据

Swap A/B 后，两个 CTA 沿第一个维度组成 Cluster：

```text
CTA 0：处理一部分输出通道
CTA 1：处理另一部分输出通道
```

两个 CTA 加载不同的权重块，但会使用同一组 Token 激活。

因此，这组 Token 激活可以通过 TMA Multicast 在两个 CTA 之间共享。

原文给出的解释是：

* 单个 CTA 的 A Tile 为 $256\times128$；
* 两个 CTA 合计处理 $512\times128$ 的 A；
* 两个 CTA 需要相同的 $32\times128$ B Tile；
* B Tile 可以通过 TMA Multicast 提供给 Cluster 中的两个 CTA。

从原始 GEMM 的角度理解，就是：

> 同一批 Token 激活，同时服务两个不同的输出通道分块。

---

## 11. H20 上的 Middle-M 和 High-M Kernel

对于已经进入 Compute Bound 的 Expert，作者选择 Pingpong 调度，并通过不同的 ClusterShape 控制 Token 方向的 Padding。

### Middle-M

```cpp
MmaTileShape = Shape<_64, _128, _128>;
ClusterShape = Shape<_1, _2, _1>;
Schedule     = Pingpong;
```

Cluster Tile 为：

$$
\langle64,256,128\rangle
$$

两个 CTA 沿 $N$ 方向排列：

```text
相同的 Token Tile
不同的输出通道 Tile
```

因此可以复用激活 $A$。

它适合：

$$
M\le64
$$

因为 64 个以内的 Token 可以放入一个 Token Tile。

---

### High-M

```cpp
MmaTileShape = Shape<_64, _128, _128>;
ClusterShape = Shape<_2, _1, _1>;
Schedule     = Pingpong;
```

Cluster Tile 为：

$$
\langle128,128,128\rangle
$$

两个 CTA 沿 $M$ 方向排列：

```text
不同的 Token Tile
相同的权重 Tile
```

因此可以复用权重 $B$。

它适合：

$$
M>64
$$

特别是 $65\le M\le128$ 时，一个 Cluster Tile 就能覆盖这些 Token。

所以，H20 上的 Kernel 选择逻辑可以概括为：

| Expert 状态            | 核心问题         | Kernel 思路                     |
| -------------------- | ------------ | ----------------------------- |
| 小 $M$、Memory Bound   | 数据搬运占主导      | Swap A/B、Cooperative、大 TMA 分块 |
| 中等 $M$、Compute Bound | Token 不超过 64 | Pingpong，Cluster 沿 $N$ 方向     |
| 较大 $M$、Compute Bound | Token 超过 64  | Pingpong，Cluster 沿 $M$ 方向     |

原文实验表明，在 H20 的中小 Batch 场景中，同时替换 Gate+Up 和 Down Layer 后，Expert Specialization 明显优于原生 CUTLASS MoE。

---

## 12. 为什么 H100/H200/H800 上的策略不一样

原文将 H100、H200、H800 统称为 Hx00。

与 H20 相比，这些 GPU 的计算吞吐更高，因此有效 Ridge Point 更高。

假设：

$$
R=\frac{P_{\mathrm{compute}}}{B_{\mathrm{memory}}}
$$

当计算能力 $P_{\mathrm{compute}}$ 大幅上升，而显存带宽没有同比例上升时，$R$ 会增大。

结果是：

> 同一个 GEMM，在 H20 上可能已经进入 Compute Bound；在 H100/H200 上却仍然可能是 Memory Bound。

因此，在中小 Batch Size 下，Hx00 上绝大部分 Expert 的 GEMM 都是 Memory Bound。

作者最初为 Hx00 准备了三个配置：

| 配置     | TileShape       | ClusterShape | 初始定位                  |
| ------ | --------------- | ------------ | --------------------- |
| Low    | `<256,32,128>`  | `<2,1,1>`    | Memory Bound，$M\le32$ |
| Middle | `<256,64,128>`  | `<2,1,1>`    | Memory Bound，$M>32$   |
| High   | `<128,128,128>` | `<1,2,1>`    | Compute Bound         |

三个配置全部使用 Cooperative 调度。

---

## 13. 为什么 Hx00 上 Gate+Up 收益很小，而 Down 收益明显

这与 K 维大小密切相关。

### 13.1 Gate+Up：K 很大

Gate+Up 的典型参数中：

$$
K=7168
$$

如果每次 Mainloop 处理：

$$
T_K=128
$$

那么循环次数为：

$$
\frac{7168}{128}=56
$$

也就是说，一个 CTA 需要执行 56 轮：

```text
TMA Load
    ↓
Shared Memory
    ↓
WGMMA
    ↓
下一轮 TMA Load
```

56 轮足以让软件流水线进入稳定状态：

* Producer Warp Group 可以持续搬数据；
* Consumer Warp Group 可以持续计算；
* 启动和排空开销被大量循环摊薄；
* 显存带宽更容易保持在较高水平。

此外，TMA 在处理边界 Tile 时，并不会因为 TileShape 更大，就从 L2/HBM 读取同等规模的无效 Padding。

例如：

```text
配置 Tile 大小：32×128
实际有效数据：16×128
```

实际从 L2 或显存读取的仍然主要是有效的 $16\times128$ 数据，而不是完整的 $32\times128$。

因此，Gate+Up 的不同 Kernel 配置：

* 外部有效读取量相近；
* 流水线都比较充分；
* 带宽利用率都比较高。

最后性能差距只有个位数微秒，Expert Specialization 的收益有限，所以 Hx00 上没有替换 Gate+Up Layer。

---

### 13.2 Down：K 很小

Down Layer 的典型参数中：

$$
K=256
$$

Mainloop 次数只有：

$$
\frac{256}{128}=2
$$

只有两轮循环时：

* 软件流水线刚填起来就结束了；
* TMA 发起开销占比更高；
* Barrier 和流水线同步占比更高；
* 每个 Tile 到底搬多少数据变得非常重要；
* 权重是否被重复加载会直接影响性能。

因此，为 Memory Bound 场景设计的 Kernel 在 Down Layer 上更容易体现优势。

原文最终只在 Hx00 上将 Down Layer 替换为 Expert Specialization。

---

## 14. 最关键的反直觉现象：Memory Bound 为什么反而选择 High Config

作者在 Hx00 测试中发现：

> 当 Expert 的 Token 数超过 64，但仍然没有进入 Compute Bound 时，High Config 反而比 Middle Config 更快。

这看起来很矛盾：

```text
当前问题仍然 Memory Bound
Middle Config 专门为 Memory Bound 设计
High Config 原本面向 Compute Bound
```

为什么 High Config 反而更快？

原因是：

> “Memory Bound Kernel”只是一个设计标签，真正影响性能的是实际搬运了多少数据。

原文使用 DeepSeek R1 TP=8 的 Down Layer 进行分析：

$$
M=96,\qquad N=7168,\qquad K=256
$$

对应的 NCU 截图中：

| 指标                 | Middle Config | High Config |
| ------------------ | ------------: | ----------: |
| Kernel 时间          |   约 377.98 μs | 约 319.10 μs |
| TMA Load Sector    |       36.70 M |     29.36 M |
| Device Memory Read |   约 643.80 MB | 约 476.59 MB |

High Config 的运行时间下降约：

$$
1-\frac{319.10}{377.98}
\approx15.6%
$$

关键是理解两个配置如何划分 $M=96$。

---

## 15. Middle Config 如何处理 $M=96$

Middle Config 使用 Swap A/B。

交换后的 Problem Shape 为：

$$
\langle7168,96,256\rangle
$$

Cluster Tile 为：

$$
\langle512,64,128\rangle
$$

沿两个空间维度的 Cluster 数量为：

$$
\left\lceil\frac{7168}{512}\right\rceil
\times
\left\lceil\frac{96}{64}\right\rceil
=
14\times2 =
28
$$

关键是：

$$
\left\lceil\frac{96}{64}\right\rceil=2
$$

96 个 Token 被拆成了两个 Tile：

```text
第一个 Token Tile：64 tokens
第二个 Token Tile：32 tokens
```

这意味着，同一个输出通道范围对应的权重，需要为两个 Token Tile 分别提供一次：

```text
权重 Tile -> 前 64 个 Token
权重 Tile -> 后 32 个 Token
```

大权重矩阵因此出现重复供应。

---

## 16. High Config 如何处理 $M=96$

High Config 使用正常矩阵方向。

Problem Shape 为：

$$
\langle96,7168,256\rangle
$$

Cluster Tile 为：

$$
\langle128,256,128\rangle
$$

Cluster 数量为：

$$
\left\lceil\frac{96}{128}\right\rceil
\times
\left\lceil\frac{7168}{256}\right\rceil
=
1\times28 =
28
$$

因为：

$$
\left\lceil\frac{96}{128}\right\rceil=1
$$

所以 96 个 Token 可以一次性放进一个 128 行 Token Tile。

同一份权重只需要服务一个 Token Tile：

```text
权重 Tile -> 全部 96 个 Token
```

虽然 128 行 Tile 中存在 32 行计算位置没有有效 Token，但这些无效位置并不会等比例转化为 HBM Padding 读取。

因此，这里的权衡是：

#### Middle Config

* Token Tile 更小；
* Token Padding 更少；
* 但大权重被重复供应。

#### High Config

* Token Tile 更大；
* 可能多一些计算 Padding；
* 但避免了大权重重复搬运。

对于 Down Layer 来说，权重矩阵远大于单次 Token 激活，因此减少权重读取更加重要。

---

## 17. 用 TMA Sector 精确验证

一个 TMA Sector 为 32 Byte。

测试中：

```text
Expert 数量       = 256
每个 Expert Tile 数 = 28
K                 = 256
每个 K Tile       = 128
K 循环次数         = 2
TMA Multicast      = 2
```

### 17.1 Middle Config

每个 Cluster、每个 K Step 需要处理：

* 权重侧：$512\times128$；
* 激活侧：$64\times128$，Multicast 给两个 CTA。

因此：

$$
S_{\mathrm{middle}}
=
\frac{
256\times28\times
\left[
\left(
512\times128
+
2\times64\times128
\right)
\times2
\right]
}{32}
$$

得到：

$$
S_{\mathrm{middle}}
=
36{,}700{,}160
$$

---

### 17.2 High Config

每个 Cluster、每个 K Step 需要处理：

* 激活侧：$128\times128$，Multicast 给两个 CTA；
* 权重侧：$256\times128$。

因此：

$$
S_{\mathrm{high}}
=
\frac{
256\times28\times
\left[
\left(
2\times128\times128
+
256\times128
\right)
\times2
\right]
}{32}
$$

得到：

$$
S_{\mathrm{high}}
=
29{,}360{,}128
$$

这两个结果与 NCU Profile 中的 TMA Load Sector 完全一致。

---

### 17.3 数据量减少了多少

Middle Config：

$$
36{,}700{,}160\times32
=
1{,}174{,}405{,}120\ \mathrm{Byte}
$$

约为：

$$
1.17\ \mathrm{GB}
$$

High Config：

$$
29{,}360{,}128\times32
=
939{,}524{,}096\ \mathrm{Byte}
$$

约为：

$$
0.94\ \mathrm{GB}
$$

TMA 数据交付量下降：

$$
1-\frac{29.36}{36.70}=20%
$$

进一步拆解可以看到：

| 配置     | 权重侧 Sector | 激活侧 Sector | 总 Sector |
| ------ | ---------: | ---------: | -------: |
| Middle |    29.36 M |     7.34 M |  36.70 M |
| High   |    14.68 M |    14.68 M |  29.36 M |

High Config：

* 激活侧数据量增加；
* 权重侧数据量减半；
* 总数据量下降 20%。

因为 Down Layer 中权重远大于激活，所以这是一个明显有利的交换。

---

## 18. 为什么只看 Cache Hit Rate 会得出错误结论

从对应 NCU 截图可以看到：

| 指标                 |    Middle |      High |
| ------------------ | --------: | --------: |
| L1/TEX Hit Rate    |    71.90% |    67.97% |
| L2 Hit Rate        |    48.76% |    52.13% |
| Device Memory Read | 643.80 MB | 476.59 MB |

High Config 的 L1/TEX Hit Rate 反而更低，但性能更好。

原因是，命中率只是比例：

$$
\text{Hit Rate}
=
\frac{\text{Hit Requests}}
{\text{Total Requests}}
$$

真正影响时间的是绝对 Miss 数据量：

$$
\text{Miss Bytes}
=
\text{Total Bytes}
\times
(1-\text{Hit Rate})
$$

High Config 的总请求量已经大幅减少，因此即使命中率略低，实际 Miss Bytes 仍然可能更少。

分析 Cache 时，正确的观察顺序应该是：

1. 总 Request/Sector 数；
2. 实际 Byte 数；
3. Miss Sector 和 Device Memory Byte；
4. 最终 Kernel 时间；
5. 最后再看 Hit Rate。

而不是只比较两个百分比。

---

## 19. 三种“数据量”不能混为一谈

NCU 中可能同时看到：

* TMA/L1TEX 到 Shared Memory 的数据量；
* L2 Cache 传输量；
* Device Memory/HBM 读取量。

它们不是同一个概念：

$$
\text{HBM Traffic}
\neq
\text{L2 Traffic}
\neq
\text{TMA Destination Traffic}
$$

例如，TMA Multicast 可能将一份缓存数据交付给两个 CTA。

从 HBM 角度看，它可能只需要读取一次；但从 CTA/Shared Memory 交付角度看，两个接收端都会形成对应的数据传输统计。

因此：

> TMA Sector 数适合分析 Kernel 内部的数据交付和 Tile 策略，但不能直接当作唯一 HBM 读取量。

---

## 20. High Config 已接近有效输入读取下界

对于单个 Expert：

$$
M=96,\qquad N=7168,\qquad K=256
$$

如果激活和权重都是 FP8，忽略 Scale 和元数据，最低有效输入数据量为：

$$
Q_{\mathrm{input}}
=
MK+KN
$$

代入：

$$
Q_{\mathrm{input}}
=
96\times256
+
256\times7168
$$

得到：

$$
Q_{\mathrm{input}}
=
1{,}859{,}584\ \mathrm{Byte}
$$

256 个 Expert：

$$
256\times1{,}859{,}584
=
476{,}053{,}504\ \mathrm{Byte}
$$

约为：

$$
476.05\ \mathrm{MB}
$$

而 High Config 的 NCU 截图中，Device Memory Read 约为：

$$
476.59\ \mathrm{MB}
$$

两者非常接近。

这说明 High Config 基本接近：

> 每份有效 FP8 激活和权重从显存读取一次。

Middle Config 的 Device Memory Read 约为 643.80 MB，则明显存在更多重复或附加数据流量。

---

## 21. 为什么最终使用 64 作为分界

Middle Config 的 Token Tile 大小为 64：

$$
N_{\mathrm{token\ tile}}
=
\left\lceil\frac{M}{64}\right\rceil
$$

当 $M$ 从 64 增加到 65 时：

$$
\left\lceil\frac{64}{64}\right\rceil=1
$$

突然变成：

$$
\left\lceil\frac{65}{64}\right\rceil=2
$$

同一份大权重因此要开始服务两个 Token Tile。

而 High Config 的 Token Tile 大小为 128：

$$
\left\lceil\frac{M}{128}\right\rceil
$$

对于：

$$
65\le M\le128
$$

始终为 1。

所以 64 不是一个随意选择的经验数字，而是一个明显的 Tile 边界：

```text
M <= 64：
Middle 只需要一个 Token Tile

M > 64：
Middle 需要两个 Token Tile
High 仍然只需要一个 Token Tile
```

基于这个发现，作者修改了 Hx00 的 Kernel 选择策略：

> 不再显式区分 Memory Bound 和 Compute Bound；当 Token 数量大于 64 时，统一使用 High Config。

这是整项工作中非常重要的认识：

> Roofline 只能告诉我们第一层瓶颈，不能直接告诉我们哪个 TileShape 最快。

即使一个问题是 Memory Bound，最终也要比较：

* 权重重复加载次数；
* 激活重复加载次数；
* TMA Sector；
* Tile 边界；
* Multicast 方向；
* L2/HBM 实际数据量。

---

## 22. 一个新的问题：三个专用 Kernel 破坏了原来的大任务池

Expert Specialization 提高了单个 Expert 与 Kernel 的匹配程度。

但它也带来了一个新问题。

原来：

```text
全部 256 个 Expert
        ↓
一个 Group GEMM Kernel
```

现在：

```text
部分 Expert -> Low-M Kernel
部分 Expert -> Middle-M Kernel
部分 Expert -> High-M Kernel
```

原来的一个大工作队列被拆成了三个小工作队列。

如果某轮路由结果是：

```text
Low-M Experts:    220
Middle-M Experts:  31
High-M Experts:     5
```

那么 High-M Kernel 的工作量可能很少，很难填满整个 GPU。

原文将其归结为多 Kernel 方案带来的负载不均衡问题。

---

## 23. Persistent Group GEMM 为什么会出现尾部效应

Group GEMM 通常采用 Persistent Kernel。

它的基本工作方式是：

```text
启动固定数量的 Worker CTA
       │
       ▼
每个 CTA 从全局队列中领取一个 Tile
       │
       ▼
完成 Tile
       │
       ▼
继续领取下一个 Tile
```

当剩余任务变少时，可能出现：

```text
CTA 0：已经没有任务，退出
CTA 1：已经没有任务，退出
CTA 2：还有一个长任务
CTA 3：已经退出
...
```

此时，大量 SM 已经空闲，但 Kernel 仍然没有结束，因为少数 CTA 还在处理最后几个任务。

这就是 Persistent Kernel 的 Tail Effect。

Expert Specialization 会加剧这一问题，因为：

* 每个 Kernel 的 Tile 数量变少；
* Tile 数量不一定能整除 Worker 数；
* 不同 Expert 的 $M_e$ 不同，单个 Tile 执行时间也可能不同；
* 三个 Kernel 各自形成独立尾部。

---

## 24. 方案一：PDL

PDL 是 Programmatic Dependent Launch。

传统情况下，如果 Kernel B 依赖 Kernel A：

```text
Kernel A 完成
      ↓
Kernel B 才能开始
```

PDL 允许后继 Kernel 更早进入待调度状态。

当 Kernel A 的部分 CTA 完成、释放 SM 资源后，Kernel B 的 CTA 有机会在这些空闲资源上开始执行，而不一定要等到 Kernel A 的整个 Grid 完成。

但是 PDL 不能做到：

* 抢占仍在执行的 CTA；
* 突破寄存器限制；
* 突破 Shared Memory 限制；
* 强制两个资源占用很大的 CTA 同时驻留在同一个 SM。

因此，如果前序 Persistent Kernel 在大部分时间都占满全部 SM，PDL 主要能回收的仍然只是尾部释放出来的资源。

作者为 Hopper Group GEMM 补充了 CUTLASS PDL 支持，但实测提升只有几微秒，因此最终没有采用。

---

## 25. 方案二：Concurrent Streams

作者最终采用的是三个 CUDA Stream：

```text
Stream 0: Low-M Kernel
Stream 1: Middle-M Kernel
Stream 2: High-M Kernel
```

三个 Kernel 在完成上游依赖后并发提交给 GPU。

当某个 Kernel 的 CTA 提前完成并释放 SM 时，另一个 Stream 中等待调度的 CTA 可以立刻使用空闲 SM。

可以把它理解为：

```text
原来：
三个独立的小任务池，各自等待自己的尾部结束

并发后：
GPU 调度器可以在三个任务池之间选择可运行的 CTA
```

这并不一定意味着同一个 SM 同时驻留三个 Kernel 的 CTA。

如果某个 Kernel：

* Shared Memory 占用接近 200 KB；
* 每个 SM 只能驻留一个 Block；

那么 Overlap 更可能表现为：

```text
一部分 SM 执行 Low-M Kernel
一部分 SM 执行 Middle-M Kernel
一部分 SM 执行 High-M Kernel
```

或者：

```text
Low-M 的部分 CTA 完成
      ↓
释放出的 SM 立即执行 High-M CTA
```

因此，Concurrent Streams 的核心价值是：

> 回收不同 Persistent Kernel 的尾部空闲 SM。

![多 Stream 让其他专用 Kernel 的 CTA 回收 Persistent Kernel 尾部释放出的 SM](/assets/blog-moe-concurrent-tail.svg)

*图 3：示意图表达的是 SM-time 资源回收，不代表同一 SM 必然共驻留三个 Kernel，也不是实际 profiler 时间线。*

原文的 PyTorch Profiler/Perfetto 时间线也观察到了三个 Kernel 之间的重叠，并且该方案的效果优于 PDL。

---

## 26. 如何正确解读 Perfetto 中的 Overlap

Perfetto 中多个 Kernel 的时间条发生重叠，只能证明：

> 它们的 GPU 生命周期在时间上有交集。

它不能自动证明：

* 它们一定在同一个 SM 上共驻留；
* Tensor Core 计算发生了真正的指令级重叠；
* 并发版本一定比串行版本更快。

真正需要比较的是完整 Makespan：

$$
T_{\mathrm{sequential}}
$$

和：

$$
T_{\mathrm{concurrent}}
$$

同时检查：

* 总执行时间；
* SM Active；
* HBM Throughput；
* L2 流量；
* Tensor Core 利用率；
* 是否发生显存带宽竞争；
* 是否破坏权重 Cache Locality。

对于这项工作，Concurrent Streams 的主要目的不是让两个满负载 GEMM 在同一个 SM 上同时执行，而是让不同 Kernel 的长尾互相填补。

---

## 27. 把整个优化过程串成一条因果链

现在可以把整项工作整理成一条完整逻辑。

### 第一步：发现全局阈值失效

原始方案根据：

$$
\sum_e M_e
$$

选择一个 Kernel。

但 MoE 真正执行的是很多个：

$$
M_e\times K\times N
$$

形状不同的 GEMM。

因此，总 Token 数无法反映单个 Expert 的计算特性。

---

### 第二步：为每个 Expert 独立选择 Kernel

准备多个配置：

```text
Low-M
Middle-M
High-M
```

根据每个 Expert 的 $M_e$ 进行选择。

为了避免 D2H 和 CPU 同步，保持固定 `group_count`，用全零 Problem Size 作为 Mask。

---

### 第三步：用 Roofline 做第一层分类

计算：

$$
I=\frac{2MNK}{MK+KN+2MN}
$$

判断某个 Expert 大致属于：

```text
Memory Bound
或
Compute Bound
```

H20 上，Gate+Up 大约需要 40 个 Token，Down 大约需要 50 个 Token 才可能进入 Compute Bound。

---

### 第四步：不能停留在 Roofline

$M=96$ 的实验说明：

> 即使问题仍然是 Memory Bound，所谓的 High/Compute Config 也可能更快。

因为真正决定性能的是：

* Tile 如何切分；
* 权重是否重复加载；
* 激活是否重复加载；
* TMA Multicast 的方向；
* TMA Sector 总数；
* HBM 实际读取量。

Roofline 是瓶颈分类工具，不是最终 Kernel 选择器。

---

### 第五步：解决多 Kernel 带来的任务碎片

三个专用 Kernel 提高了局部效率，却把一个大任务池拆成了三个小任务池。

Persistent Kernel 的尾部负载不均衡因此变严重。

最终通过 Concurrent Streams，让 GPU 调度器在不同 Kernel 之间回收空闲 SM。

---

## 28. 几个容易产生的误解

### 误解一：Memory Bound 就一定使用“Memory Kernel”

不一定。

Memory Bound 只是说明数据搬运是主要瓶颈。

最终应该选择：

$$
\text{实际数据搬运最少}
$$

或者：

$$
\text{有效带宽利用率最高}
$$

的 Kernel。

High Config 在 $M=96$ 时更快，就是因为它减少了权重重复供应。

---

### 误解二：Tile 越大，显存读取量一定越大

不一定。

TMA 对边界数据具有处理能力。Tile 中超出 Problem Size 的区域，不一定会在 L2/HBM 上形成同等大小的 Padding 读取。

大 Tile 确实可能产生更多无效计算和 Shared Memory 占用，但不能简单认为：

$$
\text{Tile 面积}
=
\text{HBM 读取量}
$$

---

### 误解三：Cache Hit Rate 越高，Kernel 一定越快

不一定。

应当同时考虑总请求量：

```text
90% 命中率 × 1 GB 请求
```

可能比：

```text
70% 命中率 × 200 MB 请求
```

产生更多的 Miss 数据。

分析时要优先看绝对 Sector 和 Byte 数。

---

### 误解四：Perfetto 中时间条重叠，就代表获得了性能收益

不一定。

Overlap 可能伴随：

* HBM 竞争；
* L2 竞争；
* Tensor Core 竞争；
* 调度等待；
* 更高的尾部延迟。

必须比较完整 Makespan。

---

### 误解五：Expert Specialization 是一个 Megakernel

不是。

这里使用的是：

$$
\text{多个独立专用 Kernel}
+
\text{多个 CUDA Stream}
$$

而不是在一个 Kernel 中动态选择三种 TileShape。

`TileShape`、`ClusterShape`、Shared Memory 大小、Warp 调度方式和 TMA 描述符大多是编译期或 Launch 期确定的。

如果强行放进一个统一 Kernel，往往需要：

* 按最大 Shared Memory 配置预留资源；
* 引入复杂的动态分支；
* 使用统一而非专用的资源配置；
* 自己实现异构 Device Scheduler。

这可能反而失去专用 Kernel 的性能优势。

---

## 29. 这项工作真正告诉我们的是什么

Expert Specialization 表面上是在选择三个 CUTLASS 配置，实际上它揭示了 GPU Kernel 优化中的三个层次。

### 第一层：问题分布

不能只看总 Token 数，还要看每个 Expert 的：

$$
M_e
$$

以及整个路由分布是否倾斜。

---

### 第二层：数据路径

不能只看 FLOPs 和理论算术强度，还要分析：

* TMA Load Sector；
* 权重重复供应；
* 激活重复供应；
* Multicast；
* L1/L2/HBM 数据量；
* Tile 边界和 Padding。

---

### 第三层：全局调度

单个 Expert 的最优 Kernel，不一定直接组合成全局最优系统。

当任务被拆分后，还需要考虑：

* 每个 Bucket 的任务数；
* Persistent Worker 数；
* Tile 是否能整除 Worker；
* 尾部 SM 空闲；
* 多 Stream 并发；
* 最终端到端 Makespan。

因此，一个更加完整的 Kernel 选择目标不是：

$$
\text{这个 Expert 应该使用哪个 Kernel}
$$

而是：

$$
\boxed{
\text{如何为全部 Expert 选择 Kernel，并使整轮 MoE 的完成时间最短}
}
$$

---

## 30. 一个更完整的成本模型

理想情况下，针对配置 $c$ 和 Expert $e$，可以估计：

$$
\hat T(c,e)
\approx
\max
\left(
\frac{
F_{\mathrm{useful}}+
F_{\mathrm{padding}}
}{
P_c
},
\frac{
Q_{\mathrm{TMA}}
}{
B_{\mathrm{TMA},c}
},
\frac{
Q_{\mathrm{HBM}}
}{
B_{\mathrm{HBM}}
}
\right)
+
T_{\mathrm{schedule}}
$$

其中：

* $F_{\mathrm{useful}}$：有效计算量；
* $F_{\mathrm{padding}}$：Tile Padding 产生的无效计算；
* $Q_{\mathrm{TMA}}$：TMA 向 Shared Memory 交付的数据量；
* $Q_{\mathrm{HBM}}$：实际显存读取量；
* $T_{\mathrm{schedule}}$：Cluster 调度、Barrier 和流水线开销。

但整轮 MoE 的时间并不是所有 Expert 时间简单相加，而更加接近：

$$
T_{\mathrm{MoE}}
=
\max_{\text{GPU Worker}}
\sum_{\text{该 Worker 上的 Tile}}
T_{\mathrm{tile}}
$$

所以 Kernel 选择和 GPU 负载均衡实际上是耦合问题。

---

## 31. 后续仍然可以改进的方向

原作者也指出，目前的方案仍然有进一步优化空间：

1. 对极小 Token 数场景，`cp.async` 可能比 TMA 更合适；CUTLASS 中已有相关实现，但原文所述实现当时仅支持 SM100。
2. TileShape 和 ClusterShape 仍然需要根据具体模型、TP 配置和矩阵形状重新调整。例如 Qwen3 TP=4 中，非 Token 方向的 Cluster Tile 也可能引入无效计算。

此外，还可以继续研究：

* 是否动态生成 Problem Size Bucket；
* 是否根据实际 TMA Sector 而不是固定 $M$ 阈值选择 Kernel；
* 是否将 Kernel 选择和任务分配联合优化；
* 是否在运行时根据路由分布选择串行、并发或合并执行；
* 是否为不同模型自动搜索 TileShape 与 ClusterShape；
* 是否使用在线性能模型替代手工阈值。

---

## 结语

这项工作的核心并不是“把三个 Kernel 放到三个 Stream 中”这么简单。

它完整展示了一个 GPU 优化问题如何逐层展开：

```text
总 Token 阈值不准确
        ↓
需要 Per-Expert Kernel 选择
        ↓
需要避免 D2H 动态 group_count
        ↓
使用 Problem Size Mask
        ↓
需要区分 Compute Bound / Memory Bound
        ↓
进一步发现 Roofline 分类仍然不够
        ↓
必须分析 Tile 边界和真实 TMA 流量
        ↓
多个专用 Kernel 又产生任务碎片
        ↓
使用 Concurrent Streams 回收尾部 SM
```

最终可以用一句话概括：

$$
\boxed{
\begin{aligned}
&
\text{根据每个 Expert 的计算形状选择专用 Kernel，}
\\&
\text{根据真实数据搬运而非标签判断配置，}
\\&
\text{再通过并发调度回收 Persistent Kernel 的尾部资源。}
\end{aligned}
}
$$

这也是 Expert Specialization 最值得借鉴的地方：

> GPU Kernel 优化不能只看一个算子有多少 FLOPs，也不能只看它是 Compute Bound 还是 Memory Bound；真正的性能来自问题分布、Tile 数据路径和全局调度三者的共同作用。
