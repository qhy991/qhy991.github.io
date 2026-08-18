---
layout: post
title: "寄存器越少越快吗？为什么 RMSNorm 40→32 大赢，Decode 56→80 也大赢"
date: 2026-08-17 10:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [CUDA, Registers, Occupancy, ILP, ptxas, FlashInfer, B200]
reading_time: 28
cover_image: /assets/blog-register-cliffs.png
excerpt: "同一张 B200 上，RMSNorm 从 40 降到 32 registers 后快 29.77%，single decode 却从 56 增到 80 registers 后快 6.56%。本文从 CTA 资源悬崖、occupancy、ILP、spill、launch_bounds 与 shape-specific dispatch 解释为什么寄存器数量不是单调性能旋钮。"
---

> 本文使用 SubCUDA 当前 machine-readable cases：FlashInfer RMSNorm toolchain cliff、SiLU r32、single-decode r80、Qwen activation register-budget stop 与 GDN state-retime negative。相关 replay 已在当前 checkout 通过；这些都是 B200 exact-shape operator evidence，不自动成为模型或 serving 收益。

CUDA 优化里最流行的建议之一是：

> “寄存器太多会降低 occupancy，所以应该尽量减少寄存器。”

这个建议只说对了一半。

同一张 B200 上，有两组看似矛盾的结果：

### RMSNorm

```text
40 registers/thread → 32 registers/thread
34.0785 → 23.9340 μs
latency −29.77%
```

### Single Decode

```text
56 registers/thread → 80 registers/thread
28.0087 → 26.1704 μs
latency −6.56%
```

一个靠寄存器变少获胜，另一个靠寄存器变多获胜。它们都正确，因为 GPU 隐藏延迟有两条主要道路：

1. **Occupancy**：让更多 warps/CTAs 同时驻留，当前 warp 等待时切换到别的 warp；
2. **ILP（Instruction-Level Parallelism）**：让同一线程保存更多独立 load、地址和 partial state，使更多指令同时在途。

寄存器数量决定两者的权衡，但方向由 kernel 的 shape、block size、工作集和依赖结构决定。

---

## 1. 寄存器不是一个数字，而是一整块 CTA 资源

GPU 为每个 SM 提供有限物理寄存器。一个 CTA 的寄存器需求大致是：

$$
R_{\mathrm{CTA}}
=
R_{\mathrm{thread}}
\times
T_{\mathrm{CTA}}
$$

仅看寄存器限制，可驻留 CTA 上限近似：

$$
N_{\mathrm{CTA,reg}}
=
\left\lfloor
\frac{R_{\mathrm{SM}}}{R_{\mathrm{CTA}}}
\right\rfloor
$$

实际还要同时满足 threads、warps、shared memory、cluster 和硬件 block limit。

关键在于 `floor`：资源不是连续变化，而是出现悬崖。

```text
需要 32768 registers/CTA → 可能驻留 2 CTA
需要 39936 registers/CTA → 可能只能驻留 1 CTA
```

少 7 个 registers/thread 可能几乎没效果，也可能突然让驻留 CTA 翻倍。

![寄存器资源从连续数字变成 CTA 驻留悬崖](/assets/blog-register-cliff-cta.svg)

*图 1：性能变化往往发生在整 CTA residency 的台阶，而不是每减少一个 register 都线性变快。*

---

## 2. 正例一：RMSNorm 为什么 40→32 带来 29.77%？

冻结 workload：

| 维度 | 值 |
| --- | --- |
| GPU | B200 / SM100a |
| Kernel | FlashInfer `FusedAddRMSNorm` |
| Dtype | BF16 |
| Batch | 2048 |
| Hidden | 8192 |
| Block | 1024 threads |
| Baseline toolchain | CUDA 12.8.93 |
| Candidate toolchain | CUDA 13.2.78 |

同一份 CUDA source，两个 ptxas 版本生成：

```text
CUDA 12.8：40 registers/thread
CUDA 13.2：32 registers/thread
```

每 CTA：

$$
40\times1024=40{,}960\ \mathrm{registers}
$$

$$
32\times1024=32{,}768\ \mathrm{registers}
$$

32-register 版本跨过双驻留门槛：

| Metric | CUDA 12.8 | CUDA 13.2 |
| --- | ---: | ---: |
| Registers/thread | 40 | 32 |
| Spill | 0 | 0 |
| Approx CTA/SM | 1 | 2 |
| Achieved occupancy | 47.4% | 88.2% |
| Effective bandwidth | 1.78 TB/s | 2.59 TB/s |
| Median latency | 34.0785 μs | 23.9340 μs |

RMSNorm 计算强度不高，需要流式读写大 tensor。更多驻留 warp 可以在一组 warp 等待 HBM/MIO 时运行另一组，因此 occupancy 的收益大于每线程状态损失。

两边 stack/local/spill 都为 0，所以这不是“新版本修复 spill”，而是纯粹的 residency cliff。

---

## 3. 同类正例：SiLU r32 为什么也快 20.17%？

FlashInfer SiLU-and-mul 的主 shape 是：

```text
tokens = 2048
d = 28672
block = 1024 threads
```

默认 39 registers，r32 为 32 registers：

| Metric | Default | r32 |
| --- | ---: | ---: |
| Registers | 39 | 32 |
| Spill | 0 | 0 |
| CTA/SM | 1 | 2 |
| Occupancy | 45.8% | 86.7% |
| DRAM bandwidth | 3.12 TB/s | 3.89 TB/s |
| Latency | 102.4969 μs | 81.8271 μs |

改善 `20.166%`。

SiLU 需要读两份 BF16 输入、计算 `expf`，再写一份 output。大 grid 足够填满 GPU，双驻留能有效隐藏 memory 与 transcendental latency。

但继续压寄存器就开始付 spill 税：

| Variant | Spill | 相对 default |
| --- | ---: | ---: |
| r32 | 0 | 约 −20.17% |
| r28 | 约 12 B | 仍快约 15% |
| r24 | 约 40–52 B | 只快约 5.5% |

跨过 occupancy cliff 有价值，但再压低会把线程状态搬到 local memory，收益迅速被反噬。

---

## 4. 为什么同一个 r32 换 Shape 会反转？

主 shape 有 2048 个 CTAs，足够覆盖所有 SM；SM 内从 1→2 CTA 很重要。

如果 tokens 只有 64，整个 grid 只有 64 CTAs，很多 SM 本来就没有 block。此时提高 SM 内 occupancy 不能解决 grid 不足，额外寄存器反而可能提高 ILP，因此 r48/r64 可能更快。

另一个 shape `2048×3584` 使用 448-thread block。默认约 39 registers 可驻留 3 CTA；某个 r48 版本实际 44 registers，只能驻留 2 CTA，回退约 20.2%。

相同寄存器数字在不同 block size 上对应不同 residency 台阶。

所以 register policy 的最小 key 应包括：

```text
(kernel, shape, block size, toolchain, architecture)
```

而不是一个全局 `-maxrregcount=32`。

---

## 5. 正例二：Single Decode 为什么 56→80 反而更快？

冻结 workload：

| 维度 | 值 |
| --- | --- |
| Kernel | FlashInfer `SingleDecodeWithKVCache` |
| Q / KV heads | 64 / 8 |
| Head dim | 128 |
| KV | 1024 / 8192 |
| Dtype | BF16 |
| Toolchain | CUDA 13.2 |

默认 target entry 56 registers，`-maxrregcount=80` 生成 80-register kernel。两边零 spill。

| Metric | Default | r80 |
| --- | ---: | ---: |
| Registers | 56 | 80 |
| Achieved occupancy | 38.0% | 24.9% |
| Cycles / instruction | 8.53 | 5.66 |
| Bandwidth | 1.16 TB/s | 1.24 TB/s |
| KV8192 latency | 28.0087 μs | 26.1704 μs |

occupancy 下降，但 latency 改善 `6.563%`。

Single decode 要保存：

- 多组 Q/K/V 地址；
- softmax max/sum；
- partial output；
- partition-KV metadata；
- merge state；
- 多个在途 global loads。

更多寄存器让线程保留更多独立状态，缩短 dependency chain、提高 ILP，避免反复重算或过早丢弃 partial。

![Occupancy 与 ILP 是两条不同的延迟隐藏路径](/assets/blog-register-occupancy-ilp.svg)

*图 2：Memory-streaming 大 grid 常受益于更多 warps；状态丰富、依赖复杂的 decode kernel 可能受益于更多每线程状态。*

---

## 6. `-maxrregcount=80` 为什么会让寄存器从 56 增到 80？

名字叫“最大寄存器数”，直觉上只会限制上限。但 nvcc 会把选项同时交给 CUDA 前端和 ptxas。前端可能据此生成不同 PTX、改变内联与 live range；最终不是简单地给同一个 SASS 加上限。

另外，FlashInfer host dispatch 会根据 occupancy 计算 `max_grid_size` 和 KV chunk 数量。寄存器变化不仅改变 kernel 内部，也可能反馈到 partition 形态。

因此这是一项：

```text
compiler resource decision
  + kernel ILP
  + host dispatch geometry
```

的组合实验，而不是单一寄存器因果。

---

## 7. 为什么 r80 也不是全局最优？

| Shape | 观察 |
| --- | --- |
| KV1024 | r80 只快约 1.58%，最优更接近 r64 |
| KV8192 | r80 快约 6.56% |
| KV≥16384 | 默认高 occupancy 可能重新占优 |
| r96 / r128 | 都没有超过 r80 |

短 KV 的循环较短，ILP headroom 小；长 KV 增加在途 load 和 partial state 的价值；再长时 occupancy 与 partition 反馈又可能反转。

不存在“B200 single decode 应统一使用 80 registers”的结论。只有 shape-bucketed policy 才可能安全。

---

## 8. `__launch_bounds__` 到底在告诉编译器什么？

CUDA 允许：

```cpp
__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
```

第二个参数不是强制 runtime 一定驻留这些 blocks，而是给编译器一个资源预算目标：若想允许至少 `minBlocksPerSM`，每 CTA 的 registers/SMEM 不能超过对应上限。

在 RMSNorm 中：

```cpp
__launch_bounds__(1024, 2)
```

可以让 CUDA 12.8 避免 40-register cliff，latency 从约 34.17 降到 25.48 μs；CUDA 13.2 已自然分配 32 registers，因此该 hint 基本免费。

它更像跨工具链的性能 guardrail，不是与 CUDA 13.2 的 29.78% 再相加的独立收益。

只写：

```cpp
__launch_bounds__(1024)
```

没有 `minBlocksPerSM`，反而可能允许 ptxas 使用更多寄存器，制造新的悬崖。

---

## 9. 负例一：把 40 Registers 压到 38/32，为什么 Activation 变慢？

Qwen3.5 D037 对 production activation kernel 扫 launch bounds：

| Arm | Registers | Median | Delta | Wins |
| --- | ---: | ---: | ---: | ---: |
| Default | 40 | 2.514400 μs | — | — |
| min 8 CTA/SM | 40 | 2.508240 μs | −0.245% | 30/30 |
| min 12 CTA/SM | 38 | 2.560520 μs | +1.834% | 0/30 |
| min 16 CTA/SM | 32 | 2.538520 μs | +0.959% | 0/30 |

所有 payload/scale byte-exact、零 spill。mb8 最好，但它根本没有降低 40 registers；38/32 arms 反而更慢，可能通过重算、调度或更短 live state 支付成本。

更关键的是，最佳 local saving 投影完整 graph 只有 `0.0069%`，低于 0.05% promotion floor，所以不跑 NCU 和 TP2 E2E。

“寄存器更少”既没有产生假设中的资源变化，也没有足够系统上限。

---

## 10. 负例二：把 Load 提前为什么导致 71 Registers 或 Spill？

GDN D001 尝试把下一组 recurrent-state `LDG.128` 提前，隐藏 long-scoreboard latency。

理论：

```text
compute group 0
  + prefetch/load group 1
```

但 load 发出后，值必须一直活到使用位置。提前越多，live range 越长，同时活跃的 state 越多。

结果：

| Arm | Registers | Stack / Spill | Median | Verdict |
| --- | ---: | ---: | ---: | --- |
| R93 control | 64 | 0 / 0 | 8.912480 μs | baseline |
| hoist n1/n2/n3 | 71 | 0 / 0 | 9.115320 μs | +2.276%，0/30 |
| hoist n4 | 64 | 8 B，LDL/STL | 未计时 | static reject |
| full group hoist | 64 | 8 B，LDL/STL | 未计时 | static reject |

ptxas 本来已经把下一组 load 合理穿插在 pack/store 中。手工移动 PTX 没有发现新自由度，只扩大 live range、改变全函数分配。

更深候选出现 spill，在 timing 前就应淘汰。

---

## 11. PTX 文本不同，为什么可能是同一个机器码？

D001 的 n1/n2/n3 PTX 文本不同，但全部下沉为同一个 1392-instruction SASS 和 71-register 资源形态。

所以它们不是三个独立机器码候选。把三个 PTX 文件各测一次，再挑最好，会把相同 binary 当成多个搜索样本。

必须保存：

- PTX hash；
- cubin/SASS hash；
- target-entry resource；
- target SASS diff；
- runtime selector。

优化对象是最终 entry，不是文本文件数量。

---

## 12. 一个统一心智模型：Occupancy、ILP 与 Spill 三角形

![寄存器预算在 Occupancy、ILP 与 Spill 之间的三角权衡](/assets/blog-register-tradeoff.svg)

*图 3：寄存器减少可以增加 resident warps，也可能迫使重算或 spill；寄存器增加可以提高 ILP，也可能跨过 occupancy cliff。*

可以粗略写成：

$$
T
\approx
T_{\mathrm{latency}}
\times
f(\mathrm{occupancy},\mathrm{ILP})
+T_{\mathrm{spill}}
+T_{\mathrm{recompute}}
+T_{\mathrm{dispatch}}
$$

没有单调关系。

### Occupancy 主导

- 1024-thread block；
- 大 grid；
- streaming memory；
- 每线程状态简单；
- 1→2 CTA cliff。

### ILP 主导

- 长循环；
- 多地址/partial state；
- load-use dependency；
- 足够 register 保存独立工作；
- grid/partition 反馈。

### Spill 主导

- cap 过紧；
- live range 未缩短；
- local load/store 出现；
- 原本想隐藏的延迟被更慢 local memory 取代。

---

## 13. 为什么 Toolchain 本身必须进入 Workload Contract？

CUDA 12.8 与 13.2 对同一 source 分配 40 vs 32 registers，产生近 30% latency 差异。

因此复现记录不能只写：

```text
CUDA 13
```

至少需要：

- nvcc version；
- ptxas version；
- target arch；
- compiler flags；
- register/launch bounds；
- source revision；
- target-entry SASS/resource hash。

升级工具链是一次候选变更，必须重新做 correctness 和性能矩阵，不能假定“新版一定更好”。

---

## 14. 如何系统地扫 Register Budget？

### Step 1：冻结 Target Entry

- exact shape/dtype/layout；
- mangled kernel symbol；
- launch geometry；
- host dispatch；
- correctness oracle。

### Step 2：建立自然基线

- default compiler allocation；
- target registers、spill、SMEM；
- occupancy-derived CTA/SM；
- no-profiler raw samples。

### Step 3：选择少量结构点

不要每个数字都扫。优先测可能跨 residency cliff 的点，以及 ILP 假设点：

```text
default
cliff−1
cliff
cliff+1
one higher-ILP point
```

### Step 4：静态淘汰

- unexpected spill/local；
- stack growth；
- target entry 未改变；
- PTX 不同但 SASS 相同；
- unsupported shape。

### Step 5：Operator A/B

- AB/BA/randomized blocks；
- 先测 A/A noise；
- correctness；
- raw samples；
- clock/foreign process guard。

### Step 6：算 E2E Ceiling

若调用频次和 critical share 太小，停止。Operator winner 不自动加载大模型。

### Step 7：Shape Bucket

每个 shape 独立选择。范围外保留自然 compiler path，不能全局设置一个 cap。

---

## 15. 结果应该怎样表述？

正确：

> 在 B200、BF16、B2048×H8192、1024-thread FlashInfer FusedAddRMSNorm 上，CUDA 13.2 target entry 从 40 降至 32 registers，允许约两 CTA/SM，operator latency 相对 CUDA 12.8 降低 29.77%。

正确：

> 在 B200、BF16、64/8 heads、D128、KV8192 的 isolated single decode 上，r80 target entry 以更低 occupancy 换取更高 ILP，operator latency降低 6.56%。

错误：

> B200 上 32 registers 最快。

错误：

> Occupancy 越高越快。

错误：

> 80 registers 是 attention 的通用最优值。

---

## 16. 当前证据边界

所有正例都是 accepted operator，不是 model E2E。SubCUDA 当前 offline replay 已验证：

- RMSNorm 90-row aggregate + 270 raw correctness rows；
- single decode 66-row aggregate + 198 raw correctness rows；
- SiLU 120-row aggregate + 360 raw correctness rows；
- D037 byte-exact register budget 与 low-ceiling stop；
- D001 random/finite exactness、0/30 和 spill static rejects。

Fresh build 仍需要 pinned FlashInfer headers、对应 CUDA toolchains 和空闲 B200；历史脚本含失效 GPU UUID，不能直接当现代一键命令。

---

## 结语：优化的是资源形态，不是寄存器数字

RMSNorm 与 SiLU 告诉我们：大 grid、1024-thread、streaming kernel 可以因为 32-register cliff 获得双驻留，显著提高 occupancy 和带宽。

Single decode 告诉我们：状态丰富、长循环的 kernel 可以从更多 registers 获得 ILP，即使 occupancy 下降仍更快。

D037 与 D001 又说明：

- 强压 register 不一定真的降低资源；
- 即使降低，也可能通过重算/调度变慢；
- 提前 load 会扩大 live range；
- cap 过紧会 spill；
- PTX 文本不同可能下沉成相同 SASS；
- local winner 还要先过 E2E ceiling。

所以，面对 register tuning，最重要的问题不是：

> “应该设成 32、64 还是 80？”

而是：

> **“这个 shape 的 block/grid 处在哪个 residency cliff？当前延迟更需要额外 warps，还是额外 per-thread state？为了改变它，我会引入 spill、重算或 dispatch 变化吗？”**

寄存器不是越少越好，也不是越多越好。只有与具体 execution shape 匹配，才是正确的资源合同。
