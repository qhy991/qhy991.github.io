---
layout: post
title: "为什么 DRAM 流量减半，Kernel 反而更慢？从 FP8 解码税、Warp ILP 到 Matched Event"
date: 2026-08-17 16:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Memory Bandwidth, FP8, Vector IO, MLP, ILP, Nsight Compute, PTX, B200]
reading_time: 30
cover_image: /assets/blog-dram-bytes-down-slower.png
excerpt: "Pure-PTX FP8 QKV 把权重 DRAM read 从 12.59 MB 降到 6.32 MB，NCU isolated duration 也更好，matched-event 却慢 5.06%；另一条 activation kernel 的 U16/U32 vector I/O 真正进入 SASS，也因 128→64/32 threads 变慢。本文解释 bytes、transaction width、occupancy、MLP、ILP 和 conversion tax 为什么必须一起进入 break-even 模型。"
---

> 本文联合三条独立证据线：SubCUDA `d1db18f` 中 Llama pure-PTX FP8 QKV 的 archived operator/attribution、D038 activation vector-I/O 的可重放 operator reject，以及 KV page-copy CUDA 13.0→13.2 的可重放 toolchain attribution。三条 workload、baseline 和 timing boundary 不同，本文只比较机制，不把百分比拼成一张排行榜。Pure-PTX QKV 的代码资产检查通过，但当前 verifier 没有声明 offline replay contract，因此不声称本轮重放了其性能 JSON。

GPU 性能分析里最有诱惑力的结论之一是：

> “DRAM bytes 减少了一半，所以 kernel 应该接近快一倍。”

在 Llama-3.2-1B 的一个 B1 QKV GEMV 中，BF16 control 每次读取约：

```text
12,594,432 bytes
```

Pure-PTX FP8 weight-only 版本读取：

```text
6,315,520 bytes
```

DRAM read 几乎减半，L1 lookup miss 也从 `412,160` 降到 `215,936`。Nsight Compute 的 isolated duration 甚至显示：

```text
8.256 → 7.232 μs
```

看起来是一个漂亮的 memory optimization。

但真正的 balanced matched-event 测量给出：

```text
FP8 row median = 12.188 μs
相对 BF16 vector128 = 1.0506×
13/13 blocks 更慢
```

也就是慢约 `5.06%`。

Profiler 没撒谎：流量真的下降了。错误在于我们把“流量下降”误写成了“完整执行时间一定下降”。

---

## 1. 一个 Kernel 的时间不只有 Bytes / Bandwidth

最简 roofline 直觉是：

$$
T_{\mathrm{memory}}
\approx
\frac{Q_{\mathrm{bytes}}}{B_{\mathrm{effective}}}
$$

但压缩权重的 kernel 还要支付：

$$
T_{\mathrm{total}}
=
T_{\mathrm{load}}
+T_{\mathrm{decode}}
+T_{\mathrm{scale}}
+T_{\mathrm{convert}}
+T_{\mathrm{math}}
+T_{\mathrm{schedule}}
+T_{\mathrm{sync}}
$$

FP8 把 $T_{\mathrm{load}}$ 降低，却可能增加：

- E4M3 unpack；
- scale load 与 broadcast；
- FP8→FP32/BF16 conversion；
- 更多 integer/bit manipulation；
- 更长 dependency chain；
- register live state；
- 更低 issue-level parallelism。

真正的 break-even 条件是：

$$
\frac{Q_{\mathrm{saved}}}{B_{\mathrm{effective}}}
>
T_{\mathrm{unpack}}
+T_{\mathrm{scale}}
+T_{\mathrm{convert}}
+\Delta T_{\mathrm{schedule}}
$$

只算左边，不算右边，roofline 就变成了愿望清单。

---

## 2. Pure-PTX QKV 做了什么？

目标 GEMV：

$$
[1,2048]\times[3072,2048]
$$

流程：

```text
BF16 checkpoint weight
  → AOT pure-PTX pack to E4M3
  → resident FP8 blob + FP32 scales

BF16 activation
  → pure-PTX FP8 GEMV
  → load FP8
  → decode + scale
  → multiply / reduce
  → BF16 Q/K/V
```

它有三种 scale contract：

- 每 output row 一个 scale；
- 每 256 weights 一个 scale；
- 每 128 weights 一个 scale。

只有 row-scale arm 通过预注册 quality gate：

```text
rel-L2 = 0.025966
cosine = 0.999663
```

Group-256 与 group-128 分别超过自己的 rel-L2 上限，不能因为可能更快就进入正式 timing claim。

---

## 3. Bytes 确实下降了，为什么 Matched Event 仍更慢？

NCU 因果证据：

| Counter | BF16 vector128 | FP8 row | 说明 |
| --- | ---: | ---: | --- |
| DRAM reads | 12,594,432 B | 6,315,520 B | 压缩有效 |
| L1 lookup misses | 412,160 | 215,936 | 请求减少 |
| Isolated NCU duration | 8.256 μs | 7.232 μs | profiler 条件下更短 |
| Long scoreboard / issue-active warp | 11.489 | 9.395 | 部分等待改善 |

但 FP8 GEMV 额外包含 28 条 static native E4M3 unpack，以及 conversion/data-move 链。

![FP8 权重压缩的 Bytes Saving 与 Decode Tax](/assets/blog-fp8-bytes-vs-decode-tax.svg)

*图 1：压缩减少 GMEM traffic，但每次使用权重都要在线解码。只有节省的 memory time 大于 unpack/scale/convert/schedule 税，matched event 才会变快。*

这说明两个测量回答不同问题：

### NCU isolated capture

用于确认：

- bytes 是否真的减少；
- cache miss、stall、instruction 是否按机制变化；
- 候选机器码是否存在。

### Balanced matched event

用于裁决：

- 在同一 harness、同一时间块、相同 launch 数下，完整 operator 是否更快。

Profiler 会序列化、插桩或改变 replay 条件。它是机制显微镜，不是最终 wall-time 法官。

---

## 4. 为什么 “13/13 更慢” 比一次 Median 更有解释力？

Pure-PTX QKV 使用 13-block balanced matched-event。Row arm：

```text
median = 12.188 μs
relative = 1.0506× BF16 vector128
13/13 blocks slower
bootstrap CI = [1.0346, 1.0602]
```

这比独立运行两个 median 更能抵消：

- clock drift；
- 温度；
- cache warm state；
- host jitter；
- order effect。

预注册 prefilter 甚至要求：

```text
≤ 6.85 μs
≤ 0.583× framework linear
```

没有 arm 接近。因此没有运行 model A/B、qualification gate 或 canonical gate。

不能把“没跑模型”写成：

> “Operator 虽慢，但模型也许会赢。”

没有证据的层级保持未运行。

---

## 5. D038：更宽 LD/ST 为什么也会变慢？

另一条 Qwen3.5 activation kernel 将 scalar byte I/O 改成：

```text
vec2: uint16_t
vec4: uint32_t
```

最终反汇编确认：

- vec2 生成 `LDG.U16 / STG.U16`；
- vec4 生成 32-bit `LDG / STG`；
- 零 spill。

所以“vectorization 被编译器优化掉”已经排除。

但 thread mapping 同时发生变化：

| Arm | Threads | 每线程处理元素 | Final I/O | Registers |
| --- | ---: | ---: | --- | ---: |
| Scalar | 128 | 1 | U8 | 40 |
| Vec2 | 64 | 2 | U16 | 24 |
| Vec4 | 32 | 4 | U32 | 49 |

每个存活 thread 要串行执行更多：

```text
dequant
→ SwiGLU
→ max/scale
→ reciprocal
→ E4M3 quant
→ store
```

而可发射 warp 数减少。

![Vector Width 与 Warp-level Parallelism 的交换](/assets/blog-vector-width-warp-ilp.svg)

*图 2：更宽 transaction 减少 scalar I/O，却把 128 条并行 element chains 压到 64/32 threads。瓶颈若在 MUFU.EX2、reciprocal 和 conversion，warp 变少会更慢。*

结果：

| Arm | Median | Delta | Wins |
| --- | ---: | ---: | ---: |
| Scalar | 2.528520 μs | baseline | — |
| Vec2 | 2.561640 μs | +1.3099% | 0/30 |
| Vec4 | 2.577160 μs | +1.9237% | 0/30 |

A/A absolute p95 只有 `0.008612 μs`；vec2/vec4 的 paired regression 分别为 `0.033440` 与 `0.048720 μs`，明显高于噪声。

更宽 I/O 生效了，kernel 仍然变慢，因为瓶颈不是 scalar memory instruction count。

---

## 6. 编译 Flag 也是数值合同：`--use_fast_math` 不能漏

D038 第一次构建没有使用生产 `--use_fast_math`。

结果不是明显错误，而是 FP32 group scale 最后几 bit 改变。Payload/scale 的 byte-exact gate 在 timing 前失败。

恢复与 incumbent 相同的 compile contract 后，candidate 才能进入 A/B。

这提醒我们：

```text
相同 CUDA source
不同编译 flag
```

可能是不同 numerical program，尤其涉及：

- reciprocal；
- exp/log；
- fused multiply-add；
- denorm；
- rounding；
- conversion。

不能为了让 candidate 通过，单独给它更宽的 tolerance；应先恢复 same-program build contract。

---

## 7. 反例中的正例：Occupancy 不变，为什么 Page Copy 快了 33%？

sgl-kernel KV page-copy 的历史 toolchain attribution 使用：

- 完全相同 CUDA source；
- 相同 launch：2 CTA × 1024 threads；
- 相同 production `block_quota=2`；
- 相同约 49.5% achieved occupancy；
- 只切换 CUDA 13.0.88 与 13.2.78。

结果：

| Shape | CUDA 13.0 | CUDA 13.2 | 13.2 reduction |
| --- | ---: | ---: | ---: |
| 65536 × 2048 B | 7810.182 μs | 5198.029 μs | −33.45% |
| 1024 × 131072 B | 7339.872 μs | 5001.011 μs | −31.87% |

主 shape 的 NCU：

| Metric | CUDA 13.0 | CUDA 13.2 |
| --- | ---: | ---: |
| Occupancy | 49.5% | 49.5% |
| Active warps | 31.7 | 31.7 |
| Memory throughput | 63.7 GB/s | 95.6 GB/s |
| Warp cycles / executed instruction | 65.6 | 46.5 |
| Issued warp / scheduler | 0.12 | 0.17 |

Occupancy 完全相同，性能却差三成。

归档 SASS/NCU 表明新工具链生成更深的 load/store 展开，让每个 warp 同时保持更多 memory operations in flight，也就是更高 MLP。

![相同 Occupancy 下，Compiler Schedule 如何改变 MLP](/assets/blog-toolchain-mlp-schedule.svg)

*图 3：Occupancy 描述“有多少 warp 可驻留”；MLP/ILP 描述每个 warp 能让多少独立操作同时在途。两者不能互相替代。*

这条正例和前两个负例共同说明：

> **关键不是 I/O 看起来多宽，而是完整机器调度能否持续给 memory/compute pipeline 喂独立工作。**

---

## 8. Occupancy、ILP、MLP 分别是什么？

### Occupancy

一个 SM 上可驻留多少 active warps，相对硬件上限的比例。

它回答：

> 当前 warp 等待时，有多少其他 warp 可以切换？

### ILP

单个 thread/warp 中有多少互不依赖的指令可以重叠执行。

它回答：

> 不切换 warp，当前 warp 自己能保持多少独立工作在途？

### MLP

有多少独立 memory requests 可以同时在途。

它回答：

> Memory latency 是否被多个并发请求摊薄？

三者关系不是单调的：

- 更多 registers 可能降低 occupancy，却提高 ILP；
- 更宽 vector 可能减少指令，却减少 threads/warps；
- 更深 unroll 可能增加 MLP，也可能增加 registers/spill；
- bytes 减少可能伴随更长 decode chain。

---

## 9. 为什么单一 Profiler Counter 不能宣布 Winner？

下面每句话都可能真实，但不足以支持“更快”：

```text
DRAM bytes 少了 50%
L1 misses 少了 48%
vector load 变宽了
registers 少了
occupancy 高了
NCU duration 短了
```

完整因果链至少需要：

```text
source change
  → target SASS / resource change
  → profiler mechanism movement
  → no-profiler matched operator wall
  → correctness
  → E2E / serving（若声明到该层）
```

Profiler counter 位于中间，不能跳过 wall-time authority。

---

## 10. 怎样建立 Compression Break-even 模型？

对 weight-only compression，可以估计：

$$
T_{\mathrm{saved}}
=
\frac{Q_{\mathrm{BF16}}-Q_{\mathrm{FP8}}}{B_{\mathrm{effective}}}
$$

新增成本：

$$
T_{\mathrm{added}}
=
N_{\mathrm{weights}}
\left(
t_{\mathrm{unpack}}
+t_{\mathrm{convert}}
+t_{\mathrm{scale}}
\right)
+\Delta T_{\mathrm{schedule}}
$$

若：

$$
T_{\mathrm{saved}}\le T_{\mathrm{added}}
$$

即使 memory counters 全部变好，kernel 也不会赢。

更现实的模型还要考虑：

- unpack 是否能与 load overlap；
- conversion pipeline throughput；
- scale reuse；
- register pressure；
- warp count；
- cache residency；
- activation/math 是否已经成为新瓶颈。

---

## 11. 怎样设计诚实的 Memory Optimization 实验？

### Gate 0：数值合同

压缩格式先过独立 pack/dequant oracle；候选 quality threshold 在性能前冻结。

### Gate 1：Binary proof

确认目标 load/store/unpack 真正进入 SASS，记录 register、spill、shared memory。

### Gate 2：Matched operator wall

使用交错或 randomized blocks；profiler 关闭；带 A/A 或 matched control。

### Gate 3：Mechanism profile

在通过或需要解释的 arm 上看 bytes、miss、stall、issue、MLP。Profiler 不覆盖 Gate 2。

### Gate 4：E2E

只有 operator material 且实际 path-hit，才运行模型/serving。

### STOP

如果候选：

- quality fail；
- 0/N wins；
- below noise；
- matched wall 回退；

就保留负结果，不用更漂亮的 counter 翻案。

---

## 12. 三条案例分别关闭了什么？

### Pure-PTX FP8 QKV

关闭：当前 row/g256/g128 portfolio 在冻结 B1/M1 QKV operator 中的 promotion。

没有关闭：更低 unpack tax、硬件 native FP8 datapath、不同 scale reuse 或更大 batch。

### D038 Vector I/O

关闭：通过减少 threads 实现 U16/U32 的当前 mapping。

没有关闭：保持 128 threads、用 cooperative vector carrier 的新设计。

### KV Page Copy Toolchain

支持：相同 occupancy 下，compiler schedule/MLP 可以产生巨大差异。

限制：历史环境/commit provenance 仍有 fresh-build debt，随机 index 也不是所有生产 trace。

---

## 13. 最后记住

1. **Bytes 是成本的一部分，不是完整时间。**
2. **更宽 I/O 若减少 warp 或增加串行 chain，可能更慢。**
3. **Occupancy 相同不代表 MLP/ILP 相同。**
4. **NCU 证明机制，matched no-profiler event 裁决性能。**
5. **压缩必须支付在线 decode/scale/conversion 税。**

真正应该问的不是：

> “流量少了多少？”

而是：

> **“节省的 memory time，能不能覆盖格式转换、依赖链、warp/issue parallelism 和调度变化的全部新增成本？”**

---

## Evidence boundary

- Source snapshot：[`SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab)。
- Pure-PTX QKV：代码/审计资产检查通过；当前 verifier 无 offline replay contract，性能与 NCU 数字来自冻结教学记录，未在本轮重放；无 model A/B。
- D038：operator JSON replay 与资产检查通过；scalar/vec2/vec4 在 production fast-math contract 下 byte-exact，vec2/vec4 均 0/30。
- KV page-copy：CSV replay 与资产检查通过；CUDA 13.0/13.2 两 shape、三轮、byte-exact；fresh build 仍需完整 pinned dependency/toolchain reconstruction。
- 三条 case 的 workload、baseline 和 timing boundary 不同，不能组成 waterfall 或跨案例百分比比较。
- 所有结论限定 B200/SM100a 与各自冻结 shape；没有 deployment-wide promotion。
- 状态与重开条件见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
