---
layout: post
title: "为什么删掉 67% 的“空 Row”仍然不快？从 Capacity、Compact Prefix 到 Persistent Fast Drain"
date: 2026-08-17 17:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [MoE, Persistent Kernel, Routing, CUDA Graph, Launch Bound, CLC, B200]
reading_time: 27
cover_image: /assets/blog-compact-route-fast-drain.png
excerpt: "B32、topK=8 的 routed MoE 为 256 个 route rows 预留容量，真实 expert-major prefix 却只有 84 或 90 行。把 launch/workspace bound 精确缩小 65%–67%，四个 rank/profile cell 仍全部 0/30。本文解释为什么 nominal capacity、metadata prefix、persistent CTA work 和真正删除的数据边不是一回事。"
---

> 本文基于 [`qhy991/SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab) 的 D022 compact route-grid case。代表性 TP0/layer0 operator JSON replay 与资产检查已通过；其余三个 cell 的归档结果同样保存在 case bundle 中。Fresh operator run 仍依赖冻结 FlashInfer DSO、P26 routes、rank-specific checkpoint weights 与两张 B200，因此本文不声称本轮重建了历史 DSO。

假设一个 MoE kernel 为 256 行 route metadata 预留空间，但真实请求只使用 84 行。

那么“空行”比例是：

$$
\frac{256-84}{256}
=67.19\%
$$

直觉上，把 launch bound 从 256 改成 84，应该能删除三分之二 CTA 工作。

真实结果却是：

```text
4 个 rank/profile cell
全部 0/30 wins
```

其中 TP0 两个 cell 甚至明确变慢；TP1 只落在 A/A envelope 内，没有可测收益。

为什么删掉 67% 的“行”，执行时间没有下降 67%，甚至连 1% 的稳定收益都没有？

因为这 172 行并不是 172 个完整 expert tiles。它们只是：

> **launch/workspace 的名义容量后缀。生产 persistent kernel 已经知道真实 prefix 长度，并能快速排空后缀。**

这篇文章要区分四个经常被混用的词：

```text
capacity
launch bound
active metadata prefix
physical work
```

只有最后一个真正决定大部分时间。

---

## 1. Route Row 是什么？

冻结 workload：

| 维度 | 值 |
| --- | ---: |
| GPU | 2×B200，TP2 operator cell |
| Model slice | Qwen3.5-122B-A10B-FP8 routed MoE |
| Batch | 32 |
| Top-K | 8 |
| Routed experts | 256 |
| Hidden | 3072 |
| Local intermediate | 512 |
| Tactic tile | 8 routed tokens / row |

Router 一轮产生：

$$
32\times8=256
$$

个 token-expert assignments。

将 assignments 按 expert-major 排序后，第 $e$ 个 expert 有 $c_e$ 个 routed tokens。

对于 tile size $T=8$，这个 expert 需要的 CTA rows 是：

$$
r_e=\left\lceil\frac{c_e}{8}\right\rceil
$$

完整真实 row 数：

$$
R_{\mathrm{exact}}
=
\sum_{e:c_e>0}
\left\lceil\frac{c_e}{8}\right\rceil
$$

两份冻结 route profile 得到：

```text
layer0: 84 rows
minimum-HHI: 90 rows
```

它们已经是 expert-major compact prefix，不是稀疏 256×something 矩阵。

---

## 2. 256 从哪里来？

`Routing::getMaxNumCtasInBatchDim` 需要在不知道本轮具体 expert distribution 时，为 workspace 和 BMM launch 提供 distribution-independent upper bound。

它返回：

```text
R_max = 256
```

这保证任何合法 B32/topK8 distribution 都有足够空间。

Route kernel 随后写入：

```text
metadata[0 : R_exact]
numNonExitingCtas = R_exact
```

后面的：

```text
metadata[R_exact : 256]
```

只是 capacity suffix。

![Route Capacity、Compact Prefix 与 Persistent Consumer](/assets/blog-route-capacity-prefix.svg)

*图 1：256 是 graph-stable capacity；84/90 是本轮 compact metadata prefix；persistent consumer 读取 `numNonExitingCtas`，不会把整个后缀当完整 expert work。*

---

## 3. “Launch 了 CTA”不等于“执行了完整 Tile”

普通 kernel 的直觉是：

```text
grid.x = 256
→ 256 CTAs 都执行相同完整函数
```

Persistent/provider kernel 更复杂。CTA 可能：

- 从全局 work queue 领取任务；
- 检查真实 work count；
- 发现没有任务后 early exit；
- 通过 CLC fast drain 取消或快速排空尚未需要的 launch suffix；
- 一个 CTA 连续处理多个 logical rows；
- 多个 CTA 竞争同一个动态 work pool。

所以 nominal grid rows 与 full-work CTAs 不是一一对应。

可以写成：

$$
T_{\mathrm{operator}}
\approx
T_{\mathrm{active\ tiles}}
+T_{\mathrm{queue/control}}
+T_{\mathrm{drain}}
+T_{\mathrm{launch/schedule}}
$$

缩小 `R_max` 只可能减少后两项的一部分；如果 `T_active tiles` 主导，而且 drain 已经很便宜，收益上限就很小。

---

## 4. CLC Fast Drain 在解决什么？

Blackwell 的 Cluster Launch Control（在这个 provider 中表现为 CLC-enabled fast drain）允许执行中的工作更快地处理“不再需要的 launch work”。

重要的是，它不是：

> “先把 256 个完整 CTA 都跑起来，再让其中 172 个白算。”

更接近：

```text
launch capacity = 256
real work count = 84
persistent kernel consumes real prefix
unused suffix is canceled / drained cheaply
```

因此：

$$
172\ \mathrm{nominal\ rows}
\not\equiv
172\ \mathrm{full\ expert\ CTA\ costs}
$$

D022 的候选把 capacity 本身改成 84/90，但没有删除：

- route sorting；
- active metadata writes；
- active expert tiles；
- FP8/BF16 math；
- expert weight reads；
- routed output writes；
- TP rank-specific work；
- persistent queue/control 的主体。

---

## 5. 四种“Compaction”不能混为一谈

![Metadata、Launch、Work 与 Dataflow 四种 Compaction](/assets/blog-route-four-compactions.svg)

*图 2：D022 只缩小 launch/workspace capacity；route metadata 本来已 compact，active compute 与数据边都没有减少。*

### 5.1 Layout compaction

把 expert-major metadata 从有洞布局变成连续 prefix。

D022 之前已经完成。

### 5.2 Capacity compaction

把 workspace/launch upper bound 从 256 改成 84/90。

这是 D022 唯一修改。

### 5.3 Work compaction

减少 persistent scheduler 真正领取并执行的 active tiles。

D022 没有改变，因为 `numNonExitingCtas` 与真实 routes 相同。

### 5.4 Dataflow compaction

删除排序、搬运、中间 tensor、weight traffic 或下游 compute edge。

D022 也没有改变。

真正大的系统收益通常来自第 3/4 类，不是第 2 类表面数字。

---

## 6. Candidate 到底改了什么？

候选保持以下全部不变：

- Router logits 与 projected counts；
- route SHA；
- expert-major metadata prefix；
- expert weights；
- production autotuned tactic；
- cubin family；
- route ABI；
- output layout；
- CUDA Graph shape；
- BF16 output semantics。

只对精确 shape：

```text
(tokens, topK, experts, tile)
= (32, 8, 256, 8)
```

把 launch/workspace row bound：

```text
256 → 84
256 → 90
```

并分别构建独立 DSO：

```text
bound256
bound84
bound90
```

这是一个很干净的 single-variable oracle。

---

## 7. 正确性为什么同时检查 Route 和 Output？

只检查 BF16 output 不够。

如果 candidate 改变 route metadata，但恰好某层输出没有暴露差异，仍然破坏了执行 contract。

D022 检查：

1. TP0/TP1 观测到的 routes 与不可变 P26 receipt 完全相同；
2. Layer0/min-HHI route SHA 一致；
3. Control A、Control B、candidate 的 BF16 output SHA 完全相同；
4. 不使用 tolerance。

如果 exact bound 小于真实 row 数，harness 在 launch 前拒绝，不允许依赖 kernel 越界或 partial output 来发现错误。

---

## 8. 四个 Cell 的结果

每个 cell：

```text
30 randomized blocks
× 100 CUDA Graph replays
control A / candidate / control B
```

| Route | Rank | Exact rows | Control A | Candidate | Delta | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer0 | TP0 | 84 | 76.292479 μs | 77.781601 μs | +1.9519% | 0/30 |
| Layer0 | TP1 | 84 | 77.947841 μs | 78.109760 μs | +0.2077% | 0/30 |
| Min-HHI | TP0 | 90 | 84.756799 μs | 85.676160 μs | +1.0847% | 0/30 |
| Min-HHI | TP1 | 90 | 85.633278 μs | 85.888000 μs | +0.2975% | 0/30 |

![D022 四个 Rank/Profile Cell 的 Operator 裁决](/assets/blog-route-grid-results.svg)

*图 3：四个 cell 全部 0/30；TP1 的小差值落在 A/A envelope 内，只能说无收益；TP0 比两个 control 都慢，构成明确回退。*

---

## 9. 为什么 TP1 不应写成“回退 0.2%”？

Layer0 TP1：

```text
Control A = 77.947841 μs
Control B = 78.277760 μs
Candidate = 78.109760 μs
```

Candidate 虽比 A 慢 0.2077%，却位于两个 control 之间。

这说明独立 Graph capture / order 本身存在约同量级系统偏移。

所以严谨结论是：

> **没有可测收益。**

不是：

> “Candidate 明确慢 0.2077%。”

相反，TP0 candidate 比更慢的 Control B 仍多：

- Layer0：约 `1.01 μs`；
- Min-HHI：约 `0.62 μs`。

因此 TP0 回退超出 A/A envelope，可以明确拒绝。

同一张表中，不同 cell 可以有“明确回退”和“无收益”两种 verdict，不能统一写成平均慢多少。

---

## 10. 为什么不继续跑 NCU / Nsys？

No-profiler operator gate 四个 cell 全部 0/30。

Contract 规定：

```text
operator 不过门
→ 不做 NCU
→ 不做 Nsys
→ 不集成 OMoE
→ 不跑 TP2 wall
```

因此我们不能进一步断言 TP0 回退具体来自：

- launch wave 变化；
- CLC 状态；
- cache alignment；
- workspace layout；
- scheduler heuristic；
- binary placement。

这些都只是可能解释，没有 profiler 证据。

负结果支持的最小结论是：

> **缩小 exact bound 没有改善完整 fused operator；nominal suffix 不是完整工作量。**

这已经足够关闭原假设。

---

## 11. 这和 Problem Size Mask 是否矛盾？

不矛盾。

前面的 Expert Specialization 文章使用固定 `group_count=256`，把不属于当前 kernel 的 expert problem 写成 `(0,0,0)`，目的是避免 GPU→CPU 回读和动态 host launch。

那里的关键思想是：

> **静态外壳可以很大，只要 inactive work 能被便宜跳过。**

D022 正好是同一原则的另一个实例：

```text
固定 capacity / graph shape
真实 compact prefix
device-side real work count
fast skip / drain
```

如果 inactive entries 已经是低成本控制流，继续缩静态上界未必有收益，反而可能破坏稳定的 launch/schedule 条件。

---

## 12. 什么时候缩小 Launch Bound 才可能有价值？

至少满足一个条件：

1. Inactive suffix 仍执行大量 prologue、descriptor、register 或 memory work；
2. Grid 太大导致额外 launch waves；
3. CLC/early-exit 不可用或 drain 成本可测；
4. Workspace capacity 造成 material allocation/cache pressure；
5. Smaller grid 改善与其他 stream 的并发；
6. Active work count 没有可靠 device-side owner；
7. Graph-stable exact bound 可以零成本更新。

但即使符合，也要做同路径 operator A/B。不能从：

```text
nominal rows reduced 67%
```

直接预测：

```text
latency reduced 67%
```

---

## 13. 真正值得重开的方向是什么？

D022 关闭的是 capacity-only shrink。

更有价值的重开方向应真正删除 active work 或数据边，例如：

- 更早消除 padding/duplicate assignments；
- 减少 route sorting / permutation traffic；
- 让 producer 直接写 consumer layout；
- 合并 route metadata 与 activation movement；
- 减少 active expert tiles，而不是 capacity rows；
- 改变 load balance，使 slow rank 的真实工作下降；
- 在 compute/communication critical path 上删除 materialized boundary。

这些都需要新 contract，不能借用 D022 的“67% nominal reduction”作为预期收益。

---

## 14. 一个通用的 Capacity Audit

遇到大数组、大 grid、大 workspace 时，先画四层：

```text
Allocated capacity
  ↓
Valid metadata extent
  ↓
Admitted work items
  ↓
Full-cost executed work
```

逐层问：

| 问题 | 证据 |
| --- | --- |
| Capacity 多大？ | allocation / launch bound |
| 实际 metadata 多长？ | device counter / hash |
| Scheduler 接受多少 work？ | queue/head/numNonExitingCtas |
| 多少 work 执行完整 prologue+math？ | marker / SASS-aware trace |
| Inactive suffix 如何退出？ | early-exit / CLC / predicate |
| 缩 capacity 是否改变 active work？ | same-route A/B |

只有最后两层下降，才有较强性能先验。

---

## 15. 最后记住

1. **Capacity 不是工作量。**
2. **Grid rows 不是 full-cost CTA 次数。**
3. **Compact metadata 不等于 compact launch，也不等于 compact compute。**
4. **Persistent kernel 可以在固定大外壳内只消费真实 prefix。**
5. **删除 nominal suffix 前，先测它当前是否已经被便宜 drain。**

D022 最值得学习的不是“精确 bound 无效”，而是一个更通用的判断：

> **不要优化静态表示中看起来很大的数字；先证明这个数字在物理执行中真的对应昂贵工作。**

---

## Evidence boundary

- Source snapshot：[`SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab)。
- Case：D022 `rejected-operator / default-off`；代表性 TP0/layer0 JSON replay 与资产检查通过。
- Workload：2×B200 TP2 operator、B32、topK8、256 routed experts、tile8；exact rows 为 84/90。
- Correctness：route SHA 与 BF16 output byte-exact；candidate 只改变 launch/workspace bound。
- 四个 rank/profile cell 都是 30×100 randomized CUDA-Graph A/B/A，全部 0/30。
- 没有运行 NCU、Nsys、OMoE integration 或 TP2 E2E；本文不指定 TP0 回退的微架构原因。
- Fresh operator 执行因冻结 DSO/routes/checkpoint/B200 环境缺失而 BLOCKED。
- 状态与重开条件见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
