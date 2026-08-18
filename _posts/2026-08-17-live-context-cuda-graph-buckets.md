---
layout: post
title: "为什么 Mask 掉的 KV 仍然要付费？从 max_len=2048 到 Live-Context CUDA Graph Buckets"
date: 2026-08-17 20:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Attention, CUDA Graph, KV Cache, Context Bucket, Serving, Qwen3, B200]
reading_time: 28
cover_image: /assets/blog-live-context-buckets.png
excerpt: "Decode Graph 为 max_len=2048 捕获时，真实 live context 只有约 190，causal mask 能把无效位置的数学贡献变成零，却不能取消 KV load、score 和 softmax work。256/512/1024/full Graph portfolio 通过严格 bucket_len > max_position 的 selector 裁掉物理尾部，并把 focused −20.1% 传到 C16 Serving +13.83%。"
---

> 本文基于 Qwen3-4B/B200 context-bucket Serving 的 Q4B12/E-Q4B-012 冻结记录：focused Graph、1920/1920 faithfulness、两顺序 C16 A/B、gate 与 rollback evidence 均存在。该 lineage 在 agentic-megakernel 中尚未迁成独立 C4 bundle，历史 land ref 也仍需统一 public binding，因此本文引用冻结 evidence，不声称 fresh replay 或当前主分支默认状态。

假设 CUDA Graph 为：

```text
max_len = 2048
```

捕获。

当前 continuous-batch decode 中，最长 active request 只有：

```text
live context ≈ 190
```

Attention 仍然可以通过 causal/valid mask 保持正确：

```text
position 190…2047
→ score = -∞
→ softmax contribution = 0
```

数学结果没问题。

但 GPU 可能仍然：

- 读取被 mask 尾部对应的 K/V pages；
- 生成 score；
- 执行 max/sum reduction；
- 处理 predicate；
- 占用 CTA/warp；
- 让 Graph replay 等待完整 max_len kernel。

这就是一个重要区别：

> **Mask 删除数值贡献，不一定删除物理工作。**

Qwen Serving 的 context-bucket portfolio 将 decode graph 分成：

```text
K256
K512
K1024
K2048 full fallback
```

每步选择覆盖所有 active rows 的最小合法 bucket。结果：

```text
focused decode: −20.1%
C16 Serving: 2803.6 → 3191.3 tok/s
+13.83%
```

---

## 1. Mask 正确性与执行剪枝是两件事

设 attention logits 为：

$$
s_i=\frac{q\cdot k_i}{\sqrt d}
$$

对无效位置加入 mask：

$$
s_i'=
\begin{cases}
s_i,& i<L_{\mathrm{live}}\\
-\infty,& i\ge L_{\mathrm{live}}
\end{cases}
$$

Softmax 后：

$$
p_i=0,\quad i\ge L_{\mathrm{live}}
$$

这证明尾部不影响输出。

却不能证明 kernel 没有先计算 $q\cdot k_i$，也不能证明没有读取 $v_i$ 或参与 reduction scheduling。

### Logical masking

```text
shape = 2048
work = 2048 positions
tail contribution = 0
```

### Physical slicing

```text
shape = 256
work = 256 positions
tail does not enter kernel
```

![Logical Mask 与 Physical KV Slice 的区别](/assets/blog-context-mask-vs-slice.svg)

*图 1：Mask 保证语义，slice 才改变 launch shape 和数据路径。Context bucket 同时利用两者：用数学等价证明可裁尾，用更短 graph 真正删除物理工作。*

---

## 2. 为什么旧 Graph 固定 2048？

CUDA Graph 需要稳定 kernel DAG 与参数形态。最安全的 baseline 是：

```text
capture at max_len=2048
always replay same shapes
mask invalid tail
```

优点：

- 所有 context 都合法；
- 不需要 selector；
- graph cache 简单；
- buffer shape 固定；
- 容易 rollback。

缺点：

- 短 context 按最大形状付费；
- HBM traffic 与 attention work 不能随 live length 缩小；
- continuous batch 大部分生命周期可能远低于 max_len。

在 fixed max_len 与 fully dynamic eager 之间，bucketed Graph 是一个折中。

---

## 3. Bucket Portfolio 如何工作？

捕获有限图集合：

```text
Graph decode @ K=256
Graph decode @ K=512
Graph decode @ K=1024
Graph decode @ K=2048
```

所有图共享：

- persistent KV tensor；
- active-row buffers；
- model weights；
- slot/length state；
- output buffers。

差异只在传给 attention 的 KV slice / captured shape。

每步计算：

$$
p_{\max}=\max_r p_r
$$

然后选择：

$$
K_{\mathrm{bucket}}
=
\min\{K\in\{256,512,1024,2048\}:K>p_{\max}\}
$$

注意是严格：

$$
K>p_{\max}
$$

不是：

$$
K\ge p_{\max}
$$

---

## 4. 为什么必须 `bucket_len > max_position`？

假设当前 token 写入位置：

```text
p = 255
```

需要包含索引 255 的 KV slice，最小长度正好 256。

如果 selector 使用：

```text
bucket_len >= max_position
```

那么当 `max_position=256` 时可能仍选 K256，但合法索引范围是：

```text
0…255
```

当前位置 256 越界。

正确条件：

```text
bucket_len > max_position
```

因此 crossing：

```text
max_position 255 → K256
max_position 256 → K512
```

![Strict Bucket Selector 与 256→512 Crossing](/assets/blog-context-bucket-selector.svg)

*图 2：Bucket length 是 tensor extent，position 是零基索引；严格大于保证当前 token 的 KV write/read 都落在 slice 内。*

这类 off-by-one 不是普通性能 bug，而会改变 KV side effect 和后续 token。

---

## 5. 为什么按 Batch 的最大 Live Position 选择？

一个 B16 decode graph 同时服务多行：

```text
row 0: 190
row 1: 82
row 2: 255
...
```

Graph 使用统一 tensor shape，必须覆盖最长 row：

$$
p_{\max}=255
\Rightarrow K256
$$

短 row 仍会在 bucket 内 mask 自己的无效位置。

这意味着 bucketed Graph 仍有“最长请求拖累同 batch”的 tail amplification：

$$
\mathrm{waste}
\propto
B\cdot K_{\mathrm{bucket}}
-\sum_r L_r
$$

但相比所有 row 固定 K2048，浪费显著下降。

如果要每行不同 bucket，就需要：

- 按 context regroup；
- 多 kernel/stream；
- ragged attention；
- 更复杂 scheduler；

那是另一个设计空间。

---

## 6. Graph Faithfulness 如何验证？

Context bucket 改变的是输入 extent，不应改变有效位置数学。

冻结验证覆盖：

- 1920/1920 graph faithfulness；
- 256→512 中途 crossing；
- 标准 gate `126/128=98.4%`；
- negative control 能定罪；
- service structural parity 0/40 failure；
- kill flag 回到 byte-identical full-max path。

需要检查的不只是单步 logits：

- bucket selector；
- current KV write；
- persistent KV cache；
- crossing 前后 Graph identity；
- token trajectory；
- fallback。

如果只在固定 190 position 测一次，无法证明 crossing 正确。

---

## 7. Focused Graph 结果

Focused B16 decode graph：

```text
−0.6857 ms / step
−20.1%
```

这说明 max_len=2048 的尾部工作在目标 cell 中确实是 material exposed work，而不是被其他 branch 完全隐藏。

与 D022 compact launch-bound 负例不同：

- D022 只缩 nominal capacity，persistent kernel 真实 active work 不变；
- context bucket 直接缩 attention 的 KV extent，真实 load/score/reduction work 下降。

这正是“删除大数字”和“删除物理工作”的区别。

---

## 8. Serving 是否穿透？

| Metric | OFF | ON | Result |
| --- | ---: | ---: | ---: |
| C16 pooled output | 2803.6 tok/s | 3191.3 tok/s | **+13.83%** |
| Opposite-order pair 1 | — | — | +15.58% |
| Opposite-order pair 2 | — | — | +11.73% |

![Focused Decode Saving 到 C16 Serving 的穿透](/assets/blog-context-bucket-serving-evidence.svg)

*图 3：Focused −20.1% 没有完整等比例传到 Serving，因为分母还包含 prefill、scheduler、sampling 和其他 kernels；但 +13.83% 说明这条 work removal 位于可见 decode critical path。*

穿透率可粗略理解为：

$$
\mathrm{E2E\ gain}
\approx
\mathrm{decode\ share}
\times
\mathrm{focused\ reduction}
\times
\mathrm{path\ hit}
-\mathrm{selector/portfolio\ overhead}
$$

它不会等于 20.1%，但方向和量级足以通过 Serving gate。

---

## 9. 为什么 B1 Attention 以前只快约 2%，这里却快 13.83%？

不矛盾，因为 workload 不同：

- B1 与 continuous B16 的并行/分母不同；
- live context / max_len 比不同；
- fixed Graph 与 bucket portfolio 不同；
- Serving epoch 的 decode share 不同；
- path-hit 频率不同。

“同一个 attention 优化”必须绑定：

```text
batch
live context distribution
max_len
bucket policy
graph state
timing boundary
```

不能跨 cell 复用百分比。

---

## 10. Bucket 越细越好吗？

更细 bucket 减少 padding work，却增加：

- Graph 数量；
- capture memory；
- warmup/capture time；
- selector complexity；
- code/binary variants；
- correctness crossing surface；
- cache pressure；
- graph invalidation/rollback 负担。

可以粗略优化：

$$
\min_{\mathcal B}
\mathbb E[T(K_{\mathrm{selected}})]
+\lambda\cdot|\mathcal B|
$$

其中 $\mathcal B$ 是 bucket set，$\lambda$ 表示每多一张 graph 的维护/内存成本。

`256/512/1024/full` 是几何级数，能在图数量有限时把 worst padding 控制在约 2× 范围内。

---

## 11. Always-Valid Full Fallback 为什么必须保留？

当：

- position ≥1024；
- selector state 异常；
- graph leaf 未准备；
- shape/capacity/dtype miss；
- buffer identity 不匹配；

系统必须回到 K2048/full path。

Fallback 的价值：

- 正确性覆盖全 domain；
- bucket portfolio 可以小而稳；
- miss 不需要动态 capture；
- rollback 一键关闭；
- graph leaf 是优化，不是唯一生存路径。

没有 fallback，selector bug 会成为生产不可用，而不只是性能回退。

---

## 12. Bucket Selector 是控制面还是 Kernel 优化？

两者都是。

控制面负责：

- 读取 active lengths；
- 求 max position；
- 选择 graph leaf；
- 更新 graph-visible state；
- 保证 fallback。

Kernel 层获得：

- 更小 K extent；
- 更少 KV load；
- 更短 score/reduction；
- 更小 launch work。

性能来自控制面选择了正确的物理执行图，不是单个 attention kernel 源码突然变快。

---

## 13. 与 Two-Graph Serving 如何组合？

Two-graph Q5 中：

```text
Graph A: B1/T64 prefill
Graph B: B16/K256 decode
```

Graph B 的 K256 正是 context-bucket portfolio 的一个 leaf。

所以组合关系是：

```text
phase selector
  ├── prefill → Graph A / T bucket
  └── decode  → Graph B / K bucket
```

这展示了 GPU Serving control 的两个正交轴：

- Phase：prefill vs decode；
- Shape bucket：T64、K256/K512/…

不要把二者压成一个“大 Graph id”。应由一个权威 dispatcher 组合选择。

---

## 14. 什么时候 Context Bucket 值得？

适合：

- max_len 远大于常见 live context；
- attention cost 随 K 明显增长；
- fixed-max graph 尾部 work 真正执行；
- context 分布有少量高频区间；
- persistent KV tensor 可 slice；
- correctness 可证明 mask-tail 等价；
- graph portfolio 成本可控。

不适合：

- kernel 已使用 ragged/live-length 真正跳过尾部；
- attention 不是关键路径；
- live context 大多接近 max_len；
- bucket crossing/selector 成本高；
- Graph 内存预算不足；
- batch 内最长请求长期拖满 bucket。

---

## 15. 最后记住

1. **Mask 删除贡献，不自动删除 load 与 compute。**
2. **缩 tensor extent 才能可靠改变物理 attention work。**
3. **Bucket length 是 extent，position 是 index，所以必须严格 `>`。**
4. **Batch graph 由最长 active row 选择 bucket。**
5. **Focused win 只有进入 Serving A/B 才能成为产品结论。**
6. **Always-valid full fallback 让 graph portfolio 可以保持小而安全。**

Context bucket 的核心不是多捕获几张图，而是：

> **把 live workload identity 显式带入 Graph selector，让静态 replay 仍能随真实物理工作缩放。**

---

## Evidence boundary

- Evidence identity：Q4B12 / E-Q4B-012，Qwen3-4B BF16、1×B200、HTTP C16、max_len 2048。
- Graph portfolio：K256/K512/K1024/full；所有 graphs 共享 persistent KV tensor。
- Correctness：1920/1920 faithfulness（含 256→512 crossing）、gate 126/128、negative control、0/40 service structural failure、full-path kill parity。
- Performance：focused decode `−0.6857 ms/step / −20.1%`；C16 `2803.6→3191.3 tok/s / +13.83%`；opposite-order `+15.58/+11.73%`。
- 状态：qualified Qwen serve path historical `WIN/default-on`；source/public ref 尚需统一，本站不声称 fresh replay。
- 不外推到 B1、其他 max_len、其他 context distribution 或未命中 bucket 的路径。
- 状态与来源边界见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
