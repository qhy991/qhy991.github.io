---
layout: post
title: "为什么 QKV 先拆开更快，后来合并又更快？从 View、Stride 到 Consumer Contract"
date: 2026-08-17 21:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [QKV, Tensor Layout, Stride, Contiguous, GEMM, Prefill, Qwen3, B200]
reading_time: 28
cover_image: /assets/blog-qkv-consumer-contract.png
excerpt: "同一个 Qwen3-4B batched-prefill shape，07-17 split Q/K/V 因消除 FlashInfer RMSNorm 的隐式 contiguous copy 而快 5.4%；07-20 换成 stride-generic SGL consumer 后，merged QKV 又因少两次 GEMM launch 而快 1.41%–3.64%。本文解释为什么 view 本身免费，layout 成本却由下游 consumer 决定。"
---

> 本文基于 Q4B3/E-Q4B-003 与 Q4B4/E-Q4B-004 两个历史 Qwen3-4B/B200 epoch。Q4B3 的 source lineage 可达，但 ledger 所指完整 run directory 当前缺失；Q4B4 有 source/merge/artifact 与 RESULT，但缺独立 raw sample/manifest。文章只使用已冻结汇总来解释机制反转，不声称 fresh replay，也不把两个 epoch 拼成同一 waterfall。

矩阵乘法上，Q、K、V 可以合并：

$$
[Q,K,V]
=
X
\begin{bmatrix}
W_Q\\W_K\\W_V
\end{bmatrix}^{T}
$$

也可以拆成三次：

$$
Q=XW_Q^T,\quad
K=XW_K^T,\quad
V=XW_V^T
$$

通常一个大 GEMM 比三个小 GEMM 更有吸引力：

- launch 少；
- M/K 相同；
- cuBLAS/DeepGEMM 更容易获得大问题规模；
- 中间调度更简单。

但 Qwen3-4B 的 batched prefill 曾出现：

```text
07-17: split Q/K/V 更快
07-20: merged QKV 又更快
```

两次结论都正确。

变化的不是数学，也不是 GPU，而是 downstream consumer 的 stride contract。

---

## 1. QKV 的实际宽度与 Layout

输出宽度大致：

```text
Q = 4096
K = 1024
V = 1024
merged N = 6144
```

Merged GEMM 产生：

```text
Y[M, 6144]
```

然后 slice：

```python
q = y[..., :4096]
k = y[..., 4096:5120]
v = y[..., 5120:6144]
```

Slice 通常只是 view：

- 不分配新 storage；
- 不移动数据；
- 改 shape / offset / stride。

但 q 的 row stride 仍是：

```text
6144
```

不是 compact q tensor 的：

```text
4096
```

K/V 同理。

![Merged QKV View 的 Stride 与潜在 Contiguous Copy](/assets/blog-qkv-view-stride.svg)

*图 1：Slice/view 本身零拷贝，但每行 Q/K/V 之间仍隔着 merged row 的其他分量。Consumer 若要求 compact `(-1,D)`，必须 materialize。*

---

## 2. 为什么 View 免费，Reshape 可能不免费？

PyTorch 中：

```python
view(...)
reshape(...)
contiguous()
```

语义不同。

### `view`

只在现有 stride 可表达目标 shape 时成功，不复制。

### `reshape`

优先返回 view；如果 stride 不能表达，就隐式分配并复制。

### `contiguous`

显式生成 compact storage。

因此下面一行：

```python
q.reshape(-1, head_dim)
```

源码看不到 `.contiguous()`，运行时仍可能出现 copy。

判断方式：

- 检查 `stride()`；
- 检查 `storage_offset()`；
- 比较 `data_ptr()` / storage identity；
- profiler 查 `copy_` / contiguous kernel；
- 对 producer→consumer 整段计时，而不是只测 slice。

---

## 3. 07-17：FlashInfer Consumer 为什么让 Split 获胜？

旧路径：

```text
one merged QKV GEMM
  → strided Q/K/V views
  → FlashInfer fused RMSNorm reshape(-1,D)
  → implicit per-head contiguous copy
```

在 B16×T512，copy bucket：

```text
5.86 ms
```

它是当时最大的 non-GEMM kernel bucket。

Candidate 将 row-concatenated weights 切成三个连续 weight views，并做三次 GEMM：

```text
Q GEMM → compact Q → QNorm
K GEMM → compact K → KNorm
V GEMM → compact V
```

虽然多了两个 GEMM launch，却大幅减少 layout materialization。

| Cell | Merged + strided consumer | Split Q/K/V | Result |
| --- | ---: | ---: | ---: |
| B16×T512 | 59.8 ms | 56.5 ms | −5.4% |
| B64×T128 | 60.3 ms | 56.7 ms | same direction |

Copy bucket：

```text
5.86 → 1.59 ms
```

GEMM decomposition bit-identical，gate `126/128=98.4%` PASS。

所以当时的成本模型是：

$$
T_{\mathrm{merged}}
=T_{\mathrm{GEMM1}}
+T_{\mathrm{copy}}
$$

$$
T_{\mathrm{split}}
=T_{\mathrm{GEMM_Q}}
+T_{\mathrm{GEMM_K}}
+T_{\mathrm{GEMM_V}}
$$

当：

$$
T_{\mathrm{copy}}
>
T_{\mathrm{split\ overhead}}
$$

Split 获胜。

---

## 4. 07-20：Consumer 改变后，为什么方向翻转？

后来 batched prelude 换成 SGL JIT QNorm/KNorm/RoPE consumer。

新 consumer 接受独立 symbolic strides：

```text
q row stride = 6144
k row stride = 6144
v row stride = 6144
```

它可以直接从 merged buffer 读取所需 head elements，不要求 `.contiguous()`。

于是 merged 路径变成：

```text
one merged QKV GEMM
  → free strided views
  → stride-generic Q/K/RoPE consumer
```

Copy 消失后，三个 GEMM 的额外 launch/调度反而成为成本。

| Cell | Split + old prelude | SGL add-norm + merged | Result |
| --- | ---: | ---: | ---: |
| B16×T512 | 50.834 ms | 50.115 ms | −0.719 ms / −1.41% |
| B16×T128 | 13.747 ms | 13.246 ms | −0.501 ms / −3.64% |

五轮 candidate 全胜；stage-4 `506/512=98.8%`，gate/control 通过。

状态仍是 `MERGED-PARTIAL`，因为绝对 `≤49.5 ms` bar 未过。

![Consumer Contract 改变导致 QKV 优化方向翻转](/assets/blog-qkv-consumer-sign-flip.svg)

*图 2：Producer 和数学不变；旧 consumer 把 strided view 变成昂贵 copy，新 consumer 直接消费 stride，merged GEMM 的少 launch 优势才重新暴露。*

---

## 5. 最关键的 Micro Counterfactual

```text
merged, no copy      0.1615 ms
split                0.1677 ms
merged + contiguous  0.2783 ms
```

三个 arm 分别回答：

### Merged / no copy

理想 stride-generic consumer，保留一个大 GEMM。

### Split

三个 compact outputs，多两个 GEMM launch。

### Merged + contiguous

一个大 GEMM，但 consumer 强制 materialize。

![Merged、Split 与 Forced-Contiguous Micro Counterfactual](/assets/blog-qkv-micro-counterfactual.svg)

*图 3：Forced copy 比 merged→split 的纯 GEMM 差异大约贵一个数量级。优化 QKV 前，应先定位 consumer 是否 materialize。*

数值：

$$
0.2783-0.1615=0.1168\ \mathrm{ms}
$$

而 merged/no-copy 相对 split 只省：

$$
0.1677-0.1615=0.0062\ \mathrm{ms}
$$

Copy 成本约是 pure merge advantage 的：

$$
\frac{0.1168}{0.0062}\approx18.8\times
$$

冻结文档以“约 13 倍量级”描述不同测量口径下的工程关系；无论采用哪种局部定义，结论相同：materialization 远大于 GEMM launch 差异。

---

## 6. 为什么不能只优化 Producer？

传统 kernel tuning 问：

```text
QKV GEMM 本身哪个更快？
```

真正应该测：

```text
producer
  → view/layout
  → norm/RoPE
  → KV store
  → attention consumer
```

Producer 输出 layout 的价值由下游解释。

同一 strided view 对不同 consumer：

| Consumer | Stride support | View cost |
| --- | --- | --- |
| FlashInfer reshape-to-compact RMSNorm | requires compact | implicit copy |
| SGL stride-generic prelude | symbolic stride | zero-copy view |
| Store kernel requiring compact K/V | compact-only | copy before store |
| Fused RoPE+KV-store accepting stride | stride-aware | direct consume |

所以 layout verdict 属于边，不属于 tensor 名称。

---

## 7. `Q is Contiguous` 到底是什么意思？

“Contiguous”必须带维度和 consumer 语义。

Merged Q slice 可能满足：

```text
last dimension contiguous
```

但不满足：

```text
flattened token-head rows compact
```

例如 shape：

```text
[M, 4096]
```

stride：

```text
[6144, 1]
```

最后一维连续，但行间有 2048 元素 gap。

Consumer 若每次按 `(row, col)` 访问，stride-aware 很容易；若想把它展平成连续 `[M*heads, dim]`，可能必须 copy。

---

## 8. 为什么 Consumer 支持 Stride 也可能有成本？

Stride-generic 不等于永远免费：

- 地址计算更复杂；
- vector alignment 可能下降；
- load coalescing 依赖 slice offset；
- TMA descriptor 需要合法 stride；
- 某些 tile/cluster tactic 不支持；
- register/IMAD 指令可能增加。

所以必须做完整 A/B。

07-20 的结果说明在该 B16 prefill cell 中，stride-generic 成本小于删除两个 GEMM launch 的收益；不代表所有 M、GPU、consumer 都如此。

---

## 9. cuBLAS Tactic 也在反转中扮演角色

合并后的：

```text
N = 6144
```

cuBLAS 选择 `2×1` CTA cluster；窄 Q GEMM 曾使用 `2×2`。

这吃掉了部分理论 merge 收益。

说明 producer composition 还会改变：

- GEMM tile；
- cluster shape；
- split-K；
- waves；
- epilogue；

不能只用“1 launch vs 3 launches”建模。

---

## 10. Sign Flip 不是旧结论被推翻

07-17 的命题：

> 在 FlashInfer compact-only RMSNorm consumer 下，split outputs 删除 implicit copy，因此更快。

07-20 的命题：

> 在 SGL stride-generic consumer 下，copy 消失，一个 merged GEMM 比三个 split GEMMs 更快。

前提不同：

```text
consumer stride contract changed
```

所以两个结论可以同时成立。

真正错误的是把结论保存成：

```text
split QKV is always faster
```

或者：

```text
merged QKV is always faster
```

正确 oracle 是：

```text
winner = f(M, producer tactic, output stride, consumer support, copy behavior)
```

---

## 11. 怎样发现隐式 Contiguous Copy？

### Static inspection

找：

```python
.contiguous()
.reshape(...)
.view(...)
flatten
transpose/permute 后进入 compact-only custom op
```

### Runtime tensor audit

记录：

```python
tensor.shape
tensor.stride()
tensor.storage_offset()
tensor.is_contiguous()
tensor.data_ptr()
```

### Profiler

查：

```text
aten::copy_
aten::contiguous
copy kernel
layout conversion
unexpected materialization
```

### Counterfactual

至少测：

```text
merged + current consumer
split + current consumer
merged + forced copy
merged + stride-generic consumer
```

这样才能把 GEMM composition 与 layout copy 分开。

---

## 12. 为什么只看 Kernel Count 会误导？

07-17：

```text
1 GEMM + copy
vs
3 GEMMs + no copy
```

Kernel 更多的 split 更快。

07-20：

```text
1 GEMM + no copy
vs
3 GEMMs + no copy
```

Kernel 更少的 merged 更快。

真正删除的不是“两个 kernel”这个抽象数字，而是：

```text
copy edge
or
extra GEMM launches
```

必须在完整 producer-consumer DAG 上标出每条物理边。

---

## 13. 如何设计 Layout-Aware Kernel Registry？

不要只按：

```text
op = qkv_projection
M / N / K
```

还应绑定：

```text
producer composition: merged | split
output stride/layout
consumer id
consumer stride capability
materialization requirement
dtype / phase / M bucket
GEMM tactic / cluster
```

一个 winner record 可以写成：

```yaml
cell:
  phase: batched_prefill
  M: 8192
  producer: merged_qkv
  output_stride: [6144, 1]
  consumer: sgl_stride_generic_qknorm_rope
  materializes: false
verdict: merged
```

换 consumer 后必须创建新 cell。

---

## 14. Evidence Debt 为什么重要？

07-17 split-QKV：

- source lineage 可达；
- 汇总数字在 matrix；
- 但 ledger 指向的 run directory 当前缺失；
- 没有独立 raw/manifest/hash。

07-20 merged flip：

- source / merge / artifact refs 存在；
- RESULT 存在；
- paired 数字存在；
- 但仍缺独立 raw sample bundle 与 clean/public binding。

因此本文可以支持：

> consumer contract 导致 sign flip 的历史机制。

不能支持：

> 本轮已经在当前 main 上 fresh 复现同样百分比。

---

## 15. 最后记住

1. **View 本身可能免费，consumer reshape 不一定免费。**
2. **Tensor 是否“好布局”由 consumer 决定。**
3. **Split 可以用更多 GEMM 换掉更昂贵的 layout copy。**
4. **Consumer 变成 stride-generic 后，merged 的少 launch 优势会重新出现。**
5. **Sign flip 表示前提改变，不表示旧实验错误。**

优化 QKV 时不要问：

> “合并还是拆分？”

而要问：

> **“当前 producer layout 到当前 consumer 的整条边，会不会 materialize；如果不 materialize，哪个 GEMM composition 在这个 M/tactic cell 更好？”**

---

## Evidence boundary

- Q4B3/E-Q4B-003：2026-07-17 BF16 batched prefill，FlashInfer compact consumer；B16T512 `59.8→56.5 ms`、B64T128 `60.3→56.7`、copy `5.86→1.59 ms`、gate 126/128。Source 可达，完整 run/raw bundle 当前缺失，`EVIDENCE_DEBT`。
- Q4B4/E-Q4B-004：2026-07-20 SGL stride-generic consumer；B16T512 `50.834→50.115 ms`、B16T128 `13.747→13.246`，五轮同向，stage-4 506/512；`MERGED-PARTIAL`，未过绝对 49.5 ms bar。
- Micro counterfactual：merged/no-copy 0.1615 ms、split 0.1677 ms、merged+copy 0.2783 ms；仅用于机制归因。
- 两个 epoch baseline 不同，不能串成 waterfall，也不代表当前 main 的 fresh result。
- 不外推到 B1 decode、FP8、其他 consumer、GPU 或 M bucket。
- 状态与 provenance debt 见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
