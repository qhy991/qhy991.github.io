---
layout: post
title: "为什么 409→15.5 ms 不是 26× Kernel 加速？从 Per-Row 假 Baseline 到 True Batched Prefill"
date: 2026-08-17 23:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Batching, Prefill, CUDA Graph, Benchmark, Baseline, Qwen3, B200]
reading_time: 29
cover_image: /assets/blog-true-batched-prefill.png
excerpt: "Qwen3-4B 的旧 B16×T128 路径用 Python loop 串行跑 16 次单请求 forward，并重复 mirror/copy 72 层 KV，耗时 409 ms；改成 N=B×T 的 true batched DAG、whole-prefill CUDA Graph 与 fused SwiGLU 后降到 15.5 ms。但公平 SGLang one_batch 是 15.03 ms，因此这不是 26× 算法领先，而是先修复错误执行结构。"
---

> 本文基于 Q4B2/E-Q4B-002 的 historical Qwen3-4B/B200 evidence。当前 Research OS 能回连 source commit 与冻结汇总，但 ledger 指向的完整 `runs/qwen3-4b-b200-batched-prefill/` 目录在本地 checkout 中缺失，原始 samples、manifest 与 hash 尚未恢复。因此本文用于解释 baseline、batch semantics 和阶段性消融，不声称 fresh replay。

一张性能表写着：

```text
409 ms → 15.5 ms
```

最容易写成：

> “新的 batched kernel 加速了 26×。”

但 409 ms 的旧路径根本不是 batched prefill。

它实际上做了：

```python
for request in batch:
    eager_forward_one_sequence(request)
    mirror_or_copy_72_layer_kv_buffers(request)
```

B16 就是 16 次串行单请求 forward。

新路径把 16×128 tokens 一次送进 batched GEMM/attention/MLP，再捕获完整 prefill Graph。

这不是把同一个成熟算法快 26×，而是把：

```text
B 次 host-controlled serial programs
```

修复成：

```text
one batched GPU program
```

公平对手 SGLang `one_batch` 在同一 B16×T128 workload 是：

```text
15.03 ms
```

OMoE 15.5 ms 仍略慢。

这篇文章要回答：

1. 什么才是 true batching？
2. 为什么 flatten `B×T` 不会让不同请求互相 attention？
3. 409→37.4→17.8→15.5 每一步到底删了什么？
4. 为什么“修复坏 baseline”和“超过成熟系统”必须分开？

---

## 1. 四种 Workload 先分开

| Workload | Shape / metric | 回答什么 |
| --- | --- | --- |
| B1 graph decode | one token step, 4K/8K, ms/token | decode DAG |
| B1 graph prefill | one prompt, 4K/8K, PP ms | long prompt path |
| True batched prefill | B16×T128/T512, B64×T128 | 多请求一次进入 batched DAG |
| HTTP C16 | dynamic continuous batch | 完整 scheduler/serving |

Prefill 的 GEMM M：

$$
M=B\times T
$$

例如：

```text
B16×T128 → M=2048
B16×T512 → M=8192
B64×T128 → M=8192
```

后两者 M 相同，但 attention 的 batch/sequence geometry 不同，所以完整 latency 不必相同。

---

## 2. 旧 Per-Row 路径为什么是假 Batched Baseline？

旧实现：

```text
Python row loop
  ├── request 0: 36-layer forward + KV copy
  ├── request 1: 36-layer forward + KV copy
  ├── ...
  └── request 15: 36-layer forward + KV copy
```

它重复支付：

- Python dispatch；
- GEMM launch；
- attention launch；
- norm/RoPE/MLP launch；
- KV mirror/copy；
- allocator/synchronization；
- cache cold/warm transitions。

记录：

| Cell | Per-row sequential |
| --- | ---: |
| B16×T128 | 409 ms |
| B16×T512 | 463 ms |
| B64×T128 | 1764 ms |

这些数字能证明旧实现结构糟糕，不能用来证明新系统比 SGLang 快 26–120×。

![Per-Row Serial 与 True Batched Prefill DAG](/assets/blog-prefill-per-row-vs-batched.svg)

*图 1：旧路径把 batch 放在 Python 控制流；新路径把 batch 放进 GEMM/attention tensor shape，让 GPU 一次处理 B×T rows。*

---

## 3. True Batched DAG 怎样构造？

```text
(B,T) token ids
  → flatten token rows N=B×T
  → one set of GEMMs over N rows
  → reshape Q/K/V to (B,H,T,D)
  → per-row causal GQA attention
  → one batched MLP path
  → lm_head at each row's real-last token
  → fixed-address KV write
  → capture whole prefill Graph
```

关键 primitives：

- cuBLAS/CUTLASS GEMM；
- cuDNN causal SDPA；
- FlashInfer/SGL norm/RoPE/SwiGLU；
- CUDA Graph replay。

---

## 4. Flatten `B×T` 会让请求互相 Attention 吗？

不会，只要 attention 恢复 batch 维：

```text
Q/K/V: [B,H,T,D]
```

每个 row $b$：

$$
\mathrm{Attention}_b
=
\mathrm{softmax}
\left(
Q_bK_b^T/\sqrt D + M_{\mathrm{causal}}
\right)V_b
$$

没有：

$$
Q_bK_{b'}^T,\quad b\ne b'
$$

Batch 维天然隔离请求；不需要显式构造一个巨大的 block-diagonal mask matrix。

Flatten 只用于 GEMM 等逐 token operation；attention 前恢复 batch identity。

---

## 5. 为什么只对 Real-Last Token 做 LM Head？

Prefill 对每行 prompt 只需要最后有效 token 的 next-token logits。

若统一 T 且没有 padding：

```text
last = T-1
```

若有真实 length：

```text
last_b = length_b - 1
```

只 gather：

$$
h_b=h[b,last_b,:]
$$

再运行 lm_head。

否则对全部 $B\times T$ hidden 做 vocabulary projection，会产生巨大无用计算和 logits materialization。

---

## 6. 四阶段消融到底删了什么？

B16×T128：

| Stage | Latency | 删除的浪费 |
| --- | ---: | --- |
| Per-row sequential | 409 ms | — |
| True batched eager | 37.4 ms | Python row loop、重复 GEMM/KV copy |
| Whole-prefill Graph | 17.8 ms | 每次 host submission / launch gap |
| Fused SwiGLU | 15.5 ms | SiLU、multiply 与 BF16 中间往返 |

![409→37.4→17.8→15.5 的结构修复阶梯](/assets/blog-prefill-structure-ladder.svg)

*图 2：第一步是 execution-model 修复，第二步是 orchestration capture，第三步才是局部 kernel/dataflow fusion。不能把总比值归因给最后一个 kernel。*

三个相邻比值：

$$
409/37.4\approx10.94\times
$$

$$
37.4/17.8\approx2.10\times
$$

$$
17.8/15.5\approx1.15\times
$$

最大收益来自去掉错误的 serial execution，而不是单 kernel 优化。

---

## 7. CUDA Graph 在这里删除了什么？

True batched eager 已经使用正确 GPU math，但每次仍由 host 发起完整 prefill DAG。

Graph capture：

```text
many Python/C++ launches per prefill
      ↓
update fixed buffers
one cudaGraphExec replay
```

所以：

```text
37.4 → 17.8 ms
```

这不是 child kernels 消失；它们仍在 device 上执行。Graph 删除 host orchestration 和 submission gaps。

因此 Graph 收益只有在 host boundary 暴露时大；如果外层已经 Graph 化，减少几百个 child nodes 可能只快 1%。

---

## 8. Fused SwiGLU 为什么只贡献最后 2.3 ms？

旧 MLP：

```text
gate/up GEMM
  → SiLU kernel
  → multiply kernel
  → BF16 hidden write/read
  → down GEMM
```

Fusion：

```text
gate/up
  → fused SwiGLU
  → down
```

它删除：

- 一个 launch；
- 中间 BF16 tensor 部分往返；
- separated activation work。

但 GEMM、attention、norm、RoPE、KV store仍存在，所以：

```text
17.8 → 15.5 ms
```

是合理的局部穿透，不应获得“26×”标签。

---

## 9. 公平 SGLang Board 说了什么？

| Cell | OMoE true batched | SGLang one_batch | Verdict |
| --- | ---: | ---: | --- |
| B16×T128 BF16 | 15.5 ms | 15.03 ms | OMoE ≈1.03× slower |
| B16×T512 BF16 | 59.5 ms | 50.89 ms | OMoE ≈1.17× slower |
| B64×T128 BF16 | 60.0 ms | 128.08 ms | OMoE ≈2.13× faster |

![True-Batched OMoE 与 Same-Workload SGLang Board](/assets/blog-prefill-fair-board.svg)

*图 3：修复 per-row baseline 后，真正的系统位置依赖 B/T shape；不能只展示 B64 winner，也不能用 409 ms 对比 SGLang 15.03 ms。*

这说明：

- B16 主 cells 当时尚未达到完整 parity；
- B64×T128 的 shape 对 OMoE 有利；
- 不能以单一 shape 宣布系统普遍领先；
- 后续优化应针对 B16×T512 的真实非 GEMM gap。

---

## 10. M 相同，为什么 B16×T512 与 B64×T128 不同？

两者：

$$
M=B\times T=8192
$$

GEMM shape 相近，但 attention 不同：

### B16×T512

- 16 sequences；
- 每条 causal matrix 512×512；
- 更长 per-row context；
- KV/attention work 大。

### B64×T128

- 64 sequences；
- 每条 128×128；
- 更高 batch parallelism；
- 更短 sequence attention。

Attention complexity近似：

$$
O(BT^2D)
$$

在固定 $BT$ 下，仍随 $T$ 增长。

所以 M bucket 只够选择 GEMM，不够描述 whole-prefill workload。

---

## 11. Profile 为什么把优化方向转向 Non-GEMM Edge？

B16×T512：

```text
whole prefill ≈59.5 ms
GEMM ≈37 ms
measured ≈59.5 TFLOP
≈1608 TF/s
```

当时 B200 BF16 anchor 约：

```text
1622 TF/s
```

GEMM 已接近该 anchor。

剩余约 20 ms 来自：

- activation；
- norm；
- SDPA；
- RoPE；
- KV copy；
- layout materialization；
- graph/dataflow seams。

继续只换 GEMM kernel 的上限有限；后续 split/merged QKV、QKNorm+RoPE+store、FP8/DeepGEMM selector 都来自这个 profile conclusion。

---

## 12. Correctness 如何证明 Batch Row 独立？

冻结 checks：

- Standard gate 128/128；
- Stage-3 B=4 identical-row path 实际走新 batched forward；
- Negative control 降到 14.1%，能定罪；
- Decode 保持 2.79 ms/token；
- B1 prefill 64/128/256 无回退。

Identical-row test：

```text
复制同一 prompt 到多个 batch rows
→ 每行输出应与单请求 reference 一致
```

它能发现：

- batch stride 错；
- row attention 混合；
- KV slot collision；
- last-token gather 错；
- padding/mask 错。

Negative control 证明 gate 不是无区分力的“总通过”。

---

## 13. 怎样定义一个公平 Baseline？

Baseline 必须与 candidate 共享：

- 同一请求集合；
- 同一 B/T；
- 同一 precision；
- 同一模型/权重；
- 同一 attention semantics；
- 同一 output token positions；
- 相同 Graph/eager boundary，或明确将 orchestration 作为 treatment；
- 相同 correctness gate。

如果 baseline 用 Python row loop，candidate 用 batched GPU program，比较回答的是：

> “修复执行结构能省多少？”

而不是：

> “我的 kernel 比成熟 batched engine 快多少？”

两种问题都可以测，但标题和结论必须匹配。

---

## 14. 为什么状态是 MERGED-PARTIAL / STOP？

True batching 是有效结构修复：

- 正确性通过；
- 执行语义正确；
- 相对旧路径巨大改善；
- 部分 shape 达到/超过 SGLang。

但完整目标尚未关闭：

- B16×T128 略慢；
- B16×T512 慢约 17%；
- 只有 B64×T128 明显领先；
- raw run bundle 仍有 evidence debt。

所以不能将 B64 winner 或 409→15.5 修复包装成全局 WIN。

---

## 15. 一个通用的 Batching Audit

看到 `batch_size > 1` 时，检查：

1. Python 是否仍 loop rows？
2. GEMM M 是否真的为 B×T？
3. Attention 是否有显式/隐式 batch dimension？
4. 不同 rows 是否共享错误 KV/position？
5. LM head 是否只算 real-last token？
6. KV write 是否 batched/fixed-address？
7. CUDA Graph 是否捕获整个 batch DAG？
8. Baseline 是否也是 one-batch execution？

只有全部回答，`B=16` 才不只是 API 参数。

---

## 16. 最后记住

1. **Batch 参数不等于 batched execution。**
2. **Python row loop 是 B 次串行程序，不是一个 batched DAG。**
3. **409→15.5 ms 是执行结构修复，不是成熟 kernel 快 26×。**
4. **公平结论来自同 workload one_batch board。**
5. **M=B×T 只描述 GEMM；attention 仍依赖 B 与 T 的分解。**
6. **修复 batching 后，profile 才能暴露真正值得优化的 non-GEMM edges。**

真正的科学叙事不是：

> “我们快了 26×。”

而是：

> **“旧路径没有 batch；修复后达到成熟 batched baseline 的同一量级，并在不同 B/T cells 上暴露了后续真正的差距。”**

---

## Evidence boundary

- Q4B2/E-Q4B-002：Qwen3-4B BF16、1×B200 true batched prefill；source commit 可回连，完整 run directory/raw/manifest/hash 当前缺失，`EVIDENCE_DEBT`。
- B16T128 staged numbers：409→37.4→17.8→15.5 ms；仅表示 serial→batched→Graph→fusion 的结构消融。
- Same-workload board：B16T128 15.5 vs SGLang 15.03；B16T512 59.5 vs 50.89；B64T128 60.0 vs 128.08。
- Correctness：gate 128/128、B4 identical-row path、negative control 14.1%、decode/B1 guardrails。
- 状态：historical `MERGED-PARTIAL / STOP`；不是当前主分支 fresh result，也不是 cross-workload 26× claim。
- 不外推到 FP8、HTTP Serving、B1 decode 或其他 GPU。
- 状态与 evidence debt 见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
