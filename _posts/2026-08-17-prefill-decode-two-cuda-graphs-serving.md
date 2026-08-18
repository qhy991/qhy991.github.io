---
layout: post
title: "为什么一张 CUDA Graph 不够？Prefill / Decode 两套执行合同如何让 Serving 快 51.2%"
date: 2026-08-17 19:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [CUDA Graph, Prefill, Decode, Serving, Dynamic Shape, Qwen3, B200, Inference]
reading_time: 31
cover_image: /assets/blog-two-graph-serving.png
excerpt: "Qwen3-4B 的固定 C16 场景中，Graph A 捕获 B1/T64 admission-prefill，Graph B 复用 B16/K256 decode；固定地址、device-owned length/slot、append-only padding 与 real-last logits 让动态内容进入静态 DAG。内部 A/B 从 3078.559 提升到 4654.812 tok/s，但仍不能与旧 SGLang denominator 跨 epoch 拼接。"
---

> 本文来自 `agentic-megakernel@fdf4898` 中登记为 `C2/S4 migration_pending` 的 Qwen two-graph lineage，以及 research evidence registry 的 Q5/E-Q4B-014 冻结记录。它有完整 internal OFF/ON 两顺序、gate、rollback 与 reproducibility 证据，因此可以支持 exact-scenario `WIN/default-on`；但源码/worker/raw bundle 尚未迁成公开独立 experiment package，同 epoch SGLang arm 也未补齐。本文不声称 fresh replay，也不发布跨系统排名。

CUDA Graph 的基本要求是：

> Capture 时 kernel DAG、参数地址和大部分执行形态稳定，replay 时只更新内容。

LLM Serving 恰好充满动态性：

- Prompt 长度不同；
- Prefill 的 token 数不同；
- Decode batch 随请求到达/结束变化；
- KV context 不断增长；
- Slot 分配变化；
- Admission、prefill 和 decode 属于不同 phase；
- 不同 phase 的最优 kernel、shape 与 memory footprint 完全不同。

最直接的方案是：

> “捕获一张足够大的 universal Graph，所有请求都 pad 到最大 shape。”

但这样可能把动态控制成本变成持续 GPU 浪费；也可能因为 prefill/decode 的执行合同不同，根本无法用一张图自然表达。

Qwen3-4B 的一个固定 C16 Serving 场景最终采用两张图：

```text
Graph A: singleton prompt admission / prefill
Graph B: continuous-batch decode
```

内部 OFF/ON A/B：

```text
3078.559 → 4654.812 output tok/s
+51.20%
```

这个数字很大，但真正值得学习的不是“多捕获一张 Graph”，而是：

> **把动态服务拆成两个语义稳定的 phase leaf，每张图只冻结自己能够诚实表达的执行合同。**

---

## 1. 为什么 Prefill 和 Decode 不是同一类 Workload？

### Prefill / Admission

输入是新 prompt，特点是：

- $T>1$；
- GEMM 的 $M$ 较大；
- 要写入一段 KV；
- 需要选择新 slot；
- 最后只关心每行真实最后一个 token 的 logits；
- prompt 长度变化明显。

### Decode

每个 active request 每步通常新增一个 token：

- rank-local / batch shape 相对稳定；
- KV length 增长；
- attention 工作随 live context 变化；
- 多个请求在 capacity 内交错；
- kernel registry 使用不同的小-$M$ bucket。

将两者塞进同一 graph，往往需要：

- 更大 padding；
- 更复杂 mask；
- 统一但次优的 kernel；
- 更多 inactive rows；
- 更难维护的 slot/length side effects。

Two-graph 的第一性原理不是“Graph 数量=2”，而是：

$$
\text{one graph}
\leftrightarrow
\text{one stable phase contract}
$$

---

## 2. 旧 C16 Wall 到底花在哪里？

前一轮 profile 将 C16 wall 粗分为：

```text
decode replay ≈ 47.5%
eager prefill/admission ≈ 52.5%
```

70.8% 的 admit groups 是 singleton `G=1`。

这意味着继续抠 B1 decode kernel，即使很成功，也无法直接消除一半以上的 admission/prefill/host submission cost。

早期 QS5 尝试把 singleton admit 路由到 fast-prefill seam：中心效应达到 `+25.997%`，但 candidate median drift `10.195% > 5%`，因此 `STOP_REPRODUCIBILITY`。

QS7 没再微调一个 kernel，而是捕获整个固定 admit boundary。

---

## 3. Graph A：B1 / T64 Admission-Prefill

历史 prompt 长度范围是 1–62，因此选择：

```text
B = 1
T_bucket = 64
```

Graph A 包含：

- 固定地址 input/token buffers；
- device-owned real length；
- device-owned slot id；
- append-only padding；
- prefill model kernels；
- persistent KV writes；
- 从 real-last position gather logits。

关键不是把 pad token 当真实 prompt 参与因果位置。

真实 token 放在前面：

```text
[real prompt tokens][padding]
```

位置、length 和 mask 仍由真实长度控制；首个生成 token 的 logits 从：

```text
real_last = real_length - 1
```

读取。

只要：

```text
real_length ≤ 64
```

Graph A 的形状稳定，而语义仍属于真实 prompt。

---

## 4. Graph B：B16 / K256 Decode

Graph B 复用已有 decode graph：

```text
capacity = 16
live-context bucket = 256
```

更大的 live context 由 512/1024/full-2048 等其他 qualified bucket/fallback 处理；Q5 exact scenario 中主要使用 K256 leaf。

每步：

- 更新 active rows 内容；
- 更新 device lengths；
- 更新 KV slot；
- 保持 graph buffer 地址不变；
- replay 捕获的 decode DAG。

![Graph A Prefill、Graph B Decode 与 Fail-Closed Fallback](/assets/blog-two-graph-architecture.svg)

*图 1：两张图不是复制整个模型，而是各自拥有一个稳定 phase contract；任何 shape、slot、dtype 或依赖 miss 都回到旧 eager/other-bucket 路径。*

---

## 5. 固定地址怎样承载动态请求？

CUDA Graph 要求地址稳定，不要求数据内容不变。

概念上：

```text
capture once:
  input_ptr  = persistent_input_buffer
  length_ptr = persistent_length_buffer
  slot_ptr   = persistent_slot_buffer
  kv_ptr     = persistent_kv_cache

per request:
  copy new tokens into input buffer
  update real length on device-visible memory
  update slot id
  replay graph
```

![固定 Graph 地址与动态 Content/Length/Slot](/assets/blog-two-graph-static-address.svg)

*图 2：Graph identity 由稳定指针和 DAG 决定；请求内容、真实长度和 slot 可以在 replay 前原地更新。Append-only padding 与 real-last gather 保持 prompt 语义。*

这是一种重要的控制面转换：

```text
Host 每次构造/launch DAG
        ↓
Host 更新小型状态 + replay 固定 DAG
```

Graph replay 删除的是 submission/Python/synchronization boundary，不是把所有 device kernels 融成一个 kernel。

---

## 6. 为什么 Graph 不是 Kernel Fusion？

CUDA Graph replay 后，device 仍执行所有 child kernels。

早期 QK layer fuse 将 nodes/token：

```text
1137 → 769
```

但 full Graph：

```text
2.80 → 2.76 ms/token
```

只快约 1%，因为 GEMM 和其他 device work 仍然主导。

Two-graph 的大收益来自另一个层级：

```text
每次 singleton admit:
Python / eager orchestration
many host submissions
sync / shape setup
dynamic boundary
        ↓
one prepared Graph A replay
```

因此 Graph A admit wall：

```text
37.555 → 3.430 ms
```

它删除的是一个粗粒度、反复暴露的控制边界。

---

## 7. Exact Scenario Contract 有多窄？

Auto default-on 只覆盖：

```text
model       = Qwen3
dtype       = BF16
GPU         = one B200
capacity    = 16
max_len     = 2048
prompt      = singleton and fits T64 envelope
buffers     = graph-stable
slot        = valid
dependency  = valid
```

任何一项不满足：

```text
fail closed
→ previous generic eager / other-bucket path
```

这叫 scenario leaf。

它不是一个无限增长的 graph table，也不是声称“Qwen 所有 serving 都适用”。

窄合同反而让：

- correctness 可冻结；
- path hit 可证明；
- rollback 清晰；
- Graph 地址/shape 可稳定；
- 性能结论可复现。

---

## 8. 正确性如何覆盖 Graph Side Effects？

Graph A/B 不只返回 logits。它们还改变：

- persistent KV cache；
- lengths；
- slots；
- active-row state；
- request progression。

冻结门包括：

- `omoe.gate 126/128 = 98.4%` PASS；
- negative control 能定罪；
- accuracy/path checks；
- service guard；
- B1 prefill/decode guardrail；
- contract miss rollback；
- fixed-scenario output behavior。

如果只比较最后 token，而不检查 slot/KV/length contract，Graph 可能在短序列上看似正确，却污染后续 replay。

---

## 9. Internal A/B 结果

| Metric | Rollback | Two-graph | Result |
| --- | ---: | ---: | ---: |
| Pool-weighted admit wall | 37.555 ms | 3.430 ms | 主要 submission gap 被删除 |
| C16 pooled output | 3078.559 tok/s | 4654.812 tok/s | **+51.20%** |
| Opposite-order pair 1 | — | — | +50.25% |
| Opposite-order pair 2 | — | — | +51.70% |
| Rollback reproducibility | — | — | 1.49% |
| Candidate reproducibility | — | — | 0.54% |

![Two-Graph Internal A/B 与外部证据边界](/assets/blog-two-graph-evidence-boundary.svg)

*图 3：内部 OFF/ON 有两顺序、低 drift 和正确性门，可支持 exact-scenario promotion；但它没有同 epoch SGLang arm，不能回答当前跨系统排名。*

相比 QS5 的 candidate drift `10.195%`，QS7 直接捕获整个 admission boundary，使 candidate 自身 drift 降到 0.54%。

这说明 Graph 的价值还包括减少测量中由 Python/submission 产生的不稳定性。

---

## 10. 为什么 +51.20% 不能和旧 SGLang Board 拼接？

2026-07-26 旧 C16 cross-system board：

| Metric | OMoE | SGLang 0.5.15 |
| --- | ---: | ---: |
| Output tok/s | 2882.1 | 5579.2 |
| TPOT p50 | 4.237 ms | 2.495 ms |
| TTFT p50 | 144 ms | 43 ms |

旧 OMoE/SGLang：

```text
0.517×
```

Two-graph 是 2026-07-28 的内部 OFF/ON：

```text
3078.559 → 4654.812
```

把新 numerator 除以旧 denominator：

```text
4654.812 / 5579.2 ≈ 0.834
```

只是算术，不是合法实验，因为：

- 不同日期/epoch；
- 不同 code tip；
- 不同 server process；
- 旧 board 的 thinking/chat 语义没有完全对齐；
- 没有同一组请求与 drift control。

正确表述：

1. 旧 board 证明当时 OMoE C16 明显落后，gap 主要在 admission/prefill/host；
2. Two-graph internal A/B 证明 fixed scenario 大幅修复这个结构问题；
3. 是否达到或超过 SGLang，必须在 two-graph tip 上重新跑 same-epoch external arm。

---

## 11. 为什么不继续深挖 Graph A 里的小 Kernel？

Graph A 优化后剩余完整 replay：

```text
3.430 ms / admit
```

按 singleton admission 比例投影，删除 Graph A 剩余全部 GPU work，对 served wall 的上限只有：

```text
3.53%
```

低于预注册 5% dispatch razor。

Direct KV store 等更窄 seam 的 ceiling 更低。

因此 QS8 被 `PARKED_BEFORE_IMPLEMENTATION`：

```text
不继续做 QK/store micro fusion
除非 workload 比例或 measured self-time 改变
```

这是很重要的研究纪律：大边界已被 Graph capture 删除后，剩余 kernel 优化的系统价值会重新排序。

---

## 12. 为什么不捕获更多 Graph？

Graph portfolio 越大，覆盖越多，但也增加：

- capture time；
- memory footprint；
- selector state；
- correctness surface；
- graph invalidation；
- binary/kernel variant 数；
- fallback/rollback 复杂度。

合理设计不是为每个 prompt length 捕获一张图，而是选择少量语义稳定 bucket：

```text
T64 prefill leaf
K256/K512/K1024/full decode leaves
always-valid fallback
```

每个 leaf 要有：

- 清楚 owner；
- path counter；
- exact/fidelity gate；
- graph-stable buffers；
- miss fallback；
- 独立性能 evidence。

---

## 13. Two-Graph Selector 应由谁拥有？

Selector 需要同时知道：

- request phase；
- real prompt length；
- current live context；
- active capacity；
- slot validity；
- buffer identity；
- graph availability；
- dtype/model/GPU contract。

这些事实分散在 scheduler、KV manager、model runner 和 graph cache 中。

正确的 SSOT 通常是一个窄的 phase dispatcher：

```text
if exact Graph-A contract:
    update device state
    replay Graph A
elif exact Graph-B bucket:
    update device state
    replay Graph B
else:
    fallback
```

不能让多个模块各自猜测“应该用哪张 graph”，否则 path identity 不可审计。

---

## 14. Two-Graph 与 Megakernel 的关系

两者解决不同层级：

### CUDA Graph

- Host submission；
- Python/control overhead；
- 固定 kernel DAG replay；
- phase/bucket specialization。

### Megakernel

- Device-resident task scheduling；
- tile-level dependency；
- persistent execution；
- 更细粒度 overlap/locality。

如果瓶颈是每次 admission 都重建/提交几十个 kernel DAG，Graph capture 往往先提供更低复杂度的收益。

只有当固定 Graph 内仍有：

- 关键路径 tile pipeline；
- CPU round trip；
- 无法表达的 device-dynamic scheduling；

Megakernel/ready queue 才可能值得。

---

## 15. 什么时候 Two-Graph 模式可迁移？

适合：

1. Prefill/decode phase shape 明显不同；
2. 每个 phase 内存在少量稳定 bucket；
3. 地址可 persistent；
4. 动态状态可通过 device buffers 更新；
5. Padding/mask 有清晰 exact contract；
6. Admission/control overhead 在 served wall 中暴露；
7. Contract miss 能 fail closed。

不适合：

- shape 组合爆炸；
- buffer 地址频繁重建；
- 每请求 control flow 完全不同；
- capture memory 过高；
- eager path 已不是瓶颈；
- graph 内剩余 GPU work 才是真正主导。

---

## 16. 最后记住

1. **Prefill 和 decode 是两个执行合同，不只是同模型的两种模式。**
2. **Graph 固定地址，不固定请求内容。**
3. **Append-only padding 必须配合真实 length/position 与 real-last logits。**
4. **CUDA Graph 删除 submission，不等于 kernel fusion。**
5. **Internal A/B 不能与旧 external denominator 跨 epoch 拼接。**
6. **大边界删除后，要重新做 headroom audit，及时停止 leaf micro-tuning。**

Two-graph 最值得借鉴的不是数字 `51.20%`，而是设计方法：

> **把动态系统拆成少量可冻结、可回滚、可证明 path identity 的 phase leaves，而不是强迫一张 universal Graph 承担所有变化。**

---

## Evidence boundary

- Concept source：`agentic-megakernel@fdf4898` 的 `qwen-two-graph-serving`，当前 `C2/S4 migration_pending`。
- Evidence identity：research registry Q5 / E-Q4B-014；base `932c9f5`、worker `a2ef683`、archive `d6d07d0`、record `65e2c7b`。
- Frozen cell：Qwen3-4B BF16、1×B200、HTTP capacity 16、max_len 2048、singleton prompt T64 envelope、B16/K256 decode leaf。
- Internal evidence：`3078.559→4654.812 tok/s`、opposite-order `+50.25/+51.70%`、candidate drift 0.54%、gate 126/128、rollback/guards green。
- 状态：exact scenario `WIN/default-on`；不是 universal Qwen serving claim。
- 同 epoch SGLang arm 缺失；不能用新 `4654.812` 与旧 `5579.2` 拼当前比值或排名。
- 源码/raw 尚未迁成公开独立 bundle，本站不声称 fresh replay。
- 状态与后续 external revalidation 见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
