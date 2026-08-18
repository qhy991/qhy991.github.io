---
layout: post
title: "为什么 2K 上快 9% 的 Kernel 到 64K 就消失？从 Amdahl、DeepEP 到瓶颈迁移"
date: 2026-08-17 11:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Long Context, Amdahl, DeepEP, FlashMLA, SGLang, Workload, B300]
reading_time: 27
cover_image: /assets/blog-64k-bottleneck-migration.png
excerpt: "同一个 fused/index projection 在 2K prefill TTFT 上曾快 7–9%，到 64K 却打平或回退。本文区分冷 prefill、cached incremental prefill 与 long decode，并从 DSA/index-score 占比、DeepEP/NCCL、production ABI、path hit 和 CUDA Graph 解释瓶颈如何随 context 迁移。"
---

> 本文基于 B300-M2 当前 GLM-5.2 实验分支的短/长上下文 E2E 表、64K cached-incremental Nsys 分类、默认路径代码审计，以及 SGLang-DGMK 公开 N6 evidence。Profiler 百分比只做因果定位，不作为正式 wall-time 加速比；旧单轮结果保留其样本量和路径限制。

一个 kernel 在 2K context 上让 TTFT 快 9%，到了 64K 却完全看不到收益，最常见的解释是：

> “长上下文噪声太大。”

噪声可能存在，但它通常不是最重要的原因。更根本的是：**工作负载变了，关键路径也变了。**

在 GLM-5.2 的一组 B300 结果中：

- `e2e_seq=2048`：`fused_qkv_a_proj` / `index_q_upproj` 的 TTFT 约快 `7–9%`；
- `e2e_seq=1024/4096`：大多打平或更慢；
- 旧 `decode_kv=64K`：TTFT 无收益，有些 arm 明显回退；
- 64K BS32 的 decode TPOT 曾出现约 `12–13%` 正向，但样本少，且不是 TTFT；
- 专门的 64K cached-incremental profile 显示时间结构已经被 DeepEP、long-KV attention 与不同 path 主导。

Kernel 没有“突然失效”。它只是从关键路径中退到了一个更小的角落。

---

## 1. “64K”至少可能指三种完全不同的 Workload

如果只写 `context=64K`，无法判断测量了什么。

### 1.1 Cold 64K Prefill

```text
新请求携带约 65K prompt
    ↓
从零建立整段 KV cache
    ↓
生成首 token
```

TTFT 包含完整 64K prefill。旧 `decode_kv` 结果属于这一类，TTFT 约 `111–115 s`。

### 1.2 Cached 64K Incremental Prefill

```text
64K prefix 已进入 cache
    ↓
新增 M=1024 / 2048 token
    ↓
增量 prefill + first token
```

专门的 capture 中 TTFT 约 `2.30 s / 4.24 s`。这里的分母与 cold prefill 完全不同。

### 1.3 Long-KV Decode

```text
KV 已约 64K
    ↓
每 step 只有少量新 token
    ↓
反复扫描/索引长 KV
```

主指标应是 TPOT/ITL 或 decode throughput，而不是首轮 TTFT。

![Cold Prefill、Cached Incremental Prefill 与 Long Decode 的不同边界](/assets/blog-64k-workload-identity.svg)

*图 1：三种 workload 都可能被简称为“64K”，但关键路径、metric 和可优化算子完全不同。*

跨这三种边界比较百分比，就像把仓库初始化、增量编译和线上请求延迟混成一个指标。

---

## 2. 旧结果到底显示了什么？

全新 prompt、BS16 的 E2E 表：

| Input | Arm | TTFT P50 | 相对 OPT0 |
| ---: | --- | ---: | ---: |
| 1024 | OPT0 | 0.397 s | 1.000× |
| 1024 | fused | 0.414 s | 1.043×，更慢 |
| 2048 | OPT0 | 0.565 s | 1.000× |
| 2048 | fused | 0.525 s | **0.929×** |
| 2048 | index | 0.514 s | **0.910×** |
| 4096 | OPT0 | 0.648 s | 1.000× |
| 4096 | fused | 0.672 s | 1.037×，更慢 |

同一 candidate 只在 2K 点明显获胜。它不是随 context 单调改善，也不是一个通用 replacement。

旧 cold-64K BS16：

| Arm | TTFT P50 | TPOT P50 |
| --- | ---: | ---: |
| OPT0 | 111.333 s | 31.29 ms |
| fused | 114.257 s | 36.27 ms |
| index | 113.306 s | 30.80 ms |
| all_gain | 114.701 s | 36.11 ms |

TTFT 分母几乎完全是 cold prefill；decode-side kernel 的小改善不可能移动 111 秒级主路径。

BS32 的 TPOT 有正向方向，但样本量小，而且 TTFT 仍无收益。不能用 decode TPOT 证明 cold-prefill TTFT winner。

---

## 3. Amdahl 定律怎样提前告诉我们“看不见”？

假设一个 kernel 占目标关键路径比例 $f$，自身 speedup 为 $s$，理想整体 speedup：

$$
S_{\mathrm{total}}
=
\frac{1}
{(1-f)+\frac{f}{s}}
$$

如果某个 GEMM 只占 5%，leaf 快 1.20×：

$$
S_{\mathrm{total}}
=
\frac{1}
{0.95+\frac{0.05}{1.20}}
\approx1.0084
$$

理想上限只有约 `0.84%`。再加上 selector、adapter、launch、stream contention 和通信，完全可能归零或反转。

![Leaf Speedup 经过 Amdahl、Path 与 Communication 后逐层缩小](/assets/blog-64k-amdahl-funnel.svg)

*图 2：局部百分比不是系统百分比；每一层只会保留真正暴露在 wall-clock 关键路径上的部分。*

最便宜的实验往往是先算上限。若 optimistic ceiling 已低于 noise/promotion floor，就不应启动完整模型。

---

## 4. 单卡 Compute 结构为什么随 Context 改变？

64K、prefill M4096 的单卡 layer microbenchmark：

| Compute group | Layer share |
| --- | ---: |
| `dsa_attn + index_score` | **约 58%** |
| MoE gate/up/down | 约 26% |
| 常被 swap 的 projection GEMMs | 往往合计 **<15%** |

即使增量新 token M 只有 1K–4K，indexer/score/attention 仍要处理长 KV。复杂度依赖 context $S$ 的部分迅速膨胀；只依赖 M 的 projection 相对占比下降。

短 context 中：

```text
projection GEMM 占比可见
attention 扫描尚短
leaf 1.1× 有机会变成 TTFT 几个百分点
```

长 context 中：

```text
DSA / index score 随 S 膨胀
projection 变成薄层
同样绝对 saving 被更大分母稀释
```

这就是 bottleneck migration。

---

## 5. 多卡 Serving 又增加一层 Communication 分母

短 prompt/decode-oriented Nsys capture 中，summed GPU kernel-time：

```text
NCCL        ≈ 31.3%
DeepEP      ≈ 21.2%
communication total ≈ 52.5%
```

专门的 64K cached-incremental prefill capture：

```text
DeepEP      ≈ 77.9%
NCCL        ≈ 1.4%
DeepGEMM    ≈ 12.2%
DSA/MLA     ≈ 2.8%
```

但必须加粗 caveat：

1. 这是 **sum of kernel durations**，不是 wall-exclusive share；
2. DeepEP low-latency dispatch 包含 busy-wait，GPU-time 可能高估 useful communication work；
3. profiler capture 会扰动 TTFT，不能用于正式加速百分比；
4. 分类适合告诉我们“去哪里调查”，不适合直接做 Amdahl 精确分母。

尽管如此，方向很清楚：多 GPU serving 中通信和 rank progress 是一等公民。优化一个小 projection 时，真正暴露的 critical share 可能比单卡 layer 表更小。

---

## 6. 为什么 Harness Winner 到生产路径只剩 1.00–1.05×？

很多历史 harness baseline 使用较慢的 FP32 block-scale 参考。生产 OPT0 已使用：

- packed int32 UE8M0 scales；
- DeepGEMM；
- PDL；
- production layout；
- CUDA Graph /真实 scheduler。

若 candidate 只接受 FP32 scale，接入生产需要：

```text
packed scale
  → unpack / adapter
  → temporary tensor
  → extra kernel / copy
  → historical candidate
```

Microbenchmark 没计 adapter 税，相对生产 stock 的真实收益可能只剩 1.00–1.05×，甚至回退。

公平比较的 baseline 不是“能算对的慢实现”，而是实际 serving incumbent。

---

## 7. 路径打偏：优化了 Contiguous PSUM，生产却走 Masked GEMM

默认 DeepEP low-latency MoE 路径：

```text
DeepEP LL
  → masked grouped GEMM
```

某些历史收益属于：

```text
DeepEP normal / contiguous GEMM
  + PSUM
```

如果 runtime 没命中 candidate path，漂亮数字与线上没有关系。即使强行切到 contig+PSUM，完整 E2E 仍可能因模式变化和开销变慢。

同样，历史 DSA candidate 可能针对 `flash_mla_sparse_fwd` layout，而线上 backend 是 `flashmla_kv`。名字都叫 sparse attention，不代表 ABI、page layout、scheduler 和 graph node 相同。

所以 path-hit proof 必须包含：

```text
op
phase
local M
N/K
dtype / scale ABI
backend
graph/eager
exact hit/miss marker
```

---

## 8. 为什么 `all_gain` 可能比单算子更差？

把两个 leaf winner 一起打开，不代表收益相加。组合会改变：

- cache footprint；
- graph topology；
- stream ordering；
- register/SM occupancy；
- allocator/workspace；
- producer/consumer arrival；
- 最慢 rank。

旧表中 `all_gain@1024` TTFT 是 `1.184×`，明显回退；`all_gain@2048` 也没有优于最佳单项 index。

组合 candidate 必须重新走 correctness、path hit 和 E2E gate。不能建立一个“winner list”后自动全开。

---

## 9. CUDA Graph 为什么会再改变一次结论？

Stable small-M decode 常被 CUDA Graph capture；large prefill M 可能 eager 或落在不同 graph bucket。

一个 candidate 可能：

- eager 减少 Python/launch 开销；
- graph replay 中这些开销本来已被摊薄；
- capture 时选择了不同 kernel；
- graph node 额外出现 adapter/copy；
- M16 win，M32 或大 M 回退。

因此比较矩阵必须是：

| Baseline | Candidate |
| --- | --- |
| native eager | optimized eager |
| native graph | optimized graph |

不能用 optimized eager 对 native graph 宣称 kernel win。

---

## 10. 长 Context 真正有效的两类改变

### 10.1 改变并行分解：Split-KV

Llama-3.1-8B standalone 中，one-partition attention：

```text
4K: 10.063 ms
8K: 17.288 ms
```

Split-KV：

```text
4K: 3.589 ms
8K: 4.050 ms
```

它不是把同一 CTA 快一点，而是沿 KV sequence 增加并行 partitions，再稳定合并 softmax partials。总信号约 2.80×/4.27×；加 dynamic tail 后约 2.85×/4.31×。当前仍 freeze pending，但机制说明大收益来自改变分解。

### 10.2 改变跨 Rank 边界：N6

GLM-5.2 100K cached-prefill N6 没有换主 GEMM，而是对齐：

- balanced expert placement；
- Router 直接输出 physical IDs；
- DeepEP 120-SM budget。

正式 P50 TTFT 改善 5.31%，吞吐提高 7.53%，独立 seed 复现。这类跨 compute/communication/control 边界的优化更可能触及长 context 多卡关键路径。

它们共同说明：当 bottleneck 已迁移，继续微调旧 winner 不如重新构造执行图。

---

## 11. 一张 Workload Matrix 应该记录什么？

![Kernel Winner 必须绑定 Workload Cell 与 Evidence Level](/assets/blog-64k-workload-matrix.svg)

*图 3：一个 candidate 的有效范围是多维 cell，不是“支持 64K”或“支持 decode”这种标签。*

至少记录：

| 类别 | 字段 |
| --- | --- |
| Model | revision、quantization、layers、experts |
| Request | 每请求 cached/new/output，global/local batch |
| KV | length distribution、page、dtype、backend |
| Parallelism | TP/DP/EP/CP、rank mapping |
| Execution | prefill/decode/chunked、graph/eager、bucket |
| Path | op、shape、ABI、hit marker、fallback |
| Timing | TTFT/TPOT/wall、warmup、samples、pair order |
| Correctness | tokens、logprobs、state、cache side effects |
| Evidence | leaf/layer/model/HTTP、raw/profile/replay |

只有所有字段相同，百分比才能直接比较。

---

## 12. 怎样判断“收益消失”是合理迁移还是实验坏了？

### 先检查 Workload Identity

- cold 还是 cached？
- prefill 还是 decode？
- 每请求长度还是 aggregate？
- local M 是否变化？

### 再检查 Path Identity

- candidate hit 吗？
- production ABI 相同吗？
- backend/layout/graph 相同吗？
- 有 silent fallback/adapter 吗？

### 再算 Amdahl Ceiling

- 目标节点 critical share；
- 绝对 saving；
- 调用频次；
- communication 稀释；
- 新开销。

### 最后区分 Noise 与 Bottleneck Migration

如果上限仍大但测不到，调查 noise、机器健康和 path drift。如果上限已经 <1%，看不到是预期，不应靠更多重复把噪声“跑成 winner”。

---

## 13. 一个适合 Agent 的候选迁移记录

```text
Source cell:
  model / GPU / phase / shape / ABI / mode

Target cell:
  changed fields

Mechanism invariant:
  why it should transfer

Path proof:
  exact selector / hit / binary

New bottleneck:
  compute / communication / control DAG

Amdahl ceiling:
  optimistic E2E impact

Rejection rule:
  correctness / path / materiality / E2E
```

这种记录比“在 A 上快，所以在 B 上试一下”更能避免重复失败。

---

## 14. 当前证据的限制

- 2K/64K E2E 表是历史 experiment matrix，不是所有 cell 都有多轮 formal pairs；
- 64K BS32 TPOT 正向样本较少，只保留为研究方向；
- Nsys 百分比是 summed kernel-time，含 DeepEP busy-wait；
- 64K cold prefill 与 cached incremental capture 不是同一批请求；
- Split-KV/dynamic-tail owning freeze pending；
- N6 只适用于冻结 100K cached-prefill cell，不代表所有长上下文。

这些限制不妨碍机制学习，但禁止把数字拼成统一排行榜。

---

## 结语：不是 Winner 消失了，是它的分母变了

2K 上的 projection winner 到 64K 消失，通常不是 kernel 变慢，而是：

```text
Context 变长
  → attention / index score 膨胀
  → communication / rank tail 更重要
  → projection critical share 下降
  → adapter / path mismatch 占比上升
  → 原 leaf saving 低于 E2E materiality
```

真正需要迁移的不是补丁，而是机制和前提。短 context 的 fixed-NK GEMM 可以留在窄 bucket；长 context 应重新审计 Split-KV、page/index chain、DeepEP/placement 和 control DAG。

所以，看到“以前快 9%，现在不快”时，不要立刻调更多 tile，也不要先怪噪声。先问：

> **“现在计时边界还是同一个问题吗？目标 kernel 仍在关键路径上吗？production path 和 baseline 还是同一个吗？它的 Amdahl 上限还有多少？”**

如果答案变了，收益消失不是实验失败，而是系统在告诉你：瓶颈已经迁移，下一轮优化也该迁移了。
