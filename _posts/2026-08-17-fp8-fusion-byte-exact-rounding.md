---
layout: post
title: "为什么 FP8 Fusion 最难的不是快，而是一个 Byte 都不能变：从 rcp.rn.f32 到 Reduction Tree"
date: 2026-08-17 07:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [FP8, Kernel Fusion, Triton, CUDA, RMSNorm, SwiGLU, Numerical Correctness]
reading_time: 30
cover_image: /assets/blog-fp8-fusion-byte-exact.png
excerpt: "Qwen3.5 的 activation quant、fused add-RMSNorm 和 shared SwiGLU+quant 展示了三种不同的 fusion 穿透率。本文解释为什么 reciprocal 末位、BF16 舍入点、reduction tree、scale layout 和下游 GEMM 都属于模型接口，以及为什么局部快 33% 仍可能因调用频次太低而不值得跑 E2E。"
---

> 本文基于 SubCUDA `d1db18f` 的 machine-readable cases 与 OMoE 对应源码/证据 commits。TP1 activation-quant 与 add-RMSNorm 的 fresh replay 仍需要外部 OMoE checkout；当前 SubCUDA verifier 对它们明确报告 BLOCKED，而不是伪造本地复现。TP2 shared-SwiGLU、低上限与负调度案例的结构化 replay 已通过。文中区分历史 accepted evidence、当前可重放证据和复现缺口。

Kernel fusion 的宣传通常很简单：

```text
原来 7 个 kernel
    ↓
现在 1 个 kernel
    ↓
launch 更少，中间 tensor 更少，所以更快
```

但在 FP8 推理里，“把公式放进同一个 kernel”只是最容易的部分。

真正困难的是保持这些接口完全不变：

- FP8 payload 的每一个 byte；
- FP32 scale 的每一个 bit；
- scale 的 shape、stride 和 major layout；
- BF16 在哪个时刻 round；
- reduction tree 以什么顺序相加；
- residual、KV、state 等原地副作用；
- 下游 GEMM 看到的最终 ABI；
- CUDA Graph 捕获和 replay 的路径身份。

如果这些 seam 中任何一个发生变化，fusion 可能更快、误差也可能更小，却不再是同一个模型。

这篇文章用三组真实 Qwen3.5 结果回答一个问题：

> **怎样让 fusion 不只是“数值差不多”，而是 byte-exact、consumer-exact，并且真的穿透到端到端？**

---

## 1. 三个正例，为什么端到端穿透率差这么大？

| Fusion | Local boundary | E2E result | 默认状态 |
| --- | ---: | ---: | --- |
| TP1 activation quant | 约 12.08×–13.58× | aggregate **+21.8389%** | ON，可显式回滚 |
| TP1 residual-add + RMSNorm | 96-site 约 9× | B16 **+13.5823%**，B32 **+11.0279%** | qualified，OFF |
| TP2 shared SwiGLU + FP8 quant | boundary −44.18% | TP2 **+0.417351%** | opt-in，OFF |

三者都删除 materialization 和 launch，但差异来自：

- 每 step 调用多少次；
- 边界占完整关键路径多少；
- 是否删除真正的 HBM round trip；
- 是否改变下游 consumer；
- 是否引入新的 vector/load/reduction 成本；
- 单卡 TP1 与双卡 TP2 的分母不同。

可以先记一个粗略优先级公式：

$$
\mathrm{E2E\ opportunity}
\propto
\mathrm{frequency}
\times
\mathrm{critical\ path\ share}
\times
\mathrm{removed\ fraction}
\times
\mathrm{correctness\ confidence}
$$

Fusion 快多少只是其中一项。

---

## 2. FP8 动态量化到底在做什么？

以 group-128 E4M3 activation quant 为例。对每个 token 的每 128 个 BF16 元素：

$$
a_{\max}=\max_i |x_i|
$$

$$
s=\frac{\max(a_{\max},10^{-4})}{448}
$$

$$
q_i=\mathrm{E4M3}\left(
\mathrm{clamp}\left(\frac{x_i}{s},-448,448\right)
\right)
$$

其中：

- `q` 是下游 FP8 GEMM 的 payload；
- `s` 是每组 scale；
- 448 是该 E4M3 finite range；
- scale 必须直接写成 consumer 需要的 MN-major `[K/128, M]`。

原始 PyTorch 表达式会经过 reshape、abs、amax、clamp、scale、reciprocal、multiply、FP8 cast、transpose/contiguous，冻结路径大约产生 7 次 GPU launch。

候选 Triton kernel 一次处理 4 个相邻 group-128，在一个 CTA 内完成全部 reduction、scale、reciprocal、clamp、FP8 cast，并直接写 consumer layout。

![Activation quant 从七段边界压缩为一个 consumer-format producer](/assets/blog-fp8-fusion-quant-seam.svg)

*图 1：主要性能来源是删除 launch、中间 tensor 和 scale transpose；inline PTX 负责固定精确 reciprocal 语义。*

---

## 3. 为什么普通的 `1 / scale` 会产生不同 FP8 Byte？

数学上，reciprocal 都是 $1/s$。但编译器可能选择近似 reciprocal，或使用不同的校正序列。

对 FP32/BF16 后续计算，末位差异常常很小。但 FP8 是稀疏的离散值集合。如果 $x/s$ 恰好靠近两个 E4M3 可表示值的中点，reciprocal 最后几 bit 的变化可能让结果跨过舍入边界：

```text
candidate A → FP8 code 0x4A
candidate B → FP8 code 0x4B
```

因此 Triton kernel 用一条 inline PTX：

```ptx
rcp.rn.f32
```

固定 round-to-nearest 的 reciprocal contract。

这条指令的作用不是独立提供 21.84% 加速，而是让更快的融合数据流通过 byte-exact gate。最终性能主要来自：

- 每次 quant 约 7→1 launch；
- 四个 group 共用一次 CTA 调度；
- 删除中间 scale/reduction tensor；
- 直接写 `[K/128,M]` consumer layout。

如果把整个收益归给一条 PTX 指令，就把“数值使能器”和“数据流优化”混为一谈。

---

## 4. Byte-Exact 不只是检查 `q`

Activation quant 的 correctness 至少需要四层：

### 4.1 Payload bytes

将 E4M3 tensor view 成 `uint8`，逐字节比较。不能只转回 FP32 做 `allclose`，因为不同 code 可能在宽松容差内看起来接近。

### 4.2 Scale bytes 与 layout

scale 的数值相同还不够。下游按 `[K/128,M]` 解释；若候选写成 `[M,K/128]`，数组内容看似一样，consumer 读取的组却完全不同。

### 4.3 下游公共 GEMM

把 baseline/candidate 的 `q + scale` 都送入同一个公共 FlashInfer FP8 GEMM，要求 BF16 output byte-exact。这样才能验证 ABI，而不只是 quantizer 自己。

### 4.4 完整 Model Graph

验证 128-step token hash、eager/graph state、snapshot/restore 和 reset。量化 byte 一致应当传递到完整模型，但仍需要端到端证明路径命中和状态稳定。

输入还必须覆盖：

- random、all-zero；
- NaN、+Inf、-Inf；
- BF16 exponent sweep；
- E4M3 half-way tie 及相邻值；
- M16/M32 × K1024/3072/8192；
- unsupported dtype/shape rollback。

随机高斯数据很少恰好落在 FP8 tie 上，只跑 random 会制造虚假的“完全一致”。

---

## 5. 为什么 Activation Quant 能带来 21.84% E2E？

Operator graph 中，source 与 candidate 的 medians：

| Shape | Source | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| M16 K1024 | 15.779208 μs | 1.218375 μs | 12.951× |
| M16 K3072 | 16.943407 μs | 1.257253 μs | 13.477× |
| M16 K8192 | 17.654930 μs | 1.331342 μs | 13.261× |
| M32 K3072 | 17.748307 μs | 1.307135 μs | 13.578× |

真实每 step 的 K1024/K3072/K8192 调用 mix 是 `48/252/48`，共 348 次 activation quant。

如果每次从约 7 launch 变为 1 launch，每 step 删除：

$$
348\times(7-1)=2088\ \mathrm{launches}
$$

128 steps 共删除：

$$
2088\times128=267{,}264\ \mathrm{launches}
$$

高频使局部 seam 真正穿透到模型：

| TP1 | Baseline | Candidate | Improvement |
| --- | ---: | ---: | ---: |
| B16 | 576.271904 tok/s | 719.023707 tok/s | +24.7716% |
| B32 | 910.440052 tok/s | 1088.771399 tok/s | +19.5874% |
| Aggregate | 762.837207 tok/s | 929.432545 tok/s | **+21.8389%** |

这不是“一条 reciprocal 快 21%”，而是一个高频 launch/materialization 边界被删除。

---

## 6. 第二个数值 Seam：Residual Add 先 Round BF16，再做 RMSNorm

Qwen3.5 的原语义是：

```text
residual = BF16_RNE(residual + delta)
normalized = RMSNorm(float(residual))
```

一个看似“更精确”的融合实现可能直接用未舍入的 FP32 sum 计算 RMSNorm：

```text
sum_fp32 = float(residual) + float(delta)
normalized = RMSNorm(sum_fp32)
residual = BF16_RNE(sum_fp32)
```

它的数学误差可能更小，却改变了模型接口。早期直接复用 FlashInfer fused add-RMSNorm 的版本表面吞吐提高约 `10.912%`，但 token/state gate 失败，正式可引用收益为 **0**。

![FP8/BF16 Fusion 中不能跨越的数值 Seam](/assets/blog-fp8-fusion-numerical-seams.svg)

*图 2：Reciprocal rounding、BF16 materialization 和 reduction tree 都是数值接口，不是“实现细节”。*

最终 CUDA kernel 显式：

1. load BF16 residual 和 delta；
2. FP32 add；
3. 立即 round-to-nearest-even 到 BF16；
4. rounded residual 同时写回 global、保留寄存器并进入 reduction；
5. 按 PyTorch H3072 顺序做 FP32 square reduction；
6. `rsqrt(mean + 1e-6)`；
7. 乘 zero-centered `(1 + weight)`；
8. BF16 output 写回原 delta buffer。

---

## 7. 为什么 Reduction Tree 也必须复现？

浮点加法不满足结合律：

$$
(a+b)+c\neq a+(b+c)
$$

两个 reduction 都可能计算“sum of squares”，但不同分组和 warp tree 会在 FP32 末位产生差异。经过 `rsqrt`、weight scale 和 BF16 cast 后，差异可能扩散到模型 token。

修复前后非常有教育意义：

| 版本 | B16 output mismatch | B32 output mismatch | Residual mismatch |
| --- | ---: | ---: | ---: |
| 错误 reduction tree | 22 | 36 | 0 / 0 |
| PyTorch-compatible tree | 0 | 0 | 0 / 0 |

Residual 已完全正确，错误只来自 norm reduction order。

这说明“副作用 exact + 最终 output allclose”仍可能漏掉结构性数值差异。对于稳定 decode graph，reduction order 就是 public interface 的一部分。

---

## 8. Add-RMSNorm 为什么也能穿透到 E2E？

Qwen3.5 每 forward 有 96 个融合位点：

```text
36 次 GDN
12 次 full attention
48 次 MoE producer
```

128-step graph 每个 capture/replay 合同是：

$$
96\times128=12{,}288\ \mathrm{fused\ calls}
$$

候选使用 192 threads，每线程处理一个 256-bit vector：

$$
192\times16\ \mathrm{BF16}=3072\ \mathrm{BF16}
$$

正好覆盖 H3072，无 tail predicate。最终 SASS 有 256-bit global load/store、32 registers、零 spill。

同模型加载五对 A/B：

| TP1 | Selector OFF | Selector ON | Improvement |
| --- | ---: | ---: | ---: |
| B16 | 829.977304 tok/s | 942.707582 tok/s | **+13.5823%** |
| B32 | 1352.549386 tok/s | 1501.707245 tok/s | **+11.0279%** |
| Aggregate | 1117.925781 tok/s | 1253.870367 tok/s | +12.1604% |

这仍是 default-off qualified candidate，不是无环境变量默认收益。

---

## 9. Shared SwiGLU + Quant：为什么局部快 44%，TP2 只快 0.417%？

TP2 shared-expert 路径原来：

```text
gate/up GEMM
    ↓
BF16 SwiGLU [32,512]
    ↓ 32 KiB write/read
group-128 E4M3 quant
    ↓
shared down GEMM
```

候选让 producer 直接写 down GEMM 需要的 E4M3 payload 和 FP32 scales，删除 BF16 intermediate 和一个 launch。

但第一次融合版本反而让 global load request 从 `65.54 KB` 膨胀到 `262.14 KB`，因为四个 BF16 都用 scalar load。只有改成两个对齐 64-bit load，才把 request 恢复到 `65.54 KB`。

这提醒我们：

> **Fusion 删除了一次 materialization，不代表融合后的内部访问自动 coalesced。必须重新 profile 新 kernel。**

最终结果：

| Boundary | Control | Candidate | Improvement |
| --- | ---: | ---: | ---: |
| 64-boundary operator | 2.814736 μs | 1.571218 μs | −44.18% |
| 加 unchanged down GEMM | 6.148090 μs | 4.902360 μs | −20.26% |
| TP2 throughput | 3547.800006 | 3562.606786 tok/s | **+0.417351%** |
| Graph wall | 1154.518291 ms | 1149.719923 ms | −4.798368 ms |

五对全胜，payload、scale、down output 与 token/rank hash exact。但 TP2 分母中还有完整 GDN、attention、routed MoE 和通信，因此 E2E 穿透远小于局部百分比。

原来使用的一条 inline `rcp.rn.f32` 后来被 CUDA 13.2 `__frcp_rn` 替代，完整 device SASS 和资源相同。这个案例最终应分类为 CUDA source fusion，而不是 PTX-only。

---

## 10. 反例一：局部 Fusion 快 33%，为什么连 E2E 都不值得跑？

D086 将 embedding RMSNorm 与 FP8 prelude quant 融合：

```text
6.145296 → 4.101072 μs/step
local reduction = 33.2649%
```

四个输出全部 byte-exact。但这个 boundary 每 decode step 只调用一次：

$$
2.044224\ \mu s\times128
=0.261661\ \mathrm{ms/graph}
$$

当前 graph wall 约 `1157.576446 ms`，乐观上限只有：

$$
\frac{0.261661}{1157.576446}
\approx0.0226\%
$$

低于 0.05% promotion floor，因此按合同停止，不浪费完整模型预算。

局部加速百分比很大，但频次和 critical-path share 太小。优秀 kernel 不一定是优秀项目优先级。

---

## 11. 反例二：PDL 为什么一个微秒都没省？

R97 在 gate/up GEMM → SwiGLU+quant 边界加入 PDL consumer。理论上 consumer 可以提前进入 GPU，在真正读 producer 输出前 `cudaGridDependencySynchronize()`。

问题是 consumer 在 wait 前没有独立前缀：第一项有效工作就是读取 gate/up。于是：

```text
consumer 提前 launch
    ↓
立刻 wait
    ↓
没有 prologue / weight prefetch 可覆盖
```

60 个随机顺序 paired repeats：

```text
median delta = 0.000 μs
candidate wins = 24/60
```

正确性 byte-exact，但没有 materiality，不进入 E2E。

PDL 只移动 launch permission，不自动创造可重叠工作。如果 wait 前的安全指令集合为空，理论上限就接近零。

---

## 12. 反例三：看起来在 Overlap，Systems 却说没有重叠

D036 将 BlockScores route metadata 放到 side stream，试图与 activation quant 重叠。孤立边界：

$$
134.470403
\longrightarrow
132.422400\ \mu s
$$

改善 `1.523%`，30/30 获胜，IDs/output exact。

但 Systems 显示两个节点没有实际 overlap，只是串行顺序改变。TP2 prescreen：

```text
3538.427223 → 3533.456200 tok/s
candidate −0.140487%
0/2 pairs
```

完整 wall 增加 `1.628528 ms/graph`，按合同停止。

![Fusion 和 Overlap 候选的频次、边界与 E2E 裁决](/assets/blog-fp8-fusion-evidence.svg)

*图 3：局部百分比只有乘上调用频次和关键路径占比，并通过完整数值与路径 gate，才可能成为系统收益。*

---

## 13. 三种 Fusion 为什么得到三种不同结论？

| 维度 | Activation Quant | Add-RMSNorm | Shared SwiGLU+Quant |
| --- | --- | --- | --- |
| 删除内容 | 约 6 个多余 launch + 中间 tensor + scale transpose | add/norm 间 materialization + launch | BF16 intermediate + quant launch |
| 每 step 频次 | 348 次 | 96 次 | shared-expert 路径上的有限边界 |
| 数值 seam | reciprocal 与 FP8 tie | BF16 round + reduction tree | SwiGLU 双重 BF16 round + quant |
| Consumer ABI | FP8 payload + MN-major scale | residual state + normalized output | shared-down FP8 + scale |
| E2E 穿透 | +21.84% TP1 | +13.58%/+11.03% TP1 | +0.417% TP2 |
| 默认状态 | ON，可回滚 | OFF qualified | OFF opt-in |

穿透率不是“fusion 质量排行榜”。不同 workload、TP、调用频次和完整 graph 分母不能直接相除。

---

## 14. Byte-Exact Fusion 的七个设计原则

### 原则一：把 Materialization Seam 当成数值接口

如果原路径写 BF16 再读，融合后必须显式复现这个 round point，除非合同提前允许改变数值顺序。

### 原则二：Reduction Tree 属于模型语义

公式相同不够。需要复现 accumulator grouping、warp reduction 顺序和 rounding 指令。

### 原则三：直接写 Consumer Layout

Fusion 的 producer 应直接输出 payload、scale、stride 和 major layout，避免下游 transpose/pack。

### 原则四：检查所有 Side Effects

Residual、KV、recurrent state、cache slot 和 untouched region 都要验证。只看返回 tensor 会漏掉后续 divergence。

### 原则五：为离散格式构造 Tie Fixtures

FP8/BF16 的 half-way、NaN/Inf、signed zero 和 exponent sweep 比普通 random 更能发现 seam 错误。

### 原则六：重新 Profile 融合后的访存

第一次 shared-SwiGLU fusion 删除 intermediate，却让 scalar load request 膨胀 4×。Fusion 不是自动 coalescing。

### 原则七：先算 Frequency × Ceiling

如果 boundary 每 step 只出现一次，哪怕快 33% 也可能不到 E2E materiality floor。先算上限，再决定是否加载 122B 模型。

---

## 15. 如何验证一个新的 FP8 Fusion？

### A. 冻结合同

- exact M/K/group/dtype/device/arch；
- payload/scale layout 与 consumer；
- eager/graph；
- state side effects；
- baseline materialization round points。

### B. 独立 Oracle

- 不复用 candidate helper；
- payload view `uint8` byte-exact；
- scale bit/shape/stride exact；
- downstream GEMM output exact；
- unsupported case fail closed。

### C. Numerical Stress

- random、zero、NaN/Inf；
- BF16 half-ULP；
- FP8 E4M3 tie；
- multi-step recurrent state；
- real checkpoint weights。

### D. Binary Proof

- registers、spill、SMEM；
- load/store width；
- reciprocal/reduction lowering；
- exact selected kernel symbol。

### E. Promotion

- no-profiler randomized operator A/B；
- frequency/Amdahl ceiling；
- one-model-load paired E2E；
- full token/state hash；
- graph call ledger；
- NSYS/NCU 只用于解释。

---

## 16. 当前证据的诚实边界

SubCUDA 当前 checkout 可以直接 replay TP2 shared-SwiGLU、D086、R97、D036 的结构化 JSON 断言；这些 replay 已通过。

TP1 activation-quant 和 add-RMSNorm 的 case contract、教学文档和 OMoE commit 均可定位，但 fresh replay 需要外部 OMoE checkout，当前本机 verifier 明确返回 BLOCKED。因此这篇文章引用的是冻结的 accepted evidence，不宣称在本机重新加载 122B 模型复现。

Shared-SwiGLU 原冻结 DSO 未保留；fresh clone 应从冻结 source 重建 binary，并以新 build hash、oracle 和 E2E 形成新 authority，而不是假装重建出的 SO 与旧 DSO byte-identical。

---

## 结语：Fusion 的本质是重写接口，不是粘贴公式

三个正例表面上都是“把两个或多个 kernel 合成一个”，但真正成功的原因各不相同：

```text
Activation quant：
  删除高频 launch/materialization
  + rcp.rn 固定 FP8 byte contract

Add-RMSNorm：
  删除 96-site materialization
  + 显式 BF16 round
  + 复现 reduction tree

Shared SwiGLU+quant：
  删除 32 KiB intermediate
  + 修复融合后 scalar-load 膨胀
  + 直接写 down-GEMM ABI
```

反例又说明：频次太低、wait 前没有独立前缀、所谓 overlap 只是重排，都会让局部 winner 无法进入系统。

所以，面对新的 FP8 fusion，最重要的问题不是：

> “能不能少一次 launch？”

而是：

> **“被删掉的 materialization 原本固定了哪些 byte、rounding、layout 和 state 语义？候选是否逐项复现，并且这条边界在完整 graph 上出现得足够频繁？”**

只有同时回答数值合同和关键路径合同，Fusion 才既是同一个模型，也是真正更快的模型。
