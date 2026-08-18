---
layout: post
title: "为什么 FP8 不能全模型一键打开？从 M Bucket、Role Selector 到 Block Scale"
date: 2026-08-17 22:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [FP8, DeepGEMM, Quantization, Selector, Prefill, Qwen3, B200]
reading_time: 30
cover_image: /assets/blog-fp8-role-selector.png
excerpt: "Qwen3-4B 的 block-scaled DeepGEMM 在 B16×T512 把 FP8 prefill 从 38.939 降到 33.680 ms，但 B1 all-role 只有 124/128；逐 role 回滚后，仅 down projection 同时通过 126/128 与 4K/8K 约 9% 加速。本文解释为什么 precision policy 必须绑定 role、M、consumer 和 scale granularity。"
---

> 本文基于 Q4B7/E-Q4B-007 与 Q4B15/E-Q4B-016 两条历史 Qwen3-4B/B200 evidence。Batched DeepGEMM 的 source/result/A-B/gate logs 与 B1 down-only 的 RESULT/META/matrix/gate logs 均可定位；但两条 run 仍缺统一 manifest/hash 和 clean public binding，因此本文只支持 historical role/M selector 结论，不声称当前主分支 fresh replay。

很多推理系统把 precision 写成模型级配置：

```text
dtype = bf16
dtype = fp8
```

这对 checkpoint storage 很方便，却不足以描述真实执行。

同一个 Qwen3-4B 模型中，冻结结果是：

```text
Batched prefill, large M:
  q/k/v/o/gate_up/down → block-scaled DeepGEMM
  38.939 → 33.680 ms
  −13.5%

B1 prefill, all roles:
  stage-1 = 124/128
  correctness reject

B1 prefill, down only:
  4K: 26.194 → 23.672 ms
  8K: 58.076 → 52.953 ms
  gate = 126/128
```

所以正确策略不是：

```text
FP8 = ON
```

而是：

```text
backend / scale policy
= f(phase, M, projection role, consumer, correctness margin)
```

---

## 1. Rowwise FP8 路径是什么？

早期 dynamic FP8：

```text
BF16 activation
  → per-token amax / scale / E4M3 cast
  → FP8 activation + FP32 row scale
  → torch._scaled_mm
  → BF16 output
```

Activation scale 通常按 token/row；weight 使用 per-output-channel 或兼容 backend 的 scale。

优点：

- scale 粒度细；
- B1 数值轨迹稳定；
- backend 覆盖广；
- 小 M 时不需要复杂 block layout。

缺点：

- quant kernel / scale handling；
- `_scaled_mm` 调度接近自身上限；
- 大 M 没充分使用 SM100 block-scaled Tensor Core 路径。

---

## 2. Block-Scaled DeepGEMM 改变了什么？

候选：

```text
BF16 activation
  → group-128 activation quant
     E4M3 values
     UE8M0 power-of-two scales
     column-major / TMA-aligned scale layout
  → deep_gemm.fp8_gemm_nt
     activation scale: 1×128 blocks
     weight scale:     128×128 blocks
  → BF16 output
```

Weights 在初始化时一次性：

```text
per_block_cast_to_fp8
transform_sf_into_required_layout
```

Activation quant 直接产出 DeepGEMM 所需布局，replay 内没有 host transform。

![Rowwise 与 Block-Scaled FP8 的 Scale Granularity](/assets/blog-fp8-scale-granularity.svg)

*图 1：Rowwise/per-channel scale 更细；1×128/128×128 block scale 更适合 SM100 TMA/Tensor Core pipeline，但数值扰动与 shape 依赖更强。*

---

## 3. 为什么 Large M 更适合 DeepGEMM？

Batched prefill：

```text
B16 × T512 → M = 8192
B16 × T128 → M = 2048
```

大 M 可以摊销：

- quant launch；
- TMA descriptor；
- scale loads；
- kernel setup；
- tile tail；
- pipeline warmup。

同时提供更多：

- CTA waves；
- Tensor Core work；
- producer/consumer overlap；
- scale block reuse。

结果：

| Cell | Rowwise `_scaled_mm` | DeepGEMM | Result |
| --- | ---: | ---: | ---: |
| B16×T512 | 38.939 ms | 33.680 ms | −13.5% / 1.156× |
| B16×T128 | 10.587 ms | 9.753 ms | −7.9% |

GEMM-only 在 M8192 约为旧 backend 的 0.810×。

---

## 4. 为什么 Batched 可以 Default-on，B1 却失败？

Batched contract：

- B1 stage-1 仍用 rowwise：`127/128=99.2%`；
- Batched stage-4 使用 DeepGEMM：`502/512=98.0%`；
- Kill switch：`ANE_FP8_GEMM=rowwise`；
- Selector：`ANE_PREFILL_GEMM=fp8` 的 qualified batched scope。

B1 如果所有 roles 都用 block scale：

```text
124/128 = 96.9%
```

低于 98% 门。

原因不是简单“UE8M0 格式有问题”。更准确地说：

- 1×128 activation block scale 比 per-row 更粗；
- 128×128 weight block scale 比 per-output-channel 更粗；
- B1 每层误差 margin 更小；
- 多 projection 的扰动沿残差/attention/MLP 累积；
- 小 M 的性能收益又更难摊销 quant/layout overhead。

同一种 scale protocol 在大 M 的吞吐优势与 B1 的数值风险可以同时成立。

---

## 5. Projection Role 为什么是 Precision Contract 的一部分？

Transformer 中的 projection 不是同一个语义位置：

```text
Q / K / V
O projection
gate_up
down projection
```

它们的：

- 输入分布；
- output width；
- residual位置；
- downstream normalization；
- error amplification；
- call frequency；
- GEMM shape；

都不同。

因此“同样是 linear”不代表相同 precision tolerance。

逐 role 回滚矩阵发现：

- All-role：124/128，失败；
- `o` only：124/128，且 latency flat；
- `down` only：126/128，并有 material 4K/8K 收益。

这证明 full-scope failure 是 role-wise 扰动累积，不是 DeepGEMM 在 B1 完全不可用。

---

## 6. 为什么 `down` 是特殊的？

MLP：

```text
hidden
  → gate/up projection
  → SwiGLU
  → down projection
  → residual path
```

Down projection：

- 输入是大 intermediate MLP hidden；
- K/N shape 与 QKV/O 不同；
- 在 4K/8K prefill 中占较大时间；
- 单独切换可保留其他 roles 的 rowwise precision；
- downstream residual/norm 对该扰动的实际 gate margin仍可接受。

Selector 保持：

```text
Q/K/V/O/gate_up → rowwise FP8
down             → block-scaled DeepGEMM
```

![Phase、M 与 Projection Role 的 Precision Selector](/assets/blog-fp8-role-selector-matrix.svg)

*图 2：Batched large-M 可全 projection 使用 block scale；B1 只让 down 进入，其他 roles 保留 rowwise。Selector 是二维/三维 policy，不是模型级 flag。*

---

## 7. B1 Down-only 的正式结果

冻结 cell：

- Qwen3-4B；
- B1 FP8 Graph prefill；
- 4K / 8K；
- 1×B200；
- 只切换 MLP down；
- 其他 projection 保持 rowwise。

| B1 FP8 Prefill | Rowwise baseline | Down DeepGEMM | Saving |
| --- | ---: | ---: | ---: |
| PP@4K | 26.194 ms | 23.672 ms | −2.522 ms / −9.63% |
| PP@8K | 58.076 ms | 52.953 ms | −5.123 ms / −8.82% |

正确性：

- Stage-1 `126/128=98.4%`；
- Dual FP8 implementation parity `512/512=100%`；
- BF16、batched prefill、decode DAG 不经过该 selector；
- TG@4K/8K 约 `2.89/2.95 ms/step`，无回退。

![Batched All-Projection 与 B1 Down-Only 的 Evidence](/assets/blog-fp8-role-evidence.svg)

*图 3：Performance 与 correctness 共同决定 scope。All-role B1 即使某些 kernel 更快，也因 124/128 不能晋级。*

---

## 8. 为什么 `o` Projection 单独也失败？

冻结 role matrix 中：

```text
o-only = 124/128
latency ≈ flat
```

这意味着：

- 没有 correctness margin；
- 没有 performance upside；

是最明确的 STOP。

即使换更精细调参可能修复一项，也必须同时修复两项，机会成本远高于已通过的 down-only。

Role selector 的意义之一就是让失败 role 不拖累成功 role，也避免为了“统一 backend”继续浪费实验预算。

---

## 9. Precision 为什么不是一个 Model Label？

同一模型可以合法同时存在：

```text
B1 Q/K/V/O/gate_up: rowwise FP8
B1 down:             block-scaled FP8
Batched projections: block-scaled DeepGEMM
某些 prelude/state:  BF16 exact
某些 rope/store:     FP8 subset
```

Checkpoint 可能是 FP8，但 runtime activation、scale、accumulation、round seam 与 role policy仍然不同。

更准确的执行精度合同：

```yaml
precision_cell:
  phase: prefill
  batch: 1
  role: down
  M_bucket: 4096_or_8192
  activation_scale: group128_ue8m0
  weight_scale: block128x128
  backend: deepgemm
  output: bf16
  fallback: rowwise_scaled_mm
```

---

## 10. 为什么不能写成一条 Waterfall？

历史数字包括：

```text
44.6
39.567
38.939
33.680 ms
```

它们来自不同日期、tip、baseline、precision board 或 A/B。

特别是：

- `39.567` 是 precision-matched OMoE/SGLang board 的 OMoE 值；
- `38.939→33.680` 才是 DeepGEMM 增量 A/B；
- B1 down-only `26.194→23.672` 又是另一 run；
- 后续 `23.02` 属于不同 tip/session 的 camera-ready board。

不能写：

```text
26.194 → 23.672 → 23.02
```

因为最后一个 denominator identity 已改变。

---

## 11. Scale Granularity 怎样影响 Error？

Row scale：

$$
s_r=\frac{\max_j|x_{rj}|}{448}
$$

Block-128 scale：

$$
s_{r,b}
=
2^{\left\lceil\log_2(\max_{j\in b}|x_{rj}|/448)\right\rceil}
$$

UE8M0 power-of-two scale 便于硬件，但相对任意 FP32 scale 更粗。

误差取决于 block 内 dynamic range：

- 一个 outlier 会提高整个 block scale；
- 小值使用更少有效 mantissa；
- 不同 role 的 activation distribution 不同；
- 多层累积会改变 top-1 margin。

所以不能只以单 kernel max error 预测模型 gate。

---

## 12. M 怎样影响 Performance Break-even？

粗略：

$$
T_{\mathrm{rowwise}}
=T_{q,r}+T_{\mathrm{scaled\ mm}}
$$

$$
T_{\mathrm{block}}
=T_{q,b}+T_{\mathrm{layout}}+T_{\mathrm{DeepGEMM}}
$$

DeepGEMM 赢需要：

$$
T_{\mathrm{scaled\ mm}}-T_{\mathrm{DeepGEMM}}
>
T_{q,b}+T_{\mathrm{layout}}-T_{q,r}
$$

大 M 时 Tensor Core/tile throughput saving 大；小 M 时 setup/quant/tail 占比高。

这就是 selector 必须绑定 M bucket 的性能原因。

---

## 13. 如何构造一个 Role × M Selector？

最小决策表：

| Phase | M regime | Role | Backend | Reason |
| --- | --- | --- | --- | --- |
| Batched prefill | large | q/k/v/o/gate_up/down | block DeepGEMM | throughput + gate passed |
| B1 prefill | small/local | down | block DeepGEMM | 4K/8K material win + 126/128 |
| B1 prefill | small/local | q/k/v/o/gate_up | rowwise | all/o block scale fail or flat |
| Decode | tiny M | all | existing qualified | outside selector scope |
| Contract miss | any | any | rowwise/BF16 fallback | fail closed |

每个 cell 必须记录：

- path hit；
- scale layout；
- weight revision；
- correctness margin；
- latency；
- rollback。

---

## 14. 为什么 Selector 比统一实现更简单？

表面上多个 backend 更复杂。

但统一实现若要同时覆盖：

- B1 correctness；
- large-M throughput；
- QKV/O/MLP shape；
- rowwise/block scale；

往往需要大量内部条件和折中 tile，反而扩大不可见状态。

显式 selector 将复杂度放在边界：

```text
small set of qualified leaves
one fail-closed dispatcher
one fallback authority
```

这是由稳定 primitive 组合复杂行为，而不是让一个 kernel 承担所有 regime。

---

## 15. Evidence Debt 与当前可说结论

Q4B7：

- Worker/integrated artifact 可定位；
- RESULT/META/A-B/gate logs 存在；
- 欠统一 manifest/hash 与 clean/public ref。

Q4B15：

- Worker/archive/merge 可定位；
- RESULT/META/A-B/matrix/gate logs 存在；
- 同样欠统一 manifest/hash/public binding。

因此可支持：

- Historical batched DeepGEMM scoped win；
- Historical B1 down-only role-scoped win；
- All-role B1 correctness STOP；
- Selector design mechanism。

不能支持：

- 当前 main fresh result；
- 所有 Qwen/FP8 workload；
- 当前跨系统排名。

---

## 16. 最后记住

1. **FP8 不是一个模型级开关，而是一组执行合同。**
2. **Scale granularity 同时改变数值误差和硬件路径。**
3. **Large M 可以摊销 block-scaled pipeline，小 M 不一定。**
4. **Projection role 决定输入分布、shape、误差传播与性能价值。**
5. **逐 role rollback 可以从 all-scope failure 中救出真正可用的 leaf。**
6. **Selector 应绑定 `{phase, M, role, consumer}`，并 fail closed。**

真正的 policy 不是：

> “这个模型用 FP8。”

而是：

> **“在这个 phase、M bucket 和 projection role 上，这套 scale layout、backend 与 consumer 同时通过 correctness 和 wall-time gate。”**

---

## Evidence boundary

- Q4B7/E-Q4B-007：2026-07-21 FP8 batched prefill；B16T512 `38.939→33.680 ms`，B16T128 `10.587→9.753 ms`；stage-4 502/512，B1 kept rowwise 127/128；historical `MERGED-PARTIAL/default-on for qualified batched scope`。
- B1 all-role block scale：124/128，correctness STOP；不发布性能结论。
- Q4B15/E-Q4B-016：B1 down-only；4K `26.194→23.672 ms`，8K `58.076→52.953 ms`；stage-1 126/128，dual parity 512/512；role-scoped WIN。
- 两条 run source/result/log 可定位，但统一 manifest/hash 与 clean public binding仍欠；本站不声称 fresh replay。
- 不把不同 epoch 的 rowwise board、DeepGEMM A/B、down-only 和 camera-ready数字串成 waterfall。
- 不外推到 decode、其他模型/GPU、其他 role 或 scale layout。
- 状态与 provenance 见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
