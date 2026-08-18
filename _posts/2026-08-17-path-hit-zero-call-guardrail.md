---
layout: post
title: "为什么 Benchmark 变快了，却可能根本没跑你的 Kernel？从 Path-hit Counter、Zero-call 到 Guardrail"
date: 2026-08-17 13:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [CUDA, Path Hit, CUDA Graph, Benchmark, Dispatch, Fallback, Qwen3, B200]
reading_time: 27
cover_image: /assets/blog-path-hit-guardrail.png
excerpt: "五个 Qwen native bundle 揭示了性能评测最隐蔽的错误：候选库已经编译、模型也能跑、计时甚至更快，但 timed Graph 可能一次都没有调用它。本文从 selector、fallback、phase、Graph capture、same-path baseline 与 negative control 构造一套先证明路径、再解释时间的方法。"
---

> 本文基于 `agentic-megakernel@fdf4898` 中五个 Qwen3-4B native experiment bundle 的 contract、checksum-bound REPORT 与教学文档。它们是历史证据的结构化迁移：部分 worker revision、逐次 raw sample 与 standalone 命令仍有归档债务。因此本文只保留原来的 `ENABLED_GUARDED / default-off` 裁决，不把 guardrail 内的小正数升级成性能 winner。

假设你写了一个新的 CUDA Kernel：

- 编译成功；
- standalone correctness 通过；
- Python adapter 能加载 `.so`；
- 模型能够生成正确 token；
- 打开开关后 benchmark 还快了 `0.06%`。

这能证明新 Kernel 更快吗？

**不能。它甚至不能证明 timed region 运行过这个 Kernel。**

在一组 Qwen3-4B / B200 实验里，native RoPE + KV-store 在 batched prefill gate 中有 `864` 次 dispatch，但 batch-1 decode Graph 的 call count 是：

```text
0
```

与此同时，计时表仍然从 `2.69474` 变成 `2.69323 ms/token`，表面快了约 `0.06%`。

如果没有 call counter，这个噪声很容易被写成：

> “Native RoPE 融合让 decode 更快。”

真实结论却是：

> **这个 decode Graph 没有执行候选；这张表只能证明新增的 default-off 分支没有破坏 decode guardrail。**

这篇文章要建立一个很简单、却经常缺失的顺序：

$$
\text{Path identity}
\rightarrow
\text{Correctness}
\rightarrow
\text{Timing}
\rightarrow
\text{Attribution}
$$

时钟必须排在路径证明之后。

---

## 1. “代码存在”到“代码被计时”之间隔着五层控制面

初学者常把一个候选想成单一路径：

```text
Python 调用
  → 新 CUDA Kernel
```

真实推理引擎更像这样：

```text
Python / model code
  → adapter 检查 dtype / shape / stride / device
  → selector 检查 phase / batch / backend / feature flag
  → unsupported case 回退到 incumbent
  → eager launch 或 CUDA Graph capture
  → Graph replay 固定已经捕获的路径
```

候选可能在任何一层被绕开：

1. `.so` 存在，但 adapter 没有加载；
2. adapter 加载了，但 shape 不支持，走 fallback；
3. shape 支持，但当前 phase 是 decode，候选只服务 prefill；
4. feature flag 打开了，但 backend selector 选择另一条 fused path；
5. selector 后来改变了，但 CUDA Graph 仍 replay 旧 capture。

![从 Native Library 到 Timed Graph 的五层路径控制面](/assets/blog-path-hit-control-plane.svg)

*图 1：路径身份不是一个布尔开关。实现、adapter、selector、phase 与 Graph capture 必须同时对齐，候选才真正进入 timed region。*

所以，“我设置了环境变量”不是 path-hit evidence；“日志里加载了库”也不是。

最强的直接证据通常是：

- 候选 dispatch 后才递增的 counter；
- 精确的 kernel symbol / launch marker；
- capture 与 replay 分开的 path receipt；
- 与 workload 预期调用次数一致的 layer/phase 计数；
- 能让错误候选被 correctness gate 判刑的 negative control。

---

## 2. 五个 Native Bundle，到底测到了什么？

这五条实验都使用自研 CUDA shared library，但它们不属于同一个 phase，也不支持同一种性能结论。

| Candidate | 真正命中的路径 | Decode Graph 是否执行候选 | Graph 结果应该怎么读 |
| --- | --- | --- | --- |
| Fused add-RMSNorm A | batched prefill gate | 不执行 | no-regression guardrail |
| Bare KV store | batched prefill，144 dispatch | 不执行 | no-regression guardrail |
| NeoX RoPE + KV store | batched prefill，864 dispatch | `CALL_COUNT=0` | zero-call guardrail |
| Per-head Q/K RMSNorm | batch-1 decode 的 separate-Q/K path | 执行 | same-path latency-neutral |
| Native SwiGLU | batch-1 decode Graph | path-hit | 小正向但低于预注册 1% win floor |

这张表的关键不是哪个数字最大，而是第三列。

如果 candidate 只在 prefill 中执行，那么 decode Graph 上的 `ON/OFF` 差异没有 candidate 因果含义。它仍然有工程价值：验证新分支、加载逻辑和 fallback 没有污染另一条主路径。但它不是 speed result。

---

## 3. Zero-call：为什么计时仍然会变化？

先看 RoPE + KV-store：

| Context | Flag OFF | Flag ON | 表面变化 | Native call count |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.69474 ms/token | 2.69323 ms/token | −0.06% | 0 |
| 8192 | 2.83005 ms/token | 2.82979 ms/token | −0.01% | 0 |

候选一次都没执行，为什么数字不是完全相同？

因为性能测量是随机变量。即使代码路径完全一致，仍然可能受到：

- GPU 时钟与温度；
- 前一轮 cache / allocator 状态；
- host 调度与 launch jitter；
- 后台系统活动；
- 非交错运行造成的时间漂移；
- 少量样本的 order effect；
- Graph 外准备区间的细小差别。

可以把一次观测写成：

$$
T_{\mathrm{obs}}
=
T_{\mathrm{path}}
+
\epsilon_{\mathrm{clock}}
+
\epsilon_{\mathrm{cache}}
+
\epsilon_{\mathrm{host}}
+
\epsilon_{\mathrm{order}}
$$

当 `CALL_COUNT=0` 时：

$$
\Delta T_{\mathrm{candidate}}=0
$$

但两个独立样本的：

$$
\Delta \epsilon \neq 0
$$

因此出现正负小差值完全正常。

![Prefill Candidate 与 Zero-call Decode Graph 的区别](/assets/blog-path-hit-zero-call.svg)

*图 2：左侧 prefill gate 真正执行候选；右侧 decode Graph 的 ON/OFF 都 replay incumbent path。右侧的小差值来自测量噪声，只能用于 no-regression。*

这是实验归因中的硬规则：

> **没有 path hit，候选性能贡献按零处理。**

---

## 4. Call Counter 应该放在哪里？

一个看似简单的 counter 也可能撒谎。

错误做法：在 adapter 入口递增。

```cpp
call_count++;

if (!supported(dtype, shape, stride, device)) {
    return fallback(...);
}

launch_native(...);
```

这只能证明“有人尝试调用 adapter”，不能证明 native kernel 被 dispatch。

更可靠的语义是：

```cpp
if (!supported(dtype, shape, stride, device)) {
    fallback_count++;
    return fallback(...);
}

launch_native(...);
native_dispatch_count++;
```

如果 launch 本身可能异步失败，还要在测试模式下配合：

- launch error check；
- stream synchronize 或后续 correctness boundary；
- 精确 kernel marker；
- output/state oracle。

Counter 的使用还需要四个约束。

### 4.1 在 timed region 外读取

如果每次 dispatch 都把 device counter 拷回 CPU，path proof 会污染你想测量的路径。常见做法是：

```text
reset counter
capture / warm up
run N replays
synchronize
read counter once
```

### 4.2 先写预期计数

“大于零”有时太弱。

Bare KV store 的 prefill gate 记录 `144` 次 dispatch：36 层 × 4 次 store。RoPE + KV store 记录 `864` 次 call。预期计数能发现：

- 只命中部分 layer；
- 只命中 K，没有命中 V；
- warmup 命中，但 timed replay 没命中；
- 第一个 shape 命中，后续 fallback；
- capture 命中一次，replay 没有使用预期图。

### 4.3 分开 capture 与 replay

CUDA Graph 把 capture 时的 launch topology 固定下来。下面两句话不是同一件事：

```text
候选在 capture 时被调用
候选对应的 kernel node 在 replay 中被执行
```

前者可以用 host dispatch counter 证明；后者最好再用 graph-node identity、kernel marker 或 profiler symbol 证明。

### 4.4 Counter 不能替代 correctness

Counter 证明“跑了”，不证明“跑对了”。更快的错误路径同样可以命中一千次。

---

## 5. Fallback 是安全边界，也是归因陷阱

生产 adapter 通常需要检查：

```text
dtype 是否支持？
tensor 是否在目标 GPU？
shape 是否属于注册 bucket？
stride 是否连续或满足 vector alignment？
architecture 是否是 sm_100？
stream / graph 状态是否兼容？
```

不满足时回退 incumbent，是正确的工程设计。

但 fallback 让 benchmark 出现一种危险状态：

```text
程序成功
token 正确
性能稳定
候选从未执行
```

所以每个 fallback-capable candidate 至少应保留：

| Receipt | 回答的问题 |
| --- | --- |
| Native dispatch count | 候选真正 launch 了几次？ |
| Fallback count + reason | 哪些输入被 incumbent 接管？ |
| Shape/dtype/stride fingerprint | 实际输入是否属于宣称 cell？ |
| Kernel / graph node identity | Timed region 执行的是哪个 binary？ |
| Negative control | Gate 真的能识别错误候选吗？ |

其中 negative control 尤其重要。

如果你故意让候选输出错误，但模型 gate 仍然通过，说明：

- 候选没有命中；或
- correctness gate 没覆盖正确的输出/状态；或
- comparator 本身没有区分力。

一个不会“判刑”的测试，不能给正常候选发无罪证明。

---

## 6. Same-path Baseline：Q/K RMSNorm 为什么必须固定 Backend？

Q/K RMSNorm 案例展示了第二类常见归因错误。

系统有两条 backend：

```text
default fused Q/K backend

separate Q/K backend
  ├── SGL JIT RMSNorm
  └── native RMSNorm
```

Native library 只能替换 separate path 内部的 RMSNorm。如果拿：

```text
default fused backend
vs.
separate backend + native RMSNorm
```

做对比，就同时改变了：

- Q/K 是否 fused；
- kernel 数量；
- 中间 tensor；
- RMSNorm 实现；
- 可能的 layout 与 Graph topology。

即使结果更快，也不能归因给 native RMSNorm。

公平实验固定：

```text
ANE_QK_BACKEND=separate
```

只切换：

```text
OMOE_QKNORM_NATIVE=0 / 1
```

结果：

| Context | Separate + incumbent | Separate + native | Change |
| ---: | ---: | ---: | ---: |
| 4096 | 2.80865 | 2.80993 ms/token | +0.046% |
| 8192 | 2.94580 | 2.94680 ms/token | +0.034% |

它的正确解读是：

- native library 存在；
- standalone oracle 通过；
- model gate 与 negative control 通过；
- same-path Graph 内没有超过 1% 的回归；
- 当前样本不支持正向或负向的小差异；
- 因此保留为 opt-in substrate，而不是 winner。

“Latency-neutral”不是失败。它证明了一个干净、可组合的 native primitive 可以进入模型，而不改变现有 Graph 性能边界。

---

## 7. Guardrail 与 Treatment 是两种完全不同的实验

### Treatment

Treatment 实验的问题是：

> 候选在 timed path 中执行时，是否让目标指标变好？

它需要：

- candidate path hit；
- baseline/candidate 只差一个目标变量；
- correctness；
- 足够样本与预声明统计规则；
- 可归因的 timing boundary。

### Guardrail

Guardrail 的问题是：

> 引入这个分支之后，其他关键路径是否没有被破坏？

它可以有意测量一个不执行候选的 path。

例如 add-RMSNorm、bare KV store 与 RoPE + KV-store 都是 prefill substrate，但还会跑 batch-1 decode Graph：

```text
候选 seam 不在 decode path
     ↓
确认新增动态库、adapter、flag 和 source branch
没有让 decode 回归超过 1%
```

这里 `1%` 是 no-regression bound，不是“超过它才算候选被执行”的阈值。

把 guardrail 写成 treatment，会产生两种错误：

1. 小正向被冒充 speedup；
2. 小负向被冒充 candidate slowdown。

如果 call count 为零，两者都没有 candidate attribution。

---

## 8. SwiGLU：路径命中了，为什么仍然不是 Winner？

Native SwiGLU 与前三个 zero-call 例子不同：它在 decode Graph 中 path-hit。

结果：

| Context | FlashInfer fused baseline | Native | Improvement |
| ---: | ---: | ---: | ---: |
| 4096 | 2.6907 | 2.6677 ms/token | 约 0.85% |
| 8192 | 2.8308 | 2.8077 ms/token | 约 0.82% |

为什么仍然不是 whole-model WIN？

因为冻结 contract 预先定义的是：

> 两个 context 都不能回归超过 1%；通过则把 native library 认定为可用、default-off substrate。

任务不是“证明 0.8% 速度提升”。而且历史记录是两个分别启动、非交错的五重复 guardrail，只保留 summary/min/max，没有完整逐次 raw pairs。

如果看到结果后，把 1% no-regression gate 改成“任何正数都算 win”，就是事后改 estimand。

更重要的是，baseline 已经是 FlashInfer fused `silu_and_mul`，不是慢速 unfused Torch：

```text
gate, up = split(gate_up)
out = bf16(silu(fp32(gate)) * fp32(up))
```

Native path 的价值是：

- 自己拥有稳定 C ABI；
- shared library 不链接 Torch / FlashInfer / SGLang / Triton / cuBLAS；
- 精确 oracle 与 path hit 通过；
- 未来可以与 producer 或 consumer 继续融合。

这叫**实现与可组合性结果**，不是产品性能结论。

---

## 9. Path Hit 以后，正确性还要检查什么？

五个 bundle 说明：不同 operator 的正确性接口完全不同。

### Bare KV Store：零容差

它不做浮点算术，只复制已经舍入的 bytes，所以必须检查：

- 写入 K/V row bitwise equal；
- 未寻址 row 保持 sentinel；
- scattered/high/reused slot；
- int32 与 int64 index；
- fallback 前不能产生 partial write。

用 cosine similarity 检查纯 store，反而放松了不该放松的合同。

### RoPE + KV Store：两个 oracle class

RoPE 做浮点旋转，可以使用预声明 BF16 ULP bound；KV scatter 不做新算术，仍必须 bitwise。

一个 fused kernel 可以同时包含“允许数值容差的输出”和“必须完全相等的状态副作用”。

### Add + RMSNorm：舍入点是接口

参考顺序是：

```text
residual_bf16 = round_bf16(residual + delta)
variance_fp32 = mean(float(residual_bf16)^2)
output_bf16 = round_bf16(residual_bf16 * rsqrt(variance + eps) * weight)
```

如果 fusion 在 FP32 中完成 add 后直接 normalize，数学实数看起来更精确，却改变了 reduction 输入和后续 token 轨迹。

所以 path hit 只是第二道门，不是最后一道门。

---

## 10. 一条性能 Claim 至少要爬过六级梯子

![从实现存在到 Serving Promotion 的六级证据梯子](/assets/blog-path-hit-claim-ladder.svg)

*图 3：每一级回答不同问题。Standalone 正确、模型 path-hit、same-path timing 与 serving promotion 不能相互替代。*

### L0：Implementation exists

`.so` 能加载，symbol 存在，C ABI 清楚。

### L1：Standalone correctness

精确 shape、stride、dtype、side effect 与 fallback oracle 通过。

### L2：Model admission

模型 token/state gate 通过；negative control 能判错。

### L3：Path identity

Timed phase、layer、shape 与 Graph replay 确实执行候选，计数符合预期。

### L4：Same-path performance

Baseline 与 candidate 只差一个目标变量，样本与统计规则预先冻结。

### L5：E2E / serving promotion

完整模型或 HTTP serving 的目标指标通过 formal pairs、holdout、回滚与部署边界。

常见错误是从 L1 直接跳到 L5：

> “Native library 正确，所以线上会更快。”

另一个错误是从 L4 跳到 deployment-wide：

> “B1 4K Graph 快，所以 prefill、B16、8K、TP2 和 HTTP 都应该打开。”

证据梯子的意义不是增加文档，而是防止不同问题的答案互相冒充。

---

## 11. 怎样设计一个不会被 Fallback 欺骗的 Benchmark？

可以使用下面这份最小模板。

### 11.1 冻结 workload identity

```yaml
model: Qwen3-4B
gpu: B200 x1
dtype: BF16
phase: prefill | decode
batch: 1 | 16
context: 4096 | 8192
backend: separate-qk | fused-qk
graph: eager | capture | replay
shape: exact rank-local dimensions
```

### 11.2 声明预期路径

```yaml
expected_native_dispatches: 144
expected_fallbacks: 0
expected_phase: batched-prefill
expected_kernel_symbol: native_kv_store_sm100
```

### 11.3 冻结 correctness

```yaml
outputs: tolerance or bitwise
state: written rows + untouched rows
indices: int32 + int64
negative_control: must convict
```

### 11.4 区分 treatment 与 guardrail

```yaml
timed_path_executes_candidate: true | false
claim_type: treatment | no-regression-guardrail
```

### 11.5 在计时后验证 receipt

```text
native count == expected
fallback count == 0
binary / graph node identity matches
correctness still passes
```

任何一项不成立，先停止性能解释。

---

## 12. 如何阅读“开关打开后快了 0.5%”？

看到一个小正数时，按下面顺序提问：

1. **候选命中了吗？** 如果 call count 是 0，收益归因直接为 0。
2. **命中的是 timed phase 吗？** Warmup/prefill 命中不能解释 decode。
3. **Baseline 是同一路径吗？** Backend、layout、Graph topology 必须固定。
4. **Counter 记录 native dispatch 还是 adapter entry？** Fallback 不能冒充命中。
5. **Correctness 覆盖 side effect 吗？** Cache、residual、state、untouched rows 都可能是接口。
6. **合同要证明 win 还是 no-regression？** 不能事后改 gate。
7. **样本支持这个小数吗？** Summary/min/max 不等于 paired raw evidence。
8. **Evidence boundary 在哪？** Operator、Graph、model 与 HTTP 是不同 estimand。

只有这些问题都回答后，百分比才开始有意义。

---

## 13. 这五条实验真正交付了什么？

它们没有交付“五个更快的 kernel”。它们交付了五个被正确分级的结果：

- Add-RMSNorm：prefill ABI 与数值合同成立，decode 只做 guardrail；
- Bare KV store：bitwise state transition、untouched-row oracle 与 144 次 prefill path hit；
- RoPE + KV store：双 oracle class、864 次 prefill path hit，以及最清楚的 zero-call 反例；
- Q/K RMSNorm：same separate-Q/K path 内 latency-neutral；
- SwiGLU：path-hit、小正向、但按预注册合同只晋级为 composable default-off substrate。

这比把所有小正数相加成一个“native kernels waterfall”更诚实，也更有复用价值。

未来 producer fusion、两张 phase-specific Graph 或新的 prefill engine 可以复用这些 ABI 与 correctness primitives；届时需要创建新的 workload cell 和新的 treatment experiment，而不是沿用旧 decode guardrail 的百分比。

---

## 14. 最后记住四句话

1. **开关打开不等于候选命中。** Selector、fallback、phase 与 Graph capture 都可能绕开它。
2. **Call count 为零时，任何正负计时差都没有 candidate attribution。**
3. **Guardrail 可以故意不执行候选；它证明“不破坏”，不证明“更快”。**
4. **同路径、正确性、原始样本和预注册 gate 都成立后，时钟才有解释权。**

因此，优化日志里最应该出现在 latency 之前的字段不是 kernel 名称，而是：

```text
phase / shape / backend / graph identity
native dispatch count
fallback count + reason
correctness receipt
```

先证明路径，再相信时钟。

---

## Evidence boundary

- Source snapshot：`agentic-megakernel@fdf4898`；五个 experiment bundle 均为 `C4/S3` 或等价 bounded record，裁决为 `ENABLED_GUARDED / default-off`。
- GPU / model：1×B200、Qwen3-4B、BF16；Graph guardrail 为 B1、4K/8K、128 decode tokens。
- Prefill-only seams 的 decode Graph 不执行候选；相关小 delta 不构成 speed result。
- Q/K RMSNorm 只比较同一 separate-Q/K backend；不与默认 fused backend 混比。
- SwiGLU 的历史五重复是 guardrail 设计，逐次 raw pairs 未完整归档；正向符号不升级成 formal win。
- 部分 worker commit、standalone runner 与 per-repeat raw 仍有 provenance debt；本站不声称 fresh model replay。
- 本系列的权威状态与重开条件见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
