---
layout: post
title: "什么时候应该停止优化？从 STOP_BEFORE_CODEGEN 到 Leaf Win / E2E Loss"
date: 2026-08-17 12:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [GPU Optimization, Experiment Design, Correctness, Amdahl, Benchmark, Agentic Workflow]
reading_time: 28
cover_image: /assets/blog-stop-optimization.png
excerpt: "真正高效的 GPU 优化不是把所有想法都跑完，而是在 applicability、ABI、correctness、noise、operator、E2E 和 formal holdout 七道门上及时停止。本文用 SGLang-DGMK、agentic-megakernel、SubCUDA 与 OMoE 的正负案例构造一套可执行决策状态机。"
---

> 本文不是“失败实验合集”，而是从多个仓库的 machine-readable verdict 中抽取实验控制流。每个案例只保留其原 evidence level：未运行的层级不会被推断，correctness-rejected arm 没有性能结论，leaf win 也不会升级成 serving win。

GPU 优化最昂贵的资源通常不是 GPU，而是研究者的注意力。

一个候选从想法到部署，可能经历：

```text
看 profile
  → 写 kernel
  → 修编译
  → 修 correctness
  → 调 tile
  → 跑 operator
  → 接模型
  → 跑 E2E
  → 收 profiler
  → 写报告
```

如果这个候选在第一天就只有 0.02% 的端到端上限，后面所有工作都在消耗错误预算。

成熟的优化系统应该把“停止”设计成一等操作，而不是把它当成失败后的无奈。

> **好的 STOP 会关闭一个机制分支、保留证据和重开条件，让未来的 Agent 不再重复同一个错误。**

---

## 1. 七道门：候选不是只有 Winner / Loser 两种状态

一个完整晋级链可以写成：

```text
G0 Applicability / Path
  ↓
G1 Build / ABI / Binary
  ↓
G2 Correctness / State / Numerics
  ↓
G3 Noise / Materiality / Ceiling
  ↓
G4 Operator / Boundary Performance
  ↓
G5 Model / Graph / Serving E2E
  ↓
G6 Formal Pairs / Holdout / Causal Profile
```

![GPU Optimization 的七道晋级与停止门](/assets/blog-stop-gates.svg)

*图 1：每一道 STOP 回答的问题不同；跳过上游 gate，后面的性能数字就没有解释权。*

一个候选可以是：

- `NOT_APPLICABLE`：生产路径不经过目标边界；
- `BUILD/ABI STOP`：没有合法 binary 或 consumer contract；
- `CORRECTNESS REJECT`：结果/state/token 不满足合同；
- `BELOW NOISE`：差异小于 A/A 波动和 materiality floor；
- `LOW CEILING`：即使局部全部消失也达不到 E2E 门槛；
- `OPERATOR REJECT`：正确但边界变慢；
- `E2E REJECT`：局部赢，完整系统输；
- `ACCEPTED_BOUNDED`：只在冻结 cell 通过；
- `PROMOTED`：通过 formal、holdout、causal 和复现门。

这些状态不能压缩成“成功/失败”。

---

## 2. G0：目标根本不在生产路径上，为什么应该在写代码前停？

历史优化中常见：

- candidate 针对 contiguous grouped GEMM，生产却走 masked GEMM；
- archive DSA 使用另一种 KV layout，线上 backend 是 `flashmla_kv`；
- op name 相似，但 `index_q_upproj` 与 attention `q_b_proj` 不是同一个 shape/consumer；
- environment profile 启用，实际 hit counter 为 0；
- graph/eager、M16/M32 或 prefill/decode 不匹配。

如果 selector 永远 miss，candidate latency 没有任何产品含义。

G0 需要：

```text
exact op / phase / local M / N / K
dtype / layout / scale ABI
backend / graph mode
path marker / hit counter
```

这一阶段最有价值的产出可能是：**不写 kernel。**

---

## 3. G1：Build / ABI Stop 不是“还没调好性能”

SGLang-DGMK N36 尝试将 FlashMLA pipeline 从 2 stage 增至 3 stage。Shared-memory 总预算看起来仍低于 B300 上限，但 24-byte validity 区把 `tma_coord` 推离 16-byte 对齐，首次同步 launch 报 misaligned address。

这时没有合法性能数字。不能说“候选可能更快，但偶尔 crash”。它是 ABI/alignment correctness stop。

N37 修复 16-byte alignment 后，planned shared memory 为 `232432 B`，距离 `232448 B` 上限只剩 16 B。Leaf 快约 `0.816%`，才有资格进入下一门。

这个例子说明：

```text
shared memory 总字节数通过
≠
每个对象的 alignment / lifetime / TMA ABI 通过
```

Build gate 应保存：

- target arch；
- entry symbol；
- registers/SMEM/stack/spill；
- alignment；
- SASS 指纹；
- exact launch；
- unsupported fail closed。

---

## 4. G2：Operator Exact 也可能不够——D058 为什么停止在模型 Token？

D058 将 TP2 finalize geometry 从 `96×4` 改为 `192×2`。

Synthetic operator：

```text
14.406328 → 12.497464 μs
−13.2502%
8/8 wins
random / finite-edge residual and norm byte-exact
```

看起来已经非常强。但 full-model qualification 在任何 timing 前比较 token hash：两个 rank 都与 baseline 不同。Harness 立即中止，`timing_started=false`。

原因是 192×2 进入另一棵 FP accumulation/reduction tree。Synthetic fixture 没覆盖真实模型触发的数值路径。

因此正确结论是：

```text
operator sampled correctness: PASS
model exactness: FAIL
E2E performance: NOT MEASURED
```

禁止写：

```text
candidate model speedup = 13.25%  ×
```

correctness-rejected arm 不需要再收 NCU 来“解释它为什么本来会快”。

---

## 5. G3a：低于 Noise Floor 的“胜利”为什么应关闭？

SubCUDA D017 将 16 个 state store 改为 `st.global.L1::evict_last`。候选 byte-exact，在 30 个 block 中赢 21 个，节省 `0.00456 μs`。

但 A/A absolute paired p95 是 `0.014276 μs`，预注册 materiality floor 是 `0.03 μs`。

$$
0.00456
<0.014276
<0.03\ \mu s
$$

候选差异连 baseline 自身波动都没超过。

这时不能通过：

- 增加更多 runs 直到 p-value 好看；
- 删除“不稳定”样本；
- 把 21/30 wins 写成趋势；
- 改用 NCU instrumented duration。

正确状态是 `REJECTED_BELOW_NOISE_FLOOR`，重开条件是新机制把 expected saving 提高到 materiality 以上。

---

## 6. G3b：局部快 33%，为什么仍不值得跑模型？

D086 将 embedding RMSNorm 与 FP8 prelude quant 融合：

```text
6.145296 → 4.101072 μs/step
−33.2649%
byte-exact
```

但该 boundary 每 decode step 只调用一次：

$$
2.044224\ \mu s\times128
=0.261661\ \mathrm{ms/graph}
$$

完整 graph wall 约 `1157.576446 ms`，乐观上限：

$$
\frac{0.261661}{1157.576446}
=0.0226\%
$$

低于 0.05% promotion floor，所以不加载 122B 模型、不跑 E2E。

这是 `REJECTED_LOW_CEILING`，不是 operator failure。Kernel 是正确且更快的，但项目优先级不成立。

先算 frequency × saving × critical exposure，是最便宜的 GPU 预算管理。

---

## 7. G4：Correct but Slower——D001 为什么 0/30 后不需要 Profile？

GDN D001 将 recurrent-state load 提前，试图隐藏 long-scoreboard latency。

结果：

```text
control: 8.912480 μs
candidate: 9.115320 μs
candidate +2.276% slower
0/30 wins
byte-exact
```

更深 retime 还出现 stack/local spill，静态淘汰。

PtXas 本来已经完成有用 interleave；手工提前只扩大 live range、把 registers 64→71，并改变全函数调度。

差异远大于 A/A noise，30 个 block 全输。此时再收 profiler 不能把候选“解释成胜利”。Profiler 只有在存在一个 bounded fixable loss、计划新候选时才有价值。

正确状态：`REJECTED_OPERATOR`。

---

## 8. G5：Leaf Win / E2E Loss 是最重要的负结果

### 8.1 SGLang N40

Contiguous SwiGLU + FP8 quant fusion：

```text
leaf: 0.045245 → 0.018443 ms
−59.1%
```

模型 correctness/path hit 通过，但 development bracket：

```text
P50 TTFT +1.56%  (更慢)
P90 TTFT +18.61%
throughput −6.01%
```

并出现 candidate-only `empty_chunked_topk`。Leaf 实现保留为研究证据，候选从服务 baseline 回滚。

### 8.2 SubCUDA D036

Route-first boundary：

```text
isolated: −1.523%, 30/30
TP2: −0.140487%, 0/2
```

Systems 还发现没有实际 overlap，只是串行重排。

![同一个局部 Winner 如何在完整系统中被反转](/assets/blog-stop-leaf-e2e.svg)

*图 2：Operator 只测局部边界；E2E 还包含频次、其他节点、通信、stream、rank tail 与新增调度成本。*

这种结果应该标为 `E2E_REJECTED`，而不是“接近打平，继续默认开着”。

---

## 9. G6：Development 看起来快，为什么 Formal 仍可能不准入？

SGLang N35 将 N32 leaf primitive 打包进服务。第一次 bracket control drift `+6.345%`，表面候选快 `2.588%`，但整组作废。

重试 control drift `−3.060%`，只比 3% 有效阈值多 `0.060` 个百分点；方向性 P50 约快 2.096%，但没有进入 formal。

实验控制 drift 本身是 gate。候选看起来快，不能覆盖 control 不稳定。

正式阶段应冻结：

- A/B 顺序；
- baseline-before / baseline-after drift；
- 机器健康；
- 样本数；
- 最少 pair wins；
- primary/tail/throughput 门；
- 独立 seed holdout。

无效 bracket 不是“低质量但可参考的正结果”，而是不能参与晋级。

---

## 10. 真正的 Promote：N6 为什么比一个漂亮 Median 多很多？

GLM-5.2 N6 的晋级链：

```text
strict workload / path marker / map SHA
  ↓
generated tokens exact against 2 references
  ↓
5 fresh-server pairs
  ↓ 4/5 P50 wins
P50 −5.31%, P90 −4.75%, throughput +7.53%
  ↓
independent client-seed holdout
  ↓ 5/5 P50 wins
P50 −5.46%, P90 −6.73%, throughput +7.40%
  ↓
matched Nsys causal explanation
```

Profile 只解释 progress calls、notify tail 和 mask-kernel removal，不参与正式百分比。

这才是 `ACCEPTED_BOUNDED`：结论只覆盖冻结 100K cached-prefill cell，不推广到 decode、其他 EP 或在线流量。

---

## 11. 一张完整的决策状态机

![从候选到 bounded promotion 的实验决策状态机](/assets/blog-stop-state-machine.svg)

*图 3：STOP 不是统一红叉；每个状态都保存已证明内容、未运行层级和明确重开条件。*

伪代码：

```python
if not path_matches_contract:
    return NOT_APPLICABLE

if not build_or_abi_valid:
    return BUILD_STOP

if not operator_and_state_correct:
    return CORRECTNESS_REJECT

if expected_e2e_ceiling < materiality_floor:
    return LOW_CEILING_STOP

if delta_within_aa_noise:
    return BELOW_NOISE_STOP

if operator_slower:
    return OPERATOR_REJECT

if not full_model_correct:
    return MODEL_CORRECTNESS_REJECT

if not e2e_faster:
    return E2E_REJECT

if not formal_pairs_and_holdout_pass:
    return CONTINUE_OR_REVERT

return ACCEPTED_BOUNDED
```

---

## 12. “没跑”与“跑了失败”必须分开

| 状态 | 可以说什么 | 不可以说什么 |
| --- | --- | --- |
| STOP_BEFORE_CODE | 上限/路径不成立，未实现 | candidate 慢 |
| BUILD_STOP | 无合法 binary/ABI | 性能不佳 |
| CORRECTNESS_REJECT | 候选不满足语义 | 若修正确性会快多少 |
| LOW_CEILING | 局部正确/可能快，但系统价值不足 | E2E 已打平 |
| OPERATOR_REJECT | 匹配边界上正确但更慢 | 模型一定更慢 |
| E2E_REJECT | 完整合同下不替换 baseline | 局部机制无价值 |
| ACCEPTED_BOUNDED | 冻结 cell 可采用 | 普遍更快 |

这种语言纪律能防止 archive 中的“没有数据”被后来的人误读成“接近成功”。

---

## 13. 为什么不能事后修改 Gate？

常见事后操作：

- candidate 只赢 P90，就把 primary 从 P50 改成 P90；
- 4/5 wins 没达到，就改成 median 更好；
- 差异低于 1%，把 gate 改成 0.1%；
- correctness 差一个 token，就改成 cosine；
- operator 输，把 NCU 某个 counter 设为新目标；
- formal 不稳，换 workload 到一个更好看的 cell。

这些都可能产生新研究问题，但必须是**新合同、新 candidate record、新实验**。不能修改旧 verdict。

Negative/null result 的价值正是关闭原分支。如果 gate 随结果变化，STOP 就失去约束力。

---

## 14. 一个可供 Agent 使用的 Candidate Record

```text
Hypothesis
  work removed / overlapped:
  critical interval:
  expected saving:
  new cost:
  rejection evidence:

Contract
  workload / model / hardware / path:
  baseline / selector:
  correctness endpoints:
  noise / materiality:
  sample plan / stop rule:

Evidence
  build / binary:
  operator correctness:
  operator timing:
  model trajectory:
  E2E pairs:
  profiles:

Decision
  state:
  what is proved:
  what was not run:
  rollback:
  reopen condition:
```

Agent 在提出下一候选前，先检索相同 mechanism + precondition 的 accepted 与 rejected records。这样 negative evidence 才能真正减少重复搜索。

---

## 15. STOP 也需要 Source of Truth

负结果至少保存：

- candidate ID；
- exact source/binary revision；
- contract hash；
- path marker；
- correctness outputs/state；
- raw samples与顺序；
- A/A noise；
- verdict；
- 未运行层级；
- reopening condition。

不能只在周报写一句“试过，没用”。否则未来 Agent 无法判断：

- 是机制无效；
- 是 shape 不匹配；
- 是代码没命中；
- 是 correctness 失败；
- 是 noise 太大；
- 还是 evidence 丢失。

---

## 16. 什么时候应该继续，而不是立即停止？

只有在 larger boundary 已经运行、结果更慢，但 profiler 指出一个**具体可删除损失**时继续：

- unexpected spill；
- 一个多余 copy/adapter；
- register/SMEM cliff；
- broken overlap；
- rank skew；
- wrong launch geometry；
- stale descriptor；
- graph node 额外出现。

下一步必须只修改这个损失，并有 bounded iteration budget。

“再调几个 tile 看看”不是 continue rationale。

---

## 结语：停止是搜索算法的一部分

这些仓库中最有价值的知识不只是 winner：

- N36 告诉我们 ABI/alignment 未过时没有性能数字；
- D058 告诉我们 sampled operator exact 不能替 model token gate；
- D017 告诉我们低于 noise floor 的正方向不算收益；
- D086 告诉我们局部快 33% 也可能不值得跑 E2E；
- D001 告诉我们 correct-but-slower 应在 operator 门关闭；
- N40/D036 告诉我们 leaf win 可以在系统中反转；
- N35 告诉我们 control drift 会让漂亮 bracket 作废；
- N6 告诉我们真正 promotion 需要 formal、holdout 和 causal explanation。

所以，一个高效的 Agentic optimization loop 不应只问：

> “下一步还能试什么？”

还要问：

> **“哪条证据已经足以关闭当前分支？停止后，我们保存了什么，未来在什么新前提下才允许重开？”**

当 STOP 变得可执行、可审计、可复用，优化才不再是无限枚举，而是一棵不断被证据剪枝的搜索树。
