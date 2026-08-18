---
layout: post
title: "为什么节点快 46.7%，Joined DAG 反而更慢？从 Slack、Max-Join 到 Launch Fusion"
date: 2026-08-17 18:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Critical Path, CUDA Graph, Kernel Fusion, PDL, Multi-Stream, Profiler, B200]
reading_time: 30
cover_image: /assets/blog-isolated-win-joined-loss.png
excerpt: "R99 用一个 32×96 CUDA launch 替换 cuBLASLt split-K + reduction，isolated NCU 快 46.74%，两次 joined no-profiler 却都回退；R92 融合 finalize、RMSNorm 与 FP8 quant，局部省 1.327 μs，TP2 仍 0/2。本文解释 local saving、branch slack、join exposure 与调度自由度之间的关系。"
---

> 本文基于 [`qhy991/SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab) 的 R99 shared-gate 与 R92 finalize+group-FP8 cases。两条 machine-readable replay 与资产检查已通过。R99 的 source/probe 已归档但 fresh build wrapper 尚未参数化；R92 候选 source/build recipe 确实缺失，因此只能重放冻结 contract/evidence，不能用当前 Round91 替代。

一个 kernel 快了 `46.74%`，完整边界却连续两次变慢。

另一个 fusion 少了两个 launch，局部省 `1.327369 μs`，完整 TP2 吞吐却：

```text
0 / 2 wins
```

这不是 GPU 性能测量“随机得无法相信”。它揭示了一个更根本的问题：

> **局部节点时间不是 DAG 完成时间。只有暴露在最终 max-join 上的节省，才能直接缩短 wall-time。**

更复杂的是，修改一个节点还会改变：

- 与其他 stream 的 overlap；
- SM/L2/HBM 竞争；
- PDL producer-consumer 窗口；
- register/shared-memory lifetime；
- CUDA Graph node topology；
- join 的最晚到达分支。

所以“把局部 saving 乘调用次数”通常只是 ceiling，不是 E2E 事实。

---

## 1. 初学者先理解 Max-Join

假设两条独立分支在 join 前运行：

```text
branch A: 20 μs
branch B: 30 μs
```

Join 最早完成时间是：

$$
T_{\mathrm{join}}
=
\max(T_A,T_B)
=30\ \mu s
$$

Branch A 有：

$$
S_A=T_B-T_A=10\ \mu s
$$

的 slack。

即使 A 快 8 μs：

```text
A: 20 → 12 μs
B: 30 μs
```

Join 仍是 30 μs。

在不考虑资源干扰时，A 的 exposed saving 近似：

$$
\Delta T_{\mathrm{exposed}}
=
\max(0,\Delta T_A-S_A)
$$

若 local saving 小于原 slack，完整 DAG 看不到直接收益。

![局部节点 Saving 与 Join Slack](/assets/blog-join-slack-exposure.svg)

*图 1：非关键分支可以大幅变快而不改变 max-join；若候选还扰动关键分支，完整时间甚至会变慢。*

---

## 2. R99 想优化什么？

Qwen3.5 TP2 的 shared gate 是：

$$
[32,3072]\times[3072,1]
$$

Incumbent 使用：

```text
cuBLASLt three-CTA split-K producer
  → separate reduction launch
  → BF16 gate[32]
```

R99 编写 source-CUDA kernel：

```text
每 token 一个 96-thread block
三个 warp 各计算连续 1024 columns
固定顺序合并三个 partial
单 launch 输出 BF16 gate[32]
```

目标不只是独立 GEMV 更快，而是未来把 gate dot 放到 finalize 的 PDL wait 前，利用逻辑独立区间。

---

## 3. Isolated 结果有多漂亮？

| Scope | Control | Candidate | Result |
| --- | ---: | ---: | ---: |
| Cold-L2 CUDA events | 10.272 μs | 8.288 μs | −1.984 μs |
| Isolated NCU replay | 11.776 μs | 6.272 μs | **−46.74%** |

Candidate：

- 一个 `32×96` launch；
- 31 registers；
- 1,036 B shared memory；
- 无 spill。

它替代了 producer 的 255-register 形态和第二个 reduction launch。

Profiler 的机制结论是真实的：

> **Shared gate 节点本身更快，launch topology 也更简单。**

但它还有另一面：

```text
0.0103 waves / SM
4.60% achieved occupancy
0.438% peak DRAM
long-scoreboard ratio = 15.347
```

这是一个非常小的 grid。独立运行时省下的时间，不一定暴露在 routed/shared joined critical path 上。

---

## 4. Joined Boundary 为什么连续回退？

R99 在九个生产 route profiles 上保持 BF16 exact，然后测量完整 routed/shared join。

| Run | Candidate − Control | Projection |
| --- | ---: | ---: |
| Joined no-profiler run 1 | +0.3573 μs / boundary | +2.195 ms / graph |
| Joined no-profiler run 2 | +0.2293 μs / boundary | +1.409 ms / graph |

两次方向一致：candidate 更慢。

![R99 Isolated Gate 与 Joined DAG 的相反裁决](/assets/blog-r99-isolated-joined.svg)

*图 2：Gate 节点自身大赢，但它不拥有 joined completion time；完整 boundary 由 routed/shared/communication 的最晚到达与共享资源竞争决定。*

可能出现的情况：

1. Incumbent gate 原本隐藏在 routed branch slack 内；
2. Candidate 虽更短，却改变 launch 时刻，与关键 branch 争用 SM/issue；
3. Small grid 不能有效填充空闲资源，反而增加一次新调度事件；
4. Join 等待的仍是 routed/shared/peer path；
5. Candidate 的资源形态改变相邻节点的开始时间。

冻结 evidence 没有唯一识别上述哪项为主因；我们只能说 local saving 没有暴露，并且 joined authority 回退。

---

## 5. 正确性为什么不是“所有输入 Exact”？

R99 使用真实 gate weight，测试：

- zeros；
- ones；
- alternating signs；
- 多 seed；
- 多 input scale；
- 九个生产 route profiles。

RMS-normalized、scale=1 的生产域输入与 joined profiles byte-exact。

但全局 `0.01×` synthetic input 出现一个 BF16 rounding difference。

因此候选不是通用 `F.linear` byte-exact 替代品，只能作为生产域 bounded experiment。

这进一步阻止它在 joined 回退后被包装成“未来可默认替换的通用 primitive”。

---

## 6. Profiler 为什么与 No-Profiler Authority 冲突？

Systems node tracing 观察到：

```text
gate nodes 代表性减少 ≈3.008 μs
profiled graph span 也偏向 candidate
```

而两次 no-profiler joined oracle 都回退。

这两组事实可以同时成立，因为 profiler 可能：

- 插入同步；
- 序列化或延长短节点；
- 改变 CUDA Graph node tracing 成本；
- 改变 stream overlap；
- 对微秒级 small grid 施加不对称扰动；
- 测量 node duration，而非最晚 join completion。

正确裁决顺序：

```text
no-profiler joined wall → admission authority
profiler nodes/timeline → mechanism explanation
```

Profiler 不能因为“看起来更符合理论”就覆盖 no-profiler 结果。

---

## 7. R92：少两个 Launch，为什么 TP2 仍不赢？

R92 优化 12 个 MoE→full-attention 边界。

旧路径：

```text
TP2 finalize
  → exact RMSNorm
  → group-128 E4M3 quant
  → full-attention GEMM
```

候选：

```text
single owner:
finalize + RMSNorm + group-FP8 quant
  → full-attention GEMM
```

它沿用 finalize 的每 token 四 CTA cluster，在通信完成后继续执行：

- 固定顺序 FP32 square reduction；
- `rsqrt.approx.ftz`；
- Qwen offset-gamma；
- 明确 BF16 norm rounding seam；
- group-128 absmax/scale/E4M3 payload。

Correctness 要求先 round 到 BF16 norm，再由这些 BF16 bytes 量化；直接从 FP32 norm 量化是另一个 numerical program。

---

## 8. R92 局部通过，E2E 如何反转？

Operator：

```text
15.678848 → 14.351480 μs
saving = 1.327369 μs
```

按 1,536 boundaries 投影：

```text
−2.038838 ms / graph
```

但 TP2 prescreen：

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Throughput | 3530.225939 tok/s | 3529.846053 tok/s |
| Wall | 1160.265680 ms | 1160.390562 ms |
| Paired wins | — | 0/2 |

Qualification token hash exact，性能 admission 仍失败。

原因不是“两微秒太小”这么简单。旧的三个 kernel 不一定串行相加：

- 它们位于 PDL 链；
- 使用双 stream；
- 与 routed/shared/communication overlap；
- 分离节点给 scheduler 更多交错自由；
- Fusion 扩大 finalize owner 的工作和资源 lifetime。

![R92 删除 Launch 但改变 Overlap 的执行图](/assets/blog-r92-fusion-overlap.svg)

*图 3：旧节点的 isolated duration 可以相加，但它们的暴露 wall 不一定相加；fusion 删除节点同时也删除调度边界，可能把原本隐藏的工作串到 owner critical path。*

---

## 9. 为什么 “Local Saving × Calls” 只能是 Ceiling？

简单投影：

$$
\Delta T_{\mathrm{projected}}
=N_{\mathrm{calls}}\times\Delta T_{\mathrm{local}}
$$

它隐含假设：

1. 每次 local interval 完全暴露；
2. 调用之间没有 overlap；
3. 候选不改变其他节点；
4. 没有资源竞争；
5. Graph schedule 不变；
6. 慢 rank / join 不变。

在多 stream/PDL/communication DAG 中，这些假设通常不成立。

更现实地：

$$
\Delta T_{\mathrm{graph}}
=
\Delta T_{\mathrm{exposed}}
-\Delta T_{\mathrm{interference}}
-\Delta T_{\mathrm{lost\ overlap}}
$$

其中 exposed saving 还受 branch slack 限制。

所以投影适合：

- 判断理论上限是否值得跑 E2E；
- 排除极低 ceiling；

不适合：

- 宣布模型收益；
- 把局部微秒直接换算 throughput。

---

## 10. Fusion 的真正代价：减少调度自由度

Fusion 常被描述为：

```text
减少 launch
减少中间 tensor
增加 locality
```

但它也会：

- 合并原本可独立调度的节点；
- 延长寄存器/SMEM live range；
- 把不同 resource phase 锁进同一 CTA；
- 让后续 consumer 必须等完整 owner；
- 减少 scheduler 在不同 stream 间穿插的机会；
- 改变 PDL trigger 与 dependency sync 的位置。

因此 fusion 是否值得，取决于：

$$
T_{\mathrm{materialization+launch\ removed}}
>
T_{\mathrm{overlap\ lost+resource\ pressure}}
$$

不是只看 kernel 数量。

---

## 11. 如何测量 Exposed Saving？

推荐三层 oracle。

### Layer 1：Isolated operator

确认 primitive 本身：

- 正确；
- path-hit；
- 有 material local saving；
- 机器码/资源符合预期。

### Layer 2：Joined boundary

把 candidate 放回真实 sibling branches、streams、waits、join：

```text
start before branches
stop after real join
```

它回答 saving 是否暴露。

### Layer 3：Model / serving

完整 Graph、TP ranks、token hash、throughput/wall。

只有 Layer 2/3 能支持对应层级 promotion。

---

## 12. Profiler 应该在什么时候进入？

先跑 no-profiler admission，再 profile：

```text
Correctness
  → isolated A/B
  → joined no-profiler A/B
  → model no-profiler A/B
  → profiler attribution
```

Profiler 用于回答：

- 哪条 branch 最晚；
- candidate 是否改变 start time；
- SM/L2/HBM 是否竞争；
- PDL overlap 是否存在；
- fusion 是否延长 owner；

而不是重新裁决 winner。

R99 已经展示：profiler node 变短与 joined wall 变长可以同时为真。

---

## 13. 怎样在写 Fusion 前做 Slack Screen？

画出：

```text
branch start
node start/end
sibling path end
join time
```

估计：

$$
S_{\mathrm{node}}
=T_{\mathrm{join}}-T_{\mathrm{node\ end}}
$$

若 predicted local saving：

$$
\Delta T_{\mathrm{local}}\le S_{\mathrm{node}}
$$

则直接 E2E 收益先验很低，除非候选还删除 shared resource contention 或 downstream dependency。

同时检查：

- Fusion 后 owner resource footprint；
- consumer release 时刻；
- 原节点是否已 overlap；
- sibling branch 是否变慢；
- slow-rank join 是否改变。

---

## 14. R99 与 R92 分别关闭了什么？

### R99

关闭：把 isolated one-launch gate 直接集成到 finalize、期待 joined win 的当前方案。

保留：若 routed/communication critical path 改变，或 gate 可以删除 materialized edge，而不只是替换隐藏节点，可新建实验。

### R92

关闭：当前 finalize+norm+quant fusion 在冻结 TP2 Graph 上 default-on promotion。

保留：若能保持原 stream/PDL overlap、缩短 owner exposed interval，或新的 Graph topology 让旧节点真正串行，可重开。

---

## 15. 最后记住

1. **Node duration 不是 Graph wall。**
2. **非关键分支的 local saving 会被 slack 吸收。**
3. **Fusion 删除 launch，也可能删除 overlap 和调度自由。**
4. **Profiler 解释节点，no-profiler joined/E2E 裁决完成时间。**
5. **Local saving × call count 是 ceiling，不是收益。**

真正的问题不是：

> “这个 kernel 快了多少？”

而是：

> **“它省下的时间有多少暴露在最终 max-join 上，并且有没有让真正的关键分支变慢？”**

---

## Evidence boundary

- Source snapshot：[`SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab)。
- R99 replay/asset check 通过：九个生产 profiles exact，第二次 joined candidate 回退 `+0.229334 μs/boundary`，不具 finalize integration 资格；synthetic 0.01-scale 有一个 BF16 difference。
- R92 replay/asset check 通过：qualification token hash exact，TP2 throughput baseline `3530.225939` 高于 candidate `3529.846053`，0/2。
- R99 profiler node/span 只作 attribution，不能覆盖两次 no-profiler joined rejection。
- R92 fresh build 因候选 source/build recipe 缺失而 BLOCKED；不能用当前 Round91 替代。
- 本文没有把局部 projection 写成 E2E 实测，也不指定未被证据唯一识别的微架构根因。
- 状态与重开条件见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
