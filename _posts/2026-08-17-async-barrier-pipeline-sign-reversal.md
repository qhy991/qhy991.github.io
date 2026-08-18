---
layout: post
title: "为什么“更异步”会变慢？从 Named Barrier、Phase Fusion 到 TMA Pending Store"
date: 2026-08-17 06:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [CUDA, Async Pipeline, Named Barrier, TMA, PTX, FlashInfer, MegaKernel]
reading_time: 28
cover_image: /assets/blog-async-sign-reversal.png
excerpt: "同一类异步技巧在不同 kernel 中为什么会变号？本文比较 16-warp single decode、4-warp paged batch decode 和 Hazy resident full-model Graph，解释 ownership、occupancy、dependency window 与 live state 如何决定 named barrier、phase fusion、pending store 和 L2 prefetch 的真实价值。"
---

> 本文联合使用两条证据线：SubCUDA 的 FlashInfer Direct-PTX/CUDA operator cases，以及 agentic-megakernel 中 Hazy 的 full-model CUDA Graph bundles。前者解释 barrier、phase 和 instruction-pipe 机制，后者负责模型级正负裁决。所有百分比都绑定原 workload 和 evidence level，不把 operator win 外推成 serving win。

GPU 优化里最容易令人上瘾的词之一是“异步”。

看到同步点，我们想删 barrier；看到内存访问，我们想 prefetch；看到 producer/consumer，我们想 double buffer；看到 TMA store，我们想先 commit、以后再 wait。

但四组真实实验给出了完全不同的结果：

| 场景 | 异步修改 | 结果 |
| --- | --- | ---: |
| FlashInfer single decode | per-`tz` named barrier | 单项约 **−4.45% latency** |
| FlashInfer batch decode | phase fusion | **慢 2.9%–4.5%** |
| Hazy full-model Graph | depth4 + pending1 TMA store | **慢 0.704%** |
| Hazy full-model Graph | future-weight L2 prefetch | 6 个 cell 无一过 1% gate |

同一个词，为什么会变号？

因为“异步”不是收益。它只是把工作和等待重新排列。真正的收益可以粗略写成：

$$
\Delta T
=
-T_{\mathrm{hidden\ stall}}
+T_{\mathrm{bookkeeping}}
+T_{\mathrm{live\ state}}
+T_{\mathrm{resource\ pressure}}
+T_{\mathrm{dependency\ extension}}
$$

只有被隐藏的关键路径 stall 大于新增成本，完整时间才会下降。

---

## 1. 初学者先记住：Barrier 保护的是所有权，不是时间

Barrier 的作用不是“让程序慢一点”，而是保证某些线程在读取共享数据前，必要的写入已经完成；或者保证 buffer 被下一阶段覆盖前，上一阶段所有读者已经离开。

判断一个 barrier 能否缩小，首先要回答：

```text
谁写这块 shared memory？
谁读这块 shared memory？
哪些线程之间真的互不依赖？
下一阶段什么时候复用这块内存？
```

如果四组 warp 各自拥有完全独立的 slice，那么全 block barrier 可能过宽。如果 consumer 会跨组读数据，拆成 named barrier 就会造成数据竞争。

![独立 warp-group ownership 与交叉读取的区别](/assets/blog-async-barrier-ownership.svg)

*图 1：Named barrier 只有在参与者、数据 ownership 和复用时刻都可分组时才合法。*

这条规则比“barrier stall 很高”更上游。Profiler 告诉你哪里在等；ownership 才决定能不能安全地少等。

---

## 2. 正例：16 Warp Single Decode 为什么适合 Named Barrier？

SubCUDA 的 Day4 目标是 FlashInfer single GQA decode：

| 维度 | 值 |
| --- | --- |
| GPU | B200 / SM100a |
| CUDA | 13.2.78 |
| Dtype | BF16 |
| Q / KV heads | 64 / 8 |
| Head dimension | 128 |
| Block | 512 threads，16 warps |
| `tz` groups | 4 组，每组 128 threads |
| Residency | 约 1 block / SM |
| Timing | 完整 decode + merge operator |

主循环的 K/V shared-memory slice 按 `(stage, tz)` 独立分开。四个 `tz` group 在循环内没有跨组数据依赖，却都使用：

```ptx
barrier.sync 0;
```

这会把 16 个 warp 锁成同一相位：一起完成 copy，一起冲向 MIO，一起做 FMA，再一起等待。即使每组数据早已就绪，也要等另外三组。

H1 将循环内 12 个全 block barrier 概念上改成：

```ptx
barrier.sync barrier_id_for_tz, 128;
```

四个 group 分别使用 barrier ID 1–4，ID 0 保留给循环外真正需要全 block 同步的位置。

### 它为什么快？

有趣的是，收益不只是“barrier stall 下降”。H1 的 barrier stall 甚至可能略升，但四组 warp 不再同时冲击同一执行管线：

| 指标 | Before | After H1 |
| --- | ---: | ---: |
| MIO throttle | 0.21 | 0.06 |
| Math throttle | 0.91 | 0.81 |
| Issue rate | 0.71 | 0.75 instr/cycle/scheduler |

单项 latency 改善约 `4.45%`。

这是一种“去同相化”效应：不是让每个 warp 做更少工作，而是让四组 warp 的 load/compute burst 错开，改善 scheduler 和管线利用。

### 为什么循环后的 barrier 不能一起改？

循环结束后，`sync_state` 会复用 shared memory 的前 16 KiB。如果 `tz0` 提前覆盖，而 `tz1` 尚未完成读取，结果会被破坏。

所以正确修改不是“把所有 barrier 换成 named barrier”，而是：

```text
循环内：slice ownership 独立 → group barrier
循环外：shared region 即将整体复用 → full block barrier
```

---

## 3. 正例中的第二层：Phase Fusion 为什么只带来小幅增益？

同一个 single-decode kernel 中，`compute_qk` 与 `update_local_state` 之间的必要中间值已经在线程私有寄存器中，不需要 shared-memory barrier 保护。

H3 将每轮：

```text
4 barriers + 2 waits
```

压缩成：

```text
2 barriers + 1 wait
```

barrier stall 从约 `0.55` 降到 `0.39`，但增量收益只有约 `0.56%`。

为什么比 named barrier 小？因为 barrier 并不是全部关键路径。删掉同步后，QK、state update、BF16 conversion、load/store 和固定 merge 仍然存在。

Day4 最终 H1 + H3 + H4 stack 在 KV8192 从：

$$
26.159\ \mu s
\longrightarrow
23.770\ \mu s
$$

改善 `9.13%`，但其中很大一部分来自 H4 将部分 BF16 widening 从拥塞 ALU/SHF 管线移到有余量的 IMAD/FMA 管线。它不是“全靠少 barrier”。

SubCUDA 当前机器可读 replay 还证明：12 个冻结 workload cell 全部获胜，改善约 `7.67%–11.98%`。但这仍是 accepted operator，不是 full-model E2E。

---

## 4. 第一次变号：Batch Decode 为什么不能复制 Named Barrier？

Day5 的 paged batch decode 看起来也是 attention decode，但执行形态完全不同：

| 维度 | Single Decode | Batch Decode |
| --- | ---: | ---: |
| Threads / block | 512 | 128 |
| Warps / block | 16 | 4 |
| Blocks / SM | 约 1 | 约 8 |
| 独立 `tz` slices | 4 | `bdz=1`，不存在 |
| 额外结构 | 较简单 | page table、DecodePlan、变长请求、merge |

在 batch kernel 中，没有四个独立 `tz` group。尝试按 `ty` 分组也不安全，因为 consumer 会跨 `ty` slot 读取 shared data。

这意味着 named barrier 不是“效果可能不好”，而是 ownership 前提不成立，静态上就应该停止，根本不应消耗 GPU 测量预算。

可迁移机制必须匹配：

$$
\text{same instruction name}
\not\Rightarrow
\text{same ownership and synchronization contract}
$$

---

## 5. 第二次变号：少了 Barrier，Phase Fusion 为什么反而慢？

Batch decode 的 phase fusion 在五个 cell 中回退约 `2.9%–4.5%`。

原路径允许：

```text
K ready
   ↓
立即开始 QK

V 稍后 ready
   ↓
进入 state/value phase
```

融合后变成：

```text
等待 K + V 都 ready
        ↓
QK 和 state/value 连续执行
```

表面上 barrier 和 wait 少了，但 QK 的最早开始时刻被推迟。依赖链变长，short-scoreboard 和 wait 成本增加。

![减少同步数量与延长依赖窗口的权衡](/assets/blog-async-dependency-window.svg)

*图 2：优化的是 critical-path start time，不是 barrier count。少一个 barrier 但晚开始 QK，完整 envelope 仍可能更长。*

此外，4-warp block 的 barrier 本来就便宜；约 8 blocks/SM 已经天然让不同 block 错峰。single decode 中“16 warp 同相冲击”的主要问题，在 batch decode 里并不存在。

因此同一个 phase-fusion 补丁会变号：它删除的成本变小了，新增的依赖成本却更大。

---

## 6. 仍然能迁移的是什么？Opaque BF16 Conversion

Day4 的 H4 将一部分 BF16→FP32 位模式扩展从 ALU/SHF 搬到 IMAD/FMA。这个机制依赖的是执行管线压力，而不是 warp-group ownership，因此在 batch decode 中仍然成立。

Direct PTX 在五个 production-Plan cell 中改善 `2.92%–3.50%`：

| Cell | Baseline | Direct PTX | 改善 |
| --- | ---: | ---: | ---: |
| B32 / KV1024 | 83.17 μs | 80.74 μs | 2.92% |
| B32 / KV8192 | 581.78 μs | 561.39 μs | 3.50% |
| B128 / KV1024 | 290.17 μs | 280.42 μs | 3.36% |
| B128 / KV8192 | 2281.50 μs | 2205.52 μs | 3.33% |
| B64 mixed | 758.53 μs | 732.22 μs | 3.47% |

更有意思的是，同一机制翻译回 CUDA source 后，编译器获得了更大的全局重排空间，目标 SASS 从 1856 条降到 1664 条，五格改善扩大到 `11.49%–13.69%`。

这说明：

> **迁移的应该是“压力从哪条管线搬到哪条管线”的机制，而不是复制一段 PTX 文本。**

而且 CUDA-source gain 与 Direct-PTX gain 实现的是同一机制，不能相加。

---

## 7. 第三次变号：更深 Ring + Pending TMA Store 为什么变慢？

Hazy 的 full-model resident CUDA Graph 实验针对 Llama-3.2-1B、BF16、B1、position 0、1×B200。

候选使用 depth-4 output ring，并允许最多一个 TMA store group 保持 pending：

```text
compute next UpGate pair
        ↕
previous store remains pending
        ↓
wait preceding group
        ↓
publish Down input
        ↓
release ring stages
```

理论上，compute 可以覆盖 store latency。但这也增加：

- 一个额外 ring stage 的 shared-memory lifetime；
- pending group bookkeeping；
- commit/wait 指令；
- buffer 何时可复用的约束；
- 更长的寄存器 live range；
- final drain 和 publish 顺序。

100/100 full-model output bitwise correct，但 no-profiler captured Graph latency：

| Arm | Median latency |
| --- | ---: |
| Stock | 687.597 μs |
| Identity depth3 | 688.350 μs |
| Depth4 control | **677.521 μs** |
| Depth4 + pending1 candidate | **692.435 μs** |

预声明 candidate 是 `depth4 + pending1`，它相对 stock 慢 `0.704%`。不能在看到结果后把 depth4 control 偷换成“候选成功”。

这组结果告诉我们：depth4 可能有价值，但“再让一个 store pending”没有隐藏新的关键 stall，新增同步和 lifetime 成本反而超过收益。

一个准确的 reopening condition 是：只有新的 profiler 证据再次显示该 store wait 位于关键路径，才值得重新设计 pending policy。

---

## 8. 第四次变号：L2 Prefetch 为什么六个 Cell 都没过门？

另一个 Hazy 实验在 instruction `i` 结束前，提前读取 `i+1` 或 `i+2` 的首个 16 KiB DownProj weight tile，希望 consumer 到来时命中 L2。

合同覆盖：

- Llama-3.1-8B；
- BF16；
- B1；
- position 0、4095、8191；
- prefetch distance 1 和 2；
- 每 cell 100 个 alternating CUDA-event samples；
- resident full-model CUDA Graph latency；
- 1% performance gate。

Source、binary 和 SASS 都证明 prefetch 指令真实存在，positions 4K/8K 的 16/16 top-1 correctness 也通过。

但六个 cell 中五个回退 `0.010%–0.267%`，唯一正向只有 `0.095%`，远低于 1% gate。

Cache hint 可能失败的时间窗口有三种：

```text
太早：数据使用前被其他权重驱逐
刚好：覆盖真实 miss latency
太晚：consumer 已经 stall，hint 无法挽回
```

即使时机合适，prefetch 仍会消耗：

- 指令 issue；
- 地址计算和寄存器；
- L2 capacity/bandwidth；
- 可能污染当前 instruction 的工作集。

“命中代码路径”和“更高 L2 hit rate”都不能替代 matched full-graph latency。

---

## 9. 把四组结果放进同一个成本模型

![四类异步修改的收益与新增成本](/assets/blog-async-outcomes.svg)

*图 3：异步技巧是否获胜，取决于它隐藏的关键 stall 与新增 ownership、live-state、resource 和 dependency 成本的差。*

可以将候选的净收益写成：

$$
G_{\mathrm{async}}
=
L_{\mathrm{exposed}}
\times
C_{\mathrm{consumable}}
-
\left(
C_{\mathrm{sync}}
+C_{\mathrm{state}}
+C_{\mathrm{resource}}
+C_{\mathrm{dependency}}
\right)
$$

其中：

- $L_{\mathrm{exposed}}$：原来真正暴露在关键路径上的 latency；
- $C_{\mathrm{consumable}}$：consumer 能否利用 producer 提前量；
- $C_{\mathrm{sync}}$：新 barrier、commit、wait 和 publish；
- $C_{\mathrm{state}}$：额外 ring、slot、descriptor、live range；
- $C_{\mathrm{resource}}$：register、SMEM、L2、MIO/FMA 竞争；
- $C_{\mathrm{dependency}}$：融合后新增长依赖链。

四组实验分别对应：

| 案例 | 隐藏的 stall | 新增成本 | 结果 |
| --- | --- | --- | --- |
| Single named barrier | 打破 16-warp 同相等待与管线突发 | 小，ownership 独立 | 赢 |
| Batch phase fusion | barrier 很便宜 | QK 被迫等待 K+V | 输 |
| Hazy pending1 | 试图覆盖 TMA store | ring/lifetime/wait 增加，consumer lead 不够 | 输 |
| Hazy L2 prefetch | 试图覆盖 weight miss | hint 时机、指令和 cache 污染 | 未过门 |

---

## 10. 为什么 Occupancy 会让同一技巧变号？

Single decode 约 1 block/SM。一个 block 内 16 个 warp 同相，block 内 scheduling 结构就是主要并发来源。

Batch decode 约 8 blocks/SM。即使每个 block 内只有 4 warp，不同 block 已经能在 scheduler 上交错。此时：

- block barrier 的相对成本更低；
- named subgroup barrier 的空间更小；
- 额外 shared memory/register 可能减少 resident blocks；
- phase fusion 产生的长依赖更难用别的 warp 隐藏。

所以不能只问“这个 kernel 有没有 barrier”，还要问：

```text
一个 block 有多少 warp？
一个 SM 能驻留多少 block？
哪些 warp 共享数据？
stall 能否被其他 block 隐藏？
新增 stage 会不会跨过 residency cliff？
```

异步设计的最小单位不是指令，而是完整的 resident execution shape。

---

## 11. 正确的实验顺序：先证伪 Ownership，再花 GPU 时间

面对一个新的 barrier/pipeline/prefetch 候选，可以按下面顺序：

### 第一步：静态 ownership proof

- producer/consumer 是哪些 threads/warps/CTAs？
- shared page 是否跨 group 读取？
- 下一阶段何时复用？
- divergent path 是否让部分参与者缺席 barrier？

ownership 不成立，立即停止。

### 第二步：零修改 binary/control

- PTX→cubin 是否与生产 SASS 相同？
- register、spill、barrier、SMEM 是否一致？
- launcher、plan 和 merge 是否公平？

### 第三步：单机制 operator A/B

- 随机或平衡 AB/BA；
- no-profiler timing；
- exact output/state；
- 检查目标 SASS 指纹；
- 预声明 materiality gate。

### 第四步：完整 Graph / Model

- graph 与 graph 比；
- capture boundary 一致；
- 完整 token/state trajectory；
- full envelope，而非所有 kernel duration 求和。

### 第五步：Profiler 只负责解释

- NSYS 确认关键路径和 overlap；
- NCU 回答 precise mechanism；
- 不能用 barrier stall 下降覆盖 matched latency 回退。

---

## 12. 何时值得重新打开一个失败分支？

失败并不意味着技巧永远错误，但 reopening 必须有新的前提：

### Named barrier

只有目标 kernel 出现同样的独立 group ownership，且没有跨组 shared read。

### Phase fusion

只有 producer 数据能同时就绪，或融合不会推迟关键 consumer 的最早开始时刻。

### Pending store / deeper ring

只有 profile 证明 wait 暴露在关键路径，consumer 能利用 lead，且资源预算不跨 residency cliff。

### L2 prefetch

只有新 workload/architecture 出现明确 cache-miss/long-scoreboard 证据，且 reuse window 可以匹配 prefetch distance。

没有新的机制证据，只换 seed、重复运行或降低门槛，不是科学的 reopening。

---

## 结语：异步不是越多越好，而是越准确越好

这些实验共同推翻了一条常见但危险的直觉：

> “如果同步会等待，那么减少同步、增加并行和提前加载总会更快。”

更准确的说法是：

```text
ownership 决定能不能异步
dependency window 决定能提前多少
occupancy 决定谁来隐藏等待
live state 决定异步要付多少资源
critical path 决定局部 overlap 是否有价值
```

Single decode 的 named barrier 获胜，是因为四个 `tz` group 真正独立，并且 16 warp 同相是现实瓶颈。Batch decode 的 phase fusion 变慢，是因为 4-warp/高 block residency 已经能隐藏 barrier，融合反而推迟 QK。Hazy 的 pending store 和 L2 prefetch 失败，是因为新增 lifetime、wait、指令和 cache 成本没有覆盖到新的关键 stall。

所以，下一次想加 double buffer、TMA pipeline、prefetch 或 named barrier 时，不要先问“还能多 overlap 多少”。先问：

> **“哪一段 latency 现在真的暴露在关键路径上？consumer 能否利用提前量？为了这点提前量，我延长了哪些状态和依赖？”**

回答清楚这三个问题，异步才从一种美丽的结构，变成一种有证据的优化。
