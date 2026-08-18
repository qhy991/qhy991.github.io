---
layout: post
title: "为什么少写一半 Store，模型只快 0.02%？从 Symmetric Memory、Rank Ownership 到 Slow-Rank Timing"
date: 2026-08-17 09:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Multi-GPU, Symmetric Memory, TP2, CUDA, Communication, Rank Ownership]
reading_time: 27
cover_image: /assets/blog-symmetric-rank-ownership.png
excerpt: "TP2 symmetric pull 删除了无消费者的第二份 16-byte payload 发布，孤立边界快 7.68%，完整模型却只提高 0.019912%。本文解释通信优化为什么要先分析 owner/consumer，再分析 transport，以及 slow-rank wall、MNNVL reduction tree 与 stream contention 如何决定真实收益。"
---

> 本文基于 SubCUDA `d1db18f` 的 D064、producer-symmetric-output、R98 与 D058 machine-readable cases。D064 是 default-off experimental E2E win，另外三条是 correctness/operator stop。所有结果都绑定 Qwen3.5 TP2、B200 和冻结 graph，不外推到其他 rank 数、transport 或 serving 模式。

多 GPU 优化里，一个非常诱人的想法是：

> “如果数据最终要被另一个 rank 读取，为什么不让 producer 直接写 symmetric memory？”

这听起来能删除 copy，也能减少中间 tensor。但真实实验给出三种完全不同的结果：

1. 只删除没有消费者的重复发布：孤立边界快 `7.68%`，TP2 小幅正向；
2. 让 GEMM 直接写 symmetric output：P2P 慢 `71.1%`，MNNVL 慢 `103.5%` 且数值不同；
3. 把独立分支放到第三 stream：所有 bytes 正确，joined boundary 仍慢 `1.438 μs`。

区别不在“用了 symmetric memory 还是普通 memory”，而在三个更上游的问题：

- **Ownership**：每份数据最终由谁消费？
- **Transport semantics**：特殊地址、可见性与 reduction 采用什么协议？
- **Critical path**：完整 step 等的是哪个 rank、哪个 join？

---

## 1. 先理解 Symmetric Memory：地址对称，不等于成本为零

在多 GPU runtime 中，symmetric buffer 通常意味着每个 rank 都有对应大小和布局的存储，peer/rank 可以通过稳定规则定位远端 slot。

概念上：

```text
rank 0: buffer[0]
rank 1: buffer[1]

source rank s 的 payload
  → 写入与 s 对应的 symmetric slot
  → consumer 按 source index 读取
```

它的价值是稳定 peer pointer、统一布局和通信协议。它不意味着：

- 写入和普通 HBM store 一样便宜；
- 所有 rank 都必须写所有 buffer；
- NVLS/MNNVL reduction 与普通 BF16 加法顺序相同；
- producer 直接写特殊地址一定比后续 publish 更快。

Symmetric 是**寻址与所有权协议**，不是“免费共享内存”。

---

## 2. Incumbent 的问题：同一份 Payload 发布了两次

D064 的 TP2 finalize 使用两个 symmetric data buffers。Incumbent 对 source rank `s` 的同一 16-byte payload 做 push-all：

```text
rank 0 input
  → buffer0 slot0   [被消费]
  → buffer1 slot0   [没有 consumer]

rank 1 input
  → buffer0 slot1   [没有 consumer]
  → buffer1 slot1   [被消费]
```

consumer 最终按 source index pull：

```text
read buffer0 slot0
+ read buffer1 slot1
```

另外两份 copy 没有消费者。

![TP2 Push-All 与 Source-Owned Symmetric Pull](/assets/blog-symmetric-pull-dataflow.svg)

*图 1：D064 没有发明新 collective，只删除了消费图中不存在的两条 store edge。*

候选只让：

```text
source rank s → buffer[s]
```

negative-zero readiness sentinel、volatile 16-byte access、BF16 rank order、residual/RMSNorm、PDL 和 graph dependency 全部保持不变。

这是一条非常通用的优化优先级：

$$
\text{先删无 consumer 的 bytes}
\quad > \quad
\text{再换更快的传输机制}
$$

---

## 3. 机器码真的少了什么？

D064 是 CUDA source change，不是 Direct PTX。编译后：

| SASS / Resource | Control | Candidate |
| --- | ---: | ---: |
| `STG.E.128.STRONG.SYS` | 4 | 2 |
| Static instructions | 1768 | 1720 |
| NCU dynamic global stores | 2307 | 1923 |
| Registers / thread | 72 | 72 |
| Shared memory | 2824 B | 2824 B |
| Spill | 0 | 0 |

资源形态不变，store 数下降，因果链相对干净：

$$
\text{remove redundant publication}
\Rightarrow
\text{fewer system-scope stores}
\Rightarrow
\text{shorter isolated finalize boundary}
$$

正确性还覆盖：

- random / finite-edge；
- 5,000,000-cycle delayed rank；
- byte-exact payload/final output；
- rank progress；
- 5 对完整 TP2 token hash。

---

## 4. 为什么孤立边界快 7.68%，模型只快 0.0199%？

孤立 slow-rank boundary：

| Arm | Median |
| --- | ---: |
| Control | 14.333036 μs |
| Symmetric pull | 13.232616 μs |
| Reduction | **7.6775%**，30/30 |

完整 TP2：

| Arm | Throughput | 128-step wall |
| --- | ---: | ---: |
| Control | 3531.509712 tok/s | 1159.843901 ms |
| Candidate | 3532.212905 tok/s | 1159.612999 ms |
| Change | **+0.019912%** | **−0.230902 ms** |

五对都获胜，方向一致，但幅度很小。

为什么不能简单做：

$$
1.10042\ \mu s
\times
10{,}752\ \mathrm{nodes}
$$

因为 isolated boundary 与完整 graph 中的 input/attention/MoE variant 不完全同分布；很多 boundary 与其他工作 overlap；complete step 还包含未修改的 GDN、attention、MoE、collective 和控制节点。

归档 profile 中 direct-input attention variant 甚至慢 `0.409%`，MoE variant 只快 `0.692%`。孤立的 7.68% 不是每个 graph node 都能获得的 saving。

---

## 5. 多 GPU 时间为什么要看 Slow Rank？

TP2 的一个共同阶段通常要等两个 rank 都完成：

$$
T_{\mathrm{stage}}
\approx
\max(T_{r0},T_{r1})
$$

如果 rank0 快 2 μs，但 rank1 不变，join 可能一微秒都不提前。反过来，平均值只改善 0.5 μs，但最慢 rank 改善 1 μs，完整 step 才可能前移。

![平均 Rank 时间与 Slow-Rank Critical Path 的区别](/assets/blog-symmetric-slow-rank.svg)

*图 2：通信和 collective 的权威 timing 应覆盖所有 rank，并以完整 envelope 或 rank-max 为主，而不是单 rank 平均。*

D064 operator 使用 slow-rank wall；完整 E2E 在同模型加载中比较完整 graph wall 和吞吐，并检查两个 rank token hash。

---

## 6. 为什么这个正结果仍然不适合维护第二套默认 DSO？

`+0.019912%` 是可重放、五对同方向的实测正信号，但收益只有约 0.23 ms / 128-step graph。

一套独立默认 DSO 会增加：

- build/release matrix；
- FlashInfer revision 绑定；
- selector 与 fallback；
- binary/SASS/hash 维护；
- TP2-only specialization；
- 与其他通信候选的组合验证。

当前 source worktree 还混入 producer-symmetric、MNNVL 等其他实验，不能整文件复制；必须从冻结 FlashInfer revision 重建最小 patch。

因此正确裁决是：

> **小而真实的 experimental E2E win，保留机制与证据，但不足以单独成为默认发布分支。**

性能真实不等于工程 ROI 足够。

---

## 7. 反例一：为什么让 GEMM 直接写 Symmetric Output 会慢 71%？

另一个候选试图把 shared-down GEMM output 直接写到 symmetric storage，删除普通 output→publish copy：

```text
control:
  GEMM → ordinary BF16 output → publish → finalize

candidate:
  GEMM → direct symmetric output → finalize
```

结果：

| Transport | Control | Candidate | Correctness |
| --- | ---: | ---: | --- |
| P2P | 20.979520 μs | 35.897280 μs | byte-exact |
| MNNVL/NVLS | 21.103680 μs | 42.954560 μs | **FAIL** |

P2P 慢 `71.106%`，MNNVL 慢 `103.541%`。

删除逻辑 copy 不等于删除物理成本。Symmetric/NVLS 地址具有不同的存储、映射和可见性语义；让高吞吐 GEMM epilogue 直接承担特殊 store，可能破坏它原本的合并写和 pipeline。

这体现一个重要边界原则：

> **Producer 直接写 consumer layout 是候选方向，不是自动定理；consumer layout 若带昂贵 transport semantics，独立 publish 反而可能是更好的隔离层。**

---

## 8. MNNVL 为什么连数值也变了？

MNNVL/NVLS 路径中，GEMM output 本身 exact，rank progress 也通过，但 residual 有 `1064/98304` 个元素不同，后续 RMSNorm/output 有 1068 个不同。

原因是 `multimem.ld_reduce ... add.acc::f32.v4.bf16x2` 使用 FP32 accumulation，而 incumbent 使用既有 BF16 pairwise rounding tree。

数学上更精确的 FP32 accumulate 仍然不是同一个模型执行合同：

$$
\mathrm{BF16}(\mathrm{BF16}(a+b)+c)
\neq
\mathrm{BF16}(a+b+c)
$$

所以 transport 也是 numerical axis。P2P exact 不能替 MNNVL 宣告 exact；每种 collective/reduction mechanism 都要单独冻结数值协议。

---

## 9. 反例二：第三 Stream 逻辑独立，为什么仍然更慢？

R98 把 shared-gate projection 放到第三 stream：

```text
routed stream ─────────────┐
shared chain ──────────────┼→ finalize
new shared-gate stream ────┘
```

依赖图允许并行，所有 output/payload/scale 也 byte-exact。但当前 runtime 已有 `140 routed / 8 shared-overlap SM` 资源划分。第三分支没有删除算术或 materialization，只增加：

- 新 stream/event；
- SM competition；
- cache/DRAM contention；
- scheduler work。

代表性 joined boundary 回退 `1.438222 μs`，投影 graph 回退 `8.836438 ms`；九个 profile 的 wins 只有 6/60–20/60，因此在 operator gate 停止。

逻辑并行只说明“可以同时提交”，不说明“GPU 有第三份物理容量”。

---

## 10. 反例三：All-Reduce Geometry 局部快 13.25%，为什么没有 E2E 数字？

D058 将 TP2 finalize geometry 从 `96×4` 改为 `192×2`。双 rank synthetic operator：

```text
14.406328 → 12.497464 μs
−13.2502%
8/8 wins
random / finite-edge byte-exact
```

但 full-model qualification 的两个 rank token hash 都从 baseline 变成另一个值。Harness 在任何 wall timing 前中止，因此**没有 E2E throughput 数字**。

原因是 192×2 进入 generic two-CTA reduction path，改变 FP accumulation tree。Synthetic inputs 没有覆盖真实模型触发的数值差异。

这个案例说明：

- collective/finalize geometry 同时是性能轴和数值轴；
- operator exactness 不是完整模型 oracle；
- correctness-rejected arm 不应该继续收 NCU/NSYS 或 E2E 性能。

![通信所有权、Transport、Stream 与 Reduction Geometry 的不同裁决](/assets/blog-symmetric-boundary-evidence.svg)

*图 3：删除重复 publication、改变特殊存储、增加 stream 和改变 reduction tree 是四类不同候选，必须分别通过 ownership、数值与 E2E gate。*

---

## 11. 数据所有权分析应该先画什么？

对每个通信 edge，记录：

```text
producer rank
payload shape / dtype / bytes
logical owner / physical buffer
所有消费者
transport / reduction semantics
stream / event / release-acquire
slot lifetime / retirement
slow-rank join
final consumer format
```

然后依次问：

1. 有没有 payload 没有消费者？
2. 能否让 consumer pull source-owned data，删除 duplicate push？
3. 能否缩窄、压缩或过滤 bytes？
4. producer 直接写 consumer layout 会不会承担更贵的 address/visibility semantics？
5. transport 是否改变 reduction order？
6. 新 stream 是否有真实空闲 SM/L2/HBM？
7. 完整 step 等的是 average rank 还是 max rank？

---

## 12. 为什么“少写一半”不等于“快一倍”？

通信边界时间可以粗略分成：

$$
T_{\mathrm{boundary}}
=
T_{\mathrm{compute}}
+T_{\mathrm{store}}
+T_{\mathrm{visibility}}
+T_{\mathrm{wait/join}}
+T_{\mathrm{fixed}}
$$

D064 只删除一部分 `store`。即使 store 指令数量减半，其他项仍然存在。

完整模型还要乘上关键路径占比：

$$
\Delta T_{\mathrm{model}}
\le
\Delta T_{\mathrm{boundary}}
\times
\mathrm{critical\ exposure}
$$

如果 boundary 与其他工作 overlap，critical exposure 远小于调用次数比例。

---

## 13. 怎样验证一个新的 Symmetric-Memory 候选？

### Ownership Gate

- 每份 write 是否有 consumer？
- consumer 按 source、destination 还是 local buffer 索引？
- rank 数变化时策略是否仍成立？

### Correctness Gate

- payload bytes；
- residual/norm/output；
- delayed rank；
- readiness sentinel；
- release/acquire；
- P2P 与 MNNVL 分开；
- full token/rank hash。

### Performance Gate

- slow-rank no-profiler boundary；
- matched graph wall；
- 五对或预声明 paired rule；
- 不把 isolated saving × node count 当成 E2E；
- 检查所有 rank 和 transport。

### Binary/Profile Gate

- SASS store/load 数；
- system-scope instruction；
- registers/SMEM/spill；
- NSYS join/overlap；
- NCU 只做 precise mechanism attribution。

### Engineering Gate

- 是否值得独立 DSO/selector；
- 是否能以最小 source patch表达；
- rollback 是否完整；
- unsupported NRanks 是否 fail closed。

---

## 14. 三层通信优化的优先顺序

### 第一层：删除无消费者的数据

D064 属于这一层。它不改变 transport，只删除重复 publication。

### 第二层：缩窄 Consumer Representation

在语义允许时，先 quantize/pack/filter，再通信；或者 producer 直接写一个便宜的 consumer layout。

### 第三层：改变 Transport

P2P、NCCL、NVLS/MNNVL、symmetric memory 各有不同 alignment、visibility、reduction 和 progress contract。应在前两层完成后再比较机制。

不能直接从“普通 buffer 多一次 copy”跳到“GEMM 直写 MNNVL 一定更快”。

---

## 15. 当前证据边界

### D064 可以说

- TP2 source-owned symmetric pull 删除重复 store；
- isolated boundary 快 7.6775%；
- TP2 五对吞吐正向 0.019912%；
- token hash exact；
- default OFF；
- 当前收益不足以单独维护第二套默认 DSO。

### 不可以说

- 其他 NRanks 同样适用；
- 所有 symmetric store 减半都会加速；
- 7.68% 可以乘 10,752 预测模型；
- MNNVL 与 P2P 数值协议相同；
- 多一条 stream 一定提高 overlap。

### 复现债务

历史 mixed worktree 已损坏/混入多项实验；fresh reproduction 必须从冻结 FlashInfer revision 重建一个只含 pull-butterfly 的最小 patch，并重新生成 DSO、SASS、operator 与 E2E authority。

---

## 结语：通信优化首先是所有权问题

多 GPU 性能分析很容易从 NCCL、NVLS、P2P 或 stream 数量开始。但 D064 的最佳修改更朴素：

```text
先画 consumer
    ↓
发现两份 store 根本没人读
    ↓
只让 source rank 发布自己的 payload
    ↓
保持 visibility、rank order 和 numerical tree
```

它得到一个小而真实的系统收益。直接写 symmetric output 和第三 stream 的反例则说明：删除逻辑节点不一定删除物理成本，允许并发也不等于有可用资源，特殊 transport 还可能改变数值顺序。

所以，下一次看到通信路径中的 copy 或 store，不要先问：

> “能不能用更先进的 transport？”

先问：

> **“这份数据的唯一 owner 是谁？到底有哪些 consumer？哪一次 write 没有人读？完整阶段最终在等哪个 rank？”**

当所有权图画对以后，许多最好的通信优化不是把 bytes 搬得更快，而是让不该存在的 bytes 从图上消失。
