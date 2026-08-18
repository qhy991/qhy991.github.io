---
layout: post
title: "GPU 内的 Ready Queue 为什么可能死锁？从 Release/Acquire、Split-KV 到 Forward Progress"
date: 2026-08-17 08:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [GPU Runtime, Release Acquire, Persistent Kernel, Split-KV, Forward Progress, CUDA Graph]
reading_time: 31
cover_image: /assets/blog-gpu-ready-queue.png
excerpt: "把调度器搬到 GPU，不只是实现一个原子队列。本文从 Hazy release/acquire 修复、长上下文 Split-KV/dynamic tail 与 SubCUDA 的 poll/PDL/third-stream 负结果出发，解释 memory visibility、epoch、slot retirement、resource reservation 和 forward progress 为什么必须一起设计。"
---

> 本文严格区分三类证据：Hazy release/acquire 与 Split-KV 是 C2/S3 archive lineage，存在 model/Graph 信号但仍有 canonical freeze debt；SubCUDA 的 D061/R98/D100 等 case 可结构化 replay；GPU-resident ready queue 与 partial DownProj accumulation 仍是 C0/S0 research plan，没有实现、正确性或性能结果。文中不会把 plan 写成 winner。

把调度器放到 GPU 上，听起来像一个很自然的优化：

```text
producer 完成一块工作
    ↓
把 task 放进 ready queue
    ↓
consumer 从队列取 task
    ↓
继续计算
```

这样可以减少 CPU 轮询、host launch 和静态 CUDA Graph 顺序带来的等待。对于长上下文 attention、paged weight pipeline、MoE dispatch 或 MegaKernel executor，它似乎是最终答案。

但一个 GPU-ready queue 要同时回答四个问题：

1. **Visibility**：consumer 看到 `ready` 时，真的能看到 producer 之前写的 payload 吗？
2. **Exactly once**：一个 task 会不会丢失、重复消费或跨 epoch 误认？
3. **Lifetime**：slot/page 什么时候可以安全复用？
4. **Forward progress**：等待 task 的 consumer 会不会占满所有 SM，让 producer 永远无法运行？

如果只实现 atomic head/tail，而没有回答这四个问题，队列可能更快地把系统带进错误或死锁。

---

## 1. 最危险的误解：看到 `ready == 1` 不等于数据已经可见

假设 producer 写一块 global-memory payload，然后设置 ready flag：

```cpp
payload[i] = result;
ready[i] = 1;
```

consumer 轮询：

```cpp
while (ready[i] == 0) {}
use(payload[i]);
```

从源代码顺序看，payload 在 ready 前写入。但 GPU/编译器可能重排、缓存或延迟可见性。若 ready store 没有 **release** 语义，consumer 的 ready load 没有匹配 **acquire**，`ready == 1` 并不自动建立 payload 的 happens-before。

正确协议至少是：

```text
Producer:
  write payload
  release publish (epoch, ready)

Consumer:
  acquire observe (epoch, ready)
  read payload
  release retire slot/page
```

![Producer release、consumer acquire 与 slot retirement](/assets/blog-gpu-ready-release-acquire.svg)

*图 1：ready flag 同时承担控制和可见性边界；epoch 防止旧 ready 被新一轮误认，retirement 保护 buffer reuse。*

Release 保证它之前的写入在匹配 acquire 后可见；acquire 保证之后的 payload load 不越过 publication。

这不是“加一个 fence 就一定慢”，而是 correctness contract。

---

## 2. Hazy 的教训：旧的 1.493× 为什么必须撤回？

早期 Hazy resident Graph 的 relaxed overlap 在短 replay 中看起来很快，约 `617.378 μs / 1.493×`。但长 replay 出现漂移。

根因不是数学公式错，而是 publication protocol 不完整：producer 写 payload 后更新 ready，consumer 轮询 ready，却没有形成 producer release → consumer acquire 的完整链。

因此这个性能数字必须按零处理并撤回。更快的错误并不是优化。

release-safe E40 修复后：

- direct / Graph 100/100 bitwise；
- resident Graph 约 `688.838–688.876 μs`；
- 相对其 individual-op Graph 保守约 `1.343×`；
- 只覆盖 Llama-3.2-1B、B1、position 0、1×B200 的单步 device Graph；
- 不是 HTTP serving；
- 当前 Hub 状态为 C2/S3 migration-pending，而不是公开独立复现完成。

这组历史最重要的顺序是：

```text
relaxed overlap 看起来快
        ↓
发现 memory-order 漂移
        ↓
撤回旧数字
        ↓
建立 release/acquire
        ↓
重新做 bitwise + Graph timing
```

不能在旧结果后补一句“后来修了 fence”，继续保留旧加速比。

---

## 3. Cache Hint 不是 Acquire：D061 为什么快但错误？

SubCUDA D061 比较多种跨 GPU polling cache policy。弱 no-allocate/cache-hint arms 看起来快 `11.8%–12.8%`，但会读到 stale data，byte correctness 失败。

只有 `acquire-system` arm 保持跨 GPU 可见性和 delayed-rank progress；但它的性能只改善 `0.0151%`，30 个样本只赢 16 个，低于 23/30 gate。

这说明两件事：

1. 更弱的 cache policy 可以“更快”，因为它少做了正确性所需的 ordering；
2. 在正确 acquire 语义下，poll policy 本身几乎没有可用 headroom。

所以 poll loop 的优化顺序应当是：

```text
先固定一致性和 delayed-rank correctness
        ↓
再测 backoff / cache hint / load width
```

不能把 stale arm 的速度拿来证明“acquire 太贵”。

---

## 4. Memory Order 正确，为什么仍可能死锁？

Release/acquire 只解决可见性，不保证 producer 一定能获得执行资源。

考虑一个 persistent consumer grid：

```text
consumer CTA 占满所有 resident slots
        ↓
每个 CTA 都在 poll ready queue
        ↓
producer kernel 等待空闲 SM slot
        ↓
producer 无法写 payload / publish ready
        ↓
consumer 永远等不到 ready
```

![Persistent waiter 占满资源导致 producer 无法前进](/assets/blog-gpu-ready-forward-progress.svg)

*图 2：这是资源死锁，不是 memory-order 错误；即使 acquire/release 完美也不会自动解除。*

可能的解决方式包括：

- 限制 persistent consumer CTA 数，保留 producer capacity；
- 把 producer 完成作为 launch 前 stream/event dependency；
- 将等待和计算拆成不同 phase；
- 使用 hardware-supported scheduler residency control；
- 让 poller 主动退让或退出，而不是无界占用；
- 对 queue 空状态使用 bounded wait + poison/abort。

减少 CTA 可能恢复 forward progress，也可能降低并行度，所以仍需完整 E2E 测量。

---

## 5. 为什么长上下文先需要 Split-KV，而不是 Ready Queue？

Llama-3.1-8B long-context profile 中，attention 约占 49.2%，DownProj 约占 15.2%。首要问题不是 task 不会排队，而是 attention 沿 KV 长度的并行分解不足。

单 partition attention 要让有限 CTA 顺序处理长 KV。KV 从 4K 增至 8K，工作增长，但 CTA 数没有相应增加。

Split-KV 将 KV 分成多个 partition。每个 partition 独立产生：

$$
m_p=\max_j s_{p,j}
$$

$$
\ell_p=\sum_j e^{s_{p,j}-m_p}
$$

$$
o_p=\sum_j e^{s_{p,j}-m_p}v_{p,j}
$$

最终稳定合并：

$$
m=\max_p m_p
$$

$$
\ell=\sum_p \ell_p e^{m_p-m}
$$

$$
o=\frac{\sum_p o_p e^{m_p-m}}{\ell}
$$

![Split-KV 恢复长序列并行度并显式归并 softmax 统计量](/assets/blog-gpu-ready-split-kv.svg)

*图 3：每个 partial 必须携带 local max、normalizer 和 weighted output；只保存 partial output 无法稳定合并 softmax。*

本地验证信号：

| Position | One partition | Split-KV | + Dynamic tail |
| --- | ---: | ---: | ---: |
| 4K | 10.063 ms | 3.589 ms | 3.535 ms |
| 8K | 17.288 ms | 4.050 ms | 4.011 ms |

相对 one-partition 总信号约 `2.85× / 4.31×`。Top-1 16/16，但 owning freeze pending，不能写成 deployment promotion。

Split-KV 的大收益来自**改变并行分解**，不是 queue 技巧。Ready queue 应当建立在正确 work decomposition 之上，而不是用更复杂调度掩盖一个并行度不足的算法。

---

## 6. Dynamic Tail 删除了什么？

固定 partition/tile 会让最后一块按完整尺寸执行，即使真实 KV tail 只有一部分有效。

Dynamic tail 根据真实有效长度缩短最后 partition，并 mask 越界 lane。

独立 NCU 信号约：

```text
3.610560 → 3.555936 ms
−1.51%
DRAM bytes 只变化约 −0.0115%
```

这说明收益不来自少读 HBM，而主要来自少做：

- 无效 HMMA；
- shared-memory staging；
- local/register work；
- tail bookkeeping 的固定计算。

这也是 ready queue 设计的重要启发：work item 不应只说“page 7 ready”，还应携带真实有效范围。否则队列只是更早地调度同样的 padding work。

---

## 7. Ready Queue 真正要替代什么？

当前 `runtime-ready-queue` 在 agentic Hub 中是 C0/S0 PLAN_ONLY。它的目标不是“让 GPU 有一个队列”，而是替代 centralized scan/join：

```text
旧：consumer 反复扫描全部 ready flags
    → 找到可以工作的 page/tile

新：producer 将 ready item 直接交付给 consumer-visible queue
    → consumer 领取真实 ready work
```

如果当前 scan/join 不在关键路径，queue 没有价值。如果 ready item 最终仍要在 centralized barrier 等齐，提前入队也没有价值。

最小 queue slot 至少需要：

```text
epoch / ticket
state
work type
payload or descriptor pointer
valid range / tail
producer identity
consumer ownership
error / poison state
```

一个可能的状态机：

```text
EMPTY(epoch)
  → WRITING
  → READY [release]
  → CLAIMED [acquire]
  → DONE [release]
  → EMPTY(next epoch)
```

这只是设计草图，不是当前实现事实。

---

## 8. Exactly-Once 为什么比 Atomic Head/Tail 更难？

即使用 atomic increment 分配 slot，仍可能出现：

### 重复消费

两个 consumer 看到同一个 ready item，缺少 CAS/claim protocol。

### 丢失 item

producer 更新 tail，但 payload publication 或 slot state 顺序不完整。

### ABA

slot 0 在 epoch 7 被消费并复用到 epoch 8；延迟 consumer 仍持有旧索引，误把新 payload 当旧 task。

### Epoch wrap

有限宽度 counter 回绕后，旧 state 与新 state 无法区分。

### Poison / error

producer 失败、rank 退出或 payload 非法时，consumer 不能无限 poll。

### Slot retirement

最后一个 consumer 完成前，producer 不能覆盖 page/descriptor。

所以 queue correctness 应覆盖：

- 多 epoch；
- slot churn；
- 延迟 producer；
- 延迟 consumer；
- 重复/丢失检测；
- counter wrap 附近；
- poison/abort；
- CUDA Graph reset/replay；
- 所有 mutated state。

---

## 9. PDL 已经能“提前 launch”，为什么还需要队列？

PDL 将依赖拆成：consumer 可以提前进入 GPU，以及真正读取 producer output 前再同步。

但它只在 wait 前有独立工作时有价值。

### R97：没有独立前缀，收益为零

gate/up → SwiGLU+quant consumer 的第一项有效工作就是读 gate/up。consumer 提前 launch 后立刻 wait：

```text
median delta = 0.000 μs
24/60 wins
```

### D100：真实 0.31 μs local overlap，E2E 仍回退

late PREEXIT + PDL consumer 确实创造约 0.31 μs local overlap，但 TP2 exact prescreen：

```text
3546.111990 → 3542.650883 tok/s
−0.0976%
0/2 pairs
```

局部机制真实，不等于 joined graph 更快。

Ready queue 也一样：它可能让 task 更早 eligible，但不会保证资源竞争和最终 join 更短。

---

## 10. 逻辑独立的 Third Stream 为什么也会变慢？

R98 将 logically independent shared-gate 放到第三 stream，所有 bytes 保持 exact。但现有执行已经使用 140/8 SM allocation；新 stream 与其他 kernel 争用资源，joined boundary 反而增加 `1.438222 μs`。

投影到完整 graph 是负 `8.836 ms`，因此未进入 TP2 E2E。

独立依赖只证明“允许并发”，不证明“物理并发有利”。Queue/stream scheduler 还要仲裁：

- SM residency；
- registers / shared memory；
- L2/HBM；
- collective progress；
- producer slot reservation；
- 最慢 rank。

优化目标是最短 complete envelope，不是最大 overlap percentage。

---

## 11. 为什么 Profiler 中的巨大 Poll Wait 可能是幻觉？

D060 曾观察到 `63.752 ms` 和 `426.099 ms` 的巨大 wait，看起来很适合实现 adaptive backoff。

审计发现两个极值都出现在 target index 0，是 first-target profiler perturbation，而不是稳定运行时现象。因此没有实现 backoff candidate。

这是一种高价值的 STOP：

```text
profiler 中发现异常
    ↓
先检查位置、重复性和无 profiler 对照
    ↓
确定是测量扰动
    ↓
不写代码
```

如果直接根据 profiler 最大值设计 queue/backoff，优化的可能只是 profiler 自己制造的现象。

---

## 12. Ready Queue 的 Forward-Progress 预算怎么写？

一个可证伪合同至少需要：

### Residency budget

```text
consumer_poller_CTAs
+ producer_CTAs
+ other_persistent_roles
≤ guaranteed resident capacity
```

不能用平均 occupancy；要按最坏 register/SMEM/cluster 形态计算。

### Bounded polling

- 指数 backoff 或 yield 是否真的释放 issue 资源？
- queue 空多久触发 poison/abort？
- Graph replay 如何重置？

### Fairness

- 长 task 是否饿死短 task？
- rank/task priority 是否放大 tail？
- queue head cache line 是否成为热点？

### Memory order

- payload → release ready；
- acquire claim → payload read；
- last-consumer → release retire；
- next producer → acquire empty/new epoch。

### Error semantics

- producer failure；
- stale epoch；
- duplicate ticket；
- queue overflow；
- unsupported work item。

任何一个缺失，性能 benchmark 都太早。

---

## 13. Partial DownProj Accumulation 为什么仍是 Plan？

另一个 C0/S0 方向是：权重/page 到达一部分，就立即开始 DownProj partial accumulation，而不是等待全部 ready。

它可能改变：

```text
arrival → first useful compute → final join
```

但当前尚未冻结：

- 沿 K、N、page 还是其他轴拆 partial；
- accumulation dtype；
- reduction order；
- partial buffer layout/bytes；
- final consumer ownership；
- 数值容差；
- producer/consumer residency。

因此不能把它自动等同于已经停止的 MatVec-K chunking，也不能从 Split-KV 的成功推导它一定有效。

重开前应先记录 DownProj edge：producer、payload、consumer、layout、bytes、ready 语义和第一真实依赖，再预注册 partial-byte 上限和数值顺序。

---

## 14. 最小可证伪实验应该长什么样？

### Phase A：证明现有 Scan/Join 是瓶颈

- device timestamp 记录每个 item ready time；
- consumer scan/join start/end；
- queue-depth/ready distribution；
- no-profiler whole-model baseline；
- 排除 first-target profiler artifact。

### Phase B：固定容量 Queue Correctness

- exactly-once；
- multi-epoch；
- delayed roles；
- slot reuse；
- poison/overflow；
- release/acquire；
- Graph replay reset。

### Phase C：Forward-Progress Stress

- reserve producer capacity；
- 逐步增加 poller CTA；
- 检查 hang、tail 与 producer start delay；
- slow-rank/slow-producer 注入。

### Phase D：Matched Performance

- native/candidate 一键切换；
- same workload/weights/graph；
- alternating raw samples；
- complete model/Graph latency；
- NSYS 证明 centralized join 关键段缩短；
- NCU 只回答 queue/poll 的精确资源问题。

Stop rule：如果 scan/join 暴露时间低于 queue bookkeeping 的乐观上限，或 candidate 两次出现 forward-progress failure，立即关闭。

---

## 15. 当前可以安全说什么？

### 已有证据

- relaxed ready publication 没有 release/acquire，旧短 replay 性能已撤回；
- release-safe Hazy Graph 通过 100/100 bitwise，并有 bounded S3 latency 信号；
- Split-KV/dynamic tail 在 4K/8K model-forward 有强本地信号，但 freeze pending；
- D061 证明弱 poll cache hint 会 stale，exact acquire 几乎没有性能 headroom；
- R97、D100 证明 PDL local overlap 不保证 E2E；
- R98 证明逻辑独立的第三 stream 也会因资源竞争变慢；
- D060 证明 profiler 极值可能是 first-target 扰动。

### 尚未实现

- GPU-resident runtime ready queue；
- partial DownProj accumulation。

它们没有 candidate、correctness、timing 或 revision，不能出现性能百分比。

---

## 结语：GPU 调度器的正确性有两条轴

GPU-resident control 最容易只关注队列算法：head、tail、CAS、work stealing。但真正的系统合同有两条正交轴：

```text
Memory correctness:
  release / acquire / epoch / lifetime / exactly-once

Execution progress:
  residency / producer capacity / polling / fairness / tail
```

Release/acquire 可以让 consumer 看到正确数据，却不能保证 producer 能运行。保留 producer slot 可以避免死锁，却不能防止 stale epoch 或 buffer 过早复用。

Split-KV 还提醒我们：很多时候，最大的收益来自重新分解工作，而不是更聪明地排同一批 work item。Dynamic tail 则说明队列应该携带真实有效范围，而不是更早调度 padding。

所以，在实现 ready queue 以前，最重要的问题不是：

> “用 ring buffer 还是 priority queue？”

而是：

> **“现有 centralized wait 真的在关键路径上吗？每个 ready item 的数据何时可见、被谁消费、何时退休？等待者会不会占掉生产者前进所需的最后一个资源槽？”**

只有同时回答 visibility、lifetime 与 forward progress，GPU-ready queue 才是一种运行时优化，而不是一个更复杂的死锁发生器。
