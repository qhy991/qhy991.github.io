---
layout: post
title: "为什么 TMA 比 cp.async 快，却仍输给普通 LDG？从一次性 State、SMEM Hop 到 Occupancy"
date: 2026-08-17 15:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [TMA, cp.async, Shared Memory, Occupancy, GDN, CUDA, B200, Memory Pipeline]
reading_time: 30
cover_image: /assets/blog-tma-cpasync-one-shot.png
excerpt: "同一块 32 KiB recurrent state，用 cp.async 搬进 shared memory 慢 44.5%，换成一次 2D TMA 后回收了约 2.07 μs，却仍比 GMEM→register 直接 load 慢 22.2%。本文从 reuse、额外数据跳、mbarrier、live range 与 CTA residency 解释为什么搬运原语的先进程度不等于端到端路径更短。"
---

> 本文基于 [`qhy991/SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab) 的 D005、D006、D001 与 D017 machine-readable cases。四条 replay 与资产检查已在当前 checkout 通过。D005/D006 的 fresh build/operator 仍缺冻结的 detached FlashInfer/CuTe candidate patch 与参数化 JIT 环境；本文引用归档 operator evidence，不把离线 JSON replay 描述成 fresh B200 run。

在 Hopper/Blackwell CUDA 优化中，下面三个词很容易形成“技术等级”错觉：

```text
普通 global load
      ↓
cp.async
      ↓
TMA
```

仿佛越往下越先进，也一定越快。

一组 B200 GDN state 实验却给出：

| State carrier | Median | 相对同组 direct source |
| --- | ---: | ---: |
| Direct `LDG.E.128` | 约 9.26 μs | baseline |
| `cp.async → 32 KiB SMEM` | 13.38556 μs | **+44.509% 更慢** |
| One 2D TMA → 32 KiB SMEM | 11.31704 μs | **+22.243% 更慢** |

TMA 的确比 `cp.async` 快：它回收了约 `2.06852 μs`。

但它仍比最简单的 direct load 慢约 `2.05920 μs`。

这不是“TMA 没有生效”。最终 SASS 中真实出现了：

```text
UTMALDG.4D
```

真正的问题是：**TMA 只消除了“由谁发 copy”的成本，没有消除一次性数据被迫经过 shared memory 的第二跳。**

---

## 1. 冻结 Workload：这块 State 有多大？

实验 operator cell：

| 维度 | 值 |
| --- | ---: |
| GPU | NVIDIA B200 / SM100a |
| Batch / token | B32 / T1 |
| Q/K heads | 8 |
| Value heads | 32 |
| K / V dimension | 128 / 128 |
| State dtype | BF16 |
| Tile | `128 × 128 × 2 B = 32 KiB` |
| Launch | 1024 CTAs × 128 threads |
| Consumption | 每 CTA 内读取一次，然后 recurrence 更新并写回 |

最重要的事实不是 32 KiB，而是：

> **这块 state 在 CTA 内只有一个计算消费者，没有跨 phase 或多次循环 reuse。**

Shared memory staging 通常依靠 reuse 摊销：从 GMEM 搬一次，之后被多个 warps、多个 MMA、多个迭代或多个输出 tile 重复读取。

这里没有这种摊销。

---

## 2. 三条数据路径，到底差在哪里？

### Direct baseline

```text
GMEM state
   ↓ LDG.E.128
register
   ↓
recurrence
```

### D005：cp.async staging

```text
GMEM state
   ↓ 许多线程协作发 cp.async
32 KiB SMEM
   ↓ wait_group + CTA barrier
register
   ↓
recurrence
```

### D006：TMA staging

```text
GMEM state
   ↓ one 2D TMA transfer
32 KiB SMEM + mbarrier
   ↓ wait + LDS
register
   ↓
recurrence
```

![Direct、cp.async 与 TMA 的 State 数据路径](/assets/blog-state-carrier-paths.svg)

*图 1：TMA 缩短了 copy-issue 控制路径，但 cp.async/TMA 都保留 GMEM→SMEM→register 两跳；direct load 只有一跳。*

可以把三者的成本写成：

$$
T_{\mathrm{direct}}
=T_{G\rightarrow R}
$$

$$
T_{\mathrm{cp.async}}
=T_{\mathrm{issue,many}}
+T_{G\rightarrow S}
+T_{\mathrm{wait/barrier}}
+T_{S\rightarrow R}
+T_{\mathrm{residency\ loss}}
-T_{\mathrm{hidden}}
$$

$$
T_{\mathrm{TMA}}
=T_{\mathrm{issue,one}}
+T_{G\rightarrow S}
+T_{\mathrm{mbarrier}}
+T_{S\rightarrow R}
+T_{\mathrm{residency\ loss}}
-T_{\mathrm{hidden}}
$$

TMA 主要降低第一项：`issue,many → issue,one`。

如果剩余几项仍大于隐藏掉的 global-memory stall，TMA 仍然输给 direct load。

---

## 3. D005：cp.async 为什么慢了 44.5%？

D005 在 Phase 0 之前发出 state copy，希望与 Q/K normalization 和 gate preparation 重叠：

```text
cp.async state copy ─────────┐
Phase 0 compute ─────────────┤
                             ↓
                     wait_group + barrier
                             ↓
                       recurrence
```

从源代码看，已经存在 overlap window。

但候选新增了四类成本。

### 3.1 Copy issue 不是免费

128 个线程通过 `(16,8)` layout 协作搬运，源代码 8×2 循环覆盖 32 KiB tile。

动态上是：

```text
2,048 次 16-byte copy / CTA
```

最终 SASS inventory 中有 16 个静态：

```text
LDGSTS.E.BYPASS.128
```

“16 个静态指令位点”不等于只搬 256 bytes。每个位点会被整个 CTA 和循环动态执行多次。

### 3.2 多了一次 SMEM→register

Baseline 的 state load 直接到寄存器；候选还要执行额外 `LDS.128`。

| 指令位点 | Direct | D005 |
| --- | ---: | ---: |
| State `LDG.E.128` | 16 | 0 |
| `LDGSTS.E.BYPASS.128` | 0 | 16 |
| `LDS.128` | 4 | 20 |

### 3.3 多了一道同步

Block barriers：

```text
1 → 2
```

Wait 必须位于 recurrence 真正读取 state 之前。若 Phase 0 不够长，仍然要等。

### 3.4 Occupancy 下降

| Resource | Direct | D005 |
| --- | ---: | ---: |
| Registers / thread | 64 | 72 |
| Dynamic SMEM | 1,356 B | 34,124 B |
| CTA ceiling | 8 / SM | 6 / SM |
| Resident-warp ceiling | 50% | 37.5% |

当当前 CTA 等待 memory/barrier 时，可切换的其他 warp 更少。软件试图制造 latency hiding，却削弱了硬件原本的 latency hiding。

结果：

```text
9.262800 → 13.385560 μs
+4.122760 μs
0/30 wins
```

Correctness 对 random/finite-edge × 1/128 steps 的 output/state 全部 byte-exact，所以回归不是漏拷或 race。

---

## 4. D006：TMA 真正解决了哪一部分？

D006 保留同一块 32 KiB SMEM state，只把 cooperative copy 换成一个二维 TMA tile transfer。

TMA descriptor 描述：

```text
[pool, value_head, V, K]
```

Warp 0 初始化 mbarrier，并声明精确 transaction bytes：

$$
2\times128\times128=32768\ \mathrm{bytes}
$$

一次 `UTMALDG.4D` 代替大量线程的 copy issue。

结果：

```text
D005 cp.async: 13.385560 μs
D006 TMA:      11.317040 μs
recovered:      2.068520 μs
```

这证明 D005 的一部分回归确实来自 cooperative copy issue。

但 TMA 相对 direct source：

```text
9.257840 → 11.317040 μs
+2.059200 μs
+22.243%
0/30 wins
```

资源仍然是：

| Resource / instruction | Direct | D006 TMA |
| --- | ---: | ---: |
| Registers / thread | 64 | 72 |
| Dynamic SMEM | 1,356 B | 34,132 B |
| `UTMALDG.4D` | 0 | 1 |
| `LDS.128` | 4 | 20 |
| Block barriers | 1 | 2 |
| CTA ceiling | 8 / SM | 6 / SM |

![cp.async、TMA 与 Direct Load 的成本分解](/assets/blog-state-carrier-costs.svg)

*图 2：TMA 消除了 D005 的大部分 copy-issue 税，但没有消除一次性 tile 的 SMEM hop、同步和 residency 损失。*

所以正确结论不是：

> “TMA 比 cp.async 快 15.45%，因此 TMA 方案成功。”

而是：

> **TMA 成功解释并回收了一部分失败成本，但仍没有越过 direct load baseline。**

---

## 5. TMA Coordinate 为什么需要故意写错的负控制？

TMA tensor map 的轴顺序非常容易错。

本例 runtime coordinates 包含：

```text
pool_slot
value_head
```

巧合的是，两者大小都是 32。

如果交换它们：

```text
[pool_slot, value_head]
        ↓ swap
[value_head, pool_slot]
```

坐标仍然在合法范围内：

```text
0 ≤ coordinate < 32
```

Kernel 不会 launch failure，却会读取错误 state。

D006 保留了 swapped-coordinate negative control：四个 correctness case 的 output/state 全部 `byte_exact=false`。

它证明 oracle 能抓住一种最危险的 TMA bug：

> **Descriptor 合法、地址也合法、kernel 能跑，但语义轴错了。**

如果没有这个负控制，“最终版本 exact”仍可能只是测试输入没有区分两个轴。

---

## 6. D001：不经过 SMEM，只把 LDG 提前可以吗？

另一个自然想法是保留 direct load，只把下一组 `LDG.128` 提前：

```text
计算 group 0
  → 提前 load group 1
  → pack/store group 0
  → 使用 group 1
```

这样没有 SMEM second hop，似乎更合理。

但 PTX 不是最终机器调度。

`ptxas` 已经在 control SASS 中把下一组四条 `LDG.128` 穿插进上一组 BF16 pack/store。手工移动 PTX 没有发现新的依赖自由度，反而延长 load result 的 live range。

结果：

| Candidate | Registers | Spill | Median | Verdict |
| --- | ---: | ---: | ---: | --- |
| R93 vector-I/O control | 64 | 0 | 8.912480 μs | baseline |
| Hoist n=1/2/3 | 71 | 0 | 9.115320 μs | +2.2759%，0/30 |
| Hoist n=4 | 64 | local store/load | 未计时 | static reject |
| Full-group hoist | 64 | local store/load | 未计时 | static reject |

这里出现另一个关键公式：

$$
\mathrm{earlier\ load}
\Rightarrow
\mathrm{longer\ live\ range}
\Rightarrow
\mathrm{more\ registers\ or\ spill}
$$

Latency hiding 不是把 load 行往前移这么简单；必须看最终 SASS 和物理资源。

---

## 7. D017：什么都不搬，只改 Cache Hint 呢？

D017 选择最轻量的修改：

```text
st.global.v4.b32
        ↓
st.global.L1::evict_last.v4.b32
```

16 条 recurrent-state stores 从 `STG.E.128` 变成 `STG.E.EL.128`。地址、字节数、算术、launch、register、SMEM、spill 全部不变。

它的确得到正方向：

| Metric | Value |
| --- | ---: |
| Direct median difference | 0.006680 μs |
| Paired median saving | 0.004560 μs |
| Wins | 21/30 |
| Bootstrap CI95 | [0.002440, 0.006960] μs |
| Relative latency | −0.074926% |

但已有 A/A absolute p95：

```text
0.014276 μs
```

预注册 operator floor：

```text
0.030000 μs
```

候选 saving 只有噪声地板约三分之一、工程门槛约 15%。因此仍然停止。

这说明即使修改：

- 真实进入机器码；
- resource-neutral；
- byte-exact；
- bootstrap 方向为正；

也不自动值得增加 production selector、额外 cubin 和维护分支。

---

## 8. 四种“隐藏 State 延迟”的策略，为什么都没有晋级？

这些不是同一个 timing run，不能拼成一条伪 waterfall。它们各自相对自己的冻结 control 裁决：

| Case | Mechanism | 做对了什么 | 为什么停止 |
| --- | --- | --- | --- |
| D005 | `cp.async → SMEM` | 真异步 copy、byte-exact | one-shot、second hop、barrier、8→6 CTA/SM，+44.5% |
| D006 | TMA → SMEM | 一次 `UTMALDG.4D`，回收 copy issue | second hop/residency 仍在，+22.2% |
| D001 | 提前 direct `LDG.128` | 保持数据路径 | ptxas 已 interleave；live range 变长，+2.28% |
| D017 | `evict_last` store hint | 资源中性、真实小正向 | saving 低于 A/A p95 和工程 floor |

![四种 State Latency-Hiding 候选的独立裁决](/assets/blog-state-carrier-verdicts.svg)

*图 3：四条 case 的 baseline 和证据边界不同，不能相互相减；共同结论是每种机制都必须覆盖其新增成本并通过自己的 admission gate。*

---

## 9. 什么时候 Direct Load 反而是最佳抽象？

Direct GMEM→register 往往适合：

- 每个元素只消费一次；
- load 已经 coalesced/vectorized；
- 编译器已有足够 interleave；
- active warps 足以隐藏 memory latency；
- staging 会降低 CTA residency；
- 数据不需要跨 warp/CTA 共享。

它的优点不是“指令老”，而是路径短：

```text
没有额外 SMEM allocation
没有 mbarrier / wait_group
没有 SMEM-to-register reload
没有 descriptor coordinate
没有 staging lifetime
```

在一次性数据上，简单路径就是更小的状态机。

---

## 10. 什么时候 cp.async 值得？

`cp.async` 更适合：

1. Tile 会被重复消费；
2. 下一 tile copy 与当前 tile compute 有稳定双缓冲窗口；
3. 线程协作 copy 的 issue 成本能被计算摊薄；
4. SMEM allocation 不跨 occupancy cliff；
5. Wait/barrier 不落在每个短 phase 的关键路径上；
6. Layout transformation 或 multicast-like reuse 真正需要 SMEM。

粗略地说：

$$
\mathrm{benefit}
\approx
\mathrm{reuse}\times T_{G\rightarrow R}
+T_{\mathrm{hidden}}
-T_{\mathrm{extra\ hop}}
-T_{\mathrm{sync}}
-T_{\mathrm{occupancy\ loss}}
$$

Reuse 为 1 时，前半部分很难覆盖后半部分。

---

## 11. 什么时候 TMA 值得？

TMA 的强项是：

- 大型多维 tile；
- 复杂地址计算；
- 少量 producer 发起大规模搬运；
- shared-memory pipeline；
- cluster multicast；
- 多个 consumer 对同一 tile 复用；
- copy issue 本身已经成为瓶颈。

但每次看到 TMA，都要问：

```text
搬到 SMEM 后会读几次？
谁是消费者？
TMA 与多少计算真正重叠？
mbarrier wait 在哪条依赖边？
SMEM / registers 会让 residency 跨哪个台阶？
descriptor axes 有负控制吗？
```

如果答案是：

```text
只读一次
只有当前 CTA 消费
短 Phase 0 不足以隐藏
8 CTA/SM 降到 6
```

那么 `UTMALDG.4D` 的存在不能替代 wall-time gate。

---

## 12. Static Instruction Count 为什么容易误导？

D005 有 16 个静态 `LDGSTS.E.BYPASS.128` 位点；D006 只有 1 个 `UTMALDG.4D`。

从 16 到 1，看起来是巨大胜利。

但 static count 不等于：

- 动态执行次数；
- 搬运字节数；
- consumer load 次数；
- barrier 等待时间；
- CTA residency；
- operator latency。

可以把机器码证据分成三层：

| Evidence | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| Opcode exists | 优化没有被编译器删除 | 更快 |
| Resource report | registers/SMEM/spill 形态 | 真实并发与关键路径 |
| No-profiler A/B | 该 exact operator cell 更快/更慢 | 模型或 serving 收益 |

三层缺一不可，但 authority 不同。

---

## 13. 怎样在写代码前判断 Staging 是否值得？

先做一个 reuse/headroom screen：

### 13.1 Reuse count

$$
R=\frac{\text{SMEM consumer reads}}{\text{GMEM tile loads}}
$$

若 $R=1$，默认怀疑 staging。

### 13.2 Overlap window

测量或静态估计：

$$
T_{\mathrm{independent\ compute}}
\quad\text{vs.}\quad
T_{\mathrm{tile\ arrival}}
$$

若 independent compute 更短，wait 仍会落在关键路径。

### 13.3 Resource cliff

计算 staging 前后：

```text
registers/thread
dynamic SMEM/CTA
resident CTAs/SM
resident warps
```

不要只看 shared-memory 总量“还没超硬件上限”。从 8 CTA 降到 6 已经是性能状态变化。

### 13.4 Extra hop

列出实际数据路径：

```text
GMEM → register
vs.
GMEM → SMEM → register
```

只有 reuse/overlap 能补偿第二跳。

### 13.5 Correctness negative control

尤其是 TMA，故意交换两个等长 axis，确认 oracle 会失败。

---

## 14. 这些负结果关闭了什么，又没有关闭什么？

已经关闭：

- 在冻结 B32/T1 exact shape 上，把一次性 32 KiB state 整块 staged 到 SMEM；
- 用 cooperative `cp.async` 或 one-shot 2D TMA 实现上述 staging；
- 只移动相同 PTX state loads，期待获得 ptxas 尚未发现的 overlap；
- 为 0.00456 μs saving 增加 production cache-policy 分支。

没有关闭：

- 同一 state tile 被多个 token/query 重用；
- persistent CTA 内多步 recurrence；
- 更小 tile 的双缓冲；
- cluster multicast 给多个 CTA；
- 新 state representation 降低 SMEM/register footprint；
- 明确更短 reuse distance 与更高 cache pressure 下的 hint；
- 能保持 8 CTA/SM 的 staging 设计。

负结果的价值不是宣布“TMA 没用”，而是精确写出它在什么前提下没用。

---

## 15. 最后记住

1. **`cp.async` 和 TMA 改变的是搬运控制与目的地，不自动减少数据路径。**
2. **一次性数据经过 SMEM，通常是在 direct load 上增加第二跳。**
3. **TMA 可以显著优于 cooperative copy，却仍输给不 staging。**
4. **提前 load 会扩大 live range；必须看最终 SASS、register 和 spill。**
5. **真实小正值仍要超过 A/A 噪声和工程 materiality floor。**

选择搬运原语时，不要问：

> “哪个 CUDA feature 更新？”

而要问：

> **“这份数据会被复用几次、能隐藏多少等待、要多走几跳、会牺牲多少 resident work？”**

---

## Evidence boundary

- Source snapshot：[`SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab)。
- Replayed cases：D005 `cp.async` staging、D006 TMA staging、D001 state-load retime、D017 `evict_last` hint；所有结构化 replay 与资产检查通过。
- Operator cell：B200/SM100a、B32/T1、Q/K8、V32、K=V=128、BF16 state、1024 CTAs × 128 threads。
- D005/D006 的 correctness 覆盖 random/finite-edge × 1/128 steps；D001/D017 的冻结 scope 较窄，不升级成模型长期状态证明。
- 不同 case 有不同 source/control，本文没有把四组 latency 拼成一个 waterfall。
- D005/D006 fresh build/operator 因候选 patch/JIT 环境未冻结而 BLOCKED；D001/D017 的统一 fresh GPU wrapper 也仍需参数化。
- 四条候选都没有 TP2 E2E promotion；graph 投影不是 E2E measurement。
- 状态与重开条件见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
