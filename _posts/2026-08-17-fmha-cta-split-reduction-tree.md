---
layout: post
title: "为什么少用 CTA 反而会算错？从 GQA Head Group、KV Split 到 Reduction Tree"
date: 2026-08-17 14:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [FMHA, GQA, CUDA, CTA, Reduction Tree, CGA, B200, Numerical Correctness]
reading_time: 29
cover_image: /assets/blog-fmha-cta-reduction.png
excerpt: "在 Q16/KV1 的 TP2 decode attention 中，Q8 tile 虽然出现在 provider metadata 里，却违反完整 GQA head-group 合同；把每个 request 的 KV split 从 4 个 CTA 减到 1/2 个，又改变了 softmax partial 的加法树。本文解释为什么 CTA 拓扑同时是性能策略、数值协议和 launch ABI。"
---

> 本文基于 [`qhy991/SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab) 的 D041 FMHA Q-head tile oracle 与 D045 multi-CTA split/reduction oracle。两条 machine-readable replay 与本地资产检查已通过；fresh operator run 仍需要冻结的 TRT-LLM/FlashInfer cubin family、生产 paged-KV 输入和空闲 B200，因此本文引用归档 operator evidence，不声称重新生成 cubin 或运行模型 E2E。

GPU 优化里，“减少 CTA”听起来几乎总是合理：

- launch 的 block 更少；
- cooperative reduction 参与者更少；
- shared memory 协调可能更简单；
- tail 也可能更短。

但在一次 paged FMHA 实验中，把每个 request 的 KV split 从 4 个 CTA 减到 2 个，结果不是“快一点或慢一点”，而是：

```text
69,763 个 BF16 元素发生变化
```

最大绝对误差只有：

```text
0.000244140625
```

数值看起来极小，却不再满足冻结的 byte-exact 模型合同。

更早一步，试图把 Q-head tile 从 16 缩到 8，甚至没有进入计时：provider launch 直接失败。

为什么 CTA 数量会同时影响：

1. kernel 能不能合法 launch；
2. attention 输出的最后几个 bit；
3. occupancy 与 SM 覆盖；
4. CGA shared-memory reduction 和 global-memory scratch 的成本？

因为 FMHA 的 CTA topology 不是“调度参数”这么简单。它同时编码了：

> **GQA 的语义分组、KV 工作切分、softmax partial 的归约树，以及跨 CTA 通信协议。**

---

## 1. 先冻结真实 Shape：Q16 / KV1 到底是什么意思？

这组 TP2 decode attention 的 rank-local shape 是：

| 维度 | 值 |
| --- | ---: |
| GPU | 1×B200 / SM100a operator cell |
| Dtype | BF16 |
| Local query heads | 16 |
| Local KV heads | 1 |
| Head dimension | 256 |
| Sequence length | 32769 |
| Page size | 128 |
| Concurrent sequences | 32 |
| Incumbent | Q16 tile，4 KV-split CTAs / request，CGA-SMEM merge |

这是 GQA（Grouped Query Attention）：多个 Q head 共享一组 K/V head。

本地比例是：

$$
\mathrm{numHeadsQPerKv}
=
\frac{16}{1}
=16
$$

这意味着每个 KV head 对应一个完整的 16-query-head semantic group。

初学者很容易把 Q tile 当成普通矩阵 tile：既然有 16 个 Q heads，那么两个 Q8 CTA 拼起来不就行了吗？

答案取决于 provider contract。

当前 FMHA provider 要求：

> 一个 CTA 必须拥有完整的 `numHeadsQPerKv` group。

所以 Q8 不是“更小但合法”，而是没有覆盖完整 semantic group。

---

## 2. Provider 里有 Q8 Cubin，为什么仍不能用？

预编译 provider 往往包含很多 cubin / tile metadata：

```text
Q8
Q16
Q32
不同 head dim
不同 split mode
不同 CGA shape
```

看到 Q8 artifact，只能证明：

> 某个 Q8 kernel binary 存在。

它不能证明：

- 当前 Q/KV head ratio 合法；
- metadata 能构造正确的 launch descriptor；
- 两个 CTA 会按预期共享一个 KV group；
- output layout 与 downstream consumer 匹配；
- 当前 split/reduction mode 支持这个 tile。

![Q16/KV1 的完整 GQA Head Group 与 Q8/Q16/Q32 Tile 合法性](/assets/blog-fmha-gqa-tile-contract.svg)

*图 1：Q8 小于 16-head semantic group，当前 provider 不能用两个 CTA 自动拼接；Q16 精确覆盖一组，Q32 合法但一次覆盖两个组宽度，可能浪费工作。*

D041 用独立 child process 测试 Q8 auto 与 forced 两种模式。两者都：

```text
CUDA launch failure
```

与此同时，两个 discovery run 中的 Q16 control 都能正常 launch。

所以这不是：

- GPU 环境整体坏了；
- CUDA context 一开始就污染；
- 输入 tensor 不合法。

它是候选违反 provider invariant 的 contract reject。

这条顺序非常重要：

```text
artifact exists
  ≠ launch contract valid
  ≠ output correct
  ≠ performance candidate
```

Q8 在第一道门就应该停止，没有 latency 数字。

---

## 3. 为什么 Launch Failure 要放进独立进程？

CUDA launch failure 可能污染当前 context 的后续调用。若把 Q8、Q16、Q32 全放在同一进程顺序执行：

```text
Q8 失败
  ↓
CUDA context 进入 error state
  ↓
后面的 Q16 也失败
```

研究者可能误以为所有 tile 都不可用。

D041 将 Q8 contract probe 放进 child process：

```text
parent
  ├── child: Q8 auto → launch failure → child exits
  ├── child: Q8 forced → launch failure → child exits
  └── clean process: Q16/Q32 oracle + timing
```

这不是测试框架细节，而是 failure containment。

对于可能触发：

- illegal memory access；
- misaligned address；
- invalid configuration；
- device-side assert；
- provider launch failure；

的探索性候选，独立进程能防止一个失败 arm 改变后续 comparator 的环境。

---

## 4. Q32 合法且 Byte-exact，为什么仍然更慢？

Q32 大于完整 16-head group，provider 可以合法构造 launch。它与 Q16/CGA4 输出：

```text
SHA 相同
不同 BF16 元素 = 0
```

因此进入 timing。

30 个 randomized blocks，每个 block 100 次 Graph replay：

| Arm | Median | 相对 Q16 | Wins |
| --- | ---: | ---: | ---: |
| Q16 / CGA4 | 155.805278 μs | baseline | — |
| Q32 / CGA4 | 157.170401 μs | **+0.876172% 更慢** | 0/30 |
| Q16 / CGA4 B | 155.528159 μs | A/A −0.177862% | control |

Q16 相对 Q32 的 paired difference CI95 是：

```text
[-1.501598, -1.280484] μs
```

它与 A/A 区间分离，不像普通噪声。

Q32 为什么可能更慢？

- 当前只有 16 个 local Q heads，tile 宽于真实 semantic group；
- 更宽 tile 可能增加 inactive lane、predicate 或资源分配；
- register/shared-memory 形态为 Q32 编译，而不是按“只用一半”免费缩小；
- occupancy、warp scheduling 或 reduction bookkeeping 可能不如 Q16；
- provider 的 Q32 artifact 为其他 head topology 设计，不保证这个 exact shape 最优。

最关键的原则是：

> **合法只表示可以比较，不表示值得晋级。**

---

## 5. 第二个轴：为什么把长 KV 分给多个 CTA？

Sequence length 是 32769。单个 CTA 扫完整 KV 会有很长的 serial work。

Multi-CTA attention 把 KV 方向切成多个 split：

```text
request 0:
  CTA 0 → KV segment 0
  CTA 1 → KV segment 1
  CTA 2 → KV segment 2
  CTA 3 → KV segment 3
```

每个 CTA 计算自己的 softmax partial，然后合并。

32 个 concurrent sequences × 4 CTAs：

$$
32\times4=128\ \mathrm{CTAs}
$$

B200 有 148 个 SM。这使一次 wave 的 CTA 数接近整卡 SM 数。

如果改成两 CTA：

$$
32\times2=64\ \mathrm{CTAs}
$$

单 CTA则只有：

$$
32\times1=32\ \mathrm{CTAs}
$$

CTA 更少可以减少 merge 参与者，却也降低并行覆盖，延长每个 CTA 的 KV loop。

但 D045 更上游的问题不是性能，而是数值树。

---

## 6. Softmax Partial 不是简单相加

对一段 logits $x_j$，稳定 softmax 需要：

$$
m=\max_j x_j
$$

$$
l=\sum_j e^{x_j-m}
$$

$$
o=\sum_j e^{x_j-m}v_j
$$

多个 KV split 各自得到：

$$
(m_s,l_s,o_s)
$$

合并两个 partial 时：

$$
m=\max(m_1,m_2)
$$

$$
l=e^{m_1-m}l_1+e^{m_2-m}l_2
$$

$$
o=e^{m_1-m}o_1+e^{m_2-m}o_2
$$

最后输出通常是：

$$
y=\frac{o}{l}
$$

浮点加法不满足结合律：

$$
(a+b)+c\neq a+(b+c)
$$

改变 split 数不仅改变参与 CTA 数，还改变：

- 每个 partial 覆盖哪些 KV 元素；
- local max 与 rescale factor；
- partial merge 的层数与顺序；
- FP32 accumulation 的舍入轨迹；
- 最后 BF16 round 的输入。

所以 one-way、two-way、four-way attention 是不同的 numerical protocol。

---

## 7. D045：减少 Split 后，哪些输出变了？

冻结相同 tensors、page table、scale、stream、output shape 和 cubin family，只改变 provider split/reduction controls：

| Arm | KV split | Reduction carrier | 不同 BF16 元素 | Max abs error | Gate |
| --- | ---: | --- | ---: | ---: | --- |
| persistent1 | 1 | none / local | 66,893 | 0.000244140625 | reject |
| cga2 | 2 | CGA shared memory | 69,763 | 0.000244140625 | reject |
| gmem2 | 2 | global scratch | 69,763 | 0.000244140625 | reject |
| gmem4 | 4 | global scratch | 0 | 0 | time |

三个观察特别重要。

### 7.1 误差很小，不等于合同通过

`0.000244140625 = 2^{-12}`，肉眼看很小。但这次 campaign 预先冻结的是 BF16 byte-exact。

为什么可能需要这么严格？

- 模型 token/state gate 依赖完全一致的轨迹；
- 线上 deterministic replay 或 cache 复用要求稳定 bytes；
- 其他 kernel winner 也在同一 exact 合同下比较；
- 看到性能后再放宽 tolerance 会产生选择偏差。

如果要允许新 reduction tree，应创建一个新 contract，预先定义 token/state tolerance，而不是在候选变快后改旧门槛。

### 7.2 `cga2` 与 `gmem2` Hash 相同

两者都产生 69,763 个差异元素，而且输出 hash 相同。

这说明差异主要来自：

```text
split count: 4 → 2
```

而不是：

```text
carrier: CGA SMEM → GMEM scratch
```

这是一个很干净的因果分解。

### 7.3 One-way 又是第三种 Hash

Single CTA 没有跨 CTA merge，但它的 local partition 与计算顺序不同，因此形成第三条舍入轨迹。

![KV Split 数量、Reduction Tree 与 CGA/GMEM Carrier 的关系](/assets/blog-fmha-split-reduction-tree.svg)

*图 2：Split count 决定 partial tree，carrier 决定 partial 如何交换。只有保持四路树的 GMEM4 与 incumbent byte-exact。*

---

## 8. 唯一 Exact 的 GMEM4，为什么还是输？

`gmem4` 保留与 incumbent 相同的 4-way split，因此输出 byte-exact，可以计时。

30 randomized blocks × 100 Graph replays：

| Arm | Median | Delta | Wins |
| --- | ---: | ---: | ---: |
| default CGA4 | 155.523043 μs | baseline | — |
| GMEM4 | 157.101922 μs | **+1.015206% 更慢** | 0/30 |
| default CGA4 B | 155.547042 μs | A/A +0.015431% | control |

Incumbent 相对 GMEM4 的 paired CI95：

```text
[-1.620002, -1.478558] μs
```

为什么 CGA shared-memory merge 更适合这个形态？

### CGA4

```text
4 CTA in cluster
  → partial state in cluster-visible shared memory
  → cluster synchronization
  → merge without global scratch round trip
```

### GMEM4

```text
4 CTA
  → write partial state to global scratch
  → update/read readiness or counters
  → another CTA reads partials
  → global merge
```

GMEM 版本可能增加：

- global store/load；
- cache line traffic；
- atomic/counter 或 polling；
- producer-consumer visibility；
- scratch allocation 与 address calculation；
- 更长的 partial lifetime。

它保住了数值树，却失去了 incumbent 的低成本 carrier。

---

## 9. CTA 更少为什么不一定更快？

可以把 operator 时间粗略分解为：

$$
T
=
T_{\mathrm{KV\ scan}}
+T_{\mathrm{partial\ merge}}
+T_{\mathrm{coordination}}
+T_{\mathrm{underfill}}
$$

减少 split CTA：

- 可能降低 $T_{\mathrm{partial\ merge}}$；
- 但增加每 CTA 的 $T_{\mathrm{KV\ scan}}$；
- 让 128 CTA 变成 64 或 32，增加 $T_{\mathrm{underfill}}$；
- 改变 reduction tree，可能直接失去 correctness eligibility。

增加或减少 CTA 都不是单调旋钮。最优点由：

- sequence length；
- concurrent request 数；
- SM 数量；
- Q/KV head ratio；
- head dimension；
- page layout；
- partial state 大小；
- CGA/GMEM communication cost；
- exactness contract；

共同决定。

---

## 10. 为什么还要独立 A/A Graph？

单看 candidate 相对 baseline 慢 1%，仍可能问：这是不是 benchmark 自己漂了？

D041/D045 都额外 capture 第二个独立 incumbent Graph：

```text
default A
candidate
default B
```

A/A 的作用是测量：

- 两次独立 Graph capture 的差别；
- random block 的自然噪声；
- order 与 cache 状态；
- harness 本身的可重复性。

D045 中：

```text
candidate GMEM4: +1.015206%, 0/30
A/A incumbent B: +0.015431%, 10/30
```

Candidate 的 paired interval 与 A/A 明显分离，所以“更慢”不是由 capture A/B 自身产生的同量级波动。

![D041/D045 的 Contract、Correctness、Operator 三层裁决](/assets/blog-fmha-oracle-gates.svg)

*图 3：Q8 在 launch contract 停止；1/2-split 在 byte correctness 停止；Q32 与 GMEM4 正确但在 30-block operator gate 以 0/30 停止。*

---

## 11. 为什么没有继续跑 NCU、Nsys 或模型 E2E？

优化工作流常见的错误是：候选输了，仍然去 profile，希望找到“它理论上应该更快”的证据。

但 profiler 不能推翻 no-profiler wall-time gate。

两条实验已经给出：

- Q8：launch contract invalid；
- Q32：byte-exact，但 operator 0/30；
- 1/2 split：correctness reject；
- GMEM4：byte-exact，但 operator 0/30。

在预声明门槛下，没有 arm 有资格进入 E2E。

继续跑 NCU 可能回答：

- GMEM 多了哪些 load/store；
- Q32 的 register/occupancy 如何；
- CGA barrier 等待多少；

但不会把失败候选变成 winner。

只有当新的可证伪设计改变前提，例如：

> 用两 CTA 保持与四 CTA 完全相同的 partial ordering；

才应创建新 candidate 和新 contract。

---

## 12. “Tile”至少有三种不同含义

谈 kernel tile 时，需要区分：

### 12.1 Arithmetic tile

一次 CTA/warp 计算多少元素，例如 Q8/Q16/Q32。

### 12.2 Semantic tile

模型接口要求哪些元素必须作为一个不可分组处理，例如完整 `numHeadsQPerKv=16`。

### 12.3 Scheduling tile

多少 CTA 共同处理一个 request，如何分 KV、如何 merge partial。

一个 arithmetic tile 可能有 binary，却不满足 semantic tile；一个 scheduling tile 可能更少，却改变 numerical protocol。

因此 kernel registry 的 key 不应只有：

```text
head_dim=256
```

至少还要绑定：

```text
q_heads_per_kv
q_tile
kv_split_count
reduction carrier
reduction order / exactness class
sequence bucket
concurrent requests
page size
GPU / SM count
```

---

## 13. 对初学者：怎样审计一个 FMHA Provider Candidate？

### 第一步：写出 GQA semantic group

$$
H_{Q/KV}=H_Q/H_{KV}
$$

确认一个 CTA 是否必须覆盖完整 group。

### 第二步：区分 artifact availability 与 launch validity

Provider metadata 有 Q8/Q16/Q32，不代表当前 shape 全合法。用最小 child-process probe 验证。

### 第三步：冻结 split 与 reduction protocol

记录：

```text
KV partition
partial state layout
merge tree
carrier (CGA SMEM / GMEM)
final round point
```

### 第四步：先做 byte/token contract

若 campaign 要求 byte-exact，任何 changed element 都停止。若要允许 tolerance，应在看到性能前预声明。

### 第五步：带独立 A/A 做 randomized paired timing

避免一次 Graph capture 或固定顺序被误当候选差异。

### 第六步：只让通过 operator gate 的 arm 进入 E2E

Profiler 负责解释，不能替代 promotion。

---

## 14. 这两条负结果真正学到了什么？

D041 与 D045 没有找到新 winner，却排除了四条非常诱人的错误直觉：

1. **“Provider 里有 cubin，就能用于当前 shape。”** 错；Q8 违反完整 GQA group invariant。
2. **“更宽 tile 至少不会更慢。”** 错；合法 exact 的 Q32 稳定 0/30。
3. **“少几个 split CTA 只改变性能。”** 错；它改变 softmax partial tree 和 BF16 bytes。
4. **“把 SMEM reduction 换成 GMEM 只改变 carrier。”** 只有 split 数相同时才成立；此时 GMEM4 exact，却稳定更慢。

这些 negative results 关闭了 provider search tree 中的具体分支，也留下明确重开条件：

- 新 provider 支持合法 Q8 composition；
- Q32 在不同 Q/KV ratio 或 request count 下有新 workload cell；
- two-way algorithm 保持 exact four-way ordering；
- 新硬件/transport 改变 CGA 与 GMEM 成本关系。

---

## 15. 最后记住

CTA topology 不只是性能参数。

在 FMHA 中，它可能同时决定：

```text
谁拥有完整 GQA head group
谁扫描哪一段 KV
partial softmax 怎样合并
浮点加法树是什么
partial state 走 SMEM 还是 GMEM
整卡能同时运行多少 CTA
```

所以正确顺序是：

$$
\text{Semantic legality}
\rightarrow
\text{Numerical protocol}
\rightarrow
\text{Operator timing}
\rightarrow
\text{E2E}
$$

不是先找最小 CTA 数，再为结果补解释。

---

## Evidence boundary

- Source snapshot：[`SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab)。
- Cases：D041 `rejected-contract-and-operator`；D045 `rejected-correctness-and-operator`；两条 replay 与资产检查在当前 checkout 通过。
- Frozen operator cell：B200、BF16、local Q16/KV1、head dim 256、sequence 32769、page 128、32 sequences。
- Timing：30 randomized blocks × 100 Graph replays；operator only，无 E2E promotion。
- Q8 launch failure、1/2-split non-exact、Q32/GMEM4 0/30 均保持原 verdict；不从 profiler 或理论投影升级。
- Fresh operator run 缺冻结 cubin family、生产 paged-KV input 与 leased B200，当前只能 replay 小型归档 JSON。
- 状态与重开条件见 [GPU Systems Evidence Register](/gpu-systems-evidence.html)。
