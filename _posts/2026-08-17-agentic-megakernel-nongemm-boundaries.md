---
layout: post
title: "为什么 MegaKernel 不该只盯着 GEMM：删掉三类小边界，HTTP 吞吐反而提升 7.116%"
date: 2026-08-17 04:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [MegaKernel, OMoE, Kernel Fusion, CUDA Graph, B200, Serving, Evidence]
reading_time: 25
cover_image: /assets/blog-agentic-megakernel-boundaries.png
excerpt: "OMoE LS18R 没有加速主 GEMM，而是让 RoPE 原地写、K/V 一次 scatter、residual add 与 RMSNorm 融合。本文解释这些重复 seam 为什么能影响 HTTP serving，并用 Hazy pipeline、L2 prefetch 与 Paged KV 的负结果说明局部 Graph win 不能越级成产品结论。"
---

> 本文来自 `agentic-megakernel` evidence hub 的 contract、manifest、scoped REPORT 与教学文档。该 Hub 是技术谱系和证据索引，不是实现仓库；部分 measured revision 与 raw artifact 仍为 local-only。文中只使用已经绑定的 bounded verdict，不把历史索引完整误写成独立复现完整。

提到 MegaKernel，很多人首先想到的是：

```text
把整个模型塞进一个 persistent kernel
        ↓
减少 launch
        ↓
让所有算子在 GPU 内部调度
```

这个方向没有错，但它很容易把注意力全部吸到 GEMM、Tensor Core 和 scheduler 上。

在 OMoE 的一次 Llama-3.1-8B 优化中，真正通过 HTTP serving 门槛的 LS18R non-GEMM bundle 没有让矩阵乘更快。两条对照路径使用相同的 block-scaled FP8 decode MLP。候选只删除了三个反复出现的边界：

1. RoPE 不再生成一份额外中间结果，而是原地更新；
2. K 和 V 不再分别 scatter，而是一次写入相邻布局；
3. residual add 不再物化后交给 RMSNorm，而是 fused add + RMSNorm。

在冻结的 capacity-16 HTTP workload 上，pooled throughput 从：

$$
3066.663
\longrightarrow
3284.878\ \mathrm{token/s}
$$

提高 `7.116%`。两个相反执行顺序的 pair 分别提高 `7.735%` 和 `6.502%`。

这个结果揭示了一个比“融合越大越好”更精确的观点：

> **MegaKernel 最有价值的能力之一，是重新设计 producer/consumer 边界，让高频的小物化、小 launch 和小搬运真正消失；最终物理实现不一定是一个巨型 kernel。**

---

## 1. 先区分两件事：大边界与大 Kernel

“优化边界变大”和“代码必须变成一个 kernel”不是同义词。

例如下面这条路径：

```text
QKV projection
    ↓
RoPE
    ↓
K scatter
    ↓
V scatter
    ↓
Paged / dense attention consumer
```

如果 K/V scatter 能接受 RoPE 之后的最终 layout，我们可以：

- 把 RoPE 与 store 合在一个 kernel；
- 保留 RoPE kernel，但让它直接写 consumer layout；
- 用一个 combined store 替代两个 scatter；
- 在 persistent executor 中把它们变成相邻 instruction。

四种物理实现都可能表达同一个更大的数据流边界。真正的目标是删除中间状态，而不是追求 kernel 数量最小。

一个稳定的原则是：

$$
\text{先决定什么工作可以消失}
\quad > \quad
\text{再决定剩余工作放进几个 kernel}
$$

---

## 2. 三个不起眼的 seam 为什么会变成大问题？

在 Transformer decode 中，一次 layer 的主 GEMM 可能很大，但相邻的 elementwise、layout 和 cache 操作会在每层、每 token 重复。

如果某个边界每层付出 $t_{\mathrm{seam}}$，模型有 $L$ 层，生成 $T$ 个 token，那么粗略成本是：

$$
T_{\mathrm{repeated\ seam}}
\approx
L \times T \times t_{\mathrm{seam}}
$$

一个只有几微秒的 seam，乘上几十层和上百个 decode step，就不再是“小东西”。它还可能带来：

- 中间 tensor 分配；
- HBM 写回和再次读取；
- 独立 kernel launch；
- CUDA Graph 节点和 dependency edge；
- 新的 buffer lifetime；
- producer 与 consumer 之间更晚的可见时刻。

![LS18R non-GEMM bundle 删除的三类重复边界](/assets/blog-agentic-megakernel-seams.svg)

*图 1：三项修改都在 GEMM 周围工作，目标是让数据直接落到下一位消费者需要的状态。*

---

## 3. 第一项：RoPE 为什么适合原地做？

RoPE 对 Q/K 的成对通道做旋转。概念上可以写成：

$$
\begin{bmatrix}x'_0\\x'_1\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}x_0\\x_1\end{bmatrix}
$$

如果下游不会再使用旋转前的 K，并且 buffer ownership 已经明确，那么 RoPE 可以原地更新对应区域，避免：

```text
读取 K
    ↓
写 K_rotated 临时 tensor
    ↓
scatter 再读一次 K_rotated
```

变成：

```text
读取 K
    ↓
原地写回 rotated K
    ↓
store 直接消费
```

但“原地”不是免费正确。必须证明：

- 没有其他 consumer 仍需要原始值；
- alias、stride 和对齐满足 kernel 假设；
- CUDA Graph replay 时地址稳定；
- 多序列 active position 不会互相覆盖；
- 异常或 fallback 不会留下半更新状态。

原地更新的本质是修改数据所有权，不只是少分配一个 tensor。

---

## 4. 第二项：为什么 K/V 应该一次 scatter？

K 与 V 通常来自同一次 projection，最终也按同一 token position 写入 KV cache。若二者地址相邻或 consumer 接受 `[K|V]` 组合布局，分别启动两个 scatter 会重复：

- 解析 token/position index；
- 计算 cache slot；
- 做 bounds/validity 判断；
- 发起 kernel；
- 访问相近元数据。

combined scatter 可以共享这些控制工作，并在一次 CTA 生命周期中写完两个 payload。

这类优化的收益经常不在 FLOPs，而在：

$$
\text{removed launches}
+ \text{removed metadata work}
+ \text{fewer cache round trips}
$$

正确性门也必须覆盖副作用，而不只是最终 token：

- 每个 K/V cache byte 是否写到正确位置；
- 未命中 slot 是否保持不变；
- 不同 active sequence 的 position 是否独立；
- K 与 V 的 layout、dtype 和 page arithmetic 是否一致。

对于纯 store，bitwise side-effect check 往往比只看最终 logits 更有定位能力。

---

## 5. 第三项：Residual Add + RMSNorm 为什么会改变数值顺序？

原始路径可能是：

```text
y = x + residual
z = RMSNorm(y)
```

如果 `y` 先以 BF16 写回，再由下一 kernel 读取，融合后在寄存器或 FP32 accumulator 中直接进入 RMSNorm，舍入次数可能变化。

RMSNorm 可以写成：

$$
\mathrm{RMSNorm}(y)
=
\frac{y}
{\sqrt{\frac{1}{d}\sum_{i=1}^{d} y_i^2 + \epsilon}}
\odot w
$$

融合删除了一次 materialization，却也可能改变：

- residual 加法的 rounding point；
- reduction 顺序；
- 中间精度；
- 最终 BF16 store 的时点。

因此 LS18R 的 correctness 不是简单要求每个 normalized byte 完全相同。它要求 residual side effect 正确，并为 teacher-forced model agreement 冻结了 98% gate。两组 hard-batch agreement 分别是 `98.320%` 和 `98.584%`，刚好说明“数值等价”需要预声明合同，不能在看到结果后再挑容差。

---

## 6. 为什么这三项组合能穿透到 HTTP Serving？

冻结 workload 是：

| 维度 | 值 |
| --- | --- |
| 模型 | Llama-3.1-8B-Instruct |
| GPU | 1×B200 |
| Server concurrency | 16 |
| Maximum sequence length | 2048 |
| Output tokens | 128 |
| Temperature | 0 |
| Timing boundary | HTTP streaming serving throughput |
| 两臂共有 | block-scaled FP8 decode MLP |

候选不是“全 BF16 对 FP8”，也不是换了 GEMM。两臂共享同一 decode MLP baseline，只切换 non-GEMM bundle。

两个 opposite-order pair 都为正：

| Pair | Throughput improvement |
| --- | ---: |
| Order pair 1 | +7.735% |
| Order pair 2 | +6.502% |
| Pooled median | **+7.116%** |

这降低了“candidate 总是恰好跑在第二个、机器更热或 cache 更暖”的解释概率。

为什么小 seam 能穿透到产品边界？一个合理的机制链是：

```text
每层删除中间 tensor / launch / HBM round trip
        ↓
decode step 的重复固定成本下降
        ↓
capacity-16 下每个请求更快推进
        ↓
20 秒 HTTP 窗口完成更多 token
```

但这里仍然要区分“证据支持的事实”和“因果推断”。S4 样本直接支持吞吐提升；每一项各自贡献多少，并没有被组合实验识别。

---

## 7. MegaKernel 思维的核心：消除边界，而非扩大代码体积

![从大 kernel 迷思到 dataflow boundary 重写](/assets/blog-agentic-megakernel-boundary.svg)

*图 2：同一个数据流重写可以由 fused kernel、combined store 或 persistent instruction 表达。*

一个有用的 MegaKernel 分析顺序是：

1. 画出 producer/consumer 和 tensor lifetime；
2. 找到只被下一节点消费的中间结果；
3. 检查 producer 能否直接写 consumer layout；
4. 检查边界是否每层/每 token 高频重复；
5. 先删除 materialization 和多余控制；
6. 再决定用 fusion、epilogue、library kernel 还是 persistent executor 实现；
7. 最后才调 tile、register 和 pipeline stage。

如果一上来就追求“一个 kernel”，很容易把原来不同 phase 的最大资源需求叠在一起：

- 共享内存按最大阶段预留；
- 寄存器 live range 变长；
- CTA geometry 对某个阶段不合适；
- 原本可并发的工作被串行化；
- 一个全局 barrier 放大最慢 worker。

---

## 8. 反例一：更深、更异步，为什么反而慢 0.704%？

Hazy depth4 + pending1 尝试使用 depth-4 output ring，并允许一个 TMA store group 保持 pending，以便下一组 UpGate 继续计算。

它在直觉上很漂亮：

```text
计算下一组
    ↕ overlap
上一组异步 store 尚未完全退休
```

正确性也通过了 100/100 bitwise full-model replay。但 captured Graph latency 从：

$$
687.597\ \mu s
\longrightarrow
692.435\ \mu s
$$

回退 `0.704%`。

更有意思的是，depth-4 但不加 pending-store policy 的 control 是 `677.521 μs`，反而更快。预声明 candidate 是“depth4 + pending1”组合，不能在看到数据后把另一个 arm 偷换成成功结论。

失败机制可能包括：

- 更长 live range；
- 更多 stage bookkeeping；
- 新的 wait/fence；
- register/shared-memory 压力；
- consumer 根本无法利用提前量。

结论是：

> **异步 primitive 只提供重叠可能；只有隐藏了关键路径上的真实 stall，重叠才是收益。**

---

## 9. 反例二：L2 Prefetch 指令真的存在，为什么仍然没用？

Hazy inter-instruction prefetch 在 instruction `i` 结束前，提前触碰 `i+1` 或 `i+2` 的首个 16 KiB weight tile，希望权重在 consumer 启动前进入 L2。

实验甚至保留了 source、binary、SASS、register、spill 和 instruction-count proof，证明“prefetch 不是被编译器删掉”。正确性也通过。

但六个 cell 中五个回退 `0.010%–0.267%`，唯一正向 cell 只改善 `0.095%`，远低于 1% gate。

Prefetch 可能失败，因为：

- 太早：数据在使用前被驱逐；
- 太晚：consumer 已经因为 miss 停住；
- 额外指令增加 dependency 或 register；
- 预取污染其他更有价值的 L2 数据；
- 原路径根本没有足够大的 memory stall。

“SASS 里有 prefetch”是 path proof，不是 performance proof。

---

## 10. 反例三：Focused Graph 快 10.56%，Serving 为什么仍被拒绝？

OMoE paged KV + native combined store 在 focused B16 decode Graph 中，将每 step 从 `4.9004 ms` 降到 `4.3829 ms`，节省 `0.5175 ms`，改善 `10.56%`。

这个局部结果是真实且 bounded accepted。它将两次 post-RoPE scatter 合成一次，并只处理 live paged KV，而不是 dense max-capacity work。

但 capacity-16 HTTP serving 的 pooled throughput 只提升 `2.02%`：

| Boundary | Result | Verdict |
| --- | ---: | --- |
| Focused decode Graph | −10.56% latency | S3 bounded accept |
| Capacity-16 HTTP serving | +2.02% throughput | S4 reject，低于 5% gate |

这两个结论不矛盾。HTTP serving 还包含 prefill、scheduler、sampling、同步、host/server overhead 和请求交错。focused boundary 节省的 `0.5175 ms` 可能已经被 overlap 隐藏，或者在产品分母中占比不足。

![同一机制在 S3 Graph 与 S4 Serving 的不同裁决](/assets/blog-agentic-megakernel-evidence-levels.svg)

*图 3：证据等级不是“分数高低”，而是不同 estimand；S3 win 不能覆盖 S4 stop。*

---

## 11. 为什么负结果是 Agentic 优化的核心资产？

如果 Agent 只记住“最后哪个 commit 最快”，它会反复走进已经关闭的分支：

- 看到 TMA 就加深 pipeline；
- 看到大权重就加 prefetch；
- 看到 focused Graph win 就直接默认开启；
- 看到 kernel 数下降就宣布 MegaKernel 成功。

一个可执行的优化记忆应该保存：

```text
Claim
  + Workload contract
  + Code/source identity
  + Correctness gate
  + Measurement boundary
  + Raw/order evidence
  + Verdict
  + Reopening condition
```

例如：

- depth4+pending1：关闭，只有 profiler 再次证明目标 wait 位于关键路径才重开；
- L2 prefetch：关闭，只有新 architecture/workload 或明确 cache-miss/stall 证据才重开；
- paged KV：实现保留、默认关闭，等待产品关键路径暴露更大占比；
- LS18R non-GEMM：只在冻结 capacity-16 serving cell bounded accepted。

这才是“Agentic MegaKernel”比自动调参更深的一层：Agent 不仅生成候选，还要知道哪些分支已经被什么证据关闭。

---

## 12. 如何寻找下一个 non-GEMM seam？

可以从下面的 edge record 开始：

```text
producer → tensor / metadata → consumer
shape / dtype / layout / bytes
allocation / lifetime / owner
stream / Graph node / synchronization
下一位 consumer 的最终格式
```

优先寻找：

1. 只被下一节点消费的大中间 tensor；
2. producer 后立刻 transpose、pack、quantize、scatter 的链；
3. 每层重复构造的 position、page、route 或 scale metadata；
4. residual/add/norm 之间的物化；
5. K/V、gate/up 等共享索引和地址计算却分两次 launch 的操作；
6. graph 中被过早 join 的独立分支；
7. host scalar read 或 per-layer descriptor setup。

然后问三个问题：

- 删除这条 seam 能减少什么真实工作？
- 新边界会增加什么 register/SMEM/lifetime/synchronization 成本？
- 这条 seam 是否位于目标 workload 的关键路径，而不是已经被 overlap 隐藏？

---

## 13. 证据边界：这篇文章没有证明什么？

LS18R 的 bounded verdict 只覆盖：

- Llama-3.1-8B-Instruct；
- 1×B200；
- capacity 16；
- max sequence length 2048；
- output 128；
- 20 秒 HTTP streaming throughput；
- 两臂共有 block-scaled FP8 decode MLP。

它没有证明：

- 所有模型的 RoPE/KV/norm fusion 都有 7%；
- 全 BF16 比 FP8 更快或更慢；
- batch-1 Graph latency同步改善；
- OMoE 普遍超过 SGLang、Hazy 或 llama.cpp；
- 三项子优化分别贡献多少。

此外，Hub 是 retrospective migration：measured worker revision 已不可达，raw archive 仍 local-only；可达的 later integration 与 archived patch 有绑定关系，但独立公开复现还没有完成。对外写作必须保留这条 provenance debt。

---

## 结语：不要只优化机器，也要优化传送带

GEMM 像工厂里的大型机器，很容易成为所有优化注意力的中心。但真正的流水线还包含传送带、包装、扫码、换箱和等待。

LS18R 的 7.116% 告诉我们：

```text
RoPE 后少一次临时交接
K/V 少一次重复 scatter 控制
Residual/Norm 少一次物化和 launch
        ↓
每层都少一点
        ↓
每 token 都少一点
        ↓
HTTP serving 窗口完成更多 token
```

Hazy 和 Paged KV 的负结果又提醒我们：更深异步、更多 prefetch、更快 focused Graph 都不会自动穿透到产品关键路径。

所以，MegaKernel 的终极问题不该是：

> “还能把多少代码塞进同一个 kernel？”

而应该是：

> **“哪些 producer/consumer 边界正在反复物化、搬运和等待？删掉它们以后，完整 workload 的关键路径真的缩短了吗？”**

这两个问题看起来只差一句话，却决定了我们是在堆叠复杂度，还是在真正删除工作。
