---
layout: post
title: "手写 PTX 真的有 CUDA 做不到的能力吗？从 16-bit 标量 Load 到 128-bit Vector I/O"
date: 2026-08-17 05:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [PTX, CUDA, SASS, B200, Vector IO, Qwen3.5, GPU Optimization]
reading_time: 28
cover_image: /assets/blog-subcuda-ptx-vector-io.png
excerpt: "SubCUDA 在 Qwen3.5 TP2 GDN 上用 Direct PTX 将大量 16-bit 标量访存改成 64/128-bit vector I/O，算子快 3.60%、模型吞吐提高 0.498704%。随后 CUDA C++ 反事实又恢复了 90.20% 的 graph-wall 节省，说明 PTX 的价值在于可控 lowering 与机制诊断，而不是独占某种硬件能力。"
---

> 本文基于 [`qhy991/SubCUDA@d1db18f`](https://github.com/qhy991/SubCUDA/commit/d1db18fbc46f873d827bc7d276988d5cef3199ab) 的可复现案例库、Round93 五对 TP2 A/B、Round95 三臂 CUDA/PTX 对照及 SASS/NCU/NSYS 证据。性能结论只覆盖冻结的 Qwen3.5 TP2 B32 offline decode，不外推到其他 batch、TP、GPU 或在线 serving。

很多 CUDA 开发者对 PTX 有两种截然相反的想象：

第一种认为 PTX 是“终极优化语言”，只要手写就一定比 CUDA 快。第二种认为现代编译器已经足够强，PTX 只会制造不可维护代码。

一次 Qwen3.5 GDN 优化把这两种极端都推翻了。

Round93 直接修改生成后的 PTX，将 Q/K/V 和 output 的大量 16-bit 标量访存换成 64/128-bit vector I/O：

```text
operator: 9.242280 → 8.909120 μs/call  （−3.60474%）
TP2:      3529.435965 → 3547.037401 token/s（+0.498704%）
```

五对完整模型 A/B 全部获胜，output、recurrent state 和双 rank token hash 保持 exact。

这看起来像“PTX 做到了 CUDA 做不到的事”。但 Round95 又写了一个完全不含 inline PTX 的 CUDA C++ counterfactual。它发出了相同类别的宽 load/store 与 packed `FFMA2`，并恢复了：

$$
90.20\%
$$

的完整 TP2 graph-wall 节省。PTX 相对 CUDA 只剩 `0.0282 μs/call` 和 `0.492789 ms/graph` 的优势。

这组实验最值得学习的结论是：

> **PTX 的价值不是拥有 CUDA 语言永远无法触达的魔法指令，而是让我们隔离 lowering、验证机器级假设，并在编译器尚未给出理想 schedule 时提供一个可控载体。**

---

## 1. CUDA、PTX 和 SASS 到底是什么关系？

写 CUDA C++ 后，代码不会直接变成 GPU 执行的指令。一个简化流程是：

```text
CUDA C++ / Triton
        ↓ 前端编译
PTX：虚拟 GPU 指令集
        ↓ ptxas
SASS：特定 GPU 的机器指令
        ↓
SM100 / B200 执行
```

![CUDA、PTX、ptxas 与 SASS 的反馈闭环](/assets/blog-subcuda-compiler-pipeline.svg)

*图 1：最终执行的是 SASS。修改 CUDA 或 PTX 都必须检查目标机制是否真正存活到 SASS。*

三个层级承担不同责任：

- **CUDA C++ / Triton**：表达算法、数据流和可维护抽象；
- **PTX**：表达虚拟 ISA 语义，允许固定某些 load width、rounding 或 barrier 形式；
- **ptxas**：做寄存器分配、指令选择、调度和最终机器码生成；
- **SASS**：B200 实际执行的指令序列。

因此“我改了 PTX”不等于“GPU 执行方式变了”。ptxas 可能重新组合、删除或变换指令。反过来，普通 CUDA 的 `uint4`、`uint2` 和 SM100 intrinsic 也可能生成与手写 PTX 相同的指令类别。

---

## 2. 目标 Kernel 在做什么？

Qwen3.5 的 Gated Delta Network（GDN）在每个 decode step 读取 Q/K/V、更新 recurrent state，并写出 output。

冻结的模型 workload 是：

| 维度 | 值 |
| --- | --- |
| 模型 | Qwen3.5-122B-A10B-FP8 |
| GPU | 2×B200，TP2 |
| Batch | 32 |
| Cached / new / output | 32,768 / 1 / 128 |
| GDN | T=1，Q/K heads=8，V heads=32，D=128 |
| State / dt_bias | BF16 |
| Graph | 完整 128-step CUDA Graph |
| Target calls | 每 rank、每 graph 4,608 次 |

单次 kernel 只有约 9 微秒。看起来微不足道，但：

$$
4608\ \mathrm{calls/graph}
\times
0.33\ \mu s/\mathrm{call}
\approx
1.52\ \mathrm{ms/graph}
$$

当一个小 kernel 被调用几千次，几百纳秒的 saving 才可能穿透到完整图。

---

## 3. 修改前：为什么 16-bit 标量访存很浪费？

BF16 元素是 16 bit，也就是 2 byte。原 kernel 中存在大量类似：

```text
load q[0]
load q[1]
load q[2]
...
load q[15]
```

如果地址连续、对齐和 predicate 允许，16 个 BF16 完全可以打包成少量更宽的事务。

Round93 的 PTX rewrite 只改变访存宽度：

```text
Q/K：16 个 BF16 scalar load → 2 个 128-bit load
V：  16 个 BF16 scalar load → 4 个 64-bit load
Out：16 个 BF16 scalar store → 4 个 64-bit store
```

![标量 BF16 I/O 与 64/128-bit Vector I/O](/assets/blog-subcuda-vector-io.svg)

*图 2：总 payload 基本不变，变化的是指令条数、sector 利用率和 LSU/MIO 压力。*

最容易误解的一点是：vector load 不一定减少 DRAM bytes。这里读取的 Q/K/V 数据没有减少，数学也没有减少。

真正的收益来自：

- 更少的 global load/store 指令；
- 更高的单指令 payload；
- 更少的 excessive sectors；
- 更低的 MIO/LG throttle；
- 更短的地址和访存指令依赖链。

---

## 4. SASS 里到底发生了什么？

Round93 的最终 SASS inventory 是：

| 指令类别 | Control | PTX Candidate |
| --- | ---: | ---: |
| `LDG.E.U16` | 35 | 3 |
| `LDG.E.64` | 0 | 4 |
| `LDG.E.128` | 16 | 18 |
| `STG.E.U16` | 16 | 0 |
| `STG.E.64` | 0 | 4 |
| Registers / thread | 64 | 64 |
| Barrier | 1 | 1 |
| Stack / local / spill | 0 | 0 |

NCU 观察到 dynamic global load/store 指令分别减少 `40.79%` 和 `37.5%`，excessive sectors 减少约 `90%`。

两臂寄存器、barrier、shared memory 和 spill 都不变，这很重要。否则我们无法判断收益来自 vector I/O，还是来自资源形态变化。

因果链可以写成：

$$
\text{wider I/O}
\Rightarrow
\text{fewer LSU instructions / wasted sectors}
\Rightarrow
\text{lower MIO pressure}
\Rightarrow
\text{shorter operator latency}
$$

---

## 5. 为什么需要“零修改 PTX 重汇编”对照？

Direct PTX 实验会引入一条新的构建/加载路径：

```text
dump PTX
    ↓
重新运行 ptxas
    ↓
生成 cubin
    ↓
通过 Driver API 加载
```

即使文本一个字符都不改，重新汇编也可能因为：

- ptxas 版本不同；
- arch flag 不同；
- launcher 不同；
- JIT cache 状态不同；
- cubin metadata 不同；

产生性能变化。

所以 control 不是“原 FlashInfer 对 candidate”两臂就够了，而是需要：

1. public/source incumbent；
2. 零修改 PTX 重汇编 cubin；
3. 真正修改后的 candidate cubin。

Round93 中零修改 cubin 与 incumbent 的偏差小于 `0.10%` control gate，SASS identity 也被校验。只有这样，后面的 `0.333160 μs` 才能归因给 PTX rewrite，而不是工具链。

这是一条非常通用的二进制实验原则：

> **先证明新实验载体本身是中性的，再测载体里的修改。**

---

## 6. 为什么正确性必须同时检查 Output 和 State？

GDN 是 recurrent kernel。当前输出正确，不代表写回的 state 正确；state 中一个 bit 的偏差可能在几十步后放大。

因此 operator gate 同时覆盖：

- random input，1 step；
- finite-edge input，1 step；
- random input，128 steps；
- finite-edge input，128 steps；
- output bytes；
- updated-state bytes。

所有 control / zero-mod / PTX candidate 的 output 和 state 都 byte-exact。

完整 TP2 还冻结双 rank token hash：

```text
5880be0e6bbfde85eb76a0417c3f08707eed83ab0a2c22a8b83a62e992d847e1
```

五对 baseline/candidate、两个 rank 和所有 repeats 都必须命中同一 hash。

这比只比较最终一个 token 更严格，因为它覆盖完整 128-step trajectory 与 rank 一致性。

---

## 7. Operator 快 3.60%，为什么模型只快 0.50%？

强 operator A/B：

| Arm | Median latency |
| --- | ---: |
| Zero-mod control | 9.242280 μs/call |
| Vector-I/O PTX | 8.909120 μs/call |
| Reduction | **3.60474%** |

完整 TP2 五对 no-profiler：

| Arm | Median throughput | Median 128-step wall |
| --- | ---: | ---: |
| Baseline | 3529.435965 token/s | 1160.525376 ms |
| PTX candidate | 3547.037401 token/s | 1154.766510 ms |
| Change | **+0.498704%** | **−5.758866 ms** |

这是典型 Amdahl 稀释。GDN 只是完整 122B 模型图的一部分；其他 attention、MoE、communication 和 control 节点没有变化。

但这里还有一个反常点：单调用 saving 乘 4,608 次只预测约 `1.28–1.54 ms`，实测 wall 却少了 `5.76 ms`。

这个差异被诚实地保留为未完全归因，而不是用故事填满。可能涉及 graph 内调度、rank interaction、cache 或测量窗口，但没有 bounded Systems 证据就不能选择一个解释。

因此 Round93 是 qualified opt-in，而不是默认开启。

---

## 8. 最关键的反事实：不用 PTX 的 CUDA C++ 能做到多少？

Round95 写了一个 shape-specialized、无 inline assembly 的完整 CUDA C++ GDN。它使用普通 `uint4/uint2` 宽访存和 SM100 `__ffma2_rn`，目标是复现同一硬件机制。

三臂 no-profiler 结果：

![Source control、CUDA C++ 与 Direct PTX 三臂对比](/assets/blog-subcuda-three-arm.svg)

*图 3：CUDA 已经恢复绝大部分收益；PTX 的剩余优势属于当前 lowering/schedule gap，而非语言级不可能。*

| TP2 Arm | Median throughput | Median wall |
| --- | ---: | ---: |
| Source control | 3545.015342 token/s | 1155.425183 ms |
| CUDA C++ | 3558.984428 token/s | 1150.890118 ms |
| Direct PTX | 3560.508964 token/s | 1150.397329 ms |

CUDA 相对 control 提高 `0.394049%`，节省 `4.535065 ms`；PTX 提高 `0.437054%`，节省 `5.027854 ms`。

因此 CUDA 恢复：

$$
\frac{4.535065}{5.027854}
\approx
90.20\%
$$

的 graph-wall saving。PTX 相对 CUDA 只多节省 `0.492789 ms`，吞吐只高 `0.042836%`。

所以三个不同命题必须分开：

| 命题 | 结论 |
| --- | --- |
| CUDA 能否表达宽 vector I/O 与 packed FMA？ | **能** |
| 当前测试的 CUDA 是否和 PTX 一样快？ | **没有，PTX 仍略快** |
| 是否证明任何 CUDA 程序永远追不上？ | **完全没有** |

---

## 9. 相同访存指令，PTX 为什么仍快一点？

Round95 的 CUDA 与 PTX 都生成：

```text
LDG.E.128 / LDG.E.64 / LDG.E.U16 = 18 / 4 / 3
STG.E.128 / STG.E.64 / STG.E.U16 = 16 / 4 / 0
FFMA2 = 128
```

动态 global load/store 和 sector 数也相同。但资源与 schedule 仍有差异：

| 指标 | CUDA C++ | PTX |
| --- | ---: | ---: |
| Static SASS instructions | 1464 | 1416 |
| Registers / thread | 70 | 64 |
| Spill | 0 | 0 |
| NCU instrumented duration | 14.432 μs | 14.208 μs |
| Long scoreboard ratio | 7.128 | 6.677 |
| Barrier stall | 1.722 | 1.206 |
| Wait stall | 1.158 | 1.067 |

PTX 的当前优势更像：

- 更短 live range；
- 更少静态指令；
- 更好的 phase ordering；
- 更低的 barrier/wait/scoreboard stall。

这正是 PTX 适合作为显微镜的地方。它让我们看到“算法和访存已经相同，剩下的是编译器调度与生命周期”。下一步应继续在 CUDA 源码中缩短 live range、表达依赖，而不是宣布 PTX 拥有独占能力。

---

## 10. 为什么强压寄存器反而会更慢？

一个早期 CUDA variant 直接用 `min_blocks_per_sm=8` 逼 ptxas 将寄存器压到 64，但没有先缩短源码 live range。结果产生：

- 12 B spill stores；
- 36 B spill loads；
- latency 退化到 `9.644480 μs`。

后来先把 Q/K 改成 pair-streamed dataflow，再请求 64-register contract，才做到无 spill 并达到 `9.056520 μs`。

正确顺序是：

```text
先缩短 live range / 改数据流
        ↓
再给编译器 occupancy / register contract
```

不是：

```text
先设置更小 max register
        ↓
希望编译器自动变快
```

寄存器数量不是越少越好。它和 occupancy、ILP、spill 形成三方权衡。

---

## 11. PTX 应该怎样进入工程流程？

一个证据驱动的 SubCUDA 流程可以写成：

```text
冻结 workload / baseline / correctness / stop rule
        ↓
零修改 PTX→cubin control
        ↓
提出一个单机制 PTX 假设
        ↓
确认 SASS 真的改变
        ↓
operator output + state gate
        ↓
无 profiler operator A/B
        ↓
完整模型 paired E2E
        ↓
NSYS / NCU 解释
        ↓
尝试翻译回 CUDA / Triton
```

PTX 最适合做三件事：

1. **机制探针**：强制 load width、rounding、barrier 或 pipeline mapping；
2. **归因工具**：在不改算法的前提下隔离编译器决策；
3. **临时最优载体**：当源码暂时无法稳定得到目标 SASS 时，提供 opt-in cubin。

它不适合成为无边界的默认方案：

- 对编译器版本和 PTX 指纹敏感；
- shape/stride/ABI gate 必须非常窄；
- 维护者难以从高层语义审查；
- 新 GPU 可能让旧 schedule 反转；
- 一个错误 predicate 或 store 时点可能只在长 trajectory 才暴露。

---

## 12. 面对一个新 PTX 候选，可以直接检查什么？

### 构建控制

- 零修改 PTX/cubin 是否与生产 SASS 和 timing 等价？
- ptxas、arch、register flag 和 launcher 是否完全记录？
- rewrite 是否有唯一 fingerprint 和 SHA gate？

### 机器机制

- 最终 SASS 的目标指令真的出现了吗？
- registers、spill、stack、shared memory 是否改变？
- load/store width、dynamic instructions、sector 和 bytes 如何变化？

### 正确性

- 除 output 外，是否有 state、cache 或其他 side effect？
- 是否覆盖 random、finite-edge、边界 predicate 和多 step？
- 模型是否保存完整 token/rank hash？

### 性能

- operator A/B 是否乱序、预热并超过 materiality floor？
- E2E 是否以无 profiler wall/throughput 为权威？
- profiler 是否只做解释？
- 局部 saving 能解释多少 wall saving，剩余是否被诚实标成未知？

### 可维护性

- CUDA/Triton 能否表达同一机制？
- PTX 的残余优势是否大到值得独立 dispatch？
- unsupported shape 是否 fail closed？

---

## 结语：把 PTX 当显微镜，而不是魔法棒

Round93 证明 Direct PTX 可以带来真实模型级收益：operator 快 `3.60%`，TP2 吞吐提高 `0.498704%`，五对全胜且 token/state exact。

Round95 又证明大部分收益不是 PTX-exclusive：CUDA C++ 复现相同宽访存和 packed FMA，并恢复 `90.20%` 的完整 graph-wall saving。

两者合在一起，比“PTX 赢了”更有价值：

```text
PTX 暴露机制
    ↓
SASS 证明机制存在
    ↓
Operator 证明局部收益
    ↓
E2E 证明系统价值
    ↓
CUDA counterfactual 测量可维护载体能恢复多少
```

如果 CUDA 已经恢复 90%，剩余 0.49 ms/graph 是否值得一套 PTX runtime，要由产品边界、维护成本和复现稳定性决定，而不是由“PTX 更底层”决定。

真正成熟的底层优化不是永远停留在更低层，而是用更低层看清问题，再尽可能把知识带回更高层。
