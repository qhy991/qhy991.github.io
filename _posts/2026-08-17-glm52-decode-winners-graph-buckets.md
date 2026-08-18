---
layout: post
title: "为什么三个 Kernel Winner 组合起来只快 5.66%？从 CUDA Graph、M Bucket 到 GLM-5.2 Decode"
date: 2026-08-17 03:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [GLM-5.2, Decode, CUDA Graph, FlashMLA, DeepGEMM, MoE, B300]
reading_time: 28
cover_image: /assets/blog-glm52-decode-winners.png
excerpt: "在 8×B300 的 GLM-5.2 decode 中，FlashMLA、fixed-NK projection 和 MoE M-tile alignment 组合把 BS128 的 median ITL 从 33.116 ms 降到 31.243 ms。本文解释为什么 winner 必须绑定 phase、local M、ABI 与 CUDA Graph，以及这组 5.66% 结果还缺少哪些 promotion 证据。"
---

> 本文基于 B300-M2 上保留的 GLM-5.2 decode raw JSONL、path-hit counter、N=100 汇总和干净实验分支进行教学化整理。重点是抽取可迁移机制，而不是把本地实验目录改写成报告。当前这组证据尚未达到 deployment-wide promotion 标准，文中会明确区分“测到了什么”和“还不能说什么”。

假设你手里有三个 microbenchmark winner：

- 一个 attention kernel 在目标 shape 上快约 27%；
- 一个 projection GEMM 快约 40%；
- 一组 MoE masked GEMM 通过缩小 M tile 快约 14%。

把它们全部接进模型，端到端是不是应该快几十个百分点？

答案通常是否定的。

在一次 8×B300、GLM-5.2-FP8 的 decode 实验中，真正保留下来的 winners stack 包含：

1. FlashMLA sparse decode 的 P1+c2 组合；
2. `o_proj` 的 fixed-N/K graph-only candidate；
3. `index_q_upproj` 的 fixed-N/K graph-only candidate；
4. MoE gate/up/down 的 expected-M-aware M-tile alignment。

在 global BS=128、KV length=32,768、output length=48 的 100 次运行中，median ITL 从：

$$
33.116\ \mathrm{ms}
\longrightarrow
31.243\ \mathrm{ms}
$$

下降 `1.873 ms`，也就是 `5.66%`。

这个结果并不小，但它远低于某些单 kernel 百分比。原因不是“优化互相抵消”这么简单，而是：

> **一个 kernel 只有在正确的 phase、shape、ABI、执行模式和真实关键路径上，才是 winner。离开这些坐标，“winner”这个词没有完整含义。**

---

## 1. 先定义指标：这里的 TPOT / ITL 到底怎么算？

实验记录的每次运行包含：

- `latency`：完整调用耗时；
- `last_ttft`：最后一个请求得到首 token 的时刻；
- `output_len=48`。

归档使用的 ITL 定义是：

$$
\mathrm{ITL}
=
\frac{\mathrm{latency}-\mathrm{last\_ttft}}
{\mathrm{output\_len}}
\times 1000\ \mathrm{ms}
$$

因此，这篇文章讨论的是首 token 之后的 decode 区间，不是 TTFT，也不是在线服务队列延迟。

冻结的主要 cell 是：

| 维度 | 值 |
| --- | --- |
| GPU | 8×B300 |
| 模型 | GLM-5.2 FP8 |
| 并行 | TP8 / DP8 / EP8，DP attention |
| KV / input length | 32,768 |
| Global batch | 128 |
| Local decode M | 16 |
| Output length | 48 |
| Baseline | `SGLANG_GLM52_OPT=0` |
| Candidate | FlashMLA P1+c2 + two fixed-NK projections + MoE alignment |
| 样本 | OPT0 100 次，winners 100 次 |

这里有一个非常容易混淆的换算：

$$
M_{\mathrm{local}}
=
\frac{\mathrm{global\ batch}}{\mathrm{DP}}
=
\frac{128}{8}
=16
$$

Kernel registry 选择的是 local `M=16` bucket，而不是 global BS=128。把 global batch 直接拿去查 kernel table，会命中完全错误的实现。

---

## 2. Winner 不是一个名字，而是一个多维坐标

把一个候选写成“`o_proj` 更快”远远不够。一个可执行的 winner 至少需要下面这些坐标：

$$
W = f(
\mathrm{op},
\mathrm{phase},
M,N,K,
\mathrm{dtype},
\mathrm{ABI},
\mathrm{graph/eager},
\mathrm{backend},
\mathrm{GPU}
)
$$

![Kernel winner 的多维选择坐标](/assets/blog-glm52-decode-oracle.svg)

*图 1：只保存“哪个 kernel 最快”是不够的；必须保存它在哪个执行坐标里最快。*

例如同一个 `o_proj`：

- 在 decode M16 上，fixed-N/K candidate 的 leaf graph 可能快 `1.39–1.44×`；
- 到 M32，优势可能只剩 `1.06–1.08×`；
- 到 prefill 大 M，另一套 tile 和 pipeline 才合理；
- 如果输入 scale layout 与生产 ABI 不同，在线 adapter 的成本可能吃掉全部收益；
- 如果 candidate 只在 eager 快、graph replay 打平，它不能成为 graph-mode winner。

所以，优化 registry 的 key 不应该只是 `op_name`，而应该接近：

```text
(op, phase, local_M, N, K, dtype, scale_layout, execution_mode, backend, arch)
```

范围外应回到 stock，范围内如果 candidate launch 失败则应暴露错误；不能把运行时失败后静默 fallback 的结果计为 winner。

---

## 3. 这组 winners stack 究竟包含什么？

![GLM-5.2 decode winners stack](/assets/blog-glm52-decode-stack.svg)

*图 2：四类修改覆盖 attention、projection 和 MoE，但它们仍只是完整 decode 关键路径的一部分。*

### 3.1 FlashMLA sparse decode：P1+c2

这条路径处理稀疏 MLA decode。P1 是 main kernel 方案，c2 是 combine 阶段的配套优化。

归档中的 leaf graph 结果显示，P1+c2 相对 stock 在 M16 大约有 `1.27×`，M32 大约 `1.14×`。但“更激进”的 r2a 虽然在某个单 bucket 上更快，却会在 serving CUDA Graph 的 KV shape 下失败，因此没有进入最终 winners stack。

这说明 graph compatibility 是 correctness contract 的一部分：

> **能在独立 harness 里运行，不等于能被真实 graph capture 和 replay。**

### 3.2 `o_proj`：只固定 N/K，不固定 M

DeepGEMM 的 `compiled_dims="nk"` 让编译器把 N/K 相关控制和选择静态化，同时保留 M16/M32 两个 decode bucket。

它没有改变 FP8 输入、packed UE8M0 scale、BF16 输出或上层 callsite。真正的 candidate 变化是更窄的编译期形状合同。

这是一种“部分求值”思路：部署中不变的 N/K 在 graph capture 前固定，request 动态的 M 仍由有限 bucket 表达。

### 3.3 `index_q_upproj`：不要被相似名字骗了

Indexer 的 `wq_b` 与 attention Q-B projection 不是同一个算子。它们可以有不同 N/K、不同并行方式和不同关键路径位置。

在 M16 上，fixed-N/K `index_q_upproj` 的 leaf graph 大约快 `1.22×`；M32 约 `1.19×`。只有正确的 prefix→logical-op 映射和 path-hit counter 才能证明 serving 真的调用了这一项。

### 3.4 MoE M-tile alignment：减少小 M 的巨大 Padding

DeepGEMM masked grouped GEMM 的 stock M alignment 是 128。decode 时每个 expert 的 `expected_m` 往往只有个位数：

```text
expected_m = 5 或 9
stock M tile = 128
```

这意味着 A load、UMMA 和 epilogue store 都可能围绕远大于有效行数的 tile 工作。

实测 crossover 形成了一个有边界的策略：

| expected M | 选择的 alignment | 原因 |
| ---: | ---: | --- |
| `1–12` | 16 | 小 M，减少 Padding |
| `13–40` | 32 | 中等 M，平衡 Padding 与 tile 效率 |
| `>40` | stock 128 | 小 tile 已开始明显回退 |

对 `expected_m=5/9`，alignment 16 的 leaf 约快 `1.13–1.14×`。但 `expected_m=65` 时它可能只剩 stock 的 `0.75×`。

更危险的是，这个 DeepGEMM knob 是进程全局状态。如果小 alignment 泄漏到 prefill，M=1024 的 grouped GEMM 曾从约 `115.7 μs` 退化到 `354.3–478.7 μs`。

因此实现必须：

```text
读取旧 alignment
    ↓
只在目标 decode 调用前设置 16/32
    ↓
执行 masked grouped GEMM
    ↓
finally 中无条件恢复 128
```

这不是“调一个参数”，而是在保护跨 phase 的全局不变量。

---

## 4. 为什么 leaf 百分比不能直接相加？

假设四个局部节点的加速分别是 $s_1,s_2,s_3,s_4$。端到端时间并不是：

$$
\mathrm{speedup}_{\mathrm{E2E}}
\neq
s_1+s_2+s_3+s_4
$$

更接近 Amdahl 形式：

$$
T' = T_{\mathrm{unmodified}}
+ \sum_i \frac{T_i}{s_i}
+ T_{\mathrm{new\ overhead}}
$$

其中 `unmodified` 仍然包括：

- DeepEP dispatch/combine；
- NCCL/TP/DP collectives；
- 其他 projections、router、quant、norm；
- KV/page/index control；
- CUDA Graph 中未替换的节点；
- 最慢 rank 和同步尾部。

旧的全模型 profile 曾显示，通信在 decode-oriented capture 中可以占 GPU kernel 时间约一半。局部 projection 再快，也只能缩短自己覆盖的那一段。

此外，并发和 graph 会让节点相互影响。一个 kernel duration 变短，不一定等比例缩短从本轮 decode 输入 ready 到下一 token commit 的完整 envelope。

---

## 5. 为什么 `fused_qkv_a_proj` 被移出了 winners？

`fused_qkv_a_proj` 的 leaf graph 在 M16/M32 也曾有 `1.28–1.35×` 的漂亮结果，但旧的端到端测试接近打平、噪声大，因此最终组合主动排除了它。

这体现了一个成熟 registry 与“全开优化包”的差别：

```text
leaf winner
    ≠
region winner
    ≠
full decode winner
```

如果一个 candidate 没有稳定缩短完整关键路径，就不应该因为“都已经写好了”而进入默认组合。

同样被排除的还有：

- prefill-only DSA candidate；
- FlashMLA r2a；
- `q_b_proj`、`index_k_proj`、`index_score` 中没有生产路径收益或参照系不公平的候选。

负结果不是浪费。它们定义了 oracle table 的空白区域，阻止错误候选在更大 workload 上静默接管。

---

## 6. Path-hit 为什么是性能证据的一部分？

实验目录保存了 winners arm 的 hit/miss counter。目标 hit 包括：

```text
fp8_gemm/fixed_nk:o_proj:decode:m16
fp8_gemm/fixed_nk:index_q_upproj:decode:m16
hotspot_plugin/flashmla_sparse_decode:dsa_decode_attn:decode:m16
moe_masked:moe_gate_proj:decode:m*
moe_masked:moe_down_proj:decode:m*
```

同时，很多不在白名单的 shape 会明确记录 `no_spec` miss，并回到生产 stock。

为什么普通启动日志不够？因为“provider 加载成功”只说明候选可用，不说明某次请求真的命中它。CUDA Graph 还会让 Python 调用次数、NVTX range 数和实际 replay 次数不再一一对应。

一个可信的 path contract 至少要有：

1. 启动时记录启用的 profile、op 白名单和 M buckets；
2. 每个候选首次命中时记录精确 `(op, phase, M)`；
3. 范围外记录 miss 原因；
4. 正式 timing 关闭会污染基线的重型观测；
5. profiler 版本单独用 NVTX / kernel symbol 对齐 graph node。

---

## 7. N=100 数据究竟证明了什么？

归档中的主要汇总是：

| Arm | n | Mean ITL | Median ITL | Stdev | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OPT0 | 100 | 33.033 ms | 33.116 ms | 0.693 | 32.947 | 33.253 |
| Winners | 100 | 31.206 ms | 31.243 ms | 0.164 | 31.103 | 31.354 |

![Decode winners 的结果与证据边界](/assets/blog-glm52-decode-evidence.svg)

*图 3：BS128 的性能信号清晰，但它仍是 bounded development evidence，不是完整 deployment promotion。*

从这些数字可以安全地说：

- 在该 author-side BS128 cell 上，winners 的 median ITL 比 OPT0 低 `5.66%`；
- 100 次样本都完成，cache hit rate 约 `0.998`；
- winners 的 ITL 分布在本轮更集中；
- hit counter 证明目标候选在 serving 中被选择。

但还不能说：

- BS256 同样更快：正式 N=30 没有完成，旧结果只有 2–3 次且缺完整对照；
- 所有 decode batch/KV 都更快；
- 四个组件各自贡献多少；
- 已满足完整 correctness promotion：该性能目录没有独立 exact-token/logprob artifact；
- 已通过严格交错 paired A/B：本轮是每个 label 启一个 server、各跑 100 次，不是逐对 AB/BA fresh-server 交错。

因此最准确的裁决是：

> **这是路径命中明确、样本量充分、方向清晰的 BS128 development result；它值得写成机制文章和下一轮复验基线，但还不应升级成 deployment-wide accepted claim。**

---

## 8. 为什么 BS256 不能从 BS128 外推？

global BS 从 128 变成 256，local M 从 16 变成 32。看起来只是翻倍，但多个 crossover 同时变化：

- `o_proj` fixed-N/K 的相对收益明显变小；
- FlashMLA 的 split/combine 选择可能改变；
- MoE `expected_m` 从约 5/9 向更大 bucket 移动；
- 每个 rank 的 expert load 分布和最慢 rank 可能变化；
- graph bucket、resident CTA 和通信占比也会变化。

本轮 BS256 的 3 次示例中，winners median 约 `34.7 ms`，但没有相同协议下的完整 OPT0 N=30 对照。没有对照就没有加速比。

“M32 也注册了 candidate”只说明它通过过某层 admission，不说明它已经通过本轮端到端裁决。

---

## 9. 从这组实验抽象出的七条原则

### 原则一：Oracle key 必须包含执行模式

eager winner 和 graph winner 是两个候选。Graph capture/replay 的节点、shape 和初始化边界会改变真实成本。

### 原则二：Global batch 与 local kernel M 必须分开记录

多 GPU 下 registry 看的是 rank-local shape。只记录 BS128 而不记录 DP8/local-M16，无法复现选择逻辑。

### 原则三：生产 ABI 是性能合同

FP8 code、packed UE8M0 scale、stride、layout 和输出 ownership 都属于候选输入。在线 unpack/adapter 税不能从 microbenchmark 中消失。

### 原则四：全局状态必须有作用域和恢复协议

MoE alignment 对 decode 是优化，对 prefill 可能是 3–4× 退化。`try/finally` 恢复不是代码风格，而是性能正确性。

### 原则五：Path-hit 与数值正确性同样先于计时

没有 hit marker 的快结果可能根本没跑 candidate；只有 hit 没有 exact output，也不能 promotion。

### 原则六：组合 winner 要重新测整条路径

单项 leaf win 不能相加。组合会改变 cache、stream、graph topology 和最慢 rank。

### 原则七：缺失的证据要直接写出来

BS256 不完整、没有独立 correctness artifact、没有严格 paired fresh-server 顺序，都应成为下一步任务，而不是被一句“结果稳定”遮住。

---

## 10. 下一轮怎样把 development result 升级为正式结论？

一个更严格的复验顺序可以是：

1. 冻结当前代码 revision、镜像、模型/量化资产、driver/CUDA 和 graph 配置；
2. 为 OPT0 和 winners 使用同一组输入与 sampling seed，做 baseline-repeat 噪声测量；
3. 记录 exact token trajectory，并比较 selected-token logprob；
4. 对 BS128 做至少 5 组 fresh-server AB/BA 交错 screen；
5. 若通过，再做 25 对或等价置信规则；
6. 用独立请求 seed 复验；
7. BS256 建立独立 cell，而不是复用 BS128 百分比；
8. 最后做 matched Nsys，确认关键路径变化和所有 rank 的进度；
9. 只对 Systems 选出的代表 kernel 使用 NCU；
10. 保留 raw order、path hits、correctness 和 profile，不只保留 summary。

如果正式 paired 结果没有复现 5.66%，这并不让当前分析失效。它会告诉我们：旧结果中还有 server 顺序、机器状态或未冻结变量的贡献。

---

## 结语：Winner 是一个带坐标的事实

这次 decode 实验最值得带走的，不是“把四个优化都打开”。真正可迁移的是下面这条规则：

```text
一个候选什么时候成立？
    ↓
哪个 op、哪个 phase、哪个 local M？
    ↓
什么 N/K、dtype、scale ABI 和 backend？
    ↓
eager 还是 CUDA Graph？
    ↓
是否真的 path hit？
    ↓
是否保持数值与完整 token trajectory？
    ↓
是否缩短完整关键路径？
```

只有这一串问题都有答案，“winner”才从一个漂亮的 microbenchmark 数字，变成可以放进推理系统的工程事实。

而 `33.116→31.243 ms` 最有价值的地方，正是它把这套坐标暴露了出来：局部 kernel 的确能赢，但系统只会为落在正确执行图上的那部分收益买单。
