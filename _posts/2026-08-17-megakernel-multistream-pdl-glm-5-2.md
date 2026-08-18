---
layout: post
title: "从 80 μs 到 36.992 μs：Megakernel、Multistream 和 PDL，究竟谁让 GLM-5.2 跑得更快？"
date: 2026-08-17 01:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [Megakernel, CUDA, Multistream, PDL, GPU Kernel, GLM-5.2, Inference]
reading_time: 32
cover_image: /assets/blog-megakernel-multistream-pdl.png
excerpt: "以 GLM-5.2-FP8 decode 的 attention-pre 子图为例，拆解 Megakernel、Multistream 与 PDL 分别解决什么问题，以及为什么关键路径上的 CUDA Graph + Multistream + PDL 最终达到 36.992 μs。"
---

> 一篇面向初学者的系统拆解：为什么 kernel fusion 做完以后，GPU 仍然有一半以上的 SM 没活干？Megakernel 到底比普通 CUDA kernel 多了什么能力？为什么最后反而是 CUDA Graph + Multistream + PDL 更快？

> **资料与版权说明**：本文基于知乎作者「是小肖啊」的原文进行结构化整理与教学式重写，实验数据与配图来自原文。原文链接：[Megakernel 实战分析](https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125)。商业转载请联系原作者授权，非商业转载请保留作者与来源。

---

## 先看结论

这次优化针对 GLM-5.2-FP8 decode 阶段的一段 attention-pre 子图：从 `Add + RMSNorm + Quant` 开始，到 `Paged Indexer Score` 结束，不包含后续 TopK 和 Sparse FlashMLA。实验运行在一张 B300 上，一次处理两个请求，每个请求 8 个 token，因此：

```text
M = 16
KV = 8192
GPU = 1 × B300，148 SM
```

性能经历了下面几步：

| 实现方式                                  |             计时口径 |          延迟 |
| ------------------------------------- | ---------------: | ----------: |
| 最初约 30 个短 kernel                      |         近似整段执行时间 |   约 `80 μs` |
| Kernel fusion 后，11 个 kernel 单流执行      |         GPU 执行区间 | `61.631 μs` |
| MegaRTP Megakernel                    |           完整实际运行 | `44.272 μs` |
| MegaRTP 扣除固定框架后的诊断值                   |         不是实际调用延迟 | `39.601 μs` |
| CUDA Graph + Multistream + PDL，简称 CMP |         GPU 执行区间 | `36.992 μs` |
| CMP                                   | Graph replay p50 | `40.960 μs` |

这里的 `p50` 指多次测量结果的中位数。

最容易误读的地方是：

> **MegaRTP 确实打破了 kernel 边界，但它在这个工作负载上的主要收益，并不来自 Megakernel 独有的 tile 级生产者—消费者流水，而是来自“独立分支并行”和“下游 kernel 提前准备”。前者可以用 Multistream 实现，后者可以用 PDL 实现。**

因此，可以把这次实验总结成下面的公式：

$$
\begin{aligned}
\text{Megakernel 的潜在收益}
={}&\text{独立分支并行}
+\text{依赖任务提前准备}
+\text{tile 级流水}\\
&-\text{固定框架开销}
-\text{统一执行模型的约束}
\end{aligned}
$$

在本例中，前两项很大，第三项在关键路径上接近于零，而后两项又不可忽略，所以普通高性能 kernel 配合 CMP 调度最终更快。

---

## 1. 阅读前先理解七个 CUDA 概念

如果不熟悉 CUDA，可以先把 GPU 想象成一座有很多车间的工厂。

| CUDA 概念                | 初学者可以怎样理解                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| **SM**                 | 一个可以独立接活的计算车间。B300 有 148 个 SM。                                                                  |
| **CTA / Thread Block** | 一支被派往某个 SM 执行的工作队。本文基本可以把 CTA 和 thread block 看成同一个概念。                                           |
| **Grid**               | 一次 kernel 启动提交的全部 CTA。比如一个 kernel 只有 12 个 CTA，它的 grid 大小就是 12。                                  |
| **Kernel**             | 一种在 GPU 上批量执行的计算任务。每次启动 kernel，都会提交一个 grid。                                                     |
| **Stream**             | GPU 的任务队列。同一条 stream 中的 kernel 必须按顺序执行。不同 stream 的任务在依赖允许时可以并发。                                 |
| **Tile**               | 一个 CTA 负责的一小块数据。例如 GEMM 的一个输出矩阵块，或者 RMSNorm 的一行。                                                |
| **Wave**               | GPU 同一时间能驻留和推进的一批 CTA。Grid 太大时会分成多轮 wave；如果一个 producer 只有一个 wave，第一块输出 ready 往往与整个 grid 完成相差不大。 |

还需要理解两个容易混淆的词。

### 1.1 Kernel fusion 是什么？

Kernel fusion 是把相邻的计算合并到同一个 kernel 中。例如原本需要：

```text
RMSNorm kernel
    ↓
Quant kernel
    ↓
Cache write kernel
```

融合以后，可以变成一个 kernel：

```text
RMSNorm + Quant + Cache write
```

这样可以减少 kernel 启动次数，也可以避免部分中间结果写回显存后再读回来。

但 fusion 之后，**这个融合 kernel 仍然是一个完整 grid**。它没有结束以前，同一条 stream 上后面的 kernel 仍然不能开始。

### 1.2 Megakernel 又是什么？

Megakernel 不只是把几段代码拼进同一个大函数。本文中的 MegaRTP 更接近一个运行在 GPU 内部的小型任务执行器：

1. Host 把不同 Op 拆成 tile task；
2. 每个 tile 被编码成一条 instruction；
3. 一个长期驻留的 persistent kernel 启动固定数量的 worker CTA；
4. worker CTA 从预先生成的队列中不断取 instruction；
5. 不同 Op 的 tile 可以在同一个 kernel 内交错执行。

如果把普通 CUDA kernel 看成“每来一批货就临时开一座工厂”，那么 persistent Megakernel 更像“工厂一直不关门，车间不断从任务单中取下一件活”。

---

## 2. Kernel fusion 已经做完，为什么 GPU 仍然很闲？

最初的 attention-pre 需要大约 30 个短 kernel，经过融合后只剩 11 个，时间从约 `80 μs` 降到了 `61.631 μs`。

听起来似乎已经不错了，但查看执行时间线后，会发现一个更大的问题：这些 kernel 的 grid 普遍很小。

![小 Grid 的 CTA 数量给活跃 SM 数量设下硬上限](/assets/blog-megakernel-small-grid.svg)

*图 1（教学重绘）：即使完全消除 Kernel 间空隙，12 个 CTA 也无法在同一时刻给 148 个 SM 都提供目标工作。*

图中每个彩色横条代表一次真实 kernel 执行，条内标出了该 kernel 的 CTA 数量。例如：

```text
B1 Indexer-K projection：12 CTA
A0 Add + RMSNorm + Quant：16 CTA
D1 Indexer-Q projection：32 CTA
E1 Paged Indexer Score：128 CTA
```

B300 有 148 个 SM。即使做最乐观的估计，一个 12-CTA kernel 也最多只能让 12 个 SM 执行它的目标工作，其余 136 个 SM 没有这个 kernel 的 CTA 可做。

可以用下面的上界估计某一时刻最多有多少 SM 在工作：

$$
\text{ActiveSMUpperBound}(t)
=
\min(\text{GridCTA}(t),148)
$$

按照每个 kernel 的持续时间加权，整段执行平均最多只有：

$$
64.65 / 148 = 43.7%
$$

的 SM 在执行目标 CTA。

这里一定要注意：

> **43.7% 不是 Nsight Compute 中的 occupancy，也不是 Tensor Core 利用率，更不是实测的 SM busy。它只是根据 grid CTA 数计算出的“活跃 SM 数量乐观上限”。实际硬件利用率只可能更低。**

### 2.1 真正的问题不是 kernel launch bubble

很多人看到大量小 kernel，会首先怀疑 kernel launch overhead。

但原实验统计了相邻 kernel 之间的空白时间。20 次测量中，所有 bubble 相加的中位数只有：

```text
0.672 μs
```

约占 `61.631 μs` 的 `1.08%`。

因此，即使把 kernel 之间的空隙完全消除，也不可能获得十几、二十微秒的收益。

真正的问题是：

> **当前 kernel 用不满 GPU，而后面本来与它无依赖的 kernel，又因为被放在同一条 stream 上，不能提前进入空闲 SM。**

---

## 3. 这 11 个 kernel 根本不是一条串行链

单流时间线看起来像下面这样：

```text
A0 → C0 → B1 producer → B1 finalizer → B0 producer → ... → E1
```

但“程序按照这个顺序 launch”并不等于“数据真的必须按照这个顺序计算”。

从 tensor 依赖关系看，这 11 个 kernel 实际上有三条分支。

![A0 之后存在三条可以并发推进的数据依赖分支](/assets/blog-megakernel-three-branch-dag.svg)

*图 2（教学重绘）：单 Stream 的 launch 顺序把三条独立分支排成直线；数据 DAG 本身并不要求这种串行。*

A0 完成以后：

* QKV-A 分支可以开始；
* Head-Gate 分支可以开始；
* Indexer-K 分支也可以开始；
* 三者彼此之间没有直接依赖。

更清晰地抽象成逻辑 Op 后，可以得到下面这张图。

<p class="source-figure-note"><strong>图 3：GLM-5.2 attention-pre 的逻辑 OpGraph</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

可以把它简化成：

```text
                         ┌── D0 Main-Q ── D2 Q-nope BMM
                         │
A0 ── B0 QKV-A ──────────┤
 │                       └──┐
 ├── C0 Head-Gate ──────────┴── D1 Indexer-Q ──┐
 │                                             ├── E1 Score
 └── B1 Indexer-K ─────────────────────────────┘
```

其中：

* A0 之后发生一次 fan-out；
* B0 和 C0 在 D1 发生一次 join；
* D1 和 B1 在 E1 再发生一次 join；
* B0 还单独产生 D0 → D2 的主 Q 分支。

图中只有 8 个逻辑 Op，但普通 CUDA trace 中有 11 个 kernel。这是因为 B0、B1、D1 都被拆成了 producer 和 finalizer 两个 kernel：producer 负责主体计算，finalizer 负责归约、后处理或缓存写回。

```text
8 个逻辑 Op + 3 个额外 finalizer = 11 个 kernel
```

这一节最重要的结论是：

> **CUDA stream 表达的是执行顺序，不会自动理解 tensor 的 producer/consumer 关系。把所有 kernel 放在同一条 stream 上，相当于把一张 DAG 人为压扁成一条链。**

---

## 4. Megakernel 到底想解决什么？

Megakernel 的目标不是单纯减少 kernel 数量，而是把调度粒度从“整个 grid”下沉到“一个 tile”。

在本文的场景中，它理论上可以带来三种重叠。

### 4.1 第一类：独立分支并行

A0 完成后，B0、C0、B1 三条路径可以同时推进。

普通单流执行时，它们是：

```text
先把 B0 整个做完
再把 C0 整个做完
再把 B1 整个做完
```

Megakernel 中，不同分支的 tile instruction 可以交错分配给不同 worker：

```text
worker 0：B0 tile
worker 1：C0 tile
worker 2：B1 tile
worker 3：B0 tile
...
```

只要输入已经就绪，就可以共同占用空闲 SM。

### 4.2 第二类：有依赖的 consumer 提前准备

假设 B 依赖 A 的输出。B 的主计算当然不能在 A 的数据完成前开始，但 B 并不是所有工作都依赖 A。

在第一次读取 A 的输出以前，B 往往可以先完成：

* 读取任务描述；
* 初始化 scheduler；
* 初始化 barrier；
* 准备 TMA（Tensor Memory Accelerator）descriptor；
* 加载 B 自己的权重；
* 加载与 A 无关的 scale。

因此，Megakernel 可以做到：

```text
A 正在计算
        +
B 的 Controller / WLoader 提前工作
        ↓
A 的数据就绪
        ↓
B 的 ALoader 补上 activation
        ↓
直接进入 MMA 主循环
```

这里没有违反数据依赖。真正读取 A 输出的操作仍然需要等待，提前的只是与 A 无关的准备工作。

### 4.3 第三类：真正的 tile 级生产者—消费者流水

普通 kernel 依赖通常以整个 grid 完成为边界：

```text
A 的所有 CTA 结束
        ↓
B 才开始
```

Megakernel 可以把依赖细化为：

```text
A 完成第 0 个输出 tile
        ↓
B 立即消费第 0 个 tile

A 继续生成第 1、2、3 个 tile
```

例如，假设 1024 行 RMSNorm 后接 `blockM=128` 的 GEMM。前 128 行完成以后，GEMM 可以先开始第一个 M tile，而不需要等待其余 896 行。

这第三类能力才是 Megakernel 最难被普通 kernel 完整复现的地方。

但它有严格前提：

1. producer 的输出必须能切成 consumer 可以独立消费的小块；
2. consumer 获得第一块数据时，GPU 上还要有资源运行它；
3. producer 不能在一个 wave 内几乎全部完成，否则第一块 ready 与整个 grid 完成相差不大；
4. 这条流水必须位于最终关键路径上。

后面会看到，本例恰好不满足最关键的条件。

---

## 5. MegaRTP 不是一个“大函数”，而是一套 GPU 内任务运行时

MegaRTP 的完整 lowering 流程可以概括为：

```text
模型图定义
    ↓
OpGraph：记录 tensor def-use
    ↓
按当前 M/KV 把每个 Op 展开成 tile
    ↓
TaskGraph：生成 task → event → task 依赖
    ↓
KernelPlan：把 task 分配给 148 个 worker queue
    ↓
instruction tensor + event counters + OperandPtrTable
    ↓
一个 persistent kernel 在 GPU 上解释执行
```

### 5.1 OpGraph：先记录“谁生产、谁消费”

OpGraph 只记录 tensor 层面的依赖：

```text
A0 产生 hidden FP8/scale
B0 和 B1 消费 hidden FP8/scale
C0 消费 norm 输出
B0 与 C0 的结果汇合到 D1
D1 与 B1 的结果汇合到 E1
```

此时还没有：

* tile 数量；
* CTA 数量；
* event id；
* threshold；
* worker queue；
* CUDA pointer。

### 5.2 TaskGraph：把 Op 依赖降低成 tile counter

<p class="source-figure-note"><strong>图 4：从 OpGraph 的 tensor 依赖，降低成 TaskGraph 的 counter event</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

在 `M=16, KV=8192` 时，8 个 Op 被展开成：

| Op     | tile task 数量 | 主要切分方式                               |
| ------ | -----------: | ------------------------------------ |
| A0     |           16 | 16 行，每行一个 tile                       |
| B0     |           84 | 1 个 M tile × 21 个 N tile × Split-K 4 |
| C0     |           48 | 1 个 M tile × 12 个 N tile × Split-K 4 |
| B1     |            8 | Indexer-K Split-K 8                  |
| D1     |          128 | 1 个 M tile × 32 个 head × Split-K 4   |
| D0     |          128 | 1 个 M tile × 64 个 head × 2 个 N tile  |
| D2     |          256 | 1 个 M tile × 64 个 head × 4 个 N tile  |
| E1     |           64 | 4 个 Q-block × 16 个 KV task           |
| **总计** |      **732** | 732 条 tile instruction               |

这里的 Split-K 是把 GEMM 的 K 维拆成多个并行 partial，最后再由 finalizer 或归约步骤合并。

TaskGraph 不再只说“B0 依赖 A0”，而是描述：

```text
哪些 producer tile 写了哪块 region
哪些 consumer tile 读取哪块 region
consumer 要等待多少次 publication
```

MegaRTP 支持四类配对策略：

| 策略            | 含义                                  | consumer 怎样等待      |
| ------------- | ----------------------------------- | ------------------ |
| `whole`       | 不分析细粒度关系                            | 等整个 producer 的计数完成 |
| `tile`        | 一个 producer tile 对应一个 consumer tile | threshold 通常为 1    |
| `tile_cover`  | 多个不重叠 tile 共同覆盖 consumer 输入         | 等所有覆盖块到齐           |
| `tile_reduce` | 多个 partial 对同一结果做贡献                 | 等全部 partial 完成     |

需要强调：counter 只负责记录“有多少 producer 已完成”，并不负责数值归约本身。

本例最终生成了 70 个 event。几个典型例子是：

```text
A0 的 16 行 hidden/scale 完成：event0 >= 16
A0 的 16 行 norm 完成：event1 >= 16
B0 的最终 Q/scale 完成：event2 >= 16
C0 的 48 个 gate partial 完成：event3 >= 48
D1 的 32 个 Indexer-Q/head tile 完成：event68 >= 32
B1 的 K Cache 写回完成：event69 >= 1
```

### 5.3 KernelPlan：把 732 条任务静态分给 148 个 worker

<p class="source-figure-note"><strong>图 5：从 Op 连接关系到 148 个 resident worker</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

MegaRTP 启动：

```text
148 CTA × 384 threads
```

设计目标可以理解为每个 SM 驻留一个长期 worker CTA。

732 条 instruction 按 round-robin 分配：

* 140 个 worker 各有 5 条有效 instruction；
* 8 个 worker 各有 4 条有效 instruction；
* 较短的队列补一条 `NoOp`；
* 最终 instruction tensor 的形状为 `[148,5,32]`。

一条 instruction 有 32 个 `uint32`，共 128 B，主要保存：

```text
opcode
wait_count / signal_count
operand_row_id
wait(event_id, threshold)
signal(event_id, increment, publication_id)
tile 坐标和动态参数
padding
```

它并没有重新保存一段 CUDA 代码。`opcode + spec_id` 只是告诉 resident CTA：应该进入哪一个已经在编译期生成好的 Op family 和模板特化。

### 5.4 OperandPtrTable：任务描述和真实地址解耦

<p class="source-figure-note"><strong>图 6：多条 tile instruction 通过 operand_row_id 共享地址表</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

正式计划只有 8 个 Op，却有 732 条 instruction。同一个 Op 的几百条 tile instruction 往往读取同一组 tensor、权重和 descriptor。

如果每条 instruction 都重复保存完整的 64-bit CUDA pointer：

* instruction 会被地址占满；
* 换一批输入时需要重写整张 instruction tensor；
* 地址变化与任务拓扑变化耦合在一起。

因此，MegaRTP 使用：

```text
OperandPtrTable
shape = [num_ops,16]
dtype = uint64
```

本例大小只有：

$$
8\times16\times8\text{ B}=1024\text{ B}
$$

每条 instruction 只保存 `operand_row_id`。例如所有 E1 tile 都保存：

```text
operand_row_id = 7
```

运行时从表的第 8 行取出：

* Q 的 TMA descriptor；
* KV Cache descriptor；
* KV scale；
* head weight；
* context length；
* block table；
* logits 地址。

于是，不同类型的变化可以分层处理：

| 发生了什么变化                                      | 需要更新什么                      |
| -------------------------------------------- | --------------------------- |
| 输入、权重、KV Cache 或 workspace 地址改变              | 更新 OperandPtrTable          |
| TMA tensor 地址或 shape 改变                      | 同步重建 TMA descriptor         |
| M、batch、KV length 改变，导致 tile 数或 threshold 改变 | 重建 TaskGraph 和 worker queue |
| 模型连接关系改变                                     | 重新生成 OpGraph 之后的计划          |

这是一种很典型的 runtime 设计：**instruction 描述“做什么”，地址表描述“数据在哪里”。**

### 5.5 动态 M/KV 怎样兼顾通用性和 GEMM 性能？

MegaRTP 并不是在 Device 端临时生成 GEMM。Host 在 `OpGraph → TaskGraph` 阶段，根据本轮的权重 shape、输入 `M` 和 tile 划分选择 GEMM 配置，并把相同配置去重：

```text
GEMM-A：K=6144, N=2624 → spec 0
GEMM-B：K=6144, N=32   → spec 1
GEMM-C：K=6144, N=2624 → 复用 spec 0
```

Codegen 只为本轮需要的 `spec` 实例化对应 C++ 类型、pipeline 和 epilogue。Runtime instruction 只保存 `opcode + spec_id + tile coordinate`，Device 根据 `spec_id` 进入已经编译好的模板分支。

所以它的分工是：

* Host 决定本轮需要哪些实现；
* 编译期生成高性能 CUDA Op family；
* Device 只做轻量分派，不现场生成矩阵乘。

这使模型图、`M` 和 KV length 可以变化，同时避免把 GEMM 退化成一个完全动态、性能较差的通用实现。

---

## 6. 一个 resident CTA 内部，实际上还有一条小流水线

Device 侧的 384 个线程被固定分成多个角色：

| Warp      | 角色         | 主要工作                                      |
| --------- | ---------- | ----------------------------------------- |
| warp 0    | Controller | 读取 instruction、准备 slot、建立 page 映射、初始化同步状态 |
| warp 1    | MMAer      | 等待 A/B 数据并执行 Tensor Core 主循环              |
| warp 2    | ALoader    | 等待上游数据 event，搬运 activation                |
| warp 3    | WLoader    | 搬运与上游无关的 weight 和 scale                   |
| warp 4–11 | Epilogue   | 读取结果、转换、写回并发布下游 event                     |

这些角色不是由 Controller 依次调用的几个函数，而是同时推进的独立 role loop。每个 instruction slot 还要在共享内存中保存独立状态，原实现每个 slot 约为 `5120 B`，双 slot 共约 `10240 B`。这里的 SMEM 是常规 shared memory，TMEM 则是 Blackwell 上供 Tensor Core 流水使用的 Tensor Memory。

<p class="source-figure-note"><strong>图 7：一个 resident CTA 内，相邻两条 instruction 的跨指令流水</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

### 6.1 为什么要有两个 instruction slot？

如果只有一个 slot，Controller 必须等待 instruction `i` 的所有角色全部结束，才能准备 `i+1`。

MegaRTP 使用双 slot：

```text
slot 0：instruction i 正在执行
slot 1：Controller 正在准备 instruction i+1
```

于是，当 `i` 的 MMAer 和 Epilogue 仍在工作时：

* Controller 可以读取 `i+1`；
* WLoader 可以预取 `i+1` 的权重；
* ALoader 只在真正读取 activation 前等待上游 event。

这叫 **cross-instruction pipeline**。

它不代表两条 instruction 的完整 GEMM 同时执行，而是：

> 下一条 instruction 的控制、初始化和权重加载，与当前 instruction 的 MMA/Epilogue 发生重叠。

### 6.2 五类同步分别保护什么？

| 同步对象                   | 含义                                                       |
| ---------------------- | -------------------------------------------------------- |
| `instruction_arrived`  | Controller 已经把当前 slot 的描述、映射和 semaphore 准备好，其他 role 可以读取 |
| Op-local `mbarrier`    | CTA 内部 Loader → MMAer → Epilogue 的数据阶段交接                 |
| global event counter   | 跨 CTA、跨 Op 的 tile 数据已经就绪                                 |
| `page_finished`        | 某个 physical SMEM/TMEM page 的最后一个使用者已经释放它                 |
| `instruction_finished` | 当前 slot 的所有角色都已收尾，Controller 可以覆盖整个 slot                 |

这些同步不能互相替代。例如，global event 表示另一个 CTA 的输出已经可读，但它不代表当前 CTA 的本地 weight stage 已经填好。

### 6.3 共享内存为什么还要做“page rename”？

双 slot 只保存两份控制状态，并没有为两条 instruction 复制两整套共享内存。

实际的数据缓冲区被切成：

```text
13 个 physical page × 16 KiB
```

每个 Op 只使用 logical page id。`pid_order` 再把 logical page 映射到本轮实际使用的 physical page。

这有点像寄存器重命名：

```text
logical page 0
        ↓ pid_order
physical page 7
```

但映射到 page 7 并不等于已经可以覆盖它。下一条 instruction 的 loader 还要等待：

```text
page_finished[7]
```

只有上一条 instruction 对 page 7 的最后一次访问完成以后，下一条 instruction 才能写入。

因此：

* `pid_order` 决定复用哪一页；
* `page_finished` 决定什么时候可以复用；
* 下一条 instruction 不必等待上一条完整结束，只等待自己真正需要覆盖的 page 被释放。

---

## 7. MegaRTP 真的产生 overlap 了吗？答案是肯定的

为了观察 Megakernel 内部执行，实验版本在每条 instruction 的三个位置记录 GPU 时间戳：

```text
Visible
    WLoader 已看到 instruction，可以开始独立准备

MainInputReady
    ALoader 已完成上游 global event wait，主输入可以读取

SemanticDone
    Epilogue 完成最终写回并发布下游 event
```

<p class="source-figure-note"><strong>图 8：一个 Megakernel 内，不同 Op instruction 的重叠时间线</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

浅蓝色表示：

```text
instruction 已经可见
        ↓
主输入真正就绪
```

这段时间不一定是纯等待。Controller 可以准备任务，WLoader 可以预取静态权重。

橙色表示：

```text
主输入就绪
        ↓
语义完成
```

从图中可以清楚看到：

1. B0、C0、B1 三条独立分支在相同时间段内推进；
2. D0 和 D1 也发生了重叠；
3. 很多下游 instruction 在主输入就绪前已经 Visible；
4. resident CTA 内确实存在跨 instruction 的准备与计算重叠。

所以，MegaRTP 的 overlap 不是概念图，而是被实际时间戳观测到了。

但这里必须区分两个问题：

> **“是否发生 overlap”与“这个 overlap 是否缩短最终关键路径”不是一回事。**

这里的 **critical path（关键路径）**，就是从子图入口到出口的最长依赖链，它决定了整个子图最早何时能够结束。一个非关键分支即使与其他计算重叠了 10 μs，只要它本来就不会决定最终结束时间，端到端 latency 也可能几乎不变。

---

## 8. 整篇文章最关键的转折：M=16 让 tile 级依赖失去了关键路径窗口

MegaRTP 使用了 70 个 counter event，看起来依赖已经非常细。但 counter 的数量多，并不等于 consumer 可以很早开始有效计算。

真正要看的是：**consumer 的 threshold 到底是多少。**

本例中，主要 GEMM 满足：

```text
M = 16
blockM = 16
```

因此，每个 GEMM 在 M 方向只有一个 tile。

更麻烦的是，上游 GEMM 沿 N 方向产生的多个输出 tile，往往共同组成下游 GEMM 的完整 K 维输入。下游只有在这些数据全部到齐以后，才能计算自己的唯一一个 M tile。

实际依赖如下：

| 依赖         | consumer 等待条件                    | 实际含义                                       |
| ---------- | -------------------------------- | ------------------------------------------ |
| A0 → B0/B1 | `event0 >= 16`                   | A0 的 16 行 hidden/scale 全部完成                |
| A0 → C0    | `event1 >= 16`                   | A0 的 16 行 norm 全部完成                        |
| B0 → D0    | `event2 >= 16`                   | 组成完整输入的 16 个 Q/scale tile 全部完成             |
| B0+C0 → D1 | `event2 >= 16` 且 `event3 >= 48`  | B0 的完整 Q/scale 和 C0 的全部 48 个 partial 都完成   |
| D1+B1 → E1 | `event68 >= 32` 且 `event69 >= 1` | 32 个 Indexer-Q/head tile 和整次 K Cache 写回都完成 |

这意味着：

> 虽然同步机制是 tile counter，但关键 consumer 实际上仍在等待上游相关结果“几乎全部完成”。

这里没有第二个 M tile 可以提前交给下游。第一块 M16 数据，就是全部 M16 数据。

因此，在关键路径上：

$$
\Delta_{\text{tile-level overlap}}
\approx 0
$$

### 8.1 D0 → D2 是真实的细粒度例外

D0 会按 head 产生 event。某个 head 的两个 N tile 完成后，对应 D2 tile 就能开始，不必等待其余 63 个 head。

trace 显示：

```text
最早 D2 输入就绪：约 29.25 μs
D0 最晚完成：约 32.90 μs
```

确实形成了大约：

```text
3.6 μs
```

的逐 head overlap。

但最终最晚结束的是另一条 E1 分支，约 `41.120 μs`。D0/D2 不在最终关键路径上，所以这段真实的 tile overlap 没有转化为本例的端到端收益。

这给出了一个非常重要的性能分析原则：

> **不要只证明 overlap 存在，还要证明它位于 critical path，并且真的让最终完成时间提前。**

---

## 9. 既然主要收益不是 tile 流水，能否用普通 kernel 重建？

经过前面的分析，MegaRTP 在本例中真正影响整体时间的机会，主要收敛为两类：

1. 独立分支并行；
2. dependent consumer 在输入就绪前完成 prologue 和权重预取。

这两类能力分别可以由 Multistream 和 PDL 表达。

于是作者保留 11 个 standalone 高性能 kernel，不再把所有算子迁移进统一 resident CTA，而是构建：

```text
CUDA Graph + Multistream + Programmatic Dependent Launch
```

简称 **CMP**。

---

## 10. Multistream：让无依赖的小 grid 共同填满 GPU

CUDA stream 只保证同一条 stream 内按顺序执行。不同 stream 的 kernel 如果没有 event 依赖，就具备并发条件。

因此，可以把三条分支放到不同 stream：

```text
stream 0：A0 → B0 → D0 → D2
stream 1：     C0 → D1 → E1
stream 2：     B1 ─────────┘
```

实际依赖仍需要 event 或 Graph edge 表达，但关键点是：

> 当 B0 只有一部分 SM 可用时，C0 或 B1 的 CTA 可以进入剩余 SM，而不必等 B0 整个 grid 完成。

Multistream 不改变 kernel 内部算法，解决的是 **DAG 被单流错误串行化** 的问题。

当然，并发不是免费的。不同 kernel 会竞争：

* SM；
* Tensor Core；
* L2 Cache；
* HBM 带宽；
* TMA/内存管线。

所以“能并发”不等于“全部同时塞进去一定最快”。真正需要优化的是整张 DAG 的关键路径和资源分配。

---

## 11. PDL：把“可以进入 GPU”和“数据已经可读”拆开

Programmatic Dependent Launch 最容易被误解成“consumer 可以提前读取 producer 的输出”。实际上不是。

PDL 把依赖拆成两个时刻：

```text
时刻 1：consumer 获得提前进入 GPU 的许可
时刻 2：consumer 真正读取 producer 输出前，等待数据完成
```

Producer 的 CTA 在安全位置调用：

```cpp
cudaTriggerProgrammaticLaunchCompletion();
```

这只表示：

> 这个 producer grid 已经到达允许下游提前 launch 的位置。

Consumer 在读取 producer 输出前仍要调用：

```cpp
cudaGridDependencySynchronize();  // GDC
```

GDC 才负责：

* 等直接依赖的 producer grid 完成；
* 建立所需的内存可见性；
* 保证后续读取不会看到未完成数据。

所以 PDL 的正确执行形态是：

```text
consumer 先进入 GPU
    ↓
初始化 scheduler / barrier / descriptor
    ↓
预取与 producer 无关的权重
    ↓
GDC
    ↓
读取 producer activation
    ↓
MMA + Epilogue
```

而不是：

```text
consumer 提前读取未完成的数据   ×
```

### 11.1 为什么还要拆分 ALoader 和 WLoader？

原始 GEMM loader 如果由同一个 warp 同时搬运 activation A 和 weight B，那么它在入口执行 GDC 后，连完全独立的权重 B 也被一起挡住。

因此，CMP 路线将 loader 拆成：

```text
ALoader：GDC → A / SFA TMA ──┐
                              ├→ full barrier → MMA
WLoader：      B / SFB TMA ──┘
```

WLoader 跳过 GDC，可以提前填充权重 stage；ALoader 等 activation ready 后再补上 A/SFA，两路在 full barrier 汇合。

这实际上复现了 MegaRTP 中的第二类 overlap：

```text
下游 instruction 的权重准备
        与
上游 producer 的计算
        重叠
```

原文给出的一个偏工程化估计是：PDL 在单条 producer-consumer 边上的收益通常大约为 `0.5～2 μs`。如果只提前 launch，收益可能只有约 `0.5 μs`；能把 prologue 和权重预取移动到 GDC 之前时，收益才会增大。本例的 `7.423 μs` 是多条依赖边累计后的整图收益，不能当成一条边的收益。

### 11.2 编译器重排也可能破坏正确性

GDC 必须位于第一次读取 producer 输出之前。即使 CUDA 源码写对了，编译器也可能因为指针限定和优化，把某些 activation load 移到 GDC 前面。

因此，这类优化不能只看源代码，还需要：

* 检查最终 SASS；
* 确认 acquire 位于相关 load 之前；
* 使用污染输入验证；
* 重复执行 Graph replay，检查是否存在偶发错误。

这也是 PDL 工程化中非常容易被忽略的一点。

---

## 12. CMP 的真实时间线：普通 kernel 也能出现明显重叠

<p class="source-figure-note"><strong>图 9：Multistream + PDL 的真实 kernel 时间线与 active-SM 曲线</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

图中：

* 浅蓝色：kernel 已进入 GPU，但主输入尚未完全就绪；
* 橙色：主输入 ready 后的正常 kernel 工作；
* 下方曲线：由 CTA timestamp 和 SMID 聚合得到的目标 CTA 覆盖情况。

浅蓝色不应被直接理解成“纯等待”。在这段窗口中，kernel 可以完成：

* prologue；
* descriptor 准备；
* barrier 初始化；
* 静态权重预取；
* 部分与 activation 无关的工作。

图中可以直接看到：

1. A0 之后的 Head-Gate、Indexer-K、QKV-A 同时推进；
2. Main-Q、Q BMM、Paged Indexer Score 在输入完全 ready 前已经进入；
3. 606 个 CTA 覆盖了全部 148 个 SM；
4. 测量区间中，平均约有 `119.078` 个 SM 在执行目标 CTA，对应目标 CTA 覆盖率约 `80.46%`。

同样要注意：`80.46%` 不是 Tensor Core 利用率，也不是 Nsight Compute occupancy。它是根据 CTA 时间戳和 SMID 得到的目标工作覆盖指标。带 kernel 内打点的 trace 版本会引入约 `4.3%～4.7%` 的扰动，因此图主要用于解释执行形态，正式性能数字来自关闭打点的版本。

---

## 13. 最有说服力的证据：相同 kernel 的 2×2 调度消融

为了区分 Multistream 和 PDL 各自的贡献，实验固定：

* 完全相同的 11 个 kernel；
* 完全相同的 tile；
* 完全相同的数学计算；
* 只改变是否保留分支并行，以及依赖边是否使用 PDL。

![保持 Kernel 与数学计算不变时，分支 DAG 和 PDL 的二乘二消融](/assets/blog-megakernel-cmp-ablation.svg)

*图 10（教学重绘）：恢复分支 DAG 是最大单项收益；PDL 再把 dependent consumer 的准备工作向前移动。*

结果如下：

| 调度方式            | Graph replay p50 | GPU 执行区间 p50 |         相对线性基线的执行区间缩短 |
| --------------- | ---------------: | -----------: | --------------------: |
| 线性完成链，无 PDL     |      `65.504 μs` |  `61.631 μs` |                    基线 |
| 线性进入顺序，使用 PDL   |      `59.360 μs` |  `54.208 μs` |  `7.423 μs`，约 `12.0%` |
| 分支 DAG，无 PDL    |      `49.152 μs` |  `44.432 μs` | `17.199 μs`，约 `27.9%` |
| 分支 DAG，同时使用 PDL |      `40.960 μs` |  `36.992 μs` | `24.639 μs`，约 `40.0%` |

这组实验说明：

### 第一，最大的单项收益来自 Multistream

仅仅把线性链恢复成真实分支 DAG，就从：

```text
61.631 μs → 44.432 μs
```

缩短了 `17.199 μs`。

原因是三个小 grid 可以共同使用原本空闲的 SM。

### 第二，PDL 进一步隐藏 dependent kernel 的准备窗口

在线性控制组中加入 PDL：

```text
61.631 μs → 54.208 μs
```

累计多个依赖边后，共缩短 `7.423 μs`。

### 第三，两者组合后基本拿到了本例的主要调度收益

```text
61.631 μs → 36.992 μs
```

GPU 执行区间累计下降约 `40.0%`。

### 第四，不能把每个 kernel 的 duration 简单相加

启用并发后，11 个 kernel 的 duration 简单求和反而从：

```text
62.400 μs → 102.864 μs
```

变大了。

这并不表示变慢，原因包括：

* 提前进入的 consumer 会把 GDC 等待算进自身 duration；
* 多个 kernel 并发后会争抢 SM、L2 和带宽；
* 多段 duration 在时间轴上发生重叠。

调度优化应该比较：

> **从第一个关键任务开始，到最后一个关键任务结束的 critical-path span。**

而不是把 profiler 中所有 kernel duration 相加。

---

## 14. 为什么 CMP 最终能够超过 MegaRTP？

### 14.1 MegaRTP 有不可忽略的固定框架开销

为了测量执行框架本身的成本，实验保留相同的 resident CTA、slot、page、TMEM 和退出流程，但不执行真实 Op。

<p class="source-figure-note"><strong>图 11：MegaRTP persistent 执行框架的固定开销</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

固定成本来自：

* persistent kernel 启动；
* CTA 状态初始化；
* page barrier 初始化；
* instruction fetch 与 publish；
* role handshake；
* TMEM 生命周期管理；
* 退出同步。

原文正文与性能总账使用的固定开销为：

```text
4.671 μs
```

于是：

```text
44.272 μs - 4.671 μs = 39.601 μs
```

其中 `39.601 μs` 只是用于分析计算与调度部分的诊断值，不是用户实际会得到的调用延迟。

> **数值说明**：所提供的固定开销配图标题写的是 `4.417 μs`，而原文正文和最终总账使用 `4.671 μs`。两者可能来自不同测量轮次。本文保留配图原样，但所有总账计算沿用正文的 `4.671 μs`，不把两组数字混用。

对于一个总长只有几十微秒的子图，4～5 μs 的固定成本已经是一个很大的比例。

### 14.2 Standalone kernel 可以保留各自最优的资源形态

MegaRTP 的 resident CTA 固定为：

```text
384 threads
```

但不同算子的最优形态可能完全不同。例如：

* 某些 Blackwell `tcgen05.mma` 使用 2-SM 协作，需要两个 CTA 成对到达；
* 高性能 TopK 可能使用 1024-thread CTA；
* 不同 GEMM 需要不同 blockM、stage、cluster、SMEM 和 TMEM 布局；
* 某些算子适合单 CTA，某些算子适合多 CTA 协作。

要把这些 kernel 迁移进统一 worker VM，不能只是复制 kernel body，还要重新设计：

* tile 的读写 region；
* event publication；
* Controller/ALoader/WLoader/MMAer/Epilogue 分工；
* page 生命周期；
* CTA 间协作协议。

CMP 则可以保留每个 standalone kernel 原本最合适的线程数和资源配置，只修改调度和必要的 PDL 入口。

### 14.3 Megakernel 独有的 tile 流水没有落在关键路径上

本例 `M=16=blockM`，关键 consumer 仍然需要等待完整的 M16 数据。于是 MegaRTP 真正难以替代的第三类能力没有产生足够收益。

综合起来：

```text
CMP 拿到了：
  独立分支并行
  + dependent consumer 提前准备
  + 各 kernel 的专用高性能实现

MegaRTP 额外拿到：
  少量 tile 级 overlap

但同时付出：
  固定 persistent VM 开销
  + 算子迁移成本
  + 统一 worker 资源约束
```

因此，在这个特定 workload 上，CMP 最终胜出。

---

## 15. Graph rewrite 为什么后来又被替换了？

最初的 CMP 实现是：

```text
按原顺序在单流 capture
    ↓
得到一条线性 CUDA Graph
    ↓
根据固定 node 编号删除错误的串行边
    ↓
重新连接真实 DAG 和 programmatic edge
```

问题是，CUDA Graph node 本身不带“A0”“QKV-A”这样的算子语义。改图代码之所以知道 `node[3]` 是哪个 kernel，只是因为当前 kernel 数量和顺序被写死。

一旦：

* 插入一个 kernel；
* 删除一个 kernel；
* 拆分或融合一个 kernel；
* 替换某个实现；

后续 node 编号都会变化，旧的改图代码甚至可能把依赖边连到错误节点。

因此，后续实现改成了：

```text
forward 中直接选择 stream
    ↓
producer 启动时绑定 Programmatic Event
    ↓
consumer stream 显式等待该 event
    ↓
外层 CUDA Graph capture 已经正确表达的关系
```

这条工程经验很重要：

> **依赖最好在模型执行代码中以有语义的 stream/event 关系表达，而不是在 capture 后依赖固定 node 位置进行“无语义手术”。**

---

## 16. 相关工作：大家其实都在研究“如何把依赖变成可调度对象”

### 16.1 Stanford Megakernels

Stanford 的 *Look Ma, No Bubbles!* 面向单 GPU、batch size 1 的 Llama decode，也采用：

* Host 构造 DAG；
* Op 拆成 instruction；
* 静态 worker queue；
* persistent kernel；
* warp specialization；
* global counter；
* 跨 instruction 权重预取；
* SMEM page 复用。

MegaRTP 继承了这套总体思路，但为了适配 GLM-5 的动态 `M/KV`、Blackwell GEMM 和可变化模型图，重新设计了 instruction ABI、TaskGraph、OperandPtrTable 和 page 生命周期。

### 16.2 Mirage Persistent Kernel 与 Event Tensor

这类工作把 task 之间的 event 关系进一步做成编译器和 runtime 对象：

* 根据实际读写 region 建立 tile 依赖；
* 合并重复 event；
* 规整 fork/join；
* 压缩 successor metadata；
* 支持静态或动态 scheduler。

但相关消融也说明：

> 细粒度依赖并不会自动产生收益。规则 dense workload 上，如果动态 scheduler 的开销过大，甚至可能比静态方案更慢。

### 16.3 TileRT：更大的收益可能来自跨卡角色分工

<p class="source-figure-note"><strong>图 12：TileRT 中 Attention 使用异构 TP7，MoE 使用 expert 内 TP8</strong><span>原始配图未随附件提供；请在 <a href="https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125" target="_blank" rel="noreferrer">知乎原文</a> 中查看。</span></p>

TileRT 的例子说明，优化不一定要把整个模型塞进一个 kernel。公开 trace 中，一次 forward 在每个 rank 上仍能看到很多 kernel；它不是用一个 kernel 覆盖完整模型，而是在每层 attention 和 FFN 内使用专用 executor，并在多卡层面改变执行图：

* rank 0 专门完成 Indexer Q/K、Score 和 Top-2048；
* rank 1–7 使用 TP7 执行主 MLA；
* MLA ranks 可以先完成 projection，拿到稀疏索引后再 gather KV；
* MoE 又切换为每个 expert 内部的 TP8，而不是把不同 expert 分给不同 GPU 的 EP8。

这个例子和本文的最终结论是一致的：

> **真正大的收益往往来自重新设计计算图、并行方式和资源分工，而不只是消除 kernel 边界。**

---

## 17. 面对一个新子图，应该怎样选择优化手段？

可以按照下面的顺序判断。

### 第一步：先把单 kernel 和必要的 fusion 做好

适合 fusion 的情况包括：

* 相邻算子之间有大量中间数据；
* 中间结果写回显存很浪费；
* epilogue 可以自然合并；
* kernel launch 和小算子开销仍明显。

不要一开始就把整个模型改写成 Megakernel。单 kernel 性能仍然是所有上层调度的基础。

### 第二步：检查 DAG 中有没有独立的小 grid 分支

如果同时满足：

* 分支之间没有真实数据依赖；
* 每个分支的 grid 都用不满 GPU；
* 并发后的资源竞争可控；

优先尝试 Multistream 或显式 CUDA Graph DAG。

这通常比迁移 Megakernel 简单得多，而且本例中它贡献了最大的单项收益。

### 第三步：检查 dependent consumer 能否提前做准备

如果 consumer 在读取 producer 输出前可以完成：

* prologue；
* descriptor 初始化；
* barrier 初始化；
* 权重和 scale 预取；

可以尝试 PDL，把 GDC 推迟到第一次 activation load 之前。

但必须检查最终 SASS 和正确性，不能只看源代码顺序。

### 第四步：判断是否真的存在关键路径 tile 流水

Megakernel 特有能力要产生明显收益，至少应同时满足：

1. producer 的输出可以按更小粒度交给 consumer；
2. consumer 可以对部分输入进行独立有效计算；
3. producer 有多个 wave，第一块输出 ready 明显早于整个 grid 完成；
4. GPU 上还有资源运行 consumer；
5. overlap 位于最终关键路径；
6. 这条流水能够持续发生，而不是只提前一个很小的尾巴。

如果这些条件不成立，tile event 可能只是在形式上更细，实际上仍然接近 whole-op barrier。

### 第五步：计算 Megakernel 的额外成本能否被摊薄

需要评估：

* persistent kernel 固定启动和退出成本；
* VM/instruction 解析成本；
* event 与 barrier 成本；
* worker queue 是否需要动态调度；
* 不同 Op 是否适合统一 CTA 线程数；
* 2-SM、cluster、1024-thread CTA 等特殊资源形态；
* 算子迁移与长期维护成本。

只有 tile 级流水收益明显大于这些成本时，Megakernel 才真正值得。

---

## 18. 哪些场景更适合 Megakernel？

本文的 attention-pre 不理想，因为 `M=16=blockM`，关键 producer 往往一个 wave 就结束，consumer 又必须等待完整输入。

更适合 Megakernel 的场景通常具有持续细粒度流水，例如 MoE：

```text
dispatch 一部分 token 到达某个 expert
        ↓
对应 expert 的 group GEMM 立刻开始
        ↓
其余 token 继续通信
        ↓
完成的输出分片继续进入 combine
```

如果 dispatch、GEMM 和 combine 能长期重叠，Megakernel 的 tile scheduler、优先级控制和局部性调度就可能发挥真正不可替代的作用。

也可以用四个问题快速判断：

| 问题                                 | 越偏向“是”，越适合 Megakernel   |
| ---------------------------------- | ----------------------- |
| 任务是否足够多、能够持续填满 worker？             | 避免 persistent worker 空转 |
| 任务耗时是否不均匀、需要细粒度调度？                 | 静态整 kernel 调度难以平衡       |
| producer 输出是否可切分，consumer 是否能立即消费？ | 形成真正 tile 流水            |
| 是否存在通信、计算和归约的长期重叠？                 | 更容易摊薄框架成本               |

---

## 19. 最后的工程结论：低延迟推理没有魔法

把所有结果放在一起：

```text
约 80 μs
   ↓ Kernel fusion
61.631 μs
   ↓ 恢复分支 DAG / Multistream
44.432 μs
   ↓ PDL 提前启动并预取
36.992 μs GPU span
40.960 μs Graph replay p50
```

MegaRTP 的实际完整运行是：

```text
44.272 μs
```

扣除固定框架后的 `39.601 μs` 只能用于机制分析，不能当作真实调用延迟。

这次实验最值得记住的，不是“Megakernel 赢了”或“Megakernel 输了”，而是下面四句话：

1. **小 kernel 的主要问题不一定是 launch overhead，也可能是小 grid 加单流顺序造成的 SM 空洞。**
2. **看到 overlap 不等于看到端到端收益，必须回到关键路径分析。**
3. **counter event 很细，不代表依赖真的可细分；consumer 的可消费粒度才是关键。**
4. **Megakernel 真正不可替代的价值，是关键路径上持续存在的 tile 级数据流水和细粒度调度，而不是“把很多 kernel 放进一个 kernel”本身。**

原文最后还强调了一个更高层的判断：当执行效率已经接近理想状态时，Megakernel 往往只是最后 `5%～10%`。模型结构和算法变化可能带来更大的收益。例如 GLM-5.2 的 IndexShare 让大约每四层只执行一次完整 Indexer，其余层复用索引；类似地，扩大 MoE 并行规模、用 FP4 MoE 降低权重访存，或者使用投机推理提高每次前向产生的 token 数，都可能比继续抠最后几个微秒更划算。

最终可以把几种手段的职责概括为：

```text
Kernel fusion：减少不必要的 kernel 边界和中间访存
Multistream：让无依赖的小 grid 共同使用空闲 SM
PDL：让有依赖的 consumer 提前进入并准备
Megakernel：进一步控制 tile 级依赖、优先级、局部性和持续流水
```

所以，面对一个新的推理 workload，真正应该问的不是：

> “能不能做成 Megakernel？”

而是：

> **“这个图中是否存在普通 kernel、Multistream 和 PDL 无法表达，并且确实位于关键路径上的持续 tile 级流水？”**

只有答案为“是”，Megakernel 的高开发与维护成本才可能值得。

---

## 附录：一张表快速区分几个容易混淆的指标

| 指标                       | 它表达什么                               | 它不表达什么                                     |
| ------------------------ | ----------------------------------- | ------------------------------------------ |
| `64.65 / 148 = 43.7%`    | 根据 grid CTA 数得到的活跃 SM 数量乐观上限        | 不是 occupancy，不是 SM busy，不是 Tensor Core 利用率 |
| `119.078 / 148 = 80.46%` | 根据 CTA timestamp/SMID 得到的目标 CTA 覆盖率 | 不是算力利用率，也不表示每个 SM 一直满载                     |
| Kernel duration 求和       | 每个 kernel 自身从开始到结束的区间之和             | 并发时不能代表端到端时间                               |
| GPU execution span       | 第一个目标 kernel 开始到最后一个目标 kernel 结束    | 不一定包含完整 Host/Graph 调用开销                    |
| Graph replay p50         | CUDA event 包围一次 Graph replay 的调用侧时间 | 与扣除框架后的诊断值不是同一口径                           |
| MegaRTP `39.601 μs`      | `44.272 - 4.671` 的机制分析值             | 不是实际可交付延迟                                  |
