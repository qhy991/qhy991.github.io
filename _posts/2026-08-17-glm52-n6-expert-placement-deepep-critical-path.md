---
layout: post
title: "为什么搬一搬 Expert 就能让 8 张 B300 更快？从 MoE 路由、DeepEP 到端到端关键路径"
date: 2026-08-17 02:00:00 +0800
author: Haiyan Qin
lang: zh-CN
math: true
series: gpu-systems
tags: [MoE, DeepEP, B300, SGLang, Distributed Inference, GPU Systems]
reading_time: 26
cover_image: /assets/blog-glm52-n6-expert-map.png
excerpt: "N6 没有把主 GEMM 变快，而是同时调整 Expert placement、Router 输出的 ID 空间和 DeepEP 的 SM 预算。本文从 logical/physical expert ID、三条执行 DAG 与关键路径出发，解释这种跨边界优化为什么能让 8×B300 的 GLM-5.2 cached-prefill 更快。"
---

> 本文从 [`qhy991/SGLang-DGMK@9e439b4`](https://github.com/qhy991/SGLang-DGMK/commit/9e439b4a9bdb6339a9ede363b19d18d4f3192b8f) 的公开代码、冻结 workload contract、原始配对样本与 compact profile evidence 中抽取可迁移知识。它不是实验流水账，也不把局部测量外推到未测试 workload。

一个常见直觉是：要让大模型推理更快，就应该优化最慢的 GEMM、写更快的 attention kernel，或者把更多工作塞进 Tensor Core。

但在一次 8×B300 的 GLM-5.2 优化中，唯一通过完整晋级门槛的 N6 并没有让主 GEMM 换算法。它做的是三件看起来不那么“像 kernel 优化”的事：

1. 重新安排 256 个 routed expert 在 8 张 GPU 上的物理位置；
2. 让 Router 直接产出 DeepEP 最终需要的 physical expert ID；
3. 将 DeepEP normal dispatch/combine 的资源预算从 136 SM 调整为 120 SM。

三项组合把正式测试的 P50 TTFT 从 `2042.12 ms` 降到 `1933.67 ms`，下降 `5.31%`；total-token throughput 从 `452544.29` 提升到 `486640.21 token/s`，提高 `7.53%`。独立 client seed 又复现了 `5.46%` 的 P50 改善和 `7.40%` 的吞吐提升。

真正值得学习的不是这几个百分比，而是背后的系统规律：

> **多 GPU 推理的关键路径经常由“谁拥有什么数据、ID 在哪个空间、各 rank 何时到齐”决定，而不是由某一个最大 kernel 的绝对速度决定。**

---

## 1. 先冻结问题：我们究竟在优化什么？

在讨论原因以前，必须先把 workload 说完整。N6 的公开结论只属于下面这个 cell：

| 维度 | 冻结值 |
| --- | --- |
| GPU | 8×NVIDIA B300 SXM6 |
| 模型 | GLM-5.2 FP8 |
| 并行 | TP8 / DP8 / EP8，`attn_cp_size=1` |
| 请求 | 90K logical shared prefix，实际 cache hit 89,984 |
| 增量输入 | 每请求 10K suffix，output=1 |
| 压力 | 110 requests，concurrency=11 |
| KV / chunk | page 64，global chunk 80,384，per-rank ceiling 10,048 |
| 执行 | FlashMLA-KV，DeepEP normal，overlap/graph 关闭 |
| 主指标 | cached-prefill 的 TTFT 和 total-token throughput |

这里最容易犯两个错误。

第一，这是 **cached prefill / first-token** 结果，不是 decode TPOT。第二，`TP8/DP8/EP8` 描述的是重叠的并行视角，不能把三个数字相乘成 512 张 GPU；实际就是 8 张卡。

完整机器可读合同在 [`workload_contract.json`](https://github.com/qhy991/SGLang-DGMK/blob/9e439b4a9bdb6339a9ede363b19d18d4f3192b8f/glm52_opt/b300_100k/workload_contract.json)。离开这个合同，N6 仍然可以是一个研究假设，但不再是已经证明的结论。

---

## 2. MoE 为什么会出现“逻辑 Expert”和“物理 Expert”？

假设模型有 256 个 routed expert。Router 对每个 token 打分，并选出 Top-K expert。模型语义中的编号可以写成：

$$
e_{\mathrm{logical}} \in \{0,1,\ldots,255\}
$$

这个编号回答的是：**模型想调用哪组权重？**

但 EP8 把 expert 分布在 8 张 GPU 上。运行时还需要回答另一个问题：**这组权重现在由哪张卡、哪个本地槽位持有？** 因此需要一个 placement map：

$$
P:\ e_{\mathrm{logical}} \longrightarrow e_{\mathrm{physical}}
$$

只要 map 是 256 个 expert 的 permutation，并且权重与 ID 使用同一映射，模型数学语义就没有改变。变的是数据要被 dispatch 到哪里，以及各 rank 收到多少工作。

可以把它想成一家有 256 个菜品、8 个厨房的餐厅：

- logical ID 是菜单编号；
- physical ID 是“这道菜现在由哪个厨房的哪个灶台负责”；
- Router 是接单员；
- DeepEP dispatch 是把订单送到正确厨房；
- expert GEMM 是做菜；
- combine 是把各厨房的结果送回原 token。

菜单没有变，但如果热门菜全部放在同一个厨房，所有客人仍会被最慢的队伍拖住。

---

## 3. 原始路径的问题：同一个 Expert ID 被加工了好几次

在原始路径里，Router 先在 logical ID 空间完成 Top-K。随后还要执行几段后处理：

```text
Router / Top-K
    ↓ int32 logical IDs
logical → physical remap
    ↓
CUDA Graph padded-row mask
    ↓
int32 → int64 conversion
    ↓
DeepEP dispatch
```

每一步的算术都很少，但会产生新的 tensor、额外读写、小 kernel launch，以及最麻烦的一件事：系统里同时存在两种“最终 expert ID”的解释。

![原始 Router 后处理链与 N6 单一写出路径](/assets/blog-glm52-n6-pipeline.svg)

*图 1：N6 没有改变 Top-K 数学，而是让最终 ID 在唯一的 canonical stage 生成。*

这类开销为什么可能重要？因为 MoE Router 每个 active MoE layer 都会调用；一个只有几微秒甚至更短的小步骤，乘上层数、请求、rank 和 prefill chunk 后，会变成高频控制/物化成本。

更重要的是，这些步骤位于 Router 与通信之间。它们不仅占时间，还推迟了 DeepEP dispatch 的最早开始时刻。

---

## 4. N6 的第一层重写：让 placement 成为数据，而不是散落的规则

N6 使用一份静态 expert map。公开文件包含 78 行、每行 256 个 ID；每一行都是合法 permutation。前 3 行保持 identity，其余 75 行使用冻结的 balanced placement。文件 SHA-256 为：

```text
36d13233672288317fd69495d4cedb46844b8ae99033d184d84aff0c99c68f09
```

这里的设计重点不是某个具体排列，而是 **SSOT**：placement map 是“logical expert 到 physical owner”的唯一权威数据。Router、权重布局和 DeepEP 必须围绕同一份 map 解释 ID。

为什么 balanced placement 可能缩短时间？多 rank MoE 的一个阶段通常要等必要的远端结果到齐。粗略地说，完成时间更接近：

$$
T_{\mathrm{stage}} \approx \max_{r \in \mathrm{ranks}}
\left(T_{\mathrm{dispatch},r}+T_{\mathrm{expert},r}+T_{\mathrm{combine},r}\right)
$$

平均负载降低并不够；真正决定 join 时刻的是最慢 rank。静态 map 的目标是让冻结流量下的热点 expert 更均匀地落到不同 physical rank，从而减小长尾。

但要谨慎：这不是“均匀 map 对所有请求都更好”。路由分布会随模型、数据、上下文、batch 和阶段变化。N6 map 只对冻结的 GLM-5.2 / EP8 / 100K-x11 cell 有正式证据。

---

## 5. N6 的第二层重写：Router 直接写出消费方格式

代码修改的核心在 `moe_fused_gate.py` 和 `topk.py`。新的 guarded path 给 Router 增加三个可选能力：

- 读取一维、CUDA、连续、int64 的 `logical_to_physical_map`；
- 根据 `num_token_non_padded` 将 padded row 的 ID 写成 `-1`；
- 直接建立 DeepEP ABI 所需的 int64 输出。

Router 仍然先在 logical ID 空间选择 Top-K。这一点非常重要：placement 不能反过来改变模型选中了哪个 expert。只有 routed columns 在最终 store 前读取 map；shared expert slot 保持原语义。

新路径可以概括为：

```text
Top-K winner（logical）
    ↓ map lookup，仅 routed columns
physical ID
    ↓ padded row ? -1 : physical ID
一次 store，直接 int64
    ↓
DeepEP dispatch
```

这是一条很通用的优化原则：

> **让 producer 直接写 consumer 的最终 layout、dtype 和语义，优先删除中间物化，而不是先生成通用格式再层层修补。**

对应的实现提交是 [`01749bb`](https://github.com/qhy991/SGLang-DGMK/commit/01749bb27781c5044898fd62ac4b70ced98c8335)，只修改了 3 个运行时文件并增加 2 个 CUDA tests，没有复制一套平行 Router。

---

## 6. 为什么“做完以后跳过第二遍”与 fusion 同样重要？

只把 map lookup 塞进 Router 还不完整。如果上层不知道 Router 已经产生 physical ID，后处理仍可能再 remap 一次，得到错误结果或重复工作。

因此 `topk.py` 还维护两个显式事实：

```text
static_placement_already_fused
padded_region_already_masked
```

后处理根据这两个事实跳过已经完成的 remap、mask 和无意义转换。最终 expert ID 只有一个 canonical producer。

这比“再加一个 fast path”更重要。很多 fusion 失败并不是新 kernel 不快，而是旧路径的一半还留着：

```text
新 kernel 做了一次
        +
旧 postprocess 又做一次
        =
更复杂、甚至更慢
```

真正的 fusion 应该同时回答两件事：

1. 新边界里增加了什么？
2. 原边界外因此可以删除什么？

---

## 7. 第三层重写：为什么 120 SM 可能比 136 SM 更好？

DeepEP normal dispatch/combine 本身会占用 GPU 资源。直觉上，给它 136 个 SM 应该比 120 个 SM 更快，但端到端系统并不只运行通信 kernel。

通信与计算可能争用：

- SM residency 和可调度 CTA 槽位；
- L2 容量与 fabric-facing traffic；
- HBM 带宽；
- rank progress 与 notify/wait 路径；
- 后续 expert compute 的启动窗口。

减少通信 kernel 的 SM 配额，可能让单次通信的局部 duration 变长，也可能为其他工作或进度机制保留空间，从而缩短完整 envelope。

但 N6 的公开正式 A/B 只测了三项组合，不能把 `136→120` 单独描述为 5.31% 的来源，也不能把 placement、Router fusion 和 SM budget 的局部百分比相加。正确表述是：

> **三项共同改变了跨模块、跨 rank 的调度边界；组合通过了端到端门槛，单项贡献没有被这组证据识别。**

---

## 8. 真正的分析对象是三张 DAG

如果只看 `forward()`，N6 像是一个 Router 小优化；如果只看 kernel 表，它又像是删掉了一些 mask kernel。两种视角都不完整。

![N6 同时改变 compute、communication 和 control DAG](/assets/blog-glm52-n6-three-dags.svg)

*图 2：N6 的价值来自三张 DAG 的边界同时对齐，而不是某一个节点孤立变快。*

### Compute DAG

Router 完成评分、Top-K 和最终 ID 写出；expert GEMM 数学不变。这里删除的是 post-router remap/mask/cast 的重复物化。

### Communication DAG

placement 决定 token 发往哪个 physical expert；DeepEP dispatch/combine 的流量分布和 rank 尾部因此改变。

### Control DAG

strict gate 决定是否进入融合路径；selected marker 证明实际命中；SM budget 改变通信工作如何与其余 GPU 工作共享资源。

最终用户看到的 TTFT 近似关键路径长度：

$$
T_{\mathrm{TTFT}} \neq \sum_i T_{\mathrm{kernel},i},
\qquad
T_{\mathrm{TTFT}} \approx \max_{p \in \mathrm{paths}} \sum_{n \in p} T_n
$$

多 stream、多 rank 和多请求会重叠。把所有 GPU 上的 kernel duration 相加，既不是延迟，也无法告诉你哪个 wait edge 真正限制了第一个 token。

---

## 9. 一个优化如何从“看起来合理”升级为“可以相信”？

N6 的证据不是一个最快数字，而是四层互相不能替代的证明。

![N6 的正确性、正式配对、独立 holdout 与因果 profile](/assets/blog-glm52-n6-evidence.svg)

*图 3：correctness 决定能不能比较，no-profiler E2E 决定是否更快，holdout 检查复现，profile 只负责解释。*

### 9.1 路径与正确性

候选必须打印：

```text
GLM-5.2 router static-placement fusion selected:
logical-to-physical + optional padded-row mask + int64 IDs
```

没有 marker 的成功请求可能走了 fallback，不能算 N6。两个 CUDA tests 覆盖 static-placement 等价、ragged/padded boundary 和 int64 ABI；独立 correctness probe 要求 generated tokens 对两个 reference exact。

### 9.2 五对 no-profiler 正式测试

正式测试保留了五对完整顺序，而不是只抄一个平均值。N6 赢得 4/5 个 P50 pair。值得注意的是：某些 pair 的 P90 或 throughput 会反向，这正是为什么需要预声明 gate、看整体中位数并继续做独立 seed，而不能挑最好的一次截图。

### 9.3 独立 client-seed holdout

holdout 的五个 P50 pair 全部获胜；P50 中位改善 `5.46%`，P90 改善 `6.73%`，吞吐提高 `7.40%`。这降低了 placement 只对单一请求顺序过拟合的风险，但仍不等于覆盖真实在线流量分布。

### 9.4 Matched Nsys 因果解释

profile 观察到：

- 同一观察窗口内 dispatch progress calls 的 device median 从 `616` 增至 `654`，增加 `6.17%`；
- cached notify/combine wait 的 P50/P90 分别下降 `43.65%`/`45.91%`；
- combine 主 kernel P50 近似不变；
- 每张 GPU 删除了约 `316–466` 个 padded-ID mask kernel。

这些证据支持“rank progress 更顺、通知尾部缩短、重复小工作被删除”的解释。它不支持“combine kernel 本身快了 5%”，也不能分离三项 treatment 各自贡献。

---

## 10. 最有价值的反例：kernel 快 59.1%，服务反而更慢

同一批研究中的 N40 融合了 routed-MoE 的 SwiGLU、FP8 quant 和 `round_to_bf16`。在 exact-shape leaf benchmark 上，它从 `0.045245 ms` 降到 `0.018443 ms`，局部快约 `59.1%`。

如果把 kernel 优化当成终点，这已经像是一个巨大胜利。但进入服务 development bracket 后：

| 指标 | N40 相对 control |
| --- | ---: |
| P50 TTFT | **慢 1.56%** |
| P90 TTFT | **慢 18.61%** |
| Throughput | **低 6.01%** |

并且候选侧出现 `empty_chunked_topk`，首次集成还暴露了 local `M=0` 的 zero-grid 边界。

这不是矛盾。leaf benchmark 只回答一个局部问题；服务路径还包含 launch、空 rank、stream、DeepEP、chunk、同步和调度行为。Amdahl 上界也会把局部收益迅速压小：

$$
\text{E2E ceiling} \lesssim
\text{critical-path share} \times \text{removable fraction}
$$

如果该 leaf 只占关键路径很小一段，哪怕自身快 59%，也可能抵不过新同步、资源竞争或长尾。

N6 与 N40 放在一起，构成了比“最终 winner”更有价值的教材：

- N6 没有耀眼的单 kernel 数字，却改变了真正的跨 rank 关键路径；
- N40 有耀眼的 leaf 数字，却没有通过端到端裁决。

---

## 11. 把 N6 抽象成五条可迁移原则

### 原则一：先优化 ID 和数据所有权，再优化搬运实现

通信成本首先由“发送什么、发送给谁”决定。placement、shard ownership 和 consumer format 经常比替换 collective kernel 更上游。

### 原则二：producer 应该直接写 consumer 的最终格式

如果 Router 的直接消费者需要 physical int64 ID，那么 logical int32 → remap → mask → cast 是值得审计的中间链。

### 原则三：最慢 rank 比平均 rank 更接近真实瓶颈

有 join 的分布式阶段通常由 tail 决定。只报告平均 expert load，可能掩盖某张卡持续成为 straggler。

### 原则四：更多 SM 不等于更短的端到端时间

资源配额是全图调度问题。单个通信 kernel 获得更多 SM，可能让与它共享 SM/L2/HBM 的其他节点更晚完成。

### 原则五：证据级别不能跳级

正确的晋级顺序是：路径命中 → correctness → no-profiler E2E → holdout → profile 解释。NCU 或 leaf win 不能替代服务结论。

---

## 12. 面对下一个 MoE 优化，可以直接问什么？

可以按下面的顺序检查：

1. Router 输出的是 logical ID 还是 consumer 最终需要的 ID？
2. placement 的 SSOT 在哪里？权重、Router 和通信是否使用同一方向的 map？
3. Router 与 dispatch 之间是否还有 remap、mask、cast、pack 或 D2H？
4. padded token、shared expert、空 rank 和 `M=0` 的语义由谁负责？
5. dispatch/combine 是否与 expert compute 争用 SM、L2 或 HBM？
6. 时间由平均 rank 决定，还是由最慢 rank 和 notify tail 决定？
7. 当前测量是 leaf、layer、full model，还是 HTTP serving？
8. optimized path 是否有 marker，unsupported shape 是否 fail closed？
9. 正式数字是否来自无 profiler 的交错样本？
10. profile 是否真的覆盖所有相关 rank、stream 和关键路径？

如果这些问题还没有答案，继续调 tile 往往太早。

---

## 结语：真正的优化对象是边界

N6 最值得学习的地方，不是“换一个 Expert Map 可以快 5%”。更准确的理解是：

```text
placement 决定数据所有权
        ↓
Router 决定 ID 在哪个语义空间落地
        ↓
DeepEP 决定数据如何跨 rank 移动
        ↓
SM budget 决定通信如何与其他 GPU 工作共享资源
        ↓
最慢 rank 和关键路径决定第一个 token 何时出现
```

当这几层边界彼此错位时，系统会用额外 kernel、额外 tensor 和额外等待把它们重新拼起来。当边界被对齐，真正消失的不只是几次 launch，而是一段高频的物化链和一部分跨 rank 尾部。

所以，下一次看到“主 GEMM 已经很快，但端到端仍然不快”时，不妨把视线从单个 kernel 移开，去找那条从数据所有权、ID 语义一直延伸到最慢 rank 的链。

这条链，往往才是系统真正的关键路径。
