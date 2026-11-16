---
layout: post
title: "同一局马里奥，两个引擎：把 9B 决策模型跑在本机"
date: 2026-11-15 09:00:00 +0800
author: Haiyan Qin
tags: [ApxInf, MLX, Metal, Apple Silicon, Clef, Decision Model]
reading_time: 11
excerpt: "左边 MLX 官方实现，右边 ApxInf：同一模型、同一策略、同一动作序列，画面快慢只来自模型思考时间。M1 Pro 上 152.4 s vs 76.5 s；顺便把整条链路做成一条命令。"
lang: zh
---

同一局马里奥，同一份权重，同一串按键。唯一不同的是，决定"下一秒跑还是跳"的那个模型跑在哪套引擎上。

视频左边是 MLX 官方实现（`clef_mlx.py`），右边是 ApxInf 的原生 Metal 决策服务。两边模型、策略、动作序列完全相同，画面快慢只来自模型思考时间。到达旗杆：**152.4 s vs 76.5 s**。

<figure class="video-figure">
  <video autoplay muted loop playsinline preload="metadata" poster="/assets/mario-clef/m1pro-race-poster.png" aria-label="M1 Pro 上 MLX 官方实现与 ApxInf 同轨迹对局，ApxInf 明显更快">
    <source src="/assets/mario-clef/m1pro-race.mp4" type="video/mp4">
  </video>
  <figcaption>M1 Pro，World 1-1，v035 配方（39 个决策）。为了看得清，成片按真实时间 4 倍速播放；两边共享的那段游戏画面约 21 s，完全相同。</figcaption>
</figure>

<div class="metric-strip">
  <div><strong>2.0×</strong><span>model time, same weights</span></div>
  <div><strong>1.78×</strong><span>wall time to the flag</span></div>
  <div><strong>0</strong><span>cloud requests</span></div>
  <div><strong>1</strong><span>command from zero</span></div>
</div>

## 怎么比才公平

这种对比最容易作弊：先跑的一方占便宜，机器一热后半段就降频。所以录制按 **A → C → C → A** 交替四局：

- 基线臂（C）每一步都**真实调用 MLX 官方实现计时**，但执行另一侧（A）的动作序列——两边走同一条路线，不会因为一边跳早了而变成两局不同的游戏；
- 每局开始前等机器安静：没有其他模型进程、1 分钟负载低于阈值、连续三次满足；
- 录制期间每 5 秒采样负载、换页和外来进程，写进监控记录。

两轮 MLX 结果是 152.4 / 150.2 s，两轮 ApxInf 是 76.5 / 74.5 s。组内差都在 3% 以内，说明机器状态稳定；跨组的差就是引擎的差。

## 模型在回答什么

马里奥不需要对话。每步暂停时，它只回答三个是/否问题：前方有敌人吗、有障碍吗、地板有坑吗。这是决策模型的形状——把状态和结构化问题映射成概率，一次前向出结果，不生成文本。

每步只喂三块裁剪图：敌人（64×80）、障碍（64×48）、坑（96×64），最近邻放大到 320×400 / 384×288 / 480×320。默认 `separate` 协议一图一题一请求。

<figure>
  <img src="/assets/mario-clef/what-it-sees.png" alt="一帧决策对应的三张裁剪图与三个 yes-no 概率" width="1280" height="720">
  <figcaption>一帧决策 = 三张裁剪 + 三个概率。位置、是否落地、截图、阈值和按键全部由 Python 负责；模型不读 RAM、不规划路线。</figcaption>
</figure>

模型的能力上界不是"能不能通关"，而是"这三张图里的三个问题答得准不准"。通关是代码逻辑、阈值和动作帧数的事。两者必须分开评估，否则你永远不知道是模型看错了，还是动作配方不行。

## 一条命令

在配套教程仓库根目录：

```sh
./scripts/mario_demo.sh
```

它把从零到页面之间的事情全部做完：检查机器、准备 MLX 0.32.3 SDK、构建决策服务（`--features coreml`，GPU 与 Neural Engine 在同一个二进制里）、从 ModelScope 下 6.2 GB 权重、建 demo 的 Python 环境、找空闲端口启动页面，然后打开浏览器。

```text
[1/6] 系统与工具         Apple M2（16 GB）
[2/6] MLX 0.32.3 SDK     复用 / 已安装
[3/6] apxinf-decide-server 构建或复用
[4/6] Clef 权重          ModelScope（失败回退 hf-mirror / 官方）
[5/6] demo Python 环境   gym-super-mario-bros 7.4.0
[6/6] 启动页面           READY http://127.0.0.1:8766
```

看到 `READY` 后，策略下拉选 `clef`、点**重开**、再点**单步**或**自动**。第一次点击要等权重加载和三个裁剪形状各自的首次编译，之后进入稳态。每一步都能在页面上看到三张裁剪、三个概率和这一决策的延迟；云端用量永远是 0。

## 同一套配方，M2 上也能跑

<figure class="video-figure">
  <video controls preload="none" poster="/assets/mario-clef/hero-poster.png" aria-label="M2 Air 上的同一套配方连续回放">
    <source src="/assets/mario-clef/hero.mp4" type="video/mp4">
  </video>
  <figcaption>M2 Air 上的回放（24 个决策、x=507）。点播放；画面按 NES 60 Hz 逐帧播放，步间停顿是模型在思考。</figcaption>
</figure>

M2 之前一直落在 reference 组合上，慢到不适合演示。这次把它的 GPU 组合纳入资格列表后，同一台机器从"能跑通"变成"能看"：

| 项 | reference（未资格时） | GPU 组合（现在） |
| --- | --- | --- |
| 加载 `load_ms` | 16.2 s | 16.7 s（同机记录） |
| 单个冷请求 | 66.6 s | 秒级（低内存压力窗口） |
| 一局 | 1 决策 383 s（换页窗口） | `single` 协议 39 决策到旗杆，中位 2.99 s/决策 |
| 云端请求 | 0 | 0 |

M2 的 ANE 我也探了一轮，结论比预期更微妙：**算子支持不是问题，成本才是**。

- 当前 neuron 程序的 **38 个运行时算子**（conv、mul、tanh、reduce_max、split 等）在 M2 上全部是 `preferred=NeuralEngine`。旧版本里"某个算子不支持"的说法，随程序重做已经不再成立。
- 真正的开销是 Core ML 的 compute-plan 检查：M2 上约 **75–90 秒/层**。column 切分实测加载 **28.9 分钟**；neuron 组合在 release 驱动上编译完 32 个程序用了约 **41 分钟**，仍未 `ready`（进程在加载结束前退出）。
- 请求侧：column 切分要 **30–94 秒**，同一请求在 GPU 组合上只要 **7.5–8.1 秒**（release，3 次）；暖缓存也不会跳过 plan 检查。

所以 M2 默认留在 GPU 组合。要让 ANE 变成可选项，得先摊销 plan 检查的成本，而不是换编码或换算子。

## 三种芯片，三种组合

ApxInf 只在**通过资格验证**的架构上启用优化组合，其余芯片退回 reference（能跑，只是慢）。构建只有一种 `coreml`：

| 芯片 | `auto` 选中 | 依据 |
| --- | --- | --- |
| M1 Pro（`applegpu_g13s`） | GPU + ANE 列切分 | GPU 与列切分都过资格 |
| M2（`applegpu_g14g`） | GPU 组合 | 2026-11-15 起纳入；demo 裁剪 parity max abs(Δp) 0.0024（gate 0.035）。ANE 见下 |
| M4（`applegpu_g16g`） | GPU + ANE 神经元切分 | 加载 93–130 s 量级，稳态更快 |
| 普通 M1 / M3 | reference | 能跑，只是慢 |

同一套对局在 M4 上也录了一版（38 s），结论方向一致：换芯片会同时换掉组合，跨机器只能各自陈述，不能直接相除。

## 三个只有真跑才会撞到的坑

<div class="callout">
<strong>1 · 二进制没有 rpath，必须带 DYLD_LIBRARY_PATH</strong>
漏了会得到 <code>dyld: Library not loaded: @rpath/libmlx.dylib … no LC_RPATH's found</code>。这是 dyld 错误，不是模型或权重问题，第一次遇到很容易往错的方向查。
</div>

<div class="callout">
<strong>2 · demo 目录只读时，账本会先炸</strong>
启动快照会读写云端账本，默认落在 demo 目录的 <code>devlocal/</code>。目录不可写时抛 <code>PermissionError: [Errno 1] Operation not permitted</code>，而且不是可读提示。设 <code>MARIO_API_LEDGER</code> 到可写位置即可——本地 <code>clef</code> 策略本身不往这个账本记账。
</div>

<div class="callout">
<strong>3 · 16 GB 是下限，不是舒适区</strong>
Clef 4-bit 峰值 7.0–8.6 GB。系统已有几个大应用时，机器进入换页主导状态：加载 16 s，单个请求却能到 66 s 甚至两分钟。你测到的不是模型速度，是内存压力。
</div>

## 自己动手

不用脚本时的核心步骤（在 ApxInf 源码目录内）：

```sh
export MLX_ROOT="$HOME/.local/share/apxinf-mlx-sdk/lib/python3.11/site-packages/mlx"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib"
cargo build --release --locked --features coreml --bin apxinf-decide-server
```

```sh
export APXINF_DECIDE_CMD="$PWD/target/release/apxinf-decide-server \
  --model $HOME/models/clef-flash-4bit --model-variant auto"
export MARIO_CLEF_PROTOCOL=separate
export APXINF_LOAD_TIMEOUT=900 APXINF_DECIDE_TIMEOUT=900
devlocal/mario-demo/.venv/bin/python examples/mario_demo/server.py --port 8766
```

离线跑一局（不打开页面）：

```sh
devlocal/mario-demo/.venv/bin/python examples/mario_demo/evaluate.py \
  --policy clef --max-decisions 12 --out devlocal/mario-demo/clef-12
```

## 边界

- **2.0× 是模型耗时，1.78× 是到旗杆的墙钟**。后者包含两边共享的约 21 s 游戏画面，这两个数不能混着说。
- **无风扇的 M2 Air 不能用来比速度**：连续推理几分钟就降频，先跑的一方天然占便宜。它适合验证"能不能跑"，不适合出加速比。
- 12 个决策全是负例时，混淆矩阵会是全 TN、准确率 1.0。这只说明这些帧上没有误报，不是识别能力，也不是通关结论。
- 换协议（`separate` / `joint` / `single`）、换阈值、换动作帧数，都要重新跑基线。

**归属**：决策服务与 Metal 后端来自 [ApxInf](https://github.com/infinigence/ApxInf)；模型是 [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit)（基座 [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)，Apache-2.0），[ModelScope](https://modelscope.cn/models/mlx-community/clef-flash-4bit) 上有同 id 镜像；Mario demo 的裁剪、问题、阈值与宏动作移植自 [FluidUse](https://github.com/FluidInference/FluidUse) 的 JevMarioDemo（Apache-2.0）。
