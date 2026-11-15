---
layout: post
title: "一条命令跑起 Mario × Clef：ApxInf Metal 决策后端实测"
date: 2026-11-15 09:00:00 +0800
author: Haiyan Qin
tags: [ApxInf, MLX, Metal, Apple Silicon, Clef, Decision Model]
reading_time: 10
cover_image: /assets/blog-mario-clef-cover.svg
excerpt: "把 9B、4-bit 的 Clef 决策模型跑在本机 Metal 上，让 Super Mario 每一步只回答三个视觉问题。一条命令完成 SDK、构建、权重和环境；附 M1 Pro / M2 / M4 的组合差异与实测数字。"
lang: zh
---

# 一条命令跑起 Mario × Clef：ApxInf Metal 决策后端实测

Super Mario 的 demo 不需要一个会聊天的模型。它只需要在每一步暂停时回答三个问题：前方有敌人吗、有障碍吗、地板有坑吗。拿到三个概率，代码按阈值决定跑还是跳。

Clef 正好是这个形状的模型：9B、affine 4-bit，把"状态 + 结构化问题"直接映射成每个候选答案的概率，一次前向出结果，不生成文本。把它的原生 Metal 实现接进 demo，就得到一条完全本地的决策链路：没有云端请求，没有密钥，三个裁剪图不出本机。

这篇记录的是把它做到"一条命令从零跑起来"的过程，以及实测出来的数字和坑。

## 先跑起来

在配套教程仓库根目录执行：

```sh
./scripts/mario_demo.sh
```

脚本会把从零到页面之间的所有事情做完：

| 步骤 | 脚本做什么 | 首次耗时 |
| --- | --- | --- |
| 1 系统检查 | 确认 Apple Silicon + macOS，检查 cargo / uv / curl、内存与磁盘 | 秒级 |
| 2 MLX SDK | 本机没有 0.32.3 就在 `~/.local/share/apxinf-mlx-sdk` 装一份（wheel 自带 C++ 头文件与 `mlx.metallib`） | 1–2 分钟 |
| 3 构建 | 没有二进制就 clone [ApxInf](https://github.com/infinigence/ApxInf) 并构建决策服务（`--features coreml`，GPU 与 Neural Engine 都在里面） | 几分钟 |
| 4 权重 | 从 ModelScope 下载 `clef-flash-4bit`（同 id 镜像，失败回退 HF 镜像与官方 Hub） | 6.2 GB |
| 5 Python 环境 | 建 venv，装模拟器依赖（`gym-super-mario-bros` 自带 ROM） | 1–2 分钟 |
| 6 启动 | 选空闲端口起页面、等 HTTP 200、打印地址并打开浏览器 | 秒级 |

每一步都可重复执行：已就绪的 SDK、二进制、权重、venv 直接复用，只补缺的部分。第二次起基本秒开。

看到 `READY` 之后，策略下拉选 `clef`，点**重开**，然后**单步**或**自动**。页面会显示 `本地请求` 计数、每步延迟和三个裁剪图；云端用量与费用保持 0。第一次点击要等模型加载加三个裁剪形状各自的首次编译，之后就进入稳态。

## 谁在跑哪一步

| 组件 | 职责 |
| --- | --- |
| Mario demo（Python） | NES 模拟器、网页、裁剪三张前方图、按阈值选动作、逐帧回放 |
| `apxinf-decide-server`（Rust） | 常驻进程；收一行 JSONL，返回每个问题的概率 |
| Clef 检查点 | 9B，affine 4-bit 骨干 + BF16 joint schema head |
| MLX 0.32.3 SDK | Metal 执行后端（构建期头文件 + 运行期动态库） |

一次"决策"发生在一个暂停的帧上：`bridge.py` 从 RAM 读脚面位置，裁出敌人（64×80）、障碍（64×48）、坑（96×64），最近邻放大后 base64；`clef_local.py` 按 `separate` 协议发三个请求（一图一题）；服务端预处理图片、渲染 Clef 提示词、跑一次前向；`controller.py` 用阈值决定长跳、短跳、跑还是等待。模型不读 RAM、不决定按键序列、也不做长程规划。

## 芯片差异：`auto` 会选什么

ApxInf 的原生 Clef 实现只会在**通过资格验证**的架构上启用优化组合，否则退回 reference。构建只需要一种 `coreml`（包含 `mlx`），GPU 与 Neural Engine 都在同一个二进制里：

| 芯片 | `auto` 选中的组合 | 说明 |
| --- | --- | --- |
| M1 Pro（`applegpu_g13s`） | GPU + ANE 列切分 | 两条路都过资格 |
| M2（`applegpu_g14g`） | GPU 组合 | 2026-11-15 起纳入 GPU 资格（demo 裁剪 parity `max\|Δp\| = 0.0024`）；ANE 未资格 |
| M4（`applegpu_g16g`） | GPU + ANE 神经元切分 | 加载 93–130 s 量级 |
| 其他（普通 M1、M3） | reference | 能跑，只是慢 |

reference 组合能跑，只是慢，不是错误。启动日志会明确写"没有已资格组合，使用 reference"。想强制某个组合做对照可以用 `APXINF_CLEF_COMPOSITION`，但它绕过资格门，只适合诊断。

## 实测

先说我手上这台：MacBook Air M2（8 核）、16 GB、macOS 27.0.1。`applegpu_g14g` 已经纳入 GPU 资格，`auto` 选 GPU 组合：

| 项 | 结果 |
| --- | --- |
| 资格证据 | 2026-11-05 同机：demo 三块裁剪的 GPU vs reference `max\|Δp\| = 0.0024`（gate 0.035） |
| 加载（`load_ms`，同机记录） | 16.7 s |
| 稳态（低内存压力窗口） | `single` 协议 39 决策跑到旗杆，中位 2.99 s/决策，整局 120.9 s |
| reference 对照（换页窗口） | 冷请求 66.6 s、1 决策 383.2 s；概率 0.023871 / 0.060975 / 0.062789 |
| 云端请求与费用 | 0 |

这台机器当时已经用掉约 18.7 GB swap、空闲页只有 60 MB 左右，所以 reference 对照那行是"换页主导"的读数，不是性能结论；GPU 组合在低压力窗口里才能看到秒级稳态。

早期在同一套 demo 上的记录：M1 Pro（`applegpu_g13s`）reference 组合约 2.2 s/请求、6.6 s/决策；`single` 协议配合调过的阈值和动作帧数，在低内存压力窗口里跑到过旗杆（39 决策、x=3161、得分 800、中位 2.99 s/决策）。M4 的 GPU+ANE 神经元组合加载在 93–130 s 量级，换来更快的稳态决策。

这三组数字不能直接互相比较：芯片、构建、内存压力、协议都不同。

## 三个真实的坑

**1. 二进制没有 rpath。** 下游可执行文件不指向 MLX SDK，运行期必须带上 `DYLD_LIBRARY_PATH`。漏了会看到：

```text
dyld[...]: Library not loaded: @rpath/libmlx.dylib
  Referenced from: .../apxinf-decide-server
  Reason: no LC_RPATH's found
```

这是 dyld 错误，不是模型或权重问题，容易误判。

**2. demo 目录只读时，账本会先炸。** 每次启动快照都会读写云账本，路径默认落在 demo 目录下的 `devlocal/`。目录不可写时会抛 `PermissionError: [Errno 1] Operation not permitted`，而且不是可读的提示。解决办法是显式设 `MARIO_API_LEDGER` 到可写位置；本地 `clef` 策略本身不往这个账本记账。

**3. 16 GB 是下限，不是舒适区。** Clef 4-bit 峰值 7.0–8.6 GB。16 GB 机器上只要系统已经有几个大应用，就会进入换页主导的状态：加载 16 s，但单个请求可以到 66 s 甚至两分钟。先关应用再测，否则读到的不是模型速度。

## 权重下载：ModelScope 快一个数量级

同一个模型在 ModelScope 上有同 id 的完整镜像（18 个文件，含两个 safetensors 分片和 joint head）。同一网络、20 MB 文件实测：

| 源 | 速度 |
| --- | --- |
| ModelScope | 约 4.3 MB/s |
| hf-mirror | 约 90 KB/s |
| 官方 HF | 约 96 KB/s |

所以脚本默认走 ModelScope，失败自动回退 hf-mirror、再回退官方 Hub：

```sh
# 默认：ModelScope（缓存显式放到可写目录）
MODELSCOPE_CACHE="$PWD/devlocal/mario-demo/.modelscope-cache" \
  modelscope download mlx-community/clef-flash-4bit --local-dir "$HOME/models/clef-flash-4bit"

# 备选：HF 镜像 / 官方
HF_ENDPOINT=https://hf-mirror.com hf download mlx-community/clef-flash-4bit \
  --local-dir "$HOME/models/clef-flash-4bit"
```

两个工具都支持断点续传。受限环境里它们的默认缓存可能不可写，手动跑时记得像上面那样显式指定缓存目录，或者用 `--model-dir` 指向已经下好的目录。

## 手动等价步骤

不想用脚本时，核心命令是这些（在 ApxInf 源码目录）：

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
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib"
devlocal/mario-demo/.venv/bin/python examples/mario_demo/server.py --port 8766
```

离线跑一局（不打开页面）：

```sh
devlocal/mario-demo/.venv/bin/python examples/mario_demo/evaluate.py \
  --policy clef --max-decisions 12 --out devlocal/mario-demo/clef-12
```

## 边界

- 12 个决策全部是负例时，混淆矩阵会是全 TN、准确率 1.0。这只说明这些帧上没有误报，不是识别能力，也不是通关结论。
- 换协议（`separate` / `joint` / `single`）、换阈值、换动作帧数，都要重新跑基线；不同配方的数字不要混着比。
- 数值对齐由 ApxInf 仓库自己的测试负责；这篇文章里的数字只证明这条链路在这台机器上跑通，并且可复现。

**归属**：决策服务与 Metal 后端来自 [ApxInf](https://github.com/infinigence/ApxInf)；模型是 [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit)（基座 [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)，Apache-2.0），[ModelScope](https://modelscope.cn/models/mlx-community/clef-flash-4bit) 上有同 id 镜像；Mario demo 的裁剪、问题、阈值与宏动作移植自 [FluidUse](https://github.com/FluidInference/FluidUse) 的 JevMarioDemo（Apache-2.0）。
