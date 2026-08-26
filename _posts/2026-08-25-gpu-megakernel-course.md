---
layout: post
title: "GPU Megakernel 实战课：从 GPU 基础到可证伪实验"
date: 2026-08-25 10:00:00 +0800
author: Haiyan Qin
lang: zh-CN
series: gpu-systems
tags: [Megakernel, Persistent Kernel, B200, CUDA, TMA, TMEM, Evidence]
reading_time: 12
cover_image: /assets/blog-megakernel-course.png
excerpt: "一套不要求 GPU 基础的 54 课中文课程：每课先用日常问题和生活类比建立直觉，再进入 Llama、Persistent Kernel、资源与可证伪实验。"
---

我把最近围绕 B200 Megakernel 的源码审计、性能实验和失败案例整理成了一套可连续阅读的中文课程：

> **[GPU Megakernel 实战课：Lessons 1–54](https://qhy991.github.io/gpu-megakernel-course-art/)**

你不需要先懂 GPU 或 CUDA。每课开头都有三句“零基础入口”：它在解决什么、可以把它想成什么、哪些细节这次可以先忽略。理解主线后，再按需查看 **[术语表](https://qhy991.github.io/gpu-megakernel-course-art/glossary/)** 和技术推导。

它不是一份“CUDA 优化技巧清单”。课程更关心三个问题：

1. GPU 上到底是谁在等待谁？
2. 一个资源 token 被释放，是否等于硬件驻留资源真的归还？
3. 一组性能数字究竟证明了什么，又不能证明什么？

![GPU Megakernel 实战课封面](/assets/blog-megakernel-course.png)

## 现在从第 1 课开始

课程现已补齐第 1–35 课，并与原有第 36–54 课接成一条连续学习线：

- 从 GPU、SM、CTA、warp、kernel 与 Llama token 数据流建立基础；
- 沿源码区分代码融合、launch 融合与数据流融合；
- 学会阅读 NCU、PTX、SASS 和原始实验档案；
- 理解 release/acquire、epoch、ACK、驻留死锁与 Ready-aware 调度；
- 再从已经测过的 Legacy 8B page-ready 优化进入 B200 实战；
- 用 Split-KV 和 Dynamic Tail 学会拆解任务并行与冗余 work；
- 读懂 occupancy、eligible warp 和 issue active；
- 讨论最大融合、Megakernel island 与 physical cut；
- 逐项审计 shared memory、register、TMEM、grid supply 和 worker identity；
- 最后用 controlled experiment tree 与 poison litmus 学会推翻自己的协议。

第一次接触 Megakernel 的读者可以从第 1 课顺序阅读；已有 GPU 系统经验的读者也可以按主题跳转。

## 课程的五个学习段

### 1. 建立 GPU 与 Megakernel 心智模型

第 1–10 课从 GPU 执行、Llama 数据流和三种融合出发，搭建最小 Megakernel，理解一枚 token 如何在常驻执行引擎中流动。

### 2. 审计性能证据与真实优化

第 11–20 课连接 NCU、PTX、SASS、实验卡和原始归档，再用 Page-ready、Split-KV 与 Dynamic Tail 解释等待、并行度和冗余工作。

### 3. 闭合同步、生命周期与调度

第 21–35 课处理 release/acquire、epoch、ACK、驻留死锁与 Ready-aware 调度，并用 CPU 模型、poison、exact binary 和 manifest 建立可复核结论。

### 4. 从真实 B200 瓶颈建立直觉

第 36–39 课使用已有报告和源码，解释为什么同样的数据量在不同 readiness 粒度下会有完全不同的等待，为什么长上下文 attention 需要 Split-KV，以及为什么 resident warp 不等于 eligible warp。

### 5. 找到融合边界并设计可证伪实验

第 40–43 课区分 physical fusion、CUDA Graph provider swap 与 Megakernel island。核心结论是：切开一个 kernel 会重置状态，却不会自动 right-size 下一段的资源包络。

第 44–48 课逐项审计 page count、scratch、register、TMEM、grid supply 与 worker identity。`occupancy=2` 只代表“允许”，不代表真的同时驻留，更不代表更快。

第 49–54 课把两岛方案拆成 R0–R5 单变量实验，并设计 deterministic READY_EARLY / ACK_EARLY poison litmus。坏协议必须以预期签名失败，测试才算真正有牙齿。

## 为什么每一课都有“证据边界”

GPU 性能研究最危险的不是数字小，而是把一个窄证据写成宽结论。例如：

- one-layer NCU 不能冒充 full-model profiler；
- CUDA Event model-forward 不能冒充 serving wall；
- source geometry 不能冒充 exact CUBIN resource；
- occupancy query 不能冒充 observed residency；
- “100 次没错”不能恢复缺失的 release/acquire 保证。

课程因此固定区分 `MEASURED`、`SOURCE-PROVEN`、`PROPOSED · UNMEASURED` 与 `WITHDRAWN`。

这些标签描述的是结论来源，不是文章质量评分。`SOURCE-PROVEN` 可以证明 page 几何或资源公式，却不能冒充实测收益；`PROPOSED · UNMEASURED` 表示设计已经具体到可以实现和推翻，但还没有性能资格。完整判定方式见 **[证据规则](https://qhy991.github.io/gpu-megakernel-course-art/evidence/)**。

## 配图和源码在哪里？

课程仓库同时保存：

- 54 篇完整博客正文；
- 54 张 16:9 教学信息图；
- ImageGen 生成与修订提示词；
- GitHub Pages 站点主题；
- 对应的证据边界与历史撤回说明。

项目地址：**[qhy991/gpu-megakernel-course-art](https://github.com/qhy991/gpu-megakernel-course-art)**

建议从 **[第 1 课：GPU 到底在做什么？](https://qhy991.github.io/gpu-megakernel-course-art/lessons/gpu-kernel-mental-model/)** 顺序阅读。如果你已经在做 persistent runtime，也可以直接跳到 **[第 50 课：2 CTA/SM 的五把锁](https://qhy991.github.io/gpu-megakernel-course-art/lessons/lesson-50-five-locks-two-cta/)** 或 **[第 54 课：Poison Litmus](https://qhy991.github.io/gpu-megakernel-course-art/lessons/lesson-54-poison-litmus/)**。
