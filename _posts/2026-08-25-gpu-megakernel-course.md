---
layout: post
title: "GPU Megakernel 实战课：从 Page Pipeline 到可证伪实验"
date: 2026-08-25 10:00:00 +0800
author: Haiyan Qin
lang: zh-CN
series: gpu-systems
tags: [Megakernel, Persistent Kernel, B200, CUDA, TMA, TMEM, Evidence]
reading_time: 12
cover_image: /assets/blog-megakernel-course.png
excerpt: "一套面向新手的 19 课中文 GPU Megakernel 课程：从 page-ready、Split-KV 和 Dynamic Tail 出发，逐步走到资源封套、TMEM、2 CTA/SM、内存序和 deterministic poison litmus。"
---

我把最近围绕 B200 Megakernel 的源码审计、性能实验和失败案例整理成了一套可连续阅读的中文课程：

> **[GPU Megakernel 实战课：Lessons 36–54](https://qhy991.github.io/gpu-megakernel-course-art/)**

它不是一份“CUDA 优化技巧清单”。课程更关心三个问题：

1. GPU 上到底是谁在等待谁？
2. 一个资源 token 被释放，是否等于硬件驻留资源真的归还？
3. 一组性能数字究竟证明了什么，又不能证明什么？

![GPU Megakernel 实战课封面](/assets/blog-megakernel-course.png)

## 为什么从第 36 课开始？

课号延续真实研究日志。第 36–54 课恰好组成一条独立完整的学习线：

- 从已经测过的 Legacy 8B page-ready 优化建立直觉；
- 用 Split-KV 和 Dynamic Tail 学会拆解任务并行与冗余 work；
- 读懂 occupancy、eligible warp 和 issue active；
- 讨论最大融合、Megakernel island 与 physical cut；
- 逐项审计 shared memory、register、TMEM、grid supply 和 worker identity；
- 最后用 controlled experiment tree 与 poison litmus 学会推翻自己的协议。

更早的课程会在证据整理完成后继续公开，但这一卷不依赖前 35 课，也适合直接开始。

## 课程的四个阶段

### 1. 从真实瓶颈建立直觉

第 36–39 课使用已有报告和源码，解释为什么同样的数据量在不同 readiness 粒度下会有完全不同的等待，为什么长上下文 attention 需要 Split-KV，以及为什么 resident warp 不等于 eligible warp。

### 2. 找到正确的融合边界

第 40–43 课区分 physical fusion、CUDA Graph provider swap 与 Megakernel island。核心结论是：切开一个 kernel 会重置状态，却不会自动 right-size 下一段的资源包络。

### 3. 打开 2 CTA/SM 的五把锁

第 44–48 课逐项审计 page count、scratch、register、TMEM、grid supply 与 worker identity。`occupancy=2` 只代表“允许”，不代表真的同时驻留，更不代表更快。

### 4. 用可证伪实验替代性能故事

第 49–54 课把两岛方案拆成 R0–R5 单变量实验，并设计 deterministic READY_EARLY / ACK_EARLY poison litmus。坏协议必须以预期签名失败，测试才算真正有牙齿。

## 为什么每一课都有“证据边界”

GPU 性能研究最危险的不是数字小，而是把一个窄证据写成宽结论。例如：

- one-layer NCU 不能冒充 full-model profiler；
- CUDA Event model-forward 不能冒充 serving wall；
- source geometry 不能冒充 exact CUBIN resource；
- occupancy query 不能冒充 observed residency；
- “100 次没错”不能恢复缺失的 release/acquire 保证。

课程因此固定区分 `MEASURED`、`SOURCE-PROVEN`、`PROPOSED · UNMEASURED` 与 `WITHDRAWN`。

## 配图和源码在哪里？

课程仓库同时保存：

- 19 篇完整博客正文；
- 19 张 16:9 教学信息图；
- ImageGen 生成与修订提示词；
- GitHub Pages 站点主题；
- 对应的证据边界与历史撤回说明。

项目地址：**[qhy991/gpu-megakernel-course-art](https://github.com/qhy991/gpu-megakernel-course-art)**

建议从 **[第 36 课：Page-ready](https://qhy991.github.io/gpu-megakernel-course-art/lessons/lesson-36-page-ready/)** 顺序阅读。如果你已经在做 persistent runtime，也可以直接跳到 **[第 50 课：2 CTA/SM 的五把锁](https://qhy991.github.io/gpu-megakernel-course-art/lessons/lesson-50-five-locks-two-cta/)** 或 **[第 54 课：Poison Litmus](https://qhy991.github.io/gpu-megakernel-course-art/lessons/lesson-54-poison-litmus/)**。

