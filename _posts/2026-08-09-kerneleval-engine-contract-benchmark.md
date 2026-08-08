---
layout: post
title: "KernelEval：生成的 Kernel 能否真正替换推理引擎算子？"
date: 2026-08-09
author: Haiyan Qin
tags: [KernelEval, GPU Kernel, Benchmark, Quantized LLM, CUDA, HIP, Metal]
reading_time: 9
cover_image: /assets/project-book-cuda.png
excerpt: "KernelEval 不只检查编译、数值正确和相对框架算子的速度，而是评估生成 Kernel 是否保持真实量化引擎 contract 并击败同硬件 incumbent。"
---

# KernelEval：生成的 Kernel 能否真正替换推理引擎算子？

自动生成 GPU Kernel 的系统通常回答三个问题：代码能否编译、结果是否正确、是否比某个框架算子快。但在量化 LLM 推理引擎中，这三个条件仍不足以证明一个 kernel 可以部署。

真实引擎使用打包的低比特权重、scale 元数据、缓存布局和固定调用接口。一个 dense FP16 矩阵乘可以产生与 Q4 算子相近的输出，却需要重新解量化和不同内存布局；它并不是 drop-in replacement。

**KernelEval** 把评估对象重新定义为：在固定引擎调用点上，保持完整 contract，并在同一硬件、同一工作负载下胜过当前实现的替代 kernel。

## 103 个真实引擎 Contract

KernelEval 以 llama.cpp/ggml 为锚点构建了 **103 个机器可读任务**，覆盖四类关键算子：

- Quantized GEMM
- Flash Attention
- RMSNorm
- TopK

任务跨 CUDA、HIP 和 Metal 后端，每个任务固定输入输出、部署形状、量化元数据、支持 batch、后端接口和同硬件 incumbent 测量。

不同后端保留原生 contract，而不是通过统一的便携封装抹平差异。

## 从生成到替换的分阶段评价

每个候选依次经历：

1. 编译；
2. 正确性验证；
3. 性能测量；
4. 与同一硬件上的 incumbent 比较。

报告不仅统计正确且可测候选上的中位性能，还保留完整任务分母。这样可以避免只对成功样本求平均、忽略大量失败任务造成的幸存者偏差。

## 高正确率不代表可替换

实验中，许多设备-算子组合达到 **94% 到 100% 的正确率**，但相对 incumbent 的中位性能跨越两个数量级以上。

在两种被评估的生成配置中，当 batch 为 512 时，MI300X、A800、H800、RTX 4090 以及 Metal 设备汇总中都没有出现可替换的 Quantized GEMM 候选。

另一方面，RMSNorm 的中位性能在 MI300X 和两款 Apple 设备上超过 incumbent parity。这说明不同算子、batch 和后端的能力边界完全不同，单一汇总分数无法描述系统是否具备部署价值。

## 为什么框架等价不够？

如果基准允许改变输入布局、重新打包权重或绕过引擎缓存，那么系统可能在一个更容易的问题上取得漂亮数字，却无法替换生产路径中的算子。

KernelEval 把 contract 视为评估的一部分：候选不仅要计算同一个数学函数，还要被现有引擎以原接口调用，并在关键 batch 区间内具有竞争力。

## 一个更严格但更有用的目标

KernelEval 不是为了降低自动 kernel 生成系统的分数，而是为了提供明确的下一步目标。它可以告诉研究者：

- 系统在哪些 operator family 上已经接近替换；
- 哪些设备上正确性仍是主要问题；
- 哪些任务正确率很高，但性能与 incumbent 仍有巨大差距；
- 一个优化是否真正对应可部署价值。

当“能生成更快代码”被进一步约束为“能替换真实引擎算子”，agentic kernel optimization 才拥有可审计的部署终点。

> KernelEval 当前为 DAI 2026 Research Track 研究预览。匿名投稿原稿暂不公开上传。

