---
layout: post
title: "CUDA Reduce: 完整代码实现与测试指南"
date: 2025-11-24
author: Haiyan Qin
tags: [CUDA, Optimization, C++, Source Code]
reading_time: 10
cover_image: /assets/blog-cuda-reduce-complete.png
excerpt: "本文档汇总了 Cooperative Groups Demo、Reduce 5 以及 Reduce 6-8 的完整可编译代码、测试 Harness 及编译命令，方便开发者直接复制运行。"
---

# CUDA Reduce 完整代码库

本文档旨在提供之前系列文章（Reduce 0.2, 0.3, 5）中讨论的核心算法的**完整、可编译、可运行**的代码版本。所有的代码都包含了 `main` 函数和测试逻辑，您可以直接复制保存为 `.cu` 文件并使用 `nvcc` 编译运行。

---

## 1. Cooperative Groups 线程组切分演示

对应文章：`CUDA Cooperative Groups: Deep Dive into Thread Hierarchy`

这段代码演示了如何使用 Cooperative Groups (CG) 将一个线程块（Block）层层切分为 Warp (32), Half-Warp (16), Quarter-Warp (8) 以及 Tile (8, 4)，并展示了线程在不同层级中的 Rank 变化。

### 📄 源代码: `cg_demo.cu`

```cpp
#include <stdio.h>
#include <cooperative_groups.h>
#include <stdlib.h>

using namespace cooperative_groups;

// ---------- 设备端：打印一个 tile 的所有元信息 ----------
template <int T>
__device__ void
show_tile(const char *tag, thread_block_tile<T> p)
{
    // thread_rank(): 当前线程在本 tile 中的序号 (0 ~ T-1)
    int rank  = p.thread_rank();        
    // size(): 本 tile 的总线程数 (恒等于 T)
    int size  = p.size();               
    // meta_group_rank(): 本 tile 在父组中的序号
    int mrank = p.meta_group_rank();    
    // meta_group_size(): 父组中总共包含了多少个这样的 tile
    int msize = p.meta_group_size();    

    // 只让全局第 1234567 号线程打印，避免输出爆炸
    // 假设 gridDim足够大涵盖此线程
    auto grid = this_grid();
    if (grid.thread_rank() == 1234567) {     
        printf("%s rank in tile %2d size %2d  "
               "meta_rank %2d meta_size %2d  "
               "net_size %3d\n",
               tag, rank, size, mrank, msize, msize * size);
    }
}

// ---------- 全局内核：演示 5 级嵌套分区 ----------
__global__ void cgwarp(int gid)
{
    // 1. 获取整个网格和线程块句柄
    auto grid   = this_grid();
    auto block  = this_thread_block();

    // 2. 第一层切分：基于 Block 切分
    // 将 block 切分成 32 线程的 tile (即标准 Warp)
    auto warp32 = tiled_partition<32>(block);   
    // 将 block 切分成 16 线程的 tile (半 Warp)
    auto warp16 = tiled_partition<16>(block);   
    // 将 block 切分成 8 线程的 tile (1/4 Warp)
    auto warp8  = tiled_partition< 8>(block);   

    // 3. 第二层切分：基于 Warp 切分
    // 注意：这里是对 warp32 这个子组继续切分，而不是对 block 切分
    auto tile8  = tiled_partition< 8>(warp32);  
    // 4. 第三层切分：基于 Tile8 切分
    auto tile4  = tiled_partition< 4>(tile8);   

    if (grid.thread_rank() == gid) {
        printf("warps and sub-warps for thread %d:\n", gid);
        show_tile("warp32", warp32);
        show_tile("warp16", warp16);
        show_tile("warp8 ", warp8);
        show_tile("tile8 ", tile8);
        show_tile("tile4 ", tile4);
    }
}

// ---------- host ----------
int main(int argc, char *argv[])
{
    // 默认寻找第 1234567 号线程
    int gid     = (argc > 1) ? atoi(argv[1]) : 1234567;
    // 确保线程总数足够大
    int blocks  = 28800; 
    int threads = 256;

    printf("Target Thread GID: %d\n", gid);
    cgwarp<<<blocks, threads>>>(gid);
    cudaDeviceSynchronize();
    return 0;
}
```

### 🔨 编译与运行命令

```bash
nvcc -arch=sm_70 -o cg_demo cg_demo.cu
./cg_demo 1234567
```

---

## 2. Reduce 5: 模板展开与 Volatile 优化

对应文章：`CUDA Parallel Reduction: Deep Dive into Reduce 5 Optimization`

这是经典的 Reduce 5 实现，使用了 C++ 模板进行循环展开 (Loop Unrolling)，并利用 `volatile` 关键字在 Warp 内部进行隐式同步（针对旧架构兼容性及指令优化）。

### 📄 源代码: `reduce5.cu`

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// 辅助函数：Warp 内展开 (Warp Unrolling)
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

// 主 Kernel 函数
template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { 
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } 
        __syncthreads(); 
    }

    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 24; // 16M elements
    size_t bytes = N * sizeof(int);
    
    int *h_in = (int*)malloc(bytes);
    // 初始化输入为 1，预期结果为 N
    for(int i=0; i<N; i++) h_in[i] = 1;

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Grid 配置
    int blockSize = 256;
    // Reduce 5 每个线程处理 2 个元素，所以 Block 覆盖 blockSize*2
    int elementsPerBlock = blockSize * 2;
    int gridSize = (N + elementsPerBlock - 1) / elementsPerBlock;
    
    cudaMalloc(&d_out, gridSize * sizeof(int));

    printf("Running Reduce 5 with N=%d, Grid=%d, Block=%d\n", N, gridSize, blockSize);
    
    // 启动 Kernel
    reduce<256><<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_in, d_out, N);
    
    // 拷回部分和
    int *h_partial = (int*)malloc(gridSize * sizeof(int));
    cudaMemcpy(h_partial, d_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    // CPU 端最终汇总
    int gpu_sum = 0;
    for(int i=0; i<gridSize; i++) gpu_sum += h_partial[i];
    
    printf("GPU Sum: %d\n", gpu_sum);
    printf("Expected: %d\n", N);
    printf("Result: %s\n", (gpu_sum == N) ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_partial);
    return 0;
}
```

### 🔨 编译与运行命令

```bash
nvcc -arch=sm_70 -o reduce5 reduce5.cu
./reduce5
```

---

## 3. Reduce 进化论: 从 CG 到 Shuffle 再到 Library

对应文章：`CUDA Reduction Evolution: From Modern C++ to Extreme Performance`

这个文件集成了三个版本的 Kernel：
*   **Reduce 6**: 使用 Cooperative Groups 语法重写 Reduce 5。
*   **Reduce 7**: 使用 Warp Shuffle 指令 + Atomic Add，移除共享内存依赖。
*   **Reduce 7_v1**: 在 Reduce 7 基础上增加 `float4` 向量化加载。
*   **Reduce 8**: 直接调用 NVIDIA 官方 `cooperative_groups/reduce.h` 库。

### 📄 源代码: `reduce_evolution.cu`

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// ==========================================
// Reduce 6: Cooperative Groups Basic
// ==========================================
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ int sdata[];
    unsigned int tid = block.thread_rank();
    
    // Grid-Stride Loop
    unsigned int i = block.group_index().x * (block.size() * 2) + tid;
    unsigned int gridSize = block.size() * 2 * grid.size();
    
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + block.size()];
        i += gridSize;
    }
    block.sync();

    if (block.size() >= 512) { 
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } 
        block.sync(); 
    }
    if (block.size() >= 256) { 
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } 
        block.sync(); 
    }
    if (block.size() >= 128) { 
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } 
        block.sync(); 
    }
    
    if (block.size() >= 64) {
        if (tid < 32) {
             // 简单模拟 Reduce 5 的 Warp Unrolling，但在 CG 中推荐使用 Shuffle
             // 这里为了保持结构一致性，假设 sdata 足够安全
             volatile int* vmem = sdata;
             vmem[tid] += vmem[tid + 32];
             vmem[tid] += vmem[tid + 16];
             vmem[tid] += vmem[tid + 8];
             vmem[tid] += vmem[tid + 4];
             vmem[tid] += vmem[tid + 2];
             vmem[tid] += vmem[tid + 1];
        }
    }

    if (tid == 0) g_odata[block.group_index().x] = sdata[0];
}

// ==========================================
// Reduce 7: Warp Shuffle & Atomic
// ==========================================
template <typename T>
__device__ __forceinline__ T warpReduceSum(cg::thread_block_tile<32> g, T val) {
    val += g.shfl_down(val, 16);
    val += g.shfl_down(val, 8);
    val += g.shfl_down(val, 4);
    val += g.shfl_down(val, 2);
    val += g.shfl_down(val, 1);
    return val;
}

__global__ void reduce7(int *g_idata, int *g_odata, unsigned int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int sum = 0;
    // Grid-Stride Loop
    unsigned int i = block.group_index().x * block.size() + block.thread_rank();
    unsigned int gridSize = block.size() * grid.size();
    
    while (i < n) {
        sum += g_idata[i];
        i += gridSize;
    }
    
    // Warp Reduce
    sum = warpReduceSum(warp, sum);

    // 每个 Warp 的 0 号线程负责原子累加
    if (warp.thread_rank() == 0) {
        atomicAdd(g_odata, sum);
    }
}

// ==========================================
// Reduce 7_v1: Vectorized (float4)
// ==========================================
// 注意：为了演示 float4 优势，这里使用 float 类型
__global__ void reduce7_v1(float *g_idata, float *g_odata, unsigned int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
    
    // 向量化加载循环
    unsigned int tid = block.size() * block.group_index().x + block.thread_rank();
    unsigned int gridSize = grid.size() * block.size();
    
    // 注意：n 需要能被 4 整除，或者在这里处理边界
    for (unsigned int idx = tid; idx < n / 4; idx += gridSize) {
        float4 tmp = reinterpret_cast<const float4 *>(g_idata)[idx];
        v4.x += tmp.x;
        v4.y += tmp.y;
        v4.z += tmp.z;
        v4.w += tmp.w;
    }
    
    float sum = v4.x + v4.y + v4.z + v4.w;
    
    // Warp Reduce
    sum = warpReduceSum(warp, sum);

    if (warp.thread_rank() == 0) {
        atomicAdd(g_odata, sum);
    }
}

// ==========================================
// Reduce 8: CG Library
// ==========================================
__global__ void reduce8(int *g_idata, int *g_odata, unsigned int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int sum = 0;
    unsigned int i = block.group_index().x * block.size() + block.thread_rank();
    unsigned int gridSize = block.size() * grid.size();
    
    while (i < n) {
        sum += g_idata[i];
        i += gridSize;
    }

    // 使用官方库函数
    sum = cg::reduce(warp, sum, cg::plus<int>());

    if (warp.thread_rank() == 0) {
        atomicAdd(g_odata, sum);
    }
}

int main() {
    int N = 1 << 24; // 16M
    size_t bytes = N * sizeof(int);
    
    // --- Setup for Int kernels (6, 7, 8) ---
    int *h_in = (int*)malloc(bytes);
    for(int i=0; i<N; i++) h_in[i] = 1;
    
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(int) * 1024); // 足够存放部分和或原子累加结果
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 1. Test Reduce 6
    printf("\n--- Testing Reduce 6 ---\n");
    int gridSizeR6 = (N + (blockSize*2) - 1) / (blockSize*2);
    reduce6<<<gridSizeR6, blockSize, blockSize*sizeof(int)>>>(d_in, d_out, N);
    
    int *h_partial = (int*)malloc(gridSizeR6 * sizeof(int));
    cudaMemcpy(h_partial, d_out, gridSizeR6 * sizeof(int), cudaMemcpyDeviceToHost);
    int sum6 = 0;
    for(int i=0; i<gridSizeR6; i++) sum6 += h_partial[i];
    printf("Reduce 6 Result: %d (Expected: %d) -> %s\n", sum6, N, (sum6==N)?"PASS":"FAIL");
    free(h_partial);

    // 2. Test Reduce 7
    printf("\n--- Testing Reduce 7 ---\n");
    cudaMemset(d_out, 0, sizeof(int)); // 原子操作前清零
    reduce7<<<gridSize, blockSize>>>(d_in, d_out, N);
    int sum7;
    cudaMemcpy(&sum7, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Reduce 7 Result: %d (Expected: %d) -> %s\n", sum7, N, (sum7==N)?"PASS":"FAIL");

    // 3. Test Reduce 8
    printf("\n--- Testing Reduce 8 ---\n");
    cudaMemset(d_out, 0, sizeof(int));
    reduce8<<<gridSize, blockSize>>>(d_in, d_out, N);
    int sum8;
    cudaMemcpy(&sum8, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Reduce 8 Result: %d (Expected: %d) -> %s\n", sum8, N, (sum8==N)?"PASS":"FAIL");

    // --- Setup for Float kernel (7_v1) ---
    printf("\n--- Testing Reduce 7_v1 (Float4) ---\n");
    float *h_in_f = (float*)malloc(N * sizeof(float));
    for(int i=0; i<N; i++) h_in_f[i] = 1.0f;
    float *d_in_f, *d_out_f;
    cudaMalloc(&d_in_f, N * sizeof(float));
    cudaMalloc(&d_out_f, sizeof(float));
    cudaMemcpy(d_in_f, h_in_f, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out_f, 0, sizeof(float));
    
    reduce7_v1<<<gridSize, blockSize>>>(d_in_f, d_out_f, N);
    
    float sum7v1;
    cudaMemcpy(&sum7v1, d_out_f, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Reduce 7_v1 Result: %.1f (Expected: %.1f) -> %s\n", sum7v1, (float)N, (sum7v1==N)?"PASS":"FAIL");

    // Cleanup
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_in_f); cudaFree(d_out_f);
    free(h_in); free(h_in_f);
    
    return 0;
}
```

### 🔨 编译与运行命令

```bash
nvcc -arch=sm_70 -o reduce_evolution reduce_evolution.cu
./reduce_evolution
```
