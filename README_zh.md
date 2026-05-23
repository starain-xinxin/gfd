# GFD — GPU-Functional-Descriptor

面向 LLM 推理负载的高性能离散 Host-to-Device 传输库。

## 问题背景

在 LLM 推理中，KV-cache 或者 token 数据分散在 CPU 锁页内存的不同位置，但需要在 GPU 显存中连续排列。逐 token 调用 `cudaMemcpy` 存在巨大的 API 开销：

- 8192 个离散 4KB token → `cudaMemcpy(N)` 仅能达到 **3.2 GB/s**
- 每次调用开销（~1-2 us）在小粒度下成为瓶颈

## 解决方案

GFD 将 gather 和合并操作卸载到专用 CPU 核心，同时保持 GPU 空闲以执行计算：

1. **GPU kernel** 将传输描述符写入无锁环形缓冲区
2. **CPU 轮询线程** 读取描述符，检测地址连续性模式
3. **并行 gather 工作线程**（15 线程，AVX-512）将离散数据汇聚到连续 staging 缓冲区
4. **CUDA Copy Engine** 通过单次 DMA 将合并后的数据传输到 GPU

结果：相比逐 token `cudaMemcpy`，带宽提升 **14-53 倍**。

## 架构

```
GPU Kernel                    CPU Polling Thread              CUDA CE
    |                              |                            |
    | write descriptors            | poll sequence numbers      |
    | to ring buffer               | detect contiguity          |
    | ---------------------------> |                            |
    |                              | dispatch parallel gather   |
    | overlapped compute           | -------------------------> |
    |                              |   15 AVX-512 workers       |
    |                              |   staging buffer ready     |
    |                              |                            |
    |                              | submit coalesced DMA  ---> | H2D transfer
    |                              |                            |
    | poll done_idx  <------------ | update done_idx            |
    | use prefetched data          |                            |
```

## 性能测试结果

**GPU:** NVIDIA RTX PRO 5000 72GB (Blackwell, sm_120)
**CPU:** 256 核，2 个 NUMA 节点
**配置:** 15 个 gather 工作线程，3 个 CE 通道，5 倍大页 staging 缓冲区
**布局:** Token 以 2x 步长分散在锁页 CPU 内存中（模拟真实 KV-cache 场景）
**迭代:** 每配置 50 次，15 次预热
**来源:** `examples/04_benchmark.cu` — 执行 `./gfd_benchmark`

三种方法对比：

- **Memcpy(N)**: N 次独立 `cudaMemcpyAsync`，从离散 CPU 地址拷贝
- **GFD Queue**: GPU 提交描述符（发射后不管）+ 等待 kernel
- **GFD Direct**: CPU 直接提交，绕过队列（并行 gather + 流水线 DMA）

### 组 A：固定 token 大小（4KB），变化 token 数量

| 配置       | 总量  | Memcpy(N)              | GFD Queue              | GFD Direct                |
| ---------- | ----- | ---------------------- | ---------------------- | ------------------------- |
| 16 x 4KB   | 64KB  | 28.4 us / 2.31 GB/s    | 53.9 us / 1.21 GB/s    | **9.7 us / 6.76 GB/s**    |
| 64 x 4KB   | 256KB | 97.6 us / 2.69 GB/s    | 60.8 us / 4.31 GB/s    | **14.2 us / 18.49 GB/s**  |
| 256 x 4KB  | 1MB   | 364.4 us / 2.88 GB/s   | 723.4 us / 1.45 GB/s   | **31.0 us / 33.81 GB/s**  |
| 1024 x 4KB | 4MB   | 1376.5 us / 3.05 GB/s  | 1043.4 us / 4.02 GB/s  | **91.7 us / 45.73 GB/s**  |
| 2048 x 4KB | 8MB   | 2711.4 us / 3.09 GB/s  | 1190.4 us / 7.05 GB/s  | **170.2 us / 49.30 GB/s** |
| 4096 x 4KB | 16MB  | 5415.1 us / 3.10 GB/s  | 1268.7 us / 13.22 GB/s | **328.3 us / 51.11 GB/s** |
| 8192 x 4KB | 32MB  | 10764.8 us / 3.12 GB/s | 1909.3 us / 17.57 GB/s | **645.0 us / 52.02 GB/s** |

### 组 B：固定 token 数量（2048），变化 token 大小

| 配置        | 总量  | Memcpy(N)              | GFD Queue              | GFD Direct                 |
| ----------- | ----- | ---------------------- | ---------------------- | -------------------------- |
| 2048 x 512B | 1MB   | 4199.2 us / 0.25 GB/s  | 1004.7 us / 1.04 GB/s  | **36.0 us / 29.15 GB/s**   |
| 2048 x 1KB  | 2MB   | 4199.3 us / 0.50 GB/s  | 948.1 us / 2.21 GB/s   | **55.0 us / 38.15 GB/s**   |
| 2048 x 2KB  | 4MB   | 2671.1 us / 1.57 GB/s  | 1122.0 us / 3.74 GB/s  | **93.4 us / 44.91 GB/s**   |
| 2048 x 4KB  | 8MB   | 2711.4 us / 3.09 GB/s  | 1161.2 us / 7.22 GB/s  | **171.4 us / 48.95 GB/s**  |
| 2048 x 8KB  | 16MB  | 2877.2 us / 5.83 GB/s  | 1278.5 us / 13.12 GB/s | **327.3 us / 51.26 GB/s**  |
| 2048 x 16KB | 32MB  | 3188.7 us / 10.52 GB/s | 1692.8 us / 19.82 GB/s | **642.6 us / 52.21 GB/s**  |
| 2048 x 32KB | 64MB  | 3833.3 us / 17.51 GB/s | 2442.6 us / 27.47 GB/s | **1268.9 us / 52.89 GB/s** |
| 2048 x 64KB | 128MB | 4806.9 us / 27.92 GB/s | 3979.2 us / 33.73 GB/s | **2533.9 us / 52.97 GB/s** |

### 组 C：固定 token 大小（64KB，LLM KV-cache 典型场景），变化 token 数量

| 配置        | 总量  | Memcpy(N)              | GFD Queue              | GFD Direct                 |
| ----------- | ----- | ---------------------- | ---------------------- | -------------------------- |
| 16 x 64KB   | 1MB   | 44.2 us / 23.74 GB/s   | 72.9 us / 14.39 GB/s   | **41.4 us / 25.35 GB/s**   |
| 64 x 64KB   | 4MB   | 161.7 us / 25.94 GB/s  | 153.1 us / 27.39 GB/s  | **96.7 us / 43.38 GB/s**   |
| 256 x 64KB  | 16MB  | 622.8 us / 26.94 GB/s  | 997.6 us / 16.82 GB/s  | **360.8 us / 46.51 GB/s**  |
| 1024 x 64KB | 64MB  | 2466.4 us / 27.21 GB/s | 2257.2 us / 29.73 GB/s | **1269.0 us / 52.88 GB/s** |
| 2048 x 64KB | 128MB | 4915.9 us / 27.30 GB/s | 3270.0 us / 41.05 GB/s | **2534.8 us / 52.95 GB/s** |

### 性能总结

| 方法                 | 最佳使用场景                              | 峰值带宽                               |
| -------------------- | ----------------------------------------- | -------------------------------------- |
| **GFD Direct**       | CPU 侧发起的传输（任意大小）              | 53 GB/s                                |
| **GFD SG Warp-Spec** | 动态 scatter-gather（MoE 路由，多 block） | 51 GB/s（多 block）                    |
| **GFD Warp-Spec**    | GPU 发起 + 传输/计算重叠                  | 43.6 GB/s（纯传输）/ 33 GB/s（含计算） |
| **GFD Queue**        | GPU 发起的预取（大 token 效果更好）       | 41 GB/s                                |
| **8-GPU 聚合**       | 多 GPU 并行 warp-spec                     | 340 GB/s（纯传输）                     |
| cudaMemcpy(N)        | 基线对比                                  | 3.1 GB/s（4KB）/ 28 GB/s（64KB）       |

**组 C 关键结论：**

- GFD Queue 在大 token（2048 × 64KB）下达到 **41 GB/s** — 每描述符开销被摊销
- GFD Direct 稳定饱和 PCIe 带宽 **53 GB/s**，与 token 大小无关
- `cudaMemcpy(N)` 在 64KB token 下提升到 27 GB/s，但仍无法匹配 GFD 的流水线化传输

## 六种传输模式

### 1. GFD Direct（CPU 发起）

CPU 构建 scatter-gather 列表并调用 `submit_direct()`。并行 gather 工作线程汇聚数据，然后发起单次合并 DMA。延迟和带宽最优，但需要 CPU 侧主动发起。

### 2. GFD Queue（GPU 发起）

GPU kernel 将描述符写入环形缓冲区后继续执行计算。CPU 轮询线程检测已提交的条目，执行 gather 和异步 DMA。GPU 轮询 `done_idx` 等待完成。支持 **通信/计算重叠**。

### 3. GFD Warp-Spec（推荐用于 LLM 推理，高级 API）

生产级 LLM 推理推荐模式。用户只需定义计算函子，框架自动处理 warp 特化、tile 调度和同步：

```cuda
#include <gfd/warp_spec.cuh>

// 用户只需定义计算逻辑
struct MyCompute {
    float* output;
    int dim;
    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32) {
            float* token = chunk.data<float>(i);
            float sum = 0;
            for (int d = 0; d < dim; d++) sum += token[d];
            output[chunk.global_idx(i)] = sum;
        }
    }
};

// 一个宏生成 kernel
GFD_WARP_SPEC_KERNEL(my_kernel, MyCompute);
```

通过 `WarpSpecSession`（单 GPU）或手动 `TiledQueue + CpuPollingThread`（多 GPU）启动：

```cpp
// 单 GPU（WarpSpecSession 管理一切）
gfd::WarpSpecConfig cfg;
cfg.total_tokens = 8192;
cfg.token_size = 16384;  // 每 token 16KB
cfg.cpu_src = cpu_data;
cfg.gpu_dst = gpu_data;

gfd::WarpSpecSession session(cfg);
session.launch(my_kernel, MyCompute{output, 4096});
session.synchronize();
auto stats = session.get_stats();
printf("BW: %.2f GB/s\n", stats.bandwidth_gbps);
```

### 4. GFD SG Warp-Spec（动态 Scatter-Gather）

适用于 MoE 推理等 token 地址在运行时动态确定的场景。与线性 Warp-Spec 模式（模式 3）不同，SG 模式不要求固定的 `cpu_base + idx * token_size` 映射，而是通过 `SGTaskQueue` 动态提交任意 `(src, dst, size)` 元组。

```cuda
#include <gfd/sg_warp_spec.cuh>

// 用户定义计算函子，接收 SGListView
struct MyCompute {
    float* output;
    __device__ void operator()(gfd::sg_warp_spec::SGListView list) {
        for (uint32_t i = list.lane_id; i < list.count; i += 32) {
            float* dst = list.dst_ptr<float>(i);
            float sum = 0;
            for (int d = 0; d < 1024; d++) sum += dst[d];
            output[list.list_id * 128 + i] = sum;
        }
    }
};

GFD_SG_WARP_SPEC_KERNEL(sg_kernel, MyCompute);
```

通过 `SGWarpSpecSession` 启动：

```cpp
gfd::SGWarpSpecConfig cfg;
cfg.num_compute_warps = 1;
cfg.num_blocks = 8;

gfd::SGWarpSpecSession session(cfg);

// Host 预填充：在 kernel 启动前提交 SG lists
for (int expert = 0; expert < num_experts; expert++) {
    session.submit_sg_list(entries[expert].data(), count,
                           expert, gfd::SG_FLAG_HOST_SUBMITTED);
}

session.launch(sg_kernel, MyCompute{d_output});
session.synchronize();
```

也支持 GPU 侧动态提交 — 计算 warp 可以在 kernel 执行期间调用 `gfd::sg::sg_submit_list()` 提交新的 SG lists。

#### SG vs Linear 性能测试（RTX PRO 5000，64 MB）

| 模式             | Block 数 | P50 延迟 | 带宽      |
| ---------------- | -------- | -------- | --------- |
| SG               | 1        | 2.16 ms  | 31.0 GB/s |
| SG               | 8        | 1.36 ms  | 49.2 GB/s |
| Linear-opt (K=1) | 1        | 1.56 ms  | 43.2 GB/s |
| Linear-opt (K=1) | 8        | 1.32 ms  | 51.0 GB/s |

### 5. 低级 Tiled API（多 GPU，手动设置）

多 GPU 或自定义 tile 调度场景，直接使用 `TiledQueue + CpuPollingThread`：

```cuda
// 手动设置（多 GPU 示例）
gfd::TiledQueue* tq;
cudaHostAlloc(&tq, sizeof(gfd::TiledQueue), cudaHostAllocMapped);
tq->scheduler.total_tiles = NUM_TILES;
tq->scheduler.tokens_per_tile = 128;
tq->scheduler.tokens_per_chunk = 32;
tq->scheduler.chunks_per_tile = 4;
tq->scheduler.token_size = TOKEN_SIZE;

// 设备侧信号缓冲区（L2 轮询 vs PCIe）
uint64_t* d_signal;
cudaMalloc(&d_signal, gfd::MAX_TILES * sizeof(uint64_t));
tq->d_tile_chunk_done = d_signal;

gfd::CpuPollingThread poller(&tq->base, gpu_buf, cpu_buf, total_size,
    true, numa_node, 0, 0, core_base, core_count);
poller.set_tiled_queue(tq);
poller.init_copy_engine();

// 启动 kernel 后启动 poller
my_kernel<<<num_sms, block_size>>>(tq, gpu_buf, cpu_buf, compute);
poller.start();
cudaDeviceSynchronize();
poller.stop();
```

### 6. 融合 Kernel 模式（简单，无 Tiling）

```cuda
__global__ void my_kernel(...) {
    // 阶段 1：请求预取（发射后不管）
    gfd::device::write_and_commit(queue, base_slot, tid, ...);

    // 阶段 2：CPU 传输数据期间执行计算
    float result = expensive_compute();

    // 阶段 3：等待传输完成
    if (tid == 0) gfd::device::wait_for_completion(queue, expected);
    __syncthreads();

    // 阶段 4：使用预取数据
    process(gpu_buf[tid]);
}
```

### Warp-Spec 性能测试

**单 GPU:** RTX PRO 5000 72GB (Blackwell, sm_120), 8192 tokens × 16KB = 128 MB

| 模式                 | 延迟    | 带宽          | 备注                       |
| -------------------- | ------- | ------------- | -------------------------- |
| **Warp-Spec 纯传输** | 2.94 ms | **43.6 GB/s** | NoOp 函子，零计算          |
| **Warp-Spec + 计算** | 4.09 ms | **32.8 GB/s** | RMSNorm+sinf 每 token      |
| 基线（全局等待）     | 19.5 ms | 6.6 GB/s      | 全部提交 → 全部等待 → 计算 |

**8 GPU 并行:** 8× RTX PRO 5000, 128 MB/GPU, NUMA 感知核心绑定

| 模式                   | 聚合带宽     | 扩展效率 |
| ---------------------- | ------------ | -------- |
| 8-GPU Warp-Spec + 计算 | **250 GB/s** | 93.6%    |
| 8-GPU 纯传输           | **340 GB/s** | 95.8%    |
| 单 GPU（参考）         | 43.6 GB/s    | —        |

## 快速开始

### Warp-Spec 模式（推荐）

```cpp
#include <gfd/gfd.h>
#include <gfd/warp_spec.cuh>

// 1. 定义计算函子
struct MyCompute {
    float* output;
    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32)
            output[chunk.global_idx(i)] = chunk.data<float>(i)[0];
    }
};
GFD_WARP_SPEC_KERNEL(my_kernel, MyCompute);

// 2. 配置并启动
gfd::WarpSpecConfig cfg;
cfg.total_tokens = 8192;
cfg.token_size = 16384;
cfg.cpu_src = cpu_pinned_buf;
cfg.gpu_dst = gpu_buf;

gfd::WarpSpecSession session(cfg);
session.launch(my_kernel, MyCompute{d_output});
session.synchronize();
printf("BW: %.2f GB/s\n", session.get_stats().bandwidth_gbps);
```

### SG Warp-Spec 模式（动态 scatter-gather）

```cpp
#include <gfd/gfd.h>
#include <gfd/sg_warp_spec.cuh>

// 1. 定义计算函子
struct MyCompute {
    __device__ void operator()(gfd::sg_warp_spec::SGListView list) {
        for (uint32_t i = list.lane_id; i < list.count; i += 32) {
            float* data = list.dst_ptr<float>(i);
            // ... 在数据上计算 ...
        }
    }
};
GFD_SG_WARP_SPEC_KERNEL(sg_kernel, MyCompute);

// 2. 配置并启动
gfd::SGWarpSpecConfig cfg;
cfg.num_compute_warps = 1;
cfg.num_blocks = 8;

gfd::SGWarpSpecSession session(cfg);

// 提交 SG lists（任意 src/dst 地址）
session.submit_sg_list(entries, count, list_id, gfd::SG_FLAG_HOST_SUBMITTED);

session.launch(sg_kernel, MyCompute{});
session.synchronize();
printf("BW: %.2f GB/s\n", session.get_stats().bandwidth_gbps);
```

### Direct 模式（CPU 发起，最低延迟）

```cpp
#include <gfd/gfd.h>

// 1. 初始化
gfd::StagingPool::instance().init(1, buffer_size);
gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, size,
                              true, 0, 0, 0, 0, 32);
poller.init_copy_engine();
poller.init_direct_ce();
poller.start();

// 2. Direct 提交（CPU 发起，绕过队列）
poller.submit_direct(sg_entries, count);

// 3. 清理
poller.stop();
gfd::StagingPool::instance().shutdown();
```

### Queue 模式（GPU 发起，通信/计算重叠）

```cpp
#include <gfd/gfd.h>
#include <gfd/device.cuh>

// ---- GPU kernel ----

// 提交 kernel：写入描述符后立即退出（发射后不管）
__global__ void submit_kernel(
    gfd::DescriptorQueue* queue,
    gfd::TokenInfo* tokens,
    void* gpu_buffer,
    int num_tokens,
    uint32_t token_size,
    uint64_t base_slot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < num_tokens);
    gfd::device::write_and_commit(
        queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0,
        gpu_buffer, token_size, num_tokens);
}

// 等待 kernel：轻量单线程完成轮询
__global__ void wait_kernel(gfd::DescriptorQueue* queue, uint64_t expected_done) {
    gfd::device::wait_for_completion(queue, expected_done);
}

// ---- Host 设置 ----

// 1. 分配共享队列（managed memory，GPU+CPU 均可访问）
gfd::DescriptorQueue* queue;
cudaMallocManaged(&queue, sizeof(gfd::DescriptorQueue));
memset(queue, 0, sizeof(gfd::DescriptorQueue));

// 2. 设置 token 元数据（离散 CPU 地址 → 连续 GPU）
gfd::TokenInfo* d_tokens;
cudaMalloc(&d_tokens, N * sizeof(gfd::TokenInfo));
// ... 每 token 填入 {cpu_addr, token_id, expert_id} ...

// 3. 初始化 CPU poller
gfd::StagingPool::instance().init(1, total_size);
gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, total_size,
                              true, 0, 0, 0, 0, 32);
poller.init_copy_engine();
poller.start();

// 4. 启动传输（GPU 提交描述符，CPU 处理 DMA）
uint64_t base_slot = 0;
int blocks = (N + 255) / 256;
submit_kernel<<<blocks, 256>>>(queue, d_tokens, gpu_buf, N, token_size, base_slot);
base_slot += N;
wait_kernel<<<1, 1>>>(queue, base_slot);
cudaDeviceSynchronize();

// 5. 清理
poller.stop();
gfd::StagingPool::instance().shutdown();
```

### 融合 Kernel（预取 + 计算重叠，无 Tiling）

```cpp
#include <gfd/gfd.h>
#include <gfd/device.cuh>

// 单 kernel：提交传输 → 计算 → 等待 → 使用数据
__global__ void fused_kernel(
    gfd::DescriptorQueue* queue,
    gfd::TokenInfo* tokens,
    void* gpu_buf, float* output,
    int N, uint32_t token_size,
    uint64_t base_slot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < N);

    // 阶段 1：提交传输请求（发射后不管）
    gfd::device::write_and_commit(
        queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0,
        gpu_buf, token_size, N);

    // 阶段 2：重叠计算（CPU gather + DMA 期间执行）
    float result = 0;
    if (active) result = expensive_compute(tid);

    // 阶段 3：等待所有传输完成
    if (tid == 0)
        gfd::device::wait_for_completion(queue, base_slot + N);
    __syncthreads();

    // 阶段 4：使用预取数据
    if (active) {
        float* my_token = (float*)((char*)gpu_buf + (size_t)tid * token_size);
        output[tid] = result + my_token[0];
    }
}
```

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="90;120"
make -j$(nproc)
```

产物：

- `libgfd.a` / `libgfd.so` — 静态库和动态库
- `gfd_basic_transfer` — 融合 kernel 功能演示（含验证）
- `gfd_benchmark` — 延迟/带宽性能基准测试（Direct + Queue 模式）
- `gfd_direct_transfer` — CPU 发起的 direct 传输演示
- `gfd_gpu_planned` — GPU 规划传输示例
- `gfd_warp_spec` — 单 GPU Warp-Spec 传输+计算示例
- `gfd_multi_gpu_warp_spec` — 8 GPU NUMA 感知 warp-spec 示例
- `gfd_multi_gpu_benchmark` — 多 GPU 带宽扩展测试
- `gfd_multi_gpu_direct` — 多 GPU direct 传输
- `gfd_sg_warp_spec` — SG scatter-gather warp-spec 示例 + 性能测试
- `gfd_test_sg_e2e` — SG 端到端测试
- `gfd_test_sg_gpu_submit` — SG GPU 动态提交测试

### 环境要求

- CUDA Toolkit ≥ 12.0（Blackwell sm_120 需要 ≥ 13.0）
- 支持 AVX-512 的 C++17 编译器
- CMake ≥ 3.18
- Linux（必需，支持大页、NUMA 绑定、CPU 亲和性绑定）
- `libnuma`（多 GPU 示例需要）

## 项目结构

```
gfd/
├── include/gfd/
│   ├── gfd.h                 # 统一 Host 头文件
│   ├── descriptor_queue.h    # 无锁环形缓冲区（16384 条目）
│   ├── tiled_queue.h         # TiledQueue: 分片完成信号 + 调度器
│   ├── device.cuh            # GPU __device__ API（基础模式）
│   ├── device_primitives.cuh # GPU 原语：slot 获取、提交、等待
│   ├── warp_spec.cuh         # Warp-Spec 框架：GFD_WARP_SPEC_KERNEL 宏
│   ├── warp_spec_session.h   # WarpSpecSession + SGWarpSpecSession（单 GPU）
│   ├── sg_task_queue.h       # SG 任务队列：SGList + DeviceSGEntry 池
│   ├── sg_device_primitives.cuh # SG GPU 原语：分配、写入、提交、等待
│   ├── sg_warp_spec.cuh      # SG warp-spec 框架：GFD_SG_WARP_SPEC_KERNEL 宏
│   ├── copy_engine.h         # 多流 CE DMA 管理器
│   ├── cpu_polling.h         # CPU 轮询线程 + gather 工作线程
│   ├── staging_pool.h        # 大页 staging 缓冲池（单例）
│   ├── pcie_topology.h       # NUMA/PCIe 拓扑发现
│   └── log.h                 # 结构化日志
├── src/
│   ├── copy_engine.cpp       # CE 管理器实现（3 流）
│   ├── cpu_polling.cpp       # 轮询循环 + tile 事件排空
│   ├── batch_processor.cpp   # 批处理、per-tile DMA、CE 回写信号
│   ├── parallel_gather.cpp   # AVX-512 并行 gather 工作线程（最多 15 个）
│   ├── direct_submit.cpp     # Direct 提交快速路径
│   ├── warp_spec_session.cpp # WarpSpecSession 生命周期管理
│   └── sg_warp_spec_session.cpp # SGWarpSpecSession 生命周期管理
├── examples/
│   ├── 01_basic_transfer.cu     # 融合 kernel 示例（含验证）
│   ├── 02_direct_transfer.cu    # CPU 发起的 direct 传输
│   ├── 03_gpu_planned_transfer.cu # GPU 规划传输模式
│   ├── 04_benchmark.cu          # 综合延迟/带宽基准测试
│   ├── 05_multi_gpu_benchmark.cu # 多 GPU 带宽扩展
│   ├── 06_multi_gpu_direct.cu   # 多 GPU direct 传输
│   ├── 07_warp_spec_simple.cu   # 单 GPU Warp-Spec 传输+计算
│   ├── 08_multi_gpu_warp_spec.cu # 8-GPU NUMA 感知 warp-spec
│   └── 09_sg_warp_spec.cu       # SG scatter-gather warp-spec + 性能测试
└── docs/
    ├── API_Reference.md      # 完整 API 文档
    └── Architecture.md       # 实现架构文档
```

## 核心优化

- **Warp 级 fence 摊销**：每 warp 仅 1 次 `__threadfence_system()`（降低 32 倍开销）
- **Warp 特化**：传输 warp + 计算 warp 以子 tile 粒度重叠 DMA 轮询与计算
- **交错 slot 获取**：每 chunk 使用 `atomicAdd`，多 SM 并发提交无队头阻塞
- **CE 回写信号**：每 tile 追加 8 字节 DMA，零 CPU 开销完成通知
- **设备内存信号路径**：`d_tile_chunk_done` 在 GPU 内存中，L2 缓存轮询（~30ns）vs PCIe（~1500ns）
- **Per-tile 进度计数**：基于计数的 `tile_done[]`，正确处理交错队列条目
- **AVX-512 流式存储**：non-temporal gather 绕过 CPU 缓存
- **最多 15 个并行 gather 工作线程**：饱和内存带宽处理离散读取
- **N 倍缓冲 staging（5x）**：gather 与 DMA 重叠执行
- **自适应批处理**：根据条目大小调整刷新阈值
- **连续性检测**：数据已连续时跳过 gather（mega-DMA 快速路径）
- **上下文固定**：消除每次调用的 CUDA 上下文切换开销
- **NUMA 感知绑定**：CPU 轮询线程 + gather 工作线程绑定到每个 GPU 的本地 NUMA 核心
- **专用信号流**：通过 `make_stream_wait_on_all()` 实现 GPU 侧信号排序，无需 CPU 阻塞
- **SG 任务队列**：两级 `SGList + DeviceSGEntry 池` 结构，支持动态 scatter-gather，无需固定地址映射
- **多 block SG 原子操作**：`atomicAdd` 列表领取 + `atomicCAS`-based max 条目消费追踪，含 per-block 背压
- **双路完成信号**：`lists_completed`（粗粒度，零开销）+ `d_list_done[list_id]`（细粒度，L2 轮询）

## 多 GPU 架构

GFD 支持 NUMA 感知的多 GPU 部署：

```
NUMA 节点 0（核心 0-63）              NUMA 节点 1（核心 64-127）
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ GPU 0: 核心  0-15           │    │ GPU 4: 核心 64-79           │
│ GPU 1: 核心 16-31           │    │ GPU 5: 核心 80-95           │
│ GPU 2: 核心 32-47           │    │ GPU 6: 核心 96-111          │
│ GPU 3: 核心 48-63           │    │ GPU 7: 核心 112-127         │
└─────────────────────────────┘    └─────────────────────────────┘
```

每个 GPU 获得独占 CPU 核心（poller 线程 + gather 工作线程），避免跨 GPU 资源争用。

8 GPU 聚合带宽达 **340 GB/s**（纯传输），扩展效率 95.8%。详见 `examples/08_multi_gpu_warp_spec.cu`。
