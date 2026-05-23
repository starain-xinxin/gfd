# GFD — GPU-Functional-Descriptor

High-performance scattered Host-to-Device transfer library for LLM inference workloads.

## Problem

In LLM inference , KV-cache tokens are scattered across CPU pinned memory but must be assembled contiguously in GPU memory. Standard `cudaMemcpy` per-token suffers from massive API overhead:

- 8192 scattered 4KB tokens → `cudaMemcpy(N)` achieves only **3.2 GB/s**
- Per-call overhead (~1-2 us) dominates at small granularities

## Solution

GFD offloads gather and coalescing to dedicated CPU cores while keeping the GPU free for compute:

1. **GPU kernels** write transfer descriptors into a lock-free ring buffer
2. **CPU polling thread** reads descriptors, detects contiguity patterns
3. **Parallel gather workers** (15 threads, AVX-512) assemble scattered data into contiguous staging buffers
4. **CUDA Copy Engine** DMA transfers the coalesced data to GPU in a single operation

Result: **14-53x** bandwidth improvement over per-token `cudaMemcpy`.

## Architecture

```
GPU Kernel                    CPU Polling Thread              CUDA CE
    │                              │                            │
    │ write descriptors            │ poll sequence numbers      │
    │ to ring buffer               │ detect contiguity          │
    │ ──────────────────────────►  │                            │
    │                              │ dispatch parallel gather   │
    │ overlapped compute           │ ──────────────────────►    │
    │                              │   15 AVX-512 workers       │
    │                              │   staging buffer ready     │
    │                              │                            │
    │                              │ submit coalesced DMA  ───► │ H2D transfer
    │                              │                            │
    │ poll done_idx  ◄──────────── │ update done_idx            │
    │ use prefetched data          │                            │
```

## Benchmark Results

**GPU:** NVIDIA RTX PRO 5000 72GB (Blackwell, sm_120)
**CPU:** 256 cores, 2 NUMA nodes
**Config:** 15 gather workers, 3 CE channels, 5x hugepage staging buffers
**Layout:** Tokens scattered at 2x stride in pinned CPU memory (realistic KV-cache pattern)
**Iterations:** 50 per config, 15 warmup
**Source:** `examples/04_benchmark.cu` — run with `./gfd_benchmark`

Three methods compared:
- **Memcpy(N)**: N individual `cudaMemcpyAsync` from scattered CPU addresses
- **GFD Queue**: GPU submit descriptors (fire-and-forget) + wait kernel
- **GFD Direct**: CPU direct-submit, bypass queue (parallel gather + pipelined DMA)

### Group A: Vary num_tokens (token_size = 4KB)

| Config     | Total | Memcpy(N)              | GFD Queue              | GFD Direct                 |
| ---------- | ----- | ---------------------- | ---------------------- | -------------------------- |
| 16 x 4KB   | 64KB  | 28.4 us / 2.31 GB/s    | 53.9 us / 1.21 GB/s    | **9.7 us / 6.76 GB/s**    |
| 64 x 4KB   | 256KB | 97.6 us / 2.69 GB/s    | 60.8 us / 4.31 GB/s    | **14.2 us / 18.49 GB/s**  |
| 256 x 4KB  | 1MB   | 364.4 us / 2.88 GB/s   | 723.4 us / 1.45 GB/s   | **31.0 us / 33.81 GB/s**  |
| 1024 x 4KB | 4MB   | 1376.5 us / 3.05 GB/s  | 1043.4 us / 4.02 GB/s  | **91.7 us / 45.73 GB/s**  |
| 2048 x 4KB | 8MB   | 2711.4 us / 3.09 GB/s  | 1190.4 us / 7.05 GB/s  | **170.2 us / 49.30 GB/s** |
| 4096 x 4KB | 16MB  | 5415.1 us / 3.10 GB/s  | 1268.7 us / 13.22 GB/s | **328.3 us / 51.11 GB/s** |
| 8192 x 4KB | 32MB  | 10764.8 us / 3.12 GB/s | 1909.3 us / 17.57 GB/s | **645.0 us / 52.02 GB/s** |

### Group B: Vary token_size (num_tokens = 2048)

| Config      | Total | Memcpy(N)              | GFD Queue              | GFD Direct                 |
| ----------- | ----- | ---------------------- | ---------------------- | -------------------------- |
| 2048 x 512B | 1MB   | 4199.2 us / 0.25 GB/s  | 1004.7 us / 1.04 GB/s  | **36.0 us / 29.15 GB/s**   |
| 2048 x 1KB  | 2MB   | 4199.3 us / 0.50 GB/s  | 948.1 us / 2.21 GB/s   | **55.0 us / 38.15 GB/s**   |
| 2048 x 2KB  | 4MB   | 2671.1 us / 1.57 GB/s  | 1122.0 us / 3.74 GB/s  | **93.4 us / 44.91 GB/s**   |
| 2048 x 4KB  | 8MB   | 2711.4 us / 3.09 GB/s  | 1161.2 us / 7.22 GB/s  | **171.4 us / 48.95 GB/s**  |
| 2048 x 8KB  | 16MB  | 2877.2 us / 5.83 GB/s  | 1278.5 us / 13.12 GB/s | **327.3 us / 51.26 GB/s**  |
| 2048 x 16KB | 32MB  | 3188.7 us / 10.52 GB/s | 1692.8 us / 19.82 GB/s | **642.6 us / 52.21 GB/s**  |
| 2048 x 32KB | 64MB  | 3833.3 us / 17.51 GB/s | 2442.6 us / 27.47 GB/s | **1268.9 us / 52.89 GB/s** |
| 2048 x 64KB | 128MB | 4806.9 us / 27.92 GB/s | 3979.2 us / 33.73 GB/s | **2533.9 us / 52.97 GB/s** |

### Group C: Vary num_tokens (token_size = 64KB, LLM KV-cache typical)

| Config      | Total  | Memcpy(N)              | GFD Queue              | GFD Direct                 |
| ----------- | ------ | ---------------------- | ---------------------- | -------------------------- |
| 16 x 64KB   | 1MB    | 44.2 us / 23.74 GB/s   | 72.9 us / 14.39 GB/s   | **41.4 us / 25.35 GB/s**  |
| 64 x 64KB   | 4MB    | 161.7 us / 25.94 GB/s  | 153.1 us / 27.39 GB/s  | **96.7 us / 43.38 GB/s**  |
| 256 x 64KB  | 16MB   | 622.8 us / 26.94 GB/s  | 997.6 us / 16.82 GB/s  | **360.8 us / 46.51 GB/s** |
| 1024 x 64KB | 64MB   | 2466.4 us / 27.21 GB/s | 2257.2 us / 29.73 GB/s | **1269.0 us / 52.88 GB/s** |
| 2048 x 64KB | 128MB  | 4915.9 us / 27.30 GB/s | 3270.0 us / 41.05 GB/s | **2534.8 us / 52.95 GB/s** |

### Summary

| Method                  | Best Use Case                                     | Peak Bandwidth      |
| ----------------------- | ------------------------------------------------- | ------------------- |
| **GFD Direct**          | CPU-initiated transfers (any size)                | 53 GB/s             |
| **GFD SG Warp-Spec**    | Dynamic scatter-gather (MoE routing, multi-block) | 51 GB/s (multi-block) |
| **GFD Warp-Spec**       | GPU-initiated with transfer+compute overlap       | 43.6 GB/s (pure), 33 GB/s (compute) |
| **GFD Queue**           | GPU-initiated prefetch (large tokens, GPU submit) | 41 GB/s             |
| **8-GPU Aggregate**     | Multi-GPU parallel warp-spec                      | 340 GB/s (pure)     |
| cudaMemcpy(N)           | Baseline comparison                               | 3.1 GB/s (4KB), 28 GB/s (64KB) |

**Key insights from Group C (64KB tokens):**
- GFD Queue reaches **41 GB/s** with large tokens (2048 × 64KB) — the per-descriptor overhead is amortized
- GFD Direct consistently saturates PCIe at **53 GB/s** regardless of token size
- `cudaMemcpy(N)` improves to 27 GB/s with 64KB tokens but still can't match GFD's pipelining

## Six Transfer Modes

### 1. GFD Direct (CPU-initiated)

CPU builds scatter-gather list and calls `submit_direct()`. Parallel gather workers assemble data, then a single coalesced DMA fires. Best latency and bandwidth, but requires CPU-side initiation.

### 2. GFD Queue (GPU-initiated)

GPU kernel writes descriptors into the ring buffer and continues computing. CPU polling thread detects committed entries, gathers, and DMAs asynchronously. GPU polls `done_idx` for completion. Enables **communication/computation overlap**.

### 3. GFD Warp-Specialized (High-Level API)

The recommended mode for production LLM inference. Users define only a compute functor; the framework handles all warp specialization, tile scheduling, and synchronization:

```cuda
#include <gfd/warp_spec.cuh>

// User defines compute logic only
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

// One macro generates the kernel
GFD_WARP_SPEC_KERNEL(my_kernel, MyCompute);
```

Launch via `WarpSpecSession` (single GPU) or manual `TiledQueue + CpuPollingThread` (multi-GPU):

```cpp
// Single-GPU (WarpSpecSession manages everything)
gfd::WarpSpecConfig cfg;
cfg.total_tokens = 8192;
cfg.token_size = 16384;  // 16KB per token
cfg.cpu_src = cpu_data;
cfg.gpu_dst = gpu_data;

gfd::WarpSpecSession session(cfg);
session.launch(my_kernel, MyCompute{output, 4096});
session.synchronize();
auto stats = session.get_stats();
printf("BW: %.2f GB/s\n", stats.bandwidth_gbps);
```

### 4. GFD SG Warp-Spec (Dynamic Scatter-Gather)

For MoE inference and workloads where token addresses are determined at runtime. Unlike the linear Warp-Spec mode (mode 3), SG mode does not require a fixed `cpu_base + idx * token_size` mapping. Instead, compute warps or the host dynamically submit arbitrary `(src, dst, size)` tuples via an `SGTaskQueue`.

```cuda
#include <gfd/sg_warp_spec.cuh>

// User defines compute logic receiving an SGListView
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

Launch via `SGWarpSpecSession`:

```cpp
gfd::SGWarpSpecConfig cfg;
cfg.num_compute_warps = 1;
cfg.num_blocks = 8;

gfd::SGWarpSpecSession session(cfg);

// Host pre-fill: submit SG lists before kernel launch
for (int expert = 0; expert < num_experts; expert++) {
    session.submit_sg_list(entries[expert].data(), count,
                           expert, gfd::SG_FLAG_HOST_SUBMITTED);
}

session.launch(sg_kernel, MyCompute{d_output});
session.synchronize();
```

GPU-side dynamic submission is also supported — compute warps can call `gfd::sg::sg_submit_list()` at runtime to submit new SG lists during kernel execution.

#### SG vs Linear Benchmark (RTX PRO 5000, 64 MB)

| Mode | Blocks | P50 Latency | Bandwidth |
|------|--------|-------------|-----------|
| SG | 1 | 2.16 ms | 31.0 GB/s |
| SG | 8 | 1.36 ms | 49.2 GB/s |
| Linear-opt (K=1) | 1 | 1.56 ms | 43.2 GB/s |
| Linear-opt (K=1) | 8 | 1.32 ms | 51.0 GB/s |

### 5. GFD Tiled (Low-Level, Manual Setup)

For multi-GPU or custom tile scheduling, use `TiledQueue` + `CpuPollingThread` directly (renumbered from mode 4):

```cuda
// Manual setup (multi-GPU example)
gfd::TiledQueue* tq;
cudaHostAlloc(&tq, sizeof(gfd::TiledQueue), cudaHostAllocMapped);
tq->scheduler.total_tiles = NUM_TILES;
tq->scheduler.tokens_per_tile = 128;
tq->scheduler.tokens_per_chunk = 32;
tq->scheduler.chunks_per_tile = 4;
tq->scheduler.token_size = TOKEN_SIZE;

// Device-side signal buffer (L2 polling vs PCIe)
uint64_t* d_signal;
cudaMalloc(&d_signal, gfd::MAX_TILES * sizeof(uint64_t));
tq->d_tile_chunk_done = d_signal;

gfd::CpuPollingThread poller(&tq->base, gpu_buf, cpu_buf, total_size,
    true, numa_node, 0, 0, core_base, core_count);
poller.set_tiled_queue(tq);
poller.init_copy_engine();

// Launch kernel then start poller
my_kernel<<<num_sms, block_size>>>(tq, gpu_buf, cpu_buf, compute);
poller.start();
cudaDeviceSynchronize();
poller.stop();
```

### 6. Fused Kernel Pattern (Simple, No Tiling)

```cuda
__global__ void my_kernel(...) {
    // Phase 1: Request prefetch (fire-and-forget)
    gfd::device::write_and_commit(queue, base_slot, tid, ...);

    // Phase 2: Compute while CPU transfers data
    float result = expensive_compute();

    // Phase 3: Wait for transfer completion
    if (tid == 0) gfd::device::wait_for_completion(queue, expected);
    __syncthreads();

    // Phase 4: Use prefetched data
    process(gpu_buf[tid]);
}
```

### Warp-Specialized Benchmark

**Single GPU:** RTX PRO 5000 72GB (Blackwell, sm_120), 8192 tokens × 16KB = 128 MB

| Mode | Latency | Bandwidth | Notes |
|------|---------|-----------|-------|
| **Warp-Spec Pure Transfer** | 2.94 ms | **43.6 GB/s** | NoOp functor, zero compute |
| **Warp-Spec + Compute** | 4.09 ms | **32.8 GB/s** | RMSNorm+sinf per token |
| Baseline (global wait) | 19.5 ms | 6.6 GB/s | Submit all → wait all → compute |

**8-GPU Parallel:** 8× RTX PRO 5000, 128 MB/GPU, NUMA-aware core pinning

| Mode | Aggregate BW | Scaling Eff. |
|------|-------------|-------------|
| 8-GPU Warp-Spec + Compute | **250 GB/s** | 93.6% |
| 8-GPU Pure Transfer | **340 GB/s** | 95.8% |
| Single GPU (reference) | 43.6 GB/s | — |

## Quick Start

### Warp-Specialized (Recommended for LLM inference)

```cpp
#include <gfd/gfd.h>
#include <gfd/warp_spec.cuh>

// 1. Define compute functor
struct MyCompute {
    float* output;
    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32)
            output[chunk.global_idx(i)] = chunk.data<float>(i)[0];
    }
};
GFD_WARP_SPEC_KERNEL(my_kernel, MyCompute);

// 2. Configure and launch
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

### SG Warp-Spec Mode (Dynamic scatter-gather)

```cpp
#include <gfd/gfd.h>
#include <gfd/sg_warp_spec.cuh>

// 1. Define compute functor
struct MyCompute {
    __device__ void operator()(gfd::sg_warp_spec::SGListView list) {
        // Process each entry in the completed SG list
        for (uint32_t i = list.lane_id; i < list.count; i += 32) {
            float* data = list.dst_ptr<float>(i);
            // ... compute on data ...
        }
    }
};
GFD_SG_WARP_SPEC_KERNEL(sg_kernel, MyCompute);

// 2. Configure and launch
gfd::SGWarpSpecConfig cfg;
cfg.num_compute_warps = 1;
cfg.num_blocks = 8;

gfd::SGWarpSpecSession session(cfg);

// Submit SG lists (arbitrary src/dst addresses)
session.submit_sg_list(entries, count, list_id, gfd::SG_FLAG_HOST_SUBMITTED);

session.launch(sg_kernel, MyCompute{});
session.synchronize();
printf("BW: %.2f GB/s\n", session.get_stats().bandwidth_gbps);
```

### Direct Mode (CPU-initiated, lowest latency)

```cpp
#include <gfd/gfd.h>

// 1. Initialize
gfd::StagingPool::instance().init(1, buffer_size);
gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, size,
                              true, 0, 0, 0, 0, 32);
poller.init_copy_engine();
poller.init_direct_ce();
poller.start();

// 2. Direct submit (CPU-initiated, bypasses queue)
poller.submit_direct(sg_entries, count);

// 3. Cleanup
poller.stop();
gfd::StagingPool::instance().shutdown();
```

### Queue Mode (GPU-initiated, overlapped compute)

```cpp
#include <gfd/gfd.h>
#include <gfd/device.cuh>

// ---- GPU kernels ----

// Submit kernel: writes descriptors and exits immediately (fire-and-forget)
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

// Wait kernel: lightweight single-thread completion poll
__global__ void wait_kernel(gfd::DescriptorQueue* queue, uint64_t expected_done) {
    gfd::device::wait_for_completion(queue, expected_done);
}

// ---- Host setup ----

// 1. Allocate shared queue (managed memory for GPU+CPU access)
gfd::DescriptorQueue* queue;
cudaMallocManaged(&queue, sizeof(gfd::DescriptorQueue));
memset(queue, 0, sizeof(gfd::DescriptorQueue));

// 2. Setup token metadata (scattered CPU addresses → contiguous GPU)
gfd::TokenInfo* d_tokens;  // device array
cudaMalloc(&d_tokens, N * sizeof(gfd::TokenInfo));
// ... fill with {cpu_addr, token_id, expert_id} per token ...

// 3. Initialize CPU poller
gfd::StagingPool::instance().init(1, total_size);
gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, total_size,
                              true, 0, 0, 0, 0, 32);
poller.init_copy_engine();
poller.start();

// 4. Launch transfer (GPU submits descriptors, CPU handles DMA)
uint64_t base_slot = 0;
int blocks = (N + 255) / 256;
submit_kernel<<<blocks, 256>>>(queue, d_tokens, gpu_buf, N, token_size, base_slot);
base_slot += N;
wait_kernel<<<1, 1>>>(queue, base_slot);
cudaDeviceSynchronize();

// 5. Cleanup
poller.stop();
gfd::StagingPool::instance().shutdown();
```

### Fused Kernel (prefetch + compute overlap, no tiling)

```cpp
#include <gfd/gfd.h>
#include <gfd/device.cuh>

// Single kernel: submit transfer → compute → wait → use data
__global__ void fused_kernel(
    gfd::DescriptorQueue* queue,
    gfd::TokenInfo* tokens,
    void* gpu_buf, float* output,
    int N, uint32_t token_size,
    uint64_t base_slot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < N);

    // Phase 1: Submit transfer request (fire-and-forget)
    gfd::device::write_and_commit(
        queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0,
        gpu_buf, token_size, N);

    // Phase 2: Overlapped compute (runs while CPU gathers + DMA)
    float result = 0;
    if (active) result = expensive_compute(tid);

    // Phase 3: Wait for all transfers to complete
    if (tid == 0)
        gfd::device::wait_for_completion(queue, base_slot + N);
    __syncthreads();

    // Phase 4: Use the prefetched data
    if (active) {
        float* my_token = (float*)((char*)gpu_buf + (size_t)tid * token_size);
        output[tid] = result + my_token[0];
    }
}
```

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="90;120"
make -j$(nproc)
```

Produces:

- `libgfd.a` / `libgfd.so` — static and shared libraries
- `gfd_basic_transfer` — fused kernel demo with verification
- `gfd_benchmark` — latency/bandwidth benchmark (Direct + Queue modes)
- `gfd_direct_transfer` — CPU-initiated direct transfer demo
- `gfd_gpu_planned` — GPU-planned transfer example
- `gfd_warp_spec` — warp-specialized transfer+compute demo (single GPU)
- `gfd_multi_gpu_warp_spec` — 8-GPU warp-spec with NUMA-aware pinning
- `gfd_multi_gpu_benchmark` — multi-GPU bandwidth benchmark
- `gfd_multi_gpu_direct` — multi-GPU direct transfer
- `gfd_sg_warp_spec` — SG scatter-gather warp-spec demo + benchmark
- `gfd_test_sg_e2e` — SG end-to-end test
- `gfd_test_sg_gpu_submit` — SG GPU dynamic submission test

### Requirements

- CUDA Toolkit ≥ 12.0 (≥ 13.0 for Blackwell sm_120)
- C++17 compiler with AVX-512 support
- CMake ≥ 3.18
- Linux (required for hugepages, NUMA binding, CPU pinning)
- `libnuma` (for multi-GPU examples)

## Project Structure

```
gfd/
├── include/gfd/
│   ├── gfd.h                 # Umbrella host header
│   ├── descriptor_queue.h    # Lock-free ring buffer (16384 entries)
│   ├── tiled_queue.h         # TiledQueue: per-tile completion signals + scheduler
│   ├── device.cuh            # GPU __device__ API (header-only, basic mode)
│   ├── device_primitives.cuh # GPU primitives: slot acquire, commit, wait
│   ├── warp_spec.cuh         # Warp-spec framework: GFD_WARP_SPEC_KERNEL macro
│   ├── warp_spec_session.h   # WarpSpecSession + SGWarpSpecSession (single-GPU)
│   ├── sg_task_queue.h       # SG task queue: SGList + DeviceSGEntry pool
│   ├── sg_device_primitives.cuh # SG GPU primitives: alloc, write, commit, wait
│   ├── sg_warp_spec.cuh      # SG warp-spec framework: GFD_SG_WARP_SPEC_KERNEL macro
│   ├── copy_engine.h         # Multi-stream CE DMA manager
│   ├── cpu_polling.h         # CPU polling thread + gather workers
│   ├── staging_pool.h        # Hugepage staging buffer pool (singleton)
│   ├── pcie_topology.h       # NUMA/PCIe topology discovery
│   └── log.h                 # Structured logging (compile-time levels)
├── src/
│   ├── copy_engine.cpp       # CE manager implementation (3 streams)
│   ├── cpu_polling.cpp       # Main polling loop + tile event drain
│   ├── batch_processor.cpp   # Batch processing, per-tile DMA, CE write-back
│   ├── parallel_gather.cpp   # AVX-512 parallel gather workers (up to 15)
│   ├── direct_submit.cpp     # Direct-submit fast path with pipelined gather+DMA
│   ├── warp_spec_session.cpp # WarpSpecSession lifecycle management
│   └── sg_warp_spec_session.cpp # SGWarpSpecSession lifecycle management
├── examples/
│   ├── 01_basic_transfer.cu     # Fused kernel demo with verification
│   ├── 02_direct_transfer.cu    # CPU-initiated direct transfer
│   ├── 03_gpu_planned_transfer.cu # GPU-planned transfer pattern
│   ├── 04_benchmark.cu          # Comprehensive latency/bandwidth benchmark
│   ├── 05_multi_gpu_benchmark.cu # Multi-GPU bandwidth scaling
│   ├── 06_multi_gpu_direct.cu   # Multi-GPU direct transfer
│   ├── 07_warp_spec_simple.cu   # Warp-specialized transfer+compute (single GPU)
│   ├── 08_multi_gpu_warp_spec.cu # 8-GPU warp-spec with NUMA-aware core pinning
│   └── 09_sg_warp_spec.cu       # SG scatter-gather warp-spec + benchmark
└── docs/
    ├── API_Reference.md      # Complete API documentation
    └── Architecture.md       # Implementation architecture
```

## Key Optimizations

- **Warp-level fence amortization**: only 1 `__threadfence_system()` per warp (32x reduction)
- **Warp specialization**: transfer warp + compute warps overlap DMA polling with compute at sub-tile granularity
- **Interleaved slot acquisition**: `atomicAdd` per-chunk enables concurrent multi-SM submission without head-of-line blocking
- **CE write-back signaling**: 8-byte DMA appended per tile for zero-CPU-overhead completion notification
- **Device-memory signal path**: `d_tile_chunk_done` in GPU memory enables L2-cached polling (~30ns) vs PCIe round-trip (~1500ns)
- **Per-tile progress counting**: count-based `tile_done[]` works correctly with interleaved queue entries
- **AVX-512 streaming stores**: non-temporal gather bypasses CPU cache
- **Up to 15 parallel gather workers**: saturates memory bandwidth for scattered reads
- **N-buffered staging (5x)**: overlaps gather with DMA
- **Adaptive batching**: threshold tuning based on entry size (256 default, 1024 for small entries)
- **Contiguity detection**: skips gather when data is already sequential (mega-DMA fast path)
- **Context pinning**: eliminates per-call CUDA context switch overhead
- **NUMA-aware pinning**: CPU poller + gather workers bound to NUMA-local cores for each GPU
- **Dedicated signal stream**: GPU-side signal ordering without CPU blocking via `make_stream_wait_on_all()`
- **SG Task Queue**: Two-level `SGList + DeviceSGEntry pool` enables dynamic scatter-gather without fixed address mapping
- **Multi-block SG atomics**: `atomicAdd` list claiming + `atomicCAS`-based max for entry consumption, with per-block backpressure
- **Dual completion path**: `lists_completed` (coarse, zero overhead) + `d_list_done[list_id]` (fine-grained, L2 polling)

## Multi-GPU Architecture

For multi-GPU deployments, GFD provides NUMA-aware core partitioning:

```
NUMA Node 0 (Cores 0-63)           NUMA Node 1 (Cores 64-127)
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ GPU 0: cores  0-15          │    │ GPU 4: cores 64-79          │
│ GPU 1: cores 16-31          │    │ GPU 5: cores 80-95          │
│ GPU 2: cores 32-47          │    │ GPU 6: cores 96-111         │
│ GPU 3: cores 48-63          │    │ GPU 7: cores 112-127        │
└─────────────────────────────┘    └─────────────────────────────┘
```

Each GPU gets exclusive CPU cores for its poller thread + gather workers, preventing cross-GPU contention. See `examples/08_multi_gpu_warp_spec.cu` for the full 8-GPU implementation.
