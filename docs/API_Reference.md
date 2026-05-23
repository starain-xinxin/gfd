# GFD API Reference

**GFD (GPU-Functional-Descriptor)** — High-performance scattered Host-to-Device transfer library for LLM inference workloads.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Headers](#headers)
- [Namespace: gfd](#namespace-gfd)
  - [Constants](#constants)
  - [Structures](#structures)
  - [Classes](#classes)
- [Namespace: gfd::device](#namespace-gfddevice)
  - [Device Functions](#device-functions)
- [Namespace: gfd::warp\_spec](#namespace-gfdwarp_spec)
  - [Warp-Spec Framework](#warp-spec-framework)
- [Namespace: gfd::sg](#namespace-gfdsg)
  - [SG Device Primitives](#sg-device-primitives)
- [Namespace: gfd::sg\_warp\_spec](#namespace-gfdsg_warp_spec)
  - [SG Warp-Spec Framework](#sg-warp-spec-framework)
- [Usage Patterns](#usage-patterns)
- [Build Integration](#build-integration)

---

## Quick Start

```cpp
// Host code
#include <gfd/gfd.h>

// Device code (fused kernels)
#include <gfd/device.cuh>

// Device code (warp-specialized kernels)
#include <gfd/warp_spec.cuh>
```

Minimal example: transfer 1024 scattered tokens (4KB each) from CPU to GPU.

```cpp
#include <gfd/gfd.h>
#include <gfd/device.cuh>

__global__ void transfer_kernel(gfd::DescriptorQueue* queue,
                                gfd::TokenInfo* tokens,
                                void* gpu_buf, int N, uint32_t size,
                                uint64_t base_slot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < N);
    gfd::device::write_and_commit(queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0, gpu_buf, size, N);
}

__global__ void wait_kernel(gfd::DescriptorQueue* queue, uint64_t done) {
    gfd::device::wait_for_completion(queue, done);
}

int main() {
    // 1. Allocate shared queue (managed memory)
    gfd::DescriptorQueue* queue;
    cudaMallocManaged(&queue, sizeof(gfd::DescriptorQueue));
    memset(queue, 0, sizeof(gfd::DescriptorQueue));

    // 2. Initialize staging pool and CPU poller
    gfd::StagingPool::instance().init(1, 4 * 1024 * 1024);
    gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, total_size,
                                  true, 0, 0, 0, 0, 32);
    poller.init_copy_engine();
    poller.start();

    // 3. Launch GPU kernels
    uint64_t base_slot = 0;
    transfer_kernel<<<blocks, threads>>>(queue, tokens, gpu_buf, N, 4096, base_slot);
    base_slot += N;
    wait_kernel<<<1, 1>>>(queue, base_slot);
    cudaDeviceSynchronize();

    // 4. Cleanup
    poller.stop();
    gfd::StagingPool::instance().shutdown();
}
```

---

## Headers

| Header | Scope | Description |
|--------|-------|-------------|
| `<gfd/gfd.h>` | Host | Umbrella header — includes all host-side components |
| `<gfd/log.h>` | Shared | Structured logging macros (`GFD_LOG_ERROR`, `GFD_LOG_INFO`, `GFD_LOG_DEBUG`) |
| `<gfd/warp_spec.cuh>` | Device | **Warp-spec framework**: `GFD_WARP_SPEC_KERNEL` macro, `ChunkView`, `TileContext`, tile scheduling |
| `<gfd/warp_spec_session.h>` | Host | `WarpSpecSession` + `WarpSpecConfig` (single-GPU lifecycle manager) |
| `<gfd/sg_task_queue.h>` | Shared | SG task queue structures: `DeviceSGEntry`, `SGList`, `SGTaskQueue`, `sg_compat::linear_to_sg_entries` |
| `<gfd/sg_device_primitives.cuh>` | Device | SG GPU primitives: `sg_alloc_entries`, `sg_write_entries`, `sg_commit_list`, `sg_submit_list`, `sg_wait_list_done` |
| `<gfd/sg_warp_spec.cuh>` | Device | SG warp-spec framework: `GFD_SG_WARP_SPEC_KERNEL` macro, `SGListView`, transfer warp loop |
| `<gfd/device.cuh>` | Device | `__device__` inline functions for basic fused kernels |
| `<gfd/device_primitives.cuh>` | Device | Low-level warp-collective primitives: `acquire_tile`, `acquire_chunk_slots`, `write_chunk`, `commit_chunk`, `wait_chunk_done`, `submit_chunk`, `submit_tile` |
| `<gfd/descriptor_queue.h>` | Shared | Ring buffer definitions shared between GPU and CPU |
| `<gfd/tiled_queue.h>` | Shared | Extended queue with per-tile completion signals + scheduler |
| `<gfd/copy_engine.h>` | Host | CUDA Driver API DMA engine manager |
| `<gfd/cpu_polling.h>` | Host | CPU polling thread with parallel gather workers |
| `<gfd/staging_pool.h>` | Host | Pre-allocated hugepage staging buffer pool |
| `<gfd/pcie_topology.h>` | Host | NUMA/PCIe topology discovery utility |

---

## Namespace: gfd

### Constants

#### Descriptor Queue Configuration

```cpp
constexpr int QUEUE_SIZE = 16384;          // Ring buffer capacity (entries)
constexpr int MAX_BATCH_SIZE = 8192;       // Max entries per batch processing
constexpr int BATCH_THRESHOLD = 4096;      // Default batch flush threshold
constexpr int BATCH_THRESHOLD_SMALL = 1024; // Flush threshold for small entries (<=2KB)
constexpr size_t ADAPTIVE_BATCH_TARGET_BYTES = 1024 * 1024; // 1MB target per batch
```

#### Descriptor Flags

```cpp
constexpr uint32_t FLAG_NONE = 0;           // No special flags
constexpr uint32_t FLAG_LAST_IN_BATCH = 1;  // Last entry in a logical batch (triggers flush)
constexpr uint32_t FLAG_URGENT = 2;         // Priority flush (bypass threshold accumulation)
constexpr uint32_t FLAG_LAST_IN_TILE = 4;   // Chunk boundary marker (every C-th entry)
constexpr uint32_t FLAG_LAST_CHUNK_IN_TILE = 8;   // Tile boundary marker (final entry of tile, triggers poller flush)
```

#### Tiled Queue Constants

```cpp
constexpr int MAX_TILES = 1024;                   // Maximum number of tiles supported
```

#### Copy Engine Configuration

```cpp
constexpr int MAX_CE_CHANNELS = 3;            // Max concurrent DMA streams
constexpr int MAX_SG_ENTRIES_PER_BATCH = 8192; // Max scatter-gather entries per submission
```

#### CPU Polling Configuration

```cpp
constexpr int MAX_GATHER_WORKERS = 15;  // Maximum gather worker threads
constexpr int NUM_GATHER_WORKERS = 7;   // Legacy default (unused in new code)
```

#### Warp-Spec Configuration

```cpp
constexpr int MAX_CHUNKS_PER_TILE = 16;  // Maximum chunks per tile (shared memory layout limit)
```

#### SG Task Queue Constants

```cpp
constexpr int MAX_SG_LISTS = 512;            // SG list ring buffer capacity
constexpr int MAX_SG_POOL_ENTRIES = 16384;   // SG entry pool capacity
constexpr uint32_t SG_FLAG_NONE = 0;         // No special flags
constexpr uint32_t SG_FLAG_HOST_SUBMITTED = 1; // List was submitted by host (not GPU)
```

#### Logging Macros (`<gfd/log.h>`)

```cpp
// Compile-time log level: -DGFD_LOG_LEVEL=N (0=silent, 1=error, 2=info+error [default], 3=debug+info+error)
GFD_LOG_ERROR(fmt, ...)   // Level >= 1, prefix: "[GFD ERROR] "
GFD_LOG_INFO(fmt, ...)    // Level >= 2, prefix: "[GFD] "
GFD_LOG_DEBUG(fmt, ...)   // Level >= 3, prefix: "[GFD DBG] "
```

---

### Structures

#### `Descriptor`

64-byte cache-line aligned descriptor for a single H2D transfer request.

```cpp
struct __align__(64) Descriptor {
    uint64_t src_addr;          // CPU source address (pinned memory)
    uint64_t dst_addr;          // GPU destination address
    uint32_t size;              // Transfer size in bytes
    uint32_t flags;             // Control flags (FLAG_*)
    uint64_t user_data;         // User-defined payload (e.g., token_id | expert_id << 32)
    volatile uint64_t sequence; // Commit marker: (slot+1) when ready, 0 when consumed
};
static_assert(sizeof(Descriptor) == 64);
```

#### `DescriptorQueue`

Lock-free ring buffer shared between GPU (producer) and CPU (consumer).

```cpp
struct DescriptorQueue {
    Descriptor entries[QUEUE_SIZE]; // Ring buffer entries
    volatile uint64_t write_idx;   // Next write slot (GPU: atomicAdd)
    volatile uint64_t read_idx;    // Next read slot (CPU updates)
    volatile uint64_t done_idx;    // Completion marker (CPU updates, GPU polls)
    uint8_t _padding[64 - 24];    // Cache-line alignment padding
};
```

**Synchronization Protocol:**
- GPU writes descriptor fields → `__threadfence_system()` → writes `sequence = slot + 1`
- CPU polls `sequence`, processes descriptor, writes `sequence = 0`
- CPU advances `done_idx` after DMA completion
- GPU polls `done_idx` for completion notification

#### `TokenInfo`

Per-token metadata for GPU-side descriptor construction.

```cpp
struct TokenInfo {
    uint64_t cpu_addr;   // Source address in CPU pinned memory
    uint32_t token_id;   // Token identifier
    uint32_t expert_id;  // Expert/shard identifier (for MoE routing)
};
```

#### `TileScheduler`

Global tile dispatch and configuration. Allocated in host-mapped memory for GPU `atomicAdd` access.

```cpp
struct __align__(64) TileScheduler {
    volatile uint32_t next_tile;      // GPU atomicAdd to acquire tile_id
    uint32_t total_tiles;             // Total number of tiles in this workload
    uint32_t tokens_per_tile;         // Tokens assigned to each tile
    uint32_t tokens_per_chunk;        // Tokens per sub-tile chunk
    uint32_t chunks_per_tile;         // = tokens_per_tile / tokens_per_chunk
    uint32_t token_size;              // Bytes per token
    uint8_t _pad[40];                 // Pad to 64 bytes
};
static_assert(sizeof(TileScheduler) == 64);
```

#### `TiledQueue`

Extended descriptor queue with tile scheduling and per-tile completion signaling for warp-specialized transfer-compute overlap.

```cpp
struct TiledQueue {
    DescriptorQueue base;                          // Lock-free descriptor ring buffer
    TileScheduler scheduler;                       // Global tile dispatch
    volatile uint64_t tile_chunk_done[MAX_TILES];  // Cumulative tokens completed per tile (host-mapped fallback)
    uint64_t* d_tile_chunk_done;                   // Device memory signal buffer (nullptr = use host-mapped)
};
```

**Signaling Semantics:**
- `tile_chunk_done[tile_id]` stores the **cumulative number of tokens completed** for that tile
- GPU polls: `tile_chunk_done[my_tile] >= threshold` to know when a chunk/tile is ready
- Supports interleaved (atomicAdd) slot acquisition from concurrent SMs
- When `d_tile_chunk_done != nullptr`, GPU polls device memory (L2 cached, ~30ns) instead of host-mapped memory (~1500ns PCIe round-trip)

**Signal Path (Device Memory):**
1. CPU poller processes entries, submits per-tile DMA
2. After DMA, CPU issues CE write-back: 8-byte copy from pinned `tile_signal_buf_[tile_id]` → `d_tile_chunk_done[tile_id]`
3. GPU polls `d_tile_chunk_done[tile_id]` via L2 cache (no PCIe latency)
4. This provides ~50x faster signal delivery vs host-mapped polling

**Allocation:** Must use `cudaHostAlloc` with `cudaHostAllocMapped` for CPU+GPU shared access.

**Setup Pattern:**
```cpp
gfd::TiledQueue* tq;
cudaHostAlloc(&tq, sizeof(gfd::TiledQueue), cudaHostAllocMapped);
memset(tq, 0, sizeof(gfd::TiledQueue));
tq->scheduler.total_tiles = NUM_TILES;
tq->scheduler.tokens_per_tile = TOKENS_PER_TILE;
tq->scheduler.tokens_per_chunk = TOKENS_PER_CHUNK;
tq->scheduler.chunks_per_tile = CHUNKS_PER_TILE;
tq->scheduler.token_size = TOKEN_SIZE;

// Optional: device-memory signals for low-latency GPU polling
uint64_t* d_signal;
cudaMalloc(&d_signal, gfd::MAX_TILES * sizeof(uint64_t));
cudaMemset(d_signal, 0, gfd::MAX_TILES * sizeof(uint64_t));
tq->d_tile_chunk_done = d_signal;
```

---

#### `SGEntry`

Scatter-gather entry for DMA submission.

```cpp
struct SGEntry {
    CUdeviceptr dst;     // GPU destination (device pointer)
    const void* src;     // CPU source (host pointer)
    size_t size;         // Transfer size in bytes
};
```

#### `DeviceSGEntry`

A single scatter-gather entry with arbitrary source and destination addresses. Used in the SG task queue for dynamic address submission.

```cpp
struct __align__(8) DeviceSGEntry {
    uint64_t src_addr;   // CPU source address
    uint64_t dst_addr;   // GPU destination address
    uint32_t size;       // Transfer size in bytes
    uint32_t tag;        // User grouping/tracking tag
};
static_assert(sizeof(DeviceSGEntry) == 24);
```

#### `SGList`

Header for a batch of entries in the SG entry pool. Forms a ring buffer in `SGTaskQueue::lists[]`.

```cpp
struct __align__(16) SGList {
    uint32_t pool_offset;        // Start index in entry pool
    uint32_t count;              // Number of entries in this list
    uint32_t list_id;            // User-assigned ID for completion tracking
    uint32_t flags;              // SG_FLAG_*
    volatile uint64_t sequence;  // Commit marker: slot+1 when ready, 0 when consumed
    uint64_t _pad;               // Pad to 32 bytes
};
static_assert(sizeof(SGList) == 32);
```

#### `SGTaskQueue`

The main scatter-gather task queue, allocated in host-mapped memory (`cudaHostAllocMapped`) for shared CPU+GPU access.

```cpp
struct SGTaskQueue {
    SGList lists[MAX_SG_LISTS];              // SG list ring buffer (512 slots)
    DeviceSGEntry entries[MAX_SG_POOL_ENTRIES]; // Entry pool (16384 entries)

    // Producer coordination (cache-line aligned)
    alignas(64) volatile uint64_t list_alloc_idx;    // Producer: atomicAdd to allocate list slot
    alignas(64) volatile uint64_t entry_alloc_idx;   // Producer: atomicAdd to allocate entries

    // Consumer coordination (cache-line aligned)
    alignas(64) volatile uint64_t list_read_idx;     // Transfer warp: atomicAdd to claim next list
    alignas(64) volatile uint64_t entry_consumed_idx; // Transfer warp: atomicCAS-based max

    // Completion signaling
    alignas(64) volatile uint64_t lists_completed;   // Coarse counter (CPU poller increments)
    volatile uint32_t terminate;                      // Termination flag

    uint64_t* d_list_done;  // Device memory per-list completion (cudaMalloc'd separately)
};
```

**Two-Level Structure:**
- `lists[]`: ring buffer of list headers — each describes a batch of entries
- `entries[]`: flat pool of SG entries — lists reference contiguous ranges via `pool_offset` + `count`

**Completion Dual Path:**
- `lists_completed`: coarse counter incremented by CPU poller after DMA. GPU polls for batch-level completion.
- `d_list_done[list_id]`: fine-grained per-list signal in device memory. GPU polls L2 cache (~30ns) for specific list completion.

**Backward Compatibility:**

```cpp
namespace sg_compat {
    void linear_to_sg_entries(DeviceSGEntry* entries,
                              const void* cpu_base, void* gpu_base,
                              uint32_t token_size, uint32_t start_idx,
                              uint32_t count, uint32_t tag = 0);
}
```

Converts a linear address range (`cpu_base + idx * token_size`) into SG entries, allowing existing linear-mode workloads to run through the SG pipeline.

#### `GpuTopology`

Per-GPU topology information.

```cpp
struct GpuTopology {
    int gpu_id;              // GPU index
    int numa_node;           // NUMA node affinity
    int pcie_bus;            // PCIe bus number
    int cpu_start;           // First CPU core for this NUMA node
    int cpu_end;             // Last CPU core for this NUMA node
    int ht_offset;           // Hyper-threading sibling offset
    int num_physical_cores;  // Physical core count on this NUMA node
};
```

---

### Classes

#### `CopyEngineManager`

Manages multiple CUDA Driver API streams for high-priority, pipelined scatter-gather DMA.

```cpp
class CopyEngineManager {
public:
    CopyEngineManager();
    ~CopyEngineManager();

    CUresult init(CUcontext ctx, int num_channels = 0);
    void     shutdown();

    CUresult pin_context();    // Push CUDA context (saves 2-4 us/batch)
    void     unpin_context();  // Pop CUDA context

    CUresult submit_scatter_gather(const SGEntry* entries, int count);
    CUresult wait_completion();
    CUresult submit_and_wait(const SGEntry* entries, int count);

    CUresult record_event_on_last_stream(CUevent event);
    CUresult make_stream_wait_on_all(CUstream stream);

    bool is_initialized() const;

    // Statistics (read-only accessors)
    uint64_t get_total_submissions() const;
    uint64_t get_total_entries() const;
    double   get_total_bytes() const;
};
```

**Methods:**

| Method | Description |
|--------|-------------|
| `init(ctx, num_channels)` | Create `num_channels` high-priority non-blocking CUDA streams. `num_channels=0` means use `MAX_CE_CHANNELS` (default: 3). |
| `shutdown()` | Synchronize and destroy all streams and events. |
| `pin_context()` | Push CUDA context to calling thread's stack. Eliminates per-call context switch overhead (~2-4 us). Must call `unpin_context()` when done. |
| `unpin_context()` | Pop CUDA context from calling thread's stack. |
| `submit_scatter_gather(entries, count)` | Enqueue async H2D copies, round-robin across channels. Records events on all dirty channels. |
| `wait_completion()` | Synchronize all channel events (blocks until all DMA completes). |
| `submit_and_wait(entries, count)` | Convenience: `submit_scatter_gather()` + `wait_completion()`. |
| `record_event_on_last_stream(event)` | Record a user-provided event on the most recently used stream. Used for staging buffer synchronization. |
| `make_stream_wait_on_all(stream)` | Insert `cuStreamWaitEvent` dependencies so `stream` won't execute until all pending CE DMAs complete. GPU-side ordering without CPU blocking. |

**Error-Checking Macro:**
```cpp
GFD_CU_CHECK(call)  // Checks CUresult, logs error with file/line, returns on failure
```

---

#### `CpuPollingThread`

CPU-side polling thread with parallel gather workers and staging DMA orchestration.

```cpp
class CpuPollingThread {
public:
    CpuPollingThread(DescriptorQueue* queue,
                     void* gpu_base,
                     void* cpu_base,
                     size_t total_cpu_size,
                     bool use_ce = true,
                     int numa_node = 0,
                     int core_offset = 0,
                     int num_ce_channels = 0,
                     int exclusive_core_base = -1,
                     int exclusive_core_count = 0);
    ~CpuPollingThread();

    bool   init_copy_engine();
    bool   init_direct_ce();
    double submit_direct(const SGEntry* entries, int count);

    void set_tiled_queue(TiledQueue* tq);

    void start();
    void stop();

    // Statistics
    uint64_t get_descriptors_processed() const;
    uint64_t get_batches_submitted() const;
    uint64_t get_coalesced_entries() const;
    uint64_t get_staging_batches() const;
    double   get_total_bytes_copied() const;
    uint64_t get_gather_us() const;
    uint64_t get_dma_wait_us() const;
    uint64_t get_dma_submit_us() const;
    uint64_t get_queue_read_us() const;
    bool     is_ce_mode() const;
    const CopyEngineManager& get_ce_manager() const;

    void reset_stats();
};
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queue` | `DescriptorQueue*` | — | Shared managed-memory descriptor queue |
| `gpu_base` | `void*` | — | GPU destination buffer base address |
| `cpu_base` | `void*` | — | CPU pinned source buffer base address |
| `total_cpu_size` | `size_t` | — | Total CPU buffer size (determines staging buffer size) |
| `use_ce` | `bool` | `true` | Use CUDA Copy Engine (false = direct memcpy to managed memory) |
| `numa_node` | `int` | `0` | NUMA node for buffer and thread affinity |
| `core_offset` | `int` | `0` | Core group offset within NUMA node (legacy) |
| `num_ce_channels` | `int` | `0` | CE channels (0 = use `MAX_CE_CHANNELS`) |
| `exclusive_core_base` | `int` | `-1` | First CPU core for exclusive use (-1 = use default) |
| `exclusive_core_count` | `int` | `0` | Number of CPU cores reserved for this poller |

**Methods:**

| Method | Description |
|--------|-------------|
| `init_copy_engine()` | Initialize CE manager, allocate staging buffers (hugepage preferred), spawn and pin gather worker threads. Must be called before `start()`. |
| `init_direct_ce()` | Initialize a separate CE manager for the direct-submit path. Call after `init_copy_engine()`. |
| `submit_direct(entries, count)` | **Scheme A fast path**: CPU-side direct DMA bypass (no descriptor queue). Thread-safe. Returns latency in microseconds, or -1 on error. Best for transfers ≤ 4MB. |
| `set_tiled_queue(TiledQueue* tq)` | Enable per-tile completion signaling. When set, `flush_batch` writes `tile_chunk_done[tile_id]` via CE write-back. If `tq->d_tile_chunk_done` is set, signals go to device memory (L2-cached). Must be called before `start()`. |
| `start()` | Launch the main polling thread (pinned to `exclusive_core_base` or calculated core). |
| `stop()` | Signal polling thread to stop and join. Drains remaining queue entries. |
| `get_ce_manager()` | Returns a const reference to the underlying `CopyEngineManager` (for external stats/diagnostics). |
| `reset_stats()` | Zero all performance counters and per-tile progress (call between test iterations). |

---

#### `StagingPool`

Singleton pre-allocated hugepage staging buffer pool. Eliminates ~28ms cold-start allocation cost.

```cpp
class StagingPool {
public:
    static constexpr int MAX_BUFFER_SETS = 8;
    static constexpr int BUFS_PER_SET = 5;
    static constexpr size_t MAX_BUF_SIZE = 128 * 1024 * 1024;

    static StagingPool& instance();

    bool init(int num_gpus, size_t buf_size, int numa_node = 0);
    int  acquire_buffers(size_t required_size, int numa_node = 0);
    void get_buffers(int set_idx, char* out_bufs[BUFS_PER_SET],
                     size_t& out_size, bool& out_hugepage);
    void release_buffers(int set_idx);
    void shutdown();

    bool   is_initialized() const;
    size_t buffer_size() const;
};
```

**Lifecycle:**

```
init()  →  acquire_buffers()  →  get_buffers()  →  release_buffers()  →  shutdown()
 ↑ once      ↑ per poller         ↑ per poller       ↑ per poller         ↑ once
```

| Method | Description |
|--------|-------------|
| `init(num_gpus, buf_size, numa_node)` | Pre-allocate `num_gpus` buffer sets, each with 5 buffers of `buf_size` bytes. Uses `mmap(MAP_HUGETLB)` + `cudaHostRegister` with NUMA binding. Falls back to `cudaMallocHost`. |
| `acquire_buffers(required_size, numa_node)` | Return an available buffer set index, or -1 if none available. |
| `get_buffers(set_idx, ...)` | Retrieve the 5 buffer pointers from a set. |
| `release_buffers(set_idx)` | Mark a buffer set as available for reuse. |
| `shutdown()` | Unregister and free all staging memory. |

---

#### `TopologyConfig`

NUMA/PCIe topology configuration and utility functions.

```cpp
struct TopologyConfig {
    int num_gpus;
    int total_numa_nodes;
    int cpus_per_numa;
    int physical_cores_per_numa;
    std::vector<GpuTopology> gpus;
    std::vector<int> gpus_per_numa;

    int recommended_ce_channels(int active_gpus_on_same_numa) const;
    void get_exclusive_cores(int gpu_id, int& out_base_cpu,
                             int& out_num_cores, int& out_stride) const;
};

TopologyConfig discover_topology(int num_gpus);
void print_topology(const TopologyConfig& topo);
```

| Function | Description |
|----------|-------------|
| `discover_topology(num_gpus)` | Probe `/sys/devices/system/node/` and `/sys/bus/pci/devices/` (Linux) for NUMA layout. Returns topology config with per-GPU core assignments. |
| `recommended_ce_channels(n)` | Returns 3 for ≤2 GPUs/NUMA, 2 for ≤4, 1 otherwise. |
| `get_exclusive_cores(gpu_id, ...)` | Compute exclusive core partition for a given GPU. |
| `print_topology(topo)` | Print topology summary via `GFD_LOG_INFO`. |

---

---

#### `WarpSpecConfig`

Configuration for a warp-specialized transfer+compute session.

```cpp
struct WarpSpecConfig {
    // === Required ===
    uint32_t total_tokens;      // Total number of tokens to transfer
    uint32_t token_size;        // Bytes per token
    void* cpu_src;              // CPU source buffer (must be pinned memory)
    void* gpu_dst;              // GPU destination buffer

    // === Optional (sensible defaults) ===
    uint32_t tokens_per_tile = 128;    // Tokens per tile
    uint32_t tokens_per_chunk = 0;     // 0 = auto-tune based on token_size
    int num_blocks = 0;                // 0 = auto (use all SMs)
    int numa_node = 0;                 // NUMA node for CPU poller
    bool use_copy_engine = true;       // Use CE for DMA acceleration
    bool double_buffer = false;        // Double-buffer ping-pong mode
    bool per_tile_mode = false;        // K=1: entire tile as single chunk

    // === Derived ===
    uint32_t effective_tokens_per_chunk() const;  // Auto-tuned chunk size
    uint32_t total_tiles() const;
    uint32_t chunks_per_tile() const;
    int block_size() const;  // (1 + K) * 32 or (1 + 2K) * 32 for double-buffer
};
```

**Auto-tuning** (`tokens_per_chunk = 0`): targets ~128KB per chunk DMA. Adjusts based on token_size (64KB for ≤512B tokens, 256KB for ≥64KB tokens). Enforces max K=8 chunks/tile. Result rounded down to power of 2.

---

#### `WarpSpecSession`

Manages the full lifecycle of a warp-specialized transfer+compute operation. Encapsulates TiledQueue, CPU poller, staging pool, and kernel launch configuration.

```cpp
class WarpSpecSession {
public:
    explicit WarpSpecSession(const WarpSpecConfig& config);
    ~WarpSpecSession();

    // Launch a GFD_WARP_SPEC_KERNEL-generated kernel
    template<typename KernelFn, typename ComputeFn>
    void launch(KernelFn kernel, ComputeFn compute_fn, cudaStream_t stream = 0);

    // Get raw launch parameters for manual invocation
    struct LaunchParams {
        TiledQueue* tq;
        void* gpu_buffer;
        const void* cpu_base;
        dim3 grid;
        dim3 block;
    };
    LaunchParams get_launch_params() const;

    // Wait for completion (kernel + all DMA)
    void synchronize();

    // Reset for another launch
    void reset();

    // Diagnostics
    struct Stats {
        uint64_t descriptors_processed;
        uint64_t bytes_transferred;
        double elapsed_ms;
        double bandwidth_gbps;
    };
    Stats get_stats() const;

    TiledQueue* get_queue() const;
    CpuPollingThread* get_poller() const;
};
```

**Usage:**
```cpp
gfd::WarpSpecConfig cfg;
cfg.total_tokens = 8192;
cfg.token_size = 16384;
cfg.cpu_src = cpu_data;
cfg.gpu_dst = gpu_data;

gfd::WarpSpecSession session(cfg);
session.launch(my_kernel, MyCompute{output});
session.synchronize();
auto stats = session.get_stats();
```

**Note:** `WarpSpecSession` does not expose `exclusive_core_base/count` for NUMA-aware multi-GPU setups. For multi-GPU, use the low-level `TiledQueue` + `CpuPollingThread` API directly (see `examples/08_multi_gpu_warp_spec.cu`).

---

#### `SGWarpSpecConfig`

Configuration for an SG warp-specialized session.

```cpp
struct SGWarpSpecConfig {
    int num_compute_warps = 0;   // 0 = transfer-only
    int num_blocks = 0;          // 0 = auto (use all SMs)
    int numa_node = 0;           // NUMA node for CPU poller
    bool use_copy_engine = true; // Use CE for DMA acceleration
    int max_lists = MAX_SG_LISTS;          // Max concurrent SG lists (512)
    int max_entries = MAX_SG_POOL_ENTRIES;  // Max entries in pool (16384)

    int block_size() const;  // (1 + max(num_compute_warps, 1)) * 32
};
```

---

#### `SGWarpSpecSession`

Manages the full lifecycle of an SG warp-specialized transfer+compute operation. Encapsulates `SGTaskQueue`, `DescriptorQueue`, CPU poller, and kernel launch configuration.

```cpp
class SGWarpSpecSession {
public:
    explicit SGWarpSpecSession(const SGWarpSpecConfig& config);
    ~SGWarpSpecSession();

    // Host-side SG list submission
    uint64_t submit_sg_list(const DeviceSGEntry* entries, uint32_t count,
                            uint32_t list_id, uint32_t flags = SG_FLAG_HOST_SUBMITTED);

    // Accessors
    SGTaskQueue* get_sg_queue() const;
    DescriptorQueue* get_desc_queue() const;
    CpuPollingThread* get_poller() const;

    // Launch an SG warp-spec kernel
    // Kernel signature: (DescriptorQueue* dq, SGTaskQueue* sq, ComputeFn fn)
    template<typename KernelFn, typename ComputeFn>
    void launch(KernelFn kernel, ComputeFn compute_fn, cudaStream_t stream = 0);

    void synchronize();
    void reset();

    struct Stats {
        uint64_t descriptors_processed;
        uint64_t bytes_transferred;
        double elapsed_ms;
        double bandwidth_gbps;
    };
    Stats get_stats() const;
};
```

**Usage:**
```cpp
gfd::SGWarpSpecConfig cfg;
cfg.num_compute_warps = 1;
cfg.num_blocks = 8;

gfd::SGWarpSpecSession session(cfg);

// Submit SG lists (host pre-fill or dynamic)
session.submit_sg_list(entries, count, list_id, gfd::SG_FLAG_HOST_SUBMITTED);

session.launch(sg_kernel, MyCompute{output});
session.synchronize();
auto stats = session.get_stats();
```

**Key Differences from `WarpSpecSession`:**
- No `cpu_src`/`gpu_dst` — addresses are in the SG entries
- Uses `SGTaskQueue` instead of `TiledQueue`
- `synchronize()` waits on `lists_completed` (DMA-confirmed by CPU poller)
- Supports GPU dynamic submission via `sg_submit_list()` device primitives

---

## Namespace: gfd::device

### Device Functions (`<gfd/device.cuh>`)

All functions are `__device__ __forceinline__` and designed for composition inside user kernels.

#### `write_descriptor`

Write one descriptor entry for a single token's transfer request.

```cpp
__device__ __forceinline__
void write_descriptor(
    DescriptorQueue* queue,
    uint64_t base_slot,      // Pre-assigned starting slot
    int token_idx,           // This thread's token index
    uint64_t src_addr,       // CPU source address
    void* gpu_dst,           // GPU destination buffer base
    uint32_t token_size,     // Bytes per token
    int num_tokens,          // Total tokens (for FLAG_LAST_IN_BATCH)
    uint64_t user_data = 0); // Optional user payload
```

**Behavior:**
- Writes to `queue->entries[(base_slot + token_idx) % QUEUE_SIZE]`
- Sets `FLAG_LAST_IN_BATCH` on the last token's entry
- Does NOT commit (CPU cannot see the entry yet)

---

#### `write_descriptor_safe`

Safe descriptor write with ring-buffer backpressure.

```cpp
__device__ __forceinline__
bool write_descriptor_safe(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    uint64_t src_addr,
    void* gpu_dst,
    uint32_t token_size,
    int num_tokens,
    uint64_t user_data = 0);
```

**Behavior:**
- Checks if the target slot has been consumed (`sequence == 0`) before writing
- Returns `false` if the ring buffer is full at this slot (GPU producing faster than CPU consuming)
- Caller should retry or yield
- Use in latency-tolerant paths where correctness > throughput

---

#### `fence_and_commit`

Two-phase commit with warp-level `__threadfence_system()` optimization.

```cpp
__device__ __forceinline__
void fence_and_commit(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    bool active);  // true if this thread has a valid descriptor
```

**Behavior:**
1. Warp leader (lane 0) executes `__threadfence_system()` — makes descriptor fields visible to CPU
2. Each active thread writes `sequence = slot + 1` (commit marker)
3. Warp leader executes `__threadfence_system()` — makes sequence visible to CPU

**Important:** Only the warp leader pays the fence cost (~1-2 us), reducing overhead by 32x compared to per-thread fences.

---

#### `wait_for_completion`

Poll until CPU signals completion of all transfers.

```cpp
__device__ __forceinline__
void wait_for_completion(
    DescriptorQueue* queue,
    uint64_t expected_done);  // base_slot + num_tokens
```

**Behavior:**
- Spin-polls `queue->done_idx` until `>= expected_done`
- Should be called by a single thread (typically `tid == 0`)
- Follow with `__syncthreads()` to broadcast completion

---

#### `write_and_commit`

Convenience function combining `write_descriptor()` + `fence_and_commit()`.

```cpp
__device__ __forceinline__
void write_and_commit(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    bool active,
    uint64_t src_addr,
    void* gpu_dst,
    uint32_t token_size,
    int num_tokens,
    uint64_t user_data = 0);
```

**Use when:** Simple transfer without overlapped compute. For maximum performance with compute overlap, call `write_descriptor()` and `fence_and_commit()` separately.

---

### Device Primitives (`<gfd/device_primitives.cuh>`)

Low-level **warp-collective** operations for warp-specialized kernels. All threads in a warp must participate (or be explicitly masked). For most users, prefer the Layer 2 Framework (`warp_spec.cuh`).

#### `acquire_tile`

Atomically acquire the next tile ID. Returns `>= scheduler.total_tiles` when exhausted.

```cpp
__device__ __forceinline__
uint32_t acquire_tile(TiledQueue* tq);
```

Only one thread per block should call this, then broadcast the result.

---

#### `acquire_chunk_slots`

Lane 0 performs `atomicAdd` to get `count` contiguous slots, broadcasts result via `__shfl_sync`.

```cpp
__device__ __forceinline__
uint64_t acquire_chunk_slots(TiledQueue* tq, uint32_t count);
```

All 32 lanes in the warp must be active.

---

#### `write_chunk`

Write `tokens_per_chunk` descriptors starting at `slot_base`. Loops if `tokens_per_chunk > 32`.

```cpp
__device__ __forceinline__
void write_chunk(
    TiledQueue* tq,
    uint64_t slot_base,
    uint32_t tile_id,
    uint32_t chunk_id,
    uint32_t tokens_per_chunk,
    uint32_t tokens_per_tile,
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size);
```

**Behavior:**
- Encodes `tile_id` in upper 32 bits of `user_data`: `((uint64_t)tile_id << 32) | global_idx`
- Sets `FLAG_LAST_IN_TILE` on the last entry of each chunk
- Sets `FLAG_LAST_CHUNK_IN_TILE` on the final entry of the last chunk in a tile

---

#### `commit_chunk`

Two-phase commit: `__threadfence_system()` → write sequence numbers → `__threadfence_system()`.

```cpp
__device__ __forceinline__
void commit_chunk(TiledQueue* tq, uint64_t slot_base, uint32_t count);
```

---

#### `wait_chunk_done`

Poll per-tile completion signal until cumulative done count meets threshold.

```cpp
__device__ __forceinline__
void wait_chunk_done(TiledQueue* tq, uint32_t tile_id, uint64_t expected);
```

**Behavior:**
- When `d_tile_chunk_done` is available, polls device memory (L2 cache ~10ns)
- Otherwise falls back to host-mapped memory (PCIe ~1500ns per load)
- Uses `__nanosleep` (sm_70+) exponential backoff after 64 spin iterations to free SM execution slots

---

#### `wait_queue_space`

Spins until there are at least `needed * 2` free slots in the ring buffer (backpressure).

```cpp
__device__ __forceinline__
void wait_queue_space(TiledQueue* tq, uint32_t needed);
```

Warp-collective: all lanes must participate.

---

#### `submit_chunk`

Convenience: `write_chunk()` + `commit_chunk()` in one call.

```cpp
__device__ __forceinline__
void submit_chunk(
    TiledQueue* tq,
    uint64_t slot_base,
    uint32_t tile_id,
    uint32_t chunk_id,
    uint32_t tokens_per_chunk,
    uint32_t tokens_per_tile,
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size);
```

---

#### `submit_tile`

Submit ALL `T` descriptors for a tile in a single acquire+write+commit cycle. Reduces system fences from `2*K` to 2 total.

```cpp
__device__ __forceinline__
void submit_tile(
    TiledQueue* tq,
    uint64_t slot_base,
    uint32_t tile_id,
    uint32_t T,              // tokens_per_tile
    uint32_t C,              // tokens_per_chunk (for flag marking)
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size);
```

---

## Namespace: gfd::warp\_spec

### Warp-Spec Framework (`<gfd/warp_spec.cuh>`)

#### `GFD_WARP_SPEC_KERNEL(name, ComputeFn)`

Macro that generates a complete warp-specialized kernel. The generated kernel handles:
- Dynamic tile acquisition via `atomicAdd` scheduling
- Per-chunk slot allocation and descriptor submission
- Fence, commit, and completion polling
- Cross-warp synchronization (transfer warp → compute warps)
- Multi-tile looping (one block processes many tiles)

```cpp
// Define compute functor
struct MyCompute {
    float* output;
    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32) {
            output[chunk.global_idx(i)] = chunk.data<float>(i)[0];
        }
    }
};

// Generate kernel (single-buffer: block = (1 + K) warps)
GFD_WARP_SPEC_KERNEL(my_kernel, MyCompute);

// Double-buffer variant (block = (1 + 2K) warps)
GFD_WARP_SPEC_KERNEL_DB(my_kernel_db, MyCompute);
```

**Generated kernel signature:**
```cpp
__global__ void my_kernel(gfd::TiledQueue* tq, void* gpu_dst, const void* cpu_src, MyCompute compute);
```

#### `gfd::warp_spec::ChunkView`

Read-only view passed to the user's compute functor. Data is guaranteed available.

```cpp
struct ChunkView {
    void* base_ptr;               // GPU address of first token in chunk
    uint32_t token_size;          // Bytes per token
    uint32_t size;                // Number of tokens in this chunk

    uint32_t tile_id;             // Global tile ID
    uint32_t chunk_id;            // Chunk index within tile (0..K-1)
    uint32_t global_token_offset; // Global index of first token

    int lane_id;                  // Thread's lane within warp (0..31)

    template<typename T>
    __device__ T* data(int local_token_idx) const;  // Pointer to i-th token

    __device__ int global_idx(int local_token_idx) const;  // Global token index
};
```

#### `gfd::warp_spec::TileContext`

Tile-level view for optional hooks. Provides access to all chunks within a tile.

```cpp
struct TileContext {
    uint32_t tile_id;
    uint32_t tokens_per_tile;
    uint32_t chunks_per_tile;
    uint32_t tokens_per_chunk;
    void* tile_base_ptr;
    uint32_t token_size;

    __device__ ChunkView get_chunk(int chunk_id) const;
};
```

---

### Pattern 5: Warp-Specialized Tiled Transfer+Compute (Interleaved)

Maximum overlap via sub-tile chunking with concurrent multi-SM submission.

```
Block = 160 threads (5 warps):
  Warp 0:      Transfer warp — atomicAdd slots, write descs, poll tile_done
  Warps 1-4:   Compute warps — each owns one 32-token chunk, computes on arrival
```

```cuda
// Warp 0 (Transfer): submit descriptors per-chunk with interleaved slots
for (int chunk = 0; chunk < 4; chunk++) {
    uint64_t chunk_base = gfd::device::acquire_chunk_slots(tq, CHUNK_TOKENS);
    gfd::device::write_chunk(tq, chunk_base, tile_id, chunk, ...);
    gfd::device::commit_chunk(tq, chunk_base, CHUNK_TOKENS);
}
// Poll tile_done for progressive chunk completion
for (int chunk = 0; chunk < 4; chunk++) {
    gfd::device::wait_chunk_done(tq, tile_id, (chunk+1) * CHUNK_TOKENS);
    chunks_ready = chunk + 1;  // signal compute warps via smem
}

// Warps 1-4 (Compute): each warp waits for its chunk
int my_chunk = warp_id - 1;
while (chunks_ready <= my_chunk) {}  // spin on smem flag
// Compute immediately on 32 tokens
compute(gpu_buffer + (tile_id * 128 + my_chunk * 32 + lane_id) * token_dim);
```

**Key Design Principles:**

1. **AtomicAdd per-chunk**: Each tile's 128 tokens are submitted in 4 chunks of 32 via `atomicAdd(&write_idx, 32)`. Different tiles' chunks naturally **interleave** in the queue based on SM scheduling order.

2. **Count-based signaling**: `tile_done[tile_id]` stores cumulative token count (not slot index). The CPU increments `tile_progress[tile_id]` as it processes entries for each tile, then CE-writes-back the count.

3. **Sub-tile overlap**: Compute warp 1 starts on chunk 0 (tokens 0-31) while the transfer warp is still polling for chunks 1-3. Transfer latency is hidden behind compute.

4. **Inter-tile fairness**: With interleaved slots, the CPU processes entries from multiple tiles in each batch, so no tile starves waiting behind others.

**Performance (64 tiles × 128 tokens × 4KB = 32MB):**

| Mode | Latency | Bandwidth | Speedup |
|------|---------|-----------|---------|
| Warp-Specialized (interleaved) | 3.91 ms | 8.59 GB/s | **19.6x** |
| Tiled (per-tile wait, whole tile) | 7.67 ms | 4.38 GB/s | 10.0x |
| Baseline (global wait) | 76.63 ms | 0.44 GB/s | 1.0x |

---

## Namespace: gfd::sg

### SG Device Primitives (`<gfd/sg_device_primitives.cuh>`)

Warp-collective GPU-side primitives for SG task queue operations. All functions assume full-warp participation (all 32 lanes active). Lane 0 performs atomic operations; results are broadcast to all lanes via `__shfl_sync`.

#### Backpressure Functions

##### `sg_wait_entry_space`

Spin-wait until the entry pool has room for `needed` entries. Prevents overflow when producers submit faster than the transfer warp consumes.

```cpp
__device__ __forceinline__
void sg_wait_entry_space(SGTaskQueue* sq, uint32_t needed);
```

**Behavior:** Lane 0 polls `(entry_alloc_idx - entry_consumed_idx) < MAX_SG_POOL_ENTRIES - needed`. Uses `__nanosleep` exponential backoff on sm_70+.

---

##### `sg_wait_list_space`

Spin-wait until the list ring buffer has a free slot.

```cpp
__device__ __forceinline__
void sg_wait_list_space(SGTaskQueue* sq);
```

**Behavior:** Lane 0 polls `(list_alloc_idx - list_read_idx) < MAX_SG_LISTS - 1`.

---

#### Allocation Functions

##### `sg_alloc_entries`

Atomically allocate `count` contiguous slots in the entry pool.

```cpp
__device__ __forceinline__
uint64_t sg_alloc_entries(SGTaskQueue* sq, uint32_t count);
```

**Returns:** Starting pool offset (modular index into `entries[]`). Lane 0 `atomicAdd`, broadcast to all lanes.

---

##### `sg_alloc_list`

Atomically allocate one list ring slot.

```cpp
__device__ __forceinline__
uint64_t sg_alloc_list(SGTaskQueue* sq);
```

**Returns:** List slot index (modular index into `lists[]`). Lane 0 `atomicAdd`, broadcast to all lanes.

---

#### Write Functions

##### `sg_write_entries`

Cooperatively write `count` `DeviceSGEntry` values from a device-accessible array into the entry pool at `pool_offset`.

```cpp
__device__ __forceinline__
void sg_write_entries(
    SGTaskQueue* sq,
    uint64_t pool_offset,
    const DeviceSGEntry* src_entries,
    uint32_t count);
```

**Behavior:** Each lane handles entries in stride-32 fashion. Modular wrapping applied to pool index.

---

##### `sg_write_entries_inline`

Write entries from parallel arrays of `src_addrs`, `dst_addrs`, `sizes`, with a shared `tag`.

```cpp
__device__ __forceinline__
void sg_write_entries_inline(
    SGTaskQueue* sq,
    uint64_t pool_offset,
    const uint64_t* src_addrs,
    const uint64_t* dst_addrs,
    const uint32_t* sizes,
    uint32_t tag,
    uint32_t count);
```

---

#### Commit Function

##### `sg_commit_list`

Two-phase fence protocol to commit a list header (consistent with existing `commit_chunk`):
1. `__threadfence_system()` — entry writes visible to CPU
2. Write list header fields (`pool_offset`, `count`, `list_id`, `flags`)
3. `__threadfence_system()` — header visible
4. Write sequence marker (commit point: `slot + 1`)
5. `__threadfence_system()` — sequence visible to consumer

```cpp
__device__ __forceinline__
void sg_commit_list(
    SGTaskQueue* sq,
    uint64_t slot,
    uint32_t pool_offset,
    uint32_t count,
    uint32_t list_id,
    uint32_t flags);
```

---

#### Convenience Function

##### `sg_submit_list`

All-in-one: backpressure check + allocate entries + write entries + allocate list + commit.

```cpp
__device__ __forceinline__
uint64_t sg_submit_list(
    SGTaskQueue* sq,
    const DeviceSGEntry* entries,
    uint32_t count,
    uint32_t list_id,
    uint32_t flags = SG_FLAG_NONE);
```

**Returns:** The allocated list slot index (for tracking).

**Typical usage from compute warp:**
```cuda
// Build entries in shared/local memory
DeviceSGEntry my_entries[N];
for (int i = 0; i < N; i++) {
    my_entries[i] = {src_addr[i], dst_addr[i], size, 0};
}

// Submit (handles backpressure internally)
uint64_t slot = gfd::sg::sg_submit_list(sq, my_entries, N, expert_id, SG_FLAG_NONE);
```

---

#### Completion Functions

##### `sg_wait_list_done`

Coarse-grained completion wait. Polls `lists_completed >= expected`.

```cpp
__device__ __forceinline__
void sg_wait_list_done(SGTaskQueue* sq, uint64_t expected);
```

**Behavior:** Low-overhead single counter. Exponential backoff via `__nanosleep` after 64 spins. No per-list granularity.

---

##### `sg_wait_list_id_done`

Fine-grained completion wait. Polls `d_list_done[list_id]` in device memory (L2 cache, ~30ns latency).

```cpp
__device__ __forceinline__
void sg_wait_list_id_done(SGTaskQueue* sq, uint32_t list_id);
```

**Behavior:** The CPU poller writes a non-zero value to `d_list_done[list_id]` after DMA completes for all entries in that list. L2-cached polling for minimal latency.

---

## Namespace: gfd::sg\_warp\_spec

### SG Warp-Spec Framework (`<gfd/sg_warp_spec.cuh>`)

Scatter-gather mode warp specialization. Unlike the tile-based framework (`warp_spec.cuh`) which uses fixed linear address mapping (`src = cpu_base + idx * token_size`), SG mode accepts dynamically submitted `(src, dst, size)` tuples via `SGTaskQueue`.

#### `GFD_SG_WARP_SPEC_KERNEL(name, ComputeFn)`

Macro that generates a complete SG warp-specialized kernel.

```cpp
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

**Generated kernel signature:**
```cpp
__global__ void sg_kernel(gfd::DescriptorQueue* dq, gfd::SGTaskQueue* sq, MyCompute compute_fn);
```

**Block layout:** 64 threads (2 warps):
- **Warp 0**: Transfer warp — polls `SGList` from `SGTaskQueue`, converts entries to `Descriptor`, submits to `DescriptorQueue`, polls DMA completion
- **Warp 1**: Compute warp — waits for list ready signal, calls user `ComputeFn` with `SGListView`

**Multi-block support:** Multiple blocks can run in parallel. Transfer warps in different blocks use `atomicAdd` on `list_read_idx` for contention-free list claiming. `atomicCAS`-based max update on `entry_consumed_idx` ensures correct backpressure tracking.

---

#### `gfd::sg_warp_spec::SGListView`

Read-only view passed to the user's compute functor. Data is guaranteed available (DMA completed).

```cpp
struct SGListView {
    uint32_t list_id;       // List identifier for tracking
    uint32_t count;         // Number of entries in this list
    uint32_t flags;         // SG flags
    int lane_id;            // Thread's lane within warp (0..31)
    uint32_t pool_offset;   // Starting offset in entry pool
    const SGTaskQueue* sq;  // Back-pointer for entry access

    // Get raw entry at index
    __device__ DeviceSGEntry get_entry(uint32_t idx) const;

    // Get typed pointer to destination buffer of entry at index
    template<typename T>
    __device__ T* dst_ptr(uint32_t idx) const;

    // Get size of entry at index
    __device__ uint32_t entry_size(uint32_t idx) const;
};
```

**Key difference from `ChunkView`:** `SGListView` accesses data by entry index (each entry has its own address/size), while `ChunkView` accesses tokens by offset from a contiguous tile base pointer.

---

#### `_SGWarpSpecState`

Internal shared memory state for cross-warp coordination. Same pattern as `_WarpSpecState` in the linear framework.

```cpp
struct _SGWarpSpecState {
    uint32_t list_id;
    uint32_t count;
    uint32_t pool_offset;
    uint32_t flags;
    volatile int list_ready;    // transfer warp → compute warp signal
    volatile int compute_done;  // compute warp → transfer warp signal
    volatile int terminated;    // transfer warp → all: no more lists
};
```

---

## Usage Patterns

### Pattern 1: Fused Prefetch + Compute (Recommended)

Maximum performance via communication/computation overlap.

```cpp
__global__ void fused_kernel(gfd::DescriptorQueue* queue,
                             gfd::TokenInfo* tokens,
                             void* gpu_buf, float* output,
                             int N, uint32_t token_size,
                             uint64_t base_slot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < N);

    // Phase 1: Submit transfer requests
    if (active) {
        gfd::device::write_descriptor(queue, base_slot, tid,
            tokens[tid].cpu_addr, gpu_buf, token_size, N);
    }
    gfd::device::fence_and_commit(queue, base_slot, tid, active);

    // Phase 2: Overlapped compute (runs while CPU+CE transfer data)
    float result = expensive_computation(tid);

    // Phase 3: Wait for transfers
    if (tid == 0) {
        gfd::device::wait_for_completion(queue, base_slot + N);
    }
    __syncthreads();

    // Phase 4: Use prefetched data
    float* my_data = (float*)((char*)gpu_buf + (size_t)tid * token_size);
    output[tid] = result + my_data[0];
}
```

### Pattern 2: Submit + Wait Split Kernels

For benchmarking or when compute is in a separate kernel.

```cpp
// Kernel 1: fire-and-forget submit
__global__ void submit_kernel(...) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < N);
    gfd::device::write_and_commit(queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0, gpu_buf, token_size, N);
}

// Kernel 2: lightweight completion wait
__global__ void wait_kernel(gfd::DescriptorQueue* queue, uint64_t done) {
    gfd::device::wait_for_completion(queue, done);
}

// Host launch
submit_kernel<<<blocks, threads>>>(...);
wait_kernel<<<1, 1>>>(queue, base_slot + N);
cudaDeviceSynchronize();
```

### Pattern 3: CPU Direct Submit (Bypass Queue)

For CPU-initiated transfers without GPU involvement. Best for ≤ 4MB.

```cpp
// Build scatter-gather list
std::vector<gfd::SGEntry> entries(N);
for (int i = 0; i < N; i++) {
    entries[i].dst  = (CUdeviceptr)(gpu_buf + i * token_size);
    entries[i].src  = cpu_ptrs[i];
    entries[i].size = token_size;
}

// Direct submit (blocks until complete)
double latency_us = poller.submit_direct(entries.data(), N);
```

### Pattern 4: Multi-GPU with Topology Discovery

```cpp
auto topo = gfd::discover_topology(num_gpus);
gfd::print_topology(topo);

for (int g = 0; g < num_gpus; g++) {
    int base_cpu, num_cores, stride;
    topo.get_exclusive_cores(g, base_cpu, num_cores, stride);
    int ce_ch = topo.recommended_ce_channels(
        topo.gpus_per_numa[topo.gpus[g].numa_node]);

    gfd::CpuPollingThread poller(queue[g], gpu_buf[g], cpu_buf[g], size,
                                  true, topo.gpus[g].numa_node, 0, ce_ch,
                                  base_cpu, num_cores);
    poller.init_copy_engine();
    poller.start();
}
```

### Pattern 6: SG Warp-Spec (Dynamic Scatter-Gather)

For MoE inference where token addresses are determined dynamically at runtime.

```cpp
#include <gfd/gfd.h>
#include <gfd/sg_warp_spec.cuh>

// 1. Define compute functor (receives SGListView instead of ChunkView)
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

// 2. Configure and launch
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
printf("BW: %.2f GB/s\n", session.get_stats().bandwidth_gbps);
```

**Key differences from Pattern 5 (linear Warp-Spec):**
- No `cpu_src`/`gpu_dst` pointers — addresses are embedded in `DeviceSGEntry`
- Compute functor receives `SGListView` instead of `ChunkView`
- Supports both host pre-fill and GPU-side dynamic submission via `sg_submit_list()`

---

## Build Integration

### CMake (Recommended)

```cmake
add_subdirectory(gfd)

add_executable(my_app my_app.cu)
target_link_libraries(my_app PRIVATE gfd_static)
set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

### Manual Compilation

```bash
# Build library
nvcc -std=c++17 -c -O3 gfd/src/copy_engine.cpp -I gfd/include -o copy_engine.o
g++ -std=c++17 -c -O3 -mavx512f gfd/src/cpu_polling.cpp -I gfd/include -o cpu_polling.o
ar rcs libgfd.a copy_engine.o cpu_polling.o

# Build application
nvcc -std=c++17 -O3 my_app.cu -I gfd/include -L. -lgfd -lcuda -lcudart -o my_app
```

### Requirements

- CUDA Toolkit ≥ 11.0 (Driver API required)
- C++17 compiler
- CMake ≥ 3.18
- Linux recommended (hugepage support, NUMA affinity, CPU pinning)
- AVX-512 recommended for optimal gather throughput (auto-detected)

---

## Performance Characteristics

**Hardware:** NVIDIA RTX PRO 5000 72GB (Blackwell, sm_120), PCIe Gen5 x16 (~63 GB/s theoretical)

| Transfer Pattern | Recommended Method | Expected Bandwidth |
|------------------|-------------------|--------------------|
| Small scattered (≤ 4MB total) | `submit_direct()` | 20-45 GB/s |
| Large contiguous (16-128MB) | GFD Warp-Spec (pure) | 43.6 GB/s |
| Large + compute overlap | GFD Warp-Spec | 33 GB/s |
| Dynamic scatter-gather (MoE) | SG Warp-Spec (multi-block) | 51 GB/s |
| CPU-initiated (any size) | `submit_direct()` | 53 GB/s |
| Multi-GPU (8x, per-GPU 128MB) | Warp-Spec + NUMA pinning | 340 GB/s aggregate |
| Baseline comparison | `cudaMemcpy(N)` | 3.1 GB/s |

Key advantages over `cudaMemcpy(N)`:
- **14-53x bandwidth** for scattered transfers
- **Overlapped compute**: GPU computes while CPU gathers + CE DMAs (warp-spec overlap)
- **Near-linear multi-GPU scaling**: 93-96% efficiency at 8 GPUs with NUMA-aware pinning
- **Zero GPU overhead**: transfer warp runs independently from compute warps
- **Device-memory signaling**: L2-cached polling for ~30ns signal latency
