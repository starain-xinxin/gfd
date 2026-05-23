# GFD Architecture Document

## Overview

GFD (GPU-Functional-Descriptor) is a high-performance scattered Host-to-Device (H2D) transfer library designed for LLM inference workloads, particularly KV-cache prefetching.

**Core Problem:** In LLM inference, tokens are scattered across CPU memory but need to be assembled contiguously in GPU memory. Standard `cudaMemcpy` per-token incurs massive API overhead (~3 GB/s at 4-16KB granularity). GFD achieves **43-53 GB/s** on PCIe Gen5 by offloading gather and coalescing to dedicated CPU cores while keeping the GPU free for compute.

**Architecture Layers:**
1. **Device Primitives** (`device_primitives.cuh`): Low-level slot acquire, commit, wait
2. **Warp-Spec Framework** (`warp_spec.cuh`): High-level macro-based kernel generation with automatic tile scheduling, warp specialization, and transfer-compute overlap
3. **SG Task Queue** (`sg_task_queue.h` + `sg_device_primitives.cuh`): Dynamic scatter-gather address submission with two-level SGList + entry pool structure
4. **SG Warp-Spec Framework** (`sg_warp_spec.cuh`): SG-mode warp specialization with per-list DMA and compute overlap
5. **Session Managers** (`warp_spec_session.h`): `WarpSpecSession` (linear) + `SGWarpSpecSession` (scatter-gather) for single-GPU lifecycle
6. **CPU Poller + Batch Processor**: Lock-free polling, parallel gather, staging DMA, per-tile/per-list CE write-back signaling
7. **Multi-GPU**: Manual `TiledQueue + CpuPollingThread` setup with NUMA-aware exclusive core partitioning

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            GPU Side                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────┐                 │
│  │              User Fused Kernel                      │                 │
│  │                                                     │                 │
│  │  Phase 1: write_descriptor() + fence_and_commit()   │                 │
│  │  Phase 2: overlapped_compute()                      │                 │
│  │  Phase 3: wait_for_completion()                     │                 │
│  │  Phase 4: use_prefetched_data()                     │                 │
│  └──────────────────────┬──────────────────────────────┘                 │
│                         │ writes descriptors                             │
│                         ▼                                                │
│  ┌─────────────────────────────────────────────────────┐                 │
│  │         DescriptorQueue (Managed Memory)            │                 │
│  │   entries[16384] │ write_idx │ read_idx │ done_idx  │                 │
│  └──────────────────────┬──────────────────────────────┘                 │
│                         │                   ▲                            │
└─────────────────────────┼───────────────────┼────────────────────────────┘
                          │ sequence commit   │ done_idx update
                          │ (PCIe write)      │ (PCIe read)
┌─────────────────────────┼───────────────────┼────────────────────────────┐
│                CPU Side │                   │                            │
│                         ▼                   │                            │
│  ┌─────────────────────────────────┐        │                            │
│  │    CpuPollingThread (Core 0)    │        │                            │
│  │                                 │        │                            │
│  │  poll sequence → read batch     │        │                            │
│  │  detect contiguity              │        │                            │
│  │  dispatch parallel_gather()    ─┼────────┼───┐                        │
│  │  submit CE DMA                  │        │   │                        │
│  │  update done_idx               ─┼────────┘   │                        │
│  └─────────────────────────────────┘            │                        │
│                                                 ▼                        │
│  ┌───────────────────────────────────────────────────────────┐           │
│  │              Gather Workers (Cores 2,4,6,...,30)          │           │
│  │                                                           │           │
│  │  Worker 0: streaming_memcpy(src[0..k] → staging)          │           │
│  │  Worker 1: streaming_memcpy(src[k..2k] → staging)         │           │
│  │  ...                                                      │           │
│  │  Worker 14: streaming_memcpy(src[14k..N] → staging)       │           │
│  └───────────────────────────┬───────────────────────────────┘           │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────┐           │
│  │         Staging Buffers (5x N-buffered, Hugepage)         │           │
│  │                                                           │           │
│  │  [Buf 0: gathering] [Buf 1: DMA in-flight] [Buf 2: free]  │           │
│  └───────────────────────────┬───────────────────────────────┘           │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────┐           │
│  │     CopyEngineManager (3 CE streams, high priority)       │           │
│  │                                                           │           │
│  │  Stream 0: cuMemcpyHtoDAsync(staging → GPU)               │           │
│  │  Stream 1: cuMemcpyHtoDAsync(staging → GPU)               │           │
│  │  Stream 2: cuMemcpyHtoDAsync(staging → GPU)               │           │
│  └───────────────────────────────────────────────────────────┘           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Descriptor Queue (Lock-Free Ring Buffer)

**File:** `include/gfd/descriptor_queue.h`

The central communication channel between GPU and CPU. A fixed-size ring buffer in CUDA managed memory, accessible by both GPU and CPU without explicit copies.

```
Ring Buffer Layout (16384 entries × 64 bytes = 1MB):

  write_idx ──────────────────────────────────────┐
  (GPU atomicAdd)                                 │
                                                  ▼
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │  ✓  │  ✓  │  ✓  │  ●  │  ●  │  ○  │  ○  │  ○  │  ...
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     ▲              ▲
     │              │
  done_idx       read_idx
  (CPU writes,   (CPU writes)
   GPU polls)

  ✓ = consumed (sequence=0)
  ● = committed (sequence=slot+1), ready for CPU
  ○ = empty (not yet written)
```

**Synchronization Protocol:**

1. **GPU Write Phase:**
   - GPU thread writes descriptor fields (`src_addr`, `dst_addr`, `size`, `flags`)
   - Warp-leader `__threadfence_system()` ensures visibility
   - GPU thread writes `sequence = slot + 1` (commit marker)
   - Warp-leader `__threadfence_system()` publishes commit

2. **CPU Read Phase:**
   - Polling thread reads `sequence` with `__ATOMIC_ACQUIRE`
   - If `sequence == expected (read_idx + 1)`, entry is ready
   - CPU copies descriptor, writes `sequence = 0` with `__ATOMIC_RELEASE`
   - Advances `read_idx`

3. **Completion Notification:**
   - After DMA completes, CPU writes `done_idx = read_idx` with `__ATOMIC_RELEASE`
   - GPU polls `done_idx` via volatile read

**Design Decisions:**

- **Sequence-number commit** (not just write_idx check): prevents reading partially-written descriptors
- **Per-entry sequence** (not global lock): allows out-of-order GPU thread completion
- **Fixed size 16384**: power-of-two for cheap modulo (`% QUEUE_SIZE` = `& (QUEUE_SIZE-1)`)
- **64-byte alignment**: one descriptor per cache line, prevents false sharing

---

### 2. Device API (GPU-Side)

**File:** `include/gfd/device.cuh`

Header-only `__device__ __forceinline__` functions designed for composition inside user kernels.

**Key Design Principles:**

1. **Warp-Level Fence Optimization:**

   ```
   Traditional: Each thread calls __threadfence_system() → 32 fences per warp
   GFD:         Only warp leader (lane 0) fences → 1 fence per warp (32x reduction)
   ```

   This is safe because `__threadfence_system()` from any thread in a warp ensures visibility of all prior writes from all threads in that warp (due to warp-synchronous execution).

2. **Two-Phase Commit:**
   - Phase 1: Fence after descriptor field writes (makes data visible)
   - Phase 2: Write sequence number (makes entry visible to CPU poller)
   - Phase 3: Fence after sequence write (ensures CPU sees commit)

3. **Separation of Concerns:**
   - `write_descriptor()`: pure data write, no synchronization
   - `fence_and_commit()`: pure synchronization, no data
   - Allows compute insertion between write and commit for latency hiding

---

### 3. CPU Polling Thread

**File:** `src/cpu_polling.cpp`, `include/gfd/cpu_polling.h`

The performance-critical CPU component. A dedicated pinned thread that continuously polls the queue, orchestrates parallel gather, and submits DMA.

#### 3.1 Main Polling Loop

```
┌───────────────────────────────────────────────┐
│                 polling_loop()                │
│                                               │
│  while (running) {                            │
│      read sequence at read_idx                │
│      if (committed) {                         │
│          accumulate into batch[]              │
│          if (threshold reached OR urgent) {   │
│              process_batch(batch, count)      │
│          }                                    │
│      } else {                                 │
│          if (batch pending) {                 │
│              gap-spin 256 cycles              │
│              if still empty → flush batch     │
│          } else {                             │
│              cpu_pause()                      │
│          }                                    │
│      }                                        │
│  }                                            │
│  drain remaining entries                      │
└───────────────────────────────────────────────┘
```

**Adaptive Batching:**

- Default threshold: 256 entries
- Small entries (≤ 2KB): threshold = 1024 (amortize per-batch overhead)
- Immediate flush on `FLAG_LAST_IN_BATCH` or `FLAG_URGENT`
- Gap-spin: if queue stalls mid-batch, spin 256 iterations before flushing partial batch

#### 3.2 Batch Processing Pipeline

`process_batch()` implements a multi-strategy optimization pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│                    process_batch(batch, count)                   │
│                                                                  │
│  ┌─── Single Entry? ──────────────────────────────────────────┐  │
│  │  YES → Direct CE submit (1 entry, no staging)              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                         │ NO                                     │
│  ┌─── Dst Contiguous? ────────────────────────────────────────┐  │
│  │  YES ─┬─ Src Contiguous? ──→ Single coalesced DMA          │  │
│  │       └─ Src Scattered?  ──→ Parallel gather + single DMA  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                         │ NO                                     │
│  ┌─── Generic Path ───────────────────────────────────────────┐  │
│  │  1. Sort by src address                                    │  │
│  │  2. Coalesce adjacent entries                              │  │
│  │  3. If coalesced > 32 && fits staging:                     │  │
│  │       parallel gather → sort by dst → merge runs → CE DMA  │  │
│  │  4. Else:                                                  │  │
│  │       direct scatter-gather CE submit                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Strategy Selection Rationale:**

| Strategy                        | When                             | Benefit                                      |
| ------------------------------- | -------------------------------- | -------------------------------------------- |
| Single coalesced DMA            | src + dst both contiguous        | One DMA call, zero gather overhead           |
| Parallel gather + single DMA    | dst contiguous, src scattered    | 15 workers gather in parallel, one large DMA |
| Generic sort + coalesce + stage | dst non-contiguous               | Maximizes DMA transfer sizes                 |
| Direct scatter-gather           | Few entries or won't fit staging | Avoids staging overhead                      |

#### 3.3 Parallel Gather Architecture

```
                 Main Thread (Core 0)
                 ┌──────────────┐
                 │ entries[0..k]│
                 └──────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   Worker 0        Worker 1         Worker 14
   (Core 2)        (Core 4)        (Core 30)
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │entries   │   │entries   │   │entries   │
  │[k..2k]   │   │[2k..3k]  │   │[14k..N]  │
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │               │               │
       ▼               ▼               ▼
  ┌─────────────────────────────────────────┐
  │        Staging Buffer (Contiguous)      │
  │  [token0|token1|token2|...|tokenN-1]    │
  └─────────────────────────────────────────┘
       │
       ▼  Single coalesced DMA
  ┌─────────────────────────────────────────┐
  │           GPU Memory (Contiguous)       │
  └─────────────────────────────────────────┘
```

**Implementation Details:**

- Lock-free task dispatch via `atomic<bool> has_work / done` flags
- Software prefetching: 16-entry lookahead with `__builtin_prefetch`
- AVX-512 streaming stores (`_mm512_stream_si512` + `_mm_sfence`): bypasses CPU cache for large sequential writes, preventing cache pollution
- Work distribution: equal-sized chunks across (workers + main thread)

#### 3.4 N-Buffered Staging

5-buffer rotation overlaps gather with DMA:

```
Time ──────────────────────────────────────────────────►

Buf 0: [Gather batch 1] [DMA batch 1        ] [free    ] [Gather batch 4]
Buf 1: [free           ] [Gather batch 2     ] [DMA batch 2        ] ...
Buf 2: [free           ] [free               ] [Gather batch 3     ] ...
Buf 3: ...
Buf 4: ...

Event synchronization ensures a buffer is not reused until its DMA completes.
```

---

### 4. Copy Engine Manager

**File:** `src/copy_engine.cpp`, `include/gfd/copy_engine.h`

Thin wrapper around CUDA Driver API for high-priority multi-stream DMA.

**Design Decisions:**

- **Driver API (not Runtime API):** Enables stream priority control and explicit context management
- **Multiple streams (3 default):** Saturates PCIe bandwidth via pipelining; hardware can overlap multiple DMA operations
- **Context pinning:** Avoids `cuCtxPushCurrent`/`cuCtxPopCurrent` per submission (saves 2-4 us)
- **Round-robin dispatch:** Distributes entries across streams for balanced utilization

```
Stream Pipeline:

Stream 0: ──[DMA A]────────[DMA D]────────[DMA G]──
Stream 1: ────[DMA B]────────[DMA E]────────[DMA H]──
Stream 2: ──────[DMA C]────────[DMA F]────────[DMA I]──

Events recorded on all streams after each submission batch.
wait_completion() synchronizes all events.
```

---

### 5. Staging Pool

**File:** `include/gfd/staging_pool.h`

**Problem Solved:** First-time allocation of hugepage-backed, CUDA-registered, NUMA-bound buffers takes ~28ms. In inference serving, this would create unacceptable latency spikes during model loading or GPU context switches.

**Solution:** Pre-allocate all staging memory once at model load time. Pollers acquire/release buffer sets without allocation overhead.

**Memory Hierarchy:**

```
Priority 1: mmap(MAP_HUGETLB) + mbind(MPOL_BIND) + cudaHostRegister
   - 2MB hugepages reduce TLB misses for large transfers
   - NUMA binding ensures local memory access
   - cudaHostRegister enables zero-copy DMA

Priority 2: cudaMallocHost + madvise(MADV_HUGEPAGE) + mbind
   - Fallback if hugepages unavailable
   - Transparent hugepage hint

Priority 3: cudaMallocHost (plain)
   - Non-Linux fallback
```

---

### 6. Topology Discovery

**File:** `include/gfd/pcie_topology.h`

Probes system topology to make optimal resource allocation decisions:

- Which CPU cores are local to each GPU's NUMA node
- How many GPUs share a NUMA node (affects CE channel allocation)
- Exclusive core partitioning to prevent thread contention

---

### 7. SG Task Queue (Dynamic Scatter-Gather)

**Files:** `include/gfd/sg_task_queue.h`, `include/gfd/sg_device_primitives.cuh`, `include/gfd/sg_warp_spec.cuh`

The SG (Scatter-Gather) task queue enables dynamic address submission for scenarios like MoE inference where token addresses are determined at runtime, rather than following a fixed linear mapping.

#### Two-Level Structure

```
SGTaskQueue
├── SGList ring buffer [512 slots]
│   Each SGList header:
│   ┌──────────────────────────────────────────┐
│   │ pool_offset | count | list_id | flags    │
│   │ sequence (commit marker)                 │
│   └──────────────────────────────────────────┘
│
└── DeviceSGEntry pool [16384 entries]
    Each entry:
    ┌──────────────────────────────────────────┐
    │ src_addr (CPU) | dst_addr (GPU) | size   │
    └──────────────────────────────────────────┘
```

**Why two levels:** Different SG lists can have different sizes (e.g., 8 entries for one expert, 256 for another). The pool allocator handles variable-sized batches without external coordination.

#### Submission Protocol

```
Compute Warp / Host
  │  1. sg_wait_entry_space() — backpressure
  │  2. sg_alloc_entries(count) — atomicAdd on entry_alloc_idx
  │  3. sg_write_entries() — warp-cooperative pool writes
  │  4. sg_alloc_list() — atomicAdd on list_alloc_idx
  │  5. sg_commit_list() — 3× __threadfence_system + sequence
  ▼
SGTaskQueue (host-mapped memory)
  │
Transfer Warp (reads committed SGLists)
  │  atomicAdd on list_read_idx (multi-block safe)
  │  → convert entries to Descriptors
  │  → commit to DescriptorQueue
  ▼
DescriptorQueue → CPU Poller (existing pipeline)
  │  → group by list_id (user_data >> 32)
  │  → per-list DMA + CE write-back
  ▼
Completion Signals
  ├── lists_completed (coarse, zero overhead)
  └── d_list_done[list_id] (fine-grained, L2 polling ~30ns)
```

#### Multi-Block Atomics

When multiple blocks run SG transfer warps:
- **List claiming:** `atomicAdd` on `list_read_idx` ensures each list is processed by exactly one block
- **Entry consumption tracking:** `atomicCAS`-based max update on `entry_consumed_idx` handles out-of-order completion across blocks
- **Backpressure:** Each block checks `entry_alloc_idx - entry_consumed_idx < MAX_SG_POOL_ENTRIES - needed` before allocating

#### SG Warp-Spec Architecture

```
Block = 64 threads (2 warps):

Warp 0 (Transfer):
  ├── atomicAdd(&list_read_idx) — claim next SGList
  ├── Poll sequence for commit
  ├── Convert entries → Descriptors (warp-parallel, 32 per iteration)
  ├── atomicAdd(&dq->write_idx) — acquire descriptor slots
  ├── Backpressure: wait for read_idx to catch up
  ├── Two-phase commit to DescriptorQueue
  ├── Update entry_consumed_idx (atomicCAS max)
  ├── Poll d_list_done[list_id] for DMA completion
  └── Signal compute warp via shared memory

Warp 1 (Compute):
  ├── Wait for list_ready signal
  ├── Build SGListView from shared state
  ├── Call user ComputeFn(SGListView)
  └── Signal compute_done to transfer warp
```

---

## Data Flow

### Complete Transfer Lifecycle

```
1. GPU Kernel Launch
   └── N threads each write one Descriptor
       └── __threadfence_system() + sequence commit

2. CPU Polling Thread detects committed entries
   └── Accumulates batch (up to 8192 or threshold)

3. Contiguity Analysis
   ├── dst contiguous + src contiguous → single DMA (fast path)
   ├── dst contiguous + src scattered  → parallel gather + single DMA
   └── generic → sort + coalesce + stage/direct

4. Parallel Gather (if needed)
   └── 15 workers + main thread: streaming_memcpy(scattered → staging)

5. CE DMA Submission
   └── cuMemcpyHtoDAsync (staging/src → GPU) on high-priority streams

6. Completion
   └── CPU writes done_idx after DMA event synchronization
       └── GPU wait kernel sees done_idx >= expected → exits
```

### Latency Breakdown (Typical 2048 × 4KB)

```
Total: ~1150 us (GFD Queue, 15 workers)

┌─────────────────────────────────────────────────┐
│ GPU submit (write+commit)    │     ~5 us        │
│ CPU poll detect + batch      │    ~10 us        │
│ Parallel gather (15 workers) │   ~200 us        │
│ CE DMA (staging → GPU)       │   ~900 us        │
│ done_idx update + GPU poll   │     ~5 us        │
└─────────────────────────────────────────────────┘

vs. cudaMemcpy(N): ~2700 us (API overhead dominates)
```

---

## Concurrency Model

### Thread Layout

```
┌────────────────────────────────────────────────────┐
│ NUMA Node 0 (Cores 0-63)                           │
│                                                    │
│  Core 0:  Main Polling Thread (spin-polls queue)   │
│  Core 2:  Gather Worker 0                          │
│  Core 4:  Gather Worker 1                          │
│  Core 6:  Gather Worker 2                          │
│  ...                                               │
│  Core 30: Gather Worker 14                         │
│                                                    │
│  (Stride 2 to avoid hyperthreading sibling         │
│   interference on physical cores)                  │
└────────────────────────────────────────────────────┘
```

### Synchronization Primitives

| Component                | Mechanism                                    | Ordering                 |
| ------------------------ | -------------------------------------------- | ------------------------ |
| GPU → CPU descriptors    | volatile sequence + `__threadfence_system()` | Release-acquire via PCIe |
| CPU → GPU done_idx       | `__atomic_store_n(RELEASE)` / volatile read  | Release-acquire          |
| Polling thread → Workers | `atomic<bool> has_work`                      | Acquire-release          |
| Workers → Polling thread | `atomic<bool> done`                          | Acquire-release          |
| Staging buffer safety    | CUevent synchronization                      | CUDA event ordering      |

### Lock-Free Design

The entire hot path (GPU write → CPU poll → gather → DMA → completion) is **lock-free**:

- No mutexes on the data path
- No condition variables (all spin-based)
- No memory allocation after initialization
- No kernel launches from the CPU poller

The only mutex is in `StagingPool` for buffer acquisition (cold path only).

---

## Memory Architecture

### Allocation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Map                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ CUDA Managed Memory (unified address space):                │
│    • DescriptorQueue (1MB) - accessible by GPU + CPU        │
│                                                             │
│ CUDA Device Memory:                                         │
│    • GPU destination buffer (user-sized)                    │
│    • TokenInfo array                                        │
│                                                             │
│ Pinned Host Memory (cudaMallocHost / mmap+cudaHostRegister):│
│    • CPU source buffer (scattered token data)               │
│    • Staging buffers (5 × buf_size, hugepage, NUMA-bound)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Managed Memory for the Queue?

- Eliminates explicit CPU↔GPU copy of control structures
- GPU can atomically write `write_idx` and read `done_idx` via PCIe BAR
- CPU can directly read/write queue entries via system memory mapping
- CUDA coherence protocol handles visibility (with `__threadfence_system()`)

### Why Pinned Host Memory for Sources?

- Required for DMA: CUDA Copy Engine needs page-locked memory
- Hugepages (2MB) reduce TLB misses during gather
- NUMA binding ensures memory is local to the polling thread's NUMA node

---

## Performance Optimizations

### 1. Warp-Level Fence Amortization

`__threadfence_system()` is expensive (~1-2 us on PCIe). By having only the warp leader execute it, we amortize the cost across 32 threads. This is the single largest GPU-side optimization.

### 2. AVX-512 Streaming Stores

Gather workers use non-temporal stores to write to staging buffers. Benefits:

- Bypasses L1/L2 cache (staging data is write-once, read-by-DMA)
- Full 64-byte cache line writes without read-for-ownership
- Prevents polluting CPU caches with transient gather data

### 3. Contiguity Detection

Before any gather or DMA, the poller checks if destination addresses are already contiguous. When tokens happen to be sequential (common in batch inference), GFD skips gather entirely and issues a single coalesced DMA.

### 4. Adaptive Batch Thresholds

Small entries (≤ 2KB) have high per-batch overhead relative to data size. GFD accumulates more (1024 vs 256) before flushing, targeting ~1MB per batch for optimal DMA efficiency.

### 5. N-Buffered Staging (Overlap Gather + DMA)

With 5 staging buffers:

- While buffer A undergoes DMA, buffer B is being gathered into
- Eliminates the serial dependency between gather and DMA
- Event-based synchronization ensures buffer safety

### 6. Context Pinning

`pin_context()` pushes the CUDA context once at thread start. Without pinning, every `cuMemcpyHtoDAsync` call would push/pop context (~2-4 us overhead × thousands of calls).

### 7. Sort + Coalesce (Generic Path)

For non-contiguous patterns, sorting by source address and merging adjacent entries reduces the number of DMA calls from N to typically N/4-N/8, dramatically reducing Driver API overhead.

---

## Configuration Tuning Guide

| Parameter               | Default          | Tuning Advice                                                           |
| ----------------------- | ---------------- | ----------------------------------------------------------------------- |
| `QUEUE_SIZE`            | 16384            | Must be power-of-two. Increase if GPU submits faster than CPU consumes. |
| `MAX_BATCH_SIZE`        | 8192             | Limit on stack arrays. Keep ≤ QUEUE_SIZE/2.                             |
| `BATCH_THRESHOLD`       | 256              | Lower = less latency, higher = better throughput for large batches.     |
| `BATCH_THRESHOLD_SMALL` | 1024             | For entries ≤ 2KB. Higher amortizes per-batch overhead.                 |
| `exclusive_core_count`  | 32               | More cores = more gather workers. Diminishing returns past 16 workers.  |
| `num_ce_channels`       | 3                | 3 for ≤2 GPUs/NUMA. Reduce if PCIe bandwidth is shared.                 |
| `NUM_STAGING_BUFS`      | 5                | 5 allows 3+ in-flight. Reduce if memory-constrained.                    |
| Staging buffer size     | `total_cpu_size` | Must fit the largest batch's contiguous range.                          |

---

## Error Handling Strategy

- **CUDA Driver API errors:** `GFD_CU_CHECK` macro logs file/line/error and returns error code (non-fatal)
- **Staging allocation failures:** Falls back progressively (hugepage → cudaMallocHost → error)
- **Queue overflow:** Not explicitly handled (ring buffer wraps). User must ensure `write_idx - done_idx < QUEUE_SIZE`
- **Worker thread failures:** Thread pinning failures are logged as warnings (non-fatal)

---

## Limitations and Constraints

1. **Unidirectional:** H2D only (Host-to-Device). D2H transfers not supported.
2. **Queue overflow protection:** User responsibility to not submit more than `QUEUE_SIZE` entries before completion.
3. **Pinned memory required:** Source buffers must be page-locked (`cudaMallocHost` or `cudaHostRegister`).
4. **Linux preferred:** Full feature set (hugepages, NUMA binding, CPU pinning) requires Linux.
5. **Single consumer:** One `CpuPollingThread` per queue (no multi-consumer support).
6. **Stack allocation:** `MAX_BATCH_SIZE` affects stack usage (~512KB for batch array). Ensure sufficient thread stack size.

---

## File Map

```
gfd/
├── CMakeLists.txt                   # Build system (static + shared lib + examples)
├── include/gfd/
│   ├── gfd.h                        # Umbrella host header
│   ├── descriptor_queue.h           # Ring buffer + Descriptor + TokenInfo + constants
│   ├── tiled_queue.h                # TiledQueue + TileScheduler + per-tile signals
│   ├── device.cuh                   # GPU __device__ API (header-only, basic mode)
│   ├── device_primitives.cuh        # GPU primitives: acquire_chunk_slots, submit_chunk, wait_chunk_done
│   ├── warp_spec.cuh                # Warp-spec framework: GFD_WARP_SPEC_KERNEL, ChunkView
│   ├── warp_spec_session.h          # WarpSpecSession + SGWarpSpecSession (single-GPU)
│   ├── sg_task_queue.h              # SG structures: DeviceSGEntry, SGList, SGTaskQueue
│   ├── sg_device_primitives.cuh     # SG GPU primitives: alloc, write, commit, wait
│   ├── sg_warp_spec.cuh             # SG warp-spec framework: GFD_SG_WARP_SPEC_KERNEL
│   ├── copy_engine.h                # CopyEngineManager class declaration
│   ├── cpu_polling.h                # CpuPollingThread class declaration
│   ├── staging_pool.h               # StagingPool singleton (header-only impl)
│   ├── pcie_topology.h              # Topology discovery (header-only impl)
│   └── log.h                        # Structured logging (compile-time levels)
├── src/
│   ├── copy_engine.cpp              # CopyEngineManager implementation (3 streams)
│   ├── cpu_polling.cpp              # CpuPollingThread: polling loop, tile event drain
│   ├── batch_processor.cpp          # Batch processing, per-tile DMA, CE write-back
│   ├── parallel_gather.cpp          # AVX-512 parallel gather workers (up to 15)
│   ├── direct_submit.cpp            # Direct-submit fast path with pipelined gather+DMA
│   ├── warp_spec_session.cpp        # WarpSpecSession lifecycle management
│   └── sg_warp_spec_session.cpp     # SGWarpSpecSession lifecycle management
├── examples/
│   ├── basic_transfer.cu            # Fused kernel demo with verification
│   ├── benchmark.cu                 # Comprehensive latency/bandwidth benchmark
│   ├── direct_transfer.cu           # CPU-initiated direct transfer demo
│   ├── gpu_planned_transfer.cu      # GPU-planned transfer pattern
│   ├── warp_spec_simple.cu          # Single-GPU warp-spec transfer+compute
│   ├── multi_gpu_warp_spec.cu       # 8-GPU warp-spec with NUMA-aware core pinning
│   ├── multi_gpu_benchmark.cu       # Multi-GPU bandwidth scaling
│   ├── multi_gpu_direct.cu          # Multi-GPU direct transfer
│   └── sg_warp_spec.cu             # SG scatter-gather warp-spec + benchmarks
└── docs/
    ├── API_Reference.md             # Complete API documentation
    └── Architecture.md              # This architecture document
```

---

## Comparison with Alternatives

| Approach             | Mechanism                      | Scattered 2048×4KB                      | Pros                        | Cons                                    |
| -------------------- | ------------------------------ | --------------------------------------- | --------------------------- | --------------------------------------- |
| `cudaMemcpy(N)`      | N individual API calls         | ~2.7 ms (3.1 GB/s)                      | Simple, portable            | API overhead dominates                  |
| `cudaMemcpy2D`       | 2D pitched copy                | N/A (requires rectangular layout)       | Single call                 | Requires specific memory layout         |
| CUDA Graphs          | Captured memcpy nodes          | ~2 ms                                   | Reduced launch overhead     | Static graph, no dynamic scatter        |
| UVM (managed memory) | Page fault + migration         | Variable (fault latency ~20-50 us/page) | Transparent                 | Unpredictable latency, page granularity |
| **GFD Queue**        | GPU submit → CPU poll → CE DMA | **~1.15 ms (7.3 GB/s)**                 | Dynamic, overlapped compute | Requires dedicated CPU cores            |
| **GFD Warp-Spec**    | Tile-chunked + warp overlap    | 128MB: **2.94 ms (43.6 GB/s)**          | Transfer+compute overlap    | Requires dedicated CPU cores + NUMA     |
| **GFD SG Warp-Spec** | Dynamic SG + warp overlap      | 64MB: **1.32 ms (51 GB/s)** (8 blocks)  | Dynamic addresses, MoE      | Requires dedicated CPU cores            |
| **GFD Direct**       | CPU direct CE submit           | **~0.15 ms (53 GB/s)**                  | Lowest latency, highest BW  | CPU-initiated only                      |

---

## Tiled Transfer-Compute Architecture (Warp Specialization)

### Overview

The tiled architecture enables **per-tile (per-SM) completion signaling** so each GPU block can start computing as soon as its own data arrives, without waiting for all tiles to complete. Combined with warp specialization, sub-tile chunking allows compute to overlap with transfer at the finest granularity.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     GPU: Warp-Specialized Tiled Kernel                       │
│                                                                              │
│  Block 0 (SM 0)          Block 1 (SM 1)          Block N (SM N)              │
│  ┌───────────────┐       ┌───────────────┐       ┌───────────────┐           │
│  │ Warp 0: Xfer  │       │ Warp 0: Xfer  │       │ Warp 0: Xfer  │           │
│  │  atomicAdd()  │       │  atomicAdd()  │       │  atomicAdd()  │           │
│  │  write descs  │       │  write descs  │       │  write descs  │           │
│  │  poll tile_done│      │  poll tile_done│      │  poll tile_done│          │
│  ├───────────────┤       ├───────────────┤       ├───────────────┤           │
│  │ Warp 1: Comp  │       │ Warp 1: Comp  │       │ Warp 1: Comp  │           │
│  │ (chunk 0)     │       │ (chunk 0)     │       │ (chunk 0)     │           │
│  │ Warp 2: Comp  │       │ Warp 2: Comp  │       │ Warp 2: Comp  │           │
│  │ (chunk 1)     │       │ (chunk 1)     │       │ (chunk 1)     │           │
│  │ Warp 3: Comp  │       │ Warp 3: Comp  │       │ Warp 3: Comp  │           │
│  │ (chunk 2)     │       │ (chunk 2)     │       │ (chunk 2)     │           │
│  │ Warp 4: Comp  │       │ Warp 4: Comp  │       │ Warp 4: Comp  │           │
│  │ (chunk 3)     │       │ (chunk 3)     │       │ (chunk 3)     │           │
│  └───────────────┘       └───────────────┘       └───────────────┘           │
│         │ atomicAdd             │ atomicAdd             │ atomicAdd          │
│         ▼ interleaved           ▼ interleaved           ▼ interleaved        │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │     DescriptorQueue (Managed Memory) — interleaved entries       │        │
│  │  [T0C0|T5C0|T12C0|T1C0|T32C0|T0C1|T7C0|T63C0|T5C1|T1C1|...]      │        │
│  └──────────────────────────────────────────┬───────────────────────┘        │
│                                             │                                │
│  ┌────────────────────────────────┐         │                                │
│  │  TiledQueue.tile_done[MAX_TILES]│ ◄──────┼──── CE write-back (8 bytes)    │
│  │  [0]=96, [1]=64, [5]=128, ...  │         │                                │
│  └────────────────────────────────┘         │                                │
└─────────────────────────────────────────────┼────────────────────────────────┘
                                              │ sequence commit (PCIe)
┌─────────────────────────────────────────────┼──────────────────────────────────┐
│                     CPU: Polling + Per-Tile DMA                                │
│                                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐                   │
│  │          CpuPollingThread                               │                   │
│  │                                                         │                   │
│  │  1. Poll entries (interleaved from many tiles)          │                   │
│  │  2. Bucket entries by tile_id (user_data >> 32)         │                   │
│  │  3. Per-tile DMA: submit contiguous DMA per tile        │                   │
│  │  4. CE write-back: append 8B write to tile_done[tid]    │                   │
│  │     tile_progress[tid] += count                         │                   │
│  │     tile_signal_buf[tid] = tile_progress[tid]           │                   │
│  │     CE DMA: signal_buf → tile_done[tid]                 │                   │
│  └─────────────────────────────────────────────────────────┘                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Key Design: Interleaved Slot Acquisition

**Problem:** If each tile pre-assigns contiguous slots, tile 0 occupies slots 0-127, tile 1 occupies 128-255, etc. The CPU must process tile 0's 128 entries before seeing tile 1's data — creating head-of-line blocking.

**Solution:** Each tile's transfer warp uses `atomicAdd(&write_idx, CHUNK_TOKENS)` per chunk (32 tokens). Multiple SMs execute concurrently, so their chunks interleave naturally:

```
Queue slot order (64 tiles, each chunk = 32 tokens):
  [T0:C0] [T5:C0] [T12:C0] [T1:C0] ... [T0:C1] [T7:C0] [T63:C0] ...

CPU sees mixed entries → groups by tile_id → per-tile DMA
```

### Key Design: Count-Based Tile Signaling

`tile_done[tile_id]` stores the **cumulative token count** for that tile (not a slot index). This works correctly regardless of slot interleaving:

```
CPU processes batch containing: 32 entries from tile 0, 32 from tile 5, 32 from tile 12
→ tile_progress[0] += 32 → CE writes tile_done[0] = 32
→ tile_progress[5] += 32 → CE writes tile_done[5] = 32
→ tile_progress[12] += 32 → CE writes tile_done[12] = 32

GPU Block 0, Warp 1 (chunk 0): waits for tile_done[0] >= 32 → starts compute
GPU Block 0, Warp 2 (chunk 1): waits for tile_done[0] >= 64 → starts compute
```

### Key Design: CE Write-Back (Zero-CPU-Overhead Signaling)

After each tile's DMA, the CPU appends an 8-byte CE write from pinned `tile_signal_buf_[tile_id]` to managed `tile_done[tile_id]`. The Copy Engine executes this write in-stream after the data DMA completes — no CPU polling, no cuEventQuery, zero CPU overhead.

```cpp
// In process_batch_tiled():
tile_progress_[tile_id] += tile_count;
tile_signal_buf_[tile_id] = tile_progress_[tile_id];  // pinned host memory
SGEntry signal = { .dst = &tile_done[tile_id], .src = &signal_buf[tile_id], .size = 8 };
ce_manager_.submit_scatter_gather(&signal, 1);        // appended to CE stream
```

### Warp Specialization Design

```
Block = 160 threads (5 warps per tile):

Warp 0 (Transfer):
  ├── Submit 4 chunks × 32 descriptors via atomicAdd
  ├── Each chunk's last entry has FLAG_LAST_IN_TILE → triggers CPU flush
  └── Poll tile_done[tile_id] for progressive chunk arrival
      └── Signal compute warps via shared memory: chunks_ready++

Warps 1-4 (Compute):
  ├── Warp 1: waits for chunks_ready >= 1, computes tokens [0..31]
  ├── Warp 2: waits for chunks_ready >= 2, computes tokens [32..63]
  ├── Warp 3: waits for chunks_ready >= 3, computes tokens [64..95]
  └── Warp 4: waits for chunks_ready >= 4, computes tokens [96..127]
```

**Timeline showing overlap:**

```
Time ──────────────────────────────────────────────────────────────────►

Warp 0:  [submit C0][submit C1][submit C2][submit C3][poll C0][poll C1][poll C2][poll C3]
Warp 1:  [────────────wait────────────────][compute chunk 0]
Warp 2:  [──────────────────wait──────────────────────][compute chunk 1]
Warp 3:  [────────────────────────wait────────────────────────][compute chunk 2]
Warp 4:  [──────────────────────────────wait──────────────────────────][compute chunk 3]

CPU:     [process mixed entries][per-tile DMA + CE writeback][...]
```

### Performance Results

**Single GPU — RTX PRO 5000 72GB (Blackwell, sm_120)**

Configuration: 8192 tokens × 16KB = 128 MB, 64 tiles × 128 tokens, K=4 chunks/tile

| Mode                                      | Latency      | Bandwidth      | Speedup   |
| ----------------------------------------- | ------------ | -------------- | --------- |
| **Warp-Spec Pure Transfer (NoOp)**        | **2.94 ms**  | **43.6 GB/s**  | **3.0x**  |
| **Warp-Spec + Compute (RMSNorm+sinf)**    | **4.09 ms**  | **32.8 GB/s**  | **2.2x**  |
| Baseline (global wait then compute)       | 19.5 ms      | 6.6 GB/s       | 1.0x      |

**8-GPU Parallel — 8× RTX PRO 5000, NUMA-aware core pinning**

Configuration: 128 MB per GPU, 16 cores per GPU (poller + 7 workers)

| Mode                        | Aggregate BW    | Scaling Efficiency |
| --------------------------- | --------------- | ------------------ |
| 8-GPU Warp-Spec + Compute   | **250 GB/s**    | 93.6%              |
| 8-GPU Pure Transfer         | **340 GB/s**    | 95.8%              |

Key observation: compute warps start processing chunk 0's data while the transfer warp is still submitting chunks 1-3. This provides ~40% bandwidth improvement over the global-wait approach.

---

---

## Device-Memory Signal Path

### Problem: Host-Mapped Polling Latency

With `tile_chunk_done[]` in host-mapped memory, each GPU read traverses PCIe (~1500ns round-trip). For tight transfer-compute overlap, this is unacceptable.

### Solution: `d_tile_chunk_done` in Device Memory

When `TiledQueue::d_tile_chunk_done` is set (non-nullptr), the CPU poller signals through device memory instead:

```
CPU Poller                           GPU SM
    │                                  │
    │  1. Process tile entries          │
    │  2. Submit per-tile DMA           │
    │  3. make_stream_wait_on_all()     │
    │     (signal_stream waits on CE)   │
    │  4. cuMemcpyHtoDAsync             │
    │     tile_signal_buf_[tid]         │
    │     → d_tile_chunk_done[tid]      │
    │     (8-byte signal write)         │
    │                                  │ ◄── L2 cache hit (~30ns)
    │                                  │     polls d_tile_chunk_done[tid]
```

**Benefits:**
- GPU polls L2 cache (~30ns) instead of PCIe (~1500ns) — **50x faster signal delivery**
- Signal stream ensures data→signal ordering without CPU blocking
- CE write-back: signal appended in-stream after data DMA, zero CPU overhead

### Implementation (src/batch_processor.cpp)

```cpp
// After per-tile DMA completes:
tile_progress_[tile_id] += tile_count;
tile_signal_buf_[tile_id] = tile_progress_[tile_id];

// Make signal_stream wait for all CE data channels
ce_manager_.make_stream_wait_on_all(signal_stream_);

// 8-byte signal write to device memory (GPU polls L2)
SGEntry signal = {
    .dst = (CUdeviceptr)&tq->d_tile_chunk_done[tile_id],
    .src = &tile_signal_buf_[tile_id],
    .size = sizeof(uint64_t)
};
cuMemcpyHtoDAsync(signal.dst, signal.src, signal.size, signal_stream_);
```

---

## Multi-GPU Architecture

### NUMA-Aware Core Partitioning

For 8-GPU systems, GFD partitions CPU cores to avoid cross-NUMA traffic:

```
NUMA Node 0 (Cores 0-63)               NUMA Node 1 (Cores 64-127)
┌───────────────────────────────┐      ┌───────────────────────────────┐
│ GPU 0: cores  0-15 (poller+7w)│      │ GPU 4: cores 64-79            │
│ GPU 1: cores 16-31 (poller+7w)│      │ GPU 5: cores 80-95            │
│ GPU 2: cores 32-47 (poller+7w)│      │ GPU 6: cores 96-111           │
│ GPU 3: cores 48-63 (poller+7w)│      │ GPU 7: cores 112-127          │
└───────────────────────────────┘      └───────────────────────────────┘
```

Each GPU's `CpuPollingThread` is created with `exclusive_core_base` and `exclusive_core_count`:
- Poller thread pinned to `core_base`
- Gather workers pinned to `core_base + 2, core_base + 4, ...` (stride 2 for HT avoidance)
- Staging buffers NUMA-bound via `mbind(MPOL_BIND)`

### Multi-GPU WarpSpecSession Limitation

`WarpSpecSession` does not expose `exclusive_core_base/count`. For multi-GPU, use the low-level API:

```cpp
// Per-GPU setup (see examples/multi_gpu_warp_spec.cu)
for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    // ... allocate TiledQueue, gpu_data, cpu_data ...

    poller[i] = new gfd::CpuPollingThread(
        &tq[i]->base, gpu_data[i], cpu_data[i], TOTAL_SIZE,
        /*use_ce=*/true, /*numa_node=*/cfg[i].numa_node,
        /*core_offset=*/0, /*num_ce_channels=*/0,
        /*exclusive_core_base=*/cfg[i].core_base,
        /*exclusive_core_count=*/cfg[i].core_count);
    poller[i]->set_tiled_queue(tq[i]);
    poller[i]->init_copy_engine();
}
```

### Scaling Results (8× RTX PRO 5000)

| GPUs | P50 Latency | Aggregate BW | Scaling Efficiency |
|------|------------|-------------|-------------------|
| 1    | 2.94 ms    | 43.6 GB/s   | 100%              |
| 2    | 2.97 ms    | 86.2 GB/s   | 98.9%             |
| 4    | 3.02 ms    | 169.5 GB/s  | 97.3%             |
| 8    | 3.10 ms    | 330.2 GB/s  | 94.7%             |

---

## SG Warp-Spec Performance Results

**Hardware:** RTX PRO 5000 72GB (Blackwell, sm_120), 64 MB total transfer

| Mode | Blocks | P50 Latency | Bandwidth |
|------|--------|-------------|-----------|
| SG | 1 | 2.16 ms | 31.0 GB/s |
| SG | 8 | 1.36 ms | 49.2 GB/s |
| Linear-opt (K=1) | 1 | 1.56 ms | 43.2 GB/s |
| Linear-opt (K=1) | 8 | 1.32 ms | 51.0 GB/s |

**Key observations:**
- SG mode with 8 blocks reaches 49.2 GB/s, within 4% of linear mode (51.0 GB/s)
- The small overhead comes from the extra indirection through the SGList + entry pool structure
- Multi-block scaling is nearly linear: 8 blocks achieve ~1.6x the bandwidth of 1 block in SG mode
- SG mode enables dynamic address patterns (MoE routing) that linear mode cannot handle

---

## Future Directions

1. **D2H support:** Reverse path for GPU→CPU result writeback
2. **Multi-queue:** Multiple independent queues for priority isolation
3. **Compression:** Inline LZ4/zstd for bandwidth amplification
4. **RDMA integration:** Direct NIC→GPU for disaggregated inference
5. **Dynamic worker scaling:** Adjust gather worker count based on load
6. **Adaptive chunk sizing:** Auto-tune CHUNK_TOKENS based on compute intensity and transfer bandwidth
7. **Batched tile signals:** Reduce per-tile CE overhead by coalescing signal writes (see tile_plan.md)
8. **submit_tile mode:** Single-shot tile submission for maximum pure-transfer bandwidth
