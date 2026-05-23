#pragma once

#include "gfd/sg_task_queue.h"

// ============================================================
// GFD SG Device Primitives
//
// Warp-collective GPU-side primitives for SG task queue operations.
// All functions assume full-warp participation (all 32 lanes active).
//
// Usage pattern:
//   1. sg_alloc_entries(sq, N)   → get pool offset for N entries
//   2. sg_write_entries(...)     → warp-cooperative write entries
//   3. sg_alloc_list(sq)         → get list ring slot
//   4. sg_commit_list(...)       → two-phase fence + sequence commit
//
// Or use the all-in-one convenience:
//   sg_submit_list(sq, entries, count, list_id, flags)
// ============================================================

namespace gfd {
namespace sg {

// ---- Backpressure: wait for entry pool space ----
// Spins until (entry_alloc_idx - entry_consumed_idx) < MAX_SG_POOL_ENTRIES - needed.
// Only lane 0 polls; all lanes sync after.
__device__ __forceinline__
void sg_wait_entry_space(SGTaskQueue* sq, uint32_t needed) {
    if ((threadIdx.x & 31) == 0) {
        while (true) {
            uint64_t alloc = sq->entry_alloc_idx;
            uint64_t consumed = *((volatile uint64_t*)&sq->entry_consumed_idx);
            if (alloc - consumed <= (uint64_t)(MAX_SG_POOL_ENTRIES - needed)) break;
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
    }
    __syncwarp();
}

// ---- Backpressure: wait for list ring space ----
// Spins until (list_alloc_idx - list_read_idx) < MAX_SG_LISTS - 1.
__device__ __forceinline__
void sg_wait_list_space(SGTaskQueue* sq) {
    if ((threadIdx.x & 31) == 0) {
        while (true) {
            uint64_t alloc = sq->list_alloc_idx;
            uint64_t read = *((volatile uint64_t*)&sq->list_read_idx);
            if (alloc - read < (uint64_t)(MAX_SG_LISTS - 1)) break;
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
    }
    __syncwarp();
}

// ---- Allocate entry pool slots (warp-collective) ----
// Lane 0 atomicAdd on entry_alloc_idx, result broadcast to all lanes.
// Returns the starting pool offset (modular into entries[]).
__device__ __forceinline__
uint64_t sg_alloc_entries(SGTaskQueue* sq, uint32_t count) {
    uint64_t base = 0;
    if ((threadIdx.x & 31) == 0) {
        base = atomicAdd((unsigned long long*)&sq->entry_alloc_idx,
                         (unsigned long long)count);
    }
    base = __shfl_sync(0xFFFFFFFF, base, 0);
    return base;
}

// ---- Write entries into pool (warp-collective) ----
// Cooperatively writes `count` DeviceSGEntries starting at pool_offset.
// Each lane handles entries in stride-32 fashion.
// `src_entries` is a device-accessible array of DeviceSGEntry.
__device__ __forceinline__
void sg_write_entries(
    SGTaskQueue* sq,
    uint64_t pool_offset,
    const DeviceSGEntry* src_entries,
    uint32_t count)
{
    const int lane_id = threadIdx.x & 31;
    for (uint32_t i = lane_id; i < count; i += 32) {
        uint32_t slot = (uint32_t)((pool_offset + i) % MAX_SG_POOL_ENTRIES);
        sq->entries[slot] = src_entries[i];
    }
}

// ---- Write entries with inline parameters (warp-collective) ----
// Each lane writes one entry per iteration. Caller provides parallel arrays.
__device__ __forceinline__
void sg_write_entries_inline(
    SGTaskQueue* sq,
    uint64_t pool_offset,
    const uint64_t* src_addrs,
    const uint64_t* dst_addrs,
    const uint32_t* sizes,
    uint32_t tag,
    uint32_t count)
{
    const int lane_id = threadIdx.x & 31;
    for (uint32_t i = lane_id; i < count; i += 32) {
        uint32_t slot = (uint32_t)((pool_offset + i) % MAX_SG_POOL_ENTRIES);
        sq->entries[slot].src_addr = src_addrs[i];
        sq->entries[slot].dst_addr = dst_addrs[i];
        sq->entries[slot].size = sizes[i];
        sq->entries[slot].tag = tag;
    }
}

// ---- Allocate a list ring slot (warp-collective) ----
// Lane 0 atomicAdd on list_alloc_idx, result broadcast.
// Returns the list slot index (modular into lists[]).
__device__ __forceinline__
uint64_t sg_alloc_list(SGTaskQueue* sq) {
    uint64_t slot = 0;
    if ((threadIdx.x & 31) == 0) {
        slot = atomicAdd((unsigned long long*)&sq->list_alloc_idx, 1ULL);
    }
    slot = __shfl_sync(0xFFFFFFFF, slot, 0);
    return slot;
}

// ---- Commit a list (warp-collective) ----
// Two-phase fence protocol (consistent with existing commit_chunk):
//   1. __threadfence_system() — entry writes visible to CPU
//   2. Write list header fields
//   3. __threadfence_system() — header visible to CPU
//   4. Write sequence (commit marker)
//   5. __threadfence_system() — sequence visible to transfer warp/CPU
__device__ __forceinline__
void sg_commit_list(
    SGTaskQueue* sq,
    uint64_t slot,
    uint32_t pool_offset,
    uint32_t count,
    uint32_t list_id,
    uint32_t flags)
{
    const int lane_id = threadIdx.x & 31;
    uint32_t ring_slot = (uint32_t)(slot % MAX_SG_LISTS);

    // Phase 1: ensure all entry writes are visible across PCIe
    __threadfence_system();
    __syncwarp();

    // Write list header (lane 0 only)
    if (lane_id == 0) {
        sq->lists[ring_slot].pool_offset = (uint32_t)(pool_offset % MAX_SG_POOL_ENTRIES);
        sq->lists[ring_slot].count = count;
        sq->lists[ring_slot].list_id = list_id;
        sq->lists[ring_slot].flags = flags;
    }
    __syncwarp();

    // Phase 2: ensure header fields visible before sequence commit
    __threadfence_system();
    __syncwarp();

    // Write sequence marker (commit point)
    if (lane_id == 0) {
        sq->lists[ring_slot].sequence = slot + 1;
    }

    // Phase 3: ensure sequence visible to consumer
    __threadfence_system();
    __syncwarp();
}

// ---- All-in-one: allocate + write + commit (warp-collective) ----
// Convenience function that performs the full submission pipeline.
// `entries` must be device-accessible.
// Returns the list slot index for tracking.
__device__ __forceinline__
uint64_t sg_submit_list(
    SGTaskQueue* sq,
    const DeviceSGEntry* entries,
    uint32_t count,
    uint32_t list_id,
    uint32_t flags = SG_FLAG_NONE)
{
    // Backpressure checks
    sg_wait_entry_space(sq, count);
    sg_wait_list_space(sq);

    // Allocate pool entries
    uint64_t pool_offset = sg_alloc_entries(sq, count);

    // Write entries
    sg_write_entries(sq, pool_offset, entries, count);

    // Allocate list slot
    uint64_t list_slot = sg_alloc_list(sq);

    // Commit
    sg_commit_list(sq, list_slot, (uint32_t)pool_offset, count, list_id, flags);

    return list_slot;
}

// ---- Wait for coarse completion (poll lists_completed) ----
// Blocks until lists_completed >= expected.
// Low-overhead: single counter, no per-list granularity.
__device__ __forceinline__
void sg_wait_list_done(SGTaskQueue* sq, uint64_t expected) {
    if ((threadIdx.x & 31) == 0) {
        int spin_count = 0;
        while (true) {
            uint64_t done = *((volatile uint64_t*)&sq->lists_completed);
            if (done >= expected) break;
#if __CUDA_ARCH__ >= 700
            if (++spin_count > 64) {
                __nanosleep(spin_count < 256 ? 100 : 1000);
            }
#endif
        }
    }
    __syncwarp();
}

// ---- Wait for specific list_id completion (poll d_list_done) ----
// Fine-grained: polls device memory d_list_done[list_id].
// The CPU poller writes a non-zero value after DMA completes.
__device__ __forceinline__
void sg_wait_list_id_done(SGTaskQueue* sq, uint32_t list_id) {
    if ((threadIdx.x & 31) == 0) {
        volatile uint64_t* signal = (volatile uint64_t*)&sq->d_list_done[list_id];
        int spin_count = 0;
        while (true) {
            uint64_t val = *signal;
            if (val != 0) break;
#if __CUDA_ARCH__ >= 700
            if (++spin_count > 64) {
                __nanosleep(spin_count < 256 ? 100 : 1000);
            }
#endif
        }
    }
    __syncwarp();
}

}  // namespace sg
}  // namespace gfd
