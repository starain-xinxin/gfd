#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gfd {

// ============================================================
// SG Task Queue: Scatter-Gather address indirection for GFD
//
// Enables dynamic (src, dst, size) address submission independent
// of tile-based scheduling. Supports both host pre-fill and
// GPU runtime dynamic submission (e.g., MoE expert routing).
//
// Architecture:
//   SGTaskQueue (host-mapped)
//     ├── SGList ring buffer  [MAX_SG_LISTS slots]
//     └── DeviceSGEntry pool  [MAX_SG_POOL_ENTRIES entries]
//
// Producers (Compute Warp / Host):
//   sg_alloc_entries() → sg_write_entries() → sg_commit_list()
//
// Consumer (Transfer Warp):
//   reads committed SGLists → converts to Descriptors → DescriptorQueue
// ============================================================

// Capacity constants
constexpr int MAX_SG_LISTS = 512;
constexpr int MAX_SG_POOL_ENTRIES = 16384;

// SG flags
constexpr uint32_t SG_FLAG_NONE = 0;
constexpr uint32_t SG_FLAG_HOST_SUBMITTED = 1;  // List was submitted by host (not GPU)

// A single scatter-gather entry: arbitrary (src, dst, size) triple.
// 32-byte aligned for coalesced GPU access (2 lanes per entry in a warp).
struct __align__(8) DeviceSGEntry {
    uint64_t src_addr;   // CPU source address
    uint64_t dst_addr;   // GPU destination address
    uint32_t size;       // Transfer size in bytes
    uint32_t tag;        // User grouping/tracking tag
};

static_assert(sizeof(DeviceSGEntry) == 24, "DeviceSGEntry must be 24 bytes");

// SG list header: describes a batch of entries in the pool.
// 32-byte aligned for clean cache-line access.
struct __align__(16) SGList {
    uint32_t pool_offset;        // Start index in entry pool
    uint32_t count;              // Number of entries in this list
    uint32_t list_id;            // User-assigned ID for completion tracking
    uint32_t flags;              // SG_FLAG_*
    volatile uint64_t sequence;  // Commit marker: slot+1 when ready, 0 when consumed
    uint64_t _pad;               // Pad to 32 bytes
};

static_assert(sizeof(SGList) == 32, "SGList must be 32 bytes");

// The main SG task queue, allocated in host-mapped memory (cudaHostAllocMapped)
// for shared CPU+GPU access.
//
// Layout:
//   - lists[]: ring buffer of SG list headers (producers write, transfer warp reads)
//   - entries[]: flat pool of SG entries (producers write, transfer warp reads)
//   - Coordination indices are cache-line separated to avoid false sharing.
struct SGTaskQueue {
    // ---- Data ----
    SGList lists[MAX_SG_LISTS];              // SG list ring buffer
    DeviceSGEntry entries[MAX_SG_POOL_ENTRIES]; // Entry pool

    // ---- Producer coordination (cache-line aligned) ----
    alignas(64) volatile uint64_t list_alloc_idx;    // Producer: atomicAdd to allocate list slot
    alignas(64) volatile uint64_t entry_alloc_idx;   // Producer: atomicAdd to allocate entries

    // ---- Consumer coordination (cache-line aligned) ----
    alignas(64) volatile uint64_t list_read_idx;     // Transfer warp: next list to process
    alignas(64) volatile uint64_t entry_consumed_idx; // Transfer warp: entries consumed so far

    // ---- Completion signaling ----
    alignas(64) volatile uint64_t lists_completed;   // Coarse completion counter (CPU increments)
    volatile uint32_t terminate;                      // Termination flag (set by session)
    uint32_t _pad_term;

    // Device memory per-list completion signal (allocated separately via cudaMalloc).
    // d_list_done[list_id] is written by CPU poller after DMA completes.
    // GPU polls this for fine-grained per-list completion.
    uint64_t* d_list_done;
};

// ============================================================
// Backward compatibility: convert linear address mapping to SG entries
// ============================================================
namespace sg_compat {

// Convert a linear address range (cpu_base + idx * token_size) into
// DeviceSGEntry array. This allows existing linear-mode workloads
// to run through the SG pipeline unchanged.
//
// entries: output array (must have capacity >= count)
// cpu_base: CPU source base address
// gpu_base: GPU destination base address
// token_size: bytes per token
// start_idx: first global token index
// count: number of tokens
// tag: user tag for all entries
inline void linear_to_sg_entries(
    DeviceSGEntry* entries,
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size,
    uint32_t start_idx,
    uint32_t count,
    uint32_t tag = 0)
{
    for (uint32_t i = 0; i < count; i++) {
        uint32_t global_idx = start_idx + i;
        entries[i].src_addr = (uint64_t)cpu_base + (uint64_t)global_idx * token_size;
        entries[i].dst_addr = (uint64_t)gpu_base + (uint64_t)global_idx * token_size;
        entries[i].size = token_size;
        entries[i].tag = tag;
    }
}

}  // namespace sg_compat
}  // namespace gfd
