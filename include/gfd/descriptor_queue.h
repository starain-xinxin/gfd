#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gfd {

// Queue configuration
constexpr int QUEUE_SIZE = 16384;
constexpr int MAX_BATCH_SIZE = 8192;
constexpr int BATCH_THRESHOLD = 4096;

// Adaptive batch threshold: for small entries, accumulate more before
// processing to reduce per-batch overhead (dispatch, DMA submit, sync).
// Target: ~1MB of data per batch submission for optimal DMA efficiency.
// Only applies to very small entries (<=2KB) where per-entry overhead
// dominates. Larger entries (4KB+) already achieve good DMA efficiency
// with the default threshold and benefit from gather-DMA overlap.
constexpr size_t ADAPTIVE_BATCH_TARGET_BYTES = 1024 * 1024;  // 1MB
constexpr int BATCH_THRESHOLD_SMALL = 1024;  // entry_size <= 2KB

// Descriptor flags
constexpr uint32_t FLAG_NONE = 0;
constexpr uint32_t FLAG_LAST_IN_BATCH = 1;
constexpr uint32_t FLAG_URGENT = 2;

// Descriptor structure (cache-line aligned, 64 bytes)
struct __align__(64) Descriptor {
    uint64_t src_addr;   // CPU memory source address
    uint64_t dst_addr;   // GPU memory destination address
    uint32_t size;       // Transfer size in bytes
    uint32_t flags;      // Control flags
    uint64_t user_data;  // User-defined (e.g. token_id, expert_id)
    volatile uint64_t sequence;  // Commit marker: slot+1 when ready, 0 when consumed
};

static_assert(sizeof(Descriptor) == 64, "Descriptor must be 64 bytes");

// Lock-free ring buffer descriptor queue
struct DescriptorQueue {
    Descriptor entries[QUEUE_SIZE];
    volatile uint64_t write_idx;  // GPU writes (atomicAdd)
    volatile uint64_t read_idx;   // CPU writes
    volatile uint64_t done_idx;   // CPU writes, GPU polls
    uint8_t _padding[64 - 24];   // Avoid false sharing
};

// Token info for GPU-side request
struct TokenInfo {
    uint64_t cpu_addr;   // Source address in CPU memory
    uint32_t token_id;   // Token identifier
    uint32_t expert_id;  // Expert identifier
};

}  // namespace gfd
