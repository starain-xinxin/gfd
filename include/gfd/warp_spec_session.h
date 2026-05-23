#pragma once

#include "gfd/tiled_queue.h"
#include "gfd/sg_task_queue.h"
#include "gfd/cpu_polling.h"
#include "gfd/staging_pool.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <chrono>

namespace gfd {

// Configuration for a warp-specialized transfer+compute session.
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
    bool double_buffer = false;        // Use double-buffer ping-pong mode
    bool per_tile_mode = false;        // K=1: entire tile as single chunk (fewer fences)

    // === Auto-tuning ===
    // Selects tokens_per_chunk based on token_size to balance DMA granularity vs overhead.
    // Returns the effective tokens_per_chunk (user-specified or auto-tuned).
    uint32_t effective_tokens_per_chunk() const {
        if (per_tile_mode) return tokens_per_tile;
        if (tokens_per_chunk > 0) return tokens_per_chunk;
        // Heuristic: aim for ~64KB-256KB per chunk DMA
        // Smaller tokens → more tokens per chunk; larger tokens → fewer
        uint64_t target_chunk_bytes = 128 * 1024;  // 128KB default target
        if (token_size <= 512) target_chunk_bytes = 64 * 1024;       // small tokens: 64KB
        else if (token_size >= 65536) target_chunk_bytes = 256 * 1024; // large tokens: 256KB
        uint32_t auto_chunk = (uint32_t)(target_chunk_bytes / token_size);
        if (auto_chunk < 4) auto_chunk = 4;
        if (auto_chunk > tokens_per_tile) auto_chunk = tokens_per_tile;
        // Enforce max K=8 chunks per tile to prevent excessive warp count
        // and register pressure. K>8 → block>288 threads → poor occupancy.
        uint32_t min_chunk = tokens_per_tile / 8;
        if (min_chunk > 0 && auto_chunk < min_chunk) auto_chunk = min_chunk;
        // Round down to power of 2 for alignment
        uint32_t p = 1;
        while (p * 2 <= auto_chunk) p *= 2;
        return p;
    }

    // === Derived (computed automatically) ===
    uint32_t total_tiles() const { return total_tokens / tokens_per_tile; }
    uint32_t chunks_per_tile() const { return tokens_per_tile / effective_tokens_per_chunk(); }
    // Single-buffer: 1 transfer + K compute warps
    // Double-buffer: 1 transfer + 2K compute warps
    int block_size() const {
        int K = chunks_per_tile();
        return double_buffer ? (1 + 2 * K) * 32 : (1 + K) * 32;
    }
};

// WarpSpecSession: manages the full lifecycle of a warp-specialized
// transfer+compute operation. Encapsulates TiledQueue, CPU poller,
// staging pool, and kernel launch configuration.
//
// Usage:
//   WarpSpecSession session(config);
//   session.launch(my_kernel, MyCompute{...});
//   session.synchronize();
//   auto stats = session.get_stats();
class WarpSpecSession {
public:
    explicit WarpSpecSession(const WarpSpecConfig& config);
    ~WarpSpecSession();

    // Non-copyable
    WarpSpecSession(const WarpSpecSession&) = delete;
    WarpSpecSession& operator=(const WarpSpecSession&) = delete;

    // ---- Launch API ----

    // Launch a GFD_WARP_SPEC_KERNEL-generated kernel with the given compute functor.
    template<typename KernelFn, typename ComputeFn>
    void launch(KernelFn kernel, ComputeFn compute_fn, cudaStream_t stream = 0);

    // Get raw launch parameters for manual kernel invocation.
    struct LaunchParams {
        TiledQueue* tq;
        void* gpu_buffer;
        const void* cpu_base;
        dim3 grid;
        dim3 block;
    };
    LaunchParams get_launch_params() const;

    // ---- Lifecycle ----

    // Wait for the current launch to complete (kernel + all DMA).
    void synchronize();

    // Reset session state for another launch (clears tile_chunk_done, next_tile).
    void reset();

    // ---- Diagnostics ----

    struct Stats {
        uint64_t descriptors_processed;
        uint64_t bytes_transferred;
        double elapsed_ms;
        double bandwidth_gbps;
    };
    Stats get_stats() const;

    // Access the underlying queue (for advanced use)
    TiledQueue* get_queue() const { return tq_; }

    // Access the CPU poller (for stats/diagnostics)
    CpuPollingThread* get_poller() const { return poller_; }

private:
    WarpSpecConfig config_;
    TiledQueue* tq_ = nullptr;
    CpuPollingThread* poller_ = nullptr;
    bool poller_started_ = false;

    std::chrono::high_resolution_clock::time_point launch_time_;

    int num_blocks_ = 0;
};

// ============================================================
// SG Warp Spec Session: scatter-gather mode session manager
//
// Unlike WarpSpecSession which uses fixed linear address mapping,
// SGWarpSpecSession manages an SGTaskQueue where compute warps
// or the host can dynamically submit (src, dst, size) entries.
// ============================================================

struct SGWarpSpecConfig {
    int num_compute_warps = 0;   // 0 = transfer-only (1 transfer warp + 0 compute)
    int num_blocks = 0;          // 0 = auto (use all SMs)
    int numa_node = 0;           // NUMA node for CPU poller
    bool use_copy_engine = true; // Use CE for DMA acceleration
    int max_lists = MAX_SG_LISTS;          // Max concurrent SG lists
    int max_entries = MAX_SG_POOL_ENTRIES;  // Max entries in pool

    // Block size: 1 transfer warp + N compute warps
    int block_size() const {
        return (1 + (num_compute_warps > 0 ? num_compute_warps : 1)) * 32;
    }
};

class SGWarpSpecSession {
public:
    explicit SGWarpSpecSession(const SGWarpSpecConfig& config);
    ~SGWarpSpecSession();

    // Non-copyable
    SGWarpSpecSession(const SGWarpSpecSession&) = delete;
    SGWarpSpecSession& operator=(const SGWarpSpecSession&) = delete;

    // ---- Host Submission API ----

    // Submit an SG list from the host side.
    // Writes entries + list header + memory barrier + sequence commit.
    // Returns the list slot index.
    uint64_t submit_sg_list(const DeviceSGEntry* entries, uint32_t count,
                            uint32_t list_id, uint32_t flags = SG_FLAG_HOST_SUBMITTED);

    // ---- Accessors ----
    SGTaskQueue* get_sg_queue() const { return sq_; }
    DescriptorQueue* get_desc_queue() const { return dq_; }
    CpuPollingThread* get_poller() const { return poller_; }

    // ---- Launch API ----

    // Launch an SG warp-spec kernel.
    // Kernel signature: (DescriptorQueue* dq, SGTaskQueue* sq, ComputeFn fn)
    template<typename KernelFn, typename ComputeFn>
    void launch(KernelFn kernel, ComputeFn compute_fn, cudaStream_t stream = 0);

    // ---- Lifecycle ----
    void synchronize();
    void reset();

    // ---- Diagnostics ----
    struct Stats {
        uint64_t descriptors_processed;
        uint64_t bytes_transferred;
        double elapsed_ms;
        double bandwidth_gbps;
    };
    Stats get_stats() const;

private:
    SGWarpSpecConfig config_;
    SGTaskQueue* sq_ = nullptr;
    DescriptorQueue* dq_ = nullptr;
    CpuPollingThread* poller_ = nullptr;
    bool poller_started_ = false;

    std::chrono::high_resolution_clock::time_point launch_time_;
    int num_blocks_ = 0;
};

}  // namespace gfd

// ---- Template implementation (requires NVCC / .cu compilation) ----
#ifdef __CUDACC__

template<typename KernelFn, typename ComputeFn>
void gfd::WarpSpecSession::launch(KernelFn kernel, ComputeFn compute_fn, cudaStream_t stream) {
    // Reset tile state for this launch
    reset();

    // Start poller AFTER kernel launch to avoid Blackwell coherence issues
    launch_time_ = std::chrono::high_resolution_clock::now();

    dim3 grid(num_blocks_);
    dim3 block(config_.block_size());
    kernel<<<grid, block, 0, stream>>>(tq_, config_.gpu_dst, config_.cpu_src, compute_fn);

    // Start CPU poller after kernel is in flight
    poller_->start();
    poller_started_ = true;
}

template<typename KernelFn, typename ComputeFn>
void gfd::SGWarpSpecSession::launch(KernelFn kernel, ComputeFn compute_fn, cudaStream_t stream) {
    // Partial reset: only reset descriptor queue and poller, NOT the SG queue.
    // SG lists may have been pre-submitted by the host before launch.
    if (poller_started_) {
        poller_->stop();
        poller_started_ = false;
    }
    sq_->terminate = 0;
    sq_->lists_completed = 0;
    if (sq_->d_list_done) {
        cudaMemset(sq_->d_list_done, 0, MAX_SG_LISTS * sizeof(uint64_t));
    }
    dq_->write_idx = 0;
    dq_->read_idx = 0;
    dq_->done_idx = 0;
    poller_->reset_stats();

    launch_time_ = std::chrono::high_resolution_clock::now();

    dim3 grid(num_blocks_);
    dim3 block(config_.block_size());
    kernel<<<grid, block, 0, stream>>>(dq_, sq_, compute_fn);

    // Start CPU poller after kernel is in flight
    poller_->start();
    poller_started_ = true;
}

#endif  // __CUDACC__
