#pragma once

#include "gfd/tiled_queue.h"
#include "gfd/copy_engine.h"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <functional>
#include <cstring>

namespace gfd {

// Maximum gather worker threads (excluding main polling thread).
constexpr int MAX_GATHER_WORKERS = 15;

// Legacy default for backwards compatibility
constexpr int NUM_GATHER_WORKERS = 7;

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

    // Initialize (CE manager for CE mode, no-op for no-CE mode)
    bool init_copy_engine();

    // Direct-Submit fast path (Scheme A):
    // Bypass descriptor queue for small transfers.
    // Thread-safe: uses a separate CE manager from the polling thread.
    // Returns latency in microseconds, or -1 on error.
    double submit_direct(const SGEntry* entries, int count);

    // Initialize direct-submit CE (call once after init_copy_engine)
    bool init_direct_ce();

    // Set optional tiled queue for per-tile completion signaling.
    // When set, flush_batch will write tile_done[tile_id] for descriptors
    // with FLAG_LAST_IN_TILE. Must be called before start().
    void set_tiled_queue(TiledQueue* tq) { tiled_queue_ = tq; }

    // Set optional SG task queue for per-list completion signaling.
    // When set, descriptors with list_id in user_data upper bits are
    // grouped by list_id and signaled via d_list_done[list_id].
    void set_sg_task_queue(struct SGTaskQueue* sq) { sg_queue_ = sq; }

    // Start/stop the polling thread
    void start();
    void stop();

    // Statistics (thread-safe: written by polling thread, read from any thread)
    uint64_t get_descriptors_processed() const { return descriptors_processed_.load(std::memory_order_relaxed); }
    uint64_t get_batches_submitted() const { return batches_submitted_.load(std::memory_order_relaxed); }
    uint64_t get_coalesced_entries() const { return coalesced_entries_.load(std::memory_order_relaxed); }
    uint64_t get_staging_batches() const { return staging_batches_.load(std::memory_order_relaxed); }
    double get_total_bytes_copied() const { return total_bytes_copied_.load(std::memory_order_relaxed); }
    const CopyEngineManager& get_ce_manager() const { return ce_manager_; }
    bool is_ce_mode() const { return use_ce_; }
    uint64_t get_gather_us() const { return total_gather_us_.load(std::memory_order_relaxed); }
    uint64_t get_dma_wait_us() const { return total_dma_wait_us_.load(std::memory_order_relaxed); }
    uint64_t get_dma_submit_us() const { return total_dma_submit_us_.load(std::memory_order_relaxed); }
    uint64_t get_queue_read_us() const { return total_queue_read_us_.load(std::memory_order_relaxed); }

    // Reset per-test timing stats (call between test runs)
    void reset_stats() {
        descriptors_processed_.store(0, std::memory_order_relaxed);
        batches_submitted_.store(0, std::memory_order_relaxed);
        coalesced_entries_.store(0, std::memory_order_relaxed);
        staging_batches_.store(0, std::memory_order_relaxed);
        total_bytes_copied_.store(0, std::memory_order_relaxed);
        total_gather_us_.store(0, std::memory_order_relaxed);
        total_dma_wait_us_.store(0, std::memory_order_relaxed);
        total_dma_submit_us_.store(0, std::memory_order_relaxed);
        total_queue_read_us_.store(0, std::memory_order_relaxed);
        memset(tile_progress_, 0, sizeof(tile_progress_));
    }

private:
    void polling_loop();

    // Flush current batch and update done_idx / async state
    void flush_batch(Descriptor* batch, int& batch_count, uint32_t& or_flags);

    bool process_batch(Descriptor* batch, int count);
    void process_batch_no_ce(Descriptor* batch, int count);

    void parallel_gather(const SGEntry* entries, int count,
                         char* staging, CUdeviceptr min_dst);
    void gather_worker_loop(int worker_id);

    static void pin_thread_to_cpu(std::thread& t, int cpu_id);
    static void pin_current_thread_to_cpu(int cpu_id);

    DescriptorQueue* queue_;
    TiledQueue* tiled_queue_ = nullptr;  // Optional: per-tile completion signaling
    struct SGTaskQueue* sg_queue_ = nullptr;  // Optional: SG per-list completion signaling
    void* gpu_base_;
    void* cpu_base_;
    size_t total_cpu_size_;
    bool use_ce_;
    int numa_node_;
    int core_offset_;
    int num_ce_channels_;
    int exclusive_core_base_;
    int exclusive_core_count_;

    CopyEngineManager ce_manager_;
    CopyEngineManager direct_ce_manager_;

    // Dedicated stream for writing device-memory signals after CE data DMAs complete.
    // Used with ce_manager_.make_stream_wait_on_all() to ensure GPU-side ordering
    // without CPU-blocking. Only created when tiled_queue_->d_tile_chunk_done is set.
    CUstream signal_stream_ = nullptr;

    static constexpr int NUM_STAGING_BUFS = 5;
    char* staging_bufs_[NUM_STAGING_BUFS] = {};
    size_t staging_buffer_size_ = 0;
    size_t staging_hp_size_[NUM_STAGING_BUFS] = {};
    bool staging_hugepage_[NUM_STAGING_BUFS] = {};
    CUevent staging_events_[NUM_STAGING_BUFS] = {};
    bool staging_event_pending_[NUM_STAGING_BUFS] = {};
    int num_staging_bufs_ = 0;  // Actual number of successfully allocated buffers
    int staging_cur_ = 0;
    int staging_inflight_count_ = 0;

    bool has_async_dma_ = false;
    uint64_t latest_async_read_ = 0;

    // Per-tile DMA splitting + CE write-back signaling:
    // When tiled mode is active, entries are grouped by tile_id and submitted
    // as separate per-tile DMAs. After each tile's DMA, a CE write-back appends
    // an 8-byte write to tile_done[tile_id], providing zero-latency signaling
    // without cuEventQuery overhead. Adaptive: if a batch covers >= 50% of
    // tiles, the global done_idx path is used instead (single coalesced DMA).
    static constexpr int MAX_PENDING_TILE_EVENTS = 32;
    struct PendingTileEvent {
        CUevent event = nullptr;
        uint64_t done_value = 0;
        uint32_t tile_ids[MAX_TILES];
        int num_tiles = 0;
    };
    PendingTileEvent pending_tile_events_[MAX_PENDING_TILE_EVENTS];
    int tile_event_head_ = 0;
    int tile_event_tail_ = 0;
    int tile_events_inflight_ = 0;

    void poll_tile_events();  // Non-blocking check of pending tile events

    // Pinned signal values for CE write-back (one per tile slot)
    uint64_t* tile_signal_buf_ = nullptr;  // cudaMallocHost'd buffer

    // Per-tile progress counters: tracks cumulative tokens completed per tile.
    // Used when tiles' descriptors are interleaved in the queue (atomicAdd mode).
    // tile_done[tile_id] is written with this value via CE write-back.
    uint64_t tile_progress_[MAX_TILES] = {};

    // Per-tile DMA processing
    void process_batch_tiled(Descriptor* batch, int count, uint64_t cur_read);

    // SG mode: per-list DMA processing with d_list_done signaling
    void process_batch_sg(Descriptor* batch, int count, uint64_t cur_read);

    struct alignas(128) GatherTask {
        alignas(64) std::atomic<bool> has_work{false};
        alignas(64) std::atomic<bool> done{false};
        const SGEntry* entries;
        char* staging;
        CUdeviceptr min_dst;
        int start;
        int end;
    };

    GatherTask gather_tasks_[MAX_GATHER_WORKERS];
    std::thread gather_threads_[MAX_GATHER_WORKERS];
    std::atomic<bool> gather_workers_running_{false};
    int active_gather_workers_ = NUM_GATHER_WORKERS;

    int staging_pool_set_ = -1;

    std::atomic<bool> running_{false};
    bool start_logged_ = false;
    std::thread thread_;

    std::atomic<uint64_t> descriptors_processed_{0};
    std::atomic<uint64_t> batches_submitted_{0};
    std::atomic<uint64_t> coalesced_entries_{0};
    std::atomic<uint64_t> staging_batches_{0};
    std::atomic<uint64_t> total_bytes_copied_{0};

    std::atomic<uint64_t> total_gather_us_{0};
    std::atomic<uint64_t> total_dma_wait_us_{0};
    std::atomic<uint64_t> total_dma_submit_us_{0};
    std::atomic<uint64_t> total_queue_read_us_{0};
};

}  // namespace gfd
