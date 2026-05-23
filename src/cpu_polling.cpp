#include "gfd/cpu_polling.h"
#include "gfd/staging_pool.h"
#include "gfd/log.h"
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <chrono>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>

static long sys_mbind(void* addr, unsigned long len, int mode,
                      const unsigned long* nodemask, unsigned long maxnode,
                      unsigned flags) {
    return syscall(__NR_mbind, addr, len, mode, nodemask, maxnode, flags);
}
#endif

// Portable spin-wait hint
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
static inline void cpu_pause() { _mm_pause(); }
#elif defined(__aarch64__)
static inline void cpu_pause() { asm volatile("yield"); }
#else
#include <thread>
static inline void cpu_pause() { std::this_thread::yield(); }
#endif

namespace gfd {

CpuPollingThread::CpuPollingThread(DescriptorQueue* queue,
                                   void* gpu_base,
                                   void* cpu_base,
                                   size_t total_cpu_size,
                                   bool use_ce,
                                   int numa_node,
                                   int core_offset,
                                   int num_ce_channels,
                                   int exclusive_core_base,
                                   int exclusive_core_count)
    : queue_(queue)
    , gpu_base_(gpu_base)
    , cpu_base_(cpu_base)
    , total_cpu_size_(total_cpu_size)
    , use_ce_(use_ce)
    , numa_node_(numa_node)
    , core_offset_(core_offset)
    , num_ce_channels_(num_ce_channels)
    , exclusive_core_base_(exclusive_core_base)
    , exclusive_core_count_(exclusive_core_count) {}

CpuPollingThread::~CpuPollingThread() {
    stop();

    gather_workers_running_.store(false, std::memory_order_release);
    for (int i = 0; i < active_gather_workers_; i++) {
        if (gather_threads_[i].joinable())
            gather_threads_[i].join();
    }

    for (int b = 0; b < NUM_STAGING_BUFS; b++) {
        if (staging_events_[b]) {
            cuEventDestroy(staging_events_[b]);
            staging_events_[b] = nullptr;
        }
    }

    // Destroy tile event objects
    for (int i = 0; i < MAX_PENDING_TILE_EVENTS; i++) {
        if (pending_tile_events_[i].event) {
            cuEventDestroy(pending_tile_events_[i].event);
            pending_tile_events_[i].event = nullptr;
        }
    }

    // Destroy signal stream
    if (signal_stream_) {
        cuStreamSynchronize(signal_stream_);
        cuStreamDestroy(signal_stream_);
        signal_stream_ = nullptr;
    }

    // Free tile signal buffer
    if (tile_signal_buf_) {
        cudaFreeHost(tile_signal_buf_);
        tile_signal_buf_ = nullptr;
    }

    if (staging_pool_set_ >= 0) {
        StagingPool::instance().release_buffers(staging_pool_set_);
        staging_pool_set_ = -1;
        for (int b = 0; b < NUM_STAGING_BUFS; b++)
            staging_bufs_[b] = nullptr;
    } else {
        for (int b = 0; b < NUM_STAGING_BUFS; b++) {
            if (staging_bufs_[b]) {
                if (staging_hugepage_[b]) {
                    cudaHostUnregister(staging_bufs_[b]);
#ifdef __linux__
                    munmap(staging_bufs_[b], staging_hp_size_[b]);
#endif
                } else {
                    cudaFreeHost(staging_bufs_[b]);
                }
                staging_bufs_[b] = nullptr;
            }
        }
    }
}

void CpuPollingThread::pin_thread_to_cpu(std::thread& t, int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpuset), &cpuset);
    if (rc != 0) {
        GFD_LOG_ERROR("Failed to pin thread to CPU %d\n", cpu_id);
    }
#endif
}

void CpuPollingThread::pin_current_thread_to_cpu(int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (rc != 0) {
        GFD_LOG_ERROR("Failed to pin current thread to CPU %d\n", cpu_id);
    }
#endif
}

bool CpuPollingThread::init_copy_engine() {
    if (!use_ce_) {
        GFD_LOG_INFO("No-CE mode: direct memcpy to managed memory\n");
        return true;
    }

    CUcontext ctx;
    CUresult res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        GFD_LOG_ERROR("Failed to get CUDA context\n");
        return false;
    }

    res = ce_manager_.init(ctx, num_ce_channels_);
    if (res != CUDA_SUCCESS) {
        GFD_LOG_ERROR("Failed to initialize CE manager\n");
        return false;
    }

    staging_buffer_size_ = total_cpu_size_;
    int bufs_allocated = 0;

    auto& pool = StagingPool::instance();
    if (pool.is_initialized() && staging_buffer_size_ <= pool.buffer_size()) {
        int set_idx = pool.acquire_buffers(staging_buffer_size_, numa_node_);
        if (set_idx >= 0) {
            staging_pool_set_ = set_idx;
            char* pool_bufs[StagingPool::BUFS_PER_SET];
            size_t pool_size;
            bool pool_hp;
            pool.get_buffers(set_idx, pool_bufs, pool_size, pool_hp);
            for (int b = 0; b < NUM_STAGING_BUFS; b++) {
                staging_bufs_[b] = pool_bufs[b];
                staging_hugepage_[b] = pool_hp;
                staging_hp_size_[b] = 0;
            }
            bufs_allocated = NUM_STAGING_BUFS;
            GFD_LOG_INFO("Staging from pool: %d x %zu bytes (hugepage=%s, NUMA %d)\n",
                         bufs_allocated, staging_buffer_size_,
                         pool_hp ? "yes" : "no", numa_node_);
        }
    }

    if (bufs_allocated == 0) {
#ifdef __linux__
        size_t hp_size = (staging_buffer_size_ + (2*1024*1024 - 1)) & ~(size_t)(2*1024*1024 - 1);
        for (int b = 0; b < NUM_STAGING_BUFS; b++) {
            void* ptr = mmap(NULL, hp_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (ptr != MAP_FAILED) {
                unsigned long nodemask = 1UL << numa_node_;
                sys_mbind(ptr, hp_size, 2 /*MPOL_BIND*/, &nodemask,
                          sizeof(nodemask) * 8 + 1, 0);
                cudaError_t reg_err = cudaHostRegister(ptr, hp_size,
                                                        cudaHostRegisterDefault);
                if (reg_err == cudaSuccess) {
                    staging_bufs_[b] = (char*)ptr;
                    staging_hp_size_[b] = hp_size;
                    staging_hugepage_[b] = true;
                    bufs_allocated++;
                    continue;
                }
                munmap(ptr, hp_size);
            }
            cudaError_t cuda_err = cudaMallocHost(&staging_bufs_[b], staging_buffer_size_);
            if (cuda_err != cudaSuccess) {
                GFD_LOG_ERROR("Failed to allocate staging buffer %d\n", b);
                staging_bufs_[b] = nullptr;
                break;
            }
            madvise(staging_bufs_[b], staging_buffer_size_, MADV_HUGEPAGE);
            unsigned long nodemask = 1UL << numa_node_;
            sys_mbind(staging_bufs_[b], staging_buffer_size_, 2 /*MPOL_BIND*/,
                      &nodemask, sizeof(nodemask) * 8 + 1, 1 /*MPOL_MF_MOVE*/);
            staging_hugepage_[b] = false;
            bufs_allocated++;
        }
#else
        for (int b = 0; b < NUM_STAGING_BUFS; b++) {
            cudaError_t cuda_err = cudaMallocHost(&staging_bufs_[b], staging_buffer_size_);
            if (cuda_err != cudaSuccess) {
                GFD_LOG_ERROR("Failed to allocate staging buffer %d\n", b);
                staging_bufs_[b] = nullptr;
                break;
            }
            staging_hugepage_[b] = false;
            bufs_allocated++;
        }
#endif
        GFD_LOG_INFO("Staging buffers: %d x %zu bytes (hugepage=%s, NUMA %d)\n",
                     bufs_allocated, staging_buffer_size_,
                     (bufs_allocated > 0 && staging_hugepage_[0]) ? "yes" : "no",
                     numa_node_);
    }

    num_staging_bufs_ = bufs_allocated;

    for (int b = 0; b < bufs_allocated; b++) {
        CUresult ev_res = cuEventCreate(&staging_events_[b],
                                         CU_EVENT_DISABLE_TIMING);
        if (ev_res != CUDA_SUCCESS) {
            GFD_LOG_ERROR("Failed to create staging event %d\n", b);
        }
        staging_event_pending_[b] = false;
    }

    int base_cpu;
    int core_stride = 2;
    int available_cores;

    if (exclusive_core_base_ >= 0 && exclusive_core_count_ > 0) {
        base_cpu = exclusive_core_base_;
        available_cores = exclusive_core_count_;
        core_stride = (available_cores >= 16) ? 2 : 1;
    } else {
        base_cpu = numa_node_ * 64 + core_offset_ * 16;
        available_cores = 8;
        core_stride = 2;
    }

    int available_slots = available_cores / core_stride;
    int max_workers;
    if (available_slots >= 16) {
        max_workers = available_slots - 1;
    } else {
        max_workers = std::max(1, available_slots - 1);
    }
    int active_workers = std::min(MAX_GATHER_WORKERS, max_workers);

    auto get_worker_cpu = [base_cpu, core_stride](int worker_idx) -> int {
        return base_cpu + (worker_idx + 1) * core_stride;
    };

    active_gather_workers_ = active_workers;
    gather_workers_running_.store(true, std::memory_order_release);
    for (int i = 0; i < active_workers; i++) {
        gather_tasks_[i].has_work.store(false, std::memory_order_release);
        gather_tasks_[i].done.store(false, std::memory_order_release);
        gather_threads_[i] = std::thread(&CpuPollingThread::gather_worker_loop,
                                         this, i);
        int cpu_id = get_worker_cpu(i);
        pin_thread_to_cpu(gather_threads_[i], cpu_id);
    }
    GFD_LOG_INFO("Gather workers: %d, NUMA %d\n", active_workers, numa_node_);

    // Allocate pinned signal buffer for CE write-back tile signaling (P1).
    // Each tile slot needs an 8-byte value in pinned host memory that the CE
    // can read from when writing to tile_done[tile_id].
    if (!tile_signal_buf_) {
        cudaError_t err = cudaMallocHost(&tile_signal_buf_, MAX_TILES * sizeof(uint64_t));
        if (err != cudaSuccess) {
            GFD_LOG_ERROR("Failed to allocate tile signal buffer\n");
            tile_signal_buf_ = nullptr;
        } else {
            memset(tile_signal_buf_, 0, MAX_TILES * sizeof(uint64_t));
        }
    }

    // Create dedicated signal stream for device-memory signal writes.
    // Signal writes go through this stream AFTER make_stream_wait_on_all()
    // ensures all CE data channels have completed. This guarantees
    // data→signal ordering without blocking the CPU poller.
    if (!signal_stream_) {
        CUresult sig_res = cuStreamCreate(&signal_stream_, CU_STREAM_NON_BLOCKING);
        if (sig_res != CUDA_SUCCESS) {
            GFD_LOG_ERROR("Failed to create signal stream\n");
            signal_stream_ = nullptr;
        }
    }

    return true;
}

void CpuPollingThread::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&CpuPollingThread::polling_loop, this);
    int base_cpu;
    if (exclusive_core_base_ >= 0) {
        base_cpu = exclusive_core_base_;
    } else {
        base_cpu = numa_node_ * 64 + core_offset_ * 16;
    }
    pin_thread_to_cpu(thread_, base_cpu);
    if (!start_logged_) {
        GFD_LOG_INFO("Polling thread pinned to CPU %d\n", base_cpu);
        start_logged_ = true;
    }
}

void CpuPollingThread::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void CpuPollingThread::polling_loop() {
    if (use_ce_) {
        CUresult pin_res = ce_manager_.pin_context();
        if (pin_res != CUDA_SUCCESS) {
            GFD_LOG_ERROR("Failed to pin CUDA context\n");
            return;
        }
    }

    Descriptor batch[MAX_BATCH_SIZE];
    int batch_count = 0;
    int adaptive_threshold = BATCH_THRESHOLD;
    uint32_t or_flags = 0;

    while (running_.load(std::memory_order_acquire)) {
        uint64_t read = queue_->read_idx;

        Descriptor* entry = &queue_->entries[read % QUEUE_SIZE];
        uint64_t expected_seq = read + 1;
        uint64_t seq = __atomic_load_n(&entry->sequence, __ATOMIC_ACQUIRE);

        if (seq == expected_seq) {
            do {
                batch[batch_count] = *entry;
                or_flags |= entry->flags;
                __atomic_store_n(&entry->sequence, 0ULL, __ATOMIC_RELEASE);
                batch_count++;
                read++;
                queue_->read_idx = read;

                if (batch_count == 1) {
                    uint32_t esz = batch[0].size;
                    if (esz <= 2048) {
                        adaptive_threshold = std::min((int)BATCH_THRESHOLD_SMALL, MAX_BATCH_SIZE);
                    } else {
                        adaptive_threshold = BATCH_THRESHOLD;
                    }
                }

                if (batch_count >= adaptive_threshold) break;
                if (batch_count >= MAX_BATCH_SIZE) break;

                // In tiled/SG mode, flush per-chunk to enable early completion signaling.
                // Without this, the poller accumulates all entries up to BATCH_THRESHOLD
                // before flushing, which deadlocks kernels waiting on per-tile/list completion.
                if ((tiled_queue_ || sg_queue_) && (or_flags & FLAG_LAST_IN_TILE)) break;

                entry = &queue_->entries[read % QUEUE_SIZE];
                expected_seq = read + 1;
                seq = __atomic_load_n(&entry->sequence, __ATOMIC_ACQUIRE);
            } while (seq == expected_seq);

            bool is_urgent = (or_flags & (FLAG_LAST_IN_BATCH | FLAG_URGENT | FLAG_LAST_IN_TILE)) != 0;

            if (batch_count >= adaptive_threshold || is_urgent) {
                flush_batch(batch, batch_count, or_flags);
            }
        } else {
            if (batch_count > 0) {
                constexpr int GAP_SPIN_LIMIT = 4096;
                bool gap_filled = false;
                for (int spin = 0; spin < GAP_SPIN_LIMIT; spin++) {
                    cpu_pause();
                    seq = __atomic_load_n(&entry->sequence, __ATOMIC_ACQUIRE);
                    if (seq == expected_seq) {
                        gap_filled = true;
                        break;
                    }
                }
                if (!gap_filled) {
                    flush_batch(batch, batch_count, or_flags);
                }
            } else {
                // Tight spin-poll pending tile events for lowest latency.
                // cuEventQuery is ~60ns; we interleave entry checks to avoid
                // starving new descriptor processing.
                if (tile_events_inflight_ > 0) {
                    while (tile_events_inflight_ > 0) {
                        poll_tile_events();
                        if (tile_events_inflight_ == 0) break;
                        // Check for new entries between polls
                        seq = __atomic_load_n(&entry->sequence, __ATOMIC_ACQUIRE);
                        if (seq == expected_seq) break;
                        cpu_pause();
                    }
                } else if (has_async_dma_) {
                    if (staging_inflight_count_ > 0) {
                        ce_manager_.wait_completion();
                        staging_inflight_count_ = 0;
                    }
                    ce_manager_.wait_completion();
                    __atomic_store_n(&queue_->done_idx, latest_async_read_,
                                     __ATOMIC_RELEASE);
                    has_async_dma_ = false;
                } else {
                    cpu_pause();
                }
            }
        }
    }

    // Drain remaining committed entries
    for (;;) {
        uint64_t read = queue_->read_idx;
        Descriptor* entry = &queue_->entries[read % QUEUE_SIZE];
        uint64_t seq = __atomic_load_n(&entry->sequence, __ATOMIC_ACQUIRE);
        if (seq != read + 1) break;
        batch[batch_count] = *entry;
        __atomic_store_n(&entry->sequence, 0ULL, __ATOMIC_RELEASE);
        batch_count++;
        queue_->read_idx = read + 1;
        if (batch_count >= MAX_BATCH_SIZE) {
            flush_batch(batch, batch_count, or_flags);
        }
    }
    flush_batch(batch, batch_count, or_flags);

    // Final flush: drain all pending tile events (blocking wait)
    if (tile_events_inflight_ > 0) {
        ce_manager_.wait_completion();
        // Signal all remaining pending tile events with progress counts
        while (tile_events_inflight_ > 0) {
            int slot = tile_event_tail_;
            PendingTileEvent& pe = pending_tile_events_[slot];
            for (int i = 0; i < pe.num_tiles; i++) {
                uint32_t tid = pe.tile_ids[i];
                __atomic_store_n(&tiled_queue_->tile_chunk_done[tid],
                                 tile_progress_[tid], __ATOMIC_RELEASE);
            }
            uint64_t cur_done = __atomic_load_n(&queue_->done_idx, __ATOMIC_RELAXED);
            if (pe.done_value > cur_done) {
                __atomic_store_n(&queue_->done_idx, pe.done_value, __ATOMIC_RELEASE);
            }
            tile_event_tail_ = (slot + 1) % MAX_PENDING_TILE_EVENTS;
            tile_events_inflight_--;
        }
        has_async_dma_ = false;
    }

    if (staging_inflight_count_ > 0) {
        ce_manager_.wait_completion();
        staging_inflight_count_ = 0;
    }
    if (has_async_dma_) {
        ce_manager_.wait_completion();
        __atomic_store_n(&queue_->done_idx, latest_async_read_,
                         __ATOMIC_RELEASE);
        has_async_dma_ = false;
    }

    if (use_ce_) {
        ce_manager_.unpin_context();
    }
}

}  // namespace gfd
