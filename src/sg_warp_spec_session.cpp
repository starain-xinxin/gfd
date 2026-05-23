#include "gfd/warp_spec_session.h"
#include "gfd/log.h"
#include <cuda.h>
#include <cstring>
#include <ctime>

namespace gfd {

SGWarpSpecSession::SGWarpSpecSession(const SGWarpSpecConfig& config)
    : config_(config)
{
    // Determine block count
    if (config_.num_blocks > 0) {
        num_blocks_ = config_.num_blocks;
    } else {
        cudaDeviceGetAttribute(&num_blocks_, cudaDevAttrMultiProcessorCount, 0);
    }

    // Allocate SGTaskQueue (host-mapped for CPU+GPU shared access)
    cudaHostAlloc(&sq_, sizeof(SGTaskQueue), cudaHostAllocMapped);
    memset(sq_, 0, sizeof(SGTaskQueue));

    // Allocate DescriptorQueue (host-mapped)
    cudaHostAlloc(&dq_, sizeof(DescriptorQueue), cudaHostAllocMapped);
    memset(dq_, 0, sizeof(DescriptorQueue));

    // Allocate device-side per-list completion signal buffer
    uint64_t* d_list_done = nullptr;
    cudaError_t alloc_err = cudaMalloc(&d_list_done, MAX_SG_LISTS * sizeof(uint64_t));
    if (alloc_err == cudaSuccess) {
        cudaMemset(d_list_done, 0, MAX_SG_LISTS * sizeof(uint64_t));
        sq_->d_list_done = d_list_done;
    } else {
        sq_->d_list_done = nullptr;
        GFD_LOG_ERROR("SGWarpSpecSession: failed to allocate d_list_done\n");
    }

    // Initialize staging pool (use a reasonable size for SG mode)
    // SG entries can point to arbitrary addresses, so staging size is estimated
    size_t staging_size = (size_t)MAX_SG_POOL_ENTRIES * 4096;  // ~64MB default
    StagingPool::instance().init(1, staging_size);

    // Create CPU poller
    // For SG mode, cpu_base and gpu_base are not used (addresses in entries),
    // but the poller needs them for initialization. Use nullptr-safe values.
    poller_ = new CpuPollingThread(
        dq_,
        nullptr,   // gpu_base: not used in SG mode
        nullptr,   // cpu_base: not used in SG mode
        staging_size,
        config_.use_copy_engine,
        config_.numa_node);

    // Set SG queue for per-list completion signaling
    poller_->set_sg_task_queue(sq_);

    if (config_.use_copy_engine) {
        if (!poller_->init_copy_engine()) {
            GFD_LOG_ERROR("SGWarpSpecSession: failed to init copy engine\n");
        }
    }
}

SGWarpSpecSession::~SGWarpSpecSession() {
    if (poller_) {
        poller_->stop();
        delete poller_;
    }
    if (sq_) {
        if (sq_->d_list_done) {
            cudaFree(sq_->d_list_done);
            sq_->d_list_done = nullptr;
        }
        cudaFreeHost(sq_);
    }
    if (dq_) {
        cudaFreeHost(dq_);
    }
}

uint64_t SGWarpSpecSession::submit_sg_list(
    const DeviceSGEntry* entries, uint32_t count,
    uint32_t list_id, uint32_t flags)
{
    // Allocate entry pool slots
    uint64_t pool_offset = sq_->entry_alloc_idx;
    sq_->entry_alloc_idx = pool_offset + count;

    // Write entries into pool
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slot = (uint32_t)((pool_offset + i) % MAX_SG_POOL_ENTRIES);
        sq_->entries[slot] = entries[i];
    }

    // Allocate list slot
    uint64_t list_slot = sq_->list_alloc_idx;
    sq_->list_alloc_idx = list_slot + 1;
    uint32_t ring_slot = (uint32_t)(list_slot % MAX_SG_LISTS);

    // Write list header
    sq_->lists[ring_slot].pool_offset = (uint32_t)(pool_offset % MAX_SG_POOL_ENTRIES);
    sq_->lists[ring_slot].count = count;
    sq_->lists[ring_slot].list_id = list_id;
    sq_->lists[ring_slot].flags = flags;

    // Memory barrier before sequence commit
    __atomic_thread_fence(__ATOMIC_RELEASE);

    // Commit: write sequence marker
    sq_->lists[ring_slot].sequence = list_slot + 1;

    // Memory barrier after sequence commit
    __atomic_thread_fence(__ATOMIC_RELEASE);

    return list_slot;
}

void SGWarpSpecSession::synchronize() {
    // Wait for all submitted lists to be fully DMA'd before terminating.
    // With multi-block, list_read_idx is atomically advanced (= "claimed"),
    // so we use lists_completed (set by CPU poller after DMA) instead.
    uint64_t expected = __atomic_load_n(&sq_->list_alloc_idx, __ATOMIC_ACQUIRE);
    while (true) {
        uint64_t completed = __atomic_load_n(&sq_->lists_completed, __ATOMIC_ACQUIRE);
        if (completed >= expected) break;
        struct timespec ts = {0, 100000};  // 100us
        nanosleep(&ts, nullptr);
    }

    // Now safe to set terminate — all lists processed
    sq_->terminate = 1;
    __atomic_thread_fence(__ATOMIC_RELEASE);

    cudaDeviceSynchronize();
    if (poller_started_) {
        poller_->stop();
        poller_started_ = false;
    }
}

void SGWarpSpecSession::reset() {
    // Stop poller if running
    if (poller_started_) {
        poller_->stop();
        poller_started_ = false;
    }

    // Reset SG queue state
    sq_->list_alloc_idx = 0;
    sq_->list_read_idx = 0;
    sq_->entry_alloc_idx = 0;
    sq_->entry_consumed_idx = 0;
    sq_->lists_completed = 0;
    sq_->terminate = 0;

    // Clear list sequences
    for (int i = 0; i < MAX_SG_LISTS; i++) {
        sq_->lists[i].sequence = 0;
    }

    // Reset d_list_done signals
    if (sq_->d_list_done) {
        cudaMemset(sq_->d_list_done, 0, MAX_SG_LISTS * sizeof(uint64_t));
    }

    // Reset descriptor queue
    dq_->write_idx = 0;
    dq_->read_idx = 0;
    dq_->done_idx = 0;

    // Reset poller stats
    poller_->reset_stats();
}

SGWarpSpecSession::Stats SGWarpSpecSession::get_stats() const {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - launch_time_).count();

    Stats stats;
    stats.descriptors_processed = poller_->get_descriptors_processed();
    stats.bytes_transferred = (uint64_t)poller_->get_total_bytes_copied();
    stats.elapsed_ms = elapsed_ms;
    stats.bandwidth_gbps = (stats.bytes_transferred > 0)
        ? stats.bytes_transferred / (elapsed_ms * 1e6) : 0.0;
    return stats;
}

}  // namespace gfd
