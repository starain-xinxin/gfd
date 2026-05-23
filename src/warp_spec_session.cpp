#include "gfd/warp_spec_session.h"
#include "gfd/log.h"
#include <cuda.h>
#include <cstring>

namespace gfd {

WarpSpecSession::WarpSpecSession(const WarpSpecConfig& config)
    : config_(config)
{
    // Validate config
    if (config_.total_tokens == 0 || config_.token_size == 0) {
        GFD_LOG_ERROR("WarpSpecSession: total_tokens and token_size must be > 0\n");
        return;
    }
    if (config_.tokens_per_tile == 0 || config_.effective_tokens_per_chunk() == 0) {
        GFD_LOG_ERROR("WarpSpecSession: tokens_per_tile and tokens_per_chunk must be > 0\n");
        return;
    }
    if (config_.tokens_per_tile % config_.effective_tokens_per_chunk() != 0) {
        GFD_LOG_ERROR("WarpSpecSession: tokens_per_tile must be divisible by tokens_per_chunk\n");
        return;
    }

    // Determine block count
    if (config_.num_blocks > 0) {
        num_blocks_ = config_.num_blocks;
    } else {
        cudaDeviceGetAttribute(&num_blocks_, cudaDevAttrMultiProcessorCount, 0);
    }

    // Allocate TiledQueue (host-mapped for CPU+GPU shared access)
    cudaHostAlloc(&tq_, sizeof(TiledQueue), cudaHostAllocMapped);
    memset(tq_, 0, sizeof(TiledQueue));

    // Configure scheduler (use effective_tokens_per_chunk for auto-tuning)
    tq_->scheduler.total_tiles = config_.total_tiles();
    tq_->scheduler.tokens_per_tile = config_.tokens_per_tile;
    tq_->scheduler.tokens_per_chunk = config_.effective_tokens_per_chunk();
    tq_->scheduler.chunks_per_tile = config_.chunks_per_tile();
    tq_->scheduler.token_size = config_.token_size;
    tq_->scheduler.next_tile = 0;

    // Allocate device-side signal buffer for fast GPU polling (L2 cache ~10ns vs PCIe ~1500ns)
    uint64_t* d_signal = nullptr;
    cudaError_t alloc_err = cudaMalloc(&d_signal, MAX_TILES * sizeof(uint64_t));
    if (alloc_err == cudaSuccess) {
        cudaMemset(d_signal, 0, MAX_TILES * sizeof(uint64_t));
        tq_->d_tile_chunk_done = d_signal;
    } else {
        tq_->d_tile_chunk_done = nullptr;  // fallback to host-mapped polling
    }

    // Initialize staging pool
    size_t total_size = (size_t)config_.total_tokens * config_.token_size;
    StagingPool::instance().init(1, total_size);

    // Create CPU poller
    poller_ = new CpuPollingThread(
        &tq_->base,
        config_.gpu_dst,
        config_.cpu_src,
        total_size,
        config_.use_copy_engine,
        config_.numa_node);

    poller_->set_tiled_queue(tq_);

    if (config_.use_copy_engine) {
        if (!poller_->init_copy_engine()) {
            GFD_LOG_ERROR("WarpSpecSession: failed to init copy engine\n");
        }
    }
}

WarpSpecSession::~WarpSpecSession() {
    if (poller_) {
        poller_->stop();
        delete poller_;
    }
    if (tq_) {
        if (tq_->d_tile_chunk_done) {
            cudaFree(tq_->d_tile_chunk_done);
            tq_->d_tile_chunk_done = nullptr;
        }
        cudaFreeHost(tq_);
    }
    StagingPool::instance().shutdown();
}

WarpSpecSession::LaunchParams WarpSpecSession::get_launch_params() const {
    LaunchParams params;
    params.tq = tq_;
    params.gpu_buffer = config_.gpu_dst;
    params.cpu_base = config_.cpu_src;
    params.grid = dim3(num_blocks_);
    params.block = dim3(config_.block_size());
    return params;
}

void WarpSpecSession::synchronize() {
    cudaDeviceSynchronize();
    if (poller_started_) {
        poller_->stop();
        poller_started_ = false;
    }
}

void WarpSpecSession::reset() {
    // Stop poller if running
    if (poller_started_) {
        poller_->stop();
        poller_started_ = false;
    }

    // Reset tile completion signals
    if (tq_->d_tile_chunk_done) {
        cudaMemset(tq_->d_tile_chunk_done, 0, MAX_TILES * sizeof(uint64_t));
    }
    memset((void*)tq_->tile_chunk_done, 0, sizeof(tq_->tile_chunk_done));

    // Reset scheduler
    tq_->scheduler.next_tile = 0;

    // Reset poller stats (includes tile_progress)
    poller_->reset_stats();

    // Clear GPU destination buffer is caller's responsibility (optional)
}

WarpSpecSession::Stats WarpSpecSession::get_stats() const {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - launch_time_).count();
    size_t total_bytes = (size_t)config_.total_tokens * config_.token_size;

    Stats stats;
    stats.descriptors_processed = poller_->get_descriptors_processed();
    stats.bytes_transferred = (uint64_t)poller_->get_total_bytes_copied();
    stats.elapsed_ms = elapsed_ms;
    stats.bandwidth_gbps = total_bytes / (elapsed_ms * 1e6);
    return stats;
}

}  // namespace gfd
