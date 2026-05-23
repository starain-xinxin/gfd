#include "gfd/cpu_polling.h"
#include "gfd/log.h"
#include <cstring>
#include <chrono>

namespace gfd {

bool CpuPollingThread::init_direct_ce() {
    if (!use_ce_) return false;

    CUcontext ctx;
    CUresult res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        GFD_LOG_ERROR("Direct CE: failed to get CUDA context\n");
        return false;
    }

    res = direct_ce_manager_.init(ctx, num_ce_channels_);
    if (res != CUDA_SUCCESS) {
        GFD_LOG_ERROR("Direct CE: failed to initialize\n");
        return false;
    }

    GFD_LOG_INFO("Direct CE initialized (separate from poller)\n");
    return true;
}

double CpuPollingThread::submit_direct(const SGEntry* entries, int count) {
    if (!direct_ce_manager_.is_initialized() || count == 0)
        return -1.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    CUresult res = direct_ce_manager_.pin_context();
    if (res != CUDA_SUCCESS) return -1.0;

    size_t total_size = 0;
    for (int i = 0; i < count; i++)
        total_size += entries[i].size;

    bool dst_contiguous = true;
    if (count > 1) {
        for (int i = 1; i < count; i++) {
            if (entries[i].dst != entries[i-1].dst + entries[i-1].size) {
                dst_contiguous = false;
                break;
            }
        }
    }

    bool src_contiguous = true;
    if (count > 1) {
        for (int i = 1; i < count; i++) {
            if (entries[i].src != (const void*)((const char*)entries[i-1].src + entries[i-1].size)) {
                src_contiguous = false;
                break;
            }
        }
    }

    if (dst_contiguous && src_contiguous && count > 1) {
        // Fully contiguous: single DMA
        SGEntry single;
        single.dst = entries[0].dst;
        single.src = entries[0].src;
        single.size = total_size;
        direct_ce_manager_.submit_scatter_gather(&single, 1);
    } else if (dst_contiguous && !src_contiguous && num_staging_bufs_ > 0 &&
               total_size <= staging_buffer_size_) {
        CUdeviceptr min_dst = entries[0].dst;

        // Pipeline threshold: use multi-chunk pipeline for large transfers
        // to overlap gather of chunk N+1 with DMA of chunk N.
        // Requires at least 2 staging buffers.
        constexpr int PIPELINE_MIN_ENTRIES = 512;
        constexpr int PIPELINE_CHUNKS = 3;

        if (count >= PIPELINE_MIN_ENTRIES && num_staging_bufs_ >= 2 &&
            active_gather_workers_ > 0) {
            // Pipelined direct submit: split into chunks, overlap gather + DMA
            int num_chunks = std::min(PIPELINE_CHUNKS, num_staging_bufs_);
            int entries_per_chunk = (count + num_chunks - 1) / num_chunks;

            for (int c = 0; c < num_chunks; c++) {
                int chunk_start = c * entries_per_chunk;
                int chunk_end = std::min(chunk_start + entries_per_chunk, count);
                if (chunk_start >= count) break;

                int chunk_count = chunk_end - chunk_start;
                char* staging = staging_bufs_[c % num_staging_bufs_];
                CUdeviceptr chunk_dst = entries[chunk_start].dst;
                size_t chunk_size = 0;
                for (int i = chunk_start; i < chunk_end; i++)
                    chunk_size += entries[i].size;

                // Wait for previous DMA on this staging buffer (if pipelining wraps)
                if (c >= num_staging_bufs_ && staging_event_pending_[c % num_staging_bufs_]) {
                    cuEventSynchronize(staging_events_[c % num_staging_bufs_]);
                    staging_event_pending_[c % num_staging_bufs_] = false;
                }

                // Gather this chunk
                parallel_gather(entries + chunk_start, chunk_count, staging, chunk_dst);

                // Submit DMA for this chunk (non-blocking)
                SGEntry run;
                run.dst = chunk_dst;
                run.src = staging;
                run.size = chunk_size;
                direct_ce_manager_.submit_scatter_gather(&run, 1);

                // Record event for this staging buffer
                if (c + 1 < num_chunks) {
                    direct_ce_manager_.record_event_on_last_stream(
                        staging_events_[c % num_staging_bufs_]);
                    staging_event_pending_[c % num_staging_bufs_] = true;
                }
            }
        } else if (count >= 64 && active_gather_workers_ > 0) {
            // Non-pipelined parallel gather
            char* staging = staging_bufs_[0];
            parallel_gather(entries, count, staging, min_dst);

            SGEntry single;
            single.dst = min_dst;
            single.src = staging;
            single.size = total_size;
            direct_ce_manager_.submit_scatter_gather(&single, 1);
        } else {
            // Serial gather for small counts
            char* staging = staging_bufs_[0];
            for (int i = 0; i < count; i++) {
                size_t offset = entries[i].dst - min_dst;
                memcpy(staging + offset, entries[i].src, entries[i].size);
            }

            SGEntry single;
            single.dst = min_dst;
            single.src = staging;
            single.size = total_size;
            direct_ce_manager_.submit_scatter_gather(&single, 1);
        }
    } else {
        direct_ce_manager_.submit_scatter_gather(entries, count);
    }

    direct_ce_manager_.wait_completion();

    // Clear any pending staging events from pipeline
    for (int b = 0; b < num_staging_bufs_; b++) {
        staging_event_pending_[b] = false;
    }

    direct_ce_manager_.unpin_context();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    total_bytes_copied_.fetch_add(total_size, std::memory_order_relaxed);
    descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
    batches_submitted_.fetch_add(1, std::memory_order_relaxed);

    return elapsed_us;
}

}  // namespace gfd
