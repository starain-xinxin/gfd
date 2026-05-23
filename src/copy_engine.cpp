#include "gfd/copy_engine.h"
#include "gfd/log.h"
#include <cstring>

namespace gfd {

CopyEngineManager::CopyEngineManager() = default;

CopyEngineManager::~CopyEngineManager() {
    shutdown();
}

CUresult CopyEngineManager::init(CUcontext ctx, int num_channels) {
    if (initialized_) return CUDA_SUCCESS;

    ctx_ = ctx;
    if (num_channels > 0 && num_channels <= MAX_CE_CHANNELS) {
        num_channels_ = num_channels;
    }

    GFD_CU_CHECK(cuCtxPushCurrent(ctx_));

    int priority_low = 0, priority_high = 0;
    GFD_CU_CHECK(cuCtxGetStreamPriorityRange(&priority_low, &priority_high));

    for (int i = 0; i < num_channels_; i++) {
        GFD_CU_CHECK(cuStreamCreateWithPriority(
            &ce_streams_[i],
            CU_STREAM_NON_BLOCKING,
            priority_high));

        GFD_CU_CHECK(cuEventCreate(
            &ce_events_[i],
            CU_EVENT_DISABLE_TIMING));
    }

    CUcontext popped;
    GFD_CU_CHECK(cuCtxPopCurrent(&popped));

    initialized_ = true;

    GFD_LOG_INFO("CE Manager: %d channels, priority=%d\n",
                 num_channels_, priority_high);

    return CUDA_SUCCESS;
}

void CopyEngineManager::shutdown() {
    if (!initialized_) return;

    if (!context_pinned_) {
        cuCtxPushCurrent(ctx_);
    }

    for (int i = 0; i < num_channels_; i++) {
        if (ce_streams_[i]) {
            cuStreamSynchronize(ce_streams_[i]);
            cuStreamDestroy(ce_streams_[i]);
            ce_streams_[i] = nullptr;
        }
        if (ce_events_[i]) {
            cuEventDestroy(ce_events_[i]);
            ce_events_[i] = nullptr;
        }
    }

    if (!context_pinned_) {
        CUcontext popped;
        cuCtxPopCurrent(&popped);
    }

    initialized_ = false;
}

CUresult CopyEngineManager::pin_context() {
    if (context_pinned_) return CUDA_SUCCESS;
    GFD_CU_CHECK(cuCtxPushCurrent(ctx_));
    context_pinned_ = true;
    return CUDA_SUCCESS;
}

void CopyEngineManager::unpin_context() {
    if (!context_pinned_) return;
    CUcontext popped;
    cuCtxPopCurrent(&popped);
    context_pinned_ = false;
}

CUresult CopyEngineManager::submit_scatter_gather(
    const SGEntry* entries, int count)
{
    if (!initialized_ || count == 0) return CUDA_SUCCESS;

    if (!context_pinned_) {
        GFD_CU_CHECK(cuCtxPushCurrent(ctx_));
    }

    // Track channels used in THIS submission only (not cumulative dirty_mask_)
    uint32_t this_mask = 0;

    // Chunk-based stream assignment: each stream gets a contiguous range
    // of entries for better hardware DMA coalescing potential
    if (count <= num_channels_) {
        // Few entries: one per stream (original behavior)
        for (int i = 0; i < count; i++) {
            int ch = next_channel_;
            next_channel_ = (next_channel_ + 1) % num_channels_;
            this_mask |= (1u << ch);
            GFD_CU_CHECK(cuMemcpyHtoDAsync(
                entries[i].dst, entries[i].src, entries[i].size,
                ce_streams_[ch]));
            total_bytes_ += entries[i].size;
        }
    } else {
        // Chunk-based: contiguous ranges per stream
        int chunk_size = (count + num_channels_ - 1) / num_channels_;
        for (int ch = 0; ch < num_channels_; ch++) {
            int start = ch * chunk_size;
            int end = start + chunk_size;
            if (end > count) end = count;
            if (start >= count) break;
            this_mask |= (1u << ch);
            for (int i = start; i < end; i++) {
                GFD_CU_CHECK(cuMemcpyHtoDAsync(
                    entries[i].dst, entries[i].src, entries[i].size,
                    ce_streams_[ch]));
                total_bytes_ += entries[i].size;
            }
        }
        next_channel_ = 0;  // Reset for next submission
    }

    // Update cumulative dirty mask (for wait_completion)
    dirty_mask_ |= this_mask;

    // Record events ONLY on channels used in THIS submission
    uint32_t mask = this_mask;
    while (mask) {
        int ch = __builtin_ctz(mask);
        GFD_CU_CHECK(cuEventRecord(ce_events_[ch], ce_streams_[ch]));
        mask &= mask - 1;
    }

    if (!context_pinned_) {
        CUcontext popped;
        GFD_CU_CHECK(cuCtxPopCurrent(&popped));
    }

    total_submissions_++;
    total_entries_ += count;

    return CUDA_SUCCESS;
}

CUresult CopyEngineManager::wait_completion() {
    if (!initialized_ || dirty_mask_ == 0) return CUDA_SUCCESS;

    if (!context_pinned_) {
        GFD_CU_CHECK(cuCtxPushCurrent(ctx_));
    }

    // Only synchronize channels that have pending work
    uint32_t mask = dirty_mask_;
    while (mask) {
        int ch = __builtin_ctz(mask);
        GFD_CU_CHECK(cuEventSynchronize(ce_events_[ch]));
        mask &= mask - 1;
    }
    dirty_mask_ = 0;

    if (!context_pinned_) {
        CUcontext popped;
        GFD_CU_CHECK(cuCtxPopCurrent(&popped));
    }

    return CUDA_SUCCESS;
}

CUresult CopyEngineManager::record_event_on_last_stream(CUevent event) {
    if (!initialized_) return CUDA_SUCCESS;
    int last_ch = (next_channel_ + num_channels_ - 1) % num_channels_;
    GFD_CU_CHECK(cuEventRecord(event, ce_streams_[last_ch]));
    return CUDA_SUCCESS;
}

CUresult CopyEngineManager::make_stream_wait_on_all(CUstream stream) {
    if (!initialized_ || dirty_mask_ == 0) return CUDA_SUCCESS;

    if (!context_pinned_) {
        GFD_CU_CHECK(cuCtxPushCurrent(ctx_));
    }

    // Record fresh events on all dirty channels, then make target stream wait
    uint32_t mask = dirty_mask_;
    while (mask) {
        int ch = __builtin_ctz(mask);
        GFD_CU_CHECK(cuEventRecord(ce_events_[ch], ce_streams_[ch]));
        GFD_CU_CHECK(cuStreamWaitEvent(stream, ce_events_[ch], 0));
        mask &= mask - 1;
    }

    if (!context_pinned_) {
        CUcontext popped;
        GFD_CU_CHECK(cuCtxPopCurrent(&popped));
    }

    return CUDA_SUCCESS;
}

CUresult CopyEngineManager::submit_and_wait(
    const SGEntry* entries, int count)
{
    CUresult res = submit_scatter_gather(entries, count);
    if (res != CUDA_SUCCESS) return res;
    return wait_completion();
}

}  // namespace gfd
