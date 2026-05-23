#pragma once

#include <cuda.h>
#include <cstdint>
#include <cstdio>

namespace gfd {

// Error-checking macro for CUDA Driver API
#define GFD_CU_CHECK(call)                                                     \
    do {                                                                        \
        CUresult err = (call);                                                  \
        if (err != CUDA_SUCCESS) {                                              \
            const char* err_str = nullptr;                                      \
            cuGetErrorString(err, &err_str);                                    \
            fprintf(stderr, "CUDA Driver error at %s:%d: %s (%d)\n",           \
                    __FILE__, __LINE__, err_str ? err_str : "unknown", err);    \
            return err;                                                         \
        }                                                                       \
    } while (0)

// Max CE channels for pipelining.
constexpr int MAX_CE_CHANNELS = 3;

// Max scatter-gather entries per single batch submission.
constexpr int MAX_SG_ENTRIES_PER_BATCH = 8192;

// A single scatter-gather entry: one (src, dst, size) triple
struct SGEntry {
    CUdeviceptr dst;    // GPU destination (device pointer)
    const void* src;    // CPU source (host pointer)
    size_t size;        // Bytes to transfer
};

// CopyEngineManager: CUDA Driver API scatter-gather DMA
//
// Uses cuMemcpyHtoDAsync on dedicated CE streams to simulate
// hardware scatter-gather. Multiple CE channels are pipelined
// so that CE programming overlaps with DMA transfer.
class CopyEngineManager {
public:
    CopyEngineManager();
    ~CopyEngineManager();

    // Initialize: create dedicated CE streams with high priority
    // Must be called after CUDA context is established
    // num_channels: override default channel count (0 = use MAX_CE_CHANNELS)
    CUresult init(CUcontext ctx, int num_channels = 0);

    // Shutdown: destroy streams and events
    void shutdown();

    // Pin/unpin the CUDA context on the calling thread.
    // When pinned, submit/wait skip cuCtxPushCurrent/cuCtxPopCurrent,
    // saving ~2-4 us per batch.
    CUresult pin_context();
    void     unpin_context();

    // Submit a batch of scatter-gather entries.
    // All entries are enqueued as async H2D copies across CE channels.
    // Returns after all copies are submitted (not completed).
    CUresult submit_scatter_gather(const SGEntry* entries, int count);

    // Wait for all in-flight CE transfers to complete.
    CUresult wait_completion();

    // Submit and wait in one call (convenience).
    CUresult submit_and_wait(const SGEntry* entries, int count);

    // Record an event on the stream used by the last submission.
    CUresult record_event_on_last_stream(CUevent event);

    // Make a target stream wait on all dirty CE channels.
    // Records events on all channels with pending work, then inserts
    // cuStreamWaitEvent dependencies so `stream` won't execute until
    // all pending CE DMAs complete. GPU-side only — does NOT block CPU.
    CUresult make_stream_wait_on_all(CUstream stream);

    // Statistics
    uint64_t get_total_submissions() const { return total_submissions_; }
    uint64_t get_total_entries() const { return total_entries_; }
    double get_total_bytes() const { return total_bytes_; }

    bool is_initialized() const { return initialized_; }

private:
    CUcontext ctx_ = nullptr;
    CUstream ce_streams_[MAX_CE_CHANNELS] = {};
    CUevent ce_events_[MAX_CE_CHANNELS] = {};
    int num_channels_ = MAX_CE_CHANNELS;
    bool initialized_ = false;
    bool context_pinned_ = false;

    int next_channel_ = 0;
    uint32_t dirty_mask_ = 0;  // Bitmask of channels used since last wait

    uint64_t total_submissions_ = 0;
    uint64_t total_entries_ = 0;
    double total_bytes_ = 0;
};

}  // namespace gfd
