#include "gfd/cpu_polling.h"
#include "gfd/log.h"
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
static inline void cpu_pause_g() { _mm_pause(); }

static void streaming_memcpy_g(void* __restrict dst, const void* __restrict src, size_t size) {
#ifdef __AVX512F__
    const char* s = (const char*)src;
    char* d = (char*)dst;
    size_t aligned = size & ~(size_t)63;

    for (size_t i = 0; i < aligned; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(s + i));
        _mm512_stream_si512((__m512i*)(d + i), v);
    }
    if (aligned < size) {
        memcpy(d + aligned, s + aligned, size - aligned);
    }
    _mm_sfence();
#else
    memcpy(dst, src, size);
#endif
}

#elif defined(__aarch64__)
static inline void cpu_pause_g() { asm volatile("yield"); }
static void streaming_memcpy_g(void* __restrict dst, const void* __restrict src, size_t size) {
    memcpy(dst, src, size);
}
#else
#include <thread>
static inline void cpu_pause_g() { std::this_thread::yield(); }
static void streaming_memcpy_g(void* __restrict dst, const void* __restrict src, size_t size) {
    memcpy(dst, src, size);
}
#endif

namespace gfd {

void CpuPollingThread::gather_worker_loop(int worker_id) {
    GatherTask& task = gather_tasks_[worker_id];
    while (gather_workers_running_.load(std::memory_order_acquire)) {
        if (task.has_work.load(std::memory_order_acquire)) {
            constexpr int PREFETCH_DIST = 16;

            for (int p = 0; p < PREFETCH_DIST && task.start + p < task.end; p++) {
                __builtin_prefetch(task.entries[task.start + p].src, 0, 0);
                if (task.start + p + 1 < task.end)
                    __builtin_prefetch(&task.entries[task.start + p + 1], 0, 3);
            }

            for (int i = task.start; i < task.end; i++) {
                if (i + PREFETCH_DIST < task.end) {
                    __builtin_prefetch(task.entries[i + PREFETCH_DIST].src, 0, 0);
                    __builtin_prefetch(&task.entries[i + PREFETCH_DIST], 0, 3);
                }
                size_t offset = task.entries[i].dst - task.min_dst;
                streaming_memcpy_g(task.staging + offset,
                                   task.entries[i].src,
                                   task.entries[i].size);
            }
            task.done.store(true, std::memory_order_release);
            task.has_work.store(false, std::memory_order_release);
        } else {
            cpu_pause_g();
        }
    }
}

void CpuPollingThread::parallel_gather(const SGEntry* entries, int count,
                                       char* staging, CUdeviceptr min_dst) {
    int num_workers = active_gather_workers_;
    int total_threads = num_workers + 1;
    int chunk = (count + total_threads - 1) / total_threads;

    for (int w = 0; w < num_workers; w++) {
        int start = (w + 1) * chunk;
        int end = std::min(start + chunk, count);
        if (start >= count) continue;
        gather_tasks_[w].entries = entries;
        gather_tasks_[w].staging = staging;
        gather_tasks_[w].min_dst = min_dst;
        gather_tasks_[w].start = start;
        gather_tasks_[w].end = end;
        gather_tasks_[w].done.store(false, std::memory_order_release);
        gather_tasks_[w].has_work.store(true, std::memory_order_release);
    }

    constexpr int MAIN_PREFETCH_DIST = 16;
    int main_end = std::min(chunk, count);
    for (int p = 0; p < MAIN_PREFETCH_DIST && p < main_end; p++) {
        __builtin_prefetch(entries[p].src, 0, 0);
        if (p + 1 < main_end)
            __builtin_prefetch(&entries[p + 1], 0, 3);
    }
    for (int i = 0; i < main_end; i++) {
        if (i + MAIN_PREFETCH_DIST < main_end) {
            __builtin_prefetch(entries[i + MAIN_PREFETCH_DIST].src, 0, 0);
            __builtin_prefetch(&entries[i + MAIN_PREFETCH_DIST], 0, 3);
        }
        size_t offset = entries[i].dst - min_dst;
        streaming_memcpy_g(staging + offset, entries[i].src, entries[i].size);
    }

    for (int w = 0; w < num_workers; w++) {
        int start = (w + 1) * chunk;
        if (start >= count) continue;
        while (!gather_tasks_[w].done.load(std::memory_order_acquire)) {
            cpu_pause_g();
        }
    }
}

}  // namespace gfd
