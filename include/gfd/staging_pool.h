#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include "gfd/log.h"
#include <atomic>
#include <mutex>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#endif

namespace gfd {

// StagingPool: Pre-allocated hugepage staging buffer pool
//
// Eliminates the ~28ms cold-start cost of mmap(MAP_HUGETLB) +
// cudaHostRegister per CpuPollingThread initialization.
//
// Usage:
//   1. Call StagingPool::instance().init() once at model load
//   2. CpuPollingThread::init_copy_engine() calls acquire_buffers()
//   3. ~CpuPollingThread calls release_buffers()
//   4. Call StagingPool::instance().shutdown() at model unload
class StagingPool {
public:
    static constexpr int MAX_BUFFER_SETS = 8;   // Up to 8 GPUs
    static constexpr int BUFS_PER_SET = 5;      // 5-buffered staging
    static constexpr size_t MAX_BUF_SIZE = 128 * 1024 * 1024;  // 128MB max

    struct BufferSet {
        char* bufs[BUFS_PER_SET] = {};
        size_t buf_size = 0;
        size_t hp_size = 0;
        bool hugepage = false;
        bool in_use = false;
        int numa_node = 0;
    };

    static StagingPool& instance() {
        static StagingPool pool;
        return pool;
    }

    // Pre-allocate all staging buffers for num_gpus pollers.
    bool init(int num_gpus, size_t buf_size, int numa_node = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;

        num_sets_ = (num_gpus > MAX_BUFFER_SETS) ? MAX_BUFFER_SETS : num_gpus;
        buf_size_ = buf_size;

        size_t hp_size = (buf_size + (2*1024*1024 - 1)) & ~(size_t)(2*1024*1024 - 1);

        int total_allocated = 0;
        for (int s = 0; s < num_sets_; s++) {
            sets_[s].buf_size = buf_size;
            sets_[s].hp_size = hp_size;
            sets_[s].in_use = false;
            sets_[s].numa_node = numa_node;

            bool set_ok = true;
            for (int b = 0; b < BUFS_PER_SET; b++) {
#ifdef __linux__
                void* ptr = mmap(NULL, hp_size, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
                if (ptr != MAP_FAILED) {
                    unsigned long nodemask = 1UL << numa_node;
                    syscall(__NR_mbind, ptr, hp_size, 2 /*MPOL_BIND*/,
                            &nodemask, sizeof(nodemask) * 8 + 1, 0);

                    cudaError_t reg = cudaHostRegister(ptr, hp_size,
                                                       cudaHostRegisterDefault);
                    if (reg == cudaSuccess) {
                        sets_[s].bufs[b] = (char*)ptr;
                        sets_[s].hugepage = true;
                        continue;
                    }
                    munmap(ptr, hp_size);
                }
                // Fallback: cudaMallocHost
                void* fallback = nullptr;
                cudaError_t err = cudaMallocHost(&fallback, buf_size);
                if (err == cudaSuccess) {
                    madvise(fallback, buf_size, MADV_HUGEPAGE);
                    unsigned long nodemask = 1UL << numa_node;
                    syscall(__NR_mbind, fallback, buf_size, 2,
                            &nodemask, sizeof(nodemask) * 8 + 1, 1);
                    sets_[s].bufs[b] = (char*)fallback;
                    sets_[s].hugepage = false;
                } else {
                    set_ok = false;
                    break;
                }
#else
                void* ptr = nullptr;
                cudaError_t err = cudaMallocHost(&ptr, buf_size);
                if (err == cudaSuccess) {
                    sets_[s].bufs[b] = (char*)ptr;
                    sets_[s].hugepage = false;
                } else {
                    set_ok = false;
                    break;
                }
#endif
            }
            if (set_ok) total_allocated++;
        }

        initialized_ = true;
        GFD_LOG_INFO("[StagingPool] Pre-allocated %d buffer sets x %d x %zu bytes "
                "(hugepage=%s)\n",
                total_allocated, BUFS_PER_SET, buf_size,
                (total_allocated > 0 && sets_[0].hugepage) ? "yes" : "no");
        return total_allocated > 0;
    }

    int acquire_buffers(size_t required_size, int numa_node = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return -1;

        for (int s = 0; s < num_sets_; s++) {
            if (!sets_[s].in_use && sets_[s].buf_size >= required_size) {
                sets_[s].in_use = true;
                return s;
            }
        }
        return -1;
    }

    void get_buffers(int set_idx, char* out_bufs[BUFS_PER_SET],
                     size_t& out_size, bool& out_hugepage) {
        if (set_idx < 0 || set_idx >= num_sets_) return;
        for (int b = 0; b < BUFS_PER_SET; b++) {
            out_bufs[b] = sets_[set_idx].bufs[b];
        }
        out_size = sets_[set_idx].buf_size;
        out_hugepage = sets_[set_idx].hugepage;
    }

    void release_buffers(int set_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (set_idx >= 0 && set_idx < num_sets_) {
            sets_[set_idx].in_use = false;
        }
    }

    bool is_initialized() const { return initialized_; }
    size_t buffer_size() const { return buf_size_; }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;

        for (int s = 0; s < num_sets_; s++) {
            for (int b = 0; b < BUFS_PER_SET; b++) {
                if (sets_[s].bufs[b]) {
                    if (sets_[s].hugepage) {
                        cudaHostUnregister(sets_[s].bufs[b]);
#ifdef __linux__
                        munmap(sets_[s].bufs[b], sets_[s].hp_size);
#endif
                    } else {
                        cudaFreeHost(sets_[s].bufs[b]);
                    }
                    sets_[s].bufs[b] = nullptr;
                }
            }
        }
        initialized_ = false;
        GFD_LOG_INFO("[StagingPool] Shutdown complete\n");
    }

private:
    StagingPool() = default;
    ~StagingPool() { shutdown(); }
    StagingPool(const StagingPool&) = delete;
    StagingPool& operator=(const StagingPool&) = delete;

    std::mutex mutex_;
    bool initialized_ = false;
    int num_sets_ = 0;
    size_t buf_size_ = 0;
    BufferSet sets_[MAX_BUFFER_SETS];
};

}  // namespace gfd
