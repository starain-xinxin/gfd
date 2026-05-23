// ============================================================
// GFD Multi-GPU Benchmark (8 GPUs, NUMA-Optimized)
//
// Measures aggregate H2D transfer bandwidth across multiple GPUs
// using GFD Direct mode with full NUMA-aware optimization.
//
// Optimizations:
//   - Per-GPU staging buffers allocated on local NUMA node
//     (bypasses shared StagingPool to avoid cross-NUMA staging)
//   - CPU source buffers pinned to local NUMA node
//   - Persistent worker threads (eliminates thread spawn overhead)
//   - Spin-barrier synchronization for tight parallel launch
//   - NUMA-aware gather worker core pinning
//
// Topology (auto-detected):
//   GPU 0-3: NUMA node 0, CPUs 0-63 + 128-191
//   GPU 4-7: NUMA node 1, CPUs 64-127 + 192-255
// ============================================================

#include <gfd/gfd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <functional>
#include <mutex>
#include <condition_variable>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// ---- Configuration ----
static constexpr int MAX_GPUS = 8;
static constexpr int NUM_TOKENS = 2048;
static constexpr int TOKEN_SIZE = 4096;       // 4KB per token
static constexpr size_t TOTAL_SIZE = (size_t)NUM_TOKENS * TOKEN_SIZE;  // 8MB per GPU
static constexpr int SCATTER_STRIDE = 2;
static constexpr int WARMUP = 15;
static constexpr int ITERS = 50;

// Per-GPU NUMA mapping
struct GPUConfig {
    int gpu_id;
    int numa_node;
    int core_base;
    int core_count;
};

// Per-GPU resources
struct GPUContext {
    int             gpu_id;
    CUcontext       cu_ctx;
    char*           cpu_buf;
    char*           gpu_buf;
    gfd::DescriptorQueue* queue;
    gfd::CpuPollingThread* poller;
    gfd::SGEntry*   sg_entries;
    std::vector<double> latencies;
};

// ---- Spin barrier (lock-free, reusable) ----
class SpinBarrier {
public:
    SpinBarrier(int count) : count_(count), waiting_(0), generation_(0) {}

    void wait() {
        int gen = generation_.load(std::memory_order_acquire);
        if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            waiting_.store(0, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_release);
        } else {
            while (generation_.load(std::memory_order_acquire) == gen) {
                __builtin_ia32_pause();
            }
        }
    }

    void reset(int count) {
        count_ = count;
        waiting_.store(0);
    }

private:
    int count_;
    std::atomic<int> waiting_;
    std::atomic<int> generation_;
};

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1;
    if (hi >= v.size()) return v.back();
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// Pin current thread to a specific CPU core
static void pin_to_cpu(int cpu) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

int main() {
    cuInit(0);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    printf("============================================================\n");
    printf("  GFD Multi-GPU Benchmark (NUMA-Optimized)\n");
    printf("============================================================\n");
    printf("GPUs detected: %d\n", num_gpus);
    printf("Per-GPU transfer: %d tokens x %d bytes = %zu MB (scattered at %dx stride)\n",
           NUM_TOKENS, TOKEN_SIZE, TOTAL_SIZE / (1024 * 1024), SCATTER_STRIDE);
    printf("Warmup: %d, Iterations: %d\n\n", WARMUP, ITERS);

    // Print GPU names
    for (int i = 0; i < num_gpus; i++) {
        char name[256];
        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuDeviceGetName(name, sizeof(name), dev);
        printf("  GPU %d: %s\n", i, name);
    }
    printf("\n");

    // NUMA-aware core assignments (16 cores per GPU):
    // GPU 0-3 (NUMA 0): cores 0-63
    // GPU 4-7 (NUMA 1): cores 64-127
    GPUConfig gpu_configs[MAX_GPUS] = {
        { 0, 0,  0, 16 },
        { 1, 0, 16, 16 },
        { 2, 0, 32, 16 },
        { 3, 0, 48, 16 },
        { 4, 1, 64, 16 },
        { 5, 1, 80, 16 },
        { 6, 1, 96, 16 },
        { 7, 1, 112, 16 },
    };

    // ---- Do NOT use shared StagingPool ----
    // Each poller self-allocates NUMA-local hugepage staging buffers
    // when pool is not initialized.

    // ---- Initialize all GPU contexts ----
    std::vector<GPUContext> contexts(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        ctx.gpu_id = i;
        GPUConfig& gcfg = gpu_configs[i];

        // Set GPU
        cudaSetDevice(i);
        CUdevice dev;
        cuDeviceGet(&dev, i);
#if CUDA_VERSION >= 13000
        CUctxCreateParams ctxParams = {};
        cuCtxCreate(&ctx.cu_ctx, &ctxParams, 0, dev);
#else
        cuCtxCreate(&ctx.cu_ctx, 0, dev);
#endif

        // Bind memory allocation to correct NUMA node
#ifdef __linux__
        unsigned long nodemask = 1UL << gcfg.numa_node;
        set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
#endif

        // Allocate pinned CPU buffer (on local NUMA node)
        cudaMallocHost(&ctx.cpu_buf, TOTAL_SIZE * SCATTER_STRIDE);
        for (int t = 0; t < NUM_TOKENS; t++) {
            char* ptr = ctx.cpu_buf + (size_t)t * TOKEN_SIZE * SCATTER_STRIDE;
            memset(ptr, (uint8_t)((i * NUM_TOKENS + t) & 0xFF), TOKEN_SIZE);
        }

        // Reset memory policy
#ifdef __linux__
        set_mempolicy(MPOL_DEFAULT, NULL, 0);
#endif

        // Allocate GPU buffer
        cudaMalloc(&ctx.gpu_buf, TOTAL_SIZE);

        // Allocate descriptor queue (pinned, for poller)
        cudaHostAlloc(&ctx.queue, sizeof(gfd::DescriptorQueue), cudaHostAllocMapped);
        memset(ctx.queue, 0, sizeof(gfd::DescriptorQueue));

        // Build SG entries (pinned host)
        cudaHostAlloc(&ctx.sg_entries, NUM_TOKENS * sizeof(gfd::SGEntry), cudaHostAllocMapped);
        for (int t = 0; t < NUM_TOKENS; t++) {
            ctx.sg_entries[t].dst  = (CUdeviceptr)(ctx.gpu_buf + (size_t)t * TOKEN_SIZE);
            ctx.sg_entries[t].src  = ctx.cpu_buf + (size_t)t * TOKEN_SIZE * SCATTER_STRIDE;
            ctx.sg_entries[t].size = TOKEN_SIZE;
        }

        // Create polling thread:
        // No shared pool → poller will self-allocate NUMA-local hugepage staging
        ctx.poller = new gfd::CpuPollingThread(
            ctx.queue, ctx.gpu_buf, ctx.cpu_buf, TOTAL_SIZE,
            /*use_ce=*/true, /*numa_node=*/gcfg.numa_node,
            /*core_offset=*/0, /*num_ce_channels=*/0,
            /*exclusive_core_base=*/gcfg.core_base,
            /*exclusive_core_count=*/gcfg.core_count);

        if (!ctx.poller->init_copy_engine()) {
            fprintf(stderr, "GPU %d: Failed to init copy engine\n", i);
            return 1;
        }
        ctx.poller->init_direct_ce();
        ctx.poller->start();
    }

    printf("All %d GPUs initialized (per-GPU NUMA-local staging)\n\n", num_gpus);

    // ---- Test 1: Per-GPU sequential bandwidth ----
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Test 1: Per-GPU Bandwidth (sequential)\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        cuCtxSetCurrent(ctx.cu_ctx);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
        }

        // Timed runs
        ctx.latencies.resize(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            ctx.latencies[iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
        }

        double p50 = percentile(ctx.latencies, 50);
        double bw = TOTAL_SIZE / (p50 * 1e3);
        printf("  GPU %d (NUMA %d): P50 = %7.1f us, BW = %6.2f GB/s\n",
               i, gpu_configs[i].numa_node, p50, bw);
    }

    // ---- Test 2: Parallel scaling with persistent threads ----
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 2: Aggregate Bandwidth (parallel, persistent threads)\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    std::vector<int> gpu_counts;
    for (int n = 1; n <= num_gpus; n *= 2) {
        gpu_counts.push_back(n);
    }
    if (gpu_counts.back() != num_gpus) {
        gpu_counts.push_back(num_gpus);
    }

    printf("  +-----------+----------+----------+----------+------------+----------+\n");
    printf("  | %9s | %8s | %8s | %8s | %10s | %8s |\n",
           "GPUs", "P50 (us)", "P90 (us)", "Max (us)", "Agg BW", "Eff.");
    printf("  +-----------+----------+----------+----------+------------+----------+\n");

    double single_gpu_bw = 0;

    for (int num_active : gpu_counts) {
        // Per-GPU results
        std::vector<std::vector<double>> per_gpu_lats(num_active);
        for (auto& v : per_gpu_lats) v.resize(ITERS);

        SpinBarrier barrier(num_active);

        // Launch persistent worker threads
        std::vector<std::thread> workers;
        for (int i = 0; i < num_active; i++) {
            workers.emplace_back([&, i]() {
                GPUContext& ctx = contexts[i];
                cuCtxSetCurrent(ctx.cu_ctx);
                // Pin this worker thread to a dedicated core
                pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                // Phase 0: Warmup
                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                }

                // Phase 1: Timed iterations
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    per_gpu_lats[i][iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                }
            });
        }

        for (auto& t : workers) t.join();

        // Compute max-across-GPUs latency per iteration
        std::vector<double> max_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            double m = 0;
            for (int i = 0; i < num_active; i++) {
                m = std::max(m, per_gpu_lats[i][iter]);
            }
            max_lats[iter] = m;
        }

        double p50 = percentile(max_lats, 50);
        double p90 = percentile(max_lats, 90);
        double max_val = *std::max_element(max_lats.begin(), max_lats.end());
        double agg_bw = (double)num_active * TOTAL_SIZE / (p50 * 1e3);

        if (num_active == 1) single_gpu_bw = agg_bw;
        double efficiency = agg_bw / (single_gpu_bw * num_active) * 100.0;

        printf("  | %4d GPU%s | %8.1f | %8.1f | %8.1f | %7.2f GB/s | %5.1f%%  |\n",
               num_active, num_active > 1 ? "s" : " ",
               p50, p90, max_val, agg_bw, efficiency);
    }
    printf("  +-----------+----------+----------+----------+------------+----------+\n");

    // ---- Test 3: NUMA locality analysis ----
    if (num_gpus >= 8) {
        printf("\n────────────────────────────────────────────────────────────\n");
        printf("  Test 3: NUMA Locality Analysis\n");
        printf("────────────────────────────────────────────────────────────\n\n");

        auto run_group = [&](const char* label, int start, int count) {
            std::vector<std::vector<double>> per_gpu_lats(count);
            for (auto& v : per_gpu_lats) v.resize(ITERS);

            SpinBarrier barrier(count);
            std::vector<std::thread> workers;

            for (int idx = 0; idx < count; idx++) {
                int i = start + idx;
                workers.emplace_back([&, i, idx]() {
                    GPUContext& ctx = contexts[i];
                    cuCtxSetCurrent(ctx.cu_ctx);
                    pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                    // Warmup
                    for (int w = 0; w < WARMUP; w++) {
                        barrier.wait();
                        ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                    }
                    // Timed
                    for (int iter = 0; iter < ITERS; iter++) {
                        barrier.wait();
                        per_gpu_lats[idx][iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                    }
                });
            }
            for (auto& t : workers) t.join();

            std::vector<double> max_lats(ITERS);
            for (int iter = 0; iter < ITERS; iter++) {
                double m = 0;
                for (int idx = 0; idx < count; idx++)
                    m = std::max(m, per_gpu_lats[idx][iter]);
                max_lats[iter] = m;
            }
            double p50 = percentile(max_lats, 50);
            double agg_bw = (double)count * TOTAL_SIZE / (p50 * 1e3);
            printf("  %s: P50 = %7.1f us, Aggregate = %6.2f GB/s\n", label, p50, agg_bw);
        };

        run_group("NUMA 0 (GPU 0-3)", 0, 4);
        run_group("NUMA 1 (GPU 4-7)", 4, 4);
        run_group("All 8 GPUs       ", 0, 8);
    }

    // ---- Test 4: Per-GPU detailed stats in parallel mode ----
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 4: Per-GPU Bandwidth Under Full Load (all %d GPUs)\n", num_gpus);
    printf("────────────────────────────────────────────────────────────\n\n");

    {
        std::vector<std::vector<double>> per_gpu_lats(num_gpus);
        for (auto& v : per_gpu_lats) v.resize(ITERS);

        SpinBarrier barrier(num_gpus);
        std::vector<std::thread> workers;

        for (int i = 0; i < num_gpus; i++) {
            workers.emplace_back([&, i]() {
                GPUContext& ctx = contexts[i];
                cuCtxSetCurrent(ctx.cu_ctx);
                pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                }
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    per_gpu_lats[i][iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                }
            });
        }
        for (auto& t : workers) t.join();

        for (int i = 0; i < num_gpus; i++) {
            double p50 = percentile(per_gpu_lats[i], 50);
            double p90 = percentile(per_gpu_lats[i], 90);
            double bw = TOTAL_SIZE / (p50 * 1e3);
            printf("  GPU %d (NUMA %d): P50 = %7.1f us, P90 = %7.1f us, BW = %6.2f GB/s\n",
                   i, gpu_configs[i].numa_node, p50, p90, bw);
        }

        // Summary
        double total_bw = 0;
        for (int i = 0; i < num_gpus; i++) {
            double p50 = percentile(per_gpu_lats[i], 50);
            total_bw += TOTAL_SIZE / (p50 * 1e3);
        }
        printf("  ─────────────────────────────────────────────────\n");
        printf("  Sum of per-GPU BW: %.2f GB/s\n", total_bw);
    }

    // ---- Summary ----
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Summary\n");
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Per-GPU transfer size:  %zu MB (%d x %d bytes)\n",
           TOTAL_SIZE / (1024 * 1024), NUM_TOKENS, TOKEN_SIZE);
    printf("  Single GPU baseline:    %.2f GB/s\n", single_gpu_bw);
    printf("  Theoretical %d-GPU max:  %.2f GB/s\n", num_gpus, single_gpu_bw * num_gpus);
    printf("  Staging: per-GPU NUMA-local hugepages (no shared pool)\n");
    printf("============================================================\n");

    // ---- Cleanup ----
    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        cuCtxSetCurrent(ctx.cu_ctx);
        ctx.poller->stop();
        delete ctx.poller;
        cudaFreeHost(ctx.sg_entries);
        cudaFreeHost(ctx.queue);
        cudaFree(ctx.gpu_buf);
        cudaFreeHost(ctx.cpu_buf);
    }

    return 0;
}
