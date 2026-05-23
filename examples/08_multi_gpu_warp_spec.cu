// ============================================================
// GFD Multi-GPU Warp-Specialized Transfer+Compute (8 GPUs)
//
// Demonstrates warp-spec transfer+compute overlap across 8 GPUs
// with NUMA-aware CPU poller pinning and dedicated CE channels.
//
// Uses low-level TiledQueue + CpuPollingThread API for full
// control over per-GPU core assignment (WarpSpecSession doesn't
// expose exclusive_core_base/count needed for multi-GPU).
//
// Tests:
//   1. Per-GPU sequential bandwidth (warp-spec + compute)
//   2. All-GPU parallel (warp-spec + compute)
//   3. All-GPU pure transfer (no compute, max bandwidth)
//   4. Scaling analysis (1/2/4/8 GPUs)
// ============================================================

#include <gfd/gfd.h>
#include <gfd/warp_spec.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>

#include <unistd.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// ---- Configuration ----
static constexpr int MAX_GPUS = 8;
static constexpr int TOTAL_TOKENS = 8192;
static constexpr int TOKENS_PER_TILE = 128;
static constexpr int TOKENS_PER_CHUNK = 32;
static constexpr int TOKEN_DIM = 4096;
static constexpr size_t TOKEN_SIZE = TOKEN_DIM * sizeof(float);  // 16KB
static constexpr size_t TOTAL_SIZE = (size_t)TOTAL_TOKENS * TOKEN_SIZE;  // 128MB
static constexpr int NUM_TILES = TOTAL_TOKENS / TOKENS_PER_TILE;
static constexpr int CHUNKS_PER_TILE = TOKENS_PER_TILE / TOKENS_PER_CHUNK;

static constexpr int WARMUP = 5;
static constexpr int ITERS = 20;

// ---- Compute Functor ----
struct VectorReduce {
    float* output;
    int token_dim;

    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32) {
            float* token = chunk.data<float>(i);
            float sq_sum = 0.0f;
            for (int d = 0; d < token_dim; d++)
                sq_sum += token[d] * token[d];
            float rms_inv = rsqrtf(sq_sum / token_dim + 1e-6f);
            float acc = 0.0f;
            for (int d = 0; d < token_dim; d++) {
                float normed = token[d] * rms_inv;
                acc += sinf(normed) * normed;
            }
            output[chunk.global_idx(i)] = acc;
        }
    }
};

GFD_WARP_SPEC_KERNEL(warp_spec_kernel, VectorReduce);

struct NoOp {
    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {}
};

GFD_WARP_SPEC_KERNEL(noop_kernel, NoOp);

// ---- Spin Barrier ----
class SpinBarrier {
public:
    SpinBarrier(int count) : count_(count), waiting_(0), generation_(0) {}
    void wait() {
        int gen = generation_.load(std::memory_order_acquire);
        if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            waiting_.store(0, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_release);
        } else {
            while (generation_.load(std::memory_order_acquire) == gen)
                __builtin_ia32_pause();
        }
    }
private:
    int count_;
    std::atomic<int> waiting_;
    std::atomic<int> generation_;
};

// ---- Per-GPU State ----
struct GPUConfig {
    int gpu_id;
    int numa_node;
    int core_base;
    int core_count;
};

struct GPUState {
    int gpu_id;
    CUcontext cu_ctx;
    int num_sms;

    // Memory
    float* cpu_data;
    float* gpu_data;
    float* d_output;

    // GFD infrastructure (manual setup for NUMA control)
    gfd::TiledQueue* tq;
    gfd::CpuPollingThread* poller;
};

// ---- Helpers ----
static void pin_to_cpu(int cpu) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1;
    if (hi >= v.size()) return v.back();
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// Reset TiledQueue state for a new launch
static void reset_tq(gfd::TiledQueue* tq) {
    memset((void*)tq->tile_chunk_done, 0, sizeof(tq->tile_chunk_done));
    tq->scheduler.next_tile = 0;
    if (tq->d_tile_chunk_done) {
        cudaMemset(tq->d_tile_chunk_done, 0, gfd::MAX_TILES * sizeof(uint64_t));
    }
}

// Run one iteration: launch kernel + start poller + sync + stop poller
template<typename KernelFn, typename ComputeFn>
static double run_once(GPUState& st, KernelFn kernel, ComputeFn compute,
                       int block_size) {
    reset_tq(st.tq);
    st.poller->reset_stats();

    auto t0 = std::chrono::high_resolution_clock::now();
    kernel<<<dim3(st.num_sms), dim3(block_size)>>>(
        st.tq, st.gpu_data, st.cpu_data, compute);
    st.poller->start();
    cudaDeviceSynchronize();
    st.poller->stop();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- Main ----
int main() {
    cuInit(0);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    int block_size = (1 + CHUNKS_PER_TILE) * 32;

    printf("============================================================\n");
    printf("  GFD Multi-GPU Warp-Specialized Transfer+Compute\n");
    printf("============================================================\n");
    printf("GPUs: %d\n", num_gpus);
    printf("Per-GPU: %d tokens x %zuKB = %.0f MB, %d tiles, K=%d\n",
           TOTAL_TOKENS, TOKEN_SIZE / 1024, TOTAL_SIZE / (1024.0 * 1024.0),
           NUM_TILES, CHUNKS_PER_TILE);
    printf("Block: %d warps (%d threads), Compute: RMSNorm+sinf\n",
           1 + CHUNKS_PER_TILE, block_size);
    printf("Warmup: %d, Iterations: %d\n\n", WARMUP, ITERS);

    // NUMA topology: 4 GPUs per NUMA node, 16 cores per GPU
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

    // ---- Initialize all GPUs ----
    std::vector<GPUState> states(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        GPUState& st = states[i];
        GPUConfig& cfg = gpu_configs[i];
        st.gpu_id = i;

        cudaSetDevice(i);
        CUdevice dev;
        cuDeviceGet(&dev, i);
#if CUDA_VERSION >= 13000
        CUctxCreateParams ctxParams = {};
        cuCtxCreate(&st.cu_ctx, &ctxParams, 0, dev);
#else
        cuCtxCreate(&st.cu_ctx, 0, dev);
#endif

        cudaDeviceGetAttribute(&st.num_sms, cudaDevAttrMultiProcessorCount, 0);

        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);
        printf("  GPU %d: %s (NUMA %d, cores %d-%d)\n",
               i, name, cfg.numa_node, cfg.core_base, cfg.core_base + cfg.core_count - 1);

        // NUMA-pinned allocation
#ifdef __linux__
        unsigned long nodemask = 1UL << cfg.numa_node;
        set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
#endif

        cudaMallocHost(&st.cpu_data, TOTAL_SIZE);
        for (int t = 0; t < TOTAL_TOKENS; t++) {
            float* dst = st.cpu_data + (size_t)t * TOKEN_DIM;
            for (int d = 0; d < TOKEN_DIM; d++)
                dst[d] = (float)(i * 10000 + t * 10 + (d % 100));
        }

#ifdef __linux__
        set_mempolicy(MPOL_DEFAULT, NULL, 0);
#endif

        cudaMalloc(&st.gpu_data, TOTAL_SIZE);
        cudaMalloc(&st.d_output, TOTAL_TOKENS * sizeof(float));

        // TiledQueue (host-mapped for GPU+CPU)
        cudaHostAlloc(&st.tq, sizeof(gfd::TiledQueue), cudaHostAllocMapped);
        memset(st.tq, 0, sizeof(gfd::TiledQueue));
        st.tq->scheduler.total_tiles = NUM_TILES;
        st.tq->scheduler.tokens_per_tile = TOKENS_PER_TILE;
        st.tq->scheduler.tokens_per_chunk = TOKENS_PER_CHUNK;
        st.tq->scheduler.chunks_per_tile = CHUNKS_PER_TILE;
        st.tq->scheduler.token_size = TOKEN_SIZE;

        // Device-side signal buffer (L2 polling instead of PCIe)
        uint64_t* d_signal = nullptr;
        cudaMalloc(&d_signal, gfd::MAX_TILES * sizeof(uint64_t));
        cudaMemset(d_signal, 0, gfd::MAX_TILES * sizeof(uint64_t));
        st.tq->d_tile_chunk_done = d_signal;

        // Staging pool (per-GPU, NUMA-local)
        gfd::StagingPool::instance().init(1, TOTAL_SIZE);

        // CpuPollingThread with exclusive NUMA-local cores
        st.poller = new gfd::CpuPollingThread(
            &st.tq->base, st.gpu_data, st.cpu_data, TOTAL_SIZE,
            /*use_ce=*/true, /*numa_node=*/cfg.numa_node,
            /*core_offset=*/0, /*num_ce_channels=*/0,
            /*exclusive_core_base=*/cfg.core_base,
            /*exclusive_core_count=*/cfg.core_count);
        st.poller->set_tiled_queue(st.tq);

        if (!st.poller->init_copy_engine()) {
            fprintf(stderr, "GPU %d: Failed to init CE\n", i);
            return 1;
        }
    }

    printf("\n");

    // ====================================================================
    // Test 1: Per-GPU Sequential (Warp-Spec + Compute)
    // ====================================================================
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Test 1: Per-GPU Warp-Spec (sequential)\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    for (int i = 0; i < num_gpus; i++) {
        GPUState& st = states[i];
        cuCtxSetCurrent(st.cu_ctx);

        // Warmup
        for (int w = 0; w < WARMUP; w++)
            run_once(st, warp_spec_kernel, VectorReduce{st.d_output, TOKEN_DIM}, block_size);

        // Timed
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++)
            lats[iter] = run_once(st, warp_spec_kernel, VectorReduce{st.d_output, TOKEN_DIM}, block_size);

        double p50 = percentile(lats, 50);
        printf("  GPU %d (NUMA %d): P50 = %.2f ms, BW = %.2f GB/s\n",
               i, gpu_configs[i].numa_node, p50, TOTAL_SIZE / (p50 * 1e6));
    }

    // ====================================================================
    // Test 2: All-GPU Parallel (Warp-Spec + Compute)
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 2: All %d GPUs Parallel (Warp-Spec + Compute)\n", num_gpus);
    printf("────────────────────────────────────────────────────────────\n\n");

    {
        std::vector<std::vector<double>> all_lats(num_gpus);
        for (auto& v : all_lats) v.resize(ITERS);

        SpinBarrier barrier(num_gpus);
        std::vector<std::thread> workers;

        for (int i = 0; i < num_gpus; i++) {
            workers.emplace_back([&, i]() {
                GPUState& st = states[i];
                cuCtxSetCurrent(st.cu_ctx);
                pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                VectorReduce compute{st.d_output, TOKEN_DIM};

                // Warmup
                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    run_once(st, warp_spec_kernel, compute, block_size);
                }
                // Timed
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    all_lats[i][iter] = run_once(st, warp_spec_kernel, compute, block_size);
                }
            });
        }
        for (auto& t : workers) t.join();

        double sum_bw = 0;
        for (int i = 0; i < num_gpus; i++) {
            double p50 = percentile(all_lats[i], 50);
            double bw = TOTAL_SIZE / (p50 * 1e6);
            sum_bw += bw;
            printf("  GPU %d: P50 = %.2f ms, BW = %.2f GB/s\n", i, p50, bw);
        }

        std::vector<double> max_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            double m = 0;
            for (int i = 0; i < num_gpus; i++)
                m = std::max(m, all_lats[i][iter]);
            max_lats[iter] = m;
        }
        double agg_p50 = percentile(max_lats, 50);
        double agg_bw = (double)num_gpus * TOTAL_SIZE / (agg_p50 * 1e6);
        printf("  ─────────────────────────────────────────\n");
        printf("  Aggregate: P50 = %.2f ms, BW = %.2f GB/s (sum = %.2f GB/s)\n",
               agg_p50, agg_bw, sum_bw);
    }

    // ====================================================================
    // Test 3: All-GPU Pure Transfer (No Compute)
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 3: All %d GPUs Pure Transfer (no compute)\n", num_gpus);
    printf("────────────────────────────────────────────────────────────\n\n");

    {
        std::vector<std::vector<double>> all_lats(num_gpus);
        for (auto& v : all_lats) v.resize(ITERS);

        SpinBarrier barrier(num_gpus);
        std::vector<std::thread> workers;

        for (int i = 0; i < num_gpus; i++) {
            workers.emplace_back([&, i]() {
                GPUState& st = states[i];
                cuCtxSetCurrent(st.cu_ctx);
                pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    run_once(st, noop_kernel, NoOp{}, block_size);
                }
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    all_lats[i][iter] = run_once(st, noop_kernel, NoOp{}, block_size);
                }
            });
        }
        for (auto& t : workers) t.join();

        double sum_bw = 0;
        for (int i = 0; i < num_gpus; i++) {
            double p50 = percentile(all_lats[i], 50);
            double bw = TOTAL_SIZE / (p50 * 1e6);
            sum_bw += bw;
            printf("  GPU %d: P50 = %.2f ms, BW = %.2f GB/s\n", i, p50, bw);
        }

        std::vector<double> max_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            double m = 0;
            for (int i = 0; i < num_gpus; i++)
                m = std::max(m, all_lats[i][iter]);
            max_lats[iter] = m;
        }
        double agg_p50 = percentile(max_lats, 50);
        double agg_bw = (double)num_gpus * TOTAL_SIZE / (agg_p50 * 1e6);
        printf("  ─────────────────────────────────────────\n");
        printf("  Aggregate: P50 = %.2f ms, BW = %.2f GB/s (sum = %.2f GB/s)\n",
               agg_p50, agg_bw, sum_bw);
    }

    // ====================================================================
    // Test 4: Scaling Analysis
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 4: Scaling Analysis\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    printf("  +-----------+----------+------------+----------+\n");
    printf("  | %9s | %8s | %10s | %8s |\n", "GPUs", "P50 (ms)", "Agg BW", "Eff.");
    printf("  +-----------+----------+------------+----------+\n");

    double single_bw = 0;
    double full_bw = 0;
    double full_eff = 0;

    std::vector<int> gpu_counts;
    for (int n = 1; n <= num_gpus; n *= 2) gpu_counts.push_back(n);
    if (gpu_counts.back() != num_gpus) gpu_counts.push_back(num_gpus);

    for (int num_active : gpu_counts) {
        std::vector<std::vector<double>> all_lats(num_active);
        for (auto& v : all_lats) v.resize(ITERS);

        SpinBarrier barrier(num_active);
        std::vector<std::thread> workers;

        for (int i = 0; i < num_active; i++) {
            workers.emplace_back([&, i]() {
                GPUState& st = states[i];
                cuCtxSetCurrent(st.cu_ctx);
                pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                VectorReduce compute{st.d_output, TOKEN_DIM};

                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    run_once(st, warp_spec_kernel, compute, block_size);
                }
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    all_lats[i][iter] = run_once(st, warp_spec_kernel, compute, block_size);
                }
            });
        }
        for (auto& t : workers) t.join();

        std::vector<double> max_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            double m = 0;
            for (int i = 0; i < num_active; i++)
                m = std::max(m, all_lats[i][iter]);
            max_lats[iter] = m;
        }
        double p50 = percentile(max_lats, 50);
        double agg_bw = (double)num_active * TOTAL_SIZE / (p50 * 1e6);

        if (num_active == 1) single_bw = agg_bw;
        double eff = agg_bw / (single_bw * num_active) * 100.0;
        full_bw = agg_bw;
        full_eff = eff;

        printf("  | %4d GPU%s | %8.2f | %7.2f GB/s | %5.1f%%  |\n",
               num_active, num_active > 1 ? "s" : " ", p50, agg_bw, eff);
    }
    printf("  +-----------+----------+------------+----------+\n");

    // Wait for any remaining GFD log output to flush before printing summary
    fflush(stdout);
    fflush(stderr);
    usleep(100000);  // 100ms

    // ====================================================================
    // Summary
    // ====================================================================
    printf("\n============================================================\n");
    printf("  Summary\n");
    printf("============================================================\n");
    printf("  Per-GPU: %d tokens x %zuKB = %.0f MB\n",
           TOTAL_TOKENS, TOKEN_SIZE / 1024, TOTAL_SIZE / (1024.0 * 1024.0));
    printf("  Total (8 GPU): %.0f MB = 1 GB\n", 8.0 * TOTAL_SIZE / (1024.0 * 1024.0));
    printf("  Block: %d warps (%d threads), K=%d chunks/tile\n",
           1 + CHUNKS_PER_TILE, block_size, CHUNKS_PER_TILE);
    printf("  Compute: RMSNorm + sinf (%d floats, 2 passes)\n", TOKEN_DIM);
    printf("  Single-GPU BW: %.2f GB/s\n", single_bw);
    printf("  %d-GPU aggregate: %.2f GB/s (%.1f%% scaling)\n",
           num_gpus, full_bw, full_eff);
    printf("============================================================\n");

    // ---- Cleanup ----
    for (int i = 0; i < num_gpus; i++) {
        GPUState& st = states[i];
        cuCtxSetCurrent(st.cu_ctx);
        st.poller->stop();
        delete st.poller;
        if (st.tq->d_tile_chunk_done) cudaFree(st.tq->d_tile_chunk_done);
        cudaFreeHost(st.tq);
        cudaFree(st.d_output);
        cudaFree(st.gpu_data);
        cudaFreeHost(st.cpu_data);
    }

    return 0;
}
