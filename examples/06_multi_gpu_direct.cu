// ============================================================
// GFD Direct 8-GPU Benchmark
//
// Comprehensive GFD Direct mode performance test across 8 GPUs.
// All GPUs execute simultaneously to measure real aggregate
// throughput under full system load.
//
// Tests:
//   Group A: Vary num_tokens (fixed token_size = 4KB)
//   Group B: Vary token_size (fixed num_tokens = 2048)
//
// Each config runs all 8 GPUs in parallel with synchronized start.
// Reports per-GPU P50 and aggregate bandwidth.
//
// NUMA-optimized:
//   - Per-GPU NUMA-local hugepage staging
//   - CPU source buffers on local NUMA node
//   - Persistent worker threads with spin-barrier sync
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

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// ---- Configuration ----
static constexpr int MAX_GPUS = 8;
static constexpr int SCATTER_STRIDE = 2;
static constexpr int WARMUP = 15;
static constexpr int ITERS = 50;

struct GPUConfig {
    int gpu_id;
    int numa_node;
    int core_base;
    int core_count;
};

static GPUConfig gpu_configs[MAX_GPUS] = {
    { 0, 0,  0, 16 },
    { 1, 0, 16, 16 },
    { 2, 0, 32, 16 },
    { 3, 0, 48, 16 },
    { 4, 1, 64, 16 },
    { 5, 1, 80, 16 },
    { 6, 1, 96, 16 },
    { 7, 1, 112, 16 },
};

// Per-GPU resources
struct GPUContext {
    CUcontext       cu_ctx;
    char*           cpu_buf;
    char*           gpu_buf;
    size_t          cpu_buf_size;
    size_t          gpu_buf_size;
    gfd::DescriptorQueue* queue;
    gfd::CpuPollingThread* poller;
};

// ---- Spin barrier ----
class SpinBarrier {
public:
    SpinBarrier() : count_(0), waiting_(0), generation_(0) {}
    void reset(int count) {
        count_ = count;
        waiting_.store(0, std::memory_order_relaxed);
    }
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

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1;
    if (hi >= v.size()) return v.back();
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static const char* fmt_size(size_t bytes, char* buf, size_t buflen) {
    if (bytes >= 1024 * 1024)
        snprintf(buf, buflen, "%zuMB", bytes / (1024 * 1024));
    else if (bytes >= 1024)
        snprintf(buf, buflen, "%zuKB", bytes / 1024);
    else
        snprintf(buf, buflen, "%zuB", bytes);
    return buf;
}

// ---- Test configuration ----
struct TestConfig {
    int num_tokens;
    int token_bytes;
};

// ---- Result ----
struct Result {
    int    num_tokens;
    int    token_bytes;
    size_t total_bytes;
    double single_p50;     // Single GPU P50 (us)
    double single_bw;      // Single GPU BW (GB/s)
    double all_p50;        // All-GPU max P50 (us)
    double all_agg_bw;     // All-GPU aggregate BW (GB/s)
    double efficiency;     // Scaling efficiency (%)
};

int main() {
    cuInit(0);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    printf("============================================================\n");
    printf("  GFD Direct 8-GPU Benchmark\n");
    printf("============================================================\n");
    printf("GPUs: %d x ", num_gpus);
    {
        char name[256];
        CUdevice dev;
        cuDeviceGet(&dev, 0);
        cuDeviceGetName(name, sizeof(name), dev);
        printf("%s\n", name);
    }
    printf("Warmup: %d, Iterations: %d, Scatter stride: %dx\n", WARMUP, ITERS, SCATTER_STRIDE);
    printf("Staging: per-GPU NUMA-local hugepages (self-allocated)\n\n");

    // ---- Test configs ----
    TestConfig group_a[] = {
        {    16,  4096 },
        {    64,  4096 },
        {   256,  4096 },
        {  1024,  4096 },
        {  2048,  4096 },
        {  4096,  4096 },
        {  8192,  4096 },
    };
    int na = sizeof(group_a) / sizeof(group_a[0]);

    TestConfig group_b[] = {
        { 2048,    512 },
        { 2048,   1024 },
        { 2048,   2048 },
        { 2048,   4096 },
        { 2048,   8192 },
        { 2048,  16384 },
        { 2048,  32768 },
    };
    int nb = sizeof(group_b) / sizeof(group_b[0]);

    // Find max sizes for allocation
    size_t max_total = 0, max_cpu = 0;
    int max_tokens = 0;
    for (int i = 0; i < na; i++) {
        size_t t = (size_t)group_a[i].num_tokens * group_a[i].token_bytes;
        if (t > max_total) max_total = t;
        if (group_a[i].num_tokens > max_tokens) max_tokens = group_a[i].num_tokens;
    }
    for (int i = 0; i < nb; i++) {
        size_t t = (size_t)group_b[i].num_tokens * group_b[i].token_bytes;
        if (t > max_total) max_total = t;
        if (group_b[i].num_tokens > max_tokens) max_tokens = group_b[i].num_tokens;
    }
    max_cpu = max_total * SCATTER_STRIDE;

    printf("Max per-GPU transfer: %zu MB, Max CPU buffer: %zu MB\n\n",
           max_total / (1024*1024), max_cpu / (1024*1024));

    // ---- Initialize GPUs ----
    std::vector<GPUContext> contexts(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        GPUConfig& gcfg = gpu_configs[i];

        cudaSetDevice(i);
        CUdevice dev;
        cuDeviceGet(&dev, i);
#if CUDA_VERSION >= 13000
        CUctxCreateParams ctxParams = {};
        cuCtxCreate(&ctx.cu_ctx, &ctxParams, 0, dev);
#else
        cuCtxCreate(&ctx.cu_ctx, 0, dev);
#endif

        // NUMA-local CPU buffer
#ifdef __linux__
        unsigned long nodemask = 1UL << gcfg.numa_node;
        set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
#endif
        cudaMallocHost(&ctx.cpu_buf, max_cpu);
        memset(ctx.cpu_buf, 0xAB, max_cpu);
        ctx.cpu_buf_size = max_cpu;
#ifdef __linux__
        set_mempolicy(MPOL_DEFAULT, NULL, 0);
#endif

        // GPU buffer
        cudaMalloc(&ctx.gpu_buf, max_total);
        ctx.gpu_buf_size = max_total;

        // Queue
        cudaHostAlloc(&ctx.queue, sizeof(gfd::DescriptorQueue), cudaHostAllocMapped);
        memset(ctx.queue, 0, sizeof(gfd::DescriptorQueue));

        // Poller (no shared pool → self-allocates NUMA-local staging)
        ctx.poller = new gfd::CpuPollingThread(
            ctx.queue, ctx.gpu_buf, ctx.cpu_buf, max_total,
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

    printf("All %d GPUs initialized\n", num_gpus);

    // ---- SG entry buffers (one per GPU, host pinned) ----
    std::vector<gfd::SGEntry*> sg_bufs(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaHostAlloc(&sg_bufs[i], max_tokens * sizeof(gfd::SGEntry), cudaHostAllocMapped);
    }

    // ---- Run a single config on all GPUs ----
    SpinBarrier barrier;

    auto run_config = [&](const TestConfig& cfg) -> Result {
        int N = cfg.num_tokens;
        int T = cfg.token_bytes;
        size_t total = (size_t)N * T;
        size_t stride = (size_t)T * SCATTER_STRIDE;

        // Build SG entries for each GPU
        for (int g = 0; g < num_gpus; g++) {
            GPUContext& ctx = contexts[g];
            for (int i = 0; i < N; i++) {
                sg_bufs[g][i].dst  = (CUdeviceptr)(ctx.gpu_buf + (size_t)i * T);
                sg_bufs[g][i].src  = ctx.cpu_buf + (size_t)i * stride;
                sg_bufs[g][i].size = T;
            }
        }

        // --- Single GPU test (GPU 0) ---
        {
            GPUContext& ctx = contexts[0];
            cuCtxSetCurrent(ctx.cu_ctx);
            for (int w = 0; w < WARMUP; w++)
                ctx.poller->submit_direct(sg_bufs[0], N);
        }
        std::vector<double> single_lats(ITERS);
        {
            GPUContext& ctx = contexts[0];
            cuCtxSetCurrent(ctx.cu_ctx);
            for (int iter = 0; iter < ITERS; iter++)
                single_lats[iter] = ctx.poller->submit_direct(sg_bufs[0], N);
        }
        double single_p50 = percentile(single_lats, 50);
        double single_bw = total / (single_p50 * 1e3);

        // --- All-GPU parallel test ---
        std::vector<std::vector<double>> per_gpu_lats(num_gpus);
        for (auto& v : per_gpu_lats) v.resize(ITERS);

        barrier.reset(num_gpus);

        std::vector<std::thread> workers;
        for (int g = 0; g < num_gpus; g++) {
            workers.emplace_back([&, g]() {
                GPUContext& ctx = contexts[g];
                cuCtxSetCurrent(ctx.cu_ctx);
#ifdef __linux__
                // Pin control thread
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(gpu_configs[g].core_base + gpu_configs[g].core_count - 1, &cpuset);
                sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
                // Warmup
                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    ctx.poller->submit_direct(sg_bufs[g], N);
                }
                // Timed
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    per_gpu_lats[g][iter] = ctx.poller->submit_direct(sg_bufs[g], N);
                }
            });
        }
        for (auto& t : workers) t.join();

        // Max latency per iteration (aggregate completes when slowest finishes)
        std::vector<double> max_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            double m = 0;
            for (int g = 0; g < num_gpus; g++)
                m = std::max(m, per_gpu_lats[g][iter]);
            max_lats[iter] = m;
        }
        double all_p50 = percentile(max_lats, 50);
        double all_agg_bw = (double)num_gpus * total / (all_p50 * 1e3);
        double eff = all_agg_bw / (single_bw * num_gpus) * 100.0;

        return { N, T, total, single_p50, single_bw, all_p50, all_agg_bw, eff };
    };

    // ---- Group A ----
    printf("\nRunning Group A: vary num_tokens (token_size = 4KB) ...\n");
    fflush(stdout);
    std::vector<Result> results_a;
    for (int i = 0; i < na; i++) {
        results_a.push_back(run_config(group_a[i]));
        printf("  [%d/%d] %d x 4KB done\n", i + 1, na, group_a[i].num_tokens);
        fflush(stdout);
    }

    // ---- Group B ----
    printf("\nRunning Group B: vary token_size (num_tokens = 2048) ...\n");
    fflush(stdout);
    std::vector<Result> results_b;
    for (int i = 0; i < nb; i++) {
        results_b.push_back(run_config(group_b[i]));
        char sz[16];
        fmt_size(group_b[i].token_bytes, sz, sizeof(sz));
        printf("  [%d/%d] 2048 x %s done\n", i + 1, nb, sz);
        fflush(stdout);
    }

    // ---- Print Results ----
    auto print_table = [&](const char* title, const std::vector<Result>& results) {
        printf("\n%s\n", title);
        printf("+------------------+--------+-----------+-----------+-----------+-----------+-------+\n");
        printf("| %-16s | %6s | %9s | %9s | %9s | %9s | %5s |\n",
               "Config", "Total", "1-GPU P50", "1-GPU BW", "8-GPU P50", "8-GPU Agg", "Eff.");
        printf("|                  |        | %9s | %9s | %9s | %9s |       |\n",
               "(us)", "(GB/s)", "(us)", "(GB/s)");
        printf("+------------------+--------+-----------+-----------+-----------+-----------+-------+\n");

        for (auto& r : results) {
            char tok_str[16], tot_str[16], label[24];
            fmt_size(r.token_bytes, tok_str, sizeof(tok_str));
            fmt_size(r.total_bytes, tot_str, sizeof(tot_str));
            snprintf(label, sizeof(label), "%d x %s", r.num_tokens, tok_str);

            printf("| %-16s | %6s | %9.1f | %9.2f | %9.1f | %9.2f | %4.1f%% |\n",
                   label, tot_str,
                   r.single_p50, r.single_bw,
                   r.all_p50, r.all_agg_bw,
                   r.efficiency);
        }
        printf("+------------------+--------+-----------+-----------+-----------+-----------+-------+\n");
    };

    printf("\n");
    printf("================================================================\n");
    printf("  Group A: Vary num_tokens (token_size = 4KB)\n");
    printf("================================================================\n");
    print_table("", results_a);

    printf("\n");
    printf("================================================================\n");
    printf("  Group B: Vary token_size (num_tokens = 2048)\n");
    printf("================================================================\n");
    print_table("", results_b);

    // ---- Peak summary ----
    double peak_single = 0, peak_agg = 0;
    for (auto& r : results_a) {
        if (r.single_bw > peak_single) peak_single = r.single_bw;
        if (r.all_agg_bw > peak_agg) peak_agg = r.all_agg_bw;
    }
    for (auto& r : results_b) {
        if (r.single_bw > peak_single) peak_single = r.single_bw;
        if (r.all_agg_bw > peak_agg) peak_agg = r.all_agg_bw;
    }

    printf("\n");
    printf("================================================================\n");
    printf("  Summary\n");
    printf("================================================================\n");
    printf("  GPUs:              %d x NVIDIA RTX PRO 5000 72GB Blackwell\n", num_gpus);
    printf("  Peak 1-GPU BW:     %.2f GB/s\n", peak_single);
    printf("  Peak 8-GPU Agg:    %.2f GB/s\n", peak_agg);
    printf("  Peak efficiency:   %.1f%%\n", peak_agg / (peak_single * num_gpus) * 100.0);
    printf("================================================================\n");

    // ---- Cleanup ----
    for (int i = 0; i < num_gpus; i++) {
        cuCtxSetCurrent(contexts[i].cu_ctx);
        contexts[i].poller->stop();
        delete contexts[i].poller;
        cudaFreeHost(sg_bufs[i]);
        cudaFreeHost(contexts[i].queue);
        cudaFree(contexts[i].gpu_buf);
        cudaFreeHost(contexts[i].cpu_buf);
    }

    return 0;
}
