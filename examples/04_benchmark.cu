// ============================================================
// GFD vs cudaMemcpy Benchmark
//
// Compares scattered H2D transfer performance across methods:
//   1. cudaMemcpy(N):  Per-token cudaMemcpyAsync (realistic scattered)
//   2. GFD Queue:      GPU submit descriptors -> CPU poll -> CE DMA
//   3. GFD Direct:     CPU direct-submit (bypass queue, small transfers)
//
// Uses submit+wait kernel split: submit kernel writes descriptors
// and exits immediately, wait kernel polls for completion.
//
// Output: P50/P90/P99 latency and bandwidth tables
//   Group A: vary num_tokens (fixed token_size = 4KB)
//   Group B: vary token_size (fixed num_tokens = 2048)
// ============================================================

#include <gfd/gfd.h>
#include <gfd/device.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <vector>

// ---- GFD benchmark kernels (submit/wait split) ----

// Submit kernel: write descriptors and exit immediately (no spin-wait)
__global__ void bench_gfd_submit_kernel(
    gfd::DescriptorQueue* queue,
    gfd::TokenInfo* tokens,
    void* gpu_buffer,
    int num_tokens,
    uint32_t token_size,
    uint64_t base_slot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < num_tokens);

    gfd::device::write_and_commit(
        queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0,
        gpu_buffer, token_size, num_tokens);
}

// Wait kernel: lightweight single-thread completion poll
__global__ void bench_gfd_wait_kernel(
    gfd::DescriptorQueue* queue,
    uint64_t expected_done)
{
    gfd::device::wait_for_completion(queue, expected_done);
}

// ---- Helpers ----

static double percentile(std::vector<double>& v, double p) {
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1;
    if (hi >= v.size()) return v.back();
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct TimingStats {
    double p50;
    double p90;
};

static TimingStats compute_stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return { percentile(v, 50), percentile(v, 90) };
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

// ---- Benchmark routines ----

static constexpr int WARMUP = 15;
static constexpr int ITERS = 50;

static TimingStats bench_memcpy_per_token(
    void* gpu_dst, const uint64_t* cpu_addrs,
    int num_tokens, size_t token_size, cudaStream_t stream)
{
    for (int i = 0; i < WARMUP; i++) {
        for (int t = 0; t < num_tokens; t++)
            cudaMemcpyAsync((char*)gpu_dst + (size_t)t * token_size,
                            (const void*)cpu_addrs[t], token_size,
                            cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    std::vector<double> times(ITERS);
    for (int i = 0; i < ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < num_tokens; t++)
            cudaMemcpyAsync((char*)gpu_dst + (size_t)t * token_size,
                            (const void*)cpu_addrs[t], token_size,
                            cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return compute_stats(times);
}

// GFD Queue: submit + wait (split kernels, no GPU spin during DMA)
static TimingStats bench_gfd_queue(
    gfd::DescriptorQueue* queue, gfd::TokenInfo* d_tokens,
    void* gpu_dst, int num_tokens, uint32_t token_size,
    uint64_t& base_slot)
{
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;

    for (int i = 0; i < WARMUP; i++) {
        bench_gfd_submit_kernel<<<blocks, threads>>>(
            queue, d_tokens, gpu_dst, num_tokens, token_size, base_slot);
        base_slot += num_tokens;
        bench_gfd_wait_kernel<<<1, 1>>>(queue, base_slot);
        cudaDeviceSynchronize();
    }
    std::vector<double> times(ITERS);
    for (int i = 0; i < ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bench_gfd_submit_kernel<<<blocks, threads>>>(
            queue, d_tokens, gpu_dst, num_tokens, token_size, base_slot);
        base_slot += num_tokens;
        bench_gfd_wait_kernel<<<1, 1>>>(queue, base_slot);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return compute_stats(times);
}

// GFD Direct: CPU direct-submit (bypass queue)
static TimingStats bench_gfd_direct(
    gfd::CpuPollingThread& poller,
    const gfd::SGEntry* entries, int count)
{
    for (int i = 0; i < WARMUP; i++)
        poller.submit_direct(entries, count);
    std::vector<double> times(ITERS);
    for (int i = 0; i < ITERS; i++)
        times[i] = poller.submit_direct(entries, count);
    return compute_stats(times);
}

// ---- Per-config result ----
struct Result {
    int    num_tokens;
    int    token_bytes;
    size_t total_bytes;
    TimingStats mcN;   // cudaMemcpy per-token
    TimingStats gfd;   // GFD queue (submit+wait)
    TimingStats dir;   // GFD direct (p50 < 0 if skipped)
};

// ---- Table printing helpers ----

static void print_latency_table(const std::vector<Result>& results, const char* pct_label,
                                 double TimingStats::*field) {
    printf("+-----------------+--------+------------+------------+------------+\n");
    printf("| %-15s | %6s | %10s | %10s | %10s |\n",
           "Config", "Total", "Memcpy(N)", "GFD Queue", "GFD Direct");
    printf("| %-15s | %6s | %10s | %10s | %10s |\n",
           "", "", pct_label, pct_label, pct_label);
    printf("+-----------------+--------+------------+------------+------------+\n");

    for (auto& r : results) {
        char tok_str[16], tot_str[16], label[24];
        fmt_size(r.token_bytes, tok_str, sizeof(tok_str));
        fmt_size(r.total_bytes, tot_str, sizeof(tot_str));
        snprintf(label, sizeof(label), "%d x %s", r.num_tokens, tok_str);

        if (r.dir.*field > 0) {
            printf("| %-15s | %6s | %10.1f | %10.1f | %10.1f |\n",
                   label, tot_str, r.mcN.*field, r.gfd.*field, r.dir.*field);
        } else {
            printf("| %-15s | %6s | %10.1f | %10.1f | %10s |\n",
                   label, tot_str, r.mcN.*field, r.gfd.*field, "-");
        }
    }
    printf("+-----------------+--------+------------+------------+------------+\n");
}

static void print_bandwidth_table(const std::vector<Result>& results) {
    printf("+-----------------+--------+------------+------------+------------+\n");
    printf("| %-15s | %6s | %10s | %10s | %10s |\n",
           "Config", "Total", "Memcpy(N)", "GFD Queue", "GFD Direct");
    printf("| %-15s | %6s | %10s | %10s | %10s |\n",
           "", "", "GB/s", "GB/s", "GB/s");
    printf("+-----------------+--------+------------+------------+------------+\n");

    for (auto& r : results) {
        char tok_str[16], tot_str[16], label[24];
        fmt_size(r.token_bytes, tok_str, sizeof(tok_str));
        fmt_size(r.total_bytes, tot_str, sizeof(tot_str));
        snprintf(label, sizeof(label), "%d x %s", r.num_tokens, tok_str);

        double bw_mcN = r.total_bytes / (r.mcN.p50 * 1e3);
        double bw_gfd = r.total_bytes / (r.gfd.p50 * 1e3);

        if (r.dir.p50 > 0) {
            double bw_dir = r.total_bytes / (r.dir.p50 * 1e3);
            printf("| %-15s | %6s | %10.2f | %10.2f | %10.2f |\n",
                   label, tot_str, bw_mcN, bw_gfd, bw_dir);
        } else {
            printf("| %-15s | %6s | %10.2f | %10.2f | %10s |\n",
                   label, tot_str, bw_mcN, bw_gfd, "-");
        }
    }
    printf("+-----------------+--------+------------+------------+------------+\n");
}

// ============================================================
// main
// ============================================================
int main() {
    cuInit(0);
    CUcontext ctx;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxParams = {};
    cuCtxCreate(&ctx, &ctxParams, 0, dev);
#else
    cuCtxCreate(&ctx, 0, dev);
#endif

    char gpu_name[256];
    cuDeviceGetName(gpu_name, sizeof(gpu_name), dev);

    // ---- Test configurations ----
    struct Config { int num_tokens; int token_bytes; };

    // Group A: vary num_tokens (fixed token_size = 4KB)
    Config group_a[] = {
        {    16,  4096 },
        {    64,  4096 },
        {   256,  4096 },
        {  1024,  4096 },
        {  2048,  4096 },
        {  4096,  4096 },
        {  8192,  4096 },
    };
    int na = sizeof(group_a) / sizeof(group_a[0]);

    // Group C: vary num_tokens (fixed token_size = 64KB)
    Config group_c[] = {
        {    16, 65536 },
        {    64, 65536 },
        {   256, 65536 },
        {  1024, 65536 },
        {  2048, 65536 },
    };
    int nc = sizeof(group_c) / sizeof(group_c[0]);

    // Group B: vary token_size (fixed num_tokens = 2048)
    Config group_b[] = {
        { 2048,    512 },
        { 2048,   1024 },
        { 2048,   2048 },
        { 2048,   4096 },
        { 2048,   8192 },
        { 2048,  16384 },
        { 2048,  32768 },
        { 2048,  65536 },
    };
    int nb = sizeof(group_b) / sizeof(group_b[0]);

    // Merge all configs for allocation sizing
    std::vector<Config> all_configs;
    for (int i = 0; i < na; i++) all_configs.push_back(group_a[i]);
    for (int i = 0; i < nb; i++) all_configs.push_back(group_b[i]);
    for (int i = 0; i < nc; i++) all_configs.push_back(group_c[i]);

    constexpr int SCATTER_STRIDE = 2;
    size_t max_total = 0, max_cpu_buf = 0;
    int max_num_tokens = 0;
    for (auto& cfg : all_configs) {
        size_t total = (size_t)cfg.num_tokens * cfg.token_bytes;
        size_t cpu   = total * SCATTER_STRIDE;
        if (total > max_total) max_total = total;
        if (cpu > max_cpu_buf) max_cpu_buf = cpu;
        if (cfg.num_tokens > max_num_tokens) max_num_tokens = cfg.num_tokens;
    }

    // ---- Allocate resources ----
    char* cpu_buf;
    cudaMallocHost(&cpu_buf, max_cpu_buf);
    memset(cpu_buf, 0xAB, max_cpu_buf);

    char* gpu_buf;
    cudaMalloc(&gpu_buf, max_total);

    gfd::DescriptorQueue* d_queue;
    cudaMallocManaged(&d_queue, sizeof(gfd::DescriptorQueue));
    memset(d_queue, 0, sizeof(gfd::DescriptorQueue));

    gfd::TokenInfo* d_tokens;
    cudaMalloc(&d_tokens, max_num_tokens * sizeof(gfd::TokenInfo));

    std::vector<gfd::TokenInfo> h_tokens(max_num_tokens);
    std::vector<uint64_t>       h_cpu_addrs(max_num_tokens);
    std::vector<gfd::SGEntry>   h_sg_entries(max_num_tokens);

    cudaStream_t bench_stream;
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&bench_stream, cudaStreamNonBlocking, hi);

    // ---- Initialize GFD ----
    gfd::StagingPool::instance().init(1, max_total);

    gfd::CpuPollingThread poller(d_queue, gpu_buf, cpu_buf, max_total,
                                  /*use_ce=*/true, /*numa_node=*/0,
                                  /*core_offset=*/0, /*num_ce_channels=*/0,
                                  /*exclusive_core_base=*/0,
                                  /*exclusive_core_count=*/32);
    if (!poller.init_copy_engine()) {
        fprintf(stderr, "Failed to init copy engine\n");
        return 1;
    }
    poller.init_direct_ce();
    poller.start();

    uint64_t gfd_base_slot = 0;

    // ---- Global warmup ----
    {
        int gwN = 512;
        int gwT = 4096;
        size_t gw_stride = (size_t)gwT * SCATTER_STRIDE;
        for (int i = 0; i < gwN; i++) {
            h_cpu_addrs[i] = (uint64_t)(cpu_buf + (size_t)i * gw_stride);
            h_tokens[i].cpu_addr  = h_cpu_addrs[i];
            h_tokens[i].token_id  = i;
            h_tokens[i].expert_id = 0;
            h_sg_entries[i].dst  = (CUdeviceptr)(gpu_buf + (size_t)i * gwT);
            h_sg_entries[i].src  = (const void*)h_cpu_addrs[i];
            h_sg_entries[i].size = gwT;
        }
        cudaMemcpy(d_tokens, h_tokens.data(), gwN * sizeof(gfd::TokenInfo),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (gwN + threads - 1) / threads;
        for (int i = 0; i < 30; i++) {
            bench_gfd_submit_kernel<<<blocks, threads>>>(
                d_queue, d_tokens, gpu_buf, gwN, gwT, gfd_base_slot);
            gfd_base_slot += gwN;
            bench_gfd_wait_kernel<<<1, 1>>>(d_queue, gfd_base_slot);
            cudaDeviceSynchronize();
        }
        poller.submit_direct(h_sg_entries.data(), gwN);
        printf("Global warmup done (30 x 512x4KB)\n");
    }

    // ---- Run all configs, collect results ----
    auto run_config = [&](const Config& cfg) -> Result {
        int N = cfg.num_tokens;
        int T = cfg.token_bytes;
        size_t total  = (size_t)N * T;
        size_t stride = (size_t)T * SCATTER_STRIDE;

        for (int i = 0; i < N; i++) {
            h_cpu_addrs[i] = (uint64_t)(cpu_buf + (size_t)i * stride);
            h_tokens[i].cpu_addr  = h_cpu_addrs[i];
            h_tokens[i].token_id  = i;
            h_tokens[i].expert_id = 0;
            h_sg_entries[i].dst  = (CUdeviceptr)(gpu_buf + (size_t)i * T);
            h_sg_entries[i].src  = (const void*)h_cpu_addrs[i];
            h_sg_entries[i].size = T;
        }
        cudaMemcpy(d_tokens, h_tokens.data(), N * sizeof(gfd::TokenInfo),
                   cudaMemcpyHostToDevice);

        Result r;
        r.num_tokens  = N;
        r.token_bytes = T;
        r.total_bytes = total;
        r.mcN = bench_memcpy_per_token(gpu_buf, h_cpu_addrs.data(), N, T, bench_stream);
        r.gfd = bench_gfd_queue(d_queue, d_tokens, gpu_buf, N, T, gfd_base_slot);
        r.dir = {-1.0, -1.0};
        r.dir = bench_gfd_direct(poller, h_sg_entries.data(), N);

        return r;
    };

    // ---- Print header ----
    printf("GPU: %s\n", gpu_name);
    printf("Max total transfer: %zu MB, Max CPU buffer: %zu MB\n",
           max_total / (1024 * 1024), max_cpu_buf / (1024 * 1024));
    printf("Scattered layout: tokens at %dx stride in pinned CPU memory\n", SCATTER_STRIDE);
    printf("Warmup: %d, Iterations: %d\n", WARMUP, ITERS);

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

    // ---- Group C ----
    printf("\nRunning Group C: vary num_tokens (token_size = 64KB) ...\n");
    fflush(stdout);
    std::vector<Result> results_c;
    for (int i = 0; i < nc; i++) {
        results_c.push_back(run_config(group_c[i]));
        printf("  [%d/%d] %d x 64KB done\n", i + 1, nc, group_c[i].num_tokens);
        fflush(stdout);
    }

    // ============================================================
    // Print results
    // ============================================================

    printf("\n");
    printf("================================================================\n");
    printf("  Group A: Vary num_tokens (token_size = 4KB)\n");
    printf("================================================================\n");

    printf("\n  [Latency P50 (us)]\n\n");
    print_latency_table(results_a, "P50 us", &TimingStats::p50);
    printf("\n  [Bandwidth (from P50 latency)]\n\n");
    print_bandwidth_table(results_a);

    printf("\n");
    printf("================================================================\n");
    printf("  Group B: Vary token_size (num_tokens = 2048)\n");
    printf("================================================================\n");

    printf("\n  [Latency P50 (us)]\n\n");
    print_latency_table(results_b, "P50 us", &TimingStats::p50);
    printf("\n  [Bandwidth (from P50 latency)]\n\n");
    print_bandwidth_table(results_b);

    printf("\n");
    printf("================================================================\n");
    printf("  Group C: Vary num_tokens (token_size = 64KB)\n");
    printf("================================================================\n");

    printf("\n  [Latency P50 (us)]\n\n");
    print_latency_table(results_c, "P50 us", &TimingStats::p50);
    printf("\n  [Bandwidth (from P50 latency)]\n\n");
    print_bandwidth_table(results_c);

    // ---- Legend ----
    printf("\n");
    printf("Legend:\n");
    printf("  Memcpy(N)  : N individual cudaMemcpyAsync from scattered CPU addresses\n");
    printf("  GFD Queue  : GPU submit descriptors (fire-and-forget) + wait kernel\n");
    printf("  GFD Direct : CPU direct-submit, bypass queue (parallel gather)\n");

    // ---- Cleanup ----
    poller.stop();
    gfd::StagingPool::instance().shutdown();
    cudaStreamDestroy(bench_stream);
    cudaFree(d_tokens);
    cudaFree(d_queue);
    cudaFree(gpu_buf);
    cudaFreeHost(cpu_buf);

    return 0;
}
