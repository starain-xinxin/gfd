#include <gfd/gfd.h>
#include <gfd/warp_spec.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

// ============================================================
// GFD Warp-Specialized Transfer+Compute Example
//
// Demonstrates the high-level API where users only define a compute
// functor and the framework handles all warp specialization, tile
// scheduling, slot acquisition, and synchronization automatically.
//
// Architecture:
//   - N tokens split into tiles of T tokens each
//   - Each tile is subdivided into K chunks of C tokens
//   - SM blocks dynamically acquire tiles via atomicAdd
//   - Warp 0 (transfer): submits descriptors, polls completion
//   - Warps 1..K (compute): process chunks as they arrive
//
// ============================================================

// ---- User Compute Functor ----
// This is ALL the user needs to write for the GPU side.
// Minimal compute: only reduce (sum). Purely transfer-bound workload.
// Moderate compute: 2-pass norm+scale per token.
// Pass 1: L2 norm (sum of squares + rsqrt)
// Pass 2: RMSNorm-style weighted accumulation with sinf activation
// ~4 FLOPs/element across 2 passes — designed to roughly match transfer time.
struct VectorReduce {
    float* output;
    int token_dim;

    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32) {
            float* token = chunk.data<float>(i);

            // Pass 1: compute RMS norm
            float sq_sum = 0.0f;
            for (int d = 0; d < token_dim; d++) {
                sq_sum += token[d] * token[d];
            }
            float rms_inv = rsqrtf(sq_sum / token_dim + 1e-6f);

            // Pass 2: normalized weighted reduction with sinf
            float acc = 0.0f;
            for (int d = 0; d < token_dim; d++) {
                float normed = token[d] * rms_inv;
                acc += sinf(normed) * normed;
            }
            output[chunk.global_idx(i)] = acc;
        }
    }
};

// Generate the kernel with one macro
GFD_WARP_SPEC_KERNEL(reduce_kernel, VectorReduce);

// Double-buffer version
GFD_WARP_SPEC_KERNEL_DB(reduce_kernel_db, VectorReduce);

// ---- No-op functor for pure transfer measurement ----
struct NoOp {
    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        // No compute — measure raw DMA throughput
    }
};
GFD_WARP_SPEC_KERNEL(noop_kernel, NoOp);

// ---- Baseline: global wait kernel (for comparison) ----
// Uses warp 0 for descriptor submission (warp-collective, chunk-based),
// then all threads wait for completion and compute.
__global__ void baseline_kernel(
    gfd::TiledQueue* tq,
    float* cpu_base,
    float* gpu_buffer,
    float* output,
    int tokens_per_tile,
    uint32_t token_size,
    int token_dim)
{
    int tile_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    uint32_t T = tq->scheduler.tokens_per_tile;
    uint32_t C = tq->scheduler.tokens_per_chunk;
    uint32_t K = tq->scheduler.chunks_per_tile;

    // Warp 0 submits descriptors chunk by chunk (like warp-spec transfer warp)
    if (warp_id == 0) {
        for (uint32_t chunk = 0; chunk < K; chunk++) {
            // Acquire slots for this chunk (warp-collective)
            uint64_t chunk_base = gfd::device::acquire_chunk_slots(tq, C);

            // Write + commit descriptors (warp-collective)
            gfd::device::submit_chunk(tq, chunk_base, tile_id, chunk,
                                      C, T, cpu_base, gpu_buffer, token_size);

            // Wait for this chunk's DMA completion
            if (lane_id == 0) {
                uint64_t expected = (uint64_t)(chunk + 1) * C;
                gfd::device::wait_chunk_done(tq, tile_id, expected);
            }
            __syncwarp();
        }
    }

    // All threads wait for warp 0 to finish
    __syncthreads();

    // Compute (all threads participate) — RMSNorm + sinf
    if (threadIdx.x < (int)T) {
        int global_idx = tile_id * tokens_per_tile + threadIdx.x;
        float* my_data = gpu_buffer + global_idx * token_dim;
        float sq_sum = 0.0f;
        for (int d = 0; d < token_dim; d++) {
            sq_sum += my_data[d] * my_data[d];
        }
        float rms_inv = rsqrtf(sq_sum / token_dim + 1e-6f);
        float acc = 0.0f;
        for (int d = 0; d < token_dim; d++) {
            float normed = my_data[d] * rms_inv;
            acc += sinf(normed) * normed;
        }
        output[global_idx] = acc;
    }
}

// ---- Helpers ----
static constexpr int WARMUP = 3;
static constexpr int ITERS = 10;

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    int idx = (int)(p / 100.0 * (v.size() - 1));
    return v[idx];
}

static void print_poller_stats(gfd::CpuPollingThread& poller, double elapsed_ms) {
    uint64_t descs = poller.get_descriptors_processed();
    uint64_t batches = poller.get_batches_submitted();
    uint64_t coalesced = poller.get_coalesced_entries();
    double bytes = poller.get_total_bytes_copied();
    uint64_t gather_us = poller.get_gather_us();
    uint64_t submit_us = poller.get_dma_submit_us();
    uint64_t wait_us = poller.get_dma_wait_us();
    printf("    [Poller] descs=%lu, batches=%lu, coalesced=%lu\n",
           (unsigned long)descs, (unsigned long)batches, (unsigned long)coalesced);
    printf("    [Poller] bytes=%.2f MB, gather=%lu us, submit=%lu us, wait=%lu us\n",
           bytes / (1024.0 * 1024.0), (unsigned long)gather_us,
           (unsigned long)submit_us, (unsigned long)wait_us);
    if (elapsed_ms > 0 && descs > 0) {
        printf("    [Poller] effective BW=%.2f GB/s, descs/ms=%.0f\n",
               bytes / (elapsed_ms * 1e6), descs / elapsed_ms);
    }
}

// ---- Summary Table ----
struct TestResult {
    const char* name;
    double total_mb;
    double p5_ms;
    double p50_ms;
    double p95_ms;
};

static void print_summary_table(const std::vector<TestResult>& results) {
    printf("\n+-------------------------------------------+----------+----------+------------+\n");
    printf("| %-41s | %8s | %8s | %10s |\n", "Test", "P50 (ms)", "P95 (ms)", "BW (GB/s)");
    printf("+-------------------------------------------+----------+----------+------------+\n");
    for (auto& r : results) {
        double bw = r.total_mb / r.p50_ms;  // MB / ms = GB/s
        printf("| %-41s | %8.2f | %8.2f | %10.2f |\n",
               r.name, r.p50_ms, r.p95_ms, bw);
    }
    printf("+-------------------------------------------+----------+----------+------------+\n");
}

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

    // ---- Configuration ----
    constexpr int TOTAL_TOKENS = 8192;
    constexpr int TOKENS_PER_TILE = 128;
    constexpr int TOKENS_PER_CHUNK = 32;
    constexpr int TOKEN_DIM = 4096;  // floats per token (16KB)
    constexpr size_t TOKEN_SIZE = TOKEN_DIM * sizeof(float);
    constexpr size_t TOTAL_SIZE = TOTAL_TOKENS * TOKEN_SIZE;
    constexpr int NUM_TILES = TOTAL_TOKENS / TOKENS_PER_TILE;
    constexpr int CHUNKS_PER_TILE = TOKENS_PER_TILE / TOKENS_PER_CHUNK;

    std::vector<TestResult> all_results;

    setbuf(stdout, NULL);  // Disable stdout buffering
    printf("=== GFD Warp-Specialized Transfer+Compute ===\n");
    printf("Total tokens: %d, Tiles: %d, Tokens/tile: %d, Chunks/tile: %d\n",
           TOTAL_TOKENS, NUM_TILES, TOKENS_PER_TILE, CHUNKS_PER_TILE);
    printf("Token size: %zu bytes (%d floats), Total: %.2f MB\n",
           TOKEN_SIZE, TOKEN_DIM, TOTAL_SIZE / (1024.0 * 1024.0));
    printf("Block config: %d warps (%d threads)\n",
           1 + CHUNKS_PER_TILE, (1 + CHUNKS_PER_TILE) * 32);
    printf("Compute: RMSNorm + sinf per token (~4 FLOPs/elem x %d elems, 2 passes)\n", TOKEN_DIM);

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    printf("GPU SMs: %d, Blocks launched: %d\n", num_sms, num_sms);
    printf("Warmup: %d, Iterations: %d\n\n", WARMUP, ITERS);

    // ---- Allocate memory (contiguous layout) ----
    float* cpu_data;
    cudaMallocHost(&cpu_data, TOTAL_SIZE);
    for (int t = 0; t < TOTAL_TOKENS; t++) {
        float* dst = cpu_data + (size_t)t * TOKEN_DIM;
        for (int d = 0; d < TOKEN_DIM; d++) {
            dst[d] = (float)(t * 1000 + d);
        }
    }

    float* gpu_data;
    cudaMalloc(&gpu_data, TOTAL_SIZE);
    cudaMemset(gpu_data, 0, TOTAL_SIZE);

    float* d_output;
    cudaMalloc(&d_output, TOTAL_TOKENS * sizeof(float));

    // ====================================================================
    // Test 0: Pure GFD Transfer (no compute) — max bandwidth reference
    // ====================================================================
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Test 0: Pure GFD Transfer (no compute)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        gfd::WarpSpecConfig config;
        config.total_tokens = TOTAL_TOKENS;
        config.token_size = TOKEN_SIZE;
        config.cpu_src = cpu_data;
        config.gpu_dst = gpu_data;
        config.tokens_per_tile = TOKENS_PER_TILE;
        config.tokens_per_chunk = TOKENS_PER_CHUNK;
        config.num_blocks = num_sms;

        gfd::WarpSpecSession session(config);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            session.launch(noop_kernel, NoOp{});
            session.synchronize();
        }

        // Measured iterations
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            auto t0 = std::chrono::high_resolution_clock::now();
            session.launch(noop_kernel, NoOp{});
            session.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[iter] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double p50 = percentile(lats, 50);
        double p5 = percentile(lats, 5);
        double p95 = percentile(lats, 95);
        printf("  P5 = %.2f ms, P50 = %.2f ms, P95 = %.2f ms\n", p5, p50, p95);
        printf("  BW (P50) = %.2f GB/s\n", TOTAL_SIZE / (p50 * 1e6));
        all_results.push_back({"0. Pure GFD Transfer (no compute)", TOTAL_SIZE / (1024.0 * 1024.0), p5, p50, p95});
        print_poller_stats(*session.get_poller(), p50);
    }

    // ====================================================================
    // Test 1: Warp-Specialized (High-Level API)
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 1: Warp-Specialized (High-Level API)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        gfd::WarpSpecConfig config;
        config.total_tokens = TOTAL_TOKENS;
        config.token_size = TOKEN_SIZE;
        config.cpu_src = cpu_data;
        config.gpu_dst = gpu_data;
        config.tokens_per_tile = TOKENS_PER_TILE;
        config.tokens_per_chunk = TOKENS_PER_CHUNK;
        config.num_blocks = num_sms;

        gfd::WarpSpecSession session(config);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            session.launch(reduce_kernel, VectorReduce{d_output, TOKEN_DIM});
            session.synchronize();
        }

        // Measured iterations
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            auto t0 = std::chrono::high_resolution_clock::now();
            session.launch(reduce_kernel, VectorReduce{d_output, TOKEN_DIM});
            session.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[iter] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double p50 = percentile(lats, 50);
        double p5 = percentile(lats, 5);
        double p95 = percentile(lats, 95);
        printf("  P5 = %.2f ms, P50 = %.2f ms, P95 = %.2f ms\n", p5, p50, p95);
        printf("  BW (P50) = %.2f GB/s\n", TOTAL_SIZE / (p50 * 1e6));
        all_results.push_back({"1. GFD Warp-Spec (K=4)", TOTAL_SIZE / (1024.0 * 1024.0), p5, p50, p95});
        print_poller_stats(*session.get_poller(), p50);

        // Verify correctness
        float* h_output = new float[TOTAL_TOKENS];
        cudaMemcpy(h_output, d_output, TOTAL_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && errors < 5; t++) {
            if (!isfinite(h_output[t]) || h_output[t] == 0.0f) errors++;
        }
        printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
        delete[] h_output;
    }

    // ====================================================================
    // Test 2: Baseline (per-tile wait, no warp spec)
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 2: Baseline (per-tile wait, no warp spec)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        // Allocate TiledQueue manually for baseline
        gfd::TiledQueue* d_tq;
        cudaHostAlloc(&d_tq, sizeof(gfd::TiledQueue), cudaHostAllocMapped);
        memset(d_tq, 0, sizeof(gfd::TiledQueue));
        d_tq->scheduler.total_tiles = NUM_TILES;
        d_tq->scheduler.tokens_per_tile = TOKENS_PER_TILE;
        d_tq->scheduler.tokens_per_chunk = TOKENS_PER_CHUNK;
        d_tq->scheduler.chunks_per_tile = CHUNKS_PER_TILE;
        d_tq->scheduler.token_size = TOKEN_SIZE;

        gfd::StagingPool::instance().init(1, TOTAL_SIZE);
        gfd::CpuPollingThread poller(&d_tq->base, gpu_data, cpu_data, TOTAL_SIZE,
                                      true, 0);
        poller.set_tiled_queue(d_tq);
        poller.init_copy_engine();

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            poller.stop();
            memset((void*)d_tq->tile_chunk_done, 0, sizeof(d_tq->tile_chunk_done));
            d_tq->scheduler.next_tile = 0;
            poller.reset_stats();
            cudaMemset(gpu_data, 0, TOTAL_SIZE);

            baseline_kernel<<<NUM_TILES, TOKENS_PER_TILE>>>(
                d_tq, cpu_data, gpu_data, d_output,
                TOKENS_PER_TILE, TOKEN_SIZE, TOKEN_DIM);
            poller.start();
            cudaDeviceSynchronize();
        }

        // Measured iterations
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            poller.stop();
            memset((void*)d_tq->tile_chunk_done, 0, sizeof(d_tq->tile_chunk_done));
            d_tq->scheduler.next_tile = 0;
            poller.reset_stats();
            cudaMemset(gpu_data, 0, TOTAL_SIZE);

            auto t0 = std::chrono::high_resolution_clock::now();
            baseline_kernel<<<NUM_TILES, TOKENS_PER_TILE>>>(
                d_tq, cpu_data, gpu_data, d_output,
                TOKENS_PER_TILE, TOKEN_SIZE, TOKEN_DIM);
            poller.start();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[iter] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        poller.stop();
        double p50 = percentile(lats, 50);
        double p5 = percentile(lats, 5);
        double p95 = percentile(lats, 95);
        printf("  P5 = %.2f ms, P50 = %.2f ms, P95 = %.2f ms\n", p5, p50, p95);
        printf("  BW (P50) = %.2f GB/s\n", TOTAL_SIZE / (p50 * 1e6));
        all_results.push_back({"2. Baseline (no overlap)", TOTAL_SIZE / (1024.0 * 1024.0), p5, p50, p95});

        // Verify
        float* h_output = new float[TOTAL_TOKENS];
        cudaMemcpy(h_output, d_output, TOTAL_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && errors < 5; t++) {
            if (!isfinite(h_output[t]) || h_output[t] == 0.0f) errors++;
        }
        printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
        delete[] h_output;

        cudaFreeHost(d_tq);
    }

    // ====================================================================
    // Test 3: Warp-Specialized Double-Buffer (Ping-Pong)
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 3: Warp-Specialized Double-Buffer (Ping-Pong)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        gfd::WarpSpecConfig config;
        config.total_tokens = TOTAL_TOKENS;
        config.token_size = TOKEN_SIZE;
        config.cpu_src = cpu_data;
        config.gpu_dst = gpu_data;
        config.tokens_per_tile = TOKENS_PER_TILE;
        config.tokens_per_chunk = TOKENS_PER_CHUNK;
        config.num_blocks = num_sms;
        config.double_buffer = true;

        printf("  Block config (DB): %d warps (%d threads)\n",
               1 + 2 * CHUNKS_PER_TILE, config.block_size());

        gfd::WarpSpecSession session(config);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            session.launch(reduce_kernel_db, VectorReduce{d_output, TOKEN_DIM});
            session.synchronize();
        }

        // Measured iterations
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            auto t0 = std::chrono::high_resolution_clock::now();
            session.launch(reduce_kernel_db, VectorReduce{d_output, TOKEN_DIM});
            session.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[iter] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double p50 = percentile(lats, 50);
        double p5 = percentile(lats, 5);
        double p95 = percentile(lats, 95);
        printf("  P5 = %.2f ms, P50 = %.2f ms, P95 = %.2f ms\n", p5, p50, p95);
        printf("  BW (P50) = %.2f GB/s\n", TOTAL_SIZE / (p50 * 1e6));
        all_results.push_back({"3. GFD Double-Buffer", TOTAL_SIZE / (1024.0 * 1024.0), p5, p50, p95});

        // Verify correctness
        float* h_output = new float[TOTAL_TOKENS];
        cudaMemcpy(h_output, d_output, TOTAL_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && errors < 5; t++) {
            if (!isfinite(h_output[t]) || h_output[t] == 0.0f) errors++;
        }
        printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
        delete[] h_output;
    }

    // ====================================================================
    // Test 4: Auto-Tuned Chunk Size
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 4: Auto-Tuned Chunk Size\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        gfd::WarpSpecConfig config;
        config.total_tokens = TOTAL_TOKENS;
        config.token_size = TOKEN_SIZE;
        config.cpu_src = cpu_data;
        config.gpu_dst = gpu_data;
        config.tokens_per_tile = TOKENS_PER_TILE;
        config.tokens_per_chunk = 0;  // auto-tune!
        config.num_blocks = num_sms;

        printf("  Auto-tuned tokens_per_chunk = %u (token_size=%zu)\n",
               config.effective_tokens_per_chunk(), TOKEN_SIZE);
        printf("  Block config: %d warps (%d threads)\n",
               1 + (int)config.chunks_per_tile(), config.block_size());

        gfd::WarpSpecSession session(config);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            session.launch(reduce_kernel, VectorReduce{d_output, TOKEN_DIM});
            session.synchronize();
        }

        // Measured iterations
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            cudaMemset(gpu_data, 0, TOTAL_SIZE);
            auto t0 = std::chrono::high_resolution_clock::now();
            session.launch(reduce_kernel, VectorReduce{d_output, TOKEN_DIM});
            session.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[iter] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double p50 = percentile(lats, 50);
        double p5 = percentile(lats, 5);
        double p95 = percentile(lats, 95);
        printf("  P5 = %.2f ms, P50 = %.2f ms, P95 = %.2f ms\n", p5, p50, p95);
        printf("  BW (P50) = %.2f GB/s\n", TOTAL_SIZE / (p50 * 1e6));
        all_results.push_back({"4. GFD Auto-Tuned", TOTAL_SIZE / (1024.0 * 1024.0), p5, p50, p95});

        // Verify
        float* h_output = new float[TOTAL_TOKENS];
        cudaMemcpy(h_output, d_output, TOTAL_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && errors < 5; t++) {
            if (!isfinite(h_output[t]) || h_output[t] == 0.0f) errors++;
        }
        printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
        delete[] h_output;
    }

    // ====================================================================
    // Test 5: Multi-Tile Warp-Spec (more tiles → more overlap opportunity)
    // Note: total entries must be < QUEUE_SIZE (16384) since no backpressure.
    // 12288 tokens / 128 per tile = 96 tiles → 0.87 tiles/SM on 110 SMs.
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 5: Multi-Tile Warp-Spec (12288 tokens, 192 MB)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        constexpr int MT_TOTAL_TOKENS = 12288;
        constexpr size_t MT_TOTAL_SIZE = (size_t)MT_TOTAL_TOKENS * TOKEN_SIZE;

        float* mt_cpu_data;
        cudaMallocHost(&mt_cpu_data, MT_TOTAL_SIZE);
        for (int t = 0; t < MT_TOTAL_TOKENS; t++) {
            for (int d = 0; d < TOKEN_DIM; d++) {
                mt_cpu_data[t * TOKEN_DIM + d] = (float)(t * 1000 + d);
            }
        }

        float* mt_gpu_data;
        cudaMalloc(&mt_gpu_data, MT_TOTAL_SIZE);

        float* mt_output;
        cudaMalloc(&mt_output, MT_TOTAL_TOKENS * sizeof(float));

        printf("  Tokens: %d, Tiles: %d, Total: %.2f MB\n",
               MT_TOTAL_TOKENS, MT_TOTAL_TOKENS / TOKENS_PER_TILE,
               MT_TOTAL_SIZE / (1024.0 * 1024.0));

        gfd::WarpSpecConfig config;
        config.total_tokens = MT_TOTAL_TOKENS;
        config.token_size = TOKEN_SIZE;
        config.cpu_src = mt_cpu_data;
        config.gpu_dst = mt_gpu_data;
        config.tokens_per_tile = TOKENS_PER_TILE;
        config.tokens_per_chunk = TOKENS_PER_CHUNK;
        config.num_blocks = num_sms;

        gfd::WarpSpecSession session(config);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            cudaMemset(mt_gpu_data, 0, MT_TOTAL_SIZE);
            session.launch(reduce_kernel, VectorReduce{mt_output, TOKEN_DIM});
            session.synchronize();
        }

        // Measured iterations
        std::vector<double> lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) {
            cudaMemset(mt_gpu_data, 0, MT_TOTAL_SIZE);
            auto t0 = std::chrono::high_resolution_clock::now();
            session.launch(reduce_kernel, VectorReduce{mt_output, TOKEN_DIM});
            session.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[iter] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double p50 = percentile(lats, 50);
        double p5 = percentile(lats, 5);
        double p95 = percentile(lats, 95);
        printf("  P5 = %.2f ms, P50 = %.2f ms, P95 = %.2f ms\n", p5, p50, p95);
        printf("  BW (P50) = %.2f GB/s\n", MT_TOTAL_SIZE / (p50 * 1e6));
        printf("  Tiles/SM = %.1f (overlap opportunity)\n",
               (float)(MT_TOTAL_TOKENS / TOKENS_PER_TILE) / num_sms);
        all_results.push_back({"5. GFD Multi-Tile (192MB)", MT_TOTAL_SIZE / (1024.0 * 1024.0), p5, p50, p95});

        // Verify
        float* h_output = new float[MT_TOTAL_TOKENS];
        cudaMemcpy(h_output, mt_output, MT_TOTAL_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < MT_TOTAL_TOKENS && errors < 5; t++) {
            if (!isfinite(h_output[t]) || h_output[t] == 0.0f) errors++;
        }
        printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
        delete[] h_output;

        cudaFree(mt_output);
        cudaFree(mt_gpu_data);
        cudaFreeHost(mt_cpu_data);
    }

    // ---- Summary Table ----
    printf("\n");
    print_summary_table(all_results);

    // Cleanup
    cudaFree(d_output);
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);

    printf("\nDone.\n");
    return 0;
}
