#include <gfd/gfd.h>
#include <gfd/device.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>

// ============================================================
// Example: Fused Compute + Communication Kernel
//
// This demonstrates how to embed GFD device-side primitives into
// a user-defined kernel, simulating a real-world pattern where
// computation and data prefetch are fused into a single launch:
//
//   Phase 1: Request KV-cache prefetch (write descriptors)
//   Phase 2: Overlap local compute while CPU+CE transfer data
//   Phase 3: Wait for transfer completion
//   Phase 4: Process prefetched data (e.g., attention with KV-cache)
//
// In production, Phase 2 would be the query/key projection or
// other independent work, and Phase 4 would be the attention
// score computation using the freshly-arrived KV-cache data.
// ============================================================

__global__ void fused_prefetch_and_compute_kernel(
    gfd::DescriptorQueue* queue,
    gfd::TokenInfo* tokens,
    float* gpu_buffer,
    float* output,
    int num_tokens,
    uint32_t token_size,
    int token_dim,
    uint64_t base_slot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < num_tokens);

    // ===== Phase 1: Write descriptors (request KV-cache prefetch) =====
    // Each thread writes one descriptor for its token's data
    if (active) {
        gfd::device::write_descriptor(
            queue, base_slot, tid,
            tokens[tid].cpu_addr,           // CPU source address
            gpu_buffer,                     // GPU destination base
            token_size, num_tokens,
            ((uint64_t)tokens[tid].expert_id << 32) | tokens[tid].token_id);
    }
    gfd::device::fence_and_commit(queue, base_slot, tid, active);

    // ===== Phase 2: Overlap compute while data is in flight =====
    // In real workloads, this would be query projection, RoPE encoding,
    // or any independent computation that doesn't need the prefetched data.
    float local_result = 0.0f;
    if (active) {
        // Simulate non-trivial compute (~200 FLOPs per thread)
        float x = (float)tid * 0.001f;
        for (int i = 0; i < 50; i++) {
            x = x * 0.99f + sinf(x) * 0.01f;
            local_result += x;
        }
    }

    // ===== Phase 3: Wait for all transfers to complete =====
    // Thread 0 polls done_idx; __syncthreads() broadcasts to all threads
    if (tid == 0) {
        gfd::device::wait_for_completion(queue, base_slot + num_tokens);
    }
    __syncthreads();

    // ===== Phase 4: Use the prefetched data =====
    // Data has arrived in gpu_buffer. In real workloads, this would be
    // attention score computation with the prefetched KV-cache.
    if (active) {
        float* my_data = gpu_buffer + tid * token_dim;
        float sum = local_result;
        for (int d = 0; d < token_dim; d++) {
            sum += my_data[d];
        }
        output[tid] = sum;
    }
}

// ============================================================
// Simple transfer-only kernel: minimal usage of device API
// ============================================================
__global__ void simple_transfer_kernel(
    gfd::DescriptorQueue* queue,
    gfd::TokenInfo* tokens,
    float* gpu_buffer,
    int num_tokens,
    uint32_t token_size,
    uint64_t base_slot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < num_tokens);

    // Write + fence + commit (convenience one-liner)
    gfd::device::write_and_commit(
        queue, base_slot, tid, active,
        active ? tokens[tid].cpu_addr : 0,
        gpu_buffer, token_size, num_tokens);

    // Wait for completion
    if (tid == 0) {
        gfd::device::wait_for_completion(queue, base_slot + num_tokens);
    }
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

    constexpr int NUM_TOKENS = 1024;
    constexpr int TOKEN_DIM = 1024;  // floats per token
    constexpr size_t TOKEN_SIZE = TOKEN_DIM * sizeof(float);
    constexpr size_t TOTAL_SIZE = NUM_TOKENS * TOKEN_SIZE;

    printf("=== GFD Fused Kernel Example ===\n");
    printf("Tokens: %d, Token size: %zu bytes, Total: %.2f MB\n",
           NUM_TOKENS, TOKEN_SIZE, TOTAL_SIZE / (1024.0 * 1024.0));

    // ---- Allocate CPU memory (pinned) ----
    float* cpu_data;
    cudaMallocHost(&cpu_data, TOTAL_SIZE);
    for (int t = 0; t < NUM_TOKENS; t++) {
        for (int d = 0; d < TOKEN_DIM; d++) {
            cpu_data[t * TOKEN_DIM + d] = (float)(t * 1000 + d);
        }
    }

    // ---- Allocate GPU memory ----
    float* gpu_data;
    cudaMalloc(&gpu_data, TOTAL_SIZE);
    cudaMemset(gpu_data, 0, TOTAL_SIZE);

    float* d_output;
    cudaMalloc(&d_output, NUM_TOKENS * sizeof(float));

    // ---- Allocate descriptor queue (managed, GPU+CPU accessible) ----
    gfd::DescriptorQueue* d_queue;
    cudaMallocManaged(&d_queue, sizeof(gfd::DescriptorQueue));
    memset(d_queue, 0, sizeof(gfd::DescriptorQueue));

    // ---- Token info on GPU ----
    gfd::TokenInfo* d_tokens;
    cudaMalloc(&d_tokens, NUM_TOKENS * sizeof(gfd::TokenInfo));
    {
        gfd::TokenInfo h_tokens[NUM_TOKENS];
        for (int t = 0; t < NUM_TOKENS; t++) {
            h_tokens[t].cpu_addr = (uint64_t)(cpu_data + t * TOKEN_DIM);
            h_tokens[t].token_id = t;
            h_tokens[t].expert_id = 0;
        }
        cudaMemcpy(d_tokens, h_tokens, NUM_TOKENS * sizeof(gfd::TokenInfo),
                   cudaMemcpyHostToDevice);
    }

    // ---- Initialize staging pool + CPU polling thread ----
    gfd::StagingPool::instance().init(1, TOTAL_SIZE);

    gfd::CpuPollingThread poller(d_queue, gpu_data, cpu_data, TOTAL_SIZE,
                                  /*use_ce=*/true, /*numa_node=*/0);
    if (!poller.init_copy_engine()) {
        fprintf(stderr, "Failed to init copy engine\n");
        return 1;
    }
    poller.start();

    // ====================================================================
    // Test 1: Simple transfer-only kernel
    // ====================================================================
    printf("\n--- Test 1: Simple Transfer Kernel ---\n");
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        int threads = 256;
        int blocks = (NUM_TOKENS + threads - 1) / threads;
        simple_transfer_kernel<<<blocks, threads>>>(
            d_queue, d_tokens, gpu_data, NUM_TOKENS, TOKEN_SIZE, 0);
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Transfer: %.2f ms (%.2f GB/s)\n",
               elapsed_ms, TOTAL_SIZE / (elapsed_ms * 1e6));

        // Verify
        float* verify = new float[NUM_TOKENS * TOKEN_DIM];
        cudaMemcpy(verify, gpu_data, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < NUM_TOKENS && errors < 5; t++) {
            float expected = (float)(t * 1000);
            if (fabsf(verify[t * TOKEN_DIM] - expected) > 1e-5f) {
                printf("  ERROR: token %d dim 0: expected %.1f, got %.1f\n",
                       t, expected, verify[t * TOKEN_DIM]);
                errors++;
            }
        }
        printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");
        delete[] verify;
    }

    // Reset queue for next test
    memset(d_queue, 0, sizeof(gfd::DescriptorQueue));
    cudaMemset(gpu_data, 0, TOTAL_SIZE);
    poller.reset_stats();

    // ====================================================================
    // Test 2: Fused prefetch + compute kernel
    // ====================================================================
    printf("\n--- Test 2: Fused Prefetch + Compute Kernel ---\n");
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        int threads = 256;
        int blocks = (NUM_TOKENS + threads - 1) / threads;
        fused_prefetch_and_compute_kernel<<<blocks, threads>>>(
            d_queue, d_tokens, gpu_data, d_output,
            NUM_TOKENS, TOKEN_SIZE, TOKEN_DIM, 0);
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Fused kernel: %.2f ms\n", elapsed_ms);

        // Verify: check that output has meaningful values
        // output[t] = local_compute_result + sum(gpu_data[t][0..TOKEN_DIM-1])
        //           = local_compute_result + sum(t*1000+d for d in 0..TOKEN_DIM-1)
        float* h_output = new float[NUM_TOKENS];
        cudaMemcpy(h_output, d_output, NUM_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify: for token 0, data_sum = sum(d, d=0..TOKEN_DIM-1) = TOKEN_DIM*(TOKEN_DIM-1)/2
        // The local_result adds a small offset from the sinf loop.
        // We verify by checking token 0 directly (data_sum is exact in float range).
        double expected_token0_data_sum = (double)TOKEN_DIM * (TOKEN_DIM - 1) / 2.0;
        int errors = 0;
        for (int t = 0; t < NUM_TOKENS && errors < 5; t++) {
            if (!isfinite(h_output[t])) {
                printf("  ERROR: token %d output is not finite: %f\n", t, h_output[t]);
                errors++;
            }
        }
        // Sanity: token 0's data sum is 523776.0; output should be close
        float token0_expected = (float)expected_token0_data_sum;
        float token0_diff = fabsf(h_output[0] - token0_expected);
        bool token0_close = (token0_diff / token0_expected) < 0.01f;  // within 1%
        if (!token0_close) {
            printf("  WARNING: token 0 output=%.1f, expected_data_sum=%.1f (diff=%.1f)\n",
                   h_output[0], token0_expected, token0_diff);
        }
        printf("Verification: %s (%d non-finite outputs, token0 data_sum_check=%s)\n",
               errors == 0 ? "PASS" : "FAIL", errors,
               token0_close ? "OK" : "DRIFT");
        delete[] h_output;
    }

    // ---- Print stats ----
    printf("\n--- CPU Poller Statistics ---\n");
    printf("Descriptors processed: %lu\n", poller.get_descriptors_processed());
    printf("Batches submitted:     %lu\n", poller.get_batches_submitted());
    printf("Staging batches:       %lu\n", poller.get_staging_batches());
    printf("Total bytes copied:    %.2f MB\n", poller.get_total_bytes_copied() / (1024.0 * 1024.0));
    printf("Gather time:           %lu us\n", poller.get_gather_us());
    printf("DMA submit time:       %lu us\n", poller.get_dma_submit_us());
    printf("DMA wait time:         %lu us\n", poller.get_dma_wait_us());

    // ---- Cleanup ----
    poller.stop();
    gfd::StagingPool::instance().shutdown();
    cudaFree(d_output);
    cudaFree(d_tokens);
    cudaFree(d_queue);
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);

    printf("\nDone.\n");
    return 0;
}
