// ============================================================
// GFD Direct Mode - Minimal Example
//
// Demonstrates the simplest usage of GFD Direct submit:
//   1. Allocate scattered CPU tokens + contiguous GPU buffer
//   2. Initialize CpuPollingThread with direct CE
//   3. Build scatter-gather list
//   4. Call submit_direct() - single function call does everything
//   5. Verify correctness
// ============================================================

#include <gfd/gfd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>

int main() {
    // Configuration
    constexpr int NUM_TOKENS = 1024;
    constexpr int TOKEN_SIZE = 4096;  // 4KB per token
    constexpr size_t TOTAL_SIZE = (size_t)NUM_TOKENS * TOKEN_SIZE;  // 4MB
    constexpr int SCATTER_STRIDE = 2;  // Tokens at 2x stride (simulates fragmentation)

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
    printf("GPU: %s\n", gpu_name);
    printf("Transfer: %d tokens x %d bytes = %zu MB (scattered at %dx stride)\n",
           NUM_TOKENS, TOKEN_SIZE, TOTAL_SIZE / (1024 * 1024), SCATTER_STRIDE);

    // ---- Allocate memory ----

    // CPU: pinned memory with scattered layout
    size_t cpu_buf_size = TOTAL_SIZE * SCATTER_STRIDE;
    char* cpu_buf;
    cudaMallocHost(&cpu_buf, cpu_buf_size);

    // Fill each token with a unique pattern for verification
    for (int i = 0; i < NUM_TOKENS; i++) {
        char* token_ptr = cpu_buf + (size_t)i * TOKEN_SIZE * SCATTER_STRIDE;
        memset(token_ptr, (uint8_t)(i & 0xFF), TOKEN_SIZE);
    }

    // GPU: contiguous destination buffer
    char* gpu_buf;
    cudaMalloc(&gpu_buf, TOTAL_SIZE);
    cudaMemset(gpu_buf, 0, TOTAL_SIZE);

    // ---- Initialize GFD ----

    // Descriptor queue (needed for CpuPollingThread, but unused in direct mode)
    gfd::DescriptorQueue* queue;
    cudaMallocManaged(&queue, sizeof(gfd::DescriptorQueue));
    memset(queue, 0, sizeof(gfd::DescriptorQueue));

    // Staging pool (pre-allocates hugepage buffers)
    gfd::StagingPool::instance().init(1, TOTAL_SIZE);

    // CPU polling thread with direct CE
    gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, TOTAL_SIZE,
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

    // ---- Build scatter-gather list ----

    gfd::SGEntry entries[NUM_TOKENS];
    for (int i = 0; i < NUM_TOKENS; i++) {
        entries[i].dst  = (CUdeviceptr)(gpu_buf + (size_t)i * TOKEN_SIZE);
        entries[i].src  = cpu_buf + (size_t)i * TOKEN_SIZE * SCATTER_STRIDE;
        entries[i].size = TOKEN_SIZE;
    }

    // ---- Execute transfer ----

    // Warmup
    for (int i = 0; i < 10; i++) {
        poller.submit_direct(entries, NUM_TOKENS);
    }

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    double latency_us = poller.submit_direct(entries, NUM_TOKENS);
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    double bandwidth_gbs = TOTAL_SIZE / (latency_us * 1e3);
    printf("\nResult:\n");
    printf("  Latency:   %.1f us\n", latency_us);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbs);

    // ---- Verify correctness ----

    char* verify_buf = new char[TOTAL_SIZE];
    cudaMemcpy(verify_buf, gpu_buf, TOTAL_SIZE, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < NUM_TOKENS; i++) {
        uint8_t expected = (uint8_t)(i & 0xFF);
        for (int j = 0; j < TOKEN_SIZE; j++) {
            if ((uint8_t)verify_buf[(size_t)i * TOKEN_SIZE + j] != expected) {
                fprintf(stderr, "MISMATCH at token %d, byte %d: got 0x%02X, expected 0x%02X\n",
                        i, j, (uint8_t)verify_buf[(size_t)i * TOKEN_SIZE + j], expected);
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    printf("  Verify:    %s\n", correct ? "PASSED" : "FAILED");

    // ---- Cleanup ----

    delete[] verify_buf;
    poller.stop();
    gfd::StagingPool::instance().shutdown();
    cudaFree(gpu_buf);
    cudaFree(queue);
    cudaFreeHost(cpu_buf);

    return correct ? 0 : 1;
}
