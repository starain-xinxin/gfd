// ============================================================
// GFD GPU-Planned Direct Transfer (Parallel)
//
// Scenario: GPU produces a scatter-gather list in parallel
// (simulating MoE expert routing), then CPU executes the
// transfer via GFD Direct in a single call.
//
// Flow:
//   1. GPU kernel computes token-to-expert routing (parallel)
//   2. GPU uses atomicAdd for stream compaction, directly
//      writes SGEntry array into pinned host memory
//   3. cudaDeviceSynchronize() ensures SG list is complete
//   4. CPU calls submit_direct() with the GPU-produced SG list
//   5. GPU verifies the transferred data
//
// Key optimization:
//   - Parallel stream compaction replaces single-thread scan
//   - GPU produces complete SGEntry[] directly (no CPU build step)
//   - Eliminates ~590us serial kernel + ~9us CPU build overhead
// ============================================================

#include <gfd/gfd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>

// ---- GPU kernel: parallel stream compaction (phase 1) ----

__global__ void plan_compact_kernel(
    int* token_ids,
    int* counter,
    int* routing_table,
    int num_tokens,
    int target_expert)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens) return;

    if (routing_table[tid] == target_expert) {
        int slot = atomicAdd(counter, 1);
        token_ids[slot] = tid;
    }
}

// ---- GPU kernel: parallel SG entry construction (phase 2) ----

__global__ void plan_build_sg_kernel(
    gfd::SGEntry* entries,
    int* token_ids,
    int num_entries,
    char* gpu_buf,
    char* cpu_buf,
    int token_size,
    int scatter_stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    int tok_id = token_ids[tid];
    entries[tid].dst  = (CUdeviceptr)(gpu_buf + (size_t)tid * token_size);
    entries[tid].src  = cpu_buf + (size_t)tok_id * token_size * scatter_stride;
    entries[tid].size = token_size;
}

// ---- GPU kernel: verify transferred data ----

__global__ void verify_kernel(
    char* gpu_buf,
    int* token_ids,
    int num_entries,
    uint32_t token_size,
    int* result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    char* token = gpu_buf + (size_t)tid * token_size;
    uint8_t expected = (uint8_t)(token_ids[tid] & 0xFF);

    for (uint32_t j = 0; j < token_size; j++) {
        if ((uint8_t)token[j] != expected) {
            atomicExch(result, 1);
            return;
        }
    }
}

int main() {
    // ---- Configuration ----
    constexpr int TOTAL_TOKENS = 4096;
    constexpr int NUM_EXPERTS = 8;
    constexpr int TOKEN_SIZE = 4096;       // 4KB per token
    constexpr int TARGET_EXPERT = 3;
    constexpr int SCATTER_STRIDE = 2;
    constexpr int EXPECTED_TOKENS = TOTAL_TOKENS / NUM_EXPERTS;  // 512
    constexpr int BLOCK_SIZE = 256;

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
    printf("Scenario: %d total tokens, %d experts, prefetch expert %d\n",
           TOTAL_TOKENS, NUM_EXPERTS, TARGET_EXPERT);
    printf("Token size: %d bytes, scattered at %dx stride\n", TOKEN_SIZE, SCATTER_STRIDE);

    // ---- Allocate memory ----

    // CPU pinned memory with scattered token layout
    size_t cpu_buf_size = (size_t)TOTAL_TOKENS * TOKEN_SIZE * SCATTER_STRIDE;
    char* cpu_buf;
    cudaMallocHost(&cpu_buf, cpu_buf_size);
    for (int i = 0; i < TOTAL_TOKENS; i++) {
        char* ptr = cpu_buf + (size_t)i * TOKEN_SIZE * SCATTER_STRIDE;
        memset(ptr, (uint8_t)(i & 0xFF), TOKEN_SIZE);
    }

    // GPU destination buffer
    size_t gpu_buf_size = (size_t)EXPECTED_TOKENS * TOKEN_SIZE * 2;
    char* gpu_buf;
    cudaMalloc(&gpu_buf, gpu_buf_size);

    // SG entries (pinned host memory, GPU writes via zero-copy, CPU reads)
    gfd::SGEntry* sg_entries;
    cudaHostAlloc(&sg_entries, EXPECTED_TOKENS * 2 * sizeof(gfd::SGEntry), cudaHostAllocMapped);

    // Atomic counter for stream compaction (device memory for speed)
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));

    // Token IDs buffer (device memory for fast GPU access)
    int* d_token_ids;
    cudaMalloc(&d_token_ids, EXPECTED_TOKENS * 2 * sizeof(int));

    // Routing table (device memory for fast GPU access)
    int* h_routing = new int[TOTAL_TOKENS];
    for (int i = 0; i < TOTAL_TOKENS; i++) {
        h_routing[i] = i % NUM_EXPERTS;
    }
    int* d_routing;
    cudaMalloc(&d_routing, TOTAL_TOKENS * sizeof(int));
    cudaMemcpy(d_routing, h_routing, TOTAL_TOKENS * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_routing;

    // Verification result
    int* d_verify;
    cudaHostAlloc(&d_verify, sizeof(int), cudaHostAllocMapped);
    *d_verify = 0;

    // Descriptor queue (needed by CpuPollingThread)
    gfd::DescriptorQueue* queue;
    cudaHostAlloc(&queue, sizeof(gfd::DescriptorQueue), cudaHostAllocMapped);
    memset(queue, 0, sizeof(gfd::DescriptorQueue));

    // ---- Initialize GFD ----

    gfd::StagingPool::instance().init(1, gpu_buf_size);

    gfd::CpuPollingThread poller(queue, gpu_buf, cpu_buf, gpu_buf_size,
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

    printf("Expected tokens for expert %d: %d (%zu KB)\n\n",
           TARGET_EXPERT, EXPECTED_TOKENS, (size_t)EXPECTED_TOKENS * TOKEN_SIZE / 1024);

    int grid_compact = (TOTAL_TOKENS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---- Warmup ----
    printf("Warming up...\n");

    for (int w = 0; w < 10; w++) {
        cudaMemset(gpu_buf, 0, gpu_buf_size);
        cudaMemset(d_counter, 0, sizeof(int));

        // Phase 1: Parallel compact token IDs (device memory)
        plan_compact_kernel<<<grid_compact, BLOCK_SIZE>>>(
            d_token_ids, d_counter, d_routing, TOTAL_TOKENS, TARGET_EXPERT);

        // Read back counter
        int n;
        cudaMemcpy(&n, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

        // Phase 2: Parallel build SG entries (writes to pinned host memory)
        int grid_sg = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        plan_build_sg_kernel<<<grid_sg, BLOCK_SIZE>>>(
            sg_entries, d_token_ids, n, gpu_buf, cpu_buf, TOKEN_SIZE, SCATTER_STRIDE);
        cudaDeviceSynchronize();

        // CPU executes the GPU-produced SG list
        poller.submit_direct(sg_entries, n);
    }

    // ---- Timed run ----
    printf("Running timed iteration...\n\n");

    cudaMemset(gpu_buf, 0, gpu_buf_size);
    cudaMemset(d_counter, 0, sizeof(int));
    *d_verify = 0;
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: GPU parallel compact (stream compaction with atomicAdd)
    plan_compact_kernel<<<grid_compact, BLOCK_SIZE>>>(
        d_token_ids, d_counter, d_routing, TOTAL_TOKENS, TARGET_EXPERT);

    // Read back counter (implicit sync)
    int num_entries;
    cudaMemcpy(&num_entries, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    auto t_compact = std::chrono::high_resolution_clock::now();

    // Step 2: GPU parallel build SG entries (writes to pinned host memory)
    int grid_sg = (num_entries + BLOCK_SIZE - 1) / BLOCK_SIZE;
    plan_build_sg_kernel<<<grid_sg, BLOCK_SIZE>>>(
        sg_entries, d_token_ids, num_entries, gpu_buf, cpu_buf, TOKEN_SIZE, SCATTER_STRIDE);
    cudaDeviceSynchronize();

    auto t_build = std::chrono::high_resolution_clock::now();

    // Step 3: CPU executes GFD Direct transfer (no CPU build needed!)
    double direct_us = poller.submit_direct(sg_entries, num_entries);

    auto t1 = std::chrono::high_resolution_clock::now();

    // ---- Verify on GPU ----
    // d_token_ids is already in device memory, use directly
    int vblocks = (num_entries + 255) / 256;
    verify_kernel<<<vblocks, 256>>>(gpu_buf, d_token_ids, num_entries, TOKEN_SIZE, d_verify);
    cudaDeviceSynchronize();

    // ---- Results ----
    double compact_us = std::chrono::duration<double, std::micro>(t_compact - t0).count();
    double build_us = std::chrono::duration<double, std::micro>(t_build - t_compact).count();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    size_t total_bytes = (size_t)num_entries * TOKEN_SIZE;
    double bw_gbs = total_bytes / (direct_us * 1e3);

    printf("Results:\n");
    printf("  Tokens routed to expert %d: %d\n", TARGET_EXPERT, num_entries);
    printf("  Data transferred: %zu KB\n", total_bytes / 1024);
    printf("\n");
    printf("  Timing breakdown:\n");
    printf("    GPU compact (parallel):  %7.1f us\n", compact_us);
    printf("    GPU build SG (parallel): %7.1f us\n", build_us);
    printf("    GFD Direct xfer:         %7.1f us\n", direct_us);
    printf("    ──────────────────────────────────\n");
    printf("    Total:                   %7.1f us\n", total_us);
    printf("\n");
    printf("  Bandwidth:  %.2f GB/s\n", bw_gbs);
    int verify_result = *d_verify;
    printf("  Verify:     %s\n", (verify_result == 0) ? "PASSED" : "FAILED");

    // ---- Cleanup ----
    poller.stop();
    gfd::StagingPool::instance().shutdown();
    cudaFreeHost(d_verify);
    cudaFree(d_token_ids);
    cudaFree(d_counter);
    cudaFreeHost(sg_entries);
    cudaFreeHost(queue);
    cudaFree(d_routing);
    cudaFree(gpu_buf);
    cudaFreeHost(cpu_buf);

    return (verify_result == 0) ? 0 : 1;
}
