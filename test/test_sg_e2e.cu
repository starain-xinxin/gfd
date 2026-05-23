#include <gfd/gfd.h>
#include <gfd/sg_warp_spec.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

// Minimal compute: just verify data arrived
struct NullCompute {
    __device__ void operator()(gfd::sg_warp_spec::SGListView list) {
        // No-op: just let the framework run
        if (list.lane_id == 0) {
            printf("GPU compute: list_id=%u count=%u\n", list.list_id, list.count);
        }
    }
};

GFD_SG_WARP_SPEC_KERNEL(sg_test_kernel, NullCompute);

int main() {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    fprintf(stderr, "=== SG E2E Test ===\n");

    cuInit(0);
    CUcontext ctx; CUdevice dev;
    cuDeviceGet(&dev, 0);
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxParams = {};
    cuCtxCreate(&ctx, &ctxParams, 0, dev);
#else
    cuCtxCreate(&ctx, 0, dev);
#endif

    constexpr int N = 4;
    constexpr size_t TOKEN_SIZE = 4096;
    constexpr size_t TOTAL = N * TOKEN_SIZE;

    float* cpu_data;
    cudaMallocHost(&cpu_data, TOTAL);
    for (size_t i = 0; i < TOTAL / sizeof(float); i++) cpu_data[i] = (float)(i + 1);

    float* gpu_data;
    cudaMalloc(&gpu_data, TOTAL);
    cudaMemset(gpu_data, 0, TOTAL);

    fprintf(stderr, "Memory allocated. cpu=%p gpu=%p\n", cpu_data, gpu_data);

    // Create session
    gfd::SGWarpSpecConfig config;
    config.num_compute_warps = 1;
    config.num_blocks = 1;
    config.use_copy_engine = true;

    fprintf(stderr, "Creating session...\n");
    gfd::SGWarpSpecSession session(config);
    fprintf(stderr, "Session created.\n");

    // Check queue state
    auto* sq = session.get_sg_queue();
    auto* dq = session.get_desc_queue();
    fprintf(stderr, "sq=%p dq=%p\n", sq, dq);
    fprintf(stderr, "sq->d_list_done=%p\n", sq->d_list_done);

    // Submit entries
    gfd::DeviceSGEntry entries[N];
    for (int i = 0; i < N; i++) {
        entries[i].src_addr = (uint64_t)cpu_data + (uint64_t)i * TOKEN_SIZE;
        entries[i].dst_addr = (uint64_t)gpu_data + (uint64_t)i * TOKEN_SIZE;
        entries[i].size = TOKEN_SIZE;
        entries[i].tag = 0;
    }

    fprintf(stderr, "Submitting SG list: %d entries...\n", N);
    session.submit_sg_list(entries, N, 0, gfd::SG_FLAG_HOST_SUBMITTED);

    fprintf(stderr, "After submit: list_alloc=%lu seq=%lu\n",
            (unsigned long)sq->list_alloc_idx,
            (unsigned long)sq->lists[0].sequence);

    // Launch
    fprintf(stderr, "Launching kernel (grid=%d, block=%d)...\n",
            1, config.block_size());
    session.launch(sg_test_kernel, NullCompute{});
    fprintf(stderr, "Kernel launched, poller started.\n");

    // Monitor progress
    for (int i = 0; i < 50; i++) {
        struct timespec ts = {0, 100000000}; // 100ms
        nanosleep(&ts, nullptr);

        uint64_t w = dq->write_idx;
        uint64_t r = dq->read_idx;
        uint64_t lr = sq->list_read_idx;
        uint64_t lc = sq->lists_completed;
        fprintf(stderr, "  [%d] dq: w=%lu r=%lu | sq: list_read=%lu completed=%lu\n",
                i, (unsigned long)w, (unsigned long)r,
                (unsigned long)lr, (unsigned long)lc);

        // If transfer warp has processed the list and poller has completed
        if (lr >= 1 && lc >= 1) {
            fprintf(stderr, "  All done!\n");
            break;
        }
        if (lr >= 1 && w > 0 && r == 0 && i > 10) {
            fprintf(stderr, "  STUCK: descriptors written but not read by poller!\n");
            // Dump first descriptor
            fprintf(stderr, "  desc[0] seq=%lu src=%lx dst=%lx sz=%u fl=%u\n",
                    (unsigned long)dq->entries[0].sequence,
                    (unsigned long)dq->entries[0].src_addr,
                    (unsigned long)dq->entries[0].dst_addr,
                    dq->entries[0].size, dq->entries[0].flags);
            break;
        }
    }

    fprintf(stderr, "Calling synchronize...\n");
    session.synchronize();
    fprintf(stderr, "Synchronized.\n");

    auto stats = session.get_stats();
    fprintf(stderr, "Stats: desc=%lu bytes=%lu elapsed=%.2f ms\n",
            (unsigned long)stats.descriptors_processed,
            (unsigned long)stats.bytes_transferred,
            stats.elapsed_ms);

    // Verify data
    float* h_gpu = new float[TOTAL / sizeof(float)];
    cudaMemcpy(h_gpu, gpu_data, TOTAL, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (size_t i = 0; i < TOTAL / sizeof(float) && errors < 5; i++) {
        if (fabsf(cpu_data[i] - h_gpu[i]) > 1e-5f) {
            if (errors < 3) fprintf(stderr, "  Mismatch at %zu: expected %.1f got %.1f\n",
                                     i, cpu_data[i], h_gpu[i]);
            errors++;
        }
    }
    fprintf(stderr, "Data correctness: %s (%d errors)\n",
            errors == 0 ? "PASS" : "FAIL", errors);

    delete[] h_gpu;
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);
    return 0;
}
