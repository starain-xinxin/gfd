#include <gfd/gfd.h>
#include <gfd/warp_spec.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================
// Double-Buffer Deadlock Regression Test
//
// Tests the double-buffer ping-pong mode with small tile counts
// (1, 2, 3, 4, 5, 7) to verify the deadlock fix.
// Previously, acquire_tile was called by all 32 lanes in the
// transfer warp, causing warp divergence and __shfl_sync UB.
// ============================================================

struct SimpleReduce {
    float* output;
    int token_dim;

    __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
        for (uint32_t i = chunk.lane_id; i < chunk.size; i += 32) {
            float* token = chunk.data<float>(i);
            float sum = 0.0f;
            for (int d = 0; d < token_dim; d++) {
                sum += token[d];
            }
            output[chunk.global_idx(i)] = sum;
        }
    }
};

GFD_WARP_SPEC_KERNEL_DB(test_kernel_db, SimpleReduce);

static bool run_test(int total_tokens, int tokens_per_tile, int tokens_per_chunk,
                     int token_dim, int num_sms) {
    const int num_tiles = total_tokens / tokens_per_tile;
    const size_t token_size = token_dim * sizeof(float);
    const size_t total_size = (size_t)total_tokens * token_size;

    printf("  tiles=%d, tokens=%d, tile_size=%d, chunk_size=%d ... ",
           num_tiles, total_tokens, tokens_per_tile, tokens_per_chunk);
    fflush(stdout);

    // Allocate
    float* cpu_data;
    cudaMallocHost(&cpu_data, total_size);
    for (int t = 0; t < total_tokens; t++) {
        for (int d = 0; d < token_dim; d++) {
            cpu_data[t * token_dim + d] = (float)(t + 1);  // non-zero
        }
    }

    float* gpu_data;
    cudaMalloc(&gpu_data, total_size);
    cudaMemset(gpu_data, 0, total_size);

    float* d_output;
    cudaMalloc(&d_output, total_tokens * sizeof(float));
    cudaMemset(d_output, 0, total_tokens * sizeof(float));

    // Configure double-buffer session
    gfd::WarpSpecConfig config;
    config.total_tokens = total_tokens;
    config.token_size = token_size;
    config.cpu_src = cpu_data;
    config.gpu_dst = gpu_data;
    config.tokens_per_tile = tokens_per_tile;
    config.tokens_per_chunk = tokens_per_chunk;
    config.num_blocks = (num_sms < num_tiles) ? num_sms : num_tiles;
    config.double_buffer = true;

    gfd::WarpSpecSession session(config);

    // Launch and synchronize
    session.launch(test_kernel_db, SimpleReduce{d_output, token_dim});
    session.synchronize();

    // Verify
    float* h_output = new float[total_tokens];
    cudaMemcpy(h_output, d_output, total_tokens * sizeof(float),
               cudaMemcpyDeviceToHost);

    int errors = 0;
    float expected_sum = (float)token_dim;  // each element = (t+1), but we
                                            // just check non-zero and finite
    for (int t = 0; t < total_tokens; t++) {
        float expected = (float)(t + 1) * token_dim;
        if (!isfinite(h_output[t]) || fabsf(h_output[t] - expected) > 1.0f) {
            if (errors < 3) {
                printf("\n    ERROR at token %d: got %f, expected %f",
                       t, h_output[t], expected);
            }
            errors++;
        }
    }

    bool pass = (errors == 0);
    printf("%s", pass ? "PASS" : "FAIL");
    if (errors > 0) printf(" (%d errors)", errors);
    printf("\n");

    delete[] h_output;
    cudaFree(d_output);
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);

    return pass;
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

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("=== Double-Buffer Deadlock Regression Test ===\n");
    printf("GPU: %s, SMs: %d\n\n", name, num_sms);

    // Use small token_dim so tests run fast
    const int TOKEN_DIM = 256;  // 1KB per token

    // tile_size=32, chunk_size=16 → K=2 chunks per tile
    const int TILE = 32;
    const int CHUNK = 16;

    int pass_count = 0;
    int total_tests = 0;

    // Test with various small tile counts: 1, 2, 3, 4, 5, 7
    int tile_counts[] = {1, 2, 3, 4, 5, 7};
    for (int tc : tile_counts) {
        total_tests++;
        if (run_test(tc * TILE, TILE, CHUNK, TOKEN_DIM, num_sms))
            pass_count++;
    }

    // Also test with K=1 (1 chunk per tile) — simplest case
    printf("\n  -- K=1 (tile=16, chunk=16) --\n");
    int tile_counts_k1[] = {1, 2, 3, 5};
    for (int tc : tile_counts_k1) {
        total_tests++;
        if (run_test(tc * 16, 16, 16, TOKEN_DIM, num_sms))
            pass_count++;
    }

    // K=4 (tile=64, chunk=16)
    printf("\n  -- K=4 (tile=64, chunk=16) --\n");
    int tile_counts_k4[] = {1, 2, 3};
    for (int tc : tile_counts_k4) {
        total_tests++;
        if (run_test(tc * 64, 64, 16, TOKEN_DIM, num_sms))
            pass_count++;
    }

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d/%d PASSED\n", pass_count, total_tests);
    printf("════════════════════════════════════════════\n");

    return (pass_count == total_tests) ? 0 : 1;
}
