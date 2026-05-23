#include <gfd/gfd.h>
#include <gfd/sg_warp_spec.cuh>
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
// GFD SG (Scatter-Gather) Warp-Spec Example
//
// Demonstrates dynamic scatter-gather address submission for
// MoE-style workloads where token routing is determined at
// runtime (e.g., expert selection in Mixture-of-Experts).
//
// Two modes:
//   1. Host pre-fill: CPU builds SG entries before kernel launch
//   2. GPU dynamic submit: Compute warp builds SG entries at runtime
//
// Architecture:
//   Compute Warp → sg_submit_list() → SGTaskQueue
//   Transfer Warp → reads SGLists → DescriptorQueue → CPU Poller
//   CPU Poller → DMA → d_list_done[list_id] signal → GPU polls
// ============================================================

// ---- NoOp compute functor for linear mode benchmark ----
struct LinearNoOp {
    __device__ void operator()(gfd::warp_spec::ChunkView) {}
};
GFD_WARP_SPEC_KERNEL(noop_linear_kernel, LinearNoOp);

// ---- NoOp SG compute for benchmark ----
struct NoOpSG {
    __device__ void operator()(gfd::sg_warp_spec::SGListView) {}
};
GFD_SG_WARP_SPEC_KERNEL(noop_sg_kernel, NoOpSG);

// ---- MoE-style compute functor ----
// After SG list DMA completes, compute L2 norm of each transferred entry.
struct MoECompute {
    float* output;

    __device__ void operator()(gfd::sg_warp_spec::SGListView list) {
        for (uint32_t i = list.lane_id; i < list.count; i += 32) {
            gfd::DeviceSGEntry entry = list.get_entry(i);
            float* data = reinterpret_cast<float*>(entry.dst_addr);
            uint32_t num_floats = entry.size / sizeof(float);

            // Compute L2 norm
            float sq_sum = 0.0f;
            for (uint32_t d = 0; d < num_floats; d++) {
                sq_sum += data[d] * data[d];
            }
            float norm = sqrtf(sq_sum);

            // Store result (use list_id * MAX + entry_index as output slot)
            uint32_t out_idx = list.list_id * list.count + i;
            output[out_idx] = norm;
        }
    }
};

// Generate the SG warp-spec kernel
GFD_SG_WARP_SPEC_KERNEL(sg_moe_kernel, MoECompute);

// ---- Combined 3-warp kernel for GPU dynamic submission ----
// Warp 0: Transfer (polls SGLists → writes descriptors → waits DMA)
// Warp 1: Compute (processes transferred data)
// Warp 2: Submitter (builds SG lists at runtime from routing table)
//
// Note: CUDA may serialize kernels on separate streams, so the submitter
// must be in the SAME kernel as the transfer warp. This kernel manually
// implements the 3-warp pattern instead of using GFD_SG_WARP_SPEC_KERNEL.
__global__ void dynamic_moe_kernel(
    gfd::DescriptorQueue* dq,
    gfd::SGTaskQueue* sq,
    MoECompute compute_fn,
    const float* cpu_base,
    float* gpu_base,
    const int* routing_table,
    int total_tokens,
    int num_experts,
    uint32_t token_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Shared state for transfer↔compute handshake
    __shared__ gfd::sg_warp_spec::_SGWarpSpecState _sg_state;

    if (threadIdx.x == 0) {
        _sg_state.list_ready = 0;
        _sg_state.compute_done = 0;
        _sg_state.terminated = 0;
    }
    __syncthreads();

    if (warp_id == 0) {
        // === TRANSFER WARP === (same as _sg_transfer_warp_loop)
        uint64_t lists_processed = 0;
        while (true) {
            uint64_t list_idx;
            if (lane_id == 0) list_idx = sq->list_read_idx;
            list_idx = __shfl_sync(0xFFFFFFFF, list_idx, 0);
            uint32_t ring_slot = (uint32_t)(list_idx % gfd::MAX_SG_LISTS);

            bool committed = false;
            if (lane_id == 0) {
                while (true) {
                    uint64_t seq = sq->lists[ring_slot].sequence;
                    if (seq == list_idx + 1) { committed = true; break; }
                    if (sq->terminate) break;
#if __CUDA_ARCH__ >= 700
                    __nanosleep(100);
#endif
                }
            }
            committed = __shfl_sync(0xFFFFFFFF, committed ? 1 : 0, 0);
            if (!committed) {
                if (lane_id == 0) { __threadfence_block(); _sg_state.terminated = 1; }
                __syncwarp();
                break;
            }

            uint32_t pool_offset = sq->lists[ring_slot].pool_offset;
            uint32_t count = sq->lists[ring_slot].count;
            uint32_t list_id = sq->lists[ring_slot].list_id;
            uint32_t flags = sq->lists[ring_slot].flags;

            if (lane_id == 0) {
                _sg_state.list_id = list_id;
                _sg_state.count = count;
                _sg_state.pool_offset = pool_offset;
                _sg_state.flags = flags;
                _sg_state.list_ready = 0;
                _sg_state.compute_done = 0;
            }
            __syncwarp();

            uint64_t desc_base = 0;
            if (lane_id == 0) {
                desc_base = atomicAdd((unsigned long long*)&dq->write_idx, (unsigned long long)count);
            }
            desc_base = __shfl_sync(0xFFFFFFFF, desc_base, 0);

            if (lane_id == 0) {
                while (true) {
                    uint64_t w = dq->write_idx;
                    uint64_t r = *((volatile uint64_t*)&dq->read_idx);
                    if (w - r < gfd::QUEUE_SIZE - count * 2) break;
                }
            }
            __syncwarp();

            for (uint32_t i = lane_id; i < count; i += 32) {
                uint32_t entry_slot = (pool_offset + i) % gfd::MAX_SG_POOL_ENTRIES;
                uint64_t desc_slot = desc_base + i;
                gfd::Descriptor* desc = &dq->entries[desc_slot % gfd::QUEUE_SIZE];
                desc->src_addr = sq->entries[entry_slot].src_addr;
                desc->dst_addr = sq->entries[entry_slot].dst_addr;
                desc->size = sq->entries[entry_slot].size;
                desc->user_data = ((uint64_t)list_id << 32) | i;
                desc->flags = (i == count - 1) ? gfd::FLAG_LAST_IN_TILE : gfd::FLAG_NONE;
            }

            __threadfence_system();
            __syncwarp();
            for (uint32_t i = lane_id; i < count; i += 32) {
                uint64_t slot = desc_base + i;
                dq->entries[slot % gfd::QUEUE_SIZE].sequence = slot + 1;
            }
            __threadfence_system();
            __syncwarp();

            if (lane_id == 0) {
                sq->lists[ring_slot].sequence = 0;
                sq->list_read_idx = list_idx + 1;
                uint64_t new_consumed = (uint64_t)pool_offset + count;
                uint64_t old_consumed = sq->entry_consumed_idx;
                if (new_consumed > old_consumed) sq->entry_consumed_idx = new_consumed;
            }
            __syncwarp();

            // Wait for DMA completion
            if (lane_id == 0) {
                if (sq->d_list_done) {
                    volatile uint64_t* signal = (volatile uint64_t*)&sq->d_list_done[list_id];
                    while (*signal == 0) {
#if __CUDA_ARCH__ >= 700
                        __nanosleep(100);
#endif
                    }
                } else {
                    while (*((volatile uint64_t*)&sq->lists_completed) < lists_processed + 1) {
#if __CUDA_ARCH__ >= 700
                        __nanosleep(100);
#endif
                    }
                }
                __threadfence_block();
                _sg_state.list_ready = 1;
            }
            __syncwarp();
            lists_processed++;

            if (lane_id == 0) {
                while (_sg_state.compute_done == 0) {}
            }
            __syncwarp();
        }

    } else if (warp_id == 1) {
        // === COMPUTE WARP ===
        while (true) {
            if (lane_id == 0) {
                while (_sg_state.list_ready == 0 && _sg_state.terminated == 0) {}
                __threadfence_block();
            }
            __syncwarp();
            if (_sg_state.terminated) break;

            gfd::sg_warp_spec::SGListView view;
            view.list_id = _sg_state.list_id;
            view.count = _sg_state.count;
            view.flags = _sg_state.flags;
            view.lane_id = lane_id;
            view.pool_offset = _sg_state.pool_offset;
            view.sq = sq;
            compute_fn(view);

            __syncwarp();
            if (lane_id == 0) {
                __threadfence_block();
                _sg_state.compute_done = 1;
                _sg_state.list_ready = 0;
            }
            __syncwarp();
        }

    } else if (warp_id == 2) {
        // === SUBMITTER WARP ===
        for (int expert = 0; expert < num_experts; expert++) {
            int count = 0;
            if (lane_id == 0) {
                for (int t = 0; t < total_tokens; t++) {
                    if (routing_table[t] == expert) count++;
                }
            }
            count = __shfl_sync(0xFFFFFFFF, count, 0);
            if (count == 0) continue;

            gfd::sg::sg_wait_entry_space(sq, count);
            uint64_t po = gfd::sg::sg_alloc_entries(sq, count);

            if (lane_id == 0) {
                int idx = 0;
                for (int t = 0; t < total_tokens && idx < count; t++) {
                    if (routing_table[t] == expert) {
                        uint32_t slot = (uint32_t)((po + idx) % gfd::MAX_SG_POOL_ENTRIES);
                        sq->entries[slot].src_addr = (uint64_t)cpu_base + (uint64_t)t * token_size;
                        sq->entries[slot].dst_addr = (uint64_t)gpu_base + (uint64_t)t * token_size;
                        sq->entries[slot].size = token_size;
                        sq->entries[slot].tag = expert;
                        idx++;
                    }
                }
            }
            __syncwarp();

            gfd::sg::sg_wait_list_space(sq);
            uint64_t ls = gfd::sg::sg_alloc_list(sq);
            gfd::sg::sg_commit_list(sq, ls, (uint32_t)po, count, expert, gfd::SG_FLAG_NONE);
        }

        // All lists submitted — wait for transfer warp to drain, then terminate
        if (lane_id == 0) {
            while (true) {
                uint64_t read = *((volatile uint64_t*)&sq->list_read_idx);
                uint64_t alloc = *((volatile uint64_t*)&sq->list_alloc_idx);
                if (read >= alloc) break;
#if __CUDA_ARCH__ >= 700
                __nanosleep(1000);
#endif
            }
            __threadfence_system();
            sq->terminate = 1;
            __threadfence_system();
        }
    }
}

// ---- Helpers ----
static constexpr int WARMUP = 2;
static constexpr int ITERS = 5;

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    int idx = (int)(p / 100.0 * (v.size() - 1));
    return v[idx];
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
    constexpr int TOTAL_TOKENS = 1024;
    constexpr int TOKEN_DIM = 1024;      // floats per token (4KB)
    constexpr size_t TOKEN_SIZE = TOKEN_DIM * sizeof(float);
    constexpr size_t TOTAL_SIZE = TOTAL_TOKENS * TOKEN_SIZE;
    constexpr int NUM_EXPERTS = 8;
    constexpr int TOKENS_PER_EXPERT = TOTAL_TOKENS / NUM_EXPERTS;

    setbuf(stdout, NULL);
    printf("=== GFD SG (Scatter-Gather) Warp-Spec Example ===\n");
    printf("Total tokens: %d, Token size: %zu bytes (%d floats)\n",
           TOTAL_TOKENS, TOKEN_SIZE, TOKEN_DIM);
    printf("Total: %.2f MB, Experts: %d, Tokens/expert: %d\n",
           TOTAL_SIZE / (1024.0 * 1024.0), NUM_EXPERTS, TOKENS_PER_EXPERT);

    // ---- Allocate memory ----
    float* cpu_data;
    cudaMallocHost(&cpu_data, TOTAL_SIZE);
    for (int t = 0; t < TOTAL_TOKENS; t++) {
        float* dst = cpu_data + (size_t)t * TOKEN_DIM;
        for (int d = 0; d < TOKEN_DIM; d++) {
            dst[d] = sinf((float)(t * 31 + d) * 0.01f);
        }
    }

    float* gpu_data;
    cudaMalloc(&gpu_data, TOTAL_SIZE);
    cudaMemset(gpu_data, 0, TOTAL_SIZE);

    // Output buffer (num_experts * max_tokens_per_expert)
    float* d_output;
    cudaMalloc(&d_output, TOTAL_TOKENS * sizeof(float));
    cudaMemset(d_output, 0, TOTAL_TOKENS * sizeof(float));

    // ====================================================================
    // Test 1: Host Pre-fill SG Mode
    //
    // CPU builds SG entries for each expert before kernel launch.
    // The transfer warp reads and converts them to descriptors.
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 1: Host Pre-fill SG Mode\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        gfd::SGWarpSpecConfig sg_config;
        sg_config.num_compute_warps = 1;
        sg_config.num_blocks = 1;  // Single block for simplicity

        gfd::SGWarpSpecSession session(sg_config);

        // Build SG entries: round-robin assignment (token i → expert i%N)
        std::vector<gfd::DeviceSGEntry> entries(TOTAL_TOKENS);
        for (int t = 0; t < TOTAL_TOKENS; t++) {
            entries[t].src_addr = (uint64_t)cpu_data + (uint64_t)t * TOKEN_SIZE;
            entries[t].dst_addr = (uint64_t)gpu_data + (uint64_t)t * TOKEN_SIZE;
            entries[t].size = TOKEN_SIZE;
            entries[t].tag = t % NUM_EXPERTS;
        }

        // Submit one list per expert from host
        for (int expert = 0; expert < NUM_EXPERTS; expert++) {
            std::vector<gfd::DeviceSGEntry> expert_entries;
            for (int t = 0; t < TOTAL_TOKENS; t++) {
                if (t % NUM_EXPERTS == expert) {
                    expert_entries.push_back(entries[t]);
                }
            }
            session.submit_sg_list(expert_entries.data(),
                                   (uint32_t)expert_entries.size(),
                                   expert, gfd::SG_FLAG_HOST_SUBMITTED);
        }

        printf("  Submitted %d SG lists (%d entries each)\n",
               NUM_EXPERTS, TOKENS_PER_EXPERT);

        // Launch kernel
        cudaMemset(gpu_data, 0, TOTAL_SIZE);
        auto t0 = std::chrono::high_resolution_clock::now();
        session.launch(sg_moe_kernel, MoECompute{d_output});
        session.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("  Elapsed: %.2f ms\n", elapsed);
        printf("  BW: %.2f GB/s\n", TOTAL_SIZE / (elapsed * 1e6));

        // Verify: check that GPU data matches CPU data
        float* h_gpu_data = new float[TOTAL_TOKENS * TOKEN_DIM];
        cudaMemcpy(h_gpu_data, gpu_data, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && errors < 10; t++) {
            for (int d = 0; d < TOKEN_DIM && errors < 10; d++) {
                float expected = cpu_data[t * TOKEN_DIM + d];
                float actual = h_gpu_data[t * TOKEN_DIM + d];
                if (fabsf(expected - actual) > 1e-5f) {
                    if (errors < 3) {
                        printf("    Mismatch at token %d, dim %d: expected %.6f, got %.6f\n",
                               t, d, expected, actual);
                    }
                    errors++;
                }
            }
        }
        printf("  Data correctness: %s (%d errors)\n",
               errors == 0 ? "PASS" : "FAIL", errors);

        // Verify compute output
        float* h_output = new float[TOTAL_TOKENS];
        cudaMemcpy(h_output, d_output, TOTAL_TOKENS * sizeof(float), cudaMemcpyDeviceToHost);
        int compute_errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && compute_errors < 5; t++) {
            if (!isfinite(h_output[t])) compute_errors++;
        }
        printf("  Compute correctness: %s (%d errors)\n",
               compute_errors == 0 ? "PASS" : "FAIL", compute_errors);

        delete[] h_output;
        delete[] h_gpu_data;

        auto stats = session.get_stats();
        printf("  [Poller] descriptors=%lu, bytes=%.2f MB\n",
               (unsigned long)stats.descriptors_processed,
               stats.bytes_transferred / (1024.0 * 1024.0));
    }

    // ====================================================================
    // Test 2: Backward Compat — linear_to_sg_entries
    //
    // Demonstrates using sg_compat::linear_to_sg_entries to convert
    // a linear address mapping into SG entries. Verifies that the
    // SG pipeline produces identical results to the original mode.
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 2: Backward Compat (linear_to_sg_entries)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        gfd::SGWarpSpecConfig sg_config;
        sg_config.num_compute_warps = 1;
        sg_config.num_blocks = 1;

        gfd::SGWarpSpecSession session(sg_config);

        // Use linear_to_sg_entries to generate entries
        constexpr int BATCH = 256;
        gfd::DeviceSGEntry linear_entries[BATCH];
        gfd::sg_compat::linear_to_sg_entries(
            linear_entries, cpu_data, gpu_data,
            TOKEN_SIZE, 0, BATCH, /*tag=*/0);

        // Submit as single list
        session.submit_sg_list(linear_entries, BATCH, 0, gfd::SG_FLAG_HOST_SUBMITTED);

        printf("  Submitted 1 SG list (%d linear entries)\n", BATCH);

        cudaMemset(gpu_data, 0, TOTAL_SIZE);
        auto t0 = std::chrono::high_resolution_clock::now();
        session.launch(sg_moe_kernel, MoECompute{d_output});
        session.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("  Elapsed: %.2f ms\n", elapsed);
        printf("  BW: %.2f GB/s\n", (BATCH * TOKEN_SIZE) / (elapsed * 1e6));

        // Verify first BATCH tokens
        float* h_gpu = new float[BATCH * TOKEN_DIM];
        cudaMemcpy(h_gpu, gpu_data, BATCH * TOKEN_SIZE, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < BATCH && errors < 5; t++) {
            for (int d = 0; d < TOKEN_DIM && errors < 5; d++) {
                float expected = cpu_data[t * TOKEN_DIM + d];
                float actual = h_gpu[t * TOKEN_DIM + d];
                if (fabsf(expected - actual) > 1e-5f) errors++;
            }
        }
        printf("  Linear compat correctness: %s (%d errors)\n",
               errors == 0 ? "PASS" : "FAIL", errors);
        delete[] h_gpu;
    }

    // ====================================================================
    // Test 3: GPU Dynamic Submission
    //
    // A single kernel with 3 warps: transfer, compute, and submitter.
    // The submitter warp dynamically builds SG lists based on routing
    // decisions, while the transfer warp processes them in real-time.
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 3: GPU Dynamic Submission\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        // Build routing table on device: round-robin (token t → expert t % N)
        int* d_routing;
        cudaMalloc(&d_routing, TOTAL_TOKENS * sizeof(int));
        std::vector<int> h_routing(TOTAL_TOKENS);
        for (int t = 0; t < TOTAL_TOKENS; t++) h_routing[t] = t % NUM_EXPERTS;
        cudaMemcpy(d_routing, h_routing.data(), TOTAL_TOKENS * sizeof(int),
                   cudaMemcpyHostToDevice);

        gfd::SGWarpSpecConfig sg_config;
        sg_config.num_compute_warps = 1;
        sg_config.num_blocks = 1;

        gfd::SGWarpSpecSession session(sg_config);

        cudaMemset(gpu_data, 0, TOTAL_SIZE);
        cudaMemset(d_output, 0, TOTAL_TOKENS * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();

        // Launch combined 3-warp kernel (96 threads)
        // Warp 0: transfer, Warp 1: compute, Warp 2: submitter
        dynamic_moe_kernel<<<1, 96>>>(
            session.get_desc_queue(), session.get_sg_queue(),
            MoECompute{d_output},
            cpu_data, gpu_data, d_routing,
            TOTAL_TOKENS, NUM_EXPERTS, TOKEN_SIZE);

        // Start poller (handles DMA + completion signaling)
        session.get_sg_queue()->terminate = 0;  // ensure not set
        // Note: we can't use session.launch() since we have a custom kernel.
        // Manually start the poller.
        auto* poller = session.get_poller();
        poller->start();

        cudaDeviceSynchronize();
        poller->stop();

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("  Elapsed: %.2f ms\n", elapsed);
        printf("  BW: %.2f GB/s\n", TOTAL_SIZE / (elapsed * 1e6));

        // Verify data
        float* h_gpu_data = new float[TOTAL_TOKENS * TOKEN_DIM];
        cudaMemcpy(h_gpu_data, gpu_data, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int t = 0; t < TOTAL_TOKENS && errors < 10; t++) {
            for (int d = 0; d < TOKEN_DIM && errors < 10; d++) {
                float expected = cpu_data[t * TOKEN_DIM + d];
                float actual = h_gpu_data[t * TOKEN_DIM + d];
                if (fabsf(expected - actual) > 1e-5f) {
                    if (errors < 3) {
                        printf("    Mismatch at token %d, dim %d: expected %.6f, got %.6f\n",
                               t, d, expected, actual);
                    }
                    errors++;
                }
            }
        }
        printf("  Data correctness: %s (%d errors)\n",
               errors == 0 ? "PASS" : "FAIL", errors);

        auto stats = session.get_stats();
        printf("  [Poller] descriptors=%lu, bytes=%.2f MB\n",
               (unsigned long)stats.descriptors_processed,
               stats.bytes_transferred / (1024.0 * 1024.0));

        delete[] h_gpu_data;
        cudaFree(d_routing);
    }

    // ---- Benchmark result storage for summary table ----
    // Test 4: single-block comparison at 3 sizes
    constexpr int BM_SIZES[] = {1024, 4096, 16384};  // 4MB, 16MB, 64MB
    constexpr int BM_NUM_SIZES = 3;
    double bm_sg_bw[BM_NUM_SIZES] = {}, bm_lin_bw[BM_NUM_SIZES] = {}, bm_opt_bw[BM_NUM_SIZES] = {};
    double bm_sg_p50[BM_NUM_SIZES] = {}, bm_lin_p50[BM_NUM_SIZES] = {}, bm_opt_p50[BM_NUM_SIZES] = {};
    // Test 5: multi-block SG scaling (64MB)
    constexpr int BM5_BLOCKS[] = {1, 2, 4, 8, 16};
    constexpr int BM5_COUNT = 5;
    double bm5_bw[BM5_COUNT] = {}, bm5_p50[BM5_COUNT] = {};
    // Test 6: multi-block linear scaling (64MB)
    constexpr int BM6_BLOCKS[] = {1, 2, 4, 8, 16, 108};
    constexpr int BM6_COUNT = 6;
    double bm6_bw[BM6_COUNT] = {}, bm6_p50[BM6_COUNT] = {};

    // ====================================================================
    // Test 4: SG vs Linear Benchmark
    //
    // Compare SG mode bandwidth against equivalent linear mode to
    // measure the overhead of address indirection. Both use NoOp
    // compute to isolate transfer overhead.
    //
    // Uses single block for both modes (fair apples-to-apples comparison
    // of the transfer pipeline). Multi-block SG is tested separately.
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 4: SG vs Linear Benchmark\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        // Test at multiple sizes to show how fixed overhead amortizes
        constexpr int SIZES[] = {1024, 4096, 16384};  // 4MB, 16MB, 64MB
        constexpr int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

        for (int si = 0; si < NUM_SIZES; si++) {
            int bench_tokens = SIZES[si];
            size_t bench_size = (size_t)bench_tokens * TOKEN_SIZE;

            float* bench_cpu;
            cudaMallocHost(&bench_cpu, bench_size);
            for (size_t i = 0; i < bench_size / sizeof(float); i++)
                bench_cpu[i] = sinf((float)i * 0.001f);

            float* bench_gpu;
            cudaMalloc(&bench_gpu, bench_size);

            printf("\n  --- %d tokens (%.0f MB) ---\n", bench_tokens,
                   bench_size / (1024.0 * 1024.0));

            // ---- SG Mode (single block, NoOp compute) ----
            constexpr int SG_BATCH = 1024;
            int sg_num_lists = bench_tokens / SG_BATCH;
            if (sg_num_lists == 0) sg_num_lists = 1;

            std::vector<double> sg_times;
            {
                gfd::SGWarpSpecConfig sg_config;
                sg_config.num_compute_warps = 1;
                sg_config.num_blocks = 1;
                gfd::SGWarpSpecSession session(sg_config);

                for (int iter = 0; iter < WARMUP + ITERS; iter++) {
                    session.reset();

                    int tokens_submitted = 0;
                    for (int batch = 0; batch < sg_num_lists; batch++) {
                        int count = std::min(SG_BATCH, bench_tokens - tokens_submitted);
                        std::vector<gfd::DeviceSGEntry> entries(count);
                        gfd::sg_compat::linear_to_sg_entries(
                            entries.data(), bench_cpu, bench_gpu,
                            TOKEN_SIZE, tokens_submitted, count, batch);
                        session.submit_sg_list(entries.data(), count,
                                               batch, gfd::SG_FLAG_HOST_SUBMITTED);
                        tokens_submitted += count;
                    }

                    cudaMemset(bench_gpu, 0, bench_size);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    session.launch(noop_sg_kernel, NoOpSG{});
                    session.synchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (iter >= WARMUP) sg_times.push_back(ms);
                }
            }

            double sg_p50 = percentile(sg_times, 50);
            double sg_bw = bench_size / (sg_p50 * 1e6);

            // ---- Linear Mode: default (K=4 chunks/tile, 128 tokens/tile) ----
            std::vector<double> lin_times;
            {
                gfd::WarpSpecConfig lin_config;
                lin_config.total_tokens = bench_tokens;
                lin_config.token_size = TOKEN_SIZE;
                lin_config.cpu_src = bench_cpu;
                lin_config.gpu_dst = bench_gpu;
                lin_config.tokens_per_tile = 128;
                lin_config.num_blocks = 1;
                lin_config.use_copy_engine = true;
                gfd::WarpSpecSession lin_session(lin_config);

                for (int iter = 0; iter < WARMUP + ITERS; iter++) {
                    cudaMemset(bench_gpu, 0, bench_size);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    lin_session.launch(noop_linear_kernel, LinearNoOp{});
                    lin_session.synchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (iter >= WARMUP) lin_times.push_back(ms);
                }
            }

            double lin_p50 = percentile(lin_times, 50);
            double lin_bw = bench_size / (lin_p50 * 1e6);

            // ---- Linear Mode: optimized (K=1, per_tile_mode, 1024 tokens/tile) ----
            // Reduces round-trips by 32x: one flush per 1024 tokens (4MB) like SG
            std::vector<double> opt_times;
            {
                gfd::WarpSpecConfig opt_config;
                opt_config.total_tokens = bench_tokens;
                opt_config.token_size = TOKEN_SIZE;
                opt_config.cpu_src = bench_cpu;
                opt_config.gpu_dst = bench_gpu;
                opt_config.tokens_per_tile = 1024;      // 1024 tokens = 4MB per tile
                opt_config.per_tile_mode = true;         // K=1: entire tile as single chunk
                opt_config.num_blocks = 1;
                opt_config.use_copy_engine = true;
                gfd::WarpSpecSession opt_session(opt_config);

                for (int iter = 0; iter < WARMUP + ITERS; iter++) {
                    cudaMemset(bench_gpu, 0, bench_size);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    opt_session.launch(noop_linear_kernel, LinearNoOp{});
                    opt_session.synchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (iter >= WARMUP) opt_times.push_back(ms);
                }
            }

            double opt_p50 = percentile(opt_times, 50);
            double opt_bw = bench_size / (opt_p50 * 1e6);

            printf("  SG:          P50 = %7.2f ms, BW = %.2f GB/s\n", sg_p50, sg_bw);
            printf("  Linear:      P50 = %7.2f ms, BW = %.2f GB/s  (K=4, 128tok/tile)\n", lin_p50, lin_bw);
            printf("  Linear-opt:  P50 = %7.2f ms, BW = %.2f GB/s  (K=1, 1024tok/tile)\n", opt_p50, opt_bw);

            bm_sg_bw[si] = sg_bw;   bm_sg_p50[si] = sg_p50;
            bm_lin_bw[si] = lin_bw;  bm_lin_p50[si] = lin_p50;
            bm_opt_bw[si] = opt_bw;  bm_opt_p50[si] = opt_p50;

            cudaFree(bench_gpu);
            cudaFreeHost(bench_cpu);
        }
    }

    // ====================================================================
    // Test 5: Multi-block SG Scaling
    //
    // Test SG mode with increasing block counts to measure how
    // multi-block parallelism affects throughput.
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 5: Multi-block SG Scaling (64 MB)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        constexpr int MB_TOKENS = 16384;  // 64MB
        constexpr size_t MB_SIZE = (size_t)MB_TOKENS * TOKEN_SIZE;
        constexpr int MB_SG_BATCH = 1024;
        constexpr int MB_NUM_LISTS = MB_TOKENS / MB_SG_BATCH;  // 16 lists

        float* mb_cpu;
        cudaMallocHost(&mb_cpu, MB_SIZE);
        for (size_t i = 0; i < MB_SIZE / sizeof(float); i++)
            mb_cpu[i] = sinf((float)i * 0.001f);
        float* mb_gpu;
        cudaMalloc(&mb_gpu, MB_SIZE);

        int block_counts[] = {1, 2, 4, 8, 16};
        int num_configs = sizeof(block_counts) / sizeof(block_counts[0]);

        for (int ci = 0; ci < num_configs; ci++) {
            int nblocks = block_counts[ci];
            std::vector<double> times;

            {
                gfd::SGWarpSpecConfig sg_config;
                sg_config.num_compute_warps = 1;
                sg_config.num_blocks = nblocks;
                gfd::SGWarpSpecSession session(sg_config);

                for (int iter = 0; iter < WARMUP + ITERS; iter++) {
                    session.reset();

                    for (int batch = 0; batch < MB_NUM_LISTS; batch++) {
                        std::vector<gfd::DeviceSGEntry> entries(MB_SG_BATCH);
                        gfd::sg_compat::linear_to_sg_entries(
                            entries.data(), mb_cpu, mb_gpu,
                            TOKEN_SIZE, batch * MB_SG_BATCH, MB_SG_BATCH, batch);
                        session.submit_sg_list(entries.data(), MB_SG_BATCH,
                                               batch, gfd::SG_FLAG_HOST_SUBMITTED);
                    }

                    cudaMemset(mb_gpu, 0, MB_SIZE);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    session.launch(noop_sg_kernel, NoOpSG{});
                    session.synchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (iter >= WARMUP) times.push_back(ms);
                }
            }

            double p50 = percentile(times, 50);
            double bw = MB_SIZE / (p50 * 1e6);
            printf("  %2d blocks: P50 = %7.2f ms, BW = %.2f GB/s\n", nblocks, p50, bw);
            bm5_bw[ci] = bw;  bm5_p50[ci] = p50;
        }

        cudaFree(mb_gpu);
        cudaFreeHost(mb_cpu);
    }

    // ====================================================================
    // Test 6: Multi-block Linear Scaling (64 MB, optimized K=1)
    //
    // Verifies that the linear warp-spec mode works correctly with
    // multiple blocks (previously caused hang due to missing
    // backpressure + partial batch tile_progress_ tracking bug).
    // ====================================================================
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 6: Multi-block Linear Scaling (64 MB, K=1)\n");
    printf("────────────────────────────────────────────────────────────\n");
    {
        constexpr int LIN_TOKENS = 16384;  // 64MB
        constexpr size_t LIN_SIZE = (size_t)LIN_TOKENS * TOKEN_SIZE;

        float* lin_cpu;
        cudaMallocHost(&lin_cpu, LIN_SIZE);
        for (size_t i = 0; i < LIN_SIZE / sizeof(float); i++)
            lin_cpu[i] = sinf((float)i * 0.001f);
        float* lin_gpu;
        cudaMalloc(&lin_gpu, LIN_SIZE);

        int block_counts[] = {1, 2, 4, 8, 16, 0};  // 0 = all SMs
        int num_configs = sizeof(block_counts) / sizeof(block_counts[0]);

        for (int ci = 0; ci < num_configs; ci++) {
            int nblocks = block_counts[ci];
            std::vector<double> times;

            {
                gfd::WarpSpecConfig lin_config;
                lin_config.total_tokens = LIN_TOKENS;
                lin_config.token_size = TOKEN_SIZE;
                lin_config.cpu_src = lin_cpu;
                lin_config.gpu_dst = lin_gpu;
                lin_config.tokens_per_tile = 1024;
                lin_config.per_tile_mode = true;
                lin_config.num_blocks = nblocks;
                lin_config.use_copy_engine = true;
                gfd::WarpSpecSession lin_session(lin_config);

                for (int iter = 0; iter < WARMUP + ITERS; iter++) {
                    cudaMemset(lin_gpu, 0, LIN_SIZE);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    lin_session.launch(noop_linear_kernel, LinearNoOp{});
                    lin_session.synchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (iter >= WARMUP) times.push_back(ms);
                }
            }

            double p50 = percentile(times, 50);
            double bw = LIN_SIZE / (p50 * 1e6);
            int display_blocks = nblocks == 0 ? 108 : nblocks;
            printf("  %3d blocks: P50 = %7.2f ms, BW = %.2f GB/s\n",
                   display_blocks, p50, bw);
            bm6_bw[ci] = bw;  bm6_p50[ci] = p50;
        }

        cudaFree(lin_gpu);
        cudaFreeHost(lin_cpu);
    }

    // ====================================================================
    // Summary: Benchmark Results
    // ====================================================================
    printf("\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Benchmark Summary\n");
    printf("════════════════════════════════════════════════════════════\n");

    // Table 1: Single-block mode comparison
    printf("\n  [Single Block] SG vs Linear (token_size=%d B)\n", TOKEN_SIZE);
    printf("  ┌──────────┬────────────────────┬────────────────────┬────────────────────┐\n");
    printf("  │   Size   │    SG (1 block)    │ Linear-def (K=4)   │ Linear-opt (K=1)   │\n");
    printf("  ├──────────┼────────────────────┼────────────────────┼────────────────────┤\n");
    for (int i = 0; i < BM_NUM_SIZES; i++) {
        double mb = (double)BM_SIZES[i] * TOKEN_SIZE / (1024.0 * 1024.0);
        printf("  │ %4.0f MB  │ %6.2f ms %6.1f G │ %6.2f ms %6.1f G │ %6.2f ms %6.1f G │\n",
               mb,
               bm_sg_p50[i],  bm_sg_bw[i],
               bm_lin_p50[i], bm_lin_bw[i],
               bm_opt_p50[i], bm_opt_bw[i]);
    }
    printf("  └──────────┴────────────────────┴────────────────────┴────────────────────┘\n");
    printf("  (G = GB/s, P50 latency)\n");

    // Table 2: Multi-block scaling (64 MB)
    printf("\n  [Multi-Block Scaling] 64 MB transfer\n");
    printf("  ┌─────────┬────────────────────┬────────────────────┐\n");
    printf("  │ Blocks  │      SG Mode       │  Linear-opt (K=1)  │\n");
    printf("  ├─────────┼────────────────────┼────────────────────┤\n");
    for (int i = 0; i < BM6_COUNT; i++) {
        int lin_blocks = BM6_BLOCKS[i];
        if (i < BM5_COUNT) {
            printf("  │  %4d   │ %6.2f ms %6.1f G │ %6.2f ms %6.1f G │\n",
                   lin_blocks,
                   bm5_p50[i], bm5_bw[i],
                   bm6_p50[i], bm6_bw[i]);
        } else {
            printf("  │  %4d   │        ---         │ %6.2f ms %6.1f G │\n",
                   lin_blocks,
                   bm6_p50[i], bm6_bw[i]);
        }
    }
    printf("  └─────────┴────────────────────┴────────────────────┘\n");

    // Peak summary
    double peak_sg = 0, peak_lin = 0, peak_opt = 0;
    for (int i = 0; i < BM_NUM_SIZES; i++) {
        if (bm_sg_bw[i] > peak_sg) peak_sg = bm_sg_bw[i];
        if (bm_lin_bw[i] > peak_lin) peak_lin = bm_lin_bw[i];
        if (bm_opt_bw[i] > peak_opt) peak_opt = bm_opt_bw[i];
    }
    double peak_sg_mb = 0, peak_lin_mb = 0;
    for (int i = 0; i < BM5_COUNT; i++)
        if (bm5_bw[i] > peak_sg_mb) peak_sg_mb = bm5_bw[i];
    for (int i = 0; i < BM6_COUNT; i++)
        if (bm6_bw[i] > peak_lin_mb) peak_lin_mb = bm6_bw[i];

    printf("\n  Peak Bandwidth:\n");
    printf("    SG      (1 block):  %6.1f GB/s\n", peak_sg);
    printf("    SG   (multi-block): %6.1f GB/s\n", peak_sg_mb);
    printf("    Linear-def (K=4):   %6.1f GB/s\n", peak_lin);
    printf("    Linear-opt (K=1):   %6.1f GB/s\n", peak_opt);
    printf("    Linear-opt (multi): %6.1f GB/s\n", peak_lin_mb);
    printf("════════════════════════════════════════════════════════════\n");

    // ---- Cleanup ----
    cudaFree(d_output);
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);

    printf("\nDone.\n");
    return 0;
}
