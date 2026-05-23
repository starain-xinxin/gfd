// Minimal test for GPU-side dynamic SG list submission.
// Uses a 3-warp combined kernel: transfer + compute + submitter.
// Verifies data correctness after GPU-submitted SG lists are
// processed through the full GFD pipeline.

#include <gfd/gfd.h>
#include <gfd/sg_warp_spec.cuh>
#include <gfd/sg_device_primitives.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

struct NullCompute {
    __device__ void operator()(gfd::sg_warp_spec::SGListView list) {}
};

// Combined 3-warp kernel: transfer (warp 0), compute (warp 1), submitter (warp 2)
__global__ void gpu_submit_test_kernel(
    gfd::DescriptorQueue* dq,
    gfd::SGTaskQueue* sq,
    NullCompute compute_fn,
    const float* cpu_base,
    float* gpu_base,
    int entries_per_list,
    int num_lists,
    uint32_t entry_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ gfd::sg_warp_spec::_SGWarpSpecState _sg_state;

    if (threadIdx.x == 0) {
        _sg_state.list_ready = 0;
        _sg_state.compute_done = 0;
        _sg_state.terminated = 0;
    }
    __syncthreads();

    if (warp_id == 0) {
        // === TRANSFER WARP ===
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
                    __nanosleep(100);
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

            // Write descriptors
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
                    while (*signal == 0) { __nanosleep(100); }
                } else {
                    while (*((volatile uint64_t*)&sq->lists_completed) < lists_processed + 1)
                        __nanosleep(100);
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
        for (int list = 0; list < num_lists; list++) {
            gfd::sg::sg_wait_entry_space(sq, entries_per_list);
            uint64_t pool_offset = gfd::sg::sg_alloc_entries(sq, entries_per_list);

            if (lane_id == 0) {
                for (int i = 0; i < entries_per_list; i++) {
                    uint32_t slot = (uint32_t)((pool_offset + i) % gfd::MAX_SG_POOL_ENTRIES);
                    int global_idx = list * entries_per_list + i;
                    sq->entries[slot].src_addr = (uint64_t)cpu_base + (uint64_t)global_idx * entry_size;
                    sq->entries[slot].dst_addr = (uint64_t)gpu_base + (uint64_t)global_idx * entry_size;
                    sq->entries[slot].size = entry_size;
                    sq->entries[slot].tag = list;
                }
            }
            __syncwarp();

            gfd::sg::sg_wait_list_space(sq);
            uint64_t list_slot = gfd::sg::sg_alloc_list(sq);
            gfd::sg::sg_commit_list(sq, list_slot, (uint32_t)pool_offset,
                                    entries_per_list, list, gfd::SG_FLAG_NONE);
        }

        // Wait for transfer warp to drain, then terminate
        if (lane_id == 0) {
            while (true) {
                uint64_t read = *((volatile uint64_t*)&sq->list_read_idx);
                uint64_t alloc = *((volatile uint64_t*)&sq->list_alloc_idx);
                if (read >= alloc) break;
                __nanosleep(1000);
            }
            __threadfence_system();
            sq->terminate = 1;
            __threadfence_system();
        }
    }
}

int main() {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    fprintf(stderr, "=== SG GPU Dynamic Submission Test ===\n");

    cuInit(0);
    CUcontext ctx; CUdevice dev;
    cuDeviceGet(&dev, 0);
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxParams = {};
    cuCtxCreate(&ctx, &ctxParams, 0, dev);
#else
    cuCtxCreate(&ctx, 0, dev);
#endif

    constexpr int NUM_LISTS = 4;
    constexpr int ENTRIES_PER_LIST = 8;
    constexpr int TOTAL_ENTRIES = NUM_LISTS * ENTRIES_PER_LIST;
    constexpr size_t ENTRY_SIZE = 4096;
    constexpr size_t TOTAL_SIZE = TOTAL_ENTRIES * ENTRY_SIZE;

    float* cpu_data;
    cudaMallocHost(&cpu_data, TOTAL_SIZE);
    for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); i++) cpu_data[i] = (float)(i + 1);

    float* gpu_data;
    cudaMalloc(&gpu_data, TOTAL_SIZE);
    cudaMemset(gpu_data, 0, TOTAL_SIZE);

    // Create SG session (handles queue allocation + poller setup)
    gfd::SGWarpSpecConfig config;
    config.num_compute_warps = 1;
    config.num_blocks = 1;
    config.use_copy_engine = true;

    gfd::SGWarpSpecSession session(config);

    fprintf(stderr, "Launching 3-warp kernel (%d lists × %d entries)...\n",
            NUM_LISTS, ENTRIES_PER_LIST);

    // Launch combined kernel (96 threads = 3 warps)
    gpu_submit_test_kernel<<<1, 96>>>(
        session.get_desc_queue(), session.get_sg_queue(),
        NullCompute{},
        cpu_data, gpu_data, ENTRIES_PER_LIST, NUM_LISTS, ENTRY_SIZE);

    // Start poller
    session.get_poller()->start();

    // Wait
    cudaDeviceSynchronize();
    session.get_poller()->stop();

    auto stats = session.get_stats();
    fprintf(stderr, "Poller: desc=%lu bytes=%.2f MB\n",
            (unsigned long)stats.descriptors_processed,
            stats.bytes_transferred / (1024.0 * 1024.0));

    // Verify
    float* h_gpu = new float[TOTAL_SIZE / sizeof(float)];
    cudaMemcpy(h_gpu, gpu_data, TOTAL_SIZE, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); i++) {
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
    return errors == 0 ? 0 : 1;
}
