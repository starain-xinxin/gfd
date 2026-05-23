#pragma once

#include "gfd/sg_device_primitives.cuh"
#include "gfd/device_primitives.cuh"

// ============================================================
// GFD SG Warp Specialization Framework
//
// Scatter-gather mode: transfer warp reads committed SGLists from
// SGTaskQueue, converts entries to Descriptors, and submits them
// through the existing DescriptorQueue -> CPU Poller pipeline.
//
// Unlike the tile-based framework (warp_spec.cuh), SG mode does
// not use fixed linear address mapping. Instead, addresses come
// from dynamically submitted DeviceSGEntry tuples.
//
// Kernel signature:
//   (DescriptorQueue* dq, SGTaskQueue* sq, ComputeFn fn)
//
// Usage:
//   struct MyCompute {
//       __device__ void operator()(gfd::sg_warp_spec::SGListView list) { ... }
//   };
//   GFD_SG_WARP_SPEC_KERNEL(my_sg_kernel, MyCompute);
// ============================================================

namespace gfd {
namespace sg_warp_spec {

// ---- SGListView: read-only view of a completed SG list ----
struct SGListView {
    uint32_t list_id;
    uint32_t count;
    uint32_t flags;
    int lane_id;
    uint32_t pool_offset;
    const SGTaskQueue* sq;

    __device__ __forceinline__
    DeviceSGEntry get_entry(uint32_t idx) const {
        uint32_t slot = (pool_offset + idx) % MAX_SG_POOL_ENTRIES;
        return sq->entries[slot];
    }

    template<typename T>
    __device__ __forceinline__
    T* dst_ptr(uint32_t idx) const {
        return reinterpret_cast<T*>(get_entry(idx).dst_addr);
    }

    __device__ __forceinline__
    uint32_t entry_size(uint32_t idx) const {
        return get_entry(idx).size;
    }
};

// Shared state for cross-warp coordination (same pattern as _WarpSpecState)
struct _SGWarpSpecState {
    uint32_t list_id;
    uint32_t count;
    uint32_t pool_offset;
    uint32_t flags;
    volatile int list_ready;    // transfer warp sets to 1 when DMA submitted
    volatile int compute_done;  // compute warp sets to 1 when done
    volatile int terminated;    // transfer warp sets to 1 when no more lists
};

// ---- Core: SG warp-specialized list processing loop ----
// Follows the same pattern as _warp_spec_tile_loop:
//   - Warp 0: transfer warp (poll SGList, write descriptors, wait done)
//   - Warp 1: compute warp (wait for list ready, call compute_fn)
//   - No __syncthreads in the main loop — uses shared memory polling
template<typename ComputeFn>
__device__ void _sg_transfer_warp_loop(
    DescriptorQueue* dq,
    SGTaskQueue* sq,
    ComputeFn compute_fn)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ _SGWarpSpecState _sg_state;

    if (threadIdx.x == 0) {
        _sg_state.list_ready = 0;
        _sg_state.compute_done = 0;
        _sg_state.terminated = 0;
        _sg_state.list_id = 0;
        _sg_state.count = 0;
        _sg_state.pool_offset = 0;
        _sg_state.flags = 0;
    }
    __syncthreads();

    if (warp_id == 0) {
        // === TRANSFER WARP ===
        uint64_t lists_processed = 0;

        while (true) {
            // Atomically claim the next list slot (multi-block safe)
            uint64_t list_idx;
            if (lane_id == 0) {
                list_idx = atomicAdd((unsigned long long*)&sq->list_read_idx, 1ULL);
            }
            list_idx = __shfl_sync(0xFFFFFFFF, list_idx, 0);

            uint32_t ring_slot = (uint32_t)(list_idx % MAX_SG_LISTS);

            // Poll for committed list (check sequence == list_idx + 1)
            bool committed = false;
            if (lane_id == 0) {
                int spin_count = 0;
                while (true) {
                    uint64_t seq = sq->lists[ring_slot].sequence;
                    if (seq == list_idx + 1) {
                        committed = true;
                        break;
                    }
                    if (sq->terminate) break;
#if __CUDA_ARCH__ >= 700
                    if (++spin_count > 64) {
                        __nanosleep(spin_count < 256 ? 100 : 1000);
                    }
#endif
                }
            }
            committed = __shfl_sync(0xFFFFFFFF, committed ? 1 : 0, 0);

            if (!committed) {
                // Signal termination
                if (lane_id == 0) {
                    __threadfence_block();
                    _sg_state.terminated = 1;
                }
                __syncwarp();
                break;
            }

            // Read list header (all lanes read for broadcast)
            uint32_t pool_offset = sq->lists[ring_slot].pool_offset;
            uint32_t count = sq->lists[ring_slot].count;
            uint32_t list_id = sq->lists[ring_slot].list_id;
            uint32_t flags = sq->lists[ring_slot].flags;

            // Reset state for this list
            if (lane_id == 0) {
                _sg_state.list_id = list_id;
                _sg_state.count = count;
                _sg_state.pool_offset = pool_offset;
                _sg_state.flags = flags;
                _sg_state.list_ready = 0;
                _sg_state.compute_done = 0;
            }
            __syncwarp();

            // Acquire descriptor queue slots
            uint64_t desc_base = 0;
            if (lane_id == 0) {
                desc_base = atomicAdd((unsigned long long*)&dq->write_idx,
                                      (unsigned long long)count);
            }
            desc_base = __shfl_sync(0xFFFFFFFF, desc_base, 0);

            // Per-block backpressure: wait until our reserved range
            // [desc_base, desc_base+count) won't overwrite unread entries.
            if (lane_id == 0) {
                while (true) {
                    uint64_t r = *((volatile uint64_t*)&dq->read_idx);
                    if (desc_base + count - r <= QUEUE_SIZE) break;
#if __CUDA_ARCH__ >= 700
                    __nanosleep(100);
#endif
                }
            }
            __syncwarp();

            // Convert SG entries -> Descriptors (warp-parallel)
            for (uint32_t i = lane_id; i < count; i += 32) {
                uint32_t entry_slot = (pool_offset + i) % MAX_SG_POOL_ENTRIES;
                uint64_t desc_slot = desc_base + i;
                Descriptor* desc = &dq->entries[desc_slot % QUEUE_SIZE];

                desc->src_addr = sq->entries[entry_slot].src_addr;
                desc->dst_addr = sq->entries[entry_slot].dst_addr;
                desc->size = sq->entries[entry_slot].size;
                desc->user_data = ((uint64_t)list_id << 32) | i;
                desc->flags = (i == count - 1) ? FLAG_LAST_IN_TILE : FLAG_NONE;
            }

            // Two-phase commit
            __threadfence_system();
            __syncwarp();

            for (uint32_t i = lane_id; i < count; i += 32) {
                uint64_t slot = desc_base + i;
                dq->entries[slot % QUEUE_SIZE].sequence = slot + 1;
            }

            __threadfence_system();
            __syncwarp();

            // Clear consumed list
            // list_read_idx already advanced by atomicAdd above
            if (lane_id == 0) {
                sq->lists[ring_slot].sequence = 0;

                // atomicMax on entry_consumed_idx (multi-block safe)
                uint64_t new_consumed = (uint64_t)pool_offset + count;
                uint64_t old_val = sq->entry_consumed_idx;
                while (new_consumed > old_val) {
                    uint64_t prev = atomicCAS((unsigned long long*)&sq->entry_consumed_idx,
                                              (unsigned long long)old_val,
                                              (unsigned long long)new_consumed);
                    if (prev == old_val) break;
                    old_val = prev;
                }
            }
            __syncwarp();

            // Wait for CPU poller to complete DMA for this list
            // Poll d_list_done[list_id] or lists_completed
            if (lane_id == 0) {
                if (sq->d_list_done) {
                    // Fine-grained: wait for this specific list
                    volatile uint64_t* signal = (volatile uint64_t*)&sq->d_list_done[list_id];
                    int spin_count = 0;
                    while (*signal == 0) {
#if __CUDA_ARCH__ >= 700
                        if (++spin_count > 64) {
                            __nanosleep(spin_count < 256 ? 100 : 1000);
                        }
#endif
                    }
                } else {
                    // Coarse: wait for lists_completed
                    int spin_count = 0;
                    while (*((volatile uint64_t*)&sq->lists_completed) < lists_processed + 1) {
#if __CUDA_ARCH__ >= 700
                        if (++spin_count > 64) {
                            __nanosleep(spin_count < 256 ? 100 : 1000);
                        }
#endif
                    }
                }

                // Signal compute warp
                __threadfence_block();
                _sg_state.list_ready = 1;
            }
            __syncwarp();

            lists_processed++;

            // Wait for compute warp to finish
            if (lane_id == 0) {
                while (_sg_state.compute_done == 0) {
                    // spin
                }
            }
            __syncwarp();
        }

    } else if (warp_id == 1) {
        // === COMPUTE WARP ===
        while (true) {
            // Wait for list_ready or terminated
            if (lane_id == 0) {
                while (_sg_state.list_ready == 0 && _sg_state.terminated == 0) {
                    // spin
                }
                __threadfence_block();
            }
            __syncwarp();

            if (_sg_state.terminated) break;

            // Build SGListView
            SGListView view;
            view.list_id = _sg_state.list_id;
            view.count = _sg_state.count;
            view.flags = _sg_state.flags;
            view.lane_id = lane_id;
            view.pool_offset = _sg_state.pool_offset;
            view.sq = sq;

            // User compute
            compute_fn(view);

            // Signal completion
            __syncwarp();
            if (lane_id == 0) {
                __threadfence_block();
                _sg_state.compute_done = 1;
                _sg_state.list_ready = 0;  // Reset for next list
            }
            __syncwarp();
        }
    }
    // else: extra warps idle
}

}  // namespace sg_warp_spec
}  // namespace gfd

// Kernel Generation Macro for SG Mode
// Block layout: 1 transfer warp + 1 compute warp = 64 threads
#define GFD_SG_WARP_SPEC_KERNEL(kernel_name, ComputeFnType)                   \
__global__ void kernel_name(                                                   \
    gfd::DescriptorQueue* dq,                                                  \
    gfd::SGTaskQueue* sq,                                                      \
    ComputeFnType compute_fn)                                                  \
{                                                                              \
    gfd::sg_warp_spec::_sg_transfer_warp_loop(dq, sq, compute_fn);            \
}
