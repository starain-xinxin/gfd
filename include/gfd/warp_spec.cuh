#pragma once

#include "gfd/device_primitives.cuh"

// ============================================================
// GFD Warp Specialization Framework (Layer 2)
//
// High-level API for building warp-specialized transfer+compute kernels.
// Users only need to define a compute functor with:
//   __device__ void operator()(gfd::warp_spec::ChunkView chunk)
//
// The framework handles:
//   - Dynamic tile acquisition (atomicAdd scheduling)
//   - Per-chunk slot allocation and descriptor submission
//   - Fence, commit, and completion polling
//   - Cross-warp synchronization (transfer warp → compute warps)
//   - Multi-tile looping (one block processes many tiles)
//
// Usage:
//   struct MyCompute {
//       float* output;
//       __device__ void operator()(gfd::warp_spec::ChunkView chunk) {
//           for (int i = chunk.lane_id; i < chunk.size; i += 32) {
//               float* data = chunk.data<float>(i);
//               output[chunk.global_idx(i)] = data[0];
//           }
//       }
//   };
//   GFD_WARP_SPEC_KERNEL(my_kernel, MyCompute);
//
// ============================================================

namespace gfd {
namespace warp_spec {

// Maximum chunks per tile supported by shared memory layout
constexpr int MAX_CHUNKS_PER_TILE = 16;

// ---- ChunkView: read-only view of a completed chunk ----
// Passed to user's compute function. Data is guaranteed to be available.
struct ChunkView {
    void* base_ptr;                // GPU address of first token in this chunk
    uint32_t token_size;           // Bytes per token
    uint32_t size;                 // Number of tokens in this chunk

    uint32_t tile_id;              // Global tile ID
    uint32_t chunk_id;             // Chunk index within tile (0..K-1)
    uint32_t global_token_offset;  // Global index of first token in chunk

    int lane_id;                   // Thread's lane within warp (0..31)

    // Get pointer to the i-th token in this chunk (0-indexed)
    template<typename T>
    __device__ __forceinline__ T* data(int local_token_idx) const {
        return reinterpret_cast<T*>(
            (char*)base_ptr + (uint64_t)local_token_idx * token_size);
    }

    // Get global token index for local token i
    __device__ __forceinline__ int global_idx(int local_token_idx) const {
        return global_token_offset + local_token_idx;
    }
};

// ---- TileContext: tile-level view for optional hooks ----
struct TileContext {
    uint32_t tile_id;
    uint32_t tokens_per_tile;
    uint32_t chunks_per_tile;
    uint32_t tokens_per_chunk;
    void* tile_base_ptr;
    uint32_t token_size;

    __device__ __forceinline__ ChunkView get_chunk(int chunk_id) const {
        ChunkView view;
        view.base_ptr = (char*)tile_base_ptr +
                        (uint64_t)chunk_id * tokens_per_chunk * token_size;
        view.token_size = token_size;
        view.size = tokens_per_chunk;
        view.tile_id = tile_id;
        view.chunk_id = chunk_id;
        view.global_token_offset = tile_id * tokens_per_tile + chunk_id * tokens_per_chunk;
        view.lane_id = threadIdx.x % 32;
        return view;
    }
};

// ---- Internal: shared memory state for cross-warp coordination ----
struct _WarpSpecState {
    uint32_t tile_id;
    volatile int chunks_ready;
    volatile int compute_done;
};

// ---- Internal: transfer warp implementation ----
// Per-chunk submission: acquires slots and submits descriptors chunk by chunk,
// polling completion after each chunk to signal compute warps progressively.
__device__ __forceinline__
void _transfer_warp_loop(
    TiledQueue* tq,
    void* gpu_buffer,
    const void* cpu_base,
    uint32_t tile_id,
    uint32_t C,    // tokens_per_chunk
    uint32_t K,    // chunks_per_tile
    uint32_t T,    // tokens_per_tile
    uint32_t token_size,
    _WarpSpecState* state)
{
    const int lane_id = threadIdx.x & 31;

    for (uint32_t chunk = 0; chunk < K; chunk++) {
        // (a) Acquire slots
        uint64_t chunk_base = device::acquire_chunk_slots(tq, C);

        // (a.1) Per-block backpressure: wait until our reserved range
        // [chunk_base, chunk_base+C) won't overwrite unread entries.
        // Critical for multi-block: each block's atomicAdd may advance
        // write_idx far ahead of read_idx under heavy contention.
        if (lane_id == 0) {
            while (true) {
                uint64_t r = *((volatile uint64_t*)&tq->base.read_idx);
                if (chunk_base + C - r <= QUEUE_SIZE) break;
#if __CUDA_ARCH__ >= 700
                __nanosleep(100);
#endif
            }
        }
        __syncwarp();

        // (b) Write + commit descriptors
        device::submit_chunk(tq, chunk_base, tile_id, chunk,
                             C, T, cpu_base, gpu_buffer, token_size);

        // (c) Wait for this chunk's DMA completion
        if (lane_id == 0) {
            uint64_t expected = (uint64_t)(chunk + 1) * C;
            device::wait_chunk_done(tq, tile_id, expected);

            // Signal compute warps
            __threadfence_block();
            state->chunks_ready = chunk + 1;
        }
        __syncwarp();
    }

    // Wait for all compute warps to finish before proceeding to next tile
    if (lane_id == 0) {
        while (state->compute_done < (int)K) {
            // spin
        }
    }
    __syncwarp();
}

// ---- Core: warp-specialized tile processing loop ----
// Template on user's compute functor type.
template<typename ComputeFn>
__device__ void _warp_spec_tile_loop(
    TiledQueue* tq,
    void* gpu_buffer,
    const void* cpu_base,
    ComputeFn compute_fn)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ _WarpSpecState _gfd_state;

    if (threadIdx.x == 0) {
        _gfd_state.chunks_ready = 0;
        _gfd_state.compute_done = 0;
    }
    __syncthreads();

    const uint32_t C = tq->scheduler.tokens_per_chunk;
    const uint32_t K = tq->scheduler.chunks_per_tile;
    const uint32_t T = tq->scheduler.tokens_per_tile;
    const uint32_t total_tiles = tq->scheduler.total_tiles;
    const uint32_t token_size = tq->scheduler.token_size;

    while (true) {
        // === Tile Acquisition (thread 0 only) ===
        if (threadIdx.x == 0) {
            _gfd_state.tile_id = device::acquire_tile(tq);
            _gfd_state.chunks_ready = 0;
            _gfd_state.compute_done = 0;
        }
        __syncthreads();

        if (_gfd_state.tile_id >= total_tiles) break;
        const uint32_t tile_id = _gfd_state.tile_id;

        if (warp_id == 0) {
            // === TRANSFER WARP ===
            _transfer_warp_loop(tq, gpu_buffer, cpu_base,
                                tile_id, C, K, T, token_size, &_gfd_state);
        } else if ((uint32_t)warp_id <= K) {
            // === COMPUTE WARP ===
            const int my_chunk = warp_id - 1;

            // Wait for my chunk's data to arrive
            if (lane_id == 0) {
                while (_gfd_state.chunks_ready <= my_chunk) {
                    // spin
                }
                __threadfence_block();
            }
            __syncwarp();

            // Build ChunkView
            ChunkView view;
            view.base_ptr = (char*)gpu_buffer +
                (uint64_t)(tile_id * T + my_chunk * C) * token_size;
            view.token_size = token_size;
            view.size = C;
            view.tile_id = tile_id;
            view.chunk_id = my_chunk;
            view.global_token_offset = tile_id * T + my_chunk * C;
            view.lane_id = lane_id;

            // === USER COMPUTE ===
            compute_fn(view);

            // Signal completion
            __syncwarp();
            if (lane_id == 0) {
                atomicAdd((int*)&_gfd_state.compute_done, 1);
            }
        }
        // else: extra warps beyond K+1 are idle (shouldn't happen with correct config)

        __syncthreads();
    }
}

// ============================================================
// Double-Buffer (Ping-Pong) Mode
//
// Overlaps Tile N's compute with Tile N+1's transfer.
// Block layout: 1 transfer warp + 2*K compute warps
//   Warps 1..K:   Compute Set A (handles even-indexed tiles for this SM)
//   Warps K+1..2K: Compute Set B (handles odd-indexed tiles for this SM)
//
// Timeline:
//   Transfer: [Tile0 submit+wait] [Tile1 submit+wait] [Tile2 submit+wait] ...
//   Set A:                        [Tile0 compute]      [Tile2 compute]     ...
//   Set B:                                             [Tile1 compute]     ...
// ============================================================

// Shared state for double-buffer mode (two tile slots)
struct _WarpSpecStateDB {
    uint32_t tile_id[2];               // tile IDs for slot 0 and 1
    volatile int chunks_ready[2];      // per-slot chunk progress
    volatile int compute_done[2];      // per-slot compute completion
    volatile int active_slot;          // which slot transfer warp is filling next
    volatile int tiles_dispatched;     // total tiles dispatched so far
};

template<typename ComputeFn>
__device__ void _warp_spec_tile_loop_double_buffer(
    TiledQueue* tq,
    void* gpu_buffer,
    const void* cpu_base,
    ComputeFn compute_fn)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ _WarpSpecStateDB _gfd_db;

    if (threadIdx.x == 0) {
        _gfd_db.chunks_ready[0] = 0;
        _gfd_db.chunks_ready[1] = 0;
        _gfd_db.compute_done[0] = 0;
        _gfd_db.compute_done[1] = 0;
        _gfd_db.active_slot = 0;
        _gfd_db.tiles_dispatched = 0;
        _gfd_db.tile_id[0] = 0xFFFFFFFF;
        _gfd_db.tile_id[1] = 0xFFFFFFFF;
    }
    __syncthreads();

    const uint32_t C = tq->scheduler.tokens_per_chunk;
    const uint32_t K = tq->scheduler.chunks_per_tile;
    const uint32_t T = tq->scheduler.tokens_per_tile;
    const uint32_t total_tiles = tq->scheduler.total_tiles;
    const uint32_t token_size = tq->scheduler.token_size;

    if (warp_id == 0) {
        // === TRANSFER WARP ===
        // Alternates between slot 0 and slot 1.
        //
        // Bug fixes applied:
        //   1. acquire_tile must be called by lane 0 only, then broadcast
        //      via __shfl_sync. Previously all 32 lanes called atomicAdd,
        //      consuming 32 tile IDs per iteration (31 wasted) and causing
        //      warp divergence on the termination check — subsequent
        //      __shfl_sync(0xFFFFFFFF) with inactive lanes is UB / deadlock.
        //   2. Before writing termination sentinels, wait for the last
        //      dispatched slot's compute to finish so no tiles are skipped.
        //   3. Add __threadfence_block after sentinel writes to chunks_ready
        //      so compute warps observe consistent termination state.
        while (true) {
            // Acquire next tile — lane 0 only, then broadcast to all lanes.
            // This avoids consuming 32 tile IDs per iteration and prevents
            // warp divergence that would make warp-collective ops UB.
            uint32_t tile_id;
            if (lane_id == 0) {
                tile_id = device::acquire_tile(tq);
            }
            tile_id = __shfl_sync(0xFFFFFFFF, tile_id, 0);

            if (tile_id >= total_tiles) {
                // Wait for the last in-flight compute to finish before
                // writing termination sentinels, so the last tile's
                // compute warps can complete without seeing a premature
                // sentinel overwrite on chunks_ready.
                if (lane_id == 0) {
                    if (_gfd_db.tiles_dispatched > 0) {
                        int last_slot = (_gfd_db.tiles_dispatched - 1) & 1;
                        while (_gfd_db.compute_done[last_slot] < (int)K) {}
                    }
                    // Signal termination: set both slots to invalid
                    _gfd_db.tile_id[0] = 0xFFFFFFFF;
                    _gfd_db.tile_id[1] = 0xFFFFFFFF;
                    __threadfence_block();
                    _gfd_db.chunks_ready[0] = (int)K + 1;  // sentinel: exit
                    _gfd_db.chunks_ready[1] = (int)K + 1;
                    __threadfence_block();
                }
                __syncwarp();
                break;  // All lanes break together — no divergence
            }

            int slot = _gfd_db.tiles_dispatched & 1;

            // Wait for previous compute on this slot to finish (if any)
            if (lane_id == 0) {
                if (_gfd_db.tiles_dispatched >= 2) {
                    while (_gfd_db.compute_done[slot] < (int)K) {}
                }
                _gfd_db.tile_id[slot] = tile_id;
                _gfd_db.chunks_ready[slot] = 0;
                _gfd_db.compute_done[slot] = 0;
                __threadfence_block();
                _gfd_db.tiles_dispatched++;
            }
            __syncwarp();

            // Submit all chunks for this tile
            for (uint32_t chunk = 0; chunk < K; chunk++) {
                uint64_t chunk_base = device::acquire_chunk_slots(tq, C);
                device::submit_chunk(tq, chunk_base, tile_id, chunk,
                                     C, T, cpu_base, gpu_buffer, token_size);

                // Wait for chunk DMA completion
                if (lane_id == 0) {
                    uint64_t expected = (uint64_t)(chunk + 1) * C;
                    device::wait_chunk_done(tq, tile_id, expected);
                    __threadfence_block();
                    _gfd_db.chunks_ready[slot] = chunk + 1;
                }
                __syncwarp();
            }
        }
    } else {
        // === COMPUTE WARPS ===
        // Warps 1..K handle slot 0 (Set A), warps K+1..2K handle slot 1 (Set B)
        int my_slot, my_chunk;
        if ((uint32_t)warp_id <= K) {
            my_slot = 0;
            my_chunk = warp_id - 1;
        } else if ((uint32_t)warp_id <= 2 * K) {
            my_slot = 1;
            my_chunk = warp_id - K - 1;
        } else {
            return;  // excess warps
        }

        int local_tile_count = 0;  // how many tiles this slot has processed

        while (true) {
            // Wait for transfer warp to dispatch a tile into our slot.
            // Our Nth tile corresponds to tiles_dispatched index = N*2 + my_slot
            // (slot 0 gets dispatches 0, 2, 4, ...; slot 1 gets 1, 3, 5, ...)
            int target_dispatch = local_tile_count * 2 + my_slot;

            if (lane_id == 0) {
                // Wait until tiles_dispatched > target_dispatch (new tile assigned)
                // or termination sentinel
                while (true) {
                    int dispatched = _gfd_db.tiles_dispatched;
                    if (dispatched > target_dispatch) break;
                    // Also check termination
                    if (_gfd_db.chunks_ready[my_slot] == (int)K + 1) break;
                }
            }
            __syncwarp();

            // Check termination
            if (_gfd_db.chunks_ready[my_slot] == (int)K + 1) break;
            uint32_t tile_id = _gfd_db.tile_id[my_slot];
            if (tile_id == 0xFFFFFFFF) break;

            // Wait for my specific chunk's data to arrive
            if (lane_id == 0) {
                while (_gfd_db.chunks_ready[my_slot] <= my_chunk) {
                    // spin until chunk data is ready
                }
                __threadfence_block();
            }
            __syncwarp();

            // Build ChunkView and compute
            ChunkView view;
            view.base_ptr = (char*)gpu_buffer +
                (uint64_t)(tile_id * T + my_chunk * C) * token_size;
            view.token_size = token_size;
            view.size = C;
            view.tile_id = tile_id;
            view.chunk_id = my_chunk;
            view.global_token_offset = tile_id * T + my_chunk * C;
            view.lane_id = lane_id;

            compute_fn(view);

            __syncwarp();
            if (lane_id == 0) {
                atomicAdd((int*)&_gfd_db.compute_done[my_slot], 1);
            }

            local_tile_count++;
        }
    }
}

}  // namespace warp_spec
}  // namespace gfd

// ============================================================
// Kernel Generation Macros
// ============================================================

// Generate a complete warp-specialized kernel from a compute functor type.
// Single-buffer mode: block = (1 + K) warps
#define GFD_WARP_SPEC_KERNEL(kernel_name, ComputeFnType)                    \
__global__ void kernel_name(                                                \
    gfd::TiledQueue* tq,                                                    \
    void* gpu_buffer,                                                       \
    const void* cpu_base,                                                   \
    ComputeFnType compute_fn)                                               \
{                                                                           \
    gfd::warp_spec::_warp_spec_tile_loop(tq, gpu_buffer, cpu_base,          \
                                         compute_fn);                       \
}

// Double-buffer mode: block = (1 + 2*K) warps
// Overlaps tile N compute with tile N+1 transfer for better latency hiding.
#define GFD_WARP_SPEC_KERNEL_DB(kernel_name, ComputeFnType)                 \
__global__ void kernel_name(                                                \
    gfd::TiledQueue* tq,                                                    \
    void* gpu_buffer,                                                       \
    const void* cpu_base,                                                   \
    ComputeFnType compute_fn)                                               \
{                                                                           \
    gfd::warp_spec::_warp_spec_tile_loop_double_buffer(                     \
        tq, gpu_buffer, cpu_base, compute_fn);                              \
}
