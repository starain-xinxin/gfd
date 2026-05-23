#pragma once

#include "gfd/tiled_queue.h"

// ============================================================
// GFD Device Primitives (Layer 1)
//
// Low-level __device__ building blocks for warp-specialized kernels.
// These are warp-collective operations: all threads in a warp must
// participate (or be explicitly masked).
//
// For most users, prefer the Layer 2 Framework API (warp_spec.cuh)
// which composes these primitives automatically.
// ============================================================

namespace gfd {
namespace device {

// ---- Tile Acquisition ----
// Atomically acquire the next tile_id. Returns >= scheduler.total_tiles
// when all tiles have been dispatched (kernel should exit).
// Only one thread per block should call this, then broadcast.
__device__ __forceinline__
uint32_t acquire_tile(TiledQueue* tq) {
    return atomicAdd((unsigned int*)&tq->scheduler.next_tile, 1);
}

// ---- Chunk Slot Acquisition (warp-collective) ----
// Lane 0 performs atomicAdd to get `count` contiguous slots.
// Result is broadcast to all lanes via __shfl_sync.
// All 32 lanes in the warp must be active.
__device__ __forceinline__
uint64_t acquire_chunk_slots(TiledQueue* tq, uint32_t count) {
    uint64_t base = 0;
    if ((threadIdx.x & 31) == 0) {
        base = atomicAdd((unsigned long long*)&tq->base.write_idx,
                         (unsigned long long)count);
    }
    base = __shfl_sync(0xFFFFFFFF, base, 0);
    return base;
}

// ---- Write Chunk Descriptors (warp-collective) ----
// A single warp writes `tokens_per_chunk` descriptors starting at slot_base.
// If tokens_per_chunk > 32, threads loop to cover all entries.
// Sets FLAG_LAST_IN_TILE on the last entry of the chunk to trigger CPU flush.
__device__ __forceinline__
void write_chunk(
    TiledQueue* tq,
    uint64_t slot_base,
    uint32_t tile_id,
    uint32_t chunk_id,
    uint32_t tokens_per_chunk,
    uint32_t tokens_per_tile,
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size)
{
    const int lane_id = threadIdx.x & 31;
    const uint32_t T = tokens_per_tile;
    const uint32_t C = tokens_per_chunk;

    const uint32_t K = T / C;  // chunks per tile
    for (uint32_t i = lane_id; i < C; i += 32) {
        uint64_t slot = slot_base + i;
        Descriptor* desc = &tq->base.entries[slot % QUEUE_SIZE];

        uint32_t local_token = chunk_id * C + i;
        uint32_t global_idx = tile_id * T + local_token;

        desc->src_addr = (uint64_t)cpu_base + (uint64_t)global_idx * token_size;
        desc->dst_addr = (uint64_t)gpu_base + (uint64_t)global_idx * token_size;
        desc->size = token_size;
        desc->user_data = ((uint64_t)tile_id << 32) | global_idx;
        // Mark chunk boundary; also mark tile boundary on last entry of last chunk
        uint32_t flags = (i == C - 1) ? FLAG_LAST_IN_TILE : FLAG_NONE;
        if (i == C - 1 && chunk_id == K - 1) flags |= FLAG_LAST_CHUNK_IN_TILE;
        desc->flags = flags;
    }
}

// ---- Commit Chunk (warp-collective) ----
// Two-phase commit with system fences:
//   1. __threadfence_system() — descriptor fields visible to CPU
//   2. Write sequence numbers (commit markers)
//   3. __threadfence_system() — sequences visible to CPU poller
__device__ __forceinline__
void commit_chunk(TiledQueue* tq, uint64_t slot_base, uint32_t count) {
    const int lane_id = threadIdx.x & 31;

    // Pre-commit fence
    __threadfence_system();
    __syncwarp();

    // Write sequence (slot+1) to mark entries as ready
    for (uint32_t i = lane_id; i < count; i += 32) {
        uint64_t slot = slot_base + i;
        tq->base.entries[slot % QUEUE_SIZE].sequence = slot + 1;
    }

    // Post-commit fence
    __threadfence_system();
    __syncwarp();
}

// ---- Wait for Chunk Completion ----
// Polls tile_chunk_done[tile_id] until cumulative done count >= expected.
// Only one thread should call this (typically lane 0 of transfer warp).
//
// When d_tile_chunk_done is available, polls device memory (L2 cache ~10ns).
// Otherwise falls back to host-mapped memory (PCIe ~1500ns per load).
//
// Opt4: Uses __nanosleep (sm_70+) exponential backoff to free SM execution
// slots for other warps while waiting. Tight-spins for the first 64 iterations
// (covers ~640ns for L2, ~96us for PCIe), then backs off progressively.
__device__ __forceinline__
void wait_chunk_done(TiledQueue* tq, uint32_t tile_id, uint64_t expected) {
    volatile uint64_t* signal = tq->d_tile_chunk_done
        ? (volatile uint64_t*)&tq->d_tile_chunk_done[tile_id]
        : (volatile uint64_t*)&tq->tile_chunk_done[tile_id];
    int spin_count = 0;
    while (true) {
        uint64_t done = *signal;
        if (done >= expected) break;
#if __CUDA_ARCH__ >= 700
        if (++spin_count > 64) {
            // Exponential backoff: 100ns for moderate waits, 1000ns for long waits.
            // Frees SM execution slots so other warps (e.g. compute warps) can progress.
            __nanosleep(spin_count < 256 ? 100 : 1000);
        }
#endif
    }
}

// ---- Backpressure: Wait for Queue Space ----
// Spins until there are at least `needed * 2` free slots in the ring buffer.
// Prevents queue overflow when GPU produces faster than CPU consumes.
// Warp-collective: all lanes must participate.
__device__ __forceinline__
void wait_queue_space(TiledQueue* tq, uint32_t needed) {
    if ((threadIdx.x & 31) == 0) {
        while (true) {
            uint64_t w = tq->base.write_idx;
            uint64_t r = *((volatile uint64_t*)&tq->base.read_idx);
            if (w - r < QUEUE_SIZE - needed * 2) break;
        }
    }
    __syncwarp();
}

// ---- Submit and Commit Chunk (convenience) ----
// Combines write_chunk + commit_chunk in one call.
__device__ __forceinline__
void submit_chunk(
    TiledQueue* tq,
    uint64_t slot_base,
    uint32_t tile_id,
    uint32_t chunk_id,
    uint32_t tokens_per_chunk,
    uint32_t tokens_per_tile,
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size)
{
    write_chunk(tq, slot_base, tile_id, chunk_id,
                tokens_per_chunk, tokens_per_tile,
                cpu_base, gpu_base, token_size);
    commit_chunk(tq, slot_base, tokens_per_chunk);
}

// ---- Submit Entire Tile (warp-collective) ----
// Submits ALL T descriptors for a tile in a single acquire+write+commit cycle.
// Reduces system fences from 2*K to 2 total per tile.
// The poller receives all descriptors at once → larger DMA batches.
__device__ __forceinline__
void submit_tile(
    TiledQueue* tq,
    uint64_t slot_base,
    uint32_t tile_id,
    uint32_t T,              // tokens_per_tile (total entries)
    uint32_t C,              // tokens_per_chunk (for FLAG_LAST_IN_TILE marking)
    const void* cpu_base,
    void* gpu_base,
    uint32_t token_size)
{
    const int lane_id = threadIdx.x & 31;

    // Write ALL T descriptors (loop if T > 32)
    for (uint32_t i = lane_id; i < T; i += 32) {
        uint64_t slot = slot_base + i;
        Descriptor* desc = &tq->base.entries[slot % QUEUE_SIZE];

        uint32_t global_idx = tile_id * T + i;
        desc->src_addr = (uint64_t)cpu_base + (uint64_t)global_idx * token_size;
        desc->dst_addr = (uint64_t)gpu_base + (uint64_t)global_idx * token_size;
        desc->size = token_size;
        desc->user_data = ((uint64_t)tile_id << 32) | global_idx;
        // Mark last entry of each chunk with FLAG_LAST_IN_TILE for progress tracking.
        // Mark the final entry of the entire tile with FLAG_LAST_CHUNK_IN_TILE
        // to trigger poller flush (larger DMA batches).
        uint32_t flags = FLAG_NONE;
        if ((i % C) == C - 1) flags |= FLAG_LAST_IN_TILE;
        if (i == T - 1) flags |= FLAG_LAST_CHUNK_IN_TILE;
        desc->flags = flags;
    }

    // Single commit for all T entries (2 system fences total)
    commit_chunk(tq, slot_base, T);
}

}  // namespace device
}  // namespace gfd
