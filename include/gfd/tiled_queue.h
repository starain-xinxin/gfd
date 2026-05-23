#pragma once

#include "gfd/descriptor_queue.h"
#include <cstdint>

namespace gfd {

constexpr int MAX_TILES = 1024;
constexpr uint32_t FLAG_LAST_IN_TILE = 4;           // Chunk boundary marker (every C-th entry)
constexpr uint32_t FLAG_LAST_CHUNK_IN_TILE = 8;     // Tile boundary marker (final entry of tile, triggers poller flush)

// TileScheduler: global tile dispatch and configuration.
// Allocated in host-mapped memory for GPU atomicAdd access.
struct __align__(64) TileScheduler {
    volatile uint32_t next_tile;      // GPU atomicAdd to acquire tile_id
    uint32_t total_tiles;             // Total number of tiles in this workload
    uint32_t tokens_per_tile;         // Tokens assigned to each tile
    uint32_t tokens_per_chunk;        // Tokens per sub-tile chunk
    uint32_t chunks_per_tile;         // = tokens_per_tile / tokens_per_chunk
    uint32_t token_size;              // Bytes per token
    uint8_t _pad[40];                 // Pad to 64 bytes
};

static_assert(sizeof(TileScheduler) == 64, "TileScheduler must be 64 bytes");

// TiledQueue: extends DescriptorQueue with tile scheduling and per-tile
// completion signaling for warp-specialized communication-computation overlap.
//
// Allocation: cudaHostAlloc with cudaHostAllocMapped for CPU+GPU shared access.
//
// Flow:
//   1. GPU Transfer Warp: atomicAdd(&scheduler.next_tile) to get tile_id
//   2. GPU Transfer Warp: atomicAdd(&base.write_idx, C) per chunk to get slots
//   3. GPU Transfer Warp: write descriptors + fence + commit sequence
//   4. CPU Poller: processes descriptors, submits DMA, writes tile_chunk_done
//   5. GPU Transfer Warp: polls tile_chunk_done[tile_id] for chunk completion
//   6. GPU Compute Warps: compute on arrived chunk data
struct TiledQueue {
    DescriptorQueue base;                          // Lock-free descriptor ring buffer
    TileScheduler scheduler;                       // Global tile dispatch
    volatile uint64_t tile_chunk_done[MAX_TILES];  // Cumulative tokens completed per tile (host-mapped fallback)
    uint64_t* d_tile_chunk_done;                   // Device memory signal buffer (nullptr = use host-mapped)
};

}  // namespace gfd
