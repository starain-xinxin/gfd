#include "gfd/cpu_polling.h"
#include "gfd/sg_task_queue.h"
#include "gfd/log.h"
#include <cuda.h>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace gfd {

// 8-bit radix sort for SGEntry arrays, keyed by a uint64_t extractor.
// Uses 4-pass (LSB) radix on the lower 32 bits for typical address sorting.
// Falls back to std::sort for very small arrays.
template<typename KeyFn>
static void radix_sort_sg(SGEntry* arr, int count, SGEntry* tmp, KeyFn key_fn) {
    if (count <= 64) {
        std::sort(arr, arr + count,
                  [&](const SGEntry& a, const SGEntry& b) {
                      return key_fn(a) < key_fn(b);
                  });
        return;
    }

    constexpr int RADIX_BITS = 8;
    constexpr int RADIX_SIZE = 1 << RADIX_BITS;
    constexpr int RADIX_MASK = RADIX_SIZE - 1;

    SGEntry* src = arr;
    SGEntry* dst = tmp;

    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * RADIX_BITS;

        int counts[RADIX_SIZE] = {};
        for (int i = 0; i < count; i++) {
            uint64_t k = key_fn(src[i]);
            int bucket = (k >> shift) & RADIX_MASK;
            counts[bucket]++;
        }

        int offsets[RADIX_SIZE];
        offsets[0] = 0;
        for (int i = 1; i < RADIX_SIZE; i++)
            offsets[i] = offsets[i-1] + counts[i-1];

        for (int i = 0; i < count; i++) {
            uint64_t k = key_fn(src[i]);
            int bucket = (k >> shift) & RADIX_MASK;
            dst[offsets[bucket]++] = src[i];
        }

        SGEntry* t = src; src = dst; dst = t;
    }

    if (src != arr) {
        memcpy(arr, src, count * sizeof(SGEntry));
    }
}

// ============================================================
// Per-tile DMA processing (P0 + P1):
// Groups entries by tile_id, submits separate DMA per tile,
// and appends a CE write-back to tile_done[tile_id] after each.
//
// Supports interleaved mode: entries from different tiles can be
// mixed in the queue (from concurrent atomicAdd by multiple SMs).
// tile_done[tile_id] stores cumulative token count (not slot index).
// ============================================================
void CpuPollingThread::process_batch_tiled(Descriptor* batch, int count, uint64_t cur_read) {
    // Group entries by tile_id (extract from user_data upper 32 bits).
    // Entries may be interleaved from multiple tiles (concurrent SM writes).
    //
    // Opt1: Compact bucket storage — avoid O(MAX_TILES) initialization.
    // Uses a small lookup table cleared lazily (only touched tile IDs are reset)
    // and a flat index array partitioned by per-bucket offsets.

    // Lazy-clear bucket map: track which tile_ids were written so we only
    // clear those at the end. Persisted across calls as thread-local static.
    static thread_local int bucket_map_storage[MAX_TILES];
    static thread_local bool bucket_map_initialized = false;
    if (!bucket_map_initialized) {
        for (int i = 0; i < MAX_TILES; i++) bucket_map_storage[i] = -1;
        bucket_map_initialized = true;
    }

    // Compact per-bucket metadata (sized to actual active tiles, not MAX_TILES)
    uint32_t active_tile_ids[MAX_BATCH_SIZE];  // at most count unique tiles
    int tile_entry_counts[MAX_BATCH_SIZE] = {};
    int num_active_tiles = 0;

    // Flat index array: all per-tile entry indices packed contiguously
    int flat_indices[MAX_BATCH_SIZE];
    // First pass: count entries per bucket
    for (int i = 0; i < count; i++) {
        uint32_t tile_id = (uint32_t)(batch[i].user_data >> 32);
        if (tile_id >= MAX_TILES) continue;

        int bucket_idx = bucket_map_storage[tile_id];
        if (bucket_idx == -1) {
            bucket_idx = num_active_tiles;
            bucket_map_storage[tile_id] = bucket_idx;
            active_tile_ids[num_active_tiles] = tile_id;
            num_active_tiles++;
        }
        tile_entry_counts[bucket_idx]++;
    }

    // Compute offsets into flat_indices (exclusive prefix sum)
    int tile_offsets[MAX_BATCH_SIZE];
    tile_offsets[0] = 0;
    for (int b = 1; b < num_active_tiles; b++)
        tile_offsets[b] = tile_offsets[b - 1] + tile_entry_counts[b - 1];

    // Second pass: scatter entry indices into flat_indices
    int fill_pos[MAX_BATCH_SIZE];
    for (int b = 0; b < num_active_tiles; b++) fill_pos[b] = tile_offsets[b];
    for (int i = 0; i < count; i++) {
        uint32_t tile_id = (uint32_t)(batch[i].user_data >> 32);
        if (tile_id >= MAX_TILES) continue;
        int bucket_idx = bucket_map_storage[tile_id];
        flat_indices[fill_pos[bucket_idx]++] = i;
    }

    // Lazy-clear: reset only the tile_ids we touched
    for (int b = 0; b < num_active_tiles; b++)
        bucket_map_storage[active_tile_ids[b]] = -1;

    // Build all SG entries and check cross-tile contiguity for mega-DMA
    SGEntry all_sg_entries[MAX_BATCH_SIZE];
    int total_sg_count = 0;
    size_t total_bytes = 0;
    int tile_sg_offsets[MAX_BATCH_SIZE];   // start offset in all_sg_entries for each bucket
    int tile_sg_counts[MAX_BATCH_SIZE];    // entry count per bucket

    for (int b = 0; b < num_active_tiles; b++) {
        tile_sg_offsets[b] = total_sg_count;
        int tile_count = tile_entry_counts[b];
        tile_sg_counts[b] = tile_count;
        int base_idx = tile_offsets[b];
        for (int i = 0; i < tile_count; i++) {
            int idx = flat_indices[base_idx + i];
            all_sg_entries[total_sg_count].dst = static_cast<CUdeviceptr>(batch[idx].dst_addr);
            all_sg_entries[total_sg_count].src = reinterpret_cast<const void*>(batch[idx].src_addr);
            all_sg_entries[total_sg_count].size = batch[idx].size;
            total_bytes += batch[idx].size;
            total_sg_count++;
        }
    }
    total_bytes_copied_.fetch_add(total_bytes, std::memory_order_relaxed);

    // Check if ALL entries across tiles are fully contiguous (common when
    // submit_tile sends sequential descriptors for sequential tiles)
    bool all_contiguous = (total_sg_count > 1);
    if (all_contiguous) {
        for (int i = 1; i < total_sg_count; i++) {
            bool d_contig = (all_sg_entries[i].dst == all_sg_entries[i-1].dst + all_sg_entries[i-1].size);
            bool s_contig = (all_sg_entries[i].src ==
                (const void*)((const char*)all_sg_entries[i-1].src + all_sg_entries[i-1].size));
            if (!d_contig || !s_contig) { all_contiguous = false; break; }
        }
    }

    // Update per-tile progress counters
    if (tile_signal_buf_) {
        for (int b = 0; b < num_active_tiles; b++) {
            uint32_t tile_id = active_tile_ids[b];
            int tile_count = tile_sg_counts[b];
            if (tile_id < MAX_TILES) {
                tile_progress_[tile_id] += tile_count;
                tile_signal_buf_[tile_id] = tile_progress_[tile_id];
            }
        }
    }

    // Submit data DMAs across CE channels (normal round-robin distribution)
    if (all_contiguous && total_sg_count > 1) {
        // Mega-DMA: single coalesced entry
        SGEntry mega;
        mega.dst = all_sg_entries[0].dst;
        mega.src = all_sg_entries[0].src;
        mega.size = total_bytes;
        ce_manager_.submit_scatter_gather(&mega, 1);
    } else {
        // Per-tile DMA submissions
        for (int b = 0; b < num_active_tiles; b++) {
            int offset = tile_sg_offsets[b];
            int tile_count = tile_sg_counts[b];
            SGEntry* sg_entries = &all_sg_entries[offset];

            // Opt3: single-entry buckets are trivially a single DMA — skip contiguity check
            if (tile_count == 1) {
                ce_manager_.submit_scatter_gather(sg_entries, 1);
                continue;
            }

            bool fully_contiguous = true;
            for (int i = 1; i < tile_count; i++) {
                bool d_contig = (sg_entries[i].dst == sg_entries[i-1].dst + sg_entries[i-1].size);
                bool s_contig = (sg_entries[i].src ==
                    (const void*)((const char*)sg_entries[i-1].src + sg_entries[i-1].size));
                if (!d_contig || !s_contig) { fully_contiguous = false; break; }
            }

            if (fully_contiguous) {
                SGEntry coalesced;
                coalesced.dst = sg_entries[0].dst;
                coalesced.src = sg_entries[0].src;
                size_t tile_bytes = 0;
                for (int i = 0; i < tile_count; i++) tile_bytes += sg_entries[i].size;
                coalesced.size = tile_bytes;
                ce_manager_.submit_scatter_gather(&coalesced, 1);
            } else {
                ce_manager_.submit_scatter_gather(sg_entries, tile_count);
            }
        }
    }

    // Write signal values AFTER all data DMAs, using the dedicated signal stream.
    // For device memory signals: make_stream_wait_on_all ensures the signal stream
    // won't execute until all CE data channels complete (GPU-side ordering, no CPU block).
    // For host-mapped signals: use CE inline submission (no ordering issue since
    // PCIe polling latency ~1500ns naturally masks any race).
    if (tile_signal_buf_) {
        bool has_device_signals = tiled_queue_ && tiled_queue_->d_tile_chunk_done;

        if (has_device_signals && signal_stream_) {
            // GPU-side barrier: signal stream waits for all dirty CE channels
            ce_manager_.make_stream_wait_on_all(signal_stream_);

            // Opt2: Batch signal writes when >4 active tiles.
            // For contiguous tile ID ranges, issue a single cuMemcpyHtoDAsync.
            // For sparse or small tile counts, use individual writes.
            if (num_active_tiles <= 4) {
                // Small count: individual writes (driver overhead amortized)
                for (int b = 0; b < num_active_tiles; b++) {
                    uint32_t tile_id = active_tile_ids[b];
                    if (tile_id < MAX_TILES) {
                        cuMemcpyHtoDAsync(
                            (CUdeviceptr)&tiled_queue_->d_tile_chunk_done[tile_id],
                            &tile_signal_buf_[tile_id],
                            sizeof(uint64_t),
                            signal_stream_);
                    }
                }
            } else {
                // Find min/max tile_id to check for contiguous range
                uint32_t min_tid = active_tile_ids[0], max_tid = active_tile_ids[0];
                for (int b = 1; b < num_active_tiles; b++) {
                    if (active_tile_ids[b] < min_tid) min_tid = active_tile_ids[b];
                    if (active_tile_ids[b] > max_tid) max_tid = active_tile_ids[b];
                }
                uint32_t range = max_tid - min_tid + 1;

                if (range == (uint32_t)num_active_tiles && max_tid < MAX_TILES) {
                    // Contiguous tile ID range: single DMA covering the whole span.
                    // tile_signal_buf_ is indexed by tile_id, so the source range
                    // [min_tid..max_tid] is already contiguous in the host buffer.
                    cuMemcpyHtoDAsync(
                        (CUdeviceptr)&tiled_queue_->d_tile_chunk_done[min_tid],
                        &tile_signal_buf_[min_tid],
                        range * sizeof(uint64_t),
                        signal_stream_);
                } else {
                    // Sparse tile IDs: fall back to individual writes
                    for (int b = 0; b < num_active_tiles; b++) {
                        uint32_t tile_id = active_tile_ids[b];
                        if (tile_id < MAX_TILES) {
                            cuMemcpyHtoDAsync(
                                (CUdeviceptr)&tiled_queue_->d_tile_chunk_done[tile_id],
                                &tile_signal_buf_[tile_id],
                                sizeof(uint64_t),
                                signal_stream_);
                        }
                    }
                }
            }
        } else {
            // Host-mapped path: CE inline submission (ordering not critical)
            SGEntry signal_entries[MAX_TILES];
            int num_signals = 0;
            for (int b = 0; b < num_active_tiles; b++) {
                uint32_t tile_id = active_tile_ids[b];
                if (tile_id < MAX_TILES) {
                    signal_entries[num_signals].dst = (CUdeviceptr)&tiled_queue_->tile_chunk_done[tile_id];
                    signal_entries[num_signals].src = &tile_signal_buf_[tile_id];
                    signal_entries[num_signals].size = sizeof(uint64_t);
                    num_signals++;
                }
            }
            if (num_signals > 0) {
                ce_manager_.submit_scatter_gather(signal_entries, num_signals);
            }
        }
    }

    descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
    batches_submitted_.fetch_add(num_active_tiles, std::memory_order_relaxed);
}

// ============================================================
// SG mode: per-list DMA processing
//
// Groups descriptors by list_id (from user_data upper 32 bits),
// submits DMA per list, and signals d_list_done[list_id] + lists_completed.
// Reuses the same CE pipeline as process_batch_tiled.
// ============================================================
void CpuPollingThread::process_batch_sg(Descriptor* batch, int count, uint64_t cur_read) {
    // Group entries by list_id (same bucket logic as process_batch_tiled)
    static thread_local int sg_bucket_map[MAX_SG_LISTS];
    static thread_local bool sg_bucket_map_initialized = false;
    if (!sg_bucket_map_initialized) {
        for (int i = 0; i < MAX_SG_LISTS; i++) sg_bucket_map[i] = -1;
        sg_bucket_map_initialized = true;
    }

    uint32_t active_list_ids[MAX_BATCH_SIZE];
    int list_entry_counts[MAX_BATCH_SIZE] = {};
    int num_active_lists = 0;

    int flat_indices[MAX_BATCH_SIZE];

    // First pass: count entries per list
    for (int i = 0; i < count; i++) {
        uint32_t list_id = (uint32_t)(batch[i].user_data >> 32);
        if (list_id >= (uint32_t)MAX_SG_LISTS) continue;

        int bucket_idx = sg_bucket_map[list_id];
        if (bucket_idx == -1) {
            bucket_idx = num_active_lists;
            sg_bucket_map[list_id] = bucket_idx;
            active_list_ids[num_active_lists] = list_id;
            num_active_lists++;
        }
        list_entry_counts[bucket_idx]++;
    }

    // Compute offsets
    int list_offsets[MAX_BATCH_SIZE];
    list_offsets[0] = 0;
    for (int b = 1; b < num_active_lists; b++)
        list_offsets[b] = list_offsets[b - 1] + list_entry_counts[b - 1];

    // Second pass: scatter indices
    int fill_pos[MAX_BATCH_SIZE];
    for (int b = 0; b < num_active_lists; b++) fill_pos[b] = list_offsets[b];
    for (int i = 0; i < count; i++) {
        uint32_t list_id = (uint32_t)(batch[i].user_data >> 32);
        if (list_id >= (uint32_t)MAX_SG_LISTS) continue;
        int bucket_idx = sg_bucket_map[list_id];
        flat_indices[fill_pos[bucket_idx]++] = i;
    }

    // Lazy-clear
    for (int b = 0; b < num_active_lists; b++)
        sg_bucket_map[active_list_ids[b]] = -1;

    // Build SG entries and submit DMA per list
    SGEntry all_sg[MAX_BATCH_SIZE];
    int total_sg = 0;
    size_t total_bytes = 0;

    for (int b = 0; b < num_active_lists; b++) {
        int lcount = list_entry_counts[b];
        int base_idx = list_offsets[b];

        SGEntry* sg_entries = &all_sg[total_sg];
        for (int i = 0; i < lcount; i++) {
            int idx = flat_indices[base_idx + i];
            sg_entries[i].dst = static_cast<CUdeviceptr>(batch[idx].dst_addr);
            sg_entries[i].src = reinterpret_cast<const void*>(batch[idx].src_addr);
            sg_entries[i].size = batch[idx].size;
            total_bytes += batch[idx].size;
        }

        // Check contiguity for coalescing
        bool fully_contiguous = (lcount > 1);
        if (fully_contiguous) {
            for (int i = 1; i < lcount; i++) {
                bool d_contig = (sg_entries[i].dst == sg_entries[i-1].dst + sg_entries[i-1].size);
                bool s_contig = (sg_entries[i].src ==
                    (const void*)((const char*)sg_entries[i-1].src + sg_entries[i-1].size));
                if (!d_contig || !s_contig) { fully_contiguous = false; break; }
            }
        }

        if (fully_contiguous && lcount > 1) {
            SGEntry coalesced;
            coalesced.dst = sg_entries[0].dst;
            coalesced.src = sg_entries[0].src;
            size_t list_bytes = 0;
            for (int i = 0; i < lcount; i++) list_bytes += sg_entries[i].size;
            coalesced.size = list_bytes;
            ce_manager_.submit_scatter_gather(&coalesced, 1);
        } else {
            ce_manager_.submit_scatter_gather(sg_entries, lcount);
        }

        total_sg += lcount;
    }
    total_bytes_copied_.fetch_add(total_bytes, std::memory_order_relaxed);

    // Signal completion: write d_list_done[list_id] and increment lists_completed
    if (sg_queue_ && sg_queue_->d_list_done && signal_stream_) {
        // GPU-side barrier: signal stream waits for all CE data channels
        ce_manager_.make_stream_wait_on_all(signal_stream_);

        // Allocate pinned signal values if needed (reuse tile_signal_buf_ or dedicate)
        // For SG mode, write 1 to d_list_done[list_id] to indicate completion
        for (int b = 0; b < num_active_lists; b++) {
            uint32_t list_id = active_list_ids[b];
            if (list_id < (uint32_t)MAX_SG_LISTS && tile_signal_buf_) {
                // Reuse tile_signal_buf_ slot for the signal value
                tile_signal_buf_[list_id] = 1;
                cuMemcpyHtoDAsync(
                    (CUdeviceptr)&sg_queue_->d_list_done[list_id],
                    &tile_signal_buf_[list_id],
                    sizeof(uint64_t),
                    signal_stream_);
            }
        }

        // Increment lists_completed (host-mapped, visible to GPU)
        if (sg_queue_) {
            uint64_t old_val = sg_queue_->lists_completed;
            sg_queue_->lists_completed = old_val + num_active_lists;
            __atomic_thread_fence(__ATOMIC_RELEASE);
        }
    } else if (sg_queue_) {
        // Host-mapped fallback: write directly after CE wait
        ce_manager_.wait_completion();
        for (int b = 0; b < num_active_lists; b++) {
            uint32_t list_id = active_list_ids[b];
            if (list_id < (uint32_t)MAX_SG_LISTS && sg_queue_->d_list_done) {
                uint64_t val = 1;
                cudaMemcpy(&sg_queue_->d_list_done[list_id], &val,
                           sizeof(uint64_t), cudaMemcpyHostToDevice);
            }
        }
        uint64_t old_val = sg_queue_->lists_completed;
        sg_queue_->lists_completed = old_val + num_active_lists;
        __atomic_thread_fence(__ATOMIC_RELEASE);
    }

    descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
    batches_submitted_.fetch_add(num_active_lists, std::memory_order_relaxed);
}

void CpuPollingThread::flush_batch(Descriptor* batch, int& batch_count, uint32_t& or_flags) {
    if (batch_count == 0) return;
    uint64_t cur_read = queue_->read_idx;

    bool has_tile_markers = tiled_queue_ && (or_flags & (FLAG_LAST_IN_TILE | FLAG_LAST_CHUNK_IN_TILE));

    // SG mode: route to per-list processing when sg_queue_ is set
    bool sg_mode = sg_queue_ && (or_flags & FLAG_LAST_IN_TILE);
    if (sg_mode && use_ce_) {
        process_batch_sg(batch, batch_count, cur_read);
        has_async_dma_ = true;
        latest_async_read_ = cur_read;
        batch_count = 0;
        or_flags = 0;
        return;
    }

    if (use_ce_) {
        // Always use per-tile DMA path when tiled_queue_ is set, even for
        // partial batches without FLAG_LAST_IN_TILE. This ensures tile_progress_
        // is always updated correctly. Without this, partial batches from gap
        // timeouts skip tile progress tracking, causing multi-block deadlocks
        // where the GPU waits for a cumulative count that never arrives.
        bool use_per_tile_dma = false;
        if (tiled_queue_ && tile_signal_buf_) {
            use_per_tile_dma = true;
        }

        if (use_per_tile_dma) {
            // P0 + P1: Per-tile DMA with CE write-back
            process_batch_tiled(batch, batch_count, cur_read);
            has_async_dma_ = true;
            latest_async_read_ = cur_read;
        } else {
            bool is_async = process_batch(batch, batch_count);
            if (is_async) {
                has_async_dma_ = true;
                latest_async_read_ = cur_read;

                // For global path with tile markers: use deferred event signaling
                if (has_tile_markers) {
                    // Count entries per tile in this batch for progress tracking
                    int tile_counts_local[MAX_TILES] = {};
                    for (int i = 0; i < batch_count; i++) {
                        uint32_t tile_id = (uint32_t)(batch[i].user_data >> 32);
                        if (tile_id < MAX_TILES) tile_counts_local[tile_id]++;
                    }

                    int slot = tile_event_head_;
                    PendingTileEvent& pe = pending_tile_events_[slot];
                    if (!pe.event) {
                        cuEventCreate(&pe.event, CU_EVENT_DISABLE_TIMING);
                    }
                    ce_manager_.record_event_on_last_stream(pe.event);
                    pe.num_tiles = 0;
                    pe.done_value = cur_read;
                    for (int i = 0; i < batch_count; i++) {
                        if ((batch[i].flags & FLAG_LAST_IN_TILE) && pe.num_tiles < MAX_TILES) {
                            uint32_t tile_id = (uint32_t)(batch[i].user_data >> 32);
                            if (tile_id < MAX_TILES) {
                                tile_progress_[tile_id] += tile_counts_local[tile_id];
                                tile_counts_local[tile_id] = 0;  // avoid double count
                                pe.tile_ids[pe.num_tiles++] = tile_id;
                            }
                        }
                    }
                    tile_event_head_ = (slot + 1) % MAX_PENDING_TILE_EVENTS;
                    tile_events_inflight_++;
                }
            } else {
                if (has_async_dma_) {
                    ce_manager_.wait_completion();
                    has_async_dma_ = false;
                }
                __atomic_store_n(&queue_->done_idx, cur_read, __ATOMIC_RELEASE);
                if (has_tile_markers) {
                    for (int i = 0; i < batch_count; i++) {
                        uint32_t tile_id = (uint32_t)(batch[i].user_data >> 32);
                        if (tile_id < MAX_TILES) {
                            tile_progress_[tile_id]++;
                            if (batch[i].flags & FLAG_LAST_IN_TILE) {
                                if (tiled_queue_->d_tile_chunk_done) {
                                    uint64_t val = tile_progress_[tile_id];
                                    cudaMemcpy(&tiled_queue_->d_tile_chunk_done[tile_id], &val,
                                               sizeof(uint64_t), cudaMemcpyHostToDevice);
                                } else {
                                    __atomic_store_n(&tiled_queue_->tile_chunk_done[tile_id],
                                                     tile_progress_[tile_id], __ATOMIC_RELEASE);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        process_batch_no_ce(batch, batch_count);
        __atomic_store_n(&queue_->done_idx, cur_read, __ATOMIC_RELEASE);
        if (has_tile_markers) {
            for (int i = 0; i < batch_count; i++) {
                uint32_t tile_id = (uint32_t)(batch[i].user_data >> 32);
                if (tile_id < MAX_TILES) {
                    tile_progress_[tile_id]++;
                    if (batch[i].flags & FLAG_LAST_IN_TILE) {
                        if (tiled_queue_->d_tile_chunk_done) {
                            uint64_t val = tile_progress_[tile_id];
                            cudaMemcpy(&tiled_queue_->d_tile_chunk_done[tile_id], &val,
                                       sizeof(uint64_t), cudaMemcpyHostToDevice);
                        } else {
                            __atomic_store_n(&tiled_queue_->tile_chunk_done[tile_id],
                                             tile_progress_[tile_id], __ATOMIC_RELEASE);
                        }
                    }
                }
            }
        }
    }

    batch_count = 0;
    or_flags = 0;
}

// Non-blocking poll of pending tile events (fallback for global-path deferred signaling).
void CpuPollingThread::poll_tile_events() {
    while (tile_events_inflight_ > 0) {
        int slot = tile_event_tail_;
        PendingTileEvent& pe = pending_tile_events_[slot];

        CUresult res = cuEventQuery(pe.event);
        if (res == CUDA_ERROR_NOT_READY) {
            break;
        }

        // Write cumulative progress count for each tile
        for (int i = 0; i < pe.num_tiles; i++) {
            uint32_t tid = pe.tile_ids[i];
            if (tiled_queue_->d_tile_chunk_done) {
                uint64_t val = tile_progress_[tid];
                cudaMemcpy(&tiled_queue_->d_tile_chunk_done[tid], &val,
                           sizeof(uint64_t), cudaMemcpyHostToDevice);
            } else {
                __atomic_store_n(&tiled_queue_->tile_chunk_done[tid],
                                 tile_progress_[tid], __ATOMIC_RELEASE);
            }
        }

        uint64_t cur_done = __atomic_load_n(&queue_->done_idx, __ATOMIC_RELAXED);
        if (pe.done_value > cur_done) {
            __atomic_store_n(&queue_->done_idx, pe.done_value, __ATOMIC_RELEASE);
        }

        tile_event_tail_ = (slot + 1) % MAX_PENDING_TILE_EVENTS;
        tile_events_inflight_--;
        has_async_dma_ = (tile_events_inflight_ > 0);
    }
}

bool CpuPollingThread::process_batch(Descriptor* batch, int count) {
    // Fast path: single entry
    if (count == 1) {
        SGEntry single;
        single.dst  = static_cast<CUdeviceptr>(batch[0].dst_addr);
        single.src  = reinterpret_cast<const void*>(batch[0].src_addr);
        single.size = batch[0].size;
        total_bytes_copied_.fetch_add(batch[0].size, std::memory_order_relaxed);

        auto ts0 = std::chrono::high_resolution_clock::now();
        ce_manager_.submit_scatter_gather(&single, 1);
        auto ts1 = std::chrono::high_resolution_clock::now();
        total_dma_submit_us_.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(ts1 - ts0).count(), std::memory_order_relaxed);

        descriptors_processed_.fetch_add(1, std::memory_order_relaxed);
        coalesced_entries_.fetch_add(1, std::memory_order_relaxed);
        batches_submitted_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Build SG entry list
    SGEntry sg_entries[MAX_BATCH_SIZE];
    for (int i = 0; i < count; i++) {
        sg_entries[i].dst  = static_cast<CUdeviceptr>(batch[i].dst_addr);
        sg_entries[i].src  = reinterpret_cast<const void*>(batch[i].src_addr);
        sg_entries[i].size = batch[i].size;
        total_bytes_copied_.fetch_add(batch[i].size, std::memory_order_relaxed);
    }

    // Check if dst addresses are contiguous
    bool dst_contiguous = true;
    if (count > 1) {
        uint32_t entry_size = sg_entries[0].size;
        CUdeviceptr expected_dst = sg_entries[0].dst + entry_size;
        for (int i = 1; i < count; i++) {
            if (sg_entries[i].dst != expected_dst || sg_entries[i].size != entry_size) {
                dst_contiguous = false;
                break;
            }
            expected_dst += entry_size;
        }
    }

    bool have_staging = (num_staging_bufs_ > 0);

    if (dst_contiguous && count > 1 && have_staging) {
        bool src_contiguous = true;
        for (int i = 1; i < count; i++) {
            if (sg_entries[i].src !=
                (const void*)((const char*)sg_entries[i-1].src + sg_entries[i-1].size)) {
                src_contiguous = false;
                break;
            }
        }

        if (src_contiguous) {
            SGEntry single;
            single.dst  = sg_entries[0].dst;
            single.src  = sg_entries[0].src;
            single.size = (size_t)count * sg_entries[0].size;

            CUresult res = ce_manager_.submit_scatter_gather(&single, 1);
            if (res != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(res, &err_str);
                GFD_LOG_ERROR("CE DMA failed: %s\n", err_str ? err_str : "unknown");
            }

            descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
            coalesced_entries_.fetch_add(1, std::memory_order_relaxed);
            batches_submitted_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        CUdeviceptr min_dst = sg_entries[0].dst;
        size_t entry_size = sg_entries[0].size;
        size_t range_size = (size_t)count * entry_size;

        if (range_size <= staging_buffer_size_ && num_staging_bufs_ > 0) {
            int cur = staging_cur_;
            auto tw0 = std::chrono::high_resolution_clock::now();

            if (staging_event_pending_[cur]) {
                cuEventSynchronize(staging_events_[cur]);
                staging_event_pending_[cur] = false;
                staging_inflight_count_--;
            }

            char* cur_buf = staging_bufs_[cur];

            auto tw1 = std::chrono::high_resolution_clock::now();
            total_dma_wait_us_.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(tw1 - tw0).count(), std::memory_order_relaxed);

            parallel_gather(sg_entries, count, cur_buf, min_dst);

            auto tg1 = std::chrono::high_resolution_clock::now();
            total_gather_us_.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(tg1 - tw1).count(), std::memory_order_relaxed);

            SGEntry run;
            run.dst  = min_dst;
            run.src  = cur_buf;
            run.size = range_size;

            auto ts0 = std::chrono::high_resolution_clock::now();
            ce_manager_.submit_scatter_gather(&run, 1);
            ce_manager_.record_event_on_last_stream(staging_events_[cur]);
            staging_event_pending_[cur] = true;
            auto ts1 = std::chrono::high_resolution_clock::now();
            total_dma_submit_us_.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(ts1 - ts0).count(), std::memory_order_relaxed);

            staging_inflight_count_++;
            staging_cur_ = (cur + 1) % num_staging_bufs_;

            staging_batches_.fetch_add(1, std::memory_order_relaxed);
            descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
            coalesced_entries_.fetch_add(count, std::memory_order_relaxed);
            batches_submitted_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }

    // Generic path: radix sort + coalesce + staging/CE
    SGEntry sort_tmp[MAX_BATCH_SIZE];
    radix_sort_sg(sg_entries, count, sort_tmp,
                  [](const SGEntry& e) -> uint64_t { return (uintptr_t)e.src; });

    SGEntry coalesced[MAX_BATCH_SIZE];
    int num_coalesced = 0;
    coalesced[0] = sg_entries[0];
    num_coalesced = 1;

    for (int i = 1; i < count; i++) {
        SGEntry& prev = coalesced[num_coalesced - 1];
        bool d_contig = (sg_entries[i].dst == prev.dst + prev.size);
        bool s_contig = (sg_entries[i].src ==
                         (const void*)((const char*)prev.src + prev.size));

        if (d_contig && s_contig) {
            prev.size += sg_entries[i].size;
        } else {
            coalesced[num_coalesced++] = sg_entries[i];
        }
    }

    constexpr int STAGING_THRESHOLD = 32;
    bool use_staging = (num_coalesced > STAGING_THRESHOLD && have_staging);

    if (use_staging) {
        CUdeviceptr min_dst = coalesced[0].dst;
        CUdeviceptr max_dst_end = coalesced[0].dst + coalesced[0].size;
        for (int i = 1; i < num_coalesced; i++) {
            if (coalesced[i].dst < min_dst) min_dst = coalesced[i].dst;
            CUdeviceptr end = coalesced[i].dst + coalesced[i].size;
            if (end > max_dst_end) max_dst_end = end;
        }
        size_t range_size = max_dst_end - min_dst;

        if (range_size > staging_buffer_size_ || num_staging_bufs_ == 0) {
            use_staging = false;
        } else {
            int cur = staging_cur_;
            if (staging_event_pending_[cur]) {
                cuEventSynchronize(staging_events_[cur]);
                staging_event_pending_[cur] = false;
                staging_inflight_count_--;
            }
            char* cur_buf = staging_bufs_[cur];

            parallel_gather(coalesced, num_coalesced, cur_buf, min_dst);

            radix_sort_sg(coalesced, num_coalesced, sort_tmp,
                          [](const SGEntry& e) -> uint64_t { return (uint64_t)e.dst; });

            SGEntry* runs = sort_tmp;
            int num_runs = 0;
            CUdeviceptr run_start = coalesced[0].dst;
            size_t run_size = coalesced[0].size;

            for (int i = 1; i < num_coalesced; i++) {
                if (coalesced[i].dst == run_start + run_size) {
                    run_size += coalesced[i].size;
                } else {
                    runs[num_runs].dst  = run_start;
                    runs[num_runs].src  = cur_buf + (run_start - min_dst);
                    runs[num_runs].size = run_size;
                    num_runs++;
                    run_start = coalesced[i].dst;
                    run_size  = coalesced[i].size;
                }
            }
            runs[num_runs].dst  = run_start;
            runs[num_runs].src  = cur_buf + (run_start - min_dst);
            runs[num_runs].size = run_size;
            num_runs++;

            CUresult res = ce_manager_.submit_scatter_gather(runs, num_runs);
            if (res != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(res, &err_str);
                GFD_LOG_ERROR("Staging DMA failed: %s\n", err_str ? err_str : "unknown");
            }
            ce_manager_.record_event_on_last_stream(staging_events_[cur]);
            staging_event_pending_[cur] = true;
            staging_inflight_count_++;
            staging_cur_ = (cur + 1) % num_staging_bufs_;

            staging_batches_.fetch_add(1, std::memory_order_relaxed);
            descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
            coalesced_entries_.fetch_add(num_coalesced, std::memory_order_relaxed);
            batches_submitted_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }

    if (!use_staging) {
        CUresult res = ce_manager_.submit_scatter_gather(coalesced, num_coalesced);
        if (res != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(res, &err_str);
            GFD_LOG_ERROR("CE scatter-gather failed: %s\n", err_str ? err_str : "unknown");
        }
    }

    descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
    coalesced_entries_.fetch_add(num_coalesced, std::memory_order_relaxed);
    batches_submitted_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void CpuPollingThread::process_batch_no_ce(Descriptor* batch, int count) {
    struct MemEntry {
        void*       dst;
        const void* src;
        size_t      size;
    };

    MemEntry entries[MAX_BATCH_SIZE];
    for (int i = 0; i < count; i++) {
        entries[i].dst  = reinterpret_cast<void*>(batch[i].dst_addr);
        entries[i].src  = reinterpret_cast<const void*>(batch[i].src_addr);
        entries[i].size = batch[i].size;
        total_bytes_copied_.fetch_add(batch[i].size, std::memory_order_relaxed);
    }

    std::sort(entries, entries + count,
              [](const MemEntry& a, const MemEntry& b) {
                  return (uintptr_t)a.src < (uintptr_t)b.src;
              });

    for (int i = 0; i < count; i++) {
        memcpy(entries[i].dst, entries[i].src, entries[i].size);
    }

    descriptors_processed_.fetch_add(count, std::memory_order_relaxed);
    coalesced_entries_.fetch_add(count, std::memory_order_relaxed);
    batches_submitted_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace gfd
