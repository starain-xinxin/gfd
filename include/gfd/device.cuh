#pragma once

#include "gfd/descriptor_queue.h"

// ============================================================
// GFD Device-Side API: __device__ building blocks for fused kernels
//
// These inline device functions are designed to be composed inside
// user-defined __global__ kernels, enabling communication/computation
// fusion. Instead of launching separate transfer and compute kernels,
// users embed GFD descriptor operations directly into their kernels.
//
// Typical fused kernel pattern:
//
//   __global__ void my_fused_kernel(...) {
//       // Phase 1: Request data prefetch
//       gfd::device::write_descriptor(...);
//       gfd::device::fence_and_commit(...);
//
//       // Phase 2: Overlap compute while CPU+CE transfer data
//       compute_something();
//
//       // Phase 3: Wait for transfer completion
//       gfd::device::wait_for_completion(...);
//       __syncthreads();
//
//       // Phase 4: Use the transferred data
//       use_prefetched_data();
//   }
//
// ============================================================

namespace gfd {
namespace device {

// ---- Per-thread descriptor write ----
//
// Each thread writes one descriptor for its own token.
// token_idx:     this thread's token index (typically blockIdx.x * blockDim.x + threadIdx.x)
// base_slot:     pre-assigned starting slot in the ring buffer (host sets this before launch)
// src_addr:      CPU memory source address for this token
// gpu_dst:       GPU destination buffer base pointer
// token_size:    bytes per token
// num_tokens:    total number of tokens (for FLAG_LAST_IN_BATCH on the last entry)
// user_data:     optional user-defined payload (e.g., packed expert_id << 32 | token_id)
__device__ __forceinline__
void write_descriptor(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    uint64_t src_addr,
    void* gpu_dst,
    uint32_t token_size,
    int num_tokens,
    uint64_t user_data = 0)
{
    uint64_t slot = base_slot + token_idx;
    Descriptor* desc = &queue->entries[slot % QUEUE_SIZE];

    desc->src_addr = src_addr;
    desc->dst_addr = (uint64_t)gpu_dst + (uint64_t)token_idx * token_size;
    desc->size = token_size;
    desc->flags = (token_idx == num_tokens - 1) ? FLAG_LAST_IN_BATCH : FLAG_NONE;
    desc->user_data = user_data;
}

// ---- Warp-optimized system fence + sequence commit ----
//
// Two-phase commit with warp-level fence optimization:
//   1. Warp-leader __threadfence_system() — makes descriptor data visible to CPU
//   2. Each active thread writes its sequence number (commit marker)
//   3. Warp-leader __threadfence_system() — makes sequence visible to CPU
//
// active:     true if this thread has a valid descriptor (token_idx < num_tokens)
// base_slot:  same base_slot used in write_descriptor()
// token_idx:  same token_idx used in write_descriptor()
__device__ __forceinline__
void fence_and_commit(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    bool active)
{
    // Pre-commit fence: ensure descriptor fields are visible to CPU
    if ((threadIdx.x & 31) == 0) {
        __threadfence_system();
    }
    __syncwarp();

    // Commit: write sequence number makes this entry visible to CPU poller
    if (active) {
        uint64_t slot = base_slot + token_idx;
        queue->entries[slot % QUEUE_SIZE].sequence = slot + 1;
    }

    // Post-commit fence: ensure sequence is visible to CPU
    if ((threadIdx.x & 31) == 0) {
        __threadfence_system();
    }
}

// ---- Single-thread completion wait ----
//
// Polls done_idx until the CPU poller has completed all transfers
// up to expected_done. Only one thread should call this (typically tid == 0),
// then __syncthreads() to broadcast completion to all threads.
//
// expected_done: base_slot + num_tokens (the read_idx value after all entries are consumed)
__device__ __forceinline__
void wait_for_completion(
    DescriptorQueue* queue,
    uint64_t expected_done)
{
    while (true) {
        uint64_t done = *((volatile uint64_t*)&queue->done_idx);
        if (done >= expected_done) break;
    }
}

// ---- Safe descriptor write with backpressure ----
//
// Checks if the target slot has been consumed (sequence == 0) before writing.
// Returns false if the ring buffer is full at this slot (GPU producing faster
// than CPU consuming). Caller should retry or yield.
//
// Use this in latency-tolerant paths where correctness is more important than
// throughput. For maximum throughput with guaranteed capacity, pre-allocate
// enough queue slots (QUEUE_SIZE > max_inflight).
__device__ __forceinline__
bool write_descriptor_safe(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    uint64_t src_addr,
    void* gpu_dst,
    uint32_t token_size,
    int num_tokens,
    uint64_t user_data = 0)
{
    uint64_t slot = base_slot + token_idx;
    Descriptor* desc = &queue->entries[slot % QUEUE_SIZE];

    // Check if slot has been consumed by CPU (sequence reset to 0)
    if (desc->sequence != 0) return false;

    desc->src_addr = src_addr;
    desc->dst_addr = (uint64_t)gpu_dst + (uint64_t)token_idx * token_size;
    desc->size = token_size;
    desc->flags = (token_idx == num_tokens - 1) ? FLAG_LAST_IN_BATCH : FLAG_NONE;
    desc->user_data = user_data;
    return true;
}

// ---- Convenience: write + fence + commit in one call ----
//
// Combines write_descriptor() + fence_and_commit() for simple cases.
// For maximum overlap, prefer calling them separately with compute in between.
__device__ __forceinline__
void write_and_commit(
    DescriptorQueue* queue,
    uint64_t base_slot,
    int token_idx,
    bool active,
    uint64_t src_addr,
    void* gpu_dst,
    uint32_t token_size,
    int num_tokens,
    uint64_t user_data = 0)
{
    if (active) {
        write_descriptor(queue, base_slot, token_idx,
                         src_addr, gpu_dst, token_size, num_tokens, user_data);
    }
    fence_and_commit(queue, base_slot, token_idx, active);
}

}  // namespace device
}  // namespace gfd
