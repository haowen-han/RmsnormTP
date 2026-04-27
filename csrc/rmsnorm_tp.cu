#include "common.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace rmsnorm_tp {

// ============================================================
// PTX memory ordering primitives (system-scope for NVLink)
// ============================================================

__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void st_release_sys_global(int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

// ============================================================
// Warp-level reduction helpers
// ============================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================
// Fused TP RMSNorm kernel
//
// Push model: each CTA writes partial sum to all remote GPUs,
// then spins on local memory for data from other GPUs.
// ============================================================

template <int kBlockThreads>
__global__ void fused_rmsnorm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float** sums_ptrs_gpu,
    int** gen_ptrs_gpu,
    int rank,
    int tp_size,
    int local_hidden_dim,
    int total_hidden_dim,
    int num_rows,
    float eps,
    int generation)
{
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // ================================================================
    // Phase 1: Compute local partial sum of squares (vectorized float4)
    // ================================================================
    const float4* input4 = reinterpret_cast<const float4*>(input + row * local_hidden_dim);
    int vec_len = local_hidden_dim / 4;  // number of float4 elements

    float local_sum = 0.0f;
    for (int i = tid; i < vec_len; i += kBlockThreads) {
        float4 v = input4[i];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Handle remaining elements (if local_hidden_dim not divisible by 4)
    int remainder_start = vec_len * 4;
    for (int i = tid + remainder_start; i < local_hidden_dim; i += kBlockThreads) {
        float v = input[row * local_hidden_dim + i];
        local_sum += v * v;
    }

    // Warp-level reduce
    local_sum = warp_reduce_sum(local_sum);

    // Block-level reduce via shared memory
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    if (lane_id == 0) atomicAdd(&block_sum, local_sum);
    __syncthreads();

    // ================================================================
    // Phase 2: Push partial sum to ALL GPUs (including self for simplicity)
    // Each thread pushes to one target rank
    // ================================================================
    if (tid < tp_size) {
        int target_rank = tid;
        // My slot in target_rank's buffer: [source_rank=rank][row]
        int slot_idx = rank * num_rows + row;
        float* target_sum_ptr = &sums_ptrs_gpu[target_rank][slot_idx];
        int* target_gen_ptr = &gen_ptrs_gpu[target_rank][slot_idx];

        // Write partial sum (regular store, ordering ensured by release below)
        *target_sum_ptr = block_sum;

        // Write generation counter with release semantics
        // This ensures the partial sum is visible before the counter update
        st_release_sys_global(target_gen_ptr, generation);
    }

    // ================================================================
    // Phase 3: Spin-wait on LOCAL memory for all ranks' data
    // Each thread waits for one source rank
    // ================================================================
    if (tid < tp_size) {
        int source_rank = tid;
        int slot_idx = source_rank * num_rows + row;
        // This pointer is in OUR local memory (push model)
        int* local_gen_ptr = &gen_ptrs_gpu[rank][slot_idx];

        // Fast spin with volatile reads (local L2 cache, cheap)
        while (ld_volatile_global(local_gen_ptr) < generation) {
            // Optional: yield to reduce power consumption
        }
        // Final acquire to establish happens-before
        ld_acquire_sys_global(local_gen_ptr);
    }

    // Ensure all threads see all remote data
    __syncthreads();
    fence_acq_rel_sys();
    __syncthreads();

    // ================================================================
    // Phase 4: Read all partial sums from LOCAL memory and compute global variance
    // ================================================================
    __shared__ float global_sum_sq;
    if (tid == 0) {
        float total = 0.0f;
        for (int r = 0; r < tp_size; r++) {
            total += sums_ptrs_gpu[rank][r * num_rows + row];
        }
        global_sum_sq = total;
    }
    __syncthreads();

    // ================================================================
    // Phase 5: Normalize with vectorized float4 access
    // ================================================================
    float variance = global_sum_sq / static_cast<float>(total_hidden_dim);
    float inv_rms = rsqrtf(variance + eps);

    float4* output4 = reinterpret_cast<float4*>(output + row * local_hidden_dim);
    const float4* weight4 = reinterpret_cast<const float4*>(weight);

    for (int i = tid; i < vec_len; i += kBlockThreads) {
        float4 v = input4[i];
        float4 w = weight4[i];
        float4 out;
        out.x = v.x * inv_rms * w.x;
        out.y = v.y * inv_rms * w.y;
        out.z = v.z * inv_rms * w.z;
        out.w = v.w * inv_rms * w.w;
        output4[i] = out;
    }

    // Handle remainder
    for (int i = tid + remainder_start; i < local_hidden_dim; i += kBlockThreads) {
        float v = input[row * local_hidden_dim + i];
        output[row * local_hidden_dim + i] = v * inv_rms * weight[i];
    }
}

// ============================================================
// Kernel launcher
// ============================================================

void launch_fused_rmsnorm(
    float* output,
    const float* input,
    const float* weight,
    float** sums_ptrs_gpu,
    int** gen_ptrs_gpu,
    int rank,
    int tp_size,
    int local_hidden_dim,
    int total_hidden_dim,
    int num_rows,
    float eps,
    int generation,
    cudaStream_t stream)
{
    constexpr int kBlockThreads = 256;
    dim3 grid(num_rows);
    dim3 block(kBlockThreads);

    fused_rmsnorm_kernel<kBlockThreads><<<grid, block, 0, stream>>>(
        output, input, weight, sums_ptrs_gpu, gen_ptrs_gpu,
        rank, tp_size, local_hidden_dim, total_hidden_dim,
        num_rows, eps, generation);
}

}  // namespace rmsnorm_tp
