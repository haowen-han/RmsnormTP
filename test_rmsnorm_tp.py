"""
Test script for FusedTPRMSNorm.

Usage (single node, 2 GPUs):
    torchrun --nproc_per_node=2 test_rmsnorm_tp.py

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 test_rmsnorm_tp.py
"""
import os
import torch
import torch.distributed as dist
import time

@torch.compile
def reference_tp_rmsnorm(x, weight, eps, tp_group):
    """Reference implementation: compute + all-reduce + normalize."""
    tp_size = dist.get_world_size(tp_group)
    # Compute local variance
    variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
    if tp_size > 1:
        dist.all_reduce(variance, op=dist.ReduceOp.SUM, group=tp_group)
        variance = variance / tp_size
    # Normalize
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def test_correctness(fused_norm, tp_group, device, rank, tp_size):
    """Test that fused kernel matches reference implementation."""
    print(f"[Rank {rank}] Testing correctness...")

    local_hidden = 4096 // tp_size
    total_hidden = 4096

    torch.manual_seed(42 + rank)
    x = torch.randn(128, local_hidden, device=device, dtype=torch.float32)
    weight = torch.randn(local_hidden, device=device, dtype=torch.float32)

    # Reference
    x_ref = x.clone()
    weight_ref = weight.clone()
    out_ref = reference_tp_rmsnorm(x_ref, weight_ref, fused_norm.eps, tp_group)

    # Fused
    x_fused = x.clone()
    weight_fused = weight.clone()
    out_fused = fused_norm.forward(x_fused, weight_fused)

    # Wait for all ranks
    dist.barrier(tp_group)

    # Compare
    max_diff = (out_ref - out_fused).abs().max().item()
    mean_diff = (out_ref - out_fused).abs().mean().item()
    print(f"[Rank {rank}] Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

    # Allow some numerical tolerance (FP32, partial sums from different order)
    if max_diff > 1e-3:
        print(f"[Rank {rank}] FAIL: max diff too large!")
        # Print sample values for debugging
        print(f"  ref[0,:5]:   {out_ref[0, :5]}")
        print(f"  fused[0,:5]: {out_fused[0, :5]}")
    else:
        print(f"[Rank {rank}] PASS")

    return max_diff < 1e-3


def test_benchmark(fused_norm, tp_group, device, rank, tp_size, num_iters=100):
    """Benchmark fused vs reference."""
    local_hidden = 4096 // tp_size
    num_rows = 1024

    torch.manual_seed(42 + rank)
    x = torch.randn(num_rows, local_hidden, device=device, dtype=torch.float32)
    weight = torch.randn(local_hidden, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = reference_tp_rmsnorm(x, weight, fused_norm.eps, tp_group)
    dist.barrier(tp_group)

    for _ in range(10):
        _ = fused_norm.forward(x, weight)
    dist.barrier(tp_group)

    # Benchmark reference
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        _ = reference_tp_rmsnorm(x, weight, fused_norm.eps, tp_group)
    dist.barrier(tp_group)
    torch.cuda.synchronize()
    t_ref = (time.time() - t0) / num_iters * 1000

    # Benchmark fused
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        _ = fused_norm.forward(x, weight)
    dist.barrier(tp_group)
    torch.cuda.synchronize()
    t_fused = (time.time() - t0) / num_iters * 1000

    if rank == 0:
        print(f"\n[Benchmark] rows={num_rows}, local_hidden={local_hidden}, tp_size={tp_size}")
        print(f"  Reference (compute + all-reduce + normalize): {t_ref:.3f} ms")
        print(f"  Fused (single kernel, NVLink direct):         {t_fused:.3f} ms")
        print(f"  Speedup: {t_ref/t_fused:.2f}x")


def test_various_shapes(fused_norm, tp_group, device, rank, tp_size):
    """Test with various hidden dimensions and row counts."""
    print(f"[Rank {rank}] Testing various shapes...")

    all_pass = True
    for total_hidden in [1024, 2048, 4096, 8192]:
        if total_hidden % tp_size != 0:
            continue
        local_hidden = total_hidden // tp_size
        for num_rows in [1, 16, 128, 512]:
            torch.manual_seed(42 + rank)
            x = torch.randn(num_rows, local_hidden, device=device, dtype=torch.float32)
            weight = torch.randn(local_hidden, device=device, dtype=torch.float32)

            out_ref = reference_tp_rmsnorm(x.clone(), weight.clone(), fused_norm.eps, tp_group)
            out_fused = fused_norm.forward(x.clone(), weight.clone())
            dist.barrier(tp_group)

            max_diff = (out_ref - out_fused).abs().max().item()
            passed = max_diff < 1e-3
            all_pass = all_pass and passed
            if not passed:
                print(f"  FAIL: hidden={total_hidden}, rows={num_rows}, diff={max_diff:.2e}")

    if all_pass:
        print(f"[Rank {rank}] All shape tests PASS")
    else:
        print(f"[Rank {rank}] Some shape tests FAILED")

    return all_pass


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create TP group (use world as TP group for simplicity)
    tp_group = dist.group.WORLD
    tp_size = world_size

    if rank == 0:
        print(f"Testing with tp_size={tp_size}")

    from rmsnorm_tp import FusedTPRMSNorm

    fused_norm = FusedTPRMSNorm(hidden_dim=4096, eps=1e-6, tp_group=tp_group)
    dist.barrier(tp_group)

    if rank == 0:
        print("CommBuffer initialized successfully\n")

    # Test correctness
    correct = test_correctness(fused_norm, tp_group, device, rank, tp_size)
    dist.barrier(tp_group)

    # Test various shapes
    test_various_shapes(fused_norm, tp_group, device, rank, tp_size)
    dist.barrier(tp_group)

    # Benchmark
    test_benchmark(fused_norm, tp_group, device, rank, tp_size)

    # Cleanup
    fused_norm.destroy()
    dist.barrier(tp_group)

    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
