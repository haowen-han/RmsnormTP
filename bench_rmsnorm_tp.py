"""
Benchmark FusedTPRMSNorm vs AllReduce across shapes, then plot.

Usage:
    torchrun --nproc_per_node=2 bench_rmsnorm_tp.py
    torchrun --nproc_per_node=4 bench_rmsnorm_tp.py
"""

import json
import os
import time
import torch
import torch.distributed as dist


def reference_tp_rmsnorm(x, weight, eps, tp_group):
    tp_size = dist.get_world_size(tp_group)
    variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
    if tp_size > 1:
        dist.all_reduce(variance, op=dist.ReduceOp.SUM, group=tp_group)
        variance = variance / tp_size
    return x * torch.rsqrt(variance + eps) * weight


def bench(fused_norm, tp_group, device, rank, total_hidden, num_rows, eps):
    local_hidden = total_hidden // fused_norm.tp_size
    x = torch.randn(num_rows, local_hidden, device=device, dtype=torch.float32)
    w = torch.randn(local_hidden, device=device, dtype=torch.float32)

    # warmup
    for _ in range(10):
        reference_tp_rmsnorm(x, w, eps, tp_group)
    dist.barrier(tp_group)
    for _ in range(10):
        fused_norm.forward(x, w)
    dist.barrier(tp_group)

    # bench reference
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        reference_tp_rmsnorm(x, w, eps, tp_group)
    dist.barrier(tp_group)
    torch.cuda.synchronize()
    t_ref = (time.time() - t0) / 50 * 1000

    # bench fused
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        fused_norm.forward(x, w)
    dist.barrier(tp_group)
    torch.cuda.synchronize()
    t_fused = (time.time() - t0) / 50 * 1000

    return t_ref, t_fused


def plot(all_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hiddens = sorted(set(e["total_hidden"] for entries in all_data.values() for e in entries))
    fig, axes = plt.subplots(1, len(hiddens), figsize=(5 * len(hiddens), 5), sharey=True)
    if len(hiddens) == 1:
        axes = [axes]

    for ax, h in zip(axes, hiddens):
        for tp, entries in sorted(all_data.items()):
            matched = [(e["num_rows"], e["speedup"]) for e in entries if e["total_hidden"] == h]
            matched.sort()
            if not matched:
                continue
            rows, speedup = zip(*matched)
            ax.plot(rows, speedup, "o-", label=f"TP={tp}", markersize=4)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Num Rows")
        ax.set_ylabel("Speedup (x)")
        ax.set_title(f"Hidden={h}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("FusedTPRMSNorm Speedup over AllReduce", fontsize=13)
    fig.tight_layout()
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, "bench_result.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = dist.group.WORLD

    from rmsnorm_tp import FusedTPRMSNorm

    total_hiddens = [1024, 2048, 4096, 8192]
    num_rows_list = [1, 16, 64, 128, 512, 1024, 2048]

    results = []

    for total_hidden in total_hiddens:
        if total_hidden % tp_size != 0:
            continue
        fused_norm = FusedTPRMSNorm(hidden_dim=total_hidden, eps=1e-6, tp_group=tp_group)
        dist.barrier(tp_group)

        for num_rows in num_rows_list:
            if num_rows > 65536:
                continue
            if rank == 0:
                print(f"tp={tp_size} hidden={total_hidden} rows={num_rows} ...", end=" ", flush=True)

            t_ref, t_fused = bench(fused_norm, tp_group, device, rank, total_hidden, num_rows, 1e-6)

            if rank == 0:
                speedup = t_ref / t_fused if t_fused > 0 else 0
                print(f"ref={t_ref:.3f}ms fused={t_fused:.3f}ms speedup={speedup:.2f}x")
                results.append({
                    "total_hidden": total_hidden, "num_rows": num_rows,
                    "t_ref": round(t_ref, 3), "t_fused": round(t_fused, 3),
                    "speedup": round(speedup, 2),
                })

        fused_norm.destroy()
        dist.barrier(tp_group)

    # Save JSON + plot (rank 0 only)
    if rank == 0:
        base = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base, f"bench_tp{tp_size}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")

        # Load all existing result files and plot together
        all_data = {tp_size: results}
        for tp in [2, 4, 8]:
            if tp == tp_size:
                continue
            path = os.path.join(base, f"bench_tp{tp}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_data[tp] = json.load(f)

        plot(all_data)


if __name__ == "__main__":
    main()
