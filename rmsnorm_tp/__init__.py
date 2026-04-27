import torch
import torch.distributed as dist
import rmsnorm_tp_cpp as _C


class TPCommBuffer:
    """Manages IPC shared memory for intra-node TP communication."""

    def __init__(self, rank: int, tp_size: int, num_max_rows: int):
        self.rank = rank
        self.tp_size = tp_size
        self.num_max_rows = num_max_rows
        self._buf = _C.CommBuffer(rank, tp_size, num_max_rows)
        self._generation = 0

    def allocate(self):
        self._buf.allocate()

    def get_ipc_handle(self):
        return self._buf.get_ipc_handle()

    def open_remote_handle(self, remote_rank, handle):
        self._buf.open_remote_handle(remote_rank, handle)

    def finalize_setup(self):
        self._buf.finalize_setup()

    def destroy(self):
        self._buf.destroy()

    def next_generation(self):
        self._generation += 1
        return self._generation

    @property
    def generation(self):
        return self._generation


class FusedTPRMSNorm:
    """Fused Tensor-Parallel RMSNorm using NVLink direct communication."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6, tp_group=None):
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.tp_group = tp_group

        self.tp_size = dist.get_world_size(tp_group) if tp_group else 1
        self.rank = dist.get_rank(tp_group) if tp_group else 0

        if self.tp_size == 1:
            self.comm_buf = None
            return

        # Determine local device
        local_device = torch.cuda.current_device()

        # Exchange device IDs via all_gather_object (works with NCCL backend)
        device_id_list = [None] * self.tp_size
        dist.all_gather_object(device_id_list, local_device, group=tp_group)
        self.device_ids = device_id_list

        # Estimate max rows
        # Typical: batch_size * seq_len, use a generous upper bound
        self.num_max_rows = 65536

        self.comm_buf = TPCommBuffer(self.rank, self.tp_size, self.num_max_rows)
        self.comm_buf.allocate()

        # Exchange IPC handles
        local_handle = self.comm_buf.get_ipc_handle()
        handle_bytes = local_handle.to_bytes()

        # All-gather handle bytes
        handle_list = [None] * self.tp_size
        dist.all_gather_object(handle_list, handle_bytes, group=tp_group)

        # Reconstruct handles and open remote
        for i in range(self.tp_size):
            if i != self.rank:
                remote_handle = _C.IpcHandle()
                remote_handle.from_bytes(handle_list[i])
                self.comm_buf.open_remote_handle(i, remote_handle)

        self.comm_buf.finalize_setup()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape [num_rows, local_hidden_dim], float32
            weight: weight tensor of shape [local_hidden_dim], float32
        Returns:
            output tensor of same shape as x
        """
        assert x.is_contiguous() and x.dtype == torch.float32
        assert weight.is_contiguous() and weight.dtype == torch.float32

        num_rows = x.shape[0]
        local_hidden_dim = x.shape[1]
        total_hidden_dim = local_hidden_dim * self.tp_size

        if self.tp_size == 1:
            # Simple single-GPU RMSNorm
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + self.eps)
            return x_norm * weight

        assert num_rows <= self.comm_buf.num_max_rows

        output = torch.empty_like(x)

        generation = self.comm_buf.next_generation()

        _C.launch_fused_rmsnorm(
            output.data_ptr(),
            x.data_ptr(),
            weight.data_ptr(),
            self.comm_buf._buf.sums_ptrs_gpu(),
            self.comm_buf._buf.gen_ptrs_gpu(),
            self.rank,
            self.tp_size,
            local_hidden_dim,
            total_hidden_dim,
            num_rows,
            self.eps,
            generation,
            torch.cuda.current_stream().cuda_stream,
        )

        return output

    def destroy(self):
        if self.comm_buf is not None:
            self.comm_buf.destroy()
