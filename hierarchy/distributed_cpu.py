"""Gloo collectives staged through CPU; model computation may remain on CUDA.

This module deliberately creates no NCCL process group. A differentiable feature
gather retains the exact SUM-backward semantics of torch.distributed.nn's
all_gather. Explicit parameter-gradient averaging therefore has the same global
objective and world-size scaling as the preceding trainer implementation.
"""
from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def setup_distributed(device):
    """Return (rank, world_size, device), using Gloo for CPU communication only."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    dev = torch.device(device)
    if world > 1 and dev.type == "cuda":
        dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(dev)
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("gloo", timeout=timedelta(seconds=1800))
        _distributed()
        if dist.get_rank() != rank or dist.get_world_size() != world:
            raise RuntimeError("Existing Gloo group disagrees with rank environment")
    return rank, world, dev


def _distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    if dist.get_world_size() == 1:
        return False
    if str(dist.get_backend()).lower() != "gloo":
        raise RuntimeError("CPU-staged collectives require a Gloo process group")
    return True


class _GatherCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        if value.ndim < 1:
            raise ValueError("Feature gather requires a batch dimension")
        ctx.rank = dist.get_rank()
        ctx.world = dist.get_world_size()
        ctx.device = value.device
        ctx.dtype = value.dtype
        ctx.rows = value.shape[0]
        staged = value.detach().to(device="cpu").contiguous()
        pieces = [torch.empty_like(staged) for _ in range(ctx.world)]
        dist.all_gather(pieces, staged)
        return torch.cat(pieces, dim=0).to(device=ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        # SUM contributions from every loss replica before selecting our input
        # slice. Reducing in FP32 prevents additional FP16 transport overflow.
        wire_dtype = torch.float64 if grad_output.dtype == torch.float64 else torch.float32
        staged = grad_output.detach().to(device="cpu", dtype=wire_dtype).contiguous()
        dist.all_reduce(staged, op=dist.ReduceOp.SUM)
        local = staged.narrow(0, ctx.rank * ctx.rows, ctx.rows).contiguous()
        return local.to(device=ctx.device, dtype=ctx.dtype)


def gather_tensor(value):
    """Gather equal-sized rank batches with differentiable SUM backward."""
    if not _distributed():
        return value
    return _GatherCPU.apply(value)


def gather_constant(value):
    """Gather labels/teacher features without recording an autograd edge."""
    if not _distributed():
        return value
    staged = value.detach().to(device="cpu").contiguous()
    if staged.ndim < 1:
        raise ValueError("Constant gather requires a batch dimension")
    pieces = [torch.empty_like(staged) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, staged)
    return torch.cat(pieces, dim=0).to(device=value.device)


def all_reduce_flag(value, op=dist.ReduceOp.MIN):
    """Reduce a scalar flag on CPU and return a Python bool or int."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("A distributed flag must contain one element")
        as_bool = value.dtype == torch.bool
        scalar = value.detach().to(device="cpu").item()
    else:
        as_bool = isinstance(value, bool)
        scalar = value
    staged = torch.tensor(int(scalar), dtype=torch.int64, device="cpu")
    if _distributed():
        dist.all_reduce(staged, op=op)
    result = int(staged.item())
    return bool(result) if as_bool else result


def _trainable(core):
    return [p for p in core.parameters() if p.requires_grad]


def _flat_cpu(values):
    # A single bulk copy in each direction; all model tensors share one device.
    if len({v.device for v in values}) != 1:
        raise RuntimeError("Trainable tensors must reside on one device")
    return torch.cat([v.detach().reshape(-1).float() for v in values]).cpu()


def _copy_flat(staged, destinations):
    flat = staged.to(device=destinations[0].device)
    offset = 0
    for value in destinations:
        value.copy_(flat[offset:offset + value.numel()].view_as(value))
        offset += value.numel()


@torch.no_grad()
def synchronize_trainable_parameters(core):
    """Broadcast trainable FP32 parameters; frozen backbone tensors stay intact."""
    if not _distributed():
        return
    parameters = _trainable(core)
    if not parameters:
        return
    staged = _flat_cpu(parameters)
    dist.broadcast(staged, src=0)
    _copy_flat(staged, parameters)


@torch.no_grad()
def average_gradients(core):
    """Average scaled gradients before GradScaler unscale, using CPU/Gloo."""
    if not _distributed():
        return
    parameters = _trainable(core)
    if not all_reduce_flag(all(p.grad is not None for p in parameters)):
        raise RuntimeError("Missing trainable gradient on at least one rank before distributed average")
    if not parameters:
        return
    gradients = [p.grad for p in parameters]
    staged = _flat_cpu(gradients)
    dist.all_reduce(staged, op=dist.ReduceOp.SUM)
    staged.div_(dist.get_world_size())
    _copy_flat(staged, gradients)
