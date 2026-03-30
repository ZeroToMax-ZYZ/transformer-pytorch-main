from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        if not is_distributed_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

        return DistributedContext(
            is_distributed=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(
        is_distributed=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
    )


def cleanup_distributed() -> None:
    if is_distributed_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def wrap_ddp(model: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.is_distributed:
        return model

    if ctx.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
        )

    return DistributedDataParallel(model)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_distributed_initialized():
        return obj

    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]
