from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist


@dataclass(frozen=True)
class RankInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int | None = None
    initialized: bool = False


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_rank_info() -> RankInfo:
    if dist.is_available() and dist.is_initialized():
        return RankInfo(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=_env_int("LOCAL_RANK", 0),
            initialized=True,
        )

    rank = _env_int("RANK", 0)
    world_size = max(_env_int("WORLD_SIZE", 1), 1)
    local_rank = (
        _env_int("LOCAL_RANK", 0)
        if "LOCAL_RANK" in os.environ
        else None
    )
    return RankInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        initialized=False,
    )


def ranked_output_path(
    output_path: str | Path,
    rank_info: RankInfo | None = None,
) -> Path:
    path = Path(output_path)
    info = rank_info if rank_info is not None else get_rank_info()
    if info.world_size <= 1 and info.rank == 0:
        return path
    return path.with_name(
        f"{path.stem}_rank{info.rank}{path.suffix}"
    )
