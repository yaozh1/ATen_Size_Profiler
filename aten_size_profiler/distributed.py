from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist


@dataclass(frozen=True)
class RankInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int | None = None
    initialized: bool = False


_last_initialized_rank_info: RankInfo | None = None
_DIST_AVAILABLE = dist.is_available()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_rank_info() -> RankInfo:
    global _last_initialized_rank_info
    if _DIST_AVAILABLE and dist.is_initialized():
        info = RankInfo(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=_env_int("LOCAL_RANK", 0),
            initialized=True,
        )
        _last_initialized_rank_info = info
        return info

    if _last_initialized_rank_info is not None:
        return _last_initialized_rank_info

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


def observe_rank_info() -> None:
    """Cache rank info while the process group is alive.

    Called on the dispatch hot path so a report exported after
    ``destroy_process_group()`` still carries the real rank. Once a rank has
    been cached this is a single global lookup.
    """
    if _last_initialized_rank_info is None and _DIST_AVAILABLE:
        if dist.is_initialized():
            get_rank_info()


def reset_rank_info_cache() -> None:
    global _last_initialized_rank_info
    _last_initialized_rank_info = None


_process_group_patch: tuple[tuple[Any, str, Any, Any], ...] | None = None


def install_rank_observer() -> None:
    """Cache rank info around process group init/destroy.

    ``observe_rank_info()`` only runs when an operator is dispatched. Without
    any operator between init and destroy (e.g. ``mp.spawn`` workers, where
    ``RANK`` is not set), the export would otherwise see rank 0 in every
    process and all of them would write the same file.
    """
    global _process_group_patch
    if _process_group_patch is not None or not _DIST_AVAILABLE:
        return
    original_init = dist.init_process_group
    original_destroy = dist.destroy_process_group

    @functools.wraps(original_init)
    def init_process_group(*args: Any, **kwargs: Any) -> Any:
        result = original_init(*args, **kwargs)
        observe_rank_info()
        return result

    @functools.wraps(original_destroy)
    def destroy_process_group(*args: Any, **kwargs: Any) -> Any:
        observe_rank_info()
        return original_destroy(*args, **kwargs)

    # Patch both the public alias and its defining module so either call
    # path is observed.
    modules = [dist]
    c10d = getattr(dist, "distributed_c10d", None)
    if c10d is not None and c10d is not dist:
        modules.append(c10d)
    patched = []
    for module in modules:
        for name, original, wrapper in (
            ("init_process_group", original_init, init_process_group),
            ("destroy_process_group", original_destroy, destroy_process_group),
        ):
            if getattr(module, name, None) is original:
                setattr(module, name, wrapper)
                patched.append((module, name, original, wrapper))
    _process_group_patch = tuple(patched)


def uninstall_rank_observer() -> None:
    global _process_group_patch
    if _process_group_patch is None:
        return
    for module, name, original, wrapper in _process_group_patch:
        if getattr(module, name, None) is wrapper:
            setattr(module, name, original)
    _process_group_patch = None


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
