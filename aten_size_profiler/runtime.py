from __future__ import annotations

import atexit
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collector import AtenCollector
from .config import ProfilerConfig, load_config
from .distributed import RankInfo, get_rank_info, ranked_output_path
from .dispatch_mode import AtenSizeMode
from .exporter import export_collector
from .iteration import IterationController, IterationHooks


@dataclass
class RuntimeState:
    config: ProfilerConfig
    mode: AtenSizeMode
    active: bool = True
    exported: bool = False
    export_suppressed: bool = False
    export_count: int = 0
    output_path: Path | None = None
    iteration_controller: IterationController | None = None
    iteration_hooks: IterationHooks | None = None


_state: RuntimeState | None = None
_atexit_registered = False


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_shutdown_at_exit)
        _atexit_registered = True


def start_profiler(config: ProfilerConfig | None = None) -> AtenSizeMode:
    global _state
    if _state is not None and _state.active:
        return _state.mode

    profiler_config = config if config is not None else load_config()
    mode = AtenSizeMode(
        collector=AtenCollector(),
        enabled=profiler_config.enabled,
        record_stride=profiler_config.record_stride,
        record_stack=profiler_config.record_stack,
    )
    iteration_controller = None
    iteration_hooks = None
    if (
        profiler_config.enabled
        and profiler_config.target_iter is not None
        and profiler_config.iteration_mode
        in {"forward", "optimizer_step"}
    ):
        iteration_controller = IterationController(
            mode,
            profiler_config.target_iter,
            profiler_config.iteration_mode,
        )
        iteration_controller.initialize()
        iteration_hooks = IterationHooks(iteration_controller)
        iteration_hooks.install()
    mode.__enter__()
    _state = RuntimeState(
        config=profiler_config,
        mode=mode,
        iteration_controller=iteration_controller,
        iteration_hooks=iteration_hooks,
    )
    _register_atexit()
    return mode


def _metadata_for_state(
    state: RuntimeState,
    rank_info: RankInfo,
) -> dict[str, Any]:
    config = state.config
    return {
        "config_file": str(config.config_file or ""),
        "configured_enabled": config.enabled,
        "configured_output_dir": str(config.output_dir),
        "configured_output_filename": config.output_filename,
        "iteration_mode": config.iteration_mode,
        "target_iter": (
            config.target_iter if config.target_iter is not None else ""
        ),
        "record_stride": config.record_stride,
        "record_stack": config.record_stack,
        "global_rank": rank_info.rank,
        "world_size": rank_info.world_size,
        "local_rank": (
            rank_info.local_rank if rank_info.local_rank is not None else ""
        ),
        "distributed_initialized": rank_info.initialized,
    }


def stop_profiler(export: bool = True) -> Path | None:
    if _state is None:
        return None
    if not export:
        _state.export_suppressed = True
    if _state.active:
        if _state.iteration_hooks is not None:
            _state.iteration_hooks.uninstall()
        _state.mode.__exit__(None, None, None)
        _state.active = False

    if (
        export
        and _state.config.enabled
        and not _state.exported
        and not _state.export_suppressed
    ):
        rank_info = get_rank_info()
        _state.output_path = export_collector(
            _state.mode.collector,
            ranked_output_path(_state.config.output_path, rank_info),
            metadata=_metadata_for_state(_state, rank_info),
            iteration=_state.config.target_iter,
            rank=rank_info.rank,
        )
        _state.exported = True
        _state.export_count += 1
    return _state.output_path


def pause_profiler() -> None:
    if _state is not None and _state.active:
        if _state.iteration_controller is not None:
            _state.iteration_controller.pause()
        else:
            _state.mode.enabled = False


def resume_profiler() -> None:
    if (
        _state is not None
        and _state.active
        and _state.config.enabled
    ):
        if _state.iteration_controller is not None:
            _state.iteration_controller.resume()
        else:
            _state.mode.enabled = True


def is_installed() -> bool:
    return _state is not None and _state.active


def is_recording() -> bool:
    return is_installed() and _state.mode.enabled


def get_collector() -> AtenCollector | None:
    return _state.mode.collector if _state is not None else None


def get_status() -> dict[str, Any]:
    if _state is None:
        return {
            "installed": False,
            "recording": False,
            "exported": False,
            "export_count": 0,
            "output_path": None,
            "iteration": None,
        }
    return {
        "installed": _state.active,
        "recording": _state.active and _state.mode.enabled,
        "exported": _state.exported,
        "export_count": _state.export_count,
        "output_path": str(_state.output_path or ""),
        "iteration": (
            _state.iteration_controller.status()
            if _state.iteration_controller is not None
            else None
        ),
    }


def get_iteration_status() -> dict[str, Any] | None:
    if _state is None or _state.iteration_controller is None:
        return None
    return _state.iteration_controller.status()


def _shutdown_at_exit() -> None:
    try:
        stop_profiler(export=True)
    except Exception as error:
        print(
            f"aten_size_profiler: export at exit failed: {error}",
            file=sys.stderr,
        )
