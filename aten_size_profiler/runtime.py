from __future__ import annotations

import atexit
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch._C import _pop_torch_dispatch_stack, _push_on_torch_dispatch_stack
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .collector import AtenCollector
from .config import (
    VALID_ITERATION_MODES,
    ProfilerConfig,
    load_config,
    validate_output_filename,
)
from .distributed import (
    RankInfo,
    get_rank_info,
    install_rank_observer,
    ranked_output_path,
    reset_rank_info_cache,
    uninstall_rank_observer,
)
from .dispatch_mode import AtenSizeMode
from .exporter import export_collector
from .iteration import IterationController, IterationHooks


@dataclass
class RuntimeState:
    config: ProfilerConfig
    mode: AtenSizeMode
    mode_entered: bool = False
    active: bool = True
    exported: bool = False
    export_suppressed: bool = False
    export_count: int = 0
    output_path: Path | None = None
    iteration_controller: IterationController | None = None
    iteration_hooks: IterationHooks | None = None


_state: RuntimeState | None = None
_atexit_registered = False


def _remove_from_dispatch_stack(mode: AtenSizeMode) -> bool:
    if not any(
        current is mode for current in _get_current_dispatch_mode_stack()
    ):
        return False
    if _get_current_dispatch_mode() is mode:
        mode.__exit__(None, None, None)
        return True

    # Modes above the profiler are lifted off and pushed back without
    # calling their __exit__/__enter__, which may be non-reentrant or reset
    # their state. The dispatch flags those modes restore on exit are left
    # as saved, so is_in_torch_dispatch_mode() may stay True afterwards.
    modes_above: list[Any] = []
    try:
        while _get_current_dispatch_mode() is not mode:
            modes_above.append(_pop_torch_dispatch_stack(None))
        _pop_torch_dispatch_stack(None)
    finally:
        for other_mode in reversed(modes_above):
            _push_on_torch_dispatch_stack(other_mode)
    return True


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_shutdown_at_exit)
        _atexit_registered = True


def start_profiler(config: ProfilerConfig | None = None) -> AtenSizeMode:
    global _state
    if _state is not None and _state.active:
        return _state.mode

    reset_rank_info_cache()
    profiler_config = config if config is not None else load_config()
    if profiler_config.iteration_mode not in VALID_ITERATION_MODES:
        allowed = ", ".join(sorted(VALID_ITERATION_MODES))
        raise ValueError(
            f"iteration_mode must be one of: {allowed}"
        )
    validate_output_filename(profiler_config.output_filename)
    mode = AtenSizeMode(
        collector=AtenCollector(),
        enabled=profiler_config.enabled,
        record_stride=profiler_config.record_stride,
        record_stack=profiler_config.record_stack,
        record_scalar_args=profiler_config.record_scalar_args,
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
    mode_entered = profiler_config.enabled
    if mode_entered:
        install_rank_observer()
        mode.__enter__()
    _state = RuntimeState(
        config=profiler_config,
        mode=mode,
        mode_entered=mode_entered,
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
    hooks = state.iteration_hooks
    iteration_status = (
        state.iteration_controller.status()
        if state.iteration_controller is not None
        else None
    )
    return {
        "config_file": str(config.config_file or ""),
        "configured_enabled": config.enabled,
        "configured_output_dir": str(config.output_dir),
        "configured_output_filename": config.output_filename,
        "iteration_mode": config.iteration_mode,
        "iteration_phase": (
            iteration_status["phase"]
            if iteration_status is not None
            else "not configured"
        ),
        "iteration_complete": (
            iteration_status["phase"] == "FINISHED"
            if iteration_status is not None
            else ""
        ),
        "target_iter": (
            config.target_iter if config.target_iter is not None else ""
        ),
        "record_stride": config.record_stride,
        "record_stack": config.record_stack,
        "record_scalar_args": config.record_scalar_args,
        "global_rank": rank_info.rank,
        "world_size": rank_info.world_size,
        "local_rank": (
            rank_info.local_rank if rank_info.local_rank is not None else ""
        ),
        "distributed_initialized": rank_info.initialized,
        "forward_root_module": (
            hooks.root_module_name or "" if hooks is not None else ""
        ),
        "other_top_level_modules": (
            sorted(hooks.other_top_level_modules)
            if hooks is not None
            else []
        ),
        "parameterized_non_root_modules": (
            sorted(hooks.parameterized_non_root_modules)
            if hooks is not None
            else []
        ),
    }


def stop_profiler(export: bool = True) -> Path | None:
    if _state is None:
        return None
    if not export:
        _state.export_suppressed = True
    rank_info = get_rank_info()
    if _state.active:
        uninstall_rank_observer()
        if _state.iteration_hooks is not None:
            _state.iteration_hooks.uninstall()
        # Stop recording even if the mode cannot be popped here.
        _state.mode.enabled = False
        _state.active = False
        if _state.mode_entered:
            _state.mode_entered = False
            if not _remove_from_dispatch_stack(_state.mode):
                warnings.warn(
                    "aten_size_profiler: the profiler TorchDispatchMode is "
                    "not on this thread's mode stack, so it was only "
                    "disabled. stop_profiler() may have run in another "
                    "thread or while modes were temporarily removed, or a "
                    "TorchDispatchMode that the profiler was started inside "
                    "of exited first; that __exit__ pops the top mode, so "
                    "it removed the profiler and left itself on the stack.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if (
        export
        and _state.config.enabled
        and not _state.exported
        and not _state.export_suppressed
    ):
        if _state.iteration_controller is not None:
            iteration_status = _state.iteration_controller.status()
            if iteration_status["phase"] != "FINISHED":
                _state.mode.collector.clear()
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
