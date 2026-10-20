from __future__ import annotations

import gc
import threading
import weakref
from enum import Enum
from typing import Any

from torch import nn
from torch.optim import Optimizer

from .dispatch_mode import AtenSizeMode


class IterationPhase(str, Enum):
    WAITING = "WAITING"
    RECORDING = "RECORDING"
    FINISHED = "FINISHED"


class IterationController:
    def __init__(
        self,
        mode: AtenSizeMode,
        target_iter: int,
        iteration_mode: str,
    ) -> None:
        if target_iter < 0:
            raise ValueError("target_iter must be non-negative")
        if iteration_mode not in {"forward", "optimizer_step"}:
            raise ValueError(
                "automatic iteration mode must be 'forward' or "
                "'optimizer_step'"
            )
        self.mode = mode
        self.target_iter = target_iter
        self.iteration_mode = iteration_mode
        self.current_iter = 0
        self.phase = IterationPhase.WAITING
        self.forward_completed = False
        self.paused = False

    def initialize(self) -> None:
        self.mode.enabled = False
        self.mode.current_iteration = None

    def _set_recording(self, recording: bool) -> None:
        should_record = recording and not self.paused
        self.mode.enabled = should_record
        self.mode.current_iteration = (
            self.current_iter if should_record else None
        )

    def on_forward_start(self) -> None:
        if self.phase is IterationPhase.FINISHED:
            return
        if (
            self.iteration_mode == "optimizer_step"
            and self.forward_completed
        ):
            return
        if self.current_iter == self.target_iter:
            self.phase = IterationPhase.RECORDING
            self._set_recording(True)
        else:
            self.phase = IterationPhase.WAITING
            self._set_recording(False)

    def on_forward_end(self) -> None:
        self.forward_completed = True
        if self.iteration_mode == "forward":
            if self.phase is IterationPhase.RECORDING:
                self.phase = IterationPhase.FINISHED
                self._set_recording(False)
            self.current_iter += 1
            if self.current_iter > self.target_iter:
                self.phase = IterationPhase.FINISHED

    def on_forward_error(self) -> None:
        if self.phase is IterationPhase.FINISHED:
            return
        if self.phase is IterationPhase.RECORDING:
            self.mode.collector.clear()
        self.forward_completed = False
        self.phase = IterationPhase.WAITING
        self._set_recording(False)

    def on_optimizer_step(self) -> None:
        if self.iteration_mode != "optimizer_step":
            return
        if not self.forward_completed:
            return
        if self.phase is IterationPhase.RECORDING:
            self.phase = IterationPhase.FINISHED
            self._set_recording(False)
        self.current_iter += 1
        self.forward_completed = False
        if self.current_iter > self.target_iter:
            self.phase = IterationPhase.FINISHED

    def pause(self) -> None:
        self.paused = True
        self._set_recording(False)

    def resume(self) -> None:
        self.paused = False
        if self.phase is IterationPhase.RECORDING:
            self._set_recording(True)

    def status(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "current_iter": self.current_iter,
            "target_iter": self.target_iter,
            "iteration_mode": self.iteration_mode,
        }


class IterationHooks:
    def __init__(self, controller: IterationController) -> None:
        self.controller = controller
        self._module_call: Any = None
        self._optimizer_init: Any = None
        self._wrapped_optimizers: dict[
            int, tuple[weakref.ReferenceType[Optimizer], Any, Any]
        ] = {}
        self._thread_state = threading.local()

    def install(self) -> None:
        self._module_call = nn.Module.__call__
        controller = self.controller
        thread_state = self._thread_state
        original_module_call = self._module_call

        def wrapped_module_call(
            module: nn.Module,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            depth = getattr(thread_state, "module_depth", 0)
            is_top_level = depth == 0
            if is_top_level:
                controller.on_forward_start()
            thread_state.module_depth = depth + 1
            forward_succeeded = False
            try:
                result = original_module_call(module, *args, **kwargs)
                forward_succeeded = True
                return result
            finally:
                thread_state.module_depth = depth
                if is_top_level:
                    if forward_succeeded:
                        controller.on_forward_end()
                    else:
                        controller.on_forward_error()

        nn.Module.__call__ = wrapped_module_call

        self._optimizer_init = Optimizer.__init__
        original_optimizer_init = self._optimizer_init
        hooks = self

        def wrapped_optimizer_init(
            optimizer: Optimizer,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            original_optimizer_init(optimizer, *args, **kwargs)
            hooks._wrap_optimizer(optimizer)

        Optimizer.__init__ = wrapped_optimizer_init
        for optimizer in gc.get_objects():
            optimizer_type = type(optimizer)
            if (
                optimizer_type is Optimizer
                or Optimizer in optimizer_type.__mro__
            ):
                self._wrap_optimizer(optimizer)

    def _wrap_optimizer(self, optimizer: Optimizer) -> None:
        optimizer_id = id(optimizer)
        if optimizer_id in self._wrapped_optimizers:
            return
        original_step = optimizer.step
        controller = self.controller

        def wrapped_step(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            controller.on_optimizer_step()
            return result

        optimizer.step = wrapped_step
        self._wrapped_optimizers[optimizer_id] = (
            weakref.ref(optimizer),
            original_step,
            wrapped_step,
        )

    def uninstall(self) -> None:
        if self._module_call is not None:
            nn.Module.__call__ = self._module_call
            self._module_call = None
        if self._optimizer_init is not None:
            Optimizer.__init__ = self._optimizer_init
            self._optimizer_init = None
        for optimizer_ref, original_step, wrapped_step in (
            self._wrapped_optimizers.values()
        ):
            optimizer = optimizer_ref()
            if optimizer is not None and optimizer.step is wrapped_step:
                optimizer.step = original_step
        self._wrapped_optimizers.clear()
