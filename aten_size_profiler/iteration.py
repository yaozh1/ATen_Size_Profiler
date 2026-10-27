from __future__ import annotations

import functools
import gc
import threading
import warnings
import weakref
from enum import Enum
from typing import Any

import torch
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
        self._scaler_step_active = False
        self._scaler_step_counted = False
        self._step_depth = 0
        self._step_start_calls = 0
        self._failed_step_pending = False

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
        if self._failed_step_pending:
            # A new forward after a failed optimizer step restarts the
            # iteration; keeping the earlier forward would double count it.
            self.on_iteration_error()
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
        if self.phase is IterationPhase.FINISHED:
            return
        self.forward_completed = True
        if self.iteration_mode == "forward":
            if self.phase is IterationPhase.RECORDING:
                self.phase = IterationPhase.FINISHED
                self._set_recording(False)
            self.current_iter += 1
            if self.current_iter > self.target_iter:
                self.phase = IterationPhase.FINISHED

    def on_forward_error(self) -> None:
        self.on_iteration_error()

    def on_iteration_error(self) -> None:
        if self.phase is IterationPhase.FINISHED:
            return
        if self.phase is IterationPhase.RECORDING:
            self.mode.collector.clear()
        self.forward_completed = False
        self._failed_step_pending = False
        self._step_start_calls = 0
        self.phase = IterationPhase.WAITING
        self._set_recording(False)

    def begin_optimizer_step(self) -> None:
        if self._step_depth == 0:
            self._step_start_calls = len(self.mode.collector.calls)
        self._step_depth += 1

    def end_optimizer_step(self, succeeded: bool) -> None:
        self._step_depth -= 1
        if succeeded or self._step_depth:
            return
        if self.phase is IterationPhase.FINISHED:
            return
        # Drop only what the failed step recorded. A retry of step() alone
        # keeps the forward/backward data; a new forward resets the iteration.
        if self.phase is IterationPhase.RECORDING:
            self.mode.collector.truncate(self._step_start_calls)
        self._failed_step_pending = True

    def begin_scaler_step(self) -> None:
        self._scaler_step_active = True
        self._scaler_step_counted = False

    def end_scaler_step(self, succeeded: bool) -> None:
        if not succeeded:
            self.on_iteration_error()
        elif not self._scaler_step_counted:
            self.on_optimizer_step()
        self._scaler_step_active = False
        self._scaler_step_counted = False

    def on_optimizer_step(self) -> None:
        if self.iteration_mode != "optimizer_step":
            return
        if self.phase is IterationPhase.FINISHED:
            return
        if self._scaler_step_active:
            self._scaler_step_counted = True
        self._failed_step_pending = False
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
        self._tensor_backward: Any = None
        self._scaler_step_patch: tuple[Any, Any] | None = None
        self._wrapped_optimizers: dict[
            int, tuple[weakref.ReferenceType[Optimizer], Any]
        ] = {}
        self._wrapped_step_types: dict[type, tuple[Any, Any]] = {}
        self._wrapped_instance_steps: list[
            tuple[weakref.ReferenceType[Optimizer], weakref.ReferenceType[Any]]
        ] = []
        self._step_guards_active = False
        self._thread_state = threading.local()
        self._forward_module_ref: weakref.ReferenceType[nn.Module] | None = None
        self.root_module_name: str | None = None
        self.other_top_level_modules: set[str] = set()
        self.parameterized_non_root_modules: set[str] = set()

    def _is_iteration_module(self, module: nn.Module) -> bool:
        root_module = (
            self._forward_module_ref()
            if self._forward_module_ref is not None
            else None
        )
        if root_module is None:
            self._forward_module_ref = weakref.ref(module)
            self.root_module_name = type(module).__qualname__
            return True
        if root_module is module:
            return True
        self._observe_non_root_module(module)
        return False

    def _observe_non_root_module(self, module: nn.Module) -> None:
        # Standalone loss modules are expected here and carry no parameters.
        # A parameterized top-level module usually means the forward root was
        # picked wrongly (e.g. a preprocessing module called before the model)
        # or there are several top-level models; both give misleading reports.
        module_name = type(module).__qualname__
        if module_name in self.other_top_level_modules:
            return
        self.other_top_level_modules.add(module_name)
        if next(module.parameters(), None) is None:
            return
        self.parameterized_non_root_modules.add(module_name)
        warnings.warn(
            "aten_size_profiler: top-level module "
            f"{module_name!r} has parameters but is not the forward root "
            f"{self.root_module_name!r}; iterations are counted only on "
            "the first top-level module called, so the report may not "
            "cover the intended model.",
            RuntimeWarning,
            stacklevel=4,
        )

    def install(self) -> None:
        self._step_guards_active = True
        self._module_call = nn.Module.__call__
        controller = self.controller
        thread_state = self._thread_state
        original_module_call = self._module_call
        hooks = self

        def wrapped_module_call(
            module: nn.Module,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            depth = getattr(thread_state, "module_depth", 0)
            is_iteration_module = (
                depth == 0 and hooks._is_iteration_module(module)
            )
            if is_iteration_module:
                controller.on_forward_start()
            thread_state.module_depth = depth + 1
            forward_succeeded = False
            try:
                result = original_module_call(module, *args, **kwargs)
                forward_succeeded = True
                return result
            finally:
                thread_state.module_depth = depth
                if is_iteration_module:
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
        if controller.iteration_mode == "optimizer_step":
            self._patch_backward()
            self._patch_grad_scaler()

    def _patch_backward(self) -> None:
        original_backward = torch.Tensor.backward
        controller = self.controller

        def wrapped_backward(
            tensor: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            try:
                return original_backward(tensor, *args, **kwargs)
            except BaseException:
                controller.on_iteration_error()
                raise

        torch.Tensor.backward = wrapped_backward
        self._tensor_backward = original_backward

    def _patch_grad_scaler(self) -> None:
        grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
        if grad_scaler is None:
            return
        original_step = grad_scaler.step
        controller = self.controller

        def wrapped_scaler_step(
            scaler: Any,
            optimizer: Optimizer,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            controller.begin_scaler_step()
            try:
                result = original_step(
                    scaler,
                    optimizer,
                    *args,
                    **kwargs,
                )
            except BaseException:
                controller.end_scaler_step(succeeded=False)
                raise
            controller.end_scaler_step(succeeded=True)
            return result

        grad_scaler.step = wrapped_scaler_step
        self._scaler_step_patch = (grad_scaler, original_step)

    def _wrap_optimizer(self, optimizer: Optimizer) -> None:
        optimizer_id = id(optimizer)
        wrapped = self._wrapped_optimizers.get(optimizer_id)
        # A freed optimizer's id can be reused by a new one, so a matching id
        # alone does not mean this optimizer is wrapped; the stale entry is
        # replaced below.
        if wrapped is not None and wrapped[0]() is optimizer:
            return
        controller = self.controller

        def step_post_hook(
            optimizer: Optimizer,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            controller.on_optimizer_step()

        handle = optimizer.register_step_post_hook(step_post_hook)
        self._wrapped_optimizers[optimizer_id] = (
            weakref.ref(optimizer),
            handle,
        )
        self._wrap_step_errors(optimizer)

    def _wrap_step_errors(self, optimizer: Optimizer) -> None:
        # Step post hooks only run on success, so a step that raises is
        # invisible to them; without this a retry would append to the data
        # recorded by the failed attempt.
        if self.controller.iteration_mode != "optimizer_step":
            return
        self._wrap_class_step(type(optimizer))
        # An LR scheduler created before install replaces ``step`` on the
        # instance with a function bound to the class step of that time, so
        # its calls never reach the class wrapper.
        instance_step = optimizer.__dict__.get("step")
        if callable(instance_step):
            guarded_step = self._guard_step(instance_step)
            optimizer.step = guarded_step
            # Weak references only: the instance step may be a bound method
            # of the optimizer and must not keep it alive.
            self._wrapped_instance_steps.append(
                (weakref.ref(optimizer), weakref.ref(guarded_step))
            )

    def _wrap_class_step(self, optimizer_type: type) -> None:
        owner = next(
            (cls for cls in optimizer_type.__mro__ if "step" in cls.__dict__),
            None,
        )
        if owner is None or owner in self._wrapped_step_types:
            return
        original_step = owner.__dict__["step"]
        if not callable(original_step):
            return
        wrapped_step = self._guard_step(original_step)
        owner.step = wrapped_step
        self._wrapped_step_types[owner] = (original_step, wrapped_step)

    def _guard_step(self, step: Any) -> Any:
        # A scheduler keeps the step it wrapped, so guards can outlive
        # uninstall; they hold the hooks weakly and pass through once
        # uninstalled.
        hooks_ref = weakref.ref(self)

        @functools.wraps(step)
        def guarded_step(*args: Any, **kwargs: Any) -> Any:
            hooks = hooks_ref()
            if hooks is None or not hooks._step_guards_active:
                return step(*args, **kwargs)
            controller = hooks.controller
            controller.begin_optimizer_step()
            succeeded = False
            try:
                result = step(*args, **kwargs)
                succeeded = True
                return result
            finally:
                controller.end_optimizer_step(succeeded)

        return guarded_step

    def uninstall(self) -> None:
        self._step_guards_active = False
        if self._module_call is not None:
            nn.Module.__call__ = self._module_call
            self._module_call = None
        if self._optimizer_init is not None:
            Optimizer.__init__ = self._optimizer_init
            self._optimizer_init = None
        if self._tensor_backward is not None:
            torch.Tensor.backward = self._tensor_backward
            self._tensor_backward = None
        if self._scaler_step_patch is not None:
            grad_scaler, original_step = self._scaler_step_patch
            grad_scaler.step = original_step
            self._scaler_step_patch = None
        for optimizer_ref, handle in (
            self._wrapped_optimizers.values()
        ):
            handle.remove()
        self._wrapped_optimizers.clear()
        for owner, (original_step, wrapped_step) in (
            self._wrapped_step_types.items()
        ):
            if owner.__dict__.get("step") is wrapped_step:
                owner.step = original_step
        self._wrapped_step_types.clear()
        for optimizer_ref, guarded_step_ref in self._wrapped_instance_steps:
            optimizer = optimizer_ref()
            guarded_step = guarded_step_ref()
            if (
                optimizer is not None
                and guarded_step is not None
                and optimizer.__dict__.get("step") is guarded_step
            ):
                optimizer.step = guarded_step.__wrapped__
        self._wrapped_instance_steps.clear()
