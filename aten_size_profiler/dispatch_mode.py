from __future__ import annotations

import traceback
from typing import Any

from torch.utils._python_dispatch import TorchDispatchMode

from .collector import (
    AtenCollector,
    CallRecord,
    OperatorAggregate,
    collect_scalar_args,
    collect_input_tensors,
)
from .distributed import observe_rank_info


def get_aten_op_name(func: Any) -> str | None:
    schema = getattr(func, "_schema", None)
    schema_name = getattr(schema, "name", "")
    if not schema_name.startswith("aten::"):
        return None

    operator_name = schema_name.removeprefix("aten::")
    overload_name = getattr(schema, "overload_name", "") or "default"
    return f"aten.{operator_name}.{overload_name}"


_EMPTY_INDICES: frozenset[int] = frozenset()
_EMPTY_NAMES: frozenset[str] = frozenset()


def _output_argument_exclusions(
    schema: Any,
) -> tuple[frozenset[int], frozenset[str]]:
    if schema is None:
        return _EMPTY_INDICES, _EMPTY_NAMES
    excluded_arg_indices: set[int] = set()
    excluded_kwargs: set[str] = set()
    for index, argument in enumerate(schema.arguments):
        is_out = getattr(argument, "is_out", None)
        if is_out is None:
            # Fallback for builds whose schema bindings lack ``is_out``.
            is_out = (
                argument.name.startswith("out")
                and argument.alias_info is not None
                and getattr(argument.alias_info, "is_write", False)
            )
        if not is_out:
            continue
        if argument.kwarg_only:
            excluded_kwargs.add(argument.name)
        else:
            excluded_arg_indices.add(index)
    return frozenset(excluded_arg_indices), frozenset(excluded_kwargs)


# Per-OpOverload results that only depend on the schema. OpOverload objects
# are long-lived singletons, so the cache is bounded by the number of
# distinct overloads a process dispatches.
_OpInfo = tuple[str, frozenset[int], frozenset[str]] | None
_op_info_cache: dict[Any, _OpInfo] = {}


def _get_op_info(func: Any) -> _OpInfo:
    try:
        return _op_info_cache[func]
    except KeyError:
        pass
    except TypeError:
        return _compute_op_info(func)
    info = _compute_op_info(func)
    _op_info_cache[func] = info
    return info


def _compute_op_info(func: Any) -> _OpInfo:
    op_name = get_aten_op_name(func)
    if op_name is None:
        return None
    excluded_arg_indices, excluded_kwargs = _output_argument_exclusions(
        getattr(func, "_schema", None)
    )
    return op_name, excluded_arg_indices, excluded_kwargs


class AtenSizeMode(TorchDispatchMode):
    supports_higher_order_operators = True

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True

    def __init__(
        self,
        collector: AtenCollector | None = None,
        *,
        enabled: bool = True,
        record_stride: bool = True,
        record_stack: bool = False,
        record_scalar_args: bool = True,
        record_memory_format: bool = False,
    ) -> None:
        super().__init__()
        self.collector = collector if collector is not None else AtenCollector()
        self.enabled = enabled
        self.record_stride = record_stride
        self.record_stack = record_stack
        self.record_scalar_args = record_scalar_args
        self.record_memory_format = record_memory_format
        self.current_iteration: int | None = None

    @property
    def calls(self) -> list[CallRecord]:
        return self.collector.calls

    @property
    def aggregates(self) -> dict[str, OperatorAggregate]:
        return self.collector.aggregates

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        actual_kwargs = {} if kwargs is None else kwargs
        if not self.enabled:
            return func(*args, **actual_kwargs)

        observe_rank_info()
        op_info = _get_op_info(func)
        if op_info is None:
            return func(*args, **actual_kwargs)
        op_name, excluded_arg_indices, excluded_kwargs = op_info

        inputs = collect_input_tensors(
            args,
            actual_kwargs,
            record_stride=self.record_stride,
            excluded_arg_indices=excluded_arg_indices,
            excluded_kwargs=excluded_kwargs,
            record_memory_format=self.record_memory_format,
        )
        scalar_args = (
            collect_scalar_args(
                args,
                actual_kwargs,
                excluded_arg_indices=excluded_arg_indices,
                excluded_kwargs=excluded_kwargs,
            )
            if self.record_scalar_args
            else ()
        )
        stack = tuple(traceback.format_stack(limit=32)) if self.record_stack else ()
        result = func(*args, **actual_kwargs)
        self.collector.record(
            op_name,
            inputs,
            stack=stack,
            iteration=self.current_iteration,
            scalar_args=scalar_args,
        )
        return result
