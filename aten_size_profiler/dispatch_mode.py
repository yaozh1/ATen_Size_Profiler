from __future__ import annotations

import traceback
from typing import Any

from torch.utils._python_dispatch import TorchDispatchMode

from .collector import (
    AtenCollector,
    CallRecord,
    OperatorAggregate,
    collect_input_tensors,
)


def get_aten_op_name(func: Any) -> str | None:
    schema = getattr(func, "_schema", None)
    schema_name = getattr(schema, "name", "")
    if not schema_name.startswith("aten::"):
        return None

    operator_name = schema_name.removeprefix("aten::")
    overload_name = getattr(schema, "overload_name", "") or "default"
    return f"aten.{operator_name}.{overload_name}"


class AtenSizeMode(TorchDispatchMode):
    def __init__(
        self,
        collector: AtenCollector | None = None,
        *,
        enabled: bool = True,
        record_stride: bool = True,
        record_stack: bool = False,
    ) -> None:
        super().__init__()
        self.collector = collector if collector is not None else AtenCollector()
        self.enabled = enabled
        self.record_stride = record_stride
        self.record_stack = record_stack
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

        op_name = get_aten_op_name(func)
        if op_name is None:
            return func(*args, **actual_kwargs)

        inputs = collect_input_tensors(
            args,
            actual_kwargs,
            record_stride=self.record_stride,
        )
        stack = tuple(traceback.format_stack(limit=32)) if self.record_stack else ()
        result = func(*args, **actual_kwargs)
        self.collector.record(
            op_name,
            inputs,
            stack=stack,
            iteration=self.current_iteration,
        )
        return result
