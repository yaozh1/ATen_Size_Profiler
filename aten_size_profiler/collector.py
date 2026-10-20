from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TensorInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    stride: tuple[int, ...] | None
    numel: int
    element_size: int
    logical_bytes: int


@dataclass(frozen=True)
class CallRecord:
    call_id: int
    op_name: str
    inputs: tuple[TensorInfo, ...]
    stack: tuple[str, ...] = ()
    iteration: int | None = None


@dataclass
class OperatorAggregate:
    op_name: str
    call_count: int = 0
    input_logical_bytes_total: int = 0
    input_signature_counts: Counter[tuple[Any, ...]] = field(
        default_factory=Counter
    )

    @property
    def unique_input_count(self) -> int:
        return len(self.input_signature_counts)


def _format_mapping_path(path: str, key: Any) -> str:
    return f"{path}[{key!r}]"


def _iter_tensor_values(
    value: Any,
    path: str,
    active_containers: set[int],
) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
        return

    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            return
        active_containers.add(container_id)
        try:
            for index, child in enumerate(value):
                yield from _iter_tensor_values(
                    child,
                    f"{path}[{index}]",
                    active_containers,
                )
        finally:
            active_containers.remove(container_id)
        return

    if isinstance(value, dict):
        container_id = id(value)
        if container_id in active_containers:
            return
        active_containers.add(container_id)
        try:
            for key, child in value.items():
                yield from _iter_tensor_values(
                    child,
                    _format_mapping_path(path, key),
                    active_containers,
                )
        finally:
            active_containers.remove(container_id)


def _describe_tensor(
    path: str,
    tensor: torch.Tensor,
    record_stride: bool = True,
) -> TensorInfo:
    shape = tuple(int(size) for size in tensor.shape)
    if not record_stride:
        stride = None
    else:
        try:
            stride = tuple(int(value) for value in tensor.stride())
        except RuntimeError:
            stride = None

    numel = int(tensor.numel())
    element_size = int(tensor.element_size())
    return TensorInfo(
        path=path,
        shape=shape,
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        stride=stride,
        numel=numel,
        element_size=element_size,
        logical_bytes=numel * element_size,
    )


def collect_input_tensors(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    record_stride: bool = True,
) -> tuple[TensorInfo, ...]:
    tensor_infos: list[TensorInfo] = []
    active_containers: set[int] = set()

    for index, value in enumerate(args):
        for path, tensor in _iter_tensor_values(
            value,
            f"args[{index}]",
            active_containers,
        ):
            tensor_infos.append(
                _describe_tensor(path, tensor, record_stride=record_stride)
            )

    for key, value in kwargs.items():
        for path, tensor in _iter_tensor_values(
            value,
            _format_mapping_path("kwargs", key),
            active_containers,
        ):
            tensor_infos.append(
                _describe_tensor(path, tensor, record_stride=record_stride)
            )

    return tuple(tensor_infos)


def _input_signature(inputs: tuple[TensorInfo, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            tensor.path,
            tensor.shape,
            tensor.dtype,
            tensor.device,
        )
        for tensor in inputs
    )


class AtenCollector:
    def __init__(self) -> None:
        self.calls: list[CallRecord] = []
        self.aggregates: dict[str, OperatorAggregate] = {}

    def record(
        self,
        op_name: str,
        inputs: tuple[TensorInfo, ...],
        stack: tuple[str, ...] = (),
        iteration: int | None = None,
    ) -> CallRecord:
        call = CallRecord(
            call_id=len(self.calls),
            op_name=op_name,
            inputs=inputs,
            stack=stack,
            iteration=iteration,
        )
        self.calls.append(call)

        aggregate = self.aggregates.get(op_name)
        if aggregate is None:
            aggregate = OperatorAggregate(op_name=op_name)
            self.aggregates[op_name] = aggregate
        aggregate.call_count += 1
        aggregate.input_logical_bytes_total += sum(
            tensor.logical_bytes for tensor in inputs
        )
        aggregate.input_signature_counts[_input_signature(inputs)] += 1
        return call

    def clear(self) -> None:
        self.calls.clear()
        self.aggregates.clear()
