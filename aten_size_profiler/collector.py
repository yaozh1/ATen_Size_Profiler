from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Set as AbstractSet
from dataclasses import dataclass, field
import json
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
    # Only set when record_memory_format is enabled; see classify_memory_layout.
    memory_layout: str | None = None


@dataclass(frozen=True)
class CallRecord:
    call_id: int
    op_name: str
    inputs: tuple[TensorInfo, ...]
    stack: tuple[str, ...] = ()
    iteration: int | None = None
    scalar_args: tuple[tuple[str, str], ...] = ()


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


_TORCH_ENUM_TYPES = (
    torch.dtype,
    torch.device,
    torch.memory_format,
    torch.layout,
)


_SCALAR_TYPES = (bool, int, float, complex, str) + _TORCH_ENUM_TYPES


def _is_scalar_value(value: Any) -> bool:
    return value is None or isinstance(value, _SCALAR_TYPES)


def _format_scalar_value(value: Any) -> str:
    if isinstance(value, _TORCH_ENUM_TYPES):
        return json.dumps(str(value), ensure_ascii=False)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=repr,
        )
    except (TypeError, ValueError):
        return repr(value)


def _iter_scalar_values(
    value: Any,
    path: str,
    active_containers: set[int],
) -> Iterator[tuple[str, str]]:
    if isinstance(value, torch.Tensor):
        return

    if _is_scalar_value(value):
        yield path, _format_scalar_value(value)
        return

    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            return
        active_containers.add(container_id)
        try:
            for index, child in enumerate(value):
                yield from _iter_scalar_values(
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
                yield from _iter_scalar_values(
                    child,
                    _format_mapping_path(path, key),
                    active_containers,
                )
        finally:
            active_containers.remove(container_id)


MEMORY_LAYOUT_MIXED = "mixed"

# dim -> (channels-last memory format, contiguous name, channels-last name).
# The names describe the memory order only, not what the dimensions mean.
_CHANNELS_LAST_LAYOUTS = {
    4: (torch.channels_last, "NCHW", "NHWC"),
    5: (torch.channels_last_3d, "NCDHW", "NDHWC"),
}


def classify_memory_layout(tensor: torch.Tensor) -> str | None:
    """Classify ``tensor`` with ``is_contiguous()``; metadata only, no sync.

    4D/5D tensors report the contiguous or channels-last name, ``ambiguous``
    when both checks hold (for example size-1 C or spatial dimensions) and
    ``non_contiguous`` when neither does. Other ranks report ``contiguous``
    or ``non_contiguous``. Returns None when the tensor does not support
    ``is_contiguous()`` (for example sparse CSR).
    """
    try:
        contiguous = tensor.is_contiguous()
        layouts = _CHANNELS_LAST_LAYOUTS.get(tensor.dim())
        if layouts is None:
            return "contiguous" if contiguous else "non_contiguous"
        memory_format, contiguous_name, channels_last_name = layouts
        channels_last = tensor.is_contiguous(memory_format=memory_format)
    except RuntimeError:
        return None
    if contiguous and channels_last:
        return "ambiguous"
    if contiguous:
        return contiguous_name
    if channels_last:
        return channels_last_name
    return "non_contiguous"


def _describe_tensor(
    path: str,
    tensor: torch.Tensor,
    record_stride: bool = True,
    record_memory_format: bool = False,
) -> TensorInfo | None:
    try:
        shape = tuple(int(size) for size in tensor.shape)
        if not record_stride:
            stride = None
        else:
            stride = tuple(int(value) for value in tensor.stride())
        numel = int(tensor.numel())
        element_size = int(tensor.element_size())
    except RuntimeError:
        return None
    # Classified separately so an unsupported is_contiguous() never drops a
    # tensor that v0.2.1 would have recorded.
    memory_layout = (
        classify_memory_layout(tensor) if record_memory_format else None
    )
    return TensorInfo(
        path=path,
        shape=shape,
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        stride=stride,
        numel=numel,
        element_size=element_size,
        logical_bytes=numel * element_size,
        memory_layout=memory_layout,
    )


def collect_input_tensors(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    record_stride: bool = True,
    excluded_arg_indices: AbstractSet[int] | None = None,
    excluded_kwargs: AbstractSet[str] | None = None,
    record_memory_format: bool = False,
) -> tuple[TensorInfo, ...]:
    tensor_infos: list[TensorInfo] = []
    active_containers: set[int] = set()
    excluded_arg_indices = excluded_arg_indices or set()
    excluded_kwargs = excluded_kwargs or set()

    for index, value in enumerate(args):
        if index in excluded_arg_indices:
            continue
        for path, tensor in _iter_tensor_values(
            value,
            f"args[{index}]",
            active_containers,
        ):
            tensor_info = _describe_tensor(
                path,
                tensor,
                record_stride=record_stride,
                record_memory_format=record_memory_format,
            )
            if tensor_info is not None:
                tensor_infos.append(tensor_info)

    for key, value in kwargs.items():
        if key in excluded_kwargs:
            continue
        for path, tensor in _iter_tensor_values(
            value,
            _format_mapping_path("kwargs", key),
            active_containers,
        ):
            tensor_info = _describe_tensor(
                path,
                tensor,
                record_stride=record_stride,
                record_memory_format=record_memory_format,
            )
            if tensor_info is not None:
                tensor_infos.append(tensor_info)

    return tuple(tensor_infos)


def collect_scalar_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    excluded_arg_indices: AbstractSet[int] | None = None,
    excluded_kwargs: AbstractSet[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    scalar_args: list[tuple[str, str]] = []
    active_containers: set[int] = set()
    excluded_arg_indices = excluded_arg_indices or set()
    excluded_kwargs = excluded_kwargs or set()

    for index, value in enumerate(args):
        if index in excluded_arg_indices:
            continue
        scalar_args.extend(
            _iter_scalar_values(
                value,
                f"args[{index}]",
                active_containers,
            )
        )

    for key, value in kwargs.items():
        if key in excluded_kwargs:
            continue
        scalar_args.extend(
            _iter_scalar_values(
                value,
                _format_mapping_path("kwargs", key),
                active_containers,
            )
        )

    return tuple(scalar_args)


def _input_signature(
    inputs: tuple[TensorInfo, ...],
    scalar_args: tuple[tuple[str, str], ...] = (),
) -> tuple[Any, ...]:
    return tuple(
        (
            tensor.path,
            tensor.shape,
            tensor.dtype,
            tensor.device,
        )
        for tensor in inputs
    ) + (scalar_args,)


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
        scalar_args: tuple[tuple[str, str], ...] = (),
    ) -> CallRecord:
        call = CallRecord(
            call_id=len(self.calls),
            op_name=op_name,
            inputs=inputs,
            stack=stack,
            iteration=iteration,
            scalar_args=scalar_args,
        )
        self.calls.append(call)
        self._aggregate(call)
        return call

    def _aggregate(self, call: CallRecord) -> None:
        aggregate = self.aggregates.get(call.op_name)
        if aggregate is None:
            aggregate = OperatorAggregate(op_name=call.op_name)
            self.aggregates[call.op_name] = aggregate
        aggregate.call_count += 1
        aggregate.input_logical_bytes_total += sum(
            tensor.logical_bytes for tensor in call.inputs
        )
        aggregate.input_signature_counts[
            _input_signature(call.inputs, call.scalar_args)
        ] += 1

    def truncate(self, call_count: int) -> None:
        """Drop calls recorded after the first ``call_count`` calls."""
        if call_count >= len(self.calls):
            return
        del self.calls[call_count:]
        self.aggregates.clear()
        for call in self.calls:
            self._aggregate(call)

    def clear(self) -> None:
        self.calls.clear()
        self.aggregates.clear()
