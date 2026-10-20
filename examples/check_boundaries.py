from __future__ import annotations

import torch

from aten_size_profiler import AtenSizeMode


def run() -> None:
    with AtenSizeMode(record_stride=True) as mode:
        zero_dim = torch.tensor(2.0)
        empty = torch.empty(0, 3)
        non_contiguous = torch.arange(12).reshape(3, 4).t()
        mutable = torch.ones(4, 3)
        torch.add(non_contiguous, zero_dim)
        torch.add(empty, 1)
        mutable.add_(1)
        torch.scalar_tensor(3.0)

    assert any(
        tensor.shape == () and tensor.stride == ()
        for call in mode.calls
        for tensor in call.inputs
    )
    assert any(
        tensor.shape == (0, 3)
        for call in mode.calls
        for tensor in call.inputs
    )
    assert any(
        tensor.shape == (4, 3) and tensor.stride == (1, 4)
        for call in mode.calls
        for tensor in call.inputs
    )
    assert any(
        call.op_name == "aten.add_.Tensor"
        for call in mode.calls
    )
    assert any(not call.inputs for call in mode.calls)

    with AtenSizeMode(record_stride=False) as no_stride_mode:
        torch.mm(torch.ones(2, 2), torch.ones(2, 2))

    assert no_stride_mode.calls
    assert all(
        tensor.stride is None
        for call in no_stride_mode.calls
        for tensor in call.inputs
    )

    print("BOUNDARY TEST PASS")


if __name__ == "__main__":
    run()
