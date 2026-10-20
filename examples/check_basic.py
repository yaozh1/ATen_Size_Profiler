from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ATEN_SIZE_PROFILER_ENABLED"] = "false"

from aten_size_profiler import AtenSizeMode, collect_input_tensors


def run() -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left = torch.randn(32, 128, device=device)
    right = torch.randn(128, 256, device=device)

    baseline_left = left.detach().clone().requires_grad_()
    baseline_right = right.detach().clone().requires_grad_()
    baseline_output = torch.relu(torch.mm(baseline_left, baseline_right))
    baseline_loss = baseline_output.square().mean()
    baseline_loss.backward()

    measured_left = left.detach().clone().requires_grad_()
    measured_right = right.detach().clone().requires_grad_()
    with AtenSizeMode() as mode:
        measured_output = torch.relu(torch.mm(measured_left, measured_right))
        measured_loss = measured_output.square().mean()
        measured_loss.backward()

    torch.testing.assert_close(measured_output, baseline_output)
    torch.testing.assert_close(measured_loss, baseline_loss)
    torch.testing.assert_close(measured_left.grad, baseline_left.grad)
    torch.testing.assert_close(measured_right.grad, baseline_right.grad)

    mm_calls = [call for call in mode.calls if call.op_name == "aten.mm.default"]
    relu_calls = [
        call for call in mode.calls if call.op_name == "aten.relu.default"
    ]
    assert mm_calls, "expected aten.mm.default to be collected"
    assert relu_calls, "expected aten.relu.default to be collected"
    assert any(
        call.op_name.endswith("backward.default")
        for call in mode.calls
    ), "expected at least one backward ATen operator"

    assert mm_calls[0].inputs[0].shape == (32, 128)
    assert mm_calls[0].inputs[1].shape == (128, 256)
    assert mm_calls[0].inputs[0].path == "args[0]"
    assert mm_calls[0].inputs[0].dtype == "torch.float32"
    assert mm_calls[0].inputs[0].device == str(measured_left.device)
    assert mm_calls[0].inputs[0].stride == (128, 1)
    assert mm_calls[0].inputs[0].numel == 32 * 128
    assert mm_calls[0].inputs[0].logical_bytes == 32 * 128 * 4

    nested_inputs = collect_input_tensors(
        ({"batch": [left, (right,)]},),
        {"extra": {"tensor": left}},
    )
    assert [tensor.path for tensor in nested_inputs] == [
        "args[0]['batch'][0]",
        "args[0]['batch'][1][0]",
        "kwargs['extra']['tensor']",
    ]

    print(f"device: {device}")
    print(f"collected calls: {len(mode.calls)}")
    print(
        "operators:",
        ", ".join(
            f"{name}={aggregate.call_count}"
            for name, aggregate in mode.aggregates.items()
        ),
    )
    print("TEST PASS")


if __name__ == "__main__":
    run()
