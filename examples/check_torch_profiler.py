from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]

# torch.profiler.profile(with_stack=True) after the iteration hooks are
# installed. The Python tracer reads the local ``self`` of every frame of the
# current ``nn.Module.__call__``; the hooks used to name it ``module``, which
# crashed the process with SIGSEGV. The profile runs after the target
# iteration, as in a training loop that profiles later steps, and once while
# recording.
SOURCE = """
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

import aten_size_profiler

model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
batch = torch.randn(2, 4)


def train_step():
    optimizer.zero_grad()
    model(batch).sum().backward()
    optimizer.step()


with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    train_step()
for _ in range(2):
    train_step()
with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    train_step()
names = {event.name for event in prof.events()}
assert "nn.Module: Linear_0" in names, sorted(names)
assert nn.Module.__call__.__code__.co_varnames[0] == "self"
print("PROFILE OK")
"""


def _environment(**updates: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ATEN_SIZE_PROFILER_")
    }
    environment["PYTHONPATH"] = str(ROOT)
    environment.update(updates)
    return environment


def _check(mode: str, output_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", SOURCE],
        cwd=ROOT,
        env=_environment(
            ATEN_SIZE_PROFILER_ENABLED="true",
            ATEN_SIZE_PROFILER_OUTPUT_DIR=str(output_dir),
            ATEN_SIZE_PROFILER_ITERATION_MODE=mode,
            ATEN_SIZE_PROFILER_TARGET_ITER="0",
        ),
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        f"{mode} exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PROFILE OK" in result.stdout, result.stdout
    workbook = load_workbook(
        output_dir / "aten_size_profile_size.xlsx",
        read_only=True,
        data_only=True,
    )
    metadata = {
        row[0]: row[1]
        for row in workbook["metadata"].iter_rows(min_row=2, values_only=True)
    }
    assert metadata["iteration_complete"] is True, metadata
    assert metadata["target_iter"] == 0, metadata


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="aten-torch-profiler-") as path:
        for mode in ("forward", "optimizer_step"):
            _check(mode, Path(path) / mode)
    print("TORCH PROFILER TEST PASS")


if __name__ == "__main__":
    run()
