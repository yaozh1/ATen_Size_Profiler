from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _environment(**updates: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ATEN_SIZE_PROFILER_")
    }
    environment["PYTHONPATH"] = str(ROOT)
    environment.update(updates)
    return environment


def _run(
    name: str,
    source: str,
    **updates: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        env=_environment(**updates),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{name} failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def run() -> None:
    _run(
        "disabled-cond",
        """
        import torch
        import aten_size_profiler as profiler

        value = torch.ones(2)

        def true_fn(inputs):
            return inputs + 1

        def false_fn(inputs):
            return inputs - 1

        result = torch.cond(
            torch.tensor(True),
            true_fn,
            false_fn,
            (value,),
        )
        torch.testing.assert_close(result, value + 1)
        assert profiler.is_installed()
        assert not profiler.is_recording()
        assert profiler.stop_profiler(export=False) is None
        """,
        ATEN_SIZE_PROFILER_ENABLED="false",
    )

    _run(
        "nested-tensor",
        """
        import torch

        nested = torch.nested.nested_tensor(
            [torch.randn(2, 4), torch.randn(3, 4)]
        )
        import aten_size_profiler

        padded = torch.nested.to_padded_tensor(nested, 0.0)
        assert padded.shape == (2, 3, 4)
        """,
        ATEN_SIZE_PROFILER_ENABLED="true",
    )

    invalid = subprocess.run(
        [sys.executable, "-c", "import aten_size_profiler"],
        cwd=ROOT,
        env=_environment(
            ATEN_SIZE_PROFILER_ENABLED="true",
            ATEN_SIZE_PROFILER_ITERATION_MODE="invalid",
        ),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert invalid.returncode != 0
    assert "iteration_mode must be one of" in invalid.stderr

    removed_manual = subprocess.run(
        [sys.executable, "-c", "import aten_size_profiler"],
        cwd=ROOT,
        env=_environment(
            ATEN_SIZE_PROFILER_ENABLED="true",
            ATEN_SIZE_PROFILER_ITERATION_MODE="manual",
        ),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert removed_manual.returncode != 0
    assert "iteration_mode must be one of" in removed_manual.stderr
    assert "manual" not in removed_manual.stderr.split(
        "iteration_mode must be one of", 1
    )[1].splitlines()[0]

    _run(
        "invalid-direct-config",
        """
        import aten_size_profiler as profiler
        from aten_size_profiler import ProfilerConfig

        profiler.stop_profiler(export=False)
        try:
            profiler.start_profiler(ProfilerConfig(iteration_mode="invalid"))
        except ValueError as error:
            assert "iteration_mode must be one of" in str(error)
        else:
            raise AssertionError("invalid direct config was accepted")
        try:
            profiler.start_profiler(ProfilerConfig(iteration_mode="manual"))
        except ValueError as error:
            assert "iteration_mode must be one of" in str(error)
        else:
            raise AssertionError("removed manual mode was accepted")
        """,
        ATEN_SIZE_PROFILER_ENABLED="true",
    )
    print("CROSS REVIEW BATCH 2 TEST PASS")


if __name__ == "__main__":
    run()
