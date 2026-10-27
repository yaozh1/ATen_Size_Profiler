from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SMOKE_CHECKS = [
    "examples/check_basic.py",
    "examples/check_boundaries.py",
]

CORE_CHECKS = [
    *SMOKE_CHECKS,
    "examples/check_export.py",
    "examples/check_auto.py",
    "examples/check_iteration.py",
    "examples/check_optimizer_lifecycle.py",
    "examples/check_output_filename.py",
    "examples/check_cross_review_batch2.py",
    "examples/check_cross_review_batch3.py",
    "examples/check_cross_review_batch4.py",
    "examples/check_cross_review_round2.py",
    "examples/check_cross_review_round3.py",
    "examples/check_codex_p1.py",
    "examples/check_mode_stack.py",
]

FULL_CHECKS = [
    *CORE_CHECKS,
    "examples/check_stability.py",
    "examples/check_real_models.py",
]


def _clean_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ATEN_SIZE_PROFILER_")
    }
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def _run_check(script: str) -> None:
    command = [sys.executable, str(ROOT / script)]
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_clean_environment(),
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"regression check failed ({result.returncode}): {script}"
        )


def _run_distributed() -> None:
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        str(ROOT / "examples/check_distributed.py"),
    ]
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_clean_environment(),
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"distributed regression check failed ({result.returncode})"
        )


def _run_installed() -> None:
    """Build a wheel from the current tree and run it outside the source tree."""
    import shutil
    import tempfile

    import torch

    environment = _clean_environment()
    environment.pop("PYTHONPATH")
    with tempfile.TemporaryDirectory(prefix="aten-size-installed-") as path:
        work_dir = Path(path)
        source_dir = work_dir / "source"
        wheel_dir = work_dir / "dist"
        venv_dir = work_dir / "venv"
        venv_python = venv_dir / "bin" / "python"
        # Build from a clean copy so stale build/ output cannot leak in.
        source_dir.mkdir()
        for name in ("pyproject.toml", "README.md"):
            shutil.copy(ROOT / name, source_dir / name)
        shutil.copytree(
            ROOT / "aten_size_profiler",
            source_dir / "aten_size_profiler",
            ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
        )
        commands = [
            [
                sys.executable, "-m", "pip", "wheel", "--no-deps",
                "--no-build-isolation", str(source_dir), "-w", str(wheel_dir),
            ],
            [
                sys.executable, "-m", "venv", "--system-site-packages",
                str(venv_dir),
            ],
        ]
        for command in commands:
            print(f"\n$ {' '.join(command)}", flush=True)
            subprocess.run(
                command, cwd=work_dir, env=environment, check=True,
            )
        wheels = list(wheel_dir.glob("aten_size_profiler-*.whl"))
        assert len(wheels) == 1, wheels
        command = [
            str(venv_python), "-m", "pip", "install", "--no-deps",
            "--force-reinstall", str(wheels[0]),
        ]
        print(f"\n$ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=work_dir, env=environment, check=True)

        script = work_dir / "check_installed_package.py"
        shutil.copy(ROOT / "examples/check_installed_package.py", script)
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            for mode in ("forward", "optimizer_step"):
                name = f"{device}_{mode}"
                output_dir = work_dir / "reports" / name
                config_path = work_dir / f"{name}.yaml"
                config_path.write_text(
                    "enabled: true\n"
                    f"output_dir: {output_dir}\n"
                    f"output_filename: installed_{name}.xlsx\n"
                    f"iteration_mode: {mode}\n"
                    "target_iter: 0\n",
                    encoding="utf-8",
                )
                check_environment = dict(environment)
                check_environment["ATEN_SIZE_PROFILER_CONFIG"] = str(
                    config_path
                )
                check_environment["ATEN_SIZE_PROFILER_OUTPUT_DIR"] = str(
                    output_dir
                )
                check_environment["ATEN_SIZE_PROFILER_OUTPUT_FILENAME"] = (
                    f"installed_{name}.xlsx"
                )
                command = [
                    str(venv_python), str(script),
                    "--device", device, "--mode", mode,
                ]
                print(f"\n$ {' '.join(command)}", flush=True)
                result = subprocess.run(
                    command, cwd=work_dir, env=check_environment, text=True,
                )
                if result.returncode != 0:
                    raise SystemExit(
                        f"installed package check failed "
                        f"({result.returncode}): {name}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("smoke", "core", "full"),
        default="core",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="also run the CPU two-Rank torchrun check",
    )
    parser.add_argument(
        "--installed",
        action="store_true",
        help="also build a wheel and run it in a venv outside the source tree",
    )
    args = parser.parse_args()

    checks = {
        "smoke": SMOKE_CHECKS,
        "core": CORE_CHECKS,
        "full": FULL_CHECKS,
    }[args.suite]
    for check in checks:
        _run_check(check)
    if args.distributed:
        _run_distributed()
    if args.installed:
        _run_installed()
    print(f"\nREGRESSION {args.suite.upper()} PASS")


if __name__ == "__main__":
    main()
