from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_OUTPUT_DIR = Path("aten_size_reports")
DEFAULT_OUTPUT_FILENAME = "aten_size_profile_size.xlsx"
CONFIG_ENV = "ATEN_SIZE_PROFILER_CONFIG"
CONFIG_FILENAME = "aten_size_profiler.yaml"
VALID_ITERATION_MODES = {"forward", "optimizer_step"}
OUTPUT_EXTENSION = ".xlsx"


@dataclass(frozen=True)
class ProfilerConfig:
    enabled: bool = True
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    record_stride: bool = True
    record_stack: bool = False
    record_scalar_args: bool = True
    target_iter: int | None = None
    iteration_mode: str = "forward"
    config_file: Path | None = None

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _parse_optional_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer or null") from error


def _parse_iteration_mode(value: Any) -> str:
    iteration_mode = str(value)
    if iteration_mode not in VALID_ITERATION_MODES:
        allowed = ", ".join(sorted(VALID_ITERATION_MODES))
        raise ValueError(
            f"iteration_mode must be one of: {allowed}"
        )
    return iteration_mode


def validate_output_filename(value: Any) -> str:
    """Return ``value`` if it is a bare ``*.xlsx`` file name, else raise."""
    if not isinstance(value, str):
        raise ValueError("output_filename must be a string")
    if not value or value != value.strip():
        raise ValueError(
            f"output_filename must be a non-empty name without leading or "
            f"trailing whitespace: {value!r}"
        )
    if "/" in value or "\\" in value:
        raise ValueError(
            f"output_filename must be a file name without directories; "
            f"use output_dir for the directory: {value!r}"
        )
    name = Path(value)
    if name.suffix.lower() != OUTPUT_EXTENSION or not name.stem:
        raise ValueError(
            f"output_filename must end with {OUTPUT_EXTENSION} and have a "
            f"non-empty stem: {value!r}"
        )
    return value


def validate_output_path(path: str | Path) -> Path:
    """Return ``path`` as a Path if it names a ``*.xlsx`` file, else raise."""
    destination = Path(path)
    if (
        destination.suffix.lower() != OUTPUT_EXTENSION
        or not destination.stem
    ):
        raise ValueError(
            f"Excel output path must end with {OUTPUT_EXTENSION} and have a "
            f"non-empty file name: {str(path)!r}"
        )
    return destination


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return dict(values)


def load_config(
    environ: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ProfilerConfig:
    environment = dict(os.environ if environ is None else environ)
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    configured_path = environment.get(CONFIG_ENV)
    if configured_path:
        config_path = Path(configured_path).expanduser()
        if not config_path.is_absolute():
            config_path = working_directory / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"configuration file not found: {config_path}")
    else:
        candidate = working_directory / CONFIG_FILENAME
        config_path = candidate if candidate.is_file() else None

    values = _load_yaml(config_path) if config_path is not None else {}
    environment_overrides = {
        "enabled": "ATEN_SIZE_PROFILER_ENABLED",
        "output_dir": "ATEN_SIZE_PROFILER_OUTPUT_DIR",
        "output_filename": "ATEN_SIZE_PROFILER_OUTPUT_FILENAME",
        "record_stride": "ATEN_SIZE_PROFILER_RECORD_STRIDE",
        "record_stack": "ATEN_SIZE_PROFILER_RECORD_STACK",
        "record_scalar_args": "ATEN_SIZE_PROFILER_RECORD_SCALAR_ARGS",
        "target_iter": "ATEN_SIZE_PROFILER_TARGET_ITER",
        "iteration_mode": "ATEN_SIZE_PROFILER_ITERATION_MODE",
    }
    for key, environment_name in environment_overrides.items():
        if environment_name in environment:
            values[key] = environment[environment_name]

    output_dir = Path(values.get("output_dir", DEFAULT_OUTPUT_DIR)).expanduser()
    target_iter = _parse_optional_int(values.get("target_iter"), "target_iter")
    return ProfilerConfig(
        enabled=_parse_bool(values.get("enabled", True), "enabled"),
        output_dir=output_dir,
        output_filename=validate_output_filename(
            str(values.get("output_filename", DEFAULT_OUTPUT_FILENAME))
        ),
        record_stride=_parse_bool(
            values.get("record_stride", True),
            "record_stride",
        ),
        record_stack=_parse_bool(
            values.get("record_stack", False),
            "record_stack",
        ),
        record_scalar_args=_parse_bool(
            values.get("record_scalar_args", True),
            "record_scalar_args",
        ),
        target_iter=target_iter,
        iteration_mode=_parse_iteration_mode(
            values.get("iteration_mode", "forward")
        ),
        config_file=config_path,
    )
