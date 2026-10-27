__version__ = "0.2.1"

from .collector import (
    AtenCollector,
    CallRecord,
    OperatorAggregate,
    TensorInfo,
    collect_input_tensors,
)
from .config import ProfilerConfig, load_config
from .distributed import RankInfo, get_rank_info, ranked_output_path
from .dispatch_mode import AtenSizeMode
from .exporter import export_collector
from .runtime import (
    get_collector,
    get_iteration_status,
    get_status,
    is_installed,
    is_recording,
    pause_profiler,
    resume_profiler,
    start_profiler,
    stop_profiler,
)

_AUTO_MODE = start_profiler()

__all__ = [
    "AtenCollector",
    "AtenSizeMode",
    "CallRecord",
    "OperatorAggregate",
    "ProfilerConfig",
    "RankInfo",
    "TensorInfo",
    "collect_input_tensors",
    "export_collector",
    "__version__",
    "get_collector",
    "get_iteration_status",
    "get_rank_info",
    "get_status",
    "is_installed",
    "is_recording",
    "load_config",
    "pause_profiler",
    "resume_profiler",
    "ranked_output_path",
    "start_profiler",
    "stop_profiler",
]
