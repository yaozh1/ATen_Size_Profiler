from __future__ import annotations

import gc
import os
import tempfile
import weakref

import torch
from torch import nn


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="aten-optimizer-lifecycle-") as path:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ATEN_SIZE_PROFILER_")
        }
        environment.update(
            {
                "ATEN_SIZE_PROFILER_ENABLED": "true",
                "ATEN_SIZE_PROFILER_OUTPUT_DIR": path,
                "ATEN_SIZE_PROFILER_OUTPUT_FILENAME": "lifecycle.xlsx",
                "ATEN_SIZE_PROFILER_TARGET_ITER": "0",
                "ATEN_SIZE_PROFILER_ITERATION_MODE": "optimizer_step",
            }
        )
        original_environment = os.environ.copy()
        os.environ.clear()
        os.environ.update(environment)
        try:
            import aten_size_profiler as profiler

            model = nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1,
            )
            output = model(torch.ones(2, 2))
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            assert profiler.get_iteration_status()["phase"] == "FINISHED"

            optimizer_ref = weakref.ref(optimizer)
            model_ref = weakref.ref(model)
            parameter = next(model.parameters())
            parameter_ref = weakref.ref(parameter)
            del scheduler, optimizer, model, parameter, output, loss
            gc.collect()
            assert optimizer_ref() is None
            assert model_ref() is None
            assert parameter_ref() is None
            profiler.stop_profiler(export=False)
        finally:
            os.environ.clear()
            os.environ.update(original_environment)
    print("OPTIMIZER LIFECYCLE TEST PASS")


if __name__ == "__main__":
    run()
