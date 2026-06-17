"""Python-facing DrakeUni runtime interface.

UniLab should enter DrakeUni through this package. The runtime layer validates
configuration, builds the batch runtime, exposes diagnostics, and keeps the C++
extension behind a stable Python API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import DrakeBatchConfig, DrakeModelInfo, DrakeRuntimeDiagnostics

if TYPE_CHECKING:
    from .batch import DrakeBatchRuntime


def available_backends() -> dict[str, bool]:
    from drakeuni.batch_env import batch_available

    return {
        "batch": bool(batch_available()),
        "debug": False,
    }


def create_runtime(config: DrakeBatchConfig):
    from .batch import DrakeBatchRuntime

    return DrakeBatchRuntime(config)


def batch_diagnostics() -> DrakeRuntimeDiagnostics:
    from drakeuni.batch_env import batch_available, batch_import_error

    detail = batch_import_error()
    return DrakeRuntimeDiagnostics(
        mode="batch",
        available=bool(batch_available()),
        batch_available=bool(batch_available()),
        batch_import_error=None if detail is None else str(detail),
    )


__all__ = [
    "DrakeBatchRuntime",
    "DrakeBatchConfig",
    "DrakeModelInfo",
    "DrakeRuntimeDiagnostics",
    "available_backends",
    "batch_diagnostics",
    "create_runtime",
]


def __getattr__(name: str):
    if name == "DrakeBatchRuntime":
        from .batch import DrakeBatchRuntime

        return DrakeBatchRuntime
    raise AttributeError(name)
