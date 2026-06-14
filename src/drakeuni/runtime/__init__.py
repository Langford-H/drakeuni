from __future__ import annotations

from typing import TYPE_CHECKING

from .types import DrakeModelInfo, DrakeRuntimeConfig, DrakeRuntimeDiagnostics

if TYPE_CHECKING:
    from .batch import DrakeBatchRuntime


def available_backends() -> dict[str, bool]:
    from drakeuni.batch_env import batch_available

    return {
        "batch": bool(batch_available()),
        "debug": False,
    }


def create_runtime(config: DrakeRuntimeConfig):
    if config.mode == "batch":
        from .batch import DrakeBatchRuntime

        return DrakeBatchRuntime(config)
    if config.mode == "debug":
        raise NotImplementedError(
            "DrakeUni debug runtime has not been migrated yet; UniLab still owns "
            "pydrake debug/replay in this Go1 batch-runtime migration."
        )
    raise ValueError(f"Unknown Drake runtime mode: {config.mode!r}")


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
    "DrakeModelInfo",
    "DrakeRuntimeConfig",
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
