from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .batch_env import DrakeEnvPool, batch_available, batch_import_error
    from .runtime import (
        DrakeModelInfo,
        DrakeRuntimeConfig,
        DrakeRuntimeDiagnostics,
        available_backends,
        create_runtime,
    )

__all__ = [
    "DrakeEnvPool",
    "DrakeModelInfo",
    "DrakeRuntimeConfig",
    "DrakeRuntimeDiagnostics",
    "available_backends",
    "batch_available",
    "batch_import_error",
    "create_runtime",
]


def __getattr__(name: str):
    if name in {"DrakeEnvPool", "batch_available", "batch_import_error"}:
        from . import batch_env

        return getattr(batch_env, name)
    if name in {
        "DrakeModelInfo",
        "DrakeRuntimeConfig",
        "DrakeRuntimeDiagnostics",
        "available_backends",
        "create_runtime",
    }:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(name)
