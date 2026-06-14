from __future__ import annotations

import sys


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


if _pydrake_loaded():
    _BATCH_IMPORT_ERROR: ImportError | None = ImportError(
        "DrakeEnvPool batch extension cannot be imported after pydrake has already "
        "been imported in this process. Start a fresh process for DrakeUni batch rollout."
    )
    DrakeEnvPool = None  # type: ignore[assignment]

    def batch_available() -> bool:
        return False

else:
    try:
        from ._drake_env_pool import DrakeEnvPool, batch_available
    except ImportError as exc:  # pragma: no cover - optional local extension.
        _BATCH_IMPORT_ERROR = exc
        DrakeEnvPool = None  # type: ignore[assignment]

        def batch_available() -> bool:
            return False

    else:
        _BATCH_IMPORT_ERROR = None


def batch_import_error() -> ImportError | None:
    return _BATCH_IMPORT_ERROR


__all__ = ["DrakeEnvPool", "batch_available", "batch_import_error"]
