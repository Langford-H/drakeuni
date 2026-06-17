from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DrakeRuntimeConfig:
    model_file: str
    num_envs: int
    sim_dt: float
    mode: Literal["batch", "debug"]
    base_name: str
    push_body_name: str | None = None
    kp: float = 35.0
    kd: float = 0.5
    nthread: int = 0

    def __post_init__(self) -> None:
        if not self.base_name:
            raise ValueError("DrakeRuntimeConfig requires a task-provided base_name")


@dataclass(frozen=True)
class DrakeRuntimeDiagnostics:
    mode: str
    available: bool
    batch_available: bool
    batch_import_error: str | None = None
    pydrake_loaded: bool = False
    nthread: int | None = None
    workspace_count: int | None = None
    num_filtered_geometries: int | None = None


@dataclass(frozen=True)
class DrakeModelInfo:
    nq: int
    nv: int
    nu: int
    home_qpos: np.ndarray
    home_qvel: np.ndarray
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    joint_ranges: np.ndarray
    sensor_names: tuple[str, ...]
    sensor_adr: np.ndarray
    sensor_dim: np.ndarray
    nsensordata: int
