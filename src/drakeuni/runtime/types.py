from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrakeBatchConfig:
    model_file: str
    num_envs: int
    sim_dt: float
    nthread: int = 0


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
    actuator_stiffness: np.ndarray
    actuator_damping: np.ndarray
    actuator_qpos_adr: np.ndarray
    actuator_qvel_adr: np.ndarray
    joint_ranges: np.ndarray
    num_bodies: int
    site_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_qpos_adr: np.ndarray
    joint_qvel_adr: np.ndarray
    joint_qpos_dim: np.ndarray
    joint_qvel_dim: np.ndarray
    sensor_names: tuple[str, ...]
    sensor_adr: np.ndarray
    sensor_dim: np.ndarray
    nsensordata: int
