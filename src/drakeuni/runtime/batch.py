from __future__ import annotations

import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from drakeuni.batch_env import DrakeEnvPool, batch_available, batch_import_error

from .mjcf_model_parser import (
    ROOT_QVEL_DIM,
    DrakeCompatibleMjcf,
    materialize_drake_compatible_mjcf,
    parse_mjcf_model_contract,
    read_keyframe_qpos,
    sensor_frames_as_pool_inputs,
)
from .types import DrakeBatchConfig, DrakeModelInfo, DrakeRuntimeDiagnostics


class DrakeBatchRuntime:
    def __init__(self, config: DrakeBatchConfig) -> None:
        if int(config.num_envs) < 1:
            raise ValueError(f"DrakeBatchRuntime requires num_envs >= 1, got {config.num_envs}")
        if DrakeEnvPool is None or not bool(batch_available()):
            detail = batch_import_error()
            message = "DrakeEnvPool batch extension has not been built."
            if detail is not None:
                message = f"{message} Import error: {detail}"
            raise ImportError(message) from detail

        self._config = config
        self._num_envs = int(config.num_envs)
        self._sim_dt = float(config.sim_dt)
        self._model_file = str(Path(config.model_file).expanduser())
        self._model_contract = parse_mjcf_model_contract(self._model_file)
        self._drake_model: DrakeCompatibleMjcf = materialize_drake_compatible_mjcf(
            self._model_file
        )
        home_qpos = read_keyframe_qpos(self._model_file, "home")
        if home_qpos is None:
            raise ValueError(f"DrakeBatchRuntime requires keyframe 'home' in {self._model_file}")
        self._home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self._nthread = _resolve_nthread(self._num_envs, int(config.nthread))
        sensor_frame_body_indices, sensor_frame_offsets = sensor_frames_as_pool_inputs(
            self._model_contract
        )
        self._pool = DrakeEnvPool(
            self._drake_model.model_file,
            self._num_envs,
            self._sim_dt,
            self._model_contract.ctrl_limits,
            self._model_contract.torque_limits,
            self._model_contract.actuator_stiffness,
            self._model_contract.actuator_damping,
            sensor_frame_body_indices,
            sensor_frame_offsets,
            self._model_contract.sensor_type,
            self._model_contract.sensor_index,
            self._model_contract.sensor_adr,
            self._model_contract.sensor_dim,
            self._model_contract.nsensordata,
            self._nthread,
        )

        nv = int(self._pool.state_dim) - 1 - int(self._home_qpos.size)
        if nv <= ROOT_QVEL_DIM:
            raise RuntimeError(f"DrakeEnvPool batch runtime returned invalid nv={nv}")
        self._home_qvel = np.zeros(nv, dtype=np.float64)
        self._model_info = DrakeModelInfo(
            nq=int(self._home_qpos.size),
            nv=nv,
            nu=int(self._pool.control_dim),
            home_qpos=self._home_qpos.copy(),
            home_qvel=self._home_qvel.copy(),
            ctrl_limits=self._model_contract.ctrl_limits.copy(),
            torque_limits=self._model_contract.torque_limits.copy(),
            actuator_stiffness=self._model_contract.actuator_stiffness.copy(),
            actuator_damping=self._model_contract.actuator_damping.copy(),
            joint_ranges=self._model_contract.joint_ranges.copy(),
            num_bodies=int(self._pool.num_bodies),
            sensor_names=self._model_contract.sensor_names,
            sensor_adr=self._model_contract.sensor_adr.copy(),
            sensor_dim=self._model_contract.sensor_dim.copy(),
            nsensordata=self._model_contract.nsensordata,
        )
        self._physics_state = np.zeros((self._num_envs, int(self._pool.state_dim)), dtype=np.float64)
        self._sensor_data = np.zeros(
            (self._num_envs, self._model_info.nsensordata),
            dtype=np.float64,
        )
        qpos = np.broadcast_to(self._home_qpos, (self._num_envs, self._model_info.nq)).copy()
        qvel = np.zeros((self._num_envs, self._model_info.nv), dtype=np.float64)
        self.reset(np.arange(self._num_envs, dtype=np.int32), qpos, qvel)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def nthread(self) -> int:
        return self._nthread

    @property
    def model_file(self) -> str:
        return self._model_file

    def model_info(self) -> DrakeModelInfo:
        info = self._model_info
        return DrakeModelInfo(
            nq=info.nq,
            nv=info.nv,
            nu=info.nu,
            home_qpos=info.home_qpos.copy(),
            home_qvel=info.home_qvel.copy(),
            ctrl_limits=info.ctrl_limits.copy(),
            torque_limits=info.torque_limits.copy(),
            actuator_stiffness=info.actuator_stiffness.copy(),
            actuator_damping=info.actuator_damping.copy(),
            joint_ranges=info.joint_ranges.copy(),
            num_bodies=info.num_bodies,
            sensor_names=info.sensor_names,
            sensor_adr=info.sensor_adr.copy(),
            sensor_dim=info.sensor_dim.copy(),
            nsensordata=info.nsensordata,
        )

    def reset(self, env_ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        indices = np.asarray(env_ids, dtype=np.int32)
        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        if indices.ndim != 1:
            raise ValueError(f"env_ids must be one-dimensional, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self._num_envs):
            raise IndexError(f"env_ids must be in [0, {self._num_envs - 1}], got {indices.tolist()}")
        if qpos_rows.shape != (indices.size, self._model_info.nq):
            raise ValueError(f"qpos must have shape ({indices.size}, {self._model_info.nq})")
        if qvel_rows.shape != (indices.size, self._model_info.nv):
            raise ValueError(f"qvel must have shape ({indices.size}, {self._model_info.nv})")
        output = self._pool.reset(indices, self._pack_state_rows(qpos_rows, qvel_rows), True)
        self._apply_output(output)
        self._apply_sensor_data(output)

    def step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
        body_forces: np.ndarray | None = None,
    ) -> dict[str, Any]:
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self._model_info.nu):
            raise ValueError(
                f"ctrl must have shape ({self._num_envs}, {self._model_info.nu}), got {values.shape}"
            )
        forces = None if body_forces is None else np.asarray(body_forces, dtype=np.float64)
        expected_force_shape = (self._num_envs, self._model_info.num_bodies, 3)
        if forces is not None and forces.shape != expected_force_shape:
            raise ValueError(f"body_forces must have shape {expected_force_shape}, got {forces.shape}")
        output = self._pool.step(self._physics_state, int(nsteps), values, forces, True)
        self._apply_output(output)
        self._apply_sensor_data(output)
        return {
            "state": self.physics_state(),
            "sensor_data": self.sensor_data(),
            "timing": dict(output.get("timing", {})),
        }

    def physics_state(self) -> np.ndarray:
        return self._physics_state.copy()

    def sensor_data(self) -> np.ndarray:
        return self._sensor_data.copy()

    def compute_body_state(
        self,
        body_ids: np.ndarray | list[int] | tuple[int, ...],
    ) -> dict[str, np.ndarray]:
        ids = np.asarray(body_ids, dtype=np.int32)
        output = self._pool.compute_body_state(self._physics_state, ids)
        return {name: np.asarray(value, dtype=np.float64).copy() for name, value in output.items()}

    def body_ids(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [self._model_contract.body_index(str(name)) for name in names],
            dtype=np.int32,
        )

    def diagnostics(self) -> DrakeRuntimeDiagnostics:
        detail = batch_import_error()
        return DrakeRuntimeDiagnostics(
            mode="batch",
            available=DrakeEnvPool is not None and bool(batch_available()),
            batch_available=bool(batch_available()),
            batch_import_error=None if detail is None else str(detail),
            pydrake_loaded=_pydrake_loaded(),
            nthread=self._nthread,
            workspace_count=int(getattr(self._pool, "workspace_count", 0)),
            num_filtered_geometries=int(getattr(self._pool, "num_filtered_geometries", 0)),
        )

    def render_capabilities(self) -> dict[str, bool]:
        return {
            "interactive": False,
            "video_capture": False,
            "physics_state_playback": True,
        }

    def close(self) -> None:
        self._drake_model.close()

    def _pack_state_rows(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        state = np.zeros((qpos.shape[0], int(self._pool.state_dim)), dtype=np.float64)
        state[:, 1 : 1 + self._model_info.nq] = qpos
        state[:, 1 + self._model_info.nq :] = qvel
        return state

    def _apply_output(self, output: dict[str, Any]) -> None:
        self._physics_state = np.asarray(output["state"], dtype=np.float64).copy()

    def _apply_sensor_data(self, output: dict[str, Any]) -> None:
        if "sensor_data" not in output:
            raise RuntimeError("DrakeEnvPool output did not include sensor_data")
        sensor_data = np.asarray(output["sensor_data"], dtype=np.float64)
        expected_shape = (self._num_envs, self._model_info.nsensordata)
        if sensor_data.shape != expected_shape:
            raise RuntimeError(
                f"DrakeEnvPool sensor_data must have shape {expected_shape}, got {sensor_data.shape}"
            )
        self._sensor_data = sensor_data.copy()


def _resolve_nthread(num_envs: int, requested: int) -> int:
    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)
