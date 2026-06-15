from __future__ import annotations

import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from drakeuni.batch_env import DrakeEnvPool, batch_available, batch_import_error

from .mjcf_model_parser import (
    ROOT_QVEL_DIM,
    parse_mjcf_model_contract,
    read_keyframe_qpos,
    tracked_points_as_pool_inputs,
)
from .types import DrakeModelInfo, DrakeRuntimeConfig, DrakeRuntimeDiagnostics


class DrakeBatchRuntime:
    def __init__(self, config: DrakeRuntimeConfig) -> None:
        if config.mode != "batch":
            raise ValueError(f"DrakeBatchRuntime requires mode='batch', got {config.mode!r}")
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
        home_qpos = read_keyframe_qpos(self._model_file, "home")
        if home_qpos is None:
            raise ValueError(f"DrakeBatchRuntime requires keyframe 'home' in {self._model_file}")
        self._home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self._kp = float(config.kp)
        self._kd = float(config.kd)
        self._nthread = _resolve_nthread(self._num_envs, int(config.nthread))
        push_body_name = config.push_body_name or config.base_name
        tracked_body_indices, tracked_offsets = tracked_points_as_pool_inputs(self._model_contract)
        self._pool = DrakeEnvPool(
            self._model_file,
            self._num_envs,
            self._sim_dt,
            self._model_contract.ctrl_limits,
            self._model_contract.torque_limits,
            self._model_contract.body_index(config.base_name),
            self._model_contract.body_index(push_body_name),
            tracked_body_indices,
            tracked_offsets,
            self._kp,
            self._kd,
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
            joint_ranges=self._model_contract.joint_ranges.copy(),
            sensor_names=self._model_contract.sensor_names,
        )
        self._physics_state = np.zeros((self._num_envs, int(self._pool.state_dim)), dtype=np.float64)
        self._sensor_packet: dict[str, np.ndarray] = {}
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
            joint_ranges=info.joint_ranges.copy(),
            sensor_names=info.sensor_names,
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
        output = self._pool.reset(indices, self._pack_state_rows(qpos_rows, qvel_rows))
        self._apply_output(output)

    def step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
        push_force: np.ndarray | None = None,
    ) -> dict[str, Any]:
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self._model_info.nu):
            raise ValueError(
                f"ctrl must have shape ({self._num_envs}, {self._model_info.nu}), got {values.shape}"
            )
        push = None if push_force is None else np.asarray(push_force, dtype=np.float64)
        if push is not None and push.shape != (self._num_envs, 3):
            raise ValueError(f"push_force must have shape ({self._num_envs}, 3), got {push.shape}")
        output = self._pool.step(self._physics_state, int(nsteps), values, push)
        self._apply_output(output)
        return {
            "state": self.physics_state(),
            "sensor": {name: values.copy() for name, values in self._sensor_packet.items()},
            "timing": dict(output.get("timing", {})),
        }

    def physics_state(self) -> np.ndarray:
        return self._physics_state.copy()

    def sensor(self, name: str) -> np.ndarray:
        try:
            return self._sensor_packet[name].copy()
        except KeyError as exc:
            raise KeyError(f"Unknown DrakeUni sensor: {name}") from exc

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
        return None

    def _pack_state_rows(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        state = np.zeros((qpos.shape[0], int(self._pool.state_dim)), dtype=np.float64)
        state[:, 1 : 1 + self._model_info.nq] = qpos
        state[:, 1 + self._model_info.nq :] = qvel
        return state

    def _apply_output(self, output: dict[str, Any]) -> None:
        self._physics_state = np.asarray(output["state"], dtype=np.float64).copy()
        raw_sensor = output.get("sensor", {})
        packet = {key: np.asarray(value, dtype=np.float64).copy() for key, value in raw_sensor.items()}
        feet_pos = packet.get("feet_pos")
        if feet_pos is not None:
            for point_index, point in enumerate(self._model_contract.tracked_points):
                if point_index < feet_pos.shape[1]:
                    packet[point.name] = feet_pos[:, point_index, :]
        feet_contact = packet.get("feet_contact_force")
        if feet_contact is not None:
            for sensor in self._model_contract.contact_sensors:
                if sensor.tracked_index is not None and sensor.tracked_index < feet_contact.shape[1]:
                    packet[sensor.name] = feet_contact[:, sensor.tracked_index, :]
                else:
                    packet[sensor.name] = np.zeros((self._num_envs, 3), dtype=np.float64)
        packet.setdefault("position", packet["base_pos"])
        self._sensor_packet = packet


def _resolve_nthread(num_envs: int, requested: int) -> int:
    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)
