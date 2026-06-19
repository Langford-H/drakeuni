from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import numpy as np

ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6

SENSOR_KIND_GYRO = 0
SENSOR_KIND_ACCELEROMETER = 1
SENSOR_KIND_VELOCIMETER = 2
SENSOR_KIND_FRAME_POS = 3
SENSOR_KIND_FRAME_LINVEL = 4
SENSOR_KIND_FRAME_ANGVEL = 5
SENSOR_KIND_FRAME_ZAXIS = 6
SENSOR_KIND_CONTACT_FORCE = 7
SENSOR_KIND_CONTACT_FOUND = 8
SENSOR_KIND_JOINT_POS = 9
SENSOR_KIND_JOINT_VEL = 10
SENSOR_KIND_JOINT_ACTUATOR_FORCE = 11
SENSOR_KIND_FRAME_QUAT = 12

ACTUATOR_KIND_POSITION = 0
ACTUATOR_KIND_VELOCITY = 1
ACTUATOR_KIND_MOTOR = 2
ACTUATOR_KIND_DAMPER = 3
ACTUATOR_KIND_GENERAL = 4

JOINT_KIND_FREE = 0
JOINT_KIND_SLIDE = 1
JOINT_KIND_HINGE = 2
JOINT_KIND_BALL = 3

FRAME_SENSOR_KIND_BY_TAG = {
    "gyro": SENSOR_KIND_GYRO,
    "accelerometer": SENSOR_KIND_ACCELEROMETER,
    "velocimeter": SENSOR_KIND_VELOCIMETER,
    "framepos": SENSOR_KIND_FRAME_POS,
    "framelinvel": SENSOR_KIND_FRAME_LINVEL,
    "frameangvel": SENSOR_KIND_FRAME_ANGVEL,
    "framezaxis": SENSOR_KIND_FRAME_ZAXIS,
    "framequat": SENSOR_KIND_FRAME_QUAT,
}

JOINT_SENSOR_KIND_BY_TAG = {
    "jointpos": SENSOR_KIND_JOINT_POS,
    "jointvel": SENSOR_KIND_JOINT_VEL,
    "jointactuatorfrc": SENSOR_KIND_JOINT_ACTUATOR_FORCE,
}

SUPPORTED_SENSOR_TAGS = frozenset({*FRAME_SENSOR_KIND_BY_TAG, *JOINT_SENSOR_KIND_BY_TAG, "contact"})

SUPPORTED_ACTUATOR_TAGS = frozenset(
    {
        "general",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "damper",
        "cylinder",
        "muscle",
        "adhesion",
        "dcmotor",
        "plugin",
    }
)

STATEFUL_OR_NONJOINT_ACTUATORS = frozenset(
    {"intvelocity", "cylinder", "muscle", "adhesion", "dcmotor", "plugin"}
)


@dataclass(frozen=True)
class MjcfFrameSensorContract:
    name: str
    tag: str
    obj_name: str
    obj_type: str
    body_name: str
    body_index: int
    offset: np.ndarray

    @property
    def dim(self) -> int:
        if self.tag == "framequat":
            return 4
        return 3

    @property
    def kind(self) -> int:
        try:
            return FRAME_SENSOR_KIND_BY_TAG[self.tag]
        except KeyError as exc:
            raise ValueError(f"Unsupported MJCF sensor tag {self.tag!r}") from exc


@dataclass(frozen=True)
class MjcfContactSensorContract:
    name: str
    geom1: str
    geom2: str
    data: str
    num: int
    reduce: str | None
    body_name: str | None
    body_index: int | None
    frame_sensor_index: int | None

    @property
    def dim(self) -> int:
        if self.data == "force":
            return 3
        if self.data == "found":
            return 1
        raise ValueError(f"Unsupported MJCF contact sensor data={self.data!r}")

    @property
    def kind(self) -> int:
        if self.data == "force":
            return SENSOR_KIND_CONTACT_FORCE
        if self.data == "found":
            return SENSOR_KIND_CONTACT_FOUND
        raise ValueError(f"Unsupported MJCF contact sensor data={self.data!r}")


@dataclass(frozen=True)
class MjcfJointSensorContract:
    name: str
    tag: str
    joint_name: str
    actuator_index: int

    @property
    def dim(self) -> int:
        return 1

    @property
    def kind(self) -> int:
        try:
            return JOINT_SENSOR_KIND_BY_TAG[self.tag]
        except KeyError as exc:
            raise ValueError(f"Unsupported MJCF joint sensor tag {self.tag!r}") from exc


@dataclass(frozen=True)
class MjcfActuatorContract:
    name: str
    tag: str
    kind: int
    joint_name: str
    ctrl_range: np.ndarray
    force_range: np.ndarray
    joint_range: np.ndarray
    gear: float
    stiffness: float
    damping: float
    gainprm: np.ndarray
    biasprm: np.ndarray

    @property
    def torque_limit(self) -> float:
        return float(np.max(np.abs(self.force_range)))

    @property
    def gain_coefficients(self) -> np.ndarray:
        return _first_three(self.gainprm)

    @property
    def bias_coefficients(self) -> np.ndarray:
        return _first_three(self.biasprm)


@dataclass(frozen=True)
class MjcfGeomCollisionContract:
    name: str
    contype: int
    conaffinity: int

    def collides_with(self, other: "MjcfGeomCollisionContract") -> bool:
        return bool((self.contype & other.conaffinity) or (other.contype & self.conaffinity))


@dataclass(frozen=True)
class MjcfJointLayout:
    name: str
    body_name: str
    kind: int
    qpos_adr: int
    qvel_adr: int
    qpos_dim: int
    qvel_dim: int


@dataclass(frozen=True)
class DrakeMjcfModelContract:
    name: str
    body_indices: dict[str, int]
    joint_layouts: tuple[MjcfJointLayout, ...]
    collision_geoms: tuple[MjcfGeomCollisionContract, ...]
    actuators: tuple[MjcfActuatorContract, ...]
    num_bodies: int
    frame_sensors: tuple[MjcfFrameSensorContract, ...]
    contact_sensors: tuple[MjcfContactSensorContract, ...]
    joint_sensors: tuple[MjcfJointSensorContract, ...]

    @property
    def ctrl_limits(self) -> np.ndarray:
        return np.asarray([actuator.ctrl_range for actuator in self.actuators], dtype=np.float64)

    @property
    def torque_limits(self) -> np.ndarray:
        return np.asarray([actuator.torque_limit for actuator in self.actuators], dtype=np.float64)

    @property
    def actuator_kind(self) -> np.ndarray:
        return np.asarray([actuator.kind for actuator in self.actuators], dtype=np.int32)

    @property
    def actuator_gear(self) -> np.ndarray:
        return np.asarray([actuator.gear for actuator in self.actuators], dtype=np.float64)

    @property
    def actuator_stiffness(self) -> np.ndarray:
        return np.asarray([actuator.stiffness for actuator in self.actuators], dtype=np.float64)

    @property
    def actuator_damping(self) -> np.ndarray:
        return np.asarray([actuator.damping for actuator in self.actuators], dtype=np.float64)

    @property
    def actuator_gainprm(self) -> np.ndarray:
        return np.asarray([actuator.gain_coefficients for actuator in self.actuators], dtype=np.float64)

    @property
    def actuator_biasprm(self) -> np.ndarray:
        return np.asarray([actuator.bias_coefficients for actuator in self.actuators], dtype=np.float64)

    @property
    def joint_ranges(self) -> np.ndarray:
        return np.asarray([actuator.joint_range for actuator in self.actuators], dtype=np.float64)

    @property
    def joint_layout_kind(self) -> np.ndarray:
        return np.asarray([layout.kind for layout in self.joint_layouts], dtype=np.int32)

    @property
    def joint_layout_qpos_adr(self) -> np.ndarray:
        return np.asarray([layout.qpos_adr for layout in self.joint_layouts], dtype=np.int32)

    @property
    def joint_layout_qvel_adr(self) -> np.ndarray:
        return np.asarray([layout.qvel_adr for layout in self.joint_layouts], dtype=np.int32)

    @property
    def joint_layout_qpos_dim(self) -> np.ndarray:
        return np.asarray([layout.qpos_dim for layout in self.joint_layouts], dtype=np.int32)

    @property
    def joint_layout_qvel_dim(self) -> np.ndarray:
        return np.asarray([layout.qvel_dim for layout in self.joint_layouts], dtype=np.int32)

    @property
    def joint_layout_names(self) -> tuple[str, ...]:
        return tuple(layout.name for layout in self.joint_layouts)

    @property
    def joint_layout_body_names(self) -> tuple[str, ...]:
        return tuple(layout.body_name for layout in self.joint_layouts)

    @property
    def collision_filter_geom_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for index, geom in enumerate(self.collision_geoms):
            for other in self.collision_geoms[index + 1 :]:
                if not geom.collides_with(other):
                    pairs.append((geom.name, other.name))
        return tuple(pairs)

    @property
    def collision_filter_geom_names1(self) -> tuple[str, ...]:
        return tuple(pair[0] for pair in self.collision_filter_geom_pairs)

    @property
    def collision_filter_geom_names2(self) -> tuple[str, ...]:
        return tuple(pair[1] for pair in self.collision_filter_geom_pairs)

    @property
    def frame_position_sensors(self) -> tuple[MjcfFrameSensorContract, ...]:
        return tuple(sensor for sensor in self.frame_sensors if sensor.tag == "framepos")

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return _unique(
            (
                *[sensor.name for sensor in self.frame_sensors],
                *[sensor.name for sensor in self.contact_sensors],
                *[sensor.name for sensor in self.joint_sensors],
            )
        )

    @property
    def sensor_dim(self) -> np.ndarray:
        by_name = {
            sensor.name: sensor
            for sensor in (*self.frame_sensors, *self.contact_sensors, *self.joint_sensors)
        }
        dims = [by_name[name].dim for name in self.sensor_names]
        return np.asarray(dims, dtype=np.int32)

    @property
    def sensor_adr(self) -> np.ndarray:
        dims = self.sensor_dim
        if dims.size == 0:
            return np.asarray([], dtype=np.int32)
        return np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(dims[:-1], dtype=np.int32)]
        )

    @property
    def nsensordata(self) -> int:
        return int(np.sum(self.sensor_dim))

    @property
    def sensor_type(self) -> np.ndarray:
        frame_kind = {sensor.name: sensor.kind for sensor in self.frame_sensors}
        contact_kind = {sensor.name: sensor.kind for sensor in self.contact_sensors}
        joint_kind = {sensor.name: sensor.kind for sensor in self.joint_sensors}
        kinds: list[int] = []
        for name in self.sensor_names:
            if name in frame_kind:
                kinds.append(frame_kind[name])
            elif name in contact_kind:
                kinds.append(contact_kind[name])
            elif name in joint_kind:
                kinds.append(joint_kind[name])
            else:  # pragma: no cover - guarded by parser construction.
                raise ValueError(f"Unknown DrakeUni MJCF sensor {name!r}")
        return np.asarray(kinds, dtype=np.int32)

    @property
    def sensor_index(self) -> np.ndarray:
        frame_index = {sensor.name: i for i, sensor in enumerate(self.frame_sensors)}
        contact_index = {sensor.name: sensor.body_index for sensor in self.contact_sensors}
        joint_index = {sensor.name: sensor.actuator_index for sensor in self.joint_sensors}
        indices: list[int] = []
        for name in self.sensor_names:
            if name in frame_index:
                indices.append(frame_index[name])
            elif name in contact_index:
                value = contact_index[name]
                indices.append(-1 if value is None else int(value))
            elif name in joint_index:
                indices.append(int(joint_index[name]))
            else:
                indices.append(-1)
        return np.asarray(indices, dtype=np.int32)

    def body_index(self, name: str) -> int:
        try:
            return self.body_indices[name]
        except KeyError as exc:
            raise ValueError(f"DrakeUni model contract does not contain body {name!r}") from exc


@dataclass(frozen=True)
class MjcfFrameContract:
    obj_type: str
    obj_name: str
    body_name: str
    offset: np.ndarray


@dataclass
class DrakeCompatibleMjcf:
    model_file: str
    tempdir: TemporaryDirectory[str] | None = None

    def close(self) -> None:
        if self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None


def parse_mjcf_model_contract(scene_path: str | Path) -> DrakeMjcfModelContract:
    path = Path(scene_path)
    roots = _load_xml_roots(path)
    defaults = _collect_default_classes(roots)
    body_indices = _extract_body_indices(roots)
    joint_layouts = _extract_joint_layouts(roots, defaults)
    collision_geoms = _extract_geom_collision_fields(roots, defaults)
    joint_ranges_by_name = _extract_joint_ranges(roots, defaults)
    actuators = _extract_actuator_contract_fields(
        roots,
        defaults,
        joint_ranges_by_name,
    )
    frame_sensors, contact_sensors, joint_sensors = _extract_sensor_contract_fields(
        roots,
        _extract_named_frames(roots, defaults),
        body_indices,
        {actuator.joint_name: index for index, actuator in enumerate(actuators)},
    )
    return DrakeMjcfModelContract(
        name=roots[0][1].attrib.get("model", path.stem),
        body_indices=body_indices,
        joint_layouts=joint_layouts,
        collision_geoms=collision_geoms,
        actuators=actuators,
        num_bodies=max(body_indices.values(), default=0) + 1,
        frame_sensors=frame_sensors,
        contact_sensors=contact_sensors,
        joint_sensors=joint_sensors,
    )


def materialize_drake_compatible_mjcf(scene_path: str | Path) -> DrakeCompatibleMjcf:
    """Write a temporary MJCF with inherited defaults expanded for Drake parsing.

    MuJoCo resolves nested defaults and body ``childclass`` inheritance before
    building the physical model. Drake's MJCF parser does not currently match
    that behavior for all UniLab robot assets, so the batch runtime feeds Drake
    a copy with joint/geom/site defaults made explicit. Visual-only geoms are
    omitted from that copy because the batch runtime only needs physical
    collision geometry.
    """

    source_scene = Path(scene_path).expanduser().resolve()
    roots = _load_xml_roots(source_scene)
    defaults = _collect_default_classes(roots)
    tempdir = TemporaryDirectory(prefix="drakeuni_mjcf_")
    temp_root = Path(tempdir.name)
    source_root = source_scene.parent
    copied_root = temp_root / source_root.name
    copytree(source_root, copied_root, dirs_exist_ok=True)
    _copy_referenced_meshdirs(roots, source_root, copied_root)

    for xml_path in copied_root.rglob("*.xml"):
        _expand_mjcf_defaults_in_file(xml_path, defaults)

    return DrakeCompatibleMjcf(
        model_file=str(copied_root / source_scene.name),
        tempdir=tempdir,
    )


def _copy_referenced_meshdirs(
    roots: Sequence[tuple[Path, ET.Element]],
    source_root: Path,
    copied_root: Path,
) -> None:
    for xml_path, root in roots:
        compiler = root.find("./compiler")
        if compiler is None:
            continue
        meshdir = compiler.attrib.get("meshdir")
        if not meshdir:
            continue
        meshdir_path = Path(meshdir)
        if meshdir_path.is_absolute():
            continue
        source_meshdir = (xml_path.parent / meshdir_path).resolve()
        if not source_meshdir.exists():
            continue
        if xml_path.parent.resolve() == source_root.resolve():
            copied_xml_dir = copied_root
        else:
            try:
                copied_xml_dir = copied_root / xml_path.parent.resolve().relative_to(source_root.resolve())
            except ValueError:
                copied_xml_dir = copied_root
        target_meshdir = (copied_xml_dir / meshdir_path).resolve()
        copytree(source_meshdir, target_meshdir, dirs_exist_ok=True)


def read_keyframe_qpos(scene_path: str | Path, name: str) -> np.ndarray | None:
    try:
        roots = _load_xml_roots(Path(scene_path))
    except ET.ParseError:
        return None
    for _, root in roots:
        for key in root.findall(".//key"):
            if key.attrib.get("name") != name:
                continue
            values = _parse_vector(key.attrib.get("qpos"))
            if values is not None:
                return values
    return None


def sensor_frames_as_pool_inputs(model_contract: DrakeMjcfModelContract) -> tuple[list[int], np.ndarray]:
    body_indices = [sensor.body_index for sensor in model_contract.frame_sensors]
    offsets = [sensor.offset for sensor in model_contract.frame_sensors]
    return body_indices, np.asarray(offsets, dtype=np.float64).reshape((-1, 3))


def _load_xml_roots(scene_path: Path) -> list[tuple[Path, ET.Element]]:
    roots: list[tuple[Path, ET.Element]] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        root = ET.parse(resolved).getroot()
        roots.append((resolved, root))
        for include in root.findall(".//include"):
            include_file = include.attrib.get("file")
            if include_file:
                visit(resolved.parent / include_file)

    visit(scene_path)
    return roots


def _parse_vector(text: str | None, *, expected: int | None = None) -> np.ndarray | None:
    if text is None:
        return None
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if expected is not None and values.shape != (expected,):
        return None
    return values if values.size else None


def _required_pair(text: str | None, description: str) -> np.ndarray:
    values = _parse_vector(text, expected=2)
    if values is None:
        raise ValueError(f"Expected two values for {description}, got {text!r}")
    return values


def _optional_pair(text: str | None, fallback: tuple[float, float]) -> np.ndarray:
    values = _parse_vector(text, expected=2)
    if values is None:
        return np.asarray(fallback, dtype=np.float64)
    return values


def _first_three(values: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    count = min(3, int(values.size))
    if count:
        out[:count] = values[:count]
    return out


def _parse_params(text: str | None, fallback: tuple[float, ...]) -> np.ndarray:
    values = _parse_vector(text)
    if values is None:
        return np.asarray(fallback, dtype=np.float64)
    return values


def _unique(names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _build_actuator_contract(
    tag: str,
    attrs: dict[str, str],
    joint_ranges_by_name: dict[str, np.ndarray],
) -> MjcfActuatorContract:
    name = attrs.get("name", "<unnamed>")
    if tag in STATEFUL_OR_NONJOINT_ACTUATORS:
        raise ValueError(
            f"DrakeUni actuator {name!r} uses MJCF <{tag}>, which requires "
            "stateful or non-joint actuator semantics not represented in the compact state"
        )
    joint_name = _require_supported_joint_transmission(name, tag, attrs)
    joint_range = joint_ranges_by_name.get(joint_name)
    ctrl_range = _resolve_ctrl_range(name, tag, attrs, joint_name, joint_range)
    force_range = _optional_pair(attrs.get("forcerange"), (-np.inf, np.inf))
    gear = _resolve_scalar_gear(name, attrs)

    if tag == "position":
        timeconst = float(attrs.get("timeconst", 0.0))
        if timeconst > 0.0:
            raise ValueError(
                f"DrakeUni actuator {name!r} uses filtered position dynamics; "
                "activation state is not represented in the compact state"
            )
        stiffness = float(attrs.get("kp", 1.0))
        damping = float(attrs.get("kv", 0.5))
        gainprm = np.asarray([stiffness, 0.0, 0.0], dtype=np.float64)
        biasprm = np.asarray([0.0, -stiffness, -damping], dtype=np.float64)
        kind = ACTUATOR_KIND_POSITION
    elif tag == "velocity":
        stiffness = 0.0
        damping = float(attrs.get("kv", 1.0))
        gainprm = np.asarray([damping, 0.0, 0.0], dtype=np.float64)
        biasprm = np.asarray([0.0, 0.0, -damping], dtype=np.float64)
        kind = ACTUATOR_KIND_VELOCITY
    elif tag == "motor":
        stiffness = 0.0
        damping = 0.0
        gainprm = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        biasprm = np.zeros(3, dtype=np.float64)
        kind = ACTUATOR_KIND_MOTOR
    elif tag == "damper":
        if "ctrlrange" not in attrs:
            raise ValueError(f"DrakeUni damper actuator {name!r} requires ctrlrange")
        if np.any(ctrl_range < 0.0):
            raise ValueError(f"DrakeUni damper actuator {name!r} requires nonnegative ctrlrange")
        stiffness = 0.0
        damping = float(attrs.get("kv", 1.0))
        if damping < 0.0:
            raise ValueError(f"DrakeUni damper actuator {name!r} requires nonnegative kv")
        gainprm = np.asarray([0.0, 0.0, -damping], dtype=np.float64)
        biasprm = np.zeros(3, dtype=np.float64)
        kind = ACTUATOR_KIND_DAMPER
    elif tag == "general":
        kind, stiffness, damping, gainprm, biasprm = _resolve_general_actuator(name, attrs)
    else:  # pragma: no cover - guarded by SUPPORTED_ACTUATOR_TAGS.
        raise ValueError(f"DrakeUni does not support MJCF actuator <{tag}> {name!r}")

    return MjcfActuatorContract(
        name=name,
        tag=tag,
        kind=kind,
        joint_name=joint_name,
        ctrl_range=ctrl_range,
        force_range=force_range,
        joint_range=joint_range if joint_range is not None else ctrl_range,
        gear=gear,
        stiffness=stiffness,
        damping=damping,
        gainprm=gainprm,
        biasprm=biasprm,
    )


def _require_supported_joint_transmission(name: str, tag: str, attrs: dict[str, str]) -> str:
    unsupported = [
        field
        for field in ("jointinparent", "site", "refsite", "body", "tendon", "cranksite", "slidersite")
        if attrs.get(field)
    ]
    if unsupported:
        raise ValueError(
            f"DrakeUni actuator {name!r} uses MJCF <{tag}> transmission fields "
            f"{unsupported}; only single joint transmissions are supported"
        )
    joint_name = attrs.get("joint")
    if not joint_name:
        raise ValueError(f"DrakeUni actuator {name!r} must target a joint")
    return joint_name


def _resolve_ctrl_range(
    name: str,
    tag: str,
    attrs: dict[str, str],
    joint_name: str,
    joint_range: np.ndarray | None,
) -> np.ndarray:
    ctrl_range = _parse_vector(attrs.get("ctrlrange"), expected=2)
    if ctrl_range is not None:
        return ctrl_range
    inherit_range = float(attrs.get("inheritrange", 0.0))
    if inherit_range > 0.0 and joint_range is not None:
        midpoint = 0.5 * float(joint_range[0] + joint_range[1])
        half_width = 0.5 * inherit_range * float(joint_range[1] - joint_range[0])
        return np.asarray([midpoint - half_width, midpoint + half_width], dtype=np.float64)
    if tag == "position" and joint_range is not None:
        return joint_range
    if tag == "damper":
        raise ValueError(f"DrakeUni damper actuator {name!r} requires ctrlrange")
    return np.asarray([-np.inf, np.inf], dtype=np.float64)


def _resolve_scalar_gear(name: str, attrs: dict[str, str]) -> float:
    gear = _parse_vector(attrs.get("gear"))
    if gear is None:
        return 1.0
    if gear.size == 0:
        return 1.0
    if gear.size > 1 and np.any(np.abs(gear[1:]) > 0.0):
        raise ValueError(
            f"DrakeUni actuator {name!r} uses multi-axis gear {gear.tolist()}; "
            "only scalar joint gear is supported"
        )
    if gear[0] == 0.0:
        raise ValueError(f"DrakeUni actuator {name!r} uses zero gear")
    return float(gear[0])


def _resolve_general_actuator(
    name: str,
    attrs: dict[str, str],
) -> tuple[int, float, float, np.ndarray, np.ndarray]:
    dyntype = attrs.get("dyntype", "none")
    gaintype = attrs.get("gaintype", "fixed")
    biastype = attrs.get("biastype", "none")
    if dyntype != "none":
        raise ValueError(
            f"DrakeUni general actuator {name!r} uses dyntype={dyntype!r}; "
            "activation state is not represented in the compact state"
        )
    if gaintype not in {"fixed", "affine"}:
        raise ValueError(f"DrakeUni general actuator {name!r} does not support gaintype={gaintype!r}")
    if biastype not in {"none", "affine"}:
        raise ValueError(f"DrakeUni general actuator {name!r} does not support biastype={biastype!r}")
    gainprm = _parse_params(attrs.get("gainprm"), (1.0, 0.0, 0.0))
    biasprm = _parse_params(attrs.get("biasprm"), (0.0, 0.0, 0.0))
    return (
        ACTUATOR_KIND_GENERAL,
        0.0,
        0.0,
        _first_three(gainprm),
        _first_three(biasprm),
    )


def _expand_mjcf_defaults_in_file(
    path: Path,
    defaults: dict[str, dict[str, dict[str, str]]],
) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag == "mujoco" and not root.attrib.get("model"):
        root.attrib["model"] = path.stem
    generated_geom_index = 0

    def walk_body(body: ET.Element, inherited_class: str | None) -> None:
        nonlocal generated_geom_index
        element_class = body.attrib.get("childclass", inherited_class)
        for child in list(body):
            if child.tag in {"joint", "geom", "site"}:
                class_name = child.attrib.get("class", element_class)
                attrs = _merged_default_attrs(defaults, class_name, child.tag, child.attrib)
                if child.tag == "geom" and _is_noncolliding_geom(attrs):
                    body.remove(child)
                    continue
                if child.tag == "geom" and not attrs.get("name"):
                    attrs["name"] = f"drakeuni_{path.parent.name}_{path.stem}_geom_{generated_geom_index}"
                    generated_geom_index += 1
                child.attrib.clear()
                child.attrib.update(attrs)
            elif child.tag == "body":
                walk_body(child, element_class)

    for body in root.findall("./worldbody/body"):
        walk_body(body, None)
    _rewrite_mesh_assets_for_drake(root, path.parent)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def _is_noncolliding_geom(attrs: dict[str, str]) -> bool:
    return attrs.get("contype") == "0" and attrs.get("conaffinity") == "0"


def _rewrite_mesh_assets_for_drake(root: ET.Element, xml_dir: Path) -> None:
    compiler = root.find("./compiler")
    meshdir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
    mesh_root = meshdir if meshdir.is_absolute() else xml_dir / meshdir
    for mesh in root.findall(".//asset/mesh"):
        file_attr = mesh.attrib.get("file")
        if not file_attr:
            continue
        source_attr = Path(file_attr)
        source = source_attr if source_attr.is_absolute() else mesh_root / source_attr
        if source.suffix.lower() != ".stl":
            continue
        target = source.with_suffix(".obj")
        if not target.exists():
            _convert_stl_to_obj(source, target)
        mesh.attrib["file"] = str(source_attr.with_suffix(".obj"))


def _convert_stl_to_obj(source: Path, target: Path) -> None:
    data = source.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            offset = 84
            for _ in range(tri_count):
                values = struct.unpack_from("<12f", data, offset)
                vertices.extend((values[3:6], values[6:9], values[9:12]))
                offset += 50
        else:
            vertices = _parse_ascii_stl_vertices(data)
    else:
        vertices = _parse_ascii_stl_vertices(data)
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError(f"Cannot convert STL mesh {source} to OBJ")
    with target.open("w", encoding="utf-8") as handle:
        handle.write("# Converted from STL by DrakeUni materializer.\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face_start in range(1, len(vertices) + 1, 3):
            handle.write(f"f {face_start} {face_start + 1} {face_start + 2}\n")


def _parse_ascii_stl_vertices(data: bytes) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices


def _collect_default_classes(
    roots: Sequence[tuple[Path, ET.Element]],
) -> dict[str, dict[str, dict[str, str]]]:
    defaults: dict[str, dict[str, dict[str, str]]] = {}

    def walk_default(
        node: ET.Element,
        inherited: dict[str, dict[str, str]],
    ) -> None:
        current = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in node:
            if child.tag in {"joint", "geom", "site", *SUPPORTED_ACTUATOR_TAGS}:
                current.setdefault(child.tag, {}).update(child.attrib)

        class_name = node.attrib.get("class")
        if class_name is not None:
            defaults[class_name] = {tag: dict(attrs) for tag, attrs in current.items()}

        for child in node:
            if child.tag == "default":
                walk_default(child, current)

    for _, root in roots:
        for default_node in root.findall("./default"):
            walk_default(default_node, {})
    return defaults


def _merged_default_attrs(
    defaults: dict[str, dict[str, dict[str, str]]],
    class_name: str | None,
    tag: str,
    attrs: dict[str, str],
) -> dict[str, str]:
    merged = dict(defaults.get(class_name or "", {}).get(tag, {}))
    merged.update(attrs)
    return merged


def _extract_body_indices(roots: Sequence[tuple[Path, ET.Element]]) -> dict[str, int]:
    body_indices: dict[str, int] = {}
    next_index = 1

    def walk_body(body: ET.Element) -> None:
        nonlocal next_index
        name = body.attrib.get("name")
        if name:
            if name in body_indices:
                raise ValueError(f"Duplicate MJCF body name {name!r}")
            body_indices[name] = next_index
        next_index += 1
        for child in body.findall("./body"):
            walk_body(child)

    for _, root in roots:
        for body in root.findall("./worldbody/body"):
            walk_body(body)
    return body_indices


def _extract_actuator_contract_fields(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
    joint_ranges_by_name: dict[str, np.ndarray],
) -> tuple[MjcfActuatorContract, ...]:
    actuators: list[MjcfActuatorContract] = []

    for _, root in roots:
        for actuator_block in root.findall(".//actuator"):
            for actuator in actuator_block:
                tag = actuator.tag.strip().lower()
                if tag not in SUPPORTED_ACTUATOR_TAGS:
                    name = actuator.attrib.get("name", "<unnamed>")
                    raise ValueError(f"DrakeUni does not support MJCF actuator <{tag}> {name!r}")
                attrs = _merged_default_attrs(
                    defaults,
                    actuator.attrib.get("class"),
                    tag,
                    actuator.attrib,
                )
                actuators.append(
                    _build_actuator_contract(
                        tag,
                        attrs,
                        joint_ranges_by_name,
                    )
                )

    if not actuators:
        raise ValueError("DrakeUni batch runtime requires MJCF actuators")
    return tuple(actuators)


def _extract_joint_ranges(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
) -> dict[str, np.ndarray]:
    ranges: dict[str, np.ndarray] = {}

    def walk_body(body: ET.Element, inherited_class: str | None) -> None:
        element_class = body.attrib.get("childclass", inherited_class)
        for joint in body.findall("./joint"):
            name = joint.attrib.get("name")
            if not name:
                continue
            attrs = _merged_default_attrs(
                defaults,
                joint.attrib.get("class", element_class),
                "joint",
                joint.attrib,
            )
            joint_range = _parse_vector(attrs.get("range"), expected=2)
            if joint_range is not None:
                ranges[name] = joint_range
        for child in body.findall("./body"):
            walk_body(child, element_class)

    for _, root in roots:
        for body in root.findall("./worldbody/body"):
            walk_body(body, None)
    return ranges


def _extract_geom_collision_fields(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
) -> tuple[MjcfGeomCollisionContract, ...]:
    geoms: list[MjcfGeomCollisionContract] = []
    generated_geom_index = 0

    def append_geom(attrs: dict[str, str]) -> None:
        nonlocal generated_geom_index
        if _is_noncolliding_geom(attrs):
            return
        name = attrs.get("name")
        if not name:
            name = f"geom{generated_geom_index}"
            generated_geom_index += 1
        geoms.append(
            MjcfGeomCollisionContract(
                name=name,
                contype=int(attrs.get("contype", "1")),
                conaffinity=int(attrs.get("conaffinity", "1")),
            )
        )

    def walk_body(body: ET.Element, inherited_class: str | None) -> None:
        element_class = body.attrib.get("childclass", inherited_class)
        for geom in body.findall("./geom"):
            attrs = _merged_default_attrs(
                defaults,
                geom.attrib.get("class", element_class),
                "geom",
                geom.attrib,
            )
            append_geom(attrs)
        for child in body.findall("./body"):
            walk_body(child, element_class)

    for _, root in roots:
        for geom in root.findall("./worldbody/geom"):
            attrs = _merged_default_attrs(defaults, geom.attrib.get("class"), "geom", geom.attrib)
            append_geom(attrs)
        for body in root.findall("./worldbody/body"):
            walk_body(body, None)

    names = [geom.name for geom in geoms]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate MJCF collision geom names: {duplicates}")
    return tuple(geoms)


def _extract_joint_layouts(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
) -> tuple[MjcfJointLayout, ...]:
    layouts: list[MjcfJointLayout] = []
    qpos_adr = 0
    qvel_adr = 0

    def append_joint(
        *,
        name: str,
        body_name: str,
        joint_type: str,
    ) -> None:
        nonlocal qpos_adr, qvel_adr
        if joint_type == "free":
            kind = JOINT_KIND_FREE
            qpos_dim = 7
            qvel_dim = 6
        elif joint_type == "slide":
            kind = JOINT_KIND_SLIDE
            qpos_dim = 1
            qvel_dim = 1
        elif joint_type == "hinge":
            kind = JOINT_KIND_HINGE
            qpos_dim = 1
            qvel_dim = 1
        elif joint_type == "ball":
            kind = JOINT_KIND_BALL
            qpos_dim = 4
            qvel_dim = 3
        else:
            raise ValueError(f"DrakeUni does not support MJCF joint type {joint_type!r}")
        layouts.append(
            MjcfJointLayout(
                name=name,
                body_name=body_name,
                kind=kind,
                qpos_adr=qpos_adr,
                qvel_adr=qvel_adr,
                qpos_dim=qpos_dim,
                qvel_dim=qvel_dim,
            )
        )
        qpos_adr += qpos_dim
        qvel_adr += qvel_dim

    def walk_body(body: ET.Element, inherited_class: str | None) -> None:
        body_name = body.attrib.get("name", "")
        element_class = body.attrib.get("childclass", inherited_class)
        for child in body:
            if child.tag == "freejoint":
                if not body_name:
                    raise ValueError("DrakeUni requires named MJCF bodies for freejoint state layout")
                append_joint(
                    name=child.attrib.get("name", ""),
                    body_name=body_name,
                    joint_type="free",
                )
            elif child.tag == "joint":
                attrs = _merged_default_attrs(
                    defaults,
                    child.attrib.get("class", element_class),
                    "joint",
                    child.attrib,
                )
                joint_type = attrs.get("type", "hinge").strip().lower()
                if joint_type == "free" and not body_name:
                    raise ValueError("DrakeUni requires named MJCF bodies for free joint state layout")
                append_joint(
                    name=attrs.get("name", ""),
                    body_name=body_name,
                    joint_type=joint_type,
                )
        for child in body.findall("./body"):
            walk_body(child, element_class)

    for _, root in roots:
        for body in root.findall("./worldbody/body"):
            walk_body(body, None)
    return tuple(layouts)


def _extract_named_frames(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str], MjcfFrameContract]:
    frames: dict[tuple[str, str], MjcfFrameContract] = {}

    def walk_body(body: ET.Element, inherited_class: str | None) -> None:
        body_name = body.attrib.get("name")
        element_class = body.attrib.get("childclass", inherited_class)
        if body_name:
            for tag in ("site", "geom"):
                for element in body.findall(f"./{tag}"):
                    obj_name = element.attrib.get("name")
                    if not obj_name:
                        continue
                    attrs = _merged_default_attrs(
                        defaults,
                        element.attrib.get("class", element_class),
                        tag,
                        element.attrib,
                    )
                    offset = _parse_vector(attrs.get("pos"), expected=3)
                    if offset is None:
                        offset = np.zeros(3, dtype=np.float64)
                    frames[(tag, obj_name)] = MjcfFrameContract(
                        obj_type=tag,
                        obj_name=obj_name,
                        body_name=body_name,
                        offset=offset,
                    )
        for child in body.findall("./body"):
            walk_body(child, element_class)

    for _, root in roots:
        for body in root.findall("./worldbody/body"):
            walk_body(body, None)
    return frames


def _extract_sensor_contract_fields(
    roots: Sequence[tuple[Path, ET.Element]],
    frames: dict[tuple[str, str], MjcfFrameContract],
    body_indices: dict[str, int],
    actuator_indices_by_joint: dict[str, int],
) -> tuple[
    tuple[MjcfFrameSensorContract, ...],
    tuple[MjcfContactSensorContract, ...],
    tuple[MjcfJointSensorContract, ...],
]:
    frame_sensors: list[MjcfFrameSensorContract] = []
    frame_object_to_index: dict[str, int] = {}

    contact_sensors: list[MjcfContactSensorContract] = []
    joint_sensors: list[MjcfJointSensorContract] = []

    for _, root in reversed(roots):
        for sensor_block in root.findall(".//sensor"):
            for sensor in sensor_block:
                tag = sensor.tag.strip().lower()
                if tag not in SUPPORTED_SENSOR_TAGS:
                    name = sensor.attrib.get("name", "<unnamed>")
                    raise ValueError(f"DrakeUni does not support MJCF sensor <{tag}> {name!r}")
                if tag == "contact":
                    contact_sensors.append(
                        _build_contact_sensor_contract(sensor, frames, body_indices, frame_object_to_index)
                    )
                    continue
                if tag in JOINT_SENSOR_KIND_BY_TAG:
                    joint_sensors.append(
                        _build_joint_sensor_contract(sensor, tag, actuator_indices_by_joint)
                    )
                    continue

                frame_sensor = _build_frame_sensor_contract(sensor, tag, frames, body_indices)
                frame_object_to_index.setdefault(frame_sensor.obj_name, len(frame_sensors))
                frame_sensors.append(frame_sensor)

    _validate_unique_sensor_names((*frame_sensors, *contact_sensors, *joint_sensors))
    return tuple(frame_sensors), tuple(contact_sensors), tuple(joint_sensors)


def _build_frame_sensor_contract(
    sensor: ET.Element,
    tag: str,
    frames: dict[tuple[str, str], MjcfFrameContract],
    body_indices: dict[str, int],
) -> MjcfFrameSensorContract:
    name = sensor.attrib.get("name")
    if not name:
        raise ValueError(f"DrakeUni MJCF <{tag}> sensor requires a name")
    if tag == "framequat" and (sensor.attrib.get("reftype") or sensor.attrib.get("refname")):
        raise ValueError(
            f"DrakeUni MJCF <framequat> sensor {name!r} does not yet support reftype/refname"
        )
    if tag in {"gyro", "accelerometer", "velocimeter"}:
        obj_type = "site"
        obj_name = sensor.attrib.get("site")
        if not obj_name:
            raise ValueError(f"DrakeUni MJCF <{tag}> sensor {name!r} requires site=")
    else:
        obj_type = sensor.attrib.get("objtype")
        obj_name = sensor.attrib.get("objname")
        if not obj_type or not obj_name:
            raise ValueError(f"DrakeUni MJCF <{tag}> sensor {name!r} requires objtype/objname")

    frame = frames.get((obj_type, obj_name))
    if frame is None:
        raise ValueError(f"DrakeUni MJCF sensor {name!r} refers to unknown {obj_type} {obj_name!r}")
    body_index = body_indices.get(frame.body_name)
    if body_index is None:
        raise ValueError(f"Frame {obj_name!r} refers to unknown body {frame.body_name!r}")
    return MjcfFrameSensorContract(
        name=name,
        tag=tag,
        obj_name=obj_name,
        obj_type=obj_type,
        body_name=frame.body_name,
        body_index=body_index,
        offset=frame.offset,
    )


def _build_contact_sensor_contract(
    sensor: ET.Element,
    frames: dict[tuple[str, str], MjcfFrameContract],
    body_indices: dict[str, int],
    frame_object_to_index: dict[str, int],
) -> MjcfContactSensorContract:
    name = sensor.attrib.get("name")
    geom1 = sensor.attrib.get("geom1", "")
    geom2 = sensor.attrib.get("geom2", "")
    if not name:
        raise ValueError("DrakeUni MJCF <contact> sensor requires a name")
    data = sensor.attrib.get("data", "force").strip().lower()
    if data not in {"force", "found"}:
        raise ValueError(
            f"DrakeUni supports MJCF contact sensor data='force' or 'found', got {data!r}"
        )
    num = int(sensor.attrib.get("num", "1"))
    if num != 1:
        raise ValueError(f"DrakeUni supports MJCF contact sensor num=1, got {num} for {name!r}")
    reduce_value = sensor.attrib.get("reduce")
    body_frame = frames.get(("geom", geom2)) or frames.get(("geom", geom1))
    body_name = None if body_frame is None else body_frame.body_name
    body_index = None if body_name is None else body_indices.get(body_name)
    if body_name is not None and body_index is None:
        raise ValueError(f"Contact sensor {name!r} refers to unknown body {body_name!r}")
    frame_sensor_index = frame_object_to_index.get(geom2)
    if frame_sensor_index is None:
        frame_sensor_index = frame_object_to_index.get(geom1)
    return MjcfContactSensorContract(
        name=name,
        geom1=geom1,
        geom2=geom2,
        data=data,
        num=num,
        reduce=reduce_value,
        body_name=body_name,
        body_index=body_index,
        frame_sensor_index=frame_sensor_index,
    )


def _build_joint_sensor_contract(
    sensor: ET.Element,
    tag: str,
    actuator_indices_by_joint: dict[str, int],
) -> MjcfJointSensorContract:
    name = sensor.attrib.get("name")
    joint_name = sensor.attrib.get("joint")
    if not name:
        raise ValueError(f"DrakeUni MJCF <{tag}> sensor requires a name")
    if not joint_name:
        raise ValueError(f"DrakeUni MJCF <{tag}> sensor {name!r} requires joint=")
    actuator_index = actuator_indices_by_joint.get(joint_name)
    if actuator_index is None:
        raise ValueError(
            f"DrakeUni MJCF <{tag}> sensor {name!r} references joint {joint_name!r}, "
            "but joint sensors are currently supported only for actuated single-dof joints"
        )
    return MjcfJointSensorContract(
        name=name,
        tag=tag,
        joint_name=joint_name,
        actuator_index=int(actuator_index),
    )


def _validate_unique_sensor_names(
    sensors: Sequence[MjcfFrameSensorContract | MjcfContactSensorContract | MjcfJointSensorContract],
) -> None:
    seen: set[str] = set()
    for sensor in sensors:
        if sensor.name in seen:
            raise ValueError(f"Duplicate MJCF sensor name {sensor.name!r}")
        seen.add(sensor.name)
