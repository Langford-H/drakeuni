from __future__ import annotations

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

FRAME_SENSOR_KIND_BY_TAG = {
    "gyro": SENSOR_KIND_GYRO,
    "accelerometer": SENSOR_KIND_ACCELEROMETER,
    "velocimeter": SENSOR_KIND_VELOCIMETER,
    "framepos": SENSOR_KIND_FRAME_POS,
    "framelinvel": SENSOR_KIND_FRAME_LINVEL,
    "frameangvel": SENSOR_KIND_FRAME_ANGVEL,
    "framezaxis": SENSOR_KIND_FRAME_ZAXIS,
}

SUPPORTED_SENSOR_TAGS = frozenset({*FRAME_SENSOR_KIND_BY_TAG, "contact"})


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
class DrakeMjcfModelContract:
    name: str
    body_indices: dict[str, int]
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    actuator_stiffness: np.ndarray
    actuator_damping: np.ndarray
    joint_ranges: np.ndarray
    num_bodies: int
    frame_sensors: tuple[MjcfFrameSensorContract, ...]
    contact_sensors: tuple[MjcfContactSensorContract, ...]

    @property
    def frame_position_sensors(self) -> tuple[MjcfFrameSensorContract, ...]:
        return tuple(sensor for sensor in self.frame_sensors if sensor.tag == "framepos")

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return _unique((*[sensor.name for sensor in self.frame_sensors], *[sensor.name for sensor in self.contact_sensors]))

    @property
    def sensor_dim(self) -> np.ndarray:
        by_name = {sensor.name: sensor for sensor in (*self.frame_sensors, *self.contact_sensors)}
        dims = [by_name[name].dim for name in self.sensor_names]
        return np.asarray(dims, dtype=np.int32)

    @property
    def sensor_adr(self) -> np.ndarray:
        dims = self.sensor_dim
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
        kinds: list[int] = []
        for name in self.sensor_names:
            if name in frame_kind:
                kinds.append(frame_kind[name])
            elif name in contact_kind:
                kinds.append(contact_kind[name])
            else:  # pragma: no cover - guarded by parser construction.
                raise ValueError(f"Unknown DrakeUni MJCF sensor {name!r}")
        return np.asarray(kinds, dtype=np.int32)

    @property
    def sensor_index(self) -> np.ndarray:
        frame_index = {sensor.name: i for i, sensor in enumerate(self.frame_sensors)}
        contact_index = {sensor.name: sensor.body_index for sensor in self.contact_sensors}
        indices: list[int] = []
        for name in self.sensor_names:
            if name in frame_index:
                indices.append(frame_index[name])
            elif name in contact_index:
                value = contact_index[name]
                indices.append(-1 if value is None else int(value))
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
    joint_ranges_by_name = _extract_joint_ranges(roots, defaults)
    (
        ctrl_limits,
        torque_limits,
        actuator_stiffness,
        actuator_damping,
        joint_ranges,
    ) = _extract_actuator_contract_fields(
        roots,
        defaults,
        joint_ranges_by_name,
    )
    frame_sensors, contact_sensors = _extract_sensor_contract_fields(
        roots,
        _extract_named_frames(roots, defaults),
        body_indices,
    )
    return DrakeMjcfModelContract(
        name=roots[0][1].attrib.get("model", path.stem),
        body_indices=body_indices,
        ctrl_limits=ctrl_limits,
        torque_limits=torque_limits,
        actuator_stiffness=actuator_stiffness,
        actuator_damping=actuator_damping,
        joint_ranges=joint_ranges,
        num_bodies=max(body_indices.values(), default=0) + 1,
        frame_sensors=frame_sensors,
        contact_sensors=contact_sensors,
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

    for xml_path in copied_root.rglob("*.xml"):
        _expand_mjcf_defaults_in_file(xml_path, defaults)

    return DrakeCompatibleMjcf(
        model_file=str(copied_root / source_scene.name),
        tempdir=tempdir,
    )


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


def _unique(names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _expand_mjcf_defaults_in_file(
    path: Path,
    defaults: dict[str, dict[str, dict[str, str]]],
) -> None:
    tree = ET.parse(path)
    root = tree.getroot()

    def walk_body(body: ET.Element, inherited_class: str | None) -> None:
        element_class = body.attrib.get("childclass", inherited_class)
        for child in list(body):
            if child.tag in {"joint", "geom", "site"}:
                class_name = child.attrib.get("class", element_class)
                attrs = _merged_default_attrs(defaults, class_name, child.tag, child.attrib)
                if child.tag == "geom" and _is_noncolliding_geom(attrs):
                    body.remove(child)
                    continue
                child.attrib.clear()
                child.attrib.update(attrs)
            elif child.tag == "body":
                walk_body(child, element_class)

    for body in root.findall("./worldbody/body"):
        walk_body(body, None)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def _is_noncolliding_geom(attrs: dict[str, str]) -> bool:
    return attrs.get("contype") == "0" and attrs.get("conaffinity") == "0"


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
            if child.tag in {"joint", "position", "geom", "site"}:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctrl_limits: list[np.ndarray] = []
    torque_limits: list[float] = []
    actuator_stiffness: list[float] = []
    actuator_damping: list[float] = []
    joint_ranges: list[np.ndarray] = []

    for _, root in roots:
        for actuator in root.findall(".//actuator/position"):
            attrs = _merged_default_attrs(
                defaults,
                actuator.attrib.get("class"),
                "position",
                actuator.attrib,
            )
            actuator_name = actuator.attrib.get("name", "<unnamed>")
            joint_name = actuator.attrib.get("joint")
            joint_range = joint_ranges_by_name.get(joint_name or "")
            ctrl_range = _parse_vector(attrs.get("ctrlrange"), expected=2)
            if ctrl_range is None:
                ctrl_range = joint_range
            if ctrl_range is None:
                raise ValueError(
                    f"Expected {actuator_name} ctrlrange or a range on joint {joint_name!r}"
                )
            force_range = _required_pair(attrs.get("forcerange"), f"{actuator_name} forcerange")
            ctrl_limits.append(ctrl_range)
            torque_limits.append(float(np.max(np.abs(force_range))))
            actuator_stiffness.append(float(attrs.get("kp", 1.0)))
            actuator_damping.append(float(attrs.get("kv", 0.5)))
            joint_ranges.append(joint_range if joint_range is not None else ctrl_range)

    if not ctrl_limits:
        raise ValueError("DrakeUni batch runtime requires MJCF position actuators")
    return (
        np.asarray(ctrl_limits, dtype=np.float64),
        np.asarray(torque_limits, dtype=np.float64),
        np.asarray(actuator_stiffness, dtype=np.float64),
        np.asarray(actuator_damping, dtype=np.float64),
        np.asarray(joint_ranges, dtype=np.float64),
    )


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
) -> tuple[tuple[MjcfFrameSensorContract, ...], tuple[MjcfContactSensorContract, ...]]:
    frame_sensors: list[MjcfFrameSensorContract] = []
    frame_object_to_index: dict[str, int] = {}

    contact_sensors: list[MjcfContactSensorContract] = []

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

                frame_sensor = _build_frame_sensor_contract(sensor, tag, frames, body_indices)
                frame_object_to_index.setdefault(frame_sensor.obj_name, len(frame_sensors))
                frame_sensors.append(frame_sensor)

    _validate_unique_sensor_names((*frame_sensors, *contact_sensors))
    return tuple(frame_sensors), tuple(contact_sensors)


def _build_frame_sensor_contract(
    sensor: ET.Element,
    tag: str,
    frames: dict[tuple[str, str], MjcfFrameContract],
    body_indices: dict[str, int],
) -> MjcfFrameSensorContract:
    name = sensor.attrib.get("name")
    if not name:
        raise ValueError(f"DrakeUni MJCF <{tag}> sensor requires a name")
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


def _validate_unique_sensor_names(
    sensors: Sequence[MjcfFrameSensorContract | MjcfContactSensorContract],
) -> None:
    seen: set[str] = set()
    for sensor in sensors:
        if sensor.name in seen:
            raise ValueError(f"Duplicate MJCF sensor name {sensor.name!r}")
        seen.add(sensor.name)
