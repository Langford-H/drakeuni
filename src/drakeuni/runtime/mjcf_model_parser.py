from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6

BASE_SENSOR_ORDER = (
    "gyro",
    "local_linvel",
    "global_linvel",
    "global_angvel",
    "position",
    "upvector",
)

BASE_SENSOR_NAMES = frozenset(
    {
        *BASE_SENSOR_ORDER,
        "accelerometer",
        "global_position",
        "orientation",
    }
)


@dataclass(frozen=True)
class MjcfTrackedPointContract:
    name: str
    obj_name: str
    obj_type: str
    body_name: str
    body_index: int
    offset: np.ndarray


@dataclass(frozen=True)
class MjcfContactSensorContract:
    name: str
    geom1: str
    geom2: str
    tracked_index: int | None


@dataclass(frozen=True)
class DrakeMjcfModelContract:
    name: str
    body_indices: dict[str, int]
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    joint_ranges: np.ndarray
    tracked_points: tuple[MjcfTrackedPointContract, ...]
    contact_sensors: tuple[MjcfContactSensorContract, ...]

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return _unique(
            (
                *BASE_SENSOR_ORDER,
                "base_pos",
                "base_quat",
                "dof_pos",
                "dof_vel",
                "feet_pos",
                "feet_contact_force",
                *(point.name for point in self.tracked_points),
                *(sensor.name for sensor in self.contact_sensors),
            )
        )

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


def parse_mjcf_model_contract(scene_path: str | Path) -> DrakeMjcfModelContract:
    path = Path(scene_path)
    roots = _load_xml_roots(path)
    defaults = _collect_default_classes(roots)
    body_indices = _extract_body_indices(roots)
    joint_ranges_by_name = _extract_joint_ranges(roots, defaults)
    ctrl_limits, torque_limits, joint_ranges = _extract_actuator_contract_fields(
        roots,
        defaults,
        joint_ranges_by_name,
    )
    tracked_points, contact_sensors = _extract_sensor_contract_fields(
        roots,
        _extract_named_frames(roots, defaults),
        body_indices,
    )
    return DrakeMjcfModelContract(
        name=roots[0][1].attrib.get("model", path.stem),
        body_indices=body_indices,
        ctrl_limits=ctrl_limits,
        torque_limits=torque_limits,
        joint_ranges=joint_ranges,
        tracked_points=tracked_points,
        contact_sensors=contact_sensors,
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


def tracked_points_as_pool_inputs(model_contract: DrakeMjcfModelContract) -> tuple[list[int], np.ndarray]:
    body_indices = [point.body_index for point in model_contract.tracked_points]
    offsets = [point.offset for point in model_contract.tracked_points]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl_limits: list[np.ndarray] = []
    torque_limits: list[float] = []
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
            joint_ranges.append(joint_range if joint_range is not None else ctrl_range)

    if not ctrl_limits:
        raise ValueError("DrakeUni batch runtime requires MJCF position actuators")
    return (
        np.asarray(ctrl_limits, dtype=np.float64),
        np.asarray(torque_limits, dtype=np.float64),
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
) -> tuple[tuple[MjcfTrackedPointContract, ...], tuple[MjcfContactSensorContract, ...]]:
    tracked_points: list[MjcfTrackedPointContract] = []
    tracked_object_to_index: dict[str, int] = {}

    for _, root in roots:
        for sensor in root.findall(".//sensor/framepos"):
            name = sensor.attrib.get("name")
            obj_type = sensor.attrib.get("objtype")
            obj_name = sensor.attrib.get("objname")
            if not name or not obj_type or not obj_name or name in BASE_SENSOR_NAMES:
                continue
            frame = frames.get((obj_type, obj_name))
            if frame is None:
                continue
            body_index = body_indices.get(frame.body_name)
            if body_index is None:
                raise ValueError(f"Frame {obj_name!r} refers to unknown body {frame.body_name!r}")
            tracked_object_to_index.setdefault(obj_name, len(tracked_points))
            tracked_points.append(
                MjcfTrackedPointContract(
                    name=name,
                    obj_name=obj_name,
                    obj_type=obj_type,
                    body_name=frame.body_name,
                    body_index=body_index,
                    offset=frame.offset,
                )
            )

    contact_sensors: list[MjcfContactSensorContract] = []
    for _, root in roots:
        for sensor in root.findall(".//sensor/contact"):
            name = sensor.attrib.get("name")
            geom1 = sensor.attrib.get("geom1", "")
            geom2 = sensor.attrib.get("geom2", "")
            if not name:
                continue
            tracked_index = tracked_object_to_index.get(geom2)
            if tracked_index is None:
                tracked_index = tracked_object_to_index.get(geom1)
            contact_sensors.append(
                MjcfContactSensorContract(
                    name=name,
                    geom1=geom1,
                    geom2=geom2,
                    tracked_index=tracked_index,
                )
            )

    return tuple(tracked_points), tuple(contact_sensors)
