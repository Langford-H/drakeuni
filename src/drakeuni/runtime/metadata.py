from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6

BASE_SENSOR_NAMES = frozenset(
    {
        "gyro",
        "local_linvel",
        "global_linvel",
        "global_angvel",
        "position",
        "upvector",
    }
)

GO1_FOOT_SENSOR_NAMES = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
GO1_FOOT_CONTACT_SENSOR_NAMES = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)

# Drake body indices for UniLab's current Go1 Drake MJCF. This stays explicit
# until DrakeUni grows a compiled metadata query layer.
GO1_BODY_INDICES = {
    "trunk": 1,
    "FR_hip": 2,
    "FR_thigh": 3,
    "FR_calf": 4,
    "FL_hip": 5,
    "FL_thigh": 6,
    "FL_calf": 7,
    "RR_hip": 8,
    "RR_thigh": 9,
    "RR_calf": 10,
    "RL_hip": 11,
    "RL_thigh": 12,
    "RL_calf": 13,
}


@dataclass(frozen=True)
class DrakeModelMetadata:
    name: str
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    joint_ranges: np.ndarray
    foot_sensor_to_body: dict[str, str]
    foot_sensor_offsets: dict[str, np.ndarray]
    contact_sensors: frozenset[str]

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return (
            "gyro",
            "local_linvel",
            "global_linvel",
            "global_angvel",
            "position",
            "upvector",
            "base_pos",
            "base_quat",
            "dof_pos",
            "dof_vel",
            "feet_pos",
            "feet_contact_force",
            *GO1_FOOT_SENSOR_NAMES,
            *GO1_FOOT_CONTACT_SENSOR_NAMES,
        )


def body_index(name: str) -> int:
    try:
        return GO1_BODY_INDICES[name]
    except KeyError as exc:
        raise ValueError(f"DrakeUni Go1 runtime only knows body {name!r}") from exc


def load_model_metadata(scene_path: str | Path) -> DrakeModelMetadata:
    path = Path(scene_path)
    roots = _load_xml_roots(path)
    defaults = _collect_default_classes(roots)
    joint_ranges_by_name = _extract_joint_ranges(roots, defaults)
    ctrl_limits, torque_limits, joint_ranges = _extract_actuator_metadata(
        roots,
        defaults,
        joint_ranges_by_name,
    )
    foot_sensor_to_body, foot_sensor_offsets, contact_sensors = _extract_sensor_metadata(
        roots,
        _extract_sites(roots),
    )
    return DrakeModelMetadata(
        name=roots[0][1].attrib.get("model", path.stem),
        ctrl_limits=ctrl_limits,
        torque_limits=torque_limits,
        joint_ranges=joint_ranges,
        foot_sensor_to_body=foot_sensor_to_body,
        foot_sensor_offsets=foot_sensor_offsets,
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


def foot_metadata(metadata: DrakeModelMetadata) -> tuple[list[int], np.ndarray]:
    body_indices: list[int] = []
    offsets: list[np.ndarray] = []
    for sensor_name in GO1_FOOT_SENSOR_NAMES:
        body_name = metadata.foot_sensor_to_body.get(sensor_name)
        if body_name is None:
            raise ValueError(f"DrakeUni Go1 runtime missing foot sensor {sensor_name!r}")
        body_indices.append(body_index(body_name))
        offsets.append(metadata.foot_sensor_offsets[sensor_name])
    return body_indices, np.asarray(offsets, dtype=np.float64)


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
            if child.tag in {"joint", "position"}:
                current.setdefault(child.tag, {}).update(child.attrib)

        class_name = node.attrib.get("class")
        if class_name:
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


def _extract_actuator_metadata(
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
            ctrl_range = _required_pair(attrs.get("ctrlrange"), f"{actuator_name} ctrlrange")
            force_range = _required_pair(attrs.get("forcerange"), f"{actuator_name} forcerange")
            ctrl_limits.append(ctrl_range)
            torque_limits.append(float(np.max(np.abs(force_range))))

            joint_name = actuator.attrib.get("joint")
            if joint_name and joint_name in joint_ranges_by_name:
                joint_ranges.append(joint_ranges_by_name[joint_name])
            else:
                joint_ranges.append(ctrl_range)

    if not ctrl_limits:
        raise ValueError("DrakeUni Go1 runtime requires position actuators with ctrlrange metadata")
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
    for _, root in roots:
        for joint in root.findall(".//worldbody//joint"):
            name = joint.attrib.get("name")
            if not name:
                continue
            attrs = _merged_default_attrs(
                defaults,
                joint.attrib.get("class"),
                "joint",
                joint.attrib,
            )
            joint_range = _parse_vector(attrs.get("range"), expected=2)
            if joint_range is not None:
                ranges[name] = joint_range
    return ranges


def _extract_sites(
    roots: Sequence[tuple[Path, ET.Element]],
) -> dict[str, tuple[str, np.ndarray]]:
    sites: dict[str, tuple[str, np.ndarray]] = {}

    def walk_body(body: ET.Element) -> None:
        body_name = body.attrib.get("name")
        if body_name:
            for site in body.findall("./site"):
                site_name = site.attrib.get("name")
                site_pos = _parse_vector(site.attrib.get("pos"), expected=3)
                if site_name and site_pos is not None:
                    sites[site_name] = (body_name, site_pos)
        for child in body.findall("./body"):
            walk_body(child)

    for _, root in roots:
        for body in root.findall("./worldbody/body"):
            walk_body(body)
    return sites


def _extract_sensor_metadata(
    roots: Sequence[tuple[Path, ET.Element]],
    sites: dict[str, tuple[str, np.ndarray]],
) -> tuple[dict[str, str], dict[str, np.ndarray], frozenset[str]]:
    foot_sensor_to_body: dict[str, str] = {}
    foot_sensor_offsets: dict[str, np.ndarray] = {}
    contact_sensors: set[str] = set()

    for _, root in roots:
        for sensor in root.findall(".//sensor/framepos"):
            name = sensor.attrib.get("name")
            obj_name = sensor.attrib.get("objname")
            if (
                not name
                or not obj_name
                or name in BASE_SENSOR_NAMES
                or sensor.attrib.get("objtype") != "site"
                or obj_name not in sites
            ):
                continue
            body_name, site_offset = sites[obj_name]
            foot_sensor_to_body[name] = body_name
            foot_sensor_offsets[name] = site_offset

        for sensor in root.findall(".//sensor/contact"):
            name = sensor.attrib.get("name")
            if name:
                contact_sensors.add(name)

    return foot_sensor_to_body, foot_sensor_offsets, frozenset(contact_sensors)
