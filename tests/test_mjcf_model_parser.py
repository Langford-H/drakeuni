from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drakeuni.runtime.mjcf_model_parser import (
    materialize_drake_compatible_mjcf,
    parse_mjcf_model_contract,
    read_keyframe_qpos,
    tracked_points_as_pool_inputs,
)

UNILAB_ROOT = Path("/Users/huanghaochen/solver/unilab/UniLab")


def _asset(path: str) -> Path:
    asset_path = UNILAB_ROOT / path
    if not asset_path.exists():
        pytest.skip(f"UniLab asset not available: {asset_path}")
    return asset_path


def test_go1_drake_scene_mjcf_model_parser_discovers_contract() -> None:
    scene = _asset("src/unilab/assets/robots/go1/scene_flat_drake.xml")
    model_contract = parse_mjcf_model_contract(scene)

    assert model_contract.body_index("trunk") == 1
    assert model_contract.body_index("FR_calf") == 4
    assert model_contract.ctrl_limits.shape == (12, 2)
    assert model_contract.torque_limits.shape == (12,)
    assert model_contract.joint_ranges.shape == (12, 2)
    assert read_keyframe_qpos(scene, "home").shape == (19,)

    tracked_names = [point.name for point in model_contract.tracked_points]
    assert tracked_names == ["FR_pos", "FL_pos", "RR_pos", "RL_pos"]
    body_indices, offsets = tracked_points_as_pool_inputs(model_contract)
    assert body_indices == [
        model_contract.body_index("FR_calf"),
        model_contract.body_index("FL_calf"),
        model_contract.body_index("RR_calf"),
        model_contract.body_index("RL_calf"),
    ]
    np.testing.assert_allclose(offsets, np.tile([0.0, 0.0, -0.213], (4, 1)))

    contact_names = [sensor.name for sensor in model_contract.contact_sensors]
    assert contact_names == [
        "FL_foot_contact",
        "FR_foot_contact",
        "RL_foot_contact",
        "RR_foot_contact",
    ]
    assert "global_position" not in model_contract.sensor_names


def test_go2_scene_mjcf_model_parser_uses_geom_frames_and_joint_range_ctrl_fallback() -> None:
    scene = _asset("src/unilab/assets/robots/go2/scene_flat.xml")
    model_contract = parse_mjcf_model_contract(scene)

    assert model_contract.body_index("base") == 1
    assert model_contract.ctrl_limits.shape == (12, 2)
    assert model_contract.torque_limits.shape == (12,)
    np.testing.assert_allclose(model_contract.ctrl_limits[0], [-1.0472, 1.0472])
    np.testing.assert_allclose(model_contract.torque_limits[[0, 2]], [23.7, 45.43])
    assert read_keyframe_qpos(scene, "home").shape == (19,)

    tracked_names = [point.name for point in model_contract.tracked_points]
    assert tracked_names == ["FR_pos", "FL_pos", "RR_pos", "RL_pos"]
    assert [point.obj_type for point in model_contract.tracked_points] == ["geom"] * 4
    assert [point.body_name for point in model_contract.tracked_points] == [
        "FR_calf",
        "FL_calf",
        "RR_calf",
        "RL_calf",
    ]
    np.testing.assert_allclose(
        np.asarray([point.offset for point in model_contract.tracked_points]),
        np.tile([-0.002, 0.0, -0.213], (4, 1)),
    )

    contacts = {sensor.name: sensor for sensor in model_contract.contact_sensors}
    assert contacts["FR_foot_contact"].tracked_index == 0
    assert contacts["FL_foot_contact"].tracked_index == 1
    assert contacts["base1_contact"].tracked_index is None
    assert "base1_contact" in model_contract.sensor_names


def test_go2_drake_compatible_mjcf_expands_physics_defaults() -> None:
    scene = _asset("src/unilab/assets/robots/go2/scene_flat.xml")
    materialized = materialize_drake_compatible_mjcf(scene)
    try:
        import xml.etree.ElementTree as ET

        go2_xml = Path(materialized.model_file).parent / "go2.xml"
        root = ET.parse(go2_xml).getroot()

        visual_only_geoms = [
            geom
            for geom in root.findall(".//worldbody//geom")
            if geom.attrib.get("contype") == "0" and geom.attrib.get("conaffinity") == "0"
        ]
        assert visual_only_geoms == []

        joints = {joint.attrib["name"]: joint.attrib for joint in root.findall(".//worldbody//joint")}
        assert joints["FL_hip_joint"]["axis"] == "1 0 0"
        assert joints["FL_thigh_joint"]["axis"] == "0 1 0"
        assert joints["FL_calf_joint"]["axis"] == "0 1 0"

        foot = next(
            geom for geom in root.findall(".//worldbody//geom") if geom.attrib.get("name") == "FL"
        )
        assert foot.attrib["size"] == "0.022"
        assert foot.attrib["pos"] == "-0.002 0 -0.213"
        assert foot.attrib["contype"] == "1"
        assert foot.attrib["conaffinity"] == "2"
    finally:
        materialized.close()
