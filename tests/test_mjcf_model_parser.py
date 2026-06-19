from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drakeuni.runtime.mjcf_model_parser import (
    ACTUATOR_KIND_DAMPER,
    ACTUATOR_KIND_GENERAL,
    ACTUATOR_KIND_MOTOR,
    ACTUATOR_KIND_POSITION,
    ACTUATOR_KIND_VELOCITY,
    SENSOR_KIND_CONTACT_FORCE,
    SENSOR_KIND_CONTACT_FOUND,
    materialize_drake_compatible_mjcf,
    parse_mjcf_model_contract,
    read_keyframe_qpos,
)

UNILAB_ROOT = Path("/Users/huanghaochen/solver/unilab/UniLab")


def _asset(path: str) -> Path:
    asset_path = UNILAB_ROOT / path
    if not asset_path.exists():
        pytest.skip(f"UniLab asset not available: {asset_path}")
    return asset_path


def _write_actuator_scene(tmp_path: Path, actuator_xml: str) -> Path:
    scene = tmp_path / "actuator_scene.xml"
    scene.write_text(
        f"""
<mujoco model="actuator_scene">
  <worldbody>
    <body name="body" pos="0 0 0">
      <joint name="hinge" type="hinge" range="-1 1"/>
      <geom name="geom" type="sphere" size="0.05" mass="1"/>
      <site name="site" pos="0 0 0"/>
    </body>
  </worldbody>
  <actuator>
    {actuator_xml}
  </actuator>
</mujoco>
""".strip()
    )
    return scene


def test_go1_drake_scene_mjcf_model_parser_discovers_contract() -> None:
    scene = _asset("src/unilab/assets/robots/go1/scene_flat_drake.xml")
    model_contract = parse_mjcf_model_contract(scene)

    assert model_contract.body_index("trunk") == 1
    assert model_contract.body_index("FR_calf") == 4
    assert model_contract.ctrl_limits.shape == (12, 2)
    assert model_contract.torque_limits.shape == (12,)
    assert model_contract.joint_ranges.shape == (12, 2)
    assert read_keyframe_qpos(scene, "home").shape == (19,)

    frame_position_names = [point.name for point in model_contract.frame_position_sensors]
    assert frame_position_names == ["position", "FR_pos", "FL_pos", "RR_pos", "RL_pos"]
    body_indices = [point.body_index for point in model_contract.frame_position_sensors]
    offsets = np.asarray([point.offset for point in model_contract.frame_position_sensors])
    assert body_indices == [
        model_contract.body_index("trunk"),
        model_contract.body_index("FR_calf"),
        model_contract.body_index("FL_calf"),
        model_contract.body_index("RR_calf"),
        model_contract.body_index("RL_calf"),
    ]
    np.testing.assert_allclose(
        offsets,
        np.vstack(([0.0, 0.0, 0.0], np.tile([0.0, 0.0, -0.213], (4, 1)))),
    )

    contact_names = [sensor.name for sensor in model_contract.contact_sensors]
    assert contact_names == [
        "FL_foot_contact",
        "FR_foot_contact",
        "RL_foot_contact",
        "RR_foot_contact",
    ]
    contacts = {sensor.name: sensor for sensor in model_contract.contact_sensors}
    assert contacts["FL_foot_contact"].data == "force"
    assert contacts["FL_foot_contact"].dim == 3
    assert contacts["FL_foot_contact"].kind == SENSOR_KIND_CONTACT_FORCE
    sensor_dims = dict(zip(model_contract.sensor_names, model_contract.sensor_dim, strict=True))
    assert sensor_dims["FL_foot_contact"] == 3
    assert "global_position" not in model_contract.sensor_names
    assert model_contract.sensor_names[:6] == (
        "gyro",
        "local_linvel",
        "position",
        "upvector",
        "global_linvel",
        "global_angvel",
    )


def test_go2_scene_mjcf_model_parser_uses_geom_frames_and_joint_range_ctrl_fallback() -> None:
    scene = _asset("src/unilab/assets/robots/go2/scene_flat.xml")
    model_contract = parse_mjcf_model_contract(scene)

    assert model_contract.body_index("base") == 1
    assert model_contract.ctrl_limits.shape == (12, 2)
    assert model_contract.torque_limits.shape == (12,)
    np.testing.assert_allclose(model_contract.ctrl_limits[0], [-1.0472, 1.0472])
    np.testing.assert_allclose(model_contract.torque_limits[[0, 2]], [23.7, 45.43])
    assert read_keyframe_qpos(scene, "home").shape == (19,)

    frame_position_names = [point.name for point in model_contract.frame_position_sensors]
    assert frame_position_names == ["global_position", "FR_pos", "FL_pos", "RR_pos", "RL_pos"]
    foot_points = model_contract.frame_position_sensors[1:]
    assert [point.obj_type for point in foot_points] == ["geom"] * 4
    assert [point.body_name for point in foot_points] == [
        "FR_calf",
        "FL_calf",
        "RR_calf",
        "RL_calf",
    ]
    np.testing.assert_allclose(
        np.asarray([point.offset for point in foot_points]),
        np.tile([-0.002, 0.0, -0.213], (4, 1)),
    )

    contacts = {sensor.name: sensor for sensor in model_contract.contact_sensors}
    assert (
        model_contract.frame_sensors[contacts["FR_foot_contact"].frame_sensor_index].name
        == "FR_pos"
    )
    assert (
        model_contract.frame_sensors[contacts["FL_foot_contact"].frame_sensor_index].name
        == "FL_pos"
    )
    assert contacts["FL_foot_contact"].data == "found"
    assert contacts["FL_foot_contact"].num == 1
    assert contacts["FL_foot_contact"].dim == 1
    assert contacts["FL_foot_contact"].kind == SENSOR_KIND_CONTACT_FOUND
    assert contacts["FL_foot_contact"].body_name == "FL_calf"
    assert contacts["base1_contact"].frame_sensor_index is None
    assert contacts["base1_contact"].data == "found"
    assert contacts["base1_contact"].body_name == "base"
    assert "base1_contact" in model_contract.sensor_names
    sensor_dims = dict(zip(model_contract.sensor_names, model_contract.sensor_dim, strict=True))
    assert sensor_dims["FL_foot_contact"] == 1
    assert "global_position" in model_contract.sensor_names
    sensor_kinds = dict(zip(model_contract.sensor_names, model_contract.sensor_type, strict=True))
    assert sensor_kinds["FL_foot_contact"] == SENSOR_KIND_CONTACT_FOUND


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


@pytest.mark.parametrize(
    ("actuator_xml", "expected_kind"),
    [
        (
            '<position name="pos" joint="hinge" forcerange="-5 5" kp="10" kv="1"/>',
            ACTUATOR_KIND_POSITION,
        ),
        (
            '<velocity name="vel" joint="hinge" ctrlrange="-2 2" forcerange="-5 5" kv="3"/>',
            ACTUATOR_KIND_VELOCITY,
        ),
        (
            '<motor name="motor" joint="hinge" ctrlrange="-2 2" forcerange="-4 4"/>',
            ACTUATOR_KIND_MOTOR,
        ),
        (
            '<damper name="damper" joint="hinge" ctrlrange="0 1" forcerange="-4 4" kv="2"/>',
            ACTUATOR_KIND_DAMPER,
        ),
        (
            '<general name="gen" joint="hinge" ctrlrange="-1 1" forcerange="-5 5" '
            'gainprm="2 0.5 0.25" biasprm="1 -2 -3"/>',
            ACTUATOR_KIND_GENERAL,
        ),
    ],
)
def test_mjcf_actuator_parser_records_supported_joint_actuator_kinds(
    tmp_path: Path,
    actuator_xml: str,
    expected_kind: int,
) -> None:
    model_contract = parse_mjcf_model_contract(_write_actuator_scene(tmp_path, actuator_xml))

    assert model_contract.actuator_kind.tolist() == [expected_kind]
    assert model_contract.ctrl_limits.shape == (1, 2)
    assert model_contract.torque_limits.shape == (1,)
    assert model_contract.actuator_gear.shape == (1,)
    assert model_contract.actuator_gainprm.shape == (1, 3)
    assert model_contract.actuator_biasprm.shape == (1, 3)


def test_mjcf_actuator_parser_inherits_unnamed_position_defaults(tmp_path: Path) -> None:
    scene = tmp_path / "default_position_scene.xml"
    scene.write_text(
        """
<mujoco model="default_position_scene">
  <default>
    <position kp="200000" kv="4000" inheritrange="1"/>
  </default>
  <worldbody>
    <body name="body">
      <joint name="slide" type="slide" range="0 0.7"/>
      <geom name="geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="slide"/>
  </actuator>
</mujoco>
""".strip()
    )

    model_contract = parse_mjcf_model_contract(scene)

    np.testing.assert_allclose(model_contract.ctrl_limits[0], [0.0, 0.7])
    np.testing.assert_allclose(model_contract.actuator_stiffness, [200000.0])
    np.testing.assert_allclose(model_contract.actuator_damping, [4000.0])


@pytest.mark.parametrize(
    "actuator_xml",
    [
        '<intvelocity name="bad" joint="hinge" ctrlrange="-1 1"/>',
        '<cylinder name="bad" joint="hinge" ctrlrange="-1 1"/>',
        '<muscle name="bad" joint="hinge" ctrlrange="-1 1"/>',
        '<adhesion name="bad" body="body" ctrlrange="-1 1"/>',
        '<dcmotor name="bad" joint="hinge" ctrlrange="-1 1"/>',
        '<plugin name="bad" joint="hinge" ctrlrange="-1 1"/>',
        '<motor name="bad" site="site" ctrlrange="-1 1"/>',
    ],
)
def test_mjcf_actuator_parser_rejects_unsupported_actuator_semantics(
    tmp_path: Path,
    actuator_xml: str,
) -> None:
    with pytest.raises(ValueError, match="DrakeUni actuator"):
        parse_mjcf_model_contract(_write_actuator_scene(tmp_path, actuator_xml))
