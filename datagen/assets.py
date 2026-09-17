# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Build the FR3 robot model and local USD assets from the source meshes."""

import argparse
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import trimesh

GripperType = Literal["franka_hand", "umi"]


class GeomGroup(IntEnum):
    WORLD_VISUAL = 0
    TASK_VISUAL = 1
    ROBOT_VISUAL = 2
    COLLISION = 3
    DEBUG = 4


@dataclass(frozen=True)
class FrankaHandConfiguration:
    attachment_quaternion: tuple[float, float, float, float]
    gripper_site_position: tuple[float, float, float]
    finger_range: tuple[float, float]
    total_opening_speed: float
    close_start_delay: float
    open_start_delay: float
    actuator_stiffness: float
    actuator_force_limit: float
    joint_damping: float
    joint_armature: float
    stock_fingertip_position: tuple[float, float, float]


@dataclass(frozen=True)
class UmiGripperConfiguration:
    holder_mesh: Path
    finger_mesh: Path
    holder_positions: tuple[tuple[float, float, float], tuple[float, float, float]]
    holder_quaternions: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    finger_positions: tuple[tuple[float, float, float], tuple[float, float, float]]
    finger_quaternions: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    fingertip_positions: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    sdf_octree_depth: int
    sdf_distance_offset: float


def _add_umi_geometry(
    hand: mujoco.MjSpec,
    configuration: UmiGripperConfiguration,
) -> None:
    holder_mesh = hand.add_mesh(
        name="umi_holder",
        file=str(configuration.holder_mesh),
    )
    holder_mesh.octree_maxdepth = configuration.sdf_octree_depth
    finger_mesh = hand.add_mesh(
        name="umi_soft_finger",
        file=str(configuration.finger_mesh),
    )
    finger_mesh.octree_maxdepth = configuration.sdf_octree_depth
    attachments = zip(
        ("left", "right"),
        configuration.holder_positions,
        configuration.holder_quaternions,
        configuration.finger_positions,
        configuration.finger_quaternions,
        strict=True,
    )
    for (
        side,
        holder_position,
        holder_quaternion,
        finger_position,
        finger_quaternion,
    ) in attachments:
        body = hand.body(f"{side}_finger")
        for geom in tuple(body.geoms):
            hand.delete(geom)
        for part, mesh_name, position, quaternion, rgba in (
            (
                "holder",
                "umi_holder",
                holder_position,
                holder_quaternion,
                (0.9, 0.9, 0.9, 1.0),
            ),
            (
                "soft_finger",
                "umi_soft_finger",
                finger_position,
                finger_quaternion,
                (1.0, 0.68, 0.05, 1.0),
            ),
        ):
            body.add_geom(
                name=f"{side}_{part}_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=list(position),
                quat=list(quaternion),
                rgba=rgba,
                group=GeomGroup.ROBOT_VISUAL,
                contype=0,
                conaffinity=0,
                mass=0.0,
            )
            body.add_geom(
                name=f"{side}_{part}_sdf",
                type=mujoco.mjtGeom.mjGEOM_SDF,
                meshname=mesh_name,
                pos=list(position),
                quat=list(quaternion),
                rgba=[0.0, 0.0, 0.0, 0.0],
                group=GeomGroup.COLLISION,
                contype=2,
                conaffinity=3,
                condim=4,
                friction=[3.0, 1.0, 0.0002],
                margin=configuration.sdf_distance_offset,
                mass=0.0,
            )


def build_fr3_spec(
    gripper_type: GripperType,
    hand_configuration: FrankaHandConfiguration,
    umi_configuration: UmiGripperConfiguration,
) -> mujoco.MjSpec:
    menagerie = Path(__file__).resolve().parents[1] / "assets/robots/franka/menagerie"
    arm = mujoco.MjSpec.from_file(str(menagerie / "franka_fr3_v2/fr3v2.xml"))
    hand = mujoco.MjSpec.from_file(str(menagerie / "franka_emika_panda/hand.xml"))
    hand.option.integrator = arm.option.integrator
    hand_body = hand.worldbody.first_body()
    hand_body.quat = list(hand_configuration.attachment_quaternion)
    hand_body.add_site(
        name="gripper",
        pos=list(hand_configuration.gripper_site_position),
    )
    for joint_name in ("finger_joint1", "finger_joint2"):
        hand.joint(joint_name).range = list(hand_configuration.finger_range)
    if gripper_type == "umi":
        _add_umi_geometry(hand, umi_configuration)
    elif gripper_type != "franka_hand":
        raise ValueError("gripper_type must be 'franka_hand' or 'umi'")
    attachment = arm.body("fr3v2_link8").add_site(name="hand_attachment")
    arm.attach(hand, prefix="panda_", site=attachment)
    return arm


def robot_spec() -> mujoco.MjSpec:
    """Compose the source robot with the task's physical parameters."""
    robot_assets = Path(__file__).resolve().parents[1] / "assets/robots"
    hand = FrankaHandConfiguration(
        attachment_quaternion=(0.9238795, 0.0, 0.0, -0.3826834),
        gripper_site_position=(0.0, 0.0, 0.1),
        finger_range=(0.0, 0.03787644),
        total_opening_speed=0.05,
        close_start_delay=0.21,
        open_start_delay=0.28,
        actuator_stiffness=20000.0,
        actuator_force_limit=100.0,
        joint_damping=150.0,
        joint_armature=0.4,
        stock_fingertip_position=(0.0, 0.0, 0.0542),
    )
    gripper = UmiGripperConfiguration(
        holder_mesh=robot_assets / "umi/franka_finger_holder.stl",
        finger_mesh=robot_assets / "umi/soft_gripper_finger.stl",
        holder_positions=(
            (0.0, -0.00272269, -0.07091285),
            (0.0, -0.00272269, -0.07091285),
        ),
        holder_quaternions=(
            (0.70701032, -0.01167946, 0.01167946, 0.70701032),
            (0.70701032, -0.01167946, 0.01167946, 0.70701032),
        ),
        finger_positions=(
            (0.0129, -0.0015086, 0.01279217),
            (0.0129, -0.0015086, 0.01279217),
        ),
        finger_quaternions=(
            (0.00148973, 0.70710521, 0.70710521, -0.00148973),
            (0.00148973, 0.70710521, 0.70710521, -0.00148973),
        ),
        fingertip_positions=(
            (-0.00099231, -0.00005533, 0.1361496),
            (-0.00099231, -0.00005533, 0.1361496),
        ),
        sdf_octree_depth=7,
        sdf_distance_offset=0.0005,
    )
    spec = build_fr3_spec("umi", hand, gripper)
    spec.option.timestep = 0.006
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    spec.option.impratio = 10.0
    for geom in spec.geoms:
        if geom.group != 2:
            geom.contype = 2
            geom.conaffinity = 3
            geom.group = 3
    for index, friction in enumerate((0.517, 0.541, 0.777, 1.457, 0.737, 0.807, 0.699), 1):
        name = f"fr3v2_joint{index}"
        joint = spec.joint(name)
        joint.damping[0] = 0.0
        joint.frictionloss = friction
        actuator = spec.actuator(name)
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.gainprm[:] = 0.0
        actuator.gainprm[0] = 1.0
        actuator.biasprm[:] = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = False
    flange = spec.body("fr3v2_link8")
    flange.explicitinertial = True
    flange.mass = 0.273
    flange.ipos = (0.074, -0.043, 0.150)
    flange.inertia = (0.0008, 0.0008, 0.0005)
    actuator = spec.actuator("panda_actuator8")
    actuator.gainprm[0] = 20000.0 * hand.finger_range[1] / 255.0
    actuator.biasprm[:3] = (0.0, -20000.0, 0.0)
    actuator.forcerange = (-100.0, 100.0)
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        spec.joint(name).damping[0] = 150.0
        spec.joint(name).armature = 0.4
    return spec


def prepare_assets(output: Path, force: bool = False) -> Path:
    """Convert source geometry to a local cache, keeping concave colliders."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "READY"
    if ready.exists() and not force:
        robot_path = output / ready.read_text().strip()
        if robot_path.is_file():
            return output
    spec = robot_spec()
    robot_assets = Path(__file__).resolve().parents[1] / "assets/robots/franka/menagerie"
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            package = "franka_emika_panda" if mesh.name.startswith("panda_") else "franka_fr3_v2"
            mesh.file = str(robot_assets / package / "assets" / Path(mesh.file).name)
    # The importer accepts mesh geoms; restore the UMI SDF approximation in USD.
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_SDF:
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
    source = output / "robot.xml"
    source.write_text(spec.to_xml())
    converted = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(source),
            usd_dir=str(output),
            fix_base=True,
            self_collision=True,
            force_usd_conversion=True,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
    )
    stage = Usd.Stage.Open(converted.usd_path)
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    material = UsdShade.Material.Define(stage, str(stage.GetDefaultPrim().GetPath()) + "/UMIMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(3.0)
    physics.CreateDynamicFrictionAttr(3.0)
    physics.CreateRestitutionAttr(0.0)
    material.GetPrim().AddAppliedSchema("NewtonMaterialAPI")
    material.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(1.0)
    material.GetPrim().CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(0.0002)
    default_material = UsdShade.Material.Define(stage, str(stage.GetDefaultPrim().GetPath()) + "/RobotMaterial")
    default_physics = UsdPhysics.MaterialAPI.Apply(default_material.GetPrim())
    default_physics.CreateStaticFrictionAttr(1.0)
    default_physics.CreateDynamicFrictionAttr(1.0)
    default_physics.CreateRestitutionAttr(0.0)
    base = stage.GetPrimAtPath(str(stage.GetDefaultPrim().GetPath()) + "/Geometry/base")
    base.RemoveAPI(UsdPhysics.RigidBodyAPI)
    base.RemoveAPI(UsdPhysics.MassAPI)
    finger_joints = []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(prim)
            if base.GetPath() in joint.GetBody1Rel().GetTargets():
                prim.SetActive(False)
            elif base.GetPath() in joint.GetBody0Rel().GetTargets():
                joint.GetBody0Rel().SetTargets([])
        if prim.GetName().startswith("panda_finger_joint"):
            UsdPhysics.PrismaticJoint(prim).CreateUpperLimitAttr(0.03787644)
            finger_joints.append(prim)
    finger_joints.sort(key=lambda prim: prim.GetName())
    finger_joints[0].AddAppliedSchema("NewtonMimicAPI")
    finger_joints[0].CreateRelationship("newton:mimicJoint").SetTargets([finger_joints[1].GetPath()])
    finger_joints[0].CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
    finger_joints[0].CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(1.0)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(default_material, materialPurpose="physics")
        if prim.HasAPI(UsdPhysics.CollisionAPI) and "_sdf" in str(prim.GetPath()):
            prim.AddAppliedSchema("MjcGeomAPI")
            prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(4)
            prim.CreateAttribute("mjc:friction", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(3.0, 1.0, 0.0002))
            prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.0005)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
            prim.AddAppliedSchema("NewtonSDFCollisionAPI")
            prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(256)
            prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int).Set(256)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()
    assets = Path(__file__).resolve().parents[1] / "assets"
    objects = assets / "objects"
    entries = [
        ("plate", objects / "plate/saucer_180mm.stl", 0.001, 0.35, True),
        ("rack_wireframe", objects / "dish_rack/rack_wireframe.obj", 1.0, 0.0, False),
    ]
    entries += [
        (name, objects / "dish_rack" / f"{name}.obj", 1.0, 0.0, True)
        for name in ("rack_wire_frame", *(f"rack_wire_beam_{i:02d}" for i in range(10)))
    ]
    for name, path, scale, mass, collision in entries:
        geometry = trimesh.load_mesh(path, process=False)
        vertices = np.asarray(geometry.vertices, dtype=np.float64) * scale
        stage = Usd.Stage.CreateNew(str(output / f"{name}.usda"))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/Asset/mesh")
        mesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexIndicesAttr(np.asarray(geometry.faces).reshape(-1).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([(0.74, 0.74, 0.74) if mass else (0.09, 0.09, 0.10)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mesh.GetPrim().AddAppliedSchema("NewtonSDFCollisionAPI")
            mesh.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
            mesh.GetPrim().CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(256)
            mesh.GetPrim().CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int).Set(256)
            material = UsdShade.Material.Define(stage, "/Asset/material")
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(1.0)
            physics.CreateDynamicFrictionAttr(1.0)
            physics.CreateRestitutionAttr(0.0)
            material.GetPrim().AddAppliedSchema("NewtonMaterialAPI")
            material.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(0.01)
            material.GetPrim().CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(0.0002)
            mesh.GetPrim().AddAppliedSchema("MjcGeomAPI")
            mesh.GetPrim().CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(4 if mass else 3)
            mesh.GetPrim().CreateAttribute("mjc:solref", Sdf.ValueTypeNames.Double2).Set(
                Gf.Vec2d(0.008, 1.0) if mass else Gf.Vec2d(0.01, 2.0)
            )
            mesh.GetPrim().CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set(
                [0.95, 0.99, 0.001, 0.5, 2.0]
            )
            mesh.GetPrim().CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(0 if mass else 1)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose="physics")
        if mass:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            inertial_mesh = trimesh.Trimesh(vertices=vertices, faces=geometry.faces, process=True)
            inertial_mesh.density = mass / inertial_mesh.volume
            moments, rotation = np.linalg.eigh(inertial_mesh.moment_inertia)
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1.0
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            properties = UsdPhysics.MassAPI.Apply(root.GetPrim())
            properties.CreateMassAttr(mass)
            properties.CreateCenterOfMassAttr(Gf.Vec3f(*inertial_mesh.center_mass))
            properties.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
            properties.CreatePrincipalAxesAttr(Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*quaternion[1:])))
        stage.GetRootLayer().Save()
    ready.write_text(str(Path(converted.usd_path).relative_to(output)) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / ".cache")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    from isaaclab.utils.version import standalone_importers_available

    app = None
    if not standalone_importers_available():
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True).app
    try:
        print(prepare_assets(arguments.output, arguments.force))
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
