# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Scalar Franka plate-rack environment for Isaac Lab Newton and PhysX."""

import gc
import logging
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import warp as wp


class PlateRackRuntime:
    """One plate-rack scene, with identical action ordering for either physics backend."""

    def __init__(
        self,
        backend: Literal["newton", "physx"],
        device: str,
        max_episode_steps: int | None,
        render_mode: str | None,
        asset_directory: Path | None,
        rack_friction: float = 0.2,
    ) -> None:
        if backend not in ("newton", "physx"):
            raise ValueError("backend must be 'newton' or 'physx'")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human' or 'rgb_array'")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not np.isfinite(rack_friction) or rack_friction < 0.0:
            raise ValueError("rack_friction must be finite and nonnegative")
        self.rack_friction = rack_friction
        self.rack_frame_position = (0.5465, -0.2792, 0.001)
        self.backend = backend
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.control_dt = 0.066
        self.app = None
        self.sim = None
        self.task_language = "Place the plate in the dish rack."
        try:
            self._initialize(asset_directory)
        except BaseException:
            logging.getLogger(__name__).exception("Plate rack initialization failed")
            self.close()
            raise

    def _initialize(self, asset_directory: Path | None) -> None:
        if self.backend == "physx":
            from isaaclab.app import AppLauncher

            self.app = AppLauncher(headless=True).app
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
        )
        from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        from isaaclab import cloner
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import (
            Articulation,
            ArticulationCfg,
            RigidObject,
            RigidObjectCfg,
        )
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import (
            CuboidCfg,
            GroundPlaneCfg,
            SimulationCfg,
            SimulationContext,
            UsdFileCfg,
        )

        from .assets import prepare_assets
        from .controller import HybridImpedanceParameters, RobotController

        if SimulationContext.instance() is not None:
            raise RuntimeError("Use one Isaac Lab runtime per process")
        directory = Path(__file__).parent / ".cache" if asset_directory is None else asset_directory
        assets = prepare_assets(directory)
        physics = (
            PhysxCfg()
            if self.backend == "physx"
            else NewtonCfg(
                load_visual_shapes=self.render_mode is not None,
                default_shape_cfg=NewtonShapeCfg(gap=0.00025),
                solver_cfg=MJWarpSolverCfg(
                    integrator="implicitfast",
                    cone="elliptic",
                    impratio=10.0,
                    nconmax=512,
                    njmax=4096,
                    use_mujoco_contacts=False,
                ),
                collision_cfg=NewtonCollisionPipelineCfg(reduce_contacts=True),
                num_substeps=1,
                use_cuda_graph=True,
            )
        )
        visualizers = []
        if self.render_mode is not None:
            visualizers.append(
                NewtonGLVisualizerCfg(
                    headless=self.render_mode == "rgb_array",
                    window_width=1280,
                    window_height=720,
                    eye=(1.35, 1.2, 1.0),
                    lookat=(0.52, 0.02, 0.2),
                )
            )
        self.sim = SimulationContext(
            SimulationCfg(
                device=self.device,
                dt=0.006,
                render_interval=11,
                physics=physics,
                use_newton_actuators=False,
                visualizer_cfgs=visualizers,
            )
        )
        self.scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=2.5))
        names = tuple(f"fr3v2_joint{i}" for i in range(1, 8)) + (
            "panda_finger_joint1",
            "panda_finger_joint2",
        )
        home = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.03787644, 0.03787644)
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_[^/]+/Robot",
            spawn=UsdFileCfg(
                usd_path=str(assets / (assets / "READY").read_text().strip()),
                variants={"Physics": "physx" if self.backend == "physx" else "mujoco"},
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(names, home, strict=True))),
            joint_ordering=names,
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["fr3v2_joint[1-7]"],
                    stiffness=0.0,
                    damping=0.0,
                    joint_effort_limit=1e6,
                ),
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint[12]"],
                    stiffness=0.0,
                    damping=150.0,
                    armature=0.4,
                    joint_effort_limit=1e6,
                ),
            },
        )
        self.robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self.robot
        stage = self.sim.stage
        for prim in list(stage.Traverse()):
            # The controller supplies the tendon force to both finger joints.
            # Keep the mimic constraint, but disable the imported zero-target servo.
            if prim.GetName() in ("panda_actuator8", "panda_split"):
                prim.SetActive(False)
            if prim.GetName().startswith("panda_finger_joint"):
                prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double).Set(0.0)
        # Compose the source rack's +90-degree yaw with its wire mesh transform.
        rack_position = (0.55725, -0.28447, 0.006861)
        rack_rotation = tuple(np.asarray((0.0069289, 0.0069289, 0.999976, -0.999976)) / np.sqrt(2.0))
        for name in (
            "rack_wireframe",
            "rack_wire_frame",
            *(f"rack_wire_beam_{i:02d}" for i in range(10)),
        ):
            cfg = UsdFileCfg(usd_path=str(assets / f"{name}.usda"))
            cfg.func(
                f"/World/envs/env_0/{name}",
                cfg,
                translation=rack_position,
                orientation=rack_rotation,
            )
            if name != "rack_wireframe":
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_0/{name}")).MakeInvisible()
        self.plate = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_[^/]+/plate0",
                spawn=UsdFileCfg(usd_path=str(assets / "plate.usda")),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5846, 0.115, -0.002)),
            )
        )
        self.scene.rigid_objects["plate0"] = self.plate
        UsdGeom.Mesh(stage.GetPrimAtPath("/World/envs/env_0/plate0/mesh")).CreateDisplayColorAttr([Gf.Vec3f(0.74)])
        plane = GroundPlaneCfg()
        plane.func("/World/ground", plane, translation=(0.0, 0.0, -0.003))
        table = CuboidCfg(size=(0.900, 1.75555, 0.005))
        yaw = np.deg2rad(0.241662) / 2.0
        table.func(
            "/World/envs/env_0/table_visual",
            table,
            translation=(0.5528, -0.027256, -0.0055),
            orientation=(0.0, 0.0, float(np.sin(yaw)), float(np.cos(yaw))),
        )
        rack_paths = [
            p.GetPath()
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.CollisionAPI) and "/rack_wire_" in str(p.GetPath())
        ]
        rack_material_path = "/World/envs/env_0/RackContactMaterial"
        rack_material_cfg = PhysxRigidBodyMaterialCfg(
            static_friction=self.rack_friction,
            dynamic_friction=self.rack_friction,
            friction_combine_mode="min",
        )
        rack_material_cfg.func(rack_material_path, rack_material_cfg)
        rack_material = UsdShade.Material(stage.GetPrimAtPath(rack_material_path))
        for path in rack_paths:
            # MJWarp uses the rack's existing priority=1; PhysX uses the minimum.
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(rack_material, materialPurpose="physics")
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.AddAppliedSchema("PhysxCollisionAPI")
                prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0005)
                prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
            if "/Robot/" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(rack_paths)
            if self.backend == "newton" and prim.HasAPI("NewtonSDFCollisionAPI"):
                prim.RemoveAPI("NewtonMeshCollisionAPI")
                prim.GetAttribute("physics:approximation").Block()
        positions = cloner.grid_transforms(1, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(
            "/World/envs/env_0",
            "/World/envs/env_{}",
            1,
            positions,
            global_paths=("/World/ground",),
        )
        cloner.replicate(plan)
        if self.backend == "physx":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.sim.reset()
        self.robot.write_joint_position_to_sim_index(position=self.robot.data.default_joint_pos.torch)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.robot.data.default_joint_vel.torch)
        self.scene.update(0.006)
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.stream):
            self.controller = RobotController(self.device)
        self.elapsed = 0
        self.controller.reset(self.robot.data.joint_pos.torch)
        # Settle once; later resets restore this fixed physical state.
        with wp.ScopedStream(self.stream):
            for _ in range(100):
                self._substep()
        self.initial_joints = self.robot.data.joint_pos.torch.clone()
        self.initial_velocities = self.robot.data.joint_vel.torch.clone()
        self.initial_plate_pose = self.plate.data.root_pose_w.torch.clone()
        self.initial_plate_velocity = self.plate.data.root_vel_w.torch.clone()
        initial_states = (
            self.initial_joints,
            self.initial_velocities,
            self.initial_plate_pose,
            self.initial_plate_velocity,
        )
        if not all(bool(torch.isfinite(state).all()) for state in initial_states):
            raise RuntimeError("The scene produced a non-finite state during settling")
        limits = HybridImpedanceParameters()
        self.action_lower = torch.tensor((*limits.joint_lower, 0.0), device=self.device)
        self.action_upper = torch.tensor((*limits.joint_upper, 1.0), device=self.device)
        if self.render_mode is not None:
            self.sim.initialize_visualizers()

    def _substep(self) -> None:
        efforts = self.controller.compute(self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch)
        self.robot.set_joint_effort_target_index(target=efforts)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(0.006)

    def reset(self) -> torch.Tensor:
        self.robot.reset()
        self.robot.write_joint_position_to_sim_index(position=self.initial_joints)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.initial_velocities)
        self.plate.reset()
        self.plate.write_root_pose_to_sim_index(root_pose=self.initial_plate_pose)
        self.plate.write_root_velocity_to_sim_index(root_velocity=self.initial_plate_velocity)
        self.controller.reset(self.initial_joints)
        self.elapsed = 0
        self.scene.update(0.0)
        return self.state()

    def state(self) -> torch.Tensor:
        positions = self.robot.data.joint_pos.torch[0]
        return torch.cat((positions[:7], (positions[7:8] / 0.03787644).clamp(0.0, 1.0)))

    def task_state(self) -> tuple[bool, bool]:
        from isaaclab.utils.math import quat_apply

        plate = self.plate
        local_position = plate.data.root_pos_w.torch - self.scene.env_origins
        failed = local_position[:, 2] < -0.15
        axis = torch.zeros((1, 3), device=self.device)
        axis[:, 2] = 1.0
        normal = quat_apply(plate.data.root_quat_w.torch, axis)
        center = local_position + 0.01 * normal
        relative = center - center.new_tensor(self.rack_frame_position)
        # Inverse of the rack frame's +90-degree yaw.
        relative = torch.stack((relative[:, 1], -relative[:, 0], relative[:, 2]), dim=1)
        normal = torch.stack((normal[:, 1], -normal[:, 0], normal[:, 2]), dim=1)
        bottom = relative + 0.1027361 * torch.nn.functional.normalize(normal * normal[:, 2:3] - axis, dim=1)
        slot = -0.027792 - 0.0808
        success = (
            ((bottom[:, 0] - slot).abs() <= 0.023529 / 2.0)
            & (normal[:, 0].abs() >= np.sqrt(0.5))
            & (bottom[:, 1] >= -0.221)
            & (bottom[:, 1] <= 0.200)
            & (bottom[:, 2] >= 0.0)
            & (bottom[:, 2] <= 0.035)
        )
        return bool(success[0]), bool(failed[0])

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, bool, bool, bool]:
        if action.shape != (8,) or action.device != torch.device(self.device):
            raise ValueError(f"expected an action on {self.device} with shape (8,)")
        if not bool(
            torch.isfinite(action).all() & ((action >= self.action_lower) & (action <= self.action_upper)).all()
        ):
            raise ValueError("action must be finite and respect joint limits and normalized gripper bounds")
        with wp.ScopedStream(self.stream):
            self.controller.prepare(action[None])
            for _ in range(11):
                self._substep()
        self.elapsed += 1
        success, failure = self.task_state()
        truncated = self.max_episode_steps is not None and self.elapsed >= self.max_episode_steps
        if self.render_mode == "human":
            self.sim.render()
        return self.state(), success, failure, truncated

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        self.sim.render()
        if self.render_mode == "rgb_array":
            return self.sim.visualizers[0].render_rgb_array()
        return None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.stop()
            self.plate = None
            self.robot = None
            self.scene = None
            self.controller = None
            self.sim.clear_instance()
            self.sim = None
            torch.cuda.synchronize(self.device)
            gc.collect()
        if self.app is not None:
            # AppLauncher owns Kit's process lifetime and preserves failure status
            # in its atexit hook. Unloading extensions here leaves stale bindings.
            self.app = None


class PlateRackEnv(gym.Env):
    """One plate-rack scene with joint-angle [rad] and normalized gripper actions."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        backend: Literal["newton", "physx"] = "newton",
        device: str = "cuda:0",
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        asset_directory: str | Path | None = None,
        rack_friction: float = 0.2,
    ) -> None:
        from .controller import HybridImpedanceParameters

        self.render_mode = render_mode
        limits = HybridImpedanceParameters()
        self.action_space = gym.spaces.Box(
            np.asarray((*limits.joint_lower, 0.0), dtype=np.float32),
            np.asarray((*limits.joint_upper, 1.0), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {"observation.state": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)}
        )
        self.runtime = PlateRackRuntime(
            backend,
            device,
            max_episode_steps,
            render_mode,
            None if asset_directory is None else Path(asset_directory),
            rack_friction,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported; this task uses a fixed initial layout")
        state = self.runtime.reset()
        return {"observation.state": state.cpu().numpy()}, {
            "success": False,
            "task_language": self.runtime.task_language,
        }

    def step(self, action: np.ndarray | torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.runtime.device)
        state, success, failure, truncated = self.runtime.step(action)
        return (
            {"observation.state": state.cpu().numpy()},
            float(success),
            success or failure,
            truncated,
            {"success": success, "failure": failure, "task_language": self.runtime.task_language},
        )

    def render(self) -> np.ndarray | None:
        return self.runtime.render()

    def close(self) -> None:
        self.runtime.close()


def register() -> None:
    """Register the scalar single-plate task for both backends."""
    for backend in ("newton", "physx"):
        gym.register(
            id=f"franka_plate_rack_{backend}",
            entry_point="datagen:PlateRackEnv",
            kwargs={"backend": backend},
        )


register()
