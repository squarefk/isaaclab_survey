# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan, run, render, or benchmark the single-plate rack task on either backend."""

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch


@dataclass(frozen=True)
class JointWaypoint:
    """A joint target [rad], normalized opening, and speed limit [rad/s]."""

    name: str
    joints: np.ndarray
    opening: float
    speed: float
    stop: bool


def plan_waypoints(
    model: mujoco.MjModel,
    plate_position: np.ndarray,
    initial_joints: np.ndarray,
) -> tuple[JointWaypoint, ...]:
    """Solve the reference fingertip poses, with vertical descent segments no longer than 5 mm.

    Args:
        model: Robot model used only for forward kinematics and IK.
        plate_position: Initial plate position [m], shape [3].
        initial_joints: Initial arm joint angles [rad], shape [7].

    Returns:
        Joint waypoints without modifying the physical simulation.
    """
    from .controller import HybridImpedanceParameters

    grasp = np.array([0.229753, -0.973249, 0.0, 0.0])
    grasp /= np.linalg.norm(grasp)
    plate_rotation = np.array([-0.5, 0.5, -0.5, 0.5])
    slot_rotation = np.empty(4)
    mujoco.mju_mulQuat(slot_rotation, plate_rotation, grasp)
    home_position = np.array([0.306891, 0.0, 0.395732])
    home_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
    rack_position = np.array([0.5465, -0.2792, 0.001])
    rack_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses = []
    position = plate_position
    direction = home_position[:2] - position[:2]
    direction /= np.linalg.norm(direction)
    yaw = np.arctan2(direction[0], -direction[1])
    yaw_quaternion = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    orientation = np.empty(4)
    mujoco.mju_mulQuat(orientation, yaw_quaternion, grasp)
    for name, radius, height, opening, speed, stop in (
        ("pre-grasp", 0.106430, 0.030733, 0.5, 1.0, True),
        ("close", 0.094355, 0.006584, 0.0, 0.5, True),
        ("lift", 0.094355, 0.106584, 0.0, 0.5, False),
    ):
        target = position + np.array([radius * direction[0], radius * direction[1], height])
        poses.append((f"plate0/{name}", target, orientation, False, opening, speed, stop))
    slot = np.array([-0.0808, 0.0, 0.0])
    for name, position, opening, speed, stop in (
        ("align", (0.288081, 0.043752, 0.446659), 0.0, 1.0, True),
        ("above-slot", (-0.027792, 0.089415, 0.406743), 0.0, 0.5, True),
        ("lower", (-0.027792, 0.089415, 0.306743), 0.0, 0.25, True),
        ("insert-entry", (-0.027792, 0.089415, 0.276743), 0.0, 0.5, False),
        ("insert", (-0.027792, 0.089415, 0.246743), 0.0, 0.5, True),
        ("release", (-0.027792, 0.089415, 0.246743), 1.0, 1.0, True),
        ("adjust", (0.012208, 0.089415, 0.186743), 1.0, 1.0 / 3.0, True),
    ):
        poses.append((f"plate0/{name}", np.asarray(position) + slot, slot_rotation, True, opening, speed, stop))
    poses.append(
        (
            "plate0/post-adjust",
            np.array([0.112208, 0.089415, 0.186743]) + slot,
            slot_rotation,
            True,
            1.0,
            1.0 / 3.0,
            True,
        )
    )
    poses.append(("plate0/home", home_position, home_quaternion, False, 1.0, 1.0, True))
    limits = HybridImpedanceParameters()
    dense_poses = []
    for pose in poses:
        name, position, quaternion, in_rack, opening, speed, stop = pose
        if name.rsplit("/", 1)[-1] in ("lower", "insert-entry", "insert"):
            previous_position = dense_poses[-1][1]
            segments = int(np.ceil(np.linalg.norm(position - previous_position) / 0.005))
            # Short Cartesian segments keep joint interpolation close to a vertical line.
            for index in range(1, segments):
                intermediate = previous_position + index / segments * (position - previous_position)
                dense_poses.append((f"{name}-{index:02d}", intermediate, quaternion, in_rack, opening, speed, False))
        dense_poses.append(pose)
    poses = dense_poses
    ik_limits = np.array(
        [
            (-2.8007, 2.8007),
            (-1.7361, 1.7361),
            (-2.8007, 2.8007),
            (-2.977, -0.2169),
            (-2.7763, 2.7763),
            (0.5398, 4.5216),
            (-2.9508, 2.9508),
        ]
    )
    low = np.maximum(limits.joint_lower, ik_limits[:, 0])
    high = np.minimum(limits.joint_upper, ik_limits[:, 1])
    home = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
    data = mujoco.MjData(model)
    site = model.site("panda_gripper").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    current, desired, inverse, error = (np.empty(4) for _ in range(4))
    rotation_error = np.empty(3)
    joints = initial_joints.copy().astype(float)
    result = []
    for name, position, quaternion, in_rack, opening, speed, stop in poses:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        if in_rack:
            position = rack_position + rack_rotation @ position
            rotation = rack_rotation @ rotation
        target = position - rotation @ np.array([0.0, 0.0, 0.0945496])
        mujoco.mju_mat2Quat(desired, rotation.ravel())
        if name.endswith("/align"):
            joints = initial_joints.copy().astype(float)
        joints = np.clip(joints, low, high)
        for _ in range(140):
            data.qpos[:7] = joints
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            position_error = target - data.site_xpos[site]
            mujoco.mju_mat2Quat(current, data.site_xmat[site])
            mujoco.mju_negQuat(inverse, current)
            mujoco.mju_mulQuat(error, desired, inverse)
            mujoco.mju_quat2Vel(rotation_error, error, 1.0)
            if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 1e-3:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            jacobian = np.vstack((jacp[:, :7], jacr[:, :7]))
            gram = jacobian @ jacobian.T
            nullspace = np.eye(7) - jacobian.T @ np.linalg.solve(gram + 1e-6 * np.eye(6), jacobian)
            delta = jacobian.T @ np.linalg.solve(
                gram + 0.08**2 * np.eye(6), np.r_[position_error, rotation_error]
            ) + nullspace @ (0.03 * (home - joints))
            delta *= min(1.0, 0.25 / max(np.linalg.norm(delta), 1e-12))
            joints = np.clip(joints + delta, low, high)
        if np.linalg.norm(position_error) > 0.001:
            raise RuntimeError(f"{name}: IK position error {np.linalg.norm(position_error):.6f} m")
        result.append(JointWaypoint(name, joints.copy().astype(np.float32), opening, speed, stop))
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("newton", "physx"), default="newton")
    parser.add_argument("--rack_friction", type=float, default=0.2)
    parser.add_argument("--settle_seconds", type=float, default=3.0, help="Hold at home after replay [s]")
    parser.add_argument(
        "--require_success", action="store_true", help="Fail unless the released plate remains inserted"
    )
    parser.add_argument("--mode", choices=("hold", "replay", "benchmark"), default="hold")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--render", choices=("human", "rgb_array"))
    parser.add_argument("--video", action="store_true", help="Record the full replay as an MP4 at simulation speed")
    parser.add_argument("--output", type=Path, default=Path("logs/plate_rack"))
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive; warmup must be nonnegative")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 1.0:
        parser.error("settle_seconds must be finite and at least one second")
    if args.require_success and args.mode != "replay":
        parser.error("--require_success requires --mode replay")
    if args.video:
        if args.mode != "replay":
            parser.error("--video requires --mode replay")
        if shutil.which("ffmpeg") is None:
            parser.error("--video requires ffmpeg with the libx264 encoder")
        args.render = "rgb_array"
    args.output.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        f"franka_plate_rack_{args.backend}",
        render_mode=args.render,
        rack_friction=args.rack_friction,
    )
    video = None
    video_frames = 0
    try:
        observation, _ = env.reset(seed=0)
        runtime = env.unwrapped.runtime
        initial = observation["observation.state"].copy()
        actions = torch.as_tensor(initial, device=runtime.device).clone()
        if args.video:
            frame = env.render()
            height, width = frame.shape[:2]
            title = "Newton / MJWarp" if args.backend == "newton" else "PhysX"
            video_path = args.output / f"plate_rack_{args.backend}.mp4"
            video = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(1.0 / runtime.control_dt),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-vf",
                    f"drawtext=text='{title} | Plate rack':fontcolor=white:fontsize=26:"
                    "box=1:boxcolor=black@0.6:boxborderw=10:x=24:y=24",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                stdin=subprocess.PIPE,
            )
            video.stdin.write(np.ascontiguousarray(frame).tobytes())
            video_frames += 1
        if args.mode == "replay":
            plate_position = (runtime.plate.data.root_pos_w.torch - runtime.scene.env_origins)[0].cpu().numpy()
            plan = plan_waypoints(runtime.controller.model, plate_position, initial[:7])
            command = initial.copy()
            stages = []
            for stage in plan:
                start = command[:7].copy()
                distance = float(np.max(np.abs(stage.joints - start)))
                # Quintic interpolation has peak normalized velocity 1.875.
                duration = max(
                    1.875 * distance / stage.speed,
                    math.sqrt(5.774 * distance / 9.0),
                    (60 * distance / 100) ** (1 / 3),
                )
                if stage.opening != command[7]:
                    duration = max(
                        duration,
                        (0.28 if stage.opening > command[7] else 0.21)
                        + abs(stage.opening - command[7]) * 0.07575288 / 0.05,
                    )
                steps = max(1, math.ceil(duration / runtime.control_dt))
                for step in range(1, steps + 1 + int(stage.stop)):
                    fraction = min(step / steps, 1.0)
                    blend = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
                    command[:7] = start + blend * (stage.joints - start)
                    command[7] = stage.opening
                    actions.copy_(torch.from_numpy(command))
                    _, _, _, _, info = env.step(actions)
                    if video is not None:
                        video.stdin.write(np.ascontiguousarray(env.render()).tobytes())
                        video_frames += 1
                state = runtime.state().cpu().numpy()
                plate_positions = [runtime.plate.data.root_pos_w.torch[0].cpu().tolist()]
                record = {
                    "stage": stage.name,
                    "joint_error_rad": float(np.max(np.abs(state[:7] - stage.joints))),
                    "opening": float(state[7]),
                    "plate_positions_m": plate_positions,
                    "plate_quaternions_xyzw": [runtime.plate.data.root_quat_w.torch[0].cpu().tolist()],
                    "success": bool(info["success"]),
                }
                stages.append(record)
                print(json.dumps(record), flush=True)
                if args.render == "rgb_array":
                    from PIL import Image

                    Image.fromarray(env.render()).save(
                        args.output / f"{args.backend}_{stage.name.replace('/', '_')}.png"
                    )
            (args.output / f"{args.backend}_replay.json").write_text(json.dumps(stages, indent=2) + "\n")
            hold_steps = math.ceil(args.settle_seconds / runtime.control_dt)
            stable_steps = 0
            start_positions = runtime.plate.data.root_pos_w.torch.clone()
            max_drift = 0.0
            for _ in range(hold_steps):
                _, _, _, _, info = env.step(actions)
                velocities = runtime.plate.data.root_vel_w.torch
                linear_speed = float(velocities[..., :3].norm(dim=-1).max())
                angular_speed = float(velocities[..., 3:].norm(dim=-1).max())
                opening = float(runtime.state()[7])
                stable = bool(info["success"]) and opening > 0.95 and linear_speed < 0.005 and angular_speed < 0.05
                stable_steps = stable_steps + 1 if stable else 0
                positions = runtime.plate.data.root_pos_w.torch
                max_drift = max(max_drift, float((positions - start_positions).norm(dim=-1).max()))
                if video is not None:
                    video.stdin.write(np.ascontiguousarray(env.render()).tobytes())
                    video_frames += 1
            report = {
                "backend": args.backend,
                "layout": "plate_rack",
                "rack_friction": args.rack_friction,
                "success": stable_steps * runtime.control_dt >= 1.0,
                "hold_seconds": hold_steps * runtime.control_dt,
                "stable_seconds": stable_steps * runtime.control_dt,
                "max_position_drift_m": max_drift,
                "final_linear_speed_m_s": linear_speed,
                "final_angular_speed_rad_s": angular_speed,
                "opening": opening,
                "plate_positions_m": [runtime.plate.data.root_pos_w.torch[0].cpu().tolist()],
                "plate_quaternions_xyzw": [runtime.plate.data.root_quat_w.torch[0].cpu().tolist()],
            }
            print(json.dumps(report, indent=2), flush=True)
            (args.output / f"{args.backend}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
            if args.render == "rgb_array":
                from PIL import Image

                runtime.sim.visualizers[0].set_camera_view(eye=(0.95, -0.90, 0.52), target=(0.48, -0.27, 0.16))
                Image.fromarray(env.render()).save(args.output / f"{args.backend}_settled.png")
            if args.require_success and not report["success"]:
                raise RuntimeError("The released plate did not remain inserted and stationary for at least one second")
        else:
            for _ in range(args.warmup):
                env.step(actions)
            torch.cuda.synchronize(runtime.device)
            started = time.perf_counter()
            for _ in range(args.steps):
                env.step(actions)
            torch.cuda.synchronize(runtime.device)
            elapsed = time.perf_counter() - started
            state = runtime.state()
            if not bool(torch.isfinite(state).all()):
                raise RuntimeError("non-finite state after rollout")
            report = {
                "backend": args.backend,
                "gpu": torch.cuda.get_device_name(),
                "num_envs": 1,
                "warmup_steps": args.warmup,
                "measured_steps": args.steps,
                "elapsed_seconds": elapsed,
                "transitions_per_second": args.steps / elapsed,
                "max_joint_drift_rad": float((state[:7] - actions[:7]).abs().max()),
                "render_mode": args.render,
            }
            print(json.dumps(report, indent=2), flush=True)
            (args.output / f"{args.backend}_{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
            if args.render == "rgb_array":
                from PIL import Image

                Image.fromarray(env.render()).save(args.output / f"{args.backend}_scene.png")
    finally:
        try:
            if video is not None:
                video.stdin.close()
                if video.wait() != 0:
                    raise RuntimeError("FFmpeg failed to encode the replay; see its error output")
                print(f"Video: {video_path} ({video_frames} frames)", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
