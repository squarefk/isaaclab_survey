# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""FR3 impedance control and joint efforts for either physics backend."""

import math
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .assets import robot_spec


@dataclass(frozen=True)
class HybridImpedanceParameters:
    cartesian_stiffness: tuple[float, ...] = (
        400.0,
        400.0,
        400.0,
        15.0,
        15.0,
        15.0,
    )
    cartesian_damping: tuple[float, ...] = (37.0, 37.0, 37.0, 2.0, 2.0, 2.0)
    joint_stiffness: tuple[float, ...] = (40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0)
    joint_damping: tuple[float, ...] = (4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0)
    cartesian_lower: tuple[float, ...] = (-1.0, -1.0, -1.0)
    cartesian_upper: tuple[float, ...] = (1.0, 1.0, 1.0)
    joint_lower: tuple[float, ...] = (-2.65, -1.68, -2.80, -2.95, -2.70, 0.45, -2.90)
    joint_upper: tuple[float, ...] = (2.65, 1.68, 2.80, -0.16, 2.70, 4.40, 2.90)
    velocity_limit: tuple[float, ...] = (2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51)
    torque_limit: tuple[float, ...] = (86.0, 86.0, 86.0, 86.0, 11.5, 11.5, 11.5)
    joint_margin: float = 0.2
    velocity_margin: float = 0.5
    cartesian_margin: float = 0.05
    joint_limit_stiffness: float = 50.0
    velocity_limit_stiffness: float = 20.0
    cartesian_limit_stiffness: float = 200.0
    torque_filter_cutoff_hz: float = 100.0
    max_torque_rate: float = 1000.0 - 1e-3
    bias_integral_rate: float = 4.0
    bias_torque_limit: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5)
    friction_velocity: float = 0.02
    velocity_idle_seconds: float = 0.1
    velocity_slew_limit: float = 8.0
    load_mass: float = 0.273
    load_center: tuple[float, float, float] = (0.074, -0.043, 0.150)
    load_inertia: tuple[float, float, float] = (0.0008, 0.0008, 0.0005)


@wp.kernel
def _update_position_reference(
    targets: wp.array3d(dtype=float),
    position_command_active: wp.array(dtype=bool),
    timestep: float,
    idle_seconds: float,
    slew_limit: float,
    velocity_limit: wp.array(dtype=float),
    previous_target: wp.array2d(dtype=float),
    elapsed: wp.array(dtype=float),
    estimated_velocity: wp.array2d(dtype=float),
):
    world = wp.tid()
    interval = elapsed[world] + timestep
    active = position_command_active[world]
    expired = not active or elapsed[world] < 0.0 or interval > idle_seconds
    moved = False
    for joint in range(7):
        if targets[world, 0, joint] != previous_target[world, joint]:
            moved = True
    for joint in range(7):
        position = targets[world, 0, joint]
        estimate = estimated_velocity[world, joint]
        if expired:
            estimate = 0.0
        elif moved:
            estimate = (position - previous_target[world, joint]) / wp.max(interval, 1e-6)
        if expired or moved:
            previous_target[world, joint] = position
        estimated_velocity[world, joint] = estimate
        target_velocity = wp.clamp(estimate, -velocity_limit[joint], velocity_limit[joint])
        maximum_change = slew_limit * timestep
        velocity = targets[world, 1, joint]
        velocity += wp.clamp(target_velocity - velocity, -maximum_change, maximum_change)
        targets[world, 1, joint] = velocity
        targets[world, 2, joint] = 0.0
    elapsed[world] = -1.0 if not active else (0.0 if expired or moved else interval)


@wp.kernel
def _configure_hybrid_dynamics(
    body_mass: wp.array2d(dtype=float),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_inertia: wp.array2d(dtype=wp.vec3),
    flange_body: int,
    load_mass: float,
    load_center: wp.vec3,
    load_inertia: wp.vec3,
):
    body_mass[0, flange_body] = load_mass
    body_ipos[0, flange_body] = load_center
    body_inertia[0, flange_body] = load_inertia


@wp.func
def _soft_limit_effort(
    value: float,
    lower: float,
    upper: float,
    margin: float,
    stiffness: float,
) -> float:
    effort = float(0.0)
    upper_violation = value - upper
    lower_violation = lower - value
    if upper_violation > -margin:
        effort = effort - stiffness * (margin + upper_violation)
    if lower_violation > -margin:
        effort = effort + stiffness * (margin + lower_violation)
    return effort


@wp.kernel
def _frame_points(
    body_position: wp.array2d(dtype=wp.vec3),
    site_position: wp.array2d(dtype=wp.vec3),
    flange_body: int,
    end_effector_site: int,
    flange_point: wp.array(dtype=wp.vec3),
    end_effector_point: wp.array(dtype=wp.vec3),
):
    world = wp.tid()
    flange_point[world] = body_position[world, flange_body]
    end_effector_point[world] = site_position[world, end_effector_site]


@wp.kernel
def _hybrid_impedance_torque(
    desired: wp.array3d(dtype=float),
    position_command_active: wp.array(dtype=bool),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qfrc_bias: wp.array2d(dtype=float),
    friction_loss: wp.array2d(dtype=float),
    friction_velocity: float,
    qpos_addresses: wp.array(dtype=int),
    dof_addresses: wp.array(dtype=int),
    actuator_addresses: wp.array(dtype=int),
    flange_jacobian_position: wp.array3d(dtype=float),
    flange_jacobian_rotation: wp.array3d(dtype=float),
    end_effector_jacobian_position: wp.array3d(dtype=float),
    end_effector_position: wp.array(dtype=wp.vec3),
    cartesian_stiffness: wp.array(dtype=float),
    cartesian_damping: wp.array(dtype=float),
    joint_stiffness: wp.array(dtype=float),
    joint_damping: wp.array(dtype=float),
    cartesian_lower: wp.array(dtype=float),
    cartesian_upper: wp.array(dtype=float),
    joint_lower: wp.array(dtype=float),
    joint_upper: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    torque_limit: wp.array(dtype=float),
    joint_margin: float,
    velocity_margin: float,
    cartesian_margin: float,
    joint_limit_stiffness: float,
    velocity_limit_stiffness: float,
    cartesian_limit_stiffness: float,
    filter_gain: float,
    maximum_torque_step: wp.array(dtype=float),
    bias_torque_limit: wp.array(dtype=float),
    bias_integral_step: float,
    bias_hold_step: float,
    command: wp.array3d(dtype=float),
    controls: wp.array2d(dtype=float),
):
    world, joint = wp.tid()
    qpos_address = qpos_addresses[joint]
    dof_address = dof_addresses[joint]
    position = qpos[world, qpos_address]
    velocity = qvel[world, dof_address]
    position_error = desired[world, 0, joint] - position
    velocity_error = desired[world, 1, joint] - velocity
    torque = joint_stiffness[joint] * position_error + joint_damping[joint] * velocity_error
    proportional = joint_stiffness[joint] * position_error

    if position_command_active[world]:
        integration_allowed = True
        moving = False
        cartesian_error_squared = 0.0
        for row in range(6):
            cartesian_position_error = float(0.0)
            cartesian_velocity_error = float(0.0)
            joint_jacobian = float(0.0)
            for source_joint in range(7):
                source_dof = dof_addresses[source_joint]
                source_error = desired[world, 0, source_joint] - qpos[world, qpos_addresses[source_joint]]
                if wp.abs(source_error) > 0.03:
                    integration_allowed = False
                if wp.abs(desired[world, 1, source_joint]) >= 0.01:
                    moving = True
                source_velocity_error = desired[world, 1, source_joint] - qvel[world, source_dof]
                jacobian = float(0.0)
                if row < 3:
                    jacobian = flange_jacobian_position[world, row, source_dof]
                else:
                    jacobian = flange_jacobian_rotation[world, row - 3, source_dof]
                cartesian_position_error += jacobian * source_error
                cartesian_velocity_error += jacobian * source_velocity_error
                if source_joint == joint:
                    joint_jacobian = jacobian
            wrench = (
                cartesian_stiffness[row] * cartesian_position_error + cartesian_damping[row] * cartesian_velocity_error
            )
            torque += joint_jacobian * wrench
            if row < 3:
                cartesian_error_squared += cartesian_position_error * cartesian_position_error
            proportional += joint_jacobian * cartesian_stiffness[row] * cartesian_position_error

        direction = 0.0
        if wp.abs(desired[world, 1, joint]) > 0.01:
            direction = wp.sign(desired[world, 1, joint])
        previous_bias = command[world, 1, joint]
        if direction != command[world, 0, joint] and direction != 0.0:
            previous_bias = 0.0
        command[world, 0, joint] = direction
        if not integration_allowed or cartesian_error_squared > 0.000025:
            proportional = 0.0
        integral_step = bias_integral_step if moving else bias_hold_step
        command[world, 1, joint] = wp.clamp(
            previous_bias + integral_step * proportional,
            -bias_torque_limit[joint],
            bias_torque_limit[joint],
        )
    torque += command[world, 1, joint]
    torque += friction_loss[0, dof_address] * wp.tanh(desired[world, 1, joint] / friction_velocity)

    point = end_effector_position[world]
    for axis in range(3):
        cartesian_force = _soft_limit_effort(
            point[axis],
            cartesian_lower[axis],
            cartesian_upper[axis],
            cartesian_margin,
            cartesian_limit_stiffness,
        )
        torque += end_effector_jacobian_position[world, axis, dof_address] * cartesian_force
    torque += _soft_limit_effort(
        position,
        joint_lower[joint],
        joint_upper[joint],
        joint_margin,
        joint_limit_stiffness,
    )
    torque += _soft_limit_effort(
        velocity,
        -velocity_limit[joint],
        velocity_limit[joint],
        velocity_margin,
        velocity_limit_stiffness,
    )
    requested = wp.clamp(torque, -torque_limit[joint], torque_limit[joint])
    previous = command[world, 2, joint]
    filtered = filter_gain * requested + (1.0 - filter_gain) * previous
    limited = wp.clamp(
        filtered,
        previous - maximum_torque_step[joint],
        previous + maximum_torque_step[joint],
    )
    command[world, 2, joint] = limited
    controls[world, actuator_addresses[joint]] = qfrc_bias[world, dof_address] + limited


class HybridImpedanceController:
    JOINTS = 7

    def __init__(
        self,
        cycle_dt: float,
        model_nv: int,
        flange_body_id: int,
        end_effector_body_id: int,
        end_effector_site_id: int,
        parameters: HybridImpedanceParameters | None = None,
    ) -> None:
        self._cycle_dt = cycle_dt
        self._model_nv = model_nv
        self._flange_body_id = flange_body_id
        self._end_effector_body_id = end_effector_body_id
        self._end_effector_site_id = end_effector_site_id
        self._parameters = parameters or HybridImpedanceParameters()

    def activate(
        self,
        device: torch.device,
        warp_device: object,
    ) -> None:
        self._previous_target = torch.zeros((1, 7), dtype=torch.float32, device=device)
        self._estimated_velocity = torch.zeros_like(self._previous_target)
        self._reference_elapsed = torch.full((1,), -1.0, dtype=torch.float32, device=device)
        self._previous_target_warp = wp.from_torch(self._previous_target)
        self._estimated_velocity_warp = wp.from_torch(self._estimated_velocity)
        self._reference_elapsed_warp = wp.from_torch(self._reference_elapsed)
        self._dynamics_configured = False
        self._position_command_active = torch.zeros(
            1,
            dtype=torch.bool,
            device=device,
        )
        self._position_command_active_warp = wp.from_torch(
            self._position_command_active,
            dtype=wp.bool,
        )
        self._target_lower = torch.tensor(
            self._parameters.joint_lower,
            dtype=torch.float32,
            device=device,
        )
        self._target_upper = torch.tensor(
            self._parameters.joint_upper,
            dtype=torch.float32,
            device=device,
        )
        self._flange_point = wp.empty(
            1,
            dtype=wp.vec3,
            device=warp_device,
        )
        self._end_effector_point = wp.empty_like(self._flange_point)
        self._flange_body = wp.array(
            np.full(1, self._flange_body_id, dtype=np.int32),
            dtype=wp.int32,
            device=warp_device,
        )
        self._end_effector_body = wp.array(
            np.full(1, self._end_effector_body_id, dtype=np.int32),
            dtype=wp.int32,
            device=warp_device,
        )
        jacobian_shape = (1, 3, self._model_nv)
        self._flange_jacobian_position = wp.empty(
            jacobian_shape,
            dtype=wp.float32,
            device=warp_device,
        )
        self._flange_jacobian_rotation = wp.empty_like(self._flange_jacobian_position)
        self._end_effector_jacobian_position = wp.empty_like(self._flange_jacobian_position)
        parameter_vectors = (
            self._parameters.cartesian_stiffness,
            self._parameters.cartesian_damping,
            self._parameters.joint_stiffness,
            self._parameters.joint_damping,
            self._parameters.cartesian_lower,
            self._parameters.cartesian_upper,
            self._parameters.joint_lower,
            self._parameters.joint_upper,
            self._parameters.velocity_limit,
            self._parameters.torque_limit,
            (self._parameters.max_torque_rate * min(self._cycle_dt, 0.001),) * self.JOINTS,
            self._parameters.bias_torque_limit,
        )
        self._parameter_vectors = tuple(
            wp.array(
                np.asarray(values, dtype=np.float32),
                dtype=wp.float32,
                device=warp_device,
            )
            for values in parameter_vectors
        )
        self._filter_gain = self._cycle_dt / (
            self._cycle_dt + 1.0 / (2.0 * math.pi * self._parameters.torque_filter_cutoff_hz)
        )

    @property
    def reference_state(self) -> tuple[torch.Tensor, ...]:
        return (
            self._previous_target,
            self._estimated_velocity,
            self._reference_elapsed,
        )

    def reset(
        self,
        positions: torch.Tensor,
        desired: torch.Tensor,
        command: torch.Tensor,
    ) -> None:
        desired.zero_()
        desired[:, 0].copy_(positions)
        self._previous_target.copy_(positions)
        self._estimated_velocity.zero_()
        self._reference_elapsed.fill_(-1.0)
        command.zero_()
        self._position_command_active.fill_(False)

    def prepare(
        self,
        target_positions: torch.Tensor,
        desired: torch.Tensor,
    ) -> None:
        desired[:, 0].copy_(torch.clamp(target_positions, self._target_lower, self._target_upper))
        self._position_command_active.fill_(True)

    def launch(
        self,
        substep: int,
        dof_addresses: object,
        command: object,
        qdd: object,
    ) -> None:
        del substep, dof_addresses, command, qdd

    def launch_torque(
        self,
        model: object,
        data: object,
        qpos_addresses: object,
        dof_addresses: object,
        actuator_addresses: object,
        desired: object,
        command: object,
    ) -> None:
        wp.launch(
            _update_position_reference,
            dim=1,
            inputs=[
                desired,
                self._position_command_active_warp,
                self._cycle_dt,
                self._parameters.velocity_idle_seconds,
                self._parameters.velocity_slew_limit,
                self._parameter_vectors[8],
            ],
            outputs=[
                self._previous_target_warp,
                self._reference_elapsed_warp,
                self._estimated_velocity_warp,
            ],
        )
        if not self._dynamics_configured:
            wp.launch(
                _configure_hybrid_dynamics,
                dim=1,
                inputs=[
                    model.body_mass,
                    model.body_ipos,
                    model.body_inertia,
                    self._flange_body_id,
                    self._parameters.load_mass,
                    wp.vec3(
                        self._parameters.load_center[0],
                        self._parameters.load_center[1],
                        self._parameters.load_center[2],
                    ),
                    wp.vec3(
                        self._parameters.load_inertia[0],
                        self._parameters.load_inertia[1],
                        self._parameters.load_inertia[2],
                    ),
                ],
            )
            mjw.set_const(model, data)
            self._dynamics_configured = True
        wp.launch(
            _frame_points,
            dim=1,
            inputs=[
                data.xpos,
                data.site_xpos,
                self._flange_body_id,
                self._end_effector_site_id,
            ],
            outputs=[self._flange_point, self._end_effector_point],
        )
        mjw.jac(
            model,
            data,
            self._flange_jacobian_position,
            self._flange_jacobian_rotation,
            self._flange_point,
            self._flange_body,
        )
        mjw.jac(
            model,
            data,
            self._end_effector_jacobian_position,
            None,
            self._end_effector_point,
            self._end_effector_body,
        )
        (
            cartesian_stiffness,
            cartesian_damping,
            joint_stiffness,
            joint_damping,
            cartesian_lower,
            cartesian_upper,
            joint_lower,
            joint_upper,
            velocity_limit,
            torque_limit,
            maximum_torque_step,
            bias_torque_limit,
        ) = self._parameter_vectors
        wp.launch(
            _hybrid_impedance_torque,
            dim=(1, self.JOINTS),
            inputs=[
                desired,
                self._position_command_active_warp,
                data.qpos,
                data.qvel,
                data.qfrc_bias,
                model.dof_frictionloss,
                self._parameters.friction_velocity,
                qpos_addresses,
                dof_addresses,
                actuator_addresses,
                self._flange_jacobian_position,
                self._flange_jacobian_rotation,
                self._end_effector_jacobian_position,
                self._end_effector_point,
                cartesian_stiffness,
                cartesian_damping,
                joint_stiffness,
                joint_damping,
                cartesian_lower,
                cartesian_upper,
                joint_lower,
                joint_upper,
                velocity_limit,
                torque_limit,
                self._parameters.joint_margin,
                self._parameters.velocity_margin,
                self._parameters.cartesian_margin,
                self._parameters.joint_limit_stiffness,
                self._parameters.velocity_limit_stiffness,
                self._parameters.cartesian_limit_stiffness,
                self._filter_gain,
                maximum_torque_step,
                bias_torque_limit,
                self._cycle_dt * self._parameters.bias_integral_rate,
                self._cycle_dt,
            ],
            outputs=[command, data.ctrl],
        )


class RobotController:
    """Compute joint efforts without advancing or writing any physical body pose."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.dt = 0.006
        self.finger_open = 0.03787644
        spec = robot_spec()
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.model = spec.compile()
        with wp.ScopedDevice(device):
            self.warp_model = mjw.put_model(self.model)
            self.data = mjw.make_data(self.model, nworld=1, nconmax=1, njmax=1)
            self.arm = HybridImpedanceController(
                self.dt,
                self.model.nv,
                self.model.body("fr3v2_link8").id,
                self.model.body("panda_hand").id,
                self.model.site("panda_gripper").id,
            )
            self.arm.activate(torch.device(device), wp.get_device(device))
            self.indices = wp.array(list(range(7)), dtype=wp.int32, device=device)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.controls = wp.to_torch(self.data.ctrl)
        self.desired = torch.zeros((1, 3, 7), device=device)
        self.command = torch.zeros_like(self.desired)
        self.finger = torch.full((1,), self.finger_open, device=device)
        self.finger_target = self.finger.clone()
        self.finger_delay = torch.zeros_like(self.finger)
        self.last_action = torch.zeros((1, 8), device=device)
        self.efforts = torch.zeros((1, 9), device=device)
        self.desired_warp = wp.from_torch(self.desired)
        self.command_warp = wp.from_torch(self.command)

    def reset(self, positions: torch.Tensor) -> None:
        self.arm.reset(positions[:, :7], self.desired, self.command)
        self.finger.copy_(positions[:, 7])
        self.finger_target.copy_(self.finger)
        self.finger_delay.zero_()
        self.last_action.copy_(torch.cat((positions[:, :7], positions[:, 7:8] / self.finger_open), dim=1))
        self.efforts.zero_()

    def prepare(self, actions: torch.Tensor) -> None:
        self.arm.prepare(actions[:, :7], self.desired)
        target = actions[:, 7] * self.finger_open
        changed = actions[:, 7] != self.last_action[:, 7]
        previous_direction = torch.sign(self.finger_target - self.finger)
        direction = torch.sign(target - self.finger)
        delay = torch.where(direction > 0.0, 0.28, 0.21)
        self.finger_delay.copy_(torch.where(changed & (direction != previous_direction), delay, self.finger_delay))
        self.finger_target.copy_(torch.where(changed, target, self.finger_target))
        self.last_action.copy_(actions)

    def compute(self, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        self.qpos.copy_(positions)
        self.qvel.copy_(velocities)
        mjw.fwd_position(self.warp_model, self.data)
        mjw.fwd_velocity(self.warp_model, self.data)
        self.arm.launch_torque(
            self.warp_model,
            self.data,
            self.indices,
            self.indices,
            self.indices,
            self.desired_warp,
            self.command_warp,
        )
        delayed = self.finger_delay.clamp(0.0, self.dt)
        self.finger_delay.sub_(delayed)
        change = 0.025 * (self.dt - delayed)
        self.finger.copy_(
            torch.maximum(
                torch.minimum(self.finger_target, self.finger + change),
                self.finger - change,
            )
        )
        self.efforts[:, :7].copy_(self.controls[:, :7])
        tendon_position = 0.5 * (positions[:, 7] + positions[:, 8])
        force = (20000.0 * (self.finger - tendon_position)).clamp(-100.0, 100.0)
        self.efforts[:, 7:].copy_(0.5 * force[:, None])
        return self.efforts
