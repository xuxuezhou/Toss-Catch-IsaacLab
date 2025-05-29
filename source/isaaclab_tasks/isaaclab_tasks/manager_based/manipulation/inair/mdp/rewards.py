# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.conditions import has_object_hand_contact, is_object_ready_to_end, is_object_static, is_orientation_aligned
from isaaclab.envs.mdp.conditions import THRESHOLD

if TYPE_CHECKING:
    from .commands import InAirReOrientationCommand

def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)
    goal_quat_w = command_term.command[:, 3:7]
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return 1.0 / (dtheta + rot_eps)


def track_delta_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)
    current_error = abs(command_term.metrics["orientation_error"])
    previous_error = abs(command_term.metrics["previous_orientation_error"])
    
    return previous_error - current_error


def above_palm(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    palm_lower_z = env.scene[robot_cfg.name].data.body_link_state_w[:, 11, 2]
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    is_above = object_z > palm_lower_z
    
    return torch.where(is_above, 1.0, -1.0)

def track_object_l2(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    palm_lower_xy = env.scene[robot_cfg.name].data.body_link_state_w[:, 11, :2]
    object_xy = env.scene[object_cfg.name].data.root_pos_w[:, :2]
    delta_xy = object_xy - palm_lower_xy

    return torch.norm(delta_xy, dim=1)

def palm_drop_penalty(
    env: ManagerBasedRLEnv,
    init_pos_z: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    palm_pos_z = robot.data.body_link_state_w[:, 11, 2]

    out_of_range = torch.abs(init_pos_z - palm_pos_z) >= 0.10
    return torch.where(out_of_range, 1.0, -1.0)

def object_vel_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    object: RigidObject = env.scene[object_cfg.name]
    object_lin_vel = object.data.root_lin_vel_w
    object_ang_vel = object.data.root_ang_vel_w
    
    object_velocity_magnitude = torch.norm(object_lin_vel, dim=1) + torch.norm(object_ang_vel, dim=1)
    return object_velocity_magnitude


def success_bonus(env) -> torch.Tensor:
    orientation_align = is_orientation_aligned(env)
    object_inhand = has_object_hand_contact(env)
    static = is_object_static(env)
    return orientation_align & object_inhand & static


def open_fingertips(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    fingertips_pos = robot.data.body_link_state_w[:, -4:, :3]  # [envs, 4, 3]

    # index=0, thumb=1, middle=2, ring=3
    index_to_thumb = torch.norm(fingertips_pos[:, 0] - fingertips_pos[:, 1], dim=-1)
    middle_to_thumb = torch.norm(fingertips_pos[:, 2] - fingertips_pos[:, 1], dim=-1)
    ring_to_thumb = torch.norm(fingertips_pos[:, 3] - fingertips_pos[:, 1], dim=-1)

    total_distance = index_to_thumb + middle_to_thumb + ring_to_thumb

    return total_distance

def grasp_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]

    fingertips_pos = robot.data.body_link_state_w[:, -4:, :3]  # [envs, 4, 3]
    object_pos = object.data.root_pos_w[:, :3]                    # [envs, 3]
    dists = torch.norm(fingertips_pos - object_pos.unsqueeze(1), dim=-1)  # [envs, 4]

    return dists.mean(dim=1)  # shape: [envs]


def undesired_forces(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    joint_wrench = robot.data.body_incoming_joint_wrench_b # (num_envs, num_links, 6)
    hand_joint_forces = joint_wrench[:, 11, :3]
    
    forces_norm = torch.norm(hand_joint_forces, dim=-1)
    mean_force = torch.mean(forces_norm, dim=-1)
    
    return mean_force
    
def dummy_reward(env):
    return torch.zeros(env.num_envs, device=env.device)