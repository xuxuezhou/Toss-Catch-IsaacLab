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

if TYPE_CHECKING:
    from .commands import InAirReOrientationCommand


def success_bonus(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Bonus reward for successfully reaching the goal.

    The object is considered to have reached the goal when the object orientation is within the threshold.
    The reward is 1.0 if the object has reached the goal, otherwise 0.0.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # obtain the threshold for the orientation error
    threshold = command_term.cfg.orientation_success_threshold
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return dtheta <= threshold


def track_pos_l2(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for tracking the object position using the L2 norm.

    The reward is the distance between the object position and the goal position.

    Args:
        env: The environment object.
        command_term: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal position
    goal_pos_e = command_term.command[:, 0:3]
    # obtain the object position in the environment frame
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins

    return torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the object orientation using the inverse of the orientation error.

    The reward is the inverse of the orientation error between the object orientation and the goal orientation.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        rot_eps: The threshold for the orientation error. Default is 1e-3.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return 1.0 / (dtheta + rot_eps)


def track_position_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the object position using the inverse of the position error.

    The reward is the inverse of the position error between the object position and the goal position.
    This encourages the object to stay close to the target position.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        pos_eps: The threshold for the position error. Default is 1e-3.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal position
    initial_pos =[0.55, -0.18, 1.32]
    num_envs = env.unwrapped.scene.num_envs
    goal_pos_w = torch.tensor(initial_pos, device=env.device).repeat(num_envs, 1)
    # goal_pos_w = command_term.command[:, :3]
    # calculate the position error
    pos_error = torch.norm(asset.data.root_pos_w - goal_pos_w, dim=1)

    return 1.0 / (pos_error + pos_eps)


def velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    velocity_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalty for high velocities when the object is close to the target pose.

    This penalty encourages the object to stabilize at the target pose by penalizing high velocities.
    The penalty is only applied when the object is close to the target pose.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        velocity_threshold: The threshold for considering velocities as high. Default is 0.01.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # check if object is close to target pose
    orientation_error = math_utils.quat_error_magnitude(asset.data.root_quat_w, command_term.command[:, 3:7])
    position_error = torch.norm(asset.data.root_pos_w - command_term.command[:, :3], dim=1)
    
    # only apply penalty when close to target
    close_to_target = (orientation_error < command_term.cfg.orientation_success_threshold) & (position_error < 0.01)
    
    # compute velocity magnitude
    lin_vel = asset.data.root_lin_vel_w
    ang_vel = asset.data.root_ang_vel_w
    velocity_magnitude = torch.norm(lin_vel, dim=1) + torch.norm(ang_vel, dim=1)
    
    # apply penalty only when close to target and velocity is high
    penalty = torch.where(
        close_to_target & (velocity_magnitude > velocity_threshold),
        -(velocity_magnitude - velocity_threshold),
        torch.zeros_like(velocity_magnitude)
    )
    
    return penalty


def acceleration_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    acceleration_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalty for high accelerations when the object is close to the target pose.

    This penalty encourages smooth motion by penalizing high accelerations.
    The penalty is only applied when the object is close to the target pose.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        acceleration_threshold: The threshold for considering accelerations as high. Default is 0.01.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # check if object is close to target pose
    orientation_error = math_utils.quat_error_magnitude(asset.data.root_quat_w, command_term.command[:, 3:7])
    position_error = torch.norm(asset.data.root_pos_w - command_term.command[:, :3], dim=1)
    
    # only apply penalty when close to target
    close_to_target = (orientation_error < command_term.cfg.orientation_success_threshold) & (position_error < 0.01)
    
    # compute acceleration magnitude
    lin_acc = asset.data.body_lin_acc_w.squeeze(1)
    ang_acc = asset.data.body_ang_acc_w.squeeze(1)
    acceleration_magnitude = torch.norm(lin_acc, dim=1) + torch.norm(ang_acc, dim=1)
    
    # apply penalty only when close to target and acceleration is high
    penalty = torch.where(
        close_to_target & (acceleration_magnitude > acceleration_threshold),
        -(acceleration_magnitude - acceleration_threshold),
        torch.zeros_like(acceleration_magnitude)
    )
    
    return penalty
