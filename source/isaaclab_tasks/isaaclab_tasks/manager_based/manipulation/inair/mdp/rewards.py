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
    
    # obtain the velocity
    lin_vel = asset.data.root_lin_vel_w
    ang_vel = asset.data.root_ang_vel_w
    
    # obtain the thresholds
    orientation_success_threshold = command_term.cfg.orientation_success_threshold
    velocity_success_threshold = command_term.cfg.velocity_success_threshold
    
    # calculate the orientation error
    orientation_dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)
    velocity_magnitude = torch.norm(lin_vel, dim=1) + torch.norm(ang_vel, dim=1)

    return (orientation_dtheta <= orientation_success_threshold) & (velocity_magnitude <= velocity_success_threshold)

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


def track_delta_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for tracking the object orientation by comparing current and previous orientation errors.
    
    This reward encourages the object to move toward the goal orientation by comparing the current
    orientation error with the previous orientation error. The reward is positive when the current
    error is less than the previous error, and negative when the current error is greater than
    the previous error.
    
    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)
    
    # get the current and previous orientation errors from the metrics
    current_error = abs(command_term.metrics["orientation_error"])
    previous_error = abs(command_term.metrics["previous_orientation_error"])
    
    return previous_error - current_error


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
    target_pos_w = [0.4, -0.4,  1.6] 
    num_envs = env.unwrapped.scene.num_envs
    goal_pos_w = torch.tensor(target_pos_w, device=env.device).repeat(num_envs, 1)
    # goal_pos_w = command_term.command[:, :3]
    # calculate the position error
    pos_error = torch.norm(asset.data.root_pos_w - goal_pos_w, dim=1)

    return 1.0 / (pos_error + pos_eps)


def above_palm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Returns 1.0 if the object is above the palm (palm_lower), -1.0 otherwise.
    """
    palm_lower_z = env.scene[asset_cfg.name].data.body_link_state_w[:, 10, 2]
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]

    is_above = object_z > palm_lower_z

    return torch.where(
        is_above,
        torch.tensor(1.0, device=object_z.device),
        torch.tensor(-1.0, device=object_z.device)
    )

def object_in_palm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    palm_lower_pos = env.scene[asset_cfg.name].data.body_link_state_w[:, 10, :3]
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, ]
    
    distance = torch.norm(palm_lower_pos - object_pos, dim=-1)
    
    return distance

def limit_object_xy_range(
    env: ManagerBasedRLEnv,
    range_limits: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Check whether the object's (x, y) position is within a specified range from the robot's palm (palm_lower).

    Args:
        env: The environment instance.
        range_limits: A tuple (limit_x, limit_y) specifying the maximum allowed displacement in x and y directions.
        asset_cfg: Configuration for the robot asset, defaulting to "robot".
        object_cfg: Configuration for the object asset, defaulting to "object".

    Returns:
        A float tensor of shape (num_envs,) where 1.0 indicates the object is within range and 0.0 otherwise.
    """
    palm_lower_xy = env.scene[asset_cfg.name].data.body_link_state_w[:, 10, :2]
    object_xy = env.scene[object_cfg.name].data.root_pos_w[:, :2]

    delta_xy = object_xy - palm_lower_xy
    abs_delta_xy = torch.abs(delta_xy)

    limit_x, limit_y = range_limits
    in_range = (abs_delta_xy[:, 0] < limit_x) & (abs_delta_xy[:, 1] < limit_y)
    binary_output = torch.where(in_range, torch.tensor(1.0, device=object_xy.device), torch.tensor(-1.0, device=object_xy.device))

    return binary_output

def track_object_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Compute the L1 distance in (x, y) between the object and the robot's palm (palm_lower).

    Args:
        env: The environment instance.
        range_limits: Unused here (kept for compatibility).
        asset_cfg: Configuration for the robot asset.
        object_cfg: Configuration for the object asset.

    Returns:
        A float tensor of shape (num_envs,) representing the distance in the xy-plane.
    """
    palm_lower_xy = env.scene[asset_cfg.name].data.body_link_state_w[:, 10, :2]
    object_xy = env.scene[object_cfg.name].data.root_pos_w[:, :2]

    delta_xy = object_xy - palm_lower_xy
    abs_delta_xy = torch.abs(delta_xy)

    # Return L1 distance: |dx| + |dy|
    return torch.sum(abs_delta_xy, dim=1)


def palm_drop_penalty(
    env: ManagerBasedRLEnv,
    init_pos_z: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    palm_pos_z = robot.data.body_link_state_w[:, 10, 2]

    out_of_range = torch.abs(init_pos_z - palm_pos_z) >= 0.10
    return torch.where(
        out_of_range,
        torch.tensor(1.0, device=palm_pos_z.device),
        torch.tensor(-1.0, device=palm_pos_z.device)
    )
    
def object_height_penalty(
    env: ManagerBasedRLEnv,
    threshold,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    is_too_high = object_z >= threshold
    
    return torch.where(is_too_high, 1.0, -1.0)


def object_vel_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    object: RigidObject = env.scene[object_cfg.name]
    object_lin_vel = object.data.root_lin_vel_w
    object_ang_vel = object.data.root_ang_vel_w
    
    object_velocity_magnitude = torch.norm(object_lin_vel, dim=1) + torch.norm(object_ang_vel, dim=1)
    return object_velocity_magnitude

def dummy_reward(env):
    return torch.zeros(env.num_envs, device=env.device)

def object_speed_z_above_threshold(
    env: ManagerBasedRLEnv,
    threshold,
) -> torch.Tensor:
    """
    Returns 1.0 if object's linear z-velocity is greater than threshold, else -1.0 (binary reward).
    """
    object_vel_z = env.scene["object"].data.root_lin_vel_w[:, 2]  # shape: (num_envs,)
    reward = torch.where(
        object_vel_z > threshold,
        torch.tensor(1.0, device=object_vel_z.device),
        torch.tensor(-1.0, device=object_vel_z.device)
    )
    return reward

def object_z_far_from_hand(
    env: ManagerBasedRLEnv,
    threshold,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Returns 1.0 if the absolute z-distance between object and hand exceeds threshold, else -1.0.
    """
    palm_lower_z = env.scene[asset_cfg.name].data.body_link_state_w[:, 10, 2]     # (num_envs,)
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]             # (num_envs,)
    dist_z = torch.abs(object_z - palm_lower_z)                                  # (num_envs,)
    
    reward = torch.where(
        dist_z > threshold,
        torch.tensor(1.0, device=dist_z.device),
        torch.tensor(-1.0, device=dist_z.device)
    )
    return reward

