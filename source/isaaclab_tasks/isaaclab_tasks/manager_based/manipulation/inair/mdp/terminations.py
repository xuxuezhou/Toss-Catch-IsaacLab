# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InAirReOrientationCommand


def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int, command_name: str) -> torch.Tensor:
    """Check if the task has been completed consecutively for a certain number of times.

    Args:
        env: The environment object.
        num_success: Threshold for the number of consecutive successes required.
        command_name: The command term to be used for extracting the goal.
    """
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    return command_term.metrics["consecutive_success"] >= num_success


def object_away_from_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the goal.

    The object is considered to be out-of-reach if the distance between the goal and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)
    asset = env.scene[object_cfg.name]

    # object pos
    asset_pos_e = asset.data.root_pos_w - env.scene.env_origins
    goal_pos_e = command_term.command[:, :3]

    return torch.norm(asset_pos_e - goal_pos_e, p=2, dim=1) > threshold


def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the robot.

    The object is considered to be out-of-reach if the distance between the robot and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        asset_cfg: The configuration for the robot entity. Default is "robot".
        object_cfg: The configuration for the object entity. Default is "object".
    """
    # extract useful elements
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]

    # compute distance
    dist = torch.norm(robot.data.root_pos_w - object.data.root_pos_w, dim=1)

    return dist > threshold


def object_away_from_palm(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the robot's palm (palm_lower).

    The object is considered to be out-of-reach if the distance between the robot's palm (palm_lower) and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot's palm and the object.
        asset_cfg: The configuration for the robot entity. Default is "robot".
        object_cfg: The configuration for the object entity. Default is "object".
    """
    # extract useful elements
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]
    
    palm_lower_state = robot.data.body_link_state_w[:, 10]  # 0-based indexing, so 6 is the 7th link
    palm_lower_pos = palm_lower_state[:, :3]  # first 3 elements are position

    # compute distance
    dist = torch.norm(palm_lower_pos - object.data.root_pos_w, dim=1)

    return dist > threshold


def land_on_floor(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plane_cfg: SceneEntityCfg = SceneEntityCfg("plane"),
) -> torch.Tensor:
    """Check if object has land on the floor.

    The task is considered to be failure if the object is on the floor.

    Args:
        env: The environment object.
        object_cfg: The configuration for the object entity. Default is "object".
        plane_cfg: The configuration for the plane entity. Default is "plane".
    """
    object = env.scene[object_cfg.name]
    plane = env.scene[plane_cfg.name]
    object_pos_z = object.data.root_pos_w[:, 2]
    
    plane_pos, _ = plane.get_world_poses()
    plane_pos_z = plane_pos[:, 2] + 0.1
    plane_pos_z_expanded = plane_pos_z.expand_as(object_pos_z)
    
    return object_pos_z <= plane_pos_z_expanded
    