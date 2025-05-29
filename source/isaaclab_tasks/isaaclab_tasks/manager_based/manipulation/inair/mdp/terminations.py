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
    plane_pos_z = plane_pos[:, 2] + 0.15
    plane_pos_z_expanded = plane_pos_z.expand_as(object_pos_z)
    
    return object_pos_z <= plane_pos_z_expanded

    