# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from typing import Tuple 

"""
Transition Conditions

"""

class THRESHOLD:
    x_thresh: float = 0.05
    y_thresh: float = 0.05
    z_thresh: float = 0.15
    
    object_static_thresh: float = 0.5
    robot_static_thresh: float = 3.5
    
    orien_thresh: float = 0.1
    force_thresh_interval: Tuple[float, float] = (-100.0, 0.0)
    
    throw_thresh: float = z_thresh + 0.1

# def is_object_in_hand(
#     env: ManagerBasedRLEnv, 
#     x_thresh: float = THRESHOLD.x_thresh, 
#     y_thresh: float = THRESHOLD.y_thresh, 
#     z_thresh: float = THRESHOLD.z_thresh
# ) -> torch.Tensor:
#     object = env.scene["object"]
#     robot = env.scene["robot"]
    
#     object_xyz = object.data.root_pos_w[:, :3]
#     hand_xyz = robot.data.body_link_state_w[:, 10, :3]

#     x_diff = torch.abs(object_xyz[:, 0] - hand_xyz[:, 0])
#     y_diff = torch.abs(object_xyz[:, 1] - hand_xyz[:, 1])
#     z_diff = object_xyz[:, 2] - hand_xyz[:, 2]

#     is_within_x = x_diff < x_thresh
#     is_within_y = y_diff < y_thresh
#     is_within_z = z_diff < z_thresh

#     is_inhand = is_within_x & is_within_y & is_within_z
#     return is_inhand


def is_object_static(
    env: ManagerBasedRLEnv, 
    threshold: float = THRESHOLD.object_static_thresh,
    objet_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    
    object = env.scene[objet_cfg.name]
    object_vel_magnitude = torch.norm(object.data.root_lin_vel_w, dim=1) + torch.norm(object.data.root_ang_vel_w, dim=1)
    is_static = object_vel_magnitude < threshold
    return is_static

def is_robot_static(
    env: ManagerBasedRLEnv,
    threshold: float = THRESHOLD.robot_static_thresh,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    joint_vel_mean = torch.mean(robot.data.joint_vel, dim=1)
    is_static = joint_vel_mean < threshold
    return is_static

def is_orientation_aligned(
    env: ManagerBasedRLEnv, 
    threshold: float = THRESHOLD.orien_thresh
) -> torch.Tensor:
    orientation_error = env.command_manager.get_term("object_pose").metrics["orientation_error"]
    return orientation_error < threshold

def has_object_hand_contact(
    env: ManagerBasedRLEnv, 
    force_thresh_interval: Tuple[float, float] = THRESHOLD.force_thresh_interval,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:    
    robot = env.scene[robot_cfg.name]
    joint_wrench = robot.data.body_incoming_joint_wrench_b # (num_envs, num_links, 6)
    joint_force_z = joint_wrench[:, :, 2]
    palm_joint_force_z = joint_force_z[:, 11] # (num_envs,)
    
    min_thresh, max_thresh = force_thresh_interval
    has_contact = (palm_joint_force_z > min_thresh) & (palm_joint_force_z < max_thresh)
    
    return has_contact

def is_static_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    # if has_object_hand_contact(env) & is_object_static(env) & is_robot_static(env):
    #     import pdb;pdb.set_trace()
    return has_object_hand_contact(env) & is_object_static(env) & is_robot_static(env) # contact means inhand 

def is_object_ready_to_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    return is_static_and_inhand(env) & is_orientation_aligned(env)

def impossible_condition(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
