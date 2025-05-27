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
    

"""
Transition Conditions

"""

class THRESHOLD:
    x_thresh: float = 0.04
    y_thresh: float = 0.06
    static_thresh: float = 0.50
    
    orien_thresh: float = 0.1
    contact_thresh: float = 0.01
    

def is_object_in_hand(env: ManagerBasedRLEnv, x_thresh: float = THRESHOLD.x_thresh, y_thresh: float = THRESHOLD.y_thresh) -> torch.Tensor:
    object = env.scene["object"]
    robot = env.scene["robot"]
    
    # Check if the object is above hand
    object_z = object.data.root_pos_w[:, 2]
    hand_z = robot.data.body_link_state_w[:, 10, 2]
    
    is_above_hand = object_z >= hand_z
    
    # Check if the object is within xy-plane in hand
    object_xy = object.data.root_pos_w[:, :2]       # shape (num_envs, 2)
    hand_xy = robot.data.body_link_state_w[:, 10, :2]

    x_diff = torch.abs(object_xy[:, 0] - hand_xy[:, 0])
    y_diff = torch.abs(object_xy[:, 1] - hand_xy[:, 1])

    is_within_x = x_diff < x_thresh
    is_within_y = y_diff < y_thresh

    is_within_xy = is_within_x & is_within_y
    return is_within_xy & is_above_hand


def is_object_static(env: ManagerBasedRLEnv, threshold: float = THRESHOLD.static_thresh) -> torch.Tensor:
    object = env.scene["object"]
    object_vel_magnitude = torch.norm(object.data.root_lin_vel_w, dim=1) + torch.norm(object.data.root_ang_vel_w, dim=1)
    
    is_static = object_vel_magnitude < threshold
    return is_static


def is_orientation_aligned(env: ManagerBasedRLEnv, threshold: float = THRESHOLD.orien_thresh) -> torch.Tensor:
    orientation_error = env.command_manager.get_term("object_pose").metrics["orientation_error"]
    return orientation_error < threshold

def has_object_hand_contact(env: ManagerBasedRLEnv, threshold: float = THRESHOLD.orien_thresh, sensor_cfg=SceneEntityCfg(name="sensor")) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    current_contact_forces = sensor.data.net_forces_w
    is_contact = torch.max(torch.norm(current_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact
    
    # filtered_contact_forces = sensor.data.force_matrix_w
    # is_contact = torch.max(torch.norm(filtered_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # return torch.any(is_contact, dim=1)

def is_static_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    return is_object_in_hand(env) & is_object_static(env) & has_object_hand_contact(env)

def has_contact_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    return has_object_hand_contact(env) & is_object_in_hand(env)

def no_contact_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (~has_object_hand_contact(env)) & is_object_in_hand(env)

def is_object_ready_to_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    return is_static_and_inhand(env) & is_orientation_aligned(env)

def impossible_condition(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
