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
    x_thresh: float = 0.10
    y_thresh: float = 0.10
    z_thresh: float = 0.20
    
    static_thresh: float = 0.5
    
    orien_thresh: float = 0.1
    contact_thresh: float = 0.01
    
    throw_thresh: float = z_thresh + 0.1

def is_object_in_hand(env: ManagerBasedRLEnv, x_thresh: float = THRESHOLD.x_thresh, y_thresh: float = THRESHOLD.y_thresh, z_thresh: float = THRESHOLD.z_thresh) -> torch.Tensor:
    object = env.scene["object"]
    robot = env.scene["robot"]
    
    object_xyz = object.data.root_pos_w[:, :3]
    hand_xyz = robot.data.body_link_state_w[:, 10, :3]

    x_diff = torch.abs(object_xyz[:, 0] - hand_xyz[:, 0])
    y_diff = torch.abs(object_xyz[:, 1] - hand_xyz[:, 1])
    z_diff = object_xyz[:, 2] - hand_xyz[:, 2]
    
    # print(f"The x difference is {x_diff}")
    # print(f"The y difference is {y_diff}")    
    # print(f"The z difference is {z_diff}")

    is_within_x = x_diff < x_thresh
    is_within_y = y_diff < y_thresh
    is_within_z = z_diff < z_thresh

    # import pdb;pdb.set_trace()
    is_inhand = is_within_x & is_within_y & is_within_z
    return is_inhand


def is_object_static(env: ManagerBasedRLEnv, threshold: float = THRESHOLD.static_thresh) -> torch.Tensor:
    object = env.scene["object"]
    object_vel_magnitude = torch.norm(object.data.root_lin_vel_w, dim=1) + torch.norm(object.data.root_ang_vel_w, dim=1)
    
    is_static = object_vel_magnitude < threshold
    # print(f"The object_vel_magnitude is {object_vel_magnitude}")
    return is_static


def is_orientation_aligned(env: ManagerBasedRLEnv, threshold: float = THRESHOLD.orien_thresh) -> torch.Tensor:
    orientation_error = env.command_manager.get_term("object_pose").metrics["orientation_error"]
    return orientation_error < threshold

def has_object_hand_contact(env: ManagerBasedRLEnv, threshold: float = THRESHOLD.orien_thresh, sensor_cfg=SceneEntityCfg(name="sensor")) -> torch.Tensor:    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.any(is_contact, dim=1)
    
    
def is_object_off_hand(env: ManagerBasedRLEnv, throw_thresh: float = THRESHOLD.throw_thresh) -> torch.Tensor:
    object = env.scene["object"]
    robot = env.scene["robot"]
    
    object_z = object.data.root_pos_w[:, 2]
    hand_z = robot.data.body_link_state_w[:, 10, 2]
    
    is_off_hand = object_z - hand_z >= throw_thresh
    return is_off_hand

def is_static_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    # if is_object_in_hand(env) & is_object_static(env):
    #     object = env.scene["object"]
    #     object_z = object.data.root_pos_w[:, 2]
    #     object_vel_magnitude = torch.norm(object.data.root_lin_vel_w, dim=1) + torch.norm(object.data.root_ang_vel_w, dim=1)
    #     print(f"Object height is {object_z}")
    #     print(f"Object velocity is {object_vel_magnitude}")
    #     import pdb;pdb.set_trace()
    return is_object_in_hand(env) & is_object_static(env)

def has_contact_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    return has_object_hand_contact(env) & is_object_in_hand(env)

def no_contact_and_inhand(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (~has_object_hand_contact(env)) & is_object_in_hand(env)

def is_object_ready_to_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    return is_static_and_inhand(env) & is_orientation_aligned(env)

def impossible_condition(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
