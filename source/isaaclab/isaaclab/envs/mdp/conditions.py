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

def is_object_above_hand(env: ManagerBasedRLEnv) -> torch.Tensor:
    object = env.scene["object"]
    robot = env.scene["robot"]
    object_z = object.data.root_pos_w[:, 2]
    hand_z = robot.data.body_link_state_w[:, 10, 2]
    return object_z >= hand_z


def is_object_static(env: ManagerBasedRLEnv, threshold: float = 0.0001) -> torch.Tensor:
    object = env.scene["object"]
    object_vel_magnitude = torch.norm(object.data.root_lin_vel_w, dim=1) + torch.norm(object.data.root_ang_vel_w, dim=1)
    return object_vel_magnitude < threshold


def is_orientation_aligned(env: ManagerBasedRLEnv, threshold: float = 0.1) -> torch.Tensor:
    orientation_error = env.command_manager.get_term("object_pose").metrics["orientation_error"]
    return orientation_error < threshold


def has_object_hand_contact(env: ManagerBasedRLEnv, threshold: float = 1e-5, sensor_cfg=SceneEntityCfg(name="sensor")) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = sensor.data.net_forces_w_history  # [envs, hist, bodies, 3]
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.any(is_contact > threshold, dim=1)


def is_object_inhand_and_static(env: ManagerBasedRLEnv) -> torch.Tensor:
    return is_object_above_hand(env) & is_object_static(env)


def is_object_ready_to_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    return is_object_static(env) & is_object_above_hand(env) & is_orientation_aligned(env)
