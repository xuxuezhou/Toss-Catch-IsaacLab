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


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InAirReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def initial_root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Asset initial root position in the environment frame.
    
    This function returns the initial position of the object when it was spawned.
    For subsequent steps, it returns zeros.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # If this is the first step (initial state), return actual position
    if env._sim_step_counter == 0:
        return asset.data.root_pos_w - env.scene.env_origins
    # Otherwise return zeros
    return torch.zeros_like(asset.data.root_pos_w)


def initial_root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Asset initial root orientation (w, x, y, z) in the environment frame.

    This function returns the initial orientation of the object when it was spawned.
    For subsequent steps, it returns identity quaternion.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # If this is the first step (initial state), return actual orientation
    if env._sim_step_counter == 0:
        quat = asset.data.root_quat_w
        # make the quaternion real-part positive if configured
        return math_utils.quat_unique(quat) if make_quat_unique else quat
    
    # Otherwise return identity quaternion
    identity_quat = torch.zeros_like(asset.data.root_quat_w)
    identity_quat[:, 0] = 1.0  # Set w component to 1 for identity quaternion
    return identity_quat
