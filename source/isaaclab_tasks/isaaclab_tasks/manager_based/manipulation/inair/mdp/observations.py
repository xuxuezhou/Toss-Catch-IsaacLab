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
from isaaclab.assets.articulation.articulation import Articulation

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

def end_effector_pos(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the end effector position in the world frame.
    
    Args:
        env: The environment object.
        robot_cfg: The configuration for the robot entity. Default is "robot".
        link_index: The index of the end effector link. Default is 6 (palm link).
        
    Returns:
        The end effector position in the world frame.
    """
    robot = env.scene[robot_cfg.name]
    link_state = robot.data.body_link_state_w[:, 6]
    end_effector_pos = link_state[:, :3]
    
    return end_effector_pos

def end_effector_lin_vel(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the end effector velocity in the world frame.
    
    Args:
        env: The environment object.
        robot_cfg: The configuration for the robot entity. Default is "robot".
        link_index: The index of the end effector link. Default is 6 (palm link).
        
    Returns:
        The end effector velocity in the world frame.
    """
    robot = env.scene[robot_cfg.name]
    link_state = robot.data.body_link_state_w[:, 6] 
    end_effector_lin_vel = link_state[:, 7:10] # [pos, quat, lin_vel, ang_vel]
    
    return end_effector_lin_vel

def end_effector_ang_vel(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the end effector velocity in the world frame.
    
    Args:
        env: The environment object.
        robot_cfg: The configuration for the robot entity. Default is "robot".
        link_index: The index of the end effector link. Default is 6 (palm link).
        
    Returns:
        The end effector velocity in the world frame.
    """
    robot = env.scene[robot_cfg.name]
    link_state = robot.data.body_link_state_w[:, 6] 
    end_effector_ang_vel = link_state[:, 10:] # [pos, quat, lin_vel, ang_vel]
    
    return end_effector_ang_vel

def hand_joint_pos_limit_normalized(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids][:,7:],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0][:,7:],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1][:,7:],
    )

def hand_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids][:,7:] - asset.data.default_joint_vel[:, asset_cfg.joint_ids][:,7:]


def contact_forces_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns contact force as observation per environment."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    filter_contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :]

    return filter_contact_forces.reshape(env.num_envs, -1)