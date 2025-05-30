# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for 3D orientation goals for objects."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.assets.articulation.articulation import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import InAirReOrientationCommandCfg


class InAirReOrientationCommand(CommandTerm):
    """Command term that generates 3D pose commands for in-air manipulation task.

    This command term generates 3D orientation commands for the object. The orientation commands
    are sampled uniformly from the 3D orientation space. The position commands are the default
    root state of the object.

    Unlike typical command terms, where the goals are resampled based on time, this command term
    does not resample the goals based on time. Instead, the goals are resampled when the object
    reaches the goal orientation. The goal orientation is considered to be reached when the
    orientation error is below a certain threshold.
    """

    cfg: InAirReOrientationCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: InAirReOrientationCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]
        self.robot: Articulation = env.scene[cfg.robot_name]

        # create buffers to store the command
        # -- command: (x, y, z)
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins
        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["previous_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["previous_object_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        
        
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_velocity_magnitude"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_velocity_magnitude"] = torch.zeros(self.num_envs, device=self.device)
        

    def __str__(self) -> str:
        msg = "InAirManipulationCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- store the previous orientation error before updating
        self.metrics["previous_orientation_error"] = self.metrics["orientation_error"].clone()
        self.metrics["previous_object_pos_error"] = self.metrics["object_pos_error"].clone()
        
        # -- compute the orientation error
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        # -- compute the pos distance error
        self.metrics["object_pos_error"] = torch.norm(self.robot.data.body_link_state_w[:, 11, :3] - self.object.data.root_pos_w[:, :3])
        
        # -- compute object velocity magnitude (both linear and angular)
        lin_vel = self.object.data.root_lin_vel_w
        ang_vel = self.object.data.root_ang_vel_w
        self.metrics["object_velocity_magnitude"] = torch.norm(lin_vel, dim=1) + torch.norm(ang_vel, dim=1)
        
        # -- compute joint velocity magnitude
        joint_vel = self.robot.data.joint_vel
        self.metrics["joint_velocity_magnitude"] = torch.norm(joint_vel, dim=1)
        
        # -- compute the number of consecutive successes
        # Check all conditions: orientation error, velocity, and acceleration
        successes = self._check_success()
        
        # All conditions must be met for success
        self.metrics["consecutive_success"] += successes.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new orientation targets
        rand_floats = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        # rotate randomly about x-axis and then y-axis
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(rand_floats[:, 0] * torch.pi, self._X_UNIT_VEC[env_ids]),
            math_utils.quat_from_angle_axis(rand_floats[:, 1] * torch.pi, self._Y_UNIT_VEC[env_ids]),
        )
        # make sure the quaternion real-part is always positive
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            goal_resets = self._check_success()
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(goal_reset_ids)
    
    def _check_success(self):
        # Condition 1: check orientation error
        orientation_success = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        
        # Condition 2: check stability
        object_vel_success = self.metrics["object_velocity_magnitude"] < self.cfg.object_vel_success_threshold  # threshold for velocity
        joint_vel_success = self.metrics["joint_velocity_magnitude"] < self.cfg.joint_vel_success_threshold  # threshold for velocity
        
        goal_resets = orientation_success & object_vel_success & joint_vel_success
        return goal_resets

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat)
