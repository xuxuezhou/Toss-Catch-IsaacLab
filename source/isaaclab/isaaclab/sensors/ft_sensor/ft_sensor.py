# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import convert_quat

from ..sensor_base import SensorBase
from .ft_sensor_data import ForceTorqueSensorData

if TYPE_CHECKING:
    from .ft_sensor_cfg import ForceTorqueSensorCfg


class ForceTorqueSensor(SensorBase):
    
    cfg: ForceTorqueSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: ForceTorqueSensorCfg):
        """Initializes the force torque sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

        self._data: ForceTorqueSensorData = ForceTorqueSensorData()
        self._articulation_view = None
        self._num_joints = 0

    def __str__(self) -> str:
        return (
            f"ForceTorqueSensor @ '{self.cfg.prim_path}': \n"
            f"\tnum joints   : {self.num_joints}\n"
            f"\tdevice       : {self._device}\n"
            f"\tdebug_vis    : {hasattr(self, 'ft_visualizer')}"
        )

    """
    Properties
    """

    @property
    def data(self) -> ForceTorqueSensorData:
        self._update_outdated_buffers()
        return self._data
    
    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def articulation_view(self) -> physx.ArticulationView:
        return self._articulation_view
    
    """
    Operations
    """
    
    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if self._data.joint_forces is None or self._data.joint_torques is None:
            carb.log_warn("[ForceTorqueSensor] Tried to reset before initialization")
            return
        if env_ids is None:
            env_ids = slice(None)
        self._data.joint_forces[env_ids] = 0.0
        self._data.joint_torques[env_ids] = 0.0


    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        # Create simulation view
        sim_view = physx.create_simulation_view(self._backend)
        sim_view.set_subspace_roots("/")
        articulation_path = self.cfg.prim_path.replace(".*", "*")
        self._articulation_view = sim_view.create_articulation_view(articulation_path)

        self._num_joints = self._articulation_view.num_dof // self._num_envs

        # Allocate buffers
        self._data.joint_forces = torch.zeros(self._num_envs, self._num_joints, device=self._device)
        self._data.joint_torques = torch.zeros_like(self._data.joint_forces)


    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Retrieve measured joint force/torque from simulator
        forces = self.articulation_view.get_measured_joint_forces().view(-1, self._num_joints)
        torques = self.articulation_view.get_measured_joint_efforts().view(-1, self._num_joints)

        self._data.joint_forces[env_ids] = forces[env_ids]
        self._data.joint_torques[env_ids] = torques[env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "ft_visualizer"):
                self.ft_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ft_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ft_visualizer"):
                self.ft_visualizer.set_visibility(False)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._articulation_view = None
