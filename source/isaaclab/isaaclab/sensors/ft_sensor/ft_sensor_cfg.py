# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FORCE_TORQUE_SENSOR_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .ft_sensor import ForceTorqueSensor


@configclass
class ForceTorqueSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor."""

    class_type: type = ForceTorqueSensor

    visualizer_cfg: VisualizationMarkersCfg = FORCE_TORQUE_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/ForceTorqueSensor")
    """The configuration object for the visualization markers. Defaults to FORCE_TORQUE_SENSOR_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """

