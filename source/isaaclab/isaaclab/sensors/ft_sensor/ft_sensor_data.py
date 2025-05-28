# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ForceTorqueSensorData:
    """Data container for the joint force-torque sensor."""

    joint_forces: torch.Tensor | None = None
    """Measured joint forces. Shape: (N, J), where N = number of environments, J = number of joints."""

    joint_torques: torch.Tensor | None = None
    """Measured joint torques. Shape: (N, J), where N = number of environments, J = number of joints."""
