# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for the xArm + Leap Hand combined robot."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

XARM_LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.expanduser("~/code/Toss-Catch-IsaacLab/source/isaaclab_assets/data/Robots/xArmLEAPHand/xarm_leap.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            # linear_damping=10.0,
            # angular_damping=10.0,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=64 / math.pi * 180.0,
            # max_depenetration_velocity=5.0,
            # max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.19, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,  # Zero out all joints including arm and hand
        },
    ),
    actuators={
        "xarm": ImplicitActuatorCfg(
            joint_names_expr=["^joint[1-7]$"],  # xArm joints 1-7 (merged)
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=0.0,
            damping=0.0,
        ),

        "leap_hand": ImplicitActuatorCfg(
            joint_names_expr=["^a_.*$"],  # Leap hand joints like a_0 ~ a_15
            effort_limit_sim=0.5,
            velocity_limit_sim=100.0,
            stiffness=200.0,
            damping=50.0,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of combined xArm + Leap Hand robot."""
