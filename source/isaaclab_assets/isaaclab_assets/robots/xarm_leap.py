# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for the xArm + Leap Hand combined robot."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

XARM_LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/xuxuezhou/code/Toss-Catch-IsaacLab/source/isaaclab_assets/data/Robots/xArmLEAPHand/xarm_leap.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            linear_damping=10.0,
            angular_damping=10.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.19, 0.50),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # ".*": 0.0,  # Zero out all joints including arm and hand
            # xArm joints
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.8,
            "joint4": 1.2,
            "joint5": 1.57,
            "joint6": -0.5,
            "joint7": 0.3,

            # hand joints
            "^a_.*$": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["^joint[1-7]$"],  # xArm joints
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=20.0,
            damping=2.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["^a_.*$"],  # Leap hand joints like a_0 ~ a_15
            effort_limit=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.2,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of combined xArm + Leap Hand robot."""
