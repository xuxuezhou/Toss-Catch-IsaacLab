# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
import isaaclab_tasks.manager_based.manipulation.inair.mdp as mdp

import isaaclab_tasks.manager_based.manipulation.inair.inair_env_cfg as inair_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets import XARM_LEAP_HAND_CFG  # isort: skip


@configclass
class xArmLeapCubeEnvCfg(inair_env_cfg.InAirObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = XARM_LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            # xArm joints
            "joint1": -1.5,
            "joint2": -0.4,     # 让上臂抬起来一点  
            "joint3": 0.8,      # 前臂自然伸出
            "joint4": 1.2,      # 手腕抬起
            "joint5": 3.6,      # 接近 180°，让掌面转向上
            "joint6": 1.57,     # 手心朝上
            "joint7": 0.3,      # 稍微偏转末端方向，避免奇异

            # hand joints
            "^a_.*$": 0.0,
        }

        # Configure the operational space controller for the arm 
        # dim: rel (6) (+ stiffness (6) + damping (6))
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["^joint[1-7]$"],  # xArm joints
            body_name="link_eef",  # End effector
            controller_cfg=OperationalSpaceControllerCfg(
                # target_types=["pose_rel", "wrench_abs"], # dim: 6 + 6
                target_types=["pose_rel"], # dim: 6 
                
                impedance_mode="fixed",
                motion_stiffness_task=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                
                # contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                # motion_control_axes_task=[1, 0, 1, 1, 1, 1],
                # contact_wrench_control_axes_task=[0, 1, 0, 0, 0, 0],
                
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                nullspace_control="position",
                nullspace_stiffness=1.0,
            ),
            nullspace_joint_pos_target="center",
            position_scale=1.0,
            orientation_scale=1.0,
            # stiffness_scale=100.0,
        )
        
        # Add a separate action for controlling the hand fingers 
        # dim: 16 joints
        self.actions.hand_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["^a_.*$"],  # Leap hand joints
            # joint_names=[".*"],  # Leap hand joints
            alpha=0.95,
            rescale_to_limits=True,
        )
        
        # contact sensor
        self.scene.sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*fingertip.*",
            # prim_path="{ENV_REGEX_NS}/Robot/palm_lower",
            # prim_path="{ENV_REGEX_NS}/Robot/.*mcp_joint.*",
            # prim_path="{ENV_REGEX_NS}/Robot/(.*fingertip.*|.*mcp_joint.*)",
            # prim_path="{ENV_REGEX_NS}/Robot/(.*fingertip.*|.*pip.*|.*dip.*|.*mcp_joint.*)",
            # prim_path="{ENV_REGEX_NS}/Robot/(palm_lower|.*fingertip.*|.*pip.*|.*dip.*|.*mcp_joint.*)",
            update_period=0.0, 
            history_length=6, 
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        )


@configclass
class xArmLeapCubeEnvCfg_PLAY(xArmLeapCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        # self.terminations.time_out = None
