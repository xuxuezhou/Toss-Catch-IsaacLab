# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

        # switch robot to leap hand
        self.scene.robot = XARM_LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Remove stiffness and damping for effort control
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0
        
        # self.scene.robot.init_state.joint_pos = {
        #     # xArm joints
        #     "joint1": 0.0,
        #     "joint2": 0.0,
        #     "joint3": 0.0,
        #     "joint4": 1.57,
        #     "joint5": 3.14, 
        #     "joint6": 1.57,
        #     "joint7": 0.0,
            
        #     # hand joints
        #     "^a_.*$": 0.0,
        # }
        
        # self.scene.robot.init_state.joint_pos = {
        #     # xArm joints
        #     "joint1": 0.0,
        #     "joint2": -0.3,
        #     "joint3": 0.8,
        #     "joint4": 1.2,
        #     "joint5": 1.57,
        #     "joint6": -0.5,
        #     "joint7": 0.3,

        #     # hand joints
        #     "^a_.*$": 0.0,
        # }
        
        # Disable automatic movements
        self.scene.robot.init_state.joint_vel = {".*": 0.0}  # Set all joint velocities to zero

        # Configure the operational space controller for the arm
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["^joint[1-7]$"],  # xArm joints
            body_name="link7",  # End effector
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 200.0),
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="center",
            position_scale=1.0,
            orientation_scale=1.0,
            stiffness_scale=100.0,
        )
        
        # Add a separate action for controlling the hand fingers
        self.actions.hand_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["^a_.*$"],  # Leap hand joints
            # joint_names=[".*"],  # Leap hand joints
            alpha=0.95,
            rescale_to_limits=True,
        )

        # Remove joint position and velocity observations as they are not needed for OSC
        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None


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
        self.terminations.time_out = None


##
# Environment configuration with no velocity observations.
##


@configclass
class xArmLeapCubeNoVelObsEnvCfg(xArmLeapCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch observation group to no velocity group
        self.observations.policy = inair_env_cfg.ObservationsCfg.NoVelocityKinematicObsGroupCfg()


@configclass
class xArmLeapCubeNoVelObsEnvCfg_PLAY(xArmLeapCubeNoVelObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
