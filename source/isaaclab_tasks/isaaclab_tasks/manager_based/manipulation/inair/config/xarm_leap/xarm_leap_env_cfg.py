# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

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
        
        self.scene.robot.init_state.joint_pos = {
            # xArm joints
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 1.57,
            "joint5": 3.14, 
            "joint6": 1.57,
            "joint7": 0.0,
            
            # hand joints
            "^a_.*$": 0.0,
        }
        
        # Disable automatic movements
        self.scene.robot.init_state.joint_vel = {".*": 0.0}  # Set all joint velocities to zero



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
