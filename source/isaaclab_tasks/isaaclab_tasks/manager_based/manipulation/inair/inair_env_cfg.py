# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_fsm_env_cfg import ManagerBasedRLFSMEnvCfg
from isaaclab.envs.mdp.conditions import THRESHOLD
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg

import isaaclab_tasks.manager_based.manipulation.inair.mdp as mdp

##
# Scene definition
##


@configclass
class InAirObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = MISSING

    # objects
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"/home/xuxuezhou/isaac-sim-assets-1/Assets/Isaac/4.5/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/sphere_physics.usd",
            # usd_path=f"/home/xuxuezhou/isaac-sim-assets-1/Assets/Isaac/4.5/Isaac/Props/Shapes/sphere_physics.usd",
            # scale=(1.75, 1.75, 1.75),
            scale=(2.25, 2.25, 2.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=100.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(0.4, -0.4,  1.2), # 2*2*2
            # pos=(0.4, -0.4,  1.0), # 3*3*3
            pos=(0.4, -0.4,  0.85), # 3*3*3 inhand
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = mdp.InAirReOrientationCommandCfg(
        asset_name="object",
        robot_name="robot",
        init_pos_offset=(0.0, 0.0, 0.0),
        update_goal_on_success=True,
        orientation_success_threshold=THRESHOLD.orien_thresh,
        object_vel_success_threshold=THRESHOLD.object_static_thresh,
        joint_vel_success_threshold=THRESHOLD.robot_static_thresh,
        make_quat_unique=True,
        marker_pos_offset=(0.0, 0.0, 0.5),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    arm_action = OperationalSpaceControllerActionCfg()
    hand_action = mdp.EMAJointPositionToLimitsActionCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        """Observations with full-kinematic state information.

        This does not include acceleration or force information.
        """

        # observation terms (order preserved)
        # -- robot terms
        hand_joint_pos = ObsTerm(func=mdp.hand_joint_pos_limit_normalized, noise=Gnoise(std=0.005))
        hand_joint_vel = ObsTerm(func=mdp.hand_joint_vel_rel, scale=0.2, noise=Gnoise(std=0.01))
        
        end_effector_pos = ObsTerm(func=mdp.end_effector_pos)
        end_effector_lin_vel = ObsTerm(func=mdp.end_effector_lin_vel)
        end_effector_ang_vel = ObsTerm(func=mdp.end_effector_ang_vel)
        

        # -- object terms
        object_pos = ObsTerm(
            func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )
        
        joint_wrench = ObsTerm(
            func=mdp.joint_wrench_obs,
            params={"asset_cfg": SceneEntityCfg(name="robot") },
        )

        # # -- action terms
        # last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    # observation groups
    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- object
    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (0.7, 1.3),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 250,
    #     },
    # )
    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.4, 1.6),
    #         "operation": "scale",
    #     },
    # )

    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},  
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )
    # reset_object = EventTerm(
    #     func=mdp.reset_object_above_palm,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [0.2, 0.2],  # Small random offset in x
    #             "y": [0.0, 0.0],  # Small random offset in y
    #             "z": [0.2, 0.2],    # Offset above palm between 5-10cm
    #         },
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #     },
    # )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            # "position_range": {".*": [0.2, 0.2]},
            "position_range": {".*": [0.0, 0.0]},
            # "position_range": {
            #     "^joint[1-7]$": [0.03, 0.03],
            #     "^a_.*$": [0.1, 0.1],
            # },
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    dummy_reward = RewTerm(
        func=mdp.dummy_reward,
        weight=1.0,
        params={},
    )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success, params={"num_success": 50, "command_name": "object_pose"}
    )

    # object_out_of_reach = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})
    object_on_floor = DoneTerm(func=mdp.land_on_floor)

    # object_out_of_reach = DoneTerm(
    #     func=mdp.object_away_from_goal, params={"threshold": 0.24, "command_name": "object_pose"}
    # )

##
# Environment configuration
##


@configclass
class InAirObjectEnvCfg(ManagerBasedRLFSMEnvCfg):
    """Configuration for the in air reorientation environment."""

    # Scene settings
    scene: InAirObjectSceneCfg = InAirObjectSceneCfg(num_envs=8192, env_spacing=1.0)
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
        # self.rerender_on_reset = True
