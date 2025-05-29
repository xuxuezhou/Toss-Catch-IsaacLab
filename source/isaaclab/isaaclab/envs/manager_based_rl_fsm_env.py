from collections.abc import Sequence
from typing import Any
import torch

from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.mdp.conditions import has_object_hand_contact, impossible_condition, is_object_ready_to_end, is_static_and_inhand
from isaaclab.envs.mdp.rewards import hand_action_l2, arm_action_l2, joint_acc_l2, joint_vel_l2, undesired_contacts
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.inair.mdp.rewards import grasp_object, open_fingertips, object_vel_penalty, palm_drop_penalty, above_palm, success_bonus, \
track_delta_orientation_l2, track_orientation_inv_l2, track_object_l2, undesired_forces
from .manager_based_rl_env import ManagerBasedRLEnv
from .manager_based_rl_fsm_env_cfg import ManagerBasedRLFSMEnvCfg


class FSMState:
    INIT = 0
    BEFORE_THROW = 1
    IN_AIR = 2
    BACK_IN_HAND = 3
    END = 4


class FSMRewardScales_0:
    """
    INIT STATE
    """
    # velocity penalty
    object_vel_penalty: float = -1.0
    joint_vel_l2: float = -1e-3
    
    # hold object (with palm)
    above_palm: float = 150.0
    open_fingertips: float = 100.0
    grasp_object: float = -0.0 
    track_object_l2: float = -50.0
    
    # maintain height
    palm_drop_penalty: float = -0.0
    
    
class FSMRewardScales_1:  
    """
    BEFORE THROW STATE
    """
    # force penalty  
    undesired_forces: float = -10.0
    # distance penalty
    track_object_l2: float = -5.0
    
    
class FSMRewardScales_2:
    """
    IN AIR STATE
    """  
    # hold object (with palm)
    above_palm: float = 100.0
    track_object_l2: float = -0.0
    open_fingertips: float = 100.0
    
    # maintain height
    palm_drop_penalty: float = -40.0
    
    # orientation improve
    track_delta_orientation_l2: float = 600.0
    track_orientation_inv_l2: float = 10.0

class FSMRewardScales_3:
    """
    BACK IN HAND STATE
    """ 
    # velocity penalty
    object_vel_penalty: float = -1.0
    joint_vel_l2: float = -1e-3
    
    # hold object (with palm)
    above_palm: float = 100.0
    track_object_l2: float = -50.0
    close_fingertips: float = -100.0
    
    # maintain height
    palm_drop_penalty: float = -40.0
    
    # orientation improve
    track_delta_orientation_l2: float = 600.0
    track_orientation_inv_l2: float = 10.0
    
    # success
    success_bonus: float = 500

    
class ManagerBasedRLFSMEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with FSM reward logic."""

    def __init__(self, cfg: ManagerBasedRLFSMEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.fsm_state = torch.full((self.num_envs,), FSMState.INIT, dtype=torch.long, device=self.device)

    def _reset_idx(self, env_ids):
        """Reset FSM state in addition to default resets."""
        super()._reset_idx(env_ids)
        self.fsm_state[env_ids] = FSMState.INIT  # Reset to INIT (0)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.reward_buf = self.compute_fsm_rewards()
        self.log_info(info)
        self.update_fsm_state()
        return obs, self.reward_buf, terminated, truncated, info
    
    def log_info(self, info):
        
        obj_vel = object_vel_penalty(self)
        joint_vel = joint_vel_l2(self, asset_cfg=SceneEntityCfg(name="robot"))
        above = above_palm(self)
        palm_drop = palm_drop_penalty(self, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot"))
        
        track_delta_ori = track_delta_orientation_l2(self, command_name="object_pose")
        track_ori = track_orientation_inv_l2(self, command_name="object_pose", object_cfg=SceneEntityCfg("object"))
        track_object = track_object_l2(self, robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object"))
        fingertip_dis = open_fingertips(self)
        grasp = grasp_object(self)
        undesired = undesired_forces(self)
        
        success = success_bonus(self)

        info["episode"] = {
            "fsm_state": self.fsm_state.clone().detach(),
            
        # """
        # INIT STATE
        # """
            "INIT/object_vel_penalty": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.object_vel_penalty * obj_vel)).detach(),
            "INIT/joint_vel_l2": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.joint_vel_l2 * joint_vel)).detach(),
            
            "INIT/above_palm": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.above_palm * above)).detach(),
            "INIT/open_fingertips": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.open_fingertips * fingertip_dis)).detach(),
            "INIT/grasp_object": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.grasp_object * grasp)).detach(),
            "INIT/track_object_l2": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.track_object_l2 * track_object)).detach(),
            
            "INIT/palm_drop_penalty": ((self.fsm_state == FSMState.INIT) * (FSMRewardScales_0.palm_drop_penalty * palm_drop)).detach(),
            
            "INIT/reward_case_0": (
                (self.fsm_state == FSMState.INIT)
                * (
                    +FSMRewardScales_0.object_vel_penalty * obj_vel
                    +FSMRewardScales_0.joint_vel_l2 * joint_vel
                    +FSMRewardScales_0.track_object_l2 * track_object
                    +FSMRewardScales_0.above_palm * above
                    +FSMRewardScales_0.open_fingertips * fingertip_dis
                    +FSMRewardScales_0.grasp_object * grasp
                    +FSMRewardScales_0.palm_drop_penalty * palm_drop
                )
            ).detach(),
            
            
        # """
        # BEFORE THROW STATE
        # """
            "BEFORE_THROW/undesired_contacts": ((self.fsm_state == FSMState.BEFORE_THROW) * (FSMRewardScales_1.undesired_forces * undesired)).detach(),
            "BEFORE_THROW/track_object_l2": ((self.fsm_state == FSMState.BEFORE_THROW) * (FSMRewardScales_1.track_object_l2 * track_object)).detach(),
            
            "BEFORE_THROW/reward_case_1": (
                (self.fsm_state == FSMState.BEFORE_THROW)
                * (
                    +FSMRewardScales_1.track_object_l2 * track_object
                    +FSMRewardScales_1.undesired_forces * undesired
                )
            ).detach(),
            
        # """
        # IN AIR STATE
        # """   
            "IN_AIR/above_palm": ((self.fsm_state == FSMState.IN_AIR) * (FSMRewardScales_2.above_palm * above)).detach(),
            "IN_AIR/open_fingertips": ((self.fsm_state == FSMState.IN_AIR) * (FSMRewardScales_2.open_fingertips * fingertip_dis)).detach(),
            "IN_AIR/track_object_l2": ((self.fsm_state == FSMState.IN_AIR) * (FSMRewardScales_2.track_object_l2 * track_object)).detach(),
            
            "IN_AIR/palm_drop_penalty": ((self.fsm_state == FSMState.IN_AIR) * (FSMRewardScales_2.palm_drop_penalty * palm_drop)).detach(),
            
            "IN_AIR/track_delta_orientation_l2": ((self.fsm_state == FSMState.IN_AIR) * (FSMRewardScales_2.track_delta_orientation_l2 * track_delta_ori)).detach(),
            "IN_AIR/track_orientation_inv_l2": ((self.fsm_state == FSMState.IN_AIR) * (FSMRewardScales_2.track_orientation_inv_l2 * track_ori)).detach(),
            
            "IN_AIR/reward_case_2": (
                (self.fsm_state == FSMState.IN_AIR) 
                * (
                    +FSMRewardScales_2.above_palm * above
                    +FSMRewardScales_2.palm_drop_penalty * palm_drop
                    +FSMRewardScales_2.track_object_l2 * track_object
                    +FSMRewardScales_2.track_delta_orientation_l2 * track_delta_ori
                    +FSMRewardScales_2.track_orientation_inv_l2 * track_ori
                    +FSMRewardScales_2.open_fingertips * fingertip_dis
                )
            ).detach(),

        # """
        # BACK IN HAND STATE
        # """   
            "BACK_IN_HAND/object_vel_penalty": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.object_vel_penalty * obj_vel)).detach(),
            "BACK_IN_HAND/joint_vel_l2": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.joint_vel_l2 * joint_vel)).detach(),
            
            "BACK_IN_HAND/above_palm": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.above_palm * above)).detach(),
            "BACK_IN_HAND/track_object_l2": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.track_object_l2 * track_object)).detach(),
            "BACK_IN_HAND/close_fingertips": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.close_fingertips * fingertip_dis)).detach(),
            
            "BACK_IN_HAND/palm_drop_penalty": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.palm_drop_penalty * palm_drop)).detach(),
            
            "BACK_IN_HAND/track_delta_orientation_l2": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.track_delta_orientation_l2 * track_delta_ori)).detach(),
            "BACK_IN_HAND/track_orientation_inv_l2": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.track_orientation_inv_l2 * track_ori)).detach(),
            "BACK_IN_HAND/success_bonus": ((self.fsm_state == FSMState.BACK_IN_HAND) * (FSMRewardScales_3.success_bonus * success)).detach(),
            
            "BACK_IN_HAND/reward_case_3": (
                (self.fsm_state == FSMState.BACK_IN_HAND)
                * (
                    +FSMRewardScales_3.object_vel_penalty * obj_vel
                    +FSMRewardScales_3.joint_vel_l2 * joint_vel
                    +FSMRewardScales_3.track_object_l2 * track_object
                    +FSMRewardScales_3.above_palm * above
                    +FSMRewardScales_3.palm_drop_penalty * palm_drop
                    +FSMRewardScales_3.track_delta_orientation_l2 * track_delta_ori
                    +FSMRewardScales_3.track_orientation_inv_l2 * track_ori
                    +FSMRewardScales_3.success_bonus * success
                    +FSMRewardScales_3.close_fingertips * fingertip_dis
                )
            ).detach(),
        }
        
        for name, state in [
            ("INIT", FSMState.INIT),
            ("BEFORE_THROW", FSMState.BEFORE_THROW),
            ("IN_AIR", FSMState.IN_AIR),
            ("BACK_IN_HAND", FSMState.BACK_IN_HAND),
            ("END", FSMState.END),
        ]:
            info["episode"][f"fsm_state_count_{name}"] = (self.fsm_state == state).sum().item()


    def compute_fsm_rewards(self):
        """Compute reward based on FSM state."""

        reward_case_0 = (
            +FSMRewardScales_0.object_vel_penalty * object_vel_penalty(self)
            +FSMRewardScales_0.joint_vel_l2 * joint_vel_l2(self, asset_cfg=SceneEntityCfg(name="robot"))
            +FSMRewardScales_0.track_object_l2 * track_object_l2(self, robot_cfg=SceneEntityCfg(name="robot"), object_cfg=SceneEntityCfg("object"))
            +FSMRewardScales_0.above_palm * above_palm(self)

            +FSMRewardScales_0.palm_drop_penalty * palm_drop_penalty(self, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot"))
            
            +FSMRewardScales_0.open_fingertips * open_fingertips(self)
            +FSMRewardScales_0.grasp_object * grasp_object(self)
        )
        reward_case_1 = (
            +FSMRewardScales_1.track_object_l2 * track_object_l2(self, robot_cfg=SceneEntityCfg(name="robot"), object_cfg=SceneEntityCfg("object"))
            +FSMRewardScales_1.undesired_forces * undesired_forces(self)
        )
        reward_case_2 = (
            +FSMRewardScales_2.above_palm * above_palm(self)
            +FSMRewardScales_2.open_fingertips * open_fingertips(self)
            +FSMRewardScales_2.track_object_l2 * track_object_l2(self, robot_cfg=SceneEntityCfg(name="robot"), object_cfg=SceneEntityCfg("object"))
            
            +FSMRewardScales_2.palm_drop_penalty * palm_drop_penalty(self, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot"))
            
            +FSMRewardScales_2.track_delta_orientation_l2 * track_delta_orientation_l2(self, command_name="object_pose")
            +FSMRewardScales_2.track_orientation_inv_l2 * track_orientation_inv_l2(self, command_name="object_pose", object_cfg=SceneEntityCfg("object"))
        )
        reward_case_3 = (
            +FSMRewardScales_3.object_vel_penalty * object_vel_penalty(self)
            +FSMRewardScales_3.joint_vel_l2 * joint_vel_l2(self, asset_cfg=SceneEntityCfg(name="robot"))
            
            +FSMRewardScales_3.above_palm * above_palm(self)
            +FSMRewardScales_3.track_object_l2 * track_object_l2(self, robot_cfg=SceneEntityCfg(name="robot"), object_cfg=SceneEntityCfg("object"))
            +FSMRewardScales_3.close_fingertips * open_fingertips(self)
            
            +FSMRewardScales_3.palm_drop_penalty * palm_drop_penalty(self, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot"))
            
            +FSMRewardScales_3.track_delta_orientation_l2 * track_delta_orientation_l2(self, command_name="object_pose")
            +FSMRewardScales_3.track_orientation_inv_l2 * track_orientation_inv_l2(self, command_name="object_pose", object_cfg=SceneEntityCfg("object"))
            +FSMRewardScales_3.success_bonus * success_bonus(self)
        )

        reward = (
            (self.fsm_state == FSMState.INIT) * reward_case_0 +
            (self.fsm_state == FSMState.BEFORE_THROW) * reward_case_1 +
            (self.fsm_state == FSMState.IN_AIR) * reward_case_2 +
            (self.fsm_state == FSMState.BACK_IN_HAND) * reward_case_3
        )
        return reward

    def update_fsm_state(self):
        """Update FSM state based on transition conditions."""
        # cond_0_to_1 = is_static_and_inhand(self)
        cond_0_to_1 = impossible_condition(self)
        cond_1_to_2 = ~has_object_hand_contact(self)
        cond_1_to_end = is_object_ready_to_end(self)
        cond_2_to_3 = has_object_hand_contact(self)
        cond_3_to_1 = is_static_and_inhand(self)
        cond_3_to_end = is_object_ready_to_end(self)

        self.fsm_state = torch.where((self.fsm_state == FSMState.INIT) & cond_0_to_1, FSMState.BEFORE_THROW, self.fsm_state)
        self.fsm_state = torch.where((self.fsm_state == FSMState.BEFORE_THROW) & cond_1_to_2, FSMState.IN_AIR, self.fsm_state)
        self.fsm_state = torch.where((self.fsm_state == FSMState.BEFORE_THROW) & cond_1_to_end, FSMState.END, self.fsm_state)
        self.fsm_state = torch.where((self.fsm_state == FSMState.IN_AIR) & cond_2_to_3, FSMState.BACK_IN_HAND, self.fsm_state)
        self.fsm_state = torch.where((self.fsm_state == FSMState.BACK_IN_HAND) & cond_3_to_1, FSMState.BEFORE_THROW, self.fsm_state)
        self.fsm_state = torch.where((self.fsm_state == FSMState.BACK_IN_HAND) & cond_3_to_end, FSMState.END, self.fsm_state)


    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """
        Resets the specified environments and additionally resets those
        in END state or where the object is under the palm.
        """
        obs_buf, extras = super().reset(seed=seed, env_ids=env_ids, options=options)

        all_env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)

        # Reset Condition 1: fsm_state == END
        end_mask = self.fsm_state == FSMState.END

        # Reset Condition 2: object_z < palm_lower_z - h
        h = 0.0
        palm_lower_z = self.scene["robot"].data.body_link_state_w[:, 10, 2]
        object_z = self.scene["object"].data.root_pos_w[:, 2]
        below_mask = (object_z < (palm_lower_z - h))

        reset_mask = (end_mask | below_mask) & torch.isin(torch.arange(self.num_envs, device=self.device), all_env_ids)
        reset_env_ids = torch.nonzero(reset_mask).flatten()

        if len(reset_env_ids) > 0:
            super()._reset_idx(reset_env_ids)

        return obs_buf, extras
