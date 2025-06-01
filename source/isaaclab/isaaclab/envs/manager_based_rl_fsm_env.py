from collections.abc import Sequence
from typing import Any, Callable
import torch

from isaaclab.envs.common import VecEnvObs
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.inair.mdp.rewards import (
    grasp_object, open_fingertips, object_vel_penalty, palm_drop_penalty, above_palm, success_bonus,
    track_delta_orientation_l2, track_orientation_inv_l2, track_object_l2, undesired_forces, track_delta_object_pos,
    desired_contact, transition_reward
)
from isaaclab.envs.mdp.rewards import joint_vel_l2, arm_joint_vel_l2, body_vel_l2

from .manager_based_rl_env import ManagerBasedRLEnv
from .manager_based_rl_fsm_env_cfg import ManagerBasedRLFSMEnvCfg
from isaaclab.envs.mdp.conditions import has_object_hand_contact, impossible_condition, is_object_ready_to_end, is_static_and_inhand

class FSMState:
    INIT = 0
    BEFORE_THROW = 1
    IN_AIR = 2
    BACK_IN_HAND = 3
    END = 4


class RewardTerm:
    def __init__(self, name: str, func: Callable, weight: float):
        self.name = name
        self.func = func
        self.weight = weight

    def compute(self, env) -> torch.Tensor:
        return self.weight * self.func(env)


FSM_REWARD_TERMS = {
    FSMState.INIT: [
        RewardTerm("object_vel_penalty", object_vel_penalty, -0.1),
        RewardTerm("arm_joint_vel_l2", lambda env: arm_joint_vel_l2(env, SceneEntityCfg("robot")), -1e-2),
        RewardTerm("joint_vel_l2", lambda env: joint_vel_l2(env, SceneEntityCfg("robot")), -1e-3),
        RewardTerm("body_vel_l2", lambda env: body_vel_l2(env, SceneEntityCfg("robot")), -1e-2),
        RewardTerm("desired_contact", desired_contact, 50), 
        RewardTerm("above_palm", above_palm, 50.0),
        # RewardTerm("track_delta_object_pos", lambda env: track_delta_object_pos(env, command_name="object_pose"), 500)
        
        # RewardTerm("open_fingertips", open_fingertips, 0.0),
        # RewardTerm("grasp_object", grasp_object, -0.0),
        # RewardTerm("track_object_l2", lambda env: track_object_l2(env, SceneEntityCfg("robot"), SceneEntityCfg("object")), -50.0),
        # RewardTerm("palm_drop_penalty", lambda env: palm_drop_penalty(env, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot")), -0.0),
    ],
    FSMState.BEFORE_THROW: [
        RewardTerm("undesired_forces", undesired_forces, -10.0),
        # RewardTerm("track_object_l2", lambda env: track_object_l2(env, SceneEntityCfg("robot"), SceneEntityCfg("object")), -5.0),
        RewardTerm("track_delta_object_pos", lambda env: track_delta_object_pos(env, command_name="object_pose"), 500),
        RewardTerm("transition_reward", lambda env: transition_reward(env), 100),
    ],
    FSMState.IN_AIR: [
        RewardTerm("above_palm", above_palm, 100.0),
        # RewardTerm("open_fingertips", open_fingertips, 100.0),
        RewardTerm("track_object_l2", lambda env: track_object_l2(env, SceneEntityCfg("robot"), SceneEntityCfg("object")), 10.0),
        RewardTerm("track_delta_object_pos", lambda env: track_delta_object_pos(env, command_name="object_pose"), 500),
        
        # RewardTerm("palm_drop_penalty", lambda env: palm_drop_penalty(env, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot")), -40.0),
        
        RewardTerm("track_delta_orientation_l2", lambda env: track_delta_orientation_l2(env, command_name="object_pose"), 600.0),
        RewardTerm("track_orientation_inv_l2", lambda env: track_orientation_inv_l2(env, command_name="object_pose", object_cfg=SceneEntityCfg("object")), 10.0),
        RewardTerm("transition_reward", lambda env: transition_reward(env), 100),
        
    ],
    FSMState.BACK_IN_HAND: [
        RewardTerm("object_vel_penalty", object_vel_penalty, -1.0),
        RewardTerm("arm_joint_vel_l2", lambda env: arm_joint_vel_l2(env, SceneEntityCfg("robot")), -1e-3),
        RewardTerm("joint_vel_l2", lambda env: joint_vel_l2(env, SceneEntityCfg("robot")), -1e-3),
        RewardTerm("body_vel_l2", lambda env: body_vel_l2(env, SceneEntityCfg("robot")), -1e-2),
        RewardTerm("above_palm", above_palm, 50.0),
        RewardTerm("desired_contact", desired_contact, 50), 
        # RewardTerm("track_object_l2", lambda env: track_object_l2(env, SceneEntityCfg("robot"), SceneEntityCfg("object")), -50.0),
        RewardTerm("track_orientation_inv_l2", lambda env: track_orientation_inv_l2(env, command_name="object_pose", object_cfg=SceneEntityCfg("object")), 10.0),
        
        # RewardTerm("close_fingertips", open_fingertips, -0.0),
        # RewardTerm("palm_drop_penalty", lambda env: palm_drop_penalty(env, init_pos_z=0.58, robot_cfg=SceneEntityCfg("robot")), -0.0),
        
        RewardTerm("track_delta_orientation_l2", lambda env: track_delta_orientation_l2(env, command_name="object_pose"), 600.0),
        RewardTerm("track_orientation_inv_l2", lambda env: track_orientation_inv_l2(env, command_name="object_pose", object_cfg=SceneEntityCfg("object")), 10.0),
        RewardTerm("success_bonus", success_bonus, 500.0),
        RewardTerm("transition_reward", lambda env: transition_reward(env), 100),
    ],
}


class ManagerBasedRLFSMEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLFSMEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.fsm_state = torch.full((self.num_envs,), FSMState.INIT, dtype=torch.long, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.fsm_state[env_ids] = FSMState.INIT

    def step(self, action: torch.Tensor):
        obs, _, terminated, truncated, info = super().step(action)
        self.reward_buf = self.compute_fsm_rewards()
        self.log_info(info)
        self.update_fsm_state()
        return obs, self.reward_buf, terminated, truncated, info

    def compute_fsm_rewards(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        for state, terms in FSM_REWARD_TERMS.items():
            mask = (self.fsm_state == state)
            if mask.any():
                for term in terms:
                    reward += mask * term.compute(self)
        return reward

    def log_info(self, info):
        info["episode"] = {"fsm_state": self.fsm_state.clone().detach()}
        for state, terms in FSM_REWARD_TERMS.items():
            mask = (self.fsm_state == state)
            for term in terms:
                val = (mask * term.compute(self)).detach()
                info["episode"][f"{state}/{term.name}"] = val
            info["episode"][f"fsm_state_count_{state}"] = mask.sum().item()

    def update_fsm_state(self):
        cond_0_to_1 = is_static_and_inhand(self)
        # cond_0_to_1 = impossible_condition(self)
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

    def reset(self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        obs_buf, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        all_env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)

        end_mask = self.fsm_state == FSMState.END
        h = 0.0
        palm_lower_z = self.scene["robot"].data.body_link_state_w[:, 10, 2]
        object_z = self.scene["object"].data.root_pos_w[:, 2]
        below_mask = (object_z < (palm_lower_z - h))

        reset_mask = (end_mask | below_mask) & torch.isin(torch.arange(self.num_envs, device=self.device), all_env_ids)
        reset_env_ids = torch.nonzero(reset_mask).flatten()

        if len(reset_env_ids) > 0:
            super()._reset_idx(reset_env_ids)

        return obs_buf, extras
