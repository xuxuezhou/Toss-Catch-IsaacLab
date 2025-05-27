# manager_based_rl_fsm_env_cfg.py
from dataclasses import dataclass, field
from isaaclab.utils import configclass
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

class ManagerBasedRLFSMEnvCfg(ManagerBasedRLEnvCfg):
  rerender_on_reset: bool = True