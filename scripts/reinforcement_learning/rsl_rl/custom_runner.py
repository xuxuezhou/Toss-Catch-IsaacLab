import torch
import statistics
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from enum import IntEnum


class FSMState(IntEnum):
    INIT = 0
    BEFORE_THROW = 1
    IN_AIR = 2
    BACK_IN_HAND = 3
    END = 4


class CustomOnPolicyRunner(OnPolicyRunner):
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # 调用原始的 log 记录训练相关数据
        super().log(locs, width, pad)

        fsm_state_key = "fsm_state"
        ep_infos = locs.get("ep_infos", [])
        if not ep_infos or fsm_state_key not in ep_infos[0]:
            return  # 没有 FSM 状态信息，不记录

        # [num_eps, num_envs]
        all_fsm_states = torch.cat(
            [ep[fsm_state_key].unsqueeze(0).to(self.device) for ep in ep_infos],
            dim=0,
        )
        unique_states = torch.unique(all_fsm_states)

        reward_keys = ep_infos[0].keys()
        for state_id in unique_states:
            state_id = int(state_id)
            try:
                state_name = FSMState(state_id).name  # 例如 "IN_AIR"
            except ValueError:
                continue  # 忽略未知状态
            prefix = f"{state_name}/"

            for key in reward_keys:
                if key.startswith(prefix):
                    values = []
                    for ep in ep_infos:
                        if key in ep:
                            val = ep[key]  # shape: [num_envs]
                            if isinstance(val, torch.Tensor):
                                mask = (ep[fsm_state_key] == state_id)
                                masked_val = val[mask]
                                if masked_val.numel() > 0:
                                    values.append(masked_val.mean().item())
                            else:
                                values.append(float(val))  # fallback
                    if values:
                        avg_value = sum(values) / len(values)
                        self.writer.add_scalar(f"FSM/{key}", avg_value, locs["it"])
