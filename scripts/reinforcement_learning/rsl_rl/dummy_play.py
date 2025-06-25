# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

# Hand joints: Projection function bewteen angle and actions
# Note: here we simply choose alpha=1, for verification only
def project_actions_to_angles(env, target, alpha=1.0, fac=1.0):
    joint_pos_limits = env.unwrapped.scene["robot"].root_physx_view.get_dof_limits()[0][7:, :]
    l = joint_pos_limits[:, 0] 
    u = joint_pos_limits[:, 1]

    l_soft = (l + u) / 2 - fac * (u - l)
    u_soft = (l + u) / 2 + fac * (u - l)
    
    hand_target = target[: ,6:]
    l_soft = l_soft.to(hand_target.device)
    u_soft = u_soft.to(hand_target.device)
    
    processed_actions = (torch.clamp(hand_target, -1, 1) + 1) / 2 * (u_soft - l_soft) + l_soft
    prev_applied_actions = env.unwrapped.action_manager.prev_action[:, 6:]
    applied_actions = alpha * processed_actions + (1-alpha) * prev_applied_actions
    
    target_angle = applied_actions
    return target_angle


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    """Modify hand config parameter alpha manually
    """
    hand_action = env.unwrapped.action_manager.get_term("hand_action")
    hand_action._alpha = 1.0

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.zeros((env.num_envs, 22), device=env.unwrapped.device)
            # print(f"[INFO] actions: {actions}")
            obs, rewards, dones, info = env.step(actions)
            
            # # get entity in environment
            # robot = env.unwrapped.scene["robot"]
            # object = env.unwrapped.scene["object"]
            # contact_sensor = env.unwrapped.scene.sensors["sensor"]
            
            # # robot
            # joint_velocities = robot.data.joint_vel
            # joint_accelerates = robot.data.joint_acc
            # joint_vel_magnitude = torch.mean(torch.square(robot.data.joint_vel), dim=1)
            # default_joint_vel_magnitude = torch.mean(torch.square(robot.data.default_joint_vel), dim=1)
            # joint_wrench = robot.data.body_incoming_joint_wrench_b # (num_envs, num_links, 6)
            
            # # arm
            # link_eef_pos = robot.data.body_link_state_w[:, 9][:, :3]
            # link_7_pos = robot.data.body_link_state_w[:, 8][:, :3]
            # arm_eef_position = robot.data.body_link_state_w[:, 9][:, :3]
            # arm_eef_orientation = robot.data.body_link_state_w[:, 9][:, 3:]
            
            # # hand
            # palm_pos = robot.data.body_link_state_w[:,6][:, :3]
            # hand_joint_forces = joint_wrench[:, 11, :3]
            # hand_joint_torques = joint_wrench[:, 11, 3:]
            # finger_joint_forces_z = joint_wrench[:, -4:, 2]
            # hand_joint_forces_mean = torch.mean(torch.norm(hand_joint_forces, dim=-1), dim=-1)
            # hand_joint_torques_mean = torch.mean(torch.norm(hand_joint_torques, dim=-1), dim=-1)
            # hand_joint_pos = robot.data.joint_pos[:, 7:]
            
            # # contact sensor
            # filter_contact_forces = contact_sensor.data.force_matrix_w
            # is_contact = torch.max(torch.norm(filter_contact_forces[:, :, :], dim=-1), dim=1)[0] > 0
            

            # body_lin_vel_sum = torch.sum(torch.norm(robot.data.body_lin_vel_w[:, :, :], dim=-1), dim=1)
            # body_ang_vel_sum = torch.sum(torch.norm(robot.data.body_ang_vel_w[:, :, :], dim=-1), dim=1)
            # body_vel_l2 = body_lin_vel_sum + body_ang_vel_sum
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
