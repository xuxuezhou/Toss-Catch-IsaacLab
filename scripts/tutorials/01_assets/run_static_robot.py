# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a xArm_leap and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

##
# Pre-defined configs
##
from isaaclab_assets import XARM_LEAP_HAND_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    xarm_leap_cfg = XARM_LEAP_HAND_CFG
    xarm_leap_cfg.prim_path = "/World/Origin.*/Robot"
    xarm_leap_cfg.actuators= {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["^joint[1-7]$"],  # xArm joints
            effort_limit=0.0,
            velocity_limit=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["^a_.*$"],  # Leap hand joints like a_0 ~ a_15
            effort_limit=0.0,
            velocity_limit=0.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    }
    
    xarm_leap_cfg.init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.19, 0.50),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # xArm joints
            "joint1": -1.0,
            "joint2": -0.4,     # 让上臂抬起来一点
            "joint3": 0.8,      # 前臂自然伸出
            "joint4": 1.2,      # 手腕抬起
            "joint5": 3.0,      # 接近 180°，让掌面转向上
            "joint6": 1.57,     # 手心朝上
            "joint7": 0.3,      # 稍微偏转末端方向，避免奇异

            # hand joints
            "^a_.*$": 0.0,
        },
    )
    xarm_leap = Articulation(cfg=xarm_leap_cfg)

    # # Disable gravity
    # current_gravity_status = xarm_leap.root_physx_view.get_disable_gravities()
    # current_gravity_status = torch.ones_like(current_gravity_status)  # set to 1 means distable gravity, 0 otherwise
    # env_ids = torch.arange(len(current_gravity_status), device=current_gravity_status.device)
    # xarm_leap.root_physx_view.set_disable_gravities(current_gravity_status, env_ids)

    # return the scene information
    scene_entities = {"xarm_leap": xarm_leap}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["xarm_leap"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # set joint positions with some noise
            print(f'[INFO]: Joint positions: {robot.cfg.init_state.joint_pos}')

            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_vel_sum = torch.sum(torch.square(robot.data.joint_vel[:, :]), dim=1)
            print(f'[INFO]: Joint positions: {joint_pos}')
            print(f'[INFO]: Joint velocities: {joint_vel}')
            print(f'[INFO]: Joint velocities sum: {joint_vel_sum}')
            
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)
        # -- write data to sim
        # robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    
    # Disable gravity
    current_gravity_status = scene_entities["xarm_leap"].root_physx_view.get_disable_gravities()
    current_gravity_status = torch.ones_like(current_gravity_status)  # set to 1 means distable gravity, 0 otherwise
    env_ids = torch.arange(len(current_gravity_status), device=current_gravity_status.device)
    scene_entities["xarm_leap"].root_physx_view.set_disable_gravities(current_gravity_status, env_ids)
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
