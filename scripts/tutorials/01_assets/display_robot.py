# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a USD articulation, set its initial joint angles, and just visualize.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/01_assets/display_robot.py \\
        --usd_path ~/code/Toss-Catch-IsaacLab/source/isaaclab_assets/data/Robots/xArmLEAPHand/xarm_leap.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Visualize a USD articulation with a fixed joint pose.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--usd_path",
    type=str,
    default="~/code/Toss-Catch-IsaacLab/source/isaaclab_assets/data/Robots/xArmLEAPHand/xarm_leap.usd",
    help="Path to the robot USD file.",
)
parser.add_argument(
    "--lock",
    action="store_true",
    help="Add a light damping actuator on all joints so the pose stays fixed.",
)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def build_scene():
    """Create lighting, ground, and spawn the articulation."""
    # Basic ground and light
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/DefaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))

    # Replace joint names/angles with the ones from your USD. Values are in radians.
    joint_preset = {
        "joint1": 0.0,
        "joint2": -1.047,
        "joint3": 0.0,
        "joint4": 0.524,
        "joint5": -3.142,
        "joint6": 1.0,
        "joint7": 0.0,
        
        "a_0": 0.0, "a_1": 0.0, "a_2": 0.5, "a_3": 0.7,
        "a_4": 0.0, "a_5": 0.0, "a_6": 0.5, "a_7": 0.7,
        "a_8": 0.0, "a_9": 0.0, "a_10": 0.5, "a_11": 0.7,
        "a_12": 0.0, "a_13": 0.0, "a_14": 0.5, "a_15": 0.7,
    }

    damping = 1.0 if args_cli.lock else 0.0

    robot_cfg = ArticulationCfg(
        prim_path="/World/Origin/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.expanduser(args_cli.usd_path),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # base pose in world frame
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_preset,
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=0.0,
                velocity_limit=0.0,
                stiffness=0.0,
                damping=damping,
            )
        },
    )

    robot = Articulation(cfg=robot_cfg)
    return {"robot": robot}


def main():
    """Main entry point."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 2.5), (0.0, 0.0, 0.5))

    scene_entities = build_scene()
    robot = scene_entities["robot"]

    # Spawn prims and write the initial pose into the simulation.
    sim.reset()
    # Push the configured root + joint state into the simulator once after reset.
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

    sim_dt = sim.get_physics_dt()
    print(f"[INFO] Spawned USD: {os.path.expanduser(args_cli.usd_path)}")
    print(f"[INFO] Initial joint pose: {robot.cfg.init_state.joint_pos}")

    while simulation_app.is_running():
        sim.step()
        robot.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
