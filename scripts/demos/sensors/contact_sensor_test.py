# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example of contact sensing between multiple cubes.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with a large cube and multiple smaller cubes."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Big cube (static)
    big_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BigCube",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # Static object
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
    )

    # Small cubes (dynamic)
    cube1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            activate_contact_sensors=True,
        ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    cube2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            activate_contact_sensors=True,      
        ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.05)),
    )

    cube3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            activate_contact_sensors=True,
        ),

        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.0, 0.05)),
        
    )

    # Contact sensor on big cube
    big_cube_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BigCube",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube1", "{ENV_REGEX_NS}/Cube2", "{ENV_REGEX_NS}/Cube3"],
        max_contact_data_count=8000
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Simulate physics
    while simulation_app.is_running():
        if count % 2000 == 0:
            count = 0
            # Reset small cubes to initial positions
            for cube_name in ["cube1", "cube2", "cube3"]:
                cube = scene[cube_name]
                root_state = cube.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                cube.write_root_pose_to_sim(root_state[:, :7])
                cube.write_root_velocity_to_sim(root_state[:, 7:])
            scene.reset()
            print("[INFO]: Resetting cubes...")

        # Perform simulation step
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Print contact information
        print("\n-------------------------------")
        print("Big Cube Contact Sensor Data:")
        print_sensor_contact_details(scene["big_cube_contact"])


def print_sensor_contact_details(sensor):
    """Improved debug output with multiple filter support"""
    print(f"\n--- Sensor: {sensor.cfg.prim_path} ---")
    
    if sensor.data.contact_forces_buffer is None:
        print("Buffer Status:")
        if len(sensor.cfg.filter_prim_paths_expr) == 0:
            print("  - No filters configured!")
        if sensor.cfg.max_contact_data_count <= 0:
            print("  - max_contact_data_count not set!")
        return

    # Iterate over all environments
    for env_id in range(sensor.data.contact_start_indices_buffer.shape[0]):
        print(f"\nEnvironment ID: {env_id}")
        
        body_id = 0  # Single body sensor
        
        # Check each filter
        for filter_idx, filter_prim in enumerate(sensor.cfg.filter_prim_paths_expr):
            print(f"\nFilter: {filter_prim}")
            
            start_idx = int(sensor.data.contact_start_indices_buffer[env_id, body_id, filter_idx].item())
            contact_count = int(sensor.data.contact_count_buffer[env_id, body_id, filter_idx].item())
            
            if contact_count > 0:
                print(f"Found {contact_count} contact points:")
                end_idx = min(start_idx + contact_count, len(sensor.data.contact_forces_buffer))
                
                for i in range(start_idx, end_idx):
                    print(f" Point {i - start_idx + 1}:")
                    print(f" Force: {sensor.data.contact_forces_buffer[i].cpu().numpy()} N")
                    print(f" Position: {sensor.data.contact_points_buffer[i].cpu().numpy()} m")
                    print(f" Normal: {sensor.data.contact_normals_buffer[i].cpu().numpy()}")
                    print(f" Separation: {sensor.data.contact_separation_distances_buffer[i].cpu().numpy()} m")
            else:
                print("No contacts detected.")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
    
    scene_cfg = ContactSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()