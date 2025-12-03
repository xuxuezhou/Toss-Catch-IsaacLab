# SPDX-License-Identifier: BSD-3-Clause
"""
OSC control demo for xArm + Leap Hand based on display_robot.py.

Runs the full CV/reach suites (from test_xarm_osc.py) plus a hand open/close check.
Gravity is enabled. PD gains on the arm are raised to avoid free-spinning wrists.

Example:
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_xarm_leap_control_demo.py --usd_path ~/code/Toss-Catch-IsaacLab/source/isaaclab_assets/data/Robots/xArmLEAPHand/xarm_leap.usd
"""

import argparse
import math
import os

import torch

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="OSC control demo for xArm + Leap Hand (arm + fingers).")
parser.add_argument("--substeps", type=int, default=10, help="physics steps per control update")
parser.add_argument("--duration", type=float, default=4.0, help="default duration per test (s)")
parser.add_argument("--cv_thresh", type=float, default=0.02, help="success threshold for mean CV error (m)")
parser.add_argument("--reach_thresh", type=float, default=0.01, help="success threshold for final reach error (m)")
parser.add_argument(
    "--suite",
    choices=["cv", "reach", "hold", "both", "all"],
    default="all",
    help="which test suite to run: constant-velocity (cv), static reach (reach), hold initial pose (hold), both (cv+reach), or all",
)
# PD tuning for the merged xArm actuator (joints 1-7)
# parser.add_argument("--xarm_stiffness", type=float, default=600.0, help="PD stiffness for xArm joints 1-7")
# parser.add_argument("--xarm_damping", type=float, default=100.0, help="PD damping for xArm joints 1-7")
parser.add_argument(
    "--usd_path",
    type=str,
    default="~/code/Toss-Catch-IsaacLab/source/isaaclab_assets/data/Robots/xArmLEAPHand/xarm_leap.usd",
    help="Path to the robot USD file.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app before any isaacsim imports
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg  # noqa: E402
from isaaclab.controllers.operational_space import OperationalSpaceController  # noqa: E402
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_assets.robots.xarm_leap import XARM_LEAP_HAND_CFG  # noqa: E402


def _color(tag: str, ok: bool) -> str:
    return f"\033[1;32m{tag}\033[0m" if ok else f"\033[1;31m{tag}\033[0m"


def build_scene():
    """Create ground, light, and the robot articulation."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/DefaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))

    robot_cfg = XARM_LEAP_HAND_CFG.replace(prim_path="/World/Origin/Robot")
    # Optional override of USD path
    robot_cfg.spawn.usd_path = os.path.expanduser(args_cli.usd_path)
    # Initial pose similar to display_robot.py but grounded
    robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint1": -1.5,
            "joint2": -0.4,
            "joint3": 0.8,
            "joint4": 1.2,
            "joint5": 3.6,
            "joint6": 1.57,
            "joint7": 0.3,
            "a_0": 0.0, "a_1": 0.0, "a_2": 0.5, "a_3": 0.7,
            "a_4": 0.0, "a_5": 0.0, "a_6": 0.5, "a_7": 0.7,
            "a_8": 0.0, "a_9": 0.0, "a_10": 0.5, "a_11": 0.7,
            "a_12": 0.0, "a_13": 0.0, "a_14": 0.5, "a_15": 0.7,
        },
    )

    # robot_cfg.actuators["xarm"] = robot_cfg.actuators["xarm"].replace(
    #     stiffness=args_cli.xarm_stiffness,
    #     damping=args_cli.xarm_damping,
    # )

    robot = Articulation(cfg=robot_cfg)
    return robot


def reset_robot(robot: Articulation):
    """Write default root/joint state into sim and clear buffers."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def build_arm_osc(robot: Articulation, device: str):
    """Create an operational-space controller for the arm joints."""
    arm_joint_ids, arm_joint_names = robot.find_joints(["^joint[1-7]$"])
    if len(arm_joint_ids) != 7:
        raise RuntimeError(f"Expected 7 arm joints, got {len(arm_joint_ids)}: {arm_joint_names}")

    ee_body_ids, ee_body_names = robot.find_bodies("link_eef")
    if len(ee_body_ids) != 1:
        raise RuntimeError(f"Expected a single ee body named 'link_eef', got {ee_body_names}")
    ee_body_id = ee_body_ids[0]

    if robot.is_fixed_base:
        jacobi_ee_body_idx = ee_body_id - 1
        jacobi_joint_idx = arm_joint_ids
    else:
        jacobi_ee_body_idx = ee_body_id
        jacobi_joint_idx = [i + 6 for i in arm_joint_ids]

    num_envs = robot.data.joint_pos.shape[0]
    num_dof = len(arm_joint_ids)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        motion_stiffness_task=(1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0),
        motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        inertial_dynamics_decoupling=True,
        gravity_compensation=True,
        nullspace_control="position",
        nullspace_stiffness=0,
    )
    osc = OperationalSpaceController(cfg=osc_cfg, num_envs=num_envs, device=device)

    jacobian_b = torch.zeros(num_envs, 6, num_dof, device=device)
    mass_matrix = torch.zeros(num_envs, num_dof, num_dof, device=device)
    gravity = torch.zeros(num_envs, num_dof, device=device)
    ee_pose_b = torch.zeros(num_envs, 7, device=device)
    ee_vel_b = torch.zeros(num_envs, 6, device=device)
    joint_pos = torch.zeros(num_envs, num_dof, device=device)
    joint_vel = torch.zeros(num_envs, num_dof, device=device)

    nullspace_joint_pos_target = torch.mean(
        robot.data.soft_joint_pos_limits[:, arm_joint_ids, :],
        dim=-1,
    )

    return {
        "osc": osc,
        "arm_joint_ids": arm_joint_ids,
        "jacobi_ee_body_idx": jacobi_ee_body_idx,
        "jacobi_joint_idx": jacobi_joint_idx,
        "ee_body_id": ee_body_id,
        "jacobian_b": jacobian_b,
        "mass_matrix": mass_matrix,
        "gravity": gravity,
        "ee_pose_b": ee_pose_b,
        "ee_vel_b": ee_vel_b,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "nullspace_joint_pos_target": nullspace_joint_pos_target,
    }


def update_arm_states(robot: Articulation, ctrl: dict):
    """Fill in arm-related tensors from the current robot state."""
    arm_joint_ids = ctrl["arm_joint_ids"]
    jacobi_ee_body_idx = ctrl["jacobi_ee_body_idx"]
    jacobi_joint_idx = ctrl["jacobi_joint_idx"]

    jacobians = robot.root_physx_view.get_jacobians()
    jacobian_w = jacobians[:, jacobi_ee_body_idx, :, jacobi_joint_idx]

    base_rot = robot.data.root_quat_w
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
    ctrl["jacobian_b"][:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
    ctrl["jacobian_b"][:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

    mass_matrices = robot.root_physx_view.get_generalized_mass_matrices()
    ctrl["mass_matrix"][:] = mass_matrices[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity_full = robot.root_physx_view.get_gravity_compensation_forces()
    ctrl["gravity"][:] = gravity_full[:, arm_joint_ids]

    ee_body_id = ctrl["ee_body_id"]
    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    ctrl["ee_pose_b"][:, :3] = ee_pos_b
    ctrl["ee_pose_b"][:, 3:] = ee_quat_b

    ee_vel_w = robot.data.body_vel_w[:, ee_body_id, :]
    root_vel_w = robot.data.root_vel_w
    rel_vel_w = ee_vel_w - root_vel_w
    ctrl["ee_vel_b"][:, 0:3] = math_utils.quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3])
    ctrl["ee_vel_b"][:, 3:6] = math_utils.quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])

    ctrl["joint_pos"][:] = robot.data.joint_pos[:, arm_joint_ids]
    ctrl["joint_vel"][:] = robot.data.joint_vel[:, arm_joint_ids]


def run_static_reach(sim, robot, arm_ctrl, hand_joint_ids, dpos_b, duration_s, substeps):
    """Move EE by dpos_b (base frame) and hold."""
    sim_dt = sim.get_physics_dt()
    ctrl_dt = sim_dt * max(1, substeps)
    steps = int(math.ceil(duration_s / ctrl_dt))

    hand_target = robot.data.joint_pos[:, hand_joint_ids].clone()

    robot.update(sim_dt)
    update_arm_states(robot, arm_ctrl)
    start_pos_b = arm_ctrl["ee_pose_b"][:, :3].clone()
    target_pos_b = start_pos_b + dpos_b.to(start_pos_b.device).view(1, 3)

    err_hist = []
    for _ in range(steps):
        if not simulation_app.is_running():
            break

        robot.update(sim_dt)
        update_arm_states(robot, arm_ctrl)

        cur_pos_b = arm_ctrl["ee_pose_b"][:, :3]
        pos_err_b = target_pos_b - cur_pos_b

        command = torch.zeros(robot.data.joint_pos.shape[0], 6, device=sim.device)
        command[:, :3] = pos_err_b
        arm_ctrl["osc"].set_command(command=command, current_ee_pose_b=arm_ctrl["ee_pose_b"])

        arm_efforts = arm_ctrl["osc"].compute(
            jacobian_b=arm_ctrl["jacobian_b"],
            current_ee_pose_b=arm_ctrl["ee_pose_b"],
            current_ee_vel_b=arm_ctrl["ee_vel_b"],
            current_ee_force_b=None,
            mass_matrix=arm_ctrl["mass_matrix"],
            gravity=arm_ctrl["gravity"],
            current_joint_pos=arm_ctrl["joint_pos"],
            current_joint_vel=arm_ctrl["joint_vel"],
            nullspace_joint_pos_target=arm_ctrl["nullspace_joint_pos_target"],
        )
        robot.set_joint_effort_target(arm_efforts, joint_ids=arm_ctrl["arm_joint_ids"])
        robot.set_joint_position_target(hand_target, joint_ids=hand_joint_ids)

        for _ in range(substeps):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

        update_arm_states(robot, arm_ctrl)
        cur_pos_b = arm_ctrl["ee_pose_b"][:, :3]
        err_hist.append(torch.linalg.norm((target_pos_b - cur_pos_b)[0]).item())

    err = torch.tensor(err_hist) if err_hist else torch.zeros(0)
    final_err = float(err[-1].item()) if err.numel() > 0 else float("nan")
    mean_err = float(err.mean().item()) if err.numel() > 0 else float("nan")
    return final_err, mean_err, err.numel()


def run_constant_velocity_follow(sim, robot, arm_ctrl, hand_joint_ids, vel_b, duration_s, substeps):
    """Follow a constant EE velocity in base frame."""
    sim_dt = sim.get_physics_dt()
    ctrl_dt = sim_dt * max(1, substeps)
    steps = int(math.ceil(duration_s / ctrl_dt))

    hand_target = robot.data.joint_pos[:, hand_joint_ids].clone()

    robot.update(sim_dt)
    update_arm_states(robot, arm_ctrl)
    target_pos_b = arm_ctrl["ee_pose_b"][:, :3].clone()

    err_hist = []
    for _ in range(steps):
        if not simulation_app.is_running():
            break

        target_pos_b = target_pos_b + vel_b.to(target_pos_b.device).view(1, 3) * ctrl_dt

        robot.update(sim_dt)
        update_arm_states(robot, arm_ctrl)

        command = torch.zeros(robot.data.joint_pos.shape[0], 6, device=sim.device)
        command[:, :3] = vel_b.to(command.device).view(1, 3) * ctrl_dt
        arm_ctrl["osc"].set_command(command=command, current_ee_pose_b=arm_ctrl["ee_pose_b"])

        arm_efforts = arm_ctrl["osc"].compute(
            jacobian_b=arm_ctrl["jacobian_b"],
            current_ee_pose_b=arm_ctrl["ee_pose_b"],
            current_ee_vel_b=arm_ctrl["ee_vel_b"],
            current_ee_force_b=None,
            mass_matrix=arm_ctrl["mass_matrix"],
            gravity=arm_ctrl["gravity"],
            current_joint_pos=arm_ctrl["joint_pos"],
            current_joint_vel=arm_ctrl["joint_vel"],
            nullspace_joint_pos_target=arm_ctrl["nullspace_joint_pos_target"],
        )
        robot.set_joint_effort_target(arm_efforts, joint_ids=arm_ctrl["arm_joint_ids"])
        robot.set_joint_position_target(hand_target, joint_ids=hand_joint_ids)

        for _ in range(substeps):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

        update_arm_states(robot, arm_ctrl)
        cur_pos_b = arm_ctrl["ee_pose_b"][:, :3]
        err_hist.append(torch.linalg.norm((target_pos_b - cur_pos_b)[0]).item())

    err = torch.tensor(err_hist) if err_hist else torch.zeros(0)
    mean_err = float(err.mean().item()) if err.numel() > 0 else float("nan")
    std_err = float(err.std().item()) if err.numel() > 0 else float("nan")
    return mean_err, std_err, err.numel()


def assemble_tests():
    """Full CV, reach, and hold suites (matching test_xarm_osc.py + hold)."""
    tests: list[dict] = []
    speeds = [0.01, 0.02]
    axes = [
        (1.0, 0.0, 0.0, "x"),
        (0.0, 1.0, 0.0, "y"),
        (0.0, 0.0, 1.0, "z"),
    ]
    for s in speeds:
        for ax in axes:
            v = (s * ax[0], s * ax[1], s * ax[2])
            tests.append(
                {
                    "domain": "cv",
                    "name": f"cv_{ax[3]}_{s:.2f}",
                    "vel": v,
                    "duration": 4.0,
                    "substeps": 10,
                }
            )

    reach_offsets = [
        (0.05, 0.0, 0.0, "reach_x+5cm"),
        (-0.05, 0.0, 0.0, "reach_x-5cm"),
        (0.0, 0.05, 0.0, "reach_y+5cm"),
        (0.0, -0.05, 0.0, "reach_y-5cm"),
        (0.0, 0.0, 0.05, "reach_z+5cm"),
        (0.0, 0.0, 0.10, "reach_z+10cm"),
    ]
    for dx, dy, dz, name in reach_offsets:
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        tests.append(
            {
                "domain": "reach",
                "name": name,
                "dpos": (dx, dy, dz),
                "duration": 3.0 if d <= 0.1 else 4.0,
                "substeps": 10,
            }
        )

    # Hold initial pose
    tests.append(
        {
            "domain": "hold",
            "name": "hold_init",
            "dpos": (0.0, 0.0, 0.0),
            "duration": 4.0,
            "substeps": 10,
        }
    )

    return tests


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81), dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.0, 0.0, 2.5), (0.0, 0.0, 1.0))

    robot = build_scene()

    sim.reset()
    reset_robot(robot)
    print(f"[INFO] Spawned USD: {os.path.expanduser(args_cli.usd_path)}")

    arm_ctrl = build_arm_osc(robot, device=sim.device)
    hand_joint_ids, _ = robot.find_joints(["^a_.*$"])
    if len(hand_joint_ids) == 0:
        raise RuntimeError("No Leap hand joints (a_*) found on the robot.")

    tests = assemble_tests()
    if args_cli.suite == "cv":
        tests = [t for t in tests if t["domain"] == "cv"]
    elif args_cli.suite == "reach":
        tests = [t for t in tests if t["domain"] == "reach"]
    elif args_cli.suite == "hold":
        tests = [t for t in tests if t["domain"] == "hold"]
    elif args_cli.suite == "both":
        tests = [t for t in tests if t["domain"] in ("cv", "reach")]
    # "all" keeps everything

    results = []

    for idx, case in enumerate(tests, start=1):
        domain = case["domain"]
        name = case["name"]
        duration = float(case.get("duration", args_cli.duration))
        substeps = int(case.get("substeps", args_cli.substeps))

        print(f"\n[TEST {idx:02d}] {domain.upper()} - {name}")
        sim.reset()
        reset_robot(robot)

        if domain == "cv":
            vel = torch.tensor(case["vel"], dtype=torch.float32, device=sim.device)
            mean_err, std_err, n_steps = run_constant_velocity_follow(
                sim,
                robot,
                arm_ctrl,
                hand_joint_ids,
                vel_b=vel,
                duration_s=duration,
                substeps=substeps,
            )
            ok = mean_err <= args_cli.cv_thresh if math.isfinite(mean_err) else False
            tag = _color("PASS" if ok else "FAIL", ok)
            print(f"    {tag}  mean_err={mean_err:.4f} m (<= {args_cli.cv_thresh:.4f}), std_err={std_err:.4f} m, steps={n_steps}")
            results.append(("cv", name, mean_err, std_err, ok))

        elif domain == "reach":
            dpos = torch.tensor(case["dpos"], dtype=torch.float32, device=sim.device)
            final_err, mean_err, n_steps = run_static_reach(
                sim,
                robot,
                arm_ctrl,
                hand_joint_ids,
                dpos_b=dpos,
                duration_s=duration,
                substeps=substeps,
            )
            ok = final_err <= args_cli.reach_thresh if math.isfinite(final_err) else False
            tag = _color("PASS" if ok else "FAIL", ok)
            print(
                f"    {tag}  final_err={final_err:.4f} m (<= {args_cli.reach_thresh:.4f}), "
                f"mean_err={mean_err:.4f} m, steps={n_steps}"
            )
            results.append(("reach", name, mean_err, final_err, ok))

        elif domain == "hold":
            dpos = torch.tensor(case["dpos"], dtype=torch.float32, device=sim.device)
            final_err, mean_err, n_steps = run_static_reach(
                sim,
                robot,
                arm_ctrl,
                hand_joint_ids,
                dpos_b=dpos,
                duration_s=duration,
                substeps=substeps,
            )
            ok = final_err <= args_cli.reach_thresh if math.isfinite(final_err) else False
            tag = _color("PASS" if ok else "FAIL", ok)
            print(
                f"    {tag}  final_err={final_err:.4f} m (<= {args_cli.reach_thresh:.4f}), "
                f"mean_err={mean_err:.4f} m, steps={n_steps}"
            )
            results.append(("hold", name, mean_err, final_err, ok))

        if not simulation_app.is_running():
            break

    # Summary
    if results:
        cv_results = [r for r in results if r[0] == "cv"]
        reach_results = [r for r in results if r[0] == "reach"]
        hold_results = [r for r in results if r[0] == "hold"]
        if cv_results:
            mean_errs = [r[2] for r in cv_results]
            pass_cnt = sum(1 for r in cv_results if r[4])
            print("===== Summary (CV) =====")
            print(
                f"Cases: {len(cv_results)} | pass: {pass_cnt}/{len(cv_results)} | "
                f"mean(err)={sum(mean_errs)/len(mean_errs):.4f} m"
            )
        if reach_results:
            final_errs = [r[3] for r in reach_results]
            pass_cnt = sum(1 for r in reach_results if r[4])
            mean_final = sum(final_errs) / len(final_errs) if final_errs else float("nan")
            print("===== Summary (Reach) =====")
            print(
                f"Cases: {len(reach_results)} | pass: {pass_cnt}/{len(reach_results)} | "
                f"mean(final_err)={mean_final:.4f} m"
            )
        if hold_results:
            final_errs = [r[3] for r in hold_results]
            pass_cnt = sum(1 for r in hold_results if r[4])
            mean_final = sum(final_errs) / len(final_errs) if final_errs else float("nan")
            print("===== Summary (Hold) =====")
            print(
                f"Cases: {len(hold_results)} | pass: {pass_cnt}/{len(hold_results)} | "
                f"mean(final_err)={mean_final:.4f} m"
            )

    # Done: close out after tests
    simulation_app.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
