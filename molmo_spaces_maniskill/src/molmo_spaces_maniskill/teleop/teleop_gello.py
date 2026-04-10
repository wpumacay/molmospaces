#!/usr/bin/env python3
"""
GELLO Teleoperation Script for ManiSkill

This script provides teleoperation capabilities for ManiSkill robots using GELLO hardware.
Supports both single-arm and dual-arm (bimanual) robots.

Usage:
    # Single-arm robot (Panda)
    python -m mani_skill.teleop.scripts.teleop_gello -e PickCube-v1 -r panda
    
    # With auto device discovery
    python -m mani_skill.teleop.scripts.teleop_gello -e PickCube-v1 -r panda --auto-discover
    
    # Dual-arm robot
    python -m mani_skill.teleop.scripts.teleop_gello -e DualArmTask-v1 -r xlerobot

Features:
    - Automatic GELLO device discovery
    - Support for single and dual-arm robots
    - Configurable recording frequency
    - Ghost visualization for alignment
    - Only saves successful trajectories
"""

import time
from dataclasses import dataclass
from typing import Annotated, List, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm
from pathlib import Path

from mani_skill.envs.sapien_env import BaseEnv
from molmo_spaces_maniskill.teleop.configs import get_robot_config, RobotTeleopConfig
from molmo_spaces_maniskill.teleop.devices import GelloDevice, GelloDeviceManager
from molmo_spaces_maniskill.teleop.wrappers import RecordEpisodeWithFreq
from molmo_spaces_maniskill.teleop.action_mappers import get_action_mapper, BaseActionMapper

# Register custom environments
from molmo_spaces_maniskill.molmoact2.tasks import *

@dataclass
class Args:
    """Teleoperation arguments."""
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    """Environment ID"""

    obs_mode: str = "rgb"
    """Observation mode"""
    
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """Robot UID (e.g., panda, fr3_robotiq_wristcam, xlerobot)"""
    
    record_dir: str = "demos"
    """Directory to save recorded trajectories"""
    
    use_timestamp: bool = True
    """Add timestamp to output directory"""
    
    save_video: bool = True
    """Save video of teleoperation"""
    
    viewer_shader: str = "default"

    enable_shadow: bool = False
    """Viewer shader"""
    
    control_freq: float = 50.0
    """Control frequency (Hz)"""
    
    record_freq: Optional[float] = 15.0
    """Recording frequency (Hz). None = record every step"""
    
    max_episode_steps: int = 50000
    """Maximum steps per episode"""
    
    target_trajs: int = 100
    """Target number of trajectories to collect"""
    
    # GELLO settings
    gello_port: Optional[str] = None
    """GELLO port (auto-detected if None)"""
    
    gello_port_2: Optional[str] = None
    """Second GELLO port for dual-arm (auto-detected if None)"""
    
    auto_discover: bool = True
    """Automatically discover GELLO devices"""
    
    alignment_threshold: float = 0.2
    """Joint alignment threshold (radians)"""


class TeleopSession:
    """
    Manages a teleoperation session.
    
    Handles device connection, environment setup, and the main teleoperation loop.
    """
    
    def __init__(self, args: Args):
        self.args = args
        self.env: Optional[BaseEnv] = None
        self.devices: List[GelloDevice] = []
        self.action_mapper: Optional[BaseActionMapper] = None
        self.device_manager = GelloDeviceManager()
        
        # Get robot configuration
        self.robot_config = get_robot_config(args.robot_uid)
        if self.robot_config is None:
            raise ValueError(f"Unknown robot: {args.robot_uid}")
    
    def setup(self) -> None:
        """Setup devices and environment."""
        print("\n" + "=" * 60)
        print("GELLO Teleoperation Setup")
        print("=" * 60)
        
        # Setup devices
        self._setup_devices()
        
        # Setup environment
        self._setup_environment()
        
        # Setup action mapper
        self.action_mapper = get_action_mapper(self.args.robot_uid, env=self.env)
        self.action_mapper.set_env(self.env)
        
        print("\n✓ Setup complete!")
        print(f"  Robot: {self.args.robot_uid}")
        print(f"  Devices: {len(self.devices)}")
        print(f"  Control freq: {self.args.control_freq} Hz")
        print(f"  Record freq: {self.args.record_freq} Hz")
    
    def _setup_devices(self) -> None:
        """Connect to GELLO devices."""
        print("\n[1/2] Connecting to GELLO devices...")
        
        if self.args.auto_discover:
            # Print discovered devices
            self.device_manager.print_discovered_devices()
            
            # Try to auto-connect for this robot
            try:
                self.devices = self.device_manager.connect_for_robot(self.args.robot_uid)
            except ValueError as e:
                print(f"⚠️  Auto-discovery failed: {e}")
                print("Falling back to manual port specification...")
                self._connect_manual_ports()
        else:
            self._connect_manual_ports()
        
        if not self.devices:
            raise RuntimeError("No GELLO devices connected!")
        
        print(f"\n✓ Connected {len(self.devices)} device(s)")
        for i, device in enumerate(self.devices):
            print(f"  [{i}] {device.name} ({device.num_joints} joints)")
    
    def _connect_manual_ports(self) -> None:
        """Connect to manually specified ports."""
        ports = []
        if self.args.gello_port:
            ports.append(self.args.gello_port)
        if self.args.gello_port_2:
            ports.append(self.args.gello_port_2)
        
        if not ports:
            raise ValueError(
                "No GELLO ports specified. Use --gello-port or --auto-discover"
            )
        
        for port in ports:
            device = self.device_manager.connect(port)
            self.devices.append(device)
    
    def _setup_environment(self) -> None:
        """Create and configure the environment."""
        print("\n[2/2] Creating environment...")
        
        # Determine output directory
        if self.args.use_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{self.args.record_dir}/{self.args.env_id}/gello_teleop_{timestamp}/"
        else:
            output_dir = f"{self.args.record_dir}/{self.args.env_id}/gello_teleop/"
        
        env_kwargs = dict()
        # Create base environment
        env_kwargs.update(dict(
            obs_mode=self.args.obs_mode,
            control_mode=self.robot_config.control_mode,
            render_mode="rgb_array",
            reward_mode="none",
            robot_uids=self.args.robot_uid,
            enable_shadow=self.args.enable_shadow,
            viewer_camera_configs=dict(shader_pack=self.args.viewer_shader),
            max_episode_steps=self.args.max_episode_steps,
        ))
        env = gym.make(self.args.env_id, **env_kwargs)
        
        # Wrap with recording
        self.env = RecordEpisodeWithFreq(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=self.args.save_video,
            info_on_video=False,
            source_type="teleoperation",
            source_desc=f"GELLO teleoperation ({self.args.robot_uid})",
            record_freq=self.args.record_freq,
            control_freq=self.args.control_freq,
        )
        
        print(f"✓ Environment: {self.args.env_id}")
        print(f"✓ Output: {output_dir}")
    
    def run(self) -> None:
        """Run the main teleoperation loop."""
        print("\n" + "=" * 60)
        print("Starting Data Collection")
        print("=" * 60)
        
        tbar = tqdm(total=self.args.target_trajs, desc="Collecting trajectories")
        num_trajs = 0
        
        # Initial reset
        seed = torch.randint(0, 1000000, (1,)).item()
        self.env.reset(seed=seed)
        
        print(f"\nControls:")
        print("  h: Show help")
        print("  q: Quit (won't save current episode)")
        print("  r: Restart episode (won't save)")
        print("\nMove GELLO to align with robot, then start teleoperating!")
        
        while num_trajs < self.args.target_trajs:
            # Wait for alignment
            aligned = self._wait_for_alignment()
            if not aligned:
                seed = torch.randint(0, 1000, (1,)).item()
                self._clear_buffer()
                self.env.reset(seed=seed)
                continue
            
            # Run teleoperation
            result = self._teleop_episode()
            
            if result == "quit":
                self._clear_buffer()
                break
            elif result == "success":
                # Save trajectory
                print("✅ Task success! Saving...")
                if self.env.save_video:
                    self.env.flush_video()
                if self.env._trajectory_buffer is not None:
                    self.env.flush_trajectory()
                
                num_trajs += 1
                tbar.update(1)
                seed = torch.randint(0, 1000, (1,)).item()
                self.env.reset(seed=seed)
            elif result == "restart":
                self._clear_buffer()
                self.env.reset(seed=seed)
        
        tbar.close()
        self._finish()
    
    def _wait_for_alignment(self) -> bool:
        """
        Wait for GELLO devices to align with robot.
        
        Uses ghost visualization to show target positions.
        
        Returns:
            True if aligned, False if user wants to skip
        """
        import sapien.utils.viewer
        from mani_skill.utils import sapien_utils
        
        viewer = self.env.render_human()
        robot = self.env.unwrapped.agent.robot
        
        # Reset robot to keyframe position before alignment
        # This ensures consistent target joints across multiple alignment attempts
        agent = self.env.unwrapped.agent
        if hasattr(agent, 'keyframes') and 'home' in agent.keyframes:
            keyframe = agent.keyframes['home']
            robot.set_qpos(keyframe.qpos)
            robot.set_qvel(robot.get_qvel() * 0)  # Zero velocity
        
        # Get target joints for alignment (now from keyframe position)
        target_joints = self.action_mapper.get_target_joints(self.env)
        
        # Print debug info about GELLO config
        print("\n" + "=" * 60)
        print("DEBUG: GELLO Configuration")
        print("=" * 60)
        for i, device in enumerate(self.devices):
            print(f"\nDevice {i}: {device.name}")
            print(f"  Port: {device.port}")
            print(f"  Robot type: {device.robot_type}")
            print(f"  Joint IDs: {device.config.joint_ids}")
            print(f"  Joint offsets: {[f'{x:.3f}' for x in device.config.joint_offsets]}")
            print(f"  Joint signs: {device.config.joint_signs}")
            print(f"  Gripper config: {device.config.gripper_config}")
        
        print(f"\nTarget joints (robot init pose):")
        if isinstance(target_joints, list):
            for i, tj in enumerate(target_joints):
                print(f"  Arm {i}: {[f'{x:.3f}' for x in tj]}")
        else:
            print(f"  {[f'{x:.3f}' for x in target_joints]}")
        print("=" * 60)
        
        # Setup ghost visualization
        transform_window = None
        for plugin in viewer.plugins:
            if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
                transform_window = plugin
                break
        
        if transform_window:
            # Select end-effector link for ghost
            ee_link_name = self.robot_config.arms[0].ee_link_name
            ee_link = sapien_utils.get_obj_by_name(
                self.env.agent.robot.links, ee_link_name
            )
            if ee_link:
                viewer.select_entity(ee_link._objs[0].entity)
                transform_window.enabled = True
                transform_window.update_ghost_objects()
                transform_window.follow = False
                self.env.render_human()
        
        last_print_time = 0
        print_interval = 1.0  # Print every 0.1 second
        
        try:
            while True:
                # Read device states
                device_states = [d.get_state() for d in self.devices]
                
                # Update robot visualization to current GELLO state
                self._update_robot_visualization(device_states)
                
                # Check alignment
                if self.action_mapper.is_bimanual:
                    errors = self.action_mapper.compute_alignment_error(
                        device_states, target_joints
                    )
                    aligned = all(e < self.args.alignment_threshold for e in errors)
                    total_error = sum(errors)
                else:
                    error = self.action_mapper.compute_alignment_error(
                        device_states, target_joints
                    )
                    aligned = error < self.args.alignment_threshold
                    total_error = error
                
                # # Print debug info periodically
                # current_time = time.time()
                # if current_time - last_print_time >= print_interval:
                #     last_print_time = current_time
                #     print("\n--- Alignment Debug ---")
                #     for i, (device, state) in enumerate(zip(self.devices, device_states)):
                #         print(f"Device {i} ({device.name}):")
                #         print(f"  GELLO joints: {[f'{x:.3f}' for x in state.joint_positions]}")
                #         print(f"  Gripper: {state.gripper_state:.3f}")
                #         print(f"  Raw joints: {[f'{x:.3f}' for x in state.raw_joints]}")
                        
                #         if isinstance(target_joints, list):
                #             tj = target_joints[i] if i < len(target_joints) else target_joints[0]
                #         else:
                #             tj = target_joints
                        
                #         # Per-joint error
                #         num_joints = min(len(state.joint_positions), len(tj))
                #         joint_errors = [abs(state.joint_positions[j] - tj[j]) for j in range(num_joints)]
                #         print(f"  Target joints: {[f'{x:.3f}' for x in tj[:num_joints]]}")
                #         print(f"  Joint errors:  {[f'{x:.3f}' for x in joint_errors]}")
                #         print(f"  Max error: {max(joint_errors):.3f} rad ({np.degrees(max(joint_errors)):.1f}°)")
                    
                #     print(f"Total error: {total_error:.3f}, threshold: {self.args.alignment_threshold}")
                #     print(f"Aligned: {aligned}")
                
                self.env.render_human()
                time.sleep(0.05)
                
                if aligned:
                    # Debug: Print state just before alignment is confirmed
                    print("\n" + "=" * 60)
                    print("ALIGNMENT ACHIEVED - Final State:")
                    print("=" * 60)
                    robot_qpos = robot.get_qpos()
                    if hasattr(robot_qpos, 'cpu'):
                        robot_qpos = robot_qpos.cpu().numpy().flatten()
                    print(f"  Ghost (target):    {[f'{x:.3f}' for x in target_joints[0][:6]]}")
                    print(f"  GELLO joints:      {[f'{x:.3f}' for x in device_states[0].joint_positions[:6]]}")
                    print(f"  Virtual robot:     {[f'{x:.3f}' for x in robot_qpos[:6]]}")
                    print("=" * 60)
                    return True
                    
        finally:
            # Cleanup ghost
            if transform_window:
                transform_window.enabled = False
                viewer.select_entity(None)
            
            # Restore robot to initial position
            initial_qpos = robot.get_qpos()
            robot.set_qpos(initial_qpos)
        
        return True
    
    def _update_robot_visualization(self, device_states) -> None:
        """Update robot visualization to match GELLO state.
        
        Uses controller's active_joint_indices to correctly handle
        interleaved qpos (e.g., bimanual robots where joints are interleaved).
        """
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos()
        
        if hasattr(qpos, 'cpu'):
            new_qpos = qpos.cpu().numpy().flatten()
        else:
            new_qpos = np.array(qpos).flatten()
        
        # Try to get joint indices from controller (handles interleaved qpos)
        agent = self.env.unwrapped.agent
        ctrl = agent.controller
        
        # Update arm joints from devices
        for arm_idx, arm_config in enumerate(self.robot_config.arms):
            if arm_idx < len(device_states):
                device_state = device_states[arm_idx]
                num_joints = arm_config.num_joints
                gello_joints = device_state.joint_positions[:num_joints]
                
                # Try to find the arm controller by name
                arm_ctrl = None
                if hasattr(ctrl, 'controllers'):
                    # Try common naming patterns
                    possible_names = [
                        arm_config.name,  # Use config name directly (e.g., "left_arm", "right_arm", "arm")
                        f"arm_{arm_idx}",
                        "arm" if len(self.robot_config.arms) == 1 else None,
                    ]
                    for name in possible_names:
                        if name and name in ctrl.controllers:
                            arm_ctrl = ctrl.controllers[name]
                            break
                
                if arm_ctrl is not None:
                    # Use controller's joint indices for correct mapping
                    joint_indices = arm_ctrl.active_joint_indices
                    if hasattr(joint_indices, 'cpu'):
                        joint_indices = joint_indices.cpu().numpy()
                    new_qpos[joint_indices] = gello_joints
                else:
                    # Fallback to slice-based (assumes contiguous qpos)
                    arm_slice = self.robot_config.get_arm_joint_slice(arm_idx)
                    new_qpos[arm_slice] = gello_joints
        
        robot.set_qpos(new_qpos)
    
    def _teleop_episode(self) -> str:
        """
        Run teleoperation for one episode.
        
        Returns:
            "success", "quit", or "restart"
        """
        viewer = self.env.render_human()
        dt = 1.0 / self.args.control_freq
        step = 0
        
        # Debug: print state at teleop start
        print("\n" + "=" * 60)
        print("TELEOP STARTING - Initial State:")
        print("=" * 60)
        robot = self.env.unwrapped.agent.robot
        current_qpos = robot.get_qpos()
        if hasattr(current_qpos, 'cpu'):
            current_qpos = current_qpos.cpu().numpy().flatten()
        
        device_states = [d.get_state() for d in self.devices]
        first_action = self.action_mapper.map_action(device_states)
        
        # Get target joints (what the ghost was showing)
        target_joints = self.action_mapper.get_target_joints(self.env)
        
        print(f"  Ghost (target):    {[f'{x:.3f}' for x in target_joints[0][:6]]}")
        print(f"  GELLO joints:      {[f'{x:.3f}' for x in device_states[0].joint_positions[:6]]}")
        print(f"  Virtual robot:     {[f'{x:.3f}' for x in current_qpos[:6]]}")
        print(f"  First action:      {[f'{x:.3f}' for x in first_action[:6]]}")
        print("=" * 60 + "\n")
        
        # CRITICAL: Sync controller's internal target with current robot qpos
        # Without this, the controller will try to move from its old target (e.g., home keyframe)
        # to the new action, causing a jump
        controller = self.env.unwrapped.agent.controller
        if hasattr(controller, 'controllers'):
            arm_ctrl = controller.controllers.get('arm')
            if arm_ctrl is not None and hasattr(arm_ctrl, '_target_qpos'):
                # Get current arm qpos
                arm_qpos = robot.get_qpos()
                if hasattr(arm_qpos, 'cpu'):
                    arm_qpos = arm_qpos.cpu()
                # Update controller's target to current position
                arm_ctrl._target_qpos = arm_qpos[:, :6].clone()
                print(f"[Sync] Updated arm controller target to current qpos")
        
        try:
            first_step = True
            while True:
                step_start = time.time()
                
                # Check keyboard input
                if viewer.window.key_press("h"):
                    self._print_help()
                    continue
                elif viewer.window.key_press("q"):
                    print("\nQuitting...")
                    return "quit"
                elif viewer.window.key_press("r"):
                    print("\nRestarting episode...")
                    return "restart"
                
                # Read device states
                device_states = [d.get_state() for d in self.devices]
                
                # Map to action
                action = self.action_mapper.map_action(device_states)
                
                # Debug first step
                if first_step:
                    print("\n--- FIRST STEP DEBUG ---")
                    before_qpos = robot.get_qpos()
                    if hasattr(before_qpos, 'cpu'):
                        before_qpos = before_qpos.cpu().numpy().flatten()
                    print(f"  Before step qpos (all): {[f'{x:.3f}' for x in before_qpos]}")
                    print(f"  Action to send (all):   {[f'{x:.3f}' for x in action]}")
                    
                    # Check arm controller's target before step
                    controller = self.env.unwrapped.agent.controller
                    arm_ctrl = controller.controllers.get('arm')
                    if arm_ctrl and hasattr(arm_ctrl, '_target_qpos'):
                        tgt = arm_ctrl._target_qpos
                        if tgt is not None:
                            if hasattr(tgt, 'cpu'):
                                tgt = tgt.cpu().numpy().flatten()
                            print(f"  Arm ctrl target BEFORE: {[f'{x:.3f}' for x in tgt]}")
                
                # Execute action through env.step (applies action via controller)
                obs, reward, terminated, truncated, info = self.env.step(action)

                print(f"info: {info}")
                
                # Debug first step - after
                if first_step:
                    after_qpos = robot.get_qpos()
                    if hasattr(after_qpos, 'cpu'):
                        after_qpos = after_qpos.cpu().numpy().flatten()
                    print(f"  After step qpos (all):  {[f'{x:.3f}' for x in after_qpos]}")
                    diff = after_qpos - before_qpos
                    print(f"  Qpos diff (all):        {[f'{x:.3f}' for x in diff]}")
                    
                    # Check arm controller's target after step
                    arm_ctrl = controller.controllers.get('arm')
                    if arm_ctrl and hasattr(arm_ctrl, '_target_qpos'):
                        tgt = arm_ctrl._target_qpos
                        if tgt is not None:
                            if hasattr(tgt, 'cpu'):
                                tgt = tgt.cpu().numpy().flatten()
                            print(f"  Arm ctrl target AFTER:  {[f'{x:.3f}' for x in tgt]}")
                    
                    # Check controller info
                    print(f"  Controller type: {type(controller).__name__}")
                    if hasattr(controller, 'controllers'):
                        print(f"  Sub-controllers: {list(controller.controllers.keys())}")
                        for name, ctrl in controller.controllers.items():
                            if hasattr(ctrl, 'single_action_space'):
                                space = ctrl.single_action_space
                                print(f"    {name}: low={space.low}, high={space.high}")
                            if hasattr(ctrl, '_normalize_action'):
                                print(f"    {name} normalize_action: {ctrl._normalize_action}")
                    
                    print("--- END FIRST STEP DEBUG ---\n")
                    first_step = False
                
                # Render
                self.env.render_human()
                
                # Check success
                if info.get('success', False):
                    return "success"
                
                # Check termination
                if terminated or truncated:
                    print(f"\n❌ Episode terminated (step {step})")
                    return self._handle_termination(viewer)
                
                # Control frequency
                elapsed = time.time() - step_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                step += 1
                
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at step {step}")
            return "restart"
    
    def _handle_termination(self, viewer) -> str:
        """Handle episode termination (not success)."""
        print("Press 'r' to restart, 'q' to quit")
        
        while True:
            self.env.render_human()
            if viewer.window.key_press("r"):
                return "restart"
            elif viewer.window.key_press("q"):
                return "quit"
            time.sleep(0.01)
    
    def _print_help(self) -> None:
        """Print help message."""
        print("""
        ╔════════════════════════════════════════╗
        ║         GELLO Teleoperation Help       ║
        ╠════════════════════════════════════════╣
        ║  h : Show this help                    ║
        ║  q : Quit (won't save current episode) ║
        ║  r : Restart episode (won't save)      ║
        ╠════════════════════════════════════════╣
        ║  Episodes are saved on task SUCCESS    ║
        ╚════════════════════════════════════════╝
        """)
    
    def _clear_buffer(self) -> None:
        """Clear trajectory and video buffers without saving."""
        self.env.clear_all_buffers()
    
    def _finish(self) -> None:
        """Finish session and print summary."""
        h5_path = self.env._h5_file.filename
        json_path = self.env._json_path
        
        self.env.close()
        
        print("\n" + "=" * 60)
        print("Data Collection Complete!")
        print("=" * 60)
        print(f"Data: {h5_path}")
        print(f"Metadata: {json_path}")


def main():
    """Main entry point."""
    args = tyro.cli(Args)
    
    session = TeleopSession(args)
    session.setup()
    session.run()


if __name__ == "__main__":
    main()

