#!/usr/bin/env python3
"""
SO101 Leader Teleoperation Script for ManiSkill

This script provides teleoperation capabilities for ManiSkill robots using LeRobot SO101 Leader.
Uses the same episode recording and management logic as GELLO teleoperation.

Usage:
    # Basic teleoperation
    python -m mani_skill.teleop.scripts.teleop_so101 -e PickPlaceSO100-v1 -r so100
    
    # With calibration
    python -m mani_skill.teleop.scripts.teleop_so101 -e PickPlaceSO100-v1 -r so100 --calibrate
    
    # Test device only
    python -m mani_skill.teleop.scripts.teleop_so101 --test-only

Features:
    - Automatic motor calibration
    - Ghost visualization for alignment
    - Only saves successful trajectories
    - Configurable recording frequency
"""

import time
from dataclasses import dataclass
from typing import Annotated, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

try:
    from molmo_spaces_maniskill.teleop.devices import SO101Device, SO101Config, SO101_AVAILABLE
except ImportError:
    SO101_AVAILABLE = False

from molmo_spaces_maniskill.teleop.wrappers import RecordEpisodeWithFreq

# Register custom environments and robots
from molmo_spaces_maniskill.molmoact2.tasks import *
from molmo_spaces_maniskill.molmoact2.robots.so100 import SO100WristCam  # Register so100_wristcam


@dataclass
class Args:
    """SO101 Leader teleoperation arguments."""
    env_id: Annotated[Optional[str], tyro.conf.arg(aliases=["-e"])] = None
    """Environment ID (use --test-only to skip environment)"""
    
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "so100_wristcam"
    """Robot UID (e.g., so100, so100_wristcam)"""
    
    port: str = "/dev/ttyACM0"
    """SO101 serial port"""
    
    obs_mode: str = "rgb"
    """Observation mode"""
    
    control_freq: float = 50.0
    """Control frequency (Hz)"""
    
    record_freq: Optional[float] = 10.0
    """Recording frequency (Hz). None = record every step"""
    
    max_episode_steps: int = 50000
    """Maximum steps per episode"""
    
    target_trajs: int = 100
    """Target number of trajectories to collect"""
    
    # SO101 settings
    use_degrees: bool = False
    """Use degrees instead of normalized range"""
    
    calibrate: bool = False
    """Run motor calibration on connect (range of motion)"""
    
    calibrate_offsets: bool = False
    """Run offset calibration (align with ghost pose)"""
    
    test_only: bool = False
    """Test SO101 reading without environment"""
    
    render: bool = True
    """Render the environment"""
    
    record_dir: str = "demos"
    """Directory to save recorded trajectories"""
    
    use_timestamp: bool = True
    """Add timestamp to output directory"""
    
    save_video: bool = True
    """Save video of teleoperation"""
    
    viewer_shader: str = "default"
    """Viewer shader (use 'default' for scenes with many lights, 'rt' for ray tracing)"""
    
    enable_shadow: bool = False
    """Enable shadows (disable for scenes with many lights)"""
    
    alignment_threshold: float = 0.15
    """Joint alignment threshold (radians)"""
    
    skip_alignment: bool = False
    """Skip alignment check and start teleoperation immediately"""
    
    show_ghost: bool = True
    """Show ghost robot for alignment"""


class SO101TeleopSession:
    """
    Teleoperation session using SO101 Leader.
    
    Uses the same episode management logic as GELLO teleoperation:
    - Only saves successful trajectories
    - Supports restart and quit commands
    - Uses RecordEpisodeWithFreq wrapper for data collection
    """
    
    def __init__(self, args: Args):
        self.args = args
        self.device: Optional[SO101Device] = None
        self.env = None
        
    def setup(self):
        """Setup device and environment."""
        print("\n" + "=" * 60)
        print("SO101 Leader Teleoperation Setup")
        print("=" * 60)
        
        # Check if LeRobot is available
        if not SO101_AVAILABLE:
            raise ImportError(
                "LeRobot is not installed. Please install it with:\n"
                "  pip install lerobot\n"
                "Or clone from: https://github.com/huggingface/lerobot"
            )
        
        # Connect to SO101
        print(f"\n[1/3] Connecting to SO101 at {self.args.port}...")
        config = SO101Config(
            port=self.args.port,
            use_degrees=self.args.use_degrees,
        )
        self.device = SO101Device(self.args.port, config=config)
        self.device.connect(calibrate=self.args.calibrate)
        
        # Load offset calibration if exists
        print(f"\n[2/3] Loading offset calibration...")
        if not self.device.load_offset_calibration():
            print("  No offset calibration found. Use --calibrate-offsets to calibrate.")
        
        # Create environment
        print(f"\n[3/3] Creating environment...")
        self._setup_environment()
        
        print(f"\n✓ Setup complete!")
        print(f"  Environment: {self.args.env_id}")
        print(f"  Robot: {self.args.robot_uid}")
        print(f"  Control freq: {self.args.control_freq} Hz")
        print(f"  Record freq: {self.args.record_freq} Hz")
        print(f"  Target trajectories: {self.args.target_trajs}")
    
    def _setup_environment(self):
        """Create and configure the environment with recording wrapper."""
        import mani_skill.envs
        
        # Determine output directory
        if self.args.use_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{self.args.record_dir}/{self.args.env_id}/so101_teleop_{timestamp}/"
        else:
            output_dir = f"{self.args.record_dir}/{self.args.env_id}/so101_teleop/"
        
        # Create base environment
        # Note: reset_robot_qpos=False because SO101 alignment handles robot positioning
        env = gym.make(
            self.args.env_id,
            obs_mode=self.args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            robot_uids=self.args.robot_uid,
            enable_shadow=self.args.enable_shadow,
            viewer_camera_configs=dict(shader_pack=self.args.viewer_shader),
            max_episode_steps=self.args.max_episode_steps,
            reset_robot_qpos=False,  # SO101 alignment handles robot positioning
        )
        
        # Wrap with recording
        self.env = RecordEpisodeWithFreq(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=self.args.save_video,
            info_on_video=False,
            source_type="teleoperation",
            source_desc=f"SO101 Leader teleoperation ({self.args.robot_uid})",
            record_freq=self.args.record_freq,
            control_freq=self.args.control_freq,
        )
        
        print(f"✓ Environment: {self.args.env_id}")
        print(f"✓ Robot: {self.args.robot_uid}")
        print(f"✓ Output: {output_dir}")
        
        # Debug: print available sensors
        obs, _ = self.env.reset()
        if "sensor_data" in obs:
            print(f"✓ Available cameras: {list(obs['sensor_data'].keys())}")
    
    def _update_robot_visualization(self, joint_positions: np.ndarray, gripper_state: float):
        """Update robot visualization to show current SO101 state."""
        env = self.env.unwrapped
        robot = env.agent.robot
        
        # Get current qpos and update arm joints
        current_qpos = robot.get_qpos()
        if hasattr(current_qpos, 'cpu'):
            current_qpos = current_qpos.cpu().numpy().flatten()
        
        # Update arm joints (first 5) and gripper (last 1)
        new_qpos = current_qpos.copy()
        new_qpos[:5] = joint_positions
        # Gripper mapping: SO101 [0,1] -> SO100 [0, -1.1]
        new_qpos[5] = -(1.0 - gripper_state) * 1.1
        
        robot.set_qpos(torch.tensor(new_qpos, dtype=torch.float32).unsqueeze(0))
        robot.set_qvel(robot.get_qvel() * 0)
    
    def _wait_for_alignment(self) -> bool:
        """Wait for user to align SO101 with target pose using ghost visualization."""
        import sapien.utils.viewer
        from mani_skill.utils import sapien_utils
        
        env = self.env.unwrapped
        robot = env.agent.robot
        
        # Render first to initialize viewer
        viewer = self.env.render_human()
        
        # Get target pose from keyframe
        # Use 'home' keyframe if available (for SO101 Leader), otherwise 'rest'
        if hasattr(env.agent, 'keyframes'):
            if 'home' in env.agent.keyframes:
                target_qpos = env.agent.keyframes['home'].qpos
            elif 'rest' in env.agent.keyframes:
                target_qpos = env.agent.keyframes['rest'].qpos
            else:
                target_qpos = robot.get_qpos().cpu().numpy().flatten()
            if hasattr(target_qpos, 'cpu'):
                target_qpos = target_qpos.cpu().numpy().flatten()
        else:
            target_qpos = robot.get_qpos().cpu().numpy().flatten()
        
        # Target arm joints (exclude gripper)
        target_arm = target_qpos[:5]
        
        # Set robot to target pose BEFORE creating ghost
        # Ghost captures current robot position when update_ghost_objects() is called
        robot.set_qpos(torch.tensor(target_qpos, dtype=torch.float32).unsqueeze(0))
        robot.set_qvel(robot.get_qvel() * 0)
        self.env.render_human()  # Render to update visual
        
        print("\n" + "=" * 60)
        print("Align SO101 with target pose (ghost)")
        print("=" * 60)
        print(f"Target joints: {[f'{j:.3f}' for j in target_arm]}")
        print(f"Threshold: {self.args.alignment_threshold} rad")
        print("Move SO101 Leader to match the ghost pose...")
        print("Press Ctrl+C to skip alignment")
        print("=" * 60)
        
        # Setup ghost visualization using TransformWindow
        transform_window = None
        if viewer is not None:
            for plugin in viewer.plugins:
                if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
                    transform_window = plugin
                    break
        
        if transform_window and self.args.show_ghost:
            try:
                # Find end-effector link (wrist_roll link for SO100)
                ee_link = sapien_utils.get_obj_by_name(robot.links, "wrist_roll")
                if ee_link is None:
                    ee_link = sapien_utils.get_obj_by_name(robot.links, "Moving_Jaw")
                if ee_link is None:
                    ee_link = sapien_utils.get_obj_by_name(robot.links, "link_5")
                
                if ee_link:
                    viewer.select_entity(ee_link._objs[0].entity)
                    transform_window.enabled = True
                    # This captures current robot pose as ghost
                    transform_window.update_ghost_objects()
                    transform_window.follow = False
                    self.env.render_human()
                    print("✓ Ghost visualization enabled")
                else:
                    print("Warning: Could not find end-effector link for ghost")
            except Exception as e:
                print(f"Warning: Could not setup ghost visualization: {e}")
        
        last_print_time = 0
        print_interval = 1.0
        
        try:
            while True:
                state = self.device.get_state()
                current_joints = state.joint_positions
                
                # Update robot visualization to show current SO101 state
                # Ghost stays at target position, robot moves with SO101
                self._update_robot_visualization(current_joints, state.gripper_state)
                
                # Compute error (offsets already applied in get_state())
                errors = np.abs(current_joints - target_arm)
                max_error = np.max(errors)
                
                # Print status periodically
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    last_print_time = current_time
                    print("\n--- Alignment Debug ---")
                    print(f"  Raw values:    {[f'{v:.1f}' for v in state.raw_joints[:5]]}")
                    print(f"  SO101 joints:  {[f'{j:.3f}' for j in current_joints]}")
                    print(f"  Target joints: {[f'{j:.3f}' for j in target_arm]}")
                    print(f"  Joint errors:  {[f'{e:.3f}' for e in errors]}")
                    print(f"  Max error: {max_error:.3f} rad ({np.degrees(max_error):.1f}°)")
                    print(f"  Gripper: {state.gripper_state:.3f}")
                
                # Check alignment
                if max_error < self.args.alignment_threshold:
                    print(f"\n✓ Aligned! Max error: {max_error:.3f} rad")
                    return True
                
                self.env.render_human()
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\n\nAlignment skipped.")
            return False
        finally:
            # Cleanup ghost
            if transform_window:
                transform_window.enabled = False
                viewer.select_entity(None)
        
    def run(self):
        """Run the main teleoperation loop (same logic as GELLO)."""
        # Run offset calibration if requested
        if self.args.calibrate_offsets:
            self._run_offset_calibration()
            return
        
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
        print("\nMove SO101 to align with robot, then start teleoperating!")
        
        while num_trajs < self.args.target_trajs:
            # Wait for alignment (unless skipped)
            if not self.args.skip_alignment:
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
        print("TELEOP STARTING")
        print("=" * 60)
        
        env = self.env.unwrapped
        
        # Reset episode-specific state (e.g., button_pressed_once) that may have been
        # accidentally triggered during alignment phase
        if hasattr(env, 'reset_episode_state'):
            env.reset_episode_state()
            print("  Episode state reset")
        robot = env.agent.robot
        
        current_qpos = robot.get_qpos()
        if hasattr(current_qpos, 'cpu'):
            current_qpos = current_qpos.cpu().numpy().flatten()
        
        state = self.device.get_state()
        gripper_action = -(1.0 - state.gripper_state) * 1.1
        first_action = np.concatenate([state.joint_positions, [gripper_action]])
        
        print(f"  SO101 joints:      {[f'{x:.3f}' for x in state.joint_positions]}")
        print(f"  Virtual robot:     {[f'{x:.3f}' for x in current_qpos[:5]]}")
        print(f"  First action:      {[f'{x:.3f}' for x in first_action]}")
        print("=" * 60 + "\n")
        
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
                
                # Read SO101 state
                state = self.device.get_state()
                
                # Create action: [arm joints, gripper]
                gripper_action = -(1.0 - state.gripper_state) * 1.1
                action = np.concatenate([state.joint_positions, [gripper_action]])
                
                # Debug first step
                if first_step:
                    print("\n--- FIRST STEP DEBUG ---")
                    before_qpos = robot.get_qpos()
                    if hasattr(before_qpos, 'cpu'):
                        before_qpos = before_qpos.cpu().numpy().flatten()
                    print(f"  Before step qpos: {[f'{x:.3f}' for x in before_qpos]}")
                    print(f"  Action to send:   {[f'{x:.3f}' for x in action]}")
                    first_step = False
                
                # Execute action
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Render
                self.env.render_human()

                # print(f"  Info: {info}")
                
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
        ║       SO101 Leader Teleop Help         ║
        ╠════════════════════════════════════════╣
        ║  h : Show this help                    ║
        ║  q : Quit (won't save current episode) ║
        ║  r : Restart episode (won't save)      ║
        ╠════════════════════════════════════════╣
        ║  Episodes are saved on task SUCCESS    ║
        ╚════════════════════════════════════════╝
        """)
    
    def _clear_buffer(self) -> None:
        """Clear trajectory buffer without saving."""
        if self.env._trajectory_buffer is not None:
            self.env._trajectory_buffer = None
    
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
    
    def _run_offset_calibration(self):
        """Run offset calibration to align SO101 with robot."""
        import sapien.utils.viewer
        from mani_skill.utils import sapien_utils
        
        env = self.env.unwrapped
        robot = env.agent.robot
        viewer = env._viewer
        
        # Get target pose (rest keyframe)
        if hasattr(env.agent, 'keyframes') and 'rest' in env.agent.keyframes:
            keyframe = env.agent.keyframes['rest']
            target_qpos = keyframe.qpos
            if hasattr(target_qpos, 'cpu'):
                target_qpos = target_qpos.cpu().numpy().flatten()
        else:
            # Use current qpos
            target_qpos = robot.get_qpos()
            if hasattr(target_qpos, 'cpu'):
                target_qpos = target_qpos.cpu().numpy().flatten()
        
        # Extract arm joints (first 5)
        target_arm = target_qpos[:5]
        
        print("\n" + "="*60)
        print("Offset Calibration")
        print("="*60)
        print(f"Target arm joints: {[f'{j:.3f}' for j in target_arm]}")
        print("\nLook at the simulation window to see the ghost (target pose).")
        print("Move SO101 to match the ghost pose...")
        
        # Render first to initialize viewer
        env.render_human()
        
        # Create ghost visualization
        transform_window = None
        for plugin in viewer.plugins:
            if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
                transform_window = plugin
                break
        
        # Get end-effector link for ghost
        ee_link = sapien_utils.get_obj_by_name(robot.get_links(), "link_5")
        
        if transform_window and ee_link:
            # Create ghost at target pose
            robot.set_qpos(torch.tensor(target_qpos, dtype=torch.float32).unsqueeze(0))
            robot.set_qvel(robot.get_qvel() * 0)
            env.render_human()
            
            # Enable ghost
            transform_window.enabled = True
            viewer.select_entity(ee_link)
            transform_window.ghost = True
            
            print("\n✓ Ghost created! The transparent robot shows the target pose.")
        else:
            print("\nWarning: Could not create ghost visualization.")
        
        print("\nMove SO101 to match the ghost, then press ENTER...")
        
        # Interactive loop - show current SO101 position while waiting
        print("\nCurrent SO101 reading (updating live):")
        try:
            while True:
                # Read SO101 state
                state = self.device.get_state()
                
                # Update robot to show current SO101 position
                current_qpos = target_qpos.copy()
                current_qpos[:5] = state.raw_joints[:5] / 100.0 * (np.pi / 2)  # Raw to radians
                current_qpos[5] = -(1.0 - state.gripper_state) * 1.1
                
                robot.set_qpos(torch.tensor(current_qpos, dtype=torch.float32).unsqueeze(0))
                robot.set_qvel(robot.get_qvel() * 0)
                env.render_human()
                
                # Print current raw values
                print(f"\r  Raw: {[f'{v:.1f}' for v in state.raw_joints[:5]]}  ", end="", flush=True)
                
                time.sleep(0.05)
                
                # Check for Enter key (non-blocking would be better, but this works)
                import sys
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    break
                    
        except KeyboardInterrupt:
            print("\n\nCalibration cancelled.")
            if transform_window:
                transform_window.enabled = False
            return
        
        print("\n")
        
        # Read final raw values
        action = self.device._bus.sync_read("Present_Position")
        raw_values = np.array([
            action.get("shoulder_pan", 0),
            action.get("shoulder_lift", 0),
            action.get("elbow_flex", 0),
            action.get("wrist_flex", 0),
            action.get("wrist_roll", 0),
        ])
        
        print(f"Raw values at target: {[f'{v:.1f}' for v in raw_values]}")
        
        # Calculate offsets
        raw_scaled = raw_values / 100.0 * (np.pi / 2)
        offsets = target_arm - raw_scaled
        
        print(f"Computed offsets: {[f'{o:.3f}' for o in offsets]}")
        
        # Update config
        self.device.config.joint_offsets = tuple(offsets.tolist())
        
        # Save to file
        offset_path = self.device._get_offset_calibration_path()
        import json
        offset_data = {
            "joint_offsets": list(offsets),
            "joint_signs": list(self.device.config.joint_signs),
            "target_joints": list(target_arm),
            "raw_values_at_target": list(raw_values),
        }
        with open(offset_path, "w") as f:
            json.dump(offset_data, f, indent=2)
        
        print(f"\n✓ Offset calibration saved to {offset_path}")
        
        # Cleanup ghost
        if transform_window:
            transform_window.enabled = False
            viewer.select_entity(None)
        
        print("\nOffset calibration complete!")
        print("Run again without --calibrate-offsets to teleoperate.")
    
    def cleanup(self):
        """Cleanup resources (called on unexpected exit, not normal finish)."""
        if self.device is not None:
            self.device.disconnect()
        # Note: env is closed in _finish() for normal exit


def test_so101_reading(args: Args):
    """Test SO101 reading without environment."""
    print("=" * 60)
    print("SO101 Leader Test Mode")
    print("=" * 60)
    
    if not SO101_AVAILABLE:
        raise ImportError("LeRobot is not installed.")
    
    config = SO101Config(
        port=args.port,
        use_degrees=args.use_degrees,
    )
    device = SO101Device(args.port, config=config)
    device.connect(calibrate=args.calibrate)
    
    print("\nReading SO101 state (Ctrl+C to stop)...")
    print("-" * 60)
    
    try:
        while True:
            state = device.get_state()
            joints_str = ", ".join([f"{j:.3f}" for j in state.joint_positions])
            print(f"\rJoints: [{joints_str}] | Gripper: {state.gripper_state:.3f}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        device.disconnect()


def main():
    args = tyro.cli(Args)
    
    if args.test_only:
        test_so101_reading(args)
        return
    
    if args.env_id is None:
        print("Error: --env-id/-e is required (or use --test-only to skip environment)")
        return
    
    session = SO101TeleopSession(args)
    session.setup()
    session.run()


if __name__ == "__main__":
    main()
