"""
Dual Arm Action Mapper

Maps multiple GELLO device inputs to dual-arm robot actions.
"""

from typing import List, Optional, Dict
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from molmo_spaces_maniskill.teleop.action_mappers.base_mapper import BaseActionMapper
from molmo_spaces_maniskill.teleop.devices.gello_device import GelloState


class DualArmActionMapper(BaseActionMapper):
    """
    Action mapper for dual-arm (bimanual) robots.
    
    Converts multiple GELLO device states to ManiSkill actions for robots like:
    - Xlerobot (dual SO100 arms)
    - UR dual-arm setup
    - Mobile manipulators with two arms
    
    Example:
        >>> mapper = DualArmActionMapper("xlerobot")
        >>> left_state = left_device.get_state()
        >>> right_state = right_device.get_state()
        >>> action = mapper.map_action([left_state, right_state])
        >>> obs, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        robot_uid: str,
        env: Optional[BaseEnv] = None,
        arm_device_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Initialize dual-arm mapper.
        
        Args:
            robot_uid: ManiSkill robot UID
            env: Optional environment for action space info
            arm_device_mapping: Optional mapping from arm index to device index.
                               Default is {0: 0, 1: 1} (arm0 -> device0, arm1 -> device1)
        """
        super().__init__(robot_uid, env)
        
        # Verify this is a bimanual robot
        if len(self.robot_config.arms) < 2:
            raise ValueError(
                f"DualArmActionMapper requires at least 2 arms, "
                f"but robot '{robot_uid}' has {len(self.robot_config.arms)} arms"
            )
        
        # Default arm-to-device mapping
        self.arm_device_mapping = arm_device_mapping or {
            i: i for i in range(len(self.robot_config.arms))
        }
    
    def map_action(
        self,
        device_states: List[GelloState],
        debug: bool = False,
    ) -> np.ndarray:
        """
        Convert multiple GELLO states to robot action.
        
        Args:
            device_states: List of GelloState from each device
            debug: Whether to print debug information
            
        Returns:
            Action array combining all arm and gripper actions
        """
        num_arms = len(self.robot_config.arms)
        if len(device_states) != num_arms:
            raise ValueError(
                f"Expected {num_arms} device states, got {len(device_states)}"
            )
        
        # Initialize action array
        action_dim = self.robot_config.action_dim
        action = np.zeros(action_dim)
        
        # Fill in base/body joints if present (keep at zero or current)
        # These are typically not controlled by GELLO
        
        # Map each arm
        for arm_idx, arm_config in enumerate(self.robot_config.arms):
            device_idx = self.arm_device_mapping.get(arm_idx, arm_idx)
            gello_state = device_states[device_idx]
            
            # Get slices for this arm
            arm_slice = self.robot_config.get_arm_joint_slice(arm_idx)
            gripper_slice = self.robot_config.get_gripper_slice(arm_idx)
            
            # Map arm joints
            num_arm_joints = arm_config.num_joints
            gello_joints = gello_state.joint_positions[:num_arm_joints]
            action[arm_slice] = gello_joints
            
            # Map gripper
            gripper_action = self._map_gripper(gello_state.gripper_state, arm_idx)
            gripper_indices = list(range(gripper_slice.start, gripper_slice.stop))
            for idx in gripper_indices:
                action[idx] = gripper_action
            
            if debug:
                print(f"\n[DEBUG] Arm {arm_idx} ({arm_config.name}):")
                print(f"  Device {device_idx} joints: {gello_joints}")
                print(f"  Device {device_idx} gripper: {gello_state.gripper_state}")
                print(f"  Arm slice: {arm_slice}")
                print(f"  Gripper slice: {gripper_slice}")
                print(f"  Gripper action: {gripper_action}")
        
        if debug:
            print(f"\n[DEBUG] Final action ({action_dim} dim): {action}")
        
        return action
    
    def get_target_joints(self, env: BaseEnv) -> List[np.ndarray]:
        """
        Get target joint positions for GELLO alignment.
        
        Returns the current robot arm joint positions for each arm.
        Uses the controller's active_joint_indices to correctly extract
        joint positions even when qpos is interleaved.
        
        Args:
            env: ManiSkill environment
            
        Returns:
            List of target joint arrays (one per arm)
        """
        robot = env.unwrapped.agent.robot
        qpos = robot.get_qpos()
        
        # Convert to numpy if needed
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()
        else:
            qpos = np.array(qpos)
        
        qpos = qpos.flatten()
        
        # Try to get joint indices from controller (handles interleaved qpos)
        agent = env.unwrapped.agent
        ctrl = agent.controller
        
        target_joints = []
        
        for arm_idx, arm_config in enumerate(self.robot_config.arms):
            # Try to find the arm controller by name
            arm_ctrl = None
            if hasattr(ctrl, 'controllers'):
                # Try common naming patterns
                possible_names = [
                    arm_config.name,  # Use config name directly (e.g., "left_arm", "right_arm")
                    f"arm_{arm_idx}",
                ]
                for name in possible_names:
                    if name and name in ctrl.controllers:
                        arm_ctrl = ctrl.controllers[name]
                        break
            
            if arm_ctrl is not None:
                joint_indices = arm_ctrl.active_joint_indices
                if hasattr(joint_indices, 'cpu'):
                    joint_indices = joint_indices.cpu().numpy()
                arm_joints = qpos[joint_indices]
            else:
                # Fallback to slice-based extraction
                arm_slice = self.robot_config.get_arm_joint_slice(arm_idx)
                arm_joints = qpos[arm_slice]
            
            target_joints.append(arm_joints)
        
        return target_joints
    
    def compute_alignment_error(
        self,
        device_states: List[GelloState],
        target_joints: List[np.ndarray],
    ) -> List[float]:
        """
        Compute alignment error for each arm.
        
        Args:
            device_states: Current device states
            target_joints: Target joint positions for each arm
            
        Returns:
            List of maximum absolute joint errors (one per arm)
        """
        errors = []
        
        for arm_idx, arm_config in enumerate(self.robot_config.arms):
            device_idx = self.arm_device_mapping.get(arm_idx, arm_idx)
            
            if device_idx >= len(device_states):
                errors.append(float('inf'))
                continue
            
            gello_joints = device_states[device_idx].joint_positions
            target = target_joints[arm_idx]
            
            # Compute error for matching joints
            num_joints = min(len(gello_joints), len(target))
            joint_errors = np.abs(gello_joints[:num_joints] - target[:num_joints])
            errors.append(np.max(joint_errors))
        
        return errors
    
    def all_arms_aligned(
        self,
        device_states: List[GelloState],
        target_joints: List[np.ndarray],
        threshold: float = 0.2,
    ) -> bool:
        """
        Check if all arms are aligned within threshold.
        
        Args:
            device_states: Current device states
            target_joints: Target joint positions for each arm
            threshold: Maximum allowed error (radians)
            
        Returns:
            True if all arms are aligned
        """
        errors = self.compute_alignment_error(device_states, target_joints)
        return all(e < threshold for e in errors)

