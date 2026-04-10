"""
Single Arm Action Mapper

Maps GELLO device input to single-arm robot actions.
"""

from typing import List, Optional
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from molmo_spaces_maniskill.teleop.action_mappers.base_mapper import BaseActionMapper
from molmo_spaces_maniskill.teleop.devices.gello_device import GelloState


class SingleArmActionMapper(BaseActionMapper):
    """
    Action mapper for single-arm robots.
    
    Converts GELLO joint states to ManiSkill actions for robots like:
    - Panda (7 DOF + gripper)
    - xArm (7 DOF + gripper)
    - UR (6 DOF + gripper)
    - Kinova Gen3 Lite (6 DOF + gripper)
    
    Example:
        >>> mapper = SingleArmActionMapper("panda")
        >>> gello_state = device.get_state()
        >>> action = mapper.map_action([gello_state])
        >>> obs, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        robot_uid: str,
        env: Optional[BaseEnv] = None,
    ):
        super().__init__(robot_uid, env)
        
        # Get arm configuration
        if len(self.robot_config.arms) != 1:
            raise ValueError(
                f"SingleArmActionMapper requires exactly 1 arm, "
                f"but robot '{robot_uid}' has {len(self.robot_config.arms)} arms"
            )
        
        self.arm_config = self.robot_config.arms[0]
    
    def map_action(
        self,
        device_states: List[GelloState],
        debug: bool = False,
    ) -> np.ndarray:
        """
        Convert GELLO state to robot action.
        
        Args:
            device_states: List with single GelloState
            debug: Whether to print debug information
            
        Returns:
            Action array [joint_positions..., gripper_action]
        """
        if len(device_states) != 1:
            raise ValueError(f"Expected 1 device state, got {len(device_states)}")
        
        gello_state = device_states[0]
        
        # Get expected action dimension
        action_dim = self.robot_config.action_dim
        action = np.zeros(action_dim)
        
        # Get joint and gripper slices
        arm_slice = self.robot_config.get_arm_joint_slice(0)
        gripper_slice = self.robot_config.get_gripper_slice(0)
        
        # Map arm joints
        num_arm_joints = self.arm_config.num_joints
        action[arm_slice] = gello_state.joint_positions[:num_arm_joints]
        
        # Map gripper
        gripper_action = self._map_gripper(gello_state.gripper_state, arm_idx=0)
        gripper_indices = list(range(gripper_slice.start, gripper_slice.stop))
        for idx in gripper_indices:
            action[idx] = gripper_action
        
        if debug:
            print(f"\n[DEBUG] SingleArmActionMapper")
            print(f"  GELLO joints: {gello_state.joint_positions}")
            print(f"  GELLO gripper: {gello_state.gripper_state}")
            print(f"  Action dim: {action_dim}")
            print(f"  Arm slice: {arm_slice}")
            print(f"  Gripper slice: {gripper_slice}")
            print(f"  Final action: {action}")
        
        return action
    
    def get_target_joints(self, env: BaseEnv) -> List[np.ndarray]:
        """
        Get target joint positions for GELLO alignment.
        
        Returns the current robot arm joint positions as the alignment target.
        
        Args:
            env: ManiSkill environment
            
        Returns:
            List with single array of target joint positions
        """
        robot = env.unwrapped.agent.robot
        qpos = robot.get_qpos()
        
        # Convert to numpy if needed
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()
        else:
            qpos = np.array(qpos)
        
        qpos = qpos.flatten()
        
        # Extract arm joints
        num_arm_joints = self.arm_config.num_joints
        target_joints = qpos[:num_arm_joints]
        
        return [target_joints]
    
    def compute_alignment_error(
        self,
        device_states: List[GelloState],
        target_joints: List[np.ndarray],
    ) -> float:
        """
        Compute alignment error between device and target positions.
        
        Also checks that GELLO joint values are within controller limits.
        If any joint is out of bounds, returns a large error to prevent alignment.
        
        Args:
            device_states: Current device states
            target_joints: Target joint positions
            
        Returns:
            Maximum absolute joint error (or large value if out of bounds)
        """
        if len(device_states) != 1 or len(target_joints) != 1:
            raise ValueError("Expected single device state and target")
        
        gello_joints = device_states[0].joint_positions
        target = target_joints[0]
        
        # Get controller joint limits if environment is available
        joint_limits = self._get_arm_joint_limits()
        
        # Compute error for matching joints
        num_joints = min(len(gello_joints), len(target))
        errors = []
        for i in range(num_joints):
            # Check if GELLO joint value is within controller limits
            if joint_limits is not None and i < len(joint_limits[0]):
                low, high = joint_limits[0][i], joint_limits[1][i]
                if gello_joints[i] < low or gello_joints[i] > high:
                    # Joint out of bounds - return large error
                    # This forces the user to recalibrate or reposition GELLO
                    errors.append(abs(gello_joints[i] - target[i]))
                    continue
            
            # Normal case: compute angular difference
            diff = gello_joints[i] - target[i]
            # Wrap to [-π, π] to handle angle periodicity
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            errors.append(abs(diff))
        
        return np.max(errors)
    
    def _get_arm_joint_limits(self) -> Optional[tuple]:
        """
        Get arm joint limits from the controller.
        
        Returns:
            Tuple of (low_limits, high_limits) arrays, or None if unavailable
        """
        if self.env is None:
            return None
        
        try:
            controller = self.env.unwrapped.agent.controller
            if hasattr(controller, 'controllers') and 'arm' in controller.controllers:
                arm_ctrl = controller.controllers['arm']
                if hasattr(arm_ctrl, 'single_action_space'):
                    space = arm_ctrl.single_action_space
                    return (space.low, space.high)
        except Exception:
            pass
        
        return None

