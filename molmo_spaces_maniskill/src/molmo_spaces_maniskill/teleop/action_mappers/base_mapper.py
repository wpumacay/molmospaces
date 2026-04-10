"""
Base Action Mapper

Abstract base class for converting device input to robot actions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from molmo_spaces_maniskill.teleop.devices.gello_device import GelloDevice, GelloState
from molmo_spaces_maniskill.teleop.configs.robot_configs import RobotTeleopConfig, get_robot_config


class BaseActionMapper(ABC):
    """
    Abstract base class for action mappers.
    
    Action mappers convert input from teleoperation devices (like GELLO)
    into actions that can be executed by ManiSkill robots.
    
    Subclasses must implement:
    - map_action(): Convert device states to robot action
    - get_target_joints(): Get target joint positions for alignment
    """
    
    def __init__(
        self,
        robot_uid: str,
        env: Optional[BaseEnv] = None,
    ):
        """
        Initialize action mapper.
        
        Args:
            robot_uid: ManiSkill robot UID
            env: Optional environment for action space info
        """
        self.robot_uid = robot_uid
        self.env = env
        
        # Get robot configuration
        self.robot_config = get_robot_config(robot_uid)
        if self.robot_config is None:
            raise ValueError(f"Unknown robot: {robot_uid}")
    
    @abstractmethod
    def map_action(
        self,
        device_states: List[GelloState],
        debug: bool = False,
    ) -> np.ndarray:
        """
        Convert device states to robot action.
        
        Args:
            device_states: List of GelloState from each device
            debug: Whether to print debug information
            
        Returns:
            Action array for the robot
        """
        pass
    
    @abstractmethod
    def get_target_joints(self, env: BaseEnv) -> List[np.ndarray]:
        """
        Get target joint positions for device alignment.
        
        Called during the alignment phase to show where each device
        should be positioned before starting teleoperation.
        
        Args:
            env: ManiSkill environment
            
        Returns:
            List of target joint arrays (one per device/arm)
        """
        pass
    
    def set_env(self, env: BaseEnv) -> None:
        """Set the environment for action space information."""
        self.env = env
    
    @property
    def num_devices(self) -> int:
        """Number of devices required for this mapper."""
        return self.robot_config.num_arms
    
    @property
    def action_dim(self) -> int:
        """Total action dimension."""
        return self.robot_config.action_dim
    
    @property
    def is_bimanual(self) -> bool:
        """Whether this is a bimanual (dual-arm) setup."""
        return self.robot_config.is_bimanual
    
    def _map_gripper(
        self,
        gripper_state: float,
        arm_idx: int = 0,
    ) -> float:
        """
        Map gripper state from device to robot action space.
        
        Args:
            gripper_state: Gripper state from device [0, 1] (0=open, 1=closed)
            arm_idx: Arm index for robots with multiple grippers
            
        Returns:
            Gripper action value
        """
        if self.env is None:
            # Default: assume normalized action space [-1, 1]
            # GELLO: 0 (open) -> 1 (closed)
            # ManiSkill normalized: -1 (closed) -> 1 (open) typically
            return 1.0 - 2.0 * gripper_state
        
        # Get gripper controller to check its limits
        controller = self.env.unwrapped.agent.controller
        gripper_controller = self._get_gripper_controller(controller, arm_idx)
        
        if gripper_controller is not None:
            action_space = gripper_controller.single_action_space
            gripper_low = action_space.low[0]
            gripper_high = action_space.high[0]
            
            # Check if action space is normalized ([-1, 1]) or actual range
            if gripper_low >= -1.01 and gripper_high <= 1.01:
                # Normalized action space
                # GELLO: 0 (open) -> 1 (closed)
                # ManiSkill normalized: typically -1 (open) -> 1 (closed) for Robotiq
                gripper_action = 2.0 * gripper_state - 1.0
                gripper_action = -gripper_action  # Invert for Robotiq convention
                gripper_action = min(gripper_action, 0.8)  # Limit max close
            else:
                # Actual range
                gripper_action = gripper_low + (gripper_high - gripper_low) * gripper_state
            
            return gripper_action
        
        # Fallback
        return 1.0 - 2.0 * gripper_state
    
    def _get_gripper_controller(self, controller, arm_idx: int = 0):
        """Get gripper controller from combined controller."""
        if hasattr(controller, 'controllers') and isinstance(controller.controllers, dict):
            # Look for gripper controller
            gripper_keys = ['gripper', f'gripper{arm_idx + 1}', f'gripper_{arm_idx}']
            for key in gripper_keys:
                if key in controller.controllers:
                    return controller.controllers[key]
        return None


