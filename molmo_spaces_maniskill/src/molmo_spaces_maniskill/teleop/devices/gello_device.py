"""
GELLO Device Wrapper

Provides a clean interface for interacting with GELLO hardware devices.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from molmo_spaces_maniskill.teleop.configs.gello_configs import GelloDeviceConfig, get_gello_config


@dataclass
class GelloState:
    """State returned from a GELLO device."""
    joint_positions: np.ndarray  # Joint positions (excluding gripper)
    gripper_state: float  # Gripper state normalized to [0, 1] (0=open, 1=closed)
    raw_joints: np.ndarray  # Raw joint values including gripper
    
    @property
    def full_state(self) -> np.ndarray:
        """Full state array [joints..., gripper]."""
        return np.concatenate([self.joint_positions, [self.gripper_state]])


class GelloDevice:
    """
    Wrapper for a single GELLO device.
    
    Provides a clean interface for reading joint states from GELLO hardware.
    
    Args:
        port: Serial port path (e.g., "/dev/serial/by-id/usb-FTDI_...-if00-port0")
        config: Optional GelloDeviceConfig. If None, will auto-detect from port.
        start_joints: Optional initial joint positions for the robot.
        
    Example:
        >>> device = GelloDevice("/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0")
        >>> state = device.get_state()
        >>> print(f"Joint positions: {state.joint_positions}")
        >>> print(f"Gripper: {state.gripper_state}")
    """
    
    def __init__(
        self,
        port: str,
        config: Optional[GelloDeviceConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        self.port = port
        
        # Get or auto-detect config
        if config is None:
            config = get_gello_config(port)
            if config is None:
                raise ValueError(
                    f"Unknown GELLO device at {port}. "
                    "Please provide a GelloDeviceConfig or register the device."
                )
        self.config = config
        
        # Create the underlying GELLO agent
        self._agent = self._create_agent(start_joints)
        
        print(f"✓ GELLO device connected: {config.name} at {port}")
    
    def _create_agent(self, start_joints: Optional[np.ndarray] = None):
        """Create the underlying GelloAgent."""
        from gello.agents.gello_agent import GelloAgent
        
        dynamixel_config = self.config.to_dynamixel_config()
        return GelloAgent(
            port=self.port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
        )
    
    def get_state(self) -> GelloState:
        """
        Get current state from the GELLO device.
        
        Returns:
            GelloState with joint positions and gripper state
        """
        # Get raw joint state from GELLO
        raw_joints = self._agent.act({})
        
        # Split into arm joints and gripper
        num_joints = self.config.num_joints
        joint_positions = raw_joints[:num_joints]
        gripper_state = raw_joints[num_joints]  # Already normalized to [0, 1]
        
        return GelloState(
            joint_positions=joint_positions,
            gripper_state=gripper_state,
            raw_joints=raw_joints,
        )
    
    def get_raw_joints(self) -> np.ndarray:
        """Get raw joint values including gripper."""
        return self._agent.act({})
    
    @property
    def name(self) -> str:
        """Device name."""
        return self.config.name
    
    @property
    def robot_type(self) -> str:
        """Robot type this GELLO is designed for."""
        return self.config.robot_type
    
    @property
    def num_joints(self) -> int:
        """Number of arm joints (excluding gripper)."""
        return self.config.num_joints
    
    @property
    def arm_side(self) -> Optional[str]:
        """Arm side for dual-arm setups (None for single-arm)."""
        return self.config.arm_side
    
    def __repr__(self) -> str:
        return f"GelloDevice({self.name}, port={self.port})"

