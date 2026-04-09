"""
GELLO Device Configurations

This module defines configurations for different GELLO devices.
Each GELLO device is identified by its USB serial ID.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GelloDeviceConfig:
    """
    Configuration for a single GELLO device.
    
    Attributes:
        name: Human-readable name for this device
        serial_id: USB serial ID (e.g., "A50285BI", "FT3M9NVB")
        robot_type: Type of robot this GELLO is designed for
        joint_ids: Dynamixel joint IDs (excluding gripper)
        joint_offsets: Joint offset values (multiples of pi/2)
        joint_signs: Joint direction signs (1 or -1)
        gripper_config: Tuple of (gripper_id, open_pos_deg, close_pos_deg)
        arm_side: For dual-arm setups, "left" or "right" (None for single-arm)
    """
    name: str
    serial_id: str
    robot_type: str
    joint_ids: Tuple[int, ...]
    joint_offsets: Tuple[float, ...]
    joint_signs: Tuple[int, ...]
    gripper_config: Tuple[int, float, float]
    arm_side: Optional[str] = None
    
    @property
    def num_joints(self) -> int:
        """Number of arm joints (excluding gripper)."""
        return len(self.joint_ids)
    
    @property
    def port_pattern(self) -> str:
        """Expected port pattern for this device."""
        return f"*{self.serial_id}*"
    
    def to_dynamixel_config(self):
        """
        Convert to DynamixelRobotConfig for gello library.
        
        Returns:
            DynamixelRobotConfig instance
        """
        # Import here to avoid circular dependency
        from gello.agents.gello_agent import DynamixelRobotConfig
        return DynamixelRobotConfig(
            joint_ids=self.joint_ids,
            joint_offsets=self.joint_offsets,
            joint_signs=self.joint_signs,
            gripper_config=self.gripper_config,
        )


# ============================================================================
# Known GELLO Device Configurations
# ============================================================================

KNOWN_GELLO_DEVICES: Dict[str, GelloDeviceConfig] = {
    # Panda / Franka GELLO (calibrated based on actual testing)
    # NOTE: Joint 3 offset changed from 0 to 2π based on alignment debug
    "A50285BI": GelloDeviceConfig(
        name="Panda GELLO",
        serial_id="A50285BI",
        robot_type="panda",
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            1 * np.pi / 2,  # 1.571 - joint 1
            2 * np.pi / 2,  # 3.142 - joint 2
            0.0,  # 6.283 - joint 3 (was 0, fixed based on debug)
            3 * np.pi / 2,  # 4.712 - joint 4
            2 * np.pi / 2,  # 3.142 - joint 5
            3 * np.pi / 2,  # 4.712 - joint 6
            2 * np.pi / 2,  # 3.142 - joint 7
        ),
        joint_signs=(1, 1, 1, 1, 1, -1, 1),
        gripper_config=(8, 159.62109375, 201.42109375),
    ),



    "FTAO9WPU": GelloDeviceConfig(
        name="YAM LEFT GELLO",
        serial_id="FTAO9WPU",
        robot_type="yam",
        joint_ids=(1, 2, 3, 4, 5, 6),  # Actual motor IDs
        joint_offsets=(
            np.pi * 1.5,       # joint 8:  2*π/2 = π
            np.pi,       # joint 9:  2*π/2 = π
            np.pi,       # joint 10: 2*π/2 = π
            np.pi,       # joint 11: 2*π/2 = π
            np.pi,       # joint 12: 2*π/2 = π
            np.pi,   # joint 13: 6*π/2 = 3π
        ),
        joint_signs=(1, -1, -1, -1, 1, -1),  # From calibration
        gripper_config=(7, 56, 114),  # ID=14, open=115°, close=73°
        arm_side="left",
    ),
    
    # YAM RIGHT GELLO (6-DOF) - Serial: FTAO9WCV
    # Actual motor IDs: 8, 9, 10, 11, 12, 13 (arm) + 14 (gripper)
    # Baudrate: 57600 (default in gello_software)
    # Calibrated via auto_calibrator.py on 2026-02-28
    # Raw positions at zero pose: [3.160, 3.139, 3.108, 3.188, 3.162, 9.400]
    # Max error: 0.046 rad (2.64°)
    "FTAO9WCV": GelloDeviceConfig(
        name="YAM RIGHT GELLO",
        serial_id="FTAO9WCV",
        robot_type="yam",
        joint_ids=(8, 9, 10, 11, 12, 13),  # Actual motor IDs
        joint_offsets=(
            np.pi,       # joint 8:  2*π/2 = π
            np.pi,       # joint 9:  2*π/2 = π
            np.pi,       # joint 10: 2*π/2 = π
            np.pi,       # joint 11: 2*π/2 = π
            np.pi,       # joint 12: 2*π/2 = π
            np.pi,   # joint 13: 6*π/2 = 3π
        ),
        joint_signs=(1, -1, 1, -1, 1, -1),  # From calibration
        gripper_config=(14, 56, 114),  # ID=14, open=115°, close=73°
        arm_side="right",
    ),

    
    # UR Left Arm GELLO
    "FT7WBEIA": GelloDeviceConfig(
        name="UR Left GELLO",
        serial_id="FT7WBEIA",
        robot_type="ur",
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            0,
            1 * np.pi / 2 + np.pi,
            np.pi / 2,
            np.pi / 2,
            np.pi - 2 * np.pi / 2,
            -1 * np.pi / 2 + 2 * np.pi,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 20, -22),
        arm_side="left",
    ),
    
    # UR Right Arm GELLO
    "FT7WBG6A": GelloDeviceConfig(
        name="UR Right GELLO",
        serial_id="FT7WBG6A",
        robot_type="ur",
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            np.pi,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            1 * np.pi,
            3 * np.pi / 2,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 286, 248),
        arm_side="right",
    ),
}


def get_gello_config(port: str) -> Optional[GelloDeviceConfig]:
    """
    Get GELLO configuration by port path.
    
    Args:
        port: Full port path (e.g., "/dev/serial/by-id/usb-FTDI_...-if00-port0")
        
    Returns:
        GelloDeviceConfig if found, None otherwise
    """
    for serial_id, config in KNOWN_GELLO_DEVICES.items():
        if serial_id in port:
            return config
    return None


def get_gello_config_by_serial(serial_id: str) -> Optional[GelloDeviceConfig]:
    """
    Get GELLO configuration by serial ID.
    
    Args:
        serial_id: USB serial ID (e.g., "A50285BI")
        
    Returns:
        GelloDeviceConfig if found, None otherwise
    """
    return KNOWN_GELLO_DEVICES.get(serial_id)


def register_gello_device(config: GelloDeviceConfig) -> None:
    """
    Register a new GELLO device configuration.
    
    Args:
        config: GelloDeviceConfig to register
    """
    KNOWN_GELLO_DEVICES[config.serial_id] = config

