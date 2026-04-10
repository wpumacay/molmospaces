"""Base classes for teleoperation devices."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DeviceState:
    """Base class for state returned from a teleoperation device."""
    joint_positions: np.ndarray  # Joint positions (excluding gripper)
    gripper_state: float  # Gripper state normalized to [0, 1] (0=open, 1=closed)
    raw_joints: np.ndarray  # Raw joint values including gripper

    @property
    def full_state(self) -> np.ndarray:
        """Full state array [joints..., gripper]."""
        return np.concatenate([self.joint_positions, [self.gripper_state]])


class BaseTeleopDevice:
    """Abstract base class for a teleoperation device."""
    
    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_state(self) -> DeviceState:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def robot_type(self) -> str:
        raise NotImplementedError

    @property
    def num_joints(self) -> int:
        raise NotImplementedError

    @property
    def arm_side(self) -> Optional[str]:
        return None

