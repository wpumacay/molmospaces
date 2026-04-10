"""
ManiSkill Teleoperation Module

This module provides tools for teleoperating robots in ManiSkill environments
using various input devices (GELLO, SpaceMouse, etc.)

Main components:
- configs: Robot and device configuration management
- devices: Hardware device abstraction (GELLO, etc.)
- wrappers: Environment wrappers for recording
- action_mappers: Convert device input to robot actions
- calibration: Automatic GELLO calibration tools
"""

from molmo_spaces_maniskill.teleop.configs import (
    GelloDeviceConfig,
    RobotTeleopConfig,
    get_robot_config,
    get_gello_config,
)
from molmo_spaces_maniskill.teleop.devices import (
    GelloDevice,
    GelloDeviceManager,
)
from molmo_spaces_maniskill.teleop.wrappers import RecordEpisodeWithFreq
from molmo_spaces_maniskill.teleop.action_mappers import (
    BaseActionMapper,
    SingleArmActionMapper,
    DualArmActionMapper,
    get_action_mapper,
)

__all__ = [
    # Configs
    "GelloDeviceConfig",
    "RobotTeleopConfig", 
    "get_robot_config",
    "get_gello_config",
    # Devices
    "GelloDevice",
    "GelloDeviceManager",
    # Wrappers
    "RecordEpisodeWithFreq",
    # Action Mappers
    "BaseActionMapper",
    "SingleArmActionMapper",
    "DualArmActionMapper",
    "get_action_mapper",
]

# Lazy import calibration to avoid circular imports
def __getattr__(name):
    if name in ("GelloCalibrator", "CalibrationResult", "calibrate_gello_interactive"):
        from molmo_spaces_maniskill.teleop.calibration import (
            GelloCalibrator,
            CalibrationResult,
            calibrate_gello_interactive,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

