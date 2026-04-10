"""Device abstraction for teleoperation hardware."""

from molmo_spaces_maniskill.teleop.devices.base_device import BaseTeleopDevice, DeviceState
from molmo_spaces_maniskill.teleop.devices.gello_device import GelloDevice, GelloState
from molmo_spaces_maniskill.teleop.devices.device_manager import GelloDeviceManager

# SO101 is optional (requires lerobot)
try:
    from molmo_spaces_maniskill.teleop.devices.so101_device import SO101Device, SO101State, SO101Config
    SO101_AVAILABLE = True
except ImportError:
    SO101_AVAILABLE = False
    SO101Device = None
    SO101State = None
    SO101Config = None

__all__ = [
    # Base classes
    "BaseTeleopDevice",
    "DeviceState",
    # GELLO
    "GelloDevice", 
    "GelloState",
    "GelloDeviceManager",
    # SO101 (optional)
    "SO101Device",
    "SO101State", 
    "SO101Config",
    "SO101_AVAILABLE",
]

