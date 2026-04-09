"""
GELLO Device Manager

Handles automatic discovery and management of GELLO devices.
Supports automatic calibration loading from saved calibration files.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from molmo_spaces_maniskill.teleop.configs.gello_configs import (
    GelloDeviceConfig,
    KNOWN_GELLO_DEVICES,
    get_gello_config,
    register_gello_device,
)
from molmo_spaces_maniskill.teleop.configs.robot_configs import (
    RobotTeleopConfig,
    get_robot_config,
)
from molmo_spaces_maniskill.teleop.devices.gello_device import GelloDevice


@dataclass
class DiscoveredDevice:
    """Information about a discovered GELLO device."""
    port: str
    serial_id: str
    config: Optional[GelloDeviceConfig]
    calibration_source: str = "builtin"  # "builtin", "calibration_file", or "unknown"
    
    @property
    def is_known(self) -> bool:
        """Whether this device has a known configuration."""
        return self.config is not None
    
    @property
    def is_calibrated(self) -> bool:
        """Whether this device has calibration data (from file or builtin)."""
        return self.config is not None
    
    @property
    def robot_type(self) -> Optional[str]:
        """Robot type if known."""
        return self.config.robot_type if self.config else None
    
    @property
    def name(self) -> str:
        """Device name or serial ID if unknown."""
        return self.config.name if self.config else f"Unknown ({self.serial_id})"


class GelloDeviceManager:
    """
    Manages discovery and connection of GELLO devices.
    
    Features:
    - Automatic scanning of /dev/serial/by-id/ for FTDI devices
    - Matching devices to known configurations
    - Loading calibration from saved files (~/.mani_skill/gello_calibrations/)
    - Creating GelloDevice instances for discovered devices
    - Support for single and dual-arm setups
    
    Example:
        >>> manager = GelloDeviceManager()
        >>> devices = manager.discover()
        >>> print(f"Found {len(devices)} GELLO device(s)")
        >>> 
        >>> # Auto-connect for a specific robot
        >>> gello_devices = manager.connect_for_robot("panda")
        
        >>> # Discover with auto-calibration loading
        >>> manager = GelloDeviceManager(auto_load_calibration=True)
        >>> devices = manager.discover()  # Will load calibration files if available
    """
    
    SERIAL_DIR = Path("/dev/serial/by-id")
    FTDI_PATTERNS = ["FTDI", "FT232R"]  # Common FTDI chip identifiers
    CALIBRATION_DIR = Path.home() / ".mani_skill" / "gello_calibrations"
    
    def __init__(self, auto_load_calibration: bool = True):
        """
        Initialize device manager.
        
        Args:
            auto_load_calibration: If True, automatically load calibration files
                                  for discovered devices
        """
        self._connected_devices: Dict[str, GelloDevice] = {}
        self._auto_load_calibration = auto_load_calibration
    
    def discover(self) -> List[DiscoveredDevice]:
        """
        Scan for connected GELLO devices.
        
        If auto_load_calibration is enabled, will also check for saved
        calibration files and load them for unknown devices.
        
        Returns:
            List of DiscoveredDevice instances
        """
        devices = []
        
        if not self.SERIAL_DIR.exists():
            print("⚠️  /dev/serial/by-id/ not found. No serial devices detected.")
            return devices
        
        for port_path in self.SERIAL_DIR.iterdir():
            port_name = port_path.name
            
            # Check if it's an FTDI device (GELLO uses FTDI chips)
            if not any(pattern in port_name for pattern in self.FTDI_PATTERNS):
                continue
            
            # Extract serial ID
            serial_id = self._extract_serial_id(port_name)
            if not serial_id:
                continue
            
            # Get configuration - check multiple sources
            config, source = self._get_device_config(serial_id)
            
            devices.append(DiscoveredDevice(
                port=str(port_path),
                serial_id=serial_id,
                config=config,
                calibration_source=source,
            ))
        
        return devices
    
    def _get_device_config(self, serial_id: str) -> Tuple[Optional[GelloDeviceConfig], str]:
        """
        Get device configuration from various sources.
        
        Priority:
        1. Built-in known devices
        2. Saved calibration files
        3. Unknown
        
        Returns:
            Tuple of (config, source) where source is "builtin", "calibration_file", or "unknown"
        """
        # Check built-in configurations first
        if serial_id in KNOWN_GELLO_DEVICES:
            return KNOWN_GELLO_DEVICES[serial_id], "builtin"
        
        # Check for saved calibration file
        if self._auto_load_calibration:
            config = self._load_calibration_file(serial_id)
            if config is not None:
                # Register it so it's available for future use
                register_gello_device(config)
                return config, "calibration_file"
        
        return None, "unknown"
    
    def _load_calibration_file(self, serial_id: str) -> Optional[GelloDeviceConfig]:
        """Load calibration from file if it exists."""
        calib_path = self.CALIBRATION_DIR / f"{serial_id}.json"
        
        if not calib_path.exists():
            return None
        
        try:
            from molmo_spaces_maniskill.teleop.calibration import load_calibration
            result = load_calibration(str(calib_path))
            if result is not None:
                print(f"✓ Loaded calibration for {serial_id} from {calib_path}")
                return result.to_gello_config()
        except Exception as e:
            print(f"⚠️  Failed to load calibration for {serial_id}: {e}")
        
        return None
    
    def _extract_serial_id(self, port_name: str) -> Optional[str]:
        """
        Extract serial ID from port name.
        
        Examples:
            "usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0" -> "FT3M9NVB"
            "usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0" -> "A50285BI"
        """
        # Pattern matches serial IDs like FT3M9NVB, A50285BI, etc.
        match = re.search(r'([A-Z0-9]{7,8})-if00', port_name)
        return match.group(1) if match else None
    
    def print_discovered_devices(self) -> None:
        """Print information about discovered devices."""
        devices = self.discover()
        
        if not devices:
            print("No GELLO devices found.")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(devices)} GELLO device(s):")
        print(f"{'='*60}")
        
        for i, device in enumerate(devices, 1):
            if device.is_known:
                if device.calibration_source == "calibration_file":
                    status = "✓ Calibrated (from file)"
                else:
                    status = "✓ Known (builtin)"
            else:
                status = "? Unknown - needs calibration"
            
            print(f"\n[{i}] {device.name}")
            print(f"    Serial ID: {device.serial_id}")
            print(f"    Port: {device.port}")
            print(f"    Status: {status}")
            if device.config:
                print(f"    Robot Type: {device.robot_type}")
                if device.config.arm_side:
                    print(f"    Arm Side: {device.config.arm_side}")
        
        # Print hint for unknown devices
        unknown_devices = [d for d in devices if not d.is_known]
        if unknown_devices:
            print(f"\n{'─'*60}")
            print("To calibrate unknown devices, run:")
            print("  python -m mani_skill.teleop.calibration.cli --port <PORT>")
        
        print(f"\n{'='*60}\n")
    
    def connect(
        self,
        port: str,
        config: Optional[GelloDeviceConfig] = None,
        start_joints: Optional[list] = None,
    ) -> GelloDevice:
        """
        Connect to a specific GELLO device.
        
        Args:
            port: Serial port path
            config: Optional configuration (auto-detected if None)
            start_joints: Optional initial joint positions
            
        Returns:
            Connected GelloDevice instance
        """
        if port in self._connected_devices:
            return self._connected_devices[port]
        
        device = GelloDevice(port, config, start_joints)
        self._connected_devices[port] = device
        return device
    
    def connect_for_robot(
        self,
        robot_uid: str,
        start_joints: Optional[list] = None,
    ) -> List[GelloDevice]:
        """
        Automatically connect GELLO devices for a specific robot.
        
        Args:
            robot_uid: ManiSkill robot UID (e.g., "panda", "xlerobot")
            start_joints: Optional initial joint positions
            
        Returns:
            List of connected GelloDevice instances (one per arm)
            
        Raises:
            ValueError: If robot config not found or required devices not available
        """
        robot_config = get_robot_config(robot_uid)
        if robot_config is None:
            raise ValueError(f"Unknown robot: {robot_uid}")
        
        # Discover available devices
        discovered = self.discover()
        discovered_by_serial = {d.serial_id: d for d in discovered}
        
        connected_devices = []
        
        for arm in robot_config.arms:
            if arm.gello_serial_id is None:
                # Try to find a matching device by robot type
                matching = [d for d in discovered if d.config and 
                           d.config.robot_type == robot_uid.split('_')[0]]
                if not matching:
                    raise ValueError(
                        f"No GELLO device configured for arm '{arm.name}' of robot '{robot_uid}'. "
                        f"Please set gello_serial_id in the robot config."
                    )
                device_info = matching[0]
            else:
                device_info = discovered_by_serial.get(arm.gello_serial_id)
                if device_info is None:
                    raise ValueError(
                        f"GELLO device {arm.gello_serial_id} for arm '{arm.name}' not found. "
                        f"Available devices: {list(discovered_by_serial.keys())}"
                    )
            
            device = self.connect(device_info.port, device_info.config, start_joints)
            connected_devices.append(device)
        
        return connected_devices
    
    def auto_connect(self) -> List[GelloDevice]:
        """
        Automatically connect to all discovered known GELLO devices.
        
        Returns:
            List of connected GelloDevice instances
        """
        discovered = self.discover()
        connected = []
        
        for device_info in discovered:
            if device_info.is_known:
                try:
                    device = self.connect(device_info.port, device_info.config)
                    connected.append(device)
                except Exception as e:
                    print(f"⚠️  Failed to connect to {device_info.name}: {e}")
        
        return connected
    
    def disconnect_all(self) -> None:
        """Disconnect all connected devices."""
        self._connected_devices.clear()
    
    @property
    def connected_devices(self) -> List[GelloDevice]:
        """List of currently connected devices."""
        return list(self._connected_devices.values())


def discover_gello_devices() -> List[DiscoveredDevice]:
    """
    Convenience function to discover GELLO devices.
    
    Returns:
        List of DiscoveredDevice instances
    """
    manager = GelloDeviceManager()
    return manager.discover()


def print_gello_devices() -> None:
    """Convenience function to print discovered GELLO devices."""
    manager = GelloDeviceManager()
    manager.print_discovered_devices()

