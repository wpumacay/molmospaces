"""
GELLO Auto-Calibration Module

Based on the official gello_software calibration approach:
https://github.com/wuphilipp/gello_software

The calibration process:
1. User places GELLO in a known position (e.g., all zeros)
2. System reads raw Dynamixel angles
3. System searches for offsets (multiples of π/2) that minimize error
4. Formula: calibrated = joint_sign * (raw - offset)

Joint Signs Reference (from official repo):
- UR: [1, 1, -1, 1, 1, 1]
- Panda: [1, -1, 1, 1, 1, -1, 1]
- xArm: [1, 1, 1, 1, 1, 1, 1]
- YAM: [1, -1, -1, -1, 1, 1]
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np


# Default joint signs for different robot types (from official gello_software)
DEFAULT_JOINT_SIGNS: Dict[str, Tuple[int, ...]] = {
    "panda": (1, -1, 1, 1, 1, -1, 1),
    "fr3": (1, -1, 1, 1, 1, -1, 1),
    "xarm": (1, 1, 1, 1, 1, 1, 1),
    "ur": (1, 1, -1, 1, 1, 1),
    "yam": (1, -1, -1, -1, 1, 1),
    "gen3_lite": (1, 1, 1, 1, 1, 1),
}

# Default number of joints for different robot types
DEFAULT_NUM_JOINTS: Dict[str, int] = {
    "panda": 7,
    "fr3": 7,
    "xarm": 7,
    "ur": 6,
    "yam": 6,
    "gen3_lite": 6,
}


@dataclass
class CalibrationResult:
    """Result of GELLO calibration."""
    serial_id: str
    robot_type: str
    joint_ids: Tuple[int, ...]
    joint_offsets: Tuple[float, ...]
    joint_signs: Tuple[int, ...]
    gripper_config: Tuple[int, float, float]  # (id, open_deg, close_deg)
    calibration_pose: str  # Name of the calibration pose used
    timestamp: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()
    
    def to_gello_config(self):
        """Convert to GelloDeviceConfig."""
        from molmo_spaces_maniskill.teleop.configs.gello_configs import GelloDeviceConfig
        return GelloDeviceConfig(
            name=f"{self.robot_type} GELLO (calibrated)",
            serial_id=self.serial_id,
            robot_type=self.robot_type,
            joint_ids=self.joint_ids,
            joint_offsets=self.joint_offsets,
            joint_signs=self.joint_signs,
            gripper_config=self.gripper_config,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "serial_id": self.serial_id,
            "robot_type": self.robot_type,
            "joint_ids": list(self.joint_ids),
            "joint_offsets": list(self.joint_offsets),
            "joint_signs": list(self.joint_signs),
            "gripper_config": list(self.gripper_config),
            "calibration_pose": self.calibration_pose,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        """Create from dictionary."""
        return cls(
            serial_id=data["serial_id"],
            robot_type=data["robot_type"],
            joint_ids=tuple(data["joint_ids"]),
            joint_offsets=tuple(data["joint_offsets"]),
            joint_signs=tuple(data["joint_signs"]),
            gripper_config=tuple(data["gripper_config"]),
            calibration_pose=data.get("calibration_pose", "unknown"),
            timestamp=data.get("timestamp", ""),
            notes=data.get("notes", ""),
        )
    
    def print_config(self):
        """Print configuration in a format suitable for gello_agent.py"""
        print("\n" + "=" * 60)
        print("Add this to gello_agent.py PORT_CONFIG_MAP:")
        print("=" * 60)
        print(f'    "{self._get_port_pattern()}": DynamixelRobotConfig(')
        print(f'        joint_ids={self.joint_ids},')
        print(f'        joint_offsets=(')
        for i, offset in enumerate(self.joint_offsets):
            pi_mult = int(np.round(offset / (np.pi / 2)))
            print(f'            {pi_mult}*np.pi/2,  # {offset:.3f}')
        print(f'        ),')
        print(f'        joint_signs={self.joint_signs},')
        print(f'        gripper_config={self.gripper_config},')
        print(f'    ),')
        print("=" * 60)
    
    def _get_port_pattern(self) -> str:
        return f"/dev/serial/by-id/usb-FTDI_*_{self.serial_id}-if00-port0"


class GelloCalibrator:
    """
    GELLO calibration tool following the official gello_software approach.
    
    The calibration finds joint offsets by:
    1. Reading raw Dynamixel positions when GELLO is at a known pose
    2. Searching for offsets (multiples of π/2) that minimize error
    
    Formula: calibrated = joint_sign * (raw - offset)
    Therefore: offset = raw - calibrated / joint_sign
    
    Example:
        >>> calibrator = GelloCalibrator(port="/dev/serial/by-id/usb-FTDI_...")
        >>> 
        >>> # Place GELLO at zero position, then:
        >>> result = calibrator.calibrate(
        ...     start_joints=[0, 0, 0, 0, 0, 0, 0],
        ...     robot_type="panda"
        ... )
        >>> result.print_config()
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 57600,
        joint_ids: Optional[List[int]] = None,
        gripper_id: Optional[int] = None,
    ):
        """
        Initialize calibrator.
        
        Args:
            port: Serial port path
            baudrate: Communication baudrate (default 57600)
            joint_ids: Optional list of Dynamixel joint IDs (if None, uses 1,2,3...)
            gripper_id: Optional gripper Dynamixel ID (if None, uses num_joints+1)
        """
        self.port = port
        self._baudrate = baudrate
        self._driver = None
        self._serial_id = self._extract_serial_id(port)
        self._custom_joint_ids = joint_ids
        self._custom_gripper_id = gripper_id
    
    def _extract_serial_id(self, port: str) -> str:
        """Extract serial ID from port path."""
        import re
        match = re.search(r'([A-Z0-9]{7,8})-if00', port)
        return match.group(1) if match else "UNKNOWN"
    
    def _connect(self, num_joints: int, has_gripper: bool = True):
        """Connect to Dynamixel servos."""
        from gello.dynamixel.driver import DynamixelDriver
        
        # Use custom joint_ids if provided, otherwise default to 1,2,3...
        if self._custom_joint_ids is not None:
            joint_ids = list(self._custom_joint_ids)
            if has_gripper:
                gripper_id = self._custom_gripper_id if self._custom_gripper_id else joint_ids[-1] + 1
                joint_ids = joint_ids + [gripper_id]
        else:
            total_joints = num_joints + (1 if has_gripper else 0)
            joint_ids = list(range(1, total_joints + 1))
        
        self._driver = DynamixelDriver(
            ids=joint_ids,
            port=self.port,
            baudrate=self._baudrate,
        )
        print(f"✓ Connected to Dynamixel servos: {joint_ids}")
        return joint_ids
    
    def _disconnect(self):
        """Disconnect from Dynamixel servos."""
        if self._driver is not None:
            try:
                self._driver.close()
            except:
                pass
            self._driver = None
    
    def _warmup(self, num_reads: int = 10):
        """Warmup by reading joint positions multiple times."""
        for _ in range(num_reads):
            self._driver.get_joints()
            time.sleep(0.01)
    
    def _get_raw_positions(self) -> np.ndarray:
        """Get raw joint positions from Dynamixel servos."""
        if self._driver is None:
            raise RuntimeError("Not connected to servos")
        return self._driver.get_joints()
    
    def calibrate(
        self,
        start_joints: Sequence[float],
        robot_type: str,
        joint_signs: Optional[Sequence[int]] = None,
        has_gripper: bool = True,
        pose_name: str = "zero",
    ) -> CalibrationResult:
        """
        Calibrate GELLO joint offsets.
        
        This follows the official gello_software approach:
        - Search for offsets in [-8π, 8π] at π/2 intervals
        - Find offset that minimizes error between calibrated and target
        
        Args:
            start_joints: Target joint positions (what GELLO should represent)
            robot_type: Type of robot (panda, xarm, ur, yam, etc.)
            joint_signs: Joint direction signs. If None, uses defaults for robot_type.
            has_gripper: Whether gripper is attached
            pose_name: Name of the calibration pose
            
        Returns:
            CalibrationResult with computed offsets
        """
        start_joints = np.asarray(start_joints)
        num_joints = len(start_joints)
        
        # Get joint signs
        if joint_signs is None:
            joint_signs = DEFAULT_JOINT_SIGNS.get(robot_type)
            if joint_signs is None:
                print(f"⚠️  No default joint signs for {robot_type}, using all +1")
                joint_signs = tuple([1] * num_joints)
            else:
                joint_signs = joint_signs[:num_joints]
        joint_signs = tuple(joint_signs)
        
        print(f"\nCalibration Settings:")
        print(f"  Robot type: {robot_type}")
        print(f"  Num joints: {num_joints}")
        print(f"  Joint signs: {joint_signs}")
        print(f"  Target pose: {start_joints}")
        print(f"  Has gripper: {has_gripper}")
        
        try:
            joint_ids = self._connect(num_joints, has_gripper)
            self._warmup()
            
            # Get current raw positions
            curr_joints = self._get_raw_positions()
            print(f"\nRaw positions: {[f'{x:.3f}' for x in curr_joints[:num_joints]]}")
            
            # Find best offsets using brute force search (official approach)
            best_offsets = self._find_best_offsets(
                curr_joints[:num_joints],
                start_joints,
                joint_signs,
            )
            
            # Get gripper config
            if has_gripper:
                gripper_id = num_joints + 1
                gripper_raw = curr_joints[num_joints]
                gripper_deg = np.rad2deg(gripper_raw)
                # Estimate open/close based on current position
                # These are rough estimates - user should verify
                gripper_config = (gripper_id, gripper_deg, gripper_deg - 42)
                print(f"\nGripper (estimated):")
                print(f"  Current: {gripper_deg:.1f}°")
                print(f"  Open: {gripper_config[1]:.1f}°")
                print(f"  Close: {gripper_config[2]:.1f}°")
            else:
                gripper_config = (num_joints + 1, 0, 0)
            
            # Verify calibration
            self._verify_calibration(curr_joints[:num_joints], best_offsets, joint_signs, start_joints)
            
            result = CalibrationResult(
                serial_id=self._serial_id,
                robot_type=robot_type,
                joint_ids=tuple(range(1, num_joints + 1)),
                joint_offsets=tuple(best_offsets),
                joint_signs=joint_signs,
                gripper_config=gripper_config,
                calibration_pose=pose_name,
                notes=f"Calibrated using official gello_software method",
            )
            
            return result
            
        finally:
            self._disconnect()
    
    def _find_best_offsets(
        self,
        raw_positions: np.ndarray,
        target_positions: np.ndarray,
        joint_signs: Tuple[int, ...],
    ) -> List[float]:
        """
        Find best offsets using brute force search.
        
        Following official gello_software approach:
        - Search in [-8π, 8π] at π/2 intervals
        - Find offset that minimizes |joint_sign * (raw - offset) - target|
        """
        best_offsets = []
        
        # Search range: -8π to 8π at π/2 intervals = 33 values
        search_offsets = np.linspace(-8 * np.pi, 8 * np.pi, 8 * 4 + 1)
        
        for i in range(len(raw_positions)):
            best_offset = 0
            best_error = float('inf')
            
            for offset in search_offsets:
                # Formula: calibrated = joint_sign * (raw - offset)
                calibrated = joint_signs[i] * (raw_positions[i] - offset)
                error = abs(calibrated - target_positions[i])
                
                if error < best_error:
                    best_error = error
                    best_offset = offset
            
            best_offsets.append(best_offset)
        
        # Print results
        print(f"\nComputed offsets:")
        print(f"  Raw values: {[f'{x:.3f}' for x in best_offsets]}")
        print(f"  As π/2 multiples: [" + 
              ", ".join([f"{int(np.round(x/(np.pi/2)))}*π/2" for x in best_offsets]) + 
              "]")
        
        return best_offsets
    
    def _verify_calibration(
        self,
        raw_positions: np.ndarray,
        offsets: List[float],
        joint_signs: Tuple[int, ...],
        target_positions: np.ndarray,
    ):
        """Verify calibration by computing calibrated positions."""
        calibrated = []
        for i in range(len(raw_positions)):
            cal = joint_signs[i] * (raw_positions[i] - offsets[i])
            calibrated.append(cal)
        
        errors = np.abs(np.array(calibrated) - target_positions)
        
        print(f"\nVerification:")
        print(f"  Calibrated: {[f'{x:.3f}' for x in calibrated]}")
        print(f"  Target:     {[f'{x:.3f}' for x in target_positions]}")
        print(f"  Errors:     {[f'{x:.4f}' for x in errors]}")
        print(f"  Max error:  {np.max(errors):.4f} rad ({np.rad2deg(np.max(errors)):.2f}°)")
    
    def calibrate_gripper_interactive(
        self,
        num_joints: int = 7,
    ) -> Tuple[int, float, float]:
        """
        Interactive gripper calibration.
        
        Guides user to move gripper to open and closed positions.
        
        Returns:
            Tuple of (gripper_id, open_position_deg, close_position_deg)
        """
        try:
            self._connect(num_joints, has_gripper=True)
            self._warmup()
            
            print("\n" + "=" * 50)
            print("Gripper Calibration")
            print("=" * 50)
            
            # Open position
            input("\n1. Move gripper to FULLY OPEN position, then press Enter...")
            raw = self._get_raw_positions()
            open_deg = np.rad2deg(raw[-1])
            print(f"   Open position: {open_deg:.2f}°")
            
            # Closed position
            input("\n2. Move gripper to FULLY CLOSED position, then press Enter...")
            raw = self._get_raw_positions()
            close_deg = np.rad2deg(raw[-1])
            print(f"   Closed position: {close_deg:.2f}°")
            
            gripper_id = num_joints + 1
            print(f"\n✓ Gripper config: ({gripper_id}, {open_deg:.2f}, {close_deg:.2f})")
            
            return (gripper_id, open_deg, close_deg)
            
        finally:
            self._disconnect()


def calibrate_gello_interactive(
    port: str,
    robot_type: str,
    start_joints: Optional[Sequence[float]] = None,
) -> CalibrationResult:
    """
    Full interactive GELLO calibration.
    
    Args:
        port: Serial port path
        robot_type: Type of robot
        start_joints: Target joint positions. If None, uses zeros.
    
    Returns:
        CalibrationResult with all calibration data
    """
    calibrator = GelloCalibrator(port)
    
    print("\n" + "=" * 60)
    print("GELLO Calibration (Official Method)")
    print("=" * 60)
    print(f"\nPort: {port}")
    print(f"Robot type: {robot_type}")
    
    # Get number of joints
    num_joints = DEFAULT_NUM_JOINTS.get(robot_type, 7)
    
    # Get target joints
    if start_joints is None:
        start_joints = [0.0] * num_joints
    
    print(f"\nINSTRUCTIONS:")
    print(f"1. Position your GELLO so that it represents: {start_joints}")
    print(f"2. For most robots, this means the 'zero' or 'home' position")
    print(f"3. Ensure all Dynamixel motors are powered")
    
    input("\nPress Enter when GELLO is in position...")
    
    # Calibrate
    result = calibrator.calibrate(
        start_joints=start_joints,
        robot_type=robot_type,
        has_gripper=True,
        pose_name="zero",
    )
    
    # Ask about gripper calibration
    print("\n" + "-" * 40)
    do_gripper = input("Calibrate gripper interactively? (y/N): ").lower() == 'y'
    if do_gripper:
        gripper_config = calibrator.calibrate_gripper_interactive(num_joints)
        result = CalibrationResult(
            serial_id=result.serial_id,
            robot_type=result.robot_type,
            joint_ids=result.joint_ids,
            joint_offsets=result.joint_offsets,
            joint_signs=result.joint_signs,
            gripper_config=gripper_config,
            calibration_pose=result.calibration_pose,
            notes="Fully calibrated with interactive gripper",
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    result.print_config()
    
    return result


def save_calibration(result: CalibrationResult, path: Optional[str] = None) -> str:
    """
    Save calibration result to JSON file.
    
    Args:
        result: CalibrationResult to save
        path: File path. If None, saves to ~/.mani_skill/gello_calibrations/
        
    Returns:
        Path to saved file
    """
    if path is None:
        calib_dir = Path.home() / ".mani_skill" / "gello_calibrations"
        calib_dir.mkdir(parents=True, exist_ok=True)
        path = str(calib_dir / f"{result.serial_id}.json")
    
    with open(path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"✓ Calibration saved to: {path}")
    return path


def load_calibration(serial_id_or_path: str) -> Optional[CalibrationResult]:
    """
    Load calibration from file.
    
    Args:
        serial_id_or_path: Either a serial ID (looks in default location)
                          or a full file path
                          
    Returns:
        CalibrationResult if found, None otherwise
    """
    # Check if it's a path
    if Path(serial_id_or_path).exists():
        path = Path(serial_id_or_path)
    else:
        # Look in default location
        path = Path.home() / ".mani_skill" / "gello_calibrations" / f"{serial_id_or_path}.json"
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return CalibrationResult.from_dict(data)


def get_calibration_for_device(port: str) -> Optional[CalibrationResult]:
    """
    Get calibration for a device by port.
    
    Extracts serial ID from port and looks up calibration.
    """
    import re
    match = re.search(r'([A-Z0-9]{7,8})-if00', port)
    if not match:
        return None
    
    serial_id = match.group(1)
    return load_calibration(serial_id)
