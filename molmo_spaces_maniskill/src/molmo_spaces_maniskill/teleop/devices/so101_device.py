"""
LeRobot SO101 Leader Device Wrapper

Provides a clean interface for interacting with SO101 Leader arm hardware.
Compatible with the LeRobot ecosystem.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np

# Try to import LeRobot components
LEROBOT_AVAILABLE = False
LEROBOT_IMPORT_ERROR = None

try:
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_IMPORT_ERROR = str(e)
except Exception as e:
    LEROBOT_IMPORT_ERROR = f"Error loading LeRobot: {e}"


@dataclass
class SO101Config:
    """Configuration for SO101 Leader device."""
    name: str = "SO101 Leader"
    port: str = ""
    use_degrees: bool = False  # If True, returns degrees; otherwise range [-100, 100]
    
    # Joint mapping to standard names
    joint_names: tuple = (
        "shoulder_pan",
        "shoulder_lift", 
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    
    # Number of arm joints (excluding gripper)
    num_joints: int = 5
    
    # No offsets applied to joint mapping - direct mapping only
    joint_offsets: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    joint_signs: tuple = (1, 1, 1, 1, 1)
    
    # Alignment offsets: used ONLY for calculating alignment error with ghost
    # These compensate for difference between SO101 home pose and robot zero pose
    # Joints 4 and 5: when SO101 is at home (raw ~100), robot should be at 0
    alignment_offsets: tuple = (0.0, 0.0, 0.0, -np.pi/2, -np.pi/2)
    
    # Calibration file path (optional)
    calibration_path: Optional[str] = None


@dataclass 
class SO101State:
    """State returned from a SO101 device."""
    joint_positions: np.ndarray  # Joint positions in radians (5 DOF arm)
    gripper_state: float  # Gripper state normalized to [0, 1] (0=open, 1=closed)
    raw_joints: np.ndarray  # Raw joint values including gripper
    
    @property
    def full_state(self) -> np.ndarray:
        """Full state array [joints..., gripper]."""
        return np.concatenate([self.joint_positions, [self.gripper_state]])


class SO101Device:
    """
    Wrapper for SO101 Leader arm from LeRobot.
    
    Provides a clean interface for reading joint states, compatible with
    the ManiSkill teleop architecture.
    
    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0" or "/dev/ttyACM0")
        config: Optional SO101Config. If None, uses defaults.
        calibration: Optional calibration dict for motors.
        
    Example:
        >>> device = SO101Device("/dev/ttyACM0")
        >>> state = device.get_state()
        >>> print(f"Joint positions: {state.joint_positions}")
        >>> print(f"Gripper: {state.gripper_state}")
    """
    
    def __init__(
        self,
        port: str,
        config: Optional[SO101Config] = None,
        calibration: Optional[Dict] = None,
    ):
        if not LEROBOT_AVAILABLE:
            error_msg = (
                "LeRobot is not available.\n"
                f"Import error: {LEROBOT_IMPORT_ERROR}\n\n"
                "To fix this, try:\n"
                "  1. pip install lerobot\n"
                "  2. Or if lerobot is installed but has missing dependencies:\n"
                "     pip install accelerate\n"
                "  3. Or clone from: https://github.com/huggingface/lerobot"
            )
            raise ImportError(error_msg)
        
        self.port = port
        self.config = config or SO101Config(port=port)
        self.config.port = port
        self._calibration = calibration
        
        # Create motor bus
        self._bus = self._create_bus()
        self._connected = False
        
    def _create_bus(self) -> "FeetechMotorsBus":
        """Create the Feetech motor bus."""
        norm_mode_body = (
            MotorNormMode.DEGREES 
            if self.config.use_degrees 
            else MotorNormMode.RANGE_M100_100
        )
        
        return FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._calibration,
        )
    
    def connect(self, calibrate: bool = False) -> None:
        """
        Connect to the SO101 device.
        
        Args:
            calibrate: If True, FORCE run calibration (ignore existing calibration).
        """
        if self._connected:
            print(f"Warning: {self.name} already connected")
            return
        
        # If calibrate=True, skip loading saved calibration and force recalibration
        if calibrate:
            print("Calibration requested, ignoring saved calibration...")
            self._calibration = None
        else:
            # Try to load saved calibration
            if self._calibration is None:
                self._calibration = self._load_calibration()
                if self._calibration is not None:
                    print(f"✓ Loaded saved calibration from {self._get_calibration_path()}")
                    # Recreate bus with calibration
                    self._bus = self._create_bus()
            
        self._bus.connect()
        
        if calibrate or not self._bus.is_calibrated:
            print(f"Running calibration...")
            self.calibrate()
        elif not self._bus.is_calibrated:
            print(f"Warning: SO101 not calibrated. Run with --calibrate to calibrate.")
        
        self._configure()
        self._connected = True
        print(f"✓ SO101 device connected: {self.name} at {self.port}")
    
    def _configure(self) -> None:
        """Configure motors for position reading."""
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
    
    def calibrate(self) -> None:
        """Run interactive calibration."""
        print(f"\nRunning calibration of {self.name}")
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move SO101 to the middle of its range of motion and press ENTER....")
        homing_offsets = self._bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self._bus.record_ranges_of_motion()

        self._calibration = {}
        for motor, m in self._bus.motors.items():
            self._calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self._bus.write_calibration(self._calibration)
        
        # Save calibration to file
        self._save_calibration()
        print("Calibration complete!")
    
    def _get_calibration_path(self) -> Path:
        """Get path to calibration file."""
        # Save in ~/.maniskill/teleop/so101_calibration.json
        calib_dir = Path.home() / ".maniskill" / "teleop"
        calib_dir.mkdir(parents=True, exist_ok=True)
        # Use port name to distinguish multiple devices
        port_name = self.port.replace("/", "_").replace("dev_", "")
        return calib_dir / f"so101_calibration_{port_name}.json"
    
    def _get_offset_calibration_path(self) -> Path:
        """Get path to offset calibration file (for robot alignment)."""
        calib_dir = Path.home() / ".maniskill" / "teleop"
        calib_dir.mkdir(parents=True, exist_ok=True)
        port_name = self.port.replace("/", "_").replace("dev_", "")
        return calib_dir / f"so101_offset_calibration_{port_name}.json"
    
    def calibrate_offsets(self, target_joints: np.ndarray) -> None:
        """
        Calibrate joint offsets by placing SO101 at target pose.
        
        Args:
            target_joints: Target joint positions in radians (5 DOF)
        """
        if not self._connected:
            raise RuntimeError("SO101 not connected. Call connect() first.")
        
        print("\n" + "="*60)
        print("SO101 Offset Calibration")
        print("="*60)
        print(f"Target joints: {[f'{j:.3f}' for j in target_joints]}")
        print("\nMove SO101 to match the target pose (ghost), then press ENTER...")
        input()
        
        # Read current raw values
        action = self._bus.sync_read("Present_Position")
        raw_values = np.array([
            action.get("shoulder_pan", 0),
            action.get("shoulder_lift", 0),
            action.get("elbow_flex", 0),
            action.get("wrist_flex", 0),
            action.get("wrist_roll", 0),
        ])
        
        print(f"Raw values at target: {[f'{v:.1f}' for v in raw_values]}")
        
        # Calculate offsets
        # raw_scaled = raw / 100 * (pi/2)
        # output = sign * raw_scaled + offset
        # We want output = target, so:
        # offset = target - sign * raw_scaled
        # Assuming sign = 1 for now
        raw_scaled = raw_values / 100.0 * (np.pi / 2)
        offsets = target_joints - raw_scaled
        
        print(f"Computed offsets: {[f'{o:.3f}' for o in offsets]}")
        
        # Update config
        self.config.joint_offsets = tuple(offsets.tolist())
        
        # Save to file
        offset_path = self._get_offset_calibration_path()
        offset_data = {
            "joint_offsets": list(offsets),
            "joint_signs": list(self.config.joint_signs),
            "target_joints": list(target_joints),
            "raw_values_at_target": list(raw_values),
        }
        with open(offset_path, "w") as f:
            json.dump(offset_data, f, indent=2)
        
        print(f"✓ Offset calibration saved to {offset_path}")
        print("="*60)
    
    def load_offset_calibration(self) -> bool:
        """Load offset calibration if it exists. Returns True if loaded."""
        offset_path = self._get_offset_calibration_path()
        if not offset_path.exists():
            return False
        
        try:
            with open(offset_path, "r") as f:
                data = json.load(f)
            
            self.config.joint_offsets = tuple(data["joint_offsets"])
            self.config.joint_signs = tuple(data["joint_signs"])
            print(f"✓ Loaded offset calibration from {offset_path}")
            print(f"  Offsets: {[f'{o:.3f}' for o in self.config.joint_offsets]}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load offset calibration: {e}")
            return False
    
    def _save_calibration(self) -> None:
        """Save calibration to file."""
        if self._calibration is None:
            return
        
        calib_path = self._get_calibration_path()
        
        # Convert MotorCalibration objects to dicts
        calib_data = {}
        for motor, cal in self._calibration.items():
            calib_data[motor] = {
                "id": cal.id,
                "drive_mode": cal.drive_mode,
                "homing_offset": cal.homing_offset,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
            }
        
        with open(calib_path, "w") as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"✓ Calibration saved to {calib_path}")
    
    def _load_calibration(self) -> Optional[Dict]:
        """Load calibration from file if it exists."""
        calib_path = self._get_calibration_path()
        
        if not calib_path.exists():
            return None
        
        try:
            with open(calib_path, "r") as f:
                calib_data = json.load(f)
            
            # Convert dicts back to MotorCalibration objects
            calibration = {}
            for motor, data in calib_data.items():
                calibration[motor] = MotorCalibration(
                    id=data["id"],
                    drive_mode=data["drive_mode"],
                    homing_offset=data["homing_offset"],
                    range_min=data["range_min"],
                    range_max=data["range_max"],
                )
            
            return calibration
        except Exception as e:
            print(f"Warning: Failed to load calibration from {calib_path}: {e}")
            return None
    
    def get_state(self) -> SO101State:
        """
        Get current state from the SO101 device.
        
        Returns:
            SO101State with joint positions and gripper state
        """
        if not self._connected:
            raise RuntimeError("SO101 not connected. Call connect() first.")
        
        # Read calibrated motor positions
        # LeRobot returns normalized values: [-100, 100] for arm joints, [0, 100] for gripper
        # The [-100, 100] range corresponds to the calibrated range of motion
        action = self._bus.sync_read("Present_Position")
        raw_values = np.array([
            action.get("shoulder_pan", 0),
            action.get("shoulder_lift", 0),
            action.get("elbow_flex", 0),
            action.get("wrist_flex", 0),
            action.get("wrist_roll", 0),
            action.get("gripper", 0),
        ])
        
        # Convert arm joints from [-100, 100] to radians
        # Simple linear mapping: [-100, 100] -> [-pi/2, pi/2]
        joint_positions = raw_values[:5].copy()
        joint_positions = joint_positions / 100.0 * (np.pi / 2)
        
        # Apply calibration offsets and signs
        joint_positions = (
            np.array(self.config.joint_signs) * joint_positions + 
            np.array(self.config.joint_offsets)
        )
        
        # Gripper: [0, 100] -> [0, 1] (0=open, 1=closed)
        gripper_state = np.clip(raw_values[5] / 100.0, 0, 1)
        
        return SO101State(
            joint_positions=joint_positions,
            gripper_state=gripper_state,
            raw_joints=raw_values,
        )
    
    def _read_raw_positions(self) -> np.ndarray:
        """Read raw encoder positions without calibration."""
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                       "wrist_flex", "wrist_roll", "gripper"]
        raw_values = []
        
        for name in motor_names:
            motor = self._bus.motors[name]
            # Read raw Present_Position register
            value = self._bus.read("Present_Position", name)
            raw_values.append(value)
        
        return np.array(raw_values, dtype=np.float32)
    
    def get_raw_joints(self) -> np.ndarray:
        """Get raw joint values including gripper."""
        state = self.get_state()
        return state.raw_joints
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        if self._connected:
            self._bus.disconnect()
            self._connected = False
            print(f"{self.name} disconnected.")
    
    @property
    def name(self) -> str:
        """Device name."""
        return self.config.name
    
    @property
    def robot_type(self) -> str:
        """Robot type this device is designed for."""
        return "so101"
    
    @property
    def num_joints(self) -> int:
        """Number of arm joints (excluding gripper)."""
        return self.config.num_joints
    
    @property
    def arm_side(self) -> Optional[str]:
        """Arm side for dual-arm setups (None for single-arm)."""
        return None
    
    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._connected
    
    def __repr__(self) -> str:
        return f"SO101Device({self.name}, port={self.port})"
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_connected') and self._connected:
            self.disconnect()

