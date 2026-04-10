#!/usr/bin/env python3
"""
GELLO Calibration CLI

Command-line tool for calibrating GELLO devices.
Based on the official gello_software calibration approach.

Usage:
    # List available devices
    python -m mani_skill.teleop.calibration.cli --list
    
    # Calibrate a device (interactive)
    python -m mani_skill.teleop.calibration.cli --port /dev/serial/by-id/usb-FTDI_... --robot panda
    
    # Calibrate with custom start joints
    python -m mani_skill.teleop.calibration.cli --port /dev/serial/by-id/usb-FTDI_... --robot panda \\
        --start-joints "0,0,0,0,0,0,0"
    
    # Show saved calibrations
    python -m mani_skill.teleop.calibration.cli --show-calibrations

Reference (official gello_software):
    https://github.com/wuphilipp/gello_software
"""

import argparse
import glob
from pathlib import Path
from typing import Optional, List
import numpy as np


def find_gello_ports() -> List[str]:
    """Auto-detect GELLO ports by looking for FTDI USB-Serial converters."""
    patterns = [
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_*",
        "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_*",
    ]
    
    ports = []
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    
    return sorted(set(ports))


def list_devices():
    """List all discovered GELLO devices."""
    from molmo_spaces_maniskill.teleop.devices import GelloDeviceManager
    
    manager = GelloDeviceManager(auto_load_calibration=True)
    manager.print_discovered_devices()
    
    # Also show raw ports
    ports = find_gello_ports()
    if ports:
        print("\nRaw FTDI ports found:")
        for port in ports:
            print(f"  {port}")


def show_calibrations():
    """Show all saved calibration files."""
    from molmo_spaces_maniskill.teleop.calibration import load_calibration
    
    calib_dir = Path.home() / ".mani_skill" / "gello_calibrations"
    
    if not calib_dir.exists():
        print("No calibration directory found.")
        print(f"Expected location: {calib_dir}")
        return
    
    files = list(calib_dir.glob("*.json"))
    
    if not files:
        print("No calibration files found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Saved Calibrations ({len(files)} files)")
    print(f"{'='*60}")
    
    for f in files:
        result = load_calibration(str(f))
        if result:
            print(f"\n[{f.stem}]")
            print(f"  Robot Type: {result.robot_type}")
            print(f"  Joint IDs: {result.joint_ids}")
            print(f"  Joint Signs: {result.joint_signs}")
            print(f"  Joint Offsets (π/2): [" + 
                  ", ".join([f"{int(np.round(x/(np.pi/2)))}" for x in result.joint_offsets]) + 
                  "]")
            print(f"  Gripper: id={result.gripper_config[0]}, " +
                  f"open={result.gripper_config[1]:.1f}°, close={result.gripper_config[2]:.1f}°")
            print(f"  Calibrated: {result.timestamp[:19]}")
    
    print(f"\n{'='*60}\n")


def calibrate(
    port: str,
    robot_type: str,
    start_joints: Optional[str] = None,
    output: Optional[str] = None,
    no_gripper: bool = False,
):
    """Run calibration."""
    from molmo_spaces_maniskill.teleop.calibration import (
        GelloCalibrator,
        calibrate_gello_interactive,
        save_calibration,
        DEFAULT_NUM_JOINTS,
    )
    from molmo_spaces_maniskill.teleop.configs.gello_configs import get_gello_config
    
    # Try to get existing config for this device (to get joint_ids)
    existing_config = get_gello_config(port)
    joint_ids = None
    gripper_id = None
    if existing_config:
        joint_ids = list(existing_config.joint_ids)
        gripper_id = existing_config.gripper_config[0]
        print(f"\n✓ Found existing config for {existing_config.name}")
        print(f"  Joint IDs: {joint_ids}")
        print(f"  Gripper ID: {gripper_id}")
    
    # Parse start joints if provided
    if start_joints:
        joints = [float(x.strip()) for x in start_joints.split(',')]
    else:
        # Default to zeros
        num_joints = len(joint_ids) if joint_ids else DEFAULT_NUM_JOINTS.get(robot_type, 7)
        joints = [0.0] * num_joints
    
    print(f"\n{'='*60}")
    print("GELLO Calibration")
    print(f"{'='*60}")
    print(f"Port: {port}")
    print(f"Robot Type: {robot_type}")
    print(f"Start Joints: {joints}")
    print(f"Has Gripper: {not no_gripper}")
    if joint_ids:
        print(f"Joint IDs: {joint_ids}")
        print(f"Gripper ID: {gripper_id}")
    
    # Run calibration
    calibrator = GelloCalibrator(port, joint_ids=joint_ids, gripper_id=gripper_id)
    
    print(f"\nINSTRUCTIONS:")
    print(f"1. Position your GELLO so it represents joint angles: {joints}")
    print(f"2. For zero calibration, this typically means straight/home position")
    print(f"3. Ensure all Dynamixel motors are powered and connected")
    
    input("\nPress Enter when GELLO is in position...")
    
    result = calibrator.calibrate(
        start_joints=joints,
        robot_type=robot_type,
        has_gripper=not no_gripper,
        pose_name="zero" if all(j == 0 for j in joints) else "custom",
    )
    
    # Interactive gripper calibration
    if not no_gripper:
        print("\n" + "-" * 40)
        do_gripper = input("Calibrate gripper interactively? (y/N): ").lower() == 'y'
        if do_gripper:
            gripper_config = calibrator.calibrate_gripper_interactive(len(joints))
            from molmo_spaces_maniskill.teleop.calibration import CalibrationResult
            result = CalibrationResult(
                serial_id=result.serial_id,
                robot_type=result.robot_type,
                joint_ids=result.joint_ids,
                joint_offsets=result.joint_offsets,
                joint_signs=result.joint_signs,
                gripper_config=gripper_config,
                calibration_pose=result.calibration_pose,
                notes="Calibrated with interactive gripper",
            )
    
    # Print config for gello_agent.py
    result.print_config()
    
    # Save calibration
    if output:
        save_path = save_calibration(result, output)
    else:
        save_path = save_calibration(result)
    
    print(f"\n✓ Calibration saved!")
    print(f"  File: {save_path}")
    print(f"\nThe device will now be auto-detected on next use.")


def main():
    parser = argparse.ArgumentParser(
        description="GELLO Calibration Tool (based on official gello_software)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Joint Signs Reference (from official gello_software):
  - Panda: [1, -1, 1, 1, 1, -1, 1]
  - xArm:  [1, 1, 1, 1, 1, 1, 1]
  - UR:    [1, 1, -1, 1, 1, 1]
  - YAM:   [1, -1, -1, -1, 1, 1]

Examples:
  # List devices
  python -m mani_skill.teleop.calibration.cli --list
  
  # Calibrate Panda GELLO at zero position
  python -m mani_skill.teleop.calibration.cli \\
      --port /dev/serial/by-id/usb-FTDI_... \\
      --robot panda
  
  # Calibrate with custom start joints
  python -m mani_skill.teleop.calibration.cli \\
      --port /dev/serial/by-id/usb-FTDI_... \\
      --robot ur \\
      --start-joints "0,-1.57,1.57,-1.57,-1.57,0"

Reference: https://github.com/wuphilipp/gello_software
        """
    )
    
    parser.add_argument("--list", "-l", action="store_true",
                       help="List discovered GELLO devices")
    parser.add_argument("--show-calibrations", "-s", action="store_true",
                       help="Show saved calibrations")
    parser.add_argument("--port", "-p", type=str,
                       help="GELLO device port")
    parser.add_argument("--robot", "-r", type=str,
                       choices=["panda", "fr3", "xarm", "ur", "yam", "gen3_lite"],
                       help="Robot type")
    parser.add_argument("--start-joints", "-j", type=str,
                       help="Start joint positions (comma-separated radians). Default: all zeros")
    parser.add_argument("--no-gripper", action="store_true",
                       help="No gripper attached")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path (default: ~/.mani_skill/gello_calibrations/)")
    
    args = parser.parse_args()
    
    if args.list:
        list_devices()
    elif args.show_calibrations:
        show_calibrations()
    elif args.port and args.robot:
        calibrate(
            port=args.port,
            robot_type=args.robot,
            start_joints=args.start_joints,
            output=args.output,
            no_gripper=args.no_gripper,
        )
    else:
        parser.print_help()
        print("\n⚠️  Please specify --list, --show-calibrations, or both --port and --robot")


if __name__ == "__main__":
    main()
