# ManiSkill Teleoperation Module

GELLO-based teleoperation for ManiSkill robots.

## Quick Start

```bash
# List connected GELLO devices
python -m mani_skill.teleop.calibration.cli --list

# Run teleoperation (auto-discover devices)
python teleop_gello.py -e PickCube-v1 -r fr3_robotiq_wristcam --auto-discover
```

## Supported Robots

| Robot | Joints | GELLO Serial |
|-------|--------|--------------|
| `panda` | 7 + gripper | A50285BI |
| `fr3_robotiq_wristcam` | 7 + gripper | A50285BI |
| `xlerobot` | 5+5 + 2 grippers | (dual-arm) |
| `ur_dual_arm` | 6+6 + 2 grippers | FT7WBEIA, FT7WBG6A |

## Calibrating New Devices

```bash
# 1. Place GELLO at zero position
# 2. Run calibration
python -m mani_skill.teleop.calibration.cli \
    --port /dev/serial/by-id/usb-FTDI_... \
    --robot panda \
    --start-joints "0,0,0,0,0,0"

# View saved calibrations
python -m mani_skill.teleop.calibration.cli --show-calibrations
```

Calibrations are saved to `~/.mani_skill/gello_calibrations/` and auto-loaded on next use.

## Python API

```python
from mani_skill.teleop.devices import GelloDeviceManager
from mani_skill.teleop.action_mappers import get_action_mapper

# Connect devices
manager = GelloDeviceManager()
devices = manager.connect_for_robot("fr3_robotiq_wristcam")

# Create action mapper
mapper = get_action_mapper("fr3_robotiq_wristcam", env=env)

# Teleoperation loop
while True:
    states = [d.get_state() for d in devices]
    action = mapper.map_action(states)
    env.step(action)
```

## Module Structure

```
teleop/
├── configs/          # Robot & GELLO configurations
├── devices/          # GELLO device management
├── wrappers/         # Recording wrapper
├── action_mappers/   # Device → robot action conversion
├── calibration/      # Auto-calibration tools
└── scripts/          # Entry point scripts
```

## Adding New Robots

```python
from mani_skill.teleop.configs.robot_configs import (
    RobotTeleopConfig, ArmConfig, register_robot_config
)

register_robot_config(RobotTeleopConfig(
    robot_uid="my_robot",
    arms=[ArmConfig(
        name="arm",
        num_joints=6,
        gripper_joints=1,
        ee_link_name="ee_link",
        gello_serial_id="XXXXXXXX",
    )],
    control_mode="pd_joint_pos",
))
```

## Joint Signs Reference

From [gello_software](https://github.com/wuphilipp/gello_software):

| Robot | Joint Signs |
|-------|-------------|
| Panda | `[1, -1, 1, 1, 1, -1, 1]` |
| xArm | `[1, 1, 1, 1, 1, 1, 1]` |
| UR | `[1, 1, -1, 1, 1, 1]` |
| YAM | `[1, -1, -1, -1, 1, 1]` |

