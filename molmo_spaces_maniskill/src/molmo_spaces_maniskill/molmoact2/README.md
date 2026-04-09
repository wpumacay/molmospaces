# MolmoAct2 Tasks — Teleoperation Guide

This guide explains how to use teleoperation hardware (GELLO arms, SO101 Leader) to collect demonstration trajectories for MolmoAct2 tasks in ManiSkill simulation.

## Overview

```
teleop/ ──────► action_mappers/ ──────► molmoact2/tasks/ ──────► demos/
 hardware          joint mapping          simulation env          saved H5
```

The teleoperation pipeline:
1. **Hardware** (GELLO/SO101) reads your arm's joint positions
2. **Action mapper** converts device state to robot action dimensions
3. **ManiSkill environment** executes the action and checks task success
4. **Recording wrapper** saves successful trajectories to H5 + video

---

## Available Tasks

### YAM Bimanual Tasks (dual-arm, GELLO ×2)

| Environment ID | Description |
|---|---|
| `BimanualYAMPickPlace-v1` | Pick cube with left arm, hand to right, place on goal |
| `BimanualYAMLiftPot-v1` | Collaboratively lift a pot with both arms |
| `BimanualYAMInsert-v1` | Two-arm insertion task |
| `BimanualYAMFlipPlate-v1` | Flip a plate using both arms |
| `BimanualYAMMicrowave-v1` | Open/close microwave with bimanual arms |

### DROID Single-Arm Tasks (FR3 + Robotiq, GELLO ×1)

| Environment ID | Description |
|---|---|
| `CubeStack-v1` | Stack colored cubes |
| `KitchenOpenDrawerPnPFork-v1` | Open drawer, pick and place fork |
| `KitchenOpenDrawerMem-v1` | Open drawer (memory variant) |
| `KitchenPourBottleCup-v1` | Pour from bottle into cup |
| `AI2LabOpenMicrowave-v1` | Open microwave door |
| `AI2LabOpenOven-v1` | Open oven door |

### SO100 Tasks (SO100/SO101 arm)

| Environment ID | Description |
|---|---|
| `SO100CloseMicrowave-v1` | Close microwave door |
| `SO100PushCubeSlot-v1` | Push cube into slot |

---

## Hardware Setup

### GELLO Devices

GELLO arms connect via USB (FTDI serial). Known device serials:

| Serial ID | Robot |
|---|---|
| `A50285BI` | Panda / FR3 (7 DOF) |
| `FTAO9WPU` | YAM Left (6 DOF) |
| `FTAO9WCV` | YAM Right (6 DOF) |

List connected devices:
```bash
python -m molmo_spaces_maniskill.teleop.calibration.cli --list
```

### SO101 Leader

Connects via serial port (typically `/dev/ttyACM0`). Requires the LeRobot library.

Test device connection:
```bash
python teleop_so101.py --test-only --port /dev/ttyACM0
```

---

## Calibration

### GELLO Calibration

Run once per device before first use:
```bash
# Place GELLO at its zero position, then run:
python -m molmo_spaces_maniskill.teleop.calibration.cli \
    --port /dev/serial/by-id/usb-FTDI_<serial>-if00-port0 \
    --robot yam \
    --start-joints "0,0,0,0,0,0"

# View saved calibrations
python -m molmo_spaces_maniskill.teleop.calibration.cli --show-calibrations
```

Calibrations are saved to `~/.mani_skill/gello_calibrations/` and auto-loaded on reconnect.

### SO101 Calibration

```bash
# Step 1: Motor range calibration (do once)
python teleop_so101.py --test-only --calibrate --port /dev/ttyACM0

# Step 2: Offset calibration (align SO101 pose to simulation home pose)
python teleop_so101.py -e SO100CloseMicrowave-v1 -r so100_wristcam \
    --port /dev/ttyACM0 --calibrate-offsets
```

---

## Running Teleoperation

### GELLO — Single Arm (FR3/Panda tasks)

```bash
python teleop_gello.py \
    -e CubeStack-v1 \
    -r fr3_robotiq_wristcam \
    --auto-discover \
    --target-trajs 50 \
    --record-dir demos
```

### GELLO — Bimanual (YAM tasks)

```bash
python teleop_gello.py \
    -e BimanualYAMPickPlace-v1 \
    -r yam_bimanual \
    --auto-discover \
    --control-freq 50 \
    --record-freq 10 \
    --target-trajs 100 \
    --save-video
```

### SO101 Leader (SO100 tasks)

```bash
python teleop_so101.py \
    -e SO100CloseMicrowave-v1 \
    -r so100_wristcam \
    --port /dev/ttyACM0 \
    --target-trajs 50
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `-e` / `--env-id` | required | Task environment ID |
| `-r` / `--robot-uid` | required | Robot type |
| `--auto-discover` | false | Auto-find GELLO devices via USB |
| `--control-freq` | 50 | Control loop frequency (Hz) |
| `--record-freq` | 15 | Trajectory save frequency (Hz) |
| `--target-trajs` | — | Stop after N successful demos |
| `--max-episode-steps` | 50000 | Max steps before episode timeout |
| `--alignment-threshold` | 0.2 | Joint error threshold (radians) for alignment check |
| `--save-video` | true | Save episode video |
| `--record-dir` | `demos` | Output directory for trajectories |

---

## Episode Workflow

1. **Alignment phase** — A "ghost" robot shows the home pose. Move your device to match it physically. The system detects when joint error drops below the threshold and auto-proceeds (or press Enter).

2. **Teleoperation phase** — Control the robot to complete the task. The simulation checks success conditions each step.

3. **Recording** — Only successful episodes are saved. The wrapper records at `--record-freq` while the control loop runs at `--control-freq`.

4. **Interactive controls** during an episode:
   - `r` — Restart episode (discard current trajectory)
   - `h` — Show help
   - `q` — Quit collection

---

## Output Format

Saved to `demos/{env_id}/gello_teleop_{timestamp}/` (or `so101_teleop_...`):

```
demos/BimanualYAMPickPlace-v1/gello_teleop_20240320_143022/
├── trajectory.h5           # All episodes: observations, actions, rewards, infos
├── trajectory_meta.json    # Metadata: source, timestamps, robot, task
└── videos/                 # Per-episode MP4 files (if --save-video)
```

---

## Quick Reference by Robot

| Robot UID | Hardware | Example Task |
|---|---|---|
| `yam_bimanual` | 2× GELLO (FTAO9WPU + FTAO9WCV) | `BimanualYAMPickPlace-v1` |
| `yam` | 1× GELLO | any single-arm YAM task |
| `fr3_robotiq_wristcam` | 1× GELLO (A50285BI) | `CubeStack-v1` |
| `panda` | 1× GELLO (A50285BI) | any panda task |
| `so100_wristcam` | SO101 Leader | `SO100CloseMicrowave-v1` |

---

## Adding a New Task

1. Create a task file in `molmoact2/tasks/<category>_tasks/your_task.py` subclassing `BaseEnv`
2. Register the environment with `@register_env("YourTask-v1", ...)`
3. Add the robot config in `teleop/configs/robot_configs.py` if using a new robot

See existing tasks in [tasks/yam_tasks/](tasks/yam_tasks/) or [tasks/droid_tasks/](tasks/droid_tasks/) for examples.
