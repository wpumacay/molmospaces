"""
Robot Teleoperation Configurations

This module defines how different ManiSkill robots map to GELLO devices
and their action space configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Callable
import numpy as np


@dataclass
class ArmConfig:
    """Configuration for a single robot arm."""
    name: str  # e.g., "arm", "arm1", "arm2"
    num_joints: int
    gripper_joints: int  # Number of gripper joints in action space
    ee_link_name: str
    gello_serial_id: Optional[str] = None  # Associated GELLO device


@dataclass  
class RobotTeleopConfig:
    """
    Teleoperation configuration for a ManiSkill robot.
    
    Attributes:
        robot_uid: ManiSkill robot UID (e.g., "panda", "fr3_robotiq_wristcam")
        arms: List of arm configurations
        control_mode: Recommended control mode for teleoperation
        action_dim: Total action dimension
        base_joints: Number of base/body joints (if mobile robot)
        gripper_normalized: Whether gripper action is normalized to [-1, 1]
    """
    robot_uid: str
    arms: List[ArmConfig]
    control_mode: str = "pd_joint_pos"
    action_dim: Optional[int] = None  # Auto-computed if None
    base_joints: int = 0
    body_joints: int = 0
    gripper_normalized: bool = True
    
    def __post_init__(self):
        if self.action_dim is None:
            # Auto-compute action dimension
            total = self.base_joints + self.body_joints
            for arm in self.arms:
                total += arm.num_joints + arm.gripper_joints
            self.action_dim = total
    
    @property
    def num_arms(self) -> int:
        return len(self.arms)
    
    @property
    def is_bimanual(self) -> bool:
        return self.num_arms > 1
    
    def get_arm_joint_slice(self, arm_idx: int = 0) -> slice:
        """Get the slice for arm joints in action array."""
        start = self.base_joints + self.body_joints
        for i in range(arm_idx):
            start += self.arms[i].num_joints + self.arms[i].gripper_joints
        arm = self.arms[arm_idx]
        return slice(start, start + arm.num_joints)
    
    def get_gripper_slice(self, arm_idx: int = 0) -> slice:
        """Get the slice for gripper joints in action array."""
        start = self.base_joints + self.body_joints
        for i in range(arm_idx):
            start += self.arms[i].num_joints + self.arms[i].gripper_joints
        arm = self.arms[arm_idx]
        gripper_start = start + arm.num_joints
        return slice(gripper_start, gripper_start + arm.gripper_joints)


# ============================================================================
# Robot Teleoperation Configurations Registry
# ============================================================================

ROBOT_TELEOP_CONFIGS: Dict[str, RobotTeleopConfig] = {}


def register_robot_config(config: RobotTeleopConfig) -> None:
    """Register a robot teleoperation configuration."""
    ROBOT_TELEOP_CONFIGS[config.robot_uid] = config


def get_robot_config(robot_uid: str) -> Optional[RobotTeleopConfig]:
    """Get robot teleoperation configuration by UID."""
    return ROBOT_TELEOP_CONFIGS.get(robot_uid)


# ============================================================================
# Built-in Robot Configurations
# ============================================================================

# Panda (Franka Emika) - 7 DOF arm + 1 gripper action
register_robot_config(RobotTeleopConfig(
    robot_uid="panda",
    arms=[ArmConfig(
        name="arm",
        num_joints=7,
        gripper_joints=1,
        ee_link_name="panda_hand_tcp",
        gello_serial_id="A50285BI",
    )],
    control_mode="pd_joint_pos",
    gripper_normalized=True,
))

# Franka FR3 with Robotiq gripper and wrist camera
register_robot_config(RobotTeleopConfig(
    robot_uid="fr3_robotiq_wristcam",
    arms=[ArmConfig(
        name="arm",
        num_joints=7,
        gripper_joints=1,
        ee_link_name="robotiq_arg2f_base_link",
        gello_serial_id="A50285BI",
    )],
    control_mode="pd_joint_pos",
    gripper_normalized=True,
))

# Panda with wrist camera
register_robot_config(RobotTeleopConfig(
    robot_uid="panda_wristcam",
    arms=[ArmConfig(
        name="arm",
        num_joints=7,
        gripper_joints=1,
        ee_link_name="panda_hand_tcp",
        gello_serial_id="A50285BI",
    )],
    control_mode="pd_joint_pos",
    gripper_normalized=True,
))


# UR with dual arms
register_robot_config(RobotTeleopConfig(
    robot_uid="ur_dual_arm",
    arms=[
        ArmConfig(
            name="left_arm",
            num_joints=6,
            gripper_joints=1,
            ee_link_name="left_ee_link",
            gello_serial_id="FT7WBEIA",
        ),
        ArmConfig(
            name="right_arm",
            num_joints=6,
            gripper_joints=1,
            ee_link_name="right_ee_link",
            gello_serial_id="FT7WBG6A",
        ),
    ],
    control_mode="pd_joint_pos",
    gripper_normalized=True,
))


# YAM - 6 DOF arm with linear gripper (single arm)
register_robot_config(RobotTeleopConfig(
    robot_uid="yam",
    arms=[ArmConfig(
        name="arm",
        num_joints=6,
        gripper_joints=1,
        ee_link_name="link_6",  # MJCF uses link_6 for end-effector
        gello_serial_id="FTAO9WCV",  # YAM RIGHT GELLO
    )],
    control_mode="pd_joint_pos",
    gripper_normalized=False,  # Linear gripper uses position directly
))

# YAM Bimanual - Two 6 DOF arms side by side
register_robot_config(RobotTeleopConfig(
    robot_uid="yam_bimanual",
    arms=[
        ArmConfig(
            name="left_arm",
            num_joints=6,
            gripper_joints=1,
            ee_link_name="left_link_6",
            gello_serial_id="FTAO9WPU",  # YAM LEFT GELLO
        ),
        ArmConfig(
            name="right_arm",
            num_joints=6,
            gripper_joints=1,
            ee_link_name="right_link_6",
            gello_serial_id="FTAO9WCV",  # YAM RIGHT GELLO
        ),
    ],
    control_mode="pd_joint_pos",
    gripper_normalized=False,
))

# SO101 - 5 DOF arm with gripper (LeRobot)
register_robot_config(RobotTeleopConfig(
    robot_uid="so100",
    arms=[ArmConfig(
        name="arm",
        num_joints=5,
        gripper_joints=1,
        ee_link_name="wrist_roll_link",  # End effector link
        gello_serial_id=None,  # Uses SO101 device, not GELLO
    )],
    control_mode="pd_joint_pos",
    gripper_normalized=True,
))

