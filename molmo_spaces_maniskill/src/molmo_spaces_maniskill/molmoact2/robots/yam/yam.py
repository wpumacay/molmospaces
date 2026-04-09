"""
YAM Robot Agent

YAM (Yet Another Manipulator) is a 6-DOF robot arm with a linear gripper.
- 6 revolute joints for the arm (joint1-joint6)
- 2 prismatic joints for the gripper (joint7, joint8)
"""

from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class YAM(BaseAgent):
    """
    YAM Robot - 6-DOF arm with linear gripper.
    
    Joint structure:
        - joint1: base rotation (-2.618 to 3.054 rad)
        - joint2: shoulder (0 to π rad)
        - joint3: elbow (0 to π rad)
        - joint4: wrist1 (-π/2 to π/2 rad)
        - joint5: wrist2 (-π/2 to π/2 rad)
        - joint6: wrist3 (-2.094 to 2.094 rad)
        - joint7: gripper left (-0.0475 to 0 m)
        - joint8: gripper right (-0.0475 to 0 m)
    """
    
    uid = "yam"
    # Use MJCF instead of URDF (URDF has physics simulation issues with joint6)
    urdf_path = None
    urdf_config = dict()  # Empty config for MJCF

    mjcf_path = "/home/shuo/research/molmospaces/molmo_spaces_maniskill/src/molmo_spaces_maniskill/molmoact2/assets/yam/yam_mujoco/yam_linear.xml"

    # Rest pose - arm in a neutral position
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0,           # joint1: base rotation
                np.pi / 4,     # joint2: shoulder
                np.pi / 2,     # joint3: elbow
                0.0,           # joint4: wrist1
                0.0,           # joint5: wrist2
                0.0,           # joint6: wrist3
                -0.02,         # joint7: gripper left (half open)
                -0.02,         # joint8: gripper right (half open)
            ]),
            pose=sapien.Pose(),
        ),
        home=Keyframe(
            qpos=np.array([
                0.0,           # joint1
                0.0,     # joint2
                0.0, # joint3
                0.0,           # joint4
                0.0,           # joint5
                0.0,           # joint6
                0.0,           # joint7: gripper closed
                0.0,           # joint8: gripper closed
            ]),
            pose=sapien.Pose(),
        ),
    )

    # Arm joints (6 DOF)
    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    
    # Gripper joints (2 prismatic joints)
    # Note: MJCF uses "left_finger"/"right_finger", URDF uses "joint7"/"joint8"
    gripper_joint_names = [
        "left_finger",
        "right_finger",
    ]
    
    # End-effector link
    ee_link_name = "link_6"  # MJCF uses "link_6" instead of "gripper"

    # Controller parameters
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 50

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm Controllers
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD EE position control
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,
            self.arm_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Gripper Controllers
        # -------------------------------------------------------------------------- #
        # Linear gripper: both fingers move together (mimic)
        # Range: -0.0475 (open) to 0 (closed)
        # Note: MJCF uses "left_finger"/"right_finger", URDF uses "joint7"/"joint8"
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.0475,
            upper=0.0,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"right_finger": {"joint": "left_finger"}},
        )

        # -------------------------------------------------------------------------- #
        # Combined Controller Configs
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, 
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos, 
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos, 
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, 
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, 
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, 
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, 
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel, 
                gripper=gripper_pd_joint_pos
            ),
        )

        return deepcopy_dict(controller_configs)

    def _after_init(self):
        """Initialize links after robot is loaded."""
        # MJCF uses "link_left_finger"/"link_right_finger", URDF uses "tip_left"/"tip_right"
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link_left_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link_right_finger"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """
        Check if the robot is grasping an object.

        Args:
            object: The object to check if the robot is grasping
            min_force: Minimum force before the robot is considered to be grasping (N)
            max_angle: Maximum angle of contact to consider grasping (degrees)
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # Direction to open the gripper (along the prismatic axis)
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 2]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 2]
        
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        """Check if the robot is static (not moving)."""
        qvel = self.robot.get_qvel()[..., :-2]  # Exclude gripper joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        """Get TCP (tool center point) position."""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """Get TCP (tool center point) pose."""
        return self.tcp.pose

