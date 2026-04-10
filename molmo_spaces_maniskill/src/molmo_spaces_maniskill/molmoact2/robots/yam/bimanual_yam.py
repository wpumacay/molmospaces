"""
Bimanual YAM Robot Agent

Two YAM arms side by side, loaded from a single bimanual MJCF file.
"""

from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class BimanualYAM(BaseAgent):
    """
    Bimanual YAM Robot - Two 6-DOF arms with linear grippers side by side.
    
    Loads from a single bimanual MJCF file with both arms already defined:
    - Left arm at y = +0.22
    - Right arm at y = -0.22
    
    Joint structure (16 total):
        Left arm (8 joints):
        - left_joint1 to left_joint6: arm joints
        - left_left_finger, left_right_finger: gripper joints
        
        Right arm (8 joints):
        - right_joint1 to right_joint6: arm joints
        - right_left_finger, right_right_finger: gripper joints
    """
    
    uid = "yam_bimanual"
    
    # Use the bimanual MJCF file directly
    mjcf_path = "/home/shuo/research/molmospaces/molmo_spaces_maniskill/src/molmo_spaces_maniskill/molmoact2/assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml"
    urdf_path = None
    urdf_config = dict()
    
    # Rest pose for both arms
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                # Left arm
                0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0, -0.02, -0.02,
                # Right arm
                0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0, -0.02, -0.02,
            ]),
            pose=sapien.Pose(),
        ),
        home=Keyframe(
            qpos=np.array([
                # Left arm
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # Right arm
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            pose=sapien.Pose(),
        ),
    )

    # Joint names for both arms
    left_arm_joint_names = [
        "left_joint1", "left_joint2", "left_joint3",
        "left_joint4", "left_joint5", "left_joint6",
    ]
    right_arm_joint_names = [
        "right_joint1", "right_joint2", "right_joint3",
        "right_joint4", "right_joint5", "right_joint6",
    ]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names
    
    left_gripper_joint_names = ["left_left_finger", "left_right_finger"]
    right_gripper_joint_names = ["right_left_finger", "right_right_finger"]
    gripper_joint_names = left_gripper_joint_names + right_gripper_joint_names
    
    # End-effector links
    left_ee_link_name = "left_link_6"
    right_ee_link_name = "right_link_6"

    # Controller parameters (same as single YAM)
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 50

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="left_hand_camera",
                pose=sapien.Pose(p=[0, 0.09, 0.06], q=[0.57922797, -0.40557979, -0.40557979, -0.57922797]),
                height=480,
                width=640,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["left_link_6"],
            ),
            CameraConfig(
                uid="right_hand_camera",
                pose=sapien.Pose(p=[0, 0.09, 0.06], q=[0.57922797, -0.40557979, -0.40557979, -0.57922797]),
                height=480,
                width=640,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["right_link_6"],
            ),
    ]

    @property
    def _controller_configs(self):
        """Controller configs for both arms."""
        # Left arm controllers
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        
        left_gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.left_gripper_joint_names,
            lower=-0.0475,
            upper=0.0,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"left_right_finger": {"joint": "left_left_finger"}},
        )
        
        # Right arm controllers
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        
        right_gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.right_gripper_joint_names,
            lower=-0.0475,
            upper=0.0,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"right_right_finger": {"joint": "right_left_finger"}},
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                left_arm=left_arm_pd_joint_pos,
                left_gripper=left_gripper_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
                right_gripper=right_gripper_pd_joint_pos,
            ),
        )

        return deepcopy_dict(controller_configs)

    def _after_init(self):
        """Initialize links after robot is loaded."""
        # Left arm links
        self.left_finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_link_left_finger"
        )
        self.left_finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_link_right_finger"
        )
        self.left_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.left_ee_link_name
        )
        
        # Right arm links
        self.right_finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_link_left_finger"
        )
        self.right_finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_link_right_finger"
        )
        self.right_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.right_ee_link_name
        )

    def is_grasping(self, object: Actor, arm: str = "left", min_force=0.5, max_angle=85):
        """
        Check if the specified arm is grasping an object.

        Args:
            object: The object to check
            arm: "left" or "right"
            min_force: Minimum force (N)
            max_angle: Maximum angle (degrees)
        """
        if arm == "left":
            finger1_link = self.left_finger1_link
            finger2_link = self.left_finger2_link
        else:
            finger1_link = self.right_finger1_link
            finger2_link = self.right_finger2_link
            
        l_contact_forces = self.scene.get_pairwise_contact_forces(finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(finger2_link, object)
        
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = finger1_link.pose.to_transformation_matrix()[..., :3, 2]
        rdirection = -finger2_link.pose.to_transformation_matrix()[..., :3, 2]
        
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        """Check if both arms are static."""
        qvel = self.robot.get_qvel()[..., :-4]  # Exclude gripper joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def left_tcp_pos(self):
        """Get left TCP position."""
        return self.left_tcp.pose.p

    @property
    def right_tcp_pos(self):
        """Get right TCP position."""
        return self.right_tcp.pose.p

    @property
    def left_tcp_pose(self):
        """Get left TCP pose."""
        return self.left_tcp.pose

    @property
    def right_tcp_pose(self):
        """Get right TCP pose."""
        return self.right_tcp.pose

