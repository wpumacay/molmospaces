"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose, Actor
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.agents.controllers.pd_ee_pose import PDEEPosController, PDEEPoseController

from molmo_spaces_maniskill.molmoact2.robots.fr3_wristcam import FR3WristCam as FR3RobotiqWristCam
from molmo_spaces_maniskill.core.env import MolmoSpacesEnv
from molmo_spaces_maniskill import MOLMOSPACES_MJCF_SCENES_DIR


def build_solid_color_target(
    scene,
    radius: float,
    thickness: float,
    name: str,
    color: np.ndarray,
    body_type: str = "dynamic",
    add_collision: bool = False,
    initial_pose: Optional[sapien.Pose] = None,
):
    """
    build a solid color target
    
    Args:
        scene: ManiSkill scene object
        radius: target half size
        thickness: target thickness
        name: actor name
        color: RGBA color array, range [0, 1]
        body_type: "dynamic", "kinematic", or "static"
        add_collision: whether to add collision
        initial_pose: initial pose
    """
    builder = scene.create_actor_builder()
    
    # only create a solid color block
    builder.add_box_visual(
        half_size=[radius, radius, thickness / 2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    if add_collision:
        builder.add_box_collision(half_size=[radius, radius, thickness / 2])
    
    # create actor based on body type
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    else:
        raise ValueError(f"Unknown body_type: {body_type}")
    
    if initial_pose is not None:
        actor.set_pose(initial_pose)
    
    return actor


# predefined color constants (for convenience)
COLORS = {
    'red': np.array([255, 0, 0, 255]) / 255,
    'green': np.array([0, 255, 0, 255]) / 255,
    'blue': np.array([0, 100, 255, 255]) / 255,
    'yellow': np.array([255, 255, 0, 255]) / 255,
    'purple': np.array([128, 0, 255, 255]) / 255,
    'orange': np.array([255, 165, 0, 255]) / 255,
    'cyan': np.array([0, 255, 255, 255]) / 255,
    'pink': np.array([255, 192, 203, 255]) / 255,
    'lime': np.array([50, 200, 50, 255]) / 255,
    'white': np.array([255, 255, 255, 255]) / 255,
    'black': np.array([0, 0, 0, 255]) / 255,
    'gray': np.array([128, 128, 128, 255]) / 255,
}

@register_env("CubeStack-v1")
class CubeStackEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to pick up a cube and place it on a goal region. there are two cubes on the table, the robot needs to pick up one of them and place it on the goal region.

    **Randomizations:**
    - the two cubes' xy position are randomized on top of a table in the region [-0.1, 0.0] x [-0.05, 0.05] and [0.0, 0.1] x [-0.05, 0.05]. It is placed flat on the table
    - the two cubes' z-axis rotation are randomized in [-pi/4, pi/4]

    **Success Conditions:**
    - the cube is placed on one of the goal regions
    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]

    # Specify some supported robot types
    
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    # set some commonly used values
    goal_radius = 0.05
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="fr3_robotiq_wristcam", robot_init_qpos_noise=0.01, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )


    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        
        # 定义相机在机器人基座坐标系中的局部位姿
        droid_exo_left_local = sapien.Pose(
            p=[-0.12, 0.32, 0.6], 
            q=[0.9622501868990583, 0.022557566113149834, 0.08418598282936919, -0.25783416049629954]
        )
        droid_exo_right_local = sapien.Pose(
            p=[-0.12, -0.32, 0.6], 
            q=[0.9622501868990583, -0.022557566113149838, 0.0841859828293692, 0.25783416049629954]
        )

        pose_base = sapien_utils.look_at([0.0, 0.0, 0.45], [-0.3, 0.0, 0.08])
        pose_steer_left = sapien_utils.look_at([-0.15, -0.35, 0.4], [-0.4, 0.0, -0.1])
        pose_steer_right = sapien_utils.look_at([-0.15, 0.35, 0.4], [-0.4, 0.0, -0.1])
        
        return [
            # CameraConfig(
            #     "base_camera",
            #     pose=droid_exo_left_local,
            #     width=256,
            #     height=256,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            #     mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            # ),
            CameraConfig(
                uid="left_camera",
                pose=droid_exo_left_local,  # 直接使用局部位姿
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            ),
            CameraConfig(
                uid="right_camera",
                pose=droid_exo_right_local,  # 添加右相机
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            ),
            # CameraConfig(
            #     uid= "steer_left_camera",
            #     pose=pose_steer_left,
            #     width=512,
            #     height=512,
            #     fov=np.pi / 2,
            #     near=0.1,
            #     far=100,
            # ),
            # CameraConfig(
            #     uid= "steer_right_camera",
            #     pose=pose_steer_right,
            #     width=512,
            #     height=512,
            #     fov=np.pi / 2,
            #     near=0.1,
            #     far=100,
            # )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.25, 0.0, 0.25], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.2, near=0.01, far=100
        )

        # droid_exo_left_local = sapien.Pose(
        #     p=[-0.12, 0.32, 0.6], 
        #     q=[0.9622501868990583, 0.022557566113149834, 0.08418598282936919, -0.25783416049629954]
        # )

        # return CameraConfig(
        #     "render_camera", pose=droid_exo_left_local, width=512, height=512, fov=np.pi/2, near=0.01, far=100, mount=self.agent.robot.links_map["base"]
        # )

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[0.0, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        # finally we specify an initial pose for the cube so that it doesn't collide with other objects initially
        self.cube1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=COLORS['red'],
            name="cube1",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.cube2 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=COLORS['green'],
            name="cube2",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.cube3 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=COLORS['blue'],
            name="cube3",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.cube4 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=COLORS['yellow'],
            name="cube4",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.cube5 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=COLORS['purple'],
            name="cube5",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        # Add a button (cylinder shape)
        builder = self.scene.create_actor_builder()
        # Create cylinder visual (button shape)
        builder.add_cylinder_visual(
            radius=0.03,
            half_length=0.01,
            material=sapien.render.RenderMaterial(
                base_color=COLORS['red']
            )
        )
        # Add collision
        builder.add_cylinder_collision(
            radius=0.03,
            half_length=0.01
        )
        # Build as static (won't move when pressed)
        self.button = builder.build_static(name="button")



        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place）


        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            
            # Override robot position after table_scene.initialize() sets it

            init_qpos = torch.tensor([[                   
                0.0,
                -1 /5 * np.pi,
                0,
                -np.pi * 4 / 5,
                0,
                np.pi * 3 / 5,
                0,
                0,
                0,
                0,
                0,
                0.04,
                0.04
            ]])
            self.agent.robot.set_qpos(init_qpos)
            self.agent.robot.set_pose(sapien.Pose(
                p = [-0.85, 0, -0.2],
                q = euler2quat(0, 0, np.pi/2)))

            # Arrange cubes: 1 in center, 4 around it in 4 directions, button on left
            # Robot is at (-0.85, 0, -0.2), looking towards positive X
            
            # Set Z height for all cubes
            z_height = self.cube_half_size
            
            # Quaternion for no rotation (wxyz format)
            q = [1, 0, 0, 0]
            
            # Use current positions as reference
            center_x = -0.4  # Current cube1 x position
            center_y = 0.0
            cube_spacing = 0.15  # Current spacing between cubes
            
            # Cube 1 (center)
            xyz1 = torch.zeros((b, 3))
            xyz1[:, 0] = center_x + torch.rand(b) * 0.02 - 0.01
            xyz1[:, 1] = center_y + torch.rand(b) * 0.02 - 0.01
            xyz1[:, 2] = z_height
            self.cube1.set_pose(Pose.create_from_pq(p=xyz1, q=q))
            
            # Cube 2 (left of center)
            xyz2 = torch.zeros((b, 3))
            xyz2[:, 0] = center_x + torch.rand(b) * 0.02 - 0.01
            xyz2[:, 1] = center_y - cube_spacing + torch.rand(b) * 0.02 - 0.01
            xyz2[:, 2] = z_height
            self.cube2.set_pose(Pose.create_from_pq(p=xyz2, q=q))
            
            # Cube 3 (right of center)
            xyz3 = torch.zeros((b, 3))
            xyz3[:, 0] = center_x + torch.rand(b) * 0.02 - 0.01
            xyz3[:, 1] = center_y + cube_spacing + torch.rand(b) * 0.02 - 0.01
            xyz3[:, 2] = z_height
            self.cube3.set_pose(Pose.create_from_pq(p=xyz3, q=q))
            
            # Cube 4 (front of center, closer to robot)
            xyz4 = torch.zeros((b, 3))
            xyz4[:, 0] = center_x - cube_spacing + torch.rand(b) * 0.02 - 0.01
            xyz4[:, 1] = center_y + torch.rand(b) * 0.02 - 0.01
            xyz4[:, 2] = z_height
            self.cube4.set_pose(Pose.create_from_pq(p=xyz4, q=q))
            
            # Cube 5 (back of center, away from robot)
            xyz5 = torch.zeros((b, 3))
            xyz5[:, 0] = center_x + cube_spacing + torch.rand(b) * 0.02 - 0.01
            xyz5[:, 1] = center_y + torch.rand(b) * 0.02 - 0.01
            xyz5[:, 2] = z_height
            self.cube5.set_pose(Pose.create_from_pq(p=xyz5, q=q))
            
            # Set button position (left of cubes)
            button_xyz = torch.zeros((b, 3))
            button_xyz[:, 0] = center_x  # Same x as center cube
            button_xyz[:, 1] = center_y - cube_spacing * 2  # Left of the left cube
            button_xyz[:, 2] = 0.015  # Slightly above table (button height/2)
            # Rotate 90 degrees around Y axis so button faces right (towards cubes)
            button_q = euler2quat(0, np.pi/2, 0)  # (w,x,y,z) format
            self.button.set_pose(Pose.create_from_pq(p=button_xyz, q=button_q))



    def _get_steered_controller(self):
        controller = self.agent.controller
        if hasattr(controller, 'controllers') and isinstance(controller.controllers, dict):
            for ctrl in controller.controllers.values():
                if isinstance(ctrl, (PDEEPosController, PDEEPoseController)):
                    return ctrl
        return None
    
    def _is_stacked(self, obj1: Actor, obj2: Actor) -> torch.Tensor:
        return (
            torch.linalg.norm(
                obj1.pose.p[..., :2] - obj2.pose.p[..., :2], dim=-1
            )
            < self.cube_half_size 
        ) & (obj1.pose.p[..., 2] < obj2.pose.p[..., 2])

    def _is_touched(self, obj: Actor) -> torch.Tensor:
        """Check if an object is touched by the end effector (gripper fingers)
        
        Args:
            obj: The object to check
            
        Returns:
            torch.Tensor: Boolean tensor of shape (num_envs,) indicating if touched
        """
        # Get contact forces between object and gripper fingers
        # For Robotiq gripper, check both finger pads
        left_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.robot.links_map["left_inner_finger_pad"], obj
        )
        right_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.robot.links_map["right_inner_finger_pad"], obj
        )
        
        # Calculate force magnitudes
        left_force = torch.linalg.norm(left_contact_forces, axis=1)
        right_force = torch.linalg.norm(right_contact_forces, axis=1)
        
        # Object is touched if either finger has contact force > threshold
        min_force = 0.1  # Newtons - lower threshold for just touching
        is_touched = (left_force > min_force) | (right_force > min_force)
        
        return is_touched

    def _is_button_pressed(self) -> torch.Tensor:
        """Check if button is pressed by the end effector
        
        Returns:
            torch.Tensor: Boolean tensor indicating if button is pressed
        """
        # Check contact forces with any part of the robot
        contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.robot.links_map["left_inner_finger_pad"], 
            self.button
        )
        contact_forces_right = self.scene.get_pairwise_contact_forces(
            self.agent.robot.links_map["right_inner_finger_pad"], 
            self.button
        )
        
        total_force = torch.linalg.norm(contact_forces, axis=1) + \
                      torch.linalg.norm(contact_forces_right, axis=1)
        
        # Button is pressed if force exceeds threshold
        return total_force > 0.5  # Higher threshold for pressing

    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position) and
        # the cube is still on the surface
        # success = self._is_obj_placed(self.cube, self.goal_region2) 
                # | self._is_obj_placed(self.cube, self.goal_region2) \
        
        button_pressed = self._is_button_pressed()
        
        success = self._is_stacked(self.cube1, self.cube3) \
            & button_pressed  # Add button press as success condition
            # & self._is_stacked(self.cube2, self.cube3) \
            # & self._is_stacked(self.cube3, self.cube4) \
 
        
        return {
            "success": success,
            "button_pressed": button_pressed,  # Add to info for debugging
        }

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
