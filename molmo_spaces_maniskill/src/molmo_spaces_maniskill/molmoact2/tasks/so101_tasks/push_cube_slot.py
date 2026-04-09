"""
Pick and Place Task for SO100 Robot

A simple pick-and-place task where the SO100 robot needs to grasp a cube
and move it to a target location.
"""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from molmo_spaces_maniskill.molmoact2.robots.so100 import SO100WristCam


@register_env("SO100PushCubeSlot-v1", max_episode_steps=100)
class SO100PushCubeSlotEnv(BaseEnv):
    """
    Push a cube into a slot for SO100 Robot.
    
    **Task Description:**
    The SO100 robot needs to pick up a cube from the table and place it at
    a target location marked by a green sphere.
    
    **Randomizations:**
    - Cube position is randomized on the table surface
    - Cube z-rotation is randomized
    - Goal position is randomized in 3D space above the table
    
    **Success Conditions:**
    - Cube is within goal_thresh distance of the goal
    - Robot is static (velocity < threshold)
    """
    
    SUPPORTED_ROBOTS = ["so100_wristcam"]
    agent: SO100WristCam
    
    # Task parameters
    cube_half_size = 0.015  # 3cm cube (smaller for SO100)
    goal_thresh = 0.025  # 2.5cm threshold
    cube_spawn_half_size = 0.04  # Smaller spawn area (closer to robot)
    max_goal_height = 0.08  # Lower max height for goal
    slot_wall_thickness = 0.01
    slot_wall_length = 0.08
    slot_inner_width = 0.05
    slot_height = 0.02
    slot_center = np.array([-0.36, 0.0, 0.0], dtype=np.float32)
    
    def __init__(
        self, 
        *args, 
        robot_uids="so100_wristcam",
        robot_init_qpos_noise=0.02,
        reset_robot_qpos=True,  # Whether to reset robot qpos to home on episode init
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reset_robot_qpos = reset_robot_qpos
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        """Configure sensor cameras."""
        pose = sapien_utils.look_at(
            eye = [-0.3, 0.0, 0.35],
            target = [-0.6, 0.0, -0.1]
        )
        return [CameraConfig("base_camera", pose, width=640, height=480, fov=np.pi / 2, near=0.01, far=100)]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure human render camera."""
        pose = sapien_utils.look_at(
            eye = [-0.3, 0.0, 0.35],
            target = [-0.6, 0.0, -0.1]
        )
        return CameraConfig("render_camera", pose, width=640, height=480, fov=np.pi / 2, near=0.01, far=100)
    
    def _load_agent(self, options: dict):
        """Load the robot agent."""
        # Position robot closer to workspace
        super()._load_agent(options, sapien.Pose(p=[-0.15, 0, 0]))
    
    def _load_scene(self, options: dict):
        """Load the scene with table and objects."""
        # Build table scene
        self.table_scene = TableSceneBuilder(
            self, 
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create walls around the workspace to avoid black background
        self._build_walls()
        
        # Create the cube to pick
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.1, 0.45, 1.0, 1.0],  # Blue cube
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self._build_slot()
        
        # Create goal indicator (green sphere)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.5],  # Semi-transparent green
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _build_slot(self):
        """Build a static U-shaped slot like the reference image."""
        slot_z = self.cube_half_size
        wall_color = [0.08, 0.08, 0.08, 1.0]

        left_wall_y = self.slot_center[1] - (self.slot_inner_width / 2 + self.slot_wall_thickness / 2)
        right_wall_y = self.slot_center[1] + (self.slot_inner_width / 2 + self.slot_wall_thickness / 2)
        bottom_wall_x = self.slot_center[0] + self.slot_wall_length / 2 - self.slot_wall_thickness / 2

        actors.build_box(
            self.scene,
            half_sizes=[self.slot_wall_length / 2, self.slot_wall_thickness / 2, self.slot_height / 2],
            color=wall_color,
            name="slot_left_wall",
            body_type="static",
            add_collision=True,
            initial_pose=sapien.Pose(p=[self.slot_center[0], left_wall_y, slot_z]),
        )
        actors.build_box(
            self.scene,
            half_sizes=[self.slot_wall_length / 2, self.slot_wall_thickness / 2, self.slot_height / 2],
            color=wall_color,
            name="slot_right_wall",
            body_type="static",
            add_collision=True,
            initial_pose=sapien.Pose(p=[self.slot_center[0], right_wall_y, slot_z]),
        )
        actors.build_box(
            self.scene,
            half_sizes=[self.slot_wall_thickness / 2, (self.slot_inner_width + 2 * self.slot_wall_thickness) / 2, self.slot_height / 2],
            color=wall_color,
            name="slot_bottom_wall",
            body_type="static",
            add_collision=True,
            initial_pose=sapien.Pose(p=[bottom_wall_x, self.slot_center[1], slot_z]),
        )
    
    def _build_walls(self):
        """Build walls around the workspace to provide background for cameras."""
        # Wall parameters
        wall_thickness = 0.01
        wall_height = 2.5
        wall_distance = 1.6  # Distance from center
        wall_color = [0.9, 0.9, 0.9, 1.0]  # Light gray
        
        # Back wall (behind robot, negative X)
        actors.build_box(
            self.scene,
            half_sizes=[wall_thickness, wall_distance, wall_height / 2],
            color=wall_color,
            name="wall_back",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-wall_distance, 0, 0 ]),
        )
        
        # Front wall (in front of robot, positive X)
        actors.build_box(
            self.scene,
            half_sizes=[wall_thickness, wall_distance, wall_height / 2],
            color=wall_color,
            name="wall_front",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[wall_distance, 0, 0]),
        )
        
        # Left wall (negative Y)
        actors.build_box(
            self.scene,
            half_sizes=[wall_distance, wall_thickness, wall_height / 2],
            color=wall_color,
            name="wall_left",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, -wall_distance, 0]),
        )
        
        # Right wall (positive Y)
        actors.build_box(
            self.scene,
            half_sizes=[wall_distance, wall_thickness, wall_height / 2],
            color=wall_color,
            name="wall_right",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, wall_distance, 0]),
        )
        

    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized positions."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Set robot position using "home" keyframe
            self.agent.robot.set_pose(self.agent.keyframes["home"].pose)
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)
            
            # Randomize cube position closer to the robot, still in front of the gripper.
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = torch.rand((b,)) * 0.04 - 0.5
            cube_xyz[:, 1] = (torch.rand((b,)) - 0.5) * 0.2
            cube_xyz[:, 2] = self.cube_half_size
            
            # Randomize cube rotation (z-axis only)
            cube_qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True
            )
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, cube_qs))
            
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = float(self.slot_center[0])
            goal_xyz[:, 1] = float(self.slot_center[1])
            goal_xyz[:, 2] = self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
    
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """Get extra observations."""
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        
        return obs
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate task success."""
        cube_to_slot = self.goal_site.pose.p - self.cube.pose.p
        obj_to_goal_dist = torch.linalg.norm(cube_to_slot, axis=1)
        is_obj_placed = obj_to_goal_dist <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        
        return {
            "success": is_obj_placed,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
    
    def compute_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """Compute dense reward for training."""
        # Reaching reward: encourage moving TCP towards cube
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward
        
        # Grasping reward
        is_grasped = info["is_grasped"]
        reward += is_grasped.float()
        
        # Placing reward: encourage moving cube towards goal
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped.float()
        
        # Static reward when placed
        qvel = self.agent.robot.get_qvel()[..., :-2]  # Exclude gripper
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"].float()
        
        # Bonus for success
        reward[info["success"]] = 5
        
        return reward
    
    def compute_normalized_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

