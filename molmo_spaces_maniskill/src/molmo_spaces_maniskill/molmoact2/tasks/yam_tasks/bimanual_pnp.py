"""
Bimanual Handover Task for YAM Robot

A handover task using two YAM arms working together.
The cube starts on the left side (reachable by left arm), 
and the goal is on the right side (reachable by right arm).
This requires cross-hand collaboration to complete the task.
"""

from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors

# Import BimanualYAM to ensure it's registered
from molmo_spaces_maniskill.molmoact2.robots.yam import BimanualYAM


@register_env("BimanualYAMPickPlace-v1", max_episode_steps=200)
class BimanualYAMPickPlaceEnv(BaseEnv):
    """
    Bimanual Handover Task for two YAM arms.
    
    **Task Description:**
    A single red cube starts on the LEFT side of the table (reachable by left arm).
    The goal position is on the RIGHT side (reachable by right arm).
    To complete the task, the arms must collaborate:
    1. Left arm picks up the cube
    2. Left arm hands over to right arm (or places in middle)
    3. Right arm places the cube at the goal
    
    **Robot Configuration:**
    Uses BimanualYAM agent which loads two YAM arms:
    - Left arm: positioned at y = +0.22
    - Right arm: positioned at y = -0.22
    - Both arms face the table (+X direction)
    
    **Randomizations:**
    - Cube initial position randomized on left side
    - Goal position randomized on right side
    
    **Success Conditions:**
    - Cube is placed at the goal position on the right side
    """
    
    # Use single bimanual YAM robot (which internally has two arms)
    SUPPORTED_ROBOTS = ["yam_bimanual"]
    agent: BimanualYAM
    
    # Task parameters
    cube_half_size = 0.02  # 4cm cube
    goal_thresh = 0.03  # 3cm threshold
    arm_separation = 0.44  # Distance between arms (y-axis), 22cm each side
    
    def __init__(
        self, 
        *args, 
        robot_uids="yam_bimanual",  # Single bimanual robot
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
        # Front view camera
        pose = sapien_utils.look_at(
            eye=[-0.65, 0.0, 0.5],
            target=[-0.4, 0.0, 0.1]
        )
        return [
            CameraConfig(
                "base_camera", 
                pose, 
                640, 
                480, 
                np.pi / 2, 
                0.01, 
                100),
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure human render camera - top-down angled view."""
        pose = sapien_utils.look_at(
            eye=[-0.65, 0.0, 0.5],
            target=[-0.4, 0.0, 0.1]
        )
        return CameraConfig("render_camera", pose, 640, 480, np.pi / 2, 0.01, 100)

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

    
    def _load_scene(self, options: dict):
        """Load the scene with table and objects."""
        # Build table scene
        self.table_scene = TableSceneBuilder(
            self, 
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create single red cube (starts on left side, goal on right side)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],  # Red
            name="cube",
            initial_pose=sapien.Pose(p=[0.1, 0.15, self.cube_half_size]),  # Left side (positive y)
        )
        
        # Create goal indicator (on right side)
        self.goal = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1, 0.5, 0.5, 0.5],  # Light red, semi-transparent
            name="goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.1, -0.15, self.cube_half_size]),  # Right side (negative y)
        )
        
        self._hidden_objects.append(self.goal)

        self._build_walls()
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized positions."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0, 0.01]))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)
            
            # Randomize cube position on LEFT side (reachable by left arm, positive y)
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.35  # x: 0.05 to 0.15
            cube_xyz[:, 1] = 0.4 + torch.rand((b,)) * 0.08  # y: 0.12 to 0.20 (left side)
            cube_xyz[:, 2] = self.cube_half_size
            cube_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, cube_qs))
            
            # Randomize goal position on RIGHT side (reachable by right arm, negative y)
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.35  # x: 0.05 to 0.15
            goal_xyz[:, 1] = -0.4 - torch.rand((b,)) * 0.08  # y: -0.12 to -0.20 (right side)
            goal_xyz[:, 2] = self.cube_half_size + torch.rand((b,)) * 0.2
            self.goal.set_pose(Pose.create_from_pq(goal_xyz))
    
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """Get extra observations."""
        obs = dict(
            cube_pose=self.cube.pose.raw_pose,
            goal_pos=self.goal.pose.p,
        )
        return obs
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate task success."""
        # Check if cube is at goal
        dist = torch.linalg.norm(
            self.goal.pose.p - self.cube.pose.p, axis=1
        )
        
        success = dist <= self.goal_thresh
        
        return {
            "success": success,
            "cube_to_goal_dist": dist,
        }
    
    def compute_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """Compute dense reward for training."""
        # Distance-based reward
        dist = torch.linalg.norm(
            self.goal.pose.p - self.cube.pose.p, axis=1
        )
        
        # Reward for moving cube closer to goal
        reward = 1 - torch.tanh(5 * dist)
        
        # Bonus for success
        reward[info["success"]] = 3
        
        return reward
    
    def compute_normalized_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3

