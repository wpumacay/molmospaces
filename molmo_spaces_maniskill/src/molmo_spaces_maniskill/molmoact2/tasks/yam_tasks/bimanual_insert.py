"""
Bimanual Peg Insertion Task for YAM Robot

A collaborative insertion task using two YAM arms working together.
Left arm holds the blue slot (base with hole), right arm holds the red peg (cuboid stick).
Both objects start in front of the robot, aligned horizontally along Y axis.
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


def build_slot_with_hole(scene, name: str, outer_half_size, hole_half_size, color=None, initial_pose=None):
    """
    Build a slot (box with a rectangular hole through it).
    
    The slot is a hollow box - a rectangular frame that a peg can pass through.
    The hole goes through along the Y axis (horizontal when placed on table).
    Each wall has a different color for easy identification.
    
    Args:
        scene: ManiSkill scene
        name: Actor name
        outer_half_size: [half_x, half_y, half_z] outer dimensions
        hole_half_size: [half_x, half_z] hole dimensions (hole extends along Y)
        color: RGBA color (ignored, using 4 different colors)
        initial_pose: Initial pose
    
    Structure (cross-section view looking along Y axis):
    ```
        +-----------+
        |  +-----+  |  <- top wall (GREEN)
        |  |     |  |
        |  | hole|  |  <- hole (empty space)
        |  |     |  |
        |  +-----+  |  <- bottom wall (YELLOW)
        +-----------+
        ^           ^
      left wall   right wall
      (BLUE)      (RED)
    ```
    """
    builder = scene.create_actor_builder()
    
    ox, oy, oz = outer_half_size  # outer half sizes
    hx, hz = hole_half_size  # hole half sizes (hole extends along Y)
    
    # Wall thickness
    wall_x = (ox - hx) / 2  # thickness of left/right walls
    wall_z = (oz - hz) / 2  # thickness of top/bottom walls
    
    # 4 different colors for 4 walls (no red, since peg is red)
    mat_left = sapien.render.RenderMaterial(base_color=[0.2, 0.4, 0.8, 1.0])    # Blue - left
    mat_right = sapien.render.RenderMaterial(base_color=[0.3, 0.7, 0.3, 1.0])   # Orange - right
    mat_top = sapien.render.RenderMaterial(base_color=[0.6, 0.5, 0.0, 1.0])
    mat_bottom =  sapien.render.RenderMaterial(base_color=[1.0, 0.5, 0.0, 1.0])    # Soft Green - top  # Dark Yellow - bottom
    
    # Left wall (negative x side) - BLUE
    left_pos = sapien.Pose(p=[-(ox - wall_x), 0, 0])
    builder.add_box_visual(pose=left_pos, half_size=[wall_x, oy, oz], material=mat_left)
    builder.add_box_collision(pose=left_pos, half_size=[wall_x, oy, oz])
    
    # Right wall (positive x side) - RED
    right_pos = sapien.Pose(p=[(ox - wall_x), 0, 0])
    builder.add_box_visual(pose=right_pos, half_size=[wall_x, oy, oz], material=mat_right)
    builder.add_box_collision(pose=right_pos, half_size=[wall_x, oy, oz])
    
    # Top wall (positive z side, between left and right walls) - GREEN
    top_pos = sapien.Pose(p=[0, 0, (oz - wall_z)])
    builder.add_box_visual(pose=top_pos, half_size=[hx, oy, wall_z], material=mat_top)
    builder.add_box_collision(pose=top_pos, half_size=[hx, oy, wall_z])
    
    # Bottom wall (negative z side, between left and right walls) - YELLOW
    bottom_pos = sapien.Pose(p=[0, 0, -(oz - wall_z)])
    builder.add_box_visual(pose=bottom_pos, half_size=[hx, oy, wall_z], material=mat_bottom)
    builder.add_box_collision(pose=bottom_pos, half_size=[hx, oy, wall_z])
    
    actor = builder.build(name=name)
    
    if initial_pose is not None:
        actor.set_pose(initial_pose)
    
    return actor


@register_env("BimanualYAMInsert-v1", max_episode_steps=200)
class BimanualYAMInsertEnv(BaseEnv):
    """
    Bimanual Peg Insertion Task for two YAM arms.
    
    **Task Description:**
    - Left arm picks up the BLUE SLOT (a rectangular frame with a hole)
    - Right arm picks up the RED PEG (a cuboid stick)
    - Both objects start in front of robot center, aligned horizontally along Y axis
    - Arms must coordinate to insert the peg through the slot's hole
    
    **Robot Configuration:**
    Uses BimanualYAM agent which loads two YAM arms:
    - Left arm: positioned at y = +0.22
    - Right arm: positioned at y = -0.22
    - Both arms face the table (+X direction)
    
    **Object Layout (top view):**
    ```
                    +X (forward)
                      ^
                      |
        [Blue Slot]   |   [Red Peg]
           (y+)       |     (y-)
        ====[ ]=======|======[====]====  <- aligned along Y axis
                      |
        <-------------+-------------> Y
                   Robot
    ```
    
    **Success Conditions:**
    - Peg center passes through slot hole (aligned in Y)
    """
    
    # Use single bimanual YAM robot (which internally has two arms)
    SUPPORTED_ROBOTS = ["yam_bimanual"]
    agent: BimanualYAM
    
    # Task parameters - Peg (cuboid stick) dimensions
    # Peg is a long thin cuboid, oriented along Y axis
    peg_half_size = [0.008, 0.05, 0.008]  # 1.6cm x 10cm x 1.6cm (long along Y)
    
    # Slot (frame with hole) dimensions
    # Outer frame size
    slot_outer_half = [0.025, 0.015, 0.025]  # 5cm x 3cm x 5cm
    # Hole size (must be larger than peg cross-section)
    slot_hole_half = [0.018, 0.018]  # 2.4cm x 2.4cm hole (peg is 1.6cm x 1.6cm)
    
    insert_thresh = 0.02  # 2cm threshold for successful insertion
    
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
        """Load the scene with table, peg (cuboid), and slot (frame with hole)."""
        # Build table scene
        self.table_scene = TableSceneBuilder(
            self, 
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create RED PEG (cuboid stick) - for right arm (negative y side)
        # Peg is horizontal, aligned with Y axis
        self.peg = actors.build_box(
            self.scene,
            half_sizes=self.peg_half_size,
            color=[1, 0, 0, 1],  # Red
            name="peg",
            initial_pose=sapien.Pose(
                p=[0.1, -0.08, self.peg_half_size[2] + 0.001],  # Right side, on table
            ),
        )
        
        # Create BLUE SLOT (frame with hole) - for left arm (positive y side)
        # Slot is horizontal, hole extends along Y axis (so peg can pass through)
        self.slot = build_slot_with_hole(
            self.scene,
            name="slot",
            outer_half_size=self.slot_outer_half,
            hole_half_size=self.slot_hole_half,
            color=[0, 0, 1, 1],  # Blue
            initial_pose=sapien.Pose(
                p=[0.1, 0.08, self.slot_outer_half[2] + 0.001],  # Left side, on table
            ),
        )

        self._build_walls()
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with objects in front of robot center, aligned along Y axis."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0, 0.01]))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)
            
            # Peg position - right side (negative y), in front of robot
            # Horizontal along Y axis
            peg_xyz = torch.zeros((b, 3))
            peg_xyz[:, 0] = -0.3  # x: in front of robot
            peg_xyz[:, 1] = -0.2  # y: right side (for right arm)
            peg_xyz[:, 2] = self.peg_half_size[2] + 0.001  # z: on table
            peg_quat = euler2quat(0, 0, np.pi / 2)
            self.peg.set_pose(Pose.create_from_pq(peg_xyz, peg_quat))
            
            # Slot position - left side (positive y), in front of robot
            slot_xyz = torch.zeros((b, 3))
            slot_xyz[:, 0] = -0.3  # x: in front of robot
            slot_xyz[:, 1] = 0.2  # y: left side (for left arm)
            slot_xyz[:, 2] = self.slot_outer_half[2] + 0.001  # z: on table
            slot_quat = euler2quat(0, 0, 0)
            self.slot.set_pose(Pose.create_from_pq(slot_xyz, slot_quat))
    
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """Get extra observations."""
        obs = dict(
            peg_pose=self.peg.pose.raw_pose,
            slot_pose=self.slot.pose.raw_pose,
        )
        return obs
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate task success.
        
        Success: Peg is truly inserted through the slot hole.
        
        For true insertion along Y axis:
        - Peg X must be within slot hole X range (aligned)
        - Peg Z must be within slot hole Z range (aligned)  
        - Peg must SPAN the slot in Y direction (peg extends on both sides of slot)
        
        Slot is at y=slot_y, with thickness slot_outer_half[1]
        Peg center at y=peg_y, with half_length peg_half_size[1]
        
        True insertion means:
        - Peg tip (peg_y - peg_half_y) < slot front (slot_y - slot_half_y)
        - Peg end (peg_y + peg_half_y) > slot back (slot_y + slot_half_y)
        """
        # Get peg and slot positions
        peg_pos = self.peg.pose.p  # Center of peg
        slot_pos = self.slot.pose.p  # Center of slot
        
        # XZ alignment check - peg must fit in hole
        # Hole half size is slot_hole_half, peg cross-section is peg_half_size[0], peg_half_size[2]
        # Allow some tolerance
        dx = torch.abs(peg_pos[:, 0] - slot_pos[:, 0])
        dz = torch.abs(peg_pos[:, 2] - slot_pos[:, 2])
        
        # Max allowed offset = hole_half - peg_half (with small tolerance)
        max_offset_x = self.slot_hole_half[0] - self.peg_half_size[0] + 0.002
        max_offset_z = self.slot_hole_half[1] - self.peg_half_size[2] + 0.002
        
        aligned_x = dx <= max_offset_x
        aligned_z = dz <= max_offset_z
        
        # Y insertion check - peg must span through the slot
        peg_half_y = self.peg_half_size[1]
        slot_half_y = self.slot_outer_half[1]
        
        # Peg front tip (smaller y) and back end (larger y)
        peg_front = peg_pos[:, 1] - peg_half_y
        peg_back = peg_pos[:, 1] + peg_half_y
        
        # Slot front face and back face
        slot_front = slot_pos[:, 1] - slot_half_y
        slot_back = slot_pos[:, 1] + slot_half_y
        
        # True insertion: peg spans through slot
        # peg_front < slot_front AND peg_back > slot_back
        inserted_through = (peg_front < slot_front) & (peg_back > slot_back)
        
        # Combined success
        success = aligned_x & aligned_z & inserted_through
        
        # Distance metrics for reward
        dist_xz = torch.sqrt(dx**2 + dz**2)
        dist_y = torch.abs(peg_pos[:, 1] - slot_pos[:, 1])
        
        return {
            "success": success,
            "aligned_x": aligned_x,
            "aligned_z": aligned_z,
            "inserted_through": inserted_through,
            "dist_xz": dist_xz,
            "dist_y": dist_y,
        }
    
    def compute_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """Compute dense reward for training."""
        # Distance between peg and slot
        peg_pos = self.peg.pose.p
        slot_pos = self.slot.pose.p
        
        # Reward for XZ alignment (most important for insertion)
        dist_xz = info["dist_xz"]
        reward_align = 1 - torch.tanh(10 * dist_xz)
        
        # Reward for Y proximity (bringing peg to slot)
        dist_y = info["dist_y"]
        reward_approach = 1 - torch.tanh(5 * dist_y)
        
        # Combined reward
        reward = reward_align + reward_approach
        
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

