"""
Bimanual Lift Task for YAM Robot

A collaborative lifting task using two YAM arms working together.
Both arms must grasp and lift a long bar together.
"""

from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors

# Import BimanualYAM to ensure it's registered
from molmo_spaces_maniskill import MOLMOSPACES_ASSETS_DIR
from molmo_spaces_maniskill.assets.actor_loader import MjcfAssetActorLoader
from molmo_spaces_maniskill.assets.articulation_loader import MjcfAssetArticulationLoader
from molmo_spaces_maniskill.molmoact2.robots.yam import BimanualYAM


@register_env("BimanualYAMMicrowave-v1", max_episode_steps=200)
class BimanualYAMMicrowaveEnv(BaseEnv):
    """
    Minimal tabletop scene for the bimanual YAM robot.

    The environment contains the default ManiSkill table and a single THOR pot
    asset placed on top of it.
    """
    
    # Use single bimanual YAM robot (which internally has two arms)
    SUPPORTED_ROBOTS = ["yam_bimanual"]
    agent: BimanualYAM
    
    pot_spawn_height = 0.1
    lift_height_thresh = 0.06
    horizon_thresh_deg = 20.0
    # Microwave_1 receptacle in the microwave root frame.
    microwave_receptacle_center = np.array([-0.0631, 0.01919, -0.0555], dtype=np.float32)
    microwave_receptacle_half_size = np.array([0.20625, 0.1125, 0.12854], dtype=np.float32)
    
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
        self.pot_spawn_quat = np.array(euler2quat(np.pi / 2, 0, np.pi / 2), dtype=np.float32)
        spawn_rot = quat2mat(self.pot_spawn_quat)
        self.pot_upright_axis_idx = int(np.argmax(np.abs(spawn_rot[2])))
        self.pot_upright_axis_sign = float(
            np.sign(spawn_rot[2, self.pot_upright_axis_idx]) or 1.0
        )
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
        """Load the ManiSkill table and a single pot asset."""
        self.table_scene = TableSceneBuilder(
            self, 
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        microwave_xml = (
            MOLMOSPACES_ASSETS_DIR
            / "mjcf"
            / "objects"
            / "thor"
            / "Kitchen Objects"
            / "Microwave"
            / "Prefabs"
            / "Microwave_1"
            / "Microwave_1_mesh.xml"
        )

        bowl_xml = (
            MOLMOSPACES_ASSETS_DIR
            / "mjcf"
            / "objects"
            / "thor"
            / "Kitchen Objects"
            / "Bowl"
            / "Prefabs"
            / "Bowl_1"
            / "Bowl_1.xml"
        )

        bowl_loader = MjcfAssetActorLoader(self.scene)
        bowl_builder = bowl_loader.load_from_xml(bowl_xml, floating_base=True)
        try:
            self.bowl = bowl_builder.build(name="bowl")
        except TypeError:
            bowl_builder.set_name("bowl")
            self.bowl = bowl_builder.build()
        
        microwave_loader = MjcfAssetArticulationLoader(self.scene)
        microwave_builder = microwave_loader.load_from_xml(microwave_xml, floating_base=False)
        try:
            self.microwave = microwave_builder.build(name="microwave")
        except TypeError:
            self.microwave = microwave_builder.build()

        self._build_walls()
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset the table scene and place the pot above the tabletop."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0, 0.01]))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            microwave_xyz = torch.zeros((b, 3))
            microwave_xyz[:, 0] = -0.05 + (torch.rand((b,)) - 0.5) * 0.04
            microwave_xyz[:, 1] = (torch.rand((b,)) - 0.5) * 0.08
            microwave_xyz[:, 2] = 0.2
            microwave_q = torch.tensor(euler2quat(np.pi /2 , 0, -np.pi / 2), device=self.device, dtype=microwave_xyz.dtype)

            bowl_xyz = torch.zeros((b, 3))
            bowl_xyz[:, 0] = -0.3 + (torch.rand((b,)) - 0.5) * 0.04
            bowl_xyz[:, 1] = -0.4 + (torch.rand((b,)) - 0.5) * 0.08
            bowl_xyz[:, 2] = 0.06
            bowl_q = torch.tensor(euler2quat(np.pi /2 , 0, -np.pi / 2), device=self.device, dtype=bowl_xyz.dtype)

            self.microwave.set_pose(Pose.create_from_pq(microwave_xyz, microwave_q))
            microwave_qpos = self.microwave.get_qpos()
            microwave_qpos[...] = 0
            self.microwave.set_qpos(microwave_qpos)
            self.bowl.set_pose(Pose.create_from_pq(bowl_xyz, bowl_q))
    
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """Get extra observations."""
        return dict(microwave_pose=self.microwave.pose.raw_pose, bowl_pose=self.bowl.pose.raw_pose)

    def _bowl_in_microwave(self) -> torch.Tensor:
        """Check whether the bowl center lies inside the microwave receptacle box."""
        bowl_pos = self.bowl.pose.p
        microwave_tf = self.microwave.pose.to_transformation_matrix()
        microwave_rot = microwave_tf[..., :3, :3]
        microwave_pos = microwave_tf[..., :3, 3]

        bowl_local = torch.matmul(
            microwave_rot.transpose(-1, -2),
            (bowl_pos - microwave_pos).unsqueeze(-1),
        ).squeeze(-1)

        receptacle_center = torch.tensor(
            self.microwave_receptacle_center,
            device=self.device,
            dtype=bowl_local.dtype,
        )
        receptacle_half_size = torch.tensor(
            self.microwave_receptacle_half_size * 0.7,
            device=self.device,
            dtype=bowl_local.dtype,
        )

        local_offset = bowl_local - receptacle_center
        return (torch.abs(local_offset) <= receptacle_half_size).all(dim=1)
    
    def evaluate(self) -> Dict[str, Any]:
        """Success when the bowl is inside the microwave and both hands released it."""
        bowl_in_microwave = self._bowl_in_microwave()
        # left_grasped = self.agent.is_grasping(self.bowl, arm="left")
        # right_grasped = self.agent.is_grasping(self.bowl, arm="right")
        # released = ~(left_grasped | right_grasped)

        return {
            "success": bowl_in_microwave ,
            "bowl_in_microwave": bowl_in_microwave,
            # "left_grasped": left_grasped,
            # "right_grasped": right_grasped,
            # "released": released,
        }
    
    def compute_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """No task reward: this env only verifies asset loading."""
        return torch.zeros(self.num_envs, device=self.device)
    
    def compute_normalized_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """No normalized reward for the visualization-only env."""
        return torch.zeros(self.num_envs, device=self.device)

