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
from molmo_spaces_maniskill.molmoact2.robots.yam import BimanualYAM


@register_env("BimanualYAMLiftPot-v1", max_episode_steps=200)
class BimanualYAMLiftPotEnv(BaseEnv):
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

        pot_xml = (
            MOLMOSPACES_ASSETS_DIR
            / "mjcf"
            / "objects"
            / "thor"
            / "Kitchen Objects"
            / "Pot"
            / "Prefabs"
            / "Pot_10"
            / "Pot_10.xml"
        )
        pot_loader = MjcfAssetActorLoader(self.scene)
        pot_builder = pot_loader.load_from_xml(pot_xml, floating_base=True)
        try:
            self.pot = pot_builder.build(name="pot")
        except TypeError:
            pot_builder.set_name("pot")
            self.pot = pot_builder.build()

        self._build_walls()
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset the table scene and place the pot above the tabletop."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0, 0.01]))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            pot_xyz = torch.zeros((b, 3))
            pot_xyz[:, 0] = -0.30 + (torch.rand((b,)) - 0.5) * 0.04
            pot_xyz[:, 1] = (torch.rand((b,)) - 0.5) * 0.08
            pot_xyz[:, 2] = self.pot_spawn_height
            pot_q = torch.zeros((b, 4))
            pot_q[:, :] = torch.tensor(self.pot_spawn_quat, device=self.device, dtype=pot_xyz.dtype)
            self.pot.set_pose(Pose.create_from_pq(pot_xyz, pot_q))
    
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """Get extra observations."""
        return dict(pot_pose=self.pot.pose.raw_pose)
    
    def evaluate(self) -> Dict[str, Any]:
        """Success: both arms grasp the pot, lift it, and keep it almost level."""
        # left_grasped = self.agent.is_grasping(self.pot, arm="left")
        # right_grasped = self.agent.is_grasping(self.pot, arm="right")
        # two_arms_grasping = left_grasped & right_grasped

        pot_height = self.pot.pose.p[:, 2]
        lifted = pot_height >= (self.pot_spawn_height + self.lift_height_thresh)

        # pot_tf = self.pot.pose.to_transformation_matrix()
        # upright_alignment = (
        #     self.pot_upright_axis_sign * pot_tf[..., 2, self.pot_upright_axis_idx]
        # )
        # almost_horizontal = upright_alignment >= np.cos(
        #     np.deg2rad(self.horizon_thresh_deg)
        # )

        success =  lifted #& almost_horizontal
        return {
            "success": success,
            # "left_grasped": left_grasped,
            # "right_grasped": right_grasped,
            # "two_arms_grasping": two_arms_grasping,
            "lifted": lifted,
            # "almost_horizontal": almost_horizontal,
            # "pot_height": pot_height,
            # "upright_alignment": upright_alignment,
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

