from pathlib import Path
from typing import Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from molmo_spaces_maniskill import MOLMOSPACES_MJCF_SCENES_DIR
from molmo_spaces_maniskill.core.env import MolmoSpacesEnv
from molmo_spaces_maniskill.molmoact2.robots.fr3_wristcam import FR3WristCam as FR3RobotiqWristCam


@register_env("DroidKitchenPourBottlePot-v1")
class DroidKitchenPourBottlePotEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Pour from the bottle into the pot. Variant of pour-bottle-cup using a pot
    as target — wider opening but a smaller xy tolerance to make alignment
    matter.

    **Success Conditions:**
    - Bottle local Y-axis points toward pot in XY plane (alignment > threshold), AND
    - Bottle is roughly horizontal (Y-axis Z component small), AND
    - Bottle XY position close to pot XY position, AND
    - Bottle Z is close to pot Z.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    bottle_to_pot_align_thresh = float(np.cos(np.deg2rad(10.0)))
    bottle_horizon_thresh = float(np.sin(np.deg2rad(10.0)))
    bottle_to_pot_xy_thresh = 0.20
    bottle_to_pot_z_thresh = 0.18

    def __init__(
        self,
        *args,
        robot_uids="fr3_robotiq_wristcam",
        robot_init_qpos_noise=0.01,
        scene_file: Path | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ithor" / "FloorPlan3_physics.xml"
        super().__init__(*args, robot_uids=robot_uids, scene_file=scene_file, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        droid_exo_right_local = sapien.Pose(
            p=[-0.12, -0.32, 0.6],
            q=[0.9622501868990583, -0.022557566113149838, 0.0841859828293692, 0.25783416049629954],
        )
        return [
            CameraConfig(
                uid="render_camera",
                pose=droid_exo_right_local,
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],
            ),
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.5]))

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.bottle = None
        self.pot = None
        for name, actor in self._env_actors.items():
            name_lower = name.lower()
            if "bottle" in name_lower and "soap" not in name_lower and self.bottle is None:
                self.bottle = actor
            if "pot_" in name_lower and self.pot is None:
                self.pot = actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            init_qpos = torch.tensor(
                [
                    [
                        0.0, -np.pi / 5, 0.0, -np.pi * 4 / 5, 0.0, np.pi * 3 / 5, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.04, 0.04,
                    ]
                ]
            )
            self.agent.robot.set_qpos(init_qpos)
            # Same pose as kitchen_pour_bottle_cup so the bottle is already in
            # workspace (~0.7 m away). Pot's default spawn is across the room,
            # so we override it next.
            self.agent.robot.set_pose(
                sapien.Pose(p=[0.3, -0.9, 1.2], q=euler2quat(0, 0, 0))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.pot is not None:
                pot_quat = self.pot.pose.q.clone()
                pot_pos = torch.tensor([[0.55, -0.90, 1.40]], device=self.device).expand(self.num_envs, 3).clone()
                self.pot.set_pose(Pose.create_from_pq(pot_pos, pot_quat))

    def _bottle_pouring_into_pot(self) -> torch.Tensor:
        if self.bottle is None or self.pot is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        bottle_pos = self.bottle.pose.p
        pot_pos = self.pot.pose.p
        bottle_tf = self.bottle.pose.to_transformation_matrix()
        bottle_y_axis = bottle_tf[..., :3, 1]

        bottle_to_pot_xy = pot_pos[..., :2] - bottle_pos[..., :2]
        bottle_to_pot_xy_dist = torch.linalg.norm(bottle_to_pot_xy, axis=1)
        safe_xy = torch.clamp(bottle_to_pot_xy_dist, min=1e-6)
        bottle_to_pot_xy_dir = bottle_to_pot_xy / safe_xy[:, None]
        bottle_y_xy = bottle_y_axis[..., :2]
        bottle_y_xy_norm = torch.linalg.norm(bottle_y_xy, axis=1)
        safe_y = torch.clamp(bottle_y_xy_norm, min=1e-6)
        bottle_y_xy_dir = bottle_y_xy / safe_y[:, None]

        align_xy = torch.sum(bottle_y_xy_dir * bottle_to_pot_xy_dir, dim=1) > self.bottle_to_pot_align_thresh
        horizon = torch.abs(bottle_y_axis[..., 2]) < self.bottle_horizon_thresh
        close_xy = bottle_to_pot_xy_dist < self.bottle_to_pot_xy_thresh
        close_z = torch.abs(bottle_pos[..., 2] - pot_pos[..., 2]) < self.bottle_to_pot_z_thresh
        return align_xy & horizon & close_xy & close_z

    def evaluate(self):
        success = self._bottle_pouring_into_pot()
        return {"success": success}

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
