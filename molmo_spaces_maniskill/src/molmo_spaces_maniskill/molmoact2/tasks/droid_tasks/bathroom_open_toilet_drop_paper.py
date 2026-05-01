from pathlib import Path
from typing import Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from molmo_spaces_maniskill import MOLMOSPACES_MJCF_SCENES_DIR
from molmo_spaces_maniskill.core.env import MolmoSpacesEnv
from molmo_spaces_maniskill.molmoact2.robots.fr3_wristcam import FR3WristCam as FR3RobotiqWristCam


# Toilet hash for FloorPlan415 (a representative bathroom).
TARGET_TOILET_NAME = "crapper_cd6fa77f725b7ec4a4ced5913731ae93_1_0_0"


@register_env("DroidBathroomOpenToiletDropPaper-v1")
class DroidBathroomOpenToiletDropPaperEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Bathroom long-horizon: open the toilet lid, drop a roll of toilet paper
    inside the toilet bowl, then close the lid.

    **Success Conditions:**
    - The toilet paper is inside the toilet body footprint, AND
    - The toilet lid is closed (joint qpos near 0).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    lid_closed_thresh = 0.15
    paper_xy_thresh = 0.20
    paper_z_below_thresh = 0.30
    paper_z_above_thresh = 0.30

    def __init__(
        self,
        *args,
        robot_uids="fr3_robotiq_wristcam",
        robot_init_qpos_noise=0.01,
        scene_file: Path | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ithor" / "FloorPlan415_physics.xml"
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
        self.toilet = None
        for name, art in self._env_articulations.items():
            if name.lower() == TARGET_TOILET_NAME.lower():
                self.toilet = art
                break
        self.toilet_paper = None
        for name, actor in self._env_actors.items():
            n = name.lower()
            if ("toilettissue" in n or "toiletpaper" in n) and self.toilet_paper is None:
                self.toilet_paper = actor

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
            self.agent.robot.set_pose(
                sapien.Pose(p=[0.5, -1.5, 0.5], q=euler2quat(0, 0, -np.pi / 2))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.toilet is not None:
                qpos = self.toilet.get_qpos()
                qpos[...] = 0.0
                self.toilet.set_qpos(qpos)

    def _lid_qpos(self) -> torch.Tensor:
        if self.toilet is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.toilet.get_qpos()[..., 0]

    def _paper_in_toilet(self) -> torch.Tensor:
        if self.toilet is None or self.toilet_paper is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        t_pos = self.toilet.pose.p
        p_pos = self.toilet_paper.pose.p
        xy = torch.linalg.norm(p_pos[..., :2] - t_pos[..., :2], axis=1)
        dz = p_pos[..., 2] - t_pos[..., 2]
        return (
            (xy < self.paper_xy_thresh)
            & (dz > -self.paper_z_below_thresh)
            & (dz < self.paper_z_above_thresh)
        )

    def evaluate(self):
        lid_qpos = self._lid_qpos()
        paper_in = self._paper_in_toilet()
        # Toilet lid joint range can be negative — closed = abs(qpos) small.
        lid_closed = torch.abs(lid_qpos) < self.lid_closed_thresh
        success = paper_in & lid_closed
        return {
            "success": success,
            "lid_qpos": lid_qpos,
            "lid_closed": lid_closed,
            "paper_in_toilet": paper_in,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
