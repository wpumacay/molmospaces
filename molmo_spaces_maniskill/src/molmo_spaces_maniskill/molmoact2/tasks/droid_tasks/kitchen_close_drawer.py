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


# Drawer hash from FloorPlan1 metadata. Same drawer family used in
# kitchen_open_drawer_pnp_fork — keeps the open/close pair on the same asset.
TARGET_DRAWER_NAME = "drawer_b4db64fe996f5874f518392f82883339_1_0_0"

# Drawer's open qpos at the start of the episode (mirrors the open task's value).
DRAWER_INIT_OPEN_QPOS = 0.26


@register_env("DroidKitchenCloseDrawer-v1")
class DroidKitchenCloseDrawerEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    A kitchen drawer starts in the open state; the robot must close it.

    **Success Conditions:**
    - The drawer prismatic joint qpos is below the closed threshold.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    drawer_closed_thresh = 0.03

    def __init__(
        self,
        *args,
        robot_uids="fr3_robotiq_wristcam",
        robot_init_qpos_noise=0.01,
        scene_file: Path | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ithor" / "FloorPlan1_physics.xml"
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
        self.drawer = None
        for name, art in self._env_articulations.items():
            if name.lower() == TARGET_DRAWER_NAME.lower():
                self.drawer = art
                break

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
            # Mirror the pose used by kitchen_open_drawer_pnp_fork.
            self.agent.robot.set_pose(
                sapien.Pose(p=[0.6, -1.7, 0.5], q=euler2quat(0, 0, -np.pi / 2))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.drawer is not None:
                qpos = self.drawer.get_qpos()
                qpos[..., 0] = DRAWER_INIT_OPEN_QPOS
                self.drawer.set_qpos(qpos)

    def _drawer_qpos(self) -> torch.Tensor:
        if self.drawer is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.drawer.get_qpos()[..., 0]

    def evaluate(self):
        drawer_qpos = self._drawer_qpos()
        drawer_closed = drawer_qpos < self.drawer_closed_thresh
        return {
            "success": drawer_closed,
            "drawer_qpos": drawer_qpos,
            "drawer_closed": drawer_closed,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
