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


# FloorPlan2 has 13 articulated drawers; pick a source and a destination.
SOURCE_DRAWER_NAME = "drawer_4f46a7c95efbc7c4096bd1cb7874b733_1_0_0"
DEST_DRAWER_NAME = "drawer_5d94906c907342962358781e87f4d24d_1_0_0"

DRAWER_INIT_OPEN_QPOS = 0.26


@register_env("DroidKitchenDrawerToDrawerFork-v1")
class DroidKitchenDrawerToDrawerForkEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Move the fork from the source drawer to the destination drawer. Both
    drawers start closed. The agent must: open source -> pick fork -> close
    source -> open dest -> place fork -> close dest. (The episode does not
    enforce closing source mid-way; only the final state matters.)

    **Success Conditions:**
    - Fork is inside the destination drawer footprint, AND
    - Destination drawer is closed.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    drawer_closed_thresh = 0.03
    fork_xy_thresh = 0.18
    fork_z_thresh = 0.12

    def __init__(
        self,
        *args,
        robot_uids="fr3_robotiq_wristcam",
        robot_init_qpos_noise=0.01,
        scene_file: Path | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ithor" / "FloorPlan2_physics.xml"
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
        self.drawer_src = None
        self.drawer_dst = None
        for name, art in self._env_articulations.items():
            n = name.lower()
            if n == SOURCE_DRAWER_NAME.lower():
                self.drawer_src = art
            if n == DEST_DRAWER_NAME.lower():
                self.drawer_dst = art
        self.fork = None
        for name, actor in self._env_actors.items():
            if "fork_" in name.lower() and self.fork is None:
                self.fork = actor

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
                sapien.Pose(p=[0.6, -1.7, 0.5], q=euler2quat(0, 0, -np.pi / 2))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.drawer_src is not None:
                qpos = self.drawer_src.get_qpos()
                qpos[..., 0] = DRAWER_INIT_OPEN_QPOS
                self.drawer_src.set_qpos(qpos)
            if self.drawer_dst is not None:
                qpos = self.drawer_dst.get_qpos()
                qpos[...] = 0.0
                self.drawer_dst.set_qpos(qpos)

    def _drawer_qpos(self, drawer) -> torch.Tensor:
        if drawer is None:
            return torch.zeros(self.num_envs, device=self.device)
        return drawer.get_qpos()[..., 0]

    def _fork_in_dest(self) -> torch.Tensor:
        if self.drawer_dst is None or self.fork is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        d_pos = self.drawer_dst.pose.p
        f_pos = self.fork.pose.p
        xy = torch.linalg.norm(f_pos[..., :2] - d_pos[..., :2], axis=1)
        dz = torch.abs(f_pos[..., 2] - d_pos[..., 2])
        return (xy < self.fork_xy_thresh) & (dz < self.fork_z_thresh)

    def evaluate(self):
        src_qpos = self._drawer_qpos(self.drawer_src)
        dst_qpos = self._drawer_qpos(self.drawer_dst)
        dst_closed = dst_qpos < self.drawer_closed_thresh
        fork_in_dest = self._fork_in_dest()
        success = fork_in_dest & dst_closed
        return {
            "success": success,
            "src_drawer_qpos": src_qpos,
            "dst_drawer_qpos": dst_qpos,
            "dst_drawer_closed": dst_closed,
            "fork_in_dest_drawer": fork_in_dest,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
