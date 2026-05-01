from pathlib import Path
from typing import Optional, Union

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


# Microwave hash for FloorPlan3 (from FloorPlan3_physics_metadata.json).
TARGET_MICROWAVE_NAME = "microwaveoven_556431620f1060c7bd31f2fd7aec68f4_1_0_0"


@register_env("DroidKitchenHeatMugMicrowave-v1")
class DroidKitchenHeatMugMicrowaveEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Open the microwave door, place the mug inside, then close the microwave door.

    **Success Conditions:**
    - The mug is inside the microwave receptacle (within xy/z tolerance of the
      microwave body), AND
    - The microwave door is closed (joint angle below the closed threshold).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    door_open_thresh = 1.0
    door_closed_thresh = 0.15
    mug_xy_thresh = 0.20
    mug_z_below_thresh = 0.05
    mug_z_above_thresh = 0.20

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
        # Same exo-right framing as kitchen_pour_bottle_cup for demo consistency.
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

        self.microwave = None
        for name, art in self._env_articulations.items():
            if name.lower() == TARGET_MICROWAVE_NAME.lower():
                self.microwave = art
                break

        # Prefer Mug, fall back to Cup if no mug actor matches.
        self.mug = None
        for name, actor in self._env_actors.items():
            if "mug" in name.lower() and self.mug is None:
                self.mug = actor
        if self.mug is None:
            for name, actor in self._env_actors.items():
                if "cup" in name.lower() and self.mug is None:
                    self.mug = actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            init_qpos = torch.tensor(
                [
                    [
                        0.0,
                        -np.pi / 5,
                        0.0,
                        -np.pi * 4 / 5,
                        0.0,
                        np.pi * 3 / 5,
                        0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.04, 0.04,
                    ]
                ]
            )
            self.agent.robot.set_qpos(init_qpos)
            # Robot positioned in front of FloorPlan3 microwave (world ~[1.0, -2.16, 1.49]).
            self.agent.robot.set_pose(
                sapien.Pose(
                    p=[0.5, -1.6, 1.0],
                    q=euler2quat(0, 0, -np.pi / 2),
                )
            )

            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)

            if self.microwave is not None:
                qpos = self.microwave.get_qpos()
                qpos[...] = 0.0
                self.microwave.set_qpos(qpos)

    def _door_qpos(self) -> torch.Tensor:
        if self.microwave is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.microwave.get_qpos()[..., 0]

    def _mug_inside_microwave(self) -> torch.Tensor:
        if self.microwave is None or self.mug is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mw_pos = self.microwave.pose.p
        mug_pos = self.mug.pose.p
        xy_dist = torch.linalg.norm(mug_pos[..., :2] - mw_pos[..., :2], axis=1)
        z_diff = mug_pos[..., 2] - mw_pos[..., 2]
        return (
            (xy_dist < self.mug_xy_thresh)
            & (z_diff > -self.mug_z_below_thresh)
            & (z_diff < self.mug_z_above_thresh)
        )

    def evaluate(self):
        door_qpos = self._door_qpos()
        mug_inside = self._mug_inside_microwave()
        door_closed = door_qpos < self.door_closed_thresh
        door_open = door_qpos > self.door_open_thresh
        success = mug_inside & door_closed
        return {
            "success": success,
            "door_qpos": door_qpos,
            "door_open": door_open,
            "door_closed": door_closed,
            "mug_inside_microwave": mug_inside,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
