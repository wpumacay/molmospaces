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


TARGET_DISHWASHER_NAME = "dishwasher_def5a30c2a98791d34988c0910c708d7_1_0_0"


@register_env("DroidKitchenOpenDishwasherPnpPlate-v1")
class DroidKitchenOpenDishwasherPnpPlateEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Open the dishwasher, place the plate inside the dishwasher rack, then close
    the dishwasher.

    **Success Conditions:**
    - The plate is placed inside the dishwasher (xy/z near dishwasher body), AND
    - The dishwasher door is closed (joint angle below the closed threshold).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    door_closed_thresh = 0.15
    plate_xy_thresh = 0.30
    plate_z_below_thresh = 0.10
    plate_z_above_thresh = 0.30

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
        self.dishwasher = None
        for name, art in self._env_articulations.items():
            if name.lower() == TARGET_DISHWASHER_NAME.lower():
                self.dishwasher = art
                break
        self.plate = None
        for name, actor in self._env_actors.items():
            if "plate" in name.lower() and self.plate is None:
                self.plate = actor

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
            # FP1 dishwasher is at (-1.76, -0.72, 0.0) — robot 0.65 m in front,
            # facing -x so the door swings toward the robot.
            self.agent.robot.set_pose(
                sapien.Pose(p=[-1.10, -0.72, 0.50], q=euler2quat(0, 0, np.pi))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.dishwasher is not None:
                qpos = self.dishwasher.get_qpos()
                qpos[...] = 0.0
                self.dishwasher.set_qpos(qpos)
            if self.plate is not None:
                plate_quat = self.plate.pose.q.clone()
                # Drop plate right above the dishwasher front (its top sits at
                # roughly z=0.85 in FP1) so it lands on the dishwasher housing
                # within easy reach.
                plate_pos = torch.tensor([[-1.65, -0.72, 1.00]], device=self.device).expand(self.num_envs, 3).clone()
                self.plate.set_pose(Pose.create_from_pq(plate_pos, plate_quat))

    def _door_qpos(self) -> torch.Tensor:
        if self.dishwasher is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.dishwasher.get_qpos()[..., 0]

    def _plate_inside_dishwasher(self) -> torch.Tensor:
        if self.dishwasher is None or self.plate is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dw_pos = self.dishwasher.pose.p
        plate_pos = self.plate.pose.p
        xy_dist = torch.linalg.norm(plate_pos[..., :2] - dw_pos[..., :2], axis=1)
        z_diff = plate_pos[..., 2] - dw_pos[..., 2]
        return (
            (xy_dist < self.plate_xy_thresh)
            & (z_diff > -self.plate_z_below_thresh)
            & (z_diff < self.plate_z_above_thresh)
        )

    def evaluate(self):
        door_qpos = self._door_qpos()
        plate_inside = self._plate_inside_dishwasher()
        door_closed = door_qpos < self.door_closed_thresh
        success = plate_inside & door_closed
        return {
            "success": success,
            "door_qpos": door_qpos,
            "door_closed": door_closed,
            "plate_inside": plate_inside,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
