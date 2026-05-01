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


@register_env("DroidKitchenSetTable-v1")
class DroidKitchenSetTableEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Arrange a place setting: place the fork to the left of the plate, the
    knife to the right, and the cup at the upper-right relative to the plate.

    **Success Conditions (relative-pose, all simultaneously):**
    - Fork is to the left of plate (negative robot-Y offset within an xy radius)
    - Knife is to the right of plate (positive robot-Y offset within an xy radius)
    - Cup/Mug is upper-right of plate
    - All three items at roughly plate height
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    side_offset = 0.18
    side_tol = 0.10
    z_tol = 0.10

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
        self.plate = None
        self.fork = None
        self.knife = None
        self.cup = None
        for name, actor in self._env_actors.items():
            name_lower = name.lower()
            if "plate_" in name_lower and self.plate is None:
                self.plate = actor
            if "fork_" in name_lower and self.fork is None:
                self.fork = actor
            if "knife_" in name_lower and "butter" not in name_lower and self.knife is None:
                self.knife = actor
            if "cup_" in name_lower and "stack" not in name_lower and self.cup is None:
                self.cup = actor
        # Fallback to mug if cup wasn't found.
        if self.cup is None:
            for name, actor in self._env_actors.items():
                if "mug_" in name.lower() and self.cup is None:
                    self.cup = actor

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
                sapien.Pose(p=[0.3, -0.9, 1.2], q=euler2quat(0, 0, 0))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)

    def _placement_correct(self, item, direction_xy, expect_offset_x, expect_offset_y) -> torch.Tensor:
        """Item should be at plate_pos + (expect_offset_x, expect_offset_y) within a tolerance."""
        if self.plate is None or item is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        plate_pos = self.plate.pose.p
        item_pos = item.pose.p
        dx = item_pos[..., 0] - plate_pos[..., 0]
        dy = item_pos[..., 1] - plate_pos[..., 1]
        dz = torch.abs(item_pos[..., 2] - plate_pos[..., 2])
        ok_x = torch.abs(dx - expect_offset_x) < self.side_tol
        ok_y = torch.abs(dy - expect_offset_y) < self.side_tol
        ok_z = dz < self.z_tol
        return ok_x & ok_y & ok_z

    def evaluate(self):
        # Robot front faces +x in the scene; "left of plate" = -y, "right" = +y.
        fork_ok = self._placement_correct(self.fork, None, 0.0, -self.side_offset)
        knife_ok = self._placement_correct(self.knife, None, 0.0, +self.side_offset)
        cup_ok = self._placement_correct(self.cup, None, +self.side_offset, +self.side_offset)
        success = fork_ok & knife_ok & cup_ok
        return {
            "success": success,
            "fork_left_of_plate": fork_ok,
            "knife_right_of_plate": knife_ok,
            "cup_upper_right_of_plate": cup_ok,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
