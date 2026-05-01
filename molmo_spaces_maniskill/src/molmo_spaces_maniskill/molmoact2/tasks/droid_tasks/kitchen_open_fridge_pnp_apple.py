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


TARGET_FRIDGE_NAME = "refrigerator_f0e3edccfd9f3c40c91e8cfc15861ab6_1_0_0"


@register_env("DroidKitchenOpenFridgePnpApple-v1")
class DroidKitchenOpenFridgePnpAppleEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Open the fridge door, place the apple inside, then close the fridge door.
    The fridge door has a strong hinge — closing it requires sustained contact.

    **Success Conditions:**
    - The apple is inside the fridge body footprint, AND
    - The fridge door is closed (joint angle below the closed threshold).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    door_closed_thresh = 0.15
    apple_xy_thresh = 0.35
    apple_z_below_thresh = 0.50
    apple_z_above_thresh = 0.50

    # FP2 fridge has 4 active joints. [0] and [1] are prismatic interior drawers;
    # [2] and [3] are revolute_unwrapped doors with range [-pi/2, 0]. We track
    # the main door at index 2.
    DOOR_JOINT_INDEX = 2

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
        self.fridge = None
        for name, art in self._env_articulations.items():
            if name.lower() == TARGET_FRIDGE_NAME.lower():
                self.fridge = art
                break
        self.apple = None
        for name, actor in self._env_actors.items():
            if "apple" in name.lower() and self.apple is None:
                self.apple = actor

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
            # FP2 fridge is at (-1.76, 0.0, 0.9). Robot 0.65 m in front, facing -x.
            self.agent.robot.set_pose(
                sapien.Pose(p=[-1.10, 0.0, 0.50], q=euler2quat(0, 0, np.pi))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.fridge is not None:
                qpos = self.fridge.get_qpos()
                qpos[...] = 0.0
                self.fridge.set_qpos(qpos)
            if self.apple is not None:
                apple_quat = self.apple.pose.q.clone()
                # Drop apple right above the fridge so it lands on top of the
                # fridge body (~z=0.95 plus a drop) instead of bouncing away.
                apple_pos = torch.tensor([[-1.65, 0.0, 1.05]], device=self.device).expand(self.num_envs, 3).clone()
                self.apple.set_pose(Pose.create_from_pq(apple_pos, apple_quat))

    def _door_qpos(self) -> torch.Tensor:
        if self.fridge is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.fridge.get_qpos()[..., self.DOOR_JOINT_INDEX]

    def _apple_inside_fridge(self) -> torch.Tensor:
        if self.fridge is None or self.apple is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fr_pos = self.fridge.pose.p
        ap_pos = self.apple.pose.p
        xy_dist = torch.linalg.norm(ap_pos[..., :2] - fr_pos[..., :2], axis=1)
        z_diff = ap_pos[..., 2] - fr_pos[..., 2]
        return (
            (xy_dist < self.apple_xy_thresh)
            & (z_diff > -self.apple_z_below_thresh)
            & (z_diff < self.apple_z_above_thresh)
        )

    def evaluate(self):
        door_qpos = self._door_qpos()
        apple_inside = self._apple_inside_fridge()
        # Door qpos is in [-pi/2, 0]; closed = magnitude near 0.
        door_closed = torch.abs(door_qpos) < self.door_closed_thresh
        success = apple_inside & door_closed
        return {
            "success": success,
            "door_qpos": door_qpos,
            "door_closed": door_closed,
            "apple_inside": apple_inside,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
