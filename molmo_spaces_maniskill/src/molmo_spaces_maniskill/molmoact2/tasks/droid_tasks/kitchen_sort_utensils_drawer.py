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


# Use one of FloorPlan2's drawers as the storage target.
TARGET_DRAWER_NAME = "drawer_4f46a7c95efbc7c4096bd1cb7874b733_1_0_0"

# Drawer pre-opened qpos so success is achievable for the place phase.
DRAWER_INIT_OPEN_QPOS = 0.26


@register_env("DroidKitchenSortUtensilsDrawer-v1")
class DroidKitchenSortUtensilsDrawerEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Place fork, knife, and spoon into an opened drawer (the drawer starts open
    so the agent can focus on multi-object pnp + final closure). Then close
    the drawer.

    **Success Conditions:**
    - All three utensils are inside the drawer footprint, AND
    - The drawer is closed (joint qpos below threshold).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    drawer_closed_thresh = 0.03
    inside_xy_thresh = 0.18
    inside_z_thresh = 0.12

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
        self.drawer = None
        for name, art in self._env_articulations.items():
            if name.lower() == TARGET_DRAWER_NAME.lower():
                self.drawer = art
                break
        self.fork = None
        self.knife = None
        self.spoon = None
        for name, actor in self._env_actors.items():
            n = name.lower()
            if "fork_" in n and self.fork is None:
                self.fork = actor
            if "knife_" in n and "butter" not in n and self.knife is None:
                self.knife = actor
            if "spoon_" in n and self.spoon is None:
                self.spoon = actor

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
            # Drawer at world (+0.81, -1.16). Robot at (+0.81, -0.50) facing -y
            # places the drawer ~0.66 m straight ahead.
            self.agent.robot.set_pose(
                sapien.Pose(p=[0.81, -0.50, 0.50], q=euler2quat(0, 0, -np.pi / 2))
            )
            for name, actor in self._env_actors.items():
                if hasattr(actor, "initial_pose"):
                    actor.set_pose(actor.initial_pose)
            if self.drawer is not None:
                qpos = self.drawer.get_qpos()
                qpos[..., 0] = DRAWER_INIT_OPEN_QPOS
                self.drawer.set_qpos(qpos)

            # Default fork/spoon spawns are across the kitchen (~3 m from the
            # robot). Place them in workspace next to the open drawer.
            # Spawn utensils on the cabinet top above the drawer (drawer is
            # at z=0.48; cabinet face/top is around z=0.80). Earlier values
            # had y too shallow so utensils free-fell to the floor.
            utensil_pos = [
                (self.fork,  [0.71, -1.16, 0.95]),
                (self.knife, [0.81, -1.16, 0.95]),
                (self.spoon, [0.91, -1.16, 0.95]),
            ]
            for actor, p in utensil_pos:
                if actor is None:
                    continue
                q = actor.pose.q.clone()
                pos = torch.tensor([p], device=self.device).expand(self.num_envs, 3).clone()
                actor.set_pose(Pose.create_from_pq(pos, q))

    def _drawer_qpos(self) -> torch.Tensor:
        if self.drawer is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.drawer.get_qpos()[..., 0]

    def _item_inside_drawer(self, item) -> torch.Tensor:
        if self.drawer is None or item is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dr_pos = self.drawer.pose.p
        it_pos = item.pose.p
        xy = torch.linalg.norm(it_pos[..., :2] - dr_pos[..., :2], axis=1)
        dz = torch.abs(it_pos[..., 2] - dr_pos[..., 2])
        return (xy < self.inside_xy_thresh) & (dz < self.inside_z_thresh)

    def evaluate(self):
        drawer_qpos = self._drawer_qpos()
        drawer_closed = drawer_qpos < self.drawer_closed_thresh
        fork_in = self._item_inside_drawer(self.fork)
        knife_in = self._item_inside_drawer(self.knife)
        spoon_in = self._item_inside_drawer(self.spoon)
        success = fork_in & knife_in & spoon_in & drawer_closed
        return {
            "success": success,
            "drawer_qpos": drawer_qpos,
            "drawer_closed": drawer_closed,
            "fork_in_drawer": fork_in,
            "knife_in_drawer": knife_in,
            "spoon_in_drawer": spoon_in,
        }

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)
