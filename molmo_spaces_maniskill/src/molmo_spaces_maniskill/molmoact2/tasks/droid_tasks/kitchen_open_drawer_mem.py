
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose, Actor
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.agents.controllers.pd_ee_pose import PDEEPosController, PDEEPoseController

from molmo_spaces_maniskill.molmoact2.robots.fr3_wristcam import FR3WristCam as FR3RobotiqWristCam
from molmo_spaces_maniskill.core.env import MolmoSpacesEnv
from molmo_spaces_maniskill import MOLMOSPACES_MJCF_SCENES_DIR


def build_solid_color_target(
    scene,
    radius: float,
    thickness: float,
    name: str,
    color: np.ndarray,
    body_type: str = "dynamic",
    add_collision: bool = False,
    initial_pose: Optional[sapien.Pose] = None,
):
    """
    build a solid color target
    
    Args:
        scene: ManiSkill scene object
        radius: target half size
        thickness: target thickness
        name: actor name
        color: RGBA color array, range [0, 1]
        body_type: "dynamic", "kinematic", or "static"
        add_collision: whether to add collision
        initial_pose: initial pose
    """
    builder = scene.create_actor_builder()
    
    # only create a solid color block
    builder.add_box_visual(
        half_size=[radius, radius, thickness / 2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    if add_collision:
        builder.add_box_collision(half_size=[radius, radius, thickness / 2])
    
    # create actor based on body type
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    else:
        raise ValueError(f"Unknown body_type: {body_type}")
    
    if initial_pose is not None:
        actor.set_pose(initial_pose)
    
    return actor


# predefined color constants (for convenience)
COLORS = {
    'red': np.array([255, 0, 0, 255]) / 255,
    'green': np.array([0, 255, 0, 255]) / 255,
    'blue': np.array([0, 100, 255, 255]) / 255,
    'yellow': np.array([255, 255, 0, 255]) / 255,
    'purple': np.array([128, 0, 255, 255]) / 255,
    'orange': np.array([255, 165, 0, 255]) / 255,
    'cyan': np.array([0, 255, 255, 255]) / 255,
    'pink': np.array([255, 192, 203, 255]) / 255,
    'lime': np.array([50, 200, 50, 255]) / 255,
    'white': np.array([255, 255, 255, 255]) / 255,
    'black': np.array([0, 0, 0, 255]) / 255,
    'gray': np.array([128, 128, 128, 255]) / 255,
}


CANDIDATE_DRAWER_NAMES= [
    "drawer_b4db64fe996f5874f518392f82883339_1_0_0",
    "drawer_b02756d512ec441961b9a3e6c3ed803f_1_0_0",
    "drawer_372d9ee41d70550432c30a66a6e5b331_1_0_0",
    "drawer_72e4c03d9599586e003dc8b46b6dfc5f_1_0_0",
    "drawer_a037f6eea70627f4134c7adf6cd82205_1_0_0"
]


@register_env("DroidKitchenOpenDrawerMem-v1")
class DroidKitchenOpenDrawerMemEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Open the drawer, pick up the fork/spork from inside, and place it on the cooking pan.
    The fork starts inside the drawer, requiring the robot to first open the drawer.

    **Success Conditions:**
    - The fork is placed on the cooking pan (within distance threshold)
    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    # set some commonly used values
    patch_half_size = 0.1  # 10cm x 10cm patch
    patch_thickness = 0.005  # 5mm thick
    cube_half_size = 0.02
    drawer_open_thresh = 0.08
    drawer_init_open_qpos = 0.18

    def __init__(self, *args, robot_uids="fr3_robotiq_wristcam", robot_init_qpos_noise=0.01, scene_file: Path | None = None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ithor" / "FloorPlan1_physics.xml" 

        super().__init__(*args, robot_uids=robot_uids, scene_file=scene_file, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )


    @property
    def _default_sensor_configs(self):
        local_pose = sapien.Pose(
            p=[0.760345, -0.25817, 0.938223], 
            q=[-0.351141, 0.265775, 0.104661, -0.891689]
        )
        return CameraConfig(
            "thirdview_camera", 
            pose=local_pose, 
            width=640, 
            height=480, 
            fov=1.2, 
            near=0.01, 
            far=100,
            mount=self.agent.robot.links_map["base"]
        )

    @property
    def _default_human_render_camera_configs(self):
        # Camera pose relative to robot base
        # World pose: [0.439655, -1.44183, 0.938223], robot at [1.2, -1.7, 0.0] with 180° rotation
        # World diff: [-0.760345, 0.25817, 0.938223]
        # After 180° rotation (x,y negated): [0.760345, -0.25817, 0.938223]
        # Camera world quaternion: [0.891689, 0.104661, 0.265775, -0.351141]
        # Robot rotation (180° around Z): [0, 0, 1, 0] (w,x,y,z) -> need to transform camera q
        # Local q = robot_q_inv * camera_world_q
        # For 180° Z rotation: q_inv = [0, 0, -1, 0], result: [-0.351141, 0.265775, 0.104661, -0.891689]
        local_pose = sapien.Pose(
            p=[0.760345, -0.25817, 0.938223], 
            q=[-0.351141, 0.265775, 0.104661, -0.891689]
        )
        return CameraConfig(
            "render_camera", 
            pose=local_pose, 
            width=640, 
            height=480, 
            fov=1.2, 
            near=0.01, 
            far=100,
            mount=self.agent.robot.links_map["base"]
        )
    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[0., 0, 0.5]))

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        # Find and store references to the candidate drawers in top-to-bottom order.
        self.drawers = []
        for drawer_name in CANDIDATE_DRAWER_NAMES:
            drawer = None
            for name, articulation in self._env_articulations.items():
                if name.lower() == drawer_name.lower():
                    drawer = articulation
                    break
            self.drawers.append(drawer)

        self.top_drawer_idx = 0
        self.bottom_drawer_idx = len(self.drawers) - 1
        self.initial_open_drawer_idx = None
        self.target_drawer_closed_once = None
        self.bottom_drawer_opened_after_target = None
        
        # Find fork/spoon and cooking pan actors.
        self.fork = None
        self.spoon = None
        self.pan = None
        for name, actor in self._env_actors.items():
            name_lower = name.lower()
            if 'fork' in name_lower:
                self.fork = actor
                print(f"Found fork: {name}")
            if 'spoon' in name_lower:
                self.spoon = actor
                print(f"Found spoon: {name}")
                
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            # Initialize robot qpos and pose
            init_qpos = torch.tensor([[                   
                0.0,
                -1/5 * np.pi,
                0,
                -np.pi * 4 / 5,
                0,
                np.pi * 3 / 5,
                0,
                0, 0, 0, 0,  # gripper/passive joints
                0.04, 0.04
            ]])
            self.agent.robot.set_qpos(init_qpos)
            self.agent.robot.set_pose(sapien.Pose(
                p = [1.35, -1.7, 0.1], 
                q = euler2quat(0, 0, np.pi)))
            
            # Reset all scene actors to their initial poses
            # Objects loaded from MJCF scene are stored in self._env_actors
            for name, actor in self._env_actors.items():
                # Get initial pose from actor (if stored) or use current pose
                if hasattr(actor, 'initial_pose'):
                    actor.set_pose(actor.initial_pose)

            # Reset all candidate drawers to the closed state.
            for drawer in self.drawers:
                if drawer is None:
                    continue
                qpos = drawer.get_qpos()
                qpos[...] = 0
                drawer.set_qpos(qpos)

            # Open exactly one non-bottom drawer and remember which one it is.
            self.initial_open_drawer_idx = torch.randint(
                0,
                self.bottom_drawer_idx,
                (b,),
                device=self.device,
            )

            # self.initial_open_drawer_idx = torch.tensor([0], device=self.device)
            self.target_drawer_closed_once = torch.zeros(
                b, dtype=torch.bool, device=self.device
            )
            self.bottom_drawer_opened_after_target = torch.zeros(
                b, dtype=torch.bool, device=self.device
            )

            for drawer_idx, drawer in enumerate(self.drawers):
                if drawer is None:
                    continue
                qpos = drawer.get_qpos()
                open_mask = self.initial_open_drawer_idx == drawer_idx
                qpos[..., 0] = torch.where(
                    open_mask,
                    torch.full_like(qpos[..., 0], self.drawer_init_open_qpos),
                    torch.zeros_like(qpos[..., 0]),
                )
                drawer.set_qpos(qpos)

            # The fork and spoon are stored in the top drawer in this scene.
            # If the random drawer is the top drawer, translate them along world Y
            # by the same distance as the drawer opening so they remain inside it.
            top_drawer_open_mask = self.initial_open_drawer_idx == self.top_drawer_idx
            for actor in (self.fork, self.spoon):
                if actor is None:
                    continue
                actor_pos = actor.pose.p.clone()
                actor_quat = actor.pose.q.clone()
                actor_pos[top_drawer_open_mask, 1] += self.drawer_init_open_qpos
                actor.set_pose(Pose.create_from_pq(actor_pos, actor_quat))
    
    def _drawer_is_open(self, drawer) -> torch.Tensor:
        if drawer is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        qpos = drawer.get_qpos()
        return torch.abs(qpos[..., 0]) > self.drawer_open_thresh

    def evaluate(self):
        """
        Memory task:
        1. One non-bottom drawer starts open.
        2. The robot must close that drawer.
        3. Then open the bottom drawer.
        4. Finally reopen the drawer it closed before.
        """
        assert self.initial_open_drawer_idx is not None
        assert self.target_drawer_closed_once is not None
        assert self.bottom_drawer_opened_after_target is not None

        drawer_open_states = []
        for drawer in self.drawers:
            drawer_open_states.append(self._drawer_is_open(drawer))
        drawer_open_states = torch.stack(drawer_open_states, dim=1)

        batch_indices = torch.arange(self.num_envs, device=self.device)
        target_drawer_open = drawer_open_states[batch_indices, self.initial_open_drawer_idx]
        bottom_drawer_open = drawer_open_states[:, self.bottom_drawer_idx]
        open_drawer_count = drawer_open_states.sum(dim=1)
        only_target_and_bottom_open = (
            bottom_drawer_open
            & target_drawer_open
            & (open_drawer_count == 2)
        )

        self.target_drawer_closed_once |= ~target_drawer_open
        self.bottom_drawer_opened_after_target |= (
            self.target_drawer_closed_once & bottom_drawer_open
        )

        success = (
            self.target_drawer_closed_once
            & self.bottom_drawer_opened_after_target
            & only_target_and_bottom_open
        )

        return {
            "success": success,
            "target_drawer_idx": self.initial_open_drawer_idx,
            "bottom_drawer_open": bottom_drawer_open,
            "target_drawer_open": target_drawer_open,
            "only_target_and_bottom_open": only_target_and_bottom_open,
            "target_drawer_closed_once": self.target_drawer_closed_once,
            "bottom_drawer_opened_after_target": self.bottom_drawer_opened_after_target,
        }

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
