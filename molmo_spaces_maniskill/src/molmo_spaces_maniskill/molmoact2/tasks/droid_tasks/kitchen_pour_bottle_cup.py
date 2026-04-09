
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




@register_env("DroidKitchenPourBottleCup-v1")
class DroidKitchenPourBottleCupEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    Pour the bottle into the cup.

    **Success Conditions:**
    - the bottle is poured into the cup (within distance threshold)
    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    # set some commonly used values
    bottle_to_cup_align_thresh = float(np.cos(np.deg2rad(10.0)))
    bottle_horizon_thresh = float(np.sin(np.deg2rad(10.0)))
    bottle_to_cup_xy_thresh = 0.17
    bottle_to_cup_z_thresh = 0.13

    def __init__(self, *args, robot_uids="fr3_robotiq_wristcam", robot_init_qpos_noise=0.01, scene_file: Path | None = None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ithor" / "FloorPlan3_physics.xml" 

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

        return []

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


        # world_pose = sapien.Pose(
        # p = [0.937723, -1.02348, 2.03332], 
        # q =[-0.00352305, -0.618955, -0.0668718, 0.782567])
        # return CameraConfig(
        #     "render_camera", 
        #     pose=world_pose, 
        #     width=512, 
        #     height=512, 
        #     fov=1.2, 
        #     near=0.01, 
        #     far=100,
        #     # mount=self.agent.robot.links_map["base"]
        # )



        droid_exo_left_local = sapien.Pose(
            p=[-0.12, 0.32, 0.6], 
            q=[0.9622501868990583, 0.022557566113149834, 0.08418598282936919, -0.25783416049629954]
        )
        droid_exo_right_local = sapien.Pose(
            p=[-0.12, -0.32, 0.6], 
            q=[0.9622501868990583, -0.022557566113149838, 0.0841859828293692, 0.25783416049629954]
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
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[0., 0, 0.5]))

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.bottle = None
        self.cup = None
        for name, actor in self._env_actors.items():
            name_lower = name.lower()
            if "bottle" in name_lower and self.bottle is None:
                self.bottle = actor
            if "cup" in name_lower and self.cup is None:
                self.cup = actor
        
                
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
                p = [0.3, -0.9, 1.2], 
                q = euler2quat(0, 0, 0)))
            
            # Reset all scene actors to their initial poses
            # Objects loaded from MJCF scene are stored in self._env_actors
            for name, actor in self._env_actors.items():
                # Debug: print actor info
                if hasattr(actor, 'initial_pose'):
                    actor.set_pose(actor.initial_pose)
    
    def _bottle_pouring_into_cup(self):
        if self.bottle is None or self.cup is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        bottle_pos = self.bottle.pose.p
        cup_pos = self.cup.pose.p

        bottle_tf = self.bottle.pose.to_transformation_matrix()
        bottle_y_axis = bottle_tf[..., :3, 1]

        # "Point to" here means the projection of the line from bottle center to
        # cup center onto the world XY plane is aligned with the projection of the
        # bottle local Y-axis onto the world XY plane.
        bottle_to_cup_xy = cup_pos[..., :2] - bottle_pos[..., :2]
        bottle_to_cup_xy_dist = torch.linalg.norm(bottle_to_cup_xy, axis=1)
        safe_xy_dist = torch.clamp(bottle_to_cup_xy_dist, min=1e-6)
        bottle_to_cup_xy_dir = bottle_to_cup_xy / safe_xy_dist[:, None]
        bottle_y_axis_xy = bottle_y_axis[..., :2]
        bottle_y_axis_xy_norm = torch.linalg.norm(bottle_y_axis_xy, axis=1)
        safe_axis_xy_norm = torch.clamp(bottle_y_axis_xy_norm, min=1e-6)
        bottle_y_axis_xy_dir = bottle_y_axis_xy / safe_axis_xy_norm[:, None]

        bottle_y_axis_points_to_cup_in_xy = (
            torch.sum(bottle_y_axis_xy_dir * bottle_to_cup_xy_dir, dim=1)
            > self.bottle_to_cup_align_thresh
        )
        bottle_y_axis_near_horizon = (
            torch.abs(bottle_y_axis[..., 2])
            < self.bottle_horizon_thresh
        )
        bottle_close_to_cup = bottle_to_cup_xy_dist < self.bottle_to_cup_xy_thresh

        bottle_close_to_cup_z = torch.abs(bottle_pos[..., 2] - cup_pos[..., 2]) < self.bottle_to_cup_z_thresh

        return (
            bottle_y_axis_points_to_cup_in_xy
            & bottle_y_axis_near_horizon
            & bottle_close_to_cup
            & bottle_close_to_cup_z
        )

    def evaluate(self):
        success = self._bottle_pouring_into_cup()

        if self.bottle is None or self.cup is None:
            zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            false_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return {
                "success": false_mask,
                "bottle_y_axis_points_to_cup_in_xy": false_mask,
                "bottle_y_axis_near_horizon": false_mask,
                "bottle_close_to_cup": false_mask,
                "bottle_close_to_cup_z": false_mask,
                "bottle_y_axis_to_cup_xy_alignment": zeros,
                "bottle_y_axis_world_z_abs": zeros,
                "bottle_to_cup_xy_dist": zeros,
                "bottle_to_cup_z_dist": zeros,
            }

        bottle_pos = self.bottle.pose.p
        cup_pos = self.cup.pose.p
        bottle_tf = self.bottle.pose.to_transformation_matrix()
        bottle_y_axis = bottle_tf[..., :3, 1]
        bottle_to_cup_xy = cup_pos[..., :2] - bottle_pos[..., :2]
        bottle_to_cup_xy_dist = torch.linalg.norm(bottle_to_cup_xy, axis=1)
        safe_xy_dist = torch.clamp(bottle_to_cup_xy_dist, min=1e-6)
        bottle_to_cup_xy_dir = bottle_to_cup_xy / safe_xy_dist[:, None]
        bottle_y_axis_xy = bottle_y_axis[..., :2]
        bottle_y_axis_xy_norm = torch.linalg.norm(bottle_y_axis_xy, axis=1)
        safe_axis_xy_norm = torch.clamp(bottle_y_axis_xy_norm, min=1e-6)
        bottle_y_axis_xy_dir = bottle_y_axis_xy / safe_axis_xy_norm[:, None]
        bottle_y_axis_to_cup_xy_alignment = torch.sum(
            bottle_y_axis_xy_dir * bottle_to_cup_xy_dir, dim=1
        )
        bottle_y_axis_world_z_abs = torch.abs(bottle_y_axis[..., 2])
        bottle_close_to_cup_z = bottle_pos[..., 2] - cup_pos[..., 2] < self.bottle_to_cup_z_thresh
        bottle_y_axis_points_to_cup_in_xy = (
            bottle_y_axis_to_cup_xy_alignment > self.bottle_to_cup_align_thresh
        )
        bottle_y_axis_near_horizon = (
            bottle_y_axis_world_z_abs < self.bottle_horizon_thresh
        )
        bottle_close_to_cup = (
            bottle_to_cup_xy_dist < self.bottle_to_cup_xy_thresh
        )

        return {
            "success": success,
            "bottle_y_axis_points_to_cup_in_xy": bottle_y_axis_points_to_cup_in_xy,
            "bottle_y_axis_near_horizon": bottle_y_axis_near_horizon,
            "bottle_close_to_cup": bottle_close_to_cup,
            "bottle_close_to_cup_z": bottle_close_to_cup_z,
            "bottle_y_axis_to_cup_xy_alignment": bottle_y_axis_to_cup_xy_alignment,
            "bottle_y_axis_world_z_abs": bottle_y_axis_world_z_abs,
            "bottle_to_cup_xy_dist": bottle_to_cup_xy_dist,
        }

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
