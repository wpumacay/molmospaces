
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

@register_env("DroidAi2LabOpenOven-v1")
class DroidAi2LabOpenOvenEnv(MolmoSpacesEnv):
    """
    **Task Description:**
    A simple task where the objective is to pick up a cube and place it on a patch. 
    The cube is in front of the robot, the patch is on the left side.

    **Randomizations:**
    - the cube's xy position is slightly randomized on top of a table in front of the robot
    - the cube's z-axis rotation is randomized in [-pi/4, pi/4]

    **Success Conditions:**
    - the cube is placed on the patch (within xy bounds and on the surface)
    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    # set some commonly used values
    patch_half_size = 0.1  # 10cm x 10cm patch
    patch_thickness = 0.005  # 5mm thick
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="fr3_robotiq_wristcam", robot_init_qpos_noise=0.01, scene_file: Path | None = None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        scene_file = MOLMOSPACES_MJCF_SCENES_DIR / "ai2lab" / "kitchen" / "scene_no_table.xml"

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
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        
        droid_exo_left_local = sapien.Pose(
            p=[-0.12, 0.32, 0.6], 
            q=[0.9622501868990583, 0.022557566113149834, 0.08418598282936919, -0.25783416049629954]
        )
        droid_exo_right_local = sapien.Pose(
            p=[-0.12, -0.32, 0.6], 
            q=[0.9622501868990583, -0.022557566113149838, 0.0841859828293692, 0.25783416049629954]
        )

        pose_base = sapien_utils.look_at([0.0, 0.0, 0.45], [-0.3, 0.0, 0.08])
        pose_steer_left = sapien_utils.look_at([-0.15, -0.35, 0.4], [-0.4, 0.0, -0.1])
        pose_steer_right = sapien_utils.look_at([-0.15, 0.35, 0.4], [-0.4, 0.0, -0.1])
        
        return [
            # CameraConfig(
            #     "base_camera",
            #     pose=droid_exo_left_local,
            #     width=256,
            #     height=256,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            #     mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            # ),
            CameraConfig(
                uid="droid_exo_left_camera",
                pose=droid_exo_left_local,  # 直接使用局部位姿
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            ),
            CameraConfig(
                uid="droid_exo_right_camera",
                pose=droid_exo_right_local,  # 添加右相机
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            ),

        ]

    @property
    def _default_human_render_camera_configs(self):
        # Camera pose relative to robot base
        # World pose was [2.298, -1.72619, 1.38011], robot at [1.8, -1.2, 0.45]
        # Local = World - Robot
        local_pose = sapien.Pose(
            p=[0.498, -0.52619, 0.93011], 
            q=[0.583889, -0.226501, 0.174033, 0.759923]
        )
        return CameraConfig(
            "render_camera", 
            pose=local_pose, 
            width=512, 
            height=512, 
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
        
        # Find and store reference to stove articulation for success checking
        self.stove = None
        for name, articulation in self._env_articulations.items():
            if "stove" in name.lower():
                self.stove = articulation
                break

        
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
                p = [1.7, -5.0, 0.35], 
                q = euler2quat(0, 0, 0)))
            
            # Reset all scene actors to their initial poses
            # Objects loaded from MJCF scene are stored in self._env_actors
            for name, actor in self._env_actors.items():
                # Get initial pose from actor (if stored) or use current pose
                if hasattr(actor, 'initial_pose'):
                    actor.set_pose(actor.initial_pose)
            
            # Get current qpos and set all joints to 0
            if self.stove is not None:
                qpos = self.stove.get_qpos()
                qpos[...] = 0  # Reset all joints to 0 (closed)
                self.stove.set_qpos(qpos)
                



    def evaluate(self):
        # Success is achieved when Door_01 is open > 0.3 radians
        # Door_01_joint is at index 1 in the stove articulation qpos
        if self.stove is not None:
            stove_qpos = self.stove.get_qpos()
            # Door_01_joint is index 1 (based on the joint list from screenshot)
            door_01_angle = stove_qpos[..., 1]  # Shape: (num_envs,) or scalar
            
            # Convert to tensor if needed
            if not isinstance(door_01_angle, torch.Tensor):
                door_01_angle = torch.tensor(door_01_angle, device=self.device)
            
            # Ensure correct shape
            if door_01_angle.dim() == 0:
                door_01_angle = door_01_angle.unsqueeze(0)
            
            success = door_01_angle > 0.3
        else:
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
