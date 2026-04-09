
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

TARGET_DRAWER_NAME = CANDIDATE_DRAWER_NAMES[0]


@register_env("DroidKitchenOpenDrawerPnpFork-v1")
class DroidKitchenOpenDrawerPnpForkEnv(MolmoSpacesEnv):
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
    drawer_init_open_qpos = 0.26  # drawer open position (same as kitchen_open_drawer_mem)

    def __init__(self, *args, robot_uids="fr3_robotiq_wristcam", robot_init_qpos_noise=0.01, scene_file: Path | None = None, start_from_stage2: bool = True, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.start_from_stage2 = start_from_stage2  # if True, start with drawer open and robot at stage 2 position
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
        local_pose = sapien.Pose(
            p=[0.760345, -0.25817, 0.938223], 
            q=[-0.351141, 0.265775, 0.104661, -0.891689]
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
        
        # Find and store reference to drawer articulation
        self.drawer = None
        for name, articulation in self._env_articulations.items():
            if name.lower() == TARGET_DRAWER_NAME.lower():
                self.drawer = articulation
                break
        
        # Find fork/spork and cooking pan actors
        self.fork = None
        self.pan = None
        for name, actor in self._env_actors.items():
            name_lower = name.lower()
            if 'fork' in name_lower:
                self.fork = actor
                print(f"Found fork: {name}")
            if 'cookingpan' in name_lower or 'cooking_pan' in name_lower or 'pan' in name_lower:
                self.pan = actor
                print(f"Found pan: {name}")
                
                
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            if self.start_from_stage2:
                # Stage 2 test mode: robot qpos from episode 0 at 55% progress (drawer already opened)
                # This qpos is extracted from the dataset at the moment when stage 1 (open drawer) is done
                init_qpos = torch.tensor([[
                    0.276107, -0.371688, 0.324967, -2.120251, 0.149047, 1.737838, 0.490432,
                    0.0, 0.0, 0.0, 0.0,  # gripper/passive joints (open)
                    0.04, 0.04
                ]])
            else:
                # Default: start from initial pose for full task
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
                p = [0.6, -1.7, 0.5], 
                q = euler2quat(0, 0, -np.pi/2)))
            
            # Reset all scene actors to their initial poses
            # Objects loaded from MJCF scene are stored in self._env_actors
            for name, actor in self._env_actors.items():
                # Get initial pose from actor (if stored) or use current pose
                if hasattr(actor, 'initial_pose'):
                    actor.set_pose(actor.initial_pose)

            # Set drawer state based on mode
            if self.drawer is not None:
                qpos = self.drawer.get_qpos()
                if self.start_from_stage2:
                    # Stage 2 test: drawer is already open
                    qpos[..., 0] = self.drawer_init_open_qpos
                else:
                    # Default: drawer closed
                    qpos[...] = 0
                self.drawer.set_qpos(qpos)
            
            # If starting from stage 2 and fork exists, move fork along with opened drawer
            if self.start_from_stage2 and self.fork is not None:
                fork_pos = self.fork.pose.p.clone()
                fork_quat = self.fork.pose.q.clone()
                # Move fork along Y axis by drawer opening distance (fork is inside drawer)
                fork_pos[:, 1] += self.drawer_init_open_qpos
                from mani_skill.utils.structs import Pose
                self.fork.set_pose(Pose.create_from_pq(fork_pos, fork_quat))
                

    def _load_lighting(self, options):
        self.enable_shadow

        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )


    def _fork_on_pan(self) -> torch.Tensor:
        """Check if fork is on the pan."""
        if self.fork is None or self.pan is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        fork_pos = self.fork.pose.p
        pan_pos = self.pan.pose.p
        
        # XY distance between fork and pan center
        xy_dist = torch.linalg.norm(fork_pos[:, :2] - pan_pos[:, :2], axis=1)
        
        # Z difference (fork should be above or on pan)
        z_diff = fork_pos[:, 2] - pan_pos[:, 2]
        
        # Success: fork is close to pan in XY (within 15cm) and slightly above (0-10cm)
        on_pan = (xy_dist < 0.15) & (z_diff > 0) & (z_diff < 0.1)
        
        return on_pan
    
    def evaluate(self):
        """
        Success: Fork is placed on the cooking pan.
        """
        # Check if fork is on pan
        fork_on_pan = self._fork_on_pan()
        
        # Optional: also check robot is not grasping (fork is released)
        # For now, just check position
        success = fork_on_pan
        
        return {
            "success": success,
            "fork_on_pan": fork_on_pan,
        }

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
