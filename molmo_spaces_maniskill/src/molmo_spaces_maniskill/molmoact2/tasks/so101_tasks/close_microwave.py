"""
Close microwave task for the SO100 robot.

The robot starts with the microwave door partially open and must close it.
"""

from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors


from molmo_spaces_maniskill import MOLMOSPACES_ASSETS_DIR
from molmo_spaces_maniskill.assets.articulation_loader import MjcfAssetArticulationLoader
from molmo_spaces_maniskill.molmoact2.robots.so100.so100_wristcam import SO100WristCam



@register_env("SO100CloseMicrowave-v1", max_episode_steps=200)
class SO100CloseMicrowaveEnv(BaseEnv):
    """
    Minimal tabletop scene for closing a microwave with the SO100 robot.
    """
    
    SUPPORTED_ROBOTS = ["so100_wristcam"]
    agent: SO100WristCam

    microwave_init_open_angle = 0.5
    microwave_closed_thresh = 0.08
    button_contact_force_thresh = 0.5
    button_center = np.array([-0.5, -0.18, 0.0], dtype=np.float32)
    button_base_half_size = np.array([0.03, 0.03, 0.01], dtype=np.float32)
    button_cap_half_size = np.array([0.018, 0.018, 0.008], dtype=np.float32)
    
    def __init__(
        self, 
        *args, 
        robot_uids="so100_wristcam",
        robot_init_qpos_noise=0.02,
        reset_robot_qpos=True,  # Whether to reset robot qpos to home on episode init
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reset_robot_qpos = reset_robot_qpos
        self.button_pressed_once = None
        self._prev_elapsed_steps = None  # Track previous step count to detect new episodes
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def reset(self, seed=None, options=None):
        """Reset the environment and clear episode state."""
        import traceback
        print(f"[DEBUG] reset() CALLED! Stack trace:")
        traceback.print_stack(limit=5)
        
        # Reset the button_pressed_once state BEFORE calling super().reset()
        # This ensures it's reset every episode, not just on reconfigure
        if self.button_pressed_once is not None:
            self.button_pressed_once[:] = False
            print(f"[DEBUG] reset(): button_pressed_once reset to {self.button_pressed_once}")
        
        # Also reset the step tracker
        self._prev_elapsed_steps = None
        
        return super().reset(seed=seed, options=options)
    
    def reset_episode_state(self):
        """Reset episode-specific state variables.
        
        This should be called at the start of actual teleoperation,
        after alignment is complete, to clear any state that was
        accidentally triggered during the alignment phase.
        """
        if self.button_pressed_once is not None:
            self.button_pressed_once[:] = False
            print(f"[DEBUG] reset_episode_state(): button_pressed_once reset to {self.button_pressed_once}")
    
    @property
    def _default_sensor_configs(self):
        """Configure sensor cameras - base camera only, hand camera comes from robot."""
        # Front view camera
        pose = sapien_utils.look_at(
            eye=[-0.65, 0.0, 0.5],
            target=[-0.4, 0.0, 0.1]
        )
        return [
            CameraConfig(
                "base_camera", 
                pose, 
                640, 
                480, 
                np.pi / 2, 
                0.01, 
                100),
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure human render camera."""
        pose = sapien_utils.look_at(
            eye=[-0.8, 0.0, 0.75],
            target=[0.0, 0.0, 0.0]
        )

        return CameraConfig("render_camera", pose, 640, 480, np.pi / 2, 0.01, 100)

    
    def _build_walls(self):
        """Build walls around the workspace to provide background for cameras."""
        # Wall parameters
        wall_thickness = 0.01
        wall_height = 2.5
        wall_distance = 1.6  # Distance from center
        wall_color = [0.9, 0.9, 0.9, 1.0]  # Light gray
        
        # Back wall (behind robot, negative X)
        actors.build_box(
            self.scene,
            half_sizes=[wall_thickness, wall_distance, wall_height / 2],
            color=wall_color,
            name="wall_back",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-wall_distance, 0, 0 ]),
        )
        
        # Front wall (in front of robot, positive X)
        actors.build_box(
            self.scene,
            half_sizes=[wall_thickness, wall_distance, wall_height / 2],
            color=wall_color,
            name="wall_front",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[wall_distance, 0, 0]),
        )
        
        # Left wall (negative Y)
        actors.build_box(
            self.scene,
            half_sizes=[wall_distance, wall_thickness, wall_height / 2],
            color=wall_color,
            name="wall_left",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, -wall_distance, 0]),
        )
        
        # Right wall (positive Y)
        actors.build_box(
            self.scene,
            half_sizes=[wall_distance, wall_thickness, wall_height / 2],
            color=wall_color,
            name="wall_right",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, wall_distance, 0]),
        )

    def _load_scene(self, options: dict):
        """Load the ManiSkill table and microwave asset."""
        self.table_scene = TableSceneBuilder(
            self, 
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        microwave_xml = (
            MOLMOSPACES_ASSETS_DIR
            / "mjcf"
            / "objects"
            / "thor"
            / "Kitchen Objects"
            / "Microwave"
            / "Prefabs"
            / "Microwave_1"
            / "Microwave_1_mesh.xml"
        )

        microwave_loader = MjcfAssetArticulationLoader(self.scene)
        microwave_builder = microwave_loader.load_from_xml(microwave_xml, floating_base=False)
        try:
            self.microwave = microwave_builder.build(name="microwave")
        except TypeError:
            self.microwave = microwave_builder.build()

        self.button_base = actors.build_box(
            self.scene,
            half_sizes=self.button_base_half_size.tolist(),
            color=[0.35, 0.35, 0.35, 1.0],
            name="button_base",
            body_type="kinematic",
            add_collision=True,
            initial_pose=sapien.Pose(
                p=[
                    float(self.button_center[0]),
                    float(self.button_center[1]),
                    float(self.button_base_half_size[2]),
                ]
            ),
        )
        self.button_cap = actors.build_box(
            self.scene,
            half_sizes=self.button_cap_half_size.tolist(),
            color=[0.85, 0.15, 0.15, 1.0],
            name="button_cap",
            body_type="kinematic",
            add_collision=True,
            initial_pose=sapien.Pose(
                p=[
                    float(self.button_center[0]),
                    float(self.button_center[1]),
                    float(2 * self.button_base_half_size[2] + self.button_cap_half_size[2]),
                ]
            ),
        )

        self._build_walls()
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset the table scene and place the microwave with door initially open."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Use "home" keyframe for robot position ([-0.65, 0, 0])
            self.agent.robot.set_pose(
                self.agent.keyframes["home"].pose
            )
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            microwave_xyz = torch.zeros((b, 3))
            microwave_xyz[:, 0] = -0.09 + (torch.rand((b,)) - 0.5) * 0.02
            microwave_xyz[:, 1] = (torch.rand((b,)) - 0.5) * 0.04
            microwave_xyz[:, 2] = 0.2
            microwave_q = torch.tensor(
                euler2quat(np.pi / 2, 0, -np.pi / 2),
                device=self.device,
                dtype=microwave_xyz.dtype,
            )

            self.microwave.set_pose(Pose.create_from_pq(microwave_xyz, microwave_q))
            microwave_qpos = self.microwave.get_qpos()
            microwave_qpos[...] = self.microwave_init_open_angle
            self.microwave.set_qpos(microwave_qpos)

            button_base_xyz = torch.zeros((b, 3), device=self.device)
            button_base_xyz[:, 0] = float(self.button_center[0])
            button_base_xyz[:, 1] = float(self.button_center[1])
            button_base_xyz[:, 2] = float(self.button_base_half_size[2])
            button_cap_xyz = torch.zeros((b, 3), device=self.device)
            button_cap_xyz[:, 0] = float(self.button_center[0])
            button_cap_xyz[:, 1] = float(self.button_center[1])
            button_cap_xyz[:, 2] = float(2 * self.button_base_half_size[2] + self.button_cap_half_size[2])
            self.button_base.set_pose(Pose.create_from_pq(button_base_xyz))
            self.button_cap.set_pose(Pose.create_from_pq(button_cap_xyz))

            # Reset button pressed state for this episode
            # Always reinitialize to ensure correct device and size
            if self.button_pressed_once is None or self.button_pressed_once.device != self.device:
                self.button_pressed_once = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            # Reset the state for the episodes being initialized
            self.button_pressed_once[env_idx] = False
            print(f"[DEBUG] _initialize_episode: env_idx={env_idx}, button_pressed_once reset to {self.button_pressed_once}")
    
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """Get extra observations."""
        return dict(
            microwave_pose=self.microwave.pose.raw_pose,
            microwave_qpos=self.microwave.get_qpos(),
            button_pose=self.button_cap.pose.raw_pose,
        )

    def _button_pressed(self) -> torch.Tensor:
        
        left_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.button_cap
        )
        right_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.button_cap
        )
        
        left_force_mag = torch.linalg.norm(left_contact_forces, axis=1)
        right_force_mag = torch.linalg.norm(right_contact_forces, axis=1)
        return (left_force_mag >= self.button_contact_force_thresh) | (
            right_force_mag >= self.button_contact_force_thresh
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """Success when the microwave door is closed and the button has been pressed."""
        microwave_qpos = self.microwave.get_qpos()
        door_closed = torch.abs(microwave_qpos[..., 0]) <= self.microwave_closed_thresh
        button_pressed = self._button_pressed()
        
        # Ensure button_pressed_once is properly initialized
        if self.button_pressed_once is None or self.button_pressed_once.device != self.device:
            self.button_pressed_once = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        
        # Detect new episode: if current step count is less than previous, a reset happened
        current_steps = self._elapsed_steps
        if self._prev_elapsed_steps is not None and current_steps < self._prev_elapsed_steps:
            self.button_pressed_once[:] = False
            print(f"[DEBUG] evaluate: New episode detected (steps {self._prev_elapsed_steps} -> {current_steps}), button_pressed_once reset to False")
        self._prev_elapsed_steps = current_steps
        
        # Update the "pressed once" state
        old_state = self.button_pressed_once.clone()
        self.button_pressed_once = self.button_pressed_once | button_pressed
        
        success = door_closed & self.button_pressed_once
        
        # Debug print
        if button_pressed.any() or success.any():
            print(f"[DEBUG] evaluate: door_closed={door_closed}, button_pressed={button_pressed}, "
                  f"button_pressed_once: {old_state} -> {self.button_pressed_once}, success={success}")

        return {
            "success": success,
            "door_closed": door_closed,
            "button_pressed": button_pressed,
            "button_pressed_once": self.button_pressed_once.clone(),  # Return a copy
            "microwave_door_qpos": microwave_qpos[..., 0],
        }
    
    def compute_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """No task reward: this env only verifies asset loading."""
        return torch.zeros(self.num_envs, device=self.device)
    
    def compute_normalized_dense_reward(
        self, 
        obs: Any, 
        action: torch.Tensor, 
        info: dict
    ) -> torch.Tensor:
        """No normalized reward for the visualization-only env."""
        return torch.zeros(self.num_envs, device=self.device)

