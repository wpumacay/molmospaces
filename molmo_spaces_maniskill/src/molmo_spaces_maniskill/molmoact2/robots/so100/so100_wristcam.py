import numpy as np
from mani_skill.agents.robots import SO100
import sapien
import torch

from mani_skill.agents.registration import register_agent
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.sensors.camera import CameraConfig
from transforms3d.euler import euler2quat

@register_agent()
class SO100WristCam(SO100):
    """SO100 robot with wrist-mounted camera."""
    
    uid = "so100_wristcam"
    
    # Inherit URDF from parent SO100 - don't override
    
    keyframes = dict(
        home=Keyframe(
            qpos=np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1]),
            pose=sapien.Pose(p=[-0.7, 0, 0], q=euler2quat(0, 0, np.pi / 2)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
    )

    @property
    def _sensor_configs(self):
        """Add wrist camera to SO100."""
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0.0, -0.05, -0.05], q=[ 0.61237244,  0.35355339, -0.61237244, -0.35355339]),
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["Fixed_Jaw"],
            ),
        ]

    
