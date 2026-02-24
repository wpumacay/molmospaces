from pathlib import Path

import sapien
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import ArticulationBuilder
from mani_skill.utils.structs import Articulation
from mani_skill.utils.structs.pose import Pose

from ..assets.robot_loader import MjcfAssetRobotLoader


class MolmoSpacesAgent(BaseAgent):
    def __init__(
        self,
        scene: ManiSkillScene,
        control_freq: int,
        control_mode: str | None = None,
        agent_idx: str | None = None,
        initial_pose: sapien.Pose | Pose | None = None,
        build_separate: bool = False,
    ):
        super().__init__(scene, control_freq, control_mode, agent_idx, initial_pose, build_separate)

    def _load_articulation(self, initial_pose: sapien.Pose | Pose | None = None) -> None:
        def build_articulation(scene_idxs: list[int] | None = None):
            if self.urdf_path is not None:
                raise RuntimeError(
                    "URDF files are not supported by our custom loader, use MJCF instead"
                )
            if self.mjcf_path is None:
                raise RuntimeError("Must provide a path to a mjcf model for a robot")

            loader = MjcfAssetRobotLoader(self.scene)
            xml_path = Path(self.mjcf_path)

            if not xml_path.is_file():
                raise RuntimeError(f"Given mjcf model @ {xml_path} doesn't exist")

            builder = loader.load_from_xml(xml_path, floating_base=not self.fix_root_link)
            builder.set_initial_pose(initial_pose)
            assert type(builder) is ArticulationBuilder, (
                "Something went wrong, must have a maniskill ArticulationBuilder"
            )

            robot_name = self.uid
            if self._agent_idx is not None:
                robot_name = f"{self.uid}-agent-{self._agent_idx}"
            builder.set_name(robot_name)

            if scene_idxs is not None:
                builder.set_scene_idxs(scene_idxs)
                builder.set_name(f"{self.uid}-agent-{self._agent_idx}-{scene_idxs}")

            robot = builder.build()
            assert robot is not None, f"Failed to load MJCF from {xml_path}"
            return robot

        if self.build_separate:
            arts = []
            for scene_idx in range(self.scene.num_envs):
                robot = build_articulation([scene_idx])
                self.scene.remove_from_state_dict_registry(robot)
                arts.append(robot)
            self.robot = Articulation.merge(
                arts, name=f"{self.uid}-agent-{self._agent_idx}", merge_links=True
            )
            self.scene.add_to_state_dict_registry(self.robot)
        else:
            self.robot = build_articulation()
        # Cache robot link names
        self.robot_link_names = [link.name for link in self.robot.get_links()]


@register_agent()
class I2RTYam(MolmoSpacesAgent):
    uid = "i2rt-yam"
    mjcf_path = "assets/mjcf/robots/i2rt_yam/yam.xml"


@register_agent()
class BiI2RTYam(MolmoSpacesAgent):
    uid = "bi-i2rt-yam"
    mjcf_path = "assets/mjcf/robots/i2rt_yam/bimanual_yam_ai2.xml"


@register_agent()
class FrankaDroid(MolmoSpacesAgent):
    uid = "franka-droid"
    mjcf_path = "assets/mjcf/robots/franka_droid/model.xml"
