from __future__ import annotations

from pathlib import Path
from typing import Generic, cast

import mujoco as mj
import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import ActorBuilder, ArticulationBuilder
from mani_skill.utils.structs import Actor, Articulation
from sapien import Pose, Scene
from sapien.physx import PhysxArticulation

from .actor_loader import MjcfAssetActorLoader
from .articulation_loader import MjcfAssetArticulationLoader
from .common import SceneT, get_orientation, has_any_non_free_joint


class MjcfSceneLoader(Generic[SceneT]):
    def __init__(self, scene: SceneT | None = None) -> None:
        self._scene: SceneT | None = scene
        self._spec: mj.MjSpec | None = None
        self._num_actors: int = 0
        self._num_articulations: int = 0

    def set_scene(self, scene: SceneT) -> MjcfSceneLoader:
        self._scene = scene
        return self

    @property
    def mjspec(self) -> mj.MjSpec | None:
        return self._spec

    def load(
        self, scene_path: Path
    ) -> tuple[dict[str, sapien.Entity | Actor], dict[str, PhysxArticulation | Articulation]]:
        assert self._scene is not None, "Must assign a valid Sapien scene before loading"

        actors: dict[str, sapien.Entity | Actor] = {}
        articulations: dict[str, PhysxArticulation | Articulation] = {}

        articulation_loader = MjcfAssetArticulationLoader[SceneT](self._scene)
        actor_loader = MjcfAssetActorLoader[SceneT](self._scene)

        if isinstance(self._scene, sapien.Scene):
            self._scene.add_ground(altitude=0, render=False)
        else:
            for subscene in self._scene.sub_scenes:
                subscene.add_ground(altitude=0, render=False)

        spec = mj.MjSpec.from_file(scene_path.as_posix())
        for root_body in spec.worldbody.bodies:
            assert isinstance(root_body, mj.MjsBody)

            world_pose = Pose(p=tuple(root_body.pos), q=tuple(get_orientation(root_body)))
            if has_any_non_free_joint(root_body):
                name = (
                    root_body.name
                    if root_body.name != ""
                    else f"articulation_{self._num_articulations}"
                )
                builder = articulation_loader.load_from_spec(
                    scene_spec=spec,
                    model_dir=scene_path.parent,
                    root_body_name=root_body.name,
                    is_part_of_scene=True,
                )
                builder.set_initial_pose(world_pose)
                if type(builder) is sapien.ArticulationBuilder:
                    articulations[name] = builder.build()
                    articulations[name].set_name(name)  # type: ignore
                elif type(builder) is ArticulationBuilder:
                    articulations[name] = builder.build(name=name)
                self._num_articulations += 1
            else:
                if root_body.name == "floor":
                    continue
                name = root_body.name if root_body.name != "" else f"actor_{self._num_actors}"
                builder = actor_loader.load_from_spec(
                    scene_spec=spec,
                    model_dir=scene_path.parent,
                    root_body_name=root_body.name,
                    is_part_of_scene=True,
                )
                builder.set_name(name)
                builder.set_initial_pose(world_pose)
                if type(builder) is sapien.ActorBuilder:
                    actors[name] = builder.build()
                elif type(builder) is ActorBuilder:
                    actors[name] = builder.build(name)
                self._num_actors += 1

        for light in spec.worldbody.lights:
            assert isinstance(light, mj.MjsLight)
            match light.type:
                case mj.mjtLightType.mjLIGHT_DIRECTIONAL:
                    cast(Scene | ManiSkillScene, self._scene).add_directional_light(
                        direction=light.dir,
                        color=light.diffuse,
                        shadow=bool(light.castshadow),
                    )
                case mj.mjtLightType.mjLIGHT_POINT:
                    pass
                case _:
                    pass

        return actors, articulations
