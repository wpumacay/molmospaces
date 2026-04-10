from __future__ import annotations

from pathlib import Path

import mujoco as mj
from sapien import ArticulationBuilder
from sapien.render import RenderMaterial

from .articulation_loader import MjcfAssetArticulationLoader
from .common import (
    SceneT,
)


class MjcfAssetRobotLoader(MjcfAssetArticulationLoader):
    def __init__(self, scene: SceneT | None = None, use_colliders_as_visuals: bool = False):
        super().__init__(scene, use_colliders_as_visuals)

    def load_from_spec(
        self,
        scene_spec: mj.MjSpec,
        model_dir: Path,
        root_body_name: str | None = None,
        floating_base: bool | None = None,
        materials: dict[str, RenderMaterial] | None = None,
        is_part_of_scene: bool = False,
    ) -> ArticulationBuilder:
        articulation_builder = super().load_from_spec(
            scene_spec, model_dir, root_body_name, floating_base, materials, is_part_of_scene
        )

        return articulation_builder
