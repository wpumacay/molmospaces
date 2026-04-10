from pathlib import Path
from typing import Any

import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Articulation
from sapien.physx import PhysxArticulation

from molmo_spaces_maniskill.assets.actor_loader import MjcfAssetActorLoader
from molmo_spaces_maniskill.assets.articulation_loader import MjcfAssetArticulationLoader
from molmo_spaces_maniskill.assets.scene_loader import MjcfSceneLoader


@register_env("MolmoSpacesEnv-v0", max_episode_steps=1000)
class MolmoSpacesEnv(BaseEnv):
    def __init__(self, *args, scene_file: Path | None = None, **kwargs) -> None:
        self._mjcf_scene_loader = MjcfSceneLoader()
        self._mjcf_actor_loader = MjcfAssetActorLoader()
        self._mjcf_articulation_loader = MjcfAssetArticulationLoader()

        self._scene_file: Path | None = scene_file

        self._env_actors: dict[str, sapien.Entity | Actor] = {}
        self._env_articulations: dict[str, PhysxArticulation | Articulation] = {}

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict) -> None:
        self._mjcf_scene_loader.set_scene(self.scene)
        self._mjcf_actor_loader.set_scene(self.scene)
        self._mjcf_articulation_loader.set_scene(self.scene)

        if self._scene_file and self._scene_file.is_file():
            self._setup_molmo_spaces_scene()

    def _setup_molmo_spaces_scene(self) -> None:
        assert self._scene_file is not None, "Must have a valid scene file path"
        actors, articulations = self._mjcf_scene_loader.load(self._scene_file)
        self._env_actors.update(actors)
        self._env_articulations.update(articulations)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0
