from typing import Any

import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from ..core.agent import BiI2RTYam, FrankaDroid, I2RTYam


@register_env("MolmoSpacesEmptyEnv-v0", max_episode_steps=1000)
class MolmoSpacesEmptyEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "franka-droid",
        "i2rt-yam",
        "bi-i2rt-yam",
    ]

    agent: I2RTYam | BiI2RTYam | FrankaDroid

    def __init__(
        self, *args, robot_uids: str | BaseAgent | list[str | BaseAgent] | None = None, **kwargs
    ) -> None:
        self._table_scene: TableSceneBuilder | None = None

        super().__init__(*args, robot_uids=robot_uids, **kwargs)  # type: ignore

    def _load_scene(self, options: dict) -> None:
        self._table_scene = TableSceneBuilder(self)
        self._table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        assert self._table_scene is not None, "Must have a TableSceneBuilder by now"
        with torch.device(self.device):
            self._table_scene.initialize(env_idx)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0
