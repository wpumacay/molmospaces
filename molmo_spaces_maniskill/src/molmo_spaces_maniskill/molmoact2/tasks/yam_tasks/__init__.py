"""YAM robot tasks."""

# Import robots first to ensure they are registered
from molmo_spaces_maniskill.molmoact2.robots.yam import YAM, BimanualYAM

from .bimanual_pnp import BimanualYAMPickPlaceEnv
from .bimanual_lift_pot import BimanualYAMLiftPotEnv
from .bimanual_insert import BimanualYAMInsertEnv
from .bimanual_flip_plate import BimanualYAMLFlipPlateEnv
from .bimanual_microwave import BimanualYAMMicrowaveEnv

__all__ = [
    "YAM",
    "BimanualYAM",
    "BimanualYAMPickPlaceEnv",
    "BimanualYAMLiftPotEnv",
    "BimanualYAMInsertEnv",
    "BimanualYAMLFlipPlateEnv",
    "BimanualYAMMicrowaveEnv",
]

