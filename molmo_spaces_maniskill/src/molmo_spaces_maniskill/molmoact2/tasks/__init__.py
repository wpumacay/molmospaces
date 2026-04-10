"""MolmoAct2 custom tasks for ManiSkill."""

from molmo_spaces_maniskill.molmoact2.tasks.so101_tasks import *
from molmo_spaces_maniskill.molmoact2.tasks.droid_tasks import *
from molmo_spaces_maniskill.molmoact2.tasks.yam_tasks import *

__all__ = [
    # Droid tasks
    "DroidKitchenOpenDrawerPnpForkEnv",
    "DroidKitchenOpenDrawerMemEnv",
    "DroidKitchenPourBottleCupEnv",
    "CubeStackEnv",
    # YAM tasks
    "BimanualYAMPickPlaceEnv",
    "BimanualYAMLiftPotEnv",
    "BimanualYAMInsertEnv",
    "BimanualYAMLFlipPlateEnv",
    "BimanualYAMMicrowaveEnv",
    # SO101 tasks
    "SO100PushCubeSlotEnv",
    "SO100CloseMicrowaveEnv",
]

