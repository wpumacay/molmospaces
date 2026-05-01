"""Droid tasks for ManiSkill."""

from .ai2lab_open_mocrowave import DroidAi2LabOpenMicrowaveEnv
from .ai2lab_open_oven import DroidAi2LabOpenOvenEnv
from .bathroom_open_toilet_drop_paper import DroidBathroomOpenToiletDropPaperEnv
from .cube_stack import CubeStackEnv
from .kitchen_close_drawer import DroidKitchenCloseDrawerEnv
from .kitchen_drawer_to_drawer_fork import DroidKitchenDrawerToDrawerForkEnv
from .kitchen_heat_mug_microwave import DroidKitchenHeatMugMicrowaveEnv
from .kitchen_open_dishwasher_pnp_plate import DroidKitchenOpenDishwasherPnpPlateEnv
from .kitchen_open_drawer_mem import DroidKitchenOpenDrawerMemEnv
from .kitchen_open_drawer_pnp_fork import DroidKitchenOpenDrawerPnpForkEnv
from .kitchen_open_fridge_pnp_apple import DroidKitchenOpenFridgePnpAppleEnv
from .kitchen_pour_bottle_cup import DroidKitchenPourBottleCupEnv
from .kitchen_pour_bottle_pot import DroidKitchenPourBottlePotEnv
from .kitchen_set_table import DroidKitchenSetTableEnv
from .kitchen_sort_utensils_drawer import DroidKitchenSortUtensilsDrawerEnv
from .livingroom_open_laptop_place_pen import DroidLivingRoomOpenLaptopPlacePenEnv

__all__ = [
    "DroidAi2LabOpenMicrowaveEnv",
    "DroidAi2LabOpenOvenEnv",
    "DroidBathroomOpenToiletDropPaperEnv",
    "CubeStackEnv",
    "DroidKitchenCloseDrawerEnv",
    "DroidKitchenDrawerToDrawerForkEnv",
    "DroidKitchenHeatMugMicrowaveEnv",
    "DroidKitchenOpenDishwasherPnpPlateEnv",
    "DroidKitchenOpenDrawerMemEnv",
    "DroidKitchenOpenDrawerPnpForkEnv",
    "DroidKitchenOpenFridgePnpAppleEnv",
    "DroidKitchenPourBottleCupEnv",
    "DroidKitchenPourBottlePotEnv",
    "DroidKitchenSetTableEnv",
    "DroidKitchenSortUtensilsDrawerEnv",
    "DroidLivingRoomOpenLaptopPlacePenEnv",
]
