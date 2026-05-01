#!/bin/bash
# Resolve molmo_spaces_maniskill project root relative to this script so the
# script works on any machine (was previously hardcoded to /home/shuo/...).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

# ============================================================
# Active command — uncomment one at a time
# ============================================================

python -m molmo_spaces_maniskill.teleop.teleop_gello \
    -r fr3_robotiq_wristcam \
    -e DroidKitchenOpenDrawerPnpFork-v1

# ============================================================
# Available droid_tasks (uncomment whichever you want to record)
# ============================================================

# Existing tasks
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenPourBottleCup-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenOpenDrawerMem-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidAi2LabOpenMicrowave-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidAi2LabOpenOven-v1

# New iTHOR-based tasks (PR #1)
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenHeatMugMicrowave-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenCloseDrawer-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenOpenDishwasherPnpPlate-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenOpenFridgePnpApple-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenPourBottlePot-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenSetTable-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenSortUtensilsDrawer-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidKitchenDrawerToDrawerFork-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidBathroomOpenToiletDropPaper-v1
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r fr3_robotiq_wristcam -e DroidLivingRoomOpenLaptopPlacePen-v1

# Other robots / sims
# python -m molmo_spaces_maniskill.teleop.teleop_gello -r yam_bimanual -e BimanualYAMMicrowave-v1
# python -m molmo_spaces_maniskill.teleop.teleop_so101 -r so100_wristcam -e SO100PushCubeSlot-v1
