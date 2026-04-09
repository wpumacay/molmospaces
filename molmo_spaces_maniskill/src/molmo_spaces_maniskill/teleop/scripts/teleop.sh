#!/bin/bash
cd /home/shuo/research/molmospaces/molmo_spaces_maniskill 

# python -m molmo_spaces_maniskill.teleop.teleop_gello \
#     -r yam_bimanual \
#     -e BimanualYAMMicrowave-v1

python -m molmo_spaces_maniskill.teleop.teleop_gello \
    -r fr3_robotiq_wristcam \
    -e DroidKitchenOpenDrawerPnpFork-v1


# python -m molmo_spaces_maniskill.teleop.teleop_so101 \
#     -r so100_wristcam \
#     -e SO100PushCubeSlot-v1